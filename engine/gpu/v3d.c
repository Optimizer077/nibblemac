/* Own userspace implementation against the installed MIT DRM V3D uAPI header.
 * CSD register field meanings: Mesa src/broadcom/common/v3d_csd.h and
 * src/broadcom/vulkan/v3dv_cmd_buffer.c (MIT, Raspberry Pi).
 * https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/broadcom/common/v3d_csd.h
 * No code from py-videocore6's GPL driver is used here.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "v3d.h"

#include <drm/drm.h>
#include <drm/v3d_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int call_ioctl(int fd, unsigned long request, void *arg) {
    int rc;
    do { rc = ioctl(fd, request, arg); } while (rc < 0 && errno == EINTR);
    return rc;
}

int v3d_open(struct v3d_device *dev, const char *node) {
    if (!dev || dev->fd != -1) { errno = EINVAL; return -1; }
    int fd = open(node ? node : "/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    char name[16] = {0};
    struct drm_version version = { .name_len = sizeof(name) - 1, .name = name };
    struct drm_v3d_get_param csd = { .param = DRM_V3D_PARAM_SUPPORTS_CSD };
    if (call_ioctl(fd, DRM_IOCTL_VERSION, &version) < 0 ||
        call_ioctl(fd, DRM_IOCTL_V3D_GET_PARAM, &csd) < 0) {
        int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
    if (strcmp(name, "v3d") || !csd.value) {
        close(fd);
        errno = ENODEV;
        return -1;
    }
    dev->fd = fd;
    return 0;
}

void v3d_close(struct v3d_device *dev) {
    if (dev && dev->fd >= 0) { close(dev->fd); dev->fd = -1; }
}

void v3d_bo_destroy(struct v3d_device *dev, struct v3d_bo *bo) {
    if (!bo) return;
    int saved = errno;
    if (bo->map && bo->map != MAP_FAILED) munmap(bo->map, bo->size);
    if (bo->handle && dev && dev->fd >= 0) {
        struct drm_gem_close close_bo = { .handle = bo->handle };
        call_ioctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &close_bo);
    }
    memset(bo, 0, sizeof(*bo));
    errno = saved;
}

int v3d_bo_create(struct v3d_device *dev, struct v3d_bo *bo, size_t size) {
    long page = sysconf(_SC_PAGESIZE);
    if (!dev || dev->fd < 0 || !bo || bo->handle || !size || page <= 0 ||
        size > UINT32_MAX - (size_t)page + 1) { errno = EINVAL; return -1; }
    size = (size + (size_t)page - 1) / (size_t)page * (size_t)page;
    struct drm_v3d_create_bo create = { .size = (uint32_t)size };
    if (call_ioctl(dev->fd, DRM_IOCTL_V3D_CREATE_BO, &create) < 0) return -1;
    *bo = (struct v3d_bo){ .handle = create.handle, .gpu = create.offset, .size = size };
    if (!bo->gpu || (uint64_t)bo->gpu + size > UINT64_C(0x100000000)) {
        errno = EOVERFLOW; v3d_bo_destroy(dev, bo); return -1;
    }
    struct drm_v3d_mmap_bo mapping = { .handle = bo->handle };
    if (call_ioctl(dev->fd, DRM_IOCTL_V3D_MMAP_BO, &mapping) < 0) {
        v3d_bo_destroy(dev, bo); return -1;
    }
    bo->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd, (off_t)mapping.offset);
    if (bo->map == MAP_FAILED) { v3d_bo_destroy(dev, bo); return -1; }
    return 0;
}

int v3d_wait(struct v3d_device *dev, const struct v3d_bo *bo, uint64_t timeout_ns) {
    if (!dev || dev->fd < 0 || !bo || !bo->handle) { errno = EINVAL; return -1; }
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) return -1;
    uint64_t start = (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
    struct drm_v3d_wait_bo wait = { .handle = bo->handle, .timeout_ns = timeout_ns };
    while (ioctl(dev->fd, DRM_IOCTL_V3D_WAIT_BO, &wait) < 0) {
        if (errno != EINTR) return -1;
        if (clock_gettime(CLOCK_MONOTONIC, &ts)) return -1;
        uint64_t elapsed = (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec - start;
        if (elapsed >= timeout_ns) { errno = ETIME; return -1; }
        wait.timeout_ns = timeout_ns - elapsed;
    }
    atomic_thread_fence(memory_order_acquire);
    return 0;
}

int v3d_submit_8qpu(struct v3d_device *dev, uint32_t shader, uint32_t uniforms,
                    const uint32_t *handles, size_t count) {
    if (!dev || dev->fd < 0 || !shader || (shader & 7) || !uniforms || (uniforms & 3) ||
        !handles || !count || count > UINT32_MAX) { errno = EINVAL; return -1; }
    /* V3D 4.2 CSD CFG0..2: workgroup count in bits 31:16.
     * CFG3: eight workgroups in one supergroup, 16 invocations/group, eight batches.
     * Keeping all eight in the same supergroup is essential: this shader indexes
     * its tile partition by physical QPU tidx, not a logical workgroup ID.
     * CFG4: eight total batches minus one. CFG5 low bits: preserve NaNs only;
     * no four-thread or single-segment flags (shader uses >32 RFs and thrsw).
     */
    struct drm_v3d_submit_csd submit = {
        .cfg = { 8u << 16, 1u << 16, 1u << 16, (7u << 12) | (8u << 8) | 16u, 7u, shader | 4u, uniforms },
        .bo_handles = (uint64_t)(uintptr_t)handles, .bo_handle_count = (uint32_t)count
    };
    atomic_thread_fence(memory_order_release);
    return call_ioctl(dev->fd, DRM_IOCTL_V3D_SUBMIT_CSD, &submit);
}
