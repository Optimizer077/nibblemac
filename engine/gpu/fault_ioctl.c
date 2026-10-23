/* Test-only LD_PRELOAD interposer. Simulated jobs never reach the GPU. */
#define _GNU_SOURCE
#include <drm/drm.h>
#include <drm/v3d_drm.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

static uint32_t setups, submits, waits;
void pf_gpu_fault_stats(uint32_t out[3]) {
    out[0] = setups; out[1] = submits; out[2] = waits;
}

int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    const char *mode = getenv("PF_GPU_TEST_FAULT");
    if (mode && request == DRM_IOCTL_V3D_GET_PARAM && !strcmp(mode, "unavailable")) {
        setups++; errno = ENODEV; return -1;
    }
    if (mode && request == DRM_IOCTL_V3D_SUBMIT_CSD &&
        (!strcmp(mode, "submit") || !strcmp(mode, "wait"))) {
        submits++;
        /* No real job is submitted, including before a simulated partial failure. */
        if (!strcmp(mode, "submit") && submits == 2) { errno = EIO; return -1; }
        return 0;
    }
    if (mode && request == DRM_IOCTL_V3D_WAIT_BO && !strcmp(mode, "wait")) {
        waits++; errno = ETIME; return -1;
    }
    int (*real_ioctl)(int, unsigned long, ...) = dlsym(RTLD_NEXT, "ioctl");
    if (!real_ioctl) { errno = ENOSYS; return -1; }
    return real_ioctl(fd, request, arg);
}
