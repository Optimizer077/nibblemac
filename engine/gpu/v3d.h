/* Standalone V3D 4.2 DRM uAPI client. No py-videocore6 driver code. */
#ifndef PINFER_V3D_H
#define PINFER_V3D_H

#include <stddef.h>
#include <stdint.h>

struct v3d_device { int fd; };
struct v3d_bo { uint32_t handle, gpu; size_t size; void *map; };

/* Functions return 0 on success, -1 with errno set on failure.
 * Initialize device.fd=-1 and BOs to zero before use. Destroy BOs before close. */
int v3d_open(struct v3d_device *dev, const char *node);
void v3d_close(struct v3d_device *dev);
int v3d_bo_create(struct v3d_device *dev, struct v3d_bo *bo, size_t size);
void v3d_bo_destroy(struct v3d_device *dev, struct v3d_bo *bo);
int v3d_wait(struct v3d_device *dev, const struct v3d_bo *bo, uint64_t timeout_ns);
/* Eight 16-lane workgroups; two hardware threads, multi-segment shader, NaNs preserved.
 * All BOs touched by the shader/uniforms must be in handles. Submission is asynchronous;
 * wait on any included BO before CPU reads/reuses memory or destroys a BO. */
int v3d_submit_8qpu(struct v3d_device *dev, uint32_t shader, uint32_t uniforms,
                    const uint32_t *handles, size_t count);
#endif
