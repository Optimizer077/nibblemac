/* nibblemac's experimental standalone Metal backend. Not yet wired into pinfer.
 * Handles are private, buffers use shared unified memory, calls are synchronous.
 * A context and its buffers must be used from a single thread at a time.
 * All functions report errors rather than silently switching to a CPU backend. */
#ifndef NIBM_METAL_H
#define NIBM_METAL_H

#include <stddef.h>
#include <stdint.h>

typedef struct nm_metal nm_metal;
typedef struct nm_buffer nm_buffer;

enum { NM_F32 = 0, NM_Q4K = 4 }; /* same dtype IDs and packed layout as .pq */

nm_metal *nm_metal_create(const char *shader_path, char *error, size_t error_size);
void nm_metal_destroy(nm_metal *context);
const char *nm_metal_device_name(nm_metal *context);
uint64_t nm_metal_recommended_memory(nm_metal *context);
nm_buffer *nm_metal_buffer_create(nm_metal *context, size_t bytes);
void nm_metal_buffer_destroy(nm_buffer *buffer);
void *nm_metal_buffer_contents(nm_buffer *buffer);

/* W [rows,cols], X [batch,cols], Y [batch,rows]. For Q4K: rows % 4 == 0,
 * cols % 256 == 0, W has rows/4 * cols/256 * 576 bytes (engine/FORMAT.md).
 * Inputs remain fp32 (unlike the legacy CPU's int8 activation quantization).
 * Buffer handles must belong to this context. Output must not alias an input.
 * Repeats are encoded in one command buffer for throughput microbenchmarks;
 * normal use passes repeats=1. gpu_seconds is the entire command buffer time. */
int nm_metal_matmul(nm_metal *context, uint32_t dtype, uint32_t rows, uint32_t cols,
                    uint32_t batch, nm_buffer *weights, nm_buffer *inputs,
                    nm_buffer *outputs, uint32_t repeats, double *gpu_seconds,
                    char *error, size_t error_size);

#endif
