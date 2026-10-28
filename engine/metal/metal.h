/* nibblemac's experimental Metal backend, standalone and hybrid inference.
 * Handles are private, buffers use shared unified memory. Calls are synchronous
 * except the explicit begin/finish submission pair.
 * A context and its buffers must be used from a single thread at a time.
 * All functions report errors rather than silently switching to a CPU backend. */
#ifndef NIBM_METAL_H
#define NIBM_METAL_H

#include <stddef.h>
#include <stdint.h>

typedef struct nm_metal nm_metal;
typedef struct nm_buffer nm_buffer;
typedef struct nm_submission nm_submission;

enum { NM_F32 = 0, NM_Q4K = 4 }; /* same dtype IDs and packed layout as .pq */

nm_metal *nm_metal_create(const char *shader_path, char *error, size_t error_size);
void nm_metal_destroy(nm_metal *context);
const char *nm_metal_device_name(nm_metal *context);
uint64_t nm_metal_recommended_memory(nm_metal *context);
nm_buffer *nm_metal_buffer_create(nm_metal *context, size_t bytes);
void nm_metal_buffer_destroy(nm_buffer *buffer);
void *nm_metal_buffer_contents(nm_buffer *buffer);
/* Upload immutable Q4K weights once, repacking without requantization when the
 * selected A8 kernel uses a GPU-coalesced layout. Input remains upstream .pq. */
nm_buffer *nm_metal_a8_weights(nm_metal *context, uint32_t rows, uint32_t cols,
                               const void *packed, size_t bytes);

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

/* Exact upstream Q4K/A8 arithmetic: block-major int8 X, fp32 per-block scales
 * and scaled sums. Integer dot products, ordered explicit FMA scale/correction.
 * W stays packed; caller performs the same activation quantization as NEON. */
int nm_metal_matmul_a8(nm_metal *context, uint32_t rows, uint32_t cols, uint32_t batch,
                       nm_buffer *w, nm_buffer *q, nm_buffer *d, nm_buffer *fs,
                       nm_buffer *y, char *error, size_t error_size);

/* Submit without waiting so the caller can compute disjoint CPU output rows.
 * Consume the returned handle exactly once with finish, before reading outputs,
 * reusing any input/workspace, or destroying the context. No CPU fallback. */
int nm_metal_matmul_a8_begin(nm_metal *context, uint32_t rows, uint32_t cols, uint32_t batch,
                             nm_buffer *w, nm_buffer *q, nm_buffer *d, nm_buffer *fs,
                             nm_buffer *y, nm_submission **pending, char *error, size_t error_size);
int nm_metal_matmul_a8_finish(nm_submission *pending, char *error, size_t error_size);

/* One submission: fp32 input -> A8 -> gate/up -> SiLU*up -> A8 -> down -> fp32.
 * Workspace is caller-owned; no CPU intermediate activations or waits. All
 * weight handles must come from nm_metal_a8_weights for this context. */
typedef struct { nm_buffer *x, *gate, *up, *q, *d, *fs, *y; } nm_mlp_buffers;
int nm_metal_mlp(nm_metal *context, uint32_t hidden, uint32_t ffn, uint32_t batch,
                  nm_buffer *wg, nm_buffer *wu, nm_buffer *wd, const nm_mlp_buffers *buffers,
                  char *error, size_t error_size);

/* Chunk-major DeltaNet state matches kern_lin.c exactly. Persistent shared S
 * remains CPU-visible after each synchronous call (save/load/reset need no
 * mirrored-state synchronization). Logs use the upstream layout. */
typedef struct { uint32_t n, heads, rep, dk, dv, qkv_stride, o_stride, keep, logged; } nm_lin_shape;
typedef struct { nm_buffer *state, *qkv, *decay, *beta, *output, *log, *pend, *work; } nm_lin_buffers;
int nm_metal_lin_rule(nm_metal *context, const nm_lin_shape *shape, const nm_lin_buffers *buffers,
                      char *error, size_t error_size);
int nm_metal_lin_apply(nm_metal *context, const nm_lin_shape *shape, const nm_lin_buffers *buffers,
                       char *error, size_t error_size);

#endif
