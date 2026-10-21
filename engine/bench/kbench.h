/* Shared declarations of the kernel microbenchmark (kbench.c + kbench_util.c). */
#ifndef KBENCH_H
#define KBENCH_H

#include <stddef.h>
#include <stdint.h>

#include "ops.h"

/* Bytes of a rows x cols matrix in `dtype` (0 = unsupported shape). */
size_t kb_matrix_bytes(uint32_t dtype, int rows, int cols);

/* Fill dst with a pseudo-random rows x cols matrix in `dtype` (quantized dtypes are packed
 * with simple absmax / min-max quantizers; the values only need to be realistic). */
void kb_fill_matrix(uint32_t dtype, int rows, int cols, uint64_t seed, void *dst);

void kb_fill_vec(float *x, size_t n, uint64_t seed);

/* Reference outputs for `nrow` rows listed in `rowidx` and n vectors, using pf_get_row:
 * ref_fp uses the fp32 inputs, ref_q (may be NULL) the int8-quantized inputs the quantized
 * kernels see. Output layout [n][nrow]. */
void kb_reference(const pf_mat *W, const float *X, int n, const int *rowidx, int nrow,
                  float *ref_fp, float *ref_q);

/* Check pf_matmul and pf_get_row against the golden file written by tools/quant.py.
 * Returns the number of failing cases (prints details). */
int kb_golden(const char *path, pf_pool *pool);

#endif
