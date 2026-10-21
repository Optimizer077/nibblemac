/* Kernel interface between the model code (forward.c) and the matrix kernels.
 *
 * Contract: forward.c only calls the functions below. Two implementations exist and are
 * chosen at build time (engine/Makefile MATVEC=ref|fast):
 *   matvec_ref.c  — portable scalar, single thread; the correctness reference
 *   matvec_fast.c — NEON kernels, quantized dtypes, thread pool (pinned cores)
 */
#ifndef PF_OPS_H
#define PF_OPS_H

#include <stdint.h>

/* A weight matrix W [rows][cols], row-major, y = W x. `dtype` is a pq_dtype; for
 * quantized dtypes `data` points at the packed layout defined in FORMAT.md. */
typedef struct {
    uint32_t dtype;
    int rows, cols;
    const void *data;
} pf_mat;

typedef struct pf_pool pf_pool; /* opaque; NULL means "run on the calling thread" */

pf_pool *pf_pool_create(int n_threads);
void pf_pool_destroy(pf_pool *pool);
int pf_pool_threads(const pf_pool *pool);

/* y[r] = sum_c W[r][c] * x[c] for all rows. */
void pf_matvec(pf_pool *pool, const pf_mat *W, const float *x, float *y);

/* Batched form: n input vectors X [n][cols] -> Y [n][rows]. Must read W once for all n
 * (this is what makes prefill and speculative verification cheap). */
void pf_matmul(pf_pool *pool, const pf_mat *W, const float *X, float *Y, int n);

/* Embedding lookup: dst[c] = float(W[row][c]). */
void pf_get_row(const pf_mat *W, int row, float *dst);

/* 1 if this build supports W->dtype. */
int pf_dtype_supported(uint32_t dtype);

#endif
