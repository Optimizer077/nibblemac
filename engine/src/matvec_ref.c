/* matvec_ref: portable scalar implementation of ops.h for F32 and BF16 weights.
 *
 * This is the correctness reference the NEON/quantized kernels (matvec_fast.c) are tested
 * against, so it favours being obviously right over being fast: every product and sum is
 * fp32, BF16 is widened exactly (it is the upper half of an fp32), and the summation order
 * is fixed (4 interleaved partial sums, combined as (s0+s1)+(s2+s3)) so results are
 * bit-reproducible across runs and thread counts. Single-threaded; the pool is a stub. */
#include <stdlib.h>
#include <string.h>

#include "ops.h"
#include "pq_format.h"

struct pf_pool { int n_threads; };

pf_pool *pf_pool_create(int n_threads) {
    pf_pool *p = malloc(sizeof *p);
    if (p) p->n_threads = n_threads < 1 ? 1 : n_threads;  /* recorded, but work stays serial */
    return p;
}

void pf_pool_destroy(pf_pool *pool) { free(pool); }

int pf_pool_threads(const pf_pool *pool) { (void)pool; return 1; }

int pf_dtype_supported(uint32_t dtype) { return dtype == PQ_F32 || dtype == PQ_BF16; }

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static float dot_f32(const float *w, const float *x, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int c = 0;
    for (; c + 4 <= n; c += 4) {
        s0 += w[c] * x[c];
        s1 += w[c + 1] * x[c + 1];
        s2 += w[c + 2] * x[c + 2];
        s3 += w[c + 3] * x[c + 3];
    }
    for (; c < n; c++) s0 += w[c] * x[c];
    return (s0 + s1) + (s2 + s3);
}

static float dot_bf16(const uint16_t *w, const float *x, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int c = 0;
    for (; c + 4 <= n; c += 4) {
        s0 += bf16_to_f32(w[c]) * x[c];
        s1 += bf16_to_f32(w[c + 1]) * x[c + 1];
        s2 += bf16_to_f32(w[c + 2]) * x[c + 2];
        s3 += bf16_to_f32(w[c + 3]) * x[c + 3];
    }
    for (; c < n; c++) s0 += bf16_to_f32(w[c]) * x[c];
    return (s0 + s1) + (s2 + s3);
}

static float dot_row(const pf_mat *W, int r, const float *x) {
    size_t off = (size_t)r * (size_t)W->cols;
    if (W->dtype == PQ_BF16) return dot_bf16((const uint16_t *)W->data + off, x, W->cols);
    return dot_f32((const float *)W->data + off, x, W->cols);
}

void pf_matvec(pf_pool *pool, const pf_mat *W, const float *x, float *y) {
    (void)pool;
    for (int r = 0; r < W->rows; r++) y[r] = dot_row(W, r, x);
}

/* Row-outer, vector-inner: each weight row is fetched from DRAM once and then stays in L1
 * for all n inputs, which is what the contract asks for. Results equal n pf_matvec calls. */
void pf_matmul(pf_pool *pool, const pf_mat *W, const float *X, float *Y, int n) {
    (void)pool;
    for (int r = 0; r < W->rows; r++)
        for (int i = 0; i < n; i++)
            Y[(size_t)i * W->rows + r] = dot_row(W, r, X + (size_t)i * W->cols);
}

void pf_get_row(const pf_mat *W, int row, float *dst) {
    size_t off = (size_t)row * (size_t)W->cols;
    if (W->dtype == PQ_BF16) {
        const uint16_t *w = (const uint16_t *)W->data + off;
        for (int c = 0; c < W->cols; c++) dst[c] = bf16_to_f32(w[c]);
    } else {
        memcpy(dst, (const float *)W->data + off, (size_t)W->cols * sizeof(float));
    }
}
