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

void pf_pool_run(pf_pool *pool, pf_task fn, void *arg, int max_threads) {
    (void)pool;
    (void)max_threads;
    fn(arg, 0, 1);
}

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

/* ---- Gated DeltaNet recurrence, scalar reference (moved here from forward_lin.c) ---- */

/* One head of the gated delta rule. S is [dk][dv] (row = key dim).
 *   S *= decay;  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
 * Two passes over S: (decay, accumulate kv) then (rank-1 update, accumulate o).
 * On return tmp[0..dv) holds delta (the rollback log records it). */
static void delta_rule(float *S, const float *q, const float *k, const float *v,
                       float decay, float beta, float *o, float *tmp, int dk, int dv) {
    float *kv = tmp;
    memset(kv, 0, (size_t)dv * sizeof(float));
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float ki = k[i];
        for (int j = 0; j < dv; j++) {
            row[j] *= decay;
            kv[j] += row[j] * ki;
        }
    }
    for (int j = 0; j < dv; j++) kv[j] = (v[j] - kv[j]) * beta;  /* kv now holds delta */
    memset(o, 0, (size_t)dv * sizeof(float));
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float ki = k[i], qi = q[i];
        for (int j = 0; j < dv; j++) {
            row[j] += ki * kv[j];
            o[j] += row[j] * qi;
        }
    }
}

static float *lin_scratch;      /* one head's state for logged (verification) passes */
static size_t lin_scratch_floats;

void pf_lin_rule(pf_pool *pool, const pf_lin_args *a) {
    (void)pool;
    const int dk = a->dk, dv = a->dv, kd = (a->heads / a->rep) * dk;
    const size_t hs = (size_t)dk * dv, logw = (size_t)dk + dv + 1;
    float tmp[512];   /* delta scratch; lin_v_dim <= 512 (checked in pinfer.c) */
    if (a->log && lin_scratch_floats < hs) {
        free(lin_scratch);
        lin_scratch = malloc(hs * sizeof(float));
        lin_scratch_floats = lin_scratch ? hs : 0;
        if (!lin_scratch) abort();
    }
    for (int h = 0; h < a->heads; h++) {
        const int kh = h / a->rep;
        float *S = a->S + h * hs;
        if (a->log) {
            memcpy(lin_scratch, S, hs * sizeof(float));
            S = lin_scratch;
        }
        for (int t = 0; t < a->n; t++) {
            const float *qkv = a->qkv + (size_t)t * a->qkv_stride;
            const float *k = qkv + kd + kh * dk;
            const float decay = a->decay[(size_t)t * a->heads + h];
            delta_rule(S, qkv + kh * dk, k, qkv + 2 * kd + h * dv, decay,
                       a->beta[(size_t)t * a->heads + h], a->o + (size_t)t * a->o_stride + h * dv,
                       tmp, dk, dv);
            if (a->log) {
                float *u = a->log + ((size_t)t * a->heads + h) * logw;
                memcpy(u, k, (size_t)dk * sizeof(float));
                memcpy(u + dk, tmp, (size_t)dv * sizeof(float));
                u[dk + dv] = decay;
            }
        }
    }
}

void pf_lin_apply(pf_pool *pool, float *S, const float *log, int keep, int heads, int dk,
                  int dv) {
    (void)pool;
    const size_t logw = (size_t)dk + dv + 1;
    for (int h = 0; h < heads; h++) {
        float *Sh = S + (size_t)h * dk * dv;
        for (int t = 0; t < keep; t++) {
            const float *u = log + ((size_t)t * heads + h) * logw;
            const float *d = u + dk, decay = u[dk + dv];
            for (int i = 0; i < dk; i++) {
                float *row = Sh + (size_t)i * dv;
                const float ki = u[i];
                for (int j = 0; j < dv; j++) {
                    row[j] *= decay;
                    row[j] += ki * d[j];
                }
            }
        }
    }
}
