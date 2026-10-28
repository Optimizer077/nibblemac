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
#include <stddef.h>

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
/* Mutable recurrence storage: optional persistent shared Metal buffers. Initially
 * zero; release through this pool before pool_destroy (not directly with free). */
void *pf_pool_state_alloc(pf_pool *pool, size_t bytes);
void pf_pool_state_free(pf_pool *pool, void *state);
/* Optional fused MLP; returns 1 if computed, 0 if this backend declines. */
int pf_pool_mlp(pf_pool *pool, const pf_mat *gate, const pf_mat *up, const pf_mat *down,
                const float *x, float *y, int n);
/* Upload an immutable subset once, bounded to 512 MiB including scratch. 0 success,
 * -1 unavailable/error. The reference backend rejects enabled GPU use. */
int pf_pool_gpu_enable(pf_pool *pool, const pf_mat *const *weights, int count, float share);
void pf_pool_gpu_pause(pf_pool *pool, int pause); /* speculative verify stays on the CPU */

/* Generic parallel job: fn(arg, tid, nthr) runs on nthr = min(pool threads, max_threads)
 * threads (tid 0 = the caller) and returns when all are done. The reference pool runs it on
 * the calling thread (nthr = 1). forward.c uses it for the per-token work outside the
 * matmuls (attention heads, conv channel ranges, activations); the split must not change
 * any per-element arithmetic. */
typedef void (*pf_task)(void *arg, int tid, int nthr);
void pf_pool_run(pf_pool *pool, pf_task fn, void *arg, int max_threads);

/* y[r] = sum_c W[r][c] * x[c] for all rows. */
void pf_matvec(pf_pool *pool, const pf_mat *W, const float *x, float *y);

/* Batched form: n input vectors X [n][cols] -> Y [n][rows]. Must read W once for all n
 * (this is what makes prefill and speculative verification cheap). */
void pf_matmul(pf_pool *pool, const pf_mat *W, const float *X, float *Y, int n);

/* Gated DeltaNet recurrence of one layer over n tokens (forward_lin.c semantics). For each
 * v-head h (k-head h / rep) and token t, with S [dk][dv]:
 *   S *= decay;  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
 * Both implementations perform exactly these per-element fp32 operations (no fused
 * multiply-add) in the same order, so they agree bit-for-bit with each other and with the
 * commit replay below.
 *
 * Speculative verification (log != NULL): token 0 updates S in place like any token (the
 * caller always commits the first token of a verify pass); tokens 1..n-1 run on a private
 * copy, so S ends as the state after token 0, and their log rows t >= 1 receive
 * k | delta | decay (row 0 is not written). Deferred commit (keep > 0): before the tokens, S
 * first advances by the first `keep` updates of `pend` (rows laid out like `log`), with
 * pf_lin_apply's arithmetic, while it is in cache anyway. A logged pass needs
 * dk <= PF_LIN_MAX_DK. */
#define PF_LIN_MAX_DK 256
typedef struct {
    int n, heads, rep, dk, dv;  /* tokens, v-heads, v-heads per k-head, head dims */
    int qkv_stride, o_stride;   /* row strides in floats of qkv and o */
    const float *qkv;           /* [n][qkv_stride]: q | k (heads/rep heads of dk) | v (heads x dv) */
    const float *decay, *beta;  /* [n][heads] */
    float *S;                   /* [heads][dk * dv] committed state, updated in place; its
                                   layout is private to the implementation (zero = reset) */
    float *o;                   /* [n][o_stride]: outputs before the gated norm */
    float *log;                 /* NULL, or [n][heads][dk + dv + 1] rollback log (pf_lin_log) */
    const float *pend;          /* [keep][heads][dk + dv + 1] updates to apply first */
    int keep;
    float *work;                /* [dk * dv] per-context scratch for a logged pass (the ref
                                   keeps its head copy there; the NEON kernel needs none) */
} pf_lin_args;
void pf_lin_rule(pf_pool *pool, const pf_lin_args *a);

/* Commit replay: S[h] = S[h] * decay + k x delta for the first `keep` logged tokens (rows of
 * `log` as written by pf_lin_rule), per element: row *= decay; row += k_i * delta_j. */
void pf_lin_apply(pf_pool *pool, float *S, const float *log, int keep, int heads, int dk,
                  int dv);

/* Embedding lookup: dst[c] = float(W[row][c]). */
void pf_get_row(const pf_mat *W, int row, float *dst);

/* 1 if this build supports W->dtype. */
int pf_dtype_supported(uint32_t dtype);

#endif
