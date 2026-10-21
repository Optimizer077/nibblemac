/* Gated DeltaNet recurrence (matvec_ref.c delta_rule) as a NEON kernel threaded over heads.
 *
 * Per head and token, with S [dk][dv]:
 *   S *= decay;  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
 * Two passes over S exactly like the scalar code: (decay, accumulate kv) then (rank-1
 * update, accumulate o). Every per-element operation keeps the scalar rounding (separate
 * multiply and add, never a fused multiply-add) and the per-column accumulation order over
 * the key index i, so the results, the updated state and the rollback log are bit-identical
 * to the scalar code and to pf_lin_commit's replay.
 *
 * The recurrence is separable by value column j, so S is processed in 32-column chunks: the
 * chunk's kv / o accumulators live in registers across the 128 key rows, and the chunk loop
 * is outside the token loop, so for n > 1 a 16 KB chunk of the 64 KB head state stays in L1
 * for all tokens and both passes. For that the state is stored chunk-major,
 * S[h][j / 32][i][j % 32] (each chunk contiguous): with the natural [i][j] layout the rows of
 * a chunk are 512 B apart and map onto only 64 of the 256 sets of the 2-way L1, which
 * thrashed (v2 measured L2-bound). The layout is private to the pf_lin_* contract (the state
 * is only ever zeroed outside it). Heads are independent and run in parallel on the pool.
 *
 * Speculative verification (engine/VERIFY.md): per chunk, the deferred commit of the
 * previous pass (`keep` logged rank-1 updates) is folded into the first sweep of token 0
 * (each row stays in registers across the updates), token 0 runs in place (it is always
 * committed) and tokens 1..n-1 continue on a 16 KB copy of the chunk on the stack of the
 * thread, so S is read from DRAM and written back once per pass, like a plain step. There
 * is no shared scratch state.
 */
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

#include "kern.h"

#define INLINE static inline __attribute__((always_inline))

#define CW 32 /* columns per chunk: 8 float32x4 accumulators, 16 KB of state per chunk */

/* The `keep` deferred updates (rows of pend, pstride floats apart) and the first sweep of
 * a token, in place, in one pass over the chunk's rows: each row stays in registers across
 * the updates. Same per-element operations as apply_chunk followed by delta_chunk's first
 * sweep; kv accumulates into acc. Folding them measured -13% for a k=3 verify pass in
 * lbench (VERIFY.md): the extra arithmetic overlaps the one DRAM read of the chunk. */
INLINE void replay_first(float *Sc, const float *pend, size_t pstride, int keep, int c, int dk,
                         int dv, const float *k0, float d0, float32x4_t *acc) {
    const float32x4_t dec = vdupq_n_f32(d0);
    for (int i = 0; i < dk; i++) {
        float *row = Sc + (size_t)i * CW;
        float32x4_t r[CW / 4];
        for (int u = 0; u < CW / 4; u++) r[u] = vld1q_f32(row + 4 * u);
        for (int t = 0; t < keep; t++) {
            const float *p = pend + (size_t)t * pstride;
            const float32x4_t pd = vdupq_n_f32(p[dk + dv]), pk = vdupq_n_f32(p[i]);
            for (int u = 0; u < CW / 4; u++)
                r[u] = vaddq_f32(vmulq_f32(r[u], pd), vmulq_f32(pk, vld1q_f32(p + dk + c + 4 * u)));
        }
        const float32x4_t ki = vdupq_n_f32(k0[i]);
        for (int u = 0; u < CW / 4; u++) {
            const float32x4_t x = vmulq_f32(r[u], dec);
            vst1q_f32(row + 4 * u, x);
            acc[u] = vaddq_f32(acc[u], vmulq_f32(x, ki));
        }
    }
}

/* Columns [c, c + CW) of one head for one token. The first pass reads the chunk's
 * [dk][CW] block from `src` and writes the decayed rows to Sc (src == Sc: in place; with
 * keep > 0, which requires src == Sc, after the deferred updates of pend); the second pass
 * works on Sc. delta[0..CW) receives the chunk's delta. */
INLINE void delta_chunk(float *Sc, const float *src, const float *q, const float *k,
                        const float *v, float decay, float beta, float *o, float *delta, int c,
                        int dk, const float *pend, size_t pstride, int keep, int dv) {
    const int cw = CW;
    const float32x4_t dec = vdupq_n_f32(decay);
    float32x4_t acc[CW / 4];
    for (int u = 0; u < cw / 4; u++) acc[u] = vdupq_n_f32(0.0f);
    if (keep > 0) replay_first(Sc, pend, pstride, keep, c, dk, dv, k, decay, acc);
    else for (int i = 0; i < dk; i++) { /* S *= decay; kv += S[i] * k[i] (new S) */
        const float *in = src + (size_t)i * cw;
        float *row = Sc + (size_t)i * cw;
        const float32x4_t ki = vdupq_n_f32(k[i]);
        for (int u = 0; u < cw / 4; u++) {
            const float32x4_t r = vmulq_f32(vld1q_f32(in + 4 * u), dec);
            vst1q_f32(row + 4 * u, r);
            acc[u] = vaddq_f32(acc[u], vmulq_f32(r, ki));
        }
    }
    const float32x4_t b = vdupq_n_f32(beta);
    for (int u = 0; u < cw / 4; u++) { /* delta = (v - kv) * beta */
        acc[u] = vmulq_f32(vsubq_f32(vld1q_f32(v + c + 4 * u), acc[u]), b);
        vst1q_f32(delta + 4 * u, acc[u]);
    }
    float32x4_t out[CW / 4];
    for (int u = 0; u < cw / 4; u++) out[u] = vdupq_n_f32(0.0f);
    for (int i = 0; i < dk; i++) { /* S[i] += k[i] * delta; o += S[i] * q[i] (new S) */
        float *row = Sc + (size_t)i * cw;
        const float32x4_t ki = vdupq_n_f32(k[i]), qi = vdupq_n_f32(q[i]);
        for (int u = 0; u < cw / 4; u++) {
            const float32x4_t r = vaddq_f32(vld1q_f32(row + 4 * u), vmulq_f32(ki, acc[u]));
            vst1q_f32(row + 4 * u, r);
            out[u] = vaddq_f32(out[u], vmulq_f32(r, qi));
        }
    }
    for (int u = 0; u < cw / 4; u++) vst1q_f32(o + c + 4 * u, out[u]);
}

/* One logged rank-1 update on columns [c, c + CW): row = row * decay + k[i] * delta, two
 * roundings (pf_lin_apply). u = the log row k | delta | decay. */
INLINE void apply_chunk(float *Sc, const float *u, int c, int dk, int dv) {
    const float32x4_t dec = vdupq_n_f32(u[dk + dv]);
    float32x4_t dl[CW / 4];
    for (int w = 0; w < CW / 4; w++) dl[w] = vld1q_f32(u + dk + c + 4 * w);
    for (int i = 0; i < dk; i++) {
        float *row = Sc + (size_t)i * CW;
        const float32x4_t ki = vdupq_n_f32(u[i]);
        for (int w = 0; w < CW / 4; w++) {
            const float32x4_t r = vmulq_f32(vld1q_f32(row + 4 * w), dec);
            vst1q_f32(row + 4 * w, vaddq_f32(r, vmulq_f32(ki, dl[w])));
        }
    }
}

static void rule_task(void *arg, int tid, int nthr) {
    const kern_lin_args *a = arg;
    const int n = a->n, dk = a->dk, dv = a->dv, kd = (a->heads / a->rep) * dk;
    const size_t hs = (size_t)dk * dv, logw = (size_t)dk + dv + 1;
    float tmp[CW];
    /* logged pass: the chunk's state after tokens 1.. (one chunk at a time, stays in L1) */
    float work[PF_LIN_MAX_DK * CW] __attribute__((aligned(64)));
    for (int h = tid; h < a->heads; h += nthr) {
        const int kh = h / a->rep;
        float *S = a->S + h * hs;
        if (a->log) /* k and decay of tokens 1..; delta is filled chunk by chunk below */
            for (int t = 1; t < n; t++) {
                float *u = a->log + ((size_t)t * a->heads + h) * logw;
                memcpy(u, a->qkv + (size_t)t * a->qkv_stride + kd + kh * dk, (size_t)dk * sizeof(float));
                u[dk + dv] = a->decay[(size_t)t * a->heads + h];
            }
        for (int c = 0; c < dv; c += CW) {
            float *Sc = S + (size_t)(c / CW) * dk * CW;
            for (int t = 0; t < n; t++) {
                const float *qkv = a->qkv + (size_t)t * a->qkv_stride;
                const int copy = a->log && t > 0; /* token 0 is committed in place */
                float *dst = copy ? work : Sc, *delta = tmp;
                if (copy) delta = a->log + ((size_t)t * a->heads + h) * logw + dk + c;
                delta_chunk(dst, t == 1 ? Sc : dst, qkv + kh * dk, qkv + kd + kh * dk,
                            qkv + 2 * kd + h * dv, a->decay[(size_t)t * a->heads + h],
                            a->beta[(size_t)t * a->heads + h],
                            a->o + (size_t)t * a->o_stride + h * dv, delta, c, dk,
                            a->pend ? a->pend + (size_t)h * logw : NULL, (size_t)a->heads * logw,
                            t == 0 ? a->keep : 0, dv);
            }
        }
    }
}

void kern_lin_rule(pf_pool *pool, const kern_lin_args *a) {
    if (a->log && a->dk > PF_LIN_MAX_DK) abort(); /* pf_lin_log_alloc refuses such models */
    kern_pool_run(pool, rule_task, (void *)a, a->heads);
}

typedef struct {
    float *S;
    const float *log;
    int keep, heads, dk, dv;
} apply_job;

/* Commit replay (pf_lin_flush), chunked like the rule so the chunk stays in L1 across the
 * kept tokens. */
static void apply_task(void *arg, int tid, int nthr) {
    const apply_job *j = arg;
    const int dk = j->dk, dv = j->dv;
    const size_t logw = (size_t)dk + dv + 1;
    for (int h = tid; h < j->heads; h += nthr) {
        float *S = j->S + (size_t)h * dk * dv;
        for (int c = 0; c < dv; c += CW)
            for (int t = 0; t < j->keep; t++)
                apply_chunk(S + (size_t)(c / CW) * dk * CW,
                            j->log + ((size_t)t * j->heads + h) * logw, c, dk, dv);
    }
}

void kern_lin_apply(pf_pool *pool, float *S, const float *log, int keep, int heads, int dk,
                    int dv) {
    apply_job j = {S, log, keep, heads, dk, dv};
    if (keep > 0) kern_pool_run(pool, apply_task, &j, heads);
}
