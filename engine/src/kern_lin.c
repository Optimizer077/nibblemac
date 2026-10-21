/* Gated DeltaNet recurrence (forward_lin.c delta_rule) as a NEON kernel threaded over heads.
 *
 * Per head and token, with S [dk][dv]:
 *   S *= decay;  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
 * Two passes over S exactly like the scalar code: (decay, accumulate kv) then (rank-1
 * update, accumulate o). Every per-element operation keeps the scalar rounding (separate
 * multiply and add, never a fused multiply-add) and the per-column accumulation order over
 * the key index i, so the results, the updated state and the rollback log are bit-identical
 * to forward_lin.c and to pf_lin_commit's replay.
 *
 * The recurrence is separable by value column j, so S is processed in 32-column chunks: the
 * chunk's kv / o accumulators live in registers across the 128 key rows, and the chunk loop
 * is outside the token loop, so for n > 1 a 16 KB chunk of the 64 KB head state stays in L1
 * for all tokens and both passes. For that the state is stored chunk-major,
 * S[h][j / 32][i][j % 32] (each chunk contiguous): with the natural [i][j] layout the rows of
 * a chunk are 512 B apart and map onto only 64 of the 256 sets of the 2-way L1, which
 * thrashed (v2 measured L2-bound). The layout is private to the pf_lin_* contract (the state
 * is only ever zeroed outside it). Heads are independent and run in parallel on the pool.
 */
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

#include "kern.h"

#define INLINE static inline __attribute__((always_inline))

#define CW 32 /* columns per chunk: 8 float32x4 accumulators, 16 KB of state per chunk */

/* Columns [c, c + CW) of one head for one token; Sc = the chunk's [dk][CW] block.
 * delta[0..CW) receives the chunk's delta. */
INLINE void delta_chunk(float *Sc, const float *q, const float *k, const float *v, float decay,
                        float beta, float *o, float *delta, int c, int dk) {
    const int cw = CW;
    const float32x4_t dec = vdupq_n_f32(decay);
    float32x4_t acc[CW / 4];
    for (int u = 0; u < cw / 4; u++) acc[u] = vdupq_n_f32(0.0f);
    for (int i = 0; i < dk; i++) { /* S *= decay; kv += S[i] * k[i] (new S) */
        float *row = Sc + (size_t)i * cw;
        const float32x4_t ki = vdupq_n_f32(k[i]);
        for (int u = 0; u < cw / 4; u++) {
            const float32x4_t r = vmulq_f32(vld1q_f32(row + 4 * u), dec);
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

/* Scratch state for logged (verification) passes: one copy per head, grown on demand. */
static float *scratch_state;
static size_t scratch_floats;

static float *scratch_get(size_t floats) {
    if (floats > scratch_floats) {
        free(scratch_state);
        scratch_state = aligned_alloc(64, ((floats * sizeof(float)) + 63) & ~(size_t)63);
        scratch_floats = scratch_state ? floats : 0;
        if (!scratch_state) abort();
    }
    return scratch_state;
}

typedef struct {
    const kern_lin_args *a;
    float *scratch; /* NULL when updating a->S in place */
} rule_job;

static void rule_task(void *arg, int tid, int nthr) {
    const rule_job *j = arg;
    const kern_lin_args *a = j->a;
    const int n = a->n, dk = a->dk, dv = a->dv, kd = (a->heads / a->rep) * dk;
    const size_t hs = (size_t)dk * dv, logw = (size_t)dk + dv + 1;
    float tmp[CW];
    for (int h = tid; h < a->heads; h += nthr) {
        const int kh = h / a->rep;
        float *S = a->S + h * hs;
        if (j->scratch) {
            memcpy(j->scratch + h * hs, S, hs * sizeof(float));
            S = j->scratch + h * hs;
        }
        if (a->log) /* k and decay of every token; delta is filled chunk by chunk below */
            for (int t = 0; t < n; t++) {
                float *u = a->log + ((size_t)t * a->heads + h) * logw;
                memcpy(u, a->qkv + (size_t)t * a->qkv_stride + kd + kh * dk, (size_t)dk * sizeof(float));
                u[dk + dv] = a->decay[(size_t)t * a->heads + h];
            }
        for (int c = 0; c < dv; c += CW)
            for (int t = 0; t < n; t++) {
                const float *qkv = a->qkv + (size_t)t * a->qkv_stride;
                float *delta = a->log ? a->log + ((size_t)t * a->heads + h) * logw + dk + c : tmp;
                delta_chunk(S + (size_t)(c / CW) * dk * CW, qkv + kh * dk, qkv + kd + kh * dk,
                            qkv + 2 * kd + h * dv, a->decay[(size_t)t * a->heads + h],
                            a->beta[(size_t)t * a->heads + h],
                            a->o + (size_t)t * a->o_stride + h * dv, delta, c, dk);
            }
    }
}

void kern_lin_rule(pf_pool *pool, const kern_lin_args *a) {
    rule_job j = {a, NULL};
    if (a->log) j.scratch = scratch_get((size_t)a->heads * a->dk * a->dv);
    kern_pool_run(pool, rule_task, &j, a->heads);
}

typedef struct {
    float *S;
    const float *log;
    int keep, heads, dk, dv;
} apply_job;

/* Commit replay, chunked like the rule so the chunk stays in L1 across the kept tokens. */
static void apply_task(void *arg, int tid, int nthr) {
    const apply_job *j = arg;
    const int dk = j->dk, dv = j->dv;
    const size_t logw = (size_t)dk + dv + 1;
    for (int h = tid; h < j->heads; h += nthr) {
        float *S = j->S + (size_t)h * dk * dv;
        for (int c = 0; c < dv; c += CW)
            for (int t = 0; t < j->keep; t++) {
                const float *u = j->log + ((size_t)t * j->heads + h) * logw, *d = u + dk + c;
                const float32x4_t dec = vdupq_n_f32(u[dk + dv]);
                float32x4_t dl[CW / 4];
                for (int w = 0; w < CW / 4; w++) dl[w] = vld1q_f32(d + 4 * w);
                for (int i = 0; i < dk; i++) { /* row = row * decay + k[i] * delta, two roundings */
                    float *row = S + (size_t)(c / CW) * dk * CW + (size_t)i * CW;
                    const float32x4_t ki = vdupq_n_f32(u[i]);
                    for (int w = 0; w < CW / 4; w++) {
                        const float32x4_t r = vmulq_f32(vld1q_f32(row + 4 * w), dec);
                        vst1q_f32(row + 4 * w, vaddq_f32(r, vmulq_f32(ki, dl[w])));
                    }
                }
            }
    }
}

void kern_lin_apply(pf_pool *pool, float *S, const float *log, int keep, int heads, int dk,
                    int dv) {
    apply_job j = {S, log, keep, heads, dk, dv};
    if (keep > 0) kern_pool_run(pool, apply_task, &j, heads);
}
