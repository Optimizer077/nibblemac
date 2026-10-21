/* Gated full attention over the KV cache (pf_full_attn) for 1..n tokens at positions pos0..
 *
 * Per block of up to PF_ATTN_TB tokens, two parallel passes over the cache:
 *   1. scores (score_task), threads over (KV group, half of the rows 0..pos_last): each K row
 *      is widened once, then dotted with the 4 query heads of its group (pf_dot4, one load of
 *      the row for 4 dots) for every token of the block, in L1-sized row blocks;
 *   2. softmax + P.V + gate (pv_task), threads over (KV group, half of its query heads): the
 *      per-head softmax exactly as before (running max, pf_softmax_exp, * 1/sum), then V rows
 *      in L1 blocks of PV_ROWS accumulated into every (token, head) of the unit in row order,
 *      two heads per pass with the output chunk in registers (pf_attn_pv_blk).
 * Per element nothing changed against the one-head-at-a-time loop this replaces (CORE.md
 * round 2/3): pf_dot4 keeps pf_dot's lane order, pf_attn_pv_blk keeps pf_attn_pv's row
 * order, so the outputs are bit-identical to it and, as before, independent of n and of the
 * thread count. The gain is traffic: the old loop streamed K and V from DRAM once per query
 * head and per token (8 passes per token), this one once per block (CONTEXT.md).
 * KV rows are stored as fp32, fp16 or int8 (ctx->kv_dtype, forward_kv.c); single-token steps
 * with an fp16 cache widen K inside the dots (pf_dot4_f16, exact). Variants measured and not
 * kept (CONTEXT.md §3d): row-outer P.V, P.V reading the fp32 cache in place (both equal within
 * noise for fp32, 3.5% slower for fp16), software prefetch of the next rows (1-6% slower). */
#include <math.h>
#include <string.h>

#include "forward.h"

#define PV_ROWS 8  /* K / V rows per L1 block (8 KB at hd = 256) */


typedef struct {
    pinfer_ctx *ctx;
    const pf_layer *L;
    const float *kc, *vc;
    int t0, n, pos0;   /* block: ctx rows t0.., n tokens, first at position pos0 */
} attn_job;

static inline float *att_row(const pinfer_ctx *ctx, int t, int h) {
    return ctx->att + ((size_t)t * ctx->n_heads + h) * (size_t)ctx->max_ctx;
}

/* query of (ctx row t, head h): normalised + rotated in place by pf_full_attn; gate at + hd */
static inline float *q_vec(const pinfer_ctx *ctx, int t, int h) {
    return ctx->ta + ((size_t)t * ctx->n_heads + h) * (size_t)(2 * ctx->hd);
}

static inline float *o_vec(const pinfer_ctx *ctx, int t, int h) {
    return ctx->to + ((size_t)t * ctx->n_heads + h) * (size_t)ctx->hd;
}

/* att[t][h][p] = (q . K[p]) * scale for the unit's rows and every token / head of group g. */
static void score_task(void *arg, int tid, int nthr) {
    const attn_job *j = arg;
    const pinfer_ctx *ctx = j->ctx;
    const int hd = ctx->hd, grp = ctx->n_heads / ctx->n_kv, np = j->pos0 + j->n;
    const int units = 2 * ctx->n_kv, half = (np + 1) / 2;
    const float scale = 1.0f / sqrtf((float)hd);
    float buf[PV_ROWS * PF_KV_MAX_HD];
    const float *rows[PV_ROWS];
    /* decode with an fp16 cache: widen K in registers inside the dots (exact, same result) */
    const int fused = j->n == 1 && ctx->kv_dtype == PF_KV_F16 && grp % 4 == 0;
    for (int u = units * tid / nthr; u < units * (tid + 1) / nthr; u++) {
        const int g = u >> 1, p1 = (u & 1) ? np : half;
        for (int pb = (u & 1) * half; pb < p1; pb += PV_ROWS) {
            const int pe = pb + PV_ROWS < p1 ? pb + PV_ROWS : p1;
            if (fused) {
                const int pos = j->pos0, pt = pe < pos + 1 ? pe : pos + 1;
                for (int h = g * grp; h < (g + 1) * grp; h += 4) {
                    const float *q[4] = {q_vec(ctx, j->t0, h), q_vec(ctx, j->t0, h + 1),
                                         q_vec(ctx, j->t0, h + 2), q_vec(ctx, j->t0, h + 3)};
                    float s[4];
                    for (int p = pb; p < pt; p++) {
                        pf_dot4_f16(q, pf_kv_row(ctx, j->kc, g, p), hd, s);
                        for (int i = 0; i < 4; i++) att_row(ctx, 0, h + i)[p] = s[i] * scale;
                    }
                }
                continue;
            }
            for (int p = pb; p < pe; p++)
                rows[p - pb] = pf_kv_load(ctx, j->kc, g, p, buf + (size_t)(p - pb) * hd);
            for (int t = 0; t < j->n; t++) {
                const int pos = j->pos0 + t, pt = pe < pos + 1 ? pe : pos + 1;  /* causal */
                for (int h = g * grp; h < (g + 1) * grp; h += 4) {
                    if (h + 4 <= (g + 1) * grp) {
                        const float *q[4] = {q_vec(ctx, j->t0 + t, h), q_vec(ctx, j->t0 + t, h + 1),
                                             q_vec(ctx, j->t0 + t, h + 2), q_vec(ctx, j->t0 + t, h + 3)};
                        float s[4];
                        for (int p = pb; p < pt; p++) {
                            pf_dot4(q, rows[p - pb], hd, s);
                            for (int i = 0; i < 4; i++) att_row(ctx, t, h + i)[p] = s[i] * scale;
                        }
                    } else {  /* group size not a multiple of 4: head by head, same order */
                        for (int i = h; i < (g + 1) * grp; i++) {
                            const float *q = q_vec(ctx, j->t0 + t, i);
                            float *att = att_row(ctx, t, i);
                            for (int p = pb; p < pt; p++) att[p] = pf_dot(q, rows[p - pb], hd) * scale;
                        }
                    }
                }
            }
        }
    }
}

/* Softmax over att[t][h][0..pos], then o[t][h] = P.V and the output gate, for the unit's
 * heads of group g and every token of the block. */
static void pv_task(void *arg, int tid, int nthr) {
    const attn_job *j = arg;
    const pinfer_ctx *ctx = j->ctx;
    const int hd = ctx->hd, grp = ctx->n_heads / ctx->n_kv, np = j->pos0 + j->n;
    const int hs = grp % 2 == 0 ? 2 : 1, units = ctx->n_kv * hs, nh = grp / hs;
    float buf[PV_ROWS * PF_KV_MAX_HD];
    for (int u = units * tid / nthr; u < units * (tid + 1) / nthr; u++) {
        const int g = u / hs, h0 = g * grp + (u % hs) * nh, h1 = h0 + nh;
        for (int t = 0; t < j->n; t++) {
            const int pos = j->pos0 + t;
            for (int h = h0; h < h1; h++) {
                float *att = att_row(ctx, t, h), mx = -INFINITY;
                for (int p = 0; p <= pos; p++)
                    if (att[p] > mx) mx = att[p];
                const float inv = 1.0f / pf_softmax_exp(att, pos + 1, mx);
                for (int p = 0; p <= pos; p++) att[p] = att[p] * inv;
                memset(o_vec(ctx, j->t0 + t, h), 0, (size_t)hd * sizeof(float));
            }
        }
        /* V rows land in buf as one sequential read per block (memcpy for fp32, the widening
         * loads otherwise); the block kernel then works from L1 with o in registers */
        for (int pb = 0; pb < np; pb += PV_ROWS) {
            const int pe = pb + PV_ROWS < np ? pb + PV_ROWS : np;
            const float *rows = buf;
            if (ctx->kv_dtype == PF_KV_F32)
                memcpy(buf, pf_kv_load(ctx, j->vc, g, pb, buf),
                       (size_t)(pe - pb) * hd * sizeof(float));
            else
                for (int p = pb; p < pe; p++) pf_kv_load(ctx, j->vc, g, p, buf + (size_t)(p - pb) * hd);
            for (int t = 0; t < j->n; t++) {
                const int cnt = (j->pos0 + t + 1 < pe ? j->pos0 + t + 1 : pe) - pb;
                if (cnt <= 0) continue;
                int h = h0;
                for (; h + 2 <= h1; h += 2)
                    pf_attn_pv_blk(o_vec(ctx, j->t0 + t, h), att_row(ctx, t, h) + pb,
                                   o_vec(ctx, j->t0 + t, h + 1), att_row(ctx, t, h + 1) + pb,
                                   cnt, rows, hd);
                if (h < h1)
                    pf_attn_pv_blk(o_vec(ctx, j->t0 + t, h), att_row(ctx, t, h) + pb, NULL, NULL,
                                   cnt, rows, hd);
            }
        }
        for (int t = 0; t < j->n; t++)
            for (int h = h0; h < h1; h++)
                pf_mul_sigmoid(o_vec(ctx, j->t0 + t, h), q_vec(ctx, j->t0 + t, h) + hd, hd);
    }
}

void pf_full_attn(pinfer_ctx *ctx, const pf_layer *L, float *kc, float *vc, int n, int pos0) {
    const int hd = ctx->hd, kvw = ctx->n_kv * hd, h2 = ctx->rot / 2, nq = ctx->n_heads;
    const int grp = nq / ctx->n_kv;
    const float eps = ctx->model.hdr->rms_eps;
    float krow[PF_KV_MAX_HD];

    pf_matmul(ctx->pool, &L->q, ctx->xn, ctx->ta, n);   /* per head: [query hd | gate hd] */
    pf_matmul(ctx->pool, &L->k, ctx->xn, ctx->tb, n);
    pf_matmul(ctx->pool, &L->v, ctx->xn, ctx->tc, n);

    double ts = pf_stats_now();
    /* All n tokens' K/V go into the cache first (causal attention includes the token itself
     * and, for later tokens of the chunk, the earlier ones); q norm + RoPE for every head. */
    for (int t = 0; t < n; t++) {
        const float *cs = ctx->rope_cos + t * h2, *sn = ctx->rope_sin + t * h2;
        for (int g = 0; g < ctx->n_kv; g++) {
            pf_rmsnorm(krow, ctx->tb + (size_t)t * kvw + g * hd, L->k_norm, hd, eps);
            pf_rope(krow, cs, sn, ctx->rot);
            pf_kv_store(ctx, kc, g, pos0 + t, krow);
            pf_kv_store(ctx, vc, g, pos0 + t, ctx->tc + (size_t)t * kvw + g * hd);
        }
        for (int h = 0; h < nq; h++) {
            float *q = q_vec(ctx, t, h);
            pf_rmsnorm(q, q, L->q_norm, hd, eps);
            pf_rope(q, cs, sn, ctx->rot);
        }
    }
    for (int t0 = 0; t0 < n; t0 += PF_ATTN_TB) {
        attn_job job = {ctx, L, kc, vc, t0, n - t0 < PF_ATTN_TB ? n - t0 : PF_ATTN_TB, pos0 + t0};
        pf_pool_run(ctx->pool, score_task, &job, 2 * ctx->n_kv);
        pf_stats_add(PFS_ATTN, PFS_KIND(ctx, n), ts);
        ts = pf_stats_now();
        pf_pool_run(ctx->pool, pv_task, &job, ctx->n_kv * (grp % 2 == 0 ? 2 : 1));
        pf_stats_add(PFS_ATTN_PV, PFS_KIND(ctx, n), ts);
        ts = pf_stats_now();
    }
    pf_matmul(ctx->pool, &L->o, ctx->to, ctx->out, n);
}
