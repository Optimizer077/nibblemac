/* Gated full attention over the KV cache (pf_full_attn) for 1..n tokens at positions pos0..
 *
 * Per block of up to PF_ATTN_TB tokens, three parallel passes:
 *   1. scores (score_task), threads over (KV group, half of the rows 0..pos_last): each K row
 *      is read once and dotted with the 4 query heads of its group (pf_scores4) for every
 *      token of the block, the 4 queries interleaved (one stream); a single token reads the
 *      rows in place, the two threads of a group taking alternate blocks of SC_ALT rows (side
 *      by side in DRAM: 3.95 instead of 3.75 GB/s, CONTEXT.md §9), a batch in L1 blocks of
 *      PV_ROWS rows shared by its tokens;
 *   2. softmax (softmax_task), threads over (token, head): exactly as before (running max,
 *      pf_softmax_exp, * 1/sum);
 *   3. P.V + gate (pv_task), threads over (KV group, column range of the head dimension):
 *      every V element is read by one thread and accumulated into the 4 heads of its group,
 *      row by row, for every token of the block (pf_attn_pv4, output chunks in registers).
 * Per element nothing changed against the one-head-at-a-time loop of CORE.md round 2/3:
 * pf_scores4 keeps pf_dot's lane order, pf_attn_pv4 keeps pf_attn_pv's row order, and the
 * row and column splits only distribute independent elements. So the outputs are
 * bit-identical to it and, as before, independent of n and of the thread count. The cache is
 * streamed once per block (the old loop: once per query head and token).
 * KV rows are stored as fp32, fp16 or int8 (ctx->kv_dtype, forward_kv.c). The kernels read
 * fp32 rows and, for single-token steps, fp16 rows in place (widened in registers, exact);
 * otherwise a block of rows is widened once into an L1 buffer and reused by every token of
 * the block. Variants measured and not kept: CONTEXT.md §3d and §9. */
#include <math.h>
#include <string.h>

#include "forward.h"

#define PV_ROWS 8   /* K / V rows per L1 block (8 KB at hd = 256) */
#define SC_ALT 16   /* single-token scores: rows per alternate block of a group's two threads */
#define QI_MAX_HD 256   /* interleaved queries (32 KB on the stack for a batch) up to this hd */

typedef struct {
    pinfer_ctx *ctx;
    const float *kc, *vc;
    int t0, n, pos0;   /* block: ctx rows t0.., n tokens, first at position pos0 */
    int ds;            /* column ranges per KV group in pass 3 */
} attn_job;

/* stand-ins for the missing heads when a group has no multiple of 4 heads: zero queries /
 * probabilities (any finite values would do), results written to scratch */
static const float zeros[PF_KV_MAX_HD];

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

/* Rows of the cache read in place: fp32 always, fp16 for a single token (widened in
 * registers). Otherwise kv_rows widens a block into an L1 buffer once for all tokens. */
static inline int in_place(const attn_job *j) {
    return j->ctx->kv_dtype == PF_KV_F32 || (j->ctx->kv_dtype == PF_KV_F16 && j->n == 1);
}

/* Rows [pb, pe) of group g as the kernels read them: the cache rows themselves (in_place),
 * else columns [d0, d1) widened into buf [PV_ROWS][hd]. *f16: the rows are binary16. */
static const void *kv_rows(const attn_job *j, const void *cache, int g, int pb, int pe, int d0,
                           int d1, float *buf, int *f16) {
    const pinfer_ctx *ctx = j->ctx;
    if (in_place(j)) {
        *f16 = ctx->kv_dtype == PF_KV_F16;
        return pf_kv_row(ctx, cache, g, pb);
    }
    *f16 = 0;
    for (int p = pb; p < pe; p++)
        pf_kv_widen(ctx, cache, g, p, d0, d1, buf + (size_t)(p - pb) * ctx->hd);
    return buf;
}

/* PRFM of columns [d0, d1) of rows [p0, p1): pass 3 walks a block 16 columns at a time
 * across its rows, an order the hardware prefetcher does not follow, so the next block is
 * requested while this one is computed (-10% / -12% P.V time fp32 / fp16 at 4096 tokens,
 * §9; unlike the row-sequential passes of §3d, where prefetching lost). */
static void prefetch_rows(const pinfer_ctx *ctx, const void *cache, int g, int p0, int p1,
                          int d0, int d1) {
    const size_t es = pf_kv_row_bytes(ctx->kv_dtype, ctx->hd) / (size_t)ctx->hd;
    for (int p = p0; p < p1; p++) {
        const char *r = pf_kv_row(ctx, cache, g, p);
        for (size_t b = (size_t)d0 * es; b < (size_t)d1 * es; b += 64) __builtin_prefetch(r + b);
    }
}

/* att[t][h][p] = (q . K[p]) * scale for the unit's rows and every token / head of group g.
 * The queries are read interleaved by 4 (pf_q4_interleave, one base pointer in the dots: 4
 * query streams next to the K rows cost up to 50% more, §9) when head_dim <= QI_MAX_HD; a
 * batch needs one quad per group for that (its tokens' queries are interleaved up front). */
static void score_task(void *arg, int tid, int nthr) {
    const attn_job *j = arg;
    const pinfer_ctx *ctx = j->ctx;
    const int hd = ctx->hd, grp = ctx->n_heads / ctx->n_kv, np = j->pos0 + j->n;
    const int units = 2 * ctx->n_kv, half = (np + 1) / 2;
    const int iq = hd <= QI_MAX_HD, iqb = iq && grp <= 4;   /* single token / batch */
    const float scale = 1.0f / sqrtf((float)hd);
    float buf[PV_ROWS * PF_KV_MAX_HD], pad[PV_ROWS], qi[PF_ATTN_TB * 4 * QI_MAX_HD];
    for (int u = units * tid / nthr; u < units * (tid + 1) / nthr; u++) {
        const int g = u >> 1, h1 = (g + 1) * grp, p1 = (u & 1) ? np : half;
        if (j->n == 1 && in_place(j)) {
            /* single token: alternate blocks of SC_ALT rows per query quad */
            float *sink = att_row(ctx, 1, g);   /* missing heads' scores: a row token 1 owns */
            for (int h = g * grp; h < h1; h += 4) {
                const float *q[4];
                float *att[4];
                for (int i = 0; i < 4; i++) {
                    q[i] = h + i < h1 ? q_vec(ctx, j->t0, h + i) : zeros;
                    att[i] = h + i < h1 ? att_row(ctx, 0, h + i) : sink;
                }
                if (iq) pf_q4_interleave(q, hd, qi);
                for (int b = (u & 1) * SC_ALT; b < np; b += 2 * SC_ALT) {
                    float *ab[4] = {att[0] + b, att[1] + b, att[2] + b, att[3] + b};
                    pf_scores4(iq ? qi : NULL, q, pf_kv_row(ctx, j->kc, g, b),
                               ctx->kv_dtype == PF_KV_F16,
                               (b + SC_ALT < np ? b + SC_ALT : np) - b, hd, scale, ab);
                }
            }
            continue;
        }
        if (iqb)   /* one quad: h = g * grp below */
            for (int t = 0; t < j->n; t++) {
                const float *q[4];
                for (int i = 0; i < 4; i++)
                    q[i] = g * grp + i < h1 ? q_vec(ctx, j->t0 + t, g * grp + i) : zeros;
                pf_q4_interleave(q, hd, qi + (size_t)t * 4 * hd);
            }
        for (int pb = (u & 1) * half; pb < p1; pb += PV_ROWS) {
            const int pe = pb + PV_ROWS < p1 ? pb + PV_ROWS : p1;
            int f16;
            const void *rows = kv_rows(j, j->kc, g, pb, pe, 0, hd, buf, &f16);
            for (int t = 0; t < j->n; t++) {
                const int pos = j->pos0 + t, cnt = (pe < pos + 1 ? pe : pos + 1) - pb; /* causal */
                for (int h = g * grp; h < h1 && cnt > 0; h += 4) {
                    const float *q[4];
                    float *att[4];
                    for (int i = 0; i < 4; i++) {
                        q[i] = h + i < h1 ? q_vec(ctx, j->t0 + t, h + i) : zeros;
                        att[i] = h + i < h1 ? att_row(ctx, t, h + i) + pb : pad;
                    }
                    pf_scores4(iqb ? qi + (size_t)t * 4 * hd : NULL, q, rows, f16, cnt, hd, scale,
                               att);
                }
            }
        }
    }
}

/* Softmax over att[t][h][0..pos] per (token, head) of the block. */
static void softmax_task(void *arg, int tid, int nthr) {
    const attn_job *j = arg;
    const pinfer_ctx *ctx = j->ctx;
    const int units = j->n * ctx->n_heads;
    for (int u = units * tid / nthr; u < units * (tid + 1) / nthr; u++) {
        const int t = u / ctx->n_heads, pos = j->pos0 + t;
        float *att = att_row(ctx, t, u % ctx->n_heads);
        const float inv = 1.0f / pf_softmax_exp(att, pos + 1, pf_softmax_max(att, pos + 1));
        for (int p = 0; p <= pos; p++) att[p] = att[p] * inv;
    }
}

/* o[t][h][d0, d1) = P.V, then the output gate, for the unit's column range of group g and
 * every head of the group and token of the block. */
static void pv_task(void *arg, int tid, int nthr) {
    const attn_job *j = arg;
    const pinfer_ctx *ctx = j->ctx;
    const int hd = ctx->hd, grp = ctx->n_heads / ctx->n_kv, np = j->pos0 + j->n;
    const int units = ctx->n_kv * j->ds, nch = (hd + 15) / 16, inplace = in_place(j);
    float buf[PV_ROWS * PF_KV_MAX_HD], scratch[PF_KV_MAX_HD];
    memset(scratch, 0, sizeof scratch);
    for (int u = units * tid / nthr; u < units * (tid + 1) / nthr; u++) {
        const int g = u / j->ds, s = u % j->ds, h1 = (g + 1) * grp;
        const int d0 = nch * s / j->ds * 16, d1 = nch * (s + 1) / j->ds * 16 < hd ?
                                                  nch * (s + 1) / j->ds * 16 : hd;
        if (d1 <= d0) continue;
        for (int t = 0; t < j->n; t++)
            for (int h = g * grp; h < h1; h++)
                memset(o_vec(ctx, j->t0 + t, h) + d0, 0, (size_t)(d1 - d0) * sizeof(float));
        for (int pb = 0; pb < np; pb += PV_ROWS) {
            const int pe = pb + PV_ROWS < np ? pb + PV_ROWS : np;
            if (inplace && pe < np)
                prefetch_rows(ctx, j->vc, g, pe, pe + PV_ROWS < np ? pe + PV_ROWS : np, d0, d1);
            int f16;
            const void *rows = kv_rows(j, j->vc, g, pb, pe, d0, d1, buf, &f16);
            for (int t = 0; t < j->n; t++) {
                const int cnt = (j->pos0 + t + 1 < pe ? j->pos0 + t + 1 : pe) - pb;
                for (int h = g * grp; h < h1 && cnt > 0; h += 4) {
                    float *o[4];
                    const float *p[4];
                    for (int i = 0; i < 4; i++) {
                        o[i] = h + i < h1 ? o_vec(ctx, j->t0 + t, h + i) : scratch;
                        p[i] = h + i < h1 ? att_row(ctx, t, h + i) + pb : zeros;
                    }
                    pf_attn_pv4(o, p, cnt, rows, f16, hd, d0, d1);
                }
            }
        }
        for (int t = 0; t < j->n; t++)
            for (int h = g * grp; h < h1; h++)
                pf_mul_sigmoid(o_vec(ctx, j->t0 + t, h) + d0, q_vec(ctx, j->t0 + t, h) + hd + d0,
                               d1 - d0);
    }
}

/* Attention of the n tokens at pos0.. (queries in ctx->ta, rows [0, pos0 + n) in the cache)
 * into ctx->to, block by block; ts = start of the profiled section (engine/bench/abench.c
 * times this part alone). */
static void attend(pinfer_ctx *ctx, const float *kc, const float *vc, int n, int pos0,
                   double ts) {
    const int nch = (ctx->hd + 15) / 16, nthr = pf_pool_threads(ctx->pool);
    /* enough column ranges for every thread (the split does not change any element) */
    int ds = (nthr + ctx->n_kv - 1) / ctx->n_kv;
    ds = ds < 1 ? 1 : ds > nch ? nch : ds;
    for (int t0 = 0; t0 < n; t0 += PF_ATTN_TB) {
        const int nb = n - t0 < PF_ATTN_TB ? n - t0 : PF_ATTN_TB;
        attn_job job = {ctx, kc, vc, t0, nb, pos0 + t0, ds};
        pf_pool_run(ctx->pool, score_task, &job, 2 * ctx->n_kv);
        pf_stats_add(PFS_ATTN, PFS_KIND(ctx, n), ts);
        ts = pf_stats_now();
        pf_pool_run(ctx->pool, softmax_task, &job, nb * ctx->n_heads);
        pf_pool_run(ctx->pool, pv_task, &job, ctx->n_kv * ds);
        pf_stats_add(PFS_ATTN_PV, PFS_KIND(ctx, n), ts);
        ts = pf_stats_now();
    }
}

void pf_full_attn(pinfer_ctx *ctx, const pf_layer *L, float *kc, float *vc, int n, int pos0) {
    const int hd = ctx->hd, kvw = ctx->n_kv * hd, h2 = ctx->rot / 2, nq = ctx->n_heads;
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
    attend(ctx, kc, vc, n, pos0, ts);
    pf_matmul(ctx->pool, &L->o, ctx->to, ctx->out, n);
}
