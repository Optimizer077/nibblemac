/* Gated DeltaNet (linear attention) mixer of Qwen3.5, plus the rollback log used by
 * speculative decoding.
 *
 * Reference: transformers 5.20.2 Qwen3_5GatedDeltaNet with torch_recurrent_gated_delta_rule
 * (the decode path). The chunked prefill kernel in transformers is the same recurrence
 * re-associated, so running the recurrent form for every token is exact, just O(T) serial.
 *
 * Loop order: first every token's conv / norms / gates, then head-outer, token-inner for the
 * recurrence, so one head's 64 KB state stays in L2 for the whole chunk instead of being
 * streamed from DRAM once per token. Heads are independent, so results are bit-identical to
 * the token-outer order. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"

/* torch.nn.functional.softplus (beta 1, threshold 20). libm here: 16 values per layer. */
static float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

typedef struct {
    pinfer_ctx *ctx;
    const pf_layer *L;
    float *conv;
    int n;
} lin_job;

/* Conv + SiLU on a channel range per thread (channels are independent; the range bounds
 * are multiples of lk_dim, so every q/k head lies inside one range and its L2 norm can
 * follow in the same thread), then the per-head gates on thread 0. Token order inside a
 * channel is kept (the conv state is sequential per channel). */
static void pre_task(void *arg, int tid, int nthr) {
    const lin_job *j = arg;
    pinfer_ctx *ctx = j->ctx;
    const int C = ctx->conv_dim, dk = ctx->lk_dim, nv = ctx->lv_heads, units = C / dk;
    const int c0 = units * tid / nthr * dk, c1 = tid + 1 == nthr ? C : units * (tid + 1) / nthr * dk;
    const float qscale = 1.0f / sqrtf((float)dk);
    for (int t = 0; t < j->n; t++) {
        float *qkv = ctx->ta + (size_t)t * C;
        pf_conv_silu(qkv, j->conv, j->L->conv_w, C, ctx->conv_k, c0, c1);
        /* q heads 0..lk_heads-1 then k heads: head u covers channels [u*dk, (u+1)*dk) */
        for (int u = c0 / dk; u < 2 * ctx->lk_heads && (u + 1) * dk <= c1; u++)
            pf_l2norm_scale(qkv + u * dk, dk, 1e-6f, u < ctx->lk_heads ? qscale : 1.0f);
    }
    if (tid != 0) return;
    for (int t = 0; t < j->n; t++)
        for (int h = 0; h < nv; h++) {  /* beta = sigmoid(b); decay = exp(g), libm (16/layer) */
            float *b = ctx->ba + (size_t)t * nv + h, *a = ctx->aa + (size_t)t * nv + h;
            *b = 1.0f / (1.0f + expf(-*b));
            *a = expf(-expf(j->L->A_log[h]) * softplus(*a + j->L->dt_bias[h]));
        }
}

/* Gated RMSNorm: rmsnorm(o) * w * silu(z) per (token, head), heads split over threads. */
static void post_task(void *arg, int tid, int nthr) {
    const lin_job *j = arg;
    pinfer_ctx *ctx = j->ctx;
    const int vd = ctx->vd, dv = ctx->lv_dim;
    const float eps = ctx->model.hdr->rms_eps;
    for (int h = tid; h < ctx->lv_heads; h += nthr)
        for (int t = 0; t < j->n; t++) {
            float *oh = ctx->to + (size_t)t * vd + h * dv;
            pf_rmsnorm(oh, oh, j->L->lin_norm, dv, eps);
            pf_mul_silu(oh, ctx->tb + (size_t)t * vd + h * dv, dv);
        }
}

void pf_linear_attn(pinfer_ctx *ctx, int layer, int n) {
    const pf_layer *L = &ctx->model.layers[layer];
    const int C = ctx->conv_dim, vd = ctx->vd, nv = ctx->lv_heads;
    const int dk = ctx->lk_dim, dv = ctx->lv_dim, rep = nv / ctx->lk_heads;
    float *rec = ctx->rec[layer], *conv = ctx->conv[layer];
    pf_lin_log *lg = ctx->lin_log_on ? ctx->lin_log : NULL;

    pf_matmul(ctx->pool, &L->qkv, ctx->xn, ctx->ta, n);
    pf_matmul(ctx->pool, &L->z, ctx->xn, ctx->tb, n);
    pf_matmul(ctx->pool, &L->b, ctx->xn, ctx->ba, n);
    pf_matmul(ctx->pool, &L->a, ctx->xn, ctx->aa, n);

    double ts = pf_stats_now();
    if (lg) {  /* verification pass: work on a copy, keep the raw inputs for pf_lin_commit */
        memcpy(lg->conv, conv, (size_t)(ctx->conv_k - 1) * C * sizeof(float));
        conv = lg->conv;
        memcpy(lg->xin[layer], ctx->ta, (size_t)n * C * sizeof(float));
        lg->n = n;
        pf_stats_add(PFS_LIN_LOG, PFS_KV, ts);
        ts = pf_stats_now();
    }

    /* Per token: conv + SiLU (shifts the conv state), q/k L2 norm (eps 1e-6 as in the
     * reference's l2norm), q scale; beta and decay overwrite ba / aa. */
    lin_job job = {ctx, L, conv, n};
    pf_pool_run(ctx->pool, pre_task, &job, C / dk);
    pf_stats_add(PFS_LIN_PRE, PFS_KIND(ctx, n), ts);
    ts = pf_stats_now();

    /* Recurrence of every head over all tokens (pf_lin_rule: scalar reference or the
     * threaded NEON kernel); with the log on it leaves `rec` untouched and records
     * k | delta | decay per (token, head) for pf_lin_commit. */
    pf_lin_args a = {n, nv, rep, dk, dv, C, vd, ctx->ta, ctx->aa, ctx->ba, rec, ctx->to,
                     lg ? lg->upd[layer] : NULL};
    pf_lin_rule(ctx->pool, &a);
    pf_stats_add(PFS_LIN_RULE, PFS_KIND(ctx, n), ts);
    ts = pf_stats_now();
    pf_pool_run(ctx->pool, post_task, &job, nv);  /* gated RMSNorm, plain weight */
    pf_stats_add(PFS_LIN_POST, PFS_KIND(ctx, n), ts);
    pf_matmul(ctx->pool, &L->out, ctx->to, ctx->out, n);
}

/* ---- speculative rollback ------------------------------------------------------------ */

void pf_lin_log_free(pinfer_ctx *ctx) {
    pf_lin_log *lg = ctx->lin_log;
    if (!lg) return;
    for (int l = 0; l < ctx->n_layers; l++) {
        if (lg->xin) free(lg->xin[l]);
        if (lg->upd) free(lg->upd[l]);
    }
    free(lg->xin);
    free(lg->upd);
    free(lg->S);
    free(lg->conv);
    free(lg);
    ctx->lin_log = NULL;
    ctx->lin_log_on = 0;
}

int pf_lin_log_alloc(pinfer_ctx *ctx, int cap) {
    if (ctx->lin_log && ctx->lin_log->cap >= cap) return PINFER_OK;
    pf_lin_log_free(ctx);
    pf_lin_log *lg = calloc(1, sizeof *lg);
    if (!lg) return PINFER_E_NOMEM;
    ctx->lin_log = lg;
    lg->cap = cap;
    const size_t upd_row = (size_t)ctx->lv_heads * (ctx->lk_dim + ctx->lv_dim + 1);
    lg->S = malloc((size_t)ctx->lk_dim * ctx->lv_dim * sizeof(float));
    lg->conv = malloc(((size_t)(ctx->conv_k - 1) * ctx->conv_dim + 1) * sizeof(float));
    lg->xin = calloc((size_t)ctx->n_layers, sizeof(float *));
    lg->upd = calloc((size_t)ctx->n_layers, sizeof(float *));
    int ok = lg->S && lg->conv && lg->xin && lg->upd;
    for (int l = 0; ok && l < ctx->n_layers; l++) {
        if (ctx->model.layers[l].is_full) continue;
        lg->xin[l] = malloc((size_t)cap * ctx->conv_dim * sizeof(float));
        lg->upd[l] = malloc((size_t)cap * upd_row * sizeof(float));
        ok = lg->xin[l] && lg->upd[l];
    }
    if (!ok) { pf_lin_log_free(ctx); return PINFER_E_NOMEM; }
    return PINFER_OK;
}

/* Advance the committed states by the first `keep` tokens of the logged pass. The replay
 * performs exactly the per-element operations of delta_rule (S = S*decay, then
 * S = S + k_i*delta_j, both rounded to fp32), so the result is bit-identical to running
 * those tokens directly. Cost: one read + write of the state per layer (~19 MB in total). */
void pf_lin_commit(pinfer_ctx *ctx, int keep) {
    pf_lin_log *lg = ctx->lin_log;
    const int C = ctx->conv_dim, S = ctx->conv_k - 1, nv = ctx->lv_heads;
    const int dk = ctx->lk_dim, dv = ctx->lv_dim;
    if (!lg || keep <= 0 || keep > lg->n) return;
    for (int l = 0; l < ctx->n_layers; l++) {
        if (ctx->model.layers[l].is_full) continue;
        /* conv: new state = last S rows of [old state rows, raw inputs 0..keep-1]; ascending
         * j reads row j + keep >= j before anything at or after it is overwritten */
        float *st = ctx->conv[l];
        for (int j = 0; j < S; j++) {
            int src = j + keep;
            const float *from = src < S ? st + (size_t)src * C
                                        : lg->xin[l] + (size_t)(src - S) * C;
            memcpy(st + (size_t)j * C, from, (size_t)C * sizeof(float));
        }
        pf_lin_apply(ctx->pool, ctx->rec[l], lg->upd[l], keep, nv, dk, dv);
    }
}
