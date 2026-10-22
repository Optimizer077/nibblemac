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
    float *conv;        /* the layer's conv state */
    float *copy, *xin;  /* logged pass: conv state copy for tokens 1.., raw inputs (rows 1..) */
    int n;
} lin_job;

/* Conv + SiLU on a channel range per thread (channels are independent; the range bounds
 * are multiples of lk_dim, so every q/k head lies inside one range and its L2 norm can
 * follow in the same thread), then the per-head gates on thread 0. Token order inside a
 * channel is kept (the conv state is sequential per channel). In a logged pass token 0
 * shifts the live state (it is always committed); before token 1 each thread copies its
 * channels of the state, and it keeps its channels of every later raw input for
 * pf_lin_commit. */
static void pre_task(void *arg, int tid, int nthr) {
    const lin_job *j = arg;
    pinfer_ctx *ctx = j->ctx;
    const int C = ctx->conv_dim, dk = ctx->lk_dim, nv = ctx->lv_heads, units = C / dk;
    const int c0 = units * tid / nthr * dk, c1 = tid + 1 == nthr ? C : units * (tid + 1) / nthr * dk;
    const size_t bytes = (size_t)(c1 - c0) * sizeof(float);
    const float qscale = 1.0f / sqrtf((float)dk);
    float *st = j->conv;
    for (int t = 0; t < j->n; t++) {
        float *qkv = ctx->ta + (size_t)t * C;
        if (j->copy && t > 0) {
            if (t == 1)
                for (int r = 0; r < ctx->conv_k - 1; r++)
                    memcpy(j->copy + (size_t)r * C + c0, st + (size_t)r * C + c0, bytes);
            st = j->copy;
            memcpy(j->xin + (size_t)t * C + c0, qkv + c0, bytes);
        }
        pf_conv_silu(qkv, st, j->L->conv_w, C, ctx->conv_k, c0, c1);
        /* q heads 0..lk_heads-1 then k heads: head u covers channels [u*dk, (u+1)*dk) */
        for (int u = c0 / dk; u < 2 * ctx->lk_heads && (u + 1) * dk <= c1; u++)
            pf_l2norm_scale(qkv + u * dk, dk, 1e-6f, u < ctx->lk_heads ? qscale : 1.0f);
    }
    if (tid != 0) return;
    for (int t = 0; t < j->n; t++)
        for (int h = 0; h < nv; h++) {  /* beta = sigmoid(b); decay = exp(g), libm (16/layer) */
            float *b = ctx->ba + (size_t)t * nv + h, *a = ctx->aa + (size_t)t * nv + h;
            *b = 1.0f / (1.0f + expf(-*b));
            /* g = A * softplus(a + dt_bias), A = -exp(A_log) or the stored A (model.h) */
            const float A = j->L->A ? j->L->A[h] : -expf(j->L->A_log[h]);
            *a = expf(A * softplus(*a + j->L->dt_bias[h]));
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

/* Floats of one token's rollback-log rows [lv_heads][lk_dim + lv_dim + 1]. */
static size_t upd_row(const pinfer_ctx *ctx) {
    return (size_t)ctx->lv_heads * (ctx->lk_dim + ctx->lv_dim + 1);
}

/* Half `half` of layer l's rollback log, at token 0. */
static float *upd_half(const pinfer_ctx *ctx, int l, int half) {
    const pf_lin_log *lg = ctx->lin_log;
    return lg->upd[l] + (size_t)half * lg->cap * upd_row(ctx);
}

void pf_linear_attn(pinfer_ctx *ctx, int layer, int n) {
    const pf_layer *L = &ctx->model.layers[layer];
    const int C = ctx->conv_dim, nv = ctx->lv_heads;
    const int dk = ctx->lk_dim, dv = ctx->lv_dim, rep = nv / ctx->lk_heads;
    pf_lin_log *lg = ctx->lin_log;
    const int logged = lg && ctx->lin_log_on;

    pf_matmul(ctx->pool, &L->qkv, ctx->xn, ctx->ta, n);
    pf_matmul(ctx->pool, &L->z, ctx->xn, ctx->tb, n);
    pf_matmul(ctx->pool, &L->b, ctx->xn, ctx->ba, n);
    pf_matmul(ctx->pool, &L->a, ctx->xn, ctx->aa, n);

    /* Per token: conv + SiLU (shifts the conv state), q/k L2 norm (eps 1e-6 as in the
     * reference l2norm), q scale; beta and decay overwrite ba / aa. A logged (verification)
     * pass also keeps what pf_lin_commit needs. */
    double ts = pf_stats_now();
    lin_job job = {ctx, L, ctx->conv[layer], logged ? lg->conv : NULL,
                   logged ? lg->xin[layer] : NULL, n};
    if (logged) lg->n = n;
    pf_pool_run(ctx->pool, pre_task, &job, C / dk);
    pf_stats_add(PFS_LIN_PRE, PFS_KIND(ctx, n), ts);
    ts = pf_stats_now();

    /* Recurrence of every head over all tokens (pf_lin_rule: scalar reference or the
     * threaded NEON kernel), after the updates the last commit deferred to this pass; a
     * logged pass leaves the state after token 0 and records k | delta | decay of tokens 1..
     * for pf_lin_commit. */
    const int keep = lg ? lg->keep[layer] : 0;
    pf_lin_args a = {n, nv, rep, dk, dv, C, ctx->vd, ctx->ta, ctx->aa, ctx->ba,
                     ctx->rec[layer], ctx->to, logged ? upd_half(ctx, layer, lg->wr) : NULL,
                     keep > 0 ? upd_half(ctx, layer, lg->wr ^ 1) + upd_row(ctx) : NULL, keep,
                     lg ? lg->S : NULL};
    pf_lin_rule(ctx->pool, &a);
    if (keep > 0) lg->keep[layer] = 0;
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
    free(lg->keep);
    free(lg->S);
    free(lg->conv);
    free(lg);
    ctx->lin_log = NULL;
    ctx->lin_log_on = 0;
}

int pf_lin_log_alloc(pinfer_ctx *ctx, int cap) {
    if (ctx->lin_log && ctx->lin_log->cap >= cap) return PINFER_OK;
    if (ctx->lk_dim > PF_LIN_MAX_DK) return PINFER_E_FORMAT;  /* logged-pass limit, ops.h */
    pf_lin_flush(ctx);
    pf_lin_log_free(ctx);
    pf_lin_log *lg = calloc(1, sizeof *lg);
    if (!lg) return PINFER_E_NOMEM;
    ctx->lin_log = lg;
    lg->cap = cap;
    lg->S = malloc((size_t)ctx->lk_dim * ctx->lv_dim * sizeof(float));
    lg->conv = malloc(((size_t)(ctx->conv_k - 1) * ctx->conv_dim + 1) * sizeof(float));
    lg->xin = calloc((size_t)ctx->n_layers, sizeof(float *));
    lg->upd = calloc((size_t)ctx->n_layers, sizeof(float *));
    lg->keep = calloc((size_t)ctx->n_layers, sizeof(int));
    int ok = lg->S && lg->conv && lg->xin && lg->upd && lg->keep;
    for (int l = 0; ok && l < ctx->n_layers; l++) {
        if (ctx->model.layers[l].is_full) continue;
        lg->xin[l] = malloc((size_t)cap * ctx->conv_dim * sizeof(float));
        lg->upd[l] = malloc(2 * (size_t)cap * upd_row(ctx) * sizeof(float));
        ok = lg->xin[l] && lg->upd[l];
    }
    if (!ok) { pf_lin_log_free(ctx); return PINFER_E_NOMEM; }
    return PINFER_OK;
}

/* Advance the committed states by the first `keep` tokens of the logged pass. Token 0 is
 * in them already; the conv state is updated now, and each recurrent state records the
 * keep - 1 updates it still owes (rows 1..keep-1 of the half just written), which its next
 * pass or pf_lin_flush applies. The replay performs exactly the per-element operations of
 * the recurrence (S = S*decay, then S = S + k_i*delta_j, both rounded to fp32), so the
 * result is bit-identical to running those tokens directly. */
void pf_lin_commit(pinfer_ctx *ctx, int keep) {
    pf_lin_log *lg = ctx->lin_log;
    const int C = ctx->conv_dim, S = ctx->conv_k - 1, e = keep - 1;
    if (!lg || keep <= 0 || keep > lg->n) return;
    for (int l = 0; l < ctx->n_layers; l++) {
        if (ctx->model.layers[l].is_full) continue;
        /* conv: new state = last S rows of [state after token 0, raw inputs 1..e]; ascending
         * j reads row j + e > j before anything at or after it is overwritten */
        float *st = ctx->conv[l];
        for (int j = 0; e > 0 && j < S; j++) {
            int src = j + e;
            const float *from = src < S ? st + (size_t)src * C
                                        : lg->xin[l] + (size_t)(src - S + 1) * C;
            memcpy(st + (size_t)j * C, from, (size_t)C * sizeof(float));
        }
        lg->keep[l] = e;
    }
    lg->wr ^= 1;   /* the half just written becomes the pending one */
    lg->n = 0;
}

/* Apply every deferred update now (one read + write of each recurrent state). */
void pf_lin_flush(pinfer_ctx *ctx) {
    pf_lin_log *lg = ctx->lin_log;
    if (!lg) return;
    for (int l = 0; l < ctx->n_layers; l++) {
        if (ctx->model.layers[l].is_full || lg->keep[l] <= 0) continue;
        pf_lin_apply(ctx->pool, ctx->rec[l], upd_half(ctx, l, lg->wr ^ 1) + upd_row(ctx),
                     lg->keep[l], ctx->lv_heads, ctx->lk_dim, ctx->lv_dim);
        lg->keep[l] = 0;
    }
}
