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

/* torch.nn.functional.softplus (beta 1, threshold 20). */
static float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

/* x * rsqrt(sum(x^2) + eps), as the FLA-compatible l2norm in the reference. */
static void l2norm(float *x, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss + 1e-6f);
    for (int i = 0; i < n; i++) x[i] *= r;
}

/* Causal depthwise conv (kernel conv_k) + SiLU, in place on one token's qkv vector.
 * state holds the previous conv_k-1 raw inputs, oldest first; it is shifted per channel. */
static void conv_step(float *qkv, float *state, const float *w, int C, int K) {
    const int S = K - 1;
    for (int c = 0; c < C; c++) {
        const float *wc = w + (size_t)c * K;
        float x = qkv[c], acc = 0.0f;
        for (int j = 0; j < S; j++) acc += wc[j] * state[(size_t)j * C + c];
        acc += wc[S] * x;
        for (int j = 0; j + 1 < S; j++) state[(size_t)j * C + c] = state[(size_t)(j + 1) * C + c];
        if (S > 0) state[(size_t)(S - 1) * C + c] = x;
        qkv[c] = pf_silu(acc);
    }
}

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

void pf_linear_attn(pinfer_ctx *ctx, int layer, int n) {
    const pf_layer *L = &ctx->model.layers[layer];
    const int C = ctx->conv_dim, kd = ctx->kd, vd = ctx->vd, nv = ctx->lv_heads;
    const int dk = ctx->lk_dim, dv = ctx->lv_dim, rep = nv / ctx->lk_heads;
    const float eps = ctx->model.hdr->rms_eps, qscale = 1.0f / sqrtf((float)dk);
    float *rec = ctx->rec[layer], *conv = ctx->conv[layer];
    pf_lin_log *lg = ctx->lin_log_on ? ctx->lin_log : NULL;
    float tmp[512];   /* delta scratch; lin_v_dim is 128 for every released Qwen3.5 */

    pf_matmul(ctx->pool, &L->qkv, ctx->xn, ctx->ta, n);
    pf_matmul(ctx->pool, &L->z, ctx->xn, ctx->tb, n);
    pf_matmul(ctx->pool, &L->b, ctx->xn, ctx->ba, n);
    pf_matmul(ctx->pool, &L->a, ctx->xn, ctx->aa, n);

    if (lg) {  /* verification pass: work on a copy, keep the raw inputs for pf_lin_commit */
        memcpy(lg->conv, conv, (size_t)(ctx->conv_k - 1) * C * sizeof(float));
        conv = lg->conv;
        memcpy(lg->xin[layer], ctx->ta, (size_t)n * C * sizeof(float));
        lg->n = n;
    }

    /* Per token: conv + SiLU, q/k L2 norm, q scale; beta and decay overwrite ba / aa. */
    for (int t = 0; t < n; t++) {
        float *qkv = ctx->ta + (size_t)t * C;
        conv_step(qkv, conv, L->conv_w, C, ctx->conv_k);
        for (int h = 0; h < ctx->lk_heads; h++) {
            l2norm(qkv + h * dk, dk);
            l2norm(qkv + kd + h * dk, dk);
            for (int i = 0; i < dk; i++) qkv[h * dk + i] *= qscale;
        }
        for (int h = 0; h < nv; h++) {
            float *b = ctx->ba + (size_t)t * nv + h, *a = ctx->aa + (size_t)t * nv + h;
            *b = pf_sigmoid(*b);
            *a = expf(-expf(L->A_log[h]) * softplus(*a + L->dt_bias[h]));  /* exp(g) */
        }
    }

    /* Per head over all tokens; the repeat_interleave of q/k heads onto v heads is kh. */
    for (int h = 0; h < nv; h++) {
        const int kh = h / rep;
        float *S = rec + (size_t)h * dk * dv;
        if (lg) {
            memcpy(lg->S, S, (size_t)dk * dv * sizeof(float));
            S = lg->S;
        }
        for (int t = 0; t < n; t++) {
            const float *qkv = ctx->ta + (size_t)t * C;
            const float *k = qkv + kd + kh * dk, *z = ctx->tb + (size_t)t * vd + h * dv;
            const float decay = ctx->aa[(size_t)t * nv + h];
            float *oh = ctx->to + (size_t)t * vd + h * dv;
            delta_rule(S, qkv + kh * dk, k, qkv + 2 * kd + h * dv, decay,
                       ctx->ba[(size_t)t * nv + h], oh, tmp, dk, dv);
            if (lg) {
                float *u = lg->upd[layer] + ((size_t)t * nv + h) * (dk + dv + 1);
                memcpy(u, k, (size_t)dk * sizeof(float));
                memcpy(u + dk, tmp, (size_t)dv * sizeof(float));
                u[dk + dv] = decay;
            }
            /* gated RMSNorm: rmsnorm(o) * w * silu(z), per head, plain weight */
            pf_rmsnorm(oh, oh, L->lin_norm, dv, eps);
            for (int j = 0; j < dv; j++) oh[j] *= pf_silu(z[j]);
        }
    }
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
        for (int h = 0; h < nv; h++) {
            float *Sh = ctx->rec[l] + (size_t)h * dk * dv;
            for (int t = 0; t < keep; t++) {
                const float *u = lg->upd[l] + ((size_t)t * nv + h) * (dk + dv + 1);
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
}
