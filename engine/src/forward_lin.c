/* Gated DeltaNet (linear attention) mixer of Qwen3.5, one token at a time.
 *
 * Reference: transformers 5.20.2 Qwen3_5GatedDeltaNet with torch_recurrent_gated_delta_rule
 * (the decode path). The chunked prefill kernel in transformers is the same recurrence
 * re-associated, so running the recurrent form for every token is exact, just O(T) serial. */
#include <math.h>
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
 *   S *= exp(g);  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
 * Two passes over S: (decay, accumulate kv) then (rank-1 update, accumulate o). */
static void delta_rule(float *S, const float *q, const float *k, const float *v,
                       float g, float beta, float *o, float *tmp, int dk, int dv) {
    const float decay = expf(g);
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
    float *rec = ctx->rec[layer];
    float tmp[512];   /* delta scratch; lin_v_dim is 128 for every released Qwen3.5 */

    pf_matmul(ctx->pool, &L->qkv, ctx->xn, ctx->ta, n);
    pf_matmul(ctx->pool, &L->z, ctx->xn, ctx->tb, n);
    pf_matmul(ctx->pool, &L->b, ctx->xn, ctx->ba, n);
    pf_matmul(ctx->pool, &L->a, ctx->xn, ctx->aa, n);

    for (int t = 0; t < n; t++) {
        float *qkv = ctx->ta + (size_t)t * C;
        const float *z = ctx->tb + (size_t)t * vd;
        float *o = ctx->to + (size_t)t * vd;
        conv_step(qkv, ctx->conv[layer], L->conv_w, C, ctx->conv_k);

        float *q = qkv, *k = qkv + kd;
        const float *v = qkv + 2 * kd;
        for (int h = 0; h < ctx->lk_heads; h++) {
            l2norm(q + h * dk, dk);
            l2norm(k + h * dk, dk);
            for (int i = 0; i < dk; i++) q[h * dk + i] *= qscale;
        }
        for (int h = 0; h < nv; h++) {
            const int kh = h / rep;  /* repeat_interleave of q/k heads onto v heads */
            const float beta = pf_sigmoid(ctx->ba[(size_t)t * nv + h]);
            const float g = -expf(L->A_log[h]) *
                            softplus(ctx->aa[(size_t)t * nv + h] + L->dt_bias[h]);
            float *oh = o + h * dv;
            delta_rule(rec + (size_t)h * dk * dv, q + kh * dk, k + kh * dk, v + h * dv,
                       g, beta, oh, tmp, dk, dv);
            /* gated RMSNorm: rmsnorm(o) * w * silu(z), per head, plain weight */
            pf_rmsnorm(oh, oh, L->lin_norm, dv, eps);
            for (int j = 0; j < dv; j++) oh[j] *= pf_silu(z[h * dv + j]);
        }
    }
    pf_matmul(ctx->pool, &L->out, ctx->to, ctx->out, n);
}
