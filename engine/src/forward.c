/* Per-token forward pass of Qwen3.5 (hybrid Gated DeltaNet + gated full attention).
 *
 * Semantics follow engine/FORMAT.md "Forward pass" (= transformers 5.20.2 modeling_qwen3_5).
 * Tokens are processed in chunks of ctx->batch: every weight matrix is applied to the whole
 * chunk with one pf_matmul (weights streamed once per chunk), while the inherently sequential
 * parts (conv state, recurrence, KV-cache attention) loop over the chunk's tokens in order.
 * All activations and states are fp32. The Gated DeltaNet mixer lives in forward_lin.c. */
#include <math.h>
#include <string.h>

#include "forward.h"

float pf_silu(float x) { return x / (1.0f + expf(-x)); }
float pf_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* y = x / rms(x) * w. `w` is the stored effective multiplier (1 + w for zero-centred norms). */
void pf_rmsnorm(float *y, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
}

/* Rotate-half RoPE on the first `rot` dims of one head: pairs (i, i + rot/2). */
static void rope(float *v, const float *c, const float *s, int rot) {
    int h = rot / 2;
    for (int i = 0; i < h; i++) {
        float a = v[i], b = v[i + h];
        v[i] = a * c[i] - b * s[i];
        v[i + h] = b * c[i] + a * s[i];
    }
}

/* cos/sin tables for positions pos0..pos0+n-1. Computed in fp32 exactly like the reference
 * (inv_freq = 1 / theta^(2i/rot) in float, angle = float(pos) * inv_freq) so long-context
 * angles round the same way. */
static void rope_tables(pinfer_ctx *ctx, int pos0, int n) {
    int h = ctx->rot / 2;
    float theta = ctx->model.hdr->rope_theta;
    for (int i = 0; i < h; i++) {
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)ctx->rot);
        for (int t = 0; t < n; t++) {
            float ang = (float)(pos0 + t) * inv;
            ctx->rope_cos[t * h + i] = cosf(ang);
            ctx->rope_sin[t * h + i] = sinf(ang);
        }
    }
}

/* Gated full attention for n tokens at positions pos0.. : ctx->xn -> ctx->out. */
static void full_attn(pinfer_ctx *ctx, int layer, int n, int pos0) {
    const pf_layer *L = &ctx->model.layers[layer];
    const int hd = ctx->hd, nq = ctx->n_heads, kvw = ctx->n_kv * hd, grp = nq / ctx->n_kv;
    const int qw = nq * hd * 2, h2 = ctx->rot / 2;
    const float eps = ctx->model.hdr->rms_eps, scale = 1.0f / sqrtf((float)hd);
    float *kc = ctx->kc[layer], *vc = ctx->vc[layer], *att = ctx->att;

    pf_matmul(ctx->pool, &L->q, ctx->xn, ctx->ta, n);   /* per head: [query hd | gate hd] */
    pf_matmul(ctx->pool, &L->k, ctx->xn, ctx->tb, n);
    pf_matmul(ctx->pool, &L->v, ctx->xn, ctx->tc, n);

    for (int t = 0; t < n; t++) {
        const int pos = pos0 + t;
        const float *cs = ctx->rope_cos + t * h2, *sn = ctx->rope_sin + t * h2;
        float *qrow = ctx->ta + (size_t)t * qw, *o = ctx->to + (size_t)t * nq * hd;
        float *krow = kc + (size_t)pos * kvw, *vrow = vc + (size_t)pos * kvw;

        /* This token's K/V go into the cache first: causal attention includes itself. */
        for (int g = 0; g < ctx->n_kv; g++) {
            pf_rmsnorm(krow + g * hd, ctx->tb + (size_t)t * kvw + g * hd, L->k_norm, hd, eps);
            rope(krow + g * hd, cs, sn, ctx->rot);
        }
        memcpy(vrow, ctx->tc + (size_t)t * kvw, (size_t)kvw * sizeof(float));

        for (int h = 0; h < nq; h++) {
            float *q = qrow + h * 2 * hd;          /* normalised in place */
            const float *gate = q + hd;
            const int g = h / grp;                 /* GQA: 4 query heads share a KV head */
            pf_rmsnorm(q, q, L->q_norm, hd, eps);
            rope(q, cs, sn, ctx->rot);

            float mx = -INFINITY;
            for (int j = 0; j <= pos; j++) {
                const float *kj = kc + (size_t)j * kvw + g * hd;
                float s = 0.0f;
                for (int d = 0; d < hd; d++) s += q[d] * kj[d];
                att[j] = s * scale;
                if (att[j] > mx) mx = att[j];
            }
            float sum = 0.0f;
            for (int j = 0; j <= pos; j++) { att[j] = expf(att[j] - mx); sum += att[j]; }
            float inv = 1.0f / sum;

            float *oh = o + h * hd;
            memset(oh, 0, (size_t)hd * sizeof(float));
            for (int j = 0; j <= pos; j++) {
                const float *vj = vc + (size_t)j * kvw + g * hd;
                float p = att[j] * inv;
                for (int d = 0; d < hd; d++) oh[d] += p * vj[d];
            }
            for (int d = 0; d < hd; d++) oh[d] *= pf_sigmoid(gate[d]);  /* output gate */
        }
    }
    pf_matmul(ctx->pool, &L->o, ctx->to, ctx->out, n);
}

/* SwiGLU: down(silu(gate x) * up x), ctx->xn -> ctx->out. */
static void mlp(pinfer_ctx *ctx, int layer, int n) {
    const pf_layer *L = &ctx->model.layers[layer];
    pf_matmul(ctx->pool, &L->gate, ctx->xn, ctx->ta, n);
    pf_matmul(ctx->pool, &L->up, ctx->xn, ctx->tb, n);
    size_t m = (size_t)n * ctx->ffn;
    for (size_t i = 0; i < m; i++) ctx->ta[i] = pf_silu(ctx->ta[i]) * ctx->tb[i];
    pf_matmul(ctx->pool, &L->down, ctx->ta, ctx->out, n);
}

static void norm_rows(pinfer_ctx *ctx, float *dst, const float *w, int n) {
    for (int t = 0; t < n; t++)
        pf_rmsnorm(dst + (size_t)t * ctx->H, ctx->x + (size_t)t * ctx->H, w, ctx->H,
                   ctx->model.hdr->rms_eps);
}

static void add_rows(pinfer_ctx *ctx, int n) {
    size_t m = (size_t)n * ctx->H;
    for (size_t i = 0; i < m; i++) ctx->x[i] += ctx->out[i];
}

/* hidden dump: entry `e` of [n_layers+1][n_total][H], rows t0..t0+n-1 */
static void dump(const pinfer_ctx *ctx, float *hidden, int e, int n_total, int t0,
                 const float *src, int n) {
    if (!hidden) return;
    size_t H = (size_t)ctx->H;
    memcpy(hidden + ((size_t)e * n_total + t0) * H, src, (size_t)n * H * sizeof(float));
}

/* One chunk of n <= ctx->batch tokens. t0 = index of the chunk inside the pf_forward call. */
static void forward_chunk(pinfer_ctx *ctx, const int32_t *tok, int n, int t0, int n_total,
                          float *logits_last, float *logits_all, float *hidden) {
    const pf_model *m = &ctx->model;
    const int H = ctx->H, pos0 = ctx->n_past;

    for (int t = 0; t < n; t++) pf_get_row(&m->embed, tok[t], ctx->x + (size_t)t * H);
    dump(ctx, hidden, 0, n_total, t0, ctx->x, n);
    rope_tables(ctx, pos0, n);

    for (int l = 0; l < ctx->n_layers; l++) {
        const pf_layer *L = &m->layers[l];
        norm_rows(ctx, ctx->xn, L->in_norm, n);
        if (L->is_full) full_attn(ctx, l, n, pos0);
        else pf_linear_attn(ctx, l, n);
        add_rows(ctx, n);
        norm_rows(ctx, ctx->xn, L->post_norm, n);
        mlp(ctx, l, n);
        add_rows(ctx, n);
        if (l + 1 < ctx->n_layers) dump(ctx, hidden, l + 1, n_total, t0, ctx->x, n);
    }

    /* Final norm + tied head. Only the rows whose logits are wanted pay for the 248k-row
     * head matmul (it is ~1/3 of all weight bytes). */
    norm_rows(ctx, ctx->xn, m->norm, n);
    dump(ctx, hidden, ctx->n_layers, n_total, t0, ctx->xn, n);
    if (logits_all)
        pf_matmul(ctx->pool, &m->embed, ctx->xn, logits_all + (size_t)t0 * ctx->vocab, n);
    if (logits_last && t0 + n == n_total)
        pf_matvec(ctx->pool, &m->embed, ctx->xn + (size_t)(n - 1) * H, logits_last);
    ctx->n_past += n;
}

int pf_forward(pinfer_ctx *ctx, const int32_t *tokens, int n, float *logits_last,
               float *logits_all, float *hidden) {
    if (!ctx || !tokens || n <= 0) return PINFER_E_ARG;
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= ctx->vocab) return PINFER_E_ARG;
    if (n > ctx->max_ctx - ctx->n_past) return PINFER_E_CTX_FULL;
    for (int t0 = 0; t0 < n; t0 += ctx->batch) {
        int nb = n - t0 < ctx->batch ? n - t0 : ctx->batch;
        forward_chunk(ctx, tokens + t0, nb, t0, n, logits_last, logits_all, hidden);
    }
    return PINFER_OK;
}

void pf_set_batch(pinfer_ctx *ctx, int batch) {
    ctx->batch = batch < 1 ? 1 : batch > PF_MAX_BATCH ? PF_MAX_BATCH : batch;
}
