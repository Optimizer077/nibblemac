/* Per-token forward pass of Qwen3.5 (hybrid Gated DeltaNet + gated full attention).
 *
 * Semantics follow engine/FORMAT.md "Forward pass" (= transformers 5.20.2 modeling_qwen3_5).
 * Tokens are processed in chunks of ctx->batch: every weight matrix is applied to the whole
 * chunk with one pf_matmul (weights streamed once per chunk), while the inherently sequential
 * parts (conv state, recurrence, KV-cache attention) loop over the chunk's tokens in order.
 * All activations and states are fp32. The Gated DeltaNet mixer lives in forward_lin.c, the
 * gated full attention in forward_attn.c. */
#include <math.h>
#include <string.h>

#include "forward.h"

/* cos/sin tables for positions pos0..pos0+n-1. Computed in fp32 exactly like the reference
 * (inv_freq = 1 / theta^(2i/rot) in float, angle = float(pos) * inv_freq) so long-context
 * angles round the same way. */
void pf_rope_tables(pinfer_ctx *ctx, int pos0, int n) {
    const double ts = pf_stats_now();
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
    pf_stats_add(PFS_ROPE, PFS_KIND(ctx, n), ts);
}

typedef struct {
    float *a;
    const float *b;
    size_t n;
} act_job;

/* a = silu(a) * b over a contiguous slice per thread (8-aligned for the vector loop). */
static void act_task(void *arg, int tid, int nthr) {
    const act_job *j = arg;
    const size_t i0 = j->n * (size_t)tid / (size_t)nthr / 8 * 8;
    const size_t i1 = tid + 1 == nthr ? j->n : j->n * (size_t)(tid + 1) / (size_t)nthr / 8 * 8;
    if (i1 > i0) pf_silu_mul(j->a + i0, j->b + i0, (int)(i1 - i0));
}

/* SwiGLU: down(silu(gate x) * up x), ctx->xn -> ctx->out. */
void pf_mlp(pinfer_ctx *ctx, const pf_layer *L, int n) {
    pf_matmul(ctx->pool, &L->gate, ctx->xn, ctx->ta, n);
    pf_matmul(ctx->pool, &L->up, ctx->xn, ctx->tb, n);
    const double ts = pf_stats_now();
    act_job job = {ctx->ta, ctx->tb, (size_t)n * ctx->ffn};
    pf_pool_run(ctx->pool, act_task, &job, 0);
    pf_stats_add(PFS_MLP_ACT, PFS_KIND(ctx, n), ts);
    pf_matmul(ctx->pool, &L->down, ctx->ta, ctx->out, n);
}

void pf_norm_rows(pinfer_ctx *ctx, float *dst, const float *w, int n) {
    const double ts = pf_stats_now();
    for (int t = 0; t < n; t++)
        pf_rmsnorm(dst + (size_t)t * ctx->H, ctx->x + (size_t)t * ctx->H, w, ctx->H,
                   ctx->model.hdr->rms_eps);
    pf_stats_add(PFS_NORM, PFS_KIND(ctx, n), ts);
}

void pf_add_rows(pinfer_ctx *ctx, int n) {
    const double ts = pf_stats_now();
    pf_add(ctx->x, ctx->out, n * ctx->H);
    pf_stats_add(PFS_NORM, PFS_KIND(ctx, n), ts);
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
                          float *logits_last, float *logits_all, float *hidden,
                          float *normed) {
    const pf_model *m = &ctx->model;
    const int H = ctx->H, pos0 = ctx->n_past;

    const double ts = pf_stats_now();
    for (int t = 0; t < n; t++) pf_get_row(&m->embed, tok[t], ctx->x + (size_t)t * H);
    pf_stats_add(PFS_EMBED, PFS_KIND(ctx, n), ts);
    dump(ctx, hidden, 0, n_total, t0, ctx->x, n);
    pf_rope_tables(ctx, pos0, n);

    for (int l = 0; l < ctx->n_layers; l++) {
        const pf_layer *L = &m->layers[l];
        pf_norm_rows(ctx, ctx->xn, L->in_norm, n);
        if (L->is_full) pf_full_attn(ctx, L, ctx->kc[l], ctx->vc[l], n, pos0);
        else pf_linear_attn(ctx, l, n);
        pf_add_rows(ctx, n);
        pf_norm_rows(ctx, ctx->xn, L->post_norm, n);
        pf_mlp(ctx, L, n);
        pf_add_rows(ctx, n);
        if (l + 1 < ctx->n_layers) dump(ctx, hidden, l + 1, n_total, t0, ctx->x, n);
    }

    /* Final norm + tied head. Only the rows whose logits are wanted pay for the 248k-row
     * head matmul (it is ~1/3 of all weight bytes). */
    pf_norm_rows(ctx, ctx->xn, m->norm, n);
    dump(ctx, hidden, ctx->n_layers, n_total, t0, ctx->xn, n);
    if (normed) memcpy(normed + (size_t)t0 * H, ctx->xn, (size_t)n * H * sizeof(float));
    if (logits_all)
        pf_matmul(ctx->pool, &m->embed, ctx->xn, logits_all + (size_t)t0 * ctx->vocab, n);
    if (logits_last && t0 + n == n_total)
        pf_matvec(ctx->pool, &m->embed, ctx->xn + (size_t)(n - 1) * H, logits_last);
    ctx->n_past += n;
}

int pf_forward(pinfer_ctx *ctx, const int32_t *tokens, int n, float *logits_last,
               float *logits_all, float *hidden) {
    return pf_forward_h(ctx, tokens, n, logits_last, logits_all, hidden, NULL);
}

int pf_forward_h(pinfer_ctx *ctx, const int32_t *tokens, int n, float *logits_last,
                 float *logits_all, float *hidden, float *normed) {
    if (!ctx || !tokens || n <= 0) return PINFER_E_ARG;
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= ctx->vocab) return PINFER_E_ARG;
    if (n > ctx->max_ctx - ctx->n_past) return PINFER_E_CTX_FULL;
    for (int t0 = 0; t0 < n; t0 += ctx->batch) {
        int nb = n - t0 < ctx->batch ? n - t0 : ctx->batch;
        forward_chunk(ctx, tokens + t0, nb, t0, n, logits_last, logits_all, hidden, normed);
    }
    return PINFER_OK;
}

void pf_set_batch(pinfer_ctx *ctx, int batch) {
    ctx->batch = batch < 1 ? 1 : batch > PF_MAX_BATCH ? PF_MAX_BATCH : batch;
}
