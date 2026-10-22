/* MTP draft head: loading (from the mtp.* tensors of the .pq file) and forward pass.
 * See mtp.h for the semantics and engine/SPEC.md for the sources they were taken from. */
#include "mtp.h"

#include <stdlib.h>
#include <string.h>

static int map_tensors(const pinfer_ctx *ctx, pf_mtp *t) {
    const pf_model *m = &ctx->model;
    const uint32_t H = (uint32_t)ctx->H, hd = (uint32_t)ctx->hd, ffn = (uint32_t)ctx->ffn;
    const uint32_t nq = (uint32_t)ctx->n_heads, nkv = (uint32_t)ctx->n_kv;
    pf_layer *L = &t->layer;
    int rc = 0;
    if (!m->hdr->has_mtp) return PINFER_E_FORMAT;
    /* model.c's typed views: every entry's bounds and size (quantized dtypes included)
     * were checked at load, so a malformed mtp.* tensor fails here, never in a kernel */
    L->is_full = 1;
    rc |= pf_model_mat(m, "mtp.fc.weight", H, 2 * H, &t->fc);
    rc |= pf_model_vec(m, "mtp.pre_fc_norm_embedding.weight", H, &t->norm_emb);
    rc |= pf_model_vec(m, "mtp.pre_fc_norm_hidden.weight", H, &t->norm_hid);
    rc |= pf_model_vec(m, "mtp.norm.weight", H, &t->norm_out);
    rc |= pf_model_vec(m, "mtp.layers.0.input_layernorm.weight", H, &L->in_norm);
    rc |= pf_model_vec(m, "mtp.layers.0.post_attention_layernorm.weight", H, &L->post_norm);
    rc |= pf_model_mat(m, "mtp.layers.0.self_attn.q_proj.weight", nq * hd * 2, H, &L->q);
    rc |= pf_model_mat(m, "mtp.layers.0.self_attn.k_proj.weight", nkv * hd, H, &L->k);
    rc |= pf_model_mat(m, "mtp.layers.0.self_attn.v_proj.weight", nkv * hd, H, &L->v);
    rc |= pf_model_mat(m, "mtp.layers.0.self_attn.o_proj.weight", H, nq * hd, &L->o);
    rc |= pf_model_vec(m, "mtp.layers.0.self_attn.q_norm.weight", hd, &L->q_norm);
    rc |= pf_model_vec(m, "mtp.layers.0.self_attn.k_norm.weight", hd, &L->k_norm);
    rc |= pf_model_mat(m, "mtp.layers.0.mlp.gate_proj.weight", ffn, H, &L->gate);
    rc |= pf_model_mat(m, "mtp.layers.0.mlp.up_proj.weight", ffn, H, &L->up);
    rc |= pf_model_mat(m, "mtp.layers.0.mlp.down_proj.weight", H, ffn, &L->down);
    return rc ? PINFER_E_FORMAT : PINFER_OK;
}

void pf_mtp_free(pinfer_ctx *ctx) {
    if (!ctx->mtp) return;
    free(ctx->mtp->kc);
    free(ctx->mtp->vc);
    free(ctx->mtp->e);
    free(ctx->mtp);
    ctx->mtp = NULL;
}

int pf_mtp_init(pinfer_ctx *ctx) {
    if (ctx->mtp) return PINFER_OK;
    pf_mtp *t = calloc(1, sizeof *t);
    if (!t) return PINFER_E_NOMEM;
    int rc = map_tensors(ctx, t);
    if (rc != PINFER_OK) { free(t); return rc; }
    const size_t kv = (size_t)ctx->max_ctx * ctx->n_kv * ctx->hd;
    t->kc = malloc(kv * sizeof(float));
    t->vc = malloc(kv * sizeof(float));
    t->e = malloc((size_t)ctx->H * sizeof(float));
    ctx->mtp = t;
    if (!t->kc || !t->vc || !t->e) { pf_mtp_free(ctx); return PINFER_E_NOMEM; }
    return PINFER_OK;
}

int pf_mtp_forward(pinfer_ctx *ctx, const float *h, const int32_t *tok, int n, int pos0,
                   float *out, float *logits_last, float *logits_all, int n_voc) {
    pf_mtp *t = ctx->mtp;
    const int H = ctx->H;
    const float eps = ctx->model.hdr->rms_eps;
    if (!t || n < 1 || n > PF_MAX_BATCH || pos0 < 0 || pos0 + n > ctx->max_ctx)
        return PINFER_E_ARG;
    for (int i = 0; i < n; i++)
        if (tok[i] < 0 || tok[i] >= ctx->vocab) return PINFER_E_ARG;

    /* fc input rows [embedding half | hidden half] (vLLM: cat([inputs_embeds, hidden])) */
    float *cat = ctx->ta;
    for (int i = 0; i < n; i++) {
        float *row = cat + (size_t)i * 2 * H;
        pf_get_row(&ctx->model.embed, tok[i], t->e);
        pf_rmsnorm(row, t->e, t->norm_emb, H, eps);
        pf_rmsnorm(row + H, h + (size_t)i * H, t->norm_hid, H, eps);
    }
    pf_matmul(ctx->pool, &t->fc, cat, ctx->x, n);

    /* one full-attention decoder block, exactly as a main-model full-attention layer */
    const pf_layer *L = &t->layer;
    pf_rope_tables(ctx, pos0, n);
    pf_norm_rows(ctx, ctx->xn, L->in_norm, n);
    pf_full_attn(ctx, L, t->kc, t->vc, n, pos0);
    pf_add_rows(ctx, n);
    pf_norm_rows(ctx, ctx->xn, L->post_norm, n);
    pf_mlp(ctx, L, n);
    pf_add_rows(ctx, n);
    pf_norm_rows(ctx, ctx->xn, t->norm_out, n);
    if (out) memcpy(out, ctx->xn, (size_t)n * H * sizeof(float));

    /* The model's head (tied: the embedding), optionally restricted to the first n_voc token
     * ids: a prefix of rows is a valid matrix for BF16 and for the 4-row-tiled quantized
     * layouts (n_voc % 4 == 0). */
    pf_mat head = ctx->model.head;
    if (n_voc > 0 && n_voc < head.rows) head.rows = n_voc & ~3;
    if (logits_all) pf_matmul(ctx->pool, &head, ctx->xn, logits_all, n);
    if (logits_last) pf_matvec(ctx->pool, &head, ctx->xn + (size_t)(n - 1) * H, logits_last);
    return PINFER_OK;
}
