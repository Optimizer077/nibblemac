/* Internal forward-pass interface: the context layout shared by pinfer.c (API, allocation)
 * and forward.c / forward_lin.c (math). Not part of the public API. */
#ifndef PF_FORWARD_H
#define PF_FORWARD_H

#include <stdint.h>

#include "model.h"
#include "ops.h"
#include "pinfer.h"

/* Tokens processed together per pf_matmul call during prefill. Larger amortises the weight
 * stream over more tokens; 64 keeps the scratch buffers at ~4 MB. */
#define PF_MAX_BATCH 64

struct pinfer_ctx {
    pf_model model;
    pf_pool *pool;
    int max_ctx;
    int n_past;
    int batch;          /* current prefill chunk size, 1..PF_MAX_BATCH (pf_set_batch) */

    /* derived dimensions */
    int H, ffn, n_layers, vocab;
    int n_heads, n_kv, hd, rot;            /* full attention */
    int lk_heads, lv_heads, lk_dim, lv_dim, conv_k, kd, vd, conv_dim;  /* linear attention */

    /* per-layer state; NULL where the layer type does not use it */
    float **conv;       /* [conv_k-1][conv_dim]: previous inputs, oldest first */
    float **rec;        /* [lv_heads][lk_dim][lv_dim]: Gated DeltaNet state S */
    float **kc, **vc;   /* [max_ctx][n_kv*hd]: KV cache (post-norm, post-RoPE keys) */

    /* scratch, PF_MAX_BATCH rows each */
    float *x, *xn, *out;        /* [B][H]: residual stream, normed input, sublayer output */
    float *ta, *tb, *tc, *to;   /* projections: qkv|q_proj|gate, z|k|up, v, mix output */
    float *ba, *aa;             /* [B][lv_heads]: in_proj_b / in_proj_a outputs */
    float *rope_cos, *rope_sin; /* [B][rot/2] */
    float *att;                 /* [max_ctx]: attention scores of one head */
};

/* Run n tokens through the model starting at position ctx->n_past and advance n_past.
 *   logits_last [vocab]   (may be NULL): logits of the last token
 *   logits_all  [n][vocab](may be NULL): logits of every token
 *   hidden [n_layers+1][n][H] (may be NULL): embeddings, then each layer's output; the last
 *       entry is the final-normed state (same convention as ref/golden hidden_states.npy).
 * Returns PINFER_OK or PINFER_E_ARG / PINFER_E_CTX_FULL. */
int pf_forward(pinfer_ctx *ctx, const int32_t *tokens, int n, float *logits_last,
               float *logits_all, float *hidden);

/* Prefill chunk size (clamped to 1..PF_MAX_BATCH). 1 = pure token-by-token decode path. */
void pf_set_batch(pinfer_ctx *ctx, int batch);

/* Gated DeltaNet mixer for n tokens (forward_lin.c): ctx->xn -> ctx->out. */
void pf_linear_attn(pinfer_ctx *ctx, int layer, int n);

/* Shared elementwise helpers (forward.c). */
void pf_rmsnorm(float *y, const float *x, const float *w, int n, float eps);
float pf_silu(float x);
float pf_sigmoid(float x);

#endif
