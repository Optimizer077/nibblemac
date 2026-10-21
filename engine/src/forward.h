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

/* Rollback log for speculative verification (forward_lin.c). While ctx->lin_log_on is set,
 * pf_linear_attn runs the conv and recurrent states on scratch copies (the committed states
 * stay untouched) and records, per token, what pf_lin_commit needs to advance the committed
 * states by any prefix of the pass: the raw conv inputs and each head's rank-1 update. */
typedef struct {
    int cap;          /* max tokens per logged pass */
    int n;            /* tokens in the last logged pass */
    float *S;         /* [lk_dim * lv_dim]: working copy of one head's state */
    float *conv;      /* [(conv_k-1) * conv_dim]: working copy of one layer's conv state */
    float **xin;      /* per linear layer [cap][conv_dim]: raw conv inputs */
    float **upd;      /* per linear layer [cap][lv_heads][lk_dim + lv_dim + 1]: k | delta | decay */
} pf_lin_log;

struct pf_mtp;        /* MTP draft head (mtp.h) */
struct pf_spec_buf;   /* speculative-decoding buffers (spec.h) */

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
    float **kc, **vc;   /* [n_kv][max_ctx][hd]: KV cache (post-norm, post-RoPE keys); one
                           contiguous stream per KV group, so a head's pass over the cache
                           is sequential for the prefetcher (forward.c pf_full_attn) */

    /* scratch, PF_MAX_BATCH rows each */
    float *x, *xn, *out;        /* [B][H]: residual stream, normed input, sublayer output */
    float *ta, *tb, *tc, *to;   /* projections: qkv|q_proj|gate, z|k|up, v, mix output */
    float *ba, *aa;             /* [B][lv_heads]: in_proj_b / in_proj_a outputs */
    float *rope_cos, *rope_sin; /* [B][rot/2] */
    float *att;                 /* [n_heads][max_ctx]: attention scores per head */

    /* speculative decoding (spec.c); all NULL / 0 until first used */
    pf_lin_log *lin_log;
    int lin_log_on;
    struct pf_mtp *mtp;
    struct pf_spec_buf *spec;
};

/* Run n tokens through the model starting at position ctx->n_past and advance n_past.
 *   logits_last [vocab]   (may be NULL): logits of the last token
 *   logits_all  [n][vocab](may be NULL): logits of every token
 *   hidden [n_layers+1][n][H] (may be NULL): embeddings, then each layer's output; the last
 *       entry is the final-normed state (same convention as ref/golden hidden_states.npy).
 * Returns PINFER_OK or PINFER_E_ARG / PINFER_E_CTX_FULL. */
int pf_forward(pinfer_ctx *ctx, const int32_t *tokens, int n, float *logits_last,
               float *logits_all, float *hidden);

/* pf_forward plus `normed` [n][H] (may be NULL): every token's final-normed hidden state,
 * the input of the MTP head. */
int pf_forward_h(pinfer_ctx *ctx, const int32_t *tokens, int n, float *logits_last,
                 float *logits_all, float *hidden, float *normed);

/* Prefill chunk size (clamped to 1..PF_MAX_BATCH). 1 = pure token-by-token decode path. */
void pf_set_batch(pinfer_ctx *ctx, int batch);

/* Gated DeltaNet mixer for n tokens (forward_lin.c): ctx->xn -> ctx->out. */
void pf_linear_attn(pinfer_ctx *ctx, int layer, int n);

/* Rollback log (forward_lin.c). alloc: room for passes of up to `cap` tokens. commit:
 * advance the committed conv/recurrent states by the first `keep` tokens of the last logged
 * pass, bit-identical to having run only those tokens. */
int pf_lin_log_alloc(pinfer_ctx *ctx, int cap);
void pf_lin_log_free(pinfer_ctx *ctx);
void pf_lin_commit(pinfer_ctx *ctx, int keep);

/* Building blocks shared with the MTP head (forward.c). All work on rows 0..n-1 of the
 * ctx scratch buffers: x = residual stream, xn = normed input, out = sublayer output. */
void pf_rope_tables(pinfer_ctx *ctx, int pos0, int n);        /* positions pos0.. */
void pf_norm_rows(pinfer_ctx *ctx, float *dst, const float *w, int n);   /* dst = norm(x) */
void pf_add_rows(pinfer_ctx *ctx, int n);                     /* x += out */
void pf_full_attn(pinfer_ctx *ctx, const pf_layer *L, float *kc, float *vc, int n, int pos0);
void pf_mlp(pinfer_ctx *ctx, const pf_layer *L, int n);

/* Phase profile (forward_stats.c), active with PF_FWD_STATS=1: t = pf_stats_now() before a
 * section, pf_stats_add(phase, kind, t) after it. kind: PFS_K1 single-token call, PFS_KB
 * batched (prefill / MTP rows), PFS_KV logged verify pass (ctx->lin_log_on). */
enum { PFS_EMBED, PFS_NORM, PFS_LIN_PRE, PFS_LIN_RULE, PFS_LIN_POST, PFS_LIN_LOG, PFS_ATTN,
       PFS_MLP_ACT, PFS_ARGMAX, PFS_ROPE, PFS_N };
enum { PFS_K1, PFS_KB, PFS_KV, PFS_KINDS };
#define PFS_KIND(ctx, n) ((ctx)->lin_log_on ? PFS_KV : (n) > 1 ? PFS_KB : PFS_K1)
extern int pf_stats_on;
double pf_stats_now(void);
void pf_stats_add(int phase, int kind, double t0);

/* Vectorized primitives (forward_ops.c): NEON on aarch64, a bit-identical scalar mirror
 * elsewhere. Per-element results never depend on the row count of the call. */
float pf_expf(float x);                       /* exp approximation used by every SiLU / sigmoid */
float pf_silu(float x);
float pf_sigmoid(float x);
void pf_silu_mul(float *a, const float *b, int n);     /* a = silu(a) * b */
void pf_mul_silu(float *y, const float *z, int n);     /* y *= silu(z) */
void pf_mul_sigmoid(float *y, const float *g, int n);  /* y *= sigmoid(g) */
void pf_add(float *y, const float *x, int n);          /* y += x */
void pf_rope(float *v, const float *c, const float *s, int rot);
void pf_rmsnorm(float *y, const float *x, const float *w, int n, float eps);
void pf_l2norm_scale(float *x, int n, float eps, float scale);
void pf_conv_silu(float *x, float *state, const float *w, int C, int K, int c0, int c1);
float pf_dot(const float *a, const float *b, int n);
float pf_softmax_exp(float *a, int n, float mx);       /* a = exp(a - mx); returns sum */
void pf_attn_pv(float *o, const float *p, int np, const float *v, int stride, int hd);
int pf_argmax(const float *v, int n);

#endif
