/* Qwen3.5 MTP (multi-token prediction) draft head. Semantics: engine/SPEC.md.
 *
 * Row t at position pos0+t takes h[t] (a final-normed hidden state: the main model's, or the
 * previous MTP step's output when chaining) and the token that follows that position:
 *   u = fc([pre_fc_norm_embedding(embed(tok)) | pre_fc_norm_hidden(h)])
 *   u = full-attention decoder block (own KV cache, RoPE at the row's position)
 *   m = mtp.norm(u);  logits = embed · m          (tied head; argmax drafts the next token)
 * The block reuses forward.c's attention / MLP code and the ctx scratch buffers, so it must
 * not run while a main-model pass is in flight. */
#ifndef PF_MTP_H
#define PF_MTP_H

#include "forward.h"

typedef struct pf_mtp {
    pf_layer layer;                 /* is_full = 1 */
    pf_mat fc;                      /* [H, 2H] */
    const float *norm_emb, *norm_hid, *norm_out;   /* effective multipliers (1 + w) */
    float *kc, *vc;                 /* [max_ctx][n_kv*hd] */
    float *e;                       /* [H] embedding row scratch */
} pf_mtp;

/* Map the mtp.* tensors and allocate the MTP KV cache (idempotent).
 * Returns PINFER_OK, PINFER_E_FORMAT (no / malformed MTP head) or PINFER_E_NOMEM. */
int pf_mtp_init(pinfer_ctx *ctx);
void pf_mtp_free(pinfer_ctx *ctx);

/* Run n rows (1 <= n <= PF_MAX_BATCH) at positions pos0..pos0+n-1.
 *   h [n][H], tok [n]: inputs (see above); writes MTP KV rows pos0..pos0+n-1
 *   out [n][H]       : m per row (may be NULL)
 *   logits_last      : logits of the last row over token ids [0, n_voc) (may be NULL)
 *   logits_all [n][n_voc] : logits of every row (may be NULL)
 * n_voc <= 0 means the full vocabulary. Returns PINFER_OK or PINFER_E_ARG. */
int pf_mtp_forward(pinfer_ctx *ctx, const float *h, const int32_t *tok, int n, int pos0,
                   float *out, float *logits_last, float *logits_all, int n_voc);

#endif
