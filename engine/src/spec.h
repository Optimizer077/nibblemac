/* Self-speculative greedy decoding with the MTP head (internal interface; the public entry
 * point is pinfer_generate_greedy in pinfer.h). Design and measurements: engine/SPEC.md.
 *
 * One round, with P = tokens committed so far and `cur` = the newest token (emitted, not yet
 * fed to the main model):
 *   1. draft d1..dk: d1 comes from the previous MTP pass; d2..dk by chaining the MTP head on
 *      its own output hidden state (one row each, positions P..P+k-2);
 *   2. verify [cur, d1..dk] in ONE batched main-model pass (positions P..P+k), with the
 *      linear-attention states run on scratch copies (pf_lin_log);
 *   3. accept the longest prefix d1..da that equals the main model's argmax, plus the
 *      main model's token after it (the bonus); commit a+1 tokens to the linear-attention
 *      states (pf_lin_commit) and set n_past = P+a+1 (stale KV rows are overwritten later);
 *   4. one batched MTP pass over the a+1 newly committed positions with the main model's
 *      hidden states (this also replaces the MTP KV rows the chain wrote) -> next d1.
 * Every emitted token is a main-model argmax given the exact committed state, so the output
 * equals plain greedy decoding. */
#ifndef PF_SPEC_H
#define PF_SPEC_H

#include "forward.h"

typedef struct pf_spec_buf {
    int k_cap;            /* buffers sized for k <= k_cap */
    float *logits;        /* [k_cap+1][vocab]: verification logits */
    float *normed;        /* [PF_MAX_BATCH][H]: final-normed hiddens (MTP input) */
    float *mrows;         /* [PF_MAX_BATCH][H]: MTP output rows */
    float *mc;            /* [2][H]: chain ping-pong */
    float *dlogits;       /* [vocab]: MTP draft logits */
    int32_t cur, d1;      /* newest emitted token; first draft of the next round */
    const float *m;       /* MTP hidden of the row that produced d1 */
} pf_spec_buf;

typedef struct {
    double mtp, verify, commit;   /* seconds spent in each part of pf_spec_decode */
} pf_spec_times;

/* Reset ctx and prefill the prompt (main model, then the MTP head over every prompt row when
 * sp->k > 0). Returns the first generated token (>= 0) or a negative PINFER_E* code. */
int pf_spec_prefill(pinfer_ctx *ctx, const int32_t *prompt, int n,
                    const pinfer_spec_params *sp);

/* When to stop besides n_max. eos < 0 disables the EOS check (benchmarks). */
typedef struct {
    int eos;
    const int32_t *stop_ids;
    int n_stop;
    int (*on_token)(int32_t tok, void *user);   /* nonzero return stops after tok */
    void *user;
} pf_stop;

/* After pf_spec_prefill: out[0] = the first token, then generate until n_max tokens or a stop
 * condition (stop may be NULL). Returns the number of tokens written (>= 0; *status = PINFER_OK
 * or the error that ended generation early, e.g. PINFER_E_CTX_FULL — out[0..ret) stays valid),
 * or a negative PINFER_E* code for invalid arguments. st / tm / status may be NULL; st and tm
 * are accumulated, not reset. */
int pf_spec_decode(pinfer_ctx *ctx, int32_t *out, int n_max, const pinfer_spec_params *sp,
                   const pf_stop *stop, pinfer_spec_stats *st, pf_spec_times *tm,
                   int *status);

/* Free MTP head, rollback log and spec buffers (pinfer_free calls this). */
void pf_spec_free(pinfer_ctx *ctx);

#endif
