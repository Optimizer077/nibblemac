/* Self-speculative greedy decoding with the MTP head (internal interface; the public entry
 * point is pinfer_generate_greedy in pinfer.h). Design and measurements: engine/SPEC.md.
 *
 * One round, with P = tokens committed so far and `cur` = the newest token (emitted, not yet
 * fed to the main model):
 *   1. draft d1..dk: d1 comes from the previous MTP pass; d2..dk by chaining the MTP head on
 *      its own output hidden state (one row each, positions P..P+k-2);
 *   2. verify [cur, d1..dk] in ONE batched main-model pass (positions P..P+k); the
 *      linear-attention states advance by cur (always committed) and run d1..dk on copies
 *      while the pass logs their updates (pf_lin_log);
 *   3. accept the longest prefix d1..da that equals the main model's argmax, plus the
 *      main model's token after it (the bonus); emit them until a stop condition, commit
 *      the e emitted tokens (e = a+1 without a stop) to the linear-attention states
 *      (pf_lin_commit: conv now, the e-1 recurrent updates deferred to the next pass) and
 *      set n_past = P+e (stale KV rows are overwritten later);
 *   4. one batched MTP pass over the e newly committed positions with the main model's
 *      hidden states (this also replaces the MTP KV rows the chain wrote) -> next d1.
 * Every emitted token is a main-model argmax given the exact committed state, so the output
 * equals plain greedy decoding. Before returning, pf_spec_decode applies the deferred
 * updates (pf_lin_flush), so the states are complete outside the loop.
 *
 * Adaptive mode (pinfer_spec_params.adaptive, spec_adapt.c): k becomes an upper bound. The
 * loop measures a plain step (one at the start, one every RECAL rounds), every pass of a
 * round and the per-draft acceptance, and picks the k (0 = plain) with the lowest expected
 * time per token; in plain mode it probes speculation again after a growing number of
 * steps. Plain steps keep their final-normed hidden states and run the MTP head over them
 * in one batch before the next speculative round, so the MTP cache stays complete. */
#ifndef PF_SPEC_H
#define PF_SPEC_H

#include "forward.h"

typedef struct {
    double t_step, t_draft, t_commit;  /* EMA seconds: plain step, one MTP pass, commit */
    double cv;           /* EMA of (t_verify / t_step - 1) / k: extra cost per verified token
                            as a fraction of a plain step; < 0 = no observation yet */
    double hits, trials; /* decayed counts of accepted / verified drafts (+ prior): p */
    int rounds;          /* speculative rounds observed */
    int k_cur;           /* k in use, 0 = plain mode */
    int calib;           /* the next plain step only (re)measures t_step */
    int since_cal;       /* speculative rounds since t_step was measured */
    int plain_run;       /* plain-mode steps since the last speculative round */
    int probe_after;     /* plain-mode steps before speculation is probed again */
} pf_spec_adapt;

typedef struct pf_spec_buf {
    int k_cap;            /* buffers sized for k <= k_cap */
    float *logits;        /* [k_cap+1][vocab]: verification logits */
    float *normed;        /* [PF_MAX_BATCH][H]: final-normed hiddens (MTP input) */
    float *mrows;         /* [PF_MAX_BATCH][H]: MTP output rows */
    float *mc;            /* [2][H]: chain ping-pong */
    float *dlogits;       /* [vocab]: MTP draft logits */
    int32_t cur, d1;      /* newest emitted token; first draft of the next round */
    const float *m;       /* MTP hidden of the row that produced d1 */
    /* adaptive mode: plain steps whose MTP rows are still pending (hiddens in normed) */
    int n_pend, pend_pos;
    int32_t pend_tok[PF_MAX_BATCH];   /* token that followed each pending row */
    pf_spec_adapt ad;
} pf_spec_buf;

typedef struct {
    double mtp, verify, commit, plain;   /* seconds spent in each part of pf_spec_decode */
    int64_t plain_steps;                 /* plain steps taken (adaptive mode, or n_max) */
} pf_spec_times;

/* Prefill the prompt (main model, then the MTP head over every prompt row when sp->k > 0).
 * append = 0 resets the context first; 1 appends to the tokens already in context.
 * Returns the first generated token (>= 0) or a negative PINFER_E* code. */
int pf_spec_prefill(pinfer_ctx *ctx, const int32_t *prompt, int n,
                    const pinfer_spec_params *sp, int append);

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

/* Adaptive policy (spec_adapt.c). next: k for the coming round (0 = plain step). plain /
 * round: record a plain step's time, or a speculative round's drafts, accepted count and
 * pass times (also decides the mode for the following rounds). */
void pf_adapt_init(pf_spec_adapt *a, int k_max);
int pf_adapt_next(pf_spec_adapt *a, int k_max);
void pf_adapt_plain(pf_spec_adapt *a, double sec);
void pf_adapt_round(pf_spec_adapt *a, int k_max, int k, int accepted, double t_mtp,
                    double t_verify, double t_commit);

#endif
