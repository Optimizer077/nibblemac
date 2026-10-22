/* Self-speculative greedy decoding with the MTP head (internal interface; the public entry
 * point is pinfer_generate_greedy in pinfer.h). Design and measurements: engine/SPEC.md,
 * engine/VERIFY.md (verify pass, rollback), engine/DRAFT.md (n-gram drafts, adaptive policy).
 *
 * One round, with P = tokens committed so far and `cur` = the newest token (emitted, not yet
 * fed to the main model):
 *   1. draft d1..dk, either from the MTP head (d1 comes from the previous MTP pass; d2..dk by
 *      chaining the MTP head on its own output hidden state, one row each, positions
 *      P..P+k-2) or, with pinfer_spec_params.ngram, by prompt lookup (ngram.h): the tokens
 *      that followed the most recent earlier occurrence of the newest 2-4 tokens;
 *   2. verify [cur, d1..dk] in ONE batched main-model pass (positions P..P+k); the
 *      linear-attention states advance by cur (always committed) and run d1..dk on copies
 *      while the pass logs their updates (pf_lin_log);
 *   3. accept the longest prefix d1..da that equals the main model's argmax, plus the
 *      main model's token after it (the bonus); emit them until a stop condition, commit
 *      the e emitted tokens (e = a+1 without a stop) to the linear-attention states
 *      (pf_lin_commit: conv now, the e-1 recurrent updates deferred to the next pass) and
 *      set n_past = P+e (stale KV rows are overwritten later);
 *   4. the MTP head runs over the e newly committed positions with the main model's hidden
 *      states (this also replaces the MTP KV rows the chain wrote) -> next d1. After an
 *      n-gram round (and after plain steps) these rows are kept pending and run in one batch
 *      before the next MTP round, so a run of n-gram rounds costs no MTP passes.
 * Every emitted token is a main-model argmax given the exact committed state, so the output
 * equals plain greedy decoding whatever the drafts are. Before returning, pf_spec_decode
 * applies the deferred updates (pf_lin_flush) and the pending MTP rows, so the states are
 * complete outside the loop.
 *
 * Adaptive mode (pinfer_spec_params.adaptive, spec_adapt.c): k becomes an upper bound. The
 * MTP chain stops at the first draft whose top-1 probability is below a threshold, and the
 * loop falls back to plain steps (probing again later) when its measured rate is below
 * plain decoding's; DRAFT.md has the policy and its offline validation (draft_sim.py). */
#ifndef PF_SPEC_H
#define PF_SPEC_H

#include "forward.h"
#include "ngram.h"

#define PF_SPEC_W (PINFER_SPEC_MAX_K + 2)   /* verify widths 1..PINFER_SPEC_MAX_K+1 */

/* N-gram drafts (spec.c ngram_pick, both modes; DRAFT.md §3): a match that extends over at
 * least PF_NG_TRUST tokens is drafted in full; one of PF_NG_ASK.. tokens only if its first
 * token equals the MTP head's d1, and then at most PF_NG_SHORT tokens; shorter matches are
 * ignored (MTP drafts instead). Without the MTP head (k = 0), matches shorter than
 * PF_NG_TRUST draft at most PF_NG_SHORT tokens. */
#define PF_NG_TRUST 16
#define PF_NG_ASK 4
#define PF_NG_SHORT 3

typedef struct {
    double tv[PF_SPEC_W]; /* EMA seconds of a verify pass of n tokens, 0 = not measured */
    double t_plain;       /* EMA seconds of the plain steps taken, 0 = none yet */
    double r_tok, r_sec;  /* decayed tokens emitted and seconds spent by speculative rounds */
    int n_spec;           /* speculative rounds since speculation (or a probe) started */
    int plain_mode;       /* 1: plain steps; speculation is probed after probe_after steps */
    int plain_run, probe_after;
    int probing;          /* plain mode: the current speculative rounds are a probe */
} pf_spec_adapt;

typedef struct pf_spec_buf {
    int k_cap;            /* logits sized for drafts <= k_cap */
    float *logits;        /* [k_cap+1][vocab]: verification logits */
    float *normed;        /* [PF_MAX_BATCH][H]: final-normed hiddens of a pass (MTP input) */
    float *mrows;         /* [PF_MAX_BATCH][H]: MTP output rows */
    float *mc;            /* [2][H]: chain ping-pong */
    float *dlogits;       /* [vocab]: MTP draft logits */
    int32_t cur, d1;      /* newest emitted token; first draft of the next round */
    float c1;             /* top-1 probability of d1 (adaptive mode) */
    int want_conf;        /* compute c1 (pf_spec_decode with adaptive) */
    const float *m;       /* MTP hidden of the row that produced d1 */
    /* committed rows whose MTP pass is still pending (plain steps, n-gram rounds) */
    float *pend_h;        /* [PF_MAX_BATCH][H]: their final-normed hiddens */
    int n_pend, pend_pos;
    int32_t pend_tok[PF_MAX_BATCH];   /* token that followed each pending row */
    pf_ngram ng;          /* token history (fed tokens + cur) for prompt lookup */
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

/* Adaptive policy (spec_adapt.c, engine/DRAFT.md). plain_now: this round is a plain step
 * (plain mode; 0 also starts a probe). chain_more: chain one more MTP draft after a draft
 * whose top-1 probability is conf? plain / round: record a plain step's time, or a
 * speculative round's verify width n, tokens emitted, total seconds (lookup, MTP passes,
 * verify, commit) and verify seconds. */
void pf_adapt_init(pf_spec_adapt *a);
int pf_adapt_plain_now(pf_spec_adapt *a);
int pf_adapt_chain_more(const pf_spec_adapt *a, float conf);
void pf_adapt_plain(pf_spec_adapt *a, double sec);
void pf_adapt_round(pf_spec_adapt *a, int n, int emitted, double sec, double t_verify);

#endif
