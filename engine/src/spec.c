/* Self-speculative greedy decoding with the MTP head and prompt-lookup drafts. Algorithm:
 * spec.h; notes: SPEC.md, VERIFY.md, DRAFT.md. */
#define _POSIX_C_SOURCE 200809L
#include "spec.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mtp.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void free_buf(pf_spec_buf *b) {
    if (!b) return;
    free(b->logits);
    free(b->normed);
    free(b->mrows);
    free(b->mc);
    free(b->dlogits);
    free(b->pend_h);
    pf_ngram_free(&b->ng);
    free(b);
}

void pf_spec_free(pinfer_ctx *ctx) {
    free_buf(ctx->spec);
    ctx->spec = NULL;
    pf_mtp_free(ctx);
    pf_lin_log_free(ctx);
}

static int valid_params(const pinfer_spec_params *sp) {
    return sp && sp->k >= 0 && sp->k <= PINFER_SPEC_MAX_K && sp->ngram >= 0 &&
           sp->ngram <= PINFER_SPEC_MAX_K;
}

/* Most drafts one round of sp verifies. */
static int max_drafts(const pinfer_spec_params *sp) {
    return sp->k > sp->ngram ? sp->k : sp->ngram;
}

/* Buffers for rounds of up to max_drafts(sp) drafts (the logits grow on demand); the MTP head
 * only if sp->k > 0, the rollback log only if a round can draft. The other buffers do not
 * depend on k, so the n-gram history survives a larger k. */
static int setup(pinfer_ctx *ctx, const pinfer_spec_params *sp) {
    const int kd = max_drafts(sp);
    int rc = sp->k > 0 ? pf_mtp_init(ctx) : PINFER_OK;
    if (rc == PINFER_OK && kd > 0) rc = pf_lin_log_alloc(ctx, kd + 1);
    if (rc != PINFER_OK) return rc;
    const size_t H = (size_t)ctx->H, V = (size_t)ctx->vocab;
    pf_spec_buf *b = ctx->spec;
    if (!b) {
        b = ctx->spec = calloc(1, sizeof *b);
        if (!b) return PINFER_E_NOMEM;
        b->k_cap = -1;
        b->normed = malloc(PF_MAX_BATCH * H * sizeof(float));
        b->mrows = malloc(PF_MAX_BATCH * H * sizeof(float));
        b->mc = malloc(2 * H * sizeof(float));
        b->dlogits = malloc(V * sizeof(float));
        b->pend_h = malloc(PF_MAX_BATCH * H * sizeof(float));
        if (!b->normed || !b->mrows || !b->mc || !b->dlogits || !b->pend_h ||
            pf_ngram_init(&b->ng, ctx->max_ctx + 1)) {
            free_buf(b);
            ctx->spec = NULL;
            return PINFER_E_NOMEM;
        }
    }
    if (b->k_cap < kd) {
        float *lg = malloc((size_t)(kd + 1) * V * sizeof(float));
        if (!lg) return PINFER_E_NOMEM;
        free(b->logits);
        b->logits = lg;
        b->k_cap = kd;
    }
    return PINFER_OK;
}

static int draft_vocab(const pinfer_ctx *ctx, const pinfer_spec_params *sp) {
    int v = sp->draft_vocab;
    return v >= 4 && v < ctx->vocab ? (v & ~3) : ctx->vocab;  /* 4-row tiles, see mtp.c */
}

/* Top-1 softmax probability of logits l[0..n) (overwrites l with the exponentials). */
static float top1_prob(float *l, int n) {
    const float s = pf_softmax_exp(l, n, pf_softmax_max(l, n));
    return s > 0.0f ? 1.0f / s : 0.0f;
}

/* MTP pass over n committed rows at positions pos0..: h = their main hiddens, tok = the
 * tokens that follow them. Leaves the next round's first draft in b->d1 / b->m (and its
 * top-1 probability in b->c1 when the adaptive policy wants it). */
static int mtp_pending(pinfer_ctx *ctx, const float *h, const int32_t *tok, int n, int pos0,
                       int nv) {
    pf_spec_buf *b = ctx->spec;
    int rc = pf_mtp_forward(ctx, h, tok, n, pos0, b->mrows, b->dlogits, NULL, nv);
    if (rc != PINFER_OK) return rc;
    b->m = b->mrows + (size_t)(n - 1) * ctx->H;
    b->d1 = pinfer_argmax(b->dlogits, nv);
    if (b->want_conf) b->c1 = top1_prob(b->dlogits, nv);
    ctx->mtp_rows = pos0 + n;
    return PINFER_OK;
}

/* MTP rows of the plain steps and n-gram rounds since the last MTP pass, in one batch. */
static int flush_pending(pinfer_ctx *ctx, int nv) {
    pf_spec_buf *b = ctx->spec;
    if (b->n_pend == 0) return PINFER_OK;
    int rc = mtp_pending(ctx, b->pend_h, b->pend_tok, b->n_pend, b->pend_pos, nv);
    b->n_pend = 0;
    return rc;
}

/* Keep n committed rows (final-normed hiddens h, positions pos.., followed by tok) for the
 * next flush instead of running the MTP head over them now. */
static int defer_rows(pinfer_ctx *ctx, const float *h, const int32_t *tok, int n, int pos,
                      int nv) {
    pf_spec_buf *b = ctx->spec;
    const size_t H = (size_t)ctx->H;
    int rc = b->n_pend + n > PF_MAX_BATCH ? flush_pending(ctx, nv) : PINFER_OK;
    if (rc != PINFER_OK) return rc;
    if (b->n_pend == 0) b->pend_pos = pos;
    memcpy(b->pend_h + (size_t)b->n_pend * H, h, (size_t)n * H * sizeof(float));
    memcpy(b->pend_tok + b->n_pend, tok, (size_t)n * sizeof(int32_t));
    b->n_pend += n;
    return PINFER_OK;
}

/* Prompt lookup for the coming round (both modes, spec.h PF_NG_*): writes the n-gram draft
 * to d and returns its length, or 0 for MTP drafts / a plain step. A match of PF_NG_ASK..
 * PF_NG_TRUST-1 tokens is checked against the MTP's d1 first, which may run the pending MTP
 * rows (d1 is needed by an MTP round anyway). On the Pi traces an unchecked short match was
 * right 10% of the time when it disagreed with d1 and 95% when it agreed (DRAFT.md §3). */
static int ngram_pick(pinfer_ctx *ctx, const pinfer_spec_params *sp, int left, int32_t *d,
                      int nv, int *rc) {
    pf_spec_buf *b = ctx->spec;
    int match = 0;
    const int max = sp->ngram < left ? sp->ngram : left;
    const int n = max > 0 ? pf_ngram_draft(&b->ng, d, max, &match) : 0;
    if (n == 0) return 0;
    if (match >= PF_NG_TRUST) return n;
    if (sp->k == 0) return n < PF_NG_SHORT ? n : PF_NG_SHORT;   /* no MTP head to ask */
    if (match < PF_NG_ASK) return 0;
    if (b->n_pend > 0 && (*rc = flush_pending(ctx, nv)) != PINFER_OK) return 0;
    if (d[0] != b->d1) return 0;
    return n < PF_NG_SHORT ? n : PF_NG_SHORT;
}

int pf_spec_prefill(pinfer_ctx *ctx, const int32_t *prompt, int n,
                    const pinfer_spec_params *sp, int append) {
    if (!ctx || !prompt || n <= 0 || !valid_params(sp)) return PINFER_E_ARG;
    if (!append) pinfer_reset(ctx);
    if (n > ctx->max_ctx - ctx->n_past) return PINFER_E_CTX_FULL;  /* nothing consumed */
    int rc = setup(ctx, sp);
    if (rc != PINFER_OK) return rc;
    pf_spec_buf *b = ctx->spec;
    float *logits = b->logits;
    const int base = ctx->n_past;
    b->m = NULL;
    b->n_pend = 0;
    b->want_conf = sp->adaptive;
    /* The n-gram history is the fed tokens plus the last emitted one, which a continuation
     * prompt leads with. Any other context (pinfer_eval, a loaded state) is not described by
     * it: start a new history (drafts only, never the output, depend on it). */
    if (append && b->ng.n == base + 1) pf_ngram_pop(&b->ng);
    else pf_ngram_reset(&b->ng);
    pf_ngram_push(&b->ng, prompt, n);
    /* Positions evaluated without the MTP head (pinfer_eval, k = 0, a loaded state) have no
     * MTP rows: zero them rather than let the drafts attend to stale memory. */
    if (sp->k > 0 && ctx->mtp_rows < base) {
        pf_kv_zero_rows(ctx, ctx->mtp->kc, ctx->mtp_rows, base);
        pf_kv_zero_rows(ctx, ctx->mtp->vc, ctx->mtp_rows, base);
    }
    if (sp->k == 0) {  /* plain greedy and n-gram drafts need no MTP state */
        rc = pf_forward(ctx, prompt, n, logits, NULL, NULL);
        if (rc != PINFER_OK) return rc;
        b->cur = pinfer_argmax(logits, ctx->vocab);
        pf_ngram_push(&b->ng, &b->cur, 1);
        return b->cur;
    }
    /* Chunk by chunk: main pass with final-normed hiddens, then the MTP over the same rows
     * (row i pairs h_i with token i+1; the last row needs the first generated token). */
    const int nv = draft_vocab(ctx, sp);
    int32_t next[PF_MAX_BATCH];
    for (int c0 = 0; c0 < n; c0 += ctx->batch) {
        const int nb = n - c0 < ctx->batch ? n - c0 : ctx->batch, last = c0 + nb == n;
        rc = pf_forward_h(ctx, prompt + c0, nb, last ? logits : NULL, NULL, NULL, b->normed);
        if (rc != PINFER_OK) return rc;
        for (int i = 0; i < nb; i++) next[i] = c0 + i + 1 < n ? prompt[c0 + i + 1] : 0;
        if (last) next[nb - 1] = b->cur = pinfer_argmax(logits, ctx->vocab);
        /* only the last row of the prompt drafts; earlier chunks just fill the MTP KV */
        rc = last ? mtp_pending(ctx, b->normed, next, nb, base + c0, nv)
                  : pf_mtp_forward(ctx, b->normed, next, nb, base + c0, NULL, NULL, NULL, nv);
        if (rc != PINFER_OK) return rc;
    }
    pf_ngram_push(&b->ng, &b->cur, 1);
    return b->cur;
}

/* Emit one token; returns 1 when generation must stop (n_max, stop token or callback). */
static int emit(int32_t *out, int *count, int n_max, int32_t tok, const pf_stop *sp) {
    out[(*count)++] = tok;
    int stop = *count >= n_max;
    if (sp) {
        if (sp->on_token && sp->on_token(tok, sp->user)) stop = 1;
        if (tok == sp->eos) stop = 1;
        for (int i = 0; i < sp->n_stop && !stop; i++) stop = tok == sp->stop_ids[i];
    }
    return stop;
}

static int decode_plain(pinfer_ctx *ctx, int32_t *out, int n_max, const pf_stop *stop,
                        int *status) {
    pf_spec_buf *b = ctx->spec;
    int count = 0;
    if (emit(out, &count, n_max, b->cur, stop)) return count;
    for (;;) {
        int rc = pf_forward(ctx, &b->cur, 1, b->logits, NULL, NULL);
        if (rc != PINFER_OK) { *status = rc; return count; }
        b->cur = pinfer_argmax(b->logits, ctx->vocab);
        pf_ngram_push(&b->ng, &b->cur, 1);
        if (emit(out, &count, n_max, b->cur, stop)) return count;
    }
}

/* One plain step inside the speculative loop: with the MTP head its final-normed hidden is
 * kept for a later batched MTP pass (flush_pending). Returns 1 when generation must stop. */
static int plain_step(pinfer_ctx *ctx, int32_t *out, int *count, int n_max,
                      const pf_stop *stop, int nv, int use_mtp, int adaptive,
                      pf_spec_times *t, int *rc) {
    pf_spec_buf *b = ctx->spec;
    const int P = ctx->n_past;
    if (use_mtp && b->n_pend == PF_MAX_BATCH && (*rc = flush_pending(ctx, nv)) != PINFER_OK)
        return 1;
    const double t0 = now_sec();
    *rc = pf_forward_h(ctx, &b->cur, 1, b->logits, NULL, NULL,
                       use_mtp ? b->pend_h + (size_t)b->n_pend * ctx->H : NULL);
    if (*rc != PINFER_OK) return 1;
    b->cur = pinfer_argmax(b->logits, ctx->vocab);
    if (use_mtp) {
        if (b->n_pend == 0) b->pend_pos = P;
        b->pend_tok[b->n_pend++] = b->cur;
    }
    pf_ngram_push(&b->ng, &b->cur, 1);
    const double dt = now_sec() - t0;
    t->plain += dt;
    t->plain_steps++;
    if (adaptive) pf_adapt_plain(&b->ad, dt);
    return emit(out, count, n_max, b->cur, stop);
}

int pf_spec_decode(pinfer_ctx *ctx, int32_t *out, int n_max, const pinfer_spec_params *sp,
                   const pf_stop *stop, pinfer_spec_stats *st, pf_spec_times *tm,
                   int *status) {
    pf_spec_buf *b = ctx ? ctx->spec : NULL;
    int dummy;
    if (!status) status = &dummy;
    *status = PINFER_OK;
    if (!b || !out || n_max <= 0 || !valid_params(sp) || max_drafts(sp) > b->k_cap)
        return *status = PINFER_E_ARG;
    if (max_drafts(sp) == 0) return decode_plain(ctx, out, n_max, stop, status);
    const int use_mtp = sp->k > 0;
    if (use_mtp && (!ctx->mtp || !b->m))   /* prefill ran without the MTP head */
        return *status = PINFER_E_ARG;
    const int nv = draft_vocab(ctx, sp), H = ctx->H, batch = ctx->batch;
    const size_t V = (size_t)ctx->vocab;
    int32_t vt[PINFER_SPEC_MAX_K + 2];   /* verify tokens: cur, d1..dk, then the bonus */
    pf_spec_times t = {0, 0, 0, 0, 0};
    int count = 0, rc = PINFER_OK;
    if (sp->adaptive) pf_adapt_init(&b->ad);
    b->want_conf = sp->adaptive;
    b->n_pend = 0;
    if (emit(out, &count, n_max, b->cur, stop)) return count;

    for (;;) {
        const int P = ctx->n_past;
        int left = n_max - count - 1;                          /* never draft past n_max */
        if (left > ctx->max_ctx - P - 1) left = ctx->max_ctx - P - 1;   /* or the context */
        if (left < 0) { rc = PINFER_E_CTX_FULL; break; }
        const double t0 = now_sec();
        /* 1a. prompt lookup: the n-gram draft, if the newest tokens occurred before and the
         * match is trusted (ngram_pick); else MTP drafts or, without them, a plain step */
        int ke = 0, use_ng = 0;
        if (!(sp->adaptive && pf_adapt_plain_now(&b->ad))) {
            if (sp->ngram > 0 && left > 0) ke = ngram_pick(ctx, sp, left, vt + 1, nv, &rc);
            if (rc != PINFER_OK) break;
            use_ng = ke > 0;
            if (!use_ng) ke = sp->k < left ? sp->k : left;
        }
        if (ke == 0) {
            if (st) st->rounds++;
            if (plain_step(ctx, out, &count, n_max, stop, nv, use_mtp, sp->adaptive, &t, &rc))
                break;
            continue;
        }

        /* 1b. or chain the MTP head: draft s+1 from (m of draft s, draft s) at position P-1+s;
         * adaptive: only while the newest draft's top-1 probability is high enough */
        vt[0] = b->cur;
        if (!use_ng) {
            if ((rc = flush_pending(ctx, nv)) != PINFER_OK) break;   /* b->m / b->d1 current */
            vt[1] = b->d1;
            const float *m = b->m;
            float conf = b->c1;
            int s = 1;
            while (s < ke && (!sp->adaptive || pf_adapt_chain_more(&b->ad, conf))) {
                float *mo = b->mc + (size_t)(s & 1) * H;
                rc = pf_mtp_forward(ctx, m, &vt[s], 1, P - 1 + s, mo, b->dlogits, NULL, nv);
                if (rc != PINFER_OK) break;
                vt[s + 1] = pinfer_argmax(b->dlogits, nv);
                if (sp->adaptive) conf = top1_prob(b->dlogits, nv);
                m = mo;
                s++;
            }
            if (rc != PINFER_OK) break;
            ke = s;
        }

        /* 2. verify cur + ke drafts in ONE batched pass; the linear-attention states take
         * cur (always committed) and the previous round's deferred updates in place, the
         * drafts run on copies (the log covers one pf_linear_attn call, so the pass must not
         * be split by a small -b) */
        const double t1 = now_sec();   /* t0..t1: lookup, flush and MTP chain */
        ctx->lin_log_on = 1;
        if (batch < ke + 1) pf_set_batch(ctx, ke + 1);
        rc = pf_forward_h(ctx, vt, ke + 1, NULL, b->logits, NULL, b->normed);
        pf_set_batch(ctx, batch);
        ctx->lin_log_on = 0;
        if (rc != PINFER_OK) break;

        /* 3. longest matching prefix + bonus token; emit, then commit the e emitted tokens
         * (e = a+1 unless a stop condition ends the round early) */
        const double t2 = now_sec();
        int a = 0, g = pinfer_argmax(b->logits, (int)V);
        while (a < ke && vt[a + 1] == g) {
            a++;
            g = pinfer_argmax(b->logits + (size_t)a * V, (int)V);
        }
        vt[a + 1] = g;
        int e = 0, done = 0;
        while (e <= a && !done) done = emit(out, &count, n_max, vt[++e], stop);
        pf_lin_commit(ctx, e);
        ctx->n_past = P + e;
        b->cur = vt[e];
        pf_ngram_push(&b->ng, vt + 1, e);
        const double t3 = now_sec();
        if (st) {
            st->rounds++; st->drafted += ke; st->accepted += a;
            if (use_ng) { st->ngram_rounds++; st->ngram_drafted += ke; st->ngram_accepted += a; }
        }
        t.verify += t2 - t1;
        t.commit += t3 - t2;

        /* 4. MTP over the newly committed positions P..P+e-1 with the main hiddens (also
         * after a stop: the MTP cache then stays complete for a continuation); after an
         * n-gram round the rows wait for the next MTP round, which needs d1 */
        if (use_mtp)
            rc = use_ng ? defer_rows(ctx, b->normed, vt + 1, e, P, nv)
                        : mtp_pending(ctx, b->normed, vt + 1, e, P, nv);
        const double t4 = now_sec();
        t.mtp += (t1 - t0) + (t4 - t3);   /* lookup, flush, chain steps and this pass */
        if (sp->adaptive) pf_adapt_round(&b->ad, ke + 1, e, t4 - t0, t2 - t1);
        if (rc != PINFER_OK || done) break;
    }
    const double tp = now_sec();
    if (rc == PINFER_OK && use_mtp) rc = flush_pending(ctx, nv);   /* pending rows, same reason */
    const double tf = now_sec();
    pf_lin_flush(ctx);   /* the last commit's deferred updates: states complete on return */
    t.mtp += tf - tp;
    t.commit += now_sec() - tf;
    if (tm) {
        tm->mtp += t.mtp; tm->verify += t.verify; tm->commit += t.commit;
        tm->plain += t.plain; tm->plain_steps += t.plain_steps;
    }
    *status = rc;
    return count;
}

pinfer_spec_params pinfer_default_spec_params(void) {
    /* adaptive k <= 4 with prompt-lookup drafts: within 0.98-1.06x of the best fixed k on every
     * benchmark prompt, faster on prose, long context and copied code (engine/DRAFT.md); with the
     * single-pass n = 5 kernels, k <= 4 beats k <= 3 (kcap-merge6_20261022_145226.json) */
    pinfer_spec_params p = {4, 32768, 1, 7};
    return p;
}

int pinfer_has_mtp(const pinfer_ctx *ctx) {
    return ctx && ctx->model.hdr->has_mtp ? 1 : 0;
}

static int generate(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt, int32_t *out,
                    int n_max, const pinfer_spec_params *sp, const pinfer_gen_opts *opts,
                    pinfer_spec_stats *stats, int *status, int append) {
    pinfer_spec_params plain = {0, 0, 0, 0};
    int dummy;
    if (!status) status = &dummy;
    if (!sp) sp = &plain;
    if (!out || n_max <= 0 || (opts && opts->n_stop > 0 && !opts->stop_ids))
        return *status = PINFER_E_ARG;
    int rc = pf_spec_prefill(ctx, prompt, n_prompt, sp, append);
    if (rc < 0) return *status = rc;
    pf_stop stop = {pinfer_eos_id(ctx), NULL, 0, NULL, NULL};
    if (opts) {
        stop.stop_ids = opts->stop_ids;
        stop.n_stop = opts->n_stop;
        stop.on_token = opts->on_token;
        stop.user = opts->user;
    }
    return pf_spec_decode(ctx, out, n_max, sp, &stop, stats, NULL, status);
}

int pinfer_generate_greedy_ex(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt,
                              int32_t *out, int n_max, const pinfer_spec_params *sp,
                              const pinfer_gen_opts *opts, pinfer_spec_stats *stats,
                              int *status) {
    return generate(ctx, prompt, n_prompt, out, n_max, sp, opts, stats, status, 0);
}

int pinfer_generate_greedy_cont(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt,
                                int32_t *out, int n_max, const pinfer_spec_params *sp,
                                const pinfer_gen_opts *opts, pinfer_spec_stats *stats,
                                int *status) {
    return generate(ctx, prompt, n_prompt, out, n_max, sp, opts, stats, status, 1);
}

int pinfer_generate_greedy(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt,
                           int32_t *out, int n_max, const pinfer_spec_params *sp,
                           pinfer_spec_stats *stats) {
    int status;
    int n = pinfer_generate_greedy_ex(ctx, prompt, n_prompt, out, n_max, sp, NULL, stats,
                                      &status);
    return status != PINFER_OK ? status : n;  /* original contract: any error -> code */
}
