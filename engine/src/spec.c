/* Self-speculative greedy decoding with the MTP head. Algorithm: spec.h; notes: SPEC.md. */
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
    free(b);
}

void pf_spec_free(pinfer_ctx *ctx) {
    free_buf(ctx->spec);
    ctx->spec = NULL;
    pf_mtp_free(ctx);
    pf_lin_log_free(ctx);
}

/* Buffers for drafts of length k (grown on demand); MTP head + rollback log only if k > 0. */
static int setup(pinfer_ctx *ctx, int k) {
    int rc = k > 0 ? pf_mtp_init(ctx) : PINFER_OK;
    if (rc == PINFER_OK && k > 0) rc = pf_lin_log_alloc(ctx, k + 1);
    if (rc != PINFER_OK) return rc;
    if (ctx->spec && ctx->spec->k_cap >= k) return PINFER_OK;
    free_buf(ctx->spec);
    pf_spec_buf *b = ctx->spec = calloc(1, sizeof *b);
    if (!b) return PINFER_E_NOMEM;
    const size_t H = (size_t)ctx->H, V = (size_t)ctx->vocab;
    b->k_cap = k;
    b->logits = malloc((size_t)(k + 1) * V * sizeof(float));
    b->normed = malloc(PF_MAX_BATCH * H * sizeof(float));
    b->mrows = malloc(PF_MAX_BATCH * H * sizeof(float));
    b->mc = malloc(2 * H * sizeof(float));
    b->dlogits = malloc(V * sizeof(float));
    if (!b->logits || !b->normed || !b->mrows || !b->mc || !b->dlogits) {
        free_buf(b);
        ctx->spec = NULL;
        return PINFER_E_NOMEM;
    }
    return PINFER_OK;
}

static int draft_vocab(const pinfer_ctx *ctx, const pinfer_spec_params *sp) {
    int v = sp->draft_vocab;
    return v > 0 && v < ctx->vocab ? (v & ~3) : ctx->vocab;  /* 4-row tiles, see mtp.c */
}

/* MTP pass over n committed rows at positions pos0..: h = their main hiddens, tok = the
 * tokens that follow them. Leaves the next round's first draft in b->d1 / b->m. */
static int mtp_pending(pinfer_ctx *ctx, const float *h, const int32_t *tok, int n, int pos0,
                       int nv) {
    pf_spec_buf *b = ctx->spec;
    int rc = pf_mtp_forward(ctx, h, tok, n, pos0, b->mrows, b->dlogits, NULL, nv);
    if (rc != PINFER_OK) return rc;
    b->m = b->mrows + (size_t)(n - 1) * ctx->H;
    b->d1 = pinfer_argmax(b->dlogits, nv);
    return PINFER_OK;
}

int pf_spec_prefill(pinfer_ctx *ctx, const int32_t *prompt, int n,
                    const pinfer_spec_params *sp) {
    if (!ctx || !prompt || n <= 0 || !sp || sp->k < 0 || sp->k > PINFER_SPEC_MAX_K)
        return PINFER_E_ARG;
    pinfer_reset(ctx);
    int rc = setup(ctx, sp->k);
    if (rc != PINFER_OK) return rc;
    pf_spec_buf *b = ctx->spec;
    float *logits = b->logits;
    b->m = NULL;
    if (sp->k == 0) {  /* plain greedy needs no MTP state */
        rc = pf_forward(ctx, prompt, n, logits, NULL, NULL);
        return rc != PINFER_OK ? rc : (b->cur = pinfer_argmax(logits, ctx->vocab));
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
        rc = last ? mtp_pending(ctx, b->normed, next, nb, c0, nv)
                  : pf_mtp_forward(ctx, b->normed, next, nb, c0, NULL, NULL, NULL, nv);
        if (rc != PINFER_OK) return rc;
    }
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
        if (emit(out, &count, n_max, b->cur, stop)) return count;
    }
}

int pf_spec_decode(pinfer_ctx *ctx, int32_t *out, int n_max, const pinfer_spec_params *sp,
                   const pf_stop *stop, pinfer_spec_stats *st, pf_spec_times *tm,
                   int *status) {
    pf_spec_buf *b = ctx ? ctx->spec : NULL;
    int dummy;
    if (!status) status = &dummy;
    *status = PINFER_OK;
    if (!b || !out || n_max <= 0 || !sp || sp->k < 0 || sp->k > b->k_cap) return PINFER_E_ARG;
    if (sp->k == 0) return decode_plain(ctx, out, n_max, stop, status);
    if (!ctx->mtp || !b->m) return PINFER_E_ARG;  /* prefill was run without the MTP head */
    const int nv = draft_vocab(ctx, sp), H = ctx->H;
    const size_t V = (size_t)ctx->vocab;
    int32_t vt[PINFER_SPEC_MAX_K + 1];   /* verify tokens: cur, d1..dk */
    pf_spec_times t = {0, 0, 0};
    int count = 0, rc = PINFER_OK;
    if (emit(out, &count, n_max, b->cur, stop)) return count;

    for (;;) {
        const int P = ctx->n_past;
        int ke = sp->k;  /* never draft past n_max or the context */
        if (ke > n_max - count - 1) ke = n_max - count - 1;
        if (ke > ctx->max_ctx - P - 1) ke = ctx->max_ctx - P - 1;
        if (ke < 0) { rc = PINFER_E_CTX_FULL; break; }

        /* 1. chain the MTP head: draft s+1 from (m of draft s, draft s) at position P-1+s */
        double t0 = now_sec();
        vt[0] = b->cur;
        if (ke > 0) vt[1] = b->d1;
        const float *m = b->m;
        for (int s = 1; s < ke && rc == PINFER_OK; s++) {
            float *mo = b->mc + (size_t)(s & 1) * H;
            rc = pf_mtp_forward(ctx, m, &vt[s], 1, P - 1 + s, mo, b->dlogits, NULL, nv);
            vt[s + 1] = pinfer_argmax(b->dlogits, nv);
            m = mo;
        }
        if (rc != PINFER_OK) break;

        /* 2. verify cur + ke drafts in one batched pass, states on scratch copies */
        double t1 = now_sec();
        ctx->lin_log_on = 1;
        rc = pf_forward_h(ctx, vt, ke + 1, NULL, b->logits, NULL, b->normed);
        ctx->lin_log_on = 0;
        if (rc != PINFER_OK) break;

        /* 3. longest matching prefix + bonus token; commit a+1 tokens */
        double t2 = now_sec();
        int a = 0, g = pinfer_argmax(b->logits, (int)V);
        while (a < ke && vt[a + 1] == g) {
            a++;
            g = pinfer_argmax(b->logits + (size_t)a * V, (int)V);
        }
        pf_lin_commit(ctx, a + 1);
        ctx->n_past = P + a + 1;
        double t3 = now_sec();
        if (st) { st->rounds++; st->drafted += ke; st->accepted += a; }
        t.mtp += t1 - t0;
        t.verify += t2 - t1;
        t.commit += t3 - t2;

        int done = 0;
        for (int i = 1; i <= a && !done; i++) done = emit(out, &count, n_max, vt[i], stop);
        if (!done) done = emit(out, &count, n_max, g, stop);
        if (done) break;

        /* 4. MTP over the newly committed positions P..P+a with the main hiddens */
        vt[a + 1] = g;
        rc = mtp_pending(ctx, b->normed, vt + 1, a + 1, P, nv);
        if (rc != PINFER_OK) break;
        b->cur = g;
        t.mtp += now_sec() - t3;
    }
    if (tm) { tm->mtp += t.mtp; tm->verify += t.verify; tm->commit += t.commit; }
    *status = rc;
    return count;
}

pinfer_spec_params pinfer_default_spec_params(void) {
    pinfer_spec_params p = {2, 32768};  /* best measured on the Pi (engine/SPEC.md) */
    return p;
}

int pinfer_has_mtp(const pinfer_ctx *ctx) {
    return ctx && ctx->model.hdr->has_mtp ? 1 : 0;
}

int pinfer_generate_greedy_ex(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt,
                              int32_t *out, int n_max, const pinfer_spec_params *sp,
                              const pinfer_gen_opts *opts, pinfer_spec_stats *stats,
                              int *status) {
    pinfer_spec_params plain = {0, 0};
    int dummy;
    if (!status) status = &dummy;
    if (!sp) sp = &plain;
    if (!out || n_max <= 0 || (opts && opts->n_stop > 0 && !opts->stop_ids))
        return *status = PINFER_E_ARG;
    int rc = pf_spec_prefill(ctx, prompt, n_prompt, sp);
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

int pinfer_generate_greedy(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt,
                           int32_t *out, int n_max, const pinfer_spec_params *sp,
                           pinfer_spec_stats *stats) {
    int status;
    int n = pinfer_generate_greedy_ex(ctx, prompt, n_prompt, out, n_max, sp, NULL, stats,
                                      &status);
    return status != PINFER_OK ? status : n;  /* original contract: any error -> code */
}
