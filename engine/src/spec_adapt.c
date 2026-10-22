/* Adaptive speculation policy (spec.h; validation and history: engine/DRAFT.md §2; Python
 * mirror: engine/bench/draft_sim.py, class Simple).
 *
 * 1. Chain length from the MTP head's confidence: d1 is always drafted (it comes free with
 *    the MTP pass after a round); draft s+1 is chained only while the top-1 probability of
 *    draft s is >= CONF_TAU, up to k. Replayed on the Pi traces this is within 0.98-1.10x of
 *    the best fixed k on 20 of 21 prompts (held-out ones included), where the earlier cost
 *    model reached 0.90-1.00x of fixed k = 2 on chat: it ran k = 3 in ~90% of the rounds of
 *    chat answers and spent 7-8% of them on plain calibration steps (DRAFT.md §2b).
 * 2. Plain mode protects machines where verifying costs about as much as decoding (the
 *    scalar host build): after MIN_ROUNDS speculative rounds (2 when a 2-token verify pass
 *    costs >= 1.5 plain steps), if the realized rate (decayed tokens / seconds of the
 *    speculative rounds) is below MARGIN x the plain-step rate, take plain steps; after
 *    PROBE0, 2x, .. PROBE_MAX plain steps, PROBE_ROUNDS speculative rounds are judged on
 *    their own. The plain step's time is measured when plain steps were taken, else
 *    extrapolated from the verify passes of widths 2 and 3 (2 tv(2) - tv(3): 129 ms on the
 *    Pi vs 126 measured; no calibration step, one costs ~1% of a chat answer); without
 *    either, only the sure case switches: a rate below 1 / (cheapest verify pass) is slower
 *    than plain decoding on any machine, since a pass of n >= 2 tokens costs at least one
 *    plain step.
 * N-gram drafts follow the same rule with and without adaptive (spec.c ngram_pick).
 * Output never depends on any of this. */
#include <string.h>

#include "spec.h"

#define CONF_TAU 0.6f   /* chain on while the newest draft's top-1 probability is >= this */
#define ALPHA 0.25      /* EMA weight of a new pass time */
#define DECAY 0.95      /* rate memory per round (~20 rounds) */
#define MARGIN 0.95     /* plain mode only if speculation is >= 5% slower than plain */
#define MIN_ROUNDS 8    /* speculative rounds before plain mode can start */
#define PROBE_ROUNDS 2  /* speculative rounds of a probe out of plain mode */
#define PROBE0 8        /* plain steps before the first probe */
#define PROBE_MAX 64

static double ema(double old, double x) { return old <= 0 ? x : old + ALPHA * (x - old); }

void pf_adapt_init(pf_spec_adapt *a) {
    memset(a, 0, sizeof *a);
    a->probe_after = PROBE0;
}

int pf_adapt_plain_now(pf_spec_adapt *a) {
    if (!a->plain_mode) return 0;
    if (a->plain_run < a->probe_after) return 1;
    if (!a->probing) {   /* a probe starts: judged on its own rounds */
        a->probing = 1;
        a->n_spec = 0;
        a->r_tok = a->r_sec = 0.0;
    }
    return 0;
}

int pf_adapt_chain_more(const pf_spec_adapt *a, float conf) {
    (void)a;
    return conf >= CONF_TAU;
}

void pf_adapt_plain(pf_spec_adapt *a, double sec) {
    a->t_plain = ema(a->t_plain, sec);
    if (a->plain_mode) a->plain_run++;
}

/* Seconds of a plain step: measured, else extrapolated from verify widths 2 and 3, else 0. */
static double t_step(const pf_spec_adapt *a) {
    if (a->t_plain > 0) return a->t_plain;
    if (a->tv[2] > 0 && a->tv[3] > 0) {
        const double t = 2.0 * a->tv[2] - a->tv[3];
        return t > 0.5 * a->tv[2] ? t : 0.5 * a->tv[2];
    }
    return 0.0;
}

void pf_adapt_round(pf_spec_adapt *a, int n, int emitted, double sec, double t_verify) {
    if (n < 2 || n >= PF_SPEC_W) return;
    a->tv[n] = ema(a->tv[n], t_verify);
    a->r_tok = DECAY * a->r_tok + emitted;
    a->r_sec = DECAY * a->r_sec + sec;
    a->n_spec++;
    double tv_min = 0.0;
    for (int i = 2; i < PF_SPEC_W; i++)
        if (a->tv[i] > 0 && (tv_min <= 0 || a->tv[i] < tv_min)) tv_min = a->tv[i];
    const double ts = t_step(a);
    /* rate r_tok / r_sec below MARGIN / tv_min (sure) or below MARGIN / ts (estimated) */
    const int slow = a->r_tok * tv_min < MARGIN * a->r_sec ||
                     (ts > 0 && a->r_tok * ts < MARGIN * a->r_sec);
    if (a->plain_mode) {   /* a probe round */
        if (a->n_spec < PROBE_ROUNDS) return;
        a->probing = 0;
        a->plain_mode = slow;
        a->probe_after = !slow ? PROBE0 : a->probe_after * 2 < PROBE_MAX ? a->probe_after * 2
                                                                          : PROBE_MAX;
        a->plain_run = 0;
        a->n_spec = 0;
        if (slow) a->r_tok = a->r_sec = 0.0;
    } else if (slow && (a->n_spec >= MIN_ROUNDS ||
                        (ts > 0 && a->tv[2] >= 1.5 * ts && a->n_spec >= 2))) {
        a->plain_mode = 1;   /* compute-bound machines (2-token verify >= 1.5 steps) early */
        a->plain_run = 0;
        a->probe_after = PROBE0;
    }
}
