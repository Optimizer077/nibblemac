/* Adaptive speculation policy (spec.h): choose the number of drafts per round from measured
 * costs and acceptance, so that speculation never runs slower than plain decoding for long.
 *
 * Model, per round with k drafts (times are EMAs of what pf_spec_decode measures):
 *   cost(k)  = t_step * (1 + cv * k) + k * t_draft + t_commit      cv = extra cost of each
 *              verified token as a fraction of a plain step, (t_verify / t_step - 1) / k
 *              averaged over the measured rounds
 *   yield(k) = 1 + p + p^2 + ... + p^k      p = per-draft acceptance, the same for every
 *              draft position (the per-position rates are close, SPEC.md 3.1), estimated as
 *              hits / trials over every verified draft with geometric decay DECAY per round
 *              and a prior of PRIOR_HITS / PRIOR_TRIALS: a round with a accepted of k drafts
 *              is a hits and a + (a < k) trials. (A first version used an EMA of "first draft
 *              accepted" whose first observation had full weight: one early miss set p = 0 and
 *              sent wikitext at 2048 context to plain mode, 6.96 vs 9.12 tok/s, CONTEXT.md §4.)
 * and the k in 0..k_max with the lowest cost / yield wins, with 3% hysteresis in favour of
 * the current k. The plain-step time comes from one calibration step at the start and one
 * every RECAL speculative rounds (the context grows, so do the step and the verify pass).
 * In plain mode p is unobservable: speculation is probed again after PROBE0 steps, doubling
 * up to PROBE_MAX while the probes keep losing. Output never depends on any of this. */
#include <string.h>

#include "spec.h"

#define ALPHA 0.25     /* EMA weight of a new observation */
#define RECAL 32       /* speculative rounds between plain-step calibrations */
#define PROBE0 8       /* plain steps before the first probe */
#define PROBE_MAX 64
#define HYST 0.97      /* a different k must be this much cheaper per token to take over */
#define DECAY 0.9      /* acceptance memory per round (~10 rounds) */
#define PRIOR_HITS 2.0 /* prior p = 2/3 with the weight of 3 drafts (SPEC.md 3.4 varied text) */
#define PRIOR_TRIALS 3.0

static double ema(double old, double x) { return old <= 0 ? x : old + ALPHA * (x - old); }

void pf_adapt_init(pf_spec_adapt *a, int k_max) {
    memset(a, 0, sizeof *a);
    a->hits = PRIOR_HITS;
    a->trials = PRIOR_TRIALS;
    a->cv = -1.0;
    a->k_cur = k_max;
    a->probe_after = PROBE0;
}

/* Expected seconds per emitted token at k drafts. */
static double per_token(const pf_spec_adapt *a, int k) {
    if (k == 0) return a->t_step;
    const double cv = a->cv >= 0 ? a->cv : 0.1, p = a->hits / a->trials;
    double y = 1.0, pk = 1.0;
    for (int i = 1; i <= k; i++) {
        pk *= p;
        y += pk;
    }
    return (a->t_step * (1.0 + cv * k) + k * a->t_draft + a->t_commit) / y;
}

static int choose(const pf_spec_adapt *a, int k_max) {
    int best = 0;
    double best_c = per_token(a, 0);
    for (int k = 1; k <= k_max; k++) {
        const double c = per_token(a, k);
        if (c < best_c) { best = k; best_c = c; }
    }
    if (a->k_cur > 0 && a->k_cur <= k_max && best != a->k_cur &&
        best_c > HYST * per_token(a, a->k_cur))
        best = a->k_cur;
    return best;
}

int pf_adapt_next(pf_spec_adapt *a, int k_max) {
    if (a->t_step <= 0) { a->calib = 1; return 0; }          /* measure a plain step first */
    if (a->rounds == 0) return k_max;                          /* no pass costs measured yet */
    if (a->k_cur == 0) return a->plain_run >= a->probe_after ? k_max : 0;
    if (a->since_cal >= RECAL) { a->calib = 1; return 0; }
    return a->k_cur;
}

void pf_adapt_plain(pf_spec_adapt *a, double sec) {
    a->t_step = ema(a->t_step, sec);
    a->since_cal = 0;
    if (a->calib) { a->calib = 0; return; }   /* calibration only: the mode stays */
    a->plain_run++;
}

void pf_adapt_round(pf_spec_adapt *a, int k_max, int k, int accepted, double t_mtp,
                    double t_verify, double t_commit) {
    if (k <= 0) return;
    if (a->t_step > 0) {
        double cv = (t_verify / a->t_step - 1.0) / k;
        if (cv < 0) cv = 0;
        a->cv = a->cv < 0 ? cv : a->cv + ALPHA * (cv - a->cv);
    }
    a->t_draft = ema(a->t_draft, t_mtp / k);
    a->t_commit = ema(a->t_commit, t_commit);
    a->hits = DECAY * a->hits + accepted;
    a->trials = DECAY * a->trials + accepted + (accepted < k ? 1 : 0);
    a->rounds++;
    a->since_cal++;
    const int best = choose(a, k_max);
    if (a->k_cur == 0) {   /* this round was a probe out of plain mode */
        a->probe_after = best > 0 ? PROBE0 : (a->probe_after * 2 < PROBE_MAX ? a->probe_after * 2
                                                                             : PROBE_MAX);
        a->plain_run = 0;
    } else if (best == 0) {
        a->plain_run = 0;
        a->probe_after = PROBE0;
    }
    a->k_cur = best;
}
