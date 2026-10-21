/* Opt-in phase profile of the forward pass outside the matmul kernels: with PF_FWD_STATS=1
 * in the environment, the timed sections of forward.c / forward_lin.c / pinfer.c accumulate
 * wall time per phase and per call kind (single token, batched prefill / MTP rows, logged
 * verify pass) and print a table to stderr at exit. Together with PF_KERN_STATS (matmul
 * time, kern_stats.c) this shows what a decode step or verify pass is made of. One getenv
 * at the first call; off = one load of a flag per section. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "forward.h"

static const char *const names[PFS_N] = {
    "embed", "norm+add", "lin_pre(conv,l2,gates)", "lin_rule", "lin_post(gated norm)",
    "lin_log copies", "attention", "mlp_act(silu*up)", "argmax", "rope_tables",
};
static const char *const kinds[PFS_KINDS] = {"n=1", "batched", "verify"};

static double sec[PFS_N][PFS_KINDS];
static long calls[PFS_N][PFS_KINDS];
int pf_stats_on = -1;

static void print_stats(void) {
    fprintf(stderr, "FSTATS %-24s %8s %10s %9s\n", "phase", "kind", "ms_total", "us/call");
    for (int k = 0; k < PFS_KINDS; k++) {
        double tot = 0;
        for (int p = 0; p < PFS_N; p++) {
            if (!calls[p][k]) continue;
            fprintf(stderr, "FSTATS %-24s %8s %10.1f %9.1f\n", names[p], kinds[k],
                    sec[p][k] * 1e3, sec[p][k] * 1e6 / (double)calls[p][k]);
            tot += sec[p][k];
        }
        if (tot > 0) fprintf(stderr, "FSTATS %-24s %8s %10.1f\n", "total", kinds[k], tot * 1e3);
    }
}

double pf_stats_now(void) {
    if (pf_stats_on < 0) {
        pf_stats_on = getenv("PF_FWD_STATS") != NULL;
        if (pf_stats_on) atexit(print_stats);
    }
    if (!pf_stats_on) return 0.0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void pf_stats_add(int phase, int kind, double t0) {
    if (pf_stats_on <= 0) return;
    sec[phase][kind] += pf_stats_now() - t0;
    calls[phase][kind]++;
}
