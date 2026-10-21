/* Opt-in matmul profile: with PF_KERN_STATS=1 in the environment, pf_matmul accumulates
 * its wall time per (rows, cols, dtype, n) and prints the table to stderr at exit. An
 * end-to-end run then shows how much of a decode / verify / prefill step is matmul at all
 * (KERNELS.md "where the time goes"). Off by default: one getenv at the first call.
 */
#include <stdio.h>
#include <stdlib.h>

#include "kern.h"

#define SLOTS 128

typedef struct {
    int rows, cols, n;
    unsigned dtype;
    long calls;
    double sec, bytes;
} stat_t;

static stat_t slots[SLOTS];
static int enabled = -1;

static void print_stats(void) {
    double total = 0;
    long calls = 0;
    fprintf(stderr, "KSTATS %8s %6s %5s %3s %8s %10s %9s %8s\n", "rows", "cols", "dtype", "n",
            "calls", "ms_total", "us/call", "GB/s");
    for (int i = 0; i < SLOTS; i++) {
        const stat_t *s = &slots[i];
        if (!s->calls) continue;
        fprintf(stderr, "KSTATS %8d %6d %5u %3d %8ld %10.1f %9.1f %8.2f\n", s->rows, s->cols,
                s->dtype, s->n, s->calls, s->sec * 1e3, s->sec * 1e6 / (double)s->calls,
                s->sec > 0 ? s->bytes / s->sec / 1e9 : 0.0);
        total += s->sec;
        calls += s->calls;
    }
    fprintf(stderr, "KSTATS total calls=%ld ms=%.1f\n", calls, total * 1e3);
}

int kern_stats_enabled(void) {
    if (enabled < 0) {
        enabled = getenv("PF_KERN_STATS") != NULL;
        if (enabled) atexit(print_stats);
    }
    return enabled;
}

void kern_stats_add(int rows, int cols, unsigned dtype, int n, double sec, size_t bytes) {
    /* open addressing on the shape; the table is tiny and never shrinks */
    unsigned h = ((unsigned)rows * 2654435761u ^ (unsigned)cols * 40503u ^ (unsigned)n * 97u ^
                  dtype) % SLOTS;
    for (int k = 0; k < SLOTS; k++, h = (h + 1) % SLOTS) {
        stat_t *s = &slots[h];
        if (s->calls && (s->rows != rows || s->cols != cols || s->n != n || s->dtype != dtype))
            continue;
        if (!s->calls) *s = (stat_t){rows, cols, n, dtype, 0, 0.0, 0.0};
        s->calls++;
        s->sec += sec;
        s->bytes += (double)bytes;
        return;
    }
}
