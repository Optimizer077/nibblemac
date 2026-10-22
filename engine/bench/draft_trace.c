/* draft_trace: per-position MTP chain drafts for offline simulation of draft policies
 * (engine/DRAFT.md). Plain greedy decoding; at every position P (cur = the token at P) it
 * prints the K drafts the speculative loop would chain from P and the top-1 probability of
 * each draft over the draft vocabulary, then advances by one plain step. The MTP KV cache
 * holds main-hidden rows for every committed position, exactly as in pf_spec_decode, so
 * the drafts at P do not depend on how earlier rounds were cut: a round at P with k drafts
 * accepts min(k, A(P)), where A(P) = how many of d1..dK equal the greedy tokens after P.
 *
 *   draft_trace -m model.pq --ids-file f [-n N] [-K 8] [-t 4] [--draft-vocab 32768]
 *
 * Output: "T P cur d1 ... dK p1 ... pK" per position (N positions, then K more plain steps
 * so that every traced draft can be scored), "G g0,g1,..." (greedy tokens, g0 = first
 * generated) and one RESULT line with the plain-step time. Build in WSL after
 * `make -C engine pi MATVEC=fast` (engine objects without the tools' main):
 *   O=build/aarch64/engine/obj-fast; aarch64-linux-gnu-gcc -O3 -mcpu=cortex-a72 -std=c11 \
 *     -ffp-contract=off -Wall -Wextra -Iengine/include -Iengine/src engine/bench/draft_trace.c \
 *     $(ls $O/[a-z]*.o | grep -v pinfer_) -o build/aarch64/draft_trace -static -pthread -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "forward.h"
#include "mtp.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int read_ids(const char *path, int32_t *ids, int cap) {
    FILE *f = fopen(path, "r");
    int n = 0;
    if (!f) return -1;
    while (n < cap && fscanf(f, "%d%*[, \n]", &ids[n]) == 1) n++;
    fclose(f);
    return n;
}

/* top-1 softmax probability of logits[0..n) */
static double top1_prob(const float *l, int n) {
    float mx = l[0];
    for (int i = 1; i < n; i++) mx = l[i] > mx ? l[i] : mx;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp((double)l[i] - (double)mx);
    return 1.0 / s;
}

int main(int argc, char **argv) {
    const char *model = NULL, *ids_file = NULL;
    int N = 96, K = 8, threads = 4, dv = 32768;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-m")) model = argv[i + 1];
        else if (!strcmp(argv[i], "--ids-file")) ids_file = argv[i + 1];
        else if (!strcmp(argv[i], "-n")) N = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-K")) K = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-t")) threads = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--draft-vocab")) dv = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    enum { CAP = 1 << 14 };
    static int32_t prompt[CAP];
    const int np = ids_file ? read_ids(ids_file, prompt, CAP) : -1;
    if (!model || np <= 0 || N < 1 || K < 1 || K > 16) {
        fprintf(stderr, "usage: draft_trace -m model.pq --ids-file f [-n N] [-K 8] [-t T]\n");
        return 2;
    }
    pinfer_params pp = pinfer_default_params();
    pp.n_threads = threads;
    pp.max_ctx = np + N + 2 * K + 4;
    int err = 0;
    pinfer_ctx *ctx = pinfer_load(model, &pp, &err);
    if (!ctx || pf_mtp_init(ctx) != PINFER_OK) { fprintf(stderr, "load failed %d\n", err); return 1; }
    const int H = ctx->H, V = ctx->vocab, nv = dv > 0 && dv < V ? (dv & ~3) : V;
    float *normed = malloc((size_t)PF_MAX_BATCH * H * sizeof(float));
    float *mrows = malloc((size_t)PF_MAX_BATCH * H * sizeof(float));
    float *mc = malloc(2 * (size_t)H * sizeof(float)), *m = malloc((size_t)H * sizeof(float));
    float *logits = malloc((size_t)V * sizeof(float)), *dl = malloc((size_t)V * sizeof(float));
    int32_t *gen = malloc((size_t)(N + K + 1) * sizeof(int32_t));
    if (!normed || !mrows || !mc || !m || !logits || !dl || !gen) return 1;

    /* prefill with final-normed hiddens, MTP rows for the prompt (as pf_spec_prefill) */
    int32_t next[PF_MAX_BATCH];
    int rc = PINFER_OK;
    for (int c0 = 0; c0 < np && rc == PINFER_OK; c0 += PF_MAX_BATCH) {
        const int nb = np - c0 < PF_MAX_BATCH ? np - c0 : PF_MAX_BATCH, last = c0 + nb == np;
        rc = pf_forward_h(ctx, prompt + c0, nb, last ? logits : NULL, NULL, NULL, normed);
        for (int i = 0; i < nb; i++) next[i] = c0 + i + 1 < np ? prompt[c0 + i + 1] : 0;
        if (last) next[nb - 1] = pinfer_argmax(logits, V);
        if (rc == PINFER_OK)
            rc = pf_mtp_forward(ctx, normed, next, nb, c0, mrows, last ? dl : NULL, NULL, nv);
        if (last) memcpy(m, mrows + (size_t)(nb - 1) * H, (size_t)H * sizeof(float));
    }
    if (rc != PINFER_OK) { fprintf(stderr, "prefill failed %d\n", rc); return 1; }
    int32_t cur = pinfer_argmax(logits, V);
    int32_t d[16];
    double pr[16], t_plain = 0.0;
    for (int step = 0; step < N + K && rc == PINFER_OK; step++) {
        const int P = ctx->n_past;   /* position of cur */
        gen[step] = cur;
        if (step < N) {   /* chain from (m, d1) as pf_spec_decode does; rows P.. are scratch */
            d[0] = pinfer_argmax(dl, nv);
            pr[0] = top1_prob(dl, nv);
            const float *mi = m;
            for (int s = 1; s < K && rc == PINFER_OK; s++) {
                float *mo = mc + (size_t)(s & 1) * H;
                rc = pf_mtp_forward(ctx, mi, &d[s - 1], 1, P - 1 + s, mo, dl, NULL, nv);
                d[s] = pinfer_argmax(dl, nv);
                pr[s] = top1_prob(dl, nv);
                mi = mo;
            }
            printf("T %d %d", P, cur);
            for (int s = 0; s < K; s++) printf(" %d", d[s]);
            for (int s = 0; s < K; s++) printf(" %.4f", pr[s]);
            printf("\n");
        }
        const double t0 = now_sec();
        rc = pf_forward_h(ctx, &cur, 1, logits, NULL, NULL, normed);   /* plain step */
        t_plain += now_sec() - t0;
        const int32_t nxt = pinfer_argmax(logits, V);
        /* the MTP row of position P with the main hidden (overwrites the chain's row P) */
        if (rc == PINFER_OK) rc = pf_mtp_forward(ctx, normed, &nxt, 1, P, mrows, dl, NULL, nv);
        memcpy(m, mrows, (size_t)H * sizeof(float));
        cur = nxt;
    }
    if (rc != PINFER_OK) { fprintf(stderr, "decode failed %d\n", rc); return 1; }
    printf("G");
    for (int i = 0; i < N + K; i++) printf("%s%d", i ? "," : " ", gen[i]);
    printf("\nRESULT {\"bench\": \"draft_trace\", \"ids_file\": \"%s\", \"n_prompt\": %d, "
           "\"n\": %d, \"K\": %d, \"draft_vocab\": %d, \"plain_ms\": %.2f}\n", ids_file, np, N, K,
           nv, 1e3 * t_plain / (N + K));
    pinfer_free(ctx);
    return 0;
}
