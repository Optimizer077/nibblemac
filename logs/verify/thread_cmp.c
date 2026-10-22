/* Throwaway check for issue #11 (VERIFY.md): two contexts decoding speculatively on two
 * threads at the same time must generate exactly what each generates alone.
 *   thread_cmp model.pq ids1.txt ids2.txt N reps threads_per_ctx */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pinfer.h"

static int read_ids(const char *path, int32_t *ids, int cap) {
    FILE *f = fopen(path, "r");
    int n = 0;
    if (!f) return -1;
    while (n < cap && fscanf(f, "%d%*[, \n]", &ids[n]) == 1) n++;
    fclose(f);
    return n;
}

typedef struct {
    const char *model;
    int32_t ids[512], out[256];
    int np, N, reps, nthr, ng, mismatches;
    const int32_t *ref;
    int nref;
} job_t;

static int gen(pinfer_ctx *c, job_t *j, int32_t *out) {
    pinfer_spec_params sp = {3, 32768, 0};
    int st;
    return pinfer_generate_greedy_ex(c, j->ids, j->np, out, j->N, &sp, NULL, NULL, &st);
}

static pinfer_ctx *load(const job_t *j) {
    pinfer_params pp = pinfer_default_params();
    pp.max_ctx = 512;
    pp.n_threads = j->nthr;
    int err;
    return pinfer_load(j->model, &pp, &err);
}

static void *worker(void *arg) {
    job_t *j = arg;
    pinfer_ctx *c = load(j);
    if (!c) { j->mismatches = -1; return NULL; }
    for (int r = 0; r < j->reps; r++) {
        int32_t out[256];
        const int n = gen(c, j, out);
        if (n != j->nref || memcmp(out, j->ref, sizeof(int32_t) * (size_t)n)) j->mismatches++;
    }
    pinfer_free(c);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 7) return fprintf(stderr, "usage: thread_cmp model ids1 ids2 N reps thr\n"), 2;
    static job_t J[2];
    int32_t ref[2][256];
    for (int i = 0; i < 2; i++) {
        J[i].model = argv[1];
        J[i].np = read_ids(argv[2 + i], J[i].ids, 512);
        J[i].N = atoi(argv[4]);
        J[i].reps = atoi(argv[5]);
        J[i].nthr = atoi(argv[6]);
        pinfer_ctx *c = load(&J[i]);   /* reference: alone, sequentially */
        if (!c || J[i].np <= 0) return 1;
        J[i].nref = gen(c, &J[i], ref[i]);
        J[i].ref = ref[i];
        pinfer_free(c);
    }
    pthread_t th[2];
    for (int i = 0; i < 2; i++) pthread_create(&th[i], NULL, worker, &J[i]);
    for (int i = 0; i < 2; i++) pthread_join(th[i], NULL);
    for (int i = 0; i < 2; i++)
        printf("context %d: %d tokens alone, %d of %d concurrent runs differ\n", i, J[i].nref,
               J[i].mismatches, J[i].reps);
    const int bad = J[0].mismatches != 0 || J[1].mismatches != 0;
    printf("%s\n", bad ? "RACE: outputs differ" : "concurrent == alone");
    printf("RESULT {\"bench\":\"thread_cmp\",\"reps\":%d,\"threads_per_ctx\":%d,\"mismatch0\":%d,"
           "\"mismatch1\":%d,\"identical\":%s}\n", J[0].reps, J[0].nthr, J[0].mismatches,
           J[1].mismatches, bad ? "false" : "true");
    return bad;
}
