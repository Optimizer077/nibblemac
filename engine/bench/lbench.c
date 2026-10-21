/* lbench: the Gated DeltaNet recurrence kernel (kern_lin.c) vs the scalar code of
 * forward_lin.c: bit-exactness (state, outputs, rollback log, commit replay) and time per
 * layer for n = 1 / 4 / 64 tokens with 1..4 threads, cold state (18 layers rotate through a
 * 18 MB arena, as in decode) and hot state. Prints RESULT lines for harness/pi.py bench.
 *   lbench [--threads 1,2,4] [--n 1,4,64] [--layers 18] [--check]
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kern.h"

#define HEADS 16
#define REP 1
#define DK 128
#define DV 128
#define C (2 * HEADS * DK + HEADS * DV) /* conv_dim: q | k | v */
#define LOGW (DK + DV + 1)

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- scalar reference: verbatim from engine/src/forward_lin.c ---- */
static void delta_rule(float *S, const float *q, const float *k, const float *v,
                       float decay, float beta, float *o, float *tmp, int dk, int dv) {
    float *kv = tmp;
    memset(kv, 0, (size_t)dv * sizeof(float));
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float ki = k[i];
        for (int j = 0; j < dv; j++) {
            row[j] *= decay;
            kv[j] += row[j] * ki;
        }
    }
    for (int j = 0; j < dv; j++) kv[j] = (v[j] - kv[j]) * beta;
    memset(o, 0, (size_t)dv * sizeof(float));
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float ki = k[i], qi = q[i];
        for (int j = 0; j < dv; j++) {
            row[j] += ki * kv[j];
            o[j] += row[j] * qi;
        }
    }
}

static void ref_layer(float *S, const float *qkv, const float *decay, const float *beta,
                      float *o, float *log, int n) {
    float tmp[DV];
    for (int h = 0; h < HEADS; h++) {
        float *Sh = S + (size_t)h * DK * DV;
        for (int t = 0; t < n; t++) {
            const float *x = qkv + (size_t)t * C;
            delta_rule(Sh, x + h * DK, x + HEADS * DK + h * DK, x + 2 * HEADS * DK + h * DV,
                       decay[t * HEADS + h], beta[t * HEADS + h], o + (size_t)t * HEADS * DV + h * DV,
                       tmp, DK, DV);
            if (log) {
                float *u = log + ((size_t)t * HEADS + h) * LOGW;
                memcpy(u, x + HEADS * DK + h * DK, DK * sizeof(float));
                memcpy(u + DK, tmp, DV * sizeof(float));
                u[DK + DV] = decay[t * HEADS + h];
            }
        }
    }
}

static void ref_commit(float *S, const float *log, int keep) { /* pf_lin_commit inner loop */
    for (int h = 0; h < HEADS; h++) {
        float *Sh = S + (size_t)h * DK * DV;
        for (int t = 0; t < keep; t++) {
            const float *u = log + ((size_t)t * HEADS + h) * LOGW, *d = u + DK;
            const float decay = u[DK + DV];
            for (int i = 0; i < DK; i++) {
                float *row = Sh + (size_t)i * DV;
                const float ki = u[i];
                for (int j = 0; j < DV; j++) {
                    row[j] *= decay;
                    row[j] += ki * d[j];
                }
            }
        }
    }
}

/* Kernel state layout: S[h][j / 32][i][j % 32] (kern_lin.c); the reference uses [h][i][j]. */
static size_t chunked_index(int h, int i, int j) {
    return ((size_t)h * (DV / 32) + j / 32) * DK * 32 + (size_t)i * 32 + j % 32;
}

static void to_chunked(const float *src, float *dst) {
    for (int h = 0; h < HEADS; h++)
        for (int i = 0; i < DK; i++)
            for (int j = 0; j < DV; j++) dst[chunked_index(h, i, j)] = src[((size_t)h * DK + i) * DV + j];
}

static void from_chunked(const float *src, float *dst) {
    for (int h = 0; h < HEADS; h++)
        for (int i = 0; i < DK; i++)
            for (int j = 0; j < DV; j++) dst[((size_t)h * DK + i) * DV + j] = src[chunked_index(h, i, j)];
}

static float frand(uint64_t *s) {
    *s ^= *s >> 12, *s ^= *s << 25, *s ^= *s >> 27;
    return (float)((*s * 2685821657736338717ull) >> 40) * (1.0f / 16777216.0f) - 0.5f;
}

int main(int argc, char **argv) {
    int threads[8] = {1, 2, 4}, nthr = 3, ns[8] = {1, 4, 64}, nn = 3, layers = 18, check = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        int *dst = !strcmp(a, "--threads") ? threads : !strcmp(a, "--n") ? ns : NULL;
        if (dst) {
            int k = 0;
            for (char *e; *v && k < 8; v = *e ? e + 1 : e) dst[k++] = (int)strtol(v, &e, 10);
            *(dst == threads ? &nthr : &nn) = k;
            i++;
        } else if (!strcmp(a, "--layers")) layers = atoi(v), i++;
        else if (!strcmp(a, "--check")) check = 1;
    }
    const size_t hs = (size_t)HEADS * DK * DV;
    const int nmax = 64;
    float *S0 = malloc(hs * sizeof(float)), *Sa = malloc(hs * sizeof(float));
    float *arena = malloc((size_t)layers * hs * sizeof(float));
    float *qkv = malloc((size_t)nmax * C * sizeof(float));
    float *decay = malloc(nmax * HEADS * sizeof(float)), *beta = malloc(nmax * HEADS * sizeof(float));
    float *oa = malloc((size_t)nmax * HEADS * DV * sizeof(float)), *ob = malloc((size_t)nmax * HEADS * DV * sizeof(float));
    float *la = malloc((size_t)nmax * HEADS * LOGW * sizeof(float)), *lb = malloc((size_t)nmax * HEADS * LOGW * sizeof(float));
    uint64_t seed = 12345;
    for (size_t i = 0; i < hs; i++) S0[i] = frand(&seed) * 0.1f;
    for (size_t i = 0; i < (size_t)nmax * C; i++) qkv[i] = frand(&seed);
    for (int i = 0; i < nmax * HEADS; i++) decay[i] = 0.9f + 0.1f * (frand(&seed) + 0.5f), beta[i] = frand(&seed) + 0.5f;
    for (int l = 0; l < layers; l++) memcpy(arena + (size_t)l * hs, S0, hs * sizeof(float));

    int fails = 0;
    pf_pool *pool = pf_pool_create(4);
    float *Sc = malloc(hs * sizeof(float)), *Sr = malloc(hs * sizeof(float)); /* chunked, back */
    for (int ni = 0; ni < nn; ni++) { /* exactness: in-place, logged (scratch) and commit */
        const int n = ns[ni];
        kern_lin_args a = {n, HEADS, REP, DK, DV, C, HEADS * DV, qkv, decay, beta, Sa, ob, NULL};
        memcpy(S0, arena, hs * sizeof(float));
        to_chunked(arena, Sc);
        memcpy(Sa, Sc, hs * sizeof(float));
        ref_layer(S0, qkv, decay, beta, oa, la, n);
        a.log = lb;
        kern_lin_rule(pool, &a); /* logged: Sa must stay untouched */
        int ok = !memcmp(Sa, Sc, hs * sizeof(float)) &&
                 !memcmp(oa, ob, (size_t)n * HEADS * DV * sizeof(float)) &&
                 !memcmp(la, lb, (size_t)n * HEADS * LOGW * sizeof(float));
        a.log = NULL;
        kern_lin_rule(pool, &a); /* in place: Sa must equal the reference state */
        from_chunked(Sa, Sr);
        ok = ok && !memcmp(Sr, S0, hs * sizeof(float));
        memcpy(Sa, Sc, hs * sizeof(float));
        memcpy(S0, arena, hs * sizeof(float));
        ref_commit(S0, la, n > 2 ? 2 : n);
        kern_lin_apply(pool, Sa, lb, n > 2 ? 2 : n, HEADS, DK, DV);
        from_chunked(Sa, Sr);
        ok = ok && !memcmp(Sr, S0, hs * sizeof(float));
        printf("exact n=%d: %s\n", n, ok ? "ok" : "FAIL");
        printf("RESULT {\"bench\":\"lbench_exact\",\"n\":%d,\"pass\":%s}\n", n, ok ? "true" : "false");
        fails += !ok;
    }
    free(Sc), free(Sr);
    pf_pool_destroy(pool);
    if (check) return fails ? 1 : 0;

    for (int ti = 0; ti < nthr; ti++) {
        pool = pf_pool_create(threads[ti]);
        for (int ni = 0; ni < nn; ni++) {
            const int n = ns[ni];
            for (int mode = 0; mode < 2; mode++) { /* 0 = NEON kernel, 1 = scalar reference */
                if (mode && threads[ti] != 1) continue;
                double best = 1e30;
                for (int rep = 0; rep < 3; rep++) {
                    const double t0 = now();
                    for (int l = 0; l < layers; l++) {
                        kern_lin_args a = {n, HEADS, REP, DK, DV, C, HEADS * DV, qkv, decay, beta,
                                           arena + (size_t)l * hs, ob, NULL};
                        if (mode) ref_layer(a.S, qkv, decay, beta, ob, NULL, n);
                        else kern_lin_rule(pool, &a);
                    }
                    const double dt = (now() - t0) / layers;
                    if (dt < best) best = dt;
                }
                printf("%s thr=%d n=%d  %8.1f us/layer  %6.2f ms/token (%d layers)\n",
                       mode ? "scalar" : "neon  ", threads[ti], n, best * 1e6, best * layers * 1e3 / n, layers);
                printf("RESULT {\"bench\":\"lbench\",\"kernel\":\"%s\",\"threads\":%d,\"n\":%d,\"layers\":%d,"
                       "\"us_per_layer\":%.1f,\"ms_per_token_all_layers\":%.3f}\n",
                       mode ? "scalar" : "neon", threads[ti], n, layers, best * 1e6, best * layers * 1e3 / n);
            }
        }
        pf_pool_destroy(pool);
    }
    printf("%s\n", fails ? "FAILED" : "all exact");
    return fails ? 1 : 0;
}
