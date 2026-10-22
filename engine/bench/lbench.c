/* lbench: the Gated DeltaNet recurrence kernel (kern_lin.c) vs the scalar code of
 * matvec_ref.c: bit-exactness (state, outputs, rollback log, commit replay, deferred commit)
 * and time per layer for n = 1 / 4 / 64 tokens with 1..4 threads, cold state (18 layers
 * rotate through an 18 MB arena, as in decode) and hot state. Prints RESULT lines for
 * harness/pi.py bench.
 *   lbench [--threads 1,2,4] [--n 1,4,64] [--layers 18] [--check]
 * Timed kinds: "inplace" = plain / prefill pass; "verify" = logged pass that also applies the
 * n-1 updates the previous round deferred (the speculative round since VERIFY.md);
 * "verify0+apply" = the same work as a logged pass plus a separate commit replay (the
 * pre-VERIFY.md round, minus its scratch copy of the state).
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
#define NMAX 64                          /* tokens per pass; inputs hold two passes */

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- scalar reference: verbatim from engine/src/matvec_ref.c ---- */
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

/* n tokens directly on S (row-major), with every token's log row when log != NULL. */
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

/* Inputs of one pass: rows `row0..` of the shared random qkv / decay / beta. */
typedef struct {
    const float *qkv, *decay, *beta;
} pass_in;

static pass_in pass_at(const float *qkv, const float *decay, const float *beta, int row0) {
    pass_in p = {qkv + (size_t)row0 * C, decay + row0 * HEADS, beta + row0 * HEADS};
    return p;
}

static kern_lin_args kargs(int n, pass_in p, float *S, float *o, float *log, const float *pend,
                           int keep) {
    kern_lin_args a = {n, HEADS, REP, DK, DV, C, HEADS * DV, p.qkv, p.decay, p.beta, S, o, log,
                       pend, keep, NULL};
    return a;
}

static int same(const void *x, const void *y, size_t floats) {
    return !memcmp(x, y, floats * sizeof(float));
}

/* Bit-exactness of every kernel mode against the reference for one n (>= 1):
 *   A  logged pass: S = state after token 0, outputs of all tokens, log rows 1..n-1;
 *   R  commit replay of rows 1..e-1 (e = min(n, 3)) = direct run of tokens 0..e-1;
 *   B  logged pass with that replay deferred into it (pend / keep), on a second input;
 *   P  in-place pass with the replay deferred into it;  I  in-place pass alone. */
static int check(pf_pool *pool, const float *S0, const float *qkv, const float *decay,
                 const float *beta, int n) {
    const size_t hs = (size_t)HEADS * DK * DV, on = (size_t)n * HEADS * DV, ln = (size_t)n * HEADS * LOGW;
    const int e = n < 3 ? n : 3;
    const pass_in pa = pass_at(qkv, decay, beta, 0), pb = pass_at(qkv, decay, beta, NMAX);
    float *Sr = malloc(hs * 4), *Se = malloc(hs * 4), *Sk = malloc(hs * 4), *Sx = malloc(hs * 4);
    float *St0 = malloc(hs * 4);
    float *oref = malloc(on * 4), *oker = malloc(on * 4), *lr = malloc(ln * 4), *lk = malloc(ln * 4);
    float *lr2 = malloc(ln * 4), *lk2 = malloc(ln * 4);
    int okA, okR, okB, okP, okI;

    /* A */
    memcpy(St0, S0, hs * 4);
    ref_layer(St0, pa.qkv, pa.decay, pa.beta, oref, NULL, 1);   /* state after token 0 */
    memcpy(Sr, S0, hs * 4);
    ref_layer(Sr, pa.qkv, pa.decay, pa.beta, oref, lr, n);      /* outputs + log of all tokens */
    to_chunked(S0, Sk);
    kern_lin_args a = kargs(n, pa, Sk, oker, lk, NULL, 0);
    kern_lin_rule(pool, &a);
    from_chunked(Sk, Sx);
    okA = same(Sx, St0, hs) && same(oker, oref, on) &&
          (n < 2 || same(lk + HEADS * LOGW, lr + HEADS * LOGW, ln - HEADS * LOGW));

    /* R: replay rows 1..e-1 of the kernel log onto the state after token 0 */
    memcpy(Se, S0, hs * 4);
    ref_layer(Se, pa.qkv, pa.decay, pa.beta, oref, NULL, e);    /* direct run of e tokens */
    kern_lin_apply(pool, Sk, lk + HEADS * LOGW, e - 1, HEADS, DK, DV);
    from_chunked(Sk, Sx);
    okR = same(Sx, Se, hs);

    /* B: from the state after token 0 of A, a logged pass on input B with A's e-1 updates
     * deferred into it; reference = e tokens of A then B */
    memcpy(Sr, Se, hs * 4);
    ref_layer(Sr, pb.qkv, pb.decay, pb.beta, oref, NULL, 1);
    memcpy(Sx, Se, hs * 4);
    ref_layer(Sx, pb.qkv, pb.decay, pb.beta, oref, lr2, n);
    to_chunked(St0, Sk);
    a = kargs(n, pb, Sk, oker, lk2, lk + HEADS * LOGW, e - 1);
    kern_lin_rule(pool, &a);
    from_chunked(Sk, Sx);
    okB = same(Sx, Sr, hs) && same(oker, oref, on) &&
          (n < 2 || same(lk2 + HEADS * LOGW, lr2 + HEADS * LOGW, ln - HEADS * LOGW));

    /* P: the same deferred updates into an in-place pass on B */
    memcpy(Sr, Se, hs * 4);
    ref_layer(Sr, pb.qkv, pb.decay, pb.beta, oref, NULL, n);
    to_chunked(St0, Sk);
    a = kargs(n, pb, Sk, oker, NULL, lk + HEADS * LOGW, e - 1);
    kern_lin_rule(pool, &a);
    from_chunked(Sk, Sx);
    okP = same(Sx, Sr, hs) && same(oker, oref, on);

    /* I: in place, nothing deferred */
    memcpy(Sr, S0, hs * 4);
    ref_layer(Sr, pa.qkv, pa.decay, pa.beta, oref, NULL, n);
    to_chunked(S0, Sk);
    a = kargs(n, pa, Sk, oker, NULL, NULL, 0);
    kern_lin_rule(pool, &a);
    from_chunked(Sk, Sx);
    okI = same(Sx, Sr, hs) && same(oker, oref, on);

    const int ok = okA && okR && okB && okP && okI;
    printf("exact n=%d: logged %s, replay %s, logged+deferred %s, inplace+deferred %s, "
           "inplace %s\n", n, okA ? "ok" : "FAIL", okR ? "ok" : "FAIL", okB ? "ok" : "FAIL",
           okP ? "ok" : "FAIL", okI ? "ok" : "FAIL");
    printf("RESULT {\"bench\":\"lbench_exact\",\"n\":%d,\"logged\":%s,\"replay\":%s,"
           "\"logged_deferred\":%s,\"inplace_deferred\":%s,\"inplace\":%s,\"pass\":%s}\n", n,
           okA ? "true" : "false", okR ? "true" : "false", okB ? "true" : "false",
           okP ? "true" : "false", okI ? "true" : "false", ok ? "true" : "false");
    free(Sr), free(Se), free(Sk), free(Sx), free(St0), free(oref), free(oker), free(lr), free(lk);
    free(lr2), free(lk2);
    return ok;
}

int main(int argc, char **argv) {
    int threads[8] = {1, 2, 4}, nthr = 3, ns[8] = {1, 4, 64}, nn = 3, layers = 18, chk = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        int *dst = !strcmp(a, "--threads") ? threads : !strcmp(a, "--n") ? ns : NULL;
        if (dst) {
            int k = 0;
            for (char *e; *v && k < 8; v = *e ? e + 1 : e) dst[k++] = (int)strtol(v, &e, 10);
            *(dst == threads ? &nthr : &nn) = k;
            i++;
        } else if (!strcmp(a, "--layers")) layers = atoi(v), i++;
        else if (!strcmp(a, "--check")) chk = 1;
    }
    for (int i = 0; i < nn; i++)
        if (ns[i] < 1 || ns[i] > NMAX) return fprintf(stderr, "n must be 1..%d\n", NMAX), 2;
    const size_t hs = (size_t)HEADS * DK * DV;
    float *S0 = malloc(hs * sizeof(float));
    float *arena = malloc((size_t)layers * hs * sizeof(float));
    float *qkv = malloc((size_t)2 * NMAX * C * sizeof(float));
    float *decay = malloc(2 * NMAX * HEADS * sizeof(float)), *beta = malloc(2 * NMAX * HEADS * sizeof(float));
    float *ob = malloc((size_t)NMAX * HEADS * DV * sizeof(float));
    float *lb = malloc((size_t)NMAX * HEADS * LOGW * sizeof(float));
    float *lw = malloc((size_t)NMAX * HEADS * LOGW * sizeof(float)); /* log written when timed */
    uint64_t seed = 12345;
    for (size_t i = 0; i < hs; i++) S0[i] = frand(&seed) * 0.1f;
    for (size_t i = 0; i < (size_t)2 * NMAX * C; i++) qkv[i] = frand(&seed);
    for (int i = 0; i < 2 * NMAX * HEADS; i++)
        decay[i] = 0.9f + 0.1f * (frand(&seed) + 0.5f), beta[i] = frand(&seed) + 0.5f;
    for (int l = 0; l < layers; l++) memcpy(arena + (size_t)l * hs, S0, hs * sizeof(float));

    int fails = 0;
    pf_pool *pool = pf_pool_create(4);
    for (int ni = 0; ni < nn; ni++) fails += !check(pool, S0, qkv, decay, beta, ns[ni]);
    { /* a valid log for the timed deferred replays: rows 1.. of a logged pass */
        kern_lin_args a = kargs(NMAX, pass_at(qkv, decay, beta, 0), arena, ob, lb, NULL, 0);
        kern_lin_rule(pool, &a);
        memcpy(arena, S0, hs * sizeof(float));
    }
    pf_pool_destroy(pool);
    if (chk) return fails ? 1 : 0;

    const pass_in pa = pass_at(qkv, decay, beta, 0);
    for (int ti = 0; ti < nthr; ti++) {
        pool = pf_pool_create(threads[ti]);
        for (int ni = 0; ni < nn; ni++) {
            const int n = ns[ni];
            /* 0 = NEON in place, 1 = scalar in place, 2 = NEON verify (n-1 deferred),
             * 3 = NEON logged + separate replay of n-1 */
            for (int mode = 0; mode < 4; mode++) {
                if (mode == 1 && threads[ti] != 1) continue;
                if (mode >= 2 && n < 2) continue;
                double best = 1e30;
                for (int rep = 0; rep < 3; rep++) {
                    const double t0 = now();
                    for (int l = 0; l < layers; l++) {
                        float *S = arena + (size_t)l * hs;
                        kern_lin_args a = kargs(n, pa, S, ob, mode >= 2 ? lw : NULL,
                                                mode == 2 ? lb + HEADS * LOGW : NULL,
                                                mode == 2 ? n - 1 : 0);
                        if (mode == 1) ref_layer(S, qkv, decay, beta, ob, NULL, n);
                        else kern_lin_rule(pool, &a);
                        if (mode == 3) kern_lin_apply(pool, S, lw + HEADS * LOGW, n - 1, HEADS, DK, DV);
                    }
                    const double dt = (now() - t0) / layers;
                    if (dt < best) best = dt;
                }
                static const char *const kind[] = {"inplace", "inplace", "verify", "verify0+apply"};
                const char *kern = mode == 1 ? "scalar" : "neon";
                printf("%-6s %-13s thr=%d n=%-2d %8.1f us/layer  %6.2f ms/token  %6.2f ms/pass "
                       "(%d layers)\n", kern, kind[mode], threads[ti], n, best * 1e6,
                       best * layers * 1e3 / n, best * layers * 1e3, layers);
                printf("RESULT {\"bench\":\"lbench\",\"kernel\":\"%s\",\"kind\":\"%s\",\"threads\":%d,"
                       "\"n\":%d,\"layers\":%d,\"us_per_layer\":%.1f,\"ms_per_token_all_layers\":%.3f,"
                       "\"ms_per_pass_all_layers\":%.3f}\n", kern, kind[mode], threads[ti], n, layers,
                       best * 1e6, best * layers * 1e3 / n, best * layers * 1e3);
            }
        }
        pf_pool_destroy(pool);
    }
    printf("%s\n", fails ? "FAILED" : "all exact");
    return fails ? 1 : 0;
}
