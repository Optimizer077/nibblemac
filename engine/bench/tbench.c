/* tbench: A/B bench of Q4_G32 / Q4K_G32 kernel variants (qvar.h, KERNELS.md Part 5).
 *
 * For every dtype x shape x batch n x thread count x variant: output compared with kern_q4 /
 * kern_q4k byte for byte (`exact`), then time per call, GOP/s (2 x rows x cols x n / time) and
 * cycles per block-vector (c/bv = time x 1.5 GHz x threads / (rows/4 x cols/32 x n); one
 * block-vector = 4 rows x 32 columns x 1 vector = 128 int8 MACs). Weights are cold (a >= 64 MB
 * arena of copies, as kbench) unless --hot. Activations are quantized once (kern_quant_act), as
 * pf_matmul does; variants run on the same pf_act, so only the matrix kernel is timed.
 *
 *   tbench [--dtypes q4,q4k] [--shapes kv,qkv,...] [--batch 1,4,7] [--threads 1,4]
 *          [--vars ref,lut,...] [--hot] [--time 0.08] [--reps 3] [--check] [--tag NAME]
 * Prints one `RESULT {json}` per configuration; exit 1 if any variant is not exact.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "kbench.h"
#include "kern.h"
#include "pq_format.h"
#include "pq_quant.h"
#include "qvar.h"

typedef struct { const char *name; int rows, cols; } shape_t;
static const shape_t SHAPES[] = {
    {"qkv", 6144, 1024}, {"z", 2048, 1024}, {"out", 1024, 2048}, {"q", 4096, 1024},
    {"kv", 512, 1024}, {"gate", 3584, 1024}, {"down", 1024, 3584}, {"head", 248320, 1024},
    {"c4qkv", 8192, 2560}, {"c4z", 4096, 2560}, {"c4out", 2560, 4096}, {"c4gate", 9216, 2560},
    {"c4down", 2560, 9216}, {"c4kv", 1024, 2560},
};
#define NSHAPES (int)(sizeof SHAPES / sizeof *SHAPES)
#define ARENA (64u << 20)

static struct {
    char dtypes[64], shapes[256], vars[256], tag[64];
    int batch[16], nbatch, threads[8], nthreads, hot, check, reps;
    double time;
} opt = {"q4,q4k", "kv,qkv,gate,down", "ref", "tbench", {1, 4, 7}, 3, {1, 4}, 2, 0, 0, 3, 0.08};

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double busy_seconds(void) { /* whole machine, from /proc/stat */
    unsigned long long u = 0, ni = 0, sy = 0, id = 0, io = 0, irq = 0, si = 0;
    FILE *f = fopen("/proc/stat", "r");
    if (f) {
        if (fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu", &u, &ni, &sy, &id, &io, &irq, &si) != 7)
            u = ni = sy = irq = si = 0;
        fclose(f);
    }
    return (double)(u + ni + sy + irq + si) / (double)sysconf(_SC_CLK_TCK);
}

static double self_seconds(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6;
}

static int in_list(const char *list, const char *name) {
    size_t n = strlen(name);
    for (const char *p = list; (p = strstr(p, name)); p += n)
        if ((p == list || p[-1] == ',') && (p[n] == ',' || p[n] == 0)) return 1;
    return 0;
}

static int parse_ints(const char *s, int *out, int max) {
    int k = 0;
    for (char *e; *s && k < max; s = *e ? e + 1 : e) out[k++] = (int)strtol(s, &e, 10);
    return k;
}

/* one call split over the pool: contiguous slabs of 4-row tiles, as matvec_fast.c's run_job */
typedef struct {
    qvar_fn fn;
    const pf_mat *W;
    const void *aux;
    const pf_act *A;
    float *Y;
    int n;
} call_t;

static void call_task(void *arg, int tid, int nthr) {
    const call_t *c = arg;
    const int units = c->W->rows / 4;
    const int u0 = (int)((long)units * tid / nthr), u1 = (int)((long)units * (tid + 1) / nthr);
    if (u0 < u1) c->fn(c->W, c->aux, c->A, c->Y, c->n, u0 * 4, u1 * 4);
}

static void run(pf_pool *pool, int thr, const qvar *v, const pf_mat *W, const void *aux,
                const pf_act *A, float *Y, int n) {
    call_t c = {v->fn, W, aux, A, Y, n};
    kern_q4_asm = v->q4asm;
    kern_pool_run(pool, call_task, &c, thr);
}

static int bench_shape(unsigned dtype, const char *dname, const shape_t *sh, pf_pool **pools) {
    const size_t bytes = pq_quant_nbytes(dtype, sh->rows, sh->cols);
    const int copies = opt.hot || opt.check ? 1 : (int)((ARENA + bytes - 1) / bytes);
    const size_t stride = (bytes + 4095) & ~(size_t)4095;
    uint8_t *arena = aligned_alloc(4096, stride * copies);
    if (!arena) { fprintf(stderr, "tbench: no memory\n"); return 1; }
    kb_fill_matrix(dtype, sh->rows, sh->cols, 42, arena);
    for (int c = 1; c < copies; c++) memcpy(arena + (size_t)c * stride, arena, bytes);
    pf_mat *Ws = malloc(sizeof(pf_mat) * copies);
    for (int c = 0; c < copies; c++) Ws[c] = (pf_mat){dtype, sh->rows, sh->cols, arena + (size_t)c * stride};

    int nmax = 1, fails = 0;
    for (int i = 0; i < opt.nbatch; i++) nmax = opt.batch[i] > nmax ? opt.batch[i] : nmax;
    const int cols = sh->cols, rows = sh->rows, nb = cols / 32;
    float *X = malloc((size_t)nmax * cols * 4), *Yr = malloc((size_t)nmax * rows * 4);
    float *Y = malloc((size_t)nmax * rows * 4);
    int8_t *q = aligned_alloc(64, ((size_t)nmax * cols + 63) & ~(size_t)63);
    float *d = malloc((size_t)nmax * nb * 4), *fs = malloc((size_t)nmax * nb * 4);
    kb_fill_vec(X, (size_t)nmax * cols, 7);

    for (int vi = 0; vi < NQVARS; vi++) {
        const qvar *v = &QVARS[vi];
        if (!in_list(opt.vars, v->name) || !(v->dtypes & (1u << dtype))) continue;
        void **aux = calloc(copies, sizeof(void *));
        size_t auxb = 0;
        if (v->prep)
            for (int c = 0; c < copies; c++) aux[c] = v->prep(&Ws[c], &auxb);
        for (int bi = 0; bi < opt.nbatch; bi++) {
            const int n = opt.batch[bi];
            kern_quant_act(X, n, cols, q, d, fs);
            const pf_act A = {cols, n, q, d, fs};
            run(pools[0], 1, &QVARS[0], &Ws[0], NULL, &A, Yr, n); /* QVARS[0] = ref */
            for (int ti = 0; ti < opt.nthreads; ti++) {
                const int thr = opt.threads[ti];
                memset(Y, 0xA5, (size_t)n * rows * 4);
                run(pools[ti], thr, v, &Ws[0], aux[0], &A, Y, n);
                const int exact = memcmp(Y, Yr, (size_t)n * rows * 4) == 0;
                fails += !exact;
                double best = 0, foreign = 0;
                long iters = 0;
                int noisy = 0;
                if (!opt.check) {
                    const double t1 = now();
                    run(pools[ti], thr, v, &Ws[0], aux[0], &A, Y, n);
                    const double one = now() - t1 + 1e-7;
                    iters = (long)(opt.time / one) + 1;
                    if (iters < copies && copies > 1) iters = copies;
                    best = 1e30, foreign = 1e30, noisy = 1;
                    double best_any = 1e30;
                    for (int rep = 0, clean = 0; rep < opt.reps + 4 && clean < opt.reps; rep++) {
                        const double b0 = busy_seconds(), s0 = self_seconds(), t0 = now();
                        for (long i = 0; i < iters; i++)
                            run(pools[ti], thr, v, &Ws[i % copies], aux[i % copies], &A, Y, n);
                        const double wall = now() - t0, per = wall / iters;
                        const double fr = (busy_seconds() - b0) - (self_seconds() - s0);
                        if (fr < foreign) foreign = fr;
                        if (per < best_any) best_any = per;
                        if (fr < 0.02 + 0.05 * wall) {
                            clean++;
                            if (per < best) best = per, noisy = 0;
                        }
                    }
                    if (noisy) best = best_any;
                }
                const double gops = best > 0 ? 2.0 * rows * cols * n / best / 1e9 : 0;
                const double cbv = best * 1.5e9 * thr / ((double)rows / 4 * cols / 32 * n);
                printf("%-8s %-4s %-6s %5dx%-5d n=%-2d thr=%d %10.1f us %6.2f GOP/s %6.2f c/bv %s%s\n",
                       v->name, dname, sh->name, rows, cols, n, thr, best * 1e6, gops, cbv,
                       exact ? "exact" : "MISMATCH", noisy ? " NOISY" : "");
                printf("RESULT {\"bench\":\"tbench\",\"tag\":\"%s\",\"var\":\"%s\",\"dtype\":\"%s\","
                       "\"shape\":\"%s\",\"rows\":%d,\"cols\":%d,\"n\":%d,\"threads\":%d,\"hot\":%d,"
                       "\"us\":%.2f,\"gops\":%.3f,\"cbv\":%.3f,\"iters\":%ld,\"aux_bytes\":%zu,"
                       "\"exact\":%s,\"noisy\":%s,\"foreign_cpu_s\":%.3f}\n",
                       opt.tag, v->name, dname, sh->name, rows, cols, n, thr, opt.hot, best * 1e6,
                       gops, cbv, iters, auxb, exact ? "true" : "false", noisy ? "true" : "false",
                       foreign);
                fflush(stdout);
            }
        }
        for (int c = 0; c < copies; c++) free(aux[c]);
        free(aux);
    }
    free(arena), free(Ws), free(X), free(Yr), free(Y), free(q), free(d), free(fs);
    return fails;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--dtypes")) snprintf(opt.dtypes, sizeof opt.dtypes, "%s", v), i++;
        else if (!strcmp(a, "--shapes")) snprintf(opt.shapes, sizeof opt.shapes, "%s", v), i++;
        else if (!strcmp(a, "--vars")) snprintf(opt.vars, sizeof opt.vars, "%s", v), i++;
        else if (!strcmp(a, "--tag")) snprintf(opt.tag, sizeof opt.tag, "%s", v), i++;
        else if (!strcmp(a, "--batch")) opt.nbatch = parse_ints(v, opt.batch, 16), i++;
        else if (!strcmp(a, "--threads")) opt.nthreads = parse_ints(v, opt.threads, 8), i++;
        else if (!strcmp(a, "--time")) opt.time = atof(v), i++;
        else if (!strcmp(a, "--reps")) opt.reps = atoi(v), i++;
        else if (!strcmp(a, "--hot")) opt.hot = 1;
        else if (!strcmp(a, "--check")) opt.check = 1;
        else if (!strcmp(a, "--list")) {
            for (int k = 0; k < NQVARS; k++) printf("%-8s %s\n", QVARS[k].name, QVARS[k].about);
            return 0;
        } else { fprintf(stderr, "unknown option %s\n", a); return 2; }
    }
    pf_pool *pools[8];
    for (int i = 0; i < opt.nthreads; i++) pools[i] = pf_pool_create(opt.threads[i]);
    int fails = 0;
    const struct { const char *name; unsigned id; } dts[] = {{"q4", PQ_Q4_G32}, {"q4k", PQ_Q4K_G32}};
    for (int di = 0; di < 2; di++) {
        if (!in_list(opt.dtypes, dts[di].name)) continue;
        for (int s = 0; s < NSHAPES; s++)
            if (in_list(opt.shapes, SHAPES[s].name))
                fails += bench_shape(dts[di].id, dts[di].name, &SHAPES[s], pools);
    }
    for (int i = 0; i < opt.nthreads; i++) pf_pool_destroy(pools[i]);
    printf("%s (%d mismatches)\n", fails ? "FAILED" : "all exact", fails);
    return fails ? 1 : 0;
}
