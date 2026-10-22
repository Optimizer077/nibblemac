/* kbench: microbenchmark + unit test of the pinfer matrix kernels (ops.h, MATVEC=fast).
 *
 * For every dtype x real Qwen3.5-0.8B shape (Qwen3.5-4B shapes c4*: on request) x thread
 * count x batch size it reports time per call, effective weight GB/s and the max error
 * against a scalar fp32 reference (both with the fp32 inputs, and with the int8-quantized
 * inputs the quantized kernels actually use).
 * Weights are cold: each call reads a different copy of the matrix from a >= 64 MB arena,
 * so like in decode every byte comes from DRAM (the L2 is 1 MB).
 *
 *   kbench [--dtypes f32,bf16,q8,q4,q4k] [--shapes qkv,z,...] [--threads 1,2,3,4]
 *          [--batch 1,2,4,8] [--time 0.15] [--reps 2] [--prefetch BYTES] [--variant NAME]
 *          [--golden FILE] [--check] [--head-f32] [--hot] [--auto] [--dirty KB] [--group N]
 *          [--gvecs N] [--gfused 0|1] [--qgroup N] [--gmode 0|1]
 * --dirty KB: write a KB buffer before every timed call (as the delta-rule recurrence
 * leaves ~1 MB of dirty state in L2 right before out_proj in the engine).
 * --group N / --gvecs N: pf_matmul splits n > N vectors into groups of at most gvecs
 * vectors for the quantized dtypes (kern_act_group / kern_group_vecs; --group 0: never);
 * --gfused 0: one job per group instead of all groups in one job (kern_group_fused);
 * --qgroup N: quantized calls of more than N vectors are split (kern_qgroup_min, 16 before
 * KERNELS.md Part 4); --gmode 0: equal group sizes instead of 7s + 5s / 4s (kern_group_mode).
 * Every RESULT carries `yhash`, an FNV-1a hash of the batched output bytes: the inputs are
 * seeded, so two kbench builds with different kernels must print the same hashes for the
 * same configuration if their outputs are bit-identical (KERNELS.md Part 3).
 * Prints one `RESULT {json}` line per configuration (for harness/pi.py bench).
 * Exit code 1 if any correctness check fails.
 */
#define _GNU_SOURCE
#include <math.h>
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

typedef struct { const char *name; int rows, cols; } shape_t;
static const shape_t SHAPES[] = {
    {"qkv", 6144, 1024}, {"z", 2048, 1024},   {"out", 1024, 2048},  {"q", 4096, 1024},
    {"kv", 512, 1024},   {"gate", 3584, 1024}, {"down", 1024, 3584}, {"ab", 16, 1024},
    {"head", 248320, 1024}, {"dhead", 32768, 1024}, /* draft-vocab head */
    /* Qwen3.5-4B (Cerno 4B, KERNELS.md Part 4): in_proj_qkv = gated q_proj, in_proj_z,
     * out_proj = o_proj, gate = up, down, k = v, in_proj_a / _b; selected by name only */
    {"c4qkv", 8192, 2560}, {"c4z", 4096, 2560}, {"c4out", 2560, 4096}, {"c4gate", 9216, 2560},
    {"c4down", 2560, 9216}, {"c4kv", 1024, 2560}, {"c4ab", 32, 2560},
};
typedef struct { const char *name; uint32_t id; } dtype_t;
static const dtype_t DTYPES[] = {{"f32", PQ_F32}, {"bf16", PQ_BF16}, {"q8", PQ_Q8_G32},
                                 {"q4", PQ_Q4_G32}, {"q4k", PQ_Q4K_G32}};
#define NSHAPES (int)(sizeof SHAPES / sizeof *SHAPES)
#define NDTYPES (int)(sizeof DTYPES / sizeof *DTYPES)
#define ARENA_BYTES (64u << 20)
#define MAX_REF_ROWS 4096

static struct {
    char dtypes[128], shapes[128], variant[64];
    int threads[8], nthreads, batch[8], nbatch, reps, check, head_f32, hot, autothr, dirty_kb;
    double time;
    const char *golden;
} opt = {"f32,bf16,q8,q4,q4k", "qkv,z,out,q,kv,gate,down,ab,head,dhead", "default",
         {1, 2, 3, 4}, 4, {1, 2, 4, 8}, 4, 2, 0, 0, 0, 0, 0, 0.15, NULL};

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* CPU seconds used by everything on the machine (from /proc/stat) and by this process.
 * Their difference over a timed run is "foreign" load from other users of the Pi. */
static double busy_seconds(void) {
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

typedef struct { double sec; int noisy; double foreign; } timing_t;

/* Best time per call over opt.reps clean repetitions. A repetition is "noisy" when other
 * processes used more than ~5% of a core during it; noisy ones are retried (up to 4 extra)
 * and only reported if no clean repetition was obtained. */
static timing_t time_calls(pf_pool *pool, uint32_t dtype, int rows, int cols, uint8_t *arena,
                           size_t stride, int copies, long it, const float *X, float *Y, int n) {
    timing_t t = {1e30, 1, 1e30};
    double best_any = 1e30;
    static uint8_t *dirty;
    if (opt.dirty_kb && !dirty) dirty = malloc((size_t)opt.dirty_kb << 10);
    int clean = 0;
    for (int attempt = 0; attempt < opt.reps + 4 && clean < opt.reps; attempt++) {
        const double b0 = busy_seconds(), s0 = self_seconds(), t0 = now();
        for (long i = 0; i < it; i++) {
            pf_mat W = {dtype, rows, cols, arena + (size_t)(i % copies) * stride};
            if (dirty) memset(dirty, (int)i, (size_t)opt.dirty_kb << 10);
            pf_matmul(pool, &W, X, Y, n);
        }
        const double wall = now() - t0;
        const double foreign = (busy_seconds() - b0) - (self_seconds() - s0);
        const double per = wall / it;
        if (foreign < t.foreign) t.foreign = foreign;
        if (per < best_any) best_any = per;
        if (foreign < 0.02 + 0.05 * wall) {
            clean++;
            if (per < t.sec) t.sec = per, t.noisy = 0;
        }
    }
    if (t.noisy) t.sec = best_any;
    return t;
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

static void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--dtypes")) snprintf(opt.dtypes, sizeof opt.dtypes, "%s", v), i++;
        else if (!strcmp(a, "--shapes")) snprintf(opt.shapes, sizeof opt.shapes, "%s", v), i++;
        else if (!strcmp(a, "--variant")) snprintf(opt.variant, sizeof opt.variant, "%s", v), i++;
        else if (!strcmp(a, "--threads")) opt.nthreads = parse_ints(v, opt.threads, 8), i++;
        else if (!strcmp(a, "--batch")) opt.nbatch = parse_ints(v, opt.batch, 8), i++;
        else if (!strcmp(a, "--time")) opt.time = atof(v), i++;
        else if (!strcmp(a, "--reps")) opt.reps = atoi(v), i++;
        else if (!strcmp(a, "--prefetch")) kern_prefetch = atoi(v), i++;
        else if (!strcmp(a, "--golden")) opt.golden = v, i++;
        else if (!strcmp(a, "--check")) opt.check = 1;
        else if (!strcmp(a, "--head-f32")) opt.head_f32 = 1;
        else if (!strcmp(a, "--hot")) opt.hot = 1; /* one copy: cache-resident if it fits */
        else if (!strcmp(a, "--auto")) opt.autothr = 1; /* library thread policy */
        else if (!strcmp(a, "--dirty")) opt.dirty_kb = atoi(v), i++; /* dirty L2 before calls */
        else if (!strcmp(a, "--group")) kern_act_group = atoi(v), i++; /* 0 = no grouping */
        else if (!strcmp(a, "--gvecs")) kern_group_vecs = atoi(v), i++; /* quantized group size */
        else if (!strcmp(a, "--gfused")) kern_group_fused = atoi(v), i++; /* groups in one job */
        else if (!strcmp(a, "--qgroup")) kern_qgroup_min = atoi(v), i++; /* quantized split above */
        else if (!strcmp(a, "--gmode")) kern_group_mode = atoi(v), i++; /* group sizes */
        else { fprintf(stderr, "unknown option %s\n", a); exit(2); }
    }
}

static unsigned long long fnv1a(const void *p, size_t n) {
    const unsigned char *b = p;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

static float max_rel(const float *y, const float *ref, int n, int nrow) {
    float emax = 0, ymax = 0;
    for (int i = 0; i < n * nrow; i++) {
        emax = fmaxf(emax, fabsf(y[i] - ref[i]));
        ymax = fmaxf(ymax, fabsf(ref[i]));
    }
    return ymax > 0 ? emax / ymax : emax;
}

/* Benchmark one dtype x shape over all thread counts and batch sizes. Returns #failures. */
static int bench_one(const dtype_t *dt, const shape_t *sh, pf_pool **pools) {
    const size_t bytes = kb_matrix_bytes(dt->id, sh->rows, sh->cols);
    if (!bytes) return 0;
    int copies = opt.check || opt.hot ? 1 : (int)((ARENA_BYTES + bytes - 1) / bytes);
    if (copies > 512) copies = 512;
    const size_t stride = (bytes + 4095) & ~(size_t)4095;
    uint8_t *arena = aligned_alloc(4096, stride * copies);
    if (!arena) { fprintf(stderr, "kbench: cannot allocate %zu MB\n", stride * copies >> 20); return 1; }
    kb_fill_matrix(dt->id, sh->rows, sh->cols, 42, arena);
    for (int c = 1; c < copies; c++) memcpy(arena + (size_t)c * stride, arena, bytes);

    int nmax = 1;
    for (int i = 0; i < opt.nbatch; i++) nmax = opt.batch[i] > nmax ? opt.batch[i] : nmax;
    const int cols = sh->cols, rows = sh->rows;
    float *X = malloc((size_t)nmax * cols * 4), *Y = malloc((size_t)nmax * rows * 4);
    float *Y1 = malloc((size_t)nmax * rows * 4);
    kb_fill_vec(X, (size_t)nmax * cols, 7);
    /* reference on a row sample (all rows for every real shape except the head) */
    const int nrow = rows < MAX_REF_ROWS ? rows : MAX_REF_ROWS;
    int *idx = malloc(nrow * sizeof(int));
    for (int k = 0; k < nrow; k++) idx[k] = (int)((long)k * rows / nrow);
    float *rfp = malloc((size_t)nmax * nrow * 4), *rq = malloc((size_t)nmax * nrow * 4);
    float *ys = malloc((size_t)nmax * nrow * 4);
    pf_mat W0 = {dt->id, rows, cols, arena};
    kb_reference(&W0, X, nmax, idx, nrow, rfp, rq);
    const int quant = dt->id >= PQ_Q8_G32;
    int fails = 0;

    for (int ti = 0; ti < opt.nthreads; ti++) {
        pf_pool *pool = pools[ti];
        for (int bi = 0; bi < opt.nbatch; bi++) {
            const int n = opt.batch[bi];
            pf_matmul(pool, &W0, X, Y, n);
            const unsigned long long yhash = fnv1a(Y, (size_t)n * rows * sizeof(float));
            for (int v = 0; v < n; v++)
                for (int k = 0; k < nrow; k++) ys[v * nrow + k] = Y[(size_t)v * rows + idx[k]];
            float e_fp = max_rel(ys, rfp, n, nrow), e_k = quant ? max_rel(ys, rq, n, nrow) : e_fp;
            /* bit-exactness: row v of the batched call == pf_matvec on vector v (speculative
             * verification compares batched logits with plain-decode logits) */
            for (int v = 0; v < n; v++)
                pf_matvec(pool, &W0, X + (size_t)v * cols, Y1 + (size_t)v * rows);
            const int exact = memcmp(Y, Y1, (size_t)n * rows * sizeof(float)) == 0;
            const int ok = e_k < 1e-4f && e_fp < (quant ? 5e-2f : 1e-4f) && exact;
            fails += !ok;
            timing_t tm = {0, 0, 0};
            long iters = 0;
            if (!opt.check) {
                iters = (long)(opt.time * 4e9 / (double)bytes) + 1;
                if (iters < copies && copies > 1) iters = copies;
                for (int i = 0; i < copies && i < 8; i++) { /* warm-up */
                    pf_mat W = {dt->id, rows, cols, arena + (size_t)i * stride};
                    pf_matmul(pool, &W, X, Y, n);
                }
                tm = time_calls(pool, dt->id, rows, cols, arena, stride, copies, iters, X, Y, n);
            }
            const double best = tm.sec;
            const double us = best * 1e6, gbs = best > 0 ? bytes / best / 1e9 : 0;
            printf("%-5s %-5s %6dx%-5d thr=%d n=%d  %10.1f us  %6.2f GB/s  err fp %.2e kern %.2e %s%s\n",
                   dt->name, sh->name, rows, cols, pf_pool_threads(pool), n, us, gbs, e_fp, e_k,
                   ok ? "" : "FAIL", tm.noisy ? " NOISY" : "");
            printf("RESULT {\"bench\":\"kbench\",\"variant\":\"%s\",\"dtype\":\"%s\",\"shape\":\"%s\","
                   "\"rows\":%d,\"cols\":%d,\"threads\":%d,\"n\":%d,\"bytes\":%zu,\"us\":%.2f,"
                   "\"gbs\":%.3f,\"rel_err_fp\":%.3e,\"rel_err_kernel\":%.3e,\"prefetch\":%d,"
                   "\"iters\":%ld,\"hot\":%d,\"auto_threads\":%d,\"dirty_kb\":%d,\"group\":%d,\"gvecs\":%d,"
                   "\"gfused\":%d,\"qgroup\":%d,\"gmode\":%d,"
                   "\"exact\":%s,\"yhash\":\"%016llx\",\"noisy\":%s,\"foreign_cpu_s\":%.3f,\"ok\":%s}\n",
                   opt.variant, dt->name, sh->name, rows, cols, pf_pool_threads(pool), n, bytes,
                   us, gbs, e_fp, e_k, kern_prefetch, iters, opt.hot, opt.autothr, opt.dirty_kb,
                   kern_act_group, kern_group_vecs, kern_group_fused, kern_qgroup_min,
                   kern_group_mode, exact ? "true" : "false", yhash,
                   tm.noisy ? "true" : "false", tm.foreign, ok ? "true" : "false");
            fflush(stdout);
        }
    }
    free(arena), free(X), free(Y), free(Y1), free(idx), free(rfp), free(rq), free(ys);
    return fails;
}

int main(int argc, char **argv) {
    parse_args(argc, argv);
    kern_auto_threads = opt.autothr; /* sweeps measure exactly the requested thread count */
    pf_pool *pools[8];
    for (int i = 0; i < opt.nthreads; i++) pools[i] = pf_pool_create(opt.threads[i]);
    int fails = 0;
    for (int i = 0; i < opt.nthreads && !opt.check; i++) { /* pool dispatch overhead */
        for (int k = 0; k < 1000; k++) kern_pool_noop(pools[i]);
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            const double t0 = now();
            for (int k = 0; k < 20000; k++) kern_pool_noop(pools[i]);
            const double dt = (now() - t0) / 20000;
            if (dt < best) best = dt;
        }
        printf("dispatch thr=%d  %.0f ns per empty call\n", pf_pool_threads(pools[i]), best * 1e9);
        printf("RESULT {\"bench\":\"kbench_dispatch\",\"variant\":\"%s\",\"threads\":%d,\"ns\":%.1f}\n",
               opt.variant, pf_pool_threads(pools[i]), best * 1e9);
    }
    if (opt.golden) {
        for (int i = 0; i < opt.nthreads; i++) fails += kb_golden(opt.golden, pools[i]);
        printf("RESULT {\"bench\":\"kbench_golden\",\"variant\":\"%s\",\"pass\":%s}\n",
               opt.variant, fails ? "false" : "true");
    }
    for (int d = 0; d < NDTYPES; d++) {
        if (!in_list(opt.dtypes, DTYPES[d].name)) continue;
        for (int s = 0; s < NSHAPES; s++) {
            if (!in_list(opt.shapes, SHAPES[s].name)) continue;
            if (DTYPES[d].id == PQ_F32 && !strcmp(SHAPES[s].name, "head") && !opt.head_f32)
                continue; /* 1 GB: only on request */
            fails += bench_one(&DTYPES[d], &SHAPES[s], pools);
        }
    }
    for (int i = 0; i < opt.nthreads; i++) pf_pool_destroy(pools[i]);
    printf("%s (%d failing checks)\n", fails ? "FAILED" : "all checks passed", fails);
    return fails ? 1 : 0;
}
