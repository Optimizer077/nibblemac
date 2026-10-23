/* Q4K pilot: real model tensors, independent sampled QPU reference, CPU/GPU co-run.
 * Usage: q4k_pilot fixture.bin 7|64 [seconds_per_sample [repetitions]]
 * Timings include uniform construction, submissions/wait and GPU output readback.
 * CPU is the shipping four-thread pf_matmul; GPU alone and simultaneous full-shape
 * work measure additional throughput, not the latency of an integrated row split.
 */
#define _GNU_SOURCE
#include "v3d.h"
#include "q4kgpu_kernels.h"
#include "kern.h"
#include "pq_quant.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { CODE, CODES, SCALES, STREAM, OUTPUT, PARAM, NBO };
typedef struct {
    int rows, cols, n, blocks, passes, refs;
    uint8_t *raw;
    float *x, *cpu, *gpu, *reference, *d, *fsum;
    int8_t *q;
    uint32_t *index;
    pf_mat w;
    pf_act act;
    pf_pool *pool;
    struct v3d_device dev;
    struct v3d_bo bo[NBO];
    const struct q4kgpu_program *program;
    uint32_t handles[NBO];
    size_t stream_stride, ybytes;
} test;

static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock"); exit(1); }
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void *alloc(size_t n) {
    void *p = malloc(n);
    if (!p) { perror("malloc"); exit(1); }
    return p;
}
static void read_bytes(FILE *f, void *p, size_t n) {
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "truncated fixture\n"); exit(1); }
}
static uint32_t bits(float f) { uint32_t v; memcpy(&v, &f, 4); return v; }
static uint64_t hash(const void *p, size_t n) {
    const uint8_t *b = p;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * UINT64_C(1099511628211);
    return h;
}
static size_t align128(size_t n) { return (n + 127) & ~(size_t)127; }
static int sumq(const pf_act *a, int b, int v) {
    int s = 0;
    for (int c = 0; c < 32; c++) s += a->q[((size_t)b * a->n + v) * 32 + c];
    return s;
}
static uint32_t word(struct q4kgpu_tag tag, const pf_act *a, int b, int v0) {
    int v = v0 + tag.vec;
    switch (tag.kind) {
    case Q4KGPU_X: return (uint32_t)(int32_t)a->q[((size_t)b * a->n + v) * 32 + tag.col];
    case Q4KGPU_DX: return bits(a->d[(size_t)b * a->n + v]);
    case Q4KGPU_FSUM: return bits(a->fsum[(size_t)b * a->n + v]);
    case Q4KGPU_SUM8: return (uint32_t)(sumq(a, b, v) * 8);
    case Q4KGPU_NEGC: return (uint32_t)(-INT64_C(524296) * sumq(a, 0, v));
    case Q4KGPU_NEGC_NEXT: return b + 1 < a->cols / 32 ?
        (uint32_t)(-INT64_C(524296) * sumq(a, b + 1, v)) : 0;
    default: fprintf(stderr, "bad uniform tag\n"); exit(1);
    }
}
static const uint8_t *superblock(test *t, int row, int b) {
    return t->raw + ((size_t)(row / 4) * (t->cols / 256) + b / 8) * 576;
}
static void pack(test *t) {
    uint32_t *codes = t->bo[CODES].map;
    float *scales = t->bo[SCALES].map;
    for (int tile = 0; tile < t->rows / 32; tile++)
        for (int b = 0; b < t->blocks; b++) for (int lane = 0; lane < 16; lane++) {
            int a = tile * 32 + lane, h = a + 16;
            const uint8_t *sa = superblock(t, a, b), *sb = superblock(t, h, b);
            const __fp16 *ha = (const __fp16 *)sa, *hb = (const __fp16 *)sb;
            size_t pos = ((size_t)tile * t->blocks + b) * 16 + lane;
            scales[pos * 4] = (float)ha[a % 4] * kern_get6(sa + 16, b % 8 * 4 + a % 4);
            scales[pos * 4 + 1] = (float)hb[h % 4] * kern_get6(sb + 16, b % 8 * 4 + h % 4);
            scales[pos * 4 + 2] = (float)ha[4 + a % 4] * kern_get6(sa + 40, b % 8 * 4 + a % 4);
            scales[pos * 4 + 3] = (float)hb[4 + h % 4] * kern_get6(sb + 40, b % 8 * 4 + h % 4);
            const uint8_t *qa = sa + 64 + b % 8 * 64 + a % 4 * 16;
            const uint8_t *qb = sb + 64 + b % 8 * 64 + h % 4 * 16;
            for (int k = 0; k < 8; k++) {
                uint32_t w = 0;
                for (int j = 0; j < 4; j++) {
                    int c = k * 4 + j, shift = c >= 16 ? 4 : 0;
                    w |= (uint32_t)((qa[c % 16] >> shift) & 15) << (j * 4);
                    w |= (uint32_t)((qb[c % 16] >> shift) & 15) << (16 + j * 4);
                }
                codes[pos * 8 + k] = w;
            }
        }
}
static void prepare(test *t) {
    const struct q4kgpu_program *k = t->program;
    for (int pass = 0; pass < t->passes; pass++) {
        int v0 = pass * (int)k->n;
        uint32_t *s = (uint32_t *)((uint8_t *)t->bo[STREAM].map + pass * t->stream_stride);
        size_t pos = 0;
        for (size_t i = 0; i < k->start_words; i++) s[pos++] = word(k->start[i], &t->act, 0, v0);
        for (int b = 0; b < t->blocks; b++)
            for (size_t i = 0; i < k->block_words; i++) s[pos++] = word(k->block[i], &t->act, b, v0);
        uint32_t *p = (uint32_t *)((uint8_t *)t->bo[PARAM].map + pass * 128);
        uint32_t nt = (uint32_t)t->rows / 256, nb = (uint32_t)t->blocks;
        p[Q4KGPU_CODES] = t->bo[CODES].gpu; p[Q4KGPU_SCALES] = t->bo[SCALES].gpu;
        p[Q4KGPU_Y] = t->bo[OUTPUT].gpu + (uint32_t)((size_t)v0 * t->rows * 4);
        p[Q4KGPU_ACT] = t->bo[STREAM].gpu + (uint32_t)(pass * t->stream_stride);
        p[Q4KGPU_NT] = nt; p[Q4KGPU_NB] = nb;
        p[Q4KGPU_TSC] = nt * nb * 512; p[Q4KGPU_TSS] = nt * nb * 256;
        p[Q4KGPU_TSY] = nt * 128; p[Q4KGPU_R4] = (uint32_t)t->rows * 4;
        p[Q4KGPU_MASK] = 0x000f000f; p[Q4KGPU_K65536] = 65536;
        p[Q4KGPU_K512] = 512; p[Q4KGPU_K64] = 64; p[Q4KGPU_K256] = 256;
    }
}
static void gpu_call(test *t) {
    prepare(t);
    for (int pass = 0; pass < t->passes; pass++)
        if (v3d_submit_8qpu(&t->dev, t->bo[CODE].gpu, t->bo[PARAM].gpu + (uint32_t)(pass * 128), t->handles, NBO)) {
            perror("submit"); exit(1);
        }
    if (v3d_wait(&t->dev, &t->bo[OUTPUT], UINT64_C(3000000000))) { perror("wait"); exit(1); }
    memcpy(t->gpu, t->bo[OUTPUT].map, t->ybytes);
}
static void cpu_call(test *t) { pf_matmul(t->pool, &t->w, t->x, t->cpu, t->n); }
static double run(test *t, bool gpu, double seconds) {
    double start = now(), end;
    long count = 0;
    do { if (gpu) gpu_call(t); else cpu_call(t); count++; end = now(); }
    while (count < 3 || end - start < seconds);
    return (end - start) / count;
}
typedef struct {
    test *t;
    pthread_barrier_t barrier;
    _Atomic int stop;
    long count;
    double start, end;
} corun;
static void *cpu_loop(void *arg) {
    corun *c = arg;
    pthread_barrier_wait(&c->barrier);
    do { cpu_call(c->t); c->count++; } while (!atomic_load_explicit(&c->stop, memory_order_acquire));
    c->end = now();
    return NULL;
}
static void concurrent(test *t, double seconds, double *cpu_sec, double *gpu_sec, double *window_ratio) {
    corun c = {.t = t};
    pthread_t thread;
    pthread_barrier_init(&c.barrier, NULL, 2);
    if (pthread_create(&thread, NULL, cpu_loop, &c)) { perror("pthread_create"); exit(1); }
    c.start = now();
    pthread_barrier_wait(&c.barrier);
    long count = 0;
    double end;
    do { gpu_call(t); count++; end = now(); } while (count < 5 || end - c.start < seconds);
    atomic_store_explicit(&c.stop, 1, memory_order_release);
    pthread_join(thread, NULL);
    pthread_barrier_destroy(&c.barrier);
    *cpu_sec = (c.end - c.start) / c.count;
    *gpu_sec = (end - c.start) / count;
    *window_ratio = (c.count + count) / (c.end - c.start);
}
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *a, int n) {
    qsort(a, (size_t)n, sizeof(*a), compare_double);
    return n % 2 ? a[n / 2] : .5 * (a[n / 2 - 1] + a[n / 2]);
}
int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) { fprintf(stderr, "usage: %s fixture 7|64 [seconds [reps]]\n", argv[0]); return 2; }
    test t = {.n = atoi(argv[2]), .dev = {.fd = -1}};
    double seconds = argc > 3 ? atof(argv[3]) : .35;
    int reps = argc > 4 ? atoi(argv[4]) : 3;
    if ((t.n != 7 && t.n != 64) || seconds < .01 || seconds > 3 || reps < 1 || reps > 7) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fixture"); return 1; }
    uint32_t h[8]; read_bytes(f, h, sizeof(h));
    t.rows = (int)h[2]; t.cols = (int)h[3]; t.refs = (int)h[5]; t.blocks = t.cols / 32;
    size_t wbytes = (size_t)(t.rows / 4) * (t.cols / 256) * 576;
    if (h[0] != 0x4b345150 || h[1] != 1 || h[4] != 64 || t.refs != 128 || h[6] != wbytes || h[7] ||
        t.rows < 256 || t.rows % 256 || t.cols < 256 || t.cols % 256 ||
        (uint64_t)(t.rows / 256) * t.blocks * 512 >= (1u << 24)) return 2;
    t.raw = alloc(wbytes); read_bytes(f, t.raw, wbytes);
    t.x = alloc((size_t)64 * t.cols * 4); read_bytes(f, t.x, (size_t)64 * t.cols * 4);
    int8_t *qref = alloc((size_t)64 * t.cols); read_bytes(f, qref, (size_t)64 * t.cols);
    float *dref = alloc((size_t)64 * t.blocks * 4); read_bytes(f, dref, (size_t)64 * t.blocks * 4);
    t.index = alloc((size_t)t.refs * 4); read_bytes(f, t.index, (size_t)t.refs * 4);
    if (t.n == 64 && fseek(f, 7 * t.refs * 4, SEEK_CUR)) return 1;
    t.reference = alloc((size_t)t.n * t.refs * 4); read_bytes(f, t.reference, (size_t)t.n * t.refs * 4);
    fclose(f);
    for (int i = 0; i < t.refs; i++) if (t.index[i] >= (uint32_t)t.rows) return 2;
    t.ybytes = (size_t)t.n * t.rows * 4;
    t.cpu = alloc(t.ybytes); t.gpu = alloc(t.ybytes);
    t.q = alloc((size_t)t.n * t.cols); t.d = alloc((size_t)t.n * t.blocks * 4); t.fsum = alloc((size_t)t.n * t.blocks * 4);
    kern_quant_act(t.x, t.n, t.cols, t.q, t.d, t.fsum);
    for (int b = 0; b < t.blocks; b++) for (int v = 0; v < t.n; v++) {
        if (memcmp(t.q + ((size_t)b * t.n + v) * 32, qref + (size_t)v * t.cols + b * 32, 32) ||
            bits(t.d[(size_t)b * t.n + v]) != bits(dref[(size_t)v * t.blocks + b])) {
            fprintf(stderr, "activation mismatch\n"); return 1;
        }
    }
    free(qref); free(dref);
    t.w = (pf_mat){PQ_Q4K_G32, t.rows, t.cols, t.raw};
    t.act = (pf_act){t.cols, t.n, t.q, t.d, t.fsum};
    kern_auto_threads = 0;
    t.pool = pf_pool_create(4);
    if (!t.pool) return 1;
    cpu_call(&t);
    uint64_t cpu_hash = hash(t.cpu, t.ybytes);
    t.program = q4kgpu_programs[t.n == 7 ? 7 : 8];
    t.passes = t.n / (int)t.program->n;
    t.stream_stride = align128((t.program->start_words + t.blocks * t.program->block_words) * 4);
    size_t sizes[NBO] = {t.program->words * 8,
        (size_t)(t.rows / 32) * t.blocks * 512 + 1536,
        (size_t)(t.rows / 32) * t.blocks * 256 + 768,
        t.stream_stride * t.passes, t.ybytes, (size_t)t.passes * 128};
    if (v3d_open(&t.dev, NULL)) { perror("v3d_open"); return 1; }
    for (int i = 0; i < NBO; i++) {
        if (v3d_bo_create(&t.dev, &t.bo[i], sizes[i])) { perror("bo"); return 1; }
        memset(t.bo[i].map, 0, t.bo[i].size); t.handles[i] = t.bo[i].handle;
    }
    memcpy(t.bo[CODE].map, t.program->code, t.program->words * 8);
    pack(&t);
    for (size_t i = 0; i < t.ybytes / 4; i++) ((float *)t.bo[OUTPUT].map)[i] = NAN;
    gpu_call(&t);
    size_t mismatches = 0;
    float max_cpu_diff = 0, max_reference = 0;
    for (size_t i = 0; i < t.ybytes / 4; i++) if (!isfinite(t.gpu[i]) || !isfinite(t.cpu[i])) return 1;
    for (int v = 0; v < t.n; v++) for (int i = 0; i < t.refs; i++) {
        float expected = t.reference[v * t.refs + i];
        size_t pos = (size_t)v * t.rows + t.index[i];
        mismatches += bits(t.gpu[pos]) != bits(expected);
        max_cpu_diff = fmaxf(max_cpu_diff, fabsf(t.cpu[pos] - expected));
        max_reference = fmaxf(max_reference, fabsf(expected));
    }
    if (mismatches || max_cpu_diff > fmaxf(1e-5f, max_reference * 1e-4f)) {
        fprintf(stderr, "reference mismatch: %zu, CPU relative %.8g\n", mismatches, max_cpu_diff / max_reference); return 1;
    }
    uint64_t gpu_hash = hash(t.gpu, t.ybytes);
    double ratios[7], windows[7];
    for (int rep = 0; rep < reps; rep++) {
        double before = run(&t, false, seconds);
        struct timespec pause = {0, 80000000}; nanosleep(&pause, NULL);
        double gpu = run(&t, true, seconds), cpu_co, gpu_co, total_calls_s;
        concurrent(&t, seconds, &cpu_co, &gpu_co, &total_calls_s);
        double after = run(&t, false, seconds), cpu = .5 * (before + after);
        if (hash(t.cpu, t.ybytes) != cpu_hash || hash(t.gpu, t.ybytes) != gpu_hash) {
            fprintf(stderr, "post-timing output differs\n"); return 1;
        }
        double ops = 2.0 * t.rows * t.cols * t.n;
        ratios[rep] = cpu * (1.0 / cpu_co + 1.0 / gpu_co);
        windows[rep] = cpu * total_calls_s;
        printf("RESULT {\"bench\":\"q4k_pilot\",\"fixture\":\"%s\",\"rows\":%d,\"cols\":%d,\"n\":%d,\"round\":%d,"
               "\"cpu_before_s\":%.9g,\"cpu_after_s\":%.9g,\"cpu_gops\":%.9g,\"gpu_s\":%.9g,\"gpu_gops\":%.9g,"
               "\"cpu_corun_s\":%.9g,\"gpu_corun_s\":%.9g,\"combined_gops\":%.9g,\"speedup\":%.9g,\"same_window_speedup\":%.9g,"
               "\"reference_words\":%d,\"reference_bad_words\":0,\"all_finite\":true,\"cpu_relative_diff\":%.9g,"
               "\"cpu_yhash\":\"%016llx\",\"gpu_yhash\":\"%016llx\",\"post_timing_exact\":true}\n",
               argv[1], t.rows, t.cols, t.n, rep + 1, before, after, ops / cpu / 1e9, gpu, ops / gpu / 1e9,
               cpu_co, gpu_co, ops * (1.0 / cpu_co + 1.0 / gpu_co) / 1e9, ratios[rep], windows[rep],
               t.n * t.refs, max_cpu_diff / max_reference, (unsigned long long)cpu_hash, (unsigned long long)gpu_hash);
        fflush(stdout);
    }
    printf("RESULT {\"bench\":\"q4k_pilot_summary\",\"fixture\":\"%s\",\"n\":%d,\"rounds\":%d,"
           "\"median_speedup\":%.9g,\"median_same_window_speedup\":%.9g}\n", argv[1], t.n, reps, median(ratios, reps), median(windows, reps));
    for (int i = NBO - 1; i >= 0; i--) v3d_bo_destroy(&t.dev, &t.bo[i]);
    v3d_close(&t.dev); pf_pool_destroy(t.pool);
    free(t.raw); free(t.x); free(t.cpu); free(t.gpu); free(t.reference); free(t.q); free(t.d); free(t.fsum); free(t.index);
    return 0;
}
