/* Gate Q4_0 test: identical raw inputs for Python QPU, own C QPU, and engine NEON.
 * Usage: v3d_q4_test fixture.bin 7|64 python-output.bin [iterations [C-output.bin]]
 * Timed calls cover submit + wait, as in the Python benchmark; setup/readback are separate.
 */
#define _POSIX_C_SOURCE 200809L
#include "v3d.h"
#include "q4gpu_kernels.h"
#include "kern.h"
#include "pq_quant.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef TEST_ROWS
#define TEST_ROWS 9216
#endif
#ifndef TEST_COLS
#define TEST_COLS 2560
#endif
enum { ROWS = TEST_ROWS, COLS = TEST_COLS, NMAX = 64, BLOCKS = COLS / 32, QPUS = 8 };
enum { CODE_BO, CODES_BO, SCALES_BO, STREAM_BO, Y_BO, PARAM_BO, NBO };
static size_t align128(size_t x) { return (x + 127) & ~(size_t)127; }
static double now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) { perror("clock_gettime"); exit(1); }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int read_exact(FILE *f, void *p, size_t size) {
    if (fread(p, 1, size, f) == size) return 0;
    errno = EIO; return -1;
}
static int cmp_double(const void *a, const void *b) {
    double aa = *(const double *)a, bb = *(const double *)b;
    return (aa > bb) - (aa < bb);
}
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint32_t negc(const int8_t *q, int b, int v, int n) {
    int sum = 0;
    for (int c = 0; c < 32; c++) sum += q[((size_t)b * n + v) * 32 + c];
    return (uint32_t)(-INT64_C(524296) * sum);
}
static uint32_t stream_word(struct q4gpu_tag tag, const pf_act *act, int b, int v0) {
    int v = v0 + tag.vec;
    switch (tag.kind) {
    case Q4GPU_X: return (uint32_t)(int32_t)act->q[((size_t)b * act->n + v) * 32 + tag.col];
    case Q4GPU_DX: return bits(act->d[(size_t)b * act->n + v]);
    case Q4GPU_NEGC: return negc(act->q, b, v, act->n);
    case Q4GPU_NEGC_NEXT: return b + 1 < BLOCKS ? negc(act->q, b + 1, v, act->n) : 0;
    default: fprintf(stderr, "invalid generated uniform descriptor\n"); exit(1);
    }
}
static int dispatch(struct v3d_device *dev, struct v3d_bo *bo, uint32_t shader,
                    unsigned passes, size_t parameter_stride, const uint32_t *handles) {
    for (unsigned pass = 0; pass < passes; pass++) {
        if (v3d_submit_8qpu(dev, shader, bo[PARAM_BO].gpu + (uint32_t)(pass * parameter_stride), handles, NBO) ||
            v3d_wait(dev, &bo[Y_BO], UINT64_C(3000000000))) return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s fixture.bin 7|64 python-y.bin [iterations [output.bin]]\n", argv[0]);
        return 2;
    }
    char *end;
    long parsed_n = strtol(argv[2], &end, 10);
    if (*end || (parsed_n != 7 && parsed_n != 64)) { fprintf(stderr, "n must be 7 or 64\n"); return 2; }
    int n = (int)parsed_n, iters = 10;
    if (argc >= 5) {
        long value = strtol(argv[4], &end, 10);
        if (*end || value < 1 || value > 100) { fprintf(stderr, "iterations must be 1..100\n"); return 2; }
        iters = (int)value;
    }
    int rc = 1;
    FILE *f = NULL;
    struct v3d_device dev = { .fd = -1 };
    struct v3d_bo bo[NBO] = {{0}};
    uint8_t *u = malloc((size_t)ROWS * COLS);
    uint16_t *d = malloc((size_t)ROWS * BLOCKS * 2);
    float *x = malloc((size_t)NMAX * COLS * 4);
    int8_t *qref = malloc((size_t)NMAX * COLS), *q = malloc((size_t)n * COLS);
    float *dxref = malloc((size_t)NMAX * BLOCKS * 4), *dx = malloc((size_t)n * BLOCKS * 4);
    float *fsum = malloc((size_t)n * BLOCKS * 4);
    size_t ybytes = (size_t)n * ROWS * 4, wbytes = (size_t)(ROWS / 4) * (COLS / 256) * 576;
    float *python = malloc(ybytes), *cpu = malloc(ybytes), *gpu = malloc(ybytes);
    uint8_t *cpu_weights = calloc(1, wbytes);
    if (!u || !d || !x || !qref || !q || !dxref || !dx || !fsum || !python || !cpu || !gpu || !cpu_weights) {
        errno = ENOMEM; perror("host allocation"); goto cleanup;
    }
    f = fopen(argv[1], "rb");
    uint32_t header[8];
    if (!f || read_exact(f, header, sizeof(header))) { perror("fixture header"); goto cleanup; }
    const uint32_t wanted[8] = {UINT32_C(0x34465051), 1, ROWS, COLS, NMAX, 0, 0, 0};
    if (memcmp(header, wanted, sizeof(wanted))) { fprintf(stderr, "unsupported fixture\n"); goto cleanup; }
    if (read_exact(f, u, (size_t)ROWS * COLS) || read_exact(f, d, (size_t)ROWS * BLOCKS * 2) ||
        read_exact(f, x, (size_t)NMAX * COLS * 4) || read_exact(f, qref, (size_t)NMAX * COLS) ||
        read_exact(f, dxref, (size_t)NMAX * BLOCKS * 4) || fgetc(f) != EOF) {
        fprintf(stderr, "truncated or oversized fixture\n"); goto cleanup;
    }
    fclose(f); f = fopen(argv[3], "rb");
    if (!f || read_exact(f, python, ybytes) || fgetc(f) != EOF) { fprintf(stderr, "invalid Python reference\n"); goto cleanup; }
    fclose(f); f = NULL;
    for (size_t i = 0; i < (size_t)ROWS * COLS; i++) {
        if (u[i] > 15) { fprintf(stderr, "invalid Q4 code\n"); goto cleanup; }
    }
    kern_quant_act(x, n, COLS, q, dx, fsum);
    for (int b = 0; b < BLOCKS; b++) for (int v = 0; v < n; v++) {
        if (memcmp(q + ((size_t)b * n + v) * 32, qref + (size_t)v * COLS + b * 32, 32) ||
            bits(dx[(size_t)b * n + v]) != bits(dxref[(size_t)v * BLOCKS + b])) {
            fprintf(stderr, "activation quantization differs from Python at b=%d v=%d\n", b, v); goto cleanup;
        }
    }
    /* Pack the same source bits into the actual engine's 4-row / 256-column tiles. */
    for (int r = 0; r < ROWS; r++) for (int b = 0; b < BLOCKS; b++) {
        uint8_t *sb = cpu_weights + ((size_t)(r / 4) * (COLS / 256) + b / 8) * 576;
        memcpy(sb + ((b % 8) * 4 + r % 4) * 2, d + (size_t)r * BLOCKS + b, 2);
        uint8_t *packed = sb + 64 + (b % 8) * 64 + (r % 4) * 16;
        const uint8_t *src = u + (size_t)r * COLS + b * 32;
        for (int j = 0; j < 16; j++) packed[j] = src[j] | (src[j + 16] << 4);
    }
    const pf_mat weights = { .dtype = PQ_Q4_G32, .rows = ROWS, .cols = COLS, .data = cpu_weights };
    const pf_act act = { .cols = COLS, .n = n, .q = q, .d = dx, .fsum = fsum };
    kern_q4(&weights, x, &act, cpu, n, 0, ROWS);

#ifdef PF_TEST_ENGINE
    /* Integration oracle: GPU prefix must equal the separately produced Python QPU
     * output, and every remaining row must equal the actual CPU kernel bit for bit. */
    {
        const float share = 0.33f;
        int split = (int)((double)ROWS * share / 256) * 256;
        const pf_mat *list[] = {&weights};
        pf_pool *pool = pf_pool_create(4);
        if (!pool || pf_pool_gpu_enable(pool, list, 1, share)) {
            fprintf(stderr, "engine GPU setup failed\n"); pf_pool_destroy(pool); goto cleanup;
        }
        pf_matmul(pool, &weights, x, gpu, n);
        pf_pool_destroy(pool);
        size_t gpu_bad = 0, cpu_bad = 0;
        float max_gpu = 0, max_cpu = 0;
        for (int v = 0; v < n; v++) for (int r = 0; r < ROWS; r++) {
            size_t pos = (size_t)v * ROWS + r;
            float expected = r < split ? python[pos] : cpu[pos];
            if (bits(gpu[pos]) != bits(expected)) {
                if (r < split) gpu_bad++; else cpu_bad++;
            }
            if (r < split) max_gpu = fmaxf(max_gpu, fabsf(gpu[pos] - expected));
            else max_cpu = fmaxf(max_cpu, fabsf(gpu[pos] - expected));
        }
        printf("RESULT {\"bench\":\"gpu-engine-split-oracle\",\"n\":%d,\"gpu_rows\":%d,"
               "\"gpu_bad_words\":%zu,\"cpu_bad_words\":%zu,\"gpu_max_abs\":%.9g,\"cpu_max_abs\":%.9g}\n",
               n, split, gpu_bad, cpu_bad, (double)max_gpu, (double)max_cpu);
        rc = gpu_bad || cpu_bad ? 1 : 0;
        goto cleanup;
    }
#endif

    const struct q4gpu_program *program = n == 7 ? &q4gpu_program_7 : &q4gpu_program_8;
    unsigned passes = (unsigned)n / program->n;
    size_t stream_words = program->start_words + BLOCKS * program->block_words;
    size_t stream_stride = align128(stream_words * 4), param_stride = align128(Q4GPU_PARAM_COUNT * 4);
    size_t code_bytes = (size_t)(ROWS / 32) * BLOCKS * 16 * 8 * 4;
    size_t scale_bytes = (size_t)(ROWS / 32) * BLOCKS * 16 * 4;
    const size_t sizes[NBO] = { program->words * 8, code_bytes + 384 * 4, scale_bytes + 48 * 4,
                               stream_stride * passes, ybytes, param_stride * passes };
    if (v3d_open(&dev, NULL)) { perror("v3d_open"); goto cleanup; }
    for (int i = 0; i < NBO; i++) {
        if (v3d_bo_create(&dev, &bo[i], sizes[i])) { perror("v3d_bo_create/mmap"); goto cleanup; }
        memset(bo[i].map, 0, bo[i].size);
    }
    memcpy(bo[CODE_BO].map, program->code, program->words * 8);
    uint32_t *codes = bo[CODES_BO].map, *scales = bo[SCALES_BO].map;
    for (int t = 0; t < ROWS / 32; t++) for (int b = 0; b < BLOCKS; b++) for (int lane = 0; lane < 16; lane++) {
        int a = t * 32 + lane, h = a + 16;
        size_t pos = ((size_t)t * BLOCKS + b) * 16 + lane;
        scales[pos] = d[(size_t)a * BLOCKS + b] | ((uint32_t)d[(size_t)h * BLOCKS + b] << 16);
        for (int w = 0; w < 8; w++) {
            uint32_t word = 0;
            for (int j = 0; j < 4; j++) {
                int c = b * 32 + w * 4 + j;
                word |= (uint32_t)u[(size_t)a * COLS + c] << (4 * j);
                word |= (uint32_t)u[(size_t)h * COLS + c] << (16 + 4 * j);
            }
            codes[pos * 8 + w] = word;
        }
    }
    for (size_t i = 0; i < ybytes / 4; i++) ((float *)bo[Y_BO].map)[i] = NAN;
    double prepare_start = now();
    for (unsigned pass = 0; pass < passes; pass++) {
        int v0 = (int)(pass * program->n);
        uint32_t *stream = (uint32_t *)((uint8_t *)bo[STREAM_BO].map + pass * stream_stride);
        size_t off = 0;
        for (size_t i = 0; i < program->start_words; i++) stream[off++] = stream_word(program->start[i], &act, 0, v0);
        for (int b = 0; b < BLOCKS; b++) for (size_t i = 0; i < program->block_words; i++) {
            stream[off++] = stream_word(program->block[i], &act, b, v0);
        }
        uint32_t *p = (uint32_t *)((uint8_t *)bo[PARAM_BO].map + pass * param_stride);
        uint32_t nt = ROWS / 32 / QPUS;
        p[Q4GPU_CODES] = bo[CODES_BO].gpu; p[Q4GPU_SCALES] = bo[SCALES_BO].gpu;
        p[Q4GPU_Y] = bo[Y_BO].gpu + (uint32_t)((size_t)v0 * ROWS * 4);
        p[Q4GPU_ACT] = bo[STREAM_BO].gpu + (uint32_t)(pass * stream_stride);
        p[Q4GPU_NT] = nt; p[Q4GPU_NB] = BLOCKS;
        p[Q4GPU_TSC] = nt * BLOCKS * 512; p[Q4GPU_TSS] = nt * BLOCKS * 64; p[Q4GPU_TSY] = nt * 128;
        p[Q4GPU_R4] = ROWS * 4; p[Q4GPU_MASK] = UINT32_C(0x000f000f);
        p[Q4GPU_K65536] = 65536; p[Q4GPU_K512] = 512; p[Q4GPU_K64] = 64;
    }
    double prepare_sec = now() - prepare_start;
    uint32_t handles[NBO];
    for (int i = 0; i < NBO; i++) handles[i] = bo[i].handle;
    if (dispatch(&dev, bo, bo[CODE_BO].gpu, passes, param_stride, handles)) { perror("first submit/wait"); goto cleanup; }
    double read_start = now();
    memcpy(gpu, bo[Y_BO].map, ybytes);
    double read_sec = now() - read_start;
    size_t python_bad = 0, cpu_bad = 0;
    float max_abs = 0, max_cpu = 0;
    bool finite = true;
    for (size_t i = 0; i < ybytes / 4; i++) {
        if (!isfinite(gpu[i]) || !isfinite(cpu[i]) || !isfinite(python[i])) finite = false;
        python_bad += bits(gpu[i]) != bits(python[i]);
        cpu_bad += bits(gpu[i]) != bits(cpu[i]);
        max_abs = fmaxf(max_abs, fabsf(gpu[i] - cpu[i])); max_cpu = fmaxf(max_cpu, fabsf(cpu[i]));
    }
    if (!finite || python_bad) { fprintf(stderr, "GPU mismatch: %zu Python words, finite=%d; no timing loop\n", python_bad, finite); goto cleanup; }
    double samples[100];
    for (int i = 0; i < iters; i++) {
        double begin = now();
        if (dispatch(&dev, bo, bo[CODE_BO].gpu, passes, param_stride, handles)) { perror("timed submit/wait"); goto cleanup; }
        samples[i] = now() - begin;
    }
    qsort(samples, (size_t)iters, sizeof(double), cmp_double);
    double sec = iters % 2 ? samples[iters / 2] : (samples[iters / 2 - 1] + samples[iters / 2]) / 2;
    /* Check again after the complete timing loop, catching stale/corrupted buffers. */
    if (memcmp(bo[Y_BO].map, python, ybytes)) { fprintf(stderr, "post-timing output changed\n"); goto cleanup; }
    if (argc == 6) {
        f = fopen(argv[5], "wb");
        if (!f || fwrite(gpu, 1, ybytes, f) != ybytes) { perror("write C output"); goto cleanup; }
        if (fclose(f)) { f = NULL; perror("close C output"); goto cleanup; }
        f = NULL;
    }
    printf("RESULT {\"bench\":\"gpu-c-q4\",\"rows\":%d,\"cols\":%d,\"n\":%d,\"passes\":%u,"
           "\"iterations\":%d,\"seconds\":%.9g,\"gops\":%.8g,\"all_finite\":true,"
           "\"checked_words\":%zu,\"python_bad_words\":%zu,\"exact_vs_python\":true,"
           "\"cpu_bad_words\":%zu,\"exact_vs_cpu\":%s,\"cpu_max_abs\":%.9g,\"cpu_relative_max\":%.9g,"
           "\"activation_quantization_exact\":true,\"uniform_build_seconds\":%.9g,\"readback_seconds\":%.9g}\n",
           ROWS, COLS, n, passes, iters, sec, 2.0 * ROWS * COLS * n / sec / 1e9, ybytes / 4,
           python_bad, cpu_bad, cpu_bad ? "false" : "true", (double)max_abs,
           max_cpu ? (double)max_abs / max_cpu : 0, prepare_sec, read_sec);
    rc = 0;
cleanup:
    if (f) fclose(f);
    for (int i = NBO - 1; i >= 0; i--) v3d_bo_destroy(&dev, &bo[i]);
    v3d_close(&dev);
    free(u); free(d); free(x); free(qref); free(q); free(dxref); free(dx); free(fsum);
    free(python); free(cpu); free(gpu); free(cpu_weights);
    return rc;
}
