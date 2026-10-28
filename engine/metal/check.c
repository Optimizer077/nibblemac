/* Deterministic correctness tests and synthetic, resident-buffer microbenchmarks.
 * No model downloads or Python packages required. CPU oracle accumulates in double. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "metal.h"

static uint32_t rng = 0x6e696262;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static float random_float(void) { return (float)(random_u32() % 2001) / 1000.0f - 1.0f; }
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static size_t weight_bytes(uint32_t dtype, uint32_t rows, uint32_t cols) {
    return dtype == NM_F32 ? (size_t)rows * cols * 4 : (size_t)(rows / 4) * (cols / 256) * 576;
}
static float half_load(const unsigned char *p) {
    __fp16 value;
    memcpy(&value, p, 2);
    return value;
}
static void half_store(unsigned char *p, float x) {
    __fp16 value = x;
    memcpy(p, &value, 2);
}
static unsigned unpack6(const unsigned char *p, unsigned i) {
    return ((p[i % 16] >> (4 * (i / 16))) & 15) |
           (((p[16 + i % 8] >> (2 * (i / 8))) & 3) << 4);
}
static void fill_weights(void *data, uint32_t dtype, uint32_t rows, uint32_t cols) {
    if (dtype == NM_F32) {
        float *p = data;
        for (size_t i = 0; i < (size_t)rows * cols; i++) p[i] = random_float();
        return;
    }
    unsigned char *p = data;
    size_t bytes = weight_bytes(dtype, rows, cols);
    for (size_t i = 0; i < bytes; i++) p[i] = (unsigned char)random_u32();
    for (size_t off = 0; off < bytes; off += 576)
        for (unsigned r = 0; r < 4; r++) {
            half_store(p + off + r * 2, 0.001f * (r + 1));
            half_store(p + off + 8 + r * 2, 0.009f * (r + 1));
        }
}
static float weight_at(const void *data, uint32_t dtype, uint32_t row, uint32_t col,
                       uint32_t cols) {
    if (dtype == NM_F32) return ((const float *)data)[(size_t)row * cols + col];
    const unsigned char *sb = (const unsigned char *)data +
        ((size_t)(row / 4) * (cols / 256) + col / 256) * 576;
    unsigned r = row % 4, k = (col % 256) / 32, lane = col % 32;
    unsigned q = (sb[64 + k * 64 + r * 16 + lane % 16] >> (4 * (lane / 16))) & 15;
    float scale = half_load(sb + r * 2) * unpack6(sb + 16, k * 4 + r);
    float minimum = half_load(sb + 8 + r * 2) * unpack6(sb + 40, k * 4 + r);
    return scale * q - minimum;
}
static int verify(const void *w, const float *x, const float *y, uint32_t dtype,
                   uint32_t rows, uint32_t cols, uint32_t batch, double *max_error) {
    double error = 0, peak = 0;
    for (uint32_t b = 0; b < batch; b++)
        for (uint32_t r = 0; r < rows; r++) {
            double sum = 0;
            for (uint32_t c = 0; c < cols; c++)
                sum += (double)weight_at(w, dtype, r, c, cols) * x[(size_t)b * cols + c];
            double got = y[(size_t)b * rows + r];
            if (!isfinite(got)) return -1;
            error = fmax(error, fabs(got - sum));
            peak = fmax(peak, fabs(sum));
        }
    *max_error = error;
    return error <= 2e-5 * fmax(peak, 1.0) ? 0 : -1;
}
static int run_case(nm_metal *ctx, uint32_t dtype, uint32_t rows, uint32_t cols,
                     uint32_t batch, int benchmark, int zero) {
    nm_buffer *w = nm_metal_buffer_create(ctx, weight_bytes(dtype, rows, cols));
    nm_buffer *x = nm_metal_buffer_create(ctx, (size_t)batch * cols * 4);
    nm_buffer *y = nm_metal_buffer_create(ctx, (size_t)batch * rows * 4);
    int result = -1;
    char error[2048] = {0};
    if (!w || !x || !y) { fprintf(stderr, "Buffer allocation failed\n"); goto done; }
    fill_weights(nm_metal_buffer_contents(w), dtype, rows, cols);
    float *inputs = nm_metal_buffer_contents(x), *outputs = nm_metal_buffer_contents(y);
    for (size_t i = 0; i < (size_t)batch * cols; i++) inputs[i] = zero ? 0 : random_float();
    // NaN sentinel catches missed rows and batch indexing bugs.
    for (size_t i = 0; i < (size_t)batch * rows; i++) outputs[i] = NAN;
    double gpu = 0, max_error = 0;
    if (nm_metal_matmul(ctx, dtype, rows, cols, batch, w, x, y, 1, &gpu, error, sizeof error))
        goto done;
    if (verify(nm_metal_buffer_contents(w), inputs, outputs, dtype, rows, cols, batch, &max_error)) {
        snprintf(error, sizeof error, "Numerical mismatch for dtype %u, %ux%u, batch %u",
                 dtype, rows, cols, batch);
        goto done;
    }
    if (!benchmark) {
        printf("PASS dtype=%u rows=%u cols=%u batch=%u zero=%d max_abs_error=%.6g\n",
               dtype, rows, cols, batch, zero, max_error);
    } else {
        // Warmup and CPU reference are excluded. Keep single-submit latency separate
        // from repeated-dispatch throughput; neither is end-to-end model performance.
        for (int i = 0; i < 3; i++)
            if (nm_metal_matmul(ctx, dtype, rows, cols, batch, w, x, y, 1, &gpu, error, sizeof error))
                goto done;
        double start = now();
        if (nm_metal_matmul(ctx, dtype, rows, cols, batch, w, x, y, 1, &gpu, error, sizeof error))
            goto done;
        double single_wall = now() - start, single_gpu = gpu;
        const uint32_t repeats = 30;
        start = now();
        if (nm_metal_matmul(ctx, dtype, rows, cols, batch, w, x, y, repeats, &gpu, error, sizeof error))
            goto done;
        double wall = now() - start;
        // Re-check after repeated dispatches, not only after the warmup.
        if (verify(nm_metal_buffer_contents(w), inputs, outputs, dtype, rows, cols, batch, &max_error)) {
            snprintf(error, sizeof error, "Numerical mismatch after benchmark repeats");
            goto done;
        }
        if (gpu <= 0 || !isfinite(gpu)) {
            snprintf(error, sizeof error, "GPU timestamps unavailable"); goto done;
        }
        printf("{\"dtype\":\"%s\",\"rows\":%u,\"cols\":%u,\"batch\":%u,"
               "\"repeats\":%u,\"weight_bytes\":%zu,\"single_wall_ms\":%.6f,"
               "\"single_gpu_ms\":%.6f,\"repeated_wall_ms_per_dispatch\":%.6f,"
               "\"repeated_gpu_ms_per_dispatch\":%.6f,\"gflops\":%.6f,"
               "\"max_abs_error\":%.9g,\"correct\":true}\n",
               dtype == NM_F32 ? "f32" : "q4k", rows, cols, batch, repeats,
               weight_bytes(dtype, rows, cols), single_wall * 1e3, single_gpu * 1e3,
               wall * 1e3 / repeats, gpu * 1e3 / repeats,
               2.0 * rows * cols * batch * repeats / gpu / 1e9, max_error);
    }
    result = 0;
done:
    if (result && error[0]) fprintf(stderr, "%s\n", error);
    nm_metal_buffer_destroy(y); nm_metal_buffer_destroy(x); nm_metal_buffer_destroy(w);
    return result;
}

/* An independent analytical layout check, not the decoder used by verify():
 * identity inputs recover every weight. Two superblocks cover all 64 possible
 * sub-scale/sub-min values and both nibble halves across multiple row tiles. */
static int test_q4k_layout(nm_metal *ctx) {
    const uint32_t rows = 8, cols = 512, batch = 512;
    nm_buffer *w = nm_metal_buffer_create(ctx, weight_bytes(NM_Q4K, rows, cols));
    nm_buffer *x = nm_metal_buffer_create(ctx, (size_t)batch * cols * 4);
    nm_buffer *y = nm_metal_buffer_create(ctx, (size_t)batch * rows * 4);
    int result = -1;
    if (!w || !x || !y) goto done;
    unsigned char *packed = nm_metal_buffer_contents(w);
    memset(packed, 0, weight_bytes(NM_Q4K, rows, cols));
    for (unsigned tile = 0; tile < rows / 4; tile++)
        for (unsigned b = 0; b < cols / 256; b++) {
            unsigned char *sb = packed + (tile * (cols / 256) + b) * 576;
            for (unsigned r = 0; r < 4; r++) {
                half_store(sb + r * 2, (tile * 4 + r + 1) / 16.0f);
                half_store(sb + 8 + r * 2, (tile * 4 + r + 1) / 8.0f);
            }
            for (unsigned i = 0; i < 32; i++) {
                unsigned values[] = {b * 32 + i, 63 - (b * 32 + i)};
                for (unsigned v = 0; v < 2; v++) {
                    unsigned char *p = sb + 16 + v * 24;
                    p[i % 16] |= (values[v] & 15) << (4 * (i / 16));
                    p[16 + i % 8] |= (values[v] >> 4) << (2 * (i / 8));
                }
            }
            for (unsigned k = 0; k < 8; k++)
                for (unsigned r = 0; r < 4; r++)
                    for (unsigned lane = 0; lane < 32; lane++) {
                        unsigned code = (lane + r + k + tile) % 16;
                        sb[64 + k * 64 + r * 16 + lane % 16] |= code << (4 * (lane / 16));
                    }
        }
    float *inputs = nm_metal_buffer_contents(x), *outputs = nm_metal_buffer_contents(y);
    memset(inputs, 0, (size_t)batch * cols * 4);
    for (unsigned c = 0; c < cols; c++) inputs[(size_t)c * cols + c] = 1;
    for (unsigned i = 0; i < batch * rows; i++) outputs[i] = NAN;
    char error[512];
    if (nm_metal_matmul(ctx, NM_Q4K, rows, cols, batch, w, x, y, 1, NULL, error, sizeof error)) {
        fprintf(stderr, "%s\n", error); goto done;
    }
    for (unsigned c = 0; c < cols; c++)
        for (unsigned r = 0; r < rows; r++) {
            unsigned k = (c % 256) / 32, sub = (c / 256) * 32 + k * 4 + r % 4;
            unsigned code = (c % 32 + r % 4 + k + r / 4) % 16;
            float expected = (r + 1) / 16.0f * sub * code - (r + 1) / 8.0f * (63 - sub);
            if (outputs[(size_t)c * rows + r] != expected) {
                fprintf(stderr, "Analytical Q4K layout mismatch at row=%u col=%u\n", r, c);
                goto done;
            }
        }
    puts("PASS Q4K analytical layout: all 64 sub-scales/minima, both nibble halves, multiple tiles");
    result = 0;
done:
    nm_metal_buffer_destroy(y); nm_metal_buffer_destroy(x); nm_metal_buffer_destroy(w);
    return result;
}

static int test_validation(nm_metal *ctx) {
    nm_buffer *w = nm_metal_buffer_create(ctx, 4096);
    nm_buffer *x = nm_metal_buffer_create(ctx, 4096);
    nm_buffer *y = nm_metal_buffer_create(ctx, 4096);
    if (!w || !x || !y) {
        nm_metal_buffer_destroy(y); nm_metal_buffer_destroy(x); nm_metal_buffer_destroy(w);
        return -1;
    }
    char err[512];
    int failed = 0;
#define REJECT(...) do { if (nm_metal_matmul(__VA_ARGS__, err, sizeof err) == 0) failed++; } while (0)
    REJECT(ctx, 99, 4, 256, 1, w, x, y, 1, NULL);
    REJECT(ctx, NM_F32, 0, 256, 1, w, x, y, 1, NULL);
    REJECT(ctx, NM_F32, 4, 0, 1, w, x, y, 1, NULL);
    REJECT(ctx, NM_F32, 4, 256, 0, w, x, y, 1, NULL);
    REJECT(ctx, NM_F32, 4, 256, 1, w, x, y, 0, NULL);
    REJECT(ctx, NM_Q4K, 3, 256, 1, w, x, y, 1, NULL);
    REJECT(ctx, NM_Q4K, 4, 255, 1, w, x, y, 1, NULL);
    REJECT(ctx, NM_F32, 4096, 4096, 1, w, x, y, 1, NULL);
    REJECT(ctx, NM_F32, UINT32_MAX, UINT32_MAX, UINT32_MAX, w, x, y, 1, NULL);
    REJECT(ctx, NM_F32, 4, 256, 1, w, x, x, 1, NULL);
    REJECT(ctx, NM_F32, 4, 256, 1, w, x, w, 1, NULL);
    REJECT(ctx, NM_F32, 4, 256, 1, NULL, x, y, 1, NULL);
    REJECT(NULL, NM_F32, 4, 256, 1, w, x, y, 1, NULL);
#undef REJECT
    if (nm_metal_buffer_create(ctx, 0)) failed++;
    nm_metal_buffer_destroy(y); nm_metal_buffer_destroy(x); nm_metal_buffer_destroy(w);
    if (!failed) puts("PASS 14 invalid argument/buffer cases rejected");
    return failed ? -1 : 0;
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "test") && strcmp(argv[1], "bench"))) {
        fprintf(stderr, "usage: %s test|bench path/to/kernels.metal\n", argv[0]);
        return 2;
    }
    char error[4096] = {0};
    nm_metal *ctx = nm_metal_create(argv[2], error, sizeof error);
    if (!ctx) { fprintf(stderr, "%s\n", error); return 1; }
    fprintf(stderr, "Metal: %s; recommended working set: %.2f GiB\n",
            nm_metal_device_name(ctx), nm_metal_recommended_memory(ctx) / 1073741824.0);
    int failed = 0;
    if (!strcmp(argv[1], "test")) {
        const uint32_t shapes[][3] = {{1, 1, 1}, {3, 33, 3}, {7, 255, 8}, {129, 1025, 3}};
        for (unsigned i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
            failed |= run_case(ctx, NM_F32, shapes[i][0], shapes[i][1], shapes[i][2], 0, 0);
        const uint32_t quant_shapes[][3] = {{4, 256, 1}, {12, 768, 3}, {128, 1024, 8}};
        for (unsigned i = 0; i < sizeof quant_shapes / sizeof quant_shapes[0]; i++)
            failed |= run_case(ctx, NM_Q4K, quant_shapes[i][0], quant_shapes[i][1], quant_shapes[i][2], 0, 0);
        failed |= run_case(ctx, NM_F32, 7, 33, 3, 0, 1);
        failed |= run_case(ctx, NM_Q4K, 12, 512, 3, 0, 1);
        failed |= test_q4k_layout(ctx);
        failed |= test_validation(ctx);
    } else {
        failed |= run_case(ctx, NM_F32, 3584, 1024, 1, 1, 0);
        failed |= run_case(ctx, NM_Q4K, 3584, 1024, 1, 1, 0);
        failed |= run_case(ctx, NM_Q4K, 1024, 3584, 1, 1, 0);
        failed |= run_case(ctx, NM_Q4K, 3584, 1024, 8, 1, 0);
        failed |= run_case(ctx, NM_Q4K, 248320, 1024, 1, 1, 0);
    }
    nm_metal_destroy(ctx);
    return failed ? 1 : 0;
}
