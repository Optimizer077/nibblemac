/* Physical-device exactness test against the existing NEON Q4K/A8 kernel. */
#include "metal.h"
#include "../src/kern.h"
#include "../src/pq_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned rng = 0x12345678;
static unsigned next(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static int check(nm_metal *ctx, int rows, int cols, int n) {
    size_t wb = pq_quant_nbytes(PQ_Q4K_G32, rows, cols), qb = (size_t)n * cols;
    size_t fb = qb / 32 * 4, yb = (size_t)n * rows * 4;
    nm_buffer *w = nm_metal_buffer_create(ctx, wb), *q = nm_metal_buffer_create(ctx, qb);
    nm_buffer *d = nm_metal_buffer_create(ctx, fb), *fs = nm_metal_buffer_create(ctx, fb);
    nm_buffer *y = nm_metal_buffer_create(ctx, yb);
    nm_buffer *gpuw = NULL;
    float *x = malloc(qb * 4), *expected = malloc(yb);
    int result = -1;
    if (!w || !q || !d || !fs || !y || !x || !expected) goto done;
    unsigned char *packed = nm_metal_buffer_contents(w);
    for (size_t i = 0; i < wb; i++) packed[i] = next();
    for (size_t i = 0; i < wb; i += 576)
        for (int r = 0; r < 8; r++) {
            __fp16 half = (r + 1) * 0.0007f;
            memcpy(packed + i + r * 2, &half, 2);
        }
    for (size_t i = 0; i < qb; i++) x[i] = ((int)(next() % 2001) - 1000) / 997.0f;
    gpuw = nm_metal_a8_weights(ctx, rows, cols, packed, wb);
    if (!gpuw) goto done;
    pf_act a = {cols, n, nm_metal_buffer_contents(q), nm_metal_buffer_contents(d), nm_metal_buffer_contents(fs)};
    kern_quant_act(x, n, cols, (int8_t *)a.q, (float *)a.d, (float *)a.fsum);
    pf_mat mat = {PQ_Q4K_G32, rows, cols, packed};
    kern_q4k(&mat, x, &a, expected, n, 0, rows);
    memset(nm_metal_buffer_contents(y), 0xff, yb);
    char error[4096];
    if (nm_metal_matmul_a8(ctx, rows, cols, n, gpuw, q, d, fs, y, error, sizeof error)) {
        fprintf(stderr, "%s\n", error); goto done;
    }
    if (memcmp(expected, nm_metal_buffer_contents(y), yb)) {
        float *got = nm_metal_buffer_contents(y);
        for (size_t i = 0; i < yb / 4; i++)
            if (memcmp(expected + i, got + i, 4)) {
                fprintf(stderr, "A8 mismatch rows=%d cols=%d n=%d i=%zu expected=%a got=%a\n",
                        rows, cols, n, i, expected[i], got[i]); break;
            }
        goto done;
    }
    printf("PASS exact Q4K/A8 NEON==Metal rows=%d cols=%d batch=%d\n", rows, cols, n);
    result = 0;
done:
    free(expected); free(x);
    nm_metal_buffer_destroy(gpuw);
    nm_metal_buffer_destroy(y); nm_metal_buffer_destroy(fs); nm_metal_buffer_destroy(d);
    nm_metal_buffer_destroy(q); nm_metal_buffer_destroy(w);
    return result;
}
static int invalid(nm_metal *ctx) {
    unsigned char packed[576] = {0};
    nm_buffer *w = nm_metal_a8_weights(ctx, 4, 256, packed, sizeof packed);
    nm_buffer *q = nm_metal_buffer_create(ctx, 256), *d = nm_metal_buffer_create(ctx, 32);
    nm_buffer *fs = nm_metal_buffer_create(ctx, 32), *y = nm_metal_buffer_create(ctx, 16);
    int failed = !w || !q || !d || !fs || !y;
    char error[512];
#define REJECT(...) do { if (!nm_metal_matmul_a8(__VA_ARGS__, error, sizeof error)) failed++; } while (0)
    REJECT(ctx, 3, 256, 1, w, q, d, fs, y);
    REJECT(ctx, 4, 255, 1, w, q, d, fs, y);
    REJECT(ctx, 4, 256, 0, w, q, d, fs, y);
    REJECT(ctx, 4, 256, 2, w, q, d, fs, y);
    REJECT(ctx, 4, 256, 1, w, q, d, fs, w);
    REJECT(ctx, 4, 256, 1, w, q, d, fs, d);
    REJECT(ctx, 4, 256, 1, NULL, q, d, fs, y);
    REJECT(NULL, 4, 256, 1, w, q, d, fs, y);
    REJECT(ctx, UINT32_MAX - 3, UINT32_MAX - 255, UINT32_MAX, w, q, d, fs, y);
#undef REJECT
    if (nm_metal_a8_weights(ctx, 4, 256, packed, sizeof packed - 1)) failed++;
    if (nm_metal_a8_weights(ctx, 3, 256, packed, sizeof packed)) failed++;
    if (nm_metal_a8_weights(ctx, 4, 256, NULL, sizeof packed)) failed++;
    nm_metal_buffer_destroy(y); nm_metal_buffer_destroy(fs); nm_metal_buffer_destroy(d);
    nm_metal_buffer_destroy(q); nm_metal_buffer_destroy(w);
    if (!failed) puts("PASS 12 Q4K/A8 invalid buffer, shape and upload cases rejected");
    return failed ? -1 : 0;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char error[4096];
    nm_metal *ctx = nm_metal_create(argv[1], error, sizeof error);
    if (!ctx) { fprintf(stderr, "%s\n", error); return 1; }
    const int batches[] = {1, 2, 3, 7, 8, 31, 64};
    int failed = 0;
    for (size_t i = 0; i < sizeof batches / sizeof batches[0]; i++) {
        failed |= check(ctx, 4, 256, batches[i]);
        failed |= check(ctx, 20, 768, batches[i]);
        failed |= check(ctx, 128, 3584, batches[i]);
    }
    failed |= invalid(ctx);
    nm_metal_destroy(ctx);
    return failed ? 1 : 0;
}
