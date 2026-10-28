/* Exact I8MM pair/tail/subrange checks against the inherited no-SDOT kernel. */
#include "../src/kern.h"
#include "../src/pq_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NM_USE_I8MM
#error "Build this test with BACKEND=cpu-i8mm"
#endif

void oracle_q4(const pf_mat *, const float *, const pf_act *, float *, int, int, int);
void oracle_q4k(const pf_mat *, const float *, const pf_act *, float *, int, int, int);
static unsigned seed = 0x69884d4d;
static unsigned next(void) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }

static int check(int dtype, int rows, int cols, int n, int mode) {
    size_t wb = pq_quant_nbytes(dtype, rows, cols), qb = (size_t)n * cols;
    unsigned char *w = malloc(wb);
    float *x = malloc(qb * 4), *d = malloc(qb / 32 * 4), *fs = malloc(qb / 32 * 4);
    int8_t *q = malloc(qb);
    size_t yb = (size_t)n * rows * 4;
    float *expected = malloc(yb), *got = malloc(yb);
    if (!w || !x || !d || !fs || !q || !expected || !got) exit(1);
    for (size_t i = 0; i < wb; i++) w[i] = next();
    for (size_t i = 0; i < wb; i += 576) for (int h = 0; h < (dtype == PQ_Q4K_G32 ? 8 : 32); h++) {
        __fp16 scale = (h + 1) * 0.0007f;
        memcpy(w + i + h * 2, &scale, 2);
    }
    for (size_t i = 0; i < qb; i++)
        x[i] = mode == 1 ? 0.0f : mode == 2 ? (i % 2 ? 1.0f : -1.0f) : ((int)(next() % 2001) - 1000) / 997.0f;
    kern_quant_act(x, n, cols, q, d, fs);
    pf_act a = {cols, n, q, d, fs};
    pf_mat mat = {dtype, rows, cols, w};
    kern_fn reference = dtype == PQ_Q4K_G32 ? oracle_q4k : oracle_q4;
    kern_fn candidate = dtype == PQ_Q4K_G32 ? kern_q4k : kern_q4;
    int failed = 0;
    for (int subrange = 0; subrange < 2; subrange++) {
        memset(expected, 0x55, yb); memset(got, 0x55, yb);
        int first = subrange ? 4 : 0, last = subrange ? rows - 4 : rows;
        reference(&mat, x, &a, expected, n, first, last);
        candidate(&mat, x, &a, got, n, first, last);
        if (memcmp(expected, got, yb)) {
            fprintf(stderr, "I8MM mismatch dtype=%d rows=%d cols=%d batch=%d mode=%d subrange=%d\n",
                    dtype, rows, cols, n, mode, subrange);
            failed = 1;
        }
    }
    free(got); free(expected); free(q); free(fs); free(d); free(x); free(w);
    return failed;
}

int main(void) {
    const int batches[] = {1, 2, 3, 4, 5, 6, 7, 8, 31, 64};
    int failed = 0, cases = 0;
    for (int dtype = PQ_Q4_G32; dtype <= PQ_Q4K_G32; dtype++) {
        for (unsigned i = 0; i < sizeof batches / sizeof batches[0]; i++) {
            failed |= check(dtype, 20, 256, batches[i], 0); cases += 2;
            failed |= check(dtype, 128, 3584, batches[i], 0); cases += 2;
        }
        failed |= check(dtype, 36, 768, 7, 1); cases += 2;
        failed |= check(dtype, 36, 768, 8, 2); cases += 2;
    }
    printf("%s %d exact I8MM/no-SDOT cases (pairs, odd tails, subranges, zero, saturated A8)\n",
           failed ? "FAIL" : "PASS", cases);
    return failed ? 1 : 0;
}
