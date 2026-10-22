/* Compiler sanity check: the max-by-magnitude loop of the quantizers, which clang 20.1.8 miscompiles
 * for AArch64 at -O3 (engine/TOOLCHAIN.md section 3.1; CONTRIBUTING.md section 5).
 *
 * Same generator as engine/bench/kbench_util.c (xorshift64*, seed 42, tile 0 = 4 x 1024 floats,
 * scale 0.05) and the same loop as its Q4 packer, over every 32-column block of the tile. The loop's
 * answer is compared with a scalar pass over volatile loads; exit status 1 if any block differs.
 *
 *   aarch64-linux-gnu-gcc -O3 -mcpu=cortex-a72 -std=c11 -ffp-contract=off clang20_maxabs.c -static -lm
 *   qemu-aarch64 -cpu cortex-a72 ./a.out        -> "wrong blocks 0 of 128" (GCC 13-16, clang 22)
 *   clang-20 --target=aarch64-linux-gnu -O3 ...  -> "row 0 block 22: loop 0.0406423, true -0.220291,
 *                                                   wrong blocks 3 of 128" (clang 20.1.8; correct at
 *                                                   -O2, -fno-unroll-loops or -fno-builtin-fabsf,
 *                                                   and on x86-64)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t rng_next(uint64_t *s) {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    return *s * 2685821657736338717ull;
}

static void gen(float *w, size_t n, uint64_t *s, float scale) {
    for (size_t i = 0; i < n; i++) {
        uint64_t r = rng_next(s);
        float a = (float)(r >> 40) * (1.0f / 16777216.0f);
        float b = (float)((r >> 16) & 0xFFFFFF) * (1.0f / 16777216.0f);
        float v = (a + b - 1.0f) * scale;
        w[i] = (r & 63) == 0 ? 6.0f * v : v;
    }
}

/* the loop under test (kbench_util.c pack_tile, PQ_Q4_G32 branch) */
static float maxabs(const float *x) {
    float mx = 0;
    for (int j = 0; j < 32; j++)
        if (fabsf(x[j]) > fabsf(mx)) mx = x[j];
    return mx;
}

static float maxabs_ref(const volatile float *x) {
    float mx = 0;
    for (int j = 0; j < 32; j++) {
        float v = x[j], av = v < 0 ? -v : v, am = mx < 0 ? -mx : mx;
        if (av > am) mx = v;
    }
    return mx;
}

int main(void) {
    const int cols = 1024;
    uint64_t s = 42 * 0x9E3779B97F4A7C15ull + 7;
    float *w = malloc(4 * cols * sizeof(float));
    if (!w) return 2;
    gen(w, 4 * cols, &s, 0.05f);
    double sum = 0;
    int bad = 0;
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < cols / 32; k++) {
            const float *x = w + i * cols + k * 32;
            float m = maxabs(x), r = maxabs_ref(x);
            sum += m;
            if (m != r && bad++ == 0)
                printf("row %d block %d: loop %.6g, true %.6g\n", i, k, m, r);
        }
    printf("checksum %.9g, wrong blocks %d of %d\n", sum, bad, 4 * cols / 32);
    free(w);
    return bad != 0;
}
