/* mm_t_scales: q4k_scales of engine/src/kern_quant.c (NEON, static inline: the file is
 * included) against the scalar decode D * get6, M * get6 on random scale lines (random bytes
 * for the two 6-bit arrays, random positive finite fp16 D and M, every 7th line subnormal),
 * bitwise. KERNELS.md Part 3 (the 5-op unpack6 and the zip-with-exponent u8_scaled).
 * Build and run (WSL, from the repository root):
 *   aarch64-linux-gnu-gcc -O2 -mcpu=cortex-a72 -std=c11 -ffp-contract=off -Wall -Wextra \
 *     -Iengine/src logs/verify/mm_t_scales.c -o build/mm_t_scales -static
 *   qemu-aarch64 -cpu cortex-a72 build/mm_t_scales
 * Result 2026-10-22 (final kern_quant.c): "12800000 values, 0 mismatches"; the previous
 * decoder (792676c) gives the same. */
#include <stdio.h>
#include <string.h>

#include "../../engine/src/kern_quant.c"

int kern_prefetch = 0, kern_act_group = 16;
int kern_get6(const uint8_t *p, int j) {
    const int lo = j < 16 ? p[j] & 15 : p[j - 16] >> 4;
    return lo | ((p[16 + j % 8] >> (2 * (j / 8))) & 3) << 4;
}

static uint64_t s = 88172645463325252ull;
static uint32_t rnd(void) { s ^= s << 13, s ^= s >> 7, s ^= s << 17; return (uint32_t)s; }

int main(void) {
    long bad = 0, n = 0;
    for (int it = 0; it < 200000; it++) {
        uint8_t line[64] __attribute__((aligned(64)));
        for (int i = 0; i < 64; i++) line[i] = (uint8_t)rnd();
        __fp16 *h = (__fp16 *)line;
        for (int i = 0; i < 8; i++) { /* positive finite fp16: exponent field 0..30 */
            uint16_t b = (uint16_t)(rnd() & 0x7FFF);
            if ((b >> 10) == 31) b &= 0x7BFF;
            if (it % 7 == 0) b &= 0x03FF; /* subnormals */
            memcpy(&h[i], &b, 2);
        }
        float d[32], o[32];
        q4k_scales(line, d, o);
        for (int j = 0; j < 32; j++) {
            const int i = j % 4;
            const float rd = (float)h[i] * (float)kern_get6(line + 16, j);
            const float ro = (float)h[4 + i] * (float)kern_get6(line + 40, j);
            bad += memcmp(&rd, &d[j], 4) != 0 || memcmp(&ro, &o[j], 4) != 0;
            n += 2;
        }
    }
    printf("t_scales: %ld values, %ld mismatches\n", n, bad);
    return bad != 0;
}
