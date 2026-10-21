/* Activation quantization (fp32 -> int8 per 32-column block) and row dequantization for
 * the quantized dtypes. Both are off the hot path: activation quantization touches cols x n
 * floats once per call; get_row is the embedding lookup (one row per token).
 */
#include <arm_neon.h>

#include "kern.h"
#include "pq_quant.h"

int kern_prefetch = 0;
int kern_act_group = 16;

void kern_quant_act(const float *X, int n, int cols, int8_t *q, float *d, float *fsum) {
    const int nb = cols / KERN_ACT_BLOCK;
    for (int v = 0; v < n; v++) {
        for (int b = 0; b < nb; b++) {
            const float *x = X + (size_t)v * cols + b * 32;
            float32x4_t xv[8], m = vdupq_n_f32(0.0f);
            for (int j = 0; j < 8; j++) {
                xv[j] = vld1q_f32(x + 4 * j);
                m = vmaxq_f32(m, vabsq_f32(xv[j]));
            }
            const float amax = vmaxvq_f32(m);
            const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
            int32x4_t qi[8], sum = vdupq_n_s32(0);
            for (int j = 0; j < 8; j++) {
                qi[j] = vcvtnq_s32_f32(vmulq_n_f32(xv[j], id));
                sum = vaddq_s32(sum, qi[j]);
            }
            const size_t k = (size_t)b * n + v; /* block-major */
            int8_t *o = q + k * 32;
            for (int j = 0; j < 8; j += 4) {
                int16x8_t a = vcombine_s16(vqmovn_s32(qi[j]), vqmovn_s32(qi[j + 1]));
                int16x8_t c = vcombine_s16(vqmovn_s32(qi[j + 2]), vqmovn_s32(qi[j + 3]));
                vst1q_s8(o + 4 * j, vcombine_s8(vqmovn_s16(a), vqmovn_s16(c)));
            }
            d[k] = amax / 127.0f;
            fsum[k] = d[k] * (float)vaddvq_s32(sum);
        }
    }
}

/* Value j (0..31) of a 6-bit array packed as in FORMAT.md (Q4K): 16 B of low nibbles
 * (byte i holds values i and i + 16), then 8 B of high bit pairs (byte i holds bits 4..5 of
 * values i, i + 8, i + 16, i + 24 at bit 0, 2, 4, 6). */
int kern_get6(const uint8_t *p, int j) {
    const int lo = j < 16 ? p[j] & 15 : p[j - 16] >> 4;
    return lo | ((p[16 + j % 8] >> (2 * (j / 8))) & 3) << 4;
}

/* Row `row` of a quantized matrix: tile row / 4, position row % 4 inside each group. */
void kern_get_row_q(const pf_mat *W, int row, float *dst) {
    const int nsb = W->cols / PQ_SB_COLS, i = row % 4;
    const size_t sbb = pq_sb_bytes(W->dtype);
    const uint8_t *p = (const uint8_t *)W->data + (size_t)(row / 4) * nsb * sbb;
    for (int sb = 0; sb < nsb; sb++, p += sbb, dst += PQ_SB_COLS) {
        const __fp16 *sc = (const __fp16 *)p;
        const uint8_t *codes = p + PQ_SB_SCALE_BYTES;
        if (W->dtype == PQ_Q8_G32) {
            for (int g = 0; g < 8; g++) {
                const float dd = (float)sc[g * 4 + i];
                const int8_t *q = (const int8_t *)codes + g * 128 + i * 32;
                for (int j = 0; j < 32; j++) dst[g * 32 + j] = dd * (float)q[j];
            }
        } else if (W->dtype == PQ_Q4_G32) {
            for (int k = 0; k < 8; k++) {
                const float dd = (float)sc[k * 4 + i];
                const uint8_t *q = codes + k * 64 + i * 16;
                for (int j = 0; j < 16; j++) {
                    dst[k * 32 + j] = dd * (float)((q[j] & 15) - 8);
                    dst[k * 32 + 16 + j] = dd * (float)((q[j] >> 4) - 8);
                }
            }
        } else { /* PQ_Q4K_G32: 6-bit sub-scale / sub-min under fp16 D, M of this row */
            const float D = (float)sc[i], M = (float)sc[4 + i];
            for (int k = 0; k < 8; k++) {
                const float dd = D * (float)kern_get6(p + 16, k * 4 + i);
                const float mm = M * (float)kern_get6(p + 40, k * 4 + i);
                const uint8_t *q = codes + k * 64 + i * 16;
                for (int j = 0; j < 16; j++) {
                    dst[k * 32 + j] = dd * (float)(q[j] & 15) - mm;
                    dst[k * 32 + 16 + j] = dd * (float)(q[j] >> 4) - mm;
                }
            }
        }
    }
}
