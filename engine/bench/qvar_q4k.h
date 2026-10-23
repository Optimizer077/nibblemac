/* Q4K_G32 sub-scale decode for the tbench variants: verbatim copy of kern_quant.c's unpack6 /
 * u8_scaled / q4k_scales (bit-identical by construction). */
#ifndef QVAR_Q4K_H
#define QVAR_Q4K_H

#include <arm_neon.h>

static inline void qv_unpack6(const uint8_t *p, uint8x16_t *a, uint8x16_t *b) {
    const uint8x16_t lo = vld1q_u8(p);
    const uint8x16_t hb = vreinterpretq_u8_u64(vld1q_dup_u64((const uint64_t *)(p + 16)));
    const uint8x16_t t = vshlq_u8(hb, vcombine_s8(vdup_n_s8(0), vdup_n_s8(-2)));
    *b = vsraq_n_u8(vandq_u8(t, vdupq_n_u8(0x30)), lo, 4);
    *a = vsliq_n_u8(lo, vandq_u8(t, vdupq_n_u8(3)), 4);
}

static inline void qv_u8_scaled(uint8x16_t v, float32x4_t mul, float *out) {
    const uint8x16_t z = vdupq_n_u8(0);
    const uint16x8_t e = vdupq_n_u16(0x4B00);
    const uint16x8_t l = vreinterpretq_u16_u8(vzip1q_u8(v, z)), h = vreinterpretq_u16_u8(vzip2q_u8(v, z));
    const uint16x8_t w[4] = {vzip1q_u16(l, e), vzip2q_u16(l, e), vzip1q_u16(h, e), vzip2q_u16(h, e)};
    for (int i = 0; i < 4; i++) {
        const float32x4_t f = vsubq_f32(vreinterpretq_f32_u16(w[i]), vdupq_n_f32(8388608.0f));
        vst1q_f32(out + 4 * i, vmulq_f32(f, mul));
    }
}

/* one superblock's d[8][4], o[8][4] (block, row) */
static inline void qv_q4k_scales(const uint8_t *p, float *d, float *o) {
    const __fp16 *h = (const __fp16 *)p;
    const float32x4_t D4 = vcvt_f32_f16(vld1_f16(h)), M4 = vcvt_f32_f16(vld1_f16(h + 4));
    uint8x16_t a, b;
    qv_unpack6(p + 16, &a, &b);
    qv_u8_scaled(a, D4, d), qv_u8_scaled(b, D4, d + 16);
    qv_unpack6(p + 40, &a, &b);
    qv_u8_scaled(a, M4, o), qv_u8_scaled(b, M4, o + 16);
}

#endif
