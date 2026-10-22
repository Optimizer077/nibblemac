/* tbench variant "isq": the Q4_G32 / Q4K_G32 kernel of kern_quant.c with its NEON ops
 * interleaved for the Cortex-A72's in-order dispatch (KERNELS.md Part 5).
 *
 * The A72 dispatches at most one F0 and one F1 uop per cycle, in program order (UAN0016A
 * section 4.1). SMLAL/SMULL run only on F0, so a run of SMLALs dispatches one per cycle and
 * every op behind it waits, whatever its pipe. GCC's schedule of kern_quant.c clusters the 16
 * multiplies of a vector. Here each block runs as a software pipeline over its vectors: while
 * the 16 multiplies of vector v dispatch, the reduction of vector v-1 and the float update of
 * vector v-2 fill the F1 slots between them. Same operations, same operands, same per-vector
 * order of the float operations => bit-identical results. The file is compiled without
 * instruction scheduling so GCC keeps this order.
 */
#pragma GCC optimize("no-schedule-insns", "no-schedule-insns2")
#include <arm_neon.h>

#include "kern.h"
#include "pq_quant.h"
#include "qvar.h"

#define INLINE static inline __attribute__((always_inline))
#define NVMAX 7

#ifndef ISQ_MAGIC
#define ISQ_MAGIC 0 /* 1: int->float by adding 1.5 x 2^23 and subtracting it as a float */
#endif

/* ---- Q4K scale decode: verbatim from kern_quant.c (bit-identical by construction) ---- */
INLINE void unpack6(const uint8_t *p, uint8x16_t *a, uint8x16_t *b) {
    const uint8x16_t lo = vld1q_u8(p);
    const uint8x16_t hb = vreinterpretq_u8_u64(vld1q_dup_u64((const uint64_t *)(p + 16)));
    const uint8x16_t t = vshlq_u8(hb, vcombine_s8(vdup_n_s8(0), vdup_n_s8(-2)));
    *b = vsraq_n_u8(vandq_u8(t, vdupq_n_u8(0x30)), lo, 4);
    *a = vsliq_n_u8(lo, vandq_u8(t, vdupq_n_u8(3)), 4);
}

INLINE void u8_scaled(uint8x16_t v, float32x4_t mul, float *out) {
    const uint8x16_t z = vdupq_n_u8(0);
    const uint16x8_t e = vdupq_n_u16(0x4B00);
    const uint16x8_t l = vreinterpretq_u16_u8(vzip1q_u8(v, z)), h = vreinterpretq_u16_u8(vzip2q_u8(v, z));
    const uint16x8_t w[4] = {vzip1q_u16(l, e), vzip2q_u16(l, e), vzip1q_u16(h, e), vzip2q_u16(h, e)};
    for (int i = 0; i < 4; i++) {
        const float32x4_t f = vsubq_f32(vreinterpretq_f32_u16(w[i]), vdupq_n_f32(8388608.0f));
        vst1q_f32(out + 4 * i, vmulq_f32(f, mul));
    }
}

INLINE void q4k_scales(const uint8_t *p, float *d, float *o) {
    const __fp16 *h = (const __fp16 *)p;
    const float32x4_t D4 = vcvt_f32_f16(vld1_f16(h)), M4 = vcvt_f32_f16(vld1_f16(h + 4));
    uint8x16_t a, b;
    unpack6(p + 16, &a, &b);
    u8_scaled(a, D4, d), u8_scaled(b, D4, d + 16);
    unpack6(p + 40, &a, &b);
    u8_scaled(a, M4, o), u8_scaled(b, M4, o + 16);
}

/* ---- one 32-column block for NV vectors, software-pipelined ----
 * step st: multiplies of vector st (M), reduction of st-1 (R), float update of st-2 (F). */
INLINE void isq_block(const uint8_t *q, float32x4_t d4, float32x4_t o4, const int8_t *xb,
                      const float *db, const float *fs, const int NV, float32x4_t *acc) {
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    int8x16_t lo[4], hi[4];
    for (int i = 0; i < 4; i++) {
        const uint8x16_t b = vld1q_u8(q + i * 16);
        lo[i] = vreinterpretq_s8_u8(vandq_u8(b, m4b));
        hi[i] = vreinterpretq_s8_u8(vshrq_n_u8(b, 4));
    }
#if ISQ_MAGIC
    const int32x4_t magi = vdupq_n_s32(0x4B400000);
    const float32x4_t magf = vreinterpretq_f32_s32(magi);
#endif
    int16x8_t s0 = vdupq_n_s16(0), s1 = s0, s2 = s0, s3 = s0, u0 = s0, u1 = s0, u2 = s0, u3 = s0;
    int16x8_t ra = s0, rb = s0, rc = s0;
    int32x4_t tq = vdupq_n_s32(0), tp = tq;
    float32x4_t cf = vdupq_n_f32(0.0f), mm = cf;
    int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16), nx0 = x0, nx1 = x1;
#pragma GCC unroll 10
    for (int st = 0; st < NV + 2; st++) {
        const int M = st < NV, R = st >= 1 && st <= NV, F = st >= 2, w = st - 2;
        const int L = st + 1 < NV;
        if (M) s0 = vmull_s8(vget_low_s8(lo[0]), vget_low_s8(x0));
        if (R) ra = vpaddq_s16(u0, u1);
        if (M) s1 = vmull_s8(vget_low_s8(lo[1]), vget_low_s8(x0));
        if (R) rb = vpaddq_s16(u2, u3);
        if (M) s2 = vmull_s8(vget_low_s8(lo[2]), vget_low_s8(x0));
#if ISQ_MAGIC
        if (F) cf = vsubq_f32(vreinterpretq_f32_s32(tp), magf);
#else
        if (F) cf = vcvtq_f32_s32(tp);
#endif
        if (M) s3 = vmull_s8(vget_low_s8(lo[3]), vget_low_s8(x0));
        if (F) mm = vmulq_n_f32(d4, db[w]);
        if (M) s0 = vmlal_high_s8(s0, lo[0], x0);
        if (L) nx0 = vld1q_s8(xb + 32 * (st + 1));
        if (M) s1 = vmlal_high_s8(s1, lo[1], x0);
        if (R) rc = vpaddq_s16(ra, rb);
        if (M) s2 = vmlal_high_s8(s2, lo[2], x0);
        if (L) nx1 = vld1q_s8(xb + 32 * (st + 1) + 16);
        if (M) s3 = vmlal_high_s8(s3, lo[3], x0);
        if (F) acc[w] = vfmaq_f32(acc[w], cf, mm);
        if (M) s0 = vmlal_s8(s0, vget_low_s8(hi[0]), vget_low_s8(x1));
        if (R) tq = vpaddlq_s16(rc);
        if (M) s1 = vmlal_s8(s1, vget_low_s8(hi[1]), vget_low_s8(x1));
#if ISQ_MAGIC
        if (R) tq = vaddq_s32(tq, magi);
#endif
        if (M) s2 = vmlal_s8(s2, vget_low_s8(hi[2]), vget_low_s8(x1));
        if (F) acc[w] = vfmsq_n_f32(acc[w], o4, fs[w]);
        if (M) s3 = vmlal_s8(s3, vget_low_s8(hi[3]), vget_low_s8(x1));
        if (M) s0 = vmlal_high_s8(s0, hi[0], x1);
        if (M) s1 = vmlal_high_s8(s1, hi[1], x1);
        if (M) s2 = vmlal_high_s8(s2, hi[2], x1);
        if (M) s3 = vmlal_high_s8(s3, hi[3], x1);
        u0 = s0, u1 = s1, u2 = s2, u3 = s3, tp = tq, x0 = nx0, x1 = nx1;
    }
}

#define ISQ_TILE(NV)                                                                          \
    static void isq_tile_##NV(const uint8_t *p, int nsb, const pf_act *A, float *Y, int rows, \
                              int r, int v0, int kq) {                                        \
        float32x4_t acc[NV];                                                                  \
        float ds[32], os[32];                                                                 \
        for (int v = 0; v < NV; v++) acc[v] = vdupq_n_f32(0.0f);                              \
        for (int sb = 0; sb < nsb; sb++, p += 576) {                                          \
            const __fp16 *sc = (const __fp16 *)p;                                             \
            if (kq) q4k_scales(p, ds, os);                                                    \
            _Pragma("GCC unroll 1") for (int k = 0; k < 8; k++) {                             \
                const float32x4_t d4 = kq ? vld1q_f32(ds + 4 * k)                             \
                                          : vcvt_f32_f16(vld1_f16(sc + 4 * k));               \
                const float32x4_t o4 = kq ? vld1q_f32(os + 4 * k) : vmulq_n_f32(d4, 8.0f);   \
                const size_t base = (size_t)(sb * 8 + k) * A->n + v0;                         \
                isq_block(p + PQ_SB_SCALE_BYTES + k * 64, d4, o4, A->q + base * 32,           \
                          A->d + base, A->fsum + base, NV, acc);                              \
            }                                                                                 \
        }                                                                                     \
        for (int v = 0; v < NV; v++) vst1q_f32(Y + (size_t)(v0 + v) * rows + r, acc[v]);      \
    }
ISQ_TILE(1)
ISQ_TILE(2)
ISQ_TILE(3)
ISQ_TILE(4)
ISQ_TILE(5)
ISQ_TILE(6)
ISQ_TILE(7)

void qv_isq(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1) {
    (void)aux;
    const int nsb = W->cols / PQ_SB_COLS, kq = W->dtype == PQ_Q4K_G32;
    for (int r = r0; r < r1; r += 4) {
        const uint8_t *p = (const uint8_t *)W->data + (size_t)(r / 4) * nsb * 576;
        for (int v = 0; v < n;) { /* passes of equal size (+-1), as kern_quant.c */
            const int left = n - v, np = (left + NVMAX - 1) / NVMAX, nv = (left + np - 1) / np;
            switch (nv) {
            case 7: isq_tile_7(p, nsb, A, Y, W->rows, r, v, kq); break;
            case 6: isq_tile_6(p, nsb, A, Y, W->rows, r, v, kq); break;
            case 5: isq_tile_5(p, nsb, A, Y, W->rows, r, v, kq); break;
            case 4: isq_tile_4(p, nsb, A, Y, W->rows, r, v, kq); break;
            case 3: isq_tile_3(p, nsb, A, Y, W->rows, r, v, kq); break;
            case 2: isq_tile_2(p, nsb, A, Y, W->rows, r, v, kq); break;
            default: isq_tile_1(p, nsb, A, Y, W->rows, r, v, kq); break;
            }
            v += nv;
        }
    }
}
