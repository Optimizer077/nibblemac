/* Quantized matvec / small-batch matmul kernels for the A72 (no SDOT): dtypes Q8_G32,
 * Q4_G32, Q4K_G32 (layouts in pq_quant.h / FORMAT.md).
 *
 * Method: activations are quantized to int8 per 32-column block once per call; the inner
 * product of a 32-column block is SMULL/SMLAL(2) into int16 lanes, reduced across the
 * 4 rows of a tile with pairwise adds (so each 4-row result lands in one int32x4 with
 * lane = row, no horizontal add per row), then converted once and scaled with
 * (weight scale x activation scale) into an fp32 accumulator.
 *
 * int16 headroom: Q8 puts 2 products (|w|,|x| <= 127 -> <= 32258) in a lane; Q4 puts 4
 * products (u4 <= 15) in a lane, and the 3 pairwise adds of the 4-row reduction keep the
 * worst case at 16 * 15 * 127 = 30480 < 32767 before widening.
 */
#include <arm_neon.h>

#include "kern.h"
#include "pq_quant.h"

#define INLINE static inline __attribute__((always_inline))

INLINE void prefetch_sb(const uint8_t *p, int bytes, int pf) {
    if (pf)
        for (int l = 0; l < bytes; l += 64) __builtin_prefetch(p + pf + l, 0, 0);
}

INLINE void store4(const float32x4_t *acc, float *Y, int rows, int r, int v0, int nv) {
    for (int v = 0; v < nv; v++) vst1q_f32(Y + (size_t)(v0 + v) * rows + r, acc[v]);
}

/* fp32 scales (or mins) of the 4 rows of group g: 4 consecutive fp16 in the scale line. */
INLINE float32x4_t scale4(const __fp16 *line, int g) { return vcvt_f32_f16(vld1_f16(line + 4 * g)); }

/* The block loop is fully unrolled for single-vector passes (short dependency chains need
 * many blocks in flight) and kept rolled for multi-vector passes (where unrolling made GCC
 * spill the 8 accumulators; the vectors already provide the parallelism). */
#define BLOCK_LOOP(NV, K, BODY)                                                       \
    if ((NV) == 1) {                                                                  \
        _Pragma("GCC unroll 8") for (int K = 0; K < 8; K++) BODY                      \
    } else {                                                                          \
        _Pragma("GCC unroll 1") for (int K = 0; K < 8; K++) BODY                      \
    }

/* ---------------------------------------------------------------- Q8_G32 */
INLINE void q8_block(const int8_t *q, float32x4_t d4, int blk, const pf_act *A, int v0, int nv,
                     float32x4_t *acc) {
    const size_t base = (size_t)blk * A->n + v0; /* block-major activations */
    const int8_t *xb = A->q + base * 32;
    const float *db = A->d + base;
    int8x16_t w[4][2];
    for (int i = 0; i < 4; i++) {
        w[i][0] = vld1q_s8(q + 32 * i);
        w[i][1] = vld1q_s8(q + 32 * i + 16);
    }
#pragma GCC unroll 8
    for (int v = 0; v < nv; v++) {
        const int8x16_t x0 = vld1q_s8(xb + 32 * v), x1 = vld1q_s8(xb + 32 * v + 16);
        int32x4_t s[4];
        for (int i = 0; i < 4; i++) {
            int16x8_t a = vmull_s8(vget_low_s8(w[i][0]), vget_low_s8(x0));
            a = vmlal_high_s8(a, w[i][0], x0);
            int16x8_t b = vmull_s8(vget_low_s8(w[i][1]), vget_low_s8(x1));
            b = vmlal_high_s8(b, w[i][1], x1);
            s[i] = vpadalq_s16(vpaddlq_s16(a), b);
        }
        const int32x4_t t = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
        acc[v] = vfmaq_f32(acc[v], vcvtq_f32_s32(t), vmulq_n_f32(d4, db[v]));
    }
}

INLINE void q8_tile(const uint8_t *p, int nsb, const pf_act *A, float *Y, int rows, int r,
                    int v0, int nv, int pf) {
    float32x4_t acc[KERN_MAX_NV];
    for (int v = 0; v < nv; v++) acc[v] = vdupq_n_f32(0.0f);
    for (int sb = 0; sb < nsb; sb++, p += 1088) {
        prefetch_sb(p, 1088, pf);
        const __fp16 *sc = (const __fp16 *)p;
        const int8_t *codes = (const int8_t *)(p + PQ_SB_SCALE_BYTES);
        BLOCK_LOOP(nv, g, { q8_block(codes + g * 128, scale4(sc, g), sb * 8 + g, A, v0, nv, acc); })
    }
    store4(acc, Y, rows, r, v0, nv);
}

/* ---------------------------------------------------------------- Q4_G32 / Q4K_G32 */
/* Both store block k (32 columns) as 4 rows x 16 B of nibbles (low = col j, high = j + 16).
 * A row's block value is d * sum(u * x) - o * sum(x) with per-(row, block) scale d and
 * offset o: Q4 has u = q + 8, so o = 8 d; Q4K has d = D * s6, o = M * m6. With one vector
 * the offset term goes to a separate accumulator `corr` (off the reduction's critical
 * path); with several vectors it is folded into acc directly (no extra registers). */
INLINE void q4_block(const uint8_t *q, float32x4_t d4, float32x4_t o4, int blk,
                     const pf_act *A, int v0, int nv, float32x4_t *acc, float32x4_t *corr) {
    const size_t base = (size_t)blk * A->n + v0; /* block-major activations */
    const int8_t *xb = A->q + base * 32;
    const float *db = A->d + base, *fs = A->fsum + base;
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    int8x16_t lo[4], hi[4];
    for (int i = 0; i < 4; i++) {
        const uint8x16_t b = vld1q_u8(q + i * 16);
        lo[i] = vreinterpretq_s8_u8(vandq_u8(b, m4b));
        hi[i] = vreinterpretq_s8_u8(vshrq_n_u8(b, 4));
    }
#pragma GCC unroll 8
    for (int v = 0; v < nv; v++) {
        const int8x16_t x0 = vld1q_s8(xb + 32 * v), x1 = vld1q_s8(xb + 32 * v + 16);
        int16x8_t s[4];
        for (int i = 0; i < 4; i++) {
            int16x8_t a = vmull_s8(vget_low_s8(lo[i]), vget_low_s8(x0));
            a = vmlal_high_s8(a, lo[i], x0);
            a = vmlal_s8(a, vget_low_s8(hi[i]), vget_low_s8(x1));
            s[i] = vmlal_high_s8(a, hi[i], x1);
        }
        const int32x4_t t = vpaddlq_s16(vpaddq_s16(vpaddq_s16(s[0], s[1]),
                                                   vpaddq_s16(s[2], s[3])));
        acc[v] = vfmaq_f32(acc[v], vcvtq_f32_s32(t), vmulq_n_f32(d4, db[v]));
        if (nv == 1) corr[0] = vfmaq_n_f32(corr[0], o4, fs[0]);
        else acc[v] = vfmsq_n_f32(acc[v], o4, fs[v]);
    }
}

/* 32 packed 6-bit values (see kern_get6) -> two u8x16: values 0..15 and 16..31. */
INLINE void unpack6(const uint8_t *p, uint8x16_t *a, uint8x16_t *b) {
    const uint8x16_t lo = vld1q_u8(p);
    const uint8x8_t hb = vld1_u8(p + 16), m3 = vdup_n_u8(3);
    const uint8x16_t h01 = vcombine_u8(vand_u8(hb, m3), vand_u8(vshr_n_u8(hb, 2), m3));
    const uint8x16_t h23 = vcombine_u8(vand_u8(vshr_n_u8(hb, 4), m3), vshr_n_u8(hb, 6));
    *a = vorrq_u8(vandq_u8(lo, vdupq_n_u8(15)), vshlq_n_u8(h01, 4));
    *b = vorrq_u8(vshrq_n_u8(lo, 4), vshlq_n_u8(h23, 4));
}

/* 16 u8 (4 blocks x 4 rows) times a per-row fp32 multiplier -> 16 floats. */
INLINE void u8_scaled(uint8x16_t v, float32x4_t mul, float *out) {
    const uint16x8_t l = vmovl_u8(vget_low_u8(v)), h = vmovl_high_u8(v);
    vst1q_f32(out + 0, vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(l))), mul));
    vst1q_f32(out + 4, vmulq_f32(vcvtq_f32_u32(vmovl_high_u16(l)), mul));
    vst1q_f32(out + 8, vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(h))), mul));
    vst1q_f32(out + 12, vmulq_f32(vcvtq_f32_u32(vmovl_high_u16(h)), mul));
}

/* Q4K: decode one superblock's 6-bit sub-scales / sub-mins into fp32 d[8][4], o[8][4]. */
INLINE void q4k_scales(const uint8_t *p, float *d, float *o) {
    const __fp16 *h = (const __fp16 *)p;
    const float32x4_t D4 = vcvt_f32_f16(vld1_f16(h)), M4 = vcvt_f32_f16(vld1_f16(h + 4));
    uint8x16_t a, b;
    unpack6(p + 16, &a, &b);
    u8_scaled(a, D4, d), u8_scaled(b, D4, d + 16);
    unpack6(p + 40, &a, &b);
    u8_scaled(a, M4, o), u8_scaled(b, M4, o + 16);
}

INLINE void q4_tile(const uint8_t *p, int nsb, const pf_act *A, float *Y, int rows, int r,
                    int v0, int nv, int pf, int kq) {
    float32x4_t acc[KERN_MAX_NV], corr[1] = {vdupq_n_f32(0.0f)};
    float ds[32], os[32]; /* Q4K scales / offsets of the current superblock, [block][row] */
    for (int v = 0; v < nv; v++) acc[v] = vdupq_n_f32(0.0f);
    for (int sb = 0; sb < nsb; sb++, p += 576) {
        prefetch_sb(p, 576, pf);
        const __fp16 *sc = (const __fp16 *)p;
        const uint8_t *codes = p + PQ_SB_SCALE_BYTES;
        if (kq) q4k_scales(p, ds, os);
        BLOCK_LOOP(nv, k, {
            const float32x4_t d4 = kq ? vld1q_f32(ds + 4 * k) : scale4(sc, k);
            const float32x4_t o4 = kq ? vld1q_f32(os + 4 * k) : vmulq_n_f32(d4, 8.0f);
            q4_block(codes + k * 64, d4, o4, sb * 8 + k, A, v0, nv, acc, corr);
        })
    }
    if (nv == 1) acc[0] = vsubq_f32(acc[0], corr[0]);
    store4(acc, Y, rows, r, v0, nv);
}

/* Split n vectors into passes of 4/3/2/1 over the same tile (later passes hit L1).
 * Passes of 8 were measured slower than 2 x 4 (register spills; KERNELS.md v8). */
#define VECTOR_PASSES(CALL)                                                           \
    for (int v = 0; v < n;) {                                                         \
        const int left = n - v;                                                       \
        if (left >= 4) { CALL(4); v += 4; }                                           \
        else if (left == 3) { CALL(3); v += 3; }                                      \
        else if (left >= 2) { CALL(2); v += 2; }                                      \
        else { CALL(1); v += 1; }                                                     \
    }

void kern_q8(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1) {
    (void)X;
    const int nsb = W->cols / PQ_SB_COLS, pf = kern_prefetch;
    for (int r = r0; r < r1; r += 4) {
        const uint8_t *p = (const uint8_t *)W->data + (size_t)(r / 4) * nsb * 1088;
#define Q8CALL(NV) q8_tile(p, nsb, A, Y, W->rows, r, v, NV, pf)
        VECTOR_PASSES(Q8CALL)
    }
}

void kern_q4(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1) {
    (void)X;
    const int nsb = W->cols / PQ_SB_COLS, pf = kern_prefetch;
    for (int r = r0; r < r1; r += 4) {
        const uint8_t *p = (const uint8_t *)W->data + (size_t)(r / 4) * nsb * 576;
#define Q4CALL(NV) q4_tile(p, nsb, A, Y, W->rows, r, v, NV, pf, 0)
        VECTOR_PASSES(Q4CALL)
    }
}

void kern_q4k(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1) {
    (void)X;
    const int nsb = W->cols / PQ_SB_COLS, pf = kern_prefetch;
    for (int r = r0; r < r1; r += 4) {
        const uint8_t *p = (const uint8_t *)W->data + (size_t)(r / 4) * nsb * 576;
#define Q4KCALL(NV) q4_tile(p, nsb, A, Y, W->rows, r, v, NV, pf, 1)
        VECTOR_PASSES(Q4KCALL)
    }
}
