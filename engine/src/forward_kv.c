/* KV-cache storage formats and the multi-head attention primitives used by forward_attn.c.
 *
 * Like forward_ops.c every function has a NEON body and a bit-identical scalar mirror:
 *   pf_scores4     scores of 4 heads against a run of rows (each dot in pf_dot's lane order);
 *   pf_attn_pv4    P.V of 4 heads over a run of rows and a column range, row order per
 *                  element (pf_attn_pv = zero + the runs in order);
 *   both read fp32 rows or fp16 rows widened in registers;
 *   pf_kv_store / pf_kv_widen  one cache row <-> fp32 for the three storage formats:
 *     0 fp32   [n_kv][max_ctx][hd] floats (the row is used in place, no copy)
 *     1 fp16   [n_kv][max_ctx][hd] IEEE binary16, round to nearest even (FCVTN with FZ=0)
 *     2 int8   [n_kv][max_ctx][hd] int8 then [n_kv][max_ctx] fp32 scales; q = rint(x*127/amax),
 *              value = q * (amax/127), amax = max |x| of the row
 * fp16 / int8 round the cache contents, so they are opt-in (pinfer_params.kv_dtype); the n = 1
 * and batched paths still agree bit for bit because every row goes through the same code. */
#include <float.h>
#include <math.h>
#include <string.h>

#include "forward.h"

#if defined(__ARM_NEON) && !defined(PF_NO_NEON)
#include <arm_neon.h>
#define PF_NEON 1
#else
#define PF_NEON 0
#endif

/* ---- binary16 <-> binary32, mirrors of FCVTN / FCVTL (round to nearest even, denormals) */
static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } x = {f};
    const uint32_t u = x.u, sign = (u >> 16) & 0x8000u, e32 = (u >> 23) & 0xffu;
    uint32_t man = u & 0x7fffffu;
    const int exp = (int)e32 - 127 + 15;
    if (e32 == 0xffu) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u | (man >> 13) : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {                       /* fp16 subnormal (or zero): value = h * 2^-24 */
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp), half = 1u << (shift - 1);
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;  /* may carry into the exponent */
    return (uint16_t)h;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu, u;
    if (exp == 0) {
        if (!man) u = sign;
        else {
            int s = 0;
            while (!(man & 0x400u)) { man <<= 1; s++; }
            u = sign | ((uint32_t)(113 - s) << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31) u = sign | 0x7f800000u | (man << 13);
    else u = sign | ((exp + 112u) << 23) | (man << 13);
    union { uint32_t u; float f; } x = {u};
    return x.f;
}

/* ---- row layout ----------------------------------------------------------------------- */
size_t pf_kv_row_bytes(int dtype, int hd) {
    return dtype == PF_KV_F16 ? (size_t)hd * 2 : dtype == PF_KV_I8 ? (size_t)hd : (size_t)hd * 4;
}

/* Bytes of one layer's K (or V) cache: rows, plus the int8 scales after them. */
size_t pf_kv_bytes(const pinfer_ctx *ctx) {
    const size_t rows = (size_t)ctx->n_kv * (size_t)ctx->max_ctx;
    return rows * pf_kv_row_bytes(ctx->kv_dtype, ctx->hd) +
           (ctx->kv_dtype == PF_KV_I8 ? rows * sizeof(float) : 0);
}

static inline size_t row_index(const pinfer_ctx *ctx, int g, int p) {
    return (size_t)g * (size_t)ctx->max_ctx + (size_t)p;
}

void pf_kv_store(const pinfer_ctx *ctx, void *cache, int g, int p, const float *row) {
    const int hd = ctx->hd;
    const size_t r = row_index(ctx, g, p);
    if (ctx->kv_dtype == PF_KV_F32) {
        memcpy((float *)cache + r * hd, row, (size_t)hd * sizeof(float));
        return;
    }
    if (ctx->kv_dtype == PF_KV_F16) {
        uint16_t *dst = (uint16_t *)cache + r * hd;
        int i = 0;
#if PF_NEON
        for (; i + 4 <= hd; i += 4)
            vst1_f16((__fp16 *)(dst + i), vcvt_f16_f32(vld1q_f32(row + i)));
#endif
        for (; i < hd; i++) dst[i] = f32_to_f16(row[i]);
        return;
    }
    /* int8: absmax is exact in any order; q = rint(x * (127 / amax)) rounds to nearest even in
     * both bodies (vcvtnq / rintf under the default rounding mode) */
    int8_t *dst = (int8_t *)cache + r * hd;
    float *scale = (float *)((int8_t *)cache + (size_t)ctx->n_kv * ctx->max_ctx * hd) + r;
    float amax = 0.0f;
    for (int i = 0; i < hd; i++) amax = fmaxf(amax, fabsf(row[i]));
    /* inv stays finite (a row of denormals quantises to zeros, like a zero row) */
    const float inv = amax > 127.0f / FLT_MAX ? 127.0f / amax : 0.0f;
    *scale = amax / 127.0f;
    int i = 0;
#if PF_NEON
    const float32x4_t iv = vdupq_n_f32(inv);
    for (; i + 8 <= hd; i += 8) {
        const int32x4_t a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + i), iv));
        const int32x4_t b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + i + 4), iv));
        vst1_s8(dst + i, vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))));
    }
#endif
    for (; i < hd; i++) {
        /* clamp in float before the conversion: (int) of a NaN (a NaN or inf activation)
         * is undefined; NaN becomes 0 like FCVTNS above, finite values are unchanged */
        const float y = rintf(row[i] * inv);
        dst[i] = (int8_t)(y > 127.0f ? 127 : y < -127.0f ? -127 : y == y ? (int)y : 0);
    }
}

/* Zero rows [p0, p1) of every group (and their int8 scales): positions whose values were
 * never computed must not feed garbage into an attention pass. */
void pf_kv_zero_rows(const pinfer_ctx *ctx, void *cache, int p0, int p1) {
    if (p1 <= p0) return;
    const size_t rb = pf_kv_row_bytes(ctx->kv_dtype, ctx->hd);
    for (int g = 0; g < ctx->n_kv; g++) {
        memset((char *)cache + row_index(ctx, g, p0) * rb, 0, (size_t)(p1 - p0) * rb);
        if (ctx->kv_dtype == PF_KV_I8)
            memset((float *)((char *)cache + (size_t)ctx->n_kv * ctx->max_ctx * ctx->hd) +
                       row_index(ctx, g, p0), 0, (size_t)(p1 - p0) * sizeof(float));
    }
}

/* fp32 values of columns [d0, d1) of row (g, p) into buf[d0, d1) (fp16: exact widening;
 * int8: q * scale; fp32: a copy). Elementwise, so any column range gives the same values. */
void pf_kv_widen(const pinfer_ctx *ctx, const void *cache, int g, int p, int d0, int d1,
                 float *buf) {
    const int hd = ctx->hd;
    const size_t r = row_index(ctx, g, p);
    int i = d0;
    if (ctx->kv_dtype == PF_KV_F32) {
        memcpy(buf + d0, (const float *)cache + r * hd + d0, (size_t)(d1 - d0) * sizeof(float));
        return;
    }
    if (ctx->kv_dtype == PF_KV_F16) {
        const uint16_t *src = (const uint16_t *)cache + r * hd;
#if PF_NEON
        for (; i + 4 <= d1; i += 4)
            vst1q_f32(buf + i, vcvt_f32_f16(vld1_f16((const __fp16 *)(src + i))));
#endif
        for (; i < d1; i++) buf[i] = f16_to_f32(src[i]);
        return;
    }
    const int8_t *src = (const int8_t *)cache + r * hd;
    const float s = ((const float *)((const int8_t *)cache +
                                     (size_t)ctx->n_kv * ctx->max_ctx * hd))[r];
#if PF_NEON
    const float32x4_t sv = vdupq_n_f32(s);
    for (; i + 8 <= d1; i += 8) {
        const int16x8_t w = vmovl_s8(vld1_s8(src + i));
        vst1q_f32(buf + i, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), sv));
        vst1q_f32(buf + i + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))), sv));
    }
#endif
    for (; i < d1; i++) buf[i] = (float)src[i] * s;
}

/* Raw storage of row (g, p). (Software prefetch of the next rows was tried and lost 1-6%,
 * CONTEXT.md §3d: the hardware prefetcher already follows the row streams.) */
const void *pf_kv_row(const pinfer_ctx *ctx, const void *cache, int g, int p) {
    return (const char *)cache + row_index(ctx, g, p) * pf_kv_row_bytes(ctx->kv_dtype, ctx->hd);
}

/* ---- attention primitives ------------------------------------------------------------- */
/* Both kernels read np rows of n elements stored back to back, fp32 or (f16 != 0) binary16
 * widened in registers (FCVTL is exact, so this equals working on the widened rows). What
 * the A72 needed (CONTEXT.md §9): named accumulators (with an accumulator array GCC stored
 * all 16 of them to the stack in every iteration of the former fp16 dot), FADDP for the
 * final sums (15 instead of ~120 scalar operations per row), and for streamed K rows the 4
 * queries behind one base pointer (4 query streams next to a K stream from L2 cost ~50%). */

/* qi[(j / 4) * 16 + 4 * h + j % 4] = q[h][j]: 4 queries interleaved by 4 elements, the layout
 * pf_scores4 reads with qi != NULL. */
void pf_q4_interleave(const float *const q[4], int n, float *qi) {
    for (int j = 0; j < n; j += 4)
        for (int h = 0; h < 4; h++)
            for (int l = 0; l < 4 && j + l < n; l++) qi[(size_t)j * 4 + 4 * h + l] = q[h][j + l];
}

#if PF_NEON
#define AI static inline __attribute__((always_inline))

/* the 16 elements at i of a row as four float32x4_t */
#define LOAD16(f16, row, i, k0, k1, k2, k3)                                              \
    do {                                                                                 \
        if (f16) {                                                                       \
            const float16x8_t x_ = vld1q_f16((const __fp16 *)(row) + (i));               \
            const float16x8_t y_ = vld1q_f16((const __fp16 *)(row) + (i) + 8);           \
            k0 = vcvt_f32_f16(vget_low_f16(x_)); k1 = vcvt_high_f32_f16(x_);             \
            k2 = vcvt_f32_f16(vget_low_f16(y_)); k3 = vcvt_high_f32_f16(y_);             \
        } else {                                                                         \
            k0 = vld1q_f32((const float *)(row) + (i));                                  \
            k1 = vld1q_f32((const float *)(row) + (i) + 4);                              \
            k2 = vld1q_f32((const float *)(row) + (i) + 8);                              \
            k3 = vld1q_f32((const float *)(row) + (i) + 12);                             \
        }                                                                                \
    } while (0)

/* FADDP adds lanes (0, 1) and (2, 3) of each operand: lane l of SUM4(a, b, c, d) is
 * (x0 + x1) + (x2 + x3) of the l-th operand x */
#define SUM4(a, b, c, d) vpaddq_f32(vpaddq_f32(a, b), vpaddq_f32(c, d))

/* Lane h = q_h . row over the first n16 elements in pf_dot's order: 16 interleaved lanes,
 * then (hsum(c0) + hsum(c1)) + (hsum(c2) + hsum(c3)) with hsum(v) = (v0 + v1) + (v2 + v3).
 * The 15 FADDPs are exactly these scalar adds of the 4 heads, same pairs, same order.
 * Queries: interleaved (qi, one base pointer) or 4 pointers q[h] (qi == NULL). */
AI float32x4_t dot4_row(const float *qi, const float *const q[4], const void *row, int f16,
                        int n16) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0;
    float32x4_t b3 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0, d0 = a0, d1 = a0, d2 = a0, d3 = a0;
    for (int i = 0; i < n16; i += 16) {
        float32x4_t k0, k1, k2, k3;
        LOAD16(f16, row, i, k0, k1, k2, k3);
        if (qi) {   /* elements i + 4l.. of head h at qi + 4i + 16l + 4h */
            const float *z = qi + (size_t)i * 4;
#define DOT4_COL(k, o, x0, x1, x2, x3)                                \
            x0 = vaddq_f32(x0, vmulq_f32(vld1q_f32(z + o), k));       \
            x1 = vaddq_f32(x1, vmulq_f32(vld1q_f32(z + o + 4), k));   \
            x2 = vaddq_f32(x2, vmulq_f32(vld1q_f32(z + o + 8), k));   \
            x3 = vaddq_f32(x3, vmulq_f32(vld1q_f32(z + o + 12), k));
            DOT4_COL(k0, 0, a0, b0, c0, d0) DOT4_COL(k1, 16, a1, b1, c1, d1)
            DOT4_COL(k2, 32, a2, b2, c2, d2) DOT4_COL(k3, 48, a3, b3, c3, d3)
#undef DOT4_COL
        } else {
#define DOT4_HEAD(q, x0, x1, x2, x3)                                  \
            x0 = vaddq_f32(x0, vmulq_f32(vld1q_f32(q + i), k0));      \
            x1 = vaddq_f32(x1, vmulq_f32(vld1q_f32(q + i + 4), k1));  \
            x2 = vaddq_f32(x2, vmulq_f32(vld1q_f32(q + i + 8), k2));  \
            x3 = vaddq_f32(x3, vmulq_f32(vld1q_f32(q + i + 12), k3));
            DOT4_HEAD(q[0], a0, a1, a2, a3) DOT4_HEAD(q[1], b0, b1, b2, b3)
            DOT4_HEAD(q[2], c0, c1, c2, c3) DOT4_HEAD(q[3], d0, d1, d2, d3)
#undef DOT4_HEAD
        }
    }
    return vpaddq_f32(vpaddq_f32(SUM4(a0, a1, a2, a3), SUM4(b0, b1, b2, b3)),
                      vpaddq_f32(SUM4(c0, c1, c2, c3), SUM4(d0, d1, d2, d3)));
}

AI void scores4(const float *qi, const float *const q[4], const void *rows, int f16, int np,
                int n, float scale, float *const att[4]) {
    const int n16 = n / 16 * 16;
    const float32x4_t sv = vdupq_n_f32(scale);
    for (int p = 0; p < np; p++) {
        const void *row = f16 ? (const void *)((const uint16_t *)rows + (size_t)p * n)
                              : (const void *)((const float *)rows + (size_t)p * n);
        float t[4];
        for (int h = 0; h < 4; h++) {   /* sequential tail, as in pf_dot: result s + t */
            float acc = 0.0f;
            for (int j = n16; j < n; j++)
                acc = acc + q[h][j] * (f16 ? f16_to_f32(((const uint16_t *)row)[j])
                                           : ((const float *)row)[j]);
            t[h] = acc;
        }
        const float32x4_t s = vmulq_f32(vaddq_f32(dot4_row(qi, q, row, f16, n16),
                                                  vld1q_f32(t)), sv);
        vst1q_lane_f32(att[0] + p, s, 0);
        vst1q_lane_f32(att[1] + p, s, 1);
        vst1q_lane_f32(att[2] + p, s, 2);
        vst1q_lane_f32(att[3] + p, s, 3);
    }
}
#endif

/* att[h][p] = (q[h] . row p) * scale for h < 4 and p < np, each dot exactly pf_dot. qi: the
 * same queries interleaved (pf_q4_interleave) or NULL; q[h] is always needed. */
void pf_scores4(const float *qi, const float *const q[4], const void *rows, int f16, int np,
                int n, float scale, float *const att[4]) {
#if PF_NEON   /* four copies with constant f16 / layout */
    if (qi) {
        if (f16) scores4(qi, q, rows, 1, np, n, scale, att);
        else scores4(qi, q, rows, 0, np, n, scale, att);
    } else {
        if (f16) scores4(NULL, q, rows, 1, np, n, scale, att);
        else scores4(NULL, q, rows, 0, np, n, scale, att);
    }
#else
    (void)qi;
    float k[PF_KV_MAX_HD];
    for (int p = 0; p < np; p++) {
        const float *row = (const float *)rows + (size_t)p * n;
        if (f16) {
            const uint16_t *r16 = (const uint16_t *)rows + (size_t)p * n;
            for (int i = 0; i < n; i++) k[i] = f16_to_f32(r16[i]);
            row = k;
        }
        for (int h = 0; h < 4; h++) att[h][p] = pf_dot(q[h], row, n) * scale;
    }
#endif
}

#if PF_NEON
/* one row of P.V for the 16 columns at d: acc_h += v * P_h (P_h = p_h[row] broadcast) */
#define PV4_ROW(row, P0, P1, P2, P3)                                                      \
    do {                                                                                  \
        float32x4_t v0, v1, v2, v3;                                                       \
        LOAD16(f16, row, d, v0, v1, v2, v3);                                              \
        a0 = vaddq_f32(a0, vmulq_f32(v0, P0)); a1 = vaddq_f32(a1, vmulq_f32(v1, P0));     \
        a2 = vaddq_f32(a2, vmulq_f32(v2, P0)); a3 = vaddq_f32(a3, vmulq_f32(v3, P0));     \
        b0 = vaddq_f32(b0, vmulq_f32(v0, P1)); b1 = vaddq_f32(b1, vmulq_f32(v1, P1));     \
        b2 = vaddq_f32(b2, vmulq_f32(v2, P1)); b3 = vaddq_f32(b3, vmulq_f32(v3, P1));     \
        c0 = vaddq_f32(c0, vmulq_f32(v0, P2)); c1 = vaddq_f32(c1, vmulq_f32(v1, P2));     \
        c2 = vaddq_f32(c2, vmulq_f32(v2, P2)); c3 = vaddq_f32(c3, vmulq_f32(v3, P2));     \
        d0 = vaddq_f32(d0, vmulq_f32(v0, P3)); d1 = vaddq_f32(d1, vmulq_f32(v1, P3));     \
        d2 = vaddq_f32(d2, vmulq_f32(v2, P3)); d3 = vaddq_f32(d3, vmulq_f32(v3, P3));     \
    } while (0)

/* Per 16 columns: the 4 heads' outputs in registers across the rows, one row per iteration
 * (the variants measured on the Pi - 4 rows per iteration with p as lanes, the next row
 * loaded ahead, two register sets - spilled or added moves and were no faster, §9) */
AI void pv4(float *const o[4], const float *const p[4], int np, const void *v, int f16, int n,
            int dbeg, int dend) {
    const float *p0 = p[0], *p1 = p[1], *p2 = p[2], *p3 = p[3];
    for (int d = dbeg; d + 16 <= dend; d += 16) {
        float32x4_t a0 = vld1q_f32(o[0] + d), a1 = vld1q_f32(o[0] + d + 4);
        float32x4_t a2 = vld1q_f32(o[0] + d + 8), a3 = vld1q_f32(o[0] + d + 12);
        float32x4_t b0 = vld1q_f32(o[1] + d), b1 = vld1q_f32(o[1] + d + 4);
        float32x4_t b2 = vld1q_f32(o[1] + d + 8), b3 = vld1q_f32(o[1] + d + 12);
        float32x4_t c0 = vld1q_f32(o[2] + d), c1 = vld1q_f32(o[2] + d + 4);
        float32x4_t c2 = vld1q_f32(o[2] + d + 8), c3 = vld1q_f32(o[2] + d + 12);
        float32x4_t d0 = vld1q_f32(o[3] + d), d1 = vld1q_f32(o[3] + d + 4);
        float32x4_t d2 = vld1q_f32(o[3] + d + 8), d3 = vld1q_f32(o[3] + d + 12);
        for (int j = 0; j < np; j++) {
            const float32x4_t P0 = vld1q_dup_f32(p0 + j), P1 = vld1q_dup_f32(p1 + j);
            const float32x4_t P2 = vld1q_dup_f32(p2 + j), P3 = vld1q_dup_f32(p3 + j);
            PV4_ROW(f16 ? (const void *)((const uint16_t *)v + (size_t)j * n)
                        : (const void *)((const float *)v + (size_t)j * n), P0, P1, P2, P3);
        }
        vst1q_f32(o[0] + d, a0); vst1q_f32(o[0] + d + 4, a1);
        vst1q_f32(o[0] + d + 8, a2); vst1q_f32(o[0] + d + 12, a3);
        vst1q_f32(o[1] + d, b0); vst1q_f32(o[1] + d + 4, b1);
        vst1q_f32(o[1] + d + 8, b2); vst1q_f32(o[1] + d + 12, b3);
        vst1q_f32(o[2] + d, c0); vst1q_f32(o[2] + d + 4, c1);
        vst1q_f32(o[2] + d + 8, c2); vst1q_f32(o[2] + d + 12, c3);
        vst1q_f32(o[3] + d, d0); vst1q_f32(o[3] + d + 4, d1);
        vst1q_f32(o[3] + d + 8, d2); vst1q_f32(o[3] + d + 12, d3);
    }
}
#undef PV4_ROW
#undef LOAD16
#undef SUM4
#undef AI
#endif

/* P.V of 4 heads over np rows, columns [d0, d1) (d0 a multiple of 16): for h < 4,
 * o[h][d] = o[h][d] + p[h][0] * v[0][d] + p[h][1] * v[1][d] + ... in row order from o's
 * current value (zero o first, then call once per block of rows, in order) - per element
 * the former pf_attn_pv_blk. The output chunks (16 columns x 4 heads) stay in registers
 * across the rows and each load of v feeds all 4 heads. */
void pf_attn_pv4(float *const o[4], const float *const p[4], int np, const void *v, int f16,
                 int n, int d0, int d1) {
    int d = d0;
#if PF_NEON
    if (f16) pv4(o, p, np, v, 1, n, d0, d1);
    else pv4(o, p, np, v, 0, n, d0, d1);
    d = d0 + (d1 - d0) / 16 * 16;
#endif
    for (; d < d1; d++)
        for (int j = 0; j < np; j++) {
            const float x = f16 ? f16_to_f32(((const uint16_t *)v)[(size_t)j * n + d])
                                : ((const float *)v)[(size_t)j * n + d];
            for (int h = 0; h < 4; h++) o[h][d] = o[h][d] + p[h][j] * x;
        }
}
