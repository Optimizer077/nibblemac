/* KV-cache storage formats and the multi-head attention primitives used by forward_attn.c.
 *
 * Like forward_ops.c every function has a NEON body and a bit-identical scalar mirror:
 *   pf_dot4        four dot products against one shared row, each in pf_dot's lane order;
 *   pf_attn_pv_blk P.V over one L1 block of rows for 1-2 heads, row order per element
 *                  (pf_attn_pv = zero + the blocks in order);
 *   pf_kv_store / pf_kv_load   one cache row <-> fp32 for the three storage formats:
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
        int v = (int)rintf(row[i] * inv);
        dst[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
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

/* fp32 view of row (g, p): the cache row itself for fp32, else the row widened into buf. */
const float *pf_kv_load(const pinfer_ctx *ctx, const void *cache, int g, int p, float *buf) {
    const int hd = ctx->hd;
    const size_t r = row_index(ctx, g, p);
    if (ctx->kv_dtype == PF_KV_F32) return (const float *)cache + r * hd;
    int i = 0;
    if (ctx->kv_dtype == PF_KV_F16) {
        const uint16_t *src = (const uint16_t *)cache + r * hd;
#if PF_NEON
        for (; i + 4 <= hd; i += 4)
            vst1q_f32(buf + i, vcvt_f32_f16(vld1_f16((const __fp16 *)(src + i))));
#endif
        for (; i < hd; i++) buf[i] = f16_to_f32(src[i]);
        return buf;
    }
    const int8_t *src = (const int8_t *)cache + r * hd;
    const float s = ((const float *)((const int8_t *)cache +
                                     (size_t)ctx->n_kv * ctx->max_ctx * hd))[r];
#if PF_NEON
    const float32x4_t sv = vdupq_n_f32(s);
    for (; i + 8 <= hd; i += 8) {
        const int16x8_t w = vmovl_s8(vld1_s8(src + i));
        vst1q_f32(buf + i, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), sv));
        vst1q_f32(buf + i + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))), sv));
    }
#endif
    for (; i < hd; i++) buf[i] = (float)src[i] * s;
    return buf;
}

/* ---- attention primitives ------------------------------------------------------------- */
#if PF_NEON
static inline float hsum(float32x4_t v) {
    return (vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1)) +
           (vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3));
}
#endif

/* out[h] = q[h] . k for h < 4, with k loaded once. Per head exactly pf_dot: 16 interleaved
 * lanes over the first n/16*16 elements combined ((c0..3)+(c4..7))+((c8..11)+(c12..15)),
 * then the sequential tail (the product q*k is commutative bit for bit). */
void pf_dot4(const float *const q[4], const float *k, int n, float *out) {
#if PF_NEON
    float32x4_t c[4][4];
    for (int h = 0; h < 4; h++)
        for (int l = 0; l < 4; l++) c[h][l] = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const float32x4_t k0 = vld1q_f32(k + i), k1 = vld1q_f32(k + i + 4);
        const float32x4_t k2 = vld1q_f32(k + i + 8), k3 = vld1q_f32(k + i + 12);
#define DOT4_HEAD(h) \
        c[h][0] = vaddq_f32(c[h][0], vmulq_f32(vld1q_f32(q[h] + i), k0)); \
        c[h][1] = vaddq_f32(c[h][1], vmulq_f32(vld1q_f32(q[h] + i + 4), k1)); \
        c[h][2] = vaddq_f32(c[h][2], vmulq_f32(vld1q_f32(q[h] + i + 8), k2)); \
        c[h][3] = vaddq_f32(c[h][3], vmulq_f32(vld1q_f32(q[h] + i + 12), k3));
        DOT4_HEAD(0) DOT4_HEAD(1) DOT4_HEAD(2) DOT4_HEAD(3)
#undef DOT4_HEAD
    }
    for (int h = 0; h < 4; h++) {
        const float s = (hsum(c[h][0]) + hsum(c[h][1])) + (hsum(c[h][2]) + hsum(c[h][3]));
        float t = 0.0f;
        for (int j = i; j < n; j++) t = t + q[h][j] * k[j];
        out[h] = s + t;
    }
#else
    for (int h = 0; h < 4; h++) out[h] = pf_dot(q[h], k, n);
#endif
}

/* pf_dot4 against an fp16 row, widened in registers: the widening is exact, so the result is
 * bit-identical to pf_dot4 on the widened row, without its store + reload through L1. */
void pf_dot4_f16(const float *const q[4], const uint16_t *k16, int n, float *out) {
#if PF_NEON
    float32x4_t c[4][4];
    for (int h = 0; h < 4; h++)
        for (int l = 0; l < 4; l++) c[h][l] = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const float16x8_t a = vld1q_f16((const __fp16 *)(k16 + i));
        const float16x8_t b = vld1q_f16((const __fp16 *)(k16 + i + 8));
        const float32x4_t k0 = vcvt_f32_f16(vget_low_f16(a)), k1 = vcvt_high_f32_f16(a);
        const float32x4_t k2 = vcvt_f32_f16(vget_low_f16(b)), k3 = vcvt_high_f32_f16(b);
#define DOT4_HEAD(h) \
        c[h][0] = vaddq_f32(c[h][0], vmulq_f32(vld1q_f32(q[h] + i), k0)); \
        c[h][1] = vaddq_f32(c[h][1], vmulq_f32(vld1q_f32(q[h] + i + 4), k1)); \
        c[h][2] = vaddq_f32(c[h][2], vmulq_f32(vld1q_f32(q[h] + i + 8), k2)); \
        c[h][3] = vaddq_f32(c[h][3], vmulq_f32(vld1q_f32(q[h] + i + 12), k3));
        DOT4_HEAD(0) DOT4_HEAD(1) DOT4_HEAD(2) DOT4_HEAD(3)
#undef DOT4_HEAD
    }
    for (int h = 0; h < 4; h++) {
        const float s = (hsum(c[h][0]) + hsum(c[h][1])) + (hsum(c[h][2]) + hsum(c[h][3]));
        float t = 0.0f;
        for (int j = i; j < n; j++) t = t + q[h][j] * f16_to_f32(k16[j]);
        out[h] = s + t;
    }
#else
    float k[PF_KV_MAX_HD];
    for (int i = 0; i < n; i++) k[i] = f16_to_f32(k16[i]);
    pf_dot4(q, k, n, out);
#endif
}

/* Raw storage of row (g, p). (Software prefetch of the next rows was tried and lost 1-6%,
 * CONTEXT.md §3d: the hardware prefetcher already follows the row streams.) */
const void *pf_kv_row(const pinfer_ctx *ctx, const void *cache, int g, int p) {
    return (const char *)cache + row_index(ctx, g, p) * pf_kv_row_bytes(ctx->kv_dtype, ctx->hd);
}

/* P.V over one block of np <= 8 rows v [np][hd] (contiguous, in L1): o0[d] += sum_j p0[j] *
 * v[j][d] and, if o1 != NULL, the same for (o1, p1), each element accumulated in row order
 * from its current value, exactly like the former row-outer pf_attn_pv. The output chunk
 * (16 floats per head) stays in registers across the block's rows, so o is loaded and stored
 * once per block instead of once per row, and two heads share each load of v. */
void pf_attn_pv_blk(float *o0, const float *p0, float *o1, const float *p1, int np,
                    const float *v, int hd) {
    int d = 0;
#if PF_NEON
    const int d16 = hd / 16 * 16;
    for (; d < d16; d += 16) {
        float32x4_t a0 = vld1q_f32(o0 + d), a1 = vld1q_f32(o0 + d + 4);
        float32x4_t a2 = vld1q_f32(o0 + d + 8), a3 = vld1q_f32(o0 + d + 12);
        if (o1) {
            float32x4_t b0 = vld1q_f32(o1 + d), b1 = vld1q_f32(o1 + d + 4);
            float32x4_t b2 = vld1q_f32(o1 + d + 8), b3 = vld1q_f32(o1 + d + 12);
            for (int j = 0; j < np; j++) {
                const float *vj = v + (size_t)j * hd + d;
                const float32x4_t pa = vdupq_n_f32(p0[j]), pb = vdupq_n_f32(p1[j]);
                const float32x4_t v0 = vld1q_f32(vj), v1 = vld1q_f32(vj + 4);
                const float32x4_t v2 = vld1q_f32(vj + 8), v3 = vld1q_f32(vj + 12);
                a0 = vaddq_f32(a0, vmulq_f32(pa, v0)); b0 = vaddq_f32(b0, vmulq_f32(pb, v0));
                a1 = vaddq_f32(a1, vmulq_f32(pa, v1)); b1 = vaddq_f32(b1, vmulq_f32(pb, v1));
                a2 = vaddq_f32(a2, vmulq_f32(pa, v2)); b2 = vaddq_f32(b2, vmulq_f32(pb, v2));
                a3 = vaddq_f32(a3, vmulq_f32(pa, v3)); b3 = vaddq_f32(b3, vmulq_f32(pb, v3));
            }
            vst1q_f32(o1 + d, b0); vst1q_f32(o1 + d + 4, b1);
            vst1q_f32(o1 + d + 8, b2); vst1q_f32(o1 + d + 12, b3);
        } else {
            for (int j = 0; j < np; j++) {
                const float *vj = v + (size_t)j * hd + d;
                const float32x4_t pa = vdupq_n_f32(p0[j]);
                a0 = vaddq_f32(a0, vmulq_f32(pa, vld1q_f32(vj)));
                a1 = vaddq_f32(a1, vmulq_f32(pa, vld1q_f32(vj + 4)));
                a2 = vaddq_f32(a2, vmulq_f32(pa, vld1q_f32(vj + 8)));
                a3 = vaddq_f32(a3, vmulq_f32(pa, vld1q_f32(vj + 12)));
            }
        }
        vst1q_f32(o0 + d, a0); vst1q_f32(o0 + d + 4, a1);
        vst1q_f32(o0 + d + 8, a2); vst1q_f32(o0 + d + 12, a3);
    }
#endif
    for (; d < hd; d++)
        for (int j = 0; j < np; j++) {
            o0[d] = o0[d] + p0[j] * v[(size_t)j * hd + d];
            if (o1) o1[d] = o1[d] + p1[j] * v[(size_t)j * hd + d];
        }
}
