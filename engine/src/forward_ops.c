/* Vectorized elementwise and reduction primitives of the forward pass.
 *
 * Every function exists twice: a NEON body (aarch64) and a scalar mirror (other targets, or
 * -DPF_NO_NEON). Both perform the same per-element operations in the same order: reductions
 * keep 4 (or 16) interleaved partial sums combined as (s0+s1)+(s2+s3), exactly the lanes of
 * a float32x4_t, and nothing is ever a fused multiply-add (the Makefile compiles with
 * -ffp-contract=off). So the two bodies give bit-identical results, and a row's result never
 * depends on how many rows a call processes: the n = 1 decode path and the batched verify /
 * prefill paths agree bit for bit, which speculative decoding requires (engine/SPEC.md).
 *
 * exp: Cephes-style expf (ln 2 range reduction in two parts, degree-6 polynomial, exponent
 * assembled from the integer part); max relative error vs libm measured in engine/CORE.md.
 * It replaces libm's expf in every SiLU / sigmoid / softmax of the model (~245k per token),
 * in both paths. libm stays for the 16 per-layer decay / beta scalars (forward_lin.c). */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "forward.h"

#if defined(__ARM_NEON) && !defined(PF_NO_NEON)
#include <arm_neon.h>
#define PF_NEON 1
#else
#define PF_NEON 0
#endif

/* ---- exp / SiLU / sigmoid ------------------------------------------------------------ */
#define EXP_HI 88.0f   /* 2^127 * p(r) stays finite */
#define EXP_LO -87.0f  /* 2^-126 is the smallest normal scale */
#define LOG2E 1.44269504088896341f
#define LN2_HI 0.693145751953125f
#define LN2_LO 1.428606765330187e-06f
#define EP0 1.9875691500e-4f
#define EP1 1.3981999507e-3f
#define EP2 8.3334519073e-3f
#define EP3 4.1665795894e-2f
#define EP4 1.6666665459e-1f
#define EP5 5.0000001201e-1f

float pf_expf(float x) {
    x = x < EXP_LO ? EXP_LO : x > EXP_HI ? EXP_HI : x;
    const float fx = rintf(x * LOG2E);
    float r = x - fx * LN2_HI;
    r = r - fx * LN2_LO;
    float p = EP0;
    p = p * r + EP1;
    p = p * r + EP2;
    p = p * r + EP3;
    p = p * r + EP4;
    p = p * r + EP5;
    float y = p * (r * r) + r;
    y = y + 1.0f;
    union { uint32_t u; float f; } s = {(uint32_t)((int)fx + 127) << 23};
    return y * s.f;
}

float pf_silu(float x) { return x / (1.0f + pf_expf(-x)); }
float pf_sigmoid(float x) { return 1.0f / (1.0f + pf_expf(-x)); }

#if PF_NEON
static inline float32x4_t expq(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(EXP_LO)), vdupq_n_f32(EXP_HI));
    const float32x4_t fx = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)));
    float32x4_t r = vsubq_f32(x, vmulq_f32(fx, vdupq_n_f32(LN2_HI)));
    r = vsubq_f32(r, vmulq_f32(fx, vdupq_n_f32(LN2_LO)));
    float32x4_t p = vdupq_n_f32(EP0);
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EP1));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EP2));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EP3));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EP4));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EP5));
    float32x4_t y = vaddq_f32(vmulq_f32(p, vmulq_f32(r, r)), r);
    y = vaddq_f32(y, vdupq_n_f32(1.0f));
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

static inline float32x4_t siluq(float32x4_t x) {
    return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0f), expq(vnegq_f32(x))));
}

static inline float32x4_t sigmoidq(float32x4_t x) {
    return vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), expq(vnegq_f32(x))));
}

/* (l0 + l1) + (l2 + l3): the scalar mirrors combine their 4 partial sums the same way. */
static inline float hsum(float32x4_t v) {
    return (vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1)) +
           (vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3));
}
#endif

/* ---- elementwise --------------------------------------------------------------------- */
/* The exp loops run two vectors per iteration: one exp + divide chain is ~80 cycles of
 * dependent latency on the A72 (measured ~20 cycles per element with one chain), so two
 * independent chains per iteration let the core overlap them. */
void pf_silu_mul(float *a, const float *b, int n) { /* a = silu(a) * b */
    int i = 0;
#if PF_NEON
    for (; i + 8 <= n; i += 8) {
        const float32x4_t s0 = siluq(vld1q_f32(a + i)), s1 = siluq(vld1q_f32(a + i + 4));
        vst1q_f32(a + i, vmulq_f32(s0, vld1q_f32(b + i)));
        vst1q_f32(a + i + 4, vmulq_f32(s1, vld1q_f32(b + i + 4)));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(a + i, vmulq_f32(siluq(vld1q_f32(a + i)), vld1q_f32(b + i)));
#endif
    for (; i < n; i++) a[i] = pf_silu(a[i]) * b[i];
}

void pf_mul_silu(float *y, const float *z, int n) { /* y *= silu(z) */
    int i = 0;
#if PF_NEON
    for (; i + 8 <= n; i += 8) {
        const float32x4_t s0 = siluq(vld1q_f32(z + i)), s1 = siluq(vld1q_f32(z + i + 4));
        vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), s0));
        vst1q_f32(y + i + 4, vmulq_f32(vld1q_f32(y + i + 4), s1));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), siluq(vld1q_f32(z + i))));
#endif
    for (; i < n; i++) y[i] = y[i] * pf_silu(z[i]);
}

void pf_mul_sigmoid(float *y, const float *g, int n) { /* y *= sigmoid(g) */
    int i = 0;
#if PF_NEON
    for (; i + 8 <= n; i += 8) {
        const float32x4_t s0 = sigmoidq(vld1q_f32(g + i)), s1 = sigmoidq(vld1q_f32(g + i + 4));
        vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), s0));
        vst1q_f32(y + i + 4, vmulq_f32(vld1q_f32(y + i + 4), s1));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), sigmoidq(vld1q_f32(g + i))));
#endif
    for (; i < n; i++) y[i] = y[i] * pf_sigmoid(g[i]);
}

void pf_add(float *y, const float *x, int n) { /* y += x */
    int i = 0;
#if PF_NEON
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vaddq_f32(vld1q_f32(y + i), vld1q_f32(x + i)));
#endif
    for (; i < n; i++) y[i] = y[i] + x[i];
}

/* Rotate-half RoPE on the first `rot` dims of one head: pairs (i, i + rot/2). */
void pf_rope(float *v, const float *c, const float *s, int rot) {
    const int h = rot / 2;
    int i = 0;
#if PF_NEON
    for (; i + 4 <= h; i += 4) {
        const float32x4_t a = vld1q_f32(v + i), b = vld1q_f32(v + i + h);
        const float32x4_t cv = vld1q_f32(c + i), sv = vld1q_f32(s + i);
        vst1q_f32(v + i, vsubq_f32(vmulq_f32(a, cv), vmulq_f32(b, sv)));
        vst1q_f32(v + i + h, vaddq_f32(vmulq_f32(b, cv), vmulq_f32(a, sv)));
    }
#endif
    for (; i < h; i++) {
        const float a = v[i], b = v[i + h];
        v[i] = a * c[i] - b * s[i];
        v[i + h] = b * c[i] + a * s[i];
    }
}

/* ---- norms --------------------------------------------------------------------------- */
/* sum of squares: 4 interleaved partial sums over the first n/4*4 elements, then the tail
 * sequentially into t, result s + t */
static float sumsq(const float *x, int n) {
    float s, t = 0.0f;
    int i = 0;
#if PF_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        acc = vaddq_f32(acc, vmulq_f32(v, v));
    }
    s = hsum(acc);
#else
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (; i + 4 <= n; i += 4)
        for (int l = 0; l < 4; l++) c[l] = c[l] + x[i + l] * x[i + l];
    s = (c[0] + c[1]) + (c[2] + c[3]);
#endif
    for (; i < n; i++) t = t + x[i] * x[i];
    return s + t;
}

/* y = x / rms(x) * w. `w` is the stored effective multiplier (1 + w for zero-centred norms). */
void pf_rmsnorm(float *y, const float *x, const float *w, int n, float eps) {
    const float r = 1.0f / sqrtf(sumsq(x, n) / (float)n + eps);
    int i = 0;
#if PF_NEON
    const float32x4_t rv = vdupq_n_f32(r);
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), rv), vld1q_f32(w + i)));
#endif
    for (; i < n; i++) y[i] = x[i] * r * w[i];
}

/* x = (x * rsqrt(sum x^2 + eps)) * scale: the FLA-compatible l2norm of the reference, then
 * the query scale (two roundings, like the reference's two multiplies). */
void pf_l2norm_scale(float *x, int n, float eps, float scale) {
    const float r = 1.0f / sqrtf(sumsq(x, n) + eps);
    int i = 0;
#if PF_NEON
    const float32x4_t rv = vdupq_n_f32(r), sv = vdupq_n_f32(scale);
    for (; i + 4 <= n; i += 4) vst1q_f32(x + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), rv), sv));
#endif
    for (; i < n; i++) x[i] = x[i] * r * scale;
}

/* ---- causal depthwise conv + SiLU ---------------------------------------------------- */
#if PF_NEON
/* K = 4: channels c..c+3. The three state rows shift down and receive x. */
static inline float32x4_t conv4(float *x, float *s0, float *s1, float *s2, const float *w,
                                int c) {
    const float32x4x4_t wv = vld4q_f32(w + (size_t)c * 4);
    const float32x4_t a0 = vld1q_f32(s0 + c), a1 = vld1q_f32(s1 + c);
    const float32x4_t a2 = vld1q_f32(s2 + c), xv = vld1q_f32(x + c);
    float32x4_t acc = vmulq_f32(wv.val[0], a0);
    acc = vaddq_f32(acc, vmulq_f32(wv.val[1], a1));
    acc = vaddq_f32(acc, vmulq_f32(wv.val[2], a2));
    acc = vaddq_f32(acc, vmulq_f32(wv.val[3], xv));
    vst1q_f32(s0 + c, a1);
    vst1q_f32(s1 + c, a2);
    vst1q_f32(s2 + c, xv);
    return acc;
}
#endif

/* In place on channels [c0, c1) of one token's C channels. state holds the previous K-1 raw
 * inputs, oldest first ([K-1][C]); it is shifted and receives x. w is [C][K] (vld4
 * deinterleaves 4 channels x 4 taps). acc = w0*e0 + w1*e1 + ... in tap order, e = state
 * rows then x. */
void pf_conv_silu(float *x, float *st, const float *w, int C, int K, int c0, int c1) {
    const int S = K - 1;
    int c = c0;
#if PF_NEON
    if (K == 4) {
        float *s0 = st, *s1 = st + C, *s2 = st + 2 * (size_t)C;
        for (; c + 8 <= c1; c += 8) {  /* two exp chains per iteration, see above */
            const float32x4_t acc0 = conv4(x, s0, s1, s2, w, c), acc1 = conv4(x, s0, s1, s2, w, c + 4);
            vst1q_f32(x + c, siluq(acc0));
            vst1q_f32(x + c + 4, siluq(acc1));
        }
        for (; c + 4 <= c1; c += 4) vst1q_f32(x + c, siluq(conv4(x, s0, s1, s2, w, c)));
    }
#endif
    for (; c < c1; c++) {
        const float *wc = w + (size_t)c * K;
        const float xc = x[c];
        float acc = wc[0] * (S > 0 ? st[c] : xc);
        for (int j = 1; j < K; j++) acc = acc + wc[j] * (j < S ? st[(size_t)j * C + c] : xc);
        for (int j = 0; j + 1 < S; j++) st[(size_t)j * C + c] = st[(size_t)(j + 1) * C + c];
        if (S > 0) st[(size_t)(S - 1) * C + c] = xc;
        x[c] = pf_silu(acc);
    }
}

/* ---- attention ----------------------------------------------------------------------- */
/* a . b with 16 interleaved partial sums over the first n/16*16 elements (4 vector
 * accumulators), sequential tail; combined ((c0..3) + (c4..7)) + ((c8..11) + (c12..15)). */
float pf_dot(const float *a, const float *b, int n) {
    float s, t = 0.0f;
    int i = 0;
#if PF_NEON
    float32x4_t c0 = vdupq_n_f32(0.0f), c1 = c0, c2 = c0, c3 = c0;
    for (; i + 16 <= n; i += 16) {
        c0 = vaddq_f32(c0, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        c1 = vaddq_f32(c1, vmulq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
        c2 = vaddq_f32(c2, vmulq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8)));
        c3 = vaddq_f32(c3, vmulq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12)));
    }
    s = (hsum(c0) + hsum(c1)) + (hsum(c2) + hsum(c3));
#else
    float c[16] = {0.0f};
    for (; i + 16 <= n; i += 16)
        for (int l = 0; l < 16; l++) c[l] = c[l] + a[i + l] * b[i + l];
    float h[4];
    for (int k = 0; k < 4; k++) h[k] = (c[4 * k] + c[4 * k + 1]) + (c[4 * k + 2] + c[4 * k + 3]);
    s = (h[0] + h[1]) + (h[2] + h[3]);
#endif
    for (; i < n; i++) t = t + a[i] * b[i];
    return s + t;
}

/* a[j] = exp(a[j] - mx) for j < n; returns the sum (4 partial sums + sequential tail). */
float pf_softmax_exp(float *a, int n, float mx) {
    float s, t = 0.0f;
    int i = 0;
#if PF_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    const float32x4_t m = vdupq_n_f32(mx);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t e = expq(vsubq_f32(vld1q_f32(a + i), m));
        vst1q_f32(a + i, e);
        acc = vaddq_f32(acc, e);
    }
    s = hsum(acc);
#else
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (; i + 4 <= n; i += 4)
        for (int l = 0; l < 4; l++) {
            a[i + l] = pf_expf(a[i + l] - mx);
            c[l] = c[l] + a[i + l];
        }
    s = (c[0] + c[1]) + (c[2] + c[3]);
#endif
    for (; i < n; i++) {
        a[i] = pf_expf(a[i] - mx);
        t = t + a[i];
    }
    return s + t;
}

/* o[d] = sum_j p[j] * v[j][d] (v rows `stride` floats apart), accumulated in j order from 0
 * per element. Row-outer: o (1 KB, L1-resident) is updated over its full width per row, so
 * v is one sequential stream. Measured per head at 100 rows on the Pi (engine/CORE.md):
 * 17 us warm / 24 us cold, against 29 / 44 for a column-chunk-outer loop that keeps the
 * sums in registers but sweeps the rows 16 times at a 1 KB stride (L1 miss on every load). */
void pf_attn_pv(float *o, const float *p, int np, const float *v, int stride, int hd) {
    int d = 0;
#if PF_NEON
    const int d16 = hd / 16 * 16;
    for (; d < d16; d += 4) vst1q_f32(o + d, vdupq_n_f32(0.0f));
    for (int j = 0; j < np; j++) {
        const float *vj = v + (size_t)j * stride;
        const float32x4_t pj = vdupq_n_f32(p[j]);
        for (d = 0; d < d16; d += 16) {
            vst1q_f32(o + d, vaddq_f32(vld1q_f32(o + d), vmulq_f32(pj, vld1q_f32(vj + d))));
            vst1q_f32(o + d + 4, vaddq_f32(vld1q_f32(o + d + 4), vmulq_f32(pj, vld1q_f32(vj + d + 4))));
            vst1q_f32(o + d + 8, vaddq_f32(vld1q_f32(o + d + 8), vmulq_f32(pj, vld1q_f32(vj + d + 8))));
            vst1q_f32(o + d + 12, vaddq_f32(vld1q_f32(o + d + 12), vmulq_f32(pj, vld1q_f32(vj + d + 12))));
        }
    }
    d = d16;
#endif
    for (; d < hd; d++) {
        float acc = 0.0f;
        for (int j = 0; j < np; j++) acc = acc + p[j] * v[(size_t)j * stride + d];
        o[d] = acc;
    }
}

/* ---- argmax -------------------------------------------------------------------------- */
/* Index of the first element equal to the maximum (= the first maximum, like torch.argmax
 * on CPU). Two passes: the maximum, then its first position. */
int pf_argmax(const float *v, int n) {
    if (!v || n <= 0) return -1;
    float mx = v[0];
    int i = 0;
#if PF_NEON
    if (n >= 4) {
        float32x4_t m = vld1q_f32(v);
        for (i = 4; i + 4 <= n; i += 4) m = vmaxq_f32(m, vld1q_f32(v + i));
        mx = vmaxvq_f32(m);
    }
#endif
    for (; i < n; i++)
        if (v[i] > mx) mx = v[i];
    i = 0;
#if PF_NEON
    const float32x4_t mv = vdupq_n_f32(mx);
    for (; i + 4 <= n; i += 4)
        if (vmaxvq_u32(vceqq_f32(vld1q_f32(v + i), mv))) break;
#endif
    for (; i < n; i++)
        if (v[i] == mx) return i;
    return 0;
}
