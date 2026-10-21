/* F32 and BF16 matvec / small-batch matmul kernels (plain row-major weights).
 *
 * Decode is DRAM-bound (~4.2 GB/s on the Pi 4), so these kernels only need to stay ahead of
 * the memory stream: 4 rows are processed together so every x load feeds 4 FMLAs, and up
 * to 4 vectors share each weight load (matmul). Wider batches are split into passes over
 * the same 4-row block, which is then re-read from L1/L2, not DRAM.
 */
#include <arm_neon.h>

#include "kern.h"
#include "pq_format.h"

#define FNV 4 /* vectors per pass: 4 rows x 4 vectors = 16 accumulators */

#define INLINE static inline __attribute__((always_inline))

/* Two float32x4 (8 columns) of row `row` starting at column c. */
INLINE void load8(const void *row, int c, int bf, float32x4_t *lo, float32x4_t *hi) {
    if (bf) {
        uint16x8_t h = vld1q_u16((const uint16_t *)row + c);
        *lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
        *hi = vreinterpretq_f32_u32(vshll_high_n_u16(h, 16));
    } else {
        *lo = vld1q_f32((const float *)row + c);
        *hi = vld1q_f32((const float *)row + c + 4);
    }
}

INLINE float scalar_w(const void *row, int c, int bf) {
    if (bf) {
        union { uint32_t u; float f; } v = {(uint32_t)((const uint16_t *)row)[c] << 16};
        return v.f;
    }
    return ((const float *)row)[c];
}

/* 4 rows starting at r, vectors v0..v0+nv-1 (nv is a compile-time constant after inlining). */
INLINE void block4(const pf_mat *W, int bf, const float *X, float *Y, int r, int v0, int nv,
                   int pf) {
    const int cols = W->cols, esz = bf ? 2 : 4;
    const char *base = (const char *)W->data + (size_t)r * cols * esz;
    const void *row[4];
    for (int i = 0; i < 4; i++) row[i] = base + (size_t)i * cols * esz;
    float32x4_t acc[4][FNV];
    for (int i = 0; i < 4; i++)
        for (int v = 0; v < nv; v++) acc[i][v] = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 32 <= cols; c += 32) {
        if (pf)
            for (int i = 0; i < 4; i++)
                __builtin_prefetch((const char *)row[i] + (size_t)c * esz + pf, 0, 0);
        for (int k = 0; k < 32; k += 8) {
            float32x4_t w[4][2];
            for (int i = 0; i < 4; i++) load8(row[i], c + k, bf, &w[i][0], &w[i][1]);
            for (int v = 0; v < nv; v++) {
                const float *x = X + (size_t)(v0 + v) * cols + c + k;
                float32x4_t x0 = vld1q_f32(x), x1 = vld1q_f32(x + 4);
                for (int i = 0; i < 4; i++) {
                    acc[i][v] = vfmaq_f32(acc[i][v], w[i][0], x0);
                    acc[i][v] = vfmaq_f32(acc[i][v], w[i][1], x1);
                }
            }
        }
    }
    for (int v = 0; v < nv; v++) {
        float32x4_t s = vpaddq_f32(vpaddq_f32(acc[0][v], acc[1][v]),
                                   vpaddq_f32(acc[2][v], acc[3][v]));
        float t[4];
        vst1q_f32(t, s);
        const float *x = X + (size_t)(v0 + v) * cols;
        for (int cc = c; cc < cols; cc++) /* column tail (cols % 32), never hit by Qwen */
            for (int i = 0; i < 4; i++) t[i] += scalar_w(row[i], cc, bf) * x[cc];
        float *y = Y + (size_t)(v0 + v) * W->rows + r;
        for (int i = 0; i < 4; i++) y[i] = t[i];
    }
}

/* Generic single row (row counts that are not a multiple of 4). */
static void row1(const pf_mat *W, int bf, const float *X, float *Y, int n, int r) {
    const int cols = W->cols;
    const void *row = (const char *)W->data + (size_t)r * cols * (bf ? 2 : 4);
    for (int v = 0; v < n; v++) {
        const float *x = X + (size_t)v * cols;
        float s = 0.0f;
        for (int c = 0; c < cols; c++) s += scalar_w(row, c, bf) * x[c];
        Y[(size_t)v * W->rows + r] = s;
    }
}

INLINE void kern_float(const pf_mat *W, int bf, const float *X, float *Y, int n, int r0, int r1) {
    const int pf = kern_prefetch;
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        int v = 0;
        for (; v + 4 <= n; v += 4) block4(W, bf, X, Y, r, v, 4, pf);
        switch (n - v) {
        case 3: block4(W, bf, X, Y, r, v, 3, pf); break;
        case 2: block4(W, bf, X, Y, r, v, 2, pf); break;
        case 1: block4(W, bf, X, Y, r, v, 1, pf); break;
        default: break;
        }
    }
    for (; r < r1; r++) row1(W, bf, X, Y, n, r);
}

void kern_f32(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1) {
    (void)A;
    kern_float(W, 0, X, Y, n, r0, r1);
}

void kern_bf16(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1) {
    (void)A;
    kern_float(W, 1, X, Y, n, r0, r1);
}
