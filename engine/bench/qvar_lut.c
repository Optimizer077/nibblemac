/* tbench variant "lut": T-MAC-style bit-serial table lookup for Q4_G32 / Q4K_G32, exact
 * (KERNELS.md Part 5).
 *
 * Integer part of a block (32 columns) for row r and vector v: t = sum_j c_j x_j with codes
 * c in 0..15 and int8 activations x. Split x = 16 h + l (l = x & 15 in 0..15, h = x >> 4 in
 * -8..7) and c into bit planes: t = sum_b 2^b (sum_g TL_g[i_bg] + 16 sum_g TH_g[i_bg]), where
 * for each group g of 4 columns the 16-entry tables hold the subset sums of l (0..60, u8) and h
 * (-32..28, s8) and i_bg (4 bits) is row r's plane-b bits of those 4 columns. One TBL looks up
 * 16 rows. Sums of 4 groups still fit a byte (<= 240, >= -128); planes are combined in int16
 * (|sum_b 2^b L_b| <= 7200, |.. H_b| <= 3840) and t = 16 H + L in int32 (|t| <= 60960).
 * Then the float update of kern_quant.c (same ops, same operands) => bit-identical output.
 *
 * Weights are repacked once (prep, "load time") into 16-row panels: [panel][block][plane]
 * [group][16 rows] index bytes = 1 byte per weight (2x the 4-bit codes) plus the decoded
 * fp32 (d4, o4) per tile and block (8 floats per 4 rows x 32 columns). Tables are built per
 * call and per thread (n x cols / 32 x 256 B). One vector at a time per panel.
 */
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

#include "kern.h"
#include "pq_quant.h"
#include "qvar.h"
#include "qvar_q4k.h"

typedef struct {
    int panels, nblk;
    uint8_t *idx; /* [panel][blk][plane 4][group 8][16] */
    float *sc;    /* [tile][blk][d0..d3, o0..o3] */
} lut_aux;

static int code_at(const pf_mat *W, int row, int col) {
    const int nsb = W->cols / PQ_SB_COLS, i = row % 4, sb = col / 256, k = (col % 256) / 32;
    const int j = col % 32;
    const uint8_t *q = (const uint8_t *)W->data + ((size_t)(row / 4) * nsb + sb) * 576 +
                       PQ_SB_SCALE_BYTES + k * 64 + i * 16 + (j % 16);
    return j < 16 ? (*q & 15) : (*q >> 4);
}

void *qv_lut_prep(const pf_mat *W, size_t *bytes) {
    lut_aux *a = calloc(1, sizeof *a);
    const int nsb = W->cols / PQ_SB_COLS, tiles = W->rows / 4;
    a->panels = W->rows / 16, a->nblk = nsb * 8;
    const size_t ib = (size_t)a->panels * a->nblk * 512, sb = (size_t)tiles * a->nblk * 8 * 4;
    a->idx = aligned_alloc(64, ib);
    a->sc = aligned_alloc(64, sb);
    *bytes = ib + sb;
    for (int P = 0; P < a->panels; P++)
        for (int blk = 0; blk < a->nblk; blk++) {
            uint8_t *o = a->idx + ((size_t)P * a->nblk + blk) * 512;
            for (int r = 0; r < 16; r++) {
                int c[32];
                for (int j = 0; j < 32; j++) c[j] = code_at(W, P * 16 + r, blk * 32 + j);
                for (int b = 0; b < 4; b++)
                    for (int g = 0; g < 8; g++) {
                        int v = 0;
                        for (int k = 0; k < 4; k++) v |= ((c[4 * g + k] >> b) & 1) << k;
                        o[(b * 8 + g) * 16 + r] = (uint8_t)v;
                    }
            }
        }
    for (int t = 0; t < tiles; t++) {
        const uint8_t *p = (const uint8_t *)W->data + (size_t)t * nsb * 576;
        for (int s = 0; s < nsb; s++, p += 576) {
            float ds[32], os[32];
            if (W->dtype == PQ_Q4K_G32) {
                qv_q4k_scales(p, ds, os);
            } else {
                for (int k = 0; k < 8; k++) {
                    const float32x4_t d4 = vcvt_f32_f16(vld1_f16((const __fp16 *)p + 4 * k));
                    vst1q_f32(ds + 4 * k, d4);
                    vst1q_f32(os + 4 * k, vmulq_n_f32(d4, 8.0f));
                }
            }
            for (int k = 0; k < 8; k++) {
                float *dst = a->sc + ((size_t)t * a->nblk + s * 8 + k) * 8;
                memcpy(dst, ds + 4 * k, 16), memcpy(dst + 4, os + 4 * k, 16);
            }
        }
    }
    return a;
}

/* tables of all (block, vector): [blk][v][group][TL 16 | TH 16] */
static void build_tables(const pf_act *A, uint8_t *tb) {
    const int nb = A->cols / 32;
    for (int blk = 0; blk < nb; blk++)
        for (int v = 0; v < A->n; v++) {
            const int8_t *x = A->q + ((size_t)blk * A->n + v) * 32;
            uint8_t *o = tb + ((size_t)blk * A->n + v) * 256;
            for (int g = 0; g < 8; g++)
                for (int idx = 0; idx < 16; idx++) {
                    int l = 0, h = 0;
                    for (int k = 0; k < 4; k++)
                        if (idx >> k & 1) l += x[4 * g + k] & 15, h += x[4 * g + k] >> 4;
                    o[g * 32 + idx] = (uint8_t)l;
                    o[g * 32 + 16 + idx] = (uint8_t)(int8_t)h;
                }
        }
}

/* t of 16 rows (4 x int32x4, lane = row % 4, register = tile) for one block and vector */
static inline void lut_block(const uint8_t *ix, const uint8_t *tb, int32x4_t *t) {
    uint8x16_t TL[8], TH[8];
    for (int g = 0; g < 8; g++) TL[g] = vld1q_u8(tb + g * 32), TH[g] = vld1q_u8(tb + g * 32 + 16);
    int16x8_t L0 = vdupq_n_s16(0), L1 = L0, H0 = L0, H1 = L0;
    for (int b = 3; b >= 0; b--) { /* Horner over the planes: L = 2 L + L_b */
        const uint8_t *ib = ix + b * 8 * 16;
        uint8x16_t i = vld1q_u8(ib), j = vld1q_u8(ib + 64);
        uint8x16_t la = vqtbl1q_u8(TL[0], i), lb = vqtbl1q_u8(TL[4], j);
        uint8x16_t ua = vqtbl1q_u8(TH[0], i), ub = vqtbl1q_u8(TH[4], j);
        for (int g = 1; g < 4; g++) { /* <= 4 groups per byte sum: no overflow */
            i = vld1q_u8(ib + g * 16), j = vld1q_u8(ib + (g + 4) * 16);
            la = vaddq_u8(la, vqtbl1q_u8(TL[g], i)), lb = vaddq_u8(lb, vqtbl1q_u8(TL[g + 4], j));
            ua = vaddq_u8(ua, vqtbl1q_u8(TH[g], i)), ub = vaddq_u8(ub, vqtbl1q_u8(TH[g + 4], j));
        }
        const int8x16_t ha = vreinterpretq_s8_u8(ua), hb = vreinterpretq_s8_u8(ub);
        const int16x8_t sl0 = vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(la), vget_low_u8(lb)));
        const int16x8_t sl1 = vreinterpretq_s16_u16(vaddl_high_u8(la, lb));
        const int16x8_t sh0 = vaddl_s8(vget_low_s8(ha), vget_low_s8(hb));
        const int16x8_t sh1 = vaddl_high_s8(ha, hb);
        L0 = vaddq_s16(vshlq_n_s16(L0, 1), sl0), L1 = vaddq_s16(vshlq_n_s16(L1, 1), sl1);
        H0 = vaddq_s16(vshlq_n_s16(H0, 1), sh0), H1 = vaddq_s16(vshlq_n_s16(H1, 1), sh1);
    }
    t[0] = vaddw_s16(vshll_n_s16(vget_low_s16(H0), 4), vget_low_s16(L0));
    t[1] = vaddw_high_s16(vshll_high_n_s16(H0, 4), L0);
    t[2] = vaddw_s16(vshll_n_s16(vget_low_s16(H1), 4), vget_low_s16(L1));
    t[3] = vaddw_high_s16(vshll_high_n_s16(H1, 4), L1);
}

static _Thread_local struct { uint8_t *p; size_t cap; } tbuf;

void qv_lut(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1) {
    const lut_aux *a = aux;
    const size_t need = (size_t)A->cols / 32 * n * 256;
    if (need > tbuf.cap) free(tbuf.p), tbuf.p = aligned_alloc(64, need), tbuf.cap = need;
    build_tables(A, tbuf.p);
    for (int P = r0 / 16; P * 16 < r1; P++)
        for (int v = 0; v < n; v++) {
            float32x4_t acc[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
            for (int blk = 0; blk < a->nblk; blk++) {
                int32x4_t t[4];
                lut_block(a->idx + ((size_t)P * a->nblk + blk) * 512,
                          tbuf.p + ((size_t)blk * n + v) * 256, t);
                const size_t k = (size_t)blk * A->n + v;
                for (int j = 0; j < 4; j++) {
                    const float *s = a->sc + ((size_t)(P * 4 + j) * a->nblk + blk) * 8;
                    const float32x4_t d4 = vld1q_f32(s), o4 = vld1q_f32(s + 4);
                    acc[j] = vfmaq_f32(acc[j], vcvtq_f32_s32(t[j]), vmulq_n_f32(d4, A->d[k]));
                    acc[j] = vfmsq_n_f32(acc[j], o4, A->fsum[k]);
                }
            }
            for (int j = 0; j < 4; j++) {
                const int r = P * 16 + j * 4;
                if (r >= r0 && r < r1) vst1q_f32(Y + (size_t)v * W->rows + r, acc[j]);
            }
        }
}
