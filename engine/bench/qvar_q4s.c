/* tbench variants "q4s" / "q4e" / "q4m": the dispatch-ordered asm register passes generated
 * by tools/gen_kern_q4s.py (q4s_gen.h: SCVTF, passes of up to 7; q4e_gen.h: the same with the
 * per-vector scale / sum loads at the start of each step; q4m_gen.h: magic-number int->float,
 * passes of up to 6), bit-identical to kern_quant.c. Measured, not integrated (KERNELS.md
 * Part 5: -9 to -13 % cycles per block-vector on Q4, +2 to +3.5 % end to end). n == 1 (decode,
 * DRAM-bound) runs kern_q4 / kern_q4k. */
#include <arm_neon.h>

#include "kern.h"
#include "pq_quant.h"
#include "qvar.h"
#include "qvar_q4k.h"

#include "q4e_gen.h"
#include "q4m_gen.h"
#include "q4s_gen.h"

/* Q4K: the decoded (d4, o4) of every block of the tile, [blk][d0..d3, o0..o3] */
static void q4k_tile_scales(const uint8_t *p, int nsb, float *ksc) {
    float ds[32], os[32];
    for (int sb = 0; sb < nsb; sb++) {
        qv_q4k_scales(p + (size_t)sb * 576, ds, os);
        for (int k = 0; k < 8; k++) {
            vst1q_f32(ksc + (sb * 8 + k) * 8, vld1q_f32(ds + 4 * k));
            vst1q_f32(ksc + (sb * 8 + k) * 8 + 4, vld1q_f32(os + 4 * k));
        }
    }
}

static void run_tiles(const q4s_fn *tiles, int nvmax, const pf_mat *W, const pf_act *A,
                      float *Y, int n, int r0, int r1) {
    const int nsb = W->cols / PQ_SB_COLS, kq = W->dtype == PQ_Q4K_G32;
    float ksc[kq ? nsb * 64 : 1];
    for (int r = r0; r < r1; r += 4) {
        const uint8_t *p = (const uint8_t *)W->data + (size_t)(r / 4) * nsb * 576;
        if (kq) q4k_tile_scales(p, nsb, ksc);
        for (int v = 0; v < n;) { /* passes of equal size (+-1), as kern_quant.c */
            const int left = n - v, np = (left + nvmax - 1) / nvmax, nv = (left + np - 1) / np;
            tiles[nv](p, (long)nsb * 8, A->q + (size_t)v * 32, A->d + v, A->fsum + v,
                      (long)A->n * 32, (long)A->n * 4, Y + (size_t)v * W->rows + r,
                      (long)W->rows * 4, ksc);
            v += nv;
        }
    }
}

void qv_q4s(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1) {
    if (n == 1) { qv_ref(W, aux, A, Y, n, r0, r1); return; }
    run_tiles(W->dtype == PQ_Q4K_G32 ? q4s_q4k_tiles : q4s_q4_tiles, 7, W, A, Y, n, r0, r1);
}

void qv_q4m(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1) {
    if (n == 1) { qv_ref(W, aux, A, Y, n, r0, r1); return; }
    run_tiles((const q4s_fn *)(W->dtype == PQ_Q4K_G32 ? q4m_q4k_tiles : q4m_q4_tiles), 6, W, A,
              Y, n, r0, r1);
}

void qv_q4e(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1) {
    if (n == 1) { qv_ref(W, aux, A, Y, n, r0, r1); return; }
    run_tiles((const q4s_fn *)(W->dtype == PQ_Q4K_G32 ? q4e_q4k_tiles : q4e_q4_tiles), 7, W, A,
              Y, n, r0, r1);
}
