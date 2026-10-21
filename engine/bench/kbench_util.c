/* kbench helpers: synthetic weights, C packers for the quantized layouts, scalar
 * references, and the golden-file check against tools/quant.py. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kbench.h"
#include "kern.h"
#include "pq_format.h"
#include "pq_quant.h"

size_t kb_matrix_bytes(uint32_t dtype, int rows, int cols) {
    if (dtype == PQ_F32) return (size_t)rows * cols * 4;
    if (dtype == PQ_BF16) return (size_t)rows * cols * 2;
    return pq_quant_nbytes(dtype, rows, cols);
}

static uint64_t rng_next(uint64_t *s) { /* xorshift64* */
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    return *s * 2685821657736338717ull;
}

/* Roughly bell-shaped values (sum of 2 uniforms) with sparse 6x outliers, like LLM weights. */
static void gen(float *w, size_t n, uint64_t *s, float scale) {
    for (size_t i = 0; i < n; i++) {
        uint64_t r = rng_next(s);
        float a = (float)(r >> 40) * (1.0f / 16777216.0f);
        float b = (float)((r >> 16) & 0xFFFFFF) * (1.0f / 16777216.0f);
        float v = (a + b - 1.0f) * scale;
        w[i] = (r & 63) == 0 ? 6.0f * v : v;
    }
}

void kb_fill_vec(float *x, size_t n, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    gen(x, n, &s, 1.0f);
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

/* Pack one 4-row tile (w: 4 rows x cols, row-major) into the layout of pq_quant.h. */
static void pack_q4k_sb(const float *w, int cols, uint8_t *p);

static void pack_tile(uint32_t dtype, const float *w, int cols, uint8_t *out) {
    const size_t sbb = pq_sb_bytes(dtype);
    for (int sb = 0; sb < cols / PQ_SB_COLS; sb++) {
        if (dtype == PQ_Q4K_G32) {
            pack_q4k_sb(w + sb * PQ_SB_COLS, cols, out + (size_t)sb * sbb);
            continue;
        }
        uint8_t *p = out + (size_t)sb * sbb, *codes = p + PQ_SB_SCALE_BYTES;
        __fp16 *sc = (__fp16 *)p;
        const float *ws = w + sb * PQ_SB_COLS;
        for (int i = 0; i < 4; i++) {
            const float *x0 = ws + (size_t)i * cols;
            if (dtype == PQ_Q8_G32) {
                for (int g = 0; g < 8; g++) {
                    const float *x = x0 + g * 32;
                    float amax = 0;
                    for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
                    sc[g * 4 + i] = (__fp16)(amax / 127.0f);
                    const float d = (float)sc[g * 4 + i], id = d > 0 ? 1.0f / d : 0.0f;
                    int8_t *q = (int8_t *)codes + g * 128 + i * 32;
                    for (int j = 0; j < 32; j++) q[j] = (int8_t)lrintf(clampf(x[j] * id, -127, 127));
                }
            } else { /* PQ_Q4_G32 */
                for (int k = 0; k < 8; k++) {
                    const float *x = x0 + k * 32;
                    float mx = 0;
                    for (int j = 0; j < 32; j++)
                        if (fabsf(x[j]) > fabsf(mx)) mx = x[j];
                    sc[k * 4 + i] = (__fp16)(mx / -8.0f);
                    const float d = (float)sc[k * 4 + i], id = d != 0 ? 1.0f / d : 0.0f;
                    uint8_t *q = codes + k * 64 + i * 16;
                    for (int j = 0; j < 16; j++) {
                        int a = (int)lrintf(clampf(x[j] * id + 8, 0, 15));
                        int b = (int)lrintf(clampf(x[j + 16] * id + 8, 0, 15));
                        q[j] = (uint8_t)(a | b << 4);
                    }
                }
            }
        }
    }
}

/* 6-bit packing of 32 values, as kern_get6 reads it. */
static void put6(uint8_t *p, const int *v) {
    memset(p, 0, 24);
    for (int j = 0; j < 32; j++) {
        p[j % 16] |= (uint8_t)((v[j] & 15) << (j < 16 ? 0 : 4));
        p[16 + j % 8] |= (uint8_t)((v[j] >> 4) << (2 * (j / 8)));
    }
}

/* Q4K superblock (4 rows x 256 cols at w, row stride cols): min/max per group, then 6-bit
 * sub-scales relative to the per-row maximum (no search; benchmark data only). */
static void pack_q4k_sb(const float *w, int cols, uint8_t *p) {
    __fp16 *h = (__fp16 *)p;
    int s6[32], m6[32];
    for (int i = 0; i < 4; i++) {
        float dg[8], mg[8], dmax = 0, mmax = 0;
        for (int g = 0; g < 8; g++) {
            const float *x = w + (size_t)i * cols + g * 32;
            float mn = x[0], mx = x[0];
            for (int j = 1; j < 32; j++) mn = fminf(mn, x[j]), mx = fmaxf(mx, x[j]);
            mn = fminf(mn, 0.0f);
            dg[g] = (mx - mn) / 15.0f, mg[g] = -mn;
            dmax = fmaxf(dmax, dg[g]), mmax = fmaxf(mmax, mg[g]);
        }
        h[i] = (__fp16)(dmax / 63.0f), h[4 + i] = (__fp16)(mmax / 63.0f);
        const float D = (float)h[i], M = (float)h[4 + i];
        for (int g = 0; g < 8; g++) {
            /* round the scale up so the 16 code levels still span the group */
            s6[g * 4 + i] = D > 0 ? (int)fminf(ceilf(dg[g] / D), 63) : 0;
            m6[g * 4 + i] = M > 0 ? (int)lrintf(fminf(mg[g] / M, 63)) : 0;
            const float d = D * s6[g * 4 + i], m = M * m6[g * 4 + i], id = d > 0 ? 1 / d : 0;
            const float *x = w + (size_t)i * cols + g * 32;
            uint8_t *q = p + PQ_SB_SCALE_BYTES + g * 64 + i * 16;
            for (int j = 0; j < 16; j++) {
                int a = (int)lrintf(clampf((x[j] + m) * id, 0, 15));
                int b = (int)lrintf(clampf((x[j + 16] + m) * id, 0, 15));
                q[j] = (uint8_t)(a | b << 4);
            }
        }
    }
    put6(p + 16, s6);
    put6(p + 40, m6);
}

void kb_fill_matrix(uint32_t dtype, int rows, int cols, uint64_t seed, void *dst) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 7;
    float *w = malloc((size_t)4 * cols * sizeof(float));
    if (!w) { fprintf(stderr, "kbench: out of memory\n"); exit(1); }
    const size_t tile_bytes = kb_matrix_bytes(dtype, 4, cols);
    for (int r = 0; r < rows; r += 4) {
        const int nr = rows - r < 4 ? rows - r : 4;
        gen(w, (size_t)nr * cols, &s, 0.05f);
        if (dtype == PQ_F32) {
            memcpy((float *)dst + (size_t)r * cols, w, (size_t)nr * cols * 4);
        } else if (dtype == PQ_BF16) {
            uint16_t *o = (uint16_t *)dst + (size_t)r * cols;
            for (size_t i = 0; i < (size_t)nr * cols; i++) { /* round to nearest even */
                uint32_t u;
                memcpy(&u, &w[i], 4);
                o[i] = (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
            }
        } else {
            pack_tile(dtype, w, cols, (uint8_t *)dst + (size_t)(r / 4) * tile_bytes);
        }
    }
    free(w);
}

void kb_reference(const pf_mat *W, const float *X, int n, const int *rowidx, int nrow,
                  float *ref_fp, float *ref_q) {
    const int cols = W->cols, nb = cols / KERN_ACT_BLOCK;
    float *row = malloc((size_t)cols * sizeof(float));
    float *xq = NULL;
    if (ref_q) { /* dequantized int8 activations, exactly as the kernels see them */
        int8_t *q = malloc((size_t)n * cols);
        float *d = malloc((size_t)n * nb * 4), *fs = malloc((size_t)n * nb * 4);
        xq = malloc((size_t)n * cols * sizeof(float));
        kern_quant_act(X, n, cols, q, d, fs); /* block-major: [b][v][32] */
        for (int v = 0; v < n; v++)
            for (int c = 0; c < cols; c++) {
                const size_t k = (size_t)(c / 32) * n + v;
                xq[(size_t)v * cols + c] = d[k] * (float)q[k * 32 + c % 32];
            }
        free(q), free(d), free(fs);
    }
    for (int k = 0; k < nrow; k++) {
        pf_get_row(W, rowidx[k], row);
        for (int v = 0; v < n; v++) {
            double a = 0, b = 0;
            for (int c = 0; c < cols; c++) {
                a += (double)row[c] * X[(size_t)v * cols + c];
                if (xq) b += (double)row[c] * xq[(size_t)v * cols + c];
            }
            ref_fp[(size_t)v * nrow + k] = (float)a;
            if (ref_q) ref_q[(size_t)v * nrow + k] = (float)b;
        }
    }
    free(row);
    free(xq);
}

/* ---- golden file: see tools/quant.py golden() for the layout ---- */
static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static float rel_err(const float *a, const float *b, size_t n) {
    float emax = 0, ymax = 0;
    for (size_t i = 0; i < n; i++) {
        emax = fmaxf(emax, fabsf(a[i] - b[i]));
        ymax = fmaxf(ymax, fabsf(b[i]));
    }
    return ymax > 0 ? emax / ymax : emax;
}

int kb_golden(const char *path, pf_pool *pool) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "golden: cannot open %s\n", path); return 1; }
    uint32_t hdr[2];
    int fails = 0;
    if (!rd(f, hdr, 8) || hdr[0] != 0x54475150u) { fclose(f); fprintf(stderr, "golden: bad file\n"); return 1; }
    for (uint32_t k = 0; k < hdr[1]; k++) {
        uint32_t h[4];
        uint64_t nbytes;
        if (!rd(f, h, 16) || !rd(f, &nbytes, 8)) { fails++; break; }
        const int rows = (int)h[1], cols = (int)h[2], n = (int)h[3];
        float *X = malloc((size_t)n * cols * 4), *Yg = malloc((size_t)n * rows * 4);
        float *Y = malloc((size_t)n * rows * 4), *Rg = malloc((size_t)n * rows * 4);
        float *Rq = malloc((size_t)n * rows * 4);
        void *data = aligned_alloc(64, (nbytes + 63) & ~(size_t)63);
        int *idx = malloc((size_t)rows * sizeof(int));
        rd(f, X, (size_t)n * cols * 4), rd(f, Yg, (size_t)n * rows * 4), rd(f, data, nbytes);
        pf_mat W = {h[0], rows, cols, data};
        for (int r = 0; r < rows; r++) idx[r] = r;
        pf_matmul(pool, &W, X, Y, n);
        kb_reference(&W, X, n, idx, rows, Rg, Rq);
        const float e_row = rel_err(Rg, Yg, (size_t)n * rows);  /* get_row vs python */
        const float e_kern = rel_err(Y, Rq, (size_t)n * rows);  /* kernel vs exact int8-x ref */
        const float e_all = rel_err(Y, Yg, (size_t)n * rows);   /* kernel vs python fp ref */
        const int ok = nbytes == kb_matrix_bytes(h[0], rows, cols) && e_row < 1e-5f &&
                       e_kern < 1e-5f && e_all < 3e-2f;
        printf("golden dtype=%u %dx%d n=%d  get_row %.2e  kernel(int8 x) %.2e  vs fp %.2e  %s\n",
               h[0], rows, cols, n, e_row, e_kern, e_all, ok ? "ok" : "FAIL");
        fails += !ok;
        free(X), free(Yg), free(Y), free(Rg), free(Rq), free(data), free(idx);
    }
    fclose(f);
    return fails;
}
