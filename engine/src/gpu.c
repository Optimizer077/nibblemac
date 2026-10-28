/* Opt-in V3D 4.2 Q4 matmul. BOs are DRM shmem/MMU objects, not CMA allocations.
 * Private to one context/pool; all jobs finish before its caller reuses scratch.
 */
#if defined(__aarch64__) && defined(__linux__)
#define _POSIX_C_SOURCE 200809L
#include "gpu.h"
#include "pq_quant.h"
#include "../gpu/v3d.h"
#include "../gpu/q4gpu_kernels.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define GPU_BUDGET ((size_t)512 << 20)
#define GPU_RESERVE ((size_t)16 << 20)
#define GPU_BATCH 64
#define GPU_JOBS 64

typedef struct {
    const pf_mat *mat;
    int rows;
    struct v3d_bo codes, scales;
} gpu_weight;
typedef struct { const pf_mat *mat; int order; } candidate;
struct pf_gpu {
    struct v3d_device dev;
    struct v3d_bo code, data;
    uint32_t programs[9];
    gpu_weight *weights;
    int count, broken, pending, n, full_rows, gpu_rows;
    size_t bytes, stream_cap, y_offset, param_offset;
};

static size_t align128(size_t x) { return (x + 127) & ~(size_t)127; }
static gpu_weight *find_weight(const pf_gpu *g, const pf_mat *w) {
    if (!g || g->broken) return NULL;
    for (int i = 0; i < g->count; i++)
        if (g->weights[i].mat->data == w->data && g->weights[i].mat->rows == w->rows &&
            g->weights[i].mat->cols == w->cols) return &g->weights[i];
    return NULL;
}
static int candidate_order(const void *aa, const void *bb) {
    const candidate *a = aa, *b = bb;
    uint64_t na = (uint64_t)a->mat->rows * a->mat->cols;
    uint64_t nb = (uint64_t)b->mat->rows * b->mat->cols;
    return na != nb ? (na < nb ? 1 : -1) : (a->order > b->order) - (a->order < b->order);
}
static uint16_t load16(const uint8_t *p) { uint16_t x; memcpy(&x, p, 2); return x; }
static const uint8_t *block(const pf_mat *w, int r, int b) {
    return (const uint8_t *)w->data + ((size_t)(r / 4) * (w->cols / 256) + b / 8) * 576;
}
static int pack_weight(gpu_weight *dst) {
    const pf_mat *w = dst->mat;
    int nb = w->cols / 32;
    uint32_t *codes = dst->codes.map, *scales = dst->scales.map;
    for (int t = 0; t < dst->rows / 32; t++) for (int b = 0; b < nb; b++)
        for (int lane = 0; lane < 16; lane++) {
            int a = t * 32 + lane, h = a + 16;
            const uint8_t *sa = block(w, a, b), *sb = block(w, h, b);
            size_t pos = ((size_t)t * nb + b) * 16 + lane;
            scales[pos] = load16(sa + ((b % 8) * 4 + a % 4) * 2) |
                ((uint32_t)load16(sb + ((b % 8) * 4 + h % 4) * 2) << 16);
            const uint8_t *qa = sa + 64 + (b % 8) * 64 + (a % 4) * 16;
            const uint8_t *qb = sb + 64 + (b % 8) * 64 + (h % 4) * 16;
            for (int k = 0; k < 8; k++) {
                uint32_t word = 0;
                for (int j = 0; j < 4; j++) {
                    int c = k * 4 + j, shift = c >= 16 ? 4 : 0;
                    word |= (uint32_t)((qa[c % 16] >> shift) & 15) << (4 * j);
                    word |= (uint32_t)((qb[c % 16] >> shift) & 15) << (16 + 4 * j);
                }
                codes[pos * 8 + k] = word;
            }
        }
    size_t nc = (size_t)(dst->rows / 32) * nb * 16 * 8;
    size_t ns = (size_t)(dst->rows / 32) * nb * 16;
    memset(codes + nc, 0, 384 * 4); memset(scales + ns, 0, 48 * 4);
    /* Keep GEM handles/GPU addresses; release CPU mappings from RSS after upload. */
    if (munmap(dst->codes.map, dst->codes.size)) return -1;
    dst->codes.map = NULL;
    if (munmap(dst->scales.map, dst->scales.size)) return -1;
    dst->scales.map = NULL;
    return 0;
}

void pf_gpu_destroy(pf_gpu *g) {
    if (!g) return;
    if (g->pending) (void)v3d_wait(&g->dev, &g->data, UINT64_C(3000000000));
    for (int i = 0; i < g->count; i++) {
        v3d_bo_destroy(&g->dev, &g->weights[i].codes);
        v3d_bo_destroy(&g->dev, &g->weights[i].scales);
    }
    v3d_bo_destroy(&g->dev, &g->data); v3d_bo_destroy(&g->dev, &g->code);
    v3d_close(&g->dev); free(g->weights); free(g);
}

pf_gpu *pf_gpu_create(const pf_mat *const *weights, int count, float share) {
    pf_gpu *g = calloc(1, sizeof(*g));
    candidate *cand = calloc((size_t)count, sizeof(*cand));
    if (!g || !cand) { free(g); free(cand); return NULL; }
    g->dev.fd = -1;
    g->weights = calloc((size_t)count, sizeof(*g->weights));
    if (!g->weights || v3d_open(&g->dev, NULL)) goto fail;
    int nc = 0;
    size_t full_bytes = 0;
    for (int i = 0; i < count; i++) {
        const pf_mat *w = weights[i];
        if (w->dtype != PQ_Q4_G32 || w->cols <= 0 || w->cols > 32768 || w->cols % 256 || w->rows < 256) continue;
        cand[nc++] = (candidate){w, i};
        full_bytes += pq_quant_nbytes(w->dtype, w->rows, w->cols);
    }
    qsort(cand, (size_t)nc, sizeof(*cand), candidate_order);
    int max_rows = 0, max_blocks = 0;
    for (int i = 0; i < nc; i++) {
        const pf_mat *w = cand[i].mat;
        if (find_weight(g, w)) continue;
        int rows = (int)((double)w->rows * share / 256) * 256;
        if (rows > 16384) rows = 16384;
        if (rows < 256) continue;
        /* The QPU multiplies the code stride with umul24; both inputs need 24 bits. */
        if ((uint64_t)(rows / 256) * (w->cols / 32) * 512 >= (1u << 24)) continue;
        size_t cb = (size_t)(rows / 32) * (w->cols / 32) * 512 + 1536;
        size_t sb = (size_t)(rows / 32) * (w->cols / 32) * 64 + 192;
        if (g->bytes + cb + sb + 131072 > GPU_BUDGET - GPU_RESERVE) continue;
        gpu_weight *gw = &g->weights[g->count];
        gw->mat = w; gw->rows = rows;
        if (v3d_bo_create(&g->dev, &gw->codes, cb) || v3d_bo_create(&g->dev, &gw->scales, sb)) {
            v3d_bo_destroy(&g->dev, &gw->codes); v3d_bo_destroy(&g->dev, &gw->scales);
            continue; /* Keep the largest fitting subset; no allocation beyond budget. */
        }
        g->count++;
        if (pack_weight(gw)) goto fail;
        g->bytes += gw->codes.size + gw->scales.size;
        if (rows > max_rows) max_rows = rows;
        if (w->cols / 32 > max_blocks) max_blocks = w->cols / 32;
    }
    if (!g->count) { errno = ENODEV; goto fail; }
    size_t code_bytes = 0;
    for (int n = 1; n <= 8; n++) code_bytes = align128(code_bytes) + q4gpu_programs[n]->words * 8;
    if (v3d_bo_create(&g->dev, &g->code, code_bytes)) goto fail;
    size_t off = 0;
    for (int n = 1; n <= 8; n++) {
        off = align128(off);
        memcpy((uint8_t *)g->code.map + off, q4gpu_programs[n]->code, q4gpu_programs[n]->words * 8);
        g->programs[n] = g->code.gpu + (uint32_t)off;
        off += q4gpu_programs[n]->words * 8;
    }
    g->stream_cap = align128((size_t)GPU_BATCH * (34 * max_blocks + 1) * 4 + GPU_JOBS * 128);
    g->y_offset = g->stream_cap;
    g->param_offset = align128(g->y_offset + (size_t)GPU_BATCH * max_rows * 4);
    size_t data_bytes = g->param_offset + GPU_JOBS * 128;
    if (v3d_bo_create(&g->dev, &g->data, data_bytes)) goto fail;
    g->bytes += g->code.size + g->data.size;
    if (g->bytes > GPU_BUDGET) { errno = ENOMEM; goto fail; }
    fprintf(stderr, "pinfer: V3D split=%.3f matrices=%d/%d BO=%.2f MiB (budget 512 MiB), Q4 body=%.2f MiB\n",
            (double)share, g->count, nc, g->bytes / 1048576.0, full_bytes / 1048576.0);
    free(cand);
    return g;
fail:
    perror("pinfer: V3D setup"); free(cand); pf_gpu_destroy(g); return NULL;
}

int pf_gpu_rows(const pf_gpu *g, const pf_mat *w) {
    gpu_weight *gw = find_weight(g, w);
    return gw ? gw->rows : 0;
}
static uint32_t float_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint32_t negc(const pf_act *a, int b, int v) {
    int sum = 0;
    for (int c = 0; c < 32; c++) sum += a->q[((size_t)b * a->n + v) * 32 + c];
    return (uint32_t)(-INT64_C(524296) * sum);
}
static uint32_t uniform_word(struct q4gpu_tag t, const pf_act *a, int b, int v0) {
    int v = v0 + t.vec;
    switch (t.kind) {
    case Q4GPU_X: return (uint32_t)(int32_t)a->q[((size_t)b * a->n + v) * 32 + t.col];
    case Q4GPU_DX: return float_bits(a->d[(size_t)b * a->n + v]);
    case Q4GPU_NEGC: return negc(a, 0, v);
    default: return b + 1 < a->cols / 32 ? negc(a, b + 1, v) : 0;
    }
}

int pf_gpu_begin(pf_gpu *g, const pf_mat *w, const pf_act *groups, int count) {
    gpu_weight *gw = find_weight(g, w);
    if (!gw || g->pending || count < 1 || count > GPU_JOBS) return -1;
    int total = 0, nj = 0;
    size_t off = 0;
    uint32_t addresses[GPU_JOBS], shaders[GPU_JOBS];
    for (int group = 0; group < count; group++) {
        const pf_act *a = &groups[group];
        if (a->cols != w->cols || a->n < 1 || total + a->n > GPU_BATCH) return -1;
        int np = (a->n + 7) / 8, v0 = 0;
        for (int pass = 0; pass < np; pass++) {
            int nv = a->n / np + (pass < a->n % np);
            const struct q4gpu_program *prog = q4gpu_programs[nv];
            size_t words = prog->start_words + (size_t)(w->cols / 32) * prog->block_words;
            off = align128(off);
            if (nj >= GPU_JOBS || off + words * 4 > g->stream_cap) return -1;
            uint32_t *stream = (uint32_t *)((uint8_t *)g->data.map + off);
            size_t pos = 0;
            for (size_t i = 0; i < prog->start_words; i++) stream[pos++] = uniform_word(prog->start[i], a, 0, v0);
            for (int b = 0; b < w->cols / 32; b++) for (size_t i = 0; i < prog->block_words; i++)
                stream[pos++] = uniform_word(prog->block[i], a, b, v0);
            uint32_t *p = (uint32_t *)((uint8_t *)g->data.map + g->param_offset + (size_t)nj * 128);
            uint32_t nt = (uint32_t)gw->rows / 256, nb = (uint32_t)w->cols / 32;
            p[Q4GPU_CODES] = gw->codes.gpu; p[Q4GPU_SCALES] = gw->scales.gpu;
            p[Q4GPU_Y] = g->data.gpu + (uint32_t)(g->y_offset + (size_t)(total + v0) * gw->rows * 4);
            p[Q4GPU_ACT] = g->data.gpu + (uint32_t)off;
            p[Q4GPU_NT] = nt; p[Q4GPU_NB] = nb;
            p[Q4GPU_TSC] = nt * nb * 512; p[Q4GPU_TSS] = nt * nb * 64; p[Q4GPU_TSY] = nt * 128;
            p[Q4GPU_R4] = (uint32_t)gw->rows * 4; p[Q4GPU_MASK] = UINT32_C(0x000f000f);
            p[Q4GPU_K65536] = 65536; p[Q4GPU_K512] = 512; p[Q4GPU_K64] = 64;
            addresses[nj] = g->data.gpu + (uint32_t)(g->param_offset + (size_t)nj * 128);
            shaders[nj] = g->programs[nv];
            off += words * 4; nj++; v0 += nv;
        }
        total += a->n;
    }
    uint32_t handles[] = {g->code.handle, g->data.handle, gw->codes.handle, gw->scales.handle};
    g->n = total; g->full_rows = w->rows; g->gpu_rows = gw->rows;
    for (int j = 0; j < nj; j++) {
        if (v3d_submit_8qpu(&g->dev, shaders[j], addresses[j], handles, 4)) {
            perror("pinfer: V3D submit (disabled)"); g->broken = 1;
            if (g->pending) (void)v3d_wait(&g->dev, &g->data, UINT64_C(3000000000));
            g->pending = 0;
            return -1;
        }
        g->pending = 1;
    }
    return 0;
}

int pf_gpu_finish(pf_gpu *g, float *y) {
    if (!g || !g->pending) return -1;
    int rc = v3d_wait(&g->dev, &g->data, UINT64_C(3000000000));
    if (rc) { perror("pinfer: V3D wait (disabled)"); g->broken = 1; return -1; }
    g->pending = 0;
    const float *src = (const float *)((const uint8_t *)g->data.map + g->y_offset);
    for (int v = 0; v < g->n; v++)
        memcpy(y + (size_t)v * g->full_rows, src + (size_t)v * g->gpu_rows, (size_t)g->gpu_rows * 4);
    return 0;
}
#endif
