/* Exact physical-GPU graph gates against the inherited CPU primitives.
 * No models, tokenizer or Python dependencies. State remains in one shared buffer
 * through normal, logged, pending-update, replay and CPU save/reset/restore steps. */
#include "metal.h"
#include "../src/kern.h"
#include "../src/forward.h"
#include "../src/pq_quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned seed = 0x6e6d6772;
static unsigned next(void) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
static float random_float(void) { return ((int)(next() % 2001) - 1000) / 16384.0f; }
/* kern_lin's test-only serial pool: no second implementation of its recurrence. */
void kern_pool_run(pf_pool *pool, kern_task fn, void *arg, int maximum) {
    (void)pool; (void)maximum; fn(arg, 0, 1);
}
static void *allocate(size_t bytes) {
    void *p = calloc(1, bytes);
    if (!p) { fputs("CPU test allocation failed\n", stderr); exit(1); }
    return p;
}
static nm_buffer *buffer(nm_metal *ctx, size_t bytes) {
    nm_buffer *p = nm_metal_buffer_create(ctx, bytes);
    if (!p) { fputs("GPU test allocation failed\n", stderr); exit(1); }
    memset(nm_metal_buffer_contents(p), 0, bytes);
    return p;
}
static int equal(const void *expected, const void *got, size_t bytes, const char *name) {
    if (!memcmp(expected, got, bytes)) return 1;
    const float *a = expected, *b = got;
    for (size_t i = 0; i < bytes / 4; i++) if (memcmp(a + i, b + i, 4)) {
        fprintf(stderr, "%s mismatch i=%zu expected=%a got=%a\n", name, i, a[i], b[i]); break;
    }
    return 0;
}
static void fill_weights(unsigned char *p, size_t bytes) {
    for (size_t i = 0; i < bytes; i++) p[i] = next();
    for (size_t off = 0; off < bytes; off += 576) for (int i = 0; i < 8; i++) {
        __fp16 value = (i + 1) * 0.0001f;
        memcpy(p + off + i * 2, &value, 2);
    }
}
static void cpu_matmul(const pf_mat *w, const float *x, float *y, int n) {
    size_t qb = (size_t)n * w->cols, fb = qb / 32 * 4;
    int8_t *q = allocate(qb);
    float *d = allocate(fb), *fs = allocate(fb);
    kern_quant_act(x, n, w->cols, q, d, fs);
    pf_act a = {w->cols, n, q, d, fs};
    kern_q4k(w, x, &a, y, n, 0, w->rows);
    free(fs); free(d); free(q);
}

static int test_mlp(nm_metal *ctx, unsigned h, unsigned f, unsigned n, int zero) {
    size_t wb = pq_quant_nbytes(PQ_Q4K_G32, f, h);
    unsigned char *wg = allocate(wb), *wu = allocate(wb), *wd = allocate(wb);
    fill_weights(wg, wb); fill_weights(wu, wb); fill_weights(wd, wb);
    nm_buffer *g = nm_metal_a8_weights(ctx, f, h, wg, wb), *u = nm_metal_a8_weights(ctx, f, h, wu, wb);
    nm_buffer *d = nm_metal_a8_weights(ctx, h, f, wd, wb);
    if (!g || !u || !d) exit(1);
    size_t hb = (size_t)n * h * 4, fb = (size_t)n * f * 4, max = hb > fb ? hb : fb;
    nm_mlp_buffers b = {buffer(ctx, hb), buffer(ctx, fb), buffer(ctx, fb), buffer(ctx, max / 4),
                        buffer(ctx, max / 32), buffer(ctx, max / 32), buffer(ctx, hb)};
    float *x = nm_metal_buffer_contents(b.x), *gate = allocate(fb), *up = allocate(fb), *y = allocate(hb);
    for (size_t i = 0; i < hb / 4; i++) x[i] = zero ? 0 : random_float() * 16;
    pf_mat Wg = {PQ_Q4K_G32, (int)f, (int)h, wg}, Wu = {PQ_Q4K_G32, (int)f, (int)h, wu};
    pf_mat Wd = {PQ_Q4K_G32, (int)h, (int)f, wd};
    cpu_matmul(&Wg, x, gate, n); cpu_matmul(&Wu, x, up, n);
    pf_silu_mul(gate, up, (int)(n * f)); cpu_matmul(&Wd, gate, y, n);
    char error[4096];
    int ok = !nm_metal_mlp(ctx, h, f, n, g, u, d, &b, error, sizeof error);
    if (!ok) fprintf(stderr, "%s\n", error);
    else ok = equal(gate, nm_metal_buffer_contents(b.gate), fb, "MLP activation") &&
              equal(up, nm_metal_buffer_contents(b.up), fb, "MLP up") &&
              equal(y, nm_metal_buffer_contents(b.y), hb, "MLP output");
    if (ok) printf("PASS exact single-submit MLP hidden=%u ffn=%u batch=%u zero=%d\n", h, f, n, zero);
    // Each validator must reject without submitting work or corrupting state.
    nm_mlp_buffers bad = b; bad.y = bad.x;
    ok &= nm_metal_mlp(ctx, h, f, n, g, u, d, &bad, error, sizeof error) != 0;
    ok &= nm_metal_mlp(ctx, h + 1, f, n, g, u, d, &b, error, sizeof error) != 0;
    ok &= nm_metal_mlp(ctx, h, f, 65, g, u, d, &b, error, sizeof error) != 0;
    ok &= nm_metal_mlp(ctx, h, f, 0, g, u, d, &b, error, sizeof error) != 0;
    ok &= nm_metal_mlp(NULL, h, f, n, g, u, d, &b, error, sizeof error) != 0;
    nm_buffer *handles[] = {g, u, d, b.x, b.gate, b.up, b.q, b.d, b.fs, b.y};
    for (size_t i = 0; i < sizeof handles / sizeof handles[0]; i++) nm_metal_buffer_destroy(handles[i]);
    free(y); free(up); free(gate); free(wd); free(wu); free(wg);
    return ok;
}

static int test_lin(nm_metal *ctx, unsigned heads, unsigned rep, unsigned dk, unsigned dv, unsigned n) {
    unsigned stride = 2 * (heads / rep) * dk + heads * dv + 4, outstride = heads * dv + 4;
    size_t sb = (size_t)heads * dk * dv * 4, qb = (size_t)n * stride * 4, db = (size_t)n * heads * 4;
    size_t ob = (size_t)n * outstride * 4, width = (size_t)heads * (dk + dv + 1), lb = (size_t)n * width * 4;
    const unsigned keep = 3;
    size_t pb = (size_t)keep * width * 4;
    nm_lin_buffers b = {buffer(ctx, sb), buffer(ctx, qb), buffer(ctx, db), buffer(ctx, db),
                        buffer(ctx, ob), buffer(ctx, lb), buffer(ctx, pb), buffer(ctx, sb)};
    float *S = allocate(sb), *o = allocate(ob), *log = allocate(lb), *snapshot = allocate(sb);
    float *qkv = nm_metal_buffer_contents(b.qkv), *decay = nm_metal_buffer_contents(b.decay);
    float *beta = nm_metal_buffer_contents(b.beta), *pend = nm_metal_buffer_contents(b.pend);
    float *shared = nm_metal_buffer_contents(b.state);
    for (size_t i = 0; i < sb / 4; i++) S[i] = random_float();
    memcpy(shared, S, sb); memcpy(snapshot, S, sb);
    for (size_t i = 0; i < qb / 4; i++) qkv[i] = random_float();
    for (size_t i = 0; i < db / 4; i++) { decay[i] = 0.875f + fabsf(random_float()); beta[i] = 0.5f + random_float(); }
    for (size_t i = 0; i < pb / 4; i++) pend[i] = random_float();
    for (unsigned t = 0; t < keep; t++) for (unsigned h = 0; h < heads; h++)
        pend[((size_t)t * heads + h) * (dk + dv + 1) + dk + dv] = 0.9375f;
    char error[4096];
    int ok = 1;
    // Sequential calls on the same state, nonzero pending updates, logged rollback,
    // then CPU restoration into the same MTLBuffer (no invalidation/upload step).
    for (unsigned pass = 0; pass < 4 && ok; pass++) {
        int logged = pass == 1 || pass == 2, pending = pass == 2 ? keep : 0;
        if (pass == 3) { memcpy(shared, snapshot, sb); memcpy(S, snapshot, sb); }
        memset(o, 0, ob); memset(nm_metal_buffer_contents(b.output), 0, ob);
        memset(log, 0, lb); memset(nm_metal_buffer_contents(b.log), 0, lb);
        pf_lin_args a = {(int)n, (int)heads, (int)rep, (int)dk, (int)dv, (int)stride, (int)outstride,
                         qkv, decay, beta, S, o, logged ? log : NULL, pending ? pend : NULL, pending, NULL};
        kern_lin_rule(NULL, &a);
        nm_lin_shape shape = {n, heads, rep, dk, dv, stride, outstride, (unsigned)pending, (unsigned)logged};
        if (nm_metal_lin_rule(ctx, &shape, &b, error, sizeof error)) { fprintf(stderr, "%s\n", error); ok = 0; }
        else ok = equal(S, shared, sb, "DeltaNet state") && equal(o, nm_metal_buffer_contents(b.output), ob, "DeltaNet output") &&
                  (!logged || equal(log, nm_metal_buffer_contents(b.log), lb, "DeltaNet log"));
    }
    nm_lin_shape shape = {n, heads, rep, dk, dv, stride, outstride, keep, 0};
    kern_lin_apply(NULL, S, pend, keep, heads, dk, dv);
    if (nm_metal_lin_apply(ctx, &shape, &b, error, sizeof error)) { fprintf(stderr, "%s\n", error); ok = 0; }
    else ok &= equal(S, shared, sb, "DeltaNet explicit replay");
    nm_lin_shape bad = shape; bad.rep = 0;
    ok &= nm_metal_lin_rule(ctx, &bad, &b, error, sizeof error) != 0;
    bad = shape; bad.n = 65;
    ok &= nm_metal_lin_rule(ctx, &bad, &b, error, sizeof error) != 0;
    bad = shape; bad.dv++;
    ok &= nm_metal_lin_rule(ctx, &bad, &b, error, sizeof error) != 0;
    nm_lin_buffers aliased = b; aliased.output = aliased.state;
    ok &= nm_metal_lin_rule(ctx, &shape, &aliased, error, sizeof error) != 0;
    bad = shape; bad.keep = 65;
    ok &= nm_metal_lin_apply(ctx, &bad, &b, error, sizeof error) != 0;
    if (ok) printf("PASS exact persistent DeltaNet heads=%u rep=%u dk=%u dv=%u batch=%u: recurrence/log/pending/replay/restore\n",
                   heads, rep, dk, dv, n);
    nm_buffer *handles[] = {b.state, b.qkv, b.decay, b.beta, b.output, b.log, b.pend, b.work};
    for (size_t i = 0; i < sizeof handles / sizeof handles[0]; i++) nm_metal_buffer_destroy(handles[i]);
    free(snapshot); free(log); free(o); free(S);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char error[4096];
    nm_metal *ctx = nm_metal_create(argv[1], error, sizeof error);
    if (!ctx) { fprintf(stderr, "%s\n", error); return 1; }
    int ok = 1;
    unsigned batches[] = {1, 3, 7, 64};
    for (size_t i = 0; i < sizeof batches / sizeof batches[0]; i++) {
        ok &= test_mlp(ctx, 256, 768, batches[i], 0);
        ok &= test_lin(ctx, 4, 2, 32, 64, batches[i]);
    }
    ok &= test_mlp(ctx, 1024, 3584, 1, 0);
    ok &= test_mlp(ctx, 1024, 3584, 8, 0);
    ok &= test_mlp(ctx, 256, 768, 3, 1);
    ok &= test_lin(ctx, 16, 1, 128, 128, 1);
    ok &= test_lin(ctx, 2, 1, 256, 512, 3);
    nm_metal_destroy(ctx);
    if (ok) puts("PASS all GPU graph numerical, persistence and invalid-input gates");
    return ok ? 0 : 1;
}
