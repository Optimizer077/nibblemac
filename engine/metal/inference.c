#include "inference.h"
#include "metal.h"
#include "../src/pq_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

typedef struct {
    const void *data;
    int rows, cols;
    nm_buffer *buffer;
} weight;
struct nm_inference {
    nm_metal *context;
    weight *weights;
    size_t count, capacity, weight_bytes, minimum_bytes;
    nm_buffer *q, *d, *fs, *y;
    size_t qcap, fcap, ycap;
    unsigned long calls;
};

nm_inference *nm_inference_create(void) {
    nm_inference *m = calloc(1, sizeof *m);
    if (!m) return NULL;
    char error[4096];
    const char *path = getenv("NM_METAL_SHADER");
    m->context = nm_metal_create(path ? path : NM_SHADER_PATH, error, sizeof error);
    if (!m->context) { fprintf(stderr, "nibblemac: %s\n", error); free(m); return NULL; }
    /* Measured on the M2 Pro: offloading small layer matrices loses to CPU SDOT.
     * Default to the large vocabulary projection; keep the threshold explicit. */
    m->minimum_bytes = 64 * 1024 * 1024;
    const char *min = getenv("NM_METAL_MIN_BYTES");
    if (min) {
        char *end;
        errno = 0;
        unsigned long long value = strtoull(min, &end, 10);
        if (!*min || *min == '-' || *end || errno == ERANGE || value > SIZE_MAX) {
            fprintf(stderr, "nibblemac: invalid NM_METAL_MIN_BYTES\n");
            nm_inference_destroy(m); return NULL;
        }
        m->minimum_bytes = (size_t)value;
    }
    fprintf(stderr, "nibblemac: Metal Q4K/A8 on %s, minimum weight bytes=%zu; other ops NEON\n",
            nm_metal_device_name(m->context), m->minimum_bytes);
    return m;
}
void nm_inference_destroy(nm_inference *m) {
    if (!m) return;
    if (m->calls) fprintf(stderr, "nibblemac: Metal calls=%lu cached_matrices=%zu packed_bytes=%zu\n",
                          m->calls, m->count, m->weight_bytes);
    for (size_t i = 0; i < m->count; i++) nm_metal_buffer_destroy(m->weights[i].buffer);
    nm_metal_buffer_destroy(m->q); nm_metal_buffer_destroy(m->d);
    nm_metal_buffer_destroy(m->fs); nm_metal_buffer_destroy(m->y);
    nm_metal_destroy(m->context); free(m->weights); free(m);
}
int nm_inference_wants(const nm_inference *m, const pf_mat *w, int n) {
    return n > 0 && w->dtype == PQ_Q4K_G32 &&
        pq_quant_nbytes(w->dtype, w->rows, w->cols) >= m->minimum_bytes;
}
static void grow(nm_inference *m, nm_buffer **p, size_t *capacity, size_t need) {
    if (*capacity >= need) return;
    nm_buffer *b = nm_metal_buffer_create(m->context, need);
    if (!b) { fprintf(stderr, "nibblemac: Metal scratch allocation failed\n"); abort(); }
    nm_metal_buffer_destroy(*p); *p = b; *capacity = need;
}
void nm_inference_matmul(nm_inference *m, const pf_mat *w, const pf_act *a, float *y) {
    nm_buffer *buffer = NULL;
    for (size_t i = 0; i < m->count; i++)
        if (m->weights[i].data == w->data && m->weights[i].rows == w->rows && m->weights[i].cols == w->cols)
            buffer = m->weights[i].buffer;
    if (!buffer) {
        if (m->count == m->capacity) {
            size_t capacity = m->capacity ? m->capacity * 2 : 64;
            weight *p = realloc(m->weights, capacity * sizeof *p);
            if (!p) abort();
            m->weights = p; m->capacity = capacity;
        }
        size_t bytes = pq_quant_nbytes(w->dtype, w->rows, w->cols);
        if (m->weight_bytes + bytes > nm_metal_recommended_memory(m->context) / 2) {
            fprintf(stderr, "nibblemac: immutable weight cache exceeds memory budget\n"); abort();
        }
        buffer = nm_metal_a8_weights(m->context, w->rows, w->cols, w->data, bytes);
        if (!buffer) { fprintf(stderr, "nibblemac: Metal weight allocation failed\n"); abort(); }
        m->weights[m->count++] = (weight){w->data, w->rows, w->cols, buffer};
        m->weight_bytes += bytes;
    }
    const size_t qb = (size_t)a->n * a->cols;
    const size_t fb = (size_t)a->n * (a->cols / 32) * 4;
    const size_t yb = (size_t)a->n * w->rows * 4;
    grow(m, &m->q, &m->qcap, qb);
    if (m->fcap < fb) {
        size_t cap = m->fcap;
        grow(m, &m->d, &cap, fb);
        grow(m, &m->fs, &m->fcap, fb);
    }
    grow(m, &m->y, &m->ycap, yb);
    memcpy(nm_metal_buffer_contents(m->q), a->q, qb);
    memcpy(nm_metal_buffer_contents(m->d), a->d, fb);
    memcpy(nm_metal_buffer_contents(m->fs), a->fsum, fb);
    char error[4096];
    if (nm_metal_matmul_a8(m->context, w->rows, w->cols, a->n, buffer, m->q, m->d, m->fs,
                            m->y, error, sizeof error)) {
        fprintf(stderr, "nibblemac: Metal inference failed: %s\n", error); abort();
    }
    memcpy(y, nm_metal_buffer_contents(m->y), yb);
    m->calls++;
}
