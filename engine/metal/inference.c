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
typedef struct { void *pointer; nm_buffer *buffer; size_t bytes; } state_buffer;
struct nm_inference {
    nm_metal *context;
    weight *weights;
    size_t count, capacity, weight_bytes, minimum_bytes;
    nm_buffer *q, *d, *fs, *y;
    size_t qcap, fcap, ycap;
    unsigned long calls;
    int recurrence, mlp;
    unsigned long lin_calls, mlp_calls;
    state_buffer states[32];
    size_t state_count, state_bytes;
    nm_lin_buffers lin;
    size_t lin_qcap, lin_dcap, lin_bcap, lin_ocap, lin_lcap, lin_pcap, lin_wcap;
    nm_mlp_buffers mlp_buffers;
    size_t mlp_xcap, mlp_gcap, mlp_ucap, mlp_qcap, mlp_dcap, mlp_fcap, mlp_ycap;
};

static int option(const char *name) {
    const char *value = getenv(name);
    if (value && strcmp(value, "0") && strcmp(value, "1")) {
        fprintf(stderr, "nibblemac: %s must be 0 or 1\n", name); return -1;
    }
    return value && !strcmp(value, "1");
}

nm_inference *nm_inference_create(void) {
    nm_inference *m = calloc(1, sizeof *m);
    if (!m) return NULL;
    char error[4096];
    const char *path = getenv("NM_METAL_SHADER");
    m->context = nm_metal_create(path ? path : NM_SHADER_PATH, error, sizeof error);
    if (!m->context) { fprintf(stderr, "nibblemac: %s\n", error); free(m); return NULL; }
    m->recurrence = option("NM_METAL_RECURRENCE");
    m->mlp = option("NM_METAL_MLP");
    if (m->recurrence < 0 || m->mlp < 0) { nm_inference_destroy(m); return NULL; }
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
    if (m->recurrence || m->mlp) fprintf(stderr, "nibblemac: experimental persistent_DeltaNet=%d single_submit_MLP=%d\n",
                                       m->recurrence, m->mlp);
    return m;
}
void nm_inference_destroy(nm_inference *m) {
    if (!m) return;
    if (m->calls) fprintf(stderr, "nibblemac: Metal calls=%lu cached_matrices=%zu packed_bytes=%zu\n",
                          m->calls, m->count, m->weight_bytes);
    if (m->lin_calls || m->mlp_calls) fprintf(stderr, "nibblemac: Metal graph lin_calls=%lu mlp_calls=%lu shared_state_bytes=%zu\n",
                                             m->lin_calls, m->mlp_calls, m->state_bytes);
    for (size_t i = 0; i < m->count; i++) nm_metal_buffer_destroy(m->weights[i].buffer);
    nm_metal_buffer_destroy(m->q); nm_metal_buffer_destroy(m->d);
    nm_metal_buffer_destroy(m->fs); nm_metal_buffer_destroy(m->y);
    for (size_t i = 0; i < m->state_count; i++) nm_metal_buffer_destroy(m->states[i].buffer);
    nm_buffer *lin[] = {m->lin.qkv, m->lin.decay, m->lin.beta, m->lin.output, m->lin.log, m->lin.pend, m->lin.work};
    for (size_t i = 0; i < sizeof lin / sizeof lin[0]; i++) nm_metal_buffer_destroy(lin[i]);
    nm_buffer *mlp[] = {m->mlp_buffers.x, m->mlp_buffers.gate, m->mlp_buffers.up, m->mlp_buffers.q,
                       m->mlp_buffers.d, m->mlp_buffers.fs, m->mlp_buffers.y};
    for (size_t i = 0; i < sizeof mlp / sizeof mlp[0]; i++) nm_metal_buffer_destroy(mlp[i]);
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
static nm_buffer *get_weight(nm_inference *m, const pf_mat *w) {
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
    return buffer;
}
void nm_inference_matmul(nm_inference *m, const pf_mat *w, const pf_act *a, float *y) {
    nm_buffer *buffer = get_weight(m, w);
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

int nm_inference_recurrence_enabled(const nm_inference *m) { return m && m->recurrence; }
void *nm_inference_state_alloc(nm_inference *m, size_t bytes) {
    if (!bytes || m->state_count == 32 || m->state_bytes + bytes > nm_metal_recommended_memory(m->context) / 8)
        return NULL;
    nm_buffer *b = nm_metal_buffer_create(m->context, bytes);
    if (!b) return NULL;
    void *pointer = nm_metal_buffer_contents(b);
    memset(pointer, 0, bytes);
    m->states[m->state_count++] = (state_buffer){pointer, b, bytes};
    m->state_bytes += bytes;
    return pointer;
}
void nm_inference_state_free(nm_inference *m, void *state) {
    if (!state) return;
    for (size_t i = 0; i < m->state_count; i++) if (m->states[i].pointer == state) {
        nm_metal_buffer_destroy(m->states[i].buffer);
        m->states[i] = m->states[--m->state_count];
        return;
    }
    fprintf(stderr, "nibblemac: unregistered shared recurrence state\n"); abort();
}
static nm_buffer *get_state(nm_inference *m, float *state) {
    for (size_t i = 0; i < m->state_count; i++) if (m->states[i].pointer == state) return m->states[i].buffer;
    fprintf(stderr, "nibblemac: missing persistent state buffer\n"); abort();
}
void nm_inference_lin_rule(nm_inference *m, const pf_lin_args *a) {
    if (a->n <= 0 || a->n > 64 || a->dk > 256 || a->keep < 0 || a->keep > 64) abort();
    nm_lin_buffers *b = &m->lin;
    const size_t qb = (size_t)a->n * a->qkv_stride * 4;
    const size_t db = (size_t)a->n * a->heads * 4;
    const size_t ob = (size_t)a->n * a->o_stride * 4;
    const size_t width = (size_t)a->heads * (a->dk + a->dv + 1);
    const size_t sb = (size_t)a->heads * a->dk * a->dv * 4;
    grow(m, &b->qkv, &m->lin_qcap, qb);
    grow(m, &b->decay, &m->lin_dcap, db);
    grow(m, &b->beta, &m->lin_bcap, db);
    grow(m, &b->output, &m->lin_ocap, ob);
    grow(m, &b->log, &m->lin_lcap, a->log ? (size_t)a->n * width * 4 : 4);
    grow(m, &b->pend, &m->lin_pcap, a->keep ? (size_t)a->keep * width * 4 : 4);
    grow(m, &b->work, &m->lin_wcap, a->log ? sb : 4);
    b->state = get_state(m, a->S);
    memcpy(nm_metal_buffer_contents(b->qkv), a->qkv, qb);
    memcpy(nm_metal_buffer_contents(b->decay), a->decay, db);
    memcpy(nm_metal_buffer_contents(b->beta), a->beta, db);
    if (a->keep) memcpy(nm_metal_buffer_contents(b->pend), a->pend, (size_t)a->keep * width * 4);
    nm_lin_shape shape = {a->n, a->heads, a->rep, a->dk, a->dv, a->qkv_stride, a->o_stride, a->keep, a->log != NULL};
    char error[4096];
    if (nm_metal_lin_rule(m->context, &shape, b, error, sizeof error)) {
        fprintf(stderr, "nibblemac: DeltaNet failed: %s\n", error); abort();
    }
    // Only meaningful output/log ranges are copied. State is already shared:
    // never download/upload the recurrent matrix between inference calls.
    for (int t = 0; t < a->n; t++)
        memcpy(a->o + (size_t)t * a->o_stride,
               (float *)nm_metal_buffer_contents(b->output) + (size_t)t * a->o_stride,
               (size_t)a->heads * a->dv * 4);
    if (a->log && a->n > 1)
        memcpy(a->log + width, (float *)nm_metal_buffer_contents(b->log) + width,
               (size_t)(a->n - 1) * width * 4);
    m->lin_calls++;
}
void nm_inference_lin_apply(nm_inference *m, float *S, const float *log, int keep,
                            int heads, int dk, int dv) {
    if (!keep) return;
    if (keep < 0 || keep > 64) abort();
    nm_lin_buffers *b = &m->lin;
    size_t bytes = (size_t)keep * heads * (dk + dv + 1) * 4;
    grow(m, &b->pend, &m->lin_pcap, bytes);
    memcpy(nm_metal_buffer_contents(b->pend), log, bytes);
    b->state = get_state(m, S);
    nm_lin_shape shape = {0, heads, 1, dk, dv, 0, 0, keep, 0};
    char error[4096];
    if (nm_metal_lin_apply(m->context, &shape, b, error, sizeof error)) {
        fprintf(stderr, "nibblemac: DeltaNet replay failed: %s\n", error); abort();
    }
    m->lin_calls++;
}
int nm_inference_mlp(nm_inference *m, const pf_mat *gate, const pf_mat *up, const pf_mat *down,
                     const float *x, float *y, int n) {
    if (!m->mlp || gate->dtype != PQ_Q4K_G32 || up->dtype != PQ_Q4K_G32 || down->dtype != PQ_Q4K_G32 ||
        gate->rows != up->rows || gate->cols != up->cols || down->cols != gate->rows ||
        down->rows != gate->cols || gate->cols % 256 || gate->rows % 256 || n < 1 || n > 64) return 0;
    nm_buffer *wg = get_weight(m, gate), *wu = get_weight(m, up), *wd = get_weight(m, down);
    size_t hb = (size_t)n * gate->cols * 4, fb = (size_t)n * gate->rows * 4;
    size_t qb = (hb > fb ? hb : fb) / 4;
    nm_mlp_buffers *b = &m->mlp_buffers;
    grow(m, &b->x, &m->mlp_xcap, hb);
    grow(m, &b->y, &m->mlp_ycap, hb);
    grow(m, &b->gate, &m->mlp_gcap, fb);
    grow(m, &b->up, &m->mlp_ucap, fb);
    grow(m, &b->q, &m->mlp_qcap, qb);
    grow(m, &b->d, &m->mlp_dcap, qb / 32 * 4);
    grow(m, &b->fs, &m->mlp_fcap, qb / 32 * 4);
    memcpy(nm_metal_buffer_contents(b->x), x, hb);
    char error[4096];
    if (nm_metal_mlp(m->context, gate->cols, gate->rows, n, wg, wu, wd, b, error, sizeof error)) {
        fprintf(stderr, "nibblemac: MLP graph failed: %s\n", error); abort();
    }
    memcpy(y, nm_metal_buffer_contents(b->y), hb);
    m->calls++; m->mlp_calls++;
    return 1;
}
