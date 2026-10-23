/* Public API (pinfer.h): context allocation, state reset, eval. The math is in forward.c. */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "spec.h"

#define STR2(x) #x
#define STR(x) STR2(x)

pinfer_params pinfer_default_params(void) {
    pinfer_params p = {4, 4096, PINFER_KV_F32, 0.0f};
    return p;
}

const char *pinfer_version(void) {
    return STR(PINFER_VERSION_MAJOR) "." STR(PINFER_VERSION_MINOR) "." STR(PINFER_VERSION_PATCH);
}

static size_t max3(size_t a, size_t b, size_t c) {
    size_t m = a > b ? a : b;
    return m > c ? m : c;
}

static void free_layer_state(float **v, int n_layers) {
    if (!v) return;
    for (int l = 0; l < n_layers; l++) free(v[l]);
    free(v);
}

/* Per-layer state arrays: one calloc per used layer (zero = the reference's initial state). */
static float **alloc_layer_state(pinfer_ctx *c, int want_full, size_t bytes) {
    float **v = calloc((size_t)c->n_layers, sizeof(float *));
    if (!v) return NULL;
    for (int l = 0; l < c->n_layers; l++) {
        if (c->model.layers[l].is_full != want_full) continue;
        v[l] = calloc(bytes, 1);
        if (!v[l]) { free_layer_state(v, c->n_layers); return NULL; }
    }
    return v;
}

static void set_dims(pinfer_ctx *c) {
    const pq_header *h = c->model.hdr;
    c->H = (int)h->hidden;
    c->ffn = (int)h->ffn;
    c->n_layers = (int)h->n_layers;
    c->vocab = (int)h->vocab;
    c->n_heads = (int)h->n_heads;
    c->n_kv = (int)h->n_kv_heads;
    c->hd = (int)h->head_dim;
    c->rot = (int)h->rot_dim;
    c->lk_heads = (int)h->lin_k_heads;
    c->lv_heads = (int)h->lin_v_heads;
    c->lk_dim = (int)h->lin_k_dim;
    c->lv_dim = (int)h->lin_v_dim;
    c->conv_k = (int)h->conv_k;
    c->kd = c->lk_heads * c->lk_dim;
    c->vd = c->lv_heads * c->lv_dim;
    c->conv_dim = 2 * c->kd + c->vd;
}

static int alloc_buffers(pinfer_ctx *c) {
    const size_t B = PF_MAX_BATCH, kvw = (size_t)c->n_kv * c->hd;
    c->conv = alloc_layer_state(c, 0, ((size_t)(c->conv_k - 1) * c->conv_dim + 1) * sizeof(float));
    c->rec = alloc_layer_state(c, 0, (size_t)c->lv_heads * c->lk_dim * c->lv_dim * sizeof(float));
    c->kc = alloc_layer_state(c, 1, pf_kv_bytes(c));
    c->vc = alloc_layer_state(c, 1, pf_kv_bytes(c));
    if (!c->conv || !c->rec || !c->kc || !c->vc) return PINFER_E_NOMEM;
    c->x = malloc(B * c->H * sizeof(float));
    c->xn = malloc(B * c->H * sizeof(float));
    c->out = malloc(B * c->H * sizeof(float));
    /* ta rows also hold the MTP head's fc input [embedding | hidden], 2H floats (mtp.c); every
     * Qwen3.5 size has ffn >= 3H, but the header allows other relations (tests/test_dims.py) */
    const size_t ta_row = max3(max3(c->conv_dim, (size_t)c->n_heads * c->hd * 2, c->ffn),
                               2 * (size_t)c->H, 1);
    c->ta = malloc(B * ta_row * sizeof(float));
    c->tb = malloc(B * max3(c->vd, kvw, c->ffn) * sizeof(float));
    c->tc = malloc(B * kvw * sizeof(float));
    c->to = malloc(B * max3(c->vd, (size_t)c->n_heads * c->hd, 1) * sizeof(float));
    c->ba = malloc(B * c->lv_heads * sizeof(float));
    c->aa = malloc(B * c->lv_heads * sizeof(float));
    c->rope_cos = malloc(B * (c->rot / 2 + 1) * sizeof(float));
    c->rope_sin = malloc(B * (c->rot / 2 + 1) * sizeof(float));
    c->att = malloc((size_t)PF_ATTN_TB * c->n_heads * c->max_ctx * sizeof(float));
    if (!c->x || !c->xn || !c->out || !c->ta || !c->tb || !c->tc || !c->to || !c->ba ||
        !c->aa || !c->rope_cos || !c->rope_sin || !c->att)
        return PINFER_E_NOMEM;
    return PINFER_OK;
}

static int enable_gpu(pinfer_ctx *c, float share) {
    /* MLP candidates first; backend sorts by size, preserving this priority on ties.
     * Embedding and output head are excluded (lookup/decode, not prefill body matmuls). */
    const pf_mat **mats = calloc((size_t)c->n_layers * 13, sizeof(*mats));
    if (!mats) {
        fprintf(stderr, "pinfer: GPU setup allocation failed; falling back to CPU-only\n");
        return PINFER_OK;
    }
    int n = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const pf_layer *layer = &c->model.layers[l];
        mats[n++] = &layer->gate; mats[n++] = &layer->up; mats[n++] = &layer->down;
    }
    for (int l = 0; l < c->n_layers; l++) {
        const pf_layer *layer = &c->model.layers[l];
        if (layer->is_full) {
            mats[n++] = &layer->q; mats[n++] = &layer->k;
            mats[n++] = &layer->v; mats[n++] = &layer->o;
        } else {
            mats[n++] = &layer->qkv; mats[n++] = &layer->z;
            mats[n++] = &layer->b; mats[n++] = &layer->a; mats[n++] = &layer->out;
        }
    }
    int rc = pf_pool_gpu_enable(c->pool, mats, n, share);
    free(mats);
    if (rc) fprintf(stderr, "pinfer: requested GPU split unavailable; falling back to CPU-only\n");
    return PINFER_OK;
}

pinfer_ctx *pinfer_load(const char *path, const pinfer_params *params, int *err) {
    pinfer_params p = params ? *params : pinfer_default_params();
    if (p.n_threads <= 0) p.n_threads = 4;
    if (p.max_ctx <= 0) p.max_ctx = 4096;
    int rc = PINFER_E_NOMEM;
    pinfer_ctx *c = path ? calloc(1, sizeof *c) : NULL;
    if (!path) rc = PINFER_E_ARG;
    if (c && (!isfinite(p.gpu_split) || p.gpu_split < 0.0f || p.gpu_split > 1.0f)) {
        fprintf(stderr, "pinfer: gpu_split must be finite and in [0,1]\n");
        rc = PINFER_E_ARG;
    } else if (c && (p.kv_dtype < PINFER_KV_F32 || p.kv_dtype > PINFER_KV_I8)) {
        fprintf(stderr, "pinfer: unknown kv_dtype %d\n", p.kv_dtype);
        rc = PINFER_E_ARG;
    } else if (c) {
        rc = pf_model_load(&c->model, path);
        if (rc == PINFER_OK) {
            set_dims(c);
            c->max_ctx = p.max_ctx;
            c->kv_dtype = p.kv_dtype;
            c->batch = PF_MAX_BATCH;
            /* forward_lin.c keeps a 512-float stack scratch and kern_lin.c walks the value
             * columns in chunks of 32; forward_attn.c stack rows of PF_KV_MAX_HD floats; the
             * int8 scales need 4-byte-aligned row data */
            if (c->lv_dim > 512 || c->lv_dim % 32 || c->hd > PF_KV_MAX_HD || c->hd % 4) {
                fprintf(stderr, "pinfer: lin_v_dim %d (needs <= 512, multiple of 32) or "
                        "head_dim %d not supported\n", c->lv_dim, c->hd);
                rc = PINFER_E_FORMAT;
            } else {
                rc = alloc_buffers(c);
            }
        }
        if (rc == PINFER_OK) {
            c->pool = pf_pool_create(p.n_threads);
            if (!c->pool) rc = PINFER_E_NOMEM;
        }
        if (rc == PINFER_OK && p.gpu_split > 0.0f) rc = enable_gpu(c, p.gpu_split);
    }
    if (rc != PINFER_OK) {
        pinfer_free(c);
        c = NULL;
    }
    if (err) *err = rc;
    return c;
}

void pinfer_free(pinfer_ctx *c) {
    if (!c) return;
    pf_spec_free(c);
    int nl = c->n_layers;
    free_layer_state(c->conv, nl);
    free_layer_state(c->rec, nl);
    free_layer_state(c->kc, nl);
    free_layer_state(c->vc, nl);
    float *bufs[] = {c->x, c->xn, c->out, c->ta, c->tb, c->tc, c->to, c->ba, c->aa,
                     c->rope_cos, c->rope_sin, c->att};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) free(bufs[i]);
    free(c->ck_buf);
    if (c->pool) pf_pool_destroy(c->pool);
    if (c->model.map) pf_model_free(&c->model);
    free(c);
}

void pinfer_reset(pinfer_ctx *c) {
    if (!c) return;
    for (int l = 0; l < c->n_layers; l++) {
        if (c->model.layers[l].is_full) continue;  /* KV entries past n_past are never read */
        memset(c->conv[l], 0, (size_t)(c->conv_k - 1) * c->conv_dim * sizeof(float));
        memset(c->rec[l], 0, (size_t)c->lv_heads * c->lk_dim * c->lv_dim * sizeof(float));
    }
    c->n_past = 0;
    c->mtp_rows = 0;
    c->ck_valid = 0;   /* new tokens will overwrite the checkpoint's KV rows (state.c) */
}

int pinfer_eval(pinfer_ctx *c, const int32_t *tokens, int n, float *logits) {
    return pf_forward(c, tokens, n, logits, NULL, NULL);
}

int pinfer_eval_select(pinfer_ctx *c, const int32_t *tokens, int n, const int32_t *ids,
                       int n_ids, float *out) {
    return pf_forward_select(c, tokens, n, ids, n_ids, out);
}

int pinfer_n_vocab(const pinfer_ctx *c) { return c ? c->vocab : 0; }
int pinfer_n_past(const pinfer_ctx *c) { return c ? c->n_past : 0; }
int pinfer_eos_id(const pinfer_ctx *c) { return c ? (int)c->model.hdr->eos_id : -1; }

/* first maximum wins, like torch.argmax (forward_ops.c) */
int pinfer_argmax(const float *logits, int n) {
    const double ts = pf_stats_now();
    const int i = pf_argmax(logits, n);
    pf_stats_add(PFS_ARGMAX, PFS_K1, ts);
    return i;
}
