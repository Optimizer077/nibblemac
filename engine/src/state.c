/* Conversation state snapshot / restore (pinfer_state_size / save / load, pinfer.h).
 *
 * The blob holds everything the context keeps for the tokens consumed so far, so restoring
 * it and feeding more tokens gives bit-identical results to never having left:
 *   header (dims, n_past, kv_dtype, MTP rows)
 *   per linear-attention layer: conv state [(conv_k-1)][conv_dim], recurrent state S
 *   per full-attention layer:   K rows [n_kv][n_past] then V rows, in the cache's storage
 *                               format (plus the int8 scales, forward_kv.c)
 *   MTP head (if its cache holds rows): K rows [n_kv][mtp_rows] then V rows
 * Sizes for Qwen3.5-0.8B, fp32 cache: 20.2 MB + 24 KB per token (+ 4 KB per MTP row). */
#include <string.h>

#include "forward.h"
#include "mtp.h"
#include "spec.h"

#define STATE_MAGIC "PINFERS"
#define STATE_VERSION 1u

typedef struct {
    char magic[8];
    uint32_t version, n_past, kv_dtype, mtp_rows;
    uint32_t n_layers, hidden, n_kv, head_dim, lk_heads, lv_heads, lk_dim, lv_dim, conv_k;
    uint32_t full_mask;
    uint32_t reserved[2];
} state_hdr;   /* 72 bytes */

static size_t lin_bytes(const pinfer_ctx *c) {
    return ((size_t)(c->conv_k - 1) * c->conv_dim + (size_t)c->lv_heads * c->lk_dim * c->lv_dim)
           * sizeof(float);
}

/* Bytes of one cache's rows [0, n) over all groups, scales included. */
static size_t rows_bytes(const pinfer_ctx *c, int n) {
    return (size_t)c->n_kv * (size_t)n * (pf_kv_row_bytes(c->kv_dtype, c->hd) +
                                          (c->kv_dtype == PF_KV_I8 ? sizeof(float) : 0));
}

static int count_layers(const pinfer_ctx *c, int full) {
    int n = 0;
    for (int l = 0; l < c->n_layers; l++) n += c->model.layers[l].is_full == full;
    return n;
}

static int mtp_rows(const pinfer_ctx *c) {
    return c->mtp && c->mtp_rows > 0 ? (c->mtp_rows < c->n_past ? c->mtp_rows : c->n_past) : 0;
}

static size_t state_bytes(const pinfer_ctx *c, int n_past, int n_mtp) {
    return sizeof(state_hdr) + (size_t)count_layers(c, 0) * lin_bytes(c) +
           (size_t)count_layers(c, 1) * 2 * rows_bytes(c, n_past) + 2 * rows_bytes(c, n_mtp);
}

size_t pinfer_state_size(const pinfer_ctx *c) {
    return c ? state_bytes(c, c->n_past, mtp_rows(c)) : 0;
}

/* Copy rows [0, n) of every group between a cache and a blob cursor (dir 0 = cache -> blob,
 * 1 = blob -> cache); the int8 scales follow the rows. Group g's rows and scales are
 * contiguous in the cache ([n_kv][max_ctx] layout), so this is 2 memcpy per group. */
static char *copy_rows(const pinfer_ctx *c, void *cache, int n, char *cur, int dir) {
    const size_t rb = pf_kv_row_bytes(c->kv_dtype, c->hd);
    const size_t gstride = (size_t)c->max_ctx * rb, len = (size_t)n * rb;
    float *scales = (float *)((char *)cache + (size_t)c->n_kv * gstride);
    for (int g = 0; g < c->n_kv; g++) {
        char *row = (char *)cache + g * gstride;
        if (dir == 0) memcpy(cur, row, len); else memcpy(row, cur, len);
        cur += len;
        if (c->kv_dtype == PF_KV_I8) {
            float *sc = scales + (size_t)g * c->max_ctx;
            const size_t sl = (size_t)n * sizeof(float);
            if (dir == 0) memcpy(cur, sc, sl); else memcpy(sc, cur, sl);
            cur += sl;
        }
    }
    return cur;
}

static char *copy_lin(const pinfer_ctx *c, int l, char *cur, int dir) {
    const size_t cb = (size_t)(c->conv_k - 1) * c->conv_dim * sizeof(float);
    const size_t sb = (size_t)c->lv_heads * c->lk_dim * c->lv_dim * sizeof(float);
    if (dir == 0) { memcpy(cur, c->conv[l], cb); memcpy(cur + cb, c->rec[l], sb); }
    else { memcpy(c->conv[l], cur, cb); memcpy(c->rec[l], cur + cb, sb); }
    return cur + cb + sb;
}

int pinfer_state_save(const pinfer_ctx *c, void *buf, size_t cap) {
    if (!c || !buf) return PINFER_E_ARG;
    const int n_mtp = mtp_rows(c);
    if (cap < state_bytes(c, c->n_past, n_mtp)) return PINFER_E_ARG;
    const pq_header *h = c->model.hdr;
    state_hdr hdr = {STATE_MAGIC, STATE_VERSION, (uint32_t)c->n_past, (uint32_t)c->kv_dtype,
                     (uint32_t)n_mtp, h->n_layers, h->hidden, h->n_kv_heads, h->head_dim,
                     h->lin_k_heads, h->lin_v_heads, h->lin_k_dim, h->lin_v_dim, h->conv_k,
                     h->full_attn_mask, {0, 0}};
    char *cur = buf;
    memcpy(cur, &hdr, sizeof hdr);
    cur += sizeof hdr;
    for (int l = 0; l < c->n_layers; l++)
        if (!c->model.layers[l].is_full) cur = copy_lin(c, l, cur, 0);
    for (int l = 0; l < c->n_layers; l++)
        if (c->model.layers[l].is_full) {
            cur = copy_rows(c, c->kc[l], c->n_past, cur, 0);
            cur = copy_rows(c, c->vc[l], c->n_past, cur, 0);
        }
    if (n_mtp > 0) {
        cur = copy_rows(c, c->mtp->kc, n_mtp, cur, 0);
        copy_rows(c, c->mtp->vc, n_mtp, cur, 0);
    }
    return PINFER_OK;
}

int pinfer_state_load(pinfer_ctx *c, const void *buf, size_t len) {
    if (!c || !buf || len < sizeof(state_hdr)) return PINFER_E_ARG;
    state_hdr hdr;
    memcpy(&hdr, buf, sizeof hdr);
    const pq_header *h = c->model.hdr;
    if (memcmp(hdr.magic, STATE_MAGIC, 8) != 0 || hdr.version != STATE_VERSION)
        return PINFER_E_FORMAT;
    if (hdr.n_layers != h->n_layers || hdr.hidden != h->hidden || hdr.n_kv != h->n_kv_heads ||
        hdr.head_dim != h->head_dim || hdr.lk_heads != h->lin_k_heads ||
        hdr.lv_heads != h->lin_v_heads || hdr.lk_dim != h->lin_k_dim ||
        hdr.lv_dim != h->lin_v_dim || hdr.conv_k != h->conv_k ||
        hdr.full_mask != h->full_attn_mask || hdr.kv_dtype != (uint32_t)c->kv_dtype)
        return PINFER_E_FORMAT;   /* another model, or another KV storage format */
    if (hdr.n_past > (uint32_t)c->max_ctx) return PINFER_E_CTX_FULL;
    if (hdr.mtp_rows > hdr.n_past || len < state_bytes(c, (int)hdr.n_past, (int)hdr.mtp_rows))
        return PINFER_E_FORMAT;
    /* MTP rows need the head's cache: allocate it now (a model without the head cannot have
     * produced them, so the dims check above already rejected that blob... unless the file
     * was converted with --no-mtp: then the rows are simply dropped). */
    int n_mtp = (int)hdr.mtp_rows;
    if (n_mtp > 0 && !c->mtp && pf_mtp_init(c) != PINFER_OK) n_mtp = 0;
    const char *cur = (const char *)buf + sizeof hdr;
    for (int l = 0; l < c->n_layers; l++)
        if (!c->model.layers[l].is_full) cur = copy_lin(c, l, (char *)cur, 1);
    for (int l = 0; l < c->n_layers; l++)
        if (c->model.layers[l].is_full) {
            cur = copy_rows(c, c->kc[l], (int)hdr.n_past, (char *)cur, 1);
            cur = copy_rows(c, c->vc[l], (int)hdr.n_past, (char *)cur, 1);
        }
    if (n_mtp > 0) {
        cur = copy_rows(c, c->mtp->kc, n_mtp, (char *)cur, 1);
        copy_rows(c, c->mtp->vc, n_mtp, (char *)cur, 1);
    }
    if (c->mtp) {   /* rows without saved values must not hold stale memory */
        pf_kv_zero_rows(c, c->mtp->kc, n_mtp, (int)hdr.n_past);
        pf_kv_zero_rows(c, c->mtp->vc, n_mtp, (int)hdr.n_past);
    }
    c->n_past = (int)hdr.n_past;
    c->mtp_rows = n_mtp;
    if (c->spec) c->spec->m = NULL;   /* no pending draft: the next generation re-drafts */
    return PINFER_OK;
}
