/* Model loading: mmap a .pq file, validate it, and expose per-layer weight views.
 *
 * Nothing is copied: every pf_mat / float pointer points straight into the mapping, so
 * load time is ~0 and the OS page cache holds the weights (shared between processes). */
#ifndef PF_MODEL_H
#define PF_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "ops.h"
#include "pq_format.h"

typedef struct {
    int is_full;                 /* 1 = gated full attention, 0 = Gated DeltaNet */
    const float *in_norm;        /* [hidden], effective multiplier (1 + w) */
    const float *post_norm;      /* [hidden], effective multiplier (1 + w) */
    pf_mat gate, up, down;       /* SwiGLU MLP */

    /* linear attention (is_full == 0) */
    pf_mat qkv;                  /* [2*kd + vd, hidden] */
    pf_mat z;                    /* [vd, hidden] */
    pf_mat b, a;                 /* [lin_v_heads, hidden] */
    pf_mat out;                  /* [hidden, vd] */
    const float *conv_w;         /* [2*kd + vd][conv_k] */
    const float *A_log, *dt_bias;/* [lin_v_heads] */
    const float *A;              /* [lin_v_heads] = -exp(A_log) as stored (GGUF imports), or NULL:
                                  * a layer has exactly one of A_log and A */
    const float *lin_norm;       /* [lin_v_dim], plain w */

    /* full attention (is_full == 1) */
    pf_mat q, k, v, o;           /* q: [n_heads*head_dim*2, hidden] (query | gate per head) */
    const float *q_norm, *k_norm;/* [head_dim], effective multiplier */
} pf_layer;

typedef struct {
    void *map;
    size_t map_bytes;
    const pq_header *hdr;
    const pq_tensor_entry *dir;
    pf_mat embed;                /* [vocab, hidden] */
    pf_mat head;                 /* [vocab, hidden] output head: embed (tied) or lm_head.weight */
    const float *norm;           /* final norm [hidden] */
    pf_layer *layers;            /* [n_layers] */
} pf_model;

/* Returns PINFER_OK or a negative PINFER_E* code; on failure *m is left empty. */
int pf_model_load(pf_model *m, const char *path);
void pf_model_free(pf_model *m);

/* Directory lookup by exact name; NULL if absent. */
const pq_tensor_entry *pf_model_find(const pf_model *m, const char *name);

/* Typed views of a named tensor (used by the layers here and by the MTP head, mtp.c):
 * vec: F32 [n]; mat: [rows, cols] of a dtype this build supports. pf_model_load has already
 * checked every entry's bounds and size (check_directory). PINFER_OK or PINFER_E_FORMAT. */
int pf_model_vec(const pf_model *m, const char *name, uint32_t n, const float **out);
int pf_model_mat(const pf_model *m, const char *name, uint32_t rows, uint32_t cols,
                 pf_mat *out);

/* 64-bit fingerprint of the loaded file: FNV-1a over the header, the tensor directory and
 * three 64-byte samples (start, middle, end) of every tensor. State blobs carry it (state.c),
 * so a blob saved with one model file is refused by another, including one with the same
 * dimensions and layout but other weights. Reads ~3 cache lines per tensor (~0.1 ms). */
uint64_t pf_model_fingerprint(const pf_model *m);

#endif
