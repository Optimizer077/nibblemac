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
    pf_mat embed;                /* [vocab, hidden]; also the tied output head */
    const float *norm;           /* final norm [hidden] */
    pf_layer *layers;            /* [n_layers] */
} pf_model;

/* Returns PINFER_OK or a negative PINFER_E* code; on failure *m is left empty. */
int pf_model_load(pf_model *m, const char *path);
void pf_model_free(pf_model *m);

/* Directory lookup by exact name; NULL if absent. */
const pq_tensor_entry *pf_model_find(const pf_model *m, const char *name);

#endif
