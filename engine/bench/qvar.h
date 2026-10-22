/* Kernel variants for tbench (KERNELS.md Part 5): alternative Q4_G32 / Q4K_G32 kernels with
 * the kern_fn contract of kern.h, compared with kern_q4 / kern_q4k for speed and bit-exact
 * output. A variant may need its own weight layout, built once by `prep` (in memory, at "load
 * time"; the .pq format is unchanged) and passed to `fn` through W->data's companion `aux`. */
#ifndef QVAR_H
#define QVAR_H

#include "kern.h"

typedef void (*qvar_fn)(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n,
                        int r0, int r1);
typedef void *(*qvar_prep)(const pf_mat *W, size_t *bytes); /* NULL: no repack */

typedef struct {
    const char *name;
    unsigned dtypes;    /* bit mask of (1 << PQ_Q4_G32) | (1 << PQ_Q4K_G32) */
    qvar_prep prep;
    qvar_fn fn;
    const char *about;
    int q4asm; /* kern_q4_asm while this variant runs (ref: 0 = the C loops) */
} qvar;

extern const qvar QVARS[];
extern const int NQVARS;

/* the variants (qvar_*.c) */
void qv_ref(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1);
void qv_lut(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1);
void *qv_lut_prep(const pf_mat *W, size_t *bytes);
void qv_isq(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1);
void qv_q4s(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1);
void qv_q4m(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1);

#endif
