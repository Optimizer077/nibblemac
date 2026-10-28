/* Experimental exact-A8 hybrid backend: packed Q4K on Metal, remaining ops on NEON. */
#ifndef NM_INFERENCE_H
#define NM_INFERENCE_H
#include "../src/kern.h"
typedef struct nm_inference nm_inference;
nm_inference *nm_inference_create(void);
void nm_inference_destroy(nm_inference *m);
int nm_inference_wants(const nm_inference *m, const pf_mat *w, int n);
void nm_inference_matmul(nm_inference *m, const pf_mat *w, const pf_act *a, float *y);
int nm_inference_recurrence_enabled(const nm_inference *m);
void *nm_inference_state_alloc(nm_inference *m, size_t bytes);
void nm_inference_state_free(nm_inference *m, void *state);
void nm_inference_lin_rule(nm_inference *m, const pf_lin_args *a);
void nm_inference_lin_apply(nm_inference *m, float *S, const float *log, int keep,
                            int heads, int dk, int dv);
int nm_inference_mlp(nm_inference *m, const pf_mat *gate, const pf_mat *up, const pf_mat *down,
                     const float *x, float *y, int n);
#endif
