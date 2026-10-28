/* Experimental exact-A8 hybrid backend: packed Q4K on Metal, remaining ops on NEON. */
#ifndef NM_INFERENCE_H
#define NM_INFERENCE_H
#include "../src/kern.h"
typedef struct nm_inference nm_inference;
nm_inference *nm_inference_create(void);
void nm_inference_destroy(nm_inference *m);
int nm_inference_wants(const nm_inference *m, const pf_mat *w, int n);
void nm_inference_matmul(nm_inference *m, const pf_mat *w, const pf_act *a, float *y);
#endif
