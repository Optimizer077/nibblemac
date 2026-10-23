/* Private opt-in batched Q4 backend. */
#ifndef PF_GPU_H
#define PF_GPU_H
#include "kern.h"
typedef struct pf_gpu pf_gpu;
pf_gpu *pf_gpu_create(const pf_mat *const *weights, int count, float share);
void pf_gpu_destroy(pf_gpu *gpu);
int pf_gpu_rows(const pf_gpu *gpu, const pf_mat *weight);
/* All activation groups are quantized before begin. Submits all groups asynchronously. */
int pf_gpu_begin(pf_gpu *gpu, const pf_mat *weight, const pf_act *groups, int count);
int pf_gpu_finish(pf_gpu *gpu, float *y); /* wait + copy into the full matrix's row stride */
#endif
