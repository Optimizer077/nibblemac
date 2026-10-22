/* Internal interface of the NEON kernels behind matvec_fast.c (not used by forward.c).
 *
 * A kernel computes Y[v][r] = sum_c W[r][c] * X[v][c] for v < n and rows r0 <= r < r1,
 * with Y laid out [n][W->rows]. r0 and r1 are multiples of 4 except possibly r1 == rows
 * for the float dtypes. Quantized kernels read the int8-quantized activations in `A`
 * (built once per call by kern_quant_act) instead of X.
 */
#ifndef PF_KERN_H
#define PF_KERN_H

#include <stddef.h>
#include <stdint.h>

#include "ops.h"

#define KERN_ACT_BLOCK 32 /* activation quantization block (columns per scale) */
#define KERN_MAX_NV 8     /* vectors handled per pass over a weight tile */

/* Activations quantized per block of 32 columns: x[v][c] ~ d * q. Block-major layout, so
 * the data of one column block for all n vectors is contiguous (one base pointer + small
 * immediate offsets in the kernels' inner loop): element (b, v) is at index b * n + v. */
typedef struct {
    int cols, n;
    const int8_t *q;     /* [cols/32][n][32], values in [-127, 127] */
    const float *d;      /* [cols/32][n] scale */
    const float *fsum;   /* [cols/32][n] d * sum(q) = sum of the dequantized block */
} pf_act;

typedef void (*kern_fn)(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n,
                        int r0, int r1);

/* Software prefetch distance in bytes ahead of the weight stream (0 = off). A tuning knob
 * for kbench; the default is the best value measured on the Pi (see KERNELS.md). */
extern int kern_prefetch;

/* pf_matmul splits batches into groups that stream the weights again (matvec_fast.c): F32 /
 * BF16 calls of more than kern_act_group vectors into equal groups of at most kern_act_group;
 * quantized calls of more than kern_qgroup_min vectors into groups of at most kern_group_vecs
 * (one register pass per tile, activations in L1), sized by kern_group_mode (0: equal sizes,
 * 1: 7s with the remainder in 5s / 4s). kern_act_group = 0: never split. kern_group_fused 1:
 * all groups of a quantized call in one job after a parallel activation quantization; 0: one
 * job per group. Defaults 16, 7, 7, 1, 1 (KERNELS.md v13, Part 3, Part 4). */
extern int kern_act_group;
extern int kern_group_vecs;
extern int kern_qgroup_min;
extern int kern_group_mode;
extern int kern_group_fused;

/* 1 (default): pf_matmul picks the thread count per call (per-dtype cap for n == 1, fewer
 * threads for small matrices). 0: always use every pool thread (kbench thread sweeps). */
extern int kern_auto_threads;

void kern_f32(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1);
void kern_bf16(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1);
void kern_q8(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1);
void kern_q4(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1);
void kern_q4k(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n, int r0, int r1);

/* Quantize n vectors X [n][cols] (cols % 32 == 0) into the block-major buffers q (n * cols
 * bytes), d and fsum (n * cols / 32 floats each). */
void kern_quant_act(const float *X, int n, int cols, int8_t *q, float *d, float *fsum);

/* Opt-in matmul profile (kern_stats.c): PF_KERN_STATS=1 in the environment. */
int kern_stats_enabled(void);
void kern_stats_add(int rows, int cols, unsigned dtype, int n, double sec, size_t bytes);

/* Dispatch an empty job to every thread of the pool and wait (dispatch-overhead test). */
void kern_pool_noop(pf_pool *pool);

/* Generic parallel job: fn(arg, tid, nthr) runs on nthr = min(pool threads, max_threads)
 * threads (tid 0 = the caller); returns when all are done. */
typedef void (*kern_task)(void *arg, int tid, int nthr);
void kern_pool_run(pf_pool *pool, kern_task fn, void *arg, int max_threads);

/* Gated DeltaNet recurrence of one layer over n tokens (kern_lin.c), the NEON + threaded
 * form of matvec_ref.c's delta_rule, bit-identical to it (same per-element operations and
 * rounding: no fused multiply-add), with the pf_lin_rule contract of ops.h (logged pass,
 * deferred commit). Heads run in parallel on the pool. */
typedef pf_lin_args kern_lin_args; /* defined in ops.h */
void kern_lin_rule(pf_pool *pool, const kern_lin_args *a);

/* Commit replay: S[h] = S[h] * decay + k x delta for the first `keep` logged tokens, same
 * arithmetic as pf_lin_commit (threaded over heads). */
void kern_lin_apply(pf_pool *pool, float *S, const float *log, int keep, int heads, int dk,
                    int dv);

/* Value j of a packed 6-bit array (Q4K sub-scales / sub-mins). */
int kern_get6(const uint8_t *p, int j);

/* Dequantize one row of a quantized matrix (pf_get_row for dtypes 2+). */
void kern_get_row_q(const pf_mat *W, int row, float *dst);

#endif
