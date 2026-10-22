/* tbench variant table; "ref" = the shipping kern_q4 / kern_q4k (KERNELS.md Part 5). */
#include "pq_quant.h"
#include "qvar.h"

void qv_ref(const pf_mat *W, const void *aux, const pf_act *A, float *Y, int n, int r0, int r1) {
    (void)aux;
    if (W->dtype == PQ_Q4K_G32) kern_q4k(W, NULL, A, Y, n, r0, r1);
    else kern_q4(W, NULL, A, Y, n, r0, r1);
}

#define BOTH ((1u << PQ_Q4_G32) | (1u << PQ_Q4K_G32))
const qvar QVARS[] = {
    {"ref", BOTH, NULL, qv_ref, "kern_q4 / kern_q4k C loops (69b31a3, kern_q4_asm = 0)", 0},
    {"eng", BOTH, NULL, qv_ref, "kern_q4 / kern_q4k as integrated (kern_q4_asm = 1)", 1},
    {"lut", BOTH, qv_lut_prep, qv_lut, "T-MAC bit-serial TBL lookup, exact int8 tables", 0},
    {"isq", BOTH, NULL, qv_isq, "SMLAL kernel, C intrinsics in interleaved order (no sched)", 0},
    {"q4s", BOTH, NULL, qv_q4s, "asm passes <= 7, multiplies paired with F1 ops, SCVTF", 0},
    {"q4m", BOTH, NULL, qv_q4m, "asm passes <= 6, as q4s with magic-number int->float", 0},
};
const int NQVARS = sizeof QVARS / sizeof *QVARS;
