/* ubench: measured issue throughput of the NEON instructions the kernels depend on, on
 * the Cortex-A72 (the public optimization guide is ambiguous about Q-form throughput).
 * Each test runs 16 independent copies of one instruction per loop iteration (dest v0..v15,
 * sources v16/v17), so latency is hidden and time / instruction = reciprocal throughput.
 * Cycles assume the core runs at 1.5 GHz while busy. One RESULT line with all numbers.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>

#define ITERS 10000000L
#define GHZ 1.5

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define R4(P, S, a, b, c, d) P "v" #a S P "v" #b S P "v" #c S P "v" #d S
#define R16(P, S) R4(P, S, 0, 1, 2, 3) R4(P, S, 4, 5, 6, 7) R4(P, S, 8, 9, 10, 11) \
    R4(P, S, 12, 13, 14, 15)

#define TEST(NAME, P, S)                                                                \
    static double NAME(void) {                                                          \
        long n = ITERS;                                                                 \
        double t0 = now();                                                              \
        __asm__ volatile("1:\n\t" R16(P, S) "subs %0, %0, 1\n\tb.ne 1b\n\t"            \
                         : "+r"(n)                                                      \
                         :                                                              \
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",  \
                           "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc"); \
        return (now() - t0) * GHZ * 1e9 / (16.0 * ITERS);                               \
    }

TEST(t_smull, "smull ", ".8h, v16.8b, v17.8b\n\t")
TEST(t_smlal, "smlal ", ".8h, v16.8b, v17.8b\n\t")
TEST(t_smlal2, "smlal2 ", ".8h, v16.16b, v17.16b\n\t")
TEST(t_smlal_s16, "smlal ", ".4s, v16.4h, v17.4h\n\t")
TEST(t_mul8h, "mul ", ".8h, v16.8h, v17.8h\n\t")
TEST(t_mla8h, "mla ", ".8h, v16.8h, v17.8h\n\t")
TEST(t_mla16b, "mla ", ".16b, v16.16b, v17.16b\n\t")
TEST(t_fmla4s, "fmla ", ".4s, v16.4s, v17.4s\n\t")
TEST(t_fmla2s, "fmla ", ".2s, v16.2s, v17.2s\n\t")
TEST(t_fmul4s, "fmul ", ".4s, v16.4s, v17.4s\n\t")
TEST(t_sadalp, "sadalp ", ".4s, v16.8h\n\t")
TEST(t_saddlp, "saddlp ", ".4s, v16.8h\n\t")
TEST(t_addp8h, "addp ", ".8h, v16.8h, v17.8h\n\t")
TEST(t_add4s, "add ", ".4s, v16.4s, v17.4s\n\t")
TEST(t_and16b, "and ", ".16b, v16.16b, v17.16b\n\t")
TEST(t_ushr16b, "ushr ", ".16b, v16.16b, 4\n\t")
TEST(t_sshll, "sshll ", ".8h, v16.8b, 0\n\t")
TEST(t_ushll16, "shll ", ".4s, v16.4h, 16\n\t")
TEST(t_scvtf, "scvtf ", ".4s, v16.4s\n\t")
TEST(t_fcvtl, "fcvtl ", ".4s, v16.4h\n\t")
TEST(t_tbl, "tbl ", ".16b, {v16.16b}, v17.16b\n\t")

/* Mixes: 8 SMULL (v0-v7) + 8 of another op (v8-v15). If the other op issues on the other
 * pipe the pair costs ~8 cycles per iteration; if it shares SMULL's pipe, ~16. Reported
 * as cycles per iteration / 16 to stay comparable with the single-op tests. */
#define S8(P, S) R4(P, S, 0, 1, 2, 3) R4(P, S, 4, 5, 6, 7)
#define O8(P, S) R4(P, S, 8, 9, 10, 11) R4(P, S, 12, 13, 14, 15)
#define MIX(NAME, P2, S2)                                                               \
    static double NAME(void) {                                                          \
        long n = ITERS;                                                                 \
        double t0 = now();                                                              \
        __asm__ volatile("1:\n\t" S8("smull ", ".8h, v16.8b, v17.8b\n\t") O8(P2, S2)     \
                         "subs %0, %0, 1\n\tb.ne 1b\n\t"                                 \
                         : "+r"(n)                                                      \
                         :                                                              \
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",  \
                           "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc"); \
        return (now() - t0) * GHZ * 1e9 / (16.0 * ITERS);                               \
    }
MIX(m_add, "add ", ".4s, v16.4s, v17.4s\n\t")
MIX(m_and, "and ", ".16b, v16.16b, v17.16b\n\t")
MIX(m_ushr, "ushr ", ".16b, v16.16b, 4\n\t")
MIX(m_addp, "addp ", ".8h, v16.8h, v17.8h\n\t")
MIX(m_saddlp, "saddlp ", ".4s, v16.8h\n\t")
MIX(m_fmla, "fmla ", ".4s, v16.4s, v17.4s\n\t")
MIX(m_scvtf, "scvtf ", ".4s, v16.4s\n\t")
MIX(m_ldr, "ld1 {", ".16b}, [sp]\n\t")

int main(void) {
    struct { const char *name; double (*fn)(void); } t[] = {
        {"smull", t_smull}, {"smlal", t_smlal}, {"smlal2", t_smlal2},
        {"smlal_4s", t_smlal_s16}, {"mul_8h", t_mul8h}, {"mla_8h", t_mla8h},
        {"mla_16b", t_mla16b}, {"fmla_4s", t_fmla4s}, {"fmla_2s", t_fmla2s},
        {"fmul_4s", t_fmul4s}, {"sadalp", t_sadalp}, {"saddlp", t_saddlp},
        {"addp_8h", t_addp8h}, {"add_4s", t_add4s}, {"and_16b", t_and16b},
        {"ushr_16b", t_ushr16b}, {"sshll_8h", t_sshll}, {"shll_4s", t_ushll16},
        {"scvtf_4s", t_scvtf}, {"fcvtl_4s", t_fcvtl}, {"tbl_16b", t_tbl},
        {"smull+add", m_add}, {"smull+and", m_and}, {"smull+ushr", m_ushr},
        {"smull+addp", m_addp}, {"smull+saddlp", m_saddlp}, {"smull+fmla", m_fmla},
        {"smull+scvtf", m_scvtf}, {"smull+ldr", m_ldr},
    };
    const int n = sizeof t / sizeof *t;
    double c[64];
    t[0].fn(); /* warm-up: let the ondemand governor raise the clock */
    for (int i = 0; i < n; i++) {
        c[i] = t[i].fn();
        printf("%-10s %.2f cycles/insn\n", t[i].name, c[i]);
    }
    printf("RESULT {\"bench\":\"ubench\",\"ghz_assumed\":%.2f,\"cycles_per_insn\":{", GHZ);
    for (int i = 0; i < n; i++) printf("%s\"%s\":%.3f", i ? "," : "", t[i].name, c[i]);
    printf("}}\n");
    return 0;
}
