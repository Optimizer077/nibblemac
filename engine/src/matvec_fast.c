/* Fast implementation of ops.h: pinned thread pool + NEON kernels (kern_*.c).
 *
 * Pool: the calling thread is worker 0; workers 1..n-1 are pinned to the 2nd..n-th CPU the
 * caller may run on (CPUs 1..n-1 normally). The caller itself is not pinned: every thread
 * the application creates later inherits its affinity mask, so pinning it to CPU 0 (before
 * 0.2.0) confined e.g. a Python app's later threads to one core. While the workers spin on
 * the other CPUs the scheduler keeps the caller on the free one (engine/HARDENING.md H4).
 * A call publishes a job by bumping a generation counter; workers spin on it
 * (a decode step issues ~150 matmuls back to back, so a wake-up must cost well under a
 * microsecond) and fall back to a futex sleep after ~SPIN_ITERS idle polls, so an idle
 * engine does not burn 3 cores. Completion is a shared counter the caller spins on.
 *
 * Work split: contiguous slabs of 4-row tiles, one per thread, so each thread reads one
 * sequential stream (the DRAM-friendly pattern, see KERNELS.md).
 */
#define _GNU_SOURCE
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "kern.h"
#include "ops.h"
#include "pq_format.h"
#include "pq_quant.h"

#define MAX_THREADS 8
#define SPIN_SEC 0.05 /* idle spin before a worker sleeps (see worker_main) */
/* Below this many weight bytes per extra thread, waking it costs more than it saves. */
#define MIN_BYTES_PER_THREAD (32 * 1024)

typedef struct {
    kern_fn fn;
    const pf_mat *W;
    const float *X;
    const pf_act *A;
    float *Y;
    int n, units, nthr;
    kern_task gfn; /* generic job (kern_pool_run); fn is NULL then */
    void *garg;
} job_t;

typedef struct {
    pf_pool *pool;
    int tid;
    int cpu; /* pinned to this CPU; -1: not pinned */
} worker_arg;

struct pf_pool {
    int n;
    pthread_t th[MAX_THREADS];
    worker_arg args[MAX_THREADS];
    job_t job;
    alignas(64) _Atomic uint32_t gen; /* futex word; workers poll this line */
    _Atomic int sleepers;
    _Atomic int quit;
    alignas(64) _Atomic int done; /* written by workers: separate line */
    char pad[60];
};

static inline void cpu_pause(void) { __asm__ volatile("yield" ::: "memory"); }

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void pin_to(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

static void run_job(const job_t *j, int tid) {
    if (tid >= j->nthr) return;
    if (j->gfn) {
        j->gfn(j->garg, tid, j->nthr);
        return;
    }
    const int u0 = (int)((long)j->units * tid / j->nthr);
    const int u1 = (int)((long)j->units * (tid + 1) / j->nthr);
    int r1 = u1 * 4;
    if (r1 > j->W->rows) r1 = j->W->rows;
    if (u0 * 4 < r1) j->fn(j->W, j->X, j->A, j->Y, j->n, u0 * 4, r1);
}

static void *worker_main(void *p) {
    worker_arg *a = p;
    pf_pool *pool = a->pool;
    pin_to(a->cpu);
    uint32_t seen = 0;
    for (;;) {
        uint32_t g = atomic_load_explicit(&pool->gen, memory_order_acquire);
        if (g == seen) {
            /* Spin for up to SPIN_SEC before sleeping: inside a decode step the gaps between
             * matmuls (scalar attention / recurrence on the caller's core) are 0.1-2 ms, and
             * a futex wake-up after a sleep costs ~0.3 ms per call (measured: out_proj ran
             * at 2.0 instead of 3.9 GB/s with a 65k-iteration spin). The clock is read every
             * 1024 polls (~10 us). */
            const double t0 = now_sec();
            for (long i = 0; g == seen; i++) {
                cpu_pause();
                g = atomic_load_explicit(&pool->gen, memory_order_acquire);
                if ((i & 1023) == 1023 && now_sec() - t0 > SPIN_SEC) break;
            }
        }
        if (g == seen) { /* idle for a while: sleep until the next job */
            atomic_fetch_add(&pool->sleepers, 1);
            while ((g = atomic_load(&pool->gen)) == seen)
                syscall(SYS_futex, &pool->gen, FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
            atomic_fetch_sub(&pool->sleepers, 1);
        }
        seen = g;
        if (atomic_load_explicit(&pool->quit, memory_order_acquire)) break;
        run_job(&pool->job, a->tid);
        atomic_fetch_add_explicit(&pool->done, 1, memory_order_release);
    }
    return NULL;
}

pf_pool *pf_pool_create(int n_threads) {
    /* Worker t is pinned to the t-th CPU of the caller's mask (CPU t when the mask holds
     * every CPU); the caller's own mask is left as it is. */
    cpu_set_t allowed;
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    const int have = pthread_getaffinity_np(pthread_self(), sizeof allowed, &allowed) == 0;
    if (have && CPU_COUNT(&allowed) > 0) ncpu = CPU_COUNT(&allowed);
    /* more threads than cores would make spinning workers share a core */
    if (n_threads <= 0 || n_threads > ncpu) n_threads = (int)ncpu;
    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
    int cpus[MAX_THREADS];
    for (int t = 0, c = 0; t < n_threads; t++) {
        while (have && c < CPU_SETSIZE && !CPU_ISSET(c, &allowed)) c++;
        cpus[t] = !have ? t : c < CPU_SETSIZE ? c++ : -1;
    }
    pf_pool *pool = aligned_alloc(64, (sizeof(pf_pool) + 63) & ~(size_t)63);
    if (!pool) return NULL;
    memset(pool, 0, sizeof *pool);
    pool->n = n_threads;
    for (int t = 1; t < n_threads; t++) {
        pool->args[t] = (worker_arg){pool, t, cpus[t]};
        if (pthread_create(&pool->th[t], NULL, worker_main, &pool->args[t]) != 0) {
            pool->n = t; /* run with what we have */
            break;
        }
    }
    return pool;
}

void pf_pool_destroy(pf_pool *pool) {
    if (!pool) return;
    atomic_store(&pool->quit, 1);
    atomic_fetch_add(&pool->gen, 1);
    syscall(SYS_futex, &pool->gen, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
    for (int t = 1; t < pool->n; t++) pthread_join(pool->th[t], NULL);
    free(pool);
}

int pf_pool_threads(const pf_pool *pool) { return pool ? pool->n : 1; }

static void dispatch(pf_pool *pool, const job_t *job) {
    if (!pool || pool->n == 1 || job->nthr <= 1) {
        run_job(job, 0);
        return;
    }
    pool->job = *job;
    atomic_store_explicit(&pool->done, 0, memory_order_relaxed);
    atomic_fetch_add(&pool->gen, 1); /* seq_cst: publishes job, pairs with sleepers check */
    if (atomic_load(&pool->sleepers) > 0)
        syscall(SYS_futex, &pool->gen, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
    run_job(&pool->job, 0);
    while (atomic_load_explicit(&pool->done, memory_order_acquire) < pool->n - 1) cpu_pause();
}

static void noop_kernel(const pf_mat *W, const float *X, const pf_act *A, float *Y, int n,
                        int r0, int r1) {
    (void)W, (void)X, (void)A, (void)Y, (void)n, (void)r0, (void)r1;
}

void kern_pool_noop(pf_pool *pool) {
    static const pf_mat dummy = {0, 4 * MAX_THREADS, 0, NULL};
    const job_t job = {noop_kernel, &dummy, NULL, NULL, NULL, 0, MAX_THREADS, pf_pool_threads(pool),
                       NULL, NULL};
    dispatch(pool, &job);
}

void kern_pool_run(pf_pool *pool, kern_task fn, void *arg, int max_threads) {
    int t = pf_pool_threads(pool);
    if (max_threads > 0 && t > max_threads) t = max_threads;
    const job_t job = {NULL, NULL, NULL, NULL, NULL, 0, 0, t, fn, arg};
    dispatch(pool, &job);
}

static kern_fn pick(uint32_t dtype) {
    switch (dtype) {
    case PQ_F32: return kern_f32;
    case PQ_BF16: return kern_bf16;
    case PQ_Q8_G32: return kern_q8;
    case PQ_Q4_G32: return kern_q4;
    case PQ_Q4K_G32: return kern_q4k;
    default: return NULL;
    }
}

int pf_dtype_supported(uint32_t dtype) { return pick(dtype) != NULL; }

static size_t weight_bytes(const pf_mat *W) {
    if (W->dtype == PQ_F32) return (size_t)W->rows * W->cols * 4;
    if (W->dtype == PQ_BF16) return (size_t)W->rows * W->cols * 2;
    return pq_quant_nbytes(W->dtype, W->rows, W->cols);
}

/* Threads used by one call. Decode (n == 1) is DRAM-bound and one A72 core already
 * saturates DRAM for cheap-to-decode dtypes, while extra streams cost a few % of bandwidth
 * (KERNELS.md), so n == 1 uses a per-dtype cap; batched calls are compute-bound and use
 * every thread. Small matrices use fewer threads so each moves >= MIN_BYTES_PER_THREAD. */
int kern_auto_threads = 1;
static const int N1_CAP[] = {
    [PQ_F32] = 2, [PQ_BF16] = 2, [PQ_Q8_G32] = 1, [PQ_Q4_G32] = 3, [PQ_Q4K_G32] = 3,
};

static int threads_for(const pf_mat *W, int n, const pf_pool *pool) {
    int t = pf_pool_threads(pool);
    if (!kern_auto_threads) return t;
    if (n == 1 && W->dtype < sizeof N1_CAP / sizeof *N1_CAP && t > N1_CAP[W->dtype])
        t = N1_CAP[W->dtype];
    const size_t cap = weight_bytes(W) * (size_t)n / MIN_BYTES_PER_THREAD;
    if ((size_t)t > cap) t = cap > 0 ? (int)cap : 1;
    const int units = (W->rows + 3) / 4;
    return t < units ? t : units;
}

/* Per-calling-thread scratch for quantized activations, grown on demand. */
static _Thread_local struct {
    void *buf;
    size_t cap;
} scratch;

static void *scratch_get(size_t bytes) {
    if (bytes > scratch.cap) {
        free(scratch.buf);
        scratch.buf = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
        scratch.cap = scratch.buf ? bytes : 0;
        if (!scratch.buf) {
            fprintf(stderr, "pinfer: out of memory for activation scratch\n");
            abort();
        }
    }
    return scratch.buf;
}

/* Groups (KERNELS.md v13, Part 3, Part 4). A batch of n vectors runs in groups, each group
 * streaming the weights again, so that a group's int8 activations stay in L1 across the tiles
 * (64 vectors x 1 KB re-read from L2 for every 4-row tile ran at ~35 instead of ~27 cycles per
 * block-vector, v13). Quantized dtypes: calls of more than kern_qgroup_min vectors run in
 * groups of at most kern_group_vecs (7), i.e. one register pass per tile. With
 * kern_group_mode 1 the groups are 7s and the remainder goes into groups of 5 or 4 (a 6-vector
 * pass costs ~8-10 % more per vector than a 7 or 5 on the A72, 2-3 vectors are memory-bound,
 * Part 4); mode 0 = equal sizes (+-1) as in Part 3. Float dtypes keep equal groups of
 * kern_act_group. Per-vector arithmetic is unchanged, so results stay bit-identical to
 * pf_matvec whatever the grouping. */
#define MAX_GROUPS 96

int kern_qgroup_min = 7;
int kern_group_mode = 1;
int kern_group_fused = 1;

/* Group starts st[0] = 0 < st[1] < ... < st[ng] = n of a call; returns ng. */
static int plan_groups(int n, int gmax, int mode, int *st) {
    int ng = 0;
    st[0] = 0;
    if (mode == 1 && gmax == 7 && n > 7 && n / 7 + 3 <= MAX_GROUPS) {
        /* remainder r of n / 7 -> its groups, taking borrow[r] 7s with it */
        static const int tail[7][4] = {{0},    {5, 5, 5, 0}, {5, 4, 0}, {5, 5, 0},
                                       {4, 0}, {5, 0},       {6, 0}};
        static const int borrow[7] = {0, 2, 1, 1, 0, 0, 0};
        const int r = n % 7, a = n / 7 - borrow[r];
        if (a >= 0) {
            for (int i = 0; i < a; i++, ng++) st[ng + 1] = st[ng] + 7;
            for (int i = 0; tail[r][i]; i++, ng++) st[ng + 1] = st[ng] + tail[r][i];
            return ng;
        }
    }
    ng = (n + gmax - 1) / gmax;
    if (ng > MAX_GROUPS) ng = MAX_GROUPS;
    for (int g = 0; g < ng; g++) st[g + 1] = st[g] + n / ng + (g < n % ng);
    return ng;
}

/* kern_group_fused: all groups of a quantized call in two jobs. The activations of every group
 * are quantized first, in parallel (group g's block-major arrays at offsets st[g] * cols and
 * st[g] * cols / 32 of one buffer), then each thread runs every group over its own slab of
 * tiles: two barriers per call instead of one per group, and no serial quantization between
 * the groups. Same kernel on the same rows and activations: bit-identical. */
typedef struct {
    kern_fn fn;
    const pf_mat *W;
    const float *X;
    float *Y;
    int ng, units;
    int8_t *q;
    float *d, *fs;
    int st[MAX_GROUPS + 1];
} group_job;

static void quant_task(void *arg, int tid, int nthr) {
    const group_job *j = arg;
    const int cols = j->W->cols, nb = cols / KERN_ACT_BLOCK;
    for (int g = tid; g < j->ng; g += nthr) {
        const int g0 = j->st[g], gn = j->st[g + 1] - g0;
        kern_quant_act(j->X + (size_t)g0 * cols, gn, cols, j->q + (size_t)g0 * cols,
                       j->d + (size_t)g0 * nb, j->fs + (size_t)g0 * nb);
    }
}

static void groups_task(void *arg, int tid, int nthr) {
    const group_job *j = arg;
    const int u0 = (int)((long)j->units * tid / nthr);
    const int u1 = (int)((long)j->units * (tid + 1) / nthr);
    int r1 = u1 * 4;
    if (r1 > j->W->rows) r1 = j->W->rows;
    if (u0 * 4 >= r1) return;
    const int cols = j->W->cols, nb = cols / KERN_ACT_BLOCK;
    for (int g = 0; g < j->ng; g++) {
        const int g0 = j->st[g], gn = j->st[g + 1] - g0;
        const pf_act a = {cols, gn, j->q + (size_t)g0 * cols, j->d + (size_t)g0 * nb,
                          j->fs + (size_t)g0 * nb};
        j->fn(j->W, NULL, &a, j->Y + (size_t)g0 * j->W->rows, gn, u0 * 4, r1);
    }
}

void pf_matmul(pf_pool *pool, const pf_mat *W, const float *X, float *Y, int n) {
    const kern_fn fn = pick(W->dtype);
    if (!fn) {
        fprintf(stderr, "pinfer: matmul: unsupported dtype %u\n", W->dtype);
        abort();
    }
    if (n <= 0) return;
    const double t0 = kern_stats_enabled() ? now_sec() : 0.0;
    const int quant = W->dtype >= PQ_Q8_G32, cols = W->cols, nb = cols / KERN_ACT_BLOCK;
    group_job gj; /* only st[0..ng] are set (no zeroing of the whole array per call) */
    gj.fn = fn, gj.W = W, gj.X = X, gj.Y = Y, gj.ng = 1, gj.units = (W->rows + 3) / 4;
    gj.st[0] = 0, gj.st[1] = n;
    if (kern_act_group > 0 && quant && kern_group_vecs > 0 && n > kern_qgroup_min)
        gj.ng = plan_groups(n, kern_group_vecs, kern_group_mode, gj.st);
    else if (kern_act_group > 0 && !quant && n > kern_act_group)
        gj.ng = plan_groups(n, kern_act_group, 0, gj.st);
    if (quant && kern_group_fused && gj.ng > 1) {
        const size_t qb = ((size_t)n * cols + 63) & ~(size_t)63;
        const size_t fb = ((size_t)n * nb * 4 + 63) & ~(size_t)63;
        uint8_t *s = scratch_get(qb + 2 * fb);
        gj.q = (int8_t *)s, gj.d = (float *)(s + qb), gj.fs = (float *)(s + qb + fb);
        kern_pool_run(pool, quant_task, &gj, gj.ng);
        kern_pool_run(pool, groups_task, &gj, threads_for(W, gj.st[1], pool));
    } else {
        for (int g = 0; g < gj.ng; g++) {
            const int g0 = gj.st[g], gn = gj.st[g + 1] - g0;
            const float *Xg = X + (size_t)g0 * cols;
            float *Yg = Y + (size_t)g0 * W->rows;
            pf_act act, *A = NULL;
            if (quant) {
                const size_t qb = ((size_t)gn * cols + 63) & ~(size_t)63, fb = (size_t)gn * nb * 4;
                uint8_t *s = scratch_get(qb + 2 * fb);
                act = (pf_act){cols, gn, (int8_t *)s, (float *)(s + qb), (float *)(s + qb + fb)};
                kern_quant_act(Xg, gn, cols, (int8_t *)act.q, (float *)act.d, (float *)act.fsum);
                A = &act;
            }
            job_t job = {fn, W, Xg, A, Yg, gn, gj.units, threads_for(W, gn, pool), NULL, NULL};
            dispatch(pool, &job);
        }
    }
    if (kern_stats_enabled())
        kern_stats_add(W->rows, W->cols, W->dtype, n, now_sec() - t0, weight_bytes(W));
}

void pf_pool_run(pf_pool *pool, pf_task fn, void *arg, int max_threads) {
    kern_pool_run(pool, fn, arg, max_threads);
}

void pf_lin_rule(pf_pool *pool, const pf_lin_args *a) { kern_lin_rule(pool, a); }

void pf_lin_apply(pf_pool *pool, float *S, const float *log, int keep, int heads, int dk,
                  int dv) {
    kern_lin_apply(pool, S, log, keep, heads, dk, dv);
}

void pf_matvec(pf_pool *pool, const pf_mat *W, const float *x, float *y) {
    pf_matmul(pool, W, x, y, 1);
}

void pf_get_row(const pf_mat *W, int row, float *dst) {
    if (W->dtype == PQ_F32) {
        memcpy(dst, (const float *)W->data + (size_t)row * W->cols, (size_t)W->cols * 4);
    } else if (W->dtype == PQ_BF16) {
        const uint16_t *s = (const uint16_t *)W->data + (size_t)row * W->cols;
        for (int c = 0; c < W->cols; c++) {
            union { uint32_t u; float f; } v = {(uint32_t)s[c] << 16};
            dst[c] = v.f;
        }
    } else {
        kern_get_row_q(W, row, dst);
    }
}
