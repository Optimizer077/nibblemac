/* Fast implementation of ops.h: pinned thread pool + NEON kernels (kern_*.c).
 *
 * Pool: the calling thread is worker 0 (pinned to CPU 0); workers 1..n-1 are pinned to
 * CPUs 1..n-1. A call publishes a job by bumping a generation counter; workers spin on it
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
#include <unistd.h>

#include "kern.h"
#include "ops.h"
#include "pq_format.h"
#include "pq_quant.h"

#define MAX_THREADS 8
#define SPIN_ITERS (1 << 16)
/* Below this many weight bytes per extra thread, waking it costs more than it saves. */
#define MIN_BYTES_PER_THREAD (32 * 1024)

typedef struct {
    kern_fn fn;
    const pf_mat *W;
    const float *X;
    const pf_act *A;
    float *Y;
    int n, units, nthr;
} job_t;

typedef struct {
    pf_pool *pool;
    int tid;
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

static void pin_to(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

static void run_job(const job_t *j, int tid) {
    if (tid >= j->nthr) return;
    const int u0 = (int)((long)j->units * tid / j->nthr);
    const int u1 = (int)((long)j->units * (tid + 1) / j->nthr);
    int r1 = u1 * 4;
    if (r1 > j->W->rows) r1 = j->W->rows;
    if (u0 * 4 < r1) j->fn(j->W, j->X, j->A, j->Y, j->n, u0 * 4, r1);
}

static void *worker_main(void *p) {
    worker_arg *a = p;
    pf_pool *pool = a->pool;
    pin_to(a->tid);
    uint32_t seen = 0;
    for (;;) {
        uint32_t g = atomic_load_explicit(&pool->gen, memory_order_acquire);
        for (int i = 0; g == seen && i < SPIN_ITERS; i++) {
            cpu_pause();
            g = atomic_load_explicit(&pool->gen, memory_order_acquire);
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
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    /* more threads than cores would make spinning workers share a core */
    if (n_threads <= 0 || n_threads > ncpu) n_threads = (int)ncpu;
    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
    pf_pool *pool = aligned_alloc(64, (sizeof(pf_pool) + 63) & ~(size_t)63);
    if (!pool) return NULL;
    memset(pool, 0, sizeof *pool);
    pool->n = n_threads;
    if (n_threads > 1) pin_to(0);
    for (int t = 1; t < n_threads; t++) {
        pool->args[t] = (worker_arg){pool, t};
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
    const job_t job = {noop_kernel, &dummy, NULL, NULL, NULL, 0, MAX_THREADS, pf_pool_threads(pool)};
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

void pf_matmul(pf_pool *pool, const pf_mat *W, const float *X, float *Y, int n) {
    const kern_fn fn = pick(W->dtype);
    if (!fn) {
        fprintf(stderr, "pinfer: matmul: unsupported dtype %u\n", W->dtype);
        abort();
    }
    if (n <= 0) return;
    pf_act act, *A = NULL;
    if (W->dtype >= PQ_Q8_G32) {
        const int cols = W->cols, nb = cols / KERN_ACT_BLOCK;
        const size_t qb = ((size_t)n * cols + 63) & ~(size_t)63, fb = (size_t)n * nb * 4;
        uint8_t *s = scratch_get(qb + 2 * fb);
        act = (pf_act){cols, n, (int8_t *)s, (float *)(s + qb), (float *)(s + qb + fb)};
        kern_quant_act(X, n, cols, (int8_t *)act.q, (float *)act.d, (float *)act.fsum);
        A = &act;
    }
    job_t job = {fn, W, X, A, Y, n, (W->rows + 3) / 4, threads_for(W, n, pool)};
    dispatch(pool, &job);
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
