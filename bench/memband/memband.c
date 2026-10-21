/*
 * memband: sustained DRAM read bandwidth on the Pi 4 with 1..4 threads.
 *
 * LLM decode streams every weight once per token, so this number is the ceiling for
 * tokens/s: tok/s <= bandwidth / model_bytes. The buffer (256 MB) is far larger than the
 * 1 MB L2, so every read goes to DRAM.
 *
 * Prints human-readable lines plus one `RESULT {json}` line for harness/pi.py.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define BUF_BYTES (256u << 20)
#define REPS 5
#define MAX_THREADS 4

typedef struct {
    const uint8_t *base;
    size_t bytes;
    int cpu;
    pthread_barrier_t *start;
    uint64_t sink;
} Job;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Read every byte once; 4 independent accumulators keep several loads in flight. */
static uint64_t read_sum(const uint8_t *p, size_t bytes) {
#ifdef __ARM_NEON
    uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i < bytes; i += 64) {
        a0 = veorq_u64(a0, vld1q_u64((const uint64_t *)(p + i)));
        a1 = veorq_u64(a1, vld1q_u64((const uint64_t *)(p + i + 16)));
        a2 = veorq_u64(a2, vld1q_u64((const uint64_t *)(p + i + 32)));
        a3 = veorq_u64(a3, vld1q_u64((const uint64_t *)(p + i + 48)));
    }
    uint64x2_t a = veorq_u64(veorq_u64(a0, a1), veorq_u64(a2, a3));
    return vgetq_lane_u64(a, 0) ^ vgetq_lane_u64(a, 1);
#else
    const uint64_t *q = (const uint64_t *)p;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (size_t i = 0; i < bytes / 8; i += 4) {
        a0 ^= q[i]; a1 ^= q[i + 1]; a2 ^= q[i + 2]; a3 ^= q[i + 3];
    }
    return a0 ^ a1 ^ a2 ^ a3;
#endif
}

static void *worker(void *arg) {
    Job *job = arg;
    /* Pin explicitly: the scheduler never places unpinned threads on isolcpus cores. */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(job->cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    pthread_barrier_wait(job->start);
    job->sink = read_sum(job->base, job->bytes);
    pthread_barrier_wait(job->start);
    return NULL;
}

/* Best-of-REPS read bandwidth in GB/s using n threads on disjoint slices. */
static double measure(const uint8_t *buf, int n) {
    double best = 0;
    for (int r = 0; r < REPS; r++) {
        pthread_barrier_t start;
        pthread_barrier_init(&start, NULL, (unsigned)n + 1);
        pthread_t tid[MAX_THREADS];
        Job jobs[MAX_THREADS];
        size_t slice = (BUF_BYTES / (size_t)n) & ~(size_t)63; /* read loop steps 64 B */
        for (int t = 0; t < n; t++) {
            jobs[t] = (Job){buf + (size_t)t * slice, slice, t % MAX_THREADS, &start, 0};
            if (pthread_create(&tid[t], NULL, worker, &jobs[t]) != 0) {
                fprintf(stderr, "pthread_create failed\n");
                exit(1);
            }
        }
        pthread_barrier_wait(&start);
        double t0 = now_sec();
        pthread_barrier_wait(&start);
        double dt = now_sec() - t0;
        for (int t = 0; t < n; t++) pthread_join(tid[t], NULL);
        pthread_barrier_destroy(&start);
        double gbs = (double)slice * n / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    return best;
}

int main(void) {
    uint8_t *buf = aligned_alloc(4096, BUF_BYTES);
    if (!buf) {
        fprintf(stderr, "allocation of %u MB failed\n", BUF_BYTES >> 20);
        return 1;
    }
    memset(buf, 1, BUF_BYTES); /* touch every page so faults are not timed */

    const int counts[] = {1, 2, 4};
    double gbs[3];
    for (int i = 0; i < 3; i++) {
        gbs[i] = measure(buf, counts[i]);
        printf("threads=%d  read %.2f GB/s\n", counts[i], gbs[i]);
    }
    printf("RESULT {\"bench\":\"memband\",\"buf_mb\":%u,\"read_gbs\":{\"1\":%.3f,\"2\":%.3f,\"4\":%.3f}}\n",
           BUF_BYTES >> 20, gbs[0], gbs[1], gbs[2]);
    free(buf);
    return 0;
}
