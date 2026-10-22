/* abench: the attention core of forward_attn.c (attend: scores, softmax, P.V + gate) on
 * synthetic data, for A/B tests of the attention kernels without a model:
 *   - time per layer and phase with the K/V caches of --layers layers rotated, so every pass
 *     streams the cache from DRAM as in a decode step (6 layers x 16 MB at 4096 tokens, fp32);
 *     GB/s = cache bytes the phase reads / its time;
 *   - a 64-bit FNV-1a hash of the outputs (o of every token and head after the gate, and the
 *     softmax probabilities), so two builds can be compared bit for bit: the same seed gives
 *     the same inputs. --dump f also writes them raw.
 *   - --stream: a plain read of the same K and V rows with the scores pass's thread split
 *     (the memband loop), i.e. the DRAM ceiling of this layout; --stream-cols: the same with
 *     the P.V pass's split (each thread reads its column half of every row of its group);
 *   - --phases (n <= PF_ATTN_TB): the three passes timed one by one, with the busiest and
 *     the least busy thread's time (wall - busiest = dispatch + waiting).
 * forward_attn.c is #included to reach its static functions; pf_stats_* are defined here and
 * collect the per-phase times. Build (WSL, project root): make -C engine/bench abench
 * (build/aarch64/abench) or abench-host (build/host/abench, scalar mirrors).
 *   abench [--ctx 4096] [--n 1] [--kv 0|1|2] [--threads 4] [--layers 6] [--reps 8]
 *          [--heads 8] [--kvh 2] [--hd 256] [--seed 1] [--dump f]
 *          [--stream | --stream-cols | --stream-alt | --phases] [--thrash [-]MB] [--fill-tok]
 * One RESULT line per run for harness/pi.py bench. */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/forward_attn.c"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* ---- phase timers (replace forward_stats.c) ------------------------------------------- */
int pf_stats_on = 1;
static double ph_sec[PFS_N];
static long ph_calls[PFS_N];

double pf_stats_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void pf_stats_add(int phase, int kind, double t0) {
    (void)kind;
    ph_sec[phase] += pf_stats_now() - t0;
    ph_calls[phase]++;
}

/* ---- inputs --------------------------------------------------------------------------- */
static unsigned long long rng = 1;
static double unif(void) {  /* (0, 1) */
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(rng >> 11) + 0.5) / 9007199254740992.0;
}
static float gauss(void) { return (float)(sqrt(-2.0 * log(unif())) * cos(6.283185307179586 * unif())); }

static void *alloc(size_t bytes) {
    void *p = aligned_alloc(64, (bytes + 63) / 64 * 64);
    if (!p) { fprintf(stderr, "abench: out of memory (%zu bytes)\n", bytes); exit(1); }
    memset(p, 0, bytes);
    return p;
}

static unsigned long long fnv(unsigned long long h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

/* ---- --stream / --stream-cols --------------------------------------------------------- */
typedef struct {
    const pinfer_ctx *ctx;
    const void *cache;
    int np, cols;
    unsigned long long sink[16];
} stream_job;

static unsigned long long read_xor(const unsigned char *b, size_t bytes) {
#if defined(__ARM_NEON)
    uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 64 <= bytes; i += 64) {
        a0 = veorq_u64(a0, vld1q_u64((const uint64_t *)(b + i)));
        a1 = veorq_u64(a1, vld1q_u64((const uint64_t *)(b + i + 16)));
        a2 = veorq_u64(a2, vld1q_u64((const uint64_t *)(b + i + 32)));
        a3 = veorq_u64(a3, vld1q_u64((const uint64_t *)(b + i + 48)));
    }
    const uint64x2_t a = veorq_u64(veorq_u64(a0, a1), veorq_u64(a2, a3));
    return vgetq_lane_u64(a, 0) ^ vgetq_lane_u64(a, 1);
#else
    unsigned long long acc = 0;
    for (size_t i = 0; i + 8 <= bytes; i += 8) acc ^= *(const unsigned long long *)(b + i);
    return acc;
#endif
}

/* units (group, half): rows [0, np/2) / [np/2, np), or with cols == 1 the column half of every
 * row, cols == 2 alternate blocks of SBLK rows (the two threads of a group side by side) */
#define SBLK 16
static void stream_task(void *arg, int tid, int nthr) {
    stream_job *j = arg;
    const int units = 2 * j->ctx->n_kv, half = (j->np + 1) / 2;
    const size_t rb = pf_kv_row_bytes(j->ctx->kv_dtype, j->ctx->hd);
    unsigned long long acc = 0;
    for (int u = units * tid / nthr; u < units * (tid + 1) / nthr; u++) {
        const int g = u >> 1;
        if (j->cols == 2) {
            const unsigned char *b = pf_kv_row(j->ctx, j->cache, g, 0);
            for (int p = (u & 1) * SBLK; p < j->np; p += 2 * SBLK)
                acc ^= read_xor(b + (size_t)p * rb, (size_t)(p + SBLK < j->np ? SBLK : j->np - p) * rb);
        } else if (j->cols) {
            const unsigned char *b = pf_kv_row(j->ctx, j->cache, g, 0);
            for (int p = 0; p < j->np; p++)
                acc ^= read_xor(b + (size_t)p * rb + (size_t)(u & 1) * (rb / 2), rb / 2);
        } else {
            const int p0 = (u & 1) * half, p1 = (u & 1) ? j->np : half;
            acc ^= read_xor(pf_kv_row(j->ctx, j->cache, g, p0), (size_t)(p1 - p0) * rb);
        }
    }
    j->sink[tid & 15] ^= acc;
}

/* ---- --thrash ------------------------------------------------------------------------ */
typedef struct {
    const unsigned char *buf;
    size_t bytes;
    unsigned long long sink[16];
} thrash_job;

static int thrash_write;   /* --thrash -MB: write the buffer (dirty lines to evict) */
static void thrash_task(void *arg, int tid, int nthr) {
    thrash_job *j = arg;
    const size_t q = j->bytes / (size_t)nthr / 64 * 64;
    if (thrash_write) memset((unsigned char *)j->buf + (size_t)tid * q, tid + j->sink[0] % 7, q);
    else j->sink[tid & 15] ^= read_xor(j->buf + (size_t)tid * q, q);
}

/* ---- --phases: one pool run timed, with each thread's busy time ----------------------- */
typedef struct {
    pf_task fn;
    void *arg;
    double busy[16];
} shim_job;

static void shim_task(void *arg, int tid, int nthr) {
    shim_job *s = arg;
    const double t0 = pf_stats_now();
    s->fn(s->arg, tid, nthr);
    s->busy[tid & 15] += pf_stats_now() - t0;
}

static double run_timed(pf_pool *pool, pf_task fn, void *arg, int max_threads, double *bmax,
                        double *bmin) {
    shim_job s = {fn, arg, {0}};
    const double t0 = pf_stats_now();
    pf_pool_run(pool, shim_task, &s, max_threads);
    const double wall = pf_stats_now() - t0;
    const int nt = pf_pool_threads(pool) < max_threads ? pf_pool_threads(pool) : max_threads;
    double hi = 0, lo = 1e9;
    for (int t = 0; t < nt && t < 16; t++) {
        hi = s.busy[t] > hi ? s.busy[t] : hi;
        lo = s.busy[t] < lo ? s.busy[t] : lo;
    }
    *bmax += hi;
    *bmin += lo;
    return wall;
}

int main(int argc, char **argv) {
    int nctx = 4096, n = 1, kv = 0, threads = 4, layers = 6, reps = 8, heads = 8, kvh = 2;
    int hd = 256, stream = 0, phases = 0, thrash = 0, fill_tok = 0;
    const char *dump = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--stream")) { stream = 1; continue; }
        if (!strcmp(a, "--stream-cols")) { stream = 2; continue; }
        if (!strcmp(a, "--stream-alt")) { stream = 3; continue; }
        if (!strcmp(a, "--phases")) { phases = 1; continue; }
        if (!strcmp(a, "--fill-tok")) { fill_tok = 1; continue; }
        if (!strcmp(a, "--fill-chunk")) { fill_tok = i + 1 < argc ? atoi(argv[++i]) : 1; continue; }
        if (!v) { fprintf(stderr, "abench: %s needs a value\n", a); return 2; }
        if (!strcmp(a, "--ctx")) nctx = atoi(v);
        else if (!strcmp(a, "--n")) n = atoi(v);
        else if (!strcmp(a, "--kv")) kv = atoi(v);
        else if (!strcmp(a, "--threads")) threads = atoi(v);
        else if (!strcmp(a, "--layers")) layers = atoi(v);
        else if (!strcmp(a, "--reps")) reps = atoi(v);
        else if (!strcmp(a, "--heads")) heads = atoi(v);
        else if (!strcmp(a, "--kvh")) kvh = atoi(v);
        else if (!strcmp(a, "--hd")) hd = atoi(v);
        else if (!strcmp(a, "--seed")) rng = strtoull(v, NULL, 10);
        else if (!strcmp(a, "--dump")) dump = v;
        else if (!strcmp(a, "--thrash")) {
            thrash = atoi(v);
            if (thrash < 0) thrash = -thrash, thrash_write = 1;
        }
        else { fprintf(stderr, "abench: unknown option %s\n", a); return 2; }
        i++;
    }
    if (n < 1 || n > PF_MAX_BATCH || n > nctx || hd > PF_KV_MAX_HD || hd % 4 || heads % kvh ||
        kv < 0 || kv > 2 || layers < 1 || reps < 1 || (phases && n > PF_ATTN_TB)) {
        fprintf(stderr, "abench: bad arguments\n");
        return 2;
    }

    pinfer_ctx c;
    memset(&c, 0, sizeof c);
    c.pool = pf_pool_create(threads);
    c.max_ctx = nctx + 64;
    c.kv_dtype = kv;
    c.n_heads = heads; c.n_kv = kvh; c.hd = hd; c.rot = hd / 4;
    c.ta = alloc((size_t)PF_MAX_BATCH * heads * 2 * hd * sizeof(float));
    c.to = alloc((size_t)PF_MAX_BATCH * heads * hd * sizeof(float));
    c.att = alloc((size_t)PF_ATTN_TB * heads * c.max_ctx * sizeof(float));

    float *row = alloc((size_t)hd * sizeof(float));
    void **kc = alloc((size_t)layers * sizeof(void *)), **vc = alloc((size_t)layers * sizeof(void *));
    if (fill_tok) {   /* the engine's page-touch order: calloc, then token by token over
                         * every layer / group / K / V, as a prefill writes them */
        for (int l = 0; l < layers; l++) {
            kc[l] = calloc(pf_kv_bytes(&c), 1);
            vc[l] = calloc(pf_kv_bytes(&c), 1);
            if (!kc[l] || !vc[l]) { fprintf(stderr, "abench: out of memory\n"); return 1; }
        }
        for (int p = 0; p < nctx; p++)
            for (int l = 0; l < layers; l++)
                for (int g = 0; g < kvh; g++) {
                    if (fill_tok > 1 && p % fill_tok == 0) {   /* pre-touch rows ahead */
                        const int r = p + fill_tok < c.max_ctx ? fill_tok : c.max_ctx - p;
                        const size_t rb = pf_kv_row_bytes(kv, hd);
                        memset((char *)kc[l] + ((size_t)g * c.max_ctx + p) * rb, 0, r * rb);
                        memset((char *)vc[l] + ((size_t)g * c.max_ctx + p) * rb, 0, r * rb);
                    }
                    for (int d = 0; d < hd; d++) row[d] = gauss();
                    pf_kv_store(&c, kc[l], g, p, row);
                    for (int d = 0; d < hd; d++) row[d] = gauss();
                    pf_kv_store(&c, vc[l], g, p, row);
                }
    } else
        for (int l = 0; l < layers; l++) {
            kc[l] = alloc(pf_kv_bytes(&c));
            vc[l] = alloc(pf_kv_bytes(&c));
            for (int g = 0; g < kvh; g++)
                for (int p = 0; p < nctx; p++) {
                    for (int d = 0; d < hd; d++) row[d] = gauss();
                    pf_kv_store(&c, kc[l], g, p, row);
                    for (int d = 0; d < hd; d++) row[d] = gauss();
                    pf_kv_store(&c, vc[l], g, p, row);
                }
        }
    for (size_t i = 0; i < (size_t)n * heads * 2 * hd; i++) c.ta[i] = gauss();

    /* --thrash MB: between layers, the pool streams MB of other memory (the weights of a
     * decode step's matmuls: caches and TLB as cold as in the engine) */
    thrash_job tj = {thrash ? alloc((size_t)thrash << 20) : NULL, (size_t)thrash << 20, {0}};
    const int pos0 = nctx - n;
    const size_t cache_rd = (size_t)kvh * nctx * pf_kv_row_bytes(kv, hd);
    FILE *df = dump ? fopen(dump, "wb") : NULL;
    if (dump && !df) { fprintf(stderr, "abench: cannot write %s\n", dump); return 1; }

    if (stream) {
        stream_job sj = {&c, NULL, nctx, stream - 1, {0}};
        double tk = 0, tv = 0;
        for (int r = 0; r < reps; r++)
            for (int l = 0; l < layers; l++) {
                double t0 = pf_stats_now();
                sj.cache = kc[l];
                pf_pool_run(c.pool, stream_task, &sj, 2 * kvh);
                double t1 = pf_stats_now();
                sj.cache = vc[l];
                pf_pool_run(c.pool, stream_task, &sj, 2 * kvh);
                tk += t1 - t0;
                tv += pf_stats_now() - t1;
            }
        const double calls = (double)reps * layers;
        const char *nm = stream == 3 ? "stream_alt" : stream == 2 ? "stream_cols" : "stream";
        printf("%s ctx=%d kv=%d threads=%d: K %.1f us (%.2f GB/s), V %.1f us (%.2f GB/s) "
               "[sink %llx]\n", nm, nctx, kv, threads, 1e6 * tk / calls,
               cache_rd * calls / tk / 1e9, 1e6 * tv / calls, cache_rd * calls / tv / 1e9,
               sj.sink[0]);
        printf("RESULT {\"bench\":\"abench_%s\",\"ctx\":%d,\"kv\":%d,\"threads\":%d,"
               "\"k_us\":%.1f,\"k_gbs\":%.3f,\"v_us\":%.1f,\"v_gbs\":%.3f}\n", nm, nctx, kv,
               threads, 1e6 * tk / calls, cache_rd * calls / tk / 1e9, 1e6 * tv / calls,
               cache_rd * calls / tv / 1e9);
        return 0;
    }

    if (phases) {   /* attend()'s loop for one block, pass by pass (ds as in attend) */
        const int nch = (hd + 15) / 16;
        int ds = (pf_pool_threads(c.pool) + kvh - 1) / kvh;
        ds = ds < 1 ? 1 : ds > nch ? nch : ds;
        double w[3] = {0}, hi[3] = {0}, lo[3] = {0};
        for (int r = 0; r <= reps; r++) {   /* r = 0: untimed warm-up */
            if (r == 1)
                for (int i = 0; i < 3; i++) w[i] = hi[i] = lo[i] = 0;
            for (int l = 0; l < layers; l++) {
                attn_job job = {&c, kc[l], vc[l], 0, n, pos0, ds};
                w[0] += run_timed(c.pool, score_task, &job, 2 * kvh, &hi[0], &lo[0]);
                w[1] += run_timed(c.pool, softmax_task, &job, n * heads, &hi[1], &lo[1]);
                w[2] += run_timed(c.pool, pv_task, &job, kvh * ds, &hi[2], &lo[2]);
                if (thrash) pf_pool_run(c.pool, thrash_task, &tj, 4);
            }
        }
        const double calls = (double)reps * layers;
        const char *nm[3] = {"scores", "softmax", "pv"};
        printf("phases ctx=%d n=%d kv=%d threads=%d, us per layer (wall / busiest / least busy):",
               nctx, n, kv, threads);
        for (int i = 0; i < 3; i++)
            printf(" %s %.1f / %.1f / %.1f%s", nm[i], 1e6 * w[i] / calls, 1e6 * hi[i] / calls,
                   1e6 * lo[i] / calls, i < 2 ? " |" : "\n");
        printf("RESULT {\"bench\":\"abench_phases\",\"ctx\":%d,\"n\":%d,\"kv\":%d,\"threads\":%d",
               nctx, n, kv, threads);
        for (int i = 0; i < 3; i++)
            printf(",\"%s_us\":%.1f,\"%s_busy_max_us\":%.1f,\"%s_busy_min_us\":%.1f", nm[i],
                   1e6 * w[i] / calls, nm[i], 1e6 * hi[i] / calls, nm[i], 1e6 * lo[i] / calls);
        printf("}\n");
        return 0;
    }

    /* untimed pass over every layer: outputs hashed (and dumped) */
    unsigned long long h = 1469598103934665603ULL;
    for (int l = 0; l < layers; l++) {
        attend(&c, kc[l], vc[l], n, pos0, pf_stats_now());
        h = fnv(h, c.to, (size_t)n * heads * hd * sizeof(float));
        if (df) fwrite(c.to, sizeof(float), (size_t)n * heads * hd, df);
        /* probabilities of the last block (att holds only the current block) */
        const int t0 = (n - 1) / PF_ATTN_TB * PF_ATTN_TB;
        for (int t = t0; t < n; t++)
            for (int i = 0; i < heads; i++) {
                const float *a = c.att + ((size_t)(t - t0) * heads + i) * c.max_ctx;
                h = fnv(h, a, (size_t)(pos0 + t + 1) * sizeof(float));
                if (df) fwrite(a, sizeof(float), (size_t)(pos0 + t + 1), df);
            }
    }
    if (df) fclose(df);
    memset(ph_sec, 0, sizeof ph_sec);
    memset(ph_calls, 0, sizeof ph_calls);
    double t0 = pf_stats_now();
    for (int r = 0; r < reps; r++)
        for (int l = 0; l < layers; l++) {
            attend(&c, kc[l], vc[l], n, pos0, pf_stats_now());
            if (thrash) pf_pool_run(c.pool, thrash_task, &tj, 4);
        }
    const double calls = (double)reps * layers, tot = pf_stats_now() - t0;
    const double ts = ph_sec[PFS_ATTN] / calls, tp = ph_sec[PFS_ATTN_PV] / calls;
    /* K and V are read once per block of PF_ATTN_TB tokens */
    const double rd = (double)cache_rd * ((n + PF_ATTN_TB - 1) / PF_ATTN_TB);
    printf("attend ctx=%d n=%d kv=%d threads=%d heads=%d/%d hd=%d: per layer scores %.1f us "
           "(%.2f GB/s), softmax+pv %.1f us (%.2f GB/s), total %.1f us | hash %016llx\n",
           nctx, n, kv, threads, heads, kvh, hd, 1e6 * ts, rd / ts / 1e9, 1e6 * tp,
           rd / tp / 1e9, 1e6 * tot / calls, h);
    printf("RESULT {\"bench\":\"abench\",\"ctx\":%d,\"n\":%d,\"kv\":%d,\"threads\":%d,"
           "\"heads\":%d,\"kvh\":%d,\"hd\":%d,\"layers\":%d,\"reps\":%d,\"scores_us\":%.1f,"
           "\"pv_us\":%.1f,\"total_us\":%.1f,\"scores_gbs\":%.3f,\"pv_gbs\":%.3f,"
           "\"hash\":\"%016llx\"}\n", nctx, n, kv, threads, heads, kvh, hd, layers, reps,
           1e6 * ts, 1e6 * tp, 1e6 * tot / calls, rd / ts / 1e9, rd / tp / 1e9, h);
    return 0;
}
