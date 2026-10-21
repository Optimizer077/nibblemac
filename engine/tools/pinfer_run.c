/*
 * pinfer_run: feed a prompt of token ids, greedily generate N tokens, report speed.
 *
 *   pinfer_run -m model.pq [-t threads] [-n N] [-c max_ctx] [-b prefill_batch]
 *              (--ids 1,2,3 | --ids-file f) [--dump-logits out.bin] [--dump-hidden out.bin]
 *              [--bench]
 *
 * Token ids only: tokenization lives in the Python package. --dump-* write raw little-endian
 * fp32 for the PROMPT tokens: logits [T][vocab], hidden [n_layers+1][T][hidden] (same layout
 * as ref/golden). Decode speed counts the N-1 single-token evals after the first generated
 * token (that one comes from the prefill logits). --bench adds one `RESULT {json}` line for
 * harness/pi.py.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "../src/forward.h"

#ifndef PF_MATVEC_NAME
#define PF_MATVEC_NAME "ref"
#endif

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Parse every integer in s (any non-digit separator). Returns count, -1 on overflow. */
static int parse_ids(const char *s, int32_t *ids, int cap) {
    int n = 0;
    while (*s) {
        if (*s < '0' || *s > '9') { s++; continue; }
        char *end;
        long v = strtol(s, &end, 10);
        if (n == cap || v > 0x7fffffffL) return -1;
        ids[n++] = (int32_t)v;
        s = end;
    }
    return n;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = sz >= 0 ? malloc((size_t)sz + 1) : NULL;
    if (buf) buf[fread(buf, 1, (size_t)sz, f)] = 0;
    fclose(f);
    return buf;
}

static int write_f32(const char *path, const float *v, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    size_t w = fwrite(v, sizeof(float), n, f);
    fclose(f);
    if (w != n) { fprintf(stderr, "%s: short write\n", path); return -1; }
    return 0;
}

static const char *base_name(const char *p) {
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

static void usage(void) {
    fprintf(stderr, "usage: pinfer_run -m model.pq [-t threads] [-n N] [-c max_ctx] "
                    "[-b batch] (--ids 1,2,3 | --ids-file f)\n"
                    "                  [--dump-logits f] [--dump-hidden f] [--bench]\n");
}

int main(int argc, char **argv) {
    const char *model = NULL, *ids_arg = NULL, *ids_file = NULL;
    const char *dump_logits = NULL, *dump_hidden = NULL;
    int threads = 4, n_gen = 32, max_ctx = 0, batch = PF_MAX_BATCH, bench = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        int used = 1;
        if (!strcmp(a, "-m") && v) model = v;
        else if (!strcmp(a, "-t") && v) threads = atoi(v);
        else if (!strcmp(a, "-n") && v) n_gen = atoi(v);
        else if (!strcmp(a, "-c") && v) max_ctx = atoi(v);
        else if (!strcmp(a, "-b") && v) batch = atoi(v);
        else if (!strcmp(a, "--ids") && v) ids_arg = v;
        else if (!strcmp(a, "--ids-file") && v) ids_file = v;
        else if (!strcmp(a, "--dump-logits") && v) dump_logits = v;
        else if (!strcmp(a, "--dump-hidden") && v) dump_hidden = v;
        else if (!strcmp(a, "--bench")) { bench = 1; used = 0; }
        else { usage(); return 2; }
        i += used;
    }
    if (!model || (!ids_arg == !ids_file) || n_gen < 0) { usage(); return 2; }

    enum { MAX_PROMPT = 1 << 16 };
    int32_t *prompt = malloc(MAX_PROMPT * sizeof(int32_t));
    char *text = ids_file ? read_file(ids_file) : NULL;
    int n_prompt = prompt ? parse_ids(ids_file ? (text ? text : "") : ids_arg, prompt,
                                      MAX_PROMPT) : -1;
    free(text);
    if (n_prompt <= 0) { fprintf(stderr, "no prompt ids\n"); return 2; }

    pinfer_params p = pinfer_default_params();
    p.n_threads = threads;
    p.max_ctx = max_ctx > 0 ? max_ctx : n_prompt + n_gen;
    int err = 0;
    double t0 = now_sec();
    pinfer_ctx *ctx = pinfer_load(model, &p, &err);
    if (!ctx) { fprintf(stderr, "load failed: %d\n", err); return 1; }
    pf_set_batch(ctx, batch);
    double t_load = now_sec() - t0;

    const int V = pinfer_n_vocab(ctx), H = ctx->H, L = ctx->n_layers;
    float *logits = malloc((size_t)V * sizeof(float));
    float *all = dump_logits ? malloc((size_t)n_prompt * V * sizeof(float)) : NULL;
    float *hid = dump_hidden ? malloc((size_t)(L + 1) * n_prompt * H * sizeof(float)) : NULL;
    int32_t *gen = malloc(((size_t)n_gen + 1) * sizeof(int32_t));
    if (!logits || !gen || (dump_logits && !all) || (dump_hidden && !hid)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    t0 = now_sec();
    int rc = pf_forward(ctx, prompt, n_prompt, logits, all, hid);
    double t_prefill = now_sec() - t0;
    if (rc != PINFER_OK) { fprintf(stderr, "prefill failed: %d\n", rc); return 1; }
    if (all && write_f32(dump_logits, all, (size_t)n_prompt * V)) return 1;
    if (hid && write_f32(dump_hidden, hid, (size_t)(L + 1) * n_prompt * H)) return 1;

    t0 = now_sec();
    for (int i = 0; i < n_gen; i++) {
        if (i > 0 && (rc = pinfer_eval(ctx, &gen[i - 1], 1, logits)) != PINFER_OK) {
            fprintf(stderr, "decode step %d failed: %d\n", i, rc);
            return 1;
        }
        gen[i] = pinfer_argmax(logits, V);
    }
    double t_decode = now_sec() - t0;

    printf("generated:");
    for (int i = 0; i < n_gen; i++) printf("%s%d", i ? "," : " ", gen[i]);
    printf("\n");
    double prefill_tps = n_prompt / t_prefill;
    double decode_tps = n_gen > 1 ? (n_gen - 1) / t_decode : 0.0;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double rss_mb = ru.ru_maxrss / 1024.0;  /* Linux reports KiB */
    printf("load %.3f s | prefill %d tok in %.3f s = %.3f tok/s | decode %d tok in %.3f s = "
           "%.3f tok/s | peak RSS %.1f MB | matvec=%s threads=%d\n",
           t_load, n_prompt, t_prefill, prefill_tps, n_gen > 1 ? n_gen - 1 : 0, t_decode,
           decode_tps, rss_mb, PF_MATVEC_NAME, threads);
    if (bench)
        printf("RESULT {\"engine\": \"pinfer-%s\", \"model\": \"%s\", \"threads\": %d, "
               "\"n_prompt\": %d, \"n_gen\": %d, \"prefill_tps\": %.4f, \"decode_tps\": %.4f, "
               "\"peak_rss_mb\": %.1f}\n",
               PF_MATVEC_NAME, base_name(model), threads, n_prompt, n_gen, prefill_tps,
               decode_tps, rss_mb);

    free(gen); free(hid); free(all); free(logits); free(prompt);
    pinfer_free(ctx);
    return 0;
}
