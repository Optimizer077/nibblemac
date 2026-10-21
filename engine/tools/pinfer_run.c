/*
 * pinfer_run: feed a prompt of token ids, greedily generate N tokens, report speed.
 *
 *   pinfer_run -m model.pq [-t threads] [-n N] [-c max_ctx] [-b prefill_batch]
 *              (--ids 1,2,3 | --ids-file f) [--dump-logits out.bin] [--dump-hidden out.bin]
 *              [--bench] [--spec k [--draft-vocab N]] [--dump-mtp out.bin]
 *
 * Token ids only: tokenization lives in the Python package. --dump-* write raw little-endian
 * fp32 for the PROMPT tokens: logits [T][vocab], hidden [n_layers+1][T][hidden] (same layout
 * as ref/golden). Decode speed counts the N-1 single-token evals after the first generated
 * token (that one comes from the prefill logits). --bench adds one `RESULT {json}` line for
 * harness/pi.py.
 * --spec k: greedy decoding with MTP self-speculation (k drafts per batched verification
 * pass; k = 0 runs the plain loop through the same code path). The output must equal the
 * plain run; RESULT also carries rounds / drafted / accepted and per-pass timings. The MTP
 * pass over the prompt is part of the prefill time. --draft-vocab N: drafts only from token
 * ids < N (cheaper draft head, same output).
 * --dump-mtp f: MTP logits of every prompt row (row i uses token i+1; the last row uses the
 * first generated token), then 3 chained draft logits from the last row: fp32 [T+3][vocab],
 * the layout of ref/golden/qwen35-0.8b-mtp (mtp_logits.npy then chain_logits.npy).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "../src/forward.h"
#include "../src/mtp.h"
#include "../src/spec.h"

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
                    "                  [--dump-logits f] [--dump-hidden f] [--bench]\n"
                    "                  [--spec k [--draft-vocab N]] [--dump-mtp f]\n");
}

/* --dump-mtp: see the header comment. normed = the prompt's final-normed hiddens [T][H]. */
static int dump_mtp(pinfer_ctx *ctx, const char *path, const int32_t *prompt, int T,
                    const float *normed, int32_t first, int K) {
    const size_t V = (size_t)ctx->vocab, H = (size_t)ctx->H;
    int rc = pf_mtp_init(ctx);
    float *lg = malloc((size_t)(T + K - 1) * V * sizeof(float));
    float *m = malloc(PF_MAX_BATCH * H * sizeof(float)), *m2 = malloc(H * sizeof(float));
    if (rc != PINFER_OK || !lg || !m || !m2) {
        fprintf(stderr, "dump-mtp: init failed (%d)\n", rc);
        return 1;
    }
    int32_t next[PF_MAX_BATCH];
    for (int c0 = 0; c0 < T && rc == PINFER_OK; c0 += PF_MAX_BATCH) {
        int nb = T - c0 < PF_MAX_BATCH ? T - c0 : PF_MAX_BATCH;
        for (int i = 0; i < nb; i++) next[i] = c0 + i + 1 < T ? prompt[c0 + i + 1] : first;
        rc = pf_mtp_forward(ctx, normed + (size_t)c0 * H, next, nb, c0, m, NULL,
                            lg + (size_t)c0 * V, 0);
    }
    const float *mi = m + (size_t)((T - 1) % PF_MAX_BATCH) * H;
    for (int s = 1; s < K && rc == PINFER_OK; s++) {   /* chain from the last prompt row */
        int32_t d = pinfer_argmax(lg + (size_t)(T + s - 2) * V, (int)V);
        rc = pf_mtp_forward(ctx, mi, &d, 1, T - 1 + s, m2, lg + (size_t)(T + s - 1) * V,
                            NULL, 0);
        memcpy(m, m2, H * sizeof(float));
        mi = m;
    }
    if (rc != PINFER_OK) fprintf(stderr, "dump-mtp: MTP forward failed (%d)\n", rc);
    int err = rc != PINFER_OK || write_f32(path, lg, (size_t)(T + K - 1) * V);
    free(lg);
    free(m);
    free(m2);
    return err;
}

int main(int argc, char **argv) {
    const char *model = NULL, *ids_arg = NULL, *ids_file = NULL;
    const char *dump_logits = NULL, *dump_hidden = NULL, *dump_mtp_f = NULL;
    int threads = 4, n_gen = 32, max_ctx = 0, batch = PF_MAX_BATCH, bench = 0;
    int spec_k = -1, draft_vocab = 0;
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
        else if (!strcmp(a, "--spec") && v) spec_k = atoi(v);
        else if (!strcmp(a, "--draft-vocab") && v) draft_vocab = atoi(v);
        else if (!strcmp(a, "--dump-mtp") && v) dump_mtp_f = v;
        else if (!strcmp(a, "--bench")) { bench = 1; used = 0; }
        else { usage(); return 2; }
        i += used;
    }
    if (!model || (!ids_arg == !ids_file) || n_gen < 0 || spec_k > PINFER_SPEC_MAX_K ||
        (spec_k >= 0 && (n_gen < 1 || dump_logits || dump_hidden || dump_mtp_f))) {
        usage();
        return 2;
    }

    enum { MAX_PROMPT = 1 << 16 };
    int32_t *prompt = malloc(MAX_PROMPT * sizeof(int32_t));
    char *text = ids_file ? read_file(ids_file) : NULL;
    int n_prompt = prompt ? parse_ids(ids_file ? (text ? text : "") : ids_arg, prompt,
                                      MAX_PROMPT) : -1;
    free(text);
    if (n_prompt <= 0) { fprintf(stderr, "no prompt ids\n"); return 2; }

    pinfer_params p = pinfer_default_params();
    p.n_threads = threads;
    /* --dump-mtp chains 3 MTP rows past the prompt */
    p.max_ctx = max_ctx > 0 ? max_ctx : n_prompt + n_gen + (dump_mtp_f ? 4 : 0);
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

    pinfer_spec_params sp = {spec_k, draft_vocab};
    pinfer_spec_stats st = {0, 0, 0};
    pf_spec_times tm = {0, 0, 0};
    double t_prefill, t_decode;
    int rc;
    if (spec_k >= 0) {  /* speculative path (k = 0: plain loop inside the same code) */
        t0 = now_sec();
        rc = pf_spec_prefill(ctx, prompt, n_prompt, &sp);
        t_prefill = now_sec() - t0;
        if (rc < 0) { fprintf(stderr, "prefill failed: %d\n", rc); return 1; }
        t0 = now_sec();
        pf_stop no_stop = {-1, NULL, 0, NULL, NULL};  /* like the plain loop: ignore EOS */
        int status = PINFER_OK;
        rc = pf_spec_decode(ctx, gen, n_gen, &sp, &no_stop, &st, &tm, &status);
        t_decode = now_sec() - t0;
        if (rc != n_gen || status != PINFER_OK) {
            fprintf(stderr, "spec decode failed: %d tokens, status %d\n", rc, status);
            return 1;
        }
    } else {
        float *normed = dump_mtp_f ? malloc((size_t)n_prompt * H * sizeof(float)) : NULL;
        if (dump_mtp_f && !normed) { fprintf(stderr, "out of memory\n"); return 1; }
        t0 = now_sec();
        rc = pf_forward_h(ctx, prompt, n_prompt, logits, all, hid, normed);
        t_prefill = now_sec() - t0;
        if (rc != PINFER_OK) { fprintf(stderr, "prefill failed: %d\n", rc); return 1; }
        if (all && write_f32(dump_logits, all, (size_t)n_prompt * V)) return 1;
        if (hid && write_f32(dump_hidden, hid, (size_t)(L + 1) * n_prompt * H)) return 1;
        if (normed && dump_mtp(ctx, dump_mtp_f, prompt, n_prompt, normed,
                               pinfer_argmax(logits, V), 4)) return 1;
        free(normed);

        t0 = now_sec();
        for (int i = 0; i < n_gen; i++) {
            if (i > 0 && (rc = pinfer_eval(ctx, &gen[i - 1], 1, logits)) != PINFER_OK) {
                fprintf(stderr, "decode step %d failed: %d\n", i, rc);
                return 1;
            }
            gen[i] = pinfer_argmax(logits, V);
        }
        t_decode = now_sec() - t0;
    }

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
    /* spec: tokens per verification pass = 1 + mean accepted; timings per pass in ms */
    double R = st.rounds > 0 ? (double)st.rounds : 1.0;
    double mean_acc = (double)st.accepted / R;
    if (spec_k >= 0)
        printf("spec k=%d draft_vocab=%d | rounds %lld drafted %lld accepted %lld | mean "
               "accepted %.3f, %.3f tok/pass | per pass: mtp %.1f ms verify %.1f ms commit "
               "%.1f ms\n", spec_k, draft_vocab, (long long)st.rounds, (long long)st.drafted,
               (long long)st.accepted, mean_acc, 1.0 + mean_acc, 1e3 * tm.mtp / R,
               1e3 * tm.verify / R, 1e3 * tm.commit / R);
    if (bench && spec_k < 0)
        printf("RESULT {\"engine\": \"pinfer-%s\", \"model\": \"%s\", \"threads\": %d, "
               "\"n_prompt\": %d, \"n_gen\": %d, \"prefill_tps\": %.4f, \"decode_tps\": %.4f, "
               "\"peak_rss_mb\": %.1f}\n",
               PF_MATVEC_NAME, base_name(model), threads, n_prompt, n_gen, prefill_tps,
               decode_tps, rss_mb);
    else if (bench)
        printf("RESULT {\"engine\": \"pinfer-%s-spec\", \"model\": \"%s\", \"threads\": %d, "
               "\"n_prompt\": %d, \"n_gen\": %d, \"prefill_tps\": %.4f, \"decode_tps\": %.4f, "
               "\"peak_rss_mb\": %.1f, \"spec_k\": %d, \"draft_vocab\": %d, \"rounds\": %lld, "
               "\"drafted\": %lld, \"accepted\": %lld, \"mean_accepted\": %.4f, "
               "\"accept_rate\": %.4f, \"tokens_per_pass\": %.4f, \"mtp_ms_per_pass\": %.2f, "
               "\"verify_ms_per_pass\": %.2f, \"commit_ms_per_pass\": %.2f}\n",
               PF_MATVEC_NAME, base_name(model), threads, n_prompt, n_gen, prefill_tps,
               decode_tps, rss_mb, spec_k, draft_vocab, (long long)st.rounds,
               (long long)st.drafted, (long long)st.accepted, mean_acc,
               st.drafted ? (double)st.accepted / (double)st.drafted : 0.0, 1.0 + mean_acc,
               1e3 * tm.mtp / R, 1e3 * tm.verify / R, 1e3 * tm.commit / R);

    free(gen); free(hid); free(all); free(logits); free(prompt);
    pinfer_free(ctx);
    return 0;
}
