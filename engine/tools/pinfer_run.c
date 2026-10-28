/*
 * pinfer_run: feed a prompt of token ids, greedily generate N tokens, report speed.
 *
 *   pinfer_run -m model.pq [-t threads] [-n N] [-c max_ctx] [-b prefill_batch]
 *              (--ids 1,2,3 | --ids-file f) [--dump-logits out.bin] [--dump-hidden out.bin]
 *              [--bench] [--warmup | --no-warmup] [--spec k [--draft-vocab N]]
 *              [--dump-mtp out.bin] [--kv-dtype 0|1|2] [--adaptive]
 *              [--save-state f] [--load-state f] [--ngram N] [--rewind-at S]
 *              [--dump-last f]
 *
 * Token ids only: tokenization lives in the Python package. --dump-* write raw little-endian
 * fp32 for the PROMPT tokens: logits [T][vocab], hidden [n_layers+1][T][hidden] (same layout
 * as ref/golden). Decode speed counts the N-1 single-token evals after the first generated
 * token (that one comes from the prefill logits). --bench adds one `RESULT {json}` line for
 * harness/pi.py and, unless --no-warmup, first runs an untimed prefill + 4 tokens in the same
 * mode and resets: the first prefill after a reboot pages 0.5-1.5 GB of mmap'd weights in
 * from the SSD (measured 20 vs 27 tok/s), which must not depend on run order. --warmup
 * enables it without --bench.
 * --spec k: greedy decoding with MTP self-speculation (k drafts per batched verification
 * pass; k = 0 runs the plain loop through the same code path). The output must equal the
 * plain run; RESULT also carries rounds / drafted / accepted and per-pass timings. The MTP
 * pass over the prompt is part of the prefill time. --draft-vocab N: drafts only from token
 * ids < N (cheaper draft head, same output).
 * --dump-mtp f: MTP logits of every prompt row (row i uses token i+1; the last row uses the
 * first generated token), then 3 chained draft logits from the last row: fp32 [T+3][vocab],
 * the layout of ref/golden/qwen35-0.8b-mtp (mtp_logits.npy then chain_logits.npy).
 * --kv-dtype: KV-cache storage, 0 fp32 (default), 1 fp16, 2 int8 (pinfer_params.kv_dtype).
 * --adaptive: with --spec k, let the engine lower k (down to plain steps) when that is faster.
 * --ngram N: prompt-lookup drafts of up to N tokens when the newest tokens repeat earlier
 * context (pinfer_spec_params.ngram, engine/DRAFT.md); with --spec 0 the only drafts.
 * --save-state f: write the context state right after the prefill (pinfer_state_save).
 * --load-state f: restore a saved state first; the prompt is then appended to it (plain path:
 * pinfer_eval semantics; --spec: pinfer_generate_greedy_cont semantics), so
 *   run A: --ids-file full.ids      run B: --ids-file head.ids --save-state s -n 1
 *   run C: --load-state s --ids-file tail.ids
 * must generate the same tokens in A and C (the state round trip is exact).
 * --rewind-at S: in-context checkpoint check (pinfer_checkpoint / pinfer_rewind). The first
 * S prompt tokens are fed (appended to a loaded state, if any), then pinfer_checkpoint; then
 * twice a detour, rewound with pinfer_rewind: other tokens (the prompt reversed, rotated by
 * the round, n_prompt - S of them) fed in the run's mode and n tokens generated from them.
 * The rest of the prompt follows as usual, so the output must equal the run on the whole
 * prompt. Prints the checkpoint and the (last) rewind time.
 * --dump-last f: the prefill's last-token logits, fp32 [vocab] (plain path). Without the other
 * --dump-* options the prefill computes only what those logits need (forward.c, "tail"), so
 * comparing this file with the last row of --dump-logits checks that path bit for bit.
 */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include <stdio.h>
#include <math.h>
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

static int state_save(pinfer_ctx *ctx, const char *path, double *ms) {
    const size_t n = pinfer_state_size(ctx);
    void *buf = malloc(n);
    double t0 = now_sec();
    int rc = buf ? pinfer_state_save(ctx, buf, n) : PINFER_E_NOMEM;
    *ms = (now_sec() - t0) * 1e3;
    FILE *f = rc == PINFER_OK ? fopen(path, "wb") : NULL;
    int err = rc != PINFER_OK || !f || fwrite(buf, 1, n, f) != n;
    if (f && fclose(f)) err = 1;
    free(buf);
    if (err) fprintf(stderr, "save-state %s failed (%d)\n", path, rc);
    else fprintf(stderr, "state saved: %zu bytes in %.1f ms\n", n, *ms);
    return err;
}

static int state_load(pinfer_ctx *ctx, const char *path, double *ms) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = sz > 0 ? malloc((size_t)sz) : NULL;
    size_t got = buf ? fread(buf, 1, (size_t)sz, f) : 0;
    fclose(f);
    double t0 = now_sec();
    int rc = got == (size_t)sz && sz > 0 ? pinfer_state_load(ctx, buf, got) : PINFER_E_IO;
    *ms = (now_sec() - t0) * 1e3;
    free(buf);
    if (rc != PINFER_OK) fprintf(stderr, "load-state %s failed (%d)\n", path, rc);
    else fprintf(stderr, "state loaded: %ld bytes in %.1f ms, n_past %d\n", sz, *ms,
                 pinfer_n_past(ctx));
    return rc != PINFER_OK;
}

/* n_past stored in a state file (header: magic[8], version, n_past), 0 if unreadable. */
static int state_n_past(const char *path) {
    FILE *f = fopen(path, "rb");
    uint32_t hdr[4] = {0, 0, 0, 0};
    if (f && fread(hdr, sizeof hdr, 1, f) != 1) hdr[3] = 0;
    if (f) fclose(f);
    return (int)hdr[3];
}

static const char *base_name(const char *p) {
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

static void usage(void) {
    fprintf(stderr, "usage: pinfer_run -m model.pq [-t threads] [-n N] [-c max_ctx] "
                    "[-b batch] (--ids 1,2,3 | --ids-file f)\n"
                    "                  [--dump-logits f] [--dump-hidden f] [--bench]\n"
                    "                  [--warmup | --no-warmup] [--spec k [--draft-vocab N]]\n"
                    "                  [--dump-mtp f] [--kv-dtype 0|1|2] [--gpu SHARE] [--adaptive]\n"
                    "                  [--save-state f] [--load-state f] [--ngram N]\n"
                    "                  [--rewind-at S] [--dump-last f]\n");
}

/* Bytes pinfer_checkpoint copies: conv + recurrent state of every linear layer (state.c). */
static size_t checkpoint_bytes(const pinfer_ctx *ctx) {
    size_t n = 0;
    for (int l = 0; l < ctx->n_layers; l++)
        if (!ctx->model.layers[l].is_full)
            n += ((size_t)(ctx->conv_k - 1) * ctx->conv_dim +
                  (size_t)ctx->lv_heads * ctx->lk_dim * ctx->lv_dim) * sizeof(float);
    return n;
}

/* --rewind-at S (header comment): head, checkpoint, two detours each followed by a rewind.
 * spec = 1: the run's speculative mode (pf_spec_prefill / pf_spec_decode), else pinfer_eval
 * semantics. gen [n_gen] and logits [vocab] are scratch. */
static int rewind_probe(pinfer_ctx *ctx, const int32_t *prompt, int S, int n_prompt,
                        const pinfer_spec_params *sp, int spec, int append, int32_t *gen,
                        int n_gen, float *logits, double *ms_ck, double *ms_rw) {
    int rc = spec ? pf_spec_prefill(ctx, prompt, S, sp, append)
                  : pf_forward(ctx, prompt, S, NULL, NULL, NULL);
    if (rc < 0) { fprintf(stderr, "rewind-at: head prefill failed: %d\n", rc); return 1; }
    const int base = pinfer_n_past(ctx), nd = n_prompt - S;
    double t0 = now_sec();
    rc = pinfer_checkpoint(ctx);
    *ms_ck = 1e3 * (now_sec() - t0);
    if (rc != PINFER_OK || pinfer_checkpoint_n_past(ctx) != base) {
        fprintf(stderr, "rewind-at: checkpoint failed: %d\n", rc);
        return 1;
    }
    int32_t *d = malloc((size_t)nd * sizeof(int32_t));
    if (!d) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int round = 0; round < 2; round++) {
        for (int i = 0; i < nd; i++) d[i] = prompt[(2 * n_prompt - 1 - i - round) % n_prompt];
        if (spec) {
            pf_stop no_stop = {-1, NULL, 0, NULL, NULL};
            int status = PINFER_OK;
            rc = pf_spec_prefill(ctx, d, nd, sp, 1);
            if (rc >= 0) rc = pf_spec_decode(ctx, gen, n_gen, sp, &no_stop, NULL, NULL, &status);
            if (rc >= 0 && status != PINFER_OK) rc = status;
        } else {
            rc = pf_forward(ctx, d, nd, logits, NULL, NULL);
            for (int i = 0; rc == PINFER_OK && i < n_gen; i++) {
                if (i > 0) rc = pinfer_eval(ctx, &gen[i - 1], 1, logits);
                gen[i] = pinfer_argmax(logits, pinfer_n_vocab(ctx));
            }
        }
        if (rc < 0) { fprintf(stderr, "rewind-at: detour %d failed: %d\n", round, rc); break; }
        t0 = now_sec();
        rc = pinfer_rewind(ctx);
        *ms_rw = 1e3 * (now_sec() - t0);
        if (rc != PINFER_OK || pinfer_n_past(ctx) != base) {
            fprintf(stderr, "rewind-at: rewind %d failed: %d (n_past %d, want %d)\n", round, rc,
                    pinfer_n_past(ctx), base);
            rc = rc != PINFER_OK ? rc : PINFER_E_ARG;
            break;
        }
    }
    free(d);
    return rc < 0;
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
    const char *save_state = NULL, *load_state = NULL, *dump_last = NULL;
    int threads = 4, n_gen = 32, max_ctx = 0, batch = PF_MAX_BATCH, bench = 0;
    int spec_k = -1, draft_vocab = 0, warmup = -1, kv_dtype = 0, adaptive = 0, ngram = 0;
    int rewind_at = 0;
    float gpu_split = 0.0f;
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
        else if (!strcmp(a, "--kv-dtype") && v) kv_dtype = atoi(v);
        else if (!strcmp(a, "--gpu") && v) {
            char *end;
            gpu_split = strtof(v, &end);
            if (end == v || *end || !isfinite(gpu_split) || gpu_split < 0.0f || gpu_split > 1.0f) {
                fprintf(stderr, "--gpu SHARE must be finite and in [0,1]\n"); return 2;
            }
        }
        else if (!strcmp(a, "--save-state") && v) save_state = v;
        else if (!strcmp(a, "--load-state") && v) load_state = v;
        else if (!strcmp(a, "--ngram") && v) ngram = atoi(v);
        else if (!strcmp(a, "--rewind-at") && v) rewind_at = atoi(v);
        else if (!strcmp(a, "--dump-last") && v) dump_last = v;
        else if (!strcmp(a, "--adaptive")) { adaptive = 1; used = 0; }
        else if (!strcmp(a, "--bench")) { bench = 1; used = 0; }
        else if (!strcmp(a, "--warmup")) { warmup = 1; used = 0; }
        else if (!strcmp(a, "--no-warmup")) { warmup = 0; used = 0; }
        else { usage(); return 2; }
        i += used;
    }
    if (!model || (!ids_arg == !ids_file) || n_gen < 0 || spec_k > PINFER_SPEC_MAX_K ||
        ngram < 0 || ngram > PINFER_SPEC_MAX_K || (ngram > 0 && spec_k < 0) ||
        (spec_k >= 0 &&
         (n_gen < 1 || dump_logits || dump_hidden || dump_mtp_f || dump_last)) ||
        (load_state && (dump_logits || dump_hidden || dump_mtp_f)) || rewind_at < 0 ||
        (rewind_at > 0 && (n_gen < 1 || dump_logits || dump_hidden || dump_mtp_f))) {
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
    if (rewind_at >= n_prompt) {
        fprintf(stderr, "--rewind-at needs at least one prompt token after it\n");
        return 2;
    }

    pinfer_params p = pinfer_default_params();
    p.n_threads = threads;
    /* --dump-mtp chains 3 MTP rows past the prompt; a loaded state adds its own tokens */
    p.max_ctx = max_ctx > 0 ? max_ctx : n_prompt + n_gen + (dump_mtp_f ? 4 : 0) +
                                            (load_state ? state_n_past(load_state) : 0);
    p.kv_dtype = kv_dtype;
    p.gpu_split = gpu_split;
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

    pinfer_spec_params sp = {spec_k, draft_vocab, adaptive, ngram};
    pinfer_spec_stats st = {0, 0, 0, 0, 0, 0};
    pf_spec_times tm = {0, 0, 0, 0, 0};
    double t_prefill, t_decode, t_warm = 0.0, ms_save = 0.0, ms_load = 0.0;
    double ms_ck = 0.0, ms_rw = 0.0;
    int rc;
    if (warmup < 0) warmup = bench;
    if (warmup) {  /* untimed prefill + up to 4 tokens, same mode, then forget everything */
        const int w = n_gen < 4 ? n_gen : 4;
        t0 = now_sec();
        if (spec_k >= 0) {
            pf_stop no_stop = {-1, NULL, 0, NULL, NULL};
            int status;
            rc = pf_spec_prefill(ctx, prompt, n_prompt, &sp, 0);
            if (rc >= 0 && w > 0)
                rc = pf_spec_decode(ctx, gen, w, &sp, &no_stop, NULL, NULL, &status);
        } else {
            rc = pf_forward(ctx, prompt, n_prompt, logits, NULL, NULL);
            for (int i = 0; rc == PINFER_OK && i < w; i++) {
                gen[0] = pinfer_argmax(logits, V);
                rc = pinfer_eval(ctx, gen, 1, logits);
            }
        }
        if (rc < 0) { fprintf(stderr, "warm-up failed: %d\n", rc); return 1; }
        pinfer_reset(ctx);
        t_warm = now_sec() - t0;
    }
    if (load_state && state_load(ctx, load_state, &ms_load)) return 1;
    int32_t *const prompt_mem = prompt;
    if (rewind_at > 0) {   /* the head, a checkpoint and two rewound detours; then the tail */
        if (rewind_probe(ctx, prompt, rewind_at, n_prompt, &sp, spec_k >= 0, load_state != NULL,
                         gen, n_gen, logits, &ms_ck, &ms_rw))
            return 1;
        prompt += rewind_at;
        n_prompt -= rewind_at;
    }
    const int append = load_state != NULL || rewind_at > 0;
    if (spec_k >= 0) {  /* speculative path (k = 0: plain loop inside the same code) */
        t0 = now_sec();
        rc = pf_spec_prefill(ctx, prompt, n_prompt, &sp, append);
        t_prefill = now_sec() - t0;
        if (rc < 0) { fprintf(stderr, "prefill failed: %d\n", rc); return 1; }
        if (save_state && state_save(ctx, save_state, &ms_save)) return 1;
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
        if (save_state && state_save(ctx, save_state, &ms_save)) return 1;
        if (all && write_f32(dump_logits, all, (size_t)n_prompt * V)) return 1;
        if (dump_last && write_f32(dump_last, logits, (size_t)V)) return 1;
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
#ifdef __APPLE__
    double rss_mb = ru.ru_maxrss / (1024.0 * 1024.0);  /* Darwin reports bytes */
#else
    double rss_mb = ru.ru_maxrss / 1024.0;  /* Linux reports KiB */
#endif
    printf("load %.3f s | warm-up %.3f s | prefill %d tok in %.3f s = %.3f tok/s | decode %d tok "
           "in %.3f s = %.3f tok/s | peak RSS %.1f MB | matvec=%s threads=%d\n",
           t_load, t_warm, n_prompt, t_prefill, prefill_tps, n_gen > 1 ? n_gen - 1 : 0,
           t_decode, decode_tps, rss_mb, PF_MATVEC_NAME, threads);
    /* spec: tokens per verification pass = 1 + mean accepted; timings per pass in ms */
    double R = st.rounds > 0 ? (double)st.rounds : 1.0;
    double mean_acc = (double)st.accepted / R;
    if (spec_k >= 0)
        printf("spec k=%d draft_vocab=%d adaptive=%d ngram=%d | rounds %lld drafted %lld "
               "accepted %lld | mean accepted %.3f, %.3f tok/pass | per pass: mtp %.1f ms "
               "verify %.1f ms commit %.1f ms | plain steps %lld (%.1f ms each) | ngram rounds "
               "%lld drafted %lld accepted %lld\n", spec_k, draft_vocab, adaptive, ngram,
               (long long)st.rounds, (long long)st.drafted, (long long)st.accepted, mean_acc,
               1.0 + mean_acc, 1e3 * tm.mtp / R, 1e3 * tm.verify / R, 1e3 * tm.commit / R,
               (long long)tm.plain_steps,
               tm.plain_steps ? 1e3 * tm.plain / (double)tm.plain_steps : 0.0,
               (long long)st.ngram_rounds, (long long)st.ngram_drafted,
               (long long)st.ngram_accepted);
    if (save_state || load_state)
        printf("state: save %.1f ms load %.1f ms size %zu bytes\n", ms_save, ms_load,
               pinfer_state_size(ctx));
    if (rewind_at > 0)
        printf("rewind: at %d, checkpoint %.1f ms, rewind %.1f ms, slot %zu bytes\n", rewind_at,
               ms_ck, ms_rw, checkpoint_bytes(ctx));
    if (bench && spec_k < 0)
        printf("RESULT {\"engine\": \"pinfer-%s\", \"model\": \"%s\", \"threads\": %d, "
               "\"n_prompt\": %d, \"n_gen\": %d, \"prefill_tps\": %.4f, \"decode_tps\": %.4f, "
               "\"peak_rss_mb\": %.1f, \"kv_dtype\": %d, \"state_save_ms\": %.1f, "
               "\"state_load_ms\": %.1f, \"rewind_at\": %d, \"checkpoint_ms\": %.2f, "
               "\"rewind_ms\": %.2f, \"checkpoint_bytes\": %zu}\n",
               PF_MATVEC_NAME, base_name(model), threads, n_prompt, n_gen, prefill_tps,
               decode_tps, rss_mb, kv_dtype, ms_save, ms_load, rewind_at, ms_ck, ms_rw,
               checkpoint_bytes(ctx));
    else if (bench)
        printf("RESULT {\"engine\": \"pinfer-%s-spec\", \"model\": \"%s\", \"threads\": %d, "
               "\"n_prompt\": %d, \"n_gen\": %d, \"prefill_tps\": %.4f, \"decode_tps\": %.4f, "
               "\"peak_rss_mb\": %.1f, \"spec_k\": %d, \"draft_vocab\": %d, \"rounds\": %lld, "
               "\"drafted\": %lld, \"accepted\": %lld, \"mean_accepted\": %.4f, "
               "\"accept_rate\": %.4f, \"tokens_per_pass\": %.4f, \"mtp_ms_per_pass\": %.2f, "
               "\"verify_ms_per_pass\": %.2f, \"commit_ms_per_pass\": %.2f, "
               "\"adaptive\": %d, \"plain_steps\": %lld, \"kv_dtype\": %d, "
               "\"state_save_ms\": %.1f, \"state_load_ms\": %.1f, \"ngram\": %d, "
               "\"ngram_rounds\": %lld, \"ngram_drafted\": %lld, \"ngram_accepted\": %lld}\n",
               PF_MATVEC_NAME, base_name(model), threads, n_prompt, n_gen, prefill_tps,
               decode_tps, rss_mb, spec_k, draft_vocab, (long long)st.rounds,
               (long long)st.drafted, (long long)st.accepted, mean_acc,
               st.drafted ? (double)st.accepted / (double)st.drafted : 0.0, 1.0 + mean_acc,
               1e3 * tm.mtp / R, 1e3 * tm.verify / R, 1e3 * tm.commit / R, adaptive,
               (long long)tm.plain_steps, kv_dtype, ms_save, ms_load, ngram,
               (long long)st.ngram_rounds, (long long)st.ngram_drafted,
               (long long)st.ngram_accepted);

    free(gen); free(hid); free(all); free(logits); free(prompt_mem);
    pinfer_free(ctx);
    return 0;
}
