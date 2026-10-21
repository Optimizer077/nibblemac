/* pinfer_ppl: perplexity with llama.cpp's llama-perplexity protocol, so numbers compare 1:1
 * (bench/quality/ppl.sh): chunks of `ctx` tokens, state reset per chunk, the first half of
 * each chunk is context only, logits at positions ctx/2 .. ctx-2 predict the next token.
 *
 *   pinfer_ppl -m model.pq -f tokens.bin [-c 512] [--chunks 64] [-t 4] [--kv-dtype 0|1|2]
 * tokens.bin: raw little-endian int32 token ids (bench/quality/make_tokens.py). Every id the
 * chunks use is checked against the vocabulary before the first forward pass (a scored
 * target indexes the logits directly). Prints progress and one `RESULT {json}` line.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "forward.h"
#include "pinfer.h"

#define SUB 64 /* positions per forward call: bounds the logits buffer to SUB x vocab floats */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* The whole file as int32 ids; NULL (n = 0) if unreadable, empty or not a multiple of 4. */
static int32_t *read_tokens(const char *path, long *n) {
    *n = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    long bytes = fseek(f, 0, SEEK_END) == 0 ? ftell(f) : -1;
    int32_t *t = bytes > 0 && bytes % 4 == 0 && fseek(f, 0, SEEK_SET) == 0
                     ? malloc((size_t)bytes) : NULL;
    if (t && fread(t, 1, (size_t)bytes, f) == (size_t)bytes) *n = bytes / 4;
    fclose(f);
    if (*n == 0) { free(t); return NULL; }
    return t;
}

/* -log softmax(l)[target], computed stably in double. */
static double nll(const float *l, int vocab, int target) {
    float mx = l[0];
    for (int i = 1; i < vocab; i++) if (l[i] > mx) mx = l[i];
    double s = 0;
    for (int i = 0; i < vocab; i++) s += exp((double)l[i] - mx);
    return log(s) + mx - l[target];
}

int main(int argc, char **argv) {
    const char *model = NULL, *tok_path = NULL;
    int ctx_len = 512, chunks = 64, threads = 4, kv_dtype = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) ctx_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunks") && i + 1 < argc) chunks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-dtype") && i + 1 < argc) kv_dtype = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s -m model.pq -f tokens.bin [-c 512] [--chunks N] [-t T] [--kv-dtype D]\n", argv[0]); return 2; }
    }
    if (!model || !tok_path || ctx_len < 4 || chunks < 1) {
        fprintf(stderr, "need -m and -f, -c >= 4 and --chunks >= 1\n");
        return 2;
    }

    long n_tok = 0;
    int32_t *tok = read_tokens(tok_path, &n_tok);
    if (!tok || n_tok < ctx_len) { fprintf(stderr, "cannot read %s or too few tokens\n", tok_path); return 1; }
    if ((long)chunks * ctx_len > n_tok) chunks = (int)(n_tok / ctx_len);

    pinfer_params p = pinfer_default_params();
    p.n_threads = threads;
    p.max_ctx = ctx_len;
    p.kv_dtype = kv_dtype;
    int err = 0;
    pinfer_ctx *ctx = pinfer_load(model, &p, &err);
    if (!ctx) { fprintf(stderr, "load failed: %d\n", err); return 1; }
    int vocab = pinfer_n_vocab(ctx);
    for (long i = 0; i < (long)chunks * ctx_len; i++)
        if (tok[i] < 0 || tok[i] >= vocab) {
            fprintf(stderr, "%s: token %ld = %d outside the vocabulary [0, %d)\n", tok_path, i,
                    tok[i], vocab);
            return 1;
        }
    float *logits = malloc((size_t)SUB * vocab * sizeof(float));
    if (!logits) { fprintf(stderr, "out of memory\n"); return 1; }

    const int first = ctx_len / 2;
    double sum = 0, sum2 = 0;
    long scored = 0;
    double t0 = now_sec();
    for (int c = 0; c < chunks; c++) {
        const int32_t *ch = tok + (long)c * ctx_len;
        pinfer_reset(ctx);
        for (int s = 0; s < ctx_len; s += SUB) {
            int n = ctx_len - s < SUB ? ctx_len - s : SUB;
            if (pf_forward(ctx, ch + s, n, NULL, logits, NULL) != PINFER_OK) {
                fprintf(stderr, "forward failed at chunk %d pos %d\n", c, s);
                return 1;
            }
            for (int j = 0; j < n; j++) {
                int pos = s + j;
                if (pos < first || pos >= ctx_len - 1) continue;
                double v = nll(logits + (size_t)j * vocab, vocab, ch[pos + 1]);
                sum += v;
                sum2 += v * v;
                scored++;
            }
        }
        fprintf(stderr, "[%d/%d] ppl %.4f  (%.1f tok/s)\n", c + 1, chunks, exp(sum / scored),
                (double)(c + 1) * ctx_len / (now_sec() - t0));
    }
    double mean = sum / scored;
    double var = sum2 / scored - mean * mean;
    double ppl = exp(mean);
    /* Same uncertainty estimate as llama-perplexity: d(ppl) = ppl * stderr(nll). */
    double ppl_err = ppl * sqrt(var > 0 && scored > 1 ? var / (scored - 1) : 0);
    double secs = now_sec() - t0;
    printf("RESULT {\"engine\":\"pinfer\",\"tool\":\"pinfer_ppl\",\"kernels\":\"%s\",\"model_file\":\"%s\","
           "\"ctx\":%d,\"chunks\":%d,\"tokens_scored\":%ld,\"threads\":%d,\"ppl\":%.4f,"
           "\"ppl_err\":%.5f,\"seconds\":%.1f,\"kv_dtype\":%d}\n",
           PF_MATVEC_NAME, model, ctx_len, chunks, scored, threads, ppl, ppl_err, secs, kv_dtype);
    free(logits);
    free(tok);
    pinfer_free(ctx);
    return 0;
}
