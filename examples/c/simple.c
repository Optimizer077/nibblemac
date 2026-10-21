/* Greedy generation with the pinfer C API only. Token ids in, token ids out
 * (tokenize with the Python package or any Qwen tokenizer):
 *   pinfer_example model.pq 760,45980,20010,220,19,369,264,2526,6165,421 [n_new]   */
#include <stdio.h>
#include <stdlib.h>
#include "pinfer.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.pq id,id,... [n_new]\n", argv[0]); return 2; }
    int32_t ids[512]; int n = 0, n_new = argc > 3 ? atoi(argv[3]) : 16, err = 0;
    for (char *s = argv[2]; *s && n < 512; ) { ids[n++] = (int32_t)strtol(s, &s, 10); if (*s) s++; }
    pinfer_params p = pinfer_default_params();
    p.max_ctx = n + n_new;
    pinfer_ctx *ctx = pinfer_load(argv[1], &p, &err);
    if (!ctx) { fprintf(stderr, "pinfer_load failed: %d\n", err); return 1; }
    float *logits = malloc(sizeof(float) * (size_t)pinfer_n_vocab(ctx));
    int rc = pinfer_eval(ctx, ids, n, logits);           /* prefill the whole prompt */
    for (int i = 0; rc == PINFER_OK && i < n_new; i++) {
        int32_t tok = pinfer_argmax(logits, pinfer_n_vocab(ctx));
        printf(i ? ",%d" : "%d", tok);
        if (tok == pinfer_eos_id(ctx) || i + 1 == n_new) break;
        rc = pinfer_eval(ctx, &tok, 1, logits);          /* decode one token */
    }
    printf("\n");
    if (rc != PINFER_OK) fprintf(stderr, "pinfer_eval failed: %d\n", rc);
    free(logits); pinfer_free(ctx);
    return rc == PINFER_OK ? 0 : 1;
}
