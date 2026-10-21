/* Throwaway check (issue #5, VERIFY.md): after speculative generation, the context state
 * (conv + recurrent states + main KV rows) must be byte-identical to plain greedy decoding
 * of the same tokens, also after a continuation (pinfer_generate_greedy_cont).
 *   state_cmp model.pq ids.txt N
 * ids.txt: comma-separated prompt ids. Compares blob bytes [72, size_plain) (header =
 * 72 bytes, state.c; the spec blob additionally holds MTP rows at the end). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pinfer.h"

static int read_ids(const char *path, int32_t *ids, int cap) {
    FILE *f = fopen(path, "r");
    int n = 0;
    if (!f) return -1;
    while (n < cap && fscanf(f, "%d%*[, \n]", &ids[n]) == 1) n++;
    fclose(f);
    return n;
}

typedef struct {
    int32_t gen[2][256];
    int ng[2], n_past;
    char *blob;
    size_t size;
} run_t;

static int run(const char *model, const int32_t *p, int np, int N, pinfer_spec_params sp,
               run_t *r) {
    pinfer_params pp = pinfer_default_params();
    pp.max_ctx = 512;
    int err;
    pinfer_ctx *c = pinfer_load(model, &pp, &err);
    if (!c) return fprintf(stderr, "load failed %d\n", err), -1;
    int st;
    r->ng[0] = pinfer_generate_greedy_ex(c, p, np, r->gen[0], N, &sp, NULL, NULL, &st);
    /* continuation: the last emitted token + 3 prompt tokens appended, N/2 more */
    int32_t more[4] = {r->gen[0][r->ng[0] - 1], p[0], p[1], p[2]};
    r->ng[1] = pinfer_generate_greedy_cont(c, more, 4, r->gen[1], N / 2, &sp, NULL, NULL, &st);
    r->n_past = pinfer_n_past(c);
    r->size = pinfer_state_size(c);
    r->blob = malloc(r->size);
    int rc = pinfer_state_save(c, r->blob, r->size);
    pinfer_free(c);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 4) return fprintf(stderr, "usage: state_cmp model ids.txt N\n"), 2;
    int32_t ids[512];
    const int np = read_ids(argv[2], ids, 512), N = atoi(argv[3]);
    pinfer_spec_params cfg[] = {{0, 0, 0}, {1, 32768, 0}, {2, 32768, 0}, {3, 32768, 0},
                                {4, 32768, 0}, {3, 0, 0}, {3, 32768, 1}, {6, 32768, 0}};
    const int ncfg = (int)(sizeof cfg / sizeof cfg[0]);
    run_t ref, r;
    if (np <= 3 || run(argv[1], ids, np, N, cfg[0], &ref) != PINFER_OK) return 1;
    int fails = 0;
    for (int i = 1; i < ncfg; i++) {
        if (run(argv[1], ids, np, N, cfg[i], &r) != PINFER_OK) return 1;
        const int tok = r.ng[0] == ref.ng[0] && r.ng[1] == ref.ng[1] &&
                        !memcmp(r.gen[0], ref.gen[0], sizeof(int32_t) * (size_t)ref.ng[0]) &&
                        !memcmp(r.gen[1], ref.gen[1], sizeof(int32_t) * (size_t)ref.ng[1]);
        const int state = r.n_past == ref.n_past && r.size >= ref.size &&
                          !memcmp(r.blob + 72, ref.blob + 72, ref.size - 72);
        printf("k=%d dv=%d adaptive=%d: tokens %s (%d+%d), state %s (n_past %d, %zu B)\n",
               cfg[i].k, cfg[i].draft_vocab, cfg[i].adaptive, tok ? "same" : "DIFFER", r.ng[0],
               r.ng[1], state ? "identical" : "DIFFERS", r.n_past, ref.size - 72);
        fails += !tok || !state;
        free(r.blob);
    }
    printf("%s\n", fails ? "FAILED" : "all identical");
    return fails ? 1 : 0;
}
