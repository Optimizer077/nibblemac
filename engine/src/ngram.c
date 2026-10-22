/* Prompt-lookup drafting: n-gram index over the token history (ngram.h, engine/DRAFT.md). */
#include "ngram.h"

#include <stdlib.h>
#include <string.h>

#define NG_TABLES (PF_NG_MAX - PF_NG_MIN + 1)

static uint32_t hash(const int32_t *t, int n, int bits) {
    uint32_t h = 0x811C9DC5u ^ (uint32_t)n;
    for (int i = 0; i < n; i++) h = (h ^ (uint32_t)t[i]) * 0x9E3779B1u;
    return h >> (32 - bits);
}

int pf_ngram_init(pf_ngram *g, int cap) {
    if (g->tok && g->cap >= cap) return 0;
    pf_ngram_free(g);
    int bits = 10;
    while (bits < 24 && (1 << bits) < 2 * cap) bits++;
    g->tok = malloc((size_t)cap * sizeof(int32_t));
    g->tab = calloc((size_t)NG_TABLES << bits, sizeof(int32_t));
    if (!g->tok || !g->tab) {
        pf_ngram_free(g);
        return -1;
    }
    g->cap = cap;
    g->bits = bits;
    g->n = g->indexed = 0;
    return 0;
}

void pf_ngram_free(pf_ngram *g) {
    free(g->tok);
    free(g->tab);
    memset(g, 0, sizeof *g);
}

void pf_ngram_reset(pf_ngram *g) {
    if (!g->tab) return;
    memset(g->tab, 0, ((size_t)NG_TABLES << g->bits) * sizeof(int32_t));
    g->n = g->indexed = 0;
}

void pf_ngram_push(pf_ngram *g, const int32_t *t, int m) {
    if (!g->tok) return;
    if (m > g->cap - g->n) m = g->cap - g->n;
    if (m <= 0) return;
    memcpy(g->tok + g->n, t, (size_t)m * sizeof(int32_t));
    g->n += m;
}

void pf_ngram_pop(pf_ngram *g) {
    if (g->n > 0) g->n--;
    if (g->indexed > g->n) g->indexed = g->n;   /* stale entries fail the checks below */
}

/* Index the n-grams that end at positions [indexed, upto). */
static void index_to(pf_ngram *g, int upto) {
    for (int q = g->indexed; q < upto; q++)
        for (int n = PF_NG_MIN; n <= PF_NG_MAX && n <= q + 1; n++)
            g->tab[((size_t)(n - PF_NG_MIN) << g->bits) + hash(g->tok + q - n + 1, n, g->bits)] =
                q + 1;
    if (upto > g->indexed) g->indexed = upto;
}

int pf_ngram_draft(pf_ngram *g, int32_t *out, int max, int *n_match) {
    const int n = g->n;
    if (n_match) *n_match = 0;
    if (!g->tok || max <= 0 || n < PF_NG_MIN + 1) return 0;
    index_to(g, n - 1);   /* earlier occurrences end before the tail's last token */
    for (int k = PF_NG_MAX; k >= PF_NG_MIN; k--) {
        if (k > n - 1) continue;
        const int32_t *tail = g->tok + n - k;
        const int q = g->tab[((size_t)(k - PF_NG_MIN) << g->bits) + hash(tail, k, g->bits)] - 1;
        if (q < k - 1 || q >= n - 1 || memcmp(g->tok + q - k + 1, tail, (size_t)k * sizeof(int32_t)))
            continue;
        const int m = n - 1 - q < max ? n - 1 - q : max;
        memcpy(out, g->tok + q + 1, (size_t)m * sizeof(int32_t));
        if (n_match) {   /* how far back the two occurrences agree: the draft's reliability */
            int e = k;
            while (e < PF_NG_EXT && q - e >= 0 && g->tok[q - e] == g->tok[n - 1 - e]) e++;
            *n_match = e;
        }
        return m;
    }
    return 0;
}
