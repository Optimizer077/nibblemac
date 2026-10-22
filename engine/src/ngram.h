/* Prompt-lookup drafting (engine/DRAFT.md): when the newest tokens repeat an n-gram that
 * occurred earlier in the context (the prompt, a document being edited, the model's own
 * output), the tokens that followed that earlier occurrence are proposed as the draft. No
 * weights; the main model verifies the draft exactly like an MTP draft.
 *
 * The index maps every n-gram (n = PF_NG_MIN..PF_NG_MAX) of the token history to the
 * position where it most recently ended: one direct-mapped hash table per n, entries
 * overwritten by newer occurrences, every hit verified against the history (a collision
 * can only lose a match, never produce a wrong one). New positions are indexed lazily by the
 * next lookup, so a commit only appends tokens. */
#ifndef PF_NGRAM_H
#define PF_NGRAM_H

#include <stdint.h>

#define PF_NG_MIN 2   /* shortest n-gram looked up */
#define PF_NG_MAX 4   /* longest; the longest match wins */
#define PF_NG_EXT 32  /* the reported match length is extended backwards up to this */

typedef struct {
    int32_t *tok;     /* [cap]: the token history, oldest first */
    int32_t *tab;     /* [PF_NG_MAX - PF_NG_MIN + 1][1 << bits]: end position + 1, 0 = empty */
    int cap, n;       /* capacity, tokens in the history */
    int indexed;      /* n-grams ending at positions < indexed are in tab */
    int bits;
} pf_ngram;

/* Room for cap tokens (idempotent for a smaller cap). Returns 0 or -1 (out of memory). */
int pf_ngram_init(pf_ngram *g, int cap);
void pf_ngram_free(pf_ngram *g);
void pf_ngram_reset(pf_ngram *g);
/* Append m tokens (tokens past the capacity are dropped: drafting only gets weaker). */
void pf_ngram_push(pf_ngram *g, const int32_t *t, int m);
/* Forget the newest token (an emitted token that was replaced, not fed). */
void pf_ngram_pop(pf_ngram *g);
/* Draft up to max tokens that followed the most recent earlier occurrence of the longest
 * matching tail n-gram of the history. Returns the draft length (0 = no match) and, if
 * match is not NULL, how many tokens the two occurrences share backwards (n-gram length
 * .. PF_NG_EXT): long shared contexts mean reliable drafts (DRAFT.md). */
int pf_ngram_draft(pf_ngram *g, int32_t *out, int max, int *match);

#endif
