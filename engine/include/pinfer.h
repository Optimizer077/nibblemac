/* pinfer — Raspberry-Pi-4-native inference for Qwen3.5. Public C API.
 *
 * The API works on token ids; tokenization lives in the Python package (HF `tokenizers`).
 * All functions return 0 on success and a negative PINFER_E* code on failure unless noted.
 */
#ifndef PINFER_H
#define PINFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PINFER_VERSION_MAJOR 0
#define PINFER_VERSION_MINOR 1
#define PINFER_VERSION_PATCH 0

enum {
    PINFER_OK = 0,
    PINFER_E_IO = -1,       /* cannot open / map the model file */
    PINFER_E_FORMAT = -2,   /* not a valid .pq file or unsupported version */
    PINFER_E_NOMEM = -3,
    PINFER_E_ARG = -4,      /* invalid argument */
    PINFER_E_CTX_FULL = -5, /* more tokens than max_ctx */
};

typedef struct pinfer_ctx pinfer_ctx;

/* KV-cache storage formats (pinfer_params.kv_dtype). fp32 is exact; fp16 halves and int8
 * quarters the cache size and its per-token read traffic at a small numeric cost
 * (engine/CONTEXT.md). Changing the dtype changes the model's outputs, so it is opt-in. */
enum {
    PINFER_KV_F32 = 0,
    PINFER_KV_F16 = 1,   /* IEEE binary16 */
    PINFER_KV_I8 = 2,    /* int8 with one fp32 scale per cached row */
};

typedef struct {
    int n_threads; /* worker threads for matrix ops; <= 0 means 4 */
    int max_ctx;   /* max tokens kept in the KV cache; <= 0 means 4096 */
    int kv_dtype;  /* PINFER_KV_*; 0 = fp32 (added in 0.1: zero-initialised means fp32) */
} pinfer_params;

pinfer_params pinfer_default_params(void);

/* Load a .pq model. Returns NULL on failure; *err (if not NULL) receives the error code. */
pinfer_ctx *pinfer_load(const char *path, const pinfer_params *params, int *err);
void pinfer_free(pinfer_ctx *ctx);

/* Forget all tokens (KV cache, recurrent and conv states). */
void pinfer_reset(pinfer_ctx *ctx);

/* Feed n tokens in order. If logits is not NULL it receives n_vocab floats for the LAST token. */
int pinfer_eval(pinfer_ctx *ctx, const int32_t *tokens, int n, float *logits);

/* ---- Speculative greedy decoding with the model's MTP head (additive, engine/SPEC.md) ----
 * The MTP head drafts k tokens by chaining itself; the main model checks all of them in ONE
 * batched pass (weights streamed once) and keeps the longest correct prefix plus one token of
 * its own. Output is identical to plain greedy decoding; only the speed changes. */
#define PINFER_SPEC_MAX_K 8

typedef struct {
    int k;           /* tokens drafted per verification pass, 0..PINFER_SPEC_MAX_K; 0 = plain */
    int draft_vocab; /* drafts are the argmax over token ids [0, draft_vocab) only, which makes
                        each draft step cheaper (the tied head is ~1/3 of the weights);
                        <= 0 = full vocabulary. Never affects the output, only acceptance. */
    int adaptive;    /* 1: k is an upper bound; the engine measures acceptance and the costs
                        of its passes and uses the k (down to plain decoding) with the best
                        expected tokens per second, probing again as the text changes, so
                        speculation never runs slower than plain decoding for long
                        (engine/CONTEXT.md). 0 (added in 0.1; zero-initialised): fixed k. */
} pinfer_spec_params;

typedef struct {
    int64_t rounds;   /* verification passes */
    int64_t drafted;  /* draft tokens proposed */
    int64_t accepted; /* draft tokens accepted (tokens per pass = 1 + accepted / rounds) */
} pinfer_spec_stats;

pinfer_spec_params pinfer_default_spec_params(void);  /* k = 2, draft_vocab = 32768 */

/* 1 if the loaded model has an MTP head (needed for k > 0). */
int pinfer_has_mtp(const pinfer_ctx *ctx);

/* Reset ctx, feed the prompt, then greedily generate up to n_max tokens into out, stopping
 * after EOS (EOS is included). sp == NULL or sp->k == 0 is plain greedy decoding. stats (may
 * be NULL) is accumulated, not reset. Returns the number of tokens written or a negative
 * PINFER_E* code (PINFER_E_FORMAT if k > 0 and the model has no MTP head). */
int pinfer_generate_greedy(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt,
                           int32_t *out, int n_max, const pinfer_spec_params *sp,
                           pinfer_spec_stats *stats);

/* Options for pinfer_generate_greedy_ex. Zero-initialise unused fields. */
typedef struct {
    const int32_t *stop_ids; /* extra stop tokens (e.g. <|im_end|>); EOS always stops */
    int n_stop;
    /* Called for every generated token, in order, as soon as it is final (with k > 0 tokens
     * arrive in bursts of up to k+1). Return nonzero to stop after this token. May be NULL. */
    int (*on_token)(int32_t tok, void *user);
    void *user;
} pinfer_gen_opts;

/* pinfer_generate_greedy with stop tokens, a per-token callback, and no lost tokens: returns
 * the number of tokens written to out (>= 0, the stop token included) and sets *status (may
 * be NULL) to PINFER_OK when it ended normally (n_max, EOS, stop token or callback) or to the
 * error that ended it early, e.g. PINFER_E_CTX_FULL when the context filled up — out[0..ret)
 * is valid either way. Returns a negative PINFER_E* code (and the same *status) only when no
 * token could be generated (bad arguments, prompt does not fit, k > 0 without an MTP head). */
int pinfer_generate_greedy_ex(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt,
                              int32_t *out, int n_max, const pinfer_spec_params *sp,
                              const pinfer_gen_opts *opts, pinfer_spec_stats *stats,
                              int *status);

/* ---- Conversation state (engine/CONTEXT.md) ----------------------------------------
 * A snapshot of everything the context holds for the tokens consumed so far: recurrent and
 * conv states, the KV rows, the MTP head's rows. Restore it and feed more tokens to continue
 * bit-identically, e.g. a system prompt or document prefilled once and reused by many
 * conversations. The blob is tied to the model file, kv_dtype and engine version. */
size_t pinfer_state_size(const pinfer_ctx *ctx);   /* bytes a save needs right now */
/* Returns PINFER_OK, or PINFER_E_ARG when cap < pinfer_state_size(ctx). */
int pinfer_state_save(const pinfer_ctx *ctx, void *buf, size_t cap);
/* Replaces the context's state. PINFER_E_FORMAT: not a state blob of this model / kv_dtype;
 * PINFER_E_CTX_FULL: more tokens than max_ctx. The context is unchanged on failure. */
int pinfer_state_load(pinfer_ctx *ctx, const void *buf, size_t len);

/* pinfer_generate_greedy_ex without the reset: the prompt (n_prompt >= 1) is appended to the
 * tokens already in context (after pinfer_eval, pinfer_state_load or an earlier generation,
 * whose last returned token is not yet in context and should lead the prompt) and generation
 * continues from there. With k > 0 the MTP head is run over the new tokens; positions that
 * were evaluated with pinfer_eval have no MTP rows (they are zeroed), which lowers draft
 * acceptance for that context but never changes the output. */
int pinfer_generate_greedy_cont(pinfer_ctx *ctx, const int32_t *prompt, int n_prompt,
                                int32_t *out, int n_max, const pinfer_spec_params *sp,
                                const pinfer_gen_opts *opts, pinfer_spec_stats *stats,
                                int *status);

int pinfer_n_vocab(const pinfer_ctx *ctx);
int pinfer_n_past(const pinfer_ctx *ctx);   /* tokens consumed since load/reset */
int pinfer_eos_id(const pinfer_ctx *ctx);
int pinfer_argmax(const float *logits, int n);
const char *pinfer_version(void);

#ifdef __cplusplus
}
#endif
#endif
