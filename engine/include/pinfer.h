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

typedef struct {
    int n_threads; /* worker threads for matrix ops; <= 0 means 4 */
    int max_ctx;   /* max tokens kept in the KV cache; <= 0 means 4096 */
} pinfer_params;

pinfer_params pinfer_default_params(void);

/* Load a .pq model. Returns NULL on failure; *err (if not NULL) receives the error code. */
pinfer_ctx *pinfer_load(const char *path, const pinfer_params *params, int *err);
void pinfer_free(pinfer_ctx *ctx);

/* Forget all tokens (KV cache, recurrent and conv states). */
void pinfer_reset(pinfer_ctx *ctx);

/* Feed n tokens in order. If logits is not NULL it receives n_vocab floats for the LAST token. */
int pinfer_eval(pinfer_ctx *ctx, const int32_t *tokens, int n, float *logits);

int pinfer_n_vocab(const pinfer_ctx *ctx);
int pinfer_n_past(const pinfer_ctx *ctx);   /* tokens consumed since load/reset */
int pinfer_eos_id(const pinfer_ctx *ctx);
int pinfer_argmax(const float *logits, int n);
const char *pinfer_version(void);

#ifdef __cplusplus
}
#endif
#endif
