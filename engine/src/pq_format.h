/* On-disk structures of the .pq model format, version 1. See engine/FORMAT.md. */
#ifndef PQ_FORMAT_H
#define PQ_FORMAT_H

#include <stdint.h>

#define PQ_MAGIC "PINFERQ"  /* 7 chars + NUL = 8 bytes */
#define PQ_VERSION 1u
#define PQ_HEADER_BYTES 4096u
#define PQ_TENSOR_ALIGN 64u
#define PQ_NAME_MAX 80

enum pq_dtype {
    PQ_F32 = 0,
    PQ_BF16 = 1,
    /* 2+ : quantized dtypes, see FORMAT.md "Quantized dtypes" */
};

enum pq_arch { PQ_ARCH_QWEN35 = 1 };

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t n_tensors;
    uint64_t dir_offset;
    uint64_t data_offset;
    uint32_t arch;
    uint32_t hidden, n_layers, vocab, ffn;
    uint32_t n_heads, n_kv_heads, head_dim, rot_dim;
    uint32_t lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_k;
    uint32_t full_attn_mask;
    uint32_t eos_id;
    uint32_t tied_embeddings;
    uint32_t has_mtp;
    float rms_eps;
    float rope_theta;
} pq_header;

typedef struct {
    char name[PQ_NAME_MAX];
    uint32_t dtype;
    uint32_t ndim;
    uint32_t shape[4];
    uint64_t offset;
    uint64_t nbytes;
    uint8_t reserved[8];
} pq_tensor_entry;

_Static_assert(sizeof(pq_tensor_entry) == 128, "pq_tensor_entry must be 128 bytes");

#endif
