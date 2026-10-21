/* .pq loader: mmap + validation + weight views. See engine/FORMAT.md.
 *
 * Validation is strict on purpose: a truncated copy on the Pi or a converter/engine version
 * skew must fail at load time with PINFER_E_FORMAT, not as garbage logits later. */
#define _GNU_SOURCE
#include "model.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "pinfer.h"
#include "pq_quant.h"

#define FAIL(...) do { fprintf(stderr, "pinfer: " __VA_ARGS__); fputc('\n', stderr); \
                       return PINFER_E_FORMAT; } while (0)

const pq_tensor_entry *pf_model_find(const pf_model *m, const char *name) {
    for (uint32_t i = 0; i < m->hdr->n_tensors; i++)
        if (strcmp(m->dir[i].name, name) == 0) return &m->dir[i];
    return NULL;
}

/* Bytes the entry's dtype and shape take: F32 / BF16 elements, or the packed size of a
 * quantized [rows, cols] matrix (FORMAT.md "Quantized dtypes": rows % 4 == 0 and
 * cols % 256 == 0). The product is built factor by factor against `limit` (the file size),
 * so a crafted shape cannot wrap the uint64 count. */
static int entry_bytes(const pq_tensor_entry *e, uint64_t limit, uint64_t *bytes) {
    const uint64_t es = e->dtype == PQ_F32 ? 4 : e->dtype == PQ_BF16 ? 2 : 0;
    if (es) {
        uint64_t n = es;
        for (uint32_t d = 0; d < e->ndim; d++) {
            if (!e->shape[d]) FAIL("%s: zero dimension", e->name);
            if (n > limit / e->shape[d]) FAIL("%s: shape larger than the file", e->name);
            n *= e->shape[d];
        }
        *bytes = n;
        return PINFER_OK;
    }
    const uint64_t sb = pq_sb_bytes(e->dtype);
    if (!sb) FAIL("%s: unknown dtype %u", e->name, e->dtype);
    const uint64_t rows = e->shape[0], cols = e->shape[1];
    if (e->ndim != 2 || !rows || !cols || rows % PQ_TILE_ROWS || cols % PQ_SB_COLS)
        FAIL("%s: quantized dtype %u needs a [rows %% %d, cols %% %d] matrix", e->name,
             e->dtype, PQ_TILE_ROWS, PQ_SB_COLS);
    const uint64_t tiles = rows / PQ_TILE_ROWS * (cols / PQ_SB_COLS);   /* < 2^54 */
    if (tiles > limit / sb) FAIL("%s: shape larger than the file", e->name);
    *bytes = tiles * sb;
    return PINFER_OK;
}

/* Every entry must lie inside the file, be 64-byte aligned, have a known dtype and exactly
 * the size its dtype and shape imply (quantized ones included: a short nbytes would let the
 * kernels read past the tensor). */
static int check_directory(pf_model *m) {
    const pq_header *h = m->hdr;
    uint64_t size = m->map_bytes;
    if (h->dir_offset < sizeof(pq_header) || h->dir_offset > size ||
        h->dir_offset % sizeof(uint64_t) ||
        (uint64_t)h->n_tensors * sizeof(pq_tensor_entry) > size - h->dir_offset)
        FAIL("tensor directory out of bounds or misaligned");
    if (h->data_offset > size) FAIL("data_offset out of bounds");
    m->dir = (const pq_tensor_entry *)((const char *)m->map + h->dir_offset);
    for (uint32_t i = 0; i < h->n_tensors; i++) {
        const pq_tensor_entry *e = &m->dir[i];
        if (memchr(e->name, 0, PQ_NAME_MAX) == NULL) FAIL("tensor %u: name not terminated", i);
        if (e->ndim < 1 || e->ndim > 4) FAIL("%s: bad ndim %u", e->name, e->ndim);
        if (e->offset % PQ_TENSOR_ALIGN || e->offset < h->data_offset ||
            e->offset > size || e->nbytes > size - e->offset)
            FAIL("%s: data out of bounds or misaligned", e->name);
        uint64_t want = 0;
        if (entry_bytes(e, size, &want) != PINFER_OK) return PINFER_E_FORMAT;
        if (want != e->nbytes)
            FAIL("%s: nbytes %llu != %llu for dtype %u and its shape", e->name,
                 (unsigned long long)e->nbytes, (unsigned long long)want, e->dtype);
    }
    return PINFER_OK;
}

int pf_model_vec(const pf_model *m, const char *name, uint32_t n, const float **out) {
    const pq_tensor_entry *e = pf_model_find(m, name);
    if (!e) FAIL("missing tensor %s", name);
    if (e->dtype != PQ_F32 || e->ndim != 1 || e->shape[0] != n)
        FAIL("%s: expected F32 [%u]", name, n);
    *out = (const float *)((const char *)m->map + e->offset);
    return PINFER_OK;
}

int pf_model_mat(const pf_model *m, const char *name, uint32_t rows, uint32_t cols,
                 pf_mat *out) {
    const pq_tensor_entry *e = pf_model_find(m, name);
    if (!e) FAIL("missing tensor %s", name);
    if (e->ndim != 2 || e->shape[0] != rows || e->shape[1] != cols)
        FAIL("%s: expected shape [%u, %u]", name, rows, cols);
    if (!pf_dtype_supported(e->dtype)) FAIL("%s: dtype %u not supported by this build",
                                            name, e->dtype);
    out->dtype = e->dtype;
    out->rows = (int)rows;
    out->cols = (int)cols;
    out->data = (const char *)m->map + e->offset;
    return PINFER_OK;
}

/* Header fields the forward pass relies on; reject anything it was not written for. The
 * upper bounds (pq_format.h PQ_MAX_*) keep every product of header fields far from wrapping;
 * engine-specific limits (lin_v_dim, head_dim) are checked in pinfer_load. */
static int check_header(const pq_header *h) {
    if (memcmp(h->magic, PQ_MAGIC, 8) != 0) FAIL("bad magic");
    if (h->version != PQ_VERSION) FAIL("unsupported version %u", h->version);
    if (h->arch != PQ_ARCH_QWEN35) FAIL("unsupported arch %u", h->arch);
    if (!h->hidden || h->hidden > PQ_MAX_DIM || !h->ffn || h->ffn > PQ_MAX_DIM ||
        !h->vocab || h->vocab > PQ_MAX_VOCAB || !h->n_layers || h->n_layers > PQ_MAX_LAYERS)
        FAIL("bad model dimensions");
    if (h->eos_id >= h->vocab) FAIL("eos_id %u outside the vocabulary", h->eos_id);
    if (!h->n_heads || h->n_heads > PQ_MAX_HEADS || !h->n_kv_heads ||
        h->n_heads % h->n_kv_heads || !h->head_dim || h->head_dim > PQ_MAX_HEAD_DIM ||
        h->rot_dim % 2 || h->rot_dim > h->head_dim)
        FAIL("bad attention dimensions");
    if (!h->lin_k_heads || h->lin_k_heads > PQ_MAX_HEADS || !h->lin_v_heads ||
        h->lin_v_heads > PQ_MAX_HEADS || h->lin_v_heads % h->lin_k_heads || !h->lin_k_dim ||
        h->lin_k_dim > PQ_MAX_HEAD_DIM || !h->lin_v_dim || h->lin_v_dim > PQ_MAX_HEAD_DIM ||
        h->conv_k < 1 || h->conv_k > PQ_MAX_CONV_K)
        FAIL("bad linear-attention dimensions");
    if (!h->tied_embeddings) FAIL("untied output head not supported");
    if (!(h->rms_eps > 0.0f) || !isfinite(h->rms_eps) || !(h->rope_theta > 0.0f) ||
        !isfinite(h->rope_theta))
        FAIL("bad eps / rope_theta");
    return PINFER_OK;
}

static int build_layer(const pf_model *m, int i, pf_layer *L) {
    const pq_header *h = m->hdr;
    char nm[PQ_NAME_MAX + 32];
    int rc = 0;
#define NM(s) (snprintf(nm, sizeof nm, "layers.%d.%s", i, s), nm)
    uint32_t H = h->hidden;
    L->is_full = (h->full_attn_mask >> i) & 1;
    rc |= pf_model_vec(m, NM("input_layernorm.weight"), H, &L->in_norm);
    rc |= pf_model_vec(m, NM("post_attention_layernorm.weight"), H, &L->post_norm);
    rc |= pf_model_mat(m, NM("mlp.gate_proj.weight"), h->ffn, H, &L->gate);
    rc |= pf_model_mat(m, NM("mlp.up_proj.weight"), h->ffn, H, &L->up);
    rc |= pf_model_mat(m, NM("mlp.down_proj.weight"), H, h->ffn, &L->down);
    if (L->is_full) {
        uint32_t hd = h->head_dim;
        rc |= pf_model_mat(m, NM("self_attn.q_proj.weight"), h->n_heads * hd * 2, H, &L->q);
        rc |= pf_model_mat(m, NM("self_attn.k_proj.weight"), h->n_kv_heads * hd, H, &L->k);
        rc |= pf_model_mat(m, NM("self_attn.v_proj.weight"), h->n_kv_heads * hd, H, &L->v);
        rc |= pf_model_mat(m, NM("self_attn.o_proj.weight"), H, h->n_heads * hd, &L->o);
        rc |= pf_model_vec(m, NM("self_attn.q_norm.weight"), hd, &L->q_norm);
        rc |= pf_model_vec(m, NM("self_attn.k_norm.weight"), hd, &L->k_norm);
    } else {
        uint32_t kd = h->lin_k_heads * h->lin_k_dim, vd = h->lin_v_heads * h->lin_v_dim;
        uint32_t conv_dim = 2 * kd + vd;
        rc |= pf_model_mat(m, NM("linear_attn.in_proj_qkv.weight"), conv_dim, H, &L->qkv);
        rc |= pf_model_mat(m, NM("linear_attn.in_proj_z.weight"), vd, H, &L->z);
        rc |= pf_model_mat(m, NM("linear_attn.in_proj_b.weight"), h->lin_v_heads, H, &L->b);
        rc |= pf_model_mat(m, NM("linear_attn.in_proj_a.weight"), h->lin_v_heads, H, &L->a);
        rc |= pf_model_mat(m, NM("linear_attn.out_proj.weight"), H, vd, &L->out);
        rc |= pf_model_vec(m, NM("linear_attn.A_log"), h->lin_v_heads, &L->A_log);
        rc |= pf_model_vec(m, NM("linear_attn.dt_bias"), h->lin_v_heads, &L->dt_bias);
        rc |= pf_model_vec(m, NM("linear_attn.norm.weight"), h->lin_v_dim, &L->lin_norm);
        /* conv1d.weight is 2-D F32 [conv_dim, conv_k]: check it by hand. */
        const pq_tensor_entry *e = pf_model_find(m, NM("linear_attn.conv1d.weight"));
        if (!e || e->dtype != PQ_F32 || e->ndim != 2 || e->shape[0] != conv_dim ||
            e->shape[1] != h->conv_k)
            FAIL("layer %d: bad or missing conv1d.weight", i);
        L->conv_w = (const float *)((const char *)m->map + e->offset);
    }
#undef NM
    return rc ? PINFER_E_FORMAT : PINFER_OK;
}

/* Copy the file image into an anonymous mapping advised for transparent huge pages. A file
 * mapping only gets huge pages through khugepaged (slow, not guaranteed), an anonymous one
 * gets them at fault time when THP is "always" or "madvise". Costs one extra copy of the
 * model in RAM; returns NULL (keep the file mapping) if anything fails. */
static void *anon_hugepage_copy(const void *src, size_t bytes) {
    const size_t huge = 2u << 20;
    size_t len = (bytes + huge - 1) / huge * huge;
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
#ifdef MADV_HUGEPAGE
    if (madvise(p, len, MADV_HUGEPAGE) != 0) perror("madvise(MADV_HUGEPAGE)");
#endif
    memcpy(p, src, bytes);
    mprotect(p, len, PROT_READ);
    FILE *f = fopen("/proc/self/smaps_rollup", "r");  /* report how much actually went huge */
    char line[128];
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, "AnonHugePages:", 14)) fprintf(stderr, "pinfer: %s", line);
    if (f) fclose(f);
    return p;
}

int pf_model_load(pf_model *m, const char *path) {
    memset(m, 0, sizeof *m);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return PINFER_E_IO; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)PQ_HEADER_BYTES) {
        close(fd);
        fprintf(stderr, "pinfer: %s: too small for a .pq file\n", path);
        return PINFER_E_FORMAT;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  /* the mapping keeps the file referenced */
    if (map == MAP_FAILED) { perror("mmap"); return PINFER_E_IO; }
    m->map = map;
    m->map_bytes = (size_t)st.st_size;
    if (getenv("PINFER_HUGEPAGES")) {  /* experiment: weights in 2 MB (THP) pages, fewer TLB misses */
        void *copy = anon_hugepage_copy(map, m->map_bytes);
        if (copy) { munmap(map, m->map_bytes); m->map = map = copy; }
    }
    m->hdr = (const pq_header *)map;

    int rc = check_header(m->hdr);
    if (rc == PINFER_OK) rc = check_directory(m);
    if (rc == PINFER_OK) rc = pf_model_mat(m, "embed_tokens.weight", m->hdr->vocab, m->hdr->hidden,
                                      &m->embed);
    if (rc == PINFER_OK) rc = pf_model_vec(m, "norm.weight", m->hdr->hidden, &m->norm);
    if (rc == PINFER_OK) {
        m->layers = calloc(m->hdr->n_layers, sizeof(pf_layer));
        if (!m->layers) rc = PINFER_E_NOMEM;
    }
    for (uint32_t i = 0; rc == PINFER_OK && i < m->hdr->n_layers; i++)
        rc = build_layer(m, (int)i, &m->layers[i]);
    if (rc != PINFER_OK) pf_model_free(m);
    return rc;
}

void pf_model_free(pf_model *m) {
    if (m->map) munmap(m->map, m->map_bytes);
    free(m->layers);
    memset(m, 0, sizeof *m);
}
