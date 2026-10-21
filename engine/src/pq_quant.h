/* Quantized dtypes of the .pq format (ids 2+). Layouts: engine/FORMAT.md "Quantized dtypes";
 * reference quantizers/packers: tools/quant.py.
 *
 * All quantized dtypes share one tiling, chosen for streaming NEON loads on the A72:
 *   - rows are taken 4 at a time (a "tile"); the kernel produces 4 outputs at once;
 *   - each tile is a sequence of 256-column "superblocks", stored back to back, so one
 *     thread reads one contiguous stream;
 *   - a superblock starts with one 64-byte line of scales (fp16, or for Q4K fp16 super-
 *     scales + 6-bit sub-scales), followed by the codes block by block (32 columns), row by
 *     row inside a block.
 * So every tensor needs rows % 4 == 0 and cols % 256 == 0 (true for all Qwen3.5 matrices).
 */
#ifndef PQ_QUANT_H
#define PQ_QUANT_H

#include <stddef.h>
#include <stdint.h>

enum pq_qdtype {
    PQ_Q8_G32 = 2,  /* int8 [-127,127] * fp16 scale per 32 cols            8.5 bits/weight */
    PQ_Q4_G32 = 3,  /* (u4 - 8) * fp16 scale per 32 cols                    4.5 bits/weight */
    PQ_Q4K_G32 = 4, /* u4 * (D * s6) - M * m6 per 32 cols; s6, m6 are 6-bit,
                       D, M fp16 per row and 256 cols (k-quant style)  4.5 bits/weight */
};

#define PQ_TILE_ROWS 4
#define PQ_SB_COLS 256
#define PQ_SB_SCALE_BYTES 64 /* one cache line of fp16 scales (and mins) per superblock */

/* Bytes of one superblock (4 rows x 256 cols) of a quantized dtype; 0 if not quantized. */
static inline size_t pq_sb_bytes(uint32_t dtype) {
    switch (dtype) {
    case PQ_Q8_G32: return PQ_SB_SCALE_BYTES + 4 * 256;      /* 1088 */
    case PQ_Q4_G32: return PQ_SB_SCALE_BYTES + 4 * 128;      /*  576 */
    case PQ_Q4K_G32: return PQ_SB_SCALE_BYTES + 4 * 128;     /*  576 */
    default: return 0;
    }
}

/* Packed size of a rows x cols matrix (0 if the shape or dtype is not supported). */
static inline size_t pq_quant_nbytes(uint32_t dtype, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || rows % PQ_TILE_ROWS || cols % PQ_SB_COLS) return 0;
    return (size_t)(rows / PQ_TILE_ROWS) * (size_t)(cols / PQ_SB_COLS) * pq_sb_bytes(dtype);
}

#endif
