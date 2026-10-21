# `.pq` model file format, version 1

One file holds one model: a fixed header, a tensor directory, then tensor data. Everything is
little-endian. Written by `tools/convert.py`, read by `engine/src/model.c` via `mmap`.

## Layout
```
offset 0      pq_header        (struct below, padded with zeros to 4096 bytes)
dir_offset    pq_tensor_entry  × n_tensors (128 bytes each)
data_offset   tensor data; data_offset is a multiple of 4096, every tensor starts at a
              multiple of 64 bytes (cache line) relative to the file start
```

## Header (`pq_header`, see `engine/src/pq_format.h`)
| field | type | value for Qwen3.5-0.8B |
|---|---|---|
| magic | char[8] | `"PINFERQ\0"` |
| version | u32 | 1 |
| n_tensors | u32 | |
| dir_offset, data_offset | u64 | |
| arch | u32 | 1 = qwen3.5 hybrid |
| hidden | u32 | 1024 |
| n_layers | u32 | 24 |
| vocab | u32 | 248320 |
| ffn | u32 | 3584 |
| n_heads, n_kv_heads, head_dim, rot_dim | u32 | 8, 2, 256, 64 |
| lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_k | u32 | 16, 16, 128, 128, 4 |
| full_attn_mask | u32 | bit i set ⇒ layer i is full attention (layers 3,7,…,23) |
| eos_id | u32 | 248044 |
| tied_embeddings | u32 | 1 |
| has_mtp | u32 | 1 if `mtp.*` tensors are present |
| rms_eps, rope_theta | f32 | 1e-6, 1e7 |

## Tensor directory entry (`pq_tensor_entry`, 128 bytes)
`char name[80]` (NUL-terminated) · `u32 dtype` · `u32 ndim` · `u32 shape[4]` · `u64 offset` ·
`u64 nbytes` · 8 bytes reserved. Matrices use PyTorch order: `shape[0]` = output rows,
`shape[1]` = input columns, row-major, so `y = W x` reads each row contiguously.

## Validation (what `pinfer_load` refuses with `PINFER_E_FORMAT`)
The loader (`engine/src/model.c`, `pinfer.c`) checks the header and the whole directory against
the file size before it reads any weight, so a truncated or crafted file fails at load time and
never inside a kernel (tests: `tests/test_malformed.py`, `engine/HARDENING.md`).
- Header: magic, version 1, arch 1; every dimension nonzero and at most the limits in
  `engine/src/pq_format.h` (hidden, ffn ≤ 65536; vocab ≤ 2²¹; n_layers ≤ 32; head counts ≤ 256;
  head dims ≤ 1024; conv_k ≤ 8), so no product of header fields can wrap; `n_heads % n_kv_heads`,
  `lin_v_heads % lin_k_heads`, `rot_dim` even and ≤ `head_dim`, `eos_id < vocab`, `rms_eps` and
  `rope_theta` positive and finite, tied embeddings.
- Engine limits (`pinfer_load`): `lin_v_dim` ≤ 512 and a multiple of 32 (the DeltaNet kernel
  works in 32-column chunks), `head_dim` ≤ 512 and a multiple of 4.
- Directory: `dir_offset` ≥ the header size and 8-byte aligned, all entries inside the file.
- Every entry: NUL-terminated name, `ndim` 1–4, `offset` 64-byte aligned, ≥ `data_offset`,
  `offset + nbytes` ≤ file size, a known dtype, and `nbytes` exactly the size its dtype and shape
  imply: elements × 4 / × 2 for F32 / BF16 (no zero dimension; the product is checked factor by
  factor against the file size, so it cannot wrap), and for a quantized dtype a 2-D shape that
  obeys the shape rule below with `nbytes = rows/4 · cols/256 · SB`. This covers tensors the
  engine maps only later (the `mtp.*` head).
- Each tensor the forward pass uses: the exact dtype class (F32 vectors) and shape it expects,
  and a dtype the build supports (the scalar `MATVEC=ref` build: F32 and BF16 only).

## dtypes
| id | name | layout |
|---|---|---|
| 0 | F32 | IEEE float32 |
| 1 | BF16 | upper 16 bits of float32 (as stored by the checkpoint) |
| 2+ | quantized | defined in the "Quantized dtypes" section (owned by the kernels work) |

## Tensor names and conventions
HF prefix `model.language_model.` is dropped; vision tensors are not stored. `mtp.*` keeps its name.
- `embed_tokens.weight` BF16 [vocab, hidden] — also the output head (tied).
- `norm.weight`, `layers.N.input_layernorm.weight`, `layers.N.post_attention_layernorm.weight`,
  `layers.N.self_attn.{q,k}_norm.weight`, `mtp.*norm*.weight`: F32, **stored as the effective
  multiplier `1 + w`** (Qwen3.5 zero-centered RMSNorm). Engine computes `x/rms(x) * stored`.
- `layers.N.linear_attn.norm.weight`: F32 **as-is** (gated RMSNorm uses plain `w`).
- `layers.N.linear_attn.conv1d.weight`: F32 [6144, 4] (depthwise, causal, kernel 4).
- `layers.N.linear_attn.A_log`, `dt_bias`: F32 [16]. Engine uses `g = -exp(A_log)·softplus(a + dt_bias)`.
- Every other 2-D weight: BF16 (v1 converter) or a quantized dtype.

## Forward pass (reference semantics, from transformers 5.20.2 `modeling_qwen3_5.py`)
Per token, `h = embed[token]`, then for each layer: `h += mix(norm_in(h))`, `h += mlp(norm_post(h))`;
finally `logits = embed · norm(h)`.
- **mlp**: `down(silu(gate(x)) * up(x))`.
- **linear-attention mix** (layers with mask bit 0): `qkv = in_proj_qkv x` (6144 = q 2048 | k 2048 |
  v 2048) → causal depthwise conv over the last 4 inputs (state = previous 3 inputs) → SiLU →
  split into 16 heads × 128 for q, k, v. `q, k` L2-normalized per head (eps 1e-6), `q *= 1/√128`.
  `beta = sigmoid(in_proj_b x)`, `g = -exp(A_log)·softplus(in_proj_a x + dt_bias)` (per head).
  State `S` per head [128 k × 128 v], zero-initialised: `S *= exp(g)`; `kv = Sᵀk`;
  `δ = (v − kv)·beta`; `S += k ⊗ δ`; `o = Sᵀq`. Then per head `o = rmsnorm(o)·w_norm·silu(z)` with
  `z = in_proj_z x` (16 × 128); output `out_proj o`.
- **full-attention mix** (mask bit 1): `q_proj x` → per head [query 256 | gate 256] (8 heads).
  `q = q_norm(query)`, `k = k_norm(k_proj x)` (2 KV heads), `v = v_proj x`. RoPE on the first
  64 dims of each head, rotate-half style within those 64 (pairs i, i+32),
  `inv_freq[i] = θ^(−2i/64)`, position = token index. Causal softmax attention with GQA (4 query
  heads per KV head), scale 1/√256. `o *= sigmoid(gate)`; output `o_proj o`.

## Quantized dtypes
Owned by the kernels work. Constants: `engine/src/pq_quant.h`; reference quantizers and
packers: `tools/quant.py` (`quantize_pack(dtype, w)`); kernels: `engine/src/kern_quant.c`.
Rationale and measurements: `engine/bench/KERNELS.md`.

**Shape rule.** `rows % 4 == 0` and `cols % 256 == 0` (true for every Qwen3.5 matrix).
`nbytes = rows/4 · cols/256 · SB` with `SB` the superblock size below. The loader enforces
both (since 0.2.0; before, a short `nbytes` let the kernels read past the tensor).

**Tiling (all quantized dtypes).** Rows are taken 4 at a time: tile `t` = rows `4t..4t+3`.
Tiles are stored in order; each tile is `cols/256` superblocks in column order; superblock
`sb` covers columns `256·sb .. 256·sb+255` of the tile's 4 rows. A superblock is one 64-byte
**scale line** followed by the **codes** of 8 blocks of 32 columns (block `k` = columns
`256·sb + 32k ..+31`), block after block, and inside a block row `i = 0..3` after row. SB is a
multiple of 64, so every scale line starts on a cache line and each thread streams one
contiguous range. "Row `i`" below means row `4t+i`; `[k][i]` arrays are stored at index `4k+i`.

| id | name | bits/weight | SB bytes | scale line (64 B) | codes of block k, row i | weight value |
|---|---|---|---|---|---|---|
| 2 | Q8_G32 | 8.5 | 1088 | fp16 `d[k][i]` (32 × 2 B) | `int8 q[32]` at `64 + 128k + 32i` | `d · q`, `q ∈ [-127, 127]` |
| 3 | Q4_G32 | 4.5 | 576 | fp16 `d[k][i]` | 16 B at `64 + 64k + 16i`, nibbles (below) | `d · (u − 8)` |
| 4 | Q4K_G32 | 4.5 | 576 | bytes 0–7 fp16 `D[i]`; 8–15 fp16 `M[i]`; 16–39 6-bit `s[k][i]`; 40–63 6-bit `m[k][i]` | as Q4_G32 | `D[i]·s · u − M[i]·m` |

- **Nibbles** (Q4_G32, Q4K_G32): byte `j` (0..15) of a block row holds column `j` in its low
  nibble and column `j + 16` in its high nibble; `u ∈ [0, 15]`.
- **6-bit arrays** (Q4K_G32): 32 values `v[0..31]` (index `4k+i`) in 24 bytes: byte `j`
  (0..15) = `(v[j] & 15) | (v[j+16] & 15) << 4`; byte `16 + j` (0..7) =
  `v[j]>>4 | (v[j+8]>>4)<<2 | (v[j+16]>>4)<<4 | (v[j+24]>>4)<<6`. `s, m ∈ [0, 63]`;
  `D, M ≥ 0` are per row and per 256 columns (k-quant style: a 32-column asymmetric group
  with the cost of 0.5 bit/weight of metadata).
- fp16 is IEEE binary16. Quantizers in `tools/quant.py`: Q8 absmax; Q4 signed-extreme scale
  with a small scale search + least-squares refit; Q4K per-group least-squares (scale, min)
  then 6-bit rounding under per-row maxima, with refits.
- Not part of the file: at run time the kernels quantize activations to int8 per 32 columns
  (scale = absmax/127), so a quantized matmul result also carries that rounding.
