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
