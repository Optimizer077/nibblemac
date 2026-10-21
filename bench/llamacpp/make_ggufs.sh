#!/usr/bin/env bash
# Convert a downloaded HF model to GGUF (bf16) and quantize it for the llama.cpp baseline.
# Run inside WSL Ubuntu after build.sh:  bash make_ggufs.sh Qwen3.5-0.8B [v0.7.0]
# Text model only (no vision projector). Outputs: models/gguf/<model>-<type>.gguf + SHA256SUMS.
set -euo pipefail

MODEL="$1"
TAG="${2:-v0.7.0}"
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
SRC=/root/src/llama.cpp-"$TAG"
PY=/root/venv-convert/bin/python
QUANT="$PROJ/build/host/llama/llama-quantize"
OUT="$PROJ/models/gguf"
mkdir -p "$OUT"

BASE="$OUT/$MODEL-bf16.gguf"
if [ ! -f "$BASE" ]; then
    "$PY" "$SRC/convert_hf_to_gguf.py" "$PROJ/models/$MODEL" --outtype bf16 --outfile "$BASE" \
        > "$OUT/$MODEL-convert.log" 2>&1
fi
for T in Q8_0 Q4_0 Q4_K_M; do
    F="$OUT/$MODEL-$T.gguf"
    [ -f "$F" ] || "$QUANT" "$BASE" "$F" "$T" > "$OUT/$MODEL-quantize-$T.log" 2>&1
done
(cd "$OUT" && sha256sum "$MODEL"-*.gguf > "$MODEL.SHA256SUMS" && ls -la "$MODEL"-*.gguf)
