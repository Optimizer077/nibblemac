#!/usr/bin/env bash
# Export Qwen3.5-0.8B with the ONNX Runtime GenAI model builder (run in WSL as root, on the PC):
#   bash bench/ort/export.sh <variant>
# variants: int4-acc4 int4-acc1 int8-acc4 int8-acc1   (acc = MatMulNBits accuracy_level:
#   4 = activations quantized to int8 (CompInt8), 1 = fp32 compute)
# Venv /root/venv-ortgen: Python 3.13.14 (uv-managed), onnxruntime-genai 0.18.0,
# onnxruntime 1.31.0, torch 2.14.1+cpu, transformers 5.20.2 (see bench/BASELINES.md).
# Output: models/baselines/ort/<variant>/ (+ export.log)
set -euo pipefail
VARIANT="$1"
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/root/venv-ortgen/bin/python
OUT="$PROJ/models/baselines/ort/$VARIANT"
CACHE=/root/cache/ortgen
mkdir -p "$OUT" "$CACHE"

case "$VARIANT" in
    int4-acc4) PREC=int4; ACC=4 ;;
    int4-acc1) PREC=int4; ACC=1 ;;
    int8-acc4) PREC=int8; ACC=4 ;;
    int8-acc1) PREC=int8; ACC=1 ;;
    *) echo "unknown variant $VARIANT"; exit 2 ;;
esac

"$PY" -m onnxruntime_genai.models.builder -m "$PROJ/models/Qwen3.5-0.8B" -o "$OUT" \
    -p "$PREC" -e cpu -c "$CACHE" --extra_options accuracy_level="$ACC" hf_token=false \
    > "$OUT/export.log" 2>&1 || { tail -30 "$OUT/export.log"; exit 1; }
tail -5 "$OUT/export.log"
(cd "$OUT" && ls -la && sha256sum ./*.onnx* > SHA256SUMS)
