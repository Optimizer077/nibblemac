#!/usr/bin/env bash
# PC (WSL x86) perplexity for all Qwen3.5-0.8B GGUFs incl. bf16, same settings as the Pi.
# Usage (WSL, project root): bash bench/quality/ppl_pc.sh [chunks=64] [threads=16]
# Binary: llama.cpp v0.7.0 @620d2225, build-host (GGML_NATIVE=ON), target llama-perplexity.
# Output: RESULT lines on stdout -> bench/quality/results_pc_c<chunks>.jsonl; logs in bench/quality/logs/
set -euo pipefail
cd "$(dirname "$0")/../.."
CHUNKS="${1:-64}"; THREADS="${2:-16}"
B=/root/src/llama.cpp-v0.7.0/build-host/bin/llama-perplexity
OUT=bench/quality/results_pc_c${CHUNKS}.jsonl
: > "$OUT"
for Q in bf16 Q8_0 Q4_K_M Q4_0; do
    bash bench/quality/ppl.sh "$B" "models/gguf/Qwen3.5-0.8B-$Q.gguf" bench/quality/wikitext-2-raw-test.txt \
        "$CHUNKS" "$THREADS" pc-x86 bench/quality/logs | tee -a "$OUT"
done
