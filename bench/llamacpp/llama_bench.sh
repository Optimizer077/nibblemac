#!/usr/bin/env bash
# Runs on the Pi from ~/pinfer: llama-bench over a thread sweep, one `RESULT {json}` line per
# configuration for harness/pi.py bench. Usage: llama_bench.sh <model.gguf> [threads] [reps]
#   pp64 = prompt processing of 64 tokens, tg64 = generating 64 tokens (decode speed).
set -euo pipefail
MODEL="$1"
THREADS="${2:-1,2,3,4}"
REPS="${3:-3}"
./llama/llama-bench -m "$MODEL" -t "$THREADS" -p 64 -n 64 -r "$REPS" -o jsonl | sed 's/^/RESULT /'
