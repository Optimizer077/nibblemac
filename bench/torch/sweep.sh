#!/usr/bin/env bash
# Runs on the Pi from ~/pinfer/baselines: PyTorch thread sweep, one process (and one
# `RESULT {json}` line) per thread count, for harness/pi.py bench.
# Usage: bash bench/torch/sweep.sh [threads=1,2,3,4] [reps=3] [dtype=fp32]
set -uo pipefail
THREADS="${1:-1,2,3,4}"; REPS="${2:-3}"; DTYPE="${3:-fp32}"
for T in ${THREADS//,/ }; do
    ../venvs/torch/bin/python bench/torch/run_bench.py --model models/Qwen3.5-0.8B \
        --threads "$T" --reps "$REPS" --dtype "$DTYPE" \
        || echo "torch run failed: threads=$T dtype=$DTYPE exit=$?" >&2
done
