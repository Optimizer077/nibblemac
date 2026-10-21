#!/usr/bin/env bash
# Runs on the Pi from ~/pinfer/baselines: MNN thread sweep, one process (and one `RESULT {json}`
# line) per thread count, for harness/pi.py bench. Our wrapper uses the shared prompt ids.
# Usage: bash bench/mnn/sweep.sh [threads=1,2,3,4] [reps=3] [precision=low] [memory=low]
# MNN prints "Can't open visual.mnn" on load: expected, the vision tower is not pushed (text only).
set -euo pipefail
THREADS="${1:-1,2,3,4}"; REPS="${2:-3}"; PREC="${3:-low}"; MEM="${4:-low}"
for T in ${THREADS//,/ }; do
    ./mnn/mnn_bench models/Qwen3.5-0.8B-MNN/config.json bench/common/prompt64.json \
        "$T" "$REPS" 64 "$PREC" "$MEM"
done
