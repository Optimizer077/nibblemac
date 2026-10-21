#!/usr/bin/env bash
# Runs on the Pi from ~/pinfer/baselines: ORT GenAI thread sweep, one process (and one
# `RESULT {json}` line) per thread count, for harness/pi.py bench.
# Usage: bash bench/ort/sweep.sh <export dir> <quant label> [threads=1,2,3,4] [reps=3] [extra args]
set -euo pipefail
MODEL="$1"; QUANT="$2"; THREADS="${3:-1,2,3,4}"; REPS="${4:-3}"
# `shift 4` with fewer than 4 arguments shifts nothing, which would pass MODEL and QUANT on
# to run_bench.py as extra arguments: shift only what is there.
shift "$(( $# < 4 ? $# : 4 ))"
for T in ${THREADS//,/ }; do
    ../venvs/ort/bin/python bench/ort/run_bench.py --model "$MODEL" --quant "$QUANT" \
        --threads "$T" --reps "$REPS" "$@"
done
