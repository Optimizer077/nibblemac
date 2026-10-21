#!/usr/bin/env bash
# PC-side driver (Git Bash, project root): run baseline sweeps one after another through
# harness/pi.py bench (each holds the Pi-wide lock; results land in bench/results/).
# Usage: bash bench/common/run_baselines.sh <group> ...   groups: mnn ort-int4 ort-int8 ort-acc1 torch
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
B() { echo "=== $1  $(date +%T)"; python harness/pi.py bench "$1" -- "cd baselines && $2" | grep -E "^saved|^RESULT" | cut -c1-400; }

for g in "$@"; do
    case "$g" in
        mnn)      B mnn-qwen35-0.8b-int4 "bash bench/mnn/sweep.sh 1,2,3,4 3" ;;
        ort-int4) B ort-qwen35-0.8b-int4-acc4 "bash bench/ort/sweep.sh models/ort/int4-acc4 int4-rtn-sym-b32-acc4 1,2,3,4 3" ;;
        ort-int8) B ort-qwen35-0.8b-int8-acc4 "bash bench/ort/sweep.sh models/ort/int8-acc4 int8-rtn-sym-b32-acc4-embfp32 1,2,3,4 3" ;;
        ort-acc1) B ort-qwen35-0.8b-int4-acc1 "bash bench/ort/sweep.sh models/ort/int4-acc1 int4-rtn-sym-b32-acc1 1,4 3"
                  B ort-qwen35-0.8b-int8-acc1 "bash bench/ort/sweep.sh models/ort/int8-acc1 int8-rtn-sym-b32-acc1-embfp32 1,4 3" ;;
        torch)    B torch-qwen35-0.8b-fp32 "bash bench/torch/sweep.sh 1,2,3,4 3" ;;
        *) echo "unknown group $g" ;;
    esac
done
