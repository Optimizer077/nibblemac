#!/bin/bash
# Speculative decoding on chat-formatted prompts (bench/common/chat*.ids): plain, fixed k=3 and
# adaptive k<=3, Q4K model, 4 threads, N generated tokens (pinfer_run ignores EOS, so keep N
# below the typical answer length). Each run is preceded by a RESULT row naming prompt and
# config. From ~/pinfer:
#   bash engine/chat_spec.sh ./engine/pinfer_run_fast [N] [REPS] [CONFIGS]
# CONFIGS: comma-separated from plain,k3,adaptive,k2,k3full,k2full (full = draft over all 248320 ids;
# commas because harness/pi.py joins its arguments with spaces)
BIN=${1:-./engine/pinfer_run_fast}
N=${2:-96}
REPS=${3:-2}
CONFIGS=${4:-plain,k3,adaptive}
M=models/qwen35-0.8b-q4k-headq4k.pq
for r in $(seq 1 "$REPS"); do
  for p in chat27 chat_code_py chat_code_c chat_list chat_math chat_story; do
    for cfg in ${CONFIGS//,/ }; do
      case $cfg in
        plain) S="" ;;
        k3) S="--spec 3 --draft-vocab 32768" ;;
        adaptive) S="--spec 3 --draft-vocab 32768 --adaptive" ;;
        k2) S="--spec 2 --draft-vocab 32768" ;;
        k3full) S="--spec 3" ;;
        k2full) S="--spec 2" ;;
      esac
      echo "RESULT {\"bench\":\"chat_spec\",\"prompt\":\"$p\",\"cfg\":\"$cfg\",\"rep\":$r}"
      $BIN -m $M -t 4 -n "$N" --ids-file "data/$p.ids" --bench $S | grep '^RESULT'
    done
  done
done
