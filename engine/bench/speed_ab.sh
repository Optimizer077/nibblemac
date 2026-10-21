#!/bin/bash
# Prompt64 speed A/B, interleaved: plain and --spec 3 --draft-vocab 32768, ROUNDS rounds of
# (reference, candidate), one RESULT line per run (engine/HARDENING.md H4). Push it to the Pi,
# then from ~/pinfer:  bash engine/speed_ab.sh NEW [OLD] [ROUNDS]
NEW=${1:?candidate binary}
OLD=${2:-./engine/pinfer_run_fast}
R=${3:-2}
M=models/qwen35-0.8b-q4k-headq4k.pq
for r in $(seq 1 $R); do
  for b in $OLD $NEW; do
    for S in "" "--spec 3 --draft-vocab 32768"; do
      $b -m $M -t 4 -n 64 --ids-file data/prompt64.ids --bench $S 2>/dev/null \
        | sed -n "s|^RESULT {|RESULT {\"ab_bin\":\"$b\",\"ab_round\":$r,\"ab_mode\":\"${S:-plain}\",|p"
    done
  done
done
