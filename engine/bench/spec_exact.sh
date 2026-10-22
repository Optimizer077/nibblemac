#!/bin/bash
# Speculative decoding must generate exactly the plain greedy tokens. For each model x prompt
# the md5 of the `generated:` line of --spec 0 is the reference; every other k must match.
# Runs on the Pi from ~/pinfer:  bash kbench/spec_exact.sh ./engine/pinfer_run_kern3
# One RESULT line per configuration (for harness/pi.py bench).
BIN=${1:-./engine/pinfer_run_fast}
N=${2:-40}
# A missing binary or a run without output must fail: two empty outputs have equal md5s.
if [ ! -x "$BIN" ]; then
  echo "RESULT {\"bench\":\"spec_exact\",\"bin\":\"$BIN\",\"match\":false,\"error\":\"binary not found\"}"
  exit 1
fi
for M in qwen35-0.8b-mix-headq4k qwen35-0.8b-q4k-headq4k; do
  for P in chat27 prompt64; do
    ref=""
    for S in "--spec 0" "--spec 2 --draft-vocab 32768" "--spec 3 --draft-vocab 32768" \
             "--spec 4 --draft-vocab 32768" "--spec 1"; do
      out=$($BIN -m models/$M.pq -t 4 -n $N --ids-file data/$P.ids $S 2>/dev/null | grep '^generated:')
      if [ -n "$out" ]; then g=$(printf '%s\n' "$out" | md5sum | cut -c1-8); else g=none; fi
      [ -z "$ref" ] && ref=$g
      if [ "$g" != none ] && [ "$g" = "$ref" ]; then m=true; else m=false; fi
      echo "RESULT {\"bench\":\"spec_exact\",\"bin\":\"$BIN\",\"model\":\"$M\",\"prompt\":\"$P\",\"n\":$N,\"spec\":\"$S\",\"md5\":\"$g\",\"match\":$m}"
    done
  done
done
