#!/bin/bash
# Speculative decoding must generate exactly the plain greedy tokens on the 2B model too
# (engine/bench/spec_exact.sh does the same for the two 0.8B files). Per prompt the md5 of the
# `generated:` line of --spec 0 is the reference; every other configuration must match.
# From ~/pinfer:  bash tmp/m2b/spec_exact_2b.sh ./engine/pinfer_run_2b1 [N] [MODEL]
BIN=${1:-./engine/pinfer_run_2b1}
N=${2:-40}
M=${3:-models/qwen35-2b-q4k-headq4k.pq}
if [ ! -x "$BIN" ] || [ ! -f "$M" ]; then
  echo "RESULT {\"bench\":\"spec_exact_2b\",\"bin\":\"$BIN\",\"match\":false,\"error\":\"binary or model not found\"}"
  exit 1
fi
for P in chat27 prompt64; do
  ref=""
  for S in "--spec 0" "--spec 2 --draft-vocab 32768" "--spec 3 --draft-vocab 32768" \
           "--spec 4 --draft-vocab 32768" "--spec 1"; do
    out=$($BIN -m "$M" -t 4 -n "$N" --ids-file "data/$P.ids" $S 2>/dev/null | grep '^generated:')
    if [ -n "$out" ]; then g=$(printf '%s\n' "$out" | md5sum | cut -c1-8); else g=none; fi
    [ -z "$ref" ] && ref=$g
    if [ "$g" != none ] && [ "$g" = "$ref" ]; then m=true; else m=false; fi
    echo "RESULT {\"bench\":\"spec_exact_2b\",\"bin\":\"$BIN\",\"model\":\"$M\",\"prompt\":\"$P\",\"n\":$N,\"spec\":\"$S\",\"md5\":\"$g\",\"match\":$m}"
  done
done
