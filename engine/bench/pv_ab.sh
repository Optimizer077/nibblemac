#!/bin/bash
# Decode-only A/B of attention variants from one restored state (engine/CONTEXT.md §3c). From ~/pinfer:
#   bash engine/pv_ab.sh <ctx> <state file> <kv dtype> bin0 bin1 ...
# Each binary decodes 32 tokens after the next wikitext token, 3 rounds alternating; prints the
# md5 of the generated tokens (must be equal for all: the variants are bit-identical) and RESULT.
C=$1; S=$2; D=$3; shift 3
M=models/qwen35-0.8b-q4k-headq4k.pq
for r in 1 2 3; do
  for B in "$@"; do
    out=$($B -m $M -t 4 -n 33 -c $((C+64)) --ids-file data/next$C.ids --spec 0 --kv-dtype $D --no-warmup --load-state $S --bench 2>/dev/null)
    g=$(echo "$out" | grep '^generated' | md5sum | cut -c1-8)
    echo "$out" | grep RESULT | sed "s/^RESULT {/RESULT {\"bin\":\"$(basename $B)\",\"round\":$r,\"gen_md5\":\"$g\",/"
  done
done
