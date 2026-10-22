#!/bin/bash
# Bit identity of a new build against the previous one on Qwen3.5-4B (Cerno 4B, Q4_0 body, Q8
# head; KERNELS.md Part 4). From ~/pinfer:
#   bash engine/cerno_check.sh ./engine/pinfer_run_new ./engine/pinfer_run_old ids-file...
# Per prompt: every row's logits (--dump-logits, the full forward pass) of both builds, and the
# new build's prefill-only last-token logits (--dump-last: the "tail" path of forward.c), which
# must equal the last row of the old build's logits; plus the 2 generated tokens.
NEW=$1
OLD=$2
shift 2
M=${M:-models/cerno-4b-q4_0.pq}
V=248320
mkdir -p tmp/cc
for f in "$@"; do
  p=$(basename "$f" .ids)
  for v in new old; do
    [ $v = new ] && bin=$NEW || bin=$OLD
    $bin -m $M -t 4 -n 2 --ids-file "$f" --dump-logits tmp/cc/all_$v.bin > tmp/cc/out_$v.txt
    grep '^load' tmp/cc/out_$v.txt | sed "s/^/  $p $v: /"
  done
  $NEW -m $M -t 4 -n 2 --ids-file "$f" --dump-last tmp/cc/last_new.bin > tmp/cc/out_tail.txt
  grep '^load' tmp/cc/out_tail.txt | sed "s/^/  $p tail: /"
  tail -c $((V * 4)) tmp/cc/all_old.bin > tmp/cc/last_old.bin
  md5sum tmp/cc/*.bin | sed "s/^/  $p /"
  a=false; b=false; g=false
  cmp -s tmp/cc/all_new.bin tmp/cc/all_old.bin && a=true
  cmp -s tmp/cc/last_new.bin tmp/cc/last_old.bin && b=true
  gn=$(grep '^generated' tmp/cc/out_new.txt); go=$(grep '^generated' tmp/cc/out_old.txt)
  gt=$(grep '^generated' tmp/cc/out_tail.txt)
  [ -n "$gn" ] && [ "$gn" = "$go" ] && [ "$gt" = "$go" ] && g=true
  echo "RESULT {\"bench\":\"cerno_check\",\"prompt\":\"$p\",\"all_logits_identical\":$a,\"tail_last_identical\":$b,\"generated_identical\":$g}"
  rm -f tmp/cc/*
done
rmdir tmp/cc 2>/dev/null
