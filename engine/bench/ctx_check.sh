#!/bin/bash
# Bit identity of the new attention path (engine/CONTEXT.md) against the previous binary on
# the Pi: hidden states + logits of prompt64 (batched and token by token) and the logits of
# the 512-token wikitext prompt, Q4K model. From ~/pinfer:
#   bash engine/ctx_check.sh ./engine/pinfer_run_ctx ./engine/pinfer_run_fast
NEW=${1:-./engine/pinfer_run_ctx}
OLD=${2:-./engine/pinfer_run_fast}
M=models/qwen35-0.8b-q4k-headq4k.pq
mkdir -p tmp/ctx
for v in new old; do
  [ $v = new ] && bin=$NEW || bin=$OLD
  $bin -m $M -t 4 -n 2 -b 64 --ids-file data/prompt64.ids --dump-hidden tmp/ctx/h64_$v.bin --dump-logits tmp/ctx/l64_$v.bin | grep '^load' | sed "s/^/  p64 b64 $v: /"
  $bin -m $M -t 4 -n 2 -b 1 --ids-file data/prompt64.ids --dump-hidden tmp/ctx/h1_$v.bin --dump-logits tmp/ctx/l1_$v.bin | grep '^load' | sed "s/^/  p64 b1 $v: /"
  $bin -m $M -t 4 -n 2 --ids-file data/ctx512.ids --dump-logits tmp/ctx/l512_$v.bin | grep '^load' | sed "s/^/  ctx512 $v: /"
done
md5sum tmp/ctx/*.bin
for f in h64 l64 h1 l1 l512; do
  if cmp -s tmp/ctx/${f}_new.bin tmp/ctx/${f}_old.bin; then echo "RESULT {\"bench\":\"ctx_check\",\"dump\":\"$f\",\"identical\":true}"; else echo "RESULT {\"bench\":\"ctx_check\",\"dump\":\"$f\",\"identical\":false}"; fi
done
rm -rf tmp/ctx
