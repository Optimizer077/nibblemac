#!/bin/bash
# fp16 KV on the Pi (NEON): prompt64 hidden states + logits, batched (-b 64, widened rows) vs
# token by token (-b 1, fused fp16 dots) of the new binary, and token by token of a reference
# binary without the fused path. All three must be identical. From ~/pinfer:
#   bash engine/f16_check.sh ./engine/pinfer_run_ctx7 ./engine/pinfer_run_ctx6
NEW=${1:-./engine/pinfer_run_ctx7}; REF=${2:-./engine/pinfer_run_ctx6}
M=models/qwen35-0.8b-q4k-headq4k.pq
mkdir -p tmp/f16
$NEW -m $M -t 4 -n 8 -b 64 --kv-dtype 1 --ids-file data/prompt64.ids --dump-hidden tmp/f16/h_b64.bin --dump-logits tmp/f16/l_b64.bin | grep '^generated' > tmp/f16/g_b64.txt
$NEW -m $M -t 4 -n 8 -b 1 --kv-dtype 1 --ids-file data/prompt64.ids --dump-hidden tmp/f16/h_b1.bin --dump-logits tmp/f16/l_b1.bin | grep '^generated' > tmp/f16/g_b1.txt
$REF -m $M -t 4 -n 8 -b 1 --kv-dtype 1 --ids-file data/prompt64.ids --dump-hidden tmp/f16/h_ref.bin --dump-logits tmp/f16/l_ref.bin | grep '^generated' > tmp/f16/g_ref.txt
md5sum tmp/f16/*
for x in h l g; do
  e=bin; [ $x = g ] && e=txt
  a=$(md5sum < tmp/f16/${x}_b64.$e); b=$(md5sum < tmp/f16/${x}_b1.$e); c=$(md5sum < tmp/f16/${x}_ref.$e)
  [ "$a" = "$b" ] && [ "$b" = "$c" ] && ok=true || ok=false
  echo "RESULT {\"bench\":\"f16_check\",\"what\":\"$x\",\"identical\":$ok}"
done
rm -rf tmp/f16
