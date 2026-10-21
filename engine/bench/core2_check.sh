#!/bin/bash
# Pi checks for the vectorized forward pass (engine/CORE.md section 2). From ~/pinfer:
#   bash engine/core2_check.sh [./engine/pinfer_run_core2] [./engine/pinfer_run_core2_noneon]
# 1. exp accuracy of the NEON path and scalar-vs-NEON bit identity (exptest_pi)
# 2. golden prompt, bf16 model: the 32 greedy tokens must equal ref/golden greedy_ids
# 3. bit identity of the prompt's hidden states and logits (q4k and bf16 models):
#    batched prefill (-b 64) vs token-by-token (-b 1) vs the -DPF_NO_NEON scalar-mirror build
B=${1:-./engine/pinfer_run_core2}
BN=${2:-./engine/pinfer_run_core2_noneon}
GOLD="628,381,1429,364,264,6669,2020,314,8806,13,1049,369,264,5243,5576,364,45285,321,14939,25161,13,561,45980,20010,220,19,369,264,2526,6165,421,628"
mkdir -p tmp/core2
echo "== 1. exptest (NEON path on the Pi)"
[ -x ./engine/exptest_pi ] && ./engine/exptest_pi
echo "== 2. golden greedy, bf16"
g=$($B -m models/qwen35-0.8b-bf16.pq -t 4 -n 32 --ids 760,45980,20010,220,19,369,264,2526,6165,421 | tee /dev/stderr | sed -n 's/^generated: //p')
if [ "$g" = "$GOLD" ]; then echo "golden greedy: 32/32 MATCH"; else echo "golden greedy: MISMATCH"; fi
for M in qwen35-0.8b-q4k-headq4k qwen35-0.8b-bf16; do
  echo "== 3. bit identity, $M, prompt64: b64 (NEON) / b1 (NEON) / b64 (no-NEON mirror)"
  for v in b64 b1 noneon; do
    case $v in b64) bin=$B; b=64;; b1) bin=$B; b=1;; noneon) bin=$BN; b=64;; esac
    $bin -m models/$M.pq -t 4 -n 2 -b $b --ids-file data/prompt64.ids \
      --dump-hidden tmp/core2/h_$v.bin --dump-logits tmp/core2/l_$v.bin | grep '^load' | sed "s/^/  $v: /"
  done
  md5sum tmp/core2/*.bin
  rm -f tmp/core2/*.bin
done
rmdir tmp/core2 2>/dev/null
