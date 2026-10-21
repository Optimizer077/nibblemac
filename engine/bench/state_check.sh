#!/bin/bash
# State save / restore on the Pi (engine/CONTEXT.md §6c): for each mode, run A = full 2112-token
# prompt; B = first 2048 tokens + --save-state; C = --load-state + the last 64 tokens. C must
# generate exactly A's tokens (NEON build); prints save / load ms and state size. From ~/pinfer:
#   bash engine/state_check.sh ./engine/pinfer_run_ctx6
B=${1:-./engine/pinfer_run_ctx6}
M=models/qwen35-0.8b-q4k-headq4k.pq
N=16
mkdir -p tmp/state
for S in "--spec 0" "--spec 3 --draft-vocab 32768" "--spec 3 --draft-vocab 32768 --adaptive"; do
  a=$($B -m $M -t 4 -n $N -c 2200 --ids-file data/ctx2112.ids $S --no-warmup | grep '^generated')
  tb0=$(date +%s.%N)
  $B -m $M -t 4 -n 1 -c 2200 --ids-file data/ctx2048.ids $S --no-warmup --save-state tmp/state/s.bin > tmp/state/b.txt 2>&1
  tb1=$(date +%s.%N)
  c=$($B -m $M -t 4 -n $N -c 2200 --ids-file data/tail2048_64.ids $S --no-warmup --load-state tmp/state/s.bin 2>tmp/state/c.err | tee tmp/state/c.txt | grep '^generated')
  tc1=$(date +%s.%N)
  size=$(stat -c %s tmp/state/s.bin)
  save=$(sed -n 's/^state: save \([0-9.]*\) ms.*/\1/p' tmp/state/b.txt)
  load=$(sed -n 's/^state: save [0-9.]* ms load \([0-9.]*\) ms.*/\1/p' tmp/state/c.txt)
  pc=$(sed -n 's/.*prefill \([0-9]*\) tok in \([0-9.]*\) s.*/\2/p' tmp/state/c.txt)
  [ "$a" = "$c" ] && m=true || m=false
  echo "RESULT {\"bench\":\"state_check\",\"mode\":\"$S\",\"match\":$m,\"state_bytes\":$size,\"save_ms\":${save:-0},\"load_ms\":${load:-0},\"run_b_s\":$(awk "BEGIN{print $tb1 - $tb0}"),\"run_c_s\":$(awk "BEGIN{print $tc1 - $tb1}"),\"tail_prefill_s\":${pc:-0}}"
done
rm -rf tmp/state
