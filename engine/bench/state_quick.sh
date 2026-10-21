#!/bin/bash
# State round trip on the Pi, short version of engine/bench/state_check.sh (prompt64 split at
# 40): A = 64 tokens + 16 generated; B = first 40 + --save-state; C = --load-state + the last
# 24. C must generate A's tokens. Also: the state of model 1 must be refused by model 2.
# Push it to the Pi, then from ~/pinfer: bash engine/state_quick.sh ./engine/pinfer_run_hardN
B=${1:?binary}
M=models/qwen35-0.8b-q4k-headq4k.pq
M2=models/qwen35-0.8b-mix-headq4k.pq
D=tmp/state_quick
mkdir -p $D
ids=$(tr -d ' \n' < data/prompt64.ids)
head=$(echo "$ids" | cut -d, -f1-40)
tail=$(echo "$ids" | cut -d, -f41-64)
for S in "--spec 0" "--spec 3 --draft-vocab 32768" "--spec 3 --draft-vocab 32768 --adaptive"; do
  a=$($B -m $M -t 4 -n 16 -c 128 --ids "$ids" $S --no-warmup 2>/dev/null | grep '^generated')
  $B -m $M -t 4 -n 1 -c 128 --ids "$head" $S --no-warmup --save-state $D/s.bin > $D/b.txt 2>&1
  c=$($B -m $M -t 4 -n 16 -c 128 --ids "$tail" $S --no-warmup --load-state $D/s.bin 2>/dev/null | grep '^generated')
  [ -n "$a" ] && [ "$a" = "$c" ] && m=true || m=false
  echo "RESULT {\"bench\":\"state_quick\",\"bin\":\"$B\",\"mode\":\"$S\",\"match\":$m,\"state_bytes\":$(stat -c %s $D/s.bin)}"
done
$B -m $M2 -t 4 -n 1 -c 128 --ids "$tail" --no-warmup --load-state $D/s.bin > $D/x.txt 2>&1
rc=$?
grep -q 'failed (-2)' $D/x.txt && r=true || r=false
echo "RESULT {\"bench\":\"state_quick\",\"bin\":\"$B\",\"mode\":\"other-model\",\"exit\":$rc,\"refused\":$r}"
rm -rf $D
