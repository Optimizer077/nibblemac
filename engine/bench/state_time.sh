#!/bin/bash
# State save / load time, reference vs candidate (the candidate fingerprints the model file on
# every save and load). 64-token state of the Q4K model, 3 repetitions each, interleaved.
# Push it to the Pi, then from ~/pinfer: bash engine/state_time.sh NEW [OLD]
NEW=${1:?candidate}
OLD=${2:-./engine/pinfer_run_fast}
M=models/qwen35-0.8b-q4k-headq4k.pq
D=tmp/state_time
mkdir -p $D
ids=$(tr -d ' \n' < data/prompt64.ids)
for r in 1 2 3; do
  for b in $OLD $NEW; do
    $b -m $M -t 4 -n 1 -c 128 --ids "$ids" --no-warmup --save-state $D/s.bin > $D/o.txt 2>&1
    s=$(sed -n 's/^state: save \([0-9.]*\) ms.*/\1/p' $D/o.txt)
    $b -m $M -t 4 -n 1 -c 128 --ids 13 --no-warmup --load-state $D/s.bin > $D/o.txt 2>&1
    l=$(sed -n 's/^state: save [0-9.]* ms load \([0-9.]*\) ms.*/\1/p' $D/o.txt)
    echo "RESULT {\"bench\":\"state_time\",\"bin\":\"$b\",\"round\":$r,\"save_ms\":${s:-null},\"load_ms\":${l:-null},\"bytes\":$(stat -c %s $D/s.bin)}"
  done
done
rm -rf $D
