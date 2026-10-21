#!/bin/bash
# Where do the threads of a decode run sit? Samples each thread's CPU (psr) and allowed set
# 20 times during a 160-token plain decode. The main thread (tid = pid) is the pool's
# unpinned caller (0.2.0) or pinned to CPU 0 (0.1.0). From ~/pinfer:
#   bash engine/caller_cpu.sh ./engine/pinfer_run_hardN
B=${1:?binary}
M=models/qwen35-0.8b-q4k-headq4k.pq
$B -m $M -t 4 -n 160 --ids-file data/prompt64.ids --no-warmup > /dev/null 2>&1 &
P=$!
sleep 4
for i in $(seq 1 20); do
  main=$(ps -L -o tid=,psr= -p $P | awk -v p=$P '$1 == p {print $2}')
  others=$(ps -L -o tid=,psr= -p $P | awk -v p=$P '$1 != p {printf "%s%s", s, $2; s=","}')
  echo "RESULT {\"bench\":\"caller_cpu\",\"bin\":\"$B\",\"sample\":$i,\"main_cpu\":${main:-null},\"worker_cpus\":[${others}],\"main_allowed\":\"$(taskset -pc $P 2>/dev/null | sed 's/.*: //')\"}"
  sleep 0.5
done
wait $P
