#!/bin/bash
# llama.cpp llama-bench (pp64 / tg64) with the process's peak RSS, which llama-bench does not
# report (the Pi has no GNU time): VmHWM of the running process is sampled every 0.2 s; it only
# grows, so the last sample is the peak up to 0.2 s before exit. From ~/pinfer:
#   bash tmp/m2b/llama_rss.sh models/Qwen3.5-2B-Q4_K_M.gguf [THREADS] [REPS]
MODEL=${1:?model.gguf}
THREADS=${2:-3,4}
REPS=${3:-3}
OUT=$(mktemp)
./llama/llama-bench -m "$MODEL" -t "$THREADS" -p 64 -n 64 -r "$REPS" -o jsonl > "$OUT" &
pid=$!
hwm=0
while kill -0 "$pid" 2>/dev/null; do
  v=$(awk '/VmHWM/ {print $2}' "/proc/$pid/status" 2>/dev/null)
  [ -n "$v" ] && hwm=$v
  sleep 0.2
done
wait "$pid"
rc=$?
sed 's/^/RESULT /' "$OUT"
rm -f "$OUT"
echo "RESULT {\"bench\":\"llama_rss\",\"model\":\"$MODEL\",\"exit\":$rc,\"peak_rss_mb\":$((hwm / 1024))}"
exit $rc
