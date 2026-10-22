#!/bin/bash
# Qwen3.5-2B speed on the Pi (engine/MODEL2B.md): plain greedy and speculative k=2 / k=3 (draft
# vocabulary 32768) on one prompt, ROUNDS rounds, one RESULT line per run (pinfer_run --bench:
# prefill_tps, decode_tps, peak_rss_mb, speculative stats). From ~/pinfer:
#   bash tmp/m2b/speed.sh ./engine/pinfer_run_2b1 prompt64 [ROUNDS] [N] [MODEL] [MODES]
# MODES: comma-separated from plain,k2,k3 (default all three; commas because harness/pi.py joins
# its arguments with spaces).
BIN=${1:?binary}
P=${2:-prompt64}
R=${3:-3}
N=${4:-64}
M=${5:-models/qwen35-2b-q4k-headq4k.pq}
MODES=${6:-plain,k2,k3}
for r in $(seq 1 "$R"); do
  for mode in ${MODES//,/ }; do
    case $mode in
      plain) S="" ;;
      k2) S="--spec 2 --draft-vocab 32768" ;;
      k3) S="--spec 3 --draft-vocab 32768" ;;
      *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    $BIN -m "$M" -t 4 -n "$N" --ids-file "data/$P.ids" --bench $S 2>/dev/null \
      | sed -n "s|^RESULT {|RESULT {\"prompt\":\"$P\",\"round\":$r,\"mode\":\"${S:-plain}\",|p"
  done
done
