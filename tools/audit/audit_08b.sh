#!/bin/bash
# Per-operation profile of Qwen3.5-0.8B (Q4K + Q4K head) with the shipping binary, for
# docs/network-audit.md. Run on the Pi from ~/pinfer through `harness/pi.py bench`:
#   python harness/pi.py bench audit-08b-profile -- bash tmp/audit/audit_08b.sh
# Two runs of prompt64 + 64 generated tokens with PF_FWD_STATS=1 PF_KERN_STATS=1:
# plain decode (--spec 0) and fixed k = 4 with draft vocab 32768. The 64-token prefill of each run
# is profiled too (KSTATS n = 64 rows, FSTATS "batched"). Profiled runs are ~2% slower than clean.
set -o pipefail
BIN=${BIN:-./engine/pinfer_run_fast}
M=models/qwen35-0.8b-q4k-headq4k.pq
freq() { cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | xargs; }
echo "bin $BIN md5 $(md5sum $BIN | cut -c1-8) model $(basename $M)"
for mode in "--spec 0" "--spec 4 --draft-vocab 32768"; do
  echo "== $mode freq $(freq)"
  PF_FWD_STATS=1 PF_KERN_STATS=1 $BIN -m $M -t 4 --ids-file data/prompt64.ids -n 64 $mode --bench 2>&1 \
    | grep -v '^generated'
done
echo "freq_after $(freq)"
