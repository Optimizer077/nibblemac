#!/bin/bash
# Per-operation profile of Cerno 4B (Q4_0 body, Q8 head) shaped like one decision, for
# docs/network-audit.md. Run on the Pi from ~/pinfer through `harness/pi.py bench`:
#   python harness/pi.py bench audit-cerno-profile -- bash tmp/audit/audit_cerno.sh
# Inputs (tmp/audit/, made by tools/audit/cerno_ids.py from a real row of the cerno-pi set):
#   c_prefix.ids  the shared prefix of one request ("Context:\n" + state + "\n\nQuestion:")
#   c_q.ids       one question's suffix (question, options, "\nAnswer: (")
# 1. untimed page-in of the weights; 2. prefix with --save-state (profiled); 3. --load-state +
# the question suffix (profiled), as a later question of a decision. pinfer_run reads the full
# head for the last token (one 248320 x 2560 Q8 call); the decision path (eval_select) reads only
# the label rows, so that call is reported separately. The state file is scratch and is removed.
set -o pipefail
BIN=${BIN:-./engine/pinfer_run_cerno2}
M=models/cerno-4b-q4_0.pq
D=tmp/audit
freq() { cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | xargs; }
echo "bin $BIN md5 $(md5sum $BIN | cut -c1-8) model $(basename $M)"
echo "prefix tokens $(tr ',' '\n' < $D/c_prefix.ids | grep -c .) question tokens $(tr ',' '\n' < $D/c_q.ids | grep -c .)"
$BIN -m $M -t 4 -c 4096 --ids-file $D/c_prefix.ids -n 1 --no-warmup 2>&1 | grep -E '^load'
echo "== prefix+save freq $(freq)"
PF_FWD_STATS=1 PF_KERN_STATS=1 $BIN -m $M -t 4 -c 4096 --ids-file $D/c_prefix.ids -n 1 --bench \
  --no-warmup --save-state $D/c_prefix.state 2>&1 | grep -v '^generated'
ls -l $D/c_prefix.state
echo "== restore+question freq $(freq)"
PF_FWD_STATS=1 PF_KERN_STATS=1 $BIN -m $M -t 4 -c 4096 --ids-file $D/c_q.ids -n 1 --bench \
  --no-warmup --load-state $D/c_prefix.state 2>&1 | grep -v '^generated'
rm -f $D/c_prefix.state
echo "freq_after $(freq)"
