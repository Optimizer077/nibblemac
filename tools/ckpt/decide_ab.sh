#!/bin/bash
# Cerno decisions on the Pi with the in-engine checkpoint (ckpt) or the save_state / load_state
# path (state) of the same library, for the before -> after timing and the bit-identity check
# (cerno-pi bench/compare_runs.py on the result files). From ~/pinfer through the harness:
#   python harness/pi.py bench ckpt-decide-ckpt-r1 -- bash tmp/ck/decide_ab.sh ckpt
#   python harness/pi.py bench ckpt-decide-state-r1 -- bash tmp/ck/decide_ab.sh state
#   python harness/pi.py bench ckpt-decide-ckpt-a -- bash tmp/ck/decide_ab.sh ckpt <request list>
# Default request list: the lead's 5-request set (oc-1500-decide, 13 questions). Files under
# tmp/ck/: py/pinfer (the Python package of the branch), lib/libpinfer.so (NEON build of the
# branch), cerno/{cerno_pi,bench,requests} (cerno-pi feat/checkpoint).
set -o pipefail
MODE=${1:-ckpt}
ONLY=${2:-support_triage,agent_guardrail,intent_smart_home,nli,medical_triage}
extra=""
[ "$MODE" = state ] && extra=--no-checkpoint
freq() { cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | xargs; }
echo "mode $MODE lib md5 $(md5sum tmp/ck/lib/libpinfer.so | cut -c1-8) freq $(freq)"
PYTHONPATH=tmp/ck/py PINFER_LIB=tmp/ck/lib/libpinfer.so venvs/pinfer/bin/python \
  tmp/ck/cerno/bench/decide_bench.py -m models/cerno-4b-q4_0.pq \
  --request tmp/ck/cerno/requests/correctness.json --warmup --only "$ONLY" $extra
rc=$?
echo "freq_after $(freq) exit $rc"
exit $rc
