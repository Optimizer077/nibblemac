#!/bin/bash
# oc_soak.sh SECONDS CMD... -- run ON THE PI from ~/pinfer, through harness/pi.py bench (it holds the
# Pi-wide lock). Repeats CMD (a 4-core load) for SECONDS while sampling the ARM clock, core voltage,
# temperature and throttle flags every 2 s, then prints one RESULT line for the harness:
#   python harness/pi.py bench oc-soak-2000 -- bash os/overclock/oc_soak.sh 300 \
#       ./engine/pinfer_run_fast -m models/qwen35-0.8b-q4k-headq4k.pq -t 4 --ids-file data/ctx2048.ids -n 1 --bench
# (a 2048-token prefill: about 70 s of 4-core compute per run, with the model already in the page cache)
# Pass: runs_failed = 0, throttled_or = 0x0 (no under-voltage, no frequency cap, no thermal limit),
# min_clock_mhz under load = the configured arm_freq, max_temp_c below the 80 C soft limit.
set -u
DUR=${1:?usage: oc_soak.sh SECONDS CMD...}; shift
[ $# -gt 0 ] || { echo "usage: oc_soak.sh SECONDS CMD..." >&2; exit 2; }
END=$(( $(date +%s) + DUR ))
SAMPLES=$(mktemp); RUNS=$(mktemp)
trap 'rm -f "$SAMPLES" "$RUNS"' EXIT

( while [ "$(date +%s)" -lt "$END" ]; do
      if "$@" > /dev/null 2>&1; then echo ok; else echo "fail $?"; fi
  done > "$RUNS" ) &
LOAD=$!

THR_OR=0   # throttle flags OR-ed in bash: the Pi's default awk (mawk) has no or() / strtonum()
while [ "$(date +%s)" -lt "$END" ]; do
    sleep 2
    clk=$(vcgencmd measure_clock arm | cut -d= -f2)
    tmp=$(vcgencmd measure_temp | tr -dc '0-9.')
    vlt=$(vcgencmd measure_volts core | tr -dc '0-9.')
    thr=$(vcgencmd get_throttled | cut -d= -f2)
    THR_OR=$(( THR_OR | thr ))
    echo "$clk $tmp $vlt" >> "$SAMPLES"
done
wait "$LOAD"

ok=$(grep -c '^ok' "$RUNS"); bad=$(grep -c '^fail' "$RUNS")
# at_max_frac: share of samples at the highest clock seen (the gaps between runs, while the next
# run loads its model, let the ondemand governor drop the clock; min_clock alone would mislead)
awk -v ok="$ok" -v bad="$bad" -v dur="$DUR" -v thr="$(printf '0x%x' "$THR_OR")" '
    { c[NR] = $1 / 1e6; if (NR == 1 || c[NR] < minc) minc = c[NR]; if (c[NR] > maxc) maxc = c[NR]; csum += c[NR]
      if ($2 > maxt) maxt = $2; tsum += $2; if ($3 > maxv) maxv = $3 }
    END { for (i = 1; i <= NR; i++) if (c[i] >= maxc - 1) atmax++
          printf "RESULT {\"seconds\":%d,\"samples\":%d,\"runs_ok\":%d,\"runs_failed\":%d,", dur, NR, ok, bad
          printf "\"min_clock_mhz\":%.0f,\"mean_clock_mhz\":%.0f,\"max_clock_mhz\":%.0f,", minc, csum / NR, maxc
          printf "\"at_max_frac\":%.3f,\"max_temp_c\":%.1f,\"mean_temp_c\":%.1f,", atmax / NR, maxt, tsum / NR
          printf "\"max_core_v\":%.4f,\"throttled_or\":\"%s\"}\n", maxv, thr }' "$SAMPLES"
grep '^fail' "$RUNS" | sort | uniq -c | sed 's/^/load failures: /'
exit 0
