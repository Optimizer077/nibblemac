#!/bin/bash
# CPU + GPU co-run test (Pi side, run from ~/pinfer inside ONE `pi.py bench` lock):
#   1. CPU alone   kbench q4, 4 threads, batch N_CPU, the 4B shapes      (variant cpu-alone)
#   2. GPU alone   bench_q4.py on GPU_SHAPE, n = N_GPU, GPU_SECS seconds  (tag gpu-alone)
#   3. both        GPU loop in the background until kbench (variant cpu-corun) ends (tag gpu-corun)
#   4. CPU alone   again (variant cpu-alone2), to see thermal / drift effects
# Every 2 s: temperature, throttle flags, ARM and V3D clocks (lines "SAMPLE ...").
# usage: bash tmp/gpu/corun.sh GPU_SHAPE N_GPU N_CPU KB_TIME KB_REPS GPU_SECS [KB_SHAPES]
set -u
G="$HOME/pinfer/tmp/gpu"
GSHAPE=${1:-gate}; NG=${2:-11}; NC=${3:-7}; KT=${4:-1.0}; KR=${5:-3}; GS=${6:-8}
KS=${7:-c4gate,c4down,c4qkv}
KB="$G/kbench_gpu1 --dtypes q4 --shapes $KS --threads 4 --batch $NC --time $KT --reps $KR"
STOP="$G/corun.stop"; GOUT="$G/corun_gpu.out"
rm -f "$STOP" "$GOUT"
sampler() {
    while true; do
        echo "SAMPLE $(date +%T.%N | cut -c1-12) $(vcgencmd measure_temp) $(vcgencmd get_throttled)" \
             "arm=$(vcgencmd measure_clock arm | cut -d= -f2) v3d=$(vcgencmd measure_clock v3d | cut -d= -f2)"
        sleep 2
    done
}
sampler & SP=$!
trap 'kill $SP 2>/dev/null; touch "$STOP"' EXIT
echo "PHASE cpu-alone $(date +%T)"
$KB --variant cpu-alone
echo "PHASE gpu-alone $(date +%T)"
bash "$G/gpurun.sh" bench_q4.py --shapes "$GSHAPE" --n "$NG" --seconds "$GS" --tag gpu-alone | tee "$G/corun_gpu_alone.out"
if ! grep -q '"exact_vs_model": true' "$G/corun_gpu_alone.out"; then
    echo "GPU kernel output not exact or hung: co-run skipped $(date +%T)"
    exit 0
fi
echo "PHASE corun $(date +%T)"
bash "$G/gpurun.sh" bench_q4.py --shapes "$GSHAPE" --n "$NG" --seconds 400 --until-file "$STOP" \
     --tag gpu-corun > "$GOUT" 2>&1 &
GP=$!
for _ in $(seq 1 300); do grep -q LOOP_START "$GOUT" 2>/dev/null && break; sleep 0.1; done
echo "GPU loop started $(date +%T)"
$KB --variant cpu-corun
touch "$STOP"
wait $GP
cat "$GOUT"
echo "PHASE cpu-alone2 $(date +%T)"
$KB --variant cpu-alone2
echo "PHASE done $(date +%T)"
