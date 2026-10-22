#!/bin/bash
# Engine-level check and timing of the in-context checkpoint on the Pi with Cerno 4B
# (engine/CONTEXT.md §10). From ~/pinfer through the harness (the bench holds the lock):
#   python harness/pi.py bench ckpt-engine-short -- bash tmp/ck/ckpt_pi.sh ids
#   python harness/pi.py bench ckpt-engine-long  -- bash tmp/ck/ckpt_pi.sh ids_long
# Inputs in tmp/ck/<set>/ (tools/audit/cerno_ids.py + paste): c_prefix.ids (the shared prefix
# of a real decision row), c_q.ids (one question's suffix), c_row.ids (both). Runs, all -n 1,
# after an untimed page-in of the weights:
#   A  the whole row                                  -> the answer-slot token
#   B  the row with --rewind-at <prefix tokens>       -> checkpoint after the prefix, two
#      detours (the row reversed, 1 token generated each) each undone by pinfer_rewind, then
#      the question: must give A's token; prints checkpoint / rewind ms
#   C  the prefix with --save-state                   -> save ms, blob size
#   D  --load-state + the question                    -> load ms; must give A's token
# RESULT lines: pinfer_run's --bench records (B carries checkpoint_ms / rewind_ms, D
# state_load_ms) and one {"bench":"ckpt_pi", ...} with the token comparison.
set -o pipefail
BIN=${BIN:-./engine/pinfer_run_ck1}
M=${M:-models/cerno-4b-q4_0.pq}
SET=${1:-ids}
D=tmp/ck/$SET
freq() { cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | xargs; }
ntok() { tr ',' '\n' < "$1" | grep -c .; }
S=$(ntok $D/c_prefix.ids)
NQ=$(ntok $D/c_q.ids)
echo "bin $BIN md5 $(md5sum $BIN | cut -c1-8) model $(basename $M) set $SET prefix $S question $NQ freq $(freq)"
run() {  # label, args...; prints the lines we keep, returns the generated token in $GEN
  local label=$1; shift
  local out
  out=$($BIN -m $M -t 4 -c 4096 -n 1 --no-warmup "$@" 2>&1) || { echo "$label failed: $out"; return 1; }
  GEN=$(echo "$out" | sed -n 's/^generated: *//p')
  echo "$out" | grep -E '^(generated|load|state|rewind|RESULT)' | sed "s/^/  $label: /"
  echo "$out" | grep -E '^RESULT'
}
$BIN -m $M -t 4 -c 4096 -n 1 --no-warmup --ids-file $D/c_row.ids 2>&1 | grep '^load' | sed 's/^/  page-in: /'
echo "== A full row  freq $(freq)"
run A --ids-file $D/c_row.ids --bench || exit 1; GA=$GEN
echo "== B rewind-at $S  freq $(freq)"
run B --ids-file $D/c_row.ids --bench --rewind-at $S || exit 1; GB=$GEN
echo "== C prefix + save-state  freq $(freq)"
run C --ids-file $D/c_prefix.ids --bench --save-state tmp/ck/$SET.state || exit 1
ls -l tmp/ck/$SET.state
echo "== D load-state + question  freq $(freq)"
run D --ids-file $D/c_q.ids --bench --load-state tmp/ck/$SET.state || exit 1; GD=$GEN
rm -f tmp/ck/$SET.state
ok=false; [ "$GA" = "$GB" ] && [ "$GA" = "$GD" ] && [ -n "$GA" ] && ok=true
echo "RESULT {\"bench\":\"ckpt_pi\",\"set\":\"$SET\",\"prefix_tokens\":$S,\"question_tokens\":$NQ,\"token_a\":\"$GA\",\"token_b\":\"$GB\",\"token_d\":\"$GD\",\"identical\":$ok,\"freq_khz\":\"$(freq)\"}"
