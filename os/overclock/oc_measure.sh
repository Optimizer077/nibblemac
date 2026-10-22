#!/bin/bash
# oc_measure.sh BIN ROUNDS SET -- the overclock A/B set; run ON THE PI from ~/pinfer through
# harness/pi.py bench (Pi-wide lock), once at the stock clock and once in the overclock test boot,
# with the same binary:
#   python harness/pi.py bench oc-1500-short -- bash os/overclock/oc_measure.sh ./engine/pinfer_run_fast 3 short
# SET
#   short  Qwen3.5-0.8B: prompt64 plain and default speculative, chat27 default speculative, ctx512 prefill
#   ctx    Qwen3.5-0.8B: decode after the 4096-token context, fp32 and fp16 KV (one run each per round)
#   cerno  Cerno 4B: prefill of the 64- and 256-token decision prompts (CERNO_IDS, default data/cerno)
# Every engine RESULT line is preceded by a label line with the case, the round, the configured
# arm_freq and the throttle flags read right after that run (non-zero flags = the run does not count).
set -u
BIN=${1:?usage: oc_measure.sh BIN ROUNDS SET}; ROUNDS=${2:?}; SET=${3:?}
Q08=models/qwen35-0.8b-q4k-headq4k.pq
C4B=models/cerno-4b-q4_0.pq
CIDS=${CERNO_IDS:-data/cerno}
SPEC="--spec 4 --draft-vocab 32768 --adaptive --ngram 7"   # the library default (pinfer_default_spec_params)
[ -x "$BIN" ] || { echo "RESULT {\"error\":\"binary not found: $BIN\"}"; exit 1; }
ARMF=$(vcgencmd get_config arm_freq | cut -d= -f2)

one() {   # one CASE ROUND ARGS...: run the engine once, label line first
    local case=$1 r=$2; shift 2
    local out; out=$("$BIN" "$@" --bench 2>/dev/null | grep '^RESULT')
    printf 'RESULT {"case":"%s","round":%d,"arm_freq":%s,"throttled":"%s","ok":%s}\n' "$case" "$r" "$ARMF" \
        "$(vcgencmd get_throttled | cut -d= -f2)" "$([ -n "$out" ] && echo true || echo false)"
    [ -n "$out" ] && echo "$out"
}

for r in $(seq 1 "$ROUNDS"); do
    case "$SET" in
    short)
        one p64-plain "$r" -m $Q08 -t 4 -n 64 --ids-file data/prompt64.ids
        one p64-spec "$r" -m $Q08 -t 4 -n 64 --ids-file data/prompt64.ids $SPEC
        one chat27-spec "$r" -m $Q08 -t 4 -n 64 --ids-file data/chat27.ids $SPEC
        one ctx512-prefill "$r" -m $Q08 -t 4 -n 1 --ids-file data/ctx512.ids ;;
    ctx)
        one ctx4096-kv0 "$r" -m $Q08 -t 4 -n 32 --kv-dtype 0 --ids-file data/ctx4096.ids
        one ctx4096-kv1 "$r" -m $Q08 -t 4 -n 32 --kv-dtype 1 --ids-file data/ctx4096.ids ;;
    cerno)
        [ -f "$CIDS/p64.ids" ] || { echo "RESULT {\"error\":\"missing $CIDS/p64.ids\"}"; exit 1; }
        one cerno-p64-prefill "$r" -m $C4B -t 4 -n 2 --ids-file "$CIDS/p64.ids"
        one cerno-p256-prefill "$r" -m $C4B -t 4 -n 2 --ids-file "$CIDS/p256.ids" ;;
    *) echo "unknown SET $SET" >&2; exit 2 ;;
    esac
done
