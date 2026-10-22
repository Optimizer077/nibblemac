#!/bin/bash
# pinfer serve, measured: for each server configuration start `pinfer serve` on localhost, run
# serve_client.py against it (RESULT lines on stdout), stop it. Meant for ONE locked harness
# call on the Pi (well under 10 minutes with N=64), from ~/pinfer:
#   python harness/pi.py bench serve -- bash tmp/serve/bench/serve_bench.sh tmp/serve 64
# DIR holds python/ (the package source), lib/libpinfer.so and bench/ (these scripts).
# Overrides: PY (python with numpy+tokenizers+jinja2), MODEL, TOKENIZER, PORT, CONFIGS.
# Server logs (one line per request) go to stderr.
DIR=${1:-tmp/serve}
N=${2:-64}
PORT=${PORT:-8091}
PY=${PY:-$HOME/pinfer/venvs/pinfer/bin/python}
MODEL=${MODEL:-models/qwen35-0.8b-q4k-headq4k.pq}
TOKENIZER=${TOKENIZER:-models/qwen35-0.8b-tokenizer}
CONFIGS=${CONFIGS:-generated,template,noreuse,nosnap}
DIR=$(cd "$DIR" && pwd) || exit 1
export PYTHONPATH=$DIR/python PINFER_LIB=${PINFER_LIB:-$DIR/lib/libpinfer.so}
URL=http://127.0.0.1:$PORT
SPID=

cleanup() {
  if [ -n "$SPID" ]; then kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=; fi
}
trap cleanup EXIT INT TERM

up() {  # wait until /health answers (model loaded), at most 120 s
  for _ in $(seq 1 240); do
    "$PY" -c "import urllib.request; urllib.request.urlopen('$URL/health', timeout=2)" \
      2>/dev/null && return 0
    kill -0 "$SPID" 2>/dev/null || return 1
    sleep 0.5
  done
  return 1
}

run_cfg() {  # label, client tests, server flags...
  local label=$1 tests=$2 rc=0
  shift 2
  "$PY" -m pinfer serve -m "$MODEL" --tokenizer "$TOKENIZER" --host 127.0.0.1 --port "$PORT" \
    "$@" 2>"$DIR/server_$label.log" &
  SPID=$!
  if up; then
    "$PY" "$DIR/bench/serve_client.py" --url "$URL" --label "$label" --tests "$tests" \
      --max-tokens "$N" --warmup || rc=$?
  else
    echo "server ($label) did not come up" >&2; rc=1
  fi
  cleanup
  echo "--- server log: $label (flags: $*)" >&2
  cat "$DIR/server_$label.log" >&2
  return $rc
}

if [ -e /proc/device-tree/model ]; then BOARD=$(tr -d '\0' </proc/device-tree/model); else BOARD=$(uname -m); fi
LIBMD5=$(md5sum "$PINFER_LIB" | cut -c1-12)
echo "RESULT {\"bench\":\"serve_env\",\"lib_md5\":\"$LIBMD5\",\"model\":\"$MODEL\",\"max_tokens\":$N,\"board\":\"$BOARD\"}"
status=0
for cfg in ${CONFIGS//,/ }; do
  case $cfg in
    generated) run_cfg generated single,conversation,alternating || status=1 ;;
    template)  run_cfg template conversation --history template || status=1 ;;
    noreuse)   run_cfg noreuse single,conversation --no-reuse || status=1 ;;
    nosnap)    run_cfg nosnap alternating --state-cache-mb 0 || status=1 ;;
    *) echo "unknown config $cfg" >&2; status=1 ;;
  esac
done
exit $status
