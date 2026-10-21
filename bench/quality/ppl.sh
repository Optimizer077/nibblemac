#!/usr/bin/env bash
# Perplexity of one GGUF on the fixed wikitext-2 slice with llama.cpp's llama-perplexity.
# Prints one `RESULT {json}` line (for harness/pi.py bench on the Pi; also used on the PC).
# Usage: ppl.sh <llama-perplexity bin> <model.gguf> <text file> [chunks=64] [threads=4] [platform] [logdir]
#
# Settings (fixed for every quantization and platform): -c 512 -b 512 --chunks N, default KV f16,
# no BOS change. llama-perplexity scores the second half of each 512-token chunk (255 tokens per
# chunk; first half is context), so N chunks = N*512 tokens read, N*255 tokens scored.
# Text: bench/quality/wikitext-2-raw-test.txt = wikitext-2-raw-v1 test (sha256 bbf94c53...),
# tokenized by llama.cpp from the start of the file; chunks are the first N*512 tokens.
set -uo pipefail
BIN="$1"; MODEL="$2"; TEXT="$3"; CHUNKS="${4:-64}"; THREADS="${5:-4}"; PLATFORM="${6:-pi4}"; LOGDIR="${7:-}"
LOG="$(mktemp)"
START=$(date +%s.%N)
"$BIN" -m "$MODEL" -f "$TEXT" -c 512 -b 512 --chunks "$CHUNKS" -t "$THREADS" > "$LOG" 2>&1
CODE=$?
END=$(date +%s.%N)
FINAL=$(grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*" "$LOG" | tail -1)
PPL=$(echo "$FINAL" | awk '{print $5}')
ERR=$(echo "$FINAL" | awk '{print $7}')
TPS=$(grep -o "[0-9.]* tokens per second" "$LOG" | tail -1 | awk '{print $1}')
NTOK=$(grep -o "tokenizing the input .*" "$LOG" | head -1)
TOKS=$(grep -o "tokenization took .*\|[0-9]* tokens" "$LOG" | head -3 | tr '\n' ' ')
SHA=$(sha256sum "$TEXT" | cut -c1-16)
[ -n "$LOGDIR" ] && mkdir -p "$LOGDIR" && cp "$LOG" "$LOGDIR/ppl_$(basename "$MODEL" .gguf)_${PLATFORM}_c${CHUNKS}.log"
if [ "$CODE" -ne 0 ] || [ -z "$PPL" ]; then
    echo "perplexity failed (exit $CODE); log tail:" >&2
    tail -20 "$LOG" >&2
    rm -f "$LOG"
    exit 1
fi
printf 'RESULT {"engine":"llama.cpp llama-perplexity","model_file":"%s","platform":"%s","text_sha256_16":"%s","ctx":512,"batch":512,"chunks":%s,"tokens_scored":%d,"threads":%s,"ppl":%s,"ppl_err":%s,"seconds":%.1f,"notes":"%s"}\n' \
    "$(basename "$MODEL")" "$PLATFORM" "$SHA" "$CHUNKS" $((CHUNKS * 255)) "$THREADS" "$PPL" "$ERR" \
    "$(echo "$END - $START" | bc)" "$(echo "$NTOK $TOKS" | tr -d '"' | cut -c1-200)"
rm -f "$LOG"
