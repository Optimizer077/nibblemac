#!/bin/bash
# Speculative decoding on chat-formatted prompts (bench/common/chat*.ids): plain, fixed k,
# adaptive k and n-gram (prompt-lookup) drafts, Q4K model, 4 threads, N generated tokens
# (pinfer_run ignores EOS, so keep N below the typical answer length). Each run is preceded by
# a RESULT row naming prompt and config. From ~/pinfer:
#   bash engine/chat_spec.sh ./engine/pinfer_run_fast [N] [REPS] [CONFIGS] [PROMPTS]
# CONFIGS: comma-separated (commas because harness/pi.py joins its arguments with spaces):
#   plain          plain greedy loop
#   kN, kNfull     fixed k = N drafts, draft vocab 32768 / all 248320 ids
#   adaptive, aN   adaptive k <= 3 / k <= N (--adaptive)
#   kNgM, aNgM     the same plus n-gram drafts of up to M tokens (--ngram M)
#   gM             n-gram drafts only (--spec 0 --ngram M), plain steps without a match
#   (k3, adaptive, k2, k3full, k2full keep their earlier meaning)
# PROMPTS: comma-separated names of data/*.ids (default: the six chat prompts). For ctx*
# prompts the run uses -c (prompt + N + 64) and --no-warmup, as engine/VERIFY.md did.
# M=path overrides the model.
BIN=${1:-./engine/pinfer_run_fast}
N=${2:-96}
REPS=${3:-2}
CONFIGS=${4:-plain,k3,adaptive}
PROMPTS=${5:-chat27,chat_code_py,chat_code_c,chat_list,chat_math,chat_story}
M=${M:-models/qwen35-0.8b-q4k-headq4k.pq}
for r in $(seq 1 "$REPS"); do
  for p in ${PROMPTS//,/ }; do
    X=""
    case $p in
      ctx*) X="-c $(( $(tr ',' '\n' < "data/$p.ids" | grep -c .) + N + 64 )) --no-warmup" ;;
    esac
    for cfg in ${CONFIGS//,/ }; do
      if [ "$cfg" = plain ]; then S=""
      elif [ "$cfg" = adaptive ]; then S="--spec 3 --draft-vocab 32768 --adaptive"
      elif [[ $cfg =~ ^k([0-9])full$ ]]; then S="--spec ${BASH_REMATCH[1]}"
      elif [[ $cfg =~ ^k([0-9])$ ]]; then S="--spec ${BASH_REMATCH[1]} --draft-vocab 32768"
      elif [[ $cfg =~ ^a([0-9])$ ]]; then
        S="--spec ${BASH_REMATCH[1]} --draft-vocab 32768 --adaptive"
      elif [[ $cfg =~ ^k([0-9])g([0-9])$ ]]; then
        S="--spec ${BASH_REMATCH[1]} --draft-vocab 32768 --ngram ${BASH_REMATCH[2]}"
      elif [[ $cfg =~ ^a([0-9])g([0-9])$ ]]; then
        S="--spec ${BASH_REMATCH[1]} --draft-vocab 32768 --adaptive --ngram ${BASH_REMATCH[2]}"
      elif [[ $cfg =~ ^g([0-9])$ ]]; then S="--spec 0 --ngram ${BASH_REMATCH[1]}"
      else
        echo "RESULT {\"bench\":\"chat_spec\",\"cfg\":\"$cfg\",\"error\":\"unknown config\"}"
        exit 2
      fi
      echo "RESULT {\"bench\":\"chat_spec\",\"prompt\":\"$p\",\"cfg\":\"$cfg\",\"rep\":$r}"
      $BIN -m $M -t 4 -n "$N" --ids-file "data/$p.ids" --bench $X $S | grep '^RESULT'
    done
  done
done
