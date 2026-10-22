#!/usr/bin/env bash
# llama.cpp perplexity on the first 8 chunks x 512 of wikitext-2 test (the slice pinfer_ppl
# --chunks 8 scores), PC x86, for any model's GGUFs: bench/quality/ppl_pc_c8.sh with the model
# name as a parameter. WSL root, project root (models/gguf/<MODEL>-<Q>.gguf):
#   bash bench/model2b/ppl_pc_c8.sh Qwen3.5-2B [bf16 Q8_0 Q4_K_M Q4_0]
cd "$(dirname "$0")/../.." || exit 1
MODEL=${1:?model name, e.g. Qwen3.5-2B}
shift
QS=${*:-bf16 Q8_0 Q4_K_M Q4_0}
for Q in $QS; do
    r=$(/root/src/llama.cpp-v0.7.0/build-host/bin/llama-perplexity -m "models/gguf/$MODEL-$Q.gguf" \
        -f bench/quality/wikitext-2-raw-test.txt -c 512 -b 512 --chunks 8 -t 16 2>&1 | grep 'Final estimate')
    echo "$MODEL $Q ${r##*PPL = }"
done
