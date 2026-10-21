#!/usr/bin/env bash
# llama.cpp perplexity on the first 8 chunks (same slice pinfer_ppl --chunks 8 scores), PC x86.
cd "$(dirname "$0")/../.."
for Q in bf16 Q8_0 Q4_K_M Q4_0; do
    r=$(/root/src/llama.cpp-v0.7.0/build-host/bin/llama-perplexity -m models/gguf/Qwen3.5-0.8B-$Q.gguf \
        -f bench/quality/wikitext-2-raw-test.txt -c 512 -b 512 --chunks 8 -t 16 2>&1 | grep 'Final estimate')
    echo "$Q ${r##*PPL = }"
done
