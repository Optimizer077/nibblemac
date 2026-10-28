// Local adapter for the pinned public llama.cpp API. Does not modify upstream.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char **argv) try {
    if (argc != 5) throw std::runtime_error("usage: adapter model.gguf ids.i32 generate quality_chunks");
    const int generate = std::stoi(argv[3]), chunks = std::stoi(argv[4]);
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open token file");
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % 4) throw std::runtime_error("invalid token file");
    std::vector<llama_token> ids(static_cast<size_t>(bytes) / 4);
    input.seekg(0);
    input.read(reinterpret_cast<char *>(ids.data()), bytes);
    if (!input || generate < 2 || chunks < 0 || (chunks && ids.size() != size_t(chunks * 512)) ||
        (!chunks && ids.size() + generate > 2048)) throw std::runtime_error("invalid arguments");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) throw std::runtime_error("model load failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (auto id : ids) if (id < 0 || id >= vocab) throw std::runtime_error("invalid token id");
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_threads = cp.n_threads_batch = 6;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context creation failed");
    auto *batch = llama_batch_ext_init(ctx);
    auto process = [&](const llama_token *tokens, int count, int pos, bool all) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < count; ++i) {
            int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
            llama_pos position = pos + i;
            llama_batch_ext_set_pos(batch, index, &position);
            llama_batch_ext_set_output_logits(batch, index, all || i == count - 1);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) throw std::runtime_error("decode failed");
    };
    auto greedy = [&]() {
        float *logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 0; i < vocab; ++i) {
            if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite logits");
            if (logits[i] > logits[best]) best = i;
        }
        return llama_token(best);
    };
    auto reset = [&]() { llama_memory_clear(llama_get_memory(ctx), true); };
    if (chunks) {
        double nll = 0;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            reset();
            for (int pos = 0; pos < 512; pos += 64) {
                process(ids.data() + chunk * 512 + pos, 64, pos, pos >= 256);
                if (pos < 256) continue;
                for (int j = 0; j < 64 && pos + j < 511; ++j) {
                    float *logits = llama_get_logits_ith(ctx, j);
                    double maximum = *std::max_element(logits, logits + vocab), sum = 0;
                    for (int k = 0; k < vocab; ++k) {
                        if (!std::isfinite(logits[k])) throw std::runtime_error("non-finite quality logits");
                        sum += std::exp(double(logits[k]) - maximum);
                    }
                    nll += maximum + std::log(sum) - logits[ids[chunk * 512 + pos + j + 1]];
                }
            }
        }
        printf("RESULT {\"ppl\":%.10f,\"nll\":%.10f,\"tokens_scored\":%d}\n", std::exp(nll / (chunks * 255)), nll, chunks * 255);
    } else {
        auto prefill = [&]() {
            for (size_t pos = 0; pos < ids.size(); pos += 64)
                process(ids.data() + pos, std::min(size_t(64), ids.size() - pos), int(pos), false);
            return greedy();
        };
        auto token = prefill();
        for (int i = 0; i < 4; ++i) { process(&token, 1, ids.size() + i, false); token = greedy(); }
        reset();
        auto start = Clock::now();
        token = prefill();
        double prefill_s = seconds(start);
        std::vector<llama_token> generated{token};
        start = Clock::now();
        for (int i = 1; i < generate; ++i) {
            process(&token, 1, ids.size() + i - 1, false);
            token = greedy();
            generated.push_back(token);
        }
        double decode_s = seconds(start);
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        printf("RESULT {\"prefill_tps\":%.8f,\"decode_tps\":%.8f,\"prefill_seconds\":%.8f,\"decode_seconds\":%.8f,\"peak_rss_mb\":%.4f,\"generated_ids\":[",
               ids.size() / prefill_s, (generate - 1) / decode_s, prefill_s, decode_s, usage.ru_maxrss / 1048576.0);
        for (size_t i = 0; i < generated.size(); ++i) printf("%s%d", i ? "," : "", generated[i]);
        printf("]}\n");
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
