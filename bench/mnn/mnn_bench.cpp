// MNN-LLM baseline wrapper: feeds the shared 64 prompt token ids (bench/common/prompt64.json)
// and generates 64 tokens greedily, timing prefill and decode with MNN's own counters
// (LlmContext::prefill_us / decode_us; decode_us covers one single-token forward per generated
// token, like llama-bench tg). One process per thread count, so VmHWM is that config's peak RSS.
//
// Usage: mnn_bench <config.json> <prompt64.json> <threads> <reps> [n_gen] [precision] [memory]
// Prints one `RESULT {json}` line.
#include <llm/llm.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using MNN::Transformer::Llm;

// Returns the integer arrays that follow each `"ids": [` in the file, in order.
static std::vector<std::vector<int>> read_id_arrays(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    std::vector<std::vector<int>> out;
    size_t pos = 0;
    while ((pos = s.find("\"ids\": [", pos)) != std::string::npos) {
        pos += 8;
        size_t end = s.find(']', pos);
        std::vector<int> ids;
        std::stringstream body(s.substr(pos, end - pos));
        std::string item;
        while (std::getline(body, item, ',')) ids.push_back(std::stoi(item));
        out.push_back(ids);
        pos = end;
    }
    return out;
}

static long vm_hwm_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return std::stol(line.substr(6));
    }
    return -1;
}

static void stats(const std::vector<double>& v, double* mean, double* sd) {
    double m = 0, s = 0;
    for (double x : v) m += x;
    m /= v.size();
    for (double x : v) s += (x - m) * (x - m);
    *mean = m;
    *sd = v.size() > 1 ? std::sqrt(s / (v.size() - 1)) : 0.0;
}

static std::string join(const std::vector<double>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        char buf[32];
        snprintf(buf, sizeof buf, "%s%.4f", i ? "," : "", v[i]);
        s += buf;
    }
    return "[" + s + "]";
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s config.json prompt64.json threads reps [n_gen] [precision] [memory]\n", argv[0]);
        return 2;
    }
    const std::string config = argv[1];
    const auto arrays = read_id_arrays(argv[2]);
    const int threads = std::stoi(argv[3]);
    const int reps = std::stoi(argv[4]);
    const int n_gen = argc > 5 ? std::stoi(argv[5]) : 64;
    const std::string precision = argc > 6 ? argv[6] : "low";
    const std::string memory = argc > 7 ? argv[7] : "low";
    if (arrays.size() < 2) {
        fprintf(stderr, "prompt file must hold prompt ids and reference ids\n");
        return 2;
    }
    const std::vector<int>& prompt = arrays[0];
    const std::vector<int>& ref = arrays[1];

    std::unique_ptr<Llm> llm(Llm::createLLM(config));
    llm->set_config("{\"thread_num\":" + std::to_string(threads) + "}");
    llm->set_config("{\"sampler_type\":\"greedy\"}");
    llm->set_config("{\"reuse_kv\":false}");
    llm->set_config("{\"async\":false}");
    llm->set_config("{\"precision\":\"" + precision + "\"}");
    llm->set_config("{\"memory\":\"" + memory + "\"}");
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    // The official Qwen3.5-0.8B-MNN export is multimodal (is_visual=true -> Omni class). We do not
    // ship visual.mnn to the Pi, so load() reports failure for the vision part only; MNN's own
    // llm_bench ignores load()'s result the same way. Text correctness is checked by ref_match.
    const bool load_ok = llm->load();
    const auto* ctx = llm->getContext();

    std::vector<double> pre_tps, dec_tps;
    std::vector<int> gen_counts, matches;
    std::vector<int> first_out;
    for (int i = 0; i <= reps; ++i) {  // i == 0 is a warm-up run, excluded (as llm_bench does)
        llm->reset();
        llm->response(prompt, nullptr, nullptr, n_gen);
        const auto& out = ctx->output_tokens;
        int n_dec = (int)out.size();
        if (n_dec > 0 && llm->is_stop(out.back())) n_dec -= 1;  // stop token: no forward after it
        int match = 0;
        while (match < (int)out.size() && match < (int)ref.size() && out[match] == ref[match]) ++match;
        if (i == 0) {
            first_out = out;
            continue;
        }
        pre_tps.push_back(ctx->prompt_len / (ctx->prefill_us / 1e6));
        dec_tps.push_back(n_dec / (ctx->decode_us / 1e6));
        gen_counts.push_back((int)out.size());
        matches.push_back(match);
    }
    double pm, ps, dm, ds;
    stats(pre_tps, &pm, &ps);
    stats(dec_tps, &dm, &ds);
    int min_match = matches.empty() ? 0 : matches[0];
    int min_gen = gen_counts.empty() ? 0 : gen_counts[0];
    for (int m : matches) min_match = std::min(min_match, m);
    for (int g : gen_counts) min_gen = std::min(min_gen, g);
    std::string first;
    for (size_t i = 0; i < first_out.size() && i < 16; ++i) first += (i ? "," : "") + std::to_string(first_out[i]);

    printf("RESULT {\"engine\":\"mnn\",\"engine_version\":\"3.6.1\",\"model_file\":\"%s\","
           "\"quant\":\"int4 HQQ asym block64 (lm_head int4, embed bf16)\",\"threads\":%d,"
           "\"n_prompt\":%d,\"n_gen\":%d,\"reps\":%d,"
           "\"prefill_tps\":%.4f,\"prefill_tps_sd\":%.4f,\"decode_tps\":%.4f,\"decode_tps_sd\":%.4f,"
           "\"prefill_samples\":%s,\"decode_samples\":%s,\"peak_rss_mb\":%.1f,"
           "\"min_generated\":%d,\"min_ref_match\":%d,\"warmup_first16\":[%s],"
           "\"load_ok\":%s,\"notes\":\"precision=%s memory=%s greedy reuse_kv=false; warm-up rep excluded\"}\n",
           config.c_str(), threads, (int)prompt.size(), n_gen, reps, pm, ps, dm, ds,
           join(pre_tps).c_str(), join(dec_tps).c_str(), vm_hwm_kb() / 1024.0, min_gen, min_match,
           first.c_str(), load_ok ? "true" : "false", precision.c_str(), memory.c_str());
    return 0;
}
