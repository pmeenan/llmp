// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External llama.cpp reference only; this does not implement llmpalooza inference.
//
// The correctness oracle of M3's DeepSeek V4 Flash slice: llama.cpp at the
// pinned build (b29c606e, the digest-pinned image) on the same GGUF, with
// flash attention on, F16 caches, one sequence, a 4,096-position context and
// 512-token ubatches. It tokenizes (llama.cpp's tokenizer stands in for the
// native one, a parallel slice), then:
//   --prompts FILE --generate N  each prompt (`name<TAB>text`, \n and \t
//       escaped), BOS first, as one ubatch from a cleared memory, then N
//       greedy tokens one at a time; every step's logits are written.
//   --ppl FILE                   the file's text, BOS first, cut to the
//       context, in ubatches of 512: every token's negative log-likelihood.
//   --dump NAMES                 the named callback tensors of the first
//       prompt's prefill, as F32 (a separate, instrumented run).
// Outputs: prompts.tokens and generated.tokens (`name<TAB>ids`), ppl.tokens,
// NAME.logits.f32, ppl.nll.f64, dump/NAME.f32 and summary.json.

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

std::string unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            out += c == 'n' ? '\n' : c == 't' ? '\t' : c;
        } else {
            out += s[i];
        }
    }
    return out;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> ids(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(),
                           /*add_special=*/false, /*parse_special=*/false);
    require(n >= 0, "tokenization failed");
    ids.resize((size_t) n);
    ids.insert(ids.begin(), llama_vocab_bos(vocab));
    return ids;
}

std::string join(const std::vector<llama_token> & ids) {
    std::string s;
    for (size_t i = 0; i < ids.size(); ++i) {
        s += (i ? " " : "") + std::to_string(ids[i]);
    }
    return s;
}

struct Dump {
    std::set<std::string> names;
    bool active = false;
    std::filesystem::path dir;
    std::vector<uint8_t> buffer;
};

bool dump_callback(ggml_tensor * t, bool ask, void * user) {
    auto * d = static_cast<Dump *>(user);
    const bool wanted = d->active && d->names.count(t->name) > 0;
    if (ask) {
        return wanted;
    }
    if (wanted && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        d->buffer.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, d->buffer.data(), 0, d->buffer.size());
        std::ofstream(d->dir / (std::string(t->name) + ".f32"), std::ios::binary)
            .write((const char *) d->buffer.data(), (std::streamsize) d->buffer.size());
    } else if (wanted && t->type == GGML_TYPE_I32 && ggml_is_contiguous(t)) {
        std::vector<int32_t> ids(ggml_nelements(t));
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
        std::vector<float> f(ids.begin(), ids.end());
        std::ofstream(d->dir / (std::string(t->name) + ".f32"), std::ios::binary)
            .write((const char *) f.data(), (std::streamsize) (f.size() * sizeof(float)));
    }
    return true;
}

void decode(llama_context * ctx, const std::vector<llama_token> & ids, int pos0) {
    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    batch.n_tokens = (int32_t) ids.size();
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = pos0 + (int) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    const int status = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    llama_batch_free(batch);
    require(status == 0, "decode failed");
}

double nll(const float * row, int vocab, llama_token target) {
    double most = row[0];
    for (int i = 1; i < vocab; ++i) most = std::max(most, (double) row[i]);
    double sum = 0;
    for (int i = 0; i < vocab; ++i) sum += std::exp((double) row[i] - most);
    return most + std::log(sum) - (double) row[target];
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        require(argc >= 3, "usage: oracle MODEL OUTDIR [--prompts FILE --generate N] [--ppl FILE] [--dump NAMES]");
        const std::string model_path = argv[1];
        const std::filesystem::path out = argv[2];
        std::string prompts_path, ppl_path, dump_names;
        int generate = 0;
        for (int i = 3; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            if (a == "--prompts") prompts_path = argv[i + 1];
            else if (a == "--ppl") ppl_path = argv[i + 1];
            else if (a == "--generate") generate = std::atoi(argv[i + 1]);
            else if (a == "--dump") dump_names = argv[i + 1];
            else throw std::runtime_error("unknown argument " + a);
        }
        std::filesystem::create_directories(out);
        llama_backend_init();

        Dump dump;
        {
            std::stringstream names(dump_names);
            std::string n;
            while (std::getline(names, n, ',')) if (!n.empty()) dump.names.insert(n);
            dump.dir = out / "dump";
            if (!dump.names.empty()) std::filesystem::create_directories(dump.dir);
        }

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;  // plain reads: no mapped copy beside the GPU's
        mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
        const auto t_load = Clock::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
        require(bool(model), "model load failed");
        const double load_s = seconds(Clock::now() - t_load);
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        const int n_vocab = llama_vocab_n_tokens(vocab);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.no_perf = false;
        if (!dump.names.empty()) {
            cp.cb_eval = dump_callback;
            cp.cb_eval_user_data = &dump;
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp),
                                                                  llama_free);
        require(bool(ctx), "context creation failed");
        llama_memory_t mem = llama_get_memory(ctx.get());

        std::ostringstream summary;
        summary << "{\"model\":\"" << model_path << "\",\"load_seconds\":" << load_s
                << ",\"n_vocab\":" << n_vocab << ",\"fusion_disabled\":"
                << (std::getenv("GGML_CUDA_DISABLE_FUSION") ? "true" : "false");

        if (!prompts_path.empty()) {
            std::ifstream in(prompts_path);
            require(bool(in), "cannot read " + prompts_path);
            std::ofstream tokens_out(out / "prompts.tokens");
            std::ofstream generated_out(out / "generated.tokens");
            summary << ",\"prompts\":[";
            std::string line;
            bool first = true;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                const size_t tab = line.find('\t');
                require(tab != std::string::npos, "a prompt line without a name");
                const std::string name = line.substr(0, tab);
                const std::vector<llama_token> ids = tokenize(vocab, unescape(line.substr(tab + 1)));
                require((int) ids.size() <= 512, name + " is longer than one ubatch");
                tokens_out << name << "\t" << join(ids) << "\n";
                llama_memory_clear(mem, true);
                dump.active = first;
                const auto t0 = Clock::now();
                decode(ctx.get(), ids, 0);
                const double prefill = seconds(Clock::now() - t0);
                dump.active = false;
                std::vector<float> steps;
                std::vector<llama_token> generated;
                const float * row = llama_get_logits_ith(ctx.get(), (int32_t) ids.size() - 1);
                double decode_s = 0;
                for (int k = 0; k < generate; ++k) {
                    require(row != nullptr, "missing logits");
                    steps.insert(steps.end(), row, row + n_vocab);
                    const llama_token next = (llama_token) (std::max_element(row, row + n_vocab) - row);
                    generated.push_back(next);
                    if (k + 1 == generate) break;
                    const auto t1 = Clock::now();
                    decode(ctx.get(), {next}, (int) ids.size() + k);
                    decode_s += seconds(Clock::now() - t1);
                    row = llama_get_logits_ith(ctx.get(), 0);
                }
                std::ofstream(out / (name + ".logits.f32"), std::ios::binary)
                    .write((const char *) steps.data(), (std::streamsize) (steps.size() * sizeof(float)));
                generated_out << name << "\t" << join(generated) << "\n";
                summary << (first ? "" : ",") << "{\"name\":\"" << name << "\",\"prompt_tokens\":" << ids.size()
                        << ",\"prefill_seconds\":" << prefill << ",\"decode_seconds\":" << decode_s
                        << ",\"generated_tokens\":" << generated.size() << "}";
                std::printf("%s: %zu prompt tokens, prefill %.3f s, decode %.3f s\n", name.c_str(), ids.size(),
                            prefill, decode_s);
                first = false;
            }
            summary << "]";
        }

        if (!ppl_path.empty()) {
            std::ifstream in(ppl_path, std::ios::binary);
            require(bool(in), "cannot read " + ppl_path);
            std::stringstream text;
            text << in.rdbuf();
            std::vector<llama_token> ids = tokenize(vocab, text.str());
            if (ids.size() > 4096) ids.resize(4096);
            std::ofstream(out / "ppl.tokens") << "ppl\t" << join(ids) << "\n";
            llama_memory_clear(mem, true);
            std::vector<double> nlls;
            const auto t0 = Clock::now();
            for (size_t at = 0; at < ids.size(); at += 512) {
                const size_t rows = std::min<size_t>(512, ids.size() - at);
                decode(ctx.get(), std::vector<llama_token>(ids.begin() + (long) at, ids.begin() + (long) (at + rows)),
                       (int) at);
                for (size_t i = 0; i < rows; ++i) {
                    if (at + i + 1 < ids.size()) {
                        nlls.push_back(nll(llama_get_logits_ith(ctx.get(), (int32_t) i), n_vocab, ids[at + i + 1]));
                    }
                }
            }
            const double s = seconds(Clock::now() - t0);
            double mean = 0;
            for (double v : nlls) mean += v;
            mean /= (double) nlls.size();
            std::ofstream(out / "ppl.nll.f64", std::ios::binary)
                .write((const char *) nlls.data(), (std::streamsize) (nlls.size() * sizeof(double)));
            summary << ",\"ppl\":{\"tokens\":" << ids.size() << ",\"scored\":" << nlls.size()
                    << ",\"mean_nll\":" << mean << ",\"ppl\":" << std::exp(mean) << ",\"seconds\":" << s << "}";
            std::printf("perplexity %.4f over %zu tokens (%.2f s)\n", std::exp(mean), nlls.size(), s);
        }
        summary << "}\n";
        std::ofstream(out / "summary.json") << summary.str();
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "oracle: %s\n", e.what());
        return 1;
    }
}
