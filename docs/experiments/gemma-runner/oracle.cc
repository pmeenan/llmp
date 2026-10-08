// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External pinned llama.cpp reference only; not linked into llmpalooza.
// Gemma26 same-format smoke control: explicit BOS, 32 raw greedy steps,
// two cleared-memory repeats and 26 teacher-forced targets. See README.md.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace {

using Clock = std::chrono::steady_clock;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

std::vector<llama_token> parse_ids(const std::string& text) {
  std::vector<llama_token> ids;
  std::istringstream in(text);
  long long v = 0;
  while (in >> v) {
    require(v >= 0 && v < (1LL << 31), "a token ID out of range");
    ids.push_back((llama_token)v);
  }
  return ids;
}

std::string join(const std::vector<llama_token>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) {
    s += (i ? " " : "") + std::to_string(ids[i]);
  }
  return s;
}

void decode(llama_context* ctx, const std::vector<llama_token>& ids, int pos0) {
  llama_batch batch = llama_batch_init((int32_t)ids.size(), 0, 1);
  batch.n_tokens = (int32_t)ids.size();
  for (size_t i = 0; i < ids.size(); ++i) {
    batch.token[i] = ids[i];
    batch.pos[i] = pos0 + (int)i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = true;
  }
  const int status = llama_decode(ctx, batch);
  llama_synchronize(ctx);
  llama_batch_free(batch);
  require(status == 0, "decode failed");
}

double nll(const float* row, int vocab, llama_token target) {
  double most = row[0];
  for (int i = 1; i < vocab; ++i) most = std::max(most, (double)row[i]);
  double sum = 0;
  for (int i = 0; i < vocab; ++i) sum += std::exp((double)row[i] - most);
  return most + std::log(sum) - (double)row[target];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc >= 3, "usage: oracle MODEL OUTDIR [--prompts FILE --generate N] [--ppl FILE]");
    const std::string model_path = argv[1];
    const std::filesystem::path out = argv[2];
    std::string prompts_path, ppl_path;
    int generate = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
      const std::string a = argv[i];
      if (a == "--prompts")
        prompts_path = argv[i + 1];
      else if (a == "--ppl")
        ppl_path = argv[i + 1];
      else if (a == "--generate")
        generate = std::atoi(argv[i + 1]);
      else
        throw std::runtime_error("unknown argument " + a);
    }
    std::filesystem::create_directories(out);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;  // plain reads: no mapped copy beside the GPU's
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    const auto t_load = Clock::now();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
    require(bool(model), "model load failed");
    const double load_s = seconds(Clock::now() - t_load);
    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const auto tokenize = [&](const std::string& text, bool add_special) {
      int n =
          llama_tokenize(vocab, text.data(), (int32_t)text.size(), nullptr, 0, add_special, false);
      require(n < 0, "token count failed");
      std::vector<llama_token> ids((size_t)-n);
      n = llama_tokenize(vocab, text.data(), (int32_t)text.size(), ids.data(), (int32_t)ids.size(),
                         add_special, false);
      require(n == (int)ids.size(), "tokenization failed");
      return ids;
    };
    const std::string literal = "The capital of France is";
    const std::string forced =
        "Paris is the capital of France. The Seine flows through the city. A triangle has three "
        "sides. Two plus two equals four.";
    const auto plain = tokenize(literal, false);
    const auto with_bos = tokenize(literal, true);
    std::ofstream(out / "plain_ids.txt") << join(plain) << "\n";
    std::ofstream(out / "with_bos_ids.txt") << join(with_bos) << "\n";
    require(with_bos.front() == llama_vocab_bos(vocab), "expected explicit BOS");
    prompts_path = (out / "prompts.tsv").string();
    std::ofstream(prompts_path) << "literal_first\t" << join(with_bos) << "\n"
                                << "literal_repeat\t" << join(with_bos) << "\n";
    ppl_path = (out / "ppl.tsv").string();
    std::ofstream(ppl_path) << "forced\t" << join(tokenize(forced, true)) << "\n";
    generate = 32;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    require(bool(ctx), "context creation failed");
    llama_memory_t mem = llama_get_memory(ctx.get());

    std::ostringstream summary;
    summary << "{\"model\":\"" << model_path << "\",\"load_seconds\":" << load_s
            << ",\"n_vocab\":" << n_vocab << ",\"fusion_disabled\":"
            << (std::getenv("GGML_CUDA_DISABLE_FUSION") ? "true" : "false");

    if (!prompts_path.empty()) {
      std::ifstream in(prompts_path);
      require(bool(in), "cannot read " + prompts_path);
      std::ofstream generated_out(out / "generated.tokens");
      summary << ",\"prompts\":[";
      std::string line;
      bool first = true;
      while (std::getline(in, line)) {
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        require(tab != std::string::npos, "a prompt line without a name");
        const std::string name = line.substr(0, tab);
        const std::vector<llama_token> ids = parse_ids(line.substr(tab + 1));
        require(!ids.empty() && (int)ids.size() <= 128, name + " is not one ubatch");
        for (const llama_token id : ids)
          require(id < n_vocab, name + ": a token outside the vocabulary");
        llama_memory_clear(mem, true);
        const auto t0 = Clock::now();
        decode(ctx.get(), ids, 0);
        const double prefill = seconds(Clock::now() - t0);
        std::vector<float> steps;
        std::vector<llama_token> generated;
        const float* row = llama_get_logits_ith(ctx.get(), (int32_t)ids.size() - 1);
        double decode_s = 0;
        for (int k = 0; k < generate; ++k) {
          require(row != nullptr, "missing logits");
          steps.insert(steps.end(), row, row + n_vocab);
          const llama_token next = (llama_token)(std::max_element(row, row + n_vocab) - row);
          generated.push_back(next);
          if (k + 1 == generate) break;
          const auto t1 = Clock::now();
          decode(ctx.get(), {next}, (int)ids.size() + k);
          decode_s += seconds(Clock::now() - t1);
          row = llama_get_logits_ith(ctx.get(), 0);
        }
        std::ofstream(out / (name + ".logits.f32"), std::ios::binary)
            .write((const char*)steps.data(), (std::streamsize)(steps.size() * sizeof(float)));
        generated_out << name << "\t" << join(generated) << "\n";
        summary << (first ? "" : ",") << "{\"name\":\"" << name
                << "\",\"prompt_tokens\":" << ids.size() << ",\"prefill_seconds\":" << prefill
                << ",\"decode_seconds\":" << decode_s
                << ",\"generated_tokens\":" << generated.size() << "}";
        std::printf("%s: %zu prompt tokens, prefill %.3f s, decode %.3f s\n", name.c_str(),
                    ids.size(), prefill, decode_s);
        first = false;
      }
      summary << "]";
    }

    if (!ppl_path.empty()) {
      std::ifstream in(ppl_path);
      require(bool(in), "cannot read " + ppl_path);
      std::string line;
      require(bool(std::getline(in, line)), "an empty perplexity file");
      const size_t tab = line.find('\t');
      require(tab != std::string::npos, "a perplexity line without a name");
      std::vector<llama_token> ids = parse_ids(line.substr(tab + 1));
      require(ids.size() >= 2 && ids.size() <= 8192,
              "the perplexity text does not fit the context");
      llama_memory_clear(mem, true);
      std::vector<double> nlls;
      std::ofstream forced_logits(out / "forced.logits.f32", std::ios::binary);
      const auto t0 = Clock::now();
      for (size_t at = 0; at < ids.size(); at += 128) {
        const size_t rows = std::min<size_t>(128, ids.size() - at);
        decode(ctx.get(),
               std::vector<llama_token>(ids.begin() + (long)at, ids.begin() + (long)(at + rows)),
               (int)at);
        for (size_t i = 0; i < rows; ++i) {
          if (at + i + 1 < ids.size()) {
            const float* forced_row = llama_get_logits_ith(ctx.get(), (int32_t)i);
            forced_logits.write((const char*)forced_row, (std::streamsize)n_vocab * sizeof(float));
            nlls.push_back(nll(forced_row, n_vocab, ids[at + i + 1]));
          }
        }
      }
      const double s = seconds(Clock::now() - t0);
      double mean = 0;
      for (double v : nlls) mean += v;
      mean /= (double)nlls.size();
      std::ofstream(out / "ppl.nll.f64", std::ios::binary)
          .write((const char*)nlls.data(), (std::streamsize)(nlls.size() * sizeof(double)));
      summary << ",\"ppl\":{\"tokens\":" << ids.size() << ",\"scored\":" << nlls.size()
              << ",\"mean_nll\":" << mean << ",\"ppl\":" << std::exp(mean) << ",\"seconds\":" << s
              << "}";
      std::printf("perplexity %.4f over %zu tokens (%.2f s)\n", std::exp(mean), nlls.size(), s);
    }
    summary << "}\n";
    std::ofstream(out / "summary.json") << summary.str();
    llama_backend_free();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "oracle: %s\n", e.what());
    return 1;
  }
}
