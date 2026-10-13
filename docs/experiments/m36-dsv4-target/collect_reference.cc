// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External pinned llama.cpp reference only. Derived from dsv4-native/oracle.cc.
// Fixed natural corpus, BOS plus256 input tokens;256 next-token losses.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace {
constexpr std::size_t kInputs = 257, kChunk = 128, kTextBytes = 17210;
constexpr int kVocabulary = 129280;
static_assert(sizeof(llama_token) == 4 && std::endian::native == std::endian::little);
void Require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}
std::string Text(const std::filesystem::path& path) {
  Require(std::filesystem::file_size(path) == kTextBytes, "wrong corpus size");
  std::string text(kTextBytes, '\0');
  std::ifstream in(path, std::ios::binary);
  Require(bool(in.read(text.data(), static_cast<std::streamsize>(text.size()))),
          "corpus read failed");
  return text;
}
std::vector<llama_token> Tokens(const llama_vocab* vocab, const std::string& text) {
  std::vector<llama_token> ids(text.size() + 16);
  const int count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                   ids.data(), static_cast<int32_t>(ids.size()), false, false);
  Require(count >= static_cast<int>(kInputs - 1) && count <= static_cast<int>(ids.size()),
          "tokenizer count outside fixed bounds");
  ids.resize(static_cast<std::size_t>(count));
  const auto bos = llama_vocab_bos(vocab);
  Require(bos >= 0 && bos < kVocabulary, "invalid BOS");
  ids.insert(ids.begin(), bos);
  for (auto id : ids) Require(id >= 0 && id < kVocabulary, "invalid token");
  return ids;
}
void Decode(llama_context* ctx, const llama_token* ids, std::size_t count, std::size_t past) {
  auto batch = llama_batch_init(static_cast<int32_t>(count), 0, 1);
  Require(batch.token && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits,
          "batch allocation failed");
  batch.n_tokens = static_cast<int32_t>(count);
  for (std::size_t i = 0; i < count; ++i) {
    batch.token[i] = ids[i];
    batch.pos[i] = static_cast<llama_pos>(past + i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = true;
  }
  const int status = llama_decode(ctx, batch);
  llama_synchronize(ctx);
  llama_batch_free(batch);
  Require(status == 0, "decode failed");
}
double Loss(const float* row, llama_token target) {
  Require(row != nullptr, "missing logits");
  double most = -INFINITY;
  for (int i = 0; i < kVocabulary; ++i) {
    Require(std::isfinite(row[i]), "nonfinite head");
    most = std::max(most, static_cast<double>(row[i]));
  }
  double sum = 0;
  for (int i = 0; i < kVocabulary; ++i) sum += std::exp(static_cast<double>(row[i]) - most);
  const double loss = most + std::log(sum) - row[target];
  Require(std::isfinite(loss) && loss >= 0, "invalid NLL");
  return loss;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(
        argc == 5 || argc == 6,
        "usage: collector prepare MODEL CORPUS OUTDIR | collector score MODEL CORPUS OUTDIR IDS");
    const std::string mode = argv[1];
    const bool prepare = mode == "prepare";
    Require((prepare && argc == 5) || (mode == "score" && argc == 6), "invalid mode");
    for (const char* key : {"GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_DISABLE_GRAPHS",
                            "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS"})
      Require(std::getenv(key) == nullptr, "unexpected reference override");
    const auto text = Text(argv[3]);
    const std::filesystem::path out = argv[4];
    Require(!std::filesystem::exists(out), "output already exists");
    llama_backend_init();
    {
      auto mp = llama_model_default_params();
      mp.vocab_only = prepare;
      mp.n_gpu_layers = prepare ? 0 : -1;
      mp.split_mode = LLAMA_SPLIT_MODE_NONE;
      mp.load_mode = LLAMA_LOAD_MODE_NONE;
      mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
      mp.load_mtp = false;
      std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
          llama_model_load_from_file(argv[2], mp), llama_model_free);
      Require(bool(model), "model load failed");
      const auto* vocab = llama_model_get_vocab(model.get());
      Require(llama_vocab_n_tokens(vocab) == kVocabulary, "wrong vocabulary");
      auto ids = Tokens(vocab, text);
      const auto full_count = ids.size();
      ids.resize(kInputs);
      if (!prepare) {
        Require(std::filesystem::file_size(argv[5]) == kInputs * sizeof(llama_token),
                "wrong captured ID extent");
        std::array<llama_token, kInputs> supplied{};
        std::ifstream in(argv[5], std::ios::binary);
        Require(bool(in.read(reinterpret_cast<char*>(supplied.data()), sizeof(supplied))),
                "captured ID read failed");
        Require(std::equal(ids.begin(), ids.end(), supplied.begin()),
                "reference tokenizer disagrees with captured IDs");
      }
      std::vector<double> losses;
      std::size_t finite_rows = 0;
      if (!prepare) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 3072;
        cp.n_batch = kChunk;
        cp.n_ubatch = kChunk;
        cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.swa_full = false;
        cp.kv_unified = false;
        cp.no_perf = true;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        Require(bool(ctx), "context creation failed");
        Require(llama_n_ctx(ctx.get()) == 3072, "context differs");
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        losses.reserve(kInputs - 1);
        for (std::size_t at = 0; at < ids.size(); at += kChunk) {
          const auto rows = std::min(kChunk, ids.size() - at);
          Decode(ctx.get(), ids.data() + at, rows, at);
          for (std::size_t row = 0; row < rows; ++row) {
            const auto target = at + row + 1 < ids.size() ? ids[at + row + 1] : ids.back();
            const auto loss =
                Loss(llama_get_logits_ith(ctx.get(), static_cast<int32_t>(row)), target);
            ++finite_rows;
            if (at + row + 1 < ids.size()) losses.push_back(loss);
          }
        }
        Require(finite_rows == kInputs && losses.size() == kInputs - 1, "incomplete coordinates");
        llama_synchronize(ctx.get());
      }
      // Successful retirement precedes output publication. Wrapper publishes final receipt only on
      // exit0.
      double mean = 0;
      for (const auto loss : losses) mean += loss;
      if (!prepare) mean /= static_cast<double>(losses.size());
      Require(std::isfinite(mean) && std::isfinite(std::exp(mean)), "aggregate overflow");
      model.reset();
      std::filesystem::create_directory(out);
      std::ofstream tokens(out / "tokens.i32le", std::ios::binary);
      tokens.write(reinterpret_cast<const char*>(ids.data()),
                   static_cast<std::streamsize>(ids.size() * sizeof(llama_token)));
      tokens.close();
      Require(bool(tokens), "token write failed");
      if (!prepare) {
        std::ofstream values(out / "losses.tsv");
        values << std::setprecision(17);
        for (std::size_t i = 0; i < losses.size(); ++i)
          values << "0\t" << i + 1 << '\t' << ids[i + 1] << '\t' << losses[i] << '\n';
        values.close();
        Require(bool(values), "loss write failed");
      }
      std::ofstream summary(out / "summary.json");
      summary << std::setprecision(17) << "{\"mode\":\"" << mode
              << "\",\"full_token_count\":" << full_count
              << ",\"input_count\":257,\"vocabulary\":129280,\"bos\":" << ids[0]
              << ",\"targets\":" << losses.size() << ",\"finite_rows\":" << finite_rows
              << ",\"context\":3072,\"chunk\":128,\"cache_k\":\"f16\",\"cache_v\":\"f16\",\"fusion_"
                 "disabled\":false,\"swa_full\":false,\"kv_unified\":false,\"mean_nll\":"
              << mean << ",\"perplexity\":" << std::exp(mean) << "}\n";
      summary.close();
      Require(bool(summary), "summary write failed");
    }
    llama_backend_free();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "collector: %s\n", e.what());
    return 1;
  }
}
