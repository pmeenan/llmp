// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// External C API harness for the pinned same-format llama.cpp comparator.
// MODEL INPUT OUTDIR prepare|prepare-rendered|score|score-unfused|score-ring CHUNK. Explicit IDs.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace {
void Require(bool good, const char* text) {
  if (!good) throw std::runtime_error(text);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(
        argc == 6,
        "usage: MODEL INPUT OUTDIR prepare|prepare-rendered|score|score-unfused|score-ring CHUNK");
    const bool prepare = std::string(argv[4]) == "prepare";
    const bool rendered = std::string(argv[4]) == "prepare-rendered";
    const bool unfused = std::string(argv[4]) == "score-unfused";
    const bool ring = std::string(argv[4]) == "score-ring";
    Require(prepare || rendered || unfused || ring || std::string(argv[4]) == "score",
            "unknown quality mode");
    const int chunk = std::stoi(argv[5]);
    Require(chunk >= 1 && chunk <= 1024, "quality chunk out of bounds");
    const auto* disable_fusion = std::getenv("GGML_CUDA_DISABLE_FUSION");
    Require(!std::getenv("GGML_CUDA_DISABLE_GRAPHS") &&
                (unfused ? disable_fusion && std::string(disable_fusion) == "1" : !disable_fusion),
            "explicit fusion/graph policy changed");
    const std::filesystem::path out(argv[3]);
    Require(std::filesystem::create_directory(out), "output must be new");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    mp.vocab_only = prepare || rendered;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    Require(bool(model), "model load failed");
    const auto* vocab = llama_model_get_vocab(model.get());
    Require(llama_vocab_n_tokens(vocab) == 262144, "unexpected vocabulary");
    if (prepare || rendered) {
      const auto length = std::filesystem::file_size(argv[2]);
      Require(length <= 4194304, "corpus exceeds bounded preparation input");
      std::string text(length, '\0');
      std::ifstream file(argv[2], std::ios::binary);
      file.read(text.data(), static_cast<std::streamsize>(length));
      Require(bool(file), "reading corpus failed");
      const int needed = -llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), nullptr,
                                         0, !rendered, rendered);
      Require(needed >= (rendered ? 1 : 1024) && needed <= 2097152,
              "unexpected corpus token count");
      std::vector<llama_token> ids(needed);
      Require(llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), ids.data(), needed,
                             !rendered, rendered) == needed,
              "tokenization failed");
      Require(ids.front() == 2, "missing explicit BOS");
      if (rendered) Require(std::count(ids.begin(), ids.end(), 2) == 1, "repeated rendered BOS");
      for (auto id : ids) Require(id >= 0 && id < 262144, "prepared ID outside vocabulary");
      std::ofstream tokens(out / "ids.i32", std::ios::binary);
      const int supplied = rendered ? needed : 1024;
      tokens.write(reinterpret_cast<const char*>(ids.data()), supplied * 4);
      tokens.close();
      Require(bool(tokens), "writing IDs failed");
      // Include corpus provenance separately; the exact supplied prefix is
      // the checked integer-token file, not a guessed UTF-8 truncation.
      std::cout << "QUALITY_PREPARE corpus_bytes=" << length << " total_tokens=" << needed;
      if (rendered)
        std::cout << " supplied_tokens=" << supplied << " bos=2 rendered=1\n";
      else
        std::cout << " supplied_tokens=1024 scored_targets=1023 bos=2\n";
      model.reset();
      llama_backend_free();
      return 0;
    }
    Require(std::filesystem::file_size(argv[2]) == 1024 * 4, "unexpected input ID length");
    std::array<llama_token, 1024> ids{};
    std::ifstream file(argv[2], std::ios::binary);
    file.read(reinterpret_cast<char*>(ids.data()), sizeof(ids));
    Require(bool(file) && ids.front() == 2, "reading input IDs failed");
    for (auto id : ids) Require(id >= 0 && id < 262144, "input ID outside vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = chunk;
    cp.n_ubatch = chunk;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = false;
    if (ring) {
      cp.swa_full = false;
      cp.kv_unified = false;
    }
    std::cout << "QUALITY_PROFILE context=" << cp.n_ctx << " batch=" << cp.n_batch
              << " ubatch=" << cp.n_ubatch << " swa_full=" << cp.swa_full
              << " kv_unified=" << cp.kv_unified << " all_outputs=1\n";
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    Require(bool(ctx), "context creation failed");
    llama_batch batch = llama_batch_init(chunk, 0, 1);
    std::vector<float> published(static_cast<std::size_t>(chunk) * 262144);
    std::ofstream logits(out / "logits.f32", std::ios::binary);
    for (int first = 0; first < 1024; first += chunk) {
      batch.n_tokens = std::min(chunk, 1024 - first);
      for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = ids[first + i];
        batch.pos[i] = first + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
      }
      Require(llama_decode(ctx.get(), batch) == 0, "decode failed");
      for (int i = 0; i < batch.n_tokens; ++i) {
        const auto* row = llama_get_logits_ith(ctx.get(), i);
        Require(row != nullptr, "missing completed full-vocabulary likelihood row");
        std::copy_n(row, 262144, published.begin() + static_cast<std::size_t>(i) * 262144);
      }
      logits.write(reinterpret_cast<const char*>(published.data()),
                   static_cast<std::streamsize>(batch.n_tokens) * 262144 * 4);
      Require(bool(logits), "writing likelihood rows failed");
      std::cout << "QUALITY_CHUNK first=" << first << " rows=" << batch.n_tokens
                << " completed=" << first + batch.n_tokens << '\n';
    }
    logits.flush();
    Require(bool(logits), "flushing likelihood rows failed");
    llama_batch_free(batch);
    ctx.reset();
    model.reset();
    llama_backend_free();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
