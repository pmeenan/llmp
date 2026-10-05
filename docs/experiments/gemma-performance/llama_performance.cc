// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// External C API harness for the pinned same-format llama.cpp comparator.
// MODEL OUTDIR warm|quality. Explicit token IDs; no template or EOG stop.
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
using Clock = std::chrono::steady_clock;
constexpr std::array<llama_token, 6> kPrompt = {2, 818, 5279, 529, 7001, 563};
constexpr int kSteps = 32;
void Require(bool good, const char* text) {
  if (!good) throw std::runtime_error(text);
}
double Seconds(Clock::duration time) { return std::chrono::duration<double>(time).count(); }
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 4, "usage: MODEL OUTDIR warm|quality|quality-all|quality-unfused");
    const std::string mode(argv[3]);
    const bool warm = mode == "warm", all_heads = mode == "quality-all";
    const bool unfused_diagnostic = mode == "quality-unfused";
    Require(warm || mode == "quality" || all_heads || unfused_diagnostic, "unknown mode");
    // Presence/values of these switches would change the comparator policy.
    const char* disabled = std::getenv("GGML_CUDA_DISABLE_FUSION");
    Require(unfused_diagnostic ? disabled && std::string(disabled) == "1" : !disabled,
            "fusion-disable environment disagrees with mode");
    Require(!std::getenv("GGML_CUDA_DISABLE_GRAPHS"), "graph-disable environment present");
    const std::filesystem::path out(argv[2]);
    std::filesystem::create_directories(out);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    Require(bool(model), "model load failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    Require(vocab == 262144, "unexpected vocabulary");
    auto cp = llama_context_default_params();
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
    Require(bool(ctx), "context creation failed");
    // Reuse the batch container. Only the last input row requests a head,
    // matching native frontier-head prefill and full-vocabulary publication.
    llama_batch batch = llama_batch_init(128, 0, 1);
    std::vector<float> published(static_cast<std::size_t>(vocab));
    const auto decode = [&](const llama_token* ids, int rows, int past) {
      batch.n_tokens = rows;
      for (int i = 0; i < rows; ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = all_heads || i == rows - 1;
      }
      Require(llama_decode(ctx.get(), batch) == 0, "decode failed");
      // The public getter synchronizes before returning its host output.
      const float* logits = llama_get_logits_ith(ctx.get(), -1);
      Require(logits != nullptr, "missing full logits");
      std::copy_n(logits, vocab, published.begin());
    };
    const auto best = [&]() -> llama_token {
      return static_cast<llama_token>(std::max_element(published.begin(), published.end()) -
                                      published.begin());
    };
    auto memory = llama_get_memory(ctx.get());
    llama_memory_clear(memory, true);
    decode(kPrompt.data(), static_cast<int>(kPrompt.size()), 0);
    int past = static_cast<int>(kPrompt.size());
    if (warm) {
      for (int i = 0; i < 8; ++i) {
        const auto token = best();
        decode(&token, 1, past++);
      }
      llama_memory_clear(memory, true);
      decode(kPrompt.data(), static_cast<int>(kPrompt.size()), 0);
      past = static_cast<int>(kPrompt.size());
      // The first stable call builds, the second captures, the third replays.
      for (int i = 0; i < 3; ++i) {
        const auto seed = best();
        decode(&seed, 1, past++);
        std::cout << "LLAMA_TIMED_PREFIX appended=" << seed << " past=" << past << '\n';
      }
    }
    const auto before = llama_perf_context(ctx.get());
    std::array<llama_token, kSteps> chosen{};
    const auto started = Clock::now();
    for (int i = 0; i < kSteps; ++i) {
      const auto token = best();
      chosen[static_cast<std::size_t>(i)] = token;
      if (warm) {
        decode(&token, 1, past++);
      } else {
        std::ofstream file(out / ("logits-" + std::to_string(i) + ".f32"), std::ios::binary);
        file.write(reinterpret_cast<const char*>(published.data()),
                   static_cast<std::streamsize>(published.size() * sizeof(float)));
        Require(bool(file), "writing full logits failed");
        if (i + 1 < kSteps) decode(&token, 1, past++);
      }
    }
    const double elapsed = Seconds(Clock::now() - started);
    const auto after = llama_perf_context(ctx.get());
    std::cout << (warm ? "LLAMA_WARM" : "LLAMA_QUALITY") << " seconds=" << elapsed
              << " completed_chunks=" << (warm ? kSteps : kSteps - 1)
              << " ggml_graph_reused_delta=" << after.n_reused - before.n_reused
              << " context=" << llama_n_ctx(ctx.get())
              << " fusion=" << (unfused_diagnostic ? "disabled-diagnostic" : "enabled")
              << " graphs=allowed head=" << (all_heads ? "all" : "frontier") << '\n';
    for (int i = 0; i < kSteps; ++i)
      std::cout << "LLAMA_TOKEN step=" << i << " id=" << chosen[static_cast<std::size_t>(i)]
                << '\n';
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
