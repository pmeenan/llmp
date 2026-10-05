// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// External C API harness for the pinned same-format llama.cpp comparator.
// MODEL IDS_I32 NEW_OUTDIR; dense31 C1/8K/u256 ring-cache recipe only.
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
    Require(argc == 4, "usage: approved31_MODEL IDS_I32 NEW_OUTDIR");
    constexpr int chunk = 256;
    Require(!std::getenv("GGML_CUDA_DISABLE_GRAPHS") && !std::getenv("GGML_CUDA_DISABLE_FUSION"),
            "production graph/fusion policy changed");
    const std::filesystem::path out(argv[3]);
    Require(std::filesystem::create_directory(out), "output must be new");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    mp.vocab_only = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    Require(bool(model), "model load failed");
    const auto* vocab = llama_model_get_vocab(model.get());
    Require(llama_vocab_n_tokens(vocab) == 262144, "unexpected vocabulary");
    Require(llama_model_n_layer(model.get()) == 60 && llama_model_n_embd(model.get()) == 5376,
            "model is outside the closed dense31 profile");
    Require(std::filesystem::file_size(argv[2]) == 8227 * 4, "unexpected input ID length");
    std::array<llama_token, 8227> ids{};
    std::ifstream file(argv[2], std::ios::binary);
    file.read(reinterpret_cast<char*>(ids.data()), sizeof(ids));
    Require(bool(file) && ids.front() == 2, "reading input IDs failed");
    for (auto id : ids) Require(id >= 0 && id < 262144, "input ID outside vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = 16384;
    cp.n_batch = 8192;
    cp.n_ubatch = chunk;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = false;
    cp.swa_full = false;
    cp.kv_unified = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    Require(bool(ctx), "context creation failed");
    Require(llama_n_ctx(ctx.get()) == 16384 && llama_n_ubatch(ctx.get()) == 256,
            "effective context/ubatch differs from ring recipe");
    llama_batch batch = llama_batch_init(8192, 0, 1);
    std::vector<float> published(262144);
    const auto decode = [&](int first, int rows) {
      batch.n_tokens = rows;
      for (int i = 0; i < rows; ++i) {
        batch.token[i] = ids[first + i];
        batch.pos[i] = first + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == rows - 1;
      }
      Require(llama_decode(ctx.get(), batch) == 0, "decode failed");
      const auto* head = llama_get_logits_ith(ctx.get(), -1);
      Require(head != nullptr, "missing completed full vocabulary head");
      std::copy_n(head, 262144, published.begin());
    };
    const auto save = [&](const char* name) {
      std::ofstream file(out / name, std::ios::binary);
      file.write(reinterpret_cast<const char*>(published.data()), 262144 * 4);
      file.flush();
      Require(bool(file), "saving completed head failed");
    };
    // Both arms page/warm weights on these six discarded input rows.
    decode(0, 6);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    const auto before_prefill = std::chrono::steady_clock::now();
    decode(0, 8192);
    const double prefill =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - before_prefill).count();
    save("prefill.f32");
    for (int i = 0; i < 3; ++i) {
      decode(8192 + i, 1);
      std::cout << "PREFILL_TIMED_PREFIX appended=" << ids[8192 + i] << " past=" << 8193 + i
                << '\n';
    }
    const auto before = llama_perf_context(ctx.get());
    std::array<llama_token, 32> chosen{};
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < 32; ++i) {
      chosen[i] = static_cast<llama_token>(std::max_element(published.begin(), published.end()) -
                                           published.begin());
      decode(8195 + i, 1);
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto after = llama_perf_context(ctx.get());
    save("final.f32");
    std::cout << "PREFILL_REFERENCE prefill_seconds=" << prefill
              << " prefill_rows=8192 intermediate_heads=0 decode_seconds=" << elapsed
              << " decode_chunks=32 timed_start=8195 completed=8227 context="
              << llama_n_ctx(ctx.get()) << " batch=8192 ubatch=" << chunk
              << " fusion=enabled graphs=allowed swa_full=false kv_unified=false cache_recipe=ring"
              << " ggml_graph_reused_delta=" << after.n_reused - before.n_reused << '\n';
    for (int i = 0; i < 32; ++i)
      std::cout << "PREFILL_TOKEN step=" << i << " argmax=" << chosen[i]
                << " forced=" << ids[8195 + i] << '\n';
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
