// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// MODEL OUTPUT_DIR OWNERS UBATCH [IDS_I32]. Real independent-sequence stock C API batch.
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
void Require(bool good, const char* detail) {
  if (!good) throw std::runtime_error(detail);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 5 || argc == 6, "MODEL OUTPUT_DIR OWNERS UBATCH [IDS_I32]");
    std::vector<std::int32_t> supplied;
    if (argc == 6) {
      supplied.resize(1024);
      std::ifstream file(argv[5], std::ios::binary);
      file.read(reinterpret_cast<char*>(supplied.data()), 1024 * sizeof(std::int32_t));
      Require(bool(file) && file.peek() == std::char_traits<char>::eof() && supplied[0] == 2 &&
                  std::ranges::all_of(supplied, [](auto id) { return id >= 0 && id < 262144; }),
              "invalid 1024-token supplied prefix");
    }
    const int prompt_rows = supplied.empty() ? 6 : 64;
    const auto count = std::stoi(argv[3]), ubatch = std::stoi(argv[4]);
    Require(count >= 1 && count <= 12 && ubatch >= count && ubatch <= 128,
            "unbounded owners or physical ubatch");
    Require(!std::getenv("GGML_CUDA_DISABLE_GRAPHS") && !std::getenv("GGML_CUDA_DISABLE_FUSION"),
            "stock graphs/fusion disabled");
    const std::filesystem::path out = argv[2];
    Require(std::filesystem::create_directory(out), "output must be new");
    if (!supplied.empty()) {
      std::ofstream file(out / "inputs.i32", std::ios::binary);
      file.write(reinterpret_cast<const char*>(supplied.data()), 1024 * sizeof(std::int32_t));
      file.flush();
      Require(bool(file), "writing supplied IDs failed");
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    Require(bool(model), "model load failed");
    Require(llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == 262144,
            "unexpected vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = static_cast<std::uint32_t>(count * 256);
    cp.n_batch = 128;
    cp.n_ubatch = static_cast<std::uint32_t>(ubatch);
    cp.n_seq_max = static_cast<std::uint32_t>(count);
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    Require(bool(ctx), "context creation failed");
    Require(llama_n_ctx_seq(ctx.get()) >= 256 &&
                llama_n_seq_max(ctx.get()) == static_cast<std::uint32_t>(count),
            "context was accidentally split below the per-owner bound");
    llama_batch batch = llama_batch_init(128, 0, 1);
    constexpr int vocab = 262144, steps = 32;
    constexpr std::array<llama_token, 6> prompt{2, 818, 5279, 529, 7001, 563};
    constexpr std::array<llama_token, 3> seeds{45518, 107, 101};
    std::vector<std::vector<float>> heads(count, std::vector<float>(vocab));
    std::array<int, 12> past{};
    std::array<std::array<llama_token, 12>, steps> chosen{};
    std::vector<float> published(static_cast<std::size_t>(steps) * count * vocab);
    const auto prefill = [&]() {
      llama_memory_clear(llama_get_memory(ctx.get()), true);
      for (int owner = 0; owner < count; ++owner) {
        batch.n_tokens = prompt_rows + owner;
        for (int i = 0; i < batch.n_tokens; ++i) {
          batch.token[i] = supplied.empty() ? (i < 6 ? prompt[i] : 563) : supplied[i];
          batch.pos[i] = i;
          batch.n_seq_id[i] = 1;
          batch.seq_id[i][0] = owner;
          batch.logits[i] = i == batch.n_tokens - 1;
        }
        Require(llama_decode(ctx.get(), batch) == 0, "independent owner prefill failed");
        const auto* row = llama_get_logits_ith(ctx.get(), batch.n_tokens - 1);
        Require(row != nullptr, "missing owner prefill head");
        std::copy_n(row, vocab, heads[owner].begin());
        past[owner] = batch.n_tokens;
      }
    };
    const auto wave = [&](int step) {
      batch.n_tokens = count;
      for (int owner = 0; owner < count; ++owner) {
        batch.token[owner] = supplied.empty() ? seeds[(step + owner) % seeds.size()]
                                              : supplied[prompt_rows + owner + step];
        batch.pos[owner] = past[owner];
        batch.n_seq_id[owner] = 1;
        batch.seq_id[owner][0] = owner;
        batch.logits[owner] = true;
      }
      Require(llama_decode(ctx.get(), batch) == 0, "joined multi-sequence decode failed");
      for (int owner = 0; owner < count; ++owner) {
        const auto* row = llama_get_logits_ith(ctx.get(), owner);
        Require(row != nullptr, "missing completed owner head");
        std::copy_n(row, vocab, heads[owner].begin());
        ++past[owner];
      }
    };
    prefill();
    for (int step = 0; step < 8; ++step) wave(step);
    prefill();
    for (int step = 0; step < 3; ++step) wave(step);
    const auto before = llama_perf_context(ctx.get());
    const auto started = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
      wave(step + 3);
      for (int owner = 0; owner < count; ++owner) {
        chosen[step][owner] = static_cast<llama_token>(
            std::max_element(heads[owner].begin(), heads[owner].end()) - heads[owner].begin());
        std::copy(heads[owner].begin(), heads[owner].end(),
                  published.begin() + (static_cast<std::size_t>(step) * count + owner) * vocab);
      }
    }
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto after = llama_perf_context(ctx.get());
    std::ofstream file(out / "heads.f32", std::ios::binary);
    file.write(reinterpret_cast<const char*>(published.data()),
               static_cast<std::streamsize>(published.size() * sizeof(float)));
    file.flush();
    Require(bool(file), "writing completed owner heads failed");
    std::cout << "JOINED_REFERENCE owners=" << count << " seconds=" << elapsed
              << " completed_waves=" << steps << " completed_units=" << steps * count
              << " first_past=" << prompt_rows + 3
              << " input_mode=" << (supplied.empty() ? "synthetic" : "supplied")
              << " unequal_past=" << (count > 1) << " context=" << llama_n_ctx(ctx.get())
              << " per_sequence_context=" << llama_n_ctx_seq(ctx.get())
              << " batch=128 ubatch=" << ubatch << " seq_max=" << llama_n_seq_max(ctx.get())
              << " fusion=enabled graphs=allowed reused_delta=" << after.n_reused - before.n_reused
              << '\n';
    for (int step = 0; step < steps; ++step)
      for (int owner = 0; owner < count; ++owner)
        std::cout << "JOINED_TOKEN step=" << step << " owner=" << owner
                  << " argmax=" << chosen[step][owner] << " forced="
                  << (supplied.empty() ? seeds[(step + 3 + owner) % seeds.size()]
                                       : supplied[prompt_rows + owner + step + 3])
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
