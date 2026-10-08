// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Original public API, ordinary greedy chain and independent sequence caches.
// MODEL INPUT_4x8192_I32 NEW_OUT gemma26|gemma31 1|4 quality|cycles [NATIVE_FIRST]
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "llama.h"

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kVocab = 262144, kRows = 8192, kPrefix = 8063, kOutputs = 129;
void Require(bool good, const char* message) {
  if (!good) throw std::runtime_error(message);
}
template <class T>
void Save(const fs::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  Require(bool(file), "exclusive output refused");
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  Require(bool(file), "complete output failed");
}
double Seconds(Clock::duration duration) { return std::chrono::duration<double>(duration).count(); }
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 7 || argc == 8,
            "MODEL INPUT_4x8192_I32 NEW_OUT gemma26|gemma31 1|4 quality|cycles [NATIVE_FIRST]");
    std::cout.precision(12);
    const std::string_view profile = argv[4], supplied_owners = argv[5], mode = argv[6];
    Require(profile == "gemma26" || profile == "gemma31", "unknown profile");
    Require(supplied_owners == "1" || supplied_owners == "4", "unknown owner count");
    Require(mode == "quality" || mode == "cycles", "unknown mode");
    Require(!std::getenv("GGML_CUDA_DISABLE_FUSION") && !std::getenv("GGML_CUDA_DISABLE_GRAPHS"),
            "stock override present");
    const bool dense = profile == "gemma31", quality = mode == "quality";
    const int owners = supplied_owners == "1" ? 1 : 4, chunk = dense ? 256 : 1024;
    Require(argc == (quality ? 8 : 7), "quality requires frozen native choices only");
    std::array<std::array<llama_token, kOutputs>, 4> supplied{};
    if (quality) {
      for (int owner = 0; owner < owners; ++owner) {
        std::ifstream chosen(fs::path(argv[7]) / ("tokens-" + std::to_string(owner) + ".i32"),
                             std::ios::binary);
        chosen.read(reinterpret_cast<char*>(supplied[owner].data()), sizeof(supplied[owner]));
        Require(bool(chosen) && chosen.peek() == std::char_traits<char>::eof() &&
                    std::ranges::all_of(supplied[owner],
                                        [](auto token) { return token >= 0 && token < kVocab; }),
                "exact frozen native 129 choices required");
      }
    }
    std::array<llama_token, 4 * kRows> input{};
    std::ifstream file(argv[2], std::ios::binary);
    file.read(reinterpret_cast<char*>(input.data()), sizeof(input));
    Require(bool(file) && file.peek() == std::char_traits<char>::eof() &&
                std::ranges::all_of(input, [](auto id) { return id >= 0 && id < kVocab; }),
            "exact four-owner valid input required");
    for (int owner = 0; owner < 4; ++owner)
      Require(input[owner * kRows] == 2, "each owner must have leading BOS");
    const fs::path out(argv[3]);
    Require(fs::create_directory(out), "new private output required");
    llama_backend_init();
    {
      auto mp = llama_model_default_params();
      mp.n_gpu_layers = 999;
      mp.load_mode = LLAMA_LOAD_MODE_NONE;
      mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
      std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
          llama_model_load_from_file(argv[1], mp), llama_model_free);
      Require(model && llama_model_n_layer(model.get()) == (dense ? 60 : 30) &&
                  llama_model_n_embd(model.get()) == (dense ? 5376 : 2816) &&
                  llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == kVocab,
              "approved model geometry differs");
      // Sampler owners outlive the context; no custom callback or output filter.
      using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
      std::array<Sampler, 4> samplers{
          Sampler(nullptr, llama_sampler_free), Sampler(nullptr, llama_sampler_free),
          Sampler(nullptr, llama_sampler_free), Sampler(nullptr, llama_sampler_free)};
      std::array<llama_sampler_seq_config, 4> config{};
      for (int owner = 0; owner < owners; ++owner) {
        samplers[owner].reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        Require(bool(samplers[owner]), "sampler allocation failed");
        llama_sampler_chain_add(samplers[owner].get(), llama_sampler_init_greedy());
        Require(llama_sampler_chain_n(samplers[owner].get()) == 1, "exact greedy chain required");
        config[owner] = {owner, samplers[owner].get()};
      }
      auto cp = llama_context_default_params();
      cp.n_ctx = kRows * owners;
      cp.n_seq_max = owners;
      cp.n_batch = cp.n_ubatch = chunk;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
      cp.type_k = cp.type_v = GGML_TYPE_F16;
      cp.swa_full = false;  // common CLI/server default, unlike public context default.
      cp.kv_unified = false;
      cp.no_perf = false;
      cp.samplers = config.data();
      cp.n_samplers = owners;
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
          llama_init_from_model(model.get(), cp), llama_free);
      Require(ctx && llama_n_ctx_seq(ctx.get()) == kRows &&
                  llama_n_seq_max(ctx.get()) == static_cast<std::uint32_t>(owners),
              "effective context geometry differs");
      struct Batch {
        llama_batch value;
        explicit Batch(int rows) : value(llama_batch_init(rows, 0, 1)) {}
        ~Batch() { llama_batch_free(value); }
      } batch(chunk);
      std::cout << "GEMMA_STOCK_CONFIG profile=" << profile << " owners=" << owners
                << " context_per_seq=8192 batch=" << chunk
                << " kv=F16 separate_sequences=1 swa_full=0 kv_unified=0 threads="
                << llama_n_threads(ctx.get())
                << " batch_threads=" << llama_n_threads_batch(ctx.get())
                << " fusion=stock graphs=allowed sampler=greedy host_sampled_logits=full\n";
      for (const auto& phase : quality ? std::vector<std::string>{"first", "repeat"}
                                       : std::vector<std::string>{"warm", "second", "third"}) {
        const fs::path directory = out / phase;
        Require(fs::create_directory(directory), "exclusive phase refused");
        std::array<std::array<llama_token, kOutputs>, 4> tokens{};
        std::array<std::vector<llama_token>, 4> history;
        std::array<std::ofstream, 4> heads;
        for (int owner = 0; owner < owners; ++owner) {
          history[owner].reserve(kRows);
          if (quality) {
            heads[owner].open(directory / ("heads-" + std::to_string(owner) + ".f32"),
                              std::ios::binary | std::ios::noreplace);
            Require(bool(heads[owner]), "exclusive head output refused");
          }
        }
        std::uint64_t backend_tokens = 0, sampled_logit_bytes = 0;
        const auto publish = [&](int owner, int output_index, int batch_index) {
          const auto token = llama_get_sampled_token_ith(ctx.get(), batch_index);
          Require(token >= 0 && token < kVocab, "backend greedy token missing");
          const auto count = llama_get_sampled_logits_count_ith(ctx.get(), batch_index);
          Require(count == kVocab, "ordinary full sampled-logit export differs");
          ++backend_tokens;
          sampled_logit_bytes += static_cast<std::uint64_t>(count) * sizeof(float);
          tokens[owner][output_index] = token;
          if (quality) {
            const auto* row = llama_get_sampled_logits_ith(ctx.get(), batch_index);
            Require(row && std::all_of(row, row + kVocab, [](float x) { return std::isfinite(x); }),
                    "nonfinite full head");
            Require(std::max_element(row, row + kVocab) - row == token,
                    "backend greedy differs from finite full-head argmax");
            heads[owner].write(reinterpret_cast<const char*>(row), kVocab * sizeof(float));
            Require(bool(heads[owner]), "complete head write failed");
          }
        };
        const auto start = Clock::now();
        llama_memory_clear(llama_get_memory(ctx.get()), false);
        for (int owner = 0; owner < owners; ++owner) llama_sampler_reset(samplers[owner].get());
        // Each owner prefills independently, as the actual serving caller does.
        // State-only chunks have no artificial synchronize; first head is boundary.
        for (int owner = 0; owner < owners; ++owner) {
          for (int past = 0; past < kPrefix;) {
            const int rows = std::min(chunk, kPrefix - past);
            batch.value.n_tokens = rows;
            for (int i = 0; i < rows; ++i) {
              const auto token = input[owner * kRows + past + i];
              batch.value.token[i] = token;
              batch.value.pos[i] = past + i;
              batch.value.n_seq_id[i] = 1;
              batch.value.seq_id[i][0] = owner;
              batch.value.logits[i] = past + i + 1 == kPrefix;
              llama_sampler_accept(samplers[owner].get(), token);
              history[owner].push_back(token);
            }
            Require(llama_decode(ctx.get(), batch.value) == 0, "prefill decode failed");
            past += rows;
            if (past == kPrefix) publish(owner, 0, rows - 1);
          }
        }
        const auto prefilled = Clock::now();
        for (int wave = 0; wave < 128; ++wave) {
          batch.value.n_tokens = owners;
          for (int owner = 0; owner < owners; ++owner) {
            const auto token = quality ? supplied[owner][wave] : tokens[owner][wave];
            batch.value.token[owner] = token;
            batch.value.pos[owner] = kPrefix + wave;
            batch.value.n_seq_id[owner] = 1;
            batch.value.seq_id[owner][0] = owner;
            batch.value.logits[owner] = true;
            llama_sampler_accept(samplers[owner].get(), token);
            history[owner].push_back(token);
          }
          Require(llama_decode(ctx.get(), batch.value) == 0, "joined scalar decode failed");
          for (int owner = 0; owner < owners; ++owner) publish(owner, wave + 1, owner);
        }
        const auto completed = Clock::now();
        Require(backend_tokens == static_cast<std::uint64_t>(owners * kOutputs),
                "sampled output count differs");
        for (int owner = 0; owner < owners; ++owner) {
          Require(history[owner].size() == 8191, "completed history differs");
          if (quality) {
            heads[owner].flush();
            Require(bool(heads[owner]), "complete head flush failed");
          }
          Save<llama_token>(directory / ("tokens-" + std::to_string(owner) + ".i32"),
                            tokens[owner]);
          Save<llama_token>(directory / ("history-" + std::to_string(owner) + ".i32"),
                            history[owner]);
        }
        std::cout << "GEMMA_STOCK_CYCLE phase=" << phase << " owners=" << owners
                  << " emitted=" << owners * kOutputs << " waves=128 cursor=8191"
                  << " prefill_seconds=" << Seconds(prefilled - start)
                  << " decode_seconds=" << Seconds(completed - prefilled)
                  << " cycle_seconds=" << Seconds(completed - start)
                  << " backend_tokens=" << backend_tokens
                  << " sampled_logit_bytes=" << sampled_logit_bytes
                  << " full_head_oracle=" << quality << " supplied_teacher=" << quality
                  << " natural_lineage="
                  << (!quality ||
                      std::equal(tokens.begin(), tokens.begin() + owners, supplied.begin()))
                  << " final_head_paid=0\n";
      }
    }
    llama_backend_free();
    std::cout << "GEMMA_STOCK_RETIRED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
