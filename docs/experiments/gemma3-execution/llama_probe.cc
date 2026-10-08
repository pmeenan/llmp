// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// External public-API caller for the original llama.cpp v0.6.0 image.
// MODEL IDS_I32 PROMPT_TEXT NEW_OUT teacher|cycle|greedy-teacher|greedy-cycle; fixed bounded C1
// screen.
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
#include <vector>

#include "llama.h"
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kVocab = 262208, kPrompt = 256, kWarm = 3, kSteps = 32, kInput = 291;
void Require(bool good, const char* message) {
  if (!good) throw std::runtime_error(message);
}
std::string Read(const fs::path& path, std::uint64_t cap) {
  const auto size = fs::file_size(path);
  Require(size <= cap, "input exceeds bound");
  std::string value(static_cast<std::size_t>(size), '\0');
  std::ifstream file(path, std::ios::binary);
  Require(bool(file.read(value.data(), static_cast<std::streamsize>(size))) &&
              file.peek() == std::char_traits<char>::eof(),
          "complete input read failed");
  return value;
}
template <class T>
void Save(const fs::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  Require(bool(file), "exclusive complete output failed");
}
double Seconds(Clock::duration value) { return std::chrono::duration<double>(value).count(); }
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 6,
            "MODEL IDS_I32 PROMPT_TEXT NEW_OUT teacher|cycle|greedy-teacher|greedy-cycle");
    const std::string mode = argv[5];
    const bool backend_sampling = mode == "greedy-teacher" || mode == "greedy-cycle";
    const bool teacher = mode == "teacher" || mode == "greedy-teacher";
    Require(teacher || mode == "cycle" || mode == "greedy-cycle", "unknown mode");
    Require(!std::getenv("GGML_CUDA_DISABLE_FUSION") && !std::getenv("GGML_CUDA_DISABLE_GRAPHS"),
            "stock fusion/graph override present");
    const auto raw = Read(argv[2], kInput * sizeof(llama_token));
    Require(raw.size() == kInput * sizeof(llama_token), "exact 291-ID input required");
    std::array<llama_token, kInput> ids{};
    std::copy_n(reinterpret_cast<const char*>(raw.data()), raw.size(),
                reinterpret_cast<char*>(ids.data()));
    Require(ids.front() == 2 &&
                std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < kVocab; }),
            "invalid token input");
    const auto text = Read(argv[3], 65536);
    const fs::path out = argv[4];
    Require(fs::create_directory(out), "output must be new");
    llama_backend_init();
    {
      auto mp = llama_model_default_params();
      mp.n_gpu_layers = 999;
      mp.load_mode = LLAMA_LOAD_MODE_NONE;
      mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
      std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
          llama_model_load_from_file(argv[1], mp), llama_model_free);
      Require(model && llama_model_n_layer(model.get()) == 34 &&
                  llama_model_n_embd(model.get()) == 2560,
              "approved Gemma3 shape differs");
      const auto* vocab = llama_model_get_vocab(model.get());
      Require(llama_vocab_n_tokens(vocab) == kVocab, "vocabulary differs");
      const auto count = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), nullptr,
                                        0, true, false);
      Require(count < 0 && count >= -8192, "bounded token count refused");
      std::vector<llama_token> tokenized(static_cast<std::size_t>(-count));
      Require(llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokenized.data(),
                             -count, true, false) == -count &&
                  tokenized.size() >= kInput &&
                  std::equal(ids.begin(), ids.end(), tokenized.begin()),
              "native/stock input IDs differ");
      auto cp = llama_context_default_params();
      cp.n_ctx = 4096;
      cp.n_batch = cp.n_ubatch = 128;
      cp.n_seq_max = 1;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
      cp.type_k = cp.type_v = GGML_TYPE_F16;
      cp.swa_full = false;
      cp.kv_unified = false;
      cp.no_perf = false;
      std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(nullptr,
                                                                            llama_sampler_free);
      llama_sampler_seq_config config{};
      if (backend_sampling) {
        sampler.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        Require(bool(sampler), "greedy sampler chain allocation failed");
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        Require(llama_sampler_chain_n(sampler.get()) == 1, "exact greedy chain required");
        config = {0, sampler.get()};
        cp.samplers = &config;
        cp.n_samplers = 1;
      }
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
          llama_init_from_model(model.get(), cp), llama_free);
      Require(ctx && llama_n_ctx_seq(ctx.get()) == 4096 && llama_n_seq_max(ctx.get()) == 1,
              "context geometry differs");
      struct Batch {
        llama_batch value = llama_batch_init(128, 0, 1);
        ~Batch() { llama_batch_free(value); }
      } batch;
      std::vector<float> published;
      published.reserve(kVocab);
      int past = 0;
      llama_token sampled = LLAMA_TOKEN_NULL;
      std::uint64_t backend_tokens = 0;
      std::uint32_t sampled_logits_min = kVocab, sampled_logits_max = 0;
      bool final_head = false;
      const auto host_best = [&]() -> llama_token {
        Require(published.size() == kVocab &&
                    std::ranges::all_of(published, [](float x) { return std::isfinite(x); }),
                "nonfinite or missing full head");
        return static_cast<llama_token>(std::max_element(published.begin(), published.end()) -
                                        published.begin());
      };
      const auto best = [&]() -> llama_token { return backend_sampling ? sampled : host_best(); };
      const auto decode = [&](std::span<const llama_token> tokens, bool head) {
        batch.value.n_tokens = static_cast<int>(tokens.size());
        for (int i = 0; i < batch.value.n_tokens; ++i) {
          batch.value.token[i] = tokens[static_cast<std::size_t>(i)];
          batch.value.pos[i] = past + i;
          batch.value.n_seq_id[i] = 1;
          batch.value.seq_id[i][0] = 0;
          batch.value.logits[i] = head && i == batch.value.n_tokens - 1;
        }
        if (backend_sampling)
          for (auto token : tokens) llama_sampler_accept(sampler.get(), token);
        Require(llama_decode(ctx.get(), batch.value) == 0, "decode failed");
        if (head && backend_sampling) {
          sampled = llama_get_sampled_token_ith(ctx.get(), -1);
          Require(sampled >= 0 && sampled < kVocab, "backend greedy token missing or invalid");
          ++backend_tokens;
          const auto count = llama_get_sampled_logits_count_ith(ctx.get(), -1);
          sampled_logits_min = std::min(sampled_logits_min, count);
          sampled_logits_max = std::max(sampled_logits_max, count);
          if (teacher || final_head) {
            Require(count == kVocab, "full sampled logits required for exact verification head");
            const auto* row = llama_get_sampled_logits_ith(ctx.get(), -1);
            Require(row != nullptr, "sampled full head missing");
            published.assign(row, row + kVocab);
            Require(host_best() == sampled, "backend greedy differs from finite full-head argmax");
          }
        } else if (head) {
          const auto* row = llama_get_logits_ith(ctx.get(), -1);
          Require(row != nullptr, "full head missing");
          published.assign(row, row + kVocab);
          host_best();
        } else {
          llama_synchronize(ctx.get());
          published.clear();
        }
        past += static_cast<int>(tokens.size());
      };
      const auto prompt = [&]() {
        decode(std::span(ids).first(128), false);
        decode(std::span(ids).subspan(128, 128), true);
      };
      const auto warm = [&]() {
        for (int i = 0; i < kWarm; ++i) decode(std::span(ids).subspan(kPrompt + i, 1), true);
      };
      llama_memory_clear(llama_get_memory(ctx.get()), true);
      if (!teacher) {
        prompt();
        warm();
        for (int i = 0; i < 8; ++i) {
          const auto token = best();
          decode(std::span(&token, 1), true);
        }
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        past = 0;
      }
      const auto start = Clock::now();
      prompt();
      const double prefill = Seconds(Clock::now() - start);
      warm();
      std::ofstream heads;
      if (teacher) {
        heads.open(out / "heads.f32", std::ios::binary | std::ios::noreplace);
        Require(bool(heads), "head output refused");
        heads.write(reinterpret_cast<const char*>(published.data()), kVocab * sizeof(float));
      }
      const auto before = llama_perf_context(ctx.get());
      std::array<llama_token, kSteps> chosen{};
      const auto decode_start = Clock::now();
      for (int i = 0; i < kSteps; ++i) {
        const auto next = best();
        chosen[static_cast<std::size_t>(i)] = next;
        const auto token = !teacher ? next : ids[kPrompt + kWarm + i];
        final_head = !teacher && i + 1 == kSteps;
        decode(std::span(&token, 1), true);
        if (teacher)
          heads.write(reinterpret_cast<const char*>(published.data()), kVocab * sizeof(float));
      }
      const double elapsed = Seconds(Clock::now() - decode_start);
      if (teacher) {
        heads.flush();
        Require(bool(heads), "complete head write failed");
      }
      Save<llama_token>(out / "chosen.i32", chosen);
      Save<float>(out / "final.f32", published);
      const auto after = llama_perf_context(ctx.get());
      std::cout << "GEMMA3_STOCK mode=" << mode << " context=4096 slots=1 chunk=128 prompt_rows=256"
                << " untimed_rows=3 decode_rows=32 past=" << past << " prefill_seconds=" << prefill
                << " decode_seconds=" << elapsed << " reused=" << after.n_reused - before.n_reused
                << " tokenized_equal=1 fusion=stock graphs=allowed backend_tokens="
                << backend_tokens << " sampled_logits_min=" << sampled_logits_min
                << " sampled_logits_max=" << sampled_logits_max << " final_head_paid=" << !teacher
                << '\n';
    }
    llama_backend_free();
    std::cout << "GEMMA3_STOCK_RETIRED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
