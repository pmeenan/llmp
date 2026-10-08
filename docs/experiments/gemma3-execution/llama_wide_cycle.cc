// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Original-image public API: one C12 natural cycle; no shape callback in timing.
// MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "gemma3_wide_schedule.h"
#include "llama.h"
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace wide = llmp::benchmarks::gemma3_wide;
constexpr auto kVocab = wide::kVocab, kOwners = wide::kOwners;
void Require(bool good, const char* message) {
  if (!good) throw std::runtime_error(message);
}
std::string Read(const fs::path& path, std::uint64_t cap) {
  const auto bytes = fs::file_size(path);
  Require(bytes <= cap, "bounded input refused");
  std::string result(static_cast<std::size_t>(bytes), '\0');
  std::ifstream file(path, std::ios::binary);
  Require(bool(file.read(result.data(), static_cast<std::streamsize>(bytes))) &&
              file.peek() == std::char_traits<char>::eof(),
          "complete input read failed");
  return result;
}
template <class T>
void Save(const fs::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  Require(bool(file), "exclusive complete output refused");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 7, "MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT");
    Require(!std::getenv("GGML_CUDA_DISABLE_FUSION") && !std::getenv("GGML_CUDA_DISABLE_GRAPHS"),
            "stock overrides present");
    std::array<std::vector<llama_token>, kOwners> ids;
    std::array<std::uint32_t, kOwners> prefix{};
    std::array<std::string, 2> texts;
    for (std::uint32_t s = 0; s < kOwners; ++s) {
      const auto input = wide::SourceIndex(s);
      const auto raw = Read(argv[2 + input * 2], 4096 * 4);
      Require(raw.size() == (input ? 807U : 295U) * 4, "exact input geometry required");
      ids[s].resize(raw.size() / 4);
      std::memcpy(ids[s].data(), raw.data(), raw.size());
      prefix[s] = static_cast<std::uint32_t>(ids[s].size()) - 39;
      Require(ids[s][0] == 2 &&
                  std::ranges::all_of(ids[s],
                                      [](auto id) { return id >= 0 && id < std::int32_t(kVocab); }),
              "invalid tokens");
      texts[input] = Read(argv[3 + input * 2], 65536);
    }
    const fs::path out = argv[6];
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
              "model shape differs");
      const auto* vocab = llama_model_get_vocab(model.get());
      Require(llama_vocab_n_tokens(vocab) == std::int32_t(kVocab), "vocab differs");
      for (std::uint32_t input = 0; input < 2; ++input) {
        const auto& text = texts[input];
        const int count = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), nullptr,
                                         0, true, false);
        Require(count < 0 && count >= -8192, "token count refused");
        std::vector<llama_token> actual(static_cast<std::size_t>(-count));
        Require(llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), actual.data(),
                               -count, true, false) == -count &&
                    actual.size() >= ids[input * 2].size() &&
                    std::equal(ids[input * 2].begin(), ids[input * 2].end(), actual.begin()),
                "tokenization differs");
      }
      std::vector<std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>> samplers;
      std::array<llama_sampler_seq_config, kOwners> configs;
      for (std::uint32_t s = 0; s < kOwners; ++s) {
        samplers.emplace_back(llama_sampler_chain_init(llama_sampler_chain_default_params()),
                              llama_sampler_free);
        Require(bool(samplers.back()), "sampler allocation failed");
        llama_sampler_chain_add(samplers.back().get(), llama_sampler_init_greedy());
        Require(llama_sampler_chain_n(samplers.back().get()) == 1, "exact greedy chain required");
        configs[s] = {static_cast<llama_seq_id>(s), samplers.back().get()};
      }
      auto cp = llama_context_default_params();
      cp.n_ctx = 4096 * kOwners;
      cp.n_seq_max = kOwners;
      cp.n_batch = cp.n_ubatch = 256;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
      cp.type_k = cp.type_v = GGML_TYPE_F16;
      cp.swa_full = false;
      cp.kv_unified = false;
      cp.no_perf = false;
      cp.samplers = configs.data();
      cp.n_samplers = kOwners;
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
          llama_init_from_model(model.get(), cp), llama_free);
      Require(ctx && llama_n_ctx_seq(ctx.get()) == 4096 && llama_n_seq_max(ctx.get()) == kOwners,
              "context geometry differs");
      struct Batch {
        llama_batch value = llama_batch_init(256, 0, 1);
        ~Batch() { llama_batch_free(value); }
      } batch;
      std::array<std::uint32_t, kOwners> past{};
      std::array<llama_token, kOwners> selected{};
      std::array<std::vector<float>, kOwners> heads;
      for (auto& row : heads) row.reserve(kVocab);
      std::uint64_t backend_tokens = 0, prefill_groups = 0, prefill_rows = 0;
      std::uint32_t sampled_min = kVocab, sampled_max = 0;
      const auto add = [&](int index, std::uint32_t s, llama_token token, std::uint32_t at,
                           bool head) {
        Require(token >= 0 && token < std::int32_t(kVocab), "natural token out of range");
        batch.value.token[index] = token;
        batch.value.pos[index] = static_cast<llama_pos>(at);
        batch.value.n_seq_id[index] = 1;
        batch.value.seq_id[index][0] = static_cast<llama_seq_id>(s);
        batch.value.logits[index] = head;
        llama_sampler_accept(samplers[s].get(), token);
      };
      const auto publish = [&](std::uint32_t s, int index, bool full) {
        selected[s] = llama_get_sampled_token_ith(ctx.get(), index);
        Require(selected[s] >= 0 && selected[s] < std::int32_t(kVocab), "backend token absent");
        ++backend_tokens;
        const auto count = llama_get_sampled_logits_count_ith(ctx.get(), index);
        sampled_min = std::min(sampled_min, count);
        sampled_max = std::max(sampled_max, count);
        Require(count == kVocab, "ordinary sampler full-row export missing");
        if (full) {
          const auto* row = llama_get_sampled_logits_ith(ctx.get(), index);
          Require(row != nullptr, "final row missing");
          heads[s].assign(row, row + kVocab);
        }
      };
      const auto prompt = [&]() {
        for (std::uint32_t at = 0; at < 768; at += 128)
          for (std::uint32_t first = 0; first < kOwners; first += 2) {
            if (at >= prefix[first]) continue;
            const bool head = at + 128 == prefix[first];
            std::array<int, 2> outputs{};
            batch.value.n_tokens = 0;
            for (std::uint32_t offset = 0; offset < 2; ++offset) {
              const auto slot = first + offset;
              Require(past[slot] == at && prefix[slot] == prefix[first], "paired cursor differs");
              for (std::uint32_t i = 0; i < 128; ++i) {
                const int index = batch.value.n_tokens++;
                add(index, slot, ids[slot][at + i], at + i, head && i + 1 == 128);
                outputs[offset] = index;
              }
            }
            Require(llama_decode(ctx.get(), batch.value) == 0, "paired prefill failed");
            for (std::uint32_t offset = 0; offset < 2; ++offset) {
              const auto slot = first + offset;
              past[slot] += 128;
              if (head) publish(slot, outputs[offset], false);
            }
            ++prefill_groups;
            prefill_rows += 256;
          }
      };
      const auto teacher = [&]() {
        for (std::uint32_t slot = 0; slot < kOwners; ++slot)
          for (std::uint32_t i = 0; i < 3; ++i) {
            batch.value.n_tokens = 1;
            add(0, slot, ids[slot][past[slot]], past[slot], true);
            Require(llama_decode(ctx.get(), batch.value) == 0, "supplied scalar startup failed");
            ++past[slot];
            publish(slot, 0, false);
          }
      };
      const auto wave = [&](bool full) {
        const auto tokens = selected;
        batch.value.n_tokens = kOwners;
        for (std::uint32_t s = 0; s < kOwners; ++s)
          add(static_cast<int>(s), s, tokens[s], past[s], true);
        Require(llama_decode(ctx.get(), batch.value) == 0, "natural C12 decode failed");
        for (std::uint32_t s = 0; s < kOwners; ++s) {
          ++past[s];
          publish(s, static_cast<int>(s), full);
        }
      };
      llama_memory_clear(llama_get_memory(ctx.get()), false);
      prompt();
      teacher();
      for (std::uint32_t i = 0; i < 8; ++i) wave(i >= 5);
      // Match native logical Clear: retain physical backing, off clock, one warm pass.
      llama_memory_clear(llama_get_memory(ctx.get()), false);
      for (auto& sampler : samplers) llama_sampler_reset(sampler.get());
      past = {};
      std::vector<llama_token> choices;
      choices.reserve(32 * kOwners);
      const auto begin = Clock::now();
      prompt();
      teacher();
      const auto prefill = std::chrono::duration<double>(Clock::now() - begin).count();
      const auto decode_begin = Clock::now();
      for (std::uint32_t i = 0; i < 32; ++i) {
        choices.insert(choices.end(), selected.begin(), selected.end());
        wave(i + 1 == 32);
      }
      const auto decode = std::chrono::duration<double>(Clock::now() - decode_begin).count();
      for (std::uint32_t s = 0; s < kOwners; ++s) {
        const auto& row = heads[s];
        Require(row.size() == kVocab &&
                    std::ranges::all_of(row, [](float f) { return std::isfinite(f); }) &&
                    selected[s] == std::max_element(row.begin(), row.end()) - row.begin() &&
                    past[s] == prefix[s] + 35,
                "finite final sampled head/cursor differs");
      }
      Require(choices.size() == 384 && backend_tokens == 576 && prefill_groups == 48 &&
                  prefill_rows == 12288 && sampled_min == kVocab && sampled_max == kVocab,
              "cycle work/publication count differs");
      Save<llama_token>(out / "chosen.i32", choices);
      Save<std::uint32_t>(out / "past.u32", past);
      std::ofstream final(out / "final.f32", std::ios::binary | std::ios::noreplace);
      for (auto& row : heads) final.write(reinterpret_cast<const char*>(row.data()), kVocab * 4);
      final.flush();
      Require(bool(final), "exclusive final heads failed");
      std::cout
          << "GEMMA3_WIDE_CYCLE_STOCK slots=12 input_identities=2 context=4096 per_owner_rows=128 "
             "wave_rows=256"
          << " paid_prefix_rows=6144 paid_teacher_rows=36 paid_generated_tokens=384 "
             "decode_steps=32 final_heads_paid=12"
          << " prefill_seconds=" << prefill << " decode_seconds=" << decode
          << " backend_tokens=" << backend_tokens << " sampled_logits_min=" << sampled_min
          << " sampled_logits_max=" << sampled_max
          << " observer=0 tokenized_equal=2 reset_zero=false swa_full=false kv_unified=false\n";
    }
    llama_backend_free();
    std::cout << "GEMMA3_WIDE_CYCLE_STOCK_RETIRED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
