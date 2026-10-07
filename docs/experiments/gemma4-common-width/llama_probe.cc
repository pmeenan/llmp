// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original-image public API: independent C1 prefill, joined C2 decode.
// MODEL IDS0 IDS1 NEW_OUT 26|31 teacher|cycle
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

#include "llama.h"
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kVocab = 262144, kSteps = 32;
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
struct Shapes {
  using Key = std::array<std::int64_t, 12>;
  std::map<Key, std::uint64_t> counts;
  std::array<bool, 2> joined{};
  static bool Observe(ggml_tensor* node, bool ask, void* opaque) {
    if (!ask || node->op != GGML_OP_FLASH_ATTN_EXT) return false;
    auto& self = *static_cast<Shapes*>(opaque);
    Key key{};
    for (int source = 0; source < 3; ++source) {
      const auto* t = node->src[source == 2 ? 3 : source];
      if (!t) return false;
      std::copy_n(t->ne, 4, key.begin() + source * 4);
    }
    ++self.counts[key];
    for (int index = 0; index < 2; ++index) {
      const auto d = index == 0 ? 256 : 512;
      const auto gqa = index == 0 ? 2 : 8;
      self.joined[index] |= key[0] == d && key[1] == 1 && (key[2] == 16 || key[2] == 32) &&
                            key[3] == 2 && key[4] == d && key[5] == 1024 &&
                            key[6] == key[2] / gqa && key[7] == 2 && key[8] == key[5] &&
                            key[9] == 1 && key[11] == 2;
    }
    return false;  // Metadata only: never request operand download.
  }
};
double Seconds(Clock::duration value) { return std::chrono::duration<double>(value).count(); }
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 7, "MODEL IDS0 IDS1 NEW_OUT 26|31 teacher|cycle");
    const std::string profile = argv[5], mode = argv[6];
    Require(profile == "26" || profile == "31", "unknown profile");
    const bool dense = profile == "31";
    const int chunk = dense ? 256 : 1024;
    const bool teacher = mode == "teacher";
    Require(teacher || mode == "cycle", "unknown mode");
    Require(!std::getenv("GGML_CUDA_DISABLE_FUSION") && !std::getenv("GGML_CUDA_DISABLE_GRAPHS"),
            "stock overrides present");
    std::array<std::vector<llama_token>, 2> ids;
    std::array<int, 2> prefix{};
    for (std::size_t s = 0; s < 2; ++s) {
      const auto raw = Read(argv[2 + s], 4096 * sizeof(llama_token));
      Require(raw.size() % 4 == 0 && raw.size() >= 291 * 4, "bounded IDs required");
      ids[s].resize(raw.size() / 4);
      prefix[s] = static_cast<int>(ids[s].size()) - 3 - kSteps;
      std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(ids[s].data()));
      Require(ids[s][0] == 2 &&
                  std::ranges::all_of(ids[s], [](auto id) { return id >= 0 && id < kVocab; }),
              "invalid tokens");
    }
    Require(ids[0] != ids[1], "different real inputs required");
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
      Require(model && llama_model_n_layer(model.get()) == (dense ? 60 : 30) &&
                  llama_model_n_embd(model.get()) == (dense ? 5376 : 2816),
              "model shape differs");
      const auto* vocab = llama_model_get_vocab(model.get());
      Require(llama_vocab_n_tokens(vocab) == kVocab, "vocabulary differs");
      std::array<std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>, 2> samplers{
          std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(nullptr,
                                                                        llama_sampler_free),
          std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(nullptr,
                                                                        llama_sampler_free)};
      std::array<llama_sampler_seq_config, 2> configs;
      for (int s = 0; s < 2; ++s) {
        samplers[s].reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        Require(bool(samplers[s]), "sampler allocation failed");
        llama_sampler_chain_add(samplers[s].get(), llama_sampler_init_greedy());
        Require(llama_sampler_chain_n(samplers[s].get()) == 1, "exact greedy chain required");
        configs[s] = {s, samplers[s].get()};
      }
      Shapes shapes;
      auto cp = llama_context_default_params();
      cp.n_ctx = 8192;
      cp.n_seq_max = 2;
      cp.n_batch = cp.n_ubatch = chunk;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
      cp.type_k = cp.type_v = GGML_TYPE_F16;
      cp.swa_full = false;
      cp.kv_unified = false;
      cp.no_perf = false;
      cp.samplers = configs.data();
      cp.n_samplers = 2;
      // Factual shape observer exists only in teacher runs, never either timing arm.
      if (teacher) {
        cp.cb_eval = Shapes::Observe;
        cp.cb_eval_user_data = &shapes;
      }
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
          llama_init_from_model(model.get(), cp), llama_free);
      Require(ctx && llama_n_ctx_seq(ctx.get()) == 4096 && llama_n_seq_max(ctx.get()) == 2,
              "C2 context geometry differs");
      struct Batch {
        llama_batch value;
        explicit Batch(int count) : value(llama_batch_init(count, 0, 1)) {}
        ~Batch() { llama_batch_free(value); }
      } batch(chunk);
      std::array<int, 2> past{};
      std::array<llama_token, 2> selected{-1, -1};
      std::array<std::vector<float>, 2> published;
      for (auto& row : published) row.reserve(kVocab);
      std::uint64_t backend_tokens = 0;
      std::uint32_t sampled_min = kVocab, sampled_max = 0;
      bool final_head = false;
      const auto publish = [&](int slot, int index) {
        selected[slot] = llama_get_sampled_token_ith(ctx.get(), index);
        Require(selected[slot] >= 0 && selected[slot] < kVocab, "backend token missing");
        ++backend_tokens;
        const auto count = llama_get_sampled_logits_count_ith(ctx.get(), index);
        sampled_min = std::min(sampled_min, count);
        sampled_max = std::max(sampled_max, count);
        if (teacher || final_head) {
          Require(count == kVocab, "full sampled logits required");
          const auto* row = llama_get_sampled_logits_ith(ctx.get(), index);
          Require(row != nullptr, "sampled row missing");
          published[slot].assign(row, row + kVocab);
          Require(std::ranges::all_of(published[slot], [](float x) { return std::isfinite(x); }) &&
                      selected[slot] == std::max_element(row, row + kVocab) - row,
                  "finite backend/full-head argmax differs");
        }
      };
      const auto add = [&](int index, int slot, llama_token token, int position, bool head) {
        batch.value.token[index] = token;
        batch.value.pos[index] = position;
        batch.value.n_seq_id[index] = 1;
        batch.value.seq_id[index][0] = slot;
        batch.value.logits[index] = head;
        llama_sampler_accept(samplers[slot].get(), token);
      };
      const auto independent = [&](int slot, std::span<const llama_token> tokens, bool head) {
        batch.value.n_tokens = static_cast<int>(tokens.size());
        for (int i = 0; i < batch.value.n_tokens; ++i)
          add(i, slot, tokens[static_cast<std::size_t>(i)], past[slot] + i,
              head && i + 1 == batch.value.n_tokens);
        Require(llama_decode(ctx.get(), batch.value) == 0, "independent decode failed");
        if (head)
          publish(slot, -1);
        else {
          published[slot].clear();
        }
        past[slot] += static_cast<int>(tokens.size());
      };
      const auto joined = [&](const std::array<llama_token, 2>& tokens) {
        batch.value.n_tokens = 2;
        for (int s = 0; s < 2; ++s) add(s, s, tokens[s], past[s], true);
        Require(llama_decode(ctx.get(), batch.value) == 0, "joined decode failed");
        for (int s = 0; s < 2; ++s) {
          publish(s, s);
          ++past[s];
        }
      };
      const auto prompt = [&] {
        for (int s = 0; s < 2; ++s) {
          for (int at = 0; at < prefix[s];) {
            const auto rows = std::min(chunk, prefix[s] - at);
            independent(s, std::span(ids[s]).subspan(at, rows), at + rows == prefix[s]);
            at += rows;
          }
        }
      };
      const auto warm = [&] {
        for (int s = 0; s < 2; ++s)
          for (int i = 0; i < 3; ++i)
            independent(s, std::span(ids[s]).subspan(prefix[s] + i, 1), true);
      };
      llama_memory_clear(llama_get_memory(ctx.get()), false);
      if (!teacher) {
        prompt();
        warm();
        for (int i = 0; i < 8; ++i) joined(selected);
        llama_memory_clear(llama_get_memory(ctx.get()), false);
        past = {};
        for (auto& sampler : samplers) llama_sampler_reset(sampler.get());
      }
      const auto begin = Clock::now();
      prompt();
      if (teacher) {
        std::vector<float> initial;
        for (const auto& row : published) initial.insert(initial.end(), row.begin(), row.end());
        Save<float>(out / "predecode.f32", initial);
      }
      const auto prefill = Seconds(Clock::now() - begin);
      warm();
      std::ofstream rows;
      const auto write_rows = [&] {
        for (const auto& row : published)
          rows.write(reinterpret_cast<const char*>(row.data()), kVocab * 4);
      };
      if (teacher) {
        rows.open(out / "heads.f32", std::ios::binary | std::ios::noreplace);
        Require(bool(rows), "teacher rows must be new");
        write_rows();
      }
      std::vector<llama_token> choices(kSteps * 2);
      const auto decode_begin = Clock::now();
      for (int i = 0; i < kSteps; ++i) {
        std::array<llama_token, 2> tokens;
        for (int s = 0; s < 2; ++s) {
          choices[static_cast<std::size_t>(i * 2 + s)] = selected[s];
          tokens[s] = teacher ? ids[s][prefix[s] + 3 + i] : selected[s];
        }
        final_head = !teacher && i + 1 == kSteps;
        joined(tokens);
        if (teacher) write_rows();
      }
      const auto decode = Seconds(Clock::now() - decode_begin);
      if (teacher) {
        rows.flush();
        Require(bool(rows), "complete teacher rows failed");
      }
      Save<llama_token>(out / "chosen.i32", choices);
      for (int slot = 0; slot < 2; ++slot) {
        Require(past[slot] == prefix[slot] + 3 + kSteps, "completed history differs");
        std::vector<llama_token> history(ids[slot].begin(), ids[slot].begin() + prefix[slot] + 3);
        for (int step = 0; step < kSteps; ++step)
          history.push_back(teacher ? ids[slot][prefix[slot] + 3 + step]
                                    : choices[step * 2 + slot]);
        Save<llama_token>(out / ("history" + std::to_string(slot) + ".i32"), history);
      }
      std::vector<float> final;
      for (const auto& row : published) final.insert(final.end(), row.begin(), row.end());
      Save<float>(out / "final.f32", final);
      Require(backend_tokens == (teacher ? 72U : 96U), "backend sample count differs");
      if (teacher) {
        for (const auto& [key, count] : shapes.counts) {
          std::cout << "GEMMA4_CONTEXT_STOCK_FLASH count=" << count;
          for (auto value : key) std::cout << ' ' << value;
          std::cout << '\n';
        }
        Require(shapes.joined[0] && shapes.joined[1],
                "ordinary stock C2 FLASH geometry not observed");
      }
      std::cout << "GEMMA4_CONTEXT_STOCK mode=" << mode << " profile=" << profile
                << " slots=2 context_per_slot=4096 chunk=" << chunk
                << " independent_prefill=1 prompt_rows0=" << prefix[0]
                << " prompt_rows1=" << prefix[1] << " untimed_rows_per_slot=3"
                << " decode_steps=32 departure_steps=0"
                << " paid_generated_tokens=64 past0=" << past[0] << " past1=" << past[1]
                << " prefill_seconds=" << prefill << " decode_seconds=" << decode
                << " backend_tokens=" << backend_tokens << " sampled_logits_min=" << sampled_min
                << " sampled_logits_max=" << sampled_max << " observer=" << teacher
                << " supplied_authenticated_input=1 sampled_logit_transfer=full final_head_paid="
                << !teacher << '\n';
    }
    llama_backend_free();
    std::cout << "GEMMA4_CONTEXT_STOCK_RETIRED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
