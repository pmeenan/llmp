// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Original-image public API: compatible two-owner prefill and joined C2 decode.
// MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT teacher|cycle [chunk=N]
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
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
#include <string_view>
#include <tuple>
#include <vector>

#include "llama.h"
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kVocab = 256000, kSteps = 32, kTail = 4;
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
  using Key = std::array<std::int64_t, 13>;
  std::map<Key, std::uint64_t> counts;
  bool joined = false, joined_prefill = false;
  int chunk = 128;
  static bool Observe(ggml_tensor* node, bool ask, void* opaque) {
    if (!ask || node->op != GGML_OP_FLASH_ATTN_EXT) return false;
    auto& self = *static_cast<Shapes*>(opaque);
    Key key{};
    for (int source = 0; source < 3; ++source) {
      const auto* t = node->src[source == 2 ? 3 : source];
      if (!t) return false;
      std::copy_n(t->ne, 4, key.begin() + source * 4);
    }
    key[12] = static_cast<std::int64_t>(std::bit_cast<float>(node->op_params[2]));
    ++self.counts[key];
    self.joined |= key[0] == 256 && key[1] == 1 && key[2] == 8 && key[3] == 2 && key[4] == 256 &&
                   key[5] >= 256 && key[6] == 4 && key[7] == 2 && key[8] == key[5] &&
                   key[11] == 2 && key[12] == 50;
    self.joined_prefill |= key[0] == 256 && key[1] == self.chunk && key[2] == 8 && key[3] == 2 &&
                           key[4] == 256 && key[6] == 4 && key[7] == 2 && key[12] == 50;
    return false;  // Metadata only: never request operand download.
  }
};
double Seconds(Clock::duration value) { return std::chrono::duration<double>(value).count(); }
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 8 || argc == 9, "MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT teacher|cycle [chunk=N]");
    int chunk = 128;
    if (argc == 9) {
      const std::string_view option = argv[8];
      Require(option.starts_with("chunk="), "unknown chunk option");
      const auto value = option.substr(6);
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), chunk);
      Require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && chunk >= 2 &&
                  chunk <= 512,
              "chunk must be in 2..512");
    }
    const std::string mode = argv[7];
    const bool teacher = mode == "teacher";
    Require(teacher || mode == "cycle", "unknown mode");
    Require(!std::getenv("GGML_CUDA_DISABLE_FUSION") && !std::getenv("GGML_CUDA_DISABLE_GRAPHS"),
            "stock overrides present");
    std::array<std::vector<llama_token>, 2> ids;
    std::array<int, 2> prefix{};
    std::array<std::string, 2> texts;
    for (std::size_t s = 0; s < 2; ++s) {
      const auto raw = Read(argv[2 + s * 2], 8192 * sizeof(llama_token));
      Require(raw.size() % 4 == 0 && raw.size() >= 295 * 4, "bounded IDs required");
      ids[s].resize(raw.size() / 4);
      prefix[s] = static_cast<int>(ids[s].size()) - 3 - kSteps - kTail;
      std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(ids[s].data()));
      Require(ids[s][0] == 2 &&
                  std::ranges::all_of(ids[s], [](auto id) { return id >= 0 && id < kVocab; }),
              "invalid tokens");
      texts[s] = Read(argv[3 + s * 2], 65536);
    }
    Require(ids[0] != ids[1], "different real inputs required");
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
      Require(model && llama_model_n_layer(model.get()) == 26 &&
                  llama_model_n_embd(model.get()) == 2304,
              "model shape differs");
      const auto* vocab = llama_model_get_vocab(model.get());
      Require(llama_vocab_n_tokens(vocab) == kVocab, "vocabulary differs");
      for (std::size_t s = 0; s < 2; ++s) {
        const auto& text = texts[s];
        const auto count = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()),
                                          nullptr, 0, true, false);
        Require(count < 0 && count >= -8192, "bounded token count refused");
        std::vector<llama_token> actual(static_cast<std::size_t>(-count));
        Require(llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), actual.data(),
                               -count, true, false) == -count &&
                    actual.size() >= ids[s].size() &&
                    std::equal(ids[s].begin(), ids[s].end(), actual.begin()),
                "native/stock tokenization differs");
      }
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
      shapes.chunk = chunk;
      auto cp = llama_context_default_params();
      cp.n_ctx = 16384;
      cp.n_seq_max = 2;
      cp.n_batch = cp.n_ubatch = static_cast<std::uint32_t>(2 * chunk);
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
      Require(ctx && llama_n_ctx_seq(ctx.get()) == 8192 && llama_n_seq_max(ctx.get()) == 2,
              "C2 context geometry differs");
      struct Batch {
        llama_batch value;
        explicit Batch(int rows) : value(llama_batch_init(rows, 0, 1)) {}
        ~Batch() { llama_batch_free(value); }
      } batch(2 * chunk);
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
      std::uint64_t prefill_groups = 0, prefill_rows = 0;
      const auto prompt = [&] {
        while (past[0] < prefix[0] || past[1] < prefix[1]) {
          const int first = past[0] == prefix[0]                         ? 1
                            : past[1] == prefix[1]                       ? 0
                            : prefix[0] - past[0] <= prefix[1] - past[1] ? 0
                                                                         : 1;
          const int first_rows = std::min(chunk, prefix[first] - past[first]);
          const bool head = past[first] + first_rows == prefix[first];
          std::array<int, 2> sizes{}, last_rows{};
          int owners = 0;
          batch.value.n_tokens = 0;
          for (const int slot : {first, 1 - first}) {
            if (past[slot] == prefix[slot]) continue;
            const int rows = std::min(chunk, prefix[slot] - past[slot]);
            if ((past[slot] + rows == prefix[slot]) != head) continue;
            ++owners;
            sizes[slot] = rows;
            for (int i = 0; i < rows; ++i) {
              const int index = batch.value.n_tokens++;
              add(index, slot, ids[slot][past[slot] + i], past[slot] + i, head && i + 1 == rows);
              last_rows[slot] = index;
            }
          }
          Require(llama_decode(ctx.get(), batch.value) == 0, "compatible prefill failed");
          for (int slot = 0; slot < 2; ++slot) {
            if (sizes[slot] == 0) continue;
            // Public positive index is the original batch-token index,
            // translated by output_resolve_row (last rows may be127/255).
            if (head)
              publish(slot, last_rows[slot]);
            else
              published[slot].clear();
            past[slot] += sizes[slot];
          }
          if (owners == 2) {
            ++prefill_groups;
            prefill_rows += static_cast<std::uint64_t>(batch.value.n_tokens);
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
      }
      const auto begin = Clock::now();
      prompt();
      const auto prefill = Seconds(Clock::now() - begin);
      if (teacher) {
        std::vector<float> frontier;
        frontier.reserve(kVocab * 2);
        for (const auto& row : published) {
          Require(row.size() == kVocab &&
                      std::ranges::all_of(row, [](float x) { return std::isfinite(x); }),
                  "finite prefill frontier required");
          frontier.insert(frontier.end(), row.begin(), row.end());
        }
        Save<float>(out / "prefill.f32", frontier);
      }
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
      std::vector<llama_token> choices((kSteps + (teacher ? kTail : 0)) * 2);
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
      if (teacher) {
        for (int s = 0; s < 2; ++s)
          for (int i = 0; i < kTail; ++i) {
            choices[kSteps * 2 + s * kTail + i] = selected[s];
            independent(s, std::span(ids[s]).subspan(past[s], 1), true);
            rows.write(reinterpret_cast<const char*>(published[s].data()), kVocab * 4);
          }
      }
      const auto decode = Seconds(Clock::now() - decode_begin);
      if (teacher) {
        rows.flush();
        Require(bool(rows), "complete teacher rows failed");
      }
      Save<llama_token>(out / "chosen.i32", choices);
      std::vector<float> final;
      for (const auto& row : published) final.insert(final.end(), row.begin(), row.end());
      Save<float>(out / "final.f32", final);
      Require(backend_tokens == (teacher ? 80U : 96U), "backend sample count differs");
      if (teacher) {
        Require(shapes.joined && shapes.joined_prefill,
                "ordinary stock C2 FLASH geometry not observed");
        for (const auto& [key, count] : shapes.counts) {
          std::cout << "GEMMA2_JOINT_PREFILL_STOCK_FLASH count=" << count;
          for (auto value : key) std::cout << ' ' << value;
          std::cout << '\n';
        }
      }
      std::cout << "GEMMA2_JOINT_PREFILL_STOCK mode=" << mode
                << " slots=2 context_per_slot=8192 chunk=" << chunk
                << " compatible_prefill=1 max_wave_rows=" << 2 * chunk
                << " prompt_rows0=" << prefix[0] << " prompt_rows1=" << prefix[1]
                << " untimed_rows_per_slot=3"
                << " decode_steps=32 departure_steps=" << (teacher ? kTail : 0)
                << " paid_generated_tokens=64 past0=" << past[0] << " past1=" << past[1]
                << " prefill_seconds=" << prefill << " decode_seconds=" << decode
                << " joined_prefill_groups=" << prefill_groups
                << " joined_prefill_rows=" << prefill_rows << " backend_tokens=" << backend_tokens
                << " sampled_logits_min=" << sampled_min << " sampled_logits_max=" << sampled_max
                << " observer=" << teacher << " tokenized_equal=2 final_head_paid=" << !teacher
                << '\n';
    }
    llama_backend_free();
    std::cout << "GEMMA2_JOINT_PREFILL_STOCK_RETIRED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
