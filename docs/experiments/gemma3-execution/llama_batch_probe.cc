// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original-image public API: paired prefill and four-owner/solo decode controls.
// MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT teacher|cycle|solo|departure
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
constexpr int kVocab = 262208, kSteps = 32, kTail = 4, kOwners = 4;
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
  bool joined = false, joined_three = false, joined_prefill = false;
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
    self.joined |= key[0] == 256 && key[1] == 1 && key[2] == 8 && key[3] == kOwners &&
                   key[4] == 256 && key[5] >= 256 && key[6] == 4 && key[7] == kOwners &&
                   key[8] == key[5] && key[11] == kOwners;
    self.joined_three |= key[0] == 256 && key[1] == 1 && key[2] == 8 && key[3] == 3 &&
                         key[4] == 256 && key[5] >= 256 && key[6] == 4 && key[7] == 3 &&
                         key[8] == key[5] && key[11] == 3;
    self.joined_prefill |= key[0] == 256 && key[1] == 128 && key[2] == 8 && key[3] == 2 &&
                           key[4] == 256 && key[6] == 4 && key[7] == 2;
    return false;  // Metadata only: never request operand download.
  }
};
double Seconds(Clock::duration value) { return std::chrono::duration<double>(value).count(); }
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 8, "MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT teacher|cycle|solo|departure");
    const std::string mode = argv[7];
    const bool solo = mode == "solo", departure = mode == "departure",
               teacher = mode == "teacher" || solo || departure;
    Require(teacher || mode == "cycle", "unknown mode");
    Require(!std::getenv("GGML_CUDA_DISABLE_FUSION") && !std::getenv("GGML_CUDA_DISABLE_GRAPHS"),
            "stock overrides present");
    std::array<std::vector<llama_token>, kOwners> ids;
    std::array<int, kOwners> prefix{};
    std::array<std::string, kOwners> texts;
    for (std::size_t s = 0; s < kOwners; ++s) {
      const auto raw = Read(argv[2 + (s % 2) * 2], 4096 * sizeof(llama_token));
      Require(raw.size() % 4 == 0 && raw.size() >= 295 * 4, "bounded IDs required");
      ids[s].resize(raw.size() / 4);
      prefix[s] = static_cast<int>(ids[s].size()) - 3 - kSteps - kTail;
      std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(ids[s].data()));
      Require(ids[s][0] == 2 &&
                  std::ranges::all_of(ids[s], [](auto id) { return id >= 0 && id < kVocab; }),
              "invalid tokens");
      texts[s] = Read(argv[3 + (s % 2) * 2], 65536);
    }
    Require(ids[0] != ids[1] && prefix[0] == 256 && prefix[1] == 256,
            "different authenticated 256-row input identities required");
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
      Require(llama_vocab_n_tokens(vocab) == kVocab, "vocabulary differs");
      for (std::size_t s = 0; s < kOwners; ++s) {
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
      std::array<std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>, kOwners> samplers{
          std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(nullptr,
                                                                        llama_sampler_free),
          std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(nullptr,
                                                                        llama_sampler_free),
          std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(nullptr,
                                                                        llama_sampler_free),
          std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(nullptr,
                                                                        llama_sampler_free)};
      std::array<llama_sampler_seq_config, kOwners> configs;
      for (int s = 0; s < kOwners; ++s) {
        samplers[s].reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        Require(bool(samplers[s]), "sampler allocation failed");
        llama_sampler_chain_add(samplers[s].get(), llama_sampler_init_greedy());
        Require(llama_sampler_chain_n(samplers[s].get()) == 1, "exact greedy chain required");
        configs[s] = {s, samplers[s].get()};
      }
      Shapes shapes;
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
      // Factual shape observer exists only in teacher runs, never either timing arm.
      if (teacher) {
        cp.cb_eval = Shapes::Observe;
        cp.cb_eval_user_data = &shapes;
      }
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
          llama_init_from_model(model.get(), cp), llama_free);
      Require(ctx && llama_n_ctx_seq(ctx.get()) == 4096 && llama_n_seq_max(ctx.get()) == kOwners,
              "C4 context geometry differs");
      struct Batch {
        llama_batch value = llama_batch_init(256, 0, 1);
        ~Batch() { llama_batch_free(value); }
      } batch;
      std::array<int, kOwners> past{};
      std::array<llama_token, kOwners> selected{-1, -1, -1, -1};
      std::array<std::vector<float>, kOwners> published;
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
      const auto joined = [&](const std::array<llama_token, kOwners>& tokens, int count = kOwners) {
        if (solo) {
          for (int s = 0; s < kOwners; ++s) independent(s, std::span(&tokens[s], 1), true);
          return;
        }
        batch.value.n_tokens = count;
        for (int s = 0; s < count; ++s) add(s, s, tokens[s], past[s], true);
        Require(llama_decode(ctx.get(), batch.value) == 0, "joined decode failed");
        for (int s = 0; s < count; ++s) {
          publish(s, s);
          ++past[s];
        }
      };
      std::uint64_t prefill_groups = 0, prefill_rows = 0;
      int c3_waves = 0, catchup_rows = 0, rejoined_waves = 0;
      const auto prompt = [&] {
        for (int first = 0; first < kOwners; first += 2) {
          while (past[first] < prefix[first]) {
            const int rows = std::min(128, prefix[first] - past[first]);
            const bool head = past[first] + rows == prefix[first];
            std::array<int, 2> last_rows{};
            batch.value.n_tokens = 0;
            for (int offset = 0; offset < 2; ++offset) {
              const int slot = first + offset;
              Require(past[slot] == past[first] && prefix[slot] == prefix[first],
                      "paired prompt geometry differs");
              for (int i = 0; i < rows; ++i) {
                const int index = batch.value.n_tokens++;
                add(index, slot, ids[slot][past[slot] + i], past[slot] + i, head && i + 1 == rows);
                last_rows[offset] = index;
              }
            }
            Require(llama_decode(ctx.get(), batch.value) == 0, "paired prefill failed");
            for (int offset = 0; offset < 2; ++offset) {
              const int slot = first + offset;
              if (head)
                publish(slot, last_rows[offset]);
              else
                published[slot].clear();
              past[slot] += rows;
            }
            ++prefill_groups;
            prefill_rows += static_cast<std::uint64_t>(batch.value.n_tokens);
          }
        }
      };
      const auto warm = [&] {
        for (int s = 0; s < kOwners; ++s)
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
      warm();
      std::ofstream rows;
      const auto write_rows = [&] {
        for (const auto& row : published)
          rows.write(reinterpret_cast<const char*>(row.data()), kVocab * 4);
      };
      std::array<bool, (kSteps + kTail + 1) * kOwners> written{};
      const auto write_head = [&](int index, int slot) {
        Require(index >= 0 && index < static_cast<int>(written.size()) && slot >= 0 &&
                    slot < kOwners && !written[index] && published[slot].size() == kVocab,
                "departure row publication invalid or duplicate");
        Require(std::ranges::all_of(published[slot], [](float x) { return std::isfinite(x); }),
                "departure row is nonfinite");
        rows.seekp(static_cast<std::streamoff>(std::uint64_t(index) * kVocab * sizeof(float)));
        rows.write(reinterpret_cast<const char*>(published[slot].data()), kVocab * 4);
        Require(bool(rows), "departure row publication failed");
        written[index] = true;
      };
      if (teacher) {
        rows.open(out / "heads.f32", std::ios::binary | std::ios::noreplace);
        Require(bool(rows), "teacher rows must be new");
        if (departure) {
          for (int s = 0; s < kOwners; ++s) write_head(s, s);
        } else
          write_rows();
      }
      std::vector<llama_token> choices((kSteps + (teacher ? kTail : 0)) * kOwners);
      const auto decode_begin = Clock::now();
      for (int i = 0; i < kSteps; ++i) {
        if (departure && i == 16) {
          Require(past[3] == 267, "paused stock cursor differs");
          for (int j = 8; j < 16; ++j) {
            choices[j * kOwners + 3] = selected[3];
            independent(3, std::span(ids[3]).subspan(past[3], 1), true);
            write_head((j + 1) * kOwners + 3, 3);
            ++catchup_rows;
          }
          Require(std::ranges::all_of(past, [](int n) { return n == 275; }),
                  "stock catchup/rejoin cursors differ");
        }
        const int count = departure && i >= 8 && i < 16 ? 3 : kOwners;
        std::array<llama_token, kOwners> tokens{};
        for (int s = 0; s < count; ++s) {
          choices[static_cast<std::size_t>(i * kOwners + s)] = selected[s];
          tokens[s] = teacher ? ids[s][prefix[s] + 3 + i] : selected[s];
        }
        final_head = !teacher && i + 1 == kSteps;
        joined(tokens, count);
        if (teacher) {
          if (departure) {
            for (int s = 0; s < count; ++s) write_head((i + 1) * kOwners + s, s);
          } else
            write_rows();
        }
        if (departure && count == 3) ++c3_waves;
        if (departure && i >= 16) ++rejoined_waves;
      }
      if (teacher) {
        for (int s = 0; s < kOwners; ++s)
          for (int i = 0; i < kTail; ++i) {
            choices[kSteps * kOwners + s * kTail + i] = selected[s];
            independent(s, std::span(ids[s]).subspan(past[s], 1), true);
            if (departure)
              write_head((kSteps + 1) * kOwners + s * kTail + i, s);
            else
              rows.write(reinterpret_cast<const char*>(published[s].data()), kVocab * 4);
          }
      }
      const auto decode = Seconds(Clock::now() - decode_begin);
      if (teacher) {
        Require(!departure || std::ranges::all_of(written, [](bool n) { return n; }),
                "departure canonical row coverage incomplete");
        rows.flush();
        Require(bool(rows), "complete teacher rows failed");
      }
      Save<llama_token>(out / "chosen.i32", choices);
      std::vector<float> final;
      for (const auto& row : published) final.insert(final.end(), row.begin(), row.end());
      Save<float>(out / "final.f32", final);
      Require(backend_tokens == (teacher ? 40U : 48U) * kOwners, "backend sample count differs");
      if (teacher) {
        Require(
            (solo || shapes.joined) && (!departure || shapes.joined_three) && shapes.joined_prefill,
            "ordinary stock C4 FLASH geometry not observed");
        for (const auto& [key, count] : shapes.counts) {
          std::cout << "GEMMA3_C4_BATCH_STOCK_FLASH count=" << count;
          for (auto value : key) std::cout << ' ' << value;
          std::cout << '\n';
        }
      }
      std::cout << "GEMMA3_C4_BATCH_STOCK mode=" << mode
                << " slots=4 context_per_slot=4096 chunk=128 input_identities=2"
                << " compatible_prefill=1 max_wave_rows=256 prompt_rows0=" << prefix[0]
                << " prompt_rows1=" << prefix[1] << " untimed_rows_per_slot=3"
                << " decode_steps=32 departure_steps=" << (teacher ? kTail : 0)
                << " paid_generated_tokens=128 past0=" << past[0] << " past1=" << past[1]
                << " past2=" << past[2] << " past3=" << past[3] << " prefill_seconds=" << prefill
                << " decode_seconds=" << decode << " joined_prefill_groups=" << prefill_groups
                << " joined_prefill_rows=" << prefill_rows << " backend_tokens=" << backend_tokens
                << " sampled_logits_min=" << sampled_min << " sampled_logits_max=" << sampled_max
                << " c3_waves=" << c3_waves << " catchup_rows=" << catchup_rows
                << " rejoined_waves=" << rejoined_waves << " observer=" << teacher
                << " tokenized_equal=4 final_head_paid=" << !teacher << '\n';
    }
    llama_backend_free();
    std::cout << "GEMMA3_C4_BATCH_STOCK_RETIRED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
