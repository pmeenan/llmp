// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original-image public API for the fixed grouped H8 schedule.
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
namespace wide = jitllm::benchmarks::gemma3_wide;
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
struct Shapes {
  std::array<std::uint64_t, 13> decode{};
  std::uint64_t prefill = 0;
  static bool Observe(ggml_tensor* node, bool ask, void* opaque) {
    if (!ask || node->op != GGML_OP_FLASH_ATTN_EXT) return false;
    auto& s = *static_cast<Shapes*>(opaque);
    const auto* q = node->src[0];
    const auto* k = node->src[1];
    const auto* mask = node->src[3];
    if (q && k && mask && q->ne[0] == 256 && q->ne[2] == 8 && k->ne[0] == 256 && k->ne[2] == 4 &&
        k->ne[3] == q->ne[3] && mask->ne[3] == q->ne[3] && mask->ne[0] == k->ne[1]) {
      if (q->ne[1] == 1 && q->ne[3] >= 4 && q->ne[3] <= 12)
        ++s.decode[static_cast<std::size_t>(q->ne[3])];
      if (q->ne[1] == 128 && q->ne[3] == 2) ++s.prefill;
      std::cout << "GEMMA3_WIDE_STOCK_FLASH queries=" << q->ne[1] << " logical=" << q->ne[3]
                << " cells=" << k->ne[1] << '\n';
    }
    return false;  // Metadata only; no callback downloads.
  }
};
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
    const auto events = wide::Schedule(prefix);
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
      cp.cb_eval = Shapes::Observe;
      cp.cb_eval_user_data = &shapes;
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
          llama_init_from_model(model.get(), cp), llama_free);
      Require(ctx && llama_n_ctx_seq(ctx.get()) == 4096 && llama_n_seq_max(ctx.get()) == kOwners,
              "context geometry differs");
      struct Batch {
        llama_batch value = llama_batch_init(256, 0, 1);
        ~Batch() { llama_batch_free(value); }
      } batch;
      std::ofstream rows(out / "heads.f32", std::ios::binary | std::ios::noreplace);
      std::ofstream map(out / "rows.tsv", std::ios::noreplace);
      Require(bool(rows) && bool(map), "exclusive teacher outputs refused");
      std::array<std::uint32_t, kOwners> past{};
      std::array<std::vector<float>, kOwners> final;
      std::vector<llama_token> choices;
      std::uint64_t backend_tokens = 0, prefill_groups = 0, prefill_rows = 0;
      std::uint32_t sampled_min = kVocab, sampled_max = 0;
      std::array<std::uint64_t, 13> waves{};
      for (const auto& event : events) {
        Require(past == event.before, "schedule target/cursor mapping differs");
        if (event.kind == wide::Kind::kRefillBegin) {
          for (std::uint32_t s = 10; s < kOwners; ++s) {
            Require(llama_memory_seq_rm(llama_get_memory(ctx.get()), static_cast<llama_seq_id>(s),
                                        -1, -1),
                    "logical request reset failed");
            llama_sampler_reset(samplers[s].get());
            past[s] = 0;
          }
          continue;
        }
        if (event.kind != wide::Kind::kPrefill && event.kind != wide::Kind::kScalar &&
            event.kind != wide::Kind::kWave)
          continue;
        const auto first = event.first, owners = event.count;
        batch.value.n_tokens = 0;
        std::array<int, kOwners> outputs;
        for (std::uint32_t s = first; s < first + owners; ++s) {
          Require(past[s] + event.rows <= ids[s].size(), "input bounds exceeded");
          for (std::uint32_t i = 0; i < event.rows; ++i) {
            const int index = batch.value.n_tokens++;
            const auto token = ids[s][past[s] + i];
            batch.value.token[index] = token;
            batch.value.pos[index] = static_cast<llama_pos>(past[s] + i);
            batch.value.n_seq_id[index] = 1;
            batch.value.seq_id[index][0] = static_cast<llama_seq_id>(s);
            batch.value.logits[index] = event.head && i + 1 == event.rows;
            llama_sampler_accept(samplers[s].get(), token);
            outputs[s] = index;
          }
        }
        Require(llama_decode(ctx.get(), batch.value) == 0, "decode failed");
        if (event.kind == wide::Kind::kWave) ++waves[owners];
        if (event.kind == wide::Kind::kPrefill) {
          ++prefill_groups;
          prefill_rows += event.rows * owners;
        }
        for (std::uint32_t s = first; s < first + owners; ++s) {
          past[s] += event.rows;
          if (!event.head) continue;
          const auto token = llama_get_sampled_token_ith(ctx.get(), outputs[s]);
          const auto count = llama_get_sampled_logits_count_ith(ctx.get(), outputs[s]);
          sampled_min = std::min(sampled_min, count);
          sampled_max = std::max(sampled_max, count);
          const auto* head = llama_get_sampled_logits_ith(ctx.get(), outputs[s]);
          Require(head && count == kVocab && token >= 0 && token < std::int32_t(kVocab) &&
                      std::all_of(
                          head, head + kVocab, [](float x) { return std::isfinite(x); }) &&
                      token == std::max_element(head, head + kVocab) - head,
                  "finite complete sampled head required");
          choices.push_back(token);
          ++backend_tokens;
          rows.write(reinterpret_cast<const char*>(head), kVocab * 4);
          const auto target = past[s] < ids[s].size() ? ids[s][past[s]] : -1;
          map << event.phase << '\t' << static_cast<int>(event.kind) << '\t' << s << '\t' << past[s]
              << '\t' << target << '\n';
          final[s].assign(head, head + kVocab);
        }
      }
      Require(choices.size() == 472 && prefill_groups == 30 && prefill_rows == 7680,
              "wide publication or prefill counters differ");
      rows.flush();
      map.flush();
      Require(bool(rows) && bool(map), "complete teacher outputs failed");
      Save<llama_token>(out / "chosen.i32", choices);
      std::vector<float> last;
      for (const auto& head : final) last.insert(last.end(), head.begin(), head.end());
      Save<float>(out / "final.f32", last);
      for (std::uint32_t n = 4; n <= 12; ++n) {
        Require(waves[n] && shapes.decode[n], "actual physical cohort witness missing");
        std::cout << "GEMMA3_WIDE_STOCK_PHASE logical=" << n << " actual_waves=" << waves[n]
                  << " flash_shapes=" << shapes.decode[n] << '\n';
      }
      Require(shapes.prefill && backend_tokens == choices.size() && sampled_min == kVocab &&
                  sampled_max == kVocab,
              "full original backend sampling export missing");
      std::cout << "GEMMA3_WIDE_STOCK_PASS rows=" << choices.size() << " slots=12 context=4096"
                << " per_owner_rows=128 wave_rows=256 input_identities=2 prefill_groups="
                << prefill_groups << " prefill_rows=" << prefill_rows
                << " backend_tokens=" << backend_tokens << " sampled_logits_min=" << sampled_min
                << " sampled_logits_max=" << sampled_max << '\n';
    }
    llama_backend_free();
    std::cout << "GEMMA3_WIDE_STOCK_RETIRED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "GEMMA3_WIDE_STOCK_FAIL " << e.what() << '\n';
    return 1;
  }
}
