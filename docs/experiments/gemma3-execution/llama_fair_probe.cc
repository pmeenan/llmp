// SPDX-FileCopyrightText: 2026 jitLLM contributors
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
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// State-only calls retain ordinary public decode overlap; final publication completes work.
#include "llama.h"
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kVocab = 262208, kPrompt = 256, kWarm = 3, kSteps = 32;
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
struct Shapes {
  using Key = std::array<std::int64_t, 12>;
  std::map<Key, std::uint64_t> counts;
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
    return false;  // Metadata only; never download operands.
  }
  void CheckDepth() const {
    bool local = false, global = false;
    for (const auto& [key, count] : counts) {
      std::cout << "GEMMA3_DEPTH_FLASH count=" << count;
      for (const auto value : key) std::cout << ' ' << value;
      std::cout << '\n';
      const bool scalar = key[0] == 256 && key[1] == 1 && key[2] == 8 && key[3] == 1 &&
                          key[4] == 256 && key[6] == 4 && key[7] == 1 && key[8] == key[5] &&
                          key[9] == 1 && key[10] == 1 && key[11] == 1;
      local |= scalar && key[5] == 1280;
      global |= scalar && key[5] == 8448;
    }
    Require(local && global, "depth scalar local/global FLASH geometry not observed");
  }
};
double Seconds(Clock::duration value) { return std::chrono::duration<double>(value).count(); }
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 6 || (argc == 7 && std::string_view(argv[6]) == "depth"),
            "MODEL IDS_I32 PROMPT_TEXT NEW_OUT teacher|cycle|greedy-teacher|greedy-cycle");
    const bool depth = argc == 7;
    const int context = depth ? 8448 : 4096, prompt_rows = depth ? 8192 : kPrompt,
              warm_rows = depth ? 0 : kWarm, steps = depth ? 64 : kSteps;
    const int input_rows = prompt_rows + warm_rows + steps;
    const std::string mode = argv[5];
    const bool backend_sampling = mode == "greedy-teacher" || mode == "greedy-cycle";
    const bool teacher = mode == "teacher" || mode == "greedy-teacher";
    Require(teacher || mode == "cycle" || mode == "greedy-cycle", "unknown mode");
    Require(!std::getenv("GGML_CUDA_DISABLE_FUSION") && !std::getenv("GGML_CUDA_DISABLE_GRAPHS"),
            "stock fusion/graph override present");
    const auto raw = Read(argv[2], input_rows * sizeof(llama_token));
    Require(raw.size() == input_rows * sizeof(llama_token), "exact recipe input rows required");
    std::vector<llama_token> ids(static_cast<std::size_t>(input_rows));
    std::copy_n(reinterpret_cast<const char*>(raw.data()), raw.size(),
                reinterpret_cast<char*>(ids.data()));
    Require(ids.front() == 2 &&
                std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < kVocab; }),
            "invalid token input");
    const auto text = Read(argv[3], depth ? 131072 : 65536);
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
      Require(count < 0 && count >= (depth ? -32768 : -8192), "bounded token count refused");
      std::vector<llama_token> tokenized(static_cast<std::size_t>(-count));
      Require(llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokenized.data(),
                             -count, true, false) == -count &&
                  tokenized.size() >= static_cast<std::size_t>(input_rows) &&
                  std::equal(ids.begin(), ids.end(), tokenized.begin()),
              "native/stock input IDs differ");
      Shapes shapes;
      auto cp = llama_context_default_params();
      cp.n_ctx = static_cast<std::uint32_t>(context);
      cp.n_batch = cp.n_ubatch = 128;
      cp.n_seq_max = 1;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
      cp.type_k = cp.type_v = GGML_TYPE_F16;
      cp.swa_full = false;
      cp.kv_unified = false;
      cp.no_perf = false;
      if (depth && teacher) {
        cp.cb_eval = Shapes::Observe;
        cp.cb_eval_user_data = &shapes;
      }
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
      Require(ctx && llama_n_ctx_seq(ctx.get()) == static_cast<std::uint32_t>(context) &&
                  llama_n_seq_max(ctx.get()) == 1,
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
          published.clear();
        }
        past += static_cast<int>(tokens.size());
      };
      const auto prompt = [&]() {
        for (int at = 0; at < prompt_rows; at += 128)
          decode(std::span(ids).subspan(static_cast<std::size_t>(at), 128),
                 at + 128 == prompt_rows);
      };
      const auto warm = [&]() {
        for (int i = 0; i < warm_rows; ++i)
          decode(std::span(ids).subspan(prompt_rows + i, 1), true);
      };
      llama_memory_clear(llama_get_memory(ctx.get()), !depth);
      if (!teacher) {
        prompt();
        warm();
        for (int i = 0; i < 8; ++i) {
          const auto token = best();
          decode(std::span(&token, 1), true);
        }
        llama_memory_clear(llama_get_memory(ctx.get()), !depth);
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
      std::vector<llama_token> chosen(static_cast<std::size_t>(steps));
      const auto decode_start = Clock::now();
      for (int i = 0; i < steps; ++i) {
        const auto next = best();
        chosen[static_cast<std::size_t>(i)] = next;
        const auto token = !teacher ? next : ids[prompt_rows + warm_rows + i];
        final_head = !teacher && i + 1 == steps;
        decode(std::span(&token, 1), true);
        if (teacher)
          heads.write(reinterpret_cast<const char*>(published.data()), kVocab * sizeof(float));
      }
      const double elapsed = Seconds(Clock::now() - decode_start);
      if (teacher) {
        heads.flush();
        Require(bool(heads), "complete head write failed");
      }
      if (depth && teacher) shapes.CheckDepth();
      Save<llama_token>(out / "chosen.i32", chosen);
      Save<float>(out / "final.f32", published);
      const auto after = llama_perf_context(ctx.get());
      std::cout << "GEMMA3_STOCK mode=" << mode << " context=" << context
                << " slots=1 chunk=128 prompt_rows=" << prompt_rows << " untimed_rows=" << warm_rows
                << " decode_rows=" << steps << " past=" << past << " prefill_seconds=" << prefill
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
