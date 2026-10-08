// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded public-API control over the original llama.cpp v0.6.0 libraries.
// MODEL INPUT_I32 NEW_OUT [gemma26|gemma31]: profile-sized prefill and four
// supplied scalar steps, then fresh prefill over the same logical prefix.
#include <algorithm>
#include <cmath>
#include <cstdint>
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
constexpr int kVocab = 262144;

void Require(bool good, const char* message) {
  if (!good) throw std::runtime_error(message);
}

void Save(const std::filesystem::path& path, std::span<const float> row) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(row.data()),
             static_cast<std::streamsize>(row.size_bytes()));
  file.flush();
  Require(bool(file), "full head publication failed");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 4 || argc == 5, "MODEL INPUT_I32 NEW_OUT [gemma26|gemma31]");
    const std::string_view profile = argc == 5 ? argv[4] : "gemma26";
    Require(profile == "gemma26" || profile == "gemma31", "unknown profile");
    const bool dense = profile == "gemma31";
    const int prefix = dense ? 128 : 256, count = prefix + 4;
    const int batch_rows = dense ? 256 : 1024;
    const std::filesystem::path out(argv[3]);
    Require(!std::filesystem::exists(out), "output already exists");
    std::vector<std::int32_t> input(static_cast<std::size_t>(count));
    std::ifstream file(argv[2], std::ios::binary);
    file.read(reinterpret_cast<char*>(input.data()),
              static_cast<std::streamsize>(input.size() * sizeof(std::int32_t)));
    Require(bool(file) && file.peek() == std::char_traits<char>::eof() && input[0] == 2 &&
                std::ranges::all_of(input, [](auto token) { return token >= 0 && token < kVocab; }),
            "input must be exactly the profile count of valid IDs with leading BOS");
    Require(std::filesystem::create_directory(out), "exclusive output refused");
    std::filesystem::permissions(out, std::filesystem::perms::owner_all);
    llama_backend_init();
    {
      auto mp = llama_model_default_params();
      mp.n_gpu_layers = 999;
      mp.load_mode = LLAMA_LOAD_MODE_NONE;
      mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
      std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
          llama_model_load_from_file(argv[1], mp), llama_model_free);
      Require(model && llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == kVocab &&
                  llama_model_n_layer(model.get()) == (dense ? 60 : 30) &&
                  llama_model_n_embd(model.get()) == (dense ? 5376 : 2816),
              "checkpoint is not the approved Gemma4 shape");
      auto cp = llama_context_default_params();
      cp.n_ctx = 8192;
      cp.n_batch = cp.n_ubatch = static_cast<std::uint32_t>(batch_rows);
      cp.n_seq_max = 1;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
      cp.type_k = cp.type_v = GGML_TYPE_F16;
      cp.swa_full = false;
      cp.kv_unified = false;
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
          llama_init_from_model(model.get(), cp), llama_free);
      Require(ctx && llama_n_ctx_seq(ctx.get()) >= 8192 && llama_n_seq_max(ctx.get()) == 1,
              "context does not match the bounded single-request geometry");
      struct Batch {
        llama_batch value;
        explicit Batch(int rows) : value(llama_batch_init(rows, 0, 1)) {}
        ~Batch() { llama_batch_free(value); }
      } batch(batch_rows);
      const auto decode = [&](int first, int count) {
        batch.value.n_tokens = count;
        for (int i = 0; i < count; ++i) {
          batch.value.token[i] = input[static_cast<std::size_t>(first + i)];
          batch.value.pos[i] = first + i;
          batch.value.n_seq_id[i] = 1;
          batch.value.seq_id[i][0] = 0;
          batch.value.logits[i] = i == count - 1;
        }
        Require(llama_decode(ctx.get(), batch.value) == 0, "supplied-token decode failed");
      };
      const auto capture = [&](const std::string& name) {
        const auto* row = llama_get_logits_ith(ctx.get(), batch.value.n_tokens - 1);
        Require(row != nullptr, "completed full head missing");
        const std::span<const float> values(row, kVocab);
        Require(std::ranges::all_of(values, [](float x) { return std::isfinite(x); }),
                "nonfinite head");
        Save(out / name, values);
        std::cout << name << " rows=1 vocab=" << kVocab
                  << " choice=" << std::max_element(values.begin(), values.end()) - values.begin()
                  << '\n';
      };
      for (int repeat = 0; repeat < 2; ++repeat) {
        const std::string label = repeat == 0 ? "" : "1-";
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        decode(0, prefix);
        capture(label + "prefix" + std::to_string(prefix) + ".f32");
        int natural_matches = 0;
        for (int i = prefix; i < count; ++i) {
          const auto* row = llama_get_logits_ith(ctx.get(), batch.value.n_tokens - 1);
          Require(row != nullptr, "pre-step head missing");
          Require(std::all_of(row, row + kVocab, [](float x) { return std::isfinite(x); }),
                  "nonfinite pre-step head");
          const auto choice = std::max_element(row, row + kVocab) - row;
          const bool match = choice == input[static_cast<std::size_t>(i)];
          natural_matches += match;
          std::cout << "SUPPLIED_STEP repeat=" << repeat << " position=" << i
                    << " natural_match=" << match << '\n';
          decode(i, 1);
        }
        std::cout << "NATURAL_PREFIX repeat=" << repeat << " matched=" << natural_matches
                  << " total=4\n";
        capture(label + "scalar" + std::to_string(count) + ".f32");
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        decode(0, count);
        capture(label + "fresh" + std::to_string(count) + ".f32");
      }
    }
    llama_backend_free();
    std::cout << "CONTINUATION_RETIRED context=0 model=0 prefix=" << count << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
