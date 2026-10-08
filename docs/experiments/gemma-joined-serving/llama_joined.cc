// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// MODEL OUTPUT_DIR OWNERS UBATCH [IDS_I32]
// New: MODEL OUT OWNERS CAP IDS_12X1024 production 26|31 scalar|joined.
// Real independent-sequence stock C API batch; C12 uses one physical twelve-row batch.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "llama.h"

namespace {
void Require(bool good, const char* detail) {
  if (!good) throw std::runtime_error(detail);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const bool production = argc == 9 && std::string_view(argv[6]) == "production";
    Require(argc == 5 || argc == 6 || production,
            "MODEL OUT OWNERS UBATCH [IDS] [production 26|31 scalar|joined]");
    constexpr int history_rows = 1024, prefill_rows = 992;
    const std::string_view variant = production ? argv[7] : "";
    const std::string_view mode = production ? argv[8] : "joined";
    const int supplied_rows = production ? 12 * history_rows : history_rows;
    std::vector<std::int32_t> supplied;
    if (argc >= 6) {
      supplied.resize(static_cast<std::size_t>(supplied_rows));
      std::ifstream file(argv[5], std::ios::binary);
      file.read(reinterpret_cast<char*>(supplied.data()), supplied_rows * sizeof(std::int32_t));
      Require(bool(file) && file.peek() == std::char_traits<char>::eof() && supplied[0] == 2 &&
                  std::ranges::all_of(supplied, [](auto id) { return id >= 0 && id < 262144; }),
              "invalid 1024-token supplied prefix");
    }
    if (production)
      for (int owner = 0; owner < 12; ++owner)
        Require(supplied[static_cast<std::size_t>(owner * history_rows)] == 2 &&
                    std::count(supplied.begin() + owner * history_rows,
                               supplied.begin() + (owner + 1) * history_rows, 2) == 1,
                "each production history requires one leading BOS");
    const int prompt_rows = production ? prefill_rows : supplied.empty() ? 6 : 64;
    const auto count = std::stoi(argv[3]), ubatch = std::stoi(argv[4]);
    Require(count >= 1 && count <= 12 && ubatch >= count && ubatch <= (production ? 1024 : 128),
            "unbounded owners or physical ubatch");
    if (production)
      Require((count == 4 || count == 8 || count == 12) && (variant == "26" || variant == "31") &&
                  (mode == "scalar" || mode == "joined") &&
                  ubatch == (variant == "26" ? 1024 : 256),
              "production recipe requires an approved profile, cohort and prefill cap");
    const int batch_rows = production ? ubatch : 128;
    Require(!std::getenv("GGML_CUDA_DISABLE_GRAPHS") && !std::getenv("GGML_CUDA_DISABLE_FUSION"),
            "stock graphs/fusion disabled");
    const std::filesystem::path out = argv[2];
    Require(std::filesystem::create_directory(out), "output must be new");
    if (!supplied.empty()) {
      std::ofstream file(out / "inputs.i32", std::ios::binary);
      file.write(reinterpret_cast<const char*>(supplied.data()),
                 supplied_rows * sizeof(std::int32_t));
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
    if (production)
      Require(llama_model_n_layer(model.get()) == (variant == "26" ? 30 : 60) &&
                  llama_model_n_embd(model.get()) == (variant == "26" ? 2816 : 5376),
              "production profile differs from the loaded checkpoint");
    auto cp = llama_context_default_params();
    cp.n_ctx = static_cast<std::uint32_t>(count * (production ? 4096 : 256));
    cp.n_batch = static_cast<std::uint32_t>(batch_rows);
    cp.n_ubatch = static_cast<std::uint32_t>(ubatch);
    cp.n_seq_max = static_cast<std::uint32_t>(count);
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = false;
    if (production) {
      cp.swa_full = false;
      cp.kv_unified = false;
    }
    std::cout << "JOINED_REFERENCE_PROFILE recipe=" << (production ? "production" : "historical")
              << " variant=" << variant << " mode=" << mode << " context=" << cp.n_ctx
              << " batch=" << cp.n_batch << " ubatch=" << cp.n_ubatch << " swa_full=" << cp.swa_full
              << " kv_unified=" << cp.kv_unified << " type_k=F16 type_v=F16\n";
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    Require(bool(ctx), "context creation failed");
    Require(llama_n_ctx_seq(ctx.get()) >= static_cast<std::uint32_t>(production ? 4096 : 256) &&
                llama_n_seq_max(ctx.get()) == static_cast<std::uint32_t>(count),
            "context was accidentally split below the per-owner bound");
    llama_batch batch = llama_batch_init(batch_rows, 0, 1);
    constexpr int vocab = 262144, steps = 32;
    constexpr std::array<llama_token, 6> prompt{2, 818, 5279, 529, 7001, 563};
    constexpr std::array<llama_token, 3> seeds{45518, 107, 101};
    std::vector<std::vector<float>> heads(count, std::vector<float>(vocab));
    std::array<int, 12> past{};
    std::array<std::array<llama_token, 12>, steps> chosen{};
    std::vector<float> published(static_cast<std::size_t>(steps + (production ? 1 : 0)) * count *
                                 vocab);
    const auto prefill = [&]() {
      llama_memory_clear(llama_get_memory(ctx.get()), true);
      for (int owner = 0; owner < count; ++owner) {
        if (production) {
          for (int first = 0; first < prefill_rows; first += ubatch) {
            batch.n_tokens = std::min(ubatch, prefill_rows - first);
            const bool final = first + batch.n_tokens == prefill_rows;
            for (int i = 0; i < batch.n_tokens; ++i) {
              batch.token[i] = supplied[static_cast<std::size_t>(owner * history_rows + first + i)];
              batch.pos[i] = first + i;
              batch.n_seq_id[i] = 1;
              batch.seq_id[i][0] = owner;
              batch.logits[i] = final && i == batch.n_tokens - 1;
            }
            Require(llama_decode(ctx.get(), batch) == 0, "production owner prefill failed");
            if (final) {
              const auto* row = llama_get_logits_ith(ctx.get(), batch.n_tokens - 1);
              Require(row != nullptr, "missing production frontier head");
              std::copy_n(row, vocab, heads[owner].begin());
            }
          }
          past[owner] = prefill_rows;
          continue;
        }
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
      // Normal joined stock work stays one physical cohort, including C12.
      const int group = production && mode == "scalar" ? 1 : count;
      for (int first = 0; first < count; first += group) {
        batch.n_tokens = std::min(group, count - first);
        for (int index = 0; index < batch.n_tokens; ++index) {
          const int owner = first + index;
          batch.token[index] =
              production
                  ? supplied[static_cast<std::size_t>(owner * history_rows + prefill_rows + step)]
              : supplied.empty() ? seeds[(step + owner) % seeds.size()]
                                 : supplied[prompt_rows + owner + step];
          batch.pos[index] = past[owner];
          batch.n_seq_id[index] = 1;
          batch.seq_id[index][0] = owner;
          batch.logits[index] = true;
        }
        Require(llama_decode(ctx.get(), batch) == 0, "joined multi-sequence decode failed");
        for (int index = 0; index < batch.n_tokens; ++index) {
          const int owner = first + index;
          const auto* row = llama_get_logits_ith(ctx.get(), index);
          Require(row != nullptr, "missing completed owner head");
          std::copy_n(row, vocab, heads[owner].begin());
          ++past[owner];
        }
      }
    };
    prefill();
    for (int step = 0; step < 8; ++step) wave(step);
    prefill();
    if (production) {
      for (int owner = 0; owner < count; ++owner) {
        Require(std::ranges::all_of(heads[owner], [](float value) { return std::isfinite(value); }),
                "non-finite production frontier");
        std::copy(heads[owner].begin(), heads[owner].end(),
                  published.begin() + static_cast<std::size_t>(owner) * vocab);
        std::cout << "JOINED_FRONTIER owner=" << owner << " completed=" << prefill_rows
                  << " argmax="
                  << std::max_element(heads[owner].begin(), heads[owner].end()) -
                         heads[owner].begin()
                  << '\n';
      }
    } else {
      for (int step = 0; step < 3; ++step) wave(step);
    }
    const auto before = llama_perf_context(ctx.get());
    const auto started = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
      wave(step + (production ? 0 : 3));
      for (int owner = 0; owner < count; ++owner) {
        chosen[step][owner] = static_cast<llama_token>(
            std::max_element(heads[owner].begin(), heads[owner].end()) - heads[owner].begin());
        std::copy(
            heads[owner].begin(), heads[owner].end(),
            published.begin() +
                (static_cast<std::size_t>(step + (production ? 1 : 0)) * count + owner) * vocab);
      }
    }
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto after = llama_perf_context(ctx.get());
    if (production)
      Require(std::ranges::all_of(published, [](float value) { return std::isfinite(value); }),
              "non-finite production completed head capture");
    std::ofstream file(out / (production ? "cohort.logits.f32" : "heads.f32"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(published.data()),
               static_cast<std::streamsize>(published.size() * sizeof(float)));
    file.flush();
    Require(bool(file), "writing completed owner heads failed");
    std::cout << "JOINED_REFERENCE owners=" << count << " seconds=" << elapsed
              << " completed_waves=" << steps << " completed_units=" << steps * count
              << " first_past=" << prompt_rows + (production ? 0 : 3)
              << " input_mode=" << (supplied.empty() ? "synthetic" : "supplied")
              << " unequal_past=" << (!production && count > 1)
              << " context=" << llama_n_ctx(ctx.get())
              << " per_sequence_context=" << llama_n_ctx_seq(ctx.get()) << " batch=" << batch_rows
              << " ubatch=" << ubatch << " recipe=" << (production ? "production" : "historical")
              << " mode=" << mode << " retained_rows=" << (steps + (production ? 1 : 0)) * count
              << " physical_decode_groups=" << (mode == "scalar" ? count : 1)
              << " seq_max=" << llama_n_seq_max(ctx.get())
              << " fusion=enabled graphs=allowed reused_delta=" << after.n_reused - before.n_reused
              << '\n';
    for (int step = 0; step < steps; ++step)
      for (int owner = 0; owner < count; ++owner)
        std::cout
            << "JOINED_TOKEN step=" << step << " owner=" << owner
            << " argmax=" << chosen[step][owner] << " forced="
            << (production
                    ? supplied[static_cast<std::size_t>(owner * history_rows + prefill_rows + step)]
                : supplied.empty() ? seeds[(step + 3 + owner) % seeds.size()]
                                   : supplied[prompt_rows + owner + step + 3])
            << '\n';
    if (production) {
      // Keep teardown diagnostics from interleaving unfinished stdout records.
      std::cout.flush();
      Require(bool(std::cout), "publishing production records failed");
    }
    if (production) llama_synchronize(ctx.get());
    llama_batch_free(batch);
    ctx.reset();
    model.reset();
    llama_backend_free();
    if (production) {
      std::cout << "JOINED_REFERENCE_RETIRED production=1\n";
      std::cout.flush();
      Require(bool(std::cout), "publishing production retirement failed");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
