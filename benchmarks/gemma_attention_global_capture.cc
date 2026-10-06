// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// ARTIFACT OUTPUT_DIR 31 4 joined norm IDS_I32 capture|control
// One real first-global operand acquisition; untimed and native-origin only.
// Bounded common-prefix units: every completed owner head is published/charged.
#include "gemma_attention_global_capture.h"

#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
using en::support::Error;
int main(int argc, char** argv) {
  umask(0077);
  if (argc != 9) return 2;
  const std::string_view witness_mode = argv[8];
  if (witness_mode != "capture" && witness_mode != "control") return 2;
  std::vector<std::int32_t> supplied;
  {
    supplied.resize(1024);
    std::ifstream file(argv[7], std::ios::binary);
    file.read(reinterpret_cast<char*>(supplied.data()), 1024 * sizeof(std::int32_t));
    if (!file || file.peek() != std::char_traits<char>::eof() || supplied[0] != 2 ||
        !std::ranges::all_of(supplied, [](auto id) { return id >= 0 && id < 262144; }))
      return 2;
  }
  const std::uint32_t prompt_rows = supplied.empty() ? 6 : 64;
  const std::string_view variant = argv[3], mode = argv[5], policy = argv[6];
  std::uint32_t count = 0;
  const std::string_view number = argv[4];
  const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), count);
  if (error != std::errc{} || end != number.data() + number.size() || count == 0 || count > 12 ||
      (variant != "26" && variant != "31") || (mode != "scalar" && mode != "joined") ||
      (policy != "ordinary" && policy != "rows" && policy != "rows-norm" && policy != "norm"))
    return 2;
  if ((policy == "rows-norm" || policy == "norm") &&
      (variant != "31" || count != 4 || supplied.empty()))
    return 2;
  if (variant != "31" || count != 4 || mode != "joined" || policy != "norm") return 2;
  const std::filesystem::path out = argv[2];
  std::error_code file_error;
  if (!std::filesystem::create_directory(out, file_error) || file_error) return 2;
  if (!supplied.empty()) {
    std::ofstream file(out / "inputs.i32", std::ios::binary);
    file.write(reinterpret_cast<const char*>(supplied.data()), 1024 * sizeof(std::int32_t));
    file.flush();
    if (!file) return 2;
  }
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
    jitllm::benchmark::GemmaAttentionGlobalCapture capture;
    bool metadata_charged = false;
  };
  auto lifetime = std::make_unique<Lifetime>();
  auto& node = lifetime->node;
  lifetime->runner = std::make_unique<en::Gemma4Runner>(
      node,
      en::Gemma4Options{
          .artifact = argv[1],
          .out = out,
          .variant = variant == "26" ? en::Gemma4Variant::k26BA4B : en::Gemma4Variant::k31B,
          .context = 256,
          .max_rows = 128,
          .slots = count,
          .fuse_norms = false,
          .row_invariant = policy == "rows" || policy == "rows-norm",
          .fuse_norm_rope = policy == "rows-norm" || policy == "norm",
          .fuse_norm_add = policy == "rows-norm" || policy == "norm"},
      0, 0);
  auto& runner = *lifetime->runner;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    constexpr auto metadata_bytes = jitllm::benchmark::GemmaAttentionGlobalCapture::kHostBytes;
    if (!node.ChargeHost(metadata_bytes, false)) return Error("capture metadata funding refused");
    lifetime->metadata_charged = true;
    // Metadata and pinned snapshot descriptors outlive execute(). Keep the
    // host charge with the complete bundle until teardown proves retirement.
    if (auto r = lifetime->capture.Setup(node, witness_mode == "capture"); !r) return r;
    jitllm::benchmark::GemmaAttentionGlobalCapture::active = &lifetime->capture;
    lifetime->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    constexpr std::uint32_t steps = 1, vocab = 262144;
    const auto heap = std::uint64_t{count} * (steps + 2) * vocab * sizeof(float);
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto budget = fixed + runner.weights().size() * en::kPagedExtent +
                        2 * node.StateCapacity() + heap + metadata_bytes;
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes() + heap +
                      metadata_bytes);
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    std::array<std::uint32_t, 12> ids{};
    for (std::uint32_t i = 0; i < count; ++i) ids[i] = i;
    if (auto r = runner.SelectSlots(std::span(ids).first(count)); !r) return r;
    if (!node.ChargeHost(heap, false)) return Error("owner publication buffers do not fit");
    struct HostGrant {
      en::PagedNode& node;
      std::uint64_t bytes;
      ~HostGrant() { node.UnchargeHost(bytes); }
    } grant{node, heap};
    return node.WithRequest(0, runner.closure(), "Gemma joined fixed-prefix screen", [&]() {
      return [&]() -> en::Status {
        std::array<std::vector<float>, 12> heads;
        std::array<std::uint32_t, 12> past{};
        std::array<std::int32_t, 12> anchors{};
        constexpr std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
        constexpr std::array<std::int32_t, 3> seeds{45518, 107, 101};
        std::vector<float> published(std::size_t{steps} * count * vocab);
        std::array<std::array<std::int32_t, 12>, steps> chosen{};
        const auto report_policy = [&](std::string_view phase, std::uint32_t owner,
                                       std::uint32_t rows, std::uint32_t segments) -> en::Status {
          const auto& p = runner.last_built_policy();
          if (p.rows != rows || p.segments != segments)
            return Error("fresh selected policy does not match requested shape");
          std::cout << "JOINED_SELECTED phase=" << phase << " owner=" << owner << " rows=" << p.rows
                    << " segments=" << p.segments << " row_products=" << p.row_products
                    << " norm_rope=" << p.norm_rope << " norm_add=" << p.norm_add
                    << " norm_fused=" << p.norm_fused << " rope_store=" << p.rope_store
                    << " shared_vecq=" << p.shared_vecq << " gemma_route=" << p.gemma_route
                    << " gemma_reduce=" << p.gemma_reduce << " lane_steps=" << p.lane_steps << '\n';
          return {};
        };
        const auto prefill = [&](bool fresh) -> en::Status {
          for (std::uint32_t i = 0; i < count; ++i) {
            if (auto r = runner.Clear(i); !r) return r;
            std::vector<std::int32_t> tokens;
            if (supplied.empty()) {
              tokens.assign(prompt.begin(), prompt.end());
              tokens.insert(tokens.end(), i, 563);
            } else {
              tokens.assign(supplied.begin(), supplied.begin() + prompt_rows + i);
            }
            past[i] = static_cast<std::uint32_t>(tokens.size());
            const en::Gemma4Runner::Work work{i, 0, tokens, &heads[i]};
            if (auto r = runner.Wave(std::span(&work, 1)); !r) return r;
            if (fresh)
              if (auto r = report_policy("prefill-first-build", i, past[i], 1); !r) return r;
          }
          return {};
        };
        bool fresh_decode = true;
        const auto wave = [&](std::uint32_t step) -> en::Status {
          lifetime->capture.BeginWave();
          std::array<en::Gemma4Runner::Work, 12> work{};
          for (std::uint32_t i = 0; i < count; ++i) {
            anchors[i] = supplied.empty() ? seeds[(step + i) % seeds.size()]
                                          : supplied[prompt_rows + i + step];
            work[i] = {i, past[i], std::span(&anchors[i], 1), &heads[i]};
          }
          if (mode == "joined") {
            for (std::uint32_t first = 0; first < count; first += en::kGemma4InvariantWaveRows) {
              const auto rows = std::min(en::kGemma4InvariantWaveRows, count - first);
              if (auto r = runner.Wave(std::span(work).subspan(first, rows)); !r) return r;
              if (fresh_decode)
                if (auto r = report_policy("decode-first-build", first, rows, rows); !r) return r;
            }
          } else {
            for (std::uint32_t i = 0; i < count; ++i) {
              if (auto r = runner.Wave(std::span(work).subspan(i, 1)); !r) return r;
              if (fresh_decode)
                if (auto r = report_policy("decode-first-build", i, 1, 1); !r) return r;
            }
          }
          fresh_decode = false;
          for (std::uint32_t i = 0; i < count; ++i) ++past[i];
          return {};
        };
        if (auto r = prefill(true); !r) return r;
        for (std::uint32_t step = 0; step < 8; ++step)
          if (auto r = wave(step); !r) return r;
        if (auto r = prefill(false); !r) return r;
        for (std::uint32_t step = 0; step < 3; ++step)
          if (auto r = wave(step); !r) return r;
        const auto before = runner.graph_stats();
        const auto started = std::chrono::steady_clock::now();
        for (std::uint32_t step = 0; step < steps; ++step) {
          if (auto r = wave(step + 3); !r) return r;
          for (std::uint32_t i = 0; i < count; ++i) {
            if (heads[i].size() != vocab) return Error("incomplete owner head publication");
            if (!std::ranges::all_of(heads[i], [](float x) { return std::isfinite(x); }))
              return Error("nonfinite completed owner head");
            chosen[step][i] = static_cast<std::int32_t>(
                std::max_element(heads[i].begin(), heads[i].end()) - heads[i].begin());
            std::copy(heads[i].begin(), heads[i].end(),
                      published.data() + (std::size_t{step} * count + i) * vocab);
          }
        }
        const auto elapsed = en::support::Seconds(std::chrono::steady_clock::now() - started);
        std::ofstream file(out / "heads.f32", std::ios::binary);
        file.write(reinterpret_cast<const char*>(published.data()),
                   static_cast<std::streamsize>(published.size() * sizeof(float)));
        file.flush();
        if (!file) return Error("writing completed owner heads failed");
        if (auto saved = lifetime->capture.Save(out); !saved) return saved;
        const auto& p = runner.last_built_policy();
        std::cout << "ATTENTION_ACQUISITION untimed=1 variant=" << variant << " owners=" << count
                  << " mode=" << mode << " policy=" << policy
                  << " acquisition_elapsed_seconds=" << elapsed << " completed_waves=" << steps
                  << " completed_units=" << steps * count << " paid_gpu_groups="
                  << steps * (mode == "scalar" ? count
                                               : (count + en::kGemma4InvariantWaveRows - 1) /
                                                     en::kGemma4InvariantWaveRows)
                  << " max_shared_rows=" << en::kGemma4InvariantWaveRows
                  << " first_past=" << prompt_rows + 3
                  << " input_mode=" << (supplied.empty() ? "synthetic" : "supplied")
                  << " unequal_past=" << (count > 1) << " context=256 max_rows=128"
                  << " captured_delta=" << runner.graph_stats().captured - before.captured
                  << " replayed_delta=" << runner.graph_stats().replayed - before.replayed
                  << " rows=" << p.rows << " segments=" << p.segments
                  << " row_products=" << p.row_products << " norm_rope=" << p.norm_rope
                  << " norm_add=" << p.norm_add << " norm_fused=" << p.norm_fused
                  << " rope_store=" << p.rope_store << " shared_vecq=" << p.shared_vecq
                  << " gemma_route=" << p.gemma_route << " gemma_reduce=" << p.gemma_reduce
                  << " policy_basis=last-built lane_steps=" << p.lane_steps
                  << " heap_funded=" << heap << '\n';
        for (std::uint32_t step = 0; step < steps; ++step)
          for (std::uint32_t i = 0; i < count; ++i)
            std::cout << "JOINED_TOKEN step=" << step << " owner=" << i
                      << " argmax=" << chosen[step][i] << " forced="
                      << (supplied.empty() ? seeds[(step + 3 + i) % seeds.size()]
                                           : supplied[prompt_rows + i + step + 3])
                      << '\n';
        return {};
      }();
    });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(lifetime->entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = lifetime.release();
  }
  if (retired) {
    jitllm::benchmark::GemmaAttentionGlobalCapture::active = nullptr;
    if (lifetime->metadata_charged) {
      node.UnchargeHost(jitllm::benchmark::GemmaAttentionGlobalCapture::kHostBytes);
      lifetime->metadata_charged = false;
    }
  }
  return ran && retired ? 0 : 1;
}
