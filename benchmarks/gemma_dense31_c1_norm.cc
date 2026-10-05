// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// ARTIFACT NEW_OUTPUT_DIR ordinary|norm IDS_I32; closed dense31 scalar C1.
// Bounded common-prefix units: every completed owner head is published/charged.
#include <algorithm>
#include <array>
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
  if (argc != 5) return 2;
  const std::string_view policy = argv[3];
  if (policy != "ordinary" && policy != "norm") return 2;
  constexpr std::uint32_t count = 1, prompt_rows = 64;
  std::vector<std::int32_t> supplied(1024);
  std::ifstream input(argv[4], std::ios::binary);
  input.read(reinterpret_cast<char*>(supplied.data()), 1024 * sizeof(std::int32_t));
  if (!input || input.peek() != std::char_traits<char>::eof() || supplied[0] != 2 ||
      !std::ranges::all_of(supplied, [](auto id) { return id >= 0 && id < 262144; }))
    return 2;
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
  };
  auto lifetime = std::make_unique<Lifetime>();
  auto& node = lifetime->node;
  lifetime->runner =
      std::make_unique<en::Gemma4Runner>(node,
                                         en::Gemma4Options{.artifact = argv[1],
                                                           .out = out,
                                                           .variant = en::Gemma4Variant::k31B,
                                                           .context = 256,
                                                           .max_rows = 128,
                                                           .slots = count,
                                                           .row_invariant = false,
                                                           .fuse_norm_rope = policy == "norm",
                                                           .fuse_norm_add = policy == "norm"},
                                         0, 0);
  auto& runner = *lifetime->runner;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    lifetime->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    constexpr std::uint32_t steps = 32, vocab = 262144;
    const auto heap = std::uint64_t{count} * (steps + 2) * vocab * sizeof(float);
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto budget =
        fixed + runner.weights().size() * en::kPagedExtent + 2 * node.StateCapacity() + heap;
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes() + heap);
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    std::array<std::uint32_t, 1> ids{};
    for (std::uint32_t i = 0; i < count; ++i) ids[i] = i;
    if (auto r = runner.SelectSlots(std::span(ids).first(count)); !r) return r;
    if (!node.ChargeHost(heap, false)) return Error("owner publication buffers do not fit");
    struct HostGrant {
      en::PagedNode& node;
      std::uint64_t bytes;
      ~HostGrant() { node.UnchargeHost(bytes); }
    } grant{node, heap};
    return node.WithRequest(0, runner.closure(), "Gemma dense31 C1 fixed-prefix screen", [&]() {
      return [&]() -> en::Status {
        std::array<std::vector<float>, 1> heads;
        std::array<std::uint32_t, 1> past{};
        std::array<std::int32_t, 1> anchors{};
        std::vector<float> published(std::size_t{steps} * count * vocab);
        std::array<std::array<std::int32_t, 1>, steps> chosen{};
        const auto report_policy = [&](std::string_view phase, std::uint32_t owner,
                                       std::uint32_t rows, std::uint32_t segments) -> en::Status {
          const auto& p = runner.last_built_policy();
          const auto norms = policy == "norm" ? 120U : 0U;
          if (p.rows != rows || p.segments != segments || p.norm_rope != norms ||
              p.norm_add != norms || p.row_products != 0 || p.norm_fused != 0 ||
              p.rope_store != 0 || p.shared_vecq != 0 || p.gemma_route != 0 || p.gemma_reduce != 0)
            return Error("fresh selected policy does not match closed C1 shape/flags");
          std::cout << "C1_SELECTED phase=" << phase << " owner=" << owner << " rows=" << p.rows
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
            std::vector<std::int32_t> tokens(supplied.begin(), supplied.begin() + prompt_rows);
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
          std::array<en::Gemma4Runner::Work, 1> work{};
          for (std::uint32_t i = 0; i < count; ++i) {
            anchors[i] = supplied[prompt_rows + i + step];
            work[i] = {i, past[i], std::span(&anchors[i], 1), &heads[i]};
          }
          if (auto r = runner.Wave(work); !r) return r;
          if (fresh_decode)
            if (auto r = report_policy("decode-first-build", 0, 1, 1); !r) return r;
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
            chosen[step][i] = static_cast<std::int32_t>(
                std::max_element(heads[i].begin(), heads[i].end()) - heads[i].begin());
            std::copy(heads[i].begin(), heads[i].end(),
                      published.data() + (std::size_t{step} * count + i) * vocab);
          }
        }
        const auto elapsed = en::support::Seconds(std::chrono::steady_clock::now() - started);
        // Matching reference validation is outside its paid interval as well.
        if (!std::ranges::all_of(published, [](float x) { return std::isfinite(x); }))
          return Error("non-finite completed C1 heads");
        std::ofstream file(out / "heads.f32", std::ios::binary);
        file.write(reinterpret_cast<const char*>(published.data()),
                   static_cast<std::streamsize>(published.size() * sizeof(float)));
        file.flush();
        if (!file) return Error("writing completed owner heads failed");
        const auto& p = runner.last_built_policy();
        std::cout << "C1_NATIVE variant=31 owners=1 mode=scalar policy=" << policy
                  << " seconds=" << elapsed << " completed_waves=" << steps
                  << " completed_units=" << steps << " paid_gpu_groups=" << steps
                  << " first_past=" << prompt_rows + 3
                  << " input_mode=supplied unequal_past=0 context=256 max_rows=128"
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
            std::cout << "C1_TOKEN step=" << step << " owner=" << i << " argmax=" << chosen[step][i]
                      << " forced=" << supplied[prompt_rows + i + step + 3] << '\n';
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
  return ran && retired ? 0 : 1;
}
