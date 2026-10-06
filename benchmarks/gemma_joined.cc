// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// ARTIFACT OUTPUT_DIR 26|31 OWNERS scalar|joined ordinary|rows|rows-norm|norm [IDS_I32]
// New recipe: append IDS_12X1024 production; policies all (26) or norm (31).
// Every completed owner head is published/charged; historical prefixes remain unchanged.
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
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
using en::support::Error;
int main(int argc, char** argv) {
  const bool production = argc == 9 && std::string_view(argv[8]) == "production";
  if (argc != 7 && argc != 8 && !production) return 2;
  constexpr std::uint32_t history_rows = 1024, prefill_rows = 992;
  const std::uint32_t supplied_rows = production ? 12 * history_rows : history_rows;
  std::vector<std::int32_t> supplied;
  if (argc >= 8) {
    supplied.resize(supplied_rows);
    std::ifstream file(argv[7], std::ios::binary);
    file.read(reinterpret_cast<char*>(supplied.data()), supplied_rows * sizeof(std::int32_t));
    if (!file || file.peek() != std::char_traits<char>::eof() || supplied[0] != 2 ||
        !std::ranges::all_of(supplied, [](auto id) { return id >= 0 && id < 262144; }))
      return 2;
  }
  if (production)
    for (std::uint32_t owner = 0; owner < 12; ++owner)
      if (supplied[owner * history_rows] != 2 ||
          std::ranges::count(std::span(supplied).subspan(owner * history_rows, history_rows), 2) !=
              1)
        return 2;
  const std::uint32_t prompt_rows = production ? prefill_rows : supplied.empty() ? 6 : 64;
  const std::string_view variant = argv[3], mode = argv[5], policy = argv[6];
  std::uint32_t count = 0;
  const std::string_view number = argv[4];
  const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), count);
  if (error != std::errc{} || end != number.data() + number.size() || count == 0 || count > 12 ||
      (variant != "26" && variant != "31") || (mode != "scalar" && mode != "joined") ||
      (!production && policy != "ordinary" && policy != "rows" && policy != "rows-norm" &&
       policy != "norm"))
    return 2;
  if (production &&
      ((count != 4 && count != 8 && count != 12) || policy != (variant == "26" ? "all" : "norm")))
    return 2;
  if (!production && (policy == "rows-norm" || policy == "norm") &&
      (variant != "31" || count != 4 || supplied.empty()))
    return 2;
  if (!production && policy == "norm" && mode != "joined") return 2;
  const std::uint32_t max_rows = production ? (variant == "26" ? 1024 : 256) : 128;
  const std::filesystem::path out = argv[2];
  std::error_code file_error;
  if (!std::filesystem::create_directory(out, file_error) || file_error) return 2;
  if (!supplied.empty()) {
    std::ofstream file(out / "inputs.i32", std::ios::binary);
    file.write(reinterpret_cast<const char*>(supplied.data()),
               supplied_rows * sizeof(std::int32_t));
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
  lifetime->runner = std::make_unique<en::Gemma4Runner>(
      node,
      en::Gemma4Options{
          .artifact = argv[1],
          .out = out,
          .variant = variant == "26" ? en::Gemma4Variant::k26BA4B : en::Gemma4Variant::k31B,
          .context = production ? 4096U : 256U,
          .max_rows = max_rows,
          .slots = count,
          .max_head_rows = production ? count : 0U,
          .row_invariant = policy == "rows" || policy == "rows-norm",
          .fuse_norm_rope = production || policy == "rows-norm" || policy == "norm",
          .fuse_norm_add = production || policy == "rows-norm" || policy == "norm",
          .fuse_gemma_route = production && variant == "26",
          .fuse_gemma_reduce = production && variant == "26",
          .owner_attention = production},
      0, 0);
  auto& runner = *lifetime->runner;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    lifetime->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    constexpr std::uint32_t steps = 32, vocab = 262144;
    // Publication plus live owner heads; production also charges the retained
    // 12-history carrier and a bounded 64KiB checkpoint/stream metadata allowance.
    const auto heap = std::uint64_t{count} * (steps + 2) * vocab * sizeof(float) +
                      (production ? std::uint64_t{count} * vocab * sizeof(float) +
                                        supplied_rows * sizeof(std::int32_t) + 65536
                                  : 0);
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto budget =
        fixed + runner.weights().size() * en::kPagedExtent + 2 * node.StateCapacity() + heap;
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes() + heap);
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
        std::vector<float> published(std::size_t{steps + (production ? 1U : 0U)} * count * vocab);
        std::array<std::array<std::int32_t, 12>, steps> chosen{};
        const auto report_policy = [&](std::string_view phase, std::uint32_t owner,
                                       std::uint32_t rows, std::uint32_t segments) -> en::Status {
          const auto& p = runner.last_built_policy();
          if (!production && (p.rows != rows || p.segments != segments))
            return Error("fresh selected policy does not match requested shape");
          std::cout << "JOINED_SELECTED phase=" << phase << " owner=" << owner << " rows=" << p.rows
                    << " segments=" << p.segments << " requested_rows=" << rows
                    << " requested_segments=" << segments
                    << " policy_basis=last-built row_products=" << p.row_products
                    << " norm_rope=" << p.norm_rope << " norm_add=" << p.norm_add
                    << " norm_fused=" << p.norm_fused << " rope_store=" << p.rope_store
                    << " owner_attention_steps=" << p.owner_attention_steps
                    << " requested_cohort8_steps=" << p.requested_cohort8_steps
                    << " shared_vecq=" << p.shared_vecq << " gemma_route=" << p.gemma_route
                    << " gemma_reduce=" << p.gemma_reduce << " lane_steps=" << p.lane_steps << '\n';
          return {};
        };
        const auto prefill = [&](bool fresh) -> en::Status {
          for (std::uint32_t i = 0; i < count; ++i) {
            if (auto r = runner.Clear(i); !r) return r;
            if (production) {
              for (std::uint32_t first = 0; first < prefill_rows; first += max_rows) {
                const auto rows = std::min(max_rows, prefill_rows - first);
                const auto tokens = std::span(supplied).subspan(i * history_rows + first, rows);
                const en::Gemma4Runner::Work work{i, first, tokens, &heads[i]};
                const bool final = first + rows == prefill_rows;
                const auto next_rows = std::min(max_rows, prefill_rows - first - rows);
                const en::Gemma4Runner::PrefillNext next{i, next_rows};
                const auto prediction =
                    final ? std::span<const en::Gemma4Runner::PrefillNext>{} : std::span(&next, 1);
                if (auto r = runner.WavePrefill(std::span(&work, 1), final, prediction,
                                                first + rows + next_rows == prefill_rows);
                    !r)
                  return r;
                if (fresh)
                  if (auto r = report_policy("prefill-first-build", i, rows, 1); !r) return r;
              }
              past[i] = prefill_rows;
              continue;
            }
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
          std::array<en::Gemma4Runner::Work, 12> work{};
          for (std::uint32_t i = 0; i < count; ++i) {
            anchors[i] = production         ? supplied[i * history_rows + prefill_rows + step]
                         : supplied.empty() ? seeds[(step + i) % seeds.size()]
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
        const auto save_states = [&](std::uint32_t positions,
                                     std::string_view phase) -> en::Status {
          if (!production) return {};
          auto ranges = runner.CheckpointRanges(positions);
          if (!ranges) return Error(ranges.error());
          std::uint64_t bytes = 0;
          for (const auto& range : *ranges) bytes += range.bytes;
          std::ofstream layout(out / (std::string(phase) + ".layout.txt"));
          layout << runner.CheckpointLayoutId() << '\n';
          layout.flush();
          if (!layout) return Error("writing owner checkpoint layout failed");
          for (std::uint32_t owner = 0; owner < count; ++owner) {
            auto slot = runner.request_slot(owner);
            if (!slot || (*slot)->completed_positions() != positions)
              return Error("owner completed positions differ from recipe");
            std::vector<jitllm::catalog::ExtentId> staging;
            auto state = node.Pinned(bytes, 0, staging);
            if (!state) return Error(state.error());
            if (auto r = runner.CopyState(owner, *state, *ranges, true); !r) return r;
            std::ofstream checkpoint(
                out / (std::string(phase) + ".owner" + std::to_string(owner) + ".state.bin"),
                std::ios::binary);
            checkpoint.write(static_cast<const char*>(*state), static_cast<std::streamsize>(bytes));
            checkpoint.flush();
            const bool written = bool(checkpoint);
            if (auto r = node.FreePinned(*state); !r) return r;
            if (!written) return Error("writing owner checkpoint failed");
          }
          std::cout << "JOINED_STATE phase=" << phase << " positions=" << positions
                    << " owners=" << count << " bytes_per_owner=" << bytes << '\n';
          return {};
        };
        const auto retain = [&](std::uint32_t row, std::uint32_t owner) -> en::Status {
          if (heads[owner].size() != vocab) return Error("incomplete owner head publication");
          std::copy(heads[owner].begin(), heads[owner].end(),
                    published.data() + (std::size_t{row} * count + owner) * vocab);
          return {};
        };
        if (auto r = prefill(true); !r) return r;
        for (std::uint32_t step = 0; step < 8; ++step)
          if (auto r = wave(step); !r) return r;
        if (auto r = prefill(false); !r) return r;
        if (production) {
          for (std::uint32_t owner = 0; owner < count; ++owner) {
            if (!std::ranges::all_of(heads[owner],
                                     [](float value) { return std::isfinite(value); }))
              return Error("non-finite production frontier");
            if (auto r = retain(0, owner); !r) return r;
            std::cout << "JOINED_FRONTIER owner=" << owner << " completed=" << prefill_rows
                      << " argmax="
                      << std::max_element(heads[owner].begin(), heads[owner].end()) -
                             heads[owner].begin()
                      << '\n';
          }
          if (auto r = save_states(prefill_rows, "frontier"); !r) return r;
        } else {
          for (std::uint32_t step = 0; step < 3; ++step)
            if (auto r = wave(step); !r) return r;
        }
        const auto before = runner.graph_stats();
        const auto started = std::chrono::steady_clock::now();
        for (std::uint32_t step = 0; step < steps; ++step) {
          if (auto r = wave(step + (production ? 0U : 3U)); !r) return r;
          for (std::uint32_t i = 0; i < count; ++i) {
            if (heads[i].size() != vocab) return Error("incomplete owner head publication");
            chosen[step][i] = static_cast<std::int32_t>(
                std::max_element(heads[i].begin(), heads[i].end()) - heads[i].begin());
            if (production) {
              if (auto r = retain(step + 1, i); !r) return r;
            } else {
              std::copy(heads[i].begin(), heads[i].end(),
                        published.data() + (std::size_t{step} * count + i) * vocab);
            }
          }
        }
        const auto elapsed = en::support::Seconds(std::chrono::steady_clock::now() - started);
        if (production &&
            !std::ranges::all_of(published, [](float value) { return std::isfinite(value); }))
          return Error("non-finite production completed head capture");
        if (auto r = save_states(history_rows, "final"); !r) return r;
        std::ofstream file(out / (production ? "cohort.logits.f32" : "heads.f32"),
                           std::ios::binary);
        file.write(reinterpret_cast<const char*>(published.data()),
                   static_cast<std::streamsize>(published.size() * sizeof(float)));
        file.flush();
        if (!file) return Error("writing completed owner heads failed");
        const auto& p = runner.last_built_policy();
        std::cout << "JOINED_NATIVE variant=" << variant << " owners=" << count << " mode=" << mode
                  << " policy=" << policy << " seconds=" << elapsed << " completed_waves=" << steps
                  << " completed_units=" << steps * count << " paid_gpu_groups="
                  << steps * (mode == "scalar" ? count
                                               : (count + en::kGemma4InvariantWaveRows - 1) /
                                                     en::kGemma4InvariantWaveRows)
                  << " max_shared_rows=" << en::kGemma4InvariantWaveRows
                  << " first_past=" << prompt_rows + (production ? 0U : 3U)
                  << " input_mode=" << (supplied.empty() ? "synthetic" : "supplied")
                  << " unequal_past=" << (!production && count > 1)
                  << " context=" << (production ? 4096 : 256) << " max_rows=" << max_rows
                  << " max_head_rows=" << (production ? count : 0)
                  << " recipe=" << (production ? "production" : "historical")
                  << " retained_rows=" << (steps + (production ? 1 : 0)) * count
                  << " owner_requested=" << production << " prefill_prediction=" << production
                  << " lookahead_attempted=" << runner.lookahead_stats().attempted
                  << " lookahead_built=" << runner.lookahead_stats().built
                  << " lookahead_cached=" << runner.lookahead_stats().cached
                  << " lookahead_refused=" << runner.lookahead_stats().refused
                  << " captured_delta=" << runner.graph_stats().captured - before.captured
                  << " replayed_delta=" << runner.graph_stats().replayed - before.replayed
                  << " rows=" << p.rows << " segments=" << p.segments
                  << " row_products=" << p.row_products << " norm_rope=" << p.norm_rope
                  << " norm_add=" << p.norm_add << " norm_fused=" << p.norm_fused
                  << " rope_store=" << p.rope_store << " shared_vecq=" << p.shared_vecq
                  << " gemma_route=" << p.gemma_route << " gemma_reduce=" << p.gemma_reduce
                  << " owner_attention_steps=" << p.owner_attention_steps
                  << " requested_cohort8_steps=" << p.requested_cohort8_steps
                  << " policy_basis=last-built lane_steps=" << p.lane_steps
                  << " heap_funded=" << heap << '\n';
        for (std::uint32_t step = 0; step < steps; ++step)
          for (std::uint32_t i = 0; i < count; ++i)
            std::cout << "JOINED_TOKEN step=" << step << " owner=" << i
                      << " argmax=" << chosen[step][i] << " forced="
                      << (production         ? supplied[i * history_rows + prefill_rows + step]
                          : supplied.empty() ? seeds[(step + 3 + i) % seeds.size()]
                                             : supplied[prompt_rows + i + step + 3])
                      << '\n';
        if (production) {
          std::cout.flush();
          if (!std::cout) return Error("publishing production records failed");
        }
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
  if (production && ran && retired) {
    std::cout << "JOINED_NATIVE_RETIRED production=1\n";
    std::cout.flush();
    if (!std::cout) return 1;
  }
  return ran && retired ? 0 : 1;
}
