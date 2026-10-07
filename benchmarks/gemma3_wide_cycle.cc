// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// One internal C12 natural cycle; quality is qualified separately, no public admission.
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/gemma3_runner.h"
#include "engine/planned.h"
#include "engine/support.h"
#include "gemma3_wide_schedule.h"
#include "platform/crash_policy.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace en = jitllm::engine;
namespace fs = std::filesystem;
using en::support::Error;
namespace wide = jitllm::benchmarks::gemma3_wide;
constexpr auto kVocab = wide::kVocab, kOwners = wide::kOwners;
std::expected<std::string, std::string> Read(const fs::path& path, std::uint64_t cap) {
  std::error_code error;
  const auto bytes = fs::file_size(path, error);
  if (error || bytes > cap) return Error("bounded input size refused");
  std::string data(static_cast<std::size_t>(bytes), '\0');
  std::ifstream file(path, std::ios::binary);
  if (!file.read(data.data(), static_cast<std::streamsize>(data.size())) ||
      file.peek() != std::char_traits<char>::eof())
    return Error("input read changed or failed");
  return data;
}
template <class T>
en::Status Save(const fs::path& path, std::span<const T> data) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size_bytes()));
  file.flush();
  return file ? en::Status{} : Error("exclusive complete output refused");
}
std::expected<std::int32_t, std::string> Best(const std::vector<float>& row) {
  if (row.size() != kVocab || !std::ranges::all_of(row, [](float x) { return std::isfinite(x); }))
    return Error("finite complete head required");
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
}
}  // namespace
int main(int argc, char** argv) {
  if (!jitllm::platform::InstallCrashPolicy("gemma3-wide-cycle") || (argc != 5 && argc != 6))
    return 2;
  const bool bounded = argc == 6;
  if (bounded && std::string_view(argv[5]) != "bounded-whole12") return 2;
  std::array<std::vector<std::int32_t>, kOwners> ids;
  std::array<std::uint32_t, kOwners> prefix{}, past{}, slots{};
  for (std::uint32_t s = 0; s < kOwners; ++s) {
    auto raw = Read(argv[2 + wide::SourceIndex(s)], 4096 * 4);
    if (!raw || raw->size() != (wide::SourceIndex(s) ? 807U : 295U) * 4) return 2;
    ids[s].resize(raw->size() / 4);
    std::memcpy(ids[s].data(), raw->data(), raw->size());
    prefix[s] = static_cast<std::uint32_t>(ids[s].size()) - 39;
    slots[s] = s;
    if (ids[s][0] != 2 ||
        !std::ranges::all_of(ids[s], [](auto t) { return t >= 0 && t < std::int32_t(kVocab); }))
      return 2;
  }
  const fs::path out = argv[4];
  if (!fs::create_directory(out)) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma3Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  life->runner = std::make_unique<en::Gemma3Runner>(node,
                                                    en::Gemma3Options{.artifact = argv[1],
                                                                      .out = out,
                                                                      .context = 4096,
                                                                      .max_rows = 128,
                                                                      .slots = kOwners,
                                                                      .max_wave_rows = 256,
                                                                      .max_head_rows = kOwners,
                                                                      .owner_decode = true,
                                                                      .packed_prefill = true,
                                                                      .bounded_roots = true,
                                                                      .bounded_whole12 = bounded,
                                                                      .fuse_norms = true,
                                                                      .fuse_quant_glu = true,
                                                                      .fuse_norm_rope = true,
                                                                      .fuse_norm_add = true},
                                                    0, 0);
  auto& runner = *life->runner;
  std::array<std::vector<float>, kOwners> heads;
  for (auto& row : heads) row.reserve(kVocab);
  std::array<std::int32_t, kOwners> selected{};
  std::uint64_t prefill_groups = 0, prefill_rows = 0;
  std::vector<std::int32_t> choices;
  choices.reserve(32 * kOwners);
  const auto prompt = [&]() -> en::Status {
    // Match the quality schedule: round-robin equal paired128 chunks, not a
    // twelve-owner prompt wave. The last chunk publishes device tokens only.
    for (std::uint32_t at = 0; at < 768; at += 128)
      for (std::uint32_t first = 0; first < kOwners; first += 2) {
        if (at >= prefix[first]) continue;
        const bool head = at + 128 == prefix[first];
        std::array<en::Gemma3Runner::Work, 2> work;
        for (std::uint32_t i = 0; i < 2; ++i) {
          const auto s = first + i;
          if (past[s] != at || prefix[s] != prefix[first]) return Error("paired cursor differs");
          work[i] = {s, at, std::span(ids[s]).subspan(at, 128), head ? nullptr : &heads[s],
                     head ? &selected[s] : nullptr};
        }
        if (auto r = runner.WavePrefill(work, head); !r) return r;
        for (auto& unit : work) {
          past[unit.slot] += 128;
          if (!head && !heads[unit.slot].empty()) return Error("state-only head leak");
        }
        ++prefill_groups;
        prefill_rows += 256;
      }
    return {};
  };
  const auto teacher = [&]() -> en::Status {
    for (std::uint32_t s = 0; s < kOwners; ++s)
      for (std::uint32_t i = 0; i < 3; ++i) {
        const en::Gemma3Runner::Work unit{s, past[s], std::span(ids[s]).subspan(past[s], 1),
                                          nullptr, &selected[s]};
        if (auto r = runner.Wave(std::span(&unit, 1)); !r) return r;
        ++past[s];
      }
    return {};
  };
  const auto wave = [&](bool token) -> en::Status {
    const auto tokens = selected;
    if (!std::ranges::all_of(tokens, [](auto t) { return t >= 0 && t < std::int32_t(kVocab); }))
      return Error("natural token out of range");
    std::array<en::Gemma3Runner::Work, kOwners> work;
    for (std::uint32_t s = 0; s < kOwners; ++s)
      work[s] = {s, past[s], std::span(&tokens[s], 1), token ? nullptr : &heads[s],
                 token ? &selected[s] : nullptr};
    if (auto r = runner.Wave(work); !r) return r;
    for (std::uint32_t s = 0; s < kOwners; ++s) ++past[s];
    return {};
  };
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    life->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + (32ULL << 20U));
    // Both phases execute24 paired chunks+36 scalar writes+8/32 joined waves.
    // One plan/graph per event conservatively funds every retained key without
    // claiming an unregistered reclaimer. Variants=1 and graph<=floor*32.
    constexpr std::uint64_t kEvents = 2 * (24 + 36) + 8 + 32;
    constexpr auto kRatio = 1 + en::kGraphNodeHostBytes / en::kPlanNodeHostBytes;
    if (runner.plan_floor_bytes() > (64ULL << 30U) / (kEvents * kRatio))
      return Error("finite cache envelope exceeds64GiB");
    const auto cache_budget = kEvents * kRatio * runner.plan_floor_bytes();
    if (auto r = node.Start(jitllm::base::Bytes(fixed + runner.weights().size() * en::kPagedExtent +
                                                node.StateCapacity() + cache_budget));
        !r)
      return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(0, runner.closure(), "Gemma3 C12 natural cycle", [&]() -> en::Status {
      if (!node.InRequest(0)) return Error("held request missing");
      if (auto r = runner.SelectSlots(slots); !r) return r;
      if (auto r = prompt(); !r) return r;
      if (auto r = teacher(); !r) return r;
      // Preserve the existing C4 warm recipe: five token-only waves then three
      // full heads, still exactly eight natural warm waves and one warm pass.
      for (std::uint32_t i = 0; i < 8; ++i) {
        if (auto r = wave(i < 5); !r) return r;
        if (i >= 5)
          for (std::uint32_t s = 0; s < kOwners; ++s) {
            auto best = Best(heads[s]);
            if (!best) return Error(best.error());
            selected[s] = *best;
          }
      }
      const auto kept = runner.state();
      for (std::uint32_t s = 0; s < kOwners; ++s)
        if (auto r = runner.Clear(s); !r) return r;
      past = {};
      if (runner.kept_state() != kept) return Error("logical Clear discarded backing");
      const auto before = runner.graph_stats();
      const auto begin = std::chrono::steady_clock::now();
      if (auto r = prompt(); !r) return r;
      if (auto r = teacher(); !r) return r;
      const auto prefill = en::support::Seconds(std::chrono::steady_clock::now() - begin);
      const auto middle = runner.graph_stats();
      const auto decode_begin = std::chrono::steady_clock::now();
      for (std::uint32_t i = 0; i < 32; ++i) {
        choices.insert(choices.end(), selected.begin(), selected.end());
        if (auto r = wave(i + 1 < 32); !r) return r;
      }
      const auto decode = en::support::Seconds(std::chrono::steady_clock::now() - decode_begin);
      const auto after = runner.graph_stats();
      for (std::uint32_t s = 0; s < kOwners; ++s) {
        if (!Best(heads[s]) || past[s] != prefix[s] + 35) return Error("final head/cursor differs");
      }
      const auto& bound = runner.plan_selections();
      if (choices.size() != 384 || runner.greedy_tokens() != 528 || prefill_groups != 48 ||
          prefill_rows != 12288 || !bound.owner_attention || !bound.packed_prefill_attention ||
          !bound.norm_rope || !bound.norm_add ||
          (bounded ? bound.bounded_owner_attention == 0 : bound.bounded_owner_attention != 0) ||
          runner.coverage().violations || !after.captured || !after.replayed ||
          bound.plans > kEvents || runner.plans_bytes() > cache_budget ||
          node.host_counted() != runner.plans_bytes() || node.host_charged() > cache_budget ||
          node.host_overcharges())
        return Error("C12 cycle controls missing");
      if (auto r = Save<std::int32_t>(out / "chosen.i32", choices); !r) return r;
      std::ofstream final(out / "final.f32", std::ios::binary | std::ios::noreplace);
      for (auto& row : heads) final.write(reinterpret_cast<const char*>(row.data()), kVocab * 4);
      final.flush();
      if (!final) return Error("final head publication failed");
      if (auto r = Save<std::uint32_t>(out / "past.u32", past); !r) return r;
      std::cout << "GEMMA3_WIDE_CYCLE slots=12 input_identities=2 context=4096 per_owner_rows=128 "
                   "wave_rows=256"
                << " paid_prefix_rows=6144 paid_teacher_rows=36 paid_generated_tokens=384 "
                   "decode_steps=32 final_heads_paid=12"
                << " prefill_seconds=" << prefill << " decode_seconds=" << decode
                << " gpu_tokens=" << runner.greedy_tokens()
                << " prefill_captured=" << middle.captured - before.captured
                << " prefill_replayed=" << middle.replayed - before.replayed
                << " prefill_eager=" << middle.eager - before.eager
                << " decode_captured=" << after.captured - middle.captured
                << " decode_replayed=" << after.replayed - middle.replayed
                << " decode_eager=" << after.eager - middle.eager
                << " selected_owner=" << bound.owner_attention << " bounded_whole12=" << bounded
                << " selected_bounded=" << bound.bounded_owner_attention
                << " selected_packed=" << bound.packed_prefill_attention
                << " actual_plans=" << bound.plans << " cache_event_bound=" << kEvents
                << " plan_graph_bytes=" << runner.plans_bytes() << " cache_budget=" << cache_budget
                << " host_overcharges=" << node.host_overcharges() << '\n';
      return {};
    });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(life->entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = life.release();
  }
  if (ran && retired) std::cout << "GEMMA3_WIDE_CYCLE_RETIRED\n";
  return ran && retired ? 0 : 1;
}
