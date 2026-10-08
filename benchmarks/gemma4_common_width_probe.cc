// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded unequal-prefix/ring screen with partial cohort departure and restore.
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
#include "engine/checkpoint_file.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"
#include "platform/crash_policy.h"

namespace {
namespace en = llmp::engine;
namespace fs = std::filesystem;
using en::support::Error;
constexpr std::uint32_t kVocab = 262144, kSteps = 32;
constexpr std::uint64_t kCopy = 8ULL << 20U;
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
  if (!llmp::platform::InstallCrashPolicy("gemma4-common-width-probe") || (argc != 8 && argc != 9))
    return 2;
  const bool dense_shared_q8 = argc == 9 && std::string_view(argv[8]) == "dense-shared-q8";
  if (argc == 9 && !dense_shared_q8) return 2;
  const std::string profile = argv[5], policy = argv[6];
  if ((profile != "26" && profile != "31") ||
      (policy != "baseline" && policy != "candidate" && policy != "bounded"))
    return 2;
  const bool dense = profile == "31", common = policy != "baseline";
  const std::uint32_t chunk = dense ? 256U : 1024U;
  const std::string mode = argv[7];
  const bool own = mode == "own", cycle = mode == "cycle";
  if (!own && !cycle) return 2;
  std::array<std::vector<std::int32_t>, 2> ids;
  std::array<std::uint32_t, 2> prefix{};
  for (std::size_t s = 0; s < 2; ++s) {
    std::error_code ec;
    const auto bytes = fs::file_size(argv[2 + s], ec);
    if (ec || bytes % 4 || bytes < 291 * 4 || bytes > 3968 * 4) return 2;
    ids[s].resize(bytes / 4);
    prefix[s] = static_cast<std::uint32_t>(ids[s].size()) - 3 - kSteps;
    std::ifstream file(argv[2 + s], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(ids[s].data()),
                   static_cast<std::streamsize>(ids[s].size() * 4)) ||
        file.peek() != std::char_traits<char>::eof() || ids[s][0] != 2 ||
        !std::ranges::all_of(ids[s], [](auto id) { return id >= 0 && id < std::int32_t(kVocab); }))
      return 2;
  }
  if (ids[0] == ids[1]) return 2;
  const fs::path out = argv[4];
  if (!fs::create_directory(out)) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  life->runner = std::make_unique<en::Gemma4Runner>(
      node,
      en::Gemma4Options{.artifact = argv[1],
                        .out = out,
                        .variant = dense ? en::Gemma4Variant::k31B : en::Gemma4Variant::k26BA4B,
                        .context = 4096,
                        .max_rows = chunk,
                        .slots = 2,
                        .max_head_rows = 2,
                        .frontier_head = false,
                        .fuse_norms = true,
                        .dense_shared_q8 = dense_shared_q8,
                        .fuse_norm_rope = true,
                        .fuse_norm_add = true,
                        .fuse_gemma_route = !dense,
                        .fuse_gemma_reduce = !dense,
                        .fuse_quant_glu = dense,
                        .owner_attention = true,
                        .common_owner_reads = common,
                        .bounded_owner_roots = policy == "bounded"},
      0, 0);
  auto& runner = *life->runner;
  std::array<std::vector<float>, 2> heads;
  for (auto& head : heads) head.reserve(kVocab);
  std::array<std::uint32_t, 2> past{};
  std::array<std::int32_t, 2> selected{-1, -1};
  void* pinned = nullptr;
  const auto snapshot = [&]() -> std::expected<std::array<std::string, 2>, std::string> {
    std::array<std::string, 2> result;
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      auto ranges = runner.CheckpointRanges(past[slot]);
      if (!ranges) return Error(ranges.error());
      llmp::base::Sha256 hash;
      for (const auto& range : *ranges)
        for (std::uint64_t at = 0; at < range.bytes;) {
          auto part = range;
          part.offset += at;
          part.bytes = std::min(kCopy, range.bytes - at);
          en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kUnproven;
          auto copied = runner.CopyState(slot, pinned, std::span(&part, 1), true, &retirement);
          if (retirement == en::LiveState::CopyRetirement::kUnproven) node.KeepPinned(pinned);
          if (!copied) return Error(copied.error());
          hash.Update(std::span(static_cast<const std::byte*>(pinned), std::size_t(part.bytes)));
          at += part.bytes;
        }
      result[slot] = llmp::base::ToHex(hash.Finish());
    }
    return result;
  };
  const auto independent = [&](std::uint32_t s, std::span<const std::int32_t> tokens, bool head,
                               bool token) -> en::Status {
    const en::Gemma4Runner::Work work{s, past[s], tokens, token ? nullptr : &heads[s],
                                      token ? &selected[s] : nullptr};
    if (auto r = runner.WavePrefill(std::span(&work, 1), head); !r) return r;
    past[s] += static_cast<std::uint32_t>(tokens.size());
    if (head && !token && !Best(heads[s])) return Error("bad independent head");
    if (!head && !heads[s].empty()) return Error("state-only head leak");
    return {};
  };
  const auto prompt = [&](bool token) -> en::Status {
    for (std::uint32_t s = 0; s < 2; ++s) {
      for (std::uint32_t at = 0; at < prefix[s];) {
        const auto rows = std::min(chunk, prefix[s] - at);
        const bool last = at + rows == prefix[s];
        if (auto r = independent(s, std::span(ids[s]).subspan(at, rows), last, token && last); !r)
          return r;
        at += rows;
      }
    }
    return {};
  };
  const auto warm = [&](bool token) -> en::Status {
    for (std::uint32_t s = 0; s < 2; ++s)
      for (std::uint32_t i = 0; i < 3; ++i)
        if (auto r = independent(s, std::span(ids[s]).subspan(prefix[s] + i, 1), true, token); !r)
          return r;
    return {};
  };
  std::uint32_t owner_steps = 0, bounded_steps = 0;
  std::uint32_t prepared_steps = 0, q8_steps = 0, vecq_steps = 0;
  const auto joined = [&](const std::array<std::int32_t, 2>& tokens, bool token) -> en::Status {
    std::array<en::Gemma4Runner::Work, 2> work;
    for (std::uint32_t s = 0; s < 2; ++s)
      work[s] = {s, past[s], std::span(&tokens[s], 1), token ? nullptr : &heads[s],
                 token ? &selected[s] : nullptr};
    if (auto r = runner.Wave(work); !r) return r;
    owner_steps = std::max(owner_steps, runner.last_built_policy().owner_attention_steps);
    bounded_steps = std::max(bounded_steps, runner.last_built_policy().bounded_owner_steps);
    prepared_steps = std::max(prepared_steps, runner.last_built_policy().prepared_mmvq_products);
    q8_steps = std::max(q8_steps, runner.last_built_policy().q8_preparations);
    vecq_steps = std::max(vecq_steps, runner.last_built_policy().shared_vecq);
    for (std::uint32_t s = 0; s < 2; ++s) {
      ++past[s];
      if (!token && !Best(heads[s])) return Error("bad joined head");
    }
    return {};
  };
  const auto clear = [&]() -> en::Status {
    for (std::uint32_t s = 0; s < 2; ++s)
      if (auto r = runner.Clear(s); !r) return r;
    past = {};
    return {};
  };
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    life->entered.push_back(&runner);
    const auto setup_begin = std::chrono::steady_clock::now();
    if (auto r = runner.Setup(); !r) return r;
    std::cout << "GEMMA4_PREPARATION_SETUP seconds="
              << en::support::Seconds(std::chrono::steady_clock::now() - setup_begin) << '\n';
    // Both modes observe initialized state after the paid endpoint. Fund
    // the same bounded copy buffer before deriving the physical budget.
    std::vector<llmp::catalog::ExtentId> extents;
    auto allocation = node.Pinned(kCopy, 0, extents);
    if (!allocation) return Error(allocation.error());
    pinned = *allocation;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + (16ULL << 20U));
    const auto budget =
        fixed + runner.weights().size() * en::kPagedExtent + 4 * node.StateCapacity();
    std::cout << "GEMMA4_PREPARATION_BUDGET fixed=" << fixed
              << " weights=" << runner.weights().size() * en::kPagedExtent
              << " state_capacity=" << node.StateCapacity()
              << " activations=" << runner.activations_needed()
              << " scratch=" << runner.pool_needed() << " host_input=" << runner.host_input_bytes()
              << " plan_floor=" << runner.plan_floor_bytes() << " total=" << budget << '\n';
    if (auto r = node.Start(llmp::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(0, runner.closure(), "Gemma4 C2 decode screen", [&]() -> en::Status {
      if (!node.InRequest(0)) return Error("held direct request required");
      const std::array<std::uint32_t, 2> slots{0, 1};
      if (auto r = runner.SelectSlots(slots); !r) return r;
      std::array<llmp::model::Gemma4Segment, 2> frontier;
      for (std::uint32_t slot = 0; slot < 2; ++slot)
        frontier[slot] = {slot, prefix[slot] + 3,
                          std::span(ids[slot]).subspan(prefix[slot] + 3, 1)};
      auto inputs = llmp::model::Gemma4Chunk(runner.profile(), runner.layout(), frontier);
      if (!inputs) return Error(inputs.error());
      std::uint32_t expected_owners = 0, expected_bounded = 0;
      for (std::uint32_t layer = 0; layer < runner.profile().layers; ++layer) {
        const bool local = runner.profile().local(layer);
        const auto first = local ? inputs->segments[0].local_n_kv : inputs->segments[0].global_n_kv;
        const auto second =
            local ? inputs->segments[1].local_n_kv : inputs->segments[1].global_n_kv;
        expected_owners += common || first == second;
        expected_bounded += policy == "bounded" && first != second;
      }
      for (const auto& segment : inputs->segments) {
        if (segment.local_cells.size() != 1 ||
            segment.local_cells[0] != segment.n_past % runner.layout().local_cells)
          return Error("actual local ring write index differs");
        std::cout << "GEMMA4_RING_LAYOUT slot=" << segment.slot
                  << " local_capacity=" << runner.layout().local_cells
                  << " n_past=" << segment.n_past << " local_write_cell=" << segment.local_cells[0]
                  << " physically_wrapped=" << (segment.n_past >= runner.layout().local_cells)
                  << " global_read=" << segment.global_n_kv << " local_read=" << segment.local_n_kv
                  << '\n';
      }
      if (cycle) {
        if (auto r = prompt(true); !r) return r;
        if (auto r = warm(true); !r) return r;
        for (std::uint32_t i = 0; i < 8; ++i) {
          std::array<std::int32_t, 2> next;
          for (std::uint32_t s = 0; s < 2; ++s) next[s] = i < 6 ? selected[s] : *Best(heads[s]);
          if (auto r = joined(next, i < 5); !r) return r;
        }
        const auto kept = runner.state();
        const auto plans = runner.plan_count();
        const auto graphs = runner.graph_count();
        if (auto r = clear(); !r) return r;
        if (runner.kept_state() != kept || runner.plan_count() != plans ||
            runner.graph_count() != graphs || plans == 0 || graphs == 0)
          return Error("warm Clear changed retained state/plans/graphs");
        std::cout << "GEMMA4_PREPARATION_WARM plans=" << plans << " graphs=" << graphs
                  << " retained_state=1\n";
      }
      const auto paid_graphs = runner.graph_stats();
      const auto begin = std::chrono::steady_clock::now();
      if (auto r = prompt(cycle); !r) return r;
      if (own) {
        std::vector<float> predecode;
        for (const auto& head : heads) predecode.insert(predecode.end(), head.begin(), head.end());
        if (auto r = Save<float>(out / "predecode.f32", predecode); !r) return r;
        auto state = snapshot();
        if (!state) return Error(state.error());
        const auto record =
            "{\"slot0\":\"" + (*state)[0] + "\",\"slot1\":\"" + (*state)[1] + "\"}\n";
        if (auto r = Save<char>(out / "predecode-state.json", record); !r) return r;
      }
      const auto prefill = en::support::Seconds(std::chrono::steady_clock::now() - begin);
      const auto prefill_graphs = runner.graph_stats();
      if (auto r = warm(cycle); !r) return r;
      std::ofstream rows;
      const auto write_rows = [&]() {
        for (const auto& head : heads)
          rows.write(reinterpret_cast<const char*>(head.data()), kVocab * 4);
      };
      if (own) {
        rows.open(out / "heads.f32", std::ios::binary | std::ios::noreplace);
        if (!rows) return Error("teacher rows must be new");
        write_rows();
      }
      std::vector<std::int32_t> choices(kSteps * 2);
      const auto decode_graphs_begin = runner.graph_stats();
      const auto decode_begin = std::chrono::steady_clock::now();
      for (std::uint32_t i = 0; i < kSteps; ++i) {
        std::array<std::int32_t, 2> tokens;
        for (std::uint32_t s = 0; s < 2; ++s) {
          choices[i * 2 + s] = cycle ? selected[s] : *Best(heads[s]);
          tokens[s] = cycle ? choices[i * 2 + s] : ids[s][prefix[s] + 3 + i];
        }
        if (auto r = joined(tokens, cycle && i + 1 != kSteps); !r) return r;
        if (own) write_rows();
      }
      const auto decode = en::support::Seconds(std::chrono::steady_clock::now() - decode_begin);
      const auto decode_graphs_end = runner.graph_stats();
      if (own) {
        rows.flush();
        if (!rows) return Error("complete teacher rows failed");
        const auto full_heads = heads;
        const auto full_state = snapshot();
        if (!full_state) return Error(full_state.error());
        const auto kept = runner.state();
        if (auto r = clear(); !r) return r;
        if (runner.kept_state() != kept) return Error("Clear discarded resident backing");
        if (auto r = prompt(true); !r) return r;
        if (auto r = warm(true); !r) return r;
        const auto before = snapshot();
        if (!before) return Error(before.error());
        std::array<std::uint64_t, 2> bytes;
        for (std::uint32_t s = 0; s < 2; ++s)
          bytes[s] = (*runner.request_slot(s))->used_state_bytes();
        const auto untouched = selected;
        const std::array<en::Gemma4Runner::Work, 2> alias{
            {{0, past[0], std::span(ids[0]).first(1), nullptr, &selected[0]},
             {1, past[1], std::span(ids[1]).first(1), nullptr, &selected[0]}}};
        auto mixed = alias;
        mixed[1].token = nullptr;
        mixed[1].logits = &heads[1];
        auto wrong = alias;
        wrong[1].token = &selected[1];
        ++wrong[1].n_past;
        if (runner.Wave(alias) || runner.Wave(mixed) || runner.Wave(wrong) || selected != untouched)
          return Error("C2 malformed publication accepted or output changed");
        for (std::uint32_t s = 0; s < 2; ++s)
          if ((*runner.request_slot(s))->completed_positions() != past[s] ||
              (*runner.request_slot(s))->used_state_bytes() != bytes[s])
            return Error("C2 refusal changed prefix metadata");
        const auto after = snapshot();
        if (!after || *after != *before) return Error("C2 refusal changed initialized state");
        for (std::uint32_t i = 0; i < kSteps; ++i) {
          std::array<std::int32_t, 2> tokens;
          for (std::uint32_t s = 0; s < 2; ++s) {
            if (selected[s] != choices[i * 2 + s]) return Error("C2 GPU/full-head choices differ");
            tokens[s] = ids[s][prefix[s] + 3 + i];
          }
          if (auto r = joined(tokens, i + 1 != kSteps); !r) return r;
        }
        for (std::uint32_t s = 0; s < 2; ++s)
          if (heads[s].size() != full_heads[s].size() ||
              std::memcmp(heads[s].data(), full_heads[s].data(), kVocab * 4) != 0)
            return Error("C2 GPU/full-head final differs");
        const auto device_state = snapshot();
        if (!device_state || *device_state != *full_state) return Error("C2 GPU state differs");
        for (std::uint32_t s = 0; s < 2; ++s) {
          if (auto r = runner.Spill(s); !r) return r;
          if (auto r = runner.Restore(s); !r) return r;
        }
        const auto restored = snapshot();
        if (!restored || *restored != *full_state) return Error("C2 spill/restore state differs");
        std::array<std::optional<en::CheckpointFile>, 2> checkpoints;
        std::array<std::vector<en::LiveState::Range>, 2> footprints;
        const auto boundary = past;
        for (std::uint32_t s = 0; s < 2; ++s) {
          footprints[s] = (*runner.request_slot(s))->state().used_ranges();
          auto checkpoint = en::CheckpointFile::Capture(
              node, out, footprints[s],
              [&](void* host, std::span<const en::LiveState::Range> ranges) {
                return runner.CopyState(s, host, ranges, true);
              });
          if (!checkpoint) return Error(checkpoint.error().detail);
          checkpoints[s] = std::move(*checkpoint);
        }
        std::array<std::vector<float>, 2> continued;
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
          for (std::uint32_t s = 0; s < 2; ++s) {
            if (auto x = independent(s, std::span(ids[s]).first(128), true, false); !x) return x;
            if (repeat == 0)
              continued[s] = heads[s];
            else if (heads[s].size() != continued[s].size() ||
                     std::memcmp(heads[s].data(), continued[s].data(), kVocab * 4))
              return Error("checkpoint continuation differs");
          }
          for (std::uint32_t s = 0; s < 2; ++s) {
            auto restored_checkpoint = checkpoints[s]->Restore(
                node,
                [&] {
                  return runner.PrepareRestore(s, boundary[s], footprints[s],
                                               runner.CheckpointLayoutId());
                },
                [&](void* host, std::span<const en::LiveState::Range> ranges) {
                  return runner.CopyState(s, host, ranges, false);
                });
            if (!restored_checkpoint) return Error(restored_checkpoint.error().detail);
            if (auto x = runner.CompleteRestore(s, boundary[s]); !x) return x;
            past[s] = boundary[s];
          }
          const auto checkpoint_state = snapshot();
          if (!checkpoint_state || *checkpoint_state != *full_state)
            return Error("ring checkpoint restored state differs");
        }
        heads = full_heads;
        const auto record = "{\"slot0\":\"" + (*full_state)[0] + "\",\"slot1\":\"" +
                            (*full_state)[1] +
                            "\",\"gpu_equal\":true,\"restore_equal\":true,"
                            "\"refusals_unchanged\":true}\n";
        if (auto r = Save<char>(out / "state.json", record); !r) return r;
        if (runner.greedy_tokens() != 70) return Error("C2 own GPU publication count differs");
      }
      if (cycle && runner.greedy_tokens() != 88)
        return Error("C2 cycle GPU publication count differs");
      if (auto r = Save<std::int32_t>(out / "chosen.i32", choices); !r) return r;
      for (std::uint32_t slot = 0; slot < 2; ++slot) {
        if (past[slot] != prefix[slot] + 3 + kSteps) return Error("completed history differs");
        std::vector<std::int32_t> history(
            ids[slot].begin(), ids[slot].begin() + static_cast<std::ptrdiff_t>(prefix[slot] + 3));
        for (std::uint32_t step = 0; step < kSteps; ++step)
          history.push_back(cycle ? choices[step * 2 + slot] : ids[slot][prefix[slot] + 3 + step]);
        if (auto r = Save<std::int32_t>(out / ("history" + std::to_string(slot) + ".i32"), history);
            !r)
          return r;
      }
      std::vector<float> final;
      final.reserve(kVocab * 2);
      for (const auto& head : heads) final.insert(final.end(), head.begin(), head.end());
      if (auto r = Save<float>(out / "final.f32", final); !r) return r;
      const auto final_state = snapshot();
      if (!final_state) return Error(final_state.error());
      for (std::size_t slot = 0; slot < final_state->size(); ++slot)
        std::cout << "final_state" << slot << "=" << (*final_state)[slot] << '\n';
      const auto& stats = runner.graph_stats();
      const auto& bound = runner.last_built_policy();
      if (!stats.captured || !stats.replayed || runner.coverage().violations ||
          owner_steps != expected_owners || bounded_steps != expected_bounded)
        return Error("C2 did not select/replay checked implementation families");
      if (vecq_steps || (dense_shared_q8 ? q8_steps == 0 || prepared_steps <= q8_steps
                                         : q8_steps != 0 || prepared_steps != 0))
        return Error("actual dense preparation policy differs");
      std::cout << "GEMMA4_CONTEXT mode=" << mode << " slots=2 context_per_slot=4096"
                << " profile=" << profile << " policy=" << policy << " actual_chunk=" << chunk
                << " independent_prefill=1 prompt_rows0=" << prefix[0]
                << " prompt_rows1=" << prefix[1] << " untimed_rows_per_slot=3"
                << " decode_steps=32 departure_steps=0"
                << " paid_generated_tokens=64 past0=" << past[0] << " past1=" << past[1]
                << " prefill_seconds=" << prefill << " decode_seconds=" << decode
                << " eager=" << stats.eager << " captured=" << stats.captured
                << " replayed=" << stats.replayed << " paid_eager="
                << prefill_graphs.eager - paid_graphs.eager + decode_graphs_end.eager -
                       decode_graphs_begin.eager
                << " paid_captured="
                << prefill_graphs.captured - paid_graphs.captured + decode_graphs_end.captured -
                       decode_graphs_begin.captured
                << " paid_replayed="
                << prefill_graphs.replayed - paid_graphs.replayed + decode_graphs_end.replayed -
                       decode_graphs_begin.replayed
                << " selected_owner=" << owner_steps << " selected_bounded_owner=" << bounded_steps
                << " dense_shared_q8=" << dense_shared_q8
                << " prepared_mmvq_products=" << prepared_steps << " q8_preparations=" << q8_steps
                << " selected_vecq=" << vecq_steps << " selected_norm_mul=" << bound.norm_fused
                << " selected_norm_rope=" << bound.norm_rope
                << " selected_norm_add=" << bound.norm_add
                << " gpu_tokens=" << runner.greedy_tokens() << '\n';
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
  if (ran && retired) std::cout << "GEMMA4_CONTEXT_RETIRED\n";
  return ran && retired ? 0 : 1;
}
