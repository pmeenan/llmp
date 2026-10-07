// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Internal grouped H8 first screen; no timing or public admission.
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
constexpr std::uint64_t kCopy = 8ULL << 20U;
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
  if (!jitllm::platform::InstallCrashPolicy("gemma3-wide-probe") || argc != 5) return 2;
  std::array<std::vector<std::int32_t>, kOwners> ids;
  std::array<std::uint32_t, kOwners> prefix{};
  for (std::uint32_t s = 0; s < kOwners; ++s) {
    auto raw = Read(argv[2 + wide::SourceIndex(s)], 4096 * 4);
    if (!raw || raw->size() % 4 || (raw->size() != 295 * 4 && raw->size() != 807 * 4)) return 2;
    ids[s].resize(raw->size() / 4);
    std::memcpy(ids[s].data(), raw->data(), raw->size());
    prefix[s] = static_cast<std::uint32_t>(ids[s].size()) - 39;
    if (prefix[s] != (wide::SourceIndex(s) ? 768U : 256U) || ids[s][0] != 2 ||
        !std::ranges::all_of(ids[s], [](auto id) { return id >= 0 && id < std::int32_t(kVocab); }))
      return 2;
  }
  const auto events = wide::Schedule(prefix);
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
                                                                      .fuse_norms = true,
                                                                      .fuse_quant_glu = true,
                                                                      .fuse_norm_rope = true,
                                                                      .fuse_norm_add = true},
                                                    0, 0);
  auto& runner = *life->runner;
  std::array<std::vector<float>, kOwners> heads;
  for (auto& row : heads) row.reserve(kVocab);
  std::array<std::uint32_t, kOwners> past{}, slots{};
  for (std::uint32_t s = 0; s < kOwners; ++s) slots[s] = s;
  std::array<std::int32_t, kOwners> selected{};
  const auto copy_head = [&](unsigned pass, const wide::Event& event, std::uint32_t slot) {
    const auto final = prefix[slot] + (slot < 4 ? 39U : slot < 10 ? 37U : 5U);
    return pass == 0 || !event.head ||
           (event.phase == events.back().phase && event.before[slot] + event.rows == final);
  };
  void* pinned = nullptr;
  const auto select = [&](std::uint32_t count) {
    return runner.SelectSlots(std::span(slots).first(count));
  };
  const auto snapshot =
      [&](std::uint32_t first,
          std::uint32_t count) -> std::expected<std::vector<std::string>, std::string> {
    if (auto r = select(kOwners); !r) return Error(r.error());
    std::vector<std::string> result;
    for (std::uint32_t s = first; s < first + count; ++s) {
      if ((*runner.request_slot(s))->completed_positions() != past[s])
        return Error("state cursor differs");
      auto ranges = runner.CheckpointRanges(past[s]);
      if (!ranges) return Error(ranges.error());
      jitllm::base::Sha256 hash;
      for (const auto& range : *ranges) {
        for (std::uint64_t at = 0; at < range.bytes;) {
          auto part = range;
          part.offset += at;
          part.bytes = std::min(kCopy, range.bytes - at);
          en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kUnproven;
          auto copied = runner.CopyState(s, pinned, std::span(&part, 1), &retirement);
          if (retirement == en::LiveState::CopyRetirement::kUnproven) node.KeepPinned(pinned);
          if (!copied) return Error(copied.error());
          hash.Update(std::span(static_cast<const std::byte*>(pinned), std::size_t(part.bytes)));
          at += part.bytes;
        }
      }
      result.push_back(jitllm::base::ToHex(hash.Finish()));
    }
    return result;
  };
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    life->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    std::vector<jitllm::catalog::ExtentId> extents;
    auto allocation = node.Pinned(kCopy, 0, extents);
    if (!allocation) return Error(allocation.error());
    pinned = *allocation;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + (32ULL << 20U));
    // Enumerate every retained shape from both exact passes using the same
    // checked chunk builder and cache equality as the runner. No reclaimer is
    // installed in this standalone harness, so its complete finite key set is
    // funded before execution, including every slot-specific prefill/scalar key.
    std::vector<jitllm::kernels::ggml::Gemma3ChunkShape> cache_keys;
    for (unsigned pass = 0; pass < 2; ++pass)
      for (const auto& event : events) {
        if (event.kind != wide::Kind::kPrefill && event.kind != wide::Kind::kScalar &&
            event.kind != wide::Kind::kWave)
          continue;
        std::array<jitllm::model::Gemma3Segment, kOwners> segments;
        for (std::uint32_t i = 0; i < event.count; ++i) {
          const auto slot = event.first + i;
          segments[i] = {slot, event.before[slot],
                         std::span(ids[slot]).subspan(event.before[slot], event.rows)};
        }
        auto input =
            jitllm::model::Gemma3Chunk(runner.profile(), runner.layout(),
                                       std::span(segments).first(event.count), true, 256, 256);
        if (!input) return Error(input.error());
        jitllm::kernels::ggml::Gemma3ChunkShape key;
        for (const auto& segment : input->segments)
          key.segments.push_back({segment.slot, segment.rows, segment.n_past, segment.global_n_kv,
                                  segment.local_n_kv});
        key.outputs = event.head ? event.count : 0;
        key.output_mode = event.head ? jitllm::kernels::ggml::Gemma3OutputMode::kHead
                                     : jitllm::kernels::ggml::Gemma3OutputMode::kStateOnly;
        key.greedy = event.head && !copy_head(pass, event, event.first);
        if (std::ranges::find(cache_keys, key) == cache_keys.end())
          cache_keys.push_back(std::move(key));
      }
    // PlannedHostBytes contains nodes*512 plus the arena. Thus one plan and its
    // graph charge <= floor*(1+16384/512). StateCapacity sums all twelve spans.
    constexpr std::uint64_t kPlanAndGraph = 1 + en::kGraphNodeHostBytes / en::kPlanNodeHostBytes;
    if (cache_keys.empty() ||
        runner.plan_floor_bytes() > (64ULL << 30U) / (cache_keys.size() * kPlanAndGraph))
      return Error("complete diagnostic cache allowance exceeds fixed 64GiB ceiling");
    const auto cache_budget = cache_keys.size() * kPlanAndGraph * runner.plan_floor_bytes();
    const auto budget =
        fixed + runner.weights().size() * en::kPagedExtent + node.StateCapacity() + cache_budget;
    std::cout << "GEMMA3_WIDE_BUDGET fixed_bytes=" << fixed
              << " state_capacity_all_owners=" << node.StateCapacity()
              << " cache_keys=" << cache_keys.size() << " cache_budget=" << cache_budget
              << " total_bytes=" << budget << '\n';
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(
        0, runner.closure(), "Gemma3 grouped wide first screen", [&]() -> en::Status {
          std::ofstream rows(out / "heads.f32", std::ios::binary | std::ios::noreplace);
          std::ofstream map(out / "rows.tsv", std::ios::noreplace);
          if (!rows || !map) return Error("exclusive teacher outputs refused");
          std::vector<std::int32_t> choices;
          std::vector<std::string> final_state;
          std::array<std::vector<float>, kOwners> final_heads;
          std::array<std::uint32_t, kOwners> final_past{};
          std::array<std::uint64_t, 13> waves{}, selected_plans{};
          std::uint64_t peer_checks = 0, prefill_groups = 0, prefill_rows = 0;
          for (unsigned pass = 0; pass < 2; ++pass) {
            if (auto r = select(kOwners); !r) return r;
            if (pass) {
              for (std::uint32_t s = 0; s < kOwners; ++s)
                if (auto r = runner.Clear(s); !r) return r;
              past = {};
            }
            std::size_t row_index = 0;
            std::vector<std::string> paused, catchup_peers, refill_peers;
            std::uint64_t before_selection = 0;
            bool refill = false;
            for (const auto& event : events) {
              if (past != event.before) return Error("schedule target/cursor mapping differs");
              const auto first = event.first, count = event.count;
              if (event.kind == wide::Kind::kPauseBegin) {
                auto state = snapshot(count, kOwners - count);
                if (!state) return Error(state.error());
                paused = std::move(*state);
                before_selection = runner.plan_selections().owner_attention;
                if (auto r = select(count); !r) return r;
                if (count != kOwners) {
                  const auto untouched = selected;
                  const en::Gemma3Runner::Work bad{count, past[count],
                                                   std::span(ids[count]).subspan(past[count], 1),
                                                   nullptr, &selected[count]};
                  if (runner.Wave(std::span(&bad, 1)) || selected != untouched)
                    return Error("paused owner publication accepted");
                }
                continue;
              }
              if (event.kind == wide::Kind::kPauseEnd) {
                const auto delta = runner.plan_selections().owner_attention - before_selection;
                if (delta < 34 * ((count + 3) / 4) || delta % (34 * ((count + 3) / 4)))
                  return Error("actual logical cohort did not select all grouped layers");
                selected_plans[count] += delta;
                auto state = snapshot(count, kOwners - count);
                if (!state || *state != paused) return Error("partial wave changed paused owners");
                ++peer_checks;
                continue;
              }
              if (event.kind == wide::Kind::kCatchupBegin) {
                auto state = snapshot(0, 4);
                if (!state) return Error(state.error());
                catchup_peers = std::move(*state);
                continue;
              }
              if (event.kind == wide::Kind::kCatchupEnd) {
                auto state = snapshot(0, 4);
                if (!state || *state != catchup_peers) return Error("catchup changed active peers");
                ++peer_checks;
                continue;
              }
              if (event.kind == wide::Kind::kRefillBegin) {
                auto state = snapshot(0, 10);
                if (!state) return Error(state.error());
                refill_peers = *state;
                for (std::uint32_t s = 10; s < kOwners; ++s) {
                  if (auto r = runner.Clear(s); !r) return r;
                  past[s] = 0;
                }
                auto after = snapshot(0, 10);
                if (!after || *after != *state) return Error("Clear changed ten refill peers");
                ++peer_checks;
                refill = true;
                continue;
              }
              if (event.kind == wide::Kind::kRefillEnd) {
                auto state = snapshot(4, 6);
                if (!state || !std::equal(state->begin(), state->end(), refill_peers.begin() + 4))
                  return Error("refill or C4 interleave changed six paused peers");
                ++peer_checks;
                refill = false;
                continue;
              }
              std::vector<std::string> refill_before;
              if (refill && event.kind == wide::Kind::kPrefill) {
                auto state = snapshot(0, 10);
                if (!state) return Error(state.error());
                refill_before = std::move(*state);
              }
              if (auto r = select(event.kind == wide::Kind::kWave ? count : kOwners); !r) return r;
              std::array<en::Gemma3Runner::Work, kOwners> work;
              for (std::uint32_t i = 0; i < count; ++i) {
                const auto s = first + i;
                if (past[s] + event.rows > ids[s].size())
                  return Error("supplied input bound exceeded");
                // Copy each final slot head; earlier device publications are tokens.
                const bool copy = copy_head(pass, event, s);
                work[i] = {s, past[s], std::span(ids[s]).subspan(past[s], event.rows),
                           copy ? &heads[s] : nullptr,
                           copy || !event.head ? nullptr : &selected[s]};
              }
              const auto actual = std::span(work).first(count);
              auto r = event.kind == wide::Kind::kPrefill ? runner.WavePrefill(actual, event.head)
                                                          : runner.Wave(actual);
              if (!r) return r;
              if (event.kind == wide::Kind::kWave) ++waves[count];
              if (event.kind == wide::Kind::kPrefill) {
                ++prefill_groups;
                prefill_rows += count * event.rows;
              }
              for (std::uint32_t i = 0; i < count; ++i) {
                const auto s = first + i;
                past[s] += event.rows;
                if (!event.head) {
                  if (!heads[s].empty()) return Error("state-only publication leak");
                  continue;
                }
                auto best = work[i].logits ? Best(heads[s])
                                           : std::expected<std::int32_t, std::string>(selected[s]);
                if (!best || *best < 0 || *best >= std::int32_t(kVocab))
                  return Error("invalid frontier publication");
                if (!pass) {
                  rows.write(reinterpret_cast<const char*>(heads[s].data()), kVocab * 4);
                  const auto target = past[s] < ids[s].size() ? ids[s][past[s]] : -1;
                  map << event.phase << '\t' << static_cast<int>(event.kind) << '\t' << s << '\t'
                      << past[s] << '\t' << target << '\n';
                  choices.push_back(*best);
                } else if (row_index >= choices.size() || choices[row_index] != *best)
                  return Error("GPU/full-head publication choices differ");
                ++row_index;
              }
              if (!refill_before.empty()) {
                auto state = snapshot(0, 10);
                if (!state || *state != refill_before)
                  return Error("paired refill changed ten peers");
                ++peer_checks;
              }
            }
            if (row_index != choices.size()) return Error("full/device row coverage differs");
            auto state = snapshot(0, kOwners);
            if (!state) return Error(state.error());
            if (!pass) {
              final_state = std::move(*state);
              final_heads = heads;
              final_past = past;
            } else {
              if (*state != final_state || past != final_past)
                return Error("GPU/full-head final states differ");
              for (std::uint32_t s = 0; s < kOwners; ++s)
                if (heads[s].size() != kVocab ||
                    std::memcmp(heads[s].data(), final_heads[s].data(), kVocab * 4))
                  return Error("GPU/full-head final rows differ");
            }
          }
          if (choices.size() != 472 || runner.greedy_tokens() != 460 || prefill_groups != 60 ||
              prefill_rows != 15360 || peer_checks != 36)
            return Error("wide funded publication or peer-check counters differ");
          rows.flush();
          map.flush();
          if (!rows || !map) return Error("complete output write failed");
          for (std::uint32_t s = 0; s < kOwners; ++s) {
            if (auto r = runner.Spill(s); !r) return r;
            if (auto r = runner.Restore(s); !r) return r;
          }
          auto restored = snapshot(0, kOwners);
          if (!restored || *restored != final_state)
            return Error("wide spill/restore state differs");
          if (auto r = Save<std::int32_t>(out / "chosen.i32", choices); !r) return r;
          std::ofstream final(out / "final.f32", std::ios::binary | std::ios::noreplace);
          for (const auto& head : final_heads)
            final.write(reinterpret_cast<const char*>(head.data()), kVocab * 4);
          final.flush();
          if (!final) return Error("exclusive complete final heads failed");
          std::ofstream state(out / "state.tsv", std::ios::noreplace);
          for (std::uint32_t s = 0; s < kOwners; ++s)
            state << s << '\t' << final_past[s] << '\t' << final_state[s] << '\n';
          state.flush();
          if (!state) return Error("complete state output failed");
          const auto& bound = runner.plan_selections();
          const auto& stats = runner.graph_stats();
          if (bound.plans != cache_keys.size() || runner.plans_bytes() > cache_budget ||
              node.host_counted() != runner.plans_bytes() || node.host_charged() > cache_budget ||
              node.host_overcharges() != 0)
            return Error("actual retained keys/charges exceed the complete finite cache envelope");
          if (!bound.packed_prefill_attention || !bound.owner_attention || !bound.norm_rope ||
              !bound.norm_add || !stats.captured || !stats.replayed ||
              runner.coverage().violations || bound.bounded_owner_attention)
            return Error("wide selected implementation/graph controls missing");
          for (std::uint32_t n = 4; n <= 12; ++n) {
            if (!selected_plans[n] || !waves[n]) return Error("logical cohort coverage missing");
            std::cout << "GEMMA3_WIDE_PHASE logical=" << n << " actual_waves=" << waves[n]
                      << " selected_owner_plans=" << selected_plans[n] << '\n';
          }
          std::cout << "GEMMA3_WIDE_PASS rows=" << choices.size() << " slots=12 input_identities=2"
                    << " context=4096 per_owner_rows=128 wave_rows=256 head_rows=12"
                    << " gpu_equal=1 state_equal=1 restore_equal=1 peer_checks=" << peer_checks
                    << " prefill_groups=" << prefill_groups << " prefill_rows=" << prefill_rows
                    << " captured=" << stats.captured << " replayed=" << stats.replayed
                    << " cache_keys=" << cache_keys.size() << " actual_plans=" << bound.plans
                    << " plan_graph_bytes=" << runner.plans_bytes()
                    << " cache_budget=" << cache_budget << " host_counted=" << node.host_counted()
                    << " host_charged=" << node.host_charged()
                    << " host_overcharges=" << node.host_overcharges()
                    << " gpu_tokens=" << runner.greedy_tokens()
                    << " activation_bytes=" << runner.activations_needed()
                    << " scratch_bytes=" << runner.pool_needed()
                    << " host_bytes=" << runner.host_input_bytes() << '\n';
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
  if (ran && retired) std::cout << "GEMMA3_WIDE_RETIRED\n";
  return ran && retired ? 0 : 1;
}
