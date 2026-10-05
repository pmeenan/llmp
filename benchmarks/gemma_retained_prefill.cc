// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Cold versus retained prefill plans; identical work, graphs disabled.
// ARTIFACT IDS_I32 NEW_OUTPUT_DIR 26|31. Fixed all1024/both256 policy.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
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
  std::cout << std::setprecision(9);
  const std::string_view variant = argc >= 5 ? argv[4] : "26";
  const std::string_view requested_policy = variant == "26" ? "all" : "both";
  if (variant != "26" && variant != "31") return 2;
  constexpr std::string_view normmul = "normmul-off";
  const std::uint32_t max_rows = variant == "26" ? 1024 : 256;
  constexpr std::uint32_t kPrefill = 8192, kWarm = 3, kSteps = 32;
  constexpr std::uint32_t kInput = kPrefill + kWarm + kSteps, kVocab = 262144;
  std::error_code error;
  if (std::filesystem::file_size(argv[2], error) != kInput * 4 || error) return 2;
  std::array<std::int32_t, kInput> ids{};
  std::ifstream file(argv[2], std::ios::binary);
  file.read(reinterpret_cast<char*>(ids.data()), sizeof(ids));
  if (!file || ids.front() != 2 ||
      !std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < 262144; }))
    return 2;
  const std::filesystem::path out(argv[3]);
  if (!std::filesystem::create_directory(out, error) || error) return 2;
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
          .variant = variant == "31" ? en::Gemma4Variant::k31B : en::Gemma4Variant::k26BA4B,
          .context = 16384,
          .max_rows = max_rows,
          .graphs = false,
          .fuse_norms = normmul == "normmul-on",
          .fuse_norm_rope = requested_policy == "both" || requested_policy == "all",
          .fuse_norm_add = requested_policy == "both" || requested_policy == "all",
          .fuse_gemma_route = requested_policy == "all",
          .fuse_gemma_reduce = requested_policy == "all"},
      0, 0);
  auto& runner = *lifetime->runner;
  auto& entered = lifetime->entered;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    constexpr std::uint64_t kOutputBytes = kVocab * 4;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    // Every call can introduce at most one plan and one captured graph.
    // PlannedHostBytes includes kPlanNodeHostBytes per launched node, so
    // floor/kPlanNodeHostBytes bounds its graph's catalog charge. This
    // derives a conservative retention budget from the exact call count,
    // rather than consuming state funding with accumulated plan charges.
    const std::uint64_t calls = 1 + kPrefill / max_rows + kWarm + kSteps;
    constexpr auto kGraphRatio = en::kGraphNodeHostBytes / en::kPlanNodeHostBytes;
    const auto floor = runner.plan_floor_bytes();
    if (floor > std::numeric_limits<std::uint64_t>::max() / calls / (1 + kGraphRatio))
      return Error("plan/graph budget overflow");
    const auto retention = calls * floor * (1 + kGraphRatio);
    const auto base = fixed + runner.weights().size() * en::kPagedExtent +
                      2 * node.StateCapacity() + kOutputBytes;
    const auto planning_scratch = en::ScratchArenaBytes();
    if (planning_scratch > std::numeric_limits<std::uint64_t>::max() - base ||
        retention > std::numeric_limits<std::uint64_t>::max() - base - planning_scratch)
      return Error("execution budget overflow");
    const auto budget = base + planning_scratch + retention;
    std::cout << "PREFILL_BUDGET fixed=" << fixed
              << " weights=" << runner.weights().size() * en::kPagedExtent
              << " state_capacity=" << node.StateCapacity() << " publication=" << kOutputBytes
              << " pinned_head_envelope=" << std::uint64_t{max_rows} * kVocab * 4
              << " max_rows=" << max_rows << " call_bound=" << calls << " plan_floor=" << floor
              << " planning_scratch=" << planning_scratch << " plan_graph_capacity=" << retention
              << " total=" << budget << '\n';
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes() + planning_scratch);
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(0, runner.closure(), "Gemma4 paid 8K screen", [&]() -> en::Status {
      if (!node.ChargeHost(kOutputBytes, false)) return Error("publication capacity refused");
      struct Charge {
        en::PagedNode& node;
        std::uint64_t bytes;
        ~Charge() { node.UnchargeHost(bytes); }
      } charged{node, kOutputBytes};
      std::vector<float> logits;
      logits.reserve(kVocab);
      if (logits.capacity() != kVocab) return Error("unexpected publication vector capacity");
      // Page all weights before the paid fresh request, matching the already
      // loaded comparator. These six rows are discarded on both engines.
      if (auto r = runner.Chunk(0, std::span(ids).first(6), logits); !r) return r;
      runner.DropPlans();
      runner.EnablePhaseAccounting();
      for (const std::string_view arm : {"cold", "retained"}) {
        const auto arm_out = std::filesystem::path(argv[3]) / arm;
        if (!std::filesystem::create_directory(arm_out, error) || error)
          return Error("creating arm output failed");
        if (auto r = runner.Clear(); !r) return r;
        const auto entries_before = runner.plan_count();
        (void)runner.TakePhaseAccounting();
        (void)node.TakeTimes(0);
        const auto started = std::chrono::steady_clock::now();
        for (std::uint32_t first = 0; first < kPrefill; first += max_rows) {
          if (auto r = runner.Chunk(first, std::span(ids).subspan(first, max_rows), logits); !r)
            return r;
        }
        const auto prefill = en::support::Seconds(std::chrono::steady_clock::now() - started);
        const auto phases = runner.TakePhaseAccounting();
        const auto jobs = node.TakeTimes(0);
        const auto entries_after = runner.plan_count();
        const auto chunks = kPrefill / max_rows;
        if (phases.planned_calls != chunks ||
            (arm == "cold" &&
             (entries_before != 0 || phases.misses != chunks || phases.hits != 0)) ||
            (arm == "retained" && (phases.hits != chunks || phases.misses != 0)))
          return Error("unexpected cold/retained prefill plan cache behavior");
        const auto last_built = runner.last_built_policy();
        std::cout << "PREFILL_PHASE arm=" << arm << " planned_calls=" << phases.planned_calls
                  << " hits=" << phases.hits << " misses=" << phases.misses
                  << " entries_before=" << entries_before << " entries_after=" << entries_after;
        constexpr std::array names{
            "checks", "inputs", "state", "planned", "staging", "execution", "publication_cleanup"};
        for (std::size_t i = 0; i < names.size(); ++i)
          std::cout << ' ' << names[i] << "_seconds=" << phases.seconds[i];
        std::cout << " job_steps=" << jobs.steps << " job_wall=" << jobs.wall
                  << " dispatch=" << jobs.dispatch << " submission=" << jobs.job
                  << " after=" << jobs.after << " stream_elapsed=" << jobs.device << '\n';
        if (logits.size() != kVocab) return Error("missing final prefill head");
        const auto save = [&](const char* name) -> en::Status {
          std::ofstream output(arm_out / name, std::ios::binary);
          output.write(reinterpret_cast<const char*>(logits.data()), kOutputBytes);
          output.flush();
          return output ? en::Status{} : en::Status(Error("writing completed head failed"));
        };
        if (auto r = save("prefill.f32"); !r) return r;
        for (std::uint32_t i = 0; i < kWarm; ++i) {
          const auto past = kPrefill + i;
          if (auto r = runner.Chunk(past, std::span(ids).subspan(past, 1), logits); !r) return r;
          std::cout << "PREFILL_TIMED_PREFIX appended=" << ids[past] << " past=" << past + 1
                    << '\n';
        }
        std::array<std::int32_t, kSteps> chosen{};
        const auto decode_started = std::chrono::steady_clock::now();
        for (std::uint32_t i = 0; i < kSteps; ++i) {
          chosen[i] = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                                logits.begin());
          const auto past = kPrefill + kWarm + i;
          if (auto r = runner.Chunk(past, std::span(ids).subspan(past, 1), logits); !r) return r;
        }
        const auto decode = en::support::Seconds(std::chrono::steady_clock::now() - decode_started);
        if (auto r = save("final.f32"); !r) return r;
        // Retain the exact completed initialized footprint outside both timers.
        // The transfer lives in separately charged pinned storage; no full host
        // payload vector is allocated. A failed copy remains node-owned until
        // proven teardown, and the whole owner bundle survives an unproven fence.
        auto ranges = runner.CheckpointRanges(kInput);
        if (!ranges) return Error(ranges.error());
        std::uint64_t state_bytes = 0;
        for (const auto& range : *ranges) state_bytes += range.bytes;
        std::vector<jitllm::catalog::ExtentId> staging;
        auto state = node.Pinned(state_bytes, 0, staging);
        if (!state) return Error(state.error());
        if (auto copied = runner.CopyState(0, *state, *ranges, true); !copied) return copied;
        std::ofstream checkpoint(arm_out / "initialized-state.bin", std::ios::binary);
        checkpoint.write(static_cast<const char*>(*state),
                         static_cast<std::streamsize>(state_bytes));
        checkpoint.flush();
        const bool written = bool(checkpoint);
        if (auto freed = node.FreePinned(*state); !freed) return freed;
        if (!written) return Error("writing completed state failed");
        std::ofstream layout(arm_out / "state-layout.txt");
        layout << runner.CheckpointLayoutId() << '\n';
        layout.flush();
        if (!layout) return Error("writing state layout failed");
        std::cout << "PREFILL_STATE positions=" << kInput << " bytes=" << state_bytes
                  << " ranges=" << ranges->size() << " layout=" << runner.CheckpointLayoutId()
                  << '\n';
        auto slot = runner.request_slot(0);
        if (!slot || (*slot)->completed_positions() != kInput)
          return Error("completed positions differ from paid work");
        const auto& decode_policy = runner.last_built_policy();
        std::cout << "PREFILL_NATIVE arm=" << arm << " graphs=false prefill_seconds=" << prefill
                  << " prefill_rows=" << kPrefill << " prefill_chunks=" << kPrefill / max_rows
                  << " intermediate_heads=" << kPrefill / max_rows - 1
                  << " decode_seconds=" << decode << " decode_chunks=" << kSteps
                  << " timed_start=" << kPrefill + kWarm << " completed=" << kInput
                  << " context=16384 chunk=" << max_rows << " masks=device"
                  << " policy=" << requested_policy << " normmul=" << normmul
                  << " last_built_rows=" << last_built.rows
                  << " last_built_norm_fused=" << last_built.norm_fused
                  << " last_built_norm_rope=" << last_built.norm_rope
                  << " last_built_norm_add=" << last_built.norm_add
                  << " last_built_gemma_route=" << last_built.gemma_route
                  << " last_built_gemma_reduce=" << last_built.gemma_reduce
                  << " gemma_route=" << decode_policy.gemma_route
                  << " gemma_reduce=" << decode_policy.gemma_reduce
                  << " shared_vecq=" << decode_policy.shared_vecq
                  << " row_products=" << decode_policy.row_products
                  << " norm_fused=" << decode_policy.norm_fused
                  << " rope_store=" << decode_policy.rope_store
                  << " norm_rope=" << decode_policy.norm_rope
                  << " norm_add=" << decode_policy.norm_add
                  << " captures=" << runner.graph_stats().captured
                  << " replays=" << runner.graph_stats().replayed << '\n';
        for (std::uint32_t i = 0; i < kSteps; ++i)
          std::cout << "PREFILL_TOKEN step=" << i << " argmax=" << chosen[i]
                    << " forced=" << ids[kPrefill + kWarm + i] << '\n';
      }
      return {};
    });
  };
  const auto ran = execute();
  const auto retired = node.TearDown(entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = lifetime.release();
  }
  return ran && retired ? 0 : 1;
}
