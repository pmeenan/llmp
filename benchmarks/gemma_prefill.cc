// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Configurable paid prefill and fixed-prefix decode screen. No quality claim.
// ARTIFACT IDS_I32 NEW_OUTPUT_DIR [26|31] [ordinary|both|all|serving] [MAX_ROWS]
// [normmul-off|normmul-on] [full|state-only] [lookahead-on|lookahead-off]
// [phases-off|phases-on] [state-chunked|state-upfront]
// [capture-ahead-off|capture-ahead-on] [features-off|features-frontier]
// [prepare-off|prepare-on] [CONTEXT PREFILL_ROWS].
// The serving policy mirrors the bounded C1 runtime recipe; legacy defaults
// retain context 16384 and 8192 prefill rows from the fixed input fixture.
// Row-cap experiments do not change production defaults.
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = llmp::engine;
using en::support::Error;
int main(int argc, char** argv) {
  if (argc < 4 || argc > 19 || argc == 16) return 2;
  const std::string_view variant = argc >= 5 ? argv[4] : "26";
  const std::string_view policy = argc >= 6 ? argv[5] : "ordinary";
  if (variant != "26" && variant != "31") return 2;
  if (policy != "ordinary" && policy != "both" && policy != "all" && policy != "serving") return 2;
  const bool serving = policy == "serving";
  const std::string_view normmul = argc >= 8 ? argv[7] : "normmul-off";
  if (normmul != "normmul-off" && normmul != "normmul-on") return 2;
  const std::string_view prefill_output = argc >= 9 ? argv[8] : "full";
  if (prefill_output != "full" && prefill_output != "state-only") return 2;
  const std::string_view lookahead = argc >= 10 ? argv[9] : "lookahead-on";
  if (lookahead != "lookahead-on" && lookahead != "lookahead-off") return 2;
  const std::string_view phases_mode = argc >= 11 ? argv[10] : "phases-off";
  if (phases_mode != "phases-off" && phases_mode != "phases-on") return 2;
  const bool account_phases = phases_mode == "phases-on";
  const std::string_view state_prepare = argc >= 12 ? argv[11] : "state-chunked";
  if (state_prepare != "state-chunked" && state_prepare != "state-upfront") return 2;
  const std::string_view capture_ahead = argc >= 13 ? argv[12] : "capture-ahead-off";
  if (capture_ahead != "capture-ahead-off" && capture_ahead != "capture-ahead-on") return 2;
  const bool ahead = capture_ahead == "capture-ahead-on";
  const std::string_view features = argc >= 14 ? argv[13] : "features-off";
  if (features != "features-off" && features != "features-frontier") return 2;
  const bool retain_features = features == "features-frontier";
  const std::string_view preparation = argc >= 15 ? argv[14] : "prepare-off";
  if (preparation != "prepare-off" && preparation != "prepare-on") return 2;
  const bool prepare_state = preparation == "prepare-on";
  std::uint32_t max_rows = 128;
  if (argc >= 7) {
    const std::string_view number(argv[6]);
    const auto [end, parsed] =
        std::from_chars(number.data(), number.data() + number.size(), max_rows);
    if (parsed != std::errc{} || end != number.data() + number.size()) return 2;
  }
  // Fixed diagnostic shapes only; no serving cap or artifact-format change.
  if (max_rows != 128 && max_rows != 256 && max_rows != 512 && max_rows != 1024 && max_rows != 2048)
    return 2;
  constexpr std::uint32_t kWarm = 3, kSteps = 32, kInputCapacity = 8227, kVocab = 262144;
  std::uint32_t context = 16384, kPrefill = 8192;
  if (argc >= 17) {
    const auto parse = [](const char* arg, std::uint32_t& value) {
      const std::string_view number(arg);
      const auto [end, error] =
          std::from_chars(number.data(), number.data() + number.size(), value);
      return error == std::errc{} && end == number.data() + number.size();
    };
    if (!parse(argv[15], context) || !parse(argv[16], kPrefill)) return 2;
  }
  if (kPrefill == 0 || kPrefill > kInputCapacity - kWarm - kSteps || context < kWarm + kSteps ||
      kPrefill > context - kWarm - kSteps || context > llmp::model::kGemma4Context ||
      max_rows > context)
    return 2;
  if (serving &&
      (context > 8192 || normmul != "normmul-on" || max_rows > (variant == "31" ? 256U : 1024U)))
    return 2;
  const std::string_view stores = argc >= 18 ? argv[17] : "stores-off";
  if (stores != "stores-off" && stores != "stores-on") return 2;
  std::uint64_t budget_override = 0;
  if (argc == 19) {
    const std::string_view number(argv[18]);
    const auto [end, error] =
        std::from_chars(number.data(), number.data() + number.size(), budget_override);
    if (error != std::errc{} || end != number.data() + number.size() || budget_override == 0 ||
        budget_override % en::kPagedExtent != 0)
      return 2;
  }
  const std::uint32_t kInput = kPrefill + kWarm + kSteps;
  const std::uint32_t chunks = (kPrefill + max_rows - 1) / max_rows;
  std::error_code error;
  if (std::filesystem::file_size(argv[2], error) != kInputCapacity * 4 || error) return 2;
  std::array<std::int32_t, kInputCapacity> ids{};
  std::ifstream file(argv[2], std::ios::binary);
  file.read(reinterpret_cast<char*>(ids.data()), sizeof(ids));
  if (!file || ids.front() != 2 ||
      !std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < 262144; }))
    return 2;
  const std::filesystem::path out(argv[3]);
  if (!std::filesystem::create_directory(out, error) || error) return 2;
  struct Lifetime {
    en::PagedNode node{{.zero_state = true, .slot_bytes = en::kSlabSlotBytes}};
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
          .context = context,
          .max_rows = max_rows,
          .max_head_rows = serving ? 1U : 0U,
          .frontier_head = !serving,
          .retain_features = retain_features,
          .fuse_norms = normmul == "normmul-on",
          .fuse_norm_rope = serving || policy == "both" || policy == "all",
          .fuse_norm_add = serving || policy == "both" || policy == "all",
          .fuse_gemma_route = policy == "all" || (serving && variant == "26"),
          .fuse_gemma_reduce = policy == "all" || (serving && variant == "26"),
          .fuse_quant_glu = serving && variant == "31",
          .prefill_lookahead = lookahead == "lookahead-on",
          .prepare_state = prepare_state,
          .group_kv_stores = stores == "stores-on",
          .capture_ahead = ahead,
          .prefill_lookahead_capacity = ahead ? 2U : 1U,
          .owner_attention = serving},
      0, 0);
  auto& runner = *lifetime->runner;
  auto& entered = lifetime->entered;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    constexpr std::uint64_t kOutputBytes = kVocab * 4;
    const std::uint64_t feature_bytes =
        retain_features ? std::uint64_t{runner.profile().width} * sizeof(float) : 0;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    // These exact hints name only later actual calls: even two future plans
    // cannot increase the unique retained key count beyond this call bound.
    // Each key retains at most one plan and one graph. PlannedHostBytes
    // includes kPlanNodeHostBytes per launched node, so
    // floor/kPlanNodeHostBytes bounds its graph's catalog charge. This
    // derives a conservative retention budget from the exact call count,
    // rather than consuming state funding with accumulated plan charges.
    const std::uint64_t calls = 1 + chunks + kWarm + kSteps;
    constexpr auto kGraphRatio = en::kGraphNodeHostBytes / en::kPlanNodeHostBytes;
    const auto floor = runner.plan_floor_bytes();
    if (floor > std::numeric_limits<std::uint64_t>::max() / calls / (1 + kGraphRatio))
      return Error("plan/graph budget overflow");
    const auto retention = calls * floor * (1 + kGraphRatio);
    // Both comparison arms fund the same maximum two independent temporary
    // plans; retained plans/graphs above cover all actual hinted future calls.
    const auto temporary = 2 * floor;
    const auto base = fixed + runner.weights().size() * en::kPagedExtent +
                      2 * node.StateCapacity() + kOutputBytes + feature_bytes;
    const auto planning_scratch = en::ScratchArenaBytes();
    if (planning_scratch > std::numeric_limits<std::uint64_t>::max() - base ||
        retention > std::numeric_limits<std::uint64_t>::max() - base - planning_scratch ||
        temporary > std::numeric_limits<std::uint64_t>::max() - base - planning_scratch - retention)
      return Error("execution budget overflow");
    const auto derived_budget = base + planning_scratch + retention + temporary;
    if (budget_override != 0 && budget_override < derived_budget)
      return Error("explicit benchmark budget is below the derived minimum");
    const auto budget = budget_override != 0 ? budget_override : derived_budget;
    std::cout << "PREFILL_BUDGET fixed=" << fixed
              << " weights=" << runner.weights().size() * en::kPagedExtent
              << " state_capacity=" << node.StateCapacity() << " publication=" << kOutputBytes
              << " feature_snapshot=" << feature_bytes
              << " pinned_head_envelope=" << std::uint64_t{serving ? 1U : max_rows} * kVocab * 4
              << " max_rows=" << max_rows << " call_bound=" << calls << " plan_floor=" << floor
              << " planning_scratch=" << planning_scratch << " plan_graph_capacity=" << retention
              << " temporary_plans=" << temporary << " derived_minimum=" << derived_budget
              << " total=" << budget << '\n';
    node.SetHostFloor(temporary + runner.host_input_bytes() + planning_scratch);
    if (auto r = node.Start(llmp::base::Bytes(budget)); !r) return r;
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
      const auto warm_extents = (*runner.request_slot(0))->state().extents();
      const auto warm_used = (*runner.request_slot(0))->used_state_bytes();
      if (auto r = runner.Clear(); !r) return r;
      auto fresh_slot = runner.request_slot(0);
      if (!fresh_slot || (*fresh_slot)->completed_positions() != 0 ||
          (*fresh_slot)->used_state_bytes() != 0 || !(*fresh_slot)->state_usable() ||
          (*fresh_slot)->state().kept_extents() != warm_extents ||
          node.has_pending_state_preparation())
        return Error("fresh paid request differs from the retained six-row zero remnant");
      std::cout << "PREFILL_FRESH completed=0 used_bytes=0 warm_rows=6 warm_used_bytes="
                << warm_used << " kept_zero_extents=" << warm_extents.size() << " zero_state=1\n";
      if (account_phases) {
        runner.EnablePhaseAccounting();
        (void)runner.TakePhaseAccounting();
        (void)node.TakeTimes(0);
      }
      const auto graph_before = runner.graph_stats();
      const auto lookahead_before = runner.lookahead_stats();
      const auto preparation_before = (*runner.request_slot(0))->state().preparation_stats();
      const auto started = std::chrono::steady_clock::now();
      // Keep all preparation inside paid prefill. The existing budget funds
      // this same initialized footprint; completed positions stay unchanged.
      // Partial failure follows normal teardown, with no second preparation.
      if (state_prepare == "state-upfront") {
        if (auto r = runner.ReserveStateThrough(0, kPrefill); !r) return r;
      }
      for (std::uint32_t first = 0; first < kPrefill;) {
        const auto rows = std::min(max_rows, kPrefill - first);
        const auto next_first = first + rows;
        const auto near = std::min(max_rows, kPrefill - next_first);
        const auto far = std::min(max_rows, kPrefill - next_first - near);
        if (auto r = runner.ChunkPrefill(
                first, std::span(ids).subspan(first, rows), logits,
                prefill_output == "full" || next_first == kPrefill, near,
                prefill_output == "full" || next_first + near == kPrefill, far,
                prefill_output == "full" || next_first + near + far == kPrefill);
            !r)
          return r;
        first = next_first;
      }
      const auto prefill_policy = runner.last_built_policy();
      const auto prefill = en::support::Seconds(std::chrono::steady_clock::now() - started);
      const auto graph_prefill = runner.graph_stats();
      const auto lookahead_prefill = runner.lookahead_stats();
      const auto preparation_prefill = (*runner.request_slot(0))->state().preparation_stats();
      if (account_phases) {
        // Completed prefill only: exclude anchors, decode, and snapshot writes.
        const auto phases = runner.TakePhaseAccounting();
        const auto jobs = node.TakeTimes(0);
        std::cout << "PREFILL_PHASE planned_calls=" << phases.planned_calls
                  << " hits=" << phases.hits << " misses=" << phases.misses;
        constexpr std::array names{"checks",
                                   "inputs",
                                   "state",
                                   "required_planning",
                                   "staging",
                                   "execution",
                                   "publication_cleanup"};
        for (std::size_t i = 0; i < names.size(); ++i)
          std::cout << ' ' << names[i] << "_seconds=" << phases.seconds[i];
        std::cout << " bind_seconds=" << phases.bind_seconds
                  << " coverage_seconds=" << phases.coverage_seconds
                  << " cache_seconds=" << phases.cache_seconds << " job_steps=" << jobs.steps
                  << " job_wall=" << jobs.wall << " dispatch=" << jobs.dispatch
                  << " submission=" << jobs.job << " after=" << jobs.after
                  << " stream_elapsed=" << jobs.device << '\n';
      }
      if (logits.size() != kVocab) return Error("missing final prefill head");
      const auto save = [&](const char* name) -> en::Status {
        std::ofstream output(out / name, std::ios::binary);
        output.write(reinterpret_cast<const char*>(logits.data()), kOutputBytes);
        output.flush();
        return output ? en::Status{} : en::Status(Error("writing completed head failed"));
      };
      // Completed frontier diagnostics stay outside paid prefill/decode. The
      // pinned copy is catalog-funded before allocation, and CheckBorrow proves
      // the exact cursor/feature epoch and held physical generations stay valid.
      std::uint64_t prefill_feature_epoch = 0;
      const auto save_feature =
          [&](const char* name,
              std::uint32_t position) -> std::expected<std::uint64_t, std::string> {
        if (!retain_features) return 0;
        auto borrowed = runner.BorrowFrozen(0, ids[position - 1]);
        if (!borrowed) return Error(borrowed.error());
        if (borrowed->prefix() != position) return Error("feature cursor mismatch");
        std::vector<llmp::catalog::ExtentId> feature_staging;
        auto pinned = node.Pinned(feature_bytes, 0, feature_staging);
        if (!pinned) return Error(pinned.error());
        if (auto copied = runner.CopyFeatures(0, position - 1, 1, *pinned); !copied)
          return Error(copied.error());
        if (auto checked = runner.CheckBorrow(*borrowed); !checked) return Error(checked.error());
        const auto* values = static_cast<const float*>(*pinned);
        if (!std::all_of(values, values + runner.profile().width,
                         [](float value) { return std::isfinite(value); }))
          return Error("nonfinite completed frontier feature");
        std::ofstream output(out / name, std::ios::binary);
        output.write(static_cast<const char*>(*pinned),
                     static_cast<std::streamsize>(feature_bytes));
        output.flush();
        const bool written = bool(output);
        if (auto freed = node.FreePinned(*pinned); !freed) return Error(freed.error());
        if (!written) return Error("writing completed frontier feature failed");
        std::cout << "PREFILL_FEATURE file=" << name << " position=" << position
                  << " bytes=" << feature_bytes << " rows=1 epoch=" << borrowed->feature_epoch()
                  << " borrowed=1 checked=1\n";
        return borrowed->feature_epoch();
      };
      if (auto r = save("prefill.f32"); !r) return r;
      if (auto epoch = save_feature("prefill-feature.f32", kPrefill); !epoch)
        return Error(epoch.error());
      else
        prefill_feature_epoch = *epoch;
      for (std::uint32_t i = 0; i < kWarm; ++i) {
        const auto past = kPrefill + i;
        if (auto r = runner.Chunk(past, std::span(ids).subspan(past, 1), logits); !r) return r;
        std::cout << "PREFILL_TIMED_PREFIX appended=" << ids[past] << " past=" << past + 1 << '\n';
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
      if (auto epoch = save_feature("final-feature.f32", kInput); !epoch)
        return Error(epoch.error());
      else if (retain_features && *epoch - prefill_feature_epoch != kWarm + kSteps)
        return Error("retained feature epoch differs from actual completed units");
      // Retain the exact completed initialized footprint outside both timers.
      // The transfer lives in separately charged pinned storage; no full host
      // payload vector is allocated. A failed copy remains node-owned until
      // proven teardown, and the whole owner bundle survives an unproven fence.
      auto ranges = runner.CheckpointRanges(kInput);
      if (!ranges) return Error(ranges.error());
      std::uint64_t state_bytes = 0;
      for (const auto& range : *ranges) state_bytes += range.bytes;
      std::vector<llmp::catalog::ExtentId> staging;
      auto state = node.Pinned(state_bytes, 0, staging);
      if (!state) return Error(state.error());
      if (auto copied = runner.CopyState(0, *state, *ranges, true); !copied) return copied;
      std::ofstream checkpoint(out / "initialized-state.bin", std::ios::binary);
      checkpoint.write(static_cast<const char*>(*state), static_cast<std::streamsize>(state_bytes));
      checkpoint.flush();
      const bool written = bool(checkpoint);
      if (auto freed = node.FreePinned(*state); !freed) return freed;
      if (!written) return Error("writing completed state failed");
      std::ofstream layout(out / "state-layout.txt");
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
      std::cout << "PREFILL_NATIVE prefill_seconds=" << prefill << " prefill_rows=" << kPrefill
                << " prefill_chunks=" << chunks << " intermediate_heads="
                << ((prefill_output == "full" || retain_features) ? chunks - 1 : 0)
                << " prefill_output=" << prefill_output << " retained_features=" << retain_features
                << " state_prepare=" << state_prepare << " lookahead=" << lookahead
                << " prepare_state=" << prepare_state << " preparation_submitted="
                << preparation_prefill.submitted - preparation_before.submitted
                << " preparation_completed="
                << preparation_prefill.completed_extents - preparation_before.completed_extents
                << " preparation_adopted="
                << preparation_prefill.adopted_extents - preparation_before.adopted_extents
                << " preparation_refused="
                << preparation_prefill.refused - preparation_before.refused
                << " preparation_failed=" << preparation_prefill.failed - preparation_before.failed
                << " capture_ahead=" << ahead << " lookahead_capacity=" << (ahead ? 2 : 1)
                << " prefill_captures=" << graph_prefill.captured - graph_before.captured
                << " prefill_replays=" << graph_prefill.replayed - graph_before.replayed
                << " prefill_built_pairs="
                << lookahead_prefill.built_pairs - lookahead_before.built_pairs
                << " prefill_cached_pairs="
                << lookahead_prefill.cached_pairs - lookahead_before.cached_pairs
                << " prefill_captured_ahead="
                << lookahead_prefill.captured_ahead - lookahead_before.captured_ahead
                << " prefill_captured_first="
                << lookahead_prefill.captured_first - lookahead_before.captured_first
                << " prefill_dropped_ahead="
                << lookahead_prefill.dropped_ahead - lookahead_before.dropped_ahead
                << " lookahead_attempted=" << runner.lookahead_stats().attempted
                << " lookahead_built=" << runner.lookahead_stats().built
                << " lookahead_cached=" << runner.lookahead_stats().cached
                << " lookahead_refused=" << runner.lookahead_stats().refused
                << " lookahead_build_seconds=" << runner.lookahead_stats().build_seconds
                << " decode_seconds=" << decode << " decode_chunks=" << kSteps
                << " timed_start=" << kPrefill + kWarm << " completed=" << kInput
                << " context=" << context << " chunk=" << max_rows << " policy=" << policy
                << " masks=device"
                << " normmul=" << normmul << " prefill_norm_fused=" << prefill_policy.norm_fused
                << " prefill_norm_rope=" << prefill_policy.norm_rope
                << " prefill_norm_add=" << prefill_policy.norm_add
                << " prefill_gemma_route=" << prefill_policy.gemma_route
                << " prefill_gemma_reduce=" << prefill_policy.gemma_reduce
                << " prefill_owner_steps=" << prefill_policy.owner_attention_steps
                << " gemma_route=" << decode_policy.gemma_route
                << " gemma_reduce=" << decode_policy.gemma_reduce
                << " owner_steps=" << decode_policy.owner_attention_steps
                << " shared_vecq=" << decode_policy.shared_vecq
                << " row_products=" << decode_policy.row_products
                << " norm_fused=" << decode_policy.norm_fused
                << " rope_store=" << decode_policy.rope_store
                << " group_stores=" << (stores == "stores-on")
                << " prefill_grouped_store_steps=" << prefill_policy.grouped_store_steps
                << " prefill_grouped_stores=" << prefill_policy.grouped_stores
                << " prefill_primitive_store_steps=" << prefill_policy.primitive_store_steps
                << " decode_grouped_store_steps=" << decode_policy.grouped_store_steps
                << " decode_grouped_stores=" << decode_policy.grouped_stores
                << " decode_primitive_store_steps=" << decode_policy.primitive_store_steps
                << " norm_rope=" << decode_policy.norm_rope
                << " norm_add=" << decode_policy.norm_add
                << " captures=" << runner.graph_stats().captured
                << " replays=" << runner.graph_stats().replayed << '\n';
      for (std::uint32_t i = 0; i < kSteps; ++i)
        std::cout << "PREFILL_TOKEN step=" << i << " argmax=" << chosen[i]
                  << " forced=" << ids[kPrefill + kWarm + i] << '\n';
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
  if (ran && retired) std::cout << "PREFILL_RETIRED completed=1\n";
  return ran && retired ? 0 : 1;
}
