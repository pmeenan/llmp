// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/serving.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/build_info.h"
#include "base/check.h"
#include "base/report.h"
#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
#include "engine/gemma2_runner.h"
#include "engine/gemma3_runner.h"
#include "engine/gemma4_runner.h"
#include "engine/qwen38_runner.h"
#include "engine/qwen_image_runner.h"
#include "execution/adaptive_depth.h"
#include "execution/adaptive_wave_mode.h"
#include "memory/reclaim.h"
#include "model/dsv4.h"
#include "model/qwen38.h"
#include "platform/crash_policy.h"
#include "platform/files.h"
#include "platform/host_probe.h"
#include "platform/kept_files.h"
#include "platform/memory_pressure.h"
#include "platform/path_trust.h"
#include "runtime/gemma_profile.h"
#include "runtime/gemma_wave.h"
#include "runtime/intake_limits.h"
#include "runtime/memory_guard.h"
#include "runtime/model_limits.h"
#include "runtime/prefill.h"
#include "runtime/swap_room.h"
#include "scheduler/programs.h"

namespace jitllm::runtime {
namespace {

// strftime_now's clock for an interpreted chat template: local time, as
// Python's datetime.now() gives it.
std::optional<chat::jinja::CivilTime> LocalTime() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  if (localtime_r(&now, &tm) == nullptr) {
    return std::nullopt;
  }
  return chat::jinja::CivilTime{.year = tm.tm_year + 1900,
                                .month = tm.tm_mon + 1,
                                .day = tm.tm_mday,
                                .hour = tm.tm_hour,
                                .minute = tm.tm_min,
                                .second = tm.tm_sec,
                                .weekday = tm.tm_wday,
                                .yearday = tm.tm_yday};
}

class HistoryFundingGuard {
 public:
  explicit HistoryFundingGuard(bool& funding)
      : funding_(funding), before_(std::exchange(funding, true)) {}
  HistoryFundingGuard(const HistoryFundingGuard&) = delete;
  HistoryFundingGuard& operator=(const HistoryFundingGuard&) = delete;
  HistoryFundingGuard(HistoryFundingGuard&&) = delete;
  HistoryFundingGuard& operator=(HistoryFundingGuard&&) = delete;
  ~HistoryFundingGuard() { funding_ = before_; }

 private:
  bool& funding_;
  bool before_;
};

namespace fs = std::filesystem;
namespace ja = jitllm::artifact;

constexpr std::uint64_t kExtent = engine::kPagedExtent;
// Extents the page-in observer tracks (the M3 models use about 100,000).
constexpr std::size_t kObservedExtents = std::size_t{1} << 18U;
// Each model's settings (prefill chunk, request slots, drafting, waves,
// floors, sampling) resolve at registration in model_settings.h's three
// layers (D-103); their fallbacks are its constants, which name what the
// engine does too.
static_assert(Llm::kMaxBranches == engine::kMaxRequestSlots);
static_assert(config::kMaxModelSlots == engine::kMaxRequestSlots);
static_assert(kDsv4SpeculativeMostSlots == engine::Dsv4Runner::kWaveRows / 2);
static_assert(config::kMaxWaveCostWidths + 1 == execution::AdaptiveWaveMode::kMaxWidth);

// A model's request slots for the start's log, and where their number came
// from (its settings).
std::string SlotsReport(const ModelSettings& s) {
  const std::uint32_t slots = s.max_slots.value;
  return std::format("{} request slot{} ({}: {})", slots, slots == 1 ? "" : "s",
                     SourceName(s.max_slots.source), s.max_slots.basis);
}

// The reclaim order's cost of idle conversation state (Server::Reclaim):
// its write-back and its restore, at the node's rates measured so far, and
// before any at the rates measured on a GB10 (spark-b, DeepSeek's four
// slots' state, 1.42 GB written back at 11.0 GB/s and read again at 14.5
// GB/s; docs/experiments/memory-pressure). Dropping it instead (no spill
// budget) costs its recomputation: its tokens at the model's
// recompute_ms_per_token.
constexpr double kSpillBytesPerSecond = 11.0e9;
constexpr double kRestoreBytesPerSecond = 14.5e9;
// A graph's capture and instantiation a GiB counted, measured on a GB10
// (0.33–0.45 s in the services' logs): what a graph's capture weighs
// against before any graph is held (Server::Reclaim).
constexpr double kGraphSecondsPerGiB = 0.36;
// How often the driver's maintenance looks (between units and when idle).
constexpr auto kMaintenanceInterval = std::chrono::milliseconds(250);

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

// An installed artifact, opened under the store's trust rules (D-056's
// integrity rests on them) and its ID checked; the runner opens it again.
std::expected<ja::Artifact, std::string> OpenTrusted(const fs::path& store, const std::string& id) {
  return OpenInstalled(store, id);
}

std::string GraphJson(const GraphCounts& g) {
  return std::format(R"({{"eager":{},"captured":{},"replayed":{},"refused":{},"kept":{}}})",
                     g.eager, g.captured, g.replayed, g.refused, g.kept);
}

GraphCounts Sum(const engine::GraphStats& a, const engine::GraphStats& b, std::size_t kept) {
  return {.eager = a.eager + b.eager,
          .captured = a.captured + b.captured,
          .replayed = a.replayed + b.replayed,
          .refused = a.refused + b.refused,
          .kept = kept};
}

// ------------------------------------------- conversations kept (D-105)

// A slot's regions' bytes in its spill file (whole extents), or their
// layouts' bytes.
std::vector<std::uint64_t> KeptRegionsOf(const engine::LiveState& live, bool mapped) {
  std::vector<std::uint64_t> out;
  out.reserve(live.regions());
  for (std::size_t i = 0; i < live.regions(); ++i) {
    out.push_back(mapped ? live.mapped_bytes(i) : live.bytes(i));
  }
  return out;
}

std::string Errno(int error) { return std::system_category().message(error); }

// Wall-clock milliseconds for a steady time point, and back: a record's
// times survive a restart, the steady clock's do not.
std::int64_t UnixMs(Clock::time_point at) {
  const auto unix = std::chrono::system_clock::now() - (Clock::now() - at);
  return std::chrono::duration_cast<std::chrono::milliseconds>(unix.time_since_epoch()).count();
}
Clock::time_point SteadyAt(std::int64_t unix_ms) {
  const auto unix = std::chrono::system_clock::time_point(std::chrono::milliseconds(unix_ms));
  return Clock::now() -
         std::chrono::duration_cast<Clock::duration>(std::chrono::system_clock::now() - unix);
}
std::int64_t NowUnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

kept::FileId KeptId(const platform::FileIdentity& id) {
  return {.device = id.device, .inode = id.inode, .generation = id.generation};
}

std::vector<kept::Range> KeptRanges(std::span<const engine::LiveState::Range> ranges) {
  std::vector<kept::Range> out;
  out.reserve(ranges.size());
  for (const auto& r : ranges) {
    out.push_back(
        {.region = static_cast<std::uint32_t>(r.region), .offset = r.offset, .bytes = r.bytes});
  }
  return out;
}

std::vector<engine::LiveState::Range> LiveRanges(std::span<const kept::Range> ranges) {
  std::vector<engine::LiveState::Range> out;
  out.reserve(ranges.size());
  for (const auto& r : ranges) {
    out.push_back({.region = r.region, .offset = r.offset, .bytes = r.bytes});
  }
  return out;
}

// What this build is, for a kept record: its version and commit, and the
// executable file's identity, so another build, or this one rebuilt or
// reinstalled, never adopts what it did not write.
std::string KeptBuild() {
  const base::BuildInfo& info = base::GetBuildInfo();
  const auto stamp = platform::RunningExecutableStamp();
  const std::string exe =
      stamp ? std::format("{}:{}:{}:{}:{}", stamp->device, stamp->inode,
                          stamp->generation ? std::to_string(*stamp->generation) : "-", stamp->size,
                          stamp->modified_ns)
            : std::string("unknown");
  return std::format("{};{}{};{};{};exe={}", info.version, info.commit,
                     info.modified ? "+modified" : "", info.sdk, info.target, exe);
}

// ---------------------------------------------------------------- the models

// DeepSeek V4 Flash (engine/dsv4_runner.h), with DSpark as its drafter.
class Dsv4 final : public Llm {
 public:
  Dsv4(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
       const config::RuntimeRoles& roles, int index)
      : entry_(entry),
        artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    node_ = &node;
    checkpoint_directory_ = roles.spill;
    speculate_ = !drafter_id_.empty() && settings.speculation.value;
    // DSpark's drafts a step, and its verify's rows: the anchor and them.
    options_.draft_rows = settings.draft_rows.value;
    options_.max_verify = settings.draft_rows.value + 1;
    execution::AdaptiveWaveMode::Costs costs{};
    for (std::size_t w = 0; w < settings.wave_costs.value.size() && w + 2 < costs.size(); ++w) {
      costs[w + 2] = settings.wave_costs.value[w];
    }
    wave_mode_ = execution::AdaptiveWaveMode(costs, WaveForce(settings.wave_form.value),
                                             CutRows(options_.max_verify));
    context_ = settings.context.value;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = context_;
    configured_rows_ = settings.prefill_chunk.value;
    max_rows_ = PrefillChunkRows(context_, std::nullopt, settings.prefill_chunk.value,
                                 model::Dsv4MostRows(model::Dsv4Flash(), context_));
    options_.max_rows = max_rows_;
    options_.graphs = true;
    // Request slots share the weights and workspace, each with its own
    // conversation state: concurrent chat requests decode in waves
    // (engine/dsv4_runner.h), whose row-local products read each weight
    // once for all of them. With DSpark each request's verify joins a wave
    // with at least two rows, so a wave of sixteen holds eight (the
    // settings' cap).
    options_.wave_slots = settings.max_slots.value;
    slots_report_ = SlotsReport(settings);
    // The output-A/HCA prefill: on full 4,096-row chunks, and on every
    // prefill chunk of 64 rows or more, where it passes every registered
    // quality control on both GGUFs under the tie-aware rule
    // (docs/experiments/ds4-output-prefix, "Default-on acceptance"): the
    // served defaults, which an owner may turn off for exactness.
    options_.prefill_outa_hca = settings.prefill_outa_hca.value;
    options_.prefill_outa_hca_partial =
        settings.prefill_outa_hca.value && settings.prefill_outa_hca_partial.value;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
    }
  }

  engine::PagedModel& paged() override { return runner_; }

  Status Setup() override {
    if (auto opened = OpenArtifacts(); !opened) {
      return opened;
    }
    if (auto setup = runner_.Setup(); !setup) {
      return setup;
    }
    for (std::size_t i = 0; i < runner_.wave_capacity(); ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) {
        return std::unexpected(slot.error());
      }
      native_slots_[i] = *slot;
    }
    return PrepareBranches(runner_.wave_capacity(), 1);
  }

  Status OpenArtifacts() {
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) {
      return std::unexpected(artifact.error());
    }
    if (artifact->model().architecture != "deepseek4") {
      return Error(std::format("artifact {} is a {}, not DeepSeek V4 (deepseek4)", artifact_id_,
                               artifact->model().architecture));
    }
    auto binding = model::BindDsv4(model::Dsv4Flash(), *artifact);
    if (!binding) {
      return std::unexpected(binding.error());
    }
    options_.frontier_head = Dsv4FrontierHeadForServing(*binding);
    if (speculate_) {
      if (auto drafter = OpenTrusted(store_, drafter_id_); !drafter) {
        return std::unexpected(drafter.error());
      }
    }
    // The tokenizer and template from the artifact's kept GGUF metadata
    // (import rule 7): the first shard's.
    return UseChatAssets(*artifact, entry_);
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override { return runner_.host_input_bytes(); }
  std::uint64_t plan_floor_bytes() const override { return runner_.plan_floor_bytes(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  std::string plan_report() const override { return runner_.plan_report(); }
  std::string serial_reason() const override { return runner_.serial_reason(); }
  std::string slots_report() const override { return slots_report_; }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  std::vector<catalog::ExtentId> kept_state() const override { return runner_.kept_state(); }
  std::vector<catalog::ExtentId> unchanged_state() const override {
    return runner_.unchanged_state();
  }
  void StateWrittenBack(bool whole) override { runner_.StateWrittenBack(whole); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  const catalog::Closure& request_closure() const override { return runner_.execution_closure(); }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  Status PrepareDefaultRequest() override {
    std::array<Branch*, 1> active{&default_branch()};
    return SelectBranches(active);
  }
  Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > runner_.wave_capacity()) {
      return Error("too many DeepSeek native conversation branches");
    }
    std::array<engine::Dsv4Runner::Slot*, engine::Dsv4Runner::kRequestSlots> selected{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (active[i] == nullptr || &active[i]->model() != this) {
        return Error("a selected DeepSeek conversation belongs to another model");
      }
      selected[i] = &NativeSlot(*active[i]);
    }
    return runner_.SelectSlots(std::span(selected).first(active.size()));
  }
  bool supports_generation_waves() const override { return runner_.waves_provisioned(); }
  std::size_t generation_wave_capacity() const override { return runner_.wave_capacity(); }
  std::uint64_t StateBytesThrough(std::uint32_t positions) const override {
    return runner_.StateBytesThrough(positions).value_or(0);
  }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  Status AfterLoad() override { return runner_.CheckHashRouting(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  double plan_seconds() const override { return runner_.plan_seconds(); }
  GraphCounts graphs() const override {
    return Sum(runner_.graph_stats(), runner_.draft_stats(), runner_.graphs());
  }
  std::string violations() const override {
    return runner_.coverage_violations() == 0
               ? std::string()
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             runner_.coverage_violations(), runner_.first_violation());
  }
  std::string extra() const override {
    return std::format(
        R"({{"turn_checkpoints":{},"turn_checkpoint_bytes":{},"used_state_bytes":{},"architecture":"deepseek4","speculation":{},"coverage_tensors":{},)"
        R"("slab_padding":{},"state_bytes":{},"drafter_state_bytes":{},"graphs":{}}})",
        turn_checkpoints(), turn_checkpoint_bytes(), runner_.used_state_bytes(),
        speculate_ ? "\"dspark\"" : "null", runner_.coverage_tensors(), runner_.slab_padding(),
        runner_.state_bytes(), runner_.drafter_state_bytes(), GraphJson(graphs()));
  }
  void Defaults(chat::Conversation& c) const override {
    // As llama-server's /apply-template rendered the references' prompts:
    // its default turns the template's thinking on.
    if (!c.enable_thinking) {
      c.enable_thinking = true;
    }
  }

 protected:
  // The default conversation's scalar entry points forward to its branch.
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                  std::vector<float>& logits) override {
    return RunChunkFor(default_branch(), all, n_past, inject, logits);
  }
  Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                  std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                  std::uint64_t& drafted) override {
    return SpecStepFor(default_branch(), all, pos, left, kept, logits, drafted);
  }
  Status Settle() override { return SettleFor(default_branch()); }
  Status ClearState() override { return ClearStateFor(default_branch()); }
  std::uint64_t target_state_base() const override { return TargetStateBaseFor(default_branch()); }
  std::uint64_t used_state_bytes() const override { return UsedStateBytesFor(default_branch()); }
  bool StateUsable() const override { return StateUsableFor(default_branch()); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    return PrepareDecodeStateFor(default_branch(), pos, left);
  }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return UsedStateRangesFor(default_branch());
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return SaveUsedStateFor(default_branch(), host, ranges);
  }
  Status RestoreUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return RestoreUsedStateFor(default_branch(), host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const override {
    return CheckpointRangesFor(default_branch(), positions);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range> footprint) override {
    return PrepareRestoreStateFor(default_branch(), footprint);
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return CopyCheckpointStateFor(default_branch(), host, ranges, to_host);
  }
  std::uint64_t target_state_bytes() const override {
    return TargetStateBytesFor(default_branch());
  }
  std::uint64_t drafter_state_base() const override {
    return DrafterStateBaseFor(default_branch());
  }
  std::uint64_t drafter_state_bytes() const override {
    return DrafterStateBytesFor(default_branch());
  }

  // Each conversation branch on its own native request slot.
  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t n_past,
                     bool inject, std::vector<float>& logits) override {
    return NativeSlot(branch).Chunk(
        n_past, all.subspan(n_past), logits,
        inject ? engine::Dsv4ChunkKind::kInject : engine::Dsv4ChunkKind::kPlain);
  }
  Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                     std::uint32_t left, std::vector<std::int32_t>& kept,
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                     bool* prefix_kept = nullptr) override {
    const std::uint32_t rows = VerifyRows(pos, left, runner_.max_verify());
    std::vector<std::int32_t> drafts;
    std::vector<float> verified;
    if (auto r = NativeSlot(branch).DraftVerify(pos, all.back(), rows, drafts, verified); !r) {
      return r;
    }
    return Undone(branch, Judge(branch, pos, rows, drafts, verified, kept, logits, drafted),
                  prefix_kept);
  }
  Status SettleFor(Branch& branch) override { return NativeSlot(branch).Rollback(); }
  Status ClearStateFor(Branch& branch) override { return NativeSlot(branch).Clear(); }
  // A conversation outside the cohort: its reuse cache cleared for peers
  // short of state capacity (Branch::ReleaseIdleState).
  Status ReleaseIdleStateFor(Branch& branch) override { return NativeSlot(branch).ClearIdle(); }
  Status SpillFor(Branch& branch) override { return NativeSlot(branch).Spill(); }
  Status RestoreFor(Branch& branch) override { return NativeSlot(branch).Restore(); }
  bool SpilledFor(const Branch& branch) const override { return NativeSlot(branch).spilled(); }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spilled_bytes();
  }
  std::uint64_t SpillWriteBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spill_write_bytes();
  }
  bool LeasedFor(const Branch& branch) const override { return NativeSlot(branch).held(); }
  std::uint64_t RefusedBytesFor(const Branch& branch) const override {
    const auto& slot = NativeSlot(branch);
    return slot.spilled() ? slot.spilled_bytes() : slot.refused_bytes();
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return NativeSlot(branch).state_refused();
  }
  bool StateUsableFor(const Branch& branch) const override {
    return NativeSlot(branch).state_usable();
  }

 public:
  // Kept across a restart (D-105).
  std::string KeptLayout() const override { return runner_.kept_layout(); }
  std::vector<std::uint64_t> KeptRegions() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).live(), true);
  }
  std::vector<std::uint64_t> KeptLayouts() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).live(), false);
  }
  // D-102's rung 2.
  Status RecoverInPlace() override {
    auto discarded = runner_.RecoverInPlace([this](std::uint32_t slot) { UnkeepSlot(slot); });
    if (!discarded) {
      return std::unexpected(discarded.error());
    }
    ForgetDiscarded(*discarded);
    return {};
  }

 protected:
  const engine::LiveState* KeptLiveFor(const Branch& branch) const override {
    return &NativeSlot(branch).live();
  }
  bool KeptWholeFor(const Branch& branch) const override { return NativeSlot(branch).kept_whole(); }
  bool OwedFor(const Branch& branch) const override { return NativeSlot(branch).owed(); }
  Status AdoptFor(Branch& branch, std::span<const engine::LiveState::Range> used) override {
    return NativeSlot(branch).Adopt(used);
  }
  void SetSpillPlaces(
      const std::function<engine::LiveState::SpillPlace(std::uint32_t)>& place) override {
    options_.spill_place = place;
  }
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) override {
    const std::uint32_t rows = speculate_ ? VerifyRows(pos, left, runner_.max_verify()) : 1U;
    return NativeSlot(branch).ReserveStateThrough(pos + rows);
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).state_base();
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).state_bytes();
  }
  std::uint64_t DrafterStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).drafter_state_base();
  }
  std::uint64_t DrafterStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).drafter_state_bytes();
  }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_ranges();
  }
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override {
    return NativeSlot(branch).SaveUsedState(host, ranges);
  }
  Status RestoreUsedStateFor(Branch& branch, void* host,
                             std::span<const engine::LiveState::Range> ranges) override {
    return NativeSlot(branch).RestoreUsedState(host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t positions) const override {
    return NativeSlot(branch).CheckpointRanges(positions);
  }
  Status PrepareRestoreStateFor(Branch& branch,
                                std::span<const engine::LiveState::Range> footprint) override {
    return NativeSlot(branch).PrepareRestoreState(footprint);
  }
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override {
    return NativeSlot(branch).CopyCheckpointState(host, ranges, to_host);
  }

  // DeepSeek keeps no speculation cursor and no adaptive draft depth: every
  // branch's are the defaults, saved and restored as no state.
  std::uint32_t CursorFor(const Branch& branch) const override {
    (void)BranchIndex(branch);
    return 0;
  }
  void SetCursorFor(Branch& branch, std::uint32_t /*value*/) override { (void)BranchIndex(branch); }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }

  bool GenerationCohortUsable() const override { return runner_.cohort_usable(); }

  // Several branches' steps in one wave (engine/dsv4_runner.h): a decode
  // step of each, or with DSpark each one's draft and one joined verify of
  // every branch's rows (each branch's share of the wave's rows) or a
  // decode step of each, as AdaptiveWaveMode chooses. A lone branch keeps
  // its ordinary step.
  Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) override {
    if (!runner_.waves_provisioned() || prepared.size() == 1) {
      return Llm::RunPreparedGenerationWave(prepared);
    }
    if (!speculate_) {
      std::vector<engine::Dsv4Runner::WaveWork> work;
      work.reserve(prepared.size());
      for (PreparedGeneration& unit : prepared) {
        if (unit.step.all.size() != std::size_t{unit.step.position} + 1) {
          return Error("a DeepSeek decode wave step is not one anchor row");
        }
        work.push_back({.slot = &NativeSlot(*unit.branch),
                        .pos = unit.step.position,
                        .anchor = unit.step.all.back(),
                        .rows = 1,
                        .drafts = nullptr,
                        .logits = &unit.row});
      }
      return runner_.DecodeWave(work);
    }
    // With DSpark, a wave of several requests is a draft-verify wave or a
    // plain decode wave (the drafter still fed), as the accepted tokens
    // against the measured cost of each width choose: never by wall time,
    // so the same waves choose the same forms (execution/adaptive_wave_mode.h).
    // A sampling member keeps every wave speculative.
    const auto width = static_cast<std::uint32_t>(prepared.size());
    const bool sampled = std::ranges::any_of(
        prepared, [this](const PreparedGeneration& unit) { return sampling(*unit.branch); });
    const auto mode = wave_cost_exploration_.Choose(wave_mode_.Choose(width, sampled), width,
                                                    sampled, settings_, calibration_record_.known);
    std::uint32_t tokens = 0;
    std::uint32_t complete = 0;
    const double planned = runner_.plan_seconds();
    const std::uint64_t captured = runner_.graph_stats().captured + runner_.draft_stats().captured;
    const auto started = Clock::now();
    auto waved = mode == execution::AdaptiveWaveMode::Mode::kPlain
                     ? PlainWave(prepared)
                     : SpeculativeWave(prepared, tokens, complete);
    const double seconds = Seconds(Clock::now() - started);
    // A sampling member's speculation accepts differently: not observed.
    if (waved && !sampled) {
      wave_mode_.Observe(width, mode, tokens, complete);
    }
    // Each form's wave time at this width: the calibration of its cost
    // (calibration.h), in force from the next start, never this service's.
    // Only steady waves count, matched between the forms: none that
    // planned or captured a graph (a shape's first waves), and a
    // draft-verify wave only when every member's full verify joined it
    // (not one cut by a mask width or the reply's end, or run alone).
    const bool speculative = mode == execution::AdaptiveWaveMode::Mode::kSpeculative;
    if (waved && runner_.plan_seconds() == planned &&
        runner_.graph_stats().captured + runner_.draft_stats().captured == captured &&
        (!speculative || complete == width)) {
      calibration_samples_.Wave(width, speculative, seconds);
    }
    return waved;
  }

 private:
  // A plain decode wave of a speculative model's requests: one row each,
  // its token chosen as an ordinary step's (greedy: the argmax). A failed
  // choice keeps the processed anchor, as an ordinary plain step's does.
  Status PlainWave(std::span<PreparedGeneration> prepared) {
    std::vector<engine::Dsv4Runner::WaveWork> work;
    std::vector<std::vector<float>> rows(prepared.size());
    work.reserve(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const PreparedGeneration& unit = prepared[i];
      if (unit.step.all.size() != std::size_t{unit.step.position} + 1) {
        return Error("a DeepSeek decode wave step is not one anchor row");
      }
      work.push_back({.slot = &NativeSlot(*unit.branch),
                      .pos = unit.step.position,
                      .anchor = unit.step.all.back(),
                      .rows = 1,
                      .drafts = nullptr,
                      .logits = &rows[i]});
    }
    if (auto ran = runner_.DecodeWave(work); !ran) {
      return ran;
    }
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      auto chosen = Choose(*unit.branch, rows[i], std::uint64_t{unit.step.position} + 1);
      if (!chosen) {
        unit.result = std::unexpected(chosen.error());
        unit.anchor_processed = true;
        continue;
      }
      unit.kept = {*chosen};
      if (unit.step.need_logits) {
        unit.logits.push_back(std::move(rows[i]));
      }
    }
    return {};
  }

  // Each request's draft and one joined verify of every request's rows.
  // `tokens` and `complete`: the tokens kept by, and the count of, the
  // requests whose verify took its full rows in the joined wave (what the
  // wave form's acceptance average counts).
  Status SpeculativeWave(std::span<PreparedGeneration> prepared, std::uint32_t& tokens,
                         std::uint32_t& complete) {
    tokens = 0;
    complete = 0;
    std::vector<engine::Dsv4Runner::WaveWork> work;
    work.reserve(prepared.size());
    // These owners never move once the borrowed descriptors are made.
    struct Frame {
      std::uint32_t rows = 0;
      std::vector<std::int32_t> drafts;
      std::vector<float> verified;
    };
    std::vector<Frame> frames(prepared.size());
    // A verify of one row (a step ending at a mask width, or the last
    // token) runs alone: a wave of wider verifies would give it the
    // multi-row launch, not its own step's arithmetic. So does a lone one.
    const auto share = [](std::size_t count) { return WaveShare(count); };
    std::vector<std::size_t> joined;
    std::vector<std::size_t> alone;
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const PreparedGeneration& unit = prepared[i];
      const std::uint32_t rows = VerifyRows(unit.step.position, unit.step.left,
                                            std::min(runner_.max_verify(), share(prepared.size())));
      (rows == 1 ? alone : joined).push_back(i);
    }
    if (joined.size() == 1) {
      alone.push_back(joined.front());
      joined.clear();
    }
    for (const std::size_t i : alone) {
      PreparedGeneration& unit = prepared[i];
      unit.result = SpecStepFor(*unit.branch, unit.step.all, unit.step.position, unit.step.left,
                                unit.kept, unit.step.need_logits ? &unit.logits : nullptr,
                                unit.drafted, &unit.failed_prefix_valid);
      if (!unit.result && !runner_.cohort_usable()) {
        return std::unexpected(unit.result.error());
      }
    }
    if (joined.empty()) {
      return {};
    }
    const std::uint32_t full = std::min(runner_.max_verify(), share(joined.size()));
    for (const std::size_t i : joined) {
      PreparedGeneration& unit = prepared[i];
      frames[i].rows = VerifyRows(unit.step.position, unit.step.left, full);
      work.push_back({.slot = &NativeSlot(*unit.branch),
                      .pos = unit.step.position,
                      .anchor = unit.step.all.back(),
                      .rows = frames[i].rows,
                      .drafts = &frames[i].drafts,
                      .logits = &frames[i].verified});
    }
    if (auto ran = runner_.DraftVerifyWave(work); !ran) {
      return ran;
    }
    for (const std::size_t i : joined) {
      PreparedGeneration& unit = prepared[i];
      Frame& frame = frames[i];
      unit.result =
          Judge(*unit.branch, unit.step.position, frame.rows, frame.drafts, frame.verified,
                unit.kept, unit.step.need_logits ? &unit.logits : nullptr, unit.drafted);
      if (unit.result && frame.rows == full) {
        tokens += static_cast<std::uint32_t>(unit.kept.size());
        ++complete;
      }
      if (!unit.result) {
        // The shared wave completed; only this branch's judgement failed.
        // Undo its own verify before keeping its prefix.
        if (auto rolled = NativeSlot(*unit.branch).DiscardVerify(); !rolled) {
          if (!runner_.cohort_usable()) {
            return Error(
                std::format("{}; rolling back failed: {}", unit.result.error(), rolled.error()));
          }
          unit.result = Error(
              std::format("{}; rolling back failed: {}", unit.result.error(), rolled.error()));
        } else {
          unit.failed_prefix_valid = true;
        }
      }
    }
    return {};
  }

  // A failed judgement after the native verify completed: the verify is
  // undone, so the step's starting prefix still holds (`prefix_kept`, as a
  // wave's failed_prefix_valid); an undo that fails says so.
  Status Undone(Branch& branch, Status judged, bool* prefix_kept) {
    if (judged) {
      return judged;
    }
    if (auto rolled = NativeSlot(branch).DiscardVerify(); !rolled) {
      return Error(std::format("{}; rolling back failed: {}", judged.error(), rolled.error()));
    }
    if (prefix_kept != nullptr) {
      *prefix_kept = true;
    }
    return judged;
  }

  // The verify's rows from `pos`: the anchor and its drafts, within the
  // tokens left, `most`, the context and the steps' mask widths (D-092).
  std::uint32_t VerifyRows(std::uint32_t pos, std::uint32_t left, std::uint32_t most) const {
    auto rows = std::min<std::uint32_t>({options_.draft_rows + 1, most, left, context_ - pos});
    rows = std::max<std::uint32_t>(rows, 1);
    while (rows > 1 && !model::Dsv4SameWidths(runner_.state_layout(), pos, rows)) {
      --rows;
    }
    return rows;
  }
  // A completed verify's verdict for `branch`: accept drafts while the
  // target agrees (greedy: its argmax; sampling: speculative sampling's
  // verdict); the first disagreement, or the row after the last draft,
  // gives the next token. Row m predicts the token at pos + m + 1.
  Status Judge(Branch& branch, std::uint32_t pos, std::uint32_t rows,
               std::vector<std::int32_t>& drafts, std::span<const float> verified,
               std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
               std::uint64_t& drafted) {
    const std::uint32_t vocab = runner_.vocab();
    if (rows == 0 || drafts.size() + 1 < rows || verified.size() != std::size_t{rows} * vocab) {
      return Error("a DeepSeek verify returned inconsistent draft/logit counts");
    }
    drafts.resize(rows - 1);
    const auto row = [&](std::uint32_t i) {
      return verified.subspan(std::size_t{i} * vocab, vocab);
    };
    std::uint32_t m = 0;
    std::int32_t next = -1;
    for (; m < rows - 1; ++m) {
      std::int32_t instead = -1;
      auto keep = Keep(branch, row(m), drafts[m], std::uint64_t{pos} + m + 1, instead);
      if (!keep) {
        return std::unexpected(keep.error());
      }
      if (!*keep) {
        next = instead;
        break;
      }
    }
    if (next < 0) {
      auto chosen = Choose(branch, row(m), std::uint64_t{pos} + m + 1);
      if (!chosen) {
        return std::unexpected(chosen.error());
      }
      next = *chosen;
    }
    if (auto r = NativeSlot(branch).Accept(m + 1); !r) {
      return r;
    }
    kept.assign(drafts.begin(), drafts.begin() + m);
    kept.push_back(next);
    if (logits != nullptr) {
      for (std::uint32_t i = 0; i <= m; ++i) {
        logits->emplace_back(row(i).begin(), row(i).end());
      }
    }
    drafted += rows - 1;
    return {};
  }
  engine::Dsv4Runner::Slot& NativeSlot(Branch& branch) {
    const auto index = BranchIndex(branch);
    base::Check(native_slots_[index] != nullptr, "native conversation slots are not ready");
    return *native_slots_[index];
  }
  const engine::Dsv4Runner::Slot& NativeSlot(const Branch& branch) const {
    const auto index = BranchIndex(branch);
    base::Check(native_slots_[index] != nullptr, "native conversation slots are not ready");
    return *native_slots_[index];
  }
  config::ModelEntry entry_;  // its tokenizer and template files, if named
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  fs::path store_;
  std::string slots_report_;
  engine::Dsv4Options options_;  // before the runner, which keeps a reference
  engine::Dsv4Runner runner_;
  std::array<engine::Dsv4Runner::Slot*, engine::Dsv4Runner::kRequestSlots> native_slots_{};
  // With DSpark: draft-verify or plain waves, by width (RunPreparedGenerationWave),
  // as the settings' wave_costs (a draft-verify wave's time over a plain
  // decode wave's, by width; model_settings.h kDsv4WaveCosts measured them
  // with each form forced) and wave_form choose.
  // A request's share of a draft-verify wave's rows among `count`: all
  // sixteen's, at least two (so DSpark takes at most eight requests).
  static std::uint32_t WaveShare(std::size_t count) {
    return std::max<std::uint32_t>(
        2, static_cast<std::uint32_t>(engine::Dsv4Runner::kWaveRows / count));
  }
  // By width, a draft-verify wave's verify rows a request where its share
  // cuts them below the full verify's `most` (AdaptiveWaveMode::Rows).
  static execution::AdaptiveWaveMode::Rows CutRows(std::uint32_t most) {
    execution::AdaptiveWaveMode::Rows rows{};
    for (std::uint32_t w = 2; w <= execution::AdaptiveWaveMode::kMaxWidth; ++w) {
      rows[w] = WaveShare(w) < most ? WaveShare(w) : 0;
    }
    return rows;
  }
  // The model's configured wave form (wave_form): chosen, or forced for
  // exactness controls.
  static execution::AdaptiveWaveMode::Force WaveForce(config::WaveForm form) {
    switch (form) {
      case config::WaveForm::kSpeculative:
        return execution::AdaptiveWaveMode::Force::kSpeculative;
      case config::WaveForm::kPlain:
        return execution::AdaptiveWaveMode::Force::kPlain;
      case config::WaveForm::kAuto:
        break;
    }
    return execution::AdaptiveWaveMode::Force::kNone;
  }
  execution::AdaptiveWaveMode wave_mode_;
  WaveCostExploration wave_cost_exploration_;
};

// Approved Gemma26/31 profiles share the serving driver and native slots.
// Ordinary Gemma31 and Gemma26 use their qualified recipes within their bounded
// envelopes; larger configurations and explicit arithmetic diagnostics retain
// their own paths.
class Gemma final : public Llm {
 public:
  Gemma(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
        const config::RuntimeRoles& roles, int index, engine::Gemma4Variant variant,
        const ServingOptions& serving)
      : entry_(entry),
        artifact_id_(entry.artifact.value_or("")),
        store_(roles.installed),
        profile_(variant == engine::Gemma4Variant::k26BA4B ? model::Gemma4_26BA4B()
                                                           : model::Gemma4_31B()),
        options_(Options(entry, settings, roles, variant, serving.gemma_row_invariant)),
        joined_(serving.gemma_joined ||
                (variant == engine::Gemma4Variant::k31B && settings.gemma31_production) ||
                (variant == engine::Gemma4Variant::k26BA4B && settings.gemma26_production)),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    node_ = &node;
    context_ = options_.context;
    max_rows_ = options_.max_rows;
    configured_rows_ = settings.prefill_chunk.value;
    checkpoint_directory_ = roles.spill;
  }
  engine::PagedModel& paged() override { return runner_; }
  Status Setup() override {
    if (entry_.drafter || settings_.speculation.value)
      return Error("Gemma serving has no qualified assistant or speculative path");
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) return Error(artifact.error());
    auto binding = model::BindGemma4(profile_, *artifact);
    if (!binding) return Error(binding.error());
    if (auto r = UseChatAssets(*artifact, entry_); !r) return r;
    // This first serving slice has no generated thought/tool channel parser.
    // Plain chat stops before a channel control can become answer text.
    for (const auto text : {"<|channel>", "<|tool_call>"})
      if (const auto token = tokenizer_->Find(text)) chat_stops_.push_back(*token);
    if (auto r = runner_.Setup(); !r) return r;
    for (std::uint32_t i = 0; i < options_.slots; ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) return Error(slot.error());
      slots_[i] = *slot;
    }
    return PrepareBranches(options_.slots, 1);
  }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override {
    // The startup guard sets this bounded heap workspace apart before any
    // scalar output or per-owner sampling vector grows. One retained frontier
    // and one prepared result per owner, plus the sampling candidate capacity.
    // The independently catalog-backed pinned output holds one row per slot.
    const auto heap = std::uint64_t{options_.slots} * profile_.vocab *
                      (2 * sizeof(float) + 2 * sizeof(execution::SamplingCandidate));
    return runner_.host_input_bytes() + heap;
  }
  std::uint64_t plan_floor_bytes() const override { return runner_.plan_floor_bytes(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  std::vector<catalog::ExtentId> kept_state() const override { return runner_.kept_state(); }
  void StateWrittenBack(bool whole) override { runner_.StateWrittenBack(whole); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  const catalog::Closure& request_closure() const override { return runner_.closure(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  GraphCounts graphs() const override {
    const auto& g = runner_.graph_stats();
    return {g.eager, g.captured, g.replayed, g.refused, runner_.graph_count()};
  }
  std::string violations() const override { return runner_.coverage().first_violation; }
  std::string extra() const override {
    const auto& selected = runner_.last_built_policy();
    return std::format(
        R"({{"architecture":"gemma4","dispatch":"{}","optional_optimizations":{},"row_invariant":{},"profile":"{}","coverage_tensors":{},"pitch_padding":{},"joined_groups":{},"joined_units":{},"gemma31_candidate":{},"gemma26_candidate":{},"max_rows":{},"norm_rope_requested":{},"norm_add_requested":{},"owner_attention_requested":{},"last_built_rows":{},"last_built_segments":{},"last_built_norm_rope":{},"last_built_norm_add":{},"last_built_owner_attention":{},"common_owner_reads_requested":{},"bounded_owner_roots_requested":{},"max_bound_bounded_owner_attention":{}}})",
        settings_.gemma31_production || settings_.gemma26_production ? "joined-candidate"
        : joined_                                                    ? "joined-diagnostic"
                                                                     : "scalar-cohort",
        joined_ || options_.row_invariant, options_.row_invariant, profile_.name,
        runner_.coverage().tensors, runner_.pitch_padding(), joined_groups_, joined_units_,
        settings_.gemma31_production, settings_.gemma26_production, options_.max_rows,
        options_.fuse_norm_rope, options_.fuse_norm_add, options_.owner_attention, selected.rows,
        selected.segments, selected.norm_rope, selected.norm_add, selected.owner_attention_steps,
        options_.common_owner_reads, options_.bounded_owner_roots,
        max_bound_bounded_owner_attention_);
  }
  std::string slots_report() const override { return SlotsReport(settings_); }
  std::string KeptLayout() const override { return runner_.CheckpointLayoutId(); }
  std::vector<std::uint64_t> KeptRegions() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).state(), true);
  }
  std::vector<std::uint64_t> KeptLayouts() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).state(), false);
  }
  Status PrepareDefaultRequest() override {
    std::array<Branch*, 1> selected{&default_branch()};
    return SelectBranches(selected);
  }
  Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > options_.slots) return Error("too many Gemma conversation owners");
    std::array<std::uint32_t, engine::kMaxRequestSlots> ids{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (!active[i] || &active[i]->model() != this) return Error("foreign Gemma conversation");
      ids[i] = BranchIndex(*active[i]);
    }
    return runner_.SelectSlots(std::span(ids).first(active.size()));
  }
  std::span<const std::int32_t> ChatStops() const override { return chat_stops_; }
  bool supports_generation_waves() const override { return true; }
  std::size_t generation_wave_capacity() const override { return options_.slots; }
  std::uint64_t StateBytesThrough(std::uint32_t positions) const override {
    const auto ranges = runner_.CheckpointRanges(positions);
    if (!ranges || !slots_[0]) return 0;
    return slots_[0]->state().UsedBytesOf(*ranges).value_or(0);
  }
  void Defaults(chat::Conversation& c) const override {
    if (!c.enable_thinking) c.enable_thinking = false;
  }

 protected:
  Status CheckConversation(const chat::Conversation& c) const override {
    if (c.enable_thinking.value_or(false))
      return Error("Gemma thought-channel output is not qualified in this serving slice");
    if (!c.tools.empty())
      return Error("Gemma generated tool-call output is not qualified in this serving slice");
    return {};
  }
  Status PrepareSamplingScratchFor(Branch& branch, std::size_t logits) override {
    const auto vocab = profile_.vocab;
    if (logits != vocab) return Error("Gemma sampling needs its complete target vocabulary");
    // TopK may reserve 2*k. Reserve the complete bound from empty storage
    // before the first sample, so subsequent rows never reallocate it.
    ReserveBranchSamplingScratch(branch, 2 * std::size_t{vocab});
    return {};
  }
  void RetireSamplingScratchFor(Branch& branch) override { DropBranchSamplingScratch(branch); }
  bool GenerationCohortUsable() const override { return runner_.cohort_usable(); }
  Status RunPreparedGenerationWave(std::span<PreparedGeneration> units) override {
    // Greedy waves take their tokens from the device: no 1 MiB row back, no
    // host scan. A wave is all one kind (the runner publishes one or the other).
    const bool greedy =
        !units.empty() && std::ranges::all_of(units, [](const auto& u) { return DeviceGreedy(u); });
    if (greedy)
      for (auto& unit : units) unit.chosen = 0;
    if (!joined_ || units.size() == 1) {
      if (!greedy) return RunScalarGenerationUnits(units);
      for (auto& unit : units) {
        unit.result =
            *RunGreedyChunkFor(*unit.branch, unit.step.all, unit.step.position, *unit.chosen);
        if (!unit.result && !GenerationCohortUsable()) return unit.result;
        if (!unit.result) unit.failed_prefix_valid = StateUsableFor(*unit.branch);
      }
      return {};
    }
    if (units.empty() || units.size() > options_.slots)
      return Error("Gemma joined generation exceeds its funded owner envelope");
    std::array<engine::Gemma4Runner::Work, engine::kMaxRequestSlots> work{};
    std::array<bool, engine::kMaxRequestSlots> seen{};
    for (std::size_t i = 0; i < units.size(); ++i) {
      const auto& unit = units[i];
      if (unit.branch == nullptr || &unit.branch->model() != this || unit.step.speculative ||
          unit.step.all.size() != std::size_t{unit.step.position} + 1)
        return Error("Gemma joined generation needs plain owned one-anchor units");
      const auto id = BranchIndex(*unit.branch);
      if (id >= options_.slots || seen[id] ||
          NativeSlot(*unit.branch).completed_positions() != unit.step.position)
        return Error("Gemma joined generation needs distinct current native cursors");
      seen[id] = true;
      work[i] = {id, unit.step.position, unit.step.all.last(1), greedy ? nullptr : &units[i].row,
                 greedy ? &*units[i].chosen : nullptr};
    }
    // Checked one-row sums keep their eight-column limit. Ordinary products
    // execute the complete admitted cohort without duplicate model passes.
    return RunGemmaGroups(
        units.size(), engine::Gemma4WaveRows(options_.row_invariant),
        [&](std::size_t first, std::size_t count) {
          auto ran = runner_.Wave(std::span(work).subspan(first, count));
          if (ran) {
            ++joined_groups_;
            joined_units_ += count;
            max_bound_bounded_owner_attention_ =
                std::max(max_bound_bounded_owner_attention_,
                         runner_.last_built_policy().bounded_owner_steps);
          }
          return ran;
        },
        [&] { return GenerationCohortUsable(); },
        [&](std::size_t first, std::size_t count, const std::string& error) {
          for (auto& unit : units.subspan(first, count)) {
            unit.result = std::unexpected(error);
            unit.failed_prefix_valid = StateUsableFor(*unit.branch);
          }
        });
  }
  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                     bool inject, std::vector<float>& logits) override {
    if (inject || past >= all.size()) return Error("Gemma needs a plain nonempty chunk");
    const engine::Gemma4Runner::Work work{BranchIndex(branch), past, all.subspan(past), &logits};
    return runner_.Wave(std::span(&work, 1));
  }
  std::optional<Status> RunGreedyChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                          std::uint32_t past, std::int32_t& token) override {
    if (past >= all.size()) return Error("Gemma needs a plain nonempty chunk");
    const engine::Gemma4Runner::Work work{BranchIndex(branch), past, all.subspan(past), nullptr,
                                          &token};
    return runner_.Wave(std::span(&work, 1));
  }
  Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                            bool inject, bool want_head, std::vector<float>& logits,
                            PrefillHint next) override {
    if (inject || past >= all.size()) return Error("Gemma needs a plain nonempty prefill chunk");
    const engine::Gemma4Runner::Work work{BranchIndex(branch), past, all.subspan(past), &logits};
    const engine::Gemma4Runner::PrefillNext hint{work.slot, next.rows};
    return runner_.WavePrefill(std::span(&work, 1), want_head,
                               std::span(&hint, next.rows == 0 ? 0U : 1U), next.want_head);
  }
  Status SettleFor(Branch& branch) override {
    return NativeSlot(branch).state_usable() ? Status{} : Error("Gemma state is quarantined");
  }
  Status ClearStateFor(Branch& branch) override { return runner_.Clear(BranchIndex(branch)); }
  bool StateUsableFor(const Branch& branch) const override {
    return NativeSlot(branch).state_usable();
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return NativeSlot(branch).refused_state_growth();
  }
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t) override {
    if (pos != NativeSlot(branch).completed_positions() || pos >= context_)
      return Error("Gemma decode position differs from its completed state");
    return runner_.ReserveStateThrough(BranchIndex(branch), pos + 1);
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).state().base(0);
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).state().bytes(0);
  }
  std::uint64_t DrafterStateBaseFor(const Branch&) const override { return 0; }
  std::uint64_t DrafterStateBytesFor(const Branch&) const override { return 0; }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return NativeSlot(branch).state().used_ranges();
  }
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override {
    return runner_.CopyState(BranchIndex(branch), host, ranges, true);
  }
  Status RestoreUsedStateFor(Branch&, void*, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned logical-position metadata");
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t pos) const override {
    if (pos != NativeSlot(branch).completed_positions())
      return Error("Gemma checkpoint boundary differs from completed state");
    return runner_.CheckpointRanges(pos);
  }
  Status PrepareRestoreStateFor(Branch&, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned logical-position metadata");
  }
  Status CheckCheckpointMetadataFor(
      const Branch&, std::uint32_t pos, std::uint32_t cursor,
      std::span<const engine::LiveState::Range> ranges) const override {
    if (pos != cursor) return Error("Gemma checkpoint cursor differs from its logical boundary");
    return runner_.ValidateFootprint(pos, ranges);
  }
  Status PreparePositionRestoreFor(Branch& branch, std::uint32_t pos, std::uint32_t cursor,
                                   std::span<const engine::LiveState::Range> ranges) override {
    if (auto r = CheckCheckpointMetadataFor(branch, pos, cursor, ranges); !r) return r;
    // Llm checks the retained record's layout against KeptLayout before
    // adoption; local snapshots are produced by this same runner.
    return runner_.PrepareRestore(BranchIndex(branch), pos, ranges, KeptLayout());
  }
  Status CompletePositionRestoreFor(Branch& branch, std::uint32_t pos) override {
    return runner_.CompleteRestore(BranchIndex(branch), pos);
  }
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override {
    return runner_.CopyState(BranchIndex(branch), host, ranges, to_host);
  }
  Status RestoreSnapshotFor(Branch& branch, void* host,
                            std::span<const engine::LiveState::Range> ranges,
                            std::uint32_t pos) override {
    if (auto r = PreparePositionRestoreFor(branch, pos, pos, ranges); !r) return r;
    if (auto r = CopyCheckpointStateFor(branch, host, ranges, false); !r) return r;
    return CompletePositionRestoreFor(branch, pos);
  }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }
  std::uint32_t CursorFor(const Branch& branch) const override {
    return NativeSlot(branch).completed_positions();
  }
  void SetCursorFor(Branch&, std::uint32_t) override {}  // checked restore/adopt publishes it
  const engine::LiveState* KeptLiveFor(const Branch& branch) const override {
    return &NativeSlot(branch).state();
  }
  bool KeptWholeFor(const Branch& branch) const override { return NativeSlot(branch).kept_whole(); }
  Status AdoptPositionFor(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t pos,
                          std::span<const engine::LiveState::Range> ranges) override {
    if (pos != tokens.size() || pos == 0)
      return Error("Gemma kept positions differ from its owned token history");
    return runner_.Adopt(BranchIndex(branch), pos, ranges, KeptLayout());
  }
  void SetSpillPlaces(
      const std::function<engine::LiveState::SpillPlace(std::uint32_t)>& place) override {
    runner_.SetSpillPlaces(place);
  }
  Status ReleaseIdleStateFor(Branch& branch) override {
    return runner_.ClearIdle(BranchIndex(branch));
  }
  Status SpillFor(Branch& branch) override { return runner_.Spill(BranchIndex(branch)); }
  Status RestoreFor(Branch& branch) override { return runner_.Restore(BranchIndex(branch)); }
  bool SpilledFor(const Branch& branch) const override { return NativeSlot(branch).is_spilled(); }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spilled_bytes();
  }
  bool LeasedFor(const Branch& branch) const override { return runner_.Held(BranchIndex(branch)); }
  std::uint64_t RefusedBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).refused_bytes();
  }
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t past, bool inject,
                  std::vector<float>& logits) override {
    return RunChunkFor(default_branch(), all, past, inject, logits);
  }
  Status SpecStep(std::span<const std::int32_t>, std::uint32_t, std::uint32_t,
                  std::vector<std::int32_t>&, std::vector<std::vector<float>>*,
                  std::uint64_t&) override {
    return Error("Gemma speculation is unavailable");
  }
  Status Settle() override { return SettleFor(default_branch()); }
  Status ClearState() override { return ClearStateFor(default_branch()); }
  bool StateUsable() const override { return StateUsableFor(default_branch()); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    return PrepareDecodeStateFor(default_branch(), pos, left);
  }
  std::uint64_t target_state_base() const override { return TargetStateBaseFor(default_branch()); }
  std::uint64_t target_state_bytes() const override {
    return TargetStateBytesFor(default_branch());
  }
  std::uint64_t drafter_state_base() const override { return 0; }
  std::uint64_t drafter_state_bytes() const override { return 0; }
  std::uint64_t used_state_bytes() const override { return UsedStateBytesFor(default_branch()); }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return UsedStateRangesFor(default_branch());
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return SaveUsedStateFor(default_branch(), host, ranges);
  }
  Status RestoreUsedState(void*, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned metadata");
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t pos) const override {
    return runner_.CheckpointRanges(pos);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned metadata");
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return CopyCheckpointStateFor(default_branch(), host, ranges, to_host);
  }

 private:
  static engine::Gemma4Options Options(const config::ModelEntry& entry,
                                       const ModelSettings& settings,
                                       const config::RuntimeRoles& roles,
                                       engine::Gemma4Variant variant, bool row_invariant) {
    const bool candidate = variant == engine::Gemma4Variant::k31B && settings.gemma31_production;
    const bool candidate26 =
        variant == engine::Gemma4Variant::k26BA4B && settings.gemma26_production;
    const bool bounded = (candidate || candidate26) && settings.context.value <= 4096 &&
                         settings.max_slots.value == 2;
    return {.artifact = roles.installed / entry.artifact.value_or(""),
            .out = roles.spill,
            .variant = variant,
            .context = settings.context.value,
            .max_rows = settings.prefill_chunk.value,
            .slots = settings.max_slots.value,
            .max_head_rows = settings.max_slots.value,
            // Stock Gemma4 gathers after the full final FFN. Narrowing earlier
            // changes quantized product arithmetic even for one published head.
            .frontier_head = !(candidate || candidate26),
            .row_invariant = row_invariant,
            .fuse_norm_rope = candidate || candidate26,
            .fuse_norm_add = candidate || candidate26,
            .fuse_gemma_route = candidate26,
            .fuse_gemma_reduce = candidate26,
            .fuse_quant_glu = candidate,
            .owner_attention = candidate || candidate26,
            .common_owner_reads = bounded,
            .bounded_owner_roots = bounded};
  }
  engine::Gemma4Runner::Slot& NativeSlot(const Branch& branch) const {
    return *slots_[BranchIndex(branch)];
  }
  std::vector<std::int32_t> chat_stops_;
  config::ModelEntry entry_;
  std::string artifact_id_;
  fs::path store_;
  const model::Gemma4Profile& profile_;
  engine::Gemma4Options options_;
  const bool joined_;
  std::uint64_t joined_groups_ = 0;
  std::uint64_t joined_units_ = 0;
  std::uint32_t max_bound_bounded_owner_attention_ = 0;
  engine::Gemma4Runner runner_;
  std::array<engine::Gemma4Runner::Slot*, engine::kMaxRequestSlots> slots_{};
};

class Gemma2 final : public Llm {
 public:
  Gemma2(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
         const config::RuntimeRoles& roles, int index)
      : entry_(entry),
        artifact_id_(entry.artifact.value_or("")),
        store_(roles.installed),
        profile_(model::Gemma2_2B()),
        options_(Options(entry, settings, roles)),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    node_ = &node;
    context_ = options_.context;
    max_rows_ = options_.max_rows;
    configured_rows_ = settings.prefill_chunk.value;
    checkpoint_directory_ = roles.spill;
  }
  engine::PagedModel& paged() override { return runner_; }
  Status Setup() override {
    if (entry_.drafter || settings_.speculation.value)
      return Error("Gemma serving has no qualified assistant or speculative path");
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) return Error(artifact.error());
    auto binding = model::BindApprovedGemma2(*artifact);
    if (!binding) return Error(binding.error());
    if (auto r = UseChatAssets(*artifact, entry_); !r) return r;
    if (auto r = runner_.Setup(); !r) return r;
    for (std::uint32_t i = 0; i < options_.slots; ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) return Error(slot.error());
      slots_[i] = *slot;
    }
    return PrepareBranches(options_.slots, 1);
  }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override {
    // The startup guard sets this bounded heap workspace apart before any
    // scalar output or per-owner sampling vector grows. One retained frontier
    // and one prepared result per owner, plus the sampling candidate capacity.
    // The independently catalog-backed pinned output holds one row per slot.
    const auto heap = std::uint64_t{options_.slots} * profile_.vocab *
                      (2 * sizeof(float) + 2 * sizeof(execution::SamplingCandidate));
    return runner_.host_input_bytes() + heap;
  }
  std::uint64_t plan_floor_bytes() const override { return runner_.plan_floor_bytes(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  std::vector<catalog::ExtentId> kept_state() const override { return runner_.kept_state(); }
  void StateWrittenBack(bool whole) override { runner_.StateWrittenBack(whole); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  const catalog::Closure& request_closure() const override { return runner_.closure(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  GraphCounts graphs() const override {
    const auto& g = runner_.graph_stats();
    return {g.eager, g.captured, g.replayed, g.refused, runner_.graph_count()};
  }
  std::string violations() const override { return runner_.coverage().first_violation; }
  std::string extra() const override {
    const auto& selected = runner_.plan_selections();
    return std::format(
        R"({{"architecture":"gemma2","recipe":"bounded-8192-two-owner","max_rows":{},"max_wave_rows":{},"joined_prefill_groups":{},"joined_prefill_rows":{},"joined_groups":{},"joined_units":{},"bound_owner_attention":{},"bound_packed_prefill_attention":{},"bound_bounded_owner_attention":{},"attention_softcap":50,"bound_norm_mul":{},"bound_quant_geglu":{},"bound_norm_rope":{},"bound_norm_add":{},"gpu_greedy_tokens":{}}})",
        options_.max_rows, options_.max_wave_rows, joined_prefill_groups_, joined_prefill_rows_,
        joined_groups_, joined_units_, selected.owner_attention, selected.packed_prefill_attention,
        selected.bounded_owner_attention, selected.norm_mul, selected.quant_geglu,
        selected.norm_rope, selected.norm_add, runner_.greedy_tokens());
  }
  std::string slots_report() const override { return SlotsReport(settings_); }
  std::string KeptLayout() const override { return runner_.CheckpointLayoutId(); }
  std::vector<std::uint64_t> KeptRegions() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).state(), true);
  }
  std::vector<std::uint64_t> KeptLayouts() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).state(), false);
  }
  Status PrepareDefaultRequest() override {
    std::array<Branch*, 1> selected{&default_branch()};
    return SelectBranches(selected);
  }
  Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > options_.slots) return Error("too many Gemma conversation owners");
    std::array<std::uint32_t, engine::kMaxRequestSlots> ids{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (!active[i] || &active[i]->model() != this) return Error("foreign Gemma conversation");
      ids[i] = BranchIndex(*active[i]);
    }
    return runner_.SelectSlots(std::span(ids).first(active.size()));
  }
  std::span<const std::int32_t> ChatStops() const override { return chat_stops_; }
  bool supports_generation_waves() const override { return true; }
  std::size_t generation_wave_capacity() const override { return options_.slots; }
  std::size_t prefill_wave_capacity() const override { return options_.slots; }
  bool CompatiblePrefill(std::uint32_t past, std::uint32_t rows, std::uint32_t peer_past,
                         std::uint32_t peer_rows) const override {
    if (rows != peer_rows || rows < 2 || past > context_ || rows > context_ - past ||
        peer_past > context_ || rows > context_ - peer_past)
      return false;
    const auto read = [](std::uint32_t end, std::uint32_t capacity) {
      return std::min(capacity, (end + 255U) / 256U * 256U);
    };
    const auto& layout = runner_.layout();
    return read(past + rows, layout.global_cells) == read(peer_past + rows, layout.global_cells) &&
           read(past + rows, layout.local_cells) == read(peer_past + rows, layout.local_cells);
  }
  std::uint64_t StateBytesThrough(std::uint32_t positions) const override {
    const auto ranges = runner_.CheckpointRanges(positions);
    if (!ranges || !slots_[0]) return 0;
    return slots_[0]->state().UsedBytesOf(*ranges).value_or(0);
  }
  void Defaults(chat::Conversation& c) const override {
    if (!c.enable_thinking) c.enable_thinking = false;
  }

 protected:
  Status CheckConversation(const chat::Conversation& c) const override {
    if (c.enable_thinking.value_or(false))
      return Error("Gemma thought-channel output is not qualified in this serving slice");
    if (!c.tools.empty())
      return Error("Gemma generated tool-call output is not qualified in this serving slice");
    return {};
  }
  Status PrepareSamplingScratchFor(Branch& branch, std::size_t logits) override {
    const auto vocab = profile_.vocab;
    if (logits != vocab) return Error("Gemma sampling needs its complete target vocabulary");
    // TopK may reserve 2*k. Reserve the complete bound from empty storage
    // before the first sample, so subsequent rows never reallocate it.
    ReserveBranchSamplingScratch(branch, 2 * std::size_t{vocab});
    return {};
  }
  void RetireSamplingScratchFor(Branch& branch) override { DropBranchSamplingScratch(branch); }
  bool GenerationCohortUsable() const override { return runner_.cohort_usable(); }
  Status RunPreparedGenerationWave(std::span<PreparedGeneration> units) override {
    // Greedy waves take their tokens from the device: no 1 MiB row back, no
    // host scan. A wave is all one kind (the runner publishes one or the other).
    const bool greedy =
        !units.empty() && std::ranges::all_of(units, [](const auto& u) { return DeviceGreedy(u); });
    if (greedy)
      for (auto& unit : units) unit.chosen = 0;
    if (units.size() == 1) {
      if (!greedy) return RunScalarGenerationUnits(units);
      for (auto& unit : units) {
        unit.result =
            *RunGreedyChunkFor(*unit.branch, unit.step.all, unit.step.position, *unit.chosen);
        if (!unit.result && !GenerationCohortUsable()) return unit.result;
        if (!unit.result) unit.failed_prefix_valid = StateUsableFor(*unit.branch);
      }
      return {};
    }
    if (units.empty() || units.size() > options_.slots)
      return Error("Gemma joined generation exceeds its funded owner envelope");
    std::array<engine::Gemma2Runner::Work, engine::kMaxRequestSlots> work{};
    std::array<bool, engine::kMaxRequestSlots> seen{};
    for (std::size_t i = 0; i < units.size(); ++i) {
      const auto& unit = units[i];
      if (unit.branch == nullptr || &unit.branch->model() != this || unit.step.speculative ||
          unit.step.all.size() != std::size_t{unit.step.position} + 1)
        return Error("Gemma joined generation needs plain owned one-anchor units");
      const auto id = BranchIndex(*unit.branch);
      if (id >= options_.slots || seen[id] ||
          NativeSlot(*unit.branch).completed_positions() != unit.step.position)
        return Error("Gemma joined generation needs distinct current native cursors");
      seen[id] = true;
      work[i] = {id, unit.step.position, unit.step.all.last(1), greedy ? nullptr : &units[i].row,
                 greedy ? &*units[i].chosen : nullptr};
    }
    auto ran = runner_.Wave(std::span(work).first(units.size()));
    if (ran) {
      ++joined_groups_;
      joined_units_ += units.size();
    } else {
      for (auto& unit : units) {
        unit.result = std::unexpected(ran.error());
        unit.failed_prefix_valid = StateUsableFor(*unit.branch);
      }
      if (!GenerationCohortUsable()) return ran;
    }
    return {};
  }
  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                     bool inject, std::vector<float>& logits) override {
    if (inject || past >= all.size()) return Error("Gemma needs a plain nonempty chunk");
    const engine::Gemma2Runner::Work work{BranchIndex(branch), past, all.subspan(past), &logits};
    return runner_.Wave(std::span(&work, 1));
  }
  std::optional<Status> RunGreedyChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                          std::uint32_t past, std::int32_t& token) override {
    if (past >= all.size()) return Error("Gemma needs a plain nonempty chunk");
    const engine::Gemma2Runner::Work work{BranchIndex(branch), past, all.subspan(past), nullptr,
                                          &token};
    return runner_.Wave(std::span(&work, 1));
  }
  Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                            bool inject, bool want_head, std::vector<float>& logits,
                            PrefillHint next) override {
    if (inject || past >= all.size()) return Error("Gemma needs a plain nonempty prefill chunk");
    const engine::Gemma2Runner::Work work{BranchIndex(branch), past, all.subspan(past), &logits};
    (void)next;
    return runner_.WavePrefill(std::span(&work, 1), want_head);
  }
  Status PreparePrefillStateFor(Branch& branch, std::uint32_t past, std::uint32_t rows) override {
    if (past != NativeSlot(branch).completed_positions() || rows == 0 || rows > options_.max_rows ||
        past > context_ || rows > context_ - past)
      return Error("Gemma prefill funding needs an owned bounded continuation");
    return runner_.ReserveStateThrough(BranchIndex(branch), past + rows);
  }
  Status RunPreparedPrefillWave(std::span<PreparedPrefill> prepared) override {
    if (prepared.empty() || prepared.size() > options_.slots)
      return Error("Gemma prefill exceeds its funded owner envelope");
    if (prepared.size() == 1) return Llm::RunPreparedPrefillWave(prepared);
    const bool want_head = prepared.front().want_head;
    std::array<engine::Gemma2Runner::Work, engine::kMaxRequestSlots> work{};
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const auto& unit = prepared[i];
      if (!unit.branch || &unit.branch->model() != this || unit.want_head != want_head ||
          unit.all.size() != std::size_t{unit.past} + unit.rows ||
          !CompatiblePrefill(prepared.front().past, prepared.front().rows, unit.past, unit.rows))
        return Error("Gemma prefill needs compatible owned chunks");
      work[i] = {BranchIndex(*unit.branch), unit.past, unit.all.last(unit.rows), unit.logits};
    }
    auto ran = runner_.WavePrefill(std::span(work).first(prepared.size()), want_head);
    if (ran && prepared.size() > 1) {
      ++joined_prefill_groups_;
      for (const auto& unit : prepared) joined_prefill_rows_ += unit.rows;
    }
    if (!ran) {
      for (auto& unit : prepared) unit.result = Error(ran.error());
      if (!GenerationCohortUsable()) return ran;
    }
    return {};
  }
  Status SettleFor(Branch& branch) override {
    return NativeSlot(branch).state_usable() ? Status{} : Error("Gemma state is quarantined");
  }
  Status ClearStateFor(Branch& branch) override { return runner_.Clear(BranchIndex(branch)); }
  bool StateUsableFor(const Branch& branch) const override {
    return NativeSlot(branch).state_usable();
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return NativeSlot(branch).refused_state_growth();
  }
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t) override {
    if (pos != NativeSlot(branch).completed_positions() || pos >= context_)
      return Error("Gemma decode position differs from its completed state");
    return runner_.ReserveStateThrough(BranchIndex(branch), pos + 1);
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).state().base(0);
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).state().bytes(0);
  }
  std::uint64_t DrafterStateBaseFor(const Branch&) const override { return 0; }
  std::uint64_t DrafterStateBytesFor(const Branch&) const override { return 0; }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return NativeSlot(branch).state().used_ranges();
  }
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override {
    return runner_.CopyState(BranchIndex(branch), host, ranges, true);
  }
  Status RestoreUsedStateFor(Branch&, void*, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned logical-position metadata");
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t pos) const override {
    if (pos != NativeSlot(branch).completed_positions())
      return Error("Gemma checkpoint boundary differs from completed state");
    return runner_.CheckpointRanges(pos);
  }
  Status PrepareRestoreStateFor(Branch&, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned logical-position metadata");
  }
  Status CheckCheckpointMetadataFor(
      const Branch&, std::uint32_t pos, std::uint32_t cursor,
      std::span<const engine::LiveState::Range> ranges) const override {
    if (pos != cursor) return Error("Gemma checkpoint cursor differs from its logical boundary");
    return runner_.ValidateFootprint(pos, ranges);
  }
  Status PreparePositionRestoreFor(Branch& branch, std::uint32_t pos, std::uint32_t cursor,
                                   std::span<const engine::LiveState::Range> ranges) override {
    if (auto r = CheckCheckpointMetadataFor(branch, pos, cursor, ranges); !r) return r;
    // Llm checks the retained record's layout against KeptLayout before
    // adoption; local snapshots are produced by this same runner.
    return runner_.PrepareRestore(BranchIndex(branch), pos, ranges, KeptLayout());
  }
  Status CompletePositionRestoreFor(Branch& branch, std::uint32_t pos) override {
    return runner_.CompleteRestore(BranchIndex(branch), pos);
  }
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override {
    return runner_.CopyState(BranchIndex(branch), host, ranges, to_host);
  }
  Status RestoreSnapshotFor(Branch& branch, void* host,
                            std::span<const engine::LiveState::Range> ranges,
                            std::uint32_t pos) override {
    if (auto r = PreparePositionRestoreFor(branch, pos, pos, ranges); !r) return r;
    if (auto r = CopyCheckpointStateFor(branch, host, ranges, false); !r) return r;
    return CompletePositionRestoreFor(branch, pos);
  }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }
  std::uint32_t CursorFor(const Branch& branch) const override {
    return NativeSlot(branch).completed_positions();
  }
  void SetCursorFor(Branch&, std::uint32_t) override {}  // checked restore/adopt publishes it
  const engine::LiveState* KeptLiveFor(const Branch& branch) const override {
    return &NativeSlot(branch).state();
  }
  bool KeptWholeFor(const Branch& branch) const override { return NativeSlot(branch).kept_whole(); }
  Status AdoptPositionFor(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t pos,
                          std::span<const engine::LiveState::Range> ranges) override {
    if (pos != tokens.size() || pos == 0)
      return Error("Gemma kept positions differ from its owned token history");
    return runner_.Adopt(BranchIndex(branch), pos, ranges, KeptLayout());
  }
  void SetSpillPlaces(
      const std::function<engine::LiveState::SpillPlace(std::uint32_t)>& place) override {
    runner_.SetSpillPlaces(place);
  }
  Status ReleaseIdleStateFor(Branch& branch) override {
    return runner_.ClearIdle(BranchIndex(branch));
  }
  Status SpillFor(Branch& branch) override { return runner_.Spill(BranchIndex(branch)); }
  Status RestoreFor(Branch& branch) override { return runner_.Restore(BranchIndex(branch)); }
  bool SpilledFor(const Branch& branch) const override { return NativeSlot(branch).is_spilled(); }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spilled_bytes();
  }
  bool LeasedFor(const Branch& branch) const override { return runner_.Held(BranchIndex(branch)); }
  std::uint64_t RefusedBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).refused_bytes();
  }
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t past, bool inject,
                  std::vector<float>& logits) override {
    return RunChunkFor(default_branch(), all, past, inject, logits);
  }
  Status SpecStep(std::span<const std::int32_t>, std::uint32_t, std::uint32_t,
                  std::vector<std::int32_t>&, std::vector<std::vector<float>>*,
                  std::uint64_t&) override {
    return Error("Gemma speculation is unavailable");
  }
  Status Settle() override { return SettleFor(default_branch()); }
  Status ClearState() override { return ClearStateFor(default_branch()); }
  bool StateUsable() const override { return StateUsableFor(default_branch()); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    return PrepareDecodeStateFor(default_branch(), pos, left);
  }
  std::uint64_t target_state_base() const override { return TargetStateBaseFor(default_branch()); }
  std::uint64_t target_state_bytes() const override {
    return TargetStateBytesFor(default_branch());
  }
  std::uint64_t drafter_state_base() const override { return 0; }
  std::uint64_t drafter_state_bytes() const override { return 0; }
  std::uint64_t used_state_bytes() const override { return UsedStateBytesFor(default_branch()); }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return UsedStateRangesFor(default_branch());
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return SaveUsedStateFor(default_branch(), host, ranges);
  }
  Status RestoreUsedState(void*, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned metadata");
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t pos) const override {
    return runner_.CheckpointRanges(pos);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned metadata");
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return CopyCheckpointStateFor(default_branch(), host, ranges, to_host);
  }

 private:
  static engine::Gemma2Options Options(const config::ModelEntry& entry,
                                       const ModelSettings& settings,
                                       const config::RuntimeRoles& roles) {
    return {.artifact = roles.installed / entry.artifact.value_or(""),
            .out = roles.spill,
            .context = settings.context.value,
            .max_rows = settings.prefill_chunk.value,
            .slots = settings.max_slots.value,
            .max_wave_rows = settings.prefill_chunk.value * settings.max_slots.value,
            .max_head_rows = settings.max_slots.value,
            .owner_decode = true,
            .packed_prefill = true,
            .bounded_roots = true,
            .fuse_norms = true,
            .fuse_quant_glu = true,
            .fuse_norm_rope = false,
            .fuse_norm_add = true};
  }
  engine::Gemma2Runner::Slot& NativeSlot(const Branch& branch) const {
    return *slots_[BranchIndex(branch)];
  }
  std::vector<std::int32_t> chat_stops_;
  config::ModelEntry entry_;
  std::string artifact_id_;
  fs::path store_;
  const model::Gemma2Profile& profile_;
  engine::Gemma2Options options_;
  std::uint64_t joined_prefill_groups_ = 0, joined_prefill_rows_ = 0;
  std::uint64_t joined_groups_ = 0;
  std::uint64_t joined_units_ = 0;
  engine::Gemma2Runner runner_;
  std::array<engine::Gemma2Runner::Slot*, engine::kMaxRequestSlots> slots_{};
};

class Gemma3 final : public Llm {
 public:
  Gemma3(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
         const config::RuntimeRoles& roles, int index, const ServingOptions& serving)
      : entry_(entry),
        artifact_id_(entry.artifact.value_or("")),
        store_(roles.installed),
        profile_(model::Gemma3_4BQat()),
        options_(Options(entry, settings, roles, serving)),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    node_ = &node;
    context_ = options_.context;
    max_rows_ = options_.max_rows;
    configured_rows_ = settings.prefill_chunk.value;
    checkpoint_directory_ = roles.spill;
  }
  engine::PagedModel& paged() override { return runner_; }
  Status Setup() override {
    if (entry_.drafter || settings_.speculation.value)
      return Error("Gemma serving has no qualified assistant or speculative path");
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) return Error(artifact.error());
    auto binding = model::BindApprovedGemma3(*artifact);
    if (!binding) return Error(binding.error());
    if (auto r = UseChatAssets(*artifact, entry_); !r) return r;
    if (auto r = runner_.Setup(); !r) return r;
    for (std::uint32_t i = 0; i < options_.slots; ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) return Error(slot.error());
      slots_[i] = *slot;
    }
    return PrepareBranches(options_.slots, 1);
  }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override {
    // The startup guard sets this bounded heap workspace apart before any
    // scalar output or per-owner sampling vector grows. One retained frontier
    // and one prepared result per owner, plus the sampling candidate capacity.
    // The independently catalog-backed pinned output holds one row per slot.
    const auto heap = std::uint64_t{options_.slots} * profile_.vocab *
                      (2 * sizeof(float) + 2 * sizeof(execution::SamplingCandidate));
    return runner_.host_input_bytes() + heap;
  }
  std::uint64_t plan_floor_bytes() const override { return runner_.plan_floor_bytes(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  std::vector<catalog::ExtentId> kept_state() const override { return runner_.kept_state(); }
  void StateWrittenBack(bool whole) override { runner_.StateWrittenBack(whole); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  const catalog::Closure& request_closure() const override { return runner_.closure(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  GraphCounts graphs() const override {
    const auto& g = runner_.graph_stats();
    return {g.eager, g.captured, g.replayed, g.refused, runner_.graph_count()};
  }
  std::string violations() const override { return runner_.coverage().first_violation; }
  std::string extra() const override {
    const auto& selected = runner_.plan_selections();
    const auto& ahead = runner_.lookahead_stats();
    return std::format(
        R"({{"architecture":"gemma3","recipe":"bounded-serving","context":{},"configured_slots":{},"max_rows":{},"max_wave_rows":{},"joined_prefill_groups":{},"joined_prefill_rows":{},"joined_groups":{},"joined_units":{},"bound_owner_attention":{},"bound_packed_prefill_attention":{},"bound_bounded_owner_attention":{},"device_masks":{},"bound_device_masks":{},"bound_norm_rope":{},"bound_norm_add":{},"gpu_greedy_tokens":{},"prefill_lookahead":{},"capture_ahead":{},"lookahead_built":{},"lookahead_cached":{},"lookahead_refused":{},"captured_first":{},"captured_ahead":{},"dropped_ahead":{}}})",
        options_.context, options_.slots, options_.max_rows, options_.max_wave_rows,
        joined_prefill_groups_, joined_prefill_rows_, joined_groups_, joined_units_,
        selected.owner_attention, selected.packed_prefill_attention,
        selected.bounded_owner_attention, options_.device_masks, selected.device_masks,
        selected.norm_rope, selected.norm_add, runner_.greedy_tokens(), options_.prefill_lookahead,
        options_.capture_ahead, ahead.built, ahead.cached, ahead.refused, ahead.captured_first,
        ahead.captured_ahead, ahead.dropped_ahead);
  }
  std::string slots_report() const override { return SlotsReport(settings_); }
  std::string KeptLayout() const override { return runner_.CheckpointLayoutId(); }
  std::vector<std::uint64_t> KeptRegions() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).state(), true);
  }
  std::vector<std::uint64_t> KeptLayouts() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).state(), false);
  }
  Status PrepareDefaultRequest() override {
    std::array<Branch*, 1> selected{&default_branch()};
    return SelectBranches(selected);
  }
  Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > options_.slots) return Error("too many Gemma conversation owners");
    std::array<std::uint32_t, engine::kMaxRequestSlots> ids{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (!active[i] || &active[i]->model() != this) return Error("foreign Gemma conversation");
      ids[i] = BranchIndex(*active[i]);
    }
    return runner_.SelectSlots(std::span(ids).first(active.size()));
  }
  std::span<const std::int32_t> ChatStops() const override { return chat_stops_; }
  bool supports_generation_waves() const override { return true; }
  std::size_t generation_wave_capacity() const override { return options_.slots; }
  std::size_t prefill_wave_capacity() const override { return options_.slots; }
  bool CompatiblePrefill(std::uint32_t past, std::uint32_t rows, std::uint32_t peer_past,
                         std::uint32_t peer_rows) const override {
    if (rows != peer_rows || rows < 2 || past > context_ || rows > context_ - past ||
        peer_past > context_ || rows > context_ - peer_past)
      return false;
    const auto read = [](std::uint32_t end, std::uint32_t capacity) {
      return std::min(capacity, (end + 255U) / 256U * 256U);
    };
    const auto& layout = runner_.layout();
    return read(past + rows, layout.global_cells) == read(peer_past + rows, layout.global_cells) &&
           read(past + rows, layout.local_cells) == read(peer_past + rows, layout.local_cells);
  }
  std::uint64_t StateBytesThrough(std::uint32_t positions) const override {
    const auto ranges = runner_.CheckpointRanges(positions);
    if (!ranges || !slots_[0]) return 0;
    return slots_[0]->state().UsedBytesOf(*ranges).value_or(0);
  }
  void Defaults(chat::Conversation& c) const override {
    if (!c.enable_thinking) c.enable_thinking = false;
  }

 protected:
  Status CheckConversation(const chat::Conversation& c) const override {
    if (c.enable_thinking.value_or(false))
      return Error("Gemma thought-channel output is not qualified in this serving slice");
    if (!c.tools.empty())
      return Error("Gemma generated tool-call output is not qualified in this serving slice");
    return {};
  }
  Status PrepareSamplingScratchFor(Branch& branch, std::size_t logits) override {
    const auto vocab = profile_.vocab;
    if (logits != vocab) return Error("Gemma sampling needs its complete target vocabulary");
    // TopK may reserve 2*k. Reserve the complete bound from empty storage
    // before the first sample, so subsequent rows never reallocate it.
    ReserveBranchSamplingScratch(branch, 2 * std::size_t{vocab});
    return {};
  }
  void RetireSamplingScratchFor(Branch& branch) override { DropBranchSamplingScratch(branch); }
  bool GenerationCohortUsable() const override { return runner_.cohort_usable(); }
  Status RunPreparedGenerationWave(std::span<PreparedGeneration> units) override {
    // Greedy waves take their tokens from the device: no 1 MiB row back, no
    // host scan. A wave is all one kind (the runner publishes one or the other).
    const bool greedy =
        !units.empty() && std::ranges::all_of(units, [](const auto& u) { return DeviceGreedy(u); });
    if (greedy)
      for (auto& unit : units) unit.chosen = 0;
    if (units.size() == 1) {
      if (!greedy) return RunScalarGenerationUnits(units);
      for (auto& unit : units) {
        unit.result =
            *RunGreedyChunkFor(*unit.branch, unit.step.all, unit.step.position, *unit.chosen);
        if (!unit.result && !GenerationCohortUsable()) return unit.result;
        if (!unit.result) unit.failed_prefix_valid = StateUsableFor(*unit.branch);
      }
      return {};
    }
    if (units.empty() || units.size() > options_.slots)
      return Error("Gemma joined generation exceeds its funded owner envelope");
    std::array<engine::Gemma3Runner::Work, engine::kMaxRequestSlots> work{};
    std::array<bool, engine::kMaxRequestSlots> seen{};
    for (std::size_t i = 0; i < units.size(); ++i) {
      const auto& unit = units[i];
      if (unit.branch == nullptr || &unit.branch->model() != this || unit.step.speculative ||
          unit.step.all.size() != std::size_t{unit.step.position} + 1)
        return Error("Gemma joined generation needs plain owned one-anchor units");
      const auto id = BranchIndex(*unit.branch);
      if (id >= options_.slots || seen[id] ||
          NativeSlot(*unit.branch).completed_positions() != unit.step.position)
        return Error("Gemma joined generation needs distinct current native cursors");
      seen[id] = true;
      work[i] = {id, unit.step.position, unit.step.all.last(1), greedy ? nullptr : &units[i].row,
                 greedy ? &*units[i].chosen : nullptr};
    }
    auto ran = runner_.Wave(std::span(work).first(units.size()));
    if (ran) {
      ++joined_groups_;
      joined_units_ += units.size();
    } else {
      for (auto& unit : units) {
        unit.result = std::unexpected(ran.error());
        unit.failed_prefix_valid = StateUsableFor(*unit.branch);
      }
      if (!GenerationCohortUsable()) return ran;
    }
    return {};
  }
  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                     bool inject, std::vector<float>& logits) override {
    if (inject || past >= all.size()) return Error("Gemma needs a plain nonempty chunk");
    const engine::Gemma3Runner::Work work{BranchIndex(branch), past, all.subspan(past), &logits};
    return runner_.Wave(std::span(&work, 1));
  }
  std::optional<Status> RunGreedyChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                          std::uint32_t past, std::int32_t& token) override {
    if (past >= all.size()) return Error("Gemma needs a plain nonempty chunk");
    const engine::Gemma3Runner::Work work{BranchIndex(branch), past, all.subspan(past), nullptr,
                                          &token};
    return runner_.Wave(std::span(&work, 1));
  }
  Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                            bool inject, bool want_head, std::vector<float>& logits,
                            PrefillHint next) override {
    if (inject || past >= all.size()) return Error("Gemma needs a plain nonempty prefill chunk");
    const engine::Gemma3Runner::Work work{BranchIndex(branch), past, all.subspan(past), &logits};
    const engine::Gemma3Runner::PrefillNext hint{work.slot, next.rows, next.after_rows};
    return runner_.WavePrefill(std::span(&work, 1), want_head,
                               std::span(&hint, next.rows == 0 ? 0U : 1U), next.want_head,
                               next.after_want_head);
  }
  Status PreparePrefillStateFor(Branch& branch, std::uint32_t past, std::uint32_t rows) override {
    if (past != NativeSlot(branch).completed_positions() || rows == 0 || rows > options_.max_rows ||
        past > context_ || rows > context_ - past)
      return Error("Gemma prefill funding needs an owned bounded continuation");
    return runner_.ReserveStateThrough(BranchIndex(branch), past + rows);
  }
  Status RunPreparedPrefillWave(std::span<PreparedPrefill> prepared) override {
    if (prepared.empty() || prepared.size() > options_.slots)
      return Error("Gemma prefill exceeds its funded owner envelope");
    if (prepared.size() == 1) return Llm::RunPreparedPrefillWave(prepared);
    const bool want_head = prepared.front().want_head;
    std::array<engine::Gemma3Runner::Work, engine::kMaxRequestSlots> work{};
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const auto& unit = prepared[i];
      if (!unit.branch || &unit.branch->model() != this || unit.want_head != want_head ||
          unit.all.size() != std::size_t{unit.past} + unit.rows ||
          !CompatiblePrefill(prepared.front().past, prepared.front().rows, unit.past, unit.rows))
        return Error("Gemma prefill needs compatible owned chunks");
      work[i] = {BranchIndex(*unit.branch), unit.past, unit.all.last(unit.rows), unit.logits};
    }
    auto ran = runner_.WavePrefill(std::span(work).first(prepared.size()), want_head);
    if (ran && prepared.size() > 1) {
      ++joined_prefill_groups_;
      for (const auto& unit : prepared) joined_prefill_rows_ += unit.rows;
    }
    if (!ran) {
      for (auto& unit : prepared) unit.result = Error(ran.error());
      if (!GenerationCohortUsable()) return ran;
    }
    return {};
  }
  Status SettleFor(Branch& branch) override {
    return NativeSlot(branch).state_usable() ? Status{} : Error("Gemma state is quarantined");
  }
  Status ClearStateFor(Branch& branch) override { return runner_.Clear(BranchIndex(branch)); }
  bool StateUsableFor(const Branch& branch) const override {
    return NativeSlot(branch).state_usable();
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return NativeSlot(branch).refused_state_growth();
  }
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t) override {
    if (pos != NativeSlot(branch).completed_positions() || pos >= context_)
      return Error("Gemma decode position differs from its completed state");
    return runner_.ReserveStateThrough(BranchIndex(branch), pos + 1);
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).state().base(0);
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).state().bytes(0);
  }
  std::uint64_t DrafterStateBaseFor(const Branch&) const override { return 0; }
  std::uint64_t DrafterStateBytesFor(const Branch&) const override { return 0; }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return NativeSlot(branch).state().used_ranges();
  }
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override {
    return runner_.CopyState(BranchIndex(branch), host, ranges, true);
  }
  Status RestoreUsedStateFor(Branch&, void*, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned logical-position metadata");
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t pos) const override {
    if (pos != NativeSlot(branch).completed_positions())
      return Error("Gemma checkpoint boundary differs from completed state");
    return runner_.CheckpointRanges(pos);
  }
  Status PrepareRestoreStateFor(Branch&, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned logical-position metadata");
  }
  Status CheckCheckpointMetadataFor(
      const Branch&, std::uint32_t pos, std::uint32_t cursor,
      std::span<const engine::LiveState::Range> ranges) const override {
    if (pos != cursor) return Error("Gemma checkpoint cursor differs from its logical boundary");
    return runner_.ValidateFootprint(pos, ranges);
  }
  Status PreparePositionRestoreFor(Branch& branch, std::uint32_t pos, std::uint32_t cursor,
                                   std::span<const engine::LiveState::Range> ranges) override {
    if (auto r = CheckCheckpointMetadataFor(branch, pos, cursor, ranges); !r) return r;
    // Llm checks the retained record's layout against KeptLayout before
    // adoption; local snapshots are produced by this same runner.
    return runner_.PrepareRestore(BranchIndex(branch), pos, ranges, KeptLayout());
  }
  Status CompletePositionRestoreFor(Branch& branch, std::uint32_t pos) override {
    return runner_.CompleteRestore(BranchIndex(branch), pos);
  }
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override {
    return runner_.CopyState(BranchIndex(branch), host, ranges, to_host);
  }
  Status RestoreSnapshotFor(Branch& branch, void* host,
                            std::span<const engine::LiveState::Range> ranges,
                            std::uint32_t pos) override {
    if (auto r = PreparePositionRestoreFor(branch, pos, pos, ranges); !r) return r;
    if (auto r = CopyCheckpointStateFor(branch, host, ranges, false); !r) return r;
    return CompletePositionRestoreFor(branch, pos);
  }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }
  std::uint32_t CursorFor(const Branch& branch) const override {
    return NativeSlot(branch).completed_positions();
  }
  void SetCursorFor(Branch&, std::uint32_t) override {}  // checked restore/adopt publishes it
  const engine::LiveState* KeptLiveFor(const Branch& branch) const override {
    return &NativeSlot(branch).state();
  }
  bool KeptWholeFor(const Branch& branch) const override { return NativeSlot(branch).kept_whole(); }
  Status AdoptPositionFor(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t pos,
                          std::span<const engine::LiveState::Range> ranges) override {
    if (pos != tokens.size() || pos == 0)
      return Error("Gemma kept positions differ from its owned token history");
    return runner_.Adopt(BranchIndex(branch), pos, ranges, KeptLayout());
  }
  void SetSpillPlaces(
      const std::function<engine::LiveState::SpillPlace(std::uint32_t)>& place) override {
    runner_.SetSpillPlaces(place);
  }
  Status ReleaseIdleStateFor(Branch& branch) override {
    return runner_.ClearIdle(BranchIndex(branch));
  }
  Status SpillFor(Branch& branch) override { return runner_.Spill(BranchIndex(branch)); }
  Status RestoreFor(Branch& branch) override { return runner_.Restore(BranchIndex(branch)); }
  bool SpilledFor(const Branch& branch) const override { return NativeSlot(branch).is_spilled(); }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spilled_bytes();
  }
  bool LeasedFor(const Branch& branch) const override { return runner_.Held(BranchIndex(branch)); }
  std::uint64_t RefusedBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).refused_bytes();
  }
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t past, bool inject,
                  std::vector<float>& logits) override {
    return RunChunkFor(default_branch(), all, past, inject, logits);
  }
  Status SpecStep(std::span<const std::int32_t>, std::uint32_t, std::uint32_t,
                  std::vector<std::int32_t>&, std::vector<std::vector<float>>*,
                  std::uint64_t&) override {
    return Error("Gemma speculation is unavailable");
  }
  Status Settle() override { return SettleFor(default_branch()); }
  Status ClearState() override { return ClearStateFor(default_branch()); }
  bool StateUsable() const override { return StateUsableFor(default_branch()); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    return PrepareDecodeStateFor(default_branch(), pos, left);
  }
  std::uint64_t target_state_base() const override { return TargetStateBaseFor(default_branch()); }
  std::uint64_t target_state_bytes() const override {
    return TargetStateBytesFor(default_branch());
  }
  std::uint64_t drafter_state_base() const override { return 0; }
  std::uint64_t drafter_state_bytes() const override { return 0; }
  std::uint64_t used_state_bytes() const override { return UsedStateBytesFor(default_branch()); }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return UsedStateRangesFor(default_branch());
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return SaveUsedStateFor(default_branch(), host, ranges);
  }
  Status RestoreUsedState(void*, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned metadata");
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t pos) const override {
    return runner_.CheckpointRanges(pos);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned metadata");
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return CopyCheckpointStateFor(default_branch(), host, ranges, to_host);
  }

 private:
  static engine::Gemma3Options Options(const config::ModelEntry& entry,
                                       const ModelSettings& settings,
                                       const config::RuntimeRoles& roles,
                                       const ServingOptions& serving) {
    return {.artifact = roles.installed / entry.artifact.value_or(""),
            .out = roles.spill,
            .context = settings.context.value,
            .max_rows = settings.prefill_chunk.value,
            .slots = settings.max_slots.value,
            .max_wave_rows = settings.prefill_chunk.value * settings.max_slots.value,
            .max_head_rows = settings.max_slots.value,
            .owner_decode = true,
            .packed_prefill = true,
            .bounded_roots = true,
            .device_masks = serving.gemma3_device_masks,
            .fuse_norms = true,
            .fuse_quant_glu = true,
            .fuse_norm_rope = true,
            .fuse_norm_add = true,
            .prefill_lookahead = serving.gemma3_prefill_lookahead,
            .capture_ahead = serving.gemma3_capture_ahead};
  }
  engine::Gemma3Runner::Slot& NativeSlot(const Branch& branch) const {
    return *slots_[BranchIndex(branch)];
  }
  std::vector<std::int32_t> chat_stops_;
  config::ModelEntry entry_;
  std::string artifact_id_;
  fs::path store_;
  const model::Gemma3Profile& profile_;
  engine::Gemma3Options options_;
  std::uint64_t joined_prefill_groups_ = 0, joined_prefill_rows_ = 0;
  std::uint64_t joined_groups_ = 0;
  std::uint64_t joined_units_ = 0;
  engine::Gemma3Runner runner_;
  std::array<engine::Gemma3Runner::Slot*, engine::kMaxRequestSlots> slots_{};
};

// Qwen3.8 Flash Next (engine/qwen38_runner.h), with its MTP block as its
// drafter.
class Qwen38 final : public Llm {
 public:
  Qwen38(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
         const config::RuntimeRoles& roles, int index)
      : entry_(entry),
        artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    node_ = &node;
    speculate_ = !drafter_id_.empty() && settings.speculation.value;
    checkpoint_directory_ = roles.spill;
    context_ = settings.context.value;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = context_;
    configured_rows_ = settings.prefill_chunk.value;
    // (The runner's fast graph builds no mask of every cell by every row, so
    // RE-037's bound does not cap its chunks.)
    max_rows_ = PrefillChunkRows(context_, std::nullopt, settings.prefill_chunk.value,
                                 model::Qwen38MostRows(context_, false));
    options_.max_rows = max_rows_;
    options_.graphs = true;
    // Request slots share weights and workspace while each branch retains
    // its own native state: the requests decode in one wave
    // (engine/qwen38_wave_plan.h), whose row-local products read each weight
    // once for a group of up to sixteen rows (C2/C4 HTTP screens: +9.8%/+20%
    // over two slots of pairs).
    options_.wave_slots = settings.max_slots.value;
    options_.request_slots = options_.wave_slots;
    slots_report_ = SlotsReport(settings);
    options_.wave_read_align = settings.wave_read_align.value;
    options_.wave_lanes = settings.wave_lanes.value;
    depth_cost_ratio_ = settings.depth_cost_ratio.value;
    shared_wave_depth_ = settings.shared_wave_depth.value;
    draft_wave_max_ = settings.draft_wave_max.value;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
      // A verify takes the anchor and the drafts: a chunk of fewer than
      // four rows holds two drafts at most.
      options_.draft_rows =
          max_rows_ >= 4 ? settings.draft_rows.value : std::min(settings.draft_rows.value, 2U);
      options_.draft_vocab = settings.draft_vocab.value;
    }
  }

  engine::PagedModel& paged() override { return runner_; }

  std::string allocation_report() const override {
    const auto b = runner_.setup_budget();
    const auto workspace = std::format(
        "\"scalar_activation_bytes\":{},\"provisioned_activation_bytes\":{},"
        "\"scalar_scratch_bytes\":{},\"provisioned_scratch_bytes\":{},"
        "\"scalar_input_staging_bytes\":{},\"provisioned_input_staging_bytes\":{},"
        "\"scalar_host_input_bytes\":{},\"provisioned_host_input_bytes\":{}",
        b.scalar_activations, b.activations, b.scalar_scratch, b.scratch, b.scalar_staging_inputs,
        b.staging_inputs, b.scalar_host_inputs, b.host_inputs);
    const auto fixed = std::format(
        "\"scalar_ple_mapped_bytes\":{},\"provisioned_ple_mapped_bytes\":{},"
        "\"scalar_pinned_bytes\":{},\"provisioned_pinned_bytes\":{},"
        "\"wave_output_pinned_bytes\":{},\"scalar_snapshot_per_branch_bytes\":{},"
        "\"provisioned_snapshot_per_branch_bytes\":{},\"runner_mapped_bytes\":{}",
        b.scalar_ple_mapped, b.ple_mapped, b.scalar_pinned, b.pinned, b.wave_output_pinned,
        b.scalar_snapshot_per_branch, b.snapshot_per_branch, b.runner_mapped);
    const auto state = std::format(
        "\"target_layout_per_branch_bytes\":{},\"drafter_layout_per_branch_bytes\":{},"
        "\"virtual_extent_per_branch_bytes\":{},\"initialized_logical_bytes\":{},"
        "\"initialized_extent_bytes\":{}",
        b.target_per_branch, b.drafter_per_branch, b.virtual_per_branch, b.initialized_logical,
        b.initialized_extent_bytes);
    return std::format(
        "{{\"format\":\"jitllm-qwen-wave-setup-budget-v1\",\"context\":{},"
        "\"prefill_rows\":{},\"wave_capacity\":{},\"branch_count\":{},"
        "\"speculative\":{},\"draft_rows\":{},{},{},{}}}",
        context_, max_rows_, runner_.wave_capacity(), branches(), speculate_, options_.draft_rows,
        workspace, fixed, state);
  }

  Status Setup() override {
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) {
      return std::unexpected(artifact.error());
    }
    if (artifact->model().architecture != "qwen4exp") {
      return Error(std::format("artifact {} is a {}, not Qwen3.8 Flash Next (qwen4exp)",
                               artifact_id_, artifact->model().architecture));
    }
    if (speculate_) {
      if (auto drafter = OpenTrusted(store_, drafter_id_); !drafter) {
        return std::unexpected(drafter.error());
      }
    }
    // The tokenizer and template: kept in the artifact's metadata, or (M3's
    // import kept config.json only) the checkpoint's, which the
    // configuration names.
    if (auto used = UseChatAssets(*artifact, entry_); !used) {
      return used;
    }
    if (auto setup = runner_.Setup(); !setup) {
      return setup;
    }
    for (std::size_t i = 0; i < runner_.request_slots(); ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) {
        return std::unexpected(slot.error());
      }
      native_slots_[i] = *slot;
    }
    return PrepareBranches(static_cast<std::uint32_t>(runner_.request_slots()), 3);
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override { return runner_.host_input_bytes(); }
  std::uint64_t plan_floor_bytes() const override { return runner_.plan_floor_bytes(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  std::string plan_report() const override { return runner_.plan_report(); }
  std::string slots_report() const override { return slots_report_; }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  std::vector<catalog::ExtentId> kept_state() const override { return runner_.kept_state(); }
  std::vector<catalog::ExtentId> unchanged_state() const override {
    return runner_.unchanged_state();
  }
  void StateWrittenBack(bool whole) override { runner_.StateWrittenBack(whole); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  const catalog::Closure& request_closure() const override { return runner_.execution_closure(); }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  Status PrepareDefaultRequest() override {
    std::array<Branch*, 1> active{&default_branch()};
    return SelectBranches(active);
  }
  Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > runner_.request_slots()) {
      return Error("too many Qwen native conversation branches");
    }
    std::array<engine::Qwen38Runner::Slot*, engine::Qwen38Runner::kRequestSlots> selected{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (active[i] == nullptr || &active[i]->model() != this) {
        return Error("a selected Qwen conversation belongs to another model");
      }
      selected[i] = &NativeSlot(*active[i]);
    }
    return runner_.SelectSlots(std::span(selected).first(active.size()));
  }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  Status AfterLoad() override { return runner_.ReadPleHash(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  double plan_seconds() const override { return runner_.plan_seconds(); }
  double demand_seconds() const override { return runner_.ple().seconds; }
  std::uint64_t demand_bytes() const override { return runner_.ple().read_bytes; }
  GraphCounts graphs() const override {
    return Sum(runner_.graph_stats(), runner_.draft_stats(), runner_.graphs());
  }
  std::string violations() const override {
    return runner_.coverage_violations() == 0
               ? std::string()
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             runner_.coverage_violations(), runner_.first_violation());
  }
  std::string extra() const override {
    const engine::PleStats& p = runner_.ple();
    return std::format(
        R"({{"turn_checkpoints":{},"turn_checkpoint_bytes":{},"used_state_bytes":{},"architecture":"qwen4exp","speculation":{},"coverage_tensors":{},)"
        R"("slab_padding":{},"state_bytes":{},"drafter_state_bytes":{},"ple_table_bytes":{},)"
        R"("ple":{{"chunks":{},"lookups":{},"rows":{},"reads":{},"read_bytes":{},)"
        R"("seconds":{:.6f}}},"graphs":{}}})",
        turn_checkpoints(), turn_checkpoint_bytes(), runner_.used_state_bytes(),
        speculate_ ? "\"mtp\"" : "null", runner_.coverage_tensors(), runner_.slab_padding(),
        runner_.state_bytes(), runner_.drafter_state_bytes(), runner_.table_bytes(), p.chunks,
        p.lookups, p.rows, p.reads, p.read_bytes, p.seconds, GraphJson(graphs()));
  }
  void Defaults(chat::Conversation& /*c*/) const override {
    // The template's own defaults (thinking on), as the oracle's /tokenize
    // rendered the references' prompts.
  }
  // A speculative step's drafts must stay inside the context (SpecStep).
  std::uint32_t usable_context() const override {
    return speculate_ ? context_ - runner_.draft_rows() : context_;
  }
  bool supports_generation_waves() const override { return runner_.waves_provisioned(); }
  std::size_t generation_wave_capacity() const override { return runner_.wave_capacity(); }
  std::uint64_t StateBytesThrough(std::uint32_t positions) const override {
    return runner_.StateBytesThrough(positions).value_or(0);
  }

 protected:
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                  std::vector<float>& logits) override {
    return RunChunkFor(default_branch(), all, n_past, inject, logits);
  }
  Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                  std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                  std::uint64_t& drafted) override {
    return SpecStepFor(default_branch(), all, pos, left, kept, logits, drafted);
  }
  Status Settle() override { return SettleFor(default_branch()); }
  Status ClearState() override { return ClearStateFor(default_branch()); }
  bool StateUsable() const override { return StateUsableFor(default_branch()); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    return PrepareDecodeStateFor(default_branch(), pos, left);
  }
  std::uint64_t target_state_base() const override { return TargetStateBaseFor(default_branch()); }
  std::uint64_t target_state_bytes() const override {
    return TargetStateBytesFor(default_branch());
  }
  std::uint64_t drafter_state_base() const override {
    return DrafterStateBaseFor(default_branch());
  }
  std::uint64_t drafter_state_bytes() const override {
    return DrafterStateBytesFor(default_branch());
  }
  std::uint64_t used_state_bytes() const override { return UsedStateBytesFor(default_branch()); }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return UsedStateRangesFor(default_branch());
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return SaveUsedStateFor(default_branch(), host, ranges);
  }
  Status RestoreUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return RestoreUsedStateFor(default_branch(), host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const override {
    return CheckpointRangesFor(default_branch(), positions);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range> footprint) override {
    return PrepareRestoreStateFor(default_branch(), footprint);
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return CopyCheckpointStateFor(default_branch(), host, ranges, to_host);
  }
  std::uint32_t cursor() const override { return CursorFor(default_branch()); }
  void set_cursor(std::uint32_t value) override { SetCursorFor(default_branch(), value); }
  void SaveDecodingState() override { SaveDecodingStateFor(default_branch()); }
  void RestoreDecodingState() override { RestoreDecodingStateFor(default_branch()); }
  execution::AdaptiveDepth TurnDecodingState() const override {
    return TurnDecodingStateFor(default_branch());
  }
  void RestoreTurnDecodingState(const execution::AdaptiveDepth& state) override {
    RestoreTurnDecodingStateFor(default_branch(), state);
  }

  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t n_past,
                     bool inject, std::vector<float>& logits) override {
    return NativeSlot(branch).Chunk(all, n_past, logits, inject);
  }
  Status SettleFor(Branch& branch) override { return NativeSlot(branch).Rollback(); }
  bool StateUsableFor(const Branch& branch) const override {
    return NativeSlot(branch).state_usable();
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).state_base();
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).state_bytes();
  }
  std::uint64_t DrafterStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).drafter_state_base();
  }
  std::uint64_t DrafterStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).drafter_state_bytes();
  }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_ranges();
  }
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override {
    return NativeSlot(branch).SaveUsedState(host, ranges);
  }
  Status RestoreUsedStateFor(Branch& branch, void* host,
                             std::span<const engine::LiveState::Range> ranges) override {
    return NativeSlot(branch).RestoreUsedState(host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t positions) const override {
    return NativeSlot(branch).CheckpointRanges(positions);
  }
  Status PrepareRestoreStateFor(Branch& branch,
                                std::span<const engine::LiveState::Range> footprint) override {
    return NativeSlot(branch).PrepareRestoreState(footprint);
  }
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override {
    return NativeSlot(branch).CopyCheckpointState(host, ranges, to_host);
  }
  std::uint32_t CursorFor(const Branch& branch) const override {
    return NativeSlot(branch).pending_rows();
  }
  void SetCursorFor(Branch& branch, std::uint32_t value) override {
    NativeSlot(branch).set_pending_rows(value);
  }

  std::size_t TokenScratchCapacityFor(const Branch& branch, std::size_t tokens,
                                      std::uint32_t left) const override {
    return speculate_ ? tokens + std::min(left, DraftDepthFor(branch)) : 0;
  }

  Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                     std::uint32_t left, std::vector<std::int32_t>& kept,
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                     bool* prefix_kept = nullptr) override {
    if (!ReserveTokenScratch(branch, all.size(), left)) {
      if (prefix_kept != nullptr) {
        *prefix_kept = true;
      }
      return Error("native verify history exceeds the execution budget");
    }
    // Sampling's position keys keep their existing fixed draft schedule.
    // Greedy chooses its depth from deterministic acceptance observations.
    const auto depth = DraftDepthFor(branch);
    if (depth > options_.context - pos) {
      return Error("the drafts would pass the context");
    }
    const auto started = Clock::now();
    std::vector<std::int32_t> drafts;
    if (auto r = NativeSlot(branch).Draft(all, drafts, nullptr, depth); !r) {
      return r;
    }
    // The verify: the anchor and its drafts, within the tokens left.
    const auto rows = std::min<std::uint32_t>(
        {static_cast<std::uint32_t>(drafts.size()) + 1, left, options_.context - pos});
    drafts.resize(rows - 1);
    auto& input = VerifyTokens(branch);
    input.assign(all.begin(), all.end());
    input.insert(input.end(), drafts.begin(), drafts.end());
    std::vector<std::int32_t> argmax;
    std::vector<float> verified;
    const bool rows_needed = logits != nullptr || sampling(branch);
    if (auto r = NativeSlot(branch).Verify(input, pos, argmax, rows_needed ? &verified : nullptr);
        !r) {
      return r;
    }
    // A full greedy step's time by its depth: the calibration of the depth
    // cost ratio (calibration.h), in force from the next start.
    if (!sampling(branch) && rows == depth + 1 && pos < CalibrationSamples::kShortContext) {
      calibration_samples_.DraftStep(depth, Seconds(Clock::now() - started));
    }
    return Undone(branch,
                  JudgeVerify(branch, drafts, pos, depth, argmax, verified, kept, logits, drafted),
                  prefix_kept);
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return NativeSlot(branch).state_refused();
  }
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) override {
    NativeSlot(branch).ForgetRefusal();  // the refusals below are not capacity's
    if (!speculate_) {
      return NativeSlot(branch).ReserveStateThrough(pos + 1);
    }
    const auto depth = DraftDepthFor(branch);
    if (depth > context_ - pos) {
      return Error("the drafts would pass the context");
    }
    const auto verify_rows = std::min({depth + 1, left, context_ - pos});
    // Draft reprocesses tokens ending before the anchor; its final pass
    // ends at pos + depth - 1. Verify includes the anchor itself.
    return NativeSlot(branch).ReserveStateThrough(pos + std::max(depth - 1, verify_rows));
  }

  Status ClearStateFor(Branch& branch) override {
    BranchDecoding(branch) = execution::AdaptiveDepth(3, depth_cost_ratio_);
    return NativeSlot(branch).Clear();
  }
  Status ReleaseIdleStateFor(Branch& branch) override {
    BranchDecoding(branch) = execution::AdaptiveDepth(3, depth_cost_ratio_);
    return NativeSlot(branch).ClearIdle();
  }
  Status SpillFor(Branch& branch) override { return NativeSlot(branch).Spill(); }
  Status RestoreFor(Branch& branch) override { return NativeSlot(branch).Restore(); }
  bool SpilledFor(const Branch& branch) const override { return NativeSlot(branch).spilled(); }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spilled_bytes();
  }
  std::uint64_t SpillWriteBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spill_write_bytes();
  }
  bool LeasedFor(const Branch& branch) const override { return NativeSlot(branch).held(); }
  std::uint64_t RefusedBytesFor(const Branch& branch) const override {
    const auto& slot = NativeSlot(branch);
    return slot.spilled() ? slot.spilled_bytes() : slot.refused_bytes();
  }

 public:
  // Kept across a restart (D-105).
  std::string KeptLayout() const override { return runner_.kept_layout(); }
  std::vector<std::uint64_t> KeptRegions() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).live(), true);
  }
  std::vector<std::uint64_t> KeptLayouts() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).live(), false);
  }
  // D-102's rung 2.
  Status RecoverInPlace() override {
    auto discarded = runner_.RecoverInPlace([this](std::uint32_t slot) { UnkeepSlot(slot); });
    if (!discarded) {
      return std::unexpected(discarded.error());
    }
    ForgetDiscarded(*discarded);
    for (std::size_t i = 0; i < branches(); ++i) {
      if ((*discarded & (engine::SlotMask{1} << i)) != 0) {
        if (auto b = branch(i); b) {
          // As a cleared state's.
          BranchDecoding(**b) = execution::AdaptiveDepth(3, depth_cost_ratio_);
        }
      }
    }
    return {};
  }

 protected:
  const engine::LiveState* KeptLiveFor(const Branch& branch) const override {
    return &NativeSlot(branch).live();
  }
  bool KeptWholeFor(const Branch& branch) const override { return NativeSlot(branch).kept_whole(); }
  bool OwedFor(const Branch& branch) const override { return NativeSlot(branch).owed(); }
  Status AdoptFor(Branch& branch, std::span<const engine::LiveState::Range> used) override {
    return NativeSlot(branch).Adopt(used);
  }
  void SetSpillPlaces(
      const std::function<engine::LiveState::SpillPlace(std::uint32_t)>& place) override {
    options_.spill_place = place;
  }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }

  bool GenerationCohortUsable() const override { return runner_.cohort_usable(); }

  Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) override {
    if (!runner_.waves_provisioned()) {
      return Llm::RunPreparedGenerationWave(prepared);
    }
    if (!speculate_) {
      std::vector<engine::Qwen38Runner::ChunkWork> chunks;
      chunks.reserve(prepared.size());
      for (PreparedGeneration& unit : prepared) {
        chunks.push_back({.slot = &NativeSlot(*unit.branch),
                          .history = unit.step.all,
                          .n_past = unit.step.position,
                          .logits = &unit.row});
      }
      return runner_.ChunkWave(chunks);
    }

    // These owners never move once the borrowed descriptors are made. Each
    // completed Draft/VerifyWave consumes its inputs before this frame ends.
    struct Frame {
      std::uint32_t depth = 0;
      std::vector<std::int32_t> drafts;
      std::vector<std::int32_t> argmax;
      std::vector<float> verified;
    };
    std::vector<Frame> frames(prepared.size());
    std::vector<engine::Qwen38Runner::DraftWork> drafts;
    drafts.reserve(prepared.size());
    const auto started = Clock::now();
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      Frame& frame = frames[i];
      // A shared wave drafts shared_wave_depth tokens a request (two: each
      // verify row past that reads its own routed experts for every member,
      // so the third draft's acceptance no longer pays for itself; C2/C4
      // HTTP screens: +8%/+15% over adaptive depth). A lone request keeps
      // its adaptive depth.
      frame.depth =
          std::min(DraftDepthFor(*unit.branch), prepared.size() > 1 ? shared_wave_depth_ : 3U);
      if (frame.depth > context_ - unit.step.position) {
        return Error("the drafts would pass the context");
      }
      drafts.push_back({.slot = &NativeSlot(*unit.branch),
                        .history = unit.step.all,
                        .drafts = &frame.drafts,
                        .passes = frame.depth});
    }
    // Past draft_wave_max requests (two) each drafts alone, on its own
    // cached graphs: a draft wave's graph keys every slot's pending rows
    // (one to four), so a wider wave's graphs would seldom repeat (C4 HTTP
    // screen: 3.6% faster).
    if (drafts.size() > draft_wave_max_) {
      for (std::size_t i = 0; i < prepared.size(); ++i) {
        if (auto ran = NativeSlot(*prepared[i].branch)
                           .Draft(prepared[i].step.all, frames[i].drafts, nullptr, frames[i].depth);
            !ran) {
          return ran;
        }
      }
    } else if (auto ran = runner_.DraftWave(drafts); !ran) {
      return ran;
    }
    std::vector<engine::Qwen38Runner::VerifyWork> verifies;
    verifies.reserve(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      Frame& frame = frames[i];
      // Keep the scalar schedule: all requested draft passes run, then the
      // verify is truncated independently by this branch's output/context.
      const auto rows =
          std::min<std::uint32_t>({static_cast<std::uint32_t>(frame.drafts.size()) + 1,
                                   unit.step.left, context_ - unit.step.position});
      frame.drafts.resize(rows - 1);
      auto& input = VerifyTokens(*unit.branch);
      input.assign(unit.step.all.begin(), unit.step.all.end());
      input.insert(input.end(), frame.drafts.begin(), frame.drafts.end());
      verifies.push_back(
          {.slot = &NativeSlot(*unit.branch),
           .history = input,
           .n_past = unit.step.position,
           .argmax = &frame.argmax,
           .logits = unit.step.need_logits || sampling(*unit.branch) ? &frame.verified : nullptr});
    }
    if (auto ran = runner_.VerifyWave(verifies); !ran) {
      return ran;
    }
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      const Frame& frame = frames[i];
      unit.result = JudgeVerify(*unit.branch, frame.drafts, unit.step.position, frame.depth,
                                frame.argmax, frame.verified, unit.kept,
                                unit.step.need_logits ? &unit.logits : nullptr, unit.drafted);
      if (!unit.result) {
        // The shared Verify completed; only this branch's host judgement
        // failed. Undo its own completed verify before preserving its prefix.
        if (auto rolled = NativeSlot(*unit.branch).DiscardVerify(); !rolled) {
          if (!runner_.cohort_usable()) {
            return Error(
                std::format("{}; rolling back failed: {}", unit.result.error(), rolled.error()));
          }
          unit.result = Error(
              std::format("{}; rolling back failed: {}", unit.result.error(), rolled.error()));
        } else {
          unit.failed_prefix_valid = true;
        }
      }
    }
    // A lone greedy request's full step by its depth: the calibration of
    // the depth cost ratio (calibration.h), in force from the next start.
    if (prepared.size() == 1 && prepared[0].result && !sampling(*prepared[0].branch) &&
        frames[0].drafts.size() == frames[0].depth &&
        prepared[0].step.position < CalibrationSamples::kShortContext) {
      calibration_samples_.DraftStep(frames[0].depth, Seconds(Clock::now() - started));
    }
    return {};
  }

 private:
  // A failed judgement after the native verify completed: the verify is
  // undone, so the step's starting prefix still holds (`prefix_kept`, as a
  // wave's failed_prefix_valid); an undo that fails says so.
  Status Undone(Branch& branch, Status judged, bool* prefix_kept) {
    if (judged) {
      return judged;
    }
    if (auto rolled = NativeSlot(branch).DiscardVerify(); !rolled) {
      return Error(std::format("{}; rolling back failed: {}", judged.error(), rolled.error()));
    }
    if (prefix_kept != nullptr) {
      *prefix_kept = true;
    }
    return judged;
  }

  std::uint32_t DraftDepthFor(const Branch& branch) const {
    return sampling(branch) || runner_.draft_rows() < 3 ? 2U : BranchDecoding(branch).Choose();
  }
  Status JudgeVerify(Branch& branch, std::span<const std::int32_t> drafts, std::uint32_t pos,
                     std::uint32_t depth, std::span<const std::int32_t> argmax,
                     std::span<const float> verified, std::vector<std::int32_t>& kept,
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted) {
    const auto rows = static_cast<std::uint32_t>(drafts.size()) + 1;
    const std::uint32_t vocab = runner_.vocab();
    // Validate the runner's whole completed result before using borrowed rows.
    if (argmax.size() != rows ||
        ((logits != nullptr || sampling(branch)) && verified.size() != std::size_t{rows} * vocab)) {
      return Error("a Qwen verify returned inconsistent token/logit counts");
    }
    const auto row = [&](std::uint32_t i) {
      return verified.subspan(std::size_t{i} * vocab, vocab);
    };
    std::uint32_t m = 0;
    std::int32_t next = -1;
    if (!sampling(branch)) {
      // Greedy: the verify's own argmaxes.
      for (; m < rows - 1; ++m) {
        if (argmax[m] != drafts[m]) {
          next = argmax[m];
          break;
        }
      }
      if (next < 0) {
        next = argmax[m];
      }
    } else {
      // Speculative sampling over the verify's rows; row m predicts the
      // token at pos + m + 1.
      for (; m < rows - 1; ++m) {
        std::int32_t instead = -1;
        auto keep = Keep(branch, row(m), drafts[m], std::uint64_t{pos} + m + 1, instead);
        if (!keep) {
          return std::unexpected(keep.error());
        }
        if (!*keep) {
          next = instead;
          break;
        }
      }
      if (next < 0) {
        auto chosen = Choose(branch, row(m), std::uint64_t{pos} + m + 1);
        if (!chosen) {
          return std::unexpected(chosen.error());
        }
        next = *chosen;
      }
    }
    if (auto r = NativeSlot(branch).Accept(m + 1); !r) {
      return r;
    }
    kept.assign(drafts.begin(), drafts.begin() + m);
    kept.push_back(next);
    if (logits != nullptr) {
      for (std::uint32_t i = 0; i <= m; ++i) {
        const std::span<const float> kept_row = row(i);
        logits->emplace_back(kept_row.begin(), kept_row.end());
      }
    }
    drafted += rows - 1;
    if (!sampling(branch)) {
      BranchDecoding(branch).Observe(depth, m + 1, rows == depth + 1);
    }
    return {};
  }
  engine::Qwen38Runner::Slot& NativeSlot(Branch& branch) {
    const auto index = BranchIndex(branch);
    base::Check(native_slots_[index] != nullptr, "native conversation slots are not ready");
    return *native_slots_[index];
  }
  const engine::Qwen38Runner::Slot& NativeSlot(const Branch& branch) const {
    const auto index = BranchIndex(branch);
    base::Check(native_slots_[index] != nullptr, "native conversation slots are not ready");
    return *native_slots_[index];
  }
  config::ModelEntry entry_;  // its tokenizer and template files, if named
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  fs::path store_;
  std::string slots_report_;
  // A wave's drafts a request, and the most requests drafting in one
  // joined wave (its settings').
  std::uint32_t shared_wave_depth_ = kQwen38SharedWaveDepth;
  std::uint32_t draft_wave_max_ = kQwen38DraftWaveMax;
  engine::Qwen38Options options_;  // before the runner, which keeps a reference
  engine::Qwen38Runner runner_;
  std::array<engine::Qwen38Runner::Slot*, engine::Qwen38Runner::kRequestSlots> native_slots_{};
};

// The Qwen-Image-2.1 pipeline (engine/qwen_image_runner.h).
class QwenImage final : public Image {
 public:
  QwenImage(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
            const config::RuntimeRoles& roles, const ServingOptions& serving, int index)
      : composition_id_(entry.composition.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    options_.store = roles.installed;
    options_.composition = composition_id_;
    options_.noise = serving.image_noise;
    options_.out = roles.spill;
    options_.prompt = serving.image_prompt;
    options_.size = settings.image_size.value;
    options_.steps = settings.image_steps.value;
  }

  engine::PagedModel& paged() override { return runner_; }
  Status Setup() override {
    // Every component under the store's trust rules first.
    const fs::path root = store_ / composition_id_;
    auto walked = platform::WalkTrusted(root, ::geteuid(), false);
    if (!walked || !walked->exists || !S_ISDIR(walked->status.st_mode) ||
        platform::OthersCanWrite(walked->status, ::geteuid(), root)) {
      return Error(
          std::format("composition {}: not a directory only root and this user can "
                      "change{}",
                      composition_id_, walked ? "" : ": " + walked.error()));
    }
    auto composition = ja::OpenComposition(root, composition_id_);
    if (!composition) {
      return Error(
          std::format("composition {}: {}", composition_id_, composition.error().ToString()));
    }
    for (const ja::CompositionComponent& c : composition->components()) {
      if (auto opened = OpenTrusted(store_, c.artifact); !opened) {
        return std::unexpected(opened.error());
      }
    }
    return runner_.Setup();
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return engine::QwenImageRunner::pool_needed(); }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  std::string extra() const override { return runner_.Report(); }
  std::string plan_report() const override { return runner_.plan_report(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  Status FirstOutput(std::string& sha) override { return runner_.FirstOutput(sha); }
  Status Finish(std::string& sha) override { return runner_.Finish(sha); }

 private:
  std::string composition_id_;
  fs::path store_;
  engine::QwenImageOptions options_;  // before the runner, which keeps a reference
  engine::QwenImageRunner runner_;
};

}  // namespace

// ---------------------------------------------------------------- memory

MemorySampler::MemorySampler() : start_(Available()) {
  low_ = start_;
  all_ = start_;
  thread_ = std::jthread([this] {
    if (auto stack = platform::InstallThreadSignalStack(); !stack) {
      std::println(stderr, "the memory sampler runs without a signal stack: {}", stack.error());
    }
    Loop();
  });
}

MemorySampler::~MemorySampler() {
  stop_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

std::uint64_t MemorySampler::Available() { return platform::AvailableMemoryBytes().value_or(0); }

void MemorySampler::Reset() { low_ = Available(); }

void MemorySampler::Loop() {
  const auto lower = [](std::atomic<std::uint64_t>& to, std::uint64_t now) {
    std::uint64_t seen = to.load();
    while (now < seen && !to.compare_exchange_weak(seen, now)) {
    }
  };
  while (!stop_) {
    const std::uint64_t now = Available();
    if (now != 0) {
      lower(low_, now);
      lower(all_, now);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void ResidentTimes::Staged(catalog::ExtentId extent, scheduler::PageInEvent event) {
  events_.fetch_add(1, std::memory_order_relaxed);
  if (event == scheduler::PageInEvent::kResident && extent.index() < at_.size()) {
    at_[extent.index()] = Clock::now();
  }
}

Clock::time_point ResidentTimes::Latest(std::span<const catalog::ExtentId> extents) const {
  Clock::time_point latest{};
  for (const catalog::ExtentId extent : extents) {
    if (extent.index() < at_.size()) {
      latest = std::max(latest, at_[extent.index()]);
    }
  }
  return latest;
}

// ---------------------------------------------------------------- an LLM

Llm::Llm() : default_branch_(*this) {}

void Llm::SetTokenMemory(RequestMemory& memory) {
  base::Check(token_history_bytes() == 0, "changing a model's token pool with live history");
  for (std::size_t slot = 0; slot < branches(); ++slot) {
    const auto b = branch(slot);
    base::Check(b && !(*b)->generation_active_ && (*b)->prompt_session_ == nullptr &&
                    !(*b)->funding_history_,
                "changing a model's token pool with an active session");
  }
  token_memory_ = &memory;
}

std::uint64_t Llm::token_history_bytes() const {
  std::uint64_t bytes = 0;
  for (std::uint32_t slot = 0; slot < branch_count_; ++slot) {
    const Branch& b = slot == 0 ? default_branch_ : *extra_branches_[slot - 1];
    bytes += b.history_charge_.bytes() + b.saved_history_charge_.bytes() +
             b.verify_tokens_charge_.bytes();
  }
  return bytes;
}

std::uint64_t Llm::IdleHistoryBytes(const Branch& branch) const {
  CheckBranch(branch);
  return branch.history_charge_.bytes() + branch.verify_tokens_charge_.bytes();
}

bool Llm::HistoryReclaimable(const Branch& branch) const {
  return BranchIdle(branch) && !branch.HeldByContinuation() && !branch.funding_history_ &&
         !branch.saved_valid_;
}

bool Llm::ReserveTokens(Branch& branch, std::vector<std::int32_t>& tokens, MemoryCharge& charge,
                        std::size_t capacity) {
  if (capacity <= tokens.capacity()) {
    return true;
  }
  const bool was_funding = std::exchange(branch.funding_history_, true);
  // Preserve amortized growth under pressure too: exact one-token growth
  // would repeatedly copy the whole conversation near the budget limit.
  const std::size_t grown =
      std::max(capacity, std::min(std::size_t{context_} + 1, tokens.capacity() * 2));
  const bool funded = ReserveTokenStorage(tokens, charge, *token_memory_, grown);
  token_memory_->Settle(0, std::chrono::milliseconds{0});
  branch.funding_history_ = was_funding;
  return funded;
}

bool Llm::ReserveTokenScratch(Branch& branch, std::size_t tokens, std::uint32_t left) {
  return ReserveTokens(branch, branch.verify_tokens_, branch.verify_tokens_charge_,
                       TokenScratchCapacityFor(branch, tokens, left));
}

void Llm::ReleaseTokens(std::vector<std::int32_t>& tokens, MemoryCharge& charge) {
  ReleaseTokenStorage(tokens, charge);
  token_memory_->Settle(0, std::chrono::milliseconds{0});
}

void Llm::PublishTokens(Branch& branch, std::vector<std::int32_t>& tokens, MemoryCharge& charge,
                        std::size_t size) {
  base::Check(size <= tokens.size(), "publishing beyond the funded history");
  if (size == 0) {
    ReleaseTokens(tokens, charge);
    ReleaseTokens(branch.history_, branch.history_charge_);
    return;
  }
  tokens.resize(size);
  branch.history_.swap(tokens);
  std::swap(branch.history_charge_, charge);
  ReleaseTokens(tokens, charge);
}

void Llm::Branch::InvalidateStateSnapshot() {
  saved_valid_ = false;
  model_.ReleaseTokens(saved_history_, saved_history_charge_);
  saved_ranges_.clear();
}

Llm::Branch::~Branch() {
  base::Check(!generation_active_, "a branch with an active generation was destroyed");
  base::Check(prompt_session_ == nullptr, "a branch with an active prompt was destroyed");
}

Status Llm::Branch::Clear() { return model_.Clear(*this); }

Status Llm::Branch::ReleaseIdleState() { return model_.ReleaseIdleState(*this); }

bool Llm::Branch::spilled() const { return model_.SpilledFor(*this); }

void Llm::Branch::Forget() { model_.Forget(*this); }

Status Llm::Branch::Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last,
                            const PrefillGoOn& go_on, PrefillRun* run) {
  return model_.Prefill(*this, tokens, last, go_on, run);
}

Status Llm::Branch::ScorePrompt(
    std::span<const std::int32_t> tokens, std::vector<float>& last,
    const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
    const PrefillGoOn& go_on, PrefillRun* run) {
  return model_.ScorePrompt(*this, tokens, last, on_row, go_on, run);
}

Status Llm::Branch::PreparePrompt(std::span<const std::int32_t> tokens,
                                  std::uint32_t stable_boundary, std::vector<float>& last,
                                  std::uint32_t& reused, const PrefillGoOn& go_on, PrefillRun* run,
                                  bool fresh) {
  return model_.PreparePrompt(*this, tokens, stable_boundary, last, reused, go_on, run, fresh);
}

std::expected<std::unique_ptr<Llm::PromptSession>, std::string> Llm::Branch::BeginPrompt(
    std::span<const std::int32_t> tokens, std::uint32_t stable_boundary, bool fresh,
    bool resume) & {
  return model_.BeginPrompt(*this, tokens, stable_boundary, fresh, resume);
}

std::expected<std::unique_ptr<Llm::PromptSession>, std::string> Llm::Branch::BeginScoringPrompt(
    std::span<const std::int32_t> tokens,
    std::function<bool(std::int32_t, std::span<const float>)> on_row, bool resume) & {
  return model_.BeginPrompt(*this, tokens, 0, !resume, resume, true, std::move(on_row));
}

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::Branch::BeginGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return model_.BeginGeneration(*this, last, options, out);
}

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::Branch::ResumeGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return model_.BeginGeneration(*this, last, options, out, true);
}

Status Llm::Branch::Generate(const std::vector<float>& last, const GenerateOptions& options,
                             Generation& out) {
  return model_.Generate(*this, last, options, out);
}

std::uint64_t Llm::Branch::state_snapshot_bytes() const {
  model_.CheckBranch(*this);
  return model_.UsedStateBytesFor(*this);
}

Status Llm::Branch::SaveState(void* host) { return model_.SaveState(*this, host); }

Status Llm::Branch::RestoreState(void* host) { return model_.RestoreState(*this, host); }

Status Llm::Clear() { return default_branch_.Clear(); }

void Llm::Forget() { default_branch_.Forget(); }

Status Llm::Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last,
                    const PrefillGoOn& go_on, PrefillRun* run) {
  return default_branch_.Prefill(tokens, last, go_on, run);
}

Status Llm::ScorePrompt(std::span<const std::int32_t> tokens, std::vector<float>& last,
                        const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                        const PrefillGoOn& go_on, PrefillRun* run) {
  return default_branch_.ScorePrompt(tokens, last, on_row, go_on, run);
}

Status Llm::PreparePrompt(std::span<const std::int32_t> tokens, std::uint32_t stable_boundary,
                          std::vector<float>& last, std::uint32_t& reused, const PrefillGoOn& go_on,
                          PrefillRun* run, bool fresh) {
  return default_branch_.PreparePrompt(tokens, stable_boundary, last, reused, go_on, run, fresh);
}

std::uint64_t Llm::turn_checkpoint_bytes() const { return default_branch_.turn_checkpoint_bytes(); }

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::BeginGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return default_branch_.BeginGeneration(last, options, out);
}

Status Llm::Generate(const std::vector<float>& last, const GenerateOptions& options,
                     Generation& out) {
  return default_branch_.Generate(last, options, out);
}

std::uint64_t Llm::state_snapshot_bytes() const { return default_branch_.state_snapshot_bytes(); }

Status Llm::SaveState(void* host) { return default_branch_.SaveState(host); }

Status Llm::RestoreState(void* host) { return default_branch_.RestoreState(host); }

std::expected<std::vector<std::int32_t>, std::string> Llm::EncodeText(std::string_view text,
                                                                      RequestMemory* memory,
                                                                      std::uint64_t* needed) const {
  std::vector<tokenizer::TokenId> ids;
  // The working set and the tokens (a byte each at most, grown by
  // doubling), charged while they are built.
  MemoryCharge charge;
  if (memory != nullptr) {
    const std::uint64_t need = tokenizer_->WorkingBytes(text) +
                               (std::uint64_t{2} * sizeof(tokenizer::TokenId) * (text.size() + 2));
    if (!charge.Add(*memory, need)) {
      if (needed != nullptr) {
        *needed = need;
      }
      return Error(std::format("encoding the text needs {} bytes of request memory", need));
    }
  }
  // Bounded by the text itself (D-102): every token covers at least a
  // byte, beside a BOS and an EOS.
  if (auto r = tokenizer_->Encode(text,
                                  {.special = tokenizer::SpecialTokens::kUserDefinedOnly,
                                   .add_bos_eos = true,
                                   .max_bytes = text.size(),
                                   .max_tokens = text.size() + 2},
                                  ids);
      !r) {
    return Error(r.error().ToString());
  }
  if (tokenizer_->bos() && (ids.empty() || ids.front() != *tokenizer_->bos())) {
    ids.insert(ids.begin(), *tokenizer_->bos());  // BOS first, as the references fed it
  }
  return ids;
}

RenderCharge Llm::RenderChargeFor(std::uint64_t message_bytes, std::size_t messages) const {
  constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
  // Four times the text, and a message's framing (role headers, turn
  // markers) for each message, however short.
  const std::uint64_t framing = std::uint64_t{messages} * kRenderBytesPerMessage;
  const std::uint64_t scaled = message_bytes > (std::numeric_limits<std::uint64_t>::max() / 8) ||
                                       framing > (std::numeric_limits<std::uint64_t>::max() / 8)
                                   ? std::numeric_limits<std::uint64_t>::max() / 4
                                   : (4 * message_bytes) + framing;
  RenderCharge c;
  c.output = std::min<std::uint64_t>(render_bytes_, scaled + kMinRenderBytes);
  c.bytes = c.output;
  if (template_ && template_->interprets()) {
    c.live = std::min<std::uint64_t>(template_->max_live_bytes(), scaled + (16 * kMiB));
    c.bytes += c.output + c.live + (2 * message_bytes);
  }
  return c;
}

std::expected<std::vector<std::int32_t>, std::string> Llm::RenderChat(
    const chat::Conversation& conversation, std::uint32_t* stable_boundary,
    const ChatRenderOptions& options, ChatRenderFailure* failure) const {
  if (stable_boundary != nullptr) {
    *stable_boundary = 0;
  }
  if (failure != nullptr) {
    *failure = ChatRenderFailure::kOther;
  }
  if (!template_) {
    return Error(std::format("{} has no chat template (D-067)", name_));
  }
  if (auto checked = CheckConversation(conversation); !checked) return Error(checked.error());
  auto rendered = template_->Render(conversation, LocalTime(), options.cancelled);
  if (!rendered) {
    if (failure != nullptr && rendered.error().cancelled) {
      *failure = ChatRenderFailure::kCancelled;
    } else if (failure != nullptr && rendered.error().bound &&
               conversation.max_render_bytes >= render_bytes_) {
      // Longer than any prompt that fits the context could render.
      *failure = ChatRenderFailure::kTooLong;
    }
    return Error(rendered.error().ToString());
  }
  // The tokenization's working set and its tokens (a byte each at most,
  // grown by doubling), charged while they are built.
  MemoryCharge charge;
  if (options.memory != nullptr) {
    const std::uint64_t window = tokenizer_->WorkingBytes(rendered->text);
    const std::uint64_t need =
        window + (std::uint64_t{2} * sizeof(tokenizer::TokenId) *
                  std::min<std::uint64_t>(rendered->text.size(), options.max_tokens));
    if (need > options.memory->capacity()) {
      // A stretch without a cut point too long to tokenize here at all,
      // whatever else the node holds: the prompt's shape, not the moment.
      if (failure != nullptr) {
        *failure = ChatRenderFailure::kUnbroken;
      }
      if (options.memory_needed != nullptr) {
        *options.memory_needed = window / tokenizer::kEncodeBytesPerWindowByte;
      }
      return Error("the conversation holds text too long without a break to tokenize");
    }
    if (!charge.Add(*options.memory, need)) {
      if (failure != nullptr) {
        *failure = ChatRenderFailure::kMemory;
      }
      if (options.memory_needed != nullptr) {
        *options.memory_needed = need;
      }
      return Error(std::format("tokenizing the rendering needs {} bytes of request memory", need));
    }
  }
  // The rendering's own bytes (already within its bound) and the caller's
  // token bound, not the tokenizer's library defaults: a prompt is bounded
  // by the model's context (D-102).
  std::vector<tokenizer::TokenId> ids;
  std::vector<std::size_t> span_tokens;
  if (auto r = tokenizer_->EncodeMarked(rendered->text, rendered->specials,
                                        {.special = tokenizer::SpecialTokens::kUserDefinedOnly,
                                         .add_bos_eos = false,
                                         .max_bytes = rendered->text.size(),
                                         .max_tokens = options.max_tokens},
                                        ids, stable_boundary == nullptr ? nullptr : &span_tokens);
      !r) {
    if (failure != nullptr && r.error().rule == tokenizer::Rule::kOutputTooLarge) {
      *failure = ChatRenderFailure::kTooLong;
    }
    if (failure != nullptr && r.error().rule == tokenizer::Rule::kCancelled) {
      *failure = ChatRenderFailure::kCancelled;
    }
    return Error(r.error().ToString());
  }
  if (stable_boundary != nullptr) {
    for (const chat::Boundary& boundary : rendered->boundaries) {
      if (boundary.kind != chat::BoundaryKind::kGenerationPrompt) {
        continue;
      }
      for (std::size_t i = 0; i < rendered->specials.size(); ++i) {
        if (rendered->specials[i].offset == boundary.offset && span_tokens[i] < ids.size()) {
          *stable_boundary = static_cast<std::uint32_t>(span_tokens[i]);
          break;
        }
      }
    }
  }
  return ids;
}

std::string Llm::Detokenize(std::span<const std::int32_t> tokens) const {
  std::string text;
  if (auto r = tokenizer_->Decode(tokens, {}, text); !r) {
    return "(not decodable)";
  }
  return text;
}

void Llm::ReserveBranchSamplingScratch(Branch& branch, std::size_t capacity) {
  CheckBranch(branch);
  branch.scratch_.reserve(capacity);
}
void Llm::DropBranchSamplingScratch(Branch& branch) {
  CheckBranch(branch);
  std::vector<execution::SamplingCandidate>().swap(branch.scratch_);
}

std::expected<std::int32_t, std::string> Llm::Choose(std::span<const float> row,
                                                     std::uint64_t position) {
  return Choose(default_branch_, row, position);
}

std::expected<std::int32_t, std::string> Llm::Choose(Branch& branch, std::span<const float> row,
                                                     std::uint64_t position) {
  CheckBranch(branch);
  if (!branch.sampling_) {
    return engine::Argmax(row);
  }
  if (auto prepared = PrepareSamplingScratchFor(branch, row.size()); !prepared)
    return std::unexpected(prepared.error());
  auto token =
      execution::Sample(row, *branch.sampling_,
                        {.seed = branch.seed_, .stream = 0, .position = position}, branch.scratch_);
  if (!token) {
    return Error(std::format("sampling: {}", execution::SamplingErrorName(token.error())));
  }
  return *token;
}

std::expected<bool, std::string> Llm::Keep(std::span<const float> row, std::int32_t draft,
                                           std::uint64_t position, std::int32_t& next) {
  return Keep(default_branch_, row, draft, position, next);
}

std::expected<bool, std::string> Llm::Keep(Branch& branch, std::span<const float> row,
                                           std::int32_t draft, std::uint64_t position,
                                           std::int32_t& next) {
  CheckBranch(branch);
  if (!branch.sampling_) {
    next = engine::Argmax(row);
    return next == draft;
  }
  auto verdict = execution::VerifyDraft(row, draft, *branch.sampling_,
                                        {.seed = branch.seed_, .stream = 0, .position = position},
                                        branch.scratch_);
  if (!verdict) {
    return Error(std::format("sampling: {}", execution::SamplingErrorName(verdict.error())));
  }
  next = verdict->token;
  return verdict->accepted;
}

Status Llm::UseTemplate(std::string_view text) {
  // A rendering's bounds follow the model (D-102): what a prompt that fits
  // its context could occupy; a value and the values held at once scale
  // with it, never below the library's own. The longest token's bytes: a
  // normal token's decoded, a special token's text.
  const std::size_t longest = tokenizer_->longest_token_bytes();
  longest_token_ = longest;
  render_bytes_ = static_cast<std::size_t>(RenderBytes(
      context_, longest, tokenizer_->normalization() == tokenizer::Normalization::kNfc));
  chat::jinja::Limits limits;
  limits.max_output_bytes = render_bytes_;
  limits.max_string_bytes = std::max(limits.max_string_bytes, render_bytes_);
  limits.max_live_bytes = std::max(limits.max_live_bytes, 2 * render_bytes_);
  auto chosen = chat::ChatTemplate::ForText(text, chat::TokenFacts::From(*tokenizer_), limits);
  if (!chosen) {
    return Error(chosen.error());
  }
  template_ = std::move(*chosen);
  auto stops = chat::StopTokens(template_->stop(), *tokenizer_);
  if (!stops) {
    return Error(stops.error().ToString());
  }
  stops_.assign(stops->begin(), stops->end());
  if (tokenizer_->eos() && std::ranges::find(stops_, *tokenizer_->eos()) == stops_.end()) {
    stops_.push_back(*tokenizer_->eos());
  }
  if (template_->how() != chat::ChatTemplate::How::kNativeByHash) {
    std::println(stderr, "{}: chat template {} rendered {} (D-067)", name_,
                 template_->sha256().substr(0, 12),
                 template_->how() == chat::ChatTemplate::How::kNativeByProbe
                     ? std::format("natively by {}, equal on its probe corpus", template_->name())
                     : std::string("by the sandboxed template interpreter"));
  }
  return {};
}

Status Llm::UseChatAssets(const ja::Artifact& artifact, const config::ModelEntry& entry) {
  auto assets = ReadChatAssets(artifact, entry, ::geteuid());
  if (!assets) {
    return std::unexpected(assets.error());
  }
  auto created = tokenizer::Tokenizer::Create(std::move(assets->spec));
  if (!created) {
    return Error(std::format("{}: {}", assets->from, created.error().ToString()));
  }
  tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
  // The reasoning markers (D-103): an override, else the vocabulary's
  // "<think>" pair, else none.
  if (auto r = ResolveReasoning(settings_, entry,
                                [this](std::string_view text) { return tokenizer_->Find(text); });
      !r) {
    return r;
  }
  const auto find = [this](const std::string& text) -> std::optional<std::int32_t> {
    return text.empty() ? std::nullopt : tokenizer_->Find(text);
  };
  think_start_ = find(settings_.reasoning_start.value);
  think_end_ = find(settings_.reasoning_end.value);
  // A model is refused when it could serve no turn.
  if (!assets->chat_template) {
    return Error(
        std::format("{} keeps no chat template and models.{}.chat_template is not set, "
                    "so it has no chat turns",
                    assets->from, entry.name));
  }
  return UseTemplate(*assets->chat_template);
}

Status Llm::PrepareBranches(std::uint32_t count, std::uint32_t draft_depth) {
  if (branches_prepared_ || count == 0 || count > kMaxBranches || draft_depth == 0 ||
      default_branch_.generation_active_ || !default_branch_.history_.empty()) {
    return Error("native branches must be prepared once before conversation use");
  }
  for (std::uint32_t slot = 1; slot < count; ++slot) {
    extra_branches_[slot - 1] = std::unique_ptr<Branch>(new Branch(*this, slot));
    extra_branches_[slot - 1]->decoding_ = execution::AdaptiveDepth(draft_depth, depth_cost_ratio_);
    extra_branches_[slot - 1]->saved_decoding_ =
        execution::AdaptiveDepth(draft_depth, depth_cost_ratio_);
  }
  default_branch_.decoding_ = execution::AdaptiveDepth(draft_depth, depth_cost_ratio_);
  default_branch_.saved_decoding_ = execution::AdaptiveDepth(draft_depth, depth_cost_ratio_);
  branch_count_ = count;
  branches_prepared_ = true;
  return {};
}

std::expected<Llm::Branch*, std::string> Llm::branch(std::size_t index) & {
  if (index >= branch_count_) {
    return Error("the model has no native conversation slot at this index");
  }
  return index == 0 ? &default_branch_ : extra_branches_[index - 1].get();
}

Status Llm::SelectBranches(std::span<Branch* const> active) {
  if (active.size() != 1 || active.front() != &default_branch_) {
    return Error("this model supports only its default native conversation");
  }
  return {};
}

void Llm::CheckBranch(const Branch& branch) const {
  base::Check(&branch.model_ == this && branch.slot_ < branch_count_ &&
                  (branch.slot_ == 0 ? &branch == &default_branch_
                                     : &branch == extra_branches_[branch.slot_ - 1].get()),
              "conversation branch does not belong to this model's native slot");
}

void Llm::CheckDefaultBranch(const Branch& branch) const {
  CheckBranch(branch);
  base::Check(&branch == &default_branch_, "this family forwards native work only for branch zero");
}

std::uint32_t Llm::BranchIndex(const Branch& branch) const {
  CheckBranch(branch);
  return branch.slot_;
}

bool Llm::AnyBranchHasRetainedState() const {
  if (default_branch_.HasRetainedState()) {
    return true;
  }
  for (std::uint32_t slot = 1; slot < branch_count_; ++slot) {
    if (extra_branches_[slot - 1]->HasRetainedState()) {
      return true;
    }
  }
  return false;
}

bool Llm::BranchIdle(const Branch& branch) const {
  CheckBranch(branch);
  return !branch.generation_active_ && branch.prompt_session_ == nullptr &&
         !branch.funding_history_ && !LeasedFor(branch);
}

std::uint64_t Llm::StateBytes(const Branch& branch) const {
  CheckBranch(branch);
  return UsedStateBytesFor(branch);
}

std::uint64_t Llm::ResidentStateBytes(const Branch& branch) const {
  CheckBranch(branch);
  return SpilledFor(branch) ? 0 : UsedStateBytesFor(branch);
}

std::uint64_t Llm::SpilledStateBytes(const Branch& branch) const {
  CheckBranch(branch);
  return SpilledBytesFor(branch);
}

Clock::time_point Llm::LastUsed(const Branch& branch) const {
  CheckBranch(branch);
  return branch.history_used_.at;
}

memory::ReclaimStamp Llm::LastStamp(const Branch& branch) const {
  CheckBranch(branch);
  return branch.history_used_.reclaim;
}

std::uint64_t Llm::SpillWriteBytes(const Branch& branch) const {
  CheckBranch(branch);
  return SpillWriteBytesFor(branch);
}

std::size_t Llm::ReusablePrefix(const Branch& branch, std::span<const std::int32_t> tokens) const {
  CheckBranch(branch);
  const auto now = Clock::now();
  if (branch.needs_clear_ || branch.history_.empty() || !StateUsableFor(branch) ||
      (!branch.HeldByContinuation() && now - branch.history_used_.at >= retention_)) {
    return 0;
  }
  const std::size_t common = CommonPrefix(branch.history_, tokens);
  if (common == branch.history_.size() && common < tokens.size()) {
    return common;
  }
  std::vector<TurnBoundary> boundaries;
  boundaries.reserve(branch.turn_checkpoints_.size());
  for (const TurnCheckpoint& checkpoint : branch.turn_checkpoints_) {
    boundaries.push_back(checkpoint.boundary);
  }
  const auto match = MatchingTurnBoundary(boundaries, common, tokens.size(), now, retention_);
  return match ? boundaries[*match].position : 0;
}

// ------------------------------------------- conversations kept (D-105)

void Llm::set_kept(Kept kept) {
  kept_ = std::move(kept);
  if (kept_.keeper == nullptr) {
    SetSpillPlaces({});
    return;
  }
  // Each slot's spill file named beneath the model's private directory,
  // kept as it is for the slots whose records validated.
  SetSpillPlaces([dir = kept_.directory, adopt = kept_.adopt,
                  spill = checkpoint_directory_](std::uint32_t slot) {
    return engine::LiveState::SpillPlace{.directory = spill,
                                         .dir = dir,
                                         .name = kept::StateFileName(slot),
                                         .keep = std::ranges::contains(adopt, slot)};
  });
}

void Llm::Unkeep(const Branch& branch) {
  if (kept_.keeper != nullptr) {
    kept_.keeper->Invalidate(kept_.model, BranchIndex(branch));
  }
}

bool Llm::SettleIdle(Branch& branch) {
  if (!OwedFor(branch)) {
    return true;
  }
  if (node_->InRequest(paged().stream()) || !BranchIdle(branch) || SpilledFor(branch) ||
      !StateUsableFor(branch)) {
    return false;
  }
  // Selected alone (no request holds the stream), its owed restore runs as
  // the next job's would have: the same bytes, now.
  std::array<Branch*, 1> alone{&branch};
  if (auto selected = SelectBranches(alone); !selected) {
    return false;
  }
  return SettleFor(branch).has_value();
}

void Llm::KeepBranch(Branch& branch) {
  if (kept_.keeper == nullptr || branch.history_.empty() || branch.needs_clear_ ||
      !KeptWholeFor(branch)) {
    return;
  }
  const engine::LiveState* live = KeptLiveFor(branch);
  if (live == nullptr || !live->spill_identity()) {
    return;
  }
  MemoryCharge token_charge;
  kept::Record r;
  if (!ReserveTokens(branch, r.tokens, token_charge, branch.history_.size())) {
    Unkeep(branch);
    Say(std::format("{}: a conversation record was not kept: token history memory refused",
                    name()));
    return;
  }
  r.identity = kept_.identity;
  r.slot = BranchIndex(branch);
  r.file = kept::StateFileName(r.slot);
  r.id = KeptId(*live->spill_identity());
  for (std::size_t i = 0; i < live->regions(); ++i) {
    r.regions.push_back(live->mapped_bytes(i));
    r.file_bytes += live->mapped_bytes(i);
  }
  for (const engine::LiveState::Range& range : live->used_ranges()) {
    r.extents.push_back({.region = static_cast<std::uint32_t>(range.region),
                         .index = static_cast<std::uint32_t>(range.offset / kExtent),
                         .digest = {}});
  }
  r.tokens = branch.history_;
  r.cursor = CursorFor(branch);
  const auto decoding = TurnDecodingStateFor(branch).Save();
  r.decoding.assign(decoding.begin(), decoding.end());
  r.used_unix_ms = UnixMs(branch.history_used_.at);
  for (const TurnCheckpoint& c : branch.turn_checkpoints_) {
    if (!c.file.identity() || c.file.name().empty() || c.boundary.position > r.tokens.size()) {
      continue;
    }
    const auto words = c.decoding.Save();
    r.checkpoints.push_back({.file = c.file.name(),
                             .id = KeptId(*c.file.identity()),
                             .file_bytes = c.file.file_bytes(),
                             .position = static_cast<std::uint32_t>(c.boundary.position),
                             .created_unix_ms = UnixMs(c.boundary.created),
                             .ranges = KeptRanges(c.file.ranges()),
                             .footprint = KeptRanges(c.footprint),
                             .cursor = c.cursor,
                             .decoding = {words.begin(), words.end()},
                             .digests = {}});
  }
  kept_.keeper->Keep(kept_.model, std::move(r), std::move(token_charge));
}

Status Llm::Adopt(Branch& branch, kept::Record& record) {
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  const engine::LiveState* live = KeptLiveFor(branch);
  if (live == nullptr || kept_.keeper == nullptr) {
    return Error("this model keeps no conversation across a restart");
  }
  // The slot's turn checkpoint files the branch does not hold (left out,
  // or all of them when nothing was adopted) go: nothing stays unused.
  const std::uint32_t slot = BranchIndex(branch);
  const auto remove_unused_checkpoints = [&]() {
    auto entries = platform::ListPrivateDirectory(kept_.directory);
    if (!entries) {
      return;
    }
    for (const platform::DirectoryEntry& entry : *entries) {
      const auto named = kept::ParseFileName(entry.name);
      if (named && named->kind == kept::NameOf::Kind::kCheckpoint && named->slot == slot &&
          std::ranges::none_of(branch.turn_checkpoints_, [&](const TurnCheckpoint& t) {
            return t.file.name() == entry.name;
          })) {
        std::ignore = platform::RemovePrivate(kept_.directory, entry.name.c_str());
      }
    }
  };
  // A record not adopted goes, so no later start tries it again.
  const auto refused = [&](std::string why) -> Status {
    const std::string name = kept::RecordFileName(slot);
    std::ignore = platform::RemovePrivate(kept_.directory, name.c_str());
    remove_unused_checkpoints();
    return Error(std::move(why));
  };
  // The draft depth's state is the conversation's; its relative cost is the
  // machine's, calibrated (D-103) and perhaps measured again since: the
  // model's own now, never a reason to refuse.
  const double cost = BranchDecoding(branch).relative_cost();
  const auto same = [&](const execution::AdaptiveDepth& depth) {
    return depth.maximum() == BranchDecoding(branch).maximum();
  };
  const auto loaded = execution::AdaptiveDepth::Load(record.decoding);
  if (!loaded.has_value() || !same(*loaded)) {
    return refused("its adaptive draft depth is not this model's");
  }
  const execution::AdaptiveDepth decoding = loaded->WithCost(cost);
  // Its extents as the slot's used ranges, each bounded to its layout.
  std::vector<engine::LiveState::Range> used;
  used.reserve(record.extents.size());
  for (const kept::Extent& extent : record.extents) {
    const std::uint64_t offset = std::uint64_t{extent.index} * kExtent;
    used.push_back({.region = extent.region,
                    .offset = offset,
                    .bytes = std::min(kExtent, live->bytes(extent.region) - offset)});
  }
  // Its checkpoints, each opened (one that does not is left out).
  std::vector<TurnCheckpoint> checkpoints;
  for (kept::Checkpoint& c : record.checkpoints) {
    const auto depth = execution::AdaptiveDepth::Load(c.decoding);
    if (!depth.has_value() || !same(*depth)) {
      Say(
          std::format("{}: a kept turn checkpoint at {} tokens was left out: its draft depth is "
                      "not this model's",
                      name(), c.position));
      continue;
    }
    if (auto checked =
            CheckCheckpointMetadataFor(branch, c.position, c.cursor, LiveRanges(c.footprint));
        !checked) {
      Say(std::format("{}: a kept turn checkpoint at {} was left out: {}", name(), c.position,
                      checked.error()));
      continue;
    }
    auto file = engine::CheckpointFile::Adopt(kept_.directory, c.file, LiveRanges(c.ranges));
    if (!file) {
      Say(std::format("{}: a kept turn checkpoint at {} tokens was left out: {}", name(),
                      c.position, file.error()));
      continue;
    }
    checkpoints.push_back(
        {.boundary = {.position = c.position, .created = SteadyAt(c.created_unix_ms)},
         .file = std::move(*file),
         .footprint = LiveRanges(c.footprint),
         .cursor = c.cursor,
         .decoding = depth->WithCost(cost)});
    if (const auto named = kept::ParseFileName(c.file); named) {
      checkpoint_serial_ = std::max(checkpoint_serial_, named->serial + 1);
    }
  }
  if (!ReserveTokens(branch, branch.history_, branch.history_charge_, record.tokens.size())) {
    return refused("native adopted history exceeds the execution budget");
  }
  if (auto adopted = AdoptPositionFor(branch, record.tokens, record.cursor, used); !adopted) {
    return refused(adopted.error());
  }
  branch.history_ = record.tokens;
  branch.needs_clear_ = false;
  branch.history_used_ = SteadyAt(record.used_unix_ms);
  SetCursorFor(branch, record.cursor);
  RestoreTurnDecodingStateFor(branch, decoding);
  branch.turn_checkpoints_ = std::move(checkpoints);
  remove_unused_checkpoints();
  // The record stays: it describes the slot's file as it is.
  std::erase_if(record.checkpoints, [&](const kept::Checkpoint& c) {
    return std::ranges::none_of(branch.turn_checkpoints_,
                                [&](const TurnCheckpoint& t) { return t.file.name() == c.file; });
  });
  kept_.keeper->Adopted(kept_.model, record);
  return {};
}

void Llm::ForgetDiscarded(std::uint32_t discarded) {
  for (std::uint32_t slot = 0; slot < branch_count_; ++slot) {
    if ((discarded & (std::uint32_t{1} << slot)) != 0) {
      Branch& branch = slot == 0 ? default_branch_ : *extra_branches_[slot - 1];
      Forget(branch);
      branch.needs_clear_ = false;  // its runner discarded it already
    }
  }
}

void Llm::PreserveKeptFiles() {
  const auto preserve = [](Branch& branch) {
    for (TurnCheckpoint& checkpoint : branch.turn_checkpoints_) {
      checkpoint.file.Preserve();
    }
  };
  preserve(default_branch_);
  for (std::uint32_t slot = 1; slot < branch_count_; ++slot) {
    preserve(*extra_branches_[slot - 1]);
  }
}

Status Llm::SpillIdle(Branch& branch) { return Spill(branch, false); }

Status Llm::SpillSetAside(Branch& branch) { return Spill(branch, true); }

Status Llm::Spill(Branch& branch, bool set_aside) {
  CheckIdleGeneration(branch);
  if (!set_aside && !BranchIdle(branch)) {
    return Error("a conversation in use is not spilled");
  }
  if (SpilledFor(branch)) {
    return {};
  }
  // State that may hold anything (a failed step's), or none, is dropped:
  // there is nothing exact to keep. A member set aside is cleared by its
  // caller, within its request (Branch::Clear).
  if (branch.needs_clear_ || branch.history_.empty() || !StateUsableFor(branch)) {
    return set_aside ? Error("no exact state to spill") : ReleaseIdleState(branch);
  }
  // Kept across a restart (D-105): its last verify's owed restore runs
  // first where nothing holds the stream, so the file holds it whole.
  if (!set_aside && kept_.keeper != nullptr) {
    (void)SettleIdle(branch);
  }
  // What it writes (only what changed since its last spill): the measured
  // rate's bytes.
  const std::uint64_t bytes = SpillWriteBytesFor(branch);
  const auto started = Clock::now();
  if (auto spilled = SpillFor(branch); !spilled) {
    if (set_aside || branch.HeldByContinuation()) {
      return spilled;
    }
    // Neither spilled nor known as it was: dropped, its next turn prefills.
    if (auto released = ReleaseIdleState(branch); !released) {
      return Error(std::format("{}; clearing it instead: {}", spilled.error(), released.error()));
    }
    return Error(std::format("{}: cleared instead", spilled.error()));
  }
  ++spill_stats_.spills;
  spill_stats_.spilled_bytes += bytes;
  spill_stats_.spill_seconds += Seconds(Clock::now() - started);
  KeepBranch(branch);  // its record follows, in the background
  return {};
}

Status Llm::RestoreSpilled(Branch& branch, bool& refused) {
  refused = false;
  if (!SpilledFor(branch)) {
    return {};
  }
  // Live again: its file may change from here on (D-105).
  Unkeep(branch);
  const std::uint64_t bytes = SpilledBytesFor(branch);
  const auto started = Clock::now();
  if (auto restored = RestoreFor(branch); !restored) {
    refused = StateRefusedFor(branch) && StateUsableFor(branch);
    return restored;
  }
  ++spill_stats_.restores;
  spill_stats_.restored_bytes += bytes;
  spill_stats_.restore_seconds += Seconds(Clock::now() - started);
  return {};
}

bool Llm::CapacityRefused(const Branch& branch) const {
  return StateRefusedFor(branch) && StateUsableFor(branch);
}

bool Llm::ReclaimFor(const Branch& branch) {
  return capacity_reclaim_ && capacity_reclaim_(branch);
}

bool Llm::sampling(const Branch& branch) const {
  CheckBranch(branch);
  return branch.sampling_.has_value();
}

execution::AdaptiveDepth& Llm::BranchDecoding(Branch& branch) {
  CheckBranch(branch);
  return branch.decoding_;
}

const execution::AdaptiveDepth& Llm::BranchDecoding(const Branch& branch) const {
  CheckBranch(branch);
  return branch.decoding_;
}

void Llm::SaveBranchDecoding(Branch& branch) {
  CheckBranch(branch);
  branch.saved_decoding_ = branch.decoding_;
}

void Llm::RestoreBranchDecoding(Branch& branch) {
  CheckBranch(branch);
  branch.decoding_ = branch.saved_decoding_;
}

Status Llm::RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t n_past,
                        bool inject, std::vector<float>& logits) {
  CheckDefaultBranch(branch);
  return RunChunk(all, n_past, inject, logits);
}

Status Llm::RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all,
                               std::uint32_t n_past, bool inject, bool /*want_head*/,
                               std::vector<float>& logits, PrefillHint /*next*/) {
  return RunChunkFor(branch, all, n_past, inject, logits);
}

Status Llm::SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                        std::uint32_t left, std::vector<std::int32_t>& kept,
                        std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                        bool* /*prefix_kept*/) {
  CheckDefaultBranch(branch);
  return SpecStep(all, pos, left, kept, logits, drafted);
}

Status Llm::SettleFor(Branch& branch) {
  CheckDefaultBranch(branch);
  return Settle();
}

Status Llm::ClearStateFor(Branch& branch) {
  CheckDefaultBranch(branch);
  return ClearState();
}

bool Llm::StateUsableFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return StateUsable();
}

Status Llm::PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) {
  CheckDefaultBranch(branch);
  return PrepareDecodeState(pos, left);
}

std::uint64_t Llm::TargetStateBaseFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return target_state_base();
}

std::uint64_t Llm::TargetStateBytesFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return target_state_bytes();
}

std::uint64_t Llm::DrafterStateBaseFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return drafter_state_base();
}

std::uint64_t Llm::DrafterStateBytesFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return drafter_state_bytes();
}

std::uint64_t Llm::UsedStateBytesFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return used_state_bytes();
}

std::vector<engine::LiveState::Range> Llm::UsedStateRangesFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return used_state_ranges();
}

Status Llm::SaveUsedStateFor(Branch& branch, void* host,
                             std::span<const engine::LiveState::Range> ranges) {
  CheckDefaultBranch(branch);
  return SaveUsedState(host, ranges);
}

Status Llm::RestoreUsedStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges) {
  CheckDefaultBranch(branch);
  return RestoreUsedState(host, ranges);
}

std::expected<std::vector<engine::LiveState::Range>, std::string> Llm::CheckpointRangesFor(
    const Branch& branch, std::uint32_t positions) const {
  CheckDefaultBranch(branch);
  return CheckpointRanges(positions);
}

Status Llm::PrepareRestoreStateFor(Branch& branch,
                                   std::span<const engine::LiveState::Range> footprint) {
  CheckDefaultBranch(branch);
  return PrepareRestoreState(footprint);
}

Status Llm::CopyCheckpointStateFor(Branch& branch, void* host,
                                   std::span<const engine::LiveState::Range> ranges, bool to_host) {
  CheckDefaultBranch(branch);
  return CopyCheckpointState(host, ranges, to_host);
}

std::uint32_t Llm::CursorFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return cursor();
}

void Llm::SetCursorFor(Branch& branch, std::uint32_t value) {
  CheckDefaultBranch(branch);
  set_cursor(value);
}

void Llm::SaveDecodingStateFor(Branch& branch) {
  CheckDefaultBranch(branch);
  SaveDecodingState();
}

void Llm::RestoreDecodingStateFor(Branch& branch) {
  CheckDefaultBranch(branch);
  RestoreDecodingState();
}

execution::AdaptiveDepth Llm::TurnDecodingStateFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return TurnDecodingState();
}

void Llm::RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) {
  CheckDefaultBranch(branch);
  RestoreTurnDecodingState(state);
}

void Llm::CheckIdleGeneration(const Branch& branch, const PromptSession* prompt) const {
  CheckBranch(branch);
  base::Check(!branch.generation_active_, "mutating a conversation with an active generation");
  base::Check(branch.prompt_session_ == prompt,
              "mutating a conversation with another active prompt");
}

void Llm::Forget(Branch& branch, const PromptSession* prompt) {
  CheckIdleGeneration(branch, prompt);
  Unkeep(branch);
  branch.turn_checkpoints_.clear();
  ReleaseTokens(branch.history_, branch.history_charge_);
  ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
  branch.needs_clear_ = true;
}

Status Llm::Clear(Branch& branch, const PromptSession* prompt) {
  CheckIdleGeneration(branch, prompt);
  Unkeep(branch);
  branch.turn_checkpoints_.clear();
  if (auto r = ClearStateFor(branch); !r) {
    return r;
  }
  ReleaseTokens(branch.history_, branch.history_charge_);
  ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
  branch.needs_clear_ = false;
  branch.history_used_ = Clock::now();
  return {};
}

Status Llm::ReleaseIdleState(Branch& branch) {
  if (branch.HeldByContinuation()) {
    return Error("an in-flight continuation holds this conversation's state");
  }
  CheckIdleGeneration(branch);
  Unkeep(branch);
  branch.turn_checkpoints_.clear();
  ReleaseTokens(branch.history_, branch.history_charge_);
  ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
  if (auto released = ReleaseIdleStateFor(branch); !released) {
    branch.needs_clear_ = true;  // its next use clears whatever is left
    return released;
  }
  branch.needs_clear_ = false;
  branch.history_used_ = Clock::now();
  return {};
}

std::size_t Llm::ExpireTurnCheckpoints(Branch& branch, Clock::time_point now) {
  if (!BranchIdle(branch) || branch.HeldByContinuation()) {
    return 0;
  }
  const std::size_t expired = std::erase_if(branch.turn_checkpoints_, [&](const TurnCheckpoint& c) {
    return now - c.boundary.created >= retention_;
  });
  // Their files went with them; a kept record names them no more.
  if (expired != 0 && kept_.keeper != nullptr &&
      kept_.keeper->Kept(kept_.model, BranchIndex(branch))) {
    KeepBranch(branch);
  }
  return expired;
}

std::uint64_t Llm::Branch::turn_checkpoint_bytes() const {
  std::uint64_t bytes = 0;
  for (const TurnCheckpoint& checkpoint : turn_checkpoints_) {
    bytes += checkpoint.file.bytes();
  }
  return bytes;
}

Status Llm::CaptureTurnCheckpoint(Branch& branch, const PrefillGoOn& go_on, bool& stopped,
                                  const PromptSession* prompt) {
  if (branch.history_.empty()) {
    return {};
  }
  const std::size_t position = branch.history_.size();
  if (std::ranges::any_of(branch.turn_checkpoints_, [&](const TurnCheckpoint& c) {
        return c.boundary.position == position;
      })) {
    return {};  // looking it up does not renew its retention period
  }
  if (auto settled = SettleFor(branch); !settled) {
    return settled;
  }
  auto ranges = CheckpointRangesFor(branch, static_cast<std::uint32_t>(position));
  if (!ranges) {
    return std::unexpected(ranges.error());
  }
  const engine::CheckpointFile::Continue progress =
      go_on ? engine::CheckpointFile::Continue([&]() { return go_on(0); })
            : engine::CheckpointFile::Continue{};
  // Named, beneath the model's private directory, when conversations are
  // kept across a restart (D-105): a record may name it.
  engine::CheckpointFile::Place place{.directory = checkpoint_directory_, .dir = -1, .name = {}};
  if (kept_.keeper != nullptr) {
    place.dir = kept_.directory;
    place.name = kept::CheckpointFileName(BranchIndex(branch), checkpoint_serial_++);
  }
  auto file = engine::CheckpointFile::Capture(
      *node_, place, *ranges,
      [&](void* host, std::span<const engine::LiveState::Range> page) {
        return CopyCheckpointStateFor(branch, host, page, true);
      },
      progress);
  if (!file) {
    if (file.error().invalid_state) {
      Forget(branch, prompt);
      return Error("saving a turn checkpoint: " + file.error().detail);
    }
    stopped = file.error().cancelled;
    if (!stopped) {
      // Positions only (D-014): the next turn that would continue from it
      // prefills instead.
      Say(std::format("{}: a turn checkpoint at {} tokens was not saved: {}", name(), position,
                      file.error().detail));
    }
    return {};  // optional cache: unavailable staging or disk does not fail the turn
  }
  TurnCheckpoint checkpoint{.boundary = {.position = position, .created = Clock::now()},
                            .file = std::move(*file),
                            .footprint = UsedStateRangesFor(branch),
                            .cursor = CursorFor(branch),
                            .decoding = TurnDecodingStateFor(branch)};
  branch.turn_checkpoints_.push_back(std::move(checkpoint));
  if (branch.turn_checkpoints_.size() > kTurnCheckpointLimit) {
    branch.turn_checkpoints_.erase(branch.turn_checkpoints_.begin());
  }
  return {};
}

std::uint32_t Llm::ReusableFor(const Branch& branch, std::span<const std::int32_t> tokens,
                               bool fresh, bool resume) const {
  if (fresh) {
    return 0;
  }
  const std::size_t prefix = ReusablePrefix(branch, tokens);
  // A generation set aside for its peers resumes from its whole prompt.
  if (prefix == 0 && resume && !branch.history_.empty() &&
      branch.history_.size() == tokens.size() && !branch.needs_clear_ && StateUsableFor(branch) &&
      CommonPrefix(branch.history_, tokens) == tokens.size()) {
    return static_cast<std::uint32_t>(tokens.size());
  }
  return static_cast<std::uint32_t>(prefix);
}

Status Llm::ReusePrompt(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t& reused,
                        bool fresh, bool resume, const PrefillGoOn& go_on, bool& stopped,
                        const PromptSession* prompt) {
  reused = 0;
  const auto now = Clock::now();
  if (fresh || branch.needs_clear_ || !StateUsableFor(branch) ||
      (!branch.HeldByContinuation() && now - branch.history_used_.at >= retention_)) {
    return Clear(branch, prompt);
  }
  std::erase_if(branch.turn_checkpoints_, [&](const TurnCheckpoint& checkpoint) {
    return now - checkpoint.boundary.created >= retention_;
  });
  const std::size_t common = CommonPrefix(branch.history_, tokens);
  // The live state continues the prompt (with `resume`, it may be the
  // whole prompt: a generation set aside for its peers resumes from it).
  const bool live = !branch.history_.empty() && common == branch.history_.size() &&
                    (common < tokens.size() || (resume && common == tokens.size()));
  std::vector<TurnBoundary> boundaries;
  boundaries.reserve(branch.turn_checkpoints_.size());
  for (const TurnCheckpoint& checkpoint : branch.turn_checkpoints_) {
    boundaries.push_back(checkpoint.boundary);
  }
  const auto match = live
                         ? std::nullopt
                         : MatchingTurnBoundary(boundaries, common, tokens.size(), now, retention_);
  if (!live && !match) {
    return Clear(branch, prompt);  // a spilled state is discarded with it
  }
  // Spill, not clear: an idle conversation's state spilled for capacity
  // comes back exactly as it left before it is reused (refused for
  // capacity: the unit runs again once the runtime made room).
  if (bool refused = false; SpilledFor(branch)) {
    if (auto restored = RestoreSpilled(branch, refused); !restored) {
      if (refused) {
        return restored;  // the caller defers it (PromptSession::Advance)
      }
      Say(
          std::format("{}: a spilled conversation of {} tokens was not restored ({}): the "
                      "prompt is prefilled from its start",
                      name(), branch.history_.size(), restored.error()));
      return Clear(branch, prompt);
    }
  }
  if (live) {
    reused = static_cast<std::uint32_t>(common);
    branch.history_used_ = now;
    return {};
  }
  TurnCheckpoint& checkpoint = branch.turn_checkpoints_[*match];
  const auto position = static_cast<std::uint32_t>(checkpoint.boundary.position);
  const auto restored_cursor = checkpoint.cursor;
  const auto restored_decoding = checkpoint.decoding;
  if (auto settled = SettleFor(branch); !settled) {
    Forget(branch, prompt);
    return settled;
  }
  const engine::CheckpointFile::Continue progress =
      go_on ? engine::CheckpointFile::Continue([&]() { return go_on(0); })
            : engine::CheckpointFile::Continue{};
  auto restored = checkpoint.file.Restore(
      *node_,
      [&]() {
        return PreparePositionRestoreFor(branch, position, restored_cursor, checkpoint.footprint);
      },
      [&](void* host, std::span<const engine::LiveState::Range> page) {
        return CopyCheckpointStateFor(branch, host, page, false);
      },
      progress);
  if (!restored) {
    if (StateRefusedFor(branch) && StateUsableFor(branch)) return Error(restored.error().detail);
    if (restored.error().cancelled) {
      stopped = true;
      return restored.error().invalid_state ? Clear(branch, prompt) : Status{};
    }
    Say(
        std::format("{}: the turn checkpoint at {} tokens was not restored ({}): the prompt is "
                    "prefilled from its start",
                    name(), position, restored.error().detail));
    // Before mutation this is an optional cache miss. After mutation, Clear
    // drops the branch before any fresh prefill; an uncertain device copy
    // may make Clear fail, in which case the runtime stops normally.
    return Clear(branch, prompt);
  }
  if (auto completed = CompletePositionRestoreFor(branch, position); !completed) {
    Forget(branch, prompt);
    return completed;
  }
  branch.history_.resize(position);
  SetCursorFor(branch, restored_cursor);
  RestoreTurnDecodingStateFor(branch, restored_decoding);
  std::erase_if(branch.turn_checkpoints_,
                [&](const TurnCheckpoint& c) { return c.boundary.position > position; });
  reused = static_cast<std::uint32_t>(position);
  branch.needs_clear_ = false;
  branch.history_used_ = now;
  return {};
}

Status Llm::PreparePrompt(Branch& branch, std::span<const std::int32_t> tokens,
                          std::uint32_t stable_boundary, std::vector<float>& last,
                          std::uint32_t& reused, const PrefillGoOn& go_on, PrefillRun* run,
                          bool fresh) {
  CheckIdleGeneration(branch);
  branch.capacity_refused_ = false;
  auto opened = BeginPrompt(branch, tokens, stable_boundary, fresh);
  if (!opened) {
    return std::unexpected(opened.error());
  }
  auto& session = **opened;
  std::size_t reclaimed = 0;
  while (!session.done()) {
    auto advanced = session.Advance(go_on, true);
    if (advanced) {
      reclaimed = 0;
      continue;
    }
    if (session.refused()) {
      // The same unit again once capacity was freed; otherwise it ends at
      // its completed prefix, as an undeferred refusal does.
      if (reclaimed < kMaxBranches && ReclaimFor(branch)) {
        ++reclaimed;
        continue;
      }
      branch.capacity_refused_ = true;
      [[maybe_unused]] const auto failed = session.Fail(std::move(advanced.error()));
    }
    break;
  }
  auto result = session.Finish();
  last = std::move(session.last_);
  reused = session.reused();
  if (run != nullptr) {
    *run = session.run();
  }
  return result;
}

Llm::PromptSession::PromptSession(Llm& model, Branch& branch, std::uint32_t stable_boundary,
                                  bool fresh, bool resume, bool scoring,
                                  std::function<bool(std::int32_t, std::span<const float>)> on_row)
    : model_(model),
      branch_(branch),
      stable_boundary_(stable_boundary),
      fresh_(fresh),
      resume_(resume),
      scoring_(scoring),
      on_row_(std::move(on_row)) {}

Llm::PromptSession::~PromptSession() {
  base::Check(finished_, "an unfinished prompt session was destroyed");
}

std::expected<std::unique_ptr<Llm::PromptSession>, std::string> Llm::BeginPrompt(
    Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t stable_boundary, bool fresh,
    bool resume, bool scoring, std::function<bool(std::int32_t, std::span<const float>)> on_row) {
  CheckBranch(branch);
  if (tokens.empty() || tokens.size() > context_ || stable_boundary >= tokens.size()) {
    return Error("the prompt or its turn boundary is outside the context");
  }
  if (max_rows_ == 0) {
    return Error("a prompt needs a positive prefill chunk size");
  }
  if (branch.generation_active_ || branch.prompt_session_ != nullptr) {
    return Error("the conversation already has an active session");
  }
  auto session = std::unique_ptr<PromptSession>(
      new PromptSession(*this, branch, stable_boundary, fresh, resume, scoring, std::move(on_row)));
  branch.prompt_session_ = session.get();
  if (!ReserveTokens(branch, session->tokens_, session->tokens_charge_, tokens.size())) {
    session->Cancel();
    (void)session->Finish();
    return Error("native prompt history exceeds the execution budget");
  }
  session->tokens_.assign(tokens.begin(), tokens.end());
  return session;
}

bool Llm::PromptSession::done() const { return complete_ || finished_; }

bool Llm::PromptSession::started() const { return phase_ != Phase::kReuse; }

std::expected<Llm::PromptSession::Unit, std::string> Llm::PromptSession::NextUnit() const {
  if (done() || advancing_) {
    return Error("the prompt has no next unit");
  }
  Unit unit{.phase = phase_};
  if (phase_ == Phase::kChunk) {
    const auto at = static_cast<std::uint32_t>(branch_.history_.size());
    const auto end =
        checkpoint_pending_ ? stable_boundary_ : static_cast<std::uint32_t>(tokens_.size());
    base::Check(at < end, "a prompt chunk has no rows");
    unit.rows = scoring_ ? 1 : PrefillRows(end - at, model_.max_rows_);
    unit.want_head = scoring_ || at + unit.rows == tokens_.size();
  }
  return unit;
}

std::uint32_t Llm::PromptSession::remaining_rows() const {
  if (done()) {
    return 0;
  }
  std::size_t at = branch_.history_.size();
  if (phase_ == Phase::kReuse) {
    // Not reused yet: what the reuse would keep (none of a stale history
    // that merely shares a few tokens with the prompt).
    at = model_.ReusableFor(branch_, tokens_, fresh_, resume_);
  }
  return at >= tokens_.size() ? 0 : static_cast<std::uint32_t>(tokens_.size() - at);
}

void Llm::PromptSession::NextPhase() {
  const auto at = branch_.history_.size();
  if (checkpoint_pending_ && at == stable_boundary_) {
    phase_ = Phase::kCheckpoint;
  } else if (at < tokens_.size()) {
    phase_ = Phase::kChunk;
  } else {
    complete_ = true;
  }
}

Status Llm::PromptSession::Fail(std::string error) {
  branch_.needs_clear_ = branch_.needs_clear_ || !model_.StateUsableFor(branch_);
  if (branch_.needs_clear_) {
    model_.ReleaseTokens(branch_.history_, branch_.history_charge_);
  }
  last_.clear();
  complete_ = true;
  ran_ = std::unexpected(std::move(error));
  return ran_;
}

bool Llm::PromptSession::CanJoin(const PromptSession& peer) const {
  if (&model_ != &peer.model_ || this == &peer || scoring_ || peer.scoring_ || model_.speculate_)
    return false;
  const auto a = NextUnit(), b = peer.NextUnit();
  return a && b && a->phase == Phase::kChunk && b->phase == Phase::kChunk &&
         a->want_head == b->want_head &&
         model_.CompatiblePrefill(static_cast<std::uint32_t>(branch_.history_.size()), a->rows,
                                  static_cast<std::uint32_t>(peer.branch_.history_.size()),
                                  b->rows);
}

std::optional<std::size_t> Llm::PromptSession::SelectForWave(
    bool decode, std::optional<std::size_t> chosen, std::span<PromptSession* const> sessions) {
  if (decode || !chosen || *chosen >= sessions.size() || sessions[*chosen] == nullptr)
    return std::nullopt;
  const auto& anchor = *sessions[*chosen];
  const auto unit = anchor.NextUnit();
  if (!unit || unit->phase != Phase::kChunk || anchor.scoring_ || anchor.model_.speculate_ ||
      anchor.model_.prefill_wave_capacity() <= 1)
    return chosen;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    const auto* peer = sessions[i];
    if (peer == nullptr || peer == &anchor || &peer->model_ != &anchor.model_ || peer->scoring_)
      continue;
    const auto next = peer->NextUnit();
    if (!next || (next->phase != Phase::kReuse && next->phase != Phase::kCheckpoint)) continue;
    const auto remaining = peer->remaining_rows();
    if (remaining == 0) continue;
    const auto at = static_cast<std::uint32_t>(peer->tokens_.size()) - remaining;
    // Reuse is a hint until its ordinary unit actually settles. A cache
    // miss may change this geometry; CanJoin checks the resulting chunks.
    const auto end = next->phase == Phase::kReuse && peer->stable_boundary_ > at
                         ? peer->stable_boundary_
                         : static_cast<std::uint32_t>(peer->tokens_.size());
    const auto rows = PrefillRows(end - at, peer->model_.max_rows_);
    const bool want_head = at + rows == peer->tokens_.size();
    if (unit->want_head == want_head &&
        anchor.model_.CompatiblePrefill(static_cast<std::uint32_t>(anchor.branch_.history_.size()),
                                        unit->rows, at, rows))
      return i;
  }
  return chosen;
}

Status Llm::PromptSession::PrepareChunk(std::uint32_t rows, bool defer_capacity) {
  const auto end = branch_.history_.size() + rows;
  if (!model_.ReserveTokens(branch_, branch_.history_, branch_.history_charge_, end)) {
    refused_ = defer_capacity;
    if (defer_capacity) return Error("native retained history exceeds the execution budget");
    return Fail("native retained history exceeds the execution budget");
  }
  return {};
}

Status Llm::PromptSession::CompleteChunk(std::uint32_t rows, double wall_seconds,
                                         double accounted_seconds, Status result,
                                         bool defer_capacity) {
  const auto at = static_cast<std::uint32_t>(branch_.history_.size());
  const auto end = at + rows;
  if (!result) {
    std::string error =
        std::format("{}'s prefill: the chunk at {}: {}", model_.name_, at, result.error());
    if (defer_capacity && model_.StateRefusedFor(branch_) && model_.StateUsableFor(branch_)) {
      refused_ = true;
      return std::unexpected(std::move(error));
    }
    return Fail(std::move(error));
  }
  branch_.history_.insert(branch_.history_.end(), tokens_.begin() + at, tokens_.begin() + end);
  branch_.history_used_ = Clock::now();
  run_.end = end;
  ++run_.chunks;
  run_.longest = std::max(run_.longest, wall_seconds);
  if (!scoring_) model_.calibration_samples_.PrefillChunk(at, rows, accounted_seconds);
  from_zero_ = from_zero_ || at == 0;
  chunk_seconds_ += accounted_seconds;
  if (!scoring_ && from_zero_ && end == tokens_.size())
    model_.calibration_samples_.Prefill(end, chunk_seconds_);
  if (scoring_ && end < tokens_.size() && on_row_ && !on_row_(tokens_[end], last_)) Stop();
  return {};
}

Status Llm::RunPreparedPrefillWave(std::span<PreparedPrefill> prepared) {
  for (auto& unit : prepared) {
    unit.result = RunPrefillChunkFor(*unit.branch, unit.all, unit.past, speculate_, unit.want_head,
                                     *unit.logits);
    if (!unit.result && !GenerationCohortUsable()) return unit.result;
  }
  return {};
}

Status Llm::RunPromptWave(std::span<PromptSession* const> sessions,
                          std::span<const PrefillGoOn> go_on, bool defer_capacity) {
  if (sessions.empty() || sessions.size() > kMaxBranches ||
      sessions.size() > prefill_wave_capacity() || go_on.size() != sessions.size())
    return Error("the prompt cohort exceeds this model's capability");
  std::array<bool, kMaxBranches> seen{};
  std::array<PromptSession::Unit, kMaxBranches> units{};
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    const auto* session = sessions[i];
    if (!session || &session->model_ != this || session->done() || session->advancing_ ||
        session->scoring_ || speculate_)
      return Error("a prompt cohort contains an inactive, scoring or foreign session");
    const auto next = session->NextUnit();
    if (!next || next->phase != PromptSession::Phase::kChunk ||
        (i != 0 && !sessions.front()->CanJoin(*session)))
      return Error("a prompt cohort needs compatible plain chunks");
    const auto id = BranchIndex(session->branch_);
    if (id >= seen.size() || seen[id]) return Error("a prompt cohort repeats a native branch");
    seen[id] = true;
    units[i] = *next;
  }
  // Keep every session borrowed through dispatch and all peer publication.
  // The HTTP driver may Finish/Cancel/BeginGeneration only after this returns.
  struct CompletedWave {
    std::span<PromptSession* const> sessions;
    ~CompletedWave() {
      for (auto* session : sessions) session->advancing_ = false;
    }
  } completed{sessions};
  for (auto* session : sessions) {
    session->refused_ = false;
    session->unit_result_ = {};
    session->advancing_ = true;
  }
  std::array<PreparedPrefill, kMaxBranches> storage{};
  std::size_t count = 0;
  const auto fail_cohort = [&](const std::string& error) {
    for (auto* session : sessions) {
      session->unit_result_ = Error(error);
      session->branch_.needs_clear_ = true;
      (void)session->Fail(error);
    }
  };
  const auto started = Clock::now();
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    auto& session = *sessions[i];
    const auto& next = units[i];
    CheckIdleGeneration(session.branch_, &session);
    if (go_on[i] && !go_on[i](next.rows)) {
      session.Stop();
      continue;
    }
    if (auto funded = session.PrepareChunk(next.rows, defer_capacity); !funded) {
      session.unit_result_ = std::move(funded);
      continue;
    }
    const auto at = static_cast<std::uint32_t>(session.branch_.history_.size());
    if (auto state = PreparePrefillStateFor(session.branch_, at, next.rows); !state) {
      session.unit_result_ =
          session.CompleteChunk(next.rows, 0, 0, std::move(state), defer_capacity);
      if (!GenerationCohortUsable()) {
        const std::string error = "the native prompt cohort became unusable during preparation";
        fail_cohort(error);
        return Error(error);
      }
      continue;
    }
    storage[count++] = {.session = &session,
                        .branch = &session.branch_,
                        .all = std::span(session.tokens_).first(at + next.rows),
                        .past = at,
                        .rows = next.rows,
                        .want_head = next.want_head,
                        .logits = &session.last_,
                        .result = {}};
  }
  auto prepared = std::span(storage).first(count);
  if (prepared.empty()) return {};
  std::ranges::sort(prepared, {}, [this](const auto& unit) { return BranchIndex(*unit.branch); });
  const StateKeeper::Quiet quiet(kept_.keeper);
  if (auto ran = RunPreparedPrefillWave(prepared); !ran) {
    fail_cohort(ran.error());
    return ran;
  }
  if (!GenerationCohortUsable()) {
    const std::string error = "the native prompt cohort became unusable after dispatch";
    fail_cohort(error);
    return Error(error);
  }
  const double seconds = Seconds(Clock::now() - started);
  std::uint32_t total_rows = 0;
  for (const auto& unit : prepared) total_rows += unit.rows;
  for (auto& unit : prepared) {
    // Wall latency belongs to every participant; calibration apportions the
    // shared cost by actual rows rather than charging the same work twice.
    const double attributed = seconds * unit.rows / total_rows;
    unit.session->unit_result_ = unit.session->CompleteChunk(
        unit.rows, seconds, attributed, std::move(unit.result), defer_capacity);
    if (unit.session->unit_result_) unit.session->NextPhase();
  }
  return {};
}

Status Llm::PromptSession::Advance(const PrefillGoOn& go_on, bool defer_capacity) {
  refused_ = false;
  auto next = NextUnit();
  if (!next) {
    return std::unexpected(next.error());
  }
  model_.CheckIdleGeneration(branch_, this);
  advancing_ = true;
  struct CompletedUnit {
    explicit CompletedUnit(bool& value) noexcept : advancing(value) {}
    CompletedUnit(const CompletedUnit&) = delete;
    CompletedUnit& operator=(const CompletedUnit&) = delete;
    CompletedUnit(CompletedUnit&&) = delete;
    CompletedUnit& operator=(CompletedUnit&&) = delete;
    bool& advancing;
    ~CompletedUnit() { advancing = false; }
  } completed{advancing_};
  if (go_on && !go_on(next->rows)) {
    Stop();
    return {};
  }
  if (next->phase == Phase::kReuse) {
    auto reused =
        model_.ReusePrompt(branch_, tokens_, reused_, fresh_, resume_, go_on, run_.stopped, this);
    if (!reused) {
      if (defer_capacity && model_.SpilledFor(branch_) && model_.StateRefusedFor(branch_) &&
          model_.StateUsableFor(branch_)) {
        // Its spilled state's restore refused for capacity, before any
        // change: the same unit again once there is room.
        refused_ = true;
        return std::unexpected(reused.error());
      }
      return Fail(reused.error());
    }
    run_.end = reused_;
    if (run_.stopped) {
      Stop();
      return {};
    }
    checkpoint_pending_ = stable_boundary_ != 0 && stable_boundary_ >= branch_.history_.size();
  } else if (next->phase == Phase::kCheckpoint) {
    auto captured = model_.CaptureTurnCheckpoint(branch_, go_on, run_.stopped, this);
    if (!captured) {
      return Fail(captured.error());
    }
    if (run_.stopped) {
      Stop();
      return {};
    }
    checkpoint_pending_ = false;
  } else {
    const auto at = static_cast<std::uint32_t>(branch_.history_.size());
    const auto end = at + next->rows;
    if (auto prepared = PrepareChunk(next->rows, defer_capacity); !prepared) return prepared;
    PrefillHint hint;
    const auto boundary =
        checkpoint_pending_ ? stable_boundary_ : static_cast<std::uint32_t>(tokens_.size());
    if (end < boundary) {
      hint.rows = scoring_ ? 1U : PrefillRows(boundary - end, model_.max_rows_);
      hint.want_head = scoring_ || end + hint.rows == tokens_.size();
      if (const auto after = end + hint.rows; after < boundary) {
        hint.after_rows = scoring_ ? 1U : PrefillRows(boundary - after, model_.max_rows_);
        hint.after_want_head = scoring_ || after + hint.after_rows == tokens_.size();
      }
    }
    const auto started = Clock::now();
    const StateKeeper::Quiet quiet(model_.kept_.keeper);  // records' hashing waits
    auto chunk =
        model_.RunPrefillChunkFor(branch_, std::span(tokens_).first(end), at, model_.speculate_,
                                  scoring_ || end == tokens_.size(), last_, hint);
    const double seconds = Seconds(Clock::now() - started);
    if (auto completed_chunk =
            CompleteChunk(next->rows, seconds, seconds, std::move(chunk), defer_capacity);
        !completed_chunk)
      return completed_chunk;
  }
  NextPhase();
  return {};
}

void Llm::PromptSession::Cancel() {
  base::Check(!advancing_, "cancelling a prompt before its unit completed");
  Stop();
}

void Llm::PromptSession::Stop() {
  if (finished_) {
    return;
  }
  run_.stopped = true;
  last_.clear();
  complete_ = true;
}

Status Llm::PromptSession::Finish() {
  base::Check(!advancing_, "finishing a prompt before its unit completed");
  if (!done()) {
    return Error("a prompt still has unprocessed units");
  }
  if (!finished_) {
    base::Check(branch_.prompt_session_ == this, "a prompt lost its branch ownership");
    if (scoring_ && ran_ && run_.chunks != 0) {
      if (auto settled = model_.SettleFor(branch_); !settled) {
        branch_.needs_clear_ = true;
        model_.ReleaseTokens(branch_.history_, branch_.history_charge_);
        last_.clear();
        ran_ = std::move(settled);
      }
    }
    model_.ReleaseTokens(tokens_, tokens_charge_);
    branch_.prompt_session_ = nullptr;
    finished_ = true;
  }
  return ran_;
}

Status Llm::Prefill(Branch& branch, std::span<const std::int32_t> tokens, std::vector<float>& last,
                    const PrefillGoOn& go_on, PrefillRun* run) {
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  // Records' hashing waits while the prompt prefills (StateKeeper::Quiet).
  const StateKeeper::Quiet quiet(kept_.keeper);
  branch.capacity_refused_ = false;
  if (branch.needs_clear_) {
    if (auto r = Clear(branch); !r) {
      return r;
    }
  }
  if (tokens.empty()) {
    return Error("nothing to prefill");
  }
  const std::size_t total = branch.history_.size() + tokens.size();
  if (total > context_) {
    return Error("the prompt exceeds the context");
  }
  MemoryCharge all_charge;
  std::vector<std::int32_t> all;
  if (!ReserveTokens(branch, all, all_charge, total)) {
    branch.capacity_refused_ = true;
    return Error("native prefill history exceeds the execution budget");
  }
  all.insert(all.end(), branch.history_.begin(), branch.history_.end());
  all.insert(all.end(), tokens.begin(), tokens.end());
  if (all.size() > context_) {
    return Error(
        std::format("{} tokens do not fit {}'s context of {}", all.size(), name_, context_));
  }
  last.clear();
  auto completed = static_cast<std::uint32_t>(branch.history_.size());
  const bool from_zero = branch.history_.empty();
  double chunk_seconds = 0;
  auto ran = RunPrefillChunks(
      static_cast<std::uint32_t>(branch.history_.size()), static_cast<std::uint32_t>(all.size()),
      max_rows_,
      [&](std::uint32_t at, std::uint32_t n) {
        const auto remaining = static_cast<std::uint32_t>(all.size()) - at - n;
        const auto next_rows = PrefillRows(remaining, max_rows_);
        const auto after_rows = PrefillRows(remaining - next_rows, max_rows_);
        const PrefillHint hint{next_rows, next_rows == remaining, after_rows,
                               after_rows == remaining - next_rows};
        auto started = Clock::now();
        auto chunk = RunPrefillChunkFor(branch, std::span(all).first(at + n), at, speculate_,
                                        at + n == all.size(), last, hint);
        // A capacity refusal ran nothing: the same chunk again once freed.
        for (std::size_t reclaimed = 0;
             !chunk && CapacityRefused(branch) && reclaimed < kMaxBranches && ReclaimFor(branch);
             ++reclaimed) {
          started = Clock::now();
          chunk = RunPrefillChunkFor(branch, std::span(all).first(at + n), at, speculate_,
                                     at + n == all.size(), last, hint);
        }
        if (chunk) {
          completed = at + n;
          // Its speed, and a whole prefill's cost a token (calibration.h).
          const double seconds = Seconds(Clock::now() - started);
          calibration_samples_.PrefillChunk(at, n, seconds);
          chunk_seconds += seconds;
          if (from_zero && completed == all.size()) {
            calibration_samples_.Prefill(completed, chunk_seconds);
          }
        } else {
          branch.capacity_refused_ = CapacityRefused(branch);
        }
        return chunk;
      },
      go_on);
  if (!ran) {
    branch.needs_clear_ = !StateUsableFor(branch);
    if (branch.needs_clear_) {
      ReleaseTokens(branch.history_, branch.history_charge_);
      ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
    } else {
      PublishTokens(branch, all, all_charge, completed);
    }
    ReleaseTokens(all, all_charge);
    last.clear();
    return Error(std::format("{}'s prefill: {}", name_, ran.error()));
  }
  if (run != nullptr) {
    *run = *ran;
  }
  // The state holds exactly the chunks that ran: stopped between chunks,
  // the history is their prefix (which the next prefill continues from,
  // through the same chunk boundaries), and there are no logits to
  // generate from.
  PublishTokens(branch, all, all_charge, ran->end);
  branch.history_used_ = Clock::now();
  if (ran->stopped) {
    last.clear();
  }
  return {};
}

Status Llm::ScorePrompt(Branch& branch, std::span<const std::int32_t> tokens,
                        std::vector<float>& last,
                        const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                        const PrefillGoOn& go_on, PrefillRun* run) {
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  branch.capacity_refused_ = false;
  if (!branch.history_.empty() || tokens.empty() || tokens.size() > context_) {
    return Error("literal scoring needs an empty history and a nonempty prompt within context");
  }
  if (branch.needs_clear_) {
    if (auto r = Clear(branch); !r) {
      return r;
    }
  }
  PrefillRun completed;
  last.clear();
  for (std::uint32_t at = 0; at < tokens.size(); ++at) {
    if (go_on && !go_on(1)) {
      completed.stopped = true;
      break;
    }
    if (!ReserveTokens(branch, branch.history_, branch.history_charge_, at + 1)) {
      branch.capacity_refused_ = true;
      return Error("native scoring history exceeds the execution budget");
    }
    const auto started = Clock::now();
    auto ran = RunChunkFor(branch, tokens.first(std::size_t{at} + 1), at, speculate_, last);
    // A capacity refusal ran nothing: the same row again once freed.
    for (std::size_t reclaimed = 0;
         !ran && CapacityRefused(branch) && reclaimed < kMaxBranches && ReclaimFor(branch);
         ++reclaimed) {
      ran = RunChunkFor(branch, tokens.first(std::size_t{at} + 1), at, speculate_, last);
    }
    if (!ran) {
      branch.capacity_refused_ = CapacityRefused(branch);
      branch.needs_clear_ = !StateUsableFor(branch);
      if (branch.needs_clear_) {
        ReleaseTokens(branch.history_, branch.history_charge_);
        ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
      }
      last.clear();
      return Error(std::format("{}'s scoring: the row at {}: {}", name_, at, ran.error()));
    }
    branch.history_.push_back(tokens[at]);
    completed.end = at + 1;
    ++completed.chunks;
    completed.longest = std::max(completed.longest, Seconds(Clock::now() - started));
    if (at + 1 < tokens.size() && on_row && !on_row(tokens[at + 1], last)) {
      completed.stopped = true;
      break;
    }
  }
  if (auto settled = SettleFor(branch); !settled) {
    branch.needs_clear_ = true;
    ReleaseTokens(branch.history_, branch.history_charge_);
    ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
    last.clear();
    return settled;
  }
  branch.history_used_ = Clock::now();
  if (completed.stopped) {
    last.clear();
  }
  if (run != nullptr) {
    *run = completed;
  }
  return {};
}

Llm::GenerationSession::GenerationSession(Llm& model, Branch& branch,
                                          const GenerateOptions& options, Generation& out)
    : model_(model), branch_(branch), options_(options), out_(out) {}

Llm::GenerationSession::~GenerationSession() {
  base::Check(finished_, "an unfinished generation session was destroyed");
}

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::BeginGeneration(
    Branch& branch, const std::vector<float>& last, const GenerateOptions& options, Generation& out,
    bool resume) {
  CheckBranch(branch);
  // A resumed generation chooses nothing from `last`: after a restore of its
  // spilled state (no prefill) there are none.
  if (last.empty() && !resume) {
    return Error(std::format("{} has no prefill's logits to generate from", name_));
  }
  if (options.max_tokens == 0) {
    return Error("generation needs a positive token budget");
  }
  if (branch.generation_active_ || branch.prompt_session_ != nullptr) {
    return Error("the conversation already has an active session");
  }
  if (resume && (out.tokens.empty() || out.stopped || out.tokens.size() >= options.max_tokens ||
                 branch.history_.size() + 1 >= context_)) {
    return Error("a resumed generation needs an unfinished generation within the context");
  }
  branch.generation_active_ = true;
  auto session =
      std::unique_ptr<GenerationSession>(new GenerationSession(*this, branch, options, out));
  if (auto began = session->Begin(last, resume); !began) {
    session->Close();
    return std::unexpected(began.error());
  }
  return session;
}

bool Llm::GenerationSession::IsStop(std::int32_t token) const {
  return options_.stop &&
         (std::ranges::find(model_.stops_, token) != model_.stops_.end() ||
          std::ranges::find(options_.extra_stops, token) != options_.extra_stops.end());
}

bool Llm::GenerationSession::Report() {
  // The new tokens since the last call, to on_tokens: false ends the
  // generation.
  if (!options_.on_tokens) {
    return true;
  }
  std::size_t visible = std::min<std::size_t>(out_.tokens.size(), options_.max_tokens);
  if (out_.stopped && visible == out_.tokens.size() && visible > 0) {
    --visible;  // the stop token
  }
  const std::span<const std::int32_t> fresh =
      std::span<const std::int32_t>(out_.tokens).subspan(reported_, visible - reported_);
  reported_ = visible;
  return options_.on_tokens(fresh);
}

Status Llm::GenerationSession::Begin(const std::vector<float>& last, bool resume) {
  if (!model_.ReserveTokens(branch_, all_, all_charge_, branch_.history_.size() + 1)) {
    return Error("native generation history exceeds the execution budget");
  }
  branch_.sampling_.reset();
  if (options_.sampling && options_.sampling->temperature > 0) {
    branch_.sampling_ = options_.sampling;
    branch_.seed_ = options_.seed;
  }
  if (resume) {
    // The anchor was chosen and reported before the preemption; the state
    // now holds exactly the tokens before it again. Sampling stays keyed by
    // absolute position, so the continuation draws as it would have.
    out_.cancelled = false;
    all_ = branch_.history_;
    all_.push_back(out_.tokens.back());
    position_ = static_cast<std::uint32_t>(branch_.history_.size());
    reported_ = std::min<std::size_t>(out_.tokens.size(), options_.max_tokens);
    start_ = Clock::now();
    return {};
  }
  auto first = model_.Choose(branch_, last, branch_.history_.size());
  if (!first) {
    return std::unexpected(first.error());
  }
  out_.tokens = {*first};
  if (options_.keep_logits) {
    out_.logits = {last};
  }
  out_.stopped = IsStop(out_.tokens.back());
  out_.cancelled = options_.on_logits && !options_.on_logits(*first, last);
  out_.cancelled = !Report() || out_.cancelled;
  // Every token so far, the anchor (the last generated, not yet in the
  // state) last; pos: how many the state holds.
  all_ = branch_.history_;
  all_.push_back(out_.tokens.back());
  position_ = static_cast<std::uint32_t>(branch_.history_.size());
  start_ = Clock::now();
  return {};
}

bool Llm::GenerationSession::done() const {
  return finished_ || !ran_ || out_.stopped || out_.cancelled ||
         out_.tokens.size() >= options_.max_tokens;
}

std::expected<Llm::GenerationSession::Step, std::string> Llm::GenerationSession::PrepareStep(
    bool defer_capacity) {
  base::Check(!prepared_ && !done(), "preparing a generation without a next step");
  refusal_.clear();
  if (position_ + 1 >= model_.context_) {
    ran_ = Error(
        std::format("{}'s conversation reached its context of {}", model_.name_, model_.context_));
    failed_prefix_valid_ = true;
    return std::unexpected(ran_.error());
  }
  left_ = static_cast<std::uint32_t>(options_.max_tokens - out_.tokens.size());
  const auto step_tokens = model_.speculate_ ? model_.StepTokenBoundFor(branch_, left_) : 1U;
  if (!model_.ReserveTokens(branch_, all_, all_charge_, all_.size() + step_tokens) ||
      !model_.ReserveTokenScratch(branch_, all_.size(), left_)) {
    refusal_ = "native generation history exceeds the execution budget";
    if (defer_capacity) {
      return std::unexpected(refusal_);
    }
    ran_ = Error(refusal_);
    failed_prefix_valid_ = true;
    return std::unexpected(ran_.error());
  }
  if (auto prepared = model_.PrepareDecodeStateFor(branch_, position_, left_); !prepared) {
    if (defer_capacity && model_.StateRefusedFor(branch_) && model_.StateUsableFor(branch_)) {
      // Refused before dispatch: the session stays active and unprepared,
      // its state and history as the last completed step left them.
      refusal_ = prepared.error().empty() ? "the state's capacity refused it" : prepared.error();
      return std::unexpected(refusal_);
    }
    ran_ = prepared;
    failed_prefix_valid_ = model_.StateUsableFor(branch_);
    return std::unexpected(ran_.error());
  }
  prepared_ = true;
  return Step{.all = all_,
              .position = position_,
              .left = left_,
              .speculative = model_.speculate_,
              .need_logits = options_.keep_logits || static_cast<bool>(options_.on_logits)};
}

Status Llm::GenerationSession::FailStep(std::string error, bool prefix_valid) {
  base::Check(prepared_, "failing a generation without a prepared step");
  prepared_ = false;
  ran_ = Error(std::move(error));
  failed_prefix_valid_ = prefix_valid;
  return ran_;
}

Status Llm::GenerationSession::FailAfterAnchor(std::string error) {
  base::Check(prepared_, "failing a generation without a prepared step");
  prepared_ = false;
  ran_ = Error(std::move(error));
  ++position_;  // the anchor was processed before the choice failed
  failed_prefix_valid_ = model_.StateUsableFor(branch_);
  return ran_;
}

bool Llm::DeviceGreedy(const PreparedGeneration& unit) {
  return unit.branch != nullptr && unit.session != nullptr && !unit.step.speculative &&
         !unit.step.need_logits && !unit.branch->sampling_.has_value();
}

Status Llm::GenerationSession::ApplyChosen(std::int32_t token) {
  base::Check(prepared_ && !model_.speculate_, "applying an unprepared ordinary generation step");
  prepared_ = false;
  return ApplyTokens({token}, {});
}

Status Llm::GenerationSession::ApplyPlain(std::vector<float> row) {
  base::Check(prepared_ && !model_.speculate_, "applying an unprepared ordinary generation step");
  prepared_ = false;
  auto next = model_.Choose(branch_, row, position_ + 1);
  if (!next) {
    ran_ = std::unexpected(next.error());
    ++position_;  // the anchor was processed before sampling failed
    failed_prefix_valid_ = model_.StateUsableFor(branch_);
    return ran_;
  }
  std::vector<std::vector<float>> logits;
  if (options_.keep_logits || options_.on_logits) {
    logits.push_back(std::move(row));
  }
  return ApplyTokens({*next}, std::move(logits));
}

Status Llm::GenerationSession::ApplySpeculative(std::vector<std::int32_t> kept,
                                                std::vector<std::vector<float>> logits,
                                                std::uint64_t drafted) {
  base::Check(prepared_ && model_.speculate_, "applying an unprepared speculative generation step");
  prepared_ = false;
  out_.drafted += drafted;
  if (kept.empty() || kept.size() > left_ ||
      ((options_.keep_logits || options_.on_logits) && logits.size() != kept.size())) {
    // Refuse a broken runner contract before indexing borrowed rows.
    // The completed verify still retires its owed commit/restore.
    constexpr std::string_view reason =
        "a speculative step returned inconsistent token/logit counts";
    if (auto settled = model_.SettleFor(branch_); !settled) {
      ran_ = Error(std::format("{}; settling failed: {}", reason, settled.error()));
    } else {
      ran_ = Error(std::string(reason));
    }
    return ran_;
  }
  out_.accepted += kept.size() - 1;
  return ApplyTokens(std::move(kept), std::move(logits));
}

Status Llm::GenerationSession::ApplyTokens(std::vector<std::int32_t> kept,
                                           std::vector<std::vector<float>> logits) {
  if (++out_.steps == 1) {
    out_.first_step = Clock::now();
  }
  // The anchor and the accepted drafts are in the state now; the last
  // kept token is the next anchor. The generation ends at a stop token,
  // though the state holds what was accepted after it.
  base::Check(kept.size() <= all_.capacity() - all_.size(),
              "completed generation exceeded its funded token bound");
  position_ += static_cast<std::uint32_t>(kept.size());
  all_.insert(all_.end(), kept.begin(), kept.end());
  for (std::size_t i = 0; i < kept.size() && out_.tokens.size() < options_.max_tokens; ++i) {
    const std::int32_t token = kept[i];
    out_.tokens.push_back(token);
    if (options_.keep_logits) {
      out_.logits.push_back(logits[i]);
    }
    if (options_.on_logits && !options_.on_logits(token, logits[i])) {
      out_.cancelled = true;
    }
    if (IsStop(token)) {
      out_.stopped = true;
    }
    // A scoring response associates each visible token with its row.
    // Deliver it now, so a stop string cannot collect later verify rows.
    // The entire accepted verify is already complete and is still settled.
    if (options_.on_logits) {
      out_.cancelled = !Report() || out_.cancelled;
    }
    if (out_.stopped || out_.cancelled) {
      break;
    }
  }
  if (!options_.on_logits) {
    out_.cancelled = !Report() || out_.cancelled;
  }
  return {};
}

void Llm::GenerationSession::EndRefused() {
  base::Check(refused() && !prepared_ && ran_.has_value(),
              "ending a generation that was not refused");
  // Named as a prefill's refusal names its chunk (positions only, D-014).
  ran_ = Error(std::format("{}'s decode step at {}: {}", model_.name_, position_,
                           std::exchange(refusal_, {})));
  failed_prefix_valid_ = model_.StateUsableFor(branch_);
}

Status Llm::GenerationSession::RunScalarStep(bool defer_capacity) {
  auto step = PrepareStep(defer_capacity);
  if (!step) {
    return std::unexpected(step.error());
  }
  const auto started = Clock::now();
  if (step->speculative) {
    std::vector<std::int32_t> kept;
    std::vector<std::vector<float>> logits;
    bool prefix_kept = false;
    auto ran =
        model_.SpecStepFor(branch_, step->all, step->position, step->left, kept,
                           step->need_logits ? &logits : nullptr, out_.drafted, &prefix_kept);
    if (!ran) {
      return FailStep(ran.error(), prefix_kept);
    }
    // Its decode speed (calibration.h: the decode floor's).
    model_.calibration_samples_.DecodeStep(step->position, static_cast<double>(kept.size()),
                                           Seconds(Clock::now() - started));
    // SpecStep already updated the legacy counter, including partial failures.
    return ApplySpeculative(std::move(kept), std::move(logits), 0);
  }
  if (!step->need_logits && !branch_.sampling_) {
    std::int32_t token = 0;
    if (auto ran = model_.RunGreedyChunkFor(branch_, step->all, step->position, token)) {
      if (!*ran) {
        return FailStep(ran->error());
      }
      model_.calibration_samples_.DecodeStep(step->position, 1.0, Seconds(Clock::now() - started));
      return ApplyChosen(token);
    }
  }
  std::vector<float> row;
  if (auto ran = model_.RunChunkFor(branch_, step->all, step->position, false, row); !ran) {
    return FailStep(ran.error());
  }
  model_.calibration_samples_.DecodeStep(step->position, 1.0, Seconds(Clock::now() - started));
  return ApplyPlain(std::move(row));
}

Status Llm::RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) {
  // Families without a shared implementation retain the ordinary one-slot
  // unit. This fallback never advertises multi-request shared dispatch.
  if (prepared.size() != 1) {
    return Error("this model has no shared generation implementation");
  }
  PreparedGeneration& unit = prepared.front();
  if (unit.step.speculative) {
    return SpecStepFor(*unit.branch, unit.step.all, unit.step.position, unit.step.left, unit.kept,
                       unit.step.need_logits ? &unit.logits : nullptr, unit.drafted,
                       &unit.failed_prefix_valid);
  }
  return RunChunkFor(*unit.branch, unit.step.all, unit.step.position, false, unit.row);
}

Status Llm::RunScalarGenerationUnits(std::span<PreparedGeneration> prepared) {
  for (auto& unit : prepared) {
    if (unit.step.speculative) return Error("independent scalar units require plain generation");
    unit.result = RunChunkFor(*unit.branch, unit.step.all, unit.step.position, false, unit.row);
    if (!unit.result && !GenerationCohortUsable()) return unit.result;
    if (!unit.result) unit.failed_prefix_valid = StateUsableFor(*unit.branch);
  }
  return {};
}

Status Llm::RunGenerationWave(std::span<GenerationSession* const> sessions, bool defer_capacity) {
  if (sessions.empty() || sessions.size() > kMaxBranches ||
      sessions.size() > generation_wave_capacity() ||
      (sessions.size() > 1 && !supports_generation_waves())) {
    return Error("the generation cohort exceeds this model's capability");
  }
  std::array<bool, kMaxBranches> selected{};
  for (GenerationSession* session : sessions) {
    if (session == nullptr || &session->model_ != this || session->done() || session->prepared_ ||
        session->finished_) {
      return Error("a generation cohort contains an inactive or foreign session");
    }
    const std::size_t slot = BranchIndex(session->branch_);
    if (selected[slot]) {
      return Error("a generation cohort repeats a native branch");
    }
    selected[slot] = true;
  }
  std::vector<PreparedGeneration> prepared;
  prepared.reserve(sessions.size());
  const auto fail_cohort = [&](const std::string& error) {
    for (GenerationSession* session : sessions) {
      if (session->prepared_) {
        [[maybe_unused]] const auto failed = session->FailStep(error);
      } else {
        session->ran_ = Error(error);
        session->refusal_.clear();  // ended, no longer retryable
        session->failed_prefix_valid_ = false;
      }
    }
  };
  for (GenerationSession* session : sessions) {
    auto step = session->PrepareStep(defer_capacity);
    if (!step) {
      // PrepareStep records this session's own terminal error, or with
      // defer_capacity its retryable refusal. A refusal confined to its
      // branch does not discard another prepared step.
      if (!GenerationCohortUsable()) {
        fail_cohort(step.error());
        return std::unexpected(step.error());
      }
      continue;
    }
    prepared.push_back({.session = session,
                        .branch = &session->branch_,
                        .step = *step,
                        .row = {},
                        .kept = {},
                        .logits = {},
                        .drafted = 0,
                        .result = {},
                        .failed_prefix_valid = false,
                        .anchor_processed = false,
                        .chosen = std::nullopt});
  }
  if (prepared.empty()) {
    return {};  // every session retains its own preparation error
  }
  std::ranges::sort(prepared, {},
                    [this](const PreparedGeneration& unit) { return BranchIndex(*unit.branch); });
  const auto started = Clock::now();
  if (auto ran = RunPreparedGenerationWave(prepared); !ran) {
    for (PreparedGeneration& unit : prepared) {
      unit.session->out_.drafted += unit.drafted;
    }
    fail_cohort(ran.error());
    return ran;
  }
  {
    // Each request's decode speed in the wave (calibration.h: the decode
    // floor's): the tokens it committed, on average, over the wave's time.
    const double seconds = Seconds(Clock::now() - started);
    std::size_t tokens = 0;
    std::uint32_t at = std::numeric_limits<std::uint32_t>::max();
    for (const PreparedGeneration& unit : prepared) {
      if (unit.result) {
        tokens += unit.step.speculative ? unit.kept.size() : 1;
      }
      at = std::min(at, unit.step.position);
    }
    calibration_samples_.DecodeStep(
        at, static_cast<double>(tokens) / static_cast<double>(prepared.size()), seconds);
  }
  for (PreparedGeneration& unit : prepared) {
    if (!unit.result) {
      unit.session->out_.drafted += unit.drafted;
      [[maybe_unused]] const auto failed =
          unit.anchor_processed
              ? unit.session->FailAfterAnchor(unit.result.error())
              : unit.session->FailStep(unit.result.error(), unit.failed_prefix_valid);
    } else {
      // Apply owns any ordinary sampling/callback error on this session.
      // Completed peers still receive their results in branch order.
      (void)(unit.step.speculative ? unit.session->ApplySpeculative(
                                         std::move(unit.kept), std::move(unit.logits), unit.drafted)
             : unit.chosen         ? unit.session->ApplyChosen(*unit.chosen)
                                   : unit.session->ApplyPlain(std::move(unit.row)));
    }
    if (!GenerationCohortUsable()) {
      const std::string error = "the native generation cohort became unusable";
      fail_cohort(error);
      return Error(error);
    }
  }
  return {};
}

void Llm::GenerationSession::Cancel() {
  base::Check(!finished_ && !prepared_, "cancelling a generation away from a completed boundary");
  out_.cancelled = true;
}

void Llm::GenerationSession::Close() {
  branch_.sampling_.reset();
  model_.RetireSamplingScratchFor(branch_);
  model_.ReleaseTokens(all_, all_charge_);
  model_.ReleaseTokens(branch_.verify_tokens_, branch_.verify_tokens_charge_);
  branch_.generation_active_ = false;
  finished_ = true;
}

Status Llm::GenerationSession::Finish() {
  if (finished_) {
    return ran_;
  }
  base::Check(!prepared_, "finishing a generation before its prepared step was applied");
  base::Check(done(), "finishing an active generation without cancellation");
  out_.decode_seconds = Seconds(Clock::now() - start_);
  branch_.sampling_.reset();
  if (!ran_) {
    if (failed_prefix_valid_) {
      if (auto settled = model_.SettleFor(branch_); !settled) {
        ran_ = Error(std::format("{}; settling failed: {}", ran_.error(), settled.error()));
      } else {
        model_.PublishTokens(branch_, all_, all_charge_, position_);
        branch_.needs_clear_ = false;
        Close();
        return ran_;
      }
    }
    branch_.needs_clear_ = true;
    model_.ReleaseTokens(branch_.history_, branch_.history_charge_);
    Close();
    return ran_;
  }
  model_.PublishTokens(branch_, all_, all_charge_, position_);
  branch_.history_used_ = Clock::now();
  if (out_.tokens.size() > options_.max_tokens) {
    out_.tokens.resize(options_.max_tokens);
  }
  if (out_.logits.size() > out_.tokens.size()) {
    out_.logits.resize(out_.tokens.size());
  }
  if (auto settled = model_.SettleFor(branch_); !settled) {
    branch_.needs_clear_ = true;
    model_.ReleaseTokens(branch_.history_, branch_.history_charge_);
    ran_ = settled;
  }
  Close();
  return ran_;
}

Status Llm::Generate(Branch& branch, const std::vector<float>& last, const GenerateOptions& options,
                     Generation& out) {
  CheckBranch(branch);
  branch.capacity_refused_ = false;
  auto began = BeginGeneration(branch, last, options, out);
  if (!began) {
    return std::unexpected(began.error());
  }
  auto& session = **began;
  std::size_t reclaimed = 0;
  while (!session.done()) {
    if (auto ran = session.RunScalarStep(true); ran) {
      reclaimed = 0;
      continue;
    }
    if (session.refused()) {
      // The same step again once capacity was freed; otherwise it ends at
      // its completed prefix, as an undeferred refusal does.
      if (reclaimed < kMaxBranches && ReclaimFor(branch)) {
        ++reclaimed;
        continue;
      }
      branch.capacity_refused_ = true;
      session.EndRefused();
    }
    break;
  }
  auto finished = session.Finish();
  // Settling the refused generation can still fail; that failure (the
  // branch then owes a clear) is not capacity's.
  branch.capacity_refused_ = branch.capacity_refused_ && !branch.needs_clear_;
  return finished;
}

Status Llm::SaveState(Branch& branch, void* host) {
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  if (!ReserveTokens(branch, branch.saved_history_, branch.saved_history_charge_,
                     branch.history_.size())) {
    return Error("native snapshot history exceeds the execution budget");
  }
  branch.saved_valid_ = false;
  if (auto r = SettleFor(branch); !r) {
    branch.InvalidateStateSnapshot();
    return r;
  }
  auto ranges = UsedStateRangesFor(branch);
  if (auto r = SaveUsedStateFor(branch, host, ranges); !r) {
    branch.InvalidateStateSnapshot();
    return r;
  }
  branch.saved_ranges_ = std::move(ranges);
  branch.saved_history_ = branch.history_;
  if (branch.saved_history_.empty()) {
    ReleaseTokens(branch.saved_history_, branch.saved_history_charge_);
  }
  branch.saved_cursor_ = CursorFor(branch);
  SaveDecodingStateFor(branch);
  branch.saved_valid_ = true;
  return {};
}

Status Llm::RestoreState(Branch& branch, void* host) {
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  if (!branch.saved_valid_) {
    return Error("no completed conversation snapshot to restore");
  }
  if (host == nullptr && !branch.saved_ranges_.empty()) {
    return Error("the conversation snapshot has no source buffer");
  }
  if (!ReserveTokens(branch, branch.history_, branch.history_charge_,
                     branch.saved_history_.size())) {
    return Error("native restored history exceeds the execution budget");
  }
  Unkeep(branch);
  if (auto r = SettleFor(branch); !r) {
    return r;
  }
  if (auto r = RestoreSnapshotFor(branch, host, branch.saved_ranges_, branch.saved_cursor_); !r) {
    if (StateRefusedFor(branch) && StateUsableFor(branch)) return r;
    branch.needs_clear_ = true;
    ReleaseTokens(branch.history_, branch.history_charge_);
    return r;
  }
  branch.history_ = branch.saved_history_;
  branch.turn_checkpoints_.clear();  // a full diagnostic restore may replace the branch
  branch.history_used_ = Clock::now();
  SetCursorFor(branch, branch.saved_cursor_);
  RestoreDecodingStateFor(branch);
  branch.needs_clear_ = false;
  return {};
}

// ---------------------------------------------------------------- the server

Server::Server(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               const ServingOptions& options, std::FILE* log)
    : config_(config),
      roles_(roles),
      options_(options),
      log_(log),
      times_(kObservedExtents),
      node_({.compute_streams = std::max<std::size_t>(config.models.size(), 1),
             .slots = engine::kPagedSlots,
             .inline_lanes = false,
             .coalesce = false,
             .copy_lane = true,
             .lazy_handoff = options.lazy_handoff,
             .zero_state = options.zero_state,
             .handle_reserve = options.handle_reserve.value_or(engine::kHandleReserve),
             .slot_bytes = engine::kSlabSlotBytes,
             .observer = &times_,
             .poll_window = std::nullopt,
             .spin_ahead = std::nullopt,
             .quiet = std::chrono::minutes(10),
             .hold_reads = &hold_reads_,
             .hold_cancellable = options.hold_cancellable}),
      retention_(std::chrono::hours(config.memory.retention_hours)),
      spill_budget_(std::uint64_t{config.memory.spill_budget_gib} << 30U) {}

Server::~Server() {
  if (started_ && !torn_down_) {
    if (auto stopped = TearDown(); !stopped) {
      Log("stopping: " + stopped.error());
    }
  }
}

void Server::Log(std::string_view text) {
  const std::string line = std::format("jitllm-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log_);
  (void)std::fflush(log_);
}

Status Server::Make(const config::ModelEntry& entry, const ModelSettings& settings, int index) {
  if (entry.composition) {
    if (options_.image_noise.empty()) {
      Log(
          std::format("model {}: not registered: an image pipeline needs --image-noise (its "
                      "initial latents) in this build",
                      entry.name));
      return {};
    }
    models_.push_back(std::make_unique<QwenImage>(node_, entry, settings, roles_, options_, index));
    return {};
  }
  if (settings.architecture == "deepseek4") {
    models_.push_back(std::make_unique<Dsv4>(node_, entry, settings, roles_, index));
  } else if (settings.architecture == "gemma4") {
    auto artifact = OpenTrusted(roles_.installed, entry.artifact.value_or(""));
    if (!artifact) return Error(artifact.error());
    auto profile = ApprovedGemmaProfile(*artifact);
    if (!profile) return Error(profile.error());
    const auto variant = *profile == &model::Gemma4_26BA4B() ? engine::Gemma4Variant::k26BA4B
                                                             : engine::Gemma4Variant::k31B;
    models_.push_back(
        std::make_unique<Gemma>(node_, entry, settings, roles_, index, variant, options_));
  } else if (settings.architecture == "gemma2") {
    models_.push_back(std::make_unique<Gemma2>(node_, entry, settings, roles_, index));
  } else if (settings.architecture == "gemma3") {
    models_.push_back(std::make_unique<Gemma3>(node_, entry, settings, roles_, index, options_));
  } else if (settings.architecture == "qwen4exp") {
    models_.push_back(std::make_unique<Qwen38>(node_, entry, settings, roles_, index));
  } else {
    return Error(
        std::format("model {}: no runner for architecture {}", entry.name, settings.architecture));
  }
  return {};
}

Status Server::SaveSnapshot(Llm& model) {
  if (!snapshot_enabled_ || snapshot_unproven_) {
    return Error("the diagnostic snapshot is disabled or its completion is unproven");
  }
  if (snapshot_model_ != nullptr) {
    snapshot_model_->InvalidateStateSnapshot();
    snapshot_model_ = nullptr;
  }
  const auto bytes = std::max<std::uint64_t>(model.state_snapshot_bytes(), 256);
  if (bytes > snapshot_capacity_) {
    if (auto freed = node_.FreePinned(snapshot_); !freed) {
      return freed;
    }
    snapshot_ = nullptr;
    snapshot_capacity_ = 0;
    // Its staging is charged to the catalog, within the budget (the
    // node's Pinned): the guard's margin, set apart beside the budget, is
    // not counted again (that double count refused DeepSeek with DSpark
    // beside Qwen3.8). Refused for the budget, the reclaim order makes
    // room first.
    std::vector<catalog::ExtentId> staging;
    auto allocated = node_.Pinned(bytes, engine::kShared, staging);
    if (!allocated && Reclaim(bytes, true, "a state snapshot's staging") != 0) {
      staging.clear();
      allocated = node_.Pinned(bytes, engine::kShared, staging);
    }
    if (!allocated) {
      return std::unexpected(allocated.error());
    }
    snapshot_ = *allocated;
    snapshot_capacity_ = bytes;
  }
  auto saved = model.SaveState(snapshot_);
  if (!saved) {
    snapshot_unproven_ = true;
    node_.KeepPinned(snapshot_);
  } else {
    snapshot_model_ = &model;
  }
  return saved;
}

Status Server::RestoreSnapshot(Llm& model) {
  if (!snapshot_enabled_ || snapshot_unproven_ || snapshot_model_ != &model) {
    return Error("the diagnostic snapshot is disabled or its completion is unproven");
  }
  auto restored = model.RestoreState(snapshot_);
  if (!restored) {
    snapshot_unproven_ = true;
    node_.KeepPinned(snapshot_);
  }
  return restored;
}

Status Server::Start(bool snapshot) {
  if (started_) {
    return Error("the server started already");
  }
  started_ = true;
  if (config_.models.empty()) {
    return Error("the configuration names no models ([models.<name>], D-096)");
  }
  // Every model's settings (D-103: derived from its artifacts, calibrated on
  // this machine, or overridden), its checkpoint's ceiling among them,
  // resolved before opening the device node or constructing runners. An
  // invalid model later in the list must not cause any earlier model's
  // resources to be allocated either.
  std::vector<ModelSettings> settings;
  std::vector<Llm::CalibrationRecord> calibrations;
  settings.reserve(config_.models.size());
  for (const config::ModelEntry& entry : config_.models) {
    auto facts = ReadArtifactFacts(entry, roles_.installed);
    if (!facts) {
      return Error(std::format("model {}: {}", entry.name, facts.error()));
    }
    if (options_.gemma31_production && options_.gemma_row_invariant &&
        (facts->gemma_profile == &model::Gemma4_31B() ||
         facts->gemma_profile == &model::Gemma4_26BA4B()))
      return Error("the bounded Gemma production recipes require ordinary arithmetic");
    // Explicit internal arithmetic diagnostics retain their existing recipe.
    const bool gemma31_recipe =
        options_.gemma31_production || (!options_.gemma_joined && !options_.gemma_row_invariant);
    // Its calibration on this machine, if one is recorded for this device,
    // driver and build (calibration.h); a reading costs a small file.
    Llm::CalibrationRecord record;
    if (!entry.composition) {
      // Keyed by the settings its measurements depend on, as they resolve
      // without calibration.
      auto uncalibrated = ResolveSettings(entry, *facts, nullptr, options_.plain, gemma31_recipe,
                                          options_.gemma3_trained_max);
      if (!uncalibrated) {
        return Error(std::format("model {}: {}", entry.name, uncalibrated.error()));
      }
      record.key = CalibrationKeyOf(entry, *uncalibrated);
      const CalibrationRead read = ReadCalibration(roles_.state, record.key, ::geteuid());
      Log(std::format("model {}: calibration {}", entry.name, read.note));
      if (read.calibration) {
        record.known = *read.calibration;
      }
    }
    auto resolved = ResolveSettings(entry, *facts, record.known.empty() ? nullptr : &record.known,
                                    options_.plain, gemma31_recipe, options_.gemma3_trained_max);
    if (!resolved) {
      return Error(std::format("model {}: {}", entry.name, resolved.error()));
    }
    for (const std::string& ignored : resolved->ignored) {
      Log(std::format("model {}: ignored {}", entry.name, ignored));
    }
    settings.push_back(std::move(*resolved));
    calibrations.push_back(std::move(record));
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  // Each model's stream and owner: its place among the registered.
  for (std::size_t i = 0; i < config_.models.size(); ++i) {
    const std::size_t before = models_.size();
    if (auto r = Make(config_.models[i], settings[i], static_cast<int>(models_.size())); !r) {
      return r;
    }
    if (models_.size() > before && models_.back()->llm()) {
      static_cast<Llm&>(*models_.back()).calibration_record() = std::move(calibrations[i]);
    }
  }
  if (models_.empty()) {
    return Error("no configured model could be registered");
  }
  std::uint64_t activations = 0;
  std::uint64_t pool = 0;
  std::uint64_t largest = 0;
  std::uint64_t host_inputs = 0;
  std::uint64_t plans = 0;
  for (const auto& m : models_) {
    const auto started = Clock::now();
    StartProgress();
    if (auto r = m->Setup(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
    activations = std::max(activations, m->activations_needed());
    host_inputs = std::max(host_inputs, m->host_input_bytes());
    plans = std::max(plans, m->plan_floor_bytes());
    pool = std::max(pool, m->pool_needed());
    largest = std::max<std::uint64_t>(largest, m->weights().size() * kExtent);
    std::string chunks;
    if (m->llm()) {
      const auto& l = static_cast<Llm&>(*m);
      chunks = std::format("; prefill chunks of {} rows", l.max_rows());
      if (l.configured_rows() && *l.configured_rows() != l.max_rows()) {
        chunks += std::format(
            " (prefill_chunk {}: capped at what the model allows below its context of {}, in "
            "whole 8-row tiles)",
            *l.configured_rows(), l.context());
      }
    }
    Log(std::format("model {}: set up in {:.2f} s, {} weight extents ({:.2f} GB read a load){}",
                    m->name(), Seconds(Clock::now() - started), m->weights().size(),
                    static_cast<double>(m->weight_read_bytes()) / 1e9, chunks));
    // Each effective setting and its source (D-103; `jitllm-runtime
    // settings` lists them with what each came from).
    Log(std::format("model {}: settings {}", m->name(), m->settings().Summary()));
    if (const auto report = m->allocation_report(); !report.empty()) {
      Log(std::format("allocation model {}: {}", m->name(), report));
    }
    if (const auto report = m->plan_report(); !report.empty()) {
      Log(std::format("model {}: plans {}", m->name(), report));
    }
    if (const auto reason = m->serial_reason(); !reason.empty()) {
      Log(std::format("model {}: serves one request at a time ({})", m->name(), reason));
    } else if (const auto slots = m->slots_report(); !slots.empty()) {
      Log(std::format("model {}: {}", m->name(), slots));
    }
  }
  const auto startup_available = MemorySampler::Available();
  const MemoryGuard startup_guard{.largest = largest,
                                  .host_inputs = host_inputs,
                                  .plans = plans + engine::ScratchArenaBytes(),
                                  .available = startup_available,
                                  .fixed = 0,
                                  .requests = RequestFloor(config_.client)};
  const auto startup_reserve = GuardReserve(startup_guard) + largest + activations + pool;
  startup_token_capacity_ =
      startup_available > startup_reserve ? startup_available - startup_reserve : 0;
  token_memory_.SetDriver(
      std::this_thread::get_id(),
      [this](std::uint64_t incoming) { return SetTokenMemory(incoming); }, 0,
      std::chrono::milliseconds{0});
  // Conversations the process before kept (D-105): read, checked and
  // their files named before the node registers them.
  if (auto kept = PrepareKept(); !kept) {
    return kept;
  }
  if (auto r = node_.MapWorkspace(activations, pool); !r) {
    return r;
  }
  workspace_ = activations + pool;
  snapshot_enabled_ = snapshot;
  // The budget (D-050's B): everything mapped for the node's life (the zone,
  // each model's own memory, the workspace, the staging) and the largest
  // model's weights, so a full swap is the only way in.
  fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  budget_ = fixed_ + largest;
  // A chunk's host-built inputs are the node's memory too, though allocated
  // per chunk and outside the catalog: counted here beside the margin, not
  // in the budget the catalog enforces. One model runs chunks at a time, so
  // the most any model's chunk builds, not their sum.
  host_inputs_ = host_inputs;
  // So are its plans and graphs (engine/planned.h), but only the floor is
  // set apart here: what one step of any model holds at once. Every plan
  // and graph past it is charged inside the budget (PagedNode::ChargeHost)
  // and given back through the reclaim order when the budget is needed.
  // Beside it, the scratch arena every plan's first build uses (one for the
  // process, at the largest estimate the models' setups measured).
  const std::uint64_t step_plans = plans;
  plans_ = plans + engine::ScratchArenaBytes();
  plans = plans_;
  const std::uint64_t available = MemorySampler::Available();
  MemoryGuard bounds = {.largest = largest,
                        .host_inputs = host_inputs,
                        .plans = plans,
                        .available = available,
                        .fixed = fixed_,
                        .requests = 0};
  // Requests' host memory (intake_limits.h, D-102): a small floor set apart
  // here, out of the catalog's budget, so the requests a route admits never
  // take the margin the driver and the models' state count on; past it the
  // route charges what requests hold to the budget as it goes
  // (SetRequestMemory), through the reclaim order.
  request_memory_ = RequestFloor(config_.client);
  bounds.requests = request_memory_;
  const auto guard = CheckMemoryGuard(bounds);
  Log(
      std::format("allocation guard: {{\"format\":\"jitllm-wave-startup-guard-v1\","
                  "\"fixed_catalog_bytes\":{},\"shared_activation_bytes\":{},"
                  "\"shared_scratch_bytes\":{},\"largest_weight_extent_bytes\":{},"
                  "\"host_input_bytes\":{},\"plan_floor_bytes\":{},\"uncounted_margin_bytes\":{},"
                  "\"available_after_fixed_bytes\":{},\"available_known\":{},"
                  "\"guard_passed\":{},\"registered_state_virtual_extent_bytes\":{},"
                  "\"request_memory_bytes\":{},\"native_token_history_bytes\":{}}}",
                  fixed_, activations, pool, largest, host_inputs, plans, kUncountedMargin,
                  available, available != 0, guard.has_value(), node_.StateCapacity(),
                  request_memory_, token_memory_.used()));
  if (!guard) {
    return std::unexpected(guard.error());
  }
  // State is charged as it grows. Keep the physical execution cap below
  // the measured available memory, leaving host-built inputs, the plans and
  // the uncounted margin outside it. A virtual context ceiling need not fit
  // before it is used; a growth that cannot fit its active closure fails.
  if (available != 0) {
    budget_ = fixed_ + ((available - GuardReserve(bounds)) / kExtent * kExtent);
  } else {
    budget_ += node_.StateCapacity();
  }
  if (auto r = node_.Start(base::Bytes(budget_)); !r) {
    return r;
  }
  StartProgress();
  for (const auto& m : models_) {
    if (auto r = m->Register(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  for (const auto& m : models_) {
    if (auto r = m->Bind(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  node_.Run();
  // Plans and graphs past the floor are charged inside the budget; a charge
  // that does not fit has the reclaim order make room first.
  node_.SetHostFloor(step_plans);
  node_.SetReclaimer([this](std::uint64_t needed, engine::PagedNode::ReclaimFor what) {
    // Cache charges displace only other plans and graphs, never
    // conversation state (a graph's only what costs less to restore than a
    // graph); pinned staging (a checkpoint's, a snapshot's) is a state need
    // and may spill idle conversations.
    using For = engine::PagedNode::ReclaimFor;
    switch (what) {
      case For::kPlan:
        return Reclaim(needed, false, "a plan past the free budget");
      case For::kGraph:
        return Reclaim(needed, false, "a graph past the free budget", nullptr,
                       memory::ReclaimKind::kGraph);
      case For::kStaging:
        break;
    }
    return Reclaim(needed, true, "pinned staging past the free budget");
  });
  // The turn checkpoints' staging, set apart once so a budget full of
  // reclaimable plans, graphs and idle state never starves a capture or a
  // restore (they use it one at a time).
  if (std::ranges::any_of(models_, [](const auto& m) { return m->llm(); })) {
    if (auto r = node_.ReserveStaging(engine::CheckpointStagingBytes()); !r) {
      return Error("setting apart the turn checkpoints' staging: " + r.error());
    }
  }
  // The state room: what the budget leaves the conversations beside the
  // fixed memory and the largest model's weights.
  const std::uint64_t room = budget_ > fixed_ + largest ? budget_ - fixed_ - largest : 0;
  state_room_ = room;
  Log(std::format(
      "serving {} models; budget {:.2f} GiB ({:.2f} GiB fixed, the workspace {:.2f}; host-built "
      "chunk inputs {:.2f} GiB, a step's plans {:.2f} GiB and {:.2f} GiB of request memory "
      "beside it); conversation state room {:.2f} GiB beside the largest weights; {:.2f} GiB "
      "available; idle conversations spill, kept {} h within {} GiB",
      models_.size(), static_cast<double>(budget_) / (1ULL << 30U),
      static_cast<double>(fixed_) / (1ULL << 30U), static_cast<double>(workspace_) / (1ULL << 30U),
      static_cast<double>(host_inputs_) / (1ULL << 30U),
      static_cast<double>(plans_) / (1ULL << 30U),
      static_cast<double>(request_memory_) / (1ULL << 30U),
      static_cast<double>(room) / (1ULL << 30U), static_cast<double>(available) / (1ULL << 30U),
      config_.memory.retention_hours, config_.memory.spill_budget_gib));
  for (const auto& m : models_) {
    if (m->llm()) {
      static_cast<Llm&>(*m).SetTokenMemory(token_memory_);
      static_cast<Llm&>(*m).set_retention(retention_);
      static_cast<Llm&>(*m).set_log([this](std::string_view text) { Log(text); });
    }
  }
  token_catalog_ready_ = true;
  if (!SetTokenMemory(0)) {
    return Error("kept token histories do not fit the execution budget");
  }
  AdoptKept();
  return {};
}

Status Server::PrepareKept() {
  if (!options_.keep_conversations) {
    return {};  // a command: what the service kept is left alone
  }
  const bool any_llm = std::ranges::any_of(models_, [](const auto& m) { return m->llm(); });
  const auto stamp = platform::RunningExecutableStamp();
  const bool wanted = config_.memory.keep_across_restart && spill_budget_ != 0;
  const bool keep = any_llm && wanted && stamp.has_value();
  const int spill = ::open(roles_.spill.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (spill < 0) {
    return Error(std::format("the spill directory {}: {}", roles_.spill.string(), Errno(errno)));
  }
  const std::string root_name(kept::kDirectory);
  if (!keep) {
    // Nothing is kept with a spill budget of 0 (idle state is dropped) or
    // with [memory] keep_across_restart false, and nothing kept before
    // stays (D-014: its retention is explicit).
    if (any_llm && wanted) {
      Log("conversations are not kept across a restart: the executable's identity is unknown");
    } else if (any_llm && !config_.memory.keep_across_restart) {
      Log("conversations are not kept across a restart ([memory] keep_across_restart false)");
    }
    std::ignore = platform::RemovePrivate(spill, root_name.c_str());
    (void)::close(spill);
    return {};
  }
  auto opened = platform::OpenPrivateDirectory(spill, root_name.c_str());
  (void)::close(spill);
  if (!opened) {
    Log(std::format("conversations are not kept across a restart: {}/{}: {}", roles_.spill.string(),
                    root_name, Errno(opened.error())));
    return {};
  }
  const int root = *opened;
  // Hashing a record's files: a few cores, off the driver's path.
  keeper_ = std::make_unique<StateKeeper>(4, [this](std::string_view line) { Log(line); });
  const std::string build = KeptBuild();
  const std::int64_t now = NowUnixMs();
  const std::int64_t retention_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(retention_).count();
  std::vector<std::string> kept_dirs;
  std::size_t adopted_count = 0;
  std::uint64_t adopted_bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    const auto entry = std::ranges::find(config_.models, l.name(), &config::ModelEntry::name);
    const std::string artifact = entry != config_.models.end() ? entry->artifact.value_or("") : "";
    const std::string layout = l.KeptLayout();
    if (!kept::ValidDirectoryName(artifact) || layout.empty()) {
      Log(
          std::format("model {}: its conversations are not kept across a restart (its artifact "
                      "ID names no directory)",
                      l.name()));
      continue;
    }
    auto dir = platform::OpenPrivateDirectory(root, artifact.c_str());
    if (!dir) {
      Log(std::format("model {}: its conversations are not kept across a restart: {}: {}", l.name(),
                      artifact, Errno(dir.error())));
      continue;
    }
    kept_dirs.push_back(artifact);
    const std::size_t index = keeper_->AddModel(*dir);
    const kept::Identity identity{
        .build = build,
        .artifact = artifact,
        .drafter = l.speculative() && entry->drafter ? *entry->drafter : std::string(),
        .layout = layout};
    kept::Expected expected{.identity = identity,
                            .slot = 0,
                            .regions = l.KeptRegions(),
                            .layouts = l.KeptLayouts(),
                            .context = l.usable_context(),
                            .vocabulary = static_cast<std::uint32_t>(l.tokenizer().size()),
                            .now_unix_ms = now,
                            .retention_ms = retention_ms};
    // Each slot's record, checked whole and its files hashed; a record
    // that does not validate is refused and its files emptied.
    std::vector<std::uint32_t> adopt;
    std::vector<std::string> keep_files;
    for (std::uint32_t slot = 0; slot < l.branches(); ++slot) {
      const std::string record_name = kept::RecordFileName(slot);
      auto text = platform::ReadPrivateFile(*dir, record_name.c_str(), kept::kMostRecordBytes);
      if (!text) {
        if (text.error() != ENOENT) {
          Log(std::format("model {}: slot {}'s kept conversation was refused: its record: {}",
                          l.name(), slot, Errno(text.error())));
        }
        continue;
      }
      expected.slot = slot;
      std::vector<std::string> dropped;
      MemoryCharge token_charge;
      auto record = kept::Decode(*text, &token_memory_, &token_charge);
      std::string why;
      if (!record) {
        why = record.error();
      } else if (auto checked = kept::Check(*record, expected, &dropped); !checked) {
        why = checked.error();
      }
      // The files: the very ones the record names, whole, hashing as it says.
      const auto verify = [&](const std::string& name, const kept::FileId& id, std::uint64_t bytes,
                              std::span<const kept::Place> places,
                              std::span<const kept::Digest> digests) -> std::string {
        auto file = platform::OpenPrivateFile(
            *dir, name.c_str(),
            {.write = false, .create = false, .truncate = false, .direct = true});
        if (!file) {
          return std::format("{}: {}", name, Errno(file.error()));
        }
        const int fd = file->fd;
        const bool same = KeptId(file->identity) == id && file->bytes == bytes;
        const auto hashed =
            same ? HashPlaces(fd, places, 8) : std::vector<std::optional<kept::Digest>>{};
        (void)::close(fd);
        if (!same) {
          return std::format("{} is not the file its record was made for", name);
        }
        for (std::size_t i = 0; i < hashed.size(); ++i) {
          if (!hashed[i] || *hashed[i] != digests[i]) {
            return std::format("{} does not hold what its record says (its digests differ)", name);
          }
        }
        return {};
      };
      if (why.empty()) {
        std::vector<kept::Place> places;
        std::vector<kept::Digest> digests;
        for (const kept::Extent& extent : record->extents) {
          places.push_back(
              {.offset = kept::ExtentOffset(record->regions, extent), .bytes = kept::kExtentBytes});
          digests.push_back(extent.digest);
        }
        why = verify(record->file, record->id, record->file_bytes, places, digests);
      }
      // Every extent the record does not list reads as zeros from here on,
      // as a fresh slot's do: a record removed before a crash can come back
      // (its removal not yet on disk), its listed extents still matching
      // while the file's others hold what a later conversation wrote there,
      // which this one would read when it grows into them (D-105). Each
      // adoption empties them again, so their emptying need not be synced.
      if (why.empty()) {
        why = EmptyUnlisted(*dir, *record);
      }
      if (!why.empty()) {
        Log(std::format("model {}: slot {}'s kept conversation was refused: {}", l.name(), slot,
                        why));
        continue;
      }
      std::erase_if(record->checkpoints, [&](const kept::Checkpoint& c) {
        const std::string bad =
            verify(c.file, c.id, c.file_bytes, kept::CheckpointPlaces(c), c.digests);
        if (!bad.empty()) {
          dropped.push_back(bad);
        }
        return !bad.empty();
      });
      for (const std::string& line : dropped) {
        Log(std::format("model {}: slot {}: a kept turn checkpoint was left out: {}", l.name(),
                        slot, line));
      }
      adopt.push_back(slot);
      keep_files.push_back(record->file);
      keep_files.push_back(record_name);
      for (const kept::Checkpoint& c : record->checkpoints) {
        keep_files.push_back(c.file);
      }
      adopted_bytes += record->extents.size() * kExtent;
      ++adopted_count;
      adoptions_.push_back(
          {.model = &l, .token_charge = std::move(token_charge), .record = std::move(*record)});
      StartProgress();
    }
    // Nothing else stays: unadopted slots' files, other records, stale
    // temporary files (D-014: what is kept is explicit).
    if (auto entries = platform::ListPrivateDirectory(*dir); entries) {
      for (const platform::DirectoryEntry& listed : *entries) {
        if (!std::ranges::contains(keep_files, listed.name)) {
          std::ignore = platform::RemovePrivate(*dir, listed.name.c_str());
        }
      }
    }
    l.set_kept({.keeper = keeper_.get(),
                .model = index,
                .directory = *dir,
                .identity = identity,
                .adopt = std::move(adopt)});
  }
  // Models no longer configured keep nothing.
  if (auto entries = platform::ListPrivateDirectory(root); entries) {
    for (const platform::DirectoryEntry& entry : *entries) {
      if (!std::ranges::contains(kept_dirs, entry.name)) {
        std::ignore = platform::RemovePrivate(root, entry.name.c_str());
      }
    }
  }
  (void)::close(root);
  Log(
      std::format("conversations are kept across a restart in {}/{} (D-105): {} adopted ({:.1f} "
                  "MiB of state)",
                  roles_.spill.string(), root_name, adopted_count,
                  static_cast<double>(adopted_bytes) / (1U << 20U)));
  return {};
}

void Server::AdoptKept() {
  for (PendingAdoption& pending : adoptions_) {
    Llm& l = *pending.model;
    const std::uint32_t slot = pending.record.slot;
    auto branch = l.branch(slot);
    Status adopted =
        branch ? l.Adopt(**branch, pending.record) : Status(std::unexpected(branch.error()));
    if (adopted) {
      Log(std::format(
          "model {}: slot {} adopted a kept conversation of {} tokens ({} turn "
          "checkpoints); its next turn restores it",
          l.name(), slot, pending.record.tokens.size(), pending.record.checkpoints.size()));
      continue;
    }
    Log(std::format("model {}: slot {}'s kept conversation was not adopted: {}", l.name(), slot,
                    adopted.error()));
    // Its file was kept as it is: emptied (a slot's fresh state reads zeros
    // from it), and its record removed.
    if (branch) {
      std::ignore = (*branch)->ReleaseIdleState();
    }
  }
  adoptions_.clear();
  token_memory_.Settle(0, std::chrono::milliseconds{0});
}

void Server::Persist(Clock::time_point deadline, const std::function<void()>& progress) {
  if (keeper_ == nullptr || !started_ || torn_down_) {
    return;
  }
  std::size_t spilled = 0;
  std::uint64_t bytes = 0;
  auto told = Clock::now();
  if (resident_ != nullptr && resident_->llm()) {
    auto& l = static_cast<Llm&>(*resident_);
    // The most recently used first, so those that do not fit are the least
    // recently used.
    std::vector<Llm::Branch*> idle;
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i);
          b && !(*b)->history().empty() && l.BranchIdle(**b) && l.ResidentStateBytes(**b) != 0) {
        idle.push_back(*b);
      }
    }
    std::ranges::stable_sort(idle, [&l](const Llm::Branch* a, const Llm::Branch* b) {
      return l.LastUsed(*a) > l.LastUsed(*b);
    });
    for (Llm::Branch* branch : idle) {
      if (Clock::now() >= deadline) {
        break;
      }
      const std::uint64_t held = l.ResidentStateBytes(*branch);
      // Within the spill budget, as every spill, but deleting only spilled
      // state used before it (the least recently used first): one that
      // fits only by displacing a more recent conversation is not kept.
      // Its whole state counts once spilled, not only what the spill
      // writes.
      if (held == 0 || held > spill_budget_ ||
          !KeepWithinSpillBudget(held, {}, l.LastUsed(*branch))) {
        continue;
      }
      if (auto r = l.SpillIdle(*branch); r) {
        ++spilled;
        bytes += held;
      }
      if (Clock::now() - told >= std::chrono::seconds(1)) {
        progress();
        told = Clock::now();
      }
    }
  }
  // The models a swap wrote back have their records already queued. The
  // keeper writes them while it makes progress (bytes hashed, records
  // written), however long that takes; a minute with nothing moving ends
  // the wait (a hung drive), as does the deadline.
  bool drained = false;
  const auto moved = [this] {
    const StateKeeper::Stats s = keeper_->stats();
    return s.hashed_bytes + s.written + s.failed + s.stale;
  };
  std::uint64_t seen = moved();
  auto quiet_until = Clock::now() + std::chrono::minutes(1);
  while (!(drained = keeper_->Drain(std::min(deadline, Clock::now() + std::chrono::seconds(1))))) {
    const auto now = Clock::now();
    if (const std::uint64_t at = moved(); at != seen) {
      seen = at;
      quiet_until = now + std::chrono::minutes(1);
    }
    if (now >= deadline || now >= quiet_until) {
      break;
    }
    progress();
  }
  const StateKeeper::Stats stats = keeper_->stats();
  Log(
      std::format("kept for the next start: {} conversations spilled ({:.1f} MiB); records {} "
                  "written in all, {} pending{}; {:.1f} MiB hashed in {:.2f} s",
                  spilled, static_cast<double>(bytes) / (1U << 20U), stats.written,
                  keeper_->pending(), drained ? "" : " (the stop's time ran out)",
                  static_cast<double>(stats.hashed_bytes) / (1U << 20U), stats.hash_seconds));
}

bool Server::DrainKept(Clock::time_point deadline) {
  return keeper_ == nullptr || keeper_->Drain(deadline);
}

std::uint64_t Server::context_body_bytes() const {
  std::uint64_t most = 0;
  for (const auto& m : models_) {
    if (m->llm()) {
      const auto& l = static_cast<const Llm&>(*m);
      most = std::max(most, ContextBodyBytes(l.usable_context(), l.longest_token()));
    }
  }
  return most;
}

std::uint64_t Server::request_capacity() const {
  if (config_.client.request_memory_bytes) {
    return *config_.client.request_memory_bytes;
  }
  return request_memory_ + state_room_;
}

bool Server::SetRequestMemory(std::uint64_t to) {
  if (!started_ || torn_down_) {
    return to <= request_memory_;
  }
  // Past the floor, charged inside the budget; what does not fit has the
  // reclaim order free it (plans, graphs, idle conversations spilled; the
  // request memory itself is never reclaimed), all of it or nothing.
  const std::uint64_t past = to > request_memory_ ? to - request_memory_ : 0;
  std::uint64_t shortfall = 0;
  if (node_.SetRequestCharge(past, &shortfall)) {
    return true;
  }
  for (int attempt = 0; attempt < 3 && shortfall != 0; ++attempt) {
    const std::uint64_t ask = ((shortfall + kExtent - 1) / kExtent * kExtent) + kExtent;
    if (Reclaim(ask, true, "request memory") == 0) {
      break;
    }
    shortfall = 0;
    if (node_.SetRequestCharge(past, &shortfall)) {
      return true;
    }
  }
  return false;
}

bool Server::SetTokenMemory(std::uint64_t incoming) {
  if (!started_ || torn_down_) {
    return false;
  }
  if (!token_catalog_ready_) {
    const auto used = token_memory_.used();
    return used <= startup_token_capacity_ && incoming <= startup_token_capacity_ - used;
  }
  std::uint64_t shortfall = 0;
  for (int attempt = 0; attempt < 4; ++attempt) {
    // Reclaim may have retired histories since this grow began.
    const auto target = token_memory_.used() + incoming;
    if (node_.SetTokenCharge(target, &shortfall)) {
      return true;
    }
    if (attempt == 3 || Reclaim(shortfall, true, "native token history", nullptr, std::nullopt,
                                false, nullptr, incoming) == 0) {
      return false;
    }
  }
  return false;
}

Status Server::TearDown() {
  if (torn_down_) {
    return {};
  }
  if (started_) {
    RecordCalibrations();
    // What the incremental spills and swaps left unwritten, and the
    // reclaims that found too little to take, or ended short when a victim
    // gave back less than it counted (counts and bytes only).
    if (auto stats = node_.Stats(); stats) {
      Log(std::format(
          "write-backs released unchanged (nothing written): {} extents, {:.1f} MiB; reclaims "
          "that took nothing (too little to take): {}; that ended short (a victim gave back "
          "less): {}",
          stats->unchanged_writebacks,
          static_cast<double>(stats->unchanged_writeback_bytes) / (1U << 20U), reclaims_short_,
          reclaims_cut_short_));
    }
    // What each model's plans and graphs held at the end, and what bringing
    // them back would have cost (the reclaim order's measured inputs):
    // counts, bytes and seconds only.
    for (std::uint32_t i = 0; i < models_.size(); ++i) {
      std::vector<memory::ReclaimCandidate> held;
      models_[i]->ReclaimCandidates(i, false, held);
      std::array<std::uint64_t, memory::kReclaimKinds> count{};
      std::array<std::uint64_t, memory::kReclaimKinds> bytes{};
      std::array<double, memory::kReclaimKinds> seconds{};
      for (const memory::ReclaimCandidate& c : held) {
        const auto k = static_cast<std::size_t>(c.kind);
        ++count[k];
        bytes[k] += c.bytes;
        seconds[k] += c.restore_seconds;
      }
      const auto p = static_cast<std::size_t>(memory::ReclaimKind::kPlan);
      const auto g = static_cast<std::size_t>(memory::ReclaimKind::kGraph);
      if (count[p] + count[g] != 0) {
        // A plan's count includes its graphs'; a graph may have no plan
        // candidate (the image pipeline's recorded step).
        const std::array<double, memory::kReclaimKinds> cost = memory::KindCosts(held);
        const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
        std::uint64_t plan_bytes = 0;
        double plan_seconds = 0;
        for (const memory::ReclaimCandidate& c : held) {
          if (c.kind != memory::ReclaimKind::kPlan) {
            continue;
          }
          std::uint64_t own = c.bytes;
          double own_seconds = c.restore_seconds;
          for (const memory::ReclaimCandidate& graph : held) {
            if (graph.kind == memory::ReclaimKind::kGraph && graph.id == c.id) {
              own -= std::min(own, graph.bytes);
              own_seconds -= std::min(own_seconds, graph.restore_seconds);
            }
          }
          plan_bytes += own;
          plan_seconds += own_seconds;
        }
        Log(std::format(
            "model {}: {} plans held {:.1f} MiB (planned in {:.3f} s, {:.2f} s a GiB), "
            "{} graphs {:.1f} MiB counted, {:.1f} MiB measured at their captures (captured in "
            "{:.3f} s, {:.2f} s a GiB)",
            models_[i]->name(), count[p], mib(plan_bytes), plan_seconds, cost[p], count[g],
            mib(bytes[g]), mib(models_[i]->graph_measured_bytes()), seconds[g], cost[g]));
      }
    }
  }
  torn_down_ = true;
  std::vector<engine::PagedModel*> models;
  models.reserve(models_.size());
  for (const auto& m : models_) {
    models.push_back(&m->paged());
    // Kept checkpoints' files stay for the next start (D-105): those no
    // record names it removes.
    if (m->llm()) {
      static_cast<Llm&>(*m).PreserveKeptFiles();
    }
  }
  return node_.TearDown(models);
}

Served* Server::Find(std::string_view name) {
  for (const auto& m : models_) {
    if (m->name() == name) {
      return m.get();
    }
  }
  return nullptr;
}

Status Server::InRequest(Served& m, const std::function<Status()>& body) {
  if (auto selected = m.PrepareDefaultRequest(); !selected) {
    return selected;
  }
  return node_.WithRequest(m.paged().stream(), m.request_closure(),
                           std::format("{}'s request", m.name()), body);
}

Status Server::SelectRequestBranches(Llm& model, std::span<Llm::Branch* const> active) {
  if (!started_ || torn_down_ || resident_ != &model || active.empty() ||
      std::ranges::none_of(models_, [&](const auto& owned) { return owned.get() == &model; })) {
    return Error("a cooperative request needs its resident model and active branches");
  }
  if (auto selected = model.SelectBranches(active); !selected) {
    return selected;
  }
  const auto stream = model.paged().stream();
  if (node_.InRequest(stream)) {
    return node_.RefreshRequest(stream, model.request_closure());
  }
  return node_.BeginRequest(stream, model.request_closure(),
                            std::format("{}'s request cohort", model.name()));
}

Status Server::EndRequestBranches(Llm& model) {
  if (!started_ || torn_down_ || resident_ != &model) {
    return Error("a retiring request cohort does not own the resident model");
  }
  const auto stream = model.paged().stream();
  // A proven failed job or state helper can already have ended its request.
  // The caller must separately prove that no work still borrows its owners.
  return node_.InRequest(stream) ? node_.EndRequest(stream) : Status{};
}

Server::ReferenceRetirement Server::RetireRequestBranches(Llm& model, bool last_owner) {
  if (!started_ || torn_down_ || resident_ != &model || !model.generation_cohort_usable()) {
    return {.result = Error("the native cohort cannot prove reference retirement"),
            .references_retired = false};
  }
  const auto stream = model.paged().stream();
  bool queued = false;
  auto fenced = node_.Job(
      model.request_closure(),
      [&queued](providers::NativeStream) {
        queued = true;
        // kQueued requires the DeviceLane to record and await its completion
        // fence, even though this job queues no arithmetic or copies itself.
        return scheduler::JobResult::kQueued;
      },
      "retiring native request-owner references", stream);
  if (!fenced || !queued) {
    return {.result = !fenced ? std::move(fenced) : Error("the retirement fence was not queued"),
            .references_retired = false};
  }
  // The known-complete stream fence is the proof. An EndRequest error does
  // not revoke it: EndRequest itself awaits ProgramDone::gone before return.
  return {.result = last_owner ? EndRequestBranches(model) : Status{}, .references_retired = true};
}

Status Server::WaitReleased() {
  // Progress-based (D-102): only evictions that stop finishing for the
  // node's quiet time are given up on, however long the whole takes.
  auto give_up = Clock::now() + node_.quiet();
  std::size_t before = std::numeric_limits<std::size_t>::max();
  for (;;) {
    std::size_t left = 0;
    if (auto r = node_.Call(
            [&]() -> Status {
              left = node_.scheduler().evictions();
              return {};
            },
            "counting evictions");
        !r) {
      return r;
    }
    if (left == 0) {
      return {};
    }
    if (left < before) {
      before = left;
      give_up = Clock::now() + node_.quiet();
    } else if (Clock::now() > give_up) {
      return Error(
          "backing no load took was not released: no eviction finished for the node's "
          "quiet time");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

Status Server::FinishSwap(SwapParts& parts) {
  if (auto r = WaitReleased(); !r) {
    return r;
  }
  parts.release = Seconds(Clock::now() - parts.ready);
  auto after = node_.Stats();
  if (!after) {
    return std::unexpected(after.error());
  }
  parts.handed_off = after->handed_off - swap_before_.handed_off;
  parts.released_unused = after->released_unused - swap_before_.released_unused;
  return {};
}

std::uint64_t Server::SpilledBytes(SpillView view) {
  std::uint64_t bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    // A model a swap wrote back holds all its state on disk until it returns.
    const bool written_back =
        m.get() != view.in &&
        (m.get() == view.out || std::ranges::find(spilled_, m.get()) != spilled_.end());
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b) {
        bytes += written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b);
      }
    }
  }
  return bytes;
}

void Server::ForEachDeletableSpill(
    SpillView view, std::optional<Clock::time_point> older_than,
    const std::function<void(Llm& model, Llm::Branch& branch, std::uint64_t counted)>& each) {
  for (const auto& m : models_) {
    if (!m->llm() || m.get() == view.in) {
      continue;  // the incoming model's stop counting once it is resident
    }
    auto& l = static_cast<Llm&>(*m);
    const bool written_back =
        m.get() == view.out || std::ranges::find(spilled_, m.get()) != spilled_.end();
    for (std::size_t i = 0; i < l.branches(); ++i) {
      auto b = l.branch(i);
      if (!b || !l.BranchIdle(**b) || (*b)->HeldByContinuation() ||
          (older_than && l.LastUsed(**b) >= *older_than)) {
        continue;
      }
      const std::uint64_t counted = written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b);
      if (counted != 0) {
        each(l, **b, counted);
      }
    }
  }
}

bool Server::SpillBudgetFits(std::uint64_t extra, SpillView view,
                             std::optional<Clock::time_point> older_than) {
  const std::uint64_t held = SpilledBytes(view);
  std::uint64_t deletable = 0;
  ForEachDeletableSpill(view, older_than,
                        [&](Llm&, Llm::Branch&, std::uint64_t counted) { deletable += counted; });
  const std::uint64_t left = held - std::min(held, deletable);
  return left <= spill_budget_ && extra <= spill_budget_ - left;
}

bool Server::KeepWithinSpillBudget(std::uint64_t extra, SpillView view,
                                   std::optional<Clock::time_point> older_than) {
  // Nothing deleted for a conversation (or a swap) that would not fit even
  // with every deletable one gone.
  return SpillBudgetFits(extra, view, older_than) && DeleteSpilledPast(extra, view, older_than);
}

bool Server::DeleteSpilledPast(std::uint64_t extra, SpillView view,
                               std::optional<Clock::time_point> older_than) {
  std::size_t deleted = 0;
  std::uint64_t deleted_bytes = 0;
  bool fits = false;
  for (;;) {
    const std::uint64_t held = SpilledBytes(view);
    if (held <= spill_budget_ && extra <= spill_budget_ - held) {
      fits = true;
      break;
    }
    // The least recently used spilled conversation no request holds.
    Llm* oldest = nullptr;
    Llm::Branch* branch = nullptr;
    ForEachDeletableSpill(view, older_than, [&](Llm& l, Llm::Branch& b, std::uint64_t) {
      if (branch == nullptr || l.LastUsed(b) < oldest->LastUsed(*branch)) {
        oldest = &l;
        branch = &b;
      }
    });
    if (branch == nullptr) {
      break;  // nothing more to delete: still short
    }
    const std::uint64_t bytes = oldest->StateBytes(*branch);
    if (auto released = branch->ReleaseIdleState(); !released) {
      Log(std::format("{}'s spilled conversation could not be deleted: {}", oldest->name(),
                      released.error()));
      break;
    }
    ++deleted;
    deleted_bytes += bytes;
  }
  if (deleted != 0) {
    Log(std::format(
        "deleted {} spilled conversations ({:.1f} MiB), the least recently used, "
        "past the spill budget of {} GiB",
        deleted, static_cast<double>(deleted_bytes) / (1U << 20U), spill_budget_ >> 30U));
  }
  return fits;
}

void AddIdleStateCandidates(Llm& model, std::uint32_t owner, bool running, const Llm::Branch* spare,
                            const IdleStateRates& rates,
                            std::vector<memory::ReclaimCandidate>& out) {
  for (std::size_t slot = 0; slot < model.branches(); ++slot) {
    auto b = model.branch(slot);
    if (!b || *b == spare || !model.BranchIdle(**b) || (*b)->history().empty()) {
      continue;
    }
    const std::uint64_t bytes = model.ResidentStateBytes(**b);
    if (bytes == 0) {
      continue;
    }
    // Spilled whole within the budget (Server::Reclaim); past all of it,
    // dropped, which a continuation holding it refuses.
    const bool dropped = bytes > rates.spill_budget;
    if (dropped && (*b)->HeldByContinuation()) {
      continue;
    }
    // A spill writes only what changed since its last one; a restore reads
    // it all.
    const auto size = static_cast<double>(bytes);
    const auto writes = static_cast<double>(model.SpillWriteBytes(**b));
    out.push_back(
        {.kind = memory::ReclaimKind::kIdleState,
         .owner = owner,
         .id = slot,
         .bytes = bytes,
         .last_use = static_cast<std::uint64_t>(model.LastUsed(**b).time_since_epoch().count()),
         .restore_seconds = dropped ? static_cast<double>((*b)->history().size()) *
                                          model.settings().recompute_ms_per_token.value / 1000.0
                                    : (writes / rates.spill_rate) + (size / rates.restore_rate),
         .running = running});
    memory::SetUse(out.back(), model.LastStamp(**b));
  }
}

bool RoomFor(std::uint64_t needed, std::optional<std::uint64_t> free,
             const std::function<std::uint64_t(std::uint64_t)>& reclaim) {
  if (needed == 0 || !free || *free >= needed) {
    return true;
  }
  return reclaim(needed - *free) >= needed - *free;
}

bool Server::RoomFor(std::uint64_t needed, std::string_view why, const Llm::Branch* spare) {
  if (needed == 0 || !started_ || torn_down_) {
    return true;
  }
  auto free = node_.FreeBytes();
  return runtime::RoomFor(
      needed, free ? std::optional(*free) : std::nullopt, [&](std::uint64_t shortfall) {
        return Reclaim(shortfall, true, why, nullptr, std::nullopt, false, spare);
      });
}

std::uint64_t Server::Reclaim(std::uint64_t needed, bool states, std::string_view why,
                              const Served* running, std::optional<memory::ReclaimKind> below_kind,
                              bool partial, const Llm::Branch* spare,
                              std::uint64_t token_incoming) {
  if (reclaiming_ || !started_ || torn_down_ || needed == 0) {
    return 0;
  }
  reclaiming_ = true;
  if (running == nullptr) {
    running = resident_;
  }
  // Kept zeroed state backing holds nothing: it goes first, before the
  // order prices anything (other models' before the running one's).
  const std::uint64_t kept = ReleaseKept(needed, running);
  if (kept >= needed) {
    reclaiming_ = false;
    return kept;
  }
  // Idle state's cost at the rates measured so far (once a GiB has moved).
  double spill_rate = kSpillBytesPerSecond;
  double restore_rate = kRestoreBytesPerSecond;
  {
    Llm::SpillStats all;
    for (const auto& m : models_) {
      if (m->llm()) {
        const Llm::SpillStats& s = static_cast<Llm&>(*m).spill_stats();
        all.spilled_bytes += s.spilled_bytes;
        all.spill_seconds += s.spill_seconds;
        all.restored_bytes += s.restored_bytes;
        all.restore_seconds += s.restore_seconds;
      }
    }
    constexpr std::uint64_t kEnough = std::uint64_t{1} << 30U;
    if (all.spilled_bytes >= kEnough && all.spill_seconds > 0) {
      spill_rate = static_cast<double>(all.spilled_bytes) / all.spill_seconds;
    }
    if (all.restored_bytes >= kEnough && all.restore_seconds > 0) {
      restore_rate = static_cast<double>(all.restored_bytes) / all.restore_seconds;
    }
  }
  const auto k = [](memory::ReclaimKind kind) { return static_cast<std::size_t>(kind); };
  std::array<std::uint64_t, memory::kReclaimKinds> count{};
  std::array<std::uint64_t, memory::kReclaimKinds> freed_by{};
  std::array<double, memory::kReclaimKinds> cost{};
  std::size_t dropped = 0;
  bool took = false;
  Llm* idle_owner = nullptr;
  struct HistoryVictim {
    Llm* model;
    Llm::Branch* branch;
    std::uint32_t owner;
    std::uint64_t slot;
  };
  std::vector<HistoryVictim> histories;
  std::vector<TokenReclaimGroup> history_groups;
  // The candidates afresh each round (memory::RunReclaim): every model's
  // plans and graphs, and with `states` the resident model's idle state.
  const auto gather = [&](std::vector<memory::ReclaimCandidate>& candidates, double& below) {
    idle_owner = nullptr;
    histories.clear();
    history_groups.clear();
    std::uint32_t resident_index = 0;
    for (std::uint32_t i = 0; i < models_.size(); ++i) {
      models_[i]->ReclaimCandidates(i, models_[i].get() == running, candidates);
      if (states && models_[i]->llm()) {
        auto& model = static_cast<Llm&>(*models_[i]);
        for (std::size_t slot = 0; slot < model.branches(); ++slot) {
          auto branch = model.branch(slot);
          if (!branch || *branch == spare || !model.HistoryReclaimable(**branch)) {
            continue;
          }
          const auto bytes = model.IdleHistoryBytes(**branch);
          if (bytes == 0) {
            continue;
          }
          histories.push_back({.model = &model, .branch = *branch, .owner = i, .slot = slot});
        }
      }
      if (models_[i].get() == resident_) {
        resident_index = i;
      }
    }
    std::ranges::sort(histories, [](const HistoryVictim& a, const HistoryVictim& b) {
      return a.model->LastUsed(*a.branch) < b.model->LastUsed(*b.branch);
    });
    std::vector<std::uint64_t> capacities;
    capacities.reserve(histories.size());
    for (const auto& history : histories) {
      capacities.push_back(history.model->IdleHistoryBytes(*history.branch));
    }
    history_groups =
        GroupTokenReclaim(capacities, token_memory_.used() + token_incoming, kExtent, needed);
    std::size_t first = 0;
    for (const auto& group : history_groups) {
      const auto& oldest = histories[first];
      double seconds = 0;
      bool group_running = false;
      for (std::size_t i = first; i < group.end; ++i) {
        const auto& h = histories[i];
        seconds += static_cast<double>(h.branch->history().size()) *
                   h.model->settings().recompute_ms_per_token.value / 1000.0;
        group_running = group_running || h.model == running;
      }
      candidates.push_back({.kind = memory::ReclaimKind::kTokenHistory,
                            .owner = oldest.owner,
                            .id = oldest.slot,
                            .bytes = group.catalog_bytes,
                            .last_use = static_cast<std::uint64_t>(
                                oldest.model->LastUsed(*oldest.branch).time_since_epoch().count()),
                            .restore_seconds = seconds,
                            .running = group_running});
      memory::SetUse(candidates.back(), oldest.model->LastStamp(*oldest.branch));
      first = group.end;
    }
    // The running model's next step finds what its last ones used.
    memory::ProtectFloor(candidates, running != nullptr ? running->plan_floor_bytes() : 0);
    if (states && resident_ != nullptr && resident_->llm()) {
      idle_owner = static_cast<Llm*>(resident_);
      AddIdleStateCandidates(
          *idle_owner, resident_index, resident_ == running, spare,
          {.spill_rate = spill_rate, .restore_rate = restore_rate, .spill_budget = spill_budget_},
          candidates);
    }
    cost = memory::KindCosts(candidates);
    if (below_kind) {
      // What the charge would cost to restore a GiB: its kind's measured
      // cost, or the GB10's when none is held.
      double charged = cost[k(*below_kind)];
      if (charged == 0) {
        charged = *below_kind == memory::ReclaimKind::kGraph ? kGraphSecondsPerGiB : 0;
      }
      below = memory::ReclaimInflation() + charged;
    }
  };
  const auto take = [&](const memory::ReclaimCandidate& c) -> std::uint64_t {
    std::uint64_t got = 0;
    if (c.kind == memory::ReclaimKind::kTokenHistory) {
      std::size_t first = 0;
      const auto prospective = [&]() {
        const auto target = token_memory_.used() + token_incoming;
        return (target / kExtent + (target % kExtent != 0 ? 1U : 0U)) * kExtent;
      };
      const auto before = prospective();
      const auto charged_before = node_.token_charged();
      const auto free_before = node_.FreeBytes();
      for (const auto& group : history_groups) {
        const auto& oldest = histories[first];
        if (oldest.owner == c.owner && oldest.slot == c.id) {
          for (std::size_t i = first; i < group.end; ++i) {
            auto& h = histories[i];
            if (h.branch != spare && h.model->HistoryReclaimable(*h.branch)) {
              std::ignore = h.branch->ReleaseIdleState();
            }
          }
          break;
        }
        first = group.end;
      }
      const auto after = prospective();
      const auto free_after = node_.FreeBytes();
      const auto actual_tokens = charged_before - std::min(charged_before, node_.token_charged());
      // Dropping a history can also release its resident native state. Count
      // the measured catalog gain once, including overlapping idle victims.
      got = free_before && free_after && *free_after >= *free_before ? *free_after - *free_before
                                                                     : actual_tokens;
      if (token_incoming != 0) {
        const auto target_drop = before - std::min(before, after);
        // The incoming allocation can change which rounded boundary matters.
        // Replace actual token release with the prospective target decrease.
        got = TokenReclaimCredit(got, actual_tokens, target_drop);
      }
    } else if (c.kind == memory::ReclaimKind::kIdleState) {
      if (idle_owner == nullptr) {
        return 0;  // only the resident model's idle state is a candidate
      }
      auto b = idle_owner->branch(c.id);
      if (!b) {
        return 0;
      }
      // Spill, not clear: within the spill budget (the least recently used
      // spilled state deleted first; once spilled its whole state counts);
      // past it, or with none, dropped. A state larger than the whole
      // budget deletes nothing else first.
      const std::uint64_t held = idle_owner->ResidentStateBytes(**b);
      if (held == 0) {
        return 0;  // gone meanwhile: nothing to give back
      }
      if (held <= spill_budget_ && KeepWithinSpillBudget(held)) {
        if (auto spilled = idle_owner->SpillIdle(**b); spilled) {
          got = c.bytes;
        } else {
          Log(std::format("{}'s idle conversation in slot {}: {}", idle_owner->name(), c.id,
                          spilled.error()));
          got = idle_owner->ResidentStateBytes(**b) == 0 ? c.bytes : 0;
        }
      } else if (auto released = (*b)->ReleaseIdleState(); released) {
        got = c.bytes;
        ++dropped;
      }
    } else {
      got = models_[c.owner]->Reclaim(c.kind, c.id);
    }
    if (got != 0) {
      took = true;
      ++count[k(c.kind)];
      freed_by[k(c.kind)] += got;
    }
    return got;
  };
  // All of it or nothing (memory::RunReclaim): a selection that cannot
  // cover what is needed reclaims nothing (the caller waits or refuses);
  // another round, without it, only when a victim gave back less than its
  // count (held, or gone meanwhile), and what the rounds before took then
  // stays taken even when the rest cannot be covered.
  const memory::ReclaimRun run = memory::RunReclaim(needed - kept, partial, gather, take);
  const std::uint64_t freed = run.freed + kept;
  if (freed == 0 && run.short_of_need) {
    ++reclaims_short_;
  } else if (!partial && freed != 0 && freed < needed) {
    ++reclaims_cut_short_;
  }
  if (freed != 0) {
    platform::ReleaseFreeHeap();
  }
  if (took) {
    const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
    using K = memory::ReclaimKind;
    // Counts, bytes and measured costs only (D-014).
    Log(std::format(
        "reclaimed {:.1f} MiB of {:.1f} MiB needed for {}: {} graphs ({:.1f} MiB), {} plans "
        "({:.1f} MiB), {} idle conversations {} ({:.1f} MiB); measured s a GiB: graphs {:.2f}, "
        "plans {:.2f}, idle state {:.2f}; {} history groups discarded ({:.1f} MiB capacity)",
        mib(freed), mib(needed), why, count[k(K::kGraph)], mib(freed_by[k(K::kGraph)]),
        count[k(K::kPlan)], mib(freed_by[k(K::kPlan)]), count[k(K::kIdleState)],
        dropped != 0 ? "dropped" : "spilled", mib(freed_by[k(K::kIdleState)]), cost[k(K::kGraph)],
        cost[k(K::kPlan)], cost[k(K::kIdleState)], count[k(K::kTokenHistory)],
        mib(freed_by[k(K::kTokenHistory)])));
  }
  reclaiming_ = false;
  return freed;
}

Status Server::MakeRoomForSwap(Served& m, std::span<const catalog::ExtentId> out) {
  // The bytes the swap lacks now: what it pages in beyond what goes out
  // and the budget's free room, from the catalog itself (what a reclaim
  // reports freed need not be what occupancy dropped by: the plans'
  // charge moves in whole extents past its floor).
  std::string sizing_error;
  const auto shortfall = [&]() -> std::uint64_t {
    std::uint64_t incoming = 0;
    std::uint64_t outgoing = 0;
    std::uint64_t free = 0;
    auto sized = node_.Call(
        [&]() -> Status {
          const catalog::Catalog& catalog = node_.catalog();
          for (const auto& [extent, generation] : m.everything().extents) {
            (void)generation;
            if (const auto view = catalog.Describe(extent);
                view && view->state != catalog::ExtentState::kResident) {
              incoming += view->descriptor.size.value();
            }
          }
          for (const catalog::ExtentId extent : out) {
            if (const auto view = catalog.Describe(extent);
                view && view->state == catalog::ExtentState::kResident) {
              outgoing += view->descriptor.size.value();
            }
          }
          const std::uint64_t occupancy = catalog.OccupancyOf(node_.domain()).Total().value();
          free = occupancy < budget_ ? budget_ - occupancy : 0;
          return {};
        },
        "sizing a swap");
    if (!sized) {
      sizing_error = sized.error();
      return 0;
    }
    return incoming > outgoing + free ? incoming - outgoing - free : 0;
  };
  auto made = MakeRoom(
      shortfall,
      [&](std::uint64_t ask) { return Reclaim(ask, false, "a swap's incoming model", &m); },
      kExtent);
  if (!sizing_error.empty()) {
    return Error(sizing_error);
  }
  if (!made) {
    return Error(std::format("{} to {}: {}", resident_ != nullptr ? resident_->name() : "a swap",
                             m.name(), made.error()));
  }
  return {};
}

bool Server::NodeHealthy() {
  if (!started_ || torn_down_) {
    return false;
  }
  auto checked = node_.Call(
      [&]() -> Status {
        return node_.scheduler().fault() ? Error("the node's scheduler faulted") : Status{};
      },
      "checking the node's health");
  return checked.has_value();
}

std::uint64_t Server::ReleaseKept(std::uint64_t needed, const Served* last) {
  std::vector<catalog::ExtentId> kept;
  for (const bool running : {false, true}) {
    for (const auto& m : models_) {
      if ((m.get() == last) != running) {
        continue;
      }
      const std::vector<catalog::ExtentId> own = m->kept_state();
      kept.insert(kept.end(), own.begin(), own.end());
    }
  }
  if (kept.empty()) {
    return 0;
  }
  std::vector<catalog::ExtentId> resident;
  std::uint64_t bytes = 0;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const catalog::ExtentId extent : kept) {
          if (bytes >= needed) {
            break;
          }
          if (const auto view = catalog.Describe(extent);
              view && view->state == catalog::ExtentState::kResident && view->discarded &&
              view->leases == 0 && view->registrations == 0) {
            resident.push_back(extent);
            bytes += view->descriptor.size.value();
          }
        }
        return {};
      },
      "listing kept state backing");
  if (!listed || resident.empty() || !node_.Evict(resident)) {
    return 0;
  }
  return bytes;
}

Status Server::EvictPaged(Served& m) {
  // Only what a swap moves: its weights and its conversation state (the
  // shared workspace and its own pinned runtime memory stay where they
  // are), with the zeroed backing its clears kept. State is written back
  // to its places.
  std::vector<catalog::ExtentId> own = m.weights();
  const std::vector<catalog::ExtentId> state = m.state();
  own.insert(own.end(), state.begin(), state.end());
  const std::vector<catalog::ExtentId> kept = m.kept_state();
  own.insert(own.end(), kept.begin(), kept.end());
  std::vector<catalog::ExtentId> resident;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const catalog::ExtentId extent : own) {
          if (const auto view = catalog.Describe(extent);
              view && view->state == catalog::ExtentState::kResident) {
            resident.push_back(extent);
          }
        }
        return {};
      },
      "listing a model's resident extents");
  if (!listed) {
    return listed;
  }
  return resident.empty() ? Status{} : node_.Evict(resident);
}

Status Server::FenceModel(Served& m) {
  const std::uint32_t stream = m.paged().stream();
  if (node_.InRequest(stream)) {
    // Its task may have ended with the cancellation: ending it only frees
    // what it held.
    std::ignore = node_.EndRequest(stream);
  }
  return node_.Job(
      catalog::Closure{}, [](providers::NativeStream) { return scheduler::JobResult::kQueued; },
      "fencing a model's stream after a hang", stream);
}

Status Server::RecoverModel(Served& m) {
  if (!NodeHealthy()) {
    return Error("the node's scheduler faulted");
  }
  if (auto fenced = FenceModel(m); !fenced) {
    return Error(std::format("its stream could not be fenced: {}", fenced.error()));
  }
  if (auto recovered = m.RecoverInPlace(); !recovered) {
    return recovered;
  }
  if (m.llm()) {
    // The hung unit may have been timed: no calibration counts it (D-103).
    static_cast<Llm&>(m).calibration_samples().Forget();
  }
  // Evicted, so its next activation loads it whole: its weights, and the
  // state it kept (idle conversations) written back.
  if (auto evicted = EvictPaged(m); !evicted) {
    return Error(std::format("evicting it: {}", evicted.error()));
  }
  if (resident_ == &m) {
    resident_ = nullptr;
  }
  if (m.llm() && m.HasRetainedState() && std::ranges::find(spilled_, &m) == spilled_.end()) {
    // As a swap's write-back: the conversations it kept are wholly on disk,
    // and their records follow (D-105).
    m.StateWrittenBack(true);
    spilled_.push_back(&m);
    auto& l = static_cast<Llm&>(m);
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b && !(*b)->spilled()) {
        l.KeepBranch(**b);
      }
    }
  }
  return {};
}

Status Server::UndoSwap(Served& out, Served& in) {
  // What of the incoming model came in goes out again (its state written
  // back to its place), then the outgoing model comes back whole: its
  // weights, and its state from what its write-back saved.
  if (auto r = EvictPaged(in); !r) {
    return r;
  }
  std::vector<catalog::ExtentId> again;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const auto& [extent, generation] : out.everything().extents) {
          (void)generation;
          if (const auto view = catalog.Describe(extent);
              view && view->state != catalog::ExtentState::kResident) {
            again.push_back(extent);
          }
        }
        return {};
      },
      "listing a failed swap's extents");
  if (!listed) {
    return listed;
  }
  if (!again.empty()) {
    std::vector<engine::LoadStats> log;
    if (auto r = node_.Load(again, std::format("{} back after a failed swap", out.name()), log);
        !r) {
      return r;
    }
  }
  return out.AfterLoad();  // its checks after a load, as after a swap
}

void Server::RecordCalibrations() {
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    Llm::CalibrationRecord& record = l.calibration_record();
    auto measured = l.calibration_samples().Take(record.known);
    if (!measured) {
      continue;
    }
    // Recorded for the next registration; this service keeps its settings,
    // so a running schedule never changes with wall time (calibration.h).
    if (auto written = WriteCalibration(roles_.state, record.key, *measured); !written) {
      Log(std::format("model {}: calibration not recorded: {}", m->name(), written.error()));
      continue;
    }
    record.known = *measured;
    Log(std::format("model {}: calibration recorded, in force from the next start: {}", m->name(),
                    FormatCalibration(*measured, record.key)));
  }
}

void Server::Maintain() {
  if (!started_ || torn_down_ || reclaiming_) {
    return;
  }
  token_memory_.Settle(0, std::chrono::milliseconds{0});
  const auto now = Clock::now();
  if (now < next_maintenance_) {
    return;
  }
  next_maintenance_ = now + kMaintenanceInterval;
  // Idle conversations past their retention, resident or spilled.
  std::size_t expired = 0;
  std::uint64_t expired_bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    for (std::size_t i = 0; i < l.branches(); ++i) {
      auto b = l.branch(i);
      if (b) {
        (void)l.ExpireTurnCheckpoints(**b, now);
      }
      if (!b || !l.BranchIdle(**b) || (*b)->HeldByContinuation() || (*b)->history().empty() ||
          now - l.LastUsed(**b) < retention_) {
        continue;
      }
      const std::uint64_t bytes = l.StateBytes(**b);
      if (auto released = (*b)->ReleaseIdleState(); released) {
        ++expired;
        expired_bytes += bytes;
      }
    }
  }
  if (expired != 0) {
    Log(std::format("deleted {} idle conversations ({:.1f} MiB) past their retention of {} h",
                    expired, static_cast<double>(expired_bytes) / (1U << 20U),
                    config_.memory.retention_hours));
  }
  // As far as it can: what a request holds stays, the rest still goes.
  (void)DeleteSpilledPast(0);
  RecordCalibrations();
  // Pressure from outside the runtime: other processes share the node's
  // memory (D-004). One reclaim of what would restore the target headroom,
  // spaced by a back-off while the pressure persists (PressureTrim), never
  // the running model's in-use floor (Reclaim).
  const platform::MemoryPressure pressure = platform::ReadMemoryPressure();
  const PressureTrim::Trim trim = pressure_.Observe(pressure, now);
  if (!trim.now) {
    return;
  }
  const std::uint64_t freed = Reclaim(trim.needed, true, trim.why, nullptr, std::nullopt, true);
  if (pressure_.Trimmed(freed, trim.needed, now)) {
    const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
    Log(std::format(
        "memory pressure from outside ({:.0f} MiB available, a full stall {:.1f}% of the last 10 "
        "s): gave back {:.1f} MiB of {:.1f} MiB, too little left to give without the running "
        "model's floor; looking again in {} s ({} trims, {} short)",
        mib(pressure.available.value_or(0)), pressure.stall ? pressure.stall->full_avg10 : 0.0,
        mib(freed), mib(trim.needed),
        std::chrono::duration_cast<std::chrono::seconds>(pressure_.backoff()).count(),
        pressure_.trims(), pressure_.short_trims()));
  }
}

Status Server::Activate(Served& m, SwapParts& parts, std::optional<bool> spill_state) {
  // Records' hashing waits while a swap moves the models (StateKeeper::
  // Quiet): their reads and the swap's share the drive.
  const StateKeeper::Quiet quiet(keeper_.get());
  parts = SwapParts{};
  parts.to = m.name();
  if (resident_ == &m) {
    return {};
  }
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  swap_before_ = *before;
  const bool restoring = std::ranges::find(spilled_, &m) != spilled_.end();
  // The incoming model's conversations a swap wrote back come back live:
  // their records go before anything of it loads (D-105), not before a
  // refusal that leaves them on disk. Those the reclaim order spilled stay
  // on disk, and kept, until a turn restores them.
  const auto unkeep_incoming = [&m]() {
    if (m.llm()) {
      auto& in = static_cast<Llm&>(m);
      for (std::size_t i = 0; i < in.branches(); ++i) {
        if (auto b = in.branch(i); b && !(*b)->spilled()) {
          in.Unkeep(**b);
        }
      }
    }
  };
  parts.requested = Clock::now();
  Clock::time_point evicted = parts.requested;
  Clock::time_point loaded;
  if (resident_ == nullptr) {
    // Nothing resident: the first load of its weights (its state is
    // resident since setup, or written back by a swap that failed and
    // could not be undone: restored with them).
    // Whatever another model still holds from a swap that failed and could
    // not be undone goes out first (state written back).
    for (const auto& other : models_) {
      if (other.get() != &m) {
        if (auto r = EvictPaged(*other); !r) {
          return r;
        }
        if (other->llm() && other->HasRetainedState() &&
            std::ranges::find(spilled_, other.get()) == spilled_.end()) {
          spilled_.push_back(other.get());
        }
      }
    }
    // Every model's plans and graphs stay charged inside the budget: room
    // for the whole load is made through the reclaim order first, as a
    // swap's is, or a load after a failed swap (or a recovery) could be
    // refused for room again and again with nothing to free it.
    if (auto room = MakeRoomForSwap(m, {}); !room) {
      return room;
    }
    std::vector<engine::LoadStats> log;
    std::vector<catalog::ExtentId> all;
    for (const auto& [extent, generation] : m.everything().extents) {
      (void)generation;
      all.push_back(extent);
    }
    unkeep_incoming();
    // A load that fails (a read error) leaves no model resident: the next
    // activation loads one whole again.
    if (auto r = node_.Load(all, std::format("{}'s first load", m.name()), log); !r) {
      return r;
    }
    loaded = Clock::now();
    parts.loaded = log.empty() ? 0 : log.front().extents;
  } else {
    Served& out = *resident_;
    parts.from = out.name();
    const bool conversation = out.llm() && out.HasRetainedState();
    parts.with_state = out.llm() && spill_state.value_or(conversation);
    std::vector<catalog::ExtentId> extents;
    std::vector<catalog::ExtentId> unchanged;
    const SpillView view{.out = &out, .in = &m};
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The spill budget holds a swap's write-back too, counted as the swap
      // leaves it: all the outgoing model's state on disk, the incoming
      // model's resident (its conversations, which then stop counting, are
      // never deleted for it). Past it, the least recently used spilled
      // conversations are deleted first, the outgoing model's among them,
      // but only once the swap's room is made below: one a request holds
      // is not, and a swap that would not fit even with every other one
      // deleted is refused here, before anything moves or is deleted.
      if (!SpillBudgetFits(0, view)) {
        return Error("the swap's retained request state exceeds the spill budget");
      }
      // Kept across a restart (D-105): each conversation's last verify's
      // owed restore runs first (no request holds the stream during a
      // swap), so the write-back holds it whole. Before the room is sized:
      // whatever it charges is counted there.
      if (l.keeps()) {
        for (std::size_t i = 0; i < l.branches(); ++i) {
          if (auto b = l.branch(i); b && l.ResidentStateBytes(**b) != 0) {
            (void)l.SettleIdle(**b);
          }
        }
      }
    }
    // Its plans and graphs stay (D-090 as amended 2026-10-02): charged
    // inside the budget, they go only when the reclaim order needs their
    // room, here first if the incoming model does not fit beside them. The
    // room is checked again after each reclaim, and a swap it cannot make
    // room for is refused before anything moves. What goes out is its
    // weights and, with its state, all of that (how much a write-back
    // writes, and whether the budget deletes some of it, does not change
    // what the swap frees; the reclaim order spills no idle state here).
    const std::vector<catalog::ExtentId> weights = out.weights();
    const std::vector<catalog::ExtentId> kept = out.kept_state();
    const std::uint64_t graphs_before = out.graphs().kept + m.graphs().kept;
    {
      std::vector<catalog::ExtentId> going = weights;
      going.insert(going.end(), kept.begin(), kept.end());
      if (parts.with_state) {
        const std::vector<catalog::ExtentId> state = out.state();
        going.insert(going.end(), state.begin(), state.end());
        for (const catalog::ExtentId extent : out.unchanged_state()) {
          if (std::ranges::find(state, extent) == state.end()) {
            going.push_back(extent);
          }
        }
      }
      if (auto room = MakeRoomForSwap(m, going); !room) {
        return room;
      }
    }
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The room is made: the budget's deletions now, which the check above
      // found enough (nothing since spilled or restored; a deletion that
      // fails still refuses the swap before anything moves).
      if (!KeepWithinSpillBudget(0, view)) {
        return Error("the swap's retained request state exceeds the spill budget");
      }
      // Its state first, so its write-backs start first; what nothing wrote
      // since its slot's spill file last held it is released without
      // writing (an incremental spill), before the swap.
      unchanged = out.unchanged_state();
      for (const catalog::ExtentId extent : out.state()) {
        if (std::ranges::find(unchanged, extent) == unchanged.end()) {
          extents.push_back(extent);
        }
      }
      // What it writes back: its resident state (a conversation the reclaim
      // order spilled already is on disk, and passed over).
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b) {
          parts.spilled_bytes += l.ResidentStateBytes(**b);
        }
      }
    }
    extents.insert(extents.end(), kept.begin(), kept.end());
    extents.insert(extents.end(), weights.begin(), weights.end());
    const std::uint64_t graphs_after = out.graphs().kept + m.graphs().kept;
    parts.dropped_graphs = graphs_before > graphs_after ? graphs_before - graphs_after : 0;
    unkeep_incoming();
    if (!unchanged.empty()) {
      if (auto r = node_.Evict(unchanged, {.unchanged = true}); !r) {
        out.StateWrittenBack(false);
        if (auto undone = UndoSwap(out, m); !undone) {
          resident_ = nullptr;  // the next activation loads one whole
        }
        return r;
      }
    }
    scheduler::SwapReport report;
    if (auto r = node_.Swap(std::move(extents), m.everything(), handoff_, report); !r) {
      if (parts.with_state) {
        out.StateWrittenBack(false);
      }
      // Never the process: whatever of the incoming model came in goes out
      // again and the outgoing one comes back whole, so both stay usable and
      // only the request that needed the swap fails.
      if (auto undone = UndoSwap(out, m); undone) {
        Log(std::format("swap {} -> {} failed and was undone: {}", out.name(), m.name(),
                        r.error()));
      } else {
        Log(
            std::format("swap {} -> {} failed ({}) and could not be undone ({}): no model is "
                        "resident until the next one loads whole",
                        out.name(), m.name(), r.error(), undone.error()));
        if (parts.with_state && std::ranges::find(spilled_, &out) == spilled_.end()) {
          spilled_.push_back(&out);
        }
        resident_ = nullptr;
      }
      return r;
    }
    if (parts.with_state) {
      out.StateWrittenBack(true);
      spilled_.push_back(&out);
      // Its conversations are wholly on disk: their records follow (D-105).
      auto& l = static_cast<Llm&>(out);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b && !(*b)->spilled()) {
          l.KeepBranch(**b);
        }
      }
    }
    evicted = report.evicted;
    loaded = report.loaded;
    parts.evicted = report.evictions;
    parts.loaded = report.loads;
  }
  parts.evict = Seconds(evicted - parts.requested);
  Clock::time_point restored = evicted;
  if (restoring) {
    restored = std::max(restored, times_.Latest(m.state()));
    parts.restore = Seconds(restored - evicted);
    std::erase(spilled_, &m);
  }
  parts.page_in = Seconds(loaded - restored);
  parts.read_bytes = m.weight_read_bytes() + (restoring ? m.state().size() * kExtent : 0);
  resident_ = &m;
  // A model whose checks after its load fail is not left resident: its
  // kernels index what those checks cover unchecked (DeepSeek's hash
  // routing, Qwen3.8's n-gram hash) or replay graphs at its places (D-090).
  // Evicted (its state written back), its next activation loads it whole
  // and checks it again; only the requests that needed it fail (D-102).
  const auto unload = [&](std::string error) -> Status {
    if (auto evicted = EvictPaged(m); !evicted) {
      error += std::format("; evicting it: {}", evicted.error());
      m.StateWrittenBack(false);
    } else if (m.llm() && m.HasRetainedState() &&
               std::ranges::find(spilled_, &m) == spilled_.end()) {
      // As a swap's write-back: its conversations are wholly on disk, and
      // their records follow (D-105).
      m.StateWrittenBack(true);
      spilled_.push_back(&m);
      auto& l = static_cast<Llm&>(m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b && !(*b)->spilled()) {
          l.KeepBranch(**b);
        }
      }
    }
    resident_ = nullptr;
    return Error(std::move(error));
  };
  if (auto r = m.AfterLoad(); !r) {
    return unload(std::format("{} after its load: {}", m.name(), r.error()));
  }
  parts.ready = Clock::now();
  parts.setup = Seconds(parts.ready - loaded);
  parts.total = Seconds(parts.ready - parts.requested);
  // Its places still pinned where it registered them (D-090).
  std::size_t unpinned = 0;
  const std::vector<catalog::ExtentId> managed = m.paged().managed_extents();
  if (auto r = node_.Call(
          [&]() -> Status {
            for (const catalog::ExtentId extent : managed) {
              unpinned += node_.scheduler().PlacePinned(extent) ? 0 : 1;
            }
            return {};
          },
          "checking places");
      !r) {
    return r;
  }
  if (unpinned != 0) {
    return unload(std::format("{} of {}'s {} extents are not pinned at their places", unpinned,
                              m.name(), managed.size()));
  }
  if (auto r = m.CheckPlaces(); !r) {
    return unload(std::format("{}'s places: {}", m.name(), r.error()));
  }
  return {};
}

}  // namespace jitllm::runtime
