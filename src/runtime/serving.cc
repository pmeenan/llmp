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
#include <utility>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/check.h"
#include "base/report.h"
#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
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
#include "platform/memory_pressure.h"
#include "platform/path_trust.h"
#include "runtime/memory_guard.h"
#include "runtime/model_limits.h"
#include "runtime/prefill.h"
#include "runtime/swap_room.h"
#include "scheduler/programs.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"

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

namespace fs = std::filesystem;
namespace ja = jitllm::artifact;

constexpr std::uint64_t kExtent = engine::kPagedExtent;
// The most a tokenizer or template file may be.
constexpr std::size_t kMaxTokenizerBytes = std::size_t{64} << 20U;
// Extents the page-in observer tracks (the M3 models use about 100,000).
constexpr std::size_t kObservedExtents = std::size_t{1} << 18U;
// Each model's prefill chunk when its prefill_chunk is not configured
// (docs/runtime-serving.md#prefill-chunks-and-cancellation): the fastest
// measured through the runtime at 8K and 32K tokens within the memory
// bound (DeepSeek: 4,096 rows 12-15% faster than 2,048, 8,192 no faster);
// then capped by the model at its context (prefill.h). DeepSeek's
// attention mask is sized for the whole context, so above 262,144 tokens
// its default stays 2,048 rows: at 1,048,576, 4,096 would fix 2.14 GiB
// more and take that from the measured 1M conversation's state.
constexpr std::uint32_t kDsv4PrefillRows = 4096;
constexpr std::uint32_t kDsv4DeepPrefillRows = 2048;
constexpr std::uint32_t kDsv4WidePrefillContext = 262144;  // the widest context at 4,096
constexpr std::uint32_t kQwen38PrefillRows = 4096;
// The draft depth of each request in a Qwen3.8 wave of more than one.
constexpr std::uint32_t kSharedWaveDepth = 2;
// DeepSeek's independent request slots (engine/dsv4_runner.h): waves of up to four.
constexpr std::uint32_t kDsv4RequestSlots = 4;

// The reclaim order's cost of idle conversation state (Server::Reclaim):
// its write-back and its restore, at the node's rates measured so far, and
// before any at the rates measured on a GB10 (spark-b, DeepSeek's four
// slots' state, 1.42 GB written back at 11.0 GB/s and read again at 14.5
// GB/s; docs/experiments/memory-pressure). Dropping it instead (no spill
// budget) costs its recomputation: 64 s a GiB measured for DeepSeek's
// prefill (16,384 tokens, 368 MiB of state in 22.9 s), the lower of the two
// models' (Qwen3.8's state is denser a token).
constexpr double kSpillBytesPerSecond = 11.0e9;
constexpr double kRestoreBytesPerSecond = 14.5e9;
constexpr double kRecomputeSecondsPerByte = 64.0 / static_cast<double>(1ULL << 30U);
// A graph's capture and instantiation a GiB counted, measured on a GB10
// (0.33–0.45 s in the services' logs): what a graph's capture weighs
// against before any graph is held (Server::Reclaim).
constexpr double kGraphSecondsPerGiB = 0.36;
// How often the driver's maintenance looks (between units and when idle).
constexpr auto kMaintenanceInterval = std::chrono::milliseconds(250);

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// A file the configuration names, under its trust rules (D-073): every
// directory on the way and the file itself root's or the runtime user's
// and writable by nobody else; opened without following links, and the
// file the walk approved.
std::expected<std::string, std::string> ReadTrustedFile(const fs::path& path, uid_t trusted,
                                                        std::size_t limit) {
  auto walked = platform::WalkTrusted(path, trusted, false);
  if (!walked) {
    return Error(std::format("{}: {}", path.string(), walked.error()));
  }
  if (!walked->exists) {
    return Error(std::format("{} does not exist", path.string()));
  }
  const int fd =
      ::open(walked->resolved.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY);
  if (fd < 0) {
    return Error(std::format("{}: {}", path.string(), Errno(errno)));
  }
  struct stat status{};
  std::string bytes;
  std::string problem;
  if (::fstat(fd, &status) != 0) {
    problem = Errno(errno);
  } else if (!S_ISREG(status.st_mode) || status.st_dev != walked->status.st_dev ||
             status.st_ino != walked->status.st_ino) {
    problem = "not the regular file the trust walk approved";
  } else if (platform::OthersCanWrite(status, trusted, fd)) {
    problem = "other users can write it";
  } else if (std::cmp_greater(status.st_size, limit)) {
    problem = std::format("larger than {} bytes", limit);
  } else {
    bytes.resize(static_cast<std::size_t>(status.st_size));
    std::size_t done = 0;
    while (done < bytes.size()) {
      const ssize_t n =
          ::pread(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(done));
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        problem = n < 0 ? Errno(errno) : "shorter than it was";
        break;
      }
      done += static_cast<std::size_t>(n);
    }
  }
  (void)::close(fd);
  if (!problem.empty()) {
    return Error(std::format("{}: {}", path.string(), problem));
  }
  return bytes;
}

// An installed artifact, opened under the store's trust rules (D-056's
// integrity rests on them) and its ID checked; the runner opens it again.
std::expected<ja::Artifact, std::string> OpenTrusted(const fs::path& store, const std::string& id) {
  ja::OpenOptions options;
  options.expected_id = id;
  options.trusted_owner = ::geteuid();
  auto opened = ja::Artifact::Open(store / id, options);
  if (!opened) {
    return Error(std::format("artifact {}: {}", id, opened.error().ToString()));
  }
  return std::move(*opened);
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

// ---------------------------------------------------------------- the models

// DeepSeek V4 Flash (engine/dsv4_runner.h), with DSpark as its drafter.
class Dsv4 final : public Llm {
 public:
  Dsv4(engine::PagedNode& node, const config::ModelEntry& entry, const config::RuntimeRoles& roles,
       bool plain, int index)
      : artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    node_ = &node;
    checkpoint_directory_ = roles.spill;
    speculate_ = !drafter_id_.empty() && entry.speculation && !plain;
    wave_mode_ = execution::AdaptiveWaveMode(kWaveCost, WaveForce(entry.wave_form));
    context_ = entry.context;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = entry.context;
    configured_rows_ = entry.prefill_chunk;
    max_rows_ = PrefillChunkRows(
        entry.context, entry.prefill_chunk,
        entry.context > kDsv4WidePrefillContext ? kDsv4DeepPrefillRows : kDsv4PrefillRows,
        model::Dsv4MostRows(model::Dsv4Flash(), entry.context));
    options_.max_rows = max_rows_;
    options_.graphs = true;
    // Four request slots share the weights and workspace, each with its own
    // conversation state: concurrent chat requests decode in waves
    // (engine/dsv4_runner.h), whose row-local products read each weight
    // once for all of them.
    options_.wave_slots = kDsv4RequestSlots;
    // The output-A/HCA prefill on every prefill chunk of 64 rows or more,
    // where it passes every registered quality control on both GGUFs under
    // the tie-aware rule (docs/experiments/ds4-output-prefix, "Default-on
    // acceptance").
    engine::SetDsv4ServedPrefill(options_);
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
    std::string kv;
    for (const ja::ListedFile& f : artifact->files()) {
      if (f.role == ja::FileRole::kSourceMetadata && f.path.ends_with(".kv.gguf") &&
          (kv.empty() || f.path.contains("-00001-of-"))) {
        kv = f.path.substr(5);
      }
    }
    if (kv.empty()) {
      return Error("the artifact keeps no GGUF metadata, so no tokenizer");
    }
    auto bytes = artifact->ReadMetadata(kv);
    if (!bytes) {
      return Error(std::format("{}: {}", kv, bytes.error().ToString()));
    }
    auto read = tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*bytes)));
    if (!read) {
      return Error(std::format("{}: {}", kv, read.error().ToString()));
    }
    auto created = tokenizer::Tokenizer::Create(std::move(read->spec));
    if (!created) {
      return Error(std::format("{}: {}", kv, created.error().ToString()));
    }
    tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
    FindThinkTokens();
    // A model is refused here, like Qwen3.8, when it could serve no turn.
    if (!read->has_chat_template) {
      return Error(std::format("{} keeps no chat template, so it has no chat turns", kv));
    }
    if (auto used = UseTemplate(read->chat_template); !used) {
      return used;
    }
    return {};
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
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
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
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted) override {
    const std::uint32_t rows = VerifyRows(pos, left, runner_.max_verify());
    std::vector<std::int32_t> drafts;
    std::vector<float> verified;
    if (auto r = NativeSlot(branch).DraftVerify(pos, all.back(), rows, drafts, verified); !r) {
      return r;
    }
    return Judge(branch, pos, rows, drafts, verified, kept, logits, drafted);
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
    const auto mode = wave_mode_.Choose(width, sampled);
    std::uint32_t tokens = 0;
    std::uint32_t complete = 0;
    auto waved = mode == execution::AdaptiveWaveMode::Mode::kPlain
                     ? PlainWave(prepared)
                     : SpeculativeWave(prepared, tokens, complete);
    // A sampling member's speculation accepts differently: not observed.
    if (waved && !sampled) {
      wave_mode_.Observe(width, mode, tokens, complete);
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
    const auto share = [](std::size_t count) {
      return std::max<std::uint32_t>(
          2, static_cast<std::uint32_t>(engine::Dsv4Runner::kWaveRows / count));
    };
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
      unit.result =
          SpecStepFor(*unit.branch, unit.step.all, unit.step.position, unit.step.left, unit.kept,
                      unit.step.need_logits ? &unit.logits : nullptr, unit.drafted);
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
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  fs::path store_;
  engine::Dsv4Options options_;  // before the runner, which keeps a reference
  engine::Dsv4Runner runner_;
  std::array<engine::Dsv4Runner::Slot*, engine::Dsv4Runner::kRequestSlots> native_slots_{};
  // With DSpark: draft-verify or plain waves, by width (RunPreparedGenerationWave).
  // The calibration: a draft-verify wave's time over a plain decode wave's,
  // by width, measured through the runtime with each form forced (GB10,
  // community GGUF, four 124-token chats, 2026-10-03: 151 / 78, 197 / 89
  // and 249 / 86 ms at widths 2, 3 and 4; docs/experiments/deepseek-
  // batching). To be measured again when a wave's cost changes.
  static constexpr execution::AdaptiveWaveMode::Costs kWaveCost = {0, 0, 1.94, 2.21, 2.90};
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
  execution::AdaptiveWaveMode wave_mode_{kWaveCost};
};

// Qwen3.8 Flash Next (engine/qwen38_runner.h), with its MTP block as its
// drafter.
class Qwen38 final : public Llm {
 public:
  Qwen38(engine::PagedNode& node, const config::ModelEntry& entry,
         const config::RuntimeRoles& roles, bool plain, int index)
      : artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        tokenizer_path_(entry.tokenizer),
        template_path_(entry.chat_template),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    node_ = &node;
    speculate_ = !drafter_id_.empty() && entry.speculation && !plain;
    checkpoint_directory_ = roles.spill;
    context_ = entry.context;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = entry.context;
    configured_rows_ = entry.prefill_chunk;
    // (The runner's fast graph builds no mask of every cell by every row, so
    // RE-037's bound does not cap its chunks.)
    max_rows_ = PrefillChunkRows(entry.context, entry.prefill_chunk, kQwen38PrefillRows,
                                 model::Qwen38MostRows(entry.context, false));
    options_.max_rows = max_rows_;
    options_.graphs = true;
    // Four execution slots share weights and workspace while each branch
    // retains its own native state: up to four requests decode in one wave
    // (engine/qwen38_wave_plan.h), whose row-local products read each weight
    // once (C2/C4 HTTP screens: +9.8%/+20% over two slots of pairs).
    options_.wave_slots = 4;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
      options_.draft_rows = max_rows_ >= 4 ? 3 : 2;
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
    const auto source = [&](std::string_view kept, const std::optional<fs::path>& named,
                            std::string_view key) -> std::expected<std::string, std::string> {
      auto meta = artifact->ReadMetadata(kept);
      if (meta) {
        return std::move(*meta);
      }
      if (meta.error().rule != ja::Rule::kFileSet) {  // kept, but not as listed
        return Error(std::format("{}: {}", kept, meta.error().ToString()));
      }
      if (!named.has_value()) {
        return Error(
            std::format("the artifact keeps no {} and models.{}.{} is not set", kept, name_, key));
      }
      return ReadTrustedFile(named.value_or(fs::path()), ::geteuid(), kMaxTokenizerBytes);
    };
    auto json = source("tokenizer.json", tokenizer_path_, "tokenizer");
    if (!json) {
      return std::unexpected(json.error());
    }
    auto spec = tokenizer::ReadHfTokenizer(*json);
    if (!spec) {
      return Error("tokenizer: " + spec.error().ToString());
    }
    auto created = tokenizer::Tokenizer::Create(std::move(*spec));
    if (!created) {
      return Error("tokenizer: " + created.error().ToString());
    }
    tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
    FindThinkTokens();
    auto text = source("chat_template.jinja", template_path_, "chat_template");
    if (!text) {
      return std::unexpected(text.error());
    }
    if (auto used = UseTemplate(*text); !used) {
      return used;
    }
    if (auto setup = runner_.Setup(); !setup) {
      return setup;
    }
    for (std::size_t i = 0; i < native_slots_.size(); ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) {
        return std::unexpected(slot.error());
      }
      native_slots_[i] = *slot;
    }
    return PrepareBranches(static_cast<std::uint32_t>(native_slots_.size()), 3);
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
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
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
    if (active.size() > native_slots_.size()) {
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

  Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                     std::uint32_t left, std::vector<std::int32_t>& kept,
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted) override {
    // Sampling's position keys keep their existing fixed draft schedule.
    // Greedy chooses its depth from deterministic acceptance observations.
    const auto depth = DraftDepthFor(branch);
    if (depth > options_.context - pos) {
      return Error("the drafts would pass the context");
    }
    std::vector<std::int32_t> drafts;
    if (auto r = NativeSlot(branch).Draft(all, drafts, nullptr, depth); !r) {
      return r;
    }
    // The verify: the anchor and its drafts, within the tokens left.
    const auto rows = std::min<std::uint32_t>(
        {static_cast<std::uint32_t>(drafts.size()) + 1, left, options_.context - pos});
    drafts.resize(rows - 1);
    std::vector<std::int32_t> input(all.begin(), all.end());
    input.insert(input.end(), drafts.begin(), drafts.end());
    std::vector<std::int32_t> argmax;
    std::vector<float> verified;
    const bool rows_needed = logits != nullptr || sampling(branch);
    if (auto r = NativeSlot(branch).Verify(input, pos, argmax, rows_needed ? &verified : nullptr);
        !r) {
      return r;
    }
    return JudgeVerify(branch, drafts, pos, depth, argmax, verified, kept, logits, drafted);
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
    BranchDecoding(branch) = execution::AdaptiveDepth(3);
    return NativeSlot(branch).Clear();
  }
  Status ReleaseIdleStateFor(Branch& branch) override {
    BranchDecoding(branch) = execution::AdaptiveDepth(3);
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
      std::vector<std::int32_t> input;
      std::vector<std::int32_t> argmax;
      std::vector<float> verified;
    };
    std::vector<Frame> frames(prepared.size());
    std::vector<engine::Qwen38Runner::DraftWork> drafts;
    drafts.reserve(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      Frame& frame = frames[i];
      // A shared wave drafts two tokens a request: each verify row past that
      // reads its own routed experts for every member, so the third draft's
      // acceptance no longer pays for itself (C2/C4 HTTP screens: +8%/+15%
      // over adaptive depth). A lone request keeps its adaptive depth.
      frame.depth =
          std::min(DraftDepthFor(*unit.branch), prepared.size() > 1 ? kSharedWaveDepth : 3U);
      if (frame.depth > context_ - unit.step.position) {
        return Error("the drafts would pass the context");
      }
      drafts.push_back({.slot = &NativeSlot(*unit.branch),
                        .history = unit.step.all,
                        .drafts = &frame.drafts,
                        .passes = frame.depth});
    }
    // Past two requests each drafts alone, on its own cached graphs: a draft
    // wave's graph keys every slot's pending rows (one to four), so a wider
    // wave's graphs would seldom repeat (C4 HTTP screen: 3.6% faster).
    if (drafts.size() > 2) {
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
      frame.input.assign(unit.step.all.begin(), unit.step.all.end());
      frame.input.insert(frame.input.end(), frame.drafts.begin(), frame.drafts.end());
      verifies.push_back(
          {.slot = &NativeSlot(*unit.branch),
           .history = frame.input,
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
    return {};
  }

 private:
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
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  std::optional<fs::path> tokenizer_path_;
  std::optional<fs::path> template_path_;
  fs::path store_;
  engine::Qwen38Options options_;  // before the runner, which keeps a reference
  engine::Qwen38Runner runner_;
  std::array<engine::Qwen38Runner::Slot*, engine::Qwen38Runner::kRequestSlots> native_slots_{};
};

// The Qwen-Image-2.1 pipeline (engine/qwen_image_runner.h).
class QwenImage final : public Image {
 public:
  QwenImage(engine::PagedNode& node, const config::ModelEntry& entry,
            const config::RuntimeRoles& roles, const ServingOptions& serving, int index)
      : composition_id_(entry.composition.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    options_.store = roles.installed;
    options_.composition = composition_id_;
    options_.noise = serving.image_noise;
    options_.out = roles.spill;
    options_.prompt = serving.image_prompt;
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

std::expected<std::vector<std::int32_t>, std::string> Llm::EncodeText(std::string_view text) const {
  std::vector<tokenizer::TokenId> ids;
  if (auto r = tokenizer_->Encode(text, {.add_bos_eos = true}, ids); !r) {
    return Error(r.error().ToString());
  }
  if (tokenizer_->bos() && (ids.empty() || ids.front() != *tokenizer_->bos())) {
    ids.insert(ids.begin(), *tokenizer_->bos());  // BOS first, as the references fed it
  }
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::expected<std::vector<std::int32_t>, std::string> Llm::RenderChat(
    const chat::Conversation& conversation, std::uint32_t* stable_boundary) const {
  if (stable_boundary != nullptr) {
    *stable_boundary = 0;
  }
  if (!template_) {
    return Error(std::format("{} has no chat template (D-067)", name_));
  }
  auto rendered = template_->Render(conversation, LocalTime());
  if (!rendered) {
    return Error(rendered.error().ToString());
  }
  std::vector<tokenizer::TokenId> ids;
  std::vector<std::size_t> span_tokens;
  if (auto r = tokenizer_->EncodeMarked(rendered->text, rendered->specials, {}, ids,
                                        stable_boundary == nullptr ? nullptr : &span_tokens);
      !r) {
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
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::string Llm::Detokenize(std::span<const std::int32_t> tokens) const {
  std::string text;
  if (auto r = tokenizer_->Decode(tokens, {}, text); !r) {
    return "(not decodable)";
  }
  return text;
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
  auto chosen = chat::ChatTemplate::ForText(text, chat::TokenFacts::From(*tokenizer_));
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

void Llm::FindThinkTokens() {
  think_start_ = tokenizer_->Find("<think>");
  think_end_ = tokenizer_->Find("</think>");
}

Status Llm::PrepareBranches(std::uint32_t count, std::uint32_t draft_depth) {
  if (branches_prepared_ || count == 0 || count > kMaxBranches || draft_depth == 0 ||
      default_branch_.generation_active_ || !default_branch_.history_.empty()) {
    return Error("native branches must be prepared once before conversation use");
  }
  for (std::uint32_t slot = 1; slot < count; ++slot) {
    extra_branches_[slot - 1] = std::unique_ptr<Branch>(new Branch(*this, slot));
    extra_branches_[slot - 1]->decoding_ = execution::AdaptiveDepth(draft_depth);
    extra_branches_[slot - 1]->saved_decoding_ = execution::AdaptiveDepth(draft_depth);
  }
  default_branch_.decoding_ = execution::AdaptiveDepth(draft_depth);
  default_branch_.saved_decoding_ = execution::AdaptiveDepth(draft_depth);
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
  return !branch.generation_active_ && branch.prompt_session_ == nullptr && !LeasedFor(branch);
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
      now - branch.history_used_.at >= retention_) {
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
  // What it writes (only what changed since its last spill): the measured
  // rate's bytes.
  const std::uint64_t bytes = SpillWriteBytesFor(branch);
  const auto started = Clock::now();
  if (auto spilled = SpillFor(branch); !spilled) {
    if (set_aside) {
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
  return {};
}

Status Llm::RestoreSpilled(Branch& branch, bool& refused) {
  refused = false;
  if (!SpilledFor(branch)) {
    return {};
  }
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

Status Llm::SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                        std::uint32_t left, std::vector<std::int32_t>& kept,
                        std::vector<std::vector<float>>* logits, std::uint64_t& drafted) {
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
  branch.turn_checkpoints_.clear();
  branch.history_.clear();
  branch.needs_clear_ = true;
}

Status Llm::Clear(Branch& branch, const PromptSession* prompt) {
  CheckIdleGeneration(branch, prompt);
  branch.turn_checkpoints_.clear();
  if (auto r = ClearStateFor(branch); !r) {
    return r;
  }
  branch.history_.clear();
  branch.needs_clear_ = false;
  branch.history_used_ = Clock::now();
  return {};
}

Status Llm::ReleaseIdleState(Branch& branch) {
  CheckIdleGeneration(branch);
  branch.turn_checkpoints_.clear();
  branch.history_.clear();
  if (auto released = ReleaseIdleStateFor(branch); !released) {
    branch.needs_clear_ = true;  // its next use clears whatever is left
    return released;
  }
  branch.needs_clear_ = false;
  branch.history_used_ = Clock::now();
  return {};
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
  auto file = engine::CheckpointFile::Capture(
      *node_, checkpoint_directory_, *ranges,
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

Status Llm::ReusePrompt(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t& reused,
                        bool fresh, bool resume, const PrefillGoOn& go_on, bool& stopped,
                        const PromptSession* prompt) {
  reused = 0;
  const auto now = Clock::now();
  if (fresh || branch.needs_clear_ || !StateUsableFor(branch) ||
      now - branch.history_used_.at >= retention_) {
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
  const auto position = checkpoint.boundary.position;
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
      *node_, [&]() { return PrepareRestoreStateFor(branch, checkpoint.footprint); },
      [&](void* host, std::span<const engine::LiveState::Range> page) {
        return CopyCheckpointStateFor(branch, host, page, false);
      },
      progress);
  if (!restored) {
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

Llm::PromptSession::PromptSession(Llm& model, Branch& branch, std::span<const std::int32_t> tokens,
                                  std::uint32_t stable_boundary, bool fresh, bool resume)
    : model_(model),
      branch_(branch),
      tokens_(tokens.begin(), tokens.end()),
      stable_boundary_(stable_boundary),
      fresh_(fresh),
      resume_(resume) {}

Llm::PromptSession::~PromptSession() {
  base::Check(finished_, "an unfinished prompt session was destroyed");
}

std::expected<std::unique_ptr<Llm::PromptSession>, std::string> Llm::BeginPrompt(
    Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t stable_boundary, bool fresh,
    bool resume) {
  CheckBranch(branch);
  if (tokens.empty() || tokens.size() >= context_ || stable_boundary >= tokens.size()) {
    return Error("the prompt or its turn boundary is outside the context");
  }
  if (max_rows_ == 0) {
    return Error("a prompt needs a positive prefill chunk size");
  }
  if (branch.generation_active_ || branch.prompt_session_ != nullptr) {
    return Error("the conversation already has an active session");
  }
  auto session = std::unique_ptr<PromptSession>(
      new PromptSession(*this, branch, tokens, stable_boundary, fresh, resume));
  branch.prompt_session_ = session.get();
  return session;
}

bool Llm::PromptSession::done() const { return complete_ || finished_; }

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
    unit.rows = std::min(model_.max_rows_, end - at);
    if (unit.rows >= kPrefillTiledFrom) {
      unit.rows -= unit.rows % kPrefillRowTile;
    }
  }
  return unit;
}

std::uint32_t Llm::PromptSession::remaining_rows() const {
  if (done()) {
    return 0;
  }
  const std::vector<std::int32_t>& history = branch_.history_;
  std::size_t at = history.size();
  if (phase_ == Phase::kReuse) {
    // Not reused yet: what the branch's history shares with the prompt.
    at = static_cast<std::size_t>(std::ranges::mismatch(history, tokens_).in1 - history.begin());
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
    branch_.history_.clear();
  }
  last_.clear();
  complete_ = true;
  ran_ = std::unexpected(std::move(error));
  return ran_;
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
    const auto started = Clock::now();
    auto chunk =
        model_.RunChunkFor(branch_, std::span(tokens_).first(end), at, model_.speculate_, last_);
    if (!chunk) {
      std::string error =
          std::format("{}'s prefill: the chunk at {}: {}", model_.name_, at, chunk.error());
      if (defer_capacity && model_.StateRefusedFor(branch_) && model_.StateUsableFor(branch_)) {
        // Refused before dispatch: the history, `last` and the state are as
        // the previous unit left them, so the same unit can run again.
        refused_ = true;
        return std::unexpected(std::move(error));
      }
      return Fail(std::move(error));
    }
    branch_.history_.insert(branch_.history_.end(), tokens_.begin() + at, tokens_.begin() + end);
    branch_.history_used_ = Clock::now();
    run_.end = end;
    ++run_.chunks;
    run_.longest = std::max(run_.longest, Seconds(Clock::now() - started));
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
    branch_.prompt_session_ = nullptr;
    finished_ = true;
  }
  return ran_;
}

Status Llm::Prefill(Branch& branch, std::span<const std::int32_t> tokens, std::vector<float>& last,
                    const PrefillGoOn& go_on, PrefillRun* run) {
  CheckIdleGeneration(branch);
  branch.capacity_refused_ = false;
  if (branch.needs_clear_) {
    if (auto r = Clear(branch); !r) {
      return r;
    }
  }
  if (tokens.empty()) {
    return Error("nothing to prefill");
  }
  std::vector<std::int32_t> all = branch.history_;
  all.insert(all.end(), tokens.begin(), tokens.end());
  if (all.size() >= context_) {
    return Error(
        std::format("{} tokens do not fit {}'s context of {}", all.size(), name_, context_));
  }
  last.clear();
  auto completed = static_cast<std::uint32_t>(branch.history_.size());
  auto ran = RunPrefillChunks(
      static_cast<std::uint32_t>(branch.history_.size()), static_cast<std::uint32_t>(all.size()),
      max_rows_,
      [&](std::uint32_t at, std::uint32_t n) {
        auto chunk = RunChunkFor(branch, std::span(all).first(at + n), at, speculate_, last);
        // A capacity refusal ran nothing: the same chunk again once freed.
        for (std::size_t reclaimed = 0;
             !chunk && CapacityRefused(branch) && reclaimed < kMaxBranches && ReclaimFor(branch);
             ++reclaimed) {
          chunk = RunChunkFor(branch, std::span(all).first(at + n), at, speculate_, last);
        }
        if (chunk) {
          completed = at + n;
        } else {
          branch.capacity_refused_ = CapacityRefused(branch);
        }
        return chunk;
      },
      go_on);
  if (!ran) {
    branch.needs_clear_ = !StateUsableFor(branch);
    if (branch.needs_clear_) {
      branch.history_.clear();
    } else {
      branch.history_.assign(all.begin(), all.begin() + completed);
    }
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
  all.resize(ran->end);
  branch.history_ = std::move(all);
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
        branch.history_.clear();
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
    branch.history_.clear();
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
  return options_.stop && std::ranges::find(model_.stops_, token) != model_.stops_.end();
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
  if (step->speculative) {
    std::vector<std::int32_t> kept;
    std::vector<std::vector<float>> logits;
    auto ran = model_.SpecStepFor(branch_, step->all, step->position, step->left, kept,
                                  step->need_logits ? &logits : nullptr, out_.drafted);
    if (!ran) {
      return FailStep(ran.error());
    }
    // SpecStep already updated the legacy counter, including partial failures.
    return ApplySpeculative(std::move(kept), std::move(logits), 0);
  }
  std::vector<float> row;
  if (auto ran = model_.RunChunkFor(branch_, step->all, step->position, false, row); !ran) {
    return FailStep(ran.error());
  }
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
                       unit.step.need_logits ? &unit.logits : nullptr, unit.drafted);
  }
  return RunChunkFor(*unit.branch, unit.step.all, unit.step.position, false, unit.row);
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
                        .anchor_processed = false});
  }
  if (prepared.empty()) {
    return {};  // every session retains its own preparation error
  }
  std::ranges::sort(prepared, {},
                    [this](const PreparedGeneration& unit) { return BranchIndex(*unit.branch); });
  if (auto ran = RunPreparedGenerationWave(prepared); !ran) {
    for (PreparedGeneration& unit : prepared) {
      unit.session->out_.drafted += unit.drafted;
    }
    fail_cohort(ran.error());
    return ran;
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
        branch_.history_.assign(all_.begin(), all_.begin() + position_);
        branch_.needs_clear_ = false;
        Close();
        return ran_;
      }
    }
    branch_.needs_clear_ = true;
    branch_.history_.clear();
    Close();
    return ran_;
  }
  branch_.history_.assign(all_.begin(), all_.begin() + position_);
  branch_.history_used_ = Clock::now();
  if (out_.tokens.size() > options_.max_tokens) {
    out_.tokens.resize(options_.max_tokens);
  }
  if (out_.logits.size() > out_.tokens.size()) {
    out_.logits.resize(out_.tokens.size());
  }
  if (auto settled = model_.SettleFor(branch_); !settled) {
    branch_.needs_clear_ = true;
    branch_.history_.clear();
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
  branch.saved_valid_ = false;
  if (auto r = SettleFor(branch); !r) {
    return r;
  }
  auto ranges = UsedStateRangesFor(branch);
  if (auto r = SaveUsedStateFor(branch, host, ranges); !r) {
    return r;
  }
  branch.saved_ranges_ = std::move(ranges);
  branch.saved_history_ = branch.history_;
  branch.saved_cursor_ = CursorFor(branch);
  SaveDecodingStateFor(branch);
  branch.saved_valid_ = true;
  return {};
}

Status Llm::RestoreState(Branch& branch, void* host) {
  CheckIdleGeneration(branch);
  if (!branch.saved_valid_) {
    return Error("no completed conversation snapshot to restore");
  }
  if (host == nullptr && !branch.saved_ranges_.empty()) {
    return Error("the conversation snapshot has no source buffer");
  }
  if (auto r = SettleFor(branch); !r) {
    return r;
  }
  if (auto r = RestoreUsedStateFor(branch, host, branch.saved_ranges_); !r) {
    branch.needs_clear_ = true;
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
             .slot_bytes = engine::kSlabSlotBytes,
             .observer = &times_}),
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

Status Server::Make(const config::ModelEntry& entry, int index, std::string_view architecture) {
  if (entry.composition) {
    if (options_.image_noise.empty()) {
      Log(
          std::format("model {}: not registered: an image pipeline needs --image-noise (its "
                      "initial latents) in this build",
                      entry.name));
      return {};
    }
    models_.push_back(std::make_unique<QwenImage>(node_, entry, roles_, options_, index));
    return {};
  }
  if (architecture == "deepseek4") {
    models_.push_back(std::make_unique<Dsv4>(node_, entry, roles_, options_.plain, index));
  } else if (architecture == "qwen4exp") {
    models_.push_back(std::make_unique<Qwen38>(node_, entry, roles_, options_.plain, index));
  } else {
    return Error(std::format("model {}: no runner for architecture {}", entry.name, architecture));
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
  // Check every LLM's checkpoint ceiling before opening the device node or
  // constructing runners. An invalid model later in the list must not
  // cause any earlier model's resources to be allocated either.
  std::vector<std::string> architectures;
  architectures.reserve(config_.models.size());
  for (const config::ModelEntry& entry : config_.models) {
    if (entry.composition) {
      architectures.emplace_back();
      continue;
    }
    auto artifact = OpenTrusted(roles_.installed, entry.artifact.value_or(""));
    if (!artifact) {
      return Error(std::format("model {}: {}", entry.name, artifact.error()));
    }
    const std::string& architecture = artifact->model().architecture;
    if (auto context = CheckModelContext(architecture, entry.context); !context) {
      return Error(std::format("model {}: {}", entry.name, context.error()));
    }
    architectures.push_back(architecture);
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  // Each model's stream and owner: its place among the registered.
  std::size_t entry_index = 0;
  for (const config::ModelEntry& entry : config_.models) {
    if (auto r = Make(entry, static_cast<int>(models_.size()), architectures[entry_index]); !r) {
      return r;
    }
    ++entry_index;
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
    if (const auto report = m->allocation_report(); !report.empty()) {
      Log(std::format("allocation model {}: {}", m->name(), report));
    }
    if (const auto report = m->plan_report(); !report.empty()) {
      Log(std::format("model {}: plans {}", m->name(), report));
    }
    if (const auto reason = m->serial_reason(); !reason.empty()) {
      Log(std::format("model {}: serves one request at a time ({})", m->name(), reason));
    }
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
  const MemoryGuard bounds = {.largest = largest,
                              .host_inputs = host_inputs,
                              .plans = plans,
                              .available = available,
                              .fixed = fixed_};
  const auto guard = CheckMemoryGuard(bounds);
  Log(
      std::format("allocation guard: {{\"format\":\"jitllm-wave-startup-guard-v1\","
                  "\"fixed_catalog_bytes\":{},\"shared_activation_bytes\":{},"
                  "\"shared_scratch_bytes\":{},\"largest_weight_extent_bytes\":{},"
                  "\"host_input_bytes\":{},\"plan_floor_bytes\":{},\"uncounted_margin_bytes\":{},"
                  "\"available_after_fixed_bytes\":{},\"available_known\":{},"
                  "\"guard_passed\":{},\"registered_state_virtual_extent_bytes\":{}}}",
                  fixed_, activations, pool, largest, host_inputs, plans, kUncountedMargin,
                  available, available != 0, guard.has_value(), node_.StateCapacity()));
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
  Log(std::format(
      "serving {} models; budget {:.2f} GiB ({:.2f} GiB fixed, the workspace {:.2f}; host-built "
      "chunk inputs {:.2f} GiB and a step's plans {:.2f} GiB beside it); conversation state "
      "room {:.2f} GiB beside the largest weights; {:.2f} GiB available; idle conversations "
      "spill, kept {} h within {} GiB",
      models_.size(), static_cast<double>(budget_) / (1ULL << 30U),
      static_cast<double>(fixed_) / (1ULL << 30U), static_cast<double>(workspace_) / (1ULL << 30U),
      static_cast<double>(host_inputs_) / (1ULL << 30U),
      static_cast<double>(plans_) / (1ULL << 30U), static_cast<double>(room) / (1ULL << 30U),
      static_cast<double>(available) / (1ULL << 30U), config_.memory.retention_hours,
      config_.memory.spill_budget_gib));
  for (const auto& m : models_) {
    if (m->llm()) {
      static_cast<Llm&>(*m).set_retention(retention_);
      static_cast<Llm&>(*m).set_log([this](std::string_view text) { Log(text); });
    }
  }
  return {};
}

Status Server::TearDown() {
  if (torn_down_) {
    return {};
  }
  if (started_) {
    // What the incremental spills and swaps left unwritten, and the
    // reclaims that found too little to take (counts and bytes only).
    if (auto stats = node_.Stats(); stats) {
      Log(std::format(
          "write-backs released unchanged (nothing written): {} extents, {:.1f} MiB; reclaims "
          "that took nothing (too little to take): {}",
          stats->unchanged_writebacks,
          static_cast<double>(stats->unchanged_writeback_bytes) / (1U << 20U), reclaims_short_));
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
  const auto give_up = Clock::now() + std::chrono::minutes(2);
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
    if (Clock::now() > give_up) {
      return Error("backing no load took was not released");
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

std::uint64_t Server::SpilledBytes() {
  std::uint64_t bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    // A model a swap wrote back holds all its state on disk until it returns.
    const bool written_back = std::ranges::find(spilled_, m.get()) != spilled_.end();
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b) {
        bytes += written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b);
      }
    }
  }
  return bytes;
}

bool Server::KeepWithinSpillBudget(std::uint64_t extra) {
  std::size_t deleted = 0;
  std::uint64_t deleted_bytes = 0;
  bool fits = false;
  for (;;) {
    const std::uint64_t held = SpilledBytes();
    if (held <= spill_budget_ && extra <= spill_budget_ - held) {
      fits = true;
      break;
    }
    // The least recently used spilled conversation no request holds.
    Llm* oldest = nullptr;
    Llm::Branch* branch = nullptr;
    for (const auto& m : models_) {
      if (!m->llm()) {
        continue;
      }
      auto& l = static_cast<Llm&>(*m);
      const bool written_back = std::ranges::find(spilled_, m.get()) != spilled_.end();
      for (std::size_t i = 0; i < l.branches(); ++i) {
        auto b = l.branch(i);
        if (!b || !l.BranchIdle(**b) ||
            (written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b)) == 0) {
          continue;
        }
        if (branch == nullptr || l.LastUsed(**b) < oldest->LastUsed(*branch)) {
          oldest = &l;
          branch = *b;
        }
      }
    }
    if (branch == nullptr) {
      break;
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

std::uint64_t Server::Reclaim(std::uint64_t needed, bool states, std::string_view why,
                              const Served* running, std::optional<memory::ReclaimKind> below_kind,
                              bool partial) {
  if (reclaiming_ || !started_ || torn_down_ || needed == 0) {
    return 0;
  }
  reclaiming_ = true;
  if (running == nullptr) {
    running = resident_;
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
  std::uint64_t freed = 0;
  std::size_t dropped = 0;
  bool took = false;
  // All of it or nothing: a selection that cannot cover what is needed
  // reclaims nothing (the caller waits or refuses). Another round only when
  // a victim gave back less than its count (held, or gone meanwhile).
  for (int round = 0; round < 4 && freed < needed; ++round) {
    std::vector<memory::ReclaimCandidate> candidates;
    Llm* idle_owner = nullptr;
    std::uint32_t resident_index = 0;
    for (std::uint32_t i = 0; i < models_.size(); ++i) {
      models_[i]->ReclaimCandidates(i, models_[i].get() == running, candidates);
      if (models_[i].get() == resident_) {
        resident_index = i;
      }
    }
    // The running model's next step finds what its last ones used.
    memory::ProtectFloor(candidates, running != nullptr ? running->plan_floor_bytes() : 0);
    if (states && resident_ != nullptr && resident_->llm()) {
      idle_owner = static_cast<Llm*>(resident_);
      for (std::size_t slot = 0; slot < idle_owner->branches(); ++slot) {
        auto b = idle_owner->branch(slot);
        if (!b || !idle_owner->BranchIdle(**b) || (*b)->history().empty()) {
          continue;
        }
        const std::uint64_t bytes = idle_owner->ResidentStateBytes(**b);
        if (bytes == 0) {
          continue;
        }
        // A spill writes only what changed since its last one; a restore
        // reads it all.
        const auto size = static_cast<double>(bytes);
        const auto writes = static_cast<double>(idle_owner->SpillWriteBytes(**b));
        candidates.push_back(
            {.kind = memory::ReclaimKind::kIdleState,
             .owner = resident_index,
             .id = slot,
             .bytes = bytes,
             .last_use =
                 static_cast<std::uint64_t>(idle_owner->LastUsed(**b).time_since_epoch().count()),
             .restore_seconds = spill_budget_ == 0 ? size * kRecomputeSecondsPerByte
                                                   : (writes / spill_rate) + (size / restore_rate),
             .running = resident_ == running});
        memory::SetUse(candidates.back(), idle_owner->LastStamp(**b));
      }
    }
    cost = memory::KindCosts(candidates);
    double below = std::numeric_limits<double>::infinity();
    if (below_kind) {
      // What the charge would cost to restore a GiB: its kind's measured
      // cost, or the GB10's when none is held.
      double charged = cost[k(*below_kind)];
      if (charged == 0) {
        charged = *below_kind == memory::ReclaimKind::kGraph ? kGraphSecondsPerGiB : 0;
      }
      below = memory::ReclaimInflation() + charged;
    }
    const memory::ReclaimPlan plan = memory::SelectReclaim(candidates, needed - freed, below);
    if (!plan.sufficient && (!partial || plan.victims.empty())) {
      ++reclaims_short_;
      break;
    }
    std::uint64_t got_round = 0;
    for (std::size_t v = 0; v < plan.victims.size() && freed < needed; ++v) {
      const memory::ReclaimCandidate& c = candidates[plan.victims[v]];
      std::uint64_t got = 0;
      if (c.kind == memory::ReclaimKind::kIdleState) {
        if (idle_owner == nullptr) {
          continue;  // only the resident model's idle state is a candidate
        }
        auto b = idle_owner->branch(c.id);
        if (!b) {
          continue;
        }
        // Spill, not clear: within the spill budget (the least recently
        // used spilled state deleted first); past it, or with none, dropped.
        if (spill_budget_ != 0 && KeepWithinSpillBudget(idle_owner->SpillWriteBytes(**b))) {
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
        memory::RaiseReclaimInflation(plan.priorities[v]);
        ++count[k(c.kind)];
        freed_by[k(c.kind)] += got;
        freed += got;
        got_round += got;
      }
    }
    if (got_round == 0) {
      break;
    }
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
        "plans {:.2f}, idle state {:.2f}",
        mib(freed), mib(needed), why, count[k(K::kGraph)], mib(freed_by[k(K::kGraph)]),
        count[k(K::kPlan)], mib(freed_by[k(K::kPlan)]), count[k(K::kIdleState)],
        dropped != 0 ? "dropped" : "spilled", mib(freed_by[k(K::kIdleState)]), cost[k(K::kGraph)],
        cost[k(K::kPlan)], cost[k(K::kIdleState)]));
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

Status Server::EvictPaged(Served& m) {
  // Only what a swap moves: its weights and its conversation state (the
  // shared workspace and its own pinned runtime memory stay where they
  // are). State is written back to its places.
  std::vector<catalog::ExtentId> own = m.weights();
  const std::vector<catalog::ExtentId> state = m.state();
  own.insert(own.end(), state.begin(), state.end());
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

void Server::Maintain() {
  if (!started_ || torn_down_ || reclaiming_) {
    return;
  }
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
      if (!b || !l.BranchIdle(**b) || (*b)->history().empty() ||
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
  (void)KeepWithinSpillBudget(0);
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
    std::vector<engine::LoadStats> log;
    std::vector<catalog::ExtentId> all;
    for (const auto& [extent, generation] : m.everything().extents) {
      (void)generation;
      all.push_back(extent);
    }
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
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The spill budget holds a swap's write-back too: past it, the least
      // recently used of its conversations are deleted, not written.
      const auto writes = [&l]() {
        std::uint64_t bytes = 0;
        for (std::size_t i = 0; i < l.branches(); ++i) {
          if (auto b = l.branch(i); b) {
            bytes += l.SpillWriteBytes(**b);
          }
        }
        return bytes;
      };
      std::size_t deleted = 0;
      while (!KeepWithinSpillBudget(writes())) {
        Llm::Branch* oldest = nullptr;
        for (std::size_t i = 0; i < l.branches(); ++i) {
          if (auto b = l.branch(i); b && l.BranchIdle(**b) && l.ResidentStateBytes(**b) != 0 &&
                                    (oldest == nullptr || l.LastUsed(**b) < l.LastUsed(*oldest))) {
            oldest = *b;
          }
        }
        if (oldest == nullptr || !oldest->ReleaseIdleState()) {
          break;
        }
        ++deleted;
      }
      if (deleted != 0) {
        Log(
            std::format("{}: {} conversations deleted, the least recently used, for the spill "
                        "budget of {} GiB before its swap-out",
                        out.name(), deleted, spill_budget_ >> 30U));
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
    const std::vector<catalog::ExtentId> weights = out.weights();
    extents.insert(extents.end(), weights.begin(), weights.end());
    // Its plans and graphs stay (D-090 as amended 2026-10-02): charged
    // inside the budget, they go only when the reclaim order needs their
    // room, here first if the incoming model does not fit beside them. The
    // room is checked again after each reclaim, and a swap it cannot make
    // room for is refused before anything moves.
    const std::uint64_t graphs_before = out.graphs().kept + m.graphs().kept;
    {
      std::vector<catalog::ExtentId> going = extents;
      going.insert(going.end(), unchanged.begin(), unchanged.end());
      if (auto room = MakeRoomForSwap(m, going); !room) {
        parts.refused = true;
        return room;
      }
    }
    const std::uint64_t graphs_after = out.graphs().kept + m.graphs().kept;
    parts.dropped_graphs = graphs_before > graphs_after ? graphs_before - graphs_after : 0;
    if (!unchanged.empty()) {
      if (auto r = node_.Evict(unchanged, {.unchanged = true}); !r) {
        out.StateWrittenBack(false);
        if (auto undone = UndoSwap(out, m); undone) {
          parts.refused = true;
        } else {
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
        parts.refused = true;
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
  if (auto r = m.AfterLoad(); !r) {
    return Error(std::format("{} after its load: {}", m.name(), r.error()));
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
    return Error(std::format("{} of {}'s {} extents are not pinned at their places", unpinned,
                             m.name(), managed.size()));
  }
  if (auto r = m.CheckPlaces(); !r) {
    return Error(std::format("{}'s places: {}", m.name(), r.error()));
  }
  return {};
}

}  // namespace jitllm::runtime
