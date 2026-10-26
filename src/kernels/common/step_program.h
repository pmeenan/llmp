// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A step program (docs/engine-components.md, "Execution"; D-107): one
// planned and lowered step of a model, as the engine runs it, whatever
// kernel source implements it. A lowering (GGML's, the image kernels',
// EXL3's, the CPU reference's) makes one from a planned step and the
// addresses it binds; the engine stages its inputs, runs it, captures it
// for replay (D-090) and checks its tensors against the catalog (BP-A1)
// through this interface alone, so the engine names no kernel library.
//
// One virtual call per step at most: Run (or, once a graph is captured,
// the launch's replay), and before it Position for a program that is never
// captured. Everything else is per plan: what the program needs (its
// scratch, its host bytes, its launches), its input and output slots, the
// tensors it touches, binding and recording.
//
// StepLaunch is the backend's launch context: the GGML launch context on
// CUDA, the device runtime's recorded work (RecordingLaunch,
// recording_launch.h) for any backend that records a stream, the fake
// device's in the tests. A program and its launch come from one backend.
//
// Vendor-free: nothing here names a kernel library or a device API.

#ifndef LLMP_KERNELS_COMMON_STEP_PROGRAM_H_
#define LLMP_KERNELS_COMMON_STEP_PROGRAM_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/comparison_boundary.h"
#include "execution/step_shape.h"
#include "format/encoding.h"
#include "ir/graph.h"
#include "providers/device_execution.h"

namespace llmp::components {
struct HostInputDef;
}
namespace llmp::execution {
class BoundPlan;
}

#include "kernels/common/comparison.h"
#include "kernels/common/comparison_region.h"
#include "kernels/common/execution_resources.h"

namespace llmp::kernels {

inline std::uint64_t NextComparisonProgramNonce() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  while (value != 0 && value != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  }
  return 0;
}

class StepLaunch;
class RoutedProgram;

// What fails when queueing, binding or recording: refused with nothing queued
// (the step may be retried or run otherwise), or of unknown effect
// (something was or may have been queued: the job fences it).
enum class StepError : std::uint8_t { kRejected, kUnknown };
struct StepFailure {
  StepError error = StepError::kRejected;
  std::string detail;
};

// What an input slot holds each step, which the engine writes into its
// staging from the step's shape and tokens (components/step.h's step
// inputs, with the same rules):
//   kTokens, kPositions  a segment's token IDs or positions (I32, one a
//                        row; a joined slot, every segment's in order;
//                        positions in `sections` copies, one after another:
//                        a multi-section rotation's);
//   kSegmentBytes, kStepBytes  bytes the caller supplies per segment or
//                        for the step;
//   kOutputRows          I32 [outputs]: the joined rows the head reads,
//                        each segment's last `outputs` (a joined slot);
//   kStoreCells          I64 [segment rows]: the cell of `cells` each of
//                        the segment's rows is stored at (position, or
//                        position mod capacity for a ring); with `outputs`,
//                        I64 [outputs]: its last `outputs` rows' only
//                        (components/step.h kOutputCells: a target's tap
//                        rows, stored at their positions' cells of a
//                        drafter's tensor); the cells as `cells` names them
//                        (components/step.h CellKey: a fixed tensor's one
//                        cell, a chunk row's, a block's);
//   kMask / kMaskF32     F16 / F32 [read, mask_rows]: the segment's mask over the
//                        first `read` cells of `cells` (ir/op.h kMaskBuild:
//                        0 where a row's position sees the position the
//                        cell holds once the step's rows are stored, causal
//                        or within `window`, else -inf; rows past the
//                        segment's -inf): a mask built on the host.
//                        `ahead` also sees the step's later rows. A
//                        frozen read's cells (`cells.frozen`) hold the
//                        positions before the segment's rows, which the
//                        step never stores;
//   kFeatures            F32 [width, rows], joined: each row's feature (a
//                        drafter's, components/step.h kFeatures), which
//                        the engine copies on the device before the run
//                        from where the step's work names it (a stage's
//                        tap cell, the slot's carried feature): never
//                        staged from the host, so never in a recording.
//   kSideRows            I32 [count · rows]: the rows of a side table each
//                        step row reads (components/step.h kSideRows: the
//                        caller computes them from the token history);
//   kBlockPositions      I32 [sections · read]: the first position of each
//                        of the segment's first `read` blocks of
//                        `cells.block` positions, in `sections` copies.
enum class InputKind : std::uint8_t {
  kTokens,
  kPositions,
  kSegmentBytes,
  kStepBytes,
  kOutputRows,
  kStoreCells,
  kMask,
  kFeatures,
  kSideRows,
  kBlockPositions,
  // Component-declared host-computed and constant inputs; components/step.h
  // defines their values. Never opaque caller bytes: the shape and metadata
  // determine them.
  kHostComputed,
  kHadamard,
  kVisibleCells,  // I32 owner rows: completed blocks capped by real/scored capacity
  kMaskF32,       // The same mask recipe as kMask, stored as F32.
};

// A state tensor's cells as an input indexes them (components/step.h
// CellKey): its capacity, and whether position p is in cell p mod capacity.
struct InputCells {
  std::uint64_t capacity = 0;
  bool ring = false;
  bool frozen = false;  // a frozen read's (components/step.h CellKey)
  // The other cell rules (components/step.h CellKey): a fixed tensor's one
  // cell, a cell per block of `block` positions, a cell per chunk row from
  // `first_row`.
  bool fixed = false;
  std::uint32_t block = 0;
  bool per_row = false;
  std::uint32_t first_row = 0;
  bool positions = false;
  bool recent = false;
  bool operator==(const InputCells&) const = default;
};

struct InputSlot {
  std::uint64_t device_address = 0;
  std::uint64_t bytes = 0;
  ir::TensorId tensor{};
  std::uint8_t segment = ir::kJoinedSegment;
  InputKind kind = InputKind::kTokens;
  // kStoreCells' and kMask's.
  InputCells cells{};
  // kMask's: the cells it reads, its window (0: causal) and its rows
  // (the segment's, padded).
  std::uint64_t read = 0;
  std::uint32_t window = 0;
  std::uint32_t mask_rows = 0;
  // kStoreCells': 0, every row's cell; else the last `outputs` rows'.
  std::uint32_t outputs = 0;
  // kPositions' and kBlockPositions' copies; kSideRows' rows a step row.
  std::uint32_t sections = 1;
  std::uint64_t count = 0;
  // Host-computed inputs (components/step.h StepInput).
  const components::HostInputDef* host = nullptr;
  std::uint32_t ratio = 0;
  bool overlap = false;
  std::uint64_t ring = 0;
  // A block drafter's mask also sees later rows of its current block.
  bool ahead = false;
  // Joined verify saves use stable slots, each max_verify_rows wide.
  // Their row indices are supplied by the runner, independently of past.
  bool commit_save = false;
  std::uint16_t component = 0;         // kSideRows' source component
  std::uint64_t visible_capacity = 0;  // kVisibleCells: excludes padded/dummy cells
  // The same between consecutive runs (planner/plan.h PlanOptions::
  // invariant): a warm run (RunWarm) neither copies nor reads it anew.
  bool invariant = false;
};

// An output the step returns: copied to the host after the run, each
// segment's to its owner alone (independent outputs), a joined one whole.
struct OutputSlot {
  std::uint64_t device_address = 0;
  std::uint64_t bytes = 0;
  ir::TensorId tensor{};
  std::uint8_t segment = ir::kJoinedSegment;
  // A drafter head's next features (its segment's rows, F32 [width, rows],
  // after its logits): what the engine carries to the slot's next pass.
  bool features = false;
};

// A range the program reads or writes, by role: a weight, a request's
// state, the step's activations, inputs and outputs (the workspace), or a
// constant the plan holds (the runner's own memory). The coverage check
// (engine/coverage.h) holds each against the catalog.
struct TensorUse {
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
  ir::Role role = ir::Role::kActivation;
  bool routed_experts = false;  // a weight in a routed-expert slab
};

// The addresses a lowering binds: weights by role and part, state by
// reference (its segment names the shape's segment, so the owner's slot),
// and the activations' range. Activations at 0 mean a placeless
// measurement: the program reports what it needs but never runs.
struct ExpertPartBinding {
  std::uint64_t address = 0, bytes = 0, pitch = 0;
  format::TensorFormat format;
  format::PartRole role = format::PartRole::kData;
};
// Owner-declared enclosing allocation for one or more bounded parts.
struct BackingAllocation {
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;
  bool operator==(const BackingAllocation&) const = default;
};
// A real, owner-bounded address range available to a lowering. `alignment`
// is the guaranteed alignment of `base`; zero means the promise is unknown.
// The range is descriptive only and does not prove current residency.
struct BoundedAddress {
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
  // Optional owner-declared enclosing allocation. The bounded part remains
  // [base, base + bytes); this declares shared backing, not residency.
  std::optional<BackingAllocation> allocation = {};
  // Optional immutable content identity supplied by this same range's owner.
  // It describes the actual part, never residency or execution authority.
  std::optional<base::Sha256Digest> content_identity = {};
  bool operator==(const BoundedAddress&) const = default;
};
struct Addresses {
  std::function<std::uint64_t(const ir::WeightRef&, format::PartRole)> weight;
  // A row-paged weight's device slot capacity, replacing its on-disk row
  // count. Zero keeps the full tensor; only row lookup tables use it.
  std::function<std::uint64_t(const ir::WeightRef&)> weight_rows;
  std::function<std::uint64_t(const ir::StateRef&)> state;
  // A verify's save of `what` (components/step.h CommitSave: F32 [width,
  // the segment's rows]) where the caller keeps it, which the verify
  // stores and its commit reads; 0: none (a placeholder while measuring).
  std::function<std::uint64_t(std::uint32_t layer, std::uint16_t component, std::string_view what,
                              std::uint8_t segment)>
      save;
  // Capacity of a joined verify save (owners times max_verify_rows).
  // Zero retains the step's packed rows, for standalone lowerings.
  std::uint64_t save_rows = 0;
  std::uint64_t activations = 0;
  // Construction's address limit, which may be the owned virtual capacity.
  // It neither proves backing nor authorizes a launch. The consumer admits
  // the returned program's exact scratch requirement before binding/running.
  std::uint64_t activation_bytes = 0;
  // Physical reservation guarantees, identical for measurement and binding.
  // Zero/absent is unknown; neither logical offsets nor placeholders prove it.
  std::uint64_t activation_alignment = 0;
  std::function<std::uint64_t(const ir::StateRef&)> state_alignment{};
  // Exact opened artifact slice/slab binding, never inferred from packed size.
  std::function<std::optional<ExpertPartBinding>(std::uint16_t, std::string_view)> expert_part{};
  // A single-part weight's actual reserved address, by its complete binding.
  // This promise is available before measurement; zero means unknown.
  std::function<std::uint64_t(const ir::WeightRef&, format::PartRole)> weight_alignment{};
  // Optional physical bounds for proof consumers. Empty means that the
  // source cannot authenticate a contiguous range for this semantic object.
  std::function<std::optional<BoundedAddress>(const ir::WeightRef&, format::PartRole)>
      weight_range{};
  // Optional exact opened-artifact and row-slot representation for row-paged
  // weights. The physical address remains the separately authenticated range.
  std::function<std::optional<components::ComparisonRowStorage>(const ir::WeightRef&,
                                                                format::PartRole)>
      row_storage{};
  std::function<std::optional<BoundedAddress>(const ir::StateRef&)> state_range{};
  std::function<std::optional<BoundedAddress>(std::uint32_t layer, std::uint16_t component,
                                              std::string_view what, std::uint8_t segment)>
      save_range{};
  bool measuring() const { return activations == 0; }
};

// A captured sequence of a stream's work, replayed as one (a CUDA graph,
// the device runtime's recorded work). Destroyed only once no replay of it
// is queued (D-090: with its plan, before the launch and the memory it
// names).
enum class CaptureMeasurement : std::uint8_t { kDisabled, kGpuDuration };

class Recorded {
 public:
  // `owner`: the launch that recorded it, which alone replays it.
  explicit Recorded(const StepLaunch* owner) : owner_(owner) {}
  Recorded(const Recorded&) = delete;
  Recorded& operator=(const Recorded&) = delete;
  Recorded(Recorded&&) = delete;
  Recorded& operator=(Recorded&&) = delete;
  virtual ~Recorded() = default;
  // The recorded nodes (0 if the backend does not count them), and the
  // host time capturing it and instantiating it took.
  virtual std::size_t nodes() const = 0;
  virtual double capture_seconds() const = 0;
  virtual double instantiate_seconds() const = 0;
  // Instrumented captures only. Refuses an unlaunched or currently pending
  // replay; call after the enclosing owner establishes completion.
  virtual std::expected<double, StepFailure> gpu_duration_ns() const {
    return std::unexpected(StepFailure{.detail = "capture duration is unavailable"});
  }
  const StepLaunch* owner() const { return owner_; }

 private:
  const StepLaunch* owner_;
};

class StepLaunch {
 public:
  using Body = std::function<std::expected<void, StepFailure>()>;
  StepLaunch() = default;
  StepLaunch(const StepLaunch&) = delete;
  StepLaunch& operator=(const StepLaunch&) = delete;
  StepLaunch(StepLaunch&&) = delete;
  StepLaunch& operator=(StepLaunch&&) = delete;
  virtual ~StepLaunch() = default;
  // Applies only to subsequent captures. Ordinary launches allocate no events.
  virtual std::expected<void, StepFailure> SetCaptureMeasurement(CaptureMeasurement mode) {
    if (mode != CaptureMeasurement::kDisabled)
      return std::unexpected(StepFailure{.detail = "capture measurement is unsupported"});
    return {};
  }
  // What `body` queues on `stream` (the launch's own), recorded instead of
  // run and made replayable. Refused (kRejected) with nothing queued when
  // the backend cannot record it; kUnknown if the attempt may have queued
  // work.
  virtual std::expected<std::unique_ptr<Recorded>, StepFailure> Capture(
      providers::NativeStream stream, const Body& body) = 0;
  // One replay of `recorded` on `stream`.
  virtual std::expected<void, StepFailure> Launch(providers::NativeStream stream,
                                                  const Recorded& recorded) = 0;
  // Gives a backend-owned generated input a chance to retire/query its actual
  // existing update/retirement fence before owners or backing are dropped.
  // This is passive: ordinary immutable-payload launches never create a
  // fence merely to submit another replay.
  virtual std::expected<void, StepFailure> PollGeneratedInputs() { return {}; }
  // Establishes and polls a real tail fence when an update or final drain
  // must mutate/drop a generated payload. Ordinary launches without generated
  // inputs have nothing to drain.
  virtual std::expected<void, StepFailure> DrainGeneratedInputs() { return PollGeneratedInputs(); }
  // A fault the launch context latched (a GGML context's): every request
  // that shares it stops (the cohort's fault).
  virtual bool faulted() const { return retired_; }
  // Denies further execution; it grants no completion or release rights.
  void Retire() { retired_ = true; }
  bool retired() const { return retired_; }

 private:
  bool retired_ = false;
};

// Complete primitive weight-read inventory emitted by the lowering. A
// dynamic entry has only the exact selected routed readers proved there.
struct RoutedWeight {
  ir::WeightRef ref;
  format::TensorFormat format;
  std::vector<std::pair<std::string, format::PartRole>> parts;
  bool dynamic = false;
  bool operator==(const RoutedWeight&) const = default;
};

using PlanReportSink = std::function<void(std::string_view)>;
// Complete records without an assembly buffer. All views are borrowed only
// for the synchronous callback; begin/end carry no text, fragments omit '\n'.
enum class PlanReportEvent : std::uint8_t { kBegin, kFragment, kEnd };
using PlanRecordSink = std::function<void(PlanReportEvent, std::string_view)>;
// Registration has only fixed keys/static warnings and uint64 counters: each
// formatted representation is below 1KiB. Three simultaneous representations
// (summary, Printable, log line) plus allocator/control headers fit this bound.
inline constexpr std::uint64_t kRegistrationPlanReportMetadataBytes = 4096;
struct PlanCoverage {
  std::uint64_t primitive_fallbacks = 0;
  std::uint64_t available_fast_fallbacks = 0;
  std::uint64_t decode_expansion_rules = 0;
  std::uint64_t missing_numerical_qualifications = 0;
  std::uint64_t unqualified_numerical_candidates = 0;
  std::uint64_t logical_activation_bytes = 0;
  std::uint64_t physical_activation_bytes = 0;
  std::uint64_t unmaterialized_logical_bytes = 0;
  std::uint64_t unmaterialized_roots = 0;
};

// Live provenance for a lowered plan. These process-local graph identities
// authenticate which source and planned graph the program actually retained;
// they are never persistent digests or caller-supplied recipe facts.
struct ComparisonPlanIdentity {
  std::uint64_t source_graph_identity = 0;
  std::uint64_t planned_graph_identity = 0;
};

class StepProgram {
 public:
  StepProgram() : comparison_program_nonce_(NextComparisonProgramNonce()) {}
  StepProgram(const StepProgram&) = delete;
  StepProgram& operator=(const StepProgram&) = delete;
  StepProgram(StepProgram&&) = delete;
  StepProgram& operator=(StepProgram&&) = delete;
  virtual ~StepProgram() = default;

  bool HoldResources(ExecutionResources owner) {
    if (!owner || resources_) return false;
    resources_ = std::move(owner);
    return true;
  }
  bool UsesResources(const ExecutionResources& owner) const {
    return owner.instance_ && resources_.instance_ == owner.instance_;
  }
  ExecutionResources RetainedResources() const { return resources_; }
  // A compiler's temporary payment follows the returned program until its
  // caller transfers the retained graph/arena to a paid plan or future. The
  // base member dies after derived storage, including failed measurements.
  bool HoldConstruction(std::shared_ptr<void> owner) {
    if (!owner || construction_) return false;
    construction_ = std::move(owner);
    return true;
  }
  virtual void ConstructionPaid() { construction_.reset(); }

  // Updates backend-owned deterministic inputs for this exact shape. Most
  // programs have no generated inputs; a backend that owns one validates and
  // prepares it before any captured or eager execution.
  virtual std::expected<void, StepFailure> TryUpdateGeneratedInputs(
      const planner::StepShape& /*shape*/) {
    return {};
  }

  // What it needs: the activations' extent it was placed in, the backend's
  // ordinary pool scratch (excluding a launch's separately reserved slices),
  // the host bytes the plan holds (its arena, its steps), and
  // the launches it queues (a captured graph's cost, kGraphNodeHostBytes
  // each).
  virtual std::uint64_t scratch_bytes() const = 0;
  virtual std::uint64_t pool_bytes() const { return 0; }
  // Exact new main-pool backing for a private view when its producer reuses
  // already paid immutable lanes. This describes resources, never capability.
  virtual std::uint64_t private_pool_bytes() const { return pool_bytes(); }
  virtual std::uint64_t host_bytes() const = 0;
  virtual std::uint64_t launches() const = 0;
  // A backend proof is optional and owns all fields it returns. Empty means
  // no comparison was requested or the program has no proof metadata.
  virtual std::optional<ComparisonRecipeSeal> comparison_recipe() const { return {}; }
  virtual std::optional<ComparisonPlanIdentity> comparison_plan_identity() const { return {}; }
  // Synchronously compare this program's actual outside-region calls with a
  // retained peer. Unsupported implementations fail closed. This grants no
  // execution or selection authority; the borrowed expansion is program-owned
  // scratch and expires at the next comparison request.
  virtual ComparisonRemainderResult CompareComparisonRemainder(
      const StepProgram& candidate,
      const components::OwnedComparisonRegionBoundary& common_boundary,
      ComparisonCertificatePhase phase) const {
    (void)candidate;
    (void)common_boundary;
    (void)phase;
    return {.status = ComparisonRemainderStatus::kUnproved,
            .reason = "backend has no authenticated remainder comparison"};
  }
  // Transparent owner decorators return their retained implementation for
  // checked backend-specific peer access. This reference is borrowed only for
  // the synchronous comparison above.
  virtual const StepProgram& comparison_program() const { return *this; }
  // Comparison-only projection over an already lowered and bound program.
  // Unsupported backends fail closed; ordinary planning/execution is unchanged.
  virtual ComparisonRegionResult ProjectComparisonRegion(
      const components::OwnedComparisonRegionBoundary&) const {
    return {.status = ComparisonRegionStatus::kUnproved,
            .reason = "backend has no authenticated region projection"};
  }
  // Descriptor-only projection for pre-admission selection. The backend must
  // derive this from the same retained call/provenance and typed-access facts
  // as bound projection, and must not publish addresses or executable tokens.
  virtual ComparisonRegionDescriptorResult ProjectComparisonRegionDescriptor(
      const components::OwnedComparisonRegionBoundary&) const {
    return {.status = ComparisonRegionStatus::kUnproved,
            .reason = "backend has no authenticated descriptor region projection"};
  }
  // Production applicability certificate over a ready local boundary. Unlike
  // comparison projection, this returns no ports, addresses or run token.
  // Bound revalidates the same certificate after ordinary admission/binding.
  virtual ComparisonRegionDescriptorResult ProjectComparisonRegionCertificate(
      const components::OwnedComparisonRegionBoundary&, ComparisonCertificatePhase) const {
    return {.status = ComparisonRegionStatus::kUnproved,
            .reason = "backend has no authenticated production region certificate"};
  }
  // Reauthenticate a previously published capability without changing the
  // program-owned expansion/port scratch or minting a new serial.
  virtual bool ValidateComparisonProjection(const ComparisonRegionProjection&) const {
    return false;
  }
  // Run the actual side-effect-free prefix for a closed projection, then the
  // exact body span separately. Callers restage original input slots before
  // each replay; the prefix supplies crossing activations from real calls.
  virtual std::expected<void, StepFailure> RunComparisonPrefix(const ComparisonRegionProjection&,
                                                               StepLaunch&,
                                                               providers::NativeStream) const {
    return std::unexpected(StepFailure{.detail = "backend has no comparison prefix executor"});
  }
  virtual std::expected<void, StepFailure> RunComparisonRegion(const ComparisonRegionProjection&,
                                                               StepLaunch&,
                                                               providers::NativeStream) const {
    return std::unexpected(StepFailure{.detail = "backend has no comparison region executor"});
  }
  // Privacy-safe static plan facts only. No values, addresses or tensor names.
  // Retained report storage is included in host_bytes(); emission is on demand.
  virtual void ReportPlan(const PlanReportSink& /*sink*/) const {}
  // Same records as allocation-free fragments, including newlines. Called
  // only on demand; implementations use owned inventory and stack formatting.
  virtual void StreamPlanReport(const PlanReportSink& sink) const {
    if (!sink) return;
    ReportPlan([&sink](std::string_view record) {
      sink(record);
      sink("\n");
    });
  }

  void ReportPlanRecords(const PlanRecordSink& sink) const {
    if (!sink) return;
    bool begun = false;
    StreamPlanReport([&](std::string_view fragment) {
      while (!fragment.empty()) {
        if (!begun) {
          sink(PlanReportEvent::kBegin, {});
          begun = true;
        }
        const auto end = fragment.find('\n');
        const auto part = fragment.substr(0, end);
        if (!part.empty()) sink(PlanReportEvent::kFragment, part);
        if (end == std::string_view::npos) break;
        sink(PlanReportEvent::kEnd, {});
        begun = false;
        fragment.remove_prefix(end + 1);
      }
    });
    if (begun) sink(PlanReportEvent::kEnd, {});
  }

  virtual PlanCoverage plan_coverage() const { return {}; }

  virtual std::span<const InputSlot> inputs() const = 0;
  virtual std::span<const OutputSlot> outputs() const = 0;
  virtual std::span<const TensorUse> tensors() const = 0;
  // The implementations it runs (D-053), if its source records them.
  virtual const execution::BoundPlan* bound() const { return nullptr; }
  // False for a program whose launch parameters take the step's positions
  // (DeepSeek's HCA prefill plan patched per position): it is never
  // recorded, ahead or beside a run, and Position gives it each step's
  // positions before its run. Asked once, when it is cached.
  virtual bool capturable() const { return true; }
  // Absent means no source authority, not an empty set of reads. Required
  // for every cached shape of a source using dynamic weight exclusions.
  virtual std::optional<std::span<const RoutedWeight>> weight_reads() const { return {}; }
  // Non-null: whole Run/Record refuse; only the owned routed continuation runs.
  virtual RoutedProgram* routed() { return nullptr; }
  virtual const RoutedProgram* routed() const { return nullptr; }

  // Once, before its first run, on the driver after the current job (a
  // plan built ahead is bound on its installation): its implementations
  // bound and its scratch checked against the launch. A measurement never
  // binds.
  virtual std::expected<void, StepFailure> Bind(StepLaunch& launch) = 0;
  // A program that is not capturable: each segment's positions before its
  // rows (`past`, by the shape's segment order), before each run, on the
  // driver between jobs (no run of it is in flight). Refused (kRejected)
  // with nothing changed.
  virtual std::expected<void, StepFailure> Position(std::span<const std::uint32_t> /*past*/) {
    return {};
  }
  // The step's work queued on `stream`, after what is queued there.
  virtual std::expected<void, StepFailure> Run(StepLaunch& launch,
                                               providers::NativeStream stream) const = 0;
  // Whether a warm run differs from Run: the plan has invariant inputs
  // (InputSlot::invariant) or a prologue (planner/plan.h StepPlan::prologue).
  virtual bool warm() const { return false; }
  // The step's work after a run of this program whose invariant inputs were
  // the same, with nothing run on its activations since (the caller's to
  // know): its prologue skipped, whose results and the invariant inputs it
  // keeps in place. Its invariant inputs' copies are the caller's to skip.
  virtual std::expected<void, StepFailure> RunWarm(StepLaunch& launch,
                                                   providers::NativeStream stream) const {
    return Run(launch, stream);
  }
  // `body` (the engine's input copies, this program's Run and the output
  // copies) recorded as one replayable sequence; a program that cannot be
  // recorded refuses (kRejected) and runs launch by launch.
  virtual std::expected<std::unique_ptr<Recorded>, StepFailure> Record(
      StepLaunch& launch, providers::NativeStream stream, const StepLaunch::Body& body) const {
    return launch.Capture(stream, body);
  }

 protected:
  std::uint64_t comparison_program_nonce() const { return comparison_program_nonce_; }
  ComparisonRegionProjection MintComparisonRegionProjection(std::uint64_t generation,
                                                            std::uint64_t serial, std::size_t first,
                                                            std::size_t end,
                                                            ComparisonRegionRecipeSeal seal,
                                                            bool inputs_preserved = false) const {
    return ComparisonRegionProjection(comparison_program_nonce_, generation, serial, first, end,
                                      std::move(seal), inputs_preserved);
  }
  std::optional<std::pair<std::size_t, std::size_t>> AuthenticatedComparisonRegionSpan(
      const ComparisonRegionProjection& projection, std::uint64_t generation,
      std::uint64_t serial) const {
    if (comparison_program_nonce_ == 0 || projection.program_nonce_ != comparison_program_nonce_ ||
        projection.generation_ != generation || projection.serial_ != serial ||
        projection.first_ > projection.end_)
      return {};
    return std::pair(projection.first_, projection.end_);
  }

 private:
  const std::uint64_t comparison_program_nonce_;
  ExecutionResources resources_;
  std::shared_ptr<void> construction_{};
};

class ForwardingStepProgram : public StepProgram {
 public:
  explicit ForwardingStepProgram(std::unique_ptr<StepProgram> program = nullptr)
      : program_(std::move(program)) {}

  void ConstructionPaid() override {
    if (program_) program_->ConstructionPaid();
    StepProgram::ConstructionPaid();
  }
  std::expected<void, StepFailure> TryUpdateGeneratedInputs(
      const planner::StepShape& shape) override {
    return program_->TryUpdateGeneratedInputs(shape);
  }
  std::uint64_t scratch_bytes() const override { return program_ ? program_->scratch_bytes() : 0; }
  std::uint64_t pool_bytes() const override { return program_ ? program_->pool_bytes() : 0; }
  std::uint64_t private_pool_bytes() const override {
    return program_ ? program_->private_pool_bytes() : 0;
  }
  std::uint64_t host_bytes() const override { return program_ ? program_->host_bytes() : 0; }
  std::uint64_t launches() const override { return program_ ? program_->launches() : 0; }
  std::optional<ComparisonRecipeSeal> comparison_recipe() const override {
    return program_ ? program_->comparison_recipe() : std::nullopt;
  }
  std::optional<ComparisonPlanIdentity> comparison_plan_identity() const override {
    return program_ ? program_->comparison_plan_identity() : std::nullopt;
  }
  ComparisonRemainderResult CompareComparisonRemainder(
      const StepProgram& candidate,
      const components::OwnedComparisonRegionBoundary& common_boundary,
      ComparisonCertificatePhase phase) const override {
    return program_->CompareComparisonRemainder(candidate, common_boundary, phase);
  }
  const StepProgram& comparison_program() const override {
    return program_ ? program_->comparison_program() : *this;
  }
  ComparisonRegionResult ProjectComparisonRegion(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    return program_->ProjectComparisonRegion(boundary);
  }
  ComparisonRegionDescriptorResult ProjectComparisonRegionDescriptor(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    return program_->ProjectComparisonRegionDescriptor(boundary);
  }
  ComparisonRegionDescriptorResult ProjectComparisonRegionCertificate(
      const components::OwnedComparisonRegionBoundary& boundary,
      ComparisonCertificatePhase phase) const override {
    return program_->ProjectComparisonRegionCertificate(boundary, phase);
  }
  bool ValidateComparisonProjection(const ComparisonRegionProjection& projection) const override {
    return program_ && program_->ValidateComparisonProjection(projection);
  }
  std::expected<void, StepFailure> RunComparisonPrefix(
      const ComparisonRegionProjection& projection, StepLaunch& launch,
      providers::NativeStream stream) const override {
    return program_->RunComparisonPrefix(projection, launch, stream);
  }
  std::expected<void, StepFailure> RunComparisonRegion(
      const ComparisonRegionProjection& projection, StepLaunch& launch,
      providers::NativeStream stream) const override {
    return program_->RunComparisonRegion(projection, launch, stream);
  }
  void ReportPlan(const PlanReportSink& sink) const override {
    if (program_) program_->ReportPlan(sink);
  }
  void StreamPlanReport(const PlanReportSink& sink) const override {
    if (program_) program_->StreamPlanReport(sink);
  }
  PlanCoverage plan_coverage() const override {
    return program_ ? program_->plan_coverage() : PlanCoverage{};
  }
  std::span<const InputSlot> inputs() const override {
    return program_ ? program_->inputs() : std::span<const InputSlot>{};
  }
  std::span<const OutputSlot> outputs() const override {
    return program_ ? program_->outputs() : std::span<const OutputSlot>{};
  }
  std::span<const TensorUse> tensors() const override {
    return program_ ? program_->tensors() : std::span<const TensorUse>{};
  }
  const execution::BoundPlan* bound() const override {
    return program_ ? program_->bound() : nullptr;
  }
  bool capturable() const override { return program_ && program_->capturable(); }
  std::optional<std::span<const RoutedWeight>> weight_reads() const override {
    return program_ ? program_->weight_reads() : std::nullopt;
  }
  RoutedProgram* routed() override { return program_ ? program_->routed() : nullptr; }
  const RoutedProgram* routed() const override { return program_ ? program_->routed() : nullptr; }
  std::expected<void, StepFailure> Bind(StepLaunch& launch) override {
    return program_->Bind(launch);
  }
  std::expected<void, StepFailure> Position(std::span<const std::uint32_t> past) override {
    return program_->Position(past);
  }
  std::expected<void, StepFailure> Run(StepLaunch& launch,
                                       providers::NativeStream stream) const override {
    return program_->Run(launch, stream);
  }
  bool warm() const override { return program_ && program_->warm(); }
  std::expected<void, StepFailure> RunWarm(StepLaunch& launch,
                                           providers::NativeStream stream) const override {
    return program_->RunWarm(launch, stream);
  }
  std::expected<std::unique_ptr<Recorded>, StepFailure> Record(
      StepLaunch& launch, providers::NativeStream stream,
      const StepLaunch::Body& body) const override {
    return program_->Record(launch, stream, body);
  }

 protected:
  StepProgram* forwarded_program() { return program_.get(); }
  const StepProgram* forwarded_program() const { return program_.get(); }
  void set_forwarded_program(std::unique_ptr<StepProgram> program) {
    program_ = std::move(program);
  }
  void ResetProgram() { program_.reset(); }

 private:
  std::unique_ptr<StepProgram> program_;
};

inline void UpdatePlanCoverage(PlanCoverage& kept, const PlanCoverage& observed) {
  kept.primitive_fallbacks = std::max(kept.primitive_fallbacks, observed.primitive_fallbacks);
  kept.available_fast_fallbacks =
      std::max(kept.available_fast_fallbacks, observed.available_fast_fallbacks);
  kept.unqualified_numerical_candidates =
      std::max(kept.unqualified_numerical_candidates, observed.unqualified_numerical_candidates);
  kept.missing_numerical_qualifications =
      std::max(kept.missing_numerical_qualifications, observed.missing_numerical_qualifications);
  kept.decode_expansion_rules =
      std::max(kept.decode_expansion_rules, observed.decode_expansion_rules);
  kept.logical_activation_bytes =
      std::max(kept.logical_activation_bytes, observed.logical_activation_bytes);
  kept.physical_activation_bytes =
      std::max(kept.physical_activation_bytes, observed.physical_activation_bytes);
  kept.unmaterialized_logical_bytes =
      std::max(kept.unmaterialized_logical_bytes, observed.unmaterialized_logical_bytes);
  kept.unmaterialized_roots = std::max(kept.unmaterialized_roots, observed.unmaterialized_roots);
}

inline void ObservePlanCoverage(PlanCoverage& kept, const StepProgram& program) {
  UpdatePlanCoverage(kept, program.plan_coverage());
}

inline std::string FormatPlanCoverageReport(std::size_t measured_shapes, const PlanCoverage& c) {
  return std::format(
      R"({{"measured_shapes":{},"max_primitive_fallbacks":{},"max_available_fast_fallbacks":{},"max_decode_expansion_rules":{},"max_missing_numerical_qualifications":{},"max_unqualified_numerical_candidates":{},"max_logical_activation_bytes":{},"max_physical_activation_bytes":{},"max_omitted_logical_bytes":{},"max_omitted_roots":{},"warning":"{}"}})",
      measured_shapes, c.primitive_fallbacks, c.available_fast_fallbacks, c.decode_expansion_rules,
      c.missing_numerical_qualifications, c.unqualified_numerical_candidates,
      c.logical_activation_bytes, c.physical_activation_bytes, c.unmaterialized_logical_bytes,
      c.unmaterialized_roots,
      c.unqualified_numerical_candidates
          ? "UNQUALIFIED CANDIDATE arithmetic is active; results are diagnostic only"
      : c.missing_numerical_qualifications
          ? "arithmetic opportunities lack qualification; enable --debug-plans for details"
      : c.available_fast_fallbacks
          ? (c.decode_expansion_rules
                 ? "fast implementations are ineligible for some measured ops; decode products "
                   "expand; enable --debug-plans for selected recipes"
                 : "fast implementations are ineligible for some measured ops; enable "
                   "--debug-plans for selected fallbacks")
          : (c.decode_expansion_rules
                 ? "decode products expand; enable --debug-plans for selected recipes"
                 : ""));
}

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_STEP_PROGRAM_H_
