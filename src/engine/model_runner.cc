// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/model_runner.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <utility>

#include "engine/prefill_lookahead.h"
#include "engine/support.h"
#include "scheduler/scheduler.h"

namespace llmp::engine {

namespace {

namespace kn = llmp::kernels;
namespace pl = llmp::planner;
namespace sc = llmp::scheduler;
using support::Address;
using support::Error;
using support::Round;

std::unexpected<StepRefusal> Refused(std::string what, bool clean = true,
                                     std::optional<std::uint32_t> slot = std::nullopt,
                                     bool state_refused = false) {
  return std::unexpected(StepRefusal{
      .what = std::move(what), .clean = clean, .slot = slot, .state_refused = state_refused});
}

constexpr std::uint32_t PhaseBit(pl::Phase phase) {
  return std::uint32_t{1} << static_cast<std::uint32_t>(phase);
}

// The I32 values of one segment's rows (or every segment's, joined) for a
// tokens or positions input, written into `out` and the rest zeroed.
std::expected<void, std::string> WriteRows(std::span<std::byte> out,
                                           std::span<const std::int32_t> values) {
  const std::uint64_t bytes = values.size_bytes();
  if (bytes > out.size()) return Error("an input's rows exceed its slot");
  if (bytes != 0) std::memcpy(out.data(), values.data(), bytes);
  std::fill(out.begin() + static_cast<std::ptrdiff_t>(bytes), out.end(), std::byte{0});
  return {};
}

}  // namespace

SlotLifecycleOptions ModelRunner::LifecycleOptions(ModelRunner* self,
                                                   SlotLifecycleOptions options) {
  // A moved place or a recovery drops every plan and recording (they name
  // the old places), then whatever the caller does beside.
  options.drop_plans = [self, also = std::move(options.drop_plans)] {
    self->DropPlans();
    if (also) also();
  };
  return options;
}

ModelRunner::ModelRunner(PagedNode& node, RunnerConfig config, std::unique_ptr<StepSource> source,
                         int owner, std::uint32_t stream)
    : node_(node),
      config_(std::move(config)),
      source_(std::move(source)),
      owner_(owner),
      stream_(stream),
      slots_(node, LifecycleOptions(this, std::move(config_.slots)), owner, stream),
      memory_(node, owner, stream),
      runs_(config_.graphs) {}

ModelRunner::~ModelRunner() = default;

std::expected<std::unique_ptr<ModelRunner>, std::string> ModelRunner::Create(
    PagedNode& node, RunnerConfig config, std::unique_ptr<StepSource> source, int owner,
    std::uint32_t stream) {
  if (source == nullptr) return Error("a model runner needs a step source");
  if (!config.launch) return Error("a model runner needs a launch");
  if (config.setup_shapes.empty()) return Error("a model runner measures at least one shape");
  if (config.weights.size() > config.slots.stages.size()) {
    return Error("weights name more stages than the slots hold");
  }
  for (const StageWeights& w : config.weights) {
    if (w.weights == nullptr || !w.weights->opened())
      return Error("a stage's weights are not open");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): the constructor is private
  return std::unique_ptr<ModelRunner>(
      new ModelRunner(node, std::move(config), std::move(source), owner, stream));
}

std::vector<catalog::ExtentId> ModelRunner::weight_extents() const {
  std::vector<catalog::ExtentId> out;
  for (const StageWeights& w : config_.weights) {
    out.insert(out.end(), w.weights->extents().begin(), w.weights->extents().end());
  }
  return out;
}

std::vector<catalog::ExtentId> ModelRunner::managed_extents() const {
  auto result = weight_extents();
  for (const auto& extents : {slots_.state(), slots_.kept_state()}) {
    result.insert(result.end(), extents.begin(), extents.end());
  }
  return result;
}

// ------------------------------------------------------------------ its life

std::expected<void, std::string> ModelRunner::CheckShape(const pl::StepShape& shape) const {
  if (shape.segments.empty()) return Error("a step has no segments");
  if (shape.stage >= slots_.stages()) return Error("a step names no stage");
  if (shape.read_align == 0 || shape.read_align % 256 != 0) {
    return Error("a step's read alignment is not a multiple of 256");
  }
  const std::uint32_t context = slots_.layout(shape.stage).context;
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const pl::Segment& s = shape.segments[i];
    if (s.slot >= slots_.slots()) return Error(std::format("a step names slot {}", s.slot));
    if (i > 0 && s.slot <= shape.segments[i - 1].slot) {
      return Error("a step's segments are not in ascending, distinct slots");
    }
    if (s.rows == 0 || s.past > context || s.rows > context - s.past) {
      return Error(std::format("slot {}'s rows are outside its context", s.slot));
    }
    if (s.read_cells < s.past + s.rows || s.read_cells > context) {
      return Error(std::format("slot {}'s reads do not cover its rows", s.slot));
    }
    if (s.outputs > s.rows || (!s.want_head && s.outputs != 0)) {
      return Error(std::format("slot {}'s outputs exceed its rows", s.slot));
    }
  }
  return {};
}

pl::PlanKey ModelRunner::KeyOf(const pl::StepShape& shape) const {
  pl::PlanKey key = key_base_;
  key.shape = shape;
  return key;
}

kn::Addresses ModelRunner::AddressesFor(const pl::StepShape& shape, bool measuring) const {
  std::vector<std::uint8_t> owners;
  owners.reserve(shape.segments.size());
  for (const pl::Segment& s : shape.segments) owners.push_back(s.slot);
  kn::Addresses a;
  a.weight = config_.weight_address
                 ? config_.weight_address
                 : std::function<std::uint64_t(const ir::WeightRef&, format::PartRole)>(
                       [](const ir::WeightRef&, format::PartRole) { return std::uint64_t{0}; });
  a.state = [this, owners = std::move(owners)](const ir::StateRef& ref) -> std::uint64_t {
    if (ref.segment >= owners.size() || ref.stage >= slots_.stages()) return 0;
    const model::StateLayoutDerived& layout = slots_.layout(ref.stage);
    for (const auto& item : layout.items) {
      if (item.layer == ref.layer && item.tensor == ref.tensor) {
        return slots_.live(owners[ref.segment], ref.stage).base(0) + item.offset;
      }
    }
    return 0;
  };
  if (!measuring) {
    a.activations = node_.activations().base;
    a.activation_bytes = node_.activations().bytes;
  }
  return a;
}

Status ModelRunner::Setup() {
  if (setup_ || released_) return Error("model runner setup is not repeatable");
  if (auto set = slots_.Setup(); !set) return set;
  for (StageWeights& w : config_.weights) {
    if (auto reserved = w.weights->Reserve(node_, w.places, w.slabs); !reserved) return reserved;
  }
  std::uint64_t activations = 0, pool = 0, inputs = 0, outputs = 0;
  for (const pl::StepShape& shape : config_.setup_shapes) {
    if (auto checked = CheckShape(shape); !checked) {
      return Error(std::format("a setup shape: {}", checked.error()));
    }
    auto program = source_->Build(shape, AddressesFor(shape, true));
    if (!program) return Error(std::format("measuring a step: {}", program.error()));
    const kn::StepProgram& p = **program;
    activations = std::max(activations, p.scratch_bytes());
    pool = std::max(pool, p.pool_bytes());
    std::uint64_t in = 0, out = 0;
    for (const kn::InputSlot& slot : p.inputs()) in += Round(slot.bytes, 256);
    for (const kn::OutputSlot& slot : p.outputs()) out += Round(slot.bytes, 256);
    inputs = std::max(inputs, in);
    outputs = std::max(outputs, out);
    plan_floor_bytes_ = std::max(plan_floor_bytes_, p.host_bytes());
  }
  // A quarter's margin over the widest measured, as every family's.
  activations_needed_ =
      Round(std::max<std::uint64_t>(activations + activations / 4, 1), kPagedExtent);
  pool_needed_ = pool == 0 ? 0 : Round(pool + pool / 4, kPagedExtent);
  inputs_staging_bytes_ = Round(std::max<std::uint64_t>(inputs + inputs / 4, 256), 256);
  outputs_staging_bytes_ = Round(std::max<std::uint64_t>(outputs + outputs / 4, 256), 256);
  host_input_bytes_ = Round(inputs_staging_bytes_ + outputs_staging_bytes_, kPagedExtent);
  auto in = memory_.Pinned(inputs_staging_bytes_);
  if (!in) return Error(in.error());
  auto out = memory_.Pinned(outputs_staging_bytes_);
  if (!out) return Error(out.error());
  inputs_staging_ = *in;
  outputs_staging_ = static_cast<std::byte*>(*out);
  runs_.SetStaging(inputs_staging_, inputs_staging_bytes_);
  setup_ = true;
  return {};
}

Status ModelRunner::Register() {
  if (!setup_ || registered_ || released_) return Error("model runner registration order");
  for (StageWeights& w : config_.weights) {
    if (auto r = w.weights->Register(node_, owner_); !r) return r;
  }
  if (auto r = slots_.Register(); !r) return r;
  std::vector<catalog::ExtentId> pinned = weight_extents();
  const auto reserved = slots_.reserved_extents();
  pinned.insert(pinned.end(), reserved.begin(), reserved.end());
  if (auto r = node_.scheduler().PinPlaces(pinned); !r) {
    return Error(std::format("pinning the runner's places: {}", sc::ToString(r.error())));
  }
  registered_ = true;
  return {};
}

Status ModelRunner::Bind() {
  if (!registered_ || bound_ || released_) return Error("model runner bind order");
  auto launch = config_.launch();
  if (!launch) return Error(launch.error());
  if (*launch == nullptr) return Error("the launch is missing");
  launch_ = std::move(*launch);
  runs_.SetLaunch(launch_.get());
  account_.Bind(
      [this](std::uint64_t bytes, bool required) { return node_.ChargeHost(bytes, required); },
      [this](std::uint64_t bytes) { node_.UnchargeHost(bytes); });
  plans_.set_account(&account_);
  key_base_ = source_->KeyBase();
  key_base_.shape = {};
  std::vector<catalog::ExtentId> shared = weight_extents();
  for (const Mapped* mapped : {&node_.activations(), &node_.pool()}) {
    shared.insert(shared.end(), mapped->extents.begin(), mapped->extents.end());
  }
  const auto own = memory_.extents();
  shared.insert(shared.end(), own.begin(), own.end());
  std::vector<const PagedWeights*> weights;
  for (const StageWeights& w : config_.weights) weights.push_back(w.weights.get());
  if (auto r = slots_.Bind(std::move(shared), std::move(weights)); !r) return r;
  bound_ = true;
  return {};
}

Status ModelRunner::Release() {
  if (released_) return {};
  released_ = true;
  // The recordings name the launch and the memory: they go first (D-090).
  DropPlans();
  launch_.reset();
  std::vector<std::string> problems;
  memory_.ReleaseMappings(problems);
  slots_.Release(node_.memory(), problems);
  for (StageWeights& w : config_.weights) {
    if (auto r = w.weights->Release(node_.memory()); !r) problems.push_back(r.error());
  }
  return support::Joined(problems);
}

// --------------------------------------------------------------------- plans

void ModelRunner::DropPlans() { plans_.Clear(); }

void ModelRunner::ReclaimCandidates(std::uint32_t owner, bool running,
                                    std::vector<memory::ReclaimCandidate>& out) {
  if (released_ || slots_.faulted()) return;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  CollectPlans(caches, owner, running, out);
}

std::uint64_t ModelRunner::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  if (released_ || slots_.faulted()) return 0;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  return ReclaimPlan(caches, kind, id);
}

std::expected<ModelRunner::Plans::Entry*, std::string> ModelRunner::Install(
    const pl::StepShape& shape, std::unique_ptr<StepPlanned> planned, double seconds,
    const std::function<void()>& transfer) {
  if (planned == nullptr || planned->program == nullptr) return Error("no program to install");
  kn::StepProgram& program = *planned->program;
  if (!planned->bound) {
    if (auto bound = program.Bind(*launch_); !bound) {
      return Error(std::format("binding a step: {}", bound.error().detail));
    }
    planned->bound = true;
  }
  // Every slot names the shape's segment it carries, or the step's: a
  // program for another shape would read or return another owner's rows.
  const auto segment_ok = [&](std::uint8_t segment) {
    return segment == ir::kJoinedSegment || segment < shape.segments.size();
  };
  for (const kn::InputSlot& in : program.inputs()) {
    if (!segment_ok(in.segment)) return Error("a step's input names no segment of its shape");
  }
  for (const kn::OutputSlot& out : program.outputs()) {
    if (!segment_ok(out.segment)) return Error("a step's output names no segment of its shape");
  }
  Coverage checked;
  CheckCoverage(node_, owner_, program.tensors(), "a step's ", checked);
  if (checked.violations != 0) {
    return Error(std::format("catalog coverage: {}", checked.first_violation));
  }
  coverage_.tensors += checked.tensors;
  const std::uint64_t host = program.host_bytes();
  const std::uint64_t launches = program.launches();
  planned->positioned = !program.capturable();
  if (transfer) transfer();
  Plans::Entry& entry = plans_.Add(KeyOf(shape), std::move(planned), host, launches, seconds);
  // Never recorded: no capture beside a run, on its first run or ahead.
  if (entry.planned->positioned) entry.runs[0].uncapturable = true;
  return &entry;
}

std::expected<ModelRunner::Plans::Entry*, std::string> ModelRunner::Planned(
    const pl::StepShape& shape) {
  if (Plans::Entry* found = plans_.Find(KeyOf(shape)); found != nullptr) return found;
  const auto started = std::chrono::steady_clock::now();
  auto program = source_->Build(shape, AddressesFor(shape, false));
  if (!program) return Error(std::format("planning a step: {}", program.error()));
  if (*program == nullptr) return Error("planning a step gave no program");
  auto planned = std::make_unique<StepPlanned>();
  planned->program = std::move(*program);
  return Install(shape, std::move(planned),
                 support::Seconds(std::chrono::steady_clock::now() - started), {});
}

std::expected<Copies, std::string> ModelRunner::OutputCopies(const kn::StepProgram& program) const {
  Copies copies;
  std::uint64_t at = 0;
  for (const kn::OutputSlot& slot : program.outputs()) {
    if (at > outputs_staging_bytes_ || slot.bytes > outputs_staging_bytes_ - at) {
      return Error("a step's outputs exceed their staging");
    }
    copies.push_back({Address(outputs_staging_ + at), slot.device_address, slot.bytes});
    at += Round(slot.bytes, 256);
  }
  return copies;
}

// --------------------------------------------------------------------- steps

std::expected<void, StepRefusal> ModelRunner::Admit(const pl::Segment& segment,
                                                    std::uint32_t read_align, bool advance) {
  if (!bound_ || released_) return Refused("the model runner is not bound");
  const std::uint32_t slot = segment.slot;
  if (slot >= slots_.slots()) return Refused(std::format("no slot {}", slot));
  if (auto active = slots_.CheckActive(slot); !active) return Refused(active.error(), true, slot);
  if (advance && segment.past != slots_.positions(slot)) {
    return Refused(std::format("slot {} holds {} positions, not {}", slot, slots_.positions(slot),
                               segment.past),
                   true, slot);
  }
  // Backed through every cell the step reads, not only its own rows' (a
  // common read reaches the widest owner's cells): reading absent backing
  // is a fault. Only this growth's refusal says capacity: an earlier one's
  // flag would mislabel a refusal for a borrow, a verify or a restore.
  slots_.ForgetRefusal(slot);
  if (auto reserved = slots_.ReserveStateThrough(
          slot, std::max(segment.past + segment.rows, segment.read_cells), read_align);
      !reserved) {
    // A clean capacity refusal leaves the slot usable and retryable, as
    // does a refusal before any growth (borrowed, a verify pending); a
    // growth failure that quarantined it is not clean. Never its peers.
    const bool refused = slots_.state_refused(slot);
    return Refused(reserved.error(), refused || slots_.usable(slot), slot, refused);
  }
  return {};
}

std::expected<StepReport, StepRefusal> ModelRunner::RunStep(const StepRequest& request) {
  if (!bound_ || released_) return Refused("the model runner is not bound");
  const pl::StepShape& shape = request.shape;
  if (auto checked = CheckShape(shape); !checked) return Refused(checked.error());
  if (request.work.size() != shape.segments.size()) {
    return Refused("a step's work does not match its segments");
  }
  if (slots_.faulted()) return Refused("the cohort is faulted", false);
  const PlanStep step;
  // Every owner's own cells (what it reads alone: its rows at its state's
  // alignment, exactly as a lone step's segment admits them) before any
  // owner's wider reads for the join (a joined alignment, a common read
  // through a peer's cells): widened first, an early owner could take what
  // a later owner's own rows need and keep it, refusing that owner in every
  // wave while capacity for both owners' own reads was there. A refusal
  // there is the owner's alone too; one in the widening is not (`widening`).
  // A widening refused for a later owner keeps the earlier owners' (clean,
  // leased growth they read once their positions reach it): bounded by the
  // join's reads, not released (docs/engine.md).
  if (shape.segments.size() > 1) {
    const std::uint32_t context = slots_.layout(shape.stage).context;
    for (pl::Segment own : shape.segments) {
      own.read_cells = static_cast<std::uint32_t>(std::min<std::uint64_t>(
          Round(std::uint64_t{own.past} + own.rows, slots_.read_align()), context));
      if (auto admitted = Admit(own, 0, request.advance); !admitted) {
        return std::unexpected(std::move(admitted.error()));
      }
    }
  }
  for (const pl::Segment& s : shape.segments) {
    if (auto admitted = Admit(s, shape.read_align, request.advance); !admitted) {
      admitted.error().widening = shape.segments.size() > 1;
      return std::unexpected(std::move(admitted.error()));
    }
  }
  if (auto places = slots_.CheckPlaces(); !places) return Refused(places.error());
  auto entry_of = Planned(shape);
  if (!entry_of) return Refused(entry_of.error());
  Plans::Entry& entry = **entry_of;
  const kn::StepProgram& program = *entry.planned->program;

  // Inputs, staged by slot.
  const auto fill = [&](std::size_t index,
                        std::span<std::byte> out) -> std::expected<void, std::string> {
    const kn::InputSlot& in = program.inputs()[index];
    const bool joined = in.segment == ir::kJoinedSegment;
    const std::size_t segment = in.segment;
    if (!joined && segment >= shape.segments.size()) return Error("an input names no segment");
    std::vector<std::int32_t> values;
    switch (in.kind) {
      case kn::InputKind::kTokens:
        if (!joined) {
          const auto tokens = request.work[in.segment].tokens;
          if (tokens.size() != shape.segments[in.segment].rows) return Error("a segment's tokens");
          return WriteRows(out, tokens);
        }
        for (std::size_t i = 0; i < shape.segments.size(); ++i) {
          const auto tokens = request.work[i].tokens;
          if (tokens.size() != shape.segments[i].rows) return Error("a segment's tokens");
          values.insert(values.end(), tokens.begin(), tokens.end());
        }
        return WriteRows(out, values);
      case kn::InputKind::kPositions:
        for (std::size_t i = 0; i < shape.segments.size(); ++i) {
          if (!joined && i != segment) continue;
          for (std::uint32_t r = 0; r < shape.segments[i].rows; ++r) {
            values.push_back(static_cast<std::int32_t>(shape.segments[i].past + r));
          }
        }
        return WriteRows(out, values);
      case kn::InputKind::kSegmentBytes: {
        if (joined) return Error("a segment's bytes name no segment");
        const auto bytes = request.work[in.segment].bytes;
        if (bytes.size() > out.size()) return Error("a segment's bytes exceed their slot");
        std::ranges::copy(bytes, out.begin());
        std::fill(out.begin() + static_cast<std::ptrdiff_t>(bytes.size()), out.end(), std::byte{0});
        return {};
      }
      case kn::InputKind::kStepBytes:
        if (request.step_bytes.size() > out.size())
          return Error("the step's bytes exceed their slot");
        std::ranges::copy(request.step_bytes, out.begin());
        std::fill(out.begin() + static_cast<std::ptrdiff_t>(request.step_bytes.size()), out.end(),
                  std::byte{0});
        return {};
    }
    return Error("an unknown input kind");
  };
  auto copies = runs_.Stage(program.inputs(), fill, 0);
  if (!copies) return Refused(copies.error());
  auto outputs = OutputCopies(program);
  if (!outputs) return Refused(outputs.error());
  if (entry.planned->positioned) {
    // Between jobs: no run of this program is in flight.
    std::vector<std::uint32_t> past;
    past.reserve(shape.segments.size());
    for (const pl::Segment& s : shape.segments) past.push_back(s.past);
    if (auto positioned = entry.planned->program->Position(past); !positioned) {
      return Refused(std::format("positioning a step: {}", positioned.error().detail));
    }
  }

  // The futures: deduplicated against this shape and each other, and the
  // cached ones protected (found) before any optional funding.
  const StepAhead& ahead = request.ahead;
  std::array<Plans::Entry*, 2> ahead_cached{};
  std::array<bool, 2> distinct{};
  for (std::size_t k = 0; k < ahead.shapes.size(); ++k) {
    if (!ahead.shapes[k] || !CheckShape(*ahead.shapes[k])) continue;
    if (*ahead.shapes[k] == shape) {
      ahead_cached[k] = &entry;
      continue;
    }
    bool duplicate = false;
    for (std::size_t prior = 0; prior < k; ++prior) {
      if (ahead.shapes[prior] && *ahead.shapes[k] == *ahead.shapes[prior]) {
        ahead_cached[k] = ahead_cached[prior];
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    distinct[k] = true;
    ahead_cached[k] = plans_.Find(KeyOf(*ahead.shapes[k]));
  }
  StepPlanRuns& runs = entry.runs[0];
  // A recording made ahead holds the layout its plan's inputs predicted;
  // one that differs from what was staged is dropped (its charge settles).
  if (runs_.graphs() && runs.has_graph() && *copies != runs.copies) {
    runs.DropGraph();
    ++lookahead_.dropped_ahead;
  }
  const bool repeats = ahead.shapes[0].has_value() && *ahead.shapes[0] == shape;
  bool capture =
      (runs.CaptureDue(runs_.graphs()) && (config_.capture_phases & PhaseBit(shape.phase)) != 0) ||
      (ahead.capture_ahead && repeats && runs_.graphs() && !runs.has_graph() && !runs.uncapturable);
  if (capture && !plans_.ChargeGraph(entry)) capture = false;
  const bool capture_first = capture && runs.eager_runs == 0;
  // The next shape's recording, beside this run, when it differs and is
  // already planned.
  Plans::Entry* upcoming = nullptr;
  Copies upcoming_copies, upcoming_outputs;
  if (ahead.capture_ahead && ahead.shapes[0] && !repeats && runs_.graphs() &&
      ahead_cached[0] != nullptr) {
    upcoming = ahead_cached[0];
    const kn::StepProgram& next = *upcoming->planned->program;
    auto layout = runs_.Layout(next.inputs(), 0);
    auto out = OutputCopies(next);
    if (upcoming->runs[0].has_graph() || upcoming->runs[0].uncapturable ||
        upcoming->runs[0].ahead_refused || !layout || !out || !plans_.ChargeGraph(*upcoming)) {
      upcoming = nullptr;
    } else {
      upcoming_copies = std::move(*layout);
      upcoming_outputs = std::move(*out);
    }
  }
  // Independent optional grants for the distinct missing futures.
  std::array<const pl::StepShape*, 2> build{};
  std::size_t build_count = 0;
  if (node_.threaded()) {
    for (std::size_t k = 0; k < ahead.shapes.size() && build_count < ahead.build; ++k) {
      if (distinct[k] && ahead_cached[k] == nullptr) build[build_count++] = &*ahead.shapes[k];
    }
  }
  PrefillLookaheadGroup<StepPlanned, 2> future(node_, plan_floor_bytes_);
  bool funded = false;
  for (std::size_t i = 0; i < build_count; ++i) {
    ++lookahead_.attempted;
    if (!future.Fund(i)) {
      ++lookahead_.refused;
      build[i] = nullptr;
    } else {
      funded = true;
    }
  }
  StepReport report;
  const std::function<void()> meanwhile = funded ? std::function<void()>([&] {
    const auto built = future.BuildAll(
        [&](std::size_t i) -> std::expected<std::unique_ptr<StepPlanned>, std::string> {
          if (build[i] == nullptr) return Error("not funded");
          auto program = source_->Build(*build[i], AddressesFor(*build[i], false));
          if (!program) return std::unexpected(std::move(program.error()));
          auto planned = std::make_unique<StepPlanned>();
          planned->program = std::move(*program);
          return planned;
        });
    for (std::size_t i = 0; i < built.size(); ++i) {
      lookahead_.built += built[i];
      report.built_ahead += built[i];
      lookahead_.build_seconds += future.seconds(i);
    }
  })
                                                 : std::function<void()>{};

  // The next shape's fresh state, prepared before the job.
  if (ahead.prepare && ahead.shapes[0] && CheckShape(*ahead.shapes[0])) {
    for (const pl::Segment& s : ahead.shapes[0]->segments) {
      // Through what that step will back (Admit): its reads, common or not.
      auto prepared = slots_.PrepareThrough(s.slot, std::max(s.past + s.rows, s.read_cells),
                                            ahead.shapes[0]->read_align);
      if (!prepared) {
        // A drain that cannot prove its page-ins retired quarantined a
        // state: then the refusal is not clean.
        const bool drained = slots_.FinishPreparations().has_value();
        return Refused(prepared.error(), drained);
      }
      report.prepared_ahead = report.prepared_ahead || *prepared;
    }
    lookahead_.prepared += report.prepared_ahead;
  }

  // What the step writes, recorded before the job (incremental spill).
  for (const pl::Segment& s : shape.segments) {
    if (auto recorded = slots_.RecordWrite(s.slot, s.past); !recorded) {
      if (!slots_.FinishPreparations()) return Refused(recorded.error(), false);
      return Refused(recorded.error(), true, s.slot);
    }
  }
  bool wrote = false, unknown = false;
  std::string failure;
  const auto posted = node_.Job(
      slots_.execution(),
      [&](providers::NativeStream native) {
        // What the owners' states owe (a verify's restore and commit)
        // before this step's own work.
        for (const pl::Segment& s : shape.segments) {
          for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
            LiveState& live = slots_.live(s.slot, stage);
            if (!live.owed()) continue;
            if (auto owed = live.QueueOwed(native); !owed) {
              failure = owed.error().detail;
              unknown = owed.error().unknown;
              return unknown ? sc::JobResult::kUnknown
                     : wrote ? sc::JobResult::kFailed
                             : sc::JobResult::kNotStarted;
            }
            wrote = true;
          }
        }
        const StepQueued q =
            runs_.Queue(runs, *copies, {}, program, *outputs, capture, graph_stats_, native);
        wrote = wrote || q.before || q.result.has_value();
        report.path = q.path;
        if (!q.result) {
          failure = q.result.error().detail;
          unknown = q.result.error().error == kn::StepError::kUnknown;
          return unknown ? sc::JobResult::kUnknown
                 : wrote ? sc::JobResult::kFailed
                         : sc::JobResult::kNotStarted;
        }
        if (upcoming != nullptr) {
          auto captured =
              runs_.CaptureAhead(upcoming->runs[0], upcoming_copies, *upcoming->planned->program,
                                 upcoming_outputs, graph_stats_, native);
          if (!captured) {
            // This run's work is queued; the capture's fault is the launch's.
            failure = captured.error().detail;
            unknown = true;
            return sc::JobResult::kUnknown;
          }
          report.captured_ahead = *captured;
        }
        return sc::JobResult::kQueued;
      },
      "a model step", stream_, meanwhile);
  // The job retired: every optional preparation drained before anything
  // else touches the state.
  const Status preparation = slots_.FinishPreparations();
  if (!posted || !failure.empty() || launch_->faulted()) {
    slots_.CheckFailedJob();
    for (const pl::Segment& s : shape.segments) {
      for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
        (void)slots_.live(s.slot, stage).Settle(false, wrote, unknown);
      }
    }
    if (launch_->faulted()) slots_.Fault();
    future.Abandon();
    // A faulted cohort (the launch's fault, or one CheckFailedJob found:
    // a scheduler fault, a lost lease) quarantined every owner, written or
    // not: never clean.
    return Refused(!failure.empty() ? failure
                   : !posted        ? posted.error()
                                    : "the launch faulted",
                   !wrote && !unknown && !slots_.faulted() && preparation.has_value());
  }
  if (!preparation) {
    // This step's writes completed but no position is published: every
    // owner becomes unusable rather than expose state at an old cursor.
    for (const pl::Segment& s : shape.segments) {
      for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
        slots_.live(s.slot, stage).Quarantine();
      }
    }
    future.Abandon();
    return Refused(preparation.error(), false);
  }
  const auto installed =
      future.InstallAfterCompletion([&](std::size_t i, std::unique_ptr<StepPlanned> planned,
                                        double seconds, const auto& transfer) {
        return Install(*build[i], std::move(planned), seconds, transfer).has_value();
      });
  for (const bool cached : installed) {
    lookahead_.cached += cached;
    report.cached_ahead += cached;
  }
  lookahead_.captured_ahead += report.captured_ahead;
  report.captured_first = capture_first && report.path == RunPath::kCaptured;
  lookahead_.captured_first += report.captured_first;
  Count(graph_stats_, report.path);

  // Each owner's own outputs, then its positions.
  report.outputs.resize(shape.segments.size());
  for (std::size_t i = 0; i < program.outputs().size(); ++i) {
    const kn::OutputSlot& slot = program.outputs()[i];
    const auto* from = support::Pointer((*outputs)[i][0]);
    const auto bytes = std::span(static_cast<const std::byte*>(from), slot.bytes);
    std::vector<std::byte>& to =
        slot.segment == ir::kJoinedSegment ? report.joined : report.outputs.at(slot.segment);
    to.insert(to.end(), bytes.begin(), bytes.end());
  }
  if (request.advance) {
    for (const pl::Segment& s : shape.segments) {
      if (auto set = slots_.SetPositions(s.slot, s.past + s.rows); !set) {
        // Admitted owners always take their positions; should one not, the
        // step's writes are done but not every cursor moved: every owner
        // becomes unusable, and no peer may run the step again (no slot).
        for (const pl::Segment& o : shape.segments) {
          for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
            slots_.live(o.slot, stage).Quarantine();
          }
        }
        return Refused(set.error(), false);
      }
    }
  }
  return report;
}

// -------------------------------------------------------------------- report

std::string ModelRunner::report_json() const {
  return std::format(
      R"({{"plans":{},"graphs":{},"plan_host_bytes":{},"graph_bytes":{},)"
      R"("reclaimed_plans":{},"reclaimed_graphs":{},)"
      R"("runs":{{"eager":{},"captured":{},"replayed":{},"refused":{}}},)"
      R"("coverage":{{"tensors":{},"violations":{}}},)"
      R"("lookahead":{{"attempted":{},"built":{},"cached":{},"refused":{},)"
      R"("captured_ahead":{},"captured_first":{},"dropped_ahead":{},"prepared":{}}},)"
      R"("activations":{},"pool":{},"host_inputs":{},"plan_floor":{}}})",
      plans_.size(), plans_.graphs(), plans_.host_bytes(), plans_.graph_bytes(),
      plans_.reclaimed_plans(), plans_.reclaimed_graphs(), graph_stats_.eager,
      graph_stats_.captured, graph_stats_.replayed, graph_stats_.refused, coverage_.tensors,
      coverage_.violations, lookahead_.attempted, lookahead_.built, lookahead_.cached,
      lookahead_.refused, lookahead_.captured_ahead, lookahead_.captured_first,
      lookahead_.dropped_ahead, lookahead_.prepared, activations_needed_, pool_needed_,
      host_input_bytes_, plan_floor_bytes_);
}

}  // namespace llmp::engine
