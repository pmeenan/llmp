// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/step_transaction.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "components/input_values.h"
#include "components/step.h"
#include "engine/batch_execution.h"
#include "engine/copy_sink.h"
#include "engine/prefill_lookahead.h"
#include "engine/step_inputs.h"
#include "engine/support.h"
#include "kernels/paging/paging.h"
#include "model/state_contract.h"
#include "providers/device_runtime.h"
#include "providers/storage.h"

namespace llmp::engine {
namespace {
namespace pl = planner;
namespace kn = kernels;
namespace sc = scheduler;
using step_inputs::WriteComputed;
using step_inputs::WriteRows;
using support::Error;
base::Sha256Digest HostInputIdentity(const StepRequest& request) {
  base::Sha256 hash;
  const auto bytes = [&](std::span<const std::byte> values) {
    const auto count = std::uint64_t{values.size()};
    hash.Update(std::as_bytes(std::span(&count, 1)));
    hash.Update(values);
  };
  bytes(request.step_bytes);
  bytes(std::as_bytes(request.side_rows));
  for (const auto& work : request.work) {
    bytes(std::as_bytes(work.tokens));
    bytes(work.bytes);
    bytes(std::as_bytes(work.history));
    const std::array<std::uint32_t, 2> feature{static_cast<std::uint32_t>(work.feature.kind),
                                               work.feature.position};
    bytes(std::as_bytes(std::span(feature)));
  }
  return hash.Finish();
}
constexpr std::uint32_t PhaseBit(pl::Phase phase) {
  return std::uint32_t{1} << static_cast<std::uint32_t>(phase);
}

std::unexpected<StepRefusal> Refused(std::string what, bool clean = true,
                                     std::optional<std::uint32_t> slot = std::nullopt) {
  return std::unexpected(StepRefusal{.what = std::move(what), .clean = clean, .slot = slot});
}
}  // namespace

std::expected<void, StepRefusal> StepTransaction::Validate(const StepRequest& request) const {
  if (runner_.unproven_.has_routed())
    return Refused("a routed step retains unproven effects", false);
  if (!runner_.bound_ || runner_.released_ || runner_.closing_)
    return Refused("the model runner is not bound");
  const pl::StepShape& shape = request.shape;
  if (auto checked = runner_.CheckShape(shape); !checked) return Refused(checked.error());
  if (request.work.size() != shape.segments.size()) {
    return Refused("a step's work does not match its segments");
  }
  if (!request.authorization.empty() && request.authorization.size() != shape.segments.size())
    return Refused("a step's authorization does not match its segments");
  for (const auto& future : request.ahead.shapes()) {
    if (future && future->phase == pl::Phase::kCommit)
      return Refused("commits are internal and cannot be lookahead steps");
  }
  const bool verify = shape.phase == pl::Phase::kVerify;
  if (shape.phase == pl::Phase::kCommit) return Refused("commits run only after a verify's Accept");
  if (verify && (shape.stage != 0 || request.advance)) {
    return Refused("a verify runs on the target and leaves its rows to their Accept");
  }
  if (verify && (runner_.config_.max_verify_rows == 0 ||
                 std::ranges::any_of(shape.segments, [&](const pl::Segment& s) {
                   return s.rows > runner_.config_.max_verify_rows;
                 }))) {
    return Refused(
        std::format("the runner verifies at most {} rows", runner_.config_.max_verify_rows));
  }
  if (shape.stage != 0 && request.advance) {
    return Refused("a drafter's step never commits positions");
  }
  return ValidateAuthorization(request, request.authorization);
}

std::expected<void, StepRefusal> StepTransaction::ValidateAuthorization(
    const StepRequest& request, std::span<const StepAuthorization> authorization) const {
  const auto& shape = request.shape;
  const bool mixed_origin = std::ranges::any_of(
      shape.segments, [](const pl::Segment& s) { return s.origin_phase.has_value(); });
  for (std::size_t segment_index = 0; segment_index < shape.segments.size(); ++segment_index) {
    const auto& segment = shape.segments[segment_index];
    const auto* permit = authorization.empty() ? nullptr : &authorization[segment_index];
    const auto& feature = request.work[segment_index].feature;
    if (mixed_origin && (!permit || permit->part_ == StepAuthorization::Part::kNone)) {
      return Refused("mixed-origin step requires an execution permit for every owner", true,
                     segment.slot);
    }
    if (permit && permit->part_ != StepAuthorization::Part::kNone &&
        (permit->runner_ != &runner_ || permit->slot_ != segment.slot ||
         permit->stage_ != shape.stage ||
         permit->phase_ != segment.origin_phase.value_or(shape.phase) ||
         permit->past_ != segment.past || permit->rows_ != segment.rows ||
         permit->outputs_ != segment.outputs || permit->want_head_ != segment.want_head ||
         permit->greedy_ != segment.greedy || permit->feature_.kind != feature.kind ||
         permit->feature_.position != feature.position))
      return Refused("step differs from its exact execution permit", true, segment.slot);
    if (shape.stage != 0 && !runner_.slots_.layout(shape.stage).borrowed.empty() &&
        !runner_.slots_.borrowed(segment.slot, 0) &&
        (!permit || !permit->borrowed_ ||
         !runner_.slots_.CheckBorrowPermit(*permit->borrowed_, segment.slot, shape.stage,
                                           segment.past, segment.rows)))
      return Refused("draft target state has no exact ordered borrow permit", true, segment.slot);
  }
  if (runner_.slots_.faulted()) return Refused("the cohort is faulted", false);
  return {};
}

std::expected<std::shared_ptr<StepTransaction>, StepRefusal> StepTransaction::Begin(
    ModelRunner& runner, const StepRequest& request) {
  StepTransaction validation(runner);
  if (auto checked = validation.Validate(request); !checked)
    return std::unexpected(std::move(checked.error()));
  // The fixed allowance includes the shared control block, array cookies
  // and allocator bookkeeping in addition to the exact array payloads.
  std::uint64_t bytes = sizeof(StepTransaction) + 512;
  const auto add = [&](std::uint64_t count, std::uint64_t size) {
    std::uint64_t storage = 0;
    return !__builtin_mul_overflow(count, size, &storage) &&
           !__builtin_add_overflow(bytes, storage, &bytes) && bytes <= SIZE_MAX;
  };
  if (!add(request.authorization.size(), sizeof(StepAuthorization)))
    return Refused("step permission inventory overflows");
  std::shared_ptr<StepTransaction> transaction;
  if (request.batch) {
    auto& data = *request.batch.data_;
    auto found = data.Find(runner, request.shape);
    if (!found) return Refused(found.error());
    auto& prepared = data.entries[*found];
    if (!prepared.transaction || !prepared.transaction->prepared_ ||
        request.authorization.size() > prepared.transaction->permits_capacity_ ||
        !prepared.transaction->PreparedInputsMatch(request))
      return Refused("step differs from its owned prepaid batch inputs or permission capacity");
    transaction = std::move(prepared.transaction);
    transaction->frame_payment_ = std::move(prepared.frame_payment);
    transaction->batch_ = request.batch;
    transaction->batch_index_ = *found;
    prepared.consumed = true;
    ++data.pending;
  } else {
    // Payment precedes the owner, both exact arrays and shared control storage.
    if (!runner.node_.ChargeHost(bytes, false))
      return Refused("step permission metadata does not fit");
    transaction = std::shared_ptr<StepTransaction>(new StepTransaction(runner));
    transaction->charge_ = bytes;
    transaction->permits_capacity_ = request.authorization.size();
    if (!request.authorization.empty())
      transaction->permits_ = std::make_unique<StepAuthorization[]>(request.authorization.size());
  }
  transaction->permits_count_ = request.authorization.size();
  for (std::size_t i = 0; i < transaction->permits_count_; ++i) {
    transaction->permits_[i] = request.authorization[i];
  }
  transaction->routed_owner_ = runner.config_.routed_owner.lock();
  return transaction;
}

std::expected<void, StepRefusal> StepTransaction::Admit(const StepRequest& request) {
  const auto& shape = request.shape;
  if (shape.segments.size() > 1) {
    for (std::size_t index = 0; index < shape.segments.size(); ++index) {
      pl::Segment own = shape.segments[index];
      own.read_cells = runner_.LoneRead(shape.stage, own.past, own.rows);
      const auto* permit = authorization().empty() ? nullptr : &authorization()[index];
      if (auto admitted = runner_.Admit(own, shape.stage, 0, request.advance, permit); !admitted) {
        return std::unexpected(std::move(admitted.error()));
      }
    }
  }
  for (std::size_t index = 0; index < shape.segments.size(); ++index) {
    const auto& s = shape.segments[index];
    const auto* permit = authorization().empty() ? nullptr : &authorization()[index];
    if (auto admitted = runner_.Admit(s, shape.stage, shape.read_align, request.advance, permit);
        !admitted) {
      admitted.error().widening = shape.segments.size() > 1;
      return std::unexpected(std::move(admitted.error()));
    }
  }
  return {};
}

Status StepTransaction::FinishPreparations() {
  auto result = runner_.slots_.FinishPreparations();
  if (!result) retirement_ = Retirement::kUnproven;
  return result;
}

void StepTransaction::JobRetired(const JobReport& report) {
  if (retirement_ == Retirement::kUnproven || report.retirement == JobReport::Retirement::kUnproven)
    retirement_ = Retirement::kUnproven;
  else if (report.retirement == JobReport::Retirement::kNotSubmitted)
    retirement_ = Retirement::kUnused;
  else
    retirement_ = Retirement::kRetired;
}

// All optional metadata is paid before allocation and retained with this
// transaction if native retirement is unknown. Builds use existing independent
// allowances; no callback retains a caller's prediction span.
struct StepTransaction::Lookahead {
  struct Slot {
    ModelRunner::Plans::Entry* cached = nullptr;
    bool keyed = false, distinct = false;
    const pl::StepShape* build = nullptr;
    std::optional<PrefillLookahead<StepPlanned>> future;
  };
  // Declared first: refunded after the arrays, key capacities and inventory die.
  HostPayment storage_payment, segment_payment, parameter_payment;
  StepAhead inventory;
  std::unique_ptr<Slot[]> slots;
  std::size_t count = 0;
  pl::PlanKey key, prior;
  void Abandon() {
    for (std::size_t i = 0; i < count; ++i)
      if (slots[i].future) slots[i].future->Abandon();
  }
};

void StepTransaction::LookaheadDelete::operator()(Lookahead* owned) const {
  if (!owned) return;
  auto payment = std::move(owned->storage_payment);
  owned->~Lookahead();
  ::operator delete(owned);
  payment.Reset();
}

// Caller spans and speculative shape references live only in this stack
// frame. The driver does not return until every Job callback/meanwhile is
// gone. The retained transaction separately owns only its paid witnesses.
struct StepTransaction::Frame {
  struct RowRead {
    ModelRunner::RowTableState* table = nullptr;
    PleRowPlan plan;
  };
  struct Saving {
    LiveState* live = nullptr;
    std::uint32_t rows = 0;
    bool any = false, queued = false;
  };
  const StepRequest* request;
  struct Admission {
    LiveState* live = nullptr;
    std::vector<LiveState::Range> ranges;
  };
  std::vector<Admission> admitted;
  StepRuns execution{false};
  base::Sha256Digest host_inputs{};
  bool batch = false, preflight = false;
  std::byte* output_base = nullptr;
  std::uint64_t output_bytes = 0;
  std::size_t feature_capacity = 0, carry_capacity = 0, tap_capacity = 0;

  ModelRunner::Plans::Entry* entry = nullptr;
  std::vector<std::vector<std::int32_t>> side_rows;
  std::vector<RowRead> row_reads;
  std::expected<Copies, std::string> copies, outputs, features, carries, taps;
  bool capture = false, capture_first = false;
  ModelRunner::Plans::Entry* upcoming = nullptr;
  ModelRunner::Plans::Entry* deferred_recording = nullptr;
  Copies upcoming_copies, upcoming_outputs;
  bool funded = false;
  StepReport report;
  std::function<void()> meanwhile;
  std::vector<Saving> saving;
  bool wrote = false, unknown = false, own_wrote = false;
  bool own_program_unknown = false;
  bool saves_queued = false, copies_queued = false;
  std::string failure;
  JobReport job_report;
  JobReport::Retirement prefix_retirement = JobReport::Retirement::kNotSubmitted;
  // One slot occupies one bit in the real cohort mask; no arbitrary owner cap.
  static constexpr auto kOwners = std::numeric_limits<SlotMask>::digits;
  std::array<std::uint32_t, kOwners> prefix_complete{}, prefix_partial{};
  std::size_t prefix_complete_count = 0, prefix_partial_count = 0;
  Status posted, preparation;
  explicit Frame(const StepRequest& input) : request(&input) {}
};

bool StepTransaction::PreparedInputsMatch(const StepRequest& request) const {
  return prepared_ && prepared_->host_inputs == HostInputIdentity(request);
}

StepTransaction::StepTransaction(ModelRunner& runner) : runner_(runner) {}

StepTransaction::~StepTransaction() {
  // The runner keeps every queued/unproven owner linked. Release's existing
  // whole-node drain contract is the only other path that destroys it.
  prepared_.reset();
  lookahead_.reset();
  permits_.reset();
  if (charge_) runner_.node_.UnchargeHost(charge_);
}

Status StepTransaction::PrepareBatch(
    PromptBatch& batch, std::span<const PendingWork> pending, std::span<const SegmentWork> work,
    [[maybe_unused]] std::span<const std::span<const std::int32_t>> histories,
    std::span<const std::byte> step_bytes, const BatchAhead& ahead, bool* over_budget) {
  if (over_budget) *over_budget = false;
  if (!batch || batch.data_->published || work.size() != pending.size())
    return Error("batch input preflight needs its unpublished actual work");
  auto& data = *batch.data_;
  auto& runner = *data.owner;
  if (!runner.node_.InRequest(runner.stream_)) {
    if (auto opened = runner.node_.BeginRequest(runner.stream_, runner.slots_.execution(),
                                                "complete batch resources");
        !opened)
      return opened;
    data.owns_request = true;
  }
  // All owners/stages back their complete actual future reads before execution.
  // UsedThrough is the canonical state declaration; zeroed capacity may remain
  // on a clean refusal, never unproven writes.
  for (const auto& shape : data.inventory.shapes())
    for (const auto& segment : shape.segments)
      if (auto grown = runner.slots_.ReserveStateThrough(
              segment.slot, std::max(segment.past + segment.rows, segment.read_cells),
              shape.read_align, shape.stage);
          !grown) {
        if (over_budget) *over_budget = runner.slots_.state_refused(segment.slot);
        return grown;
      }
  // Any initial restore/commit prefix is also part of this prepaid batch.
  for (const auto& target : pending)
    if (auto prepared = runner.slots_.PrepareOwed(target.slot); !prepared) return prepared;
  constexpr std::uint64_t kAllocatorMetadata = 64;
  const auto add = [](std::uint64_t& total, std::uint64_t n, std::uint64_t size) {
    std::uint64_t bytes = 0;
    return !__builtin_mul_overflow(n, size, &bytes) &&
           !__builtin_add_overflow(total, bytes, &total) && total <= PTRDIFF_MAX;
  };
  for (std::size_t i = 0; i < data.entries.size(); ++i) {
    if (data.inventory.fallback(i)) continue;
    auto& entry = data.entries[i];
    const auto& shape = data.inventory.shapes()[i];
    const auto& program = *entry.selected().planned->program;
    const auto owners = shape.segments.size();
    std::uint64_t feature_ports = 0, carry_ports = 0, tap_ports = 0;
    std::uint64_t proofs = 0, ranges = 0, largest_ranges = 0, side_bytes = 0, side_inputs = 0,
                  row_plans = 0;
    std::uint64_t retained_rows = 0, scratch_rows = 0;
    for (const auto& input : program.inputs()) {
      if (input.kind == kn::InputKind::kFeatures) ++feature_ports;
      if (input.kind != kn::InputKind::kSideRows) continue;
      ++side_inputs;
      if (input.bytes % sizeof(std::int32_t)) return Error("batch side-row input is not I32");
      if (!add(side_bytes, 1, input.bytes)) return Error("batch side-row storage overflows");
      const auto lookups = input.bytes / sizeof(std::int32_t);
      scratch_rows = std::max(scratch_rows, lookups);
      for (const auto& table : runner.row_tables_)
        if (table->config.weight.stage == shape.stage &&
            table->config.weight.component == input.component) {
          ++row_plans;
          if (!add(retained_rows, lookups, kPleRetainedBytesPerLookup))
            return Error("batch row-plan storage overflows");
        }
    }
    for (const auto& output : program.outputs()) carry_ports += output.features ? 1U : 0U;
    for (std::size_t stage = 1; stage < runner.slots_.stages(); ++stage)
      if (const auto* tap = model::FeatureTap(runner.slots_.layout(stage));
          shape.stage == 0 && tap && tap->contract.cells == model::CellRule::kPerRow)
        ++tap_ports;
    for (const auto& segment : shape.segments) {
      for (std::size_t stage = 0; stage < runner.slots_.stages(); ++stage) {
        if (shape.stage != 0 && stage != shape.stage) continue;
        auto through = std::max(segment.past + segment.rows, segment.read_cells);
        if (stage != shape.stage) through = std::min(through, runner.slots_.context(stage));
        const auto align =
            stage == shape.stage ? shape.read_align : runner.slots_.read_align(stage);
        auto count = model::UsedRangeCount(runner.slots_.layout(stage), through, align);
        if (!count) return Error(count.error());
        ++proofs;
        if (!add(ranges, 1, *count)) return Error("batch used-range count overflows");
        largest_ranges = std::max(largest_ranges, std::uint64_t{*count});
      }
    }
    std::uint64_t feature_copies = 0, tap_copies = 0;
    std::uint64_t feature_bound = 0;
    if (__builtin_mul_overflow(feature_ports, std::uint64_t{2}, &feature_bound) ||
        __builtin_mul_overflow(std::uint64_t{owners}, feature_bound, &feature_copies) ||
        __builtin_mul_overflow(std::uint64_t{owners}, tap_ports, &tap_copies))
      return Error("batch copy descriptor extent overflows");
    std::uint64_t bytes = sizeof(StepTransaction) + sizeof(Frame) + 512;
    if (!add(bytes, owners, sizeof(StepAuthorization) + sizeof(pl::Segment)) ||
        !add(bytes, proofs, sizeof(Frame::Admission) + kAllocatorMetadata) ||
        !add(bytes, ranges, sizeof(LiveState::Range)) ||
        !add(bytes, program.inputs().size(), sizeof(std::vector<std::int32_t>) + sizeof(RunCopy)) ||
        !add(bytes, program.outputs().size(), sizeof(RunCopy)) ||
        !add(bytes, feature_copies + carry_ports + tap_copies, sizeof(RunCopy)) ||
        !add(bytes, row_plans, sizeof(Frame::RowRead) + 3 * kAllocatorMetadata) ||
        !add(bytes, side_inputs, kAllocatorMetadata) || !add(bytes, 1, side_bytes) ||
        !add(bytes, 1, retained_rows) || !add(bytes, 10, kAllocatorMetadata))
      return Error("batch frame/transaction metadata extent overflows");
    auto paid = runner.node_.PrepayHost(bytes);
    if (!paid) {
      if (over_budget) *over_budget = true;
      return Error(paid.error());
    }
    entry.frame_payment = std::move(*paid);
    std::uint64_t temporary = 5 * kAllocatorMetadata;
    if (!add(temporary, owners, sizeof(SegmentWork) + sizeof(pl::Segment) + sizeof(std::size_t)) ||
        !add(temporary, program.inputs().size(), sizeof(std::uint16_t) + sizeof(RunCopy)) ||
        !add(temporary, scratch_rows, kPleScratchBytesPerLookup) ||
        !add(temporary, largest_ranges, sizeof(model::StateRange)))
      return Error("batch preflight scratch metadata overflows");
    auto scratch = runner.node_.PrepayHost(temporary);
    if (!scratch) {
      if (over_budget) *over_budget = true;
      return Error(scratch.error());
    }
    StepRequest request{
        .shape = shape, .step_bytes = step_bytes, .advance = shape.stage == 0, .ahead = {}};
    std::vector<SegmentWork> inputs;
    inputs.reserve(owners);
    for (const auto& segment : shape.segments) {
      const auto found = std::ranges::find(pending, segment.slot, &PendingWork::slot);
      if (found == pending.end()) return Error("batch shape names no original owner");
      const auto at = static_cast<std::size_t>(found - pending.begin());
      if (shape.stage == 0) {
        inputs.push_back(work[at]);
      } else {
        return Error("batch shapes have no non-zero stage");
      }
    }
    request.work = inputs;
    if (!data.private_resources && shape.stage == 0 && ahead) {
      ComposedStep predicted;
      predicted.shape = shape;
      predicted.work.reserve(owners);
      for (const auto& segment : shape.segments) {
        const auto found = std::ranges::find(pending, segment.slot, &PendingWork::slot);
        if (found == pending.end()) return Error("prediction shape names no actual owner");
        predicted.work.push_back(static_cast<std::size_t>(found - pending.begin()));
      }
      entry.ahead = ahead(predicted, predicted.work);
      request.ahead = entry.ahead;
    }
    auto transaction = std::shared_ptr<StepTransaction>(new StepTransaction(runner));
    transaction->permits_capacity_ = owners;
    if (owners) transaction->permits_ = std::make_unique<StepAuthorization[]>(owners);
    transaction->prepared_ = std::make_unique<Frame>(request);
    auto& frame = *transaction->prepared_;
    frame.entry = &entry.selected();
    frame.batch = frame.preflight = true;
    frame.execution = data.private_resources ? StepRuns(false) : runner.programs_.runs();
    if (data.private_resources) frame.execution.SetLaunch(data.views[shape.stage].launch());
    frame.execution.SetStaging(data.pinned ? data.pinned + entry.inputs_first : nullptr,
                               entry.measured.inputs);
    frame.output_base = data.private_resources
                            ? (data.pinned ? data.pinned + data.output_first : nullptr)
                            : runner.outputs_staging_;
    frame.output_bytes = data.private_resources ? data.route_first - data.output_first
                                                : runner.outputs_staging_bytes_;
    frame.feature_capacity = static_cast<std::size_t>(feature_copies);
    frame.carry_capacity = static_cast<std::size_t>(carry_ports);
    frame.tap_capacity = static_cast<std::size_t>(tap_copies);
    frame.admitted.reserve(static_cast<std::size_t>(proofs));
    std::vector<model::StateRange> used(static_cast<std::size_t>(largest_ranges));
    for (const auto& segment : shape.segments)
      for (std::size_t stage = 0; stage < runner.slots_.stages(); ++stage) {
        if (shape.stage != 0 && stage != shape.stage) continue;
        auto through = std::max(segment.past + segment.rows, segment.read_cells);
        if (stage != shape.stage) through = std::min(through, runner.slots_.context(stage));
        const auto align =
            stage == shape.stage ? shape.read_align : runner.slots_.read_align(stage);
        const auto& layout = runner.slots_.layout(stage);
        auto count = model::UsedRangeCount(layout, through, align);
        if (!count) return Error(count.error());
        if (auto filled =
                model::UsedThroughInto(layout, through, align, std::span(used).first(*count));
            !filled)
          return filled;
        Frame::Admission proof{.live = &runner.slots_.live(segment.slot, stage), .ranges = {}};
        proof.ranges.reserve(*count);
        for (const auto& range : std::span(used).first(*count))
          proof.ranges.push_back({.region = 0, .offset = range.offset, .bytes = range.bytes});
        if (auto backed = proof.live->CheckUsed(proof.ranges); !backed) return backed;
        frame.admitted.push_back(std::move(proof));
      }
    if (auto rows = transaction->PlanRows(frame); !rows) return Error(rows.error().what);
    if (auto staged = transaction->Stage(frame); !staged) {
      if (over_budget) *over_budget = staged.error().capacity_refused;
      return Error(staged.error().what);
    }
    if (!data.private_resources) {
      auto layout = runner.programs_.runs().Layout(program.inputs(), 0);
      if (!layout) return Error(layout.error());
      frame.copies = std::move(*layout);
      if (auto predicted = transaction->PrepareLookahead(frame); !predicted)
        return Error(predicted.error().what);
      if (auto fresh = transaction->PrepareFutureState(frame); !fresh)
        return Error(fresh.error().what);
      if (auto drained = transaction->FinishPreparations(); !drained) return drained;
    }
    if (entry.selected().planned->program->routed()) {
      auto route = runner.PrepareBatchRouting(
          entry.selected(), data.views[shape.stage],
          data.private_resources
              ? std::span<std::byte>{data.pinned + data.route_first, data.route_bytes}
              : std::span<std::byte>{},
          frame.execution, entry.selected().runs[0], frame.copies->size(), frame.outputs->size(),
          frame.carry_capacity);
      if (!route) return Error(route.error());
      entry.route = std::move(*route);
    }
    frame.host_inputs = HostInputIdentity(request);
    frame.preflight = false;
    frame.request = nullptr;  // no caller span escapes the synchronous preflight
    entry.transaction = std::move(transaction);
  }
  if (auto current = data.inventory.Revalidate(); !current) return current;
  auto& memory = data.backing->memory;
  const auto count = memory.ExtentCount();
  if (count == 0) return {};
  std::uint64_t bytes = 3 * kAllocatorMetadata +
                        sizeof(decltype(data.empty.bytes_by_domain)::value_type) +
                        3 * sizeof(void*);
  if (!add(bytes, count,
           sizeof(catalog::ExtentId) + sizeof(decltype(data.empty.extents)::value_type)))
    return Error("private batch lease source metadata overflows");
  auto paid = runner.node_.PrepayHost(bytes);
  if (!paid) {
    if (over_budget) *over_budget = true;
    return Error(paid.error());
  }
  std::vector<catalog::ExtentId> extents(count);
  if (auto filled = memory.FillExtents(extents); !filled) return filled;
  catalog::Closure closure;
  auto described = runner.node_.Call(
      [&]() -> Status {
        auto actual = runner.node_.catalog().ClosureOfExtents(extents);
        if (!actual) return Error(catalog::ToString(actual.error()));
        closure = std::move(*actual);
        return {};
      },
      "describing private batch source backing");
  if (!described) return described;
  auto extension = runner.node_.PrepareRequestExtension(runner.stream_, closure, over_budget);
  if (!extension) return Error(extension.error());
  data.extension = std::move(*extension);
  return runner.node_.ApplyRequestExtension(data.extension, over_budget);
}

std::expected<StepReport, StepRefusal> StepTransaction::Run(const StepRequest& request) {
  if (ran_ || (batch_ && (!batch_.data_->Matches(batch_index_, request.shape) ||
                          !PreparedInputsMatch(request))))
    return Refused("a transaction runs once within its live batch boundary");
  ran_ = true;
  const PlanStep protection;
  Frame ordinary(request);
  auto preflight = std::move(prepared_);
  auto& frame = preflight ? *preflight : ordinary;
  frame.request = &request;
  if (!frame.batch) {
    frame.execution = runner_.programs_.runs();
    frame.output_base = runner_.outputs_staging_;
    frame.output_bytes = runner_.outputs_staging_bytes_;
  }
  const auto execute = [&]() -> std::expected<StepReport, StepRefusal> {
    if (auto result = Plan(frame); !result) return std::unexpected(std::move(result.error()));
    if (auto result = Stage(frame); !result) return std::unexpected(std::move(result.error()));
    if (auto result = PrepareLookahead(frame); !result)
      return std::unexpected(std::move(result.error()));
    if (auto result = PrepareState(frame); !result)
      return std::unexpected(std::move(result.error()));
    Queue(frame);
    if (auto result = Settle(frame); !result) return std::unexpected(std::move(result.error()));
    return Publish(frame);
  };
  auto result = execute();
  if (!result) {
    // A prepare phase may have submitted independent work before failing.
    // Its LiveState owner retains an unproven preparation; no frame span is
    // transferred into that owner. Optional builds likewise retire/retain
    // through their existing funded group before this frame is destroyed.
    if (!FinishPreparations()) result.error().clean = false;
    if (lookahead_) lookahead_->Abandon();
  }
  runner_.ForgetRouted(releasable());
  if (batch_) {
    auto& batch = *batch_.data_;
    if (releasable()) {
      base::Check(batch.pending != 0, "a batch step publishes its retirement once");
      --batch.pending;
    } else {
      batch.unproven.Mark();
    }
  }
  return result;
}

std::expected<void, StepRefusal> StepTransaction::PlanRows(Frame& frame) {
  const auto& request = *frame.request;
  const auto& shape = request.shape;
  const auto& program = *frame.entry->planned->program;
  if (std::ranges::none_of(program.inputs(), [](const auto& input) {
        return input.kind == kn::InputKind::kSideRows;
      }))
    return {};
  auto& side_rows = frame.side_rows;
  using RowRead = Frame::RowRead;
  std::vector<RowRead> planned_rows;
  std::size_t tables = 0;
  for (const auto& input : program.inputs())
    if (input.kind == kn::InputKind::kSideRows)
      for (const auto& table : runner_.row_tables_)
        if (table->config.weight.stage == shape.stage &&
            table->config.weight.component == input.component)
          ++tables;
  planned_rows.reserve(tables);
  side_rows.resize(program.inputs().size());
  // One joined slot per component owns its table's landing and row-slot
  // numbering. Independent segmented plans cannot share that storage.
  std::vector<std::uint16_t> components;
  components.reserve(program.inputs().size());
  for (const auto& input : program.inputs()) {
    if (input.kind != kn::InputKind::kSideRows) continue;
    if (std::ranges::find(components, input.component) != components.end())
      return Refused("side-table components require one joined row input");
    components.push_back(input.component);
  }
  for (std::size_t index = 0; index < program.inputs().size(); ++index) {
    const auto& input = program.inputs()[index];
    if (input.kind != kn::InputKind::kSideRows) continue;
    auto& ids = side_rows[index];
    if (!request.side_rows.empty()) {
      if (input.bytes != request.side_rows.size_bytes())
        return Refused("the side rows must fill their slot");
      ids.assign(request.side_rows.begin(), request.side_rows.end());
    } else {
      std::uint64_t rows = 0;
      for (std::size_t i = 0; i < shape.segments.size(); ++i)
        if (input.segment == ir::kJoinedSegment || input.segment == i)
          rows += shape.segments[i].rows;
      if (rows == 0 || input.bytes % (rows * sizeof(std::int32_t)) != 0)
        return Refused("side rows have no whole number of heads");
      const auto heads = input.bytes / (rows * sizeof(std::int32_t));
      if (heads == 0) return Refused("side rows have no heads");
      ids.resize(input.bytes / sizeof(std::int32_t));
      std::size_t at = 0;
      for (std::size_t i = 0; i < shape.segments.size(); ++i) {
        if (input.segment != ir::kJoinedSegment && input.segment != i) continue;
        const auto& segment = shape.segments[i];
        const auto& work = request.work[i];
        const std::uint64_t token_first = std::uint64_t{segment.past};
        if (token_first > work.history.size() || segment.rows > work.history.size() - token_first ||
            work.tokens.size() != segment.rows ||
            !std::ranges::equal(work.tokens, work.history.subspan(token_first, segment.rows)))
          return Refused("side rows require the owner's complete matching token history");
        for (std::uint32_t row = 0; row < segment.rows; ++row) {
          auto derived = runner_.source_->SideRows(shape.stage, input.component, work.history,
                                                   static_cast<std::uint32_t>(token_first + row),
                                                   std::span(ids).subspan(at, heads));
          if (!derived) return Refused(derived.error());
          at += heads;
        }
      }
    }
    if (std::ranges::any_of(ids, [](const auto id) { return id < 0; }))
      return Refused("a side row is negative");
    for (auto& table : runner_.row_tables_) {
      if (table->config.weight.stage != shape.stage ||
          table->config.weight.component != input.component)
        continue;
      if (table->unproven.active())
        return Refused("row-table reads have unproven completion", false);
      auto plan = PlanPleRows(table->table, ids, PleLandingBound(table->capacity), table->capacity);
      if (!plan) return Refused(plan.error());
      // One side-row input is staged from its first part's remap. Every
      // other part must agree in lookup order, not just in its unique IDs.
      for (const auto& prior : planned_rows) {
        if (prior.table->config.weight.stage == shape.stage &&
            prior.table->config.weight.component == input.component &&
            prior.plan.slots != plan->slots)
          return Refused("a side table's parts disagree on row-slot mapping");
      }
      planned_rows.push_back({.table = table.get(), .plan = std::move(*plan)});
    }
  }

  frame.row_reads = std::move(planned_rows);
  return {};
}

std::expected<void, StepRefusal> StepTransaction::Plan(Frame& frame) {
  auto& transaction = *this;
  const auto& request = *frame.request;
  const auto& shape = request.shape;
  if (frame.batch) {
    for (std::size_t i = 0; i < shape.segments.size(); ++i) {
      const auto* permit = authorization().empty() ? nullptr : &authorization()[i];
      if (auto allowed =
              runner_.CheckAdmission(shape.segments[i], shape.stage, request.advance, permit);
          !allowed)
        return allowed;
    }
    for (const auto& proof : frame.admitted)
      if (auto backed = proof.live->CheckUsed(proof.ranges); !backed)
        return Refused(backed.error());
    if (auto places = runner_.slots_.CheckPlaces(); !places) return Refused(places.error());
    return {};
  }
  // Row-table input validation precedes admission: even zeroing newly
  // backed state is work a malformed set of side rows must not cause.
  // Planning binds stable virtual addresses only; PlanStep pins the plan
  // across admission and the eventual job.
  std::expected<ModelRunner::Plans::Entry*, std::string> entry_of = nullptr;
  const bool uses_side_rows = shape.stage < runner_.stage_facts_.size()
                                  ? runner_.stage_facts_[shape.stage].uses_side_rows
                                  : runner_.source_->StageFacts(shape.stage).uses_side_rows;
  if (!runner_.row_tables_.empty() || !request.side_rows.empty() || uses_side_rows) {
    entry_of = runner_.Planned(shape);
    if (!entry_of) return Refused(entry_of.error());
    frame.entry = *entry_of;
    if (auto reads = PlanRows(frame); !reads) return reads;
  }
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
  if (auto admitted = transaction.Admit(request); !admitted)
    return std::unexpected(std::move(admitted.error()));
  if (auto places = runner_.slots_.CheckPlaces(); !places) return Refused(places.error());
  if (*entry_of == nullptr) {
    entry_of = runner_.Planned(shape);
    if (!entry_of) return Refused(entry_of.error());
    frame.entry = *entry_of;
    if (auto reads = PlanRows(frame); !reads) return reads;
  }
  frame.entry = *entry_of;

  return {};
}

std::expected<void, StepRefusal> StepTransaction::Stage(Frame& frame) {
  const auto& request = *frame.request;
  const auto& shape = request.shape;
  auto& entry = *frame.entry;
  auto& program = *entry.planned->program;
  auto& side_rows = frame.side_rows;
  auto& row_reads = frame.row_reads;
  auto& copies = frame.copies;
  auto& outputs = frame.outputs;
  auto& features = frame.features;
  auto& carries = frame.carries;
  auto& taps = frame.taps;
  using RowRead = Frame::RowRead;
  if (frame.batch && !frame.preflight) {
    auto& data = *batch_.data_;
    if (!data.private_resources) {
      const auto& prepared = data.entries[batch_index_];
      auto* original = data.pinned + prepared.inputs_first;
      auto& ordinary = runner_.programs_.runs();
      if (prepared.measured.inputs)
        std::memcpy(ordinary.staging(), original,
                    static_cast<std::size_t>(prepared.measured.inputs));
      // The immutable preflight layout already names the existing global
      // staging used by these cached recordings. No new layout allocation or
      // optional capacity decision occurs between batch steps.
      frame.execution = ordinary;
    }
    features->resize(frame.feature_capacity);
    CopySink feature_sink{std::span(*features)};
    if (auto emitted = runner_.EmitFeatureCopies(program, request, feature_sink); !emitted)
      return Refused(emitted.error());
    features->resize(feature_sink.size());
    carries->resize(frame.carry_capacity);
    CopySink carry_sink{std::span(*carries)};
    if (auto emitted = runner_.EmitCarryCopies(program, shape, carry_sink); !emitted)
      return Refused(emitted.error());
    carries->resize(carry_sink.size());
    taps->resize(frame.tap_capacity);
    CopySink tap_sink{std::span(*taps)};
    if (auto emitted = runner_.EmitTapCarries(shape, tap_sink); !emitted)
      return Refused(emitted.error());
    taps->resize(tap_sink.size());
    if (entry.planned->positioned) {
      std::array<std::uint32_t, Frame::kOwners> past{};
      for (std::size_t i = 0; i != shape.segments.size(); ++i) past[i] = shape.segments[i].past;
      if (auto positioned = program.Position(std::span(past).first(shape.segments.size()));
          !positioned)
        return Refused(positioned.error().detail);
    }
    return {};
  }
  // Exact output payload and segment-view metadata are funded/allocated
  // before queueing. Joining owners shares immutable backing rather than
  // allocating one duplicate of the joined output for every owner.
  std::array<std::uint64_t, Frame::kOwners> segment_bytes{};
  std::uint64_t joined_bytes = 0;
  for (const auto& output : program.outputs()) {
    if (output.segment != ir::kJoinedSegment && output.segment >= shape.segments.size())
      return Refused("an output names no admitted segment");
    auto& count =
        output.segment == ir::kJoinedSegment ? joined_bytes : segment_bytes[output.segment];
    if (__builtin_add_overflow(count, output.bytes, &count))
      return Refused("host output payload inventory overflows");
  }
  bool over_budget = false;
  auto owned =
      HostOutputs::Allocate(runner_.node_, std::span(segment_bytes).first(shape.segments.size()),
                            joined_bytes, &over_budget);
  if (!owned)
    return std::unexpected(StepRefusal{.what = owned.error(), .capacity_refused = over_budget});
  frame.report.outputs = std::move(*owned);
  frame.report.joined = frame.report.outputs.joined();
  // Inputs, staged by slot.
  const auto fill = [&](std::size_t index,
                        std::span<std::byte> out) -> std::expected<void, std::string> {
    const kn::InputSlot& in = program.inputs()[index];
    const bool joined = in.segment == ir::kJoinedSegment;
    const std::size_t segment = in.segment;
    if (!joined && segment >= shape.segments.size()) return Error("an input names no segment");
    components::InputValueSink values(out, sizeof(std::int32_t), true);
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
          for (const auto token : tokens) values.push_back(token);
        }
        return values.Finish();
      case kn::InputKind::kPositions:
        // In `sections` copies, one after another (a multi-section
        // rotation's).
        for (std::uint32_t sec = 0; sec < std::max<std::uint32_t>(in.sections, 1); ++sec) {
          for (std::size_t i = 0; i < shape.segments.size(); ++i) {
            if (!joined && i != segment) continue;
            for (std::uint32_t r = 0; r < shape.segments[i].rows; ++r) {
              values.push_back(static_cast<std::int32_t>(shape.segments[i].past + r));
            }
          }
        }
        return values.Finish();
      case kn::InputKind::kSideRows: {
        const auto ids = std::span(side_rows[index]);
        if (ids.size_bytes() != out.size()) {
          return Error("the side rows must fill their slot");
        }
        for (const RowRead& read : row_reads) {
          if (read.table->config.weight.component == in.component) {
            return WriteRows(out, read.plan.slots);
          }
        }
        return WriteRows(out, ids);
      }
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
      case kn::InputKind::kOutputRows:
      case kn::InputKind::kStoreCells:
      case kn::InputKind::kMask:
      case kn::InputKind::kBlockPositions:
      case kn::InputKind::kVisibleCells:
      case kn::InputKind::kHostComputed:
      case kn::InputKind::kHadamard:
        // A fixed tensor's cell, and a verify's whole step's rows, are
        // every segment's.
        if (in.kind != kn::InputKind::kOutputRows &&
            !(in.kind == kn::InputKind::kHostComputed && in.host != nullptr && in.host->joined) &&
            in.kind != kn::InputKind::kHadamard && joined && !in.cells.fixed && !in.cells.per_row)
          return Error("an input names no segment");
        if (in.commit_save) {
          components::InputValueSink rows(out, sizeof(std::int64_t));
          for (const pl::Segment& s : shape.segments) {
            for (std::uint32_t r = 0; r < s.rows; ++r) {
              rows.push_back(std::int64_t{s.slot} * runner_.config_.max_verify_rows + r);
            }
          }
          return rows.Finish();
        }
        return WriteComputed(in, shape, out);
      case kn::InputKind::kFeatures:
        return Error("a step's features are copied on the device, never staged");
    }
    return Error("an unknown input kind");
  };
  copies = frame.execution.Stage(program.inputs(), fill, 0);
  if (!copies) return Refused(copies.error());
  outputs = runner_.OutputCopies(program, frame.output_base, frame.output_bytes);
  if (!outputs) return Refused(outputs.error());
  // A drafter's features, copied in on the device before its run, and its
  // next features carried to its slots' carries after it.
  if (frame.preflight) {
    features = Copies(frame.feature_capacity);
    carries = Copies(frame.carry_capacity);
    taps = Copies(frame.tap_capacity);
  } else {
    features = runner_.FeatureCopies(program, request);
    if (!features) return Refused(features.error());
  }
  if (!frame.preflight) {
    carries = runner_.CarryCopies(program, shape);
    if (!carries) return Refused(carries.error());
    taps = runner_.TapCarries(shape);
    if (!taps) return Refused(taps.error());
  }
  if (entry.planned->positioned) {
    // Between jobs: no run of this program is in flight.
    std::array<std::uint32_t, Frame::kOwners> past{};
    for (std::size_t i = 0; i < shape.segments.size(); ++i) past[i] = shape.segments[i].past;
    if (auto positioned =
            entry.planned->program->Position(std::span(past).first(shape.segments.size()));
        !positioned) {
      return Refused(std::format("positioning a step: {}", positioned.error().detail));
    }
  }

  return {};
}

std::expected<void, StepRefusal> StepTransaction::PrepareLookahead(Frame& frame) {
  if (frame.batch && !frame.preflight) return {};  // prepared before any numeric child
  const auto& request = *frame.request;
  const auto& shape = request.shape;
  auto& entry = *frame.entry;
  auto& copies = frame.copies;
  auto& capture = frame.capture;
  auto& capture_first = frame.capture_first;
  auto& upcoming = frame.upcoming;
  auto& upcoming_copies = frame.upcoming_copies;
  auto& upcoming_outputs = frame.upcoming_outputs;
  auto& funded = frame.funded;
  auto& meanwhile = frame.meanwhile;
  const StepAhead& ahead = request.ahead;
  const auto shapes = ahead.shapes();
  // Admit actual container capacity before allocating any speculative metadata.
  // Two canonical keys suffice for exact O(n^2) deduplication. The emitter
  // counts words without allocating; a changed count/fill suppresses lookahead.
  const auto prepare = [&]() -> bool {
    if (shapes.empty()) return false;
    constexpr std::uint64_t kAllocatorMetadata = 64;
    std::uint64_t bytes = 0;
    if (__builtin_mul_overflow(std::uint64_t{shapes.size()}, std::uint64_t{sizeof(Lookahead::Slot)},
                               &bytes) ||
        __builtin_add_overflow(bytes, std::uint64_t{sizeof(Lookahead)} + 2 * kAllocatorMetadata,
                               &bytes) ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
      return false;
    auto storage = runner_.node_.PrepayHost(bytes);
    if (!storage) return false;
    auto owned = std::unique_ptr<Lookahead, LookaheadDelete>(new (std::nothrow) Lookahead);
    if (!owned) return false;
    owned->storage_payment = std::move(*storage);
    owned->slots.reset(new (std::nothrow) Lookahead::Slot[shapes.size()]);
    if (!owned->slots) return false;
    owned->count = shapes.size();
    owned->inventory = ahead;
    std::size_t segments = 0;
    for (const auto& candidate : shapes)
      if (candidate) segments = std::max(segments, candidate->segments.size());
    if (segments) {
      if (__builtin_mul_overflow(std::uint64_t{segments}, std::uint64_t{2 * sizeof(pl::Segment)},
                                 &bytes) ||
          __builtin_add_overflow(bytes, 2 * kAllocatorMetadata, &bytes) ||
          bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        return false;
      auto paid = runner_.node_.PrepayHost(bytes);
      if (!paid) return false;
      owned->segment_payment = std::move(*paid);
      owned->key.shape.segments.reserve(segments);
      owned->prior.shape.segments.reserve(segments);
    }
    std::size_t words = 0;
    for (const auto& candidate : shapes) {
      if (!candidate || !runner_.CheckShape(*candidate)) continue;
      runner_.NormalizeKeyShape(*candidate, owned->key.shape);
      auto count = runner_.source_->CountShapeParameters(owned->key.shape);
      if (!count) return false;
      words = std::max(words, *count);
    }
    if (words) {
      if (__builtin_mul_overflow(std::uint64_t{words}, std::uint64_t{2 * sizeof(std::uint64_t)},
                                 &bytes) ||
          __builtin_add_overflow(bytes, 2 * kAllocatorMetadata, &bytes) ||
          bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        return false;
      auto paid = runner_.node_.PrepayHost(bytes);
      if (!paid) return false;
      owned->parameter_payment = std::move(*paid);
      owned->key.parameters.reserve(words);
      owned->prior.parameters.reserve(words);
    }
    for (std::size_t k = 0; k < shapes.size(); ++k) {
      if (!shapes[k] || !runner_.CheckShape(*shapes[k])) continue;
      if (!runner_.FillKey(*shapes[k], owned->key, true)) return false;
      auto& slot = owned->slots[k];
      slot.keyed = true;
      if (owned->key == entry.key) {
        slot.cached = &entry;
        continue;
      }
      bool duplicate = false;
      for (std::size_t prior = 0; prior < k; ++prior) {
        if (!owned->slots[prior].keyed) continue;
        if (!runner_.FillKey(*shapes[prior], owned->prior, true)) return false;
        if (owned->key == owned->prior) {
          slot.cached = owned->slots[prior].cached;
          duplicate = true;
          break;
        }
      }
      if (duplicate) continue;
      slot.distinct = true;
      slot.cached = runner_.plans_.Find(owned->key);
    }
    lookahead_ = std::move(owned);
    return true;
  };
  const bool prepared = prepare();
  if (!prepared && !shapes.empty()) ++runner_.lookahead_.refused;
  auto* slots = prepared ? lookahead_->slots.get() : nullptr;
  StepPlanRuns& runs = entry.runs[0];
  // A recording made ahead holds the layout its plan's inputs predicted;
  // one that differs from what was staged is dropped (its charge settles).
  if (runner_.programs_.runs().graphs() && runs.has_graph() && *copies != runs.copies) {
    runs.DropGraph();
    ++runner_.lookahead_.dropped_ahead;
  }
  const bool repeats = slots && slots[0].keyed && slots[0].cached == &entry;
  const bool first_capture = (runner_.config_.capture_first_phases &
                              runner_.config_.capture_phases & PhaseBit(shape.phase)) != 0 &&
                             runner_.programs_.runs().graphs() && runs.eager_runs == 0 &&
                             !runs.has_graph() && !runs.uncapturable;
  const bool requested_capture = (runs.capture_on_use || (ahead.capture_ahead && repeats)) &&
                                 (runner_.config_.capture_phases & PhaseBit(shape.phase)) != 0 &&
                                 runner_.programs_.runs().graphs() && !runs.has_graph() &&
                                 !runs.uncapturable;
  capture = first_capture ||
            (runs.CaptureDue(runner_.programs_.runs().graphs()) &&
             (runner_.config_.capture_phases & runner_.config_.automatic_capture_phases &
              PhaseBit(shape.phase)) != 0) ||
            requested_capture;
  if (capture && !runner_.plans_.ChargeGraph(entry)) capture = false;
  capture_first = capture && runs.eager_runs == 0;
  // The next shape's recording, beside this run, when it differs and is
  // already planned.
  if (ahead.capture_ahead && slots && shapes[0] && !repeats && runner_.programs_.runs().graphs() &&
      (runner_.config_.capture_phases & PhaseBit(shapes[0]->phase)) != 0 &&
      slots[0].cached != nullptr) {
    upcoming = slots[0].cached;
    const kn::StepProgram& next = *upcoming->planned->program;
    if (next.routed()) {
      // Actual selected experts do not exist yet. Publish an intent only
      // after this step succeeds; next use records under its real flight.
      frame.deferred_recording = upcoming;
      upcoming = nullptr;
    } else {
      auto layout = runner_.programs_.runs().Layout(next.inputs(), 0);
      auto out = runner_.OutputCopies(next);
      if (upcoming->runs[0].has_graph() || upcoming->runs[0].uncapturable ||
          upcoming->runs[0].ahead_refused || !layout || !out ||
          !runner_.plans_.ChargeGraph(*upcoming)) {
        upcoming = nullptr;
      } else {
        upcoming_copies = std::move(*layout);
        upcoming_outputs = std::move(*out);
      }
    }
  }
  // Independent optional grants for the requested actual missing candidates.
  std::size_t requested = 0;
  if (slots && runner_.node_.threaded()) {
    for (std::size_t k = 0; k < shapes.size() && requested < ahead.build; ++k) {
      auto& slot = slots[k];
      if (!slot.distinct || slot.cached) continue;
      ++requested;
      ++runner_.lookahead_.attempted;
      std::uint64_t padded = runner_.plan_floor_bytes_;
      const std::uint64_t scale = std::max<std::uint64_t>(2, shapes[k]->segments.size() + 1);
      if (__builtin_mul_overflow(runner_.plan_floor_bytes_, scale, &padded))
        padded = std::numeric_limits<std::uint64_t>::max();
      slot.future.emplace(runner_.node_, padded);
      if (!slot.future->Fund()) {
        slot.future.emplace(runner_.node_, runner_.plan_floor_bytes_);
        if (!slot.future->Fund()) {
          ++runner_.lookahead_.refused;
          continue;
        }
      }
      slot.build = &*lookahead_->inventory.shapes()[k];
      funded = true;
    }
  }
  meanwhile = funded ? std::function<void()>([this, &frame] {
    for (std::size_t k = 0; k < lookahead_->count; ++k) {
      auto& slot = lookahead_->slots[k];
      if (!slot.build) continue;
      const bool built =
          slot.future->Build([&]() -> std::expected<std::unique_ptr<StepPlanned>, std::string> {
            auto program =
                runner_.source_->Build(*slot.build, runner_.AddressesFor(*slot.build, false));
            if (!program) return std::unexpected(std::move(program.error()));
            if (!*program) return Error("lookahead source returned no program");
            auto planned = std::make_unique<StepPlanned>();
            planned->program = std::move(*program);
            if (PlannedHostBytes(*planned) <= slot.future->allowance())
              planned->program->ConstructionPaid();
            return planned;
          });
      runner_.lookahead_.built += built;
      frame.report.built_ahead += built;
      runner_.lookahead_.build_seconds += slot.future->seconds();
    }
  })
                     : std::function<void()>{};

  return {};
}

std::expected<void, StepRefusal> StepTransaction::PrepareFutureState(Frame& frame) {
  auto& transaction = *this;
  const auto& ahead = frame.request->ahead;
  auto& report = frame.report;
  // The next shape's fresh state, prepared before the job.
  if (ahead.prepare && !ahead.shapes().empty() && ahead.shapes()[0] &&
      runner_.CheckShape(*ahead.shapes()[0])) {
    for (const pl::Segment& s : ahead.shapes()[0]->segments) {
      // Through what that step will back (Admit): its reads, common or not.
      auto prepared =
          runner_.slots_.PrepareThrough(s.slot, std::max(s.past + s.rows, s.read_cells),
                                        ahead.shapes()[0]->read_align, ahead.shapes()[0]->stage);
      if (!prepared) {
        // A drain that cannot prove its page-ins retired quarantined a
        // state: then the refusal is not clean.
        const bool drained = transaction.FinishPreparations().has_value();
        if (!drained || runner_.slots_.faulted() || !runner_.slots_.usable(s.slot))
          return Refused(prepared.error(), false);
        ++runner_.lookahead_.refused;
        break;  // A clean speculative refusal cannot suppress required work.
      }
      report.prepared_ahead = report.prepared_ahead || *prepared;
    }
    runner_.lookahead_.prepared += report.prepared_ahead;
  }

  return {};
}

std::expected<void, StepRefusal> StepTransaction::PrepareState(Frame& frame) {
  auto& transaction = *this;
  const auto& request = *frame.request;
  const auto& shape = request.shape;
  const bool verify = shape.phase == pl::Phase::kVerify;
  auto& entry = *frame.entry;
  auto& program = *entry.planned->program;
  auto& runs = entry.runs[0];
  auto& row_reads = frame.row_reads;
  auto& copies = frame.copies;
  auto& outputs = frame.outputs;
  auto& carries = frame.carries;
  auto& capture = frame.capture;
  auto& saving = frame.saving;
  using RowRead = Frame::RowRead;
  if (!frame.batch) {
    if (auto future = PrepareFutureState(frame); !future) return future;
  }

  // A verify's saves (D-068): what it writes in each stage, by row, into
  // that stage's snapshot, queued in its job before its run.
  if (verify) {
    for (const pl::Segment& s : shape.segments) {
      for (std::size_t stage = 0; stage < runner_.slots_.stages(); ++stage) {
        if (!runner_.Verifies(s.slot, stage)) continue;
        const bool commits =
            runner_.commits_.at((std::size_t{s.slot} * runner_.slots_.stages()) + stage);
        auto writes = model::WritesFor(runner_.slots_.layout(stage), s.past, s.rows,
                                       {.commits = commits, .drafter = stage != 0});
        if (!writes) {
          const bool drained = transaction.FinishPreparations().has_value();
          return Refused(std::format("a verify's saves: {}", writes.error()), drained);
        }
        LiveState& live = runner_.slots_.live(s.slot, stage);
        live.BeginSaves();
        const std::uint64_t base = live.base(0);
        Status saved;
        bool any = false;
        for (std::uint32_t r = 0; r < s.rows && saved; ++r) {
          for (const model::StateRange& range : writes->rows[r]) {
            saved = live.Save(base + range.offset, range.bytes, r);
            if (!saved) break;
            any = true;
          }
        }
        for (const model::StateRange& range : writes->scratch) {
          if (!saved) break;
          saved = live.Save(base + range.offset, range.bytes, -1);
          any = true;
        }
        if (!saved) {
          live.BeginSaves();
          if (!transaction.FinishPreparations()) return Refused(saved.error(), false);
          return Refused(saved.error(), true, s.slot);
        }
        // Every stage it writes or commits, even one that saves nothing
        // (every tensor it writes there truncates): its rows await their
        // Accept, and a failure undoes it whole (by nothing), never
        // quarantines it.
        saving.push_back({.live = &live, .rows = s.rows, .any = any});
      }
    }
  }

  std::shared_ptr<ModelRunner::RoutedFlight> prepared_route;
  if (batch_) prepared_route = std::move(batch_.data_->entries[batch_index_].route);
  if (auto begin = runner_.BeginRouted(program, runs, *copies, *outputs, *carries, capture,
                                       std::move(prepared_route), batch_);
      !begin)
    return Refused(begin.error(), transaction.FinishPreparations().has_value());
  // Plan construction and preparation may call driver-owned lifecycle hooks.
  // Only the current exact ledger units may proceed to recorded writes.
  if (auto current = transaction.Revalidate(request); !current) {
    if (!transaction.FinishPreparations()) current.error().clean = false;
    return std::unexpected(std::move(current.error()));
  }
  // What the step writes, recorded before the job (incremental spill).
  for (const pl::Segment& s : shape.segments) {
    if (auto recorded = runner_.slots_.RecordWrite(s.slot, s.past); !recorded) {
      if (!transaction.FinishPreparations()) return Refused(recorded.error(), false);
      return Refused(recorded.error(), true, s.slot);
    }
  }
  // What of `wrote` is the step's own work (its program's queueing), apart
  // from what its owners owed: owed work runs whole or stays owed, so a
  // failure after it alone leaves their states as consistent as before.
  // A verify's saves queued: work on the stream (the job never "not
  // started"), though they write no state.
  for (const pl::Segment& s : shape.segments) {
    if (auto prepared = runner_.slots_.PrepareOwed(s.slot); !prepared) {
      const bool drained = transaction.FinishPreparations().has_value();
      return Refused(prepared.error(),
                     drained && !runner_.slots_.faulted() && runner_.slots_.usable(s.slot));
    }
  }
  for (RowRead& read : row_reads) {
    ModelRunner::RowTableState& table = *read.table;
    const int fd =
        runner_.config_.weights[table.config.weight.stage].weights->shard_fd(table.shard);
    if (fd < 0)
      return Refused("a row table's shard is not open",
                     transaction.FinishPreparations().has_value());
    if (auto loaded = ReadPleRows(*table.storage, fd, read.plan, table.landing); !loaded) {
      if (table.storage->in_flight() != 0) {
        table.unproven.Mark();
      }
      const bool drained = transaction.FinishPreparations().has_value();
      return Refused(loaded.error(), drained && !table.unproven.active());
    }
    std::ranges::copy(read.plan.sources, table.sources);
    *table.count = static_cast<std::uint32_t>(read.plan.sources.size());
  }
  return {};
}

void StepTransaction::Queue(Frame& frame) {
  auto& transaction = *this;
  const auto& request = *frame.request;
  const auto& shape = request.shape;
  auto& entry = *frame.entry;
  auto& program = *entry.planned->program;
  auto& runs = entry.runs[0];
  auto& row_reads = frame.row_reads;
  auto& copies = frame.copies;
  auto& outputs = frame.outputs;
  auto& features = frame.features;
  auto& carries = frame.carries;
  auto& taps = frame.taps;
  auto& capture = frame.capture;
  auto& upcoming = frame.upcoming;
  auto& upcoming_copies = frame.upcoming_copies;
  auto& upcoming_outputs = frame.upcoming_outputs;
  auto& report = frame.report;
  auto& meanwhile = frame.meanwhile;
  auto& saving = frame.saving;
  auto& wrote = frame.wrote;
  auto& unknown = frame.unknown;
  auto& own_wrote = frame.own_wrote;
  auto& saves_queued = frame.saves_queued;
  auto& copies_queued = frame.copies_queued;
  auto& failure = frame.failure;
  auto& job_report = frame.job_report;
  auto& posted = frame.posted;
  using RowRead = Frame::RowRead;
  using Saving = Frame::Saving;
  transaction.Queued();
  posted = runner_.node_.Job(
      runner_.slots_.execution(),
      [&](providers::NativeStream native) {
        // What the owners' states owe (a verify's restore and commit)
        // before this step's own work.
        for (const pl::Segment& s : shape.segments) {
          auto owed = runner_.slots_.QueueOwed(s.slot, native);
          wrote = wrote || owed.wrote;
          if (!owed.result) {
            if (owed.wrote || owed.result.error().unknown)
              frame.prefix_partial[frame.prefix_partial_count++] = s.slot;
            failure = owed.result.error().detail;
            unknown = owed.result.error().unknown;
            return unknown ? sc::JobResult::kUnknown
                   : wrote ? sc::JobResult::kFailed
                           : sc::JobResult::kNotStarted;
          }
          frame.prefix_complete[frame.prefix_complete_count++] = s.slot;
        }
        for (Saving& sv : saving) {
          // Nothing to save (what it writes truncates, or a commit rebuilds
          // it): undone whole by nothing.
          if (sv.any) {
            if (auto saved = sv.live->QueueSaves(native); !saved) {
              failure = saved.error().detail;
              unknown = saved.error().unknown;
              return unknown                                  ? sc::JobResult::kUnknown
                     : wrote || saves_queued || copies_queued ? sc::JobResult::kFailed
                                                              : sc::JobResult::kNotStarted;
            }
            saves_queued = true;
          }
          sv.queued = true;
        }
        // Outside any recording: their sources move with the positions.
        for (const auto& [to, from, bytes] : *taps) {
          const auto copied =
              providers::CopyAsync(native, support::Pointer(to), support::Pointer(from), bytes,
                                   providers::CopyKind::kDeviceToDevice);
          if (!copied.ok()) {
            failure = std::format("a chunk tap's carry: {}", copied.text());
            unknown = true;
            return sc::JobResult::kUnknown;
          }
          copies_queued = true;
          own_wrote = true;  // a drafter's tap cells: the step's own write
        }
        for (const auto& [to, from, bytes] : *features) {
          if (const providers::DeviceStatus copied =
                  providers::CopyAsync(native, support::Pointer(to), support::Pointer(from), bytes,
                                       providers::CopyKind::kDeviceToDevice);
              !copied.ok()) {
            failure = std::format("a step's features: {}", copied.text());
            unknown = true;
            return sc::JobResult::kUnknown;
          }
          copies_queued = true;
        }
        for (const RowRead& read : row_reads) {
          const ModelRunner::RowTableState& table = *read.table;
          if (!kernels::paging::GatherPleRows(
                  table.landing, table.sources, table.count,
                  static_cast<std::uint32_t>(read.plan.slots.size()),
                  static_cast<std::uint32_t>(table.table.row_bytes),
                  static_cast<std::byte*>(support::Pointer(table.slots.base)), native.handle)) {
            failure = "a row table's gather failed";
            unknown = true;
            return sc::JobResult::kUnknown;
          }
          saves_queued = true;
        }
        const StepQueued q = program.routed()
                                 ? runner_.QueueRoutedPrefix(native)
                                 : frame.execution.Queue(runs, *copies, {}, program, *outputs,
                                                         capture, runner_.graph_stats_, native);
        // `before` includes input staging. A rejected program promises no
        // program work; copying its inputs cannot quarantine numeric state.
        copies_queued = copies_queued || (q.before && !copies->empty());
        own_wrote = own_wrote || q.result.has_value() ||
                    (!q.result && q.result.error().error == kn::StepError::kUnknown);
        wrote = wrote || own_wrote;
        report.path = q.path;
        if (!q.result) {
          failure = q.result.error().detail;
          unknown = q.result.error().error == kn::StepError::kUnknown;
          frame.own_program_unknown = q.program_unknown;
          return unknown                                  ? sc::JobResult::kUnknown
                 : wrote || saves_queued || copies_queued ? sc::JobResult::kFailed
                                                          : sc::JobResult::kNotStarted;
        }
        if (!program.routed())
          for (const auto& [to, from, bytes] : *carries) {
            if (const providers::DeviceStatus copied =
                    providers::CopyAsync(native, support::Pointer(to), support::Pointer(from),
                                         bytes, providers::CopyKind::kDeviceToDevice);
                !copied.ok()) {
              failure = std::format("a step's carried features: {}", copied.text());
              unknown = true;
              return sc::JobResult::kUnknown;
            }
          }
        if (upcoming != nullptr) {
          auto captured = runner_.programs_.runs().CaptureAhead(
              upcoming->runs[0], upcoming_copies, *upcoming->planned->program, upcoming_outputs,
              runner_.graph_stats_, native);
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
      "a model step", runner_.stream_, meanwhile, &job_report);
  // Later routed jobs have their own effects/retirement. Their failure cannot
  // erase positive completion of this first job's ordered prefix.
  frame.prefix_retirement = job_report.retirement;
  if (program.routed() && posted && failure.empty() && !frame.execution.launch()->faulted()) {
    if (auto continued = runner_.ContinueRouted(report.path, runner_.unproven_.step());
        !continued) {
      failure = continued.error();
      unknown = true;  // prefix effects exist; no completion inferred from refusal
      posted = Error(continued.error());
      job_report.retirement = JobReport::Retirement::kUnproven;
    }
  }
}

std::expected<void, StepRefusal> StepTransaction::Settle(Frame& frame) {
  auto& transaction = *this;
  const auto& request = *frame.request;
  const auto& shape = request.shape;
  auto& saving = frame.saving;
  auto& unknown = frame.unknown;
  auto& own_wrote = frame.own_wrote;
  auto& failure = frame.failure;
  auto& job_report = frame.job_report;
  auto& posted = frame.posted;
  auto& preparation = frame.preparation;
  using Saving = Frame::Saving;
  // The job retired: every optional preparation drained before anything
  // else touches the state.
  preparation = transaction.FinishPreparations();
  transaction.JobRetired(job_report);
  // What a drafter step carried, by its slots: nothing until it ran.
  for (const pl::Segment& seg : shape.segments) {
    runner_.carry_rows_.at((std::size_t{seg.slot} * runner_.slots_.stages()) + shape.stage) = 0;
  }
  const bool prefix_retired = frame.prefix_retirement == JobReport::Retirement::kRetired;
  const bool unproven = job_report.retirement == JobReport::Retirement::kUnproven;
  const auto complete = std::span(frame.prefix_complete).first(frame.prefix_complete_count);
  const auto partial = std::span(frame.prefix_partial).first(frame.prefix_partial_count);
  // A failed own program does not undo a fully queued predecessor prefix.
  // Positive retirement, rather than the job's success Status, proves its data
  // and controls completed. One atomic preflight spans every completed owner.
  Status prefixes = runner_.slots_.CompleteOwed(
      complete, prefix_retired,
      prefix_retired ? LiveState::CopyRetirement::kProven : LiveState::CopyRetirement::kUnproven);
  if (!partial.empty()) {
    // A partial prefix has no publication proof. Its immutable pinned words
    // stay retained even if the job's later fence itself positively retired.
    (void)runner_.slots_.CompleteOwed(partial, false, LiveState::CopyRetirement::kUnproven);
    runner_.slots_.QuarantineOwners(partial);
  }
  if (!prefixes || !partial.empty()) {
    if (failure.empty()) failure = !prefixes ? prefixes.error() : "an owed prefix was only partial";
  }
  unknown = unknown || unproven;
  if (!posted || !failure.empty() || frame.execution.launch()->faulted()) {
    runner_.slots_.CheckFailedJob();
    // A program can report uncertain numeric effects even though the fence
    // positively retired. Only that program's declared writes are uncertain
    // when every earlier publication completed. Lost retirement or an unknown
    // copy, recording, control transition or cohort remains broader.
    const bool isolated_unknown =
        frame.own_program_unknown && job_report.retirement == JobReport::Retirement::kRetired &&
        prefix_retired && prefixes.has_value() && partial.empty() && preparation.has_value() &&
        !runner_.slots_.faulted() && !frame.execution.launch()->faulted();
    // What the step's own work may write: its stage, and on the target the
    // drafters' taps (Admit's rule). Another stage changed only by what it
    // owed, which ran whole or stays owed (a partial queueing is unknown).
    const auto written = [&](std::size_t stage) {
      return stage == shape.stage ||
             (shape.stage == 0 &&
              std::ranges::any_of(runner_.slots_.layout(stage).items,
                                  [](const auto& item) { return item.contract.target_writes; }));
    };
    for (const pl::Segment& s : shape.segments) {
      for (std::size_t stage = 0; stage < runner_.slots_.stages(); ++stage) {
        LiveState& live = runner_.slots_.live(s.slot, stage);
        // A verify whose saves were queued is undone whole; anything else
        // the step's own work may have written quarantines (not what its
        // owners owed, which ran whole: counting it would quarantine a
        // state the failure left consistent).
        const auto sv = std::ranges::find(saving, &live, &Saving::live);
        const bool stage_unknown = unknown && (!isolated_unknown || written(stage));
        (void)live.Settle(sv != saving.end() && sv->queued, own_wrote && written(stage),
                          stage_unknown);
      }
    }
    if (frame.execution.launch()->faulted()) runner_.slots_.Fault();
    if (lookahead_) lookahead_->Abandon();
    // A faulted cohort (the launch's fault, or one CheckFailedJob found:
    // a scheduler fault, a lost lease) quarantined every owner, written or
    // not: never clean.
    return Refused(!failure.empty() ? failure
                   : !posted        ? posted.error()
                                    : "the launch faulted",
                   !own_wrote && !unknown && prefixes.has_value() && partial.empty() &&
                       !runner_.slots_.faulted() && preparation.has_value());
  }
  // The verify ran: its rows await their Accept in every stage it writes or
  // that commits.
  for (const Saving& sv : saving) sv.live->Verified(sv.rows);
  if (!preparation) {
    // This step's writes completed but no position is published: every
    // owner becomes unusable rather than expose state at an old cursor.
    for (const pl::Segment& s : shape.segments) {
      for (std::size_t stage = 0; stage < runner_.slots_.stages(); ++stage) {
        runner_.slots_.live(s.slot, stage).Quarantine();
      }
    }
    if (lookahead_) lookahead_->Abandon();
    return Refused(preparation.error(), false);
  }
  return {};
}

std::expected<StepReport, StepRefusal> StepTransaction::Publish(Frame& frame) {
  const auto& request = *frame.request;
  const auto& shape = request.shape;
  const bool verify = shape.phase == pl::Phase::kVerify;
  auto& entry = *frame.entry;
  auto& program = *entry.planned->program;
  auto& outputs = frame.outputs;
  auto& capture_first = frame.capture_first;
  auto& report = frame.report;
  using TapRows = ModelRunner::TapRows;
  // A deferred request is consumed only after actual routed use either made
  // its recording or proved it uncapturable. A graph-payment refusal leaves
  // the intent for a later use and never counts as a capture ahead.
  if (program.routed() && (entry.runs[0].has_graph() || entry.runs[0].uncapturable))
    entry.runs[0].capture_on_use = false;
  if (frame.deferred_recording) frame.deferred_recording->runs[0].capture_on_use = true;
  if (lookahead_) {
    for (std::size_t i = 0; i < lookahead_->count; ++i) {
      auto& slot = lookahead_->slots[i];
      if (!slot.build) continue;
      const bool cached = slot.future->InstallAfterCompletion(
          [&](std::unique_ptr<StepPlanned> planned, double seconds, const auto& transfer) {
            auto installed = runner_.Install(*slot.build, std::move(planned), seconds, transfer);
            if (installed && i == 0 && request.ahead.capture_ahead &&
                (*installed)->planned->program->routed())
              (*installed)->runs[0].capture_on_use = true;
            return installed.has_value();
          });
      runner_.lookahead_.cached += cached;
      report.cached_ahead += cached;
    }
  }
  for (const kn::OutputSlot& out : program.outputs()) {
    if (out.features && out.segment < shape.segments.size()) {
      const pl::Segment& seg = shape.segments[out.segment];
      runner_.carry_rows_.at((std::size_t{seg.slot} * runner_.slots_.stages()) + shape.stage) =
          seg.outputs;
    }
  }
  runner_.lookahead_.captured_ahead += report.captured_ahead;
  if (shape.stage == 0) {
    for (const pl::Segment& seg : shape.segments) {
      for (std::size_t stage = 1; stage < runner_.slots_.stages(); ++stage) {
        const auto* tap = model::FeatureTap(runner_.slots_.layout(stage));
        if (tap == nullptr || tap->contract.cells != model::CellRule::kPerRow) continue;
        const std::size_t at = (std::size_t{seg.slot} * runner_.slots_.stages()) + stage;
        const TapRows held = runner_.HeldTapRows(seg.slot, stage);
        runner_.previous_tap_rows_[at] = verify ? held : TapRows{};
        runner_.tap_rows_[at] = {.first = seg.past,
                                 .count = seg.rows,
                                 .preceding = std::uint64_t{held.first} + held.count == seg.past
                                                  ? std::min(held.count, tap->contract.row_carry)
                                                  : 0};
      }
    }
  }
  report.captured_first = capture_first && report.path == RunPath::kCaptured;
  runner_.lookahead_.captured_first += report.captured_first;
  Count(runner_.graph_stats_, report.path);

  std::array<std::uint32_t, Frame::kOwners> publication_owners{};
  std::array<std::pair<std::uint32_t, std::uint32_t>, Frame::kOwners> positions{};
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const auto& segment = shape.segments[i];
    publication_owners[i] = segment.slot;
    positions[i] = {segment.slot, segment.past + segment.rows};
  }
  const auto owners = std::span(publication_owners).first(shape.segments.size());
  const auto refuse_publication = [&](std::string what) -> std::unexpected<StepRefusal> {
    runner_.slots_.QuarantineOwners(owners);
    return Refused(std::move(what), false);
  };
  if (request.advance)
    for (const auto& segment : shape.segments)
      if (auto checked = runner_.slots_.CheckPositions(segment.slot, segment.past + segment.rows);
          !checked)
        return refuse_publication(checked.error());
  // Copy only into the prepaid exact views; no allocation or capacity growth
  // occurs after completion or after any owner publication.
  std::array<std::size_t, Frame::kOwners> copied{};
  std::size_t joined_copied = 0;
  for (std::size_t i = 0; i < program.outputs().size(); ++i) {
    const auto& slot = program.outputs()[i];
    if (slot.bytes == 0) continue;
    const auto* from = support::Pointer((*outputs)[i][0]);
    const bool joined = slot.segment == ir::kJoinedSegment;
    auto buffer = joined ? report.joined : report.outputs.at(slot.segment);
    auto& offset = joined ? joined_copied : copied[slot.segment];
    auto to = buffer.Writable();
    if (offset > to.size() || slot.bytes > to.size() - offset)
      return refuse_publication("completed host output exceeds its prepaid view");
    std::memcpy(to.data() + offset, from, slot.bytes);
    offset += static_cast<std::size_t>(slot.bytes);
  }
  if (request.advance) {
    if (auto published = runner_.slots_.SetPositions(std::span(positions).first(owners.size()));
        !published)
      return refuse_publication(published.error());
  }
  return std::move(report);
}

}  // namespace llmp::engine
