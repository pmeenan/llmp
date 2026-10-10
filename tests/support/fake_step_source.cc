// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "fake_step_source.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "providers/fake/fake_device_execution.h"

namespace llmp::test_support {

namespace {

namespace kn = llmp::kernels;
namespace pl = llmp::planner;

std::uint64_t Round(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

template <typename T>
T* At(std::uint64_t address) {
  return reinterpret_cast<T*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::unexpected<std::string> Refuse(std::string what) { return std::unexpected(std::move(what)); }

// A tensor of cells as the fake addresses it: where position p's cell is.
struct CellTensor {
  std::uint64_t address = 0, cells = 0, cell_bytes = 0;
  enum class Rule : std::uint8_t { kPosition, kModulo, kRow } rule = Rule::kPosition;
  std::uint64_t carry = 0;  // kRow: the cells before a chunk's rows
  // The cell of `position` (chunk row `row`), or none.
  std::optional<std::uint64_t> Cell(std::int64_t position, std::uint32_t row) const {
    if (address == 0 || position < 0) return std::nullopt;
    const auto p = static_cast<std::uint64_t>(position);
    switch (rule) {
      case Rule::kPosition:
        return p < cells ? std::optional(p) : std::nullopt;
      case Rule::kModulo:
        return p % cells;
      case Rule::kRow:
        return carry + row < cells ? std::optional(carry + row) : std::nullopt;
    }
    return std::nullopt;
  }
  void* At(std::uint64_t cell) const;
};

CellTensor TensorOf(std::uint64_t address, const model::StateLayoutDerived::Item& item) {
  CellTensor t{.address = address, .cells = item.cells, .cell_bytes = item.contract.cell_bytes};
  switch (item.contract.cells) {
    case model::CellRule::kRing:
    case model::CellRule::kPositionRing:
      t.rule = CellTensor::Rule::kModulo;
      break;
    case model::CellRule::kPerRow:
      t.rule = CellTensor::Rule::kRow;
      t.carry = item.contract.window_or_ratio;
      break;
    default:
      break;
  }
  return t;
}

// What one segment's run needs, resolved at Build.
struct SegmentPlan {
  std::int32_t slot = 0;
  std::uint32_t rows = 0, outputs = 0, read_cells = 0, first = 0;  // first: row in a joined input
  std::uint64_t tokens = 0, positions = 0, out = 0;                // device addresses
  CellTensor own;                                                  // tensor 0
  std::uint64_t fixed = 0;                                         // the fixed hash's
  bool fixed_writes = true;
  std::vector<CellTensor> taps;  // the drafters' tensors the target writes
  CellTensor tap;                // a drafter's tap it reads
  CellTensor borrowed;           // a drafter's view of the target's tensor
  bool greedy = false;
  std::int64_t past = -1;    // Position's, for a positioned program (-1: the input's)
  std::uint32_t logits = 0;  // F32 logits a row (0: FakeRows)
  bool target = true;        // the target's stage: its logits mark lost state
};

void* CellTensor::At(std::uint64_t cell) const {
  return reinterpret_cast<void*>(address +
                                 (cell * cell_bytes));  // NOLINT(performance-no-int-to-ptr)
}

// A valid cell of `slot` for `position`.
std::optional<FakeCell> Valid(const CellTensor& t, std::int32_t slot, std::int64_t position,
                              std::uint32_t row) {
  const auto cell = t.Cell(position, row);
  if (!cell) return std::nullopt;
  FakeCell c;
  std::memcpy(&c, t.At(*cell), sizeof(c));
  if (c.magic != kFakeMagic || c.slot != slot || c.position != position) return std::nullopt;
  return c;
}

class FakeProgram final : public kn::StepProgram {
 public:
  FakeProgram(std::vector<SegmentPlan> segments, std::vector<kn::InputSlot> inputs,
              std::vector<kn::OutputSlot> outputs, std::vector<kn::TensorUse> uses,
              std::uint64_t scratch, std::uint64_t host, std::uint64_t launches, bool measuring,
              const FakeSourceOptions* options, std::shared_ptr<std::atomic<std::uint64_t>> queued,
              std::shared_ptr<std::atomic<std::uint64_t>> executed,
              std::shared_ptr<std::atomic<std::uint64_t>> positioned, pl::Phase phase)
      : segments_(std::move(segments)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        uses_(std::move(uses)),
        scratch_(scratch),
        host_(host),
        launches_(launches),
        measuring_(measuring),
        options_(options),
        queued_(std::move(queued)),
        executed_(std::move(executed)),
        positioned_(std::move(positioned)),
        phase_(phase) {}

  std::uint64_t scratch_bytes() const override { return scratch_; }
  std::uint64_t host_bytes() const override { return host_; }
  std::uint64_t launches() const override { return launches_; }
  std::span<const kn::InputSlot> inputs() const override { return inputs_; }
  std::span<const kn::OutputSlot> outputs() const override { return outputs_; }
  std::span<const kn::TensorUse> tensors() const override { return uses_; }
  bool capturable() const override { return !options_->positioned; }

  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch& /*launch*/) override {
    if (measuring_) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kRejected, .detail = "a measurement binds not"});
    }
    if (options_->refuse_bind) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kRejected, .detail = "the fake refuses to bind"});
    }
    bound_ = true;
    return {};
  }

  std::expected<void, kn::StepFailure> Position(std::span<const std::uint32_t> past) override {
    if (past.size() != segments_.size()) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kRejected, .detail = "positions per segment"});
    }
    for (std::size_t i = 0; i < past.size(); ++i) segments_[i].past = past[i];
    positioned_->fetch_add(1);
    return {};
  }

  std::expected<void, kn::StepFailure> Run(kn::StepLaunch& /*launch*/,
                                           providers::NativeStream stream) const override {
    if (!bound_) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kRejected, .detail = "an unbound program"});
    }
    const bool fails = !options_->fail_phase || *options_->fail_phase == phase_;
    if (fails && options_->fail_run == kn::StepError::kRejected) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kRejected, .detail = "the fake refuses to run"});
    }
    queued_->fetch_add(1);
    auto launched = providers::fake::FakeDeviceExecution::Launch(
        stream, [segments = segments_, executed = executed_] {
          Execute(segments);
          executed->fetch_add(1);
        });
    if (!launched) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kUnknown, .detail = "the fake refused work"});
    }
    if (fails && options_->fail_run == kn::StepError::kUnknown) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kUnknown, .detail = "the fake faulted"});
    }
    return {};
  }

  std::expected<std::unique_ptr<kn::Recorded>, kn::StepFailure> Record(
      kn::StepLaunch& launch, providers::NativeStream stream,
      const kn::StepLaunch::Body& body) const override {
    if (options_->refuse_record) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kRejected, .detail = "the fake refuses"});
    }
    return launch.Capture(stream, body);
  }

 private:
  // The device's work: on the fake device's thread, at run time.
  static void Execute(const std::vector<SegmentPlan>& segments) {
    for (const SegmentPlan& s : segments) {
      const auto* tokens = At<const std::int32_t>(s.tokens) + s.first;
      const auto* positions = At<const std::int32_t>(s.positions) + s.first;
      const auto position_of = [&](std::uint32_t r) {
        return s.past >= 0 ? static_cast<std::int32_t>(s.past + r) : positions[r];
      };
      // What the drafter's view of the target holds before its first row.
      std::int32_t borrowed = 0;
      if (s.borrowed.address != 0) {
        for (std::int32_t c = 0; c < position_of(0); ++c) {
          borrowed += Valid(s.borrowed, s.slot, c, 0).has_value() ? 1 : 0;
        }
      }
      std::uint64_t hash = 0;
      if (s.fixed != 0) std::memcpy(&hash, At<const void>(s.fixed), sizeof(hash));
      std::vector<std::uint64_t> hashes(s.rows);
      std::vector<std::int32_t> taps(s.rows, -1);
      for (std::uint32_t r = 0; r < s.rows; ++r) {
        const std::int32_t at = position_of(r);
        // The tap the target wrote for the position before (a drafter
        // reads the target's features there), before this row writes.
        if (const auto tapped = Valid(s.tap, s.slot, at - 1, r)) taps[r] = tapped->token;
        const FakeCell cell{
            .magic = kFakeMagic, .slot = s.slot, .position = at, .token = tokens[r]};
        if (const auto own = s.own.Cell(at, r)) std::memcpy(s.own.At(*own), &cell, sizeof(cell));
        for (const CellTensor& t : s.taps) {
          if (const auto c = t.Cell(at, r)) std::memcpy(t.At(*c), &cell, sizeof(cell));
        }
        hash = FakeHash(hash, tokens[r]);
        hashes[r] = hash;
      }
      if (s.fixed != 0 && s.fixed_writes) std::memcpy(At<void>(s.fixed), &hash, sizeof(hash));
      std::int32_t valid = 0;
      for (std::uint32_t c = 0; c < s.read_cells && c < s.own.cells; ++c) {
        FakeCell cell;
        std::memcpy(&cell, s.own.At(c), sizeof(cell));
        valid += cell.magic == kFakeMagic && cell.slot == s.slot && cell.position >= 0 &&
                 s.own.Cell(cell.position, 0) == c;
      }
      // A head's choice for row r: lost state's mark on a target whose
      // tensor 0 lacks a cell through the row's position.
      const auto choice = [&](std::uint32_t r) {
        const std::int32_t at = position_of(r);
        if (s.target && valid < at + 1) return static_cast<std::int32_t>(s.logits - 1);
        return FakeLogit(tokens[r], at, s.logits);
      };
      if (s.greedy) {
        // Its own output: its last row's token (the lowering's form).
        const std::int32_t chosen = s.logits != 0 ? choice(s.rows - 1) : tokens[s.rows - 1] + 1000;
        if (s.outputs != 0) std::memcpy(At<void>(s.out), &chosen, sizeof(chosen));
        continue;
      }
      for (std::uint32_t o = 0; o < s.outputs; ++o) {
        const std::uint32_t r = s.rows - s.outputs + o;
        if (s.logits != 0) {
          std::vector<float> row(s.logits, 0.0F);
          row[static_cast<std::size_t>(choice(r))] = 1.0F;
          std::memcpy(At<void>(s.out + (std::uint64_t{o} * s.logits * sizeof(float))), row.data(),
                      row.size() * sizeof(float));
          continue;
        }
        const FakeRow row{.slot = s.slot,
                          .position = position_of(r),
                          .token = tokens[r],
                          .cells = valid,
                          .hash = hashes[r],
                          .tap = taps[r],
                          .borrowed = borrowed};
        std::memcpy(At<void>(s.out + (o * sizeof(FakeRow))), &row, sizeof(row));
      }
    }
  }

  std::vector<SegmentPlan> segments_;
  std::vector<kn::InputSlot> inputs_;
  std::vector<kn::OutputSlot> outputs_;
  std::vector<kn::TensorUse> uses_;
  std::uint64_t scratch_ = 0, host_ = 0, launches_ = 0;
  bool measuring_ = false, bound_ = false;
  const FakeSourceOptions* options_;
  std::shared_ptr<std::atomic<std::uint64_t>> queued_, executed_, positioned_;
  pl::Phase phase_;
};

}  // namespace

pl::PlanKey FakeStepSource::KeyBase() const {
  pl::PlanKey key;
  key.spec = options_.spec;
  key.patterns[0] = 0xFA;
  return key;
}

std::expected<std::unique_ptr<kn::StepProgram>, std::string> FakeStepSource::Build(
    const pl::StepShape& shape, const kn::Addresses& addresses) {
  built_.push_back({.shape = shape, .measuring = addresses.measuring()});
  if (options_.on_build) options_.on_build(shape, addresses.measuring());
  if (options_.refuse_build && options_.refuse_build(shape)) return Refuse("the fake refuses");
  if (shape.stage >= options_.layouts.size()) return Refuse("no such stage");
  if (options_.extra_state) {
    for (const ir::StateRef& ref : options_.extra_state(shape)) {
      if (addresses.state(ref) == 0) return Refuse("a stray state tensor has no address");
    }
  }
  const model::StateLayoutDerived& layout = options_.layouts[shape.stage];
  const model::StateLayoutDerived::Item* kv = nullptr;
  const model::StateLayoutDerived::Item* fixed = nullptr;
  const model::StateLayoutDerived::Item* tap = nullptr;
  for (const auto& item : layout.items) {
    if (item.layer == 0 && item.tensor == 0) kv = &item;
    if (item.layer == 0 && item.tensor == 1 && item.contract.cells == model::CellRule::kFixed) {
      fixed = &item;
    }
    if (tap == nullptr && shape.stage != 0 && item.contract.target_writes) tap = &item;
  }
  if (kv == nullptr || kv->contract.cell_bytes < sizeof(FakeCell)) {
    return Refuse("the stage has no tensor 0 of a cell's bytes");
  }
  const bool verify = shape.phase == pl::Phase::kVerify;
  // The workspace: inputs, then outputs, at 256-byte boundaries.
  std::uint64_t at = addresses.activations;
  const auto place = [&](std::uint64_t bytes) {
    const std::uint64_t here = at;
    at += Round(std::max<std::uint64_t>(bytes, 1));
    return here;
  };
  std::vector<SegmentPlan> segments;
  std::vector<kn::InputSlot> inputs;
  std::vector<kn::OutputSlot> outputs;
  std::vector<kn::TensorUse> uses;
  const std::uint32_t total = shape.rows();
  std::uint64_t joined_tokens = 0, joined_positions = 0;
  if (options_.joined_inputs) {
    joined_tokens = place(std::uint64_t{total} * 4);
    joined_positions = place(std::uint64_t{total} * 4);
    inputs.push_back({.device_address = joined_tokens,
                      .bytes = std::uint64_t{total} * 4,
                      .tensor = {},
                      .segment = ir::kJoinedSegment,
                      .kind = kn::InputKind::kTokens});
    inputs.push_back({.device_address = joined_positions,
                      .bytes = std::uint64_t{total} * 4,
                      .tensor = {},
                      .segment = ir::kJoinedSegment,
                      .kind = kn::InputKind::kPositions});
  }
  std::uint32_t first = 0;
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const pl::Segment& seg = shape.segments[i];
    SegmentPlan s;
    s.slot = seg.slot;
    s.rows = seg.rows;
    s.outputs = seg.outputs;
    s.read_cells = seg.read_cells;
    s.greedy = seg.greedy;
    s.logits = options_.logits;
    s.target = shape.stage == 0;
    const auto segment = static_cast<std::uint8_t>(i);
    if (options_.joined_inputs) {
      s.tokens = joined_tokens;
      s.positions = joined_positions;
      s.first = first;
    } else {
      s.tokens = place(std::uint64_t{seg.rows} * 4);
      s.positions = place(std::uint64_t{seg.rows} * 4);
      inputs.push_back({.device_address = s.tokens,
                        .bytes = std::uint64_t{seg.rows} * 4,
                        .tensor = {},
                        .segment = segment,
                        .kind = kn::InputKind::kTokens});
      inputs.push_back({.device_address = s.positions,
                        .bytes = std::uint64_t{seg.rows} * 4,
                        .tensor = {},
                        .segment = segment,
                        .kind = kn::InputKind::kPositions});
    }
    first += seg.rows;
    const std::uint64_t base =
        addresses.state({.stage = shape.stage, .tensor = 0, .layer = 0, .segment = segment});
    if (base == 0) return Refuse("a state tensor has no address");
    s.own = TensorOf(base, *kv);
    uses.push_back(
        {.address = base,
         .bytes = std::min<std::uint64_t>(seg.read_cells, kv->cells) * kv->contract.cell_bytes,
         .role = ir::Role::kState});
    if (fixed != nullptr) {
      s.fixed =
          addresses.state({.stage = shape.stage, .tensor = 1, .layer = 0, .segment = segment});
      if (s.fixed == 0) return Refuse("the fixed state has no address");
      // A verify only reads state its commit rebuilds.
      s.fixed_writes = !(verify && fixed->contract.rollback == model::Rollback::kCommit);
      uses.push_back({.address = s.fixed, .bytes = fixed->bytes, .role = ir::Role::kState});
    }
    if (shape.stage == 0) {
      // The target writes every drafter's taps.
      for (std::size_t t = 1; t < options_.layouts.size(); ++t) {
        for (const auto& item : options_.layouts[t].items) {
          if (!item.contract.target_writes || item.contract.cell_bytes < sizeof(FakeCell)) continue;
          const std::uint64_t where =
              addresses.state({.stage = static_cast<std::uint16_t>(t),
                               .tensor = static_cast<std::uint16_t>(item.tensor),
                               .layer = item.layer,
                               .segment = segment});
          if (where == 0) return Refuse("a drafter's tap has no address");
          s.taps.push_back(TensorOf(where, item));
          uses.push_back({.address = where, .bytes = item.bytes, .role = ir::Role::kState});
        }
      }
    } else {
      if (tap != nullptr && tap->contract.cell_bytes >= sizeof(FakeCell)) {
        const std::uint64_t where =
            addresses.state({.stage = shape.stage,
                             .tensor = static_cast<std::uint16_t>(tap->tensor),
                             .layer = tap->layer,
                             .segment = segment});
        if (where == 0) return Refuse("the tap has no address");
        s.tap = TensorOf(where, *tap);
      }
      if (!layout.borrowed.empty() && !options_.layouts.empty()) {
        const auto& b = layout.borrowed.front();
        const auto* target = model::BorrowedItem(options_.layouts.front(), b);
        const std::uint64_t where = addresses.state({.stage = shape.stage,
                                                     .tensor = static_cast<std::uint16_t>(b.tensor),
                                                     .layer = b.layer,
                                                     .segment = segment});
        if (target == nullptr || where == 0) return Refuse("the borrowed tensor has no address");
        s.borrowed = TensorOf(where, *target);
        uses.push_back({.address = where, .bytes = target->bytes, .role = ir::Role::kState});
      }
    }
    if (seg.outputs != 0) {
      // A greedy segment's output is its one token.
      const std::uint64_t bytes =
          seg.greedy ? sizeof(std::int32_t)
                     : std::uint64_t{seg.outputs} *
                           (options_.logits != 0 ? std::uint64_t{options_.logits} * sizeof(float)
                                                 : sizeof(FakeRow));
      s.out = place(bytes);
      outputs.push_back(
          {.device_address = s.out, .bytes = bytes, .tensor = {}, .segment = segment});
    }
    segments.push_back(s);
  }
  for (const kn::InputSlot& in : inputs) {
    uses.push_back({.address = in.device_address, .bytes = in.bytes, .role = ir::Role::kInput});
  }
  for (const kn::OutputSlot& out : outputs) {
    uses.push_back({.address = out.device_address, .bytes = out.bytes, .role = ir::Role::kOutput});
  }
  if (options_.stray_output && !outputs.empty()) {
    outputs.front().segment = static_cast<std::uint8_t>(shape.segments.size());
  }
  if (options_.stray_use) {
    uses.push_back({.address = 4096, .bytes = 64, .role = ir::Role::kWeight});
  }
  const std::uint64_t scratch = at - addresses.activations;
  if (!addresses.measuring() && scratch > addresses.activation_bytes) {
    return Refuse(std::format("{} bytes of activations exceed the workspace's {}", scratch,
                              addresses.activation_bytes));
  }
  return std::make_unique<FakeProgram>(std::move(segments), std::move(inputs), std::move(outputs),
                                       std::move(uses), scratch, options_.host_bytes,
                                       options_.launches, addresses.measuring(), &options_, queued_,
                                       executed_, positioned_, shape.phase);
}

}  // namespace llmp::test_support
