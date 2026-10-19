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
  std::uint64_t save = 0, commit_indices = 0;
  bool commit_only = false;
  std::vector<CellTensor> taps;  // the drafters' tensors the target writes
  CellTensor tap;                // a drafter's tap it reads
  CellTensor borrowed;           // a drafter's view of the target's tensor
  bool greedy = false;
  std::int64_t past = -1;    // Position's, for a positioned program (-1: the input's)
  std::uint32_t logits = 0;  // F32 logits a row (0: FakeRows)
  bool target = true;        // the target's stage: its logits mark lost state
  // A drafter's features: its rows' (an input) and its next ones (an
  // output), a cell's bytes each.
  std::uint64_t features = 0, next = 0, feature_bytes = 0;
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
  // The extra inputs' slots (their indices in inputs()), recorded each run.
  void SetExtra(std::vector<std::size_t> extra) { extra_ = std::move(extra); }

  std::uint64_t scratch_bytes() const override { return scratch_; }
  std::uint64_t pool_bytes() const override { return options_->pool_bytes; }
  kn::PlanCoverage plan_coverage() const override { return options_->plan_coverage; }
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
    std::vector<std::pair<std::uint64_t, std::uint64_t>> extra;
    for (const std::size_t i : extra_)
      extra.emplace_back(inputs_[i].device_address, inputs_[i].bytes);
    auto launched = providers::fake::FakeDeviceExecution::Launch(
        stream, [segments = segments_, executed = executed_, extra = std::move(extra),
                 seen = options_->seen_inputs, options = options_, phase = phase_] {
          Execute(segments, *options, phase);
          seen->clear();
          for (const auto& [address, bytes] : extra) {
            const auto* from = At<const std::byte>(address);
            seen->emplace_back(from, from + bytes);
          }
          executed->fetch_add(1);
          if (options->after_execute) options->after_execute(phase);
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
    if (options_->fail_record)
      return std::unexpected(
          kn::StepFailure{.error = *options_->fail_record, .detail = "the fake capture failed"});
    if (options_->refuse_record) {
      return std::unexpected(
          kn::StepFailure{.error = kn::StepError::kRejected, .detail = "the fake refuses"});
    }
    return launch.Capture(stream, body);
  }

 private:
  // The device's work: on the fake device's thread, at run time.
  static void Execute(const std::vector<SegmentPlan>& segments, const FakeSourceOptions& options,
                      pl::Phase phase) {
    for (const SegmentPlan& s : segments) {
      if (s.commit_only) {
        std::int64_t row = -1;
        std::memcpy(&row, At<const void>(s.commit_indices + (s.rows - 1) * sizeof(row)),
                    sizeof(row));
        std::uint64_t hash = UINT64_MAX;
        if (row >= 0 && static_cast<std::uint64_t>(row) < s.rows)
          std::memcpy(&hash,
                      At<const void>(s.save + static_cast<std::uint64_t>(row) * sizeof(hash)),
                      sizeof(hash));
        std::memcpy(At<void>(s.fixed), &hash, sizeof(hash));
        continue;
      }
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
        // reads the target's features there), before this row writes; or
        // the feature the runner copied in for the row.
        if (s.features != 0) {
          FakeCell f;
          std::memcpy(&f, At<const void>(s.features + (r * s.feature_bytes)), sizeof(f));
          if (f.magic == kFakeMagic && f.slot == s.slot) taps[r] = f.token;
        } else if (const auto tapped = Valid(s.tap, s.slot, at - 1, r)) {
          taps[r] = tapped->token;
        }
        const FakeCell cell{
            .magic = kFakeMagic, .slot = s.slot, .position = at, .token = tokens[r]};
        if (const auto own = s.own.Cell(at, r)) std::memcpy(s.own.At(*own), &cell, sizeof(cell));
        for (const CellTensor& t : s.taps) {
          if (const auto c = t.Cell(at, r)) std::memcpy(t.At(*c), &cell, sizeof(cell));
        }
        hash = FakeHash(hash, tokens[r]);
        hashes[r] = hash;
        if (s.save != 0) std::memcpy(At<void>(s.save + r * sizeof(hash)), &hash, sizeof(hash));
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
      for (std::uint32_t o = 0; o < s.outputs && s.next != 0; ++o) {
        const std::uint32_t r = s.rows - s.outputs + o;
        const FakeCell next{.magic = kFakeMagic,
                            .slot = s.slot,
                            .position = position_of(r),
                            .token = tokens[r] + kFakeFeatureStep};
        std::memcpy(At<void>(s.next + (o * s.feature_bytes)), &next, sizeof(next));
      }
      // The original diagnostic-row fixture exposes only its last greedy
      // token. A logits head exposes every selected row's native token.
      if (s.greedy && s.logits == 0 && s.outputs != 0) {
        const std::int32_t chosen = tokens[s.rows - 1] + 1000;
        std::memcpy(At<void>(s.out), &chosen, sizeof(chosen));
        continue;
      }
      for (std::uint32_t o = 0; o < s.outputs; ++o) {
        const std::uint32_t r = s.rows - s.outputs + o;
        if (s.greedy) {
          const std::int32_t chosen = options.native_token && (!options.native_token_phase ||
                                                               *options.native_token_phase == phase)
                                          ? *options.native_token
                                      : s.logits != 0 ? choice(r)
                                                      : tokens[r] + 1000;
          std::memcpy(At<void>(s.out + std::uint64_t{o} * sizeof(chosen)), &chosen, sizeof(chosen));
          continue;
        }
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
  std::vector<std::size_t> extra_;
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

std::expected<std::vector<engine::CommitBuffer>, std::string> FakeStepSource::CommitBuffers(
    std::uint16_t stage, std::uint32_t /*rows*/) const {
  if (!options_.commits || stage >= options_.layouts.size())
    return std::vector<engine::CommitBuffer>{};
  return std::vector<engine::CommitBuffer>{
      {.layer = 0, .component = 0, .what = "hash", .row_bytes = 8}};
}

std::expected<std::unique_ptr<kn::StepProgram>, std::string> FakeStepSource::BuildPrivate(
    const pl::StepShape& shape, const kn::Addresses& addresses,
    const kn::ExecutionResources& resources) {
  if (!options_.private_backend || resources.parent() != options_.private_backend.get() ||
      !resources.Ready())
    return Refuse("fake private build requires its actual producer-owned ready view");
  auto program = Build(shape, addresses);
  if (!program) return program;
  if (!*program || !(**program).HoldResources(resources))
    return Refuse("fake private program cannot retain its actual view");
  return program;
}

std::expected<std::unique_ptr<kn::StepProgram>, std::string> FakeStepSource::Build(
    const pl::StepShape& shape, const kn::Addresses& addresses) {
  Built observed{.shape = shape,
                 .measuring = addresses.measuring(),
                 .activation_alignment = addresses.activation_alignment};
  if (!addresses.measuring() && addresses.activation_alignment != 0)
    observed.bound_alignments_hold = addresses.activations % addresses.activation_alignment == 0;
  if (shape.stage < options_.layouts.size() && addresses.state_alignment) {
    for (std::uint8_t owner = 0; owner < shape.segments.size(); ++owner) {
      for (const auto& item : options_.layouts[shape.stage].items) {
        const ir::StateRef ref{.stage = shape.stage,
                               .tensor = static_cast<std::uint16_t>(item.tensor),
                               .layer = item.layer,
                               .segment = owner};
        const auto a = addresses.state_alignment(ref);
        observed.state_alignments.push_back(a);
        if (a != 0) observed.bound_alignments_hold &= addresses.state(ref) % a == 0;
      }
    }
  }
  if (shape.stage < options_.layouts.size() && addresses.state_alignment) {
    for (std::uint8_t owner = 0; owner < shape.segments.size(); ++owner) {
      for (const auto& item : options_.layouts[shape.stage].borrowed) {
        const ir::StateRef ref{.stage = shape.stage,
                               .tensor = static_cast<std::uint16_t>(item.tensor),
                               .layer = item.layer,
                               .segment = owner};
        const auto a = addresses.state_alignment(ref);
        observed.borrowed_alignments.push_back(a);
        if (a != 0) observed.bound_alignments_hold &= addresses.state(ref) % a == 0;
      }
    }
    observed.invalid_alignment_refs_unknown =
        addresses.state_alignment({.stage = shape.stage, .tensor = 65535}) == 0 &&
        addresses.state_alignment({.stage = shape.stage, .segment = ir::kJoinedSegment}) == 0 &&
        addresses.state_alignment({.stage = 65535}) == 0;
  }
  built_.push_back(std::move(observed));
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
  // A stage that reads frozen stores no rows of its own (it may have none).
  const bool frozen = shape.stage != 0 && model::ReadsFrozen(layout);
  if ((kv == nullptr && !frozen) || (kv != nullptr && kv->contract.cell_bytes < sizeof(FakeCell))) {
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
  // A drafter's features: one joined input of a cell a row.
  const bool features = options_.features && shape.stage != 0 && tap != nullptr;
  const std::uint64_t feature_bytes = features ? tap->contract.cell_bytes : 0;
  std::uint64_t joined_features = 0;
  if (features) {
    joined_features = place(std::uint64_t{total} * feature_bytes);
    inputs.push_back({.device_address = joined_features,
                      .bytes = std::uint64_t{total} * feature_bytes,
                      .tensor = {},
                      .segment = ir::kJoinedSegment,
                      .kind = kn::InputKind::kFeatures});
  }
  std::uint32_t first = 0;
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const pl::Segment& seg = shape.segments[i];
    SegmentPlan s;
    if (features) {
      s.features = joined_features + (first * feature_bytes);
      s.feature_bytes = feature_bytes;
    }
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
    if (options_.commits && (verify || shape.phase == pl::Phase::kCommit)) {
      s.save = addresses.save(0, 0, "hash", segment);
      s.commit_only = shape.phase == pl::Phase::kCommit;
      if (!addresses.measuring() && s.save == 0) return Refuse("the commit save has no address");
    }
    if (kv != nullptr && !frozen) {
      const std::uint64_t base =
          addresses.state({.stage = shape.stage, .tensor = 0, .layer = 0, .segment = segment});
      if (base == 0) return Refuse("a state tensor has no address");
      s.own = TensorOf(base, *kv);
      uses.push_back(
          {.address = base,
           .bytes = std::min<std::uint64_t>(seg.read_cells, kv->cells) * kv->contract.cell_bytes,
           .role = ir::Role::kState});
    }
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
      // Native logits heads select one token per output row; the original
      // diagnostic-row fixture retains its single last-token contract.
      const std::uint64_t bytes =
          seg.greedy ? std::uint64_t{options_.logits != 0 ? seg.outputs : 1U} * sizeof(std::int32_t)
                     : std::uint64_t{seg.outputs} *
                           (options_.logits != 0 ? std::uint64_t{options_.logits} * sizeof(float)
                                                 : sizeof(FakeRow));
      s.out = place(bytes);
      outputs.push_back(
          {.device_address = s.out, .bytes = bytes, .tensor = {}, .segment = segment});
    }
    segments.push_back(s);
  }
  // Each drafter segment's next features, after every segment's rows.
  for (std::size_t i = 0; i < segments.size() && features; ++i) {
    SegmentPlan& s = segments[i];
    if (s.outputs == 0) continue;
    s.next = place(std::uint64_t{s.outputs} * feature_bytes);
    outputs.push_back({.device_address = s.next,
                       .bytes = std::uint64_t{s.outputs} * feature_bytes,
                       .tensor = {},
                       .segment = static_cast<std::uint8_t>(i),
                       .features = true});
  }
  if (shape.phase == pl::Phase::kCommit) {
    inputs.clear();
    for (std::size_t i = 0; i < segments.size(); ++i) {
      auto& segment = segments[i];
      segment.commit_indices = place(segment.rows * sizeof(std::int64_t));
      inputs.push_back({.device_address = segment.commit_indices,
                        .bytes = segment.rows * sizeof(std::int64_t),
                        .segment = static_cast<std::uint8_t>(i),
                        .kind = kn::InputKind::kStoreCells,
                        .cells = {.capacity = segment.rows, .per_row = true}});
    }
  }
  std::vector<std::size_t> extra;
  if (options_.extra_inputs) {
    for (kn::InputSlot in : options_.extra_inputs(shape)) {
      in.device_address = place(in.bytes);
      extra.push_back(inputs.size());
      inputs.push_back(in);
    }
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
  if (options_.extra_activation_bytes != 0) {
    const auto activation_tail = place(options_.extra_activation_bytes);
    uses.push_back({.address = activation_tail,
                    .bytes = options_.extra_activation_bytes,
                    .role = ir::Role::kActivation});
  }
  const std::uint64_t scratch = at - addresses.activations;
  if (!addresses.measuring() && scratch > addresses.activation_bytes) {
    return Refuse(std::format("{} bytes of activations exceed the workspace's {}", scratch,
                              addresses.activation_bytes));
  }
  auto program = std::make_unique<FakeProgram>(
      std::move(segments), std::move(inputs), std::move(outputs), std::move(uses), scratch,
      options_.host_bytes, options_.launches, addresses.measuring(), &options_, queued_, executed_,
      positioned_, shape.phase);
  program->SetExtra(std::move(extra));
  if (options_.construction_owner && !program->HoldConstruction(options_.construction_owner()))
    return Refuse("fake construction owner was not admitted");
  return program;
}

}  // namespace llmp::test_support
