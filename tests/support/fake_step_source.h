// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A step source for the engine core's tests on the fake device backend
// (engine/model_runner.h): its programs run as host work on the fake
// device's streams (providers/fake/fake_device_execution.h) and write
// deterministic bytes, so a test can check which slots, rows and positions
// each step touched, and that what it read was there.
//
// Each step program, for each segment (owner) of its shape:
// - reads its tokens and positions from its inputs (per segment, or one
//   joined input of every segment's rows);
// - writes, for each row, the cell of the stage's tensor 0 (layer 0; a
//   cell per position, or a ring's or position ring's cell) for the row's
//   position: a Cell {magic, slot, position, token};
// - advances the stage's fixed state (layer 0, tensor 1, if a kFixed one):
//   a 64-bit hash folded over each row's token, so chunkings and replays
//   of one prompt must agree; a verify leaves kCommit state unwritten (its
//   commit rebuilds it);
// - on the target (stage 0), writes each row's Cell into every drafter
//   tensor the target writes (target_writes: a feature tap), at the cell
//   its contract gives the row;
// - on a drafter, reads for each output row the tap (its stage's first
//   target_writes tensor) the target wrote for the position before the
//   row's, and counts the valid cells of the target's tensor it borrows
//   (its first borrowed one) before its first row;
// - returns its last `outputs` rows, each a Row {slot, position, token,
//   cells, hash, tap, borrowed} where `cells` counts the cells in [0,
//   read_cells) holding a valid Cell for their own position (all of past +
//   rows when the state persisted), `hash` is the fixed state's after the
//   row, `tap` the tapped token (-1: none valid) and `borrowed` the count;
// - for a greedy segment, its own output is instead one I32 (as the GGML
//   lowering's): its last token + 1000.
// With `logits` (a serving adapter's head, runtime/engine_llm.h), each
// output row is instead that many F32 logits, every one 0 but a 1 at
// FakeLogit(token, position): on the target only while its tensor 0 holds a
// valid cell for every position through the row's (else at logits - 1, the
// lost state's mark), on a drafter always; and a greedy segment's I32 is
// its last row's choice.
// With `features`, a drafter's rows read their feature from a kFeatures
// input (which the runner copies in on the device: its stage's tap cell, or
// its carry) instead of its tap's state, its row's `tap` that feature's
// token (-1: no valid cell of its slot), and each output row's next feature
// follows its rows as a features output: a FakeCell of its slot and position
// with its token + kFakeFeatureStep.
// `extra_inputs` adds input slots a test describes (computed ones: store
// cells, masks), each placed after the program's own; every run records
// their bytes as staged (`seen_inputs`, the last run's).
// Recording goes through the launch (RecordingLaunch), so a replay reads
// the inputs staged for it and writes the same way.

#ifndef LLMP_TESTS_SUPPORT_FAKE_STEP_SOURCE_H_
#define LLMP_TESTS_SUPPORT_FAKE_STEP_SOURCE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "engine/model_runner.h"
#include "ir/graph.h"
#include "kernels/common/step_program.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"

namespace llmp::test_support {

struct FakeCell {
  std::int32_t magic = 0;
  std::int32_t slot = 0;
  std::int32_t position = 0;
  std::int32_t token = 0;
};
inline constexpr std::int32_t kFakeMagic = 0x5EED;

struct FakeRow {
  std::int32_t slot = 0;
  std::int32_t position = 0;
  std::int32_t token = 0;
  std::int32_t cells = 0;
  std::uint64_t hash = 0;
  std::int32_t tap = -1;
  std::int32_t borrowed = 0;
};
static_assert(sizeof(FakeRow) == 32);
inline constexpr std::int32_t kFakeFeatureStep = 100000;

// The token a logits row names for `token` at `position` (below logits - 1).
inline std::int32_t FakeLogit(std::int32_t token, std::int32_t position, std::uint32_t logits) {
  const std::int64_t m = std::int64_t{logits} - 1;
  if (m <= 0) return 0;
  const std::int64_t v = (std::int64_t{token} * 7) + (std::int64_t{position} * 3) + 1;
  return static_cast<std::int32_t>(((v % m) + m) % m);
}

// The fixed state's hash after `token`, from `hash`.
inline std::uint64_t FakeHash(std::uint64_t hash, std::int32_t token) {
  return (hash * 1000003U) + static_cast<std::uint64_t>(token) + 1U;
}

struct FakeSourceOptions {
  // The stages' layouts (the slots' own), to find each state tensor.
  std::vector<model::StateLayoutDerived> layouts;
  bool joined_inputs = false;  // one joined token and position input
  std::uint64_t host_bytes = 4096;
  std::uint64_t launches = 8;
  // Test hooks: Build refuses a shape; programs refuse to record. The
  // program hooks are read when a program binds, records or runs, so a
  // test may change them between steps.
  std::function<bool(const planner::StepShape&)> refuse_build;
  std::function<std::vector<std::uint64_t>(const planner::StepShape&)> shape_parameters;
  bool refuse_record = false;
  bool refuse_bind = false;
  // Run fails: refused with nothing queued (kRejected), or of unknown
  // effect after its work was queued (kUnknown).
  std::optional<kernels::StepError> fail_run;
  // Only programs of this phase fail (none: every program).
  std::optional<planner::Phase> fail_phase;
  // Called at the start of every Build (on the driver; a lookahead's in
  // the job's meanwhile).
  std::function<void(const planner::StepShape&, bool measuring)> on_build;
  // Programs are not capturable and take each row's position from
  // StepProgram::Position (past + row) instead of their positions input.
  bool positioned = false;
  // A tensor use outside every cataloged span (the coverage check's case).
  bool stray_use = false;
  // State a program binds besides its own (a tap no contract declares):
  // Build refuses when one binds nothing.
  std::function<std::vector<ir::StateRef>(const planner::StepShape&)> extra_state;
  // An output slot naming a segment past the shape's.
  bool stray_output = false;
  // Output rows of this many F32 logits (above) instead of FakeRows; 0:
  // FakeRows.
  std::uint32_t logits = 0;
  // A drafter's features as a kFeatures input and a features output.
  bool features = false;
  // Input slots added to every program (their address and bytes set by the
  // fake: bytes from the slot's `bytes`), and what the last run staged in
  // each.
  std::function<std::vector<kernels::InputSlot>(const planner::StepShape&)> extra_inputs;
  std::shared_ptr<std::vector<std::vector<std::byte>>> seen_inputs =
      std::make_shared<std::vector<std::vector<std::byte>>>();
  base::Sha256Digest spec{};
};

class FakeStepSource final : public engine::StepSource {
 public:
  explicit FakeStepSource(FakeSourceOptions options) : options_(std::move(options)) {}
  planner::PlanKey KeyBase() const override;
  std::vector<std::uint64_t> ShapeParameters(const planner::StepShape& shape) const override {
    return options_.shape_parameters ? options_.shape_parameters(shape)
                                     : std::vector<std::uint64_t>{};
  }
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Build(
      const planner::StepShape& shape, const kernels::Addresses& addresses) override;

  // Every Build, measuring or not, in order.
  struct Built {
    planner::StepShape shape;
    bool measuring = false;
  };
  const std::vector<Built>& built() const { return built_; }
  // Runs queued by its programs (eager runs and captures' recordings
  // alike) and host work actually run on the device.
  std::uint64_t queued() const { return queued_->load(); }
  std::uint64_t executed() const { return executed_->load(); }
  // Position calls on its programs.
  std::uint64_t positioned() const { return positioned_->load(); }
  FakeSourceOptions& options() { return options_; }

 private:
  FakeSourceOptions options_;
  std::vector<Built> built_;
  // Shared with the programs; the device's host work counts on its thread.
  std::shared_ptr<std::atomic<std::uint64_t>> queued_ =
      std::make_shared<std::atomic<std::uint64_t>>(0);
  std::shared_ptr<std::atomic<std::uint64_t>> executed_ =
      std::make_shared<std::atomic<std::uint64_t>>(0);
  std::shared_ptr<std::atomic<std::uint64_t>> positioned_ =
      std::make_shared<std::atomic<std::uint64_t>>(0);
};

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_FAKE_STEP_SOURCE_H_
