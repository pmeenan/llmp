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
// - writes, for each row, the cell of the stage's per-position tensor
//   (layer 0, tensor 0) at the row's position: a Cell {magic, slot,
//   position, token};
// - advances the stage's fixed state (layer 0, tensor 1, if any): a 64-bit
//   hash folded over each row's token, so chunkings and replays of one
//   prompt must agree;
// - returns its last `outputs` rows, each a Row {slot, position, token,
//   cells} where `cells` counts the cells in [0, read_cells) holding a
//   valid Cell for their own position (all of past + rows when the state
//   persisted), and the hash after the row;
// - for a shape with greedy segments, one joined output: each segment's
//   last token + 1000, as I32.
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
};
static_assert(sizeof(FakeRow) == 24);

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
  bool refuse_record = false;
  bool refuse_bind = false;
  // Run fails: refused with nothing queued (kRejected), or of unknown
  // effect after its work was queued (kUnknown).
  std::optional<kernels::StepError> fail_run;
  // Called at the start of every Build (on the driver; a lookahead's in
  // the job's meanwhile).
  std::function<void(const planner::StepShape&, bool measuring)> on_build;
  // Programs are not capturable and take each row's position from
  // StepProgram::Position (past + row) instead of their positions input.
  bool positioned = false;
  // A tensor use outside every cataloged span (the coverage check's case).
  bool stray_use = false;
  // An output slot naming a segment past the shape's.
  bool stray_output = false;
  base::Sha256Digest spec{};
};

class FakeStepSource final : public engine::StepSource {
 public:
  explicit FakeStepSource(FakeSourceOptions options) : options_(std::move(options)) {}
  planner::PlanKey KeyBase() const override;
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
