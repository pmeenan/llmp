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

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "format/encoding.h"
#include "ir/graph.h"
#include "providers/device_execution.h"

namespace llmp::execution {
class BoundPlan;
}

namespace llmp::kernels {

class StepLaunch;

// Why queueing, binding or recording failed: refused with nothing queued
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
//                        row; a joined slot, every segment's in order);
//   kSegmentBytes, kStepBytes  bytes the caller supplies per segment or
//                        for the step;
//   kOutputRows          I32 [outputs]: the joined rows the head reads,
//                        each segment's last `outputs` (a joined slot);
//   kStoreCells          I64 [segment rows]: the cell of `cells` each of
//                        the segment's rows is stored at (position, or
//                        position mod capacity for a ring);
//   kMask                F16 [read, mask_rows]: the segment's mask over the
//                        first `read` cells of `cells` (ir/op.h kMaskBuild:
//                        0 where a row's position sees the position the
//                        cell holds once the step's rows are stored, causal
//                        or within `window`, else -inf; rows past the
//                        segment's -inf): a mask built on the host.
enum class InputKind : std::uint8_t {
  kTokens,
  kPositions,
  kSegmentBytes,
  kStepBytes,
  kOutputRows,
  kStoreCells,
  kMask,
};

// A state tensor's cells as an input indexes them (components/step.h
// CellKey): its capacity, and whether position p is in cell p mod capacity.
struct InputCells {
  std::uint64_t capacity = 0;
  bool ring = false;
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
};

// An output the step returns: copied to the host after the run, each
// segment's to its owner alone (independent outputs), a joined one whole.
struct OutputSlot {
  std::uint64_t device_address = 0;
  std::uint64_t bytes = 0;
  ir::TensorId tensor{};
  std::uint8_t segment = ir::kJoinedSegment;
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
struct Addresses {
  std::function<std::uint64_t(const ir::WeightRef&, format::PartRole)> weight;
  std::function<std::uint64_t(const ir::StateRef&)> state;
  std::uint64_t activations = 0;
  std::uint64_t activation_bytes = 0;
  bool measuring() const { return activations == 0; }
};

// A captured sequence of a stream's work, replayed as one (a CUDA graph,
// the device runtime's recorded work). Destroyed only once no replay of it
// is queued (D-090: with its plan, before the launch and the memory it
// names).
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
  // What `body` queues on `stream` (the launch's own), recorded instead of
  // run and made replayable. Refused (kRejected) with nothing queued when
  // the backend cannot record it; kUnknown if the attempt may have queued
  // work.
  virtual std::expected<std::unique_ptr<Recorded>, StepFailure> Capture(
      providers::NativeStream stream, const Body& body) = 0;
  // One replay of `recorded` on `stream`.
  virtual std::expected<void, StepFailure> Launch(providers::NativeStream stream,
                                                  const Recorded& recorded) = 0;
  // A fault the launch context latched (a GGML context's): every request
  // that shares it stops (the cohort's fault).
  virtual bool faulted() const { return false; }
};

class StepProgram {
 public:
  StepProgram() = default;
  StepProgram(const StepProgram&) = delete;
  StepProgram& operator=(const StepProgram&) = delete;
  StepProgram(StepProgram&&) = delete;
  StepProgram& operator=(StepProgram&&) = delete;
  virtual ~StepProgram() = default;

  // What it needs: the activations' extent it was placed in, the backend's
  // pool scratch, the host bytes the plan holds (its arena, its steps), and
  // the launches it queues (a captured graph's cost, kGraphNodeHostBytes
  // each).
  virtual std::uint64_t scratch_bytes() const = 0;
  virtual std::uint64_t pool_bytes() const { return 0; }
  virtual std::uint64_t host_bytes() const = 0;
  virtual std::uint64_t launches() const = 0;
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
  // `body` (the engine's input copies, this program's Run and the output
  // copies) recorded as one replayable sequence; a program that cannot be
  // recorded refuses (kRejected) and runs launch by launch.
  virtual std::expected<std::unique_ptr<Recorded>, StepFailure> Record(
      StepLaunch& launch, providers::NativeStream stream, const StepLaunch::Body& body) const {
    return launch.Capture(stream, body);
  }
};

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_STEP_PROGRAM_H_
