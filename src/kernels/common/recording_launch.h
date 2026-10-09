// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A step launch (step_program.h) over the device runtime's recorded work
// (providers/device_runtime.h: BeginRecording, EndRecording, Replay): the
// launch any backend whose programs queue plain stream work uses, and the
// fake device's in the engine core's tests. A capture records what its
// body queues on the stream and makes it replayable; a replay is one
// queued launch of it. The GGML launch context's captures go through its
// own wrapper (kernels/ggml), which keeps its pool and fault rules.

#ifndef LLMP_KERNELS_COMMON_RECORDING_LAUNCH_H_
#define LLMP_KERNELS_COMMON_RECORDING_LAUNCH_H_

#include <cstddef>
#include <expected>
#include <memory>

#include "kernels/common/step_program.h"
#include "providers/device_execution.h"

namespace llmp::kernels {

class RecordingLaunch final : public StepLaunch {
 public:
  std::expected<std::unique_ptr<Recorded>, StepFailure> Capture(providers::NativeStream stream,
                                                                const Body& body) override;
  std::expected<void, StepFailure> Launch(providers::NativeStream stream,
                                          const Recorded& recorded) override;
  // Captures and replays made through it.
  std::size_t captures() const { return captures_; }
  std::size_t launches() const { return launches_; }

 private:
  std::size_t captures_ = 0;
  std::size_t launches_ = 0;
};

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_RECORDING_LAUNCH_H_
