// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/common/recording_launch.h"

#include <chrono>
#include <format>
#include <utility>

#include "providers/device_runtime.h"

namespace llmp::kernels {

namespace {

class RecordedWork final : public Recorded {
 public:
  RecordedWork(const StepLaunch* owner, providers::RecordedWork work, double capture,
               double instantiate)
      : Recorded(owner), work_(std::move(work)), capture_(capture), instantiate_(instantiate) {}
  std::size_t nodes() const override { return 0; }
  double capture_seconds() const override { return capture_; }
  double instantiate_seconds() const override { return instantiate_; }
  const providers::RecordedWork& work() const { return work_; }

 private:
  providers::RecordedWork work_;
  double capture_;
  double instantiate_;
};

double Since(std::chrono::steady_clock::time_point from) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
}

}  // namespace

std::expected<std::unique_ptr<Recorded>, StepFailure> RecordingLaunch::Capture(
    providers::NativeStream stream, const Body& body) {
  const auto started = std::chrono::steady_clock::now();
  if (const providers::DeviceStatus begun = providers::BeginRecording(stream); !begun.ok()) {
    (void)providers::TakeLastError();
    return std::unexpected(StepFailure{.error = StepError::kRejected,
                                       .detail = std::format("recording: {}", begun.text())});
  }
  auto recorded = body();
  const double capture = Since(started);
  const auto ended_at = std::chrono::steady_clock::now();
  // The recording always ends, whatever the body did: nothing it queued ran.
  auto ended = providers::EndRecording(stream);
  if (!recorded) {
    (void)providers::TakeLastError();
    return std::unexpected(std::move(recorded.error()));
  }
  if (!ended) {
    (void)providers::TakeLastError();
    return std::unexpected(
        StepFailure{.error = StepError::kRejected,
                    .detail = std::format("ending the recording: {}", ended.error().text())});
  }
  ++captures_;
  return std::make_unique<RecordedWork>(this, std::move(*ended), capture, Since(ended_at));
}

std::expected<void, StepFailure> RecordingLaunch::Launch(providers::NativeStream stream,
                                                         const Recorded& recorded) {
  if (recorded.owner() != this) {
    return std::unexpected(
        StepFailure{.error = StepError::kRejected, .detail = "the recording is not this launch's"});
  }
  const auto* work = static_cast<const RecordedWork*>(&recorded);
  if (const providers::DeviceStatus replayed = providers::Replay(work->work(), stream);
      !replayed.ok()) {
    (void)providers::TakeLastError();
    return std::unexpected(StepFailure{.error = StepError::kUnknown,
                                       .detail = std::format("a replay: {}", replayed.text())});
  }
  ++launches_;
  return {};
}

}  // namespace llmp::kernels
