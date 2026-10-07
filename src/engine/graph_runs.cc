// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/graph_runs.h"

#include <cstring>
#include <format>
#include <utility>

#include "base/work_pulse.h"
#include "engine/support.h"
#include "providers/device_runtime.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
using support::Address;
using support::Error;
using support::Pointer;
using support::Round;

kg::KernelFailure Unknown(std::string what) {
  return {.error = kg::KernelError::kUnknown, .detail = std::move(what)};
}

}  // namespace

void Count(GraphStats& stats, RunPath path) {
  switch (path) {
    case RunPath::kEager:
      ++stats.eager;
      break;
    case RunPath::kCaptured:
      ++stats.captured;
      break;
    case RunPath::kReplayed:
      ++stats.replayed;
      break;
  }
}

std::expected<Copies, std::string> GraphRuns::Stage(
    std::span<const std::pair<ggml_tensor*, const void*>> sources, std::uint64_t base) const {
  Copies copies;
  copies.reserve(sources.size());
  std::uint64_t staged = base;
  for (const auto& [tensor, source] : sources) {
    const std::uint64_t bytes = ggml_nbytes(tensor);
    if (staged + bytes > staging_bytes_) {
      return Error("the inputs exceed their staging");
    }
    std::memcpy(staging_ + staged, source, bytes);
    copies.push_back({Address(tensor->data), bytes, staged});
    staged += Round(bytes, 256);
  }
  return copies;
}

Queued GraphRuns::Queue(PlanRuns& runs, const Copies& inputs,
                        const std::function<bool(void* stream)>& between, kg::BoundGraph& bound,
                        std::span<const RunCopy> outputs, bool capture, GraphStats& stats,
                        providers::NativeStream native) const {
  // The input copies, `between`, the plan and the outputs' copies, as one
  // run queues them and a capture records them.
  const auto queue = [&](kg::LaunchContext& launch) -> std::expected<void, kg::KernelFailure> {
    for (const auto& [to, bytes, at] : inputs) {
      if (const providers::DeviceStatus copied = providers::CopyAsync(
              native, Pointer(to), staging_ + at, bytes, providers::CopyKind::kHostToDevice);
          !copied.ok()) {
        return std::unexpected(Unknown(std::format("an input copy: {}", copied.text())));
      }
    }
    if (between && !between(native.handle)) {
      return std::unexpected(Unknown("the work queued between the inputs and the plan"));
    }
    if (auto r = bound.Run(launch); !r) {
      return r;
    }
    for (const auto& [to, from, bytes] : outputs) {
      if (const providers::DeviceStatus copied = providers::CopyAsync(
              native, Pointer(to), Pointer(from), bytes, providers::CopyKind::kDeviceToHost);
          !copied.ok()) {
        return std::unexpected(Unknown(std::format("an output's copy: {}", copied.text())));
      }
    }
    return {};
  };
  Queued q;
  if (graphs_ && runs.graph.has_value()) {
    // What the graph copies must be where the host staged it.
    if (inputs != runs.copies) {
      q.result = std::unexpected(
          kg::KernelFailure{.error = kg::KernelError::kRejected,
                            .detail = "the inputs' staging differs from the captured graph's"});
      return q;
    }
    q.path = RunPath::kReplayed;
    q.result = launch_->Launch(*runs.graph);
    return q;
  }
  // With nothing queued between the inputs and the plan, a capture runs
  // the plan launch by launch first and is recorded beside it: the capture
  // and instantiation take host time the run's device work covers, rather
  // than leaving the device idle before a first replay. What `between`
  // queues is captured before anything runs, as it always was.
  const bool beside = capture && !between;
  if (beside) {
    ++runs.eager_runs;
    q.before = true;  // any input copy before a refusal
    q.result = queue(*launch_);
    if (!q.result) {
      return q;
    }
  }
  if (capture) {
    const std::size_t free_before =
        providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
    // A capture and its instantiation run on the CPU, seconds for a large
    // plan. In the service they run inside a device job on the lane thread,
    // which has no pulse, so these beats only reach a caller on the driver
    // thread; there a capture is covered by its unit's allowance instead
    // (D-102).
    (void)base::Pulse();
    auto captured = launch_->Capture(queue);
    (void)base::Pulse();
    const std::size_t free_after =
        providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
    (void)providers::TakeLastError();
    if (captured) {
      stats.capture_seconds += captured->capture_seconds();
      stats.instantiate_seconds += captured->instantiate_seconds();
      stats.nodes += captured->nodes();
      stats.memory_bytes +=
          static_cast<std::int64_t>(free_before) - static_cast<std::int64_t>(free_after);
      runs.seconds = captured->capture_seconds() + captured->instantiate_seconds();
      runs.measured_bytes = free_before > free_after ? free_before - free_after : 0;
      runs.graph.emplace(std::move(*captured));
      runs.copies = inputs;
      q.path = RunPath::kCaptured;
      q.before = true;  // the upload
      if (!beside) {
        q.result = launch_->Launch(*runs.graph);
      }
      return q;
    }
    if (captured.error().error == kg::KernelError::kUnknown) {
      q.result = std::unexpected(captured.error());
      return q;
    }
    // Refused, with nothing captured queued: this plan runs launch by launch.
    runs.uncapturable = true;
    if (stats.refused++ == 0) {
      stats.first_refusal = captured.error().detail;
    }
    if (beside) {
      return q;  // it ran already
    }
  }
  ++runs.eager_runs;
  q.before = true;  // any input copy before a refusal
  q.result = queue(*launch_);
  return q;
}

}  // namespace jitllm::engine
