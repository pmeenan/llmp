// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/step_runs.h"

#include <format>
#include <utility>

#include "base/work_pulse.h"
#include "engine/plan_cache.h"
#include "engine/support.h"
#include "providers/device_runtime.h"

namespace llmp::engine {

namespace {

namespace kn = llmp::kernels;
using support::Error;
using support::Pointer;
using support::Round;

kn::StepFailure Unknown(std::string what) {
  return {.error = kn::StepError::kUnknown, .detail = std::move(what)};
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

std::uint64_t PlannedHostBytes(const StepPlanned& planned) {
  if (planned.program == nullptr) return 0;
  return planned.program->host_bytes();
}

std::expected<Copies, std::string> StepRuns::Layout(std::span<const kn::InputSlot> inputs,
                                                    std::uint64_t base) const {
  Copies copies;
  copies.reserve(inputs.size());
  std::uint64_t staged = base;
  for (const kn::InputSlot& input : inputs) {
    // Copied in on the device by the runner, never from the staging.
    if (input.kind == kn::InputKind::kFeatures) continue;
    if (staged > staging_bytes_ || input.bytes > staging_bytes_ - staged) {
      return Error("the inputs exceed their staging");
    }
    copies.push_back({input.device_address, input.bytes, staged});
    staged += Round(input.bytes, 256);
  }
  return copies;
}

std::expected<Copies, std::string> StepRuns::Stage(std::span<const kn::InputSlot> inputs,
                                                   const Fill& fill, std::uint64_t base) const {
  auto copies = Layout(inputs, base);
  if (!copies) return copies;
  std::size_t copy = 0;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].kind == kn::InputKind::kFeatures) continue;
    const auto& [to, bytes, at] = (*copies)[copy++];
    if (auto filled = fill(i, std::span(staging_ + at, bytes)); !filled) {
      return std::unexpected(std::move(filled.error()));
    }
  }
  return copies;
}

std::expected<void, kn::StepFailure> StepRuns::Record(
    const Copies& inputs, const std::function<bool(providers::NativeStream)>& between,
    const kn::StepProgram& program, std::span<const RunCopy> outputs,
    providers::NativeStream native) const {
  for (const auto& [to, bytes, at] : inputs) {
    if (const providers::DeviceStatus copied = providers::CopyAsync(
            native, Pointer(to), staging_ + at, bytes, providers::CopyKind::kHostToDevice);
        !copied.ok()) {
      return std::unexpected(Unknown(std::format("an input copy: {}", copied.text())));
    }
  }
  if (between && !between(native)) {
    return std::unexpected(Unknown("the work queued between the inputs and the program"));
  }
  if (auto ran = program.Run(*launch_, native); !ran) return ran;
  for (const auto& [to, from, bytes] : outputs) {
    if (const providers::DeviceStatus copied = providers::CopyAsync(
            native, Pointer(to), Pointer(from), bytes, providers::CopyKind::kDeviceToHost);
        !copied.ok()) {
      return std::unexpected(Unknown(std::format("an output's copy: {}", copied.text())));
    }
  }
  return {};
}

StepQueued StepRuns::QueueRouted(StepPlanRuns& runs, const kn::RoutedAccess& access,
                                 kn::RoutedProgram& program, bool continuation,
                                 const Copies& inputs, std::span<const RunCopy> outputs,
                                 std::span<std::byte> route_staging, bool capture,
                                 GraphStats& stats, providers::NativeStream native) const {
  StepQueued result;
  const auto index = continuation ? 1U : 0U;
  auto& recording = runs.routed_graphs[index];
  const auto body = [&]() -> std::expected<void, kn::StepFailure> {
    result.before = true;  // conservative once the first launch/copy is attempted
    if (!continuation) {
      for (const auto& [to, bytes, at] : inputs) {
        if (at > staging_bytes_ || bytes > staging_bytes_ - at)
          return std::unexpected(Unknown("routed input staging changed"));
        const auto copy = providers::CopyAsync(native, Pointer(to), staging_ + at, bytes,
                                               providers::CopyKind::kHostToDevice);
        if (!copy.ok()) return std::unexpected(Unknown(copy.text()));
      }
      if (auto prefix = program.Prefix(access, *launch_, native); !prefix) return prefix;
      for (const auto& copy : program.layout().readback) {
        std::uint64_t host_bytes = 0, device_last = 0;
        if (copy.rows == 0 || copy.bytes == 0 ||
            (copy.rows > 1 && copy.device_stride < copy.bytes) ||
            __builtin_mul_overflow(copy.rows, copy.bytes, &host_bytes) ||
            __builtin_mul_overflow(copy.rows - 1, copy.device_stride, &device_last) ||
            __builtin_add_overflow(device_last, copy.bytes, &device_last) || copy.device == 0 ||
            device_last > UINT64_MAX - copy.device || copy.host_offset > route_staging.size() ||
            host_bytes > route_staging.size() - copy.host_offset)
          return std::unexpected(Unknown("routed readback staging changed"));
        for (std::uint64_t row = 0; row != copy.rows; ++row) {
          const auto copied = providers::CopyAsync(
              native, route_staging.data() + copy.host_offset + row * copy.bytes,
              Pointer(copy.device + row * copy.device_stride), copy.bytes,
              providers::CopyKind::kDeviceToHost);
          if (!copied.ok()) return std::unexpected(Unknown(copied.text()));
        }
      }
    } else {
      if (auto tail = program.Continue(access, *launch_, native); !tail) return tail;
      for (const auto& [to, from, bytes] : outputs) {
        const auto copied = providers::CopyAsync(native, Pointer(to), Pointer(from), bytes,
                                                 providers::CopyKind::kDeviceToHost);
        if (!copied.ok()) return std::unexpected(Unknown(copied.text()));
      }
    }
    return {};
  };
  if (graphs_ && recording) {
    result.before = true;
    result.path = RunPath::kReplayed;
    result.result = program.ReplaySegment(access, *launch_, native, continuation, *recording);
    return result;
  }
  if (capture && !runs.uncapturable) {
    const auto free_before =
        providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
    (void)base::Pulse();
    auto captured = program.RecordSegment(access, *launch_, native, continuation, body);
    (void)base::Pulse();
    const auto free_after =
        providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
    (void)providers::TakeLastError();
    if (captured) {
      stats.capture_seconds += (*captured)->capture_seconds();
      stats.instantiate_seconds += (*captured)->instantiate_seconds();
      stats.nodes += (*captured)->nodes();
      stats.memory_bytes +=
          static_cast<std::int64_t>(free_before) - static_cast<std::int64_t>(free_after);
      runs.seconds += (*captured)->capture_seconds() + (*captured)->instantiate_seconds();
      runs.measured_bytes += free_before > free_after ? free_before - free_after : 0;
      recording = std::move(*captured);
      result.path = RunPath::kCaptured;
      result.before = true;
      result.result = program.ReplaySegment(access, *launch_, native, continuation, *recording);
      return result;
    }
    if (captured.error().error == kn::StepError::kUnknown) {
      result.result = std::unexpected(captured.error());
      return result;
    }
    // Capture produced no runnable graph. Keep any other successfully captured
    // segment; this segment stays eager on this plan's subsequent runs.
    runs.uncapturable = true;
    if (stats.refused++ == 0) stats.first_refusal = captured.error().detail;
  }
  result.result = body();
  return result;
}

std::expected<bool, kn::StepFailure> StepRuns::CaptureInto(
    StepPlanRuns& runs, const Copies& inputs,
    const std::function<bool(providers::NativeStream)>& between, const kn::StepProgram& program,
    std::span<const RunCopy> outputs, GraphStats& stats, providers::NativeStream native) const {
  const std::size_t free_before =
      providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
  // A capture runs on the CPU, seconds for a large plan, inside a device
  // job: these beats reach the driver's pulse (D-102, D-106).
  (void)base::Pulse();
  auto captured = program.Record(*launch_, native, [&]() -> std::expected<void, kn::StepFailure> {
    return Record(inputs, between, program, outputs, native);
  });
  (void)base::Pulse();
  const std::size_t free_after =
      providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
  (void)providers::TakeLastError();
  if (captured) {
    const kn::Recorded& recorded = **captured;
    stats.capture_seconds += recorded.capture_seconds();
    stats.instantiate_seconds += recorded.instantiate_seconds();
    stats.nodes += recorded.nodes();
    stats.memory_bytes +=
        static_cast<std::int64_t>(free_before) - static_cast<std::int64_t>(free_after);
    runs.seconds = recorded.capture_seconds() + recorded.instantiate_seconds();
    runs.measured_bytes = free_before > free_after ? free_before - free_after : 0;
    runs.graph = std::move(*captured);
    runs.copies = inputs;
    return true;
  }
  if (captured.error().error == kn::StepError::kUnknown) {
    return std::unexpected(std::move(captured.error()));
  }
  // Refused, with nothing captured queued: this plan runs launch by launch.
  runs.uncapturable = true;
  if (stats.refused++ == 0) stats.first_refusal = captured.error().detail;
  return false;
}

std::expected<bool, kn::StepFailure> StepRuns::CaptureAhead(
    StepPlanRuns& runs, const Copies& inputs, const kn::StepProgram& program,
    std::span<const RunCopy> outputs, GraphStats& stats, providers::NativeStream native) const {
  if (program.routed() || runs.routed_graphs[0] || runs.routed_graphs[1])
    return std::unexpected(
        kn::StepFailure{.detail = "ordinary capture refuses routed programs or recordings"});
  if (!graphs_ || launch_ == nullptr || runs.has_graph() || runs.uncapturable ||
      runs.ahead_refused) {
    return false;
  }
  auto captured = CaptureInto(runs, inputs, {}, program, outputs, stats, native);
  if (captured && !*captured) {
    // A plan never yet run may refuse what a capture beside its run would
    // not: leave that path open.
    runs.uncapturable = false;
    runs.ahead_refused = true;
  }
  return captured;
}

StepQueued StepRuns::Queue(StepPlanRuns& runs, const Copies& inputs,
                           const std::function<bool(providers::NativeStream)>& between,
                           const kn::StepProgram& program, std::span<const RunCopy> outputs,
                           bool capture, GraphStats& stats, providers::NativeStream native) const {
  StepQueued q;
  if (program.routed() || runs.routed_graphs[0] || runs.routed_graphs[1]) {
    q.result = std::unexpected(
        kn::StepFailure{.detail = "ordinary queue refuses routed programs or recordings"});
    return q;
  }
  if (launch_ == nullptr) {
    q.result = std::unexpected(
        kn::StepFailure{.error = kn::StepError::kRejected, .detail = "no launch is bound"});
    return q;
  }
  if (graphs_ && runs.graph) {
    // What the recording copies must be where the host staged it.
    if (inputs != runs.copies) {
      q.result = std::unexpected(
          kn::StepFailure{.error = kn::StepError::kRejected,
                          .detail = "the inputs' staging differs from the captured graph's"});
      return q;
    }
    q.path = RunPath::kReplayed;
    q.before = true;
    q.result = launch_->Launch(native, *runs.graph);
    return q;
  }
  // With nothing queued between the inputs and the program, a capture runs
  // the program launch by launch first and is recorded beside it (the
  // capture's host time overlaps the device's run); what `between` queues
  // is captured before anything runs.
  const bool beside = capture && !between;
  if (beside) {
    ++runs.eager_runs;
    q.before = true;
    q.result = Record(inputs, between, program, outputs, native);
    if (!q.result) return q;
  }
  if (capture) {
    auto captured = CaptureInto(runs, inputs, between, program, outputs, stats, native);
    if (!captured) {
      q.result = std::unexpected(std::move(captured.error()));
      return q;
    }
    if (*captured) {
      q.path = RunPath::kCaptured;
      q.before = true;
      if (!beside) q.result = launch_->Launch(native, *runs.graph);
      return q;
    }
    if (beside) return q;  // it ran already
  }
  ++runs.eager_runs;
  q.before = true;
  q.result = Record(inputs, between, program, outputs, native);
  return q;
}

}  // namespace llmp::engine
