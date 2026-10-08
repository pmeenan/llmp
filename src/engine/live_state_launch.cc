// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/live_state_launch.h"

#include <expected>
#include <format>
#include <string>
#include <utility>

#include "engine/support.h"
#include "kernels/ggml/llmp_ops.h"
#include "providers/device_execution.h"
#include "scheduler/commands.h"

namespace llmp::engine {

namespace {

namespace kg = llmp::kernels::ggml;
namespace sc = llmp::scheduler;
using support::Error;

}  // namespace

std::expected<void, kg::KernelFailure> LiveStateLaunch::QueueSaves(const LiveState& live,
                                                                   kg::LaunchContext& launch) {
  return kg::CopyRanges(launch, live.save_, live.save_count_);
}

std::expected<void, kg::KernelFailure> LiveStateLaunch::QueueOwed(LiveState& live,
                                                                  kg::LaunchContext& launch) {
  if (live.commit_keep_ != 0 && !live.launch_commit_) {
    // The two paths never mix: a core commit would be handed no stream.
    return std::unexpected(kg::KernelFailure{
        .error = kg::KernelError::kRejected,
        .detail = std::format("{}'s commit queues on a job's stream (LiveState::QueueOwed)",
                              live.model_)});
  }
  if (live.owed()) live.InvalidatePlaces();
  if (live.restore_count_ != 0) {
    // Still owed until the copy is queued.
    if (auto r = kg::CopyRanges(launch, live.restore_, live.restore_count_); !r) {
      return r;
    }
    live.restore_count_ = 0;
  }
  if (live.commit_keep_ != 0) {
    // A SetCommitThrough commit queues on its own launch context; the
    // stream it is handed goes unused.
    if (auto r = live.commit_(providers::NativeStream{}, live.commit_keep_); !r) {
      return std::unexpected(kg::KernelFailure{
          .error = r.error().unknown ? kg::KernelError::kUnknown : kg::KernelError::kRejected,
          .detail = std::move(r.error().detail)});
    }
    live.commit_keep_ = 0;
  }
  return {};
}

LiveState::Status LiveStateLaunch::Rollback(LiveState& live, PagedNode& node,
                                            const catalog::Closure& closure, std::uint32_t stream,
                                            kg::LaunchContext& launch, std::string_view what) {
  if (auto usable = live.Usable(); !usable) {
    return usable;
  }
  if (!live.owed()) {
    return {};
  }
  std::string failed;
  auto posted = node.Job(
      closure,
      [&](providers::NativeStream) {
        if (auto r = QueueOwed(live, launch); !r) {
          failed = r.error().detail;
          return sc::JobResult::kUnknown;
        }
        return sc::JobResult::kQueued;
      },
      what, stream);
  if (!posted || !failed.empty()) {
    // What was not queued stays owed; what was has completed (the job
    // retired); a launch of unknown effect quarantines.
    live.Settle(false, false, !failed.empty() || launch.faulted());
    return Error(failed.empty() ? posted.error() : failed);
  }
  return {};
}

void LiveStateLaunch::SetCommit(LiveState& live, RunnerResources& resources, LaunchCommit commit) {
  live.SetCommit([&resources, commit = std::move(commit)](providers::NativeStream /*stream*/,
                                                          std::uint32_t keep) -> LiveState::Queued {
    if (auto r = commit(resources.launch(), keep); !r) {
      return std::unexpected(
          LiveState::QueueFailure{.unknown = r.error().error == kg::KernelError::kUnknown,
                                  .detail = std::move(r.error().detail)});
    }
    return {};
  });
  live.launch_commit_ = true;
}

}  // namespace llmp::engine
