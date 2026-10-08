// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A live state's speculation work queued through a legacy GGML runner's
// launch context (kernels/ggml/launch.h), as those runners queued it before
// the engine core split (M3.6): the saves and the owed restore are the GGML
// module's range copies (kernels/ggml/llmp_ops.h CopyRanges), refused while
// the context is faulted, and a failed launch faults it; a commit runs the
// runner's own kernels on the runner's launch context. The engine core's
// LiveState queues the same descriptors on a job's native stream instead
// (live_state.h). CUDA builds only (llmp_engine); goes with the legacy
// runners.

#ifndef LLMP_ENGINE_LIVE_STATE_LAUNCH_H_
#define LLMP_ENGINE_LIVE_STATE_LAUNCH_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <string_view>
#include <utility>

#include "catalog/catalog.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "engine/runner_resources.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"

namespace llmp::engine {

// A legacy runner's commit of a verify's first `keep` rows on its launch
// context.
using LaunchCommit = std::function<std::expected<void, kernels::ggml::KernelFailure>(
    kernels::ggml::LaunchContext& launch, std::uint32_t keep)>;

struct LiveStateLaunch {
  static std::expected<void, kernels::ggml::KernelFailure> QueueSaves(
      const LiveState& live, kernels::ggml::LaunchContext& launch);
  static std::expected<void, kernels::ggml::KernelFailure> QueueOwed(
      LiveState& live, kernels::ggml::LaunchContext& launch);
  static LiveState::Status Rollback(LiveState& live, PagedNode& node,
                                    const catalog::Closure& closure, std::uint32_t stream,
                                    kernels::ggml::LaunchContext& launch, std::string_view what);
  static void SetCommit(LiveState& live, RunnerResources& resources, LaunchCommit commit);
};

// LiveState::QueueSaves through `launch` (in the verify's job, before its
// run).
inline std::expected<void, kernels::ggml::KernelFailure> QueueSavesThrough(
    const LiveState& live, kernels::ggml::LaunchContext& launch) {
  return LiveStateLaunch::QueueSaves(live, launch);
}
// LiveState::QueueOwed through `launch`; each part stays owed unless queued.
// A commit set by LiveState::SetCommit owed is refused before anything is
// queued (the core's QueueOwed queues it).
inline std::expected<void, kernels::ggml::KernelFailure> QueueOwedThrough(
    LiveState& live, kernels::ggml::LaunchContext& launch) {
  return LiveStateLaunch::QueueOwed(live, launch);
}
// LiveState::Rollback through `launch`: what is owed, now, as a job of its
// own over `closure`; a failure, or a context faulted by then,
// quarantines.
inline LiveState::Status RollbackThrough(LiveState& live, PagedNode& node,
                                         const catalog::Closure& closure, std::uint32_t stream,
                                         kernels::ggml::LaunchContext& launch,
                                         std::string_view what) {
  return LiveStateLaunch::Rollback(live, node, closure, stream, launch, what);
}
// The state's commit: `commit` on `resources`' launch context (the context
// the runner's jobs queue through), whatever stream the state names. The
// two paths do not mix, and each refuses the other's commit before queuing
// anything: such a commit is queued only through QueueOwedThrough or
// RollbackThrough (the core's QueueOwed would not see the context's
// fault), and those queue only such a commit (they hand the state's commit
// no stream).
inline void SetCommitThrough(LiveState& live, RunnerResources& resources, LaunchCommit commit) {
  LiveStateLaunch::SetCommit(live, resources, std::move(commit));
}

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_LIVE_STATE_LAUNCH_H_
