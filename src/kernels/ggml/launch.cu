// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "common.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"

namespace llmp::kernels::ggml {

// GGML's scratch pool over the caller's workspace: a stack of 256-byte
// aligned blocks. GGML frees a block when the host launcher that took it
// returns, while its kernels may still run; reusing the offset is safe in
// stream order, and the workspace itself outlives the launches (launch.h).
// GGML uses what alloc returns without checking it, so Run bounds each
// operation's scratch first, and a request past the bound is a broken
// invariant, not an error.
class WorkspacePool final : public ggml_cuda_pool {
 public:
  WorkspacePool(std::uint64_t base, std::uint64_t size) : base_(base), size_(size) {}

  void* alloc(std::size_t size, std::size_t* actual_size) override {
    const std::uint64_t offset = (top_ + kAlign - 1) / kAlign * kAlign;
    base::Check(offset <= limit_ && size <= limit_ - offset,
                "a GGML launcher drew more scratch than its operation's bound");
    blocks_.push_back(Block{.offset = offset, .end = offset + size, .live = true});
    top_ = offset + size;
    peak_ = std::max(peak_, top_);
    *actual_size = size;
    return reinterpret_cast<void*>(base_ + offset);  // NOLINT(performance-no-int-to-ptr)
  }

  void free(void* ptr, std::size_t size) override {
    const std::uint64_t offset = reinterpret_cast<std::uintptr_t>(ptr) - base_;
    bool found = false;
    for (Block& block : blocks_) {
      if (block.live && block.offset == offset && block.end - block.offset == size) {
        block.live = false;
        found = true;
        break;
      }
    }
    base::Check(found, "GGML freed a block the pool did not hand out");
    while (!blocks_.empty() && !blocks_.back().live) {
      blocks_.pop_back();
    }
    top_ = blocks_.empty() ? 0 : blocks_.back().end;
  }

  // Bounds the next operation's scratch.
  void Limit(std::uint64_t limit) {
    base::Check(limit <= size_, "an operation's scratch bound fits the workspace");
    limit_ = limit;
  }
  bool empty() const { return blocks_.empty(); }
  std::uint64_t peak() const { return peak_; }
  void ResetPeak() { peak_ = top_; }

 private:
  static constexpr std::uint64_t kAlign = 256;
  struct Block {
    std::uint64_t offset = 0;
    std::uint64_t end = 0;
    bool live = false;
  };

  std::uint64_t base_;
  std::uint64_t size_;
  std::uint64_t limit_ = 0;
  std::uint64_t top_ = 0;
  std::uint64_t peak_ = 0;
  std::vector<Block> blocks_;
};

namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

}  // namespace

// A concurrent lane (LaunchContext::ConfigureLanes): its stream, the event
// others wait on, and a GGML context over a pool of its own.
struct LaunchContext::Lane {
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  std::unique_ptr<WorkspacePool> pool;
  std::unique_ptr<ggml_backend_cuda_context> context;
};

std::expected<std::unique_ptr<LaunchContext>, KernelFailure> LaunchContext::Create(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    Workspace workspace, CublasHandle* cublas) {
  if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
    return Rejected("a launch context needs a device");
  }
  // Also makes the provider's context current, where the runtime binds.
  auto native = execution.Submission(stream);
  if (!native || native->handle == nullptr) {
    return Rejected("the provider refused the stream" +
                    (native ? std::string() : ": " + native.error().detail));
  }
  if (workspace.size.value() > 0 && (workspace.base == 0 || workspace.base % 256 != 0 ||
                                     workspace.base + workspace.size.value() < workspace.base)) {
    return Rejected("the workspace is not a 256-byte aligned device range");
  }
  // GGML's device table, read once for the process; its failures are
  // recorded like a launcher's. So is any error still pending on this
  // thread, which refuses the context too.
  const ggml_cuda_device_info& info = ggml_cuda_info();
  if (auto error = internal::TakeCudaError()) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail = "a CUDA error reading the device table, or pending before it: " + *error});
  }
  if (device >= info.device_count || info.devices[device].warp_size == 0) {
    return Rejected(std::format("GGML sees no device {}", device));
  }
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess || current != device) {
    return Rejected(std::format("device {} is not current on this thread", device));
  }
  if (cublas != nullptr && (cublas->device() != device || cublas->stream() != stream ||
                            cublas->native_stream().handle != native->handle)) {
    return Rejected("the cuBLAS handle is bound to another device or stream");
  }
  if (cublas != nullptr) {
    // Both workspaces are written from the same stream's work.
    const Workspace other = cublas->workspace();
    if (workspace.size.value() > 0 && other.size.value() > 0 &&
        workspace.base < other.base + other.size.value() &&
        other.base < workspace.base + workspace.size.value()) {
      return Rejected("the scratch workspace overlaps the cuBLAS workspace");
    }
  }
  auto context = std::make_unique<ggml_backend_cuda_context>(device);
  auto pool = std::make_unique<WorkspacePool>(workspace.base, workspace.size.value());
  context->streams[device][0] = static_cast<cudaStream_t>(native->handle);
  context->pools[device][0].reset(pool.get());
  if (cublas != nullptr) {
    // GGML's cublas_handle() returns a handle it finds, and creates one
    // (with a workspace of its own) only when it finds none.
    context->cublas_handles[device][0] = cublas->native();
    ++cublas->borrowers_;
  }
  return std::unique_ptr<LaunchContext>(new LaunchContext(
      device, execution, stream, *native, std::move(context), std::move(pool), workspace, cublas));
}

LaunchContext::LaunchContext(int device, providers::DeviceExecution& execution,
                             providers::StreamId stream, providers::NativeStream native,
                             std::unique_ptr<ggml_backend_cuda_context> context,
                             std::unique_ptr<WorkspacePool> pool, Workspace workspace,
                             CublasHandle* cublas)
    : id_([] {
        static std::atomic<std::uint64_t> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
      }()),
      device_(device),
      execution_(execution),
      stream_(stream),
      native_(native),
      pool_(std::move(pool)),
      context_(std::move(context)),
      workspace_(workspace),
      cublas_(cublas) {}

LaunchContext::~LaunchContext() {
  if (capturing_) {
    // Never left capturing (Capture always ends it); ended here all the same.
    cudaGraph_t graph = nullptr;
    (void)cudaStreamEndCapture(static_cast<cudaStream_t>(native_.handle), &graph);
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
  }
  // Take back what was lent, so GGML's destructor finds nothing to destroy.
  context_->streams[device_][0] = nullptr;
  (void)context_->pools[device_][0].release();
  if (cublas_ != nullptr) {
    context_->cublas_handles[device_][0] = nullptr;
    --cublas_->borrowers_;
  }
  ReleaseLanes();
}

void LaunchContext::ReleaseLanes() {
  for (const std::unique_ptr<Lane>& lane : lanes_) {
    // Take back the stream and pool lent to the lane's GGML context, so its
    // destructor (with the lane, below) finds nothing of ours to destroy.
    lane->context->streams[device_][0] = nullptr;
    (void)lane->context->pools[device_][0].release();
    // Work in flight completes first; CUDA frees the stream after it.
    (void)cudaEventDestroy(lane->event);
    (void)cudaStreamDestroy(lane->stream);
  }
  lanes_.clear();
  if (stream_event_ != nullptr) {
    (void)cudaEventDestroy(static_cast<cudaEvent_t>(stream_event_));
    stream_event_ = nullptr;
  }
  lane_scratch_ = 0;
  active_ = 0;
}

std::expected<void, KernelFailure> LaunchContext::ConfigureLanes(std::uint32_t count,
                                                                 base::Bytes lane_scratch) {
  if (!lanes_.empty() || capturing_ || faulted_) {
    return Rejected("lanes are made once, outside a capture, on a usable context");
  }
  const std::uint64_t each = (lane_scratch.value() + 255) / 256 * 256;
  if (count == 0 || each * count > workspace_.size.value()) {
    return Rejected(std::format("{} lanes of {} bytes of scratch exceed the workspace ({} bytes)",
                                count, each, workspace_.size.value()));
  }
  cudaEvent_t own = nullptr;
  if (cudaEventCreateWithFlags(&own, cudaEventDisableTiming) != cudaSuccess) {
    (void)cudaGetLastError();
    return Rejected("an event for the lanes");
  }
  stream_event_ = own;
  const std::uint64_t start = workspace_.size.value() - (each * count);
  for (std::uint32_t i = 0; i < count; ++i) {
    auto lane = std::make_unique<Lane>();
    if (cudaStreamCreateWithFlags(&lane->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&lane->event, cudaEventDisableTiming) != cudaSuccess) {
      (void)cudaGetLastError();
      // This lane has no GGML context yet; the ones made are released as
      // the destructor releases them (each once).
      if (lane->stream != nullptr) {
        (void)cudaStreamDestroy(lane->stream);
      }
      ReleaseLanes();
      return Rejected("a stream or event for a lane");
    }
    lane->pool = std::make_unique<WorkspacePool>(workspace_.base + start + (i * each), each);
    lane->context = std::make_unique<ggml_backend_cuda_context>(device_);
    lane->context->streams[device_][0] = lane->stream;
    lane->context->pools[device_][0].reset(lane->pool.get());
    lanes_.push_back(std::move(lane));
  }
  lane_scratch_ = each;
  return {};
}

void LaunchContext::SelectLane(std::uint32_t lane) {
  base::Check(lane <= lanes_.size(), "a lane the context has");
  active_ = lane;
}

void* LaunchContext::StreamOf(std::uint32_t lane) const {
  return lane == 0 ? native_.handle : lanes_[lane - 1]->stream;
}

std::expected<void, KernelFailure> LaunchContext::LaneWait(std::uint32_t waiter, std::uint32_t on) {
  if (waiter > lanes_.size() || on > lanes_.size() || waiter == on) {
    return Rejected("a lane wait between two of the context's lanes");
  }
  auto* const event = static_cast<cudaEvent_t>(on == 0 ? stream_event_ : lanes_[on - 1]->event);
  if (cudaEventRecord(event, static_cast<cudaStream_t>(StreamOf(on))) != cudaSuccess ||
      cudaStreamWaitEvent(static_cast<cudaStream_t>(StreamOf(waiter)), event, 0) != cudaSuccess) {
    const cudaError_t error = cudaGetLastError();
    faulted_ = true;
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown,
                      .detail = std::string("a lane's event: ") + cudaGetErrorString(error)});
  }
  return {};
}

base::Bytes LaunchContext::scratch_size(std::uint32_t lane) const {
  if (lane != 0) {
    return base::Bytes(lane <= lanes_.size() ? lane_scratch_ : 0);
  }
  return base::Bytes(workspace_.size.value() - (lane_scratch_ * lanes_.size()));
}

ggml_backend_cuda_context* LaunchContext::ActiveContext() {
  return active_ == 0 ? context_.get() : lanes_[active_ - 1]->context.get();
}

WorkspacePool& LaunchContext::ActivePool() {
  return active_ == 0 ? *pool_ : *lanes_[active_ - 1]->pool;
}

std::expected<void, KernelFailure> LaunchContext::Begin(base::Bytes scratch) {
  if (faulted_) {
    return Rejected("the launch context faulted earlier; its stream awaits recovery");
  }
  if (scratch > scratch_size(active_)) {
    return Rejected(std::format("the operation needs {} bytes of scratch; lane {}'s pool has {}",
                                scratch.value(), active_, scratch_size(active_).value()));
  }
  // The run is queued work on the stream, which the provider must know of
  // before any launch (launch.h).
  auto native = execution_.Submission(stream_);
  if (!native) {
    if (native.error().error == providers::ProviderError::kUnknown) {
      // A device fault: the stream's state is undetermined, so the context
      // faults and waits for recovery like after a launch error.
      faulted_ = true;
      return std::unexpected(
          KernelFailure{.error = KernelError::kUnknown,
                        .detail = "the stream faulted: " + native.error().detail});
    }
    return Rejected("the provider refused the stream: " + native.error().detail);
  }
  base::Check(native->handle == native_.handle, "a stream keeps its native handle");
  // GGML checks each launch with cudaGetLastError, so an error left by
  // another runtime call on this thread would be taken for the launch's.
  if (const cudaError_t stale = cudaGetLastError(); stale != cudaSuccess) {
    faulted_ = true;
    return std::unexpected(KernelFailure{
        .error = KernelError::kUnknown,
        .detail = std::string("a CUDA error before the launch: ") + cudaGetErrorString(stale)});
  }
  if (auto stray = internal::TakeCudaError()) {
    // Recorded outside a run: nothing of ours is known to be at fault.
    faulted_ = true;
    return std::unexpected(KernelFailure{.error = KernelError::kUnknown,
                                         .detail = "a CUDA error outside a launch: " + *stray});
  }
  ActivePool().Limit(scratch.value());
  return {};
}

std::expected<void, KernelFailure> LaunchContext::End() {
  base::Check(ActivePool().empty(), "GGML launchers return their scratch before they return");
  // Some launchers queue kernels without checking the launch (MMF's), so
  // the runtime's record, clear since Begin, is read too. Reading it
  // clears it, so it is not taken for a later launch's; a sticky error
  // stays.
  const cudaError_t unchecked = cudaGetLastError();
  auto error = internal::TakeCudaError();
  if (!error && unchecked != cudaSuccess) {
    error = std::string("an unchecked kernel launch failed: ") + cudaGetErrorString(unchecked);
  }
  // A lane lends no cuBLAS handle; one GGML made there would queue
  // outside the run's order and hold memory nothing counts.
  if (!error && active_ != 0 &&
      lanes_[active_ - 1]->context->cublas_handles[device_][0] != nullptr) {
    error = std::string("an operation on a lane created a cuBLAS handle");
  }
  if (error) {
    faulted_ = true;
    return std::unexpected(KernelFailure{.error = KernelError::kUnknown, .detail = *error});
  }
  return {};
}

// ------------------------------------------------------------------ graphs

CapturedGraph::CapturedGraph(CapturedGraph&& other) noexcept
    : owner_(std::exchange(other.owner_, 0)),
      exec_(std::exchange(other.exec_, nullptr)),
      nodes_(other.nodes_),
      capture_seconds_(other.capture_seconds_),
      instantiate_seconds_(other.instantiate_seconds_) {}

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& other) noexcept {
  if (this != &other) {
    if (exec_ != nullptr) {
      (void)cudaGraphExecDestroy(exec_);
    }
    owner_ = std::exchange(other.owner_, 0);
    exec_ = std::exchange(other.exec_, nullptr);
    nodes_ = other.nodes_;
    capture_seconds_ = other.capture_seconds_;
    instantiate_seconds_ = other.instantiate_seconds_;
  }
  return *this;
}

CapturedGraph::~CapturedGraph() {
  if (exec_ != nullptr) {
    // A replay still in flight completes first: CUDA frees the executable
    // graph once it has.
    (void)cudaGraphExecDestroy(exec_);
  }
}

namespace {

std::int64_t Now() { return std::chrono::steady_clock::now().time_since_epoch().count(); }

double Since(std::int64_t ticks) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch() -
                                       std::chrono::steady_clock::duration(ticks))
      .count();
}

}  // namespace

std::expected<void, KernelFailure> LaunchContext::BeginCapture() {
  if (capturing_) {
    return Rejected("the context is already capturing");
  }
  // As a run begins: the context usable, the stream counted as queued
  // work (an upload follows a capture), no error pending.
  if (auto begun = Begin(base::Bytes(0)); !begun) {
    return begun;
  }
  auto* const stream = static_cast<cudaStream_t>(native_.handle);
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  if (const cudaError_t queried = cudaStreamIsCapturing(stream, &status);
      queried != cudaSuccess || status != cudaStreamCaptureStatusNone) {
    (void)cudaGetLastError();
    return Rejected("the stream is being captured, or its capture state cannot be read");
  }
  // Thread-local: a call on this thread that a capture cannot hold (a
  // synchronization, an allocation) fails and invalidates the capture
  // instead of running; other threads' calls are unaffected.
  if (const cudaError_t begun = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
      begun != cudaSuccess) {
    (void)cudaGetLastError();
    return Rejected(std::string("cudaStreamBeginCapture: ") + cudaGetErrorString(begun));
  }
  capturing_ = true;
  capture_started_ = Now();
  return {};
}

std::expected<CapturedGraph, KernelFailure> LaunchContext::EndCapture(
    std::expected<void, KernelFailure> recorded) {
  auto* const stream = static_cast<cudaStream_t>(native_.handle);
  cudaGraph_t graph = nullptr;
  const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
  capturing_ = false;
  const double captured = Since(capture_started_);
  // What a launcher recorded during the capture, and the runtime's record,
  // belong to the capture: taken so that no later run inherits them.
  const std::optional<std::string> launcher = internal::TakeCudaError();
  const cudaError_t runtime = cudaGetLastError();
  if (!recorded || ended != cudaSuccess || graph == nullptr || launcher || runtime != cudaSuccess) {
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    // Nothing captured ran. Only a device fault, which the provider or the
    // stream now reports, makes the context's state unknown.
    if (auto native = execution_.Submission(stream_);
        !native && native.error().error == providers::ProviderError::kUnknown) {
      faulted_ = true;
      return std::unexpected(
          KernelFailure{.error = KernelError::kUnknown,
                        .detail = "the stream faulted during a capture: " + native.error().detail});
    }
    const cudaError_t probe = cudaStreamQuery(stream);
    (void)cudaGetLastError();
    if (probe != cudaSuccess && probe != cudaErrorNotReady) {
      faulted_ = true;
      return std::unexpected(KernelFailure{
          .error = KernelError::kUnknown,
          .detail =
              std::string("the stream faulted during a capture: ") + cudaGetErrorString(probe)});
    }
    faulted_ = false;  // a run refused during the capture queued nothing
    std::string why;
    if (!recorded) {
      why = recorded.error().detail;
    } else if (launcher) {
      why = *launcher;
    } else if (ended != cudaSuccess) {
      why = std::string("cudaStreamEndCapture: ") + cudaGetErrorString(ended);
    } else {
      why = std::string("a call the capture could not hold: ") + cudaGetErrorString(runtime);
    }
    return Rejected("the capture failed: " + why);
  }
  const std::int64_t instantiating = Now();
  std::size_t nodes = 0;
  (void)cudaGraphGetNodes(graph, nullptr, &nodes);
  cudaGraphExec_t exec = nullptr;
  const cudaError_t instantiated = cudaGraphInstantiate(&exec, graph, 0);
  (void)cudaGraphDestroy(graph);  // the executable graph keeps what it needs
  if (instantiated != cudaSuccess || exec == nullptr) {
    (void)cudaGetLastError();
    return Rejected(std::string("cudaGraphInstantiate: ") + cudaGetErrorString(instantiated));
  }
  // Uploaded now, so that the first replay costs what every other does.
  if (const cudaError_t uploaded = cudaGraphUpload(exec, stream); uploaded != cudaSuccess) {
    (void)cudaGetLastError();
    (void)cudaGraphExecDestroy(exec);
    faulted_ = true;
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown,
                      .detail = std::string("cudaGraphUpload: ") + cudaGetErrorString(uploaded)});
  }
  return CapturedGraph(id_, exec, nodes, captured, Since(instantiating));
}

std::expected<void, KernelFailure> LaunchContext::Launch(const CapturedGraph& graph) {
  if (graph.owner_ != id_ || graph.exec_ == nullptr) {
    return Rejected("the graph was captured by another launch context, or moved from");
  }
  if (capturing_) {
    return Rejected("a graph replay while capturing");
  }
  if (auto begun = Begin(base::Bytes(0)); !begun) {
    return begun;
  }
  if (const cudaError_t launched =
          cudaGraphLaunch(graph.exec_, static_cast<cudaStream_t>(native_.handle));
      launched != cudaSuccess) {
    (void)cudaGetLastError();
    faulted_ = true;
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown,
                      .detail = std::string("cudaGraphLaunch: ") + cudaGetErrorString(launched)});
  }
  return End();
}

base::Bytes LaunchContext::scratch_peak() const { return base::Bytes(pool_->peak()); }

void LaunchContext::ResetScratchPeak() { pool_->ResetPeak(); }

}  // namespace llmp::kernels::ggml
