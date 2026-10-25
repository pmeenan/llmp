// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "common.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/scratch_stack.h"

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
  // For nonzero allocations, successive stack entries start at distinct
  // 256-byte boundaries. Dead interior entries have distinct starts too.
  // Private run metadata is therefore allocated/paid before any submission.
  static std::uint64_t Blocks(std::uint64_t bytes) { return ScratchStack::Capacity(bytes); }
  static std::uint64_t DescriptorBytes(std::uint64_t bytes) {
    return Blocks(bytes) * sizeof(ScratchStack::Block);
  }
  void MakePrivate() {
    capacity_ = Blocks(size_);
    if (capacity_ != 0)
      fixed_ = std::make_unique_for_overwrite<ScratchStack::Block[]>(
          static_cast<std::size_t>(capacity_));
    stack_ = ScratchStack(std::span(fixed_.get(), static_cast<std::size_t>(capacity_)));
    bounded_ = true;
  }
  void* alloc(std::size_t size, std::size_t* actual_size) override {
    *actual_size = size;
    // No descriptor for zero bytes: freeing any such view is a no-op, even
    // when nested with an allocation at the same offset.
    if (size == 0) return reinterpret_cast<void*>(base_);
    if (bounded_) {
      auto offset = stack_.Allocate(size, limit_);
      base::Check(offset.has_value(), "private scratch fits its paid backing and descriptors");
      return reinterpret_cast<void*>(base_ + *offset);
    }
    const std::uint64_t offset = (top_ + kAlign - 1) / kAlign * kAlign;
    base::Check(offset <= limit_ && size <= limit_ - offset,
                "a GGML launcher drew more scratch than its operation's bound");
    blocks_.push_back(Block{.offset = offset, .end = offset + size, .live = true});
    top_ = offset + size;
    peak_ = std::max(peak_, top_);
    return reinterpret_cast<void*>(base_ + offset);
  }
  void free(void* ptr, std::size_t size) override {
    if (size == 0) return;
    const std::uint64_t offset = reinterpret_cast<std::uintptr_t>(ptr) - base_;
    if (bounded_) {
      base::Check(stack_.Free(offset, size), "private scratch frees its own live allocation once");
      return;
    }
    bool found = false;
    for (Block& block : blocks_) {
      if (block.live && block.offset == offset && block.end - block.offset == size) {
        block.live = false;
        found = true;
        break;
      }
    }
    base::Check(found, "GGML freed a block the pool did not hand out");
    while (!blocks_.empty() && !blocks_.back().live) blocks_.pop_back();
    top_ = blocks_.empty() ? 0 : blocks_.back().end;
  }
  void Limit(std::uint64_t limit) {
    base::Check(limit <= size_, "an operation's scratch bound fits the workspace");
    limit_ = limit;
  }
  bool empty() const { return bounded_ ? stack_.empty() : blocks_.empty(); }
  std::uint64_t peak() const { return bounded_ ? stack_.peak() : peak_; }
  void ResetPeak() {
    if (bounded_)
      stack_.ResetPeak();
    else
      peak_ = top_;
  }

 private:
  static constexpr std::uint64_t kAlign = 256;
  struct Block {
    std::uint64_t offset = 0, end = 0;
    bool live = false;
  };
  std::uint64_t base_, size_, limit_ = 0, top_ = 0, peak_ = 0;
  std::vector<Block> blocks_;
  std::unique_ptr<ScratchStack::Block[]> fixed_;
  ScratchStack stack_;
  std::uint64_t capacity_ = 0;
  bool bounded_ = false;
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

static_assert(PerDeviceOnce::kDevices == GGML_CUDA_MAX_DEVICES);

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

std::expected<void, KernelFailure> LaunchContext::LendCublas(CublasHandle* cublas) {
  if (capturing_ || lane_source_ || active_ != 0 || !lanes_.empty() || !pool_->empty())
    return Rejected("the descriptor context is not idle for a library loan");
  if (cublas && (cublas->device() != device_ || cublas->stream() != stream_ ||
                 cublas->native_stream().handle != native_.handle))
    return Rejected("the cuBLAS handle is bound to another device or stream");
  if (cublas == cublas_) return {};
  if (cublas_) --cublas_->borrowers_;
  cublas_ = cublas;
  context_->cublas_handles[device_][0] = cublas ? cublas->native() : nullptr;
  if (cublas_) ++cublas_->borrowers_;
  return {};
}

std::expected<std::uint64_t, KernelFailure> LaunchContext::PrivateHostBytes(Workspace workspace) {
  constexpr std::uint64_t kAllocationMetadataBytes = 64;
  // Context, pool, their owner, and the actual fixed descriptor capacity.
  // The context's short device name uses its existing standard-library inline
  // representation. A separate named 64-byte name/control allowance also
  // covers a provider/library implementation using dynamic short-name storage.
  std::uint64_t bytes = sizeof(LaunchContext) + sizeof(ggml_backend_cuda_context) +
                        sizeof(WorkspacePool) + 4 * kAllocationMetadataBytes +
                        (workspace.size.value() ? kAllocationMetadataBytes : 0);
  std::uint64_t descriptors = 0;
  if (__builtin_mul_overflow(WorkspacePool::Blocks(workspace.size.value()),
                             std::uint64_t{WorkspacePool::DescriptorBytes(256)}, &descriptors) ||
      __builtin_add_overflow(bytes, descriptors, &bytes))
    return Rejected("private launch metadata size overflows");
  return bytes;
}

std::expected<std::unique_ptr<LaunchContext>, KernelFailure> LaunchContext::CreatePrivate(
    LaunchContext& parent, Workspace workspace) {
  if (parent.faulted() || parent.capturing_ || parent.lane_source_ ||
      workspace.base + workspace.size.value() < workspace.base ||
      (workspace.base < parent.workspace_.base + parent.workspace_.size.value() &&
       parent.workspace_.base < workspace.base + workspace.size.value()))
    return Rejected("private workspace needs a healthy original parent and distinct backing");
  auto made = Create(parent.device_, parent.execution_, parent.stream_, workspace, parent.cublas_);
  if (!made) return made;
  (*made)->pool_->MakePrivate();
  (*made)->lane_source_ = &parent;
  return made;
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
  if (lane_source_ || !lanes_.empty() || capturing_ || faulted_) {
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
  base::Check(lane <= lanes(), "a lane the context has");
  active_ = lane;
}

void* LaunchContext::StreamOf(std::uint32_t lane) const {
  return lane_source_ ? lane_source_->StreamOf(lane)
                      : (lane == 0 ? native_.handle : lanes_[lane - 1]->stream);
}

std::expected<void, KernelFailure> LaunchContext::LaneWait(std::uint32_t waiter, std::uint32_t on) {
  if (lane_source_) {
    auto waited = lane_source_->LaneWait(waiter, on);
    if (!waited && waited.error().error == KernelError::kUnknown) faulted_ = true;
    return waited;
  }
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
    if (lane_source_) return lane_source_->scratch_size(lane);
    return base::Bytes(lane <= lanes_.size() ? lane_scratch_ : 0);
  }
  return base::Bytes(workspace_.size.value() - (lane_scratch_ * lanes_.size()));
}

std::uint64_t LaunchContext::scratch_base() const {
  if (active_ == 0) {
    return workspace_.base;
  }
  if (lane_source_) {
    const auto& source = *lane_source_;
    return source.workspace_.base + source.workspace_.size.value() -
           (source.lane_scratch_ * source.lanes_.size()) + ((active_ - 1) * source.lane_scratch_);
  }
  // ConfigureLanes: lane i's pool follows lane i - 1's, from the workspace's
  // end less every lane's scratch.
  const std::uint64_t start = workspace_.size.value() - (lane_scratch_ * lanes_.size());
  return workspace_.base + start + ((active_ - 1) * lane_scratch_);
}

ggml_backend_cuda_context* LaunchContext::ActiveContext() {
  return active_ == 0 ? context_.get()
                      : (lane_source_ ? lane_source_->lanes_[active_ - 1]->context.get()
                                      : lanes_[active_ - 1]->context.get());
}

WorkspacePool& LaunchContext::ActivePool() {
  return active_ == 0 ? *pool_
                      : (lane_source_ ? *lane_source_->lanes_[active_ - 1]->pool
                                      : *lanes_[active_ - 1]->pool);
}

std::expected<void, KernelFailure> LaunchContext::Begin(base::Bytes scratch) {
  if (faulted()) {
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
  if (!error && active_ != 0 && ActiveContext()->cublas_handles[device_][0] != nullptr) {
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
    : start_event_(std::exchange(other.start_event_, nullptr)),
      stop_event_(std::exchange(other.stop_event_, nullptr)),
      timing_stream_(other.timing_stream_),
      timing_device_(other.timing_device_),
      timing_fault_(other.timing_fault_),
      submitted_(other.submitted_),
      launched_(other.launched_),
      timing_unknown_(other.timing_unknown_),
      owner_(std::exchange(other.owner_, 0)),
      exec_(std::exchange(other.exec_, nullptr)),
      nodes_(other.nodes_),
      capture_seconds_(other.capture_seconds_),
      instantiate_seconds_(other.instantiate_seconds_) {}

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& other) noexcept {
  if (this != &other) {
    Release();
    start_event_ = std::exchange(other.start_event_, nullptr);
    stop_event_ = std::exchange(other.stop_event_, nullptr);
    timing_stream_ = other.timing_stream_;
    timing_device_ = other.timing_device_;
    timing_fault_ = other.timing_fault_;
    submitted_ = other.submitted_;
    launched_ = other.launched_;
    timing_unknown_ = other.timing_unknown_;
    owner_ = std::exchange(other.owner_, 0);
    exec_ = std::exchange(other.exec_, nullptr);
    nodes_ = other.nodes_;
    capture_seconds_ = other.capture_seconds_;
    instantiate_seconds_ = other.instantiate_seconds_;
  }
  return *this;
}

void CapturedGraph::Release() {
  if (start_event_ && submitted_ && timing_fault_ && *timing_fault_) timing_unknown_ = true;
  if (start_event_ && submitted_ && !timing_unknown_) {
    // Instrumentation only: the owning stream outlives this graph (D-090).
    // Upload counts too, even when this graph has never been replayed.
    cudaError_t completed = cudaSetDevice(timing_device_);
    if (completed == cudaSuccess) {
      completed = cudaStreamQuery(static_cast<cudaStream_t>(timing_stream_));
      if (completed == cudaErrorNotReady) {
        (void)cudaGetLastError();
        completed = cudaStreamSynchronize(static_cast<cudaStream_t>(timing_stream_));
      }
    }
    timing_unknown_ = completed != cudaSuccess;
  }
  if (start_event_ && timing_unknown_) {
    // Quarantine on unknown completion: leave these native handles to CUDA
    // context teardown. Destroying an event named by pending work is unsafe.
    // The faulted context refuses further capture/replay; no retry accumulates
    // handles. This grants no retirement rights for plan memory or backing.
    if (timing_fault_) *timing_fault_ = true;
  } else {
    if (exec_) (void)cudaGraphExecDestroy(exec_);
    if (start_event_) (void)cudaEventDestroy(static_cast<cudaEvent_t>(start_event_));
    if (stop_event_) (void)cudaEventDestroy(static_cast<cudaEvent_t>(stop_event_));
  }
  exec_ = nullptr;
  start_event_ = nullptr;
  stop_event_ = nullptr;
}

CapturedGraph::~CapturedGraph() { Release(); }

std::expected<double, KernelFailure> CapturedGraph::gpu_duration_ns() const {
  if (!exec_ || !start_event_ || !launched_)
    return Rejected("capture duration requires an instrumented, successfully launched graph");
  if (timing_fault_ && *timing_fault_) timing_unknown_ = true;
  if (timing_unknown_)
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown, .detail = "capture completion is unknown"});
  // A previous replay's stop event may already be complete. Only current
  // stream-tail completion proves the latest launch has finished. Unrelated
  // work subsequently queued on this stream also delays this query.
  cudaError_t completed = cudaSetDevice(timing_device_);
  if (completed == cudaSuccess)
    completed = cudaStreamQuery(static_cast<cudaStream_t>(timing_stream_));
  if (completed == cudaErrorNotReady) {
    (void)cudaGetLastError();
    return Rejected("capture replay is still pending");
  }
  float milliseconds = 0;
  if (completed == cudaSuccess)
    completed = cudaEventElapsedTime(&milliseconds, static_cast<cudaEvent_t>(start_event_),
                                     static_cast<cudaEvent_t>(stop_event_));
  if (completed != cudaSuccess) {
    timing_unknown_ = true;
    if (timing_fault_) *timing_fault_ = true;
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown,
                      .detail = std::string("capture duration: ") + cudaGetErrorString(completed)});
  }
  const double nanoseconds = static_cast<double>(milliseconds) * 1'000'000.0;
  if (!std::isfinite(nanoseconds) || nanoseconds <= 0)
    return Rejected("capture duration is not finite and positive");
  return nanoseconds;
}

namespace {

std::int64_t Now() { return std::chrono::steady_clock::now().time_since_epoch().count(); }

double Since(std::int64_t ticks) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch() -
                                       std::chrono::steady_clock::duration(ticks))
      .count();
}

}  // namespace

std::expected<void, KernelFailure> LaunchContext::BeginCapture(bool measure_gpu_duration) {
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
  cudaEvent_t start = nullptr, stop = nullptr;
  if (measure_gpu_duration) {
    cudaError_t created = cudaEventCreate(&start);
    if (created == cudaSuccess) created = cudaEventCreate(&stop);
    if (created != cudaSuccess) {
      if (start) (void)cudaEventDestroy(start);
      if (stop) (void)cudaEventDestroy(stop);
      return Rejected(std::string("capture timing events: ") + cudaGetErrorString(created));
    }
  }
  // Thread-local: a call on this thread that a capture cannot hold (a
  // synchronization, an allocation) fails and invalidates the capture
  // instead of running; other threads' calls are unaffected.
  if (const cudaError_t begun = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
      begun != cudaSuccess) {
    (void)cudaGetLastError();
    if (start) (void)cudaEventDestroy(start);
    if (stop) (void)cudaEventDestroy(stop);
    return Rejected(std::string("cudaStreamBeginCapture: ") + cudaGetErrorString(begun));
  }
  capturing_ = true;
  capture_started_ = Now();
  capture_start_event_ = start;
  capture_stop_event_ = stop;
  if (start) {
    const auto recorded = cudaEventRecordWithFlags(start, stream, cudaEventRecordExternal);
    if (recorded != cudaSuccess) {
      auto ended = EndCapture(Rejected("capture timing start event failed"));
      return std::unexpected(ended.error());
    }
  }
  return {};
}

std::expected<CapturedGraph, KernelFailure> LaunchContext::EndCapture(
    std::expected<void, KernelFailure> recorded) {
  auto* const stream = static_cast<cudaStream_t>(native_.handle);
  CapturedGraph captured_graph(id_, nullptr, 0, 0, 0);
  captured_graph.start_event_ = std::exchange(capture_start_event_, nullptr);
  captured_graph.stop_event_ = std::exchange(capture_stop_event_, nullptr);
  captured_graph.timing_stream_ = stream;
  captured_graph.timing_device_ = device_;
  captured_graph.timing_fault_ = &faulted_;
  if (captured_graph.stop_event_) {
    const auto stopped = cudaEventRecordWithFlags(
        static_cast<cudaEvent_t>(captured_graph.stop_event_), stream, cudaEventRecordExternal);
    if (stopped != cudaSuccess && recorded) recorded = Rejected("capture timing stop event failed");
  }
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
  captured_graph.exec_ = exec;
  captured_graph.nodes_ = nodes;
  captured_graph.capture_seconds_ = captured;
  // Set before the API: a failed upload may still have submitted work.
  captured_graph.submitted_ = true;
  // Uploaded now, so that the first replay costs what every other does.
  if (const cudaError_t uploaded = cudaGraphUpload(exec, stream); uploaded != cudaSuccess) {
    (void)cudaGetLastError();
    if (captured_graph.start_event_) captured_graph.timing_unknown_ = true;
    faulted_ = true;
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown,
                      .detail = std::string("cudaGraphUpload: ") + cudaGetErrorString(uploaded)});
  }
  captured_graph.instantiate_seconds_ = Since(instantiating);
  return captured_graph;
}

std::expected<void, KernelFailure> LaunchContext::Launch(const CapturedGraph& graph) {
  if (graph.owner_ != id_ || graph.exec_ == nullptr) {
    return Rejected("the graph was captured by another launch context, or moved from");
  }
  if (capturing_) {
    return Rejected("a graph replay while capturing");
  }
  if (graph.start_event_ && graph.timing_unknown_) {
    faulted_ = true;
    return Rejected("instrumented graph completion is unknown");
  }
  if (auto begun = Begin(base::Bytes(0)); !begun) {
    return begun;
  }
  graph.submitted_ = true;
  if (const cudaError_t launched =
          cudaGraphLaunch(graph.exec_, static_cast<cudaStream_t>(native_.handle));
      launched != cudaSuccess) {
    (void)cudaGetLastError();
    if (graph.start_event_) graph.timing_unknown_ = true;
    faulted_ = true;
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown,
                      .detail = std::string("cudaGraphLaunch: ") + cudaGetErrorString(launched)});
  }
  auto ended = End();
  if (ended)
    graph.launched_ = true;
  else if (graph.start_event_)
    graph.timing_unknown_ = true;
  return ended;
}

base::Bytes LaunchContext::scratch_peak() const { return base::Bytes(pool_->peak()); }

void LaunchContext::ResetScratchPeak() { pool_->ResetPeak(); }

}  // namespace llmp::kernels::ggml
