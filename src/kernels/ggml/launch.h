// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The K-C launch context (D-053; docs/backend-proof.md#dispatch-and-implementations-d-053):
// GGML's CUDA operation launchers run with a ggml_backend_cuda_context that
// jitLLM fills with the stream to launch on and a scratch pool over
// workspace the caller declared and charged. GGML creates nothing of its
// own: no stream, pool, cuBLAS handle, workspace or process-wide setting
// (ggml_support.cu), except that soft_max, the quantized tile products
// (MMQ) and the tensor-core flash attention raise the dynamic shared memory
// limit of their own kernels, once per kernel and device
// (cudaFuncSetAttribute; softmax.cu, mmq.cuh, fattn-mma-f16.cuh; ops_ext.h's
// flash-attention plan raises it too, as the launcher will, before it
// queries the kernel's occupancy). It keeps only unlocked host-side caches:
// which kernels may use programmatic dependent launch, and which of those
// limits it has raised. A context may also lend GGML a cuBLAS
// handle jitLLM created for the same stream (cublas.h); without one, an
// operation that needs cuBLAS is refused. Run checks an operation's scratch bound against the
// workspace, calls its launchers, and reports the first CUDA error they
// recorded instead of aborting.
//
// A launch only queues work; completion is the caller's, through a fence
// after the last launch. The pool reuses workspace offsets in stream order
// as GGML frees them when a host launcher returns, so the workspace must
// stay mapped and charged until that fence has completed
// (docs/async-model.md). After an error, what was queued is undetermined:
// the context refuses further runs, and its stream and workspace wait for
// recovery. One context per stream, used on the device submission lane
// only. Each run takes the stream again through DeviceExecution::Submission,
// which counts it as queued work, so the provider refuses to destroy the
// stream until a fence after the run has completed and been released; the
// context must not outlive the provider or the stream.
//
// Captured graphs (D-090). Capture records, instead of queuing, what runs
// and plain stream copies queue on the context's stream, as one CUDA graph
// (cudaStreamBeginCapture in thread-local mode, so this thread's calls a
// capture cannot hold fail instead of running); Launch queues one replay,
// one operation on the stream (RE-029). A replay launches the captured
// kernels with the captured parameters: every address they hold (operands,
// the pool's blocks, which the pool hands out in the same order from the
// same empty stack each run, and the workspace) must still be mapped at the
// same place and hold what it held, which is the caller's to guarantee
// (D-090: pinned places), and whatever varies between replays must be data
// the graph reads, never a launch parameter. A capture queues nothing, so a
// refused one leaves the context as it was (kRejected), unless the stream
// reports a device fault (kUnknown, which faults the context). A graph
// replays only on the context that captured it, and must be destroyed before
// the workspace or any memory it names is unmapped.

#ifndef JITLLM_KERNELS_GGML_LAUNCH_H_
#define JITLLM_KERNELS_GGML_LAUNCH_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "kernels/ggml/tensors.h"
#include "providers/device_execution.h"

struct ggml_backend_cuda_context;
struct CUgraphExec_st;  // cudaGraphExec_t's

namespace jitllm::kernels::ggml {

class CublasHandle;
class WorkspacePool;
class LaunchContext;

// An instantiated graph a context captured (LaunchContext::Capture). Move
// only; destroying it frees the executable graph (after a replay in flight
// completes, as CUDA does).
class CapturedGraph {
 public:
  CapturedGraph(CapturedGraph&& other) noexcept;
  CapturedGraph& operator=(CapturedGraph&& other) noexcept;
  CapturedGraph(const CapturedGraph&) = delete;
  CapturedGraph& operator=(const CapturedGraph&) = delete;
  ~CapturedGraph();

  // The captured graph's nodes: kernels, copies and memsets.
  std::size_t nodes() const { return nodes_; }
  // Host time capturing it, and instantiating and uploading it.
  double capture_seconds() const { return capture_seconds_; }
  double instantiate_seconds() const { return instantiate_seconds_; }

 private:
  friend class LaunchContext;
  CapturedGraph(std::uint64_t owner, CUgraphExec_st* exec, std::size_t nodes,
                double capture_seconds, double instantiate_seconds)
      : owner_(owner),
        exec_(exec),
        nodes_(nodes),
        capture_seconds_(capture_seconds),
        instantiate_seconds_(instantiate_seconds) {}

  // The capturing context's identity, never reused in the process (not
  // its address, which a later context may take).
  std::uint64_t owner_ = 0;
  CUgraphExec_st* exec_ = nullptr;
  std::size_t nodes_ = 0;
  double capture_seconds_ = 0;
  double instantiate_seconds_ = 0;
};

class LaunchContext {
 public:
  // A device range the pool hands out: mapped with access and charged by
  // the caller for as long as the context's launches can use it.
  struct Workspace {
    std::uint64_t base = 0;
    base::Bytes size;
  };

  // Launches on `stream`, a CUDA provider's stream on `device`, lending
  // GGML `cublas` if given: a handle for the same device and stream, which
  // must outlive the context.
  static std::expected<std::unique_ptr<LaunchContext>, KernelFailure> Create(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      Workspace workspace, CublasHandle* cublas = nullptr);

  LaunchContext(const LaunchContext&) = delete;
  LaunchContext& operator=(const LaunchContext&) = delete;
  LaunchContext(LaunchContext&&) = delete;
  LaunchContext& operator=(LaunchContext&&) = delete;
  ~LaunchContext();

  // Calls launch(context) for GGML launchers that together draw at most
  // `scratch` from the pool at once, counting each block the pool hands out
  // from a 256-byte boundary (so the bound includes the rounding). Refused, with nothing queued, if
  // the workspace is smaller, the provider refuses the stream or the context is faulted; kUnknown
  // if a launcher recorded a CUDA error, which faults the context.
  template <typename Launch>
  std::expected<void, KernelFailure> Run(base::Bytes scratch, Launch&& launch) {
    if (auto begun = Begin(scratch); !begun) {
      return begun;
    }
    std::forward<Launch>(launch)(*ActiveContext());
    return End();
  }

  // Concurrent lanes (graph_plan.h AssignLanes; executor.h BoundGraph::Run):
  // `count` streams of the context's own, each with a GGML context and a
  // scratch pool of `lane_scratch` bytes carved from the workspace's end,
  // which the stream's pool then no longer uses. A lane borrows no cuBLAS
  // handle. Made once, outside any capture; refused if the workspace has
  // no room for them. A lane's work belongs to the stream's run or capture
  // only through LaneWait: the caller makes each lane wait for the stream
  // before its first launch and the stream wait for each lane before the
  // run ends (BoundGraph::Run does), so the stream's fence covers it.
  std::expected<void, KernelFailure> ConfigureLanes(std::uint32_t count, base::Bytes lane_scratch);
  std::uint32_t lanes() const { return static_cast<std::uint32_t>(lanes_.size()); }
  // Where the next Run launches: 0, the context's stream, or a lane.
  void SelectLane(std::uint32_t lane);
  // Makes `waiter`'s stream wait for what `on`'s stream has queued so far
  // (0 is the context's stream).
  std::expected<void, KernelFailure> LaneWait(std::uint32_t waiter, std::uint32_t on);
  // The scratch a run on `lane` may draw.
  base::Bytes scratch_size(std::uint32_t lane) const;

  // Captures what `record(*this)` queues on the context's stream (its runs,
  // and copies the caller queues on the same stream) into a graph, which it
  // instantiates and uploads; `record` returns std::expected<void,
  // KernelFailure>. Nothing it records runs. Refused (kRejected), with the
  // context as it was, if the context is faulted or already capturing, the
  // stream is being captured, or the capture fails: `record` fails, or a call
  // it makes cannot be captured. kUnknown, faulting the context, if the
  // stream reports a device fault or the upload fails.
  template <typename Record>
  std::expected<CapturedGraph, KernelFailure> Capture(Record&& record) {
    if (auto begun = BeginCapture(); !begun) {
      return std::unexpected(begun.error());
    }
    std::expected<void, KernelFailure> recorded = std::forward<Record>(record)(*this);
    return EndCapture(std::move(recorded));
  }

  // Queues one replay of `graph`, which this context captured: one
  // operation on the stream. Refused, with nothing queued, if another
  // context captured it or the context is faulted; kUnknown if the launch
  // reports an error, which faults the context.
  std::expected<void, KernelFailure> Launch(const CapturedGraph& graph);
  bool capturing() const { return capturing_; }

  // The most scratch any run has held at once, since the context was made
  // or the peak was last reset.
  base::Bytes scratch_peak() const;
  // Between runs: the next peak counts only the runs after this.
  void ResetScratchPeak();
  bool faulted() const { return faulted_; }
  // Borrowed-operation adapters must fence the provider stream used here,
  // not another stream that happens to have the same numerical ID.
  bool UsesStream(const providers::DeviceExecution& execution, providers::StreamId stream) const {
    return &execution_ == &execution && stream_ == stream;
  }
  int device() const { return device_; }
  Workspace workspace() const { return workspace_; }
  // The lent cuBLAS handle, if any, while the stream is selected: a lane
  // has none (its stream is not the handle's), so an operation needing one
  // is refused there (implementations.h UsesCublas).
  const CublasHandle* cublas() const { return active_ == 0 ? cublas_ : nullptr; }

 private:
  LaunchContext(int device, providers::DeviceExecution& execution, providers::StreamId stream,
                providers::NativeStream native, std::unique_ptr<ggml_backend_cuda_context> context,
                std::unique_ptr<WorkspacePool> pool, Workspace workspace, CublasHandle* cublas);

  std::expected<void, KernelFailure> Begin(base::Bytes scratch);
  std::expected<void, KernelFailure> End();
  // Destroys the lanes' streams and events and their GGML contexts, each
  // once, what was lent to those taken back first (the destructor, and
  // ConfigureLanes after a partial failure).
  void ReleaseLanes();
  ggml_backend_cuda_context* ActiveContext();
  WorkspacePool& ActivePool();
  void* StreamOf(std::uint32_t lane) const;

  struct Lane;
  std::expected<void, KernelFailure> BeginCapture();
  std::expected<CapturedGraph, KernelFailure> EndCapture(
      std::expected<void, KernelFailure> recorded);

  std::uint64_t id_;  // what its graphs record as their owner
  int device_;
  providers::DeviceExecution& execution_;
  providers::StreamId stream_;
  providers::NativeStream native_;
  std::unique_ptr<WorkspacePool> pool_;
  // Declared after the pool: destroyed first, once it has handed the pool
  // and stream back.
  std::unique_ptr<ggml_backend_cuda_context> context_;
  Workspace workspace_;
  CublasHandle* cublas_;
  std::vector<std::unique_ptr<Lane>> lanes_;
  std::uint32_t active_ = 0;
  std::uint64_t lane_scratch_ = 0;
  void* stream_event_ = nullptr;  // cudaEvent_t: the stream's, for LaneWait
  bool faulted_ = false;
  bool capturing_ = false;
  std::int64_t capture_started_ = 0;  // steady_clock ticks, while capturing
};

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_LAUNCH_H_
