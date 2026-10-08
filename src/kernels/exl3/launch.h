// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Llmpalooza's launchers of ExLlamaV3's kernels (K-L, D-053;
// docs/backend-proof.md, "EXL3-derived operations"), CUDA builds only. They
// replace upstream's ATen host wrappers: each checks its operands and plan
// on the host (validate.h), then launches the locked kernel
// (llmp_exl3_kernels.h) on a llmpalooza stream exactly as upstream's wrapper
// would at that plan, and returns a launch error as a fault instead of
// ending the process.
//
// A context holds what launch_contract.h requires of every launch:
// - the stream, a CUDA provider's (DeviceExecution::Submission), which
//   counts each launch as queued work, so the provider refuses to destroy
//   the stream until a fence after it has completed and been released;
// - a lock area the caller declared and charged (kLockBytes, the size of
//   upstream's per-device area), which the context zeroes on its stream
//   before its first launch, as upstream zeroes its own on first use. No two
//   live contexts may share lock slots: Create refuses an area overlapping
//   another live context's, and one context's launches run in its stream's
//   order. The registry knows only live contexts: a lock area passes to a
//   context on another stream only once a fence after the previous
//   holder's last launch has completed, or the new context's zeroing
//   races those launches. That is the rule for every operand whose memory
//   passes between streams (docs/async-model.md: retirement after a
//   completion proof), and it is kept where memory is reassigned, not
//   here: the registry cannot tell a lock area handed to another stream
//   from one freed after its fence and allocated again at the same
//   address, so refusing reuse here would refuse correct callers. A
//   launch that completes returns its slots to zero; one that faults
//   leaves them undetermined, so the context refuses every later launch,
//   and recovery zeroes the area again in a new context;
// - the co-resident block limit of each cooperative kernel on the device
//   (occupancy per SM times SMs), which bounds every cooperative grid
//   before launch. It is the whole device's, not a share of it: two
//   contexts on two streams launching grids at that limit back to back
//   both complete on GB10 (unit.Exl3LinearTest.
//   TwoContextsAtTheCoresidentLimitBothComplete), consistent with the
//   device admitting each cooperative grid whole. CUDA documents
//   co-residency per launch, not across streams, so that is a measured
//   behaviour of this driver and device, not a documented guarantee.
// The multi-GEMM kernel's selection state (launch_contract.h, rule 2) is
// never used: MultiGemm launches without expert indices, weights, ranges
// or sliced sources, where the kernel reads none of it.
//
// A launch only queues work; completion is the caller's, through a fence
// after the last launch. The lock area and every operand stay mapped and
// charged until that fence has completed (docs/async-model.md), including
// after the context is destroyed. One context per stream, used on the
// device submission lane only; the context must not outlive the provider
// or the stream.

#ifndef LLMP_KERNELS_EXL3_LAUNCH_H_
#define LLMP_KERNELS_EXL3_LAUNCH_H_

#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <optional>

#include "kernels/exl3/launch_contract.h"
#include "kernels/exl3/upstream_gemv.h"
#include "kernels/exl3/validate.h"
#include "providers/device_execution.h"

namespace llmp::kernels::exl3 {

class LaunchContext {
 public:
  // Launches on `stream`, a CUDA provider's stream on `device`, which must
  // be current on the calling thread, with the lock slots at `locks`
  // (kLockBytes, 256-byte aligned, mapped with access). Queues the zeroing
  // of the lock area.
  static std::expected<std::unique_ptr<LaunchContext>, KernelFailure> Create(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      std::uint64_t locks);

  LaunchContext(const LaunchContext&) = delete;
  LaunchContext& operator=(const LaunchContext&) = delete;
  LaunchContext(LaunchContext&&) = delete;
  LaunchContext& operator=(LaunchContext&&) = delete;
  ~LaunchContext();

  // exl3_gemm_kernel at `plan`: grid (blocks), kBlockDim[shape] threads,
  // kGemmSharedMemory bytes, cooperatively (exl3_gemm.cu's forced and tuned
  // launches alike).
  std::expected<void, KernelFailure> Gemm(const LinearOperands& operands, const GemmPlan& plan);
  // exl3_gemv_kernel at `plan`, cooperatively (exl3_gemv_try_launch):
  // mode 0 for one row, 1 for up to eight.
  std::expected<void, KernelFailure> Gemv(const LinearOperands& operands, const GemvPlan& plan);
  // exl3_mgemm_kernel at `plan`, cooperatively, as upstream's gated MLP
  // calls exl3_mgemm for gate and up: one input, two outputs, no indices,
  // weights, expert range or slices.
  std::expected<void, KernelFailure> MultiGemm(const MultiLinearOperands& operands,
                                               const MultiGemmPlan& plan);
  // reconstruct_kernel (grid columns / 128 × k / 16, 256 threads) or
  // reconstruct_had_kernel (columns / 128 × k / 128, 256 threads), as
  // reconstruct_slice and reconstruct_had_slice launch them.
  std::expected<void, KernelFailure> Reconstruct(const ReconstructOperands& operands);
  // had_{hf,ff}_r_128_kernel (grid rows × columns / 128, 32 threads), as
  // had_r_128 launches it with scale 1.
  std::expected<void, KernelFailure> Hadamard(const HadamardOperands& operands);
  // add_kernel_hhh (grid ceil(rows × columns / 1024), 1024 threads), as
  // add_gr launches it.
  std::expected<void, KernelFailure> Bias(const BiasOperands& operands);

  // Blocks of the kernel that fit on the device at once.
  std::expected<int, KernelFailure> GemmCoresident(int bits, int shape, Output output);
  std::expected<int, KernelFailure> MultiGemmCoresident(int bits, int shape, Output output);
  std::expected<int, KernelFailure> GemvCoresident(int bits, Output output, int m, int config);
  // Where upstream's EXL3-O profile would launch the GEMV (upstream_gemv.h),
  // with this device's class and co-resident limits.
  std::expected<std::optional<GemvPlan>, KernelFailure> UpstreamGemv(const Weights& weights,
                                                                     Output output, int m);

  // Runs `launch(stream)` for work of the caller's (the reconstruction
  // GEMM, lt.h) with the same stream accounting and fault handling as the
  // launchers: refused if the context is faulted or the provider refuses
  // the stream; a failure `launch` reports, or a CUDA error it leaves,
  // faults the context.
  template <typename Launch>
  std::expected<void, KernelFailure> Run(Launch&& launch) {
    auto stream = Begin();
    if (!stream) {
      return std::unexpected(stream.error());
    }
    return End(launch(*stream));
  }

  int device() const { return device_; }
  int sm_count() const { return sm_count_; }
  CcClass cc_class() const { return cc_class_; }
  std::uint64_t locks() const { return locks_; }
  bool faulted() const { return faulted_; }

 private:
  LaunchContext(int device, providers::DeviceExecution& execution, providers::StreamId stream,
                providers::NativeStream native, std::uint64_t locks, int sm_count,
                CcClass cc_class);

  std::expected<void*, KernelFailure> Begin();
  // `launched` is the launch call's own result.
  std::expected<void, KernelFailure> End(std::expected<void, KernelFailure> launched);
  std::expected<int, KernelFailure> Coresident(const void* kernel, int threads, int shared);

  int device_;
  providers::DeviceExecution& execution_;
  providers::StreamId stream_;
  providers::NativeStream native_;
  std::uint64_t locks_;
  int sm_count_;
  CcClass cc_class_;
  bool faulted_ = false;
  std::map<const void*, int> coresident_;
};

}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_LAUNCH_H_
