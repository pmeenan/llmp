// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/exl3/launch.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "kernels/exl3/upstream_gemv.h"
#include "kernels/exl3/validate.h"
#include "llmp_exl3_kernels.h"

namespace llmp::kernels::exl3 {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

std::unexpected<KernelFailure> Fault(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kUnknown, .detail = std::move(detail)});
}

std::expected<void, KernelFailure> Launched(cudaError_t error) {
  if (error != cudaSuccess) {
    return Fault(std::string("the launch failed: ") + cudaGetErrorString(error));
  }
  return {};
}

// The lock areas of live contexts, which no other context may share
// (launch_contract.h, rule 2).
std::mutex& LockAreasMutex() {
  static std::mutex mutex;
  return mutex;
}
std::vector<std::uint64_t>& LockAreas() {
  static std::vector<std::uint64_t> areas;
  return areas;
}

bool AreasOverlap(std::uint64_t a, std::uint64_t b) {
  return a < b + kLockBytes && b < a + kLockBytes;
}

constexpr std::array<int, 4> kCompiledRates{4, 5, 6, 8};

// Kernel arguments: the CUDA runtime copies each from the address given.
// NOLINTBEGIN(performance-no-int-to-ptr): device addresses.
void* Pointer(std::uint64_t address) { return reinterpret_cast<void*>(address); }
// NOLINTEND(performance-no-int-to-ptr)

// The addresses of a launch's arguments, in the kernel's parameter order.
template <typename... T>
std::array<void*, sizeof...(T)> Arguments(T&... values) {
  return {static_cast<void*>(&values)...};
}

}  // namespace

std::expected<std::unique_ptr<LaunchContext>, KernelFailure> LaunchContext::Create(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    std::uint64_t locks) {
  if (locks == 0 || locks % 256 != 0 || locks + kLockBytes < locks) {
    return Rejected("the lock area is not a 256-byte aligned device range");
  }
  // The host checks' constants are the kernels' (validate.h).
  if (llmp_exl3::GemmSharedMemory() != kGemmSharedMemory ||
      llmp_exl3::GemvMaxRows() != kGemvMaxRows) {
    return Rejected("the kernels' constants differ from the host checks'");
  }
  // Also makes the provider's context current, where the runtime binds.
  auto native = execution.Submission(stream);
  if (!native || native->handle == nullptr) {
    return Rejected("the provider refused the stream" +
                    (native ? std::string() : ": " + native.error().detail));
  }
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess || current != device) {
    return Rejected(std::format("device {} is not current on this thread", device));
  }
  if (const cudaError_t stale = cudaGetLastError(); stale != cudaSuccess) {
    return Rejected(std::string("a CUDA error is pending: ") + cudaGetErrorString(stale));
  }
  int sm_count = 0;
  int major = 0;
  int minor = 0;
  if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
      sm_count < 1) {
    (void)cudaGetLastError();
    return Rejected(std::format("device {} does not report its SMs and capability", device));
  }
  // Every GEMM kernel takes kGemmSharedMemory of dynamic shared memory, which
  // needs the opt-in limit raised first (exl3_gemm.cu does it on first use).
  for (const int bits : kCompiledRates) {
    for (int shape = 1; shape <= kShapes; ++shape) {
      for (const bool fp32 : {false, true}) {
        for (const void* kernel : {llmp_exl3::GemmKernel(bits, shape, fp32),
                                   llmp_exl3::MultiGemmKernel(bits, shape, fp32)}) {
          if (kernel == nullptr ||
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   kGemmSharedMemory) != cudaSuccess) {
            (void)cudaGetLastError();
            return Rejected(std::format(
                "the K = {} shape {} GEMM kernels cannot take {} bytes of shared memory", bits,
                shape, kGemmSharedMemory));
          }
        }
      }
    }
  }
  {
    const std::scoped_lock lock(LockAreasMutex());
    std::vector<std::uint64_t>& areas = LockAreas();
    if (std::ranges::any_of(areas, [&](std::uint64_t area) { return AreasOverlap(area, locks); })) {
      return Rejected("the lock area overlaps a live launch context's");
    }
    // Queued before any launch: kernels require zeroed lock slots.
    // NOLINTNEXTLINE(performance-no-int-to-ptr): a device address.
    if (const cudaError_t error = cudaMemsetAsync(reinterpret_cast<void*>(locks), 0, kLockBytes,
                                                  static_cast<cudaStream_t>(native->handle));
        error != cudaSuccess) {
      (void)cudaGetLastError();
      return Rejected(std::string("zeroing the lock area failed: ") + cudaGetErrorString(error));
    }
    areas.push_back(locks);
  }
  return std::unique_ptr<LaunchContext>(new LaunchContext(device, execution, stream, *native, locks,
                                                          sm_count, UpstreamCcClass(major, minor)));
}

LaunchContext::LaunchContext(int device, providers::DeviceExecution& execution,
                             providers::StreamId stream, providers::NativeStream native,
                             std::uint64_t locks, int sm_count, CcClass cc_class)
    : device_(device),
      execution_(execution),
      stream_(stream),
      native_(native),
      locks_(locks),
      sm_count_(sm_count),
      cc_class_(cc_class) {}

LaunchContext::~LaunchContext() {
  const std::scoped_lock lock(LockAreasMutex());
  std::vector<std::uint64_t>& areas = LockAreas();
  if (const auto found = std::ranges::find(areas, locks_); found != areas.end()) {
    areas.erase(found);
  }
}

std::expected<void*, KernelFailure> LaunchContext::Begin() {
  if (faulted_) {
    return Rejected("the launch context faulted earlier; its stream awaits recovery");
  }
  // The launch is queued work on the stream, which the provider must know
  // of first.
  auto native = execution_.Submission(stream_);
  if (!native) {
    if (native.error().error == providers::ProviderError::kUnknown) {
      faulted_ = true;
      return Fault("the stream faulted: " + native.error().detail);
    }
    return Rejected("the provider refused the stream: " + native.error().detail);
  }
  if (native->handle != native_.handle) {
    return Rejected("the provider's stream changed its native handle");
  }
  // An error left by another runtime call on this thread would be taken
  // for the launch's.
  if (const cudaError_t stale = cudaGetLastError(); stale != cudaSuccess) {
    faulted_ = true;
    return Fault(std::string("a CUDA error before the launch: ") + cudaGetErrorString(stale));
  }
  return native_.handle;
}

std::expected<void, KernelFailure> LaunchContext::End(std::expected<void, KernelFailure> launched) {
  const cudaError_t error = cudaGetLastError();
  if (!launched) {
    faulted_ = true;
    return Fault(launched.error().detail);
  }
  if (error != cudaSuccess) {
    faulted_ = true;
    return Fault(std::string("the launch left a CUDA error: ") + cudaGetErrorString(error));
  }
  return {};
}

std::expected<int, KernelFailure> LaunchContext::Coresident(const void* kernel, int threads,
                                                            int shared) {
  if (kernel == nullptr) {
    return Rejected("this build has no such kernel");
  }
  if (const auto found = coresident_.find(kernel); found != coresident_.end()) {
    return found->second;
  }
  int per_sm = 0;
  if (const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &per_sm, kernel, threads, static_cast<std::size_t>(shared));
      error != cudaSuccess) {
    (void)cudaGetLastError();
    return Rejected(std::string("the occupancy query failed: ") + cudaGetErrorString(error));
  }
  const int blocks = per_sm * sm_count_;
  coresident_.emplace(kernel, blocks);
  return blocks;
}

std::expected<int, KernelFailure> LaunchContext::GemmCoresident(int bits, int shape,
                                                                Output output) {
  if (shape < 1 || shape > kShapes) {
    return Rejected(std::format("tile shape {}", shape));
  }
  return Coresident(llmp_exl3::GemmKernel(bits, shape, output == Output::kF32),
                    kBlockDim.at(static_cast<std::size_t>(shape)), kGemmSharedMemory);
}

std::expected<int, KernelFailure> LaunchContext::MultiGemmCoresident(int bits, int shape,
                                                                     Output output) {
  if (shape < 1 || shape > kShapes) {
    return Rejected(std::format("tile shape {}", shape));
  }
  return Coresident(llmp_exl3::MultiGemmKernel(bits, shape, output == Output::kF32),
                    kBlockDim.at(static_cast<std::size_t>(shape)), kGemmSharedMemory);
}

std::expected<int, KernelFailure> LaunchContext::GemvCoresident(int bits, Output output, int m,
                                                                int config) {
  return Coresident(llmp_exl3::GemvKernel(bits, output == Output::kF32, m == 1 ? 0 : 1, config),
                    GemvThreads(config), 0);
}

std::expected<std::optional<GemvPlan>, KernelFailure> LaunchContext::UpstreamGemv(
    const Weights& weights, Output output, int m) {
  if (weights.bits != 4 || m < 1 || m > kGemvMaxRows) {
    return std::optional<GemvPlan>();  // no GEMV instance: upstream keeps the GEMM
  }
  auto narrow = GemvCoresident(weights.bits, output, m, 0);
  if (!narrow) {
    return std::unexpected(narrow.error());
  }
  auto wide = GemvCoresident(weights.bits, output, m, 1);
  if (!wide) {
    return std::unexpected(wide.error());
  }
  return UpstreamGemvPlan(cc_class_, m, weights.k, weights.n, weights.bits, kCodebookMcg, *narrow,
                          *wide);
}

std::expected<void, KernelFailure> LaunchContext::Gemm(const LinearOperands& o,
                                                       const GemmPlan& plan) {
  auto coresident = GemmCoresident(o.weights.bits, plan.shape, o.output);
  if (!coresident) {
    return std::unexpected(coresident.error());
  }
  if (auto checked = CheckGemm(o, plan, *coresident, locks_); !checked) {
    return checked;
  }
  const void* kernel = llmp_exl3::GemmKernel(o.weights.bits, plan.shape, o.output == Output::kF32);
  auto stream = Begin();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  // exl3_gemm_kernel(EXL3_GEMM_ARGS).
  void* a = Pointer(o.x);
  void* b = Pointer(o.weights.trellis);
  void* c = Pointer(o.y);
  int size_m = o.m;
  int size_k = o.weights.k;
  int size_n = o.weights.n;
  void* locks = Pointer(locks_);
  void* suh = Pointer(o.weights.suh);
  void* a_had = Pointer(o.a_had);
  void* svh = Pointer(o.weights.svh);
  auto args = Arguments(a, b, c, size_m, size_k, size_n, locks, suh, a_had, svh);
  return End(Launched(cudaLaunchCooperativeKernel(
      kernel, dim3(static_cast<unsigned>(plan.blocks)),
      dim3(static_cast<unsigned>(kBlockDim.at(static_cast<std::size_t>(plan.shape)))), args.data(),
      kGemmSharedMemory, static_cast<cudaStream_t>(*stream))));
}

std::expected<void, KernelFailure> LaunchContext::Gemv(const LinearOperands& o,
                                                       const GemvPlan& plan) {
  if (plan.config != 0 && plan.config != 1) {
    return Rejected(std::format("GEMV configuration {}", plan.config));
  }
  if (o.m < 1) {
    return Rejected(std::format("{} rows", o.m));
  }
  auto coresident = GemvCoresident(o.weights.bits, o.output, o.m, plan.config);
  if (!coresident) {
    return std::unexpected(coresident.error());
  }
  if (auto checked = CheckGemv(o, plan, *coresident, locks_); !checked) {
    return checked;
  }
  const void* kernel = llmp_exl3::GemvKernel(o.weights.bits, o.output == Output::kF32,
                                             o.m == 1 ? 0 : 1, plan.config);
  auto stream = Begin();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  // exl3_gemv_kernel takes exl3_gemm_kernel's arguments (EXL3_GEMM_ARGS).
  void* a = Pointer(o.x);
  void* b = Pointer(o.weights.trellis);
  void* c = Pointer(o.y);
  int size_m = o.m;
  int size_k = o.weights.k;
  int size_n = o.weights.n;
  void* locks = Pointer(locks_);
  void* suh = Pointer(o.weights.suh);
  void* a_had = Pointer(o.a_had);
  void* svh = Pointer(o.weights.svh);
  auto args = Arguments(a, b, c, size_m, size_k, size_n, locks, suh, a_had, svh);
  return End(
      Launched(cudaLaunchCooperativeKernel(kernel, dim3(static_cast<unsigned>(plan.blocks)),
                                           dim3(static_cast<unsigned>(GemvThreads(plan.config))),
                                           args.data(), 0, static_cast<cudaStream_t>(*stream))));
}

std::expected<void, KernelFailure> LaunchContext::MultiGemm(const MultiLinearOperands& o,
                                                            const MultiGemmPlan& plan) {
  auto coresident = MultiGemmCoresident(o.first.bits, plan.shape, o.output);
  if (!coresident) {
    return std::unexpected(coresident.error());
  }
  if (auto checked = CheckMultiGemm(o, plan, *coresident, locks_); !checked) {
    return checked;
  }
  const void* kernel =
      llmp_exl3::MultiGemmKernel(o.first.bits, plan.shape, o.output == Output::kF32);
  auto stream = Begin();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  // exl3_mgemm_kernel(EXL3_MGEMM_ARGS), as exl3_mgemm_gr passes them for
  // upstream's gated MLP: A (1 × m × k), C (2 × m × n), no indices, weights,
  // range, per-matrix widths or slices, one token.
  void* a = Pointer(o.x);
  void* b_list = Pointer(o.trellis_table);
  void* c = Pointer(o.y);
  int size_m = o.m;
  int size_k = o.first.k;
  int size_n = o.first.n;
  void* locks = Pointer(locks_);
  void* suh_list = Pointer(o.suh_table);
  void* a_had = Pointer(o.a_had);
  void* svh_list = Pointer(o.svh_table);
  void* indices = nullptr;
  void* weights = nullptr;
  int bszm_in = 1;
  int bszm_out = 2;
  int min_index = -1;
  int max_index = -1;
  int num_tokens = 1;
  void* size_n_list = nullptr;
  void* c_list = nullptr;
  void* n_stride_list = nullptr;
  void* had_src_list = nullptr;
  int num_had_src = 0;
  auto args = Arguments(a, b_list, c, size_m, size_k, size_n, locks, suh_list, a_had, svh_list,
                        indices, weights, bszm_in, bszm_out, min_index, max_index, num_tokens,
                        size_n_list, c_list, n_stride_list, had_src_list, num_had_src);
  return End(Launched(cudaLaunchCooperativeKernel(
      kernel, dim3(static_cast<unsigned>(plan.blocks), 1, static_cast<unsigned>(plan.concurrency)),
      dim3(static_cast<unsigned>(kBlockDim.at(static_cast<std::size_t>(plan.shape)))), args.data(),
      kGemmSharedMemory, static_cast<cudaStream_t>(*stream))));
}

std::expected<void, KernelFailure> LaunchContext::Reconstruct(const ReconstructOperands& o) {
  if (auto checked = CheckReconstruct(o); !checked) {
    return checked;
  }
  const void* kernel = o.fused ? llmp_exl3::ReconstructHadKernel(o.weights.bits)
                               : llmp_exl3::ReconstructKernel(o.weights.bits);
  if (kernel == nullptr) {
    return Rejected(std::format("no reconstruction kernel for K = {}", o.weights.bits));
  }
  auto stream = Begin();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  void* unpacked = Pointer(o.w);
  void* packed = Pointer(o.weights.trellis);
  int packed_blocks_n = o.weights.n / 16;
  int packed_n_offset = o.column / 16;
  const auto columns = static_cast<unsigned>(o.columns / kHadamardBlock);
  if (o.fused) {
    // reconstruct_had_slice: svh pre-offset to the slice's first column.
    void* suh = Pointer(o.weights.suh);
    void* svh = Pointer(o.weights.svh + (static_cast<std::uint64_t>(o.column) * 2));
    auto args = Arguments(unpacked, packed, suh, svh, packed_blocks_n, packed_n_offset);
    return End(Launched(
        cudaLaunchKernel(kernel, dim3(columns, static_cast<unsigned>(o.weights.k / kHadamardBlock)),
                         dim3(256), args.data(), 0, static_cast<cudaStream_t>(*stream))));
  }
  auto args = Arguments(unpacked, packed, packed_blocks_n, packed_n_offset);
  return End(
      Launched(cudaLaunchKernel(kernel, dim3(columns, static_cast<unsigned>(o.weights.k / 16)),
                                dim3(256), args.data(), 0, static_cast<cudaStream_t>(*stream))));
}

std::expected<void, KernelFailure> LaunchContext::Hadamard(const HadamardOperands& o) {
  if (auto checked = CheckHadamard(o); !checked) {
    return checked;
  }
  const void* kernel =
      llmp_exl3::HadamardKernel(o.type == Output::kF32, o.input_scale, !o.input_scale);
  if (kernel == nullptr) {
    return Rejected("no such Hadamard kernel");
  }
  auto stream = Begin();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  void* input = Pointer(o.x);
  void* output = Pointer(o.y);
  void* scale = Pointer(o.scale);
  float r_scale = kHadamardScale;
  auto args = Arguments(input, output, scale, r_scale);
  return End(Launched(cudaLaunchKernel(
      kernel,
      dim3(static_cast<unsigned>(o.rows), static_cast<unsigned>(o.columns / kHadamardBlock)),
      dim3(32), args.data(), 0, static_cast<cudaStream_t>(*stream))));
}

std::expected<void, KernelFailure> LaunchContext::Bias(const BiasOperands& o) {
  if (auto checked = CheckBias(o); !checked) {
    return checked;
  }
  auto stream = Begin();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  void* x = Pointer(o.x);
  void* y = Pointer(o.bias);
  void* z = Pointer(o.y);
  std::uint64_t numel_x =
      static_cast<std::uint64_t>(o.rows) * static_cast<std::uint64_t>(o.columns);
  auto numel_y = static_cast<std::uint64_t>(o.columns);
  const auto blocks = static_cast<unsigned>((numel_x + kAddThreads - 1) / kAddThreads);
  auto args = Arguments(x, y, z, numel_x, numel_y);
  return End(Launched(cudaLaunchKernel(llmp_exl3::AddKernelHhh(), dim3(blocks), dim3(kAddThreads),
                                       args.data(), 0, static_cast<cudaStream_t>(*stream))));
}

}  // namespace llmp::kernels::exl3
