// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Native ownership adapter. Original numerical definitions live in a separate
// private translation unit and receive only borrowed pointers and facts.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/dsv4_ds4_product.h"
#include "kernels/ggml/dsv4_ds4_product_raw.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/mul_mat_q_borrowed.cuh"
#include "kernels/ggml/ops_ext.h"

namespace jitllm::kernels::ggml {
namespace {
constexpr std::uint64_t kCublasWorkspace = 32ULL << 20;
constexpr std::uint64_t kOutAShared = 40960;
constexpr std::uint64_t kTailBytes = 256 * 144;

auto Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}
Ds4ProductRead Read(Ds4ProductWrite d) { return {.data = d.data, .bytes = d.bytes}; }
bool Disjoint(LaunchContext::Workspace w, Ds4ProductRead d) {
  if (w.size.value() == 0 || d.data == nullptr) return true;
  std::uint64_t end = 0, dend = 0;
  const auto start = reinterpret_cast<std::uintptr_t>(d.data);
  return !__builtin_add_overflow(w.base, w.size.value(), &end) &&
         !__builtin_add_overflow(static_cast<std::uint64_t>(start), d.bytes, &dend) &&
         (end <= start || dend <= w.base);
}
bool WorkspaceFits(const LaunchContext& launch, std::initializer_list<Ds4ProductRead> operands) {
  for (const auto d : operands) {
    if (!Disjoint(launch.workspace(), d)) return false;
    if (launch.cublas() != nullptr && !Disjoint(launch.cublas()->workspace(), d)) return false;
  }
  return true;
}
std::expected<ds4_product::Device, KernelFailure> Device(const LaunchContext& launch) {
  const auto& info = ggml_cuda_info();
  if (launch.device() < 0 || launch.device() >= info.device_count)
    return Rejected("ds4 product has no native device facts");
  const auto& d = info.devices[launch.device()];
  if (d.cc != 1210 || d.warp_size != 32)
    return Rejected("original ds4 product closure requires sm_121 device facts");
  return ds4_product::Device{
      .cc = d.cc, .multiprocessors = d.nsm, .warp = d.warp_size, .shared_optin = d.smpbo};
}
ds4_product::Rope Rope(const Ds4ProductRope& r) {
  return {.positions = static_cast<const std::int32_t*>(r.positions.data),
          .first = r.first,
          .step = r.step,
          .original_context = r.original_context,
          .rotary = r.rotary,
          .base = r.base,
          .scale = r.scale,
          .extension = r.extension,
          .attention = r.attention,
          .beta_fast = r.beta_fast,
          .beta_slow = r.beta_slow,
          .inverse = r.inverse};
}
cudaError_t ZeroTail(const Ds4D4Sidecar& q, cudaStream_t stream) {
  const auto payload = static_cast<std::uint64_t>(q.rows) * (q.columns / 128) * 144;
  return cudaMemsetAsync(static_cast<char*>(q.storage.data) + payload, 0, kTailBytes, stream);
}
std::expected<ds4_product::MmqPlan, KernelFailure> MmqPlan(const LaunchContext& launch,
                                                           const Ds4Q8Product& d) {
  const auto device = Device(launch);
  if (!device) return std::unexpected(device.error());
  ds4_product::MmqPlan p;
  if (!ds4_product::PlanMmq(*device, static_cast<int>(d.weights.rows),
                            static_cast<int>(d.input.rows), static_cast<int>(d.input.columns), p))
    return Rejected("original ds4 MMQ has no legal bounded launch for this shape");
  return p;
}
// Bounded automatic metadata only. Kernels retain copied scalar arguments
// and device addresses, never these host descriptors after the launch.
struct NativeQ8View {
  ggml_tensor weights{}, source{}, output{};
  BorrowedMmqD4 quantized{};
  explicit NativeQ8View(const Ds4Q8Product& d) {
    const auto matrix = [](ggml_tensor& tensor, ggml_type type, void* data, std::uint32_t width,
                           std::uint32_t rows) {
      tensor.type = type;
      tensor.data = data;
      tensor.ne[0] = width;
      tensor.ne[1] = rows;
      tensor.ne[2] = tensor.ne[3] = 1;
      tensor.nb[0] = ggml_type_size(type);
      tensor.nb[1] = ggml_row_size(type, width);
      tensor.nb[2] = tensor.nb[3] = tensor.nb[1] * rows;
    };
    matrix(weights, GGML_TYPE_Q8_0, const_cast<void*>(d.weights.raw.data), d.weights.columns,
           d.weights.rows);
    matrix(source, GGML_TYPE_F32, const_cast<void*>(d.input.storage.data), d.input.columns,
           d.input.rows);
    matrix(output, GGML_TYPE_F32, d.output.storage.data, d.output.columns, d.output.rows);
    output.op = GGML_OP_MUL_MAT;
    output.src[0] = &weights;
    output.src[1] = &source;
    quantized = {.data = d.quantized.storage.data,
                 .bytes = d.quantized.storage.bytes,
                 .source = d.quantized.source,
                 .generation = d.quantized.generation};
  }
};
}  // namespace

std::expected<void, KernelFailure> RunDs4Embedding(LaunchContext& launch, const Ds4Embedding& d) {
  if (auto c = CheckDs4Embedding(d); !c) return c;
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  if (!WorkspaceFits(launch, {d.tokens, d.weights.storage, Read(d.output.storage)}))
    return Rejected("ds4 embedding overlaps native workspace");
  return launch.Run(base::Bytes(0), [d](ggml_backend_cuda_context& c) {
    CUDA_CHECK(ds4_product::Embedding(
        static_cast<const std::int32_t*>(d.tokens.data), d.weights.storage.data,
        static_cast<float*>(d.output.storage.data), d.weights.rows, d.output.rows,
        d.weights.columns, d.hyper_connections, c.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4F16Conversion(LaunchContext& launch,
                                                       const Ds4F16Conversion& d) {
  if (auto c = CheckDs4F16Conversion(d); !c) return c;
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  if (!WorkspaceFits(launch, {d.input.storage, Read(d.output.storage)}))
    return Rejected("ds4 F16 conversion overlaps native workspace");
  return launch.Run(base::Bytes(0), [d](ggml_backend_cuda_context& c) {
    CUDA_CHECK(ds4_product::F16Conversion(
        static_cast<const float*>(d.input.storage.data), d.output.storage.data,
        static_cast<std::uint64_t>(d.input.rows) * d.input.columns, c.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4QkvNorm(LaunchContext& launch, const Ds4QkvNorm& d) {
  if (auto c = CheckDs4QkvNorm(d); !c) return c;
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  if (!WorkspaceFits(launch, {d.query.storage, d.query_weight, Read(d.query_output.storage),
                              d.kv.storage, d.kv_weight, Read(d.kv_output.storage)}))
    return Rejected("ds4 fused QKV RMS overlaps native workspace");
  return launch.Run(base::Bytes(0), [d](ggml_backend_cuda_context& c) {
    CUDA_CHECK(ds4_product::QkvNorm(static_cast<const float*>(d.query.storage.data),
                                    static_cast<const float*>(d.query_weight.data),
                                    static_cast<float*>(d.query_output.storage.data),
                                    d.query.columns, static_cast<const float*>(d.kv.storage.data),
                                    static_cast<const float*>(d.kv_weight.data),
                                    static_cast<float*>(d.kv_output.storage.data), d.kv.columns,
                                    d.query.rows, d.epsilon, c.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4F16Product(LaunchContext& launch, const Ds4F16Product& d) {
  if (auto c = CheckDs4F16Product(d); !c) return c;
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  const auto* handle = launch.cublas();
  if (handle == nullptr || handle->workspace().size.value() != kCublasWorkspace ||
      !WorkspaceFits(launch, {d.weights.storage, d.input.storage, Read(d.output.storage)}))
    return Rejected("original ds4 F16 needs its disjoint charged 32 MiB lent cuBLAS workspace");
  return launch.Run(base::Bytes(0), [d](ggml_backend_cuda_context& c) {
    auto* handle = internal::CublasHandleOf(c);
    cublasMath_t math = CUBLAS_DEFAULT_MATH;
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
    CUBLAS_CHECK(cublasGetMathMode(handle, &math));
    if (internal::CudaErrorPending()) return;
    CUBLAS_CHECK(cublasGetPointerMode(handle, &mode));
    if (internal::CudaErrorPending()) return;
    if (math != CUBLAS_TF32_TENSOR_OP_MATH || mode != CUBLAS_POINTER_MODE_HOST) {
      ggml_cuda_error("ds4 F16 lent handle state", __func__, __FILE__, __LINE__,
                      "original math/pointer mode is not present");
      return;
    }
    const float alpha = 1.0F, beta = 0.0F;
    // Original ds4_gpu_matmul_f16_preconv_tensor geometry and output type.
    // Physical leading dimensions are explicit; no operand is repacked here.
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(d.weights.rows),
        static_cast<int>(d.input.rows), static_cast<int>(d.input.columns), &alpha,
        d.weights.storage.data, CUDA_R_16F, static_cast<int>(d.weights.row_stride / 2),
        d.input.storage.data, CUDA_R_16F, static_cast<int>(d.input.row_stride / 2), &beta,
        d.output.storage.data, CUDA_R_32F, static_cast<int>(d.output.row_stride / 4),
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  });
}

std::expected<void, KernelFailure> RunDs4D4(LaunchContext& launch, const Ds4ProductMatrix& input,
                                            std::uint64_t generation, const Ds4D4Sidecar& d) {
  if (auto c = CheckDs4D4(input, generation, d); !c) return c;
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  if (!WorkspaceFits(launch, {input.storage, Read(d.storage)}))
    return Rejected("ds4 D4 sidecar overlaps native workspace");
  return launch.Run(base::Bytes(0), [input, d](ggml_backend_cuda_context& c) {
    CUDA_CHECK(ZeroTail(d, c.stream()));
    if (internal::CudaErrorPending()) return;
    CUDA_CHECK(ds4_product::D4(static_cast<const float*>(input.storage.data), d.storage.data,
                               static_cast<int>(input.rows), static_cast<int>(input.columns),
                               c.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4F16Vector(LaunchContext& launch, const Ds4F16Vector& d) {
  const auto bytes = PlanDs4F16Vector(d);
  if (!bytes) return std::unexpected(bytes.error());
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  if (!WorkspaceFits(launch, {d.weights.storage, d.input.storage, Read(d.output.storage)}))
    return Rejected("ds4 small F16 operands overlap native workspace");
  const auto split = *bytes == 0 ? 1 : *bytes / (static_cast<std::uint64_t>(d.weights.rows) * 4);
  return launch.Run(base::Bytes(*bytes), [d, bytes = *bytes, split](ggml_backend_cuda_context& c) {
    ggml_cuda_pool_alloc<char> partial(c.pool());
    if (bytes != 0) partial.alloc(static_cast<std::size_t>(bytes));
    if (internal::CudaErrorPending()) return;
    CUDA_CHECK(ds4_product::F16Vector(
        d.weights.storage.data, static_cast<const float*>(d.input.storage.data),
        static_cast<float*>(d.output.storage.data), static_cast<int>(d.weights.rows),
        static_cast<int>(d.input.rows), static_cast<int>(d.input.columns), static_cast<int>(split),
        partial.get(), bytes, c.stream()));
  });
}

std::expected<std::uint64_t, KernelFailure> PlanDs4Q8Product(const LaunchContext& launch,
                                                             const Ds4Q8Product& d) {
  if (auto c = CheckDs4Q8Product(d); !c) return std::unexpected(c.error());
  const auto device = Device(launch);
  if (!device) return std::unexpected(device.error());
  if (!WorkspaceFits(launch, {d.input.storage, Read(d.output.storage), Read(d.quantized.storage),
                              d.weights.raw, d.weights.scales, d.weights.codes}))
    return Rejected("ds4 Q8 operands overlap native workspace");
  if (d.path == Ds4Q8Path::kDenseD2r) {
    if (ds4_product::DenseD2rSharedBytes() > device->shared_optin)
      return Rejected("original dense D2R shared memory is unavailable");
    return 0;
  }
  const auto p = MmqPlan(launch, d);
  if (!p) return std::unexpected(p.error());
  // One draw, starting at the pool's 256-byte boundary. No original pool.
  return p->fixup;
}

std::expected<void, KernelFailure> RunDs4Q8Product(LaunchContext& launch, const Ds4Q8Product& d) {
  const auto bytes = PlanDs4Q8Product(launch, d);
  if (!bytes) return std::unexpected(bytes.error());
  ds4_product::MmqPlan p;
  if (d.path == Ds4Q8Path::kMmq) {
    const auto plan = MmqPlan(launch, d);
    if (!plan) return std::unexpected(plan.error());
    p = *plan;
  }
  return launch.Run(base::Bytes(*bytes), [d, p](ggml_backend_cuda_context& c) {
    const auto stream = c.stream();
    CUDA_CHECK(ZeroTail(d.quantized, stream));
    if (internal::CudaErrorPending()) return;
    if (!d.prepared) {
      CUDA_CHECK(ds4_product::D4(static_cast<const float*>(d.input.storage.data),
                                 d.quantized.storage.data, static_cast<int>(d.input.rows),
                                 static_cast<int>(d.input.columns), stream));
      if (internal::CudaErrorPending()) return;
    }
    if (d.path == Ds4Q8Path::kDenseD2r) {
      CUDA_CHECK(ds4_product::DenseD2r(
          d.weights.scales.data, d.weights.codes.data, d.quantized.storage.data,
          static_cast<float*>(d.output.storage.data), static_cast<int>(d.weights.rows),
          static_cast<int>(d.input.rows), static_cast<int>(d.input.columns), stream));
      return;
    }
    ggml_cuda_pool_alloc<char> fixup(c.pool());
    if (p.fixup != 0) fixup.alloc(static_cast<std::size_t>(p.fixup));
    if (internal::CudaErrorPending()) return;
    CUDA_CHECK(ds4_product::Mmq(
        d.weights.raw.data, d.quantized.storage.data, static_cast<float*>(d.output.storage.data),
        static_cast<int>(d.weights.rows), static_cast<int>(d.input.rows),
        static_cast<int>(d.input.columns), p, fixup.get(), p.fixup, stream));
  });
}

std::expected<std::uint64_t, KernelFailure> PlanDs4Q8NativeMmq(const LaunchContext& launch,
                                                               const Ds4Q8Product& d) {
  if (auto checked = CheckDs4Q8Product(d); !checked) return std::unexpected(checked.error());
  if (!d.prepared || d.path != Ds4Q8Path::kMmq || d.weights.raw.data == nullptr)
    return Rejected("native Q8 consumer requires the original prepared D4/raw-MMQ descriptor");
  if (auto device = Device(launch); !device) return std::unexpected(device.error());
  // Preserve the complete original view, including any co-resident aligned
  // planes, as disjoint charged operands of the same native Run.
  if (!WorkspaceFits(launch, {d.input.storage, Read(d.output.storage), Read(d.quantized.storage),
                              d.weights.raw, d.weights.scales, d.weights.codes}))
    return Rejected("native Q8 control overlaps an original operand workspace");
  NativeQ8View view(d);
  return PlanMulMatQBorrowedD4(launch, &view.output, view.quantized, d.generation);
}

std::expected<void, KernelFailure> RunDs4Q8NativeMmq(LaunchContext& launch, const Ds4Q8Product& d) {
  auto bytes = PlanDs4Q8NativeMmq(launch, d);
  if (!bytes) return std::unexpected(bytes.error());
  return launch.Run(base::Bytes(*bytes), [d](ggml_backend_cuda_context& context) {
    CUDA_CHECK(ZeroTail(d.quantized, context.stream()));
    if (internal::CudaErrorPending()) return;
    NativeQ8View view(d);
    internal::LaunchMulMatQBorrowedD4(context, &view.output, d.quantized.storage.data);
    if (internal::CudaErrorPending()) return;
    CUDA_CHECK(ds4_product::Sanitize(static_cast<float*>(d.output.storage.data),
                                     static_cast<std::uint64_t>(d.output.rows) * d.output.columns,
                                     context.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4HeadRope(LaunchContext& launch, const Ds4HeadRope& d) {
  if (auto c = CheckDs4HeadRope(d); !c) return c;
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  if (!WorkspaceFits(launch, {Read(d.input.storage), d.rope.positions}))
    return Rejected("ds4 head RoPE overlaps native workspace");
  return launch.Run(base::Bytes(0), [d](ggml_backend_cuda_context& c) {
    CUDA_CHECK(ds4_product::HeadRope(static_cast<float*>(d.input.storage.data), d.input.rows,
                                     d.heads, d.head_width, Rope(d.rope), d.normalize, d.epsilon,
                                     c.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4Q81(LaunchContext& launch, const Ds4ProductMatrix& input,
                                             std::uint64_t generation, const Ds4Q81Sidecar& d) {
  if (auto c = CheckDs4Q81(input, generation, d); !c) return c;
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  if (!WorkspaceFits(launch, {input.storage, Read(d.storage)}))
    return Rejected("ds4 Q8_1 sidecar overlaps native workspace");
  return launch.Run(base::Bytes(0), [input, d](ggml_backend_cuda_context& c) {
    CUDA_CHECK(ds4_product::Q81(static_cast<const float*>(input.storage.data), d.storage.data,
                                static_cast<int>(input.rows), static_cast<int>(input.columns),
                                c.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4Q8Vector(LaunchContext& launch, const Ds4Q8Vector& d) {
  if (auto c = CheckDs4Q8Vector(d); !c) return c;
  if (auto c = Device(launch); !c) return std::unexpected(c.error());
  if (!WorkspaceFits(launch, {d.input.storage, Read(d.output.storage), Read(d.quantized.storage),
                              d.weights.raw, d.weights.scales, d.weights.codes}))
    return Rejected("ds4 small Q8 operands overlap native workspace");
  return launch.Run(base::Bytes(0), [d](ggml_backend_cuda_context& c) {
    if (!d.prepared) {
      CUDA_CHECK(ds4_product::Q81(static_cast<const float*>(d.input.storage.data),
                                  d.quantized.storage.data, static_cast<int>(d.input.rows),
                                  static_cast<int>(d.input.columns), c.stream()));
      if (internal::CudaErrorPending()) return;
    }
    CUDA_CHECK(ds4_product::Q8Vector(
        d.weights.raw.data, d.weights.scales.data, d.weights.codes.data, d.quantized.storage.data,
        static_cast<float*>(d.output.storage.data), static_cast<int>(d.weights.rows),
        static_cast<int>(d.input.rows), static_cast<int>(d.input.columns),
        d.path == Ds4Q8VectorPath::kAligned, c.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4OutA(LaunchContext& launch, const Ds4OutA& d) {
  if (auto c = CheckDs4OutA(d); !c) return c;
  const auto device = Device(launch);
  if (!device) return std::unexpected(device.error());
  if (kOutAShared > device->shared_optin ||
      !WorkspaceFits(launch,
                     {d.heads.storage, Read(d.low.storage), Read(d.rope_table), d.weights.scales,
                      d.weights.codes, Read(d.quantized.storage), d.rope.positions}))
    return Rejected("ds4 own out-a shared memory or disjoint workspace is unavailable");
  return launch.Run(base::Bytes(0), [d](ggml_backend_cuda_context& c) {
    if (d.quantized.storage.data != nullptr) {
      CUDA_CHECK(ZeroTail(d.quantized, c.stream()));
      if (internal::CudaErrorPending()) return;
    }
    CUDA_CHECK(ds4_product::OutA(d.weights.scales.data, d.weights.codes.data,
                                 static_cast<const float*>(d.heads.storage.data),
                                 static_cast<float*>(d.low.storage.data), d.rope_table.data,
                                 d.quantized.storage.data, d.heads.rows, Rope(d.rope), c.stream()));
  });
}
}  // namespace jitllm::kernels::ggml
