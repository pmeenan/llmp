// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// GGML's cuBLAS matrix multiplication under llmpalooza's launch context, on
// llmpalooza's cuBLAS handle, on a GB10 (label `gpu`; docs/backend-proof.md,
// P1):
// - the handle carries upstream's state (TF32 math, the provider's stream)
//   and GGML finds it instead of creating its own;
// - each cuBLAS entry point the Qwen2 shapes use (GemmEx, strided and
//   pointer-array batched) gives identical results in cudaMalloc memory,
//   device VMM and host VMM (BP-N3, D-034), close to a CPU reference;
// - the scratch each draws is the plan's bound, and for the output head it
//   equals GGML's pool peaks recorded in P0;
// - what the path cannot run, or upstream would not route to it, is
//   refused before anything is queued.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace {

using llmp::base::Bytes;
using llmp::kernels::ggml::CublasGemm;
using llmp::kernels::ggml::CublasHandle;
using llmp::kernels::ggml::CublasOperand;
using llmp::kernels::ggml::KernelError;
using llmp::kernels::ggml::LaunchContext;
using llmp::kernels::ggml::TensorArena;
using llmp::providers::Access;
using llmp::providers::BackingKind;
using llmp::providers::DeviceExecution;
using llmp::providers::FenceState;
using llmp::providers::StreamId;
using llmp::providers::VmmProvider;

constexpr std::int64_t kWidth = 896;     // Qwen2.5-0.5B's hidden size
constexpr std::int64_t kVocab = 151936;  // and its vocabulary
constexpr std::int64_t kHeadDim = 64;
constexpr std::int64_t kHeads = 14;
constexpr std::int64_t kKvHeads = 2;
constexpr std::int64_t kTokens = 32;   // above MMF's 16 columns: upstream picks cuBLAS
constexpr std::int64_t kKv = 256;      // KV cells in use
constexpr std::int64_t kKvSize = 512;  // KV cells in the cache
constexpr Bytes kScratch(16ULL << 20);

enum class Memory : std::uint8_t { kCudaMalloc, kDeviceVmm, kHostVmm };

// Deterministic inputs in [-1, 1).
float Value(std::uint64_t seed, std::uint64_t i) {
  std::uint64_t x = (seed * 0x9E3779B97F4A7C15ULL) ^ (i + 0x632BE59BD9B4E019ULL);
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 29;
  return static_cast<float>(static_cast<double>(x >> 40) / static_cast<double>(1ULL << 23)) - 1.0f;
}

// F16 values as the device sees them.
float Half(std::uint64_t seed, std::uint64_t i, float scale) {
  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(Value(seed, i) * scale));
}

std::vector<std::uint32_t> Bits(const std::vector<float>& values) {
  std::vector<std::uint32_t> bits;
  bits.reserve(values.size());
  for (const float value : values) {
    bits.push_back(std::bit_cast<std::uint32_t>(value));
  }
  return bits;
}

// The largest difference relative to 1 + |reference|; infinite if the
// lengths differ or any value is not finite, which std::max would skip.
double Worst(const std::vector<float>& got, const std::vector<double>& want) {
  if (got.size() != want.size() || got.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
      return std::numeric_limits<double>::infinity();
    }
    worst = std::max(worst, std::abs(got[i] - want[i]) / (1.0 + std::abs(want[i])));
  }
  return worst;
}

class GgmlCublasTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memory_ = std::move(llmp::providers::cuda::OpenDeviceMemory(0).value());
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    staging_ = Vmm(BackingKind::kHost, Bytes(64ULL << 20));
  }

  void TearDown() override {
    Finish();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : malloced_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
    for (const auto& [reservation, backing, size] : mapped_) {
      ASSERT_TRUE(memory_->Unmap(reservation, Bytes(0), size).has_value());
      ASSERT_TRUE(memory_->Release(backing).has_value());
      ASSERT_TRUE(memory_->Free(reservation).has_value());
    }
  }

  std::uint64_t Vmm(BackingKind kind, Bytes size) {
    const std::uint64_t granule = memory_->Granularity().value();
    const Bytes rounded((size.value() + granule - 1) / granule * granule);
    std::size_t kind_class = 0;
    for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
      if (memory_->Classes()[i].kind == kind) {
        kind_class = i;
      }
    }
    const auto reservation = memory_->Reserve(rounded).value();
    const auto backing = memory_->Create(kind_class, rounded).value();
    EXPECT_TRUE(memory_->Map(reservation, Bytes(0), backing).has_value());
    EXPECT_TRUE(memory_->SetAccess(reservation, Bytes(0), rounded, Access::kReadWrite).has_value());
    mapped_.push_back({reservation, backing, rounded});
    return memory_->RangeOf(reservation).value().base;
  }

  std::uint64_t Allocate(Memory memory, std::size_t size) {
    switch (memory) {
      case Memory::kCudaMalloc: {
        void* pointer = nullptr;
        EXPECT_EQ(cudaMalloc(&pointer, size), cudaSuccess);
        malloced_.push_back(pointer);
        return reinterpret_cast<std::uintptr_t>(pointer);
      }
      case Memory::kDeviceVmm:
        return Vmm(BackingKind::kDevice, Bytes(size));
      case Memory::kHostVmm:
        return Vmm(BackingKind::kHost, Bytes(size));
    }
    return 0;
  }

  // Upstream's handle, on a workspace of upstream's size for this device.
  std::unique_ptr<CublasHandle> Handle() {
    int major = 0;
    int minor = 0;
    EXPECT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
    EXPECT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
    const Bytes size = CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor));
    auto handle = CublasHandle::Create(0, *execution_, stream_,
                                       {.base = Vmm(BackingKind::kDevice, size), .size = size});
    EXPECT_TRUE(handle.has_value()) << (handle ? "" : handle.error().detail);
    return handle ? std::move(*handle) : nullptr;
  }

  std::unique_ptr<LaunchContext> Launcher(CublasHandle* cublas, Bytes scratch = kScratch) {
    const std::uint64_t base = Vmm(BackingKind::kDevice, scratch);
    auto launch =
        LaunchContext::Create(0, *execution_, stream_, {.base = base, .size = scratch}, cublas);
    EXPECT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    return launch ? std::move(*launch) : nullptr;
  }

  // Everything queued on `stream` so far has run.
  void Finish(StreamId stream) {
    const auto fence = execution_->Record(stream).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }
  void Finish() { Finish(stream_); }

  // Copies host bytes to `address` through the staging region, and waits,
  // so the region can be reused.
  void Upload(std::uint64_t address, const void* data, std::size_t size) {
    for (std::size_t done = 0; done < size;) {
      const std::size_t part = std::min<std::size_t>(size - done, 64ULL << 20);
      std::memcpy(reinterpret_cast<void*>(staging_),  // NOLINT(performance-no-int-to-ptr)
                  static_cast<const std::byte*>(data) + done, part);
      ASSERT_TRUE(execution_->Copy(stream_, address + done, staging_, Bytes(part)).has_value());
      Finish();
      done += part;
    }
  }

  std::vector<float> Download(std::uint64_t address, std::size_t count) {
    std::vector<float> values(count);
    const std::size_t size = count * sizeof(float);
    for (std::size_t done = 0; done < size;) {
      const std::size_t part = std::min<std::size_t>(size - done, 64ULL << 20);
      EXPECT_TRUE(execution_->Copy(stream_, staging_, address + done, Bytes(part)).has_value());
      Finish();
      const void* staged =
          reinterpret_cast<const void*>(staging_);  // NOLINT(performance-no-int-to-ptr)
      std::memcpy(reinterpret_cast<std::byte*>(values.data()) + done, staged, part);
      done += part;
    }
    return values;
  }

  std::unique_ptr<VmmProvider> memory_;
  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::uint64_t staging_ = 0;
  std::vector<void*> malloced_;
  struct Mapped {
    llmp::providers::ReservationId reservation;
    llmp::providers::BackingId backing;
    Bytes size;
  };
  std::vector<Mapped> mapped_;
};

TEST_F(GgmlCublasTest, TheHandleCarriesUpstreamsStateAndGgmlFindsIt) {
  EXPECT_EQ(CublasHandle::UpstreamWorkspace(1210), Bytes(32ULL << 20));
  EXPECT_EQ(CublasHandle::UpstreamWorkspace(860), Bytes(4ULL << 20));
  auto handle = Handle();
  ASSERT_NE(handle, nullptr);
  cublasMath_t math = CUBLAS_DEFAULT_MATH;
  ASSERT_EQ(cublasGetMathMode(handle->native(), &math), CUBLAS_STATUS_SUCCESS);
  EXPECT_EQ(math, CUBLAS_TF32_TENSOR_OP_MATH);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cublasGetStream(handle->native(), &stream), CUBLAS_STATUS_SUCCESS);
  EXPECT_EQ(static_cast<void*>(stream), handle->native_stream().handle);
  int sm_target = -1;
  ASSERT_EQ(cublasGetSmCountTarget(handle->native(), &sm_target), CUBLAS_STATUS_SUCCESS);
  EXPECT_EQ(sm_target, 0);  // cuBLAS's default: every SM

  auto launch = Launcher(handle.get());
  ASSERT_NE(launch, nullptr);
  EXPECT_EQ(launch->cublas(), handle.get());
  const auto looked_up = launch->Run(Bytes(0), [&handle](ggml_backend_cuda_context& context) {
    // What every GGML cuBLAS call site asks for.
    EXPECT_EQ(llmp::kernels::ggml::internal::CublasHandleOf(context), handle->native());
    EXPECT_FALSE(llmp::kernels::ggml::internal::HoldsCublasWorkspace(context));
  });
  EXPECT_TRUE(looked_up.has_value());
  launch.reset();  // before the handle it borrows
}

TEST_F(GgmlCublasTest, AHandleMustFitItsContextAndWorkspace) {
  // cuBLAS refuses a workspace aligned to less than 256 bytes.
  const std::uint64_t base = Vmm(BackingKind::kDevice, Bytes(4ULL << 20));
  const auto misaligned =
      CublasHandle::Create(0, *execution_, stream_, {.base = base + 128, .size = Bytes(1 << 20)});
  ASSERT_FALSE(misaligned.has_value());
  EXPECT_EQ(misaligned.error().error, KernelError::kRejected);

  // cuBLAS's own numerics switches would change the executed plan.
  for (const char* name : {"NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG"}) {
    ASSERT_EQ(setenv(name, "0", 1), 0);  // NOLINT(concurrency-mt-unsafe): one thread
    const auto switched =
        CublasHandle::Create(0, *execution_, stream_, {.base = base, .size = Bytes(1 << 20)});
    ASSERT_EQ(unsetenv(name), 0);  // NOLINT(concurrency-mt-unsafe)
    ASSERT_FALSE(switched.has_value()) << name;
    EXPECT_NE(switched.error().detail.find(name), std::string::npos) << switched.error().detail;
  }

  auto handle = Handle();
  ASSERT_NE(handle, nullptr);
  // A handle for another stream, or a scratch workspace inside the
  // handle's, is refused.
  const StreamId other = execution_->CreateStream().value();
  const auto elsewhere = LaunchContext::Create(
      0, *execution_, other, {.base = base, .size = Bytes(1 << 20)}, handle.get());
  ASSERT_FALSE(elsewhere.has_value());
  EXPECT_EQ(elsewhere.error().error, KernelError::kRejected);
  // Taking the stream counted as queued work: a fence releases it.
  Finish(other);
  ASSERT_TRUE(execution_->DestroyStream(other).has_value());
  const auto overlapping = LaunchContext::Create(
      0, *execution_, stream_, {.base = handle->workspace().base, .size = Bytes(1 << 20)},
      handle.get());
  ASSERT_FALSE(overlapping.has_value());
  EXPECT_EQ(overlapping.error().error, KernelError::kRejected);
}

TEST_F(GgmlCublasTest, OperandsMayNotOverlapAWorkspace) {
  auto handle = Handle();
  auto launch = Launcher(handle.get());
  auto arena = TensorArena::Create(8).value();
  ggml_context* context = arena.context();
  const std::uint64_t memory = Vmm(BackingKind::kDevice, Bytes(8ULL << 20));
  ggml_tensor* w = ggml_new_tensor_2d(context, GGML_TYPE_F16, kWidth, 1024);
  ggml_tensor* x = ggml_new_tensor_2d(context, GGML_TYPE_F32, kWidth, kTokens);
  TensorArena::Bind(x, memory);
  ggml_tensor* y = ggml_mul_mat(context, w, x);
  TensorArena::Bind(y, memory + (4ULL << 20));
  // Weights in the scratch would be overwritten by the input's F16 copy
  // before the GEMM read them; in the cuBLAS workspace, by cuBLAS.
  for (const std::uint64_t at : {launch->workspace().base, handle->workspace().base}) {
    TensorArena::Bind(w, at);
    const auto refused = llmp::kernels::ggml::MulMatCublas(*launch, y);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().error, KernelError::kRejected);
  }
  EXPECT_FALSE(launch->faulted());
  launch.reset();
}

TEST_F(GgmlCublasTest, DestructionRestoresTheCallersContext) {
  auto handle = Handle();
  CUcontext owner = nullptr;
  ASSERT_EQ(cuCtxGetCurrent(&owner), CUDA_SUCCESS);
  ASSERT_NE(owner, nullptr);
  // The handle is destroyed in its own context whatever is current, and
  // the caller's (here none) is current again afterwards.
  ASSERT_EQ(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  handle.reset();
  CUcontext after = owner;
  ASSERT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
  EXPECT_EQ(after, nullptr);
  ASSERT_EQ(cuCtxSetCurrent(owner), CUDA_SUCCESS);
}

// One cuBLAS product of each kind the Qwen2 plans use, in one memory.
struct Products {
  std::vector<float> linear;   // a projection, 32 rows: GemmEx, F16
  std::vector<float> kq;       // K·Q over grouped heads: batched pointers, F32 (TF32)
  std::vector<float> kqv;      // V·softmax: batched pointers, F16
  std::vector<float> strided;  // two heads without grouping: strided batched, F16
  std::vector<float> packed;   // K·Q with K gathered from a wider cache: packed conversion
  std::vector<float> single;   // one F32 matrix: Sgemm (TF32)
  std::vector<float> bf16;     // BF16 weights: GemmEx, BF16 inputs, F32 output
  std::vector<float> samples;  // grouping within each of two samples: batched pointers
  std::vector<float> loose;    // weights 4-byte aligned: GemmEx on other kernels
};

constexpr std::int64_t kWideHeads = 3;  // KV heads in the wider cache

class GgmlCublasMemoryTest : public GgmlCublasTest {
 protected:
  Products Run(Memory memory) {
    auto arena = TensorArena::Create(64).value();
    ggml_context* context = arena.context();

    // A projection: F16 weights, F32 rows.
    std::vector<ggml_fp16_t> weights(kWidth * 1024);
    std::vector<float> rows(kWidth * kTokens);
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weights[i] = ggml_fp32_to_fp16(Value(1, i) * 0.1f);
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
      rows[i] = Value(2, i);
    }
    ggml_tensor* w = ggml_new_tensor_2d(context, GGML_TYPE_F16, kWidth, 1024);
    ggml_tensor* x = ggml_new_tensor_2d(context, GGML_TYPE_F32, kWidth, kTokens);
    TensorArena::Bind(w, Allocate(memory, weights.size() * sizeof(ggml_fp16_t)));
    TensorArena::Bind(x, Allocate(memory, rows.size() * sizeof(float)));
    Upload(reinterpret_cast<std::uintptr_t>(w->data), weights.data(),
           weights.size() * sizeof(ggml_fp16_t));
    Upload(reinterpret_cast<std::uintptr_t>(x->data), rows.data(), rows.size() * sizeof(float));
    ggml_tensor* linear = ggml_mul_mat(context, w, x);
    TensorArena::Bind(linear, Allocate(memory, ggml_nbytes(linear)));

    // Attention without flash attention, as llama.cpp builds it: K and V
    // views of an F16 cache (V transposed), grouped over the query heads.
    std::vector<ggml_fp16_t> k_cache(kHeadDim * kKvHeads * kKvSize);
    std::vector<ggml_fp16_t> v_cache(kKvSize * kHeadDim * kKvHeads);
    std::vector<float> queries(kHeadDim * kHeads * kTokens);
    for (std::size_t i = 0; i < k_cache.size(); ++i) {
      k_cache[i] = ggml_fp32_to_fp16(Value(3, i));
      v_cache[i] = ggml_fp32_to_fp16(Value(4, i));
    }
    for (std::size_t i = 0; i < queries.size(); ++i) {
      queries[i] = Value(5, i) * 0.125f;
    }
    ggml_tensor* kc = ggml_new_tensor_2d(context, GGML_TYPE_F16, kHeadDim * kKvHeads, kKvSize);
    ggml_tensor* vc = ggml_new_tensor_2d(context, GGML_TYPE_F16, kKvSize, kHeadDim * kKvHeads);
    ggml_tensor* qt = ggml_new_tensor_3d(context, GGML_TYPE_F32, kHeadDim, kHeads, kTokens);
    TensorArena::Bind(kc, Allocate(memory, ggml_nbytes(kc)));
    TensorArena::Bind(vc, Allocate(memory, ggml_nbytes(vc)));
    TensorArena::Bind(qt, Allocate(memory, ggml_nbytes(qt)));
    Upload(reinterpret_cast<std::uintptr_t>(kc->data), k_cache.data(), ggml_nbytes(kc));
    Upload(reinterpret_cast<std::uintptr_t>(vc->data), v_cache.data(), ggml_nbytes(vc));
    Upload(reinterpret_cast<std::uintptr_t>(qt->data), queries.data(), ggml_nbytes(qt));
    ggml_tensor* k = ggml_permute(context,
                                  ggml_view_3d(context, kc, kHeadDim, kKvHeads, kKv,
                                               ggml_row_size(kc->type, kHeadDim), kc->nb[1], 0),
                                  0, 2, 1, 3);
    ggml_tensor* q = ggml_permute(context, qt, 0, 2, 1, 3);
    ggml_tensor* kq = ggml_mul_mat(context, k, q);
    EXPECT_TRUE(ggml_prec_set_acc(kq, GGML_PREC_F32));
    TensorArena::Bind(kq, Allocate(memory, ggml_nbytes(kq)));
    ggml_tensor* v =
        ggml_view_3d(context, vc, kKv, kHeadDim, kKvHeads, vc->nb[1], vc->nb[1] * kHeadDim, 0);
    ggml_tensor* kqv = ggml_mul_mat(context, v, kq);
    TensorArena::Bind(kqv, Allocate(memory, ggml_nbytes(kqv)));

    // Two heads each with their own keys: no grouping, packed operands.
    ggml_tensor* keys = ggml_new_tensor_3d(context, GGML_TYPE_F16, kHeadDim, kKv, kKvHeads);
    ggml_tensor* per_head = ggml_new_tensor_3d(context, GGML_TYPE_F32, kHeadDim, kTokens, kKvHeads);
    TensorArena::Bind(keys, reinterpret_cast<std::uintptr_t>(kc->data));  // same bytes
    TensorArena::Bind(per_head, reinterpret_cast<std::uintptr_t>(qt->data));
    ggml_tensor* strided = ggml_mul_mat(context, keys, per_head);
    TensorArena::Bind(strided, Allocate(memory, ggml_nbytes(strided)));

    // Two of three KV heads of a wider cache: the view has gaps, so the
    // path gathers it into packed rows (convert_nc).
    std::vector<ggml_fp16_t> wide_cache(kHeadDim * kWideHeads * kKvSize);
    for (std::size_t i = 0; i < wide_cache.size(); ++i) {
      wide_cache[i] = ggml_fp32_to_fp16(Value(6, i));
    }
    ggml_tensor* wc = ggml_new_tensor_2d(context, GGML_TYPE_F16, kHeadDim * kWideHeads, kKvSize);
    TensorArena::Bind(wc, Allocate(memory, ggml_nbytes(wc)));
    Upload(reinterpret_cast<std::uintptr_t>(wc->data), wide_cache.data(), ggml_nbytes(wc));
    ggml_tensor* gapped =
        ggml_permute(context,
                     ggml_view_3d(context, wc, kHeadDim, kKvHeads, kKv,
                                  ggml_row_size(wc->type, kHeadDim), wc->nb[1], 0),
                     0, 2, 1, 3);
    ggml_tensor* packed = ggml_mul_mat(context, gapped, q);
    EXPECT_TRUE(ggml_prec_set_acc(packed, GGML_PREC_F32));
    TensorArena::Bind(packed, Allocate(memory, ggml_nbytes(packed)));

    // One F32 matrix.
    std::vector<float> square(kHeadDim * 128);
    for (std::size_t i = 0; i < square.size(); ++i) {
      square[i] = Value(7, i);
    }
    ggml_tensor* sq = ggml_new_tensor_2d(context, GGML_TYPE_F32, kHeadDim, 128);
    TensorArena::Bind(sq, Allocate(memory, ggml_nbytes(sq)));
    Upload(reinterpret_cast<std::uintptr_t>(sq->data), square.data(), ggml_nbytes(sq));
    // Against the first head's rows alone: one matrix.
    ggml_tensor* first = ggml_view_2d(context, per_head, kHeadDim, kTokens, per_head->nb[1], 0);
    ggml_tensor* single = ggml_mul_mat(context, sq, first);
    TensorArena::Bind(single, Allocate(memory, ggml_nbytes(single)));

    // BF16 weights against the projection's rows.
    std::vector<ggml_bf16_t> brain(kWidth * 256);
    for (std::size_t i = 0; i < brain.size(); ++i) {
      brain[i] = ggml_fp32_to_bf16(Value(9, i) * 0.1f);
    }
    ggml_tensor* wb = ggml_new_tensor_2d(context, GGML_TYPE_BF16, kWidth, 256);
    TensorArena::Bind(wb, Allocate(memory, ggml_nbytes(wb)));
    Upload(reinterpret_cast<std::uintptr_t>(wb->data), brain.data(), ggml_nbytes(wb));
    ggml_tensor* bf16 = ggml_mul_mat(context, wb, x);
    TensorArena::Bind(bf16, Allocate(memory, ggml_nbytes(bf16)));

    // Two samples of four channels over two weight channels each.
    std::vector<ggml_fp16_t> sample_weights(kHeadDim * 128 * 2 * 2);
    std::vector<float> sample_rows(kHeadDim * kTokens * 4 * 2);
    for (std::size_t i = 0; i < sample_weights.size(); ++i) {
      sample_weights[i] = ggml_fp32_to_fp16(Value(10, i));
    }
    for (std::size_t i = 0; i < sample_rows.size(); ++i) {
      sample_rows[i] = Value(11, i);
    }
    ggml_tensor* sw = ggml_new_tensor_4d(context, GGML_TYPE_F16, kHeadDim, 128, 2, 2);
    ggml_tensor* sr = ggml_new_tensor_4d(context, GGML_TYPE_F32, kHeadDim, kTokens, 4, 2);
    TensorArena::Bind(sw, Allocate(memory, ggml_nbytes(sw)));
    TensorArena::Bind(sr, Allocate(memory, ggml_nbytes(sr)));
    Upload(reinterpret_cast<std::uintptr_t>(sw->data), sample_weights.data(), ggml_nbytes(sw));
    Upload(reinterpret_cast<std::uintptr_t>(sr->data), sample_rows.data(), ggml_nbytes(sr));
    ggml_tensor* samples = ggml_mul_mat(context, sw, sr);
    TensorArena::Bind(samples, Allocate(memory, ggml_nbytes(samples)));

    // F16 weights 4 bytes past a 256-byte boundary.
    ggml_tensor* lw = ggml_new_tensor_2d(context, GGML_TYPE_F16, kHeadDim, 128);
    TensorArena::Bind(lw, Allocate(memory, ggml_nbytes(lw) + 256) + 4);
    Upload(reinterpret_cast<std::uintptr_t>(lw->data), sample_weights.data(), ggml_nbytes(lw));
    ggml_tensor* loose = ggml_mul_mat(context, lw, first);
    TensorArena::Bind(loose, Allocate(memory, ggml_nbytes(loose)));

    auto handle = Handle();
    // Each product in a fresh context, so its peak is its own.
    const auto expect = [this, &handle](ggml_tensor* node, CublasGemm gemm, ggml_type compute,
                                        CublasOperand weights, std::uint64_t alignment) {
      auto launch = Launcher(handle.get());
      const auto plan = llmp::kernels::ggml::PlanMulMatCublas(*launch, node);
      ASSERT_TRUE(plan.has_value()) << plan.error().detail;
      EXPECT_EQ(plan->gemm, gemm);
      EXPECT_EQ(plan->compute, compute);
      EXPECT_EQ(plan->weights, weights);
      EXPECT_EQ(plan->alignment, alignment);
      const auto ran = llmp::kernels::ggml::MulMatCublas(*launch, node);
      ASSERT_TRUE(ran.has_value()) << ran.error().detail;
      EXPECT_EQ(launch->scratch_peak().value(), plan->scratch);  // the bound is exact
      EXPECT_FALSE(launch->faulted());
    };
    // Alignments: 256 unless a row of 64 F16 elements (128 bytes) or the
    // loose weights set it.
    expect(linear, CublasGemm::kGemmEx, GGML_TYPE_F16, CublasOperand::kDirect, 256);
    expect(kq, CublasGemm::kGemmBatchedEx, GGML_TYPE_F32, CublasOperand::kConverted, 256);
    expect(kqv, CublasGemm::kGemmBatchedEx, GGML_TYPE_F16, CublasOperand::kDirect, 128);
    expect(strided, CublasGemm::kGemmStridedBatchedEx, GGML_TYPE_F16, CublasOperand::kDirect, 128);
    expect(packed, CublasGemm::kGemmBatchedEx, GGML_TYPE_F32, CublasOperand::kPacked, 256);
    expect(single, CublasGemm::kSgemm, GGML_TYPE_F32, CublasOperand::kDirect, 256);
    expect(bf16, CublasGemm::kGemmEx, GGML_TYPE_BF16, CublasOperand::kDirect, 256);
    expect(samples, CublasGemm::kGemmBatchedEx, GGML_TYPE_F16, CublasOperand::kDirect, 128);
    expect(loose, CublasGemm::kGemmEx, GGML_TYPE_F16, CublasOperand::kDirect, 4);

    Products products{.linear = Download(reinterpret_cast<std::uintptr_t>(linear->data),
                                         static_cast<std::size_t>(ggml_nelements(linear))),
                      .kq = Download(reinterpret_cast<std::uintptr_t>(kq->data),
                                     static_cast<std::size_t>(ggml_nelements(kq))),
                      .kqv = Download(reinterpret_cast<std::uintptr_t>(kqv->data),
                                      static_cast<std::size_t>(ggml_nelements(kqv))),
                      .strided = Download(reinterpret_cast<std::uintptr_t>(strided->data),
                                          static_cast<std::size_t>(ggml_nelements(strided))),
                      .packed = Download(reinterpret_cast<std::uintptr_t>(packed->data),
                                         static_cast<std::size_t>(ggml_nelements(packed))),
                      .single = Download(reinterpret_cast<std::uintptr_t>(single->data),
                                         static_cast<std::size_t>(ggml_nelements(single))),
                      .bf16 = Download(reinterpret_cast<std::uintptr_t>(bf16->data),
                                       static_cast<std::size_t>(ggml_nelements(bf16))),
                      .samples = Download(reinterpret_cast<std::uintptr_t>(samples->data),
                                          static_cast<std::size_t>(ggml_nelements(samples))),
                      .loose = Download(reinterpret_cast<std::uintptr_t>(loose->data),
                                        static_cast<std::size_t>(ggml_nelements(loose)))};
    return products;
  }
};

TEST_F(GgmlCublasMemoryTest, HostAndDeviceVmmMatchCudaMallocBitForBit) {
  const Products control = Run(Memory::kCudaMalloc);
  for (const Memory memory : {Memory::kDeviceVmm, Memory::kHostVmm}) {
    const Products got = Run(memory);
    EXPECT_EQ(Bits(got.linear), Bits(control.linear));
    EXPECT_EQ(Bits(got.kq), Bits(control.kq));
    EXPECT_EQ(Bits(got.kqv), Bits(control.kqv));
    EXPECT_EQ(Bits(got.strided), Bits(control.strided));
    EXPECT_EQ(Bits(got.packed), Bits(control.packed));
    EXPECT_EQ(Bits(got.single), Bits(control.single));
    EXPECT_EQ(Bits(got.bf16), Bits(control.bf16));
    EXPECT_EQ(Bits(got.samples), Bits(control.samples));
    EXPECT_EQ(Bits(got.loose), Bits(control.loose));
  }

  // CPU references in double precision, from the values the device holds.
  std::vector<double> linear(1024 * kTokens);
  for (std::int64_t t = 0; t < kTokens; ++t) {
    for (std::int64_t o = 0; o < 1024; ++o) {
      double sum = 0.0;
      for (std::int64_t c = 0; c < kWidth; ++c) {
        sum += static_cast<double>(Half(1, static_cast<std::uint64_t>((o * kWidth) + c), 0.1f)) *
               Half(2, static_cast<std::uint64_t>((t * kWidth) + c), 1.0f);
      }
      linear[static_cast<std::size_t>((t * 1024) + o)] = sum;
    }
  }
  // K[cell][kv head][d] and Q[token][head][d], as uploaded.
  const auto key = [](std::int64_t cell, std::int64_t kv_head, std::int64_t d) {
    return static_cast<double>(
        Half(3, static_cast<std::uint64_t>((cell * kHeadDim * kKvHeads) + (kv_head * kHeadDim) + d),
             1.0f));
  };
  const auto query = [](std::int64_t token, std::int64_t head, std::int64_t d) {
    return static_cast<double>(
        Value(5, static_cast<std::uint64_t>((token * kHeadDim * kHeads) + (head * kHeadDim) + d)) *
        0.125f);
  };
  std::vector<double> kq(kKv * kTokens * kHeads);
  for (std::int64_t h = 0; h < kHeads; ++h) {
    for (std::int64_t t = 0; t < kTokens; ++t) {
      for (std::int64_t cell = 0; cell < kKv; ++cell) {
        double sum = 0.0;
        for (std::int64_t d = 0; d < kHeadDim; ++d) {
          sum += key(cell, h / (kHeads / kKvHeads), d) * query(t, h, d);
        }
        kq[static_cast<std::size_t>((((h * kTokens) + t) * kKv) + cell)] = sum;
      }
    }
  }
  // KQV reads KQ as the device computed it, rounded to F16 by the path.
  std::vector<double> kqv(kHeadDim * kTokens * kHeads);
  for (std::int64_t h = 0; h < kHeads; ++h) {
    const std::int64_t g = h / (kHeads / kKvHeads);
    for (std::int64_t t = 0; t < kTokens; ++t) {
      for (std::int64_t d = 0; d < kHeadDim; ++d) {
        double sum = 0.0;
        for (std::int64_t cell = 0; cell < kKv; ++cell) {
          const double value =
              Half(4, static_cast<std::uint64_t>((((g * kHeadDim) + d) * kKvSize) + cell), 1.0f);
          sum += value *
                 ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                     control.kq[static_cast<std::size_t>((((h * kTokens) + t) * kKv) + cell)]));
        }
        kqv[static_cast<std::size_t>((((h * kTokens) + t) * kHeadDim) + d)] = sum;
      }
    }
  }
  // The same cache bytes as [head][cell][d] keys, and the query bytes as
  // [head][token][d] rows.
  std::vector<double> strided(kKv * kTokens * kKvHeads);
  for (std::int64_t h = 0; h < kKvHeads; ++h) {
    for (std::int64_t t = 0; t < kTokens; ++t) {
      for (std::int64_t cell = 0; cell < kKv; ++cell) {
        double sum = 0.0;
        for (std::int64_t d = 0; d < kHeadDim; ++d) {
          sum += static_cast<double>(Half(
                     3, static_cast<std::uint64_t>((((h * kKv) + cell) * kHeadDim) + d), 1.0f)) *
                 Value(5, static_cast<std::uint64_t>((((h * kTokens) + t) * kHeadDim) + d)) *
                 0.125f;
        }
        strided[static_cast<std::size_t>((((h * kTokens) + t) * kKv) + cell)] = sum;
      }
    }
  }
  // Heads 0 and 1 of the wider cache: K[cell][kv head][d] at a row of three.
  std::vector<double> packed(kKv * kTokens * kHeads);
  for (std::int64_t h = 0; h < kHeads; ++h) {
    for (std::int64_t t = 0; t < kTokens; ++t) {
      for (std::int64_t cell = 0; cell < kKv; ++cell) {
        double sum = 0.0;
        for (std::int64_t d = 0; d < kHeadDim; ++d) {
          const std::int64_t kv_head = h / (kHeads / kKvHeads);
          sum +=
              static_cast<double>(Half(6,
                                       static_cast<std::uint64_t>((cell * kHeadDim * kWideHeads) +
                                                                  (kv_head * kHeadDim) + d),
                                       1.0f)) *
              query(t, h, d);
        }
        packed[static_cast<std::size_t>((((h * kTokens) + t) * kKv) + cell)] = sum;
      }
    }
  }
  // The F32 matrix against the first head's rows.
  std::vector<double> single(128 * kTokens);
  for (std::int64_t t = 0; t < kTokens; ++t) {
    for (std::int64_t o = 0; o < 128; ++o) {
      double sum = 0.0;
      for (std::int64_t d = 0; d < kHeadDim; ++d) {
        sum += static_cast<double>(Value(7, static_cast<std::uint64_t>((o * kHeadDim) + d))) *
               Value(5, static_cast<std::uint64_t>((t * kHeadDim) + d)) * 0.125f;
      }
      single[static_cast<std::size_t>((t * 128) + o)] = sum;
    }
  }
  // BF16 weights and rows, accumulated in F32.
  std::vector<double> bf16(256 * kTokens);
  for (std::int64_t t = 0; t < kTokens; ++t) {
    for (std::int64_t o = 0; o < 256; ++o) {
      double sum = 0.0;
      for (std::int64_t c = 0; c < kWidth; ++c) {
        const auto w = static_cast<std::uint64_t>((o * kWidth) + c);
        const auto r = static_cast<std::uint64_t>((t * kWidth) + c);
        sum += static_cast<double>(ggml_bf16_to_fp32(ggml_fp32_to_bf16(Value(9, w) * 0.1f))) *
               ggml_bf16_to_fp32(ggml_fp32_to_bf16(Value(2, r)));
      }
      bf16[static_cast<std::size_t>((t * 256) + o)] = sum;
    }
  }
  // Sample s, channel c reads weight channel c / 2 of the same sample.
  std::vector<double> samples(128 * kTokens * 4 * 2);
  for (std::int64_t s = 0; s < 2; ++s) {
    for (std::int64_t c = 0; c < 4; ++c) {
      for (std::int64_t t = 0; t < kTokens; ++t) {
        for (std::int64_t o = 0; o < 128; ++o) {
          double sum = 0.0;
          for (std::int64_t d = 0; d < kHeadDim; ++d) {
            const auto w =
                static_cast<std::uint64_t>((((((s * 2) + (c / 2)) * 128) + o) * kHeadDim) + d);
            const auto r =
                static_cast<std::uint64_t>((((((s * 4) + c) * kTokens) + t) * kHeadDim) + d);
            sum += static_cast<double>(Half(10, w, 1.0f)) * Value(11, r);
          }
          samples[static_cast<std::size_t>((((((s * 4) + c) * kTokens) + t) * 128) + o)] = sum;
        }
      }
    }
  }
  // The loose weights hold the first 64 × 128 sample weights.
  std::vector<double> loose(128 * kTokens);
  for (std::int64_t t = 0; t < kTokens; ++t) {
    for (std::int64_t o = 0; o < 128; ++o) {
      double sum = 0.0;
      for (std::int64_t d = 0; d < kHeadDim; ++d) {
        sum += static_cast<double>(Half(10, static_cast<std::uint64_t>((o * kHeadDim) + d), 1.0f)) *
               Value(5, static_cast<std::uint64_t>((t * kHeadDim) + d)) * 0.125f;
      }
      loose[static_cast<std::size_t>((t * 128) + o)] = sum;
    }
  }
  // F16 accumulation (COMPUTE_16F) for the F16 products; TF32 for the F32
  // ones. These catch a misplaced operand, head or stride, which is off by
  // O(1).
  EXPECT_LT(Worst(control.linear, linear), 5e-2);
  EXPECT_LT(Worst(control.kq, kq), 1e-2);
  EXPECT_LT(Worst(control.kqv, kqv), 5e-2);
  EXPECT_LT(Worst(control.strided, strided), 5e-2);
  EXPECT_LT(Worst(control.packed, packed), 1e-2);
  EXPECT_LT(Worst(control.single, single), 1e-2);
  EXPECT_LT(Worst(control.bf16, bf16), 1e-2);
  EXPECT_LT(Worst(control.samples, samples), 5e-2);
  EXPECT_LT(Worst(control.loose, loose), 5e-2);
}

// GGML's pool peaks for the output head, recorded from the FP16 bridge in
// P0 (docs/experiments/backend-proof-p0/README.md, "FP16 executed plan and
// workspace"): its F16 output temporary plus the F16 copy of its input.
TEST_F(GgmlCublasTest, TheOutputHeadDrawsTheRecordedPoolPeaks) {
  auto arena = TensorArena::Create(8).value();
  ggml_tensor* head = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F16, kWidth, kVocab);
  TensorArena::Bind(head, Vmm(BackingKind::kDevice, Bytes(ggml_nbytes(head))));
  // Zeros would do; the values only have to be finite.
  const std::vector<ggml_fp16_t> zeros(static_cast<std::size_t>(kWidth) * 4096,
                                       ggml_fp32_to_fp16(0.0f));
  for (std::size_t done = 0; done < ggml_nbytes(head); done += zeros.size() * 2) {
    Upload(reinterpret_cast<std::uintptr_t>(head->data) + done, zeros.data(),
           std::min(zeros.size() * 2, ggml_nbytes(head) - done));
  }
  auto handle = Handle();
  for (const auto& [rows, peak] : {std::pair<std::int64_t, std::uint64_t>{17, 5'196'288},
                                   std::pair<std::int64_t, std::uint64_t>{32, 9'781'248}}) {
    ggml_tensor* x = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, kWidth, rows);
    TensorArena::Bind(x, Vmm(BackingKind::kDevice, Bytes(ggml_nbytes(x))));
    Upload(reinterpret_cast<std::uintptr_t>(x->data), zeros.data(), ggml_nbytes(x));
    ggml_tensor* logits = ggml_mul_mat(arena.context(), head, x);
    TensorArena::Bind(logits, Vmm(BackingKind::kDevice, Bytes(ggml_nbytes(logits))));
    auto launch = Launcher(handle.get());
    const auto plan = llmp::kernels::ggml::PlanMulMatCublas(*launch, logits);
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    EXPECT_EQ(plan->gemm, CublasGemm::kGemmEx);
    EXPECT_EQ(plan->weights, CublasOperand::kDirect);
    EXPECT_EQ(plan->input, CublasOperand::kConverted);
    EXPECT_FALSE(plan->f32_output);
    EXPECT_EQ(plan->alignment, 256U);
    EXPECT_EQ(plan->scratch, peak) << rows;
    ASSERT_TRUE(llmp::kernels::ggml::MulMatCublas(*launch, logits).has_value());
    EXPECT_EQ(launch->scratch_peak(), Bytes(peak)) << rows;
    const std::vector<float> got =
        Download(reinterpret_cast<std::uintptr_t>(logits->data), static_cast<std::size_t>(kVocab));
    EXPECT_TRUE(std::ranges::all_of(got, [](float v) { return v == 0.0f; }));
  }
}

TEST_F(GgmlCublasTest, WhatThePathCannotRunIsRefusedBeforeLaunch) {
  auto arena = TensorArena::Create(16).value();
  ggml_context* context = arena.context();
  const std::uint64_t memory = Vmm(BackingKind::kDevice, Bytes(64ULL << 20));
  ggml_tensor* w = ggml_new_tensor_2d(context, GGML_TYPE_F16, kWidth, 1024);
  ggml_tensor* x = ggml_new_tensor_2d(context, GGML_TYPE_F32, kWidth, kTokens);
  TensorArena::Bind(w, memory);
  TensorArena::Bind(x, memory + (8ULL << 20));
  ggml_tensor* y = ggml_mul_mat(context, w, x);
  TensorArena::Bind(y, memory + (16ULL << 20));

  // Without a handle GGML would create its own, with its own workspace.
  auto bare = Launcher(nullptr);
  const auto no_handle = llmp::kernels::ggml::MulMatCublas(*bare, y);
  ASSERT_FALSE(no_handle.has_value());
  EXPECT_EQ(no_handle.error().error, KernelError::kRejected);
  EXPECT_FALSE(bare->faulted());

  auto handle = Handle();
  // Scratch for the F16 input copy and output, but less than both.
  auto tight = Launcher(handle.get(), Bytes(2ULL << 20));
  ggml_tensor* wide = ggml_new_tensor_2d(context, GGML_TYPE_F32, kWidth, 1024);
  TensorArena::Bind(wide, memory + (24ULL << 20));
  ggml_tensor* big = ggml_mul_mat(context, w, wide);
  TensorArena::Bind(big, memory + (40ULL << 20));
  const auto too_big = llmp::kernels::ggml::MulMatCublas(*tight, big);
  ASSERT_FALSE(too_big.has_value());
  EXPECT_EQ(too_big.error().error, KernelError::kRejected);
  EXPECT_FALSE(tight->faulted());

  auto launch = Launcher(handle.get());
  // Sixteen rows go to MMF upstream, one to MMVF: not this path's.
  for (const std::int64_t rows : {16, 1}) {
    ggml_tensor* narrow = ggml_view_2d(context, x, kWidth, rows, x->nb[1], 0);
    ggml_tensor* product = ggml_mul_mat(context, w, narrow);
    TensorArena::Bind(product, memory + (16ULL << 20));
    const auto refused = llmp::kernels::ggml::MulMatCublas(*launch, product);
    ASSERT_FALSE(refused.has_value()) << rows;
    EXPECT_EQ(refused.error().error, KernelError::kRejected);
  }
  EXPECT_FALSE(launch->faulted());
  launch.reset();
  tight.reset();
}

}  // namespace
