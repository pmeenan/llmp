// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// GGML's CUDA launchers under jitLLM's launch context on a GB10 (label
// `gpu`; docs/backend-proof.md, P1), the memory and launch-context tests
// also on a discrete GPU the build targets (`gpu-discrete`, D-082):
// - operands in cudaMalloc memory (the control), device VMM and host VMM
//   give identical results (BP-N3, D-034), each close to a CPU reference;
// - runtime-API launches bind to the provider's context and run in order
//   with its copies on the same stream;
// - fused and unfused RMSNorm-mul, and MMVF and MMF at one column, are
//   separate implementations of one operation;
// - an operation that does not fit is refused before launch, and a launch
//   error returns as a fault instead of aborting;
// - a plan selects between fused and unfused RMSNorm-mul through the
//   implementation registry (D-053, BP-S1), each selection exact across
//   memory kinds and close to a CPU reference, and a stale or foreign
//   implementation selects no kernel (BP-S2, BP-S4).

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
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

// GGML's launcher, called directly to provoke a launch error, and its
// error hook (ggml_support.cu), to leave an error pending.
void ggml_cuda_op_rms_norm(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line,
                     const char* msg);

namespace {

using jitllm::base::Bytes;
using jitllm::kernels::ggml::KernelError;
using jitllm::kernels::ggml::LaunchContext;
using jitllm::kernels::ggml::TensorArena;
using jitllm::providers::Access;
using jitllm::providers::BackingKind;
using jitllm::providers::DeviceExecution;
using jitllm::providers::FenceState;
using jitllm::providers::StreamId;
using jitllm::providers::VmmProvider;
using jitllm::test_support::FailedCode;

constexpr std::int64_t kWidth = 896;  // Qwen2.5-0.5B's hidden size
constexpr std::int64_t kOutputs = 1024;
constexpr std::int64_t kRows = 5;  // MMF's range, below cuBLAS's
constexpr float kEps = 1e-6f;

enum class Memory : std::uint8_t { kCudaMalloc, kDeviceVmm, kHostVmm };

// Deterministic inputs in [-1, 1).
float Value(std::uint64_t seed, std::uint64_t i) {
  std::uint64_t x = (seed * 0x9E3779B97F4A7C15ULL) ^ (i + 0x632BE59BD9B4E019ULL);
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 29;
  return static_cast<float>(static_cast<double>(x >> 40) / static_cast<double>(1ULL << 23)) - 1.0f;
}

// Forwards to another provider, and reports its next Submission as an
// unknown outcome: a device fault.
class FaultingExecution final : public DeviceExecution {
 public:
  explicit FaultingExecution(DeviceExecution& inner) : inner_(inner) {}
  bool fault_next = false;

  std::expected<StreamId, jitllm::providers::Failure> CreateStream() override {
    return inner_.CreateStream();
  }
  std::expected<void, jitllm::providers::Failure> DestroyStream(StreamId stream) override {
    return inner_.DestroyStream(stream);
  }
  std::expected<void, jitllm::providers::Failure> Copy(StreamId stream, std::uint64_t destination,
                                                       std::uint64_t source, Bytes size) override {
    return inner_.Copy(stream, destination, source, size);
  }
  std::expected<jitllm::providers::NativeStream, jitllm::providers::Failure> Submission(
      StreamId stream) override {
    if (std::exchange(fault_next, false)) {
      return std::unexpected(jitllm::providers::Failure{
          .error = jitllm::providers::ProviderError::kUnknown, .detail = "a scripted fault"});
    }
    return inner_.Submission(stream);
  }
  std::expected<void, jitllm::providers::Failure> Wait(StreamId stream,
                                                       jitllm::providers::FenceId fence) override {
    return inner_.Wait(stream, fence);
  }
  std::expected<jitllm::providers::FenceId, jitllm::providers::Failure> Record(
      StreamId stream) override {
    return inner_.Record(stream);
  }
  std::expected<FenceState, jitllm::providers::Failure> Query(
      jitllm::providers::FenceId fence) override {
    return inner_.Query(fence);
  }
  std::expected<void, jitllm::providers::Failure> Release(
      jitllm::providers::FenceId fence) override {
    return inner_.Release(fence);
  }

 private:
  DeviceExecution& inner_;
};

// Bit patterns, so that +0 and -0 or two NaNs are told apart.
std::vector<std::uint32_t> Bits(const std::vector<float>& values) {
  std::vector<std::uint32_t> bits;
  bits.reserve(values.size());
  for (const float value : values) {
    bits.push_back(std::bit_cast<std::uint32_t>(value));
  }
  return bits;
}

class GgmlKernelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memory_ = std::move(jitllm::providers::cuda::OpenDeviceMemory(0).value());
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    staging_ = Vmm(BackingKind::kHost, Bytes(16ULL << 20));
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

  // Everything queued so far has run.
  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

  std::unique_ptr<LaunchContext> Launcher(Bytes workspace = Bytes(0)) {
    const std::uint64_t base =
        workspace.value() == 0 ? 0 : Allocate(Memory::kDeviceVmm, workspace.value());
    auto launch = LaunchContext::Create(0, *execution_, stream_, {.base = base, .size = workspace});
    EXPECT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    return launch ? std::move(*launch) : nullptr;
  }

  // Copies host bytes into `address` through the staging region.
  void Upload(std::uint64_t address, const void* data, std::size_t size) {
    ASSERT_LE(staging_used_ + size, 16ULL << 20);
    std::memcpy(reinterpret_cast<void*>(staging_ + staging_used_), data, size);  // NOLINT
    ASSERT_TRUE(
        execution_->Copy(stream_, address, staging_ + staging_used_, Bytes(size)).has_value());
    staging_used_ += (size + 255) / 256 * 256;
  }

  std::vector<float> Download(std::uint64_t address, std::size_t count) {
    const std::uint64_t at = staging_ + staging_used_;
    EXPECT_TRUE(execution_->Copy(stream_, at, address, Bytes(count * sizeof(float))).has_value());
    Finish();
    std::vector<float> values(count);
    std::memcpy(values.data(), reinterpret_cast<const void*>(at), count * sizeof(float));  // NOLINT
    return values;
  }

  std::unique_ptr<VmmProvider> memory_;
  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::uint64_t staging_ = 0;
  std::uint64_t staging_used_ = 0;
  std::vector<void*> malloced_;
  struct Mapped {
    jitllm::providers::ReservationId reservation;
    jitllm::providers::BackingId backing;
    Bytes size;
  };
  std::vector<Mapped> mapped_;
};

// The operands of one run, all in one kind of memory.
struct Results {
  std::vector<float> normed;   // rms_norm, then mul by the weight: two launches
  std::vector<float> fused;    // the same through GGML's fused launcher
  std::vector<float> mmf;      // kRows columns through MMF
  std::vector<float> mmvf;     // one column through MMVF
  std::vector<float> mmf_one;  // the same column through MMF
  std::vector<float> added;    // the input plus the normed rows (a residual add)
  std::vector<float> biased;   // the input plus the weight, broadcast over rows
};

class GgmlMemoryTest : public GgmlKernelsTest {
 protected:
  Results Run(Memory memory) {
    std::vector<float> input(kWidth * kRows);
    std::vector<float> weight(kWidth);
    std::vector<ggml_fp16_t> matrix(kWidth * kOutputs);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = Value(1, i) * 4.0f;
    }
    for (std::size_t i = 0; i < weight.size(); ++i) {
      weight[i] = Value(2, i) + 1.0f;
    }
    for (std::size_t i = 0; i < matrix.size(); ++i) {
      matrix[i] = ggml_fp32_to_fp16(Value(3, i) * 0.1f);
    }
    const std::size_t rows_bytes = input.size() * sizeof(float);
    const std::uint64_t input_at = Allocate(memory, rows_bytes);
    const std::uint64_t weight_at = Allocate(memory, weight.size() * sizeof(float));
    const std::uint64_t matrix_at = Allocate(memory, matrix.size() * sizeof(ggml_fp16_t));
    const std::uint64_t normed_at = Allocate(memory, rows_bytes);
    const std::uint64_t scaled_at = Allocate(memory, rows_bytes);
    const std::uint64_t fused_at = Allocate(memory, rows_bytes);
    const std::uint64_t product_at = Allocate(memory, kOutputs * kRows * sizeof(float));
    const std::uint64_t vector_at = Allocate(memory, kOutputs * sizeof(float));
    const std::uint64_t vector_mmf_at = Allocate(memory, kOutputs * sizeof(float));
    const std::uint64_t added_at = Allocate(memory, rows_bytes);
    const std::uint64_t biased_at = Allocate(memory, rows_bytes);
    Upload(input_at, input.data(), rows_bytes);
    Upload(weight_at, weight.data(), weight.size() * sizeof(float));
    Upload(matrix_at, matrix.data(), matrix.size() * sizeof(ggml_fp16_t));

    auto arena = TensorArena::Create(16).value();
    ggml_context* context = arena.context();
    ggml_tensor* x = ggml_new_tensor_2d(context, GGML_TYPE_F32, kWidth, kRows);
    ggml_tensor* w = ggml_new_tensor_1d(context, GGML_TYPE_F32, kWidth);
    ggml_tensor* m = ggml_new_tensor_2d(context, GGML_TYPE_F16, kWidth, kOutputs);
    TensorArena::Bind(x, input_at);
    TensorArena::Bind(w, weight_at);
    TensorArena::Bind(m, matrix_at);
    ggml_tensor* norm = ggml_rms_norm(context, x, kEps);
    ggml_tensor* scaled = ggml_mul(context, norm, w);
    TensorArena::Bind(norm, normed_at);
    TensorArena::Bind(scaled, scaled_at);
    ggml_tensor* fused_norm = ggml_rms_norm(context, x, kEps);
    ggml_tensor* fused = ggml_mul(context, fused_norm, w);
    TensorArena::Bind(fused, fused_at);
    // The fused launcher never writes its norm; it needs no memory.
    ggml_tensor* product = ggml_mul_mat(context, m, scaled);
    TensorArena::Bind(product, product_at);
    ggml_tensor* column = ggml_view_2d(context, scaled, kWidth, 1, scaled->nb[1], 0);
    ggml_tensor* vector = ggml_mul_mat(context, m, column);
    TensorArena::Bind(vector, vector_at);
    ggml_tensor* vector_mmf = ggml_mul_mat(context, m, column);
    TensorArena::Bind(vector_mmf, vector_mmf_at);
    ggml_tensor* added = ggml_add(context, x, scaled);
    TensorArena::Bind(added, added_at);
    ggml_tensor* biased = ggml_add(context, x, w);
    TensorArena::Bind(biased, biased_at);

    auto launch = Launcher();
    EXPECT_TRUE(jitllm::kernels::ggml::RmsNorm(*launch, norm).has_value());
    EXPECT_TRUE(jitllm::kernels::ggml::Mul(*launch, scaled).has_value());
    EXPECT_TRUE(jitllm::kernels::ggml::RmsNormMul(*launch, fused_norm, fused).has_value());
    auto mmf = jitllm::kernels::ggml::MulMatF(*launch, product);
    EXPECT_TRUE(mmf.has_value()) << (mmf ? "" : mmf.error().detail);
    auto mmvf = jitllm::kernels::ggml::MulMatVecF(*launch, vector);
    EXPECT_TRUE(mmvf.has_value()) << (mmvf ? "" : mmvf.error().detail);
    EXPECT_TRUE(jitllm::kernels::ggml::MulMatF(*launch, vector_mmf).has_value());
    EXPECT_TRUE(jitllm::kernels::ggml::Add(*launch, added).has_value());
    EXPECT_TRUE(jitllm::kernels::ggml::Add(*launch, biased).has_value());
    // MMVF is upstream's choice for one column only.
    EXPECT_EQ(FailedCode(jitllm::kernels::ggml::MulMatVecF(*launch, product)),
              KernelError::kRejected);
    EXPECT_EQ(launch->scratch_peak(), Bytes(0));

    return Results{.normed = Download(scaled_at, input.size()),
                   .fused = Download(fused_at, input.size()),
                   .mmf = Download(product_at, kOutputs * kRows),
                   .mmvf = Download(vector_at, kOutputs),
                   .mmf_one = Download(vector_mmf_at, kOutputs),
                   .added = Download(added_at, input.size()),
                   .biased = Download(biased_at, input.size())};
  }
};

// CPU references in double precision.
std::vector<double> ReferenceNorm() {
  std::vector<double> out(kWidth * kRows);
  for (std::int64_t r = 0; r < kRows; ++r) {
    double sum = 0.0;
    for (std::int64_t c = 0; c < kWidth; ++c) {
      const double v = Value(1, static_cast<std::uint64_t>((r * kWidth) + c)) * 4.0;
      sum += v * v;
    }
    const double scale = 1.0 / std::sqrt((sum / kWidth) + kEps);
    for (std::int64_t c = 0; c < kWidth; ++c) {
      const auto i = static_cast<std::uint64_t>((r * kWidth) + c);
      out[i] = Value(1, i) * 4.0 * scale * (Value(2, static_cast<std::uint64_t>(c)) + 1.0);
    }
  }
  return out;
}

void ExpectClose(const std::vector<float>& got, const std::vector<double>& want, double tolerance,
                 const char* what) {
  ASSERT_EQ(got.size(), want.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(got[i] - want[i]) / (1.0 + std::abs(want[i])));
  }
  EXPECT_LT(worst, tolerance) << what;
}

TEST_F(GgmlMemoryTest, HostAndDeviceVmmMatchCudaMallocBitForBit) {
  const Results control = Run(Memory::kCudaMalloc);
  for (const Memory memory : {Memory::kDeviceVmm, Memory::kHostVmm}) {
    const Results got = Run(memory);
    EXPECT_EQ(Bits(got.normed), Bits(control.normed));
    EXPECT_EQ(Bits(got.fused), Bits(control.fused));
    EXPECT_EQ(Bits(got.mmf), Bits(control.mmf));
    EXPECT_EQ(Bits(got.mmvf), Bits(control.mmvf));
    EXPECT_EQ(Bits(got.mmf_one), Bits(control.mmf_one));
    EXPECT_EQ(Bits(got.added), Bits(control.added));
    EXPECT_EQ(Bits(got.biased), Bits(control.biased));
  }

  const std::vector<double> norm = ReferenceNorm();
  ExpectClose(control.normed, norm, 1e-5, "rms_norm then mul");
  ExpectClose(control.fused, norm, 1e-5, "fused rms_norm-mul");
  // The products, from the CPU's own (F32) normed rows and F16 weights.
  std::vector<double> product(kOutputs * kRows);
  for (std::int64_t r = 0; r < kRows; ++r) {
    for (std::int64_t o = 0; o < kOutputs; ++o) {
      double sum = 0.0;
      for (std::int64_t c = 0; c < kWidth; ++c) {
        const auto wi = static_cast<std::uint64_t>((o * kWidth) + c);
        sum += static_cast<double>(ggml_fp16_to_fp32(ggml_fp32_to_fp16(Value(3, wi) * 0.1f))) *
               control.normed[static_cast<std::size_t>((r * kWidth) + c)];
      }
      product[static_cast<std::size_t>((r * kOutputs) + o)] = sum;
    }
  }
  // F16 activations inside the kernels: a few units in the third place.
  ExpectClose(control.mmf, product, 2e-2, "MMF");
  const std::vector<double> first(product.begin(), product.begin() + kOutputs);
  ExpectClose(control.mmvf, first, 2e-2, "MMVF");
  ExpectClose(control.mmf_one, first, 2e-2, "MMF at one column");
  // F32 additions round the same on the CPU.
  for (std::size_t i = 0; i < control.added.size(); ++i) {
    const float x = Value(1, i) * 4.0f;
    ASSERT_EQ(control.added[i], x + control.normed[i]) << i;
    ASSERT_EQ(control.biased[i], x + (Value(2, i % static_cast<std::size_t>(kWidth)) + 1.0f)) << i;
  }
}

TEST_F(GgmlKernelsTest, LaunchesUseTheProvidersContextAndStreamOrder) {
  CUdevice device = 0;
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  CUcontext primary = nullptr;
  ASSERT_EQ(cuDevicePrimaryCtxRetain(&primary, device), CUDA_SUCCESS);
  auto launch = Launcher();
  ASSERT_NE(launch, nullptr);
  EXPECT_TRUE(launch->UsesStream(*execution_, stream_));
  FaultingExecution other_provider(*execution_);
  EXPECT_FALSE(launch->UsesStream(other_provider, stream_));
  const auto other_stream = execution_->CreateStream().value();
  EXPECT_FALSE(launch->UsesStream(*execution_, other_stream));
  ASSERT_TRUE(execution_->DestroyStream(other_stream).has_value());
  CUcontext current = nullptr;
  ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
  EXPECT_EQ(current, primary);
  int runtime_device = -1;
  ASSERT_EQ(cudaGetDevice(&runtime_device), cudaSuccess);
  EXPECT_EQ(runtime_device, 0);

  // A kernel between two provider copies on the stream sees the first and
  // is seen by the second.
  const std::uint64_t rows = Allocate(Memory::kDeviceVmm, kWidth * sizeof(float));
  const std::uint64_t out = Allocate(Memory::kDeviceVmm, kWidth * sizeof(float));
  std::vector<float> ones(kWidth, 3.0f);
  auto arena = TensorArena::Create(4).value();
  ggml_tensor* x = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, kWidth);
  TensorArena::Bind(x, rows);
  ggml_tensor* norm = ggml_rms_norm(arena.context(), x, 0.0f);
  TensorArena::Bind(norm, out);
  Upload(rows, ones.data(), ones.size() * sizeof(float));
  ASSERT_TRUE(jitllm::kernels::ggml::RmsNorm(*launch, norm).has_value());
  const std::vector<float> got = Download(out, kWidth);
  for (const float v : got) {
    EXPECT_FLOAT_EQ(v, 1.0f);
  }
  ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
  EXPECT_EQ(current, primary);
  EXPECT_EQ(cuDevicePrimaryCtxRelease(device), CUDA_SUCCESS);

  // Every run is queued work the provider knows of: once earlier work is
  // fenced and released, a later run again keeps the stream from being
  // destroyed until a fence covers it.
  Finish();
  ASSERT_TRUE(jitllm::kernels::ggml::RmsNorm(*launch, norm).has_value());
  EXPECT_FALSE(execution_->DestroyStream(stream_).has_value());
}

TEST_F(GgmlKernelsTest, AnUnknownSubmissionFaultsTheContext) {
  FaultingExecution faulting(*execution_);
  auto launch = LaunchContext::Create(0, faulting, stream_, {.base = 0, .size = Bytes(0)});
  ASSERT_TRUE(launch.has_value()) << launch.error().detail;
  faulting.fault_next = true;
  const auto failed = (*launch)->Run(Bytes(0), [](ggml_backend_cuda_context&) {});
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().error, KernelError::kUnknown);
  EXPECT_TRUE((*launch)->faulted());
  EXPECT_EQ(FailedCode((*launch)->Run(Bytes(0), [](ggml_backend_cuda_context&) {})),
            KernelError::kRejected);
}

TEST_F(GgmlKernelsTest, AnErrorPendingOnTheThreadRefusesOneContext) {
  ggml_cuda_error("stmt", "func", "file", 1, "a pending error");
  auto refused = LaunchContext::Create(0, *execution_, stream_, {.base = 0, .size = Bytes(0)});
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, KernelError::kRejected);
  EXPECT_NE(refused.error().detail.find("a pending error"), std::string::npos)
      << refused.error().detail;
  EXPECT_NE(Launcher(), nullptr);  // taken, the error refuses nothing more
}

// BP-A4's negative control (invariant 1): a GGML kernel launched over
// device VMM whose backing has been unmapped faults; it never reads what
// that memory held. The fault ends the CUDA context, so it runs in a child
// process: the run over mapped memory completes, the input's backing is
// unmapped, and the same run's fence then reports the fault.
class GgmlStaleMemoryDeathTest : public GgmlKernelsTest {
 protected:
  void RunOverUnmappedBacking() {
    const std::uint64_t rows = Allocate(Memory::kDeviceVmm, kWidth * sizeof(float));
    const Mapped input = mapped_.back();
    const std::uint64_t out = Allocate(Memory::kDeviceVmm, kWidth * sizeof(float));
    auto launch = Launcher();
    auto arena = TensorArena::Create(4).value();
    ggml_tensor* x = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, kWidth);
    TensorArena::Bind(x, rows);
    ggml_tensor* norm = ggml_rms_norm(arena.context(), x, kEps);
    TensorArena::Bind(norm, out);
    const std::vector<float> ones(kWidth, 1.0F);
    Upload(rows, ones.data(), ones.size() * sizeof(float));
    const auto near_one = [](const std::vector<float>& values) {
      return std::ranges::all_of(values, [](float v) { return std::fabs(v - 1.0F) < 1e-5F; });
    };
    if (!launch || !jitllm::kernels::ggml::RmsNorm(*launch, norm).has_value() ||
        !near_one(Download(out, kWidth))) {
      std::cerr << "the run over mapped memory failed\n";
      std::_Exit(2);
    }
    if (!memory_->Unmap(input.reservation, Bytes(0), input.size).has_value()) {
      std::cerr << "the unmap failed\n";
      std::_Exit(3);
    }
    std::ignore = jitllm::kernels::ggml::RmsNorm(*launch, norm);  // queued; faults on the device
    const auto fence = execution_->Record(stream_);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (fence && std::chrono::steady_clock::now() < deadline) {
      const auto state = execution_->Query(*fence);
      if (!state) {
        std::cerr << "faulted: " << state.error().detail << "\n";
        std::_Exit(0);
      }
      if (*state == FenceState::kComplete) {
        std::cerr << "completed over unmapped backing\n";
        std::_Exit(4);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    std::cerr << (fence ? "no result" : "faulted at the record") << "\n";
    std::_Exit(fence ? 5 : 0);
  }
};

TEST_F(GgmlStaleMemoryDeathTest, AKernelOverUnmappedBackingFaults) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(RunOverUnmappedBacking(), ::testing::ExitedWithCode(0), "faulted");
}

TEST_F(GgmlKernelsTest, WhatDoesNotFitIsRefusedAndALaunchErrorIsAFault) {
  auto launch = Launcher();
  ASSERT_NE(launch, nullptr);
  // No workspace: an operation that needs scratch is refused.
  EXPECT_EQ(FailedCode(launch->Run(Bytes(1), [](ggml_backend_cuda_context&) {})),
            KernelError::kRejected);

  // 70,000 channels exceed the row kernel's grid (65,535).
  constexpr std::int64_t kChannels = 70000;
  const std::uint64_t rows = Allocate(Memory::kDeviceVmm, kChannels * sizeof(float));
  auto arena = TensorArena::Create(4).value();
  ggml_tensor* x = ggml_new_tensor_3d(arena.context(), GGML_TYPE_F32, 1, 1, kChannels);
  TensorArena::Bind(x, rows);
  ggml_tensor* norm = ggml_rms_norm(arena.context(), x, kEps);
  TensorArena::Bind(norm, rows);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::RmsNorm(*launch, norm)), KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());

  // Operands GGML's launchers would abort on, or index past: none is
  // launched. Nothing here reaches the device, so one address serves all.
  auto shapes = TensorArena::Create(32).value();
  ggml_context* context = shapes.context();
  ggml_tensor* empty = ggml_new_tensor_1d(context, GGML_TYPE_F32, 0);
  ggml_tensor* strided = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2 * (kWidth + 1));
  ggml_tensor* matrix = ggml_new_tensor_2d(context, GGML_TYPE_F16, kWidth, kOutputs);
  for (ggml_tensor* tensor : {empty, strided, matrix}) {
    TensorArena::Bind(tensor, rows);
  }
  ggml_tensor* sum = ggml_add(context, empty, empty);  // divides by its extents
  TensorArena::Bind(sum, rows);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::Add(*launch, sum)), KernelError::kRejected);
  // Two rows at a stride of kWidth + 1: normalized in place, the kernel
  // would write them densely.
  ggml_tensor* spaced = ggml_view_2d(context, strided, kWidth, 2, (kWidth + 1) * sizeof(float), 0);
  ggml_tensor* in_place = ggml_rms_norm_inplace(context, spaced, kEps);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::RmsNorm(*launch, in_place)), KernelError::kRejected);
  // An odd activation column stride, which MMVF's launcher asserts on: MMVF
  // loads float2 pairs, so the alignment check refuses it first.
  ggml_tensor* column = ggml_view_2d(context, strided, kWidth, 1, (kWidth + 1) * sizeof(float), 0);
  ggml_tensor* vector = ggml_mul_mat(context, matrix, column);
  TensorArena::Bind(vector, rows);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::MulMatVecF(*launch, vector)), KernelError::kRejected);
  // A misaligned operand, which would be a sticky fault for the process.
  ggml_tensor* odd = ggml_new_tensor_1d(context, GGML_TYPE_F32, kWidth);
  TensorArena::Bind(odd, rows + 2);
  ggml_tensor* odd_norm = ggml_rms_norm(context, odd, kEps);
  TensorArena::Bind(odd_norm, rows);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::RmsNorm(*launch, odd_norm)), KernelError::kRejected);
  // A norm scaled by itself: the fused kernel would read the norm it never
  // writes.
  ggml_tensor* self_norm = ggml_rms_norm(context, strided, kEps);
  ggml_tensor* squared = ggml_mul(context, self_norm, self_norm);
  TensorArena::Bind(self_norm, rows);
  TensorArena::Bind(squared, rows + 8192U);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::RmsNormMul(*launch, self_norm, squared)),
            KernelError::kRejected);
  // In place into a transposed view: the kernel would write it densely.
  ggml_tensor* square = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 4);
  ggml_tensor* addend = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 4);
  TensorArena::Bind(square, rows);
  TensorArena::Bind(addend, rows + 4096U);
  ggml_tensor* transposed_sum = ggml_add_inplace(context, ggml_transpose(context, square), addend);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::Add(*launch, transposed_sum)),
            KernelError::kRejected);
  // An output one row into its input.
  ggml_tensor* two_rows = ggml_new_tensor_2d(context, GGML_TYPE_F32, kWidth, 2);
  TensorArena::Bind(two_rows, rows);
  ggml_tensor* shifted = ggml_rms_norm(context, two_rows, kEps);
  TensorArena::Bind(shifted, rows + (kWidth * sizeof(float)));
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::RmsNorm(*launch, shifted)), KernelError::kRejected);
  // An odd channel stride: MMVF loads activations as float2 from every
  // channel's offset.
  ggml_tensor* channels = ggml_view_3d(context, strided, kWidth, 1, 2, kWidth * sizeof(float),
                                       (kWidth + 1) * sizeof(float), 0);
  ggml_tensor* per_channel = ggml_mul_mat(context, matrix, channels);
  TensorArena::Bind(per_channel, rows + 65536U);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::MulMatVecF(*launch, per_channel)),
            KernelError::kRejected);
  // A dimension of one with an unpacked stride: GGML counts it contiguous,
  // and the broadcast launcher's dimension merging would misindex it.
  ggml_tensor* loose = ggml_view_4d(context, strided, 4, 1, 2, 1, 8, 16, 32, 0);
  ggml_tensor* four = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
  TensorArena::Bind(four, rows + 4096U);
  ggml_tensor* loose_sum = ggml_add(context, loose, four);
  TensorArena::Bind(loose_sum, rows + 65536U);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::Add(*launch, loose_sum)), KernelError::kRejected);
  // Strides that wrap: ggml_nbytes sums them to a few hundred bytes, while
  // the kernel would read gigabytes away.
  ggml_tensor* small = ggml_new_tensor_2d(context, GGML_TYPE_F16, 64, 16);
  TensorArena::Bind(small, rows);
  ggml_tensor* wrapped =
      ggml_view_4d(context, strided, 64, 1, 2, 2, 256,
                   (~std::size_t{0} - (std::size_t{1} << 33)) + 1, (std::size_t{1} << 33) + 256, 0);
  ggml_tensor* wrapped_product = ggml_mul_mat(context, small, wrapped);
  TensorArena::Bind(wrapped_product, rows + 65536U);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::MulMatVecF(*launch, wrapped_product)),
            KernelError::kRejected);
  // A view made before its source was bound again keeps the old address.
  ggml_tensor* source = ggml_new_tensor_1d(context, GGML_TYPE_F32, kWidth);
  TensorArena::Bind(source, rows);
  ggml_tensor* stale = ggml_view_1d(context, source, kWidth, 0);
  TensorArena::Bind(source, rows + 4096U);
  ggml_tensor* stale_norm = ggml_rms_norm(context, stale, kEps);
  TensorArena::Bind(stale_norm, rows + 65536U);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::RmsNorm(*launch, stale_norm)),
            KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());

  // GGML's launcher itself, past the check: the launch fails, and the
  // error comes back instead of aborting the process.
  const auto failed = launch->Run(Bytes(0), [norm](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm(context, norm);
  });
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().error, KernelError::kUnknown);
  // The detail names the failed CUDA call and where GGML made it.
  EXPECT_NE(failed.error().detail.find("common.cuh"), std::string::npos) << failed.error().detail;
  EXPECT_TRUE(launch->faulted());
  EXPECT_EQ(FailedCode(launch->Run(Bytes(0), [](ggml_backend_cuda_context&) {})),
            KernelError::kRejected);
}

// Plan selection between two implementations of one operation (D-053).
// Operands of one RMSNorm-mul: rows, a weight, the unfused norm's
// intermediate and the output.
struct NormShape {
  std::int64_t width = 0;
  std::int64_t rows = 0;
};

// A quiet NaN that no kernel writes.
constexpr std::uint32_t kPoison = 0x7fc00badU;

// What one selection wrote.
struct NormResult {
  std::vector<float> output;
  std::vector<float> intermediate;
};

class GgmlPlanTest : public GgmlKernelsTest {
 protected:
  // RMSNorm-mul by the one operation of `plan`, resolved against
  // `registry`, over inputs in `memory`; output and intermediate start as
  // NaN, so whatever is not written shows.
  NormResult Run(const jitllm::execution::Registry& registry, const jitllm::execution::Plan& plan,
                 NormShape shape, Memory memory) {
    const auto bound = jitllm::execution::Resolve(registry, plan);
    EXPECT_TRUE(bound.has_value()) << (bound ? "" : bound.error().detail);
    if (!bound) {
      return {};
    }
    const auto kernel = jitllm::kernels::ggml::RmsNormMulKernel::Bind(bound->at(0));
    EXPECT_TRUE(kernel.has_value()) << (kernel ? "" : kernel.error().detail);
    if (!kernel) {
      return {};
    }
    const auto count = static_cast<std::size_t>(shape.width * shape.rows);
    std::vector<float> input(count);
    std::vector<float> weight(static_cast<std::size_t>(shape.width));
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = Value(1, i) * 4.0f;
    }
    for (std::size_t i = 0; i < weight.size(); ++i) {
      weight[i] = Value(2, i) + 1.0f;
    }
    const std::vector<float> poison(count, std::bit_cast<float>(kPoison));
    const std::size_t bytes = count * sizeof(float);
    const std::uint64_t input_at = Allocate(memory, bytes);
    const std::uint64_t weight_at = Allocate(memory, weight.size() * sizeof(float));
    const std::uint64_t intermediate_at = Allocate(memory, bytes);
    const std::uint64_t output_at = Allocate(memory, bytes);
    Upload(input_at, input.data(), bytes);
    Upload(weight_at, weight.data(), weight.size() * sizeof(float));
    Upload(intermediate_at, poison.data(), bytes);
    Upload(output_at, poison.data(), bytes);

    auto arena = TensorArena::Create(4).value();
    ggml_context* context = arena.context();
    ggml_tensor* x = ggml_new_tensor_2d(context, GGML_TYPE_F32, shape.width, shape.rows);
    ggml_tensor* w = ggml_new_tensor_1d(context, GGML_TYPE_F32, shape.width);
    TensorArena::Bind(x, input_at);
    TensorArena::Bind(w, weight_at);
    ggml_tensor* norm = ggml_rms_norm(context, x, kEps);
    ggml_tensor* scaled = ggml_mul(context, norm, w);
    TensorArena::Bind(norm, intermediate_at);
    TensorArena::Bind(scaled, output_at);

    auto launch = Launcher();
    EXPECT_TRUE(kernel->Check(norm, scaled).has_value());
    const auto ran = kernel->Run(*launch, norm, scaled);
    EXPECT_TRUE(ran.has_value()) << (ran ? "" : ran.error().detail);
    EXPECT_EQ(launch->scratch_peak(), Bytes(0));
    return {.output = Download(output_at, count), .intermediate = Download(intermediate_at, count)};
  }
};

// The CPU reference in double precision.
std::vector<double> ReferenceNormMul(NormShape shape) {
  std::vector<double> out(static_cast<std::size_t>(shape.width * shape.rows));
  for (std::int64_t r = 0; r < shape.rows; ++r) {
    double sum = 0.0;
    for (std::int64_t c = 0; c < shape.width; ++c) {
      const double v = Value(1, static_cast<std::uint64_t>((r * shape.width) + c)) * 4.0;
      sum += v * v;
    }
    const double scale = 1.0 / std::sqrt((sum / static_cast<double>(shape.width)) + kEps);
    for (std::int64_t c = 0; c < shape.width; ++c) {
      const auto i = static_cast<std::uint64_t>((r * shape.width) + c);
      out[i] = Value(1, i) * 4.0 * scale * (Value(2, static_cast<std::uint64_t>(c)) + 1.0);
    }
  }
  return out;
}

const char* const kFused = "ggml.rms_norm_mul.fused";
const char* const kUnfused = "ggml.rms_norm_mul.unfused";

jitllm::execution::Plan NormPlan(const jitllm::execution::Registry& registry, const char* name) {
  const std::vector<jitllm::execution::Choice> choices = {
      {.operation = jitllm::execution::Operation::kRmsNormMul, .implementation = name}};
  return jitllm::execution::Plan::Build(registry, choices).value();
}

TEST_F(GgmlPlanTest, EachSelectionIsExactAcrossMemoryAndCloseToTheReference) {
  const auto registry =
      jitllm::execution::Registry::Create(jitllm::kernels::ggml::Implementations()).value();
  std::size_t candidates = 0;
  for (std::size_t i = 0; i < registry.size(); ++i) {
    candidates += registry.at(i).operation == jitllm::execution::Operation::kRmsNormMul ? 1 : 0;
  }
  EXPECT_EQ(candidates, 2U);
  const auto fused = NormPlan(registry, kFused);
  const auto unfused = NormPlan(registry, kUnfused);
  EXPECT_NE(fused.identity(), unfused.identity());

  // Qwen2.5-0.5B's width, which GGML's launchers run in 32-thread blocks,
  // and a width they run in 1,024-thread blocks.
  for (const NormShape shape :
       {NormShape{.width = kWidth, .rows = kRows}, NormShape{.width = 4096, .rows = 3}}) {
    const NormResult fused_control = Run(registry, fused, shape, Memory::kCudaMalloc);
    const NormResult unfused_control = Run(registry, unfused, shape, Memory::kCudaMalloc);
    for (const Memory memory : {Memory::kDeviceVmm, Memory::kHostVmm}) {
      EXPECT_EQ(Bits(Run(registry, fused, shape, memory).output), Bits(fused_control.output));
      EXPECT_EQ(Bits(Run(registry, unfused, shape, memory).output), Bits(unfused_control.output));
    }
    // Each against the reference, as GgmlMemoryTest judges them.
    const std::vector<double> reference = ReferenceNormMul(shape);
    ExpectClose(fused_control.output, reference, 1e-5, kFused);
    ExpectClose(unfused_control.output, reference, 1e-5, kUnfused);
    // The fused kernel never writes the intermediate; the unfused one does.
    EXPECT_TRUE(std::ranges::all_of(fused_control.intermediate, [](float v) {
      return std::bit_cast<std::uint32_t>(v) == kPoison;
    }));
    EXPECT_TRUE(
        std::ranges::none_of(unfused_control.intermediate, [](float v) { return std::isnan(v); }));

    // Fused against unfused: reported, not judged (BP-S1 judges each
    // against its own bridge arm, in P2).
    std::size_t differ = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < fused_control.output.size(); ++i) {
      const float a = fused_control.output[i];
      const float b = unfused_control.output[i];
      if (std::bit_cast<std::uint32_t>(a) != std::bit_cast<std::uint32_t>(b)) {
        ++differ;
        worst = std::max(worst, std::abs(static_cast<double>(a) - b) / (1.0 + std::abs(b)));
      }
    }
    std::cout << "rms_norm_mul " << shape.width << " x " << shape.rows << ": fused and unfused "
              << "differ in " << differ << " of " << fused_control.output.size()
              << " elements, largest relative difference " << worst << "\n";
    RecordProperty("differ_" + std::to_string(shape.width), static_cast<int>(differ));
  }
}

TEST_F(GgmlPlanTest, AStaleOrForeignImplementationSelectsNoKernel) {
  using jitllm::execution::PlanError;
  const auto registry =
      jitllm::execution::Registry::Create(jitllm::kernels::ggml::Implementations()).value();

  // A plan made in a build whose fused implementation had another GGML
  // tree (BP-S2): stale here, and never run as the unfused one.
  std::vector<jitllm::execution::Implementation> older = jitllm::kernels::ggml::Implementations();
  for (auto& implementation : older) {
    implementation.revision = "0000";
  }
  const auto old_registry = jitllm::execution::Registry::Create(older).value();
  const auto stale = jitllm::execution::Resolve(registry, NormPlan(old_registry, kFused));
  ASSERT_FALSE(stale.has_value());
  EXPECT_EQ(stale.error().error, PlanError::kStale);
  // Nor does the stale declaration itself select a kernel.
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::RmsNormMulKernel::Bind(older[0])),
            KernelError::kRejected);

  // A plan naming an implementation of a module this build lacks (BP-S4).
  std::vector<jitllm::execution::Implementation> with_module =
      jitllm::kernels::ggml::Implementations();
  with_module.push_back({.name = "module.rms_norm_mul.other",
                         .operation = jitllm::execution::Operation::kRmsNormMul,
                         .source = "module",
                         .revision = "1",
                         .build = "1",
                         .variant = "1"});
  const auto module_registry = jitllm::execution::Registry::Create(with_module).value();
  const auto unsupported =
      jitllm::execution::Resolve(registry, NormPlan(module_registry, "module.rms_norm_mul.other"));
  ASSERT_FALSE(unsupported.has_value());
  EXPECT_EQ(unsupported.error().error, PlanError::kUnsupported);
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::RmsNormMulKernel::Bind(with_module.back())),
            KernelError::kRejected);

  // A bound kernel refuses operands it cannot take, before any launch: the
  // unfused norm needs memory of its own.
  const auto unfused = jitllm::execution::Resolve(registry, NormPlan(registry, kUnfused)).value();
  const auto kernel = jitllm::kernels::ggml::RmsNormMulKernel::Bind(unfused.at(0)).value();
  EXPECT_EQ(kernel.name(), kUnfused);
  const std::uint64_t rows = Allocate(Memory::kDeviceVmm, 3 * kWidth * sizeof(float));
  auto arena = TensorArena::Create(4).value();
  ggml_tensor* x = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, kWidth);
  ggml_tensor* w = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, kWidth);
  TensorArena::Bind(x, rows);
  TensorArena::Bind(w, rows + (kWidth * sizeof(float)));
  ggml_tensor* norm = ggml_rms_norm(arena.context(), x, kEps);
  ggml_tensor* scaled = ggml_mul(arena.context(), norm, w);
  TensorArena::Bind(scaled, rows + (2 * kWidth * sizeof(float)));
  auto launch = Launcher();
  EXPECT_EQ(FailedCode(kernel.Run(*launch, norm, scaled)), KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());
}

}  // namespace
