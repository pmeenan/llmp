// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The remaining operations of the FP16 bridge's recorded plan under
// jitLLM's launch context on a GB10 (label `gpu`; docs/backend-proof.md,
// P1; kernels/ggml/ops.h):
// - each is bit-identical in cudaMalloc memory, device VMM and host VMM
//   (BP-N3, D-034), and close to a CPU reference;
// - at Qwen2.5-0.5B's shapes each launches exactly what the bridge's
//   recorded plan launched there (docs/experiments/backend-proof-p0/
//   fp16-plan.json): the same kernel, grid, block and dynamic shared
//   memory, or the same copy, on the context's stream;
// - upstream's device-dependent fusion gate agrees with the fused kernels'
//   own selection, and the registry declares, binds and runs every new
//   implementation (D-053).
//
// Launches are recorded by tests/support's launch recorder, which wraps the
// CUDA runtime's launch and copy entry points when this test links
// (--wrap). GGML's `<<<>>>` launches reach the runtime through
// __cudaLaunchKernel, its programmatic-dependent launches through
// cudaLaunchKernelExC. The PDL attribute those carry is not compared, as
// the FP16 gate allows. The first seven launches of the fused decode step
// are also written as the recorder's JSON lines and must equal
// plan_record_sample.txt, which plan_compare.py matches with the record;
// with JITLLM_TEST_PLAN_RECORD set they are also written to that file.

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "launch_recorder.h"
#include "plan_record.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace {

using jitllm::base::Bytes;
using jitllm::kernels::ggml::KernelError;
using jitllm::kernels::ggml::KernelFailure;
using jitllm::kernels::ggml::LaunchContext;
using jitllm::kernels::ggml::TensorArena;
using jitllm::providers::Access;
using jitllm::providers::BackingKind;
using jitllm::providers::DeviceExecution;
using jitllm::providers::FenceState;
using jitllm::providers::StreamId;
using jitllm::providers::VmmProvider;
using jitllm::test_support::Event;
using jitllm::test_support::EventKind;
using jitllm::test_support::FailedCode;

// Qwen2.5-0.5B's shapes.
constexpr std::int64_t kWidth = 896;
constexpr std::int64_t kHead = 64;
constexpr std::int64_t kHeads = 14;
constexpr std::int64_t kKvHeads = 2;
constexpr std::int64_t kKvWidth = kHead * kKvHeads;
constexpr std::int64_t kFfn = 4864;
constexpr std::int64_t kCells = 1024;
constexpr float kRopeBase = 1000000.0f;
constexpr float kScale = 0.125f;  // 64^-1/2

// The model's RoPE, as llama.cpp calls ggml_rope_ext for it.
ggml_tensor* QwenRope(ggml_context* context, ggml_tensor* x, ggml_tensor* positions) {
  return ggml_rope_ext(context, x, positions, nullptr, static_cast<int>(kHead), GGML_ROPE_TYPE_NEOX,
                       32768, kRopeBase, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

enum class Memory : std::uint8_t { kCudaMalloc, kDeviceVmm, kHostVmm };

// Deterministic inputs in [-1, 1).
float Value(std::uint64_t seed, std::uint64_t i) {
  std::uint64_t x = (seed * 0x9E3779B97F4A7C15ULL) ^ (i + 0x632BE59BD9B4E019ULL);
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 29;
  return static_cast<float>(static_cast<double>(x >> 40) / static_cast<double>(1ULL << 23)) - 1.0f;
}

std::vector<float> Values(std::uint64_t seed, std::size_t count, float scale = 1.0f) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = Value(seed, i) * scale;
  }
  return values;
}

std::vector<ggml_fp16_t> Halves(std::uint64_t seed, std::size_t count, float scale) {
  std::vector<ggml_fp16_t> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = ggml_fp32_to_fp16(Value(seed, i) * scale);
  }
  return values;
}

// Bit patterns, so that +0 and -0 or two NaNs are told apart.
std::vector<std::uint32_t> Bits(const std::vector<float>& values) {
  std::vector<std::uint32_t> bits;
  bits.reserve(values.size());
  for (const float value : values) {
    bits.push_back(std::bit_cast<std::uint32_t>(value));
  }
  return bits;
}

void ExpectClose(const std::vector<float>& got, const std::vector<double>& want, double tolerance,
                 const char* what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(got[i] - want[i]) / (1.0 + std::abs(want[i])));
  }
  EXPECT_LT(worst, tolerance) << what;
  std::cout << what << ": largest relative difference from the reference " << worst << "\n";
}

// The recorded plan's kernels these operations launch, by their id in
// fp16-plan.json's kernel table, with the mangled names recorded there.
std::string_view PlanKernel(int id) {
  switch (id) {
    case 3:
      return "_Z10cpy_scalarIXadL_ZN36_INTERNAL_ae89f889_6_cpy_cu_bfba63e212cpy_1_"
             "scalarIffEEvPKcPcE"
             "EEvS3_S4_lllllllllllllll";
    case 4:
      return "_Z11k_bin_bcastIXadL_ZN42_INTERNAL_d5c41c42_11_binbcast_cu_6840010b6op_"
             "addEffEEfffJPKf"
             "EEvPKT0_PKT1_PT2_jjj5uint3SB_SB_SB_SB_jjjjjjjjjjjDpT3_";
    case 6:
      return "_Z16k_get_rows_floatIffEvPKT_PKiPT0_ll5uint3mmmmmmmmm";
    case 7:
      return "_Z20k_get_rows_float_vecIfEvPKT_PKiPS0_ll5uint3mmmmmmmmm";
    case 8:
      return "_Z10k_set_rowsIfl6__halfEvPKT_PKT0_PT1_llllllllllllll5uint3S9_S9_S9_S9_";
    case 16:
      return "_Z13mul_mat_vec_fI6__halfS0_Li1ELi224ELb0ELb0EEvPKT_PKfPKi31ggml_cuda_mm_fusion_args_"
             "devicePfi5uint3iiiSA_iiiSA_iiii";
    case 17:
      return "_Z13mul_mat_vec_fI6__halfS0_Li1ELi224ELb1ELb0EEvPKT_PKfPKi31ggml_cuda_mm_fusion_args_"
             "devicePfi5uint3iiiSA_iiiSA_iiii";
    case 18:
      return "_Z13mul_mat_vec_fI6__halfS0_Li1ELi256ELb0ELb0EEvPKT_PKfPKi31ggml_cuda_mm_fusion_args_"
             "devicePfi5uint3iiiSA_iiiSA_iiii";
    case 19:
      return "_Z13mul_mat_vec_fI6__halfS0_Li1ELi256ELb1ELb0EEvPKT_PKfPKi31ggml_cuda_mm_fusion_args_"
             "devicePfi5uint3iiiSA_iiiSA_iiii";
    case 20:
      return "_Z13mul_mat_vec_fI6__halffLi1ELi224ELb1ELb0EEvPKT_PKfPKi31ggml_cuda_mm_fusion_args_"
             "devicePfi5uint3iiiSA_iiiSA_iiii";
    case 24:
      return "_Z9rope_neoxILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiiiiiiiPKifff14rope_corr_dimsfPKfPKlib";
    case 25:
      return "_Z9rope_neoxILb1ELb0EffEvPKT1_PT2_iiiiiiiiiiiPKifff14rope_corr_dimsfPKfPKlib";
    case 26:
      return "_Z12soft_max_f32ILb1ELi0ELi0EfEvPKfPKT2_S1_Pf15soft_max_params";
    case 27:
      return "_Z12soft_max_f32ILb1ELi256ELi256EfEvPKfPKT2_S1_Pf15soft_max_params";
    case 28:
      return "_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_b65f5324_8_unary_cu_eb6d53667op_"
             "siluEfEEf"
             "EvPKT0_S3_PS1_llll";
    default:
      return "";
  }
}

// One launch of the recorded plan: a kernel by id with its grid, block and
// dynamic shared memory, or (id 0) a device-to-device copy of `bytes`.
struct Planned {
  int id = 0;
  std::array<unsigned, 3> grid{};
  std::array<unsigned, 3> block{};
  std::size_t shared = 0;
  std::size_t bytes = 0;
};

Planned PlanLaunch(int id, std::array<unsigned, 3> grid, std::array<unsigned, 3> block,
                   std::size_t shared = 0) {
  return {.id = id, .grid = grid, .block = block, .shared = shared, .bytes = 0};
}

Planned PlanCopy(std::size_t bytes) {
  return {.id = 0, .grid = {}, .block = {}, .shared = 0, .bytes = bytes};
}

class GgmlOpsTest : public ::testing::Test {
 protected:
  static constexpr std::uint64_t kStaging = 64ULL << 20;

  void SetUp() override {
    memory_ = std::move(jitllm::providers::cuda::OpenDeviceMemory(0).value());
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    staging_ = Vmm(BackingKind::kHost, Bytes(kStaging));
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

  // Everything queued so far has run; the staging region is free again.
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
    staging_used_ = 0;
  }

  std::unique_ptr<LaunchContext> Launcher() {
    auto launch = LaunchContext::Create(0, *execution_, stream_, {.base = 0, .size = Bytes(0)});
    EXPECT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    return launch ? std::move(*launch) : nullptr;
  }

  // Copies host bytes into `address` through the staging region.
  void Upload(std::uint64_t address, const void* data, std::size_t size) {
    if (staging_used_ + size > kStaging) {
      Finish();
    }
    ASSERT_LE(size, kStaging);
    std::memcpy(reinterpret_cast<void*>(staging_ + staging_used_), data, size);  // NOLINT
    ASSERT_TRUE(
        execution_->Copy(stream_, address, staging_ + staging_used_, Bytes(size)).has_value());
    staging_used_ += (size + 255) / 256 * 256;
  }

  // Binds `tensor` to new memory of `memory` kind, holding `data` if given.
  template <typename T = float>
  ggml_tensor* Place(Memory memory, ggml_tensor* tensor, const std::vector<T>& data = {}) {
    const std::size_t size = ggml_nbytes(tensor);
    TensorArena::Bind(tensor, Allocate(memory, size));
    if (!data.empty()) {
      EXPECT_EQ(data.size() * sizeof(T), size);
      Upload(reinterpret_cast<std::uintptr_t>(tensor->data), data.data(), size);
    }
    return tensor;
  }

  template <typename T = float>
  std::vector<T> Download(const ggml_tensor* tensor) {
    const std::size_t size = ggml_nbytes(tensor);
    Finish();
    EXPECT_LE(size, kStaging);
    EXPECT_TRUE(
        execution_
            ->Copy(stream_, staging_, reinterpret_cast<std::uintptr_t>(tensor->data), Bytes(size))
            .has_value());
    Finish();
    std::vector<T> values(size / sizeof(T));
    std::memcpy(values.data(), reinterpret_cast<const void*>(staging_), size);  // NOLINT
    return values;
  }

  // What `run` launched, on this context's stream.
  template <typename Run>
  std::vector<Event> Record(Run&& run) {
    std::vector<Event> launched;
    {
      jitllm::test_support::Recording recording;
      std::forward<Run>(run)();
      launched = recording.Take();
    }
    const void* native = execution_->Submission(stream_).value().handle;
    for (const Event& launch : launched) {
      EXPECT_EQ(launch.stream, native) << launch.name;
    }
    return launched;
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

void Launched(const std::expected<void, KernelFailure>& result, const char* what) {
  EXPECT_TRUE(result.has_value()) << what << ": " << (result ? "" : result.error().detail);
}

// Every operation's outputs in one kind of memory.
struct Outputs {
  std::vector<float> picked;               // get_rows, the scalar kernel
  std::vector<float> picked_vec;           // get_rows, the vector kernel
  std::vector<ggml_fp16_t> cache;          // set_rows into a poisoned F16 cache
  std::vector<float> rotated;              // RoPE of Q
  std::vector<ggml_fp16_t> fused_cache;    // RoPE of K fused with its write
  std::vector<ggml_fp16_t> unfused_cache;  // RoPE of K, then set_rows
  std::vector<float> probabilities;        // soft_max over 256 cells
  std::vector<float> probabilities_wide;   // over 768 cells
  std::vector<float> merged;               // cont of permuted heads
  std::vector<float> merged_row;           // the same for one row: a copy
  std::vector<float> glu;                  // SwiGLU
  std::vector<float> biased;               // MMVF fused with a bias
  std::vector<float> biased_unfused;       // MMVF, then add
  std::vector<float> gated;                // MMVF fused gate, up and SwiGLU
  std::vector<float> gated_unfused;        // MMVF twice, then SwiGLU
};

constexpr std::int64_t kRows = 5;
constexpr std::array<std::int32_t, kRows> kPositions = {0, 1, 5, 76, 576};
constexpr std::array<std::int32_t, kRows> kPicks = {7, 0, 3, 3, 6};
constexpr std::array<std::int64_t, kRows> kSlots = {9, 2, 15, 0, 4};
constexpr std::uint16_t kPoisonHalf = 0x7e00;  // an F16 NaN no kernel writes

TEST_F(GgmlOpsTest, F16ApeRowsWidenExactlyAcrossMemoryDomains) {
  const auto source = Halves(29, kWidth * 8, 4.0f);
  const std::vector<std::int32_t> picks(kPicks.begin(), kPicks.end());
  std::vector<float> want;
  for (const auto row : picks) {
    for (std::int64_t col = 0; col < kWidth; ++col) {
      want.push_back(ggml_fp16_to_fp32(source[static_cast<std::size_t>((row * kWidth) + col)]));
    }
  }
  for (const auto memory : {Memory::kCudaMalloc, Memory::kDeviceVmm, Memory::kHostVmm}) {
    auto arena = TensorArena::Create(8).value();
    auto launch = Launcher();
    auto* rows =
        Place(memory, ggml_new_tensor_2d(arena.context(), GGML_TYPE_F16, kWidth, 8), source);
    auto* ids = Place(memory, ggml_new_tensor_1d(arena.context(), GGML_TYPE_I32, kRows), picks);
    auto* gathered = Place(memory, ggml_get_rows(arena.context(), rows, ids));
    Launched(jitllm::kernels::ggml::GetRows(*launch, gathered), "F16 APE gather");
    EXPECT_EQ(Bits(Download(gathered)), Bits(want));
  }
}

class GgmlOpsMemoryTest : public GgmlOpsTest {
 protected:
  Outputs Run(Memory memory) {
    auto arena = TensorArena::Create(128).value();
    ggml_context* c = arena.context();
    auto launch = Launcher();
    Outputs out;

    // get_rows: five ids of eight rows, and 128 ids of 130 rows.
    ggml_tensor* rows =
        Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, 8), Values(1, kWidth * 8));
    ggml_tensor* ids = Place(memory, ggml_new_tensor_1d(c, GGML_TYPE_I32, kRows),
                             std::vector<std::int32_t>(kPicks.begin(), kPicks.end()));
    ggml_tensor* picked = Place(memory, ggml_get_rows(c, rows, ids));
    ggml_tensor* many_rows =
        Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, 130), Values(2, kWidth * 130));
    std::vector<std::int32_t> many(128);
    for (std::size_t i = 0; i < many.size(); ++i) {
      many[i] = static_cast<std::int32_t>((i * 37) % 130);
    }
    ggml_tensor* many_ids = Place(memory, ggml_new_tensor_1d(c, GGML_TYPE_I32, 128), many);
    ggml_tensor* picked_vec = Place(memory, ggml_get_rows(c, many_rows, many_ids));
    EXPECT_FALSE(jitllm::kernels::ggml::GetRowsVectorized(picked));
    EXPECT_TRUE(jitllm::kernels::ggml::GetRowsVectorized(picked_vec));
    Launched(jitllm::kernels::ggml::GetRows(*launch, picked), "get_rows");
    Launched(jitllm::kernels::ggml::GetRows(*launch, picked_vec), "get_rows (vector)");

    // set_rows: five K rows into a 16-cell cache.
    const std::vector<ggml_fp16_t> poison(kKvWidth * 16, std::bit_cast<ggml_fp16_t>(kPoisonHalf));
    ggml_tensor* cache = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F16, kKvWidth, 16), poison);
    ggml_tensor* values = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F32, kKvWidth, kRows),
                                Values(3, kKvWidth * kRows, 4.0f));
    ggml_tensor* slots = Place(memory, ggml_new_tensor_1d(c, GGML_TYPE_I64, kRows),
                               std::vector<std::int64_t>(kSlots.begin(), kSlots.end()));
    Launched(jitllm::kernels::ggml::SetRows(*launch, ggml_set_rows(c, cache, values, slots)),
             "set_rows");

    // RoPE of Q, and of K fused with its write or not.
    ggml_tensor* positions = Place(memory, ggml_new_tensor_1d(c, GGML_TYPE_I32, kRows),
                                   std::vector<std::int32_t>(kPositions.begin(), kPositions.end()));
    ggml_tensor* q = Place(memory, ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, kHeads, kRows),
                           Values(4, kHead * kHeads * kRows, 2.0f));
    ggml_tensor* rotated = Place(memory, QwenRope(c, q, positions));
    Launched(jitllm::kernels::ggml::Rope(*launch, rotated), "rope");
    ggml_tensor* k = Place(memory, ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, kKvHeads, kRows),
                           Values(5, kKvWidth * kRows, 2.0f));
    ggml_tensor* fused_cache =
        Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F16, kKvWidth, 16), poison);
    ggml_tensor* fused_rope = QwenRope(c, k, positions);  // never written
    ggml_tensor* fused_write = ggml_set_rows(
        c, fused_cache, ggml_view_2d(c, fused_rope, kKvWidth, kRows, fused_rope->nb[2], 0), slots);
    Launched(jitllm::kernels::ggml::RopeSetRows(*launch, fused_rope, fused_write),
             "rope fused with set_rows");
    ggml_tensor* unfused_cache =
        Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F16, kKvWidth, 16), poison);
    ggml_tensor* k_rope = Place(memory, QwenRope(c, k, positions));
    ggml_tensor* unfused_write = ggml_set_rows(
        c, unfused_cache, ggml_view_2d(c, k_rope, kKvWidth, kRows, k_rope->nb[2], 0), slots);
    Launched(jitllm::kernels::ggml::Rope(*launch, k_rope), "rope of K");
    Launched(jitllm::kernels::ggml::SetRows(*launch, unfused_write), "set_rows of K");

    // soft_max over 256 and 768 cells, with a causal mask.
    ggml_tensor* probabilities = nullptr;
    ggml_tensor* probabilities_wide = nullptr;
    for (const std::int64_t cells : {256, 768}) {
      std::vector<float> mask(static_cast<std::size_t>(cells * kRows));
      for (std::int64_t r = 0; r < kRows; ++r) {
        for (std::int64_t col = 0; col < cells; ++col) {
          mask[static_cast<std::size_t>((r * cells) + col)] =
              col <= kPositions[static_cast<std::size_t>(r)]
                  ? 0.0f
                  : -std::numeric_limits<float>::infinity();
        }
      }
      ggml_tensor* scores =
          Place(memory, ggml_new_tensor_3d(c, GGML_TYPE_F32, cells, kRows, kHeads),
                Values(6, static_cast<std::size_t>(cells * kRows * kHeads), 8.0f));
      ggml_tensor* mask_tensor =
          Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F32, cells, kRows), mask);
      ggml_tensor* soft = Place(memory, ggml_soft_max_ext(c, scores, mask_tensor, kScale, 0.0f));
      Launched(jitllm::kernels::ggml::SoftMax(*launch, soft), "soft_max");
      (cells == 256 ? probabilities : probabilities_wide) = soft;
    }

    // cont of the attention's heads: the scalar kernel, and for one row a
    // copy.
    ggml_tensor* heads = Place(memory, ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, kRows, kHeads),
                               Values(7, kHead * kRows * kHeads));
    ggml_tensor* merged =
        Place(memory, ggml_cont_2d(c, ggml_permute(c, heads, 0, 2, 1, 3), kWidth, kRows));
    Launched(jitllm::kernels::ggml::Cont(*launch, merged), "cont");
    ggml_tensor* head_row = Place(memory, ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, 1, kHeads),
                                  Values(8, kHead * kHeads));
    ggml_tensor* merged_row =
        Place(memory, ggml_cont_2d(c, ggml_permute(c, head_row, 0, 2, 1, 3), kWidth, 1));
    Launched(jitllm::kernels::ggml::Cont(*launch, merged_row), "cont of one row");

    // SwiGLU.
    ggml_tensor* gate_in = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F32, kFfn, kRows),
                                 Values(9, kFfn * kRows, 6.0f));
    ggml_tensor* up_in = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F32, kFfn, kRows),
                               Values(10, kFfn * kRows, 2.0f));
    ggml_tensor* glu = Place(memory, ggml_swiglu_split(c, gate_in, up_in));
    Launched(jitllm::kernels::ggml::SwiGlu(*launch, glu), "swiglu");

    // MMVF with a bias, fused and not.
    ggml_tensor* x =
        Place(memory, ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth), Values(11, kWidth));
    ggml_tensor* w = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, kWidth),
                           Halves(12, kWidth * kWidth, 0.1f));
    ggml_tensor* bias =
        Place(memory, ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth), Values(13, kWidth));
    ggml_tensor* product = ggml_mul_mat(c, w, x);  // never written
    ggml_tensor* biased = Place(memory, ggml_add(c, product, bias));
    Launched(jitllm::kernels::ggml::MulMatVecBias(*launch, product, biased), "fused bias");
    ggml_tensor* product_unfused = Place(memory, ggml_mul_mat(c, w, x));
    ggml_tensor* biased_unfused = Place(memory, ggml_add(c, product_unfused, bias));
    Launched(jitllm::kernels::ggml::MulMatVecF(*launch, product_unfused), "MMVF");
    Launched(jitllm::kernels::ggml::Add(*launch, biased_unfused), "add");

    // MMVF with gate, up and SwiGLU, fused and not.
    ggml_tensor* w_gate = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, kFfn),
                                Halves(14, kWidth * kFfn, 0.1f));
    ggml_tensor* w_up = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, kFfn),
                              Halves(15, kWidth * kFfn, 0.1f));
    ggml_tensor* gate = ggml_mul_mat(c, w_gate, x);  // never written
    ggml_tensor* up = ggml_mul_mat(c, w_up, x);      // never written
    ggml_tensor* gated = Place(memory, ggml_swiglu_split(c, gate, up));
    Launched(jitllm::kernels::ggml::MulMatVecGlu(*launch, gate, up, gated), "fused gate and up");
    ggml_tensor* gate_unfused = Place(memory, ggml_mul_mat(c, w_gate, x));
    ggml_tensor* up_unfused = Place(memory, ggml_mul_mat(c, w_up, x));
    ggml_tensor* gated_unfused = Place(memory, ggml_swiglu_split(c, gate_unfused, up_unfused));
    Launched(jitllm::kernels::ggml::MulMatVecF(*launch, gate_unfused), "MMVF gate");
    Launched(jitllm::kernels::ggml::MulMatVecF(*launch, up_unfused), "MMVF up");
    Launched(jitllm::kernels::ggml::SwiGlu(*launch, gated_unfused), "swiglu of the products");
    EXPECT_EQ(launch->scratch_peak(), Bytes(0));
    EXPECT_FALSE(launch->faulted());

    out.picked = Download(picked);
    out.picked_vec = Download(picked_vec);
    out.cache = Download<ggml_fp16_t>(cache);
    out.rotated = Download(rotated);
    out.fused_cache = Download<ggml_fp16_t>(fused_cache);
    out.unfused_cache = Download<ggml_fp16_t>(unfused_cache);
    out.probabilities = Download(probabilities);
    out.probabilities_wide = Download(probabilities_wide);
    out.merged = Download(merged);
    out.merged_row = Download(merged_row);
    out.glu = Download(glu);
    out.biased = Download(biased);
    out.biased_unfused = Download(biased_unfused);
    out.gated = Download(gated);
    out.gated_unfused = Download(gated_unfused);
    return out;
  }
};

std::vector<std::uint16_t> HalfBits(const std::vector<ggml_fp16_t>& values) {
  std::vector<std::uint16_t> bits;
  bits.reserve(values.size());
  for (const ggml_fp16_t value : values) {
    bits.push_back(std::bit_cast<std::uint16_t>(value));
  }
  return bits;
}

// RoPE NEOX in double: position p rotates the pair (i, i + 32) of a head
// by p * base^(-2i/64).
std::vector<double> ReferenceRope(const std::vector<float>& x, std::int64_t heads) {
  std::vector<double> out(x.size());
  for (std::int64_t t = 0; t < kRows; ++t) {
    const double position = kPositions[static_cast<std::size_t>(t)];
    for (std::int64_t h = 0; h < heads; ++h) {
      const std::int64_t base = ((t * heads) + h) * kHead;
      for (std::int64_t i = 0; i < kHead / 2; ++i) {
        const double theta = position * std::pow(static_cast<double>(kRopeBase),
                                                 -2.0 * static_cast<double>(i) / kHead);
        const double x0 = x[static_cast<std::size_t>(base + i)];
        const double x1 = x[static_cast<std::size_t>(base + i + (kHead / 2))];
        out[static_cast<std::size_t>(base + i)] = (x0 * std::cos(theta)) - (x1 * std::sin(theta));
        out[static_cast<std::size_t>(base + i + (kHead / 2))] =
            (x0 * std::sin(theta)) + (x1 * std::cos(theta));
      }
    }
  }
  return out;
}

std::vector<double> ReferenceSoftMax(std::int64_t cells) {
  const std::vector<float> scores =
      Values(6, static_cast<std::size_t>(cells * kRows * kHeads), 8.0f);
  std::vector<double> out(scores.size());
  for (std::int64_t h = 0; h < kHeads; ++h) {
    for (std::int64_t r = 0; r < kRows; ++r) {
      const std::int64_t row = ((h * kRows) + r) * cells;
      const std::int64_t visible = kPositions[static_cast<std::size_t>(r)] + 1;
      double top = -std::numeric_limits<double>::infinity();
      for (std::int64_t col = 0; col < std::min(visible, cells); ++col) {
        top = std::max(top, scores[static_cast<std::size_t>(row + col)] * double{kScale});
      }
      double sum = 0.0;
      for (std::int64_t col = 0; col < std::min(visible, cells); ++col) {
        sum += std::exp((scores[static_cast<std::size_t>(row + col)] * double{kScale}) - top);
      }
      for (std::int64_t col = 0; col < cells; ++col) {
        out[static_cast<std::size_t>(row + col)] =
            col < visible
                ? std::exp((scores[static_cast<std::size_t>(row + col)] * double{kScale}) - top) /
                      sum
                : 0.0;
      }
    }
  }
  return out;
}

double Silu(double x) { return x / (1.0 + std::exp(-x)); }

// A product of F16 weights (as stored) and F32 activations, in double.
std::vector<double> ReferenceProduct(std::uint64_t seed, std::int64_t outputs,
                                     const std::vector<float>& x) {
  const std::vector<ggml_fp16_t> w = Halves(seed, static_cast<std::size_t>(kWidth * outputs), 0.1f);
  std::vector<double> out(static_cast<std::size_t>(outputs));
  for (std::int64_t o = 0; o < outputs; ++o) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < kWidth; ++i) {
      sum += static_cast<double>(ggml_fp16_to_fp32(w[static_cast<std::size_t>((o * kWidth) + i)])) *
             x[static_cast<std::size_t>(i)];
    }
    out[static_cast<std::size_t>(o)] = sum;
  }
  return out;
}

// Fused and unfused outputs: reported, not judged; each is exact against
// its own bridge arm in P2 (BP-S1).
void ReportDifferences(const std::vector<float>& fused, const std::vector<float>& unfused,
                       const char* what) {
  std::size_t differ = 0;
  for (std::size_t i = 0; i < fused.size(); ++i) {
    differ +=
        std::bit_cast<std::uint32_t>(fused[i]) != std::bit_cast<std::uint32_t>(unfused[i]) ? 1 : 0;
  }
  std::cout << what << ": fused and unfused differ in " << differ << " of " << fused.size()
            << " elements\n";
}

TEST_F(GgmlOpsMemoryTest, EachOperationIsExactAcrossMemoryAndCloseToAReference) {
  const Outputs control = Run(Memory::kCudaMalloc);
  for (const Memory memory : {Memory::kDeviceVmm, Memory::kHostVmm}) {
    const Outputs got = Run(memory);
    EXPECT_EQ(Bits(got.picked), Bits(control.picked));
    EXPECT_EQ(Bits(got.picked_vec), Bits(control.picked_vec));
    EXPECT_EQ(HalfBits(got.cache), HalfBits(control.cache));
    EXPECT_EQ(Bits(got.rotated), Bits(control.rotated));
    EXPECT_EQ(HalfBits(got.fused_cache), HalfBits(control.fused_cache));
    EXPECT_EQ(HalfBits(got.unfused_cache), HalfBits(control.unfused_cache));
    EXPECT_EQ(Bits(got.probabilities), Bits(control.probabilities));
    EXPECT_EQ(Bits(got.probabilities_wide), Bits(control.probabilities_wide));
    EXPECT_EQ(Bits(got.merged), Bits(control.merged));
    EXPECT_EQ(Bits(got.merged_row), Bits(control.merged_row));
    EXPECT_EQ(Bits(got.glu), Bits(control.glu));
    EXPECT_EQ(Bits(got.biased), Bits(control.biased));
    EXPECT_EQ(Bits(got.biased_unfused), Bits(control.biased_unfused));
    EXPECT_EQ(Bits(got.gated), Bits(control.gated));
    EXPECT_EQ(Bits(got.gated_unfused), Bits(control.gated_unfused));
  }

  // Copies are exact: gathered rows, permuted heads, and F32 rounded once
  // to F16 in the cache, which is untouched elsewhere.
  const std::vector<float> rows = Values(1, kWidth * 8);
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::int64_t i = 0; i < kWidth; ++i) {
      ASSERT_EQ(control.picked[(r * kWidth) + static_cast<std::size_t>(i)],
                rows[(static_cast<std::size_t>(kPicks[r]) * kWidth) + static_cast<std::size_t>(i)]);
    }
  }
  const std::vector<float> many_rows = Values(2, kWidth * 130);
  for (std::size_t r = 0; r < 128; ++r) {
    for (std::int64_t i = 0; i < kWidth; ++i) {
      ASSERT_EQ(control.picked_vec[(r * kWidth) + static_cast<std::size_t>(i)],
                many_rows[(((r * 37) % 130) * kWidth) + static_cast<std::size_t>(i)]);
    }
  }
  const std::vector<float> values = Values(3, kKvWidth * kRows, 4.0f);
  for (std::size_t cell = 0; cell < 16; ++cell) {
    const auto* slot = std::ranges::find(kSlots, static_cast<std::int64_t>(cell));
    for (std::int64_t i = 0; i < kKvWidth; ++i) {
      const auto got = std::bit_cast<std::uint16_t>(
          control.cache[(cell * kKvWidth) + static_cast<std::size_t>(i)]);
      const std::uint16_t want =
          slot == kSlots.end()
              ? kPoisonHalf
              : std::bit_cast<std::uint16_t>(ggml_fp32_to_fp16(
                    values[(static_cast<std::size_t>(slot - kSlots.begin()) * kKvWidth) +
                           static_cast<std::size_t>(i)]));
      ASSERT_EQ(got, want) << cell << " " << i;
    }
  }
  const std::vector<float> heads = Values(7, kHead * kRows * kHeads);
  for (std::int64_t r = 0; r < kRows; ++r) {
    for (std::int64_t h = 0; h < kHeads; ++h) {
      for (std::int64_t i = 0; i < kHead; ++i) {
        ASSERT_EQ(control.merged[static_cast<std::size_t>((r * kWidth) + (h * kHead) + i)],
                  heads[static_cast<std::size_t>((((h * kRows) + r) * kHead) + i)]);
      }
    }
  }
  EXPECT_EQ(Bits(control.merged_row), Bits(Values(8, kHead * kHeads)));

  // RoPE under fast math: sinf and cosf of angles up to 576 radians, from
  // a single-precision power (6.5e-5 on spark-b, 2026-09-27).
  ExpectClose(control.rotated, ReferenceRope(Values(4, kHead * kHeads * kRows, 2.0f), kHeads), 5e-4,
              "rope");
  // The fused K write is the RoPE of K rounded to F16, as the unfused one.
  const std::vector<double> k_rope = ReferenceRope(Values(5, kKvWidth * kRows, 2.0f), kKvHeads);
  for (const auto* cache : {&control.fused_cache, &control.unfused_cache}) {
    std::vector<float> written;
    std::vector<double> want;
    for (std::size_t r = 0; r < kRows; ++r) {
      for (std::int64_t i = 0; i < kKvWidth; ++i) {
        written.push_back(
            ggml_fp16_to_fp32((*cache)[(static_cast<std::size_t>(kSlots[r]) * kKvWidth) +
                                       static_cast<std::size_t>(i)]));
        want.push_back(k_rope[(r * kKvWidth) + static_cast<std::size_t>(i)]);
      }
    }
    ExpectClose(written, want, 1e-3, cache == &control.fused_cache ? "fused K write" : "K write");
  }
  std::cout << "rope then set_rows and the fused write: "
            << (HalfBits(control.fused_cache) == HalfBits(control.unfused_cache) ? "identical"
                                                                                 : "differ")
            << "\n";

  ExpectClose(control.probabilities, ReferenceSoftMax(256), 1e-6, "soft_max, 256 cells");
  ExpectClose(control.probabilities_wide, ReferenceSoftMax(768), 1e-6, "soft_max, 768 cells");

  const std::vector<float> gate_in = Values(9, kFfn * kRows, 6.0f);
  const std::vector<float> up_in = Values(10, kFfn * kRows, 2.0f);
  std::vector<double> glu(gate_in.size());
  for (std::size_t i = 0; i < glu.size(); ++i) {
    glu[i] = Silu(gate_in[i]) * up_in[i];
  }
  ExpectClose(control.glu, glu, 1e-6, "swiglu");

  // The products: F16 weights, so a few units in the third place.
  const std::vector<float> x = Values(11, kWidth);
  const std::vector<float> bias = Values(13, kWidth);
  std::vector<double> biased = ReferenceProduct(12, kWidth, x);
  for (std::size_t i = 0; i < biased.size(); ++i) {
    biased[i] += bias[i];
  }
  ExpectClose(control.biased, biased, 2e-2, "fused bias");
  ExpectClose(control.biased_unfused, biased, 2e-2, "MMVF then add");
  ReportDifferences(control.biased, control.biased_unfused, "bias");
  const std::vector<double> gate = ReferenceProduct(14, kFfn, x);
  const std::vector<double> up = ReferenceProduct(15, kFfn, x);
  std::vector<double> gated(gate.size());
  for (std::size_t i = 0; i < gated.size(); ++i) {
    gated[i] = Silu(gate[i]) * up[i];
  }
  // Fused, F16 weights accumulate in F32 (the GLU's parameters are read as
  // the precision), so the fused products are far closer.
  ExpectClose(control.gated, gated, 1e-4, "fused gate and up");
  ExpectClose(control.gated_unfused, gated, 2e-2, "MMVF, MMVF, then swiglu");
  ReportDifferences(control.gated, control.gated_unfused, "gate and up");
}

// Each operation at Qwen2.5-0.5B's shapes launches what the bridge's
// recorded plan launched there. The shapes are llama.cpp's graph at the
// pin, built the same way; the rows are the recorded chunks' (1 for
// decode, 16, 17, 32 and 512 for prefill), and the recorded plan's
// sequences named are control-fused (CF), control-unfused (CU),
// heldout-fused (HF) and heldout-unfused (HU), with their sequence ids.
class GgmlOpsPlanMatchTest : public GgmlOpsTest {
 protected:
  // Compares what was recorded with the plan's launches, by kernel name
  // (normalized), grid, block and dynamic shared memory, or copy size.
  static void Matches(const std::vector<Event>& got, const std::vector<Planned>& want,
                      const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (want[i].id == 0) {
        EXPECT_EQ(got[i].kind, EventKind::kCopy) << what << ": " << got[i].name;
        EXPECT_EQ(got[i].bytes, want[i].bytes) << what;
        continue;
      }
      EXPECT_EQ(jitllm::test_support::NormalizedKernelName(got[i].name),
                jitllm::test_support::NormalizedKernelName(PlanKernel(want[i].id)))
          << what << ": kernel " << want[i].id;
      EXPECT_EQ(got[i].grid, want[i].grid) << what << ": kernel " << want[i].id;
      EXPECT_EQ(got[i].block, want[i].block) << what << ": kernel " << want[i].id;
      EXPECT_EQ(got[i].shared, want[i].shared) << what << ": kernel " << want[i].id;
    }
  }

  // A tensor in cudaMalloc memory; only indices need contents here.
  template <typename T = float>
  ggml_tensor* At(ggml_tensor* tensor, const std::vector<T>& data = {}) {
    return Place(Memory::kCudaMalloc, tensor, data);
  }

  template <typename T>
  static std::vector<T> Iota(std::int64_t count) {
    std::vector<T> values(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<T>(i);
    }
    return values;
  }

  static unsigned U(std::int64_t value) { return static_cast<unsigned>(value); }
};

TEST_F(GgmlOpsPlanMatchTest, PrefillAndDecodeOperationsLaunchAsRecorded) {
  auto launch = Launcher();
  for (const std::int64_t n : {1, 16, 17, 32, 512}) {
    auto arena = TensorArena::Create(64).value();
    ggml_context* c = arena.context();
    const std::string rows = std::to_string(n) + " rows";

    // get_rows over the last layer's rows (CF/CU 0 and 1, HF/HU 0-4): the
    // scalar kernel at [n, 4, 1], the vector kernel at 512 rows.
    ggml_tensor* hidden = At(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, n));
    ggml_tensor* out_ids = At(ggml_new_tensor_1d(c, GGML_TYPE_I32, n), Iota<std::int32_t>(n));
    ggml_tensor* picked = At(ggml_get_rows(c, hidden, out_ids));
    Matches(Record([&] { Launched(jitllm::kernels::ggml::GetRows(*launch, picked), "get_rows"); }),
            {n == 512 ? PlanLaunch(7, {512, 1, 1}, {256, 1, 1})
                      : PlanLaunch(6, {U(n), 4, 1}, {256, 1, 1})},
            "get_rows, " + rows);

    // Q's RoPE; K's RoPE and its KV write, unfused (CU, HU) and fused (CF,
    // HF); V's write of one element per row into the transposed cache.
    ggml_tensor* positions = At(ggml_new_tensor_1d(c, GGML_TYPE_I32, n), Iota<std::int32_t>(n));
    ggml_tensor* q =
        ggml_reshape_3d(c, At(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, n)), kHead, kHeads, n);
    ggml_tensor* q_rope = At(QwenRope(c, q, positions));
    Matches(Record([&] { Launched(jitllm::kernels::ggml::Rope(*launch, q_rope), "rope"); }),
            {PlanLaunch(25, {U(kHeads * n), 1, 1}, {1, 256, 1})}, "Q rope, " + rows);
    ggml_tensor* k = ggml_reshape_3d(c, At(ggml_new_tensor_2d(c, GGML_TYPE_F32, kKvWidth, n)),
                                     kHead, kKvHeads, n);
    ggml_tensor* k_cache = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kKvWidth, kCells));
    ggml_tensor* k_ids = At(ggml_new_tensor_1d(c, GGML_TYPE_I64, n), Iota<std::int64_t>(n));
    ggml_tensor* k_rope = At(QwenRope(c, k, positions));
    ggml_tensor* k_write =
        ggml_set_rows(c, k_cache, ggml_view_2d(c, k_rope, kKvWidth, n, k_rope->nb[2], 0), k_ids);
    const unsigned write_blocks = U(((kKvWidth * n) + 255) / 256);
    Matches(Record([&] {
              Launched(jitllm::kernels::ggml::Rope(*launch, k_rope), "rope");
              Launched(jitllm::kernels::ggml::SetRows(*launch, k_write), "set_rows");
            }),
            {PlanLaunch(25, {U(kKvHeads * n), 1, 1}, {1, 256, 1}),
             PlanLaunch(8, {write_blocks, 1, 1}, {256, 1, 1})},
            "K rope then write, " + rows);
    ggml_tensor* fused_rope = QwenRope(c, k, positions);
    ggml_tensor* fused_write = ggml_set_rows(
        c, k_cache, ggml_view_2d(c, fused_rope, kKvWidth, n, fused_rope->nb[2], 0), k_ids);
    Matches(Record([&] {
              Launched(jitllm::kernels::ggml::RopeSetRows(*launch, fused_rope, fused_write),
                       "rope_set_rows");
            }),
            {PlanLaunch(24, {U(kKvHeads * n), 1, 1}, {1, 256, 1})},
            "K rope fused with its write, " + rows);
    ggml_tensor* v_cache = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kCells, kKvWidth));
    ggml_tensor* v =
        ggml_reshape_2d(c, At(ggml_new_tensor_2d(c, GGML_TYPE_F32, kKvWidth, n)), 1, kKvWidth * n);
    ggml_tensor* v_ids =
        At(ggml_new_tensor_1d(c, GGML_TYPE_I64, kKvWidth * n), Iota<std::int64_t>(kKvWidth * n));
    ggml_tensor* v_write =
        ggml_set_rows(c, ggml_reshape_2d(c, v_cache, 1, kCells * kKvWidth), v, v_ids);
    Matches(Record([&] { Launched(jitllm::kernels::ggml::SetRows(*launch, v_write), "set_rows"); }),
            {PlanLaunch(8, {write_blocks, 1, 1}, {256, 1, 1})}, "V write, " + rows);

    // soft_max over 256 cells (CF/CU 0 and 1, HF/HU 0-2) and 768 (HF/HU 3
    // and 4, whose chunks run at 512 rows and at one).
    for (const std::int64_t cells : {256, 768}) {
      if ((cells == 256 && n == 512) || (cells == 768 && n != 1 && n != 512)) {
        continue;
      }
      ggml_tensor* scores = At(ggml_new_tensor_3d(c, GGML_TYPE_F32, cells, n, kHeads));
      ggml_tensor* mask = At(ggml_new_tensor_2d(c, GGML_TYPE_F32, cells, n));
      ggml_tensor* soft = At(ggml_soft_max_ext(c, scores, mask, kScale, 0.0f));
      Matches(Record([&] { Launched(jitllm::kernels::ggml::SoftMax(*launch, soft), "soft_max"); }),
              {cells == 256 ? PlanLaunch(27, {U(n), 14, 1}, {256, 1, 1}, 1152)
                            : PlanLaunch(26, {U(n), 14, 1}, {1024, 1, 1}, 3200)},
              "soft_max over " + std::to_string(cells) + " cells, " + rows);
    }

    // The heads merged back: a 3,584-byte copy at one row (CF/CU 1, HF/HU 2
    // and 4), else the scalar kernel at [14n, 1, 1].
    ggml_tensor* kqv = At(ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, n, kHeads));
    ggml_tensor* merged = At(ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), kWidth, n));
    Matches(Record([&] { Launched(jitllm::kernels::ggml::Cont(*launch, merged), "cont"); }),
            {n == 1 ? PlanCopy(kWidth * sizeof(float))
                    : PlanLaunch(3, {U(kHeads * n), 1, 1}, {64, 1, 1})},
            "cont, " + rows);

    // SwiGLU (prefill in every arm; decode unfused, CU 1 and HU 2 and 4).
    ggml_tensor* glu = At(ggml_swiglu_split(c, At(ggml_new_tensor_2d(c, GGML_TYPE_F32, kFfn, n)),
                                            At(ggml_new_tensor_2d(c, GGML_TYPE_F32, kFfn, n))));
    Matches(Record([&] { Launched(jitllm::kernels::ggml::SwiGlu(*launch, glu), "swiglu"); }),
            {PlanLaunch(28, {U(((kFfn * n) + 255) / 256), 1, 1}, {256, 1, 1})}, "swiglu, " + rows);
    Finish();
  }
  EXPECT_FALSE(launch->faulted());
}

// The decode step's matrix products (CF 1, HF 2 and 4 fused; CU 1, HU 2 and
// 4 unfused).
TEST_F(GgmlOpsPlanMatchTest, DecodeProductsLaunchAsRecordedFusedAndNot) {
  auto launch = Launcher();
  auto arena = TensorArena::Create(64).value();
  ggml_context* c = arena.context();
  ggml_tensor* x = At(ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth));
  ggml_tensor* ffn_x = At(ggml_new_tensor_1d(c, GGML_TYPE_F32, kFfn));
  ggml_tensor* wq = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, kWidth));
  ggml_tensor* wk = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, kKvWidth));
  ggml_tensor* w_gate = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, kFfn));
  ggml_tensor* w_up = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, kFfn));
  ggml_tensor* w_down = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kFfn, kWidth));
  ggml_tensor* bq = At(ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth));
  ggml_tensor* bk = At(ggml_new_tensor_1d(c, GGML_TYPE_F32, kKvWidth));
  ggml_tensor* residual = At(ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth));

  // Fused: Q (and O with the residual) and K, V with their biases at
  // [896] and [128] (kernel 17); down with the residual (19); gate, up and
  // SwiGLU (20, whose F16 weights accumulate in F32).
  ggml_tensor* q = ggml_mul_mat(c, wq, x);
  ggml_tensor* q_biased = At(ggml_add(c, q, bq));
  ggml_tensor* k = ggml_mul_mat(c, wk, x);
  ggml_tensor* k_biased = At(ggml_add(c, k, bk));
  ggml_tensor* down = ggml_mul_mat(c, w_down, ffn_x);
  ggml_tensor* out = At(ggml_add(c, down, residual));
  ggml_tensor* gate = ggml_mul_mat(c, w_gate, x);
  ggml_tensor* up = ggml_mul_mat(c, w_up, x);
  ggml_tensor* glu = At(ggml_swiglu_split(c, gate, up));
  for (ggml_tensor* product : {q, k, down, gate, up}) {
    EXPECT_TRUE(jitllm::kernels::ggml::MulMatVecFusible(*launch, product));
  }
  Matches(
      Record([&] {
        Launched(jitllm::kernels::ggml::MulMatVecBias(*launch, q, q_biased), "Q");
        Launched(jitllm::kernels::ggml::MulMatVecBias(*launch, k, k_biased), "K");
        Launched(jitllm::kernels::ggml::MulMatVecBias(*launch, down, out), "down");
        Launched(jitllm::kernels::ggml::MulMatVecGlu(*launch, gate, up, glu), "gate and up");
      }),
      {PlanLaunch(17, {896, 1, 1}, {224, 1, 1}, 256), PlanLaunch(17, {128, 1, 1}, {224, 1, 1}, 256),
       PlanLaunch(19, {896, 1, 1}, {256, 1, 1}, 256),
       PlanLaunch(20, {4864, 1, 1}, {224, 1, 1}, 256)},
      "fused decode products");

  // Unfused: MMVF (16, and 18 for down) and the bias add at [4, 1, 1] (4).
  ggml_tensor* q_written = At(ggml_mul_mat(c, wq, x));
  ggml_tensor* q_added = At(ggml_add(c, q_written, bq));
  ggml_tensor* k_written = At(ggml_mul_mat(c, wk, x));
  ggml_tensor* down_written = At(ggml_mul_mat(c, w_down, ffn_x));
  ggml_tensor* gate_written = At(ggml_mul_mat(c, w_gate, x));
  Matches(
      Record([&] {
        Launched(jitllm::kernels::ggml::MulMatVecF(*launch, q_written), "Q");
        Launched(jitllm::kernels::ggml::Add(*launch, q_added), "bias");
        Launched(jitllm::kernels::ggml::MulMatVecF(*launch, k_written), "K");
        Launched(jitllm::kernels::ggml::MulMatVecF(*launch, down_written), "down");
        Launched(jitllm::kernels::ggml::MulMatVecF(*launch, gate_written), "gate");
      }),
      {PlanLaunch(16, {896, 1, 1}, {224, 1, 1}, 128), PlanLaunch(4, {4, 1, 1}, {128, 1, 1}),
       PlanLaunch(16, {128, 1, 1}, {224, 1, 1}, 128), PlanLaunch(18, {896, 1, 1}, {256, 1, 1}, 128),
       PlanLaunch(16, {4864, 1, 1}, {224, 1, 1}, 128)},
      "unfused decode products");

  // Upstream fuses only one column: at two, the device's gate refuses, and
  // so does the fused implementation.
  ggml_tensor* two = At(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, 2));
  ggml_tensor* q2 = ggml_mul_mat(c, wq, two);
  EXPECT_FALSE(jitllm::kernels::ggml::MulMatVecFusible(*launch, q2));
  EXPECT_EQ(
      FailedCode(jitllm::kernels::ggml::MulMatVecBias(
          *launch, q2, At(ggml_add(c, q2, At(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, 2)))))),
      KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());
}

// The fused decode step's first seven launches (CF 1 from token 6: the
// attention norm, Q with its bias, Q's RoPE, K and V with theirs, K's RoPE
// with its write, V's write), recorded and written as the recorder's JSON
// lines, equal plan_record_sample.txt, which plan_compare.py matches with
// the record. Registers and shared memory are the runtime's. The recording
// covers the whole test, the inputs' uploads before the chunk included, so
// that an nsys trace of this test alone lines up with it (plan_compare.py
// --nsys).
TEST_F(GgmlOpsPlanMatchTest, DecodeStepStartRecordsAsTheSample) {
  jitllm::test_support::Recording recording;
  auto launch = Launcher();
  auto arena = TensorArena::Create(64).value();
  ggml_context* c = arena.context();
  ggml_tensor* x = At(ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth));
  ggml_tensor* norm_weight = At(ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth));
  ggml_tensor* norm = At(ggml_rms_norm(c, x, 1e-6f));
  ggml_tensor* normed = At(ggml_mul(c, norm, norm_weight));
  auto biased = [&](std::int64_t width) {
    ggml_tensor* weight = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, width));
    ggml_tensor* bias = At(ggml_new_tensor_1d(c, GGML_TYPE_F32, width));
    ggml_tensor* product = ggml_mul_mat(c, weight, normed);
    return std::pair{product, At(ggml_add(c, product, bias))};
  };
  const auto [q, q_biased] = biased(kWidth);
  const auto [k, k_biased] = biased(kKvWidth);
  const auto [v, v_biased] = biased(kKvWidth);
  ggml_tensor* positions = At(ggml_new_tensor_1d(c, GGML_TYPE_I32, 1), Iota<std::int32_t>(1));
  ggml_tensor* q_rope = At(QwenRope(c, ggml_reshape_3d(c, q_biased, kHead, kHeads, 1), positions));
  ggml_tensor* k_cache = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kKvWidth, kCells));
  ggml_tensor* k_ids = At(ggml_new_tensor_1d(c, GGML_TYPE_I64, 1), Iota<std::int64_t>(1));
  ggml_tensor* k_rope = QwenRope(c, ggml_reshape_3d(c, k_biased, kHead, kKvHeads, 1), positions);
  ggml_tensor* k_write =
      ggml_set_rows(c, k_cache, ggml_view_2d(c, k_rope, kKvWidth, 1, k_rope->nb[2], 0), k_ids);
  ggml_tensor* v_cache = At(ggml_new_tensor_2d(c, GGML_TYPE_F16, kCells, kKvWidth));
  ggml_tensor* v_ids =
      At(ggml_new_tensor_1d(c, GGML_TYPE_I64, kKvWidth), Iota<std::int64_t>(kKvWidth));
  ggml_tensor* v_write = ggml_set_rows(c, ggml_reshape_2d(c, v_cache, 1, kCells * kKvWidth),
                                       ggml_reshape_2d(c, v_biased, 1, kKvWidth), v_ids);
  const std::vector<Event> uploads = recording.Take();
  Launched(jitllm::kernels::ggml::RmsNormMul(*launch, norm, normed), "attention norm");
  Launched(jitllm::kernels::ggml::MulMatVecBias(*launch, q, q_biased), "Q");
  Launched(jitllm::kernels::ggml::Rope(*launch, q_rope), "Q's RoPE");
  Launched(jitllm::kernels::ggml::MulMatVecBias(*launch, k, k_biased), "K");
  Launched(jitllm::kernels::ggml::MulMatVecBias(*launch, v, v_biased), "V");
  Launched(jitllm::kernels::ggml::RopeSetRows(*launch, k_rope, k_write), "K's RoPE and write");
  Launched(jitllm::kernels::ggml::SetRows(*launch, v_write), "V's write");
  std::vector<Event> events = recording.Take();
  Finish();
  EXPECT_FALSE(launch->faulted());
  EXPECT_TRUE(recording.Take().empty());

  // Everything on the context's stream: the uploads are the provider's copies.
  const void* native = execution_->Submission(stream_).value().handle;
  std::string outside;
  for (const Event& upload : uploads) {
    EXPECT_EQ(upload.kind, EventKind::kCopy);
    EXPECT_EQ(upload.api, "driver");
    EXPECT_EQ(upload.stream, native);
    outside += jitllm::test_support::EventLine(upload);
  }
  std::string lines = jitllm::test_support::ChunkLine(
      jitllm::test_support::Chunk{.evaluation = 1, .chunk = 1, .rows = 1, .n_past = 32});
  for (Event& event : events) {
    EXPECT_EQ(event.stream, native) << event.name;
    event.stream = reinterpret_cast<const void*>(0x10);  // NOLINT(performance-no-int-to-ptr)
    lines += jitllm::test_support::EventLine(event);
  }
  lines += jitllm::test_support::EndChunkLine();
  const std::string_view sample = jitllm::test_support::PlanRecordSample();
  EXPECT_EQ(lines, sample.substr(sample.find('\n') + 1));

  // For plan_compare.py by hand: the recording with this build's libraries.
  if (const char* out = std::getenv("JITLLM_TEST_PLAN_RECORD")) {  // NOLINT(concurrency-mt-unsafe)
    std::ofstream file(out);
    file << jitllm::test_support::HeaderLine("ggml_ops_test: the decode step's first launches",
                                             jitllm::test_support::LoadedCublas())
         << outside << lines;
    EXPECT_TRUE(file.good()) << out;
  }
}

// The registry declares every new implementation, binds it by identity and
// runs it over its nodes (D-053); a stale or foreign declaration binds none.
TEST_F(GgmlOpsTest, TheRegistryDeclaresBindsAndRunsEveryNewImplementation) {
  using jitllm::execution::Operation;
  const std::vector<jitllm::execution::Implementation> declared =
      jitllm::kernels::ggml::Implementations();
  const auto registry = jitllm::execution::Registry::Create(declared).value();
  const std::array<std::pair<const char*, Operation>, 9> expected = {{
      {"ggml.get_rows", Operation::kGetRows},
      {"ggml.set_rows", Operation::kSetRows},
      {"ggml.rope.neox", Operation::kRope},
      {"ggml.rope_set_rows.fused", Operation::kRopeSetRows},
      {"ggml.soft_max", Operation::kSoftMax},
      {"ggml.cont", Operation::kCont},
      {"ggml.swiglu", Operation::kSwiGlu},
      {"ggml.mul_mat_add.mmvf_fused", Operation::kMulMatAdd},
      {"ggml.mul_mat_glu.mmvf_fused", Operation::kMulMatGlu},
  }};
  for (const auto& [name, operation] : expected) {
    const std::size_t index = registry.Find(name).value_or(registry.size());
    ASSERT_LT(index, registry.size()) << name;
    EXPECT_EQ(registry.at(index).operation, operation) << name;
    const std::vector<jitllm::execution::Choice> choices = {
        {.operation = operation, .implementation = name}};
    const auto plan = jitllm::execution::Plan::Build(registry, choices).value();
    const auto bound = jitllm::execution::Resolve(registry, plan).value();
    const auto kernel = jitllm::kernels::ggml::Kernel::Bind(bound.at(0));
    ASSERT_TRUE(kernel.has_value()) << name;
    EXPECT_EQ(kernel->name(), name);
    EXPECT_EQ(kernel->operation(), operation);
  }
  // RMSNorm-mul's declarations are RmsNormMulKernel's, not Kernel's.
  EXPECT_FALSE(jitllm::kernels::ggml::Kernel::Bind(declared[0]).has_value());
  // A declaration from a build with another module digest binds nothing.
  jitllm::execution::Implementation stale = declared.back();
  stale.revision = "ggml tree 0; jitllm module 0";
  EXPECT_EQ(FailedCode(jitllm::kernels::ggml::Kernel::Bind(stale)), KernelError::kRejected);

  // A bound kernel runs its operation over its nodes, as the operation
  // does, and refuses the wrong number of nodes before any launch.
  const auto swiglu =
      jitllm::kernels::ggml::Kernel::Bind(declared[registry.Find("ggml.swiglu").value_or(0)])
          .value();
  EXPECT_EQ(swiglu.arity(), 1U);
  auto arena = TensorArena::Create(8).value();
  ggml_context* c = arena.context();
  ggml_tensor* gate =
      Place(Memory::kDeviceVmm, ggml_new_tensor_1d(c, GGML_TYPE_F32, kFfn), Values(21, kFfn, 6.0f));
  ggml_tensor* up =
      Place(Memory::kDeviceVmm, ggml_new_tensor_1d(c, GGML_TYPE_F32, kFfn), Values(22, kFfn, 2.0f));
  ggml_tensor* by_kernel = Place(Memory::kDeviceVmm, ggml_swiglu_split(c, gate, up));
  ggml_tensor* direct = Place(Memory::kDeviceVmm, ggml_swiglu_split(c, gate, up));
  auto launch = Launcher();
  const std::array<ggml_tensor*, 1> nodes = {by_kernel};
  EXPECT_TRUE(swiglu.Check(std::span<const ggml_tensor* const>(nodes.data(), 1)).has_value());
  Launched(swiglu.Run(*launch, nodes), "swiglu by kernel");
  Launched(jitllm::kernels::ggml::SwiGlu(*launch, direct), "swiglu");
  EXPECT_EQ(Bits(Download(by_kernel)), Bits(Download(direct)));
  const std::array<ggml_tensor*, 2> two = {by_kernel, direct};
  EXPECT_EQ(FailedCode(swiglu.Run(*launch, two)), KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());
}

// The pinned GGML definition uses the tanh approximation, rather than ERF.
double GemmaGelu(double x) {
  return 0.5 * x * (1.0 + std::tanh(0.79788456080286535588 * x * (1.0 + 0.044715 * x * x)));
}

TEST_F(GgmlOpsTest, GemmaGeGluMatchesTanhGeluAcrossRowsViewsAndMemoryDomains) {
  namespace kg = jitllm::kernels::ggml;
  const std::array<float, 14> values{0.0f,
                                     -0.0f,
                                     1e-8f,
                                     -1e-8f,
                                     0.1f,
                                     -0.1f,
                                     1.0f,
                                     -1.0f,
                                     4.0f,
                                     -4.0f,
                                     100.0f,
                                     -100.0f,
                                     std::numeric_limits<float>::max(),
                                     -std::numeric_limits<float>::max()};
  for (const auto memory : {Memory::kCudaMalloc, Memory::kDeviceVmm, Memory::kHostVmm}) {
    auto arena = TensorArena::Create(192).value();
    auto* c = arena.context();
    auto launch = Launcher();
    for (const std::int64_t width : {2112, 704}) {
      const std::int64_t used = width == 704 ? 8 : 1;
      constexpr std::int64_t rows = 4;
      std::vector<float> both(static_cast<std::size_t>(2 * width * used * rows));
      std::vector<double> want;
      for (std::int64_t row = 0; row < used * rows; ++row) {
        for (std::int64_t j = 0; j < width; ++j) {
          const auto ix = static_cast<std::size_t>(row * 2 * width + j);
          const float gate = values[static_cast<std::size_t>(row + j) % values.size()];
          const float up = static_cast<float>((row + j) % 5 - 2) * 0.25f;
          both[ix] = gate;
          both[ix + static_cast<std::size_t>(width)] = up;
          want.push_back(GemmaGelu(gate) * up);
        }
      }
      auto* packed =
          Place(memory, ggml_new_tensor_3d(c, GGML_TYPE_F32, 2 * width, used, rows), both);
      auto* gate = ggml_view_3d(c, packed, width, used, rows, packed->nb[1], packed->nb[2], 0);
      auto* up = ggml_view_3d(c, packed, width, used, rows, packed->nb[1], packed->nb[2],
                              static_cast<std::size_t>(width) * sizeof(float));
      auto* glu = Place(memory, ggml_geglu_split(c, gate, up));
      Launched(kg::GeGlu(*launch, glu), "batched strided Gemma GeGLU");
      const auto got = Download(glu);
      for (const auto value : got) {
        EXPECT_TRUE(std::isfinite(value));
      }
      ExpectClose(got, want, 3e-6, "Gemma GeGLU tanh reference");
      for (std::int64_t row = 0; row < used * rows; ++row) {
        auto* g = ggml_view_1d(c, packed, width, static_cast<std::size_t>(row) * packed->nb[1]);
        auto* u = ggml_view_1d(c, packed, width,
                               static_cast<std::size_t>(row) * packed->nb[1] +
                                   static_cast<std::size_t>(width) * sizeof(float));
        auto* solo = Place(memory, ggml_geglu_split(c, g, u));
        Launched(kg::GeGlu(*launch, solo), "solo Gemma GeGLU");
        const auto begin = got.begin() + row * width;
        EXPECT_EQ(Bits(Download(solo)), Bits(std::vector<float>(begin, begin + width)));
      }
      // A packed gate/up pair can be overwritten exactly in place.
      auto* g = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F32, width, rows),
                      Values(90, static_cast<std::size_t>(width * rows), 7));
      auto* u = Place(memory, ggml_new_tensor_2d(c, GGML_TYPE_F32, width, rows),
                      Values(91, static_cast<std::size_t>(width * rows)));
      auto* ordinary = Place(memory, ggml_geglu_split(c, g, u));
      Launched(kg::GeGlu(*launch, ordinary), "packed GeGLU");
      const auto expected = Download(ordinary);
      auto* inplace = ggml_geglu_split(c, g, u);
      TensorArena::Bind(inplace, reinterpret_cast<std::uintptr_t>(g->data));
      Launched(kg::GeGlu(*launch, inplace), "GeGLU in place over gate");
      EXPECT_EQ(Bits(Download(inplace)), Bits(expected));
      EXPECT_EQ(launch->scratch_peak(), Bytes(0));
    }
  }
}

TEST_F(GgmlOpsTest, GemmaGeluUnaryAndRegistryUseTheTanhPrimitive) {
  namespace kg = jitllm::kernels::ggml;
  auto arena = TensorArena::Create(16).value();
  auto* c = arena.context();
  auto launch = Launcher();
  std::vector<float> input = Values(87, 2112 * 4, 8);
  input[0] = std::numeric_limits<float>::max();
  input[1] = -input[0];
  input[2] = -0.0f;
  input[3] = 1e-8f;
  auto* x = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F32, 2112, 4), input);
  auto* gelu = Place(Memory::kDeviceVmm, ggml_gelu(c, x));
  Launched(kg::Unary(*launch, gelu), "Gemma GELU-tanh");
  std::vector<double> want;
  for (const float v : input) {
    want.push_back(GemmaGelu(v));
  }
  const auto expected = Download(gelu);
  for (const auto value : expected) {
    EXPECT_TRUE(std::isfinite(value));
  }
  ExpectClose(expected, want, 1e-5, "Gemma GELU-tanh reference");
  auto* inplace = ggml_gelu_inplace(c, x);
  Launched(kg::Unary(*launch, inplace), "GELU-tanh in place");
  EXPECT_EQ(Bits(Download(inplace)), Bits(expected));
  auto* up = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F32, 2112, 4),
                   Values(88, input.size()));
  auto* glu = Place(Memory::kDeviceVmm, ggml_geglu_split(c, x, up));
  const auto registry = jitllm::execution::Registry::Create(kg::Implementations()).value();
  const auto ix = registry.Find("ggml.geglu");
  ASSERT_TRUE(ix);
  const std::array<jitllm::execution::Choice, 1> choices{
      {{.operation = jitllm::execution::Operation::kGeGlu, .implementation = "ggml.geglu"}}};
  const auto plan = jitllm::execution::Plan::Build(registry, choices).value();
  const auto bound = jitllm::execution::Resolve(registry, plan).value();
  auto kernel = kg::Kernel::Bind(bound.at(0)).value();
  const std::array<ggml_tensor*, 1> nodes{glu};
  Launched(kernel.Run(*launch, nodes), "GeGLU registry");
  const auto by_registry = Download(glu);
  Launched(kg::GeGlu(*launch, glu), "GeGLU direct");
  EXPECT_EQ(Bits(Download(glu)), Bits(by_registry));
  auto* erf = Place(Memory::kDeviceVmm, ggml_geglu_erf_split(c, x, up));
  EXPECT_EQ(FailedCode(kg::GeGlu(*launch, erf)), KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());
}

TEST_F(GgmlOpsTest, GemmaFloatingGeGluFusionMatchesItsPrimitiveFallbackAndRefusesBatchFusion) {
  namespace kg = jitllm::kernels::ggml;
  auto arena = TensorArena::Create(48).value();
  auto* c = arena.context();
  auto launch = Launcher();
  for (const std::int64_t width : {2112, 704}) {
    auto* wg = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F16, 2816, width),
                     Halves(82, static_cast<std::size_t>(2816 * width), 0.025f));
    auto* wu = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F16, 2816, width),
                     Halves(83, static_cast<std::size_t>(2816 * width), 0.025f));
    for (const std::int64_t rows : {1, 4}) {
      auto* x = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F32, 2816, rows),
                      Values(84, static_cast<std::size_t>(2816 * rows)));
      auto* gate = Place(Memory::kDeviceVmm, ggml_mul_mat(c, wg, x));
      auto* up = Place(Memory::kDeviceVmm, ggml_mul_mat(c, wu, x));
      ggml_prec_set_acc(gate, GGML_PREC_F32);
      ggml_prec_set_acc(up, GGML_PREC_F32);
      auto* glu = Place(Memory::kDeviceVmm, ggml_geglu_split(c, gate, up));
      const auto fallback = [&] {
        Launched(rows == 1 ? kg::MulMatVecF(*launch, gate) : kg::MulMatF(*launch, gate),
                 "Gemma floating gate");
        Launched(rows == 1 ? kg::MulMatVecF(*launch, up) : kg::MulMatF(*launch, up),
                 "Gemma floating up");
        Launched(kg::GeGlu(*launch, glu), "Gemma floating GeGLU fallback");
      };
      fallback();
      const auto want = Download(glu);
      const auto measure = [&](bool fused) {
        cudaEvent_t start{}, end{};
        EXPECT_EQ(cudaEventCreate(&start), cudaSuccess);
        EXPECT_EQ(cudaEventCreate(&end), cudaSuccess);
        auto stream =
            reinterpret_cast<cudaStream_t>(execution_->Submission(stream_).value().handle);
        for (int i = 0; i < 8; ++i) {
          if (fused) {
            Launched(kg::MulMatVecGeGlu(*launch, gate, up, glu), "fusion warmup");
          } else {
            fallback();
          }
        }
        EXPECT_EQ(cudaEventRecord(start, stream), cudaSuccess);
        for (int i = 0; i < 64; ++i) {
          if (fused) {
            Launched(kg::MulMatVecGeGlu(*launch, gate, up, glu), "fusion timing");
          } else {
            fallback();
          }
        }
        EXPECT_EQ(cudaEventRecord(end, stream), cudaSuccess);
        EXPECT_EQ(cudaEventSynchronize(end), cudaSuccess);
        float ms = 0;
        EXPECT_EQ(cudaEventElapsedTime(&ms, start, end), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(start), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(end), cudaSuccess);
        return ms * 1000 / 64;
      };
      if (rows == 4) {
        EXPECT_EQ(FailedCode(kg::MulMatVecGeGlu(*launch, gate, up, glu)), KernelError::kRejected);
        if (std::getenv("JITLLM_GEMMA_ACTIVATION_TIMING") != nullptr &&
            (width == 704 || std::getenv("JITLLM_GEMMA_ACTIVATION_TIMING_N704_ONLY") == nullptr)) {
          const auto a1 = measure(false), a2 = measure(false);
          std::cout << "GEMMA_GEGLU_FLOAT k=2816 n=" << width << " rows=4 fallback1_us=" << a1
                    << " fallback2_us=" << a2 << " fused=unsupported\n";
        }
        continue;
      }
      ASSERT_TRUE(kg::MulMatVecFusible(*launch, up));
      Launched(kg::MulMatVecGeGlu(*launch, gate, up, glu), "Gemma floating fused GeGLU");
      const auto got = Download(glu);
      std::vector<double> reference(want.begin(), want.end());
      ExpectClose(got, reference, 3e-6, "fused versus primitive GeGLU");
      const auto registry = jitllm::execution::Registry::Create(kg::Implementations()).value();
      const std::array<jitllm::execution::Choice, 1> choices{
          {{.operation = jitllm::execution::Operation::kMulMatGeGlu,
            .implementation = "ggml.mul_mat_geglu.mmvf_fused"}}};
      const auto plan = jitllm::execution::Plan::Build(registry, choices).value();
      const auto bound = jitllm::execution::Resolve(registry, plan).value();
      auto kernel = kg::Kernel::Bind(bound.at(0)).value();
      const std::array<ggml_tensor*, 3> nodes{gate, up, glu};
      Launched(kernel.Run(*launch, nodes), "fused GeGLU registry");
      EXPECT_EQ(Bits(Download(glu)), Bits(got));
      if (std::getenv("JITLLM_GEMMA_ACTIVATION_TIMING") != nullptr &&
          (width == 704 || std::getenv("JITLLM_GEMMA_ACTIVATION_TIMING_N704_ONLY") == nullptr)) {
        // Include both products and the activation; inputs/weights stay resident.
        const auto a1 = measure(false), b = measure(true), a2 = measure(false);
        std::cout << "GEMMA_GEGLU_FLOAT k=2816 n=" << width << " rows=1 fallback1_us=" << a1
                  << " fused_us=" << b << " fallback2_us=" << a2 << "\n";
      }
    }
  }
}

TEST_F(GgmlOpsTest, GemmaAdversarialOddRowsAndUpOverwritePreserveEveryElement) {
  namespace kg = jitllm::kernels::ggml;
  auto arena = TensorArena::Create(32).value();
  auto* c = arena.context();
  auto launch = Launcher();
  constexpr std::int64_t width = 257;
  constexpr std::int64_t row_width = 2 * width + 7;
  constexpr std::int64_t rows = 6;
  const std::array<float, 12> gates{-0.0f,
                                    0.0f,
                                    std::numeric_limits<float>::denorm_min(),
                                    -std::numeric_limits<float>::denorm_min(),
                                    -6.0f,
                                    -5.0f,
                                    -4.0f,
                                    std::nextafter(-4.0f, 0.0f),
                                    1e-20f,
                                    -1e-20f,
                                    std::numeric_limits<float>::max(),
                                    -std::numeric_limits<float>::max()};
  std::vector<float> input(static_cast<std::size_t>(row_width * rows), 12345.0f);
  std::vector<double> want;
  for (std::int64_t row = 0; row < rows; ++row) {
    for (std::int64_t column = 0; column < width; ++column) {
      const auto at = static_cast<std::size_t>(row * row_width + column);
      const float gate = gates[static_cast<std::size_t>(row + column) % gates.size()];
      const float up = static_cast<float>((row + column) % 7 - 3) / 8.0f;
      input[at] = gate;
      input[at + width] = up;
      want.push_back(GemmaGelu(gate) * up);
    }
  }
  auto* storage =
      Place(Memory::kDeviceVmm, ggml_new_tensor_4d(c, GGML_TYPE_F32, row_width, 1, 3, 2), input);
  auto* gate =
      ggml_view_4d(c, storage, width, 1, 3, 2, storage->nb[1], storage->nb[2], storage->nb[3], 0);
  auto* up = ggml_view_4d(c, storage, width, 1, 3, 2, storage->nb[1], storage->nb[2],
                          storage->nb[3], width * sizeof(float));
  auto* glu = Place(Memory::kDeviceVmm, ggml_geglu_split(c, gate, up));
  const auto events = Record([&] { Launched(kg::GeGlu(*launch, glu), "odd padded GeGLU"); });
  ASSERT_EQ(events.size(), 1U);
  const auto got = Download(glu);
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_TRUE(std::isfinite(got[i])) << i;
    EXPECT_LE(std::abs(static_cast<double>(got[i]) - want[i]), 1e-5 * (1 + std::abs(want[i]))) << i;
  }
  EXPECT_EQ(Bits(Download(storage)), Bits(input));  // including row-padding canaries
  auto* packed_gate = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F32, width, rows),
                            Values(101, got.size()));
  auto* packed_up = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F32, width, rows),
                          Values(102, got.size()));
  auto* separate = Place(Memory::kDeviceVmm, ggml_geglu_split(c, packed_gate, packed_up));
  Launched(kg::GeGlu(*launch, separate), "independent GeGLU output");
  const auto expected = Download(separate);
  auto* overwrite_up = ggml_geglu_split(c, packed_gate, packed_up);
  TensorArena::Bind(overwrite_up, reinterpret_cast<std::uintptr_t>(packed_up->data));
  Launched(kg::GeGlu(*launch, overwrite_up), "overwrite packed up");
  EXPECT_EQ(Bits(Download(overwrite_up)), Bits(expected));
}

TEST_F(GgmlOpsTest, GemmaAdversarialRefusalsSubmitNoGpuWorkOrOutputWrites) {
  namespace kg = jitllm::kernels::ggml;
  auto arena = TensorArena::Create(32).value();
  auto* c = arena.context();
  auto launch = Launcher();
  auto* gate = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F32, 257, 3));
  auto* up = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F32, 257, 3));
  const std::vector<float> canary(257 * 3, -12345.0f);
  auto* glu = Place(Memory::kDeviceVmm, ggml_geglu_split(c, gate, up), canary);
  auto* old_view = ggml_view_2d(c, gate, 257, 3, gate->nb[1], 0);
  void* output = glu->data;
  Finish();
  const auto events = Record([&] {
    ++up->ne[1];
    EXPECT_EQ(FailedCode(kg::GeGlu(*launch, glu)), KernelError::kRejected);
    --up->ne[1];
    glu->data = static_cast<char*>(gate->data) + sizeof(float);
    EXPECT_EQ(FailedCode(kg::GeGlu(*launch, glu)), KernelError::kRejected);
    glu->data = output;
    gate->nb[0] += sizeof(float);
    EXPECT_EQ(FailedCode(kg::GeGlu(*launch, glu)), KernelError::kRejected);
    gate->nb[0] -= sizeof(float);
    glu->op_params[1] = 1;
    EXPECT_EQ(FailedCode(kg::GeGlu(*launch, glu)), KernelError::kRejected);
    glu->op_params[1] = 0;
    void* original_gate = gate->data;
    gate->data = up->data;
    glu->src[0] = old_view;
    EXPECT_EQ(FailedCode(kg::GeGlu(*launch, glu)), KernelError::kRejected);
    gate->data = original_gate;
    glu->src[0] = gate;
  });
  EXPECT_TRUE(events.empty());
  EXPECT_EQ(Bits(Download(glu)), Bits(canary));
  EXPECT_FALSE(launch->faulted());

  auto* x = Place(Memory::kDeviceVmm, ggml_new_tensor_1d(c, GGML_TYPE_F32, 32));
  auto* wg = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F16, 32, 64));
  auto* wu = Place(Memory::kDeviceVmm, ggml_new_tensor_2d(c, GGML_TYPE_F16, 32, 64));
  auto* g = ggml_mul_mat(c, wg, x);
  auto* u = ggml_mul_mat(c, wu, x);
  auto* fused = Place(Memory::kDeviceVmm, ggml_geglu_split(c, g, u));
  Finish();
  const auto precision_events = Record([&] {
    ggml_prec_set_acc(g, GGML_PREC_F32);
    EXPECT_EQ(FailedCode(kg::MulMatVecGeGlu(*launch, g, u, fused)), KernelError::kRejected);
    ggml_prec_set_acc(g, GGML_PREC_DEFAULT);
    ggml_prec_set_acc(u, GGML_PREC_F32);
    EXPECT_EQ(FailedCode(kg::MulMatVecGeGlu(*launch, g, u, fused)), KernelError::kRejected);
    ggml_prec_set_acc(g, GGML_PREC_F32);
    void* saved = fused->data;
    fused->data = wg->data;
    EXPECT_EQ(FailedCode(kg::MulMatVecGeGlu(*launch, g, u, fused)), KernelError::kRejected);
    fused->data = saved;
  });
  EXPECT_TRUE(precision_events.empty());
  EXPECT_FALSE(launch->faulted());
  const auto registry = jitllm::execution::Registry::Create(kg::Implementations()).value();
  const std::array<jitllm::execution::Choice, 1> wrong{
      {{.operation = jitllm::execution::Operation::kSwiGlu, .implementation = "ggml.geglu"}}};
  EXPECT_FALSE(jitllm::execution::Plan::Build(registry, wrong));
}

TEST_F(GgmlOpsTest, GemmaAdversarialF32AndBf16FusionPreserveBroadcastChannels) {
  namespace kg = jitllm::kernels::ggml;
  auto arena = TensorArena::Create(32).value();
  auto* c = arena.context();
  auto launch = Launcher();
  constexpr std::int64_t k = 64;
  constexpr std::int64_t n = 64;
  for (const auto type : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
    auto* wg = ggml_new_tensor_2d(c, type, k, n);
    auto* wu = ggml_new_tensor_2d(c, type, k, n);
    const auto gv = Values(103, k * n, 0.5f);
    const auto uv = Values(104, k * n, 0.5f);
    if (type == GGML_TYPE_F32) {
      Place(Memory::kDeviceVmm, wg, gv);
      Place(Memory::kDeviceVmm, wu, uv);
    } else {
      std::vector<ggml_bf16_t> gb, ub;
      for (std::size_t i = 0; i < gv.size(); ++i) {
        gb.push_back(ggml_fp32_to_bf16(gv[i]));
        ub.push_back(ggml_fp32_to_bf16(uv[i]));
      }
      Place(Memory::kDeviceVmm, wg, gb);
      Place(Memory::kDeviceVmm, wu, ub);
    }
    auto* x = Place(Memory::kDeviceVmm, ggml_new_tensor_3d(c, GGML_TYPE_F32, k, 1, 3),
                    Values(105, k * 3));
    auto* g = Place(Memory::kDeviceVmm, ggml_mul_mat(c, wg, x));
    auto* u = Place(Memory::kDeviceVmm, ggml_mul_mat(c, wu, x));
    auto* glu = Place(Memory::kDeviceVmm, ggml_geglu_split(c, g, u));
    // F32/BF16 defaults already use F32 accumulation. Weights broadcast
    // across three independent input channels, with one column per channel.
    ASSERT_TRUE(kg::MulMatVecFusible(*launch, u));
    Launched(kg::MulMatVecF(*launch, g), "broadcast gate fallback");
    Launched(kg::MulMatVecF(*launch, u), "broadcast up fallback");
    Launched(kg::GeGlu(*launch, glu), "broadcast activation fallback");
    const auto expected = Download(glu);
    Launched(kg::MulMatVecGeGlu(*launch, g, u, glu), "broadcast GeGLU fusion");
    EXPECT_EQ(Bits(Download(glu)), Bits(expected)) << ggml_type_name(type);
  }
}

}  // namespace
