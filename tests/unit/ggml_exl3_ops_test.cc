// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML operations the native EXL3 plan adds (kernels/ggml/ops.h;
// docs/backend-proof.md, "Native EXL3 operation plan") under llmpalooza's
// launch context on a GB10 (label `gpu`; the over-read probe and the host
// checks also on a discrete GPU the build targets, `gpu-discrete`, D-082):
// - the casts (ggml.convert): F32 to F16 rounds to nearest even and F16 to
//   F32 widens, bit for bit as the host does;
// - the embedding (ggml.get_rows over a BF16 table) widens exactly;
// - the forced vector attention (ggml.flash_attn_ext.vec) is close to an
//   FP64 reference over K and V padded to 256 cells, never reads past its
//   operands' padded cells, and launches what the record lists
//   (docs/experiments/backend-proof-p0/exl3-op-plan.json): at each recorded
//   phase kind the same kernels, grids, blocks and shared memory, with
//   PlanFlashAttnVec's parallel blocks and scratch equal to what
//   launch_fattn draws (the record's attention_pool_scratch_bytes, each
//   pool block rounded to 256 bytes);
// - the host checks refuse what these launchers would not take;
// - the over-read probe: at the plan's shapes, with every operand in
//   device VMM flush against an unmapped granule (after it, then before
//   it), each operation completes and writes what it writes in cudaMalloc
//   memory.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "launch_recorder.h"
#include "plan_record.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace {

using llmp::base::Bytes;
using llmp::kernels::ggml::KernelError;
using llmp::kernels::ggml::LaunchContext;
using llmp::kernels::ggml::TensorArena;
using llmp::providers::DeviceExecution;
using llmp::providers::FenceState;
using llmp::providers::StreamId;
using llmp::test_support::Event;
using llmp::test_support::FailedCode;

constexpr std::int64_t kHead = 64;
constexpr std::int64_t kHeads = 14;
constexpr std::int64_t kKvHeads = 2;
constexpr float kScale = 0.125f;

std::uint16_t HalfBits(float value) {
  return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
}
float FromHalf(std::uint16_t bits) { return static_cast<float>(std::bit_cast<_Float16>(bits)); }

class GgmlExl3OpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
  }
  void TearDown() override {
    Finish();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : device_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
  }

  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

  std::uint64_t Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    EXPECT_EQ(cudaMalloc(&pointer, bytes), cudaSuccess);
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }

  std::unique_ptr<LaunchContext> Launcher(std::uint64_t workspace = 0) {
    const std::uint64_t base = workspace == 0 ? 0 : Allocate(workspace);
    auto launch =
        LaunchContext::Create(0, *execution_, stream_, {.base = base, .size = Bytes(workspace)});
    EXPECT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    return launch ? std::move(*launch) : nullptr;
  }

  template <typename T>
  ggml_tensor* Place(ggml_tensor* tensor, const std::vector<T>& data = {}) {
    const std::size_t size = ggml_nbytes(tensor);
    const std::uint64_t address = Allocate(size);
    TensorArena::Bind(tensor, address);
    if (!data.empty()) {
      EXPECT_EQ(data.size() * sizeof(T), size);
      EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), data.data(), size,  // NOLINT
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      // A copy from pageable memory may return before its DMA lands, and
      // the provider's stream does not wait for the legacy stream
      // (CU_STREAM_NON_BLOCKING): wait for it before any launch reads it.
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    return tensor;
  }

  template <typename T>
  std::vector<T> Download(const ggml_tensor* tensor) {
    Finish();
    std::vector<T> values(ggml_nbytes(tensor) / sizeof(T));
    EXPECT_EQ(cudaMemcpy(values.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }

  template <typename Run>
  std::vector<Event> Record(Run&& run) {
    std::vector<Event> launched;
    {
      llmp::test_support::Recording recording;
      std::forward<Run>(run)();
      launched = recording.Take();
    }
    return launched;
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
};

void Launched(const std::expected<void, llmp::kernels::ggml::KernelFailure>& result) {
  EXPECT_TRUE(result.has_value()) << (result ? "" : result.error().detail);
}

TEST_F(GgmlExl3OpsTest, ConvertRoundsToNearestEvenAndWidensExactly) {
  auto arena = TensorArena::Create(16).value();
  ggml_context* c = arena.context();
  const std::int64_t rows = 7;
  const std::int64_t width = 896;
  std::mt19937 random(7);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::normal_distribution<float> normal(0.0f, 3.0f);
  std::vector<float> values(static_cast<std::size_t>(rows * width));
  for (float& v : values) {
    v = normal(random);
  }
  // Ties, the largest half, overflow, subnormals and signed zero.
  values[0] = 1.0f + (1.0f / 2048.0f);
  values[1] = 65504.0f;
  values[2] = 65520.0f;
  values[3] = 3.0e-7f;
  values[4] = -0.0f;
  values[5] = 1.0f + (3.0f / 2048.0f);
  ggml_tensor* src = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, width, rows), values);
  ggml_tensor* dst = Place<std::uint16_t>(ggml_new_tensor_2d(c, GGML_TYPE_F16, width, rows));
  ggml_tensor* narrow = ggml_cpy(c, src, dst);
  auto launch = Launcher();
  const std::vector<Event> events =
      Record([&] { Launched(llmp::kernels::ggml::Convert(*launch, narrow)); });
  const auto halves = Download<std::uint16_t>(dst);
  for (std::size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(halves[i], HalfBits(values[i])) << i << ": " << values[i];
  }
  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(llmp::test_support::NormalizedKernelName(events[0].name),
            "_Z21cpy_scalar_contiguousIf6__halfEvPKcPcl");
  EXPECT_EQ(events[0].grid[0], ((rows * width) + 63) / 64);  // the record's [448] at 32 rows
  EXPECT_EQ(events[0].block[0], 64U);

  ggml_tensor* back =
      Place<float>(ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, width / kHead, rows));
  ggml_tensor* src16 = ggml_new_tensor_3d(c, GGML_TYPE_F16, kHead, width / kHead, rows);
  TensorArena::Bind(src16, reinterpret_cast<std::uintptr_t>(dst->data));
  ggml_tensor* widen = ggml_cpy(c, src16, back);
  Launched(llmp::kernels::ggml::Convert(*launch, widen));
  const auto widened = Download<float>(back);
  for (std::size_t i = 0; i < halves.size(); ++i) {
    ASSERT_EQ(std::bit_cast<std::uint32_t>(widened[i]),
              std::bit_cast<std::uint32_t>(FromHalf(halves[i])))
        << i;
  }

  // Refused: another shape of the same count (GGML's builder asserts the
  // count), the same type, BF16, overlapping operands.
  ggml_tensor* other = Place<std::uint16_t>(ggml_new_tensor_1d(c, GGML_TYPE_F16, width * rows));
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckConvert(ggml_cpy(c, src, other))),
            KernelError::kRejected);
  ggml_tensor* same = Place<float>(ggml_new_tensor_2d(c, GGML_TYPE_F32, width, rows));
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckConvert(ggml_cpy(c, src, same))),
            KernelError::kRejected);
  ggml_tensor* bf16 = Place<std::uint16_t>(ggml_new_tensor_2d(c, GGML_TYPE_BF16, width, rows));
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckConvert(ggml_cpy(c, src, bf16))),
            KernelError::kRejected);
  ggml_tensor* over = ggml_new_tensor_2d(c, GGML_TYPE_F16, width, rows);
  TensorArena::Bind(over, reinterpret_cast<std::uintptr_t>(src->data) + 256);
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckConvert(ggml_cpy(c, src, over))),
            KernelError::kRejected);
}

TEST_F(GgmlExl3OpsTest, EmbeddingWidensTheBf16TableExactly) {
  auto arena = TensorArena::Create(8).value();
  ggml_context* c = arena.context();
  const std::int64_t width = 896;
  const std::int64_t vocab = 300;
  std::mt19937 random(11);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::vector<std::uint16_t> table(static_cast<std::size_t>(vocab * width));
  for (std::uint16_t& v : table) {
    v = static_cast<std::uint16_t>(random());
    if ((v & 0x7F80U) == 0x7F80U) {
      v = static_cast<std::uint16_t>(v & 0xBFFFU);  // no NaN or infinity
    }
  }
  const std::vector<std::int32_t> ids = {5, 299, 0, 17, 17, 128, 64};
  ggml_tensor* rows = Place(ggml_new_tensor_2d(c, GGML_TYPE_BF16, width, vocab), table);
  ggml_tensor* index =
      Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, static_cast<std::int64_t>(ids.size())), ids);
  ggml_tensor* node = ggml_get_rows(c, rows, index);
  Place<float>(node);
  auto launch = Launcher();
  const std::vector<Event> events =
      Record([&] { Launched(llmp::kernels::ggml::GetRows(*launch, node)); });
  const auto out = Download<float>(node);
  for (std::size_t r = 0; r < ids.size(); ++r) {
    for (std::int64_t j = 0; j < width; ++j) {
      const std::uint16_t bits =
          table[(static_cast<std::size_t>(ids[r]) * static_cast<std::size_t>(width)) +
                static_cast<std::size_t>(j)];
      ASSERT_EQ(std::bit_cast<std::uint32_t>(
                    out[(r * static_cast<std::size_t>(width)) + static_cast<std::size_t>(j)]),
                static_cast<std::uint32_t>(bits) << 16U)
          << r << ", " << j;
    }
  }
  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(events[0].name, "_Z16k_get_rows_floatI13__nv_bfloat16fEvPKT_PKiPT0_ll5uint3mmmmmmmmm");
  EXPECT_EQ(events[0].grid, (std::array<unsigned, 3>{7, 4, 1}));  // the record's [n, 4, 1]
  EXPECT_EQ(events[0].block, (std::array<unsigned, 3>{256, 1, 1}));
}

struct AttentionCase {
  int rows = 0;
  int past = 0;
  int padded = 0;
  int parallel_blocks = 0;  // exl3-op-plan.json's shape_parameters
  std::uint64_t record_scratch = 0;
};

// q [rows, heads, head] F32; k, v [padded, kv heads, head] F16 bits; FP64.
std::vector<double> Reference(const AttentionCase& a, const std::vector<float>& q,
                              const std::vector<std::uint16_t>& k,
                              const std::vector<std::uint16_t>& v) {
  constexpr std::size_t kH = kHeads;
  constexpr std::size_t kD = kHead;
  constexpr std::size_t kG = kHeads / kKvHeads;
  constexpr std::size_t kKv = kKvHeads;
  const auto rows = static_cast<std::size_t>(a.rows);
  const auto past = static_cast<std::size_t>(a.past);
  std::vector<double> out(rows * kH * kD);
  for (std::size_t i = 0; i < rows; ++i) {
    const std::size_t attended = past + i + 1;
    for (std::size_t h = 0; h < kH; ++h) {
      const std::size_t g = h / kG;
      std::vector<double> scores(attended);
      double most = -std::numeric_limits<double>::infinity();
      for (std::size_t j = 0; j < attended; ++j) {
        double dot = 0;
        for (std::size_t d = 0; d < kD; ++d) {
          dot += static_cast<double>(q[(((i * kH) + h) * kD) + d]) *
                 FromHalf(k[(((j * kKv) + g) * kD) + d]);
        }
        scores[j] = dot * kScale;
        most = std::max(most, scores[j]);
      }
      double sum = 0;
      for (double& s : scores) {
        s = std::exp(s - most);
        sum += s;
      }
      for (std::size_t d = 0; d < kD; ++d) {
        double acc = 0;
        for (std::size_t j = 0; j < attended; ++j) {
          acc += scores[j] * FromHalf(v[(((j * kKv) + g) * kD) + d]);
        }
        out[(((i * kH) + h) * kD) + d] = acc / sum;
      }
    }
  }
  return out;
}

TEST_F(GgmlExl3OpsTest, VectorAttentionMatchesTheRecordAndAnFp64Reference) {
  // The recorded phase kinds (exl3-op-plan.json): prefills of 32, 145 and
  // 1,024 rows, and single-token steps with K padded to 256, 1,024 and
  // 1,280.
  const std::vector<AttentionCase> cases = {
      {.rows = 32, .past = 0, .padded = 256, .parallel_blocks = 3, .record_scratch = 354'816},
      {.rows = 145, .past = 0, .padded = 256, .parallel_blocks = 3, .record_scratch = 1'607'760},
      {.rows = 1024, .past = 0, .padded = 1024, .parallel_blocks = 3, .record_scratch = 11'356'160},
      {.rows = 1, .past = 32, .padded = 256, .parallel_blocks = 4, .record_scratch = 14'784},
      // The P3 addendum's kind (exl3-op-plan-g.json): K padded to 1,024.
      {.rows = 1, .past = 1023, .padded = 1024, .parallel_blocks = 13, .record_scratch = 48'048},
      {.rows = 1, .past = 1024, .padded = 1280, .parallel_blocks = 13, .record_scratch = 48'048},
  };
  std::mt19937 random(23);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (const AttentionCase& a : cases) {
    SCOPED_TRACE(testing::Message() << a.rows << " rows at " << a.past);
    auto arena = TensorArena::Create(32).value();
    ggml_context* c = arena.context();
    std::vector<float> q(static_cast<std::size_t>(a.rows) * kHeads * kHead);
    for (float& x : q) {
      x = normal(random);
    }
    // Cells below past + rows hold values; the padded cells hold zero.
    std::vector<std::uint16_t> k(static_cast<std::size_t>(a.padded) * kKvHeads * kHead, 0);
    std::vector<std::uint16_t> v(k.size(), 0);
    const std::size_t live = static_cast<std::size_t>(a.past + a.rows) * kKvHeads * kHead;
    for (std::size_t i = 0; i < live; ++i) {
      k[i] = HalfBits(normal(random));
      v[i] = HalfBits(normal(random));
    }
    std::vector<std::uint16_t> mask(static_cast<std::size_t>(a.rows) *
                                    static_cast<std::size_t>(a.padded));
    for (int i = 0; i < a.rows; ++i) {
      for (int j = 0; j < a.padded; ++j) {
        mask[(static_cast<std::size_t>(i) * static_cast<std::size_t>(a.padded)) +
             static_cast<std::size_t>(j)] = j <= a.past + i ? 0x0000 : 0xFC00;
      }
    }
    ggml_tensor* tq = Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, kHeads, a.rows), q);
    ggml_tensor* tk = Place(ggml_new_tensor_3d(c, GGML_TYPE_F16, kHead, kKvHeads, a.padded), k);
    ggml_tensor* tv = Place(ggml_new_tensor_3d(c, GGML_TYPE_F16, kHead, kKvHeads, a.padded), v);
    ggml_tensor* tm = Place(ggml_new_tensor_2d(c, GGML_TYPE_F16, a.padded, a.rows), mask);
    ggml_tensor* node =
        ggml_flash_attn_ext(c, ggml_permute(c, tq, 0, 2, 1, 3), ggml_permute(c, tk, 0, 2, 1, 3),
                            ggml_permute(c, tv, 0, 2, 1, 3), tm, kScale, 0.0f, 0.0f);
    ASSERT_TRUE(ggml_prec_set_acc(node, GGML_PREC_F32));
    Place<float>(node);

    auto planning = Launcher();
    const auto plan = llmp::kernels::ggml::PlanFlashAttnVec(*planning, node);
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    EXPECT_EQ(plan->parallel_blocks, a.parallel_blocks);
    EXPECT_EQ(plan->mask_prepass, a.rows >= 1024);
    // The record's scratch, with each pool block from a 256-byte boundary.
    const auto round = [](std::uint64_t bytes) { return (bytes + 255) / 256 * 256; };
    const std::uint64_t outputs = static_cast<std::uint64_t>(a.rows) * kHeads * kHead;
    const std::uint64_t rows = static_cast<std::uint64_t>(a.rows) * kHeads;
    const auto blocks = static_cast<std::uint64_t>(a.parallel_blocks);
    const std::uint64_t prepass =
        a.rows >= 1024 ? static_cast<std::uint64_t>((a.rows + 1) / 2) * sizeof(int) : 0;
    EXPECT_EQ(prepass + (blocks * outputs * 4) + (blocks * rows * 8), a.record_scratch);
    EXPECT_EQ(plan->scratch,
              round(prepass) + round(blocks * outputs * 4) + round(blocks * rows * 8));

    // Too little workspace is refused before anything is queued.
    auto starved = Launcher(plan->scratch - 256);
    EXPECT_EQ(FailedCode(llmp::kernels::ggml::FlashAttnVec(*starved, node)),
              KernelError::kRejected);

    auto launch = Launcher(plan->scratch);
    const std::vector<Event> events =
        Record([&] { Launched(llmp::kernels::ggml::FlashAttnVec(*launch, node)); });
    const auto out = Download<float>(node);
    const auto want = Reference(a, q, k, v);
    double error = 0;
    double norm = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
      ASSERT_TRUE(std::isfinite(out[i])) << i;
      error += (out[i] - want[i]) * (out[i] - want[i]);
      norm += want[i] * want[i];
    }
    // The vector kernel computes in F32 (relative RMS 2.3-4.1e-6 against
    // FP64 in P0's operation study).
    EXPECT_LT(std::sqrt(error / norm), 2e-5);

    // The launches, as the record lists them: the mask pre-pass from 1,024
    // rows, the vector kernel, the combine.
    const int columns = a.rows == 1 ? 1 : 2;
    const auto tiles = static_cast<unsigned>((a.rows + columns - 1) / columns);
    std::size_t e = 0;
    if (a.rows >= 1024) {
      ASSERT_GE(events.size(), 1U);
      EXPECT_EQ(events[e].name, "_Z25flash_attn_mask_to_KV_maxILi2EEvPK7__half2Piill");
      EXPECT_EQ(events[e].grid, (std::array<unsigned, 3>{tiles, 1, 1}));
      EXPECT_EQ(events[e].block, (std::array<unsigned, 3>{128, 1, 1}));
      ++e;
    }
    ASSERT_EQ(events.size(), e + 2);
    EXPECT_EQ(events[e].name,
              columns == 1 ? "_Z18flash_attn_ext_vecILi64ELi1EL9ggml_type1ELS0_1ELb0EEvPKcS2_S2_S2_"
                             "S2_PKiPfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil"
                           : "_Z18flash_attn_ext_vecILi64ELi2EL9ggml_type1ELS0_1ELb0EEvPKcS2_S2_S2_"
                             "S2_PKiPfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil");
    EXPECT_EQ(events[e].grid,
              (std::array<unsigned, 3>{tiles, static_cast<unsigned>(a.parallel_blocks), kHeads}));
    EXPECT_EQ(events[e].block, (std::array<unsigned, 3>{32, 4, 1}));
    EXPECT_EQ(events[e].shared + events[e].static_shared, columns == 1 ? 4352U : 4608U);
    EXPECT_EQ(events[e + 1].name, "_Z26flash_attn_combine_resultsILi64EEvPKfPK6float2Pfi");
    EXPECT_EQ(events[e + 1].grid,
              (std::array<unsigned, 3>{static_cast<unsigned>(a.rows), kHeads, 1}));
    EXPECT_EQ(events[e + 1].block, (std::array<unsigned, 3>{64, 1, 1}));
    EXPECT_EQ(events[e + 1].shared, static_cast<std::size_t>(a.parallel_blocks) * 8);
  }
}

// The over-read probe for the EXL3 plan's GGML operations, at its shapes:
// every operand (and the attention's pool workspace) in device VMM, first
// ending flush against an unmapped granule, then starting flush after
// one. Each operation must complete without a fault and write the same
// bytes as with its operands in cudaMalloc memory, so none reads or writes
// outside its tensors: the embedding over a BF16 table (the last row
// included), the casts both ways, the fused norm, RoPE, the residual add,
// SwiGLU, and the vector attention at odd and even row counts, with and
// without the mask pre-pass.
TEST_F(GgmlExl3OpsTest, OperationsReadAndWriteOnlyTheirTensors) {
  enum class Where : std::uint8_t { kMalloc, kEnd, kStart };
  auto memory = std::move(llmp::providers::cuda::OpenDeviceMemory(0).value());
  std::size_t device_class = 0;
  for (std::size_t i = 0; i < memory->Classes().size(); ++i) {
    if (memory->Classes()[i].kind == llmp::providers::BackingKind::kDevice) {
      device_class = i;
    }
  }
  const std::uint64_t granule = memory->Granularity().value();
  struct Mapped {
    llmp::providers::ReservationId reservation;
    llmp::providers::BackingId backing;
    std::uint64_t offset;
    std::uint64_t size;
  };
  std::vector<Mapped> mapped;
  const auto allocate = [&](Where where, std::uint64_t bytes) -> std::uint64_t {
    if (where == Where::kMalloc) {
      return Allocate(bytes);
    }
    const std::uint64_t size = (bytes + granule - 1) / granule * granule;
    const auto reservation = memory->Reserve(Bytes(size + granule)).value();
    const auto backing = memory->Create(device_class, Bytes(size)).value();
    const std::uint64_t offset = where == Where::kStart ? granule : 0;
    EXPECT_TRUE(memory->Map(reservation, Bytes(offset), backing).has_value());
    EXPECT_TRUE(memory
                    ->SetAccess(reservation, Bytes(offset), Bytes(size),
                                llmp::providers::Access::kReadWrite)
                    .has_value());
    mapped.push_back({reservation, backing, offset, size});
    const std::uint64_t base = memory->RangeOf(reservation).value().base + offset;
    return where == Where::kEnd ? base + size - bytes : base;
  };
  // Completion without a fault (a fault is an error, not a crash here).
  const auto drained = [&]() -> bool {
    const auto fence = execution_->Record(stream_);
    if (!fence) {
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
      const auto state = execution_->Query(*fence);
      if (!state) {
        return false;
      }
      if (*state == FenceState::kComplete) {
        return execution_->Release(*fence).has_value();
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
  };
  std::mt19937 random(31);  // NOLINT(bugprone-random-generator-seed): reproducible
  const auto words = [&](std::size_t bytes, bool half) {
    std::vector<std::uint16_t> out(bytes / 2);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (auto& v : out) {
      const float x = normal(random);
      v = half ? HalfBits(x) : static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(x) >> 16U);
    }
    return out;
  };
  const auto floats = [&](std::size_t count) {
    std::vector<float> out(count);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& v : out) {
      v = normal(random);
    }
    return out;
  };

  // One case: its operands' contents (the same at every placement; empty
  // for an output, which starts as 0xA5 bytes), and a build that makes its
  // tensors, placing each operand in order before any view of it is made.
  using Place = std::function<ggml_tensor*(ggml_tensor*)>;
  using Run =
      std::function<std::expected<void, llmp::kernels::ggml::KernelFailure>(LaunchContext&)>;
  struct Built {
    std::vector<ggml_tensor*> outputs;
    Run run;
    // The pool scratch its launch draws, once its tensors are placed.
    std::function<std::uint64_t()> scratch;
  };
  const auto bytes_of = [](const auto& values) {
    const auto view = std::as_bytes(std::span(values));
    return std::vector<std::byte>(view.begin(), view.end());
  };
  const auto check = [&](const std::string& what,
                         const std::vector<std::vector<std::byte>>& operands,
                         const std::function<Built(ggml_context*, const Place&)>& build) {
    SCOPED_TRACE(what);
    std::vector<std::vector<std::byte>> first;
    for (const Where where : {Where::kMalloc, Where::kEnd, Where::kStart}) {
      auto arena = TensorArena::Create(64).value();
      std::size_t placed = 0;
      const Place place = [&](ggml_tensor* tensor) {
        const std::size_t size = ggml_nbytes(tensor);
        const std::uint64_t address = allocate(where, size);
        TensorArena::Bind(tensor, address);
        void* data = reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
        if (placed < operands.size() && !operands[placed].empty()) {
          EXPECT_EQ(operands[placed].size(), size) << "operand " << placed;
          EXPECT_EQ(cudaMemcpy(data, operands[placed].data(),
                               std::min(size, operands[placed].size()), cudaMemcpyHostToDevice),
                    cudaSuccess);
        } else {
          EXPECT_EQ(cudaMemset(data, 0xA5, size), cudaSuccess);
        }
        ++placed;
        return tensor;
      };
      const Built built = build(arena.context(), place);
      ASSERT_EQ(placed, operands.size());
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const std::uint64_t scratch = built.scratch ? built.scratch() : 0;
      const std::uint64_t pool = scratch == 0 ? 0 : allocate(where, scratch);
      auto launch =
          LaunchContext::Create(0, *execution_, stream_, {.base = pool, .size = Bytes(scratch)});
      ASSERT_TRUE(launch.has_value()) << launch.error().detail;
      const auto ran = built.run(**launch);
      ASSERT_TRUE(ran.has_value()) << ran.error().detail;
      ASSERT_TRUE(drained()) << "a fault with operands at placement " << static_cast<int>(where);
      launch->reset();
      for (std::size_t o = 0; o < built.outputs.size(); ++o) {
        std::vector<std::byte> out(ggml_nbytes(built.outputs[o]));
        ASSERT_EQ(
            cudaMemcpy(out.data(), built.outputs[o]->data, out.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
        if (where == Where::kMalloc) {
          first.push_back(std::move(out));
        } else {
          EXPECT_TRUE(out == first[o])
              << "output " << o << " at placement " << static_cast<int>(where);
        }
      }
    }
  };

  namespace kg = llmp::kernels::ggml;
  constexpr std::int64_t kWidth = 896;
  constexpr std::int64_t kFfn = 4864;
  constexpr std::int64_t kVocab = 1000;
  // The embedding: a BF16 table, rows 0 and vocab - 1 among the IDs.
  {
    const std::vector<std::int32_t> ids = {kVocab - 1, 0, 17, kVocab - 1, 5, 998, 3};
    check("get_rows", {bytes_of(words(kWidth * kVocab * 2, false)), bytes_of(ids), {}},
          [&](ggml_context* c, const Place& place) {
            ggml_tensor* table = place(ggml_new_tensor_2d(c, GGML_TYPE_BF16, kWidth, kVocab));
            ggml_tensor* index = place(ggml_new_tensor_1d(c, GGML_TYPE_I32, 7));
            ggml_tensor* node = place(ggml_get_rows(c, table, index));
            return Built{.outputs = {node},
                         .run = [node](LaunchContext& l) { return kg::GetRows(l, node); },
                         .scratch = {}};
          });
  }
  for (const std::int64_t rows : {1, 145, 1023}) {
    const std::string at = " at " + std::to_string(rows) + " rows";
    // The casts: F32 to F16 over [width, rows] and K's [head, 2, rows];
    // F16 to F32 over [head, heads, rows].
    for (const auto& [ne0, ne1, narrow] :
         {std::tuple{kWidth, std::int64_t{1}, true}, std::tuple{kHead, kKvHeads, true},
          std::tuple{kHead, kHeads, false}}) {
      const auto count = static_cast<std::size_t>(ne0 * ne1 * rows);
      check(std::string(narrow ? "cast to F16" : "cast to F32") + at,
            {narrow ? bytes_of(floats(count)) : bytes_of(words(count * 2, true)), {}},
            [&, ne0 = ne0, ne1 = ne1, narrow = narrow](ggml_context* c, const Place& place) {
              ggml_tensor* src = place(
                  ggml_new_tensor_3d(c, narrow ? GGML_TYPE_F32 : GGML_TYPE_F16, ne0, ne1, rows));
              ggml_tensor* dst = place(
                  ggml_new_tensor_3d(c, narrow ? GGML_TYPE_F16 : GGML_TYPE_F32, ne0, ne1, rows));
              ggml_tensor* node = ggml_cpy(c, src, dst);
              return Built{.outputs = {dst},
                           .run = [node](LaunchContext& l) { return kg::Convert(l, node); },
                           .scratch = {}};
            });
    }
    const auto hidden = static_cast<std::size_t>(kWidth * rows);
    check("rms_norm_mul" + at, {bytes_of(floats(hidden)), bytes_of(floats(kWidth)), {}},
          [&](ggml_context* c, const Place& place) {
            ggml_tensor* x = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, rows));
            ggml_tensor* w = place(ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth));
            ggml_tensor* norm = ggml_rms_norm(c, x, 1e-6f);
            ggml_tensor* mul = place(ggml_mul(c, norm, w));
            return Built{
                .outputs = {mul},
                .run = [norm, mul](LaunchContext& l) { return kg::RmsNormMul(l, norm, mul); },
                .scratch = {}};
          });
    for (const std::int64_t heads : {kHeads, kKvHeads}) {
      std::vector<std::int32_t> positions(static_cast<std::size_t>(rows));
      for (std::size_t i = 0; i < positions.size(); ++i) {
        positions[i] = static_cast<std::int32_t>(300 + i);
      }
      check("rope over " + std::to_string(heads) + " heads" + at,
            {bytes_of(floats(static_cast<std::size_t>(kHead * heads * rows))),
             bytes_of(positions),
             {}},
            [&](ggml_context* c, const Place& place) {
              ggml_tensor* x = place(ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, heads, rows));
              ggml_tensor* pos = place(ggml_new_tensor_1d(c, GGML_TYPE_I32, rows));
              ggml_tensor* node = place(ggml_rope_ext(c, x, pos, nullptr, static_cast<int>(kHead),
                                                      GGML_ROPE_TYPE_NEOX, 32768, 1000000.0f, 1.0f,
                                                      0.0f, 1.0f, 32.0f, 1.0f));
              return Built{.outputs = {node},
                           .run = [node](LaunchContext& l) { return kg::Rope(l, node); },
                           .scratch = {}};
            });
    }
    check("add" + at, {bytes_of(floats(hidden)), bytes_of(floats(hidden)), {}},
          [&](ggml_context* c, const Place& place) {
            ggml_tensor* a = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, rows));
            ggml_tensor* b = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, rows));
            ggml_tensor* node = place(ggml_add(c, a, b));
            return Built{.outputs = {node},
                         .run = [node](LaunchContext& l) { return kg::Add(l, node); },
                         .scratch = {}};
          });
    const auto ffn = static_cast<std::size_t>(kFfn * rows);
    check("swiglu" + at, {bytes_of(floats(ffn)), bytes_of(floats(ffn)), {}},
          [&](ggml_context* c, const Place& place) {
            ggml_tensor* gate = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, kFfn, rows));
            ggml_tensor* up = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, kFfn, rows));
            ggml_tensor* node = place(ggml_swiglu_split(c, gate, up));
            return Built{.outputs = {node},
                         .run = [node](LaunchContext& l) { return kg::SwiGlu(l, node); },
                         .scratch = {}};
          });
  }
  // The vector attention: odd rows with two columns per block (145,
  // 1,023), one row, and 1,024 rows with the mask pre-pass; K and V padded
  // with zero cells, the pool workspace placed as the operands are.
  for (const auto& [rows, past] :
       {std::pair{145, 0}, std::pair{1023, 0}, std::pair{1024, 0}, std::pair{1, 1023}}) {
    const int padded = (past + rows + 255) / 256 * 256;
    const std::size_t cells = static_cast<std::size_t>(padded) * kKvHeads * kHead;
    const std::size_t live = static_cast<std::size_t>(past + rows) * kKvHeads * kHead;
    std::vector<std::uint16_t> k(cells, 0);
    std::vector<std::uint16_t> v(cells, 0);
    const auto live_k = words(live * 2, true);
    const auto live_v = words(live * 2, true);
    std::ranges::copy(live_k, k.begin());
    std::ranges::copy(live_v, v.begin());
    std::vector<std::uint16_t> mask(static_cast<std::size_t>(rows) *
                                    static_cast<std::size_t>(padded));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < padded; ++j) {
        mask[(static_cast<std::size_t>(i) * static_cast<std::size_t>(padded)) +
             static_cast<std::size_t>(j)] = j <= past + i ? 0x0000 : 0xFC00;
      }
    }
    check("attention at " + std::to_string(rows) + " rows",
          {bytes_of(floats(static_cast<std::size_t>(rows) * kHeads * kHead)),
           bytes_of(k),
           bytes_of(v),
           bytes_of(mask),
           {}},
          [&, rows = rows, padded = padded](ggml_context* c, const Place& place) {
            ggml_tensor* q = place(ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, kHeads, rows));
            ggml_tensor* tk = place(ggml_new_tensor_3d(c, GGML_TYPE_F16, kHead, kKvHeads, padded));
            ggml_tensor* tv = place(ggml_new_tensor_3d(c, GGML_TYPE_F16, kHead, kKvHeads, padded));
            ggml_tensor* m = place(ggml_new_tensor_2d(c, GGML_TYPE_F16, padded, rows));
            ggml_tensor* node = ggml_flash_attn_ext(
                c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, tk, 0, 2, 1, 3),
                ggml_permute(c, tv, 0, 2, 1, 3), m, kScale, 0.0f, 0.0f);
            EXPECT_TRUE(ggml_prec_set_acc(node, GGML_PREC_F32));
            place(node);
            return Built{.outputs = {node},
                         .run = [node](LaunchContext& l) { return kg::FlashAttnVec(l, node); },
                         .scratch = [this, node]() -> std::uint64_t {
                           auto planning = Launcher();
                           const auto plan = kg::PlanFlashAttnVec(*planning, node);
                           EXPECT_TRUE(plan.has_value());
                           return plan ? plan->scratch : 0;
                         }};
          });
  }

  for (const Mapped& m : mapped) {
    EXPECT_TRUE(memory->Unmap(m.reservation, Bytes(m.offset), Bytes(m.size)).has_value());
    EXPECT_TRUE(memory->Release(m.backing).has_value());
    EXPECT_TRUE(memory->Free(m.reservation).has_value());
  }
}

TEST_F(GgmlExl3OpsTest, AttentionChecksRefuseWhatTheLauncherWouldNotTake) {
  auto arena = TensorArena::Create(64).value();
  ggml_context* c = arena.context();
  const auto make = [&](int rows, int padded, ggml_type kv, ggml_type mask_type, float softcap,
                        bool prec) {
    ggml_tensor* q = Place<float>(ggml_new_tensor_3d(c, GGML_TYPE_F32, kHead, kHeads, rows));
    ggml_tensor* k = Place<std::uint16_t>(ggml_new_tensor_3d(c, kv, kHead, kKvHeads, padded));
    ggml_tensor* v = Place<std::uint16_t>(ggml_new_tensor_3d(c, kv, kHead, kKvHeads, padded));
    ggml_tensor* m = Place<std::uint16_t>(ggml_new_tensor_2d(c, mask_type, padded, rows));
    ggml_tensor* node =
        ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, k, 0, 2, 1, 3),
                            ggml_permute(c, v, 0, 2, 1, 3), m, kScale, 0.0f, softcap);
    if (prec) {
      ggml_prec_set_acc(node, GGML_PREC_F32);
    }
    Place<float>(node);
    return node;
  };
  EXPECT_TRUE(
      llmp::kernels::ggml::CheckFlashAttnVec(make(8, 256, GGML_TYPE_F16, GGML_TYPE_F16, 0, true))
          .has_value());
  // K not padded to 256 cells; BF16 K and V; no F32 precision; a soft cap.
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckFlashAttnVec(
                make(8, 200, GGML_TYPE_F16, GGML_TYPE_F16, 0, true))),
            KernelError::kRejected);
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckFlashAttnVec(
                make(8, 256, GGML_TYPE_BF16, GGML_TYPE_F16, 0, true))),
            KernelError::kRejected);
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckFlashAttnVec(
                make(8, 256, GGML_TYPE_F16, GGML_TYPE_F16, 0, false))),
            KernelError::kRejected);
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckFlashAttnVec(
                make(8, 256, GGML_TYPE_F16, GGML_TYPE_F16, 30.0f, true))),
            KernelError::kRejected);
  // From 1,024 rows the mask pre-pass reads two rows per tile unbounded: an
  // odd row count with one mask row per query would be read past its end.
  EXPECT_TRUE(llmp::kernels::ggml::CheckFlashAttnVec(
                  make(1024, 1024, GGML_TYPE_F16, GGML_TYPE_F16, 0, true))
                  .has_value());
  EXPECT_EQ(FailedCode(llmp::kernels::ggml::CheckFlashAttnVec(
                make(1025, 1280, GGML_TYPE_F16, GGML_TYPE_F16, 0, true))),
            KernelError::kRejected);
}

}  // namespace
