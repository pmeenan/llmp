// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A model-fixture check of the selected Q4_1 head's existing scalar/joined
// VecQ kernels. The supervised recipe authenticates flat operands captured
// from Draft, never raw checkpoint data, before setting the fixture path.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_fast.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {
namespace kg = llmp::kernels::ggml;
namespace dv = llmp::providers;
constexpr std::int64_t kWidth = 2560, kRows = 47172, kColumns = 3;

class Qwen38QuantHeadTest : public ::testing::Test {
 protected:
  std::unique_ptr<dv::DeviceExecution> execution;
  dv::StreamId stream;
  std::unique_ptr<kg::LaunchContext> launch;
  std::unique_ptr<kg::TensorArena> arena;
  std::vector<void*> allocations;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  std::filesystem::path fixture;

  void SetUp() override {
    const char* path = std::getenv("LLMP_Q4_HEAD_FIXTURE");
    if (path == nullptr) GTEST_SKIP() << "needs the authenticated selected-head model fixture";
    fixture = path;
    ASSERT_TRUE(std::filesystem::is_directory(fixture));
    auto opened = dv::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(opened);
    execution = std::move(*opened);
    auto created = execution->CreateStream();
    ASSERT_TRUE(created);
    stream = *created;
    auto made = kg::LaunchContext::Create(
        0, *execution, stream,
        {.base = Allocate(256U << 20U), .size = llmp::base::Bytes(256U << 20U)});
    ASSERT_TRUE(made);
    launch = std::move(*made);
    auto tensors = kg::TensorArena::Create(256);
    ASSERT_TRUE(tensors);
    arena = std::make_unique<kg::TensorArena>(std::move(*tensors));
  }
  bool Finish() {
    if (!execution) return true;
    auto fence = execution->Record(stream);
    if (!fence) return false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      auto state = execution->Query(*fence);
      if (!state) return false;
      if (*state == dv::FenceState::kComplete) return execution->Release(*fence).has_value();
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
  void TearDown() override {
    if (!execution) return;
    if (!Finish()) {
      ADD_FAILURE() << "operand retirement unproven";
      (void)launch.release();
      (void)arena.release();
      (void)execution.release();
      return;
    }
    if (executable) EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    if (graph) EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    launch.reset();
    EXPECT_TRUE(execution->DestroyStream(stream));
    for (void* pointer : allocations) EXPECT_EQ(cudaFree(pointer), cudaSuccess);
  }
  std::uint64_t Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    EXPECT_EQ(cudaMalloc(&pointer, bytes), cudaSuccess);
    allocations.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }
  template <class T>
  std::vector<T> Read(const char* name, std::size_t count) {
    std::ifstream file(fixture / name, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(T))) {
      ADD_FAILURE() << "invalid operand " << name;
      return {};
    }
    file.seekg(0);
    std::vector<T> values(count);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (!file) {
      ADD_FAILURE() << "reading operand " << name;
      return {};
    }
    return values;
  }
  ggml_tensor* Place(ggml_tensor* tensor) {
    kg::TensorArena::Bind(tensor, Allocate(ggml_nbytes(tensor)));
    return tensor;
  }
  ggml_tensor* GuardedOutput(ggml_tensor* tensor) {
    constexpr std::size_t guard = 64;
    const auto base = Allocate(ggml_nbytes(tensor) + 2 * guard);
    EXPECT_EQ(cudaMemset(reinterpret_cast<void*>(base), 0x5a, ggml_nbytes(tensor) + 2 * guard),
              cudaSuccess);
    kg::TensorArena::Bind(tensor, base + guard);
    return tensor;
  }
  void OutputGuards(const ggml_tensor* tensor) {
    std::array<std::byte, 64> before{}, after{};
    const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
    ASSERT_EQ(cudaMemcpy(before.data(), reinterpret_cast<const void*>(address - before.size()),
                         before.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(after.data(), reinterpret_cast<const void*>(address + ggml_nbytes(tensor)),
                         after.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::ranges::all_of(before, [](std::byte b) { return b == std::byte{0x5a}; }));
    EXPECT_TRUE(std::ranges::all_of(after, [](std::byte b) { return b == std::byte{0x5a}; }));
  }
  std::vector<float> Download(ggml_tensor* tensor) {
    if (!Finish()) {
      ADD_FAILURE() << "operand completion unproven";
      return {};
    }
    std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    EXPECT_EQ(cudaMemcpy(values.data(), tensor->data, values.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }
  void CheckPrefixes(std::span<const std::int64_t> prefixes);
  void Same(std::span<const float> got, std::span<const float> want) {
    ASSERT_EQ(got.size(), want.size());
    ASSERT_FALSE(got.empty());
    EXPECT_TRUE(std::ranges::all_of(got, [](float x) { return std::isfinite(x); }));
    EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size_bytes()), 0);
  }
};

void Qwen38QuantHeadTest::CheckPrefixes(std::span<const std::int64_t> prefixes) {
  const auto weights = Read<std::byte>("head.q4_1", 75475200);
  const auto ids = Read<std::int32_t>("head.ids.i32", kRows);
  const auto first = Read<float>("first.inputs.f32", kWidth * kColumns);
  const auto changed = Read<float>("changed.inputs.f32", kWidth * kColumns);
  const auto first_drafts = Read<std::int32_t>("first.drafts.i32", kColumns);
  const auto changed_drafts = Read<std::int32_t>("changed.drafts.i32", kColumns);
  ASSERT_EQ(first_drafts.size(), kColumns);
  ASSERT_EQ(changed_drafts.size(), kColumns);
  const auto first_heads = Read<float>("first.logits.f32", kRows * kColumns);
  const auto changed_heads = Read<float>("changed.logits.f32", kRows * kColumns);
  ASSERT_EQ(weights.size(), 75475200U);
  ASSERT_EQ(ids.size(), kRows);
  ASSERT_EQ(first.size(), kWidth * kColumns);
  ASSERT_EQ(changed.size(), first.size());
  ASSERT_EQ(first_heads.size(), kRows * kColumns);
  ASSERT_EQ(changed_heads.size(), first_heads.size());
  ASSERT_NE(first, changed);
  // Both tested grids include column one. Prove the protected-peer replay
  // changes that column's operand and complete head rather than replaying
  // an accidentally identical slice from two otherwise different frontiers.
  ASSERT_NE(std::memcmp(first.data() + kWidth, changed.data() + kWidth,
                        static_cast<std::size_t>(kWidth) * sizeof(float)),
            0);
  ASSERT_NE(std::memcmp(first_heads.data() + kRows, changed_heads.data() + kRows,
                        static_cast<std::size_t>(kRows) * sizeof(float)),
            0);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ASSERT_GE(ids[i], 0);
    ASSERT_LT(ids[i], 248320);
    if (i) ASSERT_LT(ids[i - 1], ids[i]);
  }
  auto* c = arena->context();
  auto* parent = Place(ggml_new_tensor_2d(c, GGML_TYPE_Q4_1, kWidth, kRows));
  ASSERT_EQ(parent->nb[1], 1600U);
  ASSERT_EQ(cudaMemcpy(parent->data, weights.data(), weights.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  for (const auto rows : prefixes) {
    ASSERT_GE(rows, 1);
    ASSERT_LE(rows, kRows);
    SCOPED_TRACE(rows);
    ASSERT_NE(std::memcmp(first_heads.data() + kRows, changed_heads.data() + kRows,
                          static_cast<std::size_t>(rows) * sizeof(float)),
              0);
    auto* w = rows == kRows ? parent : ggml_view_2d(c, parent, kWidth, rows, parent->nb[1], 0);
    for (const std::int64_t columns : {2, 3}) {
      SCOPED_TRACE(columns);
      auto* input = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, columns));
      auto* q8 = Place(kg::QuantizeQ8(c, input));
      ASSERT_EQ(ggml_nbytes(q8), 2880U * static_cast<std::uint64_t>(columns));
      auto* joint = GuardedOutput(kg::VecQ(c, w, q8, nullptr, columns, false));
      kg::SetVecQOneToken(joint);
      ASSERT_TRUE(kg::CheckVecQ(joint));
      std::vector<ggml_tensor*> scalar_q8, scalar;
      for (std::int64_t column = 0; column < columns; ++column) {
        auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, 1);
        kg::TensorArena::Bind(x, reinterpret_cast<std::uintptr_t>(input->data) +
                                     static_cast<std::uint64_t>(column) * input->nb[1]);
        scalar_q8.push_back(Place(kg::QuantizeQ8(c, x)));
        scalar.push_back(GuardedOutput(kg::VecQ(c, w, scalar_q8.back(), nullptr, 1, false)));
        ASSERT_TRUE(kg::CheckVecQ(scalar.back()));
      }
      std::vector<float> uploaded;
      const auto upload = [&](std::span<const float> values) {
        values = values.first(static_cast<std::size_t>(kWidth * columns));
        uploaded.assign(values.begin(), values.end());
        EXPECT_EQ(
            cudaMemcpy(input->data, values.data(), values.size_bytes(), cudaMemcpyHostToDevice),
            cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      };
      const auto queue = [&] {
        EXPECT_TRUE(kg::RunQuantizeQ8(*launch, q8));
        EXPECT_TRUE(kg::RunVecQ(*launch, joint));
      };
      const auto check = [&](std::span<const float> expected,
                             std::span<const std::int32_t> drafts) {
        std::vector<float> prefix_heads;
        for (std::int64_t column = 0; column < columns; ++column) {
          const auto begin = expected.begin() + column * kRows;
          prefix_heads.insert(prefix_heads.end(), begin, begin + rows);
        }
        const std::span<const float> want(prefix_heads);
        const auto got = Download(joint);
        Same(got, want);
        OutputGuards(joint);
        for (std::size_t column = 0; column < scalar.size(); ++column) {
          EXPECT_TRUE(kg::RunQuantizeQ8(*launch, scalar_q8[column]));
          EXPECT_TRUE(kg::RunVecQ(*launch, scalar[column]));
          const auto alone = Download(scalar[column]);
          OutputGuards(scalar[column]);
          Same(alone, want.subspan(column * static_cast<std::size_t>(rows),
                                   static_cast<std::size_t>(rows)));
          if (alone.size() != static_cast<std::size_t>(rows) || got.size() != want.size()) continue;
          const auto best = std::max_element(alone.begin(), alone.end()) - alone.begin();
          const auto joined =
              std::max_element(got.begin() + static_cast<std::ptrdiff_t>(
                                                 column * static_cast<std::size_t>(rows)),
                               got.begin() + static_cast<std::ptrdiff_t>(
                                                 (column + 1) * static_cast<std::size_t>(rows))) -
              got.begin() - static_cast<std::ptrdiff_t>(column * static_cast<std::size_t>(rows));
          EXPECT_EQ(ids[static_cast<std::size_t>(best)], ids[static_cast<std::size_t>(joined)]);
          const auto expected_begin =
              want.begin() + static_cast<std::ptrdiff_t>(column * static_cast<std::size_t>(rows));
          const auto expected_best =
              std::max_element(expected_begin, expected_begin + rows) - expected_begin;
          EXPECT_EQ(ids[static_cast<std::size_t>(best)],
                    ids[static_cast<std::size_t>(expected_best)]);
          if (rows == kRows) EXPECT_EQ(ids[static_cast<std::size_t>(best)], drafts[column]);
        }
        const auto inputs_after = Download(input);
        ASSERT_EQ(inputs_after.size(), uploaded.size());
        EXPECT_EQ(
            std::memcmp(inputs_after.data(), uploaded.data(), uploaded.size() * sizeof(float)), 0);
      };
      upload(first);
      queue();
      check(first_heads, first_drafts);
      auto submission = execution->Submission(stream);
      ASSERT_TRUE(submission);
      auto native = reinterpret_cast<cudaStream_t>(submission->handle);
      ASSERT_EQ(cudaStreamBeginCapture(native, cudaStreamCaptureModeThreadLocal), cudaSuccess);
      queue();
      ASSERT_EQ(cudaStreamEndCapture(native, &graph), cudaSuccess);
      ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
      for (const auto& [values, heads] :
           {std::pair{&first, &first_heads}, std::pair{&changed, &changed_heads}}) {
        upload(*values);
        ASSERT_EQ(cudaMemset(joint->data, 0xff, ggml_nbytes(joint)), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(executable, native), cudaSuccess);
        check(*heads, values == &first ? first_drafts : changed_drafts);
      }
      // One changed owner/column cannot perturb other columns' arithmetic.
      auto peer = first;
      std::copy(changed.begin() + kWidth, changed.begin() + 2 * kWidth, peer.begin() + kWidth);
      auto peer_heads = first_heads;
      std::copy(changed_heads.begin() + kRows, changed_heads.begin() + 2 * kRows,
                peer_heads.begin() + kRows);
      auto peer_drafts = first_drafts;
      peer_drafts[1] = changed_drafts[1];
      upload(peer);
      ASSERT_EQ(cudaMemset(joint->data, 0xff, ggml_nbytes(joint)), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ASSERT_EQ(cudaGraphLaunch(executable, native), cudaSuccess);
      check(peer_heads, peer_drafts);
      ASSERT_TRUE(Finish());
      ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
      executable = nullptr;
      ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
      graph = nullptr;
    }
  }
}

TEST_F(Qwen38QuantHeadTest, ActualSelectedHeadsKeepEveryScalarColumnAndRefreshChangedReplay) {
  const std::array<std::int64_t, 1> rows{kRows};
  CheckPrefixes(rows);
}

TEST_F(Qwen38QuantHeadTest, ActualSelectedPrefixesKeepEveryScalarRowTailAndRefreshChangedReplay) {
  const std::array<std::int64_t, 6> rows{1, 31, 32, 33, 16385, 47171};
  CheckPrefixes(rows);
}
}  // namespace
