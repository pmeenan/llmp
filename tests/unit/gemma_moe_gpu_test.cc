// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "kernels/ggml/executor.h"
#include "kernels/ggml/gemma_moe.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "kernels/ggml/launch.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {
namespace kg = llmp::kernels::ggml;
namespace pr = llmp::providers;
using llmp::base::Bytes;
class GemmaMoeGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    execution = std::move(pr::cuda::OpenDeviceExecution(0).value());
    stream = execution->CreateStream().value();
    launch = std::move(*kg::LaunchContext::Create(0, *execution, stream, {.size = Bytes(0)}));
    arena = std::make_unique<kg::TensorArena>(kg::TensorArena::Create(2048).value());
  }
  bool Finish() {
    if (retirement_failed) return false;
    auto fence = execution->Record(stream);
    if (!fence) {
      ADD_FAILURE() << "completion fence could not be recorded";
      retirement_failed = true;
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      auto state = execution->Query(*fence);
      if (!state || std::chrono::steady_clock::now() >= deadline) {
        ADD_FAILURE() << "device work retirement is unproven; retain fixture owners";
        retirement_failed = true;
        return false;
      }
      if (*state == pr::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution->Release(*fence)) {
      ADD_FAILURE() << "completed fence release failed";
      retirement_failed = true;
      return false;
    }
    return true;
  }
  void TearDown() override {
    if (!Finish()) {
      // No completion proof: keep graph/context/provider and device storage
      // alive until process teardown. No destructor or cudaFree is retirement.
      std::ignore = new std::vector<kg::CapturedGraph>(std::move(graphs));
      std::ignore = launch.release();
      std::ignore = execution.release();
      std::ignore = arena.release();
      return;
    }
    graphs.clear();
    launch.reset();
    EXPECT_TRUE(execution->DestroyStream(stream));
    for (auto* p : memory) EXPECT_EQ(cudaFree(p), cudaSuccess);
  }
  ggml_context* c() { return arena->context(); }
  ggml_tensor* Place(ggml_tensor* t) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, ggml_nbytes(t)), cudaSuccess);
    memory.push_back(p);
    kg::TensorArena::Bind(t, reinterpret_cast<std::uintptr_t>(p));
    return t;
  }
  template <class T>
  void Upload(const ggml_tensor* t, const std::vector<T>& v) {
    ASSERT_EQ(v.size() * sizeof(T), ggml_nbytes(t));
    ASSERT_EQ(cudaMemcpy(t->data, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice),
              cudaSuccess);
    // Pageable copies on the legacy stream must finish before provider work.
    const auto completed = cudaDeviceSynchronize();
    if (completed != cudaSuccess) retirement_failed = true;
    ASSERT_EQ(completed, cudaSuccess);
  }
  template <class T>
  std::vector<T> Read(const ggml_tensor* t) {
    if (!Finish()) return {};
    std::vector<T> v(ggml_nbytes(t) / sizeof(T));
    const auto status = cudaMemcpy(v.data(), t->data, v.size() * sizeof(T), cudaMemcpyDeviceToHost);
    EXPECT_EQ(status, cudaSuccess);
    if (status != cudaSuccess) retirement_failed = true;
    return v;
  }
  template <class T>
  void Export(const std::string& stem, const std::vector<T>& v) {
    const auto* root = std::getenv("LLMP_GEMMA_MOE_ORACLE_ROOT");
    if (!root) return;
    const auto path = std::filesystem::path(root) / (stem + ".bin");
    ASSERT_FALSE(std::filesystem::exists(path));
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(T)));
    ASSERT_TRUE(f);
  }
  std::unique_ptr<pr::DeviceExecution> execution;
  pr::StreamId stream;
  std::unique_ptr<kg::LaunchContext> launch;
  std::unique_ptr<kg::TensorArena> arena;
  std::vector<void*> memory;
  std::vector<kg::CapturedGraph> graphs;
  bool retirement_failed = false;
};
TEST_F(GemmaMoeGpu, RegisteredBothOutputPlanPreservesTailAndFreshCapturedOperands) {
  auto registry = llmp::execution::Registry::Create(kg::Implementations());
  ASSERT_TRUE(registry);
  for (const auto rows : {1, 2, 4, 8, 128}) {
    auto* logits = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 128, rows));
    auto* probabilities = ggml_soft_max(c(), logits);
    auto* reshaped = ggml_reshape_3d(c(), probabilities, 1, 128, rows);
    auto* ids = ggml_argsort_top_k(c(), probabilities, 8);
    auto* gathered = ggml_get_rows(c(), reshaped, ids);
    auto* selected = ggml_reshape_2d(c(), gathered, 8, rows);
    auto* sum = ggml_sum_rows(c(), selected);
    auto* clamp =
        ggml_clamp(c(), sum, kg::kGemmaRouteClamp, std::numeric_limits<float>::infinity());
    auto* normalized = ggml_div(c(), selected, clamp);
    auto* weights = ggml_reshape_3d(c(), normalized, 1, 8, rows);
    std::vector<ggml_tensor*> nodes{probabilities, reshaped, ids->view_src, ids,        gathered,
                                    selected,      sum,      clamp,         normalized, weights};
    auto* experts = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 2816, 8, rows));
    auto* scales = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 8, rows));
    auto* scaled = ggml_mul(c(), experts, scales);
    auto* weighted = ggml_mul(c(), scaled, weights);
    nodes.push_back(scaled);
    nodes.push_back(weighted);
    auto* value = static_cast<ggml_tensor*>(nullptr);
    std::array<ggml_tensor*, 8> parts{};
    for (std::size_t i = 0; i < parts.size(); ++i) {
      parts[i] = ggml_view_2d(c(), weighted, 2816, rows, weighted->nb[2], i * weighted->nb[1]);
      nodes.push_back(parts[i]);
    }
    value = parts[0];
    for (std::size_t i = 1; i < parts.size(); ++i) {
      value = ggml_add(c(), value, parts[i]);
      nodes.push_back(value);
    }
    for (auto* tensor : nodes)
      if (!tensor->view_src) Place(tensor);
    kg::BindViews(nodes);
    kg::DeviceChoices choices;
    choices.fuse_gemma_route = choices.fuse_gemma_reduce = true;
    auto planned = kg::PlanGraph(nodes, false, choices);
    ASSERT_TRUE(planned);
    ASSERT_EQ(planned->steps.size(), 2U);
    EXPECT_EQ(planned->steps[0].nodes.size(), 10U);
    EXPECT_EQ(planned->steps[1].nodes.size(), 17U);
    auto bound = kg::BoundGraph::Bind(*registry, *planned);
    ASSERT_TRUE(bound);
    auto scratch = kg::PlanScratch(*launch, *planned);
    ASSERT_TRUE(scratch);
    EXPECT_EQ(*scratch, 0U);
    // Uniform logits exercise deterministic ties. Every expert is dyadic,
    // with independent row/column values and scale2, so reduction is exact.
    std::vector<float> input(static_cast<std::size_t>(rows) * 128, 0),
        expert_data(static_cast<std::size_t>(rows) * 8 * 2816),
        scale_data(static_cast<std::size_t>(rows) * 8, 2.0f);
    for (std::size_t i = 0; i < expert_data.size(); ++i) expert_data[i] = float(i % 31) / 32;
    std::vector<std::int32_t> sentinel(static_cast<std::size_t>(rows) * 128, -777777);
    Upload(logits, input);
    Upload(experts, expert_data);
    Upload(scales, scale_data);
    Upload(ids->view_src, sentinel);
    ASSERT_TRUE(bound->Run(*launch));
    const auto first_ids = Read<std::int32_t>(ids->view_src);
    const auto first_weights = Read<float>(normalized);
    const auto first_values = Read<float>(value);
    ASSERT_FALSE(retirement_failed);
    ASSERT_EQ(first_ids.size(), sentinel.size());
    ASSERT_EQ(first_weights.size(), static_cast<std::size_t>(rows) * 8);
    ASSERT_EQ(first_values.size(), static_cast<std::size_t>(rows) * 2816);
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
      for (std::size_t j = 0; j < 128; ++j)
        EXPECT_EQ(first_ids[r * 128 + j], j < 8 ? static_cast<std::int32_t>(j) : -777777);
      for (std::size_t j = 0; j < 8; ++j) EXPECT_EQ(first_weights[r * 8 + j], .125f);
      for (std::size_t col = 0; col < 2816; ++col) {
        float expected = 0;
        for (std::size_t j = 0; j < 8; ++j)
          expected += expert_data[(r * 8 + j) * 2816 + col] * .25f;
        EXPECT_EQ(first_values[r * 2816 + col], expected);
      }
    }
    auto capture = launch->Capture([&](kg::LaunchContext& l) { return bound->Run(l); });
    ASSERT_TRUE(capture);
    EXPECT_EQ(capture->nodes(), 2U);
    graphs.push_back(std::move(*capture));
    for (auto& x : expert_data) x = -x;
    Upload(experts, expert_data);
    ASSERT_TRUE(launch->Launch(graphs.back()));
    const auto fresh = Read<float>(value);
    ASSERT_FALSE(retirement_failed);
    ASSERT_EQ(fresh.size(), first_values.size());
    for (std::size_t i = 0; i < fresh.size(); ++i) EXPECT_EQ(fresh[i], -first_values[i]);
    EXPECT_EQ(Read<std::int32_t>(ids->view_src), first_ids);
    const auto address = ids->data;
    ids->data = static_cast<std::byte*>(ids->data) + 4;
    auto rejected = bound->Run(*launch);
    EXPECT_FALSE(rejected);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().error, kg::KernelError::kRejected);
    ids->data = address;
    // A stale binding is refused before the first launch; restoring the
    // checked address leaves the same bound plan usable.
    ASSERT_TRUE(bound->Run(*launch));
    ASSERT_TRUE(Finish());
  }
}
TEST_F(GemmaMoeGpu, LiteralRoutingReductionJoinsAndFreshCaptureKeepTheTailUntouched) {
  for (const auto rows : {1, 2, 4, 8, 128}) {
    auto* x = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 128, rows));
    auto* w = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 8, rows));
    auto* root = Place(ggml_argsort(c(), x, GGML_SORT_ORDER_DESC));
    auto* ids = ggml_view_2d(c(), root, 8, rows, root->nb[1], 0);
    kg::GemmaRouting route{
        {x, Bytes(ggml_nbytes(x))}, {w, Bytes(ggml_nbytes(w))}, {ids, Bytes(ggml_nbytes(root))}};
    auto* e = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 2816, 8, rows));
    auto* s = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 8, rows));
    auto* rw = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 8, rows));
    auto* out = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 2816, rows));
    kg::GemmaScaledReduction reduction{{e, Bytes(ggml_nbytes(e))},
                                       {s, Bytes(ggml_nbytes(s))},
                                       {rw, Bytes(ggml_nbytes(rw))},
                                       {out, Bytes(ggml_nbytes(out))}};
    std::vector<float> logits(static_cast<std::size_t>(rows) * 128),
        experts(static_cast<std::size_t>(rows) * 8 * 2816),
        scales(static_cast<std::size_t>(rows) * 8), weights(static_cast<std::size_t>(rows) * 8);
    std::vector<std::int32_t> sentinel(static_cast<std::size_t>(rows) * 128, -777777);
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
      for (std::size_t i = 0; i < 128; ++i)
        logits[r * 128 + i] = r % 3 == 0   ? 0.0f
                              : r % 3 == 1 ? float((i * 17 + r) % 128) * .125f
                                           : (i == 17 || i == 65 ? 80.0f : -80.0f);
      for (std::size_t j = 0; j < 8; ++j) {
        scales[r * 8 + j] = .75f + float(j) * .0625f;
        weights[r * 8 + j] = .03125f * float(j + 1);
        for (std::size_t col = 0; col < 2816; ++col)
          experts[(r * 8 + j) * 2816 + col] =
              float(static_cast<int>((col * 17 + j * 13 + r * 7) % 97) - 48) * .03125f;
      }
    }
    // Cancellation makes contribution order observable, unlike only small
    // dyadic sums. The first two scales/weights are exactly 1, so their
    // opposite large values cancel before the six small contributions.
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
      scales[r * 8] = scales[r * 8 + 1] = 1.0f;
      weights[r * 8] = weights[r * 8 + 1] = 1.0f;
      experts[(r * 8) * 2816] = 1e20f;
      experts[(r * 8 + 1) * 2816] = -1e20f;
      for (std::size_t j = 2; j < 8; ++j) experts[(r * 8 + j) * 2816] = .125f;
    }
    Upload(x, logits);
    Upload(root, sentinel);
    Upload(e, experts);
    Upload(s, scales);
    Upload(rw, weights);
    const auto run = [&]() -> std::expected<void, kg::KernelFailure> {
      if (auto q = kg::RunGemmaRouting(*launch, route); !q) return q;
      return kg::RunGemmaScaledReduction(*launch, reduction);
    };
    ASSERT_TRUE(run());
    auto iw = Read<float>(w);
    auto ii = Read<std::int32_t>(root);
    auto values = Read<float>(out);
    ASSERT_FALSE(retirement_failed);
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
      std::array<std::size_t, 128> expected{};
      std::iota(expected.begin(), expected.end(), 0);
      std::stable_sort(expected.begin(), expected.end(), [&](std::size_t a, std::size_t b) {
        return logits[r * 128 + a] > logits[r * 128 + b];
      });
      double total = 0;
      const auto maximum = logits[r * 128 + expected[0]];
      for (std::size_t j = 0; j < 8; ++j)
        total += std::exp(double(logits[r * 128 + expected[j]] - maximum));
      for (std::size_t j = 0; j < 8; ++j) {
        EXPECT_EQ(ii[r * 128 + j], expected[j]);
        EXPECT_NEAR(iw[r * 8 + j],
                    std::exp(double(logits[r * 128 + expected[j]] - maximum)) / total, 2e-6);
      }
      for (std::size_t j = 8; j < 128; ++j) EXPECT_EQ(ii[r * 128 + j], sentinel[r * 128 + j]);
      for (std::size_t col = 0; col < 2816; ++col) {
        float want = 0;
        for (std::size_t j = 0; j < 8; ++j)
          want += (experts[(r * 8 + j) * 2816 + col] * scales[r * 8 + j]) * weights[r * 8 + j];
        EXPECT_EQ(values[r * 2816 + col], want);  // Dyadic literal operands: exact arithmetic.
      }
    }
    // Independently submitted rows use the same arithmetic as the joined call.
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
      auto* sx = ggml_view_2d(c(), x, 128, 1, x->nb[1], r * x->nb[1]);
      auto* sw = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 8, 1));
      auto* sr = Place(ggml_argsort(c(), sx, GGML_SORT_ORDER_DESC));
      auto* si = ggml_view_2d(c(), sr, 8, 1, sr->nb[1], 0);
      kg::GemmaRouting solo{{sx, Bytes(512)}, {sw, Bytes(32)}, {si, Bytes(512)}};
      ASSERT_TRUE(kg::RunGemmaRouting(*launch, solo));
      const auto actual = Read<float>(sw);
      const auto actual_ids = Read<std::int32_t>(sr);
      ASSERT_FALSE(retirement_failed);
      EXPECT_EQ(std::memcmp(actual.data(), iw.data() + r * 8, 32), 0);
      EXPECT_EQ(std::memcmp(actual_ids.data(), ii.data() + r * 128, 32), 0);
      auto* se = ggml_view_3d(c(), e, 2816, 8, 1, e->nb[1], e->nb[2], r * e->nb[2]);
      auto* ss = ggml_view_3d(c(), s, 1, 8, 1, s->nb[1], s->nb[2], r * s->nb[2]);
      auto* srw = ggml_view_3d(c(), rw, 1, 8, 1, rw->nb[1], rw->nb[2], r * rw->nb[2]);
      auto* so = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 2816, 1));
      kg::GemmaScaledReduction sd{
          {se, Bytes(2816 * 8 * 4)}, {ss, Bytes(32)}, {srw, Bytes(32)}, {so, Bytes(2816 * 4)}};
      ASSERT_TRUE(kg::RunGemmaScaledReduction(*launch, sd));
      const auto actual_values = Read<float>(so);
      ASSERT_FALSE(retirement_failed);
      EXPECT_EQ(std::memcmp(actual_values.data(), values.data() + r * 2816, 2816 * 4), 0);
    }
    auto capture = launch->Capture([&](kg::LaunchContext&) { return run(); });
    ASSERT_TRUE(capture);
    EXPECT_EQ(capture->nodes(), 2U);
    graphs.push_back(std::move(*capture));
    for (auto& v : logits) v = -v;
    for (auto& v : experts)
      v = std::nextafter(-v, 1.0f);  // Non-dyadic phase for pinned-image comparison.
    Upload(x, logits);
    Upload(e, experts);
    Upload(root, sentinel);
    ASSERT_TRUE(launch->Launch(graphs.back()));
    const auto fresh_w = Read<float>(w);
    const auto fresh_i = Read<std::int32_t>(root);
    const auto fresh_out = Read<float>(out);
    ASSERT_FALSE(retirement_failed);
    ASSERT_TRUE(run());
    EXPECT_EQ(Read<float>(w), fresh_w);
    EXPECT_EQ(Read<std::int32_t>(root), fresh_i);
    EXPECT_EQ(Read<float>(out), fresh_out);
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r)
      for (std::size_t j = 8; j < 128; ++j) EXPECT_EQ(fresh_i[r * 128 + j], -777777);
    ASSERT_FALSE(retirement_failed);
    const auto stem = "rows" + std::to_string(rows);
    Export(stem + "-logits", logits);
    Export(stem + "-experts", experts);
    Export(stem + "-scales", scales);
    Export(stem + "-weights", weights);
    Export(stem + "-route", fresh_w);
    Export(stem + "-ids", fresh_i);
    Export(stem + "-values", fresh_out);
  }
}
TEST_F(GemmaMoeGpu, RefusalQueuesNothingAndLeavesOutputsAndContextUsable) {
  auto* x = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 128, 1));
  auto* w = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 8, 1));
  auto* root = Place(ggml_argsort(c(), x, GGML_SORT_ORDER_DESC));
  auto* ids = ggml_view_2d(c(), root, 8, 1, root->nb[1], 0);
  Upload(x, std::vector<float>(128, 0));
  Upload(w, std::vector<float>(8, -123));
  Upload(root, std::vector<std::int32_t>(128, -456));
  kg::GemmaRouting d{{x, Bytes(512)}, {w, Bytes(32)}, {ids, Bytes(511)}};
  auto q = kg::RunGemmaRouting(*launch, d);
  ASSERT_FALSE(q);
  EXPECT_EQ(q.error().error, kg::KernelError::kRejected);
  EXPECT_EQ(Read<float>(w), std::vector<float>(8, -123));
  EXPECT_EQ(Read<std::int32_t>(root), std::vector<std::int32_t>(128, -456));
  d.ids.bytes = Bytes(512);
  ASSERT_TRUE(kg::RunGemmaRouting(*launch, d));
  EXPECT_EQ(Read<float>(w), std::vector<float>(8, .125f));
}
}  // namespace
