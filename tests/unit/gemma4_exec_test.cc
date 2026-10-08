// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Actual 26B geometry/types, structured synthetic weights, complete local and
// global layer plans against an independent unrolled scalar oracle. This does
// not load the checkpoint or establish whole-model/reference quality parity.
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
#include <expected>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "engine/gemma4_plan.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "gemma4_fixture.h"
#include "gemma4_scalar.h"
#include "ggml-impl.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {
namespace kg = llmp::kernels::ggml;
namespace en = llmp::engine;
namespace md = llmp::model;
namespace scalar = llmp::test_support::gemma4_scalar;
namespace fixture = llmp::test_support::gemma4;
using llmp::base::Bytes;
class Gemma4ExecTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(*llmp::providers::cuda::OpenDeviceExecution(0));
    stream_ = execution_->CreateStream().value();
    int major = 0, minor = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
    const auto cb = kg::CublasHandle::UpstreamWorkspace(100 * major + 10 * minor);
    auto handle = kg::CublasHandle::Create(0, *execution_, stream_,
                                           {.base = Allocate(cb.value()), .size = cb});
    ASSERT_TRUE(handle);
    cublas_ = std::move(*handle);
    auto launch = kg::LaunchContext::Create(
        0, *execution_, stream_, {.base = Allocate(128U << 20U), .size = Bytes{128U << 20U}},
        cublas_.get());
    ASSERT_TRUE(launch);
    launch_ = std::move(*launch);
    registry_ = std::make_unique<llmp::execution::Registry>(
        std::move(*llmp::execution::Registry::Create(kg::Implementations())));
    resources_ = fixture::Resources(26);
    binding_ = std::move(*md::BindGemma4(p_, "gemma4", resources_));
    state_ = std::move(*md::Gemma4State(p_, 4096, 4));
    model_.profile = &p_;
    model_.binding = &binding_;
    model_.state = &state_;
    for (std::uint32_t i = 0; i < 4; ++i)
      model_.slots.push_back({Allocate(state_.bytes), state_.bytes});
  }
  void TearDown() override {
    const auto fence = execution_->Record(stream_);
    ASSERT_TRUE(fence);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    auto state = llmp::providers::FenceState::kPending;
    while ((state = execution_->Query(*fence).value()) == llmp::providers::FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, llmp::providers::FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(*fence));
    launch_.reset();
    cublas_.reset();
    ASSERT_TRUE(execution_->DestroyStream(stream_));
    for (auto* p : allocations_) EXPECT_EQ(cudaFree(p), cudaSuccess);
  }
  std::uint64_t Allocate(std::uint64_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, std::max<std::uint64_t>(bytes, 256)), cudaSuccess);
    allocations_.push_back(p);
    return reinterpret_cast<std::uintptr_t>(p);
  }
  std::string Role(const md::Gemma4Tensor& t) const {
    std::uint32_t ordinary = 0, array = 0;
    for (const auto& r : resources_) {
      const auto index = r.expert_array ? array++ : ordinary++;
      if (r.expert_array == t.expert_array && index == t.index) return r.roles[0];
    }
    return {};
  }
  void Weights(const kg::Gemma4Graph& g) {
    for (const auto& w : g.weights) {
      auto& domain = w.resource.expert_array ? model_.arrays : model_.resources;
      domain.resize(std::max(domain.size(), std::size_t{w.resource.index} + 1));
      if (domain[w.resource.index].address != 0) continue;
      const auto* t = w.tensor;
      const auto bytes = w.resource.readable +
                         (w.resource.expert_array ? (std::uint64_t{p_.experts} - 1) * t->nb[2] : 0);
      std::vector<std::byte> data(static_cast<std::size_t>(bytes));
      const auto role = Role(w.resource);
      if (w.resource.ne.size() == 1) {
        std::vector<float> values(static_cast<std::size_t>(t->ne[0]));
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] = role == "rope_freqs.weight"
                          ? (i < 64 ? 1.0f : 1e30f)
                          : static_cast<float>(scalar::Vector(role, static_cast<std::uint32_t>(i)));
        std::memcpy(data.data(), values.data(), values.size() * sizeof(float));
      } else {
        std::vector<float> row(static_cast<std::size_t>(t->ne[0]));
        std::vector<std::byte> encoded(t->nb[1]);
        auto& coefficients = decoded_[role];
        coefficients.resize(static_cast<std::size_t>(t->ne[1] * t->ne[2]));
        for (std::int64_t e = 0; e < t->ne[2]; ++e) {
          for (std::int64_t j = 0; j < t->ne[1]; ++j) {
            std::ranges::fill(
                row, static_cast<float>(scalar::Coefficient(role, static_cast<std::uint32_t>(j),
                                                            static_cast<std::uint32_t>(e))));
            const auto written =
                ggml_quantize_chunk(t->type, row.data(), encoded.data(), 0, 1, t->ne[0], nullptr);
            EXPECT_EQ(written, t->nb[1]);
            const auto block = ggml_blck_size(t->type);
            std::vector<float> decoded(static_cast<std::size_t>(block));
            if (t->type == GGML_TYPE_F32)
              std::memcpy(decoded.data(), encoded.data(), sizeof(float));
            else
              ggml_get_type_traits(t->type)->to_float(encoded.data(), decoded.data(), block);
            EXPECT_EQ(*std::min_element(decoded.begin(), decoded.end()),
                      *std::max_element(decoded.begin(), decoded.end()));
            coefficients[static_cast<std::size_t>(e * t->ne[1] + j)] = decoded[0];
            std::memcpy(data.data() + static_cast<std::size_t>(e) * t->nb[2] +
                            static_cast<std::size_t>(j) * t->nb[1],
                        encoded.data(), encoded.size());
          }
        }
      }
      const auto address = Allocate(bytes);
      domain[w.resource.index] = {address, bytes};
      EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), data.data(), data.size(),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  void State(std::uint32_t layer, std::uint32_t slot, std::uint32_t past) {
    const auto d = p_.head_dim(layer), heads = p_.kv_heads(layer);
    for (const bool value : {false, true}) {
      const auto& t = state_.tensors[std::size_t{layer} * 2 + (value ? 1U : 0U)];
      std::vector<ggml_fp16_t> data(std::size_t{t.width} * t.cells);
      for (std::uint32_t cell = 0; cell < t.cells; ++cell) {
        const auto position = scalar::AbsoluteCell(past, cell, t.cells, t.local);
        for (std::uint32_t h = 0; h < heads; ++h)
          for (std::uint32_t i = 0; i < d; ++i) {
            data[std::size_t{cell} * t.width + h * d + i] =
                ggml_fp32_to_fp16(scalar::Initial(position, i, h, value, positive_values_));
          }
      }
      EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(model_.slots[slot].address + t.offset),
                           data.data(), t.bytes, cudaMemcpyHostToDevice),
                cudaSuccess);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  struct Run {
    md::Gemma4ChunkInputs chunk;
    std::vector<float> input;
    en::Gemma4HostInputs sources;
    std::unique_ptr<en::Gemma4Planned> planned;
    std::vector<float> output;
    std::vector<ggml_fp16_t> keys, values;
  };
  Run Execute(std::uint32_t layer, std::span<const std::uint32_t> slots, std::uint32_t rows,
              std::uint32_t past, bool shared, bool head = false, bool fused_norms = false,
              bool device_masks = false, bool rope_store = false, bool keep_rope = false,
              bool norm_chains = false, bool moe_chains = false) {
    Run run;
    std::vector<std::vector<std::int32_t>> tokens(slots.size(), std::vector<std::int32_t>(rows, 1));
    std::vector<md::Gemma4Segment> segments;
    kg::Gemma4ChunkShape shape;
    std::vector<std::int32_t> frontier;
    for (std::size_t i = 0; i < slots.size(); ++i) {
      const auto slot = slots[i];
      segments.push_back({slot, past + slot * 7, tokens[i]});
      State(layer, slot, past + slot * 7);
      for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t j = 0; j < p_.width; ++j) {
          run.input.push_back(0.15f + static_cast<float>((j * 3 + slot * 11 + r * 7) % 31) / 32);
        }
      if (head) frontier.push_back(static_cast<std::int32_t>((i + 1) * rows - 1));
    }
    auto chunk = md::Gemma4Chunk(p_, state_, segments, !device_masks);
    EXPECT_TRUE(chunk) << (chunk ? "" : chunk.error());
    if (!chunk) return run;
    run.chunk = std::move(*chunk);
    for (const auto& s : run.chunk.segments)
      shape.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
    shape.outputs = static_cast<std::uint32_t>(frontier.size());
    model_.options = {.expert_stride = {},
                      .first_layer = layer,
                      .layer_count = 1,
                      .hidden_input = true,
                      .head = head,
                      .shared_q8 = shared,
                      .device_masks = device_masks,
                      .rope_store = rope_store};
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(p_, slots.size()));
    EXPECT_TRUE(arena);
    auto graph = kg::BuildGemma4Graph(*arena, p_, binding_, state_, shape, model_.options);
    EXPECT_TRUE(graph) << (graph ? "" : graph.error().detail);
    if (!graph) return run;
    Weights(*graph);
    std::vector<std::string> kept{std::string("blk.") + std::to_string(layer) + ".slot." +
                                  std::to_string(slots[0]) + ".attention"};
    if (keep_rope)
      kept.push_back(std::string("blk.") + std::to_string(layer) + ".slot." +
                     std::to_string(slots[0]) + ".k_rope");
    auto choices = kg::DeviceChoicesOf(*launch_);
    choices.fuse_norms = fused_norms;
    choices.fuse_norm_rope = norm_chains;
    choices.fuse_norm_add = norm_chains;
    choices.fuse_gemma_route = moe_chains;
    choices.fuse_gemma_reduce = moe_chains;
    auto measured = en::PlanGemma4Chunk(model_, shape, choices, 0, 0, kept);
    EXPECT_TRUE(measured) << (measured ? "" : measured.error());
    if (!measured) return run;
    const auto activation = Allocate((*measured)->placement.extent);
    auto planned = en::PlanGemma4Chunk(model_, shape, choices, activation,
                                       (*measured)->placement.extent, kept);
    EXPECT_TRUE(planned) << (planned ? "" : planned.error());
    if (!planned) return run;
    run.planned = std::move(*planned);
    auto bound = en::BindPlanned(*run.planned, *launch_, *registry_, "Gemma synthetic layer");
    EXPECT_TRUE(bound) << (bound ? "" : bound.error());
    if (!bound) return run;
    std::size_t fused = 0, unfused = 0, rope_stores = 0;
    for (const auto& step : run.planned->plan.steps) {
      fused += step.implementation == "ggml.rms_norm_mul.fused";
      unfused += step.implementation == "ggml.rms_norm_mul.unfused";
      rope_stores += step.operation == llmp::execution::Operation::kRopeSetRows;
    }
    std::cout << "GEMMA_BOUND_POLICY layer=" << layer << " rows=" << rows
              << " slots=" << slots.size() << " shared_q8=" << shared
              << " fuse_norms=" << fused_norms << " norm_fused=" << fused
              << " norm_unfused=" << unfused << " rope_store_fused=" << rope_stores << '\n';
    EXPECT_EQ(rope_stores, rope_store ? slots.size() - (keep_rope ? 1U : 0U) : 0U);
    const auto budget = en::Gemma4SourceBytes(run.planned->graph).value();
    auto sources = en::Gemma4Sources(run.planned->graph, run.chunk, frontier, run.input, budget);
    EXPECT_TRUE(sources) << (sources ? "" : sources.error());
    if (!sources) return run;
    run.sources = std::move(*sources);
    for (const auto& [t, src] : run.sources.sources)
      EXPECT_EQ(cudaMemcpy(t->data, src, ggml_nbytes(t), cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto launched = run.planned->bound->Run(*launch_);
    EXPECT_TRUE(launched) << (launched ? "" : launched.error().detail);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto* out = head ? run.planned->graph.logits : run.planned->graph.hidden;
    run.output.resize(ggml_nbytes(out) / sizeof(float));
    EXPECT_EQ(cudaMemcpy(run.output.data(), out->data, ggml_nbytes(out), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (const auto slot : slots)
      for (const bool value : {false, true}) {
        const auto& t = state_.tensors[std::size_t{layer} * 2 + (value ? 1U : 0U)];
        auto& data = value ? run.values : run.keys;
        const auto first = data.size();
        data.resize(first + static_cast<std::size_t>(t.bytes / 2));
        EXPECT_EQ(cudaMemcpy(data.data() + first,
                             reinterpret_cast<const void*>(model_.slots[slot].address + t.offset),
                             t.bytes, cudaMemcpyDeviceToHost),
                  cudaSuccess);
      }
    return run;
  }
  void Reference(const Run& run, std::uint32_t layer, bool strict = true, bool encoded = false,
                 bool rounded_phase = false, bool observed_attention = false,
                 bool rounded_projection = false) {
    const scalar::CoefficientOracle coefficients = [&](std::string_view role, std::uint32_t row,
                                                       std::uint32_t expert) {
      if (!encoded) return scalar::Coefficient(role, row, expert);
      const auto& values = decoded_.at(std::string(role));
      const auto width = values.size() / (role.contains("_exps.weight") ? p_.experts : 1U);
      return values[std::size_t{expert} * width + row];
    };
    ASSERT_NE(run.planned, nullptr);
    std::vector<double> expected;
    for (const auto& s : run.chunk.segments) {
      const std::vector<double> input(
          run.input.begin() + static_cast<std::ptrdiff_t>(std::size_t{s.first_row} * p_.width),
          run.input.begin() +
              static_cast<std::ptrdiff_t>(std::size_t{s.first_row + s.rows} * p_.width));
      std::vector<double> attention;
      if (observed_attention) {
        auto* tensor = run.planned->graph.Named(std::string("blk.") + std::to_string(layer) +
                                                ".slot." + std::to_string(s.slot) + ".attention");
        std::vector<float> values(ggml_nbytes(tensor) / sizeof(float));
        ASSERT_EQ(
            cudaMemcpy(values.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost),
            cudaSuccess);
        attention.assign(values.begin(), values.end());
      }
      auto ref = scalar::Layer(p_, layer, s.n_past, input, s.rows,
                               p_.local(layer) ? state_.local_cells : state_.global_cells,
                               positive_values_, coefficients, rounded_phase, attention,
                               rounded_projection);
      expected.insert(expected.end(), ref.output.begin(), ref.output.end());
    }
    ASSERT_EQ(run.output.size(), expected.size());
    double error = 0, norm = 0, max = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const double delta = run.output[i] - expected[i];
      error += delta * delta;
      norm += expected[i] * expected[i];
      max = std::max(max, std::abs(delta));
      EXPECT_TRUE(std::isfinite(run.output[i]));
    }
    std::cout << "Gemma layer " << layer << " shared=" << run.planned->graph.options.shared_q8
              << " slots=" << run.chunk.segments.size() << " positive_values=" << positive_values_
              << " encoded=" << encoded << " rounded_phase=" << rounded_phase
              << " observed_attention=" << observed_attention
              << " rounded_projection=" << rounded_projection << " NMSE=" << error / norm
              << " max=" << max << '\n';
    if (strict) {
      EXPECT_LT(error / norm, 1e-5);
      EXPECT_LT(max, 0.015);
    }
  }
  void Stage(const Run& run) {
    const auto submission = execution_->Submission(stream_);
    ASSERT_TRUE(submission);
    auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
    for (const auto& [t, src] : run.sources.sources) {
      ASSERT_EQ(cudaMemcpyAsync(t->data, src, ggml_nbytes(t), cudaMemcpyHostToDevice, stream),
                cudaSuccess);
    }
  }
  void Repeat(const Run& run) {
    ASSERT_NE(run.planned, nullptr);
    ASSERT_TRUE(run.planned->bound.has_value());
    Stage(run);
    ASSERT_TRUE(run.planned->bound->Run(*launch_));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto* out =
        run.planned->graph.options.head ? run.planned->graph.logits : run.planned->graph.hidden;
    std::vector<float> again(run.output.size());
    ASSERT_EQ(cudaMemcpy(again.data(), out->data, ggml_nbytes(out), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(again.data(), run.output.data(), ggml_nbytes(out)), 0);
    auto capture = launch_->Capture(
        [&](kg::LaunchContext& launch) { return run.planned->bound->Run(launch); });
    ASSERT_TRUE(capture) << (capture ? "" : capture.error().detail);
    Stage(run);
    ASSERT_TRUE(launch_->Launch(*capture));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(again.data(), out->data, ggml_nbytes(out), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(again.data(), run.output.data(), ggml_nbytes(out)), 0);
    const auto layer = run.planned->graph.options.first_layer;
    std::size_t offset = 0;
    for (const auto& seg : run.chunk.segments) {
      for (const auto value : {false, true}) {
        const auto& t = state_.tensors[std::size_t{layer} * 2 + (value ? 1U : 0U)];
        std::vector<ggml_fp16_t> actual(static_cast<std::size_t>(t.bytes / 2));
        ASSERT_EQ(
            cudaMemcpy(actual.data(),
                       reinterpret_cast<const void*>(model_.slots[seg.slot].address + t.offset),
                       t.bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        const auto& original = value ? run.values : run.keys;
        EXPECT_EQ(
            std::memcmp(actual.data(), original.data() + offset, static_cast<std::size_t>(t.bytes)),
            0);
      }
      offset += static_cast<std::size_t>(state_.tensors[std::size_t{layer} * 2].bytes / 2);
    }
  }
  double Time(const Run& run, bool captured = false, std::uint64_t* peak = nullptr) {
    launch_->ResetScratchPeak();
    std::optional<kg::CapturedGraph> graph;
    if (captured) {
      auto recorded =
          launch_->Capture([&](auto& context) { return run.planned->bound->Run(context); });
      EXPECT_TRUE(recorded);
      if (!recorded) return 0;
      graph.emplace(std::move(*recorded));
    }
    const auto submit = [&] {
      Stage(run);
      EXPECT_TRUE(graph ? launch_->Launch(*graph) : run.planned->bound->Run(*launch_));
    };
    for (int i = 0; i < 4; ++i) {
      submit();
    }
    cudaEvent_t start{}, end{};
    EXPECT_EQ(cudaEventCreate(&start), cudaSuccess);
    EXPECT_EQ(cudaEventCreate(&end), cudaSuccess);
    auto stream = reinterpret_cast<cudaStream_t>(execution_->Submission(stream_).value().handle);
    EXPECT_EQ(cudaEventRecord(start, stream), cudaSuccess);
    for (int i = 0; i < 32; ++i) {
      submit();
    }
    EXPECT_EQ(cudaEventRecord(end, stream), cudaSuccess);
    EXPECT_EQ(cudaEventSynchronize(end), cudaSuccess);
    float milliseconds = 0;
    EXPECT_EQ(cudaEventElapsedTime(&milliseconds, start, end), cudaSuccess);
    EXPECT_EQ(cudaEventDestroy(start), cudaSuccess);
    EXPECT_EQ(cudaEventDestroy(end), cudaSuccess);
    auto* out = run.planned->graph.logits != nullptr ? run.planned->graph.logits
                                                     : run.planned->graph.hidden;
    std::vector<float> actual(run.output.size());
    EXPECT_EQ(cudaMemcpy(actual.data(), out->data, ggml_nbytes(out), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(actual.data(), run.output.data(), ggml_nbytes(out)), 0);
    if (peak != nullptr) *peak = launch_->scratch_peak().value();
    return static_cast<double>(milliseconds) * 1000 / 32;
  }
  bool positive_values_ = true;
  std::unordered_map<std::string, std::vector<double>> decoded_;
  const md::Gemma4Profile& p_ = md::Gemma4_26BA4B();
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  std::unique_ptr<kg::CublasHandle> cublas_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::unique_ptr<llmp::execution::Registry> registry_;
  std::vector<void*> allocations_;
  std::vector<md::Gemma4Resource> resources_;
  md::Gemma4Binding binding_;
  md::Gemma4StateLayout state_;
  en::Gemma4Model model_;
};
TEST_F(Gemma4ExecTest, Full512FrequencyFactorRopeAndFusedStoreMatchScalar) {
  auto arena = kg::TensorArena::Create(64);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  const auto place = [&](ggml_tensor* t, const void* input = nullptr) {
    const auto address = Allocate(ggml_nbytes(t));
    kg::TensorArena::Bind(t, address);
    EXPECT_EQ(cudaMemset(reinterpret_cast<void*>(address), 0, ggml_nbytes(t)), cudaSuccess);
    if (input != nullptr)
      EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), input, ggml_nbytes(t),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
    return t;
  };
  std::vector<float> input(512 * 2 * 3);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(std::sin(double(i) * 0.071));
  std::array<float, 256> factors{};
  for (std::size_t i = 0; i < factors.size(); ++i)
    factors[i] = i < 64 ? 1.0f + static_cast<float>(i % 5) * 0.25f : 1e30f;
  const std::array<std::int32_t, 3> positions{1279, 1280, 262143};
  const std::array<std::int64_t, 3> indices{254, 255, 0};
  auto* x = place(ggml_new_tensor_3d(c, GGML_TYPE_F32, 512, 2, 3), input.data());
  auto* p = place(ggml_new_tensor_1d(c, GGML_TYPE_I32, 3), positions.data());
  auto* f = place(ggml_new_tensor_1d(c, GGML_TYPE_F32, 256), factors.data());
  const auto node = [&] {
    return ggml_rope_ext(c, x, p, f, 512, GGML_ROPE_TYPE_NEOX, 262144, 1000000, 1, 0, 1, 32, 1);
  };
  auto* ordinary = place(node());
  auto* extended = place(node());
  ASSERT_TRUE(kg::Rope(*launch_, ordinary));
  ASSERT_TRUE(kg::RopeExt(*launch_, extended));
  auto* fused = node();  // The fused launcher never allocates or writes this node.
  auto* rows = ggml_view_2d(c, fused, 1024, 3, fused->nb[2], 0);
  auto* ids = place(ggml_new_tensor_1d(c, GGML_TYPE_I64, 3), indices.data());
  auto* cache = place(ggml_new_tensor_2d(c, GGML_TYPE_F16, 1024, 256));
  auto* write = ggml_set_rows(c, cache, rows, ids);
  kg::BindViews(kg::GraphOrder(std::vector<ggml_tensor*>{write}));
  auto launched = kg::RopeSetRows(*launch_, fused, write);
  ASSERT_TRUE(launched) << (launched ? "" : launched.error().detail);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> actual(input.size()), ext(input.size());
  std::vector<ggml_fp16_t> stored(1024 * 256);
  EXPECT_EQ(cudaMemcpy(actual.data(), ordinary->data, actual.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(ext.data(), extended->data, ext.size() * sizeof(float), cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cudaMemcpy(stored.data(), cache->data, stored.size() * sizeof(ggml_fp16_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(std::memcmp(actual.data(), ext.data(), actual.size() * sizeof(float)), 0);
  double max = 0, short_max = 0, independently_rounded_max = 0;
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t h = 0; h < 2; ++h)
      for (std::size_t i = 0; i < 256; ++i) {
        const auto a = (r * 2 + h) * 512 + i, b = a + 256;
        const float theta_scale = std::pow(1000000.0f, -2.0f / 512);
        const float angle_f32 = static_cast<float>(positions[r]) *
                                std::pow(theta_scale, static_cast<float>(i)) / factors[i];
        const double angle =
            double(positions[r]) * std::pow(1000000.0, -double(2 * i) / 512) / factors[i];
        const double rounded_a =
            input[a] * std::cos(double(angle_f32)) - input[b] * std::sin(double(angle_f32));
        const double rounded_b =
            input[a] * std::sin(double(angle_f32)) + input[b] * std::cos(double(angle_f32));
        independently_rounded_max =
            std::max({independently_rounded_max, std::abs(actual[a] - rounded_a),
                      std::abs(actual[b] - rounded_b)});
        const auto expected_a = input[a] * std::cos(angle) - input[b] * std::sin(angle);
        const auto expected_b = input[a] * std::sin(angle) + input[b] * std::cos(angle);
        const auto pair_error =
            std::max(std::abs(actual[a] - expected_a), std::abs(actual[b] - expected_b));
        max = std::max(max, pair_error);
        if (r < 2) short_max = std::max(short_max, pair_error);
        EXPECT_NEAR(double(actual[a]) * actual[a] + double(actual[b]) * actual[b],
                    double(input[a]) * input[a] + double(input[b]) * input[b], 2e-6);
        const auto offset = static_cast<std::size_t>(indices[r]) * 1024 + h * 512 + i;
        EXPECT_EQ(stored[offset], ggml_fp32_to_fp16(actual[a]));
        EXPECT_EQ(stored[offset + 256], ggml_fp32_to_fp16(actual[b]));
      }
  std::cout << "Gemma full512 factor RoPE max absolute error " << max << '\n';
  std::cout << "short-position ideal FP64 error " << short_max
            << " independently rounded F32-phase error " << independently_rounded_max << '\n';
  EXPECT_LT(short_max, 0.001);
  // Long-position ideal/F32-host phase differences are diagnostics, not a
  // newly accepted model-quality bound. CUDA ordinary/ext/fused bytes agree.
}
TEST_F(Gemma4ExecTest, CompleteLocalGlobalLayersMatchScalarAndOwnRepeat) {
  for (const auto layer : {0U, 5U})
    for (const auto shared : {false, true}) {
      const std::array<std::uint32_t, 1> slot{0};
      auto run = Execute(layer, slot, 2, 1279, shared);
      ASSERT_TRUE(run.planned != nullptr && run.planned->bound.has_value());
      Reference(run, layer);
      Repeat(run);
    }
}
TEST_F(Gemma4ExecTest, NearCancellationPrimitiveChainRetainsDiagnosticAndExactReplay) {
  positive_values_ = false;  // Original zero-centered cache fixture, retained.
  for (const auto layer : {0U, 5U}) {
    const std::array<std::uint32_t, 1> slot{0};
    auto run = Execute(layer, slot, 2, 1279, false);
    ASSERT_TRUE(run.planned != nullptr && run.planned->bound.has_value());
    Reference(run, layer, false, false);
    Reference(run, layer, false, true);
    Reference(run, layer, false, true, true);
    Reference(run, layer, false, true, true, true);
    Reference(run, layer, false, true, true, true, true);
    Repeat(run);
    auto fused = Execute(layer, slot, 2, 1279, false, false, true);
    ASSERT_TRUE(fused.planned != nullptr && fused.planned->bound.has_value());
    Reference(fused, layer, false, true, true, true, true);
    double difference = 0;
    for (std::size_t i = 0; i < run.output.size(); ++i)
      difference = std::max(difference, std::abs(double(fused.output[i]) - run.output[i]));
    std::cout << "Near-cancellation norm-fusion layer=" << layer << " max_difference=" << difference
              << '\n';
    Repeat(fused);
  }
}
TEST_F(Gemma4ExecTest, IndependentOneRowSegmentsMatchSoloAndKeepState) {
  for (const auto layer : {0U, 5U})
    for (const auto shared : {false, true})
      for (const auto count : {2U, 4U}) {
        std::array<std::uint32_t, 4> slots{0, 1, 2, 3};
        auto batch = Execute(layer, std::span(slots).first(count), 1, 1279, shared);
        Reference(batch, layer);
        ASSERT_NE(batch.planned, nullptr);
        Repeat(batch);
        const auto cells = p_.local(layer) ? state_.local_cells : state_.global_cells;
        const auto state_count = std::size_t{cells} * p_.head_dim(layer) * p_.kv_heads(layer);
        for (std::uint32_t i = 0; i < count; ++i) {
          const std::array<std::uint32_t, 1> one{i};
          auto solo = Execute(layer, one, 1, 1279, shared);
          ASSERT_NE(solo.planned, nullptr);
          ASSERT_EQ(solo.output.size(), p_.width);
          if (shared)
            EXPECT_EQ(
                std::memcmp(solo.output.data(), batch.output.data() + std::size_t{i} * p_.width,
                            p_.width * sizeof(float)),
                0);
          else
            for (std::uint32_t j = 0; j < p_.width; ++j)
              EXPECT_NEAR(solo.output[j], batch.output[std::size_t{i} * p_.width + j], 0.002);
          EXPECT_EQ(std::memcmp(solo.keys.data(), batch.keys.data() + std::size_t{i} * state_count,
                                state_count * 2),
                    0);
          EXPECT_EQ(
              std::memcmp(solo.values.data(), batch.values.data() + std::size_t{i} * state_count,
                          state_count * 2),
              0);
        }
      }
}
TEST_F(Gemma4ExecTest, NormPolicyBindsAcrossIndependentSoloTwoAndFourSegments) {
  const std::array<std::uint32_t, 4> slots{0, 1, 2, 3};
  for (const auto layer : {0U, 5U})
    for (const auto count : {1U, 2U, 4U}) {
      const auto selected = std::span(slots).first(count);
      auto primitive = Execute(layer, selected, 1, 1279, false);
      auto fused = Execute(layer, selected, 1, 1279, false, false, true);
      ASSERT_TRUE(primitive.planned != nullptr && primitive.planned->bound.has_value());
      ASSERT_TRUE(fused.planned != nullptr && fused.planned->bound.has_value());
      Reference(primitive, layer);
      Reference(fused, layer);
      ASSERT_EQ(primitive.output.size(), fused.output.size());
      for (std::size_t i = 0; i < primitive.output.size(); ++i)
        EXPECT_NEAR(primitive.output[i], fused.output[i], 0.002);
      Repeat(fused);
    }
}
TEST_F(Gemma4ExecTest, TiedQuantHeadSoftcapMatchesIndependentScalar) {
  const std::array<std::uint32_t, 1> slot{0};
  auto run = Execute(5, slot, 1, 1279, false, true);
  ASSERT_NE(run.planned, nullptr);
  const std::vector<double> input(run.input.begin(), run.input.end());
  const auto layer = scalar::Layer(p_, 5, 1279, input, 1, state_.global_cells);
  const auto normalized = scalar::Norm(layer.output, "output_norm.weight");
  const auto logits = scalar::Product(normalized, "token_embd.weight", p_.vocab);
  ASSERT_EQ(run.output.size(), logits.size());
  double max = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const auto expected = 30 * std::tanh(logits[i] / 30);
    max = std::max(max, std::abs(run.output[i] - expected));
    EXPECT_TRUE(std::isfinite(run.output[i]));
    EXPECT_LE(std::abs(run.output[i]), 30);
  }
  std::cout << "Gemma tied quant head max absolute error " << max << '\n';
  EXPECT_LT(max, 0.03);
  Repeat(run);
}
TEST_F(Gemma4ExecTest, ActualRopeStoresMatchPrimitiveBytesWithRaggedSegmentsAndFreshCapture) {
  // Each case owns only its bounded caches; no full-model or long-context
  // inference is performed. The long case includes the last absolute position.
  for (const auto* profile : {&md::Gemma4_26BA4B(), &md::Gemma4_31B()})
    for (const auto layer : {0U, 5U})
      for (const auto slots : {1U, 2U, 4U, 16U, 0U}) {
        const auto allocation_start = allocations_.size();
        {
          const bool long_case = slots == 0;
          const auto count = long_case ? 1U : slots;
          const auto d = profile->head_dim(layer), heads = profile->kv_heads(layer);
          const auto width = d * heads;
          const auto capacity = profile->local(layer) ? 1280U : (long_case ? 262144U : 4096U);
          std::vector<std::uint32_t> starts;
          std::uint32_t rows = 0;
          for (std::uint32_t i = 0; i < count; ++i) {
            starts.push_back(rows);
            rows += long_case ? 4U : i % 4 + 1;
          }
          auto arena = kg::TensorArena::Create(512);
          ASSERT_TRUE(arena);
          auto* c = arena->context();
          const auto upload = [&](ggml_tensor* t, const auto& values) {
            kg::TensorArena::Bind(t, Allocate(ggml_nbytes(t)));
            EXPECT_EQ(ggml_nbytes(t), values.size() * sizeof(values[0]));
            EXPECT_EQ(cudaMemcpy(t->data, values.data(), ggml_nbytes(t), cudaMemcpyHostToDevice),
                      cudaSuccess);
            return t;
          };
          std::vector<float> values(std::size_t{width} * rows);
          for (std::size_t i = 0; i < values.size(); ++i)
            values[i] =
                static_cast<float>(static_cast<int>((i * 17 + i / width * 11) % 103) - 51) / 37.0f;
          auto* joined = upload(ggml_new_tensor_3d(c, GGML_TYPE_F32, d, heads, rows), values);
          std::vector<std::int32_t> positions(rows);
          auto* position_tensor = upload(ggml_new_tensor_1d(c, GGML_TYPE_I32, rows), positions);
          std::vector<float> factors(d / 2);
          for (std::size_t i = 0; i < factors.size(); ++i)
            factors[i] = i < 64 ? 1.0f + static_cast<float>(i % 13) / 16 : 1e30f;
          auto* factor_tensor = profile->local(layer)
                                    ? nullptr
                                    : upload(ggml_new_tensor_1d(c, GGML_TYPE_F32, d / 2), factors);
          constexpr std::uint16_t untouched = 0x3555;
          std::vector<std::uint16_t> initial(std::size_t{width} * capacity, untouched);
          std::vector<ggml_tensor*> primitive, fused;
          std::vector<ggml_tensor*> original_caches, fused_caches, indices;
          std::vector<std::vector<std::int64_t>> ids(count);
          for (std::uint32_t i = 0; i < count; ++i) {
            const auto n = long_case ? 4U : i % 4 + 1;
            auto* part = ggml_view_3d(c, joined, d, heads, n, joined->nb[1], joined->nb[2],
                                      std::size_t{starts[i]} * joined->nb[2]);
            auto* pos = ggml_view_1d(c, position_tensor, n, std::size_t{starts[i]} * 4);
            const auto rotate = [&] {
              return ggml_rope_ext(
                  c, part, pos, factor_tensor,
                  static_cast<int>(profile->local(layer) ? profile->local_rope_dims
                                                         : profile->global_rope_dims),
                  GGML_ROPE_TYPE_NEOX, static_cast<int>(profile->context),
                  profile->local(layer) ? profile->local_rope_base : profile->global_rope_base, 1,
                  0, 1, 32, 1);
            };
            auto* original = rotate();
            kg::TensorArena::Bind(original, Allocate(ggml_nbytes(original)));
            auto* candidate = rotate();  // Fused writer never needs this allocation.
            auto* a = upload(ggml_new_tensor_2d(c, GGML_TYPE_F16, width, capacity), initial);
            auto* b = upload(ggml_new_tensor_2d(c, GGML_TYPE_F16, width, capacity), initial);
            ids[i].resize(n);
            auto* index = upload(ggml_new_tensor_1d(c, GGML_TYPE_I64, n), ids[i]);
            primitive.push_back(original);
            primitive.push_back(ggml_set_rows(
                c, a, ggml_view_2d(c, original, width, n, original->nb[2], 0), index));
            fused.push_back(candidate);
            fused.push_back(ggml_set_rows(
                c, b, ggml_view_2d(c, candidate, width, n, candidate->nb[2], 0), index));
            original_caches.push_back(a);
            fused_caches.push_back(b);
            indices.push_back(index);
            ASSERT_TRUE(kg::CheckRope(original));
            ASSERT_TRUE(kg::CheckSetRows(primitive.back()));
            ASSERT_TRUE(kg::CheckRopeSetRows(candidate, fused.back()));
          }
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);  // Complete default-stream setup.
          const auto submission = execution_->Submission(stream_);
          ASSERT_TRUE(submission);
          const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
          const auto stage = [&](bool fresh) {
            for (std::uint32_t i = 0; i < count; ++i) {
              const auto past = long_case
                                    ? 262139U + (fresh ? 1U : 0U)
                                    : (i % 2 == 0 ? 1279U : 0U) + i * 7 + (fresh ? 1280U : 0U);
              for (std::size_t r = 0; r < ids[i].size(); ++r) {
                positions[starts[i] + r] = static_cast<std::int32_t>(past + r);
                ids[i][r] = static_cast<std::int64_t>(profile->local(layer) ? (past + r) % capacity
                                                                            : past + r);
              }
              EXPECT_EQ(cudaMemcpyAsync(indices[i]->data, ids[i].data(), ggml_nbytes(indices[i]),
                                        cudaMemcpyHostToDevice, stream),
                        cudaSuccess);
            }
            EXPECT_EQ(cudaMemcpyAsync(position_tensor->data, positions.data(),
                                      ggml_nbytes(position_tensor), cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
          };
          const auto run_fused =
              [&](kg::LaunchContext& launch) -> std::expected<void, kg::KernelFailure> {
            for (std::size_t i = 0; i < fused.size(); i += 2)
              if (auto result = kg::RopeSetRows(launch, fused[i], fused[i + 1]); !result)
                return result;
            return {};
          };
          auto capture = launch_->Capture(run_fused);
          ASSERT_TRUE(capture) << (capture ? "" : capture.error().detail);
          std::vector<std::vector<std::uint16_t>> expected(count, initial);
          for (const auto fresh : {false, true}) {
            stage(fresh);
            for (std::size_t i = 0; i < primitive.size(); i += 2) {
              ASSERT_TRUE(kg::Rope(*launch_, primitive[i]));
              ASSERT_TRUE(kg::SetRows(*launch_, primitive[i + 1]));
            }
            if (!fresh) {
              ASSERT_TRUE(run_fused(*launch_));
            } else {
              ASSERT_TRUE(launch_->Launch(*capture));
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            for (std::uint32_t i = 0; i < count; ++i) {
              std::vector<float> rotated(ggml_nbytes(primitive[i * 2]) / 4);
              ASSERT_EQ(cudaMemcpy(rotated.data(), primitive[i * 2]->data,
                                   ggml_nbytes(primitive[i * 2]), cudaMemcpyDeviceToHost),
                        cudaSuccess);
              for (std::size_t r = 0; r < ids[i].size(); ++r)
                for (std::uint32_t j = 0; j < width; ++j)
                  expected[i][static_cast<std::size_t>(ids[i][r]) * width + j] =
                      std::bit_cast<std::uint16_t>(ggml_fp32_to_fp16(rotated[r * width + j]));
              std::vector<std::uint16_t> actual(initial.size());
              for (auto* cache : {original_caches[i], fused_caches[i]}) {
                ASSERT_EQ(cudaMemcpy(actual.data(), cache->data, ggml_nbytes(cache),
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
                EXPECT_EQ(std::memcmp(actual.data(), expected[i].data(), ggml_nbytes(cache)), 0);
              }
            }
          }
        }
        // Captures/descriptors are gone and submitted work is proven complete.
        while (allocations_.size() > allocation_start) {
          ASSERT_EQ(cudaFree(allocations_.back()), cudaSuccess);
          allocations_.pop_back();
        }
      }
}

TEST_F(Gemma4ExecTest, KeptSegmentRotationsUseTheOrdinaryProducerAndRemainReadable) {
  const std::array<std::uint32_t, 4> slots{0, 1, 2, 3};
  for (const auto layer : {0U, 5U}) {
    auto run =
        Execute(layer, std::span(slots).first(2), 2, 1279, false, false, false, true, true, true);
    ASSERT_NE(run.planned, nullptr);
    auto* kept = run.planned->graph.Named("blk." + std::to_string(layer) + ".slot.0.k_rope");
    ASSERT_NE(kept, nullptr);
    std::vector<float> actual(ggml_nbytes(kept) / 4);
    ASSERT_EQ(cudaMemcpy(actual.data(), kept->data, ggml_nbytes(kept), cudaMemcpyDeviceToHost),
              cudaSuccess);
    const auto width = p_.head_dim(layer) * p_.kv_heads(layer);
    for (std::uint32_t r = 0; r < 2; ++r) {
      const auto cell = run.chunk.segments[0].local_cells[r];
      const auto global = run.chunk.segments[0].global_cells[r];
      const auto index = static_cast<std::size_t>(p_.local(layer) ? cell : global) * width;
      for (std::uint32_t j = 0; j < width; ++j)
        EXPECT_EQ(ggml_fp32_to_fp16(actual[std::size_t{r} * width + j]), run.keys[index + j]);
    }
    Repeat(run);
  }
}

TEST_F(Gemma4ExecTest, CheckedMoePoliciesCompleteLocalGlobalLayersAndCapture) {
  const std::array<std::uint32_t, 4> slots{0, 1, 2, 3};
  for (const auto layer : {0U, 5U})
    for (const auto count : {1U, 2U, 4U}) {
      auto fused = Execute(layer, std::span(slots).first(count), count == 4 ? 1 : 2, 1279, false,
                           false, false, true, false, false, true, true);
      ASSERT_NE(fused.planned, nullptr);
      std::size_t routing = 0, reduction = 0;
      for (const auto& step : fused.planned->plan.steps) {
        routing += step.implementation == "ggml.gemma.route.fused";
        reduction += step.implementation == "ggml.gemma.scaled_reduce.fused";
      }
      EXPECT_EQ(routing, 1U);
      EXPECT_EQ(reduction, 1U);
      Reference(fused, layer);
      Repeat(fused);
    }
}

TEST_F(Gemma4ExecTest, CheckedNormChainsCompleteLocalGlobalLayersAndCapture) {
  const std::array<std::uint32_t, 2> slots{0, 1};
  for (const auto layer : {0U, 5U}) {
    auto fused = Execute(layer, slots, 2, 1279, false, false, false, false, false, false, true);
    ASSERT_NE(fused.planned, nullptr);
    std::size_t rope = 0, residual = 0;
    for (const auto& step : fused.planned->plan.steps) {
      rope += step.implementation == "ggml.rms_norm_mul_rope.fused";
      residual += step.implementation == "ggml.rms_norm_mul_add.fused";
    }
    EXPECT_GT(rope, 0U);
    EXPECT_GT(residual, 0U);
    Reference(fused, layer);
    Repeat(fused);
  }
}

TEST_F(Gemma4ExecTest, ExplicitRopeStoreKeepsCompleteLayersAndCapturedCachesExact) {
  const std::array<std::uint32_t, 4> slots{0, 1, 2, 3};
  for (const auto layer : {0U, 5U})
    for (const auto count : {1U, 4U}) {
      const auto selected = std::span(slots).first(count);
      auto primitive = Execute(layer, selected, 1, 1279, false, false, false, true, false);
      auto fused = Execute(layer, selected, 1, 1279, false, false, false, true, true);
      ASSERT_NE(primitive.planned, nullptr);
      ASSERT_NE(fused.planned, nullptr);
      EXPECT_EQ(primitive.output, fused.output);
      EXPECT_EQ(primitive.keys, fused.keys);
      EXPECT_EQ(primitive.values, fused.values);
      Reference(fused, layer);
      Repeat(fused);
      if (std::getenv("LLMP_GEMMA_STORE_SCREEN") != nullptr) {  // NOLINT(concurrency-mt-unsafe)
        const auto a = Time(primitive), b = Time(fused), after = Time(primitive);
        std::uint64_t ap = 0, bp = 0;
        const auto ca = Time(primitive, true, &ap), cb = Time(fused, true, &bp),
                   cafter = Time(primitive, true);
        std::cout << "GEMMA_STORE_LAYER_SCREEN layer=" << layer << " slots=" << count
                  << " rows=1 primitive_us=" << a << " fused_us=" << b
                  << " primitive_after_us=" << after << " capture_primitive_us=" << ca
                  << " capture_fused_us=" << cb << " capture_primitive_after_us=" << cafter
                  << " primitive_activation=" << primitive.planned->placement.extent
                  << " fused_activation=" << fused.planned->placement.extent
                  << " primitive_scratch=" << primitive.planned->scratch
                  << " fused_scratch=" << fused.planned->scratch << " primitive_pool_peak=" << ap
                  << " fused_pool_peak=" << bp << '\n';
      }
    }
}

TEST_F(Gemma4ExecTest, DeviceMasksKeepTheCompleteLayerAndCapturedStateExact) {
  const std::array<std::uint32_t, 4> slots{0, 1, 2, 3};
  for (const auto layer : {0U, 5U})
    for (const auto count : {1U, 4U}) {
      const auto selected = std::span(slots).first(count);
      auto host = Execute(layer, selected, 1, 1279, false);
      auto device = Execute(layer, selected, 1, 1279, false, false, false, true);
      ASSERT_NE(host.planned, nullptr);
      ASSERT_NE(device.planned, nullptr);
      EXPECT_EQ(
          std::memcmp(host.output.data(), device.output.data(), host.output.size() * sizeof(float)),
          0);
      EXPECT_TRUE(device.sources.masks.empty());
      Reference(device, layer);
      Repeat(device);
      if (std::getenv("LLMP_GEMMA_MASK_SCREEN") != nullptr) {  // NOLINT(concurrency-mt-unsafe)
        const auto a = Time(host), b = Time(device), after = Time(host);
        std::uint64_t host_peak = 0, device_peak = 0;
        const auto ca = Time(host, true, &host_peak), cb = Time(device, true, &device_peak),
                   cafter = Time(host, true);
        std::cout << "GEMMA_MASK_LAYER_SCREEN layer=" << layer << " slots=" << count
                  << " rows=1 host_us=" << a << " device_us=" << b << " host_after_us=" << after
                  << " host_source_bytes=" << en::Gemma4SourceBytes(host.planned->graph).value()
                  << " device_source_bytes=" << en::Gemma4SourceBytes(device.planned->graph).value()
                  << " capture_host_us=" << ca << " capture_device_us=" << cb
                  << " capture_host_after_us=" << cafter
                  << " host_staged_bytes=" << host.planned->inputs_bytes
                  << " device_staged_bytes=" << device.planned->inputs_bytes
                  << " host_pool_peak=" << host_peak << " device_pool_peak=" << device_peak
                  << " host_activation_bytes=" << host.planned->placement.extent
                  << " device_activation_bytes=" << device.planned->placement.extent
                  << " host_plan_scratch=" << host.planned->scratch
                  << " device_plan_scratch=" << device.planned->scratch << '\n';
      }
    }
}

TEST_F(Gemma4ExecTest, DeviceMasksKeepThirtyThreeQueryChunksAndCapturedStateExact) {
  // Local capacity remains 1,280 at this chunk size; existing slot storage
  // is unchanged. The mask now has two physical 32-query tiles.
  state_ = std::move(*md::Gemma4State(p_, 4096, 33));
  const std::array<std::uint32_t, 1> slots{0};
  for (const auto layer : {0U, 5U}) {
    auto host = Execute(layer, slots, 33, 1279, false);
    auto device = Execute(layer, slots, 33, 1279, false, false, false, true);
    ASSERT_NE(host.planned, nullptr);
    ASSERT_NE(device.planned, nullptr);
    EXPECT_EQ(
        std::memcmp(host.output.data(), device.output.data(), host.output.size() * sizeof(float)),
        0);
    Reference(device, layer);
    Repeat(device);
    if (std::getenv("LLMP_GEMMA_MASK_SCREEN") != nullptr) {  // NOLINT(concurrency-mt-unsafe)
      const auto a = Time(host), b = Time(device), after = Time(host);
      std::uint64_t host_peak = 0, device_peak = 0;
      const auto ca = Time(host, true, &host_peak), cb = Time(device, true, &device_peak),
                 cafter = Time(host, true);
      std::cout << "GEMMA_MASK_LAYER_SCREEN layer=" << layer << " slots=1 rows=33 host_us=" << a
                << " device_us=" << b << " host_after_us=" << after
                << " host_source_bytes=" << en::Gemma4SourceBytes(host.planned->graph).value()
                << " device_source_bytes=" << en::Gemma4SourceBytes(device.planned->graph).value()
                << " capture_host_us=" << ca << " capture_device_us=" << cb
                << " capture_host_after_us=" << cafter
                << " host_staged_bytes=" << host.planned->inputs_bytes
                << " device_staged_bytes=" << device.planned->inputs_bytes
                << " host_pool_peak=" << host_peak << " device_pool_peak=" << device_peak
                << " host_activation_bytes=" << host.planned->placement.extent
                << " device_activation_bytes=" << device.planned->placement.extent
                << " host_plan_scratch=" << host.planned->scratch
                << " device_plan_scratch=" << device.planned->scratch << '\n';
    }
  }
}

TEST_F(Gemma4ExecTest, OptionalPaidWholeLayerScreen) {
  // Ordinary test gate pays correctness, not timing. This bounded experiment
  // measures EVERY primitive/preparation/attention/store in the complete plan.
  if (std::getenv("LLMP_GEMMA_LAYER_SCREEN") == nullptr) return;  // NOLINT(concurrency-mt-unsafe)
  const std::array<std::uint32_t, 4> slots{0, 1, 2, 3};
  for (const auto layer : {0U, 5U})
    for (const auto count : {1U, 4U}) {
      const auto selected = std::span(slots).first(count);
      auto ordinary = Execute(layer, selected, 1, 1279, false);
      auto shared = Execute(layer, selected, 1, 1279, true);
      ASSERT_NE(ordinary.planned, nullptr);
      ASSERT_NE(shared.planned, nullptr);
      Reference(ordinary, layer);
      Reference(shared, layer);
      auto fused_norms = Execute(layer, selected, 1, 1279, false, false, true);
      ASSERT_TRUE(fused_norms.planned != nullptr && fused_norms.planned->bound.has_value());
      Reference(fused_norms, layer);
      const auto before = Time(ordinary), candidate = Time(shared), fused = Time(fused_norms),
                 after = Time(ordinary);
      std::cout << "GEMMA_LAYER_SCREEN layer=" << layer << " slots=" << count
                << " primitive_us=" << before << " shared_q8_us=" << candidate
                << " fused_norms_us=" << fused << " primitive_after_us=" << after << '\n';
    }
}
}  // namespace
