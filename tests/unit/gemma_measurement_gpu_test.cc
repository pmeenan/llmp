// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Descriptor-only funding oracle on actual device selectors; no model work is submitted.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma2_plan.h"
#include "engine/gemma3_plan.h"
#include "engine/gemma4_plan.h"
#include "engine/support.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "gemma2_fixture.h"
#include "gemma3_fixture.h"
#include "gemma4_fixture.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/implementations.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {
namespace en = jitllm::engine;
namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;

template <int Family>
struct Adapter;

template <>
struct Adapter<2> {
  using Model = en::Gemma2Model;
  using Shape = kg::Gemma2ChunkShape;
  using Segment = md::Gemma2Segment;
  using Options = kg::Gemma2GraphOptions;
  static constexpr std::uint32_t kContext = 8192;
  static constexpr std::uint32_t kRows = 128;
  static constexpr std::uint32_t kBudget = 256;
  static constexpr std::uint32_t kSlots = 2;
  static const auto& Profile([[maybe_unused]] std::uint32_t size) { return md::Gemma2_2B(); }
  static auto Binding(const auto& p, [[maybe_unused]] std::uint32_t size) {
    return md::BindGemma2(p, "gemma2", jitllm::test_support::gemma2::Resources());
  }
  static auto State(const auto& p) { return md::Gemma2State(p, kContext, kRows); }
  static auto Chunk(const auto& p, const auto& state, const auto& segments) {
    return md::Gemma2Chunk(p, state, segments, false, 256, kBudget);
  }
  static auto Build(kg::TensorArena& arena, const auto& p, const auto& binding, const auto& state,
                    const Shape& shape, const Options& o, const kg::DeviceChoices& choices) {
    return kg::BuildGemma2Graph(arena, p, binding, state, shape, o, choices.dense_mmvq_shape);
  }
  static auto Plan(const Model& model, const Shape& shape, const kg::DeviceChoices& choices,
                   std::optional<en::ActivationMeasurement> measurement = std::nullopt) {
    return en::PlanGemma2Chunk(model, shape, choices, 0, 0, {}, measurement);
  }
  static auto Source(const auto& graph) { return en::Gemma2SourceBytes(graph); }
  static auto Host(const auto& p, const auto& state, const auto& segments) {
    return md::Gemma2HostInputBytes(p, state, segments, false, 256, kBudget);
  }
  static std::size_t Tensors(const auto& p, std::size_t owners, const Options& o) {
    (void)o;
    return kg::Gemma2GraphTensors(p, owners);
  }
  static void Output(Shape& shape, unsigned mode, std::uint32_t owners, std::uint32_t total,
                     bool features) {
    shape.output_mode = mode == 2 ? kg::Gemma2OutputMode::kStateOnly : kg::Gemma2OutputMode::kHead;
    shape.outputs = mode == 2 ? 0U : mode == 1 ? total : owners;
    shape.greedy = mode == 3;
    (void)features;
  }
  static Options Policy() {
    Options o;
    o.max_total_rows = kBudget;
    o.shared_q8 = true;
    o.owner_decode = true;
    o.device_masks = true;
    o.narrow_final = true;
    o.packed_prefill = true;
    o.owner_prefill = true;
    o.flexible_owner_prefill = true;
    return o;
  }
};

template <>
struct Adapter<3> {
  using Model = en::Gemma3Model;
  using Shape = kg::Gemma3ChunkShape;
  using Segment = md::Gemma3Segment;
  using Options = kg::Gemma3GraphOptions;
  static constexpr std::uint32_t kContext = 4096;
  static constexpr std::uint32_t kRows = 128;
  static constexpr std::uint32_t kBudget = 256;
  static constexpr std::uint32_t kSlots = 2;
  static const auto& Profile([[maybe_unused]] std::uint32_t size) { return md::Gemma3_4BQat(); }
  static auto Binding(const auto& p, [[maybe_unused]] std::uint32_t size) {
    return md::BindGemma3(p, "gemma3", jitllm::test_support::gemma3::Resources());
  }
  static auto State(const auto& p) { return md::Gemma3State(p, kContext, kRows); }
  static auto Chunk(const auto& p, const auto& state, const auto& segments) {
    return md::Gemma3Chunk(p, state, segments, false, 256, kBudget);
  }
  static auto Build(kg::TensorArena& arena, const auto& p, const auto& binding, const auto& state,
                    const Shape& shape, const Options& o, const kg::DeviceChoices& choices) {
    return kg::BuildGemma3Graph(arena, p, binding, state, shape, o, choices.dense_mmvq_shape);
  }
  static auto Plan(const Model& model, const Shape& shape, const kg::DeviceChoices& choices,
                   std::optional<en::ActivationMeasurement> measurement = std::nullopt) {
    return en::PlanGemma3Chunk(model, shape, choices, 0, 0, {}, measurement);
  }
  static auto Source(const auto& graph) { return en::Gemma3SourceBytes(graph); }
  static auto Host(const auto& p, const auto& state, const auto& segments) {
    return md::Gemma3HostInputBytes(p, state, segments, false, 256, kBudget);
  }
  static std::size_t Tensors(const auto& p, std::size_t owners, const Options& o) {
    (void)o;
    return kg::Gemma3GraphTensors(p, owners);
  }
  static void Output(Shape& shape, unsigned mode, std::uint32_t owners, std::uint32_t total,
                     bool features) {
    shape.output_mode = mode == 2 ? kg::Gemma3OutputMode::kStateOnly : kg::Gemma3OutputMode::kHead;
    shape.outputs = mode == 2 ? 0U : mode == 1 ? total : owners;
    shape.greedy = mode == 3;
    (void)features;
  }
  static Options Policy() {
    Options o;
    o.max_total_rows = kBudget;
    o.shared_q8 = true;
    o.owner_decode = true;
    o.device_masks = true;
    o.narrow_final = true;
    o.packed_prefill = true;
    o.owner_prefill = true;
    o.flexible_owner_prefill = true;
    return o;
  }
};

template <>
struct Adapter<4> {
  using Model = en::Gemma4Model;
  using Shape = kg::Gemma4ChunkShape;
  using Segment = md::Gemma4Segment;
  using Options = kg::Gemma4GraphOptions;
  static constexpr std::uint32_t kContext = 4096;
  static constexpr std::uint32_t kRows = 1024;
  static constexpr std::uint32_t kBudget = 1024;
  static constexpr std::uint32_t kSlots = 4;
  static const auto& Profile([[maybe_unused]] std::uint32_t size) {
    return size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
  }
  static auto Binding(const auto& p, [[maybe_unused]] std::uint32_t size) {
    return md::BindGemma4(p, "gemma4", jitllm::test_support::gemma4::Resources(size));
  }
  static auto State(const auto& p) { return md::Gemma4State(p, kContext, kRows); }
  static auto Chunk(const auto& p, const auto& state, const auto& segments) {
    return md::Gemma4Chunk(p, state, segments, false);
  }
  static auto Build(kg::TensorArena& arena, const auto& p, const auto& binding, const auto& state,
                    const Shape& shape, const Options& o, const kg::DeviceChoices& choices) {
    return kg::BuildGemma4Graph(arena, p, binding, state, shape, o, choices.dense_mmvq_shape);
  }
  static auto Plan(const Model& model, const Shape& shape, const kg::DeviceChoices& choices,
                   std::optional<en::ActivationMeasurement> measurement = std::nullopt) {
    return en::PlanGemma4Chunk(model, shape, choices, 0, 0, {}, measurement);
  }
  static auto Source(const auto& graph) { return en::Gemma4SourceBytes(graph); }
  static auto Host(const auto& p, const auto& state, const auto& segments) {
    return md::Gemma4HostInputBytes(p, state, segments, false);
  }
  static std::size_t Tensors(const auto& p, std::size_t owners, const Options& o) {
    return kg::Gemma4GraphTensors(p, owners, o);
  }
  static void Output(Shape& shape, unsigned mode, std::uint32_t owners, std::uint32_t total,
                     bool features) {
    shape.output_mode = mode == 2 ? kg::Gemma4OutputMode::kStateOnly : kg::Gemma4OutputMode::kHead;
    shape.outputs = mode == 2 ? 0U : mode == 1 ? total : owners;
    shape.greedy = mode == 3;
    shape.feature_outputs = features ? total : 0U;
  }
  static Options Policy() {
    Options o;
    o.dense_shared_q8 = true;
    o.device_masks = true;
    o.attention_mode = kg::Gemma4AttentionMode::kOwners;
    o.common_owner_reads = true;
    o.bounded_owner_roots = true;
    return o;
  }
};

void SameDescriptions(const kg::GraphPlan& a, const kg::GraphPlan& b) {
  ASSERT_EQ(a.steps.size(), b.steps.size());
  for (std::size_t i = 0; i < a.steps.size(); ++i) {
    const auto& x = a.steps[i];
    const auto& y = b.steps[i];
    EXPECT_EQ(x.operation, y.operation);
    EXPECT_EQ(x.implementation, y.implementation);
    EXPECT_EQ(x.lane, y.lane);
    ASSERT_EQ(x.nodes.size(), y.nodes.size());
    for (std::size_t j = 0; j < x.nodes.size(); ++j) {
      EXPECT_STREQ(x.nodes[j]->name, y.nodes[j]->name);
      EXPECT_EQ(x.nodes[j]->op, y.nodes[j]->op);
      EXPECT_EQ(x.nodes[j]->type, y.nodes[j]->type);
      EXPECT_TRUE(std::ranges::equal(x.nodes[j]->ne, y.nodes[j]->ne));
    }
  }
}

template <int Family>
void Oracle(kg::LaunchContext& launch, std::uint32_t size) {
  using A = Adapter<Family>;
  const auto& profile = A::Profile(size);
  auto binding = A::Binding(profile, size);
  ASSERT_TRUE(binding);
  auto state = A::State(profile);
  ASSERT_TRUE(state);
  auto choices = kg::DeviceChoicesOf(launch);
  choices.fuse_norms = true;
  choices.fuse_norm_rope = true;
  choices.fuse_norm_add = true;
  choices.fuse_gemma_route = Family == 4 && size == 26;
  choices.fuse_gemma_reduce = Family == 4 && size == 26;
  choices.fuse_quant_glu = Family != 4 || size == 31;
  const auto options = A::Policy();
  std::vector<std::int32_t> tokens(A::kRows, 1);
  typename A::Shape seed;
  seed.segments.push_back({0, 1, 0, 256, 256});
  seed.outputs = 1;
  auto arena = kg::TensorArena::Create(A::Tensors(profile, 1, options));
  ASSERT_TRUE(arena);
  auto graph = A::Build(*arena, profile, *binding, *state, seed, options, choices);
  ASSERT_TRUE(graph);
  typename A::Model model;
  model.profile = &profile;
  model.binding = &*binding;
  model.state = &*state;
  model.options = options;
  std::uint64_t address = std::uint64_t{1} << 36U;
  for (const auto& w : graph->weights) {
    auto* domain = &model.resources;
    std::uint64_t bytes = w.resource.readable;
    if constexpr (Family == 4) {
      if (w.resource.expert_array) {
        domain = &model.arrays;
        bytes += (std::uint64_t{profile.experts} - 1) * w.tensor->nb[2];
      }
    }
    domain->resize(std::max(domain->size(), std::size_t{w.resource.index} + 1));
    (*domain)[w.resource.index] = {address, bytes};
    address += en::support::Round(bytes, 256) + 256;
  }
  model.slots.resize(A::kSlots);
  for (auto& slot : model.slots) {
    slot = {address, state->bytes};
    address += state->bytes + 256;
  }
  std::uint64_t exact_max = 0, shortcut_max = 0, measured = 0, shortcuts = 0, fallbacks = 0;
  auto registry = jitllm::execution::Registry::Create(kg::Implementations());
  ASSERT_TRUE(registry);
  for (const auto budget : {A::kBudget, 2U}) {
    for (std::uint32_t owners = 1; owners <= A::kSlots; ++owners) {
      const auto endpoints = en::support::ChunkMeasurementRows(A::kRows, budget, owners, false);
      const auto cases = en::support::ChunkMeasurementRows(A::kRows, budget, owners, true);
      for (const auto& rows : cases) {
        const bool supplemental = std::ranges::find(endpoints, rows) == endpoints.end();
        std::uint32_t total = 0;
        for (const auto n : rows) total += n;
        for (unsigned endpoint = 0; endpoint < 4; ++endpoint) {
          if (endpoint >= 2 && (owners != 2 || (Family != 4 && endpoint == 3))) continue;
          if constexpr (Family == 4)
            if (endpoint >= 2 && rows != std::vector<std::uint32_t>{1, 1}) continue;
          std::vector<typename A::Segment> segments;
          for (std::uint32_t i = 0; i < owners; ++i) {
            const bool high =
                endpoint == 1 || (endpoint == 2 && i == 1) || (endpoint == 3 && i == 0);
            segments.push_back(
                {i, high ? A::kContext - rows[i] : 0U, std::span(tokens).first(rows[i])});
          }
          auto input = A::Chunk(profile, *state, segments);
          auto host = A::Host(profile, *state, segments);
          ASSERT_TRUE(input) << *jitllm::test_support::Failed(input);
          ASSERT_TRUE(host) << *jitllm::test_support::Failed(host);
          for (const bool features : {false, true}) {
            if (features && Family != 4) continue;
            for (unsigned mode = 0; mode < 4; ++mode) {
              if (mode == 1 ? budget != 2 : budget != A::kBudget) continue;
              if (features && mode >= 2) continue;
              typename A::Shape shape;
              for (const auto& s : input->segments)
                shape.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
              A::Output(shape, mode, owners, total, features);
              auto exact = A::Plan(model, shape, choices);
              ASSERT_TRUE(exact) << *jitllm::test_support::Failed(exact);
              if (!supplemental) {
                exact_max = std::max(exact_max, (*exact)->placement.extent);
                shortcut_max = std::max(shortcut_max, (*exact)->placement.extent);
                continue;
              }
              auto bounded =
                  A::Plan(model, shape, choices, en::ActivationMeasurement{shortcut_max});
              ASSERT_TRUE(bounded) << *jitllm::test_support::Failed(bounded);
              ++measured;
              ASSERT_TRUE((*bounded)->measurement_only);
              EXPECT_FALSE(en::BindPlanned(**bounded, launch, *registry, "measurement oracle"));
              SameDescriptions((*exact)->plan, (*bounded)->plan);
              EXPECT_EQ(en::PlannedHostBytes(**exact), en::PlannedHostBytes(**bounded));
              EXPECT_EQ((*exact)->inputs_bytes, (*bounded)->inputs_bytes);
              const auto exact_source = A::Source((*exact)->graph);
              const auto bound_source = A::Source((*bounded)->graph);
              ASSERT_TRUE(exact_source);
              ASSERT_TRUE(bound_source);
              EXPECT_EQ(*exact_source + *host, *bound_source + *host);
              const auto exact_scratch = kg::PlanScratch(launch, (*exact)->plan);
              const auto bound_scratch = kg::PlanScratch(launch, (*bounded)->plan);
              ASSERT_TRUE(exact_scratch);
              ASSERT_TRUE(bound_scratch);
              EXPECT_EQ(*exact_scratch, *bound_scratch);
              if ((*bounded)->placement.measurement_bound) {
                ++shortcuts;
                EXPECT_LE((*exact)->placement.extent, (*bounded)->placement.extent);
                EXPECT_LE((*bounded)->placement.extent, shortcut_max);
              } else {
                ++fallbacks;
                EXPECT_EQ((*exact)->placement.extent, (*bounded)->placement.extent);
              }
              exact_max = std::max(exact_max, (*exact)->placement.extent);
              shortcut_max = std::max(shortcut_max, (*bounded)->placement.extent);
              EXPECT_EQ(exact_max, shortcut_max);
            }
          }
        }
      }
    }
  }
  EXPECT_GT(measured, 0U);
  EXPECT_GT(shortcuts, 0U);
  EXPECT_EQ(exact_max, shortcut_max);
  std::cout << "MEASUREMENT_ORACLE family=" << Family << " size=" << size
            << " supplemental=" << measured << " shortcuts=" << shortcuts
            << " fallbacks=" << fallbacks << " activation_max=" << exact_max << '\n';
}

TEST(GemmaMeasurementGpu, AllConfiguredCompositionsPreserveExactFunding) {
  auto execution = jitllm::providers::cuda::OpenDeviceExecution(0);
  ASSERT_TRUE(execution);
  auto stream = (*execution)->CreateStream();
  ASSERT_TRUE(stream);
  auto launch = kg::LaunchContext::Create(0, **execution, *stream,
                                          {.base = 0, .size = jitllm::base::Bytes(0)});
  ASSERT_TRUE(launch);
  Oracle<2>(**launch, 2);
  Oracle<3>(**launch, 3);
  Oracle<4>(**launch, 26);
  Oracle<4>(**launch, 31);
  const auto retire = [&]() {
    const auto fence = (*execution)->Record(*stream);
    if (!fence) {
      ADD_FAILURE() << "completion record: " << fence.error().detail;
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      const auto state = (*execution)->Query(*fence);
      if (!state) {
        ADD_FAILURE() << "completion query: " << state.error().detail;
        return false;
      }
      if (*state == jitllm::providers::FenceState::kComplete) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        ADD_FAILURE() << "descriptor launch context retirement timed out";
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    const auto released = (*execution)->Release(*fence);
    if (!released) {
      ADD_FAILURE() << "completion release: " << released.error().detail;
      return false;
    }
    return true;
  };
  if (!retire()) {
    // A destructor cannot establish completion. Retain both owners until exit.
    std::ignore = launch->release();
    std::ignore = execution->release();
    return;
  }
  launch->reset();
  const auto destroyed = (*execution)->DestroyStream(*stream);
  EXPECT_TRUE(destroyed) << jitllm::test_support::Failed(destroyed)->detail;
}
}  // namespace
