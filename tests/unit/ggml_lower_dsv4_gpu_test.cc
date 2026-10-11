// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The shared local forms against the existing helpers on identical operands.
// No model weights or serving-policy/performance claim.
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
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "components/step.h"
#include "execution/registry.h"
#include "ir/builder.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "planner/pattern.h"
#include "planner/plan.h"
#include "providers/cuda/cuda_device_execution.h"
#include "tests/support/dsv4_score_original.h"

namespace {
namespace co = llmp::components;
namespace ex = llmp::execution;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
using llmp::format::Encoding;

std::vector<float> Values(std::size_t count, float scale = 1.0F) {
  std::vector<float> out(count);
  for (std::size_t i = 0; i < count; ++i)
    out[i] = scale * static_cast<float>(static_cast<int>((i * 31) % 97) - 48) / 64.0F;
  return out;
}

class GgmlLowerDsv4Gpu : public ::testing::Test {
 protected:
  void SetUp() override {
    auto opened = llmp::providers::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(opened);
    execution = std::move(*opened);
    auto created = execution->CreateStream();
    ASSERT_TRUE(created);
    stream = *created;
    stream_ready = true;
    auto context = kg::LaunchContext::Create(
        0, *execution, stream, {.base = Allocate(16U << 20), .size = llmp::base::Bytes(16U << 20)});
    ASSERT_TRUE(context);
    launch = std::move(*context);
    auto declared = ex::Registry::Create(kg::Implementations());
    ASSERT_TRUE(declared);
    registry.emplace(std::move(*declared));
  }
  bool Finish() {
    if (retirement_failed) return false;
    if (!execution || !stream_ready) return true;
    auto fence = execution->Record(stream);
    if (!fence) {
      retirement_failed = true;
      ADD_FAILURE() << "record completion";
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      auto state = execution->Query(*fence);
      if (!state || std::chrono::steady_clock::now() >= deadline) {
        retirement_failed = true;
        ADD_FAILURE() << "unproven retirement; retain fixture owners";
        return false;
      }
      if (*state == llmp::providers::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution->Release(*fence)) {
      retirement_failed = true;
      return false;
    }
    return true;
  }
  void TearDown() override {
    if (!Finish()) {
      std::ignore = new decltype(graphs)(std::move(graphs));
      std::ignore = new decltype(lowered)(std::move(lowered));
      // Pending copies borrow these bytes just as launches borrow device roots.
      std::ignore = new decltype(staged_values)(std::move(staged_values));
      std::ignore = new decltype(arenas)(std::move(arenas));
      std::ignore = launch.release();
      std::ignore = cublas.release();
      std::ignore = execution.release();
      return;
    }
    graphs.clear();
    lowered.clear();
    arenas.clear();
    launch.reset();
    cublas.reset();
    if (execution && stream_ready) EXPECT_TRUE(execution->DestroyStream(stream));
    for (void* p : allocations) EXPECT_EQ(cudaFree(p), cudaSuccess);
  }
  std::uint64_t Allocate(std::uint64_t size) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, std::max<std::uint64_t>(size, 256)), cudaSuccess);
    allocations.push_back(p);
    return reinterpret_cast<std::uintptr_t>(p);
  }
  ggml_context* Context(std::size_t tensors = 32) {
    auto arena = kg::TensorArena::Create(tensors);
    EXPECT_TRUE(arena);
    if (!arena) return nullptr;
    arenas.push_back(std::make_unique<kg::TensorArena>(std::move(*arena)));
    return arenas.back()->context();
  }
  ggml_tensor* Place(ggml_tensor* t) {
    kg::TensorArena::Bind(t, Allocate(ggml_nbytes(t)));
    return t;
  }
  // Pageable H2D cudaMemcpy may return before DMA completes on the default
  // stream. All fixture writes use the launch stream and explicit retirement;
  // staged_values owns their host bytes even on an unknown-completion path.
  void CopyToInput(ggml_tensor* t, const std::vector<std::byte>& bytes) {
    auto native = execution->Submission(stream);
    ASSERT_TRUE(native) << (native ? "" : native.error().detail);
    const auto copied = cudaMemcpyAsync(t->data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice,
                                        static_cast<cudaStream_t>(native->handle));
    if (copied != cudaSuccess) retirement_failed = true;
    ASSERT_EQ(copied, cudaSuccess);
  }
  template <class T>
  void Upload(ggml_tensor* t, const std::vector<T>& values) {
    ASSERT_EQ(ggml_nbytes(t), values.size() * sizeof(T));
    ASSERT_TRUE(Finish());  // retire every previous borrower before changing its bytes
    auto& saved = staged_values[t];
    saved.resize(ggml_nbytes(t));
    std::memcpy(saved.data(), values.data(), saved.size());
    CopyToInput(t, saved);
    ASSERT_TRUE(Finish());
  }
  void Restage(const kg::LoweredStep& step) {
    for (const auto& leaf : step.leaves) {
      const auto at = staged_values.find(leaf.tensor);
      if (at == staged_values.end()) continue;
      CopyToInput(leaf.tensor, at->second);
      if (::testing::Test::HasFatalFailure()) return;
    }
    ASSERT_TRUE(Finish());
  }
  // Logical F32 values, including the legacy HC tails' per-token pitch.
  std::vector<float> Read(const ggml_tensor* t) {
    if (!Finish()) return {};
    EXPECT_EQ(t->type, GGML_TYPE_F32);
    std::vector<std::byte> raw(ggml_nbytes(t));
    const auto copied = cudaMemcpy(raw.data(), t->data, raw.size(), cudaMemcpyDeviceToHost);
    if (copied != cudaSuccess) retirement_failed = true;
    EXPECT_EQ(copied, cudaSuccess);
    std::vector<float> out;
    for (std::int64_t d = 0; d < t->ne[3]; ++d)
      for (std::int64_t c = 0; c < t->ne[2]; ++c)
        for (std::int64_t b = 0; b < t->ne[1]; ++b)
          for (std::int64_t a = 0; a < t->ne[0]; ++a) {
            const auto offset =
                static_cast<std::size_t>(a) * t->nb[0] + static_cast<std::size_t>(b) * t->nb[1] +
                static_cast<std::size_t>(c) * t->nb[2] + static_cast<std::size_t>(d) * t->nb[3];
            float value = 0;
            std::memcpy(&value, raw.data() + offset, sizeof(value));
            out.push_back(value);
          }
    return out;
  }
  template <class T>
  std::vector<T> ReadTyped(const ggml_tensor* t) {
    if (!Finish()) return {};
    std::vector<T> result(ggml_nbytes(t) / sizeof(T));
    const auto copied = cudaMemcpy(result.data(), t->data, ggml_nbytes(t), cudaMemcpyDeviceToHost);
    if (copied != cudaSuccess) retirement_failed = true;
    EXPECT_EQ(copied, cudaSuccess);
    return result;
  }
  kg::LoweredStep* Lower(ir::Graph graph, std::vector<co::StepInput> inputs,
                         const pl::StepShape& shape, const ex::Registry* selected = nullptr,
                         bool aligned_roots = false) {
    co::StepGraph step{
        .graph = std::move(graph), .inputs = std::move(inputs), .weights = {}, .state = {}};
    std::vector<ir::TensorId> staged;
    for (const auto& input : step.inputs) staged.push_back(input.tensor);
    const auto& choices = selected == nullptr ? *registry : *selected;
    // cudaMalloc's owned allocation is independently checked at binding;
    // the planner derives each independently placed root from this guarantee.
    auto plan = pl::Plan(step.graph.Clone(), choices, {.device_class = "sm_121", .arch = 121}, {},
                         {.staged = staged,
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot,
                          .shape = &shape,
                          .activation_alignment = aligned_roots ? 256U : 0U});
    EXPECT_TRUE(plan) << (plan ? "" : plan.error().detail);
    if (!plan) return nullptr;
    const kn::Addresses addresses{.weight = {},
                                  .weight_rows = {},
                                  .state = {},
                                  .save = {},
                                  .activations = Allocate(plan->placement.extent),
                                  .activation_bytes = plan->placement.extent};
    auto result = kg::LowerStep(
        {.plan = *plan, .step = step, .shape = shape, .addresses = addresses, .context = 65536},
        choices, kg::DeviceChoicesOf(*launch));
    EXPECT_TRUE(result) << (result ? "" : result.error().detail);
    if (!result) return nullptr;
    lowered.push_back(std::make_unique<kg::LoweredStep>(std::move(*result)));
    return lowered.back().get();
  }
  kg::LoweredStep* LowerStateBanks(const co::StepGraph& step, const pl::StepShape& shape,
                                   kn::Addresses addresses, const pl::StateSpans& facts,
                                   bool enabled) {
    pl::Policy policy;
    for (const auto& pattern : pl::Patterns())
      if (pattern.name != "state_bank_attention") policy.disabled.emplace_back(pattern.name);
    if (enabled) policy.enabled = {"state_bank_attention"};
    pl::Canonicalize(policy);
    std::vector<ir::TensorId> staged;
    for (const auto& input : step.inputs) staged.push_back(input.tensor);
    auto plan =
        pl::Plan(step.graph.Clone(), *registry, {.device_class = "sm_121", .arch = 121}, policy,
                 {.staged = staged,
                  .written = kg::WrittenBytes,
                  .within = kg::WithinRoot,
                  .shape = &shape,
                  .state_spans = facts});
    EXPECT_TRUE(plan) << (plan ? "" : plan.error().detail);
    if (!plan) return nullptr;
    EXPECT_EQ(std::ranges::count_if(
                  plan->applied,
                  [](const auto& applied) { return applied.pattern == "state_bank_attention"; }),
              enabled ? shape.segments.size() : 0U);
    addresses.activations = Allocate(plan->placement.extent);
    addresses.activation_bytes = plan->placement.extent;
    auto result = kg::LowerStep({.plan = *plan,
                                 .step = step,
                                 .shape = shape,
                                 .addresses = addresses,
                                 .context = 512,
                                 .state_spans = facts},
                                *registry, kg::DeviceChoicesOf(*launch));
    EXPECT_TRUE(result) << (result ? "" : result.error().detail);
    if (!result) return nullptr;
    lowered.push_back(std::make_unique<kg::LoweredStep>(std::move(*result)));
    return lowered.back().get();
  }
  ggml_tensor* Leaf(kg::LoweredStep& step, ir::TensorId id) {
    const auto it =
        std::ranges::find_if(step.leaves, [&](const auto& leaf) { return leaf.primitive == id; });
    EXPECT_NE(it, step.leaves.end());
    return it == step.leaves.end() ? nullptr : it->tensor;
  }
  void RunAndReplay(kg::LoweredStep& step) {
    auto bound = kg::BoundGraph::Bind(*registry, step.plan);
    ASSERT_TRUE(bound) << (bound ? "" : bound.error().detail);
    Restage(step);
    const auto ran = bound->Run(*launch);
    ASSERT_TRUE(ran) << (ran ? "" : ran.error().detail);
    ASSERT_TRUE(Finish());
    std::vector<std::vector<float>> before;
    for (const auto* t : step.keep) before.push_back(Read(t));
    auto previous = captures.find(&step);
    if (previous == captures.end()) {
      auto captured = launch->Capture([&](kg::LaunchContext& l) { return bound->Run(l); });
      ASSERT_TRUE(captured) << (captured ? "" : captured.error().detail);
      graphs.push_back(std::move(*captured));
      previous = captures.emplace(&step, graphs.size() - 1).first;
    }
    // As in ModelRunner, staged inputs are repopulated before every
    // replay; their placement may be reused after their final reader.
    // Reuse this same captured graph after positions/scores/values change.
    Restage(step);
    ASSERT_TRUE(launch->Launch(graphs[previous->second]));
    ASSERT_TRUE(Finish());
    for (std::size_t i = 0; i < step.keep.size(); ++i) {
      const auto after = Read(step.keep[i]);
      ASSERT_EQ(after.size(), before[i].size());
      for (std::size_t j = 0; j < after.size(); ++j)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(after[j]),
                  std::bit_cast<std::uint32_t>(before[i][j]));
    }
  }
  void ScoreOracle(std::int64_t cells, std::int64_t rows, bool pitched, bool repeated = false,
                   std::int64_t top = 1);
  std::unique_ptr<llmp::providers::DeviceExecution> execution;
  llmp::providers::StreamId stream{};
  bool stream_ready = false, retirement_failed = false;
  std::unique_ptr<kg::CublasHandle> cublas;
  std::unique_ptr<kg::LaunchContext> launch;
  std::optional<ex::Registry> registry;
  std::vector<void*> allocations;
  std::map<ggml_tensor*, std::vector<std::byte>> staged_values;
  std::vector<kg::CapturedGraph> graphs;
  std::map<const kg::LoweredStep*, std::size_t> captures;
  std::vector<std::unique_ptr<kg::TensorArena>> arenas;
  std::vector<std::unique_ptr<kg::LoweredStep>> lowered;
};

TEST_F(GgmlLowerDsv4Gpu, StateBanksMatchPrimitiveCopiesAndReplayChangedJoinedOperands) {
  namespace md = llmp::model;
  // Keep dense D256/D512 controls and exercise the existing D512 sparse
  // compacted path at its ordinary 4096-cell floor (sparse_any stays off).
  for (const auto [head, compressed_read] :
       {std::pair<std::uint64_t, std::uint64_t>{256, 256}, {512, 256}, {512, 3584}}) {
    const std::uint64_t compressed_capacity = compressed_read + 256;
    const std::uint64_t cells = 512 + compressed_read;
    const bool sparse = cells >= 4096;
    for (const bool joined : {false, true}) {
      SCOPED_TRACE(head);
      SCOPED_TRACE(cells);
      SCOPED_TRACE(joined);
      const std::vector<std::pair<std::uint32_t, md::StateContract>> contracts{
          {0,
           {.tensors = {{.name = "prefix", .cell_bytes = 4},
                        {.name = "raw", .cell_bytes = head * 2},
                        {.name = "compressed",
                         .cell_bytes = head * 2,
                         .cells = md::CellRule::kFixed,
                         .window_or_ratio = static_cast<std::uint32_t>(compressed_capacity)}}}}};
      const auto layout = md::Layout(contracts, 512, 128);
      ASSERT_TRUE(layout);
      ASSERT_EQ(layout->items[1].offset, 2048U);  // inside an extent, after unrelated state
      pl::StepShape shape{.segments = {{.rows = 1, .outputs = 0, .want_head = false}},
                          .read_align = 2048};
      if (joined) {
        shape.segments[0].rows = 2;
        shape.segments.push_back({.slot = 4, .rows = 3, .outputs = 0, .want_head = false});
      }
      std::vector<std::uint64_t> state_bases;
      for (std::size_t owner = 0; owner < shape.segments.size(); ++owner)
        state_bases.push_back(Allocate(layout->bytes));
      const pl::StateSpans facts =
          [&](std::span<const pl::StateCellSlice> slices) -> std::optional<pl::StateSpanFact> {
        if (slices.empty()) return std::nullopt;
        const auto owner = slices.front().ref.segment;
        if (owner >= shape.segments.size()) return std::nullopt;
        std::vector<md::StateSlice> requested;
        for (const auto& slice : slices) {
          if (slice.ref.stage != 0 || slice.ref.segment != owner) return std::nullopt;
          requested.push_back({.layer = slice.ref.layer,
                               .tensor = slice.ref.tensor,
                               .first = slice.first,
                               .cells = slice.cells});
        }
        const auto bank = md::ContiguousStateSpan(*layout, requested);
        return bank ? std::optional<pl::StateSpanFact>{{.bank = *bank,
                                                        .region_bytes = layout->bytes}}
                    : std::nullopt;
      };
      kn::Addresses addresses{};
      addresses.state = [&](const ir::StateRef& ref) {
        return state_bases.at(ref.segment) + layout->items.at(ref.tensor).offset;
      };
      ir::Builder b;
      const auto queries = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({head, 8, shape.rows()})));
      const auto sinks = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8})));
      std::vector<co::StepInput> inputs{{.tensor = queries, .kind = co::InputKind::kFeatures},
                                        {.tensor = sinks, .kind = co::InputKind::kFeatures}};
      std::vector<co::StepState> state;
      std::vector<ir::TensorId> masks;
      std::uint64_t first = 0;
      for (std::size_t owner = 0; owner < shape.segments.size(); ++owner) {
        const auto segment = static_cast<std::uint8_t>(owner);
        b.SetOwner(segment);
        const auto rows = shape.segments[owner].rows;
        const auto q =
            b.View(queries, ir::ShapeOf({head, 8, rows}),
                   {4, head * 4, head * 8 * 4, head * 8 * shape.rows() * 4}, first * head * 8 * 4);
        const ir::StateRef raw_ref{.tensor = 1, .segment = segment};
        const ir::StateRef comp_ref{.tensor = 2, .segment = segment};
        const auto raw = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({head, 512})), raw_ref);
        const auto compressed =
            b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({head, compressed_capacity})), comp_ref);
        state.push_back({.tensor = raw, .ref = raw_ref, .item = 1, .read = true});
        state.push_back({.tensor = compressed, .ref = comp_ref, .item = 2, .read = true});
        const auto a =
            b.View(raw, ir::ShapeOf({head, 1, 512}), {2, head * 2, head * 2, head * 1024}, 0);
        const auto z = b.View(compressed, ir::ShapeOf({head, 1, compressed_read}),
                              {2, head * 2, head * 2, head * compressed_capacity * 2}, 0);
        const auto copied = b.Concat(a, z, 2);
        const auto mask = b.Input(ir::Dense(Encoding::kF16, ir::ShapeOf({cells, 32})));
        masks.push_back(mask);
        inputs.push_back({.tensor = mask, .kind = co::InputKind::kFeatures, .segment = segment});
        b.Output(b.Attention(b.Permute(q, {0, 2, 1, 3}), b.Permute(copied, {0, 2, 1, 3}),
                             b.Permute(copied, {0, 2, 1, 3}), mask, sinks,
                             {.scale = 1.0F / std::sqrt(static_cast<float>(head)),
                              .sinks = true,
                              .accumulate = ir::Accumulate::kF32,
                              .n_kv_max = 384}));
        first += rows;
      }
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
      const co::StepGraph step{
          .graph = std::move(*graph), .inputs = inputs, .weights = {}, .state = state};
      auto* primitive = LowerStateBanks(step, shape, addresses, facts, false);
      auto* bank = LowerStateBanks(step, shape, addresses, facts, true);
      ASSERT_NE(primitive, nullptr);
      ASSERT_NE(bank, nullptr);
      ASSERT_EQ(primitive->keep.size(), shape.segments.size());
      ASSERT_EQ(bank->keep.size(), shape.segments.size());
      EXPECT_EQ(std::ranges::count_if(primitive->nodes,
                                      [](const auto* node) { return node->op == GGML_OP_CONCAT; }),
                shape.segments.size());
      EXPECT_FALSE(std::ranges::any_of(
          bank->nodes, [](const auto* node) { return node->op == GGML_OP_CONCAT; }));
      for (const auto* program : {primitive, bank}) {
        std::size_t attention_steps = 0;
        for (const auto& selected : program->plan.steps) {
          if (selected.operation != ex::Operation::kFlashAttn) continue;
          ++attention_steps;
          ASSERT_EQ(selected.implementation, kg::kFlashAttnMmaName);
          ASSERT_EQ(selected.nodes.size(), 1U);
          const auto* node = selected.nodes.front();
          const auto valid = kg::CheckFlashAttnMma(node);
          ASSERT_TRUE(valid) << (valid ? "" : valid.error().detail);
          EXPECT_EQ(node->op_params[kg::kFlashAttnSparseParam], 0);
          // This is the exact planner used by the selected implementation's
          // launch, not an inference from tensor dimensions alone.
          const auto recipe = kg::PlanFlashAttnMma(*launch, node, false);
          ASSERT_TRUE(recipe) << (recipe ? "" : recipe.error().detail);
          EXPECT_EQ(recipe->sparse, sparse);
          if (sparse) {
            EXPECT_EQ(recipe->columns, 1);
            EXPECT_GT(recipe->scratch, 0U);
          }
        }
        EXPECT_EQ(attention_steps, shape.segments.size());
      }
      std::vector<std::vector<float>> initial;
      for (const int shift : {0, 1}) {
        SCOPED_TRACE(shift);
        for (auto* program : {primitive, bank}) {
          Upload(Leaf(*program, queries),
                 Values(head * 8 * shape.rows(), shift == 0 ? 1.0F : -0.5F));
          Upload(Leaf(*program, sinks), Values(8, shift == 0 ? 0.25F : 0.75F));
          for (const auto& leaf : state) {
            const auto values = Values(head * step.graph.tensor(leaf.tensor).ne(1),
                                       0.5F + static_cast<float>(shift + leaf.ref.segment));
            std::vector<ggml_fp16_t> half(values.size());
            ggml_fp32_to_fp16_row(values.data(), half.data(),
                                  static_cast<std::int64_t>(half.size()));
            Upload(Leaf(*program, leaf.tensor), half);
          }
          for (std::size_t owner = 0; owner < masks.size(); ++owner) {
            std::vector<float> mask(cells * 32, -std::numeric_limits<float>::infinity());
            for (std::size_t row = 0; row < shape.segments[owner].rows; ++row) {
              std::size_t visible = 0;
              for (std::size_t cell = 0; cell < cells; ++cell) {
                // Sparse cases wrap the selected prefix across the far end
                // of the compressed bank, including after capture restaging.
                const auto offset = sparse ? row * 11 + static_cast<std::size_t>(shift) * 37 : 0;
                if ((cell < 128 && (cell + row + static_cast<std::size_t>(shift)) % 3 != 0) ||
                    (cell >= 512 && (cell - 512 + offset) % compressed_read <
                                        32 * (row + 1 + owner + static_cast<std::size_t>(shift)))) {
                  mask[row * cells + cell] = 0;
                  ++visible;
                }
              }
              ASSERT_LE(visible, 384U);  // the original n_kv_max promise
            }
            std::vector<ggml_fp16_t> half(mask.size());
            ggml_fp32_to_fp16_row(mask.data(), half.data(), static_cast<std::int64_t>(half.size()));
            Upload(Leaf(*program, masks[owner]), half);
          }
        }
        RunAndReplay(*primitive);
        RunAndReplay(*bank);
        for (std::size_t owner = 0; owner < shape.segments.size(); ++owner) {
          const auto expected = Read(primitive->keep[owner]);
          const auto actual = Read(bank->keep[owner]);
          ASSERT_EQ(actual.size(), expected.size());
          for (std::size_t i = 0; i < actual.size(); ++i)
            ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                      std::bit_cast<std::uint32_t>(expected[i]));
          if (shift == 0)
            initial.push_back(actual);
          else
            EXPECT_NE(actual, initial[owner]);
        }
      }
      EXPECT_EQ(captures.count(primitive), 1U);
      EXPECT_EQ(captures.count(bank), 1U);
    }
  }
}

TEST_F(GgmlLowerDsv4Gpu, QHeadsMatchExistingHelperAndReplayWithNewPositions) {
  std::map<std::int32_t, std::vector<float>> one_chunk;
  for (const std::uint32_t rows : {16U, 65U}) {
    ir::Builder b;
    const auto q = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({512, 64, rows})));
    const auto pos = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({rows})));
    const ir::NormRopeParams params{.eps = 1e-6F,
                                    .rope = {.mode = ir::RopeMode::kNormal,
                                             .dims = 64,
                                             .offset = 448,
                                             .original_context = 65536,
                                             .base = 1000000.0F,
                                             .scale = 0.125F,
                                             .ext_factor = 1.0F,
                                             .attn_factor = 1.25F}};
    const auto out = b.Fused(ir::FusedKind::kDsv4QHead, std::array{q, pos}, params)[0];
    b.Output(out);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    const pl::StepShape shape{
        .segments = {{.rows = 7, .past = 127, .outputs = 0, .want_head = false},
                     {.slot = 1, .rows = rows - 7, .past = 812, .outputs = 0, .want_head = false}}};
    auto* lowered_step = Lower(std::move(*graph),
                               {{.tensor = q, .kind = co::InputKind::kFeatures},
                                {.tensor = pos, .kind = co::InputKind::kPositions}},
                               shape);
    ASSERT_NE(lowered_step, nullptr);
    auto* qt = Leaf(*lowered_step, q);
    auto* pt = Leaf(*lowered_step, pos);
    ASSERT_NE(qt, nullptr);
    ASSERT_NE(pt, nullptr);
    Upload(qt, Values(static_cast<std::size_t>(512 * 64 * rows)));
    const auto& r = params.rope;
    auto* direct =
        Place(kg::Dsv4QHead(Context(), qt, pt,
                            {.eps = params.eps,
                             .original_context = static_cast<std::int32_t>(r.original_context),
                             .base = r.base,
                             .scale = r.scale,
                             .extension = r.ext_factor,
                             .attention = r.attn_factor,
                             .beta_fast = r.beta_fast,
                             .beta_slow = r.beta_slow}));
    for (const std::int32_t shift : {0, 4096}) {
      std::vector<std::int32_t> positions(rows);
      for (std::uint32_t i = 0; i < rows; ++i)
        positions[i] = shift + static_cast<std::int32_t>(i < 7 ? 127 + i : 812 + i - 7);
      Upload(pt, positions);
      ASSERT_TRUE(kg::RunDsv4QHead(*launch, direct));
      const auto expected = Read(direct);
      RunAndReplay(*lowered_step);
      EXPECT_EQ(Read(lowered_step->keep[0]), expected);
      if (rows == 16)
        one_chunk[shift] = expected;
      else
        EXPECT_TRUE(
            std::equal(one_chunk.at(shift).begin(), one_chunk.at(shift).end(), expected.begin()));
    }
  }
}

TEST_F(GgmlLowerDsv4Gpu, CompressorMaskedDummyMatchesFallbackAndExistingHelper) {
  ir::Builder b;
  b.SetOwner(0);
  const auto input = [&] { return b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({2, 1}))); };
  const auto state = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({2, 2})));
  const auto scores = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({2, 2})));
  const auto current = input(), current_scores = input();
  const auto ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2})));
  const ir::CompressorParams params{.ratio = 1, .overlap = true};
  b.Output(b.Fused(ir::FusedKind::kDsv4Compress,
                   std::array{state, scores, current, current_scores, ids}, params)[0]);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << graph.error().detail;
  const pl::StepShape shape{.segments = {{.rows = 1, .outputs = 0, .want_head = false}}};
  const std::vector<co::StepInput> inputs{
      {.tensor = state, .kind = co::InputKind::kFeatures},
      {.tensor = scores, .kind = co::InputKind::kFeatures},
      {.tensor = current, .kind = co::InputKind::kFeatures},
      {.tensor = current_scores, .kind = co::InputKind::kFeatures},
      {.tensor = ids,
       .kind = co::InputKind::kPoolReads,
       .segment = 0,
       .cells = {.capacity = 2, .ring = true, .recent = true},
       .ratio = 1,
       .overlap = true,
       .ring = 2}};
  auto implementations = kg::Implementations();
  std::erase_if(implementations, [](const auto& i) { return i.name == kg::kDsv4CompressName; });
  auto primitive = ex::Registry::Create(std::move(implementations));
  ASSERT_TRUE(primitive);
  auto* fast = Lower(graph->Clone(), inputs, shape);
  auto* fallback = Lower(std::move(*graph), inputs, shape, &*primitive);
  ASSERT_NE(fast, nullptr);
  ASSERT_NE(fallback, nullptr);
  const auto values = co::InputValues(inputs.back(), shape, {});
  ASSERT_TRUE(values) << values.error();
  EXPECT_EQ(*values, (std::vector<std::int64_t>{3, 2}));
  const std::vector<std::int32_t> reads(values->begin(), values->end());
  const auto neg_inf = -std::numeric_limits<float>::infinity();
  for (auto* step : {fast, fallback}) {
    Upload(Leaf(*step, state), std::vector<float>{1, 100, 2, 200});
    Upload(Leaf(*step, scores), std::vector<float>{neg_inf, 0, 0, 0});
    Upload(Leaf(*step, current), std::vector<float>{9, 900});
    Upload(Leaf(*step, current_scores), std::vector<float>{0, 0});
    Upload(Leaf(*step, ids), reads);
    RunAndReplay(*step);
    EXPECT_EQ(Read(step->keep[0]), std::vector<float>{900});
    Upload(Leaf(*step, current_scores), std::vector<float>{0, neg_inf});
    RunAndReplay(*step);
    EXPECT_EQ(Read(step->keep[0]), std::vector<float>{0});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (const float score : {0.0F, neg_inf, nan}) {
      for (const float value : {900.0F, nan, std::numeric_limits<float>::infinity()}) {
        SCOPED_TRACE(step == fast ? "fast" : "primitive");
        SCOPED_TRACE(score);
        SCOPED_TRACE(value);
        Upload(Leaf(*step, current), std::vector<float>{9, value});
        Upload(Leaf(*step, current_scores), std::vector<float>{0, score});
        RunAndReplay(*step);
        const auto got = Read(step->keep[0]);
        ASSERT_EQ(got.size(), 1U);
        if (!std::isfinite(value) && (std::isnan(value) || score != 0.0F))
          EXPECT_TRUE(std::isnan(got[0]));
        else
          EXPECT_EQ(got[0], score == 0.0F ? value : 0.0F);
      }
    }
    // Restore partial masking for the direct-helper comparison below.
    Upload(Leaf(*step, current), std::vector<float>{9, 900});
    Upload(Leaf(*step, current_scores), std::vector<float>{0, 0});
    RunAndReplay(*step);
  }
  const auto lowered_values = Read(fast->keep[0]);
  Restage(*fast);
  auto* direct = Place(kg::Dsv4Compress(Context(), Leaf(*fast, state), Leaf(*fast, scores),
                                        Leaf(*fast, current), Leaf(*fast, current_scores),
                                        Leaf(*fast, ids), 1, true));
  ASSERT_TRUE(kg::RunDsv4Compress(*launch, direct));
  EXPECT_EQ(Read(direct), lowered_values);
}

TEST_F(GgmlLowerDsv4Gpu, CompressorReadsPitchedOwnerViewsAcrossBlockBoundaries) {
  struct Case {
    std::uint32_t ratio;
    bool overlap;
    std::uint32_t rows;
    std::uint32_t past;
  };
  for (const auto test : {Case{4, true, 7, 0}, Case{4, true, 9, 127}, Case{128, false, 130, 127}}) {
    SCOPED_TRACE(test.ratio);
    SCOPED_TRACE(test.past);
    constexpr std::uint64_t head = 8;
    const std::uint64_t halves = test.overlap ? 2 : 1;
    const std::uint64_t channels = head * halves, pitch = channels + 4;
    const std::uint64_t ring = test.ratio * halves;
    const std::uint64_t blocks =
        co::CompressorBlocks(test.rows, test.past, test.ratio, test.overlap);
    const std::uint64_t count = blocks * test.ratio * halves;
    ir::Builder b;
    b.SetOwner(0);
    std::array<ir::TensorId, 4> roots{}, views{};
    std::array<std::vector<float>, 4> values;
    std::vector<co::StepInput> inputs;
    for (std::size_t source = 0; source < roots.size(); ++source) {
      const std::uint64_t rows = source < 2 ? ring : test.rows;
      roots[source] = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({pitch, rows})));
      inputs.push_back({.tensor = roots[source], .kind = co::InputKind::kFeatures});
      views[source] = b.View(roots[source], ir::ShapeOf({channels, rows}),
                             {4, pitch * 4, pitch * rows * 4, pitch * rows * 4}, 8);
      // Unread padding is NaN so offset/stride mistakes cannot accidentally
      // agree with the logical rows. Each owner begins two floats in.
      values[source].assign(pitch * rows, std::numeric_limits<float>::quiet_NaN());
      for (std::uint64_t row = 0; row < rows; ++row)
        for (std::uint64_t ch = 0; ch < channels; ++ch)
          values[source][row * pitch + 2 + ch] =
              source % 2 == 0 ? static_cast<float>(ch + 1 + 3 * row + (source < 2 ? 0 : 100))
                              : (ch % head == 0 ? -std::numeric_limits<float>::infinity() : 0.0F);
    }
    const auto ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({count})));
    inputs.push_back({.tensor = ids,
                      .kind = co::InputKind::kPoolReads,
                      .segment = 0,
                      .cells = {.capacity = ring, .ring = true, .recent = true},
                      .ratio = test.ratio,
                      .overlap = test.overlap,
                      .ring = ring});
    b.Output(b.Fused(ir::FusedKind::kDsv4Compress,
                     std::array{views[0], views[1], views[2], views[3], ids},
                     ir::CompressorParams{.ratio = test.ratio, .overlap = test.overlap})[0]);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    const pl::StepShape shape{
        .segments = {{.rows = test.rows, .past = test.past, .outputs = 0, .want_head = false}}};
    const auto generated = co::InputValues(inputs.back(), shape, {});
    ASSERT_TRUE(generated) << generated.error();
    ASSERT_EQ(generated->size(), count);
    const std::vector<std::int32_t> reads(generated->begin(), generated->end());
    std::vector<float> expected(head * blocks, 0.0F);
    for (std::uint64_t block = 0; block < blocks; ++block) {
      for (std::uint64_t ch = 1; ch < head; ++ch) {
        float sum = 0;
        std::uint32_t seen = 0;
        for (std::uint64_t j = 0; j < test.ratio * halves; ++j) {
          const bool second = test.overlap && j >= test.ratio;
          const auto read =
              static_cast<std::uint64_t>(reads[(second ? blocks * test.ratio : 0) +
                                               block * test.ratio + (second ? j - test.ratio : j)]);
          if (read >= ring + test.rows) continue;
          const auto source = read < ring ? 0U : 2U;
          const auto row = read < ring ? read : read - ring;
          sum += values[source][row * pitch + 2 + ch + (second ? head : 0)];
          ++seen;
        }
        ASSERT_TRUE(std::has_single_bit(seen));
        expected[block * head + ch] = sum / static_cast<float>(seen);
      }
    }
    auto implementations = kg::Implementations();
    std::erase_if(implementations, [](const auto& i) { return i.name == kg::kDsv4CompressName; });
    auto primitive = ex::Registry::Create(std::move(implementations));
    ASSERT_TRUE(primitive);
    for (const auto* selected : {&*registry, &*primitive}) {
      auto* step = Lower(graph->Clone(), inputs, shape, selected);
      ASSERT_NE(step, nullptr);
      for (std::size_t i = 0; i < roots.size(); ++i) Upload(Leaf(*step, roots[i]), values[i]);
      Upload(Leaf(*step, ids), reads);
      RunAndReplay(*step);
      EXPECT_EQ(Read(step->keep[0]), expected);
    }
  }
}

TEST_F(GgmlLowerDsv4Gpu, HcCarriesAndDownstreamOwnersMatchExistingHelpers) {
  constexpr std::uint64_t width = 4096;
  for (const auto encoding : {Encoding::kF32, Encoding::kF16, Encoding::kBF16}) {
    std::vector<std::vector<float>> one_row;
    for (const std::uint32_t rows : {1U, 4U}) {
      SCOPED_TRACE(static_cast<int>(encoding));
      SCOPED_TRACE(rows);
      ir::Builder b;
      const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, 4, rows})));
      const auto fn = b.Input(ir::Dense(encoding, ir::ShapeOf({width * 4, 24})));
      const auto scale = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({3})));
      const auto base = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({24})));
      const auto norm = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width})));
      const ir::SinkhornPreParams params{
          .streams = 4, .eps = 1e-6F, .hc_eps = 1e-6F, .iterations = 20};
      const auto pre =
          b.Fused(ir::FusedKind::kDsv4HcPre, std::array{x, fn, scale, base, norm}, params);
      for (const auto output : pre.span()) b.Output(output);
      // A later activation beside the carry roots detects uncharged tail or
      // partial writes; unequal owners consume each packed carry independently.
      const auto next = b.Scale(pre[0], 0.25F);
      b.Output(next);
      pl::StepShape shape;
      shape.segments.push_back({.rows = 1, .outputs = 0, .want_head = false});
      if (rows > 1)
        shape.segments.push_back({.slot = 1, .rows = rows - 1, .outputs = 0, .want_head = false});
      std::uint64_t first = 0;
      for (std::uint32_t owner = 0; owner < shape.segments.size(); ++owner) {
        b.SetOwner(static_cast<std::uint8_t>(owner));
        const std::uint64_t n = shape.segments[owner].rows;
        const auto part = [&](ir::TensorId t, llmp::format::Shape shape_part,
                              std::uint64_t offset) {
          // Each owner is a contiguous last-axis run; inactive strides
          // describe that run, not the entire joined root.
          return b.View(t, shape_part, ir::NaturalStrides(ir::Dense(Encoding::kF32, shape_part)),
                        offset);
        };
        const auto y = part(next, ir::ShapeOf({width, n}), first * width * 4);
        const auto residual = part(x, ir::ShapeOf({width, 4, n}), first * width * 4 * 4);
        const auto post = part(pre[1], ir::ShapeOf({4, n}), first * 4 * 4);
        const auto comb = part(pre[2], ir::ShapeOf({4, 4, n}), first * 16 * 4);
        const auto result = b.Fused(ir::FusedKind::kHcNormPost, std::array{y, residual, post, comb},
                                    ir::HcNormPostParams{.eps = params.eps, .streams = 4});
        for (const auto output : result.span()) b.Output(output);
        first += n;
      }
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph) << graph.error().detail;
      std::vector<co::StepInput> inputs;
      for (const auto t : {x, fn, scale, base, norm})
        inputs.push_back({.tensor = t, .kind = co::InputKind::kFeatures});
      auto* step = Lower(std::move(*graph), inputs, shape);
      ASSERT_NE(step, nullptr);
      Upload(Leaf(*step, x), Values(width * 4 * rows));
      const auto weights = Values(width * 4 * 24, 0.01F);
      if (encoding == Encoding::kF32) {
        Upload(Leaf(*step, fn), weights);
      } else if (encoding == Encoding::kF16) {
        std::vector<ggml_fp16_t> rounded(weights.size());
        ggml_fp32_to_fp16_row(weights.data(), rounded.data(),
                              static_cast<std::int64_t>(weights.size()));
        Upload(Leaf(*step, fn), rounded);
      } else {
        std::vector<ggml_bf16_t> rounded(weights.size());
        ggml_fp32_to_bf16_row(weights.data(), rounded.data(),
                              static_cast<std::int64_t>(weights.size()));
        Upload(Leaf(*step, fn), rounded);
      }
      Upload(Leaf(*step, scale), std::vector<float>{0.6F, 0.7F, 0.8F});
      Upload(Leaf(*step, base), Values(24, 0.1F));
      Upload(Leaf(*step, norm), std::vector<float>(width, 1.0F));
      auto* c = Context();
      auto* partials = Place(kg::Dsv4HcMix(c, Leaf(*step, x), Leaf(*step, fn)));
      auto* blob = Place(kg::Dsv4HcPre(
          c, partials, Leaf(*step, x), Leaf(*step, scale), Leaf(*step, base), Leaf(*step, norm),
          params.eps, params.hc_eps, static_cast<std::int32_t>(params.iterations)));
      ASSERT_TRUE(kg::RunDsv4HcMix(*launch, partials));
      ASSERT_TRUE(kg::RunDsv4HcPre(*launch, blob));
      ASSERT_TRUE(Finish());
      auto* direct_y = ggml_view_2d(c, blob, width, rows, width * 4, 0);
      auto* direct_post = ggml_view_2d(c, blob, 4, rows, kg::kDsv4HcTail * 4, width * rows * 4);
      auto* direct_comb =
          ggml_view_3d(c, blob, 4, 4, rows, 16, kg::kDsv4HcTail * 4, width * rows * 4 + 16);
      RunAndReplay(*step);
      EXPECT_EQ(Read(step->keep[0]), Read(direct_y));
      const auto posts = Read(direct_post), combinations = Read(direct_comb);
      EXPECT_EQ(Read(step->keep[1]), posts);
      EXPECT_EQ(Read(step->keep[2]), combinations);
      if (rows == 1) {
        for (const auto* kept : step->keep) one_row.push_back(Read(kept));
      } else {
        // The first owner's row has identical input and parameters alone
        // and in a 1+3 wave, including both downstream outputs.
        for (std::size_t i = 0; i < one_row.size(); ++i) {
          const auto joined = Read(step->keep[i]);
          EXPECT_TRUE(std::equal(one_row[i].begin(), one_row[i].end(), joined.begin()));
        }
      }
      auto expected_y = Read(direct_y);
      for (auto& v : expected_y) v *= 0.25F;
      EXPECT_EQ(Read(step->keep[3]), expected_y);
      first = 0;
      for (std::size_t owner = 0; owner < shape.segments.size(); ++owner) {
        const auto n = shape.segments[owner].rows;
        auto* dc = Context();
        auto* y = Place(ggml_new_tensor_2d(dc, GGML_TYPE_F32, width, n));
        Upload(y, std::vector<float>(
                      expected_y.begin() + static_cast<std::ptrdiff_t>(first * width),
                      expected_y.begin() + static_cast<std::ptrdiff_t>((first + n) * width)));
        auto* residual = Place(ggml_new_tensor_3d(dc, GGML_TYPE_F32, width, 4, n));
        const auto original_x = Values(width * 4 * rows);
        Upload(residual,
               std::vector<float>(
                   original_x.begin() + static_cast<std::ptrdiff_t>(first * width * 4),
                   original_x.begin() + static_cast<std::ptrdiff_t>((first + n) * width * 4)));
        auto* post = Place(ggml_new_tensor_2d(dc, GGML_TYPE_F32, 4, n));
        auto* comb = Place(ggml_new_tensor_3d(dc, GGML_TYPE_F32, 4, 4, n));
        Upload(post,
               std::vector<float>(posts.begin() + static_cast<std::ptrdiff_t>(first * 4),
                                  posts.begin() + static_cast<std::ptrdiff_t>((first + n) * 4)));
        Upload(comb, std::vector<float>(
                         combinations.begin() + static_cast<std::ptrdiff_t>(first * 16),
                         combinations.begin() + static_cast<std::ptrdiff_t>((first + n) * 16)));
        auto* output = Place(ggml_dsv4_hc_post(dc, y, residual, post, comb));
        auto* flat = ggml_reshape_2d(dc, output, width * 4, n);
        auto* normalized = Place(kg::Dsv4HcNormF16(dc, flat, params.eps, GGML_TYPE_F32));
        ASSERT_TRUE(kg::RunDsv4HcPostNormF16(*launch, output, normalized));
        EXPECT_EQ(Read(step->keep[4 + 2 * owner]), Read(output));
        EXPECT_EQ(Read(step->keep[5 + 2 * owner]), Read(normalized));
        first += n;
      }
    }
  }
}

// Independent scalar ordering of supplied F32 bits. No score arithmetic and
// no radix-key imitation: numeric descending, NaN last, stable lower-index ties.
std::vector<std::int32_t> VisibleIds(const std::vector<float>& scores, std::int32_t count,
                                     std::size_t top) {
  const auto visible = static_cast<std::size_t>(
      std::clamp<std::int64_t>(count, 0, static_cast<std::int64_t>(scores.size())));
  std::vector<std::int32_t> ids(visible);
  std::iota(ids.begin(), ids.end(), 0);
  std::ranges::stable_sort(ids, [&](auto a, auto b) {
    const auto x = scores[static_cast<std::size_t>(a)], y = scores[static_cast<std::size_t>(b)];
    if (std::isnan(x) || std::isnan(y)) return !std::isnan(x) && std::isnan(y);
    return x > y;
  });
  ids.resize(std::min(top, visible));
  std::ranges::sort(ids);
  ids.resize(top, -1);
  return ids;
}

TEST_F(GgmlLowerDsv4Gpu, ContF16PreservesBitsAcrossCopyPathsAndSameCapture) {
  constexpr std::array<std::uint16_t, 10> bits{0x7d01, 0xfd35, 0x8000, 0,      1,
                                               0x83ff, 0x7c00, 0xfc00, 0x3c00, 0xbc00};
  struct Case {
    std::size_t first, row;
    bool transpose;
    kg::ContCopy copy;
  };
  const std::array cases{
      Case{2, 16, false, kg::ContCopy::kMemcpy}, Case{2, 40, false, kg::ContCopy::kMemcpy2d},
      Case{0, 0, false, kg::ContCopy::kScalar},  Case{0, 40, false, kg::ContCopy::kScalar},
      Case{4, 0, false, kg::ContCopy::kScalar},  Case{4, 6, false, kg::ContCopy::kScalar},
      Case{4, 40, false, kg::ContCopy::kScalar}, Case{2, 16, true, kg::ContCopy::kTranspose}};
  for (const auto& test : cases) {
    auto* c = Context();
    auto* root = Place(ggml_new_tensor_2d(c, GGML_TYPE_F16, 20, 3));
    auto* values = ggml_view_2d(c, root, 8, 3, test.row, 2);
    values->nb[0] = test.first;
    if (test.transpose) values = ggml_transpose(c, values);
    kg::TensorArena::Bind(values, reinterpret_cast<std::uintptr_t>(root->data) + 2);
    auto* output = Place(ggml_cont(c, values));
    const auto copy = kg::CheckCont(output);
    ASSERT_TRUE(copy) << copy.error().detail;
    EXPECT_EQ(*copy, test.copy);
    std::optional<std::size_t> captured;
    for (std::size_t revision = 0; revision < 3; ++revision) {
      std::vector<std::uint16_t> source(60), expected;
      for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = bits[(i + revision) % bits.size()];
      for (std::int64_t row = 0; row < values->ne[1]; ++row)
        for (std::int64_t cell = 0; cell < values->ne[0]; ++cell)
          expected.push_back(source[1 + static_cast<std::size_t>(row) * values->nb[1] / 2 +
                                    static_cast<std::size_t>(cell) * values->nb[0] / 2]);
      Upload(root, source);
      ASSERT_TRUE(kg::Cont(*launch, output));
      EXPECT_EQ(ReadTyped<std::uint16_t>(output), expected);
      if (!captured) {
        auto graph = launch->Capture([&](kg::LaunchContext& l) { return kg::Cont(l, output); });
        ASSERT_TRUE(graph);
        graphs.push_back(std::move(*graph));
        captured = graphs.size() - 1;
      }
      ASSERT_TRUE(launch->Launch(graphs[*captured]));
      EXPECT_EQ(ReadTyped<std::uint16_t>(output), expected);
    }
  }
}

TEST_F(GgmlLowerDsv4Gpu, PrefixOverwriteCopiesBitsAcrossPitchedRepeatedViewsAndSameCapture) {
  constexpr std::array<std::uint32_t, 10> bits{0x7fa00001, 0xffa12345, 0x80000000, 0,
                                               1,          0x807fffff, 0x7f800000, 0xff800000,
                                               0x3f800000, 0xbf800000};
  constexpr std::int64_t cells = 17, rows = 3, pitch = 40;
  for (const std::uint64_t cell_stride : {0U, 8U})
    for (const std::uint64_t row_stride : std::array<std::uint64_t, 2>{0, pitch * 4})
      for (const std::uint64_t count_stride : {0U, 8U})
        for (const std::uint32_t fill : {0xff800000U, 0x80000000U, 0x7fa12345U}) {
          SCOPED_TRACE(::testing::PrintToString(
              std::make_tuple(cell_stride, row_stride, count_stride, fill)));
          auto* c = Context();
          auto* root = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, pitch, rows));
          auto* values = ggml_view_2d(c, root, cells, rows, row_stride, 4);
          values->nb[0] = cell_stride;
          kg::TensorArena::Bind(values, reinterpret_cast<std::uintptr_t>(root->data) + 4);
          auto* count_root = Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, rows * 2 + 1));
          auto* counts = ggml_view_1d(c, count_root, rows, 4);
          counts->nb[0] = count_stride;
          kg::TensorArena::Bind(counts, reinterpret_cast<std::uintptr_t>(count_root->data) + 4);
          auto* output = Place(kg::PrefixOverwrite(c, values, counts, fill));
          ASSERT_TRUE(kg::CheckPrefixOverwrite(output));
          std::optional<std::size_t> captured;
          for (std::size_t revision = 0; revision < 4; ++revision) {
            SCOPED_TRACE(revision);
            std::vector<std::uint32_t> source(pitch * rows);
            for (std::size_t i = 0; i < source.size(); ++i)
              source[i] = bits[(i + revision) % bits.size()];
            const std::array<std::int32_t, 4> limits{INT32_MIN, 0, 8, INT32_MAX};
            std::vector<std::int32_t> visible(rows * 2 + 1, -99);
            for (std::size_t row = 0; row < rows; ++row)
              visible[1 + 2 * row] = limits[(row + revision) % limits.size()];
            std::vector<std::uint32_t> expected;
            for (std::uint64_t row = 0; row < rows; ++row)
              for (std::uint64_t cell = 0; cell < cells; ++cell)
                expected.push_back(static_cast<std::int64_t>(cell) <
                                           visible[1 + row * count_stride / 4]
                                       ? source[1 + row * row_stride / 4 + cell * cell_stride / 4]
                                       : fill);
            Upload(root, source);
            Upload(count_root, visible);
            ASSERT_TRUE(kg::RunPrefixOverwrite(*launch, output));
            EXPECT_EQ(ReadTyped<std::uint32_t>(output), expected);
            if (!captured) {
              auto graph = launch->Capture(
                  [&](kg::LaunchContext& l) { return kg::RunPrefixOverwrite(l, output); });
              ASSERT_TRUE(graph);
              graphs.push_back(std::move(*graph));
              captured = graphs.size() - 1;
            }
            ASSERT_TRUE(launch->Launch(graphs[*captured]));
            EXPECT_EQ(ReadTyped<std::uint32_t>(output), expected);
          }
        }
}

TEST_F(GgmlLowerDsv4Gpu, RoundedIndexScoreExecutesPortableFallbackWithLiteralNonfiniteTails) {
  // The primitive matrix product borrows an explicitly owned cuBLAS handle.
  // It outlives this context and all graphs, and retires before the stream.
  ASSERT_TRUE(Finish());
  const auto workspace = launch->workspace();
  int major = 0, minor = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
  const auto blas_bytes = kg::CublasHandle::UpstreamWorkspace(100 * major + 10 * minor);
  auto handle = kg::CublasHandle::Create(
      0, *execution, stream, {.base = Allocate(blas_bytes.value()), .size = blas_bytes});
  ASSERT_TRUE(handle) << handle.error().detail;
  cublas = std::move(*handle);
  launch.reset();
  auto context = kg::LaunchContext::Create(0, *execution, stream, workspace, cublas.get());
  ASSERT_TRUE(context) << context.error().detail;
  launch = std::move(*context);
  std::vector<ex::Implementation> primitive_impls;
  for (const auto& impl : kg::Implementations())
    if (impl.name != kg::kRoundedIndexScoreName) primitive_impls.push_back(impl);
  auto primitive = ex::Registry::Create(primitive_impls);
  ASSERT_TRUE(primitive);
  for (const bool pitched : {false, true}) {
    SCOPED_TRACE(pitched);
    ir::Builder b;
    const std::uint64_t query_pitch = pitched ? 130 : 128;
    const std::uint64_t key_pitch = pitched ? 130 : 128;
    const std::uint64_t weight_pitch = pitched ? 66 : 64;
    const std::uint64_t offset = pitched ? 1 : 0;
    const auto q_root = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({query_pitch, 64, 3})));
    const auto k_root = b.Input(ir::Dense(Encoding::kF16, ir::ShapeOf({key_pitch, 1, 4})));
    const auto w_root = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({weight_pitch, 3})));
    const auto q =
        b.View(q_root, ir::ShapeOf({128, 64, 3}),
               {4, query_pitch * 4, query_pitch * 64 * 4, query_pitch * 64 * 3 * 4}, offset * 4);
    const auto k = b.View(k_root, ir::ShapeOf({128, 1, 4}),
                          {2, key_pitch * 2, key_pitch * 2, key_pitch * 4 * 2}, offset * 2);
    const auto w =
        b.View(w_root, ir::ShapeOf({64, 3}),
               {4, weight_pitch * 4, weight_pitch * 3 * 4, weight_pitch * 3 * 4}, offset * 4);
    const auto counts = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
    const auto scores = b.RoundedIndexScore(q, k, w, counts);
    const auto ids = b.VisibleTopK(scores, counts, 2);
    const auto mask = b.SelectedMask(b.Fill(ir::ShapeOf({2, 3}), 0, Encoding::kF16), ids,
                                     {.raw_cells = 3, .cells = 4});
    for (const auto output : {scores, ids, mask}) b.Output(output);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    std::vector<co::StepInput> inputs;
    for (const auto input : {q_root, k_root, w_root, counts})
      inputs.push_back({.tensor = input, .kind = co::InputKind::kFeatures});
    const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 3}}};
    auto* step = Lower(std::move(*graph), std::move(inputs), shape, &*primitive, true);
    ASSERT_NE(step, nullptr);
    int prefixes = 0, conversions = 0;
    for (const auto* node : step->nodes) {
      EXPECT_NE(kg::LlmpOpOf(node), kg::LlmpOp::kDsv4LidTopK);
      EXPECT_NE(kg::LlmpOpOf(node), kg::LlmpOp::kRoundedIndexScore);
      EXPECT_NE(node->op, GGML_OP_LIGHTNING_INDEXER);
      prefixes += kg::LlmpOpOf(node) == kg::LlmpOp::kPrefixOverwrite;
      conversions += node->op == GGML_OP_CPY;
    }
    EXPECT_EQ(prefixes, 1);
    EXPECT_EQ(conversions, 2);
    auto bound = kg::BoundGraph::Bind(*primitive, step->plan);
    ASSERT_TRUE(bound) << bound.error().detail;
    std::optional<std::size_t> captured;
    const float inf = std::numeric_limits<float>::infinity();
    for (std::size_t revision = 0; revision < 3; ++revision) {
      std::vector<float> query(query_pitch * 64 * 3, 0), weights(weight_pitch * 3, 0);
      std::vector<std::uint16_t> keys(key_pitch * 4, 0);
      keys[offset] = 0x3c00;
      keys[offset + key_pitch] = 0xbc00;
      keys[offset + key_pitch * 2] = 0x7c00;
      keys[offset + key_pitch * 3] = 0x7e01;
      for (std::size_t row = 0; row < 3; ++row) {
        query[offset + row * query_pitch * 64] = revision == 0 ? 1.0002F : 2.0002F;
        weights[offset + row * weight_pitch] =
            revision == 2 ? std::numeric_limits<float>::quiet_NaN() : 2.0F;
      }
      const std::vector<std::int32_t> visible = revision == 0
                                                    ? std::vector<std::int32_t>{1, INT32_MAX, -1}
                                                    : std::vector<std::int32_t>{4, 2, 0};
      Upload(Leaf(*step, q_root), query);
      Upload(Leaf(*step, k_root), keys);
      Upload(Leaf(*step, w_root), weights);
      Upload(Leaf(*step, counts), visible);
      std::vector<float> expected(12, -inf);
      std::vector<std::int32_t> expected_ids;
      std::vector<std::uint16_t> expected_mask(21, 0xfc00);
      for (std::size_t row = 0; row < 3; ++row) {
        const std::vector<float> logical =
            revision == 2 ? std::vector<float>(4, std::numeric_limits<float>::quiet_NaN())
                          : std::vector<float>{revision == 0 ? 2.0F : 4.0F, 0, inf, 0};
        for (std::size_t cell = 0; cell < 4; ++cell)
          if (static_cast<std::int64_t>(cell) < visible[row])
            expected[row * 4 + cell] = logical[cell];
        const auto selected = VisibleIds(logical, visible[row], 2);
        expected_ids.insert(expected_ids.end(), selected.begin(), selected.end());
        expected_mask[row * 7] = expected_mask[row * 7 + 1] = 0;
        for (const auto id : selected)
          if (id >= 0) expected_mask[row * 7 + 3 + static_cast<std::size_t>(id)] = 0;
      }
      const auto check = [&] {
        const auto got = Read(step->keep[0]);
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
          if (std::isnan(expected[i]))
            EXPECT_TRUE(std::isnan(got[i])) << i;
          else
            EXPECT_EQ(got[i], expected[i]) << i;
        }
        EXPECT_EQ(ReadTyped<std::int32_t>(step->keep[1]), expected_ids);
        EXPECT_EQ(ReadTyped<std::uint16_t>(step->keep[2]), expected_mask);
      };
      Restage(*step);
      const auto ran = bound->Run(*launch);
      ASSERT_TRUE(ran) << ran.error().detail;
      check();
      if (!captured) {
        auto capture = launch->Capture([&](kg::LaunchContext& l) { return bound->Run(l); });
        ASSERT_TRUE(capture) << capture.error().detail;
        graphs.push_back(std::move(*capture));
        captured = graphs.size() - 1;
      }
      Restage(*step);
      ASSERT_TRUE(launch->Launch(graphs[*captured]));
      check();
    }
  }
}

TEST_F(GgmlLowerDsv4Gpu, VisibleTopKAndSelectedMaskMatchArbitraryBitsAcrossPersistentReplay) {
  constexpr std::array<std::uint32_t, 16> bits{
      0x7fc01234, 0xffc05678, 0x7fa00001, 0xff800000, 0x80000001, 0x807fffff, 0x80000000, 0, 1,
      0x007fffff, 0x3f800000, 0x3f800000, 0x7f800000, 0x7f800000, 0xbf800000, 0xffa00001};
  constexpr std::array<std::uint16_t, 5> raw_bits{0x8000, 0x7d35, 0xfd21, 1, 0xfc00};
  for (const std::int64_t cells : {17, 4097}) {
    for (const std::int64_t rows : {1, 8, 9}) {
      std::vector<std::int64_t> tops{1, 2, 17, cells + 3};
      if (cells == 17 && rows == 1) tops.push_back(65536);
      for (const auto top : tops) {
        SCOPED_TRACE(::testing::PrintToString(std::make_tuple(cells, rows, top)));
        auto* c = Context();
        const auto score_pitch = static_cast<std::size_t>(cells * 2 + 3);
        auto* score_root = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, cells * 2 + 3, rows));
        auto* scores = ggml_view_2d(c, score_root, cells, rows, score_pitch * 4, 4);
        scores->nb[0] = 8;
        kg::TensorArena::Bind(scores, reinterpret_cast<std::uintptr_t>(score_root->data) + 4);
        auto* count_root = Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, rows * 2 + 1));
        auto* counts = ggml_view_1d(c, count_root, rows, 4);
        counts->nb[0] = 8;
        kg::TensorArena::Bind(counts, reinterpret_cast<std::uintptr_t>(count_root->data) + 4);
        auto* raw_root = Place(ggml_new_tensor_2d(c, GGML_TYPE_F16, 13, rows));
        auto* raw = ggml_view_2d(c, raw_root, 5, rows, 26, 2);
        raw->nb[0] = 4;
        kg::TensorArena::Bind(raw, reinterpret_cast<std::uintptr_t>(raw_root->data) + 2);
        auto* selected = Place(kg::VisibleTopK(c, scores, counts, top));
        auto* mask = Place(kg::SelectedMask(c, raw, selected, 8, cells));
        ASSERT_TRUE(kg::CheckVisibleTopK(selected));
        ASSERT_TRUE(kg::CheckSelectedMask(mask));
        const auto scratch = kg::PlanVisibleTopK(*launch, selected);
        ASSERT_TRUE(scratch);
        EXPECT_EQ(*scratch > 0, cells > 2048 && rows <= 8);
        const auto run = [&](kg::LaunchContext& l) -> std::expected<void, kg::KernelFailure> {
          if (auto r = kg::RunVisibleTopK(l, selected); !r) return r;
          return kg::RunSelectedMask(l, mask);
        };
        std::optional<std::size_t> captured;
        for (int revision = 0; revision < 7; ++revision) {
          SCOPED_TRACE(revision);
          std::vector<float> score_values(score_pitch * static_cast<std::size_t>(rows), 99);
          std::vector<std::int32_t> count_values(static_cast<std::size_t>(rows * 2 + 1), -99);
          std::vector<std::uint16_t> raw_values(static_cast<std::size_t>(rows * 13), 0xffff);
          std::vector<std::int32_t> expected_ids;
          std::vector<std::uint16_t> expected_mask(static_cast<std::size_t>((8 + cells) * rows),
                                                   0xfc00);
          const std::array<std::int32_t, 7> visible{INT32_MIN,
                                                    0,
                                                    1,
                                                    static_cast<std::int32_t>(cells / 2),
                                                    static_cast<std::int32_t>(cells - 1),
                                                    static_cast<std::int32_t>(cells),
                                                    INT32_MAX};
          for (std::size_t row = 0; row < static_cast<std::size_t>(rows); ++row) {
            std::vector<float> logical(static_cast<std::size_t>(cells));
            for (std::size_t cell = 0; cell < logical.size(); ++cell) {
              logical[cell] = std::bit_cast<float>(
                  bits[(cell + row + static_cast<std::size_t>(revision)) % bits.size()]);
              score_values[row * score_pitch + 1 + cell * 2] = logical[cell];
            }
            count_values[1 + row * 2] =
                visible[(row + static_cast<std::size_t>(revision)) % visible.size()];
            const auto ids =
                VisibleIds(logical, count_values[1 + row * 2], static_cast<std::size_t>(top));
            expected_ids.insert(expected_ids.end(), ids.begin(), ids.end());
            for (std::size_t i = 0; i < raw_bits.size(); ++i) {
              const auto v = raw_bits[(i + static_cast<std::size_t>(revision)) % raw_bits.size()];
              raw_values[row * 13 + 1 + 2 * i] = v;
              expected_mask[row * static_cast<std::size_t>(8 + cells) + i] = v;
            }
            for (const auto id : ids)
              if (id >= 0)
                expected_mask[row * static_cast<std::size_t>(8 + cells) + 8 +
                              static_cast<std::size_t>(id)] = 0;
          }
          Upload(score_root, score_values);
          Upload(count_root, count_values);
          Upload(raw_root, raw_values);
          ASSERT_TRUE(run(*launch));
          EXPECT_EQ(ReadTyped<std::int32_t>(selected), expected_ids);
          EXPECT_EQ(ReadTyped<std::uint16_t>(mask), expected_mask);
          if (!captured) {
            auto graph = launch->Capture(run);
            ASSERT_TRUE(graph);
            graphs.push_back(std::move(*graph));
            captured = graphs.size() - 1;
          }
          ASSERT_TRUE(launch->Launch(graphs[*captured]));
          EXPECT_EQ(ReadTyped<std::int32_t>(selected), expected_ids);
          EXPECT_EQ(ReadTyped<std::uint16_t>(mask), expected_mask);
        }
      }
    }
  }
}

TEST_F(GgmlLowerDsv4Gpu, SelectedMaskAcceptsDuplicateInvalidRepeatedAndCrossRowIds) {
  constexpr std::int64_t cells = 8193, rows = 3, top = 12, raw_cells = 7;
  for (const std::size_t col_stride : {0U, 1U, 2U}) {
    for (const std::size_t row_stride : {0U, 3U, 27U}) {
      SCOPED_TRACE(::testing::PrintToString(std::make_tuple(col_stride, row_stride)));
      auto* c = Context();
      auto* root = Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, 96));
      auto* ids = ggml_view_2d(c, root, top, rows, row_stride * 4, 4);
      ids->nb[0] = col_stride * 4;
      kg::TensorArena::Bind(ids, reinterpret_cast<std::uintptr_t>(root->data) + 4);
      auto* raw = Place(ggml_new_tensor_2d(c, GGML_TYPE_F16, 3, rows));
      const std::vector<std::uint16_t> raw_values{0x7d35, 0x8000, 1, 0xfd12, 0,
                                                  0xfc00, 0x7c00, 2, 0x8001};
      Upload(raw, raw_values);
      auto* mask = Place(kg::SelectedMask(c, raw, ids, raw_cells, cells));
      ASSERT_TRUE(kg::CheckSelectedMask(mask));
      const auto run = [&](kg::LaunchContext& l) { return kg::RunSelectedMask(l, mask); };
      auto graph = launch->Capture(run);
      ASSERT_TRUE(graph);
      graphs.push_back(std::move(*graph));
      for (int revision = 0; revision < 3; ++revision) {
        const std::array<std::int32_t, 12> values{-1,   INT32_MIN, INT32_MAX, 8193, 0,    0,
                                                  4088, 4089,      4090,      8185, 8192, 8192};
        std::vector<std::int32_t> source(96);
        for (std::size_t i = 0; i < source.size(); ++i)
          source[i] = values[(i + static_cast<std::size_t>(revision)) % values.size()];
        Upload(root, source);  // writes through the same root that the aliased view reads
        std::vector<std::uint16_t> expected(static_cast<std::size_t>((raw_cells + cells) * rows),
                                            0xfc00);
        for (std::size_t row = 0; row < rows; ++row) {
          std::copy_n(raw_values.begin() + static_cast<std::ptrdiff_t>(row * 3), 3,
                      expected.begin() + static_cast<std::ptrdiff_t>(row * (raw_cells + cells)));
          for (std::size_t i = 0; i < top; ++i) {
            const auto id = source[1 + row * row_stride + i * col_stride];
            if (id >= 0 && id < cells)
              expected[row * (raw_cells + cells) + raw_cells + static_cast<std::size_t>(id)] = 0;
          }
        }
        ASSERT_TRUE(launch->Launch(graphs.back()));
        EXPECT_EQ(ReadTyped<std::uint16_t>(mask), expected);
      }
    }
  }
}

// Compare the standalone native producer with a test-only copy of the original
// kernel and host launch formation. This is a native recipe equivalence test;
// the independent wider scalar and portable primitive tests remain separate.
void GgmlLowerDsv4Gpu::ScoreOracle(std::int64_t cells, std::int64_t rows, bool pitched,
                                   bool repeated, std::int64_t top) {
  SCOPED_TRACE(::testing::PrintToString(std::make_tuple(cells, rows, pitched, repeated, top)));
  constexpr std::int64_t raw_cells = 256, dim = 128, heads = 64;
  constexpr float inf = std::numeric_limits<float>::infinity();
  auto* c = Context(64);
  const std::size_t qp = pitched ? 130 : 128, kp = pitched ? 136 : 128, wp = pitched ? 66 : 64,
                    qo = pitched ? 2 : 0, ko = pitched ? 8 : 0, wo = pitched ? 1 : 0;
  const auto t = static_cast<std::size_t>(rows), n = static_cast<std::size_t>(cells);
  auto* qr =
      Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, static_cast<std::int64_t>(qp), heads, rows));
  auto* kr = Place(ggml_new_tensor_2d(c, GGML_TYPE_F16, static_cast<std::int64_t>(kp), cells));
  auto* wr = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, static_cast<std::int64_t>(wp), rows));
  auto* q = ggml_view_3d(c, qr, dim, heads, rows, repeated ? 0 : qp * 4,
                         repeated ? 0 : qp * heads * 4, qo * 4);
  auto* k = ggml_view_3d(c, kr, dim, 1, cells, kp * 2, repeated ? 0 : kp * 2, ko * 2);
  auto* k_old = ggml_view_2d(c, kr, dim, cells, repeated ? 0 : kp * 2, ko * 2);
  auto* w = ggml_view_2d(c, wr, heads, rows, repeated ? 0 : wp * 4, wo * 4);
  for (auto* view : {q, k, k_old, w})
    kg::TensorArena::Bind(view,
                          reinterpret_cast<std::uintptr_t>(view->view_src->data) + view->view_offs);
  auto* counts = Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, rows));
  auto* window = Place(ggml_new_tensor_2d(c, GGML_TYPE_F16, raw_cells, rows));
  auto* fresh = Place(kg::RoundedIndexScore(c, q, k, w, counts));
  auto* original = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, cells, rows));
  auto* fresh_ids = Place(kg::VisibleTopK(c, fresh, counts, top));
  auto* original_ids = Place(kg::VisibleTopK(c, original, counts, top));
  auto* legacy_ids = Place(kg::Dsv4LidTopK(c, q, k_old, w, counts, top));
  auto* fresh_mask = Place(kg::SelectedMask(c, window, fresh_ids, raw_cells, cells));
  auto* original_mask = Place(kg::SelectedMask(c, window, original_ids, raw_cells, cells));
  auto* legacy_mask = Place(kg::Dsv4SparseMask(c, window, legacy_ids, nullptr, raw_cells, cells));
  ASSERT_TRUE(kg::CheckRoundedIndexScore(fresh));
  ASSERT_TRUE(kg::CheckDsv4LidTopK(legacy_ids));
  ASSERT_TRUE(kg::CheckDsv4SparseMask(legacy_mask));
  ASSERT_EQ(fresh->src[3], legacy_ids->src[3]);
  ASSERT_EQ(fresh_ids->src[1], counts);
  ASSERT_EQ(original_ids->src[1], counts);
  ASSERT_EQ(fresh_mask->ne[0], raw_cells + fresh->ne[0]);
  const auto scratch = kg::PlanDsv4LidTopK(*launch, legacy_ids);
  ASSERT_TRUE(scratch);
  if (rows == 513 && cells == 65536) EXPECT_EQ(*scratch, 128U << 20);
  // Aligned widths use the same actual dense attention recipe downstream.
  // Other widths and the group boundary are checked through all score/mask bits.
  ggml_tensor* fresh_out = nullptr;
  ggml_tensor* legacy_out = nullptr;
  if (cells % 256 == 0 && rows <= 9) {
    auto* aq = Place(ggml_new_tensor_4d(c, GGML_TYPE_F32, 256, rows, 8, 1));
    auto* kv = Place(ggml_new_tensor_4d(c, GGML_TYPE_F16, 256, raw_cells + cells, 1, 1));
    Upload(aq, std::vector<float>(256 * t * 8, 0));
    std::vector<std::uint16_t> values(256 * (256 + n), 0);
    for (std::size_t cell = 0; cell < n; ++cell) {
      values[(256 + cell) * 256] = ggml_fp32_to_fp16(static_cast<float>(cell % 128 + 1) / 16);
      values[(256 + cell) * 256 + 1] = ggml_fp32_to_fp16(static_cast<float>(cell / 128 + 1) / 16);
    }
    Upload(kv, values);
    fresh_out = Place(ggml_flash_attn_ext(c, aq, kv, kv, fresh_mask, 1, 0, 0));
    legacy_out = Place(ggml_flash_attn_ext(c, aq, kv, kv, legacy_mask, 1, 0, 0));
    ggml_prec_set_acc(fresh_out, GGML_PREC_F32);
    ggml_prec_set_acc(legacy_out, GGML_PREC_F32);
    ASSERT_TRUE(kg::PlanFlashAttnMma(*launch, fresh_out));
    ASSERT_TRUE(kg::PlanFlashAttnMma(*launch, legacy_out));
  }
  const auto run = [&](kg::LaunchContext& l) -> std::expected<void, kg::KernelFailure> {
    if (auto r = kg::RunRoundedIndexScore(l, fresh); !r) return r;
    if (auto r = llmp::test_support::OriginalLidScores(l, legacy_ids, original); !r) return r;
    for (auto* ids : {fresh_ids, original_ids})
      if (auto r = kg::RunVisibleTopK(l, ids); !r) return r;
    if (auto r = kg::RunDsv4LidTopK(l, legacy_ids); !r) return r;
    for (auto* mask : {fresh_mask, original_mask})
      if (auto r = kg::RunSelectedMask(l, mask); !r) return r;
    if (auto r = kg::RunDsv4SparseMask(l, legacy_mask); !r) return r;
    if (fresh_out != nullptr) {
      if (auto r = kg::FlashAttnMma(l, fresh_out); !r) return r;
      if (auto r = kg::FlashAttnMma(l, legacy_out); !r) return r;
    }
    return {};
  };
  std::optional<std::size_t> captured;
  std::vector<std::string_view> witnesses{
      "head_order",      "fma_left",     "fma_right",    "query_round",          "visible_nan",
      "all_visible_nan", "positive_inf", "negative_inf", "visible_negative_inf", "nan_key",
      "nan_weight",      "zero_ties",    "finite_ties",  "visibility",           "unseen_inf",
      "subnormals",      "tile_edges"};
  if (rows > 9) witnesses = {"head_order", "fma_left", "visibility"};
  for (const auto name : witnesses) {
    SCOPED_TRACE(name);
    std::vector<float> query_values(qp * heads * t, 0), weights(wp * t, 0);
    std::vector<std::uint16_t> keys(kp * n, 0);
    std::vector<std::int32_t> visible(t, 2);
    const auto key = [&](std::int64_t cell, std::int64_t d, float value) {
      keys[ko + static_cast<std::size_t>(cell) * kp + static_cast<std::size_t>(d)] =
          ggml_fp32_to_fp16(value);
    };
    for (std::int64_t row = 0; row < rows; ++row) {
      const auto query = [&](std::int64_t h, std::int64_t d, float value) {
        query_values[qo +
                     (static_cast<std::size_t>(row) * heads + static_cast<std::size_t>(h)) * qp +
                     static_cast<std::size_t>(d)] = value;
      };
      const auto weight = [&](std::int64_t h, float value) {
        weights[wo + static_cast<std::size_t>(row) * wp + static_cast<std::size_t>(h)] = value;
      };
      if (name == "head_order") {
        query(0, 0, 1);
        query(8, 0, 1);
        query(1, 1, 1);
        weight(0, 0x1p24F);
        weight(1, 1);
        weight(8, -0x1p24F);
        key(0, 0, 1);
        key(0, 1, 1);
        key(1, 1, 0.5F);
      } else if (name == "fma_left" || name == "fma_right") {
        const std::int64_t a = name == "fma_left" ? 0 : 8;
        const std::int64_t b = name == "fma_left" ? 8 : 0;
        query(a, 0, 1);
        query(a, 1, 0x1p-6F);
        weight(a, 1 - 0x1p-13F);
        query(b, 2, 1);
        weight(b, -1);
        query(16, 3, 1);
        weight(16, -0x1p-27F);
        key(0, 0, 1);
        key(0, 1, 0x1p-7F);
        key(0, 2, 1);
        key(1, 3, 1);
      } else if (name == "query_round") {
        query(0, 0, 1.0001F);
        query(0, 1, 1);
        weight(0, 1);
        key(0, 1, 1);
        key(1, 0, 1);
      } else if (name == "visible_nan" || name == "all_visible_nan") {
        query(0, 0, 1);
        query(1, 0, 1);
        weight(0, 1);
        weight(1, -1);
        key(0, 0, inf);
        if (name == "all_visible_nan")
          key(1, 0, inf);
        else
          visible[static_cast<std::size_t>(row)] = 1;
      } else if (name == "positive_inf" || name == "negative_inf" ||
                 name == "visible_negative_inf") {
        query(0, 0, 1);
        weight(0, name == "positive_inf" ? 1 : -1);
        key(0, 0, inf);
        if (name == "visible_negative_inf") visible[static_cast<std::size_t>(row)] = 1;
      } else if (name == "nan_key" || name == "nan_weight") {
        query(0, 0, 1);
        weight(0, name == "nan_weight" ? std::numeric_limits<float>::quiet_NaN() : 1);
        key(0, 0, name == "nan_key" ? std::numeric_limits<float>::quiet_NaN() : 1);
        key(1, 0, 1);
      } else if (name == "zero_ties") {
        query(0, 0, 1);
        weight(0, -1);
        key(0, 0, -1);
        key(1, 0, 0);
      } else if (name == "finite_ties") {
        query(0, 0, 1);
        weight(0, 1);
        key(0, 0, 1);
        key(1, 0, 1);
        key(2, 0, 1);
        visible[static_cast<std::size_t>(row)] = 3;
      } else if (name == "visibility") {
        query(0, 0, 1);
        weight(0, 1);
        for (std::int64_t cell = 0; cell < 4; ++cell) key(cell, 0, static_cast<float>(4 - cell));
        visible[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(row % 5);
      } else if (name == "subnormals") {
        query(0, 0, 0x1p-24F);
        weight(0, 1);
        key(0, 0, 0x1p-24F);
        key(1, 0, -0x1p-24F);
      } else if (name == "tile_edges") {
        query(0, 0, 1);
        weight(0, 1);
        for (std::int64_t cell = 0; cell < cells; ++cell)
          key(cell, 0, static_cast<float>((cell * 13) % 31 - 15));
        const std::array<std::int32_t, 9> edges{0, 1, 63, 64, 65, 255, 256, 257, INT32_MAX};
        visible[static_cast<std::size_t>(row)] =
            edges[static_cast<std::size_t>(row) % edges.size()];
      } else {
        ASSERT_EQ(name, "unseen_inf");
        query(0, 0, 1);
        weight(0, 1);
        key(0, 0, 1);
        key(cells - 1, 0, inf);
        visible[static_cast<std::size_t>(row)] = 1;
      }
    }
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(raw_cells) * t, 0xfc00);
    for (std::size_t row = 0; row < t; ++row) raw[row * 256] = ggml_fp32_to_fp16(-0.25F);
    Upload(qr, query_values);
    Upload(kr, keys);
    Upload(wr, weights);
    Upload(counts, visible);
    Upload(window, raw);
    const auto check = [&] {
      const auto got = ReadTyped<std::uint32_t>(fresh);
      const auto want = ReadTyped<std::uint32_t>(original);
      ASSERT_EQ(got, want);  // includes NaN payloads, signed zero and every padded tail
      const auto ids = ReadTyped<std::int32_t>(fresh_ids);
      EXPECT_EQ(ids, ReadTyped<std::int32_t>(original_ids));
      EXPECT_EQ(ids, ReadTyped<std::int32_t>(legacy_ids));
      const auto mask = ReadTyped<std::uint16_t>(fresh_mask);
      EXPECT_EQ(mask, ReadTyped<std::uint16_t>(original_mask));
      EXPECT_EQ(mask, ReadTyped<std::uint16_t>(legacy_mask));
      // Independent CPU selection of the observed native scores checks all
      // F32 ordering/visibility; it does not define the score reduction.
      for (std::size_t row = 0; row < t; ++row) {
        std::vector<float> scores(n);
        for (std::size_t cell = 0; cell < n; ++cell) {
          scores[cell] = std::bit_cast<float>(got[row * n + cell]);
          if (static_cast<std::int64_t>(cell) >= visible[row])
            ASSERT_EQ(got[row * n + cell], 0xff800000U);
        }
        const auto wanted_ids = VisibleIds(scores, visible[row], static_cast<std::size_t>(top));
        EXPECT_EQ(
            std::vector<std::int32_t>(
                ids.begin() + static_cast<std::ptrdiff_t>(row * static_cast<std::size_t>(top)),
                ids.begin() +
                    static_cast<std::ptrdiff_t>((row + 1) * static_cast<std::size_t>(top))),
            wanted_ids);
        std::vector<std::uint16_t> wanted_mask(256 + n, 0xfc00);
        wanted_mask[0] = ggml_fp32_to_fp16(-0.25F);
        for (const auto id : wanted_ids)
          if (id >= 0) wanted_mask[256 + static_cast<std::size_t>(id)] = 0;
        ASSERT_TRUE(std::equal(wanted_mask.begin(), wanted_mask.end(),
                               mask.begin() + static_cast<std::ptrdiff_t>(row * (256 + n))));
      }
      if (fresh_out != nullptr)
        EXPECT_EQ(ReadTyped<std::uint32_t>(fresh_out), ReadTyped<std::uint32_t>(legacy_out));
    };
    const auto ran = run(*launch);
    ASSERT_TRUE(ran) << (ran ? "" : ran.error().detail);
    check();
    if (!captured) {
      auto graph = launch->Capture(run);
      ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
      graphs.push_back(std::move(*graph));
      captured = graphs.size() - 1;
    }
    // The first capture is retained across every subsequent changed operand.
    ASSERT_TRUE(launch->Launch(graphs[*captured]));
    check();
  }
}

TEST_F(GgmlLowerDsv4Gpu, NativeRoundedScoreMatchesOriginalWitnessesPitchesAndSameCapture) {
  for (std::int64_t rows = 1; rows <= 9; ++rows)
    for (const bool pitched : {false, true}) ScoreOracle(256, rows, pitched);
  for (const std::int64_t rows : {1, 3, 8, 9}) ScoreOracle(4096, rows, true);
  ScoreOracle(4096, 3, true, false, 2);
}

TEST_F(GgmlLowerDsv4Gpu, NativeRoundedScoreMatchesOriginalPartialTilesAndRepeatedViews) {
  for (const std::int64_t cells : {63, 65, 257}) ScoreOracle(cells, 9, true);
  ScoreOracle(256, 3, true, true);
}

TEST_F(GgmlLowerDsv4Gpu, NativeRoundedScoreRetainsOriginalScratchGroupBoundary) {
  ASSERT_TRUE(Finish());
  launch.reset();
  auto context = kg::LaunchContext::Create(
      0, *execution, stream, {.base = Allocate(144U << 20), .size = llmp::base::Bytes(144U << 20)});
  ASSERT_TRUE(context);
  launch = std::move(*context);
  ScoreOracle(65536, 513, true);
}

TEST_F(GgmlLowerDsv4Gpu, NativeRoundedScoreLowersWithOwnedAlignedRootsAndRestagesCapture) {
  ir::Builder b;
  const auto q = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({128, 64, 3})));
  const auto k = b.Input(ir::Dense(Encoding::kF16, ir::ShapeOf({128, 1, 256})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({64, 3})));
  const auto counts = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
  // A computed query root gets its own placement guarantee, not the
  // input's old numeric ID. This remains exact for these finite integers.
  const auto scores = b.RoundedIndexScore(b.Scale(q, 1.0F), k, w, counts);
  b.Output(scores);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  std::vector<co::StepInput> inputs;
  for (auto id : {q, k, w, counts})
    inputs.push_back({.tensor = id, .kind = co::InputKind::kFeatures});
  const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 3}}};
  auto* step = Lower(std::move(*graph), std::move(inputs), shape, nullptr, true);
  ASSERT_NE(step, nullptr);
  ASSERT_EQ(step->nodes.size(), 2U);
  ASSERT_EQ(kg::LlmpOpOf(step->nodes[1]), kg::LlmpOp::kRoundedIndexScore);
  ASSERT_EQ(step->keep[0], step->nodes[1]);
  for (int revision = 0; revision < 3; ++revision) {
    std::vector<float> queries(128 * 64 * 3, 0), weights(64 * 3, 0);
    std::vector<std::uint16_t> keys(128 * 256, 0);
    for (std::size_t row = 0; row < 3; ++row) {
      queries[row * 128 * 64] = static_cast<float>(revision + 1);
      weights[row * 64] = 2;
    }
    keys[0] = 0x3c00;
    keys[128] = 0xbc00;
    const std::vector<std::int32_t> visible{revision, 2, 0};
    Upload(Leaf(*step, q), queries);
    Upload(Leaf(*step, k), keys);
    Upload(Leaf(*step, w), weights);
    Upload(Leaf(*step, counts), visible);
    RunAndReplay(*step);
    const auto values = ReadTyped<std::uint32_t>(step->keep[0]);
    for (std::size_t row = 0; row < 3; ++row)
      for (std::size_t cell = 0; cell < 256; ++cell) {
        const auto want =
            static_cast<std::int64_t>(cell) >= visible[row] ? 0xff800000U
            : cell == 0 ? std::bit_cast<std::uint32_t>(2.0F * static_cast<float>(revision + 1))
                        : 0U;
        ASSERT_EQ(values[row * 256 + cell], want);
      }
  }
}

}  // namespace
