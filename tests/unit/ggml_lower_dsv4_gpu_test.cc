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
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "compiler_inventory.h"
#include "components/step.h"
#include "dsv4_fixture.h"
#include "engine/component_source.h"
#include "engine/execution_contract.h"
#include "engine/paged_node.h"
#include "execution/registry.h"
#include "families/dsv4.h"
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
#include "kernels/ggml/step_program.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "planner/pattern.h"
#include "planner/plan.h"
#include "platform/host_probe.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_runtime.h"
#include "spec/mappings/dsv4.h"
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
    const auto bytes = std::max<std::uint64_t>(size, 256);
    if (allocation_limit_ != 0 &&
        (allocated_ > allocation_limit_ || bytes > allocation_limit_ - allocated_)) {
      ADD_FAILURE() << "prepaid fixture allocation ceiling";
      return 0;
    }
    allocated_ += bytes;
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    allocations.push_back(p);
    return reinterpret_cast<std::uintptr_t>(p);
  }
  ggml_context* Context(std::size_t tensors = 32) {
    if (allocation_limit_ != 0 && (tensors > 128 || arenas.size() >= 16)) {
      ADD_FAILURE() << "prepaid fixture descriptor ceiling";
      return nullptr;
    }
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
                         bool aligned_roots = false, const pl::Policy& policy = {},
                         std::uint64_t allocation_limit = 0) {
    if (allocation_limit_ != 0 &&
        (lowered.size() >= 16 || graph.nodes().size() > 128 || graph.tensor_count() > 256)) {
      ADD_FAILURE() << "prepaid fixture plan ceiling";
      return nullptr;
    }
    co::StepGraph step{
        .graph = std::move(graph), .inputs = std::move(inputs), .weights = {}, .state = {}};
    std::vector<ir::TensorId> staged;
    for (const auto& input : step.inputs) staged.push_back(input.tensor);
    const auto& choices = selected == nullptr ? *registry : *selected;
    // cudaMalloc's owned allocation is independently checked at binding;
    // the planner derives each independently placed root from this guarantee.
    auto plan =
        pl::Plan(step.graph.Clone(), choices, {.device_class = "sm_121", .arch = 121}, policy,
                 {.staged = staged,
                  .written = kg::WrittenBytes,
                  .within = kg::WithinRoot,
                  .shape = &shape,
                  .activation_alignment = aligned_roots ? 256U : 0U});
    EXPECT_TRUE(plan) << (plan ? "" : plan.error().detail);
    if (!plan) return nullptr;
    if (allocation_limit_ != 0 &&
        (plan->graph.nodes().size() > 128 || plan->graph.tensor_count() > 256)) {
      ADD_FAILURE() << "prepaid fixture lowered graph ceiling";
      return nullptr;
    }
    if (allocation_limit_ != 0) {
      // Same descriptor inventory as Lowering::Run, before it allocates the
      // arena. LowerInput.context is model context, not descriptor capacity.
      if (!step.saves.empty() || plan->graph.tensor_count() > 112) {
        ADD_FAILURE() << "paid route native inventory";
        return nullptr;
      }
      std::uint64_t descriptors = plan->graph.tensor_count() + 16;
      const auto add = [&](std::uint64_t count) {
        if (count > 128 - descriptors) return false;
        descriptors += count;
        return true;
      };
      bool fits = true;
      for (const auto& applied : plan->applied)
        for (const auto replaced : applied.replaced) {
          if (replaced.v >= step.graph.nodes().size()) {
            fits = false;
            continue;
          }
          fits = add(step.graph.node(replaced).nout) && fits;
        }
      for (const auto& n : plan->graph.nodes()) {
        if (n.key == ir::Primitive(ir::Op::kRoute)) fits = add(10) && fits;
        if (n.key == ir::Primitive(ir::Op::kFill) || n.key == ir::Primitive(ir::Op::kConvert))
          fits = add(1) && fits;
        if (n.key == ir::Primitive(ir::Op::kExpertReduce)) {
          if (n.nsrc == 0 || plan->graph.tensor(n.src[0]).ne(1) > 32)
            fits = false;
          else
            fits = add(3 + 2 * plan->graph.tensor(n.src[0]).ne(1)) && fits;
        }
        if (n.key.op == ir::Op::kFused || n.key.op == ir::Op::kMatMul ||
            n.key.op == ir::Op::kWeight || n.key.op == ir::Op::kGatedDeltaNet)
          fits = add(8) && fits;
      }
      if (!fits || descriptors * ggml_tensor_overhead() + 256 * sizeof(void*) > (1U << 20)) {
        ADD_FAILURE() << "paid route native descriptor arena exceeded";
        return nullptr;
      }
    }
    if (allocation_limit != 0) {
      EXPECT_LE(plan->placement.extent, allocation_limit);
      if (plan->placement.extent > allocation_limit) return nullptr;
    }
    const kn::Addresses addresses{.weight = {},
                                  .weight_rows = {},
                                  .state = {},
                                  .save = {},
                                  .activations = Allocate(plan->placement.extent),
                                  .activation_bytes = plan->placement.extent};
    auto result = kg::LowerStep({*plan, step, shape, addresses, 65536}, choices,
                                kg::DeviceChoicesOf(*launch));
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
    if (!enabled) policy.disabled.emplace_back("state_bank_attention");
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
    auto result = kg::LowerStep({*plan, step, shape, addresses, 512, facts}, *registry,
                                kg::DeviceChoicesOf(*launch));
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
  std::uint64_t allocation_limit_ = 0, allocated_ = 0;
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

// The new control retains the complete fixture on every unknown-retirement
// path, including failures inside SetUp, Upload, lowering, run or capture.
class HcPostGpuOwner final : public GgmlLowerDsv4Gpu {
 public:
  using GgmlLowerDsv4Gpu::allocation_limit_;
  using GgmlLowerDsv4Gpu::Context;
  using GgmlLowerDsv4Gpu::Finish;
  using GgmlLowerDsv4Gpu::launch;
  using GgmlLowerDsv4Gpu::Leaf;
  using GgmlLowerDsv4Gpu::Lower;
  using GgmlLowerDsv4Gpu::Place;
  using GgmlLowerDsv4Gpu::Read;
  using GgmlLowerDsv4Gpu::ReadTyped;
  using GgmlLowerDsv4Gpu::retirement_failed;
  using GgmlLowerDsv4Gpu::RunAndReplay;
  using GgmlLowerDsv4Gpu::SetUp;
  using GgmlLowerDsv4Gpu::TearDown;
  using GgmlLowerDsv4Gpu::Upload;
  kg::LoweredStep* LowerGrouped(ir::Graph graph, ir::TensorId weight, ir::TensorId input,
                                std::uint32_t tokens, std::uint64_t weight_bytes) {
    co::StepGraph step{.graph = std::move(graph),
                       .inputs = {{.tensor = input, .kind = co::InputKind::kFeatures}},
                       .weights = {{.tensor = weight,
                                    .ref = {},
                                    .parts = {{"weight", llmp::format::PartRole::kData}}}},
                       .state = {}};
    pl::Policy policy;
    // Operator comparison only: propose this reduction explicitly, without a
    // checkpoint-quality grant or a blanket diagnostic numerical bypass.
    policy.numerical_candidates = {pl::NumericalChange::kQ8VectorReduction};
    pl::Canonicalize(policy);
    const pl::StepShape shape{.segments = {{.rows = tokens, .outputs = 0, .want_head = false}}};
    auto plan =
        pl::Plan(step.graph.Clone(), *registry, {.device_class = "sm_121", .arch = 121}, policy,
                 {.staged = std::vector<ir::TensorId>{input},
                  .written = kg::WrittenBytes,
                  .within = kg::WithinRoot,
                  .shape = &shape});
    EXPECT_TRUE(plan) << (plan ? "" : plan.error().detail);
    if (!plan) return nullptr;
    EXPECT_LE(plan->placement.extent, 2U << 20);
    if (plan->placement.extent > (2U << 20)) return nullptr;
    const auto storage = Allocate(weight_bytes);
    const kn::Addresses addresses{
        .weight =
            [storage](const ir::WeightRef& ref, llmp::format::PartRole role) {
              return ref == ir::WeightRef{} && role == llmp::format::PartRole::kData ? storage : 0;
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = Allocate(plan->placement.extent),
        .activation_bytes = plan->placement.extent};
    auto result =
        kg::LowerStep({*plan, step, shape, addresses}, *registry, kg::DeviceChoicesOf(*launch));
    EXPECT_TRUE(result) << (result ? "" : result.error().detail);
    if (!result) return nullptr;
    lowered.push_back(std::make_unique<kg::LoweredStep>(std::move(*result)));
    return lowered.back().get();
  }
  // Keep the bound recipe along with every allocation and capture until the
  // owning fixture proves retirement, even when a launch/capture assertion fails.
  void RunRouteAndReplay(kg::LoweredStep& step) {
    if (allocation_limit_ != 0) {
      ASSERT_LT(route_bounds.size(), 16U);
      ASSERT_LT(graphs.size(), 16U);
      ASSERT_LE(step.plan.steps.size(), 128U);
      for (const auto& item : step.plan.steps) ASSERT_LE(item.nodes.size(), 128U);
    }
    auto bound = kg::BoundGraph::Bind(*registry, step.plan);
    ASSERT_TRUE(bound) << (bound ? "" : bound.error().detail);
    route_bounds.push_back(std::make_unique<kg::BoundGraph>(std::move(*bound)));
    const auto* recipe = route_bounds.back().get();
    Restage(step);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_TRUE(recipe->Run(*launch));
    ASSERT_TRUE(Finish());
    std::vector<std::vector<std::uint32_t>> before;
    for (const auto* output : step.keep) {
      before.push_back(ReadRouteBits(output));
      ASSERT_FALSE(retirement_failed);
      ASSERT_FALSE(::testing::Test::HasFailure());
      ASSERT_EQ(before.back().size() * sizeof(std::uint32_t), ggml_nbytes(output));
    }
    auto captured = launch->Capture([&](kg::LaunchContext& l) { return recipe->Run(l); });
    ASSERT_TRUE(captured);
    graphs.push_back(std::move(*captured));
    captures.emplace(&step, graphs.size() - 1);
    Restage(step);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_TRUE(launch->Launch(graphs.back()));
    ASSERT_TRUE(Finish());
    for (std::size_t i = 0; i < step.keep.size(); ++i) {
      const auto after = ReadRouteBits(step.keep[i]);
      ASSERT_FALSE(retirement_failed);
      ASSERT_FALSE(::testing::Test::HasFailure());
      ASSERT_EQ(after.size(), before[i].size());
      EXPECT_EQ(after, before[i]);
    }
  }
  std::vector<std::uint32_t> ReadRouteBits(const ggml_tensor* tensor) {
    if (allocation_limit_ == 0) return ReadTyped<std::uint32_t>(tensor);
    if (!Finish()) return {};
    if (!tensor || ggml_nbytes(tensor) > 2048 || readbacks.size() >= 128) {
      ADD_FAILURE() << "paid route readback bound";
      return {};
    }
    auto& bytes = readbacks.emplace_back(ggml_nbytes(tensor) / sizeof(std::uint32_t));
    const auto copied =
        cudaMemcpy(bytes.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost);
    if (copied != cudaSuccess) retirement_failed = true;
    EXPECT_EQ(copied, cudaSuccess);
    return bytes;
  }
  std::vector<std::vector<std::uint32_t>> readbacks;
  void ReplayRoute(kg::LoweredStep& step) {
    ASSERT_FALSE(retirement_failed);
    ASSERT_FALSE(::testing::Test::HasFailure());
    const auto capture = captures.find(&step);
    ASSERT_NE(capture, captures.end());
    Restage(step);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_TRUE(launch->Launch(graphs[capture->second]));
    ASSERT_TRUE(Finish());
  }
  std::vector<std::unique_ptr<kg::BoundGraph>> route_bounds;
  std::optional<ex::Registry> primitive;

 private:
  void TestBody() override {}
};
class GgmlHcPostGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    owner = std::make_unique<HcPostGpuOwner>();
    owner->SetUp();
    if (HasFatalFailure()) return;
    auto implementations = kg::Implementations();
    std::erase_if(implementations, [](const auto& impl) { return impl.name == kg::kHcPostName; });
    auto primitive = ex::Registry::Create(std::move(implementations));
    ASSERT_TRUE(primitive);
    owner->primitive.emplace(std::move(*primitive));
  }
  void TearDown() override {
    if (!owner) return;
    if (!owner->Finish()) {
      std::ignore = owner.release();
      return;
    }
    owner->TearDown();
    if (owner->retirement_failed) std::ignore = owner.release();
  }
  std::unique_ptr<HcPostGpuOwner> owner;
};

TEST_F(GgmlHcPostGpu, UngatedExpertCombinePreservesOrderedFmaAndCapturedTupleBits) {
  // Eight complete six-slot tuples from authenticated DS40d; all six route
  // weights and shared outputs were bit-identical in all three captured arms.
  // Full actual4096+16384 replay is separately sealed, not inferred from these
  // samples. Here tile tuples, independent round-sensitive values and signed
  // zeros across every output and row; direct composite and primitive matcher.
  constexpr std::array<std::uint32_t, 6> weight_bits{0x3eba2bccU, 0x3e948d3cU, 0x3ebec02bU,
                                                     0x3e776d82U, 0x3de76a46U, 0x3df3d5ecU};
  constexpr std::array<std::array<std::uint32_t, 8>, 8> tuples{
      {{0xbca7e119U, 0xbc4df1c0U, 0xbe774f2fU, 0x3e4f0ef2U, 0xbdbcc83eU, 0x3dc4f359U, 0x3f2746d9U,
        0x3f1a2c72U},
       {0x3dbad7b6U, 0x3e2297c1U, 0x3e292285U, 0xbe0d5e0cU, 0xbb027760U, 0x3d35f29dU, 0xbe10d1e9U,
        0xbced16a8U},
       {0xbe329264U, 0x3eb37fa5U, 0xbc4f7546U, 0xbe914a2cU, 0xbc95a475U, 0xbdeee262U, 0xbf219c18U,
        0xbf2ea5a8U},
       {0x3b3b1fe0U, 0x3bc06f00U, 0x3d9b28deU, 0x3e27542cU, 0x3c6ce998U, 0xbe1ba21cU, 0xbee2aeb7U,
        0xbec70a20U},
       {0x3e1d19c4U, 0x3d83ae61U, 0xbcb2076cU, 0x3d48deb8U, 0xbc8cc977U, 0x3e80e34bU, 0xbe197daaU,
        0xbd32df66U},
       {0xbd8cc5fcU, 0xbe24b2adU, 0x3e45d332U, 0xbd0ce3a3U, 0x3e0ed6bfU, 0xbb8603d0U, 0x3e8f89a8U,
        0x3e9343afU},
       {0x3e6dc1efU, 0xbdd80800U, 0x3e06b699U, 0xbdb9cd28U, 0xbd320ca4U, 0xbcce224cU, 0xbde1f878U,
        0xbd18ea56U},
       {0x3d956ccdU, 0xbd2e4ab8U, 0xbe508f9aU, 0x3dc11a3fU, 0x3e3f540aU, 0xbdaf47bcU, 0xbe708e2aU,
        0xbe869be9U}}};
  constexpr std::uint64_t width = 4096, used = 6;
  // 12 retained lowered plans <=4MiB each, host staging/results<8MiB,
  // existing launch workspace16MiB: below80MiB total
  // payload. Every asynchronous borrower lives in the completion-aware owner.
  for (const std::uint32_t rows : {1U, 3U, 8U}) {
    std::vector<float> down(width * used * rows), weights(used * rows), shared(width * rows);
    std::vector<std::int32_t> ids(used * rows);
    for (std::size_t t = 0; t < rows; ++t) {
      for (std::size_t k = 0; k < used; ++k) {
        weights[t * used + k] = std::bit_cast<float>(weight_bits[k]);
        ids[t * used + k] = static_cast<std::int32_t>((k * 37 + t) % 256);
      }
      for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t k = 0; k < used; ++k)
          down[(t * used + k) * width + i] = std::bit_cast<float>(tuples[i % 8][k]);
        shared[t * width + i] = std::bit_cast<float>(tuples[i % 8][6]);
      }
      // Independently varied finite inputs alongside the captured tuples.
      for (std::size_t k = 0; k < used; ++k)
        down[(t * used + k) * width + 8] =
            static_cast<float>(static_cast<int>((t + 3) * (k + 7) % 19) - 9) / 7.0F;
      for (std::size_t k = 0; k < used; ++k) down[(t * used + k) * width + 9] = -0.0F;
      shared[t * width + 9] = -0.0F;
    }
    std::vector<std::uint32_t> fused(width * rows), separate(width * rows);
    for (std::size_t t = 0; t < rows; ++t)
      for (std::size_t i = 0; i < width; ++i) {
        float sum = 0.0F;
        volatile float first = down[(t * used) * width + i] * weights[t * used];
        float plain = first;
        for (std::size_t k = 0; k < used; ++k) {
          const auto x = down[(t * used + k) * width + i], w = weights[t * used + k];
          sum = std::fma(x, w, sum);
          if (k) {
            volatile float product = x * w;
            volatile float added = plain + product;
            plain = added;
          }
        }
        fused[t * width + i] = std::bit_cast<std::uint32_t>(sum + shared[t * width + i]);
        separate[t * width + i] = std::bit_cast<std::uint32_t>(plain + shared[t * width + i]);
        ASSERT_TRUE(std::isfinite(std::bit_cast<float>(fused[t * width + i])));
        if (i < 8) EXPECT_EQ(fused[t * width + i], tuples[i][7]);
      }
    EXPECT_NE(fused, separate);
    EXPECT_EQ(fused[9], 0U);  // +0 FMA accumulator versus -0 initial product.
    EXPECT_EQ(separate[9], 0x80000000U);
    for (const bool direct : {false, true})
      for (const bool native : {false, true}) {
        SCOPED_TRACE(rows);
        SCOPED_TRACE(direct);
        SCOPED_TRACE(native);
        ir::Builder b;
        auto d = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, used, rows})));
        const bool padded_ids = rows == 1;
        auto id_root =
            b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({padded_ids ? 256ULL : used, rows})));
        auto id = padded_ids ? b.View(id_root, ir::ShapeOf({used, rows}), {4, 1024, 1024, 1024}, 0)
                             : id_root;
        auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({used, rows})));
        auto sh = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})));
        const std::array sources = {d, id, w, sh};
        b.Output(direct ? b.Fused(ir::FusedKind::kMoeCombine, sources)[0]
                        : b.Add(b.ExpertReduce(d, w, id, {}), sh));
        b.Output(id);  // Keep the actual typed ID input live in both decompositions.
        auto graph = std::move(b).Finish();
        ASSERT_TRUE(graph);
        pl::Policy policy;
        if (!native) policy.disabled = {"moe_combine"};
        pl::Canonicalize(policy);
        const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
        auto* step = owner->Lower(std::move(*graph),
                                  {{.tensor = d, .kind = co::InputKind::kFeatures},
                                   {.tensor = id_root, .kind = co::InputKind::kFeatures},
                                   {.tensor = w, .kind = co::InputKind::kFeatures},
                                   {.tensor = sh, .kind = co::InputKind::kFeatures}},
                                  shape, nullptr, false, policy, 4U << 20);
        ASSERT_NE(step, nullptr);
        EXPECT_EQ(std::ranges::count_if(step->plan.steps,
                                        [](const auto& entry) {
                                          return entry.implementation == kg::kDsv4CombineName;
                                        }),
                  native ? 1 : 0);
        owner->Upload(owner->Leaf(*step, d), down);
        auto id_storage = ids;
        if (padded_ids) id_storage.resize(256, -1);
        owner->Upload(owner->Leaf(*step, id_root), id_storage);
        owner->Upload(owner->Leaf(*step, w), weights);
        owner->Upload(owner->Leaf(*step, sh), shared);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(step->keep.size(), 2U);
        if (native) {
          auto* result = step->keep[0];
          ASSERT_TRUE(kg::CheckDsv4Combine(result));
          // Refuse invalid physical bindings before any launch; restore every
          // descriptor before execution and graph capture.
          for (auto* operand : {result, result->src[0], result->src[1], result->src[2]}) {
            auto* original = operand->data;
            operand->data = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(original) + 1);
            EXPECT_FALSE(kg::CheckDsv4Combine(result));
            operand->data = original;
          }
          auto* original = result->data;
          for (auto* input : {result->src[0], result->src[1], result->src[2]}) {
            result->data = input->data;
            EXPECT_FALSE(kg::CheckDsv4Combine(result));
          }
          result->data = original;
          ASSERT_TRUE(kg::CheckDsv4Combine(result));
        }
        ASSERT_EQ(step->keep[1]->nb[0], 4U);
        ASSERT_EQ(step->keep[1]->nb[1], padded_ids ? 1024U : used * 4U);
        if (padded_ids) {
          EXPECT_EQ(step->keep[1]->view_src, owner->Leaf(*step, id_root));
          EXPECT_EQ(step->keep[1]->view_offs, 0U);
          EXPECT_EQ(ggml_nbytes(step->keep[1]), 24U);
        }
        owner->RunRouteAndReplay(*step);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(step->keep.size(), 2U);
        EXPECT_EQ(owner->ReadTyped<std::int32_t>(step->keep[1]), ids);
        EXPECT_EQ(owner->ReadTyped<std::uint32_t>(step->keep[0]), native ? fused : separate);
      }
  }
}

// 4 GiB pays the existing common hard-limit compiler inventory (peak below
// 2560 MiB), serial construction and registry/binding temporaries. Additional
// 128 MiB pays <=16 retained 128-descriptor plans, <=16 128-descriptor arenas,
// host staging/results and the checked <=32 MiB CUDA fixture allocations.
// The accounting owner remains alive until every borrower has retired.
class GgmlBiasedRouteGpu : public ::testing::Test {
 protected:
  struct Lifetime {
    llmp::engine::PagedNode node{llmp::engine::NodeSettings{}};
    std::unique_ptr<HcPostGpuOwner> gpu;
  };
  std::unique_ptr<Lifetime> paid;
  void SetUp() override {
    static_assert(sizeof(ir::Node) <= 160 && sizeof(ir::TensorDesc) <= 160);
    static_assert(sizeof(void*) == 8 && sizeof(std::string) <= 32);
    // The 16-plan fixture allowance absorbs at most 16 KiB above the prior
    // LoweredStep cap, within the existing 128 MiB retained-fixture charge.
    static_assert(sizeof(kg::PlanStep) <= 128 && sizeof(kg::LoweredStep) <= 2048);
    // The enlarged owned leaf facts add at most 128 KiB for sixteen
    // 256-leaf plans, covered by the same retained-fixture charge.
    static_assert(sizeof(kg::LoweredLeaf) <= 160 && sizeof(kn::TensorUse) <= 128);
    constexpr std::uint64_t budget = std::uint64_t{5} << 30;
    constexpr std::uint64_t charge = (std::uint64_t{4} << 30) + (128U << 20);
    ASSERT_LE(ggml_tensor_overhead(), 4096U);
    const auto available = llmp::platform::AvailableMemoryBytes();
    ASSERT_TRUE(available);
    ASSERT_GE(*available, budget + (std::uint64_t{6} << 30));
    paid = std::make_unique<Lifetime>();
    ASSERT_TRUE(paid->node.Open());
    paid->node.SetHostFloor(0);
    ASSERT_TRUE(paid->node.Start(llmp::base::Bytes(budget)));
    const auto before = paid->node.catalog().OccupancyOf(paid->node.domain());
    ASSERT_TRUE(paid->node.TryChargeHost(charge));
    const auto after = paid->node.catalog().OccupancyOf(paid->node.domain());
    ASSERT_EQ(after.Total().value() - before.Total().value(), charge);
    ASSERT_EQ(after.pinned.value() - before.pinned.value(), charge);
    paid->gpu = std::make_unique<HcPostGpuOwner>();
    paid->gpu->allocation_limit_ = 32U << 20;
    paid->gpu->SetUp();
  }
  void TearDown() override {
    if (!paid) return;
    if (paid->gpu) {
      if (!paid->gpu->Finish()) {
        std::ignore = paid.release();
        return;
      }
      paid->gpu->TearDown();
      if (paid->gpu->retirement_failed) {
        std::ignore = paid.release();
        return;
      }
      paid->gpu.reset();  // Retire every allocation/metadata borrower before its payment.
    }
    const auto stopped = paid->node.TearDown({});
    EXPECT_TRUE(stopped);
    if (!stopped) std::ignore = paid.release();
  }
};

constexpr std::array<std::uint32_t, 256> kLayer3Logits{
    0xc0855f68U, 0xc043c96cU, 0xc0c498b4U, 0xc0c05d10U, 0xc0b0fe7aU, 0xc095e434U, 0xc04b1700U,
    0xc0bc159cU, 0xc078f9b2U, 0xc06bbbf2U, 0xc0711cacU, 0xc08b19e8U, 0xc0b08550U, 0xc07b3e7aU,
    0xc0c461e7U, 0xc0425646U, 0xc063f939U, 0xc072ee99U, 0xc08556a8U, 0xc091f156U, 0xc05858d0U,
    0xc07a0b02U, 0xc0b01240U, 0xc0bc96eeU, 0xc0282ff2U, 0xc0489738U, 0xc0a1d8a9U, 0xc07267caU,
    0xc0244502U, 0xc0751d1cU, 0xc08a7f36U, 0xc09d3efaU, 0xc073b481U, 0xbfe405b3U, 0xc072ab16U,
    0xc04774b6U, 0xc08c5a3cU, 0xc065d109U, 0xc05a8d0eU, 0xc040a20aU, 0xc09c1198U, 0xc056862aU,
    0xc06be434U, 0xc03687eeU, 0xc08dbc87U, 0xc03491caU, 0xc07cd232U, 0xbfb10047U, 0xc02af8c1U,
    0xc090af9fU, 0xc07b9830U, 0xc07ec89cU, 0xc08f3c56U, 0xc044cb98U, 0xc089bf07U, 0xc08813d0U,
    0xc05c6eabU, 0xc054b034U, 0xc0903cdbU, 0xc087def0U, 0xc0801347U, 0xc0d5233aU, 0xbfcefab8U,
    0xc0562b43U, 0xc07881c9U, 0xc0563fb8U, 0xc00c30baU, 0xc08bc590U, 0xc0835026U, 0xc04c4ab8U,
    0xc0729ac2U, 0xc038af3aU, 0xc080571eU, 0xc067eb4aU, 0xc015f203U, 0xc034acdcU, 0xc07e8b3bU,
    0xc0521b59U, 0xc023995eU, 0xc020bf21U, 0xbfe092beU, 0xc058d454U, 0xc0a376beU, 0xc06ec277U,
    0xc0e15e27U, 0xc04bb044U, 0xc077ad59U, 0xc0488713U, 0xc08a1c6aU, 0xc03c61c6U, 0xc05fb9b8U,
    0xc064838fU, 0xc067f606U, 0xc02a276dU, 0xc02a48deU, 0xc0817d6dU, 0xc095e75eU, 0xc0742085U,
    0xc0278e32U, 0xc08edd56U, 0xc04aa450U, 0xc03f0588U, 0xc0746b68U, 0xc0db8f55U, 0x3df2e4c2U,
    0xc0839c84U, 0xc04043bdU, 0xc05f81b0U, 0xc080c0e6U, 0xc07070a8U, 0xc054d084U, 0xc03bcefaU,
    0xc04b477aU, 0xc096dd30U, 0xbfa14eadU, 0xc02c9140U, 0xc0b348e1U, 0xc022c7ecU, 0xc04598f8U,
    0xc031ad64U, 0xc071fa13U, 0xc0affe1aU, 0xc0644079U, 0xc06abbf6U, 0xc0507fbbU, 0xc0a135c9U,
    0xc080a660U, 0xc0a7e2a4U, 0xc034c3c1U, 0xc00ed07fU, 0xc08974baU, 0xc0581d97U, 0xc0322cd2U,
    0xc0502c3cU, 0xc06a1d4cU, 0xc02cad78U, 0xc06211e0U, 0xc062581aU, 0xc072a9a3U, 0xc03eb776U,
    0xc06dccb0U, 0xc09e77c2U, 0xc08bb947U, 0xc04b358aU, 0xc0b10c92U, 0xc09427d6U, 0xc05be9baU,
    0xc03fc17aU, 0xc04eb1bdU, 0xc088a70aU, 0xc08c5235U, 0xc0278711U, 0xc0a19786U, 0xc080c8feU,
    0xc04f9764U, 0xc086a84eU, 0xc06ba774U, 0xc0aa3e2cU, 0xc08c45f6U, 0xc0805452U, 0xc03c07aeU,
    0xc09a4783U, 0xc09db1c5U, 0xc0832465U, 0xc07cca66U, 0xc073faecU, 0xc04ce453U, 0xc08455b4U,
    0xc04e410bU, 0xc0887295U, 0xc09578c5U, 0xc09d0d89U, 0xc097ce29U, 0xbfee13b6U, 0xc04910a5U,
    0xc04bcbd8U, 0xc086e215U, 0xc047408eU, 0xc042c702U, 0xc0a06b28U, 0xc08155a6U, 0xc0866d8fU,
    0xc04a58f0U, 0xc0854ec5U, 0xc08213caU, 0xc06b1ff8U, 0xc04b0d55U, 0xc0843a32U, 0xc07e6382U,
    0xc089ef58U, 0xc07d7d58U, 0xc0bfb331U, 0xc07f8539U, 0xc060a095U, 0xc092d232U, 0xc050ad13U,
    0xc08497bfU, 0xc0a4b4e0U, 0xc05a15c0U, 0xc0548873U, 0xc0576b32U, 0xc0bda70cU, 0xc09a3b0bU,
    0xc07e85f4U, 0xbfe86678U, 0xc0f4dd6cU, 0xc049a401U, 0xc08b2e83U, 0xc0243545U, 0xc02ec863U,
    0xc084f264U, 0xc04698acU, 0xc03037c0U, 0xc070c7f4U, 0xc06f7030U, 0xc028c857U, 0xc0a02d6cU,
    0xc0659ea8U, 0xc06bbb6fU, 0xc08dda8eU, 0xc090b5e2U, 0xbff4ed7fU, 0xc07088d8U, 0xc0db1c88U,
    0xc049cefaU, 0xc09d3a3cU, 0xc05df620U, 0xc08eda56U, 0xc06d7769U, 0xc020007aU, 0xc06c37e6U,
    0xc04ac914U, 0xc0872dafU, 0xc039918aU, 0xc0b48773U, 0xbf4d2a25U, 0xc06ab772U, 0xc09f106cU,
    0xc0652b88U, 0xc061b778U, 0xc06ccd54U, 0xc0849a46U, 0xc080ca05U, 0xc0398449U, 0xc098ffd2U,
    0xc066f0f2U, 0xc074d4bfU, 0xc087656bU, 0xc05a4e8aU, 0xc047cec7U, 0xc08d9a3fU, 0xc08efba0U,
    0xc049127aU, 0xc06c0b5eU, 0xc013bbbeU, 0xc0a9a7a8U};

constexpr std::array<std::uint32_t, 256> kLayer3Bias{
    0x410ffb98U, 0x410fbf29U, 0x410febe1U, 0x410fa9dbU, 0x410e6a13U, 0x410ff7a4U, 0x410ffb46U,
    0x410e1226U, 0x410fe1beU, 0x410fa893U, 0x410fe686U, 0x410fd7b8U, 0x410ffeeeU, 0x41101328U,
    0x410fb5efU, 0x410fcaa4U, 0x41101db8U, 0x410fa114U, 0x4110122dU, 0x410feb12U, 0x41101356U,
    0x4110035eU, 0x410ff7ceU, 0x410eb198U, 0x410ff7b3U, 0x410fe79bU, 0x410faa3cU, 0x4110076dU,
    0x410fef4bU, 0x41102b32U, 0x41103669U, 0x410fe28eU, 0x410feb9bU, 0x41100b8aU, 0x4110000eU,
    0x411006a2U, 0x410fee3cU, 0x41100a4fU, 0x410ff2ccU, 0x410f599cU, 0x410f3bb4U, 0x41100770U,
    0x410ffed7U, 0x410ff2f4U, 0x410fd9a2U, 0x41100b6cU, 0x410fd78cU, 0x410ff2abU, 0x41103ec3U,
    0x410ffb33U, 0x410fa06dU, 0x41100fa8U, 0x410ff6f2U, 0x41101a0fU, 0x41102260U, 0x41101c96U,
    0x41101e44U, 0x41100afdU, 0x4106565aU, 0x410fbcfdU, 0x41100a49U, 0x410f4f5dU, 0x41100a4eU,
    0x411002f0U, 0x410ec7b3U, 0x41100702U, 0x410ffe9dU, 0x41100743U, 0x410f9480U, 0x41101a87U,
    0x410feea0U, 0x410fdda2U, 0x411002bbU, 0x41101b09U, 0x411006f6U, 0x410ffb60U, 0x410fc7a0U,
    0x411026d2U, 0x410ff764U, 0x410ffea4U, 0x41100f28U, 0x410fc3d3U, 0x410ec016U, 0x410f9d97U,
    0x410d9d32U, 0x410ff7caU, 0x410fe764U, 0x410ffe34U, 0x410fdeefU, 0x411012b0U, 0x41100ef5U,
    0x411002e9U, 0x410fff3fU, 0x41100badU, 0x41100649U, 0x410fcbf7U, 0x411002d0U, 0x411015d1U,
    0x410ff29eU, 0x410ffb0cU, 0x410fead4U, 0x410fe882U, 0x410fff6fU, 0x410e59d2U, 0x410fff65U,
    0x410ff781U, 0x41102233U, 0x41101af6U, 0x4110166bU, 0x41100703U, 0x4110026eU, 0x410ffe84U,
    0x410ff37fU, 0x410fea6fU, 0x410fff5cU, 0x41100381U, 0x410e96bcU, 0x41100354U, 0x410ff336U,
    0x410ffba4U, 0x4110174bU, 0x410f7086U, 0x411002a7U, 0x411006c8U, 0x41101a35U, 0x41100295U,
    0x41101a33U, 0x410ed833U, 0x41100331U, 0x410fef5fU, 0x410f9e31U, 0x410ffb8cU, 0x410ff794U,
    0x41101fa1U, 0x411015caU, 0x410ff2b7U, 0x410fd179U, 0x410fffefU, 0x410ffebaU, 0x410f83dcU,
    0x411022cbU, 0x41100bc1U, 0x410ffeceU, 0x410fd35fU, 0x410fef34U, 0x410fc9bdU, 0x41100ed4U,
    0x410f540aU, 0x410ffa6eU, 0x41101b02U, 0x411012b9U, 0x4110063bU, 0x4110134dU, 0x411013ceU,
    0x410fce3dU, 0x410ff2f5U, 0x41101a1eU, 0x4110266fU, 0x410e64cfU, 0x41100f65U, 0x410fe343U,
    0x410feecdU, 0x41101b53U, 0x410fb1a8U, 0x410fa209U, 0x410ffad2U, 0x411006f6U, 0x410f8d2cU,
    0x410ff663U, 0x4110065cU, 0x41100ee7U, 0x410fff97U, 0x41100a2eU, 0x410fb143U, 0x410ffebdU,
    0x410ff3dbU, 0x41101ddaU, 0x41100f36U, 0x410ffa7fU, 0x41102e68U, 0x41104cefU, 0x410f73c3U,
    0x41102b52U, 0x410fdb3aU, 0x410f90b8U, 0x410f9f05U, 0x410ffe6bU, 0x410ffacdU, 0x410fed18U,
    0x410feef7U, 0x411028e6U, 0x410f1af2U, 0x410ff69bU, 0x41101eccU, 0x411007d7U, 0x410ffea3U,
    0x411023d3U, 0x4110182aU, 0x410fef9dU, 0x411007b9U, 0x410ffb62U, 0x410fda98U, 0x410fe878U,
    0x41102b3dU, 0x410ff6afU, 0x410cccbaU, 0x411006a3U, 0x4110267eU, 0x410fdf83U, 0x410feeddU,
    0x41100a75U, 0x410d75bdU, 0x410fef83U, 0x41100f88U, 0x410ff3baU, 0x410fe76aU, 0x4110441fU,
    0x411009c0U, 0x41101f1aU, 0x41102741U, 0x410fff3fU, 0x410ff6b1U, 0x410eac47U, 0x410f84b4U,
    0x410ffaa8U, 0x41101a9fU, 0x41100f5dU, 0x411013c5U, 0x41101af5U, 0x410ffb66U, 0x410ffe89U,
    0x41100e3aU, 0x410fd7bcU, 0x410fffb7U, 0x410f51c7U, 0x410fea38U, 0x41101b5aU, 0x410f8aadU,
    0x410ff365U, 0x410fe710U, 0x410ec921U, 0x410ffc14U, 0x410fff58U, 0x410ff22cU, 0x4110026fU,
    0x410ee19dU, 0x41101650U, 0x410ff304U, 0x41100bb7U, 0x41101e5fU, 0x410ffeaeU, 0x411031ffU,
    0x410feaafU, 0x410fe715U, 0x4110026fU, 0x410ffe6bU};

constexpr std::array<std::uint32_t, 6> kLayer3Ids{0x00000068U, 0x000000ebU, 0x00000072U,
                                                  0x0000002fU, 0x0000003eU, 0x00000050U};

constexpr std::array<std::uint32_t, 6> kLayer3Weights{0x3ecba768U, 0x3e8ec504U, 0x3e6a5856U,
                                                      0x3e5de4f6U, 0x3e479160U, 0x3e3b587bU};

constexpr std::array<std::uint32_t, 6> kLayer3WeightsPrimitive{
    0x3ecba768U, 0x3e8ec504U, 0x3e6a5856U, 0x3e5de4f6U, 0x3e479160U, 0x3e3b587bU};

constexpr std::array<std::uint32_t, 256> kLayer8Logits{
    0xc083e7ccU, 0xbf7ad53aU, 0xc04b88e6U, 0xc07cdc98U, 0xc0363f9aU, 0xc005656aU, 0xc0468230U,
    0xc0292beeU, 0xc02d8e58U, 0xc033256eU, 0xc05113f7U, 0xc03da4faU, 0xc052d2f6U, 0xc040600aU,
    0xc092bf67U, 0xc058ae8bU, 0xc0414e81U, 0xc02ade47U, 0xc0307d3fU, 0xc04e1a0aU, 0xbffa1886U,
    0xc00cfb2aU, 0xc029f74cU, 0xc0006331U, 0xc080a44cU, 0xc03d0524U, 0xc094ace7U, 0xc065a714U,
    0xc003d1e8U, 0xc05115bbU, 0xc0565bf0U, 0xc007eedbU, 0xc04feaa1U, 0xc085686cU, 0xbfd57b5cU,
    0xc059bee5U, 0xc02c6e50U, 0xc04b2558U, 0xc0495100U, 0xc036064eU, 0xc0190e1dU, 0xc05ba16cU,
    0xc0175c70U, 0xc03d3b98U, 0xc051d2d0U, 0xc038b4c8U, 0xc0109094U, 0xc011a08bU, 0xc05fa07eU,
    0xc08e79ccU, 0xc0357358U, 0xc035b961U, 0xc02a6913U, 0xbef1797aU, 0xc0526898U, 0xc01dcedeU,
    0xc03ded8cU, 0xc0181f88U, 0xc0499260U, 0xc050ab34U, 0xc05ca7a7U, 0xbfd39d32U, 0xc094eceaU,
    0xc01e9fbeU, 0xc021dcf6U, 0xc05965cfU, 0xc035e4f0U, 0xc0b399c8U, 0xc032e725U, 0xc05118a3U,
    0xc0423569U, 0xc0618736U, 0xc0120762U, 0x3f363f76U, 0xc0645f41U, 0xc08965b6U, 0xc01608acU,
    0xc0046d0eU, 0xc0700abdU, 0xc020d8f6U, 0xc095b9c9U, 0xc057be32U, 0xc09421c8U, 0x3e8f904bU,
    0xc01bf4eaU, 0xc0214071U, 0xc04e8610U, 0xc04d0b38U, 0xc00bedfcU, 0xc03bb83eU, 0xc061f76cU,
    0xc0843d7eU, 0xc003dd54U, 0xc03aa49eU, 0xbf980525U, 0xc07433fdU, 0xc060048cU, 0xc00d129bU,
    0xc001a93aU, 0xc03de8a8U, 0xbfd47f3aU, 0xc03f7b4cU, 0xc05dc3aeU, 0xc081c1a6U, 0xc018c6afU,
    0xc019748eU, 0xc0134864U, 0xc05b3cf8U, 0xc0416daeU, 0xc0938358U, 0x3fb815efU, 0xc04d48acU,
    0xbffa34c7U, 0xc003f4eeU, 0xc036490eU, 0xc00444ecU, 0xc05c22daU, 0xc04a4a24U, 0xc03d9b16U,
    0xc0ad1d82U, 0xc06a7d7aU, 0xc014de99U, 0xc01d015eU, 0xc014abb4U, 0xbfc6ad36U, 0xc0808400U,
    0xc00371a2U, 0xc0aa12f4U, 0xc02c5d6aU, 0xc04f973cU, 0xc02cae09U, 0xc0856f36U, 0xc0294d28U,
    0xc04b8a34U, 0xc054baeaU, 0xc0408232U, 0xc09b5070U, 0xc021b328U, 0xc03d47ecU, 0xc08ff002U,
    0xc0261c80U, 0xc049ea3cU, 0xc05c52c4U, 0xc07269e7U, 0xc09cede2U, 0xc054327fU, 0xc0067e15U,
    0xc0248a16U, 0xc0174eb9U, 0xc05d8d51U, 0xc00fa2bbU, 0xc075140fU, 0xc0111a08U, 0xc0b2c914U,
    0xc04bacc1U, 0xc054150cU, 0xc09e741dU, 0xc01d97b6U, 0xc044782aU, 0xc03703f7U, 0xc062bf4aU,
    0xc086afebU, 0xbf394746U, 0xc0230083U, 0xc037d60cU, 0xc0407a82U, 0xc0290425U, 0xc041bdecU,
    0xc0773d48U, 0xc064ada3U, 0xc02bfb32U, 0xc0539720U, 0xbffbdcfeU, 0xc011053aU, 0xbff6ad19U,
    0xc085e9c6U, 0xc01d9d0eU, 0xc01e7032U, 0xc073b060U, 0xc0270132U, 0xc029251bU, 0xc05c64ffU,
    0xc02aa098U, 0xc06b2a22U, 0xc0527dc0U, 0xc034ee6fU, 0xc0713920U, 0xc0208e10U, 0xc06102a6U,
    0xc00b1fc6U, 0xc047503cU, 0xc0510789U, 0xc0236ae4U, 0xbfbbfb1eU, 0xc07a5988U, 0xc04aac2eU,
    0xc04e196aU, 0xc03cccf0U, 0xc06a5214U, 0xc01e850cU, 0x3f862b5aU, 0xc0143782U, 0xc0349c86U,
    0xc03231efU, 0xc042f9e2U, 0xbfe9921cU, 0xc074989aU, 0xc024073cU, 0xc06d2179U, 0xc0161d0fU,
    0xc03b2738U, 0xbfc4160eU, 0xc0486fccU, 0xc088f2e8U, 0xc02cb1aeU, 0xc0116149U, 0xc0482394U,
    0xc05cf30fU, 0xc05ce946U, 0xc053069eU, 0xc046ec17U, 0xc094b0feU, 0xc0373a98U, 0xc07a0740U,
    0xc08f6d53U, 0xbfdabd87U, 0xc08a896eU, 0xbfc48ffeU, 0xc0327ca4U, 0xc011167fU, 0xbfdbf09dU,
    0xc064b49dU, 0xc00614edU, 0xc04d85feU, 0xc0114454U, 0xc0277fa5U, 0xc068e8d5U, 0xc0030320U,
    0xc0362219U, 0xc00c90afU, 0xc05a417bU, 0xbff06c94U, 0xc015ed8bU, 0xc095306eU, 0xc063160aU,
    0xbfc23384U, 0xc0a4238dU, 0xc058c22cU, 0xc0a0656fU, 0xc02fb33aU, 0xc01fd4adU, 0xbfcfaf12U,
    0xc0519382U, 0xc0220c22U, 0xc0b52908U, 0xc02f20ecU};

constexpr std::array<std::uint32_t, 256> kLayer8Bias{
    0x411b9941U, 0x411b5ff3U, 0x411b691cU, 0x411ba5b5U, 0x411ba28bU, 0x411b860dU, 0x411ba417U,
    0x411b7d0fU, 0x411b959cU, 0x411b9993U, 0x411b999dU, 0x411b8e8eU, 0x411b4f2eU, 0x411b4373U,
    0x411bc245U, 0x411baa46U, 0x411bb9baU, 0x411b0a99U, 0x411b9098U, 0x411b961fU, 0x411b6428U,
    0x411acc72U, 0x411ba934U, 0x411ba12dU, 0x411afe2fU, 0x411b2e11U, 0x411b9b65U, 0x411b7e8eU,
    0x411bbdfcU, 0x411bd54dU, 0x411a8635U, 0x411b7599U, 0x411b7557U, 0x411b852aU, 0x411b6c7eU,
    0x411b6c75U, 0x411b6028U, 0x411b91e9U, 0x411b99b9U, 0x411b84e9U, 0x411b91dbU, 0x411b8d61U,
    0x411b891fU, 0x411bc576U, 0x411b7309U, 0x411b8c3aU, 0x411b9c57U, 0x411ba8f7U, 0x411b9aa9U,
    0x411a5752U, 0x411ba591U, 0x411ba4c0U, 0x411b8cfbU, 0x411b941eU, 0x411bcdb5U, 0x411ba2a6U,
    0x411b50b9U, 0x411b65c8U, 0x411ba9a8U, 0x411ba20cU, 0x411b9597U, 0x411b7cb6U, 0x411bcad4U,
    0x411b5426U, 0x411ba60eU, 0x411b4bbcU, 0x411b84e1U, 0x41185c09U, 0x411b6b38U, 0x411b623bU,
    0x411b9a12U, 0x411ba94cU, 0x411b4b75U, 0x411b774bU, 0x411b846dU, 0x411ba209U, 0x411b9f7fU,
    0x411b903eU, 0x411b610dU, 0x411ba296U, 0x411b144cU, 0x411b9da0U, 0x411ba29fU, 0x411ba57eU,
    0x411b2e54U, 0x411af564U, 0x411ba284U, 0x411b81baU, 0x411ba21eU, 0x411ba65eU, 0x411b30bbU,
    0x411b98c7U, 0x411b7f8cU, 0x411b995bU, 0x411b8a28U, 0x411b74c8U, 0x411b8192U, 0x411ba0b4U,
    0x411b3fb1U, 0x411a66a3U, 0x411b8cd9U, 0x411ba477U, 0x411b8905U, 0x411b48a5U, 0x411b3618U,
    0x411b9ffbU, 0x411b8cc9U, 0x411b9d64U, 0x411bc43cU, 0x411b90e7U, 0x411b6c9dU, 0x411b8047U,
    0x411bae9fU, 0x411b8871U, 0x411b8191U, 0x411b7d48U, 0x411bb177U, 0x411bb6a1U, 0x411bb129U,
    0x411b8998U, 0x411ba0a4U, 0x411b4dedU, 0x411b9261U, 0x411b8ccfU, 0x411b6957U, 0x411bb20eU,
    0x411b8097U, 0x411ae2afU, 0x411b6c93U, 0x411ba8a3U, 0x411b94f2U, 0x411bac89U, 0x411b9c7dU,
    0x411b9db9U, 0x411b9142U, 0x411b6950U, 0x411b81c1U, 0x411b6fdeU, 0x411b951dU, 0x411b403bU,
    0x411b575aU, 0x411a668fU, 0x411ba10eU, 0x411b693bU, 0x411b9d77U, 0x411b6c9cU, 0x411b9585U,
    0x411b6aabU, 0x411b761dU, 0x411b9e17U, 0x411b5698U, 0x411ba5f7U, 0x411b8844U, 0x41147bbcU,
    0x411bc9cdU, 0x411b5dd7U, 0x411b9d5aU, 0x411b9066U, 0x411b81b4U, 0x411b7d46U, 0x411b9e8dU,
    0x411b948bU, 0x411b8a46U, 0x411b9dc4U, 0x411b911bU, 0x411b7f77U, 0x411b65f9U, 0x411b8abbU,
    0x411ad023U, 0x411af266U, 0x411ba193U, 0x411b94cfU, 0x411b859aU, 0x411b83a5U, 0x411b9783U,
    0x411b8ca6U, 0x411b7538U, 0x411b5c27U, 0x411b72a4U, 0x411b6836U, 0x411b910fU, 0x411b19caU,
    0x411b7047U, 0x411b4c5dU, 0x411b6c80U, 0x411b4ed9U, 0x411bbaeeU, 0x411b59f3U, 0x411ba23bU,
    0x411b6f66U, 0x411ba8a0U, 0x411b9dc1U, 0x411b4428U, 0x411ba1fbU, 0x411b9623U, 0x411b6491U,
    0x411b98baU, 0x411b8934U, 0x411b8ea4U, 0x411ba251U, 0x41193b33U, 0x411b5b9dU, 0x411b8c95U,
    0x411b9a35U, 0x411bc6a3U, 0x411ba230U, 0x411b99e5U, 0x411b7544U, 0x411b3222U, 0x411b85e5U,
    0x411ba5d3U, 0x411b896fU, 0x411b9528U, 0x411a80d1U, 0x411b685bU, 0x411b9786U, 0x411b9d02U,
    0x411ba089U, 0x411b5154U, 0x411bb43bU, 0x411b26f2U, 0x411b6ca8U, 0x411ba11fU, 0x411b9d12U,
    0x411bd692U, 0x411b7dd1U, 0x411acf1cU, 0x411b9101U, 0x411baaa8U, 0x411b6842U, 0x411b3b94U,
    0x411b9a33U, 0x411bbd6bU, 0x411b917aU, 0x411b9cceU, 0x411b87edU, 0x411b8e1bU, 0x411b5957U,
    0x411b9583U, 0x411b97e6U, 0x411b95c0U, 0x411b8deaU, 0x411ba674U, 0x411ba28bU, 0x411b96b7U,
    0x411ba4eeU, 0x411b7f0aU, 0x411b992eU, 0x4119ebceU, 0x411b968dU, 0x411ba1c1U, 0x411b8536U,
    0x411b88fcU, 0x411b6b50U, 0x41198a7aU, 0x411b9583U};

constexpr std::array<std::uint32_t, 6> kLayer8Ids{0x0000006eU, 0x00000049U, 0x000000c8U,
                                                  0x00000053U, 0x00000035U, 0x000000a2U};

constexpr std::array<std::uint32_t, 6> kLayer8Weights{0x3eabd037U, 0x3e8cf20cU, 0x3e9b488dU,
                                                      0x3e758d5cU, 0x3e3a36bbU, 0x3e28263fU};

constexpr std::array<std::uint32_t, 6> kLayer8WeightsPrimitive{
    0x3eabd038U, 0x3e8cf20dU, 0x3e9b488eU, 0x3e758d5eU, 0x3e3a36bcU, 0x3e282640U};

TEST_F(GgmlBiasedRouteGpu, CompleteCapturedAndMultirowNativeRouteMatchesLegacyAndChangedReplay) {
  auto& owner = *paid->gpu;
  struct Case {
    std::uint32_t rows, used;
    bool normalize;
    float scale;
    unsigned captured;
  };
  constexpr std::array cases{Case{1, 6, true, 1.5F, 3},   Case{1, 6, true, 1.5F, 8},
                             Case{2, 1, true, 1.0F, 0},   Case{3, 6, false, 0.75F, 0},
                             Case{8, 32, true, 1.5F, 0},  Case{8, 6, true, 1.5F, 0},
                             Case{3, 32, false, 0.75F, 0}};
  for (const auto scenario : cases) {
    SCOPED_TRACE(scenario.captured);
    SCOPED_TRACE(scenario.rows);
    SCOPED_TRACE(scenario.used);
    auto* c = owner.Context();
    ASSERT_NE(c, nullptr);
    auto* logits = owner.Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, scenario.rows));
    auto* bias = owner.Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, 256));
    auto* legacy = owner.Place(kg::Dsv4Route(c, logits, bias, nullptr, nullptr, scenario.used,
                                             scenario.normalize, 0x1p-14F, scenario.scale));
    std::vector<float> x(256U * scenario.rows), bias_values(256);
    const auto fill = [&](unsigned revision) {
      for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(static_cast<int>((i * 71 + revision * 23) % 191) - 95) / 8.0F;
      for (std::size_t i = 0; i < bias_values.size(); ++i)
        bias_values[i] =
            static_cast<float>(static_cast<int>((i * 13 + revision * 17) % 31) - 15) / 64.0F;
      x[0] = -0.0F;
      x[1] = 0.0F;
      x[2] = 20.0F;
      x[3] = std::nextafter(20.0F, 21.0F);
      if (scenario.used == 1) {
        std::fill_n(x.begin(), 256, -100.0F);
        std::ranges::fill(bias_values, 0.0F);
        x[revision] = -0.0F;
        x[revision + 1] = 0.0F;  // Equal score: lower expert index wins.
      }
      if (revision == 0 && scenario.captured) {
        const auto& xb = scenario.captured == 3 ? kLayer3Logits : kLayer8Logits;
        const auto& bb = scenario.captured == 3 ? kLayer3Bias : kLayer8Bias;
        for (std::size_t i = 0; i < 256; ++i) {
          x[i] = std::bit_cast<float>(xb[i]);
          bias_values[i] = std::bit_cast<float>(bb[i]);
        }
      }
    };
    for (bool native : {true, false}) {
      if (!native && scenario.captured == 0) continue;
      ir::Builder b;
      const auto input = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256, scenario.rows})));
      const auto bias_id = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256})));
      const auto result = b.Route(input, bias_id, {}, {},
                                  {.kind = ir::RouteKind::kSqrtSoftplusBias,
                                   .normalize = scenario.normalize,
                                   .experts = 256,
                                   .used = scenario.used,
                                   .scale = scenario.scale});
      b.Output(result[0]);
      b.Output(result[1]);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph);
      pl::Policy policy;
      if (!native) policy.disabled = {"biased_route"};
      pl::Canonicalize(policy);
      const pl::StepShape shape{
          .segments = {{.rows = scenario.rows, .outputs = 0, .want_head = false}}};
      auto* step = owner.Lower(std::move(*graph),
                               {{.tensor = input, .kind = co::InputKind::kFeatures},
                                {.tensor = bias_id, .kind = co::InputKind::kFeatures}},
                               shape, nullptr, false, policy, 65536);
      ASSERT_NE(step, nullptr);
      ASSERT_EQ(std::ranges::count_if(
                    step->plan.steps,
                    [](const auto& s) { return s.implementation == kg::kDsv4RouteName; }),
                native ? 1 : 0);
      for (unsigned revision = 0; revision < (native ? 2U : 1U); ++revision) {
        fill(revision);
        owner.Upload(logits, x);
        owner.Upload(bias, bias_values);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_TRUE(kg::RunDsv4Route(*owner.launch, legacy));
        ASSERT_TRUE(owner.Finish());
        const auto blob = owner.ReadRouteBits(legacy);
        ASSERT_FALSE(owner.retirement_failed);
        ASSERT_FALSE(HasFailure());
        ASSERT_EQ(blob.size(), 2U * scenario.used * scenario.rows);
        std::vector<std::uint32_t> ids(scenario.used * scenario.rows), weights(ids.size());
        for (std::size_t row = 0; row < scenario.rows; ++row)
          for (std::size_t slot = 0; slot < scenario.used; ++slot) {
            ids[row * scenario.used + slot] = blob[row * scenario.used * 2 + slot];
            weights[row * scenario.used + slot] =
                blob[row * scenario.used * 2 + scenario.used + slot];
          }
        if (scenario.used == 1) EXPECT_EQ(ids[0], revision);
        owner.Upload(owner.Leaf(*step, input), x);
        owner.Upload(owner.Leaf(*step, bias_id), bias_values);
        ASSERT_FALSE(HasFatalFailure());
        if (revision == 0)
          owner.RunRouteAndReplay(*step);
        else
          owner.ReplayRoute(*step);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(step->keep.size(), 2U);
        const auto actual_ids = owner.ReadRouteBits(step->keep[0]);
        ASSERT_FALSE(owner.retirement_failed);
        ASSERT_FALSE(HasFailure());
        ASSERT_EQ(actual_ids.size(), ids.size());
        EXPECT_EQ(actual_ids, ids);
        const auto actual = owner.ReadRouteBits(step->keep[1]);
        ASSERT_FALSE(owner.retirement_failed);
        ASSERT_FALSE(HasFailure());
        ASSERT_EQ(actual.size(), weights.size());
        if (native) EXPECT_EQ(actual, weights);
        if (revision == 0 && scenario.captured) {
          const auto& expected_ids = scenario.captured == 3 ? kLayer3Ids : kLayer8Ids;
          const auto& expected = scenario.captured == 3 ? kLayer3Weights : kLayer8Weights;
          const auto& primitive =
              scenario.captured == 3 ? kLayer3WeightsPrimitive : kLayer8WeightsPrimitive;
          EXPECT_TRUE(std::ranges::equal(ids, expected_ids));
          EXPECT_TRUE(std::ranges::equal(weights, expected));
          EXPECT_TRUE(std::ranges::equal(actual, native ? expected : primitive));
          if (!native && scenario.captured == 8) EXPECT_NE(actual, weights);
        }
      }
    }
  }
}

TEST_F(GgmlHcPostGpu, SelectedHashRouteMatchesAllLegacyBitsAndSeparatesPrimitiveRecipe) {
  // Complete authenticated DS39d router output (1024 bytes), selected IDs
  // and both frozen weight recipes. All three captured arms agree on inputs.
  // These small operands are a regression fixture, not model qualification.
  constexpr std::array<std::uint32_t, 256> captured{
      0xbf2aa426U, 0xbfc26098U, 0xc04e9de3U, 0x3f85ebc7U, 0xbea053b7U, 0xbfa7d476U, 0x3fe49f08U,
      0xbeee033fU, 0x3ead70e1U, 0xc0167f74U, 0xbf8929daU, 0xbf8a8aacU, 0x3ede2f80U, 0xbec1ce92U,
      0x3e9eebf1U, 0xc00fc868U, 0xbf8a629aU, 0x3d193384U, 0xbf81bed2U, 0xbfff041aU, 0xc008e3b8U,
      0x3ebbdca7U, 0x3eaf8309U, 0xbf666ec9U, 0xc0570230U, 0xbfb0b5c4U, 0xc07441deU, 0xbde359c4U,
      0x3efee02dU, 0xc03dcaaeU, 0xbf52ad2eU, 0xbf218c50U, 0xbf2e7a9eU, 0x3def3b5dU, 0x3e6e1656U,
      0xbfe46620U, 0xbfd53809U, 0xc00c92acU, 0xbf7df49eU, 0xbfb91d6cU, 0x3f5c2be9U, 0xbeeba8faU,
      0xbe770f73U, 0xbd16dac2U, 0xbf480cd0U, 0x400ec4d3U, 0x3eb3f948U, 0xbedfef70U, 0xbffa2465U,
      0xbfac3c36U, 0xc00137cdU, 0x3e8a3a7bU, 0x3f5b47d0U, 0x3fabe4ecU, 0xc00a7620U, 0xbf6fb982U,
      0xbfe57ca6U, 0xbf874c06U, 0xbe65db9eU, 0xbfcb8970U, 0x3f2efd38U, 0x4013f7e6U, 0x3f317640U,
      0xc01ff736U, 0xbf845466U, 0xbf1c990cU, 0x3e4e7330U, 0xbff90c30U, 0xbfe5f98eU, 0x3f5fc9f8U,
      0xbecdac84U, 0xc01e3388U, 0xc07f4fc0U, 0xc01fe32cU, 0xbfb13e90U, 0xc00c167eU, 0xbf3b5582U,
      0xbf950a82U, 0x3e8e6136U, 0xbf7ece26U, 0x3f46ef68U, 0xbfb14b98U, 0x3f679df9U, 0xbfe85ef6U,
      0x3efb17e0U, 0xc06a4220U, 0xbf87de30U, 0xbfbbb6c3U, 0xbfb52e6fU, 0x3eb0cefaU, 0x3f4612b2U,
      0xbee83c25U, 0xbfa15dc2U, 0xc0482d12U, 0xbf2d3f59U, 0xbf338369U, 0xbef9f424U, 0xbd1827a2U,
      0x3f6820deU, 0x3eff6e5aU, 0xbebc78c4U, 0xbf34776eU, 0x3f2ad7f9U, 0xbfc051a2U, 0x3fda7fb7U,
      0xbf3af48aU, 0xbeb8e0ceU, 0xbfd8eab9U, 0xbf4e9f92U, 0xbff2bd08U, 0xbff6e8c4U, 0xc01018faU,
      0x3e1edd27U, 0x3f9a68d2U, 0xc002c654U, 0xbf68a5d5U, 0xbed1ba04U, 0xbf747cb2U, 0x3fa7cdbeU,
      0xbe9454eaU, 0x3f194027U, 0x3f021dc2U, 0xbfedf504U, 0xbfaa2d39U, 0xbea1a564U, 0x3fc0296fU,
      0xbed01fd2U, 0xc02a966eU, 0x3f9c03b8U, 0x3e889f84U, 0x3f116963U, 0xc011ab4cU, 0xc03aa6f3U,
      0x3e6ecf31U, 0xc05f83c0U, 0xbeb8a74eU, 0x3f5ba3c1U, 0xbfea5d8dU, 0x3f804834U, 0xbf918850U,
      0xbf85015cU, 0xbfb47bb6U, 0xc01bdf1fU, 0xbf84d0d4U, 0xbce7d688U, 0xbfd27376U, 0xbf8162deU,
      0x3f406137U, 0xbed5e5ddU, 0xbf7a26ebU, 0xbf55c9caU, 0xbf509010U, 0x3ff65cceU, 0xbfe3d5b3U,
      0xbf6e2564U, 0xbffe739fU, 0x3ebc14acU, 0xbf19e78cU, 0xbf675d76U, 0xbd2fce60U, 0xbf8c78cfU,
      0x3fa62f60U, 0x3f8013b6U, 0xbf0e9ab1U, 0x3c0705c0U, 0xbf142944U, 0x3e10a992U, 0xbffd0093U,
      0xbf3e3ec6U, 0xbf22751aU, 0xbd339ebeU, 0x3d999e24U, 0xc007e04cU, 0x3ec526ddU, 0xbe9b2835U,
      0xc06f594eU, 0xbfe1f09cU, 0xbf4104deU, 0xbc515d00U, 0xbf82f099U, 0xbfb2aa7aU, 0x3f6a14e1U,
      0xbf6b0c82U, 0x3f67d5f5U, 0xbed4261dU, 0x3d1b58f8U, 0xbf2f38d2U, 0xbecd07c0U, 0x3e3bc5b7U,
      0xc00a6848U, 0xbfa5ef4aU, 0xc00723b7U, 0xbf528e1bU, 0xbec6b70cU, 0x3e235888U, 0x3f479252U,
      0xbf8b51c4U, 0x3e06cc6bU, 0x3fa5aa98U, 0xbf3ef767U, 0x3e618e25U, 0xc0043baeU, 0xbf8a69e8U,
      0xbfa8ae22U, 0xc001cc10U, 0xbfad2c7aU, 0xbf889d19U, 0xc00c18d5U, 0xbef6367bU, 0xc0250730U,
      0xc0243058U, 0xbfaece6cU, 0xbd06b1dbU, 0x3f56f538U, 0xbfb3c43fU, 0xbf4f9aa0U, 0xbe0e4df0U,
      0xbdd50e40U, 0xbff26d3aU, 0x3e86d020U, 0x3e926bceU, 0xbf817836U, 0xbef5d102U, 0xbf806096U,
      0xbf0f6c2cU, 0xbed12c2fU, 0xc02912b1U, 0xbf063b4dU, 0xbfebec52U, 0xc00680d2U, 0xbe43099aU,
      0xbfe376c6U, 0xbf55a71dU, 0xbf8e73b6U, 0xc0372dc1U, 0x3e1d6e4aU, 0xbfc13844U, 0x3f99f6c6U,
      0xbf8c3a84U, 0x3da495b0U, 0x3f182ebfU, 0xbf1e5f65U, 0xbf902ad1U, 0xbfe064c1U, 0xbfcf6bfeU,
      0xbf527888U, 0xbf262b61U, 0x3c2318e0U, 0x3ff0f2e2U, 0xbe834d24U, 0x3eebe54fU, 0xbfb38e7eU,
      0xbec3f626U, 0xbf43de86U, 0x3f685d0aU, 0xbed1491dU};
  const std::vector<std::int32_t> captured_ids{90, 239, 52, 116, 20, 204};
  const std::vector<std::uint32_t> legacy_bits{0x3eba2bccU, 0x3e948d3cU, 0x3ebec02bU,
                                               0x3e776d82U, 0x3de76a46U, 0x3df3d5ecU};
  const std::vector<std::uint32_t> primitive_bits{0x3eba2bcbU, 0x3e948d3bU, 0x3ebec029U,
                                                  0x3e776d80U, 0x3de76a42U, 0x3df3d5e8U};
  for (const unsigned scenario : {0U, 1U, 2U, 3U}) {
    SCOPED_TRACE(scenario);
    const std::uint32_t rows = scenario == 0 ? 1U : 3U;
    const std::uint32_t used = scenario == 3 ? 32U : 6U;
    const bool normalize = scenario != 2;
    const float scale = scenario == 2 ? 0.75F : 1.5F;
    std::vector<float> values(256U * rows);
    std::vector<std::int32_t> ids(used * rows);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = scenario == 0 ? std::bit_cast<float>(captured[i])
                                : static_cast<float>(static_cast<int>((i * 71) % 191) - 95) / 8.0F;
    if (scenario != 0) {
      values[0] = -0.0F;
      values[1] = 0.0F;
      values[2] = 20.0F;
      values[3] = std::nextafter(20.0F, 21.0F);
      values[4] = -100.0F;
    }
    for (std::size_t i = 0; i < ids.size(); ++i)
      ids[i] =
          scenario == 0 ? captured_ids[i] : static_cast<std::int32_t>((i * 37 + i / used) % 256);
    if (scenario == 1) ids[1] = ids[0];  // selected IDs may repeat
    auto* c = owner->Context();
    ASSERT_NE(c, nullptr);
    auto* logits = owner->Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, rows));
    auto* table = owner->Place(ggml_new_tensor_2d(c, GGML_TYPE_I32, used, rows + 2));
    auto* tokens = owner->Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, rows));
    std::vector<std::int32_t> table_values(used * (rows + 2), 255);
    std::vector<std::int32_t> token_values(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      token_values[row] = static_cast<std::int32_t>(rows - row);
      std::copy_n(ids.begin() + static_cast<std::ptrdiff_t>(row * used), used,
                  table_values.begin() + static_cast<std::ptrdiff_t>((rows - row) * used));
    }
    owner->Upload(logits, values);
    owner->Upload(table, table_values);
    owner->Upload(tokens, token_values);
    ASSERT_FALSE(HasFatalFailure());
    auto* legacy = owner->Place(
        kg::Dsv4Route(c, logits, nullptr, table, tokens, used, normalize, 0x1p-14F, scale));
    ASSERT_TRUE(kg::CheckDsv4Route(legacy));
    ASSERT_TRUE(kg::RunDsv4Route(*owner->launch, legacy));
    ASSERT_TRUE(owner->Finish());
    const auto blob = owner->ReadTyped<std::uint32_t>(legacy);
    ASSERT_EQ(blob.size(), 2U * used * rows);
    std::vector<std::uint32_t> oracle(used * rows);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t slot = 0; slot < used; ++slot) {
        EXPECT_EQ(blob[row * 2 * used + slot], static_cast<std::uint32_t>(ids[row * used + slot]));
        oracle[row * used + slot] = blob[row * 2 * used + used + slot];
      }
    if (scenario == 0) EXPECT_EQ(oracle, legacy_bits);
    for (const bool native : {true, false}) {
      if (!native && scenario != 0) continue;
      SCOPED_TRACE(native);
      ir::Builder b;
      const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256, rows})));
      const auto selected = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({used, rows})));
      const auto out = b.Route(x, {}, selected, {},
                               {.kind = ir::RouteKind::kHash,
                                .normalize = normalize,
                                .experts = 256,
                                .used = used,
                                .scale = scale});
      b.Output(out[0]);
      b.Output(out[1]);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph);
      pl::Policy policy;
      if (!native) policy.disabled = {"hash_route"};
      pl::Canonicalize(policy);
      const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
      auto* step = owner->Lower(std::move(*graph),
                                {{.tensor = x, .kind = co::InputKind::kFeatures},
                                 {.tensor = selected, .kind = co::InputKind::kFeatures}},
                                shape, nullptr, false, policy, 2U << 20);
      ASSERT_NE(step, nullptr);
      ASSERT_EQ(std::ranges::count_if(
                    step->plan.steps,
                    [](const auto& item) { return item.implementation == kg::kDsv4RouteName; }),
                native ? 1 : 0);
      owner->Upload(owner->Leaf(*step, x), values);
      owner->Upload(owner->Leaf(*step, selected), ids);
      ASSERT_FALSE(HasFatalFailure());
      owner->RunRouteAndReplay(*step);
      ASSERT_FALSE(HasFatalFailure());
      ASSERT_EQ(step->keep.size(), 2U);
      EXPECT_EQ(owner->ReadTyped<std::int32_t>(step->keep[0]), ids);
      const auto weights = owner->ReadTyped<std::uint32_t>(step->keep[1]);
      EXPECT_EQ(weights, native ? oracle : primitive_bits);
      if (!native) EXPECT_NE(weights, legacy_bits);
    }
  }
}

// Explicit prepared IR: independently decoded exact Q8_0 weights and integral
// inputs (each block has max127, hence Q8_1 scale1) give an FP64 scalar oracle.
// This tests the actual selected/lowered recipe, not a hand-built VecQ helper.
TEST_F(GgmlHcPostGpu, PreparedRoutedTailBroadcastAndPerSlotMatchIndependentScalar) {
  constexpr std::uint64_t k = 64, n = 16, experts = 4, used = 8;
  for (const bool per_slot : {false, true})
    for (const std::uint32_t tokens : {1U, 2U, 3U, 4U}) {
      SCOPED_TRACE(per_slot);
      SCOPED_TRACE(tokens);
      ir::Builder b;
      const auto w = b.Input({.encoding = Encoding::kQ8_0,
                              .layout = llmp::format::Layout::kGgmlBlocks,
                              .shape = ir::ShapeOf({k, n, experts})});
      const auto x =
          b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({k, per_slot ? used : 1, tokens})));
      const auto ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({used, tokens})));
      b.Output(
          b.MatMulId(w, b.Prepare(x, Encoding::kQ8_1, llmp::format::Layout::kGgmlBlocks), ids));
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
      pl::Policy policy;
      policy.vector_rows = 4;
      const pl::StepShape shape{.segments = {{.rows = tokens, .outputs = 0, .want_head = false}}};
      auto* step = owner->Lower(std::move(*graph),
                                {{.tensor = w, .kind = co::InputKind::kFeatures},
                                 {.tensor = x, .kind = co::InputKind::kFeatures},
                                 {.tensor = ids, .kind = co::InputKind::kFeatures}},
                                shape, nullptr, false, policy);
      ASSERT_NE(step, nullptr);
      std::vector<std::byte> weights(k * n * experts / 32 * 34);
      for (std::size_t block = 0; block < weights.size() / 34; ++block) {
        weights[block * 34] = std::byte{0};
        weights[block * 34 + 1] = std::byte{0x1c};  // half1/256
        for (std::size_t j = 0; j < 32; ++j)
          weights[block * 34 + 2 + j] = std::bit_cast<std::byte>(
              static_cast<std::int8_t>(static_cast<int>((block * 3 + j * 5) % 15) - 7));
      }
      std::vector<std::int32_t> routes(used * tokens);
      for (std::size_t i = 0; i < routes.size(); ++i)
        routes[i] = static_cast<std::int32_t>((i * 3 + i / used) % experts);
      owner->Upload(owner->Leaf(*step, w), weights);
      owner->Upload(owner->Leaf(*step, ids), routes);
      ASSERT_FALSE(HasFatalFailure());
      for (const int revision : {0, 1}) {
        std::vector<float> values(k * (per_slot ? used : 1) * tokens);
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] =
              i % 32 == 0
                  ? 127.0F
                  : static_cast<float>(
                        static_cast<int>((i * 7 + static_cast<std::size_t>(revision) * 3) % 17) -
                        8);
        owner->Upload(owner->Leaf(*step, x), values);
        ASSERT_FALSE(HasFatalFailure());
        std::vector<double> expected(n * used * tokens);
        for (std::size_t t = 0; t < tokens; ++t)
          for (std::size_t u = 0; u < used; ++u)
            for (std::size_t row = 0; row < n; ++row) {
              double sum = 0;
              const auto expert = static_cast<std::size_t>(routes[t * used + u]);
              for (std::size_t j = 0; j < k; ++j) {
                const auto block = (expert * n + row) * (k / 32) + j / 32;
                const double weight = static_cast<double>(std::bit_cast<std::int8_t>(
                                          weights[block * 34 + 2 + j % 32])) /
                                      256.0;
                sum += weight * values[(t * (per_slot ? used : 1) + (per_slot ? u : 0)) * k + j];
              }
              expected[(t * used + u) * n + row] = sum;
            }
        owner->RunAndReplay(*step);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(step->keep.size(), 1U);
        const auto actual = owner->Read(step->keep[0]);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
          EXPECT_NEAR(actual[i], expected[i], 1e-6) << i;
      }
      EXPECT_EQ(owner->ReadTyped<std::byte>(owner->Leaf(*step, w)), weights);
    }
}

TEST_F(GgmlHcPostGpu, PreparedGluKindsMatchIndependentScalarOnTailRows) {
  constexpr std::uint64_t k = 64, n = 16, used = 8;
  for (const bool routed : {false, true})
    for (const auto kind :
         {ir::GluKind::kSwiGlu, ir::GluKind::kSwiGluClamp, ir::GluKind::kGeGluTanh})
      for (const std::uint32_t tokens : {1U, 2U, 3U, 4U}) {
        SCOPED_TRACE(routed);
        SCOPED_TRACE(static_cast<unsigned>(kind));
        SCOPED_TRACE(tokens);
        const std::uint64_t experts = routed ? 4 : 1;
        ir::Builder b;
        const llmp::format::TensorFormat format{.encoding = Encoding::kQ8_0,
                                                .layout = llmp::format::Layout::kGgmlBlocks,
                                                .shape = ir::ShapeOf({k, n, experts})};
        const auto gate = b.Input(format), up = b.Input(format);
        const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({k, tokens})));
        const auto q8 = b.Prepare(x, Encoding::kQ8_1, llmp::format::Layout::kGgmlBlocks);
        const ir::GluParams glu{.kind = kind,
                                .limit = kind == ir::GluKind::kSwiGluClamp ? 0.7F : 0.0F};
        std::vector<co::StepInput> inputs{{.tensor = gate, .kind = co::InputKind::kFeatures},
                                          {.tensor = up, .kind = co::InputKind::kFeatures},
                                          {.tensor = x, .kind = co::InputKind::kFeatures}};
        ir::TensorId ids;
        if (routed) {
          ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({used, tokens})));
          inputs.push_back({.tensor = ids, .kind = co::InputKind::kFeatures});
          b.Output(b.Fused(ir::FusedKind::kMatMulIdGlu, std::array{gate, up, q8, ids},
                           ir::MoeExpertsParams{.glu = glu, .hidden = n})[0]);
        } else {
          b.Output(b.Fused(ir::FusedKind::kMatMulGlu, std::array{gate, up, q8}, glu)[0]);
        }
        auto graph = std::move(b).Finish();
        ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
        pl::Policy policy;
        policy.vector_rows = 4;
        const pl::StepShape shape{.segments = {{.rows = tokens, .outputs = 0, .want_head = false}}};
        auto* step =
            owner->Lower(std::move(*graph), std::move(inputs), shape, nullptr, false, policy);
        ASSERT_NE(step, nullptr);
        std::array<std::vector<std::byte>, 2> weights;
        for (std::size_t which = 0; which < weights.size(); ++which) {
          auto& bytes = weights[which];
          bytes.resize(k * n * experts / 32 * 34);
          for (std::size_t block = 0; block < bytes.size() / 34; ++block) {
            bytes[block * 34] = std::byte{0};
            bytes[block * 34 + 1] = std::byte{0x1c};
            for (std::size_t j = 0; j < 32; ++j)
              bytes[block * 34 + 2 + j] = std::bit_cast<std::byte>(static_cast<std::int8_t>(
                  static_cast<int>((block * 3 + j * 5 + which * 7) % 15) - 7));
          }
          owner->Upload(owner->Leaf(*step, which == 0 ? gate : up), bytes);
        }
        std::vector<float> values(k * tokens);
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] = i % 32 == 0 ? 127.0F : static_cast<float>(static_cast<int>((i * 7) % 17) - 8);
        owner->Upload(owner->Leaf(*step, x), values);
        std::vector<std::int32_t> routes(used * tokens);
        for (std::size_t i = 0; i < routes.size(); ++i)
          routes[i] = static_cast<std::int32_t>((i * 3 + i / used) % experts);
        if (routed) owner->Upload(owner->Leaf(*step, ids), routes);
        ASSERT_FALSE(HasFatalFailure());
        std::vector<double> expected(n * (routed ? used : 1) * tokens);
        double separation_from_swiglu = 0;
        for (std::size_t t = 0; t < tokens; ++t)
          for (std::size_t u = 0; u < (routed ? used : 1); ++u)
            for (std::size_t row = 0; row < n; ++row) {
              std::array<double, 2> product{};
              const auto expert = routed ? static_cast<std::size_t>(routes[t * used + u]) : 0;
              for (std::size_t j = 0; j < k; ++j) {
                const auto block = (expert * n + row) * (k / 32) + j / 32;
                for (std::size_t which = 0; which < weights.size(); ++which)
                  product[which] += static_cast<double>(std::bit_cast<std::int8_t>(
                                        weights[which][block * 34 + 2 + j % 32])) /
                                    256.0 * values[t * k + j];
              }
              double g = product[0], up_value = product[1];
              const double swiglu = g / (1.0 + std::exp(-g)) * up_value;
              if (kind == ir::GluKind::kSwiGluClamp) {
                g = std::min(g, static_cast<double>(glu.limit));
                up_value = std::clamp(up_value, -static_cast<double>(glu.limit),
                                      static_cast<double>(glu.limit));
              }
              const double value = kind == ir::GluKind::kGeGluTanh
                                       ? 0.5 * g *
                                             (1.0 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) * g *
                                                              (1.0 + 0.044715 * g * g))) *
                                             up_value
                                       : g / (1.0 + std::exp(-g)) * up_value;
              separation_from_swiglu = std::max(separation_from_swiglu, std::abs(value - swiglu));
              expected[(t * (routed ? used : 1) + u) * n + row] = value;
            }
        if (kind != ir::GluKind::kSwiGlu) EXPECT_GT(separation_from_swiglu, 1e-3);
        owner->RunAndReplay(*step);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(step->keep.size(), 1U);
        const auto actual = owner->Read(step->keep[0]);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
          EXPECT_NEAR(actual[i], expected[i], 5e-5 * std::max(1.0, std::abs(expected[i]))) << i;
      }
}

TEST_F(GgmlHcPostGpu, GroupedVecQMatchesOriginalHelperForAllOutputsAndChangedInputReplay) {
  constexpr std::uint64_t k = 64, n = 16;
  for (const std::uint32_t tokens : {1U, 3U, 8U})
    for (const std::uint64_t groups : {2U, 8U}) {
      SCOPED_TRACE(tokens);
      SCOPED_TRACE(groups);
      ir::Builder b;
      const llmp::format::TensorFormat format{.encoding = Encoding::kQ8_0,
                                              .layout = llmp::format::Layout::kGgmlBlocks,
                                              .shape = ir::ShapeOf({k, n * groups})};
      const auto weight = b.Weight(format, {});
      const auto input = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({k, groups, tokens})));
      const auto product =
          b.MatMul(b.Reshape(weight, ir::ShapeOf({k, n, groups})), b.Permute(input, {0, 2, 1, 3}));
      b.Output(b.Cont(b.Permute(product, {0, 2, 1, 3}), ir::ShapeOf({n * groups, tokens})));
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph);
      // Ordinary Q8_0 blocks: exact half scale1, independently varied signed codes.
      std::vector<std::byte> weights(k * n * groups / 32 * 34);
      for (std::size_t block = 0; block < weights.size() / 34; ++block) {
        weights[block * 34] = std::byte{0};
        weights[block * 34 + 1] = std::byte{0x3c};
        for (std::size_t j = 0; j < 32; ++j) {
          const auto code =
              static_cast<std::int8_t>(static_cast<int>((block * 7 + j * 3) % 15) - 7);
          weights[block * 34 + 2 + j] = std::bit_cast<std::byte>(code);
        }
      }
      auto* step = owner->LowerGrouped(std::move(*graph), weight, input, tokens, weights.size());
      ASSERT_NE(step, nullptr);
      ASSERT_EQ(step->plan.steps.size(), 3U);
      EXPECT_EQ(step->plan.steps[0].implementation, kg::kQuantizeQ8Name);
      EXPECT_EQ(step->plan.steps[1].implementation, kg::kVecQName);
      auto* w = owner->Leaf(*step, weight);
      auto* x = owner->Leaf(*step, input);
      ASSERT_NE(w, nullptr);
      ASSERT_NE(x, nullptr);
      owner->Upload(w, weights);
      ASSERT_FALSE(HasFatalFailure());
      auto* context = owner->Context();
      ASSERT_NE(context, nullptr);
      auto* grouped =
          ggml_reshape_3d(context, w, static_cast<std::int64_t>(k), static_cast<std::int64_t>(n),
                          static_cast<std::int64_t>(groups));
      auto* quantized = owner->Place(kg::QuantizeQ8(context, x));
      auto* oracle = owner->Place(kg::VecQ(context, grouped, quantized, nullptr, tokens, true));
      for (int revision = 0; revision < 2; ++revision) {
        const auto values =
            Values(k * groups * tokens, 0.73F + 0.031F * static_cast<float>(revision));
        owner->Upload(x, values);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_TRUE(kg::CheckQuantizeQ8(quantized));
        ASSERT_TRUE(kg::CheckVecQ(oracle));
        ASSERT_TRUE(kg::RunQuantizeQ8(*owner->launch, quantized));
        ASSERT_TRUE(kg::RunVecQ(*owner->launch, oracle));
        ASSERT_TRUE(owner->Finish());
        const auto expected = owner->Read(oracle);
        owner->RunAndReplay(*step);
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(step->keep.size(), 1U);
        const auto actual = owner->Read(step->keep.front());
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
          ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                    std::bit_cast<std::uint32_t>(expected[i]))
              << i;
      }
      EXPECT_EQ(owner->ReadTyped<std::byte>(w), weights);
    }
}

TEST_F(GgmlHcPostGpu, NativeFmaAndPrimitiveFallbackMatchIndependentFullOutputs) {
  // Complete four-stream input tuples from the first round-sensitive
  // features of DS35's attention and FFN post boundaries, respectively.
  // These are input regressions, not replacements for full model controls.
  constexpr std::array<std::array<std::uint32_t, 25>, 2> captured{{
      {0xc001b03eU, 0x3db3dd20U, 0x3db3dd20U, 0x3db3dd20U, 0x3db3dd20U, 0x3d554f3eU, 0x32c37ebeU,
       0x32b80626U, 0x381c2e6fU, 0x3f686721U, 0x3b58d86eU, 0x3c54832cU, 0x3aa44bfbU, 0x2fb92853U,
       0x3f669390U, 0x3a04d607U, 0x3dfd7888U, 0x2ed159cdU, 0x3dc3ead5U, 0x3f7c35c4U, 0x341a355dU,
       0x3dbcc688U, 0x39b15898U, 0x3aadce32U, 0x3f5ffeb8U},
      {0x3f1a2c72U, 0x3d08feb4U, 0xbcafb3a7U, 0xbcafb3a7U, 0xbcaf6166U, 0x3a32bbe1U, 0x3627d308U,
       0x37302434U, 0x3f384361U, 0x3eec7406U, 0x3a82a0fcU, 0x383bb443U, 0x3efe25dcU, 0x34b138bdU,
       0x3f7f034bU, 0x3d94dc0bU, 0x36bc41c8U, 0x364bb109U, 0x3811afd4U, 0x3f6d5dceU, 0x3daf584cU,
       0x3f09c5b4U, 0x3b390c5bU, 0x386c1538U, 0x3ed60335U},
  }};
  std::size_t recipe_differences = 0;
  for (std::size_t scenario = 0; scenario < 3; ++scenario) {
    const bool actual_tuple = scenario != 0;
    const std::uint32_t width = actual_tuple ? 1U : 17U;
    const std::uint32_t rows = actual_tuple ? 1U : 3U;
    std::vector<float> current = Values(width * rows, 0.7F);
    std::vector<float> residual = Values(width * 4 * rows, 0.3F);
    std::vector<float> post = Values(4 * rows, 0.6F);
    std::vector<float> comb = Values(16 * rows, 0.8F);
    if (actual_tuple) {
      const auto& tuple = captured[scenario - 1];
      current[0] = std::bit_cast<float>(tuple[0]);
      for (std::size_t s = 0; s < 4; ++s) {
        residual[s] = std::bit_cast<float>(tuple[1 + s]);
        post[s] = std::bit_cast<float>(tuple[5 + s]);
      }
      for (std::size_t c = 0; c < 16; ++c) comb[c] = std::bit_cast<float>(tuple[9 + c]);
    } else {
      // Independent cancellation and signed-zero cases, on every token.
      for (std::size_t t = 0; t < rows; ++t) {
        current[t * width] = -1.0F;
        current[t * width + 1] = -0.0F;
        for (std::size_t d = 0; d < 4; ++d) {
          post[t * 4 + d] = 1.0F;
          for (std::size_t s = 0; s < 4; ++s)
            comb[t * 16 + s * 4 + d] = s == 0 ? 1.0F - 0x1p-23F : 1.0F;
          residual[t * width * 4 + d * width] = d == 0 ? 1.0F + 0x1p-23F : 0.0F;
          residual[t * width * 4 + d * width + 1] = -0.0F;
        }
      }
    }
    std::array<std::vector<float>, 2> expected;
    for (auto& values : expected) values.resize(width * 4 * rows);
    for (std::size_t t = 0; t < rows; ++t)
      for (std::size_t d = 0; d < 4; ++d)
        for (std::size_t i = 0; i < width; ++i) {
          volatile float initial = current[t * width + i] * post[t * 4 + d];
          float fused = initial, separate = initial;
          for (std::size_t s = 0; s < 4; ++s) {
            const float x = residual[t * width * 4 + s * width + i];
            const float c = comb[t * 16 + s * 4 + d];
            fused = std::fma(x, c, fused);
            volatile float product = x * c;
            separate = separate + product;
          }
          const auto index = t * width * 4 + d * width + i;
          expected[0][index] = fused;
          expected[1][index] = separate;
          recipe_differences +=
              std::bit_cast<std::uint32_t>(fused) != std::bit_cast<std::uint32_t>(separate);
        }
    if (!actual_tuple) {
      EXPECT_EQ(expected[0][0], -0x1p-46F);
      EXPECT_EQ(expected[1][0], 0.0F);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(expected[0][1]), 0x80000000U);
    }
    for (bool native : {true, false}) {
      SCOPED_TRACE(scenario);
      SCOPED_TRACE(native);
      ir::Builder b;
      const auto y = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})));
      const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, 4, rows})));
      const auto p = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({4, rows})));
      const auto c = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({4, 4, rows})));
      b.Output(b.HcPost(y, x, p, c, {.streams = 4}));
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph);
      const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
      auto* step = owner->Lower(std::move(*graph),
                                {{.tensor = y, .kind = co::InputKind::kFeatures},
                                 {.tensor = x, .kind = co::InputKind::kFeatures},
                                 {.tensor = p, .kind = co::InputKind::kFeatures},
                                 {.tensor = c, .kind = co::InputKind::kFeatures}},
                                shape, native ? nullptr : &*owner->primitive);
      ASSERT_NE(step, nullptr);
      ASSERT_EQ(step->plan.steps.size(), native ? 1U : 39U);
      ASSERT_EQ(
          std::ranges::count_if(step->plan.steps,
                                [](const auto& s) { return s.implementation == kg::kHcPostName; }),
          native ? 1 : 0);
      owner->Upload(owner->Leaf(*step, y), current);
      owner->Upload(owner->Leaf(*step, x), residual);
      owner->Upload(owner->Leaf(*step, p), post);
      owner->Upload(owner->Leaf(*step, c), comb);
      ASSERT_FALSE(HasFatalFailure());
      owner->RunAndReplay(*step);
      ASSERT_FALSE(HasFatalFailure());
      ASSERT_EQ(step->keep.size(), 1U);
      const auto got = owner->Read(step->keep.front());
      const auto& oracle = expected[native ? 0U : 1U];
      ASSERT_EQ(got.size(), oracle.size());
      for (std::size_t i = 0; i < got.size(); ++i)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(got[i]), std::bit_cast<std::uint32_t>(oracle[i]))
            << i;
    }
  }
  EXPECT_GT(recipe_differences, 0U);
}

TEST_F(GgmlLowerDsv4Gpu, SinkhornPrimitiveMatchesNativeHelperAndChangedInputCapture) {
  for (const std::uint32_t rows : {1U, 9U, 512U}) {
    for (const std::uint32_t iterations : {1U, 5U, 20U}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(iterations);
      ir::Builder b;
      b.SetOwner(0);
      const auto mixes = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({24, rows})));
      const auto scale = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({3})));
      const auto base = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({24})));
      const auto comb = b.HcMix(
          mixes, scale, base,
          {.mix = ir::HcMixKind::kSinkhorn, .streams = 4, .iterations = iterations, .eps = 1e-6F});
      b.Output(comb);
      // Retain an actual downstream use as well as the complete native output.
      b.Output(b.Scale(comb, 0.5F));
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
      const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
      auto* step = Lower(std::move(*graph),
                         {{.tensor = mixes, .kind = co::InputKind::kFeatures},
                          {.tensor = scale, .kind = co::InputKind::kFeatures},
                          {.tensor = base, .kind = co::InputKind::kFeatures}},
                         shape);
      ASSERT_NE(step, nullptr);
      ASSERT_EQ(
          std::ranges::count_if(step->plan.steps,
                                [](const auto& s) { return s.implementation == kg::kHcCombName; }),
          1);
      auto* oracle =
          Place(ggml_dsv4_hc_comb(Context(), Leaf(*step, mixes), Leaf(*step, scale),
                                  Leaf(*step, base), 1e-6F, static_cast<std::int32_t>(iterations)));
      ASSERT_TRUE(kg::CheckHcComb(oracle));
      for (int revision = 0; revision < 3; ++revision) {
        SCOPED_TRACE(revision);
        auto values = Values(24 * rows, 0.1F + 0.03F * static_cast<float>(revision));
        auto bias = Values(24, 0.01F + 0.02F * static_cast<float>(revision));
        if (revision == 2) {
          values[8] = std::numeric_limits<float>::infinity();
          values[9] = -std::numeric_limits<float>::infinity();
          bias[10] = std::numeric_limits<float>::quiet_NaN();
        }
        Upload(Leaf(*step, mixes), values);
        Upload(Leaf(*step, scale),
               std::vector<float>{0.2F, 0.3F, 0.6F + 0.1F * static_cast<float>(revision)});
        Upload(Leaf(*step, base), bias);
        ASSERT_TRUE(kg::HcComb(*launch, oracle));
        ASSERT_TRUE(Finish());
        const auto expected = ReadTyped<std::uint32_t>(oracle);
        RunAndReplay(*step);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        EXPECT_EQ(ReadTyped<std::uint32_t>(step->keep[0]), expected);
        const auto downstream = Read(step->keep[1]);
        ASSERT_EQ(downstream.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
          const float value = std::bit_cast<float>(expected[i]);
          if (std::isnan(value))
            EXPECT_TRUE(std::isnan(downstream[i]));
          else
            EXPECT_EQ(downstream[i], value * 0.5F);
        }
      }
    }
  }
}

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
                         .fixed_cells = static_cast<std::uint32_t>(compressed_capacity)}}}}};
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
          if (selected.operation != llmp::ir::Primitive(llmp::ir::Op::kAttention)) continue;
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
    const auto out = b.Fused(ir::FusedKind::kRmsNormMulRope, std::array{q, pos}, params)[0];
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
  b.Output(b.Fused(ir::FusedKind::kSoftmaxPoolCompress,
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
       .kind = co::InputKind::kHostComputed,
       .segment = 0,
       .cells = {.capacity = 2, .ring = true, .recent = true},
       .host = {.def = &co::CompressorPoolReadsInput(), .ratio = 1, .overlap = true},
       .ratio = 1,
       .overlap = true,
       .ring = 2}};
  auto implementations = kg::Implementations();
  std::erase_if(implementations, [](const auto& i) { return i.name == kg::kDsv4CompressName; });
  auto primitive = ex::Registry::Create(std::move(implementations));
  ASSERT_TRUE(primitive);
  auto fast_graph = graph->Clone();
  ASSERT_TRUE(fast_graph) << fast_graph.error().detail;
  auto* fast = Lower(std::move(*fast_graph), inputs, shape);
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
                      .kind = co::InputKind::kHostComputed,
                      .segment = 0,
                      .cells = {.capacity = ring, .ring = true, .recent = true},
                      .host = {.def = &co::CompressorPoolReadsInput(),
                               .ratio = test.ratio,
                               .overlap = test.overlap},
                      .ratio = test.ratio,
                      .overlap = test.overlap,
                      .ring = ring});
    b.Output(b.Fused(ir::FusedKind::kSoftmaxPoolCompress,
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
      auto clone = graph->Clone();
      ASSERT_TRUE(clone) << clone.error().detail;
      auto* step = Lower(std::move(*clone), inputs, shape, selected);
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
          b.Fused(ir::FusedKind::kSinkhornHcPre, std::array{x, fn, scale, base, norm}, params);
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

// A fixed synthetic corridor, separate from full ComponentStepSource admission.
// Its real backend owner is prepared and retained through every captured use.
namespace rounded_corridor {
namespace pr = llmp::providers;
constexpr std::uint64_t kHost = 64U << 20, kDevice = 192U << 20;
constexpr std::uint64_t kActivation = 2U << 20, kPool = 16U << 20, kPinned = 1U << 20;
static_assert(sizeof(ir::Node) <= 384 && sizeof(ir::TensorDesc) <= 192);
struct Owner {
  std::shared_ptr<llmp::catalog::Catalog> metadata_catalog;
  std::shared_ptr<llmp::engine::ExecutionMetadataAllowance> metadata_paid;
  std::unique_ptr<pr::DeviceExecution> execution;
  pr::StreamId stream{};
  bool streaming = false, unknown = false;
  std::unique_ptr<kg::CublasHandle> cublas;
  std::unique_ptr<kg::GgmlBackend> backend;
  std::unique_ptr<kn::StepLaunch> launch;
  std::unique_ptr<co::StepGraph> step;
  std::optional<pl::StepPlan> plan;
  std::unique_ptr<kn::StepProgram> program;
  std::unique_ptr<kn::Recorded> captured;
  std::array<void*, 4> allocations{};
  std::size_t allocation_count = 0;
  std::uint64_t allocated = 0, activation = 0;
  kg::SelectedWorkspace selected_state{};
  void* pinned = nullptr;
  bool Finish() {
    if (unknown) return false;
    if (!streaming) return true;
    auto fence = execution->Record(stream);
    if (!fence) {
      unknown = true;
      return false;
    }
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      auto state = execution->Query(*fence);
      if (!state || std::chrono::steady_clock::now() >= until) {
        unknown = true;
        return false;
      }
      if (*state == pr::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution->Release(*fence)) {
      unknown = true;
      return false;
    }
    return true;
  }
  bool Close() {
    if (!Finish()) return false;
    captured.reset();
    program.reset();
    plan.reset();
    step.reset();
    launch.reset();
    if (backend) backend->LaunchReleased();
    backend.reset();
    cublas.reset();
    if (streaming && !execution->DestroyStream(stream)) return false;
    streaming = false;
    for (auto& address : allocations) {
      if (address && cudaFree(address) != cudaSuccess) return false;
      address = nullptr;
    }
    if (pinned) {
      pr::FreePinned(pinned);
      pinned = nullptr;
    }
    execution.reset();
    return true;
  }
  std::uint64_t Allocate(std::uint64_t bytes) {
    if (!bytes || bytes > kDevice - allocated || allocation_count == allocations.size()) return 0;
    void* address = nullptr;
    if (cudaMalloc(&address, bytes) != cudaSuccess) return 0;
    allocations[allocation_count++] = address;
    allocated += bytes;
    return reinterpret_cast<std::uintptr_t>(address);
  }
  bool Copy(pr::NativeStream native, void* to, const void* from, std::size_t bytes,
            pr::CopyKind kind) {
    if (pr::CopyAsync(native, to, from, bytes, kind).ok()) return true;
    unknown = true;
    return false;
  }
  std::uint64_t Address(ir::TensorId original, std::uint64_t bytes) const {
    if (original.v >= plan->tensors.size()) return 0;
    const auto mapped = plan->tensors[original.v];
    const auto root = plan->graph.root(mapped);
    const auto at = std::ranges::find(plan->placement.offsets, root,
                                      &std::pair<ir::TensorId, std::uint64_t>::first);
    if (at == plan->placement.offsets.end()) return 0;
    const auto offset = at->second + plan->graph.tensor(mapped).view_offset;
    if (offset > kActivation || bytes > kActivation - offset) return 0;
    return activation + offset;
  }
};
struct Retire {
  void operator()(Owner* owner) const {
    if (!owner) return;
    if (!owner->Close()) {
      ADD_FAILURE() << "unproved corridor retirement; whole owner retained";
      return;
    }
    delete owner;
  }
};
}  // namespace rounded_corridor

TEST(GgmlRoundedConsumerGpu, DeclaredCountsSelectOneFullMaskAcrossChangedCapture) {
  namespace rc = rounded_corridor;
  namespace pr = llmp::providers;
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available);
  ASSERT_GE(*available, rc::kHost + rc::kDevice + (6ULL << 30));
  ASSERT_LE(kg::Implementations().size(), 4096U);
  // Fixed complete 64MiB host/192MiB device inventory is admitted before
  // construction. The wrapper separately confines process/cache/runtime.
  std::unique_ptr<rc::Owner, rc::Retire> life(new rc::Owner);
  auto execution = pr::cuda::OpenDeviceExecution(0);
  ASSERT_TRUE(execution);
  life->execution = std::move(*execution);
  auto stream = life->execution->CreateStream();
  ASSERT_TRUE(stream);
  life->stream = *stream;
  life->streaming = true;
  auto facts = pr::QueryDeviceFacts(0);
  ASSERT_TRUE(facts);
  const auto workspace = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
  ASSERT_LE(workspace.value(), rc::kDevice - rc::kPool - rc::kActivation - (32U << 20));
  const auto cublas_address = life->Allocate(workspace.value());
  ASSERT_NE(cublas_address, 0U);
  auto cublas = kg::CublasHandle::Create(0, *life->execution, life->stream,
                                         {.base = cublas_address, .size = workspace});
  ASSERT_TRUE(cublas);
  life->cublas = std::move(*cublas);
  auto backend = kg::GgmlBackend::Create(0, *life->execution, life->stream, nullptr,
                                         {.lower = {.index_scratch_limit = 1U << 20}});
  ASSERT_TRUE(backend);
  life->backend = std::move(*backend);
  life->metadata_catalog = std::make_shared<llmp::catalog::Catalog>();
  const auto metadata_domain = life->metadata_catalog->AddDomain("corridor host inventory");
  auto paid = llmp::engine::ExecutionMetadataAllowance::Admit(
      *life->metadata_catalog, metadata_domain, 16U << 20U, *available, 6ULL << 30U);
  ASSERT_TRUE(paid) << (paid ? "" : paid.error());
  life->metadata_paid = *paid;
  // The existing complete 64MiB host envelope includes this finite category.
  const auto charge = [paid = *paid, keep = life->metadata_catalog](std::uint64_t bytes) {
    (void)keep;
    return paid->Charge(bytes);
  };
  const auto refund = [paid = *paid, keep = life->metadata_catalog](std::uint64_t bytes) {
    (void)keep;
    EXPECT_LE(bytes, paid->used());
    paid->Release(bytes);
  };
  ASSERT_TRUE(life->backend->FundConstruction(charge, refund));
  ASSERT_TRUE(life->backend->Declare({}, workspace.value(), charge, refund));
  ASSERT_FALSE(life->backend->Backend().owner->Ready());
  ASSERT_TRUE(life->backend->LendCublas(life->cublas.get()));
  const auto table = life->backend->Backend();
  ASSERT_NE(table.owner, nullptr);
  ASSERT_TRUE(table.owner->Ready());
  const auto& native = table.owner->backend();
  ASSERT_LE(life->backend->persistent_bytes(), 32U << 20);
  life->selected_state.bytes = life->backend->persistent_bytes();
  life->selected_state.base =
      life->selected_state.bytes ? life->Allocate(life->selected_state.bytes) : 0;
  const auto pool = life->Allocate(rc::kPool);
  ASSERT_NE(pool, 0U);
  auto launch = life->backend->Launch({.base = pool, .size = llmp::base::Bytes(rc::kPool)},
                                      life->selected_state);
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  life->launch = std::move(*launch);
  life->activation = life->Allocate(rc::kActivation);
  ASSERT_NE(life->activation, 0U);
  auto pinned = pr::AllocatePinned(rc::kPinned);
  ASSERT_TRUE(pinned);
  life->pinned = *pinned;
  std::memset(life->pinned, 0, rc::kPinned);

  ir::Builder b(32);
  b.SetOwner(0);
  co::StepContext context(b, {{.slot = 0, .rows = 4, .past = 2}}, false);
  const auto q = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({128, 64, 4})));
  const auto k = b.Input(ir::Dense(Encoding::kF16, ir::ShapeOf({128, 1, 256})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({64, 4})));
  const auto raw = b.Input(ir::Dense(Encoding::kF16, ir::ShapeOf({8, 4})));
  auto counts = context.VisibleCells(0, {.capacity = 257, .block = 4, .positions = true}, 256, 64);
  ASSERT_TRUE(counts);
  const auto scores = b.RoundedIndexScore(q, k, w, *counts);
  const auto ids = b.VisibleTopK(scores, *counts, 2);
  const auto mask = b.SelectedMask(raw, ids, {.raw_cells = 8, .cells = 256});
  for (auto out : {scores, ids, mask}) b.Output(out);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  auto inputs = context.TakeInputs();
  for (auto in : {q, k, w, raw})
    inputs.push_back({.tensor = in, .kind = co::InputKind::kFeatures, .segment = 0});
  life->step = std::make_unique<co::StepGraph>(co::StepGraph{
      .graph = std::move(*graph), .inputs = std::move(inputs), .weights = {}, .state = {}});
  const pl::StepShape shape{.segments = {{.rows = 4, .past = 2, .read_cells = 256}},
                            .read_align = 256};
  auto planned = pl::Plan(life->step->graph.Clone(), *native.registry, native.device, {},
                          {.capturable = true,
                           .on_stream = native.on_stream,
                           .written = native.written,
                           .within = native.within,
                           .shape = &shape,
                           .activation_alignment = 256});
  ASSERT_TRUE(planned);
  life->plan.emplace(std::move(*planned));
  ASSERT_EQ(life->plan->steps.size(), 3U);
  ASSERT_TRUE(life->plan->applied.empty());
  ASSERT_TRUE(life->plan->decomposed.empty());
  const std::array<std::string_view, 3> names{kg::kRoundedIndexScoreName, kg::kVisibleTopKName,
                                              kg::kSelectedMaskName};
  for (std::size_t i = 0; i < names.size(); ++i)
    ASSERT_EQ(native.registry->at(life->plan->steps[i].implementation).name, names[i]);
  // With these three undecomposed native nodes, Lowering::Run reserves
  // tensor_count +16 (8+16), with no recipe/reduction/conversion extras.
  ASSERT_EQ(life->plan->graph.tensor_count(), 8U);
  ASSERT_LE(life->plan->placement.extent, rc::kActivation);
  const kn::Addresses addresses{.weight = {},
                                .weight_rows = {},
                                .state = {},
                                .save = {},
                                .activations = life->activation,
                                .activation_bytes = rc::kActivation,
                                .activation_alignment = 256};
  auto made = native.lower({*life->plan, *life->step, shape, addresses, 257});
  ASSERT_TRUE(made);
  life->program = std::move(*made);
  ASSERT_LE(life->program->host_bytes(), 1U << 20);
  ASSERT_LE(life->program->pool_bytes(), rc::kPool);
  ASSERT_EQ(life->program->launches(), 3U);
  ASSERT_EQ(life->program->inputs().size(), 5U);
  ASSERT_TRUE(life->program->Bind(*life->launch));
  auto submission = life->execution->Submission(life->stream);
  ASSERT_TRUE(submission);
  auto* bytes = static_cast<std::byte*>(life->pinned);
  auto* query = reinterpret_cast<float*>(bytes);
  auto* keys = reinterpret_cast<std::uint16_t*>(bytes + 131072);
  auto* weights = reinterpret_cast<float*>(bytes + 196608);
  auto* raw_values = reinterpret_cast<std::uint16_t*>(bytes + 197632);
  auto* count_values = reinterpret_cast<std::int32_t*>(bytes + 197696);
  auto* score_values = reinterpret_cast<float*>(bytes + 198144);
  auto* id_values = reinterpret_cast<std::int32_t*>(bytes + 202240);
  auto* mask_values = reinterpret_cast<std::uint16_t*>(bytes + 202496);
  static_assert(202496 + 264 * 4 * 2 < rc::kPinned);
  const std::array<std::uint16_t, 8> literal{0, 0x8000, 0xfc00, 0x7c00, 1, 0x3c00, 0xbc00, 0x7e31};
  const auto count_input = std::ranges::find(life->step->inputs, *counts, &co::StepInput::tensor);
  ASSERT_NE(count_input, life->step->inputs.end());
  for (std::uint32_t revision = 0; revision != 3; ++revision) {
    ASSERT_TRUE(life->Finish());
    auto current = shape;
    current.segments[0].past = std::array{2U, 6U, 253U}[revision];
    auto visible = co::InputValues(*count_input, current, {});
    ASSERT_TRUE(visible);
    ASSERT_EQ(visible->size(), 4U);
    for (std::size_t row = 0; row != 4; ++row) {
      query[row * 8192] = static_cast<float>(revision + 1);
      weights[row * 64] = 2;
      count_values[row] = static_cast<std::int32_t>((*visible)[row]);
      for (std::size_t cell = 0; cell != 8; ++cell)
        raw_values[row * 8 + cell] = literal[(cell + revision) % 8];
    }
    // Change the winning cell as well as completed counts between replays.
    for (std::size_t cell = 0; cell != 256; ++cell)
      keys[cell * 128] = cell == revision ? 0x4000 : 0x3c00;
    const auto stage = [&] {
      for (const auto& input : life->program->inputs()) {
        const void* from = input.tensor == q         ? static_cast<void*>(query)
                           : input.tensor == k       ? static_cast<void*>(keys)
                           : input.tensor == w       ? static_cast<void*>(weights)
                           : input.tensor == raw     ? static_cast<void*>(raw_values)
                           : input.tensor == *counts ? static_cast<void*>(count_values)
                                                     : nullptr;
        ASSERT_NE(from, nullptr);
        const std::uint64_t expected_bytes = input.tensor == q     ? 131072
                                             : input.tensor == k   ? 65536
                                             : input.tensor == w   ? 1024
                                             : input.tensor == raw ? 64
                                                                   : 16;
        ASSERT_EQ(input.bytes, expected_bytes);
        ASSERT_TRUE(life->Copy(*submission, reinterpret_cast<void*>(input.device_address), from,
                               input.bytes, pr::CopyKind::kHostToDevice));
      }
      ASSERT_TRUE(life->Finish());
    };
    const auto check = [&] {
      ASSERT_TRUE(life->Finish());
      for (const auto [tensor, to, size] :
           {std::tuple{scores, static_cast<void*>(score_values), std::uint64_t{4096}},
            std::tuple{ids, static_cast<void*>(id_values), std::uint64_t{32}},
            std::tuple{mask, static_cast<void*>(mask_values), std::uint64_t{2112}}}) {
        const auto address = life->Address(tensor, size);
        ASSERT_NE(address, 0U);
        ASSERT_TRUE(life->Copy(*submission, to, reinterpret_cast<void*>(address), size,
                               pr::CopyKind::kDeviceToHost));
      }
      ASSERT_TRUE(life->Finish());
      for (std::size_t row = 0; row != 4; ++row) {
        for (std::size_t cell = 0; cell != 256; ++cell)
          if (cell < static_cast<std::size_t>(count_values[row]))
            EXPECT_EQ(score_values[row * 256 + cell],
                      static_cast<float>(2 * (revision + 1) * (cell == revision ? 2 : 1)));
          else
            EXPECT_EQ(std::bit_cast<std::uint32_t>(score_values[row * 256 + cell]), 0xff800000U);
        const std::vector<float> actual(score_values + row * 256, score_values + (row + 1) * 256);
        const auto selected = VisibleIds(actual, count_values[row], 2);
        for (std::size_t i = 0; i != 2; ++i) EXPECT_EQ(id_values[row * 2 + i], selected[i]);
        for (std::size_t cell = 0; cell != 8; ++cell)
          EXPECT_EQ(mask_values[row * 264 + cell], raw_values[row * 8 + cell]);
        for (std::int32_t cell = 0; cell != 256; ++cell) {
          const bool chosen = id_values[row * 2] == cell || id_values[row * 2 + 1] == cell;
          EXPECT_EQ(mask_values[row * 264 + 8 + static_cast<std::size_t>(cell)],
                    chosen ? 0 : 0xfc00);
        }
      }
    };
    stage();
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_TRUE(life->program->Run(*life->launch, *submission));
    check();
    ASSERT_FALSE(::testing::Test::HasFailure());
    if (!life->captured) {
      auto captured = life->program->Record(*life->launch, *submission, [&] {
        return life->program->Run(*life->launch, *submission);
      });
      ASSERT_TRUE(captured);
      life->captured = std::move(*captured);
      ASSERT_GT(life->captured->nodes(), 0U);
      ASSERT_LE(life->captured->nodes(), 16U);
    }
    stage();
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_TRUE(life->launch->Launch(*submission, *life->captured));
    check();
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
  RecordProperty("revisions", 3);
  RecordProperty("completed_count_capacity", 64);
  RecordProperty("raw_mask_bits_checked", 192);
  RecordProperty("full_mask_elements_checked", 6336);
}

// Actual mapped source execution, deliberately excluding FFN/routing and model admission.
namespace mapped_csa {
namespace en = llmp::engine;
namespace md = llmp::model;
namespace sp = llmp::spec;
namespace fam = llmp::test_support::families;
namespace fx = llmp::test_support::dsv4;
// Fixed one-layer, one-owner source inventory (external funding derivation).
// These are source-specific construction ceilings, not the global IR limits.
constexpr std::uint64_t kNodes = 2048, kTensors = 8192;
constexpr std::uint64_t kDescriptors = kTensors + 16 + 16 * kNodes;
constexpr std::uint64_t kGraphs =
    llmp::test_support::compiler_inventory::GraphBytes(kNodes, kTensors);
constexpr std::uint64_t kWorksets = 4096 * (kNodes + kTensors);
constexpr std::uint64_t kNative =
    llmp::test_support::compiler_inventory::NativeBytes(kDescriptors, kTensors);
constexpr std::uint64_t kRetained = 32ULL << 20, kMisc = 32ULL << 20;
constexpr std::uint64_t kHost = 256ULL << 20;
static_assert(kGraphs + kWorksets + kNative + kRetained + kMisc <= kHost);
static_assert(sizeof(ir::Node) + 96 <= llmp::test_support::compiler_inventory::kGraphNodeBytes &&
              sizeof(ir::TensorDesc) <= 192);
constexpr std::uint64_t kPacked = 64ULL << 20, kWeights = 16ULL << 20;
constexpr std::uint64_t kStateOffset = kWeights, kActivationOffset = 32ULL << 20;
constexpr std::uint64_t kActivationBytes = kPacked - kActivationOffset;
constexpr std::uint64_t kStateReadback = 512ULL << 10;
struct Owner : rounded_corridor::Owner {
  sp::Source tensors;
  sp::ModelSpec spec;
  md::StateLayoutDerived layout;
  std::unique_ptr<en::ComponentStepSource> source;
  std::map<ir::WeightRef, std::pair<llmp::format::PartRole, std::uint64_t>> weight_offsets;
  std::uint64_t weight_bytes = 0, packed = 0;
  std::array<std::uint32_t, 160> baseline{};
  std::vector<std::byte> state_baseline;
};
struct Retire {
  void operator()(Owner* owner) const {
    if (!owner) return;
    if (!owner->Finish()) {
      ADD_FAILURE() << "unknown mapped-source completion; complete owner retained";
      return;
    }
    owner->captured.reset();
    owner->program.reset();
    owner->source.reset();  // Release execution metadata while its backend is alive.
    if (!owner->Close()) {
      ADD_FAILURE() << "unknown mapped-source retirement; complete owner retained";
      return;
    }
    delete owner;
  }
};
std::uint64_t Bytes(const ir::TensorDesc& t) {
  if (!ir::IsContiguous(t) || t.is_view()) return 0;
  return t.nb[3] * t.ne(3);
}
void Describe(Owner& life, const pl::StepShape& shape) {
  auto p = fam::Dsv4Flash();
  p.name = "synthetic-csa-source";
  p.layers = 1;
  p.width = 64;
  p.hc = 4;
  p.heads = 8;
  p.head_dim = 256;
  p.rope_dims = 8;
  p.q_lora = 32;
  p.o_lora = 16;
  p.o_groups = 2;
  p.window = 6;
  p.experts = 8;
  p.experts_used = 2;
  p.expert_ffn = 32;
  p.hash_layers = 0;
  p.sinkhorn_iterations = 3;
  p.indexer_heads = 64;
  p.indexer_head_dim = 128;
  p.indexer_top_k = 3;
  p.vocab = 40;
  p.yarn_original_context = 64;
  p.compress_rope_base = 1000;
  p.rope_scale = 4;
  p.compress_ratios = {4};
  life.tensors =
      fx::SourceOf("deepseek4", fx::Resources(p, fx::kF16Types), sp::mappings::Dsv4Mapping());
  auto values = fam::Dsv4Values(p);
  ASSERT_TRUE(values) << values.error();
  auto spec = sp::Synthesize(sp::mappings::Dsv4Mapping(), life.tensors, *values);
  ASSERT_TRUE(spec) << spec.error();
  auto& stage = spec->stages.at(0);
  ASSERT_EQ(stage.layers.size(), 1U);
  ASSERT_EQ(stage.layers[0].components.size(), 6U);
  ASSERT_EQ(stage.layers[0].components[0].kind, "hc.mix");
  ASSERT_EQ(stage.layers[0].components[1].kind, "attention.mqa");
  ASSERT_EQ(stage.layers[0].components[2].kind, "hc.combine");
  stage.layers[0].components.resize(3);  // One declared synthetic attention pipeline.
  ASSERT_TRUE(stage.side.empty());
  ASSERT_TRUE(stage.taps.empty());
  ASSERT_TRUE(sp::Validate(*spec));
  life.spec = std::move(*spec);
  auto layout = co::StageLayout(life.spec, 0, 257, 4);
  ASSERT_TRUE(layout) << layout.error();
  ASSERT_LE(layout->items.size(), 32U);
  ASSERT_LE(layout->bytes, kStateReadback);
  life.layout = std::move(*layout);
  auto step =
      co::BuildStep(life.spec, 0, shape, {.tensors = life.tensors.tensors, .state = &life.layout},
                    {.rounded_indexer = true});
  ASSERT_TRUE(step) << step.error();
  ASSERT_LE(step->graph.nodes().size(), kNodes);
  ASSERT_LE(step->graph.tensor_count(), kTensors);
  ASSERT_LE(step->weights.size(), 64U);
  ASSERT_LE(step->inputs.size(), 128U);
  for (const auto& weight : step->weights) {
    ASSERT_FALSE(weight.ref.expert_array);
    ASSERT_EQ(weight.parts.size(), 1U);
    const auto bytes = Bytes(step->graph.tensor(weight.tensor));
    ASSERT_GT(bytes, 0U);
    ASSERT_LE(bytes, rounded_corridor::kPinned);
    const auto offset = (life.weight_bytes + 255) & ~std::uint64_t{255};
    ASSERT_LE(offset, kWeights);
    ASSERT_LE(bytes, kWeights - offset);
    ASSERT_TRUE(
        life.weight_offsets.emplace(weight.ref, std::pair{weight.parts.front().second, offset})
            .second);
    life.weight_bytes = offset + bytes;
  }
  life.step = std::make_unique<co::StepGraph>(std::move(*step));
  life.state_baseline.resize(life.layout.bytes);
}
}  // namespace mapped_csa

TEST(GgmlRoundedConsumerGpu, RealMappedSourceBuildsAndExecutesItsAttentionPipeline) {
  namespace mc = mapped_csa;
  namespace pr = llmp::providers;
  namespace en = llmp::engine;
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available);
  ASSERT_GE(*available, mc::kHost + rounded_corridor::kDevice + (6ULL << 30));
  std::unique_ptr<mc::Owner, mc::Retire> life(new mc::Owner);
  const pl::StepShape shape{.phase = pl::Phase::kScore,
                            .segments = {{.rows = 4, .past = 0, .read_cells = 256, .outputs = 4}},
                            .read_align = 256};
  ASSERT_LE(ggml_tensor_overhead(), 512U);
  mc::Describe(*life, shape);
  ASSERT_FALSE(HasFatalFailure());
  auto execution = pr::cuda::OpenDeviceExecution(0);
  ASSERT_TRUE(execution);
  life->execution = std::move(*execution);
  auto stream = life->execution->CreateStream();
  ASSERT_TRUE(stream);
  life->stream = *stream;
  life->streaming = true;
  auto facts = pr::QueryDeviceFacts(0);
  ASSERT_TRUE(facts);
  const auto workspace = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
  ASSERT_LE(workspace.value(),
            rounded_corridor::kDevice - rounded_corridor::kPool - mc::kPacked - (32U << 20));
  const auto workspace_address = life->Allocate(workspace.value());
  ASSERT_NE(workspace_address, 0U);
  auto cublas = kg::CublasHandle::Create(0, *life->execution, life->stream,
                                         {.base = workspace_address, .size = workspace});
  ASSERT_TRUE(cublas);
  life->cublas = std::move(*cublas);
  auto backend = kg::GgmlBackend::Create(0, *life->execution, life->stream, nullptr,
                                         {.lower = {.index_scratch_limit = 1U << 20}});
  ASSERT_TRUE(backend);
  life->backend = std::move(*backend);
  life->metadata_catalog = std::make_shared<llmp::catalog::Catalog>();
  const auto metadata_domain = life->metadata_catalog->AddDomain("corridor host inventory");
  auto paid = llmp::engine::ExecutionMetadataAllowance::Admit(
      *life->metadata_catalog, metadata_domain, 16U << 20U, *available, 6ULL << 30U);
  ASSERT_TRUE(paid) << (paid ? "" : paid.error());
  life->metadata_paid = *paid;
  // The existing complete 64MiB host envelope includes this finite category.
  const auto charge = [paid = *paid, keep = life->metadata_catalog](std::uint64_t bytes) {
    (void)keep;
    return paid->Charge(bytes);
  };
  const auto refund = [paid = *paid, keep = life->metadata_catalog](std::uint64_t bytes) {
    (void)keep;
    EXPECT_LE(bytes, paid->used());
    paid->Release(bytes);
  };
  ASSERT_TRUE(life->backend->FundConstruction(charge, refund));
  ASSERT_TRUE(life->backend->Declare({}, workspace.value(), charge, refund));
  ASSERT_TRUE(life->backend->LendCublas(life->cublas.get()));
  ASSERT_TRUE(life->backend->Backend().owner->Ready());
  ASSERT_LE(life->backend->persistent_bytes(), 32U << 20);
  life->selected_state.bytes = life->backend->persistent_bytes();
  life->selected_state.base =
      life->selected_state.bytes ? life->Allocate(life->selected_state.bytes) : 0;
  const auto pool = life->Allocate(rounded_corridor::kPool);
  ASSERT_NE(pool, 0U);
  auto launch = life->backend->Launch(
      {.base = pool, .size = llmp::base::Bytes(rounded_corridor::kPool)}, life->selected_state);
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  life->launch = std::move(*launch);
  life->packed = life->Allocate(mc::kPacked);
  ASSERT_NE(life->packed, 0U);
  life->activation = life->packed + mc::kActivationOffset;
  auto pinned = pr::AllocatePinned(rounded_corridor::kPinned);
  ASSERT_TRUE(pinned);
  life->pinned = *pinned;
  auto submission = life->execution->Submission(life->stream);
  ASSERT_TRUE(submission);
  for (const auto& weight : life->step->weights) {
    const auto& desc = life->step->graph.tensor(weight.tensor);
    const auto bytes = mc::Bytes(desc);
    const auto& name = weight.parts.front().first;
    const bool norm = name.find("norm") != std::string::npos;
    if (desc.format.encoding == Encoding::kF32) {
      auto* data = static_cast<float*>(life->pinned);
      for (std::size_t i = 0; i < bytes / 4; ++i)
        data[i] = norm ? 1.0F : static_cast<float>(static_cast<int>(i % 7) - 3) / 128.0F;
    } else {
      ASSERT_EQ(desc.format.encoding, Encoding::kF16);
      auto* data = static_cast<std::uint16_t*>(life->pinned);
      for (std::size_t i = 0; i < bytes / 2; ++i)
        data[i] = static_cast<std::uint16_t>(0x2000U + (i % 29) * 16U);
    }
    ASSERT_TRUE(life->Copy(
        *submission,
        reinterpret_cast<void*>(life->packed + life->weight_offsets.at(weight.ref).second),
        life->pinned, bytes, pr::CopyKind::kHostToDevice));
    ASSERT_TRUE(life->Finish());
  }
  auto domain = en::NumericalDomain::Create({{.stage = 0,
                                              .name = life->spec.stages.front().name,
                                              .uses = {en::NumericalUse::kScore},
                                              .phases = {{.phase = pl::Phase::kScore,
                                                          .max_owners = 1,
                                                          .max_owner_rows = 4,
                                                          .max_total_rows = 4}}}});
  ASSERT_TRUE(domain) << domain.error();
  auto source = en::ComponentStepSource::Create(
      {.spec = &life->spec,
       .bindings = {.tensors = life->tensors.tensors, .state = &life->layout},
       .bound_digest = life->spec.Digest(),
       .slots = 1,
       .policy = pl::Policy{.enabled = {"rounded_indexer"}, .disabled = {"rounded_index_mask"}},
       .backend = life->backend->Backend(),
       .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain))});
  ASSERT_TRUE(source) << source.error();
  life->source = std::move(*source);
  ASSERT_TRUE(life->source->PrepareExecution());
  ASSERT_TRUE(life->source->CheckExecutionUse(en::NumericalUse::kScore));
  ASSERT_FALSE(life->source->CheckExecutionUse(en::NumericalUse::kSpeculative));
  auto identity = life->source->ExecutionIdentity();
  ASSERT_TRUE(identity && identity->has_value());
  // Address lookups borrow the guarded test owner during synchronous Build.
  // The native/declaration refund callbacks above retain their actual paid
  // allowance and catalog independently; they never borrow this raw owner.
  const auto state_address = [owner = life.get()](const ir::StateRef& ref) -> std::uint64_t {
    if (ref.stage != 0 || ref.segment != 0) return 0;
    const auto at = std::ranges::find_if(owner->layout.items, [&](const auto& item) {
      return item.layer == ref.layer && item.tensor == ref.tensor;
    });
    return at == owner->layout.items.end() ? 0 : owner->packed + mc::kStateOffset + at->offset;
  };
  const kn::Addresses addresses{
      .weight = [owner = life.get()](const ir::WeightRef& ref,
                                     llmp::format::PartRole role) -> std::uint64_t {
        const auto at = owner->weight_offsets.find(ref);
        return at == owner->weight_offsets.end() || at->second.first != role
                   ? 0
                   : owner->packed + at->second.second;
      },
      .weight_rows = {},
      .state = state_address,
      .save = {},
      .activations = life->activation,
      .activation_bytes = mc::kActivationBytes,
      .activation_alignment = 256,
      .state_alignment = [state_address](const ir::StateRef& ref) -> std::uint64_t {
        const auto address = state_address(ref);
        return address != 0 ? std::gcd(std::uint64_t{256}, address) : 0;
      }};
  auto made = life->source->Build(shape, addresses);  // The literal owned lower callback.
  ASSERT_TRUE(made) << made.error();
  life->program = std::move(*made);
  ASSERT_LE(life->program->host_bytes(), mc::kRetained);
  ASSERT_LE(life->program->scratch_bytes(), mc::kActivationBytes);
  ASSERT_LE(life->program->pool_bytes(), rounded_corridor::kPool);
  ASSERT_LE(life->program->launches(), 4096U);
  ASSERT_TRUE(life->program->capturable());
  ASSERT_TRUE(life->program->Bind(*life->launch));
  const auto* bound = life->program->bound();
  ASSERT_NE(bound, nullptr);
  for (auto expected : std::array<std::string_view, 3>{
           kg::kRoundedIndexScoreName, kg::kVisibleTopKName, kg::kSelectedMaskName}) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < bound->size(); ++i) count += bound->at(i).name == expected;
    ASSERT_EQ(count, 1U) << expected;
  }
  ASSERT_EQ(life->program->outputs().size(), 1U);
  ASSERT_EQ(life->program->outputs().front().bytes, 160U * sizeof(float));
  for (std::uint32_t revision = 0; revision < 2; ++revision) {
    const std::array<std::int32_t, 4> tokens{1 + static_cast<int>(revision), 3, 5, 7};
    for (unsigned pass = 0; pass < 2; ++pass) {
      ASSERT_TRUE(life->Finish());
      std::memset(life->pinned, 0, life->layout.bytes);
      ASSERT_TRUE(life->Copy(*submission, reinterpret_cast<void*>(life->packed + mc::kStateOffset),
                             life->pinned, life->layout.bytes, pr::CopyKind::kHostToDevice));
      ASSERT_TRUE(life->Finish());
      for (const auto& slot : life->program->inputs()) {
        ASSERT_LE(slot.bytes, rounded_corridor::kPinned);
        if (slot.kind == kn::InputKind::kMask) {
          // Same typed cell-position semantics as ModelRunner's host stager.
          ASSERT_EQ(slot.segment, 0U);
          ASSERT_FALSE(slot.ahead);
          ASSERT_FALSE(slot.cells.frozen);
          co::StepInput cells{.kind = co::InputKind::kCellPositions,
                              .segment = slot.segment,
                              .cells = {.capacity = slot.cells.capacity,
                                        .ring = slot.cells.ring,
                                        .frozen = slot.cells.frozen,
                                        .fixed = slot.cells.fixed,
                                        .block = slot.cells.block,
                                        .per_row = slot.cells.per_row,
                                        .first_row = slot.cells.first_row,
                                        .positions = slot.cells.positions,
                                        .recent = slot.cells.recent},
                              .read = slot.read};
          auto held = co::InputValues(cells, shape, {});
          ASSERT_TRUE(held) << held.error();
          ASSERT_EQ(held->size(), slot.read);
          ASSERT_GE(slot.mask_rows, 4U);
          ASSERT_EQ(slot.bytes, slot.read * slot.mask_rows * sizeof(std::uint16_t));
          auto* mask = static_cast<std::uint16_t*>(life->pinned);
          std::fill_n(mask, slot.bytes / sizeof(std::uint16_t), 0xfc00U);
          for (std::uint32_t row = 0; row < 4; ++row) {
            const std::int64_t position = shape.segments.front().past + row;
            for (std::uint64_t cell = 0; cell < slot.read; ++cell)
              if ((*held)[cell] >= 0 && (*held)[cell] <= position &&
                  (slot.window == 0 || position - (*held)[cell] < slot.window))
                mask[row * slot.read + cell] = 0;
          }
          ASSERT_TRUE(life->Copy(*submission, reinterpret_cast<void*>(slot.device_address),
                                 life->pinned, slot.bytes, pr::CopyKind::kHostToDevice));
          ASSERT_TRUE(life->Finish());
          continue;
        }
        const auto in = std::ranges::find(life->step->inputs, slot.tensor, &co::StepInput::tensor);
        ASSERT_NE(in, life->step->inputs.end());
        ASSERT_LE(slot.bytes, rounded_corridor::kPinned);
        const auto encoding = life->step->graph.tensor(in->tensor).format.encoding;
        if (in->kind == co::InputKind::kHadamard) {
          const auto values = co::HadamardValues(in->read);
          ASSERT_EQ(slot.bytes, values.size() * sizeof(float));
          std::memcpy(life->pinned, values.data(), slot.bytes);
        } else {
          auto values = co::InputValues(*in, shape, tokens);
          ASSERT_TRUE(values) << values.error();
          if (encoding == Encoding::kI64) {
            ASSERT_EQ(slot.bytes, values->size() * 8);
            std::memcpy(life->pinned, values->data(), slot.bytes);
          } else {
            ASSERT_EQ(encoding, Encoding::kI32);
            ASSERT_EQ(slot.bytes, values->size() * 4);
            auto* out = static_cast<std::int32_t*>(life->pinned);
            for (std::size_t i = 0; i < values->size(); ++i) {
              ASSERT_GE((*values)[i], std::numeric_limits<std::int32_t>::min());
              ASSERT_LE((*values)[i], std::numeric_limits<std::int32_t>::max());
              out[i] = static_cast<std::int32_t>((*values)[i]);
            }
          }
        }
        ASSERT_TRUE(life->Copy(*submission, reinterpret_cast<void*>(slot.device_address),
                               life->pinned, slot.bytes, pr::CopyKind::kHostToDevice));
        ASSERT_TRUE(life->Finish());
      }
      if (pass == 0) {
        ASSERT_TRUE(life->program->Run(*life->launch, *submission));
      } else {
        if (!life->captured) {
          auto recorded = life->program->Record(*life->launch, *submission, [&] {
            return life->program->Run(*life->launch, *submission);
          });
          ASSERT_TRUE(recorded);
          ASSERT_GT((*recorded)->nodes(), 0U);
          ASSERT_LE((*recorded)->nodes(), 4096U);
          life->captured = std::move(*recorded);
        }
        ASSERT_TRUE(life->launch->Launch(*submission, *life->captured));
      }
      ASSERT_TRUE(life->Finish());
      const auto& output = life->program->outputs().front();
      ASSERT_TRUE(life->Copy(*submission, life->pinned,
                             reinterpret_cast<void*>(output.device_address), output.bytes,
                             pr::CopyKind::kDeviceToHost));
      ASSERT_TRUE(life->Finish());
      const auto* rows = static_cast<const std::uint32_t*>(life->pinned);
      for (std::size_t i = 0; i < 160; ++i) {
        ASSERT_TRUE(std::isfinite(std::bit_cast<float>(rows[i])));
        if (pass == 0)
          life->baseline[i] = rows[i];
        else
          ASSERT_EQ(rows[i], life->baseline[i]);
      }
      ASSERT_TRUE(life->Copy(*submission, life->pinned,
                             reinterpret_cast<void*>(life->packed + mc::kStateOffset),
                             life->layout.bytes, pr::CopyKind::kDeviceToHost));
      ASSERT_TRUE(life->Finish());
      if (pass == 0)
        std::memcpy(life->state_baseline.data(), life->pinned, life->layout.bytes);
      else
        ASSERT_EQ(std::memcmp(life->state_baseline.data(), life->pinned, life->layout.bytes), 0);
    }
  }
  EXPECT_EQ(life->source->ExecutionIdentity(), identity);
  RecordProperty("source_rows_compared", 8);
  RecordProperty("source_head_values_compared", 320);
  RecordProperty("source_state_bytes", std::to_string(life->layout.bytes));
  RecordProperty("source_programs", 1);
  RecordProperty("source_primitive_nodes", std::to_string(life->step->graph.nodes().size()));
  RecordProperty("source_primitive_tensors", std::to_string(life->step->graph.tensor_count()));
  RecordProperty("source_bound_steps", std::to_string(bound->size()));
  RecordProperty("source_weight_bytes", std::to_string(life->weight_bytes));
  RecordProperty("source_host_inventory", std::to_string(mc::kHost));
  RecordProperty("source_device_inventory", std::to_string(rounded_corridor::kDevice));
}

}  // namespace
