// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// End-to-end qualification of the owned comparison-pair coordinator. This is
// correctness/lifetime coverage only; duration API controls produce no rank data.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "components/comparison_boundary.h"
#include "components/graph_io.h"
#include "components/step.h"
#include "engine/comparison_pair.h"
#include "engine/graph_plan.h"
#include "engine/paged_node.h"
#include "engine/runner_resources.h"
#include "engine/step_inputs.h"
#include "execution/registry.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "kernels/common/lower.h"
#include "kernels/common/recording_launch.h"
#include "kernels/ggml/step_program.h"
#include "kernels/reference/exec.h"
#include "model/state_contract.h"
#include "planner/alternative.h"
#include "planner/policy.h"
#include "platform/host_probe.h"
#include "spec/spec.h"

namespace {
namespace co = llmp::components;
namespace en = llmp::engine;
namespace execution = llmp::execution;
namespace fmt = llmp::format;
namespace ir = llmp::ir;
namespace kn = llmp::kernels;
namespace kg = llmp::kernels::ggml;
namespace model = llmp::model;
namespace pl = llmp::planner;
namespace ref = llmp::kernels::reference;
namespace spec = llmp::spec;

constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::uint64_t kArenaBytes = 1ULL << 20U;
constexpr std::uint64_t kPoolBytes = 1ULL << 20U;

fmt::TensorFormat F32(std::initializer_list<std::uint64_t> dims) {
  return {
      .encoding = fmt::Encoding::kF32, .layout = fmt::Layout::kDense, .shape = *fmt::ShapeOf(dims)};
}

struct TwoLayerTokenSource {
  spec::ModelSpec model;
  std::vector<co::BoundTensor> tensors;
  model::StateLayoutDerived state;
  pl::StepShape shape;
  co::StepGraph step;
};

spec::Param SpecParam(std::string name, spec::ParamValue value) {
  return {.name = std::move(name), .value = std::move(value)};
}

spec::ComponentRef DenseAttention(std::uint64_t width, std::uint32_t layer) {
  const auto suffix = std::to_string(layer);
  spec::ComponentRef attention{.kind = "attention.dense"};
  attention.params = {SpecParam("placement", std::string("pre")),
                      SpecParam("eps", 1.0e-6),
                      SpecParam("heads", std::int64_t{1}),
                      SpecParam("kv_heads", std::int64_t{1}),
                      SpecParam("key_dim", static_cast<std::int64_t>(width)),
                      SpecParam("value_dim", static_cast<std::int64_t>(width)),
                      SpecParam("scale", 1.0 / std::sqrt(static_cast<double>(width))),
                      SpecParam("softcap", 0.0),
                      SpecParam("window", std::int64_t{0}),
                      SpecParam("rotation", std::string("neox")),
                      SpecParam("rope_dims", static_cast<std::int64_t>(width)),
                      SpecParam("rope_base", 10000.0),
                      SpecParam("rope_scale", 1.0)};
  attention.weights = {{.role = "k", .tensor = "attn_k_" + suffix},
                       {.role = "norm", .tensor = "attn_norm_" + suffix},
                       {.role = "output", .tensor = "attn_output_" + suffix},
                       {.role = "q", .tensor = "attn_q_" + suffix},
                       {.role = "v", .tensor = "attn_v_" + suffix}};
  return attention;
}

std::expected<TwoLayerTokenSource, std::string> MakeTwoLayerTokenSource() {
  // Keep the already-qualified dense-attention shape; this fixture adds a
  // second occurrence without introducing a new kernel specialization.
  constexpr std::uint64_t width = 64;
  spec::ComponentRef embedding{.kind = "embedding.token"};
  embedding.weights = {{.role = "table", .tensor = "token_embd"}};
  spec::Stage stage{.name = "target", .context = 256, .embedding = std::move(embedding)};
  stage.stream.width = width;
  stage.layers = {{.components = {DenseAttention(width, 0)}},
                  {.components = {DenseAttention(width, 1)}}};
  stage.heads.final_norm = spec::ComponentRef{.kind = "norm.rms"};
  stage.heads.final_norm->params = {SpecParam("eps", 1.0e-6)};
  stage.heads.final_norm->weights = {{.role = "weight", .tensor = "output_norm"}};
  stage.heads.output = spec::ComponentRef{.kind = "head.linear"};
  stage.heads.output->weights = {{.role = "weight", .tensor = "output_head"}};
  spec::ModelSpec definition{.stages = {std::move(stage)}};
  spec::Canonicalize(definition);
  std::vector<co::BoundTensor> tensors;
  tensors.push_back({.name = "token_embd", .format = F32({width, 8})});
  for (std::uint32_t layer = 0; layer < 2; ++layer) {
    const auto suffix = std::to_string(layer);
    tensors.push_back({.name = "attn_k_" + suffix, .format = F32({width, width})});
    tensors.push_back({.name = "attn_norm_" + suffix, .format = F32({width})});
    tensors.push_back({.name = "attn_output_" + suffix, .format = F32({width, width})});
    tensors.push_back({.name = "attn_q_" + suffix, .format = F32({width, width})});
    tensors.push_back({.name = "attn_v_" + suffix, .format = F32({width, width})});
  }
  tensors.push_back({.name = "output_head", .format = F32({width, 8})});
  tensors.push_back({.name = "output_norm", .format = F32({width})});
  auto valid = co::ValidateSpec(definition, tensors);
  if (!valid) return std::unexpected(valid.error());
  auto state = co::StageLayout(definition, 0, 256, 1);
  if (!state) return std::unexpected(state.error());
  pl::StepShape shape{
      .stage = 0,
      .phase = pl::Phase::kPrefill,
      .segments = {{.slot = 0, .rows = 1, .past = 0, .read_cells = 256, .outputs = 1}}};
  auto step = co::BuildStep(definition, 0, shape, {.tensors = tensors, .state = &*state});
  if (!step) return std::unexpected(step.error());
  return TwoLayerTokenSource{.model = std::move(definition),
                             .tensors = std::move(tensors),
                             .state = std::move(*state),
                             .shape = std::move(shape),
                             .step = std::move(*step)};
}

class PaidPairFunder final : public ir::StorageFunder {
 public:
  explicit PaidPairFunder(std::shared_ptr<en::ExecutionMetadataAllowance> paid)
      : paid_(std::move(paid)) {}
  bool Charge(std::uint64_t bytes) const override { return paid_->Charge(bytes); }
  void Release(std::uint64_t bytes) const override { paid_->Release(bytes); }

 private:
  std::shared_ptr<en::ExecutionMetadataAllowance> paid_;
};

struct CudaBuffer {
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  void* pointer = nullptr;
  ~CudaBuffer() {
    if (pointer) EXPECT_EQ(cudaFree(pointer), cudaSuccess);
  }
  bool Allocate(std::size_t bytes) { return cudaMalloc(&pointer, bytes) == cudaSuccess; }
  void Free() {
    if (pointer) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
      pointer = nullptr;
    }
  }
  std::uint64_t address() const { return reinterpret_cast<std::uintptr_t>(pointer); }
};

struct GpuLife final : en::PagedModel {
  en::PagedNode node{{.compute_streams = 1, .slots = 1}};
  en::RunnerResources resources{node, 0, 0};
  std::unique_ptr<kg::GgmlBackend> backend;
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  CudaBuffer source_weight;
  CudaBuffer model_weights;
  CudaBuffer arena;
  CudaBuffer pool;
  CudaBuffer persistent;
  CudaBuffer model_state;
  std::unique_ptr<kn::StepLaunch> launch;
  std::unique_ptr<en::ComparisonPair> pair;
  std::unique_ptr<kn::StepProgram> selected_program;
  std::unique_ptr<kn::Recorded> recording;
  GpuLife* quarantine_next = nullptr;
  bool opened = false;
  llmp::catalog::Closure empty;

  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return empty; }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override { return {}; }
  en::Status RetireAfterFence() override {
    recording.reset();
    selected_program.reset();
    pair.reset();
    launch.reset();
    if (backend) backend->LaunchReleased();
    persistent.Free();
    model_state.Free();
    pool.Free();
    arena.Free();
    source_weight.Free();
    model_weights.Free();
    return {};
  }
  en::Status Release() override {
    backend.reset();
    std::vector<std::string> errors;
    resources.Release(errors);
    if (!errors.empty()) return std::unexpected(errors.front());
    opened = false;
    return {};
  }

  static GpuLife*& QuarantineHead() {
    static GpuLife* head = nullptr;
    return head;
  }
  static void Quarantine(std::unique_ptr<GpuLife> owner) {
    auto* retained = owner.release();
    retained->quarantine_next = QuarantineHead();
    QuarantineHead() = retained;
  }

  static bool Retire(std::unique_ptr<GpuLife>& owner, std::string_view context) {
    if (!owner) return true;
    if (!owner->opened) {
      owner.reset();
      return true;
    }
    const std::array<en::PagedModel*, 1> models{owner.get()};
    auto retired = owner->node.TearDown(models);
    if (!retired) {
      ADD_FAILURE() << context << " could not prove completion; preserving complete GPU owner: "
                    << retired.error();
      Quarantine(std::move(owner));
      return false;
    }
    owner.reset();
    return true;
  }
};

// Concrete enclosing owner: the backend/device, GPU allocations and launch
// all outlive the private semantic pair and its bound programs/recordings.
struct GpuPairOwner {
  explicit GpuPairOwner(std::unique_ptr<GpuLife> life_in) : life(std::move(life_in)) {}
  GpuPairOwner(const GpuPairOwner&) = delete;
  GpuPairOwner& operator=(const GpuPairOwner&) = delete;
  ~GpuPairOwner() { GpuLife::Retire(life, "pair owner retirement"); }

  std::unique_ptr<GpuLife> life;
};

class GgmlComparisonPairGpu : public testing::Test {
 protected:
  struct RecipeCase {
    std::uint64_t width = 16;
    bool pure_prefix = true;
    bool measure_capture = false;
    std::uint32_t repetitions = 1;
    bool internal_output = true;
    bool ready_boundary_prefix = false;
    std::uint16_t component = 0;
    std::uint32_t layer = 0;
    std::string part_name = "weight";
    float prefix_epsilon = 1.0e-6F;
    float epsilon = 1.0e-6F;
    std::uint64_t parent_extra = 0;
    std::uint64_t view_offset_extra = 0;
    std::uint64_t alignment = 16;
    std::uint32_t allocation_class_x = 1;
    std::uint32_t allocation_class_w = 2;
  };

  void RunPairQualification(bool select_recipe);
  void RunPairQualification(bool select_recipe, const RecipeCase& recipe_case,
                            std::vector<execution::RecipeComparisonKey>* described_keys,
                            std::vector<std::uint64_t>* source_identities,
                            bool measurement_bind = false);

  void InitializeLife() {
    ASSERT_TRUE(GpuLife::Retire(life_, "retiring previous fixture case"));
    life_ = std::make_unique<GpuLife>();
    auto opened = life_->node.Open();
    ASSERT_TRUE(opened) << (opened ? "" : opened.error());
    life_->opened = true;
    auto available = llmp::platform::AvailableMemoryBytes();
    ASSERT_TRUE(available);
    auto admitted = en::ExecutionMetadataAllowance::Admit(
        life_->node.catalog(), life_->node.domain(), 16 * kMiB, *available, 6ULL << 30U);
    ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
    life_->paid = std::move(*admitted);
    auto backend =
        kg::GgmlBackend::Create(0, life_->node.execution(), life_->node.stream(0), nullptr);
    ASSERT_TRUE(backend) << (backend ? "" : backend.error());
    life_->backend = std::move(*backend);
    const auto charge = [paid = life_->paid](std::uint64_t bytes) { return paid->Charge(bytes); };
    const auto release = [paid = life_->paid](std::uint64_t bytes) { paid->Release(bytes); };
    ASSERT_TRUE(life_->resources.OpenLibrary("owned comparison-pair GPU qualification"));
    ASSERT_TRUE(life_->backend->FundConstruction(charge, release));
    ASSERT_TRUE(life_->backend->Declare({}, life_->resources.library_bytes(), charge, release));
    ASSERT_TRUE(life_->backend->LendLibrary(&life_->resources.library()));
  }

  void SetUp() override { ASSERT_NO_FATAL_FAILURE(InitializeLife()); }

  void TearDown() override { GpuLife::Retire(life_, "fixture teardown"); }

  std::unique_ptr<GpuLife> life_;
};

std::optional<std::uint64_t> PlacementOffset(const pl::StepPlan& plan, ir::TensorId original) {
  if (!original.valid() || original.v >= plan.tensors.size()) return std::nullopt;
  const auto mapped = plan.tensors[original.v];
  if (!mapped.valid()) return std::nullopt;
  const auto root = plan.graph.root(mapped);
  const auto found = std::ranges::find_if(plan.placement.offsets, [&](const auto& item) {
    return plan.graph.root(item.first) == root;
  });
  if (found == plan.placement.offsets.end()) return std::nullopt;
  return found->second;
}

std::expected<execution::Registry, execution::PlanRejection> TestOnlyRecipeRegistry(
    const execution::Registry& source, const execution::RecipeComparisonKey& key) {
  std::vector<execution::Implementation> declarations;
  declarations.reserve(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) declarations.push_back(source.at(i));
  auto ranks = source.ranks();
  ranks.recipe_comparisons.clear();
  ranks.recipe_comparisons.push_back(
      {.key = key,
       .incumbent_ns = 120,
       .alternative_ns = 100,
       .samples = 3,
       .protocol = llmp::base::Sha256{}.Update("test-only-not-a-measurement").Finish(),
       .receipt = llmp::base::Sha256{}.Update("test-only-recipe-row").Finish(),
       .provenance = "synthetic test-only row; not a measurement",
       .shape_class = "descriptor-pair-GPU-control"});
  return execution::Registry::Create(std::move(declarations), std::move(ranks));
}

void GgmlComparisonPairGpu::RunPairQualification(bool select_recipe) {
  RunPairQualification(select_recipe, RecipeCase{}, nullptr, nullptr, false);
}

void GgmlComparisonPairGpu::RunPairQualification(
    bool select_recipe, const RecipeCase& recipe_case,
    std::vector<execution::RecipeComparisonKey>* described_keys,
    std::vector<std::uint64_t>* source_identities, bool measurement_bind) {
  auto owner = std::make_unique<GpuPairOwner>(std::move(life_));
  auto& life = *owner->life;
  auto backend_recipe = life.backend->Backend();
  const auto& recipe = backend_recipe.owner ? backend_recipe.owner->backend() : backend_recipe;
  ASSERT_NE(recipe.registry, nullptr);

  ir::Builder builder;
  builder.SetComponent(recipe_case.component);
  builder.SetOwner(0);
  const auto x = builder.Input(F32({recipe_case.width, 1}), "comparison-pair-x");
  const ir::WeightRef weight_ref{.stage = 0,
                                 .component = recipe_case.component,
                                 .role = 0,
                                 .layer = recipe_case.layer,
                                 .expert_array = false};
  const auto w = builder.Weight(F32({recipe_case.width}), weight_ref, recipe_case.part_name);
  ir::TensorId prefix = x;
  if (recipe_case.ready_boundary_prefix) {
    builder.SetJoined();
    prefix = builder.RmsNorm(x, recipe_case.prefix_epsilon);
  } else if (recipe_case.pure_prefix) {
    prefix = builder.RmsNorm(x, recipe_case.prefix_epsilon);
  }
  const auto normalized = builder.RmsNorm(prefix, recipe_case.epsilon);
  const auto scaled = builder.Mul(normalized, w);
  if (recipe_case.ready_boundary_prefix) builder.SetOwner(0);
  // Concat preserves the larger full-program output; Relu keeps its suffix
  // outside the typed comparison schema while ordinary execution remains valid.
  const auto suffix = (recipe_case.ready_boundary_prefix || recipe_case.internal_output)
                          ? builder.Unary(builder.Concat(scaled, prefix, 0), ir::UnaryFn::kRelu)
                          : scaled;
  builder.Output(suffix);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  const std::size_t expected_nodes =
      recipe_case.ready_boundary_prefix
          ? 7U
          : (recipe_case.pure_prefix && recipe_case.internal_output
                 ? 7U
                 : (recipe_case.internal_output ? 6U : (recipe_case.pure_prefix ? 5U : 4U)));
  ASSERT_EQ(graph->nodes().size(), expected_nodes);
  const auto x_bytes = ir::StorageBytes(graph->tensor(x));
  const auto w_bytes = ir::StorageBytes(graph->tensor(w));
  ASSERT_TRUE(x_bytes && w_bytes);

  auto funder = std::make_shared<PaidPairFunder>(life.paid);
  auto funded_graph = graph->Clone(funder);
  ASSERT_TRUE(funded_graph) << (funded_graph ? "" : funded_graph.error().detail);
  const auto original_identity = funded_graph->identity();
  const bool component_input_domain =
      select_recipe || measurement_bind || recipe_case.ready_boundary_prefix;
  const auto ordinary_input_kind =
      component_input_domain ? co::InputKind::kStream : co::InputKind::kFeatures;
  co::StepGraph source{
      .graph = std::move(*funded_graph),
      .inputs = {{.tensor = x, .kind = ordinary_input_kind, .component = recipe_case.component}},
      .weights = {{.tensor = w,
                   .ref = weight_ref,
                   .parts = {{recipe_case.part_name, fmt::PartRole::kData}}}},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  pl::StepShape shape{.stage = 0, .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  CudaBuffer& w_parent = life.source_weight;
  const auto parent_bytes = std::max(*x_bytes, *w_bytes) + recipe_case.parent_extra + 512;
  ASSERT_TRUE(w_parent.Allocate(static_cast<std::size_t>(parent_bytes)));
  const auto aligned_offset = [&](const CudaBuffer& buffer) {
    const auto address = buffer.address();
    const auto remainder = address % recipe_case.alignment;
    return remainder == 0 ? std::uint64_t{0} : recipe_case.alignment - remainder;
  };
  const auto w_offset = aligned_offset(w_parent) + recipe_case.view_offset_extra;
  if (recipe_case.view_offset_extra == 0)
    EXPECT_EQ(w_offset, 0U);
  else
    ASSERT_GE(w_offset, recipe_case.view_offset_extra);
  ASSERT_LE(w_offset + *w_bytes, parent_bytes);
  const auto w_address = w_parent.address() + w_offset;
  std::vector<co::ComparisonExternalBinding> external{
      {.original_root = x,
       .allocation_class = recipe_case.allocation_class_x,
       .allocation_bytes = *x_bytes,
       .offset = 0,
       .bytes = *x_bytes,
       .alignment = 16},
      {.original_root = w,
       .part = fmt::PartRole::kData,
       .allocation_class = recipe_case.allocation_class_w,
       .allocation_bytes = parent_bytes,
       .offset = w_offset,
       .bytes = *w_bytes,
       .alignment = recipe_case.alignment}};
  CudaBuffer& arena = owner->life->arena;
  CudaBuffer& pool = owner->life->pool;
  CudaBuffer& persistent = owner->life->persistent;
  auto& pair = owner->life->pair;
  auto& launch = owner->life->launch;
  auto made = en::ComparisonPair::Create(
      funder, std::move(source), std::move(shape), std::move(external), 0, 0,
      component_input_domain ? en::ComparisonInputDomain::kComponentPorts
                             : en::ComparisonInputDomain::kTokenStep);
  ASSERT_TRUE(made) << (made ? "" : made.error().detail);
  pair = std::move(*made);
  const auto* pattern = pl::FindPattern("rms_norm_mul");
  ASSERT_NE(pattern, nullptr);
  pl::ComparisonSiteSelector selector{
      .source_graph_identity = original_identity,
      .kind = pl::ComparisonSiteKind::kRewrite,
      .pattern = "rms_norm_mul",
      .version = pattern->version,
      .member_origins =
          recipe_case.ready_boundary_prefix
              ? std::vector<std::vector<ir::NodeId>>{{ir::NodeId{3}}, {ir::NodeId{4}}}
              : std::vector<std::vector<ir::NodeId>>{{ir::NodeId{recipe_case.pure_prefix ? 3U
                                                                                         : 2U}},
                                                     {ir::NodeId{recipe_case.pure_prefix ? 4U
                                                                                         : 3U}}},
      .member_owner_segments = {},
      .boundary_ports = {}};
  const pl::PlanOptions options{.capturable = true, .within = recipe.within};
  auto planned = pair->Plan(selector, *recipe.registry, recipe.device, {}, options);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(pair->plan(en::ComparisonPairArm::kIncumbent)->plan.applied.size(), 1U);
  EXPECT_EQ(pair->plan(en::ComparisonPairArm::kIncumbent)->plan.applied.front().pattern,
            "rms_norm_mul");
  EXPECT_TRUE(
      std::ranges::none_of(pair->plan(en::ComparisonPairArm::kAlternative)->plan.applied,
                           [](const auto& item) { return item.pattern == "rms_norm_mul"; }));

  ASSERT_TRUE(arena.Allocate(kArenaBytes));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life.backend->persistent_bytes();
  if (persistent_bytes != 0) ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  auto made_launch =
      life.backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                           {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(made_launch) << (made_launch ? "" : made_launch.error());
  launch = std::move(*made_launch);

  std::array<std::vector<kn::ComparisonBindingRange>, 2> binding_storage;
  for (std::size_t arm = 0; arm < binding_storage.size(); ++arm) {
    const auto which =
        arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
    const auto* retained = pair->plan(which);
    ASSERT_NE(retained, nullptr);
    const auto input_offset = PlacementOffset(retained->plan, x);
    ASSERT_TRUE(input_offset);
    ASSERT_LE(*input_offset, kArenaBytes);
    ASSERT_LE(*x_bytes, kArenaBytes - *input_offset);
    binding_storage[arm].push_back({.original_root = x,
                                    .allocation_class = recipe_case.allocation_class_x,
                                    .allocation_bytes = *x_bytes,
                                    .offset = 0,
                                    .bytes = *x_bytes,
                                    .alignment = 16,
                                    .base = arena.address() + *input_offset});
    binding_storage[arm].push_back({.original_root = w,
                                    .part = fmt::PartRole::kData,
                                    .allocation_class = recipe_case.allocation_class_w,
                                    .allocation_bytes = parent_bytes,
                                    .offset = w_offset,
                                    .bytes = *w_bytes,
                                    .alignment = recipe_case.alignment,
                                    .base = w_parent.address()});
  }
  const std::array<std::span<const kn::ComparisonBindingRange>, 2> binding_views{
      binding_storage[0], binding_storage[1]};
  ASSERT_TRUE(pair->InstallBindings(binding_views));

  kn::Addresses addresses{
      .weight =
          [weight_ref, w_address](const ir::WeightRef& ref, fmt::PartRole part) {
            return ref == weight_ref && part == fmt::PartRole::kData ? w_address : 0;
          },
      .weight_rows = {},
      .state = {},
      .save = {},
      .activations = arena.address(),
      .activation_bytes = kArenaBytes,
      .activation_alignment = 256,
      .state_alignment = {},
      .expert_part = {},
      .weight_alignment = {},
      .weight_range =
          [weight_ref, w_address, parent_bytes, w_offset, alignment = recipe_case.alignment](
              const ir::WeightRef& ref, fmt::PartRole part) -> std::optional<kn::BoundedAddress> {
        if (ref != weight_ref || part != fmt::PartRole::kData || w_offset > parent_bytes)
          return std::nullopt;
        return kn::BoundedAddress{
            .base = w_address, .bytes = parent_bytes - w_offset, .alignment = alignment};
      }};
  if (select_recipe) {
    auto source_bindings = en::ComparisonSourceBindings::Create(funder, pair->source(), 0);
    ASSERT_TRUE(source_bindings);
    auto incumbent_ranges = source_bindings->Resolve(
        pair->plan(en::ComparisonPairArm::kIncumbent)->plan, addresses, recipe.written);
    ASSERT_TRUE(incumbent_ranges) << (incumbent_ranges ? "" : incumbent_ranges.error().detail);
    auto alternative_ranges = source_bindings->Resolve(
        pair->plan(en::ComparisonPairArm::kAlternative)->plan, addresses, recipe.written);
    ASSERT_TRUE(alternative_ranges)
        << (alternative_ranges ? "" : alternative_ranges.error().detail);
    auto view = source_bindings->boundary_view();
    ASSERT_TRUE(view);
    auto source_boundary = co::MakeComparisonBoundary(*view, funder);
    ASSERT_TRUE(source_boundary) << (source_boundary ? "" : source_boundary.error().detail);
    kn::ComparisonRequest source_request{.mode = kn::ComparisonRunMode::kFull,
                                         .bindings = alternative_ranges->ranges(),
                                         .state = pair->source().state,
                                         .active_stage = 0,
                                         .expected_boundary = &*source_boundary};
    kn::LowerInput source_lower_input(pair->plan(en::ComparisonPairArm::kAlternative)->plan,
                                      co::GraphIO(pair->source()), addresses, std::nullopt,
                                      &source_request);
    auto source_lowered = life.backend->Lower(source_lower_input);
    ASSERT_TRUE(source_lowered) << (source_lowered ? "" : source_lowered.error());
  }
  std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
  for (std::size_t arm = 0; arm < programs.size(); ++arm) {
    const auto which =
        arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
    auto lower = pair->LoweringInput(which, addresses);
    ASSERT_TRUE(lower) << (lower ? "" : lower.error().detail);
    if (component_input_domain) {
      EXPECT_FALSE(lower->token.has_value());
      ASSERT_EQ(lower->io.inputs.size(), 1U);
    } else {
      EXPECT_TRUE(lower->token.has_value());
    }
    auto lowered = life.backend->Lower(*lower);
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error());
    const auto whole_recipe = (*lowered)->comparison_recipe();
    ASSERT_TRUE(whole_recipe);
    if (recipe_case.internal_output) {
      EXPECT_EQ(whole_recipe->state, kn::ComparisonProofState::kUnproved);
      EXPECT_EQ(whole_recipe->missing, kn::ComparisonMissing::kOperationSchema);
    }
    programs[arm] = std::move(*lowered);
  }
  const auto expected_alternative_identity =
      pair->plan(en::ComparisonPairArm::kAlternative)->plan.graph.identity();
  kn::StepProgram* program = nullptr;
  if (select_recipe) {
    ASSERT_TRUE(pair->InstallDescriptorPrograms(std::move(programs[0]), std::move(programs[1])));
    auto described = en::DescribeRecipeComparison(*pair, *recipe.registry);
    ASSERT_TRUE(described) << (described ? "" : described.error().detail);
    ASSERT_TRUE(*described) << "actual local RMSNorm region did not close";
    if (described_keys) described_keys->push_back(**described);
    if (source_identities) source_identities->push_back(original_identity);
    auto test_registry = TestOnlyRecipeRegistry(*recipe.registry, **described);
    ASSERT_TRUE(test_registry) << (test_registry ? "" : test_registry.error().detail);
    ASSERT_EQ(test_registry->SelectionDigest(), recipe.registry->SelectionDigest());
    const auto* synthetic_row = test_registry->FindRecipeComparison(**described);
    ASSERT_NE(synthetic_row, nullptr);
    EXPECT_EQ(synthetic_row->provenance, "synthetic test-only row; not a measurement");
    auto selected = en::SelectRecipeComparison(std::move(pair), *test_registry);
    ASSERT_TRUE(selected) << (selected ? "" : selected.error().detail);
    pair.reset();
    life.selected_program = std::move(*selected);
    program = life.selected_program.get();
    ASSERT_NE(program, nullptr);
    const auto selected_identity = program->comparison_plan_identity();
    ASSERT_TRUE(selected_identity);
    EXPECT_EQ(selected_identity->source_graph_identity, original_identity);
    EXPECT_EQ(selected_identity->planned_graph_identity, expected_alternative_identity)
        << "synthetic test-only latency row must choose the already-lowered alternative";
    const auto selected_whole = program->comparison_recipe();
    ASSERT_TRUE(selected_whole);
    if (recipe_case.internal_output) {
      EXPECT_EQ(selected_whole->state, kn::ComparisonProofState::kUnproved);
      EXPECT_EQ(selected_whole->missing, kn::ComparisonMissing::kOperationSchema);
    }
    EXPECT_EQ(program->bound(), nullptr);
    ASSERT_TRUE(program->Bind(*launch)) << "selected wrapper revalidates bound local closure";
    ASSERT_NE(program->bound(), nullptr);
  } else if (measurement_bind) {
    ASSERT_TRUE(pair->InstallDescriptorPrograms(std::move(programs[0]), std::move(programs[1])));
    const auto preparation = recipe_case.ready_boundary_prefix
                                 ? en::ComparisonRecipePreparation::kStatelessReadyBoundaryOperands
                                 : en::ComparisonRecipePreparation::
                                       kFullStatelessRestagedInputsPurePrefixCapturedFullBody;
    auto described = en::DescribeRecipeComparison(*pair, *recipe.registry, preparation);
    ASSERT_TRUE(described) << (described ? "" : described.error().detail);
    ASSERT_TRUE(*described) << "actual local RMSNorm region did not close before Bind";
    if (described_keys) described_keys->push_back(**described);
    if (source_identities) source_identities->push_back(original_identity);
    const auto incumbent_identity =
        pair->plan(en::ComparisonPairArm::kIncumbent)->plan.graph.identity();
    ASSERT_NE(incumbent_identity, 0U);
    ASSERT_TRUE(pair->BindPrograms(*launch, *launch))
        << "measurement Bind must use the already-owned lowered programs";
    auto second_description = en::DescribeRecipeComparison(*pair, *recipe.registry);
    ASSERT_FALSE(second_description) << "a measurement Bind attempt is terminal for describing";
    auto closed = en::CloseComparisonPair(std::move(pair));
    ASSERT_TRUE(closed) << (closed ? "" : closed.error().detail);
    pair = std::move(*closed);
    ASSERT_TRUE(pair->closed());
    EXPECT_EQ(pair->plan(en::ComparisonPairArm::kIncumbent)->plan.graph.identity(),
              incumbent_identity);
    EXPECT_EQ(pair->plan(en::ComparisonPairArm::kAlternative)->plan.graph.identity(),
              expected_alternative_identity);
  } else {
    for (auto& actual : programs) ASSERT_TRUE(actual->Bind(*launch));
    ASSERT_TRUE(pair->InstallPrograms(std::move(programs[0]), std::move(programs[1])));
    auto closed = en::CloseComparisonPair(std::move(pair));
    ASSERT_TRUE(closed) << (closed ? "" : closed.error().detail);
    pair = std::move(*closed);
    ASSERT_TRUE(pair->closed());
  }

  const auto* stable_owner = owner.get();
  auto stable_owner_handle = std::move(owner);
  ASSERT_EQ(stable_owner_handle.get(), stable_owner);
  auto& stable = *stable_owner_handle;
  if (select_recipe)
    ASSERT_EQ(program->comparison_plan_identity()->source_graph_identity, original_identity);

  const auto stream = life.node.execution().Submission(life.node.stream(0));
  ASSERT_TRUE(stream) << (stream ? "" : stream.error().detail);
  if (select_recipe) {
    auto recorded =
        program->Record(*launch, *stream, [&]() { return program->Run(*launch, *stream); });
    ASSERT_TRUE(recorded) << (recorded ? "" : recorded.error().detail);
    life.recording = std::move(*recorded);
  } else {
    ASSERT_NE(pair, nullptr);
    if (recipe_case.measure_capture) {
      ASSERT_TRUE(launch->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
      auto refused = launch->Capture(*stream, []() -> std::expected<void, kn::StepFailure> {
        return std::unexpected(kn::StepFailure{.detail = "intentional capture-only refusal"});
      });
      auto disabled = launch->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled);
      auto ordinary =
          launch->Capture(*stream, []() -> std::expected<void, kn::StepFailure> { return {}; });
      auto fence = life.node.execution().Record(life.node.stream(0));
      const auto completed = cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle));
      ASSERT_TRUE(fence);
      const auto status = life.node.execution().Query(*fence);
      const auto released = life.node.execution().Release(*fence);
      ASSERT_EQ(completed, cudaSuccess);
      ASSERT_TRUE(status);
      ASSERT_EQ(*status, llmp::providers::FenceState::kComplete);
      ASSERT_TRUE(released);
      ASSERT_TRUE(disabled);
      ASSERT_FALSE(refused);
      EXPECT_EQ(refused.error().error, kn::StepError::kRejected);
      ASSERT_FALSE(launch->faulted());
      ASSERT_TRUE(ordinary);
      EXPECT_EQ((*ordinary)->nodes(), 0U) << "disabled capture must not retain timing nodes";
      EXPECT_FALSE((*ordinary)->gpu_duration_ns());
    }
    for (const auto arm :
         {en::ComparisonPairArm::kIncumbent, en::ComparisonPairArm::kAlternative}) {
      ASSERT_TRUE(launch->SetCaptureMeasurement(recipe_case.measure_capture
                                                    ? kn::CaptureMeasurement::kGpuDuration
                                                    : kn::CaptureMeasurement::kDisabled));
      auto captured = pair->CaptureRegion(arm, *launch, *stream, recipe_case.repetitions);
      ASSERT_TRUE(launch->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
      ASSERT_TRUE(captured);
      const auto inventory = pair->captured_region_inventory(arm, *launch);
      ASSERT_TRUE(inventory);
      const auto body_nodes = arm == en::ComparisonPairArm::kIncumbent ? 1U : 2U;
      EXPECT_EQ(inventory->recorded_nodes,
                body_nodes * recipe_case.repetitions + (recipe_case.measure_capture ? 2U : 0U));
      EXPECT_EQ(inventory->repetitions, recipe_case.repetitions);
      kn::RecordingLaunch foreign;
      EXPECT_FALSE(pair->captured_region_duration_ns(arm, foreign));
      EXPECT_FALSE(pair->captured_region_duration_ns(arm, *launch))
          << "even completed upload is not a completed replay";
    }
  }

  ASSERT_EQ(&stable, stable_owner);
  auto reference = ref::Executor::Create(graph.value());
  ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
  std::optional<std::vector<float>> first_expected;
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    std::vector<float> host_x(recipe_case.width), host_w(recipe_case.width);
    for (std::size_t i = 0; i < host_x.size(); ++i) {
      const auto centered =
          static_cast<float>(i) - static_cast<float>(recipe_case.width - 1) / 2.0F;
      host_x[i] = centered * 0.125F + (frame == 0 ? 0.0F : centered * centered * 0.015625F);
      host_w[i] = 0.5F + static_cast<float>(i % 5) * 0.125F;
    }
    ASSERT_TRUE(reference->BindValues(x, host_x));
    ASSERT_TRUE(reference->BindValues(w, host_w));
    ASSERT_TRUE(reference->Run());
    const auto compared_tensor = select_recipe ? suffix : scaled;
    auto expected = reference->Values(compared_tensor);
    ASSERT_TRUE(expected) << (expected ? "" : expected.error().detail);
    if (frame == 0) {
      first_expected = *expected;
    } else {
      ASSERT_TRUE(first_expected);
      ASSERT_EQ(first_expected->size(), expected->size());
      EXPECT_GT(std::ranges::max(std::views::iota(std::size_t{0}, expected->size()) |
                                 std::views::transform([&](std::size_t i) {
                                   return std::abs((*expected)[i] - (*first_expected)[i]);
                                 })),
                1.0e-4F)
          << "changed-input replay must exercise a numerically distinct result";
    }

    const auto check_output = [&](std::uint64_t address, std::uint64_t bytes) {
      ASSERT_EQ(bytes, expected->size() * sizeof(float));
      std::vector<float> actual(expected->size());
      ASSERT_EQ(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(address),
                           static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]) && std::isfinite((*expected)[i]));
        EXPECT_LE(std::abs(actual[i] - (*expected)[i]),
                  1.0e-5F * (1.0F + std::abs((*expected)[i])));
      }
    };
    const auto stage_inputs = [&](std::span<const kn::InputSlot> slots) {
      ASSERT_EQ(slots.size(), 1U);
      bool found_input = false;
      for (const auto& slot : slots) {
        ASSERT_EQ(slot.tensor, x) << "unexpected bound input slot " << slot.tensor.v;
        ASSERT_EQ(slot.kind,
                  component_input_domain ? kn::InputKind::kStepBytes : kn::InputKind::kFeatures);
        ASSERT_EQ(slot.bytes, host_x.size() * sizeof(float));
        ASSERT_FALSE(found_input);
        found_input = true;
        ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(slot.device_address), host_x.data(),
                             static_cast<std::size_t>(slot.bytes), cudaMemcpyHostToDevice),
                  cudaSuccess);
      }
      ASSERT_TRUE(found_input);
    };
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(w_address), host_w.data(),
                         static_cast<std::size_t>(*w_bytes), cudaMemcpyHostToDevice),
              cudaSuccess);
    if (select_recipe) {
      ASSERT_NO_FATAL_FAILURE(stage_inputs(program->inputs()));
      ASSERT_TRUE(launch->Launch(*stream, *life.recording))
          << "complete selected schedule replay failed";
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
      bool found_output = false;
      for (const auto& output : program->outputs()) {
        if (output.tensor != suffix) continue;
        ASSERT_FALSE(found_output);
        found_output = true;
        ASSERT_NO_FATAL_FAILURE(check_output(output.device_address, output.bytes));
      }
      ASSERT_TRUE(found_output) << "complete selected program omitted the Relu suffix output";
    } else {
      ASSERT_NE(pair, nullptr);
      for (const auto arm :
           {en::ComparisonPairArm::kIncumbent, en::ComparisonPairArm::kAlternative}) {
        if (recipe_case.ready_boundary_prefix) {
          bool poisoned_prefix = false;
          bool poisoned_output = false;
          for (const auto& port : pair->value_ports(arm)) {
            if (port.kind != kn::ComparisonRegionValueKind::kInput &&
                port.kind != kn::ComparisonRegionValueKind::kOutput)
              continue;
            if (port.role == ir::Role::kWeight) continue;
            ASSERT_EQ(port.role, ir::Role::kActivation);
            if (port.kind == kn::ComparisonRegionValueKind::kInput) {
              ASSERT_EQ(port.source_tensor, prefix);
              poisoned_prefix = true;
            } else {
              ASSERT_EQ(port.source_tensor, compared_tensor);
              poisoned_output = true;
            }
            ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(port.address), 0xA5,
                                      static_cast<std::size_t>(port.bytes),
                                      static_cast<cudaStream_t>(stream->handle)),
                      cudaSuccess);
          }
          ASSERT_TRUE(poisoned_prefix);
          ASSERT_TRUE(poisoned_output);
          // The local body consumes a prefix result. The original input is
          // restaged after poisoning so the real prefix must recreate it.
        }
        ASSERT_NO_FATAL_FAILURE(stage_inputs(pair->input_slots(arm)));
        ASSERT_TRUE(pair->PrepareFrame(arm, *launch, *stream));
        ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
        // On the second frame the previous stop event is already complete.
        // A bounded host gate outside the authenticated graph keeps this replay
        // pending so the duration API cannot accidentally report that old stop.
        struct HostGate {
          std::atomic<bool> release{false};
          std::atomic<bool> expired{false};
        };
        std::shared_ptr<HostGate> gate;
        cudaError_t queued = cudaSuccess;
        if (recipe_case.measure_capture && frame == 1) {
          gate = std::make_shared<HostGate>();
          auto* keep = new std::shared_ptr<HostGate>(gate);
          queued = cudaLaunchHostFunc(
              static_cast<cudaStream_t>(stream->handle),
              [](void* opaque) {
                std::unique_ptr<std::shared_ptr<HostGate>> keep(
                    static_cast<std::shared_ptr<HostGate>*>(opaque));
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!(*keep)->release.load(std::memory_order_acquire)) {
                  if (std::chrono::steady_clock::now() >= deadline) {
                    (*keep)->expired.store(true, std::memory_order_release);
                    break;
                  }
                  std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
              },
              keep);
          if (queued != cudaSuccess) delete keep;
        }
        const auto replayed = pair->ReplayCapturedRegion(arm, *launch, *stream);
        const auto pending = gate ? pair->captured_region_duration_ns(arm, *launch)
                                  : std::expected<double, kn::StepFailure>{0};
        if (gate) gate->release.store(true, std::memory_order_release);
        const auto completed = cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle));
        ASSERT_EQ(queued, cudaSuccess);
        ASSERT_TRUE(replayed);
        ASSERT_EQ(completed, cudaSuccess);
        if (gate) {
          ASSERT_FALSE(gate->expired.load(std::memory_order_acquire));
          ASSERT_FALSE(pending) << "a completed old stop cannot prove the latest replay complete";
          EXPECT_EQ(pending.error().error, kn::StepError::kRejected);
        }
        const auto duration = pair->captured_region_duration_ns(arm, *launch);
        if (recipe_case.measure_capture) {
          ASSERT_TRUE(duration) << (duration ? "" : duration.error().detail);
          EXPECT_TRUE(std::isfinite(*duration));
          EXPECT_GT(*duration, 0);
        } else {
          EXPECT_FALSE(duration);
        }
        bool found_output = false;
        for (const auto& port : pair->value_ports(arm)) {
          if (port.kind != kn::ComparisonRegionValueKind::kOutput ||
              port.source_tensor != compared_tensor)
            continue;
          ASSERT_FALSE(found_output);
          found_output = true;
          ASSERT_NO_FATAL_FAILURE(check_output(port.address, port.bytes));
        }
        ASSERT_TRUE(found_output) << "closed region omitted its checked output";
      }
    }
  }

  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
  ASSERT_TRUE(GpuLife::Retire(stable_owner_handle->life, "retiring completed pair case"));
}

TEST_F(GgmlComparisonPairGpu, BoundPairReplaysCapturedRegionForChangedInputs) {
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(false));
}

TEST_F(GgmlComparisonPairGpu, NonclobberingRmsBodyCaptures128InvocationsForChangedInputs) {
  RecipeCase repeated;
  repeated.repetitions = 128;
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(false, repeated, nullptr, nullptr, true));
}

TEST_F(GgmlComparisonPairGpu, MeasuredCaptureReplaysRestagedInputsAndReportsBodyDuration) {
  kn::RecordingLaunch unsupported;
  EXPECT_TRUE(unsupported.SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
  EXPECT_FALSE(unsupported.SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
  RecipeCase measured;
  measured.measure_capture = true;
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(false, measured, nullptr, nullptr, true));
}

TEST_F(GgmlComparisonPairGpu, DescriptorPairBindsSameProgramsThenCapturesForChangedInputs) {
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(false, RecipeCase{}, nullptr, nullptr, true));
}

TEST_F(GgmlComparisonPairGpu, JoinedReadyBoundaryMatchesBuildStepAndReplaysChangedInputs) {
  spec::Stage stage{
      .name = "target", .role = spec::StageRole::kTarget, .artifact = 0, .context = 16};
  stage.embedding = spec::ComponentRef{.kind = "embedding.token",
                                       .version = 1,
                                       .params = {{.name = "scale", .value = 1.0}},
                                       .weights = {{.role = "table", .tensor = "table"}}};
  stage.stream.width = 16;
  stage.layers = {{{spec::ComponentRef{.kind = "norm.rms",
                                       .version = 1,
                                       .params = {{.name = "eps", .value = 1.0e-5}},
                                       .weights = {{.role = "weight", .tensor = "gamma"}}}}}};
  stage.heads.output = spec::ComponentRef{
      .kind = "head.linear", .version = 1, .weights = {{.role = "weight", .tensor = "head"}}};
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kLlm, .stages = {std::move(stage)}};
  spec::Canonicalize(model);
  const std::array<spec::SourceTensor, 3> tensors{
      spec::SourceTensor{.name = "table", .format = F32({16, 8})},
      spec::SourceTensor{.name = "gamma", .format = F32({16})},
      spec::SourceTensor{.name = "head", .format = F32({16, 8})}};
  const pl::StepShape shape{.stage = 0, .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  auto built = co::BuildStep(model, 0, shape, {.tensors = tensors});
  ASSERT_TRUE(built) << (built ? "" : built.error());

  std::optional<ir::NodeId> norm_id;
  std::optional<ir::NodeId> mul_id;
  for (std::uint32_t i = 0; i < built->graph.nodes().size(); ++i) {
    const ir::NodeId id{i};
    const auto& node = built->graph.node(id);
    if (node.key == ir::Primitive(ir::Op::kRmsNorm)) norm_id = id;
    if (node.key == ir::Primitive(ir::Op::kMul) && norm_id && node.nsrc == 2 &&
        (node.src[0] == built->graph.node(*norm_id).out[0] ||
         node.src[1] == built->graph.node(*norm_id).out[0]))
      mul_id = id;
  }
  ASSERT_TRUE(norm_id.has_value());
  ASSERT_TRUE(mul_id.has_value());
  const auto& norm = built->graph.node(*norm_id);
  const auto& mul = built->graph.node(*mul_id);
  ASSERT_EQ(norm.batch, ir::Batch::kJoined);
  ASSERT_EQ(mul.batch, ir::Batch::kJoined);
  const auto norm_weight =
      built->graph.tensor(mul.src[0]).role == ir::Role::kWeight ? mul.src[0] : mul.src[1];
  ASSERT_EQ(built->graph.tensor(norm_weight).role, ir::Role::kWeight);
  const auto* weight_ref = std::get_if<ir::WeightRef>(&built->graph.tensor(norm_weight).binding);
  ASSERT_NE(weight_ref, nullptr);
  const auto step_weight = std::ranges::find_if(
      built->weights, [&](const auto& item) { return item.tensor == norm_weight; });
  ASSERT_NE(step_weight, built->weights.end());
  ASSERT_EQ(step_weight->parts.size(), 1U);
  EXPECT_EQ(step_weight->parts.front().second, fmt::PartRole::kData);

  llmp::base::Sha256Digest expected_logical{};
  llmp::base::Sha256Digest expected_boundary{};
  {
    const co::GraphBuildIO io{built->graph, built->graph.inputs(), built->graph.outputs(),
                              built->weights};
    // The portable source view authenticates the complete original BuildStep
    // inventory. Only true graph inputs and immutable weight parts are external;
    // the activation feeding RMSNorm is an internal producer result.
    std::vector<co::ComparisonExternalBinding> external;
    std::uint32_t allocation_class = 1;
    for (const auto input : io.inputs) {
      const auto root = io.graph.root(input);
      if (std::ranges::any_of(
              external, [&](const auto& fact) { return fact.original_root == root && !fact.part; }))
        continue;
      const auto bytes = ir::StorageBytes(io.graph.tensor(root));
      ASSERT_TRUE(bytes);
      external.push_back({.original_root = root,
                          .allocation_class = allocation_class++,
                          .allocation_bytes = *bytes,
                          .bytes = *bytes,
                          .alignment = 1});
    }
    for (const auto& weight : io.weights) {
      const auto root = io.graph.root(weight.tensor);
      const auto parts = fmt::PartsOf(io.graph.tensor(root).format);
      ASSERT_TRUE(parts);
      ASSERT_EQ(parts->size(), weight.parts.size());
      for (std::size_t i = 0; i < parts->size(); ++i) {
        const auto& part = parts->parts()[i];
        external.push_back({.original_root = root,
                            .part = part.role,
                            .allocation_class = allocation_class++,
                            .allocation_bytes = part.bytes,
                            .bytes = part.bytes,
                            .alignment = 1});
      }
    }
    const std::array<std::uint32_t, 3> offsets{0, 1, 2};
    const std::array<ir::NodeId, 2> members{*norm_id, *mul_id};
    auto boundary = co::MakeComparisonRegionBoundary({{io, {}, 0, external}, offsets, members},
                                                     std::make_shared<PaidPairFunder>(life_->paid));
    ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
    expected_logical = boundary->digest().logical;
    expected_boundary = boundary->digest().combined;
  }

  RecipeCase ready_case{.pure_prefix = false,
                        .internal_output = false,
                        .ready_boundary_prefix = true,
                        .component = norm.component,
                        .layer = weight_ref->layer,
                        .part_name = step_weight->parts.front().first,
                        .epsilon = 1.0e-5F,
                        .alignment = 16};
  std::vector<execution::RecipeComparisonKey> keys;
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(false, ready_case, &keys, nullptr, true));
  ASSERT_EQ(keys.size(), 1U);
  EXPECT_EQ(keys.front().subject, expected_logical)
      << "the real joined body must match BuildStep's RMSNorm source boundary";
  EXPECT_EQ(keys.front().boundary, expected_boundary)
      << "the real joined body must preserve BuildStep's weight and storage facts";
}

TEST_F(GgmlComparisonPairGpu, SelectedRecipePromotesAndReplaysCompleteScheduleForChangedInputs) {
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(true));
}

TEST_F(GgmlComparisonPairGpu, ActualRecipeKeysNormalizeOccurrenceAndLocationButRetainParameters) {
  std::vector<execution::RecipeComparisonKey> keys;
  std::vector<std::uint64_t> source_identities;
  const RecipeCase isolated{.pure_prefix = false,
                            .internal_output = false,
                            .component = 3,
                            .layer = 1,
                            .part_name = "norm.weight",
                            .epsilon = 1.0e-6F,
                            .parent_extra = 0,
                            .view_offset_extra = 0,
                            .alignment = 16,
                            .allocation_class_x = 1,
                            .allocation_class_w = 2};
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(true, isolated, &keys, &source_identities));

  ASSERT_NO_FATAL_FAILURE(InitializeLife());
  const RecipeCase prefixed_internal{.pure_prefix = true,
                                     .internal_output = true,
                                     .component = 91,
                                     .layer = 47,
                                     .part_name = "layer47.projection",
                                     .epsilon = 1.0e-6F,
                                     .parent_extra = 8192,
                                     .view_offset_extra = 512,
                                     .alignment = 512,
                                     .allocation_class_x = 17,
                                     .allocation_class_w = 23};
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(true, prefixed_internal, &keys, &source_identities));

  ASSERT_NO_FATAL_FAILURE(InitializeLife());
  auto changed_parameter = prefixed_internal;
  changed_parameter.epsilon = 2.0e-6F;
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(true, changed_parameter, &keys, &source_identities));

  ASSERT_EQ(keys.size(), 3U);
  ASSERT_EQ(source_identities.size(), 3U);
  EXPECT_EQ(keys[0], keys[1])
      << "the local recipe should ignore isolated-vs-prefix-produced value roles, "
         "renamed occurrence labels, parent capacity/offset, and excess alignment";
  EXPECT_NE(keys[0], keys[2]) << "the actual RMSNorm parameter remains recipe identity";
  EXPECT_NE(source_identities[0], source_identities[1]);
  EXPECT_NE(source_identities[1], source_identities[2]);
  EXPECT_NE(source_identities[0], source_identities[2]);
}

TEST_F(GgmlComparisonPairGpu, CumulativeAdoptionKeepsBothRealNormSitesAndRunsFullTokenGraph) {
  std::vector<execution::RecipeComparisonKey> ready_keys;
  RecipeCase ready{.width = 64,
                   .pure_prefix = false,
                   .internal_output = false,
                   .ready_boundary_prefix = true,
                   .component = 3,
                   .layer = 0,
                   .part_name = "norm.weight",
                   .epsilon = 1.0e-6F};
  ASSERT_NO_FATAL_FAILURE(RunPairQualification(false, ready, &ready_keys, nullptr, true));
  ASSERT_EQ(ready_keys.size(), 1U);
  ASSERT_NO_FATAL_FAILURE(InitializeLife());

  auto source_result = MakeTwoLayerTokenSource();
  ASSERT_TRUE(source_result) << (source_result ? "" : source_result.error());
  std::optional<TwoLayerTokenSource> source_owner{std::move(*source_result)};
  auto& source = *source_owner;
  auto& step = source.step;
  struct PrenormSite {
    ir::TensorId output{};
    ir::TensorId weight{};
  };
  std::array<PrenormSite, 2> prenorm_sites{};
  std::array<ir::TensorId, 2> prenorm_weights{};
  for (const auto& weight : step.weights) {
    if (weight.ref.layer >= prenorm_weights.size() ||
        !std::ranges::any_of(weight.parts, [&](const auto& part) {
          return part.first == std::format("attn_norm_{}", weight.ref.layer) &&
                 part.second == fmt::PartRole::kData;
        }))
      continue;
    auto& target = prenorm_weights[weight.ref.layer];
    ASSERT_FALSE(target.valid());
    target = weight.tensor;
  }
  ASSERT_TRUE(prenorm_weights[0].valid());
  ASSERT_TRUE(prenorm_weights[1].valid());
  for (std::uint32_t n = 0; n < step.graph.nodes().size(); ++n) {
    const auto& node = step.graph.node(ir::NodeId{n});
    if (node.key != ir::Primitive(ir::Op::kMul) || node.sources().size() != 2 ||
        node.outputs().size() != 1)
      continue;
    for (std::size_t wi = 0; wi < 2; ++wi) {
      const auto weight = step.graph.root(node.sources()[wi]);
      const auto layer = std::ranges::find(prenorm_weights, weight);
      if (layer == prenorm_weights.end()) continue;
      const auto layer_index = static_cast<std::size_t>(layer - prenorm_weights.begin());
      const auto normalized = step.graph.root(node.sources()[1 - wi]);
      const auto norm_producer = step.graph.producer(normalized);
      if (!norm_producer.valid() ||
          step.graph.node(norm_producer).key != ir::Primitive(ir::Op::kRmsNorm))
        continue;
      auto& site = prenorm_sites[layer_index];
      ASSERT_FALSE(site.output.valid()) << "each layer has exactly one prenorm RMS/Mul site";
      site = {.output = node.outputs().front(), .weight = weight};
    }
  }
  ASSERT_TRUE(prenorm_sites[0].output.valid());
  ASSERT_TRUE(prenorm_sites[1].output.valid());
  const co::GraphBuildIO io = co::GraphIO(step);
  auto backend = life_->backend->Backend();
  const auto& base_backend = backend.owner ? backend.owner->backend() : backend;
  ASSERT_NE(base_backend.registry, nullptr);
  auto test_registry = TestOnlyRecipeRegistry(*base_backend.registry, ready_keys.front());
  ASSERT_TRUE(test_registry) << (test_registry ? "" : "test recipe registry unavailable");
  ASSERT_EQ(test_registry->SelectionDigest(), base_backend.registry->SelectionDigest());
  auto miss_key = ready_keys.front();
  miss_key.subject =
      llmp::base::Sha256{}.Update("valid but nonmatching cumulative subject").Finish();
  ASSERT_NE(miss_key, ready_keys.front());
  auto miss_registry = TestOnlyRecipeRegistry(*base_backend.registry, miss_key);
  ASSERT_TRUE(miss_registry) << (miss_registry ? "" : "test miss registry unavailable");
  ASSERT_EQ(miss_registry->SelectionDigest(), base_backend.registry->SelectionDigest());
  backend = base_backend;
  backend.registry = &*miss_registry;

  constexpr std::uint64_t arena_bytes = 8 * kMiB;
  constexpr std::uint64_t pool_bytes = 8 * kMiB;
  ASSERT_TRUE(life_->arena.Allocate(arena_bytes));
  ASSERT_TRUE(life_->pool.Allocate(pool_bytes));
  ASSERT_TRUE(life_->model_state.Allocate(source.state.bytes));
  ASSERT_EQ(cudaMemset(life_->model_state.pointer, 0, static_cast<std::size_t>(source.state.bytes)),
            cudaSuccess);

  struct WeightPlace {
    ir::WeightRef ref;
    fmt::PartRole part;
    ir::TensorId tensor;
    std::uint64_t offset;
    std::uint64_t bytes;
    std::vector<float> values;
  };
  std::vector<WeightPlace> weight_places;
  std::uint64_t weight_bytes = 0;
  for (const auto& weight : step.weights) {
    const auto parts = fmt::PartsOf(step.graph.tensor(weight.tensor).format);
    ASSERT_TRUE(parts);
    ASSERT_EQ(parts->size(), weight.parts.size());
    for (const auto& part : parts->parts()) {
      weight_bytes = (weight_bytes + 255U) & ~std::uint64_t{255U};
      ASSERT_LE(part.readable, std::numeric_limits<std::uint64_t>::max() - weight_bytes);
      WeightPlace place{.ref = weight.ref,
                        .part = part.role,
                        .tensor = weight.tensor,
                        .offset = weight_bytes,
                        .bytes = part.readable,
                        .values = {}};
      ASSERT_EQ(part.role, fmt::PartRole::kData);
      ASSERT_EQ(part.readable % sizeof(float), 0U);
      place.values.resize(static_cast<std::size_t>(part.readable / sizeof(float)));
      const bool norm = step.graph.tensor(weight.tensor).shape().rank == 1;
      for (std::size_t i = 0; i < place.values.size(); ++i)
        place.values[i] = norm ? 1.0F : static_cast<float>(static_cast<int>(i % 13) - 6) / 64.0F;
      weight_bytes += part.readable;
      weight_places.push_back(std::move(place));
    }
  }
  ASSERT_GT(weight_bytes, 0U);
  ASSERT_TRUE(life_->model_weights.Allocate(weight_bytes));
  std::map<ir::TensorId, std::vector<float>> reference_weights;
  for (const auto& place : weight_places) {
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(life_->model_weights.address() + place.offset),
                         place.values.data(), static_cast<std::size_t>(place.bytes),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    reference_weights.emplace(place.tensor, place.values);
  }

  std::map<ir::StateRef, const model::StateLayoutDerived::Item*> state_items;
  for (const auto& state_fact : step.state) {
    ASSERT_LT(state_fact.item, source.state.items.size());
    ASSERT_TRUE(state_items.emplace(state_fact.ref, &source.state.items[state_fact.item]).second);
  }
  kn::Addresses addresses{};
  addresses.weight = [&](const ir::WeightRef& ref, fmt::PartRole part) {
    const auto found = std::ranges::find_if(
        weight_places, [&](const auto& place) { return place.ref == ref && place.part == part; });
    return found == weight_places.end() ? std::uint64_t{0}
                                        : life_->model_weights.address() + found->offset;
  };
  addresses.weight_alignment = [](const ir::WeightRef&, fmt::PartRole) {
    return std::uint64_t{256};
  };
  addresses.weight_range = [&](const ir::WeightRef& ref,
                               fmt::PartRole part) -> std::optional<kn::BoundedAddress> {
    const auto found = std::ranges::find_if(
        weight_places, [&](const auto& place) { return place.ref == ref && place.part == part; });
    if (found == weight_places.end()) return std::nullopt;
    return kn::BoundedAddress{.base = life_->model_weights.address() + found->offset,
                              .bytes = found->bytes,
                              .alignment = 256};
  };
  addresses.state = [&](const ir::StateRef& ref) {
    const auto found = state_items.find(ref);
    return found == state_items.end() ? std::uint64_t{0}
                                      : life_->model_state.address() + found->second->offset;
  };
  addresses.state_alignment = [](const ir::StateRef&) { return std::uint64_t{256}; };
  addresses.state_range = [&](const ir::StateRef& ref) -> std::optional<kn::BoundedAddress> {
    const auto found = state_items.find(ref);
    if (found == state_items.end()) return std::nullopt;
    return kn::BoundedAddress{.base = life_->model_state.address() + found->second->offset,
                              .bytes = found->second->bytes,
                              .alignment = 256};
  };
  addresses.activations = life_->arena.address();
  addresses.activation_bytes = arena_bytes;
  addresses.activation_alignment = 256;

  pl::StateSpans state_spans =
      [&](std::span<const pl::StateCellSlice> slices) -> std::optional<pl::StateSpanFact> {
    if (slices.empty()) return std::nullopt;
    std::vector<model::StateSlice> requested;
    requested.reserve(slices.size());
    for (const auto& slice : slices) {
      if (slice.ref.stage != 0 || slice.ref.segment != 0) return std::nullopt;
      requested.push_back({.layer = slice.ref.layer,
                           .tensor = slice.ref.tensor,
                           .first = slice.first,
                           .cells = slice.cells});
    }
    const auto bank = model::ContiguousStateSpan(source.state, requested);
    if (!bank) return std::nullopt;
    for (const auto& slice : slices) {
      const auto found = state_items.find(slice.ref);
      if (found == state_items.end() || addresses.state(slice.ref) < found->second->offset ||
          addresses.state(slice.ref) - found->second->offset != life_->model_state.address())
        return std::nullopt;
    }
    return pl::StateSpanFact{.bank = *bank, .region_bytes = source.state.bytes};
  };
  pl::PlanOptions options{.capturable = true,
                          .staged = std::vector<ir::TensorId>{},
                          .on_stream = base_backend.on_stream,
                          .written = base_backend.written,
                          .within = base_backend.within,
                          .shape = &source.shape,
                          .state_spans = state_spans,
                          .alignment_staged = std::vector<ir::TensorId>{}};
  for (const auto& input : step.inputs) {
    if (input.kind != co::InputKind::kCellPositions) options.staged->push_back(input.tensor);
    if (input.kind != co::InputKind::kCellPositions && input.kind != co::InputKind::kCommitRows)
      options.alignment_staged->push_back(input.tensor);
  }
  ASSERT_TRUE(en::PrepareComponentPlanOptionsInPlace(io, backend, addresses, options));
  const pl::Policy base_policy{};
  auto ruled = pl::ForStep(base_policy, source.shape.rows(),
                           static_cast<std::uint32_t>(source.shape.segments.size()));
  ASSERT_TRUE(ruled) << (ruled ? "" : ruled.error());
  const pl::Policy policy = std::move(*ruled);

  struct LoweredSiteProof {
    std::uint64_t graph_identity = 0;
    std::array<bool, 2> decomposed{};
    const kn::StepProgram* program = nullptr;
  };
  std::vector<LoweredSiteProof> lower_proofs;
  const kn::ProgramValidator observe_plan = [&](const kn::LowerInput& input,
                                                const kn::StepProgram& program) {
    LoweredSiteProof proof{.graph_identity = input.plan.graph.identity(),
                           .program = &program.comparison_program()};
    for (std::size_t i = 0; i < prenorm_sites.size(); ++i) {
      const auto& site = prenorm_sites[i];
      if (site.output.v >= input.plan.tensors.size() || site.weight.v >= input.plan.tensors.size())
        continue;
      const auto output = input.plan.tensors[site.output.v];
      const auto weight = input.plan.tensors[site.weight.v];
      if (!output.valid() || output.v >= input.plan.graph.tensor_count() || !weight.valid() ||
          weight.v >= input.plan.graph.tensor_count())
        continue;
      const auto& planned_output =
          input.plan.graph.node(input.plan.graph.producer(input.plan.graph.root(output)));
      if (planned_output.key != ir::Primitive(ir::Op::kMul) || planned_output.sources().size() != 2)
        continue;
      bool has_weight = false;
      bool has_primitive_norm = false;
      for (const auto source_tensor : planned_output.sources()) {
        const auto root = input.plan.graph.root(source_tensor);
        has_weight |= root == weight;
        const auto producer = input.plan.graph.producer(root);
        has_primitive_norm |= producer.valid() && input.plan.graph.node(producer).key ==
                                                      ir::Primitive(ir::Op::kRmsNorm);
      }
      proof.decomposed[i] = has_weight && has_primitive_norm;
    }
    lower_proofs.push_back(proof);
    return std::expected<void, std::string>{};
  };

  const auto persistent_bytes = life_->backend->persistent_bytes();
  if (persistent_bytes != 0) ASSERT_TRUE(life_->persistent.Allocate(persistent_bytes));
  auto made_launch =
      life_->backend->Launch({.base = life_->pool.address(), .size = llmp::base::Bytes(pool_bytes)},
                             {.base = life_->persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(made_launch) << (made_launch ? "" : made_launch.error());
  life_->launch = std::move(*made_launch);

  const auto miss_lower_begin = lower_proofs.size();
  auto miss_options = options;
  auto miss_program = en::BuildCumulativeRecipeProgram(io, step, source.shape, 0, 256, backend,
                                                       policy, addresses, std::move(miss_options),
                                                       observe_plan, backend.owner, nullptr);
  ASSERT_TRUE(miss_program) << (miss_program ? "" : miss_program.error());
  ASSERT_GT(lower_proofs.size(), miss_lower_begin);
  const auto miss_identity = (*miss_program)->comparison_plan_identity();
  ASSERT_TRUE(miss_identity);
  EXPECT_EQ(miss_identity->source_graph_identity, step.graph.identity());
  EXPECT_EQ(miss_identity->planned_graph_identity, lower_proofs[miss_lower_begin].graph_identity)
      << "a valid row miss must return the exact first lowered incumbent";
  EXPECT_FALSE(lower_proofs[miss_lower_begin].decomposed[0]);
  EXPECT_FALSE(lower_proofs[miss_lower_begin].decomposed[1]);
  EXPECT_EQ(&(*miss_program)->comparison_program(), lower_proofs[miss_lower_begin].program)
      << "a valid row miss must return the exact first lowered incumbent";
  EXPECT_EQ((*miss_program)->bound(), nullptr);
  miss_program->reset();
  backend.registry = &*test_registry;

  const auto hit_lower_begin = lower_proofs.size();
  auto cumulative =
      en::BuildCumulativeRecipeProgram(io, step, source.shape, 0, 256, backend, policy, addresses,
                                       std::move(options), observe_plan, backend.owner, nullptr);
  ASSERT_TRUE(cumulative) << (cumulative ? "" : cumulative.error());
  ASSERT_NE(*cumulative, nullptr);
  // Keep the selected program inside the completion-aware fixture owner before
  // any bind or launch can enqueue work.  Its retirement path fences it with
  // the buffers and launch resources on every assertion exit.
  life_->selected_program = std::move(*cumulative);
  auto* program = life_->selected_program.get();
  ASSERT_NE(program, nullptr);
  const auto identity = program->comparison_plan_identity();
  ASSERT_TRUE(identity);
  EXPECT_EQ(identity->source_graph_identity, step.graph.identity());
  EXPECT_GT(program->pool_bytes(), 0U)
      << "attention outside the local RMS bodies keeps its real nonzero pool requirement";
  std::string report;
  program->StreamPlanReport([&](std::string_view part) { report.append(part); });
  const auto final_proof = std::ranges::find(lower_proofs, identity->planned_graph_identity,
                                             &LoweredSiteProof::graph_identity);
  ASSERT_NE(final_proof, lower_proofs.end())
      << "the selected actual program must have passed the plan observer";
  ASSERT_GT(lower_proofs.size(), hit_lower_begin);
  EXPECT_FALSE(lower_proofs[hit_lower_begin].decomposed[0]);
  EXPECT_FALSE(lower_proofs[hit_lower_begin].decomposed[1]);
  EXPECT_NE(identity->planned_graph_identity, lower_proofs[hit_lower_begin].graph_identity)
      << "the selected schedule must differ from the original incumbent";
  EXPECT_TRUE(std::ranges::any_of(
      lower_proofs.begin() + static_cast<std::ptrdiff_t>(hit_lower_begin + 1), final_proof,
      [](const auto& proof) { return proof.decomposed[0] != proof.decomposed[1]; }))
      << "the first accepted site should appear before the second site is adopted";
  EXPECT_TRUE(final_proof->decomposed[0])
      << "layer 0's actual prenorm output and immutable weight must map to RMS+Mul";
  EXPECT_TRUE(final_proof->decomposed[1])
      << "layer 1's actual prenorm output and immutable weight must map to RMS+Mul";

  const std::array<std::int32_t, 1> token_values{3};
  std::vector<float> expected_values;
  {
    auto reference = ref::Executor::Create(step.graph);
    ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
    for (const auto& weight : step.weights)
      ASSERT_TRUE(reference->BindValues(weight.tensor, reference_weights.at(weight.tensor)));
    for (const auto& state_fact : step.state) {
      const auto elements = ir::Elements(step.graph.tensor(state_fact.tensor).shape());
      ASSERT_TRUE(reference->BindValues(state_fact.tensor, std::vector<float>(elements, 0.0F)));
    }
    for (const auto& input : step.inputs) {
      auto values = co::InputValues(input, source.shape, token_values);
      ASSERT_TRUE(values) << (values ? "" : values.error());
      ASSERT_TRUE(reference->BindInts(input.tensor, *values));
    }
    ASSERT_TRUE(reference->Run()) << "original full token graph reference executes";
    const auto reference_values = reference->Values(step.graph.outputs().front());
    ASSERT_TRUE(reference_values) << (reference_values ? "" : reference_values.error().detail);
    expected_values = std::move(*reference_values);
  }
  struct InputPayload {
    std::uint64_t address = 0;
    std::vector<std::byte> bytes;
  };
  std::vector<InputPayload> input_payloads;
  input_payloads.reserve(program->inputs().size());
  for (const auto& slot : program->inputs()) {
    std::vector<std::byte> payload(static_cast<std::size_t>(slot.bytes));
    if (slot.kind == kn::InputKind::kTokens || slot.kind == kn::InputKind::kPositions) {
      const auto input = std::ranges::find(step.inputs, slot.tensor, &co::StepInput::tensor);
      ASSERT_NE(input, step.inputs.end());
      auto values = co::InputValues(*input, source.shape, token_values);
      ASSERT_TRUE(values);
      std::vector<std::int32_t> narrowed(values->size());
      std::ranges::transform(*values, narrowed.begin(),
                             [](std::int64_t value) { return static_cast<std::int32_t>(value); });
      ASSERT_TRUE(en::step_inputs::WriteRows(payload, narrowed));
    } else {
      ASSERT_TRUE(en::step_inputs::WriteComputed(slot, source.shape, payload));
    }
    input_payloads.push_back({slot.device_address, std::move(payload)});
  }
  // The selected executable and copied reference/staging data must no longer
  // depend on the compiler source graph or its planning callbacks.
  source_owner.reset();

  ASSERT_TRUE(program->Bind(*life_->launch));
  const auto stream = life_->node.execution().Submission(life_->node.stream(0));
  ASSERT_TRUE(stream) << (stream ? "" : stream.error().detail);
  for (const auto& payload : input_payloads) {
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(payload.address), payload.bytes.data(),
                         payload.bytes.size(), cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  ASSERT_TRUE(program->Run(*life_->launch, *stream));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
  ASSERT_EQ(program->outputs().size(), 1U);
  ASSERT_EQ(program->outputs().front().bytes, expected_values.size() * sizeof(float));
  std::vector<float> actual(expected_values.size());
  ASSERT_EQ(cudaMemcpy(actual.data(),
                       reinterpret_cast<const void*>(program->outputs().front().device_address),
                       static_cast<std::size_t>(program->outputs().front().bytes),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], expected_values[i], 1.0e-4F) << i;
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
}

}  // namespace
