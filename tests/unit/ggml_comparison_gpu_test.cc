// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The requested comparison path enters the real backend's LowerFor boundary.
// These synthetic fixtures execute GPU work for proof and lifetime validation;
// Normal test runs collect no timing or model evidence. An explicit external
// timing invocation reuses these complete reset/reference callbacks.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <print>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "artifact_writer.h"
#include "base/bytes.h"
#include "components/comparison_boundary.h"
#include "components/input_values.h"
#include "components/step.h"
#include "engine/comparison_pair.h"
#include "engine/execution_contract.h"
#include "engine/graph_plan.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/runner_owner.h"
#include "engine/runner_resources.h"
#include "engine/step_inputs.h"
#include "execution/fusion_ranks.h"
#include "fake_step_source.h"
#include "format/encoding.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/comparison_region.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/step_program.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/reference/decode.h"
#include "kernels/reference/exec.h"
#include "model/state_contract.h"
#include "planner/alternative.h"
#include "planner/pattern.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "platform/host_probe.h"
#include "providers/device_execution.h"
#include "spec/spec.h"

namespace {
namespace co = llmp::components;
namespace en = llmp::engine;
namespace ir = llmp::ir;
namespace kn = llmp::kernels;
namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace planner = llmp::planner;
namespace reference = llmp::kernels::reference;
namespace spec = llmp::spec;
namespace ts = llmp::test_support;
constexpr std::uint64_t kMiB = 1ULL << 20U;

struct TokenStateSource {
  spec::ModelSpec model;
  std::vector<co::BoundTensor> tensors;
  llmp::model::StateLayoutDerived layout;
  llmp::planner::StepShape shape;
  co::StepGraph step;
};

spec::Param SpecParam(std::string name, spec::ParamValue value) {
  return {.name = std::move(name), .value = std::move(value)};
}

spec::WeightBinding SpecWeight(std::string role, std::string tensor) {
  return {.role = std::move(role), .tensor = std::move(tensor)};
}

llmp::format::TensorFormat F32Format(std::initializer_list<std::uint64_t> dimensions) {
  return {.encoding = llmp::format::Encoding::kF32, .shape = ir::ShapeOf(dimensions)};
}

std::expected<TokenStateSource, std::string> MakeSmallTokenStateSource(
    std::uint32_t rows = 1, bool owner_attention = false) {
  constexpr std::uint64_t width = 64;
  const std::uint64_t head_dim = owner_attention ? 256 : width;
  const std::uint64_t heads = owner_attention ? 8 : 1;
  const std::uint64_t kv_heads = owner_attention ? 4 : 1;
  spec::ComponentRef embedding{.kind = "embedding.token"};
  embedding.weights = {SpecWeight("table", "token_embd")};
  spec::ComponentRef attention{.kind = "attention.dense"};
  attention.params = {SpecParam("placement", std::string("pre")),
                      SpecParam("eps", 1.0e-6),
                      SpecParam("heads", static_cast<std::int64_t>(heads)),
                      SpecParam("kv_heads", static_cast<std::int64_t>(kv_heads)),
                      SpecParam("key_dim", static_cast<std::int64_t>(head_dim)),
                      SpecParam("value_dim", static_cast<std::int64_t>(head_dim)),
                      SpecParam("scale", 0.125),
                      SpecParam("softcap", 0.0),
                      SpecParam("window", std::int64_t{0}),
                      SpecParam("rotation", std::string("neox")),
                      SpecParam("rope_dims", static_cast<std::int64_t>(head_dim)),
                      SpecParam("rope_base", 10000.0),
                      SpecParam("rope_scale", 1.0)};
  attention.weights = {SpecWeight("k", "attn_k"), SpecWeight("norm", "attn_norm"),
                       SpecWeight("output", "attn_output"), SpecWeight("q", "attn_q"),
                       SpecWeight("v", "attn_v")};
  spec::Stage stage{.name = "target", .context = 256, .embedding = std::move(embedding)};
  stage.stream.width = width;
  stage.layers = {{.components = {std::move(attention)}}};
  stage.heads.final_norm = spec::ComponentRef{.kind = "norm.rms"};
  stage.heads.final_norm->params = {SpecParam("eps", 1.0e-6)};
  stage.heads.final_norm->weights = {SpecWeight("weight", "output_norm")};
  stage.heads.output = spec::ComponentRef{.kind = "head.linear"};
  stage.heads.output->weights = {SpecWeight("weight", "output_head")};
  spec::ModelSpec model{.stages = {std::move(stage)}};
  spec::Canonicalize(model);
  std::vector<co::BoundTensor> tensors{
      {.name = "attn_k", .format = F32Format({width, head_dim * kv_heads})},
      {.name = "attn_norm", .format = F32Format({width})},
      {.name = "attn_output", .format = F32Format({head_dim * heads, width})},
      {.name = "attn_q", .format = F32Format({width, head_dim * heads})},
      {.name = "attn_v", .format = F32Format({width, head_dim * kv_heads})},
      {.name = "output_head", .format = F32Format({width, 8})},
      {.name = "output_norm", .format = F32Format({width})},
      {.name = "token_embd", .format = F32Format({width, 8})}};
  auto valid = co::ValidateSpec(model, tensors);
  if (!valid) return std::unexpected(valid.error());
  auto layout = co::StageLayout(model, 0, 256, 1);
  if (!layout) return std::unexpected(layout.error());
  llmp::planner::StepShape shape{
      .phase = llmp::planner::Phase::kPrefill,
      .segments = {{.slot = 0, .rows = rows, .past = 0, .read_cells = 256, .outputs = rows}}};
  if (owner_attention) {
    shape.phase = planner::Phase::kDecode;
    shape.segments.push_back({.slot = 1, .rows = 1, .past = 0, .read_cells = 256, .outputs = 1});
  }
  auto step = co::BuildStep(model, 0, shape, {.tensors = tensors, .state = &*layout});
  if (!step) return std::unexpected(step.error());
  return TokenStateSource{.model = std::move(model),
                          .tensors = std::move(tensors),
                          .layout = std::move(*layout),
                          .shape = std::move(shape),
                          .step = std::move(*step)};
}

// The smallest real shared GDN source for the Ready NormGate applicability control.
// All roles are synthetic F32 values already carrying the declared folds.
std::expected<TokenStateSource, std::string> MakeSmallGdnStateSource() {
  spec::ComponentRef embedding{.kind = "embedding.token"};
  embedding.weights = {SpecWeight("table", "token_embd")};
  spec::ComponentRef gdn{.kind = "attention.gdn"};
  gdn.params = {SpecParam("placement", std::string("pre")),
                SpecParam("eps", 1.0e-6),
                SpecParam("head_dim", std::int64_t{128}),
                SpecParam("k_heads", std::int64_t{16}),
                SpecParam("v_heads", std::int64_t{48}),
                SpecParam("conv", std::int64_t{4}),
                SpecParam("output_gate", std::string("sigmoid")),
                SpecParam("a_log", false),
                SpecParam("norm_zero_centered", false)};
  std::vector<co::BoundTensor> tensors;
  const auto add = [&](std::string role, std::initializer_list<std::uint64_t> dims) {
    const auto name = "gdn_" + role;
    gdn.weights.push_back(SpecWeight(std::move(role), name));
    tensors.push_back({.name = name, .format = F32Format(dims)});
  };
  add("qkv", {64, 10240});
  add("gate", {64, 6144});
  add("beta", {64, 48});
  add("alpha", {64, 48});
  add("dt_bias", {48});
  add("a", {48});
  add("conv", {4, 10240});
  add("norm", {128});
  add("output", {6144, 64});
  add("input_norm", {64});
  tensors.push_back({.name = "token_embd", .format = F32Format({64, 8})});
  tensors.push_back({.name = "output_head", .format = F32Format({64, 8})});
  spec::Stage stage{.name = "target", .context = 256, .embedding = std::move(embedding)};
  stage.stream.width = 64;
  stage.layers = {{.components = {std::move(gdn)}}};
  stage.folds = {"ssm.a_neg_exp", "gdn.value_heads_tiled"};
  stage.heads.output = spec::ComponentRef{.kind = "head.linear"};
  stage.heads.output->weights = {SpecWeight("weight", "output_head")};
  spec::ModelSpec model{.stages = {std::move(stage)}};
  spec::Canonicalize(model);
  auto valid = co::ValidateSpec(model, tensors);
  if (!valid) return std::unexpected(valid.error());
  auto layout = co::StageLayout(model, 0, 256, 1);
  if (!layout) return std::unexpected(layout.error());
  planner::StepShape shape{.stage = 0,
                           .phase = planner::Phase::kDecode,
                           .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  auto step = co::BuildStep(model, 0, shape, {.tensors = tensors, .state = &*layout});
  if (!step) return std::unexpected(step.error());
  return TokenStateSource{.model = std::move(model),
                          .tensors = std::move(tensors),
                          .layout = std::move(*layout),
                          .shape = std::move(shape),
                          .step = std::move(*step)};
}

struct CudaBuffer {
  void* pointer = nullptr;
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
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
  void* Release() { return std::exchange(pointer, nullptr); }
};

class ForeignStepLaunch final : public llmp::kernels::StepLaunch {
 public:
  std::expected<std::unique_ptr<llmp::kernels::Recorded>, llmp::kernels::StepFailure> Capture(
      llmp::providers::NativeStream, const Body&) override {
    return std::unexpected(llmp::kernels::StepFailure{.detail = "foreign launch"});
  }
  std::expected<void, llmp::kernels::StepFailure> Launch(llmp::providers::NativeStream,
                                                         const llmp::kernels::Recorded&) override {
    return std::unexpected(llmp::kernels::StepFailure{.detail = "foreign launch"});
  }
};

class PaidBoundaryFunder final : public ir::StorageFunder {
 public:
  explicit PaidBoundaryFunder(std::shared_ptr<en::ExecutionMetadataAllowance> paid)
      : paid_(std::move(paid)) {}
  bool Charge(std::uint64_t bytes) const override { return paid_->Charge(bytes); }
  void Release(std::uint64_t bytes) const override { paid_->Release(bytes); }

 private:
  std::shared_ptr<en::ExecutionMetadataAllowance> paid_;
};

bool WholeTimingConfigured() {
  const auto* value = std::getenv("LLMP_U6_WHOLE_TIMING");
  return value && (std::string_view(value) == "1" || std::string_view(value) == "preflight");
}

bool WholeTimingRequested() {
  const auto* value = std::getenv("LLMP_U6_WHOLE_TIMING");
  return value && std::string_view(value) == "1";
}

std::string TimingQuote(std::string_view value) {
  // Only internal ASCII operation names and hexadecimal digests reach this helper.
  return std::format("\"{}\"", value);
}

std::string RecipeArmJson(const llmp::execution::RecipeArmIdentity& arm) {
  return std::format(
      "{{\"planned_region\":{},\"physical_recipe\":{},"
      "\"backend_recipe\":{},\"backend_build\":{},"
      "\"backend_selection\":{}}}",
      TimingQuote(llmp::base::ToHex(arm.planned_region)),
      TimingQuote(llmp::base::ToHex(arm.physical_recipe)),
      TimingQuote(llmp::base::ToHex(arm.backend_recipe)),
      TimingQuote(llmp::base::ToHex(arm.backend_build)),
      TimingQuote(llmp::base::ToHex(arm.backend_selection)));
}

std::string RecipeKeyJson(const llmp::execution::RecipeComparisonKey& key) {
  const auto operation =
      key.discovery_operation.op == ir::Op::kFused
          ? std::format("fused.{}", ir::FusedKindName(key.discovery_operation.fused))
          : std::string(ir::OpName(key.discovery_operation.op));
  return std::format(
      "{{\"version\":{},\"region_canonicalizer\":{},\"discovery_operation\":{},"
      "\"mode\":{},\"subject\":{},\"boundary\":{},\"policy\":{},\"device\":{},"
      "\"selection_environment\":{},\"preparation_domain\":{},\"incumbent\":{},"
      "\"alternative\":{}}}",
      key.version, key.region_canonicalizer, TimingQuote(operation),
      TimingQuote(key.mode == llmp::execution::RecipeRunMode::kFull ? "full" : "warm"),
      TimingQuote(llmp::base::ToHex(key.subject)), TimingQuote(llmp::base::ToHex(key.boundary)),
      TimingQuote(llmp::base::ToHex(key.policy)), TimingQuote(llmp::base::ToHex(key.device)),
      TimingQuote(llmp::base::ToHex(key.selection_environment)),
      TimingQuote(llmp::base::ToHex(key.preparation_domain)), RecipeArmJson(key.incumbent),
      RecipeArmJson(key.alternative));
}

// Opt-in measurement of an already qualified, genuine whole-program pair.
// The callback restores/stages one arm, invokes it once, finishes, and checks
// complete outputs/state/weights. Durations are the capture's internal events.
template <class Run>
void MeasureWholeStatePair(std::string_view shape_class,
                           const llmp::execution::RecipeComparisonKey& key,
                           const std::array<std::unique_ptr<kn::Recorded>, 2>& captures,
                           Run&& run) {
  if (!WholeTimingConfigured()) return;
  ASSERT_FALSE(::testing::Test::HasFailure());
  for (const auto& capture : captures) {
    ASSERT_TRUE(capture);
    ASSERT_GT(capture->nodes(), 2U);
  }
  constexpr std::string_view protocol =
      "llmp.u6.whole-state-captured.v1;invocations=1;"
      "duration=internal-captured-events;reset=full-state-and-input-before-every-arm;"
      "reference=complete-original-output-state-immutable-weight-check-after-every-arm;"
      "fixed-ab-warmups=4;alternative-bookend-before=median9;"
      "alternating-ab-ba-warmups=4;alternating-ab-ba-sample-pairs=9;"
      "alternative-bookend-after=median9;records=52;drift=10pct;retry=none";
  const auto digest = llmp::base::ToHex(llmp::base::Sha256{}.Update(protocol).Finish());
  std::println(
      "{{\"event\":\"whole_state_pair\",\"shape_class\":{},"
      "\"recipe_key\":{},\"protocol\":{},\"description\":{}}}",
      TimingQuote(shape_class), RecipeKeyJson(key), TimingQuote(digest), TimingQuote(protocol));
  if (!WholeTimingRequested()) return;
  const auto time = [&](std::size_t arm, std::string_view phase, int pair, std::size_t order,
                        double& ns) {
    run(arm);
    if (::testing::Test::HasFailure()) return;
    const auto measured = captures[arm]->gpu_duration_ns();
    ASSERT_TRUE(measured) << (measured ? "" : measured.error().detail);
    ns = *measured;
    ASSERT_TRUE(std::isfinite(ns));
    ASSERT_GT(ns, 0.0);
    std::println(
        "{{\"event\":\"sample\",\"shape_class\":{},\"phase\":{},"
        "\"pair\":{},\"order\":{},\"arm\":{},\"ns\":{}}}",
        TimingQuote(shape_class), TimingQuote(phase), pair, order, arm, ns);
    std::fflush(stdout);
  };
  double ns = 0;
  for (int pair = 0; pair < 4; ++pair)
    for (std::size_t arm = 0; arm < 2; ++arm) {
      time(arm, "fixed_ab_warmup", pair, arm, ns);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
  std::array<double, 9> before{}, after{};
  std::array<std::array<double, 9>, 2> samples{};
  for (std::size_t i = 0; i < 9; ++i) {
    time(1, "bookend_before", -1, i, before[i]);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
  for (int pair = -4; pair < 9; ++pair)
    for (std::size_t order = 0; order < 2; ++order) {
      const auto arm = (static_cast<std::size_t>(pair + 4) + order) % 2;
      time(arm, pair < 0 ? "alternating_warmup" : "paired_timing", pair, order, ns);
      ASSERT_FALSE(::testing::Test::HasFailure());
      if (pair >= 0) samples[arm][static_cast<std::size_t>(pair)] = ns;
    }
  for (std::size_t i = 0; i < 9; ++i) {
    time(1, "bookend_after", -1, i, after[i]);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
  const auto median = [](auto values) {
    std::ranges::sort(values);
    return values[4];
  };
  const auto first = median(before), last = median(after);
  ASSERT_LE(std::max(first, last), 1.1 * std::min(first, last)) << "no retry after drift refusal";
  std::println(
      "{{\"event\":\"whole_state_timing_result\",\"shape_class\":{},"
      "\"protocol\":{},\"incumbent_median_ns\":{},\"alternative_median_ns\":{},"
      "\"bookend_before_ns\":{},\"bookend_after_ns\":{},\"pairs\":9}}",
      TimingQuote(shape_class), TimingQuote(digest), median(samples[0]), median(samples[1]), first,
      last);
}

// Test-owned rank data over the same immutable native implementation inventory.
// Keep the registry alive after its adopted program, just as production does.
struct WholeAdoption {
  std::unique_ptr<llmp::execution::Registry> registry;
  std::unique_ptr<kn::StepProgram> program;
};

std::expected<WholeAdoption, std::string> AdoptWholeAlternative(
    const co::StepGraph& step, const planner::StepShape& shape,
    const planner::ComparisonInventory& inventory, const co::ComparisonBoundaryView& source,
    const kn::StepBackend& backend, const kn::Addresses& addresses, planner::PlanOptions options,
    const kn::ComparisonRecipeSeal& incumbent, const kn::ComparisonRecipeSeal& alternative,
    std::shared_ptr<const ir::StorageFunder> funder, const planner::Policy& policy = {},
    std::uint32_t context = 0) {
  const auto& native = backend.owner ? backend.owner->backend() : backend;
  std::vector<std::uint32_t> offsets{0};
  std::vector<ir::NodeId> members;
  for (const auto& group : inventory.site.accepted_member_origins) {
    members.insert(members.end(), group.begin(), group.end());
    offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }
  auto boundary = co::MakeComparisonRegionBoundary({source, offsets, members}, std::move(funder));
  if (!boundary) return std::unexpected(std::string(boundary.error().detail));
  auto key = en::RecipeComparisonKeyForWholePrograms(
      inventory.source_boundary, *boundary, inventory.site, shape.phase,
      native.registry->SelectionDigest(), incumbent, alternative);
  if (!key) return std::unexpected("actual whole pair has no authenticated exact key");
  std::vector<llmp::execution::Implementation> declarations;
  for (std::size_t i = 0; i < native.registry->size(); ++i)
    declarations.push_back(native.registry->at(i));
  auto ranks = native.registry->ranks();
  ranks.recipe_comparisons = {
      {.key = *key,
       .incumbent_ns = 100,
       .alternative_ns = 50,
       .samples = 3,
       .protocol = llmp::base::Sha256{}.Update("whole-state-test-protocol").Finish(),
       .receipt = llmp::base::Sha256{}.Update("whole-state-test-receipt").Finish(),
       .provenance = "synthetic-reviewed-test",
       .shape_class = "owned-state-pair"}};
  auto made = llmp::execution::Registry::Create(std::move(declarations), std::move(ranks));
  if (!made) return std::unexpected(std::string(made.error().detail));
  WholeAdoption result{.registry = std::make_unique<llmp::execution::Registry>(std::move(*made)),
                       .program = {}};
  auto measured = native;
  measured.registry = result.registry.get();
  auto program =
      en::BuildCumulativeRecipeProgram(co::GraphIO(step), step, shape, 0, context, measured, policy,
                                       addresses, std::move(options), {}, backend.owner, nullptr);
  if (!program) return std::unexpected(program.error());
  const auto actual = (*program)->comparison_recipe();
  if (!actual || *actual != alternative)
    return std::unexpected("production whole lookup did not select its exact own alternative");
  result.program = std::move(*program);
  return result;
}

class PaidCudaBacking final : public kn::ResourceDependency {
 public:
  PaidCudaBacking(en::HostPayment paid, void* allocation, bool& alive)
      : paid_(std::move(paid)), allocation_(allocation), alive_(alive) {
    alive_ = true;
  }
  static std::expected<kn::ResourceKeep, std::string> Make(en::PagedNode& node,
                                                           CudaBuffer& allocation, bool& alive) {
    auto paid = node.PrepayHost(sizeof(PaidCudaBacking) + 64);
    if (!paid) return std::unexpected(paid.error());
    auto* backing = new (std::nothrow) PaidCudaBacking(std::move(*paid), nullptr, alive);
    if (!backing) return std::unexpected("private CUDA backing owner allocation failed");
    backing->allocation_ = allocation.Release();
    return backing->Adopt();
  }

 private:
  void Destroy() override {
    auto paid = std::move(paid_);
    void* allocation = allocation_;
    bool& alive = alive_;
    delete this;
    EXPECT_EQ(cudaFree(allocation), cudaSuccess);
    alive = false;
    paid.Reset();
  }
  en::HostPayment paid_;
  void* allocation_ = nullptr;
  bool& alive_;
};

class ForwardingDeviceExecution final : public llmp::providers::DeviceExecution {
 public:
  explicit ForwardingDeviceExecution(llmp::providers::DeviceExecution& inner) : inner_(inner) {}
  void FailNextRecord() { fail_next_record_ = true; }
  void ReturnPendingNextQuery() { return_pending_next_query_ = true; }
  std::expected<llmp::providers::StreamId, llmp::providers::Failure> CreateStream() override {
    return inner_.CreateStream();
  }
  std::expected<void, llmp::providers::Failure> DestroyStream(
      llmp::providers::StreamId stream) override {
    return inner_.DestroyStream(stream);
  }
  std::expected<void, llmp::providers::Failure> Copy(llmp::providers::StreamId stream,
                                                     std::uint64_t destination,
                                                     std::uint64_t source,
                                                     llmp::base::Bytes size) override {
    return inner_.Copy(stream, destination, source, size);
  }
  std::expected<void, llmp::providers::Failure> Zero(llmp::providers::StreamId stream,
                                                     std::uint64_t destination,
                                                     llmp::base::Bytes size) override {
    return inner_.Zero(stream, destination, size);
  }
  std::expected<llmp::providers::NativeStream, llmp::providers::Failure> Submission(
      llmp::providers::StreamId stream) override {
    return inner_.Submission(stream);
  }
  std::expected<void, llmp::providers::Failure> Wait(llmp::providers::StreamId stream,
                                                     llmp::providers::FenceId fence) override {
    return inner_.Wait(stream, fence);
  }
  std::expected<llmp::providers::FenceId, llmp::providers::Failure> Record(
      llmp::providers::StreamId stream) override {
    if (std::exchange(fail_next_record_, false))
      return std::unexpected(
          llmp::providers::Failure{.error = llmp::providers::ProviderError::kUnknown,
                                   .detail = "test injected unknown fence-record outcome"});
    return inner_.Record(stream);
  }
  std::expected<llmp::providers::FenceState, llmp::providers::Failure> Query(
      llmp::providers::FenceId fence) override {
    if (std::exchange(return_pending_next_query_, false))
      return llmp::providers::FenceState::kPending;
    return inner_.Query(fence);
  }
  std::expected<void, llmp::providers::Failure> Release(llmp::providers::FenceId fence) override {
    return inner_.Release(fence);
  }

 private:
  llmp::providers::DeviceExecution& inner_;
  bool fail_next_record_ = false;
  bool return_pending_next_query_ = false;
};

struct Life final : en::PagedModel {
  en::PagedNode node{{.compute_streams = 1, .slots = 1}};
  std::unique_ptr<ForwardingDeviceExecution> execution;
  en::RunnerResources resources{node, 0, 0};
  std::unique_ptr<kg::GgmlBackend> backend;
  // Keep every allocation and program reachable by submitted token work in
  // the model owner until PagedNode has fenced the stream.
  CudaBuffer test_activation;
  CudaBuffer test_weights;
  CudaBuffer test_weights_rebound;
  CudaBuffer test_state;
  CudaBuffer test_state_rebound;
  CudaBuffer test_pool;
  CudaBuffer test_persistent;
  std::unique_ptr<kn::StepLaunch> test_launch;
  std::unique_ptr<kn::StepProgram> test_program;
  std::unique_ptr<kn::StepProgram> test_peer_program;
  std::unique_ptr<en::ComparisonPair> test_pair;
  std::optional<kn::ParentResources> test_parent_resources;
  kg::SelectedWorkspace selected_state{};
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  bool opened = false;
  bool defer_release = false;
  bool released = false;
  bool backing_alive = false;
  std::string release_error;
  llmp::catalog::Closure empty;
  Life* quarantine_next = nullptr;

  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return empty; }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override { return {}; }
  en::Status RetireAfterFence() override {
    const bool had_launch = test_launch != nullptr;
    test_peer_program.reset();
    test_program.reset();
    test_pair.reset();
    test_parent_resources.reset();
    test_launch.reset();
    if (had_launch && backend) backend->LaunchReleased();
    test_persistent.Free();
    test_pool.Free();
    test_state_rebound.Free();
    test_state.Free();
    test_weights_rebound.Free();
    test_weights.Free();
    test_activation.Free();
    return {};
  }
  en::Status Release() override {
    if (defer_release) return {};
    return ReleaseNow();
  }
  en::Status ReleaseNow() {
    if (released) return {};
    backend.reset();
    std::vector<std::string> errors;
    resources.Release(errors);
    if (!errors.empty()) return std::unexpected(errors.front());
    released = true;
    opened = false;
    return {};
  }

  static Life*& QuarantineHead() {
    static Life* head = nullptr;
    return head;
  }
  static void Quarantine(std::unique_ptr<Life> owner) {
    auto* retained = owner.release();
    retained->quarantine_next = QuarantineHead();
    QuarantineHead() = retained;
  }
};

class GgmlComparisonLowerForGpu : public testing::Test {
 protected:
  void CheckGdnNormGateReadyKey(bool expanded);
  void SetUp() override {
    life_ = std::make_unique<Life>();
    auto opened = life_->node.Open();
    ASSERT_TRUE(opened) << (opened ? "" : opened.error());
    life_->opened = true;
    life_->execution = std::make_unique<ForwardingDeviceExecution>(life_->node.execution());
    const auto available = llmp::platform::AvailableMemoryBytes();
    ASSERT_TRUE(available);
    auto admitted = en::ExecutionMetadataAllowance::Admit(
        life_->node.catalog(), life_->node.domain(), 16 * kMiB, *available, 6ULL << 30U);
    ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
    life_->paid = std::move(*admitted);
    kg::GgmlBackendOptions options{.lower = {.expert_stride = [](const ir::WeightRef& ref) {
                                     return ref.expert_array ? std::uint64_t{2048}
                                                             : std::uint64_t{0};
                                   }}};
    auto backend = kg::GgmlBackend::Create(0, *life_->execution, life_->node.stream(0), nullptr,
                                           std::move(options));
    ASSERT_TRUE(backend) << (backend ? "" : backend.error());
    life_->backend = std::move(*backend);
    const auto charge = [paid = life_->paid](std::uint64_t bytes) { return paid->Charge(bytes); };
    const auto release = [paid = life_->paid](std::uint64_t bytes) { paid->Release(bytes); };
    ASSERT_TRUE(life_->resources.OpenLibrary("comparison proof declaration workspace"));
    ASSERT_TRUE(life_->backend->FundConstruction(charge, release));
    ASSERT_TRUE(life_->backend->Declare({}, life_->resources.library_bytes(), charge, release));
    ASSERT_TRUE(life_->backend->LendLibrary(&life_->resources.library()));
  }

  void TearDown() override {
    if (!life_) return;
    if (life_->opened) {
      const std::array<en::PagedModel*, 1> models{life_.get()};
      if (!runner_active_) life_->defer_release = false;
      auto retired = life_->node.TearDown(models);
      EXPECT_TRUE(retired) << (retired ? "" : retired.error());
      if (runner_active_ && retired) {
        runner_active_ = false;
      }
      if (!retired) {
        Life::Quarantine(std::move(life_));
        return;
      }
    }
    life_.reset();
  }

  struct Fixture {
    ir::Graph graph;
    std::optional<llmp::planner::StepPlan> plan;
    ir::TensorId input{};
    kn::Addresses addresses{};

    co::GraphBuildIO io() const { return {graph, graph.inputs(), graph.outputs(), {}}; }
  };

  Fixture MakeFixture() {
    ir::Builder builder;
    builder.SetOwner(0);
    const auto input =
        builder.Input(llmp::ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 4})));
    builder.Output(builder.RmsNorm(input, 1.0e-6F));
    auto graph = std::move(builder).Finish().value();
    co::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), {}};
    kn::Addresses addresses{};  // owned by Fixture for the later synchronous lower call
    auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    auto plan = en::PlanComponentGraph(io, recipe, {}, addresses);
    EXPECT_TRUE(plan) << (plan ? "" : plan.error());
    if (!plan) return {std::move(graph), std::nullopt, input, addresses};
    return {std::move(graph), std::optional<llmp::planner::StepPlan>(std::move(*plan)), input,
            addresses};
  }

  std::unique_ptr<Life> life_;
  bool runner_active_ = false;
};

// The runner below must exercise its ordinary row-table transaction on a CUDA
// stream. FakeStepSource's host callback is intentionally not used for that
// execution; this adapter retains its real slots/bounds and queues only an
// output clear on the runner's stream.
struct RowTableParentObservation {
  std::optional<kn::InputSlot> side_rows;
  struct Table {
    std::uint64_t address = 0;
    std::uint64_t rows = 0;
    std::optional<kn::BoundedAddress> range;
    std::optional<co::ComparisonRowStorage> storage;
  } tables[2];
};

class CudaOutputOnlyParentProgram final : public kn::StepProgram {
 public:
  CudaOutputOnlyParentProgram(std::unique_ptr<kn::StepProgram> inner,
                              std::shared_ptr<RowTableParentObservation> observation)
      : inner_(std::move(inner)), observation_(std::move(observation)) {
    for (const auto& input : inner_->inputs()) {
      if (input.kind == kn::InputKind::kSideRows && input.component == 17) {
        observation_->side_rows = input;
        break;
      }
    }
  }
  std::uint64_t scratch_bytes() const override { return inner_->scratch_bytes(); }
  std::uint64_t pool_bytes() const override { return inner_->pool_bytes(); }
  std::uint64_t host_bytes() const override { return inner_->host_bytes() + sizeof(*this); }
  std::uint64_t launches() const override { return inner_->launches(); }
  std::span<const kn::InputSlot> inputs() const override { return inner_->inputs(); }
  std::span<const kn::OutputSlot> outputs() const override { return inner_->outputs(); }
  std::span<const kn::TensorUse> tensors() const override { return inner_->tensors(); }
  kn::PlanCoverage plan_coverage() const override { return inner_->plan_coverage(); }
  bool capturable() const override { return false; }
  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch& launch) override {
    return inner_->Bind(launch);
  }
  std::expected<void, kn::StepFailure> Position(std::span<const std::uint32_t> past) override {
    return inner_->Position(past);
  }
  std::expected<void, kn::StepFailure> TryUpdateGeneratedInputs(
      const planner::StepShape& shape) override {
    return inner_->TryUpdateGeneratedInputs(shape);
  }
  std::expected<void, kn::StepFailure> Run(kn::StepLaunch&,
                                           llmp::providers::NativeStream stream) const override {
    for (const auto& output : inner_->outputs()) {
      if (output.bytes == 0 || output.device_address == 0) continue;
      const auto status = cudaMemsetAsync(reinterpret_cast<void*>(output.device_address), 0,
                                          output.bytes, static_cast<cudaStream_t>(stream.handle));
      if (status != cudaSuccess)
        return std::unexpected(
            kn::StepFailure{.error = kn::StepError::kUnknown,
                            .detail = "CUDA row-table parent output clear failed"});
    }
    return {};
  }

 private:
  std::unique_ptr<kn::StepProgram> inner_;
  std::shared_ptr<RowTableParentObservation> observation_;
};

class CudaRowTableParentSource final : public en::StepSource {
 public:
  CudaRowTableParentSource(ts::FakeSourceOptions options,
                           std::shared_ptr<RowTableParentObservation> observation,
                           std::array<ir::WeightRef, 2> table_refs)
      : inner_(std::make_unique<ts::FakeStepSource>(std::move(options))),
        observation_(std::move(observation)),
        table_refs_(table_refs) {}
  planner::PlanKey KeyBase() const override { return inner_->KeyBase(); }
  en::Status PrepareExecution() override { return inner_->PrepareExecution(); }
  en::Status CheckExecutionUse(en::NumericalUse use) const override {
    return inner_->CheckExecutionUse(use);
  }
  en::Status CheckExecutionShape(const planner::StepShape& shape) const override {
    return inner_->CheckExecutionShape(shape);
  }
  std::expected<std::optional<llmp::base::Sha256Digest>, std::string> ExecutionIdentity()
      const override {
    return inner_->ExecutionIdentity();
  }
  bool UsesSideRows(std::uint16_t stage) const override { return inner_->UsesSideRows(stage); }
  en::Status SideRows(std::uint16_t stage, std::uint16_t component,
                      std::span<const std::int32_t> history, std::uint32_t position,
                      std::span<std::int32_t> out) const override {
    return inner_->SideRows(stage, component, history, position, out);
  }
  std::expected<std::vector<en::CommitBuffer>, std::string> CommitBuffers(
      std::uint16_t stage, std::uint32_t rows) const override {
    return inner_->CommitBuffers(stage, rows);
  }
  std::expected<std::unique_ptr<kn::StepProgram>, std::string> Build(
      const planner::StepShape& shape, const kn::Addresses& addresses) override {
    if (!addresses.measuring()) {
      for (std::size_t i = 0; i < table_refs_.size(); ++i) {
        observation_->tables[i].address =
            addresses.weight(table_refs_[i], llmp::format::PartRole::kData);
        observation_->tables[i].rows = addresses.weight_rows(table_refs_[i]);
        observation_->tables[i].range =
            addresses.weight_range(table_refs_[i], llmp::format::PartRole::kData);
        observation_->tables[i].storage =
            addresses.row_storage(table_refs_[i], llmp::format::PartRole::kData);
      }
    }
    auto inner = inner_->Build(shape, addresses);
    if (!inner) return std::unexpected(inner.error());
    auto wrapped = std::unique_ptr<kn::StepProgram>(
        new (std::nothrow) CudaOutputOnlyParentProgram(std::move(*inner), observation_));
    if (!wrapped) return std::unexpected("row-table parent wrapper allocation failed");
    return wrapped;
  }

 private:
  std::unique_ptr<ts::FakeStepSource> inner_;
  std::shared_ptr<RowTableParentObservation> observation_;
  std::array<ir::WeightRef, 2> table_refs_;
};

llmp::format::TensorFormat RowTableFormat(llmp::format::Encoding encoding) {
  return {.encoding = encoding,
          .layout = llmp::format::Layout::kDense,
          .shape = {.ne = {16, 8, 1, 1}, .rank = 2}};
}

std::expected<std::filesystem::path, std::string> PublishRowTableArtifact(
    const std::filesystem::path& store, llmp::format::TensorFormat f32,
    llmp::format::TensorFormat f16, std::span<const float> f32_rows,
    std::span<const __half> f16_rows) {
  constexpr std::uint64_t kGroupBytes = 4096;
  llmp::test::V1Plan plan;
  plan.groups = {{.kind = llmp::artifact::GroupKind::kGlobal,
                  .resources = {{.name = "side_f32", .format = f32, .rows = true}}},
                 {.kind = llmp::artifact::GroupKind::kGlobal,
                  .resources = {{.name = "side_f16", .format = f16, .rows = true}}}};
  auto files = llmp::test::Render(plan);
  if (files.shards.size() != 1 || files.index["chunk_sha256"].array().size() != 2)
    return std::unexpected("two row-table resources did not render into expected chunks");
  std::uint64_t header_bytes = 0;
  if (files.shards[0].size() < sizeof(header_bytes))
    return std::unexpected("row-table shard has a truncated header");
  std::memcpy(&header_bytes, files.shards[0].data(), sizeof(header_bytes));
  std::uint64_t data_offset = 0;
  if (__builtin_add_overflow(header_bytes, std::uint64_t{sizeof(header_bytes)}, &data_offset) ||
      data_offset > files.shards[0].size() ||
      2 * kGroupBytes > files.shards[0].size() - data_offset ||
      f32_rows.size_bytes() > kGroupBytes || f16_rows.size_bytes() > kGroupBytes)
    return std::unexpected("row-table shard data range is invalid");
  std::memcpy(files.shards[0].data() + data_offset, f32_rows.data(), f32_rows.size_bytes());
  std::memcpy(files.shards[0].data() + data_offset + kGroupBytes, f16_rows.data(),
              f16_rows.size_bytes());
  for (std::size_t group = 0; group < 2; ++group) {
    const std::string_view bytes(files.shards[0].data() + data_offset + group * kGroupBytes,
                                 kGroupBytes);
    files.index["chunk_sha256"].array()[group] =
        llmp::base::ToHex(llmp::base::Sha256{}.Update(bytes).Finish());
  }
  return llmp::test::Publish(store, files);
}

TEST_F(GgmlComparisonLowerForGpu, OwnerPreparedSideRowsDriveF32AndF16GetRowsPairs) {
  constexpr std::uint32_t kWidth = 16;
  constexpr std::uint32_t kLogicalRows = 8;
  constexpr std::uint32_t kRequestRows = 4;
  constexpr std::uint64_t kArenaBytes = 1U << 20U;
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  ASSERT_NE(scratch, nullptr);

  std::array<float, kWidth * kLogicalRows> f32_table{};
  std::array<__half, kWidth * kLogicalRows> f16_table{};
  for (std::size_t row = 0; row < kLogicalRows; ++row) {
    for (std::size_t col = 0; col < kWidth; ++col) {
      const float value =
          (static_cast<float>(row) - 3.0F) * 0.125F + (static_cast<float>(col) - 7.0F) * 0.03125F;
      f32_table[row * kWidth + col] = value;
      f16_table[row * kWidth + col] = __float2half(value);
    }
  }
  auto artifact_path =
      PublishRowTableArtifact(std::filesystem::path(scratch) / "row-table-comparison-owner",
                              RowTableFormat(llmp::format::Encoding::kF32),
                              RowTableFormat(llmp::format::Encoding::kF16), f32_table, f16_table);
  ASSERT_TRUE(artifact_path) << (artifact_path ? "" : artifact_path.error());
  auto weights = std::make_unique<en::PagedWeights>();
  ASSERT_TRUE(weights->Open(*artifact_path));

  const std::array refs{ir::WeightRef{.stage = 0, .component = 17, .role = 17, .layer = 0},
                        ir::WeightRef{.stage = 0, .component = 17, .role = 18, .layer = 0}};
  auto observation = std::make_shared<RowTableParentObservation>();
  const std::array<std::int32_t, 4> first_rows{7, 2, 7, 5};
  const std::array<std::int32_t, 4> second_rows{3, 7, 3, 2};
  ts::FakeSourceOptions fake_options;
  const std::array<std::pair<std::uint32_t, md::StateContract>, 1> contracts{
      {{0, {.tensors = {{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 16}}}}}};
  auto layout = md::Layout(contracts, 256, kRequestRows);
  ASSERT_TRUE(layout) << (layout ? "" : layout.error());
  fake_options.layouts = {*layout};
  fake_options.extra_inputs = [](const planner::StepShape& shape) {
    std::uint64_t rows = 0;
    for (const auto& segment : shape.segments) rows += segment.rows;
    return std::vector<kn::InputSlot>{{.bytes = rows * sizeof(std::int32_t),
                                       .segment = ir::kJoinedSegment,
                                       .kind = kn::InputKind::kSideRows,
                                       .component = 17}};
  };
  auto parent_source =
      std::make_unique<CudaRowTableParentSource>(std::move(fake_options), observation, refs);
  en::RunnerConfig config;
  config.slots = {.model = "row-table-comparison-parent",
                  .slots = 1,
                  .stages = {{.name = "target", .layout = *layout}},
                  .read_align = 256,
                  .spill_place = {},
                  .spill_directory = scratch,
                  .invalidate = {},
                  .drop_plans = {}};
  const planner::StepShape parent_shape{.stage = 0,
                                        .phase = planner::Phase::kDecode,
                                        .segments = {{.slot = 0,
                                                      .rows = kRequestRows,
                                                      .past = 0,
                                                      .read_cells = 256,
                                                      .outputs = kRequestRows}}};
  config.setup_shapes = {parent_shape};
  config.weights.push_back({.weights = std::move(weights),
                            .places = {en::GroupPlace::kNone, en::GroupPlace::kNone},
                            .slabs = {}});
  config.row_tables = {{.weight = refs[0],
                        .resource = 0,
                        .format = RowTableFormat(llmp::format::Encoding::kF32),
                        .part = llmp::format::PartRole::kData,
                        .rows_per_token = 1},
                       {.weight = refs[1],
                        .resource = 1,
                        .format = RowTableFormat(llmp::format::Encoding::kF16),
                        .part = llmp::format::PartRole::kData,
                        .rows_per_token = 1}};
  const auto persistent_bytes = life_->backend->persistent_bytes();
  if (persistent_bytes != 0)
    config.kernel_memory.push_back(
        {.name = "row-table comparison launch state", .bytes = persistent_bytes});
  config.prepare = [this, persistent_bytes](
                       std::span<const std::uint64_t> bases) -> std::expected<void, std::string> {
    if (bases.size() != static_cast<std::size_t>(persistent_bytes != 0))
      return std::unexpected("row-table parent launch state was not mapped");
    life_->selected_state = {persistent_bytes != 0 ? bases.front() : 0, persistent_bytes};
    return {};
  };
  config.retire = [this] {
    life_->selected_state = {};
    auto released = life_->ReleaseNow();
    if (!released) life_->release_error = released.error();
  };
  config.graphs = false;
  config.launch =
      [this](
          std::uint64_t pool_bytes) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
    auto launch = life_->backend->Launch(
        {.base = life_->node.pool().base, .size = llmp::base::Bytes(pool_bytes)},
        life_->selected_state);
    if (launch) life_->defer_release = true;
    return launch;
  };
  auto created =
      en::ModelRunner::Create(life_->node, std::move(config), std::move(parent_source), 0, 0);
  ASSERT_TRUE(created) << (created ? "" : created.error());
  auto adopted = en::RunnerOwner::Adopt(life_->node, *created);
  ASSERT_TRUE(adopted) << (adopted ? "" : adopted.error());
  auto runner = std::move(*adopted);
  runner_active_ = true;
  ASSERT_TRUE(runner->Setup());
  ASSERT_TRUE(life_->node.MapWorkspace(runner->activations_needed(),
                                       std::max(runner->pool_needed(), en::kPagedExtent)));
  const auto fixed = life_->node.catalog().OccupancyOf(life_->node.domain()).Total().value();
  life_->node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() +
                           (16ULL << 20U));
  ASSERT_TRUE(life_->node.Start(llmp::base::Bytes(fixed + 32 * en::kPagedExtent)));
  ASSERT_TRUE(runner->Register());
  life_->node.Run();
  ASSERT_TRUE(runner->Bind());
  auto kept_parent = runner->KeepResources();
  ASSERT_TRUE(kept_parent) << (kept_parent ? "" : kept_parent.error().what);
  life_->test_parent_resources.emplace(std::move(*kept_parent));

  ASSERT_TRUE(observation->side_rows);
  ASSERT_EQ(observation->side_rows->bytes, kRequestRows * sizeof(std::int32_t));
  std::array<std::array<std::int32_t, kRequestRows>, 2> dense_frames{};
  auto run_parent_rows = [&](std::span<const std::int32_t> logical_ids,
                             std::array<std::int32_t, kRequestRows>& dense_ids) {
    en::StepRequest request{
        .shape = parent_shape, .side_rows = logical_ids, .advance = false, .ahead = {}};
    const std::array<std::int32_t, kRequestRows> tokens{1, 2, 3, 4};
    const std::array<en::SegmentWork, 1> work{en::SegmentWork{.tokens = tokens, .history = tokens}};
    request.work = work;
    auto run = runner->RunStep(request);
    ASSERT_TRUE(run) << run.error().what;
    auto stream = life_->node.execution().Submission(life_->node.stream(0));
    ASSERT_TRUE(stream) << (stream ? "" : stream.error().detail);
    ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dense_ids.data(),
                         reinterpret_cast<const void*>(observation->side_rows->device_address),
                         sizeof(dense_ids), cudaMemcpyDeviceToHost),
              cudaSuccess);
    std::vector<std::int32_t> unique(logical_ids.begin(), logical_ids.end());
    std::ranges::sort(unique);
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    std::array<std::int32_t, kRequestRows> expected_dense{};
    for (std::size_t i = 0; i < logical_ids.size(); ++i)
      expected_dense[i] = static_cast<std::int32_t>(
          std::ranges::lower_bound(unique, logical_ids[i]) - unique.begin());
    EXPECT_EQ(dense_ids, expected_dense);
    for (std::size_t table_index = 0; table_index < 2; ++table_index) {
      const auto& observed_table = observation->tables[table_index];
      ASSERT_NE(observed_table.address, 0U);
      ASSERT_TRUE(observed_table.range);
      ASSERT_TRUE(observed_table.storage);
      EXPECT_EQ(observed_table.range->base, observed_table.address);
      EXPECT_EQ(observed_table.rows, observed_table.storage->slot_capacity);
      EXPECT_EQ(observed_table.range->bytes,
                observed_table.storage->slot_capacity * observed_table.storage->row_bytes);
      EXPECT_EQ(observed_table.storage->logical_rows, kLogicalRows);
      std::vector<std::byte> gathered(static_cast<std::size_t>(observed_table.range->bytes));
      ASSERT_EQ(cudaMemcpy(gathered.data(), reinterpret_cast<const void*>(observed_table.address),
                           gathered.size(), cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (std::size_t slot = 0; slot < unique.size(); ++slot) {
        for (std::size_t col = 0; col < kWidth; ++col) {
          float actual = 0.0F;
          const auto at =
              slot * observed_table.storage->row_bytes + col * (table_index == 1 ? 2 : 4);
          if (table_index == 1) {
            __half value{};
            std::memcpy(&value, gathered.data() + at, sizeof(value));
            actual = __half2float(value);
          } else {
            std::memcpy(&actual, gathered.data() + at, sizeof(actual));
          }
          const auto source_index = static_cast<std::size_t>(unique[slot]) * kWidth + col;
          const float expected =
              table_index == 1 ? __half2float(f16_table[source_index]) : f32_table[source_index];
          EXPECT_FLOAT_EQ(actual, expected);
        }
      }
    }
  };
  ASSERT_NO_FATAL_FAILURE(run_parent_rows(first_rows, dense_frames[0]));
  // Establish sentinels in slots that the first gather does not populate. A
  // second normal transaction restores the actual gathered rows while keeping
  // the unused slot bytes observable for the later full-range immutability
  // checks.
  for (const auto& table : observation->tables) {
    ASSERT_TRUE(table.range);
    ASSERT_NE(table.address, 0U);
    ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(table.address), 0x5a,
                         static_cast<std::size_t>(table.range->bytes)),
              cudaSuccess);
  }
  ASSERT_NO_FATAL_FAILURE(run_parent_rows(first_rows, dense_frames[0]));

  auto run_case = [&](std::size_t table_index, llmp::format::Encoding encoding,
                      std::span<const float> logical_table, std::span<const std::int32_t> first_ids,
                      std::span<const std::int32_t> second_ids) {
    ASSERT_NO_FATAL_FAILURE(run_parent_rows(first_ids, dense_frames[0]));
    const ir::WeightRef table_ref = refs[table_index];
    const auto table_format = RowTableFormat(encoding);
    ir::Builder builder;
    builder.SetComponent(17);
    builder.SetOwner(0);
    const auto ids = builder.Input(
        ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({kRequestRows})), "dense-side-rows");
    const auto table = builder.Weight(table_format, table_ref, "row-table");
    const ir::WeightRef scale_ref{.stage = 0, .component = 17, .role = 19, .layer = 0};
    const auto scale = builder.Weight(
        ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth})), scale_ref, "rms-scale");
    const auto gathered = builder.GetRows(table, ids);
    const auto normalized = builder.RmsNorm(gathered, 1.0e-6F);
    const auto output = builder.Mul(normalized, scale);
    builder.Output(output);
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    co::StepGraph step{.graph = std::move(*graph),
                       .inputs = {{.tensor = ids,
                                   .kind = co::InputKind::kSideRows,
                                   .segment = ir::kJoinedSegment,
                                   .component = 17}},
                       .weights = {{.tensor = table,
                                    .ref = table_ref,
                                    .parts = {{"row-table", llmp::format::PartRole::kData}}},
                                   {.tensor = scale,
                                    .ref = scale_ref,
                                    .parts = {{"rms-scale", llmp::format::PartRole::kData}}}},
                       .state = {},
                       .logits = output};
    const planner::StepShape shape{.stage = 0,
                                   .phase = planner::Phase::kDecode,
                                   .segments = {{.slot = 0,
                                                 .rows = kRequestRows,
                                                 .past = 0,
                                                 .read_cells = 256,
                                                 .outputs = kRequestRows}}};
    const std::uint64_t scale_bytes = kWidth * sizeof(float);
    ASSERT_TRUE(life_->test_activation.Allocate(kArenaBytes));
    ASSERT_TRUE(life_->test_weights.Allocate(kArenaBytes));
    std::vector<float> scale_values(kWidth);
    for (std::size_t i = 0; i < scale_values.size(); ++i)
      scale_values[i] = 0.75F + static_cast<float>(i % 7) * 0.03125F;
    ASSERT_EQ(cudaMemcpy(life_->test_weights.pointer, scale_values.data(), scale_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    const auto table_owner = observation->tables[table_index];
    const auto activation_address = life_->test_activation.address();
    const auto scale_address = life_->test_weights.address();
    ASSERT_TRUE(life_->test_parent_resources);
    ASSERT_TRUE(life_->test_parent_resources->ready());
    auto* const comparison_launch = life_->test_parent_resources->launch();
    ASSERT_NE(comparison_launch, nullptr);
    kn::Addresses addresses{};
    addresses.weight = [table_ref, table_owner, scale_ref, scale_address](
                           const ir::WeightRef& ref, llmp::format::PartRole part) {
      if (part != llmp::format::PartRole::kData) return std::uint64_t{0};
      if (ref == table_ref) return table_owner.address;
      if (ref == scale_ref) return scale_address;
      return std::uint64_t{0};
    };
    addresses.weight_rows = [table_ref, table_owner](const ir::WeightRef& ref) {
      return ref == table_ref ? table_owner.rows : std::uint64_t{0};
    };
    addresses.weight_range = [table_ref, table_owner, scale_ref, scale_address](
                                 const ir::WeightRef& ref,
                                 llmp::format::PartRole part) -> std::optional<kn::BoundedAddress> {
      if (part != llmp::format::PartRole::kData) return std::nullopt;
      if (ref == table_ref) return table_owner.range;
      if (ref == scale_ref)
        return kn::BoundedAddress{.base = scale_address, .bytes = scale_bytes, .alignment = 16};
      return std::nullopt;
    };
    addresses.row_storage =
        [table_ref, table_owner](
            const ir::WeightRef& ref,
            llmp::format::PartRole part) -> std::optional<co::ComparisonRowStorage> {
      if (ref == table_ref && part == llmp::format::PartRole::kData) return table_owner.storage;
      return std::nullopt;
    };
    addresses.activations = activation_address;
    addresses.activation_bytes = kArenaBytes;
    addresses.activation_alignment = 256;
    auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    ASSERT_NE(recipe.registry, nullptr);
    auto funded = step.graph.Clone(std::make_shared<PaidBoundaryFunder>(life_->paid));
    ASSERT_TRUE(funded) << (funded ? "" : funded.error().detail);
    step.graph = std::move(*funded);
    auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
    const auto* pattern = planner::FindPattern("rms_norm_mul");
    ASSERT_NE(pattern, nullptr);
    std::vector<std::vector<ir::NodeId>> members;
    for (std::uint32_t i = 0; i < step.graph.nodes().size(); ++i) {
      const auto& node = step.graph.node(ir::NodeId{i});
      if (node.key == ir::Primitive(ir::Op::kRmsNorm)) members.push_back({ir::NodeId{i}});
      if (node.key == ir::Primitive(ir::Op::kMul)) members.push_back({ir::NodeId{i}});
    }
    ASSERT_EQ(members.size(), 2U);
    auto pair_result = en::ComparisonPair::Create(
        funder, std::move(step), planner::StepShape(shape),
        [&] {
          std::vector<co::ComparisonExternalBinding> external;
          const auto id_bytes = kRequestRows * sizeof(std::int32_t);
          external.push_back({.original_root = ids,
                              .allocation_class = 1,
                              .allocation_bytes = id_bytes,
                              .bytes = id_bytes,
                              .alignment = 4});
          const auto storage = *table_owner.storage;
          const auto slot_bytes = table_owner.range->bytes;
          external.push_back({.original_root = table,
                              .part = llmp::format::PartRole::kData,
                              .allocation_class = 2,
                              .allocation_bytes = slot_bytes,
                              .bytes = slot_bytes,
                              .alignment = table_owner.range->alignment,
                              .resource_part_identity = storage.content_identity,
                              .row_storage = storage});
          external.push_back({.original_root = scale,
                              .part = llmp::format::PartRole::kData,
                              .allocation_class = 3,
                              .allocation_bytes = scale_bytes,
                              .bytes = scale_bytes,
                              .alignment = 16});
          return external;
        }(),
        256, 0, en::ComparisonInputDomain::kTokenStep);
    ASSERT_TRUE(pair_result) << (pair_result ? "" : pair_result.error().detail);
    life_->test_pair = std::move(*pair_result);
    auto& pair = life_->test_pair;
    planner::ComparisonSiteSelector selector{
        .source_graph_identity = pair->source().graph.identity(),
        .kind = planner::ComparisonSiteKind::kRewrite,
        .pattern = "rms_norm_mul",
        .version = pattern->version,
        .member_origins = members,
        .member_owner_segments = {},
        .boundary_ports = {}};
    ASSERT_TRUE(pair->Plan(selector, *recipe.registry, recipe.device, {},
                           {.capturable = true, .within = recipe.within}));
    const auto& incumbent_plan = pair->plan(en::ComparisonPairArm::kIncumbent)->plan;
    const auto& alternative_plan = pair->plan(en::ComparisonPairArm::kAlternative)->plan;
    const auto selected_names = [&](const planner::StepPlan& plan) {
      std::vector<std::string> names;
      names.reserve(plan.steps.size());
      for (const auto& selected : plan.steps)
        names.push_back(recipe.registry->at(selected.implementation).name);
      return names;
    };
    const auto incumbent_names = selected_names(incumbent_plan);
    const auto alternative_names = selected_names(alternative_plan);
    EXPECT_TRUE(std::ranges::any_of(
        incumbent_names, [](const auto& name) { return name == "ggml.rms_norm_mul.fused"; }));
    EXPECT_TRUE(std::ranges::any_of(alternative_names, [](const auto& name) {
      return name == "ggml.rms_norm";
    })) << "the alternative must retain its actual RMSNorm call";
    EXPECT_TRUE(std::ranges::any_of(alternative_names, [](const auto& name) {
      return name == "ggml.mul";
    })) << "the alternative must retain its actual Mul call";
    EXPECT_NE(incumbent_names, alternative_names)
        << "the proof must compare the actual fused body with its own decomposition";
    auto source_bindings = en::ComparisonSourceBindings::Create(funder, pair->source(), 0);
    ASSERT_TRUE(source_bindings) << (source_bindings ? "" : source_bindings.error().detail);
    std::array<std::vector<kn::ComparisonBindingRange>, 2> binding_storage;
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      auto resolved = source_bindings->Resolve(pair->plan(which)->plan, addresses);
      ASSERT_TRUE(resolved) << (resolved ? "" : resolved.error().detail);
      binding_storage[arm].assign(resolved->ranges().begin(), resolved->ranges().end());
    }
    const auto source_view = source_bindings->boundary_view();
    ASSERT_TRUE(source_view);
    auto full_boundary = co::MakeComparisonBoundary(*source_view, funder);
    ASSERT_TRUE(full_boundary) << (full_boundary ? "" : full_boundary.error().detail);
    ASSERT_NE(pair->inventory(), nullptr);
    std::vector<std::uint32_t> member_offsets{0};
    std::vector<ir::NodeId> source_members;
    for (const auto& accepted_group : pair->inventory()->site.accepted_member_origins) {
      ASSERT_FALSE(accepted_group.empty());
      source_members.insert(source_members.end(), accepted_group.begin(), accepted_group.end());
      ASSERT_LE(source_members.size(), std::numeric_limits<std::uint32_t>::max());
      member_offsets.push_back(static_cast<std::uint32_t>(source_members.size()));
    }
    auto region_boundary =
        co::MakeComparisonRegionBoundary({*source_view, member_offsets, source_members}, funder);
    ASSERT_TRUE(region_boundary) << (region_boundary ? "" : region_boundary.error().detail);

    std::array<kn::ComparisonRequest, 2> requests;
    std::array<kn::ComparisonRegionCapture, 2> captures;
    std::array<std::optional<planner::ComparisonOriginDag>, 2> origin_copies;
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const auto which =
          arm == 0 ? en::ComparisonPairArm::kIncumbent : en::ComparisonPairArm::kAlternative;
      const auto* owned_plan = pair->plan(which);
      ASSERT_NE(owned_plan, nullptr);
      ASSERT_TRUE(owned_plan->final_origins);
      auto copied_origins = owned_plan->final_origins->Clone(funder);
      ASSERT_TRUE(copied_origins) << (copied_origins ? "" : copied_origins.error().detail);
      origin_copies[arm].emplace(std::move(*copied_origins));
      requests[arm] = {.mode = kn::ComparisonRunMode::kFull,
                       .bindings = binding_storage[arm],
                       .state = pair->source().state,
                       .active_stage = 0,
                       .expected_boundary = &*full_boundary};
      captures[arm].seed = &*region_boundary;
      captures[arm].origins = &*origin_copies[arm];
      captures[arm].final_nodes = &owned_plan->final_node_origins;
      kn::LowerInput input(owned_plan->plan, pair->source(), pair->shape(), addresses, 256, {},
                           &requests[arm], &captures[arm]);
      auto lowered = life_->backend->Lower(input);
      ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error());
      if (arm == 0)
        life_->test_program = std::move(*lowered);
      else
        life_->test_peer_program = std::move(*lowered);
    }
    ASSERT_TRUE(life_->test_program);
    ASSERT_TRUE(life_->test_peer_program);
    const auto certificate = [&](const kn::StepProgram& program,
                                 kn::ComparisonCertificatePhase phase) {
      return program.ProjectComparisonRegionCertificate(*region_boundary, phase);
    };
    auto descriptor_left =
        certificate(*life_->test_program, kn::ComparisonCertificatePhase::kDescriptor);
    auto descriptor_right =
        certificate(*life_->test_peer_program, kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(descriptor_left.status, kn::ComparisonRegionStatus::kClosed)
        << descriptor_left.reason;
    ASSERT_EQ(descriptor_right.status, kn::ComparisonRegionStatus::kClosed)
        << descriptor_right.reason;
    ASSERT_EQ(descriptor_left.seal.state, kn::ComparisonProofState::kDescriptorVerified);
    ASSERT_EQ(descriptor_right.seal.state, kn::ComparisonProofState::kDescriptorVerified);
    auto remainder_left = life_->test_program->CompareComparisonRemainder(
        *life_->test_peer_program, *region_boundary, kn::ComparisonCertificatePhase::kDescriptor);
    auto remainder_right = life_->test_peer_program->CompareComparisonRemainder(
        *life_->test_program, *region_boundary, kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(remainder_left.status, kn::ComparisonRemainderStatus::kSame) << remainder_left.reason;
    ASSERT_EQ(remainder_right.status, kn::ComparisonRemainderStatus::kSame)
        << remainder_right.reason;
    auto ready_key = en::RecipeComparisonKeyForReadyCertificates(
        *region_boundary, ir::Fused(ir::FusedKind::kRmsNormMul), pair->shape().phase,
        recipe.registry->SelectionDigest(), descriptor_left.seal, descriptor_right.seal);
    ASSERT_TRUE(ready_key) << "actual local descriptor certificates must form the production key";

    ASSERT_NO_FATAL_FAILURE(run_parent_rows(second_ids, dense_frames[1]));
    auto second_descriptor_left =
        certificate(*life_->test_program, kn::ComparisonCertificatePhase::kDescriptor);
    auto second_descriptor_right =
        certificate(*life_->test_peer_program, kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(second_descriptor_left.status, kn::ComparisonRegionStatus::kClosed)
        << second_descriptor_left.reason;
    ASSERT_EQ(second_descriptor_right.status, kn::ComparisonRegionStatus::kClosed)
        << second_descriptor_right.reason;
    const auto second_remainder_left = life_->test_program->CompareComparisonRemainder(
        *life_->test_peer_program, *region_boundary, kn::ComparisonCertificatePhase::kDescriptor);
    const auto second_remainder_right = life_->test_peer_program->CompareComparisonRemainder(
        *life_->test_program, *region_boundary, kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(second_remainder_left.status, kn::ComparisonRemainderStatus::kSame)
        << second_remainder_left.reason;
    ASSERT_EQ(second_remainder_right.status, kn::ComparisonRemainderStatus::kSame)
        << second_remainder_right.reason;
    const auto second_key = en::RecipeComparisonKeyForReadyCertificates(
        *region_boundary, ir::Fused(ir::FusedKind::kRmsNormMul), pair->shape().phase,
        recipe.registry->SelectionDigest(), second_descriptor_left.seal,
        second_descriptor_right.seal);
    ASSERT_TRUE(second_key);
    EXPECT_EQ(*second_key, *ready_key)
        << "normal row-table remapping must preserve the descriptor ready key";

    ASSERT_TRUE(life_->test_program->Bind(*comparison_launch));
    ASSERT_TRUE(life_->test_peer_program->Bind(*comparison_launch));
    auto bound_left = certificate(*life_->test_program, kn::ComparisonCertificatePhase::kBound);
    auto bound_right =
        certificate(*life_->test_peer_program, kn::ComparisonCertificatePhase::kBound);
    ASSERT_EQ(bound_left.status, kn::ComparisonRegionStatus::kClosed) << bound_left.reason;
    ASSERT_EQ(bound_right.status, kn::ComparisonRegionStatus::kClosed) << bound_right.reason;
    ASSERT_EQ(bound_left.seal.state, kn::ComparisonProofState::kBoundVerified);
    ASSERT_EQ(bound_right.seal.state, kn::ComparisonProofState::kBoundVerified);
    const auto same_persistent_fields = [](const auto& bound, const auto& descriptor) {
      EXPECT_EQ(bound.version, descriptor.version);
      EXPECT_EQ(bound.missing, descriptor.missing);
      EXPECT_EQ(bound.subject, descriptor.subject);
      EXPECT_EQ(bound.boundary, descriptor.boundary);
      EXPECT_EQ(bound.planned_region, descriptor.planned_region);
      EXPECT_EQ(bound.physical_recipe, descriptor.physical_recipe);
      EXPECT_EQ(bound.policy, descriptor.policy);
      EXPECT_EQ(bound.registry, descriptor.registry);
      EXPECT_EQ(bound.backend_recipe, descriptor.backend_recipe);
      EXPECT_EQ(bound.backend_build, descriptor.backend_build);
      EXPECT_EQ(bound.backend_selection, descriptor.backend_selection);
      EXPECT_EQ(bound.backend_device, descriptor.backend_device);
    };
    same_persistent_fields(bound_left.seal, descriptor_left.seal);
    same_persistent_fields(bound_right.seal, descriptor_right.seal);
    auto bound_remainder_left = life_->test_program->CompareComparisonRemainder(
        *life_->test_peer_program, *region_boundary, kn::ComparisonCertificatePhase::kBound);
    auto bound_remainder_right = life_->test_peer_program->CompareComparisonRemainder(
        *life_->test_program, *region_boundary, kn::ComparisonCertificatePhase::kBound);
    ASSERT_EQ(bound_remainder_left.status, kn::ComparisonRemainderStatus::kSame)
        << bound_remainder_left.reason;
    ASSERT_EQ(bound_remainder_right.status, kn::ComparisonRemainderStatus::kSame)
        << bound_remainder_right.reason;

    auto native = life_->node.execution().Submission(life_->node.stream(0));
    ASSERT_TRUE(native) << (native ? "" : native.error().detail);
    auto reference = reference::Executor::Create(pair->source().graph);
    ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
    ASSERT_TRUE(reference->BindValues(table, logical_table));
    ASSERT_TRUE(reference->BindValues(scale, scale_values));
    for (std::size_t frame_index = 0; frame_index < 2; ++frame_index) {
      const auto logical_ids = frame_index == 0 ? first_ids : second_ids;
      ASSERT_NO_FATAL_FAILURE(run_parent_rows(logical_ids, dense_frames[frame_index]));
      const auto& dense_ids = dense_frames[frame_index];
      std::array<float, kRequestRows> reference_ids{};
      for (std::size_t i = 0; i < logical_ids.size(); ++i)
        reference_ids[i] = static_cast<float>(logical_ids[i]);
      ASSERT_TRUE(reference->BindValues(ids, reference_ids));
      ASSERT_TRUE(reference->Run());
      auto expected = reference->Values(output);
      ASSERT_TRUE(expected) << (expected ? "" : expected.error().detail);
      ASSERT_EQ(expected->size(), kRequestRows * kWidth);
      const auto observed_table_owner = observation->tables[table_index];
      ASSERT_TRUE(observed_table_owner.range);
      std::vector<std::byte> table_snapshot(
          static_cast<std::size_t>(observed_table_owner.range->bytes));
      ASSERT_EQ(cudaMemcpy(table_snapshot.data(),
                           reinterpret_cast<const void*>(observed_table_owner.address),
                           table_snapshot.size(), cudaMemcpyDeviceToHost),
                cudaSuccess);
      std::vector<float> scale_snapshot(kWidth);
      ASSERT_EQ(cudaMemcpy(scale_snapshot.data(), reinterpret_cast<const void*>(scale_address),
                           scale_snapshot.size() * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (auto* program : {life_->test_program.get(), life_->test_peer_program.get()}) {
        const auto inputs = program->inputs();
        const auto side_slot = std::ranges::find_if(inputs, [&](const auto& slot) {
          return slot.tensor == ids && slot.kind == kn::InputKind::kSideRows &&
                 slot.component == 17;
        });
        ASSERT_NE(side_slot, inputs.end());
        ASSERT_EQ(side_slot->bytes, sizeof(dense_ids));
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(activation_address), 0xa5, kArenaBytes,
                                  static_cast<cudaStream_t>(native->handle)),
                  cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(reinterpret_cast<void*>(side_slot->device_address),
                            reinterpret_cast<const void*>(observation->side_rows->device_address),
                            sizeof(dense_ids), cudaMemcpyDeviceToDevice,
                            static_cast<cudaStream_t>(native->handle)),
            cudaSuccess);
        ASSERT_TRUE(program->Run(*comparison_launch, *native));
        ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
        const auto outputs = program->outputs();
        const auto output_slot = std::ranges::find_if(
            outputs, [&](const auto& slot) { return slot.tensor == output && !slot.features; });
        ASSERT_NE(output_slot, outputs.end());
        std::vector<std::byte> actual_bytes(static_cast<std::size_t>(output_slot->bytes));
        ASSERT_EQ(cudaMemcpy(actual_bytes.data(),
                             reinterpret_cast<const void*>(output_slot->device_address),
                             actual_bytes.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(actual_bytes.size(), expected->size() * sizeof(float));
        for (std::size_t i = 0; i < expected->size(); ++i) {
          float actual = 0.0F;
          std::memcpy(&actual, actual_bytes.data() + i * sizeof(float), sizeof(actual));
          ASSERT_TRUE(std::isfinite(actual));
          ASSERT_TRUE(std::isfinite((*expected)[i]));
          EXPECT_LE(std::abs(actual - (*expected)[i]), 1.0e-5F * (1.0F + std::abs((*expected)[i])));
        }
        std::vector<std::byte> table_after(table_snapshot.size());
        ASSERT_EQ(cudaMemcpy(table_after.data(),
                             reinterpret_cast<const void*>(observed_table_owner.address),
                             table_after.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(table_after, table_snapshot) << "GGML arms must not modify any row slot";
        std::vector<float> scale_after(kWidth);
        ASSERT_EQ(cudaMemcpy(scale_after.data(), reinterpret_cast<const void*>(scale_address),
                             scale_after.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(scale_after, scale_snapshot) << "RMS scale is immutable for both arms";
      }
    }
    ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
    life_->test_peer_program.reset();
    life_->test_program.reset();
    life_->test_pair.reset();
    life_->test_activation.Free();
    life_->test_weights.Free();
  };

  ASSERT_NO_FATAL_FAILURE(
      run_case(0, llmp::format::Encoding::kF32, f32_table, first_rows, second_rows));
  std::array<float, kWidth * kLogicalRows> f16_as_float{};
  for (std::size_t i = 0; i < f16_as_float.size(); ++i)
    f16_as_float[i] = __half2float(f16_table[i]);
  ASSERT_NO_FATAL_FAILURE(
      run_case(1, llmp::format::Encoding::kF16, f16_as_float, first_rows, second_rows));

  const std::array<en::PagedModel*, 2> models{life_.get(), runner.get()};
  auto retired = life_->node.TearDown(models);
  ASSERT_TRUE(retired) << (retired ? "" : retired.error());
  runner.reset();
  runner_active_ = false;
}

TEST_F(GgmlComparisonLowerForGpu, Q8GetRowsExtKeepsTheOrdinaryCumulativeOwner) {
  constexpr std::uint64_t kWidth = 256;
  constexpr std::uint64_t kLogicalRows = 8;
  constexpr std::uint64_t kStepRows = 4;
  constexpr std::uint64_t kArenaBytes = 1U << 20U;
  const auto device_backend = life_->backend->Backend();
  ASSERT_NE(device_backend.owner, nullptr);
  const auto& owned_backend = device_backend.owner->backend();
  ASSERT_NE(owned_backend.registry, nullptr);

  auto declarations = std::vector<llmp::execution::Implementation>{};
  declarations.reserve(owned_backend.registry->size());
  for (std::size_t i = 0; i < owned_backend.registry->size(); ++i)
    declarations.push_back(owned_backend.registry->at(i));
  auto ranks = owned_backend.registry->ranks();
  const auto measured = std::ranges::find_if(ranks.recipe_comparisons, [](const auto& row) {
    return row.key.discovery_operation == ir::Fused(ir::FusedKind::kRmsNormMul);
  });
  ASSERT_NE(measured, ranks.recipe_comparisons.end())
      << "the native registry should contain real RMS/Mul comparison rows";
  auto nonmatching = *measured;
  nonmatching.key.subject = llmp::base::Sha256{}.Update("test-only Q8 fallback row").Finish();
  ranks.recipe_comparisons = {std::move(nonmatching)};
  auto test_registry = llmp::execution::Registry::Create(std::move(declarations), std::move(ranks));
  ASSERT_TRUE(test_registry) << (test_registry ? "" : test_registry.error().detail);
  ASSERT_EQ(test_registry->ranks().recipe_comparisons.size(), 1U);

  const ir::WeightRef table_ref{.stage = 0, .component = 17, .role = 30, .layer = 0};
  const ir::WeightRef scale_ref{.stage = 0, .component = 17, .role = 31, .layer = 0};
  const auto table_format =
      llmp::format::TensorFormat{.encoding = llmp::format::Encoding::kQ8_0,
                                 .layout = llmp::format::Layout::kGgmlBlocks,
                                 .shape = ir::ShapeOf({kWidth, kLogicalRows})};
  const auto row_bytes = llmp::format::RowBytes(table_format, llmp::format::PartRole::kData);
  ASSERT_TRUE(row_bytes);
  const auto table_bytes = *row_bytes * kLogicalRows;
  ASSERT_TRUE(life_->test_activation.Allocate(kArenaBytes));
  ASSERT_TRUE(life_->test_weights.Allocate(table_bytes));
  ASSERT_TRUE(life_->test_weights_rebound.Allocate(kWidth * sizeof(float)));
  std::vector<std::byte> q8_rows(static_cast<std::size_t>(table_bytes), std::byte{0});
  ASSERT_EQ(cudaMemcpy(life_->test_weights.pointer, q8_rows.data(), q8_rows.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  std::vector<float> scale_values(kWidth, 1.0F);
  ASSERT_EQ(cudaMemcpy(life_->test_weights_rebound.pointer, scale_values.data(),
                       scale_values.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

  ir::Builder builder;
  builder.SetComponent(17);
  builder.SetOwner(0);
  const auto ids = builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({kStepRows})),
                                 "q8-side-rows");
  const auto table = builder.Weight(table_format, table_ref, "q8-row-table");
  const auto scale = builder.Weight(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth})),
                                    scale_ref, "rms-scale");
  const auto gathered = builder.GetRows(table, ids);
  const auto normalized = builder.RmsNorm(gathered, 1.0e-6F);
  const auto output = builder.Mul(normalized, scale);
  builder.Output(output);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = ids,
                                 .kind = co::InputKind::kSideRows,
                                 .segment = ir::kJoinedSegment,
                                 .component = 17}},
                     .weights = {{.tensor = table,
                                  .ref = table_ref,
                                  .parts = {{"q8-row-table", llmp::format::PartRole::kData}}},
                                 {.tensor = scale,
                                  .ref = scale_ref,
                                  .parts = {{"rms-scale", llmp::format::PartRole::kData}}}},
                     .state = {},
                     .logits = output};
  const planner::StepShape shape{.stage = 0,
                                 .phase = planner::Phase::kDecode,
                                 .segments = {{.slot = 0,
                                               .rows = static_cast<std::uint32_t>(kStepRows),
                                               .past = 0,
                                               .read_cells = 256,
                                               .outputs = static_cast<std::uint32_t>(kStepRows)}}};
  const auto io = co::GraphIO(step);

  const auto table_address = life_->test_weights.address();
  const auto scale_address = life_->test_weights_rebound.address();
  const auto content_identity = llmp::base::Sha256{}.Update(q8_rows).Finish();
  const co::ComparisonRowStorage storage{.ref = table_ref,
                                         .part = llmp::format::PartRole::kData,
                                         .logical_format = table_format,
                                         .content_identity = content_identity,
                                         .resource = 1,
                                         .group = 1,
                                         .shard = 0,
                                         .group_bytes = 4096,
                                         .resource_offset = 0,
                                         .resource_bytes = table_bytes,
                                         .logical_rows = kLogicalRows,
                                         .slot_capacity = kLogicalRows,
                                         .row_bytes = *row_bytes};
  kn::Addresses addresses{};
  addresses.weight = [=](const ir::WeightRef& ref, llmp::format::PartRole part) {
    if (part != llmp::format::PartRole::kData) return std::uint64_t{0};
    if (ref == table_ref) return table_address;
    if (ref == scale_ref) return scale_address;
    return std::uint64_t{0};
  };
  addresses.weight_rows = [=](const ir::WeightRef& ref) {
    return ref == table_ref ? kLogicalRows : std::uint64_t{0};
  };
  addresses.weight_range = [=](const ir::WeightRef& ref,
                               llmp::format::PartRole part) -> std::optional<kn::BoundedAddress> {
    if (part != llmp::format::PartRole::kData) return {};
    if (ref == table_ref)
      return kn::BoundedAddress{.base = table_address, .bytes = table_bytes, .alignment = 16};
    if (ref == scale_ref)
      return kn::BoundedAddress{
          .base = scale_address, .bytes = kWidth * sizeof(float), .alignment = 16};
    return {};
  };
  addresses.row_storage =
      [=](const ir::WeightRef& ref,
          llmp::format::PartRole part) -> std::optional<co::ComparisonRowStorage> {
    if (ref == table_ref && part == llmp::format::PartRole::kData) return storage;
    return {};
  };
  addresses.weight_alignment = [=](const ir::WeightRef& ref, llmp::format::PartRole part) {
    return part == llmp::format::PartRole::kData && (ref == table_ref || ref == scale_ref)
               ? std::uint64_t{16}
               : std::uint64_t{0};
  };
  addresses.activations = life_->test_activation.address();
  addresses.activation_bytes = kArenaBytes;
  addresses.activation_alignment = 256;

  auto backend = owned_backend;
  backend.registry = &*test_registry;
  const auto step_policy = planner::ForStep(planner::Policy{}, shape.rows(), shape.segments.size());
  ASSERT_TRUE(step_policy) << (step_policy ? "" : step_policy.error());
  planner::PlanOptions options{};
  options.capturable = true;
  options.within = backend.within;
  options.written = backend.written;
  options.on_stream = backend.on_stream;
  options.shape = &shape;

  auto source_bindings = en::ComparisonSourceBindings::Create(
      std::make_shared<PaidBoundaryFunder>(life_->paid), step, 0);
  ASSERT_TRUE(source_bindings) << (source_bindings ? "" : source_bindings.error().detail);

  struct ObservedProofs {
    const kn::StepProgram* initial = nullptr;
    std::optional<kn::ComparisonRegionDescriptorResult> first_certificate;
    std::vector<kn::ComparisonRegionStatus> candidate_certificates;
    std::vector<kn::ComparisonProofState> candidate_certificate_states;
    std::vector<std::string> candidate_certificate_reasons;
    std::vector<kn::ComparisonRemainderStatus> remainders;
    std::vector<std::string> remainder_reasons;
    std::vector<std::vector<ir::NodeId>> expansions;
    std::vector<kn::ComparisonRegionStatus> expanded_certificates;
    std::vector<std::string> expanded_certificate_reasons;
    bool expansion_contains_get_rows = false;
    std::vector<std::string> actual_implementations;
    std::vector<std::vector<std::string>> selected_implementations;
    std::size_t lower_count = 0;
  } observed;
  const kn::ProgramValidator observe =
      [&](const kn::LowerInput& input,
          const kn::StepProgram& program) -> std::expected<void, std::string> {
    ++observed.lower_count;
    bool saw_ext = false;
    std::vector<std::string> selected_names;
    selected_names.reserve(input.plan.steps.size());
    for (const auto& selected : input.plan.steps) {
      const auto name = test_registry->at(selected.implementation).name;
      observed.actual_implementations.push_back(name);
      selected_names.push_back(name);
      saw_ext |= name == kg::kGetRowsExtName;
    }
    observed.selected_implementations.push_back(std::move(selected_names));
    if (input.comparison_region == nullptr || input.comparison_region->seed == nullptr)
      return std::expected<void, std::string>{};
    if (!saw_ext)
      return std::unexpected(std::string("the actual Q8 plan did not select GetRowsExt"));
    const auto& boundary = *input.comparison_region->seed;
    auto certificate = program.ProjectComparisonRegionCertificate(
        boundary, kn::ComparisonCertificatePhase::kDescriptor);
    if (observed.initial == nullptr) {
      observed.initial = &program.comparison_program();
      observed.first_certificate = certificate;
    } else {
      observed.candidate_certificates.push_back(certificate.status);
      observed.candidate_certificate_states.push_back(certificate.seal.state);
      observed.candidate_certificate_reasons.emplace_back(certificate.reason);

      auto resolved = source_bindings->Resolve(input.plan, addresses, backend.written);
      if (!resolved)
        return std::unexpected(std::string("source binding resolution failed: ") +
                               std::string(resolved.error().detail));
      const auto source_view = source_bindings->boundary_view();
      if (!source_view)
        return std::unexpected(std::string("source binding boundary view is unavailable"));

      auto remainder = observed.initial->CompareComparisonRemainder(
          program, boundary, kn::ComparisonCertificatePhase::kDescriptor);
      observed.remainders.push_back(remainder.status);
      observed.remainder_reasons.emplace_back(remainder.reason);
      std::vector<ir::NodeId> additions(remainder.required_members.begin(),
                                        remainder.required_members.end());
      if (remainder.status == kn::ComparisonRemainderStatus::kExpand) {
        observed.expansion_contains_get_rows |=
            std::ranges::find(additions, step.graph.producer(gathered)) != additions.end();
      }

      auto reverse = program.CompareComparisonRemainder(
          *observed.initial, boundary, kn::ComparisonCertificatePhase::kDescriptor);
      if (reverse.status == kn::ComparisonRemainderStatus::kExpand) {
        additions.insert(additions.end(), reverse.required_members.begin(),
                         reverse.required_members.end());
        observed.expansion_contains_get_rows |=
            std::ranges::find(reverse.required_members, step.graph.producer(gathered)) !=
            reverse.required_members.end();
      }
      if (remainder.status == kn::ComparisonRemainderStatus::kExpand ||
          reverse.status == kn::ComparisonRemainderStatus::kExpand) {
        std::ranges::sort(additions);
        additions.erase(std::unique(additions.begin(), additions.end()), additions.end());

        std::vector<std::uint32_t> offsets{0};
        std::vector<ir::NodeId> members;
        std::uint32_t prior_group = 0;
        bool have_group = false;
        for (const auto& node : boundary.nodes()) {
          if (have_group && node.group != prior_group) {
            offsets.push_back(static_cast<std::uint32_t>(members.size()));
          }
          members.push_back(node.original_node);
          prior_group = node.group;
          have_group = true;
        }
        offsets.push_back(static_cast<std::uint32_t>(members.size()));
        for (const auto added : additions) {
          if (!added.valid() || added.v >= step.graph.nodes().size())
            return std::unexpected(
                std::string("remainder expansion contains an invalid source node"));
          if (std::ranges::find(members, added) != members.end()) continue;
          members.push_back(added);
          offsets.push_back(static_cast<std::uint32_t>(members.size()));
        }
        auto expanded = co::MakeComparisonRegionBoundary(
            {*source_view, offsets, members}, std::make_shared<PaidBoundaryFunder>(life_->paid));
        if (!expanded)
          return std::unexpected(std::string("expanded source boundary failed: ") +
                                 std::string(expanded.error().detail));
        const auto expanded_left = observed.initial->ProjectComparisonRegionCertificate(
            *expanded, kn::ComparisonCertificatePhase::kDescriptor);
        observed.expanded_certificates.push_back(expanded_left.status);
        observed.expanded_certificate_reasons.emplace_back(expanded_left.reason);
        const auto expanded_right = program.ProjectComparisonRegionCertificate(
            *expanded, kn::ComparisonCertificatePhase::kDescriptor);
        observed.expanded_certificates.push_back(expanded_right.status);
        observed.expanded_certificate_reasons.emplace_back(expanded_right.reason);
        observed.expansions.push_back(std::move(additions));
      }
    }
    return std::expected<void, std::string>{};
  };
  auto cumulative =
      en::BuildCumulativeRecipeProgram(io, step, shape, 0, 256, backend, *step_policy, addresses,
                                       std::move(options), observe, device_backend.owner, nullptr);
  ASSERT_TRUE(cumulative) << (cumulative ? "" : cumulative.error());
  ASSERT_GT(observed.lower_count, 1U)
      << "the real recipe registry must reach an alternative trial after one incumbent Lower";
  ASSERT_NE(observed.initial, nullptr);
  ASSERT_TRUE(observed.first_certificate);
  ASSERT_EQ(observed.first_certificate->status, kn::ComparisonRegionStatus::kClosed);
  ASSERT_EQ(observed.first_certificate->seal.state, kn::ComparisonProofState::kDescriptorVerified);
  ASSERT_NE(std::ranges::find(observed.actual_implementations, kg::kGetRowsExtName),
            observed.actual_implementations.end());
  // Native whole-program descriptors now cover this ordinary reader; local
  // region access remains unsupported and must not confer a local adoption.
  ASSERT_EQ((*cumulative)->comparison_recipe()->state,
            kn::ComparisonProofState::kDescriptorVerified);
  ASSERT_EQ((*cumulative)->comparison_recipe()->missing, kn::ComparisonMissing::kNone);
  EXPECT_EQ(&(*cumulative)->comparison_program(), observed.initial)
      << "unsupported local GetRowsExt remainder proof must keep the exact first ordinary owner";
  ASSERT_FALSE(observed.candidate_certificates.empty());
  EXPECT_TRUE(std::ranges::all_of(observed.candidate_certificates, [](auto status) {
    return status == kn::ComparisonRegionStatus::kClosed;
  })) << ::testing::PrintToString(observed.candidate_certificates);
  EXPECT_TRUE(std::ranges::all_of(observed.candidate_certificate_states, [](auto state) {
    return state == kn::ComparisonProofState::kDescriptorVerified;
  })) << ::testing::PrintToString(observed.candidate_certificate_states);
  ASSERT_EQ(observed.candidate_certificates.size(), observed.remainders.size());
  EXPECT_TRUE(std::ranges::any_of(
      observed.remainders,
      [](auto status) { return status == kn::ComparisonRemainderStatus::kExpand; }))
      << "the actual outside GetRowsExt calls must request complete source expansion; plans="
      << ::testing::PrintToString(observed.selected_implementations)
      << "; candidate certificates=" << ::testing::PrintToString(observed.candidate_certificates)
      << "; certificate reasons="
      << ::testing::PrintToString(observed.candidate_certificate_reasons)
      << "; remainders=" << ::testing::PrintToString(observed.remainders)
      << "; remainder reasons=" << ::testing::PrintToString(observed.remainder_reasons);
  EXPECT_TRUE(observed.expansion_contains_get_rows)
      << "the typed expansion must include the actual GetRows source producer";
  ASSERT_FALSE(observed.expanded_certificates.empty());
  EXPECT_TRUE(std::ranges::all_of(
      observed.expanded_certificates,
      [](auto status) { return status == kn::ComparisonRegionStatus::kUnproved; }))
      << "the authenticated union must reach the unsupported GetRowsExt access schema; statuses="
      << ::testing::PrintToString(observed.expanded_certificates)
      << "; reasons=" << ::testing::PrintToString(observed.expanded_certificate_reasons);
  EXPECT_TRUE(std::ranges::all_of(observed.expanded_certificate_reasons, [](const auto& reason) {
    return reason == "actual region call lacks a supported typed access recipe";
  })) << ::testing::PrintToString(observed.expanded_certificate_reasons);

  const auto pool_bytes = std::max<std::uint64_t>((*cumulative)->pool_bytes(), 1U << 20U);
  ASSERT_TRUE(life_->test_pool.Allocate(pool_bytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  if (persistent_bytes != 0) ASSERT_TRUE(life_->test_persistent.Allocate(persistent_bytes));
  auto launch = life_->backend->Launch(
      {.base = life_->test_pool.address(), .size = llmp::base::Bytes(pool_bytes)},
      {.base = life_->test_persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  life_->test_launch = std::move(*launch);
  life_->test_program = std::move(*cumulative);
  ASSERT_TRUE(life_->test_program->Bind(*life_->test_launch));
  ASSERT_EQ(life_->test_program->comparison_recipe()->state,
            kn::ComparisonProofState::kBoundVerified);
  ASSERT_EQ(life_->test_program->comparison_recipe()->missing, kn::ComparisonMissing::kNone);
  ASSERT_EQ(cudaMemset(life_->test_activation.pointer, 0xa5, kArenaBytes), cudaSuccess);
  const std::array<std::int32_t, kStepRows> side_rows{0, 1, 2, 3};
  bool staged = false;
  for (const auto& slot : life_->test_program->inputs()) {
    if (slot.tensor != ids) continue;
    ASSERT_EQ(slot.kind, kn::InputKind::kSideRows);
    ASSERT_EQ(slot.component, 17U);
    ASSERT_EQ(slot.bytes, sizeof(side_rows));
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(slot.device_address), side_rows.data(),
                         sizeof(side_rows), cudaMemcpyHostToDevice),
              cudaSuccess);
    staged = true;
  }
  ASSERT_TRUE(staged);
  auto native = life_->execution->Submission(life_->node.stream(0));
  ASSERT_TRUE(native) << native.error().detail;
  ASSERT_TRUE(life_->test_program->Run(*life_->test_launch, *native));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_EQ(life_->test_program->outputs().size(), 1U);
  const auto& output_slot = life_->test_program->outputs().front();
  ASSERT_EQ(output_slot.bytes, kStepRows * kWidth * sizeof(float));
  std::vector<float> output_values(static_cast<std::size_t>(kStepRows * kWidth));
  ASSERT_EQ(
      cudaMemcpy(output_values.data(), reinterpret_cast<const void*>(output_slot.device_address),
                 output_slot.bytes, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (const auto value : output_values) EXPECT_FLOAT_EQ(value, 0.0F);
}

TEST_F(GgmlComparisonLowerForGpu, SmallTokenBuildStepCarriesFoldedCellPositionsAndWrittenState) {
  auto source = MakeSmallTokenStateSource();
  ASSERT_TRUE(source) << (source ? "" : source.error());
  ASSERT_EQ(source->step.state.size(), 2U);
  EXPECT_TRUE(std::ranges::any_of(source->step.graph.nodes(), [](const ir::Node& node) {
    return node.key.op == ir::Op::kCacheStore;
  }));
  const auto cell_inputs =
      std::ranges::count(source->step.inputs, co::InputKind::kCellPositions, &co::StepInput::kind);
  ASSERT_EQ(cell_inputs, 1U);
  const auto& input =
      *std::ranges::find(source->step.inputs, co::InputKind::kCellPositions, &co::StepInput::kind);
  std::size_t mask_readers = 0;
  for (const auto& node : source->step.graph.nodes()) {
    for (std::size_t i = 0; i < node.nsrc; ++i) {
      if (node.source(i) != input.tensor) continue;
      EXPECT_EQ(node.key, ir::Primitive(ir::Op::kMaskBuild));
      EXPECT_EQ(i, 1U);
      ++mask_readers;
    }
  }
  EXPECT_GT(mask_readers, 0U);

  const co::GraphBuildIO io = co::GraphIO(source->step);
  std::vector<co::ComparisonExternalBinding> external;
  std::vector<co::ComparisonUnstoredInput> unstored;
  std::uint32_t allocation_class = 1;
  for (const auto input_id : io.inputs) {
    const auto input_slot =
        std::ranges::find(source->step.inputs, input_id, &co::StepInput::tensor);
    ASSERT_NE(input_slot, source->step.inputs.end());
    if (input_slot->kind == co::InputKind::kCellPositions) {
      unstored.push_back({.original_input = input_id});
      continue;
    }
    const auto bytes = ir::StorageBytes(source->step.graph.tensor(input_id));
    ASSERT_TRUE(bytes && *bytes != 0);
    external.push_back({.original_root = input_id,
                        .allocation_class = allocation_class++,
                        .allocation_bytes = *bytes,
                        .bytes = *bytes,
                        .alignment = 16});
  }
  for (const auto& weight : source->step.weights) {
    const auto parts = llmp::format::PartsOf(source->step.graph.tensor(weight.tensor).format);
    ASSERT_TRUE(parts);
    ASSERT_EQ(weight.parts.size(), parts->size());
    for (const auto& part : parts->parts()) {
      external.push_back({.original_root = weight.tensor,
                          .part = part.role,
                          .allocation_class = allocation_class++,
                          .allocation_bytes = part.readable,
                          .bytes = part.readable,
                          .alignment = 16});
    }
  }
  for (const auto& state : source->step.state) {
    const auto bytes = ir::StorageBytes(source->step.graph.tensor(state.tensor));
    ASSERT_TRUE(bytes && *bytes != 0);
    external.push_back({.original_root = state.tensor,
                        .allocation_class = allocation_class++,
                        .allocation_bytes = *bytes,
                        .bytes = *bytes,
                        .alignment = 16});
  }
  const co::ComparisonBoundaryView view{io, source->step.state, 0, external, unstored};
  auto boundary =
      co::MakeComparisonBoundary(view, std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  ASSERT_EQ(boundary->unstored_inputs().size(), 1U);
  EXPECT_EQ(boundary->unstored_inputs()[0].original_input, input.tensor);
  std::vector<ir::TensorId> identity(source->step.graph.tensor_count());
  for (std::uint32_t i = 0; i < identity.size(); ++i) identity[i] = ir::TensorId{i};
  auto mapped = co::ValidateMappedComparisonBoundary(*boundary, source->step.graph, identity);
  ASSERT_TRUE(mapped) << (mapped ? "" : mapped.error().detail);
}

TEST_F(GgmlComparisonLowerForGpu, TokenBuildStepGetsReadyAndBoundCertificateThenRunsNormally) {
  constexpr std::size_t kArenaBytes = 4U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  constexpr float kF16StateTolerance = 1.0F / 1024.0F;  // components_test.cc reference bound
  constexpr float kOutputTolerance = 1.0e-4F;           // components_test.cc reference bound
  {
    auto source = MakeSmallTokenStateSource();
    ASSERT_TRUE(source) << (source ? "" : source.error());
    auto& step = source->step;
    const co::GraphBuildIO io = co::GraphIO(step);
    ASSERT_TRUE(std::ranges::any_of(
        step.state, [](const co::StepState& state_fact) { return state_fact.written; }));
    const auto cell_input =
        std::ranges::find(step.inputs, co::InputKind::kCellPositions, &co::StepInput::kind);
    ASSERT_NE(cell_input, step.inputs.end());
    const std::array<co::ComparisonUnstoredInput, 1> unstored{
        co::ComparisonUnstoredInput{.original_input = cell_input->tensor}};

    CudaBuffer& activation = life_->test_activation;
    CudaBuffer& weights = life_->test_weights;
    CudaBuffer& state = life_->test_state;
    CudaBuffer& pool = life_->test_pool;
    CudaBuffer& persistent = life_->test_persistent;
    ASSERT_TRUE(activation.Allocate(kArenaBytes));
    CudaBuffer relocated_activation;
    ASSERT_TRUE(relocated_activation.Allocate(kArenaBytes));
    ASSERT_TRUE(pool.Allocate(kPoolBytes));
    ASSERT_TRUE(state.Allocate(kArenaBytes));
    ASSERT_EQ(cudaMemset(state.pointer, 0, source->layout.bytes), cudaSuccess);
    const auto backend_view = life_->backend->Backend();
    const auto& recipe = backend_view.owner ? backend_view.owner->backend() : backend_view;
    const planner::Policy policy{};
    planner::PlanOptions plan_options{};
    plan_options.capturable = true;
    plan_options.shape = &source->shape;
    plan_options.within = recipe.within;
    plan_options.written = recipe.written;
    plan_options.on_stream = recipe.on_stream;
    std::vector<ir::TensorId> staged_inputs;
    std::vector<ir::TensorId> aligned_staged_inputs;
    for (const auto& input : step.inputs) {
      if (input.kind != co::InputKind::kCellPositions) staged_inputs.push_back(input.tensor);
      if (input.kind != co::InputKind::kCellPositions && input.kind != co::InputKind::kCommitRows)
        aligned_staged_inputs.push_back(input.tensor);
    }
    plan_options.staged = staged_inputs;
    plan_options.alignment_staged = aligned_staged_inputs;
    plan_options.activation_alignment = 256;
    plan_options.root_alignment = [](const ir::Graph& graph, ir::TensorId root) {
      if (!root.valid() || root.v >= graph.tensor_count()) return std::uint64_t{0};
      const auto& desc = graph.tensor(root);
      return std::holds_alternative<ir::WeightRef>(desc.binding) ||
                     std::holds_alternative<ir::StateRef>(desc.binding)
                 ? std::uint64_t{256}
                 : std::uint64_t{0};
    };

    std::map<ir::WeightRef, std::uint64_t> weight_offsets;
    std::uint64_t weight_bytes = 0;
    for (const auto& weight : step.weights) {
      const auto bytes = ir::StorageBytes(step.graph.tensor(weight.tensor));
      ASSERT_TRUE(bytes && *bytes > 0);
      weight_bytes = (weight_bytes + 255U) & ~std::uint64_t{255U};
      ASSERT_TRUE(weight_offsets.emplace(weight.ref, weight_bytes).second);
      ASSERT_LE(*bytes, kArenaBytes - weight_bytes);
      weight_bytes += *bytes;
    }
    ASSERT_GT(weight_bytes, 0U);
    ASSERT_LE(weight_bytes, kArenaBytes);
    ASSERT_TRUE(weights.Allocate(kArenaBytes));
    std::map<ir::TensorId, std::vector<float>> weight_values;
    for (const auto& weight : step.weights) {
      const auto& desc = step.graph.tensor(weight.tensor);
      const auto offset = weight_offsets.at(weight.ref);
      std::vector<float> values(ir::Elements(desc.shape()));
      const bool norm = desc.shape().rank == 1;
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = norm ? 1.0F : static_cast<float>(static_cast<int>(i % 13) - 6) / 64.0F;
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(weights.address() + offset), values.data(),
                           values.size() * sizeof(float), cudaMemcpyHostToDevice),
                cudaSuccess);
      weight_values.emplace(weight.tensor, std::move(values));
    }
    std::map<ir::StateRef, std::uint64_t> state_offsets;
    for (const auto& state_fact : step.state) {
      ASSERT_LT(state_fact.item, source->layout.items.size());
      ASSERT_TRUE(
          state_offsets.emplace(state_fact.ref, source->layout.items[state_fact.item].offset)
              .second);
    }
    kn::Addresses addresses{};
    addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
      const auto found = weight_offsets.find(ref);
      return found == weight_offsets.end() ? std::uint64_t{0} : weights.address() + found->second;
    };
    addresses.state = [&](const ir::StateRef& ref) {
      const auto found = state_offsets.find(ref);
      return found == state_offsets.end() ? std::uint64_t{0} : state.address() + found->second;
    };
    addresses.activations = activation.address();
    addresses.activation_bytes = kArenaBytes;
    addresses.activation_alignment = 256;

    // The ordinary planner establishes actual placed input ranges before the
    // owned source boundary is sealed. BoundComparison repeats this same plan.
    auto step_policy_result = planner::ForStep(
        policy, source->shape.rows(), static_cast<std::uint32_t>(source->shape.segments.size()));
    ASSERT_TRUE(step_policy_result) << (step_policy_result ? "" : step_policy_result.error());
    const planner::Policy step_policy = std::move(*step_policy_result);
    auto preliminary = planner::Plan(step.graph.Clone(), *recipe.registry, recipe.device,
                                     step_policy, plan_options);
    ASSERT_TRUE(preliminary) << (preliminary ? "" : preliminary.error().detail);
    std::vector<co::ComparisonExternalBinding> external;
    std::vector<kn::ComparisonBindingRange> bindings;
    const auto input_offset = [&](ir::TensorId original) -> std::optional<std::uint64_t> {
      if (!original.valid() || original.v >= preliminary->tensors.size()) return {};
      const auto mapped = preliminary->tensors[original.v];
      if (!mapped.valid() || mapped.v >= preliminary->graph.tensor_count()) return {};
      const auto root = preliminary->graph.root(mapped);
      const auto found = std::ranges::find(preliminary->placement.offsets, root,
                                           &std::pair<ir::TensorId, std::uint64_t>::first);
      if (found == preliminary->placement.offsets.end()) return {};
      return found->second;
    };
    for (const auto input_id : io.inputs) {
      const auto input_fact = std::ranges::find(step.inputs, input_id, &co::StepInput::tensor);
      ASSERT_NE(input_fact, step.inputs.end());
      if (input_fact->kind == co::InputKind::kCellPositions) continue;
      const auto bytes = ir::StorageBytes(step.graph.tensor(input_id));
      const auto offset = input_offset(input_id);
      ASSERT_TRUE(bytes && *bytes > 0 && offset);
      ASSERT_LE(*offset, kArenaBytes);
      ASSERT_LE(*bytes, kArenaBytes - *offset);
      external.push_back({.original_root = input_id,
                          .allocation_class = 1,
                          .allocation_bytes = kArenaBytes,
                          .offset = *offset,
                          .bytes = *bytes,
                          .alignment = 16});
      bindings.push_back({.original_root = input_id,
                          .allocation_class = 1,
                          .allocation_bytes = kArenaBytes,
                          .offset = *offset,
                          .bytes = *bytes,
                          .alignment = 16,
                          .base = activation.address()});
    }
    for (const auto& weight : step.weights) {
      const auto parts = llmp::format::PartsOf(step.graph.tensor(weight.tensor).format);
      ASSERT_TRUE(parts);
      ASSERT_EQ(weight.parts.size(), parts->size());
      const auto offset = weight_offsets.at(weight.ref);
      for (const auto& part : parts->parts()) {
        external.push_back({.original_root = weight.tensor,
                            .part = part.role,
                            .allocation_class = 2,
                            .allocation_bytes = kArenaBytes,
                            .offset = offset,
                            .bytes = part.readable,
                            .alignment = 16});
        bindings.push_back({.original_root = weight.tensor,
                            .part = part.role,
                            .allocation_class = 2,
                            .allocation_bytes = kArenaBytes,
                            .offset = offset,
                            .bytes = part.readable,
                            .alignment = 16,
                            .base = weights.address()});
      }
    }
    for (const auto& state_fact : step.state) {
      const auto& item = source->layout.items[state_fact.item];
      external.push_back({.original_root = state_fact.tensor,
                          .allocation_class = 3,
                          .allocation_bytes = source->layout.bytes,
                          .offset = item.offset,
                          .bytes = item.bytes,
                          .alignment = 256});
      bindings.push_back({.original_root = state_fact.tensor,
                          .allocation_class = 3,
                          .allocation_bytes = source->layout.bytes,
                          .offset = item.offset,
                          .bytes = item.bytes,
                          .alignment = 256,
                          .base = state.address()});
    }
    const co::ComparisonBoundaryView boundary_view{io, step.state, 0, external, unstored};
    auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
    auto boundary = co::MakeComparisonBoundary(boundary_view, funder);
    ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
    const auto pattern = planner::FindPattern("rms_norm_mul");
    ASSERT_NE(pattern, nullptr);
    ir::NodeId norm{}, mul{}, outside_norm{}, outside_mul{};
    for (std::uint32_t i = 0; i < step.graph.nodes().size(); ++i) {
      const auto& node = step.graph.node(ir::NodeId{i});
      if (node.key.op != ir::Op::kMul) continue;
      for (std::size_t side = 0; side < node.nsrc; ++side) {
        const auto source_id = node.source(side);
        const auto producer = step.graph.producer(source_id);
        if (!producer.valid() || step.graph.node(producer).key.op != ir::Op::kRmsNorm) continue;
        if (!std::holds_alternative<ir::WeightRef>(
                step.graph.tensor(node.source(1 - side)).binding))
          continue;
        const auto& norm_node = step.graph.node(producer);
        const auto input_producer = step.graph.producer(norm_node.source(0));
        if (input_producer.valid() && step.graph.node(input_producer).key.op == ir::Op::kGetRows) {
          if (!norm.valid()) {
            norm = producer;
            mul = ir::NodeId{i};
          }
        } else if (!outside_norm.valid()) {
          outside_norm = producer;
          outside_mul = ir::NodeId{i};
        }
      }
    }
    ASSERT_TRUE(norm.valid() && mul.valid()) << "the actual token graph needs its norm/weight pair";
    ASSERT_TRUE(outside_norm.valid() && outside_mul.valid())
        << "the actual token graph needs an outside norm/weight pair";
    const planner::ComparisonSiteSelector selector{.source_graph_identity = step.graph.identity(),
                                                   .kind = planner::ComparisonSiteKind::kRewrite,
                                                   .pattern = "rms_norm_mul",
                                                   .version = pattern->version,
                                                   .member_origins = {{norm}, {mul}},
                                                   .member_owner_segments = {},
                                                   .boundary_ports = {}};
    auto inventory = planner::BoundComparison(boundary_view, selector, funder, *recipe.registry,
                                              recipe.device, step_policy, plan_options);
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    ASSERT_EQ(inventory->site.accepted_member_origins, selector.member_origins);
    const std::array members{norm, mul};
    const std::array<std::uint32_t, 2> member_offsets{0, 2};
    auto region_boundary = co::MakeComparisonRegionBoundary(
        {.source = boundary_view, .member_offsets = member_offsets, .members = members}, funder);
    ASSERT_TRUE(region_boundary) << (region_boundary ? "" : region_boundary.error().detail);
    const kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                        .bindings = bindings,
                                        .state = step.state,
                                        .active_stage = 0,
                                        .expected_boundary = &inventory->source_boundary};
    const kn::ComparisonRegionCapture capture{
        .seed = &*region_boundary,
        .origins = &*inventory->incumbent.final_origins,
        .final_nodes = &inventory->incumbent.final_node_origins};
    auto alternative = planner::MakeAlternative(*inventory);
    ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
    auto alternative_plan =
        planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                 recipe.device, step_policy, plan_options);
    ASSERT_TRUE(alternative_plan) << (alternative_plan ? "" : alternative_plan.error().detail);
    ASSERT_TRUE(alternative_plan->final_origins);
    const kn::ComparisonRegionCapture alternative_capture{
        .seed = &*region_boundary,
        .origins = &*alternative_plan->final_origins,
        .final_nodes = &alternative_plan->final_node_origins};
    const planner::ComparisonSiteSelector outside_selector{
        .source_graph_identity = step.graph.identity(),
        .kind = planner::ComparisonSiteKind::kRewrite,
        .pattern = "rms_norm_mul",
        .version = pattern->version,
        .member_origins = {{outside_norm}, {outside_mul}},
        .member_owner_segments = {},
        .boundary_ports = {}};
    auto outside_inventory =
        planner::BoundComparison(boundary_view, outside_selector, funder, *recipe.registry,
                                 recipe.device, step_policy, plan_options);
    ASSERT_TRUE(outside_inventory) << (outside_inventory ? "" : outside_inventory.error().detail);
    ASSERT_EQ(outside_inventory->site.accepted_member_origins, outside_selector.member_origins);
    auto outside_alternative = planner::MakeAlternative(*outside_inventory);
    ASSERT_TRUE(outside_alternative)
        << (outside_alternative ? "" : outside_alternative.error().detail);
    auto outside_plan =
        planner::PlanAlternative(*outside_inventory, std::move(*outside_alternative),
                                 *recipe.registry, recipe.device, step_policy, plan_options);
    ASSERT_TRUE(outside_plan) << (outside_plan ? "" : outside_plan.error().detail);
    ASSERT_TRUE(outside_plan->final_origins);
    bool outside_norm_primitive = false;
    bool outside_mul_primitive = false;
    for (const auto& selected : outside_plan->plan.steps) {
      for (const auto planned_node : selected.nodes) {
        if (!planned_node.valid() || planned_node.v >= outside_plan->plan.from.size()) continue;
        const auto original = outside_plan->plan.from[planned_node.v];
        if (original == outside_norm)
          outside_norm_primitive =
              outside_plan->plan.graph.node(planned_node).key.op == ir::Op::kRmsNorm;
        if (original == outside_mul)
          outside_mul_primitive =
              outside_plan->plan.graph.node(planned_node).key.op == ir::Op::kMul;
      }
    }
    ASSERT_TRUE(outside_norm_primitive && outside_mul_primitive)
        << "the independently selected outside alternative must be the real RMS then MUL plan";
    struct ActualRange {
      ir::TensorId root{};
      std::uint64_t offset = 0;
      std::uint64_t bytes = 0;
    };
    ir::TensorId causal_mask{};
    std::uint32_t mask_node_index = 0;
    for (std::uint32_t i = 0; i < step.graph.nodes().size(); ++i) {
      const auto& node = step.graph.node(ir::NodeId{i});
      if (node.key.op == ir::Op::kMaskBuild && node.nout == 1) {
        causal_mask = node.out[0];
        mask_node_index = i;
        break;
      }
    }
    ASSERT_TRUE(causal_mask.valid()) << "the dense token source must emit its causal mask";
    std::uint32_t last_mask_reader = mask_node_index;
    for (std::uint32_t i = mask_node_index + 1; i < step.graph.nodes().size(); ++i) {
      const auto& node = step.graph.node(ir::NodeId{i});
      if (std::ranges::contains(node.sources(), causal_mask)) last_mask_reader = i;
    }
    auto lower = [&](const planner::StepPlan& plan, const kn::ComparisonRequest& compare,
                     const kn::ComparisonRegionCapture& region, std::uint64_t activation_base,
                     std::uint64_t weight_base = 0, std::uint64_t state_base = 0) {
      auto arm_addresses = addresses;
      arm_addresses.activations = activation_base;
      if (weight_base != 0) {
        arm_addresses.weight = [&, weight_base](const ir::WeightRef& ref, llmp::format::PartRole) {
          const auto found = weight_offsets.find(ref);
          return found == weight_offsets.end() ? std::uint64_t{0} : weight_base + found->second;
        };
      }
      if (state_base != 0) {
        arm_addresses.state = [&, state_base](const ir::StateRef& ref) {
          const auto found = state_offsets.find(ref);
          return found == state_offsets.end() ? std::uint64_t{0} : state_base + found->second;
        };
      }
      std::vector<kn::ComparisonBindingRange> arm_bindings(compare.bindings.begin(),
                                                           compare.bindings.end());
      for (auto& binding : arm_bindings) {
        if (std::ranges::contains(io.inputs, binding.original_root)) binding.base = activation_base;
        if (weight_base != 0 &&
            std::ranges::any_of(step.weights, [&](const co::StepWeight& weight) {
              return weight.tensor == binding.original_root;
            }))
          binding.base = weight_base;
        if (state_base != 0 && std::ranges::any_of(step.state, [&](const co::StepState& fact) {
              return fact.tensor == binding.original_root;
            }))
          binding.base = state_base;
      }
      auto arm_compare = compare;
      arm_compare.bindings = arm_bindings;
      const kn::LowerInput input(plan, step, source->shape, arm_addresses, 16, {}, &arm_compare,
                                 &region);
      return life_->backend->Lower(input);
    };
    auto wrong_kind =
        std::ranges::find(step.inputs, unstored[0].original_input, &co::StepInput::tensor);
    ASSERT_NE(wrong_kind, step.inputs.end());
    wrong_kind->kind = co::InputKind::kTokens;
    auto forged = lower(inventory->incumbent.plan, request, capture, activation.address());
    wrong_kind->kind = co::InputKind::kCellPositions;
    ASSERT_FALSE(forged);
    EXPECT_NE(forged.error().find("not cell positions"), std::string::npos);

    auto lowered_program = lower(inventory->incumbent.plan, request, capture, activation.address());
    ASSERT_TRUE(lowered_program) << (lowered_program ? "" : lowered_program.error());
    std::string physical_report;
    (*lowered_program)->StreamPlanReport([&](std::string_view text) {
      physical_report.append(text);
    });
    const auto actual_range = [&](ir::TensorId original) -> std::optional<ActualRange> {
      const auto& plan = inventory->incumbent.plan;
      if (!original.valid() || original.v >= plan.tensors.size()) return {};
      const auto mapped = plan.tensors[original.v];
      if (!mapped.valid() || mapped.v >= plan.graph.tensor_count()) return {};
      const auto root = plan.graph.root(mapped);
      const auto number = [](std::string_view record,
                             std::string_view key) -> std::optional<std::uint64_t> {
        std::string field;
        field.reserve(key.size() + 3);
        field.push_back('"');
        field.append(key);
        field.append("\":");
        const auto found = record.find(field);
        if (found == std::string_view::npos) return {};
        const auto start = found + field.size();
        std::uint64_t value = 0;
        const auto parsed =
            std::from_chars(record.data() + start, record.data() + record.size(), value);
        return parsed.ec == std::errc{} ? std::optional<std::uint64_t>{value} : std::nullopt;
      };
      for (std::size_t begin = 0; begin < physical_report.size();) {
        const auto end = physical_report.find('\n', begin);
        const auto line =
            std::string_view(physical_report)
                .substr(begin,
                        end == std::string::npos ? physical_report.size() - begin : end - begin);
        if (line.find("\"kind\":\"physical_root\"") != std::string_view::npos) {
          const auto actual_root = number(line, "root");
          const auto offset = number(line, "offset");
          const auto bytes = number(line, "bytes");
          if (actual_root && *actual_root == root.v && offset && bytes && *bytes != 0)
            return ActualRange{.root = root, .offset = *offset, .bytes = *bytes};
        }
        if (end == std::string::npos) break;
        begin = end + 1;
      }
      return {};
    };
    const auto mask_range = actual_range(causal_mask);
    ASSERT_TRUE(mask_range) << "the lowered physical report must include the causal-mask root";
    bool found_dead_reuse = false;
    ir::NodeId reused_by{};
    ActualRange reused_range{};
    for (std::uint32_t i = last_mask_reader + 1; i < step.graph.nodes().size(); ++i) {
      for (const auto output : step.graph.node(ir::NodeId{i}).outputs()) {
        const auto range = actual_range(output);
        if (!range || range->root == mask_range->root) continue;
        const auto mask_end = mask_range->offset + mask_range->bytes;
        const auto output_end = range->offset + range->bytes;
        if (mask_range->offset < output_end && range->offset < mask_end) {
          found_dead_reuse = true;
          reused_by = ir::NodeId{i};
          reused_range = *range;
          break;
        }
      }
      if (found_dead_reuse) break;
    }
    ASSERT_TRUE(found_dead_reuse)
        << "the actual lowered physical report should reuse the causal-mask allocation only after "
           "its final read";
    RecordProperty("actual_dead_reuse_after_mask_last_read",
                   std::to_string(mask_node_index) + ":" + std::to_string(last_mask_reader) + "->" +
                       std::to_string(reused_by.v) +
                       " offsets=" + std::to_string(mask_range->offset) + ":" +
                       std::to_string(reused_range.offset));
    auto alternative_program =
        lower(alternative_plan->plan, request, alternative_capture, relocated_activation.address());
    ASSERT_TRUE(alternative_program) << (alternative_program ? "" : alternative_program.error());
    const kn::ComparisonRegionCapture outside_capture{
        .seed = &*region_boundary,
        .origins = &*outside_plan->final_origins,
        .final_nodes = &outside_plan->final_node_origins};
    auto outside_program =
        lower(outside_plan->plan, request, outside_capture, activation.address());
    ASSERT_TRUE(outside_program) << (outside_program ? "" : outside_program.error());
    const auto token_input =
        std::ranges::find(step.inputs, co::InputKind::kTokens, &co::StepInput::kind);
    ASSERT_NE(token_input, step.inputs.end());
    const auto incumbent_input_slot = std::ranges::find(
        (*lowered_program)->inputs(), token_input->tensor, &kn::InputSlot::tensor);
    const auto alternative_input_slot = std::ranges::find(
        (*alternative_program)->inputs(), token_input->tensor, &kn::InputSlot::tensor);
    ASSERT_NE(incumbent_input_slot, (*lowered_program)->inputs().end());
    ASSERT_NE(alternative_input_slot, (*alternative_program)->inputs().end());
    EXPECT_NE(incumbent_input_slot->device_address, alternative_input_slot->device_address)
        << "the actual input activation placement must differ across these retained programs";
    std::string actual_schedule;
    (*lowered_program)->StreamPlanReport([&](std::string_view text) {
      actual_schedule.append(text);
    });
    const auto incumbent_schedule_bytes = actual_schedule.size();
    actual_schedule.push_back('\n');
    (*alternative_program)->StreamPlanReport([&](std::string_view text) {
      actual_schedule.append(text);
    });
    const auto outside_schedule_start = actual_schedule.size();
    actual_schedule.push_back('\n');
    (*outside_program)->StreamPlanReport([&](std::string_view text) {
      actual_schedule.append(text);
    });
    const auto outside_schedule =
        std::string_view(actual_schedule).substr(outside_schedule_start + 1);
    EXPECT_NE(outside_schedule.find(kg::kRmsNormMulUnfused), std::string_view::npos)
        << outside_schedule;
    RecordProperty("actual_d64_incumbent_and_primitive_schedules", actual_schedule);
    const auto persistent_bytes = life_->backend->persistent_bytes();
    if (persistent_bytes != 0) ASSERT_TRUE(persistent.Allocate(persistent_bytes));
    auto launched =
        life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                               {.base = persistent.address(), .bytes = persistent_bytes});
    ASSERT_TRUE(launched) << (launched ? "" : launched.error());
    life_->test_launch = std::move(*launched);
    ASSERT_TRUE(life_->test_weights_rebound.Allocate(kArenaBytes));
    auto rebound_program = lower(inventory->incumbent.plan, request, capture, activation.address(),
                                 life_->test_weights_rebound.address());
    ASSERT_TRUE(rebound_program) << (rebound_program ? "" : rebound_program.error());
    ASSERT_TRUE(life_->test_state_rebound.Allocate(source->layout.bytes));
    auto rebound_state_program =
        lower(inventory->incumbent.plan, request, capture, activation.address(), 0,
              life_->test_state_rebound.address());
    ASSERT_TRUE(rebound_state_program)
        << (rebound_state_program ? "" : rebound_state_program.error());
    const auto certificate_descriptor =
        (*lowered_program)
            ->ProjectComparisonRegionCertificate(*region_boundary,
                                                 kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(certificate_descriptor.status, kn::ComparisonRegionStatus::kClosed)
        << certificate_descriptor.reason;
    ASSERT_EQ(certificate_descriptor.seal.state, kn::ComparisonProofState::kDescriptorVerified);
    const auto alternative_certificate_descriptor =
        (*alternative_program)
            ->ProjectComparisonRegionCertificate(*region_boundary,
                                                 kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(alternative_certificate_descriptor.status, kn::ComparisonRegionStatus::kClosed)
        << alternative_certificate_descriptor.reason << "\n"
        << actual_schedule;
    ASSERT_EQ(alternative_certificate_descriptor.seal.state,
              kn::ComparisonProofState::kDescriptorVerified);
    EXPECT_NE(certificate_descriptor.seal.planned_region,
              alternative_certificate_descriptor.seal.planned_region);
    EXPECT_NE(certificate_descriptor.seal.physical_recipe,
              alternative_certificate_descriptor.seal.physical_recipe);
    const auto outside_certificate_descriptor =
        (*outside_program)
            ->ProjectComparisonRegionCertificate(*region_boundary,
                                                 kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(outside_certificate_descriptor.status, kn::ComparisonRegionStatus::kClosed)
        << outside_certificate_descriptor.reason;
    ASSERT_EQ(outside_certificate_descriptor.seal.state,
              kn::ComparisonProofState::kDescriptorVerified);
    EXPECT_EQ(outside_certificate_descriptor.seal.boundary, certificate_descriptor.seal.boundary);
    EXPECT_EQ(outside_certificate_descriptor.seal.physical_recipe,
              certificate_descriptor.seal.physical_recipe);
    const auto bound_before_bind =
        (*lowered_program)
            ->CompareComparisonRemainder(**alternative_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kBound);
    EXPECT_EQ(bound_before_bind.status, kn::ComparisonRemainderStatus::kUnproved)
        << "descriptor projections cannot grant bound remainder authority";
    const auto rebound_remainder =
        (*lowered_program)
            ->CompareComparisonRemainder(**rebound_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kDescriptor);
    EXPECT_EQ(rebound_remainder.status, kn::ComparisonRemainderStatus::kUnproved)
        << "an otherwise identical graph bound to a different immutable weight allocation must not "
           "share remainder authority";
    const auto rebound_state_certificate =
        (*rebound_state_program)
            ->ProjectComparisonRegionCertificate(*region_boundary,
                                                 kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(rebound_state_certificate.status, kn::ComparisonRegionStatus::kClosed)
        << rebound_state_certificate.reason;
    const auto rebound_state_remainder =
        (*lowered_program)
            ->CompareComparisonRemainder(**rebound_state_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kDescriptor);
    EXPECT_EQ(rebound_state_remainder.status, kn::ComparisonRemainderStatus::kUnproved)
        << "a different external state allocation must not share remainder authority";
    const auto remainder_descriptor =
        (*lowered_program)
            ->CompareComparisonRemainder(**alternative_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kDescriptor);
    std::string remainder_expansion;
    for (const auto node : remainder_descriptor.required_members) {
      if (!remainder_expansion.empty()) remainder_expansion.push_back(',');
      remainder_expansion += std::to_string(node.v);
    }
    ASSERT_EQ(remainder_descriptor.status, kn::ComparisonRemainderStatus::kSame)
        << remainder_descriptor.reason << "\nincumbent schedule:\n"
        << "required members: " << remainder_expansion << "\nincumbent schedule:\n"
        << actual_schedule.substr(0, incumbent_schedule_bytes) << "\nalternative schedule:\n"
        << actual_schedule.substr(incumbent_schedule_bytes + 1);
    EXPECT_TRUE(remainder_descriptor.required_members.empty());
    const auto outside_remainder =
        (*lowered_program)
            ->CompareComparisonRemainder(**outside_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(outside_remainder.status, kn::ComparisonRemainderStatus::kExpand)
        << outside_remainder.reason << "\n"
        << actual_schedule;
    EXPECT_TRUE(std::ranges::contains(outside_remainder.required_members, outside_norm));
    EXPECT_TRUE(std::ranges::contains(outside_remainder.required_members, outside_mul));
    const auto self_remainder_descriptor =
        (*lowered_program)
            ->CompareComparisonRemainder(**lowered_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(self_remainder_descriptor.status, kn::ComparisonRemainderStatus::kSame)
        << self_remainder_descriptor.reason;
    EXPECT_TRUE(self_remainder_descriptor.required_members.empty());
    EXPECT_NE((*lowered_program)->ProjectComparisonRegion(*region_boundary).status,
              kn::ComparisonRegionStatus::kClosed)
        << "the production certificate grants no executable comparison authority";

    auto bound = (*lowered_program)->Bind(*life_->test_launch);
    ASSERT_TRUE(bound) << (bound ? "" : bound.error().detail);
    auto alternative_bound = (*alternative_program)->Bind(*life_->test_launch);
    ASSERT_TRUE(alternative_bound) << (alternative_bound ? "" : alternative_bound.error().detail);
    auto outside_bound = (*outside_program)->Bind(*life_->test_launch);
    ASSERT_TRUE(outside_bound) << (outside_bound ? "" : outside_bound.error().detail);
    const auto certificate_bound =
        (*lowered_program)
            ->ProjectComparisonRegionCertificate(*region_boundary,
                                                 kn::ComparisonCertificatePhase::kBound);
    ASSERT_EQ(certificate_bound.status, kn::ComparisonRegionStatus::kClosed)
        << certificate_bound.reason;
    ASSERT_EQ(certificate_bound.seal.state, kn::ComparisonProofState::kBoundVerified);
    const auto alternative_certificate_bound =
        (*alternative_program)
            ->ProjectComparisonRegionCertificate(*region_boundary,
                                                 kn::ComparisonCertificatePhase::kBound);
    ASSERT_EQ(alternative_certificate_bound.status, kn::ComparisonRegionStatus::kClosed)
        << alternative_certificate_bound.reason;
    ASSERT_EQ(alternative_certificate_bound.seal.state, kn::ComparisonProofState::kBoundVerified);
    const auto remainder_bound =
        (*lowered_program)
            ->CompareComparisonRemainder(**alternative_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kBound);
    ASSERT_EQ(remainder_bound.status, kn::ComparisonRemainderStatus::kSame)
        << remainder_bound.reason << "\n"
        << actual_schedule;
    EXPECT_TRUE(remainder_bound.required_members.empty());
    const auto outside_remainder_bound =
        (*lowered_program)
            ->CompareComparisonRemainder(**outside_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kBound);
    ASSERT_EQ(outside_remainder_bound.status, kn::ComparisonRemainderStatus::kExpand)
        << outside_remainder_bound.reason;
    EXPECT_TRUE(std::ranges::contains(outside_remainder_bound.required_members, outside_norm));
    EXPECT_TRUE(std::ranges::contains(outside_remainder_bound.required_members, outside_mul));
    const auto self_remainder_bound =
        (*lowered_program)
            ->CompareComparisonRemainder(**lowered_program, *region_boundary,
                                         kn::ComparisonCertificatePhase::kBound);
    ASSERT_EQ(self_remainder_bound.status, kn::ComparisonRemainderStatus::kSame)
        << self_remainder_bound.reason;
    EXPECT_TRUE(self_remainder_bound.required_members.empty());
    EXPECT_EQ(certificate_bound.seal.subject, certificate_descriptor.seal.subject);
    EXPECT_EQ(certificate_bound.seal.boundary, certificate_descriptor.seal.boundary);
    EXPECT_EQ(certificate_bound.seal.planned_region, certificate_descriptor.seal.planned_region);
    EXPECT_EQ(certificate_bound.seal.physical_recipe, certificate_descriptor.seal.physical_recipe);
    EXPECT_EQ(certificate_bound.seal.backend_device, certificate_descriptor.seal.backend_device);
    EXPECT_NE(certificate_bound.seal.physical_recipe,
              alternative_certificate_bound.seal.physical_recipe);

    const std::array<std::int32_t, 1> tokens{3};
    for (const auto& slot : (*lowered_program)->inputs()) {
      ASSERT_GT(slot.bytes, 0U);
      std::vector<std::byte> payload(static_cast<std::size_t>(slot.bytes));
      if (slot.kind == kn::InputKind::kTokens || slot.kind == kn::InputKind::kPositions) {
        const auto input = std::ranges::find(step.inputs, slot.tensor, &co::StepInput::tensor);
        ASSERT_NE(input, step.inputs.end());
        const auto values = co::InputValues(*input, source->shape, tokens);
        ASSERT_TRUE(values) << (values ? "" : values.error());
        std::vector<std::int32_t> narrowed(values->size());
        std::ranges::transform(*values, narrowed.begin(),
                               [](std::int64_t value) { return static_cast<std::int32_t>(value); });
        ASSERT_TRUE(en::step_inputs::WriteRows(payload, narrowed));
      } else {
        auto emitted = en::step_inputs::WriteComputed(slot, source->shape, payload);
        ASSERT_TRUE(emitted) << (emitted ? "" : emitted.error());
      }
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(slot.device_address), payload.data(),
                           payload.size(), cudaMemcpyHostToDevice),
                cudaSuccess)
          << "input kind=" << static_cast<int>(slot.kind);
    }
    auto submission = life_->execution->Submission(life_->node.stream(0));
    ASSERT_TRUE(submission) << (submission ? "" : submission.error().detail);
    life_->test_program = std::move(*lowered_program);
    ASSERT_TRUE(life_->test_program->Run(*life_->test_launch, *submission));
    auto fence = life_->execution->Record(life_->node.stream(0));
    ASSERT_TRUE(fence) << (fence ? "" : fence.error().detail);
    for (;;) {
      auto status = life_->execution->Query(*fence);
      ASSERT_TRUE(status) << (status ? "" : status.error().detail);
      if (*status == llmp::providers::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_TRUE(life_->execution->Release(*fence));

    auto cpu = reference::Executor::Create(step.graph);
    ASSERT_TRUE(cpu) << (cpu ? "" : cpu.error().detail);
    for (const auto& weight : step.weights)
      ASSERT_TRUE(cpu->BindValues(weight.tensor, weight_values.at(weight.tensor)));
    for (const auto& state_fact : step.state) {
      const auto elements = ir::Elements(step.graph.tensor(state_fact.tensor).shape());
      ASSERT_TRUE(cpu->BindValues(state_fact.tensor, std::vector<float>(elements, 0.0F)));
    }
    for (const auto& input : step.inputs) {
      auto values = co::InputValues(input, source->shape, tokens);
      ASSERT_TRUE(values) << (values ? "" : values.error());
      ASSERT_TRUE(cpu->BindInts(input.tensor, *values));
    }
    ASSERT_TRUE(cpu->Run()) << "normal CPU reference executes the full token graph";
    ASSERT_EQ(life_->test_program->outputs().size(), 1U);
    const auto output = cpu->Values(step.graph.outputs().front());
    ASSERT_TRUE(output) << (output ? "" : output.error().detail);
    ASSERT_EQ(life_->test_program->outputs().front().bytes, output->size() * sizeof(float));
    std::vector<float> actual_output(output->size());
    ASSERT_EQ(cudaMemcpy(actual_output.data(),
                         reinterpret_cast<const void*>(
                             life_->test_program->outputs().front().device_address),
                         life_->test_program->outputs().front().bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (std::size_t i = 0; i < output->size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual_output[i]) && std::isfinite((*output)[i]));
      EXPECT_NEAR(actual_output[i], (*output)[i], kOutputTolerance) << i;
    }
    for (const auto& state_fact : step.state) {
      const auto expected = cpu->Values(state_fact.tensor);
      ASSERT_TRUE(expected) << (expected ? "" : expected.error().detail);
      const auto& item = source->layout.items[state_fact.item];
      std::vector<__half> actual(expected->size());
      ASSERT_LE(actual.size() * sizeof(__half), item.bytes);
      ASSERT_EQ(
          cudaMemcpy(actual.data(), reinterpret_cast<const void*>(state.address() + item.offset),
                     actual.size() * sizeof(__half), cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (std::size_t i = 0; i < expected->size(); ++i) {
        const float value = __half2float(actual[i]);
        ASSERT_TRUE(std::isfinite(value) && std::isfinite((*expected)[i]));
        EXPECT_NEAR(value, (*expected)[i], kF16StateTolerance) << i;
      }
    }
    life_->test_program.reset();
  }
  auto& activation = life_->test_activation;
  auto& state = life_->test_state;
  const auto backend_view = life_->backend->Backend();
  const auto& recipe = backend_view.owner ? backend_view.owner->backend() : backend_view;
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  const auto submission = life_->execution->Submission(life_->node.stream(0));
  ASSERT_TRUE(submission);
  // Reuse this real source/resource fixture for complete native-whole KV and
  // attention decisions. The old local-cut controls above remain independent.
  std::vector<std::string> first_whole_schedule;
  const auto run_whole_attention = [&](std::uint32_t rows, bool owners) {
    SCOPED_TRACE(owners ? "owner-attention" : std::to_string(rows));
    if (owners) {
      // This larger real pair coexists with production's candidate pair.
      // Grow the same paid catalog owner, rather than bypassing its funder.
      const auto available = llmp::platform::AvailableMemoryBytes();
      ASSERT_TRUE(available);
      constexpr auto capacity = 64 * kMiB;
      ASSERT_LE(life_->paid->capacity(), capacity);
      const auto occupied = life_->node.catalog().OccupancyOf(life_->node.domain()).Total().value();
      const auto delta = capacity - life_->paid->capacity();
      std::uint64_t budget = 0;
      ASSERT_FALSE(__builtin_add_overflow(occupied, delta, &budget));
      const auto grown = life_->paid->Grow(life_->node.catalog(), life_->node.domain(), capacity,
                                           budget, *available, 6ULL << 30U);
      ASSERT_TRUE(grown) << (grown ? "" : grown.error());
    }
    auto whole_source = MakeSmallTokenStateSource(rows, owners);
    ASSERT_TRUE(whole_source) << (whole_source ? "" : whole_source.error());
    auto& whole_step = whole_source->step;
    const auto whole_io = co::GraphIO(whole_step);
    const auto& whole_shape = whole_source->shape;
    std::map<ir::WeightRef, std::uint64_t> offsets;
    std::map<ir::TensorId, std::vector<float>> values;
    std::uint64_t used = 0;
    for (const auto& weight : whole_step.weights) {
      used = (used + 255U) & ~std::uint64_t{255U};
      const auto bytes = ir::StorageBytes(whole_step.graph.tensor(weight.tensor));
      ASSERT_TRUE(bytes);
      ASSERT_LE(*bytes, kArenaBytes - used);
      offsets.emplace(weight.ref, used);
      auto& payload = values[weight.tensor];
      payload.resize(ir::Elements(whole_step.graph.tensor(weight.tensor).shape()));
      for (std::size_t i = 0; i < payload.size(); ++i)
        payload[i] = whole_step.graph.tensor(weight.tensor).shape().rank == 1
                         ? 1.0F
                         : static_cast<float>(static_cast<int>(i % 13) - 6) / 64.0F;
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(life_->test_weights_rebound.address() + used),
                           payload.data(), *bytes, cudaMemcpyHostToDevice),
                cudaSuccess);
      used += *bytes;
    }
    const auto state_offset = [&](const ir::StateRef& ref) {
      const auto found = std::ranges::find(whole_step.state, ref, &co::StepState::ref);
      if (found == whole_step.state.end()) return std::uint64_t{UINT64_MAX};
      return whole_source->layout.items[found->item].offset +
             std::uint64_t{ref.segment} * whole_source->layout.bytes;
    };
    const auto state_bytes = whole_source->layout.bytes * whole_shape.segments.size();
    ASSERT_LE(state_bytes, kArenaBytes);
    std::vector<std::byte> state_image(state_bytes);
    for (const auto& fact : whole_step.state) {
      ASSERT_EQ(whole_step.graph.tensor(fact.tensor).format.encoding, llmp::format::Encoding::kF16);
      const auto count = ir::Elements(whole_step.graph.tensor(fact.tensor).shape());
      const auto offset = state_offset(fact.ref);
      ASSERT_LE(offset + count * sizeof(__half), state_image.size());
      for (std::size_t i = 0; i < count; ++i) {
        const auto value = __float2half(static_cast<float>(static_cast<int>(i % 17) - 8) / 256.0F);
        std::memcpy(state_image.data() + offset + i * sizeof(value), &value, sizeof(value));
      }
    }
    kn::Addresses whole_addresses{.weight = {},
                                  .weight_rows = {},
                                  .state = {},
                                  .save = {},
                                  .save_rows = 0,
                                  .activations = activation.address(),
                                  .activation_bytes = kArenaBytes,
                                  .activation_alignment = 256};
    whole_addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
      return life_->test_weights_rebound.address() + offsets.at(ref);
    };
    whole_addresses.weight_range =
        [&](const ir::WeightRef& ref, llmp::format::PartRole) -> std::optional<kn::BoundedAddress> {
      const auto found = std::ranges::find(whole_step.weights, ref, &co::StepWeight::ref);
      if (found == whole_step.weights.end()) return {};
      const auto bytes = ir::StorageBytes(whole_step.graph.tensor(found->tensor));
      if (!bytes) return {};
      return kn::BoundedAddress{.base = whole_addresses.weight(ref, llmp::format::PartRole::kData),
                                .bytes = *bytes,
                                .alignment = 256};
    };
    whole_addresses.state = [&](const ir::StateRef& ref) {
      return state.address() + state_offset(ref);
    };
    whole_addresses.state_range =
        [&](const ir::StateRef& ref) -> std::optional<kn::BoundedAddress> {
      const auto found = std::ranges::find(whole_step.state, ref, &co::StepState::ref);
      if (found == whole_step.state.end()) return {};
      return kn::BoundedAddress{.base = whole_addresses.state(ref),
                                .bytes = whole_source->layout.items[found->item].bytes,
                                .alignment = 256};
    };
    auto whole_policy = planner::ForStep({}, whole_shape.rows(),
                                         static_cast<std::uint32_t>(whole_shape.segments.size()));
    ASSERT_TRUE(whole_policy);
    auto options = en::PrepareComponentPlanOptions(whole_io, recipe, whole_addresses,
                                                   {.capturable = true, .shape = &whole_shape});
    ASSERT_TRUE(options) << (options ? "" : options.error());
    // This is a token program: ordinary typed staging owns these graph inputs.
    std::vector<ir::TensorId> staged_inputs;
    std::vector<ir::TensorId> aligned_staged_inputs;
    for (const auto& input : whole_step.inputs) {
      if (input.kind != co::InputKind::kCellPositions) staged_inputs.push_back(input.tensor);
      if (input.kind != co::InputKind::kCellPositions && input.kind != co::InputKind::kCommitRows)
        aligned_staged_inputs.push_back(input.tensor);
    }
    options->staged = staged_inputs;
    options->alignment_staged = aligned_staged_inputs;
    auto ordinary = planner::Plan(whole_step.graph.Clone(), *recipe.registry, recipe.device,
                                  *whole_policy, *options);
    ASSERT_TRUE(ordinary) << (ordinary ? "" : ordinary.error().detail);
    std::vector<std::string> schedule;
    for (const auto& call : ordinary->steps)
      schedule.emplace_back(recipe.registry->at(call.implementation).name);
    if (!owners && rows == 1) first_whole_schedule = schedule;
    if (!owners && rows == 4 && schedule == first_whole_schedule) return;
    RecordProperty(owners ? "native_whole_owner_attention_calls"
                          : "native_whole_attention_rows_" + std::to_string(rows),
                   static_cast<int>(schedule.size()));
    auto bindings_owner = en::ComparisonSourceBindings::Create(funder, whole_step, 0);
    ASSERT_TRUE(bindings_owner);
    auto first_ranges = bindings_owner->Resolve(*ordinary, whole_addresses, recipe.written);
    ASSERT_TRUE(first_ranges) << (first_ranges ? "" : first_ranges.error().detail);
    const auto view = bindings_owner->boundary_view();
    ASSERT_TRUE(view);
    const auto* actual_pattern = planner::FindPattern(owners ? "owner_attention" : "rope_store");
    ASSERT_NE(actual_pattern, nullptr);
    const std::array filter{actual_pattern->result};
    auto discovered = planner::DiscoverComparisonPlan(
        *view, filter, false, funder, *recipe.registry, recipe.device, *whole_policy, *options);
    ASSERT_TRUE(discovered) << (discovered ? "" : discovered.error().detail);
    const auto tickets = discovered->discovery.tickets();
    const auto ticket = std::ranges::find_if(tickets, [&](const auto& candidate) {
      return candidate.pattern == actual_pattern->name && candidate.boundary_representable;
    });
    ASSERT_NE(ticket, tickets.end()) << "the default source must select its authentic target site";
    planner::ComparisonSiteSelector actual_site{
        .source_graph_identity = whole_step.graph.identity(),
        .kind = ticket->kind,
        .pattern = std::string(ticket->pattern),
        .version = ticket->version,
        .member_origins = {},
        .member_owner_segments = ticket->owner_segments,
        .boundary_ports = ticket->boundary_ports};
    for (std::size_t group = 0; group + 1 < ticket->member_offsets.size(); ++group)
      actual_site.member_origins.emplace_back(
          ticket->member_origins.begin() + ticket->member_offsets[group],
          ticket->member_origins.begin() + ticket->member_offsets[group + 1]);
    auto actual_inventory = planner::BoundComparison(*view, actual_site, funder, *recipe.registry,
                                                     recipe.device, *whole_policy, *options);
    ASSERT_TRUE(actual_inventory)
        << (actual_inventory ? "" : actual_inventory.error().detail)
        << " error=" << (actual_inventory ? -1 : static_cast<int>(actual_inventory.error().error));
    auto expansion = planner::MakeAlternative(*actual_inventory);
    ASSERT_TRUE(expansion);
    auto own = planner::PlanAlternative(*actual_inventory, std::move(*expansion), *recipe.registry,
                                        recipe.device, *whole_policy, *options);
    ASSERT_TRUE(own) << (own ? "" : own.error().detail);
    std::array<std::unique_ptr<kn::StepProgram>, 2> whole_programs;
    std::array<kn::ComparisonRecipeSeal, 2> whole_descriptors;
    const std::array plans{&actual_inventory->incumbent.plan, &own->plan};
    for (std::size_t arm = 0; arm < plans.size(); ++arm) {
      auto ranges = bindings_owner->Resolve(*plans[arm], whole_addresses, recipe.written);
      ASSERT_TRUE(ranges) << (ranges ? "" : ranges.error().detail);
      const kn::ComparisonRequest whole_request{
          .mode = kn::ComparisonRunMode::kFull,
          .scope = kn::ComparisonProofScope::kWholeProgramNative,
          .bindings = ranges->ranges(),
          .state = whole_step.state,
          .active_stage = 0,
          .expected_boundary = &actual_inventory->source_boundary};
      const kn::LowerInput input(*plans[arm], whole_step, whole_shape, whole_addresses, 256, {},
                                 &whole_request);
      auto lowered = life_->backend->Lower(input);
      ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error());
      const auto seal = (*lowered)->comparison_recipe();
      ASSERT_TRUE(seal);
      ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified)
          << "arm=" << arm << " missing=" << static_cast<int>(seal->missing);
      whole_descriptors[arm] = *seal;
      whole_programs[arm] = std::move(*lowered);
    }
    auto adopted = AdoptWholeAlternative(
        whole_step, whole_shape, *actual_inventory, *view, backend_view, whole_addresses, *options,
        whole_descriptors[0], whole_descriptors[1], funder, *whole_policy, 256);
    ASSERT_TRUE(adopted) << (adopted ? "" : adopted.error());
    whole_programs[1] = std::move(adopted->program);
    for (std::size_t arm = 0; arm < plans.size(); ++arm) {
      ASSERT_TRUE(whole_programs[arm]->Bind(*life_->test_launch));
      EXPECT_TRUE(en::SameWholeProgramRecipe(whole_descriptors[arm],
                                             *whole_programs[arm]->comparison_recipe()));
    }
    std::array<std::unique_ptr<kn::Recorded>, 2> captures;
    if (WholeTimingConfigured())
      ASSERT_TRUE(life_->test_launch->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
    for (std::size_t arm = 0; arm < captures.size(); ++arm) {
      auto recording = whole_programs[arm]->Record(*life_->test_launch, *submission, [&]() {
        return whole_programs[arm]->Run(*life_->test_launch, *submission);
      });
      ASSERT_TRUE(recording) << (recording ? "" : recording.error().detail);
      captures[arm] = std::move(*recording);
    }
    if (WholeTimingConfigured())
      ASSERT_TRUE(life_->test_launch->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
    const auto run_frame = [&](std::uint32_t frame, std::size_t arm, bool replay) {
      auto current_shape = whole_shape;
      for (auto& segment : current_shape.segments) segment.past = frame * 3;
      std::vector<std::int32_t> current_tokens(current_shape.rows());
      for (std::size_t i = 0; i < current_tokens.size(); ++i)
        current_tokens[i] = static_cast<std::int32_t>((i + 2 + frame * 3) % 8);
      auto expected = reference::Executor::Create(whole_step.graph);
      ASSERT_TRUE(expected);
      for (const auto& weight : whole_step.weights)
        ASSERT_TRUE(expected->BindValues(weight.tensor, values.at(weight.tensor)));
      for (const auto& fact : whole_step.state) {
        const std::array<reference::PartView, 1> part{
            {{.role = llmp::format::PartRole::kData,
              .bytes = std::span(state_image)
                           .subspan(state_offset(fact.ref),
                                    whole_source->layout.items[fact.item].bytes)}}};
        ASSERT_TRUE(expected->Bind(fact.tensor, part));
      }
      for (const auto& input : whole_step.inputs) {
        const auto computed = co::InputValues(input, current_shape, current_tokens);
        ASSERT_TRUE(computed);
        ASSERT_TRUE(expected->BindInts(input.tensor, *computed));
      }
      ASSERT_TRUE(expected->Run());
      SCOPED_TRACE(std::to_string(frame) + ":" + std::to_string(arm) + ":" +
                   std::to_string(replay));
      ASSERT_EQ(cudaMemsetAsync(activation.pointer, 0xff, kArenaBytes,
                                static_cast<cudaStream_t>(submission->handle)),
                cudaSuccess);
      ASSERT_EQ(
          cudaMemcpyAsync(state.pointer, state_image.data(), state_image.size(),
                          cudaMemcpyHostToDevice, static_cast<cudaStream_t>(submission->handle)),
          cudaSuccess);
      std::vector<std::vector<std::byte>> payloads;
      for (const auto& slot : whole_programs[arm]->inputs()) {
        auto& payload = payloads.emplace_back(static_cast<std::size_t>(slot.bytes));
        if (slot.kind == kn::InputKind::kTokens || slot.kind == kn::InputKind::kPositions) {
          const auto input =
              std::ranges::find(whole_step.inputs, slot.tensor, &co::StepInput::tensor);
          ASSERT_NE(input, whole_step.inputs.end());
          const auto computed = co::InputValues(*input, current_shape, current_tokens);
          ASSERT_TRUE(computed) << (computed ? "" : computed.error());
          std::vector<std::int32_t> words;
          words.reserve(computed->size());
          for (const auto value : *computed) {
            ASSERT_GE(value, INT32_MIN);
            ASSERT_LE(value, INT32_MAX);
            words.push_back(static_cast<std::int32_t>(value));
          }
          ASSERT_TRUE(en::step_inputs::WriteRows(payload, words));
        } else {
          const auto staged = en::step_inputs::WriteComputed(slot, current_shape, payload);
          ASSERT_TRUE(staged) << (staged ? "" : staged.error());
        }
        ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), payload.data(),
                                  payload.size(), cudaMemcpyHostToDevice,
                                  static_cast<cudaStream_t>(submission->handle)),
                  cudaSuccess);
      }
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(submission->handle)), cudaSuccess);
      if (replay)
        ASSERT_TRUE(life_->test_launch->Launch(*submission, *captures[arm]));
      else
        ASSERT_TRUE(whole_programs[arm]->Run(*life_->test_launch, *submission));
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(submission->handle)), cudaSuccess);
      ASSERT_TRUE(life_->test_launch->DrainGeneratedInputs());
      for (const auto& weight : whole_step.weights) {
        const auto& original = values.at(weight.tensor);
        std::vector<float> unchanged(original.size());
        ASSERT_EQ(cudaMemcpy(unchanged.data(),
                             reinterpret_cast<void*>(life_->test_weights_rebound.address() +
                                                     offsets.at(weight.ref)),
                             original.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(std::memcmp(unchanged.data(), original.data(), original.size() * sizeof(float)),
                  0)
            << weight.tensor.v;
      }
      for (const auto& output_slot : whole_programs[arm]->outputs()) {
        const auto want = expected->Values(output_slot.tensor);
        ASSERT_TRUE(want);
        const auto width = whole_step.graph.tensor(output_slot.tensor).ne(0);
        std::uint64_t first = 0;
        for (std::size_t owner = 0; owner < output_slot.segment; ++owner)
          first += current_shape.segments[owner].outputs;
        const auto expected_slice =
            std::span(*want).subspan(first * width, output_slot.bytes / sizeof(float));
        std::vector<float> got(expected_slice.size());
        ASSERT_EQ(cudaMemcpy(got.data(), reinterpret_cast<void*>(output_slot.device_address),
                             got.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (std::size_t i = 0; i < got.size(); ++i) {
          ASSERT_TRUE(std::isfinite(got[i]));
          EXPECT_NEAR(got[i], expected_slice[i], kOutputTolerance);
        }
      }
      for (const auto& fact : whole_step.state) {
        const auto want = expected->Values(fact.tensor);
        ASSERT_TRUE(want);
        std::vector<__half> got(want->size());
        ASSERT_EQ(cudaMemcpy(got.data(), reinterpret_cast<void*>(whole_addresses.state(fact.ref)),
                             got.size() * sizeof(__half), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (std::size_t i = 0; i < got.size(); ++i) {
          ASSERT_TRUE(std::isfinite(__half2float(got[i])));
          EXPECT_NEAR(__half2float(got[i]), (*want)[i], kF16StateTolerance);
        }
      }
    };
    for (const std::uint32_t frame : {0U, 1U})
      for (std::size_t arm = 0; arm < captures.size(); ++arm)
        for (const bool replay : {false, true}) {
          run_frame(frame, arm, replay);
          ASSERT_FALSE(HasFailure());
        }
    const auto shape_class =
        std::format("attention-{}-r{}", owners ? "owner2" : "rope-store", rows);
    MeasureWholeStatePair(shape_class, adopted->registry->ranks().recipe_comparisons.front().key,
                          captures, [&](std::size_t arm) { run_frame(0, arm, true); });
    ASSERT_FALSE(HasFailure());
    captures = {};
    whole_programs = {};
  };
  run_whole_attention(1, false);
  ASSERT_FALSE(HasFatalFailure());
  run_whole_attention(4, false);
  ASSERT_FALSE(HasFatalFailure());
  run_whole_attention(1, true);
}

TEST(GgmlComparisonRemainderLifetime, UsesActualWriteAndReadSublaunchOrder) {
  const auto value = [](std::size_t first_write, std::size_t first_read, std::size_t last_read,
                        std::uint8_t last_read_subcall = 0, bool kept = false) {
    return kg::ComparisonLifetimeFact{.address = 0x2000,
                                      .storage_id = 1,
                                      .definition_id = 1,
                                      .storage_offset = 0,
                                      .bytes = 64,
                                      .first_write_call = first_write,
                                      .first_write_subcall = 0,
                                      .first_read_call = first_read,
                                      .first_read_subcall = 0,
                                      .last_read_call = last_read,
                                      .last_read_subcall = last_read_subcall,
                                      .has_read = true,
                                      .external_source = false,
                                      .kept = kept};
  };
  const auto preserves = [](std::uintptr_t address, std::uint64_t bytes, std::size_t call,
                            std::uint8_t subcall, const kg::ComparisonLifetimeFact& fact) {
    return kg::ComparisonWritePreservesLifetime(address, bytes, call, subcall, fact);
  };

  EXPECT_FALSE(preserves(0x2010, 4, 1, 0, value(0, 0, 3)));
  EXPECT_TRUE(preserves(0x2010, 4, 2, 0, value(0, 0, 1)));
  EXPECT_TRUE(preserves(0x2010, 4, 1, 0, value(3, 3, 4)));
  EXPECT_TRUE(preserves(0x2010, 4, 2, 0, value(0, 0, 2, 0)));
  EXPECT_FALSE(preserves(0x2010, 4, 2, 0, value(0, 0, 2, 1)));
  EXPECT_FALSE(preserves(0x2010, 4, 3, 0, value(0, 0, 1, 0, true)));
  EXPECT_TRUE(preserves(0x3000, 4, 1, 0, value(0, 0, 3)));
  EXPECT_FALSE(preserves(std::numeric_limits<std::uintptr_t>::max() - 3, 8, 1, 0, value(0, 0, 3)));
}

TEST(GgmlComparisonRemainderLifetime, RejectsMalformedLiveValueIntervals) {
  const kg::ComparisonLifetimeFact malformed{.address = 0,
                                             .bytes = 64,
                                             .first_write_call = 0,
                                             .first_read_call = 0,
                                             .last_read_call = 4,
                                             .has_read = true};
  EXPECT_FALSE(kg::ComparisonWritePreservesLifetime(0x2000, 4, 1, 0, malformed));
  EXPECT_FALSE(kg::ComparisonWritePreservesLifetime(0, 4, 1, 0, malformed));
}

TEST_F(GgmlComparisonLowerForGpu, CommitRowsBindingIsProtectedFromLocalValueAliases) {
  const auto run_case = [&](bool alias_commit_to_body) {
    constexpr std::uint64_t kArenaBytes = 1U << 20U;
    CudaBuffer activation, weight_storage, saved_storage;
    ASSERT_TRUE(activation.Allocate(kArenaBytes));
    ASSERT_TRUE(weight_storage.Allocate(16 * sizeof(float)));
    ASSERT_TRUE(saved_storage.Allocate(16 * sizeof(float)));

    ir::Builder builder;
    builder.SetOwner(0);
    const auto x =
        builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 1})), "features");
    const ir::WeightRef weight_ref{.stage = 0, .component = 0, .role = 0, .layer = 0};
    const auto weight = builder.Weight(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16})),
                                       weight_ref, "norm_weight");
    const auto norm = builder.RmsNorm(x, 1.0e-6F);
    const auto value = builder.Mul(norm, weight);
    const auto saved =
        builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 1})), "saved_rows");
    const auto suffix = builder.Add(value, saved);
    builder.Output(value);
    builder.Output(suffix);
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    co::StepGraph step{.graph = std::move(*graph),
                       .inputs = {{.tensor = x, .kind = co::InputKind::kFeatures, .segment = 0},
                                  {.tensor = saved,
                                   .kind = co::InputKind::kCommitRows,
                                   .segment = 0,
                                   .component = 0,
                                   .layer = 0,
                                   .what = "cache"}},
                       .weights = {{.tensor = weight,
                                    .ref = weight_ref,
                                    .parts = {{"norm_weight", llmp::format::PartRole::kData}}}},
                       .state = {}};
    const auto shape = planner::StepShape{
        .stage = 0,
        .phase = llmp::execution::Phase::kCommit,
        .segments = {{.slot = 0, .rows = 1, .past = 0, .read_cells = 1, .outputs = 1}}};
    const auto io = co::GraphIO(step);
    const auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    planner::PlanOptions options{};
    options.capturable = true;
    options.within = recipe.within;
    options.shape = &shape;
    const std::vector<ir::TensorId> staged{x};
    options.staged = staged;
    options.alignment_staged = staged;
    auto step_policy = planner::ForStep({}, shape.rows(), 1);
    ASSERT_TRUE(step_policy) << (step_policy ? "" : step_policy.error());
    auto preliminary =
        planner::Plan(step.graph.Clone(), *recipe.registry, recipe.device, *step_policy, options);
    ASSERT_TRUE(preliminary) << (preliminary ? "" : preliminary.error().detail);
    const auto placement = [&](ir::TensorId original) -> std::optional<std::uint64_t> {
      if (!original.valid() || original.v >= preliminary->tensors.size()) return {};
      const auto mapped = preliminary->tensors[original.v];
      if (!mapped.valid()) return {};
      const auto root = preliminary->graph.root(mapped);
      const auto found = std::ranges::find(preliminary->placement.offsets, root,
                                           &std::pair<ir::TensorId, std::uint64_t>::first);
      return found == preliminary->placement.offsets.end()
                 ? std::optional<std::uint64_t>{}
                 : std::optional<std::uint64_t>{found->second};
    };
    const auto x_offset = placement(x);
    const auto value_offset = placement(value);
    ASSERT_TRUE(x_offset && value_offset);
    ASSERT_LE(*x_offset, kArenaBytes - 16 * sizeof(float));
    ASSERT_LE(*value_offset, kArenaBytes - 16 * sizeof(float));
    const std::uint64_t saved_address =
        alias_commit_to_body ? activation.address() + *value_offset : saved_storage.address();
    if (alias_commit_to_body)
      ASSERT_EQ(saved_address, activation.address() + *value_offset);
    else
      ASSERT_NE(saved_address, activation.address() + *value_offset);

    std::vector<co::ComparisonExternalBinding> external{
        {.original_root = x,
         .allocation_class = 1,
         .allocation_bytes = kArenaBytes,
         .offset = *x_offset,
         .bytes = 16 * sizeof(float),
         .alignment = 16},
        {.original_root = saved,
         .allocation_class = alias_commit_to_body ? 1U : 2U,
         .allocation_bytes = alias_commit_to_body ? kArenaBytes : 16 * sizeof(float),
         .offset = alias_commit_to_body ? *value_offset : 0,
         .bytes = 16 * sizeof(float),
         .alignment = 16},
        {.original_root = weight,
         .part = llmp::format::PartRole::kData,
         .allocation_class = 3,
         .allocation_bytes = 16 * sizeof(float),
         .bytes = 16 * sizeof(float),
         .alignment = 16}};
    std::vector<kn::ComparisonBindingRange> bindings;
    bindings.push_back({.original_root = x,
                        .allocation_class = 1,
                        .allocation_bytes = kArenaBytes,
                        .offset = *x_offset,
                        .bytes = 16 * sizeof(float),
                        .alignment = 16,
                        .base = activation.address()});
    bindings.push_back(
        {.original_root = saved,
         .allocation_class = alias_commit_to_body ? 1U : 2U,
         .allocation_bytes = alias_commit_to_body ? kArenaBytes : 16 * sizeof(float),
         .offset = alias_commit_to_body ? *value_offset : 0,
         .bytes = 16 * sizeof(float),
         .alignment = 16,
         .base = alias_commit_to_body ? activation.address() : saved_storage.address()});
    bindings.push_back({.original_root = weight,
                        .part = llmp::format::PartRole::kData,
                        .allocation_class = 3,
                        .allocation_bytes = 16 * sizeof(float),
                        .bytes = 16 * sizeof(float),
                        .alignment = 16,
                        .base = weight_storage.address()});
    kn::Addresses addresses{};
    addresses.activations = activation.address();
    addresses.activation_bytes = kArenaBytes;
    addresses.activation_alignment = 256;
    addresses.weight = [&](const ir::WeightRef&, llmp::format::PartRole) {
      return weight_storage.address();
    };
    addresses.save = [&](std::uint32_t layer, std::uint16_t component, std::string_view what,
                         std::uint8_t segment) {
      return layer == 0 && component == 0 && what == "cache" && segment == 0 ? saved_address : 0;
    };
    auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
    const co::ComparisonBoundaryView source_view{io, step.state, 0, external};
    auto source_boundary = co::MakeComparisonBoundary(source_view, funder);
    ASSERT_TRUE(source_boundary) << (source_boundary ? "" : source_boundary.error().detail);
    const auto* pattern = planner::FindPattern("rms_norm_mul");
    ASSERT_NE(pattern, nullptr);
    const planner::ComparisonSiteSelector selector{
        .source_graph_identity = step.graph.identity(),
        .kind = planner::ComparisonSiteKind::kRewrite,
        .pattern = "rms_norm_mul",
        .version = pattern->version,
        .member_origins = {{step.graph.producer(norm)}, {step.graph.producer(value)}},
        .member_owner_segments = {},
        .boundary_ports = {}};
    auto inventory = planner::BoundComparison(source_view, selector, funder, *recipe.registry,
                                              recipe.device, *step_policy, options);
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    const std::array members{step.graph.producer(norm), step.graph.producer(value)};
    const std::array<std::uint32_t, 2> offsets{0, 2};
    auto region = co::MakeComparisonRegionBoundary(
        {.source = source_view, .member_offsets = offsets, .members = members}, funder);
    ASSERT_TRUE(region) << (region ? "" : region.error().detail);
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .bindings = bindings,
                                  .active_stage = 0,
                                  .expected_boundary = &inventory->source_boundary};
    const kn::ComparisonRegionCapture capture{
        .seed = &*region,
        .origins = &*inventory->incumbent.final_origins,
        .final_nodes = &inventory->incumbent.final_node_origins};
    kn::LowerInput lower_input(inventory->incumbent.plan, step, shape, addresses, 16, {}, &request,
                               &capture);
    auto program = life_->backend->Lower(lower_input);
    ASSERT_TRUE(program) << (program ? "" : program.error());
    const auto certificate = (*program)->ProjectComparisonRegionCertificate(
        *region, kn::ComparisonCertificatePhase::kDescriptor);
    if (alias_commit_to_body) {
      ASSERT_EQ(certificate.status, kn::ComparisonRegionStatus::kUnproved);
      EXPECT_EQ(certificate.seal.state, kn::ComparisonProofState::kUnproved);
      EXPECT_FALSE(certificate.reason.empty());
    } else {
      ASSERT_EQ(certificate.status, kn::ComparisonRegionStatus::kClosed) << certificate.reason;
      ASSERT_EQ(certificate.seal.state, kn::ComparisonProofState::kDescriptorVerified);
    }
  };

  ASSERT_NO_FATAL_FAILURE(run_case(false));
  ASSERT_NO_FATAL_FAILURE(run_case(true));
}

TEST_F(GgmlComparisonLowerForGpu, NonpackedSaveAuxiliariesStayOutsideTheLocalCertificate) {
  enum class SaveAlias { kNone, kBodyOutput, kWeight, kRows, kState, kOtherSave, kLiveInput };
  const auto run_case = [&](SaveAlias alias, bool provide_owner_range) {
    constexpr std::uint64_t kArenaBytes = 1U << 20U;
    CudaBuffer local_activation, local_weight_storage, local_save_storage, outside_weight_storage;
    const bool alias_body = alias == SaveAlias::kBodyOutput;
    CudaBuffer& activation = alias_body ? local_activation : life_->test_activation;
    CudaBuffer& weight_storage = alias_body ? local_weight_storage : life_->test_weights;
    CudaBuffer& save_storage = alias_body ? local_save_storage : life_->test_state;
    ASSERT_TRUE(activation.Allocate(kArenaBytes));
    ASSERT_TRUE(weight_storage.Allocate(16 * sizeof(float)));
    if (alias == SaveAlias::kWeight)
      ASSERT_TRUE(outside_weight_storage.Allocate(16 * sizeof(float)));
    // The save occupies the middle eight values; keep one exact guard value on
    // either side so the normal SET_ROWS path checks its complete extent.
    ASSERT_TRUE(save_storage.Allocate(20 * sizeof(float)));
    CudaBuffer state_storage, live_input_storage;
    if (alias == SaveAlias::kState) ASSERT_TRUE(state_storage.Allocate(32 * sizeof(float)));

    ir::Builder builder;
    builder.SetOwner(0);
    const auto x =
        builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 2})), "features");
    const ir::WeightRef weight_ref{.stage = 0, .component = 0, .role = 0, .layer = 0};
    const ir::WeightRef outside_weight_ref{.stage = 0, .component = 0, .role = 1, .layer = 0};
    const auto weight = builder.Weight(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16})),
                                       weight_ref, "norm_weight");
    const auto outside_weight =
        alias == SaveAlias::kWeight
            ? builder.Weight(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16})),
                             outside_weight_ref, "outside_weight")
            : ir::TensorId{};
    const auto norm = builder.RmsNorm(x, 1.0e-6F);
    const auto value = builder.Mul(norm, weight);
    const auto outside_value =
        outside_weight.valid() ? builder.Mul(x, outside_weight) : ir::TensorId{};
    const std::array<std::uint64_t, llmp::format::kMaxDims> view_strides{4, 64, 128, 128};
    const auto saved_view = builder.View(value, ir::ShapeOf({4, 2}), view_strides, 0);
    const auto saved_other = alias == SaveAlias::kOtherSave
                                 ? builder.View(value, ir::ShapeOf({4, 2}), view_strides, 0)
                                 : ir::TensorId{};
    const auto rows =
        builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({2})), "save_rows");
    const auto live_input =
        alias == SaveAlias::kLiveInput
            ? builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16})),
                            "live_outside")
            : ir::TensorId{};
    const ir::StateRef state_ref{.stage = 0, .tensor = 0, .layer = 0, .segment = 0};
    const auto protected_state =
        alias == SaveAlias::kState
            ? builder.State(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 2})),
                            state_ref, "protected_state")
            : ir::TensorId{};
    const auto outside_state_value =
        protected_state.valid() ? builder.Add(x, protected_state) : ir::TensorId{};
    builder.Keep(saved_view);
    if (saved_other.valid()) builder.Keep(saved_other);
    builder.Keep(rows);
    if (live_input.valid()) builder.Keep(live_input);
    if (outside_state_value.valid()) builder.Keep(outside_state_value);
    builder.Output(value);
    if (outside_value.valid()) builder.Output(outside_value);
    if (outside_state_value.valid()) builder.Output(outside_state_value);
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    ASSERT_FALSE(ir::IsContiguous(graph->tensor(saved_view)));
    std::vector<co::StepInput> inputs{{.tensor = x, .kind = co::InputKind::kFeatures, .segment = 0},
                                      {.tensor = rows,
                                       .kind = co::InputKind::kStoreCells,
                                       .segment = 0,
                                       .cells = {.capacity = 2, .per_row = true}}};
    if (live_input.valid())
      inputs.push_back({.tensor = live_input, .kind = co::InputKind::kFeatures, .segment = 0});
    std::vector<co::StepState> states;
    if (protected_state.valid())
      states.push_back(
          {.tensor = protected_state, .ref = state_ref, .read = true, .written = false});
    std::vector<co::CommitSave> saves{{.layer = 0,
                                       .component = 0,
                                       .what = "cache",
                                       .segment = 0,
                                       .tensor = saved_view,
                                       .rows = rows}};
    if (saved_other.valid())
      saves.push_back({.layer = 0,
                       .component = 0,
                       .what = "cache2",
                       .segment = 0,
                       .tensor = saved_other,
                       .rows = rows});
    std::vector<co::StepWeight> weights{
        {.tensor = weight,
         .ref = weight_ref,
         .parts = {{"norm_weight", llmp::format::PartRole::kData}}}};
    if (outside_weight.valid())
      weights.push_back({.tensor = outside_weight,
                         .ref = outside_weight_ref,
                         .parts = {{"outside_weight", llmp::format::PartRole::kData}}});
    co::StepGraph step{.graph = std::move(*graph),
                       .inputs = std::move(inputs),
                       .weights = std::move(weights),
                       .state = std::move(states),
                       .features = value,
                       .saves = std::move(saves)};
    const auto shape = planner::StepShape{
        .stage = 0,
        .phase = llmp::execution::Phase::kVerify,
        .segments = {{.slot = 0, .rows = 2, .past = 0, .read_cells = 2, .outputs = 2}}};
    const auto io = co::GraphIO(step);
    const auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    planner::PlanOptions options{};
    options.capturable = true;
    options.within = recipe.within;
    options.shape = &shape;
    std::vector<ir::TensorId> staged{x, rows};
    if (live_input.valid()) staged.push_back(live_input);
    options.staged = staged;
    options.alignment_staged = staged;
    auto step_policy = planner::ForStep({}, shape.rows(), 1);
    ASSERT_TRUE(step_policy) << (step_policy ? "" : step_policy.error());
    auto preliminary =
        planner::Plan(step.graph.Clone(), *recipe.registry, recipe.device, *step_policy, options);
    ASSERT_TRUE(preliminary) << (preliminary ? "" : preliminary.error().detail);
    const auto placement = [&](ir::TensorId original) -> std::optional<std::uint64_t> {
      if (!original.valid() || original.v >= preliminary->tensors.size()) return {};
      const auto mapped = preliminary->tensors[original.v];
      if (!mapped.valid()) return {};
      const auto root = preliminary->graph.root(mapped);
      const auto found = std::ranges::find(preliminary->placement.offsets, root,
                                           &std::pair<ir::TensorId, std::uint64_t>::first);
      return found == preliminary->placement.offsets.end()
                 ? std::optional<std::uint64_t>{}
                 : std::optional<std::uint64_t>{found->second};
    };
    const auto x_offset = placement(x);
    const auto rows_offset = placement(rows);
    const auto value_offset = placement(value);
    ASSERT_TRUE(x_offset && rows_offset && value_offset);
    const auto live_offset = live_input.valid() ? placement(live_input) : std::nullopt;
    if (live_input.valid()) ASSERT_TRUE(live_offset);
    ASSERT_LE(*x_offset, kArenaBytes - 32 * sizeof(float));
    ASSERT_LE(*rows_offset, kArenaBytes - 2 * sizeof(std::int64_t));
    ASSERT_LE(*value_offset, kArenaBytes - 32 * sizeof(float));
    const std::uint64_t ordinary_save_address = save_storage.address() + sizeof(float);
    const std::uint64_t save_address =
        alias == SaveAlias::kBodyOutput  ? activation.address() + *value_offset
        : alias == SaveAlias::kWeight    ? outside_weight_storage.address()
        : alias == SaveAlias::kRows      ? activation.address() + *rows_offset
        : alias == SaveAlias::kState     ? state_storage.address()
        : alias == SaveAlias::kLiveInput ? activation.address() + *live_offset
                                         : ordinary_save_address;

    std::vector<co::ComparisonExternalBinding> external{
        {.original_root = x,
         .allocation_class = 1,
         .allocation_bytes = kArenaBytes,
         .offset = *x_offset,
         .bytes = 32 * sizeof(float),
         .alignment = 16},
        co::ComparisonExternalBinding{.original_root = rows,
                                      .allocation_class = 1,
                                      .allocation_bytes = kArenaBytes,
                                      .offset = *rows_offset,
                                      .bytes = 2 * sizeof(std::int64_t),
                                      .alignment = alignof(std::int64_t)},
        co::ComparisonExternalBinding{.original_root = weight,
                                      .part = llmp::format::PartRole::kData,
                                      .allocation_class = 2,
                                      .allocation_bytes = 16 * sizeof(float),
                                      .bytes = 16 * sizeof(float),
                                      .alignment = 16}};
    if (outside_weight.valid())
      external.push_back({.original_root = outside_weight,
                          .part = llmp::format::PartRole::kData,
                          .allocation_class = 4,
                          .allocation_bytes = 16 * sizeof(float),
                          .bytes = 16 * sizeof(float),
                          .alignment = 16});
    if (live_input.valid())
      external.push_back({.original_root = live_input,
                          .allocation_class = 1,
                          .allocation_bytes = kArenaBytes,
                          .offset = *live_offset,
                          .bytes = 16 * sizeof(float),
                          .alignment = 16});
    if (protected_state.valid())
      external.push_back({.original_root = protected_state,
                          .allocation_class = 3,
                          .allocation_bytes = 32 * sizeof(float),
                          .bytes = 32 * sizeof(float),
                          .alignment = 16});
    std::vector<kn::ComparisonBindingRange> bindings{
        {.original_root = x,
         .allocation_class = 1,
         .allocation_bytes = kArenaBytes,
         .offset = *x_offset,
         .bytes = 32 * sizeof(float),
         .alignment = 16,
         .base = activation.address()},
        kn::ComparisonBindingRange{.original_root = rows,
                                   .allocation_class = 1,
                                   .allocation_bytes = kArenaBytes,
                                   .offset = *rows_offset,
                                   .bytes = 2 * sizeof(std::int64_t),
                                   .alignment = alignof(std::int64_t),
                                   .base = activation.address()},
        kn::ComparisonBindingRange{.original_root = weight,
                                   .part = llmp::format::PartRole::kData,
                                   .allocation_class = 2,
                                   .allocation_bytes = 16 * sizeof(float),
                                   .bytes = 16 * sizeof(float),
                                   .alignment = 16,
                                   .base = weight_storage.address()}};
    if (outside_weight.valid())
      bindings.push_back({.original_root = outside_weight,
                          .part = llmp::format::PartRole::kData,
                          .allocation_class = 4,
                          .allocation_bytes = 16 * sizeof(float),
                          .bytes = 16 * sizeof(float),
                          .alignment = 16,
                          .base = outside_weight_storage.address()});
    if (live_input.valid())
      bindings.push_back({.original_root = live_input,
                          .allocation_class = 1,
                          .allocation_bytes = kArenaBytes,
                          .offset = *live_offset,
                          .bytes = 16 * sizeof(float),
                          .alignment = 16,
                          .base = activation.address()});
    if (protected_state.valid())
      bindings.push_back({.original_root = protected_state,
                          .allocation_class = 3,
                          .allocation_bytes = 32 * sizeof(float),
                          .bytes = 32 * sizeof(float),
                          .alignment = 16,
                          .base = state_storage.address()});
    kn::Addresses addresses{};
    addresses.activations = activation.address();
    addresses.activation_bytes = kArenaBytes;
    addresses.activation_alignment = 256;
    addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
      return outside_weight.valid() && ref == outside_weight_ref ? outside_weight_storage.address()
                                                                 : weight_storage.address();
    };
    addresses.state = [&](const ir::StateRef& ref) {
      return protected_state.valid() && ref == state_ref ? state_storage.address() : 0;
    };
    addresses.save = [&](std::uint32_t layer, std::uint16_t component, std::string_view what,
                         std::uint8_t segment) {
      if (layer != 0 || component != 0 || segment != 0) return std::uint64_t{0};
      if (what == "cache") return save_address;
      if (what == "cache2")
        return alias == SaveAlias::kOtherSave ? save_address
                                              : ordinary_save_address + 10 * sizeof(float);
      return std::uint64_t{0};
    };
    if (provide_owner_range) {
      addresses.save_range = [&](std::uint32_t layer, std::uint16_t component,
                                 std::string_view what,
                                 std::uint8_t segment) -> std::optional<kn::BoundedAddress> {
        if (layer != 0 || component != 0 || segment != 0 || (what != "cache" && what != "cache2"))
          return std::nullopt;
        const auto range_base = what == "cache" ? save_address : addresses.save(0, 0, what, 0);
        return kn::BoundedAddress{
            .base = range_base, .bytes = 9 * sizeof(float), .alignment = alignof(float)};
      };
    }
    auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
    const co::ComparisonBoundaryView source_view{io, step.state, 0, external};
    const auto* pattern = planner::FindPattern("rms_norm_mul");
    ASSERT_NE(pattern, nullptr);
    const planner::ComparisonSiteSelector selector{
        .source_graph_identity = step.graph.identity(),
        .kind = planner::ComparisonSiteKind::kRewrite,
        .pattern = "rms_norm_mul",
        .version = pattern->version,
        .member_origins = {{step.graph.producer(norm)}, {step.graph.producer(value)}},
        .member_owner_segments = {},
        .boundary_ports = {}};
    auto inventory = planner::BoundComparison(source_view, selector, funder, *recipe.registry,
                                              recipe.device, *step_policy, options);
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    auto alternative = planner::MakeAlternative(*inventory);
    ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
    auto alternative_plan =
        planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                 recipe.device, *step_policy, options);
    ASSERT_TRUE(alternative_plan) << (alternative_plan ? "" : alternative_plan.error().detail);
    ASSERT_TRUE(alternative_plan->final_origins);
    const auto selected_recipe = [](const planner::StepPlan& plan) {
      std::vector<std::pair<std::size_t, std::size_t>> recipe;
      recipe.reserve(plan.steps.size());
      for (const auto& call : plan.steps)
        recipe.emplace_back(call.implementation, call.nodes.size());
      return recipe;
    };
    ASSERT_NE(selected_recipe(inventory->incumbent.plan), selected_recipe(alternative_plan->plan))
        << "the save comparison must exercise two actual selected recipes";
    const std::array members{step.graph.producer(norm), step.graph.producer(value)};
    const std::array<std::uint32_t, 2> offsets{0, 2};
    auto region = co::MakeComparisonRegionBoundary(
        {.source = source_view, .member_offsets = offsets, .members = members}, funder);
    ASSERT_TRUE(region) << (region ? "" : region.error().detail);
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .bindings = bindings,
                                  .state = step.state,
                                  .active_stage = 0,
                                  .expected_boundary = &inventory->source_boundary};
    const kn::ComparisonRegionCapture capture{
        .seed = &*region,
        .origins = &*inventory->incumbent.final_origins,
        .final_nodes = &inventory->incumbent.final_node_origins};
    const kn::ComparisonRegionCapture alternative_capture{
        .seed = &*region,
        .origins = &*alternative_plan->final_origins,
        .final_nodes = &alternative_plan->final_node_origins};
    kn::LowerInput lower_input(inventory->incumbent.plan, step, shape, addresses, 16, {}, &request,
                               &capture);
    kn::LowerInput alternative_input(alternative_plan->plan, step, shape, addresses, 16, {},
                                     &request, &alternative_capture);
    auto program = life_->backend->Lower(lower_input);
    ASSERT_TRUE(program) << (program ? "" : program.error());
    auto alternative_program = life_->backend->Lower(alternative_input);
    ASSERT_TRUE(alternative_program) << (alternative_program ? "" : alternative_program.error());
    const auto certificate = (*program)->ProjectComparisonRegionCertificate(
        *region, kn::ComparisonCertificatePhase::kDescriptor);
    if (alias_body) {
      ASSERT_EQ(certificate.status, kn::ComparisonRegionStatus::kUnproved);
      EXPECT_FALSE(certificate.reason.empty());
    } else {
      ASSERT_EQ(certificate.status, kn::ComparisonRegionStatus::kClosed) << certificate.reason;
      ASSERT_EQ(certificate.seal.state, kn::ComparisonProofState::kDescriptorVerified);
      const auto alternative_certificate =
          (*alternative_program)
              ->ProjectComparisonRegionCertificate(*region,
                                                   kn::ComparisonCertificatePhase::kDescriptor);
      ASSERT_EQ(alternative_certificate.status, kn::ComparisonRegionStatus::kClosed)
          << alternative_certificate.reason;
      const auto remainder_descriptor = (*program)->CompareComparisonRemainder(
          **alternative_program, *region, kn::ComparisonCertificatePhase::kDescriptor);
      if (!provide_owner_range) {
        EXPECT_EQ(remainder_descriptor.status, kn::ComparisonRemainderStatus::kUnproved);
        EXPECT_FALSE(remainder_descriptor.reason.empty());
      } else if (alias == SaveAlias::kWeight || alias == SaveAlias::kRows ||
                 alias == SaveAlias::kState || alias == SaveAlias::kOtherSave ||
                 alias == SaveAlias::kLiveInput) {
        if (alias == SaveAlias::kRows) {
          ASSERT_EQ(remainder_descriptor.status, kn::ComparisonRemainderStatus::kExpand)
              << remainder_descriptor.reason;
          ASSERT_EQ(remainder_descriptor.required_members.size(), 2U);
          EXPECT_EQ(remainder_descriptor.required_members[0], step.graph.producer(saved_view));
          EXPECT_EQ(remainder_descriptor.required_members[1], step.graph.producer(rows));
        } else {
          ASSERT_EQ(remainder_descriptor.status, kn::ComparisonRemainderStatus::kUnproved)
              << remainder_descriptor.reason;
          EXPECT_FALSE(remainder_descriptor.reason.empty());
        }
        if (alias == SaveAlias::kLiveInput) {
          ASSERT_EQ(remainder_descriptor.status, kn::ComparisonRemainderStatus::kUnproved);
          EXPECT_EQ(remainder_descriptor.reason,
                    "saved destination overlaps an activation value that remains live");
        }
        return;
      } else {
        ASSERT_EQ(remainder_descriptor.status, kn::ComparisonRemainderStatus::kSame)
            << "reason=" << remainder_descriptor.reason << " required original members:" << [&] {
                 std::string ids;
                 for (const auto member : remainder_descriptor.required_members)
                   ids += " " + std::to_string(member.v);
                 return ids;
               }();
        EXPECT_TRUE(remainder_descriptor.required_members.empty());
      }

      if (!provide_owner_range) return;

      // Exercise the ordinary bound program and the actual generated CONT →
      // SET_ROWS save. The Life owner retains the submitted program, launch,
      // library, and buffers through the authenticated fence/teardown path.
      std::array<float, 16> weight_values{};
      weight_values.fill(1.0F);
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(weight_storage.address()), weight_values.data(),
                           sizeof(weight_values), cudaMemcpyHostToDevice),
                cudaSuccess);
      std::array<float, 32> input_values{};
      for (std::size_t i = 0; i < input_values.size(); ++i)
        input_values[i] = static_cast<float>(i + 1) * 0.25F;
      std::array<std::array<float, 32>, 2> input_frames{input_values, input_values};
      for (std::size_t i = 0; i < input_frames[1].size(); ++i)
        input_frames[1][i] = input_values[i] * -0.5F + 0.125F;

      constexpr float kGuard = -8765.5F;
      std::array<float, 10> saved_values{};
      saved_values.fill(kGuard);
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(save_storage.address()), saved_values.data(),
                           sizeof(saved_values), cudaMemcpyHostToDevice),
                cudaSuccess);

      const auto persistent_bytes = life_->backend->persistent_bytes();
      ASSERT_TRUE(life_->test_pool.Allocate(1U << 20U));
      if (persistent_bytes != 0) ASSERT_TRUE(life_->test_persistent.Allocate(persistent_bytes));
      auto launched = life_->backend->Launch(
          {.base = life_->test_pool.address(), .size = llmp::base::Bytes(1U << 20U)},
          {.base = life_->test_persistent.address(), .bytes = persistent_bytes});
      ASSERT_TRUE(launched) << (launched ? "" : launched.error());
      life_->test_launch = std::move(*launched);
      auto bound = (*program)->Bind(*life_->test_launch);
      ASSERT_TRUE(bound) << (bound ? "" : bound.error().detail);
      auto alternative_bound = (*alternative_program)->Bind(*life_->test_launch);
      ASSERT_TRUE(alternative_bound) << (alternative_bound ? "" : alternative_bound.error().detail);
      const auto bound_remainder = (*program)->CompareComparisonRemainder(
          **alternative_program, *region, kn::ComparisonCertificatePhase::kBound);
      ASSERT_EQ(bound_remainder.status, kn::ComparisonRemainderStatus::kSame)
          << bound_remainder.reason;
      EXPECT_TRUE(bound_remainder.required_members.empty());

      life_->test_program = std::move(*program);
      life_->test_peer_program = std::move(*alternative_program);
      std::array<std::array<std::vector<float>, 2>, 2> arm_outputs;
      std::array<std::array<std::array<float, 8>, 2>, 2> arm_saves{};
      std::array<std::int64_t, 2> row_values{};
      for (std::size_t arm = 0; arm < 2; ++arm) {
        auto* current = arm == 0 ? life_->test_program.get() : life_->test_peer_program.get();
        ASSERT_NE(current, nullptr);
        for (std::size_t frame = 0; frame < input_frames.size(); ++frame) {
          saved_values.fill(kGuard);
          ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(save_storage.address()), saved_values.data(),
                               sizeof(saved_values), cudaMemcpyHostToDevice),
                    cudaSuccess);
          for (const auto& slot : current->inputs()) {
            ASSERT_GT(slot.bytes, 0U);
            std::vector<std::byte> payload(static_cast<std::size_t>(slot.bytes));
            if (slot.tensor == x) {
              ASSERT_EQ(slot.kind, kn::InputKind::kFeatures);
              ASSERT_EQ(payload.size(), sizeof(input_values));
              std::memcpy(payload.data(), input_frames[frame].data(), sizeof(input_frames[frame]));
            } else {
              ASSERT_EQ(slot.tensor, rows);
              ASSERT_EQ(slot.kind, kn::InputKind::kStoreCells);
              auto emitted = en::step_inputs::WriteComputed(slot, shape, payload);
              ASSERT_TRUE(emitted) << (emitted ? "" : emitted.error());
              ASSERT_EQ(payload.size(), sizeof(row_values));
              std::memcpy(row_values.data(), payload.data(), sizeof(row_values));
            }
            ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(slot.device_address), payload.data(),
                                 payload.size(), cudaMemcpyHostToDevice),
                      cudaSuccess);
          }
          auto submission = life_->execution->Submission(life_->node.stream(0));
          ASSERT_TRUE(submission) << (submission ? "" : submission.error().detail);
          ASSERT_TRUE(current->Run(*life_->test_launch, *submission));
          auto fence = life_->execution->Record(life_->node.stream(0));
          ASSERT_TRUE(fence) << (fence ? "" : fence.error().detail);
          for (;;) {
            auto status = life_->execution->Query(*fence);
            ASSERT_TRUE(status) << (status ? "" : status.error().detail);
            if (*status == llmp::providers::FenceState::kComplete) break;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
          }
          ASSERT_TRUE(life_->execution->Release(*fence));
          ASSERT_EQ(
              cudaMemcpy(saved_values.data(), reinterpret_cast<const void*>(save_storage.address()),
                         sizeof(saved_values), cudaMemcpyDeviceToHost),
              cudaSuccess);
          EXPECT_EQ(saved_values.front(), kGuard);
          EXPECT_EQ(saved_values.back(), kGuard);
          std::copy_n(saved_values.begin() + 1, arm_saves[arm][frame].size(),
                      arm_saves[arm][frame].begin());
          ASSERT_EQ(current->outputs().size(), 1U);
          const auto& output = current->outputs().front();
          ASSERT_EQ(output.bytes, sizeof(input_values));
          arm_outputs[arm][frame].resize(input_values.size());
          ASSERT_EQ(cudaMemcpy(arm_outputs[arm][frame].data(),
                               reinterpret_cast<const void*>(output.device_address), output.bytes,
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
        }
      }
      for (std::size_t frame = 0; frame < input_frames.size(); ++frame) {
        auto cpu = reference::Executor::Create(step.graph);
        ASSERT_TRUE(cpu) << (cpu ? "" : cpu.error().detail);
        ASSERT_TRUE(cpu->BindValues(
            weight, std::vector<float>(weight_values.begin(), weight_values.end())));
        ASSERT_TRUE(cpu->BindValues(
            x, std::vector<float>(input_frames[frame].begin(), input_frames[frame].end())));
        ASSERT_TRUE(cpu->BindInts(rows, row_values));
        ASSERT_TRUE(cpu->Run());
        const auto expected_output = cpu->Values(value);
        ASSERT_TRUE(expected_output) << (expected_output ? "" : expected_output.error().detail);
        const auto expected_saved = cpu->Values(saved_view);
        ASSERT_TRUE(expected_saved) << (expected_saved ? "" : expected_saved.error().detail);
        ASSERT_EQ(expected_output->size(), arm_outputs[0][frame].size());
        ASSERT_EQ(expected_saved->size(), arm_saves[0][frame].size());
        for (std::size_t i = 0; i < expected_output->size(); ++i) {
          ASSERT_TRUE(std::isfinite((*expected_output)[i]));
          for (const auto& arm_output : arm_outputs) {
            ASSERT_TRUE(std::isfinite(arm_output[frame][i]));
            EXPECT_NEAR(arm_output[frame][i], (*expected_output)[i], 1.0e-4F)
                << "frame " << frame << " output " << i;
          }
        }
        for (std::size_t i = 0; i < expected_saved->size(); ++i) {
          ASSERT_TRUE(std::isfinite((*expected_saved)[i]));
          for (const auto& arm_save : arm_saves) {
            ASSERT_TRUE(std::isfinite(arm_save[frame][i]));
            EXPECT_NEAR(arm_save[frame][i], (*expected_saved)[i], 1.0e-4F)
                << "frame " << frame << " save " << i;
          }
        }
      }
    }
  };

  ASSERT_NO_FATAL_FAILURE(run_case(SaveAlias::kBodyOutput, true));
  ASSERT_NO_FATAL_FAILURE(run_case(SaveAlias::kWeight, true));
  ASSERT_NO_FATAL_FAILURE(run_case(SaveAlias::kRows, true));
  ASSERT_NO_FATAL_FAILURE(run_case(SaveAlias::kState, true));
  ASSERT_NO_FATAL_FAILURE(run_case(SaveAlias::kOtherSave, true));
  ASSERT_NO_FATAL_FAILURE(run_case(SaveAlias::kLiveInput, true));
  ASSERT_NO_FATAL_FAILURE(run_case(SaveAlias::kNone, true));
  ASSERT_NO_FATAL_FAILURE(run_case(SaveAlias::kNone, false));
}

TEST_F(GgmlComparisonLowerForGpu, PackedSavedSubviewKeepsItsFlattenedSourceOffset) {
  constexpr std::uint64_t kArenaBytes = 1U << 20U;
  CudaBuffer activation, weight_storage, save_storage;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(weight_storage.Allocate(32 * sizeof(float)));
  ASSERT_TRUE(save_storage.Allocate(16 * 2 * sizeof(float)));

  ir::Builder builder;
  builder.SetOwner(0);
  const auto x =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({32, 2})), "features");
  const ir::WeightRef weight_ref{.stage = 0, .component = 0, .role = 0, .layer = 0};
  const auto weight = builder.Weight(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({32})),
                                     weight_ref, "norm_weight");
  const auto norm = builder.RmsNorm(x, 1.0e-6F);
  const auto value = builder.Mul(norm, weight);
  const std::array<std::uint64_t, llmp::format::kMaxDims> view_strides{4, 64, 128, 128};
  const auto saved_view = builder.View(x, ir::ShapeOf({16, 2}), view_strides, 64);
  const auto rows =
      builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({2})), "save_rows");
  builder.Keep(saved_view);
  builder.Keep(rows);
  builder.Output(value);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  ASSERT_TRUE(ir::IsContiguous(graph->tensor(saved_view)));
  ASSERT_EQ(graph->tensor(saved_view).view_offset, 64U);

  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = x, .kind = co::InputKind::kFeatures, .segment = 0},
                                {.tensor = rows,
                                 .kind = co::InputKind::kStoreCells,
                                 .segment = 0,
                                 .cells = {.capacity = 2, .per_row = true}}},
                     .weights = {{.tensor = weight,
                                  .ref = weight_ref,
                                  .parts = {{"norm_weight", llmp::format::PartRole::kData}}}},
                     .state = {},
                     .saves = {{.layer = 0,
                                .component = 0,
                                .what = "cache",
                                .segment = 0,
                                .tensor = saved_view,
                                .rows = rows}}};
  const planner::StepShape shape{
      .stage = 0,
      .phase = llmp::execution::Phase::kVerify,
      .segments = {{.slot = 0, .rows = 2, .past = 0, .read_cells = 2, .outputs = 2}}};
  const auto io = co::GraphIO(step);
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  planner::PlanOptions options{};
  options.capturable = true;
  options.within = recipe.within;
  options.shape = &shape;
  const std::vector<ir::TensorId> staged{x, rows};
  options.staged = staged;
  options.alignment_staged = staged;
  auto step_policy = planner::ForStep({}, shape.rows(), 1);
  ASSERT_TRUE(step_policy) << (step_policy ? "" : step_policy.error());
  auto preliminary =
      planner::Plan(step.graph.Clone(), *recipe.registry, recipe.device, *step_policy, options);
  ASSERT_TRUE(preliminary) << (preliminary ? "" : preliminary.error().detail);
  const auto placement = [&](ir::TensorId original) -> std::optional<std::uint64_t> {
    if (!original.valid() || original.v >= preliminary->tensors.size()) return {};
    const auto mapped = preliminary->tensors[original.v];
    if (!mapped.valid()) return {};
    const auto root = preliminary->graph.root(mapped);
    const auto found = std::ranges::find(preliminary->placement.offsets, root,
                                         &std::pair<ir::TensorId, std::uint64_t>::first);
    return found == preliminary->placement.offsets.end()
               ? std::optional<std::uint64_t>{}
               : std::optional<std::uint64_t>{found->second};
  };
  const auto x_offset = placement(x);
  const auto rows_offset = placement(rows);
  ASSERT_TRUE(x_offset && rows_offset);
  ASSERT_LE(*x_offset, kArenaBytes - 32 * 2 * sizeof(float));
  ASSERT_LE(*rows_offset, kArenaBytes - 2 * sizeof(std::int64_t));

  const std::array external{co::ComparisonExternalBinding{.original_root = x,
                                                          .allocation_class = 1,
                                                          .allocation_bytes = kArenaBytes,
                                                          .offset = *x_offset,
                                                          .bytes = 32 * 2 * sizeof(float),
                                                          .alignment = 16},
                            co::ComparisonExternalBinding{.original_root = rows,
                                                          .allocation_class = 1,
                                                          .allocation_bytes = kArenaBytes,
                                                          .offset = *rows_offset,
                                                          .bytes = 2 * sizeof(std::int64_t),
                                                          .alignment = alignof(std::int64_t)},
                            co::ComparisonExternalBinding{.original_root = weight,
                                                          .part = llmp::format::PartRole::kData,
                                                          .allocation_class = 2,
                                                          .allocation_bytes = 32 * sizeof(float),
                                                          .bytes = 32 * sizeof(float),
                                                          .alignment = 16}};
  const std::array bindings{kn::ComparisonBindingRange{.original_root = x,
                                                       .allocation_class = 1,
                                                       .allocation_bytes = kArenaBytes,
                                                       .offset = *x_offset,
                                                       .bytes = 32 * 2 * sizeof(float),
                                                       .alignment = 16,
                                                       .base = activation.address()},
                            kn::ComparisonBindingRange{.original_root = rows,
                                                       .allocation_class = 1,
                                                       .allocation_bytes = kArenaBytes,
                                                       .offset = *rows_offset,
                                                       .bytes = 2 * sizeof(std::int64_t),
                                                       .alignment = alignof(std::int64_t),
                                                       .base = activation.address()},
                            kn::ComparisonBindingRange{.original_root = weight,
                                                       .part = llmp::format::PartRole::kData,
                                                       .allocation_class = 2,
                                                       .allocation_bytes = 32 * sizeof(float),
                                                       .bytes = 32 * sizeof(float),
                                                       .alignment = 16,
                                                       .base = weight_storage.address()}};
  kn::Addresses addresses{};
  addresses.activations = activation.address();
  addresses.activation_bytes = kArenaBytes;
  addresses.activation_alignment = 256;
  addresses.weight = [&](const ir::WeightRef&, llmp::format::PartRole) {
    return weight_storage.address();
  };
  addresses.save = [&](std::uint32_t layer, std::uint16_t component, std::string_view what,
                       std::uint8_t segment) {
    return layer == 0 && component == 0 && what == "cache" && segment == 0 ? save_storage.address()
                                                                           : 0;
  };
  addresses.save_range = [&](std::uint32_t layer, std::uint16_t component, std::string_view what,
                             std::uint8_t segment) -> std::optional<kn::BoundedAddress> {
    if (layer != 0 || component != 0 || what != "cache" || segment != 0) return std::nullopt;
    return kn::BoundedAddress{
        .base = save_storage.address(), .bytes = 16 * 2 * sizeof(float), .alignment = 16};
  };

  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  const co::ComparisonBoundaryView source_view{io, step.state, 0, external};
  const auto* pattern = planner::FindPattern("rms_norm_mul");
  ASSERT_NE(pattern, nullptr);
  const planner::ComparisonSiteSelector selector{
      .source_graph_identity = step.graph.identity(),
      .kind = planner::ComparisonSiteKind::kRewrite,
      .pattern = "rms_norm_mul",
      .version = pattern->version,
      .member_origins = {{step.graph.producer(norm)}, {step.graph.producer(value)}},
      .member_owner_segments = {},
      .boundary_ports = {}};
  auto inventory = planner::BoundComparison(source_view, selector, funder, *recipe.registry,
                                            recipe.device, *step_policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  auto alternative = planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto alternative_plan = planner::PlanAlternative(
      *inventory, std::move(*alternative), *recipe.registry, recipe.device, *step_policy, options);
  ASSERT_TRUE(alternative_plan) << (alternative_plan ? "" : alternative_plan.error().detail);
  ASSERT_TRUE(alternative_plan->final_origins);
  const std::array members{step.graph.producer(norm), step.graph.producer(value)};
  const std::array<std::uint32_t, 2> offsets{0, 2};
  auto region = co::MakeComparisonRegionBoundary(
      {.source = source_view, .member_offsets = offsets, .members = members}, funder);
  ASSERT_TRUE(region) << (region ? "" : region.error().detail);
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                .bindings = bindings,
                                .active_stage = 0,
                                .expected_boundary = &inventory->source_boundary};
  const kn::ComparisonRegionCapture capture{
      .seed = &*region,
      .origins = &*inventory->incumbent.final_origins,
      .final_nodes = &inventory->incumbent.final_node_origins};
  const kn::ComparisonRegionCapture alternative_capture{
      .seed = &*region,
      .origins = &*alternative_plan->final_origins,
      .final_nodes = &alternative_plan->final_node_origins};
  kn::LowerInput input(inventory->incumbent.plan, step, shape, addresses, 16, {}, &request,
                       &capture);
  kn::LowerInput alternative_input(alternative_plan->plan, step, shape, addresses, 16, {}, &request,
                                   &alternative_capture);
  auto program = life_->backend->Lower(input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  auto alternative_program = life_->backend->Lower(alternative_input);
  ASSERT_TRUE(alternative_program) << (alternative_program ? "" : alternative_program.error());
  const auto certificate = (*program)->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(certificate.status, kn::ComparisonRegionStatus::kClosed) << certificate.reason;
  ASSERT_EQ(certificate.seal.state, kn::ComparisonProofState::kDescriptorVerified);
  const auto alternative_certificate =
      (*alternative_program)
          ->ProjectComparisonRegionCertificate(*region,
                                               kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(alternative_certificate.status, kn::ComparisonRegionStatus::kClosed)
      << alternative_certificate.reason;
  const auto descriptor_remainder = (*program)->CompareComparisonRemainder(
      **alternative_program, *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor_remainder.status, kn::ComparisonRemainderStatus::kSame)
      << "reason=" << descriptor_remainder.reason << " required original members:" << [&] {
           std::string ids;
           for (const auto member : descriptor_remainder.required_members)
             ids += " " + std::to_string(member.v);
           return ids;
         }();
  EXPECT_TRUE(descriptor_remainder.required_members.empty());

  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_TRUE(life_->test_pool.Allocate(1U << 20U));
  if (persistent_bytes != 0) ASSERT_TRUE(life_->test_persistent.Allocate(persistent_bytes));
  auto launched = life_->backend->Launch(
      {.base = life_->test_pool.address(), .size = llmp::base::Bytes(1U << 20U)},
      {.base = life_->test_persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launched) << (launched ? "" : launched.error());
  life_->test_launch = std::move(*launched);
  ASSERT_TRUE((*program)->Bind(*life_->test_launch));
  ASSERT_TRUE((*alternative_program)->Bind(*life_->test_launch));
  const auto bound_remainder = (*program)->CompareComparisonRemainder(
      **alternative_program, *region, kn::ComparisonCertificatePhase::kBound);
  EXPECT_EQ(bound_remainder.status, kn::ComparisonRemainderStatus::kSame) << bound_remainder.reason;
}

TEST_F(GgmlComparisonLowerForGpu, MalformedOriginalMapReturnsLoweringError) {
  auto fixture = MakeFixture();
  ASSERT_TRUE(fixture.plan);
  ASSERT_GE(fixture.plan->tensors.size(), 2U);
  fixture.plan->tensors.pop_back();
  kn::ComparisonRequest request{};
  kn::LowerInput input(*fixture.plan, fixture.io(), fixture.addresses, std::nullopt, &request);
  auto result = life_->backend->Lower(input);
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().find("wrong size"), std::string::npos);
}

TEST_F(GgmlComparisonLowerForGpu, UndersizedMeasuredInputRangeReturnsLoweringError) {
  auto fixture = MakeFixture();
  ASSERT_TRUE(fixture.plan);
  std::array<kn::ComparisonBindingRange, 1> ranges{
      kn::ComparisonBindingRange{.original_root = fixture.input,
                                 .allocation_class = 1,
                                 .allocation_bytes = 1,
                                 .bytes = 1,
                                 .alignment = 1}};
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull, .bindings = ranges};
  kn::LowerInput input(*fixture.plan, fixture.io(), fixture.addresses, std::nullopt, &request);
  auto result = life_->backend->Lower(input);
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().find("smaller than its readable/written extent"), std::string::npos);
}

TEST_F(GgmlComparisonLowerForGpu, MissingOwnedBoundaryCannotPromoteProof) {
  auto fixture = MakeFixture();
  ASSERT_TRUE(fixture.plan);
  std::array<kn::ComparisonBindingRange, 1> ranges{
      kn::ComparisonBindingRange{.original_root = fixture.input,
                                 .allocation_class = 1,
                                 .allocation_bytes = 1024,
                                 .bytes = 1024,
                                 .alignment = 16}};
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull, .bindings = ranges};
  kn::LowerInput input(*fixture.plan, fixture.io(), fixture.addresses, std::nullopt, &request);
  auto result = life_->backend->Lower(input);
  ASSERT_TRUE(result) << (result ? "" : result.error());
  const auto seal = (*result)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kBoundary);
}

TEST_F(GgmlComparisonLowerForGpu, OwnedBoundaryCanPromoteACompleteDescriptorRecipe) {
  auto fixture = MakeFixture();
  ASSERT_TRUE(fixture.plan);
  const std::array<kn::ComparisonBindingRange, 1> ranges{
      kn::ComparisonBindingRange{.original_root = fixture.input,
                                 .allocation_class = 1,
                                 .allocation_bytes = 1024,
                                 .bytes = 1024,
                                 .alignment = 16}};
  const co::ComparisonExternalBinding external{.original_root = fixture.input,
                                               .allocation_class = 1,
                                               .allocation_bytes = 1024,
                                               .bytes = 1024,
                                               .alignment = 16};
  const auto io = fixture.io();
  const std::array<co::ComparisonExternalBinding, 1> externals{external};
  const co::ComparisonBoundaryView view{io, {}, 0, externals};
  auto boundary =
      co::MakeComparisonBoundary(view, std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = ranges, .expected_boundary = &*boundary};
  kn::LowerInput input(*fixture.plan, io, fixture.addresses, std::nullopt, &request);
  auto result = life_->backend->Lower(input);
  ASSERT_TRUE(result) << (result ? "" : result.error());
  const auto seal = (*result)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
  EXPECT_NE(seal->subject, llmp::base::Sha256Digest{});
  EXPECT_NE(seal->boundary, llmp::base::Sha256Digest{});
  EXPECT_NE(seal->backend_recipe, llmp::base::Sha256Digest{});
  EXPECT_NE(seal->backend_build, llmp::base::Sha256Digest{});
  EXPECT_NE(seal->backend_selection, llmp::base::Sha256Digest{});
  const auto effective = life_->backend->Backend();
  ASSERT_TRUE(effective.owner);
  const auto& owned_backend = effective.owner->backend();
  ASSERT_TRUE(owned_backend.registry);
  const auto declaration = effective.owner->Ready();
  ASSERT_TRUE(declaration) << (declaration ? "" : declaration.error());
  EXPECT_NE(declaration->comparison_selection, llmp::base::Sha256Digest{});
  EXPECT_EQ(seal->registry, owned_backend.registry->SelectionDigest());
  EXPECT_EQ(seal->backend_selection, declaration->comparison_selection);
}

TEST_F(GgmlComparisonLowerForGpu, ExactSiteAlternativeKeepsTheAcceptedRmsNormMulOrigins) {
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.role = 1};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto input = builder.Input(ir::Dense(kF32, ir::ShapeOf({16, 4})));
  const auto weight = builder.Weight(ir::Dense(kF32, ir::ShapeOf({16})), weight_ref, "norm.weight");
  builder.Keep(input);
  const auto normalized = builder.RmsNorm(input, 1.0e-6F);
  const auto scaled = builder.Mul(normalized, weight);
  builder.Output(scaled);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  const std::array<co::StepWeight, 1> weights{
      co::StepWeight{.tensor = weight,
                     .ref = weight_ref,
                     .parts = {{"norm.weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), weights};
  const auto input_bytes = ir::StorageBytes(graph->tensor(input));
  const auto weight_parts = llmp::format::PartsOf(graph->tensor(weight).format);
  ASSERT_TRUE(input_bytes && weight_parts && weight_parts->size() == 1);
  const auto weight_part = (*weight_parts)[0];
  const std::array<co::ComparisonExternalBinding, 2> external{
      co::ComparisonExternalBinding{.original_root = input,
                                    .allocation_class = 1,
                                    .allocation_bytes = *input_bytes,
                                    .bytes = *input_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = weight_part.readable,
                                    .bytes = weight_part.readable,
                                    .alignment = 16}};
  const co::ComparisonBoundaryView boundary{io, {}, 0, external};
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* pattern = llmp::planner::FindPattern("rms_norm_mul");
  ASSERT_NE(pattern, nullptr);
  const llmp::planner::ComparisonSiteSelector selector{
      .source_graph_identity = graph->identity(),
      .kind = llmp::planner::ComparisonSiteKind::kRewrite,
      .pattern = "rms_norm_mul",
      .version = pattern->version,
      .member_origins = {{ir::NodeId{2}}, {ir::NodeId{3}}},
      .member_owner_segments = {},
      .boundary_ports = {}};
  const llmp::planner::Policy policy{};
  const llmp::planner::PlanOptions options{.capturable = true, .within = recipe.within};
  auto inventory = llmp::planner::BoundComparison(boundary, selector, funder, *recipe.registry,
                                                  recipe.device, policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  ASSERT_EQ(inventory->site.accepted_member_origins.size(), 2U);
  EXPECT_EQ(inventory->site.accepted_member_origins, selector.member_origins);
  ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
  EXPECT_EQ(inventory->incumbent.plan.applied[0].pattern, "rms_norm_mul");
  EXPECT_EQ(inventory->incumbent.plan.applied[0].replaced,
            (std::vector<ir::NodeId>{ir::NodeId{2}, ir::NodeId{3}}));
  auto alternative = llmp::planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto planned = llmp::planner::PlanAlternative(*inventory, std::move(*alternative),
                                                *recipe.registry, recipe.device, policy, options);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  EXPECT_TRUE(std::ranges::none_of(planned->plan.applied, [](const auto& applied) {
    return applied.pattern == "rms_norm_mul";
  }));
}

TEST_F(GgmlComparisonLowerForGpu, LocalRegionRunsItsActualPrefixAndBodyWithoutItsSuffix) {
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.role = 1};
  constexpr float kEpsilon = 1.0e-6F;
  ir::Builder builder;
  builder.SetOwner(0);
  const auto input = builder.Input(ir::Dense(kF32, ir::ShapeOf({16, 4})));
  const auto weight =
      builder.Weight(ir::Dense(kF32, ir::ShapeOf({16})), weight_ref, "region.norm.weight");
  constexpr ir::WeightRef alias_weight_ref{.role = 2};
  const auto alias_weight = builder.Weight(ir::Dense(kF32, ir::ShapeOf({16, 4})), alias_weight_ref,
                                           "region.alias.weight");
  const auto alias_product = builder.Mul(input, alias_weight);
  const auto prefix = builder.RmsNorm(input, kEpsilon);
  const auto normalized = builder.RmsNorm(prefix, kEpsilon);
  const auto product = builder.Mul(normalized, weight);
  // This terminal metadata member has no real consumer inside the region.
  // Its exact mapped output view must close without inventing another call.
  const auto scaled = builder.Reshape(product, ir::ShapeOf({4, 16}));
  const auto suffix_prefix = builder.Reshape(prefix, ir::ShapeOf({4, 16}));
  // This unrelated suffix lowers normally but is intentionally outside the
  // local RMS/Mul recipe domain.
  const auto suffix = builder.Unary(builder.Concat(scaled, suffix_prefix, 0), ir::UnaryFn::kRelu);
  builder.Output(suffix);
  builder.Output(alias_product);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);

  const auto norm_node = graph->producer(normalized);
  const auto mul_node = graph->producer(product);
  const auto output_view_node = graph->producer(scaled);
  ASSERT_TRUE(norm_node.valid());
  ASSERT_TRUE(mul_node.valid());
  ASSERT_TRUE(output_view_node.valid());
  const std::array members{norm_node, mul_node, output_view_node};
  const std::array<std::uint32_t, 3> member_offsets{0, 2, 3};
  const auto input_bytes = ir::StorageBytes(graph->tensor(input));
  const auto weight_parts = llmp::format::PartsOf(graph->tensor(weight).format);
  const auto alias_weight_parts = llmp::format::PartsOf(graph->tensor(alias_weight).format);
  ASSERT_TRUE(input_bytes && weight_parts && weight_parts->size() == 1 && alias_weight_parts &&
              alias_weight_parts->size() == 1);
  const auto weight_part = weight_parts->parts().front();
  const auto alias_weight_part = alias_weight_parts->parts().front();
  const std::array<co::StepWeight, 2> weights{
      co::StepWeight{.tensor = weight,
                     .ref = weight_ref,
                     .parts = {{"region.norm.weight", llmp::format::PartRole::kData}}},
      co::StepWeight{.tensor = alias_weight,
                     .ref = alias_weight_ref,
                     .parts = {{"region.alias.weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), weights};
  const std::array<co::ComparisonExternalBinding, 3> external{
      co::ComparisonExternalBinding{.original_root = input,
                                    .allocation_class = 1,
                                    .allocation_bytes = *input_bytes,
                                    .bytes = *input_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = weight_part.readable,
                                    .bytes = weight_part.readable,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = alias_weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 3,
                                    .allocation_bytes = alias_weight_part.readable,
                                    .bytes = alias_weight_part.readable,
                                    .alignment = 16}};
  const co::ComparisonBoundaryView source{io, {}, 0, external};
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  auto region_boundary = co::MakeComparisonRegionBoundary(
      {.source = source, .member_offsets = member_offsets, .members = members}, funder);
  ASSERT_TRUE(region_boundary) << (region_boundary ? "" : region_boundary.error().detail);

  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* pattern = planner::FindPattern("rms_norm_mul");
  ASSERT_NE(pattern, nullptr);
  const planner::ComparisonSiteSelector selector{.source_graph_identity = graph->identity(),
                                                 .kind = planner::ComparisonSiteKind::kRewrite,
                                                 .pattern = "rms_norm_mul",
                                                 .version = pattern->version,
                                                 .member_origins = {{norm_node}, {mul_node}},
                                                 .member_owner_segments = {},
                                                 .boundary_ports = {}};
  const planner::Policy policy{};
  const planner::PlanOptions options{.capturable = true, .within = recipe.within};
  auto inventory = planner::BoundComparison(source, selector, funder, *recipe.registry,
                                            recipe.device, policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  auto alternative = planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto planned = planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                          recipe.device, policy, options);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
  EXPECT_EQ(inventory->incumbent.plan.applied.front().pattern, "rms_norm_mul");
  EXPECT_TRUE(std::ranges::none_of(planned->plan.applied, [](const auto& applied) {
    return applied.pattern == "rms_norm_mul";
  }));

  constexpr std::size_t kArenaBytes = 1U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  CudaBuffer activation, weight_buffer, alias_weight_buffer, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(weight_buffer.Allocate(weight_part.readable));
  ASSERT_TRUE(alias_weight_buffer.Allocate(alias_weight_part.readable));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  std::optional<std::unique_ptr<kn::StepLaunch>> launch;
  std::optional<llmp::providers::NativeStream> native;

  std::array<std::vector<float>, 2> host_values;
  host_values[0].resize(16 * 4);
  host_values[1].resize(16);
  for (std::size_t i = 0; i < host_values[0].size(); ++i) {
    host_values[0][i] = 0.02F * static_cast<float>(static_cast<int>(i % 17) - 8);
  }
  for (std::size_t i = 0; i < host_values[1].size(); ++i)
    host_values[1][i] = 0.5F + 0.025F * static_cast<float>(i);
  const std::vector<float> alias_host_values(16 * 4, 0.75F);
  ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(weight_buffer.address()), host_values[1].data(),
                       host_values[1].size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(reinterpret_cast<void*>(alias_weight_buffer.address()), alias_host_values.data(),
                 alias_host_values.size() * sizeof(float), cudaMemcpyHostToDevice),
      cudaSuccess);

  auto reference = reference::Executor::Create(*graph);
  ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
  ASSERT_TRUE(reference->BindValues(input, host_values[0]));
  ASSERT_TRUE(reference->BindValues(weight, host_values[1]));
  ASSERT_TRUE(reference->BindValues(alias_weight, alias_host_values));
  ASSERT_TRUE(reference->Run());
  auto expected_region = reference->Values(scaled);
  auto expected_prefix = reference->Values(prefix);
  ASSERT_TRUE(expected_region && expected_prefix);

  const std::array<const planner::StepPlan*, 2> plans{&inventory->incumbent.plan, &planned->plan};
  {
    // Authenticate a real legal in-place multiply shape whose destination is
    // deliberately placed on the immutable weight bytes. Generic fused MUL
    // permits destination==weight, so only the local physical access closure
    // should reject promotion of this reachable bound program.
    const auto& plan = *plans[0];
    const auto input_root = plan.graph.root(plan.tensors.at(input.v));
    const auto alias_root = plan.graph.root(plan.tensors.at(alias_product.v));
    const auto placement = [&](ir::TensorId root) -> std::optional<std::uint64_t> {
      const auto found = std::ranges::find(plan.placement.offsets, root,
                                           &std::pair<ir::TensorId, std::uint64_t>::first);
      return found == plan.placement.offsets.end() ? std::nullopt : std::optional(found->second);
    };
    const auto input_offset = placement(input_root);
    const auto output_offset = placement(alias_root);
    ASSERT_TRUE(input_offset && output_offset);
    ASSERT_LE(*output_offset, kArenaBytes);
    ASSERT_LE(alias_weight_part.readable, kArenaBytes - *output_offset);
    std::array<co::ComparisonExternalBinding, 3> shared_backing{
        co::ComparisonExternalBinding{.original_root = input,
                                      .allocation_class = 7,
                                      .allocation_bytes = kArenaBytes,
                                      .offset = *input_offset,
                                      .bytes = *input_bytes,
                                      .alignment = 16},
        co::ComparisonExternalBinding{.original_root = weight,
                                      .part = llmp::format::PartRole::kData,
                                      .allocation_class = 2,
                                      .allocation_bytes = weight_part.readable,
                                      .bytes = weight_part.readable,
                                      .alignment = 16},
        co::ComparisonExternalBinding{.original_root = alias_weight,
                                      .part = llmp::format::PartRole::kData,
                                      .allocation_class = 7,
                                      .allocation_bytes = kArenaBytes,
                                      .offset = *output_offset,
                                      .bytes = alias_weight_part.readable,
                                      .alignment = 16}};
    const co::ComparisonBoundaryView shared_source{io, {}, 0, shared_backing};
    auto shared_base_boundary = co::MakeComparisonBoundary(shared_source, funder);
    ASSERT_TRUE(shared_base_boundary)
        << (shared_base_boundary ? "" : shared_base_boundary.error().detail);
    kn::Addresses alias_addresses{};
    alias_addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
      return ref.role == alias_weight_ref.role ? activation.address() + *output_offset
                                               : weight_buffer.address();
    };
    alias_addresses.activations = activation.address();
    alias_addresses.activation_bytes = kArenaBytes;
    alias_addresses.activation_alignment = 256;
    const std::array<kn::ComparisonBindingRange, 3> alias_ranges{
        kn::ComparisonBindingRange{.original_root = input,
                                   .allocation_class = 7,
                                   .allocation_bytes = kArenaBytes,
                                   .offset = *input_offset,
                                   .bytes = *input_bytes,
                                   .alignment = 16,
                                   .base = activation.address()},
        kn::ComparisonBindingRange{.original_root = weight,
                                   .part = llmp::format::PartRole::kData,
                                   .allocation_class = 2,
                                   .allocation_bytes = weight_part.readable,
                                   .bytes = weight_part.readable,
                                   .alignment = 16,
                                   .base = weight_buffer.address()},
        kn::ComparisonBindingRange{.original_root = alias_weight,
                                   .part = llmp::format::PartRole::kData,
                                   .allocation_class = 7,
                                   .allocation_bytes = kArenaBytes,
                                   .offset = *output_offset,
                                   .bytes = alias_weight_part.readable,
                                   .alignment = 16,
                                   .base = activation.address()}};
    kn::ComparisonRequest alias_request{.mode = kn::ComparisonRunMode::kFull,
                                        .bindings = alias_ranges,
                                        .expected_boundary = &*shared_base_boundary};
    const std::array alias_members{graph->producer(alias_product)};
    const std::array<std::uint32_t, 2> alias_member_offsets{0, 1};
    auto alias_region = co::MakeComparisonRegionBoundary(
        {.source = shared_source, .member_offsets = alias_member_offsets, .members = alias_members},
        funder);
    ASSERT_TRUE(alias_region) << (alias_region ? "" : alias_region.error().detail);
    const kn::ComparisonRegionCapture alias_capture{
        .seed = &*alias_region,
        .origins = &*inventory->incumbent.final_origins,
        .final_nodes = &inventory->incumbent.final_node_origins};
    kn::LowerInput alias_input(plan, io, alias_addresses, std::nullopt, &alias_request,
                               &alias_capture);
    auto alias_program = life_->backend->Lower(alias_input);
    ASSERT_TRUE(alias_program) << (alias_program ? "" : alias_program.error());
    const auto alias_projection =
        (*alias_program)->ProjectComparisonRegionDescriptor(*alias_region);
    EXPECT_NE(alias_projection.status, kn::ComparisonRegionStatus::kClosed)
        << "a materialized region write overlapped an immutable external weight range";
    EXPECT_EQ(alias_projection.reason,
              "actual region read/write exceeds its authenticated storage extent");
  }

  std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
  std::array<std::optional<kn::ComparisonRegionProjection>, 2> tokens;
  std::array<kn::ComparisonRegionRecipeSeal, 2> descriptor_seals{};
  std::array<std::vector<kn::ComparisonRegionValuePort>, 2> value_ports;
  for (std::size_t arm = 0; arm < plans.size(); ++arm) {
    const auto& plan = *plans[arm];
    ASSERT_LE(plan.placement.extent, kArenaBytes);
    const auto endpoint = plan.tensors.at(scaled.v);
    ASSERT_TRUE(endpoint.valid());
    EXPECT_EQ(plan.graph.node(plan.graph.producer(endpoint)).key, ir::Primitive(ir::Op::kReshape));
    const auto placement_offset = [&](ir::TensorId original) -> std::optional<std::uint64_t> {
      if (!original.valid() || original.v >= plan.tensors.size() ||
          !plan.tensors[original.v].valid())
        return {};
      const auto root = plan.graph.root(plan.tensors[original.v]);
      const auto found = std::ranges::find(plan.placement.offsets, root,
                                           &std::pair<ir::TensorId, std::uint64_t>::first);
      return found == plan.placement.offsets.end() ? std::nullopt
                                                   : std::optional<std::uint64_t>(found->second);
    };
    kn::Addresses addresses{};
    addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
      return ref.role == alias_weight_ref.role ? alias_weight_buffer.address()
                                               : weight_buffer.address();
    };
    addresses.activations = activation.address();
    addresses.activation_bytes = kArenaBytes;
    addresses.activation_alignment = 256;
    std::vector<kn::ComparisonBindingRange> ranges;
    for (const auto& fact : inventory->source_boundary.external()) {
      ASSERT_LT(fact.canonical_root, inventory->source_boundary.roots().size());
      const auto root = inventory->source_boundary.roots()[fact.canonical_root].original_root;
      std::uint64_t base = weight_buffer.address();
      if (!fact.part) {
        const auto offset = placement_offset(root);
        ASSERT_TRUE(offset);
        base = activation.address() + *offset;
      } else if (root == alias_weight) {
        base = alias_weight_buffer.address();
      }
      ranges.push_back({.original_root = root,
                        .part = fact.part,
                        .allocation_class = fact.canonical_allocation_class,
                        .allocation_bytes = fact.allocation_bytes,
                        .offset = fact.offset,
                        .bytes = fact.bytes,
                        .alignment = fact.alignment,
                        .base = base,
                        .resource_part_identity = fact.resource_part_identity});
    }
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .bindings = ranges,
                                  .expected_boundary = &inventory->source_boundary};
    ASSERT_TRUE(arm == 0 ? inventory->incumbent.final_origins.has_value()
                         : planned->final_origins.has_value());
    auto* final_origins =
        arm == 0 ? &*inventory->incumbent.final_origins : &*planned->final_origins;
    const auto* final_nodes =
        arm == 0 ? &inventory->incumbent.final_node_origins : &planned->final_node_origins;
    const kn::ComparisonRegionCapture capture{
        .seed = &*region_boundary, .origins = final_origins, .final_nodes = final_nodes};
    kn::LowerInput lower_input(plan, io, addresses, std::nullopt, &request, &capture);
    auto program = life_->backend->Lower(lower_input);
    ASSERT_TRUE(program) << (program ? "" : program.error());
    const auto whole_before_bind = (*program)->comparison_recipe();
    ASSERT_TRUE(whole_before_bind);
    ASSERT_EQ(whole_before_bind->state, kn::ComparisonProofState::kUnproved);
    EXPECT_TRUE(whole_before_bind->missing == kn::ComparisonMissing::kOperationSchema ||
                whole_before_bind->missing == kn::ComparisonMissing::kPhysicalClosure);
    const auto descriptor = (*program)->ProjectComparisonRegionDescriptor(*region_boundary);
    ASSERT_EQ(descriptor.status, kn::ComparisonRegionStatus::kClosed) << descriptor.reason;
    ASSERT_EQ(descriptor.seal.state, kn::ComparisonProofState::kDescriptorVerified);
    ASSERT_EQ(descriptor.seal.version, 3U);
    EXPECT_TRUE(descriptor.required_members.empty());
    EXPECT_NE(descriptor.seal.backend_device, llmp::base::Sha256Digest{});
    const auto certificate_descriptor = (*program)->ProjectComparisonRegionCertificate(
        *region_boundary, kn::ComparisonCertificatePhase::kDescriptor);
    ASSERT_EQ(certificate_descriptor.status, kn::ComparisonRegionStatus::kClosed)
        << certificate_descriptor.reason;
    ASSERT_EQ(certificate_descriptor.seal.state, kn::ComparisonProofState::kDescriptorVerified);
    EXPECT_TRUE(certificate_descriptor.required_members.empty());
    EXPECT_NE((*program)->ProjectComparisonRegion(*region_boundary).status,
              kn::ComparisonRegionStatus::kClosed)
        << "a production descriptor certificate must not mint executable authority";
    descriptor_seals[arm] = descriptor.seal;
    if (!launch) {
      auto made_launch =
          life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                                 {.base = persistent.address(), .bytes = persistent_bytes});
      ASSERT_TRUE(made_launch) << (made_launch ? "" : made_launch.error());
      launch.emplace(std::move(*made_launch));
      auto made_submission = life_->node.execution().Submission(life_->node.stream(0));
      ASSERT_TRUE(made_submission) << (made_submission ? "" : made_submission.error().detail);
      native.emplace(std::move(*made_submission));
    }
    ASSERT_TRUE((*program)->Bind(**launch));
    const auto certificate_bound = (*program)->ProjectComparisonRegionCertificate(
        *region_boundary, kn::ComparisonCertificatePhase::kBound);
    ASSERT_EQ(certificate_bound.status, kn::ComparisonRegionStatus::kClosed)
        << certificate_bound.reason;
    ASSERT_EQ(certificate_bound.seal.state, kn::ComparisonProofState::kBoundVerified);
    const auto& certificate_seal = certificate_bound.seal;
    const auto& descriptor_certificate_seal = certificate_descriptor.seal;
    EXPECT_EQ(certificate_seal.version, descriptor_certificate_seal.version);
    EXPECT_EQ(certificate_seal.subject, descriptor_certificate_seal.subject);
    EXPECT_EQ(certificate_seal.boundary, descriptor_certificate_seal.boundary);
    EXPECT_EQ(certificate_seal.planned_region, descriptor_certificate_seal.planned_region);
    EXPECT_EQ(certificate_seal.physical_recipe, descriptor_certificate_seal.physical_recipe);
    EXPECT_EQ(certificate_seal.policy, descriptor_certificate_seal.policy);
    EXPECT_EQ(certificate_seal.registry, descriptor_certificate_seal.registry);
    EXPECT_EQ(certificate_seal.backend_recipe, descriptor_certificate_seal.backend_recipe);
    EXPECT_EQ(certificate_seal.backend_build, descriptor_certificate_seal.backend_build);
    EXPECT_EQ(certificate_seal.backend_selection, descriptor_certificate_seal.backend_selection);
    EXPECT_EQ(certificate_seal.backend_device, descriptor_certificate_seal.backend_device);
    const auto plan_identity = (*program)->comparison_plan_identity();
    ASSERT_TRUE(plan_identity);
    EXPECT_EQ(plan_identity->source_graph_identity, graph->identity());
    EXPECT_NE(plan_identity->planned_graph_identity, 0U);
    auto projected = (*program)->ProjectComparisonRegion(*region_boundary);
    ASSERT_EQ(projected.status, kn::ComparisonRegionStatus::kClosed) << projected.reason;
    ASSERT_TRUE(projected.projection);
    ASSERT_EQ(projected.projection->seal().state, kn::ComparisonProofState::kBoundVerified);
    ASSERT_EQ(projected.projection->seal().missing, kn::ComparisonMissing::kNone);
    const auto& descriptor_seal = descriptor_seals[arm];
    const auto& bound_seal = projected.projection->seal();
    EXPECT_EQ(bound_seal.version, descriptor_seal.version);
    EXPECT_EQ(bound_seal.subject, descriptor_seal.subject);
    EXPECT_EQ(bound_seal.boundary, descriptor_seal.boundary);
    EXPECT_EQ(bound_seal.planned_region, descriptor_seal.planned_region);
    EXPECT_EQ(bound_seal.physical_recipe, descriptor_seal.physical_recipe);
    EXPECT_EQ(bound_seal.policy, descriptor_seal.policy);
    EXPECT_EQ(bound_seal.registry, descriptor_seal.registry);
    EXPECT_EQ(bound_seal.backend_recipe, descriptor_seal.backend_recipe);
    EXPECT_EQ(bound_seal.backend_build, descriptor_seal.backend_build);
    EXPECT_EQ(bound_seal.backend_selection, descriptor_seal.backend_selection);
    EXPECT_EQ(bound_seal.backend_device, descriptor_seal.backend_device);
    EXPECT_TRUE((*program)->ValidateComparisonProjection(*projected.projection));
    value_ports[arm].assign(projected.value_ports.begin(), projected.value_ports.end());
    tokens[arm].emplace(std::move(*projected.projection));
    programs[arm] = std::move(*program);
  }
  EXPECT_NE(tokens[0]->seal().physical_recipe, tokens[1]->seal().physical_recipe);
  ASSERT_EQ(value_ports[0].size(), region_boundary->ports().size());
  ASSERT_EQ(value_ports[1].size(), region_boundary->ports().size());
  for (std::size_t port_index = 0; port_index < region_boundary->ports().size(); ++port_index) {
    const auto& expected = region_boundary->ports()[port_index];
    for (const auto& ports : value_ports) {
      EXPECT_EQ(ports[port_index].ordinal, port_index);
      EXPECT_EQ(ports[port_index].source_tensor, expected.original_tensor);
      EXPECT_EQ(ports[port_index].kind == kn::ComparisonRegionValueKind::kInput,
                expected.kind == co::ComparisonRegionPortKind::kInput);
    }
  }

  std::array<std::optional<kn::ComparisonRegionProjection>, 2> stale_tokens;
  for (std::size_t arm = 0; arm < plans.size(); ++arm) {
    stale_tokens[arm].emplace(std::move(*tokens[arm]));
    EXPECT_FALSE(programs[arm]->ValidateComparisonProjection(*tokens[arm]));
    auto refreshed = programs[arm]->ProjectComparisonRegion(*region_boundary);
    ASSERT_EQ(refreshed.status, kn::ComparisonRegionStatus::kClosed) << refreshed.reason;
    ASSERT_TRUE(refreshed.projection);
    ASSERT_EQ(refreshed.value_ports.size(), value_ports[arm].size());
    EXPECT_FALSE(programs[arm]->ValidateComparisonProjection(*stale_tokens[arm]));
    EXPECT_FALSE(programs[arm]->RunComparisonRegion(*stale_tokens[arm], **launch, *native));
    tokens[arm].reset();
    tokens[arm].emplace(std::move(*refreshed.projection));
    EXPECT_TRUE(programs[arm]->ValidateComparisonProjection(*tokens[arm]));
  }
  EXPECT_FALSE(programs[1]->ValidateComparisonProjection(*stale_tokens[0]));
  EXPECT_FALSE(programs[1]->RunComparisonRegion(*stale_tokens[0], **launch, *native));
  EXPECT_FALSE(programs[0]->ValidateComparisonProjection(*tokens[1]));

  const auto original_boundary_digest = tokens[0]->seal().boundary;
  auto relabelled_external = external;
  relabelled_external[0].allocation_class += 17;
  relabelled_external[1].allocation_class += 29;
  const auto make_region = [&](std::uint16_t stage,
                               std::span<const co::ComparisonExternalBinding> facts) {
    const co::ComparisonBoundaryView view{io, {}, stage, facts};
    return co::MakeComparisonRegionBoundary(
        {.source = view, .member_offsets = member_offsets, .members = members}, funder);
  };
  auto relabelled_boundary = make_region(0, relabelled_external);
  ASSERT_TRUE(relabelled_boundary)
      << (relabelled_boundary ? "" : relabelled_boundary.error().detail);
  auto relabelled_projection = programs[0]->ProjectComparisonRegion(*relabelled_boundary);
  ASSERT_EQ(relabelled_projection.status, kn::ComparisonRegionStatus::kClosed)
      << relabelled_projection.reason;
  ASSERT_TRUE(relabelled_projection.projection);
  EXPECT_EQ(relabelled_projection.projection->seal().boundary, original_boundary_digest);

  // The original input feeds only the checked prefix, not this local RMS/Mul
  // body. Changing its external alignment contract must not perturb the
  // body's canonical boundary or either local recipe identity.
  auto unrelated_alignment = external;
  unrelated_alignment[0].alignment = 32;
  auto unrelated_alignment_boundary = make_region(0, unrelated_alignment);
  ASSERT_TRUE(unrelated_alignment_boundary)
      << (unrelated_alignment_boundary ? "" : unrelated_alignment_boundary.error().detail);
  auto unrelated_alignment_projection =
      programs[0]->ProjectComparisonRegion(*unrelated_alignment_boundary);
  ASSERT_EQ(unrelated_alignment_projection.status, kn::ComparisonRegionStatus::kClosed)
      << unrelated_alignment_projection.reason;
  ASSERT_TRUE(unrelated_alignment_projection.projection);
  EXPECT_EQ(unrelated_alignment_projection.projection->seal().boundary, original_boundary_digest);
  EXPECT_EQ(unrelated_alignment_projection.projection->seal().planned_region,
            tokens[0]->seal().planned_region);
  EXPECT_EQ(unrelated_alignment_projection.projection->seal().physical_recipe,
            tokens[0]->seal().physical_recipe);

  auto wrong_stage = make_region(1, external);
  ASSERT_TRUE(wrong_stage) << (wrong_stage ? "" : wrong_stage.error().detail);
  auto stage_projection = programs[0]->ProjectComparisonRegion(*wrong_stage);
  EXPECT_NE(stage_projection.status, kn::ComparisonRegionStatus::kClosed);

  auto changed_external = external;
  changed_external[1].alignment = 32;
  auto changed_boundary = make_region(0, changed_external);
  ASSERT_TRUE(changed_boundary) << (changed_boundary ? "" : changed_boundary.error().detail);
  auto changed_projection = programs[0]->ProjectComparisonRegion(*changed_boundary);
  EXPECT_NE(changed_projection.status, kn::ComparisonRegionStatus::kClosed);

  auto changed_alias = external;
  changed_alias[1].allocation_class = changed_alias[0].allocation_class;
  changed_alias[1].allocation_bytes = changed_alias[0].allocation_bytes;
  changed_alias[1].offset = changed_alias[0].offset;
  auto changed_alias_boundary = make_region(0, changed_alias);
  ASSERT_TRUE(changed_alias_boundary)
      << (changed_alias_boundary ? "" : changed_alias_boundary.error().detail);
  auto alias_projection = programs[0]->ProjectComparisonRegion(*changed_alias_boundary);
  EXPECT_NE(alias_projection.status, kn::ComparisonRegionStatus::kClosed);

  // A failed projection still advances the program-minted serial. Acquire
  // fresh tokens for both arms before any execution or capture.
  for (std::size_t arm = 0; arm < plans.size(); ++arm) {
    tokens[arm].reset();
    auto refreshed = programs[arm]->ProjectComparisonRegion(*region_boundary);
    ASSERT_EQ(refreshed.status, kn::ComparisonRegionStatus::kClosed) << refreshed.reason;
    ASSERT_TRUE(refreshed.projection);
    tokens[arm].emplace(std::move(*refreshed.projection));
  }

  const auto find_slot = [](auto slots, ir::TensorId tensor) {
    using Slot = std::remove_cvref_t<decltype(slots.front())>;
    const auto found = std::ranges::find(slots, tensor, &Slot::tensor);
    return found == slots.end() ? static_cast<const Slot*>(nullptr) : &*found;
  };
  const auto compare_prefix = [&](std::size_t arm, std::span<const float> expected_values) {
    const auto prefix_port = std::ranges::find_if(value_ports[arm], [&](const auto& port) {
      return port.kind == kn::ComparisonRegionValueKind::kInput && port.source_tensor == prefix;
    });
    ASSERT_NE(prefix_port, value_ports[arm].end());
    ASSERT_EQ(prefix_port->bytes, expected_values.size_bytes());
    std::vector<float> actual(expected_values.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(prefix_port->address),
                         prefix_port->bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (std::size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i]) && std::isfinite(expected_values[i])) << i;
      EXPECT_LT(std::abs(static_cast<double>(actual[i]) - expected_values[i]) /
                    (1.0 + std::abs(static_cast<double>(expected_values[i]))),
                1.0e-5)
          << "prefix output element " << i;
    }
  };
  const auto compare_region = [&](std::size_t arm, std::span<const float> expected_values) {
    const auto output_port = std::ranges::find_if(value_ports[arm], [&](const auto& port) {
      return port.kind == kn::ComparisonRegionValueKind::kOutput && port.source_tensor == scaled;
    });
    ASSERT_NE(output_port, value_ports[arm].end());
    ASSERT_EQ(output_port->bytes, expected_values.size_bytes());
    std::vector<float> actual(expected_values.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(output_port->address),
                         output_port->bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (std::size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i]) && std::isfinite(expected_values[i])) << i;
      // Match ggml_kernels_test's fused and primitive RMSNormMul reference
      // check: worst abs(error)/(1 + abs(reference)) must stay below 1e-5.
      EXPECT_LT(std::abs(static_cast<double>(actual[i]) - expected_values[i]) /
                    (1.0 + std::abs(static_cast<double>(expected_values[i]))),
                1.0e-5)
          << "region output element " << i;
    }
  };

  std::array<std::unique_ptr<kn::Recorded>, 2> recordings;
  const auto original_input = host_values[0];
  for (std::size_t arm = 0; arm < plans.size(); ++arm) {
    const auto& plan = *plans[arm];
    const auto mapped_input = plan.tensors.at(input.v);
    ASSERT_TRUE(mapped_input.valid());
    // Public slots are keyed by the original GraphBuildIO tensor IDs. The
    // plan mapping is private to lowering and must not be used as a slot key.
    const auto* input_slot = find_slot(programs[arm]->inputs(), input);
    ASSERT_NE(input_slot, nullptr);
    ASSERT_NE(input_slot->device_address, 0U);
    ASSERT_EQ(input_slot->bytes, host_values[0].size() * sizeof(float));
    const auto stage_input = [&](std::span<const float> values) {
      return cudaMemcpy(reinterpret_cast<void*>(input_slot->device_address), values.data(),
                        values.size_bytes(), cudaMemcpyHostToDevice) == cudaSuccess;
    };
    const auto execute_prefix = [&]() {
      return programs[arm]->RunComparisonPrefix(*tokens[arm], **launch, *native);
    };
    host_values[0] = original_input;
    ASSERT_TRUE(reference->BindValues(input, host_values[0]));
    ASSERT_TRUE(reference->Run());
    expected_region = reference->Values(scaled);
    expected_prefix = reference->Values(prefix);
    ASSERT_TRUE(expected_region && expected_prefix);
    for (std::size_t frame = 0; frame < 3; ++frame) {
      if (frame != 0) {
        for (std::size_t i = 0; i < host_values[0].size(); ++i)
          host_values[0][i] += 0.003F * static_cast<float>(static_cast<int>(i % 7) - 3);
        ASSERT_TRUE(reference->BindValues(input, host_values[0]));
        ASSERT_TRUE(reference->Run());
        expected_region = reference->Values(scaled);
        expected_prefix = reference->Values(prefix);
        ASSERT_TRUE(expected_region && expected_prefix);
      }
      ASSERT_TRUE(stage_input(host_values[0]));
      ASSERT_TRUE(execute_prefix());
      ASSERT_EQ(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(native->handle)), cudaSuccess);
      compare_prefix(arm, *expected_prefix);

      if (frame == 0) {
        ASSERT_TRUE(programs[arm]->RunComparisonRegion(*tokens[arm], **launch, *native));
      } else if (frame == 1) {
        auto record =
            programs[arm]->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
              return programs[arm]->RunComparisonRegion(*tokens[arm], **launch, *native);
            });
        ASSERT_TRUE(record) << (record ? "" : record.error().detail);
        // The source-registered fused recipe emits one CUDA kernel. The
        // source-registered unfused wrapper emits RMSNorm then Mul: two.
        // The captured body itself therefore detects an accidental suffix.
        ASSERT_EQ((*record)->nodes(), arm == 0 ? 1U : 2U);
        recordings[arm] = std::move(*record);
        ASSERT_TRUE((*launch)->Launch(*native, *recordings[arm]));
      } else {
        ASSERT_TRUE(recordings[arm]);
        ASSERT_TRUE((*launch)->Launch(*native, *recordings[arm]));
      }
      ASSERT_EQ(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(native->handle)), cudaSuccess);
      compare_region(arm, *expected_region);
    }
  }
  for (auto& recording : recordings) recording.reset();
  for (auto& program : programs) program.reset();
  launch->reset();
  life_->backend->LaunchReleased();
}

TEST_F(GgmlComparisonLowerForGpu, RepeatedEdgesProjectBeyondUniqueTensorCapacity) {
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  constexpr float kEpsilon = 1.0e-6F;
  ir::Builder builder;
  builder.SetOwner(0);
  const auto input = builder.Input(ir::Dense(kF32, ir::ShapeOf({16, 4})));
  const auto weight =
      builder.Weight(ir::Dense(kF32, ir::ShapeOf({16})), {.role = 1}, "region.occurrence.weight");
  const auto normalized = builder.RmsNorm(input, kEpsilon);
  const auto scaled = builder.Mul(normalized, weight);
  builder.Output(scaled);
  std::array<ir::TensorId, 6> products{};
  for (auto& product : products) {
    product = builder.Mul(input, weight);
    builder.Output(product);
  }
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  const auto input_bytes = ir::StorageBytes(graph->tensor(input));
  const auto weight_parts = llmp::format::PartsOf(graph->tensor(weight).format);
  ASSERT_TRUE(input_bytes && weight_parts && weight_parts->size() == 1);
  const auto weight_part = weight_parts->parts().front();
  const std::array<co::StepWeight, 1> weights{
      co::StepWeight{.tensor = weight,
                     .ref = {.role = 1},
                     .parts = {{"region.occurrence.weight", llmp::format::PartRole::kData}}}};
  const std::array<co::ComparisonExternalBinding, 2> external{
      co::ComparisonExternalBinding{.original_root = input,
                                    .allocation_class = 1,
                                    .allocation_bytes = *input_bytes,
                                    .bytes = *input_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = weight_part.readable,
                                    .bytes = weight_part.readable,
                                    .alignment = 16}};
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), weights};
  const co::ComparisonBoundaryView source{io, {}, 0, external};
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* pattern = planner::FindPattern("rms_norm_mul");
  ASSERT_NE(pattern, nullptr);
  const auto norm_node = graph->producer(normalized);
  const auto norm_mul_node = graph->producer(scaled);
  ASSERT_TRUE(norm_node.valid() && norm_mul_node.valid());
  const planner::ComparisonSiteSelector selector{.source_graph_identity = graph->identity(),
                                                 .kind = planner::ComparisonSiteKind::kRewrite,
                                                 .pattern = "rms_norm_mul",
                                                 .version = pattern->version,
                                                 .member_origins = {{norm_node}, {norm_mul_node}},
                                                 .member_owner_segments = {},
                                                 .boundary_ports = {}};
  const planner::Policy policy{};
  const planner::PlanOptions options{.capturable = true, .within = recipe.within};
  auto inventory = planner::BoundComparison(source, selector, funder, *recipe.registry,
                                            recipe.device, policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);

  std::array<ir::NodeId, 8> members{norm_node, norm_mul_node};
  for (std::size_t i = 0; i < products.size(); ++i) {
    members[i + 2] = graph->producer(products[i]);
    ASSERT_TRUE(members[i + 2].valid());
  }
  const std::array<std::uint32_t, 2> member_offsets{0, 8};
  auto region_boundary = co::MakeComparisonRegionBoundary(
      {.source = source, .member_offsets = member_offsets, .members = members}, funder);
  ASSERT_TRUE(region_boundary) << (region_boundary ? "" : region_boundary.error().detail);
  ASSERT_EQ(graph->tensor_count(), 10U);
  ASSERT_EQ(region_boundary->ports().size(), 21U);
  ASSERT_GT(region_boundary->ports().size(), 2U * graph->tensor_count());

  constexpr std::size_t kArenaBytes = 1U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  CudaBuffer activation, weight_buffer, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(weight_buffer.Allocate(weight_part.readable));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());

  kn::Addresses addresses{};
  addresses.weight = [&](const ir::WeightRef&, llmp::format::PartRole) {
    return weight_buffer.address();
  };
  addresses.activations = activation.address();
  addresses.activation_bytes = kArenaBytes;
  addresses.activation_alignment = 256;
  const auto& plan = inventory->incumbent.plan;
  const auto input_planned = plan.tensors.at(input.v);
  ASSERT_TRUE(input_planned.valid());
  const auto input_root = plan.graph.root(input_planned);
  const auto input_offset = std::ranges::find(plan.placement.offsets, input_root,
                                              &std::pair<ir::TensorId, std::uint64_t>::first);
  ASSERT_NE(input_offset, plan.placement.offsets.end());
  std::array<kn::ComparisonBindingRange, 2> ranges{
      kn::ComparisonBindingRange{.original_root = input,
                                 .allocation_class = 1,
                                 .allocation_bytes = *input_bytes,
                                 .bytes = *input_bytes,
                                 .alignment = 16,
                                 .base = activation.address() + input_offset->second},
      kn::ComparisonBindingRange{.original_root = weight,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 2,
                                 .allocation_bytes = weight_part.readable,
                                 .bytes = weight_part.readable,
                                 .alignment = 16,
                                 .base = weight_buffer.address()}};
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                .bindings = ranges,
                                .expected_boundary = &inventory->source_boundary};
  const kn::ComparisonRegionCapture capture{
      .seed = &*region_boundary,
      .origins = &*inventory->incumbent.final_origins,
      .final_nodes = &inventory->incumbent.final_node_origins};
  kn::LowerInput lower_input(plan, io, addresses, std::nullopt, &request, &capture);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  ASSERT_TRUE((*program)->Bind(**launch));
  auto projected = (*program)->ProjectComparisonRegion(*region_boundary);
  ASSERT_EQ(projected.status, kn::ComparisonRegionStatus::kClosed) << projected.reason;
  ASSERT_TRUE(projected.projection);
  EXPECT_EQ(projected.value_ports.size(), 21U);
  for (std::size_t i = 0; i < projected.value_ports.size(); ++i) {
    EXPECT_EQ(projected.value_ports[i].ordinal, i);
    EXPECT_EQ(projected.value_ports[i].source_tensor, region_boundary->ports()[i].original_tensor);
    EXPECT_EQ(projected.value_ports[i].kind == kn::ComparisonRegionValueKind::kInput,
              region_boundary->ports()[i].kind == co::ComparisonRegionPortKind::kInput);
  }
  auto aliased_external = external;
  aliased_external[1].allocation_class = aliased_external[0].allocation_class;
  aliased_external[1].allocation_bytes = aliased_external[0].allocation_bytes;
  aliased_external[1].offset = aliased_external[0].offset;
  const co::ComparisonBoundaryView aliased_source{io, {}, 0, aliased_external};
  auto aliased_region = co::MakeComparisonRegionBoundary(
      {.source = aliased_source, .member_offsets = member_offsets, .members = members}, funder);
  ASSERT_TRUE(aliased_region) << (aliased_region ? "" : aliased_region.error().detail);
  const auto aliased_projection = (*program)->ProjectComparisonRegion(*aliased_region);
  EXPECT_NE(aliased_projection.status, kn::ComparisonRegionStatus::kClosed)
      << "a changed external alias partition must not reuse the authentic binding seal";
}

TEST_F(GgmlComparisonLowerForGpu, TypedNormGateHelperCanPromoteItsClosedDescriptorRecipe) {
  constexpr auto kFormat = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.role = 1};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto rows = builder.Input(ir::Dense(kFormat, ir::ShapeOf({128, 1, 1, 1})));
  const auto weight = builder.Weight(ir::Dense(kFormat, ir::ShapeOf({128})), weight_ref);
  const auto gate = builder.Input(ir::Dense(kFormat, ir::ShapeOf({128, 1, 1, 1})));
  const std::array sources{rows, weight, gate};
  const auto result =
      builder.Fused(ir::FusedKind::kNormGate, sources,
                    ir::NormGateParams{.eps = 1.0e-6F, .gate = ir::NormGateFn::kSilu})[0];
  builder.Output(result);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures},
                 {.tensor = gate, .kind = co::InputKind::kFeatures}};
  step.weights = {co::StepWeight{
      .tensor = weight, .ref = weight_ref, .parts = {{"weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io = co::GraphIO(step);
  kn::Addresses addresses{};
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  auto plan = en::PlanComponentGraph(io, recipe, {}, addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());

  const auto row_bytes = ir::StorageBytes(step.graph.tensor(rows));
  const auto gate_bytes = ir::StorageBytes(step.graph.tensor(gate));
  const auto weight_bytes = ir::StorageBytes(step.graph.tensor(weight));
  ASSERT_TRUE(row_bytes && gate_bytes && weight_bytes);
  const std::array<kn::ComparisonBindingRange, 3> bindings{
      kn::ComparisonBindingRange{.original_root = rows,
                                 .allocation_class = 1,
                                 .allocation_bytes = *row_bytes,
                                 .bytes = *row_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = weight,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 2,
                                 .allocation_bytes = *weight_bytes,
                                 .bytes = *weight_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = gate,
                                 .allocation_class = 3,
                                 .allocation_bytes = *gate_bytes,
                                 .bytes = *gate_bytes,
                                 .alignment = 16}};
  const std::array<co::ComparisonExternalBinding, 3> external{
      co::ComparisonExternalBinding{.original_root = rows,
                                    .allocation_class = 1,
                                    .allocation_bytes = *row_bytes,
                                    .bytes = *row_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = *weight_bytes,
                                    .bytes = *weight_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = gate,
                                    .allocation_class = 3,
                                    .allocation_bytes = *gate_bytes,
                                    .bytes = *gate_bytes,
                                    .alignment = 16}};
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, external},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                .bindings = bindings,
                                .state = step.state,
                                .active_stage = 0,
                                .expected_boundary = &*boundary};
  const llmp::planner::StepShape shape{.stage = 0,
                                       .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  kn::LowerInput lower_input(*plan, step, shape, addresses, 0, {}, &request);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  const auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
}

TEST_F(GgmlComparisonLowerForGpu, ExactNormGateAlternativeSealsBothBoundArms) {
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.role = 1};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto rows = builder.Input(ir::Dense(kF32, ir::ShapeOf({128, 1})));
  const auto weight =
      builder.Weight(ir::Dense(kF32, ir::ShapeOf({128})), weight_ref, "norm.weight");
  const auto gate = builder.Input(ir::Dense(kF32, ir::ShapeOf({128, 1})));
  builder.Keep(rows);
  builder.Keep(gate);
  const std::array sources{rows, weight, gate};
  builder.Output(
      builder.Fused(ir::FusedKind::kNormGate, sources,
                    ir::NormGateParams{.eps = 1.0e-6F, .gate = ir::NormGateFn::kSigmoid})[0]);
  auto fused = std::move(builder).Finish();
  ASSERT_TRUE(fused);
  auto graph = ir::Expand(*fused, false);
  ASSERT_TRUE(graph);
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures},
                 {.tensor = gate, .kind = co::InputKind::kFeatures}};
  step.weights = {co::StepWeight{.tensor = weight,
                                 .ref = weight_ref,
                                 .parts = {{"norm.weight", llmp::format::PartRole::kData}}}};
  const auto io = co::GraphIO(step);

  const auto input_bytes = ir::StorageBytes(step.graph.tensor(rows));
  const auto gate_bytes = ir::StorageBytes(step.graph.tensor(gate));
  const auto parts = llmp::format::PartsOf(step.graph.tensor(weight).format);
  ASSERT_TRUE(input_bytes && gate_bytes && parts && parts->size() == 1);
  const std::array external{
      co::ComparisonExternalBinding{.original_root = rows,
                                    .allocation_class = 1,
                                    .allocation_bytes = *input_bytes,
                                    .bytes = *input_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = parts->parts()[0].readable,
                                    .bytes = parts->parts()[0].readable,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = gate,
                                    .allocation_class = 3,
                                    .allocation_bytes = *gate_bytes,
                                    .bytes = *gate_bytes,
                                    .alignment = 16}};
  const co::ComparisonBoundaryView view{io, {}, 0, external};
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* pattern = llmp::planner::FindPattern("gdn_norm_gate");
  ASSERT_NE(pattern, nullptr);
  std::vector<ir::NodeId> origins;
  std::size_t norms = 0, muls = 0, unaries = 0;
  for (std::uint32_t i = 0; i < step.graph.nodes().size(); ++i) {
    const auto op = step.graph.node(ir::NodeId{i}).key.op;
    if (op == ir::Op::kRmsNorm) {
      ++norms;
      origins.emplace_back(i);
    }
    if (op == ir::Op::kMul) {
      ++muls;
      origins.emplace_back(i);
    }
    if (op == ir::Op::kUnary) {
      ++unaries;
      origins.emplace_back(i);
    }
  }
  ASSERT_EQ(norms, 1U);
  ASSERT_EQ(muls, 2U);
  ASSERT_EQ(unaries, 1U);
  ASSERT_EQ(origins.size(), 4U);
  std::vector<llmp::planner::ComparisonOriginSet> members;
  for (const auto origin : origins) members.push_back({origin});
  const llmp::planner::ComparisonSiteSelector selector{
      .source_graph_identity = step.graph.identity(),
      .kind = llmp::planner::ComparisonSiteKind::kRewrite,
      .pattern = "gdn_norm_gate",
      .version = pattern->version,
      .member_origins = std::move(members),
      .member_owner_segments = {},
      .boundary_ports = {}};
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  const llmp::planner::Policy policy{};
  const llmp::planner::PlanOptions options{.capturable = true, .within = recipe.within};
  auto inventory = llmp::planner::BoundComparison(view, selector, funder, *recipe.registry,
                                                  recipe.device, policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  auto alternative = llmp::planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto planned = llmp::planner::PlanAlternative(*inventory, std::move(*alternative),
                                                *recipe.registry, recipe.device, policy, options);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
  EXPECT_EQ(inventory->incumbent.plan.applied[0].pattern, "gdn_norm_gate");
  EXPECT_TRUE(std::ranges::none_of(planned->plan.applied, [](const auto& applied) {
    return applied.pattern == "gdn_norm_gate";
  }));

  constexpr std::size_t kArenaBytes = 1U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  CudaBuffer activation, weight_buffer, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(weight_buffer.Allocate(parts->parts()[0].readable));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  const std::array<const llmp::planner::StepPlan*, 2> plans{&inventory->incumbent.plan,
                                                            &planned->plan};
  const llmp::planner::StepShape shape{.stage = 0,
                                       .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  for (const auto* plan : plans) {
    ASSERT_LE(plan->placement.extent, kArenaBytes);
    kn::Addresses addresses{
        .weight = [&](const ir::WeightRef&,
                      llmp::format::PartRole) { return weight_buffer.address(); },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = activation.address(),
        .activation_bytes = kArenaBytes,
        .activation_alignment = 256};
    std::vector<kn::ComparisonBindingRange> ranges;
    for (const auto& fact : inventory->source_boundary.external()) {
      ASSERT_LT(fact.canonical_root, inventory->source_boundary.roots().size());
      const auto original_root =
          inventory->source_boundary.roots()[fact.canonical_root].original_root;
      const auto mapped = fact.part ? ir::TensorId{} : plan->tensors[original_root.v];
      std::uint64_t base = weight_buffer.address();
      if (!fact.part) {
        ASSERT_TRUE(mapped.valid());
        const auto root = plan->graph.root(mapped);
        const auto placed = std::ranges::find_if(plan->placement.offsets, [&](const auto& entry) {
          return plan->graph.root(entry.first) == root;
        });
        ASSERT_NE(placed, plan->placement.offsets.end());
        base = activation.address() + placed->second;
      }
      ranges.push_back({.original_root = original_root,
                        .part = fact.part,
                        .allocation_class = fact.canonical_allocation_class,
                        .allocation_bytes = fact.allocation_bytes,
                        .offset = fact.offset,
                        .bytes = fact.bytes,
                        .alignment = fact.alignment,
                        .base = base,
                        .resource_part_identity = fact.resource_part_identity});
    }
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .bindings = ranges,
                                  .expected_boundary = &inventory->source_boundary};
    kn::LowerInput lower_input(*plan, step, shape, addresses, 0, {}, &request);
    auto program = life_->backend->Lower(lower_input);
    ASSERT_TRUE(program) << (program ? "" : program.error());
    auto seal = (*program)->comparison_recipe();
    ASSERT_TRUE(seal);
    ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
    ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
    ASSERT_TRUE((*program)->Bind(**launch));
    seal = (*program)->comparison_recipe();
    ASSERT_TRUE(seal);
    EXPECT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified);
    EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
  }
  life_->backend->LaunchReleased();
}

TEST_F(GgmlComparisonLowerForGpu, LocalNormGateSigmoidClosesBothSchedulesAndMatchesReference) {
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.stage = 0, .component = 17, .role = 1, .layer = 0};
  constexpr std::uint32_t kHeads = 48;
  constexpr std::uint32_t kWidth = 128;
  constexpr float kEpsilon = 1.0e-6F;
  ir::Builder builder;
  builder.SetComponent(17);
  builder.SetOwner(0);
  const auto rows = builder.Input(ir::Dense(kF32, ir::ShapeOf({kWidth, kHeads, 1, 1})));
  builder.SetJoined();
  const auto weight =
      builder.Weight(ir::Dense(kF32, ir::ShapeOf({kWidth})), weight_ref, "norm.weight");
  const auto gate = builder.Input(ir::Dense(kF32, ir::ShapeOf({kWidth, kHeads, 1, 1})));
  const std::array sources{rows, weight, gate};
  const auto gated =
      builder.Fused(ir::FusedKind::kNormGate, sources,
                    ir::NormGateParams{.eps = kEpsilon, .gate = ir::NormGateFn::kSigmoid})[0];
  builder.Output(builder.Reshape(gated, ir::ShapeOf({kWidth * kHeads, 1})));
  auto fused = std::move(builder).Finish();
  ASSERT_TRUE(fused) << (fused ? "" : fused.error().detail);
  auto graph = ir::Expand(*fused, false);
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);

  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures, .segment = 0, .component = 17},
                 {.tensor = gate, .kind = co::InputKind::kFeatures, .component = 17}};
  step.weights = {co::StepWeight{.tensor = weight,
                                 .ref = weight_ref,
                                 .parts = {{"norm.weight", llmp::format::PartRole::kData}}}};
  ASSERT_EQ(step.graph.outputs().size(), 1U);
  step.logits = step.graph.outputs().front();
  const auto io = co::GraphIO(step);
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* pattern = planner::FindPattern("gdn_norm_gate");
  ASSERT_NE(pattern, nullptr);
  std::vector<ir::NodeId> origins;
  for (std::uint32_t i = 0; i < step.graph.nodes().size(); ++i) {
    const auto op = step.graph.node(ir::NodeId{i}).key.op;
    if (op == ir::Op::kRmsNorm || op == ir::Op::kMul || op == ir::Op::kUnary)
      origins.emplace_back(i);
  }
  ASSERT_EQ(origins.size(), 4U);
  const planner::ComparisonSiteSelector selector{
      .source_graph_identity = step.graph.identity(),
      .kind = planner::ComparisonSiteKind::kRewrite,
      .pattern = "gdn_norm_gate",
      .version = pattern->version,
      .member_origins = {{origins[0]}, {origins[1]}, {origins[2]}, {origins[3]}},
      .member_owner_segments = {},
      .boundary_ports = {}};

  const auto row_bytes = ir::StorageBytes(step.graph.tensor(rows));
  const auto weight_bytes = ir::StorageBytes(step.graph.tensor(weight));
  const auto gate_bytes = ir::StorageBytes(step.graph.tensor(gate));
  ASSERT_TRUE(row_bytes && weight_bytes && gate_bytes);
  const std::array external{co::ComparisonExternalBinding{.original_root = rows,
                                                          .allocation_class = 1,
                                                          .allocation_bytes = *row_bytes,
                                                          .bytes = *row_bytes,
                                                          .alignment = 16},
                            co::ComparisonExternalBinding{.original_root = weight,
                                                          .part = llmp::format::PartRole::kData,
                                                          .allocation_class = 2,
                                                          .allocation_bytes = *weight_bytes,
                                                          .bytes = *weight_bytes,
                                                          .alignment = 16},
                            co::ComparisonExternalBinding{.original_root = gate,
                                                          .allocation_class = 3,
                                                          .allocation_bytes = *gate_bytes,
                                                          .bytes = *gate_bytes,
                                                          .alignment = 16}};
  const co::ComparisonBoundaryView view{io, {}, 0, external};
  constexpr std::uint64_t kArenaBytes = 1U << 20U;
  constexpr std::uint64_t kPoolBytes = 1U << 20U;
  auto& activation = life_->test_activation;
  auto& weight_buffer = life_->test_weights;
  auto& pool = life_->test_pool;
  auto& persistent = life_->test_persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(weight_buffer.Allocate(*weight_bytes));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());

  const planner::StepShape shape{.stage = 0,
                                 .phase = llmp::execution::Phase::kDecode,
                                 .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  const planner::Policy policy{};
  const planner::PlanOptions options{.capturable = true, .within = recipe.within, .shape = &shape};
  auto inventory =
      planner::BoundComparison(view, selector, std::make_shared<PaidBoundaryFunder>(life_->paid),
                               *recipe.registry, recipe.device, policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  auto alternative = planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto candidate = planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                            recipe.device, policy, options);
  ASSERT_TRUE(candidate) << (candidate ? "" : candidate.error().detail);
  ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
  ASSERT_EQ(inventory->incumbent.plan.applied.front().pattern, "gdn_norm_gate");
  EXPECT_TRUE(std::ranges::none_of(candidate->plan.applied, [](const auto& applied) {
    return applied.pattern == "gdn_norm_gate";
  }));
  const auto selected_names = [&](const planner::StepPlan& plan) {
    std::vector<std::string> names;
    for (const auto& selected : plan.steps)
      names.emplace_back(recipe.registry->at(selected.implementation).name);
    return names;
  };
  const auto incumbent_names = selected_names(inventory->incumbent.plan);
  const auto alternative_names = selected_names(candidate->plan);
  ASSERT_TRUE(std::ranges::contains(incumbent_names, kg::kGdnNormGateName));
  ASSERT_FALSE(std::ranges::contains(alternative_names, kg::kGdnNormGateName));
  ASSERT_NE(incumbent_names, alternative_names);

  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  auto full_boundary = co::MakeComparisonBoundary(view, funder);
  ASSERT_TRUE(full_boundary) << (full_boundary ? "" : full_boundary.error().detail);
  std::vector<std::uint32_t> member_offsets{0};
  std::vector<ir::NodeId> members;
  for (const auto& group : inventory->site.accepted_member_origins) {
    ASSERT_FALSE(group.empty());
    members.insert(members.end(), group.begin(), group.end());
    ASSERT_LE(members.size(), std::numeric_limits<std::uint32_t>::max());
    member_offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }
  ASSERT_EQ(members, origins);
  auto region = co::MakeComparisonRegionBoundary({view, member_offsets, members}, funder);
  ASSERT_TRUE(region) << (region ? "" : region.error().detail);

  const auto offset_for = [](const planner::StepPlan& plan,
                             ir::TensorId original) -> std::optional<std::uint64_t> {
    if (!original.valid() || original.v >= plan.tensors.size()) return {};
    const auto mapped = plan.tensors[original.v];
    if (!mapped.valid()) return {};
    const auto root = plan.graph.root(mapped);
    const auto found = std::ranges::find(plan.placement.offsets, root,
                                         &std::pair<ir::TensorId, std::uint64_t>::first);
    return found == plan.placement.offsets.end() ? std::nullopt
                                                 : std::optional<std::uint64_t>{found->second};
  };
  std::array<const planner::StepPlan*, 2> plans{&inventory->incumbent.plan, &candidate->plan};
  std::array<std::vector<kn::ComparisonBindingRange>, 2> binding_storage;
  std::array<kn::ComparisonRequest, 2> requests;
  std::array<kn::ComparisonRegionCapture, 2> captures;
  std::array<std::optional<planner::ComparisonOriginDag>, 2> origin_copies;
  std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
  for (std::size_t arm = 0; arm < plans.size(); ++arm) {
    ASSERT_LE(plans[arm]->placement.extent, kArenaBytes);
    for (const auto& fact : external) {
      auto base = weight_buffer.address();
      if (!fact.part) {
        const auto offset = offset_for(*plans[arm], fact.original_root);
        ASSERT_TRUE(offset);
        base = activation.address() + *offset;
      }
      binding_storage[arm].push_back({.original_root = fact.original_root,
                                      .part = fact.part,
                                      .allocation_class = fact.allocation_class,
                                      .allocation_bytes = fact.allocation_bytes,
                                      .offset = fact.offset,
                                      .bytes = fact.bytes,
                                      .alignment = fact.alignment,
                                      .base = base,
                                      .resource_part_identity = fact.resource_part_identity});
    }
    const auto& owned = arm == 0 ? inventory->incumbent : *candidate;
    ASSERT_TRUE(owned.final_origins);
    auto copied = owned.final_origins->Clone(funder);
    ASSERT_TRUE(copied) << (copied ? "" : copied.error().detail);
    origin_copies[arm].emplace(std::move(*copied));
    requests[arm] = {.mode = kn::ComparisonRunMode::kFull,
                     .bindings = binding_storage[arm],
                     .state = step.state,
                     .active_stage = 0,
                     .expected_boundary = &*full_boundary};
    captures[arm] = {.seed = &*region,
                     .origins = &*origin_copies[arm],
                     .final_nodes = &owned.final_node_origins};
    kn::Addresses addresses{};
    addresses.weight = [&](const ir::WeightRef&, llmp::format::PartRole) {
      return weight_buffer.address();
    };
    addresses.activations = activation.address();
    addresses.activation_bytes = kArenaBytes;
    addresses.activation_alignment = 256;
    if (arm == 0) {
      auto truncated_ranges = binding_storage[arm];
      const auto gate_range =
          std::ranges::find(truncated_ranges, gate, &kn::ComparisonBindingRange::original_root);
      ASSERT_NE(gate_range, truncated_ranges.end());
      ASSERT_GT(gate_range->bytes, sizeof(float));
      gate_range->bytes -= sizeof(float);
      kn::ComparisonRequest truncated_request{.mode = kn::ComparisonRunMode::kFull,
                                              .bindings = truncated_ranges,
                                              .state = step.state,
                                              .active_stage = 0,
                                              .expected_boundary = &*full_boundary};
      kn::LowerInput truncated_input(*plans[arm], step, shape, addresses, 0, {}, &truncated_request,
                                     &captures[arm]);
      auto truncated_program = life_->backend->Lower(truncated_input);
      if (truncated_program) {
        auto truncated_projection = (*truncated_program)
                                        ->ProjectComparisonRegionCertificate(
                                            *region, kn::ComparisonCertificatePhase::kDescriptor);
        EXPECT_NE(truncated_projection.status, kn::ComparisonRegionStatus::kClosed)
            << "a shortened authenticated gate range must not close the local proof";
        EXPECT_FALSE(truncated_projection.reason.empty());
      } else {
        EXPECT_FALSE(truncated_program.error().empty());
      }
    }
    kn::LowerInput input(*plans[arm], step, shape, addresses, 0, {}, &requests[arm],
                         &captures[arm]);
    auto lowered = life_->backend->Lower(input);
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error());
    programs[arm] = std::move(*lowered);
  }

  auto descriptor0 = programs[0]->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor0.status, kn::ComparisonRegionStatus::kClosed) << descriptor0.reason;
  ASSERT_EQ(descriptor0.seal.state, kn::ComparisonProofState::kDescriptorVerified);
  auto descriptor1 = programs[1]->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor1.status, kn::ComparisonRegionStatus::kClosed) << descriptor1.reason;
  ASSERT_EQ(descriptor1.seal.state, kn::ComparisonProofState::kDescriptorVerified);
  const auto same_persistent_seal = [](const auto& left, const auto& right) {
    return left.version == right.version && left.missing == right.missing &&
           left.subject == right.subject && left.boundary == right.boundary &&
           left.planned_region == right.planned_region &&
           left.physical_recipe == right.physical_recipe && left.policy == right.policy &&
           left.registry == right.registry && left.backend_recipe == right.backend_recipe &&
           left.backend_build == right.backend_build &&
           left.backend_selection == right.backend_selection &&
           left.backend_device == right.backend_device;
  };
  const auto descriptor_remainder = programs[0]->CompareComparisonRemainder(
      *programs[1], *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor_remainder.status, kn::ComparisonRemainderStatus::kSame)
      << descriptor_remainder.reason;
  EXPECT_TRUE(descriptor_remainder.required_members.empty());
  const auto descriptor_remainder_reverse = programs[1]->CompareComparisonRemainder(
      *programs[0], *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor_remainder_reverse.status, kn::ComparisonRemainderStatus::kSame)
      << descriptor_remainder_reverse.reason;
  EXPECT_TRUE(descriptor_remainder_reverse.required_members.empty());
  const auto ready_key = en::RecipeComparisonKeyForReadyCertificates(
      *region, ir::Fused(ir::FusedKind::kNormGate), shape.phase, recipe.registry->SelectionDigest(),
      descriptor0.seal, descriptor1.seal);
  ASSERT_TRUE(ready_key) << "the actual default local NormGate pair must produce a ready key";

  for (auto& program : programs) {
    ASSERT_TRUE(program->Bind(**launch));
  }
  auto bound0 = programs[0]->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kBound);
  auto bound1 = programs[1]->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kBound);
  ASSERT_EQ(bound0.status, kn::ComparisonRegionStatus::kClosed) << bound0.reason;
  ASSERT_EQ(bound1.status, kn::ComparisonRegionStatus::kClosed) << bound1.reason;
  ASSERT_EQ(bound0.seal.state, kn::ComparisonProofState::kBoundVerified);
  ASSERT_EQ(bound1.seal.state, kn::ComparisonProofState::kBoundVerified);
  EXPECT_TRUE(same_persistent_seal(bound0.seal, descriptor0.seal));
  EXPECT_TRUE(same_persistent_seal(bound1.seal, descriptor1.seal));
  const auto bound_remainder = programs[0]->CompareComparisonRemainder(
      *programs[1], *region, kn::ComparisonCertificatePhase::kBound);
  ASSERT_EQ(bound_remainder.status, kn::ComparisonRemainderStatus::kSame) << bound_remainder.reason;
  const auto bound_remainder_reverse = programs[1]->CompareComparisonRemainder(
      *programs[0], *region, kn::ComparisonCertificatePhase::kBound);
  ASSERT_EQ(bound_remainder_reverse.status, kn::ComparisonRemainderStatus::kSame)
      << bound_remainder_reverse.reason;
  life_->test_program = std::move(programs[0]);
  life_->test_peer_program = std::move(programs[1]);
  life_->test_launch = std::move(*launch);
  const std::array<kn::StepProgram*, 2> live_programs{life_->test_program.get(),
                                                      life_->test_peer_program.get()};

  const auto row_count = static_cast<std::size_t>(kWidth) * kHeads;
  std::vector<float> gamma(kWidth), rows_values(row_count), gate_values(row_count);
  for (std::size_t i = 0; i < gamma.size(); ++i)
    gamma[i] = 0.75F + static_cast<float>(i % 11) * 0.025F;
  ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(weight_buffer.address()), gamma.data(),
                       gamma.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  auto reference = reference::Executor::Create(step.graph);
  ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
  ASSERT_TRUE(reference->BindValues(weight, gamma));
  auto native = life_->execution->Submission(life_->node.stream(0));
  ASSERT_TRUE(native) << (native ? "" : native.error().detail);
  for (std::size_t frame = 0; frame < 2; ++frame) {
    for (std::size_t i = 0; i < row_count; ++i) {
      const auto centered = static_cast<float>(static_cast<int>((i + frame * 7) % 31) - 15);
      rows_values[i] = centered * (frame == 0 ? 0.013F : -0.021F) + 0.19F;
      gate_values[i] = static_cast<float>(static_cast<int>((i * 3 + frame * 5) % 23) - 11) *
                           (frame == 0 ? 0.17F : 0.11F) +
                       0.03F;
    }
    ASSERT_TRUE(reference->BindValues(rows, rows_values));
    ASSERT_TRUE(reference->BindValues(gate, gate_values));
    ASSERT_TRUE(reference->Run());
    auto expected = reference->Values(step.graph.outputs().front());
    ASSERT_TRUE(expected) << (expected ? "" : expected.error().detail);
    ASSERT_EQ(expected->size(), row_count);
    for (std::size_t arm = 0; arm < live_programs.size(); ++arm) {
      ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(activation.address()), 0xa5,
                           static_cast<std::size_t>(kArenaBytes)),
                cudaSuccess);
      for (const auto& slot : live_programs[arm]->inputs()) {
        ASSERT_GT(slot.bytes, 0U);
        std::span<const float> values;
        if (slot.tensor == rows)
          values = rows_values;
        else if (slot.tensor == gate)
          values = gate_values;
        else
          FAIL() << "unexpected local NormGate input tensor " << slot.tensor.v;
        ASSERT_EQ(slot.bytes, values.size_bytes());
        ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(slot.device_address), values.data(),
                             values.size_bytes(), cudaMemcpyHostToDevice),
                  cudaSuccess);
      }
      ASSERT_TRUE(live_programs[arm]->Run(*life_->test_launch, *native));
      auto fence = life_->execution->Record(life_->node.stream(0));
      ASSERT_TRUE(fence) << (fence ? "" : fence.error().detail);
      for (;;) {
        auto status = life_->execution->Query(*fence);
        ASSERT_TRUE(status) << (status ? "" : status.error().detail);
        if (*status == llmp::providers::FenceState::kComplete) break;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      ASSERT_TRUE(life_->execution->Release(*fence));
      ASSERT_EQ(live_programs[arm]->outputs().size(), 1U);
      const auto& output = live_programs[arm]->outputs().front();
      ASSERT_EQ(output.bytes, row_count * sizeof(float));
      std::vector<float> actual(row_count);
      ASSERT_EQ(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(output.device_address),
                           output.bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]) && std::isfinite((*expected)[i])) << i;
        EXPECT_LE(std::abs(static_cast<double>(actual[i]) - (*expected)[i]),
                  1.0e-5 * (1.0 + std::abs(static_cast<double>((*expected)[i]))))
            << "arm=" << arm << " frame=" << frame << " element=" << i;
      }
      std::vector<float> observed_gamma(gamma.size());
      ASSERT_EQ(
          cudaMemcpy(observed_gamma.data(), reinterpret_cast<const void*>(weight_buffer.address()),
                     observed_gamma.size() * sizeof(float), cudaMemcpyDeviceToHost),
          cudaSuccess);
      EXPECT_EQ(observed_gamma, gamma) << "arm=" << arm << " frame=" << frame;
    }
  }
}

void GgmlComparisonLowerForGpu::CheckGdnNormGateReadyKey(bool expanded) {
  // This is descriptor applicability only: the complete GDN prefix/remainder
  // is deliberately outside the executable local comparison domain.
  constexpr std::uint64_t kArenaBytes = 32 * kMiB;
  ASSERT_TRUE(life_->test_activation.Allocate(kArenaBytes));
  ASSERT_TRUE(life_->test_weights.Allocate(kArenaBytes));
  ASSERT_TRUE(life_->test_state.Allocate(kArenaBytes));
  ASSERT_TRUE(life_->test_pool.Allocate(kArenaBytes));
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  auto actual = MakeSmallGdnStateSource();
  ASSERT_TRUE(actual) << (actual ? "" : actual.error());

  // Match RunReadyNormGatePair's zero-prefix graph, including its 2D gate
  // root and rank4 reshape; do not insert storage-protecting operations.
  ir::Builder builder;
  builder.SetComponent(17);
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  ir::TensorId gate_input{};
  if (expanded) {
    // The real pre-norm stream exists before the recurrence produces rows.
    builder.SetJoined();
    gate_input = builder.Input(ir::Dense(kF32, ir::ShapeOf({64, 1})));
  }
  builder.SetOwner(0);
  const auto rows = builder.Input(ir::Dense(kF32, ir::ShapeOf({128, 48, 1, 1})));
  constexpr ir::WeightRef ref{.stage = 0, .component = 17, .role = 1, .layer = 0};
  const auto gamma = builder.Weight(ir::Dense(kF32, ir::ShapeOf({128})), ref, "norm.weight");
  builder.SetJoined();
  ir::TensorId gate_root{}, gate{}, normalized{}, scaled{}, gate_weight{};
  constexpr ir::WeightRef gate_ref{.stage = 0, .component = 17, .role = 2, .layer = 0};
  if (expanded) {
    // Match the real BuildStep order: norm/scale, then the gate projection.
    // The pre-norm stream's last use is this MatMul; the GDN residual reads
    // the original stream, so no artificial keep or downstream use is needed.
    normalized = builder.RmsNorm(rows, 1.0e-6F);
    scaled = builder.Mul(normalized, gamma);
    gate_weight = builder.Weight(ir::Dense(kF32, ir::ShapeOf({64, 6144})), gate_ref, "gate.weight");
    gate_root = builder.MatMul(gate_weight, gate_input);
    gate = builder.Reshape(gate_root, ir::ShapeOf({128, 48, 1, 1}));
  } else {
    gate_root = builder.Input(ir::Dense(kF32, ir::ShapeOf({6144, 1})));
    gate = builder.Reshape(gate_root, ir::ShapeOf({128, 48, 1, 1}));
    normalized = builder.RmsNorm(rows, 1.0e-6F);
    scaled = builder.Mul(normalized, gamma);
  }
  const auto sigmoid = builder.Unary(gate, ir::UnaryFn::kSigmoid);
  const auto result = builder.Mul(scaled, sigmoid);
  const auto output = builder.Reshape(result, ir::ShapeOf({6144, 1}));
  builder.Output(output);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph producer{
      .graph = std::move(*graph),
      .inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures, .segment = 0, .component = 17},
                 {.tensor = expanded ? gate_input : gate_root,
                  .kind = co::InputKind::kFeatures,
                  .component = 17}},
      .weights = {{.tensor = gamma,
                   .ref = ref,
                   .parts = {{"norm.weight", llmp::format::PartRole::kData}}}},
      .state = {},
      .logits = output};
  if (expanded) {
    producer.inputs[1].tensor = gate_input;
    producer.weights.push_back({.tensor = gate_weight,
                                .ref = gate_ref,
                                .parts = {{"gate.weight", llmp::format::PartRole::kData}}});
  }
  const planner::StepShape producer_shape{.stage = 0,
                                          .phase = planner::Phase::kDecode,
                                          .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  std::array<std::optional<llmp::execution::RecipeComparisonKey>, 2> keys;
  auto observe = [&](co::StepGraph& step, const planner::StepShape& shape,
                     const md::StateLayoutDerived* layout, std::size_t which) {
    SCOPED_TRACE(which == 0 ? "actual shared GDN BuildStep"
                 : expanded ? "expanded gate-projection producer"
                            : "zero-prefix Ready producer");
    auto admitted = step.graph.Clone(funder);
    ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error().detail);
    step.graph = std::move(*admitted);
    const auto* pattern = planner::FindPattern("gdn_norm_gate");
    ASSERT_NE(pattern, nullptr);
    ir::TensorId norm_weight{};
    if (which == 0) {
      for (const auto& weight : step.weights)
        if (std::ranges::any_of(weight.parts,
                                [](const auto& part) { return part.first == "gdn_norm"; }))
          norm_weight = weight.tensor;
    } else {
      norm_weight = gamma;
    }
    ASSERT_TRUE(norm_weight.valid());
    ir::NodeId norm{}, mul{}, unary{}, final_mul{};
    for (std::uint32_t n = 0; n < step.graph.nodes().size(); ++n) {
      const auto& node = step.graph.node(ir::NodeId{n});
      if (node.key.op != ir::Op::kMul || node.nsrc != 2) continue;
      for (std::size_t side = 0; side < 2; ++side) {
        if (node.source(1 - side) != norm_weight) continue;
        const auto parent = step.graph.producer(node.source(side));
        if (parent.valid() && step.graph.node(parent).key.op == ir::Op::kRmsNorm) {
          ASSERT_FALSE(norm.valid());
          norm = parent;
          mul = ir::NodeId{n};
        }
      }
    }
    ASSERT_TRUE(norm.valid() && mul.valid());
    for (std::uint32_t n = 0; n < step.graph.nodes().size(); ++n) {
      const auto& node = step.graph.node(ir::NodeId{n});
      if (node.key.op != ir::Op::kMul || node.nsrc != 2) continue;
      for (std::size_t side = 0; side < 2; ++side) {
        if (node.source(side) != step.graph.node(mul).outputs()[0]) continue;
        const auto parent = step.graph.producer(node.source(1 - side));
        if (parent.valid() && step.graph.node(parent).key.op == ir::Op::kUnary) {
          ASSERT_FALSE(final_mul.valid());
          unary = parent;
          final_mul = ir::NodeId{n};
        }
      }
    }
    ASSERT_TRUE(unary.valid() && final_mul.valid());
    const planner::ComparisonSiteSelector selector{
        .source_graph_identity = step.graph.identity(),
        .kind = planner::ComparisonSiteKind::kRewrite,
        .pattern = "gdn_norm_gate",
        .version = pattern->version,
        .member_origins = {{norm}, {mul}, {unary}, {final_mul}},
        .member_owner_segments = {},
        .boundary_ports = {}};
    std::map<ir::WeightRef, kn::BoundedAddress> weights;
    std::uint64_t weight_end = 0;
    for (const auto& weight : step.weights) {
      const auto parts = llmp::format::PartsOf(step.graph.tensor(weight.tensor).format);
      ASSERT_TRUE(parts);
      ASSERT_EQ(parts->size(), 1U);
      const auto& part = parts->parts().front();
      weight_end = (weight_end + 255U) & ~std::uint64_t{255U};
      ASSERT_LE(part.readable, kArenaBytes - weight_end);
      ASSERT_TRUE(
          weights
              .emplace(weight.ref,
                       kn::BoundedAddress{.base = life_->test_weights.address() + weight_end,
                                          .bytes = part.readable,
                                          .alignment = 256})
              .second);
      weight_end += part.readable;
    }
    std::map<ir::StateRef, kn::BoundedAddress> states;
    for (const auto& state : step.state) {
      ASSERT_NE(layout, nullptr);
      ASSERT_LT(state.item, layout->items.size());
      const auto& item = layout->items[state.item];
      ASSERT_LE(item.offset + item.bytes, kArenaBytes);
      ASSERT_TRUE(states
                      .emplace(state.ref,
                               kn::BoundedAddress{.base = life_->test_state.address() + item.offset,
                                                  .bytes = item.bytes,
                                                  .alignment = 256})
                      .second);
    }
    kn::Addresses addresses{};
    addresses.weight = [&](const ir::WeightRef& key, llmp::format::PartRole) {
      return weights.at(key).base;
    };
    addresses.weight_range = [&](const ir::WeightRef& key,
                                 llmp::format::PartRole) -> std::optional<kn::BoundedAddress> {
      return weights.at(key);
    };
    addresses.weight_alignment = [](const ir::WeightRef&, llmp::format::PartRole) {
      return std::uint64_t{256};
    };
    addresses.state = [&](const ir::StateRef& key) { return states.at(key).base; };
    addresses.state_range = [&](const ir::StateRef& key) -> std::optional<kn::BoundedAddress> {
      return states.at(key);
    };
    addresses.state_alignment = [](const ir::StateRef&) { return std::uint64_t{256}; };
    addresses.activations = life_->test_activation.address();
    addresses.activation_bytes = kArenaBytes;
    addresses.activation_alignment = 256;
    planner::PlanOptions options{.capturable = true, .within = recipe.within, .shape = &shape};
    std::vector<ir::TensorId> staged, aligned;
    if (which == 0) {
      options.written = recipe.written;
      options.on_stream = recipe.on_stream;
      for (const auto& input : step.inputs) {
        if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
        if (input.kind != co::InputKind::kCellPositions && input.kind != co::InputKind::kCommitRows)
          aligned.push_back(input.tensor);
      }
      options.staged = staged;
      options.alignment_staged = aligned;
      options.activation_alignment = 256;
      options.root_alignment = [](const ir::Graph& g, ir::TensorId root) {
        const auto& desc = g.tensor(root);
        return std::holds_alternative<ir::WeightRef>(desc.binding) ||
                       std::holds_alternative<ir::StateRef>(desc.binding)
                   ? std::uint64_t{256}
                   : std::uint64_t{0};
      };
    }
    const auto policy = planner::ForStep({}, shape.rows(), shape.segments.size());
    ASSERT_TRUE(policy) << (policy ? "" : policy.error());
    auto preliminary =
        planner::Plan(step.graph.Clone(), *recipe.registry, recipe.device, *policy, options);
    ASSERT_TRUE(preliminary) << (preliminary ? "" : preliminary.error().detail);
    ASSERT_LE(preliminary->placement.extent, kArenaBytes);
    auto bindings = en::ComparisonSourceBindings::Create(funder, step, 0);
    ASSERT_TRUE(bindings) << (bindings ? "" : bindings.error().detail);
    auto initial = bindings->Resolve(*preliminary, addresses, recipe.written);
    ASSERT_TRUE(initial) << (initial ? "" : initial.error().detail);
    const auto view = bindings->boundary_view();
    ASSERT_TRUE(view);
    auto source_boundary = co::MakeComparisonBoundary(*view, funder);
    ASSERT_TRUE(source_boundary);
    const auto mapped_boundary = co::ValidateMappedComparisonBoundary(
        *source_boundary, preliminary->graph, preliminary->tensors);
    ASSERT_TRUE(mapped_boundary) << (mapped_boundary ? "" : mapped_boundary.error().detail);
    auto inventory = planner::BoundComparison(*view, selector, funder, *recipe.registry,
                                              recipe.device, *policy, options);
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    auto alternative = planner::MakeAlternative(*inventory);
    ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
    auto candidate = planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                              recipe.device, *policy, options);
    ASSERT_TRUE(candidate) << (candidate ? "" : candidate.error().detail);
    std::vector<std::uint32_t> offsets{0};
    std::vector<ir::NodeId> members;
    for (const auto& group : inventory->site.accepted_member_origins) {
      ASSERT_FALSE(group.empty());
      members.insert(members.end(), group.begin(), group.end());
      offsets.push_back(static_cast<std::uint32_t>(members.size()));
    }
    auto made_region = co::MakeComparisonRegionBoundary({*view, offsets, members}, funder);
    ASSERT_TRUE(made_region) << (made_region ? "" : made_region.error().detail);
    std::optional<co::OwnedComparisonRegionBoundary> region;
    region.emplace(std::move(*made_region));
    std::array<kn::ComparisonRegionRecipeSeal, 2> seals;
    std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
    std::array<std::vector<ir::NodeId>, 2> expansions;
    for (std::size_t arm = 0; arm < 2; ++arm) {
      SCOPED_TRACE(arm);
      auto& owned = arm == 0 ? inventory->incumbent : *candidate;
      ASSERT_TRUE(owned.final_origins);
      ASSERT_LE(owned.plan.placement.extent, kArenaBytes);
      std::string selected;
      bool norm_gate_selected = false;
      for (const auto& op : owned.plan.steps) {
        const auto name = recipe.registry->at(op.implementation).name;
        selected += std::string(name) + " ";
        norm_gate_selected |= name == kg::kGdnNormGateName;
      }
      EXPECT_EQ(norm_gate_selected, arm == 0) << selected;
      std::fprintf(stderr, "GDN Ready control source=%zu arm=%zu recipes=%s\n", which, arm,
                   selected.c_str());
      auto resolved = bindings->Resolve(owned.plan, addresses, recipe.written);
      ASSERT_TRUE(resolved) << (resolved ? "" : resolved.error().detail);
      const kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                          .bindings = resolved->ranges(),
                                          .state = step.state,
                                          .active_stage = 0,
                                          .expected_boundary = &inventory->source_boundary};
      const kn::ComparisonRegionCapture capture{.seed = &*region,
                                                .origins = &*owned.final_origins,
                                                .final_nodes = &owned.final_node_origins};
      kn::LowerInput input(owned.plan, step, shape, addresses, which == 0 ? 256 : 0, {}, &request,
                           &capture);
      auto program = life_->backend->Lower(input);
      ASSERT_TRUE(program) << (program ? "" : program.error());
      const auto identity = (*program)->comparison_plan_identity();
      ASSERT_TRUE(identity);
      EXPECT_EQ(identity->source_graph_identity, step.graph.identity());
      EXPECT_EQ(identity->planned_graph_identity, owned.plan.graph.identity());
      programs[arm] = std::move(*program);
      if (expanded) {
        const auto projection = programs[arm]->ProjectComparisonRegionDescriptor(*region);
        if (arm == 1) {
          ASSERT_EQ(projection.status, kn::ComparisonRegionStatus::kExpand) << projection.reason;
          ASSERT_FALSE(projection.required_members.empty());
        }
        expansions[arm].assign(projection.required_members.begin(),
                               projection.required_members.end());
      }
    }
    if (expanded) {
      std::vector<std::uint8_t> membership(step.graph.nodes().size(), 0);
      for (const auto member : members) membership[member.v] = 1;
      std::vector<ir::NodeId> additions;
      additions.reserve(step.graph.nodes().size());
      const auto merged =
          en::MergeComparisonExpansions(expansions[0], expansions[1], membership, additions);
      ASSERT_EQ(merged, en::ComparisonExpansionMerge::kProgress);
      ASSERT_EQ(additions.size(), 1U);
      EXPECT_EQ(step.graph.node(additions.front()).key, ir::Primitive(ir::Op::kMatMul));
      EXPECT_EQ(additions.front(),
                step.graph.producer(step.graph.root(step.graph.node(unary).source(0))));
      for (const auto addition : additions) {
        members.push_back(addition);
        offsets.push_back(static_cast<std::uint32_t>(members.size()));
      }
      auto candidate_region = co::MakeComparisonRegionBoundary({*view, offsets, members}, funder);
      ASSERT_TRUE(candidate_region) << (candidate_region ? "" : candidate_region.error().detail);
      region.emplace(std::move(*candidate_region));
    }
    for (std::size_t arm = 0; arm < 2; ++arm) {
      SCOPED_TRACE(arm);
      const auto certificate = programs[arm]->ProjectComparisonRegionCertificate(
          *region, kn::ComparisonCertificatePhase::kDescriptor);
      ASSERT_EQ(certificate.status, kn::ComparisonRegionStatus::kClosed) << certificate.reason;
      ASSERT_EQ(certificate.seal.state, kn::ComparisonProofState::kDescriptorVerified);
      seals[arm] = certificate.seal;
      if (which == 0 && arm == 1) {
        // Applicability may describe disjointly interleaved calls, but this
        // does not authorize executable span replay or whole-GDN adoption.
        const auto executable = programs[arm]->ProjectComparisonRegionDescriptor(*region);
        EXPECT_NE(executable.status, kn::ComparisonRegionStatus::kClosed);
        const auto remainder = programs[arm]->CompareComparisonRemainder(
            *programs[arm], *region, kn::ComparisonCertificatePhase::kDescriptor);
        EXPECT_NE(remainder.status, kn::ComparisonRemainderStatus::kSame);
      }
    }
    keys[which] = en::RecipeComparisonKeyForReadyCertificates(
        *region, ir::Fused(ir::FusedKind::kNormGate), shape.phase,
        recipe.registry->SelectionDigest(), seals[0], seals[1]);
    ASSERT_TRUE(keys[which]);
    std::fprintf(
        stderr, "GDN Ready control source=%zu complete_key=%s backend_build=%s\n", which,
        llmp::base::ToHex(llmp::execution::RecipeComparisonKeyDigest(*keys[which])).c_str(),
        llmp::base::ToHex(keys[which]->incumbent.backend_build).c_str());
  };
  observe(actual->step, actual->shape, &actual->layout, 0);
  ASSERT_TRUE(keys[0]);
  observe(producer, producer_shape, nullptr, 1);
  ASSERT_TRUE(keys[1]);
  const auto& a = *keys[0];
  const auto& b = *keys[1];
  EXPECT_EQ(a.subject, b.subject) << "subject " << llmp::base::ToHex(a.subject) << " / "
                                  << llmp::base::ToHex(b.subject);
  EXPECT_EQ(a.boundary, b.boundary)
      << "boundary " << llmp::base::ToHex(a.boundary) << " / " << llmp::base::ToHex(b.boundary);
  EXPECT_EQ(a.incumbent.planned_region, b.incumbent.planned_region) << "incumbent planned region";
  EXPECT_EQ(a.alternative.planned_region, b.alternative.planned_region)
      << "alternative planned region";
  if (expanded) {
    // The real full source leaves this small crossing disjoint; the isolated
    // producer reuses its dead stream input for output/sigmoid storage.
    EXPECT_NE(a.incumbent.physical_recipe, b.incumbent.physical_recipe);
    EXPECT_NE(a.alternative.physical_recipe, b.alternative.physical_recipe);
  } else {
    EXPECT_EQ(a.incumbent.physical_recipe, b.incumbent.physical_recipe);
    EXPECT_NE(a.alternative.physical_recipe, b.alternative.physical_recipe)
        << "the real interleaved gate producer has a different physical alias class";
  }
  // Every complete-key field remains significant. This bounded control
  // observes inequality; it does not qualify the isolated producer for this
  // default shared GDN alternative's physical recipe.
  EXPECT_EQ(a.version, b.version);
  EXPECT_EQ(a.region_canonicalizer, b.region_canonicalizer);
  EXPECT_EQ(a.discovery_operation, b.discovery_operation);
  EXPECT_EQ(a.mode, b.mode);
  EXPECT_EQ(a.policy, b.policy);
  EXPECT_EQ(a.device, b.device);
  EXPECT_EQ(a.selection_environment, b.selection_environment);
  EXPECT_EQ(a.preparation_domain, b.preparation_domain);
  EXPECT_EQ(a.incumbent.backend_recipe, b.incumbent.backend_recipe);
  EXPECT_EQ(a.incumbent.backend_build, b.incumbent.backend_build);
  EXPECT_EQ(a.incumbent.backend_selection, b.incumbent.backend_selection);
  EXPECT_EQ(a.alternative.backend_recipe, b.alternative.backend_recipe);
  EXPECT_EQ(a.alternative.backend_build, b.alternative.backend_build);
  EXPECT_EQ(a.alternative.backend_selection, b.alternative.backend_selection);
  EXPECT_NE(a, b);
}

TEST_F(GgmlComparisonLowerForGpu, DefaultGdnBuildStepNormGateCertificatesPreserveItsPhysicalKey) {
  CheckGdnNormGateReadyKey(false);
}

TEST_F(GgmlComparisonLowerForGpu, ExpandedGdnGateProjectionPreservesTheActualStorageDifference) {
  CheckGdnNormGateReadyKey(true);
}

TEST_F(GgmlComparisonLowerForGpu,
       LocalNeoxNormRopeFrequencyClosesAndRejectsUnsafeActualDescriptors) {
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.stage = 0, .component = 17, .role = 1, .layer = 0};
  constexpr std::uint32_t kHeads = 1;
  constexpr std::uint32_t kWidth = 128;
  constexpr float kEpsilon = 1.0e-6F;
  ir::Builder builder;
  builder.SetComponent(17);
  builder.SetOwner(0);
  const auto rows = builder.Input(ir::Dense(kF32, ir::ShapeOf({kWidth, kHeads, 1, 1})));
  builder.SetJoined();
  const auto weight =
      builder.Weight(ir::Dense(kF32, ir::ShapeOf({kWidth})), weight_ref, "norm.weight");
  const auto gate = builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({1})));
  constexpr ir::WeightRef frequency_ref{.stage = 0, .component = 17, .role = 2, .layer = 0};
  const auto frequency =
      builder.Weight(ir::Dense(kF32, ir::ShapeOf({kWidth / 2})), frequency_ref, "rope.frequency");
  builder.SetOwner(0);
  const auto normalized = builder.RmsNorm(rows, kEpsilon);
  const auto scaled = builder.Mul(normalized, weight);
  const ir::RopeParams rope_params{
      .mode = ir::RopeMode::kNeox, .dims = kWidth, .original_context = 4096};
  const auto gated = builder.Rope(scaled, gate, frequency, rope_params);
  builder.Output(builder.Reshape(gated, ir::ShapeOf({kWidth, 1})));
  auto fused = std::move(builder).Finish();
  ASSERT_TRUE(fused) << (fused ? "" : fused.error().detail);
  auto graph = ir::Expand(*fused, false);
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);

  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures, .segment = 0, .component = 17},
                 {.tensor = gate, .kind = co::InputKind::kPositions, .component = 17}};
  step.weights = {co::StepWeight{.tensor = weight,
                                 .ref = weight_ref,
                                 .parts = {{"norm.weight", llmp::format::PartRole::kData}}}};
  step.weights.push_back({.tensor = frequency,
                          .ref = frequency_ref,
                          .parts = {{"rope.frequency", llmp::format::PartRole::kData}}});
  ASSERT_EQ(step.graph.outputs().size(), 1U);
  step.logits = step.graph.outputs().front();
  const auto io = co::GraphIO(step);
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* pattern = planner::FindPattern("interleaved_norm_rope");
  ASSERT_NE(pattern, nullptr);
  std::vector<ir::NodeId> origins;
  for (std::uint32_t i = 0; i < step.graph.nodes().size(); ++i) {
    const auto op = step.graph.node(ir::NodeId{i}).key.op;
    if (op == ir::Op::kRmsNorm || op == ir::Op::kMul || op == ir::Op::kRope)
      origins.emplace_back(i);
  }
  ASSERT_EQ(origins.size(), 3U);
  const planner::ComparisonSiteSelector selector{
      .source_graph_identity = step.graph.identity(),
      .kind = planner::ComparisonSiteKind::kRewrite,
      .pattern = "interleaved_norm_rope",
      .version = pattern->version,
      .member_origins = {{origins[0]}, {origins[1]}, {origins[2]}},
      .member_owner_segments = {},
      .boundary_ports = {}};

  const auto row_bytes = ir::StorageBytes(step.graph.tensor(rows));
  const auto weight_bytes = ir::StorageBytes(step.graph.tensor(weight));
  const auto gate_bytes = ir::StorageBytes(step.graph.tensor(gate));
  ASSERT_TRUE(row_bytes && weight_bytes && gate_bytes);
  const std::vector<co::ComparisonExternalBinding> external{
      co::ComparisonExternalBinding{.original_root = rows,
                                    .allocation_class = 1,
                                    .allocation_bytes = *row_bytes,
                                    .bytes = *row_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = *weight_bytes,
                                    .bytes = *weight_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = gate,
                                    .allocation_class = 3,
                                    .allocation_bytes = *gate_bytes,
                                    .bytes = *gate_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = frequency,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 4,
                                    .allocation_bytes = kWidth / 2 * sizeof(float),
                                    .bytes = kWidth / 2 * sizeof(float),
                                    .alignment = 16}};
  const co::ComparisonBoundaryView view{io, {}, 0, external};
  constexpr std::uint64_t kArenaBytes = 1U << 20U;
  constexpr std::uint64_t kPoolBytes = 1U << 20U;
  auto& activation = life_->test_activation;
  auto& weight_buffer = life_->test_weights;
  auto& pool = life_->test_pool;
  auto& persistent = life_->test_persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(weight_buffer.Allocate(*weight_bytes + kWidth / 2 * sizeof(float)));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());

  const planner::StepShape shape{.stage = 0,
                                 .phase = llmp::execution::Phase::kDecode,
                                 .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  const planner::Policy policy{};
  const planner::PlanOptions options{.capturable = true, .within = recipe.within, .shape = &shape};
  auto inventory =
      planner::BoundComparison(view, selector, std::make_shared<PaidBoundaryFunder>(life_->paid),
                               *recipe.registry, recipe.device, policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  auto alternative = planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto candidate = planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                            recipe.device, policy, options);
  ASSERT_TRUE(candidate) << (candidate ? "" : candidate.error().detail);
  ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
  ASSERT_EQ(inventory->incumbent.plan.applied.front().pattern, "interleaved_norm_rope");
  EXPECT_TRUE(std::ranges::none_of(candidate->plan.applied, [](const auto& applied) {
    return applied.pattern == "interleaved_norm_rope";
  }));
  const auto selected_names = [&](const planner::StepPlan& plan) {
    std::vector<std::string> names;
    for (const auto& selected : plan.steps)
      names.emplace_back(recipe.registry->at(selected.implementation).name);
    return names;
  };
  const auto incumbent_names = selected_names(inventory->incumbent.plan);
  const auto alternative_names = selected_names(candidate->plan);
  ASSERT_TRUE(std::ranges::contains(incumbent_names, kg::kGemmaNormRopeName));
  ASSERT_FALSE(std::ranges::contains(alternative_names, kg::kGemmaNormRopeName));
  ASSERT_NE(incumbent_names, alternative_names);

  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  auto full_boundary = co::MakeComparisonBoundary(view, funder);
  ASSERT_TRUE(full_boundary) << (full_boundary ? "" : full_boundary.error().detail);
  std::vector<std::uint32_t> member_offsets{0};
  std::vector<ir::NodeId> members;
  for (const auto& group : inventory->site.accepted_member_origins) {
    ASSERT_FALSE(group.empty());
    members.insert(members.end(), group.begin(), group.end());
    ASSERT_LE(members.size(), std::numeric_limits<std::uint32_t>::max());
    member_offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }
  ASSERT_EQ(members, origins);
  auto region = co::MakeComparisonRegionBoundary({view, member_offsets, members}, funder);
  ASSERT_TRUE(region) << (region ? "" : region.error().detail);

  const auto offset_for = [](const planner::StepPlan& plan,
                             ir::TensorId original) -> std::optional<std::uint64_t> {
    if (!original.valid() || original.v >= plan.tensors.size()) return {};
    const auto mapped = plan.tensors[original.v];
    if (!mapped.valid()) return {};
    const auto root = plan.graph.root(mapped);
    const auto found = std::ranges::find(plan.placement.offsets, root,
                                         &std::pair<ir::TensorId, std::uint64_t>::first);
    return found == plan.placement.offsets.end() ? std::nullopt
                                                 : std::optional<std::uint64_t>{found->second};
  };
  std::array<const planner::StepPlan*, 2> plans{&inventory->incumbent.plan, &candidate->plan};
  std::array<std::vector<kn::ComparisonBindingRange>, 2> binding_storage;
  std::array<kn::ComparisonRequest, 2> requests;
  std::array<kn::ComparisonRegionCapture, 2> captures;
  std::array<std::optional<planner::ComparisonOriginDag>, 2> origin_copies;
  std::array<std::unique_ptr<kn::StepProgram>, 2> programs;
  for (std::size_t arm = 0; arm < plans.size(); ++arm) {
    ASSERT_LE(plans[arm]->placement.extent, kArenaBytes);
    for (const auto& fact : external) {
      auto base = weight_buffer.address() + (fact.original_root == frequency ? *weight_bytes : 0);
      if (!fact.part) {
        const auto offset = offset_for(*plans[arm], fact.original_root);
        ASSERT_TRUE(offset);
        base = activation.address() + *offset;
      }
      binding_storage[arm].push_back({.original_root = fact.original_root,
                                      .part = fact.part,
                                      .allocation_class = fact.allocation_class,
                                      .allocation_bytes = fact.allocation_bytes,
                                      .offset = fact.offset,
                                      .bytes = fact.bytes,
                                      .alignment = fact.alignment,
                                      .base = base,
                                      .resource_part_identity = fact.resource_part_identity});
    }
    const auto& owned = arm == 0 ? inventory->incumbent : *candidate;
    ASSERT_TRUE(owned.final_origins);
    auto copied = owned.final_origins->Clone(funder);
    ASSERT_TRUE(copied) << (copied ? "" : copied.error().detail);
    origin_copies[arm].emplace(std::move(*copied));
    requests[arm] = {.mode = kn::ComparisonRunMode::kFull,
                     .bindings = binding_storage[arm],
                     .state = step.state,
                     .active_stage = 0,
                     .expected_boundary = &*full_boundary};
    captures[arm] = {.seed = &*region,
                     .origins = &*origin_copies[arm],
                     .final_nodes = &owned.final_node_origins};
    kn::Addresses addresses{};
    addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
      return weight_buffer.address() + (ref == frequency_ref ? *weight_bytes : 0);
    };
    addresses.activations = activation.address();
    addresses.activation_bytes = kArenaBytes;
    addresses.activation_alignment = 256;
    if (arm == 0) {
      auto truncated_ranges = binding_storage[arm];
      const auto gate_range =
          std::ranges::find(truncated_ranges, gate, &kn::ComparisonBindingRange::original_root);
      ASSERT_NE(gate_range, truncated_ranges.end());
      ASSERT_GE(gate_range->bytes, sizeof(std::int32_t));
      gate_range->bytes -= sizeof(std::int32_t);
      kn::ComparisonRequest truncated_request{.mode = kn::ComparisonRunMode::kFull,
                                              .bindings = truncated_ranges,
                                              .state = step.state,
                                              .active_stage = 0,
                                              .expected_boundary = &*full_boundary};
      kn::LowerInput truncated_input(*plans[arm], step, shape, addresses, 0, {}, &truncated_request,
                                     &captures[arm]);
      auto truncated_program = life_->backend->Lower(truncated_input);
      if (truncated_program) {
        auto truncated_projection = (*truncated_program)
                                        ->ProjectComparisonRegionCertificate(
                                            *region, kn::ComparisonCertificatePhase::kDescriptor);
        EXPECT_NE(truncated_projection.status, kn::ComparisonRegionStatus::kClosed)
            << "a shortened authenticated gate range must not close the local proof";
        EXPECT_FALSE(truncated_projection.reason.empty());
      } else {
        EXPECT_FALSE(truncated_program.error().empty());
      }
    }
    kn::LowerInput input(*plans[arm], step, shape, addresses, 0, {}, &requests[arm],
                         &captures[arm]);
    if (arm == 0) {
      auto actual = kg::LowerStep(input, *recipe.registry, {});
      ASSERT_TRUE(actual) << (actual ? "" : actual.error().detail);
      const auto call = std::ranges::find(actual->plan.steps, kg::kGemmaNormRopeName,
                                          &kg::PlanStep::implementation);
      ASSERT_NE(call, actual->plan.steps.end());
      ASSERT_EQ(call->nodes.size(), 3U);
      auto* norm = call->nodes[0];
      auto* mul = call->nodes[1];
      auto* rope = call->nodes[2];
      ASSERT_TRUE(kg::CheckGemmaNormRope(norm, mul, rope));
      ASSERT_TRUE(rope->src[2]);
      const auto saved_norm = *norm, saved_mul = *mul, saved_rope = *rope;
      const auto saved_positions = *rope->src[1], saved_frequency = *rope->src[2];
      const auto restore = [&] {
        *rope->src[1] = saved_positions;
        *rope->src[2] = saved_frequency;
        *norm = saved_norm;
        *mul = saved_mul;
        *rope = saved_rope;
      };
      const auto reject = [&] {
        EXPECT_FALSE(kg::CheckGemmaNormRope(norm, mul, rope));
        restore();
        ASSERT_TRUE(kg::CheckGemmaNormRope(norm, mul, rope));
      };
      // These are real arena-owned descriptors with real device addresses.
      // Never execute a malformed descriptor, and restore every changed field.
      auto* gamma_source = mul->src[0] == norm ? mul->src[1] : mul->src[0];
      for (const auto* source : {norm->src[0], gamma_source, rope->src[1], rope->src[2]}) {
        rope->data = source->data;
        reject();
      }
      rope->src[1]->type = GGML_TYPE_F32;
      reject();
      rope->src[1]->ne[0] = 0;
      reject();
      rope->src[2]->type = GGML_TYPE_F16;
      reject();
      --rope->src[2]->ne[0];
      reject();
      rope->src[0] = norm;
      reject();
      mul->src[0] = norm->src[0];
      reject();
      norm->src[0] = mul;
      reject();
      ASSERT_TRUE(kg::CheckGemmaNormRope(norm, mul, rope));
    }
    if (arm == 1) {
      auto actual = kg::LowerStep(input, *recipe.registry, {});
      ASSERT_TRUE(actual) << (actual ? "" : actual.error().detail);
      const auto call =
          std::ranges::find(actual->plan.steps, kg::kRopeExtName, &kg::PlanStep::implementation);
      ASSERT_NE(call, actual->plan.steps.end());
      ASSERT_EQ(call->nodes.size(), 1U);
      auto* rope = call->nodes.front();
      ASSERT_TRUE(kg::CheckRope(rope));
      ASSERT_TRUE(kg::CheckRopeExt(rope));
      ASSERT_TRUE(rope->src[2]);
      const auto saved_rope = *rope;
      const auto saved_positions = *rope->src[1], saved_frequency = *rope->src[2];
      const auto restore = [&] {
        *rope->src[1] = saved_positions;
        *rope->src[2] = saved_frequency;
        *rope = saved_rope;
      };
      const auto reject = [&] {
        EXPECT_FALSE(kg::CheckRope(rope));
        restore();
        ASSERT_TRUE(kg::CheckRope(rope));
      };
      for (const auto* metadata : {rope->src[1], rope->src[2]}) {
        rope->data = metadata->data;
        reject();
      }
      rope->src[1]->type = GGML_TYPE_F32;
      reject();
      rope->src[1]->ne[0] = 0;
      reject();
      rope->src[2]->type = GGML_TYPE_F16;
      reject();
      --rope->src[2]->ne[0];
      reject();
      // The ordinary RoPE launch explicitly permits exact in-place rotation.
      rope->data = rope->src[0]->data;
      ASSERT_TRUE(kg::CheckRope(rope));
      restore();
    }
    if (arm == 0) {
      const auto output_offset = offset_for(*plans[arm], step.graph.outputs().front());
      ASSERT_TRUE(output_offset);
      const auto destination = activation.address() + *output_offset;
      for (const auto target : {rows, weight, gate, frequency}) {
        auto alias_ranges = binding_storage[arm];
        auto range =
            std::ranges::find(alias_ranges, target, &kn::ComparisonBindingRange::original_root);
        ASSERT_NE(range, alias_ranges.end());
        range->base = destination;
        auto alias_addresses = addresses;
        if (target == weight || target == frequency)
          alias_addresses.weight = [&, target, destination](const ir::WeightRef& ref,
                                                            llmp::format::PartRole) {
            const auto original = ref == frequency_ref ? frequency : weight;
            return original == target
                       ? destination
                       : weight_buffer.address() + (ref == frequency_ref ? *weight_bytes : 0);
          };
        kn::ComparisonRequest alias_request{.mode = kn::ComparisonRunMode::kFull,
                                            .bindings = alias_ranges,
                                            .state = step.state,
                                            .active_stage = 0,
                                            .expected_boundary = &*full_boundary};
        kn::LowerInput alias_input(*plans[arm], step, shape, alias_addresses, 0, {}, &alias_request,
                                   &captures[arm]);
        auto unsafe = life_->backend->Lower(alias_input);
        if (unsafe) {
          auto descriptor = (*unsafe)->ProjectComparisonRegionCertificate(
              *region, kn::ComparisonCertificatePhase::kDescriptor);
          EXPECT_NE(descriptor.status, kn::ComparisonRegionStatus::kClosed);
          auto bound = (*unsafe)->Bind(**launch);
          if (bound) {
            auto projected = (*unsafe)->ProjectComparisonRegionCertificate(
                *region, kn::ComparisonCertificatePhase::kBound);
            EXPECT_NE(projected.status, kn::ComparisonRegionStatus::kClosed);
          }
        } else
          EXPECT_FALSE(unsafe.error().empty());
      }
    }
    auto lowered = life_->backend->Lower(input);
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error());
    programs[arm] = std::move(*lowered);
  }

  auto descriptor0 = programs[0]->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor0.status, kn::ComparisonRegionStatus::kClosed) << descriptor0.reason;
  ASSERT_EQ(descriptor0.seal.state, kn::ComparisonProofState::kDescriptorVerified);
  auto descriptor1 = programs[1]->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor1.status, kn::ComparisonRegionStatus::kClosed) << descriptor1.reason;
  ASSERT_EQ(descriptor1.seal.state, kn::ComparisonProofState::kDescriptorVerified);
  const auto same_persistent_seal = [](const auto& left, const auto& right) {
    return left.version == right.version && left.missing == right.missing &&
           left.subject == right.subject && left.boundary == right.boundary &&
           left.planned_region == right.planned_region &&
           left.physical_recipe == right.physical_recipe && left.policy == right.policy &&
           left.registry == right.registry && left.backend_recipe == right.backend_recipe &&
           left.backend_build == right.backend_build &&
           left.backend_selection == right.backend_selection &&
           left.backend_device == right.backend_device;
  };
  const auto descriptor_remainder = programs[0]->CompareComparisonRemainder(
      *programs[1], *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor_remainder.status, kn::ComparisonRemainderStatus::kSame)
      << descriptor_remainder.reason;
  EXPECT_TRUE(descriptor_remainder.required_members.empty());
  const auto descriptor_remainder_reverse = programs[1]->CompareComparisonRemainder(
      *programs[0], *region, kn::ComparisonCertificatePhase::kDescriptor);
  ASSERT_EQ(descriptor_remainder_reverse.status, kn::ComparisonRemainderStatus::kSame)
      << descriptor_remainder_reverse.reason;
  EXPECT_TRUE(descriptor_remainder_reverse.required_members.empty());
  const auto ready_key = en::RecipeComparisonKeyForReadyCertificates(
      *region, ir::Fused(ir::FusedKind::kRmsNormMulRope), shape.phase,
      recipe.registry->SelectionDigest(), descriptor0.seal, descriptor1.seal);
  ASSERT_TRUE(ready_key) << "the actual default local NEOX norm/RoPE pair must produce a ready key";

  for (auto& program : programs) {
    ASSERT_TRUE(program->Bind(**launch));
  }
  auto bound0 = programs[0]->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kBound);
  auto bound1 = programs[1]->ProjectComparisonRegionCertificate(
      *region, kn::ComparisonCertificatePhase::kBound);
  ASSERT_EQ(bound0.status, kn::ComparisonRegionStatus::kClosed) << bound0.reason;
  ASSERT_EQ(bound1.status, kn::ComparisonRegionStatus::kClosed) << bound1.reason;
  ASSERT_EQ(bound0.seal.state, kn::ComparisonProofState::kBoundVerified);
  ASSERT_EQ(bound1.seal.state, kn::ComparisonProofState::kBoundVerified);
  EXPECT_TRUE(same_persistent_seal(bound0.seal, descriptor0.seal));
  EXPECT_TRUE(same_persistent_seal(bound1.seal, descriptor1.seal));
  const auto bound_remainder = programs[0]->CompareComparisonRemainder(
      *programs[1], *region, kn::ComparisonCertificatePhase::kBound);
  ASSERT_EQ(bound_remainder.status, kn::ComparisonRemainderStatus::kSame) << bound_remainder.reason;
  const auto bound_remainder_reverse = programs[1]->CompareComparisonRemainder(
      *programs[0], *region, kn::ComparisonCertificatePhase::kBound);
  ASSERT_EQ(bound_remainder_reverse.status, kn::ComparisonRemainderStatus::kSame)
      << bound_remainder_reverse.reason;
  life_->test_program = std::move(programs[0]);
  life_->test_peer_program = std::move(programs[1]);
  life_->test_launch = std::move(*launch);
  const std::array<kn::StepProgram*, 2> live_programs{life_->test_program.get(),
                                                      life_->test_peer_program.get()};

  const auto row_count = static_cast<std::size_t>(kWidth) * kHeads;
  std::vector<float> gamma(kWidth), rows_values(row_count);
  for (std::size_t i = 0; i < gamma.size(); ++i)
    gamma[i] = 0.75F + static_cast<float>(i % 11) * 0.025F;
  ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(weight_buffer.address()), gamma.data(),
                       gamma.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  auto reference = reference::Executor::Create(step.graph);
  ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
  ASSERT_TRUE(reference->BindValues(weight, gamma));
  auto native = life_->execution->Submission(life_->node.stream(0));
  ASSERT_TRUE(native) << (native ? "" : native.error().detail);
  std::vector<float> frequencies(kWidth / 2);
  for (std::size_t i = 0; i < frequencies.size(); ++i)
    frequencies[i] = 0.75F + static_cast<float>(i % 13) * 0.03125F;
  ASSERT_EQ(
      cudaMemcpy(reinterpret_cast<void*>(weight_buffer.address() + *weight_bytes),
                 frequencies.data(), frequencies.size() * sizeof(float), cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_TRUE(reference->BindValues(frequency, frequencies));
  for (std::size_t frame = 0; frame < 2; ++frame) {
    for (std::size_t i = 0; i < row_count; ++i) {
      const auto centered = static_cast<float>(static_cast<int>((i + frame * 7) % 31) - 15);
      rows_values[i] = centered * (frame == 0 ? 0.013F : -0.021F) + 0.19F;
    }
    ASSERT_TRUE(reference->BindValues(rows, rows_values));
    const std::array<std::int32_t, 1> positions{static_cast<std::int32_t>(3 + frame * 19)};
    const std::array<std::int64_t, 1> reference_positions{positions[0]};
    ASSERT_TRUE(reference->BindInts(gate, reference_positions));
    ASSERT_TRUE(reference->Run());
    auto expected = reference->Values(step.graph.outputs().front());
    ASSERT_TRUE(expected) << (expected ? "" : expected.error().detail);
    ASSERT_EQ(expected->size(), row_count);
    for (std::size_t arm = 0; arm < live_programs.size(); ++arm) {
      ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(activation.address()), 0xa5,
                           static_cast<std::size_t>(kArenaBytes)),
                cudaSuccess);
      for (const auto& slot : live_programs[arm]->inputs()) {
        ASSERT_GT(slot.bytes, 0U);
        const void* data = nullptr;
        std::size_t size = 0;
        if (slot.tensor == rows) {
          data = rows_values.data();
          size = rows_values.size() * sizeof(float);
        } else if (slot.tensor == gate) {
          data = positions.data();
          size = sizeof(positions);
        } else
          FAIL() << "unexpected NEOX input";
        ASSERT_EQ(slot.bytes, size);
        ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(slot.device_address), data, size,
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
      }
      ASSERT_TRUE(live_programs[arm]->Run(*life_->test_launch, *native));
      auto fence = life_->execution->Record(life_->node.stream(0));
      ASSERT_TRUE(fence) << (fence ? "" : fence.error().detail);
      for (;;) {
        auto status = life_->execution->Query(*fence);
        ASSERT_TRUE(status) << (status ? "" : status.error().detail);
        if (*status == llmp::providers::FenceState::kComplete) break;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      ASSERT_TRUE(life_->execution->Release(*fence));
      ASSERT_EQ(live_programs[arm]->outputs().size(), 1U);
      const auto& output = live_programs[arm]->outputs().front();
      ASSERT_EQ(output.bytes, row_count * sizeof(float));
      std::vector<float> actual(row_count);
      ASSERT_EQ(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(output.device_address),
                           output.bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]) && std::isfinite((*expected)[i])) << i;
        EXPECT_LE(std::abs(static_cast<double>(actual[i]) - (*expected)[i]),
                  1.0e-5 * (1.0 + std::abs(static_cast<double>((*expected)[i]))))
            << "arm=" << arm << " frame=" << frame << " element=" << i;
      }
      std::vector<float> observed_gamma(gamma.size());
      ASSERT_EQ(
          cudaMemcpy(observed_gamma.data(), reinterpret_cast<const void*>(weight_buffer.address()),
                     observed_gamma.size() * sizeof(float), cudaMemcpyDeviceToHost),
          cudaSuccess);
      EXPECT_EQ(observed_gamma, gamma) << "arm=" << arm << " frame=" << frame;
      std::vector<float> observed_frequency(frequencies.size());
      ASSERT_EQ(cudaMemcpy(observed_frequency.data(),
                           reinterpret_cast<const void*>(weight_buffer.address() + *weight_bytes),
                           observed_frequency.size() * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
      EXPECT_EQ(observed_frequency, frequencies);
      const auto position_slot =
          std::ranges::find(live_programs[arm]->inputs(), gate, &kn::InputSlot::tensor);
      ASSERT_NE(position_slot, live_programs[arm]->inputs().end());
      std::array<std::int32_t, 1> observed_positions{};
      ASSERT_EQ(cudaMemcpy(observed_positions.data(),
                           reinterpret_cast<const void*>(position_slot->device_address),
                           sizeof(observed_positions), cudaMemcpyDeviceToHost),
                cudaSuccess);
      EXPECT_EQ(observed_positions, positions);
    }
  }
}

TEST_F(GgmlComparisonLowerForGpu, ActualGdnBuildStepBothNormGatePlansMatchFullReferenceAndState) {
  // Predeclared ComponentsQwen38 reference convention: max absolute gap /
  // max(1, max absolute reference value), independently for logits and each
  // complete F32 state tensor. This is correctness, not a speed measurement.
  constexpr double kTolerance = 1.0e-4;
  constexpr std::uint64_t kArenaBytes = 32 * kMiB;
  auto source = MakeSmallGdnStateSource();
  ASSERT_TRUE(source) << (source ? "" : source.error());
  auto& step = source->step;
  ASSERT_EQ(step.state.size(), 2U);
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  auto admitted_graph = step.graph.Clone(funder);
  ASSERT_TRUE(admitted_graph) << (admitted_graph ? "" : admitted_graph.error().detail);
  step.graph = std::move(*admitted_graph);
  ASSERT_TRUE(life_->test_activation.Allocate(kArenaBytes));
  ASSERT_TRUE(life_->test_weights.Allocate(kArenaBytes));
  ASSERT_TRUE(life_->test_state.Allocate(source->layout.bytes));
  ASSERT_TRUE(life_->test_pool.Allocate(kArenaBytes));
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(life_->test_persistent.Allocate(persistent_bytes));
  auto launch = life_->backend->Launch(
      {.base = life_->test_pool.address(), .size = llmp::base::Bytes(kArenaBytes)},
      {.base = life_->test_persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  life_->test_launch = std::move(*launch);

  std::map<ir::WeightRef, kn::BoundedAddress> weights;
  std::map<ir::TensorId, std::vector<float>> weight_values;
  std::uint64_t weight_end = 0;
  ir::TensorId norm_weight{};
  bool initialized_folded_a = false;
  for (const auto& weight : step.weights) {
    const auto& format = step.graph.tensor(weight.tensor).format;
    ASSERT_EQ(format.encoding, llmp::format::Encoding::kF32);
    const auto parts = llmp::format::PartsOf(format);
    ASSERT_TRUE(parts);
    ASSERT_EQ(parts->size(), 1U);
    ASSERT_EQ(weight.parts.size(), 1U);
    const auto& name = weight.parts.front().first;
    if (name == "gdn_norm") norm_weight = weight.tensor;
    std::vector<float> values(ir::Elements(format.shape));
    for (std::size_t i = 0; i < values.size(); ++i) {
      const float fraction = static_cast<float>((i * 17U + 11U) % 97U) / 96.0F;
      // kGdnRanges: folded A strictly negative, dt_bias and convolution small.
      values[i] = name == "gdn_a"          ? -1.2F + fraction
                  : name == "gdn_dt_bias"  ? -0.5F + fraction
                  : name == "gdn_conv"     ? -0.6F + 1.2F * fraction
                  : format.shape.rank == 1 ? 0.9F + 0.2F * fraction
                                           : -0.03F + 0.06F * fraction;
    }
    if (name == "gdn_a") {
      ASSERT_TRUE(std::ranges::all_of(values, [](float value) { return value < 0.0F; }));
      initialized_folded_a = true;
    }
    const auto& part = parts->parts().front();
    ASSERT_EQ(values.size() * sizeof(float), part.readable);
    weight_end = (weight_end + 255U) & ~std::uint64_t{255U};
    ASSERT_LE(part.readable, kArenaBytes - weight_end);
    const kn::BoundedAddress bound{.base = life_->test_weights.address() + weight_end,
                                   .bytes = part.readable,
                                   .alignment = 256};
    ASSERT_TRUE(weights.emplace(weight.ref, bound).second);
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(bound.base), values.data(), part.readable,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    weight_values.emplace(weight.tensor, std::move(values));
    weight_end += part.readable;
  }
  ASSERT_TRUE(norm_weight.valid());
  ASSERT_TRUE(initialized_folded_a);
  std::map<ir::StateRef, kn::BoundedAddress> states;
  std::map<ir::TensorId, std::vector<float>> initial_values;
  std::vector<std::byte> initial_state(source->layout.bytes, std::byte{0});
  for (const auto& state : step.state) {
    ASSERT_TRUE(state.read && state.written);
    ASSERT_LT(state.item, source->layout.items.size());
    const auto& item = source->layout.items[state.item];
    ASSERT_EQ(item.contract.encoding, llmp::format::Encoding::kF32);
    std::vector<float> values(ir::Elements(step.graph.tensor(state.tensor).shape()));
    ASSERT_EQ(values.size() * sizeof(float), item.bytes);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(static_cast<int>((i * 13U + 7U) % 61U) - 30) / 2048.0F;
    ASSERT_LE(item.offset + item.bytes, initial_state.size());
    std::memcpy(initial_state.data() + item.offset, values.data(), item.bytes);
    initial_values.emplace(state.tensor, std::move(values));
    ASSERT_TRUE(states
                    .emplace(state.ref,
                             kn::BoundedAddress{.base = life_->test_state.address() + item.offset,
                                                .bytes = item.bytes,
                                                .alignment = 256})
                    .second);
  }
  kn::Addresses addresses{};
  addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
    return weights.at(ref).base;
  };
  addresses.weight_range = [&](const ir::WeightRef& ref,
                               llmp::format::PartRole) -> std::optional<kn::BoundedAddress> {
    return weights.at(ref);
  };
  addresses.weight_alignment = [](const ir::WeightRef&, llmp::format::PartRole) {
    return std::uint64_t{256};
  };
  addresses.state = [&](const ir::StateRef& ref) { return states.at(ref).base; };
  addresses.state_range = [&](const ir::StateRef& ref) -> std::optional<kn::BoundedAddress> {
    return states.at(ref);
  };
  addresses.state_alignment = [](const ir::StateRef&) { return std::uint64_t{256}; };
  addresses.activations = life_->test_activation.address();
  addresses.activation_bytes = kArenaBytes;
  addresses.activation_alignment = 256;
  planner::PlanOptions options{
      .capturable = true, .within = recipe.within, .shape = &source->shape};
  options.written = recipe.written;
  options.on_stream = recipe.on_stream;
  std::vector<ir::TensorId> staged, aligned;
  for (const auto& input : step.inputs) {
    if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
    if (input.kind != co::InputKind::kCellPositions && input.kind != co::InputKind::kCommitRows)
      aligned.push_back(input.tensor);
  }
  options.staged = staged;
  options.alignment_staged = aligned;
  options.activation_alignment = 256;
  options.root_alignment = [](const ir::Graph& graph, ir::TensorId root) {
    const auto& desc = graph.tensor(root);
    return std::holds_alternative<ir::WeightRef>(desc.binding) ||
                   std::holds_alternative<ir::StateRef>(desc.binding)
               ? std::uint64_t{256}
               : std::uint64_t{0};
  };
  auto policy = planner::ForStep({}, source->shape.rows(), source->shape.segments.size());
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  auto preliminary =
      planner::Plan(step.graph.Clone(), *recipe.registry, recipe.device, *policy, options);
  ASSERT_TRUE(preliminary) << (preliminary ? "" : preliminary.error().detail);
  auto bindings = en::ComparisonSourceBindings::Create(funder, step, 0);
  ASSERT_TRUE(bindings) << (bindings ? "" : bindings.error().detail);
  auto resolved = bindings->Resolve(*preliminary, addresses, recipe.written);
  ASSERT_TRUE(resolved) << (resolved ? "" : resolved.error().detail);
  const auto view = bindings->boundary_view();
  ASSERT_TRUE(view);
  ir::NodeId norm{}, mul{}, unary{}, final_mul{};
  for (std::uint32_t n = 0; n < step.graph.nodes().size(); ++n) {
    const auto& node = step.graph.node(ir::NodeId{n});
    if (node.key.op != ir::Op::kMul || node.nsrc != 2) continue;
    for (std::size_t side = 0; side < 2; ++side) {
      const auto parent = step.graph.producer(node.source(side));
      if (node.source(1 - side) == norm_weight && parent.valid() &&
          step.graph.node(parent).key.op == ir::Op::kRmsNorm) {
        ASSERT_FALSE(norm.valid());
        norm = parent;
        mul = ir::NodeId{n};
      }
    }
  }
  ASSERT_TRUE(norm.valid() && mul.valid());
  for (std::uint32_t n = 0; n < step.graph.nodes().size(); ++n) {
    const auto& node = step.graph.node(ir::NodeId{n});
    if (node.key.op != ir::Op::kMul || node.nsrc != 2) continue;
    for (std::size_t side = 0; side < 2; ++side) {
      const auto parent = step.graph.producer(node.source(1 - side));
      if (node.source(side) == step.graph.node(mul).outputs()[0] && parent.valid() &&
          step.graph.node(parent).key.op == ir::Op::kUnary) {
        ASSERT_FALSE(final_mul.valid());
        unary = parent;
        final_mul = ir::NodeId{n};
      }
    }
  }
  ASSERT_TRUE(unary.valid() && final_mul.valid());
  const auto* pattern = planner::FindPattern("gdn_norm_gate");
  ASSERT_NE(pattern, nullptr);
  const planner::ComparisonSiteSelector selector{
      .source_graph_identity = step.graph.identity(),
      .kind = planner::ComparisonSiteKind::kRewrite,
      .pattern = "gdn_norm_gate",
      .version = pattern->version,
      .member_origins = {{norm}, {mul}, {unary}, {final_mul}},
      .member_owner_segments = {},
      .boundary_ports = {}};
  auto inventory = planner::BoundComparison(*view, selector, funder, *recipe.registry,
                                            recipe.device, *policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  auto alternative = planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto candidate = planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                            recipe.device, *policy, options);
  ASSERT_TRUE(candidate) << (candidate ? "" : candidate.error().detail);
  for (std::size_t arm = 0; arm < 2; ++arm) {
    const auto& plan = (arm == 0 ? inventory->incumbent : *candidate).plan;
    ASSERT_LE(plan.placement.extent, kArenaBytes);
    bool selected_norm_gate = false, selected_gdn_step = false;
    for (const auto& selected : plan.steps) {
      const auto name = recipe.registry->at(selected.implementation).name;
      selected_norm_gate |= name == kg::kGdnNormGateName;
      selected_gdn_step |= name == kg::kGdnStepName;
    }
    ASSERT_EQ(selected_norm_gate, arm == 0);
    ASSERT_TRUE(selected_gdn_step);
    auto arm_ranges = bindings->Resolve(plan, addresses, recipe.written);
    ASSERT_TRUE(arm_ranges) << (arm_ranges ? "" : arm_ranges.error().detail);
    // These are ordinary complete programs, without executable local-prefix
    // or whole-GDN remainder authority and without transferred producer keys.
    kn::LowerInput input(plan, step, source->shape, addresses, 256);
    auto program = life_->backend->Lower(input);
    ASSERT_TRUE(program) << (program ? "" : program.error());
    auto& retained = arm == 0 ? life_->test_program : life_->test_peer_program;
    retained = std::move(*program);
    auto bound = retained->Bind(*life_->test_launch);
    ASSERT_TRUE(bound) << (bound ? "" : bound.error().detail);
  }

  constexpr std::array<std::int32_t, 1> tokens{3};
  auto cpu = reference::Executor::Create(step.graph);
  ASSERT_TRUE(cpu) << (cpu ? "" : cpu.error().detail);
  for (const auto& weight : step.weights)
    ASSERT_TRUE(cpu->BindValues(weight.tensor, weight_values.at(weight.tensor)));
  for (const auto& state : step.state)
    ASSERT_TRUE(cpu->BindValues(state.tensor, initial_values.at(state.tensor)));
  for (const auto& input : step.inputs) {
    const auto values = co::InputValues(input, source->shape, tokens);
    ASSERT_TRUE(values) << (values ? "" : values.error());
    ASSERT_TRUE(cpu->BindInts(input.tensor, *values));
  }
  ASSERT_TRUE(cpu->Run()) << "original full GDN IR reference";
  const auto compare = [&](std::span<const float> actual, std::span<const float> expected) {
    ASSERT_EQ(actual.size(), expected.size());
    double worst = 0.0, scale = 1.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i]) && std::isfinite(expected[i])) << i;
      worst = std::max(worst, std::abs(static_cast<double>(actual[i]) - expected[i]));
      scale = std::max(scale, std::abs(static_cast<double>(expected[i])));
    }
    EXPECT_LE(worst / scale, kTolerance);
  };
  for (std::size_t arm = 0; arm < 2; ++arm) {
    SCOPED_TRACE(arm == 0 ? "default NormGate incumbent" : "own-decomposition alternative");
    auto& program = arm == 0 ? life_->test_program : life_->test_peer_program;
    // NaN-poison the complete arena so this arm cannot inherit valid logits
    // or temporaries from its peer if an expected write is missing.
    ASSERT_EQ(cudaMemset(life_->test_activation.pointer, 0xff, kArenaBytes), cudaSuccess);
    // Reset the complete identical state image, including convolution history
    // and recurrence, before each ordinary program run.
    ASSERT_EQ(cudaMemcpy(life_->test_state.pointer, initial_state.data(), initial_state.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    for (const auto& slot : program->inputs()) {
      ASSERT_GT(slot.bytes, 0U);
      std::vector<std::byte> payload(static_cast<std::size_t>(slot.bytes));
      if (slot.kind == kn::InputKind::kTokens || slot.kind == kn::InputKind::kPositions) {
        const auto input = std::ranges::find(step.inputs, slot.tensor, &co::StepInput::tensor);
        ASSERT_NE(input, step.inputs.end());
        const auto values = co::InputValues(*input, source->shape, tokens);
        ASSERT_TRUE(values) << (values ? "" : values.error());
        std::vector<std::int32_t> narrowed(values->size());
        std::ranges::transform(*values, narrowed.begin(),
                               [](std::int64_t value) { return static_cast<std::int32_t>(value); });
        ASSERT_TRUE(en::step_inputs::WriteRows(payload, narrowed));
      } else {
        auto written = en::step_inputs::WriteComputed(slot, source->shape, payload);
        ASSERT_TRUE(written) << (written ? "" : written.error());
      }
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(slot.device_address), payload.data(),
                           payload.size(), cudaMemcpyHostToDevice),
                cudaSuccess);
    }
    auto submission = life_->execution->Submission(life_->node.stream(0));
    ASSERT_TRUE(submission) << (submission ? "" : submission.error().detail);
    ASSERT_TRUE(program->Run(*life_->test_launch, *submission));
    auto fence = life_->execution->Record(life_->node.stream(0));
    ASSERT_TRUE(fence) << (fence ? "" : fence.error().detail);
    for (;;) {
      const auto status = life_->execution->Query(*fence);
      ASSERT_TRUE(status) << (status ? "" : status.error().detail);
      if (*status == llmp::providers::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_TRUE(life_->execution->Release(*fence));
    ASSERT_EQ(program->outputs().size(), step.graph.outputs().size());
    for (std::size_t i = 0; i < program->outputs().size(); ++i) {
      SCOPED_TRACE("complete logits");
      const auto expected = cpu->Values(step.graph.outputs()[i]);
      ASSERT_TRUE(expected) << (expected ? "" : expected.error().detail);
      const auto& slot = program->outputs()[i];
      ASSERT_EQ(slot.bytes, expected->size() * sizeof(float));
      std::vector<float> actual(expected->size());
      ASSERT_EQ(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(slot.device_address),
                           slot.bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);
      compare(actual, *expected);
    }
    for (const auto& state : step.state) {
      SCOPED_TRACE(state.ref.tensor);
      const auto expected = cpu->Values(state.tensor);
      ASSERT_TRUE(expected) << (expected ? "" : expected.error().detail);
      const auto& item = source->layout.items[state.item];
      ASSERT_EQ(expected->size() * sizeof(float), item.bytes);
      EXPECT_FALSE(std::ranges::equal(*expected, initial_values.at(state.tensor)))
          << "both state tensors must exercise nontrivial updates";
      std::vector<float> actual(expected->size());
      ASSERT_EQ(cudaMemcpy(actual.data(),
                           reinterpret_cast<const void*>(life_->test_state.address() + item.offset),
                           item.bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);
      compare(actual, *expected);
    }
  }
}

TEST_F(GgmlComparisonLowerForGpu, LocalNormGateParametersAndGateViewChangeTheActualKey) {
  constexpr std::uint32_t kHeads = 48;
  constexpr std::uint32_t kWidth = 128;
  constexpr ir::WeightRef weight_ref{.stage = 0, .component = 17, .role = 1, .layer = 0};
  struct VariantResult {
    std::optional<llmp::execution::RecipeComparisonKey> key;
    kn::ComparisonRegionRecipeSeal seal;
    bool closed = false;
    std::string reason;
  };
  std::string failure;
  enum class GateLayout { kRank4, kReshapeRank4, kOffsetView };
  auto observe = [&](float epsilon, ir::NormGateFn gate_fn, GateLayout layout,
                     llmp::format::Encoding gate_encoding =
                         llmp::format::Encoding::kF32) -> std::optional<VariantResult> {
    constexpr auto kF32 = llmp::format::Encoding::kF32;
    ir::Builder builder;
    builder.SetComponent(17);
    builder.SetOwner(0);
    const auto rows = builder.Input(ir::Dense(kF32, ir::ShapeOf({kWidth, kHeads, 1, 1})));
    builder.SetJoined();
    const auto weight =
        builder.Weight(ir::Dense(kF32, ir::ShapeOf({kWidth})), weight_ref, "norm.weight");
    ir::TensorId gate_source{};
    ir::TensorId gate_root{};
    if (layout == GateLayout::kReshapeRank4) {
      gate_root = builder.Input(ir::Dense(kF32, ir::ShapeOf({kWidth, kHeads})));
      gate_source = builder.Reshape(gate_root, ir::ShapeOf({kWidth, kHeads, 1, 1}));
    } else if (layout == GateLayout::kOffsetView) {
      gate_root = builder.Input(ir::Dense(kF32, ir::ShapeOf({kWidth, kHeads + 1})));
      constexpr std::uint64_t kRowBytes = kWidth * sizeof(float);
      const std::array<std::uint64_t, llmp::format::kMaxDims> strides{
          sizeof(float), kRowBytes, kRowBytes * (kHeads + 1), kRowBytes * (kHeads + 1)};
      const auto view = builder.View(gate_root, ir::ShapeOf({kWidth, kHeads}), strides, kRowBytes);
      gate_source = builder.Reshape(view, ir::ShapeOf({kWidth, kHeads, 1, 1}));
    } else {
      gate_root = builder.Input(ir::Dense(gate_encoding, ir::ShapeOf({kWidth, kHeads, 1, 1})));
      gate_source = gate_root;
    }
    const std::array sources{rows, weight, gate_source};
    const auto gated = builder.Fused(ir::FusedKind::kNormGate, sources,
                                     ir::NormGateParams{.eps = epsilon, .gate = gate_fn})[0];
    builder.Output(builder.Reshape(gated, ir::ShapeOf({kWidth * kHeads, 1})));
    auto fused = std::move(builder).Finish();
    if (!fused) {
      failure = "variant graph construction: " + fused.error().detail;
      return {};
    }
    auto expanded = ir::Expand(*fused, false);
    if (!expanded) {
      failure = "variant decomposition: " + expanded.error().detail;
      return {};
    }
    co::StepGraph step{.graph = std::move(*expanded), .inputs = {}, .weights = {}, .state = {}};
    step.inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures, .component = 17},
                   {.tensor = gate_root, .kind = co::InputKind::kFeatures, .component = 17}};
    step.weights = {co::StepWeight{.tensor = weight,
                                   .ref = weight_ref,
                                   .parts = {{"norm.weight", llmp::format::PartRole::kData}}}};
    step.logits = step.graph.outputs().front();
    const co::GraphBuildIO io = co::GraphIO(step);
    const auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    const auto* pattern = planner::FindPattern("gdn_norm_gate");
    if (!pattern) {
      failure = "GDN NormGate rewrite pattern is absent";
      return {};
    }
    planner::ComparisonSiteSelector selector{.source_graph_identity = step.graph.identity(),
                                             .kind = planner::ComparisonSiteKind::kRewrite,
                                             .pattern = "gdn_norm_gate",
                                             .version = pattern->version,
                                             .member_origins = {},
                                             .member_owner_segments = {},
                                             .boundary_ports = {}};
    for (std::uint32_t n = 0; n < step.graph.nodes().size(); ++n) {
      const auto op = step.graph.node(ir::NodeId{n}).key.op;
      if (op == ir::Op::kRmsNorm || op == ir::Op::kMul || op == ir::Op::kUnary)
        selector.member_origins.push_back({ir::NodeId{n}});
    }
    const auto row_bytes = ir::StorageBytes(step.graph.tensor(rows));
    const auto weight_bytes = ir::StorageBytes(step.graph.tensor(weight));
    const auto gate_bytes = ir::StorageBytes(step.graph.tensor(gate_root));
    if (!row_bytes || !weight_bytes || !gate_bytes) {
      failure = "variant source storage extent is invalid";
      return {};
    }
    const std::array external{co::ComparisonExternalBinding{.original_root = rows,
                                                            .allocation_class = 1,
                                                            .allocation_bytes = *row_bytes,
                                                            .bytes = *row_bytes,
                                                            .alignment = 16},
                              co::ComparisonExternalBinding{.original_root = weight,
                                                            .part = llmp::format::PartRole::kData,
                                                            .allocation_class = 2,
                                                            .allocation_bytes = *weight_bytes,
                                                            .bytes = *weight_bytes,
                                                            .alignment = 16},
                              co::ComparisonExternalBinding{.original_root = gate_root,
                                                            .allocation_class = 3,
                                                            .allocation_bytes = *gate_bytes,
                                                            .bytes = *gate_bytes,
                                                            .alignment = 16}};
    const co::ComparisonBoundaryView view{io, {}, 0, external};
    const planner::StepShape shape{.stage = 0,
                                   .phase = llmp::execution::Phase::kDecode,
                                   .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
    const planner::Policy policy{};
    const planner::PlanOptions options{
        .capturable = true, .within = recipe.within, .shape = &shape};
    auto inventory =
        planner::BoundComparison(view, selector, std::make_shared<PaidBoundaryFunder>(life_->paid),
                                 *recipe.registry, recipe.device, policy, options);
    if (!inventory) {
      failure = "variant BoundComparison: " + inventory.error().detail;
      return {};
    }
    std::vector<std::uint32_t> offsets{0};
    std::vector<ir::NodeId> members;
    for (const auto& group : inventory->site.accepted_member_origins) {
      if (group.empty()) {
        failure = "variant has an empty accepted source group";
        return {};
      }
      members.insert(members.end(), group.begin(), group.end());
      offsets.push_back(static_cast<std::uint32_t>(members.size()));
    }
    auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
    auto region = co::MakeComparisonRegionBoundary({view, offsets, members}, funder);
    if (!region) {
      failure = "variant region boundary: " + region.error().detail;
      return {};
    }
    auto full_boundary = co::MakeComparisonBoundary(view, funder);
    if (!full_boundary) {
      failure = "variant full boundary: " + full_boundary.error().detail;
      return {};
    }
    const auto offset_for = [](const planner::StepPlan& plan,
                               ir::TensorId original) -> std::optional<std::uint64_t> {
      if (!original.valid() || original.v >= plan.tensors.size()) return {};
      const auto mapped = plan.tensors[original.v];
      if (!mapped.valid()) return {};
      const auto root = plan.graph.root(mapped);
      const auto found = std::ranges::find(plan.placement.offsets, root,
                                           &std::pair<ir::TensorId, std::uint64_t>::first);
      return found == plan.placement.offsets.end() ? std::nullopt
                                                   : std::optional<std::uint64_t>{found->second};
    };
    auto* activation = static_cast<std::byte*>(life_->test_activation.pointer);
    std::vector<kn::ComparisonBindingRange> ranges;
    for (const auto& fact : external) {
      auto base = life_->test_weights.address();
      if (!fact.part) {
        const auto offset = offset_for(inventory->incumbent.plan, fact.original_root);
        if (!offset) {
          failure = "variant input has no actual planned placement";
          return {};
        }
        base = reinterpret_cast<std::uintptr_t>(activation) + *offset;
      }
      ranges.push_back({.original_root = fact.original_root,
                        .part = fact.part,
                        .allocation_class = fact.allocation_class,
                        .allocation_bytes = fact.allocation_bytes,
                        .offset = fact.offset,
                        .bytes = fact.bytes,
                        .alignment = fact.alignment,
                        .base = base,
                        .resource_part_identity = fact.resource_part_identity});
    }
    kn::Addresses addresses{};
    addresses.weight = [&](const ir::WeightRef&, llmp::format::PartRole) {
      return life_->test_weights.address();
    };
    addresses.activations = reinterpret_cast<std::uintptr_t>(activation);
    addresses.activation_bytes = 1U << 20U;
    addresses.activation_alignment = 256;
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .bindings = ranges,
                                  .expected_boundary = &*full_boundary};
    auto& owned = inventory->incumbent;
    if (!owned.final_origins) {
      failure = "variant has no retained final-origin map";
      return {};
    }
    kn::ComparisonRegionCapture capture{.seed = &*region,
                                        .origins = &*owned.final_origins,
                                        .final_nodes = &owned.final_node_origins};
    kn::LowerInput input(owned.plan, step, shape, addresses, 0, {}, &request, &capture);
    auto first = life_->backend->Lower(input);
    if (!first) {
      failure = "variant first Lower: " + first.error();
      return {};
    }
    auto second = life_->backend->Lower(input);
    if (!second) {
      failure = "variant second Lower: " + second.error();
      return {};
    }
    const auto first_certificate = (*first)->ProjectComparisonRegionCertificate(
        *region, kn::ComparisonCertificatePhase::kDescriptor);
    const auto second_certificate = (*second)->ProjectComparisonRegionCertificate(
        *region, kn::ComparisonCertificatePhase::kDescriptor);
    if (first_certificate.status != kn::ComparisonRegionStatus::kClosed ||
        second_certificate.status != kn::ComparisonRegionStatus::kClosed)
      return VariantResult{.key = {},
                           .seal = first_certificate.seal,
                           .closed = false,
                           .reason = std::string(first_certificate.reason) + " / " +
                                     std::string(second_certificate.reason)};
    auto key = en::RecipeComparisonKeyForReadyCertificates(
        *region, ir::Fused(ir::FusedKind::kNormGate), shape.phase,
        recipe.registry->SelectionDigest(), first_certificate.seal, second_certificate.seal);
    if (!key) {
      failure = "variant actual descriptor pair did not produce a ready key";
      return {};
    }
    return VariantResult{.key = *key, .seal = first_certificate.seal, .closed = true, .reason = {}};
  };

  ASSERT_TRUE(life_->test_activation.Allocate(1U << 20U));
  ASSERT_TRUE(life_->test_weights.Allocate(kWidth * sizeof(float)));
  const auto sigmoid = observe(1.0e-6F, ir::NormGateFn::kSigmoid, GateLayout::kRank4);
  ASSERT_TRUE(sigmoid) << failure;
  const auto different_epsilon = observe(1.0e-4F, ir::NormGateFn::kSigmoid, GateLayout::kRank4);
  ASSERT_TRUE(different_epsilon) << failure;
  const auto different_gate = observe(1.0e-6F, ir::NormGateFn::kSilu, GateLayout::kRank4);
  ASSERT_TRUE(different_gate) << failure;
  const auto reshaped_gate = observe(1.0e-6F, ir::NormGateFn::kSigmoid, GateLayout::kReshapeRank4);
  ASSERT_TRUE(reshaped_gate) << failure;
  const auto offset_gate = observe(1.0e-6F, ir::NormGateFn::kSigmoid, GateLayout::kOffsetView);
  ASSERT_TRUE(offset_gate) << failure;
  const auto unsupported_encoding =
      observe(1.0e-6F, ir::NormGateFn::kSigmoid, GateLayout::kRank4, llmp::format::Encoding::kF16);
  ASSERT_TRUE(sigmoid->closed && different_epsilon->closed && different_gate->closed &&
              reshaped_gate->closed && offset_gate->closed);
  EXPECT_FALSE(unsupported_encoding);
  EXPECT_NE(failure.find("no implementation computes"), std::string::npos) << failure;
  EXPECT_NE(failure.find("unary f16"), std::string::npos) << failure;
  EXPECT_NE(sigmoid->seal.physical_recipe, different_epsilon->seal.physical_recipe);
  EXPECT_NE(sigmoid->seal.physical_recipe, different_gate->seal.physical_recipe);
  EXPECT_NE(sigmoid->seal.physical_recipe, reshaped_gate->seal.physical_recipe);
  EXPECT_NE(sigmoid->seal.physical_recipe, offset_gate->seal.physical_recipe);
  ASSERT_TRUE(sigmoid->key && different_epsilon->key && different_gate->key && reshaped_gate->key &&
              offset_gate->key);
  EXPECT_NE(*sigmoid->key, *different_epsilon->key);
  EXPECT_NE(*sigmoid->key, *different_gate->key);
  EXPECT_NE(*sigmoid->key, *reshaped_gate->key);
  EXPECT_NE(*sigmoid->key, *offset_gate->key);
}

TEST_F(GgmlComparisonLowerForGpu, NestedHcPostNormCapturesActualDefaultSchedules) {
  for (const bool component_ports : {false, true}) {
    SCOPED_TRACE(component_ports ? "ComponentPorts" : "TokenFeatures");
    if (component_ports) ASSERT_TRUE(life_->backend->LendLibrary(&life_->resources.library()));
    constexpr std::uint64_t kWidth = 4096;
    constexpr std::uint64_t kStreams = 4;
    constexpr std::uint64_t kTokens = 2;
    constexpr float kEpsilon = 1.0e-6F;
    ir::Builder builder;
    builder.SetOwner(0);
    const auto y =
        builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, kTokens})));
    const auto x = builder.Input(
        ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, kStreams, kTokens})));
    const auto post_weights =
        builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kStreams, kTokens})));
    const auto comb = builder.Input(
        ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kStreams, kStreams, kTokens})));
    const auto post = builder.HcPost(y, x, post_weights, comb, {.streams = kStreams});
    const auto flat = builder.Reshape(post, ir::ShapeOf({kWidth * kStreams, kTokens}));
    const auto flat_norm = builder.RmsNorm(flat, kEpsilon);
    const auto consumer = builder.RmsNorm(post, kEpsilon);
    builder.Output(post);
    builder.Output(flat_norm);
    builder.Output(consumer);
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);

    co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
    step.inputs = {{.tensor = y, .kind = co::InputKind::kFeatures},
                   {.tensor = x, .kind = co::InputKind::kFeatures},
                   {.tensor = post_weights, .kind = co::InputKind::kFeatures},
                   {.tensor = comb, .kind = co::InputKind::kFeatures}};
    const auto io = co::GraphIO(step);
    const auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    const planner::StepShape shape{.stage = 0,
                                   .segments = {{.slot = 0, .rows = kTokens, .outputs = 0}}};
    constexpr std::uint64_t kActivationBytes = 1U << 20U;
    const planner::PlanOptions options{
        .capturable = true, .within = recipe.within, .shape = &shape};
    auto seed_plan =
        planner::Plan(step.graph.Clone(), *recipe.registry, recipe.device, {}, options);
    ASSERT_TRUE(seed_plan) << (seed_plan ? "" : seed_plan.error().detail);
    ASSERT_EQ(seed_plan->applied.size(), 1U);
    EXPECT_EQ(seed_plan->applied.front().pattern, "hc_post_flat_norm");
    ASSERT_LE(seed_plan->placement.extent, kActivationBytes);
    CudaBuffer activation;
    ASSERT_TRUE(activation.Allocate(kActivationBytes));
    kn::Addresses addresses{};
    addresses.activations = activation.address();
    addresses.activation_bytes = kActivationBytes;
    addresses.activation_alignment = 256;
    const auto placement_offset = [](const planner::StepPlan& plan,
                                     ir::TensorId root) -> std::optional<std::uint64_t> {
      const auto mapped = plan.tensors.at(root.v);
      if (!mapped.valid()) return std::nullopt;
      const auto planned_root = plan.graph.root(mapped);
      const auto found = std::ranges::find(plan.placement.offsets, planned_root,
                                           &std::pair<ir::TensorId, std::uint64_t>::first);
      if (found == plan.placement.offsets.end()) return std::nullopt;
      return found->second;
    };
    const auto* pattern = planner::FindPattern("hc_post_flat_norm");
    ASSERT_NE(pattern, nullptr);
    const std::array members{step.graph.producer(post), step.graph.producer(flat),
                             step.graph.producer(flat_norm)};
    const std::vector<planner::ComparisonOriginSet> origins{
        planner::ComparisonOriginSet{members[0]}, planner::ComparisonOriginSet{members[1]},
        planner::ComparisonOriginSet{members[2]}};
    const planner::ComparisonSiteSelector selector{.source_graph_identity = step.graph.identity(),
                                                   .kind = planner::ComparisonSiteKind::kRewrite,
                                                   .pattern = "hc_post_flat_norm",
                                                   .version = pattern->version,
                                                   .member_origins = origins,
                                                   .member_owner_segments = {},
                                                   .boundary_ports = {}};
    const std::array roots{y, x, post_weights, comb};
    std::array<co::ComparisonExternalBinding, roots.size()> external{};
    std::array<std::uint64_t, roots.size()> bytes{};
    std::array<std::uint64_t, roots.size()> offsets{};
    for (std::size_t i = 0; i < roots.size(); ++i) {
      const auto size = ir::StorageBytes(step.graph.tensor(roots[i]));
      ASSERT_TRUE(size);
      bytes[i] = *size;
      const auto offset = placement_offset(*seed_plan, roots[i]);
      ASSERT_TRUE(offset);
      offsets[i] = *offset;
      external[i] = {.original_root = roots[i],
                     .allocation_class = static_cast<std::uint32_t>(i + 1),
                     .allocation_bytes = *size,
                     .offset = 0,
                     .bytes = *size,
                     .alignment = 256};
    }
    auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
    auto inventory = planner::BoundComparison({io, {}, 0, external}, selector, funder,
                                              *recipe.registry, recipe.device, {}, options);
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    auto alternative = planner::MakeAlternative(*inventory);
    ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
    auto planned = planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                            recipe.device, {}, options);
    ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
    ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
    EXPECT_EQ(inventory->incumbent.plan.applied.front().pattern, "hc_post_flat_norm");
    EXPECT_TRUE(std::ranges::none_of(planned->plan.applied, [](const auto& applied) {
      return applied.pattern == "hc_post_flat_norm";
    }));

    const auto schedule = [&](const planner::StepPlan& plan) {
      std::vector<std::string> names;
      for (const auto& selected : plan.steps)
        names.emplace_back(recipe.registry->at(selected.implementation).name);
      return names;
    };
    const auto incumbent_schedule = schedule(inventory->incumbent.plan);
    const auto alternative_schedule = schedule(planned->plan);
    const auto names = [](const std::vector<std::string>& values) {
      std::string result;
      for (const auto& value : values) {
        if (!result.empty()) result += ", ";
        result += value;
      }
      return result;
    };
    SCOPED_TRACE(testing::Message()
                 << "incumbent schedule=[" << names(incumbent_schedule)
                 << "], alternative schedule=[" << names(alternative_schedule) << "]");
    EXPECT_EQ(incumbent_schedule,
              (std::vector<std::string>{kg::kDsv4HcPostNormF16Name, "ggml.rms_norm"}));
    EXPECT_EQ(alternative_schedule, (std::vector<std::string>{std::string(kg::kHcPostName),
                                                              "ggml.rms_norm", "ggml.rms_norm"}));
    ASSERT_TRUE(std::ranges::find(incumbent_schedule, kg::kDsv4HcPostNormF16Name) !=
                incumbent_schedule.end());
    ASSERT_TRUE(std::ranges::find(alternative_schedule, kg::kHcPostName) !=
                alternative_schedule.end());

    std::vector<std::uint32_t> member_offsets{0};
    std::vector<ir::NodeId> region_members;
    for (const auto& group : inventory->site.accepted_member_origins) {
      region_members.insert(region_members.end(), group.begin(), group.end());
      member_offsets.push_back(static_cast<std::uint32_t>(region_members.size()));
    }
    auto region = co::MakeComparisonRegionBoundary(
        {{io, {}, 0, external}, member_offsets, region_members}, funder);
    ASSERT_TRUE(region) << (region ? "" : region.error().detail);
    const std::array<const planner::StepPlan*, 2> plans{&inventory->incumbent.plan, &planned->plan};
    CudaBuffer pool, persistent;
    ASSERT_TRUE(pool.Allocate(1U << 20U));
    const auto persistent_bytes = life_->backend->persistent_bytes();
    ASSERT_GT(persistent_bytes, 0U);
    ASSERT_TRUE(persistent.Allocate(persistent_bytes));
    auto launch =
        life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(1U << 20U)},
                               {.base = persistent.address(), .bytes = persistent_bytes});
    ASSERT_TRUE(launch) << (launch ? "" : launch.error());
    std::array<std::unique_ptr<kn::StepProgram>, plans.size()> programs;
    std::array<std::unique_ptr<kn::Recorded>, plans.size()> recordings;
    for (const auto* plan : plans) {
      std::vector<kn::ComparisonBindingRange> ranges;
      for (std::size_t i = 0; i < roots.size(); ++i) {
        const auto offset = placement_offset(*plan, roots[i]);
        ASSERT_TRUE(offset);
        ranges.push_back({.original_root = roots[i],
                          .allocation_class = static_cast<std::uint32_t>(i + 1),
                          .allocation_bytes = bytes[i],
                          .offset = 0,
                          .bytes = bytes[i],
                          .alignment = 256,
                          .base = activation.address() + *offset});
      }
      kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                    .bindings = ranges,
                                    .expected_boundary = &inventory->source_boundary};
      const auto& owned = plan == plans[0] ? inventory->incumbent : *planned;
      auto final_origins = owned.final_origins->Clone(funder);
      ASSERT_TRUE(final_origins);
      kn::ComparisonRegionCapture capture{
          .seed = &*region, .origins = &*final_origins, .final_nodes = &owned.final_node_origins};
      kn::LowerInput lower_input(
          *plan, io, addresses,
          component_ports ? std::nullopt : std::optional<kn::TokenLowerInput>{{step, shape, 0, {}}},
          &request, &capture);
      auto program = life_->backend->Lower(lower_input);
      ASSERT_TRUE(program) << (program ? "" : program.error());
      const auto seal = (*program)->comparison_recipe();
      ASSERT_TRUE(seal);
      ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified)
          << "missing=" << static_cast<int>(seal->missing);
      ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
      EXPECT_EQ(seal->pool_bytes, 0U);
      EXPECT_EQ(seal->private_pool_bytes, 0U);
      const auto arm = static_cast<std::size_t>(plan == plans[0] ? 0 : 1);
      const auto descriptor = (*program)->ProjectComparisonRegionDescriptor(*region);
      ASSERT_EQ(descriptor.status, kn::ComparisonRegionStatus::kClosed) << descriptor.reason;
      if (component_ports)
        for (const auto& slot : (*program)->inputs())
          EXPECT_EQ(slot.kind, kn::InputKind::kStepBytes);
      programs[arm] = std::move(*program);
      ASSERT_TRUE(programs[arm]->Bind(**launch));
      const auto bound = programs[arm]->comparison_recipe();
      ASSERT_TRUE(bound);
      ASSERT_EQ(bound->state, kn::ComparisonProofState::kBoundVerified);
      ASSERT_EQ(bound->missing, kn::ComparisonMissing::kNone);
      const auto remainder = programs[arm]->CompareComparisonRemainder(
          *programs[arm], *region, kn::ComparisonCertificatePhase::kBound);
      ASSERT_EQ(remainder.status, kn::ComparisonRemainderStatus::kSame) << remainder.reason;
      const auto projected = programs[arm]->ProjectComparisonRegion(*region);
      ASSERT_EQ(projected.status, kn::ComparisonRegionStatus::kClosed) << projected.reason;
    }
    std::vector<kn::ComparisonBindingRange> valid_ranges;
    for (std::size_t i = 0; i < roots.size(); ++i) {
      const auto offset = placement_offset(*plans[0], roots[i]);
      ASSERT_TRUE(offset);
      valid_ranges.push_back({.original_root = roots[i],
                              .allocation_class = static_cast<std::uint32_t>(i + 1),
                              .allocation_bytes = bytes[i],
                              .offset = 0,
                              .bytes = bytes[i],
                              .alignment = 256,
                              .base = activation.address() + *offset});
    }
    auto wrong_alias = valid_ranges;
    wrong_alias[1].allocation_class = 1;
    kn::ComparisonRequest wrong_alias_request{.mode = kn::ComparisonRunMode::kFull,
                                              .bindings = wrong_alias,
                                              .expected_boundary = &inventory->source_boundary};
    kn::LowerInput wrong_alias_input(*plans[0], step, shape, addresses, 0, {},
                                     &wrong_alias_request);
    auto wrong_alias_program = life_->backend->Lower(wrong_alias_input);
    EXPECT_FALSE(wrong_alias_program);

    const std::array wrong_output_order{consumer, flat_norm, post};
    const co::GraphBuildIO wrong_io{step.graph, io.inputs, wrong_output_order, io.weights};
    kn::ComparisonRequest wrong_order_request{.mode = kn::ComparisonRunMode::kFull,
                                              .bindings = valid_ranges,
                                              .expected_boundary = &inventory->source_boundary};
    kn::LowerInput wrong_order_input(*plans[0], wrong_io, addresses, std::nullopt,
                                     &wrong_order_request);
    auto wrong_order_program = life_->backend->Lower(wrong_order_input);
    EXPECT_FALSE(wrong_order_program);

    auto native = life_->node.execution().Submission(life_->node.stream(0));
    ASSERT_TRUE(native) << (native ? "" : native.error().detail);
    const std::array<std::size_t, roots.size()> input_sizes{
        static_cast<std::size_t>(kWidth * kTokens),
        static_cast<std::size_t>(kWidth * kStreams * kTokens),
        static_cast<std::size_t>(kStreams * kTokens),
        static_cast<std::size_t>(kStreams * kStreams * kTokens)};
    std::array<std::vector<float>, roots.size()> host_inputs;
    for (std::size_t port = 0; port < roots.size(); ++port) {
      host_inputs[port].resize(input_sizes[port]);
      for (std::size_t i = 0; i < host_inputs[port].size(); ++i) {
        const auto centered = static_cast<float>(static_cast<int>(i % 19) - 9);
        host_inputs[port][i] = port == 2   ? 0.5F + 0.025F * centered
                               : port == 3 ? 0.2F + 0.01F * centered
                                           : 0.005F * centered;
      }
    }
    auto reference = reference::Executor::Create(step.graph);
    ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
    const auto run_reference =
        [&]() -> std::expected<std::array<std::vector<float>, 3>, reference::ExecFailure> {
      for (std::size_t i = 0; i < roots.size(); ++i)
        if (auto bound = reference->BindValues(roots[i], host_inputs[i]); !bound)
          return std::unexpected(bound.error());
      if (auto ran = reference->Run(); !ran) return std::unexpected(ran.error());
      std::array<std::vector<float>, 3> values;
      for (std::size_t i = 0; i < values.size(); ++i) {
        auto result = reference->Values(step.graph.outputs()[i]);
        if (!result) return std::unexpected(result.error());
        values[i] = std::move(*result);
      }
      return values;
    };
    const auto copy_inputs = [&](const planner::StepPlan& plan) {
      for (std::size_t i = 0; i < roots.size(); ++i) {
        const auto offset = placement_offset(plan, roots[i]);
        if (!offset) return false;
        const auto destination = activation.address() + *offset;
        if (cudaMemcpy(reinterpret_cast<void*>(destination), host_inputs[i].data(),
                       host_inputs[i].size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess)
          return false;
      }
      return true;
    };
    const auto compare_outputs = [&](std::size_t arm,
                                     const std::array<std::vector<float>, 3>& expected) {
      const auto& plan = *plans[arm];
      for (std::size_t port = 0; port < expected.size(); ++port) {
        const auto id = step.graph.outputs()[port];
        const auto offset = placement_offset(plan, id);
        ASSERT_TRUE(offset);
        std::vector<float> actual(expected[port].size());
        ASSERT_EQ(
            cudaMemcpy(actual.data(), reinterpret_cast<const void*>(activation.address() + *offset),
                       actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(actual.size(), expected[port].size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
          ASSERT_TRUE(std::isfinite(actual[i]) && std::isfinite(expected[port][i]))
              << port << ":" << i;
          // Match reference_exec_test.cc's HC expansion bound: 1e-5 * (1 + |reference|).
          EXPECT_LE(std::abs(static_cast<double>(actual[i]) - expected[port][i]),
                    1.0e-5 * (1.0 + std::abs(static_cast<double>(expected[port][i]))))
              << "output " << port << " element " << i;
        }
      }
    };
    for (std::size_t arm = 0; arm < programs.size(); ++arm) {
      auto recording =
          programs[arm]->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
            return programs[arm]->Run(**launch, *native);
          });
      ASSERT_TRUE(recording) << (recording ? "" : recording.error().detail);
      recordings[arm] = std::move(*recording);
    }
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
      if (frame != 0) {
        for (std::size_t port = 0; port < host_inputs.size(); ++port)
          for (std::size_t i = 0; i < host_inputs[port].size(); ++i)
            host_inputs[port][i] += static_cast<float>(port + 1) * 0.003F;
      }
      auto expected = run_reference();
      ASSERT_TRUE(expected) << (expected ? "" : expected.error().detail);
      for (std::size_t arm = 0; arm < recordings.size(); ++arm) {
        for (const auto output : step.graph.outputs()) {
          const auto offset = placement_offset(*plans[arm], output);
          const auto output_bytes = ir::StorageBytes(step.graph.tensor(output));
          ASSERT_TRUE(offset && output_bytes);
          ASSERT_EQ(cudaMemset(reinterpret_cast<void*>(activation.address() + *offset), 0xff,
                               static_cast<std::size_t>(*output_bytes)),
                    cudaSuccess);
        }
        ASSERT_TRUE(copy_inputs(*plans[arm]));
        ASSERT_TRUE((*launch)->Launch(*native, *recordings[arm]));
        ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
        compare_outputs(arm, *expected);
      }
    }
    for (auto& recording : recordings) recording.reset();
    for (auto& program : programs) program.reset();
    launch->reset();
    life_->backend->LaunchReleased();
  }
}

TEST_F(GgmlComparisonLowerForGpu, RmsNormMul1024x1HasCompleteDescriptorRecipe) {
  constexpr auto kFormat = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.stage = 0, .component = 0, .role = 0, .layer = 0};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto rows = builder.Input(ir::Dense(kFormat, ir::ShapeOf({1024, 1})));
  const auto weight = builder.Weight(ir::Dense(kFormat, ir::ShapeOf({1024})), weight_ref);
  const auto norm = builder.RmsNorm(rows, 1.0e-6F);
  builder.Output(builder.Mul(norm, weight));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  const std::array<co::StepWeight, 1> weights{co::StepWeight{
      .tensor = weight, .ref = weight_ref, .parts = {{"weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), weights};
  kn::Addresses addresses{};
  auto backend = life_->backend->Backend();
  const auto& recipe_backend = backend.owner ? backend.owner->backend() : backend;
  llmp::planner::Policy policy{};
  policy.enabled = {"rms_norm_mul"};
  auto plan = llmp::planner::Plan(graph->Clone(), *recipe_backend.registry,
                                  {.device_class = "sm_121", .arch = 121}, policy);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
  ASSERT_EQ(plan->applied.size(), 1U);
  ASSERT_EQ(plan->applied[0].pattern, "rms_norm_mul");
  bool fused_node = false;
  for (const auto& node : plan->graph.nodes()) {
    if (node.key == ir::Fused(ir::FusedKind::kRmsNormMul)) fused_node = true;
  }
  ASSERT_TRUE(fused_node);
  bool fused_implementation = false;
  for (const auto& selected : plan->steps) {
    if (recipe_backend.registry->at(selected.implementation).name == "ggml.rms_norm_mul.fused")
      fused_implementation = true;
  }
  ASSERT_TRUE(fused_implementation);

  const auto row_bytes = ir::StorageBytes(graph->tensor(rows));
  const auto weight_bytes = ir::StorageBytes(graph->tensor(weight));
  ASSERT_TRUE(row_bytes && weight_bytes);
  const std::array<kn::ComparisonBindingRange, 2> bindings{
      kn::ComparisonBindingRange{.original_root = rows,
                                 .allocation_class = 1,
                                 .allocation_bytes = *row_bytes,
                                 .bytes = *row_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = weight,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 2,
                                 .allocation_bytes = *weight_bytes,
                                 .bytes = *weight_bytes,
                                 .alignment = 16}};
  const std::array<co::ComparisonExternalBinding, 2> external{
      co::ComparisonExternalBinding{.original_root = rows,
                                    .allocation_class = 1,
                                    .allocation_bytes = *row_bytes,
                                    .bytes = *row_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = *weight_bytes,
                                    .bytes = *weight_bytes,
                                    .alignment = 16}};
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, external},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = bindings, .expected_boundary = &*boundary};
  kn::LowerInput lower_input(*plan, io, addresses, std::nullopt, &request);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  const auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
}

TEST_F(GgmlComparisonLowerForGpu, SectionsNormRopeBothPlansCaptureChangedIndependentPositions) {
  for (unsigned fixture = 0; fixture < 6; ++fixture) {
    SCOPED_TRACE(fixture);
    if (fixture) ASSERT_TRUE(life_->backend->LendLibrary(&life_->resources.library()));
    const bool bf16 = (fixture & 1U) != 0;
    const bool neox = fixture >= 2 && fixture < 4;
    const bool single = fixture >= 4;
    const std::uint32_t dim = neox || single ? 128U : 64U;
    const std::vector<std::uint32_t> dims = single ? std::vector<std::uint32_t>{128}
                                                   : (neox ? std::vector<std::uint32_t>{16, 56, 56}
                                                           : std::vector<std::uint32_t>{32, 32});
    const std::uint32_t kHeads = single ? 1U : 3U;
    const std::uint32_t kRows = single ? 2U : 5U;
    const ir::WeightRef gamma_ref{.role = 1};
    ir::Builder b;
    b.SetOwner(0);
    const auto x =
        b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({dim, kHeads, kRows})));
    const auto gamma =
        b.Weight(ir::Dense(bf16 ? llmp::format::Encoding::kBF16 : llmp::format::Encoding::kF32,
                           ir::ShapeOf({dim})),
                 gamma_ref, "gamma");
    std::vector<ir::TensorId> positions;
    for (std::size_t section = 0; section < dims.size(); ++section)
      positions.push_back(b.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({kRows}))));
    auto value =
        b.Mul(b.RmsNorm(x, 1e-6F), bf16 ? b.Convert(gamma, llmp::format::Encoding::kF32) : gamma);
    std::uint32_t offset = 0;
    for (std::size_t section = 0; section < dims.size(); ++section) {
      value = b.Rope(value, positions[section], {},
                     {.mode = neox ? ir::RopeMode::kNeox : ir::RopeMode::kNormal,
                      .dims = dims[section],
                      .offset = offset,
                      .base = 10000.0F});
      offset += dims[section];
    }
    b.Output(value);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    co::StepGraph source{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
    source.inputs.push_back({.tensor = x, .kind = co::InputKind::kStream});
    for (auto p : positions) source.inputs.push_back({.tensor = p, .kind = co::InputKind::kSide});
    source.weights.push_back(
        {.tensor = gamma, .ref = gamma_ref, .parts = {{"gamma", llmp::format::PartRole::kData}}});
    std::vector<co::ComparisonExternalBinding> external;
    for (auto tensor : source.graph.inputs()) {
      const auto bytes = *ir::StorageBytes(source.graph.tensor(tensor));
      external.push_back(
          {.original_root = tensor,
           .part = tensor == gamma ? std::optional{llmp::format::PartRole::kData} : std::nullopt,
           .allocation_class = static_cast<std::uint32_t>(external.size() + 1),
           .allocation_bytes = bytes,
           .bytes = bytes,
           .alignment = 16});
    }
    const auto gamma_bytes = *ir::StorageBytes(source.graph.tensor(gamma));
    external.push_back({.original_root = gamma,
                        .part = llmp::format::PartRole::kData,
                        .allocation_class = static_cast<std::uint32_t>(external.size() + 1),
                        .allocation_bytes = gamma_bytes,
                        .bytes = gamma_bytes,
                        .alignment = 16});
    const auto io = co::GraphIO(source);
    const co::ComparisonBoundaryView view{io, {}, 0, external};
    auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
    auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    const llmp::planner::Policy policy{};
    const llmp::planner::PlanOptions options{.capturable = true, .within = recipe.within};
    const std::array filter{ir::Fused(ir::FusedKind::kRmsNormMulRopeSections)};
    auto discovered = llmp::planner::DiscoverComparisonPlan(
        view, filter, false, funder, *recipe.registry, recipe.device, policy, options);
    ASSERT_TRUE(discovered) << (discovered ? "" : discovered.error().detail);
    const auto ticket = std::ranges::find_if(discovered->discovery.tickets(), [](const auto& t) {
      return t.kind == llmp::planner::ComparisonSiteKind::kRewrite &&
             t.pattern == "rms_norm_mul_rope_sections";
    });
    ASSERT_NE(ticket, discovered->discovery.tickets().end());
    llmp::planner::ComparisonSiteSelector selector{.source_graph_identity = source.graph.identity(),
                                                   .kind = ticket->kind,
                                                   .pattern = std::string(ticket->pattern),
                                                   .version = ticket->version,
                                                   .member_origins = {},
                                                   .member_owner_segments = ticket->owner_segments,
                                                   .boundary_ports = ticket->boundary_ports};
    for (std::size_t member = 0; member + 1 < ticket->member_offsets.size(); ++member)
      selector.member_origins.emplace_back(
          ticket->member_origins.begin() + ticket->member_offsets[member],
          ticket->member_origins.begin() + ticket->member_offsets[member + 1]);
    auto inventory = llmp::planner::BoundComparison(view, selector, funder, *recipe.registry,
                                                    recipe.device, policy, options);
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    auto own = llmp::planner::MakeAlternative(*inventory);
    ASSERT_TRUE(own);
    auto alternative = llmp::planner::PlanAlternative(*inventory, std::move(*own), *recipe.registry,
                                                      recipe.device, policy, options);
    ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
    std::vector<std::uint32_t> offsets{0};
    std::vector<ir::NodeId> members;
    for (const auto& group : inventory->site.accepted_member_origins) {
      members.insert(members.end(), group.begin(), group.end());
      offsets.push_back(static_cast<std::uint32_t>(members.size()));
    }
    auto region = co::MakeComparisonRegionBoundary({view, offsets, members}, funder);
    ASSERT_TRUE(region);
    constexpr std::size_t kArena = 1U << 20U;
    CudaBuffer activation, weights, pool, persistent;
    ASSERT_TRUE(activation.Allocate(kArena));
    ASSERT_TRUE(weights.Allocate(dim * (bf16 ? 2U : 4U)));
    ASSERT_TRUE(pool.Allocate(kArena));
    ASSERT_TRUE(persistent.Allocate(life_->backend->persistent_bytes()));
    auto launch = life_->backend->Launch(
        {.base = pool.address(), .size = llmp::base::Bytes(kArena)},
        {.base = persistent.address(), .bytes = life_->backend->persistent_bytes()});
    ASSERT_TRUE(launch);
    auto native = life_->execution->Submission(life_->node.stream(0));
    ASSERT_TRUE(native);
    const auto stream = static_cast<cudaStream_t>(native->handle);
    std::vector<float> gamma_values(dim);
    std::vector<std::uint16_t> gamma_bits(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      gamma_values[i] = 0.75F + static_cast<float>(i % 9) / 16.0F;
      std::memcpy(&gamma_bits[i], reinterpret_cast<const std::byte*>(&gamma_values[i]) + 2, 2);
    }
    const void* gamma_data = bf16 ? static_cast<const void*>(gamma_bits.data())
                                  : static_cast<const void*>(gamma_values.data());
    ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(weights.address()), gamma_data,
                              dim * (bf16 ? 2U : 4U), cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    auto cpu = reference::Executor::Create(source.graph);
    ASSERT_TRUE(cpu);
    ASSERT_TRUE(cpu->BindValues(gamma, gamma_values));
    const std::array<const llmp::planner::OwnedComparisonPlan*, 2> plans{&inventory->incumbent,
                                                                         &*alternative};
    for (const auto* owned : plans) {
      const auto& plan = owned->plan;
      ASSERT_LE(plan.placement.extent, kArena);
      kn::Addresses addresses{
          .weight = [&](const ir::WeightRef&, llmp::format::PartRole) { return weights.address(); },
          .weight_rows = {},
          .state = {},
          .save = {},
          .activations = activation.address(),
          .activation_bytes = kArena,
          .activation_alignment = 256};
      std::vector<kn::ComparisonBindingRange> ranges;
      for (const auto& fact : inventory->source_boundary.external()) {
        const auto original = inventory->source_boundary.roots()[fact.canonical_root].original_root;
        std::uint64_t base = weights.address();
        if (!fact.part) {
          const auto mapped = plan.tensors[original.v];
          const auto root = plan.graph.root(mapped);
          const auto place = std::ranges::find_if(plan.placement.offsets, [&](const auto& p) {
            return plan.graph.root(p.first) == root;
          });
          ASSERT_NE(place, plan.placement.offsets.end());
          base = activation.address() + place->second;
        }
        ranges.push_back({.original_root = original,
                          .part = fact.part,
                          .allocation_class = fact.canonical_allocation_class,
                          .allocation_bytes = fact.allocation_bytes,
                          .offset = fact.offset,
                          .bytes = fact.bytes,
                          .alignment = fact.alignment,
                          .base = base,
                          .resource_part_identity = fact.resource_part_identity});
      }
      kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                    .bindings = ranges,
                                    .expected_boundary = &inventory->source_boundary};
      auto origins = owned->final_origins->Clone(funder);
      ASSERT_TRUE(origins);
      kn::ComparisonRegionCapture capture{
          .seed = &*region, .origins = &*origins, .final_nodes = &owned->final_node_origins};
      kn::LowerInput lower(plan, io, addresses, std::nullopt, &request, &capture);
      auto program = life_->backend->Lower(lower);
      ASSERT_TRUE(program) << (program ? "" : program.error());
      const auto whole_descriptor = (*program)->comparison_recipe();
      ASSERT_TRUE(whole_descriptor);
      ASSERT_EQ(whole_descriptor->state, kn::ComparisonProofState::kDescriptorVerified);
      ASSERT_EQ(whole_descriptor->missing, kn::ComparisonMissing::kNone);
      const auto descriptor = (*program)->ProjectComparisonRegionDescriptor(*region);
      ASSERT_EQ(descriptor.status, kn::ComparisonRegionStatus::kClosed) << descriptor.reason;
      ASSERT_TRUE((*program)->Bind(**launch));
      const auto whole_bound = (*program)->comparison_recipe();
      ASSERT_TRUE(whole_bound);
      ASSERT_EQ(whole_bound->state, kn::ComparisonProofState::kBoundVerified);
      ASSERT_EQ(whole_bound->missing, kn::ComparisonMissing::kNone);
      const auto remainder = (*program)->CompareComparisonRemainder(
          **program, *region, kn::ComparisonCertificatePhase::kBound);
      ASSERT_EQ(remainder.status, kn::ComparisonRemainderStatus::kSame) << remainder.reason;
      // A projection borrows program-owned tables; acquire the value ports
      // after the remainder analysis has finished rebuilding those tables.
      const auto bound = (*program)->ProjectComparisonRegion(*region);
      ASSERT_EQ(bound.status, kn::ComparisonRegionStatus::kClosed) << bound.reason;
      auto recorded =
          (*program)->Record(**launch, *native, [&] { return (*program)->Run(**launch, *native); });
      ASSERT_TRUE(recorded);
      std::printf(
          "SectionsNormRope dim=%u heads=%u rows=%u sections=%zu gamma=%s mode=%s "
          "arm=%s kernels=%llu calls=",
          dim, kHeads, kRows, dims.size(), bf16 ? "BF16" : "F32", neox ? "NEOX" : "NORMAL",
          owned == plans[0] ? "incumbent" : "alternative",
          static_cast<unsigned long long>((*recorded)->nodes()));
      for (const auto& selected : plan.steps) {
        const auto name = recipe.registry->at(selected.implementation).name;
        std::printf("%.*s ", static_cast<int>(name.size()), name.data());
      }
      std::printf("\n");
      std::vector<float> previous;
      for (unsigned frame = 0; frame < 2; ++frame) {
        std::vector<float> features(dim * kHeads * kRows);
        for (std::size_t i = 0; i < features.size(); ++i)
          features[i] =
              static_cast<float>(static_cast<int>((i * 7 + frame * 11) % 47) - 23) / 32.0F;
        ASSERT_TRUE(cpu->BindValues(x, features));
        std::vector<std::vector<std::int32_t>> host_positions(dims.size());
        for (std::size_t section = 0; section < dims.size(); ++section) {
          for (unsigned row = 0; row < kRows; ++row)
            host_positions[section].push_back(
                static_cast<std::int32_t>(row + section * 13 + frame * (section + 2)));
          const std::vector<std::int64_t> cpu_positions(host_positions[section].begin(),
                                                        host_positions[section].end());
          ASSERT_TRUE(cpu->BindInts(positions[section], cpu_positions));
        }
        ASSERT_TRUE(cpu->Run());
        const auto expected = cpu->Values(value);
        ASSERT_TRUE(expected);
        ASSERT_EQ(
            cudaMemsetAsync(reinterpret_cast<void*>(activation.address()), 0xff, kArena, stream),
            cudaSuccess);
        ASSERT_EQ((*program)->inputs().size(), positions.size() + 1);
        for (const auto& slot : (*program)->inputs()) {
          ASSERT_EQ(slot.kind, kn::InputKind::kStepBytes);
          ASSERT_EQ(slot.bytes, slot.tensor == x ? features.size() * sizeof(float)
                                                 : kRows * sizeof(std::int32_t));
          const void* payload = features.data();
          if (slot.tensor != x) {
            const auto position = std::ranges::find(positions, slot.tensor);
            ASSERT_NE(position, positions.end());
            payload = host_positions[static_cast<std::size_t>(position - positions.begin())].data();
          }
          ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), payload,
                                    slot.bytes, cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
        }
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_TRUE((*launch)->Launch(*native, **recorded));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const auto output =
            std::ranges::find(bound.value_ports, kn::ComparisonRegionValueKind::kOutput,
                              &kn::ComparisonRegionValuePort::kind);
        ASSERT_NE(output, bound.value_ports.end());
        ASSERT_EQ(output->bytes, features.size() * sizeof(float));
        std::vector<float> actual(features.size());
        ASSERT_EQ(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(output->address),
                             actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
          ASSERT_TRUE(std::isfinite(actual[i]));
          EXPECT_LE(std::abs(actual[i] - (*expected)[i]) / std::max(std::abs((*expected)[i]), 1.0F),
                    1e-4F);
        }
        if (frame) EXPECT_NE(previous, actual);
        previous = std::move(actual);
      }
    }
    life_->backend->LaunchReleased();
  }
}

TEST_F(GgmlComparisonLowerForGpu, RmsNormMulRopeNeoxFixtureHasCompleteDescriptorRecipe) {
  constexpr auto kFormat = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.role = 1};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto q = builder.Input(ir::Dense(kFormat, ir::ShapeOf({128, 1, 1})));
  const auto weight = builder.Weight(ir::Dense(kFormat, ir::ShapeOf({128})), weight_ref);
  const auto positions = builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({1})));
  builder.Keep(q);
  builder.Keep(positions);
  const llmp::ir::RopeParams rope{
      .mode = ir::RopeMode::kNeox, .dims = 128, .original_context = 4096};
  const auto normalized = builder.RmsNorm(q, 1.0e-6F);
  const auto scaled = builder.Mul(normalized, weight);
  builder.Output(builder.Rope(scaled, positions, {}, rope));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = q, .kind = co::InputKind::kFeatures},
                 {.tensor = positions, .kind = co::InputKind::kPositions}};
  step.weights = {co::StepWeight{.tensor = weight,
                                 .ref = weight_ref,
                                 .parts = {{"norm.weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io = co::GraphIO(step);
  const auto backend = life_->backend->Backend();
  const auto& recipe_backend = backend.owner ? backend.owner->backend() : backend;
  const llmp::planner::Policy policy{};
  auto planned = llmp::planner::Plan(step.graph.Clone(), *recipe_backend.registry,
                                     {.device_class = "sm_121", .arch = 121}, policy);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(planned->applied.size(), 1U);
  ASSERT_EQ(planned->applied[0].pattern, "interleaved_norm_rope");
  bool selected_norm_rope = false;
  for (const auto& selected : planned->steps) {
    if (recipe_backend.registry->at(selected.implementation).name == kg::kGemmaNormRopeName) {
      selected_norm_rope = true;
      ASSERT_EQ(selected.nodes.size(), 1U);
      const auto& fused = planned->graph.node(selected.nodes[0]);
      ASSERT_EQ(fused.key, ir::Fused(ir::FusedKind::kRmsNormMulRope));
      const auto params = fused.Params<ir::NormRopeParams>();
      EXPECT_FLOAT_EQ(params.eps, 1.0e-6F);
      EXPECT_EQ(params.rope.mode, ir::RopeMode::kNeox);
      EXPECT_EQ(params.rope.dims, 128U);
      EXPECT_EQ(params.rope.original_context, 4096U);
    }
  }
  ASSERT_TRUE(selected_norm_rope);

  const auto q_bytes = ir::StorageBytes(step.graph.tensor(q));
  const auto position_bytes = ir::StorageBytes(step.graph.tensor(positions));
  const auto weight_bytes = ir::StorageBytes(step.graph.tensor(weight));
  ASSERT_TRUE(q_bytes && position_bytes && weight_bytes);
  const std::array<kn::ComparisonBindingRange, 3> bindings{
      kn::ComparisonBindingRange{.original_root = q,
                                 .allocation_class = 1,
                                 .allocation_bytes = *q_bytes,
                                 .bytes = *q_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = weight,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 2,
                                 .allocation_bytes = *weight_bytes,
                                 .bytes = *weight_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = positions,
                                 .allocation_class = 3,
                                 .allocation_bytes = *position_bytes,
                                 .bytes = *position_bytes,
                                 .alignment = 16}};
  const std::array<co::ComparisonExternalBinding, 3> external{
      co::ComparisonExternalBinding{.original_root = q,
                                    .allocation_class = 1,
                                    .allocation_bytes = *q_bytes,
                                    .bytes = *q_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = *weight_bytes,
                                    .bytes = *weight_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = positions,
                                    .allocation_class = 3,
                                    .allocation_bytes = *position_bytes,
                                    .bytes = *position_bytes,
                                    .alignment = 16}};
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, external},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = bindings, .expected_boundary = &*boundary};
  const llmp::planner::StepShape shape{.stage = 0,
                                       .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  kn::Addresses addresses{};
  kn::LowerInput lower_input(*planned, step, shape, addresses, 0, {}, &request);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  const auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
}

TEST_F(GgmlComparisonLowerForGpu, MatMulGlu128x1HasCompleteFloatMmvfDescriptorRecipe) {
  constexpr std::uint64_t kWidth = 128;
  const ir::WeightRef gate_ref{.role = 1};
  const ir::WeightRef up_ref{.role = 2};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto input =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, 1})));
  const auto gate = builder.Weight(
      ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({kWidth, kWidth})), gate_ref);
  const auto up = builder.Weight(
      ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({kWidth, kWidth})), up_ref);
  builder.Keep(input);
  const auto gate_product = builder.MatMul(gate, input);
  const auto up_product = builder.MatMul(up, input);
  builder.Output(
      builder.Glu(gate_product, up_product, ir::GluParams{.kind = ir::GluKind::kSwiGlu}));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = input, .kind = co::InputKind::kFeatures}};
  step.weights = {
      co::StepWeight{.tensor = gate,
                     .ref = gate_ref,
                     .parts = {{"gate.weight", llmp::format::PartRole::kData}}},
      co::StepWeight{
          .tensor = up, .ref = up_ref, .parts = {{"up.weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io = co::GraphIO(step);
  const auto backend = life_->backend->Backend();
  const auto& recipe_backend = backend.owner ? backend.owner->backend() : backend;
  auto planned =
      llmp::planner::Plan(step.graph.Clone(), *recipe_backend.registry,
                          {.device_class = "sm_121", .arch = 121}, llmp::planner::Policy{});
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(planned->applied.size(), 1U);
  EXPECT_EQ(planned->applied[0].pattern, "float_mul_mat_glu");
  ASSERT_EQ(planned->graph.outputs().size(), 1U);
  EXPECT_EQ(planned->graph.node(planned->graph.producer(planned->graph.outputs().front())).key,
            ir::Fused(ir::FusedKind::kMatMulGlu));
  bool selected_mmvf = false;
  for (const auto& selected : planned->steps) {
    if (recipe_backend.registry->at(selected.implementation).name == kg::kMulMatGluFused) {
      selected_mmvf = true;
      ASSERT_EQ(selected.nodes.size(), 1U);
      EXPECT_EQ(planned->graph.node(selected.nodes.front()).key,
                ir::Fused(ir::FusedKind::kMatMulGlu));
    }
  }
  ASSERT_TRUE(selected_mmvf);

  const auto input_bytes = ir::StorageBytes(step.graph.tensor(input));
  const auto gate_bytes = ir::StorageBytes(step.graph.tensor(gate));
  const auto up_bytes = ir::StorageBytes(step.graph.tensor(up));
  ASSERT_TRUE(input_bytes && gate_bytes && up_bytes);
  const std::array<kn::ComparisonBindingRange, 3> bindings{
      kn::ComparisonBindingRange{.original_root = input,
                                 .allocation_class = 1,
                                 .allocation_bytes = *input_bytes,
                                 .bytes = *input_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = gate,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 2,
                                 .allocation_bytes = *gate_bytes,
                                 .bytes = *gate_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = up,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 3,
                                 .allocation_bytes = *up_bytes,
                                 .bytes = *up_bytes,
                                 .alignment = 16}};
  const std::array<co::ComparisonExternalBinding, 3> external{
      co::ComparisonExternalBinding{.original_root = input,
                                    .allocation_class = 1,
                                    .allocation_bytes = *input_bytes,
                                    .bytes = *input_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = gate,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = *gate_bytes,
                                    .bytes = *gate_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = up,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 3,
                                    .allocation_bytes = *up_bytes,
                                    .bytes = *up_bytes,
                                    .alignment = 16}};
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, external},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = bindings, .expected_boundary = &*boundary};
  const llmp::planner::StepShape shape{.stage = 0,
                                       .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  kn::Addresses addresses{};
  kn::LowerInput lower_input(*planned, step, shape, addresses, 0, {}, &request);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  const auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
}

TEST_F(GgmlComparisonLowerForGpu, ExactSiteMatMulGluAlternativeSealsBothBoundArms) {
  constexpr std::uint64_t kWidth = 128;
  const ir::WeightRef gate_ref{.role = 1};
  const ir::WeightRef up_ref{.role = 2};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto input =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, 1})));
  const auto gate =
      builder.Weight(ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({kWidth, kWidth})),
                     gate_ref, "gate.weight");
  const auto up = builder.Weight(
      ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({kWidth, kWidth})), up_ref, "up.weight");
  const auto gate_product = builder.MatMul(gate, input);
  const auto up_product = builder.MatMul(up, input);
  builder.Output(
      builder.Glu(gate_product, up_product, ir::GluParams{.kind = ir::GluKind::kSwiGlu}));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = input, .kind = co::InputKind::kStream}};
  step.weights = {
      co::StepWeight{.tensor = gate,
                     .ref = gate_ref,
                     .parts = {{"gate.weight", llmp::format::PartRole::kData}}},
      co::StepWeight{
          .tensor = up, .ref = up_ref, .parts = {{"up.weight", llmp::format::PartRole::kData}}}};
  const auto io = co::GraphIO(step);
  const auto input_bytes = ir::StorageBytes(step.graph.tensor(input));
  const auto gate_parts = llmp::format::PartsOf(step.graph.tensor(gate).format);
  const auto up_parts = llmp::format::PartsOf(step.graph.tensor(up).format);
  ASSERT_TRUE(input_bytes && gate_parts && gate_parts->size() == 1 && up_parts &&
              up_parts->size() == 1);
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* pattern = llmp::planner::FindPattern("float_mul_mat_glu");
  ASSERT_NE(pattern, nullptr);
  std::vector<ir::NodeId> origins;
  std::size_t matmuls = 0, glus = 0;
  for (std::uint32_t i = 0; i < step.graph.nodes().size(); ++i) {
    const auto op = step.graph.node(ir::NodeId{i}).key.op;
    if (op == ir::Op::kMatMul) {
      ++matmuls;
      origins.emplace_back(i);
    } else if (op == ir::Op::kGlu) {
      ++glus;
      origins.emplace_back(i);
    }
  }
  ASSERT_EQ(matmuls, 2U);
  ASSERT_EQ(glus, 1U);
  ASSERT_EQ(origins.size(), 3U);
  std::vector<llmp::planner::ComparisonOriginSet> members;
  for (const auto origin : origins) members.push_back({origin});
  const llmp::planner::ComparisonSiteSelector selector{
      .source_graph_identity = step.graph.identity(),
      .kind = llmp::planner::ComparisonSiteKind::kRewrite,
      .pattern = "float_mul_mat_glu",
      .version = pattern->version,
      .member_origins = std::move(members),
      .member_owner_segments = {},
      .boundary_ports = {}};
  constexpr std::size_t kArenaBytes = 1U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  CudaBuffer activation, gate_weights, up_weights, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(gate_weights.Allocate(gate_parts->parts()[0].readable));
  ASSERT_TRUE(up_weights.Allocate(up_parts->parts()[0].readable));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  kn::Addresses addresses{.weight =
                              [&](const ir::WeightRef& ref, llmp::format::PartRole) {
                                if (ref == gate_ref) return gate_weights.address();
                                if (ref == up_ref) return up_weights.address();
                                return std::uint64_t{0};
                              },
                          .weight_rows = {},
                          .state = {},
                          .save = {},
                          .activations = activation.address(),
                          .activation_bytes = kArenaBytes,
                          .activation_alignment = 256};
  addresses.weight_range = [&](const ir::WeightRef& ref,
                               llmp::format::PartRole role) -> std::optional<kn::BoundedAddress> {
    if (role != llmp::format::PartRole::kData) return {};
    if (ref == gate_ref)
      return kn::BoundedAddress{gate_weights.address(), gate_parts->parts()[0].readable, 256};
    if (ref == up_ref)
      return kn::BoundedAddress{up_weights.address(), up_parts->parts()[0].readable, 256};
    return {};
  };
  addresses.weight_alignment = [](const ir::WeightRef&, llmp::format::PartRole) {
    return std::uint64_t{256};
  };
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  const llmp::planner::Policy policy{.active_stage = 0};
  auto prepared = en::PrepareComponentPlanOptions(io, recipe, addresses,
                                                  {.capturable = true, .within = recipe.within});
  ASSERT_TRUE(prepared);
  const auto options = *prepared;
  auto initial = en::PlanComponentGraph(io, recipe, policy, addresses, options);
  ASSERT_TRUE(initial);
  auto source_bindings = en::ComparisonSourceBindings::Create(funder, io, 0);
  ASSERT_TRUE(source_bindings);
  auto source_ranges = source_bindings->Resolve(*initial, addresses, recipe.written);
  ASSERT_TRUE(source_ranges) << (source_ranges ? "" : source_ranges.error().detail);
  const auto source_view = source_bindings->boundary_view();
  ASSERT_TRUE(source_view);
  const auto view = *source_view;
  auto inventory = llmp::planner::BoundComparison(view, selector, funder, *recipe.registry,
                                                  recipe.device, policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  auto alternative = llmp::planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto planned = llmp::planner::PlanAlternative(*inventory, std::move(*alternative),
                                                *recipe.registry, recipe.device, policy, options);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
  EXPECT_EQ(inventory->incumbent.plan.applied[0].pattern, "float_mul_mat_glu");
  EXPECT_TRUE(std::ranges::none_of(planned->plan.applied, [](const auto& applied) {
    return applied.pattern == "float_mul_mat_glu";
  }));

  std::vector<std::uint32_t> region_offsets{0};
  std::vector<ir::NodeId> region_members;
  for (const auto& group : inventory->site.accepted_member_origins) {
    region_members.insert(region_members.end(), group.begin(), group.end());
    region_offsets.push_back(static_cast<std::uint32_t>(region_members.size()));
  }
  auto region = co::MakeComparisonRegionBoundary({view, region_offsets, region_members}, funder);
  ASSERT_TRUE(region) << (region ? "" : region.error().detail);

  const std::array<const llmp::planner::StepPlan*, 2> plans{&inventory->incumbent.plan,
                                                            &planned->plan};
  auto native = life_->execution->Submission(life_->node.stream(0));
  ASSERT_TRUE(native);
  auto stream = static_cast<cudaStream_t>(native->handle);
  std::vector<__half> gate_data(kWidth * kWidth), up_data(kWidth * kWidth);
  for (std::size_t i = 0; i < gate_data.size(); ++i) {
    gate_data[i] = __float2half_rn(static_cast<float>(static_cast<int>(i % 31) - 15) / 256.0F);
    up_data[i] = __float2half_rn(static_cast<float>(static_cast<int>(i % 23) - 11) / 256.0F);
  }
  ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(gate_weights.address()), gate_data.data(),
                            gate_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(up_weights.address()), up_data.data(),
                            up_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::array<kn::ComparisonRecipeSeal, 2> descriptors;
  const auto check_changed = [&](kn::StepProgram& program, std::size_t nodes, double bound) {
    auto recorded =
        program.Record(**launch, *native, [&] { return program.Run(**launch, *native); });
    ASSERT_TRUE(recorded);
    EXPECT_EQ((*recorded)->nodes(), nodes);
    std::vector<float> previous;
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
      std::vector<float> features(kWidth);
      for (std::size_t i = 0; i < features.size(); ++i)
        features[i] = static_cast<float>(static_cast<int>((i * 13 + frame * 7) % 37) - 18) /
                      (frame == 0 ? 32.0F : 16.0F);
      ASSERT_EQ(
          cudaMemsetAsync(reinterpret_cast<void*>(activation.address()), 0xff, kArenaBytes, stream),
          cudaSuccess);
      ASSERT_EQ(program.inputs().size(), 1U);
      const auto& slot = program.inputs()[0];
      ASSERT_EQ(slot.bytes, features.size() * sizeof(float));
      ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), features.data(),
                                slot.bytes, cudaMemcpyHostToDevice, stream),
                cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      ASSERT_TRUE((*launch)->Launch(*native, **recorded));
      auto fence = life_->execution->Record(life_->node.stream(0));
      ASSERT_TRUE(fence);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      EXPECT_EQ(*life_->execution->Query(*fence), llmp::providers::FenceState::kComplete);
      ASSERT_TRUE(life_->execution->Release(*fence));
      ASSERT_EQ(program.outputs().size(), 1U);
      const auto* output_port = &program.outputs().front();
      ASSERT_EQ(output_port->tensor, io.outputs.front());
      ASSERT_EQ(output_port->bytes, kWidth * sizeof(float));
      std::vector<float> actual(kWidth);
      ASSERT_EQ(
          cudaMemcpy(actual.data(), reinterpret_cast<const void*>(output_port->device_address),
                     actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (std::size_t row = 0; row < actual.size(); ++row) {
        double g = 0, u = 0;
        for (std::size_t k = 0; k < features.size(); ++k) {
          g += static_cast<double>(__half2float(gate_data[row * kWidth + k])) * features[k];
          u += static_cast<double>(__half2float(up_data[row * kWidth + k])) * features[k];
        }
        const double expected = g / (1.0 + std::exp(-g)) * u;
        ASSERT_TRUE(std::isfinite(actual[row]));
        EXPECT_LE(std::abs(actual[row] - expected) / (1.0 + std::abs(expected)), bound);
      }
      if (frame != 0) EXPECT_NE(previous, actual);
      previous = std::move(actual);
    }
  };
  for (const auto* plan : plans) {
    ASSERT_LE(plan->placement.extent, kArenaBytes);
    std::vector<kn::ComparisonBindingRange> ranges;
    for (const auto& fact : inventory->source_boundary.external()) {
      ASSERT_LT(fact.canonical_root, inventory->source_boundary.roots().size());
      const auto original_root =
          inventory->source_boundary.roots()[fact.canonical_root].original_root;
      std::uint64_t base = 0;
      if (fact.part) {
        if (original_root == gate)
          base = gate_weights.address();
        else if (original_root == up)
          base = up_weights.address();
        else
          FAIL() << "unexpected external weight root";
      } else {
        ASSERT_EQ(original_root, input);
        ASSERT_LT(original_root.v, plan->tensors.size());
        const auto mapped = plan->tensors[original_root.v];
        ASSERT_TRUE(mapped.valid());
        const auto root = plan->graph.root(mapped);
        const auto placed = std::ranges::find_if(plan->placement.offsets, [&](const auto& entry) {
          return plan->graph.root(entry.first) == root;
        });
        ASSERT_NE(placed, plan->placement.offsets.end());
        base = activation.address() + placed->second;
      }
      ranges.push_back({.original_root = original_root,
                        .part = fact.part,
                        .allocation_class = fact.canonical_allocation_class,
                        .allocation_bytes = fact.allocation_bytes,
                        .offset = fact.offset,
                        .bytes = fact.bytes,
                        .alignment = fact.alignment,
                        .base = base,
                        .resource_part_identity = fact.resource_part_identity});
    }
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .scope = kn::ComparisonProofScope::kWholeProgramNative,
                                  .bindings = ranges,
                                  .expected_boundary = &inventory->source_boundary};
    const auto& owned = plan == plans[0] ? inventory->incumbent : *planned;
    ASSERT_TRUE(owned.final_origins);
    auto copied_origins = owned.final_origins->Clone(funder);
    ASSERT_TRUE(copied_origins);
    kn::ComparisonRegionCapture capture{
        .seed = &*region, .origins = &*copied_origins, .final_nodes = &owned.final_node_origins};
    // A bound weight overlapping the actual output is not an admissible
    // comparison resource. Inspect only; malformed descriptors never execute.
    const auto output_root = plan->graph.root(plan->graph.outputs().front());
    const auto output_placement = std::ranges::find(plan->placement.offsets, output_root,
                                                    &std::pair<ir::TensorId, std::uint64_t>::first);
    ASSERT_NE(output_placement, plan->placement.offsets.end());
    const auto alias_address = activation.address() + output_placement->second;
    auto unsafe_addresses = addresses;
    unsafe_addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
      return ref == gate_ref ? alias_address : up_weights.address();
    };
    auto unsafe_ranges = ranges;
    for (auto& range : unsafe_ranges)
      if (range.original_root == gate) range.base = alias_address;
    auto unsafe_request = request;
    unsafe_request.bindings = unsafe_ranges;
    kn::LowerInput unsafe_input(*plan, io, unsafe_addresses, std::nullopt, &unsafe_request,
                                &capture);
    auto unsafe_program = life_->backend->Lower(unsafe_input);
    if (unsafe_program) {
      const auto unsafe_projection = (*unsafe_program)->ProjectComparisonRegionDescriptor(*region);
      EXPECT_NE(unsafe_projection.status, kn::ComparisonRegionStatus::kClosed);
    }
    kn::LowerInput lower_input(*plan, io, addresses, std::nullopt, &request, &capture);
    auto program = life_->backend->Lower(lower_input);
    ASSERT_TRUE(program) << (program ? "" : program.error());
    auto seal = (*program)->comparison_recipe();
    ASSERT_TRUE(seal);
    ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
    ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
    const auto descriptor = (*program)->ProjectComparisonRegionDescriptor(*region);
    ASSERT_EQ(descriptor.status, kn::ComparisonRegionStatus::kClosed) << descriptor.reason;
    const auto program_descriptor = seal;
    ASSERT_TRUE((*program)->Bind(**launch));
    seal = (*program)->comparison_recipe();
    ASSERT_TRUE(seal);
    EXPECT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified);
    EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
    const auto projected = (*program)->ProjectComparisonRegion(*region);
    EXPECT_EQ(projected.status, kn::ComparisonRegionStatus::kClosed) << projected.reason;

    ASSERT_EQ(projected.status, kn::ComparisonRegionStatus::kClosed) << projected.reason;
    descriptors[plan == plans[0] ? 0 : 1] = *program_descriptor;
    check_changed(**program, plan == plans[0] ? 1U : 3U, plan == plans[0] ? 1e-4 : 2e-2);
  }
  const auto key = en::RecipeComparisonKeyForWholePrograms(
      inventory->source_boundary, *region, inventory->site, llmp::execution::Phase::kPrefill,
      recipe.registry->SelectionDigest(), descriptors[0], descriptors[1]);
  ASSERT_TRUE(key);
  for (std::size_t choice = 0; choice < 3; ++choice) {
    SCOPED_TRACE(choice);
    std::vector<llmp::execution::Implementation> declarations;
    for (std::size_t i = 0; i < recipe.registry->size(); ++i)
      declarations.push_back(recipe.registry->at(i));
    auto ranks = recipe.registry->ranks();
    auto registered = *key;
    if (choice == 2)
      registered.subject = llmp::base::Sha256{}.Update("different original component").Finish();
    ranks.recipe_comparisons = {
        {.key = registered,
         .incumbent_ns = choice == 1 ? 50U : 100U,
         .alternative_ns = choice == 1 ? 100U : 50U,
         .samples = 3,
         .protocol = llmp::base::Sha256{}.Update("component-test-protocol").Finish(),
         .receipt = llmp::base::Sha256{}.Update("component-test-receipt").Finish(),
         .provenance = "synthetic-owned-mechanism-control",
         .shape_class = "component-f16-glu"}};
    auto measured = llmp::execution::Registry::Create(std::move(declarations), std::move(ranks));
    ASSERT_TRUE(measured) << (measured ? "" : measured.error().detail);
    auto component_backend = recipe;
    component_backend.registry = &*measured;
    auto production = en::BuildComponentProgram(io, component_backend, policy, addresses, options,
                                                std::nullopt, {}, backend.owner);
    ASSERT_TRUE(production) << (production ? "" : production.error());
    const auto descriptor = (*production)->comparison_recipe();
    ASSERT_TRUE(descriptor);
    EXPECT_EQ(*descriptor, descriptors[choice == 0 ? 1 : 0]);
    ASSERT_TRUE((*production)->Bind(**launch));
    EXPECT_TRUE(en::SameWholeProgramRecipe(descriptors[choice == 0 ? 1 : 0],
                                           *(*production)->comparison_recipe()));
    // A winning decomposition vetoes the fused call; an incumbent win or
    // exact-source miss retains it. Both changed frames still execute.
    ASSERT_NE((*production)->bound(), nullptr);
    EXPECT_EQ((*production)->bound()->size(), choice == 0 ? 3U : 1U);
    check_changed(**production, choice == 0 ? 3U : 1U, choice == 0 ? 2e-2 : 1e-4);
  }
  life_->backend->LaunchReleased();
}

TEST_F(GgmlComparisonLowerForGpu, LocalAddFusionsCloseBothPlansAndReplayChangedFrames) {
  for (int fixture = 0; fixture < 3; ++fixture) {
    SCOPED_TRACE(fixture);
    if (fixture != 0) ASSERT_TRUE(life_->backend->LendLibrary(&life_->resources.library()));
    const bool norm_add = fixture == 0;
    const bool half_weights = fixture == 1;
    const auto pattern_name = norm_add ? "rms_norm_mul_add" : "float_mul_mat_add";
    constexpr std::uint64_t kWidth = 256;
    const ir::WeightRef gate_ref{.role = 1};
    const ir::WeightRef up_ref{.role = 2};
    ir::Builder builder;
    builder.SetOwner(0);
    const auto input =
        builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, 1})));
    const auto gate = builder.Weight(
        ir::Dense(half_weights ? llmp::format::Encoding::kF16 : llmp::format::Encoding::kF32,
                  norm_add ? ir::ShapeOf({kWidth}) : ir::ShapeOf({kWidth, kWidth})),
        gate_ref, "gate.weight");
    const auto up =
        norm_add ? builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, 1})))
                 : builder.Weight(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth})),
                                  up_ref, "bias.weight");
    const auto value =
        norm_add ? builder.Mul(builder.RmsNorm(input, 1e-6F), gate) : builder.MatMul(gate, input);
    builder.Output(builder.Add(value, up));
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
    step.inputs = {{.tensor = input, .kind = co::InputKind::kStream}};
    if (norm_add) step.inputs.push_back({.tensor = up, .kind = co::InputKind::kSide});
    step.weights = {co::StepWeight{.tensor = gate,
                                   .ref = gate_ref,
                                   .parts = {{"gate.weight", llmp::format::PartRole::kData}}}};
    if (!norm_add)
      step.weights.push_back(
          {.tensor = up, .ref = up_ref, .parts = {{"bias.weight", llmp::format::PartRole::kData}}});
    const auto io = co::GraphIO(step);
    const auto input_bytes = ir::StorageBytes(step.graph.tensor(input));
    const auto gate_parts = llmp::format::PartsOf(step.graph.tensor(gate).format);
    const auto up_parts = llmp::format::PartsOf(step.graph.tensor(up).format);
    ASSERT_TRUE(input_bytes && gate_parts && gate_parts->size() == 1 && up_parts &&
                up_parts->size() == 1);
    const std::array external{
        co::ComparisonExternalBinding{.original_root = input,
                                      .allocation_class = 1,
                                      .allocation_bytes = *input_bytes,
                                      .bytes = *input_bytes,
                                      .alignment = 16},
        co::ComparisonExternalBinding{.original_root = gate,
                                      .part = llmp::format::PartRole::kData,
                                      .allocation_class = 2,
                                      .allocation_bytes = gate_parts->parts()[0].readable,
                                      .bytes = gate_parts->parts()[0].readable,
                                      .alignment = 16},
        co::ComparisonExternalBinding{
            .original_root = up,
            .part = norm_add ? std::nullopt : std::optional{llmp::format::PartRole::kData},
            .allocation_class = 3,
            .allocation_bytes = up_parts->parts()[0].readable,
            .bytes = up_parts->parts()[0].readable,
            .alignment = 16}};
    const co::ComparisonBoundaryView view{io, {}, 0, external};
    auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    const auto* pattern = llmp::planner::FindPattern(pattern_name);
    ASSERT_NE(pattern, nullptr);
    std::vector<ir::NodeId> origins;
    std::size_t matmuls = 0, glus = 0;
    for (std::uint32_t i = 0; i < step.graph.nodes().size(); ++i) {
      const auto op = step.graph.node(ir::NodeId{i}).key.op;
      if (op == ir::Op::kMatMul || op == ir::Op::kRmsNorm || op == ir::Op::kMul) {
        ++matmuls;
        origins.emplace_back(i);
      } else if (op == ir::Op::kAdd) {
        ++glus;
        origins.emplace_back(i);
      }
    }
    ASSERT_EQ(matmuls, norm_add ? 2U : 1U);
    ASSERT_EQ(glus, 1U);
    ASSERT_EQ(origins.size(), norm_add ? 3U : 2U);
    std::vector<llmp::planner::ComparisonOriginSet> members;
    for (const auto origin : origins) members.push_back({origin});
    const llmp::planner::ComparisonSiteSelector selector{
        .source_graph_identity = step.graph.identity(),
        .kind = llmp::planner::ComparisonSiteKind::kRewrite,
        .pattern = pattern_name,
        .version = pattern->version,
        .member_origins = std::move(members),
        .member_owner_segments = {},
        .boundary_ports = {}};
    auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
    const llmp::planner::Policy policy{};
    const llmp::planner::PlanOptions options{.capturable = true, .within = recipe.within};
    auto inventory = llmp::planner::BoundComparison(view, selector, funder, *recipe.registry,
                                                    recipe.device, policy, options);
    if (fixture == 2) {
      // The normal float_mul_mat_add matcher currently admits F16 only.
      // Preserve F32 backend proof support without inventing an accepted site.
      EXPECT_FALSE(inventory);
      continue;
    }
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    auto alternative = llmp::planner::MakeAlternative(*inventory);
    ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
    auto planned = llmp::planner::PlanAlternative(*inventory, std::move(*alternative),
                                                  *recipe.registry, recipe.device, policy, options);
    ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
    ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
    EXPECT_EQ(inventory->incumbent.plan.applied[0].pattern, pattern_name);
    EXPECT_TRUE(std::ranges::none_of(planned->plan.applied, [&](const auto& applied) {
      return applied.pattern == pattern_name;
    }));

    std::vector<std::uint32_t> region_offsets{0};
    std::vector<ir::NodeId> region_members;
    for (const auto& group : inventory->site.accepted_member_origins) {
      region_members.insert(region_members.end(), group.begin(), group.end());
      region_offsets.push_back(static_cast<std::uint32_t>(region_members.size()));
    }
    auto region = co::MakeComparisonRegionBoundary({view, region_offsets, region_members}, funder);
    ASSERT_TRUE(region) << (region ? "" : region.error().detail);

    const std::array<std::uint32_t, 2> add_offsets{0, 1};
    const std::array<ir::NodeId, 1> add_member{origins.back()};
    auto add_region = co::MakeComparisonRegionBoundary({view, add_offsets, add_member}, funder);
    ASSERT_TRUE(add_region);

    constexpr std::size_t kArenaBytes = 1U << 20U;
    constexpr std::size_t kPoolBytes = 1U << 20U;
    CudaBuffer activation, gate_weights, up_weights, pool, persistent;
    ASSERT_TRUE(activation.Allocate(kArenaBytes));
    ASSERT_TRUE(gate_weights.Allocate(gate_parts->parts()[0].readable));
    ASSERT_TRUE(up_weights.Allocate(up_parts->parts()[0].readable));
    ASSERT_TRUE(pool.Allocate(kPoolBytes));
    const auto persistent_bytes = life_->backend->persistent_bytes();
    ASSERT_GT(persistent_bytes, 0U);
    ASSERT_TRUE(persistent.Allocate(persistent_bytes));
    auto launch =
        life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                               {.base = persistent.address(), .bytes = persistent_bytes});
    ASSERT_TRUE(launch) << (launch ? "" : launch.error());
    const std::array<const llmp::planner::StepPlan*, 2> plans{&inventory->incumbent.plan,
                                                              &planned->plan};

    auto native = life_->execution->Submission(life_->node.stream(0));
    ASSERT_TRUE(native);
    auto stream = static_cast<cudaStream_t>(native->handle);
    std::vector<float> gate_values(norm_add ? kWidth : kWidth * kWidth), bias_values(kWidth);
    std::vector<__half> half_values(gate_values.size());
    for (std::size_t i = 0; i < gate_values.size(); ++i) {
      gate_values[i] = norm_add ? 0.75F + static_cast<float>(i % 7) / 32.0F
                                : static_cast<float>(static_cast<int>(i % 31) - 15) / 256.0F;
      half_values[i] = __float2half_rn(gate_values[i]);
      if (half_weights) gate_values[i] = __half2float(half_values[i]);
    }
    for (std::size_t i = 0; i < bias_values.size(); ++i)
      bias_values[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 64.0F;
    ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(gate_weights.address()),
                              half_weights ? static_cast<const void*>(half_values.data())
                                           : static_cast<const void*>(gate_values.data()),
                              gate_parts->parts()[0].readable, cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(up_weights.address()), bias_values.data(),
                              up_parts->parts()[0].readable, cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (const auto* plan : plans) {
      ASSERT_LE(plan->placement.extent, kArenaBytes);
      kn::Addresses addresses{.weight =
                                  [&](const ir::WeightRef& ref, llmp::format::PartRole) {
                                    if (ref == gate_ref) return gate_weights.address();
                                    if (ref == up_ref) return up_weights.address();
                                    return std::uint64_t{0};
                                  },
                              .weight_rows = {},
                              .state = {},
                              .save = {},
                              .activations = activation.address(),
                              .activation_bytes = kArenaBytes,
                              .activation_alignment = 256};
      std::vector<kn::ComparisonBindingRange> ranges;
      for (const auto& fact : inventory->source_boundary.external()) {
        ASSERT_LT(fact.canonical_root, inventory->source_boundary.roots().size());
        const auto original_root =
            inventory->source_boundary.roots()[fact.canonical_root].original_root;
        std::uint64_t base = 0;
        if (fact.part) {
          if (original_root == gate)
            base = gate_weights.address();
          else if (original_root == up)
            base = up_weights.address();
          else
            FAIL() << "unexpected external weight root";
        } else {
          ASSERT_TRUE(original_root == input || (norm_add && original_root == up));
          ASSERT_LT(original_root.v, plan->tensors.size());
          const auto mapped = plan->tensors[original_root.v];
          ASSERT_TRUE(mapped.valid());
          const auto root = plan->graph.root(mapped);
          const auto placed = std::ranges::find_if(plan->placement.offsets, [&](const auto& entry) {
            return plan->graph.root(entry.first) == root;
          });
          ASSERT_NE(placed, plan->placement.offsets.end());
          base = activation.address() + placed->second;
        }
        ranges.push_back({.original_root = original_root,
                          .part = fact.part,
                          .allocation_class = fact.canonical_allocation_class,
                          .allocation_bytes = fact.allocation_bytes,
                          .offset = fact.offset,
                          .bytes = fact.bytes,
                          .alignment = fact.alignment,
                          .base = base,
                          .resource_part_identity = fact.resource_part_identity});
      }
      kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                    .bindings = ranges,
                                    .expected_boundary = &inventory->source_boundary};
      const auto& owned = plan == plans[0] ? inventory->incumbent : *planned;
      ASSERT_TRUE(owned.final_origins);
      auto copied_origins = owned.final_origins->Clone(funder);
      ASSERT_TRUE(copied_origins);
      kn::ComparisonRegionCapture capture{
          .seed = &*region, .origins = &*copied_origins, .final_nodes = &owned.final_node_origins};
      // A bound weight overlapping the actual output is not an admissible
      // comparison resource. Inspect only; malformed descriptors never execute.
      const auto output_root = plan->graph.root(plan->graph.outputs().front());
      const auto output_placement = std::ranges::find(
          plan->placement.offsets, output_root, &std::pair<ir::TensorId, std::uint64_t>::first);
      ASSERT_NE(output_placement, plan->placement.offsets.end());
      const auto alias_address = activation.address() + output_placement->second;
      auto unsafe_addresses = addresses;
      unsafe_addresses.weight = [&](const ir::WeightRef& ref, llmp::format::PartRole) {
        return ref == gate_ref ? alias_address : up_weights.address();
      };
      auto unsafe_ranges = ranges;
      for (auto& range : unsafe_ranges)
        if (range.original_root == gate) range.base = alias_address;
      auto unsafe_request = request;
      unsafe_request.bindings = unsafe_ranges;
      kn::LowerInput unsafe_input(*plan, io, unsafe_addresses, std::nullopt, &unsafe_request,
                                  &capture);
      auto unsafe_program = life_->backend->Lower(unsafe_input);
      if (unsafe_program) {
        const auto unsafe_projection =
            (*unsafe_program)->ProjectComparisonRegionDescriptor(*region);
        EXPECT_NE(unsafe_projection.status, kn::ComparisonRegionStatus::kClosed);
      }
      kn::LowerInput lower_input(*plan, io, addresses, std::nullopt, &request, &capture);
      auto program = life_->backend->Lower(lower_input);
      ASSERT_TRUE(program) << (program ? "" : program.error());
      auto seal = (*program)->comparison_recipe();
      ASSERT_TRUE(seal);
      ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
      ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
      const auto add_descriptor = (*program)->ProjectComparisonRegionDescriptor(*add_region);
      EXPECT_EQ(add_descriptor.status, plan == plans[0] ? kn::ComparisonRegionStatus::kExpand
                                                        : kn::ComparisonRegionStatus::kClosed)
          << add_descriptor.reason;
      const auto descriptor = (*program)->ProjectComparisonRegionDescriptor(*region);
      ASSERT_EQ(descriptor.status, kn::ComparisonRegionStatus::kClosed) << descriptor.reason;
      ASSERT_TRUE((*program)->Bind(**launch));
      seal = (*program)->comparison_recipe();
      ASSERT_TRUE(seal);
      EXPECT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified);
      EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
      if (plan == plans[1]) {
        const auto remainder = (*program)->CompareComparisonRemainder(
            **program, *add_region, kn::ComparisonCertificatePhase::kBound);
        EXPECT_EQ(remainder.status, kn::ComparisonRemainderStatus::kSame) << remainder.reason;
      }
      const auto projected = (*program)->ProjectComparisonRegion(*region);
      EXPECT_EQ(projected.status, kn::ComparisonRegionStatus::kClosed) << projected.reason;

      ASSERT_EQ(projected.status, kn::ComparisonRegionStatus::kClosed) << projected.reason;
      auto recorded =
          (*program)->Record(**launch, *native, [&] { return (*program)->Run(**launch, *native); });
      ASSERT_TRUE(recorded);
      EXPECT_EQ((*recorded)->nodes(), plan == plans[0] ? 1U : 2U);
      std::vector<float> previous;
      for (std::uint32_t frame = 0; frame < 2; ++frame) {
        if (norm_add)
          for (std::size_t i = 0; i < bias_values.size(); ++i)
            bias_values[i] = static_cast<float>(static_cast<int>((i + frame * 3) % 11) - 5) /
                             (frame == 0 ? 64.0F : 32.0F);
        std::vector<float> features(kWidth);
        for (std::size_t i = 0; i < features.size(); ++i)
          features[i] = static_cast<float>(static_cast<int>((i * 13 + frame * 7) % 37) - 18) /
                        (frame == 0 ? 32.0F : 16.0F);
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(activation.address()), 0xff, kArenaBytes,
                                  stream),
                  cudaSuccess);
        ASSERT_EQ((*program)->inputs().size(), norm_add ? 2U : 1U);
        for (const auto& slot : (*program)->inputs()) {
          ASSERT_EQ(slot.kind, kn::InputKind::kStepBytes);
          const auto& payload = slot.tensor == input ? features : bias_values;
          ASSERT_TRUE(slot.tensor == input || (norm_add && slot.tensor == up));
          ASSERT_EQ(slot.bytes, payload.size() * sizeof(float));
          ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), payload.data(),
                                    slot.bytes, cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
        }
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_TRUE((*launch)->Launch(*native, **recorded));
        auto fence = life_->execution->Record(life_->node.stream(0));
        ASSERT_TRUE(fence);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(*life_->execution->Query(*fence), llmp::providers::FenceState::kComplete);
        ASSERT_TRUE(life_->execution->Release(*fence));
        const auto output_port =
            std::ranges::find(projected.value_ports, kn::ComparisonRegionValueKind::kOutput,
                              &kn::ComparisonRegionValuePort::kind);
        ASSERT_NE(output_port, projected.value_ports.end());
        ASSERT_EQ(output_port->bytes, kWidth * sizeof(float));
        std::vector<float> actual(kWidth);
        ASSERT_EQ(cudaMemcpy(actual.data(), reinterpret_cast<const void*>(output_port->address),
                             actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        // Existing F16 biased-product contract: both paths use default
        // accumulation (ggml_ops_test.cc). F32 norm/product checks retain 1e-5.
        const double bound = half_weights ? 2e-2 : 1e-5;
        double mean_square = 0;
        for (float feature : features) mean_square += static_cast<double>(feature) * feature;
        mean_square /= static_cast<double>(features.size());
        for (std::size_t row = 0; row < actual.size(); ++row) {
          double expected = bias_values[row];
          if (norm_add) {
            expected += features[row] / std::sqrt(mean_square + 1e-6) * gate_values[row];
          } else {
            for (std::size_t k = 0; k < features.size(); ++k)
              expected += static_cast<double>(gate_values[row * kWidth + k]) * features[k];
          }
          ASSERT_TRUE(std::isfinite(actual[row]));
          EXPECT_LE(std::abs(actual[row] - expected) / (1.0 + std::abs(expected)), bound);
        }
        if (frame != 0) EXPECT_NE(previous, actual);
        previous = std::move(actual);
      }
    }
    life_->backend->LaunchReleased();
  }
}

TEST_F(GgmlComparisonLowerForGpu, UnrelatedMatMulCannotBorrowTheMmvfGluClosure) {
  constexpr std::uint64_t kWidth = 128;
  constexpr auto kF16 = llmp::format::Encoding::kF16;
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  const ir::WeightRef gate_ref{.role = 1};
  const ir::WeightRef up_ref{.role = 2};
  const ir::WeightRef unrelated_ref{.role = 3};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto input = builder.Input(ir::Dense(kF32, ir::ShapeOf({kWidth, 1})));
  const auto gate = builder.Weight(ir::Dense(kF16, ir::ShapeOf({kWidth, kWidth})), gate_ref);
  const auto up = builder.Weight(ir::Dense(kF16, ir::ShapeOf({kWidth, kWidth})), up_ref);
  const auto unrelated =
      builder.Weight(ir::Dense(kF16, ir::ShapeOf({kWidth, kWidth})), unrelated_ref);
  builder.Keep(input);
  const auto gate_product = builder.MatMul(gate, input);
  const auto up_product = builder.MatMul(up, input);
  const auto unrelated_product = builder.MatMul(unrelated, input);
  builder.Output(
      builder.Glu(gate_product, up_product, ir::GluParams{.kind = ir::GluKind::kSwiGlu}));
  builder.Output(unrelated_product);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = input, .kind = co::InputKind::kFeatures}};
  step.weights = {
      co::StepWeight{
          .tensor = gate, .ref = gate_ref, .parts = {{"gate", llmp::format::PartRole::kData}}},
      co::StepWeight{.tensor = up, .ref = up_ref, .parts = {{"up", llmp::format::PartRole::kData}}},
      co::StepWeight{.tensor = unrelated,
                     .ref = unrelated_ref,
                     .parts = {{"unrelated", llmp::format::PartRole::kData}}}};
  const auto io = co::GraphIO(step);
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  kn::Addresses addresses{};
  auto plan = en::PlanComponentGraph(io, recipe, {}, addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  const auto input_bytes = ir::StorageBytes(step.graph.tensor(input));
  const auto gate_parts = llmp::format::PartsOf(step.graph.tensor(gate).format);
  const auto up_parts = llmp::format::PartsOf(step.graph.tensor(up).format);
  const auto unrelated_parts = llmp::format::PartsOf(step.graph.tensor(unrelated).format);
  ASSERT_TRUE(input_bytes && gate_parts && gate_parts->size() == 1 && up_parts &&
              up_parts->size() == 1 && unrelated_parts && unrelated_parts->size() == 1);
  const std::array external{
      co::ComparisonExternalBinding{.original_root = input,
                                    .allocation_class = 1,
                                    .allocation_bytes = *input_bytes,
                                    .bytes = *input_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = gate,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = gate_parts->parts()[0].readable,
                                    .bytes = gate_parts->parts()[0].readable,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = up,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 3,
                                    .allocation_bytes = up_parts->parts()[0].readable,
                                    .bytes = up_parts->parts()[0].readable,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = unrelated,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 4,
                                    .allocation_bytes = unrelated_parts->parts()[0].readable,
                                    .bytes = unrelated_parts->parts()[0].readable,
                                    .alignment = 16}};
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, external},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  std::vector<kn::ComparisonBindingRange> ranges;
  for (const auto& external_fact : external) {
    ranges.push_back({.original_root = external_fact.original_root,
                      .part = external_fact.part,
                      .allocation_class = external_fact.allocation_class,
                      .allocation_bytes = external_fact.allocation_bytes,
                      .bytes = external_fact.bytes,
                      .alignment = external_fact.alignment});
  }
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = ranges, .expected_boundary = &*boundary};
  const llmp::planner::StepShape shape{.stage = 0,
                                       .segments = {{.slot = 0, .rows = 1, .outputs = 2}}};
  kn::LowerInput lower_input(*plan, step, shape, addresses, 0, {}, &request);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  const auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kPhysicalClosure);
}

TEST_F(GgmlComparisonLowerForGpu, QsaPrepRemainsUnprovedWithoutCompletePhysicalClosure) {
  constexpr auto kFormat = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef weight_ref{.stage = 0, .component = 0, .role = 0, .layer = 0};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto q = builder.Input(ir::Dense(kFormat, ir::ShapeOf({128, 1})));
  const auto weight = builder.Weight(ir::Dense(kFormat, ir::ShapeOf({128})), weight_ref);
  const auto positions =
      builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({4})), {},
                    ir::InputParams{.fact = ir::InputFact::kPositions, .sections = 4});
  const llmp::ir::RopeParams rope{.mode = ir::RopeMode::kInterleavedMulti,
                                  .dims = 64,
                                  .base = 10000.0F,
                                  .scale = 1.0F,
                                  .ext_factor = 0.0F,
                                  .attn_factor = 1.0F,
                                  .sections = {1, 1, 1, 1}};
  const auto normalized = builder.RmsNorm(q, 1.0e-6F);
  const auto scaled = builder.Mul(normalized, weight);
  builder.Output(builder.Rope(scaled, positions, {}, rope));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = q, .kind = co::InputKind::kFeatures},
                 {.tensor = positions, .kind = co::InputKind::kPositions, .sections = 4}};
  step.weights = {co::StepWeight{
      .tensor = weight, .ref = weight_ref, .parts = {{"weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io = co::GraphIO(step);
  const auto backend = life_->backend->Backend();
  const auto& recipe_backend = backend.owner ? backend.owner->backend() : backend;
  llmp::planner::Policy policy{};
  policy.enabled = {"rms_norm_mul_rope"};
  auto planned = llmp::planner::Plan(step.graph.Clone(), *recipe_backend.registry,
                                     {.device_class = "sm_121", .arch = 121}, policy);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(planned->applied.size(), 1U);
  ASSERT_EQ(planned->applied[0].pattern, "interleaved_norm_rope");

  const auto q_bytes = ir::StorageBytes(step.graph.tensor(q));
  const auto position_bytes = ir::StorageBytes(step.graph.tensor(positions));
  const auto weight_bytes = ir::StorageBytes(step.graph.tensor(weight));
  ASSERT_TRUE(q_bytes && position_bytes && weight_bytes);
  const std::array<kn::ComparisonBindingRange, 3> bindings{
      kn::ComparisonBindingRange{.original_root = q,
                                 .allocation_class = 1,
                                 .allocation_bytes = *q_bytes,
                                 .bytes = *q_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = weight,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 2,
                                 .allocation_bytes = *weight_bytes,
                                 .bytes = *weight_bytes,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = positions,
                                 .allocation_class = 3,
                                 .allocation_bytes = *position_bytes,
                                 .bytes = *position_bytes,
                                 .alignment = 16}};
  const std::array<co::ComparisonExternalBinding, 3> external{
      co::ComparisonExternalBinding{.original_root = q,
                                    .allocation_class = 1,
                                    .allocation_bytes = *q_bytes,
                                    .bytes = *q_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = weight,
                                    .part = llmp::format::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = *weight_bytes,
                                    .bytes = *weight_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = positions,
                                    .allocation_class = 3,
                                    .allocation_bytes = *position_bytes,
                                    .bytes = *position_bytes,
                                    .alignment = 16}};
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, external},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = bindings, .expected_boundary = &*boundary};
  const llmp::planner::StepShape shape{.stage = 0,
                                       .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  kn::Addresses addresses{};
  kn::LowerInput lower_input(*planned, step, shape, addresses, 0, {}, &request);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kPhysicalClosure);
  ForeignStepLaunch foreign;
  const auto bound = (*program)->Bind(foreign);
  ASSERT_FALSE(bound);
  EXPECT_NE(bound.error().detail.find("a measurement does not bind"), std::string::npos);
  seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kPhysicalClosure);
}

TEST_F(GgmlComparisonLowerForGpu, IndependentNormRopeHelperCannotAuthorizeQsaPrepInTheSameProgram) {
  constexpr auto kF32 = llmp::format::Encoding::kF32;
  constexpr ir::WeightRef plain_weight_ref{.role = 1};
  constexpr ir::WeightRef qsa_weight_ref{.role = 2};
  ir::Builder builder;
  builder.SetOwner(0);
  const auto plain_q = builder.Input(ir::Dense(kF32, ir::ShapeOf({128, 1, 1})));
  const auto plain_weight = builder.Weight(ir::Dense(kF32, ir::ShapeOf({128})), plain_weight_ref);
  const auto plain_positions =
      builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({1})));
  const ir::RopeParams neox{.mode = ir::RopeMode::kNeox, .dims = 128, .original_context = 4096};
  const auto plain_norm = builder.RmsNorm(plain_q, 1.0e-6F);
  const auto plain_scaled = builder.Mul(plain_norm, plain_weight);
  builder.Output(builder.Rope(plain_scaled, plain_positions, {}, neox));

  const auto qsa_q = builder.Input(ir::Dense(kF32, ir::ShapeOf({128, 1})));
  const auto qsa_weight = builder.Weight(ir::Dense(kF32, ir::ShapeOf({128})), qsa_weight_ref);
  const auto qsa_positions =
      builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({4})), {},
                    ir::InputParams{.fact = ir::InputFact::kPositions, .sections = 4});
  const ir::RopeParams qsa_rope{.mode = ir::RopeMode::kInterleavedMulti,
                                .dims = 64,
                                .base = 10000.0F,
                                .scale = 1.0F,
                                .ext_factor = 0.0F,
                                .attn_factor = 1.0F,
                                .sections = {1, 1, 1, 1}};
  const auto qsa_norm = builder.RmsNorm(qsa_q, 1.0e-6F);
  const auto qsa_scaled = builder.Mul(qsa_norm, qsa_weight);
  builder.Output(builder.Rope(qsa_scaled, qsa_positions, {}, qsa_rope));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);

  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = plain_q, .kind = co::InputKind::kFeatures},
                 {.tensor = plain_positions, .kind = co::InputKind::kPositions},
                 {.tensor = qsa_q, .kind = co::InputKind::kFeatures},
                 {.tensor = qsa_positions, .kind = co::InputKind::kPositions, .sections = 4}};
  step.weights = {co::StepWeight{.tensor = plain_weight,
                                 .ref = plain_weight_ref,
                                 .parts = {{"plain.weight", llmp::format::PartRole::kData}}},
                  co::StepWeight{.tensor = qsa_weight,
                                 .ref = qsa_weight_ref,
                                 .parts = {{"qsa.weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io = co::GraphIO(step);
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const llmp::planner::Policy policy{};
  auto planned = llmp::planner::Plan(step.graph.Clone(), *recipe.registry,
                                     {.device_class = "sm_121", .arch = 121}, policy);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(planned->applied.size(), 2U);
  EXPECT_EQ(planned->applied[0].pattern, "interleaved_norm_rope");
  EXPECT_EQ(planned->applied[1].pattern, "interleaved_norm_rope");
  ASSERT_TRUE(std::ranges::any_of(planned->steps, [&](const auto& selected) {
    return recipe.registry->at(selected.implementation).name == kg::kGemmaNormRopeName;
  }));

  std::vector<kn::ComparisonBindingRange> ranges;
  std::vector<co::ComparisonExternalBinding> external;
  const auto bind = [&](ir::TensorId root, std::uint32_t allocation_class,
                        std::optional<llmp::format::PartRole> part = std::nullopt) {
    const auto bytes = ir::StorageBytes(step.graph.tensor(root));
    ASSERT_TRUE(bytes);
    ranges.push_back({.original_root = root,
                      .part = part,
                      .allocation_class = allocation_class,
                      .allocation_bytes = *bytes,
                      .bytes = *bytes,
                      .alignment = 16});
    external.push_back({.original_root = root,
                        .part = part,
                        .allocation_class = allocation_class,
                        .allocation_bytes = *bytes,
                        .bytes = *bytes,
                        .alignment = 16});
  };
  bind(plain_q, 1);
  bind(plain_weight, 2, llmp::format::PartRole::kData);
  bind(plain_positions, 3);
  bind(qsa_q, 4);
  bind(qsa_weight, 5, llmp::format::PartRole::kData);
  bind(qsa_positions, 6);
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, external},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = ranges, .expected_boundary = &*boundary};
  const llmp::planner::StepShape shape{.stage = 0,
                                       .segments = {{.slot = 0, .rows = 1, .outputs = 2}}};
  kn::Addresses addresses{};
  kn::LowerInput lower_input(*planned, step, shape, addresses, 0, {}, &request);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  const auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kPhysicalClosure);
}

TEST_F(GgmlComparisonLowerForGpu, IndexedActivationStoreRemainsUnprovedWithoutValueClosure) {
  ir::Builder builder;
  builder.SetOwner(0);
  const auto seed = builder.Input(ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({64, 2})));
  const auto target = builder.Cont(seed);
  const auto rows = builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({64, 1})));
  const auto indices = builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({1})));
  builder.Output(builder.CacheStore(target, rows, indices));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = seed, .kind = co::InputKind::kFeatures},
                                {.tensor = rows, .kind = co::InputKind::kFeatures},
                                {.tensor = indices,
                                 .kind = co::InputKind::kStoreCells,
                                 .segment = 0,
                                 .cells = {.capacity = 2}}},
                     .weights = {},
                     .state = {}};
  const co::GraphBuildIO io = co::GraphIO(step);
  kn::Addresses addresses{};
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  auto plan = en::PlanComponentGraph(io, recipe, {}, addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());

  std::array<kn::ComparisonBindingRange, 3> ranges{};
  std::array<co::ComparisonExternalBinding, 3> external{};
  const std::array roots{seed, rows, indices};
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const auto bytes = ir::StorageBytes(step.graph.tensor(roots[i]));
    ASSERT_TRUE(bytes);
    ranges[i] = {.original_root = roots[i],
                 .allocation_class = static_cast<std::uint32_t>(i + 1),
                 .allocation_bytes = *bytes,
                 .bytes = *bytes,
                 .alignment = 16};
    external[i] = {.original_root = roots[i],
                   .allocation_class = static_cast<std::uint32_t>(i + 1),
                   .allocation_bytes = *bytes,
                   .bytes = *bytes,
                   .alignment = 16};
  }
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, external},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = ranges, .expected_boundary = &*boundary};
  const llmp::planner::StepShape shape{.stage = 0,
                                       .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  kn::LowerInput lower_input(*plan, step, shape, addresses, 0, {}, &request);
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  const auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kPhysicalClosure);
}

TEST_F(GgmlComparisonLowerForGpu, ThreeStoreFixtureUsesOneSharedGeneratedCellInput) {
  constexpr std::size_t kStores = 3;
  ir::Builder builder;
  builder.SetOwner(0);
  const auto rows =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, kStores})));
  const auto cells = builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({kStores})));
  std::vector<ir::TensorId> states(kStores);
  for (std::uint16_t i = 0; i < kStores; ++i) {
    const ir::StateRef ref{.stage = 0, .tensor = i, .layer = 0, .segment = 0};
    states[i] = builder.State(ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({16, 8})), ref);
    (void)builder.CacheStore(states[i], rows, cells);
  }
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures},
                                {.tensor = cells,
                                 .kind = co::InputKind::kStoreCells,
                                 .segment = 0,
                                 .cells = {.capacity = 8}}},
                     .weights = {},
                     .state = {}};
  for (std::uint16_t i = 0; i < kStores; ++i) {
    const ir::StateRef ref{.stage = 0, .tensor = i, .layer = 0, .segment = 0};
    step.state.push_back({.tensor = states[i], .ref = ref, .item = i, .written = true});
  }
  const co::GraphBuildIO io = co::GraphIO(step);
  kn::Addresses addresses{};
  const auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const llmp::planner::StepShape shape{
      .stage = 0,
      .segments = {
          {.slot = 0, .rows = static_cast<std::uint32_t>(kStores), .past = 0, .outputs = 0}}};
  const auto plan = en::PlanComponentGraph(io, recipe, {}, addresses, {.shape = &shape});
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  ASSERT_EQ(plan->steps.size(), 1U);
  EXPECT_EQ(plan->steps.front().batch, "grouped_stores");
  EXPECT_EQ(plan->steps.front().singles.size(), kStores);
  EXPECT_EQ(step.inputs[1].kind, co::InputKind::kStoreCells);
  EXPECT_EQ(step.inputs[1].cells.capacity, 8U);
  EXPECT_EQ(step.state.size(), kStores);
  for (const auto& state : step.state) EXPECT_TRUE(state.written);
}

TEST_F(GgmlComparisonLowerForGpu, ThreeStoreOwnedIndicesReplayAtUpdatedPastAcrossAllStates) {
  constexpr std::size_t kStores = 3;
  constexpr std::uint32_t kWidth = 16;
  constexpr std::uint32_t kCapacity = 8;
  constexpr std::size_t kActivationBytes = 1U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  ir::Builder builder;
  builder.SetOwner(0);
  const auto rows =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, kStores})));
  const auto cells = builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({kStores})));
  std::vector<ir::TensorId> states(kStores);
  for (std::uint16_t i = 0; i < kStores; ++i) {
    const ir::StateRef ref{.stage = 0, .tensor = i, .layer = 0, .segment = 0};
    states[i] = builder.State(
        ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({kWidth, kCapacity})), ref);
    (void)builder.CacheStore(states[i], rows, cells);
  }
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures},
                                {.tensor = cells,
                                 .kind = co::InputKind::kStoreCells,
                                 .segment = 0,
                                 .cells = {.capacity = kCapacity}}},
                     .weights = {},
                     .state = {}};
  for (std::size_t i = 0; i < kStores; ++i) {
    const ir::StateRef ref{
        .stage = 0, .tensor = static_cast<std::uint16_t>(i), .layer = 0, .segment = 0};
    step.state.push_back(
        {.tensor = states[i], .ref = ref, .item = static_cast<std::uint16_t>(i), .written = true});
  }
  const co::GraphBuildIO io = co::GraphIO(step);
  const llmp::planner::StepShape shape{
      .stage = 0, .segments = {{.slot = 0, .rows = kStores, .past = 0, .outputs = 0}}};

  CudaBuffer activation, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kActivationBytes));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  std::array<CudaBuffer, kStores> state_buffers{};
  std::array<std::uint64_t, kStores> state_addresses{};
  for (std::size_t i = 0; i < kStores; ++i) {
    ASSERT_TRUE(state_buffers[i].Allocate(kWidth * kCapacity * sizeof(__half)));
    state_addresses[i] = state_buffers[i].address();
    ASSERT_EQ(cudaMemset(state_buffers[i].pointer, 0, kWidth * kCapacity * sizeof(__half)),
              cudaSuccess);
  }
  kn::Addresses addresses{.weight = {},
                          .weight_rows = {},
                          .state =
                              [&](const ir::StateRef& ref) {
                                return ref.tensor < state_addresses.size()
                                           ? state_addresses[ref.tensor]
                                           : std::uint64_t{0};
                              },
                          .save = {},
                          .save_rows = 0,
                          .activations = activation.address(),
                          .activation_bytes = kActivationBytes,
                          .activation_alignment = 256};
  addresses.state_range = [&](const ir::StateRef& ref) -> std::optional<kn::BoundedAddress> {
    if (ref.tensor >= state_addresses.size()) return {};
    return kn::BoundedAddress{.base = state_addresses[ref.tensor],
                              .bytes = kWidth * kCapacity * sizeof(__half),
                              .alignment = 256};
  };
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  auto plan = en::PlanComponentGraph(io, recipe, {}, addresses, {.shape = &shape});
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  ASSERT_EQ(plan->steps.size(), 1U);
  EXPECT_EQ(plan->steps.front().batch, "grouped_stores");
  EXPECT_EQ(plan->steps.front().nodes.size(), kStores);
  EXPECT_EQ(plan->steps.front().singles.size(), kStores);

  const auto address_of = [&](const planner::StepPlan& selected,
                              ir::TensorId original) -> std::optional<std::uint64_t> {
    if (!original.valid() || original.v >= selected.tensors.size()) return {};
    const auto mapped = selected.tensors[original.v];
    if (!mapped.valid()) return {};
    const auto root = selected.graph.root(mapped);
    const auto found = std::ranges::find_if(selected.placement.offsets, [&](const auto& entry) {
      return selected.graph.root(entry.first) == root;
    });
    if (found == selected.placement.offsets.end() || found->second >= kActivationBytes) return {};
    return activation.address() + found->second;
  };
  const auto rows_address = address_of(*plan, rows);
  const auto cells_address = address_of(*plan, cells);
  ASSERT_TRUE(rows_address);
  ASSERT_TRUE(cells_address);
  std::array<float, kWidth * kStores> host_rows{};
  for (std::size_t i = 0; i < host_rows.size(); ++i)
    host_rows[i] = static_cast<float>(i + 1) * 0.125F - 1.0F;
  ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(*rows_address), host_rows.data(), sizeof(host_rows),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  const std::array<std::int64_t, kStores> poison_indices{99, 101, 103};
  ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(*cells_address), poison_indices.data(),
                       sizeof(poison_indices), cudaMemcpyHostToDevice),
            cudaSuccess);

  const auto rows_bytes = ir::StorageBytes(step.graph.tensor(rows));
  const auto cells_bytes = ir::StorageBytes(step.graph.tensor(cells));
  ASSERT_TRUE(rows_bytes && cells_bytes);
  auto source_bindings = en::ComparisonSourceBindings::Create(
      std::make_shared<PaidBoundaryFunder>(life_->paid), step, 0);
  ASSERT_TRUE(source_bindings);
  auto resolved = source_bindings->Resolve(*plan, addresses, recipe.written);
  ASSERT_TRUE(resolved) << (resolved ? "" : resolved.error().detail);
  const auto bindings = resolved->ranges();
  const auto source_view = source_bindings->boundary_view();
  ASSERT_TRUE(source_view);
  const auto boundary_view = *source_view;
  const auto* grouped_pattern = planner::FindPattern("grouped_stores");
  ASSERT_NE(grouped_pattern, nullptr);
  planner::ComparisonSiteSelector selector{.source_graph_identity = step.graph.identity(),
                                           .kind = planner::ComparisonSiteKind::kBatch,
                                           .pattern = "grouped_stores",
                                           .version = grouped_pattern->version,
                                           .member_origins = {},
                                           .member_owner_segments = {},
                                           .boundary_ports = {}};
  for (std::uint32_t n = 0; n < step.graph.nodes().size(); ++n)
    if (step.graph.node(ir::NodeId{n}).key.op == ir::Op::kCacheStore)
      selector.member_origins.push_back({ir::NodeId{n}});
  ASSERT_EQ(selector.member_origins.size(), kStores);
  const planner::PlanOptions comparison_options{
      .capturable = true, .within = recipe.within, .shape = &shape};
  auto inventory = planner::BoundComparison(
      boundary_view, selector, std::make_shared<PaidBoundaryFunder>(life_->paid), *recipe.registry,
      recipe.device, {}, comparison_options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  ASSERT_EQ(inventory->incumbent.plan.steps.size(), 1U);
  EXPECT_EQ(inventory->incumbent.plan.steps.front().batch, "grouped_stores");
  EXPECT_EQ(inventory->incumbent.plan.steps.front().nodes.size(), kStores);
  auto alternative = planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto alternative_plan = planner::PlanAlternative(
      *inventory, std::move(*alternative), *recipe.registry, recipe.device, {}, comparison_options);
  ASSERT_TRUE(alternative_plan) << (alternative_plan ? "" : alternative_plan.error().detail);
  EXPECT_FALSE(std::ranges::any_of(alternative_plan->plan.steps, [&](const auto& candidate) {
    return candidate.batch == "grouped_stores" && candidate.nodes.size() == kStores;
  }));
  const auto grouped_alternative = std::ranges::find_if(
      alternative_plan->plan.steps,
      [](const auto& candidate) { return candidate.batch == "grouped_stores"; });
  ASSERT_NE(grouped_alternative, alternative_plan->plan.steps.end());
  EXPECT_EQ(grouped_alternative->nodes.size(), kStores - 1);
  EXPECT_EQ(address_of(alternative_plan->plan, rows), rows_address);
  EXPECT_EQ(address_of(alternative_plan->plan, cells), cells_address);
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                .scope = kn::ComparisonProofScope::kWholeProgramNative,
                                .bindings = bindings,
                                .state = step.state,
                                .active_stage = 0,
                                .expected_boundary = &inventory->source_boundary};
  kn::LowerInput lower_input(inventory->incumbent.plan, step, shape, addresses, 0, {}, &request);
  kn::LowerInput alternative_input(alternative_plan->plan, step, shape, addresses, 0, {}, &request);
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  auto alternative_program = life_->backend->Lower(alternative_input);
  ASSERT_TRUE(alternative_program) << (alternative_program ? "" : alternative_program.error());
  // A second backend-owned program over the same immutable source facts gets
  // an independent generated payload owner. The recording must retain both.
  auto second_program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(second_program) << (second_program ? "" : second_program.error());
  auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified)
      << "missing=" << static_cast<int>(seal->missing);
  ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
  auto alternative_seal = (*alternative_program)->comparison_recipe();
  ASSERT_TRUE(alternative_seal);
  ASSERT_EQ(alternative_seal->state, kn::ComparisonProofState::kDescriptorVerified)
      << "missing=" << static_cast<int>(alternative_seal->missing);
  ASSERT_EQ(alternative_seal->missing, kn::ComparisonMissing::kNone);
  EXPECT_EQ(alternative_seal->subject, seal->subject);
  EXPECT_EQ(alternative_seal->boundary, seal->boundary);
  EXPECT_NE(alternative_seal->physical_recipe, seal->physical_recipe);
  std::vector<std::uint32_t> whole_offsets{0};
  std::vector<ir::NodeId> whole_members;
  for (const auto& group : inventory->site.accepted_member_origins) {
    whole_members.insert(whole_members.end(), group.begin(), group.end());
    whole_offsets.push_back(static_cast<std::uint32_t>(whole_members.size()));
  }
  auto whole_boundary = co::MakeComparisonRegionBoundary(
      {.source = boundary_view, .member_offsets = whole_offsets, .members = whole_members},
      std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(whole_boundary);
  const auto whole_key = en::RecipeComparisonKeyForWholePrograms(
      inventory->source_boundary, *whole_boundary, inventory->site, shape.phase,
      recipe.registry->SelectionDigest(), *seal, *alternative_seal);
  ASSERT_TRUE(whole_key);
  EXPECT_EQ((*program)->comparison_plan_identity()->source_graph_identity, step.graph.identity());
  const auto descriptor_seal = *seal;
  const auto alternative_descriptor_seal = *alternative_seal;
  // Install this exact complete-program pair as synthetic rank-store data;
  // production must derive the identical key without a local certificate.
  std::vector<llmp::execution::Registry> production_registries;
  production_registries.reserve(4);
  std::vector<std::unique_ptr<kn::StepProgram>> production;
  const std::array<std::array<std::uint64_t, 2>, 4> latencies{
      {{100, 50}, {50, 100}, {100, 100}, {0, 0}}};
  for (std::size_t choice = 0; choice < latencies.size(); ++choice) {
    SCOPED_TRACE(choice);
    std::vector<llmp::execution::Implementation> declarations;
    for (std::size_t i = 0; i < recipe.registry->size(); ++i)
      declarations.push_back(recipe.registry->at(i));
    auto ranks = recipe.registry->ranks();
    ranks.recipe_comparisons = {
        {.key = *whole_key,
         .incumbent_ns = latencies[choice][0],
         .alternative_ns = latencies[choice][1],
         .samples = choice == 3 ? 0U : 3U,
         .protocol = llmp::base::Sha256{}.Update("whole-test-protocol").Finish(),
         .receipt = llmp::base::Sha256{}.Update("whole-test-receipt").Finish(),
         .provenance = "synthetic-reviewed-test",
         .shape_class = "three-stores",
         .kind = choice == 3 ? llmp::execution::RecipeComparisonKind::kRejectIncumbentQuality
                             : llmp::execution::RecipeComparisonKind::kLatency}};
    auto measured_registry =
        llmp::execution::Registry::Create(std::move(declarations), std::move(ranks));
    ASSERT_TRUE(measured_registry) << (measured_registry ? "" : measured_registry.error().detail);
    production_registries.push_back(std::move(*measured_registry));
    auto production_backend = recipe;
    production_backend.registry = &production_registries.back();
    std::uint32_t lowered_count = 0;
    auto selected = en::BuildCumulativeRecipeProgram(
        io, step, shape, 0, 0, production_backend, {}, addresses, comparison_options,
        [&](const kn::LowerInput&, const kn::StepProgram&) -> std::expected<void, std::string> {
          ++lowered_count;
          return {};
        },
        backend.owner, nullptr);
    ASSERT_TRUE(selected) << (selected ? "" : selected.error());
    ASSERT_EQ(lowered_count, 2U) << "whole decisions finish cumulative selection";
    ASSERT_TRUE((*selected)->Bind(**launch));
    ASSERT_NE((*selected)->bound(), nullptr);
    const bool choose_alternative = choice == 0 || choice == 3;
    EXPECT_EQ((*selected)->bound()->size(), choose_alternative
                                                ? alternative_plan->plan.steps.size()
                                                : inventory->incumbent.plan.steps.size());
    EXPECT_TRUE(en::SameWholeProgramRecipe(
        choose_alternative ? alternative_descriptor_seal : descriptor_seal,
        *(*selected)->comparison_recipe()));
    production.push_back(std::move(*selected));
  }
  ASSERT_TRUE((*program)->Bind(**launch));
  ASSERT_TRUE((*second_program)->Bind(**launch));
  ASSERT_TRUE((*alternative_program)->Bind(**launch));
  seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  ASSERT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified);
  ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
  alternative_seal = (*alternative_program)->comparison_recipe();
  ASSERT_TRUE(alternative_seal);
  ASSERT_EQ(alternative_seal->state, kn::ComparisonProofState::kBoundVerified);
  ASSERT_EQ(alternative_seal->missing, kn::ComparisonMissing::kNone);

  EXPECT_TRUE(en::SameWholeProgramRecipe(descriptor_seal, *seal));
  EXPECT_TRUE(en::SameWholeProgramRecipe(alternative_descriptor_seal, *alternative_seal));

  auto native = life_->node.execution().Submission(life_->node.stream(0));
  ASSERT_TRUE(native) << native.error().detail;
  const auto expect_state_at = [&](std::uint32_t past) {
    std::array<__half, kWidth * kCapacity> expected{};
    for (std::uint32_t row = 0; row < kStores; ++row)
      for (std::uint32_t col = 0; col < kWidth; ++col)
        expected[(past + row) * kWidth + col] = __float2half(host_rows[row * kWidth + col]);
    for (const auto& state : state_buffers) {
      std::array<__half, kWidth * kCapacity> actual{};
      ASSERT_EQ(cudaMemcpy(actual.data(), state.pointer, sizeof(actual), cudaMemcpyDeviceToHost),
                cudaSuccess);
      SCOPED_TRACE(past);
      EXPECT_EQ(std::memcmp(actual.data(), expected.data(), sizeof(expected)), 0);
    }
  };
  auto run_shape = shape;
  auto recorder =
      (*program)->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
        if (auto ran = (*program)->Run(**launch, *native); !ran) return ran;
        if (auto second = (*second_program)->Run(**launch, *native); !second) return second;
        if (auto repeated = (*program)->Run(**launch, *native); !repeated) return repeated;
        if (auto second = (*second_program)->Run(**launch, *native); !second) return second;
        auto during_capture = run_shape;
        during_capture.segments.front().past = 1;
        EXPECT_FALSE((*program)->TryUpdateGeneratedInputs(during_capture));
        return {};
      });
  ASSERT_TRUE(recorder) << (recorder ? "" : recorder.error().detail);
  // Two live recordings of the same owner each retain their own prepaid
  // source entry; replay both without an intervening completion fence.
  auto second_recorder =
      (*program)->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
        if (auto ran = (*program)->Run(**launch, *native); !ran) return ran;
        return (*second_program)->Run(**launch, *native);
      });
  ASSERT_TRUE(second_recorder) << (second_recorder ? "" : second_recorder.error().detail);
  auto alternative_recorder =
      (*alternative_program)
          ->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
            return (*alternative_program)->Run(**launch, *native);
          });
  ASSERT_TRUE(alternative_recorder)
      << (alternative_recorder ? "" : alternative_recorder.error().detail);
  ASSERT_TRUE((*launch)->Launch(*native, **recorder));
  auto invalid_while_submitted = run_shape;
  invalid_while_submitted.segments.front().past = 6;
  EXPECT_FALSE((*program)->TryUpdateGeneratedInputs(invalid_while_submitted));
  ASSERT_TRUE((*launch)->Launch(*native, **second_recorder));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  expect_state_at(0);
  std::array<std::array<__half, kWidth * kCapacity>, kStores> grouped_state{};
  for (std::size_t i = 0; i < kStores; ++i)
    ASSERT_EQ(cudaMemcpy(grouped_state[i].data(), state_buffers[i].pointer,
                         sizeof(grouped_state[i]), cudaMemcpyDeviceToHost),
              cudaSuccess);
  for (auto& state : state_buffers)
    ASSERT_EQ(cudaMemsetAsync(state.pointer, 0, kWidth * kCapacity * sizeof(__half),
                              static_cast<cudaStream_t>(native->handle)),
              cudaSuccess);
  ASSERT_TRUE((*launch)->Launch(*native, **alternative_recorder));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  expect_state_at(0);
  for (std::size_t i = 0; i < kStores; ++i) {
    std::array<__half, kWidth * kCapacity> actual{};
    ASSERT_EQ(
        cudaMemcpy(actual.data(), state_buffers[i].pointer, sizeof(actual), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(std::memcmp(actual.data(), grouped_state[i].data(), sizeof(actual)), 0);
  }

  run_shape.segments.front().past = 6;  // rows would exceed the fixed eight-cell target.
  EXPECT_FALSE((*program)->TryUpdateGeneratedInputs(run_shape));
  run_shape.segments.front().past = 0;
  ASSERT_TRUE((*program)->TryUpdateGeneratedInputs(run_shape));
  ASSERT_TRUE((*second_program)->TryUpdateGeneratedInputs(run_shape));
  ASSERT_TRUE((*alternative_program)->TryUpdateGeneratedInputs(run_shape));
  for (auto& state : state_buffers)
    ASSERT_EQ(cudaMemsetAsync(state.pointer, 0, kWidth * kCapacity * sizeof(__half),
                              static_cast<cudaStream_t>(native->handle)),
              cudaSuccess);
  ASSERT_TRUE((*launch)->Launch(*native, **recorder));
  ASSERT_TRUE((*launch)->Launch(*native, **second_recorder));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  expect_state_at(0);  // rejected past6 left the previously accepted bytes intact.
  for (auto& state : state_buffers)
    ASSERT_EQ(cudaMemsetAsync(state.pointer, 0, kWidth * kCapacity * sizeof(__half),
                              static_cast<cudaStream_t>(native->handle)),
              cudaSuccess);
  ASSERT_TRUE((*launch)->Launch(*native, **alternative_recorder));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  expect_state_at(0);
  for (std::size_t i = 0; i < kStores; ++i) {
    std::array<__half, kWidth * kCapacity> actual{};
    ASSERT_EQ(
        cudaMemcpy(actual.data(), state_buffers[i].pointer, sizeof(actual), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(std::memcmp(actual.data(), grouped_state[i].data(), sizeof(actual)), 0);
  }

  run_shape.segments.front().past = 2;
  ASSERT_TRUE((*program)->TryUpdateGeneratedInputs(run_shape));
  ASSERT_TRUE((*second_program)->TryUpdateGeneratedInputs(run_shape));
  ASSERT_TRUE((*alternative_program)->TryUpdateGeneratedInputs(run_shape));
  for (auto& state : state_buffers)
    ASSERT_EQ(cudaMemsetAsync(state.pointer, 0, kWidth * kCapacity * sizeof(__half),
                              static_cast<cudaStream_t>(native->handle)),
              cudaSuccess);
  ASSERT_TRUE((*launch)->Launch(*native, **recorder));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  expect_state_at(2);
  std::array<std::array<__half, kWidth * kCapacity>, kStores> grouped_state_at_two{};
  for (std::size_t i = 0; i < kStores; ++i)
    ASSERT_EQ(cudaMemcpy(grouped_state_at_two[i].data(), state_buffers[i].pointer,
                         sizeof(grouped_state_at_two[i]), cudaMemcpyDeviceToHost),
              cudaSuccess);
  for (auto& state : state_buffers)
    ASSERT_EQ(cudaMemsetAsync(state.pointer, 0, kWidth * kCapacity * sizeof(__half),
                              static_cast<cudaStream_t>(native->handle)),
              cudaSuccess);
  ASSERT_TRUE((*launch)->Launch(*native, **alternative_recorder));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  expect_state_at(2);
  for (std::size_t i = 0; i < kStores; ++i) {
    std::array<__half, kWidth * kCapacity> actual{};
    ASSERT_EQ(
        cudaMemcpy(actual.data(), state_buffers[i].pointer, sizeof(actual), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(std::memcmp(actual.data(), grouped_state_at_two[i].data(), sizeof(actual)), 0);
  }
  // Exercise the actually adopted owners, including incumbent and neutral
  // decisions, through ordinary execution and one-invocation captures.
  for (std::size_t choice = 0; choice < production.size(); ++choice) {
    SCOPED_TRACE(choice);
    auto& adopted = *production[choice];
    const auto feature =
        std::ranges::find(adopted.inputs(), kn::InputKind::kFeatures, &kn::InputSlot::kind);
    ASSERT_NE(feature, adopted.inputs().end());
    const auto stage = [&](std::uint32_t past) {
      auto next = shape;
      next.segments.front().past = past;
      ASSERT_TRUE(adopted.TryUpdateGeneratedInputs(next));
      for (std::size_t j = 0; j < host_rows.size(); ++j)
        host_rows[j] = (static_cast<float>(j + 1) * 0.0625F - 1.0F) * (past == 0 ? 1.0F : -0.75F);
      for (auto& state : state_buffers)
        ASSERT_EQ(cudaMemsetAsync(state.pointer, 0, kWidth * kCapacity * sizeof(__half),
                                  static_cast<cudaStream_t>(native->handle)),
                  cudaSuccess);
      ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(feature->device_address), host_rows.data(),
                                sizeof(host_rows), cudaMemcpyHostToDevice,
                                static_cast<cudaStream_t>(native->handle)),
                cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
    };
    stage(0);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_TRUE(adopted.Run(**launch, *native));
    ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
    ASSERT_TRUE((*launch)->DrainGeneratedInputs());
    expect_state_at(0);
    auto recorded =
        adopted.Record(**launch, *native, [&]() { return adopted.Run(**launch, *native); });
    ASSERT_TRUE(recorded) << (recorded ? "" : recorded.error().detail);
    for (const std::uint32_t past : {1U, 4U}) {
      stage(past);
      ASSERT_FALSE(HasFatalFailure());
      ASSERT_TRUE((*launch)->Launch(*native, **recorded));
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
      ASSERT_TRUE((*launch)->DrainGeneratedInputs());
      expect_state_at(past);
    }
    recorded->reset();
  }
  ForeignStepLaunch foreign;
  EXPECT_FALSE(production.front()->Bind(foreign));
  EXPECT_FALSE(production.front()->Run(**launch, *native));
  EXPECT_FALSE(production.front()->TryUpdateGeneratedInputs(shape));
  recorder->reset();
  second_recorder->reset();
  alternative_recorder->reset();
  second_program->reset();
  alternative_program->reset();
  program->reset();
  production.clear();
  launch->reset();
  life_->backend->LaunchReleased();
}

TEST_F(GgmlComparisonLowerForGpu, QuantizedStoreGroupsCompareAndReplayFullStateBytes) {
  constexpr std::uint32_t kWidth = 32;
  constexpr std::uint32_t kCapacity = 8;
  constexpr std::size_t kActivationBytes = 1U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  const std::array<std::vector<llmp::format::Encoding>, 3> cases{
      std::vector{llmp::format::Encoding::kQ8_0, llmp::format::Encoding::kQ4_0,
                  llmp::format::Encoding::kQ8_0},
      std::vector{llmp::format::Encoding::kQ4_0, llmp::format::Encoding::kQ4_0},
      std::vector{llmp::format::Encoding::kF16, llmp::format::Encoding::kF16,
                  llmp::format::Encoding::kF16}};
  CudaBuffer pool, persistent;
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  for (const auto& encodings : cases) {
    const std::size_t kStores = encodings.size();

    ir::Builder builder;
    builder.SetOwner(0);
    const auto rows =
        builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, kStores})));
    const auto cells =
        builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({kStores})));
    std::vector<ir::TensorId> states(kStores);
    for (std::uint16_t i = 0; i < kStores; ++i) {
      const auto encoding = encodings[i];
      const auto cache_format = llmp::format::TensorFormat{
          .encoding = encoding,
          .layout = encoding == llmp::format::Encoding::kF16 ? llmp::format::Layout::kDense
                                                             : llmp::format::Layout::kGgmlBlocks,
          .shape = ir::ShapeOf({kWidth, kCapacity})};
      const ir::StateRef ref{.stage = 0, .tensor = i, .layer = 0, .segment = 0};
      states[i] = builder.State(cache_format, ref);
      (void)builder.CacheStore(states[i], rows, cells);
    }
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    co::StepGraph step{.graph = std::move(*graph),
                       .inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures},
                                  {.tensor = cells,
                                   .kind = co::InputKind::kStoreCells,
                                   .segment = 0,
                                   .cells = {.capacity = kCapacity}}},
                       .weights = {},
                       .state = {}};
    for (std::size_t i = 0; i < kStores; ++i) {
      step.state.push_back(
          {.tensor = states[i],
           .ref = {.stage = 0, .tensor = static_cast<std::uint16_t>(i), .layer = 0, .segment = 0},
           .item = static_cast<std::uint16_t>(i),
           .written = true});
    }
    const co::GraphBuildIO io = co::GraphIO(step);
    const llmp::planner::StepShape shape{
        .stage = 0,
        .segments = {
            {.slot = 0, .rows = static_cast<std::uint32_t>(kStores), .past = 0, .outputs = 0}}};

    CudaBuffer activation;
    ASSERT_TRUE(activation.Allocate(kActivationBytes));
    std::vector<CudaBuffer> state_buffers(kStores);
    std::vector<std::uint64_t> state_addresses(kStores);
    std::vector<std::uint64_t> state_bytes(kStores);
    for (std::size_t i = 0; i < kStores; ++i) {
      const auto bytes = ir::StorageBytes(step.graph.tensor(states[i]));
      ASSERT_TRUE(bytes);
      const auto parts = llmp::format::PartsOf(step.graph.tensor(states[i]).format);
      ASSERT_TRUE(parts);
      const auto data = parts->Find(llmp::format::PartRole::kData);
      ASSERT_TRUE(data);
      state_bytes[i] = std::max(*bytes, data->readable);
      ASSERT_TRUE(state_buffers[i].Allocate(state_bytes[i]));
      state_addresses[i] = state_buffers[i].address();
      ASSERT_EQ(cudaMemset(state_buffers[i].pointer, 0, state_bytes[i]), cudaSuccess);
    }
    kn::Addresses addresses{.weight = {},
                            .weight_rows = {},
                            .state =
                                [&](const ir::StateRef& ref) {
                                  return ref.tensor < state_addresses.size()
                                             ? state_addresses[ref.tensor]
                                             : std::uint64_t{0};
                                },
                            .save = {},
                            .save_rows = 0,
                            .activations = activation.address(),
                            .activation_bytes = kActivationBytes,
                            .activation_alignment = 256};
    addresses.state_range = [&](const ir::StateRef& ref) -> std::optional<kn::BoundedAddress> {
      if (ref.tensor >= state_addresses.size()) return {};
      return kn::BoundedAddress{
          .base = state_addresses[ref.tensor], .bytes = state_bytes[ref.tensor], .alignment = 256};
    };
    auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    auto plan = en::PlanComponentGraph(io, recipe, {}, addresses, {.shape = &shape});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error());
    ASSERT_EQ(plan->steps.size(), 1U);
    ASSERT_EQ(plan->steps.front().batch, "grouped_stores");
    ASSERT_EQ(plan->steps.front().nodes.size(), kStores);
    const auto address_of = [&](const planner::StepPlan& selected,
                                ir::TensorId original) -> std::optional<std::uint64_t> {
      if (!original.valid() || original.v >= selected.tensors.size()) return {};
      const auto mapped = selected.tensors[original.v];
      if (!mapped.valid()) return {};
      const auto root = selected.graph.root(mapped);
      const auto found = std::ranges::find_if(selected.placement.offsets, [&](const auto& entry) {
        return selected.graph.root(entry.first) == root;
      });
      if (found == selected.placement.offsets.end() || found->second >= kActivationBytes) return {};
      return activation.address() + found->second;
    };
    const auto rows_address = address_of(*plan, rows);
    const auto cells_address = address_of(*plan, cells);
    ASSERT_TRUE(rows_address);
    ASSERT_TRUE(cells_address);
    std::vector<float> host_rows(kWidth * kStores);
    for (std::size_t i = 0; i < host_rows.size(); ++i)
      host_rows[i] = static_cast<float>(static_cast<int>((i * 17) % 41) - 20) * 0.03125F;
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(*rows_address), host_rows.data(),
                         host_rows.size() * sizeof(host_rows[0]), cudaMemcpyHostToDevice),
              cudaSuccess);
    const std::vector<std::int64_t> poison_indices(kStores, 99);
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(*cells_address), poison_indices.data(),
                         poison_indices.size() * sizeof(poison_indices[0]), cudaMemcpyHostToDevice),
              cudaSuccess);

    const auto rows_extent = ir::StorageBytes(step.graph.tensor(rows));
    const auto cells_extent = ir::StorageBytes(step.graph.tensor(cells));
    ASSERT_TRUE(rows_extent && cells_extent);
    auto source_bindings = en::ComparisonSourceBindings::Create(
        std::make_shared<PaidBoundaryFunder>(life_->paid), step, 0);
    ASSERT_TRUE(source_bindings);
    auto resolved = source_bindings->Resolve(*plan, addresses, recipe.written);
    ASSERT_TRUE(resolved) << (resolved ? "" : resolved.error().detail);
    const auto bindings = resolved->ranges();
    const auto source_view = source_bindings->boundary_view();
    ASSERT_TRUE(source_view);
    const auto boundary_view = *source_view;
    const auto* grouped_pattern = planner::FindPattern("grouped_stores");
    ASSERT_NE(grouped_pattern, nullptr);
    planner::ComparisonSiteSelector selector{.source_graph_identity = step.graph.identity(),
                                             .kind = planner::ComparisonSiteKind::kBatch,
                                             .pattern = "grouped_stores",
                                             .version = grouped_pattern->version,
                                             .member_origins = {},
                                             .member_owner_segments = {},
                                             .boundary_ports = {}};
    for (std::uint32_t n = 0; n < step.graph.nodes().size(); ++n)
      if (step.graph.node(ir::NodeId{n}).key.op == ir::Op::kCacheStore)
        selector.member_origins.push_back({ir::NodeId{n}});
    ASSERT_EQ(selector.member_origins.size(), kStores);
    const planner::PlanOptions options{
        .capturable = true, .within = recipe.within, .shape = &shape};
    auto inventory = planner::BoundComparison(boundary_view, selector,
                                              std::make_shared<PaidBoundaryFunder>(life_->paid),
                                              *recipe.registry, recipe.device, {}, options);
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    ASSERT_EQ(inventory->incumbent.plan.steps.size(), 1U);
    ASSERT_EQ(inventory->incumbent.plan.steps.front().batch, "grouped_stores");
    ASSERT_EQ(inventory->incumbent.plan.steps.front().nodes.size(), kStores);
    auto alternative = planner::MakeAlternative(*inventory);
    ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
    auto alternative_plan = planner::PlanAlternative(*inventory, std::move(*alternative),
                                                     *recipe.registry, recipe.device, {}, options);
    ASSERT_TRUE(alternative_plan) << (alternative_plan ? "" : alternative_plan.error().detail);
    EXPECT_FALSE(std::ranges::any_of(alternative_plan->plan.steps, [&](const auto& candidate) {
      return candidate.batch == "grouped_stores" && candidate.nodes.size() == kStores;
    }));
    ASSERT_EQ(address_of(alternative_plan->plan, rows), rows_address);
    ASSERT_EQ(address_of(alternative_plan->plan, cells), cells_address);
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .scope = kn::ComparisonProofScope::kWholeProgramNative,
                                  .bindings = bindings,
                                  .state = step.state,
                                  .active_stage = 0,
                                  .expected_boundary = &inventory->source_boundary};
    kn::LowerInput original_input(inventory->incumbent.plan, step, shape, addresses, 0, {},
                                  &request);
    kn::LowerInput alternative_input(alternative_plan->plan, step, shape, addresses, 0, {},
                                     &request);
    auto original_program = life_->backend->Lower(original_input);
    ASSERT_TRUE(original_program) << (original_program ? "" : original_program.error());
    auto alternative_program = life_->backend->Lower(alternative_input);
    ASSERT_TRUE(alternative_program) << (alternative_program ? "" : alternative_program.error());
    const auto generated_slot = [&](const kn::StepProgram& candidate) -> const kn::InputSlot* {
      const kn::InputSlot* found = nullptr;
      for (const auto& slot : candidate.inputs()) {
        if (slot.tensor != cells) continue;
        if (found != nullptr) return nullptr;
        found = &slot;
      }
      return found;
    };
    const auto* original_cells_slot = generated_slot(**original_program);
    const auto* alternative_cells_slot = generated_slot(**alternative_program);
    ASSERT_NE(original_cells_slot, nullptr);
    ASSERT_NE(alternative_cells_slot, nullptr);
    for (const auto* slot : {original_cells_slot, alternative_cells_slot}) {
      ASSERT_EQ(slot->kind, kn::InputKind::kStoreCells);
      ASSERT_EQ(slot->device_address, *cells_address);
      ASSERT_EQ(slot->bytes, poison_indices.size() * sizeof(poison_indices[0]));
    }
    for (const auto* program : {&*original_program, &*alternative_program}) {
      const auto seal = (*program)->comparison_recipe();
      ASSERT_TRUE(seal);
      ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified)
          << "missing=" << static_cast<int>(seal->missing);
      ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
    }
    const auto original_descriptor = *(*original_program)->comparison_recipe();
    const auto alternative_descriptor = *(*alternative_program)->comparison_recipe();
    auto adopted = AdoptWholeAlternative(step, shape, *inventory, boundary_view, backend, addresses,
                                         options, original_descriptor, alternative_descriptor,
                                         std::make_shared<PaidBoundaryFunder>(life_->paid));
    ASSERT_TRUE(adopted) << (adopted ? "" : adopted.error());
    alternative_program->reset();
    *alternative_program = std::move(adopted->program);
    ASSERT_TRUE((*original_program)->Bind(**launch));
    ASSERT_TRUE((*alternative_program)->Bind(**launch));
    for (const auto* program : {&*original_program, &*alternative_program}) {
      const auto seal = (*program)->comparison_recipe();
      ASSERT_TRUE(seal);
      ASSERT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified)
          << "missing=" << static_cast<int>(seal->missing);
      ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
    }
    EXPECT_TRUE(
        en::SameWholeProgramRecipe(original_descriptor, *(*original_program)->comparison_recipe()));
    EXPECT_TRUE(en::SameWholeProgramRecipe(alternative_descriptor,
                                           *(*alternative_program)->comparison_recipe()));
    auto native = life_->node.execution().Submission(life_->node.stream(0));
    ASSERT_TRUE(native) << native.error().detail;
    const auto expected_state = [&](std::uint32_t past) {
      std::vector<std::vector<std::byte>> expected(kStores);
      for (std::size_t i = 0; i < kStores; ++i) {
        expected[i].resize(state_bytes[i], std::byte{0});
        const llmp::format::TensorFormat tensor_format{
            .encoding = encodings[i],
            .layout = encodings[i] == llmp::format::Encoding::kF16
                          ? llmp::format::Layout::kDense
                          : llmp::format::Layout::kGgmlBlocks,
            .shape = ir::ShapeOf({kWidth, kCapacity})};
        const auto row_values = std::span<const float>(host_rows);
        const auto encoded = reference::EncodeRows(tensor_format, 0, past, row_values, expected[i]);
        EXPECT_TRUE(encoded) << "state=" << i;
      }
      return expected;
    };
    const auto run_and_check = [&](const std::unique_ptr<kn::Recorded>& recorded,
                                   std::uint32_t past) {
      for (std::size_t i = 0; i < kStores; ++i)
        EXPECT_EQ(cudaMemsetAsync(state_buffers[i].pointer, 0, state_bytes[i],
                                  static_cast<cudaStream_t>(native->handle)),
                  cudaSuccess);
      ASSERT_TRUE((*launch)->Launch(*native, *recorded));
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
      ASSERT_TRUE((*launch)->DrainGeneratedInputs());
      const auto expected = expected_state(past);
      for (std::size_t i = 0; i < kStores; ++i) {
        std::vector<std::byte> actual(state_bytes[i]);
        ASSERT_EQ(cudaMemcpy(actual.data(), state_buffers[i].pointer, state_bytes[i],
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(actual, expected[i]) << "state=" << i << " past=" << past;
      }
    };
    ASSERT_TRUE((*alternative_program)->Run(**launch, *native));
    ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
    ASSERT_TRUE((*launch)->DrainGeneratedInputs());
    const auto ordinary_expected = expected_state(0);
    for (std::size_t i = 0; i < kStores; ++i) {
      std::vector<std::byte> actual(state_bytes[i]);
      ASSERT_EQ(cudaMemcpy(actual.data(), state_buffers[i].pointer, state_bytes[i],
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      EXPECT_EQ(actual, ordinary_expected[i]);
    }
    if (WholeTimingConfigured())
      ASSERT_TRUE((*launch)->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
    auto original_record =
        (*original_program)
            ->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
              return (*original_program)->Run(**launch, *native);
            });
    ASSERT_TRUE(original_record) << (original_record ? "" : original_record.error().detail);
    auto alternative_record =
        (*alternative_program)
            ->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
              return (*alternative_program)->Run(**launch, *native);
            });
    ASSERT_TRUE(alternative_record)
        << (alternative_record ? "" : alternative_record.error().detail);
    if (WholeTimingConfigured())
      ASSERT_TRUE((*launch)->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
    run_and_check(*original_record, 0);
    run_and_check(*alternative_record, 0);

    auto rejected = shape;
    rejected.segments.front().past = kCapacity - static_cast<std::uint32_t>(kStores) + 1;
    ASSERT_FALSE((*original_program)->TryUpdateGeneratedInputs(rejected));
    ASSERT_FALSE((*alternative_program)->TryUpdateGeneratedInputs(rejected));
    run_and_check(*original_record, 0);
    run_and_check(*alternative_record, 0);
    for (auto& value : host_rows) value = value * -0.75F + 0.125F;
    ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(*rows_address), host_rows.data(),
                         host_rows.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    auto accepted = shape;
    accepted.segments.front().past = 2;
    ASSERT_TRUE((*original_program)->TryUpdateGeneratedInputs(accepted));
    ASSERT_TRUE((*alternative_program)->TryUpdateGeneratedInputs(accepted));
    run_and_check(*original_record, 2);
    run_and_check(*alternative_record, 2);
    std::array<std::unique_ptr<kn::Recorded>, 2> timing_captures{std::move(*original_record),
                                                                 std::move(*alternative_record)};
    MeasureWholeStatePair(
        encodings[0] == llmp::format::Encoding::kF16 ? "stores-f16-f16-f16-r3"
        : kStores == 3                               ? "stores-q8-q4-q8-r3"
                                                     : "stores-q4-q4-r2",
        adopted->registry->ranks().recipe_comparisons.front().key, timing_captures,
        [&](std::size_t arm) {
          ASSERT_EQ(cudaMemset(activation.pointer, 0xff, kActivationBytes), cudaSuccess);
          ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(*rows_address), host_rows.data(),
                               host_rows.size() * sizeof(float), cudaMemcpyHostToDevice),
                    cudaSuccess);
          run_and_check(timing_captures[arm], 2);
        });
    ASSERT_FALSE(HasFailure());
    timing_captures = {};
    original_record->reset();
    alternative_record->reset();
    original_program->reset();
    alternative_program->reset();
  }
  launch->reset();
  life_->backend->LaunchReleased();
}

TEST_F(GgmlComparisonLowerForGpu, QuantizedRopeStoreAlternativeMatchesReferenceAcrossUpdatedPast) {
  constexpr std::uint32_t kWidth = 64;
  constexpr std::uint32_t kCapacity = 8;
  constexpr std::uint32_t kRows = 2;
  constexpr std::size_t kActivationBytes = 1U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  for (const auto encoding : {llmp::format::Encoding::kQ8_0, llmp::format::Encoding::kQ4_0}) {
    const llmp::format::TensorFormat cache_format{.encoding = encoding,
                                                  .layout = llmp::format::Layout::kGgmlBlocks,
                                                  .shape = ir::ShapeOf({kWidth, kCapacity + 1})};
    auto row_bytes = llmp::format::RowBytes(cache_format, llmp::format::PartRole::kData);
    ASSERT_TRUE(row_bytes) << (row_bytes ? "" : llmp::format::FormatErrorName(row_bytes.error()));

    ir::Builder builder;
    builder.SetOwner(0);
    const auto features =
        builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth + 1, 1, kRows})));
    const auto positions =
        builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({kRows})));
    const auto cells = builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({kRows})));
    const ir::StateRef state_ref{.stage = 0, .tensor = 0, .layer = 0, .segment = 0};
    const auto cache = builder.State(cache_format, state_ref);
    const auto cache_window =
        builder.View(cache, ir::ShapeOf({kWidth, kCapacity}), builder.tensor(cache).nb, *row_bytes);
    const auto target = builder.View(cache_window, ir::ShapeOf({kWidth, kCapacity}),
                                     builder.tensor(cache_window).nb, 0);
    const auto feature_window = builder.View(features, ir::ShapeOf({kWidth, 1, kRows}),
                                             builder.tensor(features).nb, sizeof(float));
    const auto feature_view = builder.View(feature_window, ir::ShapeOf({kWidth, 1, kRows}),
                                           builder.tensor(feature_window).nb, 0);
    const ir::RopeParams rope_params{.mode = ir::RopeMode::kNeox,
                                     .dims = kWidth,
                                     .original_context = 4096,
                                     .base = 10000.0F,
                                     .scale = 1.0F,
                                     .beta_fast = 32.0F,
                                     .beta_slow = 1.0F};
    const auto rope = builder.Rope(feature_view, positions, {}, rope_params);
    const auto rows = builder.Reshape(rope, ir::ShapeOf({kWidth, kRows}));
    const auto stored = builder.CacheStore(target, rows, cells);
    builder.Keep(features);
    builder.Output(stored);
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);

    co::StepGraph step{
        .graph = std::move(*graph),
        .inputs = {{.tensor = features, .kind = co::InputKind::kFeatures},
                   {.tensor = positions, .kind = co::InputKind::kPositions},
                   {.tensor = cells,
                    .kind = co::InputKind::kStoreCells,
                    .segment = 0,
                    .cells = {.capacity = kCapacity}}},
        .weights = {},
        .state = {{.tensor = cache, .ref = state_ref, .read = true, .written = true}}};
    const auto io = co::GraphIO(step);
    const planner::StepShape initial_shape{
        .stage = 0, .segments = {{.slot = 0, .rows = kRows, .past = 0, .outputs = 0}}};
    auto backend = life_->backend->Backend();
    const auto& recipe = backend.owner ? backend.owner->backend() : backend;
    const auto* pattern = planner::FindPattern("rope_store");
    ASSERT_NE(pattern, nullptr);
    planner::ComparisonSiteSelector selector{.source_graph_identity = step.graph.identity(),
                                             .kind = planner::ComparisonSiteKind::kRewrite,
                                             .pattern = "rope_store",
                                             .version = pattern->version,
                                             .member_origins = {{step.graph.producer(rope)},
                                                                {step.graph.producer(rows)},
                                                                {step.graph.producer(stored)}},
                                             .member_owner_segments = {},
                                             .boundary_ports = {}};

    const std::array roots{features, positions, cells, cache};
    std::array<std::uint64_t, roots.size()> root_bytes{};
    std::array<co::ComparisonExternalBinding, roots.size()> external{};
    for (std::size_t i = 0; i < roots.size(); ++i) {
      const auto bytes = ir::StorageBytes(step.graph.tensor(roots[i]));
      ASSERT_TRUE(bytes);
      root_bytes[i] = *bytes;
      if (roots[i] == cache) {
        const auto parts = llmp::format::PartsOf(step.graph.tensor(cache).format);
        ASSERT_TRUE(parts);
        const auto data = parts->Find(llmp::format::PartRole::kData);
        ASSERT_TRUE(data);
        root_bytes[i] = std::max(root_bytes[i], data->readable);
      }
      external[i] = {.original_root = roots[i],
                     .allocation_class = static_cast<std::uint32_t>(i + 1),
                     .allocation_bytes = root_bytes[i],
                     .bytes = root_bytes[i],
                     .alignment = 256};
    }
    const auto state_storage_bytes = ir::StorageBytes(step.graph.tensor(cache));
    ASSERT_TRUE(state_storage_bytes);
    const co::ComparisonBoundaryView boundary_view{io, step.state, 0, external};
    const planner::PlanOptions options{
        .capturable = true, .within = recipe.within, .shape = &initial_shape};
    auto inventory = planner::BoundComparison(boundary_view, selector,
                                              std::make_shared<PaidBoundaryFunder>(life_->paid),
                                              *recipe.registry, recipe.device, {}, options);
    ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
    ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
    EXPECT_EQ(inventory->incumbent.plan.applied.front().pattern, "rope_store");
    ASSERT_EQ(inventory->incumbent.plan.steps.size(), 1U);
    EXPECT_EQ(recipe.registry->at(inventory->incumbent.plan.steps.front().implementation).name,
              kg::kRopeSetRowsQuantName);
    auto alternative = planner::MakeAlternative(*inventory);
    ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
    auto planned = planner::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                            recipe.device, {}, options);
    ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
    EXPECT_TRUE(std::ranges::none_of(planned->plan.applied, [](const auto& applied) {
      return applied.pattern == "rope_store";
    }));

    constexpr std::size_t kArms = 2;
    std::array<CudaBuffer, kArms> activations{}, state_buffers{};
    CudaBuffer pool, persistent;
    ASSERT_TRUE(pool.Allocate(kPoolBytes));
    ASSERT_TRUE(persistent.Allocate(life_->backend->persistent_bytes()));
    auto launch = life_->backend->Launch(
        {.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
        {.base = persistent.address(), .bytes = life_->backend->persistent_bytes()});
    ASSERT_TRUE(launch) << (launch ? "" : launch.error());
    for (std::size_t arm = 0; arm < kArms; ++arm) {
      ASSERT_TRUE(activations[arm].Allocate(kActivationBytes));
      ASSERT_TRUE(state_buffers[arm].Allocate(root_bytes[3]));
      ASSERT_EQ(cudaMemset(state_buffers[arm].pointer, 0, root_bytes[3]), cudaSuccess);
    }
    const std::array<const planner::StepPlan*, kArms> plans{&inventory->incumbent.plan,
                                                            &planned->plan};
    std::array<std::unique_ptr<kn::StepProgram>, kArms> programs{};
    std::array<kn::Addresses, kArms> arm_addresses{};
    std::array<kn::ComparisonRecipeSeal, kArms> descriptors{};
    std::array<std::array<std::uint64_t, 3>, kArms> input_addresses{};
    const auto get_input_address = [&](const planner::StepPlan& plan, std::size_t arm,
                                       ir::TensorId original) -> std::optional<std::uint64_t> {
      if (!original.valid() || original.v >= plan.tensors.size()) return {};
      const auto mapped = plan.tensors[original.v];
      if (!mapped.valid()) return {};
      const auto root = plan.graph.root(mapped);
      const auto placed = std::ranges::find_if(plan.placement.offsets, [&](const auto& entry) {
        return plan.graph.root(entry.first) == root;
      });
      if (placed == plan.placement.offsets.end() || placed->second >= kActivationBytes) return {};
      return activations[arm].address() + placed->second;
    };
    for (std::size_t arm = 0; arm < kArms; ++arm) {
      ASSERT_LE(plans[arm]->placement.extent, kActivationBytes);
      kn::Addresses addresses{.weight = {},
                              .weight_rows = {},
                              .state =
                                  [&, arm](const ir::StateRef& ref) {
                                    return ref == state_ref ? state_buffers[arm].address()
                                                            : std::uint64_t{0};
                                  },
                              .save = {},
                              .save_rows = 0,
                              .activations = activations[arm].address(),
                              .activation_bytes = kActivationBytes,
                              .activation_alignment = 256};
      addresses.state_range = [&,
                               arm](const ir::StateRef& ref) -> std::optional<kn::BoundedAddress> {
        if (ref != state_ref) return {};
        return kn::BoundedAddress{
            .base = state_buffers[arm].address(), .bytes = root_bytes[3], .alignment = 256};
      };
      arm_addresses[arm] = addresses;
      const std::array input_roots{features, positions, cells};
      std::array<kn::ComparisonBindingRange, roots.size()> ranges{};
      for (std::size_t i = 0; i < input_roots.size(); ++i) {
        const auto address = get_input_address(*plans[arm], arm, input_roots[i]);
        ASSERT_TRUE(address);
        input_addresses[arm][i] = *address;
        ranges[i] = {.original_root = input_roots[i],
                     .allocation_class = static_cast<std::uint32_t>(i + 1),
                     .allocation_bytes = root_bytes[i],
                     .offset = 0,
                     .bytes = root_bytes[i],
                     .alignment = 256,
                     .base = *address};
      }
      ranges[3] = {.original_root = cache,
                   .allocation_class = 4,
                   .allocation_bytes = root_bytes[3],
                   .bytes = root_bytes[3],
                   .alignment = 256,
                   .base = state_buffers[arm].address()};
      kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                    .scope = kn::ComparisonProofScope::kWholeProgramNative,
                                    .bindings = ranges,
                                    .state = step.state,
                                    .active_stage = 0,
                                    .expected_boundary = &inventory->source_boundary};
      kn::LowerInput lower_input(*plans[arm], step, initial_shape, addresses, 0, {}, &request);
      auto lowered = life_->backend->Lower(lower_input);
      ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error());
      programs[arm] = std::move(*lowered);
      const auto seal = programs[arm]->comparison_recipe();
      ASSERT_TRUE(seal);
      ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified)
          << "missing=" << static_cast<int>(seal->missing);
      ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
      descriptors[arm] = *seal;
    }
    auto adopted = AdoptWholeAlternative(step, initial_shape, *inventory, boundary_view, backend,
                                         arm_addresses[1], options, descriptors[0], descriptors[1],
                                         std::make_shared<PaidBoundaryFunder>(life_->paid));
    ASSERT_TRUE(adopted) << (adopted ? "" : adopted.error());
    programs[1] = std::move(adopted->program);
    for (std::size_t arm = 0; arm < kArms; ++arm) {
      ASSERT_TRUE(programs[arm]->Bind(**launch));
      EXPECT_TRUE(
          en::SameWholeProgramRecipe(descriptors[arm], *programs[arm]->comparison_recipe()));
    }
    for (const auto& program : programs) {
      const auto seal = program->comparison_recipe();
      ASSERT_TRUE(seal);
      ASSERT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified);
      ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
    }

    auto reference = reference::Executor::Create(step.graph);
    ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
    std::vector<std::byte> initial_state(root_bytes[3]);
    const std::array<reference::PartView, 1> state_parts{
        {{.role = llmp::format::PartRole::kData, .bytes = initial_state}}};
    ASSERT_TRUE(reference->Bind(cache, state_parts));
    std::vector<float> host_features((kWidth + 1) * kRows);
    for (std::size_t i = 0; i < host_features.size(); ++i)
      host_features[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.0625F;
    std::array<std::int32_t, kRows> host_positions{};
    const auto stream = life_->node.execution().Submission(life_->node.stream(0));
    ASSERT_TRUE(stream) << stream.error().detail;
    std::array<std::unique_ptr<kn::Recorded>, kArms> recordings{};
    if (WholeTimingConfigured())
      ASSERT_TRUE((*launch)->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
    for (std::size_t arm = 0; arm < kArms; ++arm) {
      const auto input_slots = programs[arm]->inputs();
      const auto slot_for = [&](ir::TensorId tensor) -> const kn::InputSlot* {
        const kn::InputSlot* found = nullptr;
        for (const auto& slot : input_slots) {
          if (slot.tensor != tensor) continue;
          if (found) return nullptr;
          found = &slot;
        }
        return found;
      };
      for (const auto [tensor, kind, slot_index] :
           {std::tuple{features, kn::InputKind::kFeatures, std::size_t{0}},
            std::tuple{positions, kn::InputKind::kPositions, std::size_t{1}},
            std::tuple{cells, kn::InputKind::kStoreCells, std::size_t{2}}}) {
        const auto* slot = slot_for(tensor);
        ASSERT_NE(slot, nullptr);
        ASSERT_EQ(slot->kind, kind);
        ASSERT_EQ(slot->device_address, input_addresses[arm][slot_index]);
      }
      auto record = programs[arm]->Record(**launch, *stream,
                                          [&]() { return programs[arm]->Run(**launch, *stream); });
      ASSERT_TRUE(record) << (record ? "" : record.error().detail);
      recordings[arm] = std::move(*record);
    }
    if (WholeTimingConfigured())
      ASSERT_TRUE((*launch)->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
    bool ordinary_checked = false;
    const auto run_frame = [&](std::uint32_t past, int only_arm = -1) {
      if (only_arm >= 0) {
        ASSERT_TRUE(reference->Bind(cache, state_parts));
        ASSERT_EQ(cudaMemset(activations[static_cast<std::size_t>(only_arm)].pointer, 0xff,
                             kActivationBytes),
                  cudaSuccess);
        ASSERT_EQ(
            cudaMemset(state_buffers[static_cast<std::size_t>(only_arm)].pointer, 0, root_bytes[3]),
            cudaSuccess);
      }
      auto frame = initial_shape;
      frame.segments.front().past = past;
      if (past != 0) {
        for (const auto& program : programs) ASSERT_TRUE(program->TryUpdateGeneratedInputs(frame));
      }
      std::array<std::byte, sizeof(host_positions)> position_bytes{};
      co::InputValueSink sink(position_bytes, sizeof(std::int32_t));
      ASSERT_TRUE(co::EmitInputValues(step.inputs[1], frame, {}, sink));
      std::memcpy(host_positions.data(), position_bytes.data(), position_bytes.size());
      for (std::size_t arm = 0; arm < kArms; ++arm) {
        ASSERT_EQ(
            cudaMemcpy(reinterpret_cast<void*>(input_addresses[arm][0]), host_features.data(),
                       host_features.size() * sizeof(host_features[0]), cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(input_addresses[arm][1]),
                             host_positions.data(), sizeof(host_positions), cudaMemcpyHostToDevice),
                  cudaSuccess);
      }
      const std::array<reference::PartView, 1> positions_part{
          {{.role = llmp::format::PartRole::kData,
            .bytes = std::as_bytes(std::span(host_positions))}}};
      ASSERT_TRUE(reference->BindValues(features, host_features));
      ASSERT_TRUE(reference->Bind(positions, positions_part));
      std::array<std::int64_t, kRows> ids{};
      for (std::uint32_t i = 0; i < kRows; ++i) ids[i] = past + i;
      ASSERT_TRUE(reference->BindInts(cells, ids));
      ASSERT_TRUE(reference->Run());
      const auto expected = reference->Bytes(cache);
      ASSERT_EQ(expected.size(), *state_storage_bytes);
      if (!ordinary_checked) {
        ASSERT_TRUE(programs[1]->Run(**launch, *stream));
        ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
        ASSERT_TRUE((*launch)->DrainGeneratedInputs());
        std::vector<std::byte> actual(root_bytes[3]);
        ASSERT_EQ(cudaMemcpy(actual.data(), state_buffers[1].pointer, root_bytes[3],
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_TRUE(std::ranges::equal(expected, std::span(actual).first(expected.size())));
        ASSERT_EQ(cudaMemset(state_buffers[1].pointer, 0, root_bytes[3]), cudaSuccess);
        ordinary_checked = true;
      }
      for (std::size_t arm = 0; arm < kArms; ++arm) {
        if (only_arm >= 0 && arm != static_cast<std::size_t>(only_arm)) continue;
        ASSERT_TRUE((*launch)->Launch(*stream, *recordings[arm]));
        ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
        ASSERT_TRUE((*launch)->DrainGeneratedInputs());
        std::vector<std::byte> actual(root_bytes[3]);
        ASSERT_EQ(cudaMemcpy(actual.data(), state_buffers[arm].pointer, root_bytes[3],
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_TRUE(std::ranges::equal(expected, std::span(actual).first(expected.size())))
            << "arm=" << arm << " past=" << past;
        EXPECT_TRUE(std::ranges::all_of(std::span(actual).subspan(expected.size()),
                                        [](std::byte byte) { return byte == std::byte{0}; }))
            << "arm=" << arm << " past=" << past << " readable tail";
      }
    };
    run_frame(0);
    auto invalid_shape = initial_shape;
    invalid_shape.segments.front().past = kCapacity - kRows + 1;
    for (const auto& program : programs) {
      auto rejected = program->TryUpdateGeneratedInputs(invalid_shape);
      ASSERT_FALSE(rejected) << "an out-of-range RopeStore update must be rejected";
    }
    run_frame(0);  // The rejected update preserves the accepted indices and positions.
    run_frame(5);
    MeasureWholeStatePair(
        encoding == llmp::format::Encoding::kQ8_0 ? "rope-store-q8-r2" : "rope-store-q4-r2",
        adopted->registry->ranks().recipe_comparisons.front().key, recordings,
        [&](std::size_t arm) { run_frame(5, static_cast<int>(arm)); });
    ASSERT_FALSE(HasFailure());
    recordings = {};
    programs = {};
    launch->reset();
    life_->backend->LaunchReleased();
    if (encoding == llmp::format::Encoding::kQ8_0) {
      kg::GgmlBackendOptions backend_options{
          .lower = {.expert_stride = [](const ir::WeightRef& ref) {
            return ref.expert_array ? std::uint64_t{2048} : std::uint64_t{0};
          }}};
      auto next_backend = kg::GgmlBackend::Create(0, *life_->execution, life_->node.stream(0),
                                                  nullptr, std::move(backend_options));
      ASSERT_TRUE(next_backend) << (next_backend ? "" : next_backend.error());
      const auto charge = [paid = life_->paid](std::uint64_t bytes) { return paid->Charge(bytes); };
      const auto release = [paid = life_->paid](std::uint64_t bytes) { paid->Release(bytes); };
      ASSERT_TRUE((*next_backend)->FundConstruction(charge, release));
      ASSERT_TRUE((*next_backend)->Declare({}, life_->resources.library_bytes(), charge, release));
      ASSERT_TRUE((*next_backend)->LendLibrary(&life_->resources.library()));
      life_->backend = std::move(*next_backend);
    }
  }
}

TEST_F(GgmlComparisonLowerForGpu, RopeStoreViewAlternativeMatchesReferenceAcrossUpdatedPast) {
  constexpr std::uint32_t kWidth = 64;
  constexpr std::uint32_t kCapacity = 16;
  constexpr std::uint32_t kRows = 3;
  constexpr std::size_t kActivationBytes = 1U << 20U;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  constexpr std::size_t kArms = 2;
  constexpr std::size_t kStateBytes = kWidth * 32 * sizeof(__half);
  ir::Builder builder;
  builder.SetOwner(0);
  const auto root =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, 1, 5})));
  const auto feature_base = builder.View(root, ir::ShapeOf({kWidth, 1, 4}), builder.tensor(root).nb,
                                         kWidth * sizeof(float));
  const auto input_view = builder.View(feature_base, ir::ShapeOf({kWidth, 1, kRows}),
                                       builder.tensor(feature_base).nb, kWidth * sizeof(float));
  const auto positions =
      builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({kRows})));
  const auto cells = builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({kRows})));
  const ir::StateRef state_ref{.stage = 0, .tensor = 0, .layer = 0, .segment = 0};
  const auto cache =
      builder.State(ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({kWidth, 32})), state_ref);
  const auto target = builder.View(cache, ir::ShapeOf({kWidth, kCapacity}),
                                   builder.tensor(cache).nb, kWidth * sizeof(__half));
  const ir::RopeParams rope_params{.mode = ir::RopeMode::kNeox,
                                   .dims = kWidth,
                                   .base = 10000.0F,
                                   .scale = 1.0F,
                                   .beta_fast = 4.0F,
                                   .beta_slow = 1.0F};
  const auto rope = builder.Rope(input_view, positions, {}, rope_params);
  const auto flat = builder.Reshape(rope, ir::ShapeOf({kWidth, kRows}));
  const auto rows = builder.View(flat, ir::ShapeOf({kWidth, kRows}), builder.tensor(flat).nb, 0);
  const auto stored = builder.CacheStore(target, rows, cells);
  builder.Keep(root);
  builder.Output(stored);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);

  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = root, .kind = co::InputKind::kFeatures},
                                {.tensor = positions, .kind = co::InputKind::kPositions},
                                {.tensor = cells,
                                 .kind = co::InputKind::kStoreCells,
                                 .segment = 0,
                                 .cells = {.capacity = kCapacity}}},
                     .weights = {},
                     .state = {{.tensor = cache, .ref = state_ref, .read = true, .written = true}}};
  const co::GraphBuildIO io = co::GraphIO(step);
  const llmp::planner::StepShape initial_shape{
      .stage = 0, .segments = {{.slot = 0, .rows = kRows, .past = 0, .outputs = 0}}};
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* rope_store_pattern = planner::FindPattern("rope_store");
  ASSERT_NE(rope_store_pattern, nullptr);
  const planner::ComparisonSiteSelector selector{.source_graph_identity = step.graph.identity(),
                                                 .kind = planner::ComparisonSiteKind::kRewrite,
                                                 .pattern = "rope_store",
                                                 .version = rope_store_pattern->version,
                                                 .member_origins = {{step.graph.producer(rope)},
                                                                    {step.graph.producer(flat)},
                                                                    {step.graph.producer(rows)},
                                                                    {step.graph.producer(stored)}},
                                                 .member_owner_segments = {},
                                                 .boundary_ports = {}};

  const std::array roots{root, positions, cells, cache};
  std::array<std::uint64_t, roots.size()> root_bytes{};
  std::array<co::ComparisonExternalBinding, roots.size()> external{};
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const auto bytes = ir::StorageBytes(step.graph.tensor(roots[i]));
    ASSERT_TRUE(bytes);
    root_bytes[i] = *bytes;
    external[i] = {.original_root = roots[i],
                   .allocation_class = static_cast<std::uint32_t>(i + 1),
                   .allocation_bytes = *bytes,
                   .bytes = *bytes,
                   .alignment = 256};
  }
  const co::ComparisonBoundaryView boundary_view{io, step.state, 0, external};
  auto inventory = planner::BoundComparison(
      boundary_view, selector, std::make_shared<PaidBoundaryFunder>(life_->paid), *recipe.registry,
      recipe.device, {}, {.capturable = true, .within = recipe.within, .shape = &initial_shape});
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  ASSERT_EQ(inventory->incumbent.plan.applied.size(), 1U);
  EXPECT_EQ(inventory->incumbent.plan.applied.front().pattern, "rope_store");
  ASSERT_EQ(inventory->incumbent.plan.steps.size(), 1U);
  EXPECT_EQ(recipe.registry->at(inventory->incumbent.plan.steps.front().implementation).name,
            kg::kRopeSetRowsFused);
  auto alternative = planner::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto planned = planner::PlanAlternative(
      *inventory, std::move(*alternative), *recipe.registry, recipe.device, {},
      {.capturable = true, .within = recipe.within, .shape = &initial_shape});
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  EXPECT_TRUE(std::ranges::none_of(planned->plan.applied,
                                   [](const auto& item) { return item.pattern == "rope_store"; }));
  EXPECT_TRUE(std::ranges::any_of(planned->plan.steps, [&](const auto& item) {
    return recipe.registry->at(item.implementation).name == kg::kRopeName;
  }));
  EXPECT_TRUE(std::ranges::any_of(planned->plan.steps, [&](const auto& item) {
    return recipe.registry->at(item.implementation).name == kg::kSetRowsName;
  }));

  std::array<CudaBuffer, kArms> activations{}, states{};
  CudaBuffer pool, persistent;
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  ASSERT_TRUE(persistent.Allocate(life_->backend->persistent_bytes()));
  auto launch = life_->backend->Launch(
      {.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
      {.base = persistent.address(), .bytes = life_->backend->persistent_bytes()});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  for (std::size_t i = 0; i < kArms; ++i) {
    ASSERT_TRUE(activations[i].Allocate(kActivationBytes));
    ASSERT_TRUE(states[i].Allocate(kStateBytes));
    ASSERT_EQ(cudaMemset(states[i].pointer, 0, kStateBytes), cudaSuccess);
  }
  std::array<const planner::StepPlan*, kArms> plans{&inventory->incumbent.plan, &planned->plan};
  std::array<std::unique_ptr<kn::StepProgram>, kArms> programs{};
  std::array<kn::Addresses, kArms> arm_addresses{};
  std::array<kn::ComparisonRecipeSeal, kArms> descriptors{};
  std::array<std::array<std::uint64_t, 3>, kArms> input_addresses{};
  const auto input_address = [&](const planner::StepPlan& plan, std::size_t arm,
                                 ir::TensorId original) -> std::optional<std::uint64_t> {
    if (original.v >= plan.tensors.size()) return {};
    const ir::TensorId mapped = plan.tensors[original.v];
    if (!mapped.valid()) return {};
    const ir::TensorId mapped_root = plan.graph.root(mapped);
    const auto placed = std::ranges::find_if(plan.placement.offsets, [&](const auto& entry) {
      return plan.graph.root(entry.first) == mapped_root;
    });
    if (placed == plan.placement.offsets.end() || placed->second >= kActivationBytes) return {};
    return activations[arm].address() + placed->second;
  };
  for (std::size_t arm = 0; arm < kArms; ++arm) {
    ASSERT_LE(plans[arm]->placement.extent, kActivationBytes);
    const std::array this_arm_inputs{root, positions, cells};
    std::array<kn::ComparisonBindingRange, 4> ranges{};
    kn::Addresses addresses{.weight = {},
                            .weight_rows = {},
                            .state =
                                [&, arm](const ir::StateRef& ref) {
                                  return ref == state_ref ? states[arm].address()
                                                          : std::uint64_t{0};
                                },
                            .save = {},
                            .save_rows = 0,
                            .activations = activations[arm].address(),
                            .activation_bytes = kActivationBytes,
                            .activation_alignment = 256};
    addresses.state_range = [&, arm](const ir::StateRef& ref) -> std::optional<kn::BoundedAddress> {
      if (ref != state_ref) return {};
      return kn::BoundedAddress{
          .base = states[arm].address(), .bytes = root_bytes[3], .alignment = 256};
    };
    arm_addresses[arm] = addresses;
    for (std::size_t i = 0; i < this_arm_inputs.size(); ++i) {
      const auto address = input_address(*plans[arm], arm, this_arm_inputs[i]);
      ASSERT_TRUE(address);
      input_addresses[arm][i] = *address;
      ranges[i] = {.original_root = this_arm_inputs[i],
                   .allocation_class = static_cast<std::uint32_t>(i + 1),
                   .allocation_bytes = root_bytes[i],
                   .bytes = root_bytes[i],
                   .alignment = 256,
                   .base = *address};
    }
    ranges[3] = {.original_root = cache,
                 .allocation_class = 4,
                 .allocation_bytes = root_bytes[3],
                 .bytes = root_bytes[3],
                 .alignment = 256,
                 .base = states[arm].address()};
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .scope = kn::ComparisonProofScope::kWholeProgramNative,
                                  .bindings = ranges,
                                  .state = step.state,
                                  .active_stage = 0,
                                  .expected_boundary = &inventory->source_boundary};
    kn::LowerInput lower_input(*plans[arm], step, initial_shape, addresses, 0, {}, &request);
    auto program = life_->backend->Lower(lower_input);
    ASSERT_TRUE(program) << (program ? "" : program.error());
    programs[arm] = std::move(*program);
    auto seal = programs[arm]->comparison_recipe();
    ASSERT_TRUE(seal);
    ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified)
        << "missing=" << static_cast<int>(seal->missing);
    ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);
    descriptors[arm] = *seal;
  }
  auto adopted = AdoptWholeAlternative(
      step, initial_shape, *inventory, boundary_view, backend, arm_addresses[1],
      planner::PlanOptions{.capturable = true, .within = recipe.within, .shape = &initial_shape},
      descriptors[0], descriptors[1], std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(adopted) << (adopted ? "" : adopted.error());
  programs[1] = std::move(adopted->program);
  for (std::size_t arm = 0; arm < kArms; ++arm) {
    ASSERT_TRUE(programs[arm]->Bind(**launch));
    EXPECT_TRUE(en::SameWholeProgramRecipe(descriptors[arm], *programs[arm]->comparison_recipe()));
  }
  auto first_seal = programs[0]->comparison_recipe();
  auto second_seal = programs[1]->comparison_recipe();
  ASSERT_TRUE(first_seal && second_seal);
  ASSERT_EQ(first_seal->state, kn::ComparisonProofState::kBoundVerified);
  ASSERT_EQ(second_seal->state, kn::ComparisonProofState::kBoundVerified);
  EXPECT_EQ(first_seal->subject, second_seal->subject);
  EXPECT_EQ(first_seal->boundary, second_seal->boundary);
  EXPECT_NE(first_seal->physical_recipe, second_seal->physical_recipe);

  std::array<kn::ComparisonBindingRange, 4> valid_ranges{};
  const std::array bound_roots{root, positions, cells};
  for (std::size_t i = 0; i < bound_roots.size(); ++i) {
    valid_ranges[i] = {.original_root = bound_roots[i],
                       .allocation_class = static_cast<std::uint32_t>(i + 1),
                       .allocation_bytes = root_bytes[i],
                       .bytes = root_bytes[i],
                       .alignment = 256,
                       .base = input_addresses[0][i]};
  }
  valid_ranges[3] = {.original_root = cache,
                     .allocation_class = 4,
                     .allocation_bytes = root_bytes[3],
                     .bytes = root_bytes[3],
                     .alignment = 256,
                     .base = states[0].address()};
  kn::Addresses control_addresses{
      .weight = {},
      .weight_rows = {},
      .state = [&](const ir::StateRef& ref) { return ref == state_ref ? states[0].address() : 0; },
      .save = {},
      .save_rows = 0,
      .activations = activations[0].address(),
      .activation_bytes = kActivationBytes,
      .activation_alignment = 256};

  const auto original_cells_kind = step.inputs[2].kind;
  step.inputs[2].kind = co::InputKind::kFeatures;
  kn::ComparisonRequest unowned_request{.mode = kn::ComparisonRunMode::kFull,
                                        .bindings = valid_ranges,
                                        .state = step.state,
                                        .active_stage = 0,
                                        .expected_boundary = &inventory->source_boundary};
  kn::LowerInput unowned_input(*plans[0], step, initial_shape, control_addresses, 0, {},
                               &unowned_request);
  auto unowned_program = life_->backend->Lower(unowned_input);
  step.inputs[2].kind = original_cells_kind;
  ASSERT_TRUE(unowned_program) << (unowned_program ? "" : unowned_program.error());
  const auto unowned_seal = (*unowned_program)->comparison_recipe();
  ASSERT_TRUE(unowned_seal);
  EXPECT_EQ(unowned_seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(unowned_seal->missing, kn::ComparisonMissing::kPhysicalClosure);

  auto alias_ranges = valid_ranges;
  alias_ranges[3].base = alias_ranges[1].base;
  kn::ComparisonRequest alias_request{.mode = kn::ComparisonRunMode::kFull,
                                      .bindings = alias_ranges,
                                      .state = step.state,
                                      .active_stage = 0,
                                      .expected_boundary = &inventory->source_boundary};
  kn::LowerInput alias_input(*plans[0], step, initial_shape, control_addresses, 0, {},
                             &alias_request);
  auto alias_program = life_->backend->Lower(alias_input);
  ASSERT_FALSE(alias_program);
  EXPECT_NE(alias_program.error().find("overlapping ranges disagree on backing alias class"),
            std::string::npos);

  auto shifted_ranges = valid_ranges;
  shifted_ranges[3].offset = 256;
  shifted_ranges[3].bytes -= 256;
  kn::ComparisonRequest shifted_request{.mode = kn::ComparisonRunMode::kFull,
                                        .bindings = shifted_ranges,
                                        .state = step.state,
                                        .active_stage = 0,
                                        .expected_boundary = &inventory->source_boundary};
  kn::LowerInput shifted_input(*plans[0], step, initial_shape, control_addresses, 0, {},
                               &shifted_request);
  auto shifted_program = life_->backend->Lower(shifted_input);
  ASSERT_FALSE(shifted_program);
  EXPECT_NE(shifted_program.error().find("smaller than its readable/written extent"),
            std::string::npos);

  auto changed_state = step.state;
  changed_state.front().read = false;
  const co::ComparisonBoundaryView changed_state_view{io, changed_state, 0, external};
  auto changed_state_boundary = co::MakeComparisonBoundary(
      changed_state_view, std::make_shared<PaidBoundaryFunder>(life_->paid));
  EXPECT_FALSE(changed_state_boundary);

  reference::Executor* reference_ptr = nullptr;
  auto reference = reference::Executor::Create(step.graph);
  ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
  reference_ptr = &*reference;
  std::vector<std::byte> initial_state(kStateBytes);
  const std::array<reference::PartView, 1> state_parts{
      reference::PartView{.role = llmp::format::PartRole::kData, .bytes = initial_state}};
  ASSERT_TRUE(reference_ptr->Bind(cache, state_parts));

  std::array<std::unique_ptr<kn::Recorded>, kArms> recordings{};
  if (WholeTimingConfigured())
    ASSERT_TRUE((*launch)->SetCaptureMeasurement(kn::CaptureMeasurement::kGpuDuration));
  const auto stream = life_->node.execution().Submission(life_->node.stream(0));
  ASSERT_TRUE(stream) << (stream ? "" : stream.error().detail);
  std::array<float, kWidth * 5> host_input{};
  std::array<std::int32_t, kRows> host_positions{};
  const auto fill_inputs = [&](std::uint32_t past, float delta) -> bool {
    for (std::size_t i = 0; i < host_input.size(); ++i)
      host_input[i] = static_cast<float>(i + 1) * 0.03125F + delta;
    auto position_shape = initial_shape;
    position_shape.segments.front().past = past;
    std::array<std::byte, sizeof(host_positions)> position_bytes{};
    co::InputValueSink position_sink(position_bytes, sizeof(std::int32_t));
    if (auto emitted = co::EmitInputValues(step.inputs[1], position_shape, {}, position_sink);
        !emitted)
      return false;
    std::memcpy(host_positions.data(), position_bytes.data(), position_bytes.size());
    return true;
  };
  const auto copy_device_inputs = [&]() -> bool {
    for (std::size_t arm = 0; arm < kArms; ++arm) {
      if (cudaMemcpy(reinterpret_cast<void*>(input_addresses[arm][0]), host_input.data(),
                     sizeof(host_input), cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(reinterpret_cast<void*>(input_addresses[arm][1]), host_positions.data(),
                     sizeof(host_positions), cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    }
    return true;
  };
  const auto run_reference =
      [&](std::uint32_t past) -> std::expected<void, reference::ExecFailure> {
    std::array<std::int64_t, kRows> host_cells{};
    for (std::uint32_t row = 0; row < kRows; ++row) host_cells[row] = past + row;
    if (auto bound = reference_ptr->BindValues(root, host_input); !bound) return bound;
    const std::array<reference::PartView, 1> position_parts{reference::PartView{
        .role = llmp::format::PartRole::kData, .bytes = std::as_bytes(std::span(host_positions))}};
    if (auto bound = reference_ptr->Bind(positions, position_parts); !bound) return bound;
    if (auto bound = reference_ptr->BindInts(cells, host_cells); !bound) return bound;
    return reference_ptr->Run();
  };

  ASSERT_TRUE(fill_inputs(0, 0.0F));
  ASSERT_TRUE(copy_device_inputs());
  for (std::size_t arm = 0; arm < kArms; ++arm) {
    auto recorded =
        programs[arm]->Record(**launch, *stream, [&]() -> std::expected<void, kn::StepFailure> {
          return programs[arm]->Run(**launch, *stream);
        });
    ASSERT_TRUE(recorded) << (recorded ? "" : recorded.error().detail);
    recordings[arm] = std::move(*recorded);
  }
  if (WholeTimingConfigured())
    ASSERT_TRUE((*launch)->SetCaptureMeasurement(kn::CaptureMeasurement::kDisabled));
  ASSERT_TRUE(run_reference(0)) << "CPU reference could not execute the initial state write";
  const auto ordinary_expected = reference_ptr->Bytes(cache);
  ASSERT_TRUE(programs[1]->Run(**launch, *stream));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  std::array<__half, kWidth * 32> ordinary_state{};
  ASSERT_EQ(cudaMemcpy(ordinary_state.data(), states[1].pointer, sizeof(ordinary_state),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(ordinary_expected.size(), sizeof(ordinary_state));
  double ordinary_worst = 0.0;
  for (std::size_t i = 0; i < ordinary_state.size(); ++i) {
    __half expected_half{};
    std::memcpy(&expected_half, ordinary_expected.data() + i * sizeof(__half), sizeof(__half));
    const auto got = __half2float(ordinary_state[i]);
    const auto want = __half2float(expected_half);
    ASSERT_TRUE(std::isfinite(got));
    ordinary_worst = std::max(ordinary_worst,
                              std::abs(static_cast<double>(got) - want) / (1.0 + std::abs(want)));
  }
  EXPECT_LT(ordinary_worst, 1.0e-3);
  for (const auto& recording : recordings) ASSERT_TRUE((*launch)->Launch(*stream, *recording));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  auto expected = reference_ptr->Bytes(cache);
  ASSERT_EQ(expected.size(), kStateBytes);
  const auto compare_state = [&](std::size_t arm, std::span<const std::byte> expected_bytes,
                                 std::uint32_t past) {
    std::array<__half, kWidth * 32> actual{};
    EXPECT_EQ(expected_bytes.size(), sizeof(actual));
    EXPECT_EQ(
        cudaMemcpy(actual.data(), states[arm].pointer, sizeof(actual), cudaMemcpyDeviceToHost),
        cudaSuccess);
    double worst = 0.0;
    constexpr std::size_t kRowBytes = kWidth * sizeof(__half);
    const auto target_offset = step.graph.tensor(target).view_offset;
    EXPECT_EQ(target_offset % kRowBytes, 0U);
    const std::size_t first_written_row = target_offset / kRowBytes + past;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      __half expected_half{};
      if ((i + 1) * sizeof(expected_half) <= expected_bytes.size())
        std::memcpy(&expected_half, expected_bytes.data() + i * sizeof(expected_half),
                    sizeof(expected_half));
      const double want = __half2float(expected_half);
      const double got = __half2float(actual[i]);
      EXPECT_TRUE(std::isfinite(want));
      EXPECT_TRUE(std::isfinite(got));
      if (!std::isfinite(want) || !std::isfinite(got)) continue;
      const std::size_t row = i / kWidth;
      if (row < first_written_row || row >= first_written_row + kRows) {
        const auto* actual_row =
            reinterpret_cast<const std::byte*>(actual.data()) + row * kRowBytes;
        const auto* expected_row = expected_bytes.data() + row * kRowBytes;
        EXPECT_EQ(std::memcmp(actual_row, expected_row, kRowBytes), 0)
            << "untouched state row " << row;
      } else {
        worst = std::max(worst, std::abs(got - want) / (1.0 + std::abs(want)));
      }
    }
    // Match ggml_ops_test.cc's full F16 K-write normalization and 1e-3 bound.
    EXPECT_LT(worst, 1.0e-3);
    return actual;
  };
  const auto first_fused = compare_state(0, expected, 0);
  const auto first_primitive = compare_state(1, expected, 0);
  EXPECT_EQ(std::memcmp(first_fused.data(), first_primitive.data(), sizeof(first_fused)), 0);

  auto changed_shape = initial_shape;
  changed_shape.segments.front().past = 4;
  for (auto& program : programs) {
    auto update = program->TryUpdateGeneratedInputs(changed_shape);
    ASSERT_TRUE(update) << (update ? "" : update.error().detail);
  }
  ASSERT_TRUE(fill_inputs(4, 0.5F));
  ASSERT_TRUE(copy_device_inputs());
  ASSERT_TRUE(run_reference(4)) << "CPU reference could not execute the changed-input replay";
  expected = reference_ptr->Bytes(cache);
  for (const auto& recording : recordings) ASSERT_TRUE((*launch)->Launch(*stream, *recording));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  const auto second_fused = compare_state(0, expected, 4);
  const auto second_primitive = compare_state(1, expected, 4);
  EXPECT_EQ(std::memcmp(second_fused.data(), second_primitive.data(), sizeof(second_fused)), 0);

  auto invalid_shape = changed_shape;
  invalid_shape.segments.front().past = kCapacity - kRows + 1;
  for (auto& program : programs) {
    auto update = program->TryUpdateGeneratedInputs(invalid_shape);
    ASSERT_FALSE(update);
  }
  for (std::uint32_t row = 0; row < kRows; ++row)
    EXPECT_EQ(host_positions[row],
              static_cast<std::int32_t>(changed_shape.segments.front().past + row));
  ASSERT_TRUE(run_reference(4)) << "CPU reference could not execute the retained accepted frame";
  expected = reference_ptr->Bytes(cache);
  for (const auto& recording : recordings) ASSERT_TRUE((*launch)->Launch(*stream, *recording));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
  ASSERT_TRUE((*launch)->DrainGeneratedInputs());
  const auto retained_fused = compare_state(0, expected, 4);
  const auto retained_primitive = compare_state(1, expected, 4);
  EXPECT_EQ(std::memcmp(retained_fused.data(), retained_primitive.data(), sizeof(retained_fused)),
            0);

  MeasureWholeStatePair(
      "rope-store-f16-view-r3", adopted->registry->ranks().recipe_comparisons.front().key,
      recordings, [&](std::size_t arm) {
        ASSERT_EQ(cudaMemset(activations[arm].pointer, 0xff, kActivationBytes), cudaSuccess);
        ASSERT_EQ(cudaMemset(states[arm].pointer, 0, kStateBytes), cudaSuccess);
        ASSERT_TRUE(reference_ptr->Bind(cache, state_parts));
        ASSERT_TRUE(fill_inputs(4, 0.5F));
        ASSERT_TRUE(copy_device_inputs());
        ASSERT_TRUE(run_reference(4));
        ASSERT_TRUE((*launch)->Launch(*stream, *recordings[arm]));
        ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream->handle)), cudaSuccess);
        ASSERT_TRUE((*launch)->DrainGeneratedInputs());
        (void)compare_state(arm, reference_ptr->Bytes(cache), 4);
      });
  ASSERT_FALSE(HasFailure());
  for (auto& recording : recordings) recording.reset();
  for (auto& program : programs) program.reset();
  launch->reset();
  life_->backend->LaunchReleased();
}

TEST_F(GgmlComparisonLowerForGpu,
       PrivateGeneratedStoreCaptureRetainsBackingAcrossProgramAndViewDrop) {
  constexpr std::uint32_t kWidth = 16;
  constexpr std::size_t kStores = 3;
  constexpr std::uint32_t kRows = 3;
  constexpr std::uint32_t kCapacity = 8;
  constexpr std::uint64_t kActivationBytes = 1U << 20U;
  constexpr std::uint64_t kPrivatePoolBytes = 1U << 20U;

  const std::array<std::pair<std::uint32_t, md::StateContract>, 1> contracts{
      {{0, {.tensors = {{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 16}}}}}};
  auto layout = md::Layout(contracts, kCapacity, kRows);
  ASSERT_TRUE(layout) << (layout ? "" : layout.error());
  const planner::StepShape parent_setup_shape{
      .stage = 0,
      .phase = planner::Phase::kDecode,
      .segments = {{.slot = 0, .rows = 1, .past = 0, .read_cells = 1, .outputs = 1}}};
  ts::FakeSourceOptions fake_options;
  fake_options.layouts = {*layout};
  auto fake_source = std::make_unique<ts::FakeStepSource>(std::move(fake_options));
  auto* fake_source_observer = fake_source.get();
  en::RunnerConfig config;
  config.slots.model = "generated-index-parent";
  config.slots.slots = 1;
  config.slots.stages = {{.name = "target", .layout = *layout}};
  config.slots.read_align = 256;
  config.slots.spill_directory = std::filesystem::temp_directory_path();
  const auto selected_bytes = life_->backend->persistent_bytes();
  if (selected_bytes)
    config.kernel_memory.push_back(
        {.name = "comparison selected kernel state", .bytes = selected_bytes});
  config.prepare = [this, selected_bytes](
                       std::span<const std::uint64_t> bases) -> std::expected<void, std::string> {
    if (bases.size() != static_cast<std::size_t>(selected_bytes != 0))
      return std::unexpected("selected kernel state was not mapped");
    life_->selected_state = {selected_bytes ? bases.front() : 0, selected_bytes};
    return {};
  };
  config.retire = [this] {
    life_->selected_state = {};
    auto released = life_->ReleaseNow();
    if (!released) life_->release_error = released.error();
  };
  config.setup_shapes = {parent_setup_shape};
  config.launch =
      [this](
          std::uint64_t pool_bytes) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
    auto launch = life_->backend->Launch(
        {.base = life_->node.pool().base, .size = llmp::base::Bytes(pool_bytes)},
        life_->selected_state);
    if (launch) life_->defer_release = true;
    return launch;
  };
  auto created =
      en::ModelRunner::Create(life_->node, std::move(config), std::move(fake_source), 0, 0);
  ASSERT_TRUE(created) << (created ? "" : created.error());
  auto adopted = en::RunnerOwner::Adopt(life_->node, *created);
  ASSERT_TRUE(adopted) << (adopted ? "" : adopted.error());
  auto runner = std::move(*adopted);
  runner_active_ = true;
  auto setup = runner->Setup();
  ASSERT_TRUE(setup) << (setup ? "" : setup.error());
  ASSERT_TRUE(life_->node.MapWorkspace(runner->activations_needed(),
                                       std::max(runner->pool_needed(), en::kPagedExtent)));
  const auto fixed = life_->node.catalog().OccupancyOf(life_->node.domain()).Total().value();
  life_->node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() +
                           (16ULL << 20U));
  ASSERT_TRUE(life_->node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
  ASSERT_TRUE(runner->Register());
  life_->node.Run();
  auto bound = runner->Bind();
  ASSERT_TRUE(bound) << (bound ? "" : bound.error());
  auto parent = runner->KeepResources();
  ASSERT_TRUE(parent) << (parent ? "" : parent.error().what);
  kn::StepLaunch* const primary = parent->launch();
  ASSERT_NE(primary, nullptr);

  ir::Builder builder;
  builder.SetOwner(0);
  const auto rows =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({kWidth, kStores})));
  const auto cells = builder.Input(ir::Dense(llmp::format::Encoding::kI64, ir::ShapeOf({kStores})));
  std::array<ir::TensorId, kStores> states{};
  for (std::uint16_t i = 0; i < kStores; ++i) {
    const ir::StateRef ref{.stage = 0, .tensor = i, .layer = 0, .segment = 0};
    states[i] = builder.State(
        ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({kWidth, kCapacity})), ref);
    (void)builder.CacheStore(states[i], rows, cells);
  }
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = rows, .kind = co::InputKind::kFeatures},
                                {.tensor = cells,
                                 .kind = co::InputKind::kStoreCells,
                                 .segment = 0,
                                 .cells = {.capacity = kCapacity}}},
                     .weights = {},
                     .state = {}};
  for (std::uint16_t i = 0; i < kStores; ++i) {
    const ir::StateRef ref{.stage = 0, .tensor = i, .layer = 0, .segment = 0};
    step.state.push_back({.tensor = states[i], .ref = ref, .item = i, .written = true});
  }
  const co::GraphBuildIO io = co::GraphIO(step);
  const planner::StepShape shape{.stage = 0,
                                 .segments = {{.slot = 0, .rows = kRows, .past = 0, .outputs = 0}}};

  CudaBuffer activation, private_pool;
  std::array<CudaBuffer, kStores> state_buffers{};
  ASSERT_TRUE(activation.Allocate(kActivationBytes));
  ASSERT_TRUE(private_pool.Allocate(kPrivatePoolBytes));
  std::array<std::uint64_t, kStores> state_addresses{};
  for (std::size_t i = 0; i < kStores; ++i) {
    ASSERT_TRUE(state_buffers[i].Allocate(kWidth * kCapacity * sizeof(__half)));
    state_addresses[i] = state_buffers[i].address();
    ASSERT_EQ(cudaMemset(state_buffers[i].pointer, 0, kWidth * kCapacity * sizeof(__half)),
              cudaSuccess);
  }
  const auto private_pool_address = private_pool.address();
  auto backing = PaidCudaBacking::Make(life_->node, private_pool, life_->backing_alive);
  ASSERT_TRUE(backing) << (backing ? "" : backing.error());
  auto backend = life_->backend->Backend();
  ASSERT_TRUE(backend.owner);
  auto private_resources =
      backend.owner->Resources(backend.owner, {private_pool_address, kPrivatePoolBytes},
                               std::move(*parent), std::move(*backing));
  ASSERT_TRUE(private_resources) << (private_resources ? "" : private_resources.error().what);

  kn::Addresses addresses{.weight = {},
                          .weight_rows = {},
                          .state =
                              [state_addresses](const ir::StateRef& ref) {
                                return ref.tensor < state_addresses.size()
                                           ? state_addresses[ref.tensor]
                                           : std::uint64_t{0};
                              },
                          .save = {},
                          .activations = activation.address(),
                          .activation_bytes = kActivationBytes,
                          .activation_alignment = 256};
  const auto& recipe = backend.owner->backend();
  auto plan = en::PlanComponentGraph(io, recipe, {}, addresses, {.shape = &shape});
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  const auto address_of = [&](ir::TensorId original) -> std::optional<std::uint64_t> {
    if (!original.valid() || original.v >= plan->tensors.size()) return {};
    const auto mapped = plan->tensors[original.v];
    if (!mapped.valid()) return {};
    const auto root = plan->graph.root(mapped);
    const auto found = std::ranges::find_if(plan->placement.offsets, [&](const auto& entry) {
      return plan->graph.root(entry.first) == root;
    });
    if (found == plan->placement.offsets.end() || found->second >= kActivationBytes) return {};
    return activation.address() + found->second;
  };
  const auto rows_address = address_of(rows);
  const auto cells_address = address_of(cells);
  ASSERT_TRUE(rows_address);
  ASSERT_TRUE(cells_address);
  std::array<float, kWidth * kRows> host_rows{};
  for (std::size_t i = 0; i < host_rows.size(); ++i)
    host_rows[i] = static_cast<float>(i + 1) * 0.25F;
  ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(*rows_address), host_rows.data(), sizeof(host_rows),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  const auto rows_bytes = ir::StorageBytes(step.graph.tensor(rows));
  const auto cells_bytes = ir::StorageBytes(step.graph.tensor(cells));
  ASSERT_TRUE(rows_bytes && cells_bytes);
  std::vector<kn::ComparisonBindingRange> bindings{
      {.original_root = rows,
       .allocation_class = 1,
       .allocation_bytes = kActivationBytes,
       .offset = *rows_address - activation.address(),
       .bytes = *rows_bytes,
       .alignment = 16,
       .base = activation.address()},
      {.original_root = cells,
       .allocation_class = 1,
       .allocation_bytes = kActivationBytes,
       .offset = *cells_address - activation.address(),
       .bytes = *cells_bytes,
       .alignment = 16,
       .base = activation.address()},
  };
  std::vector<co::ComparisonExternalBinding> external{
      co::ComparisonExternalBinding{.original_root = rows,
                                    .allocation_class = 1,
                                    .allocation_bytes = kActivationBytes,
                                    .offset = *rows_address - activation.address(),
                                    .bytes = *rows_bytes,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = cells,
                                    .allocation_class = 1,
                                    .allocation_bytes = kActivationBytes,
                                    .offset = *cells_address - activation.address(),
                                    .bytes = *cells_bytes,
                                    .alignment = 16}};
  for (std::size_t i = 0; i < kStores; ++i) {
    const auto bytes = ir::StorageBytes(step.graph.tensor(states[i]));
    ASSERT_TRUE(bytes);
    bindings.push_back({.original_root = states[i],
                        .allocation_class = static_cast<std::uint32_t>(i + 2),
                        .allocation_bytes = *bytes,
                        .bytes = *bytes,
                        .alignment = 16,
                        .base = state_addresses[i]});
    external.push_back({.original_root = states[i],
                        .allocation_class = static_cast<std::uint32_t>(i + 2),
                        .allocation_bytes = *bytes,
                        .bytes = *bytes,
                        .alignment = 16});
  }
  co::ComparisonBoundaryView boundary_view{io, step.state, 0, external};
  auto boundary =
      co::MakeComparisonBoundary(boundary_view, std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                .bindings = bindings,
                                .state = step.state,
                                .active_stage = 0,
                                .expected_boundary = &*boundary};
  kn::LowerInput input(*plan, step, shape, addresses, 0, {}, &request);
  auto program = private_resources->Lower(input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  ASSERT_TRUE((*program)->Bind(*private_resources->launch()));
  const auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  ASSERT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified);
  ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);

  auto native = life_->node.execution().Submission(life_->node.stream(0));
  ASSERT_TRUE(native);
  auto recorded = primary->Capture(*native, [&]() -> std::expected<void, kn::StepFailure> {
    return (*program)->Run(*private_resources->launch(), *native);
  });
  ASSERT_TRUE(recorded) << (recorded ? "" : recorded.error().detail);
  ASSERT_TRUE(life_->backing_alive);
  ASSERT_TRUE(primary->Launch(*native, **recorded));
  life_->execution->FailNextRecord();
  auto unknown_drain = primary->DrainGeneratedInputs();
  ASSERT_FALSE(unknown_drain);
  EXPECT_EQ(unknown_drain.error().error, kn::StepError::kUnknown);
  EXPECT_TRUE(life_->backing_alive);
  EXPECT_FALSE(primary->Launch(*native, **recorded));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  // Retrying through the real provider after its unknown Record outcome
  // authenticates a tail fence and reopens the submission gate.
  ASSERT_TRUE(primary->DrainGeneratedInputs());
  std::array<__half, kWidth * kCapacity> expected{};
  for (std::size_t row = 0; row < kRows; ++row)
    for (std::size_t col = 0; col < kWidth; ++col)
      expected[row * kWidth + col] = __float2half(host_rows[row * kWidth + col]);
  std::array<std::array<__half, kWidth * kCapacity>, kStores> first{};
  for (std::size_t i = 0; i < kStores; ++i) {
    ASSERT_EQ(cudaMemcpy(first[i].data(), state_buffers[i].pointer, sizeof(first[i]),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(first[i].data(), expected.data(), sizeof(expected)), 0);
  }

  life_->execution->ReturnPendingNextQuery();
  ASSERT_TRUE(primary->Launch(*native, **recorded));
  auto pending_drain = primary->DrainGeneratedInputs();
  ASSERT_FALSE(pending_drain);
  EXPECT_EQ(pending_drain.error().error, kn::StepError::kRejected);
  EXPECT_TRUE(life_->backing_alive);
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_TRUE(primary->DrainGeneratedInputs());

  program->reset();
  *private_resources = {};
  EXPECT_TRUE(life_->backing_alive);
  for (auto& state : state_buffers)
    ASSERT_EQ(cudaMemset(state.pointer, 0, kWidth * kCapacity * sizeof(__half)), cudaSuccess);
  ASSERT_TRUE(primary->Launch(*native, **recorded));
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native->handle)), cudaSuccess);
  ASSERT_TRUE(primary->DrainGeneratedInputs());
  for (std::size_t i = 0; i < kStores; ++i) {
    std::array<__half, kWidth * kCapacity> replayed{};
    ASSERT_EQ(cudaMemcpy(replayed.data(), state_buffers[i].pointer, sizeof(replayed),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(first[i].data(), replayed.data(), sizeof(replayed)), 0);
  }
  EXPECT_TRUE(life_->backing_alive);  // The live recording still owns the private resource.
  recorded->reset();
  ASSERT_TRUE(primary->DrainGeneratedInputs());
  EXPECT_FALSE(life_->backing_alive);

  const std::array<en::PagedModel*, 1> models{life_.get()};
  auto retired = life_->node.TearDown(models);
  ASSERT_TRUE(retired) << (retired ? "" : retired.error());
  EXPECT_TRUE(life_->released);
  EXPECT_TRUE(life_->release_error.empty()) << life_->release_error;
  EXPECT_EQ(fake_source_observer->queued(), 0U);
  runner.reset();
  runner_active_ = false;
}

TEST_F(GgmlComparisonLowerForGpu, RejectsMeasuredExpertLeafBeyondItsDeclaredRange) {
  ir::Builder builder;
  builder.SetOwner(0);
  const ir::WeightRef ref{.expert_array = true};
  const auto weight =
      builder.Weight(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 16, 2})), ref);
  const auto input =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 1, 2})));
  builder.Output(builder.MatMul(weight, input));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  const std::array<co::StepWeight, 1> weights{co::StepWeight{
      .tensor = weight, .ref = ref, .parts = {{"weight", llmp::format::PartRole::kData}}}};
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), weights};
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  kn::Addresses addresses{};
  auto plan = en::PlanComponentGraph(io, recipe, {}, addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  const auto weight_bytes = llmp::ir::StorageBytes(graph->tensor(weight));
  ASSERT_TRUE(weight_bytes);
  ASSERT_EQ(*weight_bytes, 2048U);
  const std::array<kn::ComparisonBindingRange, 2> ranges{
      kn::ComparisonBindingRange{.original_root = input,
                                 .allocation_class = 1,
                                 .allocation_bytes = 4096,
                                 .bytes = 4096,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = weight,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 2,
                                 .allocation_bytes = *weight_bytes,
                                 .bytes = *weight_bytes,
                                 .alignment = 16}};
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull, .bindings = ranges};
  kn::LowerInput lower_input(*plan, io, addresses, std::nullopt, &request);
  auto lowered = life_->backend->Lower(lower_input);
  ASSERT_FALSE(lowered);
  EXPECT_NE(lowered.error().find("does not cover every actual lowered leaf extent"),
            std::string::npos);
}

TEST_F(GgmlComparisonLowerForGpu, RealBoundProgramPromotesAndLaunchReleaseDemotesProof) {
  constexpr std::size_t kArenaBytes = 4096;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  CudaBuffer activation, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));

  ir::Builder builder;
  builder.SetOwner(0);
  const auto source =
      builder.Input(llmp::ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 4})));
  builder.Output(builder.RmsNorm(source, 1.0e-6F));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), {}};
  kn::Addresses addresses{};
  addresses.activations = activation.address();
  addresses.activation_bytes = kArenaBytes;
  addresses.activation_alignment = 256;
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  auto plan = en::PlanComponentGraph(io, recipe, {}, addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());

  const std::array<kn::ComparisonBindingRange, 1> ranges{
      kn::ComparisonBindingRange{.original_root = source,
                                 .allocation_class = 1,
                                 .allocation_bytes = kArenaBytes,
                                 .bytes = kArenaBytes,
                                 .alignment = 256,
                                 .base = activation.address()}};
  const std::array<co::ComparisonExternalBinding, 1> externals{
      co::ComparisonExternalBinding{.original_root = source,
                                    .allocation_class = 1,
                                    .allocation_bytes = kArenaBytes,
                                    .bytes = kArenaBytes,
                                    .alignment = 256}};
  const co::ComparisonBoundaryView boundary_view{io, {}, 0, externals};
  auto boundary =
      co::MakeComparisonBoundary(boundary_view, std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = ranges, .expected_boundary = &*boundary};
  kn::LowerInput lower_input(*plan, io, addresses, std::nullopt, &request);
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error()) << " persistent=" << persistent.address()
                      << "/" << persistent_bytes << " pool=" << pool.address() << "/" << kPoolBytes;
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  ASSERT_TRUE((*program)->Bind(**launch));
  auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);

  life_->backend->LaunchReleased();
  seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
}

TEST_F(GgmlComparisonLowerForGpu, UnprovedBoundaryReasonSurvivesSuccessfulBind) {
  constexpr std::size_t kArenaBytes = 4096;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  CudaBuffer activation, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));

  auto fixture = MakeFixture();
  ASSERT_TRUE(fixture.plan);
  fixture.addresses.activations = activation.address();
  fixture.addresses.activation_bytes = kArenaBytes;
  fixture.addresses.activation_alignment = 256;
  const auto io = fixture.io();
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  auto plan = en::PlanComponentGraph(io, recipe, {}, fixture.addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  fixture.plan = std::move(*plan);

  const std::array<kn::ComparisonBindingRange, 1> ranges{
      kn::ComparisonBindingRange{.original_root = fixture.input,
                                 .allocation_class = 1,
                                 .allocation_bytes = kArenaBytes,
                                 .bytes = kArenaBytes,
                                 .alignment = 256,
                                 .base = activation.address()}};
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull, .bindings = ranges};
  kn::LowerInput lower_input(*fixture.plan, io, fixture.addresses, std::nullopt, &request);
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  ASSERT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  ASSERT_EQ(seal->missing, kn::ComparisonMissing::kBoundary);

  ASSERT_TRUE((*program)->Bind(**launch));
  seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kBoundary);
}

TEST_F(GgmlComparisonLowerForGpu, DescriptorBindingRefusalRecordsBindingsReason) {
  constexpr std::size_t kArenaBytes = 4096;
  constexpr std::size_t kPoolBytes = 1U << 20U;
  CudaBuffer activation, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  ASSERT_GT(persistent_bytes, 0U);
  ASSERT_TRUE(persistent.Allocate(persistent_bytes));

  auto fixture = MakeFixture();
  ASSERT_TRUE(fixture.plan);
  fixture.addresses.activations = activation.address();
  fixture.addresses.activation_bytes = kArenaBytes;
  fixture.addresses.activation_alignment = 256;
  const auto io = fixture.io();
  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  auto plan = en::PlanComponentGraph(io, recipe, {}, fixture.addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  fixture.plan = std::move(*plan);

  const std::array<kn::ComparisonBindingRange, 1> ranges{
      kn::ComparisonBindingRange{.original_root = fixture.input,
                                 .allocation_class = 1,
                                 .allocation_bytes = kArenaBytes,
                                 .bytes = kArenaBytes,
                                 .alignment = 256,
                                 .base = activation.address()}};
  const std::array<co::ComparisonExternalBinding, 1> externals{
      co::ComparisonExternalBinding{.original_root = fixture.input,
                                    .allocation_class = 1,
                                    .allocation_bytes = kArenaBytes,
                                    .bytes = kArenaBytes,
                                    .alignment = 256}};
  auto boundary = co::MakeComparisonBoundary({io, {}, 0, externals},
                                             std::make_shared<PaidBoundaryFunder>(life_->paid));
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  kn::ComparisonRequest request{
      .mode = kn::ComparisonRunMode::kFull, .bindings = ranges, .expected_boundary = &*boundary};
  kn::LowerInput lower_input(*fixture.plan, io, fixture.addresses, std::nullopt, &request);
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());
  auto program = life_->backend->Lower(lower_input);
  ASSERT_TRUE(program) << (program ? "" : program.error());
  auto seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  ASSERT_EQ(seal->state, kn::ComparisonProofState::kDescriptorVerified);
  ASSERT_EQ(seal->missing, kn::ComparisonMissing::kNone);

  ForeignStepLaunch foreign;
  const auto bound = (*program)->Bind(foreign);
  ASSERT_FALSE(bound);
  seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kBindings);

  ASSERT_TRUE((*program)->Bind(**launch));
  seal = (*program)->comparison_recipe();
  ASSERT_TRUE(seal);
  EXPECT_EQ(seal->state, kn::ComparisonProofState::kBoundVerified);
  EXPECT_EQ(seal->missing, kn::ComparisonMissing::kNone);
}

}  // namespace
