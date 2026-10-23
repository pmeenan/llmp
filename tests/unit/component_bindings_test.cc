// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_bindings.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact_writer.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "expected_error.h"
#include "ir/graph.h"
#include "runtime/component_admission.h"
#include "runtime/memory_guard.h"
#include "spec/stored.h"

#if defined(LLMP_BINDING_CUDA)
#include "kernels/ggml/executor.h"
#include "kernels/ggml/lower.h"
#endif

namespace {
namespace rt = llmp::runtime;
namespace co = llmp::components;
namespace sp = llmp::spec;
namespace fm = llmp::format;
namespace en = llmp::engine;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
using llmp::test::V1Plan;
using llmp::test::V1Resource;
using llmp::test_support::Failed;

fm::TensorFormat F(std::initializer_list<std::uint64_t> shape,
                   fm::Encoding encoding = fm::Encoding::kF32) {
  return llmp::ir::Dense(encoding, *fm::ShapeOf(std::span(shape.begin(), shape.size())));
}
sp::ComponentRef C(std::string kind, std::vector<sp::WeightBinding> weights,
                   std::vector<sp::Param> params = {}) {
  return {.kind = std::move(kind),
          .version = 1,
          .params = std::move(params),
          .weights = std::move(weights)};
}
sp::WeightBinding W(std::string role, std::string tensor) {
  return {.role = std::move(role), .tensor = std::move(tensor)};
}
sp::ModelSpec Tiny() {
  sp::Stage stage{.name = "target", .role = sp::StageRole::kTarget, .artifact = 0, .context = 256};
  stage.embedding = C("embedding.token", {W("table", "table")}, {{"scale", 1.0}});
  stage.stream.width = 32;
  stage.layers = {{{C("norm.rms", {W("weight", "norm")}, {{"eps", 0.00001}})}}};
  stage.heads.output = C("head.linear", {W("weight", "head")}, {{"rows", std::int64_t{17}}});
  sp::ModelSpec model{.pipeline = sp::PipelineKind::kLlm, .stages = {std::move(stage)}};
  sp::Canonicalize(model);
  return model;
}
V1Plan Plan(sp::ModelSpec model = Tiny(), fm::Encoding encoding = fm::Encoding::kF32) {
  sp::Canonicalize(model);
  V1Plan plan;
  plan.spec = sp::ToJson(model);
  plan.groups = {
      {.kind = llmp::artifact::GroupKind::kGlobal,
       .resources = {{.name = "table", .aliases = {"head"}, .format = F({32, 32}, encoding)},
                     {.name = "norm", .format = F({32})}}}};
  return plan;
}

class ComponentBindingsTest : public testing::Test {
 protected:
  std::filesystem::path scratch;
  llmp::catalog::Catalog catalog;
  llmp::catalog::DomainId domain = catalog.AddDomain("metadata");
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance;
  void SetUp() override {
    const char* configured = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    scratch = (configured ? std::filesystem::path(configured)
                          : std::filesystem::path(testing::TempDir())) /
              "component-bindings" / testing::UnitTest::GetInstance()->current_test_info()->name();
    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
    std::filesystem::create_directories(scratch, ec);
    ASSERT_FALSE(ec);
  }
  auto Open(const V1Plan& plan) {
    return llmp::artifact::Artifact::Open(llmp::test::Publish(scratch, llmp::test::Render(plan)));
  }
  std::expected<std::unique_ptr<rt::StoredBindingPayment>, std::string> Pay(
      const llmp::artifact::Artifact& artifact, std::uint64_t shortfall = 0) {
    auto bound = rt::StoredBindingBytes(artifact);
    if (!bound) return std::unexpected(bound.error());
    const auto total = *bound + en::ExecutionMetadataAllowance::OverheadBytes() - shortfall;
    auto paid = en::ExecutionMetadataAllowance::Admit(catalog, domain, total, total + 1, 1);
    if (!paid) return std::unexpected(paid.error());
    allowance = *paid;
    return rt::StoredBindingPayment::Create(artifact, allowance);
  }
};

TEST_F(ComponentBindingsTest, PaymentPrecedesParseAndMovesWithAllOwnedBindingData) {
  auto artifact = Open(Plan());
  ASSERT_TRUE(artifact) << Failed(artifact, &llmp::artifact::Error::ToString).value_or("");
  EXPECT_FALSE(rt::ReadStoredComponents(*artifact, {}));
  EXPECT_FALSE(Pay(*artifact, 1));
  EXPECT_EQ(allowance->used(), 0U);
  auto payment = Pay(*artifact);
  ASSERT_TRUE(payment) << Failed(payment).value_or("");
  const auto charged = allowance->used();
  EXPECT_GT(charged, 0U);
  EXPECT_FALSE(rt::StoredBindingPayment::Create(*artifact, allowance));
  {
    auto bound = rt::ReadStoredComponents(*artifact, std::move(*payment));
    ASSERT_TRUE(bound) << Failed(bound).value_or("");
    auto moved = std::move(*bound);
    EXPECT_EQ(allowance->used(), charged);
    EXPECT_EQ(moved.logit_width, 17U);
    EXPECT_EQ(moved.model, Tiny());
  }
  EXPECT_EQ(allowance->used(), 0U);
  EXPECT_GT(catalog.OccupancyOf(domain).Total().value(), charged);      // node-lifetime prepayment
  EXPECT_TRUE(rt::StoredBindingPayment::Create(*artifact, allowance));  // same allowance, retry
}

TEST_F(ComponentBindingsTest, StartupPaysSourceOnlyDraftersWithoutAdmittingSpecLessTargets) {
  auto plan = Plan();
  auto with_spec = Open(plan);
  ASSERT_TRUE(with_spec);
  const auto with_spec_bound = rt::StartupArtifactBytes(*with_spec);
  ASSERT_TRUE(with_spec_bound);
  plan.spec.reset();
  auto source_only = Open(plan);
  ASSERT_TRUE(source_only);
  EXPECT_FALSE(source_only->spec_json());
  ASSERT_TRUE(sp::SourceOf(*source_only));
  const auto source_bound = rt::StartupArtifactBytes(*source_only);
  ASSERT_TRUE(source_bound) << Failed(source_bound).value_or("");
  EXPECT_LT(*source_bound, *with_spec_bound);
  // Funding source metadata cannot turn a source-only artifact into a target.
  EXPECT_FALSE(rt::StoredBindingBytes(*source_only));
  EXPECT_FALSE(Pay(*source_only));
  EXPECT_FALSE(rt::ReadComponentAdmission(*source_only));
  // Alias records and copied names remain paid even without a spec document.
  plan.groups.front().resources.front().aliases.push_back(std::string(64, 'x'));
  auto larger = Open(plan);
  ASSERT_TRUE(larger) << Failed(larger, &llmp::artifact::Error::ToString).value_or("");
  const auto larger_bound = rt::StartupArtifactBytes(*larger);
  ASSERT_TRUE(larger_bound);
  EXPECT_GT(*larger_bound, *source_bound);
  constexpr std::uint64_t extent = llmp::artifact::kChunkBytes;
  const auto paid_bytes = (*source_bound + 4096 + extent - 1) / extent * extent;
  constexpr std::uint64_t available = 128ULL << 30U;
  EXPECT_FALSE(
      rt::StartupArtifactPayment::Admit(catalog, domain, *source_bound, paid_bytes - 1, available));
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), 0U);
  {
    auto payment =
        rt::StartupArtifactPayment::Admit(catalog, domain, *source_bound, paid_bytes, available);
    ASSERT_TRUE(payment) << Failed(payment).value_or("");
    EXPECT_EQ((*payment)->bytes(), paid_bytes);
    auto moved = std::move(*payment);
    EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), paid_bytes);
  }
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), 0U);
}

TEST_F(ComponentBindingsTest, NativeFormatsAliasesAndHeadRowsAdmitWithoutAnImportMapping) {
  auto artifact = Open(Plan(Tiny(), fm::Encoding::kBF16));
  ASSERT_TRUE(artifact) << Failed(artifact, &llmp::artifact::Error::ToString).value_or("");
  auto payment = Pay(*artifact);
  ASSERT_TRUE(payment);
  auto bound = rt::ReadStoredComponents(*artifact, std::move(*payment));
  ASSERT_TRUE(bound) << Failed(bound).value_or("");
  EXPECT_TRUE(rt::CheckComponentArtifactAdmission(*artifact));  // the stored spec is executable
  ASSERT_EQ(bound->places.size(), 3U);  // two role references share one physical resource
  std::size_t table_roles = 0;
  for (const auto& [key, place] : bound->places) {
    EXPECT_EQ(place.artifact, 0U);
    EXPECT_FALSE(place.array);
    const auto& resource = artifact->resources()[place.index];
    if (resource.name == "table") {
      ++table_roles;
      EXPECT_EQ(resource.format->encoding, fm::Encoding::kBF16);
    }
    EXPECT_EQ(key.second, fm::PartRole::kData);
  }
  EXPECT_EQ(table_roles, 2U);
  EXPECT_EQ(bound->logit_width, 17U);
  auto malformed = Tiny();
  malformed.stages[0].heads.output->params[0].value = std::int64_t{33};
  auto bad_artifact = Open(Plan(std::move(malformed)));
  ASSERT_TRUE(bad_artifact);
  EXPECT_FALSE(rt::CheckComponentArtifactAdmission(*bad_artifact));
  auto bad_payment = Pay(*bad_artifact);
  ASSERT_TRUE(bad_payment);
  auto bad = rt::ReadStoredComponents(*bad_artifact, std::move(*bad_payment));
  ASSERT_FALSE(bad);
  EXPECT_NE(bad.error().find("vocabulary"), std::string::npos);
  EXPECT_EQ(allowance->used(), 0U);
}

TEST_F(ComponentBindingsTest, StoredDrafterFoldCoverageIncludesOwnRolesAndRespectsBorrowedHolder) {
  auto target = Tiny();
  target.stages[0].folds = {"norm.plus_one"};
  target.stages[0].taps = {{.name = "features", .layer = 0}};
  auto target_plan = Plan(target);
  target_plan.source_folds = {{"norm.plus_one", {"norm"}}};
  auto target_artifact = llmp::artifact::Artifact::Open(
      llmp::test::Publish(scratch / "target", llmp::test::Render(target_plan)));
  ASSERT_TRUE(target_artifact);
  auto target_source = sp::SourceOf(*target_artifact);
  ASSERT_TRUE(target_source);

  auto composed = target;
  auto draft = target.stages[0];
  draft.name = "assistant";
  draft.role = sp::StageRole::kDrafter;
  draft.artifact = 1;
  draft.drafting = sp::MtpDraft{};
  draft.embedding->kind = "embedding.projected";
  draft.embedding->params = {
      {"scale", 1.0}, {"feature", std::int64_t{32}}, {"tap", std::string("features")}};
  draft.embedding->weights.push_back(W("projection", "draft.projection"));
  draft.reads = {{.stage = "target", .kind = sp::ReadKind::kTap, .name = "features"}};
  draft.embedding->weights[0].stage = "target";
  draft.heads.output->weights[0].stage = "target";
  draft.layers[0].components[0].weights[0].tensor = "draft.first";
  draft.layers[0].components.push_back(
      C("norm.rms", {W("weight", "draft.second")}, {{"eps", 0.00001}}));
  auto borrowed = C("norm.rms", {W("weight", "norm")}, {{"eps", 0.00001}});
  borrowed.weights[0].stage = "target";
  draft.layers[0].components.push_back(std::move(borrowed));
  composed.stages.push_back(std::move(draft));
  sp::Canonicalize(composed);
  ASSERT_TRUE(sp::Validate(composed));
  auto draft_plan = Plan(composed);
  draft_plan.architecture = "gemma4-assistant";
  draft_plan.groups = {{.kind = llmp::artifact::GroupKind::kGlobal,
                        .resources = {{.name = "draft.first", .format = F({32})},
                                      {.name = "draft.second", .format = F({32})},
                                      {.name = "draft.projection", .format = F({64, 32})}}}};
  draft_plan.source_folds = {{"norm.plus_one", {"draft.first", "draft.second"}}};
  const auto publish = [&](const V1Plan& plan, std::string_view name) {
    return llmp::artifact::Artifact::Open(
        llmp::test::Publish(scratch / name, llmp::test::Render(plan)));
  };
  auto complete = publish(draft_plan, "complete");
  ASSERT_TRUE(complete);
  auto admitted = rt::ReadComponentDrafterAdmission(*complete);
  ASSERT_TRUE(admitted) << Failed(admitted).value_or("");
  ASSERT_TRUE(admitted->model);
  EXPECT_TRUE(rt::CheckComposedAdmission(composed, *target_source, admitted->source));
  EXPECT_TRUE(rt::CheckComposedAdmission(composed, *target_artifact, *complete));

  // Both roles remain bound and stored; only the authenticated receipt omits
  // one. Catalog membership and listed-tensor membership alone cannot detect it.
  draft_plan.source_folds[0].second.pop_back();
  auto partial = publish(draft_plan, "partial");
  ASSERT_TRUE(partial);
  ASSERT_TRUE(sp::StoredSpec(*partial));
  const auto refused = rt::ReadComponentDrafterAdmission(*partial);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().contains("complete component tensor roles"));
  auto partial_source = sp::SourceOf(*partial);
  ASSERT_TRUE(partial_source);
  EXPECT_FALSE(rt::CheckComposedAdmission(composed, *target_source, *partial_source));
  EXPECT_FALSE(rt::CheckComposedAdmission(composed, *target_artifact, *partial));

  // The borrowed norm stays in the target's closure, not the drafter's.
  auto wrong_holder = admitted->source;
  wrong_holder.tensors.push_back({.name = "norm", .format = F({32})});
  wrong_holder.preprocessing[0].tensors.push_back("norm");
  EXPECT_FALSE(rt::CheckComposedAdmission(composed, *target_source, wrong_holder));
  auto absent_target_receipt = *target_source;
  absent_target_receipt.preprocessing.clear();
  EXPECT_FALSE(rt::CheckComposedAdmission(composed, absent_target_receipt, admitted->source));

  // A source-only legacy artifact cannot bypass the required stored contract.
  draft_plan.spec.reset();
  draft_plan.source_folds.clear();
  auto legacy = publish(draft_plan, "legacy");
  ASSERT_TRUE(legacy);
  auto legacy_admitted = rt::ReadComponentDrafterAdmission(*legacy);
  ASSERT_FALSE(legacy_admitted);
  EXPECT_TRUE(legacy_admitted.error().contains("carries no spec of its own"));
  const auto legacy_source = sp::SourceOf(*legacy);
  ASSERT_TRUE(legacy_source);
  EXPECT_FALSE(rt::CheckComposedAdmission(composed, *target_source, *legacy_source));
}

TEST_F(ComponentBindingsTest, VisitorUsesHoldingStageArtifactAndTheEmitOwners) {
  auto model = Tiny();
  auto next = model.stages[0];
  next.name = "drafter";
  next.role = sp::StageRole::kDrafter;
  next.artifact = 0;
  next.embedding->weights[0].stage = "target";
  model.stages[0].artifact = 1;  // stage number is deliberately not artifact number
  model.stages.push_back(std::move(next));
  sp::Canonicalize(model);
  const std::vector<co::BoundTensor> first{
      {"table", F({32, 32})}, {"head", F({32, 32})}, {"norm", F({32})}};
  const std::vector<co::BoundTensor> second{{"table", F({32, 32}, fm::Encoding::kBF16)},
                                            {"head", F({32, 32}, fm::Encoding::kBF16)},
                                            {"norm", F({32})}};
  const std::array<std::span<const co::BoundTensor>, 2> tables{first, second};
  std::optional<llmp::ir::WeightRef> owner;
  std::size_t borrowed = 0;
  const auto visited = co::VisitWeightBindings(
      model, tables, [&](const co::WeightBindingView& v) -> std::expected<void, std::string> {
        if (v.parts.front().first == "table") {
          EXPECT_EQ(v.use_component, 0U);
          EXPECT_EQ(v.use_role, 0U);
          EXPECT_EQ(v.artifact, 1U);
          EXPECT_EQ(v.ref.stage, 0U);
          EXPECT_EQ(v.format.encoding, fm::Encoding::kBF16);
          if (v.use_stage == 0)
            owner = v.ref;
          else {
            if (!owner) return std::unexpected("missing earlier owner");
            EXPECT_EQ(v.ref, *owner);
            ++borrowed;
          }
        }
        return {};
      });
  ASSERT_TRUE(visited) << Failed(visited).value_or("");
  EXPECT_EQ(borrowed, 1U);
  // Mutating only the borrowed artifact's format is rejected by the same
  // catalog role checker used in graph emission; no partial callback occurs.
  auto wrong = second;
  wrong[0].format = F({16, 32});
  const std::array<std::span<const co::BoundTensor>, 2> bad_tables{first, wrong};
  std::size_t visits = 0;
  EXPECT_FALSE(co::VisitWeightBindings(model, bad_tables,
                                       [&](const auto&) -> std::expected<void, std::string> {
                                         ++visits;
                                         return {};
                                       }));
  EXPECT_EQ(visits, 0U);
}

TEST_F(ComponentBindingsTest, RowTablePlacementRefusesSharedGroupsBeforeExclusion) {
  auto model = Tiny();
  auto& stage = model.stages[0];
  stage.stream.streams = 2;
  stage.stream.mixing = C("residual.hc", {}, {{"mix", std::string("sigmoid")}});
  stage.side = {{.name = "ngrams",
                 .source = C("side.ngram", {W("table", "side")},
                             {{"ngram", std::int64_t{2}},
                              {"heads_per_ngram", std::int64_t{1}},
                              {"row", std::int64_t{4}},
                              {"eos", std::int64_t{0}},
                              {"multipliers", std::vector<std::int64_t>{7, 11}},
                              {"offsets", std::vector<std::int64_t>{0}},
                              {"vocab", std::vector<std::int64_t>{32}}}),
                 .layers = {0}}};
  auto ple = C("residual.ple",
               {W("key", "key"), W("value", "value"), W("norm_key", "nk"), W("norm_query", "nq"),
                W("norm_conv", "nc"), W("conv", "conv")},
               {{"eps", 0.00001},
                {"input", std::string("ngrams")},
                {"conv", std::int64_t{2}},
                {"dilation", std::int64_t{1}}});
  stage.layers[0].components = {std::move(ple)};
  auto plan = Plan(model);
  auto& resources = plan.groups[0].resources;
  for (auto resource : std::vector<V1Resource>{{.name = "key", .format = F({4, 64})},
                                               {.name = "value", .format = F({4, 32})},
                                               {.name = "nk", .format = F({64})},
                                               {.name = "nq", .format = F({64})},
                                               {.name = "nc", .format = F({64})},
                                               {.name = "conv", .format = F({2, 64})}})
    resources.push_back(std::move(resource));
  plan.groups.push_back({.kind = llmp::artifact::GroupKind::kTable,
                         .resources = {{.name = "side", .format = F({4, 32}), .rows = true}}});
  auto artifact = Open(plan);
  ASSERT_TRUE(artifact);
  auto payment = Pay(*artifact);
  ASSERT_TRUE(payment);
  auto binding = rt::ReadStoredComponents(*artifact, std::move(*payment));
  ASSERT_TRUE(binding) << Failed(binding).value_or("");
  ASSERT_EQ(binding->rows.size(), 1U);
  EXPECT_EQ(binding->rows[0].rows_per_token, 1U);
  EXPECT_EQ(binding->rows[0].format, F({4, 32}));
  const auto serialized = sp::ToJson(binding->model);
  artifact = std::unexpected(llmp::artifact::Error{});  // owned names/lists survive the reader
  EXPECT_EQ(sp::ToJson(binding->model), serialized);
  EXPECT_EQ(binding->model.stages[0].side[0].source.Find("multipliers")->value,
            sp::ParamValue(std::vector<std::int64_t>{7, 11}));
  plan.groups.back().resources.push_back({.name = "unrelated", .format = F({4})});
  auto shared = Open(plan);
  ASSERT_TRUE(shared);
  auto shared_payment = Pay(*shared);
  ASSERT_TRUE(shared_payment);
  auto refused = rt::ReadStoredComponents(*shared, std::move(*shared_payment));
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().find("shares its group"), std::string::npos);
}

TEST_F(ComponentBindingsTest, PreservesOrdinaryAndFormatDeclaredMultipartExpertMaps) {
  auto model = Tiny();
  auto moe =
      C("ffn.moe", {W("gate", "gate"), W("up", "up"), W("down", "down"), W("router", "router")},
        {{"placement", std::string("none")},
         {"eps", 0.00001},
         {"activation", std::string("silu")},
         {"hidden", std::int64_t{32}},
         {"experts", std::int64_t{2}},
         {"used", std::int64_t{1}},
         {"expert_hidden", std::int64_t{32}},
         {"router", std::string("softmax_topk")},
         {"router_input", std::string("stream")}});
  for (const auto* name : {"gate_experts", "up_experts", "down_experts"})
    moe.weights.push_back({.role = name, .tensor = name, .experts = true});
  model.stages[0].layers[0].components = {moe};
  auto plan = Plan(model);
  plan.expert_count = 2;
  for (const auto* name : {"gate", "up", "down"})
    plan.groups[0].resources.push_back({.name = name, .format = F({32, 32})});
  plan.groups[0].resources.push_back({.name = "router", .format = F({32, 2})});
  plan.groups.push_back({.kind = llmp::artifact::GroupKind::kExpert,
                         .layer = 0,
                         .arrays = {{.name = "gate_experts", .format = F({32, 32})},
                                    {.name = "up_experts", .format = F({32, 32})},
                                    {.name = "down_experts", .format = F({32, 32})}}});
  auto artifact = Open(plan);
  ASSERT_TRUE(artifact);
  auto payment = Pay(*artifact);
  ASSERT_TRUE(payment);
  auto bindings = rt::ReadStoredComponents(*artifact, std::move(*payment));
  ASSERT_TRUE(bindings) << Failed(bindings).value_or("");
  std::size_t arrays = 0;
  for (const auto& [key, place] : bindings->places)
    if (place.array) {
      ++arrays;
      EXPECT_TRUE(key.first.expert_array);
      EXPECT_EQ(artifact->expert_arrays()[place.index].count, 2U);
    }
  EXPECT_EQ(arrays, 3U);
  std::erase_if(moe.weights, [](const auto& w) { return w.experts; });
  plan.groups.back().arrays.clear();
  const fm::TensorFormat split{.encoding = fm::Encoding::kMxfp8,
                               .layout = fm::Layout::kSplit,
                               .shape = *fm::ShapeOf(std::array<std::uint64_t, 2>{32, 32})};
  for (const auto* name : {"gate_experts", "up_experts", "down_experts"}) {
    for (const auto part : {fm::PartRole::kCodes, fm::PartRole::kBlockScales}) {
      const std::string tensor =
          std::string(name) + (part == fm::PartRole::kCodes ? ".codes" : ".scales");
      moe.weights.push_back({.role = name, .tensor = tensor, .part = part, .experts = true});
      plan.groups.back().arrays.push_back({.name = tensor, .format = split, .part = part});
    }
  }
  model.stages[0].layers[0].components.front() = std::move(moe);
  sp::Canonicalize(model);
  plan.spec = sp::ToJson(model);
  auto multipart = Open(plan);
  ASSERT_TRUE(multipart) << Failed(multipart, &llmp::artifact::Error::ToString).value_or("");
  auto multipart_payment = Pay(*multipart);
  ASSERT_TRUE(multipart_payment);
  auto multipart_bound = rt::ReadStoredComponents(*multipart, std::move(*multipart_payment));
  ASSERT_TRUE(multipart_bound) << Failed(multipart_bound).value_or("");
  EXPECT_EQ(std::ranges::count_if(multipart_bound->places,
                                  [](const auto& entry) { return entry.second.array; }),
            6);
  for (const auto& [key, place] : multipart_bound->places) {
    if (!place.array) continue;
    const auto& array = multipart->expert_arrays()[place.index];
    EXPECT_EQ(array.part, key.second);
    EXPECT_TRUE(fm::PartsOf(*array.format)->Find(key.second));
  }
}

#if defined(LLMP_BINDING_CUDA)
namespace kg = llmp::kernels::ggml;
// Host-only binding adapter. It cannot dispatch or claim a production backend
// declaration. All selection/lowering/implementation checks below are real.
class HostProgram final : public kn::StepProgram {
 public:
  HostProgram(kg::LoweredStep step, const llmp::execution::Registry& registry)
      : step_(std::move(step)), registry_(registry) {}
  std::uint64_t scratch_bytes() const override { return step_.extent; }
  std::uint64_t host_bytes() const override { return 0; }
  std::uint64_t launches() const override { return step_.plan.steps.size(); }
  std::span<const kn::InputSlot> inputs() const override { return step_.inputs; }
  std::span<const kn::OutputSlot> outputs() const override { return step_.outputs; }
  std::span<const kn::TensorUse> tensors() const override { return step_.tensors; }
  const llmp::execution::BoundPlan* bound() const override {
    return bound_ ? &bound_->bound() : nullptr;
  }
  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch&) override {
    auto result = kg::BoundGraph::Bind(registry_, step_.plan);
    if (!result)
      return std::unexpected(kn::StepFailure{kn::StepError::kRejected, result.error().detail});
    bound_.emplace(std::move(*result));
    return {};
  }
  std::expected<void, kn::StepFailure> Run(kn::StepLaunch&,
                                           llmp::providers::NativeStream) const override {
    return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "host-only binding control"});
  }

 private:
  kg::LoweredStep step_;
  const llmp::execution::Registry& registry_;
  std::optional<kg::BoundGraph> bound_;
};
struct NoLaunch final : kn::StepLaunch {
  std::expected<std::unique_ptr<kn::Recorded>, kn::StepFailure> Capture(
      llmp::providers::NativeStream, const Body&) override {
    return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "host-only"});
  }
  std::expected<void, kn::StepFailure> Launch(llmp::providers::NativeStream,
                                              const kn::Recorded&) override {
    return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "host-only"});
  }
};
// Deliberately test authority, not GgmlBackend::Declare: it owns the immutable
// host table below and pays the real source's seal. No device is prepared.
class HostBackendOwner final : public kn::BackendOwner {
 public:
  HostBackendOwner(kn::StepBackend table, std::shared_ptr<en::ExecutionMetadataAllowance> allowance)
      : table_(std::move(table)), allowance_(std::move(allowance)) {}
  const kn::StepBackend& backend() const override { return table_; }
  bool Charge(std::uint64_t bytes) const override { return allowance_->Charge(bytes); }
  void Release(std::uint64_t bytes) const override { allowance_->Release(bytes); }
  std::expected<llmp::execution::BackendDeclaration, std::string> Ready() const override {
    const auto recipe = llmp::base::Sha256{}.Update("host-binding-control").Finish();
    return llmp::execution::BackendDeclaration{recipe, recipe, recipe};
  }

 private:
  const kn::StepBackend table_;
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance_;
};
TEST_F(ComponentBindingsTest, ActualStoredSourcePlansLowersAndBindsWithoutDeviceExecution) {
  // This real cuBLAS descriptor needs a 16-byte-aligned output column stride.
  // The metadata-only controls above retain the independent 17-row head case.
  auto model = Tiny();
  model.stages[0].heads.output->params[0].value = std::int64_t{32};
  auto artifact = Open(Plan(std::move(model)));
  ASSERT_TRUE(artifact);
  auto payment = Pay(*artifact);
  ASSERT_TRUE(payment);
  auto bindings = rt::ReadStoredComponents(*artifact, std::move(*payment));
  ASSERT_TRUE(bindings) << Failed(bindings).value_or("");
  auto layout = co::StageLayout(bindings->model, 0, 256, 2);
  ASSERT_TRUE(layout) << Failed(layout).value_or("");
  auto registry = llmp::execution::Registry::Create(kg::Implementations());
  ASSERT_TRUE(registry);
  std::size_t lowers = 0;
  auto source_allowance =
      en::ExecutionMetadataAllowance::Admit(catalog, domain, 1U << 20, (1U << 20) + 1, 1);
  ASSERT_TRUE(source_allowance);
  kn::StepBackend table{
      .registry = &*registry,
      .device = {.device_class = "sm_121", .arch = 121},
      .written = kg::WrittenBytes,
      .within = kg::WithinRoot,
      .lower = [&](const kn::LowerInput& input)
          -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
        ++lowers;
        kg::DeviceChoices device;
        device.mul_mat =
            [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
          return kg::MulMatPath::kCublas;
        };
        auto lowered = kg::LowerStep(input, *registry, device);
        if (!lowered) return std::unexpected(lowered.error().detail);
        return std::unique_ptr<kn::StepProgram>(new HostProgram(std::move(*lowered), *registry));
      }};
  auto owner = std::make_shared<HostBackendOwner>(std::move(table), *source_allowance);
  auto numerical = en::NumericalDomain::Create({{.stage = 0,
                                                 .name = "target",
                                                 .uses = {en::NumericalUse::kScore},
                                                 .phases = {{.phase = pl::Phase::kScore,
                                                             .max_owners = 1,
                                                             .max_owner_rows = 2,
                                                             .max_total_rows = 2}}}});
  ASSERT_TRUE(numerical);
  auto source = en::ComponentStepSource::Create(
      {.spec = &bindings->model,
       .bindings = {.tensors = bindings->source.tensors, .state = &*layout},
       .slots = 1,
       .policy = pl::Policy{},
       .backend = {.owner = owner},
       .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*numerical))});
  ASSERT_TRUE(source) << Failed(source).value_or("");
  // Even an undeclared source's plan key must describe its future launch
  // resources; a currently unconfigured measurement context is not authority.
  const auto with_lanes = [&](kn::LaneConfiguration lanes) {
    auto backend = owner->backend();
    backend.lanes = lanes;
    return en::ComponentStepSource::Create(
        {.spec = &bindings->model,
         .bindings = {.tensors = bindings->source.tensors, .state = &*layout},
         .slots = 1,
         .policy = pl::Policy{},
         .backend = std::move(backend)});
  };
  auto serial = with_lanes({});
  auto four = with_lanes({.count = 4, .scratch_bytes_per_lane = 256});
  auto larger = with_lanes({.count = 4, .scratch_bytes_per_lane = 512});
  ASSERT_TRUE(serial);
  ASSERT_TRUE(four);
  ASSERT_TRUE(larger);
  EXPECT_NE((*serial)->KeyBase(), (*four)->KeyBase());
  EXPECT_NE((*four)->KeyBase(), (*larger)->KeyBase());
  EXPECT_FALSE(with_lanes({.count = 0, .scratch_bytes_per_lane = 256}));
  EXPECT_FALSE(with_lanes({.count = 4, .scratch_bytes_per_lane = 0}));
  EXPECT_FALSE(with_lanes({.count = 5, .scratch_bytes_per_lane = 256}));
  // Synthetic nonoverlapping host descriptor addresses, never device accesses.
  kn::Addresses addresses{};
  std::size_t lookups = 0;
  addresses.weight = [&](const llmp::ir::WeightRef& ref, fm::PartRole part) -> std::uint64_t {
    ++lookups;
    const auto at = bindings->places.find({ref, part});
    EXPECT_NE(at, bindings->places.end());
    return at == bindings->places.end() ? 0
                                        : (std::uint64_t{1} << 40) + (at->second.index + 1) * 65536;
  };
  addresses.activations = std::uint64_t{1} << 42;
  addresses.activation_bytes = 1U << 20;
  addresses.activation_alignment = 256;
  pl::StepShape shape{.phase = pl::Phase::kScore,
                      .segments = {{.slot = 0, .rows = 2, .read_cells = 256, .outputs = 2}},
                      .read_align = 256};
  EXPECT_FALSE((*source)->Build(shape, addresses));
  EXPECT_EQ(lowers, 0U);
  auto prepared = (*source)->PrepareExecution();
  ASSERT_TRUE(prepared) << Failed(prepared).value_or("");
  auto measuring = addresses;
  measuring.activations = 0;
  measuring.activation_bytes = 0;
  auto measured = (*source)->Build(shape, measuring);
  ASSERT_TRUE(measured) << Failed(measured).value_or("");
  EXPECT_GT((*measured)->scratch_bytes(), 0U);
  auto program = (*source)->Build(shape, addresses);
  ASSERT_TRUE(program) << Failed(program).value_or("");
  EXPECT_EQ(lowers, 2U);
  EXPECT_GT(lookups, 0U);
  ASSERT_EQ((*program)->outputs().size(), 1U);
  EXPECT_EQ((*program)->outputs()[0].bytes, 32U * 2U * sizeof(float));
  NoLaunch launch;
  auto bound = (*program)->Bind(launch);
  ASSERT_TRUE(bound) << Failed(bound, &kn::StepFailure::detail).value_or("");
  ASSERT_NE((*program)->bound(), nullptr);
  EXPECT_FALSE((*program)->Run(launch, {}));
  auto identity = (*source)->ExecutionIdentity();
  ASSERT_TRUE(identity);
  EXPECT_TRUE(identity->has_value());  // test-owned declaration only; no prepared CUDA backend
}
#endif
}  // namespace
