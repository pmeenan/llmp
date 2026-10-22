// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "execution/registry.h"
#include "ir/builder.h"
#include "planner/plan.h"

namespace llmp::planner {
namespace {

execution::Registry ReportRegistry() {
  return execution::Registry::Create(
             {execution::Implementation{
                 .name = "report.owned.fill",
                 .operation = ir::Primitive(ir::Op::kFill),
                 .source = "test",
                 .revision = "report-revision",
                 .build = "host",
                 .variant = "checked",
                 .capability = execution::Capability{.op = ir::Primitive(ir::Op::kFill)}}})
      .value();
}

StepPlan ReportFixture() {
  ir::Builder builder;
  builder.Output(builder.Fill(ir::ShapeOf({4}), 1234.625F));
  StepPlan plan{.graph = std::move(builder).Finish().value(),
                .steps = {},
                .regions = {},
                .placement = {},
                .applied = {},
                .from = {},
                .tensors = {},
                .decomposed = {}};
  const auto root = plan.graph.outputs()[0];
  const auto node = plan.graph.producer(root);
  plan.primitive_digest = plan.graph.Digest();
  plan.graph_digest = plan.primitive_digest;
  plan.steps.push_back({.nodes = {node}, .implementation = 0});
  static constexpr char group[] = "quoted\" slash\\ tab\t line\n\0\1";
  plan.steps.back().group = {group, sizeof(group) - 1};
  plan.placement.offsets.emplace_back(root, UINT64_MAX);
  plan.placement.extent = UINT64_MAX;
  return plan;
}

TEST(PlannerReportInventoryTest, ActualCandidateWeightObservationNeverGrantsSelection) {
  const auto matrix = ir::Primitive(ir::Op::kMatMul);
  const auto prepare = ir::Primitive(ir::Op::kPrepare);
  auto registry = execution::Registry::Create(
      {execution::Implementation{.name = "report.matmul",
                                 .operation = matrix,
                                 .source = "test",
                                 .revision = "observation",
                                 .build = "host",
                                 .variant = "observation",
                                 .capability = execution::Capability{.op = matrix}},
       execution::Implementation{.name = "report.prepare",
                                 .operation = prepare,
                                 .source = "test",
                                 .revision = "observation",
                                 .build = "host",
                                 .variant = "observation",
                                 .capability = execution::Capability{.op = prepare}}});
  ASSERT_TRUE(registry);
  ir::Builder b;
  b.SetComponent(7);
  const auto x = b.Input(ir::Dense(format::Encoding::kF32, ir::ShapeOf({256, 3})));
  const format::TensorFormat weights{.encoding = format::Encoding::kQ8_0,
                                     .layout = format::Layout::kGgmlBlocks,
                                     .shape = ir::ShapeOf({256, 128})};
  const ir::WeightRef ref{.stage = 2, .component = 7, .role = 3, .layer = 5};
  const auto w = b.Weight(weights, ref);
  b.Output(b.MatMul(w, x));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  const auto bound = base::Sha256{}.Update("actual bound composition").Finish();
  const auto semantic = base::Sha256{}.Update("actual semantic model").Finish();
  const PlanOptions options{.observed_bound_model = bound, .observed_model_spec = semantic};
  Policy policy;
  policy.active_stage = 2;
  auto control = Plan(graph->Clone(), *registry,
                      {.device_class = "sm_86", .arch = 86, .device_model = "unqualified GPU"},
                      policy, options);
  ASSERT_TRUE(control);
  EXPECT_TRUE(control->applied.empty());  // Observation identities cannot authorize a rewrite.
  ASSERT_FALSE(control->missing_numerical.empty());
  for (const auto& missing : control->missing_numerical) EXPECT_EQ(missing.device_arch, 86U);
  std::string exact_report;
  StreamPlanReport(*control, *registry, [&](std::string_view part) { exact_report.append(part); });
  EXPECT_NE(exact_report.find("\"device_arch\":86"), std::string::npos);
  EXPECT_NE(exact_report.find("no matching numerical evidence"), std::string::npos);
  policy.numerical_candidates = {NumericalChange::kQ8VectorReduction};
  auto candidate = Plan(graph->Clone(), *registry, {}, policy, options);
  ASSERT_TRUE(candidate);
  const auto found = std::ranges::find(candidate->applied, "vecq_inputs", &Applied::pattern);
  ASSERT_NE(found, candidate->applied.end());
  ASSERT_EQ(found->numerical_proofs.size(), 1U);
  const auto& proof = found->numerical_proofs.front();
  EXPECT_TRUE(proof.candidate);
  EXPECT_EQ(proof.qualification.evidence, base::Sha256Digest{});
  EXPECT_EQ(proof.qualification.source, base::Sha256Digest{});
  EXPECT_EQ(proof.qualification.semantic, base::Sha256Digest{});
  EXPECT_EQ(proof.qualification.weight_role, UINT16_MAX);
  ASSERT_TRUE(proof.observed.weight);
  EXPECT_EQ(*proof.observed.weight, ref);
  EXPECT_EQ(proof.observed.operand_format, weights);
  EXPECT_EQ(proof.observed.leaf_format, weights);
  EXPECT_EQ(proof.observed.input_format.encoding, format::Encoding::kF32);
  EXPECT_EQ(proof.observed.active_stage, 2U);
  EXPECT_EQ(proof.observed.bound_model, bound);
  EXPECT_EQ(proof.observed.model_spec, semantic);
  // Adding/changing source observations changes neither selection nor plan IDs.
  auto unbound = Plan(graph->Clone(), *registry, {}, policy);
  ASSERT_TRUE(unbound);
  EXPECT_EQ(unbound->graph_digest, candidate->graph_digest);
  EXPECT_EQ(unbound->policy_digest, candidate->policy_digest);
  EXPECT_EQ(unbound->applied.size(), candidate->applied.size());
  std::optional<PlanReportInventory> retained;
  {
    const auto bytes = PlanReportInventory::StorageBytes(*candidate, *registry);
    ASSERT_TRUE(bytes);
    auto inventory = PlanReportInventory::Capture(*candidate, *registry, *bytes);
    ASSERT_TRUE(inventory);
    retained = std::move(*inventory);
  }
  candidate->applied.clear();
  std::string stream;
  retained->Stream([&](std::string_view part) { stream.append(part); });
  bool observed = false;
  std::string_view text = stream;
  while (!text.empty()) {
    const auto end = text.find('\n');
    ASSERT_NE(end, std::string_view::npos);
    const auto line = text.substr(0, end);
    auto json = base::json::Parse(line);
    ASSERT_TRUE(json);
    if (json->root().find("kind")->string() == "pattern") {
      const auto recipient = json->root().find("numerical_recipients")->at(0);
      EXPECT_EQ(recipient.find("status")->string(), "UNQUALIFIED_CANDIDATE");
      EXPECT_EQ(recipient.find("evidence")->string(), std::string(64, '0'));
      EXPECT_EQ(recipient.find("source")->string(), std::string(64, '0'));
      const auto facts = recipient.find("observed");
      ASSERT_TRUE(facts);
      EXPECT_EQ(facts->find("status")->string(), "BOUND_WEIGHT_REF");
      EXPECT_EQ(facts->find("source_status")->string(), "BOUND_MODEL_IDENTITY");
      EXPECT_EQ(facts->find("stage")->int64(), 2);
      EXPECT_EQ(facts->find("component")->int64(), 7);
      EXPECT_EQ(facts->find("weight_role")->int64(), 3);
      EXPECT_EQ(facts->find("layer")->int64(), 5);
      EXPECT_EQ(facts->find("bound_model")->string(), base::ToHex(bound));
      EXPECT_EQ(facts->find("model_spec")->string(), base::ToHex(semantic));
      EXPECT_EQ(facts->find("weight_operand")->find("encoding")->string(), "q8_0");
      EXPECT_EQ(facts->find("input")->find("shape")->at(1).int64(), 3);
      observed = true;
    }
    text.remove_prefix(end + 1);
  }
  EXPECT_TRUE(observed);
}

TEST(PlannerReportInventoryTest, OwnsAllFactsAfterGraphRegistryAndStringsDie) {
  std::optional<PlanReportInventory> retained;
  std::string expected;
  base::Sha256Digest evidence{};
  evidence[0] = 42;
  {
    auto registry = ReportRegistry();
    auto plan = ReportFixture();
    std::string pattern_name = "temporary-pattern\nname";
    std::string group_name = "temporary-group\"\n";
    plan.steps[0].group = group_name;
    plan.applied.push_back({.pattern = pattern_name,
                            .replaced = {{0}},
                            .made = {{0}},
                            .numerical_change = NumericalChange::kBf16ProductInputAndResult,
                            .numerical_evidence = evidence,
                            .numerical_source = evidence,
                            .numerical_semantic = evidence});
    for (std::uint16_t role = 0; role < 2; ++role) {
      auto receipt = evidence;
      receipt[1] = static_cast<std::uint8_t>(role + 1);
      plan.applied.back().numerical_proofs.push_back(
          {.site = {static_cast<std::uint32_t>(role + 10)},
           .weight_operand = static_cast<std::uint8_t>(role),
           .qualification = {.change = NumericalChange::kBf16ProductInputAndResult,
                             .evidence = receipt,
                             .source = evidence,
                             .semantic = evidence,
                             .stage = 2,
                             .component = 7,
                             .anchor = ir::Primitive(ir::Op::kMatMul),
                             .weight_role = role},
           .actual = {.rows = 4, .k = 64, .n = 8 + 8ULL * role}});
    }
    const auto without_proofs = [&] {
      auto proofs = std::move(plan.applied.back().numerical_proofs);
      const auto bytes = PlanReportInventory::StorageBytes(plan, registry).value();
      plan.applied.back().numerical_proofs = std::move(proofs);
      return bytes;
    }();
    EXPECT_GE(PlanReportInventory::StorageBytes(plan, registry).value() - without_proofs,
              2 * sizeof(NumericalProof));
    plan.decomposed.push_back("temporary.fallback\"op");
    plan.missing_numerical.push_back({.change = NumericalChange::kQ8VectorReduction,
                                      .component = 17,
                                      .anchor = ir::Primitive(ir::Op::kMatMul)});
    StreamPlanReport(plan, registry, [&](std::string_view part) { expected.append(part); });
    auto bytes = PlanReportInventory::StorageBytes(plan, registry);
    ASSERT_TRUE(bytes);
    auto report = PlanReportInventory::Capture(plan, registry, *bytes);
    ASSERT_TRUE(report);
    EXPECT_EQ(report->storage_bytes(), *bytes);
    retained = std::move(*report);
    pattern_name.assign(8192, 'x');
    group_name.assign(8192, 'y');
    plan.decomposed[0].assign(8192, 'z');
  }
  std::string actual;
  retained->Stream([&](std::string_view part) { actual.append(part); });
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(actual.find("1234.625"), std::string::npos);
  EXPECT_NE(actual.find(base::ToHex(evidence)), std::string::npos);
  EXPECT_NE(actual.find("18446744073709551615"), std::string::npos);
  std::string_view remaining = actual;
  std::size_t records = 0;
  bool missing = false;
  while (!remaining.empty()) {
    const auto end = remaining.find('\n');
    ASSERT_NE(end, std::string_view::npos);
    const auto record = remaining.substr(0, end);
    auto parsed = base::json::Parse(record);
    ASSERT_TRUE(parsed) << record;
    if (parsed->root().find("kind")->string() == "pattern") {
      const auto recipients = parsed->root().find("numerical_recipients");
      ASSERT_TRUE(recipients);
      ASSERT_EQ(recipients->size(), 2U);
      for (std::size_t i = 0; i < 2; ++i) {
        const auto proof = recipients->at(i);
        EXPECT_EQ(proof.find("primitive_site")->int64(), 10 + static_cast<std::int64_t>(i));
        EXPECT_EQ(proof.find("weight_operand")->int64(), static_cast<std::int64_t>(i));
        EXPECT_EQ(proof.find("stage")->int64(), 2);
        EXPECT_EQ(proof.find("component")->int64(), 7);
        EXPECT_EQ(proof.find("weight_role")->int64(), static_cast<std::int64_t>(i));
        EXPECT_EQ(proof.find("n")->int64(), 8 + 8 * static_cast<std::int64_t>(i));
        auto receipt = evidence;
        receipt[1] = static_cast<std::uint8_t>(i + 1);
        EXPECT_EQ(proof.find("evidence")->string(), base::ToHex(receipt));
        EXPECT_EQ(proof.find("source")->string(), base::ToHex(evidence));
      }
    }
    if (parsed->root().find("kind")->string() == "missing_numerical_qualification") {
      missing = true;
      EXPECT_EQ(parsed->root().find("component")->int64(), 17);
    }
    ++records;
    remaining.remove_prefix(end + 1);
  }
  EXPECT_EQ(records, 6U);
  EXPECT_TRUE(missing);
  retained->Stream({});
}

TEST(PlannerReportInventoryTest, OwnsSourceMultiplicityOutputsAndRootProducerLinks) {
  std::optional<PlanReportInventory> inventory;
  ir::TensorId input{}, output{};
  ir::NodeId producer{};
  {
    auto registry = execution::Registry::Create({execution::Implementation{
        .name = "report.add",
        .operation = ir::Primitive(ir::Op::kAdd),
        .source = "test",
        .revision = "1",
        .build = "host",
        .variant = "exact",
        .capability = execution::Capability{.op = ir::Primitive(ir::Op::kAdd)}}});
    ASSERT_TRUE(registry);
    ir::Builder builder;
    builder.SetComponent(7);
    const auto x = builder.Input(ir::Dense(format::Encoding::kF32, ir::ShapeOf({4})));
    builder.Output(builder.Add(x, x));
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph);
    auto plan = Plan(std::move(*graph), *registry, {.device_class = "cpu"}, {});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    input = plan->graph.inputs()[0];
    output = plan->graph.outputs()[0];
    producer = plan->graph.producer(output);
    const auto bytes = PlanReportInventory::StorageBytes(*plan, *registry);
    ASSERT_TRUE(bytes);
    auto owned = PlanReportInventory::Capture(*plan, *registry, *bytes);
    ASSERT_TRUE(owned);
    inventory = std::move(*owned);
  }
  std::string records;
  inventory->Stream([&](std::string_view fragment) { records.append(fragment); });
  bool selected = false, linked = false;
  for (std::size_t begin = 0; begin < records.size();) {
    const auto end = records.find('\n', begin);
    ASSERT_NE(end, std::string::npos);
    auto parsed = base::json::Parse(std::string_view(records).substr(begin, end - begin));
    ASSERT_TRUE(parsed);
    const auto record = parsed->root();
    if (record.find("kind")->string() == "selection") {
      const auto nodes = record.find("nodes");
      ASSERT_TRUE(nodes);
      ASSERT_EQ(nodes->size(), 1U);
      const auto node = nodes->at(0);
      EXPECT_EQ(node.find("id")->int64(), producer.v);
      EXPECT_EQ(node.find("component")->int64(), 7);
      const auto sources = node.find("sources"), outputs = node.find("outputs");
      ASSERT_TRUE(sources && outputs);
      ASSERT_EQ(sources->size(), 2U);
      EXPECT_EQ(sources->at(0).int64(), input.v);
      EXPECT_EQ(sources->at(1).int64(), input.v);
      ASSERT_EQ(outputs->size(), 1U);
      EXPECT_EQ(outputs->at(0).int64(), output.v);
      selected = true;
    }
    if (record.find("kind")->string() == "activation" && record.find("root")->int64() == output.v) {
      EXPECT_EQ(record.find("producer")->int64(), producer.v);
      EXPECT_EQ(record.find("op")->string(), "add");
      EXPECT_EQ(record.find("component")->int64(), 7);
      linked = true;
    }
    begin = end + 1;
  }
  EXPECT_TRUE(selected);
  EXPECT_TRUE(linked);
}

TEST(PlannerReportInventoryTest, CandidateAndIncumbentStatusesSurviveOwnedCapture) {
  auto registry = ReportRegistry();
  auto plan = ReportFixture();
  plan.applied.push_back({.pattern = "candidate",
                          .replaced = {{0}},
                          .made = {{0}},
                          .numerical_change = NumericalChange::kQ8VectorReduction,
                          .numerical_candidate = true});
  plan.applied.push_back(
      {.pattern = "retained",
       .replaced = {{0}},
       .made = {{0}},
       .numerical_change = NumericalChange::kMxfp8ProductInput,
       .numerical_evidence = base::Sha256{}.Update("accepted source recipe").Finish(),
       .numerical_incumbent = true});
  const auto bytes = PlanReportInventory::StorageBytes(plan, registry);
  ASSERT_TRUE(bytes);
  auto inventory = PlanReportInventory::Capture(plan, registry, *bytes);
  ASSERT_TRUE(inventory);
  plan.applied.clear();
  std::string output;
  inventory->Stream([&](std::string_view fragment) { output.append(fragment); });
  bool candidate = false, retained = false;
  for (std::size_t begin = 0; begin < output.size();) {
    const auto end = output.find('\n', begin);
    ASSERT_NE(end, std::string::npos);
    auto parsed = base::json::Parse(std::string_view(output).substr(begin, end - begin));
    ASSERT_TRUE(parsed);
    const auto status = parsed->root().find("numerical_status");
    if (status && status->string() == "UNQUALIFIED_CANDIDATE") {
      candidate = true;
      EXPECT_EQ(parsed->root().find("numerical_evidence")->string(), std::string(64, '0'));
    }
    if (status && status->string() == "ACCEPTED_INCUMBENT_RECIPE") retained = true;
    begin = end + 1;
  }
  EXPECT_TRUE(candidate);
  EXPECT_TRUE(retained);
}

TEST(PlannerReportInventoryTest, RejectsStalePaidSizeBeforeFillingAndUsesCompactResources) {
  auto registry = ReportRegistry();
  auto plan = ReportFixture();
  // Repeated selections share the registry's one copied identity/name. No
  // serialized characters are retained, including escaping and numeric text.
  for (unsigned i = 0; i < 1000; ++i) plan.steps.push_back(plan.steps.front());
  auto bytes = PlanReportInventory::StorageBytes(plan, registry);
  ASSERT_TRUE(bytes);
  EXPECT_FALSE(PlanReportInventory::Capture(plan, registry, *bytes - 1));
  EXPECT_FALSE(PlanReportInventory::Capture(plan, registry, *bytes + 1));
  auto report = PlanReportInventory::Capture(plan, registry, *bytes);
  ASSERT_TRUE(report);
  std::uint64_t serialized = 0;
  report->Stream([&](std::string_view part) { serialized += part.size(); });
  EXPECT_LT(report->storage_bytes(), serialized);
  plan.decomposed.push_back("new-resource-after-payment");
  EXPECT_FALSE(PlanReportInventory::Capture(plan, registry, *bytes));
}

}  // namespace
}  // namespace llmp::planner
