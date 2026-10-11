// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Host-only audit of actual bound recipes, without a runtime/source opt-in.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/step.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "ggml.h"
#include "ir/builder.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/lower.h"
#include "planner/pattern.h"
#include "planner/plan.h"

namespace {
namespace co = llmp::components;
namespace ex = llmp::execution;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace pl = llmp::planner;
using llmp::format::Encoding;
using llmp::test_support::Failed;

ex::Registry Registry() {
  std::vector<ex::Implementation> all;
  for (const auto& d : kg::CapabilityDeclarations())
    all.push_back({.name = std::string(d.name),
                   .operation = d.operation,
                   .source = "ggml",
                   .revision = "planning",
                   .build = "cpu",
                   .variant = std::string(kg::AcceptsTextOf(d.name)),
                   .capability = *d.capability});
  return ex::Registry::Create(std::move(all)).value();
}
kg::DeviceChoices Device(kg::MulMatPath path = kg::MulMatPath::kTensorCore) {
  kg::DeviceChoices d;
  d.mul_mat = [path](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return path;
  };
  return d;
}
struct Fixture {
  co::StepGraph step;
  pl::StepShape shape;
  std::optional<pl::StepPlan> plan{};
  ir::TensorId weight{}, extra_weight{}, input{}, product{}, extra{}, indices{};
};
Fixture Make(std::span<const std::uint32_t> widths, bool annotated = true,
             bool store_indices = false) {
  std::uint32_t rows = 0;
  std::vector<co::SegmentRows> segments;
  pl::StepShape shape;
  for (std::size_t i = 0; i < widths.size(); ++i) {
    const auto slot = static_cast<std::uint8_t>(5 - i);
    segments.push_back({.slot = slot, .first_row = rows, .rows = widths[i], .outputs = 1});
    shape.segments.push_back({.slot = slot, .rows = widths[i], .outputs = 1});
    rows += widths[i];
  }
  ir::Builder b;
  const auto input = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({32, rows})));
  const auto indices =
      store_indices ? b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({1}))) : ir::TensorId{};
  const ir::WeightRef wr{.role = 1}, nr{.role = 2}, er{.role = 3};
  const auto norm = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({32})), nr);
  const auto weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({8, 8})), wr);
  const auto extra_weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({8, 8})), er);
  const auto hidden = b.Reshape(b.Mul(b.RmsNorm(input, 1e-6F), norm), ir::ShapeOf({8, 4U * rows}));
  const auto product = b.MatMul(weight, hidden);
  const auto extra = b.MatMul(extra_weight, hidden);
  b.Output(product);
  b.Output(extra);
  co::StepGraph step{
      .graph = std::move(b).Finish().value(),
      .inputs = {{.tensor = input, .kind = co::InputKind::kFeatures}},
      .weights = {{.tensor = norm, .ref = nr, .parts = {{"norm", llmp::format::PartRole::kData}}},
                  {.tensor = weight,
                   .ref = wr,
                   .parts = {{"weight", llmp::format::PartRole::kData}}},
                  {.tensor = extra_weight,
                   .ref = er,
                   .parts = {{"extra", llmp::format::PartRole::kData}}}},
      .state = {}};
  if (store_indices) step.inputs.push_back({.tensor = indices, .kind = co::InputKind::kStoreCells});
  if (annotated)
    step.owner_products =
        co::ProveOwnerProducts(step.graph, step.inputs, segments, std::array{product}).value();
  return {.step = std::move(step),
          .shape = std::move(shape),
          .weight = weight,
          .extra_weight = extra_weight,
          .input = input,
          .product = product,
          .extra = extra,
          .indices = indices};
}
void Plan(Fixture& f, const ex::Registry& registry, bool fuse_preparation = false) {
  pl::Policy policy{.enabled = {"bf16_inputs", "owner_row_invariant"}};
  // The initial backend domain authenticates unchanged primitive preparation.
  // A fused preparation is an explicit refusal control below, not a positive.
  if (!fuse_preparation) policy.disabled = {"rms_norm_mul"};
  std::vector<ir::TensorId> staged;
  for (const auto& input : f.step.inputs) staged.push_back(input.tensor);
  auto p =
      pl::Plan(f.step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, policy,
               {.staged = std::move(staged),
                .written = &kg::WrittenBytes,
                .shape = &f.shape,
                .owner_products =
                    f.step.owner_products.products().empty() ? nullptr : &f.step.owner_products});
  ASSERT_TRUE(p) << p.error().detail;
  f.plan.emplace(std::move(*p));
}
auto Lower(const Fixture& f, const ex::Registry& registry,
           const kg::DeviceChoices& device = Device(),
           const llmp::kernels::Addresses& addresses = {}) {
  return kg::LowerStep({.plan = *f.plan, .step = f.step, .shape = f.shape, .addresses = addresses},
                       registry, device);
}
std::size_t Count(const kg::LoweredStep& p, std::string_view implementation) {
  return static_cast<std::size_t>(
      std::ranges::count(p.plan.steps, implementation, &kg::PlanStep::implementation));
}

// Rebuild valid IR, then recompute ordinary selection and placement. Retain
// the authentic proof/scopes with fully composed original tensor/node maps;
// a negative must reach the owner audit, not fail on malformed plan metadata.
void Rebuild(Fixture& f, const ex::Registry& registry,
             const std::function<void(const ir::Node&, pl::RewriteContext&)>& rewrite) {
  auto old = std::move(*f.plan);
  f.plan.reset();
  ir::Builder b;
  std::vector<ir::TensorId> map(old.graph.tensor_count());
  std::vector<ir::NodeId> from;
  pl::RewriteContext copy(old.graph, b, map);
  for (std::uint32_t i = 0; i < old.graph.nodes().size(); ++i) {
    const auto& n = old.graph.node({i});
    b.SetComponent(n.component);
    if (n.batch == ir::Batch::kPerOwner)
      b.SetOwner(old.graph.tensor(n.out[0]).segment);
    else
      b.SetJoined();
    rewrite(n, copy);
    ASSERT_TRUE(b.ok()) << b.error()->detail;
    from.resize(b.node_count(), ir::NodeId{i});
  }
  for (auto t : old.graph.outputs()) b.Output(copy.Map(t));
  for (auto t : old.graph.keep()) b.Keep(copy.Map(t));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << Failed(graph, &ir::VerifyError::detail).value_or("");
  pl::Policy policy;
  for (const auto& p : pl::Patterns()) policy.disabled.emplace_back(p.name);
  pl::Canonicalize(policy);
  auto next = pl::Plan(std::move(*graph), registry, {.device_class = "sm_121", .arch = 121}, policy,
                       {.written = &kg::WrittenBytes});
  ASSERT_TRUE(next) << Failed(next, &pl::PlanFailure::detail).value_or("");
  const auto composed = [&](ir::TensorId t) {
    return t.valid() ? next->tensors.at(map.at(t.v).v) : ir::TensorId{};
  };
  auto tensors = old.tensors;
  for (auto& t : tensors) t = composed(t);
  auto scopes = old.owner_products;
  for (auto& s : scopes) s.result = composed(s.result);
  for (auto& n : next->from) n = old.from.at(from.at(n.v).v);
  next->tensors = std::move(tensors);
  next->owner_products = std::move(scopes);
  next->owner_digest = old.owner_digest;
  next->owner_row_invariant = old.owner_row_invariant;
  f.plan.emplace(std::move(*next));
}

TEST(GgmlOwnerProductsTest, OrdinaryOwnerRecipesAndPhysicalPartitionsAreRetained) {
  const auto registry = Registry();
  for (const auto& widths :
       {std::vector{1U}, std::vector{3U}, std::vector{1U, 1U}, std::vector{1U, 3U},
        std::vector{3U, 1U}, std::vector{3U, 3U}, std::vector{4U, 4U}}) {
    auto f = Make(widths);
    Plan(f, registry);
    ASSERT_TRUE(f.plan);
    auto lowered = Lower(f, registry);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    std::vector<std::int64_t> columns;
    for (const auto& step : lowered->plan.steps) {
      if (step.implementation != kg::kMulMatTensorCore) continue;
      for (const auto* n : step.nodes)
        if (n->src[0] != nullptr && n->src[0]->type == GGML_TYPE_BF16 &&
            (widths.size() == 1 || n->src[1]->op == GGML_OP_VIEW))
          columns.push_back(n->src[1]->ne[1]);
    }
    if (widths.size() == 1) {
      // The second unannotated product has the same lone width.
      EXPECT_EQ(columns, (std::vector<std::int64_t>{4 * widths[0], 4 * widths[0]}));
    } else {
      std::vector<std::int64_t> expected;
      for (const auto w : widths) expected.push_back(4 * w);
      EXPECT_EQ(columns, expected);
    }
    if (widths == std::vector{1U, 1U}) {
      EXPECT_TRUE(f.plan->owner_row_invariant);
      EXPECT_EQ(Count(*lowered, kg::kMulMatTensorCore), 2U);
      EXPECT_EQ(Count(*lowered, kg::kMulMatVecFRows),
                1U);  // unannotated product
    }
  }
}

TEST(GgmlOwnerProductsTest, UnannotatedAndExplicitVectorPoliciesKeepTheirMeaning) {
  const auto registry = Registry();
  auto plain = Make(std::array{1U, 1U}, false);
  Plan(plain, registry);
  ASSERT_TRUE(plain.plan);
  const auto unannotated = Lower(plain, registry);
  ASSERT_TRUE(unannotated) << Failed(unannotated, &kg::KernelFailure::detail).value_or("");
  EXPECT_EQ(Count(*unannotated, kg::kMulMatVecFRows), 2U);
  for (const auto explicit_policy : {0, 1, 2}) {
    auto f = Make(std::array{1U, 1U});
    Plan(f, registry);
    ASSERT_TRUE(f.plan);
    auto d = Device();
    d.vector_floats = explicit_policy == 0;
    d.row_invariant = explicit_policy == 1;
    if (explicit_policy == 2) d.vector_float_node = [](const ggml_tensor*) { return true; };
    const auto lowered = Lower(f, registry, d);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    EXPECT_EQ(Count(*lowered, kg::kMulMatVecFRows), 3U);
  }
  auto f = Make(std::array{3U});
  Plan(f, registry);
  ASSERT_TRUE(f.plan);
  auto d = Device();
  d.row_invariant = true;
  EXPECT_FALSE(Lower(f, registry, d));
}

TEST(GgmlOwnerProductsTest, EveryRequestedOwnerNeedsExactlyOneAuthenticScope) {
  const auto registry = Registry();
  for (int changed = 0; changed < 9; ++changed) {
    SCOPED_TRACE(changed);
    auto f = Make(std::array{1U, 3U});
    Plan(f, registry);
    ASSERT_TRUE(f.plan);
    switch (changed) {
      case 0:
        f.plan->owner_products.clear();
        break;
      case 1:
        f.plan->owner_products.push_back(f.plan->owner_products.front());
        break;
      case 2:
        f.plan->owner_products[0].slot = 8;
        break;
      case 3:
        f.plan->owner_products[0].columns = 5;
        break;
      case 4:
        f.plan->owner_products[0].result = {};
        break;
      case 5:
        f.plan->owner_digest[0] ^= 1;
        break;
      case 6:
        f.step.owner_products = {};
        break;
      case 7:
        f.shape.segments[0].rows = 2;
        break;
      case 8:
        f.step.inputs[0].kind = co::InputKind::kOutputRows;
        break;
    }
    auto out = Lower(f, registry);
    ASSERT_FALSE(out);
    EXPECT_NE(out.error().detail.find("owner products:"), std::string::npos) << out.error().detail;
  }
}

TEST(GgmlOwnerProductsTest, OriginalBindingAndInputMapsCannotBeSubstituted) {
  const auto registry = Registry();
  for (int changed = 0; changed < 4; ++changed) {
    auto f = Make(std::array{1U, 3U});
    Plan(f, registry);
    ASSERT_TRUE(f.plan);
    if (changed == 0) f.plan->tensors[f.weight.v] = f.plan->tensors[f.extra_weight.v];
    if (changed == 1) f.step.weights[1].ref = f.step.weights[2].ref;
    if (changed == 2) f.step.weights[1].parts[0].second = llmp::format::PartRole::kCodes;
    if (changed == 3) f.plan->tensors[f.input.v] = f.plan->tensors[f.extra_weight.v];
    auto out = Lower(f, registry);
    ASSERT_FALSE(out);
    EXPECT_NE(out.error().detail.find("owner products:"), std::string::npos) << out.error().detail;
  }
}

TEST(GgmlOwnerProductsTest, ValidRebuiltIrCannotChangeProtectedOperandsOrOrder) {
  const auto registry = Registry();
  // -1 is an unchanged rebuild, followed by offset, stride, weight,
  // activation conversion, accumulation precision and concat-order changes.
  for (int changed = -1; changed < 6; ++changed) {
    SCOPED_TRACE(changed);
    auto f = Make(std::array{1U, 3U});
    Plan(f, registry);
    ASSERT_TRUE(f.plan);
    const auto product = f.plan->owner_products[0].result;
    const auto view = f.plan->graph.node(f.plan->graph.producer(product)).src[1];
    const auto extra_weight = f.plan->tensors[f.extra_weight.v];
    const auto concat = f.plan->tensors[f.product.v];
    Rebuild(f, registry, [&](const ir::Node& n, pl::RewriteContext& copy) {
      auto& b = copy.builder();
      if (n.out[0] == view && (changed == 0 || changed == 1)) {
        const auto& d = copy.original().tensor(view);
        auto nb = d.nb;
        if (changed == 1) nb[1] += 32;
        const auto offset = d.view_offset - copy.original().tensor(n.src[0]).view_offset;
        copy.Set(view, b.View(copy.Map(n.src[0]), d.shape(), nb, offset + (changed == 0 ? 32 : 0)));
      } else if (n.out[0] == product && changed >= 2 && changed <= 4) {
        auto input = copy.Map(n.src[1]);
        if (changed == 3) input = b.Convert(input, Encoding::kBF16);
        auto params = n.Params<ir::ProductParams>();
        if (changed == 4) params.accumulate = ir::Accumulate::kF32;
        copy.Set(product,
                 b.MatMul(copy.Map(changed == 2 ? extra_weight : n.src[0]), input, params));
      } else if (n.out[0] == concat && changed == 5) {
        copy.Set(concat, b.Concat(copy.Map(n.src[1]), copy.Map(n.src[0]), 1));
      } else {
        copy.Copy(n);
      }
    });
    ASSERT_TRUE(f.plan);
    const auto out = Lower(f, registry);
    if (changed < 0) {
      ASSERT_TRUE(out) << Failed(out, &kg::KernelFailure::detail).value_or("");
    } else {
      ASSERT_FALSE(out);
      EXPECT_NE(out.error().detail.find("owner products:"), std::string::npos)
          << out.error().detail;
    }
  }
}

TEST(GgmlOwnerProductsTest, ProductAndEveryConcatRootRejectAliasStores) {
  const auto registry = Registry();
  for (int changed = 0; changed < 4; ++changed) {
    SCOPED_TRACE(changed);
    // A singleton, a joined owner output, an intermediate concat, and the
    // final concat. The three-owner case distinguishes both concat roots.
    auto f = Make(changed == 0 ? std::vector{1U} : std::vector{1U, 1U, 1U}, true, true);
    Plan(f, registry);
    ASSERT_TRUE(f.plan);
    auto target = f.plan->owner_products[0].result;
    if (changed == 2) {
      const auto at =
          std::ranges::find(f.plan->graph.nodes(), ir::Primitive(ir::Op::kConcat), &ir::Node::key);
      ASSERT_NE(at, f.plan->graph.nodes().end());
      target = at->out[0];
      ASSERT_NE(target, f.plan->tensors[f.product.v]);
    }
    if (changed == 3) target = f.plan->tensors[f.product.v];
    const auto indices = f.plan->tensors[f.indices.v];
    Rebuild(f, registry, [&](const ir::Node& n, pl::RewriteContext& copy) {
      copy.Copy(n);
      if (n.out[0] != target) return;
      auto& b = copy.builder();
      const auto& d = copy.original().tensor(target);
      const auto alias = b.View(copy.Map(target), d.shape(), d.nb, 0);
      const auto value = b.Fill(ir::ShapeOf({d.ne(0), 1}), -3.0F);
      b.Keep(b.CacheStore(alias, value, copy.Map(indices)));
    });
    ASSERT_TRUE(f.plan);
    const auto out = Lower(f, registry);
    ASSERT_FALSE(out);
    EXPECT_EQ(out.error().detail,
              "lowering onto GGML: owner products: overwritten product or concat output");

    // The rebuilt store is an otherwise valid, selected and placed ordinary
    // operation. Only the requested owner-preservation contract forbids it.
    f.step.owner_products = {};
    f.plan->owner_products.clear();
    f.plan->owner_digest = {};
    const auto ordinary = Lower(f, registry);
    ASSERT_TRUE(ordinary) << Failed(ordinary, &kg::KernelFailure::detail).value_or("");
  }
}

TEST(GgmlOwnerProductsTest, SuppliedAddressesKeepOffsetsAndRefuseOverlapOrMisalignment) {
  const auto registry = Registry();
  auto f = Make(std::array{1U, 3U});
  Plan(f, registry);
  ASSERT_TRUE(f.plan);
  constexpr std::uint64_t activation_base = 0x200000000ULL, weight_base = 0x400000000ULL;
  const auto placed = [&](ir::TensorId t) {
    const auto root = f.plan->graph.root(t);
    const auto at = std::ranges::find(f.plan->placement.offsets, root,
                                      &std::pair<ir::TensorId, std::uint64_t>::first);
    EXPECT_NE(at, f.plan->placement.offsets.end());
    return at == f.plan->placement.offsets.end() ? 0 : activation_base + at->second;
  };
  const auto input_base = placed(f.plan->tensors[f.step.owner_products.products()[0].input.v]);
  const auto output_base = placed(f.plan->owner_products[0].result);
  for (int changed = 0; changed < 4; ++changed) {
    SCOPED_TRACE(changed);
    // Distinct supplied addresses, output/weight overlap, a misaligned
    // activation region, and a misaligned original weight respectively.
    const llmp::kernels::Addresses addresses{
        .weight =
            [=](const ir::WeightRef& ref, llmp::format::PartRole part) {
              EXPECT_EQ(part, llmp::format::PartRole::kData);
              if (ref.role == 1 && changed == 1) return output_base;
              return weight_base + std::uint64_t{ref.role} * 65536 +
                     (ref.role == 1 && changed == 3 ? 2U : 0U);
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = activation_base + (changed == 2 ? 1U : 0U),
        .activation_bytes = f.plan->placement.extent};
    const auto out = Lower(f, registry, Device(), addresses);
    if (changed != 0) {
      ASSERT_FALSE(out);
      EXPECT_NE(out.error().detail.find(changed == 1 ? "overlapping" : "misaligned"),
                std::string::npos)
          << out.error().detail;
      continue;
    }
    ASSERT_TRUE(out) << Failed(out, &kg::KernelFailure::detail).value_or("");
    EXPECT_FALSE(out->measuring);
    std::vector<std::uint64_t> inputs;
    for (const auto& step : out->plan.steps)
      for (const auto* node : step.nodes)
        if (node->op == GGML_OP_MUL_MAT &&
            reinterpret_cast<std::uintptr_t>(node->src[0]->data) == weight_base + 65536)
          inputs.push_back(reinterpret_cast<std::uintptr_t>(node->src[1]->data));
    EXPECT_EQ(inputs, (std::vector<std::uint64_t>{input_base, input_base + 128}));
  }
}

TEST(GgmlOwnerProductsTest, DifferentConcreteMemberIsNotCoveredByTheSelectedGroup) {
  const auto registry = Registry();
  auto f = Make(std::array{1U, 3U});
  Plan(f, registry);
  ASSERT_TRUE(f.plan);
  std::size_t calls = 0;
  auto d = Device();
  d.mul_mat = [&calls](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    // The two authenticated original-owner descriptors expect MMF. Actual
    // planning then resolves another member of the same registered group.
    return ++calls <= 2 ? kg::MulMatPath::kTensorCore : kg::MulMatPath::kVector;
  };
  const auto out = Lower(f, registry, d);
  ASSERT_FALSE(out);
  EXPECT_NE(out.error().detail.find("actual concrete recipe differs"), std::string::npos)
      << out.error().detail;
}

TEST(GgmlOwnerProductsTest, UnsupportedOrMissingConcreteSelectorRefuses) {
  const auto registry = Registry();
  auto f = Make(std::array{1U});
  Plan(f, registry);
  ASSERT_TRUE(f.plan);
  EXPECT_FALSE(Lower(f, registry, Device(kg::MulMatPath::kCublas)));
  auto d = Device();
  d.mul_mat = {};
  EXPECT_FALSE(Lower(f, registry, d));
  const auto vector = Lower(f, registry, Device(kg::MulMatPath::kVector));
  ASSERT_TRUE(vector) << Failed(vector, &kg::KernelFailure::detail).value_or("");
  EXPECT_EQ(Count(*vector, kg::kMulMatVector), 2U);

  auto fused = Make(std::array{1U, 3U});
  Plan(fused, registry, true);
  ASSERT_TRUE(fused.plan);
  ASSERT_TRUE(std::ranges::any_of(
      fused.plan->applied, [](const pl::Applied& a) { return a.pattern == "rms_norm_mul"; }));
  const auto refused = Lower(fused, registry);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().detail,
            "lowering onto GGML: owner products: unmapped or overwritten preparation");
}
}  // namespace
