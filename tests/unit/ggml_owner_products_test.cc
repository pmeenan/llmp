// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Host-only audit of actual bound recipes, without a runtime/source opt-in.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
#include "kernels/ggml/llmp_ops.h"
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
  return kg::LowerStep({*f.plan, f.step, f.shape, addresses}, registry, device);
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
  next->owner_rows_policy_enabled = old.owner_rows_policy_enabled;
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

Fixture MakeSelectedHc(std::span<const std::uint32_t> widths, std::uint64_t width = 32,
                       std::uint64_t rank = 16, bool all_outputs = false, std::uint64_t hc = 4) {
  pl::StepShape shape;
  const float scale = 1.0F / static_cast<float>(hc);
  std::uint64_t rows = 0, outputs = 0;
  for (std::size_t i = 0; i < widths.size(); ++i) {
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(i),
                              .rows = widths[i],
                              .outputs = all_outputs ? widths[i] : 1});
    rows += widths[i];
    outputs += shape.segments.back().outputs;
  }
  ir::Builder b;
  const auto streams = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, hc, rows})));
  const auto ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({outputs})));
  const auto stores = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({1})));
  const ir::WeightRef nr{.role = 1}, dr{.role = 2}, ur{.role = 3};
  const auto norm = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({width * hc})), nr);
  const auto down = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({width * hc, rank})), dr);
  const auto up = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({rank, width * hc})), ur);
  const auto mix = [&](ir::TensorId x, std::uint64_t columns) {
    auto xn = b.Mul(b.Reshape(b.RmsNorm(x, 1e-6F), ir::ShapeOf({width * hc, columns})), norm);
    auto lo = b.Unary(b.Scale(b.MatMul(down, xn), scale), ir::UnaryFn::kSilu);
    auto gate = b.Unary(b.MatMul(up, lo), ir::UnaryFn::kSigmoid);
    auto gated = b.Reshape(b.Mul(xn, gate), ir::ShapeOf({width, hc, columns}));
    const auto nb = b.tensor(gated).nb;
    const std::array<std::uint64_t, 4> strides{nb[0], nb[2], nb[3], nb[3]};
    auto folded = b.Cont(b.View(gated, ir::ShapeOf({width, columns}), strides, 0));
    for (std::uint64_t s = 1; s < hc; ++s)
      folded = b.Add(folded, b.View(gated, ir::ShapeOf({width, columns}), strides, s * nb[1]));
    return b.Scale(folded, scale);
  };
  // Both full-input and selected-output HC pairs are in the actual plan.
  b.Output(mix(streams, rows));
  auto selected = b.Reshape(b.GetRows(b.Reshape(streams, ir::ShapeOf({width * hc, rows})), ids),
                            ir::ShapeOf({width, hc, outputs}));
  b.Output(mix(selected, outputs));
  co::StepGraph step{
      .graph = std::move(b).Finish().value(), .inputs = {}, .weights = {}, .state = {}};
  step.inputs = {{.tensor = streams, .kind = co::InputKind::kFeatures},
                 {.tensor = ids, .kind = co::InputKind::kOutputRows},
                 {.tensor = stores, .kind = co::InputKind::kStoreCells}};
  step.weights = {{.tensor = norm, .ref = nr, .parts = {{"norm", llmp::format::PartRole::kData}}},
                  {.tensor = down, .ref = dr, .parts = {{"down", llmp::format::PartRole::kData}}},
                  {.tensor = up, .ref = ur, .parts = {{"up", llmp::format::PartRole::kData}}}};
  return {.step = std::move(step), .shape = std::move(shape), .indices = stores};
}
void PlanSelectedHc(Fixture& f, const ex::Registry& registry, bool enabled = true) {
  pl::Policy policy{.enabled = {"hc_prep"}, .disabled = {"rms_norm_mul"}};
  if (enabled)
    policy.enabled.push_back("owner_row_invariant");
  else
    policy.disabled.push_back("owner_row_invariant");
  pl::Canonicalize(policy);
  std::vector<ir::TensorId> staged;
  for (const auto& i : f.step.inputs) staged.push_back(i.tensor);
  auto planned =
      pl::Plan(f.step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, policy,
               {.staged = std::move(staged),
                .written = &kg::WrittenBytes,
                .within = &kg::WithinRoot,
                .shape = &f.shape});
  ASSERT_TRUE(planned) << Failed(planned, &pl::PlanFailure::detail).value_or("");
  f.plan.emplace(std::move(*planned));
}
// Authenticate all four physical products against the original weight leaves
// and the planned graph, including the primitive fallback. A custom-kernel
// name filter alone can silently return no operations when hc_prep refuses.
void CheckHcProducts(const Fixture& f, const kg::LoweredStep& lowered, const ex::Registry& registry,
                     bool fused, std::vector<const ggml_tensor*>& products) {
  const auto& original = f.step.graph;
  const auto& planned = f.plan->graph;
  std::vector<const ir::Node*> old_products, plan_products;
  for (const auto& n : original.nodes())
    if (n.key == ir::Primitive(ir::Op::kMatMul)) old_products.push_back(&n);
  for (const auto& n : planned.nodes())
    if (n.key == ir::Primitive(ir::Op::kMatMul)) plan_products.push_back(&n);
  ASSERT_EQ(old_products.size(), 4U);
  ASSERT_EQ(plan_products.size(), 4U);
  EXPECT_EQ(std::ranges::count_if(
                planned.nodes(),
                [](const ir::Node& n) { return n.key == ir::Fused(ir::FusedKind::kHcPrep); }),
            fused ? 2 : 0);
  EXPECT_EQ(std::ranges::count_if(
                planned.nodes(),
                [](const ir::Node& n) { return n.key == ir::Fused(ir::FusedKind::kHcFold); }),
            fused ? 2 : 0);
  std::vector<const kg::PlanStep*> operations;
  for (const auto& operation : lowered.plan.steps)
    if (operation.operation == ex::Operation::kMatMul) operations.push_back(&operation);
  ASSERT_EQ(operations.size(), 4U);
  products.clear();
  for (std::size_t i = 0; i < operations.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& old = *old_products[i];
    const auto& made = *plan_products[i];
    const auto& weight = f.step.weights[1 + i % 2];
    ASSERT_EQ(old.src[0], weight.tensor);
    ASSERT_EQ(made.src[0], f.plan->tensors[weight.tensor.v]);
    EXPECT_EQ(original.tensor(old.src[0]).format, planned.tensor(made.src[0]).format);
    EXPECT_EQ(original.tensor(old.src[0]).binding, planned.tensor(made.src[0]).binding);
    EXPECT_EQ(old.params, made.params);
    const auto id = planned.producer(made.out[0]);
    const pl::Selected* selection = nullptr;
    for (const auto& step : f.plan->steps) {
      if (std::ranges::find(step.nodes, id) == step.nodes.end()) continue;
      ASSERT_EQ(selection, nullptr);
      selection = &step;
    }
    ASSERT_NE(selection, nullptr);
    ASSERT_EQ(selection->nodes.size(), 1U);
    if (fused)
      EXPECT_EQ(registry.at(selection->implementation).name, kg::kGemmBf16Name);
    else
      EXPECT_EQ(selection->group, "ggml.mul_mat");
    const auto& operation = *operations[i];
    EXPECT_EQ(operation.implementation, fused                         ? kg::kGemmBf16Name
                                        : f.plan->owner_row_invariant ? kg::kMulMatVecFRows
                                                                      : kg::kMulMatTensorCore);
    ASSERT_EQ(operation.nodes.size(), 1U);
    const auto* node = operation.nodes[0];
    const ggml_tensor* leaf = nullptr;
    for (const auto& candidate : lowered.leaves) {
      if (candidate.primitive != weight.tensor) continue;
      ASSERT_EQ(leaf, nullptr);
      leaf = candidate.tensor;
    }
    ASSERT_NE(leaf, nullptr);
    ASSERT_EQ(node->src[0], leaf);
    ASSERT_NE(node->src[1], nullptr);
    EXPECT_EQ(leaf->type, GGML_TYPE_BF16);
    EXPECT_EQ(node->src[1]->type, fused ? GGML_TYPE_BF16 : GGML_TYPE_F32);
    EXPECT_EQ(node->type, fused && i % 2 == 1 ? GGML_TYPE_BF16 : GGML_TYPE_F32);
    for (std::size_t dim = 0; dim < 4; ++dim) {
      EXPECT_EQ(static_cast<std::uint64_t>(node->ne[dim]), planned.tensor(made.out[0]).ne(dim));
      EXPECT_EQ(static_cast<std::uint64_t>(node->src[0]->ne[dim]),
                planned.tensor(made.src[0]).ne(dim));
      EXPECT_EQ(static_cast<std::uint64_t>(node->src[1]->ne[dim]),
                planned.tensor(made.src[1]).ne(dim));
    }
    if (fused) {
      EXPECT_EQ(node->op, GGML_OP_CUSTOM);
      EXPECT_EQ(kg::LlmpOpOf(node), kg::LlmpOp::kGemmBf16);
      const auto& from = planned.node(planned.producer(planned.root(made.src[1])));
      EXPECT_EQ(from.key,
                ir::Fused(i % 2 == 0 ? ir::FusedKind::kHcPrep : ir::FusedKind::kScaledUnary));
      const auto* source = node->src[1];
      while (source->view_src != nullptr) source = source->view_src;
      EXPECT_EQ(source->op, GGML_OP_CUSTOM);
      EXPECT_EQ(kg::LlmpOpOf(source), i % 2 == 0 ? kg::LlmpOp::kHcPrep : kg::LlmpOp::kHcLo);
    } else {
      EXPECT_EQ(node->op, GGML_OP_MUL_MAT);
      EXPECT_EQ(f.plan->tensors[old.out[0].v], made.out[0]);
      EXPECT_EQ(f.plan->tensors[old.src[1].v], made.src[1]);
      EXPECT_EQ(node->src[1]->op, i % 2 == 0 ? GGML_OP_MUL : GGML_OP_UNARY);
    }
    products.push_back(node);
  }
}

TEST(GgmlOwnerProductsTest, SelectedOutputRowsProtectOnlyTheirFinalPair) {
  const auto registry = Registry();
  for (const auto& widths : {std::vector{1U, 1U}, std::vector{3U, 3U}, std::vector{1U, 3U},
                             std::vector{3U, 1U}, std::vector{4U, 4U}}) {
    auto f = MakeSelectedHc(widths);
    PlanSelectedHc(f, registry);
    ASSERT_TRUE(f.plan);
    const auto lowered = Lower(f, registry);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    std::vector<const ggml_tensor*> products;
    ASSERT_NO_FATAL_FAILURE(CheckHcProducts(f, *lowered, registry, true, products));
    ASSERT_EQ(products.size(), 4U);
    for (std::size_t i = 0; i < products.size(); ++i) {
      const bool wide = i >= 2 || widths == std::vector{1U, 1U};
      EXPECT_EQ(kg::GemvBf16Wide(products[i]), wide);
      EXPECT_EQ(kg::UsesGemvBf16Vector(products[i]), wide);
    }
  }
}
TEST(GgmlOwnerProductsTest, UnoptedSingletonAndOldWideFallbackStayUnchanged) {
  const auto registry = Registry();
  for (const auto& widths : {std::vector{3U}, std::vector{3U, 3U}}) {
    auto f = MakeSelectedHc(widths);
    PlanSelectedHc(f, registry, false);
    ASSERT_TRUE(f.plan);
    const auto lowered = Lower(f, registry);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    std::vector<const ggml_tensor*> products;
    ASSERT_NO_FATAL_FAILURE(CheckHcProducts(f, *lowered, registry, true, products));
    ASSERT_EQ(products.size(), 4U);
    for (const auto* node : products) {
      EXPECT_EQ(kg::LlmpOpInt(node, 0), 1);
      EXPECT_FALSE(kg::GemvBf16Wide(node));
    }
  }
  // width8192/hc4 does not enter hc_prep (width <=4096). Keep its real
  // BF16-weight/F32-input primitive products as a separate fallback control.
  for (const auto& widths : {std::vector{1U, 1U}, std::vector{3U, 3U}}) {
    auto primitive = MakeSelectedHc(widths, 8192);
    PlanSelectedHc(primitive, registry);
    ASSERT_TRUE(primitive.plan);
    const auto plain = Lower(primitive, registry);
    ASSERT_TRUE(plain) << Failed(plain, &kg::KernelFailure::detail).value_or("");
    std::vector<const ggml_tensor*> products;
    ASSERT_NO_FATAL_FAILURE(CheckHcProducts(primitive, *plain, registry, false, products));
    ASSERT_EQ(products.size(), 4U);
  }
  // Eight streams keep the actual HC prep supported at width4096 while
  // its down-product K32768 exceeds the existing vector branch's limit.
  auto old = MakeSelectedHc(std::array{1U, 1U}, 4096, 16, false, 8);
  PlanSelectedHc(old, registry);
  ASSERT_TRUE(old.plan);
  const auto lowered = Lower(old, registry);
  ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
  std::vector<const ggml_tensor*> products;
  ASSERT_NO_FATAL_FAILURE(CheckHcProducts(old, *lowered, registry, true, products));
  ASSERT_EQ(products.size(), 4U);
  for (std::size_t i = 0; i < products.size(); ++i) {
    EXPECT_EQ(kg::LlmpOpInt(products[i], 0), 2);
    EXPECT_TRUE(kg::GemvBf16Wide(products[i]));
    // The low product retains cuBLAS fallback, the high product fits vector.
    EXPECT_EQ(kg::UsesGemvBf16Vector(products[i]), i % 2 == 1);
  }
  auto protected_wide = MakeSelectedHc(std::array{3U, 3U}, 4096, 16, false, 8);
  PlanSelectedHc(protected_wide, registry);
  ASSERT_TRUE(protected_wide.plan);
  EXPECT_TRUE(protected_wide.plan->owner_rows_policy_enabled);
  EXPECT_FALSE(protected_wide.plan->owner_row_invariant);
  const auto& graph = protected_wide.plan->graph;
  EXPECT_EQ(std::ranges::count_if(
                graph.nodes(),
                [](const ir::Node& n) { return n.key == ir::Fused(ir::FusedKind::kHcPrep); }),
            2);
  const auto refused = Lower(protected_wide, registry);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().detail,
            "lowering onto GGML: selected HC: protected vector geometry is unsupported");
  // The same graph/shape remains valid without the protected recipe policy.
  PlanSelectedHc(protected_wide, registry, false);
  const auto ordinary = Lower(protected_wide, registry);
  ASSERT_TRUE(ordinary) << Failed(ordinary, &kg::KernelFailure::detail).value_or("");
  ASSERT_NO_FATAL_FAILURE(CheckHcProducts(protected_wide, *ordinary, registry, true, products));
  ASSERT_EQ(products.size(), 4U);
  for (const auto* node : products) EXPECT_EQ(kg::LlmpOpInt(node, 0), 1);
}
TEST(GgmlOwnerProductsTest, MultipleOutputsKeepTheOrdinaryHcRecipe) {
  const auto registry = Registry();
  for (const auto& widths : {std::vector{3U, 3U}, std::vector{3U, 1U}}) {
    for (const auto& [width, streams] :
         {std::pair{32U, 4U}, std::pair{4096U, 8U}, std::pair{8192U, 4U}}) {
      SCOPED_TRACE(width);
      auto f = MakeSelectedHc(widths, width, 16, true, streams);
      PlanSelectedHc(f, registry);
      ASSERT_TRUE(f.plan);
      const auto lowered = Lower(f, registry);
      ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
      const bool fused = width <= 4096;
      std::vector<const ggml_tensor*> products;
      ASSERT_NO_FATAL_FAILURE(CheckHcProducts(f, *lowered, registry, fused, products));
      ASSERT_EQ(products.size(), 4U);
      for (const auto* node : products) {
        if (fused) {
          EXPECT_EQ(kg::LlmpOpInt(node, 0), 1);  // unchanged ordinary HC selector
          EXPECT_FALSE(kg::GemvBf16Wide(node));
          EXPECT_FALSE(kg::UsesGemvBf16Vector(node));
        }
        EXPECT_EQ(node->ne[1], widths[0] + widths[1]);
      }
      if (!fused) continue;  // no selected-HC corridor exists in the primitive plan
      // Coherent multi-output geometry is ordinary; changing the metadata
      // without changing the actual gather must still refuse before lowering.
      --f.shape.segments[0].outputs;
      const auto refused = Lower(f, registry);
      ASSERT_FALSE(refused);
      EXPECT_NE(refused.error().detail.find("selected HC:"), std::string::npos);
    }
  }
}
TEST(GgmlOwnerProductsTest, SelectedPairRequiresActualCanonicalStagingAndMaps) {
  const auto registry = Registry();
  for (int change = 0; change < 7; ++change) {
    SCOPED_TRACE(change);
    auto f = MakeSelectedHc(std::array{3U, 3U});
    PlanSelectedHc(f, registry);
    ASSERT_TRUE(f.plan);
    if (change == 0) f.step.inputs[1].kind = co::InputKind::kTokens;
    if (change == 1) f.step.inputs.push_back(f.step.inputs[1]);
    if (change == 2) f.step.inputs[1].segment = 0;
    if (change == 3) f.step.inputs[1].sections = 2;
    if (change == 4) f.shape.segments[0].outputs = 2;
    if (change == 5) f.plan->tensors[f.step.weights[1].tensor.v] = {};
    if (change == 6) ++f.step.weights[1].ref.role;
    const auto lowered = Lower(f, registry);
    ASSERT_FALSE(lowered);
    EXPECT_NE(lowered.error().detail.find("selected HC:"), std::string::npos)
        << lowered.error().detail;
  }
}
TEST(GgmlOwnerProductsTest, SelectedPairRejectsValidAliasedOutputWrites) {
  const auto registry = Registry();
  for (int which = 0; which < 4; ++which) {
    auto f = MakeSelectedHc(std::array{3U, 3U});
    PlanSelectedHc(f, registry);
    ASSERT_TRUE(f.plan);
    std::vector<ir::TensorId> final;
    for (const auto& n : f.plan->graph.nodes()) {
      if (n.key == ir::Fused(ir::FusedKind::kHcPrep) && f.plan->graph.tensor(n.out[0]).ne(1) == 2)
        final.push_back(n.out[0]);
      if (n.key == ir::Primitive(ir::Op::kMatMul) && f.plan->graph.tensor(n.out[0]).ne(1) == 2)
        final.push_back(n.out[0]);
      if (n.key == ir::Fused(ir::FusedKind::kHcFold) && f.plan->graph.tensor(n.out[0]).ne(1) == 2)
        final.push_back(n.out[0]);
    }
    ASSERT_EQ(final.size(), 4U);
    const auto target = final[static_cast<std::size_t>(which)];
    const auto indices = f.plan->tensors[f.indices.v];
    Rebuild(f, registry, [&](const ir::Node& n, pl::RewriteContext& copy) {
      copy.Copy(n);
      if (n.out[0] != target) return;
      auto& b = copy.builder();
      const auto& d = copy.original().tensor(target);
      auto value = b.Fill(ir::ShapeOf({d.ne(0), 1}), -3.0F);
      if (d.format.encoding == Encoding::kBF16) value = b.Convert(value, Encoding::kBF16);
      b.Keep(b.CacheStore(copy.Map(target), value, copy.Map(indices)));
    });
    ASSERT_TRUE(f.plan);
    const auto lowered = Lower(f, registry);
    ASSERT_FALSE(lowered);
    EXPECT_NE(lowered.error().detail.find("selected HC:"), std::string::npos)
        << lowered.error().detail;
  }
}

TEST(GgmlOwnerProductsTest, SelectedPairChecksSuppliedBindingsAndNativeDescriptorDomain) {
  const auto registry = Registry();
  auto f = MakeSelectedHc(std::array{1U, 3U});
  PlanSelectedHc(f, registry);
  ASSERT_TRUE(f.plan);
  constexpr std::uint64_t activation = 0x200000000ULL, weights = 0x400000000ULL;
  for (int change = 0; change < 3; ++change) {
    const llmp::kernels::Addresses addresses{
        .weight =
            [=](const ir::WeightRef& ref, llmp::format::PartRole) {
              return weights + std::uint64_t{ref.role} * 65536 +
                     (change == 1 && ref.role == 2 ? 2U : 0U);
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = activation + (change == 2 ? 1U : 0U),
        .activation_bytes = f.plan->placement.extent};
    const auto result = Lower(f, registry, Device(), addresses);
    if (change) {
      EXPECT_FALSE(result);
      continue;
    }
    ASSERT_TRUE(result) << Failed(result, &kg::KernelFailure::detail).value_or("");
    std::vector<const ggml_tensor*> products;
    ASSERT_NO_FATAL_FAILURE(CheckHcProducts(f, *result, registry, true, products));
    ASSERT_EQ(products.size(), 4U);
    for (const auto* node : {products[2], products[3]}) {
      EXPECT_TRUE(kg::UsesGemvBf16Vector(node));
      auto changed = *node;
      auto input = *node->src[1];
      changed.src[1] = &input;
      input.nb[1] += 16;
      EXPECT_FALSE(kg::CheckGemmBf16(&changed));
      input = *node->src[1];
      changed.data = input.data;
      EXPECT_FALSE(kg::CheckGemmBf16(&changed));
    }
  }
  for (int change = 0; change < 3; ++change) {
    auto bad = MakeSelectedHc(std::array{3U, 3U});
    PlanSelectedHc(bad, registry);
    ASSERT_TRUE(bad.plan);
    if (change == 0) bad.shape.segments[1].slot = bad.shape.segments[0].slot;
    if (change == 1) bad.shape.segments[0].rows = 0;
    if (change == 2) {
      bad.shape.segments.clear();
      for (std::uint8_t i = 0; i < 9; ++i)
        bad.shape.segments.push_back({.slot = i, .rows = 3, .outputs = 1});
    }
    EXPECT_FALSE(Lower(bad, registry));
  }
}
TEST(GgmlOwnerProductsTest, SelectedPairSurvivesHonestRebuildAndRejectsChangedPrecision) {
  const auto registry = Registry();
  for (const bool change : {false, true}) {
    auto f = MakeSelectedHc(std::array{3U, 3U});
    PlanSelectedHc(f, registry);
    ASSERT_TRUE(f.plan);
    bool touched = false;
    Rebuild(f, registry, [&](const ir::Node& n, pl::RewriteContext& copy) {
      if (change && !touched && n.key == ir::Primitive(ir::Op::kMatMul) &&
          copy.original().tensor(n.out[0]).ne(1) == 2) {
        auto params = n.Params<ir::ProductParams>();
        params.accumulate = ir::Accumulate::kF32;
        copy.Set(n.out[0], copy.builder().MatMul(copy.Map(n.src[0]), copy.Map(n.src[1]), params));
        touched = true;
      } else
        copy.Copy(n);
    });
    ASSERT_TRUE(f.plan);
    const auto result = Lower(f, registry);
    if (change) {
      ASSERT_TRUE(touched);
      ASSERT_FALSE(result);
      EXPECT_NE(result.error().detail.find("selected HC:"), std::string::npos);
    } else
      ASSERT_TRUE(result) << Failed(result, &kg::KernelFailure::detail).value_or("");
  }
}

// These synthetic shapes test descriptor authority, not measured tuning data.
Fixture VectorFixture(std::uint32_t rows) {
  ir::Builder b;
  const auto input = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({32, rows})));
  const ir::WeightRef wr{.role = 1}, other{.role = 2};
  const auto weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({32, 16})), wr);
  const auto extra_weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({32, 24})), other);
  const auto product = b.MatMul(weight, input);
  const auto extra = b.MatMul(extra_weight, input);
  b.Output(product);
  b.Output(extra);
  co::StepGraph step{.graph = std::move(b).Finish().value(),
                     .inputs = {{.tensor = input, .kind = co::InputKind::kFeatures}},
                     .weights = {{.tensor = weight,
                                  .ref = wr,
                                  .parts = {{"vector", llmp::format::PartRole::kData}}},
                                 {.tensor = extra_weight,
                                  .ref = other,
                                  .parts = {{"ordinary", llmp::format::PartRole::kData}}}},
                     .state = {}};
  return {.step = std::move(step),
          .shape = {.segments = {{.slot = 0, .rows = rows, .outputs = rows}}},
          .weight = weight,
          .extra_weight = extra_weight,
          .input = input,
          .product = product,
          .extra = extra};
}
void PlanVector(Fixture& f, const ex::Registry& registry, bool tuned, bool aligned = true) {
  const auto rows = f.step.graph.tensor(f.input).ne(1);
  pl::DeviceFacts device{.device_class = "sm_121", .arch = 121};
  if (tuned)
    device.tuned.push_back({.implementation = kg::kMulMatVecFRows,
                            .rows = rows,
                            .k = 32,
                            .n = 16,
                            .output_encoding = Encoding::kF32});
  const std::array facts{pl::RootAlignmentFact{f.weight, 256},
                         pl::RootAlignmentFact{f.extra_weight, 256}};
  auto plan = pl::Plan(f.step.graph.Clone(), registry, device, {},
                       {.written = &kg::WrittenBytes,
                        .shape = &f.shape,
                        .root_alignment = aligned ? pl::GraphRootAlignments(f.step.graph, facts)
                                                  : pl::RootAlignment{},
                        .activation_alignment = 256});
  ASSERT_TRUE(plan) << plan.error().detail;
  f.plan.emplace(std::move(*plan));
}

TEST(GgmlBf16VectorSelectionTest, ExactTuningSelectsOnlyActualProductInBothLowerings) {
  const auto registry = Registry();
  for (const auto rows : {1U, 8U}) {
    auto f = VectorFixture(rows);
    PlanVector(f, registry, true);
    ASSERT_TRUE(f.plan);
    const auto measured = Lower(f, registry);
    ASSERT_TRUE(measured) << measured.error().detail;
    EXPECT_TRUE(measured->measuring);
    EXPECT_EQ(Count(*measured, kg::kMulMatVecFRows), 1U);
    EXPECT_EQ(Count(*measured, kg::kMulMatTensorCore), 1U);
    const llmp::kernels::Addresses addresses{
        .weight =
            [](const ir::WeightRef& ref, llmp::format::PartRole) {
              return std::uint64_t{0x1000000} + std::uint64_t{ref.role} * 65536;
            },
        .weight_rows = {},
        .state = {},
        .save = {},
        .activations = 0x2000000,
        .activation_bytes = f.plan->placement.extent};
    const auto bound = Lower(f, registry, Device(), addresses);
    ASSERT_TRUE(bound) << bound.error().detail;
    EXPECT_FALSE(bound->measuring);
    EXPECT_EQ(Count(*bound, kg::kMulMatVecFRows), 1U);
    EXPECT_EQ(Count(*bound, kg::kMulMatTensorCore), 1U);
    EXPECT_EQ(bound->extent, measured->extent);
  }
}

TEST(GgmlBf16VectorSelectionTest, MissingTuningAlignmentOrWiderPhysicalRowsKeepOrdinarySelection) {
  const auto registry = Registry();
  for (const auto rows : {8U, 12U, 16U}) {
    for (const auto tuned : {false, true}) {
      auto f = VectorFixture(rows);
      PlanVector(f, registry, tuned, false);
      ASSERT_TRUE(f.plan);
      const auto out = Lower(f, registry);
      ASSERT_TRUE(out) << out.error().detail;
      EXPECT_EQ(Count(*out, kg::kMulMatVecFRows), 0U);
    }
    auto f = VectorFixture(rows);
    PlanVector(f, registry, rows > 8);
    ASSERT_TRUE(f.plan);
    const auto out = Lower(f, registry);
    ASSERT_TRUE(out) << out.error().detail;
    EXPECT_EQ(Count(*out, kg::kMulMatVecFRows), 0U);
  }
}

TEST(GgmlBf16VectorSelectionTest, PhysicalBindingRechecksAlignmentAndAliasing) {
  const auto registry = Registry();
  auto f = VectorFixture(8);
  PlanVector(f, registry, true);
  ASSERT_TRUE(f.plan);
  const auto product = f.plan->tensors.at(f.product.v);
  const auto place = std::ranges::find(f.plan->placement.offsets, product,
                                       &std::pair<ir::TensorId, std::uint64_t>::first);
  ASSERT_NE(place, f.plan->placement.offsets.end());
  const auto overlap = std::uint64_t{0x2000000} + place->second;
  for (const bool misaligned : {false, true}) {
    const llmp::kernels::Addresses addresses{.weight =
                                                 [=](const ir::WeightRef&, llmp::format::PartRole) {
                                                   return misaligned ? std::uint64_t{0x1000002}
                                                                     : overlap;
                                                 },
                                             .weight_rows = {},
                                             .state = {},
                                             .save = {},
                                             .activations = 0x2000000,
                                             .activation_bytes = f.plan->placement.extent};
    EXPECT_FALSE(Lower(f, registry, Device(), addresses));
  }
}

TEST(GgmlBf16VectorSelectionTest, DescriptorAndTypedPinRefuseUnsupportedDomains) {
  const auto* capability = kg::CapabilityOf(kg::kMulMatVecFRows);
  ASSERT_NE(capability, nullptr);
  EXPECT_TRUE(capability->needs_tuning);
  for (int change = 0; change < 16; ++change) {
    SCOPED_TRACE(change);
    ir::ProductParams params;
    std::array operands{
        ex::OperandFormat{.operand = 0,
                          .encoding = Encoding::kBF16,
                          .ne = {32, 16, 1, 1},
                          .memory = ex::OperandMemory{{2, 64, 1024, 1024}, 0, 256}},
        ex::OperandFormat{.operand = 1,
                          .encoding = Encoding::kF32,
                          .ne = {32, 8, 1, 1},
                          .memory = ex::OperandMemory{{4, 128, 1024, 1024}, 0, 256}},
        ex::OperandFormat{.operand = ex::kOutputOperand,
                          .encoding = Encoding::kF32,
                          .ne = {16, 8, 1, 1},
                          .memory = ex::OperandMemory{{4, 64, 512, 512}, 0, 256}}};
    std::array tuning{ex::TunedShape{.implementation = kg::kMulMatVecFRows,
                                     .rows = 8,
                                     .k = 32,
                                     .n = 16,
                                     .output_encoding = Encoding::kF32}};
    switch (change) {
      case 1:
        operands[0].encoding = Encoding::kF16;
        break;
      case 2:
        operands[1].encoding = Encoding::kBF16;
        break;
      case 3:
        operands[2].encoding = Encoding::kBF16;
        break;
      case 4:
        operands[0].memory->nb[1] += 4;
        break;
      case 5:
        operands[1].memory->view_offset = 4;
        break;
      case 6:
        operands[0].ne[2] = 2;
        break;
      case 7:
        params.reserved[0] = 1;
        break;
      case 8:
        params.accumulate = ir::Accumulate::kF32;
        break;
      case 9:
        tuning[0].output_encoding.reset();
        break;
      case 10:
        tuning[0].n = 24;
        break;
      case 11:
        operands[0].memory.reset();
        break;
      default:
        break;
    }
    // Real graph queries carry the fixed canonical Node::params array;
    // hypothetical descriptor queries may carry just the declared struct.
    std::array<std::byte, ir::kMaxParamBytes + 1> storage{};
    const auto declared = std::as_bytes(std::span(&params, 1));
    std::ranges::copy(declared, storage.begin());
    if (change == 13) storage[ir::kMaxParamBytes - 1] = std::byte{1};
    const auto count = change == 14   ? sizeof(params) - 1
                       : change == 15 ? storage.size()
                       : change >= 12 ? ir::kMaxParamBytes
                                      : sizeof(params);
    const ex::CapabilityQuery query{.op = ir::Primitive(ir::Op::kMatMul),
                                    .operands = operands,
                                    .rows = 8,
                                    .k = 32,
                                    .n = 16,
                                    .arch = 121,
                                    .tuned = tuning,
                                    .params = std::span(storage).first(count)};
    EXPECT_EQ(capability->accepts(query), change == 0 || change == 12);
  }
}

}  // namespace
