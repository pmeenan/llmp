// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML lowering (kernels/ggml/lower.h) of Gemma 4's steps (26B-A4B, 31B
// and their assistants), structurally, on the CPU (every profile builds the
// planner and the lowering), under each serving recipe the gemma4 mapping's
// policy gives: within the qualified envelope (a context of at most 8,192 and
// four owners: every row to the head, the fused norm-mul, norm-rope and
// norm-add, the 26B-A4B's fused routing and scaled reduce, the 31B's
// quantized GeGLU, owner attention over joined one-row owners of equal
// reads, and at 4,096 and two owners bounded roots over common reads), and
// outside it (the frontier head's narrowed final layer and the fused
// norm-mul alone); with device masks and host masks. For every shape the
// engine measures at setup, the step planned under the mapping's policy for
// that envelope and lowered has leaves that each bind a weight resource, a
// layer's, owner's or tensor's cache or a step input; its plan's steps cover
// its nodes; its owners' outputs and staged inputs are the shape's; and the
// policy's patterns apply (owner attention within the envelope and nowhere
// outside it, the 26B-A4B's routing and reduce as the fused launches, every
// other form on its primitives). The assistant's draft steps lower with host
// masks (its frozen reads' masks) and device masks (the frozen producer).
// (These graphs were once compared node for node with the M3 graph
// builders, retired with their runners after the migration's gate.)

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <optional>
#include <print>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "components/step.h"
#include "engine/support.h"
#include "execution/registry.h"
#include "families/gemma.h"
#include "format/encoding.h"
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/tensors.h"
#include "model/state_contract.h"
#include "planner/pattern.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma4.h"
#include "spec_fixtures.h"
#include "state_layouts.h"

namespace {

namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace fam = llmp::test_support::families;
namespace ex = llmp::execution;
namespace pl = llmp::planner;
namespace co = llmp::components;
namespace ir = llmp::ir;

constexpr std::uint64_t kExtent = 2ULL << 20U;
constexpr std::uint32_t kGb10 = 121;

ex::Registry PlanningRegistry() {
  std::vector<ex::Implementation> v;
  for (auto d : kg::PlanningImplementations()) {
    v.push_back(std::move(d));
  }
  return std::move(ex::Registry::Create(std::move(v)).value());
}

const pl::DeviceFacts& Gb10() {
  static const pl::DeviceFacts facts{.device_class = "sm_121", .arch = kGb10};
  return facts;
}

std::optional<llmp::format::Encoding> EncodingOf(ggml_type type) {
  for (const auto& info : llmp::format::Encodings()) {
    if (info.ggml && info.ggml->type == static_cast<std::uint32_t>(type)) return info.id;
  }
  return std::nullopt;
}

// The GB10's routing as both sides take it (ggml_lower_test's model): MMVQ
// where upstream's ggml_cuda_should_use_mmvq takes a product's columns (an
// expert product's tokens), MMQ otherwise; cuBLAS for float products.
kg::DeviceChoices Device() {
  kg::DeviceChoices d;
  d.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  d.quant = [](const ggml_tensor* n) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
    const auto e = EncodingOf(n->src[0]->type);
    const auto columns =
        static_cast<std::uint64_t>(n->op == GGML_OP_MUL_MAT_ID ? n->ne[2] : n->src[1]->ne[1]);
    return e && kg::MmvqRouted(*e, kGb10, columns) ? kg::QuantMulMatPath::kVector
                                                   : kg::QuantMulMatPath::kTile;
  };
  d.quant_glu_fusible = [](const ggml_tensor*, const ggml_tensor* up, const ggml_tensor*) {
    const auto e = EncodingOf(up->src[0]->type);
    return up->ne[1] == 1 && e && kg::MmvqRouted(*e, kGb10, 1);
  };
  return d;
}

// ------------------------------------------------------------ the recipes

// Serving envelopes spanning bounded, joined and owners-alone scheduling.
struct Recipe {
  std::string name;
  std::uint32_t context = 0;
  std::uint32_t slots = 0;
  std::uint32_t max_rows = 0;
  bool bounded = false;
};

std::vector<Recipe> Recipes(std::uint32_t size) {
  const std::uint32_t rows = size == 26 ? 1024 : 256;  // the envelope's chunks
  return {
      {.name = "bounded", .context = 4096, .slots = 2, .max_rows = rows, .bounded = true},
      {.name = "envelope", .context = 8192, .slots = 4, .max_rows = rows},
      {.name = "outside", .context = 16384, .slots = 2, .max_rows = 128},
  };
}

// ------------------------------------------------------------ the shapes

enum class Out : std::uint8_t { kHead, kAll, kStateOnly, kGreedy };
struct Case {
  pl::StepShape ir;
  bool state_only = false;
  std::string Name() const {
    std::string s;
    for (const auto& seg : ir.segments) {
      s += std::format("[{}:{}@{} {}]", seg.slot, seg.rows, seg.past, seg.read_cells);
    }
    return s + std::format(" outputs {}{}{}", ir.outputs(), state_only ? " state" : "",
                           !ir.segments.empty() && ir.segments.front().greedy ? " greedy" : "");
  }
};

std::uint64_t PadUp(std::uint64_t value, std::uint64_t to) { return (value + to - 1) / to * to; }

// The engine's setup measurement loop at the recipe's options: the owners'
// rows (the chunk budgets' splits), the pasts at either end of the context
// (and, within the bounded envelope, mixed ends of two owners), and each
// head the step may return. A step's read is its end rounded up to 256
// cells within the global cells.
std::vector<Case> SetupCases(const fam::Gemma4Profile& p, const Recipe& r) {
  std::vector<Case> out;
  const std::uint32_t global_cells = static_cast<std::uint32_t>(PadUp(r.context, 256));
  const std::uint32_t local_cells = static_cast<std::uint32_t>(
      PadUp(std::min<std::uint64_t>(r.context, p.window + r.max_rows), 256));
  const std::uint32_t head_rows = r.slots;
  for (const auto row_budget : {r.max_rows, head_rows}) {
    for (std::uint32_t count = 1; count <= r.slots; ++count) {
      for (const auto& owner_rows :
           llmp::engine::support::ChunkMeasurementRows(row_budget, row_budget, count, false)) {
        const bool mixed = r.bounded && count == 2 &&
                           std::ranges::all_of(owner_rows, [](auto n) { return n == 1; });
        for (const std::uint32_t endpoint : {0U, 1U, 2U, 3U}) {
          if (endpoint >= 2 && !mixed) continue;
          std::vector<std::array<std::uint32_t, 3>> segments;  // slot, rows, past
          for (std::uint32_t i = 0; i < count; ++i) {
            const auto rows = owner_rows[i];
            const bool high =
                endpoint == 1 || (endpoint == 2 && i == 1) || (endpoint == 3 && i == 0);
            auto past = high ? r.context - rows : 0U;
            if (high && endpoint >= 2) {
              constexpr std::uint64_t operand_limit = 64ULL << 20U;
              const auto local_limit =
                  operand_limit / (std::uint64_t{p.local_head_dim} * p.local_kv_heads * 2);
              auto common_limit = std::min(
                  std::uint64_t{16384},
                  operand_limit / (std::uint64_t{p.global_head_dim} * p.global_kv_heads * 2));
              if (local_cells > local_limit) common_limit = std::min(common_limit, local_limit);
              past = std::min(past, static_cast<std::uint32_t>(common_limit - 1));
            }
            segments.push_back({i, rows, past});
          }
          for (const Out o : {Out::kHead, Out::kAll, Out::kStateOnly, Out::kGreedy}) {
            const bool all = o == Out::kAll;
            if (all ? row_budget != head_rows : row_budget != r.max_rows) continue;
            Case c;
            c.state_only = o == Out::kStateOnly;
            for (const auto& [slot, rows, past] : segments) {
              c.ir.segments.push_back(
                  {.slot = static_cast<std::uint8_t>(slot),
                   .rows = rows,
                   .past = past,
                   .read_cells = static_cast<std::uint32_t>(
                       std::min<std::uint64_t>(global_cells, PadUp(past + rows, 256))),
                   .outputs = o == Out::kStateOnly ? 0U
                              : all                ? rows
                                                   : 1U,
                   .want_head = o != Out::kStateOnly,
                   .greedy = o == Out::kGreedy});
            }
            out.push_back(std::move(c));
          }
        }
      }
    }
    if (head_rows == r.max_rows) break;
  }
  return out;
}

// ------------------------------------------------------------ the lowering

struct Lowered {
  std::optional<co::StepGraph> step;
  std::optional<pl::StepPlan> plan;
  std::optional<kg::LoweredStep> lowered;
  std::map<const ggml_tensor*, std::string> leaves;
  std::size_t owners = 0, routes = 0;
};

// The step's leaves by what they bind, as the builders' are named.
std::map<std::uint32_t, std::string> StepLeaves(const co::StepGraph& step,
                                                const md::StateLayoutDerived* target) {
  std::map<std::uint32_t, std::string> out;
  const auto where = [](const co::StepInput& in) {
    return std::format("{}.{}", in.segment, in.cells.ring ? "local" : "global");
  };
  std::map<std::uint32_t, const co::StepInput*> inputs;
  for (const co::StepInput& in : step.inputs) {
    inputs[in.tensor.v] = &in;
    switch (in.kind) {
      case co::InputKind::kTokens:
        out[in.tensor.v] = "tokens";
        break;
      case co::InputKind::kPositions:
        out[in.tensor.v] = "positions";
        break;
      case co::InputKind::kOutputRows:
        out[in.tensor.v] = "out_ids";
        break;
      case co::InputKind::kStoreCells:
        out[in.tensor.v] = "cells." + where(in);
        break;
      case co::InputKind::kFeatures:
        out[in.tensor.v] = "features";
        break;
      default:
        break;
    }
  }
  for (const co::StepState& st : step.state) {
    if (st.borrowed && target != nullptr) {
      // The target's cache it reads: its name, its kind and the owner.
      const auto& item = target->items.at(st.item);
      out[st.tensor.v] =
          std::format("{}.{}.{}", item.contract.name,
                      item.contract.window != 0 ? "local" : "global", st.ref.segment);
      continue;
    }
    out[st.tensor.v] =
        std::format("{}.{}.{}", st.ref.tensor == 0 ? "k" : "v", st.ref.layer, st.ref.segment);
  }
  for (const co::StepWeight& w : step.weights) out[w.tensor.v] = "w." + w.parts.front().first;
  for (const ir::Node& n : step.graph.nodes()) {
    if (n.key.op == ir::Op::kMaskBuild) {
      const auto held = inputs.find(n.src[1].v);
      if (held != inputs.end()) out[n.out[0].v] = "mask." + where(*held->second);
    }
  }
  return out;
}

struct Ir {
  ex::Registry registry = PlanningRegistry();
  llmp::spec::ModelSpec spec;
  std::vector<llmp::spec::Source> sources;  // by artifact
  std::vector<std::span<const co::BoundTensor>> tables;
  std::vector<md::StateLayoutDerived> layouts;  // by stage
  std::vector<const md::StateLayoutDerived*> stages;
  pl::Policy policy;
  std::uint32_t context = 0;
};

Lowered Lower(Ir& ir_, std::uint16_t stage, const pl::StepShape& shape, bool device_masks) {
  Lowered out;
  const llmp::spec::Stage& s = ir_.spec.stages.at(stage);
  auto step = co::BuildStep(ir_.spec, stage, shape,
                            {.tensors = ir_.tables.at(s.artifact),
                             .state = &ir_.layouts.at(stage),
                             .stages = ir_.stages,
                             .artifacts = ir_.tables},
                            {.narrow = !pl::Disabled(ir_.policy, "narrow_final")});
  EXPECT_TRUE(step.has_value()) << step.error();
  if (!step) return out;
  std::vector<ir::TensorId> staged;
  for (const co::StepInput& in : step->inputs) {
    if (in.kind != co::InputKind::kCellPositions) staged.push_back(in.tensor);
  }
  auto plan = pl::Plan(
      step->graph.Clone(), ir_.registry, Gb10(), ir_.policy,
      {.capturable = true, .staged = staged, .written = &kg::WrittenBytes, .shape = &shape});
  EXPECT_TRUE(plan.has_value()) << (plan ? "" : plan.error().detail);
  if (!plan) return out;
  out.owners = static_cast<std::size_t>(
      std::ranges::count(plan->applied, "owner_attention", &pl::Applied::pattern));
  for (const ir::Node& n : plan->graph.nodes()) {
    out.routes += n.key == ir::Primitive(ir::Op::kRoute) ? 1 : 0;
  }
  out.step.emplace(std::move(*step));
  out.plan.emplace(std::move(*plan));
  const kn::Addresses addresses{};
  const kn::LowerInput input{*out.plan, *out.step, shape, addresses, ir_.context};
  auto lowered = kg::LowerStep(input, ir_.registry, Device(),
                               {.device_masks = device_masks, .frozen_host_masks = true});
  EXPECT_TRUE(lowered.has_value()) << (lowered ? "" : lowered.error().detail);
  if (!lowered) return out;
  const auto names = StepLeaves(*out.step, stage != 0 ? &ir_.layouts.at(0) : nullptr);
  for (const kg::LoweredLeaf& leaf : lowered->leaves) {
    const auto at = names.find(leaf.primitive.v);
    out.leaves[leaf.tensor] = at == names.end() ? "?" : at->second;
  }
  out.lowered.emplace(std::move(*lowered));
  return out;
}

// Every leaf binds something the step names (a weight resource, a cache, an
// input), every plan step's nodes are the graph's own, and the plan covers
// every node that launches something.
void ExpectWellFormed(const Lowered& w) {
  const auto& nodes = w.lowered->nodes;
  std::set<const ggml_tensor*> mine(nodes.begin(), nodes.end());
  EXPECT_EQ(mine.size(), nodes.size());
  for (const auto& [tensor, name] : w.leaves) {
    EXPECT_NE(name, "?") << ggml_get_name(tensor);
    EXPECT_FALSE(mine.contains(tensor)) << name;
  }
  std::set<const ggml_tensor*> planned;
  for (const kg::PlanStep& step : w.lowered->plan.steps) {
    EXPECT_FALSE(step.nodes.empty());
    for (const ggml_tensor* node : step.nodes) {
      EXPECT_TRUE(mine.contains(node));
      planned.insert(node);
    }
  }
  for (const ggml_tensor* node : nodes) {
    if (!kg::LaunchesNothing(node)) EXPECT_TRUE(planned.contains(node)) << ggml_op_desc(node);
  }
}

// The inputs the step stages: each one a named leaf, a mask only in the
// host-mask build, of its rows and cells.
void ExpectInputs(const Lowered& w, bool device_masks) {
  std::map<std::uint64_t, const ggml_tensor*> by_address;
  for (const auto& [t, name] : w.leaves) {
    by_address[static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data))] = t;
  }
  for (const kn::InputSlot& slot : w.lowered->inputs) {
    const auto found = by_address.find(slot.device_address);
    ASSERT_NE(found, by_address.end());
    EXPECT_NE(w.leaves.at(found->second), "?");
    if (slot.kind == kn::InputKind::kMask) {
      EXPECT_FALSE(device_masks);
      EXPECT_EQ(slot.bytes, slot.read * slot.mask_rows * 2);
    }
  }
}

Ir TargetIr(std::uint32_t size, const Recipe& r) {
  const auto& p = size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B();
  const auto values = fam::Gemma4Values(p);
  Ir ir_;
  ir_.sources.push_back(llmp::test_support::Gemma4Source(size));
  ir_.spec =
      *llmp::spec::Synthesize(llmp::spec::mappings::Gemma4Mapping(), ir_.sources[0], *values);
  ir_.tables.emplace_back(ir_.sources[0].tensors);
  auto layout = co::StageLayoutUniform(ir_.spec, 0, r.context, r.max_rows, kExtent);
  EXPECT_TRUE(layout.has_value()) << layout.error();
  ir_.layouts.push_back(std::move(*layout));
  ir_.stages.push_back(&ir_.layouts[0]);
  const auto text = llmp::spec::DefaultPolicy(
      llmp::spec::mappings::Gemma4Mapping(), *values, {.context = r.context, .slots = r.slots},
      {.architecture = ir_.sources[0].architecture, .container = ir_.sources[0].container});
  EXPECT_TRUE(text.has_value());
  ir_.policy = pl::ParsePolicy(*text).value();
  ir_.context = r.context;
  return ir_;
}

class GgmlLowerGemma4 : public testing::TestWithParam<std::uint32_t> {};

TEST_P(GgmlLowerGemma4, ServingRecipesLowerEverySetupShape) {
  const std::uint32_t size = GetParam();
  const auto& p = size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B();
  std::size_t shapes = 0, owner_forms = 0, nodes = 0, routes = 0;
  for (const Recipe& r : Recipes(size)) {
    Ir ir_ = TargetIr(size, r);
    ASSERT_EQ(ir_.layouts[0].context, r.context);
    // Structural eligibility is common to all serving envelopes; concrete
    // owner/row geometry determines the implementations below.
    EXPECT_FALSE(pl::Disabled(ir_.policy, "narrow_final")) << r.name;
    EXPECT_FALSE(pl::Disabled(ir_.policy, "owner_attention")) << r.name;
    EXPECT_TRUE(pl::Disabled(ir_.policy, "owner_padded_reads")) << r.name;
    for (const Case& c : SetupCases(p, r)) {
      for (const bool device_masks : {true, false}) {
        SCOPED_TRACE(std::format("{} {}{}", r.name, c.Name(), device_masks ? "" : ", host masks"));
        const Lowered w = Lower(ir_, 0, c.ir, device_masks);
        ASSERT_TRUE(w.lowered.has_value());
        ExpectWellFormed(w);
        if (::testing::Test::HasFatalFailure()) return;
        ExpectInputs(w, device_masks);
        if (::testing::Test::HasFatalFailure()) return;
        // Gemma 4's native owner-attention domain is one returned row
        // per owner, independent of the mapping's scheduling envelope.
        const bool decode = std::ranges::all_of(c.ir.segments, [](const auto& segment) {
          return segment.rows == 1 && segment.outputs == 1;
        });
        EXPECT_EQ(w.owners, c.ir.segments.size() > 1 && decode ? 1U : 0U);
        // Each owner's own outputs.
        std::uint32_t outputs = 0;
        for (const kn::OutputSlot& o : w.lowered->outputs) {
          EXPECT_LT(o.segment, c.ir.segments.size());
          ++outputs;
        }
        EXPECT_EQ(outputs, c.state_only ? 0U : c.ir.segments.size());
        // The 26B-A4B's routing and scaled reduce, one pair a layer, as
        // their fused launches wherever the graph matches.
        const auto fused = [&](std::string_view name) {
          return std::ranges::count_if(w.lowered->plan.steps, [&](const kg::PlanStep& step) {
            return step.implementation == name;
          });
        };
        // A state-only step has no use for the last layer's feed-forward
        // (nothing reads its rows), so it is not run.
        const std::ptrdiff_t pairs = size == 26 ? p.layers - (c.state_only ? 1U : 0U) : 0;
        EXPECT_EQ(fused(kg::kGemmaRouteName), pairs);
        EXPECT_EQ(fused(kg::kGemmaReduceName), pairs);
        owner_forms += w.owners != 0 ? 1 : 0;
        routes += w.routes;
        nodes += w.lowered->nodes.size();
        ++shapes;
      }
    }
  }
  std::println("gemma{}: {} lowered steps ({} with owner attention), {} nodes, {} fused routings",
               size, shapes, owner_forms, nodes, routes);
  EXPECT_GT(shapes, 0U);
  EXPECT_GT(owner_forms, 0U);
  EXPECT_EQ(routes != 0, size == 26);
}

INSTANTIATE_TEST_SUITE_P(Gemma4, GgmlLowerGemma4, testing::Values(26U, 31U),
                         [](const testing::TestParamInfo<std::uint32_t>& info) {
                           return std::format("gemma{}", info.param);
                         });

// Owner attention's Gemma 4 forms (16 or 32 heads, D 512) are the GB10's
// alone, other devices keeping independent attention;
// Gemma 2 and 3's (D 256, 8 heads) wherever the build has code.
TEST(GgmlLowerGemma4Test, OwnerAttentionOfGemma4FormsOnTheGb10Only) {
  const ex::Capability* owners = kg::CapabilityOf(kg::kFlashAttnOwnersName);
  ASSERT_NE(owners, nullptr);
  ASSERT_NE(owners->accepts, nullptr);
  const ir::OwnerAttnParams params{.attn = {}, .owners = 2, .cohort = 2, .offset = 0};
  using llmp::format::Encoding;
  using llmp::format::Layout;
  for (const auto [d, heads] : {std::pair<std::uint64_t, std::uint64_t>{256, 8},
                                {256, 16},
                                {512, 16},
                                {256, 32},
                                {512, 32}}) {
    SCOPED_TRACE(std::format("D {} heads {}", d, heads));
    const std::uint64_t cells = 1024;
    const std::uint64_t kv = heads / (d == 256 ? 2 : 8);
    std::vector<ex::OperandFormat> operands{{.operand = 0,
                                             .encoding = Encoding::kF32,
                                             .layout = Layout::kDense,
                                             .ne = {d, 1, heads, 2}},
                                            {.operand = 1,
                                             .encoding = Encoding::kF16,
                                             .layout = Layout::kDense,
                                             .ne = {cells, 32, 1, 2}}};
    for (std::uint8_t o = 2; o < 6; ++o) {
      operands.push_back({.operand = o,
                          .encoding = Encoding::kF16,
                          .layout = Layout::kDense,
                          .ne = {d, cells, kv, 1}});
    }
    ex::CapabilityQuery q{.op = owners->op,
                          .operands = operands,
                          .rows = 1,
                          .k = d,
                          .n = d,
                          .arch = kGb10,
                          .params = std::as_bytes(std::span(&params, 1))};
    EXPECT_TRUE(owners->accepts(q));
    q.arch = 86;
    EXPECT_EQ(owners->accepts(q), kg::CompiledFor(86));
  }
}

// The assistant's draft passes: one query an owner at its target's
// committed endpoint, over the target's last local and global caches,
// with host masks of its frozen reads (inputs, each frozen); with device
// masks, the frozen producer.
TEST(GgmlLowerGemma4Test, AssistantLowersItsDraftPasses) {
  constexpr std::uint32_t kContext = 8192;
  constexpr std::uint32_t kMaxRows = 64;
  std::size_t compared = 0;
  for (const std::uint32_t size : {26U, 31U}) {
    SCOPED_TRACE(size);
    const auto& p = size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B();
    const auto& a = size == 26 ? fam::Gemma4Assistant26() : fam::Gemma4Assistant31();
    const auto values = fam::Gemma4AssistantValues(a, p);
    ASSERT_TRUE(values.has_value());
    const auto target_values = fam::Gemma4Values(p);
    Ir ir_;
    ir_.sources.push_back(llmp::test_support::Gemma4Source(size));
    ir_.sources.push_back(llmp::test_support::Gemma4AssistantSource(size));
    const auto target = *llmp::spec::Synthesize(llmp::spec::mappings::Gemma4Mapping(),
                                                ir_.sources[0], *target_values);
    auto composed = llmp::spec::ComposeDrafter(target, llmp::spec::mappings::Gemma4Mapping(),
                                               llmp::spec::mappings::Gemma4AssistantMapping(),
                                               ir_.sources[1], *values);
    ASSERT_TRUE(composed.has_value()) << composed.error();
    ir_.spec = std::move(*composed);
    for (const auto& s : ir_.sources) ir_.tables.emplace_back(s.tensors);
    for (std::uint16_t stage = 0; stage < 2; ++stage) {
      auto layout = co::StageLayoutUniform(ir_.spec, stage, kContext, kMaxRows, kExtent);
      ASSERT_TRUE(layout.has_value()) << layout.error();
      ir_.layouts.push_back(std::move(*layout));
    }
    for (const auto& l : ir_.layouts) ir_.stages.push_back(&l);
    const auto text =
        llmp::spec::DefaultPolicy(llmp::spec::mappings::Gemma4AssistantMapping(), *values);
    ASSERT_TRUE(text.has_value());
    ir_.policy = pl::ParsePolicy(*text).value();
    ir_.context = kContext;
    for (const auto& prefixes : std::vector<std::vector<std::uint32_t>>{
             {300}, {300, 1500}, {7, 2048, 6000}, {1, 256, 1300, 8191}}) {
      SCOPED_TRACE(prefixes.size());
      pl::StepShape shape{.stage = 1, .phase = pl::Phase::kDraft, .segments = {}};
      for (std::size_t i = 0; i < prefixes.size(); ++i) {
        const std::uint32_t read = std::min(kContext, (prefixes[i] + 255) / 256 * 256);
        shape.segments.push_back({.slot = static_cast<std::uint8_t>(2 * i),
                                  .rows = 1,
                                  .past = prefixes[i],
                                  .read_cells = read,
                                  .outputs = 1});
      }
      const Lowered w = Lower(ir_, 1, shape, false);
      ASSERT_TRUE(w.lowered.has_value());
      ExpectWellFormed(w);
      if (::testing::Test::HasFatalFailure()) return;
      ExpectInputs(w, false);
      if (::testing::Test::HasFatalFailure()) return;
      std::size_t masks = 0;
      for (const kn::InputSlot& slot : w.lowered->inputs) {
        if (slot.kind == kn::InputKind::kMask) {
          EXPECT_TRUE(slot.cells.frozen);
          ++masks;
        }
      }
      // A local and a global mask an owner.
      EXPECT_EQ(masks, 2 * prefixes.size());
      // Device masks: the frozen producer in each mask's place.
      const Lowered d = Lower(ir_, 1, shape, true);
      ASSERT_TRUE(d.lowered.has_value());
      std::size_t producers = 0;
      for (const ggml_tensor* n : d.lowered->nodes) {
        if (kg::LlmpOpOf(n) == kg::LlmpOp::kCausalMask) {
          EXPECT_EQ(kg::LlmpOpInt(n, 6), static_cast<std::int32_t>(kg::MaskPolicy::kFrozen));
          ++producers;
        }
      }
      EXPECT_EQ(producers, 2 * prefixes.size());
      ++compared;
    }
  }
  EXPECT_EQ(compared, 8U);
}

// A target's step with outputs writes its drafter's tap: the final norm's
// output rows stored at their positions' cells, the cells an input of the
// segment's last rows' store cells.
TEST(GgmlLowerGemma4Test, TargetStepsLowerTheirDraftersTaps) {
  constexpr std::uint32_t kContext = 4096;
  constexpr std::uint32_t kMaxRows = 64;
  const auto& p = fam::Gemma4_26BA4B();
  const auto& a = fam::Gemma4Assistant26();
  Ir ir_;
  ir_.sources.push_back(llmp::test_support::Gemma4Source(26));
  ir_.sources.push_back(llmp::test_support::Gemma4AssistantSource(26));
  const auto target_values = fam::Gemma4Values(p);
  const auto values = fam::Gemma4AssistantValues(a, p);
  ASSERT_TRUE(target_values.has_value() && values.has_value());
  const auto target = *llmp::spec::Synthesize(llmp::spec::mappings::Gemma4Mapping(), ir_.sources[0],
                                              *target_values);
  auto composed = llmp::spec::ComposeDrafter(target, llmp::spec::mappings::Gemma4Mapping(),
                                             llmp::spec::mappings::Gemma4AssistantMapping(),
                                             ir_.sources[1], *values);
  ASSERT_TRUE(composed.has_value()) << composed.error();
  ir_.spec = std::move(*composed);
  for (const auto& s : ir_.sources) ir_.tables.emplace_back(s.tensors);
  for (std::uint16_t stage = 0; stage < 2; ++stage) {
    auto layout = co::StageLayoutUniform(ir_.spec, stage, kContext, kMaxRows, kExtent);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    ir_.layouts.push_back(std::move(*layout));
  }
  for (const auto& l : ir_.layouts) ir_.stages.push_back(&l);
  ir_.policy =
      pl::ParsePolicy(*llmp::spec::DefaultPolicy(llmp::spec::mappings::Gemma4Mapping(),
                                                 *target_values, {.context = kContext, .slots = 2},
                                                 {.architecture = ir_.sources[0].architecture,
                                                  .container = ir_.sources[0].container}))
          .value();
  ir_.context = kContext;
  const pl::StepShape shape{
      .segments = {{.slot = 0, .rows = 7, .past = 0, .read_cells = 256, .outputs = 1},
                   {.slot = 1, .rows = 3, .past = 300, .read_cells = 512, .outputs = 3}}};
  const Lowered w = Lower(ir_, 0, shape, true);
  ASSERT_TRUE(w.lowered.has_value());
  std::vector<std::uint32_t> taps;
  for (const kn::InputSlot& slot : w.lowered->inputs) {
    if (slot.kind == kn::InputKind::kStoreCells && slot.outputs != 0) {
      EXPECT_EQ(slot.bytes, std::uint64_t{slot.outputs} * sizeof(std::int64_t));
      EXPECT_TRUE(slot.cells.ring);
      taps.push_back(slot.outputs);
    }
  }
  EXPECT_EQ(taps, (std::vector<std::uint32_t>{1, 3}));
}

}  // namespace
