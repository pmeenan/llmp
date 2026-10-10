// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML lowering (kernels/ggml/lower.h) against today's graph builders,
// structurally, on the CPU (every profile builds the descriptor-only
// builders, PlanGraph and the lowering). For every Gemma 2 and Gemma 3
// shape the runners measure at setup at the serving defaults (Gemma 2 at
// 8,192 with two owners; Gemma 3 at 4,096 with two and 8,448 with one),
// and Gemma 3's joined decode of 2 to 12 owners with every count of long
// owners (its runner's four-root carriers), with device masks and with host
// masks, the step planned under the architecture mapping's policy and
// lowered is BuildGemma2Graph's or BuildGemma3Graph's graph under the
// serving recipe (runtime/serving.cc: owner decode with bounded roots,
// owner and flexible owner prefill, the narrowed final layer of
// frontier_head, shared Q8 preparation, the fused norm-mul, norm-add and
// quantized GeGLU, and for Gemma 3 the fused norm-rope and grouped KV
// stores), node for node: the operation, type, extents, strides, view
// offset, operation parameters and flags, every source (a computed one by
// its position, a leaf by what it binds: the same weight resource, layer's,
// owner's and tensor's cache, or step input), and PlanGraph's steps over
// it are the builders' (the same implementations over the same nodes). The
// planner selects over the GGML module's own capabilities (capabilities.h),
// on a model of the GB10's product routing that is upstream's
// (MmvqRouted) on both sides.

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
#include "format/encoding.h"
#include "format/layout.h"
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/gemma2_graph.h"
#include "kernels/ggml/gemma3_graph.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/tensors.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec_fixtures.h"

namespace {

namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace ex = llmp::execution;
namespace pl = llmp::planner;
namespace co = llmp::components;
namespace ir = llmp::ir;

constexpr std::uint64_t kExtent = 2ULL << 20U;
constexpr std::uint32_t kGb10 = 121;

// The GGML module's capabilities as a registry (placeholder identities: a
// CPU build has no kernels to bind).
ex::Registry PlanningRegistry() {
  std::vector<ex::Implementation> v;
  for (const kg::CapabilityDeclaration& d : kg::CapabilityDeclarations()) {
    v.push_back({.name = std::string(d.name),
                 .operation = d.operation,
                 .source = "ggml",
                 .revision = "planning",
                 .build = "cpu",
                 .variant = std::string(kg::AcceptsTextOf(d.name)),
                 .capability = *d.capability});
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

// The device's own choices on the GB10, as both sides take them: upstream's
// product routing (MMVQ where ggml_cuda_should_use_mmvq takes it, MMQ
// otherwise; MulMatVecQGluFusible's one column), no cuBLAS product.
kg::DeviceChoices Device() {
  kg::DeviceChoices d;
  d.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  d.quant = [](const ggml_tensor* n) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
    const auto e = EncodingOf(n->src[0]->type);
    return e && kg::MmvqRouted(*e, kGb10, static_cast<std::uint64_t>(n->src[1]->ne[1]))
               ? kg::QuantMulMatPath::kVector
               : kg::QuantMulMatPath::kTile;
  };
  d.quant_glu_fusible = [](const ggml_tensor*, const ggml_tensor* up, const ggml_tensor*) {
    const auto e = EncodingOf(up->src[0]->type);
    return up->ne[1] == 1 && e && kg::MmvqRouted(*e, kGb10, 1);
  };
  d.dense_mmvq_shape = [](ggml_type type, std::int64_t columns) {
    const auto e = EncodingOf(type);
    return columns >= 1 && columns <= 8 && e &&
           kg::MmvqRouted(*e, kGb10, static_cast<std::uint64_t>(columns));
  };
  return d;
}

// The recipes (runtime/serving.cc): as the mappings' policies for the
// planner, as flags for PlanGraph over the builders' graphs.

kg::DeviceChoices Recipe(bool gemma3) {
  kg::DeviceChoices d = Device();
  d.fuse_norms = true;
  d.fuse_quant_glu = true;
  d.fuse_norm_add = true;
  d.fuse_norm_rope = gemma3;
  d.group_set_rows = gemma3;
  return d;
}

struct Gemma2Family {
  using Shape = kg::Gemma2ChunkShape;
  using Options = kg::Gemma2GraphOptions;
  static constexpr bool kGemma3 = false;
  static std::string_view Policy() {
    return llmp::spec::mappings::Gemma2Mapping().policies.front().text;
  }
  static const md::Gemma2Profile& P() { return md::Gemma2_2B(); }
  static auto Resources() { return llmp::test_support::gemma2::Resources(); }
  static auto Binding() { return md::BindGemma2(P(), "gemma2", Resources()).value(); }
  static auto State(std::uint32_t context, std::uint32_t rows) {
    return md::Gemma2State(P(), context, rows).value();
  }
  static llmp::spec::ModelSpec Spec() {
    auto values = llmp::spec::mappings::Gemma2V0Values(P());
    return *llmp::spec::Synthesize(llmp::spec::mappings::Gemma2Mapping(),
                                   llmp::test_support::Gemma2Source(), *values);
  }
  static llmp::spec::Source Source() { return llmp::test_support::Gemma2Source(); }
  static std::size_t Tensors(std::size_t segments) { return kg::Gemma2GraphTensors(P(), segments); }
  template <typename... A>
  static auto Build(A&&... a) {
    return kg::BuildGemma2Graph(std::forward<A>(a)...);
  }
  static constexpr auto kStateOnly = kg::Gemma2OutputMode::kStateOnly;
  static constexpr auto kHead = kg::Gemma2OutputMode::kHead;
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> Contexts() {
    return {{md::kGemma2Context, 2}};
  }
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> Cohorts() { return {}; }
};

struct Gemma3Family {
  using Shape = kg::Gemma3ChunkShape;
  using Options = kg::Gemma3GraphOptions;
  static constexpr bool kGemma3 = true;
  static std::string_view Policy() {
    return llmp::spec::mappings::Gemma3Mapping().policies.front().text;
  }
  static const md::Gemma3Profile& P() { return md::Gemma3_4BQat(); }
  static auto Resources() { return llmp::test_support::gemma3::Resources(); }
  static auto Binding() { return md::BindGemma3(P(), "gemma3", Resources()).value(); }
  static auto State(std::uint32_t context, std::uint32_t rows) {
    return md::Gemma3State(P(), context, rows).value();
  }
  static llmp::spec::ModelSpec Spec() {
    auto values = llmp::spec::mappings::Gemma3V0Values(P());
    return *llmp::spec::Synthesize(llmp::spec::mappings::Gemma3Mapping(),
                                   llmp::test_support::Gemma3Source(), *values);
  }
  static llmp::spec::Source Source() { return llmp::test_support::Gemma3Source(); }
  static std::size_t Tensors(std::size_t segments) { return kg::Gemma3GraphTensors(P(), segments); }
  template <typename... A>
  static auto Build(A&&... a) {
    return kg::BuildGemma3Graph(std::forward<A>(a)...);
  }
  static constexpr auto kStateOnly = kg::Gemma3OutputMode::kStateOnly;
  static constexpr auto kHead = kg::Gemma3OutputMode::kHead;
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> Contexts() {
    return {{4096, 2}, {8448, 1}};
  }
  // The runner's joined decode beyond the serving envelope (its four-root
  // carriers): at 4,096 with twelve owners.
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> Cohorts() { return {{4096, 12}}; }
};

// A step: per owner (slot, rows, past), and its outputs.
enum class Out : std::uint8_t { kHead, kAll, kGreedy, kStateOnly };
struct Case {
  std::vector<std::array<std::uint32_t, 3>> segments;
  Out out = Out::kHead;
  std::string Name() const {
    std::string s;
    for (const auto& [slot, rows, past] : segments)
      s += std::format("[{}:{}@{}]", slot, rows, past);
    return s + std::format(" out {}", static_cast<int>(out));
  }
};

// The shapes a runner measures at setup (gemma2_runner.cc Setup's loop,
// with shared Q8's supplemental rows) with `slots` owners of 128 rows.
std::vector<Case> SetupCases(std::uint32_t context, std::uint32_t slots) {
  constexpr std::uint32_t kMaxRows = 128;
  std::vector<Case> out;
  const std::uint32_t wave_rows = kMaxRows * slots;
  const std::uint32_t head_rows = slots;
  for (const std::uint32_t budget : {wave_rows, head_rows}) {
    for (std::uint32_t count = 1; count <= slots; ++count) {
      for (const auto& rows :
           llmp::engine::support::ChunkMeasurementRows(kMaxRows, budget, count, true)) {
        const std::uint32_t most = *std::ranges::max_element(rows);
        for (const std::uint32_t past : {0U, context - most}) {
          for (const bool mixed : {false, true}) {
            if (mixed && (count != 2 || past != 0)) continue;
            Case c;
            for (std::uint32_t i = 0; i < count; ++i) {
              const std::uint32_t n = rows[i];
              c.segments.push_back({i, n, (past == 0 && (!mixed || i == 0)) ? 0U : context - n});
            }
            for (const Out o : {Out::kHead, Out::kAll, Out::kStateOnly, Out::kGreedy}) {
              if ((o == Out::kAll) != (budget == head_rows && budget != wave_rows)) continue;
              c.out = o;
              out.push_back(c);
            }
          }
        }
      }
    }
    if (head_rows == wave_rows) break;
  }
  return out;
}

// The joined one-row steps of 2 to `slots` owners a runner measures at
// setup (gemma3_runner.cc: every count of long owners, the last ones near
// the context's end), returning every owner's row, its token, or nothing.
std::vector<Case> CohortCases(std::uint32_t context, std::uint32_t slots) {
  std::vector<Case> out;
  for (std::uint32_t count = 2; count <= slots; ++count) {
    for (std::uint32_t long_count = 0; long_count < count; ++long_count) {
      Case c;
      for (std::uint32_t i = 0; i < count; ++i) {
        c.segments.push_back({i, 1, i < count - long_count ? 0U : context - 1});
      }
      for (const Out o : {Out::kHead, Out::kGreedy, Out::kStateOnly}) {
        c.out = o;
        out.push_back(c);
      }
    }
  }
  return out;
}

template <typename State>
std::uint32_t ReadOf(const std::array<std::uint32_t, 3>& s, const State& state, bool local) {
  const std::uint32_t read = (s[1] + s[2] + 255) / 256 * 256;
  return std::min(read, local ? state.local_cells : state.global_cells);
}

template <typename F, typename State>
typename F::Shape LegacyShape(const Case& c, const State& state) {
  typename F::Shape shape;
  std::uint32_t rows = 0;
  for (const auto& s : c.segments) {
    shape.segments.push_back({s[0], s[1], s[2], ReadOf(s, state, false), ReadOf(s, state, true)});
    rows += s[1];
  }
  const auto count = static_cast<std::uint32_t>(c.segments.size());
  shape.output_mode = c.out == Out::kStateOnly ? F::kStateOnly : F::kHead;
  shape.outputs = c.out == Out::kStateOnly ? 0U : c.out == Out::kAll ? rows : count;
  shape.greedy = c.out == Out::kGreedy;
  return shape;
}

template <typename State>
pl::StepShape IrShape(const Case& c, const State& state) {
  pl::StepShape shape;
  for (const auto& s : c.segments) {
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(s[0]),
                              .rows = s[1],
                              .past = s[2],
                              .read_cells = ReadOf(s, state, false),
                              .outputs = c.out == Out::kStateOnly ? 0U
                                         : c.out == Out::kAll     ? s[1]
                                                                  : 1U,
                              .want_head = c.out != Out::kStateOnly,
                              .greedy = c.out == Out::kGreedy});
  }
  return shape;
}

// ------------------------------------------------------------ the builders

struct Legacy {
  std::optional<kg::TensorArena> arena;
  std::vector<ggml_tensor*> nodes;
  std::vector<ggml_tensor*> inputs;
  std::map<const ggml_tensor*, std::string> leaves;
  kg::GraphPlan plan;
};

// Every leaf at an address of its own, every computed node at another
// (PlaceAndPlan's first pass), so that the gates see disjoint operands.
void BindAll(std::span<ggml_tensor* const> nodes) {
  std::uint64_t next = std::uint64_t{1} << 40U;
  std::set<ggml_tensor*> bound;
  for (ggml_tensor* node : nodes) {
    for (ggml_tensor* src : node->src) {
      while (src != nullptr && src->view_src != nullptr) src = src->view_src;
      if (src == nullptr || src->op != GGML_OP_NONE || bound.contains(src)) continue;
      kg::TensorArena::Bind(src, next);
      next += (ggml_nbytes(src) + 511) / 256 * 256 + 256;
      bound.insert(src);
    }
  }
  kg::BindDistinct(nodes, std::uint64_t{1} << 46U);
}

// The builders' leaves by what they bind (components_order_test's names).
template <typename Graph>
std::map<const ggml_tensor*, std::string> LegacyLeaves(const Graph& g) {
  std::map<const ggml_tensor*, std::string> out;
  const auto add = [&](const ggml_tensor* t, std::string name) {
    if (t != nullptr) out[t] = std::move(name);
  };
  add(g.tokens, "tokens");
  add(g.positions, "positions");
  add(g.out_ids, "out_ids");
  for (std::size_t s = 0; s < g.segments.size(); ++s) {
    const auto& seg = g.segments[s];
    add(seg.global_cells, std::format("cells.{}.global", s));
    add(seg.local_cells, std::format("cells.{}.local", s));
    if (seg.global_mask != nullptr && seg.global_mask->op == GGML_OP_NONE) {
      add(seg.global_mask, std::format("mask.{}.global", s));
      add(seg.local_mask, std::format("mask.{}.local", s));
    }
    for (std::size_t l = 0; l < seg.caches.size(); ++l) {
      add(seg.caches[l].first, std::format("k.{}.{}", l, s));
      add(seg.caches[l].second, std::format("v.{}.{}", l, s));
    }
  }
  for (const auto& w : g.weights) add(w.tensor, std::format("w.{}", w.resource.index));
  return out;
}

template <typename F, typename State>
Legacy BuildLegacy(const Case& c, const State& state, bool device_masks) {
  Legacy out;
  auto arena = kg::TensorArena::Create(F::Tensors(c.segments.size()));
  EXPECT_TRUE(arena.has_value());
  out.arena.emplace(std::move(*arena));
  const kg::DeviceChoices device = Recipe(F::kGemma3);
  typename F::Options options;
  options.device_masks = device_masks;
  options.shared_q8 = true;
  options.max_total_rows = 256;  // the runners' wave rows: two owners of 128
  // The serving recipe's owner forms and frontier head.
  options.narrow_final = true;
  options.owner_decode = true;
  options.packed_prefill = true;
  options.owner_prefill = true;
  options.flexible_owner_prefill = true;
  options.bounded_roots = true;
  auto g = F::Build(*out.arena, F::P(), F::Binding(), state, LegacyShape<F>(c, state), options,
                    device.dense_mmvq_shape);
  EXPECT_TRUE(g.has_value()) << g.error().detail;
  if (!g) return out;
  out.nodes = g->nodes;
  out.inputs = g->inputs;
  out.leaves = LegacyLeaves(*g);
  std::vector<ggml_tensor*> keep;
  for (ggml_tensor* k : {g->logits, g->greedy, g->hidden}) {
    if (k != nullptr) keep.push_back(k);
  }
  BindAll(out.nodes);
  auto plan = kg::PlanGraph(out.nodes, false, device, keep);
  EXPECT_TRUE(plan.has_value()) << plan.error().detail;
  if (plan) out.plan = std::move(*plan);
  return out;
}

// ------------------------------------------------------------ the lowering

struct Ir {
  ex::Registry registry = PlanningRegistry();
  llmp::spec::ModelSpec spec;
  llmp::spec::Source source;
  std::optional<md::StateLayoutDerived> layout{};
};

struct Lowered {
  std::size_t owners = 0;  // owner_attention applied
  std::optional<co::StepGraph> step;
  std::optional<pl::StepPlan> plan;
  std::optional<kg::LoweredStep> lowered;
  std::map<const ggml_tensor*, std::string> leaves;
};

// The step's leaves by what they bind, as the builders' are named.
template <typename Resources>
std::map<std::uint32_t, std::string> StepLeaves(const co::StepGraph& step,
                                                const Resources& resources) {
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
      default:
        break;
    }
  }
  for (const co::StepState& st : step.state) {
    out[st.tensor.v] =
        std::format("{}.{}.{}", st.ref.tensor == 0 ? "k" : "v", st.ref.layer, st.ref.segment);
  }
  for (const co::StepWeight& w : step.weights) {
    const std::string& name = w.parts.front().first;
    const auto at = std::ranges::find_if(
        resources, [&](const auto& r) { return std::ranges::contains(r.roles, name); });
    out[w.tensor.v] = std::format("w.{}", static_cast<std::size_t>(at - resources.begin()));
  }
  for (const ir::Node& n : step.graph.nodes()) {
    if (n.key.op == ir::Op::kMaskBuild) {
      const auto held = inputs.find(n.src[1].v);
      if (held != inputs.end()) out[n.out[0].v] = "mask." + where(*held->second);
    }
  }
  return out;
}

template <typename F, typename State>
Lowered Lower(Ir& ir_, const Case& c, const State& state, bool device_masks,
              std::string_view policy_text) {
  Lowered out;
  const pl::StepShape shape = IrShape(c, state);
  auto step =
      co::BuildStep(ir_.spec, 0, shape, {.tensors = ir_.source.tensors, .state = &*ir_.layout});
  EXPECT_TRUE(step.has_value()) << step.error();
  if (!step) return out;
  std::vector<ir::TensorId> staged;
  for (const co::StepInput& in : step->inputs) {
    if (in.kind != co::InputKind::kCellPositions) staged.push_back(in.tensor);
  }
  const pl::Policy policy = pl::ParsePolicy(policy_text).value();
  auto plan = pl::Plan(
      step->graph.Clone(), ir_.registry, Gb10(), policy,
      {.capturable = true, .staged = staged, .written = &kg::WrittenBytes, .shape = &shape});
  EXPECT_TRUE(plan.has_value()) << (plan ? "" : plan.error().detail);
  if (!plan) return out;
  out.owners = static_cast<std::size_t>(
      std::ranges::count(plan->applied, "owner_attention", &pl::Applied::pattern));
  out.step.emplace(std::move(*step));
  out.plan.emplace(std::move(*plan));
  const kn::Addresses addresses{};  // a measurement: placeholder addresses
  const kn::LowerInput input{.plan = *out.plan,
                             .step = *out.step,
                             .shape = shape,
                             .addresses = addresses,
                             .context = ir_.layout->context};
  auto lowered = kg::LowerStep(input, ir_.registry, Device(), {.device_masks = device_masks});
  EXPECT_TRUE(lowered.has_value()) << (lowered ? "" : lowered.error().detail);
  if (!lowered) return out;
  const auto names = StepLeaves(*out.step, F::Resources());
  for (const kg::LoweredLeaf& leaf : lowered->leaves) {
    const auto at = names.find(leaf.primitive.v);
    out.leaves[leaf.tensor] = at == names.end() ? "?" : at->second;
  }
  out.lowered.emplace(std::move(*lowered));
  return out;
}

// ------------------------------------------------------------ comparisons

void ExpectSameGraph(const Legacy& l, const Lowered& w) {
  const auto& a = l.nodes;
  const auto& b = w.lowered->nodes;
  ASSERT_EQ(a.size(), b.size());
  std::map<const ggml_tensor*, std::size_t> ai, bi;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ai[a[i]] = i;
    bi[b[i]] = i;
  }
  const auto name = [](const std::map<const ggml_tensor*, std::size_t>& index,
                       const std::map<const ggml_tensor*, std::string>& leaves,
                       const ggml_tensor* t) -> std::string {
    if (t == nullptr) return "-";
    if (const auto at = index.find(t); at != index.end()) return std::format("#{}", at->second);
    const auto leaf = leaves.find(t);
    return leaf == leaves.end() ? "?" : leaf->second;
  };
  for (std::size_t i = 0; i < a.size(); ++i) {
    const ggml_tensor* x = a[i];
    const ggml_tensor* y = b[i];
    SCOPED_TRACE(std::format("node {}: {} ({})", i, ggml_op_desc(x), ggml_get_name(x)));
    ASSERT_EQ(x->op, y->op) << ggml_op_desc(y);
    ASSERT_EQ(x->type, y->type);
    for (std::size_t d = 0; d < GGML_MAX_DIMS; ++d) {
      ASSERT_EQ(x->ne[d], y->ne[d]) << "ne " << d;
      ASSERT_EQ(x->nb[d], y->nb[d]) << "nb " << d;
    }
    ASSERT_EQ(x->view_offs, y->view_offs);
    ASSERT_EQ(x->flags, y->flags);
    ASSERT_EQ(std::memcmp(x->op_params, y->op_params, sizeof(x->op_params)), 0) << "op params";
    for (std::size_t s = 0; s < GGML_MAX_SRC; ++s) {
      ASSERT_EQ(name(ai, l.leaves, x->src[s]), name(bi, w.leaves, y->src[s])) << "source " << s;
      if (x->src[s] != nullptr && !ai.contains(x->src[s])) {
        // A leaf: its type, extents and flags (a weight's row padding).
        ASSERT_EQ(x->src[s]->type, y->src[s]->type) << "source " << s;
        ASSERT_EQ(x->src[s]->flags, y->src[s]->flags) << "source " << s;
        for (std::size_t d = 0; d < GGML_MAX_DIMS; ++d) {
          ASSERT_EQ(x->src[s]->ne[d], y->src[s]->ne[d]) << "source " << s;
        }
      }
    }
  }
}

void ExpectSamePlan(const Legacy& l, const Lowered& w) {
  std::map<const ggml_tensor*, std::size_t> ai, bi;
  for (std::size_t i = 0; i < l.nodes.size(); ++i) ai[l.nodes[i]] = i;
  for (std::size_t i = 0; i < w.lowered->nodes.size(); ++i) bi[w.lowered->nodes[i]] = i;
  const auto& a = l.plan.steps;
  const auto& b = w.lowered->plan.steps;
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(a[i].implementation, b[i].implementation) << "step " << i;
    ASSERT_EQ(a[i].operation, b[i].operation) << "step " << i;
    ASSERT_EQ(a[i].nodes.size(), b[i].nodes.size()) << "step " << i;
    for (std::size_t k = 0; k < a[i].nodes.size(); ++k) {
      ASSERT_EQ(ai.at(a[i].nodes[k]), bi.at(b[i].nodes[k])) << "step " << i;
    }
  }
}

// The inputs the step stages: the builders' inputs, each as its slot.
void ExpectInputs(const Legacy& l, const Lowered& w, bool device_masks) {
  std::multiset<std::string> want, got;
  for (const ggml_tensor* t : l.inputs) {
    want.insert(std::format("{} {}", l.leaves.at(t), ggml_nbytes(t)));
  }
  std::map<std::uint64_t, const ggml_tensor*> by_address;
  for (const auto& [t, name] : w.leaves) {
    by_address[static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data))] = t;
  }
  for (const kn::InputSlot& slot : w.lowered->inputs) {
    const ggml_tensor* t = by_address.at(slot.device_address);
    got.insert(std::format("{} {}", w.leaves.at(t), slot.bytes));
    if (slot.kind == kn::InputKind::kMask) {
      EXPECT_FALSE(device_masks);
      EXPECT_EQ(slot.bytes, slot.read * slot.mask_rows * 2);
    }
  }
  EXPECT_EQ(got, want);
}

template <typename F>
void ExpectFamilyLowers() {
  std::size_t shapes = 0, nodes = 0, steps = 0, owner_forms = 0;
  std::uint64_t biggest = 0;
  std::vector<std::tuple<std::uint32_t, std::uint32_t, bool>> runs;
  for (const auto& [context, slots] : F::Contexts()) runs.emplace_back(context, slots, false);
  for (const auto& [context, slots] : F::Cohorts()) runs.emplace_back(context, slots, true);
  for (const auto& [context, slots, cohorts] : runs) {
    const auto state = F::State(context, 128);
    Ir ir_{.spec = F::Spec(), .source = F::Source()};
    auto layout = co::StageLayout(ir_.spec, 0, context, 128, kExtent);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    ir_.layout.emplace(std::move(*layout));
    for (const Case& c : cohorts ? CohortCases(context, slots) : SetupCases(context, slots)) {
      for (const bool device_masks : {true, false}) {
        SCOPED_TRACE(
            std::format("context {} {}{}", context, c.Name(), device_masks ? "" : ", host masks"));
        const Legacy l = BuildLegacy<F>(c, state, device_masks);
        const Lowered w = Lower<F>(ir_, c, state, device_masks, F::Policy());
        ASSERT_TRUE(w.lowered.has_value());
        ExpectSameGraph(l, w);
        if (::testing::Test::HasFatalFailure()) return;
        ExpectSamePlan(l, w);
        if (::testing::Test::HasFatalFailure()) return;
        ExpectInputs(l, w, device_masks);
        // Each owner's own outputs, and nothing placed past the extent.
        std::uint32_t outputs = 0;
        for (const kn::OutputSlot& o : w.lowered->outputs) {
          EXPECT_LT(o.segment, c.segments.size());
          outputs += 1;
        }
        EXPECT_EQ(outputs, c.out == Out::kStateOnly ? 0U : c.segments.size());
        biggest = std::max(biggest, w.lowered->extent);
        owner_forms += w.owners;
        nodes += w.lowered->nodes.size();
        steps += w.lowered->plan.steps.size();
        ++shapes;
      }
    }
  }
  std::println(
      "{} lowered steps ({} with owner attention), {} nodes, {} plan steps, the largest "
      "activation extent {} bytes",
      shapes, owner_forms, nodes, steps, biggest);
  EXPECT_GT(shapes, 0U);
  EXPECT_GT(owner_forms, 0U);
}

TEST(GgmlLowerTest, Gemma2LowersToTheBuildersGraphs) { ExpectFamilyLowers<Gemma2Family>(); }

TEST(GgmlLowerTest, Gemma3LowersToTheBuildersGraphs) { ExpectFamilyLowers<Gemma3Family>(); }

// A selection PlanGraph would not make is refused, never run otherwise:
// without the norm-mul pattern the IR selects the norm and its mul alone
// (which unfused PlanGraph runs as one step of both launchers, the IR's two
// steps), but on a device whose own choices fuse norms PlanGraph runs the
// two as the fused RMSNorm-mul.
TEST(GgmlLowerTest, RefusesASelectionPlanGraphDoesNotMake) {
  Ir ir_{.spec = Gemma2Family::Spec(), .source = Gemma2Family::Source()};
  const auto state = Gemma2Family::State(md::kGemma2Context, 128);
  auto layout = co::StageLayout(ir_.spec, 0, md::kGemma2Context, 128, kExtent);
  ASSERT_TRUE(layout.has_value());
  ir_.layout.emplace(std::move(*layout));
  const Case c{.segments = {{0, 1, 7}}, .out = Out::kHead};
  const pl::StepShape shape = IrShape(c, state);
  auto step =
      co::BuildStep(ir_.spec, 0, shape, {.tensors = ir_.source.tensors, .state = &*ir_.layout});
  ASSERT_TRUE(step.has_value());
  std::vector<ir::TensorId> staged;
  for (const co::StepInput& in : step->inputs) {
    if (in.kind != co::InputKind::kCellPositions) staged.push_back(in.tensor);
  }
  const pl::Policy policy = pl::ParsePolicy(
                                "max_decode_rows 16\ndisable grouped_stores\ndisable "
                                "rms_norm_mul\ndisable rms_norm_mul_rope\ndisable rope_store\n")
                                .value();
  auto plan = pl::Plan(step->graph.Clone(), ir_.registry, Gb10(), policy,
                       {.staged = staged, .written = &kg::WrittenBytes});
  ASSERT_TRUE(plan.has_value());
  const kn::Addresses addresses{};
  const kn::LowerInput input{
      .plan = *plan, .step = *step, .shape = shape, .addresses = addresses, .context = 8192};
  // Unfused, the two run as PlanGraph's one step of both launchers.
  ASSERT_TRUE(kg::LowerStep(input, ir_.registry, Device()).has_value());
  kg::DeviceChoices fusing = Device();
  fusing.fuse_norms = true;
  auto lowered = kg::LowerStep(input, ir_.registry, fusing);
  ASSERT_FALSE(lowered.has_value());
  EXPECT_NE(lowered.error().detail.find("PlanGraph's step"), std::string::npos)
      << lowered.error().detail;
}

// Upstream's routing as the capabilities model it (mmvq.cu
// ggml_cuda_should_use_mmvq at the pin): the GB10's bound for its types,
// the RTX 3080 Ti's (sm_86) default, Ada's k-quant bounds, and nothing for
// an unquantized type.
TEST(GgmlLowerTest, CapabilitiesRouteAsUpstream) {
  using llmp::format::Encoding;
  EXPECT_TRUE(kg::MmvqRouted(Encoding::kQ4_0, 121, 8));
  EXPECT_FALSE(kg::MmvqRouted(Encoding::kQ4_0, 121, 9));
  EXPECT_TRUE(kg::MmvqRouted(Encoding::kQ2_K, 121, 6));
  EXPECT_FALSE(kg::MmvqRouted(Encoding::kQ2_K, 121, 7));
  EXPECT_TRUE(kg::MmvqRouted(Encoding::kQ8_0, 86, 8));
  EXPECT_FALSE(kg::MmvqRouted(Encoding::kQ2_K, 89, 5));
  EXPECT_TRUE(kg::MmvqRouted(Encoding::kQ3_K, 89, 6));
  EXPECT_FALSE(kg::MmvqRouted(Encoding::kF16, 121, 1));
  // The fused quantized GLU takes one column of gate and up weights of one
  // type (MulMatVecQGluFusible), so a mixed pair stays two products.
  const ex::Capability* glu = kg::CapabilityOf("ggml.mul_mat_geglu.mmvq_fused");
  ASSERT_NE(glu, nullptr);
  ASSERT_NE(glu->accepts, nullptr);
  const ir::GluParams geglu{.kind = ir::GluKind::kGeGluTanh};
  using llmp::format::Layout;
  std::array<ex::OperandFormat, 3> operands{
      {{.operand = 0, .encoding = Encoding::kQ4_0, .layout = Layout::kGgmlBlocks},
       {.operand = 1, .encoding = Encoding::kQ4_0, .layout = Layout::kGgmlBlocks},
       {.operand = 2, .encoding = Encoding::kF32, .layout = Layout::kDense}}};
  ex::CapabilityQuery query{.op = glu->op,
                            .operands = operands,
                            .rows = 1,
                            .k = 2560,
                            .n = 10240,
                            .arch = 121,
                            .params = std::as_bytes(std::span(&geglu, 1))};
  EXPECT_TRUE(glu->accepts(query));
  query.rows = 2;
  EXPECT_FALSE(glu->accepts(query));
  query.rows = 1;
  operands[1].encoding = Encoding::kQ6_K;
  EXPECT_FALSE(glu->accepts(query));
  // Every declared capability is a capability of its implementation's IR
  // key and, but for the module's own families, stands alone.
  for (const kg::CapabilityDeclaration& d : kg::CapabilityDeclarations()) {
    ASSERT_NE(d.capability, nullptr) << d.name;
    EXPECT_EQ(kg::CapabilityOf(d.name), d.capability) << d.name;
    EXPECT_FALSE(kg::AcceptsTextOf(d.name).empty()) << d.name;
    EXPECT_TRUE(d.capability->group.empty() || d.capability->group == "ggml.mul_mat" ||
                d.capability->group == "ggml.mul_mat_id" ||
                d.capability->group == "ggml.flash_attn_ext")
        << d.name;
  }
}

}  // namespace
