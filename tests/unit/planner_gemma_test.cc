// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The planner (planner/plan.h) on the Gemma 2 and Gemma 3 specs' steps
// against today's planning of the same steps (kernels/ggml/graph_plan.h
// PlanGraph over BuildGemma2Graph/BuildGemma3Graph) with each family's
// production recipe (runtime/serving.cc: fused norm-mul, quantized GeGLU and
// norm-add for both, norm-rope and grouped KV stores for Gemma 3 only,
// device masks, shared Q8 preparation), on a fake registry that declares
// what that recipe selects as capabilities, and a model of the GB10's
// product routing (MMVQ through 8 columns, the fused GeGLU at one) on both
// sides:
// - the fusions: the fused steps PlanGraph plans and the patterns the
//   planner applies (and its grouped-store batches) cover the same node
//   sets, by kind, the step's nodes matched to the builders' in their
//   execution order (components_order_test shows the primitive step is the
//   builder's graph node for node on its shapes; here every shape checks
//   each matched pair's operation, and each product's and weight
//   multiply's weights, so a pattern that took another layer's or owner's
//   node would not match);
// - shared Q8: the products sharing each preparation, by weight;
// - placement: for every shape the runners measure at setup (their
//   ChunkMeasurementRows loop at the serving defaults), the planned step's
//   activation extent is no more than PlaceActivations' for the builders'
//   graph, the peak-memory gate's early warning, and the placement holds:
//   no two tensors live at once share bytes, by lifetimes recomputed here
//   from the planned steps. The owner-attention forms
//   (owner decode, packed and owner prefill) are slice G1b's: here both
//   sides plan those shapes' per-owner attention.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <print>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "components/step.h"
#include "engine/support.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/gemma2_graph.h"
#include "kernels/ggml/gemma3_graph.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/set_rows_group.h"
#include "kernels/ggml/tensors.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "planner/placement.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec_fixtures.h"

namespace llmp::planner {
namespace {

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace ex = llmp::execution;
using format::Encoding;
using format::Layout;

constexpr std::uint64_t kExtent = 2ULL << 20U;

// ------------------------------------------------- the fake registry

constexpr std::array kQuantized = {Encoding::kQ4_0, Encoding::kQ8_0};
constexpr std::array kF32 = {Encoding::kF32};
constexpr std::array kQ8_1 = {Encoding::kQ8_1};
constexpr std::array kF16 = {Encoding::kF16};
// MMVQ and MMQ: quantized weights and F32 activations; the prepared
// product: the activations already Q8_1.
constexpr std::array kProductRules = {ex::OperandRule{.operand = 0, .encodings = kQuantized},
                                      ex::OperandRule{.operand = 1, .encodings = kF32}};
constexpr std::array kPreparedRules = {ex::OperandRule{.operand = 0, .encodings = kQuantized},
                                       ex::OperandRule{.operand = 1, .encodings = kQ8_1}};
constexpr std::array kGluRules = {ex::OperandRule{.operand = 0, .encodings = kQuantized},
                                  ex::OperandRule{.operand = 1, .encodings = kQuantized},
                                  ex::OperandRule{.operand = 2, .encodings = kF32}};
constexpr std::array kStoreRules = {ex::OperandRule{.operand = 0, .encodings = kF16},
                                    ex::OperandRule{.operand = 1, .encodings = kF32}};

// CheckGemmaNormRope's full D256/D512 rotation; CheckGemmaNormAdd's widths.
bool NormRopeWidths(const ex::CapabilityQuery& q) { return q.k == 256 || q.k == 512; }
bool NormAddWidths(const ex::CapabilityQuery& q) {
  return q.k == 2304 || q.k == 2560 || q.k == 2816 || q.k == 5376;
}

ex::Implementation Fake(std::string name, ex::Operation operation, ex::Capability capability) {
  return {.name = std::move(name),
          .operation = operation,
          .source = "fake",
          .revision = "r1",
          .build = "cpu",
          .variant = "model of the GB10's GGML selection",
          .capability = capability};
}

ex::Capability Any(ir::OpKey key) { return ex::Capability{.op = key}; }

ex::Registry GemmaRegistry() {
  using ex::Operation;
  using ir::FusedKind;
  using ir::Op;
  const auto prim = [](Op op) { return Any(ir::Primitive(op)); };
  std::vector<ex::Implementation> v = {
      Fake("fake.get_rows", Operation::kGetRows, prim(Op::kGetRows)),
      Fake("fake.scale", Operation::kScale, prim(Op::kScale)),
      Fake("fake.rms_norm", Operation::kRmsNorm, prim(Op::kRmsNorm)),
      Fake("fake.mul", Operation::kMul, prim(Op::kMul)),
      Fake("fake.add", Operation::kAdd, prim(Op::kAdd)),
      Fake("fake.rope", Operation::kRope, prim(Op::kRope)),
      Fake("fake.set_rows", Operation::kSetRows, prim(Op::kCacheStore)),
      Fake("fake.flash_attn", Operation::kFlashAttn, prim(Op::kAttention)),
      Fake("fake.concat", Operation::kConcat, prim(Op::kConcat)),
      Fake("fake.cont", Operation::kCont, prim(Op::kCont)),
      Fake("fake.geglu", Operation::kGeGlu, prim(Op::kGlu)),
      Fake("fake.unary", Operation::kUnary, prim(Op::kUnary)),
      Fake("fake.argmax", Operation::kTopK, prim(Op::kArgmax)),
      Fake("fake.mask", Operation::kFill, prim(Op::kMaskBuild)),
      Fake("fake.q8_1", Operation::kQuantize,
           ex::Capability{.op = ir::Primitive(Op::kPrepare), .shape = {.max_rows = 8}}),
      Fake("fake.mul_mat.mmvq", Operation::kMatMul,
           ex::Capability{.op = ir::Primitive(Op::kMatMul),
                          .operands = kProductRules,
                          .shape = {.max_rows = 8},
                          .group = "fake.mul_mat"}),
      Fake("fake.mul_mat.mmq", Operation::kMatMul,
           ex::Capability{.op = ir::Primitive(Op::kMatMul),
                          .operands = kProductRules,
                          .shape = {.min_rows = 9},
                          .group = "fake.mul_mat"}),
      Fake("fake.mmvq.prepared", Operation::kMatMul,
           ex::Capability{.op = ir::Primitive(Op::kMatMul),
                          .operands = kPreparedRules,
                          .shape = {.max_rows = 8}}),
      Fake("fake.rms_norm_mul.fused", Operation::kRmsNormMul,
           Any(ir::Fused(FusedKind::kRmsNormMul))),
      Fake("fake.rms_norm_mul_rope.fused", Operation::kRmsNormMulRope,
           ex::Capability{.op = ir::Fused(FusedKind::kRmsNormMulRope), .accepts = &NormRopeWidths}),
      Fake("fake.rms_norm_mul_add.fused", Operation::kRmsNormMulAdd,
           ex::Capability{.op = ir::Fused(FusedKind::kRmsNormMulAdd), .accepts = &NormAddWidths}),
      Fake("fake.mul_mat_geglu.mmvq_fused", Operation::kMulMatGeGlu,
           ex::Capability{.op = ir::Fused(FusedKind::kMatMulGlu),
                          .operands = kGluRules,
                          .shape = {.max_rows = 1}}),
      Fake("fake.set_rows.grouped", Operation::kSetRows,
           ex::Capability{
               .op = ir::Fused(FusedKind::kGroupedStores), .operands = kStoreRules, .batch = 16}),
      Fake("fake.rope_set_rows.fused", Operation::kRopeSetRows,
           Any(ir::Fused(FusedKind::kRopeStore))),
  };
  return std::move(ex::Registry::Create(std::move(v)).value());
}

const DeviceFacts& Gb10() {
  static const DeviceFacts facts{.device_class = "sm_121", .arch = 121};
  return facts;
}

// The recipes as policies: the architecture mappings' text.
const std::string_view kGemma2Policy = spec::mappings::Gemma2Mapping().policy;
const std::string_view kGemma3Policy = spec::mappings::Gemma3Mapping().policy;

Policy PolicyOf(std::string_view text, bool shared_q8) {
  Policy p = ParsePolicy(text).value();
  if (!shared_q8) {
    p.disabled.emplace_back("prepared_inputs");
    Canonicalize(p);
  }
  return p;
}

// What PlanGraph's DeviceChoices are under the recipe, with the same model
// of the device's routing.
kg::DeviceChoices LegacyChoices(bool gemma3) {
  kg::DeviceChoices d;
  d.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  d.quant = [](const ggml_tensor* n) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
    return n->src[1]->ne[1] <= 8 ? kg::QuantMulMatPath::kVector : kg::QuantMulMatPath::kTile;
  };
  d.quant_glu_fusible = [](const ggml_tensor*, const ggml_tensor* up, const ggml_tensor*) {
    return up->ne[1] == 1;
  };
  d.dense_mmvq_shape = [](ggml_type, std::int64_t columns) { return columns >= 1 && columns <= 8; };
  d.fuse_norms = true;
  d.fuse_quant_glu = true;
  d.fuse_norm_add = true;
  d.fuse_norm_rope = gemma3;
  d.group_set_rows = gemma3;
  return d;
}

// ------------------------------------------------- the families

struct Gemma2Family {
  using Profile = md::Gemma2Profile;
  using Shape = kg::Gemma2ChunkShape;
  using Options = kg::Gemma2GraphOptions;
  static constexpr bool kGemma3 = false;
  static const Profile& P() { return md::Gemma2_2B(); }
  static auto Resources() { return test_support::gemma2::Resources(); }
  static auto Binding() { return md::BindGemma2(P(), "gemma2", Resources()).value(); }
  static auto State(std::uint32_t context, std::uint32_t rows) {
    return md::Gemma2State(P(), context, rows).value();
  }
  static spec::ModelSpec Spec() {
    auto values = spec::mappings::Gemma2V0Values(P());
    return *spec::Synthesize(spec::mappings::Gemma2Mapping(), test_support::Gemma2Source(),
                             *values);
  }
  static spec::Source Source() { return test_support::Gemma2Source(); }
  static std::size_t Tensors(std::size_t segments) { return kg::Gemma2GraphTensors(P(), segments); }
  template <typename... A>
  static auto Build(A&&... a) {
    return kg::BuildGemma2Graph(std::forward<A>(a)...);
  }
  static constexpr auto kStateOnly = kg::Gemma2OutputMode::kStateOnly;
  static constexpr auto kHead = kg::Gemma2OutputMode::kHead;
};

struct Gemma3Family {
  using Profile = md::Gemma3Profile;
  using Shape = kg::Gemma3ChunkShape;
  using Options = kg::Gemma3GraphOptions;
  static constexpr bool kGemma3 = true;
  static const Profile& P() { return md::Gemma3_4BQat(); }
  static auto Resources() { return test_support::gemma3::Resources(); }
  static auto Binding() { return md::BindGemma3(P(), "gemma3", Resources()).value(); }
  static auto State(std::uint32_t context, std::uint32_t rows) {
    return md::Gemma3State(P(), context, rows).value();
  }
  static spec::ModelSpec Spec() {
    auto values = spec::mappings::Gemma3V0Values(P());
    return *spec::Synthesize(spec::mappings::Gemma3Mapping(), test_support::Gemma3Source(),
                             *values);
  }
  static spec::Source Source() { return test_support::Gemma3Source(); }
  static std::size_t Tensors(std::size_t segments) { return kg::Gemma3GraphTensors(P(), segments); }
  template <typename... A>
  static auto Build(A&&... a) {
    return kg::BuildGemma3Graph(std::forward<A>(a)...);
  }
  static constexpr auto kStateOnly = kg::Gemma3OutputMode::kStateOnly;
  static constexpr auto kHead = kg::Gemma3OutputMode::kHead;
};

// A step as both sides take it: per owner (slot, rows, past), and its
// outputs: the head (one row an owner), every row, greedy, or none.
enum class Out : std::uint8_t { kHead, kAll, kGreedy, kStateOnly };
struct Case {
  std::vector<std::array<std::uint32_t, 3>> segments;  // slot, rows, past
  Out out = Out::kHead;
  std::string Name() const {
    std::string s;
    for (const auto& [slot, rows, past] : segments) {
      s += std::format("[{}:{}@{}]", slot, rows, past);
    }
    return s + std::format(" out {}", static_cast<int>(out));
  }
};

// The shapes a runner measures at setup (gemma2_runner.cc Setup's loop)
// with `slots` owners of up to `max_rows` rows.
std::vector<Case> SetupCases(std::uint32_t context, std::uint32_t max_rows, std::uint32_t slots) {
  std::vector<Case> out;
  const std::uint32_t wave_rows = max_rows * slots;
  const std::uint32_t head_rows = slots;
  for (const std::uint32_t budget : {wave_rows, head_rows}) {
    for (std::uint32_t count = 1; count <= slots; ++count) {
      for (const auto& rows :
           engine::support::ChunkMeasurementRows(max_rows, budget, count, true)) {
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
StepShape IrShape(const Case& c, const State& state) {
  StepShape shape;
  for (const auto& s : c.segments) {
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(s[0]),
                              .rows = s[1],
                              .past = s[2],
                              .read_cells = ReadOf(s, state, false),
                              .outputs = c.out == Out::kStateOnly ? 0U
                                         : c.out == Out::kAll     ? s[1]
                                                                  : 1U,
                              .greedy = c.out == Out::kGreedy});
  }
  return shape;
}

// ------------------------------------------------- the legacy side

struct Legacy {
  std::optional<kg::TensorArena> arena;
  std::vector<ggml_tensor*> nodes;
  std::vector<ggml_tensor*> inputs;
  std::vector<ggml_tensor*> keep;
  std::map<const ggml_tensor*, std::size_t> weight;  // leaf -> resource index
  kg::GraphPlan plan;
  std::uint64_t extent = 0;
};

// Every leaf the nodes read at an address of its own, every computed node
// at another (PlaceAndPlan's first pass), so the fusion checks see
// disjoint operands.
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

// `production`: with the owner-attention forms the runners serve with
// (owner decode, packed, owner and flexible owner prefill, bounded roots).
template <typename F, typename State>
Legacy PlanLegacy(const Case& c, const State& state, bool shared_q8, bool production = false) {
  Legacy out;
  auto arena = kg::TensorArena::Create(F::Tensors(c.segments.size()));
  EXPECT_TRUE(arena.has_value());
  out.arena.emplace(std::move(*arena));
  const auto binding = F::Binding();
  const kg::DeviceChoices device = LegacyChoices(F::kGemma3);
  typename F::Options options;
  options.device_masks = true;
  options.shared_q8 = shared_q8;
  options.narrow_final = true;   // the runners' frontier_head: the components narrow so
  options.max_total_rows = 256;  // the runners' wave rows: two owners of 128
  if (production) {
    options.owner_decode = true;
    options.packed_prefill = true;
    options.owner_prefill = true;
    options.flexible_owner_prefill = true;
    options.bounded_roots = true;
  }
  auto g = F::Build(*out.arena, F::P(), binding, state, LegacyShape<F>(c, state), options,
                    device.dense_mmvq_shape);
  EXPECT_TRUE(g.has_value()) << g.error().detail;
  if (!g) return out;
  out.nodes = g->nodes;
  out.inputs = g->inputs;
  for (ggml_tensor* k : {g->logits, g->greedy, g->hidden}) {
    if (k != nullptr) out.keep.push_back(k);
  }
  for (const auto& w : g->weights) out.weight[w.tensor] = w.resource.index;
  BindAll(out.nodes);
  auto plan = kg::PlanGraph(out.nodes, false, device, out.keep);
  EXPECT_TRUE(plan.has_value()) << plan.error().detail;
  if (!plan) return out;
  out.plan = std::move(*plan);
  auto placed = kg::PlaceActivations(out.nodes, out.plan, out.inputs, 256, out.keep);
  EXPECT_TRUE(placed.has_value()) << placed.error().detail;
  if (placed) out.extent = placed->extent;
  return out;
}

// ------------------------------------------------- the planner side

struct Planned {
  std::optional<components::StepGraph> step;
  std::optional<StepPlan> plan;
};

template <typename F>
struct Ir {
  spec::ModelSpec spec = F::Spec();
  spec::Source source = F::Source();
  std::optional<model::StateLayoutDerived> layout;
  ex::Registry registry = GemmaRegistry();
};

template <typename F, typename State>
Planned PlanIr(Ir<F>& ir, const Case& c, const State& state, const Policy& policy) {
  Planned out;
  auto step = components::BuildStep(ir.spec, 0, IrShape(c, state),
                                    {.tensors = ir.source.tensors, .state = &*ir.layout});
  EXPECT_TRUE(step.has_value()) << step.error();
  if (!step) return out;
  // The device masks derive their cells' positions (the lowering's
  // producer): those inputs are not staged.
  std::vector<ir::TensorId> staged;
  for (const components::StepInput& in : step->inputs) {
    if (in.kind != components::InputKind::kCellPositions) staged.push_back(in.tensor);
  }
  // A copy of the primitive graph for the plan (graphs are move-only):
  // rebuilt from the same step.
  auto again = components::BuildStep(ir.spec, 0, IrShape(c, state),
                                     {.tensors = ir.source.tensors, .state = &*ir.layout});
  EXPECT_TRUE(again.has_value());
  auto plan = Plan(std::move(again->graph), ir.registry, Gb10(), policy, {.staged = staged});
  EXPECT_TRUE(plan.has_value()) << (plan ? "" : plan.error().detail);
  out.step.emplace(std::move(*step));
  if (plan) out.plan.emplace(std::move(*plan));
  return out;
}

// The primitive graph's nodes the builders' graphs list (computed nodes
// and views; not the positions' views the masks read), in order: the
// i-th is the builders' i-th node (components_order_test).
std::vector<std::size_t> Comparable(const ir::Graph& g) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < g.nodes().size(); ++i) {
    const ir::Node& n = g.nodes()[i];
    const ir::Op op = n.key.op;
    if (op == ir::Op::kInput || op == ir::Op::kWeight || op == ir::Op::kState) continue;
    if (op == ir::Op::kView) {
      const ir::TensorId src = n.src[0];
      if (g.node(g.producer(src)).key.op == ir::Op::kInput &&
          g.tensor(src).format.encoding == Encoding::kI32) {
        continue;
      }
    }
    out.push_back(i);
  }
  return out;
}

// The IR operation a builders' node is (components_order_test's map), or
// none where the map has no entry.
std::optional<ir::Op> LegacyOp(const ggml_tensor* t) {
  switch (t->op) {
    case GGML_OP_GET_ROWS:
      return ir::Op::kGetRows;
    case GGML_OP_SCALE:
      return ir::Op::kScale;
    case GGML_OP_RMS_NORM:
      return ir::Op::kRmsNorm;
    case GGML_OP_MUL:
      return ir::Op::kMul;
    case GGML_OP_ADD:
      return ir::Op::kAdd;
    case GGML_OP_MUL_MAT:
      return ir::Op::kMatMul;
    case GGML_OP_RESHAPE:
      return ir::Op::kReshape;
    case GGML_OP_VIEW:
      return ir::Op::kView;
    case GGML_OP_PERMUTE:
      return ir::Op::kPermute;
    case GGML_OP_ROPE:
      return ir::Op::kRope;
    case GGML_OP_SET_ROWS:
      return ir::Op::kCacheStore;
    case GGML_OP_FLASH_ATTN_EXT:
      return ir::Op::kAttention;
    case GGML_OP_CONCAT:
      return ir::Op::kConcat;
    case GGML_OP_GLU:
      return ir::Op::kGlu;
    case GGML_OP_UNARY:
      return ir::Op::kUnary;
    default:
      return std::nullopt;
  }
}

// Weight leaves by what they bind, to their resource's index.
template <typename Resources>
std::map<ir::WeightRef, std::size_t> WeightIndex(const components::StepGraph& step,
                                                 const Resources& resources) {
  std::map<ir::WeightRef, std::size_t> weight;
  for (const components::StepWeight& w : step.weights) {
    const std::string& name = w.parts.front().first;
    const auto at = std::ranges::find_if(
        resources, [&](const auto& r) { return std::ranges::contains(r.roles, name); });
    weight[w.ref] = static_cast<std::size_t>(at - resources.begin());
  }
  return weight;
}

// The comparable nodes pair up by operation, and every product and weight
// multiply reads the same weight resources on both sides.
template <typename Resources>
void ExpectSameNodes(const Legacy& l, const components::StepGraph& step,
                     const Resources& resources) {
  const ir::Graph& g = step.graph;
  const auto comparable = Comparable(g);
  ASSERT_EQ(comparable.size(), l.nodes.size());
  const auto weight = WeightIndex(step, resources);
  for (std::size_t i = 0; i < comparable.size(); ++i) {
    const ggml_tensor* t = l.nodes[i];
    const ir::Node& n = g.node(ir::NodeId{static_cast<std::uint32_t>(comparable[i])});
    if (const auto op = LegacyOp(t)) {
      ASSERT_EQ(n.key.op, *op) << "node " << i << " " << ggml_op_desc(t);
    }
    if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL) continue;
    std::vector<std::size_t> want;
    std::vector<std::size_t> got;
    for (const ggml_tensor* src : t->src) {
      if (src != nullptr && l.weight.contains(src)) want.push_back(l.weight.at(src));
    }
    for (const ir::TensorId src : n.sources()) {
      if (!src.valid()) continue;
      if (const auto* ref = std::get_if<ir::WeightRef>(&g.tensor(src).binding)) {
        got.push_back(weight.at(*ref));
      }
    }
    ASSERT_EQ(got, want) << "node " << i << " " << ggml_op_desc(t);
  }
}

// Every two placed tensors whose lifetimes meet hold disjoint bytes, the
// lifetimes recomputed from the planned steps: from the step computing a
// root (a staged input from the start) to the last step reading it or a
// view of it, or reading its prepared form (kPrepare's source); the
// outputs, kept tensors and the last launching node's results to the end.
void ExpectPlacedDisjoint(const StepPlan& p) {
  const ir::Graph& g = p.graph;
  const auto end = static_cast<std::int64_t>(p.steps.size());
  struct Life {
    std::int64_t first = 0;
    std::int64_t last = -1;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
  };
  std::map<std::uint32_t, Life> life;
  for (const auto& [t, offset] : p.placement.offsets) {
    const ir::Role role = g.tensor(t).role;
    const auto bytes = RootBytes(g.tensor(t));
    ASSERT_TRUE(bytes.has_value());
    life[t.v] = {.first = role == ir::Role::kInput || role == ir::Role::kConstant ? -1 : end + 1,
                 .offset = offset,
                 .bytes = (*bytes + 255) / 256 * 256};
    EXPECT_LE(offset + life[t.v].bytes, p.placement.extent);
  }
  const auto read = [&](ir::TensorId t, std::int64_t s) {
    if (const auto it = life.find(g.root(t).v); it != life.end()) {
      it->second.last = std::max(it->second.last, s);
    }
  };
  for (std::size_t s = 0; s < p.steps.size(); ++s) {
    const auto step = static_cast<std::int64_t>(s);
    for (const ir::NodeId id : p.steps[s].nodes) {
      const ir::Node& n = g.node(id);
      for (const ir::TensorId o : n.outputs()) {
        if (const auto it = life.find(o.v); it != life.end()) {
          it->second.first = std::min(it->second.first, step);
        }
      }
      for (const ir::TensorId src : n.sources()) {
        if (!src.valid()) continue;
        read(src, step);
        const ir::Node& made = g.node(g.producer(g.root(src)));
        if (made.key == ir::Primitive(ir::Op::kPrepare)) read(made.src[0], step);
      }
    }
  }
  for (const auto list : {g.outputs(), g.keep()}) {
    for (const ir::TensorId t : list) read(t, end);
  }
  for (std::size_t i = g.nodes().size(); i-- > 0;) {
    if (LaunchesNothing(g.nodes()[i].key)) continue;
    for (const ir::TensorId o : g.nodes()[i].outputs()) read(o, end);
    break;
  }
  std::vector<std::pair<std::uint32_t, Life>> order(life.begin(), life.end());
  for (auto& [t, l] : order) {
    ASSERT_LE(l.first, end) << "tensor " << t << " is computed by no step";
    l.last = std::max(l.last, l.first);
  }
  std::ranges::sort(order, {}, [](const auto& e) { return e.second.first; });
  std::vector<std::pair<std::uint32_t, Life>> live;
  for (const auto& [t, l] : order) {
    std::erase_if(live, [&](const auto& e) { return e.second.last < l.first; });
    for (const auto& [u, m] : live) {
      ASSERT_TRUE(l.offset + l.bytes <= m.offset || m.offset + m.bytes <= l.offset)
          << "tensors " << t << " and " << u << " share bytes while both live";
    }
    live.emplace_back(t, l);
  }
}

using Fusions = std::vector<std::pair<std::string, std::vector<std::size_t>>>;

std::string Kind(std::string_view implementation) {
  if (implementation == kg::kRmsNormMulFused) return "rms_norm_mul";
  if (implementation == kg::kGemmaNormRopeName) return "rms_norm_mul_rope";
  if (implementation == kg::kGemmaNormAddName) return "rms_norm_mul_add";
  if (implementation == kg::kMulMatGeGluQFused || implementation == kg::kMulMatGluQFused)
    return "mul_mat_glu";
  if (implementation == kg::kSetRowsGroupedName) return "grouped_stores";
  if (implementation == kg::kRopeSetRowsFused) return "rope_store";
  return "";
}

Fusions LegacyFusions(const Legacy& l) {
  std::map<const ggml_tensor*, std::size_t> index;
  for (std::size_t i = 0; i < l.nodes.size(); ++i) index[l.nodes[i]] = i;
  Fusions out;
  for (const kg::PlanStep& s : l.plan.steps) {
    std::string kind = Kind(s.implementation);
    if (kind.empty()) {
      if (s.nodes.size() == 1 || s.implementation == kg::kRmsNormMulUnfused) continue;
      kind = std::format("other {}", s.implementation);
    }
    std::vector<std::size_t> nodes;
    for (const ggml_tensor* n : s.nodes) nodes.push_back(index.at(n));
    std::ranges::sort(nodes);
    out.emplace_back(std::move(kind), std::move(nodes));
  }
  std::ranges::sort(out);
  return out;
}

Fusions IrFusions(const ir::Graph& primitive, const StepPlan& plan) {
  std::map<std::size_t, std::size_t> index;
  const auto comparable = Comparable(primitive);
  for (std::size_t i = 0; i < comparable.size(); ++i) index[comparable[i]] = i;
  Fusions out;
  for (const Applied& a : plan.applied) {
    std::vector<std::size_t> nodes;
    for (const ir::NodeId n : a.replaced) nodes.push_back(index.at(n.v));
    std::ranges::sort(nodes);
    out.emplace_back(std::string(a.pattern), std::move(nodes));
  }
  for (const Selected& s : plan.steps) {
    if (s.batch.empty()) continue;
    std::vector<std::size_t> nodes;
    for (const ir::NodeId n : s.nodes) nodes.push_back(index.at(plan.from.at(n.v).v));
    std::ranges::sort(nodes);
    out.emplace_back(std::string(s.batch), std::move(nodes));
  }
  std::ranges::sort(out);
  return out;
}

// Each shared preparation's products, by weight resource, in order.
std::vector<std::vector<std::size_t>> LegacyShared(const Legacy& l) {
  std::vector<std::vector<std::size_t>> out;
  std::map<const ggml_tensor*, std::size_t> prep;
  for (const ggml_tensor* n : l.nodes) {
    if (n->op == GGML_OP_CUSTOM && kg::LlmpOpOf(n) == kg::LlmpOp::kQuantizeQ8) {
      prep[n] = out.size();
      out.emplace_back();
    }
  }
  for (const ggml_tensor* n : l.nodes) {
    if (n->op == GGML_OP_CUSTOM && kg::LlmpOpOf(n) == kg::LlmpOp::kMmvqPrepared) {
      out.at(prep.at(n->src[1])).push_back(l.weight.at(n->src[0]));
    }
  }
  for (auto& g : out) std::ranges::sort(g);
  return out;
}

template <typename Resources>
std::vector<std::vector<std::size_t>> IrShared(const components::StepGraph& step,
                                               const StepPlan& plan, const Resources& resources) {
  const auto weight = WeightIndex(step, resources);
  const ir::Graph& g = plan.graph;
  std::vector<std::vector<std::size_t>> out;
  std::map<std::uint32_t, std::size_t> prep;
  for (const ir::Node& n : g.nodes()) {
    if (n.key == ir::Primitive(ir::Op::kPrepare)) {
      prep[n.out[0].v] = out.size();
      out.emplace_back();
    }
  }
  for (const ir::Node& n : g.nodes()) {
    if (n.key == ir::Primitive(ir::Op::kMatMul) && prep.contains(n.src[1].v)) {
      const auto& ref = std::get<ir::WeightRef>(g.tensor(n.src[0]).binding);
      out.at(prep.at(n.src[1].v)).push_back(weight.at(ref));
    }
  }
  for (auto& v : out) std::ranges::sort(v);
  return out;
}

// The shapes both comparisons run: the setup shapes at the serving
// defaults, and components_order_test's.
template <typename F>
std::vector<std::pair<std::uint32_t, std::vector<Case>>> AllCases() {
  std::vector<std::pair<std::uint32_t, std::vector<Case>>> out;
  if constexpr (F::kGemma3) {
    out.emplace_back(4096, SetupCases(4096, 128, 2));
    out.emplace_back(8448, SetupCases(8448, 128, 1));
  } else {
    out.emplace_back(md::kGemma2Context, SetupCases(md::kGemma2Context, 128, 2));
  }
  return out;
}

template <typename F>
void ExpectFamilyMatches(std::string_view policy_text) {
  std::size_t shapes = 0, fused = 0, shared = 0;
  double least = 2.0, most = 0.0, served = 0.0;
  for (auto& [context, cases] : AllCases<F>()) {
    const auto state = F::State(context, 128);
    Ir<F> ir;
    auto layout = components::StageLayout(ir.spec, 0, context, 128, kExtent);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    ir.layout.emplace(std::move(*layout));
    for (const Case& c : cases) {
      SCOPED_TRACE(std::format("context {} {}", context, c.Name()));
      // The fusions, without shared Q8 (its custom nodes would change the
      // node correspondence).
      {
        const Legacy l = PlanLegacy<F>(c, state, false);
        const Planned p = PlanIr(ir, c, state, PolicyOf(policy_text, false));
        ASSERT_TRUE(p.plan.has_value());
        const ir::Graph& primitive = p.step->graph;
        ExpectSameNodes(l, *p.step, F::Resources());
        if (::testing::Test::HasFatalFailure()) return;
        const Fusions want = LegacyFusions(l);
        ASSERT_EQ(IrFusions(primitive, *p.plan), want);
        fused += want.size();
      }
      // Shared Q8 as served: the same products share each preparation, and
      // the planned step's activations fit the builders' placement.
      {
        const Legacy l = PlanLegacy<F>(c, state, true);
        const Planned p = PlanIr(ir, c, state, PolicyOf(policy_text, true));
        ASSERT_TRUE(p.plan.has_value());
        const auto want = LegacyShared(l);
        EXPECT_EQ(IrShared(*p.step, *p.plan, F::Resources()), want);
        shared += want.size();
        ASSERT_GT(l.extent, 0U);
        EXPECT_LE(p.plan->placement.extent, l.extent);
        ExpectPlacedDisjoint(*p.plan);
        if (::testing::Test::HasFatalFailure()) return;
        const double ratio =
            static_cast<double>(p.plan->placement.extent) / static_cast<double>(l.extent);
        least = std::min(least, ratio);
        most = std::max(most, ratio);
        // And no more than the builders' with the owner-attention forms,
        // which the planner's per-owner attention replaces until slice G1b.
        const Legacy owners = PlanLegacy<F>(c, state, true, true);
        ASSERT_GT(owners.extent, 0U);
        EXPECT_LE(p.plan->placement.extent, owners.extent);
        served = std::max(served, static_cast<double>(p.plan->placement.extent) /
                                      static_cast<double>(owners.extent));
      }
      ++shapes;
    }
  }
  std::println(
      "{} shapes, {} fused steps, {} shared preparations; activation extent / PlaceActivations' "
      "from {:.4f} to {:.4f}; / the served recipe's at most {:.4f}",
      shapes, fused, shared, least, most, served);
  EXPECT_GT(fused, 0U);
  EXPECT_GT(shared, 0U);
}

TEST(PlannerGemmaTest, Gemma2PlansAsTodaysRecipe) {
  ExpectFamilyMatches<Gemma2Family>(kGemma2Policy);
}

TEST(PlannerGemmaTest, Gemma3PlansAsTodaysRecipe) {
  ExpectFamilyMatches<Gemma3Family>(kGemma3Policy);
}

// The two recipes differ as data: one policy text each, in its mapping,
// canonical.
TEST(PlannerGemmaTest, TheRecipesArePolicies) {
  const Policy g2 = ParsePolicy(kGemma2Policy).value();
  const Policy g3 = ParsePolicy(kGemma3Policy).value();
  EXPECT_EQ(ToText(g2), kGemma2Policy);
  EXPECT_EQ(ToText(g3), kGemma3Policy);
  EXPECT_FALSE(Disabled(g2, "owner_attention"));
  EXPECT_FALSE(Disabled(g3, "owner_attention"));
  EXPECT_TRUE(Disabled(g2, "rms_norm_mul_rope"));
  EXPECT_FALSE(Disabled(g3, "rms_norm_mul_rope"));
  EXPECT_TRUE(Disabled(g2, "grouped_stores"));
  EXPECT_FALSE(Disabled(g3, "grouped_stores"));
  EXPECT_NE(PatternsDigest(g2), PatternsDigest(g3));
  EXPECT_NE(Digest(g2), Digest(g3));
}

}  // namespace
}  // namespace llmp::planner
