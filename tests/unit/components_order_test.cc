// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Gemma 2 and Gemma 3 components' primitive steps (components/step.h
// BuildStep) against today's graph builders (kernels/ggml/gemma2_graph.h,
// gemma3_graph.h) with their default options: node for node in GGML's
// execution order (GraphOrder), the same operation, result type, extents,
// view strides, computed sources and the parameters that decide the
// arithmetic (norm epsilons, scales, rotations, attention's scale, softcap
// and accumulation, the GLU and the softcap's tanh), and every leaf a node
// reads: the same weight resource, the same layer's, owner's and tensor's
// cache (K or V), and the same step input (tokens, positions, output rows,
// each owner's global or ring cells and masks). The steps' own masks
// (kMaskBuild over a view of the positions) are inputs in the host-mask
// builds, checked for their window, and the device producer in the
// device-mask builds, checked for its rows, capacity, window and context.
// What the lowering (M3.6 slice I6) adds on top: binding, planning, the
// patterns.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <string>
#include <vector>

#include "components/step.h"
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/gemma2_graph.h"
#include "kernels/ggml/gemma3_graph.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/tensors.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec_fixtures.h"

namespace llmp::components {
namespace {

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
constexpr std::uint64_t kExtent = 2ULL << 20U;

std::optional<ir::Op> OpOf(const ggml_tensor* t) {
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
    case GGML_OP_CUSTOM:
      if (kg::LlmpOpOf(t) == kg::LlmpOp::kArgmax) return ir::Op::kArgmax;
      if (kg::LlmpOpOf(t) == kg::LlmpOp::kGemma4Mask) return ir::Op::kMaskBuild;
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

std::optional<format::Encoding> TypeOf(ggml_type t) {
  switch (t) {
    case GGML_TYPE_F32:
      return format::Encoding::kF32;
    case GGML_TYPE_F16:
      return format::Encoding::kF16;
    case GGML_TYPE_I32:
      return format::Encoding::kI32;
    default:
      return std::nullopt;
  }
}

template <typename T>
T Param(const ggml_tensor* t, std::size_t index) {
  T v{};
  std::memcpy(&v, reinterpret_cast<const char*>(t->op_params) + (index * 4), sizeof(T));
  return v;
}

// The IR nodes a GGML graph has: computed nodes, less the masks of a
// host-mask build and the views of the positions they read.
std::vector<std::size_t> Comparable(const ir::Graph& g, bool device_masks) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < g.nodes().size(); ++i) {
    const ir::Node& n = g.nodes()[i];
    const ir::Op op = n.key.op;
    if (op == ir::Op::kInput || op == ir::Op::kWeight || op == ir::Op::kState) continue;
    if (op == ir::Op::kMaskBuild && !device_masks) continue;
    if (op == ir::Op::kView) {
      const ir::TensorId src = n.src[0];
      const ir::Node& from = g.node(g.producer(src));
      if (from.key.op == ir::Op::kInput &&
          g.tensor(src).format.encoding == format::Encoding::kI32) {
        continue;  // a segment's positions, a mask's input
      }
    }
    out.push_back(i);
  }
  return out;
}

// What the legacy graph's leaves are: "tokens", "positions", "out_ids",
// "cells.<owner>.<global|local>", "mask.<owner>.<global|local>" (host
// masks), "k.<layer>.<owner>", "v.<layer>.<owner>" and "w.<resource>".
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
    add(seg.global_mask, std::format("mask.{}.global", s));
    add(seg.local_mask, std::format("mask.{}.local", s));
    for (std::size_t l = 0; l < seg.caches.size(); ++l) {
      add(seg.caches[l].first, std::format("k.{}.{}", l, s));
      add(seg.caches[l].second, std::format("v.{}.{}", l, s));
    }
  }
  for (const auto& w : g.weights) add(w.tensor, std::format("w.{}", w.resource.index));
  return out;
}

// The same names for the step's leaves and the tensors the comparison
// skips (host masks, the positions' views). `resources` names each
// resource's tensors (a fixture's table).
template <typename Resource>
std::map<std::uint32_t, std::string> StepLeaves(const StepGraph& step,
                                                const std::vector<Resource>& resources) {
  std::map<std::uint32_t, std::string> out;
  const auto where = [](const StepInput& in) {
    return std::format("{}.{}", in.segment, in.cells.ring ? "local" : "global");
  };
  std::map<std::uint32_t, const StepInput*> inputs;
  for (const StepInput& in : step.inputs) {
    inputs[in.tensor.v] = &in;
    switch (in.kind) {
      case InputKind::kTokens:
        out[in.tensor.v] = "tokens";
        break;
      case InputKind::kPositions:
        out[in.tensor.v] = "positions";
        break;
      case InputKind::kOutputRows:
        out[in.tensor.v] = "out_ids";
        break;
      case InputKind::kStoreCells:
        out[in.tensor.v] = "cells." + where(in);
        break;
      case InputKind::kCellPositions:
        out[in.tensor.v] = "held." + where(in);
        break;
      case InputKind::kStream:
        out[in.tensor.v] = "stream";
        break;
    }
  }
  for (const StepState& st : step.state) {
    out[st.tensor.v] =
        std::format("{}.{}.{}", st.ref.tensor == 0 ? "k" : "v", st.ref.layer, st.ref.segment);
  }
  for (const StepWeight& w : step.weights) {
    const std::string& name = w.parts.front().first;
    const auto at = std::ranges::find_if(
        resources, [&](const Resource& r) { return std::ranges::contains(r.roles, name); });
    out[w.tensor.v] = at == resources.end()
                          ? "w.?" + name
                          : std::format("w.{}", static_cast<std::size_t>(at - resources.begin()));
  }
  const ir::Graph& g = step.graph;
  for (const ir::Node& n : g.nodes()) {
    if (n.key.op == ir::Op::kMaskBuild) {
      const auto held = inputs.find(n.src[1].v);
      if (held != inputs.end()) out[n.out[0].v] = "mask." + where(*held->second);
    }
    if (n.key.op == ir::Op::kView) {
      const auto at = out.find(n.src[0].v);
      if (at != out.end() && at->second == "positions") out[n.out[0].v] = "positions";
    }
  }
  return out;
}

// The facts a mask is checked against.
struct MaskFacts {
  std::uint32_t window = 0;  // the ring layers'
  std::uint32_t context = 0;
  std::size_t segments = 0;
};

void ExpectSameSteps(const std::vector<ggml_tensor*>& nodes,
                     const std::map<const ggml_tensor*, std::string>& legacy_leaves,
                     const StepGraph& step, const std::map<std::uint32_t, std::string>& step_leaves,
                     bool device_masks, const MaskFacts& masks) {
  const ir::Graph& g = step.graph;
  const std::vector<std::size_t> ir_nodes = Comparable(g, device_masks);
  ASSERT_GT(nodes.size(), 400U);  // every layer's nodes, not an empty graph
  ASSERT_EQ(nodes.size(), ir_nodes.size());
  std::map<const ggml_tensor*, std::size_t> ggml_index;
  std::map<std::uint32_t, std::size_t> ir_index;  // node's first output -> position
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    ggml_index[nodes[i]] = i;
    ir_index[g.nodes()[ir_nodes[i]].out[0].v] = i;
  }
  std::map<std::uint32_t, const StepInput*> inputs;
  for (const StepInput& in : step.inputs) inputs[in.tensor.v] = &in;
  // Every host mask's window: the ring's (the local layers') or none.
  std::size_t host_masks = 0;
  for (const ir::Node& n : g.nodes()) {
    if (n.key.op != ir::Op::kMaskBuild) continue;
    const StepInput* held = inputs.at(n.src[1].v);
    const auto p = n.Params<ir::MaskParams>();
    EXPECT_EQ(p.kind, held->cells.ring ? ir::MaskKind::kWindow : ir::MaskKind::kCausal);
    EXPECT_EQ(p.window, held->cells.ring ? masks.window : 0U);
    ++host_masks;
  }
  EXPECT_EQ(host_masks, 2 * masks.segments);  // causal and window, per owner
  std::size_t leaves = 0;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const ggml_tensor* t = nodes[i];
    const ir::Node& n = g.nodes()[ir_nodes[i]];
    const ir::TensorDesc& out = g.tensor(n.out[0]);
    SCOPED_TRACE(std::format("node {}: {} ({})", i, ggml_op_desc(t), ggml_get_name(t)));
    ASSERT_EQ(OpOf(t), n.key.op);
    ASSERT_EQ(TypeOf(t->type), out.format.encoding);
    for (std::size_t d = 0; d < 4; ++d) {
      ASSERT_EQ(static_cast<std::uint64_t>(t->ne[d]), out.ne(d)) << "ne " << d;
      if (t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_RESHAPE) {
        ASSERT_EQ(static_cast<std::uint64_t>(t->nb[d]), out.nb.at(d)) << "nb " << d;
      }
    }
    if (t->op == GGML_OP_VIEW) EXPECT_EQ(t->view_offs, out.view_offset);
    // Sources, in order: a computed node by its position, a leaf by what it
    // is. The device producer reads the positions alone; the IR's mask also
    // reads its cells' positions, which the producer derives.
    std::vector<std::string> want;
    std::vector<std::string> got;
    for (const ggml_tensor* s : t->src) {
      if (s == nullptr) continue;
      if (const auto at = ggml_index.find(s); at != ggml_index.end()) {
        want.push_back(std::format("#{}", at->second));
      } else {
        const auto leaf = legacy_leaves.find(s);
        want.push_back(leaf != legacy_leaves.end() ? leaf->second : "?");
      }
    }
    auto sources = n.sources();
    if (n.key.op == ir::Op::kMaskBuild) sources = sources.first(1);
    for (const ir::TensorId s : sources) {
      if (!s.valid()) continue;
      if (const auto at = ir_index.find(s.v); at != ir_index.end()) {
        got.push_back(std::format("#{}", at->second));
      } else {
        const auto leaf = step_leaves.find(s.v);
        got.push_back(leaf != step_leaves.end() ? leaf->second : "?");
        ++leaves;
      }
    }
    // GGML's set_rows lists the rows, the indices, then the cache; the
    // IR's store the cache, the rows, then the indices.
    if (t->op == GGML_OP_SET_ROWS && want.size() == 3) std::ranges::rotate(want, want.end() - 1);
    ASSERT_EQ(want, got);
    // The parameters that decide the arithmetic, bit for bit.
    switch (n.key.op) {
      case ir::Op::kRmsNorm:
        EXPECT_EQ(Param<float>(t, 0), n.Params<ir::NormParams>().eps);
        break;
      case ir::Op::kScale: {
        const auto p = n.Params<ir::ScaleParams>();
        EXPECT_EQ(Param<float>(t, 0), p.scale);
        EXPECT_EQ(Param<float>(t, 1), p.bias);
        break;
      }
      case ir::Op::kRope: {
        const auto p = n.Params<ir::RopeParams>();
        EXPECT_EQ(Param<std::int32_t>(t, 1), static_cast<std::int32_t>(p.dims));
        EXPECT_EQ(Param<std::int32_t>(t, 2), GGML_ROPE_TYPE_NEOX);
        EXPECT_EQ(p.mode, ir::RopeMode::kNeox);
        EXPECT_EQ(Param<std::int32_t>(t, 4), static_cast<std::int32_t>(p.original_context));
        EXPECT_EQ(Param<float>(t, 5), p.base);
        EXPECT_EQ(Param<float>(t, 6), p.scale);
        EXPECT_EQ(Param<float>(t, 7), p.ext_factor);
        EXPECT_EQ(Param<float>(t, 8), p.attn_factor);
        EXPECT_EQ(Param<float>(t, 9), p.beta_fast);
        EXPECT_EQ(Param<float>(t, 10), p.beta_slow);
        break;
      }
      case ir::Op::kAttention: {
        const auto p = n.Params<ir::AttnParams>();
        EXPECT_EQ(Param<float>(t, 0), p.scale);
        EXPECT_EQ(Param<float>(t, 1), p.max_bias);
        EXPECT_EQ(Param<float>(t, 2), p.softcap);
        EXPECT_EQ(Param<std::int32_t>(t, 3), GGML_PREC_F32);
        EXPECT_EQ(p.accumulate, ir::Accumulate::kF32);
        break;
      }
      case ir::Op::kMaskBuild: {
        // The device producer: the owner's rows of the positions, its
        // cells' capacity, the window and the context.
        const auto p = n.Params<ir::MaskParams>();
        const ir::TensorDesc& rows = g.tensor(n.src[0]);
        const StepInput* held = inputs.at(n.src[1].v);
        EXPECT_EQ(kg::LlmpOpInt(t, 0), static_cast<std::int32_t>(rows.view_offset / 4));
        EXPECT_EQ(kg::LlmpOpInt(t, 1), static_cast<std::int32_t>(rows.ne(0)));
        EXPECT_EQ(kg::LlmpOpInt(t, 2), static_cast<std::int32_t>(held->cells.capacity));
        EXPECT_EQ(kg::LlmpOpInt(t, 3), static_cast<std::int32_t>(p.window));
        EXPECT_EQ(kg::LlmpOpInt(t, 4), static_cast<std::int32_t>(masks.context));
        break;
      }
      case ir::Op::kGlu:
        EXPECT_EQ(Param<std::int32_t>(t, 0), GGML_GLU_OP_GEGLU);
        EXPECT_EQ(n.Params<ir::GluParams>().kind, ir::GluKind::kGeGluTanh);
        break;
      case ir::Op::kUnary:
        EXPECT_EQ(ggml_get_unary_op(t), GGML_UNARY_OP_TANH);
        EXPECT_EQ(n.Params<ir::UnaryParams>().fn, ir::UnaryFn::kTanh);
        break;
      case ir::Op::kMatMul:
        EXPECT_EQ(n.Params<ir::ProductParams>().accumulate, ir::Accumulate::kDefault);
        break;
      default:
        break;
    }
  }
  EXPECT_GT(leaves, 0U);
}

spec::ModelSpec Gemma2Spec() {
  auto values = spec::mappings::Gemma2V0Values(md::Gemma2_2B());
  return *spec::Synthesize(spec::mappings::Gemma2Mapping(), test_support::Gemma2Source(), *values);
}
spec::ModelSpec Gemma3Spec() {
  auto values = spec::mappings::Gemma3V0Values(md::Gemma3_4BQat());
  return *spec::Synthesize(spec::mappings::Gemma3Mapping(), test_support::Gemma3Source(), *values);
}

struct Case {
  std::string name;
  std::vector<std::array<std::uint32_t, 5>> segments;  // slot, rows, past, global read, local read
  std::uint32_t outputs = 0;                           // per segment; 0: state-only
  bool greedy = false;
};

// Gemma 2's context is 8192 and its ring 4096 + 256 cells; Gemma 3's ring
// 1024 + 256. "wrapped greedy" and "sixteen owners" wrap both rings;
// "three decodes" reads past Gemma 2's context (Gemma 3 only).
std::vector<Case> Cases() {
  // Sixteen owners (the most a step joins): mixed rows, some rings wrapped.
  Case sixteen{.name = "sixteen owners", .segments = {}, .outputs = 1};
  for (std::uint32_t i = 0; i < 16; ++i) {
    const std::uint32_t rows = 1 + (i % 3);
    const std::uint32_t past = 397 * i;
    const std::uint32_t read = (past + rows + 255) / 256 * 256;
    sixteen.segments.push_back({i, rows, past, read, read});
  }
  return {
      sixteen,
      {.name = "prefill", .segments = {{0, 7, 0, 256, 256}}, .outputs = 1},
      {.name = "prefill outputs", .segments = {{0, 7, 0, 256, 256}}, .outputs = 3},
      {.name = "two owners",
       .segments = {{3, 3, 4351, 4608, 4352}, {1, 1, 7, 256, 256}},
       .outputs = 1},
      {.name = "three decodes",
       .segments = {{0, 1, 300, 512, 512}, {2, 1, 40, 256, 256}, {5, 1, 9000, 9216, 4352}},
       .outputs = 1,
       .greedy = true},
      {.name = "wrapped greedy",
       .segments = {{1, 1, 6000, 6144, 4352}, {0, 1, 1500, 1536, 1536}},
       .outputs = 1,
       .greedy = true},
      {.name = "state only", .segments = {{0, 16, 32, 256, 256}}, .outputs = 0},
      {.name = "state only, two owners",
       .segments = {{0, 16, 4400, 4608, 4352}, {1, 5, 0, 256, 256}},
       .outputs = 0},
  };
}

planner::StepShape ShapeOf(const Case& c) {
  planner::StepShape shape;
  for (const auto& s : c.segments) {
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(s[0]),
                              .rows = s[1],
                              .past = s[2],
                              .read_cells = s[3],
                              .outputs = c.outputs,
                              .greedy = c.greedy});
  }
  return shape;
}

template <typename GemmaShape>
GemmaShape LegacyShape(const Case& c) {
  GemmaShape shape;
  for (const auto& s : c.segments) shape.segments.push_back({s[0], s[1], s[2], s[3], s[4]});
  shape.outputs = c.outputs * static_cast<std::uint32_t>(c.segments.size());
  shape.output_mode =
      c.outputs == 0 ? decltype(shape.output_mode)::kStateOnly : decltype(shape.output_mode)::kHead;
  shape.greedy = c.greedy;
  return shape;
}

// The case's reads at the state's capacities: the step reads
// min(capacity, read) of each tensor, as the legacy shape names them.
template <typename State>
bool Fits(Case& c, const State& state) {
  for (auto& s : c.segments) {
    if (s[1] + s[2] > state.context) return false;
    s[3] = std::min(s[3], state.global_cells);
    s[4] = std::min(s[4], state.local_cells);
  }
  return true;
}

TEST(ComponentsOrderTest, Gemma2StepsAreTheBuildersGraphsNodeForNode) {
  const auto& p = md::Gemma2_2B();
  const auto resources = test_support::gemma2::Resources();
  const auto binding = md::BindGemma2(p, "gemma2", resources);
  ASSERT_TRUE(binding.has_value());
  const auto spec = Gemma2Spec();
  const auto source = test_support::Gemma2Source();
  const auto state = md::Gemma2State(p, p.context, 256);
  ASSERT_TRUE(state.has_value());
  const auto layout = StageLayout(spec, 0, p.context, 256, kExtent);
  ASSERT_TRUE(layout.has_value());
  std::size_t compared = 0;
  for (const bool device_masks : {false, true}) {
    for (const Case& c : Cases()) {
      SCOPED_TRACE(std::format("{}{}", c.name, device_masks ? ", device masks" : ""));
      auto legacy_case = c;
      if (!Fits(legacy_case, *state)) continue;
      auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(p, c.segments.size()));
      ASSERT_TRUE(arena.has_value());
      auto legacy = kg::BuildGemma2Graph(*arena, p, *binding, *state,
                                         LegacyShape<kg::Gemma2ChunkShape>(legacy_case),
                                         {.device_masks = device_masks});
      ASSERT_TRUE(legacy.has_value()) << legacy.error().detail;
      auto step =
          BuildStep(spec, 0, ShapeOf(legacy_case), {.tensors = source.tensors, .state = &*layout});
      ASSERT_TRUE(step.has_value()) << step.error();
      ExpectSameSteps(
          legacy->nodes, LegacyLeaves(*legacy), *step, StepLeaves(*step, resources), device_masks,
          {.window = p.window, .context = state->context, .segments = c.segments.size()});
      ++compared;
    }
  }
  EXPECT_EQ(compared, 2 * (Cases().size() - 1));
}

TEST(ComponentsOrderTest, Gemma3StepsAreTheBuildersGraphsNodeForNode) {
  const auto& p = md::Gemma3_4BQat();
  const auto resources = test_support::gemma3::Resources();
  const auto binding = md::BindGemma3(p, "gemma3", resources);
  ASSERT_TRUE(binding.has_value());
  const auto spec = Gemma3Spec();
  const auto source = test_support::Gemma3Source();
  const auto state = md::Gemma3State(p, 16384, 256);
  ASSERT_TRUE(state.has_value());
  const auto layout = StageLayout(spec, 0, 16384, 256, kExtent);
  ASSERT_TRUE(layout.has_value());
  std::size_t compared = 0;
  for (const bool device_masks : {false, true}) {
    for (const Case& c : Cases()) {
      SCOPED_TRACE(std::format("{}{}", c.name, device_masks ? ", device masks" : ""));
      auto legacy_case = c;
      ASSERT_TRUE(Fits(legacy_case, *state));
      auto arena = kg::TensorArena::Create(kg::Gemma3GraphTensors(p, c.segments.size()));
      ASSERT_TRUE(arena.has_value());
      auto legacy = kg::BuildGemma3Graph(*arena, p, *binding, *state,
                                         LegacyShape<kg::Gemma3ChunkShape>(legacy_case),
                                         {.device_masks = device_masks});
      ASSERT_TRUE(legacy.has_value()) << legacy.error().detail;
      auto step =
          BuildStep(spec, 0, ShapeOf(legacy_case), {.tensors = source.tensors, .state = &*layout});
      ASSERT_TRUE(step.has_value()) << step.error();
      ExpectSameSteps(
          legacy->nodes, LegacyLeaves(*legacy), *step, StepLeaves(*step, resources), device_masks,
          {.window = p.window, .context = state->context, .segments = c.segments.size()});
      ++compared;
    }
  }
  EXPECT_EQ(compared, 2 * Cases().size());
}

}  // namespace
}  // namespace llmp::components
