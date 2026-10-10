// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Gemma 4 components' primitive steps (components/step.h BuildStep)
// against today's graph builders (kernels/ggml/gemma4_graph.h,
// gemma4_assistant_graph.h) with their default (non-fused) options, node
// for node in GGML's execution order (GraphOrder): the same operation,
// result type, extents, view strides, computed sources and the parameters
// that decide the arithmetic, and every leaf a node reads (the weight
// resource, the layer's, owner's and tensor's cache, the step input), as
// components_order_test does for Gemma 2 and Gemma 3, with host and device
// masks; for the 26B-A4B, 31B and their assistants (draft steps).
//
// What differs, by design, and how the comparison takes it: the router's
// selection and the experts' reduce are the op set's composite ops
// (ir::kRoute, ir::kExpertReduce, which a plan runs as their decompositions
// or as an implementation of their own, and partitioning splits); each
// stands for the run of GGML nodes its decomposition is (soft_max through
// the weights' reshape; the two products by each slot's scale, gathered
// before it as the builders gather it, and weight, the slot views and the
// ordered adds), compared by the run's position, its outside sources (the
// reduce's ids, which only name its slots' experts, aside) and the
// parameters it carries. Everything else is node for node.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "components/step.h"
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/gemma4_assistant_graph.h"
#include "kernels/ggml/gemma4_graph.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/tensors.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma4.h"
#include "spec_fixtures.h"

namespace llmp::components {
namespace {

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
constexpr std::uint64_t kExtent = 2ULL << 20U;
constexpr std::uint32_t kContext = 8192;
constexpr std::uint32_t kMaxRows = 64;

std::optional<ir::Op> OpOf(const ggml_tensor* t) {
  switch (t->op) {
    case GGML_OP_GET_ROWS:
      return ir::Op::kGetRows;
    case GGML_OP_REPEAT:
      return ir::Op::kRepeat;
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
    case GGML_OP_MUL_MAT_ID:
      return ir::Op::kMatMulId;
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

// A legacy entry: one GGML node, or a run of them a composite IR op
// stands for (its first and last node, and which nodes are its outputs).
struct Entry {
  std::size_t first = 0;
  std::size_t last = 0;
  std::optional<ir::Op> composite{};
  std::vector<std::size_t> outputs{};  // node indices, by the IR op's outputs
};

// The legacy nodes as entries: each Gemma router's run (soft_max through
// the reshape named "expert_weights") and each reduce's run (the product
// named "expert_scaled_down" through the last ordered add) collapsed.
std::vector<Entry> Entries(const std::vector<ggml_tensor*>& nodes, std::uint32_t used) {
  std::vector<Entry> out;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const std::string name = ggml_get_name(nodes[i]);
    if (nodes[i]->op == GGML_OP_SOFT_MAX && name.ends_with("router_probabilities")) {
      // soft_max, reshape, argsort, view (ids), get_rows, reshape, sum_rows,
      // clamp, div, reshape (weights).
      std::size_t ids = i + 3;
      std::size_t weights = i + 9;
      EXPECT_TRUE(std::string(ggml_get_name(nodes.at(ids))).ends_with("expert_ids"));
      EXPECT_TRUE(std::string(ggml_get_name(nodes.at(weights))).ends_with("expert_weights"));
      EXPECT_EQ(nodes.at(i + 2)->op, GGML_OP_ARGSORT);
      EXPECT_EQ(Param<std::int32_t>(nodes.at(i + 2), 0),
                static_cast<std::int32_t>(GGML_SORT_ORDER_DESC));
      EXPECT_EQ(nodes.at(ids)->ne[0], static_cast<std::int64_t>(used));
      EXPECT_EQ(nodes.at(i + 7)->op, GGML_OP_CLAMP);
      EXPECT_EQ(Param<float>(nodes.at(i + 7), 0), 0x1p-14F);
      out.push_back(
          {.first = i, .last = weights, .composite = ir::Op::kRoute, .outputs = {ids, weights}});
      i = weights;
      continue;
    }
    if (nodes[i]->op == GGML_OP_MUL && name.ends_with("expert_scaled_down")) {
      // mul (by each slot's scale, gathered before it by the reshape,
      // repeat and get_rows the components emit too), mul (by weight), a
      // view a slot, an add a slot after the first.
      const std::size_t last = i + (2 * used);
      EXPECT_EQ(nodes.at(i + 1)->op, GGML_OP_MUL);
      EXPECT_EQ(nodes.at(i + 2)->op, GGML_OP_VIEW);
      EXPECT_EQ(nodes.at(last)->op, GGML_OP_ADD);
      out.push_back(
          {.first = i, .last = last, .composite = ir::Op::kExpertReduce, .outputs = {last}});
      i = last;
      continue;
    }
    out.push_back({.first = i, .last = i, .composite = std::nullopt, .outputs = {i}});
  }
  return out;
}

// The facts a mask is checked against.
struct MaskFacts {
  std::uint32_t window = 0;
  std::uint32_t context = 0;
  std::size_t masks = 0;  // host masks the step has
};

void ExpectSameSteps(const std::vector<ggml_tensor*>& nodes,
                     const std::map<const ggml_tensor*, std::string>& legacy_leaves,
                     const StepGraph& step, const std::map<std::uint32_t, std::string>& step_leaves,
                     bool device_masks, const MaskFacts& masks, std::uint32_t used) {
  const ir::Graph& g = step.graph;
  const std::vector<std::size_t> ir_nodes = Comparable(g, device_masks);
  const std::vector<Entry> entries = Entries(nodes, used);
  ASSERT_GT(nodes.size(), 100U);
  ASSERT_EQ(entries.size(), ir_nodes.size());
  // Each legacy node's name as a source: its entry's position, with the
  // output index for a composite run's outputs.
  std::map<const ggml_tensor*, std::string> legacy_name;
  for (std::size_t e = 0; e < entries.size(); ++e) {
    const Entry& entry = entries[e];
    for (std::size_t k = entry.first; k <= entry.last; ++k) {
      legacy_name[nodes[k]] = std::format("#{}.inside", e);
    }
    for (std::size_t o = 0; o < entry.outputs.size(); ++o) {
      legacy_name[nodes[entry.outputs[o]]] =
          entry.outputs.size() == 1 ? std::format("#{}", e) : std::format("#{}.{}", e, o);
    }
  }
  std::map<std::uint32_t, std::string> ir_name;
  for (std::size_t i = 0; i < ir_nodes.size(); ++i) {
    const ir::Node& n = g.nodes()[ir_nodes[i]];
    for (std::size_t o = 0; o < n.nout; ++o) {
      ir_name[n.out[o].v] = n.nout == 1 ? std::format("#{}", i) : std::format("#{}.{}", i, o);
    }
  }
  std::map<std::uint32_t, const StepInput*> inputs;
  for (const StepInput& in : step.inputs) inputs[in.tensor.v] = &in;
  std::size_t host_masks = 0;
  for (const ir::Node& n : g.nodes()) {
    if (n.key.op != ir::Op::kMaskBuild) continue;
    const StepInput* held = inputs.at(n.src[1].v);
    const auto p = n.Params<ir::MaskParams>();
    EXPECT_EQ(p.kind, held->cells.ring ? ir::MaskKind::kWindow : ir::MaskKind::kCausal);
    EXPECT_EQ(p.window, held->cells.ring ? masks.window : 0U);
    ++host_masks;
  }
  EXPECT_EQ(host_masks, masks.masks);
  const auto source_of = [&](const ggml_tensor* s) -> std::string {
    if (const auto at = legacy_name.find(s); at != legacy_name.end()) return at->second;
    const auto leaf = legacy_leaves.find(s);
    return leaf != legacy_leaves.end() ? leaf->second : "?";
  };
  std::size_t leaves = 0;
  std::size_t composites = 0;
  for (std::size_t e = 0; e < entries.size(); ++e) {
    const Entry& entry = entries[e];
    const ir::Node& n = g.nodes()[ir_nodes[e]];
    std::vector<std::string> got;
    auto sources = n.sources();
    if (n.key.op == ir::Op::kMaskBuild) sources = sources.first(1);
    for (std::size_t k = 0; k < sources.size(); ++k) {
      const ir::TensorId s = sources[k];
      if (!s.valid()) continue;
      // A reduce's ids name its slots' experts (expert parallelism); its
      // run reads them only through the scales' gather before it.
      if (n.key.op == ir::Op::kExpertReduce && k == 2) continue;
      if (const auto at = ir_name.find(s.v); at != ir_name.end()) {
        got.push_back(at->second);
      } else {
        const auto leaf = step_leaves.find(s.v);
        got.push_back(leaf != step_leaves.end() ? leaf->second : "?");
        ++leaves;
      }
    }
    if (entry.composite) {
      // A run: the IR op, its sources those the run reads from outside it
      // (as a set: the decomposition reads them in its own order).
      SCOPED_TRACE(std::format("entry {}: {} run of {} nodes", e, ir::OpName(*entry.composite),
                               entry.last - entry.first + 1));
      ASSERT_EQ(n.key.op, *entry.composite);
      std::set<std::string> want;
      for (std::size_t k = entry.first; k <= entry.last; ++k) {
        for (const ggml_tensor* s : nodes[k]->src) {
          if (s == nullptr) continue;
          const std::string name = source_of(s);
          if (!name.starts_with(std::format("#{}", e))) want.insert(name);
        }
      }
      EXPECT_EQ(want, std::set<std::string>(got.begin(), got.end()));
      if (n.key.op == ir::Op::kRoute) {
        const auto p = n.Params<ir::RouteParams>();
        EXPECT_EQ(p.kind, ir::RouteKind::kSoftmaxTopK);
        EXPECT_TRUE(p.normalize);
        EXPECT_EQ(p.used, used);
        EXPECT_EQ(p.scale, 1.0F);
        EXPECT_EQ(static_cast<std::uint64_t>(nodes[entry.first]->ne[0]), p.experts);
      }
      // Its outputs: the run's, extent for extent.
      for (std::size_t o = 0; o < entry.outputs.size(); ++o) {
        const ggml_tensor* t = nodes[entry.outputs[o]];
        const ir::TensorDesc& out = g.tensor(n.out[o]);
        EXPECT_EQ(static_cast<std::uint64_t>(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]),
                  out.ne(0) * out.ne(1) * out.ne(2) * out.ne(3));
        EXPECT_EQ(TypeOf(t->type), out.format.encoding);
      }
      ++composites;
      continue;
    }
    const ggml_tensor* t = nodes[entry.first];
    const ir::TensorDesc& out = g.tensor(n.out[0]);
    SCOPED_TRACE(std::format("entry {}: {} ({})", e, ggml_op_desc(t), ggml_get_name(t)));
    ASSERT_EQ(OpOf(t), n.key.op);
    ASSERT_EQ(TypeOf(t->type), out.format.encoding);
    for (std::size_t d = 0; d < 4; ++d) {
      ASSERT_EQ(static_cast<std::uint64_t>(t->ne[d]), out.ne(d)) << "ne " << d;
      if (t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_RESHAPE) {
        ASSERT_EQ(static_cast<std::uint64_t>(t->nb[d]), out.nb.at(d)) << "nb " << d;
      }
    }
    if (t->op == GGML_OP_VIEW) {
      EXPECT_EQ(t->view_offs, out.view_offset);
    }
    std::vector<std::string> want;
    for (const ggml_tensor* s : t->src) {
      if (s != nullptr) want.push_back(source_of(s));
    }
    // GGML's set_rows lists the rows, the indices, then the cache; the
    // IR's store the cache, the rows, then the indices.
    if (t->op == GGML_OP_SET_ROWS && want.size() == 3) std::ranges::rotate(want, want.end() - 1);
    ASSERT_EQ(want, got);
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
  // A routed step's runs were compared (a router and a reduce a layer that
  // computes its feed-forward block).
  if (used != 0) {
    EXPECT_GT(composites, 0U);
    EXPECT_EQ(composites % 2, 0U);
  } else {
    EXPECT_EQ(composites, 0U);
  }
}

// The legacy weight leaves' names: "w.<kind><index>" (o: ordinary, a: an
// expert array), as BindGemma4 numbers them; "w.target" for the
// assistant's view of its target's table.
std::string WeightName(bool array, std::uint32_t index) {
  return std::format("w.{}{}", array ? 'a' : 'o', index);
}

// Each fixture resource's legacy name, by its tensor name.
std::map<std::string, std::string> ResourceNames(const std::vector<md::Gemma4Resource>& resources) {
  std::map<std::string, std::string> out;
  std::uint32_t ordinary = 0;
  std::uint32_t arrays = 0;
  for (const md::Gemma4Resource& r : resources) {
    out[r.roles.front()] = WeightName(r.expert_array, r.expert_array ? arrays++ : ordinary++);
  }
  return out;
}

std::map<const ggml_tensor*, std::string> LegacyLeaves(const kg::Gemma4Graph& g) {
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
  for (const auto& w : g.weights) {
    add(w.tensor, WeightName(w.resource.expert_array, w.resource.index));
  }
  return out;
}

std::map<std::uint32_t, std::string> StepLeaves(const StepGraph& step,
                                                const std::map<std::string, std::string>& names,
                                                const model::StateLayoutDerived* target) {
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
      case InputKind::kFeatures:
        out[in.tensor.v] = "features";
        break;
      default:
        out[in.tensor.v] = "?";
        break;
    }
  }
  for (const StepState& st : step.state) {
    if (st.borrowed) {
      // The target's tensor it reads: its kind and owner.
      const auto& item = target->items.at(st.item);
      out[st.tensor.v] =
          std::format("{}.{}.{}", item.contract.name,
                      item.contract.window_or_ratio != 0 ? "local" : "global", st.ref.segment);
      continue;
    }
    out[st.tensor.v] =
        std::format("{}.{}.{}", st.ref.tensor == 0 ? "k" : "v", st.ref.layer, st.ref.segment);
  }
  for (const StepWeight& w : step.weights) {
    const std::string& name = w.parts.front().first;
    if (w.ref.stage != 0 && target != nullptr) {
      // Not the target's: the assistant's own.
      const auto at = names.find(name);
      out[w.tensor.v] = at == names.end() ? "w.?" + name : at->second;
      continue;
    }
    if (target != nullptr) {
      out[w.tensor.v] = "w.target";
      continue;
    }
    const auto at = names.find(name);
    out[w.tensor.v] = at == names.end() ? "w.?" + name : at->second;
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

spec::ModelSpec Gemma4Spec(std::uint32_t size) {
  const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
  auto values = spec::mappings::Gemma4V0Values(p);
  return *spec::Synthesize(spec::mappings::Gemma4Mapping(), test_support::Gemma4Source(size),
                           *values);
}

struct Case {
  std::string name;
  std::vector<std::array<std::uint32_t, 5>> segments;  // slot, rows, past, global read, local read
  std::uint32_t outputs = 0;                           // per segment; 0: state-only
  bool greedy = false;
};

// The layout's ring holds 1,280 cells (the window and 64 rows, padded);
// "wrapped greedy", "two owners" and "sixteen owners" wrap it.
std::vector<Case> Cases() {
  Case sixteen{.name = "sixteen owners", .segments = {}, .outputs = 1};
  for (std::uint32_t i = 0; i < 16; ++i) {
    const std::uint32_t rows = 1 + (i % 3);
    const std::uint32_t past = 397 * i;
    const std::uint32_t read = (past + rows + 255) / 256 * 256;
    sixteen.segments.push_back({i, rows, past, read, std::min(read, 1280U)});
  }
  return {
      sixteen,
      {.name = "prefill", .segments = {{0, 7, 0, 256, 256}}, .outputs = 1},
      {.name = "prefill outputs", .segments = {{0, 7, 0, 256, 256}}, .outputs = 3},
      {.name = "two owners",
       .segments = {{3, 3, 1500, 1792, 1280}, {1, 1, 7, 256, 256}},
       .outputs = 1},
      {.name = "three decodes",
       .segments = {{0, 1, 300, 512, 512}, {2, 1, 40, 256, 256}, {5, 1, 5000, 5120, 1280}},
       .outputs = 1,
       .greedy = true},
      {.name = "wrapped greedy",
       .segments = {{1, 1, 6000, 6144, 1280}, {0, 1, 1300, 1536, 1280}},
       .outputs = 1,
       .greedy = true},
      {.name = "state only", .segments = {{0, 16, 32, 256, 256}}, .outputs = 0},
      {.name = "state only, two owners",
       .segments = {{0, 16, 4400, 4608, 1280}, {1, 5, 0, 256, 256}},
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

kg::Gemma4ChunkShape LegacyShape(const Case& c) {
  kg::Gemma4ChunkShape shape;
  for (const auto& s : c.segments) shape.segments.push_back({s[0], s[1], s[2], s[3], s[4]});
  shape.outputs = c.outputs * static_cast<std::uint32_t>(c.segments.size());
  shape.output_mode =
      c.outputs == 0 ? kg::Gemma4OutputMode::kStateOnly : kg::Gemma4OutputMode::kHead;
  shape.greedy = c.greedy;
  return shape;
}

class ComponentsGemma4Order : public testing::TestWithParam<std::uint32_t> {};

TEST_P(ComponentsGemma4Order, StepsAreTheBuildersGraphsNodeForNode) {
  const std::uint32_t size = GetParam();
  const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
  const auto resources = test_support::gemma4::Resources(size);
  const auto binding = md::BindGemma4(p, "gemma4", resources);
  ASSERT_TRUE(binding.has_value()) << binding.error();
  const auto spec = Gemma4Spec(size);
  const auto source = test_support::Gemma4Source(size);
  const auto state = md::Gemma4State(p, kContext, kMaxRows);
  ASSERT_TRUE(state.has_value());
  const auto layout = StageLayout(spec, 0, kContext, kMaxRows, kExtent);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  ASSERT_EQ(state->local_cells, 1280U);
  const auto names = ResourceNames(resources);
  std::size_t compared = 0;
  for (const bool device_masks : {false, true}) {
    for (const Case& c : Cases()) {
      SCOPED_TRACE(std::format("{}{}", c.name, device_masks ? ", device masks" : ""));
      auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(p, c.segments.size()));
      ASSERT_TRUE(arena.has_value());
      kg::Gemma4GraphOptions options;
      options.device_masks = device_masks;
      options.narrow_final = true;  // the runners' frontier_head: the components narrow
      auto legacy = kg::BuildGemma4Graph(*arena, p, *binding, *state, LegacyShape(c), options);
      ASSERT_TRUE(legacy.has_value()) << legacy.error().detail;
      auto step = BuildStep(spec, 0, ShapeOf(c), {.tensors = source.tensors, .state = &*layout});
      ASSERT_TRUE(step.has_value()) << step.error();
      ExpectSameSteps(
          legacy->nodes, LegacyLeaves(*legacy), *step, StepLeaves(*step, names, nullptr),
          device_masks,
          {.window = p.window, .context = state->context, .masks = 2 * c.segments.size()},
          p.experts_used);
      ++compared;
    }
  }
  EXPECT_EQ(compared, 2 * Cases().size());
}

INSTANTIATE_TEST_SUITE_P(Gemma4, ComponentsGemma4Order, testing::Values(26U, 31U),
                         [](const testing::TestParamInfo<std::uint32_t>& info) {
                           return std::format("gemma{}", info.param);
                         });

// The assistant's draft steps against today's assistant graph: one query
// an owner at its target's committed endpoint, over the target's last
// local and global caches (host masks: the builder's are inputs).
TEST(ComponentsGemma4OrderTest, AssistantStepsAreTheBuildersGraphsNodeForNode) {
  for (const std::uint32_t size : {26U, 31U}) {
    SCOPED_TRACE(size);
    const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
    const auto& a = size == 26 ? md::Gemma4Assistant26() : md::Gemma4Assistant31();
    const auto target_resources = test_support::gemma4::Resources(size);
    const auto assistant_resources = test_support::gemma4::AssistantResources(size);
    const auto target_binding = md::BindGemma4(p, "gemma4", target_resources);
    const auto assistant_binding =
        md::BindGemma4Assistant(a, "gemma4-assistant", assistant_resources);
    ASSERT_TRUE(target_binding.has_value() && assistant_binding.has_value());
    const auto values = spec::mappings::Gemma4AssistantV0Values(a, p);
    ASSERT_TRUE(values.has_value());
    const auto spec = spec::ComposeDrafter(Gemma4Spec(size), spec::mappings::Gemma4Mapping(),
                                           spec::mappings::Gemma4AssistantMapping(),
                                           test_support::Gemma4AssistantSource(size), *values);
    ASSERT_TRUE(spec.has_value()) << spec.error();
    const auto target_source = test_support::Gemma4Source(size);
    const auto assistant_source = test_support::Gemma4AssistantSource(size);
    const std::array<std::span<const BoundTensor>, 2> artifacts{target_source.tensors,
                                                                assistant_source.tensors};
    const auto state = md::Gemma4State(p, kContext, kMaxRows);
    const auto target_layout = StageLayout(*spec, 0, kContext, kMaxRows, kExtent);
    const auto drafter_layout = StageLayout(*spec, 1, kContext, kMaxRows, kExtent);
    ASSERT_TRUE(state && target_layout && drafter_layout);
    const std::array<const model::StateLayoutDerived*, 2> layouts{&*target_layout,
                                                                  &*drafter_layout};
    // Each assistant resource's legacy name (the assistant binding's index).
    std::map<std::string, std::string> names;
    for (std::uint32_t i = 0; i < assistant_resources.size(); ++i) {
      names[assistant_resources[i].roles.front()] = WeightName(false, i);
    }
    for (const auto& prefixes :
         std::vector<std::vector<std::uint32_t>>{{300}, {300, 1500}, {7, 2048, 6000}}) {
      SCOPED_TRACE(prefixes.size());
      kg::Gemma4AssistantShape legacy_shape;
      planner::StepShape shape{.stage = 1, .segments = {}};
      for (std::size_t i = 0; i < prefixes.size(); ++i) {
        const std::uint32_t read = (prefixes[i] + 255) / 256 * 256;
        legacy_shape.segments.push_back({.slot = static_cast<std::uint32_t>(2 * i),
                                         .prefix = prefixes[i],
                                         .local_n_kv = std::min(read, state->local_cells),
                                         .global_n_kv = read});
        shape.segments.push_back({.slot = static_cast<std::uint8_t>(2 * i),
                                  .rows = 1,
                                  .past = prefixes[i],
                                  .read_cells = read,
                                  .outputs = 1});
      }
      auto arena = kg::TensorArena::Create(kg::Gemma4AssistantGraphTensors(prefixes.size()));
      ASSERT_TRUE(arena.has_value());
      auto legacy = kg::BuildGemma4AssistantGraph(*arena, a, *assistant_binding, p, *target_binding,
                                                  *state, legacy_shape);
      ASSERT_TRUE(legacy.has_value()) << legacy.error().detail;
      auto step = BuildStep(*spec, 1, shape,
                            {.tensors = assistant_source.tensors,
                             .state = &*drafter_layout,
                             .stages = layouts,
                             .artifacts = artifacts});
      ASSERT_TRUE(step.has_value()) << step.error();
      std::map<const ggml_tensor*, std::string> leaves;
      leaves[legacy->tokens] = "tokens";
      leaves[legacy->features] = "features";
      leaves[legacy->positions] = "positions";
      leaves[legacy->embedding] = "w.target";
      for (std::size_t s = 0; s < legacy->segments.size(); ++s) {
        const auto& seg = legacy->segments[s];
        for (std::size_t kind = 0; kind < 2; ++kind) {
          const char* type = kind == 0 ? "local" : "global";
          leaves[seg.caches[kind].first] = std::format("k.{}.{}", type, s);
          leaves[seg.caches[kind].second] = std::format("v.{}.{}", type, s);
          leaves[seg.masks[kind]] = std::format("mask.{}.{}", s, type);
        }
      }
      for (const auto& w : legacy->weights) leaves[w.tensor] = WeightName(false, w.resource.index);
      ExpectSameSteps(legacy->nodes, leaves, *step, StepLeaves(*step, names, &*target_layout),
                      false,
                      {.window = p.window, .context = kContext, .masks = 2 * prefixes.size()}, 0);
    }
  }
}

// The mappings' default policies are policies the planner takes.
TEST(ComponentsGemma4OrderTest, DefaultPoliciesParse) {
  for (const spec::Mapping* m : spec::Mappings()) {
    for (const spec::PolicyRule& rule : m->policies) {
      const auto parsed = planner::ParsePolicy(rule.text);
      EXPECT_TRUE(parsed.has_value()) << m->architecture << ": " << parsed.error();
    }
  }
}

}  // namespace
}  // namespace llmp::components
