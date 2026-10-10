// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen3.8 components' primitive steps (components/step.h BuildStep,
// the qwen4exp mapping's spec) against today's graph builder's unfused
// form (kernels/ggml/qwen38_graph.h BuildQwen38Graph, Qwen38GraphOptions
// fused false), node for node in GGML's execution order (GraphOrder): the
// same operation, result type, extents, view strides, computed sources and
// the parameters that decide the arithmetic, and every leaf a node reads
// (the weight resource, the layer's state tensor, the step input), for a
// GGUF quantization's tensors and the ModelOpt import's (GGML's NVFP4
// experts; the unfused form takes no CUTLASS layout), at prefill and
// decode shapes whose attention reads no more cells than the indexer's
// budget (below it the selection keeps every cell and neither graph
// builds it).
//
// What differs, by design, and how the comparison takes it:
//   - the router's selection and the experts' reduce are the op set's
//     composite ops (ir::kRoute, ir::kExpertReduce): each stands for the
//     run of GGML nodes its decomposition is (soft_max through the
//     weights' reshape; the products by each slot's scale, where the
//     experts have one, and weight, the slot views and the ordered adds),
//     compared by the run's position, its outside sources and the
//     parameters it carries, as components_gemma4_order_test does;
//   - the gated delta rule is one node of two outputs where GGML views its
//     one result twice: the views are the node's outputs;
//   - an MXFP8 product is one node where the unfused form dequantizes then
//     multiplies past the vector product's 8 rows, and reads its weight's
//     parts (codes and scales) as one leaf; the n-gram table's NVFP4 rows
//     likewise (its tensor scale a part of the table);
//   - the shared expert's gate is the gate row times each row, summed
//     (where GGML reads the weight's row through a get_rows of row 0 first,
//     which its products' refusal of a one-row output needs);
//   - masks are the framework's (the unfused form's host masks are inputs).
// The fast form's fusions (llmp.hc.*, gdn.*, qsa.*, moe.*, the MXFP8
// tensor-core products) are the planner's patterns, slice G3b.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/qwen38_graph.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen38.h"
#include "planner/step_shape.h"
#include "qwen38_fixture.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"

namespace llmp::components {
namespace {

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace fx = llmp::test_support::qwen38;

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
    case GGML_OP_TRANSPOSE:
      return ir::Op::kPermute;
    case GGML_OP_CONT:
      return ir::Op::kCont;
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
    case GGML_OP_SQRT:
      return ir::Op::kUnary;
    case GGML_OP_SUM_ROWS:
      return ir::Op::kSumRows;
    case GGML_OP_CLAMP:
      return ir::Op::kClamp;
    case GGML_OP_SSM_CONV:
      return ir::Op::kSsmConv;
    case GGML_OP_GATED_DELTA_NET:
      return ir::Op::kGatedDeltaNet;
    case GGML_OP_CUSTOM:
      switch (kg::LlmpOpOf(t)) {
        case kg::LlmpOp::kArgmax:
          return ir::Op::kArgmax;
        case kg::LlmpOp::kMxfp8MulMatVec:
          return ir::Op::kMatMul;
        case kg::LlmpOp::kNvfp4Rows:
          return ir::Op::kGetRows;
        default:
          return std::nullopt;
      }
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
    case GGML_TYPE_I64:
      return format::Encoding::kI64;
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

bool IsDequant(const ggml_tensor* t) {
  return t->op == GGML_OP_CUSTOM && kg::LlmpOpOf(t) == kg::LlmpOp::kMxfp8Dequant;
}

// The IR nodes a GGML graph has: computed nodes, less the masks of a
// host-mask build and the views of the positions they read.
std::vector<std::size_t> Comparable(const ir::Graph& g, const std::set<std::size_t>& skip) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < g.nodes().size(); ++i) {
    const ir::Node& n = g.nodes()[i];
    const ir::Op op = n.key.op;
    if (skip.contains(i)) continue;
    if (op == ir::Op::kInput || op == ir::Op::kWeight || op == ir::Op::kState) continue;
    if (op == ir::Op::kMaskBuild) continue;
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

// The nodes that keep the QSA layers' block keys (their pooled, normed,
// rotated keys stored at each complete block's cell every step): the
// unfused builder pools at a selecting step alone. Each store into an
// indexer_blocks state and the nodes that feed only it, back to the raw
// keys' store.
std::set<std::size_t> BlockKeyNodes(const ir::Graph& g,
                                    const std::map<std::uint32_t, std::string>& step_leaves) {
  std::vector<std::size_t> uses(g.tensor_count(), 0);
  for (const ir::Node& n : g.nodes()) {
    for (const ir::TensorId s : n.sources()) {
      if (s.valid()) ++uses[s.v];
    }
  }
  std::set<std::size_t> out;
  std::vector<std::size_t> todo;
  for (std::size_t i = 0; i < g.nodes().size(); ++i) {
    const ir::Node& n = g.nodes()[i];
    if (n.key.op != ir::Op::kCacheStore) continue;
    const auto at = step_leaves.find(n.src[0].v);
    if (at != step_leaves.end() && at->second.starts_with("indexer_blocks.")) {
      out.insert(i);
      todo.push_back(i);
    }
  }
  // A producer joins once every use of its outputs is a joined node's.
  std::vector<std::size_t> used_by_set(g.tensor_count(), 0);
  while (!todo.empty()) {
    const ir::Node& n = g.nodes()[todo.back()];
    todo.pop_back();
    for (const ir::TensorId s : n.sources()) {
      if (!s.valid()) continue;
      ++used_by_set[s.v];
      const std::size_t p = g.producer(s).v;
      const ir::Op op = g.nodes()[p].key.op;
      if (op == ir::Op::kInput || op == ir::Op::kWeight || op == ir::Op::kState ||
          op == ir::Op::kCacheStore || out.contains(p)) {
        continue;
      }
      bool all = true;
      for (const ir::TensorId o : g.nodes()[p].outputs()) {
        if (o.valid() && used_by_set[o.v] != uses[o.v]) all = false;
      }
      if (all) {
        out.insert(p);
        todo.push_back(p);
      }
    }
  }
  return out;
}

// A legacy entry: one GGML node, a run a composite IR op stands for, or a
// dequantization and the product it feeds.
struct Entry {
  std::size_t first = 0;
  std::size_t last = 0;
  std::optional<ir::Op> composite{};
  std::vector<std::size_t> outputs{};  // node indices, by the IR op's outputs
};

// The legacy nodes as entries, and the nodes the IR has no node for (the
// gated delta rule's views: aliases of its outputs; the shared gate's
// row: an alias of its weight).
struct Legacy {
  std::vector<Entry> entries;
  std::map<const ggml_tensor*, std::string> aliases;  // a skipped node's name as a source
};

Legacy Entries(const std::vector<ggml_tensor*>& nodes, std::uint32_t used, bool scaled,
               const std::map<const ggml_tensor*, std::string>& leaves) {
  Legacy out;
  std::set<const ggml_tensor*> route_weights;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const ggml_tensor* t = nodes[i];
    // A gated delta rule's views: its attention at offset 0, else its state.
    if (t->op == GGML_OP_VIEW && t->src[0] != nullptr && t->src[0]->op == GGML_OP_GATED_DELTA_NET) {
      out.aliases[t] = t->view_offs == 0 ? "gdn.0" : "gdn.1";
      out.aliases[t] += std::format("@{}", static_cast<const void*>(t->src[0]));
      continue;
    }
    // The shared gate's row read as F32: an alias of its weight.
    if (t->op == GGML_OP_RESHAPE && t->src[0] != nullptr && leaves.contains(t->src[0]) &&
        i + 1 < nodes.size() && nodes[i + 1]->op == GGML_OP_GET_ROWS && nodes[i + 1]->src[0] == t &&
        nodes[i + 1]->src[1] != nullptr && leaves.contains(nodes[i + 1]->src[1]) &&
        leaves.at(nodes[i + 1]->src[1]) == "row_zero") {
      out.aliases[t] = leaves.at(t->src[0]);
      out.aliases[nodes[i + 1]] = leaves.at(t->src[0]);
      ++i;
      continue;
    }
    if (t->op == GGML_OP_SOFT_MAX) {
      // soft_max, reshape, argsort, view (ids), get_rows, reshape, sum_rows,
      // clamp, div, reshape (weights).
      const std::size_t ids = i + 3;
      const std::size_t weights = i + 9;
      EXPECT_EQ(nodes.at(i + 2)->op, GGML_OP_ARGSORT);
      EXPECT_EQ(Param<std::int32_t>(nodes.at(i + 2), 0),
                static_cast<std::int32_t>(GGML_SORT_ORDER_DESC));
      EXPECT_EQ(nodes.at(ids)->ne[0], static_cast<std::int64_t>(used));
      EXPECT_EQ(nodes.at(i + 7)->op, GGML_OP_CLAMP);
      EXPECT_EQ(Param<float>(nodes.at(i + 7), 0), 0x1p-14F);
      out.entries.push_back(
          {.first = i, .last = weights, .composite = ir::Op::kRoute, .outputs = {ids, weights}});
      route_weights.insert(nodes.at(weights));
      i = weights;
      continue;
    }
    // The reduce: the product by each slot's scale (where the experts have
    // one), by its weight, a view a slot, an add a slot after the first.
    if (t->op == GGML_OP_MUL && route_weights.contains(t->src[1])) {
      const std::size_t first = scaled ? i - 1 : i;
      if (scaled) {
        EXPECT_EQ(out.entries.back().first, first);
        out.entries.pop_back();
      }
      const std::size_t last = i + (2 * used) - 1;
      EXPECT_EQ(nodes.at(i + 1)->op, GGML_OP_VIEW);
      EXPECT_EQ(nodes.at(last)->op, GGML_OP_ADD);
      out.entries.push_back(
          {.first = first, .last = last, .composite = ir::Op::kExpertReduce, .outputs = {last}});
      i = last;
      continue;
    }
    // A dequantization is its product's weight: an alias of its codes.
    if (IsDequant(t)) {
      out.aliases[t] = leaves.contains(t->src[0]) ? leaves.at(t->src[0]) : "?";
      continue;
    }
    out.entries.push_back({.first = i, .last = i, .composite = std::nullopt, .outputs = {i}});
  }
  return out;
}

// The legacy nodes' outside sources' names: a node matched to an IR node
// by that node's name, an alias (a gated delta rule's views, the shared
// gate's row), a leaf by its name; "?" for a node not matched yet.
struct Names {
  const std::map<const ggml_tensor*, std::string>& leaves;
  const Legacy& legacy;
  std::map<const ggml_tensor*, std::string> matched;

  std::string Of(const ggml_tensor* s) const {
    if (const auto alias = legacy.aliases.find(s); alias != legacy.aliases.end()) {
      if (alias->second.starts_with("gdn.")) {
        const auto at = matched.find(s->src[0]);
        return at == matched.end() ? "?"
                                   : std::format("{}.{}", at->second, alias->second.substr(4, 1));
      }
      return alias->second;
    }
    if (const auto at = matched.find(s); at != matched.end()) return at->second;
    const auto leaf = leaves.find(s);
    return leaf != leaves.end() ? leaf->second : "?";
  }
};

// What an entry reads, as names: a composite run's outside sources (a
// set), or a node's sources in order (an MXFP8 product's codes and input,
// the n-gram rows' table and rows; set_rows' cache first; attention's
// mask a name of its own).
std::vector<std::string> Wanted(const std::vector<ggml_tensor*>& nodes, const Entry& entry,
                                const Names& names) {
  std::vector<std::string> want;
  if (entry.composite) {
    std::set<const ggml_tensor*> inside;
    for (std::size_t k = entry.first; k <= entry.last; ++k) inside.insert(nodes[k]);
    std::set<std::string> outside;
    for (std::size_t k = entry.first; k <= entry.last; ++k) {
      for (const ggml_tensor* s : nodes[k]->src) {
        if (s != nullptr && !inside.contains(s)) outside.insert(names.Of(s));
      }
    }
    return {outside.begin(), outside.end()};
  }
  const ggml_tensor* t = nodes[entry.last];
  if (t->op == GGML_OP_CUSTOM && kg::LlmpOpOf(t) == kg::LlmpOp::kMxfp8MulMatVec) {
    want = {names.Of(t->src[0]), names.Of(t->src[2])};
  } else if (t->op == GGML_OP_CUSTOM && kg::LlmpOpOf(t) == kg::LlmpOp::kNvfp4Rows) {
    want = {names.Of(t->src[0]), names.Of(t->src[1])};
  } else {
    for (const ggml_tensor* s : t->src) {
      if (s != nullptr) want.push_back(names.Of(s));
    }
  }
  if (t->op == GGML_OP_SET_ROWS && want.size() == 3) std::ranges::rotate(want, want.end() - 1);
  if (t->op == GGML_OP_FLASH_ATTN_EXT && want.size() == 4) want[3] = "mask";
  return want;
}

// The IR node's sources as names.
std::vector<std::string> Got(const ir::Node& n, const std::map<std::uint32_t, std::string>& ir_name,
                             const std::map<std::uint32_t, std::string>& step_leaves,
                             bool composite) {
  std::vector<std::string> got;
  const auto sources = n.sources();
  for (std::size_t k = 0; k < sources.size(); ++k) {
    const ir::TensorId s = sources[k];
    if (!s.valid()) continue;
    if (n.key.op == ir::Op::kExpertReduce && k == 2) continue;
    if (const auto at = ir_name.find(s.v); at != ir_name.end()) {
      got.push_back(at->second);
    } else {
      const auto leaf = step_leaves.find(s.v);
      got.push_back(leaf != step_leaves.end() ? leaf->second : "?");
    }
  }
  if (n.key.op == ir::Op::kAttention && got.size() >= 4) got[3] = "mask";
  if (composite) {
    std::ranges::sort(got);
    got.erase(std::ranges::unique(got).begin(), got.end());
  }
  return got;
}

// Whether the IR node is the entry: its op, its extents and type (but a
// gated delta rule's two outputs), its sources.
bool Same(const std::vector<ggml_tensor*>& nodes, const Entry& entry, const ir::Graph& g,
          const ir::Node& n, const std::vector<std::string>& got, const Names& names) {
  if (entry.composite) return n.key.op == *entry.composite && Wanted(nodes, entry, names) == got;
  const ggml_tensor* t = nodes[entry.last];
  if (OpOf(t) != n.key.op) return false;
  if (n.key.op != ir::Op::kGatedDeltaNet) {
    const ir::TensorDesc& out = g.tensor(n.out[0]);
    if (TypeOf(t->type) != out.format.encoding) return false;
    for (std::size_t d = 0; d < 4; ++d) {
      if (static_cast<std::uint64_t>(t->ne[d]) != out.ne(d)) return false;
    }
  }
  return Wanted(nodes, entry, names) == got;
}

// The step against the legacy graph: every IR node matched, in order, to a
// legacy entry of the same operation, extents and sources (and then its
// strides and parameters checked); entries matched out of the legacy order
// are returned (their first nodes' indices), the rest are in order.
std::vector<std::size_t> ExpectSameSteps(
    const std::vector<ggml_tensor*>& nodes,
    const std::map<const ggml_tensor*, std::string>& legacy_leaves, const StepGraph& step,
    const std::map<std::uint32_t, std::string>& step_leaves, std::uint32_t used, bool scaled) {
  const ir::Graph& g = step.graph;
  const std::vector<std::size_t> ir_nodes = Comparable(g, BlockKeyNodes(g, step_leaves));
  const Legacy legacy = Entries(nodes, used, scaled, legacy_leaves);
  const std::vector<Entry>& entries = legacy.entries;
  std::vector<std::size_t> reordered;
  EXPECT_GT(nodes.size(), 100U);
  EXPECT_EQ(entries.size(), ir_nodes.size());
  if (entries.size() != ir_nodes.size()) return reordered;
  std::map<std::uint32_t, std::string> ir_name;
  for (std::size_t i = 0; i < ir_nodes.size(); ++i) {
    const ir::Node& n = g.nodes()[ir_nodes[i]];
    for (std::size_t o = 0; o < n.nout; ++o) {
      ir_name[n.out[o].v] = n.nout == 1 ? std::format("#{}", i) : std::format("#{}.{}", i, o);
    }
  }
  Names names{.leaves = legacy_leaves, .legacy = legacy, .matched = {}};
  std::vector<bool> taken(entries.size(), false);
  std::size_t next = 0;
  std::size_t leaves = 0;
  std::size_t composites = 0;
  for (std::size_t i = 0; i < ir_nodes.size(); ++i) {
    const ir::Node& n = g.nodes()[ir_nodes[i]];
    const bool composite = n.key.op == ir::Op::kRoute || n.key.op == ir::Op::kExpertReduce;
    const std::vector<std::string> got = Got(n, ir_name, step_leaves, composite);
    for (const std::string& s : got) leaves += s.starts_with("#") ? 0 : 1;
    std::optional<std::size_t> found;
    for (std::size_t c = next; c < entries.size() && c < next + 600; ++c) {
      if (!taken[c] && Same(nodes, entries[c], g, n, got, names)) {
        found = c;
        break;
      }
    }
    if (!found) {
      // The first difference against the next legacy entry, for the log.
      const Entry& want = entries[next];
      const ggml_tensor* t = nodes[want.last];
      ADD_FAILURE() << std::format("IR node {} ({}) matches no legacy entry; the next is {} ({})",
                                   i, ir::OpName(n.key.op), next, ggml_op_desc(t))
                    << "\n  IR sources: " << testing::PrintToString(got)
                    << "\n  legacy's:   " << testing::PrintToString(Wanted(nodes, want, names));
      return reordered;
    }
    const Entry& entry = entries[*found];
    taken[*found] = true;
    if (*found != next) reordered.push_back(entries[*found].first);
    while (next < entries.size() && taken[next]) ++next;
    for (std::size_t o = 0; o < entry.outputs.size(); ++o) {
      names.matched[nodes[entry.outputs[o]]] = entry.outputs.size() == 1 && n.nout == 1
                                                   ? std::format("#{}", i)
                                                   : std::format("#{}.{}", i, o);
    }
    if (n.key.op == ir::Op::kGatedDeltaNet)
      names.matched[nodes[entry.last]] = std::format("#{}", i);
    if (entry.composite) {
      ++composites;
      if (n.key.op == ir::Op::kRoute) {
        const auto p = n.Params<ir::RouteParams>();
        EXPECT_EQ(p.kind, ir::RouteKind::kSoftmaxTopK);
        EXPECT_TRUE(p.normalize);
        EXPECT_EQ(p.used, used);
        EXPECT_EQ(p.scale, 1.0F);
        EXPECT_EQ(static_cast<std::uint64_t>(nodes[entry.first]->ne[0]), p.experts);
      }
      continue;
    }
    const ggml_tensor* t = nodes[entry.last];
    const ir::TensorDesc& out = g.tensor(n.out[0]);
    SCOPED_TRACE(std::format("IR node {}: {}", i, ggml_op_desc(t)));
    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
        t->op == GGML_OP_RESHAPE) {
      for (std::size_t d = 0; d < 4; ++d) {
        EXPECT_EQ(static_cast<std::uint64_t>(t->nb[d]), out.nb.at(d)) << "nb " << d;
      }
    }
    if (t->op == GGML_OP_VIEW) EXPECT_EQ(t->view_offs, out.view_offset);
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
      case ir::Op::kClamp: {
        const auto p = n.Params<ir::ClampParams>();
        EXPECT_EQ(Param<float>(t, 0), p.min);
        EXPECT_EQ(Param<float>(t, 1), p.max);
        break;
      }
      case ir::Op::kConcat:
        EXPECT_EQ(Param<std::int32_t>(t, 0), n.Params<ir::ConcatParams>().dim);
        break;
      case ir::Op::kRope: {
        const auto p = n.Params<ir::RopeParams>();
        EXPECT_EQ(Param<std::int32_t>(t, 1), static_cast<std::int32_t>(p.dims));
        EXPECT_EQ(Param<std::int32_t>(t, 2), GGML_ROPE_TYPE_IMROPE);
        EXPECT_EQ(p.mode, ir::RopeMode::kInterleavedMulti);
        EXPECT_EQ(Param<std::int32_t>(t, 4), static_cast<std::int32_t>(p.original_context));
        EXPECT_EQ(Param<float>(t, 5), p.base);
        EXPECT_EQ(Param<float>(t, 6), p.scale);
        EXPECT_EQ(Param<float>(t, 7), p.ext_factor);
        EXPECT_EQ(Param<float>(t, 8), p.attn_factor);
        EXPECT_EQ(Param<float>(t, 9), p.beta_fast);
        EXPECT_EQ(Param<float>(t, 10), p.beta_slow);
        for (std::size_t s = 0; s < 4; ++s) {
          EXPECT_EQ(Param<std::int32_t>(t, 11 + s), static_cast<std::int32_t>(p.sections.at(s)));
        }
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
      case ir::Op::kGlu:
        EXPECT_EQ(Param<std::int32_t>(t, 0), GGML_GLU_OP_SWIGLU);
        EXPECT_EQ(n.Params<ir::GluParams>().kind, ir::GluKind::kSwiGlu);
        break;
      case ir::Op::kUnary: {
        const ir::UnaryFn fn = n.Params<ir::UnaryParams>().fn;
        if (t->op == GGML_OP_SQRT) {
          EXPECT_EQ(fn, ir::UnaryFn::kSqrt);
          break;
        }
        const auto op = ggml_get_unary_op(t);
        const bool same = (op == GGML_UNARY_OP_SILU && fn == ir::UnaryFn::kSilu) ||
                          (op == GGML_UNARY_OP_SIGMOID && fn == ir::UnaryFn::kSigmoid) ||
                          (op == GGML_UNARY_OP_SOFTPLUS && fn == ir::UnaryFn::kSoftplus) ||
                          (op == GGML_UNARY_OP_ABS && fn == ir::UnaryFn::kAbs) ||
                          (op == GGML_UNARY_OP_SGN && fn == ir::UnaryFn::kSgn);
        EXPECT_TRUE(same) << ggml_unary_op_name(op);
        break;
      }
      default:
        break;
    }
  }
  EXPECT_GT(leaves, 0U);
  EXPECT_GT(composites, 0U);
  EXPECT_EQ(composites % 2, 0U);
  return reordered;
}

// The legacy graph's leaves by name: a weight by its fixture resource's
// index ("w.<i>"), a layer's state by its contract's name, the inputs.
std::map<const ggml_tensor*, std::string> LegacyLeaves(const kg::Qwen38Graph& g,
                                                       const md::Qwen38Binding& b) {
  std::map<const ggml_tensor*, std::string> out;
  const auto add = [&](const ggml_tensor* t, std::string name) {
    if (t != nullptr) out[t] = std::move(name);
  };
  const auto w = [&](const ggml_tensor* t, const md::Qwen38Tensor& r) {
    if (t != nullptr && !r.type.empty()) out[t] = std::format("w.{}", r.index);
  };
  const auto linear = [&](const kg::Qwen38Mxfp8Tensors& t, const md::Qwen38Mxfp8& r) {
    w(t.codes, r.codes);
    w(t.scales, r.scales);
    w(t.matrix, r.matrix);
  };
  add(g.tokens, "tokens");
  add(g.positions, "positions");
  add(g.cells, "cells");
  add(g.mask, "mask");
  add(g.ple_rows, "ple_rows");
  add(g.state_row, "state_row");
  add(g.row_zero, "row_zero");
  add(g.out_ids, "out_ids");
  w(g.token_embd, b.token_embd);
  w(g.ple_table, b.ple_table);
  w(g.ple_table_scale, b.ple_table_scale);
  w(g.output, b.output);
  w(g.output_hc_norm, b.output_hc_norm);
  w(g.output_hc_down, b.output_hc_down);
  w(g.output_hc_up, b.output_hc_up);
  for (std::size_t il = 0; il < g.layers.size(); ++il) {
    const kg::Qwen38LayerTensors& l = g.layers[il];
    const md::Qwen38Layer& r = b.layers[il];
    w(l.hc_attn_norm, r.hc_attn_norm);
    w(l.hc_attn_down, r.hc_attn_down);
    w(l.hc_attn_up, r.hc_attn_up);
    w(l.hc_attn_inject, r.hc_attn_inject);
    w(l.hc_ffn_norm, r.hc_ffn_norm);
    w(l.hc_ffn_down, r.hc_ffn_down);
    w(l.hc_ffn_up, r.hc_ffn_up);
    w(l.hc_ffn_inject, r.hc_ffn_inject);
    for (const auto& [t, rr] : {std::pair{&l.qkv, &r.qkv},
                                {&l.z, &r.z},
                                {&l.beta, &r.beta},
                                {&l.alpha, &r.alpha},
                                {&l.ssm_out, &r.ssm_out},
                                {&l.q, &r.q},
                                {&l.k, &r.k},
                                {&l.v, &r.v},
                                {&l.o, &r.o},
                                {&l.idx_qk, &r.idx_qk},
                                {&l.idx_q, &r.idx_q},
                                {&l.idx_k, &r.idx_k},
                                {&l.gate_shexp, &r.gate_shexp},
                                {&l.up_shexp, &r.up_shexp},
                                {&l.down_shexp, &r.down_shexp}}) {
      linear(*t, *rr);
    }
    w(l.dt_bias, r.dt_bias);
    w(l.ssm_a, r.ssm_a);
    w(l.conv1d, r.conv1d);
    w(l.ssm_norm, r.ssm_norm);
    w(l.q_norm, r.q_norm);
    w(l.k_norm, r.k_norm);
    w(l.idx_q_norm, r.idx_q_norm);
    w(l.idx_k_norm, r.idx_k_norm);
    w(l.ple_key, r.ple_key);
    w(l.ple_value, r.ple_value);
    w(l.ple_norm_key, r.ple_norm_key);
    w(l.ple_norm_query, r.ple_norm_query);
    w(l.ple_norm_conv, r.ple_norm_conv);
    w(l.ple_conv1d, r.ple_conv1d);
    w(l.router, r.router);
    w(l.shared_gate, r.shared_gate);
    w(l.gate_exps, r.gate_exps);
    w(l.up_exps, r.up_exps);
    w(l.down_exps, r.down_exps);
    w(l.gate_exps_scale, r.gate_exps_scale);
    w(l.up_exps_scale, r.up_exps_scale);
    w(l.down_exps_scale, r.down_exps_scale);
    add(l.cache_k, std::format("k.{}", il));
    add(l.cache_v, std::format("v.{}", il));
    add(l.cache_idx, std::format("indexer_k.{}", il));
    add(l.cache_pool, std::format("indexer_blocks.{}", il));
    add(l.conv_state, std::format("conv.{}", il));
    add(l.recurrent, std::format("recurrent.{}", il));
    add(l.ple_state, std::format("ple_conv.{}", il));
  }
  return out;
}

// The step's leaves by the same names: a weight by its first part's
// resource, a state leaf by its contract's name and layer.
std::map<std::uint32_t, std::string> StepLeaves(const StepGraph& step,
                                                const std::vector<md::Qwen38Resource>& resources,
                                                const model::StateLayoutDerived& layout) {
  std::map<std::string, std::uint32_t> index;
  for (std::uint32_t i = 0; i < resources.size(); ++i) index[resources[i].roles.front()] = i;
  std::map<std::uint32_t, std::string> out;
  for (const StepInput& in : step.inputs) {
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
        out[in.tensor.v] = in.cells.fixed ? "state_row" : "cells";
        break;
      case InputKind::kSideRows:
        out[in.tensor.v] = "ple_rows";
        break;
      default:
        out[in.tensor.v] = "?";
        break;
    }
  }
  for (const StepState& st : step.state) {
    const auto& item = layout.items.at(st.item);
    out[st.tensor.v] = std::format("{}.{}", item.contract.name, st.ref.layer);
  }
  for (const StepWeight& w : step.weights) {
    const auto at = index.find(w.parts.front().first);
    out[w.tensor.v] =
        at == index.end() ? "w.?" + w.parts.front().first : std::format("w.{}", at->second);
  }
  return out;
}

struct Case {
  std::string name;
  std::uint32_t rows = 1;
  std::uint32_t past = 0;
  std::uint32_t read = 256;
  std::uint32_t outputs = 1;
  bool greedy = false;
};

// Prefill and decode shapes the unfused form builds the same way (no
// convolution padding: rows of at most 32, or whole 32s).
std::vector<Case> Cases() {
  return {
      {.name = "prefill", .rows = 7, .past = 0, .read = 256, .outputs = 1},
      {.name = "prefill outputs", .rows = 5, .past = 300, .read = 512, .outputs = 5},
      {.name = "whole windows", .rows = 64, .past = 0, .read = 256, .outputs = 2},
      {.name = "decode", .rows = 1, .past = 900, .read = 1024, .outputs = 1},
      {.name = "greedy decode", .rows = 1, .past = 40, .read = 256, .outputs = 1, .greedy = true},
  };
}

void CompareForm(fx::Form form) {
  const auto& p = md::Qwen38Flash();
  const auto resources = fx::Resources(p, form);
  const auto bound = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(bound.has_value()) << bound.error();
  const bool gguf = form == fx::Form::kGguf;
  const spec::Source source =
      fx::SourceOf("qwen4exp", gguf ? "gguf" : "safetensors",
                   std::string(gguf ? fx::kGgufConverter : fx::kModelOptConverter), resources,
                   spec::mappings::Qwen38Mapping());
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> vocab;
  for (std::int64_t h = 0; h < 16; ++h) {
    offsets.push_back(h * 20000096);
    vocab.push_back(20000096 - (h * 8));
  }
  const auto hash =
      md::CheckQwen38PleHash(p, std::vector<std::int64_t>{3, 5, 7}, offsets, vocab, fx::kTableRows);
  ASSERT_TRUE(hash.has_value());
  const auto values = spec::mappings::Qwen38V0Values(p, *bound, *hash);
  ASSERT_TRUE(values.has_value()) << values.error();
  const auto spec = spec::Synthesize(spec::mappings::Qwen38Mapping(), source, *values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const auto layout = StageLayout(*spec, 0, kContext, kMaxRows);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  const auto state = md::Qwen38State(p, kContext, kMaxRows, true);
  ASSERT_TRUE(state.has_value());
  for (const Case& c : Cases()) {
    SCOPED_TRACE(c.name);
    const planner::StepShape shape{.segments = {{.slot = 0,
                                                 .rows = c.rows,
                                                 .past = c.past,
                                                 .read_cells = c.read,
                                                 .outputs = c.outputs,
                                                 .greedy = c.greedy}}};
    auto step = BuildStep(*spec, 0, shape, {.tensors = source.tensors, .state = &*layout});
    ASSERT_TRUE(step.has_value()) << step.error();
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    const kg::Qwen38ChunkShape legacy_shape{.rows = c.rows,
                                            .n_kv = c.read,
                                            .cells = state->cells,
                                            .outputs = c.outputs,
                                            .qsa_select = false,
                                            .qsa_blocks = 0,
                                            .token = c.greedy};
    auto graph = kg::BuildQwen38Graph(*arena, p, *bound, legacy_shape,
                                      {.expert_stride = {}, .fused = false});
    ASSERT_TRUE(graph.has_value()) << graph.error().detail;
    const auto leaves = LegacyLeaves(*graph, *bound);
    const std::vector<std::size_t> reordered = ExpectSameSteps(
        graph->nodes, leaves, *step, StepLeaves(*step, resources, *layout), p.experts_used, !gguf);
    // Out of order only around the n-gram layer: the unfused form's
    // n-gram block reads its history, value and key (its convolution
    // store's operands) before the layer before it finishes (its shared
    // expert and combine), where the components finish that layer first.
    // Its nodes between its history's reshape and its store.
    const std::string history = std::format("ple_conv.{}", p.ple_layer);
    std::size_t first = graph->nodes.size();
    std::size_t store = 0;
    for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
      const ggml_tensor* t = graph->nodes[i];
      const auto leaf = [&](const ggml_tensor* s) {
        return s != nullptr && leaves.contains(s) && leaves.at(s) == history;
      };
      if (t->op == GGML_OP_RESHAPE && leaf(t->src[0])) first = std::min(first, i);
      if (t->op == GGML_OP_SET_ROWS && leaf(t->src[2])) store = i;
    }
    EXPECT_LT(first, store);
    std::size_t outside = 0;
    for (const std::size_t node : reordered) outside += node < first || node > store ? 1 : 0;
    std::printf("%s: %zu legacy nodes, %zu of them out of order (n-gram layer's %zu to %zu)\n",
                c.name.c_str(), graph->nodes.size(), reordered.size(), first, store);
    EXPECT_EQ(outside, 0U) << reordered.size() << " entries out of order";
  }
}

TEST(ComponentsQwen38OrderTest, AGgufsStepsAreTheUnfusedBuildersNodeForNode) {
  CompareForm(fx::Form::kGguf);
}

TEST(ComponentsQwen38OrderTest, TheModelOptImportsStepsAreTheUnfusedBuildersNodeForNode) {
  CompareForm(fx::Form::kGgmlExperts);
}

}  // namespace
}  // namespace llmp::components
