// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The DeepSeek V4 components' primitive steps (components/step.h
// BuildStep, the deepseek4 and dflash mappings' spec) against today's
// graph builder's reference form (kernels/ggml/dsv4_graph.h BuildDsv4Graph
// and BuildDsparkGraph, Dsv4GraphOptions fused false: llama.cpp's graph),
// node for node in GGML's execution order (GraphOrder): the same
// operation, result type, extents, view strides, computed sources and the
// parameters that decide the arithmetic, and every leaf a node reads (the
// weight by its tensor's name, the layer's state tensor, the step input),
// over one owner's prefill, decode, greedy decode and state-only chunks,
// a chunk that writes its drafter's injection, and DSpark's draft block.
//
// What differs, by design, and how the comparison takes it:
//   - the routing, its experts' reduce and the compressors' pooling are
//     the op set's composite ops (ir::kRoute, kExpertReduce, kPool): each
//     stands for the run of GGML nodes its decomposition is (the routing's
//     softplus through its scaled weights; the products by each slot's
//     weight, the slot views and the ordered adds; the pooling's permutes,
//     softmax, product, sum and permute back), compared by the run's last
//     node's position, its outside sources and its parameters;
//   - the token-hash routing's lookup is a side input (side.hash), made
//     after the embedding, where GGML runs it inside its layer's routing:
//     those three entries come out of order;
//   - an attention mix's post weights and combination (hc.mix's carries)
//     are made right after its mixing product, where the reference form
//     makes them just before the combine that reads them (its FFN mix
//     already records them there): out of order, eight entries a layer;
//   - a view of a leaf (a weight's or a state tensor's, such as a
//     compressor's ring), which computes nothing and which GGML's traversal
//     may reach from a later consumer first: it may come out of order;
//   - the token embedding's rows are a step input GGML's graph does not
//     make (looked up on the host); the output's grouped weight is a leaf
//     GGML makes three-dimensional where the IR reshapes it; masks are the
//     framework's (the reference form's host masks are inputs), their
//     views to a step's rows the IR's alone; a fill has no source.
// The fast plan's fusions (llmp.dsv4.*, the vector products, the sparse
// attention) are the planner's patterns, slice G4b; the waves' and the
// row-invariant verify's forms are its too.

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
#include "dsv4_fixture.h"
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/tensors.h"
#include "model/dspark.h"
#include "model/dsv4.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/dsv4.h"

namespace llmp::components {
namespace {

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace fx = llmp::test_support::dsv4;

constexpr std::uint32_t kContext = 65536;
constexpr std::uint32_t kMaxRows = 512;

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
    case GGML_OP_ROPE_BACK:
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
    case GGML_OP_FILL:
      return ir::Op::kFill;
    case GGML_OP_TOP_K:
      return ir::Op::kTopK;
    case GGML_OP_LIGHTNING_INDEXER:
      return ir::Op::kIndexScore;
    case GGML_OP_DSV4_HC_COMB:
      return ir::Op::kHcMix;
    case GGML_OP_DSV4_HC_PRE:
      return ir::Op::kHcPre;
    case GGML_OP_DSV4_HC_POST:
      return ir::Op::kHcPost;
    case GGML_OP_CUSTOM:
      return kg::LlmpOpOf(t) == kg::LlmpOp::kArgmax ? std::optional(ir::Op::kArgmax) : std::nullopt;
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

// The IR nodes a GGML graph has: computed nodes, less the masks and their
// views, the positions' views, the token embedding's lookup (named "embd",
// as GGML's host input is) and a weight's reshape (the grouped output
// weight, a 3D leaf in GGML).
std::vector<std::size_t> Comparable(const ir::Graph& g, ir::TensorId tokens,
                                    std::map<std::uint32_t, std::string>& aliases) {
  std::vector<std::size_t> out;
  const auto from = [&](ir::TensorId t) -> const ir::Node& { return g.node(g.producer(t)); };
  for (std::size_t i = 0; i < g.nodes().size(); ++i) {
    const ir::Node& n = g.nodes()[i];
    const ir::Op op = n.key.op;
    if (op == ir::Op::kInput || op == ir::Op::kWeight || op == ir::Op::kState) continue;
    if (op == ir::Op::kMaskBuild) continue;
    if (op == ir::Op::kView) {
      const ir::Node& source = from(n.src[0]);
      if (source.key.op == ir::Op::kMaskBuild) continue;  // a mask's rows
      if (source.key.op == ir::Op::kInput &&
          g.tensor(n.src[0]).format.encoding == format::Encoding::kI32) {
        continue;  // a segment's positions, a mask's input, an anchor
      }
    }
    if (op == ir::Op::kReshape && from(n.src[0]).key.op == ir::Op::kWeight) continue;
    // The embedding's rows of the target's table (GGML's are a host input).
    if (op == ir::Op::kGetRows && from(n.src[0]).key.op == ir::Op::kWeight && n.src[1] == tokens &&
        g.tensor(n.out[0]).format.encoding == format::Encoding::kF32) {
      aliases[n.out[0].v] = "embd";
      continue;
    }
    out.push_back(i);
  }
  return out;
}

// A legacy entry: one GGML node, or the run a composite IR op stands for
// (its nodes, its outputs by the IR op's, its position its last node's).
struct Entry {
  std::vector<std::size_t> nodes;
  std::optional<ir::Op> composite{};
  std::vector<std::size_t> outputs{};
  std::size_t at() const { return nodes.back(); }
};

const ggml_tensor* ConsumerOf(const std::vector<ggml_tensor*>& nodes, const ggml_tensor* t,
                              ggml_op op, std::size_t src) {
  for (const ggml_tensor* n : nodes) {
    if (n->op == op && n->src[src] == t) return n;
  }
  return nullptr;
}

// The legacy views of the step's integer inputs (an anchor, the injected
// rows' positions): the IR's are no nodes; named as their inputs.
bool InputView(const ggml_tensor* t, const std::map<const ggml_tensor*, std::string>& leaves) {
  if (t->op != GGML_OP_VIEW || t->src[0] == nullptr) return false;
  const auto at = leaves.find(t->src[0]);
  return at != leaves.end() && (at->second == "tokens" || at->second == "positions");
}

std::vector<Entry> Entries(const std::vector<ggml_tensor*>& nodes,
                           const std::map<const ggml_tensor*, std::string>& leaves,
                           std::uint32_t used) {
  std::map<const ggml_tensor*, std::size_t> index;
  for (std::size_t i = 0; i < nodes.size(); ++i) index[nodes[i]] = i;
  std::set<std::size_t> taken;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (InputView(nodes[i], leaves)) taken.insert(i);
  }
  std::vector<Entry> out;
  const auto is_leaf = [&](const ggml_tensor* t, std::string_view prefix) {
    const auto at = leaves.find(t);
    return at != leaves.end() && at->second.starts_with(prefix);
  };
  // The pooling runs: around each softmax.
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const ggml_tensor* s = nodes[i];
    if (s->op != GGML_OP_SOFT_MAX) continue;
    const ggml_tensor* scores_cont = s->src[0];
    const ggml_tensor* mul = ConsumerOf(nodes, s, GGML_OP_MUL, 1);
    const ggml_tensor* sum = mul != nullptr ? ConsumerOf(nodes, mul, GGML_OP_SUM_ROWS, 0) : nullptr;
    const ggml_tensor* back = sum != nullptr ? ConsumerOf(nodes, sum, GGML_OP_PERMUTE, 0) : nullptr;
    const ggml_tensor* cont = back != nullptr ? ConsumerOf(nodes, back, GGML_OP_CONT, 0) : nullptr;
    EXPECT_NE(cont, nullptr);
    if (cont == nullptr) return out;
    Entry e;
    e.composite = ir::Op::kPool;
    const std::array<const ggml_tensor*, 9> run{mul->src[0]->src[0],
                                                mul->src[0],
                                                scores_cont->src[0],
                                                scores_cont,
                                                s,
                                                mul,
                                                sum,
                                                back,
                                                cont};
    for (const ggml_tensor* t : run) {
      e.nodes.push_back(index.at(t));
    }
    std::ranges::sort(e.nodes);
    e.outputs = {index.at(cont)};
    for (const std::size_t n : e.nodes) taken.insert(n);
    out.push_back(std::move(e));
  }
  // The routings: softplus through the scaled weights (a hash routing's
  // lookup aside: the side input's entry); then each one's reduce.
  std::set<const ggml_tensor*> weights;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const ggml_tensor* t = nodes[i];
    if (t->op != GGML_OP_UNARY || ggml_get_unary_op(t) != GGML_UNARY_OP_SOFTPLUS) continue;
    std::size_t last = i;
    while (last < nodes.size() && nodes[last]->op != GGML_OP_SCALE) ++last;
    Entry e;
    e.composite = ir::Op::kRoute;
    std::optional<std::size_t> ids;
    for (std::size_t k = i; k <= last; ++k) {
      const ggml_tensor* n = nodes[k];
      if (n->op == GGML_OP_GET_ROWS && is_leaf(n->src[0], "blk.")) {
        ids = k;  // the hash lookup: its own entry
        continue;
      }
      if (n->op == GGML_OP_VIEW && n->src[0]->op == GGML_OP_ARGSORT) ids = k;
      e.nodes.push_back(k);
    }
    EXPECT_TRUE(ids.has_value());
    e.outputs = {ids.value_or(0), last};
    for (const std::size_t n : e.nodes) taken.insert(n);
    weights.insert(nodes[last]);
    out.push_back(std::move(e));
  }
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const ggml_tensor* t = nodes[i];
    if (t->op != GGML_OP_MUL || !weights.contains(t->src[1])) continue;
    // The product, a view a slot, an add a slot after the first.
    const std::size_t last = i + (2 * std::size_t{used}) - 1;
    EXPECT_EQ(nodes.at(i + 1)->op, GGML_OP_VIEW);
    EXPECT_EQ(nodes.at(last)->op, GGML_OP_ADD);
    Entry e;
    e.composite = ir::Op::kExpertReduce;
    for (std::size_t k = i; k <= last; ++k) e.nodes.push_back(k);
    e.outputs = {last};
    for (const std::size_t n : e.nodes) taken.insert(n);
    out.push_back(std::move(e));
  }
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (taken.contains(i)) continue;
    out.push_back({.nodes = {i}, .composite = std::nullopt, .outputs = {i}});
  }
  std::ranges::sort(out, {}, &Entry::at);
  return out;
}

// The legacy nodes' outside sources' names: a node matched to an IR node
// by that node's name, a leaf by its name; "?" for a node not matched yet.
struct Names {
  const std::map<const ggml_tensor*, std::string>& leaves;
  std::map<const ggml_tensor*, std::string> matched;

  std::string Of(const ggml_tensor* s) const {
    if (const auto at = matched.find(s); at != matched.end()) return at->second;
    if (InputView(s, leaves)) return leaves.at(s->src[0]);
    const auto leaf = leaves.find(s);
    return leaf != leaves.end() ? leaf->second : "?";
  }
};

std::vector<std::string> Wanted(const std::vector<ggml_tensor*>& nodes, const Entry& entry,
                                const Names& names) {
  std::vector<std::string> want;
  if (entry.composite) {
    std::set<const ggml_tensor*> inside;
    for (const std::size_t k : entry.nodes) inside.insert(nodes[k]);
    std::set<std::string> outside;
    for (const std::size_t k : entry.nodes) {
      for (const ggml_tensor* s : nodes[k]->src) {
        if (s != nullptr && !inside.contains(s)) outside.insert(names.Of(s));
      }
    }
    return {outside.begin(), outside.end()};
  }
  const ggml_tensor* t = nodes[entry.at()];
  if (t->op == GGML_OP_FILL) return {};  // the IR's fill has no source
  for (const ggml_tensor* s : t->src) {
    if (s != nullptr) want.push_back(names.Of(s));
  }
  if (t->op == GGML_OP_SET_ROWS && want.size() == 3) std::ranges::rotate(want, want.end() - 1);
  return want;
}

std::vector<std::string> Got(const ir::Graph& g, const ir::Node& n,
                             const std::map<std::uint32_t, std::string>& ir_name,
                             const std::map<std::uint32_t, std::string>& step_leaves,
                             bool composite) {
  std::vector<std::string> got;
  const auto sources = n.sources();
  for (std::size_t k = 0; k < sources.size(); ++k) {
    ir::TensorId s = sources[k];
    if (!s.valid()) continue;
    if (n.key.op == ir::Op::kExpertReduce && k == 2) continue;
    // A mask's view: the mask.
    const ir::Node& from = g.node(g.producer(s));
    if (from.key.op == ir::Op::kView &&
        g.node(g.producer(from.src[0])).key.op == ir::Op::kMaskBuild) {
      s = from.src[0];
    }
    if (g.node(g.producer(s)).key.op == ir::Op::kMaskBuild) {
      got.emplace_back("mask");
      continue;
    }
    if (const auto at = ir_name.find(s.v); at != ir_name.end()) {
      got.push_back(at->second);
    } else {
      const auto leaf = step_leaves.find(g.root(s).v);
      got.push_back(leaf != step_leaves.end() ? leaf->second : "?");
    }
  }
  if (composite) {
    std::ranges::sort(got);
    got.erase(std::ranges::unique(got).begin(), got.end());
  }
  if (n.key.op == ir::Op::kFill) got.clear();
  return got;
}

bool Same(const std::vector<ggml_tensor*>& nodes, const Entry& entry, const ir::Graph& g,
          const ir::Node& n, const std::vector<std::string>& got, const Names& names) {
  if (entry.composite) return n.key.op == *entry.composite && Wanted(nodes, entry, names) == got;
  const ggml_tensor* t = nodes[entry.at()];
  if (OpOf(t) != n.key.op) return false;
  const ir::TensorDesc& out = g.tensor(n.out[0]);
  if (TypeOf(t->type) != out.format.encoding) return false;
  for (std::size_t d = 0; d < 4; ++d) {
    if (static_cast<std::uint64_t>(t->ne[d]) != out.ne(d)) return false;
  }
  // A view's place, a scale's arithmetic: two views of one tensor or two
  // scales of one source differ by them.
  if (t->op == GGML_OP_VIEW) {
    if (t->view_offs != out.view_offset) return false;
    for (std::size_t d = 0; d < 4; ++d) {
      if (static_cast<std::uint64_t>(t->nb[d]) != out.nb.at(d)) return false;
    }
  }
  if (t->op == GGML_OP_SCALE) {
    const auto p = n.Params<ir::ScaleParams>();
    if (Param<float>(t, 0) != p.scale || Param<float>(t, 1) != p.bias) return false;
  }
  return Wanted(nodes, entry, names) == got;
}

// The step against the legacy graph: every IR node matched to a legacy
// entry of the same operation, extents and sources (then its strides and
// parameters checked); the entries matched out of order are returned
// (their last nodes).
std::vector<std::size_t> ExpectSameSteps(
    const std::vector<ggml_tensor*>& nodes,
    const std::map<const ggml_tensor*, std::string>& legacy_leaves, const StepGraph& step,
    const std::map<std::uint32_t, std::string>& step_leaves, ir::TensorId tokens,
    std::uint32_t used) {
  const ir::Graph& g = step.graph;
  std::map<std::uint32_t, std::string> ir_name;
  const std::vector<std::size_t> ir_nodes = Comparable(g, tokens, ir_name);
  const std::vector<Entry> entries = Entries(nodes, legacy_leaves, used);
  std::vector<std::size_t> reordered;
  EXPECT_GT(nodes.size(), 50U);
  EXPECT_EQ(entries.size(), ir_nodes.size());
  Names names{.leaves = legacy_leaves, .matched = {}};
  std::vector<bool> taken(entries.size(), false);
  std::size_t next = 0;
  std::size_t composites = 0;
  std::vector<std::size_t> order;  // each IR node's legacy entry's position
  for (std::size_t i = 0; i < ir_nodes.size(); ++i) {
    const ir::Node& n = g.nodes()[ir_nodes[i]];
    for (std::size_t o = 0; o < n.nout; ++o) {
      ir_name[n.out[o].v] = n.nout == 1 ? std::format("#{}", i) : std::format("#{}.{}", i, o);
    }
    const bool composite = n.key.op == ir::Op::kRoute || n.key.op == ir::Op::kExpertReduce ||
                           n.key.op == ir::Op::kPool;
    const std::vector<std::string> got = Got(g, n, ir_name, step_leaves, composite);
    std::optional<std::size_t> found;
    for (std::size_t c = next; c < entries.size() && c < next + 2000; ++c) {
      if (!taken[c] && Same(nodes, entries[c], g, n, got, names)) {
        found = c;
        break;
      }
    }
    if (!found) {
      if (next >= entries.size()) {
        ADD_FAILURE() << std::format("IR node {} ({}) past every legacy entry", i,
                                     ir::OpName(n.key.op));
        return reordered;
      }
      const Entry& want = entries[next];
      const ggml_tensor* t = nodes[want.at()];
      ADD_FAILURE() << std::format("IR node {} ({}) matches no legacy entry; the next is {} ({}{})",
                                   i, ir::OpName(n.key.op), next, ggml_op_desc(t),
                                   want.composite ? ", a composite" : "")
                    << "\n  IR sources: " << testing::PrintToString(got)
                    << "\n  legacy's:   " << testing::PrintToString(Wanted(nodes, want, names))
                    << "\n  IR extents: " << g.tensor(n.out[0]).ne(0) << ","
                    << g.tensor(n.out[0]).ne(1) << "," << g.tensor(n.out[0]).ne(2) << ","
                    << g.tensor(n.out[0]).ne(3) << " legacy " << t->ne[0] << "," << t->ne[1] << ","
                    << t->ne[2] << "," << t->ne[3];
      return reordered;
    }
    const Entry& entry = entries[*found];
    taken[*found] = true;
    order.push_back(entry.at());
    while (next < entries.size() && taken[next]) ++next;
    for (std::size_t o = 0; o < entry.outputs.size(); ++o) {
      names.matched[nodes[entry.outputs[o]]] = entry.outputs.size() == 1 && n.nout == 1
                                                   ? std::format("#{}", i)
                                                   : std::format("#{}.{}", i, o);
    }
    if (entry.composite) {
      ++composites;
      if (n.key.op == ir::Op::kRoute) {
        const auto p = n.Params<ir::RouteParams>();
        EXPECT_TRUE(p.normalize);
        EXPECT_EQ(p.scale, 1.5F);
      }
      continue;
    }
    const ggml_tensor* t = nodes[entry.at()];
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
      case ir::Op::kConcat:
        EXPECT_EQ(Param<std::int32_t>(t, 0), n.Params<ir::ConcatParams>().dim);
        break;
      case ir::Op::kFill:
        EXPECT_EQ(Param<float>(t, 0), n.Params<ir::FillParams>().value);
        break;
      case ir::Op::kTopK:
        EXPECT_EQ(static_cast<std::uint32_t>(t->ne[0]), n.Params<ir::TopKParams>().k);
        break;
      case ir::Op::kRope: {
        const auto p = n.Params<ir::RopeParams>();
        EXPECT_EQ(t->op == GGML_OP_ROPE_BACK, p.inverse);
        EXPECT_EQ(Param<std::int32_t>(t, 1), static_cast<std::int32_t>(p.dims));
        EXPECT_EQ(Param<std::int32_t>(t, 2), 0);  // NORMAL pairs
        EXPECT_EQ(p.mode, ir::RopeMode::kNormal);
        EXPECT_EQ(Param<std::int32_t>(t, 4), static_cast<std::int32_t>(p.original_context));
        EXPECT_EQ(Param<float>(t, 5), p.base);
        EXPECT_EQ(Param<float>(t, 6), p.scale);
        EXPECT_EQ(Param<float>(t, 7), p.ext_factor);
        EXPECT_EQ(Param<float>(t, 8), p.attn_factor);
        EXPECT_EQ(Param<float>(t, 9), p.beta_fast);
        EXPECT_EQ(Param<float>(t, 10), p.beta_slow);
        EXPECT_EQ(Param<std::int32_t>(t, 15), static_cast<std::int32_t>(p.offset));
        break;
      }
      case ir::Op::kAttention: {
        const auto p = n.Params<ir::AttnParams>();
        EXPECT_EQ(Param<float>(t, 0), p.scale);
        EXPECT_EQ(Param<float>(t, 1), p.max_bias);
        EXPECT_EQ(Param<float>(t, 2), p.softcap);
        EXPECT_EQ(Param<std::int32_t>(t, 3), GGML_PREC_F32);
        EXPECT_EQ(p.accumulate, ir::Accumulate::kF32);
        EXPECT_TRUE(p.sinks);
        break;
      }
      case ir::Op::kGlu:
        EXPECT_EQ(Param<std::int32_t>(t, 0), GGML_GLU_OP_SWIGLU_CLAMP);
        EXPECT_EQ(n.Params<ir::GluParams>().kind, ir::GluKind::kSwiGluClamp);
        break;
      case ir::Op::kHcMix: {
        const auto p = n.Params<ir::HcParams>();
        EXPECT_EQ(p.mix, ir::HcMixKind::kSinkhorn);
        EXPECT_EQ(Param<float>(t, 0), p.eps);
        EXPECT_EQ(Param<std::int32_t>(t, 1), static_cast<std::int32_t>(p.iterations));
        break;
      }
      case ir::Op::kMatMul: {
        const auto p = n.Params<ir::ProductParams>();
        // The router's F32 sums, the indexer's Hadamard hint.
        EXPECT_EQ(Param<std::int32_t>(t, 0) == GGML_PREC_F32, p.accumulate == ir::Accumulate::kF32);
        break;
      }
      default:
        break;
    }
  }
  EXPECT_GT(composites, 0U);
  // Out of order: the entries outside a longest increasing run of the
  // matched positions (the fewest moved to make the orders one).
  std::vector<std::size_t> tails;  // index into `order` of each length's least tail
  std::vector<std::size_t> before(order.size(), order.size());
  for (std::size_t k = 0; k < order.size(); ++k) {
    const auto at =
        std::ranges::lower_bound(tails, order[k], {}, [&](std::size_t j) { return order[j]; });
    if (at != tails.begin()) before[k] = *(at - 1);
    if (at == tails.end()) {
      tails.push_back(k);
    } else {
      *at = k;
    }
  }
  std::vector<bool> kept(order.size(), false);
  for (std::size_t k = tails.empty() ? order.size() : tails.back(); k < order.size();
       k = before[k]) {
    kept[k] = true;
  }
  for (std::size_t k = 0; k < order.size(); ++k) {
    if (!kept[k]) reordered.push_back(order[k]);
  }
  return reordered;
}

// The legacy graph's leaves by name: a weight by its fixture resource's
// tensor name, a layer's state by its contract's name and layer, the
// inputs by what they hold.
std::map<const ggml_tensor*, std::string> LegacyLeaves(const kg::Dsv4Graph& g,
                                                       const md::Dsv4Binding& b,
                                                       const std::vector<md::Dsv4Resource>& r,
                                                       std::string_view state_prefix = {}) {
  std::map<const ggml_tensor*, std::string> out;
  const auto add = [&](const ggml_tensor* t, std::string name) {
    if (t != nullptr) out[t] = std::move(name);
  };
  const auto w = [&](const ggml_tensor* t, const md::Dsv4Tensor& tensor) {
    if (t != nullptr && !tensor.type.empty()) out[t] = r.at(tensor.index).roles.front();
  };
  add(g.embd, "embd");
  add(g.tokens, "tokens");
  add(g.positions, "positions");
  add(g.raw_k_idxs, "cells");
  add(g.raw_mask, "mask");
  add(g.out_ids, "out_ids");
  add(g.lid_rot, "hadamard");
  add(g.top_k_zeros, "zeros");
  for (const kg::Dsv4CompInputs* in : {&g.csa, &g.hca, &g.lid}) {
    add(in->persist_src, "kept_rows");
    add(in->persist_dst, "kept_cells");
    add(in->read_idxs, "pool_reads");
    add(in->write_idxs, "block_cells");
    add(in->write_pos, "block_starts");
    add(in->mask, "mask");
  }
  add(g.csa.state_pos, "phases.4");
  add(g.lid.state_pos, "phases.4");
  add(g.hca.state_pos, "phases.128");
  w(g.output_norm, b.output_norm);
  w(g.output, b.output);
  w(g.hc_head_fn, b.hc_head_fn);
  w(g.hc_head_base, b.hc_head_base);
  w(g.hc_head_scale, b.hc_head_scale);
  for (std::size_t il = 0; il < g.layers.size(); ++il) {
    const kg::Dsv4LayerTensors& l = g.layers[il];
    const md::Dsv4Layer& x = b.layers[il];
    for (const auto& [t, tensor] : {std::pair{l.attn_norm, &x.attn_norm},
                                    {l.attn_sinks, &x.attn_sinks},
                                    {l.q_a, &x.q_a},
                                    {l.q_a_norm, &x.q_a_norm},
                                    {l.q_b, &x.q_b},
                                    {l.kv, &x.kv},
                                    {l.kv_norm, &x.kv_norm},
                                    {l.out_a, &x.out_a},
                                    {l.out_b, &x.out_b},
                                    {l.hc_attn_fn, &x.hc_attn_fn},
                                    {l.hc_attn_base, &x.hc_attn_base},
                                    {l.hc_attn_scale, &x.hc_attn_scale},
                                    {l.hc_ffn_fn, &x.hc_ffn_fn},
                                    {l.hc_ffn_base, &x.hc_ffn_base},
                                    {l.hc_ffn_scale, &x.hc_ffn_scale},
                                    {l.comp_kv, &x.comp_kv},
                                    {l.comp_gate, &x.comp_gate},
                                    {l.comp_ape, &x.comp_ape},
                                    {l.comp_norm, &x.comp_norm},
                                    {l.idx_q_b, &x.idx_q_b},
                                    {l.idx_proj, &x.idx_proj},
                                    {l.idx_comp_kv, &x.idx_comp_kv},
                                    {l.idx_comp_gate, &x.idx_comp_gate},
                                    {l.idx_comp_ape, &x.idx_comp_ape},
                                    {l.idx_comp_norm, &x.idx_comp_norm},
                                    {l.ffn_norm, &x.ffn_norm},
                                    {l.router, &x.router},
                                    {l.router_bias, &x.router_bias},
                                    {l.tid2eid, &x.tid2eid},
                                    {l.up_exps, &x.up_exps},
                                    {l.gate_exps, &x.gate_exps},
                                    {l.down_exps, &x.down_exps},
                                    {l.up_shexp, &x.up_shexp},
                                    {l.gate_shexp, &x.gate_shexp},
                                    {l.down_shexp, &x.down_shexp}}) {
      w(t, *tensor);
    }
    const std::string p(state_prefix);
    add(l.raw_k, std::format("{}kv.{}", p, il));
    add(l.csa_k, std::format("{}compressed.{}", p, il));
    add(l.csa_state_kv, std::format("{}compressed_kv.{}", p, il));
    add(l.csa_state_score, std::format("{}compressed_score.{}", p, il));
    add(l.hca_k, std::format("{}compressed.{}", p, il));
    add(l.hca_state_kv, std::format("{}compressed_kv.{}", p, il));
    add(l.hca_state_score, std::format("{}compressed_score.{}", p, il));
    add(l.lid_k, std::format("{}index.{}", p, il));
    add(l.lid_state_kv, std::format("{}index_kv.{}", p, il));
    add(l.lid_state_score, std::format("{}index_score.{}", p, il));
  }
  return out;
}

// The step's leaves by the same names.
std::map<std::uint32_t, std::string> StepLeaves(
    const spec::ModelSpec& spec, const StepGraph& step,
    std::span<const model::StateLayoutDerived* const> layouts) {
  std::map<std::uint32_t, std::string> out;
  for (const StepInput& in : step.inputs) {
    switch (in.kind) {
      case InputKind::kTokens:
      case InputKind::kSideRows:  // a token-hash side input's rows: the tokens
        out[in.tensor.v] = "tokens";
        break;
      case InputKind::kPositions:
        out[in.tensor.v] = "positions";
        break;
      case InputKind::kOutputRows:
        out[in.tensor.v] = "out_ids";
        break;
      case InputKind::kStoreCells:
        out[in.tensor.v] = "cells";
        break;
      case InputKind::kPositionPhases:
        out[in.tensor.v] = std::format("phases.{}", in.ratio);
        break;
      case InputKind::kKeptRows:
        out[in.tensor.v] = "kept_rows";
        break;
      case InputKind::kKeptCells:
        out[in.tensor.v] = "kept_cells";
        break;
      case InputKind::kPoolReads:
        out[in.tensor.v] = "pool_reads";
        break;
      case InputKind::kBlockCells:
        out[in.tensor.v] = "block_cells";
        break;
      case InputKind::kBlockStarts:
        out[in.tensor.v] = "block_starts";
        break;
      case InputKind::kHadamard:
        out[in.tensor.v] = "hadamard";
        break;
      default:
        out[in.tensor.v] = "?";
        break;
    }
  }
  for (const StepState& st : step.state) {
    const auto& item = layouts[st.ref.stage]->items.at(st.item);
    out[st.tensor.v] = std::format("{}{}.{}", st.ref.stage == 0 ? "" : "dspark.",
                                   item.contract.name, st.ref.layer);
  }
  // A drafter's own weights by its stage's name (a target's tensors and
  // its drafter's share names).
  for (const StepWeight& w : step.weights) {
    out[w.tensor.v] = std::format("{}{}", w.ref.stage == 0 ? "" : "dspark.", w.parts.front().first);
  }
  (void)spec;
  return out;
}

ir::TensorId TokensOf(const StepGraph& step) {
  for (const StepInput& in : step.inputs) {
    if (in.kind == InputKind::kTokens) return in.tensor;
  }
  return {};
}

struct Fixture {
  md::Dsv4Profile p = md::Dsv4Flash();
  md::DsparkProfile d = md::DsparkDeepSeekV4Flash();
  std::vector<md::Dsv4Resource> resources = fx::Resources(p);
  std::vector<md::Dsv4Resource> drafter_resources = fx::DrafterResources(d);
  spec::Source target = fx::SourceOf("deepseek4", resources, spec::mappings::Dsv4Mapping());
  spec::Source drafter = fx::SourceOf("dflash", drafter_resources, spec::mappings::DsparkMapping());
  spec::ModelSpec spec;
  md::Dsv4Binding binding;
  md::DsparkBinding drafter_binding;
  model::StateLayoutDerived layout;
  model::StateLayoutDerived drafter_layout;
  md::Dsv4StateLayout state;
};

std::optional<Fixture> Make() {
  Fixture f;
  auto values = spec::mappings::Dsv4V0Values(f.p);
  auto target = spec::Synthesize(spec::mappings::Dsv4Mapping(), f.target, *values);
  EXPECT_TRUE(target.has_value()) << (target ? "" : target.error());
  if (!target) return std::nullopt;
  auto drafter_values = spec::mappings::DsparkV0Values(f.d);
  auto spec = spec::ComposeDrafter(*target, spec::mappings::Dsv4Mapping(),
                                   spec::mappings::DsparkMapping(), f.drafter, *drafter_values);
  EXPECT_TRUE(spec.has_value()) << (spec ? "" : spec.error());
  if (!spec) return std::nullopt;
  f.spec = std::move(*spec);
  auto binding = md::BindDsv4(f.p, "deepseek4", f.resources);
  EXPECT_TRUE(binding.has_value()) << (binding ? "" : binding.error());
  if (!binding) return std::nullopt;
  f.binding = std::move(*binding);
  auto drafter_binding = md::BindDspark(f.d, "dflash", f.drafter_resources, f.p, f.binding);
  EXPECT_TRUE(drafter_binding.has_value()) << (drafter_binding ? "" : drafter_binding.error());
  if (!drafter_binding) return std::nullopt;
  f.drafter_binding = std::move(*drafter_binding);
  auto layout = StageLayout(f.spec, 0, kContext, kMaxRows);
  auto drafter_layout = StageLayout(f.spec, 1, kContext, 16);
  auto state = md::Dsv4State(f.p, kContext, kMaxRows, md::Dsv4Window::kRing);
  EXPECT_TRUE(layout && drafter_layout && state);
  if (!layout || !drafter_layout || !state) return std::nullopt;
  f.layout = std::move(*layout);
  f.drafter_layout = std::move(*drafter_layout);
  f.state = std::move(*state);
  return f;
}

struct Case {
  std::string name;
  std::uint32_t rows = 1;
  std::uint32_t past = 0;
  std::uint32_t outputs = 1;
  bool greedy = false;
  bool inject = false;
};

// Whether a node is a mix's post weight chain (the views of the mixing
// product and its scale and base, the product, sum, sigmoid and scale).
bool PostChain(const ggml_tensor* t) {
  const auto mixes = [](const ggml_tensor* x) {
    return x != nullptr && x->op == GGML_OP_VIEW && x->src[0] != nullptr &&
           x->src[0]->op == GGML_OP_MUL_MAT;
  };
  switch (t->op) {
    case GGML_OP_VIEW:
      return mixes(t) || (t->src[0] != nullptr && t->src[0]->op == GGML_OP_NONE);
    case GGML_OP_MUL:
      return mixes(t->src[0]);
    case GGML_OP_ADD:
      return t->src[0] != nullptr && t->src[0]->op == GGML_OP_MUL && mixes(t->src[0]->src[0]);
    case GGML_OP_UNARY:
      return ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID && PostChain(t->src[0]);
    case GGML_OP_SCALE:
      return t->src[0] != nullptr && t->src[0]->op == GGML_OP_UNARY && PostChain(t->src[0]);
    default:
      return false;
  }
}

// The reordered entries the design moves (above): an attention mix's
// carries (its post weight chain and the Sinkhorn combination), a hash
// routing's lookup, and a view of a state tensor (a compressor's ring,
// which GGML's traversal reaches from the next layer's compressor before
// the layer before it finishes).
bool Expected(const ggml_tensor* t) {
  if (t->op == GGML_OP_DSV4_HC_COMB) return true;
  if (t->op == GGML_OP_GET_ROWS) {
    return t->src[0] != nullptr && t->src[0]->op == GGML_OP_NONE &&
           t->src[0]->type == GGML_TYPE_I32;
  }
  return PostChain(t);
}

void CompareCase(const Fixture& f, const Case& c) {
  SCOPED_TRACE(c.name);
  const std::array<const model::StateLayoutDerived*, 2> layouts{&f.layout, &f.drafter_layout};
  const std::array<std::span<const BoundTensor>, 2> tensors{f.target.tensors, f.drafter.tensors};
  const Bindings bindings{
      .tensors = f.target.tensors, .state = &f.layout, .stages = layouts, .artifacts = tensors};
  // A target without its drafter: one stage of the composed spec.
  spec::ModelSpec alone = f.spec;
  if (!c.inject) {
    alone.stages.resize(1);
    alone.stages[0].taps.clear();
  }
  const planner::StepShape shape{
      .phase = c.greedy ? planner::Phase::kDecode : planner::Phase::kPrefill,
      .segments = {{.slot = 0,
                    .rows = c.rows,
                    .past = c.past,
                    .read_cells = f.state.raw_cells,
                    .outputs = c.outputs,
                    .greedy = c.greedy}}};
  auto step = BuildStep(alone, 0, shape, bindings);
  ASSERT_TRUE(step.has_value()) << step.error();
  auto chunk = md::Dsv4Chunk(f.p, f.state, c.past, c.rows, true, true);
  ASSERT_TRUE(chunk.has_value()) << chunk.error();
  kg::Dsv4ChunkShape legacy_shape = kg::Dsv4ShapeOf(f.state, *chunk, c.outputs);
  legacy_shape.token = c.greedy;
  std::optional<kg::Dsv4Injection> inject;
  if (c.inject) {
    inject = kg::Dsv4Injection{.profile = &f.d,
                               .binding = &f.drafter_binding,
                               .rows = std::min<std::int64_t>(c.rows, f.d.ring),
                               .ring = f.d.ring};
  }
  auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(f.p));
  ASSERT_TRUE(arena.has_value());
  const kg::Dsv4GraphOptions options{
      .features =
          c.inject ? std::vector<std::uint32_t>(f.d.target_layers.begin(), f.d.target_layers.end())
                   : std::vector<std::uint32_t>{},
      .inject = inject,
      .state_only = c.outputs == 0};
  auto graph = kg::BuildDsv4Graph(*arena, f.p, f.binding, legacy_shape, options);
  ASSERT_TRUE(graph.has_value()) << graph.error().detail;
  auto leaves = LegacyLeaves(*graph, f.binding, f.resources);
  if (graph->inject) {
    leaves[graph->inject->cells] = "kept_cells";
    leaves[graph->inject->fc] = "dspark.fc.weight";
    leaves[graph->inject->enc_norm] = "dspark.enc.output_norm.weight";
    for (std::size_t il = 0; il < graph->inject->kv.size(); ++il) {
      leaves[graph->inject->kv[il]] = std::format("dspark.blk.{}.attn_kv.weight", il);
      leaves[graph->inject->kv_norm[il]] = std::format("dspark.blk.{}.attn_kv_a_norm.weight", il);
      leaves[graph->inject->ring[il]] = std::format("dspark.kv.{}", il);
    }
  }
  const std::vector<std::size_t> reordered =
      ExpectSameSteps(graph->nodes, leaves, *step, StepLeaves(alone, *step, layouts),
                      TokensOf(*step), f.p.experts_used);
  std::string unexpected;
  for (const std::size_t node : reordered) {
    const ggml_tensor* t = graph->nodes[node];
    if (!Expected(t)) unexpected += std::format(" {}:{}", node, ggml_op_desc(t));
  }
  testing::Test::RecordProperty(c.name + " reordered", static_cast<int>(reordered.size()));
  EXPECT_EQ(unexpected, "") << c.name << ": " << reordered.size() << " of " << graph->nodes.size()
                            << " entries out of order";
}

TEST(ComponentsDsv4OrderTest, TheTargetsStepsAreTheReferenceFormsNodeForNode) {
  auto f = Make();
  ASSERT_TRUE(f.has_value());
  for (const Case& c : std::vector<Case>{
           {.name = "prefill", .rows = 7, .past = 0, .outputs = 1},
           {.name = "prefill outputs", .rows = 64, .past = 300, .outputs = 2},
           {.name = "an HCA block", .rows = 130, .past = 0, .outputs = 1},
           {.name = "decode", .rows = 1, .past = 900, .outputs = 1},
           {.name = "greedy decode", .rows = 1, .past = 40, .outputs = 1, .greedy = true},
           {.name = "state-only", .rows = 64, .past = 128, .outputs = 0},
           {.name = "injected", .rows = 9, .past = 20, .outputs = 1, .inject = true},
       }) {
    CompareCase(*f, c);
  }
}

TEST(ComponentsDsv4OrderTest, TheDraftBlockIsTheReferenceFormsNodeForNode) {
  auto f = Make();
  ASSERT_TRUE(f.has_value());
  const std::array<const model::StateLayoutDerived*, 2> layouts{&f->layout, &f->drafter_layout};
  const std::array<std::span<const BoundTensor>, 2> tensors{f->target.tensors, f->drafter.tensors};
  const Bindings bindings{.tensors = f->drafter.tensors,
                          .state = &f->drafter_layout,
                          .stages = layouts,
                          .artifacts = tensors};
  constexpr std::uint32_t kRows = 5;
  const planner::StepShape shape{.stage = 1,
                                 .phase = planner::Phase::kDraft,
                                 .segments = {{.slot = 0,
                                               .rows = kRows,
                                               .past = 812,
                                               .read_cells = 256,
                                               .outputs = kRows,
                                               .greedy = true}}};
  auto step = BuildStep(f->spec, 1, shape, bindings);
  ASSERT_TRUE(step.has_value()) << step.error();
  auto arena = kg::TensorArena::Create(kg::DsparkGraphTensors(f->d, kRows));
  ASSERT_TRUE(arena.has_value());
  auto graph = kg::BuildDsparkGraph(*arena, f->d, f->drafter_binding, kRows, f->d.ring, {});
  ASSERT_TRUE(graph.has_value()) << graph.error().detail;
  // The drafter's leaves: its blocks' by its resources' names; the target's
  // table and head by theirs.
  auto leaves = LegacyLeaves(graph->core, f->drafter_binding.blocks, f->drafter_resources);
  leaves[graph->core.output] = "output.weight";
  leaves[graph->tokens] = "tokens";
  leaves[graph->markov_w1] = "markov_w1.weight";
  leaves[graph->markov_w2] = "markov_w2.weight";
  // Its ring's: the drafter stage's, as its own steps name them.
  std::map<std::uint32_t, std::string> step_leaves = StepLeaves(f->spec, *step, layouts);
  for (auto& [id, name] : step_leaves) {
    if (name.starts_with("dspark.")) name = name.substr(7);
  }
  const std::vector<std::size_t> reordered = ExpectSameSteps(
      graph->core.nodes, leaves, *step, step_leaves, TokensOf(*step), f->d.blocks.experts_used);
  std::string unexpected;
  for (const std::size_t node : reordered) {
    const ggml_tensor* t = graph->core.nodes[node];
    if (!Expected(t)) unexpected += std::format(" {}:{}", node, ggml_op_desc(t));
  }
  testing::Test::RecordProperty("draft block reordered", static_cast<int>(reordered.size()));
  EXPECT_EQ(unexpected, "") << reordered.size() << " of " << graph->core.nodes.size()
                            << " entries out of order";
}

}  // namespace
}  // namespace llmp::components
