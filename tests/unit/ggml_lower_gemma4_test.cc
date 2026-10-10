// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML lowering (kernels/ggml/lower.h) of Gemma 4's steps against
// today's graph builders (kernels/ggml/gemma4_graph.h, gemma4_assistant_
// graph.h), structurally, on the CPU, under each serving recipe today's
// runner plans (runtime/serving.cc's Gemma options): within the qualified
// envelope (a context of at most 8,192 and four owners: every row to the
// head, the fused norm-mul, norm-rope and norm-add, the 26B-A4B's fused
// routing and scaled reduce, the 31B's quantized GeGLU, owner attention
// over joined one-row owners of equal reads, and at 4,096 and two owners
// bounded roots over common reads), and outside it (the frontier head's
// narrowed final layer and the fused norm-mul alone); with device masks
// and host masks. For every shape the runner measures at setup
// (gemma4_runner.cc Setup's loop), the step planned under the gemma4
// mapping's policy for that envelope and lowered launches the builder's
// GGML nodes: the same operation, type, extents, strides, view offset,
// operation parameters and flags, over the same operands (each a view's
// type, extents, strides and offset of the same storage: a leaf by what it
// binds, the weight, the layer's, owner's and tensor's cache, the step
// input; a computed root by its own description, recursively), in the
// same order, and PlanGraph's steps over them are the builder's (the same
// implementations over the same nodes). Steps with owner attention compare
// as multisets: the builder's owner transform (gemma4_attention.cc) leaves
// the per-owner views it replaced in the graph and orders the later
// owners' stores and the joined masks where the transform reaches them,
// which changes no launch's operands. Views, reshapes and permutes launch
// nothing and are compared as the operands they describe. The assistant's
// draft steps lower to its builder's graph with host masks (its frozen
// reads' masks) the same way.

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
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/gemma4_assistant_graph.h"
#include "kernels/ggml/gemma4_graph.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/tensors.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "planner/pattern.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma4.h"
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

// A serving envelope and today's recipe there (serving.cc Gemma::Options:
// the candidate within a context of 8,192 and four owners, bounded at
// 4,096 and two).
struct Recipe {
  std::string name;
  std::uint32_t context = 0;
  std::uint32_t slots = 0;
  std::uint32_t max_rows = 0;
  bool candidate = false;
  bool bounded = false;
};

std::vector<Recipe> Recipes(std::uint32_t size) {
  const std::uint32_t rows = size == 26 ? 1024 : 256;  // the envelope's chunks
  return {
      {.name = "bounded",
       .context = 4096,
       .slots = 2,
       .max_rows = rows,
       .candidate = true,
       .bounded = true},
      {.name = "envelope", .context = 8192, .slots = 4, .max_rows = rows, .candidate = true},
      {.name = "outside", .context = 16384, .slots = 2, .max_rows = 128},
  };
}

kg::DeviceChoices LegacyChoices(std::uint32_t size, const Recipe& r) {
  kg::DeviceChoices d = Device();
  d.fuse_norms = true;
  d.fuse_norm_rope = r.candidate;
  d.fuse_norm_add = r.candidate;
  d.fuse_gemma_route = r.candidate && size == 26;
  d.fuse_gemma_reduce = r.candidate && size == 26;
  d.fuse_quant_glu = r.candidate && size == 31;
  return d;
}

kg::Gemma4GraphOptions LegacyOptions(const Recipe& r, bool device_masks) {
  kg::Gemma4GraphOptions o;
  o.device_masks = device_masks;
  o.narrow_final = !r.candidate;  // frontier_head outside the envelope
  o.attention_mode =
      r.candidate ? kg::Gemma4AttentionMode::kOwners : kg::Gemma4AttentionMode::kIndependent;
  o.common_owner_reads = r.bounded;
  o.bounded_owner_roots = r.bounded;
  return o;
}

// ------------------------------------------------------------ the shapes

enum class Out : std::uint8_t { kHead, kAll, kStateOnly, kGreedy };
struct Case {
  kg::Gemma4ChunkShape legacy;
  pl::StepShape ir;
  std::string Name() const {
    std::string s;
    for (const auto& seg : legacy.segments) {
      s += std::format("[{}:{}@{} {}/{}]", seg.slot, seg.rows, seg.n_past, seg.global_n_kv,
                       seg.local_n_kv);
    }
    return s + std::format(" outputs {}{}{}", legacy.outputs,
                           legacy.output_mode == kg::Gemma4OutputMode::kStateOnly ? " state" : "",
                           legacy.greedy ? " greedy" : "");
  }
};

// gemma4_runner.cc Setup's measurement loop at the recipe's options.
std::vector<Case> SetupCases(const md::Gemma4Profile& p, const md::Gemma4StateLayout& layout,
                             const Recipe& r) {
  std::vector<Case> out;
  const std::uint32_t head_rows = r.slots;
  std::vector<std::int32_t> tokens(r.max_rows, 1);
  for (const auto row_budget : {r.max_rows, head_rows}) {
    for (std::uint32_t count = 1; count <= r.slots; ++count) {
      for (const auto& owner_rows :
           llmp::engine::support::ChunkMeasurementRows(row_budget, row_budget, count, false)) {
        const bool mixed = r.bounded && count == 2 &&
                           std::ranges::all_of(owner_rows, [](auto n) { return n == 1; });
        for (const std::uint32_t endpoint : {0U, 1U, 2U, 3U}) {
          if (endpoint >= 2 && !mixed) continue;
          std::vector<md::Gemma4Segment> segments;
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
              if (layout.local_cells > local_limit)
                common_limit = std::min(common_limit, local_limit);
              past = std::min(past, static_cast<std::uint32_t>(common_limit - 1));
            }
            segments.push_back({i, past, std::span(tokens).first(rows)});
          }
          auto in = md::Gemma4Chunk(p, layout, segments, false);
          EXPECT_TRUE(in.has_value());
          if (!in) return out;
          for (const Out o : {Out::kHead, Out::kAll, Out::kStateOnly, Out::kGreedy}) {
            const bool all = o == Out::kAll;
            if (all ? row_budget != head_rows : row_budget != r.max_rows) continue;
            Case c;
            for (const auto& s : in->segments) {
              c.legacy.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
              const std::uint32_t outputs = o == Out::kStateOnly ? 0U : all ? s.rows : 1U;
              c.ir.segments.push_back({.slot = static_cast<std::uint8_t>(s.slot),
                                       .rows = s.rows,
                                       .past = s.n_past,
                                       .read_cells = s.global_n_kv,
                                       .outputs = outputs,
                                       .want_head = o != Out::kStateOnly,
                                       .greedy = o == Out::kGreedy});
            }
            c.legacy.output_mode = o == Out::kStateOnly ? kg::Gemma4OutputMode::kStateOnly
                                                        : kg::Gemma4OutputMode::kHead;
            c.legacy.greedy = o == Out::kGreedy;
            c.legacy.outputs = o == Out::kStateOnly ? 0U
                               : all                ? static_cast<std::uint32_t>(in->tokens.size())
                                                    : count;
            out.push_back(std::move(c));
          }
        }
      }
    }
    if (head_rows == r.max_rows) break;
  }
  return out;
}

// ------------------------------------------------------------ comparison

// Each launched node by what it computes and over what (see the header):
// a hash of its operation, type, extents, strides, view offset, operation
// parameters, flags and operands, an operand a view's description of its
// storage root's (a leaf's name, or a computed root's own hash).
class Canon {
 public:
  // `cache_extents` off: a cache leaf by its name and type alone (the
  // assistant's builder binds each read's first cells as a leaf of their
  // width; its views give the extents read).
  Canon(const std::map<const ggml_tensor*, std::string>& leaves, bool cache_extents)
      : leaves_(leaves), cache_extents_(cache_extents) {}

  static bool Launches(const ggml_tensor* t) {
    return !(t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
             t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE);
  }

  std::uint64_t Node(const ggml_tensor* t) {
    if (const auto at = memo_.find(t); at != memo_.end()) return at->second;
    std::uint64_t h = kSeed;
    if (t->op == GGML_OP_NONE) {
      const auto leaf = leaves_.find(t);
      const std::string name = leaf == leaves_.end() ? std::string("?leaf") : leaf->second;
      Mix(h, name);
      Mix(h, t->type);
      if (cache_extents_ || !(name.starts_with("k.") || name.starts_with("v."))) {
        for (const auto n : t->ne) Mix(h, n);
        for (const auto n : t->nb) Mix(h, n);
      }
      Mix(h, t->flags);
    } else {
      Mix(h, t->op);
      Mix(h, t->type);
      for (const auto n : t->ne) Mix(h, n);
      for (const auto n : t->nb) Mix(h, n);
      Mix(h, t->view_offs);
      Mix(h, t->flags);
      MixBytes(h, t->op_params, sizeof(t->op_params));
      for (const ggml_tensor* s : t->src) Mix(h, s == nullptr ? std::uint64_t{0} : Operand(s));
    }
    memo_[t] = h;
    return h;
  }

  // A source as the node reads it: a view by its description and root.
  std::uint64_t Operand(const ggml_tensor* s) {
    if (s->view_src == nullptr) return Node(s);
    std::uint64_t h = kSeed ^ 0x5bd1e995U;
    Mix(h, s->type);
    for (const auto n : s->ne) Mix(h, n);
    for (const auto n : s->nb) Mix(h, n);
    Mix(h, s->view_offs);
    Mix(h, Node(s->view_src));
    return h;
  }

  // A plan step: its implementation over its nodes (a view of a fused
  // window by its description).
  std::uint64_t Step(const kg::PlanStep& step) {
    std::uint64_t h = kSeed;
    Mix(h, std::string(step.implementation));
    for (const ggml_tensor* n : step.nodes) Mix(h, Launches(n) ? Node(n) : Operand(n));
    return h;
  }

 private:
  static constexpr std::uint64_t kSeed = 1469598103934665603ULL;
  template <typename T>
  static void Mix(std::uint64_t& h, T v) {
    MixBytes(h, &v, sizeof(v));
  }
  static void Mix(std::uint64_t& h, const std::string& s) {
    MixBytes(h, s.data(), s.size());
    Mix(h, s.size());
  }
  static void MixBytes(std::uint64_t& h, const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
      h ^= p[i];
      h *= 1099511628211ULL;
    }
  }
  const std::map<const ggml_tensor*, std::string>& leaves_;
  bool cache_extents_ = true;
  std::map<const ggml_tensor*, std::uint64_t> memo_;
};

struct Side {
  std::vector<std::uint64_t> launched;  // in order
  std::vector<std::uint64_t> steps;     // in order
  std::vector<std::string> described;   // the launched nodes, for messages
  std::vector<std::string> step_names;  // the steps, for messages
};

Side Describe(std::span<ggml_tensor* const> nodes, const kg::GraphPlan& plan,
              const std::map<const ggml_tensor*, std::string>& leaves, bool cache_extents = true) {
  Canon canon(leaves, cache_extents);
  Side out;
  for (const ggml_tensor* n : nodes) {
    if (!Canon::Launches(n)) continue;
    out.launched.push_back(canon.Node(n));
    // For a failure's message: the node and each operand's description.
    std::string text =
        std::format("{}[{},{},{},{}] {} flags {} params {} {} {} {}", ggml_op_desc(n), n->ne[0],
                    n->ne[1], n->ne[2], n->ne[3], ggml_get_name(n), n->flags, n->op_params[0],
                    n->op_params[1], n->op_params[2], n->op_params[3]);
    for (const ggml_tensor* s : n->src) {
      if (s == nullptr) continue;
      const ggml_tensor* root = s->view_src != nullptr ? s->view_src : s;
      const auto leaf = leaves.find(root);
      text += std::format(" | {} {} [{},{},{},{}] nb [{},{},{},{}] +{} of {} ({:x})",
                          ggml_op_desc(s), ggml_type_name(s->type), s->ne[0], s->ne[1], s->ne[2],
                          s->ne[3], s->nb[0], s->nb[1], s->nb[2], s->nb[3], s->view_offs,
                          leaf != leaves.end() ? leaf->second : std::string(ggml_op_desc(root)),
                          canon.Operand(s));
    }
    out.described.push_back(std::move(text));
  }
  for (const kg::PlanStep& s : plan.steps) {
    out.steps.push_back(canon.Step(s));
    std::string text(s.implementation);
    for (const ggml_tensor* n : s.nodes) text += std::format(" {}", ggml_op_desc(n));
    out.step_names.push_back(std::move(text));
  }
  return out;
}

void ExpectSame(const Side& legacy, const Side& lowered, bool ordered) {
  ASSERT_EQ(legacy.launched.size(), lowered.launched.size());
  if (ordered) {
    for (std::size_t i = 0; i < legacy.launched.size(); ++i) {
      ASSERT_EQ(legacy.launched[i], lowered.launched[i])
          << "launched node " << i << ": " << legacy.described[i] << " / " << lowered.described[i];
    }
    for (std::size_t i = 0; i < std::min(legacy.steps.size(), lowered.steps.size()); ++i) {
      ASSERT_EQ(legacy.steps[i], lowered.steps[i])
          << "step " << i << ": " << legacy.step_names[i] << " / " << lowered.step_names[i];
    }
    ASSERT_EQ(legacy.steps.size(), lowered.steps.size());
    return;
  }
  std::multiset<std::uint64_t> a(legacy.launched.begin(), legacy.launched.end());
  std::multiset<std::uint64_t> b(lowered.launched.begin(), lowered.launched.end());
  if (a != b) {
    for (std::size_t i = 0; i < legacy.launched.size(); ++i) {
      if (!b.contains(legacy.launched[i])) {
        ADD_FAILURE() << "the lowering has no " << legacy.described[i] << " (builder's node " << i
                      << ")";
        break;
      }
    }
    for (std::size_t i = 0; i < lowered.launched.size(); ++i) {
      if (!a.contains(lowered.launched[i])) {
        ADD_FAILURE() << "the builder has no " << lowered.described[i] << " (lowered node " << i
                      << ")";
        break;
      }
    }
    FAIL() << "the launched nodes differ";
  }
  const std::multiset<std::uint64_t> sa(legacy.steps.begin(), legacy.steps.end());
  const std::multiset<std::uint64_t> sb(lowered.steps.begin(), lowered.steps.end());
  if (sa != sb) {
    for (std::size_t i = 0; i < legacy.steps.size(); ++i) {
      if (!sb.contains(legacy.steps[i])) {
        ADD_FAILURE() << "the lowering has no step " << legacy.step_names[i];
        break;
      }
    }
    for (std::size_t i = 0; i < lowered.steps.size(); ++i) {
      if (!sa.contains(lowered.steps[i])) {
        ADD_FAILURE() << "the builder has no step " << lowered.step_names[i];
        break;
      }
    }
    FAIL() << "the plan steps differ";
  }
}

// ------------------------------------------------------------ the builders

std::string WeightName(const std::vector<md::Gemma4Resource>& resources, bool array,
                       std::uint32_t index) {
  std::uint32_t seen = 0;
  for (const md::Gemma4Resource& r : resources) {
    if (r.expert_array != array) continue;
    if (seen++ == index) return "w." + r.roles.front();
  }
  return "w.?";
}

struct Legacy {
  std::optional<kg::TensorArena> arena;
  std::vector<ggml_tensor*> nodes;
  std::vector<ggml_tensor*> inputs;
  std::map<const ggml_tensor*, std::string> leaves;
  std::map<std::string, std::uint64_t> expert_stride;  // by weight name
  kg::GraphPlan plan;
  bool owners = false;  // the owner transform applied
};

// Every leaf at an address of its own, every computed node at another.
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

Legacy BuildLegacy(std::uint32_t size, const Recipe& r, const Case& c,
                   const md::Gemma4StateLayout& state, bool device_masks) {
  Legacy out;
  const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
  const auto resources = llmp::test_support::gemma4::Resources(size);
  const auto binding = md::BindGemma4(p, "gemma4", resources);
  EXPECT_TRUE(binding.has_value());
  if (!binding) return out;
  const kg::Gemma4GraphOptions options = LegacyOptions(r, device_masks);
  auto arena =
      kg::TensorArena::Create(kg::Gemma4GraphTensors(p, c.legacy.segments.size(), options));
  EXPECT_TRUE(arena.has_value());
  out.arena.emplace(std::move(*arena));
  auto g = kg::BuildGemma4Graph(*out.arena, p, *binding, state, c.legacy, options);
  EXPECT_TRUE(g.has_value()) << g.error().detail;
  if (!g) return out;
  out.nodes = g->nodes;
  out.inputs = g->inputs;
  out.owners = g->attention_quad_mask != 0;
  const auto add = [&](const ggml_tensor* t, std::string name) {
    if (t != nullptr) out.leaves[t] = std::move(name);
  };
  add(g->tokens, "tokens");
  add(g->positions, "positions");
  add(g->out_ids, "out_ids");
  for (std::size_t s = 0; s < g->segments.size(); ++s) {
    const auto& seg = g->segments[s];
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
  for (const auto& w : g->weights) {
    const std::string name = WeightName(resources, w.resource.expert_array, w.resource.index);
    add(w.tensor, name);
    if (w.resource.expert_array) out.expert_stride[name] = w.tensor->nb[2];
  }
  std::vector<ggml_tensor*> keep;
  for (ggml_tensor* k : {g->logits, g->greedy, g->hidden}) {
    if (k != nullptr) keep.push_back(k);
  }
  BindAll(out.nodes);
  auto plan = kg::PlanGraph(out.nodes, false, LegacyChoices(size, r), keep);
  EXPECT_TRUE(plan.has_value()) << plan.error().detail;
  if (plan) out.plan = std::move(*plan);
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
                      item.contract.window_or_ratio != 0 ? "local" : "global", st.ref.segment);
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

Lowered Lower(Ir& ir_, std::uint16_t stage, const pl::StepShape& shape, bool device_masks,
              const std::map<std::string, std::uint64_t>& expert_stride) {
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
  std::map<ir::WeightRef, std::uint64_t> strides;
  for (const co::StepWeight& w : step->weights) {
    const auto at = expert_stride.find("w." + w.parts.front().first);
    if (at != expert_stride.end()) strides[w.ref] = at->second;
  }
  out.step.emplace(std::move(*step));
  out.plan.emplace(std::move(*plan));
  const kn::Addresses addresses{};
  const kn::LowerInput input{.plan = *out.plan,
                             .step = *out.step,
                             .shape = shape,
                             .addresses = addresses,
                             .context = ir_.context};
  auto lowered =
      kg::LowerStep(input, ir_.registry, Device(),
                    {.device_masks = device_masks,
                     .frozen_host_masks = true,
                     .expert_stride = [&strides](const ir::WeightRef& ref) -> std::uint64_t {
                       const auto at = strides.find(ref);
                       return at == strides.end() ? 0 : at->second;
                     }});
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

// The inputs the step stages: the builder's inputs, each as its slot.
void ExpectInputs(std::span<ggml_tensor* const> inputs,
                  const std::map<const ggml_tensor*, std::string>& legacy, const Lowered& w) {
  std::multiset<std::string> want, got;
  for (const ggml_tensor* t : inputs) {
    want.insert(std::format("{} {}", legacy.at(t), ggml_nbytes(t)));
  }
  std::map<std::uint64_t, const ggml_tensor*> by_address;
  for (const auto& [t, name] : w.leaves) {
    by_address[static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data))] = t;
  }
  for (const kn::InputSlot& slot : w.lowered->inputs) {
    const ggml_tensor* t = by_address.at(slot.device_address);
    got.insert(std::format("{} {}", w.leaves.at(t), slot.bytes));
  }
  EXPECT_EQ(got, want);
}

Ir TargetIr(std::uint32_t size, const Recipe& r) {
  const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
  const auto values = llmp::spec::mappings::Gemma4V0Values(p);
  Ir ir_;
  ir_.sources.push_back(llmp::test_support::Gemma4Source(size));
  ir_.spec =
      *llmp::spec::Synthesize(llmp::spec::mappings::Gemma4Mapping(), ir_.sources[0], *values);
  ir_.tables.emplace_back(ir_.sources[0].tensors);
  auto layout = co::StageLayout(ir_.spec, 0, r.context, r.max_rows, kExtent);
  EXPECT_TRUE(layout.has_value()) << layout.error();
  ir_.layouts.push_back(std::move(*layout));
  ir_.stages.push_back(&ir_.layouts[0]);
  const auto text = llmp::spec::DefaultPolicy(llmp::spec::mappings::Gemma4Mapping(), *values,
                                              {.context = r.context, .slots = r.slots});
  EXPECT_TRUE(text.has_value());
  ir_.policy = pl::ParsePolicy(*text).value();
  ir_.context = r.context;
  return ir_;
}

class GgmlLowerGemma4 : public testing::TestWithParam<std::uint32_t> {};

TEST_P(GgmlLowerGemma4, ServingRecipesLowerToTheBuildersGraphs) {
  const std::uint32_t size = GetParam();
  const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
  std::size_t shapes = 0, owner_forms = 0, launches = 0, routes = 0;
  for (const Recipe& r : Recipes(size)) {
    const auto state = md::Gemma4State(p, r.context, r.max_rows);
    ASSERT_TRUE(state.has_value());
    Ir ir_ = TargetIr(size, r);
    ASSERT_EQ(ir_.layouts[0].context, r.context);
    // The policy is the recipe's: narrowing only outside the envelope.
    EXPECT_EQ(pl::Disabled(ir_.policy, "narrow_final"), r.candidate) << r.name;
    EXPECT_EQ(pl::Disabled(ir_.policy, "owner_attention"), !r.candidate) << r.name;
    for (const Case& c : SetupCases(p, *state, r)) {
      for (const bool device_masks : {true, false}) {
        SCOPED_TRACE(std::format("{} {}{}", r.name, c.Name(), device_masks ? "" : ", host masks"));
        const Legacy l = BuildLegacy(size, r, c, *state, device_masks);
        ASSERT_FALSE(l.nodes.empty());
        const Lowered w = Lower(ir_, 0, c.ir, device_masks, l.expert_stride);
        ASSERT_TRUE(w.lowered.has_value());
        // The owner forms where the builder transformed, and nowhere else.
        EXPECT_EQ(w.owners != 0, l.owners);
        const Side a = Describe(l.nodes, l.plan, l.leaves);
        const Side b = Describe(w.lowered->nodes, w.lowered->plan, w.leaves);
        ExpectSame(a, b, !l.owners);
        if (::testing::Test::HasFatalFailure()) return;
        ExpectInputs(l.inputs, l.leaves, w);
        std::uint32_t outputs = 0;
        for (const kn::OutputSlot& o : w.lowered->outputs) {
          EXPECT_LT(o.segment, c.ir.segments.size());
          ++outputs;
        }
        EXPECT_EQ(outputs, c.legacy.output_mode == kg::Gemma4OutputMode::kStateOnly
                               ? 0U
                               : c.ir.segments.size());
        owner_forms += l.owners ? 1 : 0;
        routes += w.routes;
        launches += a.launched.size();
        ++shapes;
      }
    }
  }
  std::println(
      "gemma{}: {} lowered steps ({} with owner attention), {} launches, {} fused routings", size,
      shapes, owner_forms, launches, routes);
  EXPECT_GT(shapes, 0U);
  EXPECT_GT(owner_forms, 0U);
  EXPECT_EQ(routes != 0, size == 26);
}

INSTANTIATE_TEST_SUITE_P(Gemma4, GgmlLowerGemma4, testing::Values(26U, 31U),
                         [](const testing::TestParamInfo<std::uint32_t>& info) {
                           return std::format("gemma{}", info.param);
                         });

// Owner attention's Gemma 4 forms (16 or 32 heads, D 512) are the GB10's
// alone, as today's runner keeps independent attention on other devices;
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
    EXPECT_EQ(owners->accepts(q), d == 256 && heads == 8 && kg::CompiledFor(86));
  }
}

// The assistant's draft passes: one query an owner at its target's
// committed endpoint, over the target's last local and global caches,
// with host masks of its frozen reads (its builder's are inputs), lowered
// to its builder's graph; with device masks, the frozen producer.
TEST(GgmlLowerGemma4Test, AssistantLowersToItsBuildersGraphs) {
  constexpr std::uint32_t kContext = 8192;
  constexpr std::uint32_t kMaxRows = 64;
  std::size_t compared = 0;
  for (const std::uint32_t size : {26U, 31U}) {
    SCOPED_TRACE(size);
    const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
    const auto& a = size == 26 ? md::Gemma4Assistant26() : md::Gemma4Assistant31();
    const auto target_resources = llmp::test_support::gemma4::Resources(size);
    const auto assistant_resources = llmp::test_support::gemma4::AssistantResources(size);
    const auto target_binding = md::BindGemma4(p, "gemma4", target_resources);
    const auto assistant_binding =
        md::BindGemma4Assistant(a, "gemma4-assistant", assistant_resources);
    ASSERT_TRUE(target_binding.has_value() && assistant_binding.has_value());
    const auto values = llmp::spec::mappings::Gemma4AssistantV0Values(a, p);
    ASSERT_TRUE(values.has_value());
    const auto target_values = llmp::spec::mappings::Gemma4V0Values(p);
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
      auto layout = co::StageLayout(ir_.spec, stage, kContext, kMaxRows, kExtent);
      ASSERT_TRUE(layout.has_value()) << layout.error();
      ir_.layouts.push_back(std::move(*layout));
    }
    for (const auto& l : ir_.layouts) ir_.stages.push_back(&l);
    const auto text =
        llmp::spec::DefaultPolicy(llmp::spec::mappings::Gemma4AssistantMapping(), *values);
    ASSERT_TRUE(text.has_value());
    ir_.policy = pl::ParsePolicy(*text).value();
    ir_.context = kContext;
    const auto state = md::Gemma4State(p, kContext, kMaxRows);
    ASSERT_TRUE(state.has_value());
    // Each assistant resource's name.
    std::map<std::uint32_t, std::string> names;
    for (std::uint32_t i = 0; i < assistant_resources.size(); ++i) {
      names[i] = "w." + assistant_resources[i].roles.front();
    }
    for (const auto& prefixes : std::vector<std::vector<std::uint32_t>>{
             {300}, {300, 1500}, {7, 2048, 6000}, {1, 256, 1300, 8191}}) {
      SCOPED_TRACE(prefixes.size());
      kg::Gemma4AssistantShape legacy_shape;
      pl::StepShape shape{.stage = 1, .phase = pl::Phase::kDraft, .segments = {}};
      for (std::size_t i = 0; i < prefixes.size(); ++i) {
        const std::uint32_t read = std::min(kContext, (prefixes[i] + 255) / 256 * 256);
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
      std::map<const ggml_tensor*, std::string> leaves;
      leaves[legacy->tokens] = "tokens";
      leaves[legacy->features] = "features";
      leaves[legacy->positions] = "positions";
      leaves[legacy->embedding] = "w.token_embd.weight";
      for (std::size_t s = 0; s < legacy->segments.size(); ++s) {
        const auto& seg = legacy->segments[s];
        for (std::size_t kind = 0; kind < 2; ++kind) {
          const char* type = kind == 0 ? "local" : "global";
          leaves[seg.caches[kind].first] = std::format("k.{}.{}", type, s);
          leaves[seg.caches[kind].second] = std::format("v.{}.{}", type, s);
          leaves[seg.masks[kind]] = std::format("mask.{}.{}", s, type);
        }
      }
      for (const auto& w : legacy->weights) leaves[w.tensor] = names.at(w.resource.index);
      BindAll(legacy->nodes);
      auto plan = kg::PlanGraph(legacy->nodes, false, Device(),
                                std::vector<ggml_tensor*>{legacy->logits, legacy->next_features});
      ASSERT_TRUE(plan.has_value()) << plan.error().detail;
      const Lowered w = Lower(ir_, 1, shape, false, {});
      ASSERT_TRUE(w.lowered.has_value());
      ExpectSame(Describe(legacy->nodes, *plan, leaves, false),
                 Describe(w.lowered->nodes, w.lowered->plan, w.leaves, false), true);
      if (::testing::Test::HasFatalFailure()) return;
      ExpectInputs(legacy->inputs, leaves, w);
      for (const kn::InputSlot& slot : w.lowered->inputs) {
        if (slot.kind == kn::InputKind::kMask) EXPECT_TRUE(slot.cells.frozen);
      }
      // Device masks: the frozen producer in each mask's place.
      const Lowered d = Lower(ir_, 1, shape, true, {});
      ASSERT_TRUE(d.lowered.has_value());
      std::size_t producers = 0;
      for (const ggml_tensor* n : d.lowered->nodes) {
        if (kg::LlmpOpOf(n) == kg::LlmpOp::kGemma4Mask) {
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
  const auto& p = md::Gemma4_26BA4B();
  const auto& a = md::Gemma4Assistant26();
  Ir ir_;
  ir_.sources.push_back(llmp::test_support::Gemma4Source(26));
  ir_.sources.push_back(llmp::test_support::Gemma4AssistantSource(26));
  const auto target_values = llmp::spec::mappings::Gemma4V0Values(p);
  const auto values = llmp::spec::mappings::Gemma4AssistantV0Values(a, p);
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
    auto layout = co::StageLayout(ir_.spec, stage, kContext, kMaxRows, kExtent);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    ir_.layouts.push_back(std::move(*layout));
  }
  for (const auto& l : ir_.layouts) ir_.stages.push_back(&l);
  ir_.policy =
      pl::ParsePolicy(*llmp::spec::DefaultPolicy(llmp::spec::mappings::Gemma4Mapping(),
                                                 *target_values, {.context = kContext, .slots = 2}))
          .value();
  ir_.context = kContext;
  const pl::StepShape shape{
      .segments = {{.slot = 0, .rows = 7, .past = 0, .read_cells = 256, .outputs = 1},
                   {.slot = 1, .rows = 3, .past = 300, .read_cells = 512, .outputs = 3}}};
  const Lowered w = Lower(ir_, 0, shape, true, {});
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
