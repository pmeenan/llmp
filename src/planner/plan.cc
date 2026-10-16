// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <iterator>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"
#include "planner/lanes.h"
#include "planner/pattern.h"
#include "planner/placement.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::planner {
namespace {

using ir::NodeId;
using ir::TensorId;

// Decompositions may emit composite ops and fused kinds of their own; a
// plan takes at most this many rounds of them.
constexpr int kMaxPasses = 4;

std::unexpected<PlanFailure> Failed(PlanError error, std::string detail) {
  return std::unexpected(PlanFailure{.error = error, .detail = std::move(detail)});
}

std::string Describe(const ir::Graph& graph, NodeId id) {
  const ir::Node& n = graph.node(id);
  std::string text = std::format("node {} ({}", id.v, ir::OpKeyName(n.key));
  for (const TensorId s : n.sources()) {
    if (s.valid()) {
      const ir::TensorDesc& t = graph.tensor(s);
      text += std::format(" {}[{},{},{},{}]", format::Info(t.format.encoding).name, t.ne(0),
                          t.ne(1), t.ne(2), t.ne(3));
    }
  }
  return text + ")";
}

// `graph` with the flagged nodes replaced by their decompositions, those
// `drop` flags (if given) left out, and each of `repeats` (if given) left
// out for the earlier node it names; `map` receives each old tensor's new
// id (none for a dropped node's) and `from` each new node's old one.
std::expected<ir::Graph, ir::VerifyError> ExpandNodes(
    const ir::Graph& graph, const std::vector<bool>& expand, std::vector<TensorId>& map,
    std::vector<NodeId>& from, const std::vector<bool>* drop = nullptr,
    const std::vector<NodeId>* repeats = nullptr) {
  ir::Builder b;
  map.assign(graph.tensor_count(), TensorId{});
  from.clear();
  RewriteContext rc(graph, b, map);
  std::vector<TensorId> sources;
  std::vector<format::Encoding> encodings;
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const ir::Node& node = graph.nodes()[i];
    if (drop != nullptr && (*drop)[i]) continue;
    if (repeats != nullptr && (*repeats)[i].valid()) {
      // A repeat: the earlier node's outputs stand for its own.
      const ir::Node& earlier = graph.node((*repeats)[i]);
      for (std::size_t k = 0; k < node.nout; ++k) rc.Set(node.out[k], rc.Map(earlier.out[k]));
      continue;
    }
    if (!expand[i]) {
      rc.Copy(node);
    } else {
      b.SetComponent(node.component);
      if (node.batch == ir::Batch::kPerOwner && node.nout > 0) {
        b.SetOwner(graph.tensor(node.out[0]).segment);
      } else {
        b.SetJoined();
      }
      sources.clear();
      for (const TensorId s : node.sources()) {
        sources.push_back(rc.Map(s));
      }
      encodings.clear();
      for (const TensorId o : node.outputs()) {
        encodings.push_back(graph.tensor(o).format.encoding);
      }
      const ir::Outputs outs = ir::Decompose(b, node.key, node.params, sources, encodings);
      for (std::size_t k = 0; k < node.nout && k < outs.n; ++k) {
        rc.Set(node.out.at(k), outs[k]);
      }
    }
    if (!b.ok()) {
      ir::VerifyError e = *b.error();
      e.node = i;
      return std::unexpected(std::move(e));
    }
    from.resize(b.node_count(), NodeId{i});
  }
  for (const TensorId o : graph.outputs()) {
    b.Output(rc.Map(o));
  }
  for (const TensorId k : graph.keep()) {
    b.Keep(rc.Map(k));
  }
  return std::move(b).Finish();
}

// A node whose run matters beyond its outputs: a leaf, a write in place (a
// cache store, a fused node's aliased output) or communication.
bool Effect(const ir::Node& n) {
  switch (n.key.op) {
    case ir::Op::kInput:
    case ir::Op::kWeight:
    case ir::Op::kState:
    case ir::Op::kCacheStore:
    case ir::Op::kAllReduce:
    case ir::Op::kAllGather:
    case ir::Op::kAllToAll:
    case ir::Op::kSend:
    case ir::Op::kRecv:
      return true;
    case ir::Op::kFused:
      for (std::size_t i = 0; i < n.nout; ++i)
        if (ir::AliasedSource(n.key, i)) return true;
      return false;
    default:
      return false;
  }
}

// The roots `graph` writes in place, by tensor: a cache store's target, a
// fused node's aliased source, a state, and every root a node with an
// Effect (but a weight or input leaf) reads or makes. A value read from
// one may differ between two reads, so it is neither another node's
// repeat nor a prologue's.
std::vector<bool> WrittenRoots(const ir::Graph& graph) {
  std::vector<bool> written(graph.tensor_count(), false);
  const auto mark = [&](TensorId t) {
    if (t.valid()) written.at(graph.root(t).v) = true;
  };
  for (const ir::Node& n : graph.nodes()) {
    if (!Effect(n) || n.key.op == ir::Op::kInput || n.key.op == ir::Op::kWeight) continue;
    for (const TensorId t : n.sources()) mark(t);
    for (const TensorId t : n.outputs()) mark(t);
    if (n.key.op == ir::Op::kFused)
      for (std::size_t i = 0; i < n.nout; ++i)
        if (const auto alias = ir::AliasedSource(n.key, i)) mark(n.src.at(*alias));
  }
  return written;
}

// The nodes of `graph` nothing needs: none of their outputs is the
// graph's output or kept, or read by a node that is needed, and they have
// no Effect. A rewrite leaves such nodes where its fused node reads their
// sources directly (one pass, readers after their producers). Empty when
// every node is needed.
std::vector<bool> Unread(const ir::Graph& graph) {
  std::vector<bool> needed(graph.tensor_count(), false);
  for (const auto list : {graph.outputs(), graph.keep()})
    for (const TensorId t : list) {
      needed.at(t.v) = true;
      needed.at(graph.root(t).v) = true;
    }
  std::vector<bool> unread(graph.nodes().size(), false);
  bool any = false;
  for (std::uint32_t i = static_cast<std::uint32_t>(graph.nodes().size()); i-- > 0;) {
    const ir::Node& n = graph.nodes()[i];
    if (!Effect(n) && std::ranges::none_of(n.outputs(), [&](TensorId o) {
          return needed.at(o.v) || needed.at(graph.root(o).v);
        })) {
      unread[i] = any = true;
      continue;
    }
    for (const TensorId src : n.sources())
      if (src.valid()) {
        needed.at(src.v) = true;
        needed.at(graph.root(src).v) = true;
      }
  }
  if (!any) unread.clear();
  return unread;
}

// For each node of `graph` that repeats an earlier one, the earlier one:
// the same operation and parameters over the same sources (as repeats
// resolve), with outputs of the same roles, formats, strides and segments,
// whichever component emitted it (two blocks joining the same rows). Only
// nodes with sources and no Effect whose outputs are roots the graph
// neither outputs nor keeps (nor an Effect reads) repeat. Empty when none
// does.
std::vector<NodeId> Repeats(const ir::Graph& graph) {
  std::vector<TensorId> canon(graph.tensor_count());
  for (std::uint32_t t = 0; t < canon.size(); ++t) canon[t] = TensorId{t};
  // Outputs and kept tensors are never another's; a node reading or
  // making a root written in place repeats none (WrittenRoots).
  std::vector<bool> external(graph.tensor_count(), false);
  for (const auto list : {graph.outputs(), graph.keep()})
    for (const TensorId t : list) external.at(t.v) = true;
  const std::vector<bool> written = WrittenRoots(graph);
  const auto eligible = [&](const ir::Node& n) {
    return n.nsrc > 0 && n.nout > 0 && !Effect(n) &&
           std::ranges::none_of(
               n.sources(), [&](TensorId t) { return t.valid() && written.at(graph.root(t).v); }) &&
           std::ranges::none_of(n.outputs(), [&](TensorId o) {
             return external.at(o.v) || graph.tensor(o).is_view() || written.at(graph.root(o).v);
           });
  };
  const auto same = [&](const ir::Node& a, const ir::Node& b) {
    if (a.key != b.key || a.params != b.params || a.nsrc != b.nsrc || a.nout != b.nout ||
        a.batch != b.batch)
      return false;
    for (std::size_t k = 0; k < a.nsrc; ++k) {
      const TensorId x = a.src[k], y = b.src[k];
      if (x.valid() != y.valid() || (x.valid() && canon[x.v] != canon[y.v])) return false;
    }
    for (std::size_t k = 0; k < a.nout; ++k) {
      const ir::TensorDesc& x = graph.tensor(a.out[k]);
      const ir::TensorDesc& y = graph.tensor(b.out[k]);
      if (x.role != y.role || x.format != y.format || x.nb != y.nb || x.segment != y.segment)
        return false;
    }
    return true;
  };
  // Candidates by operation and first source.
  std::map<std::pair<std::uint64_t, std::uint32_t>, std::vector<std::uint32_t>> seen;
  std::vector<NodeId> repeats;
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const ir::Node& n = graph.nodes()[i];
    if (!eligible(n)) continue;
    const TensorId first = n.src[0];
    const std::pair<std::uint64_t, std::uint32_t> key{
        (std::uint64_t{static_cast<std::uint16_t>(n.key.op)} << 16U) |
            static_cast<std::uint16_t>(n.key.fused),
        first.valid() ? canon[first.v].v : TensorId::kNone};
    auto& bucket = seen[key];
    const auto at =
        std::ranges::find_if(bucket, [&](std::uint32_t j) { return same(graph.nodes()[j], n); });
    if (at == bucket.end()) {
      bucket.push_back(i);
      continue;
    }
    if (repeats.empty()) repeats.assign(graph.nodes().size(), NodeId{});
    repeats[i] = NodeId{*at};
    for (std::size_t k = 0; k < n.nout; ++k) canon[n.out[k].v] = graph.nodes()[*at].out[k];
  }
  return repeats;
}

struct Selection {
  std::vector<Selected> steps;
  std::vector<bool> missing;  // by node: no implementation
  bool complete = true;
};

Selected One(const execution::Registry& registry, NodeId id, std::size_t implementation) {
  return Selected{.nodes = {id},
                  .implementation = implementation,
                  .group = registry.at(implementation).capability->group};
}

Selection Select(const ir::Graph& graph, const SelectionView& view,
                 std::span<const OwnerProductScope> scopes) {
  Selection out;
  out.missing.assign(graph.nodes().size(), false);
  MatchContext context(graph, view);
  const execution::Registry& registry = view.registry();
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const NodeId id{i};
    const ir::Node& node = graph.node(id);
    if (LaunchesNothing(node.key) || context.taken(id)) {
      continue;
    }
    bool batched = false;
    for (const Pattern& p : Patterns()) {
      auto m = BatchAt(p, context, id);
      if (!m || std::ranges::any_of(m->nodes, [&](NodeId member) {
            return std::ranges::any_of(
                scopes, [&](const auto& scope) { return graph.producer(scope.result) == member; });
          })) {
        continue;
      }
      // Each member keeps its own selection, for a run of one (lanes.h).
      Selected step{.nodes = m->nodes,
                    .implementation = m->implementation,
                    .group = registry.at(m->implementation).capability->group,
                    .batch = p.name};
      bool singles = true;
      for (const NodeId member : m->nodes) {
        const auto own = view.Candidates(graph, graph.node(member));
        if (own.empty()) {
          singles = false;
          break;
        }
        step.singles.push_back(own.front());
        step.single_groups.push_back(registry.at(own.front()).capability->group);
      }
      if (!singles) {
        continue;
      }
      context.Take(m->nodes);
      out.steps.push_back(std::move(step));
      batched = true;
      break;
    }
    if (batched) {
      continue;
    }
    const auto candidates = view.Candidates(graph, node);
    if (candidates.empty()) {
      out.missing[i] = true;
      out.complete = false;
      continue;
    }
    out.steps.push_back(One(registry, id, candidates.front()));
  }
  return out;
}

// The steps of `steps` that read only `invariant` (planned input ids),
// weights and each other's results (a plan's prologue), and the tensors
// the other steps read from them (kept in place between runs). A step
// with any node that reads anything else, or has an Effect (a state, a
// store, communication), is not the prologue's, and what it computes is
// not invariant; nor is anything a root written in place holds.
struct Prologue {
  std::vector<std::uint32_t> steps;
  std::vector<TensorId> preserved;
  // Of `invariant`, by position, those it holds fixed (the rest, a root
  // written in place, are copied every run).
  std::vector<bool> kept;
};
Prologue PrologueOf(const ir::Graph& graph, std::span<const Selected> steps,
                    std::span<const TensorId> invariant, const RootWithin& within) {
  std::vector<bool> fixed(graph.tensor_count(), false);
  std::vector<bool> demoted(graph.nodes().size(), false);
  // A node with an Effect (a store, a state, communication) is never the
  // prologue's, nor is what a root written in place holds.
  const std::vector<bool> written = WrittenRoots(graph);
  const auto effect = [](const ir::Node& n) { return Effect(n) && n.key.op != ir::Op::kWeight; };
  std::vector<bool> prologue(steps.size(), false);
  for (bool changed = true; changed;) {
    changed = false;
    std::ranges::fill(fixed, false);
    for (const TensorId t : invariant)
      if (!written.at(graph.root(t).v)) fixed.at(t.v) = true;
    for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
      const ir::Node& n = graph.nodes()[i];
      // An input only where declared (above); a weight always.
      if (n.key == ir::Primitive(ir::Op::kInput)) continue;
      bool all = !demoted[i] && !effect(n);
      for (const TensorId src : n.sources())
        if (src.valid() && (!fixed.at(src.v) || written.at(graph.root(src).v))) all = false;
      if (all)
        for (const TensorId o : n.outputs())
          if (!written.at(graph.root(o).v)) fixed.at(o.v) = true;
    }
    // A root placed inside another (RootWithin: a blob's later output, a
    // preparation in its norm's place) shares its bytes: both fixed or
    // neither, or one's run would write the other's kept bytes.
    if (within) {
      for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
        for (const TensorId o : graph.nodes()[i].outputs()) {
          const TensorId root = graph.root(o);
          const TensorId outer = within(graph, root);
          if (!outer.valid() || fixed.at(root.v) == fixed.at(outer.v)) continue;
          const TensorId kept = fixed.at(root.v) ? root : outer;
          const NodeId producer = graph.producer(kept);
          if (producer.valid() && !demoted[producer.v]) {
            demoted[producer.v] = true;
            changed = true;
          }
        }
      }
      if (changed) continue;
    }
    for (std::size_t s = 0; s < steps.size(); ++s) {
      const bool whole = std::ranges::all_of(steps[s].nodes, [&](NodeId id) {
        return std::ranges::all_of(graph.node(id).outputs(),
                                   [&](TensorId o) { return fixed.at(o.v); });
      });
      prologue[s] = whole && !steps[s].nodes.empty();
      if (whole) continue;
      // A step that runs anyway: what it computes is not the prologue's.
      for (const NodeId id : steps[s].nodes) {
        if (!demoted[id.v] && std::ranges::any_of(graph.node(id).outputs(),
                                                  [&](TensorId o) { return fixed.at(o.v); })) {
          demoted[id.v] = true;
          changed = true;
        }
      }
    }
  }
  Prologue out;
  for (std::size_t s = 0; s < steps.size(); ++s) {
    if (prologue[s]) {
      out.steps.push_back(static_cast<std::uint32_t>(s));
      continue;
    }
    for (const NodeId id : steps[s].nodes)
      for (const TensorId src : graph.node(id).sources())
        if (src.valid() && fixed.at(src.v) && !std::ranges::contains(out.preserved, src))
          out.preserved.push_back(src);
  }
  // The step's outputs and kept tensors computed by the prologue stay too.
  for (const auto list : {graph.outputs(), graph.keep()})
    for (const TensorId t : list)
      if (fixed.at(t.v) && !std::ranges::contains(out.preserved, t)) out.preserved.push_back(t);
  // And the root a kept one lies in.
  if (within) {
    for (std::size_t k = 0; k < out.preserved.size(); ++k) {
      const TensorId outer = within(graph, graph.root(out.preserved[k]));
      if (outer.valid() && !std::ranges::contains(out.preserved, outer))
        out.preserved.push_back(outer);
    }
  }
  for (const TensorId t : invariant) out.kept.push_back(fixed.at(t.v));
  return out;
}

}  // namespace

std::string_view PlanErrorName(PlanError error) {
  switch (error) {
    case PlanError::kNoImplementation:
      return "no implementation";
    case PlanError::kPattern:
      return "pattern";
    case PlanError::kPlacement:
      return "placement";
    case PlanError::kBounds:
      return "bounds";
    case PlanError::kPolicy:
      return "policy";
  }
  return "unknown";
}

std::expected<StepPlan, PlanFailure> Plan(ir::Graph primitive, const execution::Registry& registry,
                                          const DeviceFacts& device, const Policy& policy,
                                          const PlanOptions& options) {
  if (auto valid = Validate(policy); !valid) {
    return Failed(PlanError::kPolicy, valid.error());
  }
  if (options.activation_alignment != 0 &&
      (options.activation_alignment & (options.activation_alignment - 1)) != 0)
    return Failed(PlanError::kPlacement, "the activation base alignment is not a power of two");
  if (options.alignment_staged) {
    for (const auto t : *options.alignment_staged) {
      if (!t.valid() || t.v >= primitive.tensor_count() || primitive.tensor(t).is_view() ||
          !std::ranges::contains(primitive.inputs(), t) ||
          (options.staged && !std::ranges::contains(*options.staged, t)))
        return Failed(PlanError::kPlacement, "an aligned input is not actually staged");
    }
  }
  SelectionView view(
      registry, device, policy, options.row_invariant, options.capturable, options.state_spans,
      options.root_alignment,
      {.base = options.activation_alignment,
       .offset = options.alignment,
       .staged = options.alignment_staged ? options.alignment_staged : options.staged,
       .within = options.within});
  std::optional<OwnerRewritten> owners;
  std::vector<OwnerProductScope> scopes;
  std::vector<NodeId> preserve;
  if (options.owner_products != nullptr) {
    if (options.shape != nullptr) {
      for (const auto& p : options.owner_products->products()) {
        if (p.owners.size() != options.shape->segments.size())
          return Failed(PlanError::kPattern, "owner product and step partitions differ");
        for (std::size_t i = 0; i < p.owners.size(); ++i)
          if (p.owners[i].slot != options.shape->segments[i].slot ||
              p.owners[i].rows != options.shape->segments[i].rows)
            return Failed(PlanError::kPattern, "owner product and step rows differ");
      }
    }
    auto split = SplitOwnerProducts(primitive, *options.owner_products);
    if (!split) return Failed(PlanError::kPattern, split.error());
    owners = std::move(*split);
    view.Forward(primitive, owners->rewrite.graph, owners->rewrite.tensors);
    scopes = owners->scopes;
    for (const auto& scope : scopes)
      preserve.push_back(owners->rewrite.graph.producer(scope.result));
  }
  const ir::Graph& pattern_input = owners ? owners->rewrite.graph : primitive;
  auto rewritten = ApplyPatterns(pattern_input, view, options.shape, options.state_spans, preserve);
  if (!rewritten) {
    return Failed(
        PlanError::kPattern,
        std::format("a rewrite at node {}: {}", rewritten.error().node, rewritten.error().detail));
  }
  view.Forward(pattern_input, rewritten->graph, rewritten->tensors);
  for (auto& scope : scopes) scope.result = rewritten->tensors.at(scope.result.v);
  if (owners) {
    for (auto& a : rewritten->applied)
      for (auto& n : a.replaced) n = owners->rewrite.from.at(n.v);
    std::size_t max_from = 0;
    for (const auto& nid : rewritten->from) {
      if (nid.v >= max_from) max_from = nid.v + 1;
    }
    std::vector<std::vector<NodeId>> by_old(max_from);
    for (std::uint32_t i = 0; i < rewritten->from.size(); ++i) {
      if (rewritten->from[i].v < max_from) {
        by_old[rewritten->from[i].v].push_back(NodeId{i});
      }
    }
    for (auto& a : owners->rewrite.applied) {
      std::vector<NodeId> made;
      for (const NodeId old_id : a.made) {
        if (old_id.v < by_old.size()) {
          made.insert(made.end(), by_old[old_id.v].begin(), by_old[old_id.v].end());
        }
      }
      std::ranges::sort(made, {}, &NodeId::v);
      a.made = std::move(made);
    }
    for (auto& n : rewritten->from) n = owners->rewrite.from.at(n.v);
    std::vector<TensorId> composed;
    for (auto t : owners->rewrite.tensors)
      composed.push_back(t.valid() ? rewritten->tensors.at(t.v) : t);
    rewritten->tensors = std::move(composed);
    rewritten->applied.insert(rewritten->applied.begin(),
                              std::make_move_iterator(owners->rewrite.applied.begin()),
                              std::make_move_iterator(owners->rewrite.applied.end()));
  }
  ir::Graph graph = std::move(rewritten->graph);
  std::vector<TensorId> tensors = std::move(rewritten->tensors);
  std::vector<NodeId> from = std::move(rewritten->from);
  std::vector<Applied> applied = std::move(rewritten->applied);
  // `next` for `graph`, its tensors (`map`) and nodes (`back`) carried
  // through every map onto the primitive graph.
  const auto adopt = [&](ir::Graph next, std::span<const TensorId> map,
                         std::span<const NodeId> back) {
    view.Forward(graph, next, map);
    for (auto& scope : scopes) scope.result = map[scope.result.v];
    for (TensorId& t : tensors) {
      t = t.valid() ? map[t.v] : t;
    }
    std::vector<NodeId> composed(back.size());
    for (std::size_t j = 0; j < back.size(); ++j) {
      composed[j] = from.at(back[j].v);
    }
    std::size_t max_back = 0;
    for (const auto& nid : back) {
      if (nid.v >= max_back) max_back = nid.v + 1;
    }
    std::vector<std::vector<NodeId>> by_old(max_back);
    for (std::uint32_t j = 0; j < back.size(); ++j) {
      if (back[j].v < max_back) by_old[back[j].v].push_back(NodeId{j});
    }
    for (Applied& a : applied) {
      std::vector<NodeId> made;
      for (const NodeId old_id : a.made) {
        if (old_id.v < by_old.size()) {
          made.insert(made.end(), by_old[old_id.v].begin(), by_old[old_id.v].end());
        }
      }
      std::ranges::sort(made, {}, &NodeId::v);
      a.made = std::move(made);
    }
    from = std::move(composed);
    graph = std::move(next);
  };
  // A node computed again (the same rows joined twice) runs once.
  if (const auto repeats = Repeats(graph); !repeats.empty()) {
    std::vector<TensorId> map;
    std::vector<NodeId> back;
    auto joined = ExpandNodes(graph, std::vector<bool>(graph.nodes().size(), false), map, back,
                              nullptr, &repeats);
    if (!joined) {
      return Failed(PlanError::kPattern, std::format("joining repeated nodes at node {}: {}",
                                                     joined.error().node, joined.error().detail));
    }
    adopt(std::move(*joined), map, back);
  }
  // What the rewrites left unread is not run (a split pattern's fused node
  // reading the vectors another node joined for the replaced readers).
  if (const auto unread = Unread(graph); !unread.empty()) {
    std::vector<TensorId> map;
    std::vector<NodeId> back;
    auto pruned =
        ExpandNodes(graph, std::vector<bool>(graph.nodes().size(), false), map, back, &unread);
    if (!pruned) {
      return Failed(PlanError::kPattern, std::format("pruning unread nodes at node {}: {}",
                                                     pruned.error().node, pruned.error().detail));
    }
    adopt(std::move(*pruned), map, back);
  }
  std::vector<std::string> decomposed;
  Selection selection;
  for (int pass = 0;; ++pass) {
    selection = Select(graph, view, scopes);
    if (selection.complete) {
      break;
    }
    // The primitive-fallback rule: a node no implementation computes is
    // replaced by its decomposition, where it has one.
    for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
      if (!selection.missing[i]) {
        continue;
      }
      const ir::Node& n = graph.nodes()[i];
      if (!ir::HasDecomposition(n.key)) {
        return Failed(PlanError::kNoImplementation,
                      std::format("no implementation computes {} on this device",
                                  Describe(graph, NodeId{i})));
      }
      decomposed.push_back(ir::OpKeyName(n.key));
    }
    if (pass + 1 == kMaxPasses) {
      return Failed(PlanError::kBounds, "decompositions did not reach implemented operations");
    }
    std::vector<TensorId> map;
    std::vector<NodeId> back;
    auto expanded = ExpandNodes(graph, selection.missing, map, back);
    if (!expanded) {
      return Failed(PlanError::kPattern,
                    std::format("a decomposition at node {}: {}", expanded.error().node,
                                expanded.error().detail));
    }
    adopt(std::move(*expanded), map, back);
  }
  for (const auto& scope : scopes) {
    if (!scope.result.valid() ||
        graph.node(graph.producer(scope.result)).key != ir::Primitive(ir::Op::kMatMul) ||
        graph.tensor(scope.result).ne(1) != scope.columns)
      return Failed(PlanError::kPattern, "an owner product lost its primitive scope");
  }
  std::vector<Selected> steps = std::move(selection.steps);
  // Lane tags follow the primitive nodes into what stands for them.
  LaneTags tags;
  if (!options.lanes.empty()) {
    std::uint32_t max_node = 0;
    for (const auto& [node, _] : options.lanes) {
      if (node.valid()) {
        max_node = std::max(max_node, node.v);
      }
    }
    std::vector<LaneTag> lane_lut(max_node + 1);
    std::vector<bool> has_lane(max_node + 1, false);
    for (const auto& [node, tag] : options.lanes) {
      if (node.valid()) {
        lane_lut[node.v] = tag;
        has_lane[node.v] = true;
      }
    }
    for (std::uint32_t j = 0; j < from.size(); ++j) {
      const NodeId fn = from[j];
      if (fn.valid() && fn.v < lane_lut.size() && has_lane[fn.v]) {
        tags.emplace_back(NodeId{j}, lane_lut[fn.v]);
      }
    }
  }
  std::vector<Region> regions = AssignLanes(steps, tags, registry, options.on_stream);
  std::vector<TensorId> staged;
  if (options.staged) {
    for (const TensorId t : *options.staged) {
      if (!t.valid() || t.v >= tensors.size() || !tensors[t.v].valid()) {
        return Failed(PlanError::kPlacement, "a staged input the plan does not hold");
      }
      staged.push_back(tensors[t.v]);
    }
  } else {
    staged.assign(graph.inputs().begin(), graph.inputs().end());
  }
  // The prologue (PlanOptions::invariant), without lanes.
  Prologue prologue;
  std::vector<TensorId> invariant;
  if (!options.invariant.empty() && regions.empty()) {
    std::vector<TensorId> planned;
    for (const TensorId t : options.invariant) {
      if (!t.valid() || t.v >= tensors.size() || !tensors[t.v].valid() ||
          !std::ranges::contains(graph.inputs(), tensors[t.v]))
        return Failed(PlanError::kPlacement, "an invariant input the plan does not hold");
      planned.push_back(tensors[t.v]);
    }
    prologue = PrologueOf(graph, steps, planned, options.within);
    // Only what the prologue holds fixed is kept between runs (an input a
    // store or communication reads is copied every run).
    for (std::size_t k = 0; k < planned.size(); ++k)
      if (prologue.kept[k]) invariant.push_back(options.invariant[k]);
  }
  auto placement = PlaceActivations(graph, steps, regions, staged, options.alignment,
                                    options.written, options.within, prologue.preserved);
  if (!placement) {
    return Failed(PlanError::kPlacement, placement.error());
  }
  auto alignments = view.Alignments(graph);
  if (!alignments.empty()) {
    std::uint32_t max_tensor = 0;
    for (const auto& [tid, _] : placement->offsets) {
      if (tid.valid()) {
        max_tensor = std::max(max_tensor, tid.v);
      }
    }
    std::vector<std::uint64_t> offset_lut(max_tensor + 1, 0);
    std::vector<bool> has_offset(max_tensor + 1, false);
    for (const auto& [tid, off] : placement->offsets) {
      if (tid.valid()) {
        offset_lut[tid.v] = off;
        has_offset[tid.v] = true;
      }
    }
    for (const auto& fact : alignments) {
      if (fact.root.valid() && fact.root.v < has_offset.size() && has_offset[fact.root.v]) {
        const std::uint64_t off = offset_lut[fact.root.v];
        if (off % fact.alignment != 0 || (options.activation_alignment != 0 &&
                                          options.activation_alignment % fact.alignment != 0)) {
          return Failed(PlanError::kPlacement,
                        "an activation's placement breaks its alignment promise");
        }
      }
    }
  }
  return StepPlan{
      .graph = std::move(graph),
      .steps = std::move(steps),
      .regions = std::move(regions),
      .placement = std::move(*placement),
      .applied = std::move(applied),
      .from = std::move(from),
      .tensors = std::move(tensors),
      .decomposed = std::move(decomposed),
      .owner_row_invariant =
          !Disabled(policy, "owner_row_invariant") && options.shape != nullptr &&
          options.shape->segments.size() > 1 &&
          std::ranges::all_of(options.shape->segments, [](const auto& s) { return s.rows == 1; }),
      .owner_rows_policy_enabled = !Disabled(policy, "owner_row_invariant"),
      .root_alignments = std::move(alignments),
      .owner_products = std::move(scopes),
      .owner_digest =
          options.owner_products ? options.owner_products->digest() : base::Sha256Digest{},
      .invariant = std::move(invariant),
      .prologue = std::move(prologue.steps)};
}

PlanKey KeyOf(const base::Sha256Digest& spec, const StepShape& shape, const Policy& policy,
              std::uint32_t partition) {
  return PlanKey{.spec = spec,
                 .shape = shape,
                 .patterns = PatternsDigest(policy),
                 .policy = Digest(policy),
                 .partition = partition};
}

}  // namespace llmp::planner
