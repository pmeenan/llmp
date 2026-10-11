// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
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

// `graph` with the flagged nodes replaced by their decompositions; `map`
// receives each old tensor's new id and `from` each new node's old one.
std::expected<ir::Graph, ir::VerifyError> ExpandNodes(const ir::Graph& graph,
                                                      const std::vector<bool>& expand,
                                                      std::vector<TensorId>& map,
                                                      std::vector<NodeId>& from) {
  ir::Builder b;
  map.assign(graph.tensor_count(), TensorId{});
  from.clear();
  RewriteContext rc(graph, b, map);
  std::vector<TensorId> sources;
  std::vector<format::Encoding> encodings;
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const ir::Node& node = graph.nodes()[i];
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

Selection Select(const ir::Graph& graph, const SelectionView& view) {
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
      if (!m) {
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
  const SelectionView view(registry, device, policy, options.row_invariant, options.capturable,
                           options.state_spans, options.root_alignment);
  auto rewritten = ApplyPatterns(primitive, view, options.shape, options.state_spans);
  if (!rewritten) {
    return Failed(
        PlanError::kPattern,
        std::format("a rewrite at node {}: {}", rewritten.error().node, rewritten.error().detail));
  }
  ir::Graph graph = std::move(rewritten->graph);
  std::vector<TensorId> tensors = std::move(rewritten->tensors);
  std::vector<NodeId> from = std::move(rewritten->from);
  std::vector<Applied> applied = std::move(rewritten->applied);
  std::vector<std::string> decomposed;
  Selection selection;
  for (int pass = 0;; ++pass) {
    selection = Select(graph, view);
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
    for (TensorId& t : tensors) {
      t = t.valid() ? map.at(t.v) : t;
    }
    std::vector<NodeId> composed(back.size());
    for (std::size_t j = 0; j < back.size(); ++j) {
      composed[j] = from.at(back[j].v);
    }
    for (Applied& a : applied) {
      std::vector<NodeId> made;
      for (std::uint32_t j = 0; j < back.size(); ++j) {
        if (std::ranges::contains(a.made, back[j])) {
          made.push_back(NodeId{j});
        }
      }
      a.made = std::move(made);
    }
    from = std::move(composed);
    graph = std::move(*expanded);
  }
  std::vector<Selected> steps = std::move(selection.steps);
  // Lane tags follow the primitive nodes into what stands for them.
  LaneTags tags;
  if (!options.lanes.empty()) {
    for (std::uint32_t j = 0; j < from.size(); ++j) {
      const auto tag =
          std::ranges::find(options.lanes, from[j], &std::pair<NodeId, LaneTag>::first);
      if (tag != options.lanes.end()) {
        tags.emplace_back(NodeId{j}, tag->second);
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
  auto placement = PlaceActivations(graph, steps, regions, staged, options.alignment,
                                    options.written, options.within);
  if (!placement) {
    return Failed(PlanError::kPlacement, placement.error());
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
          std::ranges::all_of(options.shape->segments, [](const auto& s) { return s.rows == 1; })};
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
