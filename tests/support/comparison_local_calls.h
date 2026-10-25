// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "kernels/common/lower.h"
#include "planner/alternative.h"

namespace llmp::tests {

// Test-only snapshot of one actual selected local call and its complete
// original-source membership. This intentionally carries no process identity
// and cannot authorize production selection.
struct ComparisonLocalCall {
  base::Sha256Digest implementation_identity{};
  std::string operation;
  std::vector<ir::NodeId> original_members;
};

inline std::vector<ComparisonLocalCall> CaptureComparisonLocalCalls(
    const kernels::LowerInput& input, const execution::Registry& registry) {
  std::vector<ComparisonLocalCall> calls;
  if (input.comparison_region == nullptr || input.comparison_region->origins == nullptr ||
      input.comparison_region->final_nodes == nullptr)
    return calls;
  const auto node_origins = input.comparison_region->final_nodes->node_origins.values;
  if (input.comparison_region->final_nodes->source_graph_identity != input.io.graph.identity() ||
      input.comparison_region->final_nodes->planned_graph_identity != input.plan.graph.identity() ||
      node_origins.size() != input.plan.graph.nodes().size())
    return calls;
  for (const auto& selected : input.plan.steps) {
    if (selected.implementation >= registry.size()) return {};
    std::vector<ir::NodeId> members;
    for (const auto node : selected.nodes) {
      if (!node.valid() || node.v >= node_origins.size()) return {};
      const auto count = input.comparison_region->origins->OriginCount(node_origins[node.v]);
      if (!count || *count == 0 || *count > input.io.graph.nodes().size()) return {};
      const auto first = members.size();
      members.resize(first + *count);
      std::size_t written = 0;
      if (!input.comparison_region->origins->CopyOrigins(
              node_origins[node.v], std::span(members).subspan(first), written) ||
          written != *count)
        return {};
    }
    if (members.empty()) return {};
    if (std::ranges::any_of(members, [&](ir::NodeId origin) {
          return !origin.valid() || origin.v >= input.io.graph.nodes().size();
        }))
      return {};
    std::ranges::sort(members, [](ir::NodeId a, ir::NodeId b) { return a.v < b.v; });
    members.erase(std::unique(members.begin(), members.end()), members.end());
    const auto& implementation = registry.at(selected.implementation);
    calls.push_back({.implementation_identity = execution::IdentityOf(implementation),
                     .operation = std::string(ir::OpKeyName(implementation.operation)),
                     .original_members = std::move(members)});
  }
  return calls;
}

// Serialize the complete calls wholly inside this exact boundary. A partial
// call intersection is unsupported; a seed representative is never used to
// infer a call's original membership.
inline std::optional<base::Sha256Digest> HashContainedComparisonLocalCalls(
    std::span<const ComparisonLocalCall> calls,
    const components::OwnedComparisonRegionBoundary& boundary) {
  std::size_t selected_count = 0;
  for (const auto& call : calls) {
    if (call.original_members.empty()) return std::nullopt;
    std::size_t contained = 0;
    for (const auto origin : call.original_members) {
      if (!origin.valid()) return std::nullopt;
      contained += std::ranges::any_of(
          boundary.nodes(), [&](const auto& member) { return member.original_node == origin; });
    }
    if (contained == 0) continue;
    if (contained != call.original_members.size()) return std::nullopt;
    ++selected_count;
  }
  if (selected_count == 0) return std::nullopt;

  base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(&selected_count, 1)));
  for (const auto& call : calls) {
    if (!std::ranges::all_of(call.original_members, [&](ir::NodeId origin) {
          return std::ranges::any_of(
              boundary.nodes(), [&](const auto& member) { return member.original_node == origin; });
        }))
      continue;
    std::vector<std::uint32_t> canonical_members;
    canonical_members.reserve(call.original_members.size());
    for (const auto origin : call.original_members) {
      const auto found = std::ranges::find(boundary.nodes(), origin,
                                           &components::ComparisonRegionNode::original_node);
      if (found == boundary.nodes().end()) return std::nullopt;
      canonical_members.push_back(found->canonical);
    }
    std::ranges::sort(canonical_members);
    canonical_members.erase(std::unique(canonical_members.begin(), canonical_members.end()),
                            canonical_members.end());
    hash.Update(call.operation);
    hash.Update(std::as_bytes(std::span(&call.implementation_identity, 1)));
    const auto member_count = canonical_members.size();
    hash.Update(std::as_bytes(std::span(canonical_members)));
    hash.Update(std::as_bytes(std::span(&member_count, 1)));
  }
  return hash.Finish();
}

}  // namespace llmp::tests
