// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/owner_products.h"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"

namespace llmp::planner {
std::expected<OwnerRewritten, std::string> SplitOwnerProducts(
    const ir::Graph& graph, const components::OwnerProducts& proof) {
  if (proof.graph_digest() != components::OwnerGraphIdentity(graph))
    return std::unexpected("owner products: proof belongs to another graph");
  ir::Builder b;
  std::vector<ir::TensorId> tensors(graph.tensor_count());
  std::vector<ir::NodeId> from;
  std::vector<OwnerProductScope> scopes;
  std::vector<Applied> applied;
  RewriteContext copy(graph, b, tensors);
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const ir::NodeId id{i};
    const ir::Node& n = graph.node(id);
    const auto p = std::ranges::find(proof.products(), id, &components::OwnerProduct::node);
    if (p != proof.products().end()) {
      for (const auto& owner : p->owners)
        if (owner.columns > 16)
          return std::unexpected("owner products: owner width exceeds the initial recipe domain");
    }
    if (p == proof.products().end() || p->owners.size() == 1) {
      copy.Copy(n);
      if (p != proof.products().end()) {
        const auto& owner = p->owners.front();
        scopes.push_back({.primitive = id,
                          .result = copy.Map(p->output),
                          .slot = owner.slot,
                          .segment = owner.segment,
                          .columns = owner.columns});
      }
    } else {
      const auto& x = graph.tensor(p->input);
      b.SetComponent(n.component);
      Applied a{.pattern = "owner_products", .replaced = {id}, .made = {}};
      const auto begin = b.node_count();
      ir::TensorId joined;
      for (const auto& owner : p->owners) {
        b.SetOwner(owner.segment);
        auto shape = x.shape();
        shape.ne[1] = owner.columns;
        const auto input = b.View(copy.Map(p->input), shape, x.nb, owner.byte_offset);
        const auto result = b.MatMul(copy.Map(p->weight), input, n.Params<ir::ProductParams>());
        scopes.push_back({.primitive = id,
                          .result = result,
                          .slot = owner.slot,
                          .segment = owner.segment,
                          .columns = owner.columns});
        b.SetJoined();
        joined = joined.valid() ? b.Concat(joined, result, 1) : result;
      }
      copy.Set(p->output, joined);
      for (std::size_t j = begin; j < b.node_count(); ++j)
        a.made.push_back(ir::NodeId{static_cast<std::uint32_t>(j)});
      applied.push_back(std::move(a));
    }
    if (!b.ok())
      return std::unexpected(std::format("owner rewrite at {}: {}", i, b.error()->detail));
    from.resize(b.node_count(), id);
  }
  for (auto t : graph.outputs()) b.Output(copy.Map(t));
  for (auto t : graph.keep()) b.Keep(copy.Map(t));
  auto made = std::move(b).Finish();
  if (!made) return std::unexpected(made.error().detail);
  return OwnerRewritten{.rewrite = {.graph = std::move(*made),
                                    .applied = std::move(applied),
                                    .from = std::move(from),
                                    .tensors = std::move(tensors)},
                        .scopes = std::move(scopes)};
}
}  // namespace llmp::planner
