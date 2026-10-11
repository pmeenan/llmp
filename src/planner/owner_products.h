// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Expand proved physical owner columns before any width-based preparation.
// The existing graph retains its canonical inputs/state and one placement.
#ifndef LLMP_PLANNER_OWNER_PRODUCTS_H_
#define LLMP_PLANNER_OWNER_PRODUCTS_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "components/owner_products.h"
#include "ir/graph.h"
#include "planner/pattern.h"

namespace llmp::planner {
struct OwnerProductScope {
  ir::NodeId primitive{};
  ir::TensorId result{};  // transformed product's result, never the concat
  std::uint8_t slot = 0;
  std::uint8_t segment = 0;
  std::uint64_t columns = 0;
};
struct OwnerRewritten {
  Rewritten rewrite;
  std::vector<OwnerProductScope> scopes;
};
// The first bounded region is an ordinary dense product with <=16 physical
// columns per owner. Unsupported regions refuse; never split rounded input.
std::expected<OwnerRewritten, std::string> SplitOwnerProducts(
    const ir::Graph& graph, const components::OwnerProducts& proof);
}  // namespace llmp::planner
#endif  // LLMP_PLANNER_OWNER_PRODUCTS_H_
