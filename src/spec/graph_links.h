// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_SPEC_GRAPH_LINKS_H_
#define LLMP_SPEC_GRAPH_LINKS_H_
#include <cstdint>
#include <optional>
namespace llmp::spec {
// Source construction metadata, not an artifact schema extension. Value zero
// is an owner's input; component i produces value i+1. Complete backward edges
// are copied into the compiled graph's source-owned plan parameters.
struct GraphLink {
  std::uint16_t input = 0;
  std::optional<std::uint16_t> auxiliary{};
  bool operator==(const GraphLink&) const = default;
};
}  // namespace llmp::spec
#endif  // LLMP_SPEC_GRAPH_LINKS_H_
