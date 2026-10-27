// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/decode/selected_source.h"

#include <array>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "kernels/common/composite_source.h"
#include "kernels/decode/selected_grouped.h"
#include "kernels/decode/selected_products.h"
#include "kernels/decode/selected_rows.h"

namespace llmp::kernels::decode {
namespace {

using Child = std::expected<std::shared_ptr<const kernels::SelectedSource>, std::string>;

// The grouped, rows and products children that serve the encoding. A child
// that serves it must construct: its failure (the build lacks its
// declarations) is the source's, never a silently smaller source.
std::expected<std::shared_ptr<const kernels::SelectedSource>, std::string> Decode(
    std::optional<format::Encoding> encoding, Child (*grouped)(std::optional<format::Encoding>)) {
  const std::array<Child (*)(std::optional<format::Encoding>), 3> children{
      grouped, &RowsSelectedSource, &ProductsSelectedSource};
  std::vector<std::shared_ptr<const kernels::SelectedSource>> sources;
  for (const auto child : children) {
    auto made = child(encoding);
    if (!made) return std::unexpected("a decode source failed: " + made.error());
    sources.push_back(std::move(*made));
  }
  if (sources.empty()) {
    return std::unexpected("no decode sources available for requested encoding");
  }
  return CompositeSelectedSource(std::move(sources));
}

}  // namespace

std::expected<std::shared_ptr<const kernels::SelectedSource>, std::string> DecodeSelectedSource(
    std::optional<format::Encoding> encoding) {
  return Decode(encoding, &GroupedSelectedSource);
}

std::expected<std::shared_ptr<const kernels::SelectedSource>, std::string>
DecodeResidentSelectedSource(std::optional<format::Encoding> encoding) {
  return Decode(encoding, &ResidentGroupedSelectedSource);
}

}  // namespace llmp::kernels::decode
