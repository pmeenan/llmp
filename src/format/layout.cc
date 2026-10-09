// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "format/layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "base/check.h"

namespace llmp::format {
namespace {

constexpr auto kStep = static_cast<std::uint32_t>(kGgmlRowPadding);

constexpr std::array kLayouts = {
    LayoutInfo{.id = Layout::kDense,
               .name = "dense",
               .over_read_row_step = 0,
               .rows_addressable = true,
               .split_parts = false},
    LayoutInfo{.id = Layout::kGgmlBlocks,
               .name = "ggml_blocks",
               .over_read_row_step = kStep,
               .rows_addressable = true,
               .split_parts = false},
    LayoutInfo{.id = Layout::kGgmlNvfp4Blocks,
               .name = "ggml_nvfp4_blocks",
               .over_read_row_step = kStep,
               .rows_addressable = true,
               .split_parts = false},
    LayoutInfo{.id = Layout::kSplit,
               .name = "split",
               .over_read_row_step = 0,
               .rows_addressable = true,
               .split_parts = true},
    LayoutInfo{.id = Layout::kRowInterleaved,
               .name = "row_interleaved",
               .over_read_row_step = 0,
               .rows_addressable = true,
               .split_parts = false},
    LayoutInfo{.id = Layout::kCutlassScaleAtoms,
               .name = "cutlass_scale_atoms",
               .over_read_row_step = 0,
               .rows_addressable = false,
               .split_parts = true},
    LayoutInfo{.id = Layout::kExl3Tiles,
               .name = "exl3_tiles",
               .over_read_row_step = 0,
               .rows_addressable = false,
               .split_parts = true},
};

constexpr bool InEnumOrder() {
  for (std::size_t i = 0; i < kLayouts.size(); ++i) {
    if (static_cast<std::size_t>(kLayouts[i].id) != i) {
      return false;
    }
  }
  return kLayouts.back().id == Layout::kExl3Tiles;
}
static_assert(InEnumOrder(), "the table lists every layout in enum order");

}  // namespace

std::span<const LayoutInfo> Layouts() { return kLayouts; }

const LayoutInfo& Info(Layout layout) {
  const auto index = static_cast<std::size_t>(layout);
  base::Check(index < kLayouts.size(), "format::Info of a layout outside the enum");
  return kLayouts[index];
}

std::optional<Layout> LayoutByName(std::string_view name) {
  const auto* found =
      std::ranges::find_if(kLayouts, [&](const LayoutInfo& l) { return l.name == name; });
  if (found == kLayouts.end()) {
    return std::nullopt;
  }
  return found->id;
}

}  // namespace llmp::format
