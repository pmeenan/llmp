// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Serving's state layout (components::StageLayout, model::ExtentLayout's
// per-tensor rule) of a real spec's stage, checked against the rule and
// measured beside the two uniform layouts it replaces: every tensor packed
// at 256 bytes, and every tensor on its own extent. For the models tests
// (components_*_models_test, spec_models_test), which record the numbers.

#ifndef LLMP_TESTS_SUPPORT_STATE_LAYOUTS_H_
#define LLMP_TESTS_SUPPORT_STATE_LAYOUTS_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <set>
#include <string>
#include <string_view>

#include "components/step.h"
#include "model/state_contract.h"
#include "spec/spec.h"

namespace llmp::test_support {

// A uniform-alignment state layout (`align` for every non-bank tensor), for
// tests comparing against the per-tensor rule.
inline std::expected<model::StateLayoutDerived, std::string> StageLayoutUniform(
    const spec::ModelSpec& spec, std::size_t stage, std::uint32_t context, std::uint32_t max_rows,
    std::uint64_t align) {
  auto contracts = components::StageContracts(spec, stage);
  if (!contracts) return std::unexpected(contracts.error());
  return model::Layout(*contracts, context, max_rows, align);
}

// One layout's footprint, in extents of model::kStateExtent.
struct StateFootprint {
  std::uint64_t region = 0;   // the region's extents (a slot's virtual state)
  std::uint64_t initial = 0;  // extents backing 256 positions (a prompt's first demand)
  std::uint64_t full = 0;     // extents backing the whole context
  std::uint64_t growths = 0;  // times the backed extents grow over `prompt` positions in chunks
};

inline std::uint64_t ExtentsThrough(const model::StateLayoutDerived& layout,
                                    std::uint32_t positions) {
  constexpr std::uint64_t kExtent = model::kStateExtent;
  auto used = model::UsedThrough(layout, std::min(positions, layout.context), 256);
  EXPECT_TRUE(used.has_value());
  if (!used) return 0;
  std::set<std::uint64_t> extents;
  for (const auto& r : *used)
    if (r.bytes != 0)
      for (std::uint64_t e = r.offset / kExtent; e <= (r.offset + r.bytes - 1) / kExtent; ++e)
        extents.insert(e);
  return extents.size();
}

inline StateFootprint FootprintOf(const model::StateLayoutDerived& layout, std::uint32_t prompt) {
  constexpr std::uint64_t kExtent = model::kStateExtent;
  StateFootprint f{.region = (layout.bytes + kExtent - 1) / kExtent,
                   .initial = ExtentsThrough(layout, 256),
                   .full = ExtentsThrough(layout, layout.context)};
  std::uint64_t last = 0;
  const std::uint32_t chunk = std::max<std::uint32_t>(layout.max_rows, 1);
  for (std::uint32_t p = chunk; p <= std::min(prompt, layout.context); p += chunk) {
    const auto now = ExtentsThrough(layout, p);
    if (now != last) ++f.growths;
    last = now;
  }
  return f;
}

struct StateFootprints {
  StateFootprint rule, packed, spread;
  std::size_t big = 0;    // growing tensors of an extent or more
  std::size_t banks = 0;  // bank continuations
};

// Checks the rule on `spec`'s stage at a serving context and chunk, and
// prints (and records) its footprint beside the uniform layouts'.
inline StateFootprints ExpectServingLayout(std::string_view name, const spec::ModelSpec& spec,
                                           std::size_t stage, std::uint32_t context,
                                           std::uint32_t rows, std::uint32_t prompt = 8192) {
  constexpr std::uint64_t kExtent = model::kStateExtent;
  const auto rule = components::StageLayout(spec, stage, context, rows);
  const auto packed = StageLayoutUniform(spec, stage, context, rows, 256);
  const auto spread = StageLayoutUniform(spec, stage, context, rows, kExtent);
  EXPECT_TRUE(rule.has_value() && packed.has_value() && spread.has_value()) << name;
  if (!rule || !packed || !spread) return {};
  EXPECT_EQ(rule->extent, kExtent);
  EXPECT_EQ(rule->items.size(), packed->items.size());
  std::size_t big = 0, banks = 0;
  for (std::size_t i = 0; i < rule->items.size(); ++i) {
    const auto& item = rule->items[i];
    if (item.contract.bank_with_previous) {
      ++banks;
      EXPECT_GT(i, 0U) << name;
      if (i == 0) continue;
      EXPECT_EQ(item.offset, rule->items[i - 1].offset + rule->items[i - 1].bytes)
          << name << " bank " << item.contract.name << " of layer " << item.layer;
    } else if (item.bytes >= kExtent && model::GrowsWithPositions(item.contract)) {
      ++big;
      EXPECT_EQ(item.offset % kExtent, 0U) << name << " " << item.contract.name;
    } else {
      EXPECT_EQ(item.offset % 256, 0U) << name << " " << item.contract.name;
    }
    if (i > 0) EXPECT_GE(item.offset, rule->items[i - 1].offset + rule->items[i - 1].bytes);
  }
  const auto r = FootprintOf(*rule, prompt);
  const auto p = FootprintOf(*packed, prompt);
  const auto s = FootprintOf(*spread, prompt);
  const std::string line = std::format(
      "state {} stage {} context {} chunk {}: {} tensors ({} growing of an extent or more, {} bank "
      "continuations); extents rule/packed/each-own: region {}/{}/{}, 256 positions {}/{}/{}, "
      "full {}/{}/{}, growths over {} positions {}/{}/{}",
      name, stage, context, rows, rule->items.size(), big, banks, r.region, p.region, s.region,
      r.initial, p.initial, s.initial, r.full, p.full, s.full, prompt, r.growths, p.growths,
      s.growths);
  std::cout << line << "\n";
  ::testing::Test::RecordProperty(std::format("state_{}_{}", name, stage), line);
  return {.rule = r, .packed = p, .spread = s, .big = big, .banks = banks};
}

}  // namespace llmp::test_support

namespace llmp::components {
using test_support::StageLayoutUniform;
}  // namespace llmp::components

#endif  // LLMP_TESTS_SUPPORT_STATE_LAYOUTS_H_
