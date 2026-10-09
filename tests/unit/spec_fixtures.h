// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Gemma 2 and Gemma 3 tensor tables (tests/unit/data) as mapping
// sources (spec/mapping.h), as SourceOfV0 reads their artifacts.
#ifndef LLMP_TESTS_UNIT_SPEC_FIXTURES_H_
#define LLMP_TESTS_UNIT_SPEC_FIXTURES_H_

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <utility>
#include <vector>

#include "artifact/format_map.h"
#include "artifact/json.h"
#include "artifact/representation.h"
#include "gemma2_fixture.h"
#include "gemma3_fixture.h"
#include "spec/mapping.h"

namespace llmp::test_support {

// A fixture's tensor table as a source, as SourceOfV0 reads an artifact.
template <typename Resource>
inline spec::Source SourceOf(std::string architecture, const std::vector<Resource>& resources) {
  spec::Source s{.architecture = std::move(architecture), .container = "gguf"};
  for (const Resource& r : resources) {
    std::string text = std::format(
        R"({{"family":"ggml","ne":[{}],"type":"{}"}})",
        [&] {
          std::string ne;
          for (const auto d : r.ne) {
            ne += (ne.empty() ? "" : ",") + std::to_string(d);
          }
          return ne;
        }(),
        r.type);
    const auto doc = artifact::json::Parse(text);
    EXPECT_TRUE(doc.has_value()) << text;
    const auto rep = artifact::ParseRepresentation(doc->root());
    EXPECT_TRUE(rep.has_value()) << text;
    const auto f = artifact::FormatOf(*rep);
    EXPECT_TRUE(f.has_value()) << text;
    for (const std::string& role : r.roles) {
      s.tensors.push_back({.name = role, .format = *f});
    }
  }
  return s;
}

inline spec::Source Gemma2Source() { return SourceOf("gemma2", gemma2::Resources()); }
inline spec::Source Gemma3Source() { return SourceOf("gemma3", gemma3::Resources()); }

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_UNIT_SPEC_FIXTURES_H_
