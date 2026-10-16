// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Gemma 2, 3 and 4 tensor tables (tests/unit/data) as mapping sources
// (spec/mapping.h), as the importer describes a GGUF's tensors.
#ifndef LLMP_TESTS_UNIT_SPEC_FIXTURES_H_
#define LLMP_TESTS_UNIT_SPEC_FIXTURES_H_

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "artifact/format_map.h"
#include "artifact/json.h"
#include "artifact/representation.h"
#include "base/json.h"
#include "gemma2_fixture.h"
#include "gemma3_fixture.h"
#include "gemma4_assistant_fixture.h"
#include "gemma4_fixture.h"
#include "spec/mapping.h"

namespace llmp::test_support {

// A fixture's tensor table as a source, as the importer describes a GGUF.
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

// A Gemma 4 fixture's tables (26 or 31) as sources: an expert array as one
// tensor of its slices, the expert count its ne[2] (the importer's form).
inline spec::Source Gemma4SourceOf(std::string architecture,
                                   const std::vector<families::Gemma4Resource>& resources) {
  spec::Source s{.architecture = std::move(architecture), .container = "gguf"};
  for (const families::Gemma4Resource& r : resources) {
    families::Gemma4Resource one = r;
    one.roles = {r.roles.front()};
    const spec::Source single = SourceOf("x", std::vector<families::Gemma4Resource>{one});
    spec::SourceTensor t = single.tensors.front();
    if (r.expert_array) {
      t.format.shape.ne[2] = r.count;
      t.format.shape.rank = 3;
      t.experts = true;
    }
    s.tensors.push_back(std::move(t));
  }
  return s;
}
inline spec::Source Gemma4Source(std::uint32_t size) {
  return Gemma4SourceOf("gemma4", gemma4::Resources(size));
}
inline spec::Source Gemma4AssistantSource(std::uint32_t size) {
  return Gemma4SourceOf("gemma4-assistant", gemma4::AssistantResources(size));
}

// A fixture's GGUF metadata as source values: numbers, flags, strings,
// and lists of integers or flags (as integers) or numbers; other values
// (string arrays, summaries) left out.
inline std::vector<spec::SourceValue> MetadataValues(base::json::Value metadata) {
  std::vector<spec::SourceValue> out;
  for (std::size_t i = 0; i < metadata.size(); ++i) {
    const auto v = metadata.member(i);
    spec::SourceValue sv{.key = std::string(metadata.key(i)), .value = {}};
    if (v.is_bool()) {
      sv.value = v.boolean();
    } else if (v.is_number() && v.is_integer()) {
      sv.value = *v.int64();
    } else if (v.is_number()) {
      sv.value = *v.float64();
    } else if (v.is_string()) {
      sv.value = std::string(v.string());
    } else if (v.is_array() && v.size() > 0 && (v.at(0).is_bool() || v.at(0).is_integer())) {
      std::vector<std::int64_t> list;
      for (std::size_t j = 0; j < v.size(); ++j) {
        list.push_back(v.at(j).is_bool() ? (v.at(j).boolean() ? 1 : 0) : *v.at(j).int64());
      }
      sv.value = std::move(list);
    } else {
      continue;
    }
    out.push_back(std::move(sv));
  }
  return out;
}
inline std::vector<spec::SourceValue> Gemma4FixtureValues(std::uint32_t size) {
  const auto doc = gemma4::Fixture(size);
  return MetadataValues(gemma4::Get(doc.root(), "metadata"));
}
inline std::vector<spec::SourceValue> Gemma4AssistantFixtureValues(std::uint32_t size) {
  const auto* dir = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  base::Check(dir != nullptr, "missing assistant fixtures");
  std::ifstream file(std::string(dir) + "/gemma4-assistant/assistant" + std::to_string(size) +
                     ".json");
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = base::json::Parse(text);
  base::Check(doc.has_value(), "invalid assistant fixtures");
  return MetadataValues(gemma4::Get(doc->root(), "metadata"));
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_UNIT_SPEC_FIXTURES_H_
