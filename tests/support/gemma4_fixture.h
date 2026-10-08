// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_SUPPORT_GEMMA4_FIXTURE_H_
#define LLMP_TESTS_SUPPORT_GEMMA4_FIXTURE_H_
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/representation.h"
#include "base/check.h"
#include "base/json.h"
#include "model/gemma4.h"
namespace llmp::test_support::gemma4 {
namespace md = llmp::model;
namespace json = llmp::base::json;
inline json::Value Get(json::Value value, std::string_view key) {
  auto found = value.find(key);
  llmp::base::Check(found.has_value(), "Gemma4 fixture has a missing key");
  return *found;
}
inline json::Document Fixture(std::uint32_t size) {
  const auto* dir = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  llmp::base::Check(dir != nullptr, "Gemma4 test fixture directory is missing");
  std::ifstream file(std::string(dir) + "/gemma4/gemma4_" + std::to_string(size) + "b.json");
  llmp::base::Check(file.good(), "Gemma4 fixture is missing");
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = json::Parse(text);
  llmp::base::Check(doc.has_value(), "Gemma4 fixture is not JSON");
  return std::move(*doc);
}
inline std::uint64_t PreparedReadable(const md::Gemma4Resource& resource) {
  std::string text = "{\"family\":\"ggml\",\"type\":\"" + resource.type + "\",\"ne\":[";
  for (const auto dim : resource.ne) {
    if (text.back() != '[') {
      text += ',';
    }
    text += std::to_string(dim);
  }
  text += "]}";
  const auto doc = llmp::artifact::json::Parse(text);
  llmp::base::Check(doc.has_value(), "Gemma4 test representation is not JSON");
  const auto repr = llmp::artifact::ParseRepresentation(doc->root());
  llmp::base::Check(repr.has_value(), "Gemma4 test representation is invalid");
  return repr->readable;
}
inline std::vector<md::Gemma4Resource> Resources(std::uint32_t size) {
  const auto fixture = Fixture(size);
  const auto tensors = Get(fixture.root(), "tensors");
  std::vector<md::Gemma4Resource> resources;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto t = tensors.at(i);
    md::Gemma4Resource r;
    r.roles = {std::string(Get(t, "name").string())};
    const auto id = *Get(t, "ggml_type").int64();
    const auto types = llmp::artifact::GgmlTypes();
    const auto type = std::ranges::find_if(types, [id](const auto& x) { return x.id == id; });
    llmp::base::Check(type != types.end(), "Gemma4 fixture GGML type is unknown");
    r.type = type->name;
    const auto ne = Get(t, "ne");
    for (std::size_t j = 0; j < ne.size(); ++j) {
      r.ne.push_back(static_cast<std::uint64_t>(*ne.at(j).int64()));
    }
    if (r.ne.size() == 3) {
      r.expert_array = true;
      r.count = static_cast<std::uint32_t>(r.ne.back());
      r.ne.pop_back();
    }
    // Actual GGUF shapes/types become prepared resources; canonical GGML
    // readability also includes any final-row kernel over-read.
    r.readable = PreparedReadable(r);
    resources.push_back(std::move(r));
  }
  return resources;
}
}  // namespace llmp::test_support::gemma4
#endif  // LLMP_TESTS_SUPPORT_GEMMA4_FIXTURE_H_
