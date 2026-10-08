// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Actual approved tensor table; generation/source provenance is in the JSON.
#ifndef LLMP_TESTS_SUPPORT_GEMMA3_FIXTURE_H_
#define LLMP_TESTS_SUPPORT_GEMMA3_FIXTURE_H_
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/representation.h"
#include "base/check.h"
#include "base/json.h"
#include "model/gemma3.h"
namespace llmp::test_support::gemma3 {
namespace md = llmp::model;
namespace json = llmp::base::json;
inline json::Value Get(json::Value value, std::string_view key) {
  const auto found = value.find(key);
  llmp::base::Check(found.has_value(), "Gemma3 fixture key is missing");
  return *found;
}
inline json::Document Fixture() {
  const auto* dir = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  llmp::base::Check(dir != nullptr, "Gemma3 test data directory is missing");
  std::ifstream file(std::string(dir) + "/gemma3/gemma3_4b_qat.json");
  llmp::base::Check(file.good(), "Gemma3 fixture is missing");
  const std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = json::Parse(text);
  llmp::base::Check(doc.has_value(), "Gemma3 fixture is invalid JSON");
  return std::move(*doc);
}
inline std::vector<md::Gemma3Resource> Resources() {
  const auto fixture = Fixture();
  const auto tensors = Get(fixture.root(), "tensors");
  std::vector<md::Gemma3Resource> resources;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto tensor = tensors.at(i);
    md::Gemma3Resource resource;
    resource.roles = {std::string(Get(tensor, "name").string())};
    resource.type = Get(tensor, "type").string();
    const auto ne = Get(tensor, "ne");
    for (std::size_t dim = 0; dim < ne.size(); ++dim) {
      resource.ne.push_back(static_cast<std::uint64_t>(*ne.at(dim).int64()));
    }
    std::string representation = "{\"family\":\"ggml\",\"type\":\"" + resource.type + "\",\"ne\":[";
    for (const auto dim : resource.ne) {
      if (representation.back() != '[') {
        representation += ',';
      }
      representation += std::to_string(dim);
    }
    representation += "]}";
    const auto doc = llmp::artifact::json::Parse(representation);
    llmp::base::Check(doc.has_value(), "Gemma3 representation is invalid JSON");
    const auto parsed = llmp::artifact::ParseRepresentation(doc->root());
    llmp::base::Check(parsed.has_value(), "Gemma3 representation is invalid");
    llmp::base::Check(parsed->bytes == static_cast<std::uint64_t>(*Get(tensor, "bytes").int64()),
                      "Gemma3 recorded bytes differ from native type helpers");
    llmp::base::Check(parsed->bytes == parsed->readable,
                      "approved Gemma3 rows unexpectedly require padding");
    const auto* traits = llmp::artifact::FindGgmlType(resource.type);
    llmp::base::Check(traits != nullptr && static_cast<std::int64_t>(traits->id) ==
                                               *Get(tensor, "ggml_type").int64(),
                      "Gemma3 recorded type ID differs from native helpers");
    resource.readable = parsed->readable;
    resources.push_back(std::move(resource));
  }
  return resources;
}
inline md::Gemma3Resource& Role(std::vector<md::Gemma3Resource>& resources, std::string_view role) {
  const auto found = std::ranges::find_if(resources, [role](const auto& resource) {
    return std::ranges::contains(resource.roles, role);
  });
  llmp::base::Check(found != resources.end(), "Gemma3 role is missing");
  return *found;
}

}  // namespace llmp::test_support::gemma3
#endif  // LLMP_TESTS_SUPPORT_GEMMA3_FIXTURE_H_
