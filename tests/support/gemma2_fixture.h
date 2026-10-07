// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Actual approved tensor table; generation/source provenance is in the JSON.
#ifndef JITLLM_TESTS_SUPPORT_GEMMA2_FIXTURE_H_
#define JITLLM_TESTS_SUPPORT_GEMMA2_FIXTURE_H_
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
#include "model/gemma2.h"
namespace jitllm::test_support::gemma2 {
namespace md = jitllm::model;
namespace json = jitllm::base::json;
inline json::Value Get(json::Value value, std::string_view key) {
  const auto found = value.find(key);
  jitllm::base::Check(found.has_value(), "Gemma2 fixture key is missing");
  return *found;
}
inline json::Document Fixture() {
  const auto* dir = std::getenv("JITLLM_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  jitllm::base::Check(dir != nullptr, "Gemma2 test data directory is missing");
  std::ifstream file(std::string(dir) + "/gemma2/gemma2_2b_q8.json");
  jitllm::base::Check(file.good(), "Gemma2 fixture is missing");
  const std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = json::Parse(text);
  jitllm::base::Check(doc.has_value(), "Gemma2 fixture is invalid JSON");
  return std::move(*doc);
}
inline std::vector<md::Gemma2Resource> Resources() {
  const auto fixture = Fixture();
  const auto tensors = Get(fixture.root(), "tensors");
  std::vector<md::Gemma2Resource> resources;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto tensor = tensors.at(i);
    md::Gemma2Resource resource;
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
    const auto doc = jitllm::artifact::json::Parse(representation);
    jitllm::base::Check(doc.has_value(), "Gemma2 representation is invalid JSON");
    const auto parsed = jitllm::artifact::ParseRepresentation(doc->root());
    jitllm::base::Check(parsed.has_value(), "Gemma2 representation is invalid");
    jitllm::base::Check(parsed->bytes == static_cast<std::uint64_t>(*Get(tensor, "bytes").int64()),
                        "Gemma2 recorded bytes differ from native type helpers");
    const auto* traits = jitllm::artifact::FindGgmlType(resource.type);
    jitllm::base::Check(traits != nullptr && static_cast<std::int64_t>(traits->id) ==
                                                 *Get(tensor, "ggml_type").int64(),
                        "Gemma2 recorded type ID differs from native helpers");
    resource.readable = parsed->readable;
    resources.push_back(std::move(resource));
  }
  return resources;
}
inline md::Gemma2Resource& Role(std::vector<md::Gemma2Resource>& resources, std::string_view role) {
  const auto found = std::ranges::find_if(resources, [role](const auto& resource) {
    return std::ranges::contains(resource.roles, role);
  });
  jitllm::base::Check(found != resources.end(), "Gemma2 role is missing");
  return *found;
}

}  // namespace jitllm::test_support::gemma2
#endif  // JITLLM_TESTS_SUPPORT_GEMMA2_FIXTURE_H_
