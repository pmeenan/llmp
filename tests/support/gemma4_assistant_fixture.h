// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_SUPPORT_GEMMA4_ASSISTANT_FIXTURE_H_
#define LLMP_TESTS_SUPPORT_GEMMA4_ASSISTANT_FIXTURE_H_
#include "gemma4_fixture.h"
#include "model/gemma4_assistant.h"
namespace llmp::test_support::gemma4 {
inline std::vector<md::Gemma4Resource> AssistantResources(std::uint32_t size) {
  const auto* dir = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  base::Check(dir != nullptr, "missing assistant fixtures");
  std::ifstream file(std::string(dir) + "/gemma4-assistant/assistant" + std::to_string(size) +
                     ".json");
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = json::Parse(text);
  base::Check(doc.has_value(), "invalid assistant fixtures");
  const auto tensors = Get(doc->root(), "tensors");
  std::vector<md::Gemma4Resource> resources;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto t = tensors.at(i);
    md::Gemma4Resource r;
    r.roles = {std::string(Get(t, "name").string())};
    r.type = Get(t, "type").string();
    const auto ne = Get(t, "ne");
    for (std::size_t j = 0; j < ne.size(); ++j)
      r.ne.push_back(static_cast<std::uint64_t>(*ne.at(j).int64()));
    r.readable = PreparedReadable(r);
    resources.push_back(std::move(r));
  }
  return resources;
}
}  // namespace llmp::test_support::gemma4
#endif  // LLMP_TESTS_SUPPORT_GEMMA4_ASSISTANT_FIXTURE_H_
