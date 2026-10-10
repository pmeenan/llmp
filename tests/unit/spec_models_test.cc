// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The model spec synthesized from the installed Gemma 2, Gemma 3 and
// Gemma 4 v0 artifacts (spec/mapping.h SynthesizeV0, the Gemma 4
// assistant composed with its target by ComposeDrafterV0) with today's
// profiles: the same spec their tensor tables give (spec_test), bound to
// each artifact's ID.
// Needs the model files (label `models`); skipped without them.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/spec.h"
#include "spec_fixtures.h"

namespace llmp::spec {
namespace {

std::filesystem::path Models() {
  // NOLINTNEXTLINE(concurrency-mt-unsafe): tests read, never set, the environment
  if (const char* dir = std::getenv("LLMP_TEST_MODELS"); dir != nullptr) {
    return dir;
  }
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  return home != nullptr ? std::filesystem::path(home) / ".local/share/llmp" : "";
}

struct Case {
  std::string_view architecture;
  std::string_view path;
  std::size_t tensors;  // the resources' roles
  std::size_t layers;
};

class SpecModelsTest : public testing::TestWithParam<Case> {};

TEST_P(SpecModelsTest, SynthesizesTheSpecFromTheArtifact) {
  const Case& c = GetParam();
  const std::filesystem::path path = Models() / c.path;
  if (std::error_code error; !std::filesystem::exists(path, error)) {
    GTEST_SKIP() << "no artifact in " << path;
  }
  auto artifact = artifact::Artifact::Open(path, {.expected_id = {}});
  ASSERT_TRUE(artifact.has_value()) << artifact.error().ToString();
  const Mapping* mapping = FindMapping(artifact->model().architecture);
  ASSERT_NE(mapping, nullptr);
  EXPECT_EQ(mapping->architecture, c.architecture);
  const auto values = c.architecture == "gemma2" ? mappings::Gemma2V0Values(model::Gemma2_2B())
                                                 : mappings::Gemma3V0Values(model::Gemma3_4BQat());
  ASSERT_TRUE(values.has_value()) << values.error();

  const auto source = SourceOfV0(*artifact);
  ASSERT_TRUE(source.has_value()) << source.error();
  EXPECT_EQ(source->container, "gguf");
  EXPECT_EQ(source->tensors.size(), c.tensors);
  const auto spec = SynthesizeV0(*artifact, *mapping, *values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  ASSERT_EQ(spec->stages.size(), 1U);
  EXPECT_EQ(spec->stages[0].layers.size(), c.layers);
  EXPECT_EQ(spec->stages[0].folds, std::vector<std::string>{"norm.plus_one"});
  // Every binding names a tensor the artifact resolves.
  std::size_t bindings = 0;
  const auto resolves = [&](const ComponentRef& component) {
    for (const WeightBinding& w : component.weights) {
      EXPECT_TRUE(artifact->FindResource(w.tensor).has_value()) << w.tensor;
      ++bindings;
    }
  };
  const Stage& s = spec->stages[0];
  resolves(*s.embedding);
  for (const Layer& l : s.layers) {
    for (const ComponentRef& component : l.components) {
      resolves(component);
    }
  }
  resolves(*s.heads.final_norm);
  resolves(*s.heads.output);
  // Every resource once, and the tied head's embedding again.
  EXPECT_EQ(bindings, artifact->resources().size() + 1);

  // The tensor table's spec (spec_fixtures.h) is this one.
  const auto table = Synthesize(
      *mapping,
      c.architecture == "gemma2" ? test_support::Gemma2Source() : test_support::Gemma3Source(),
      *values);
  ASSERT_TRUE(table.has_value()) << table.error();
  EXPECT_EQ(*table, *spec);

  auto back = FromJson(ToJson(*spec));
  ASSERT_TRUE(back.has_value()) << back.error();
  EXPECT_EQ(*back, *spec);
  const std::array<std::string_view, 1> ids = {artifact->id()};
  const auto bound = BoundDigest(*spec, ids);
  ASSERT_TRUE(bound.has_value()) << bound.error();
  EXPECT_NE(*bound, spec->Digest());
  RecordProperty("spec_digest", base::ToHex(spec->Digest()));
  RecordProperty("spec_bytes", static_cast<int>(ToJson(*spec).size()));
}

INSTANTIATE_TEST_SUITE_P(
    Gemma, SpecModelsTest,
    testing::Values(Case{.architecture = "gemma2",
                         .path = "gemma2-import-20261007/artifacts/"
                                 "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870",
                         .tensors = 288,
                         .layers = 26},
                    Case{.architecture = "gemma3",
                         .path = "gemma3-import-20261007/artifacts/"
                                 "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb",
                         .tensors = 444,
                         .layers = 34}),
    [](const testing::TestParamInfo<Case>& info) { return std::string(info.param.architecture); });
// The Gemma 4 26B-A4B and 31B targets from their installed artifacts, and
// the 26B-A4B's assistant composed with its target: the specs their tensor
// tables give (spec_test), every binding resolved in its stage's artifact.
TEST(SpecModelsTest, Gemma4SpecsFromTheirArtifacts) {
  const std::filesystem::path store = Models() / "m3-artifacts";
  const auto open = [&](std::string_view id) -> std::optional<artifact::Artifact> {
    const std::filesystem::path path = store / id;
    if (std::error_code error; !std::filesystem::exists(path, error)) return std::nullopt;
    auto a = artifact::Artifact::Open(path, {.expected_id = {}});
    EXPECT_TRUE(a.has_value()) << a.error().ToString();
    return a ? std::optional(std::move(*a)) : std::nullopt;
  };
  const auto target26 = open("4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3");
  const auto target31 = open("32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08");
  const auto assistant26 = open("1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42");
  if (!target26 || !target31 || !assistant26) GTEST_SKIP() << "no Gemma 4 artifacts in " << store;
  const auto resolves = [](const artifact::Artifact& a, const ComponentRef& c,
                           std::string_view own) {
    for (const WeightBinding& w : c.weights) {
      if (w.stage.empty() != own.empty()) continue;
      const bool found =
          a.FindResource(w.tensor).has_value() ||
          std::ranges::contains(a.expert_arrays(), w.tensor, &artifact::ExpertArray::name);
      EXPECT_TRUE(found) << w.tensor;
    }
  };
  std::optional<ModelSpec> spec26;
  for (const auto& [a, size] : {std::pair{&*target26, 26U}, std::pair{&*target31, 31U}}) {
    SCOPED_TRACE(size);
    const Mapping* mapping = FindMapping(a->model().architecture);
    ASSERT_EQ(mapping, &mappings::Gemma4Mapping());
    const auto& p = size == 26 ? model::Gemma4_26BA4B() : model::Gemma4_31B();
    const auto values = mappings::Gemma4V0Values(p);
    ASSERT_TRUE(values.has_value());
    const auto spec = SynthesizeV0(*a, *mapping, *values);
    ASSERT_TRUE(spec.has_value()) << spec.error();
    EXPECT_TRUE(spec->stages[0].folds.empty());
    const auto table = Synthesize(*mapping, test_support::Gemma4Source(size), *values);
    ASSERT_TRUE(table.has_value()) << table.error();
    EXPECT_EQ(*table, *spec);
    const Stage& s = spec->stages[0];
    resolves(*a, *s.embedding, {});
    for (const Layer& l : s.layers) {
      for (const ComponentRef& c : l.components) resolves(*a, c, {});
    }
    if (size == 26) spec26 = *spec;
  }
  const auto values =
      mappings::Gemma4AssistantV0Values(model::Gemma4Assistant26(), model::Gemma4_26BA4B());
  ASSERT_TRUE(values.has_value());
  const auto composed = ComposeDrafterV0(*spec26, mappings::Gemma4Mapping(), *assistant26,
                                         mappings::Gemma4AssistantMapping(), *values);
  ASSERT_TRUE(composed.has_value()) << composed.error();
  const auto table =
      ComposeDrafter(*spec26, mappings::Gemma4Mapping(), mappings::Gemma4AssistantMapping(),
                     test_support::Gemma4AssistantSource(26), *values);
  ASSERT_TRUE(table.has_value()) << table.error();
  EXPECT_EQ(*table, *composed);
  const Stage& d = composed->stages.at(1);
  resolves(*assistant26, *d.embedding, {});
  resolves(*target26, *d.embedding, "target");
  for (const Layer& l : d.layers) {
    for (const ComponentRef& c : l.components) resolves(*assistant26, c, {});
  }
  resolves(*assistant26, *d.heads.output, {});
  const std::array<std::string_view, 2> ids = {target26->id(), assistant26->id()};
  EXPECT_TRUE(BoundDigest(*composed, ids).has_value());
}

}  // namespace
}  // namespace llmp::spec
