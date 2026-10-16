// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The model spec the published Gemma 2, Gemma 3 and Gemma 4 v1 artifacts
// store (spec/stored.h StoredSpec; the Gemma 4 assistant composed with its
// target by ComposeDrafter, from the values serving composes it with): the
// same spec their tensor tables give with the families' values
// (spec_fixtures.h, tests/support/families/gemma.h; spec_test), every
// binding resolved in its artifact, bound to each artifact's ID.
// Needs the model files (label `models`); skipped without them.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "families/gemma.h"
#include "runtime/component_admission.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma4.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "spec_fixtures.h"
#include "v1_frozen.h"

namespace llmp::spec {
namespace {

namespace fam = test_support::families;

// The published v1 artifact the store's v1-ids.json names `name`.
std::optional<artifact::Artifact> OpenPublished(std::string_view name) {
  const auto id = test_support::V1IdNamed(name);
  if (!id) return std::nullopt;
  auto a = artifact::Artifact::Open(std::filesystem::path(test_support::V1Store()) / *id,
                                    {.expected_id = {}});
  EXPECT_TRUE(a.has_value()) << a.error().ToString();
  return a ? std::optional(std::move(*a)) : std::nullopt;
}

struct Case {
  std::string_view architecture;
  std::string_view name;  // the store's v1-ids.json name
  std::size_t tensors;    // the resources' roles
  std::size_t layers;
};

class SpecModelsTest : public testing::TestWithParam<Case> {};

TEST_P(SpecModelsTest, StoredSpecIsTheTensorTablesSpec) {
  const Case& c = GetParam();
  auto artifact = OpenPublished(c.name);
  if (!artifact) GTEST_FAIL() << "no published v1 " << c.name << " in " << test_support::V1Store();
  const Mapping* mapping = FindMapping(artifact->model().architecture);
  ASSERT_NE(mapping, nullptr);
  EXPECT_EQ(mapping->architecture, c.architecture);
  const auto values = c.architecture == "gemma2" ? fam::Gemma2Values(fam::Gemma2_2B())
                                                 : fam::Gemma3Values(fam::Gemma3_4BQat());
  ASSERT_TRUE(values.has_value()) << values.error();

  const auto source = SourceOf(*artifact);
  ASSERT_TRUE(source.has_value()) << source.error();
  EXPECT_EQ(source->container, "gguf");
  EXPECT_EQ(source->tensors.size(), c.tensors);
  const auto spec = StoredSpec(*artifact);
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
    testing::Values(Case{.architecture = "gemma2", .name = "gemma2", .tensors = 288, .layers = 26},
                    Case{.architecture = "gemma3", .name = "gemma3", .tensors = 444, .layers = 34}),
    [](const testing::TestParamInfo<Case>& info) { return std::string(info.param.architecture); });

// The Gemma 4 26B-A4B and 31B targets' stored specs, and the 26B-A4B's
// assistant composed with its target as serving composes it: the specs
// their tensor tables give (spec_test), every binding resolved in its
// stage's artifact.
TEST(SpecModelsTest, Gemma4SpecsFromTheirArtifacts) {
  const auto target26 = OpenPublished("gemma26");
  const auto target31 = OpenPublished("gemma31");
  const auto assistant26 = OpenPublished("gemma26-assistant");
  if (!target26 || !target31 || !assistant26) {
    GTEST_FAIL() << "no published v1 Gemma 4 artifacts in " << test_support::V1Store();
  }
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
    const auto& p = size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B();
    const auto values = fam::Gemma4Values(p);
    ASSERT_TRUE(values.has_value());
    const auto spec = StoredSpec(*a);
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
  // The assistant as serving composes it: the approved pairing's values
  // from the metadata it keeps, by the mapping its approval names.
  const auto kept = runtime::ComponentDrafterValues(*target26, *assistant26);
  ASSERT_TRUE(kept.has_value()) << kept.error();
  const Mapping* drafter = runtime::ComposedDrafterMapping(*assistant26);
  ASSERT_NE(drafter, nullptr);
  const auto source = SourceOf(*assistant26);
  ASSERT_TRUE(source.has_value()) << source.error();
  const auto composed =
      ComposeDrafter(*spec26, mappings::Gemma4Mapping(), *drafter, *source, *kept);
  ASSERT_TRUE(composed.has_value()) << composed.error();
  const auto values = fam::Gemma4AssistantValues(fam::Gemma4Assistant26(), fam::Gemma4_26BA4B());
  ASSERT_TRUE(values.has_value());
  const auto table = ComposeDrafter(*spec26, mappings::Gemma4Mapping(), *drafter,
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
