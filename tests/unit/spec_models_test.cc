// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The model spec synthesized from the installed Gemma 2 and Gemma 3 v0
// artifacts (spec/mapping.h SynthesizeV0) with today's profiles: the same
// spec their tensor tables give (spec_test), bound to each artifact's ID.
// Needs the model files (label `models`); skipped without them.

#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
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
  auto artifact = artifact::Artifact::Open(path, {.expected_id = {}, .trusted_owner = ::geteuid()});
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
  EXPECT_EQ(spec->folds, std::vector<std::string>{"norm.plus_one"});
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

}  // namespace
}  // namespace llmp::spec
