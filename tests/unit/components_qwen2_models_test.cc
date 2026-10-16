// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The three approved fixtures' published v1 artifacts (metadata only): their
// stored specs over their tensors, multipart binding, state contract and
// actual-shape IR. These
// are not executions and do not qualify G5c's bridge/Tier C controls.
#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <span>
#include <string>

#include "artifact/artifact.h"
#include "components/step.h"
#include "spec/mapping.h"
#include "spec/stored.h"
#include "v1_frozen.h"

namespace {
namespace fs = std::filesystem;
namespace ar = llmp::artifact;
namespace co = llmp::components;
namespace sp = llmp::spec;

class ComponentsQwen2Models : public ::testing::TestWithParam<int> {};

TEST_P(ComponentsQwen2Models, ApprovedArtifactBindsEveryPartAndBuildsTheRecordedShapes) {
  // The store's v1-ids.json names.
  constexpr std::array names{"qwen2-fp16", "qwen2-exl3-4.0", "qwen2-exl3-4.5"};
  const char* name = names.at(static_cast<std::size_t>(GetParam()));
  const auto id = llmp::test_support::V1IdNamed(name);
  if (!id) GTEST_FAIL() << "no published v1 " << name << " in " << llmp::test_support::V1Store();
  const fs::path root = fs::path(llmp::test_support::V1Store()) / *id;
  const auto artifact = ar::Artifact::Open(root);
  ASSERT_TRUE(artifact.has_value()) << root;
  const auto source = sp::SourceOf(*artifact);
  ASSERT_TRUE(source.has_value()) << source.error();
  const auto spec = sp::StoredSpec(*artifact);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const std::array<std::span<const co::BoundTensor>, 1> artifacts{source->tensors};
  const auto valid = co::ValidateSpec(*spec, artifacts);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  const auto layout = co::StageLayout(*spec, 0, 2048, 1024);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  const co::Bindings bindings{.tensors = source->tensors, .state = &*layout};
  for (const std::uint32_t rows : {1U, 32U, 144U, 145U, 1023U, 1024U}) {
    SCOPED_TRACE(rows);
    const auto step = co::BuildStep(*spec, 0,
                                    {.segments = {{.slot = 0,
                                                   .rows = rows,
                                                   .past = 0,
                                                   .read_cells = (rows + 255U) / 256U * 256U,
                                                   .outputs = rows}}},
                                    bindings);
    ASSERT_TRUE(step.has_value()) << step.error();
    EXPECT_TRUE(step->logits.valid());
    EXPECT_EQ(step->state.size(), 48U);
  }
}
INSTANTIATE_TEST_SUITE_P(Fixtures, ComponentsQwen2Models, ::testing::Values(0, 1, 2));
}  // namespace
