// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The three approved fixture artifacts (metadata only) through the shared
// mapping, multipart binding, state contract and actual-shape IR. These
// are not executions and do not qualify G5c's bridge/Tier C controls.
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <system_error>

#include "artifact/artifact.h"
#include "components/step.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen2.h"

namespace {
namespace fs = std::filesystem;
namespace ar = llmp::artifact;
namespace co = llmp::components;
namespace md = llmp::model;
namespace sp = llmp::spec;

class ComponentsQwen2Models : public ::testing::TestWithParam<int> {};

TEST_P(ComponentsQwen2Models, ApprovedArtifactBindsEveryPartAndBuildsTheRecordedShapes) {
  constexpr std::array ids{"b93cdc326ba4f4c1c71da613503ecd848cd2a0caf122214f26e04588a12a9073",
                           "6e96e499ea9b859ea726327d4f82fb1739cb116ce4e9d3541d1ba67506efb4b4",
                           "00d77caf056c45b6fa05ebdbeff8b38da79d0556a2fd4f1b661e7e21ef9ed83a"};
  // Match the existing artifact-layout / EXL3 proof's store override.
  const char* store = std::getenv("LLMP_EXL3_FIXTURE_STORE");  // NOLINT(concurrency-mt-unsafe)
  const char* home = std::getenv("HOME");                      // NOLINT(concurrency-mt-unsafe)
  const fs::path root =
      (store && *store
           ? fs::path(store)
           : fs::path(home ? home : "/") / ".local/share/llmp/artifact-layout-20260922/installed") /
      ids.at(static_cast<std::size_t>(GetParam()));
  std::error_code error;
  if (!fs::is_directory(root, error)) GTEST_SKIP() << "fixture not installed: " << root;
  const auto artifact = ar::Artifact::Open(root);
  ASSERT_TRUE(artifact.has_value()) << root;
  const bool exl3 = GetParam() != 0;
  const auto& profile = exl3 ? md::Qwen25Instruct05BExl3() : md::Qwen25Instruct05B();
  // Independently validate the fixture against its existing strict binding.
  if (exl3) {
    ASSERT_TRUE(md::BindQwen2Exl3(profile, *artifact));
  } else {
    ASSERT_TRUE(md::BindQwen2(profile, *artifact));
  }
  const auto& mapping = sp::mappings::Qwen2Mapping();
  const auto values = sp::mappings::Qwen2V0Values(profile);
  ASSERT_TRUE(values.has_value()) << values.error();
  const auto source = sp::SourceOfV0(*artifact, &mapping);
  ASSERT_TRUE(source.has_value()) << source.error();
  const auto spec = sp::Synthesize(mapping, *source, *values);
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
