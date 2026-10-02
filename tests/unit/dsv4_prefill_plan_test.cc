// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::engine {
namespace {
namespace kg = kernels::ggml;
namespace md = model;

TEST(Dsv4PrefillPlan, InternalOptionDefaultsOff) {
  EXPECT_FALSE(Dsv4Options{}.prefill_outa_hca);
  EXPECT_FALSE(Dsv4Model{}.prefill_outa_hca);
}

TEST(Dsv4PrefillPlan, RefusesInvalidFirstPositionBeforeGraphAllocation) {
  md::Dsv4StateLayout state;
  state.context = 8192;
  Dsv4Model m;
  m.state = &state;
  for (const auto& [rows, first] :
       {std::pair{2048, 0U}, std::pair{4096, 4097U}, std::pair{4096, UINT32_MAX}}) {
    auto planned = PlanDsv4Chunk(m, {.rows = rows}, {}, {}, 0, 0, {}, first);
    ASSERT_FALSE(planned.has_value());
    EXPECT_EQ(planned.error(), "the prefill plan's first position leaves its 4K context");
  }
}

TEST(Dsv4PrefillPlan, AuthenticatesActualPositionsBeforeEmbeddingOrSubmission) {
  md::Dsv4Binding binding;
  // Valid positions reach the existing type refusal; malformed positions
  // must be refused first without dereferencing an artifact or a device.
  binding.token_embd.type = "unsupported-test-type";
  Dsv4Model m;
  m.profile = &md::Dsv4Flash();
  m.binding = &binding;
  kg::Dsv4Graph graph;
  graph.prefill_first_position = 4096;
  md::Dsv4ChunkInputs input;
  input.positions.resize(4096);
  std::ranges::iota(input.positions, 4096);
  std::vector<std::int32_t> tokens(4096, 0);
  Dsv4HostInputs host;
  auto valid = BuildDsv4Inputs(m, graph, input, tokens, {}, host);
  const auto type = kg::GgmlTypeOf(binding.token_embd.type);
  ASSERT_FALSE(valid.has_value());
  ASSERT_FALSE(type.has_value());
  EXPECT_EQ(valid.error(), type.error().detail);
  constexpr std::string_view kError = "the cached HCA first position differs from the actual chunk";
  const auto refused = [&] {
    const auto result = BuildDsv4Inputs(m, graph, input, tokens, {}, host);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), kError);
    EXPECT_TRUE(host.embd.empty());
    EXPECT_TRUE(host.sources.empty());
  };
  graph.prefill_first_position = 0;
  refused();
  graph.prefill_first_position = 4096;
  input.positions[17] += 1;
  refused();
  input.positions[17] -= 1;
  input.positions.front() = -1;
  refused();
  input.positions.front() = 4096;
  input.positions.pop_back();
  refused();
  input.positions.resize(2048);
  tokens.resize(2048);
  refused();
}

}  // namespace
}  // namespace jitllm::engine
