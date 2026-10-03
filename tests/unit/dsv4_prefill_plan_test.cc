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
  state.window = md::Dsv4Window::kRing;
  Dsv4Model m;
  m.state = &state;
  m.prefill_outa_hca = true;
  // Any prefill chunk of 64 to 4,096 rows takes HCA (a prompt's last,
  // partial chunk included); a first position is refused for any other
  // chunk, without the option, past the compressed width it qualifies at,
  // or past the context.
  const kg::Dsv4ChunkShape hca{.rows = 2947, .hca_n_kv = 256};
  EXPECT_TRUE(Dsv4PrefillHca(m, hca));
  EXPECT_TRUE(Dsv4PrefillHca(m, {.rows = 64, .hca_n_kv = 256}));
  EXPECT_TRUE(Dsv4PrefillHca(m, {.rows = 4096, .hca_n_kv = 256}));
  EXPECT_FALSE(Dsv4PrefillHca(m, {.rows = 63, .hca_n_kv = 256}));
  EXPECT_FALSE(Dsv4PrefillHca(m, {.rows = 4097, .hca_n_kv = 256}));
  EXPECT_FALSE(Dsv4PrefillHca(m, {.rows = 2947, .hca_n_kv = 512}));
  Dsv4Model off = m;
  off.prefill_outa_hca = false;
  EXPECT_FALSE(Dsv4PrefillHca(off, hca));
  Dsv4Model exact = m;
  exact.exact = true;
  EXPECT_FALSE(Dsv4PrefillHca(exact, hca));
  for (const auto& [rows, first] : {std::pair{32, 0U}, std::pair{4097, 0U}, std::pair{4096, 4097U},
                                    std::pair{2947, 5246U}, std::pair{4096, UINT32_MAX}}) {
    auto planned = PlanDsv4Chunk(m, {.rows = rows, .hca_n_kv = 256}, {}, {}, 0, 0, {}, first);
    ASSERT_FALSE(planned.has_value());
    EXPECT_EQ(planned.error(),
              "the prefill plan's first position is not an HCA prefill chunk's in its context");
  }
  auto unqualified = PlanDsv4Chunk(off, hca, {}, {}, 0, 0, {}, 0U);
  ASSERT_FALSE(unqualified.has_value());
  EXPECT_EQ(unqualified.error(),
            "the prefill plan's first position is not an HCA prefill chunk's in its context");
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
  // A shorter chunk (a prompt's last) at the cached position authenticates.
  input.positions.resize(2048);
  tokens.resize(2048);
  auto partial = BuildDsv4Inputs(m, graph, input, tokens, {}, host);
  ASSERT_FALSE(partial.has_value());
  EXPECT_EQ(partial.error(), type.error().detail);
  input.positions[5] += 1;
  refused();
}

}  // namespace
}  // namespace jitllm::engine
