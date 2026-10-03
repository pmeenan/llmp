// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::engine {
namespace {
namespace kg = kernels::ggml;
namespace md = model;

TEST(Dsv4PrefillPlan, InternalOptionDefaultsOff) {
  EXPECT_FALSE(Dsv4Options{}.prefill_outa_hca);
  EXPECT_FALSE(Dsv4Model{}.prefill_outa_hca);
  EXPECT_FALSE(Dsv4Options{}.prefill_outa_hca_partial);
  EXPECT_FALSE(Dsv4Model{}.prefill_outa_hca_partial);
}

TEST(Dsv4PrefillPlan, RefusesInvalidFirstPositionBeforeGraphAllocation) {
  md::Dsv4StateLayout state;
  state.context = 8192;
  state.window = md::Dsv4Window::kRing;
  Dsv4Model m;
  m.state = &state;
  m.prefill_outa_hca = true;
  // A full 4,096-row chunk takes HCA; with the internal partial option any
  // prefill chunk of 64 to 4,096 rows (a prompt's last, partial chunk
  // included). A first position is refused for any other chunk, without
  // the option, past the compressed width it qualifies at, or past the
  // context.
  const kg::Dsv4ChunkShape hca{.rows = 2947, .hca_n_kv = 256};
  EXPECT_TRUE(Dsv4PrefillHca(m, {.rows = 4096, .hca_n_kv = 256}));
  EXPECT_FALSE(Dsv4PrefillHca(m, hca));
  EXPECT_FALSE(Dsv4PrefillHca(m, {.rows = 4096, .hca_n_kv = 512}));
  {
    Dsv4Model partial = m;
    partial.prefill_outa_hca_partial = true;
    EXPECT_TRUE(Dsv4PrefillHca(partial, hca));
    EXPECT_TRUE(Dsv4PrefillHca(partial, {.rows = 64, .hca_n_kv = 256}));
    EXPECT_TRUE(Dsv4PrefillHca(partial, {.rows = 4096, .hca_n_kv = 256}));
    EXPECT_FALSE(Dsv4PrefillHca(partial, {.rows = 63, .hca_n_kv = 256}));
    EXPECT_FALSE(Dsv4PrefillHca(partial, {.rows = 4097, .hca_n_kv = 256}));
    EXPECT_FALSE(Dsv4PrefillHca(partial, {.rows = 2947, .hca_n_kv = 512}));
    auto planned = PlanDsv4Chunk(partial, {.rows = 2947, .hca_n_kv = 256}, {}, {}, 0, 0, {}, 5246U);
    ASSERT_FALSE(planned.has_value());  // past the context
  }
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

// An HCA plan is not keyed by position: each run sets the chunk's position
// on its HCA nodes and on the position its inputs authenticate.
TEST(Dsv4PrefillPlan, AnHcaPlanTakesEachRunsFirstPosition) {
  md::Dsv4StateLayout state;
  state.context = 32768;
  Dsv4Model m;
  m.state = &state;
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena.has_value());
  ggml_context* c = arena->context();
  constexpr std::int64_t kRows = 4096;
  auto* q = ggml_new_tensor_3d(c, GGML_TYPE_F32, 512, kRows, 64);
  auto* k = ggml_new_tensor_3d(c, GGML_TYPE_F16, 512, 8192, 1);
  auto* mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, 8192, kRows);
  auto* hca = ggml_flash_attn_ext(c, q, k, k, mask, 0.04f, 0.0f, 0.0f);
  auto* other = ggml_flash_attn_ext(c, q, k, k, mask, 0.04f, 0.0f, 0.0f);
  kg::MarkDsv4HcaTokentile(hca, 0);
  kg::Dsv4Graph g;
  g.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, kRows);
  g.nodes = {hca, other};
  const auto first_of = [](const ggml_tensor* node) {
    std::uint32_t first = 0;
    std::memcpy(&first, &node->op_params[kg::kDsv4HcaFirstParam], sizeof(first));
    return first;
  };
  // Not an HCA plan: refused, nothing changed.
  EXPECT_FALSE(SetDsv4HcaFirstPosition(m, g, 8192, false).has_value());
  EXPECT_EQ(first_of(hca), 0U);
  g.prefill_first_position = 0;
  ASSERT_TRUE(SetDsv4HcaFirstPosition(m, g, 8192, false).has_value());
  EXPECT_EQ(first_of(hca), 8192U);
  EXPECT_EQ(g.prefill_first_position, std::optional<std::uint32_t>(8192));
  EXPECT_NE(other->op_params[kg::kDsv4HcaTagParam], kg::kDsv4HcaTag);  // untouched
  // Past the context: refused, the plan keeps its position.
  EXPECT_FALSE(SetDsv4HcaFirstPosition(m, g, 32768 - kRows + 1, false).has_value());
  EXPECT_EQ(first_of(hca), 8192U);
  // A captured plan: its replay would keep the captured position, so the
  // position cannot change; refused, nothing changed.
  const auto captured = SetDsv4HcaFirstPosition(m, g, 4096, true);
  ASSERT_FALSE(captured.has_value());
  EXPECT_EQ(captured.error(), "an HCA prefill plan was captured as a graph");
  EXPECT_EQ(first_of(hca), 8192U);
  EXPECT_EQ(g.prefill_first_position, std::optional<std::uint32_t>(8192));
  ASSERT_TRUE(SetDsv4HcaFirstPosition(m, g, 32768 - kRows, false).has_value());
  EXPECT_EQ(first_of(hca), 32768U - 4096U);
}

// A verify never takes the output-A/HCA prefill: a first position is
// refused for it, before any graph is allocated, whatever its rows.
TEST(Dsv4PrefillPlan, RefusesHcaForAVerify) {
  md::Dsv4StateLayout state;
  state.context = 8192;
  state.window = md::Dsv4Window::kRing;
  Dsv4Model m;
  m.state = &state;
  m.prefill_outa_hca = true;
  const kg::Dsv4ChunkShape shape{.rows = 4096, .hca_n_kv = 256};
  ASSERT_TRUE(Dsv4PrefillHca(m, shape));
  auto planned = PlanDsv4Chunk(m, shape, {}, {}, 0, 0, {.verify = true}, 0U);
  ASSERT_FALSE(planned.has_value());
  EXPECT_EQ(planned.error(),
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
