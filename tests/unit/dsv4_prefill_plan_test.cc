// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
#include "ggml.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::engine {
namespace {
namespace kg = kernels::ggml;
namespace md = model;

TEST(Dsv4PrefillPlan, StartupMeasurementRejectsRuntimeStorageBeforeModelAccess) {
  Dsv4Model target;
  DsparkModel draft;
  const ActivationMeasurement measurement{256};
  const auto rejected = [](const auto& result) {
    EXPECT_FALSE(result);
    if (!result) EXPECT_NE(result.error().find("measurement-only"), std::string::npos);
  };
  rejected(PlanDsv4Chunk(target, {}, {}, {}, 256, 256, {}, std::nullopt, measurement));
  rejected(PlanDsv4Wave(target, {}, {}, {}, 256, 256, nullptr, {}, measurement));
  rejected(PlanDsparkDraft(draft, 3, {}, 256, 256, measurement));
  rejected(PlanDsparkWave(draft, {}, 3, {}, 256, 256, true, measurement));
  EXPECT_TRUE(Dsv4Options{}.startup_activation_threshold);
}

TEST(Dsv4PrefillPlan, CompletedTokenBatchRefusesAnyInvalidOwnerBeforePublication) {
  const std::array<std::int32_t, 3> valid{0, 8, 16};
  const std::array<std::int32_t, 3> negative{0, -1, 16};
  const std::array<std::int32_t, 3> outside{0, 8, 17};
  EXPECT_TRUE(CheckGreedyTokens(valid, 17));
  EXPECT_FALSE(CheckGreedyTokens(negative, 17));
  EXPECT_FALSE(CheckGreedyTokens(outside, 17));
  EXPECT_FALSE(CheckGreedyTokens(valid, 0));
  EXPECT_FALSE(CheckGreedyTokens({}, 17));
}

TEST(Dsv4PrefillPlan, PlainTokenOutputAuthenticatesFlavorCountTypeAndActivationSpan) {
  auto arena = kg::TensorArena::Create(16);
  ASSERT_TRUE(arena);
  auto* input = ggml_new_tensor_2d(arena->context(), GGML_TYPE_F32, 513, 2);
  auto* token = kg::Argmax(arena->context(), input, false, kg::ArgmaxFlavor::kHostGreedy);
  kg::TensorArena::Bind(input, 0x100000);
  kg::TensorArena::Bind(token, 0x200000);
  auto checked = GreedyOutputBytes(token, 2, 0x200000, 8);
  ASSERT_TRUE(checked);
  EXPECT_EQ(*checked, 8);
  EXPECT_FALSE(GreedyOutputBytes(token, 0, 0x200000, 8));
  EXPECT_FALSE(GreedyOutputBytes(token, 1, 0x200000, 8));
  EXPECT_FALSE(GreedyOutputBytes(token, 2, 0x200000, 7));
  EXPECT_FALSE(GreedyOutputBytes(token, 2, 0x200004, 8));
  kg::TensorArena::Bind(token, 0x200001);
  EXPECT_FALSE(GreedyOutputBytes(token, 2, 0x200000, 16));
  kg::TensorArena::Bind(token, 0x200000);
  token->type = GGML_TYPE_F32;
  EXPECT_FALSE(GreedyOutputBytes(token, 2, 0x200000, 8));
  token->type = GGML_TYPE_I32;
  token->src[1] = input;
  EXPECT_FALSE(GreedyOutputBytes(token, 2, 0x200000, 8));
  token->src[1] = nullptr;
  auto* native = kg::Argmax(arena->context(), input);
  kg::TensorArena::Bind(native, 0x200000);
  EXPECT_TRUE(kg::CheckArgmax(native));
  EXPECT_FALSE(GreedyOutputBytes(native, 2, 0x200000, 8));
  auto* unknown = kg::Argmax(arena->context(), input, false, static_cast<kg::ArgmaxFlavor>(2));
  kg::TensorArena::Bind(unknown, 0x200000);
  EXPECT_FALSE(kg::CheckArgmax(unknown));
  auto* probability = kg::Argmax(arena->context(), input, true, kg::ArgmaxFlavor::kHostGreedy);
  kg::TensorArena::Bind(probability, 0x200000);
  EXPECT_FALSE(kg::CheckArgmax(probability));
}

// Off for the harnesses unless asked; serving turns on both, partial
// chunks included (docs/experiments/ds4-output-prefix, "Tie-aware
// re-scoring and partial chunks").
TEST(Dsv4PrefillPlan, HarnessesDefaultOffServingTakesEveryChunk) {
  EXPECT_FALSE(Dsv4Options{}.prefill_outa_hca);
  EXPECT_FALSE(Dsv4Model{}.prefill_outa_hca);
  EXPECT_FALSE(Dsv4Options{}.prefill_outa_hca_partial);
  EXPECT_FALSE(Dsv4Model{}.prefill_outa_hca_partial);
  EXPECT_TRUE(Dsv4Options{}.device_raw_masks);
  EXPECT_FALSE(Dsv4Model{}.device_raw_masks);  // Low-level plan reference is explicit.
  EXPECT_TRUE(Dsv4Options{}.device_tokens);
  EXPECT_FALSE(kg::Dsv4ChunkShape{}.token);  // Low-level plans retain explicit output policy.
  Dsv4Options served;
  SetDsv4ServedPrefill(served);
  EXPECT_TRUE(served.prefill_outa_hca);
  EXPECT_TRUE(served.prefill_outa_hca_partial);
}

TEST(Dsv4PrefillPlan, TokenPolicyRefusesNonPlainShapesBeforeModelAccess) {
  Dsv4Model model;
  auto refused = PlanDsv4Chunk(model, {.rows = 2, .token = true}, {}, {}, 0, 0);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error(), "plain device token plans require one non-speculative target row");
  const kg::Dsv4ChunkShape shape{.rows = 1, .token = true};
  EXPECT_FALSE(PlanDsv4Chunk(model, shape, {}, {}, 0, 0, {.verify = true}));
  EXPECT_FALSE(PlanDsv4Chunk(model, shape, {}, {}, 0, 0, {}, 0U));
  model.exact = true;
  EXPECT_FALSE(PlanDsv4Chunk(model, shape, {}, {}, 0, 0));
}

TEST(Dsv4PrefillPlan, RefusesInvalidFirstPositionBeforeGraphAllocation) {
  md::Dsv4StateLayout state;
  state.context = 8192;
  state.window = md::Dsv4Window::kRing;
  Dsv4Model m;
  m.state = &state;
  m.prefill_outa_hca = true;
  // A full 4,096-row chunk takes HCA; with the partial option (serving's
  // default) any prefill chunk of 64 to 4,096 rows (a prompt's last,
  // partial chunk included). A first position is refused for any other chunk, without
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
  auto state = md::Dsv4State(md::Dsv4Flash(), 8192, 4096, md::Dsv4Window::kRing);
  ASSERT_TRUE(state);
  m.state = &*state;
  auto arena = kg::TensorArena::Create(256);
  ASSERT_TRUE(arena);
  kg::Dsv4Graph graph;
  graph.prefill_first_position = 4096;
  md::Dsv4ChunkInputs input;
  const auto set_rows = [&](std::uint32_t rows) {
    input = *md::Dsv4Chunk(md::Dsv4Flash(), *state, 4096, rows, false);
    graph.positions = ggml_new_tensor_1d(arena->context(), GGML_TYPE_I32, rows);
    graph.raw_k_idxs = ggml_new_tensor_1d(arena->context(), GGML_TYPE_I64, rows);
    graph.raw_mask = ggml_new_tensor_2d(arena->context(), GGML_TYPE_F16, state->raw_cells, rows);
  };
  set_rows(4096);
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
  set_rows(2048);
  tokens.resize(2048);
  auto partial = BuildDsv4Inputs(m, graph, input, tokens, {}, host);
  ASSERT_FALSE(partial.has_value());
  EXPECT_EQ(partial.error(), type.error().detail);
  input.positions[5] += 1;
  refused();
}

TEST(DsparkMaskPlan, JoinedHostAndDeviceSourcesStageTheSameActualBlockRows) {
  const char* configured =
      std::getenv("JITLLM_TEST_ARTIFACT_CORPUS");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path corpus = configured ? configured : "artifact-corpus";
  const auto directory = corpus / "golden" / "tiny";
  ASSERT_TRUE(std::filesystem::is_directory(directory));
  auto artifact = artifact::Artifact::Open(std::filesystem::directory_iterator(directory)->path());
  ASSERT_TRUE(artifact);
  ASSERT_FALSE(artifact->resources().empty());
  auto target_profile = md::Dsv4Flash();
  target_profile.width = 8;
  target_profile.vocab = 4;
  md::Dsv4Binding binding;
  binding.token_embd = {.index = 0, .type = "F16", .ne = {8, 4}};
  Dsv4Model target;
  target.profile = &target_profile;
  target.binding = &binding;
  target.artifact = &*artifact;
  auto profile = md::DsparkDeepSeekV4Flash();
  profile.mask_token = 3;
  profile.blocks.width = target_profile.width;
  profile.blocks.vocab = target_profile.vocab;
  auto state = md::DsparkState(profile, 3);
  ASSERT_TRUE(state);
  const auto table_offset = artifact->resources()[0].offset.value();
  std::vector<std::byte> table(table_offset + 8 * 4 * 2);
  for (std::size_t i = 0; i < 32; ++i) {
    const auto value = ggml_fp32_to_fp16(static_cast<float>(i) / 8);
    std::memcpy(table.data() + table_offset + 2 * i, &value, sizeof(value));
  }
  std::vector<float> reference;
  for (const bool device : {false, true}) {
    DsparkModel draft;
    draft.profile = &profile;
    draft.state = &*state;
    draft.device_masks = device;
    auto arena = kg::TensorArena::Create(64);
    ASSERT_TRUE(arena);
    auto* c = arena->context();
    kg::DsparkWaveGraph graph;
    graph.first = {0, 3};
    graph.slots.resize(2);
    graph.joined.embd = ggml_new_tensor_2d(c, GGML_TYPE_F32, 8, 6);
    graph.joined.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 6);
    graph.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, 6);
    std::array<md::DsparkBlockInputs, 2> blocks;
    for (std::size_t i = 0; i < 2; ++i) {
      blocks[i] = *md::DsparkBlock(profile, *state, i ? 300 : 254, i ? 2 : 1, 3, !device);
      auto& slot = graph.slots[i];
      slot.device_raw_mask = device;
      slot.raw_k_idxs = ggml_new_tensor_1d(c, GGML_TYPE_I64, 3);
      slot.raw_mask = device ? kg::CausalRingMask(c, graph.joined.positions, 256,
                                                  static_cast<std::int32_t>(graph.first[i]), 3, 256,
                                                  128, INT32_MAX, kg::CausalMaskRows::kExact,
                                                  GGML_TYPE_F16, kg::MaskPolicy::kBlock)
                             : ggml_new_tensor_2d(c, GGML_TYPE_F16, 256, 3);
      if (device) graph.joined.nodes.push_back(slot.raw_mask);
    }
    const std::array<const md::DsparkBlockInputs*, 2> blocks_in{&blocks[0], &blocks[1]};
    DsparkWaveHostInputs out;
    auto result = BuildDsparkWaveInputs(target, draft, graph, blocks_in, table, out);
    ASSERT_TRUE(result) << (result ? "" : result.error());
    EXPECT_EQ(out.tokens, (std::vector<std::int32_t>{1, 3, 3, 2, 3, 3}));
    EXPECT_EQ(out.positions, (std::vector<std::int32_t>{254, 255, 256, 300, 301, 302}));
    std::vector<float> expected_embd;
    for (const auto token : out.tokens)
      for (std::uint32_t feature = 0; feature < target_profile.width; ++feature)
        expected_embd.push_back(
            static_cast<float>(static_cast<std::uint32_t>(token) * target_profile.width + feature) /
            8);
    EXPECT_EQ(out.embd, expected_embd);
    if (!device)
      reference = out.embd;
    else
      EXPECT_EQ(out.embd, reference);
    ASSERT_EQ(out.sources.size(), device ? 5U : 7U);
    EXPECT_EQ(out.sources[0],
              (std::pair<ggml_tensor*, const void*>{graph.joined.embd, out.embd.data()}));
    EXPECT_EQ(out.sources[1],
              (std::pair<ggml_tensor*, const void*>{graph.tokens, out.tokens.data()}));
    EXPECT_EQ(out.sources[2],
              (std::pair<ggml_tensor*, const void*>{graph.joined.positions, out.positions.data()}));
    std::size_t at = 3;
    for (std::size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(out.sources[at++], (std::pair<ggml_tensor*, const void*>{graph.slots[i].raw_k_idxs,
                                                                         blocks[i].cells.data()}));
      if (!device)
        EXPECT_EQ(out.sources[at++], (std::pair<ggml_tensor*, const void*>{graph.slots[i].raw_mask,
                                                                           blocks[i].mask.data()}));
    }
  }
}

// Valid input producers reach the deliberately unsupported embedding type;
// malformed sources must refuse before allocating embeddings or staging.
TEST(DsparkMaskPlan, AuthenticatesScalarAndJoinedBlockSourcesBeforeAllocation) {
  md::Dsv4Binding binding;
  binding.token_embd.type = "unsupported-test-type";
  Dsv4Model target;
  target.profile = &md::Dsv4Flash();
  target.binding = &binding;
  const auto& profile = md::DsparkDeepSeekV4Flash();
  auto state = md::DsparkState(profile, 3);
  ASSERT_TRUE(state);
  const auto type = kg::GgmlTypeOf(binding.token_embd.type);
  ASSERT_FALSE(type);
  for (const bool device : {false, true}) {
    DsparkModel draft;
    draft.profile = &profile;
    draft.state = &*state;
    draft.device_masks = device;
    auto arena = kg::TensorArena::Create(128);
    ASSERT_TRUE(arena);
    auto* c = arena->context();
    kg::DsparkWaveGraph graph;
    graph.first = {0, 3};
    graph.slots.resize(2);
    graph.joined.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 6);
    graph.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, 6);
    std::array<md::DsparkBlockInputs, 2> blocks;
    for (std::size_t i = 0; i < 2; ++i) {
      blocks[i] = *md::DsparkBlock(profile, *state, i ? 300 : 254, 0, 3, !device);
      auto& slot = graph.slots[i];
      slot.device_raw_mask = device;
      slot.raw_k_idxs = ggml_new_tensor_1d(c, GGML_TYPE_I64, 3);
      slot.raw_mask = device ? kg::CausalRingMask(c, graph.joined.positions, state->ring,
                                                  static_cast<std::int32_t>(graph.first[i]), 3,
                                                  static_cast<std::int32_t>(state->ring),
                                                  static_cast<std::int32_t>(profile.blocks.window),
                                                  INT32_MAX, kg::CausalMaskRows::kExact,
                                                  GGML_TYPE_F16, kg::MaskPolicy::kBlock)
                             : ggml_new_tensor_2d(c, GGML_TYPE_F16, state->ring, 3);
      if (device) graph.joined.nodes.push_back(slot.raw_mask);
    }
    const std::array<const md::DsparkBlockInputs*, 2> inputs{&blocks[0], &blocks[1]};
    DsparkWaveHostInputs out;
    const auto checked = [&](bool valid) {
      const auto result = BuildDsparkWaveInputs(target, draft, graph, inputs, {}, out);
      ASSERT_FALSE(result);
      if (valid)
        EXPECT_EQ(result.error(), type.error().detail);
      else
        EXPECT_NE(result.error(), type.error().detail);
      EXPECT_TRUE(out.embd.empty());
      EXPECT_TRUE(out.tokens.empty());
      EXPECT_TRUE(out.positions.empty());
      EXPECT_TRUE(out.sources.empty());
    };
    // Valid source authentication ends before embedding allocation. The
    // joined token/position assembly is allowed only after authentication.
    auto result = BuildDsparkWaveInputs(target, draft, graph, inputs, {}, out);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), type.error().detail);
    EXPECT_EQ(out.tokens.size(), 6U);
    EXPECT_EQ(out.positions.size(), 6U);
    out = {};
    blocks[1].pos0 += 1;
    checked(false);
    blocks[1].pos0 -= 1;
    blocks[1].positions[1] += 1;
    checked(false);
    blocks[1].positions[1] -= 1;
    blocks[1].positions[0] = -1;
    checked(false);
    blocks[1].positions[0] = 300;
    blocks[1].cells[2] += 1;
    checked(false);
    blocks[1].cells[2] -= 1;
    graph.first[1] = 0;
    checked(false);
    graph.first[1] = 3;
    graph.slots[1].raw_k_idxs->nb[0] = 4;
    checked(false);
    graph.slots[1].raw_k_idxs->nb[0] = 8;
    graph.joined.positions->src[0] = graph.tokens;
    checked(false);
    graph.joined.positions->src[0] = nullptr;
    if (device) {
      auto* mask = graph.slots[1].raw_mask;
      for (std::size_t param = 0; param < 8; ++param) {
        std::array<std::byte, sizeof(mask->op_params)> saved{};
        std::memcpy(saved.data(), mask->op_params, saved.size());
        const std::int32_t value = kg::JitllmOpInt(mask, static_cast<int>(param)) ^ 1;
        std::memcpy(reinterpret_cast<std::byte*>(mask->op_params) + 32 + param * 4, &value,
                    sizeof(value));
        checked(false);
        std::memcpy(mask->op_params, saved.data(), saved.size());
      }
      mask->src[0] = graph.tokens;
      checked(false);
      mask->src[0] = graph.joined.positions;
      graph.joined.nodes.push_back(mask);
      checked(false);
      graph.joined.nodes.pop_back();
      graph.slots[1].device_raw_mask = false;
      checked(false);
      graph.slots[1].device_raw_mask = true;
      blocks[1].mask.push_back(0);
      checked(false);
      blocks[1].mask.clear();
    } else {
      blocks[1].mask.pop_back();
      checked(false);
      blocks[1].mask.push_back(0);
      graph.slots[1].raw_mask->src[0] = graph.joined.positions;
      checked(false);
      graph.slots[1].raw_mask->src[0] = nullptr;
    }
    // A scalar owns its own position descriptor, with first_row exactly 0.
    kg::DsparkGraph scalar;
    scalar.core.device_raw_mask = device;
    scalar.core.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 3);
    scalar.core.raw_k_idxs = ggml_new_tensor_1d(c, GGML_TYPE_I64, 3);
    scalar.core.raw_mask =
        device
            ? kg::CausalRingMask(c, scalar.core.positions, state->ring, 0, 3,
                                 static_cast<std::int32_t>(state->ring),
                                 static_cast<std::int32_t>(profile.blocks.window), INT32_MAX,
                                 kg::CausalMaskRows::kExact, GGML_TYPE_F16, kg::MaskPolicy::kBlock)
            : ggml_new_tensor_2d(c, GGML_TYPE_F16, state->ring, 3);
    if (device) scalar.core.nodes.push_back(scalar.core.raw_mask);
    Dsv4HostInputs one;
    auto valid = BuildDsparkInputs(target, draft, scalar, blocks[0], {}, one);
    ASSERT_FALSE(valid);
    EXPECT_EQ(valid.error(), type.error().detail);
    EXPECT_TRUE(one.embd.empty());
    EXPECT_TRUE(one.sources.empty());
    blocks[0].cells[0] += 1;
    auto bad = BuildDsparkInputs(target, draft, scalar, blocks[0], {}, one);
    ASSERT_FALSE(bad);
    EXPECT_NE(bad.error(), type.error().detail);
    EXPECT_TRUE(one.embd.empty());
    EXPECT_TRUE(one.sources.empty());
  }
}

TEST(Dsv4RawMaskPlan, JoinedDeviceAndHostSourcesStageCompletelyWithIdenticalRowData) {
  // The existing tiny artifact supplies a real resource offset for a small
  // F16 embedding fixture; the rest of this test is graph/input-only.
  const char* configured =
      std::getenv("JITLLM_TEST_ARTIFACT_CORPUS");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path corpus = configured ? configured : "artifact-corpus";
  const auto directory = corpus / "golden" / "tiny";
  ASSERT_TRUE(std::filesystem::is_directory(directory));
  auto artifact = artifact::Artifact::Open(std::filesystem::directory_iterator(directory)->path());
  ASSERT_TRUE(artifact);
  ASSERT_FALSE(artifact->resources().empty());
  auto profile = md::Dsv4Flash();
  profile.width = 8;
  profile.vocab = 4;
  md::Dsv4Binding binding;
  binding.token_embd = {.index = 0, .type = "F16", .ne = {8, 4}};
  auto state = md::Dsv4State(md::Dsv4Flash(), 8704, 4096, md::Dsv4Window::kRing);
  ASSERT_TRUE(state);
  std::vector<std::byte> table(artifact->resources()[0].offset.value() + 4 * 8 * 2);
  std::vector<float> reference_embd;
  for (const bool device : {false, true}) {
    auto arena = kg::TensorArena::Create(256);
    ASSERT_TRUE(arena);
    auto* c = arena->context();
    kg::Dsv4WaveGraph graph;
    graph.slots.resize(2);
    graph.first = {0, 3};
    graph.joined.embd = ggml_new_tensor_2d(c, GGML_TYPE_F32, 8, 4);
    graph.joined.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
    graph.joined.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
    for (auto* comp : {&graph.joined.csa, &graph.joined.hca, &graph.joined.lid})
      comp->state_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
    graph.joined.lid_rot = ggml_new_tensor_2d(c, GGML_TYPE_F32, 128, 128);
    std::array<md::Dsv4ChunkInputs, 2> chunks;
    std::array<Dsv4WaveSlotInputs, 2> slots;
    const std::array<std::vector<std::int32_t>, 2> tokens{{{0, 1, 2}, {3}}};
    for (std::size_t slot = 0; slot < 2; ++slot) {
      chunks[slot] =
          *md::Dsv4Chunk(md::Dsv4Flash(), *state, slot ? 4608 : 4351, slot ? 1 : 3, false, !device);
      const auto rows = chunks[slot].rows;
      auto& g = graph.slots[slot];
      g.raw_k_idxs = ggml_new_tensor_1d(c, GGML_TYPE_I64, rows);
      g.device_raw_mask = device;
      g.raw_mask = device ? kg::CausalRingMask(c, graph.joined.positions, state->raw_cells,
                                               static_cast<std::int32_t>(graph.first[slot]),
                                               static_cast<std::int32_t>(rows),
                                               static_cast<std::int32_t>(state->raw_cells), 128,
                                               static_cast<std::int32_t>(state->context),
                                               kg::CausalMaskRows::kExact)
                          : ggml_new_tensor_2d(c, GGML_TYPE_F16, state->raw_cells, rows);
      if (device) graph.joined.nodes.push_back(g.raw_mask);
      const std::array<const md::Dsv4CompPlan*, 3> plans{&chunks[slot].csa, &chunks[slot].hca,
                                                         &chunks[slot].lid};
      const std::array<kg::Dsv4CompInputs*, 3> comps{&g.csa, &g.hca, &g.lid};
      for (std::size_t k = 0; k < 3; ++k) {
        comps[k]->persist_src = ggml_new_tensor_1d(
            c, GGML_TYPE_I32, static_cast<std::int64_t>(plans[k]->persist_src.size()));
        comps[k]->persist_dst = ggml_new_tensor_1d(
            c, GGML_TYPE_I32, static_cast<std::int64_t>(plans[k]->persist_dst.size()));
        comps[k]->read_idxs = ggml_new_tensor_1d(
            c, GGML_TYPE_I32, static_cast<std::int64_t>(plans[k]->read_idxs.size()));
        comps[k]->write_idxs = ggml_new_tensor_1d(
            c, GGML_TYPE_I64, static_cast<std::int64_t>(plans[k]->write_idxs.size()));
        comps[k]->write_pos = ggml_new_tensor_1d(
            c, GGML_TYPE_I32, static_cast<std::int64_t>(plans[k]->write_pos.size()));
      }
      g.csa_visible = ggml_new_tensor_1d(c, GGML_TYPE_I32, rows);
      g.hca_visible = ggml_new_tensor_1d(c, GGML_TYPE_I32, rows);
      slots[slot] = {.chunk = &chunks[slot], .tokens = tokens[slot], .inject_cells = {}};
    }
    Dsv4Model model{.artifact = &*artifact,
                    .profile = &profile,
                    .binding = &binding,
                    .state = &*state,
                    .places = {},
                    .rot = kg::HadamardMatrix(profile.indexer_head_dim),
                    .device_raw_masks = device};
    Dsv4WaveHostInputs host;
    auto staged = BuildDsv4WaveInputs(model, graph, slots, table, host);
    ASSERT_TRUE(staged) << (staged ? "" : staged.error());
    EXPECT_EQ(host.tokens, (std::vector<std::int32_t>{0, 1, 2, 3}));
    EXPECT_EQ(host.positions, (std::vector<std::int32_t>{4351, 4352, 4353, 4608}));
    if (device)
      EXPECT_EQ(host.embd, reference_embd);
    else
      reference_embd = host.embd;
    const auto inputs = graph.inputs();
    ASSERT_EQ(host.sources.size(), inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      EXPECT_EQ(host.sources[i].first, inputs[i]);
      if (ggml_nbytes(inputs[i]) != 0) EXPECT_NE(host.sources[i].second, nullptr) << i;
    }
    for (const auto& slot : graph.slots)
      EXPECT_EQ(std::ranges::count(inputs, slot.raw_mask), device ? 0 : 1);
    graph.first[1] += 1;
    Dsv4WaveHostInputs untouched;
    EXPECT_FALSE(BuildDsv4WaveInputs(model, graph, slots, table, untouched));
    EXPECT_TRUE(untouched.embd.empty());
    EXPECT_TRUE(untouched.tokens.empty());
  }
}

TEST(Dsv4RawMaskPlan, AuthenticatesExactRowsOffsetsAndEveryProducerParentBeforeAllocation) {
  auto state = md::Dsv4State(md::Dsv4Flash(), 8704, 4096, md::Dsv4Window::kRing);
  ASSERT_TRUE(state);
  md::Dsv4Binding binding;
  binding.token_embd.type = "unsupported-test-type";
  Dsv4Model model{.profile = &md::Dsv4Flash(),
                  .binding = &binding,
                  .state = &*state,
                  .places = {},
                  .rot = {},
                  .device_raw_masks = true};
  auto arena = kg::TensorArena::Create(256);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto input = md::Dsv4Chunk(md::Dsv4Flash(), *state, 4351, 3, false, false);
  ASSERT_TRUE(input);
  EXPECT_TRUE(input->raw_mask.empty());
  kg::Dsv4Graph graph;
  graph.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 3);
  graph.raw_k_idxs = ggml_new_tensor_1d(c, GGML_TYPE_I64, 3);
  graph.raw_mask = kg::CausalRingMask(
      c, graph.positions, state->raw_cells, 0, 3, static_cast<std::int32_t>(state->raw_cells), 128,
      static_cast<std::int32_t>(state->context), kg::CausalMaskRows::kExact);
  graph.device_raw_mask = true;
  graph.nodes = {graph.positions, graph.raw_mask};
  const std::vector<std::int32_t> tokens(3, 0);
  const auto call = [&] {
    Dsv4HostInputs host;
    auto result = BuildDsv4Inputs(model, graph, *input, tokens, {}, host);
    EXPECT_TRUE(host.embd.empty());
    EXPECT_TRUE(host.sources.empty());
    return result;
  };
  const auto type = kg::GgmlTypeOf(binding.token_embd.type);
  ASSERT_FALSE(type);
  auto valid = call();
  ASSERT_FALSE(valid);
  EXPECT_EQ(valid.error(), type.error().detail);
  const auto refused = [&] {
    auto result = call();
    ASSERT_FALSE(result);
    EXPECT_NE(result.error(), type.error().detail);
  };
  for (std::size_t parent = 1; parent < GGML_MAX_SRC; ++parent) {
    graph.raw_mask->src[parent] = graph.positions;
    refused();
    graph.raw_mask->src[parent] = nullptr;
  }
  auto* source = graph.raw_mask->src[0];
  // A larger packed scalar source would otherwise pass the segment bounds
  // and stage beyond the actual host position vector.
  graph.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
  graph.raw_mask->src[0] = graph.positions;
  refused();
  graph.positions = source;
  graph.raw_mask->src[0] = graph.raw_k_idxs;
  refused();
  graph.raw_mask->src[0] = source;
  for (const std::size_t param : {0U, 1U, 2U, 3U, 4U, 5U}) {
    auto* bytes = reinterpret_cast<std::byte*>(graph.raw_mask->op_params) + 32 + 4 * param;
    std::int32_t value = 0;
    std::memcpy(&value, bytes, 4);
    const auto bad = value + 1;
    std::memcpy(bytes, &bad, 4);
    refused();
    std::memcpy(bytes, &value, 4);
  }
  graph.nodes.push_back(graph.raw_mask);
  refused();
  graph.nodes.pop_back();
  graph.raw_mask->view_src = graph.positions;
  refused();
  graph.raw_mask->view_src = nullptr;
  graph.device_raw_mask = false;
  refused();
  graph.device_raw_mask = true;
  input->positions[1] += 1;
  refused();
  input->positions[1] -= 1;
  input->raw_cells[2] += 1;
  refused();
  input->raw_cells[2] -= 1;
  input->raw_mask.push_back(0);
  refused();
  input->raw_mask.clear();
  EXPECT_EQ(call().error(), type.error().detail);
}

}  // namespace
}  // namespace jitllm::engine
