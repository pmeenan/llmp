// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma4.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/representation.h"
#include "base/check.h"
#include "base/json.h"
#include "expected_error.h"

namespace {
namespace md = jitllm::model;
namespace json = jitllm::base::json;
constexpr std::uint64_t kExtent = 2U << 20U;
json::Value Get(json::Value value, std::string_view key) {
  auto found = value.find(key);
  jitllm::base::Check(found.has_value(), "Gemma4 fixture has a missing key");
  return *found;
}
json::Document Fixture(std::uint32_t size) {
  const auto* dir = std::getenv("JITLLM_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  jitllm::base::Check(dir != nullptr, "Gemma4 test fixture directory is missing");
  std::ifstream file(std::string(dir) + "/gemma4/gemma4_" + std::to_string(size) + "b.json");
  jitllm::base::Check(file.good(), "Gemma4 fixture is missing");
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = json::Parse(text);
  jitllm::base::Check(doc.has_value(), "Gemma4 fixture is not JSON");
  return std::move(*doc);
}
std::uint64_t PreparedReadable(const md::Gemma4Resource& resource) {
  std::string text = "{\"family\":\"ggml\",\"type\":\"" + resource.type + "\",\"ne\":[";
  for (const auto dim : resource.ne) {
    if (text.back() != '[') {
      text += ',';
    }
    text += std::to_string(dim);
  }
  text += "]}";
  const auto doc = jitllm::artifact::json::Parse(text);
  jitllm::base::Check(doc.has_value(), "Gemma4 test representation is not JSON");
  const auto repr = jitllm::artifact::ParseRepresentation(doc->root());
  jitllm::base::Check(repr.has_value(), "Gemma4 test representation is invalid");
  return repr->readable;
}
std::vector<md::Gemma4Resource> Resources(std::uint32_t size) {
  const auto fixture = Fixture(size);
  const auto tensors = Get(fixture.root(), "tensors");
  std::vector<md::Gemma4Resource> resources;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto t = tensors.at(i);
    md::Gemma4Resource r;
    r.roles = {std::string(Get(t, "name").string())};
    const auto id = *Get(t, "ggml_type").int64();
    const auto types = jitllm::artifact::GgmlTypes();
    const auto type = std::ranges::find_if(types, [id](const auto& x) { return x.id == id; });
    jitllm::base::Check(type != types.end(), "Gemma4 fixture GGML type is unknown");
    r.type = type->name;
    const auto ne = Get(t, "ne");
    for (std::size_t j = 0; j < ne.size(); ++j) {
      r.ne.push_back(static_cast<std::uint64_t>(*ne.at(j).int64()));
    }
    if (r.ne.size() == 3) {
      r.expert_array = true;
      r.count = static_cast<std::uint32_t>(r.ne.back());
      r.ne.pop_back();
    }
    // Actual GGUF shapes/types become prepared resources; canonical GGML
    // readability also includes any final-row kernel over-read.
    r.readable = PreparedReadable(r);
    resources.push_back(std::move(r));
  }
  return resources;
}
md::Gemma4Resource& Role(std::vector<md::Gemma4Resource>& resources, std::string_view role) {
  const auto found = std::ranges::find_if(
      resources, [role](const auto& r) { return std::ranges::contains(r.roles, role); });
  jitllm::base::Check(found != resources.end(), "Gemma4 test role is missing");
  return *found;
}
void CheckProfile(const md::Gemma4Profile& p, std::uint32_t size) {
  const auto fixture = Fixture(size);
  const auto m = Get(fixture.root(), "metadata");
  const auto number = [&](std::string_view key) { return *Get(m, key).float64(); };
  EXPECT_EQ(number("gemma4.block_count"), p.layers);
  EXPECT_EQ(number("gemma4.embedding_length"), p.width);
  EXPECT_EQ(number("gemma4.attention.head_count"), p.heads);
  EXPECT_EQ(number("gemma4.context_length"), p.context);
  EXPECT_EQ(number("gemma4.feed_forward_length"), p.ffn);
  EXPECT_EQ(number("gemma4.attention.key_length"), p.global_head_dim);
  EXPECT_EQ(number("gemma4.attention.value_length"), p.global_head_dim);
  EXPECT_EQ(number("gemma4.attention.key_length_swa"), p.local_head_dim);
  EXPECT_EQ(number("gemma4.attention.value_length_swa"), p.local_head_dim);
  EXPECT_EQ(number("gemma4.rope.dimension_count"), p.global_rope_dims);
  EXPECT_EQ(number("gemma4.rope.dimension_count_swa"), p.local_rope_dims);
  EXPECT_EQ(number("gemma4.rope.freq_base"), p.global_rope_base);
  EXPECT_EQ(number("gemma4.rope.freq_base_swa"), p.local_rope_base);
  EXPECT_EQ(number("gemma4.attention.sliding_window"), p.window);
  EXPECT_EQ(static_cast<float>(number("gemma4.attention.layer_norm_rms_epsilon")), p.rms_eps);
  EXPECT_EQ(number("gemma4.final_logit_softcapping"), p.final_softcap);
  EXPECT_EQ(number("gemma4.attention.shared_kv_layers"), 0);
  EXPECT_EQ(number("gemma4.embedding_length_per_layer_input"), 0);
  EXPECT_EQ(number("vocab_count"), p.vocab);
  const auto pattern = Get(m, "gemma4.attention.sliding_window_pattern");
  const auto kv = Get(m, "gemma4.attention.head_count_kv");
  ASSERT_EQ(pattern.size(), p.layers);
  ASSERT_EQ(kv.size(), p.layers);
  for (std::uint32_t i = 0; i < p.layers; ++i) {
    EXPECT_EQ(pattern.at(i).boolean(), p.local(i));
    EXPECT_EQ(*kv.at(i).int64(), p.kv_heads(i));
  }
  if (p.experts != 0) {
    EXPECT_EQ(number("gemma4.expert_count"), p.experts);
    EXPECT_EQ(number("gemma4.expert_used_count"), p.experts_used);
    EXPECT_EQ(number("gemma4.expert_feed_forward_length"), p.expert_ffn);
  } else {
    EXPECT_FALSE(m.find("gemma4.expert_count"));
  }
  const auto factors = Get(fixture.root(), "rope_factors");
  std::vector<float> values;
  for (std::size_t i = 0; i < factors.size(); ++i) {
    values.push_back(static_cast<float>(*factors.at(i).float64()));
  }
  EXPECT_TRUE(md::CheckGemma4RopeFactors(p, values));
  ASSERT_EQ(values.size(), 256U);
  EXPECT_EQ(values[63], 1.0f);
  EXPECT_GT(values[64], 1e29f);
  EXPECT_EQ(p.global_rope_dims, 512U);
}
TEST(Gemma4, ProfilesMatchBothApprovedGgufMetadataContracts) {
  CheckProfile(md::Gemma4_26BA4B(), 26);
  CheckProfile(md::Gemma4_31B(), 31);
}
TEST(Gemma4, BothActualTensorContractsBindWithTiedHeadsAndGlobalProjections) {
  for (const auto size : {26U, 31U}) {
    const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
    auto bound = md::BindGemma4(p, "gemma4", Resources(size));
    ASSERT_TRUE(bound) << *jitllm::test_support::Failed(bound);
    EXPECT_EQ(bound->output.index, bound->token_embd.index);
    EXPECT_EQ(bound->rope_freqs.ne, (std::vector<std::uint64_t>{256}));
    EXPECT_EQ(bound->layers[5].v.index, bound->layers[5].k.index);
    EXPECT_TRUE(bound->layers[5].tied_kv);
    EXPECT_FALSE(bound->layers[0].tied_kv);
    EXPECT_NE(bound->layers[0].v.index, bound->layers[0].k.index);
    if (size == 26) {
      EXPECT_EQ(bound->layers[0].gate_up_exps->type, "Q4_K");
      EXPECT_EQ(bound->layers[0].down_exps->type, "Q5_1");
      EXPECT_EQ(bound->layers[29].down_exps->type, "Q8_0");
    } else {
      EXPECT_FALSE(bound->layers[0].router);
    }
  }
}
TEST(Gemma4, BindingRefusesWrongRolesShapesTypesCountsAndUnfundedStorage) {
  const auto& p = md::Gemma4_26BA4B();
  const auto original = Resources(26);
  EXPECT_FALSE(md::BindGemma4(p, "gemma3", original));
  auto resources = original;
  resources.push_back(resources.front());
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  resources = original;
  resources.front().roles = {"unexpected.weight"};
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  resources = original;
  Role(resources, "rope_freqs.weight").ne = {64};
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  resources = original;
  Role(resources, "blk.0.attn_q_norm.weight").type = "F16";
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  resources = original;
  Role(resources, "blk.0.attn_q.weight").type = "I8";
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  resources = original;
  Role(resources, "blk.0.ffn_down_exps.weight").count = 127;
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  resources = original;
  Role(resources, "blk.0.ffn_down_exps.weight").readable = 0;
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  resources = original;
  Role(resources, "token_embd.weight").count = 1;
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  auto partial = p;
  partial.global_rope_dims = 128;
  EXPECT_FALSE(md::BindGemma4(partial, "gemma4", original));
}
TEST(Gemma4, ExplicitHeadAliasWorksButAnIndependentHeadIsRefused) {
  auto resources = Resources(31);
  Role(resources, "token_embd.weight").roles.push_back("output.weight");
  EXPECT_TRUE(md::BindGemma4(md::Gemma4_31B(), "gemma4", resources));
  auto output = Role(resources, "token_embd.weight");
  output.roles = {"output.weight"};
  resources = Resources(31);
  resources.push_back(std::move(output));
  EXPECT_FALSE(md::BindGemma4(md::Gemma4_31B(), "gemma4", resources));
}
TEST(Gemma4, BindingRequiresQuantizedReadablePaddingAndAlignedExpertMembers) {
  const auto& p = md::Gemma4_26BA4B();
  auto resources = Resources(26);
  // Resource contracts represent prepared storage, including GGML's final
  // partial 512-element row over-read, rather than just GGUF payload bytes.
  ASSERT_TRUE(md::BindGemma4(p, "gemma4", resources));
  auto& gate = Role(resources, "blk.0.ffn_gate_up_exps.weight");
  ASSERT_EQ(gate.type, "Q4_K");
  ASSERT_EQ(gate.readable, 2230416U);  // 2230272 payload + 144 tail
  --gate.readable;
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  ++gate.readable;
  auto& query = Role(resources, "blk.0.attn_q.weight");
  ASSERT_EQ(query.type, "Q8_0");
  --query.readable;
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  ++query.readable;
  auto& down = Role(resources, "blk.0.ffn_down_exps.weight");
  ASSERT_EQ(down.type, "Q5_1");
  ASSERT_EQ(down.readable, 1487088U);  // 1486848 payload + 240 tail
  --down.readable;
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  ++down.readable;
  down.group_offset = 1;
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
  down.group_offset = 256;
  EXPECT_TRUE(md::BindGemma4(p, "gemma4", resources));
  down.readable = std::numeric_limits<std::uint64_t>::max();
  EXPECT_FALSE(md::BindGemma4(p, "gemma4", resources));
}
TEST(Gemma4, SplitExpertGateAndUpMustBeCompleteAndExclusive) {
  auto resources = Resources(26);
  for (std::uint32_t il = 0; il < 30; ++il) {
    const auto prefix = "blk." + std::to_string(il) + ".";
    auto& fused = Role(resources, prefix + "ffn_gate_up_exps.weight");
    fused.roles = {prefix + "ffn_gate_exps.weight"};
    fused.ne[1] /= 2;
    fused.readable = PreparedReadable(fused);
    auto up = fused;
    up.roles = {prefix + "ffn_up_exps.weight"};
    resources.push_back(std::move(up));
  }
  EXPECT_TRUE(md::BindGemma4(md::Gemma4_26BA4B(), "gemma4", resources));
  auto extra = Role(resources, "blk.0.ffn_gate_exps.weight");
  extra.roles = {"blk.0.ffn_gate_up_exps.weight"};
  extra.ne[1] *= 2;
  extra.readable *= 2;
  resources.push_back(extra);
  EXPECT_FALSE(md::BindGemma4(md::Gemma4_26BA4B(), "gemma4", resources));
  resources.pop_back();
  std::erase_if(resources, [](const auto& r) { return r.roles[0] == "blk.0.ffn_up_exps.weight"; });
  EXPECT_FALSE(md::BindGemma4(md::Gemma4_26BA4B(), "gemma4", resources));
}
TEST(Gemma4, RopeFactorsPreserveTheFiniteProportionalSuffixAndRefuseMalformedValues) {
  std::vector<float> factors(256, 1e30f);
  const auto& p = md::Gemma4_26BA4B();
  EXPECT_TRUE(md::CheckGemma4RopeFactors(p, factors));
  factors[0] = 0;
  EXPECT_FALSE(md::CheckGemma4RopeFactors(p, factors));
  factors[0] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(md::CheckGemma4RopeFactors(p, factors));
  factors[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(md::CheckGemma4RopeFactors(p, factors));
  factors.resize(64);
  EXPECT_FALSE(md::CheckGemma4RopeFactors(p, factors));
}
TEST(Gemma4, StateIsBoundedAlignedAndKeepsSeparateNormalizedValues) {
  for (const auto* p : {&md::Gemma4_26BA4B(), &md::Gemma4_31B()}) {
    auto state = md::Gemma4State(*p, p->context, 4096);
    ASSERT_TRUE(state);
    EXPECT_EQ(state->local_cells, 5120U);
    EXPECT_EQ(state->global_cells, 262144U);
    ASSERT_EQ(state->tensors.size(), p->layers * 2U);
    for (std::size_t i = 0; i < state->tensors.size(); ++i) {
      const auto& t = state->tensors[i];
      EXPECT_EQ(t.offset % kExtent, 0U);
      EXPECT_LE(t.offset + t.bytes, state->bytes);
      if (i != 0) {
        EXPECT_GE(t.offset, state->tensors[i - 1].offset + state->tensors[i - 1].bytes);
      }
    }
    const auto& k = state->tensors[10];
    const auto& v = state->tensors[11];
    EXPECT_FALSE(k.local);
    EXPECT_FALSE(v.local);
    EXPECT_NE(k.offset, v.offset);
    EXPECT_EQ(k.width, p->global_kv_heads * 512);
    const auto reps = state->Representations(*p);
    ASSERT_TRUE(reps);
    EXPECT_FALSE((*reps)[0].CanTruncate());
    EXPECT_TRUE((*reps)[10].Can(md::StateCapability::kTruncate));
    for (const auto& r : *reps) {
      EXPECT_TRUE(md::IsValid(r));
    }
    EXPECT_TRUE(md::Gemma4UsedState(*p, *state, 0)->empty());
    const auto used = md::Gemma4UsedState(*p, *state, 1);
    ASSERT_TRUE(used);
    ASSERT_EQ(used->size(), state->tensors.size());
    EXPECT_EQ((*used)[10].bytes, std::uint64_t{k.width} * 256 * 2);
    auto at_end = md::Gemma4UsedState(*p, *state, p->context);
    ASSERT_TRUE(at_end);
    EXPECT_EQ((*at_end)[0].bytes, state->tensors[0].bytes);
    EXPECT_EQ((*at_end)[10].bytes, k.bytes);
  }
}
TEST(Gemma4, LayoutAndFootprintRefuseInvalidOrMutatedBounds) {
  const auto& p = md::Gemma4_26BA4B();
  EXPECT_FALSE(md::Gemma4State(p, 0, 1));
  EXPECT_FALSE(md::Gemma4State(p, p.context + 1, 1));
  EXPECT_FALSE(md::Gemma4State(p, p.context, 0));
  EXPECT_FALSE(md::Gemma4State(p, p.context, 8193));
  EXPECT_FALSE(md::Gemma4State(p, 8, 9));
  auto state = *md::Gemma4State(p, 8192, 4);
  EXPECT_FALSE(md::Gemma4UsedState(p, state, 8193));
  EXPECT_FALSE(md::Gemma4UsedState(p, state, 1, 257));
  state.tensors[0].offset += 2;
  EXPECT_FALSE(state.Representations(p));
  EXPECT_FALSE(md::Gemma4UsedState(p, state, 1));
  state = *md::Gemma4State(p, 8192, 4);
  --state.bytes;
  EXPECT_FALSE(md::Gemma4ChunkWrites(p, state, 0, 1));
}
TEST(Gemma4, BatchSegmentsRetainIndependentPositionsCellsMasksAndOutputRows) {
  const auto& p = md::Gemma4_26BA4B();
  const auto state = *md::Gemma4State(p, 8192, 4);
  const std::array<std::int32_t, 2> a{2, 100};
  const std::array<std::int32_t, 1> b{3};
  const std::array<md::Gemma4Segment, 2> segments{
      {{.slot = 7, .n_past = 1279, .tokens = a}, {.slot = 2, .n_past = 0, .tokens = b}}};
  const auto in = md::Gemma4Chunk(p, state, segments, true);
  ASSERT_TRUE(in);
  EXPECT_EQ(in->positions, (std::vector<std::int32_t>{1279, 1280, 0}));
  EXPECT_EQ(in->out_ids, (std::vector<std::int32_t>{0, 1, 2}));
  EXPECT_EQ(in->segments[0].local_cells, (std::vector<std::int64_t>{1279, 0}));
  EXPECT_EQ(in->segments[1].local_cells, (std::vector<std::int64_t>{0}));
  EXPECT_EQ(in->segments[1].first_row, 2U);
  EXPECT_EQ(in->segments[1].slot, 2U);
  EXPECT_EQ(in->segments[1].global_mask[0], 0U);
  EXPECT_EQ(in->segments[1].global_mask[1], 0xFC00U);
  const auto& first = in->segments[0];
  EXPECT_EQ(first.local_mask[0], 0xFC00U);  // physical0 now holds future position1280
  EXPECT_EQ(first.local_mask[1279], 0U);
  EXPECT_EQ(first.local_mask[first.local_n_kv + 256], 0xFC00U);  // exactly1024 positions old
  EXPECT_EQ(first.local_mask[first.local_n_kv + 257], 0U);
  for (std::size_t i = 0; i < segments.size(); ++i) {
    const std::array<md::Gemma4Segment, 1> solo{segments[i]};
    const auto one = md::Gemma4Chunk(p, state, solo, true);
    ASSERT_TRUE(one);
    EXPECT_EQ(one->segments[0].global_mask, in->segments[i].global_mask);
    EXPECT_EQ(one->segments[0].local_mask, in->segments[i].local_mask);
  }
}
TEST(Gemma4, RingWritesSplitAtWrapAndAreInsideInitializedFootprints) {
  const auto& p = md::Gemma4_26BA4B();
  const auto s = *md::Gemma4State(p, 8192, 4);
  const auto writes = md::Gemma4ChunkWrites(p, s, 1278, 4);
  ASSERT_TRUE(writes);
  const auto used = md::Gemma4UsedState(p, s, 1282);
  ASSERT_TRUE(used);
  std::uint64_t actual = 0, expected = 0;
  for (const auto& t : s.tensors) {
    expected += std::uint64_t{t.width} * 4 * 2;
  }
  for (const auto& w : *writes) {
    actual += w.bytes;
    EXPECT_TRUE(std::ranges::any_of(*used, [&](const auto& r) {
      return w.offset >= r.offset && w.offset + w.bytes <= r.offset + r.bytes;
    }));
  }
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(writes->size(), 110U);  // 25localK/V split,5globalK/V single
  EXPECT_EQ((*writes)[0].offset, 1278U * 4096U);
  EXPECT_EQ((*writes)[0].bytes, 8192U);
  EXPECT_EQ((*writes)[1].offset, 0U);
  EXPECT_EQ((*writes)[1].bytes, 8192U);
}
TEST(Gemma4, DefaultInputStorageDoesNotGrowWithTheAttendedContext) {
  const auto& p = md::Gemma4_31B();
  const auto s = *md::Gemma4State(p, p.context, 8192);
  const std::array<std::int32_t, 1> token{2};
  const std::array<md::Gemma4Segment, 1> first{{{.slot = 0, .n_past = 0, .tokens = token}}},
      last{{{.slot = 0, .n_past = p.context - 1, .tokens = token}}};
  EXPECT_EQ(*md::Gemma4HostInputBytes(p, s, first), 28U + sizeof(md::Gemma4SegmentInputs));
  EXPECT_EQ(*md::Gemma4HostInputBytes(p, s, last), 28U + sizeof(md::Gemma4SegmentInputs));
  const auto in = md::Gemma4Chunk(p, s, last);
  ASSERT_TRUE(in);
  EXPECT_TRUE(in->segments[0].global_mask.empty());
  EXPECT_TRUE(in->segments[0].local_mask.empty());
  EXPECT_EQ(in->segments[0].global_n_kv, p.context);
  std::vector<std::int32_t> widest(8192, 2);
  const std::array<md::Gemma4Segment, 1> big{
      {{.slot = 0, .n_past = p.context - 8192, .tokens = widest}}};
  EXPECT_FALSE(md::Gemma4Chunk(p, s, big, true));
  EXPECT_TRUE(md::Gemma4Chunk(p, s, big));
}
TEST(Gemma4, BatchAdmissionRefusesOverlapOversizeAndBadTokensBeforeBuildingInputs) {
  const auto& p = md::Gemma4_26BA4B();
  const auto s = *md::Gemma4State(p, 8192, 4);
  const std::array<std::int32_t, 3> tokens{2, 3, 4};
  std::array<md::Gemma4Segment, 2> segments{
      {{.slot = 0, .n_past = 0, .tokens = tokens},
       {.slot = 0, .n_past = 0, .tokens = std::span(tokens).first(1)}}};
  EXPECT_FALSE(md::Gemma4Chunk(p, s, segments));
  segments[1].slot = 1;
  EXPECT_TRUE(md::Gemma4Chunk(p, s, segments));
  segments[1].tokens = tokens;
  EXPECT_FALSE(md::Gemma4Chunk(p, s, segments));
  segments[1].tokens = std::span(tokens).first(1);
  segments[1].n_past = 8192;
  EXPECT_FALSE(md::Gemma4Chunk(p, s, segments));
  segments[1].n_past = 0;
  segments[1].slot = 16;
  EXPECT_FALSE(md::Gemma4Chunk(p, s, segments));
  const std::array<std::int32_t, 1> invalid{-1};
  segments[1].slot = 1;
  segments[1].tokens = invalid;
  EXPECT_FALSE(md::Gemma4Chunk(p, s, segments));
  EXPECT_FALSE(md::Gemma4ChunkWrites(p, s, 8192, 1));
  EXPECT_FALSE(md::Gemma4ChunkWrites(p, s, 0, 0));
}
TEST(Gemma4, FakeCacheBatchMatchesIndependentUnrolledWindowReferences) {
  const auto& p = md::Gemma4_26BA4B();
  const auto state = *md::Gemma4State(p, 8192, 4);
  const std::array<std::int32_t, 2> a{2, 3}, b{4, 5};
  const std::array<md::Gemma4Segment, 2> segments{
      {{.slot = 3, .n_past = 1279, .tokens = a}, {.slot = 8, .n_past = 4095, .tokens = b}}};
  const auto in = md::Gemma4Chunk(p, state, segments, true);
  ASSERT_TRUE(in);
  for (const auto& x : in->segments) {
    // Fake persistent values distinguish slots and every absolute position.
    const auto value = [&](std::uint32_t pos) { return std::int64_t{x.slot} * 10000 + pos + 1; };
    std::vector<std::int64_t> local(state.local_cells), global(state.global_cells);
    for (std::uint32_t pos = 0; pos < x.n_past; ++pos) {
      local[pos % state.local_cells] = value(pos);
      global[pos] = value(pos);
    }
    // A graph writes its entire chunk before attention reads any query.
    for (std::uint32_t row = 0; row < x.rows; ++row) {
      local[static_cast<std::size_t>(x.local_cells[row])] = value(x.n_past + row);
      global[static_cast<std::size_t>(x.global_cells[row])] = value(x.n_past + row);
    }
    for (std::uint32_t row = 0; row < x.rows; ++row) {
      std::int64_t local_sum = 0, global_sum = 0, expected_local = 0, expected_global = 0;
      for (std::uint32_t cell = 0; cell < x.local_n_kv; ++cell) {
        if (x.local_mask[std::size_t{row} * x.local_n_kv + cell] == 0) {
          local_sum += local[cell];
        }
      }
      for (std::uint32_t cell = 0; cell < x.global_n_kv; ++cell) {
        if (x.global_mask[std::size_t{row} * x.global_n_kv + cell] == 0) {
          global_sum += global[cell];
        }
      }
      const auto query = x.n_past + row;
      for (std::uint32_t pos = 0; pos <= query; ++pos) {
        expected_global += value(pos);
        if (query - pos < 1024) {
          expected_local += value(pos);
        }
      }
      EXPECT_EQ(local_sum, expected_local);
      EXPECT_EQ(global_sum, expected_global);
    }
  }
}
TEST(Gemma4, StateCursorAllowsGlobalTruncationAndRefusesUnsnapshottedRingRollback) {
  const auto state = *md::Gemma4State(md::Gemma4_26BA4B(), 8192, 4);
  const auto reps = state.Representations(md::Gemma4_26BA4B());
  ASSERT_TRUE(reps);
  auto local = md::StateCursor::Create((*reps)[0], 8192);
  auto global = md::StateCursor::Create((*reps)[10], 8192);
  ASSERT_TRUE(local);
  ASSERT_TRUE(global);
  ASSERT_TRUE(local->Append(1026));
  ASSERT_TRUE(global->Append(1026));
  ASSERT_TRUE(local->Commit(1024));
  ASSERT_TRUE(global->Commit(1024));
  EXPECT_FALSE(local->Truncate(1024));
  EXPECT_TRUE(global->Truncate(1024));
}
TEST(Gemma4, Whole4096RowWriteRetainsEveryQueryWindowAcrossRingWrap) {
  const auto& p = md::Gemma4_26BA4B();
  const auto state = *md::Gemma4State(p, 16384, 4096);
  const std::vector<std::int32_t> tokens(4096, 1);
  const std::array<md::Gemma4Segment, 1> segments{{{.slot = 15, .n_past = 5119, .tokens = tokens}}};
  const auto in = md::Gemma4Chunk(p, state, segments, true);
  ASSERT_TRUE(in);
  ASSERT_EQ(state.local_cells, 5120U);
  const auto& x = in->segments[0];
  std::vector<std::int32_t> cache(state.local_cells, -1);
  // Store absolute positions, then write the complete prefill before any
  // query reads. The expected window is independent of physical ring order.
  for (std::uint32_t position = 0; position < x.n_past + x.rows; ++position) {
    cache[position % state.local_cells] = static_cast<std::int32_t>(position);
  }
  for (std::uint32_t row = 0; row < x.rows; ++row) {
    const auto position = x.n_past + row;
    std::size_t visible = 0;
    for (std::uint32_t cell = 0; cell < x.local_n_kv; ++cell) {
      if (x.local_mask[std::size_t{row} * x.local_n_kv + cell] == 0) {
        ASSERT_GE(cache[cell], 0);
        ASSERT_LE(static_cast<std::uint32_t>(cache[cell]), position);
        ASSERT_LT(position - static_cast<std::uint32_t>(cache[cell]), 1024U);
        ++visible;
      }
    }
    ASSERT_EQ(visible, 1024U) << row;
  }
  const auto writes = md::Gemma4ChunkWrites(p, state, x.n_past, x.rows);
  const auto initialized = md::Gemma4UsedState(p, state, x.n_past + x.rows);
  ASSERT_TRUE(writes);
  ASSERT_TRUE(initialized);
  for (const auto& write : *writes) {
    EXPECT_TRUE(std::ranges::any_of(*initialized, [&](const auto& range) {
      return write.offset >= range.offset &&
             write.offset + write.bytes <= range.offset + range.bytes;
    }));
  }
}
TEST(Gemma4, MaskStrideBoundaryRefusesBeforeAllocationAndBatchEnvelopeDoesNotOverflow) {
  const auto& p = md::Gemma4_31B();
  const auto state = *md::Gemma4State(p, p.context, 8192);
  std::vector<std::int32_t> tokens(4096, 1);
  std::array<md::Gemma4Segment, 1> oversized{
      {{.slot = 0, .n_past = p.context - 4096, .tokens = tokens}}};
  // Exactly 2^31 bytes is already beyond the signed GGML byte stride.
  EXPECT_FALSE(md::Gemma4HostInputBytes(p, state, oversized, true));
  EXPECT_FALSE(md::Gemma4Chunk(p, state, oversized, true));
  const std::array<md::Gemma4Segment, 3> independent{
      {{.slot = 0, .n_past = p.context - 4095, .tokens = std::span(tokens).first(4095)},
       {.slot = 15, .n_past = p.context - 4095, .tokens = std::span(tokens).first(4095)},
       {.slot = 7, .n_past = p.context - 2, .tokens = std::span(tokens).first(2)}}};
  // Each mask is legal; their caller-funded aggregate exceeds 32 bits.
  // Query only the envelope here: no multi-gigabyte masks are allocated.
  const auto envelope = md::Gemma4HostInputBytes(p, state, independent, true);
  ASSERT_TRUE(envelope);
  EXPECT_EQ(*envelope, std::uint64_t{8192} * (28 + 2U * (state.global_cells + state.local_cells)) +
                           3 * sizeof(md::Gemma4SegmentInputs));
  EXPECT_GT(*envelope, std::numeric_limits<std::uint32_t>::max());
}
}  // namespace
