// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma2.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/representation.h"
#include "base/check.h"
#include "base/json.h"
#include "expected_error.h"
#include "gemma2_fixture.h"

namespace {
namespace md = llmp::model;
namespace json = llmp::base::json;
using llmp::test_support::gemma2::Fixture;
using llmp::test_support::gemma2::Get;
using llmp::test_support::gemma2::Resources;
using llmp::test_support::gemma2::Role;

TEST(Gemma2FoundationTest, ApprovedProfileMatchesActualMetadataAndAllTensorStorage) {
  const auto& p = md::Gemma2_2B();
  const auto fixture = Fixture();
  const auto metadata = Get(fixture.root(), "metadata");
  const auto number = [&](std::string_view key) { return *Get(metadata, key).float64(); };
  EXPECT_EQ(Get(metadata, "general.architecture").string(), "gemma2");
  EXPECT_EQ(number("gemma2.block_count"), p.layers);
  EXPECT_EQ(number("gemma2.embedding_length"), p.width);
  EXPECT_EQ(number("gemma2.feed_forward_length"), p.ffn);
  EXPECT_EQ(number("gemma2.attention.head_count"), p.heads);
  EXPECT_EQ(number("gemma2.attention.head_count_kv"), p.kv_heads);
  EXPECT_EQ(number("gemma2.attention.key_length"), p.key_dim);
  EXPECT_EQ(number("gemma2.attention.value_length"), p.value_dim);
  EXPECT_EQ(number("gemma2.context_length"), p.context);
  EXPECT_EQ(number("gemma2.attention.sliding_window"), p.window);
  EXPECT_EQ(number("vocab_count"), p.vocab);
  EXPECT_FALSE(metadata.find("gemma2.rope.freq_base"));
  EXPECT_FALSE(metadata.find("gemma2.rope.scaling.factor"));
  EXPECT_EQ(p.rope_base, 10000.0F);  // d812 defaults for absent metadata.
  EXPECT_EQ(p.rope_scale, 1.0F);
  EXPECT_EQ(p.attention_scale, 0.0625F);  // 1/sqrt(D256), not width/heads.
  EXPECT_EQ(number("gemma2.attn_logit_softcapping"), p.attention_softcap);
  EXPECT_EQ(number("gemma2.final_logit_softcapping"), p.final_softcap);
  EXPECT_FLOAT_EQ(static_cast<float>(number("gemma2.attention.layer_norm_rms_epsilon")), p.rms_eps);
  const auto resources = Resources();
  ASSERT_EQ(resources.size(), 288);
  const auto bound = md::BindGemma2(p, "gemma2", resources);
  ASSERT_TRUE(bound) << *llmp::test_support::Failed(bound);
  EXPECT_EQ(bound->token_embd.type, "Q8_0");
  EXPECT_EQ(bound->token_embd.readable, 626688272);
  EXPECT_EQ(bound->output, bound->token_embd);
  ASSERT_EQ(bound->layers.size(), 26);
  for (const auto& layer : bound->layers) {
    EXPECT_NE(layer.k.index, layer.v.index);
    EXPECT_EQ(layer.k.ne, layer.v.ne);
    EXPECT_EQ(layer.q.type, "Q8_0");
  }
}

TEST(Gemma2FoundationTest, RejectsIncompleteDuplicateAndUnusedRolesAndIndependentHead) {
  const auto& p = md::Gemma2_2B();
  auto resources = Resources();
  resources.pop_back();
  EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
  resources = Resources();
  resources.back().roles = resources.front().roles;
  EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
  resources = Resources();
  resources.back().roles.push_back("unread.weight");
  EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
  resources = Resources();
  resources.back().roles.clear();
  EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
  resources = Resources();
  Role(resources, "output_norm.weight").roles.push_back("output.weight");
  EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
  resources = Resources();
  Role(resources, "token_embd.weight").roles.push_back("output.weight");
  const auto alias = md::BindGemma2(p, "gemma2", resources);
  ASSERT_TRUE(alias) << *llmp::test_support::Failed(alias);
  EXPECT_EQ(alias->output, alias->token_embd);
}

TEST(Gemma2FoundationTest, RejectsWrongTypesDimensionsShortStorageAndUnapprovedProfiles) {
  const auto& p = md::Gemma2_2B();
  for (const auto role :
       {"token_embd.weight", "blk.0.attn_q.weight", "blk.25.post_ffw_norm.weight"}) {
    auto resources = Resources();
    --Role(resources, role).readable;
    EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
    resources = Resources();
    Role(resources, role).type = "F16";
    EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
    for (const auto dim : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
      resources = Resources();
      Role(resources, role).ne[0] = dim;
      EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
    }
    resources = Resources();
    Role(resources, role).ne.push_back(1);
    EXPECT_FALSE(md::BindGemma2(p, "gemma2", resources));
  }
  const auto resources = Resources();
  auto edited = p;
  --edited.vocab;
  EXPECT_FALSE(md::BindGemma2(edited, "gemma2", resources));
  edited = p;
  edited.attention_softcap = 0;
  EXPECT_FALSE(md::BindGemma2(edited, "gemma2", resources));
  EXPECT_FALSE(md::BindGemma2(p, "gemma4", resources));
}

TEST(Gemma2FoundationTest, EveryStoredTensorMustFundCanonicalReadableTail) {
  const auto& p = md::Gemma2_2B();
  const auto resources = Resources();
  const auto fixture = Fixture();
  const auto tensors = Get(fixture.root(), "tensors");
  std::size_t padded = 0;
  for (std::size_t i = 0; i < resources.size(); ++i) {
    const auto stored = static_cast<std::uint64_t>(*Get(tensors.at(i), "bytes").int64());
    const auto tail = resources[i].readable - stored;
    EXPECT_EQ(tail, resources[i].type == "Q8_0" && resources[i].ne[0] % 512 != 0 ? 272U : 0U);
    if (tail != 0) {
      ++padded;
      auto shortened = resources;
      shortened[i].readable = stored;
      EXPECT_FALSE(md::BindGemma2(p, "gemma2", shortened)) << i;
    }
    auto shortened = resources;
    --shortened[i].readable;
    EXPECT_FALSE(md::BindGemma2(p, "gemma2", shortened)) << i;
  }
  EXPECT_EQ(padded, 131);
}

TEST(Gemma2FoundationTest, PublicDescriptorsAreCheckedWithoutReconstruction) {
  const auto& p = md::Gemma2_2B();
  auto bound = md::BindGemma2(p, "gemma2", Resources());
  ASSERT_TRUE(bound);
  EXPECT_TRUE(md::CheckGemma2Binding(p, *bound));
  auto b = *bound;
  b.layers[0].k.index = b.layers[0].v.index;
  EXPECT_FALSE(md::CheckGemma2Binding(p, b));
  b = *bound;
  b.layers[0].q.index = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(md::CheckGemma2Binding(p, b));
  b = *bound;
  b.layers[25].ffn_post_norm.ne[0] = std::numeric_limits<std::uint64_t>::max();
  EXPECT_FALSE(md::CheckGemma2Binding(p, b));
  b = *bound;
  --b.layers[0].q.readable;
  EXPECT_FALSE(md::CheckGemma2Binding(p, b));
  b = *bound;
  b.output.index = b.output_norm.index;
  EXPECT_FALSE(md::CheckGemma2Binding(p, b));
  b = *bound;
  b.layers.pop_back();
  EXPECT_FALSE(md::CheckGemma2Binding(p, b));
  auto changed = p;
  changed.final_softcap = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(md::CheckGemma2Binding(changed, *bound));
  EXPECT_FALSE(md::BindGemma2(changed, "gemma2", Resources()));
  // Resource order is not identity: a complete unique permutation is valid.
  auto reversed = Resources();
  std::ranges::reverse(reversed);
  auto permutation = md::BindGemma2(p, "gemma2", reversed);
  ASSERT_TRUE(permutation);
  EXPECT_TRUE(md::CheckGemma2Binding(p, *permutation));
}

TEST(Gemma2FoundationTest, AlternatingLocalGlobalScheduleHasNoGemma3Pattern) {
  const auto& p = md::Gemma2_2B();
  std::size_t local = 0;
  for (std::uint32_t i = 0; i < p.layers; ++i) {
    EXPECT_EQ(p.local(i), i % 2 == 0);
    local += p.local(i);
  }
  EXPECT_EQ(local, 13);
  EXPECT_FALSE(p.local(25));
}
TEST(Gemma2StateTest, ScheduleRepresentationsAndInitializedFootprintsAreChecked) {
  const auto& p = md::Gemma2_2B();
  const std::vector<std::uint32_t> globals{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25};
  auto state = md::Gemma2State(p, 8192, 16);
  ASSERT_TRUE(state) << *llmp::test_support::Failed(state);
  EXPECT_EQ(state->global_cells, 8192);
  EXPECT_EQ(state->local_cells, 4352);
  ASSERT_EQ(state->tensors.size(), 52);
  auto representations = state->Representations(p);
  ASSERT_TRUE(representations);
  std::uint64_t allowance = 0;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    EXPECT_EQ(!p.local(il), std::ranges::contains(globals, il));
    for (std::uint32_t value = 0; value < 2; ++value) {
      const auto index = std::size_t{il} * 2 + value;
      const auto& tensor = state->tensors[index];
      const auto& representation = (*representations)[index];
      EXPECT_EQ(tensor.width, 1024);
      EXPECT_EQ(tensor.local, p.local(il));
      EXPECT_EQ(tensor.offset % (2U << 20U), 0);
      EXPECT_TRUE(md::IsValid(representation));
      EXPECT_TRUE(representation.Can(md::StateCapability::kAppend));
      EXPECT_EQ(representation.CanTruncate(), !p.local(il));
      EXPECT_EQ(representation.max_snapshots, 0);
      allowance += representation.block_bytes.value();
    }
  }
  EXPECT_EQ(allowance, state->bytes);
  EXPECT_TRUE(p.local(24));
  auto empty = md::Gemma2UsedState(p, *state, 0);
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->empty());
  auto initialized = md::Gemma2UsedState(p, *state, 4353);
  ASSERT_TRUE(initialized);
  ASSERT_EQ(initialized->size(), 52);
  for (std::size_t i = 0; i < initialized->size(); ++i) {
    EXPECT_EQ((*initialized)[i].offset, state->tensors[i].offset);
    EXPECT_EQ((*initialized)[i].bytes,
              std::uint64_t{p.local(static_cast<std::uint32_t>(i / 2)) ? 4352U : 4608U} * 2048);
    EXPECT_LE((*initialized)[i].bytes, state->tensors[i].bytes);
  }
  auto malformed = *state;
  ++malformed.tensors[0].offset;
  EXPECT_FALSE(malformed.Representations(p));
  EXPECT_FALSE(md::Gemma2UsedState(p, malformed, 1));
  EXPECT_FALSE(md::Gemma2State(p, 0, 1));
  EXPECT_FALSE(md::Gemma2State(p, p.context + 1, 1));
  EXPECT_FALSE(md::Gemma2State(p, 16, 17));
  EXPECT_FALSE(md::Gemma2UsedState(p, *state, 8193));
  EXPECT_FALSE(md::Gemma2UsedState(p, *state, 1, 255));
}

TEST(Gemma2StateTest, RaggedInputsKeepIndependentCausalityAndSplitRingWrites) {
  const auto& p = md::Gemma2_2B();
  auto state = md::Gemma2State(p, 8192, 16);
  ASSERT_TRUE(state);
  const std::vector<std::int32_t> a{1, 2, 3}, b{4};
  const std::vector<md::Gemma2Segment> segments{{3, 4351, a}, {1, 7, b}};
  auto envelope = md::Gemma2HostInputBytes(p, *state, segments, true);
  auto input = md::Gemma2Chunk(p, *state, segments, true);
  ASSERT_TRUE(envelope);
  ASSERT_TRUE(input) << *llmp::test_support::Failed(input);
  EXPECT_EQ(input->positions, (std::vector<std::int32_t>{4351, 4352, 4353, 7}));
  EXPECT_EQ(input->out_ids, (std::vector<std::int32_t>{0, 1, 2, 3}));
  ASSERT_EQ(input->segments.size(), 2);
  const auto& first = input->segments[0];
  EXPECT_EQ(first.local_cells, (std::vector<std::int64_t>{4351, 0, 1}));
  EXPECT_EQ(first.global_cells, (std::vector<std::int64_t>{4351, 4352, 4353}));
  EXPECT_EQ(input->segments[1].first_row, 3);
  // Writing all rows first must not expose future wrapped rows to query0.
  EXPECT_EQ(first.local_mask[0], 0xFC00);
  EXPECT_EQ(first.local_mask[4351], 0);
  EXPECT_EQ(first.local_mask[255], 0xFC00);
  EXPECT_EQ(first.local_mask[256], 0);
  EXPECT_EQ(first.local_mask[4352], 0);
  EXPECT_EQ(first.global_mask[4352], 0xFC00);
  EXPECT_EQ(input->segments[1].global_mask[7], 0);
  EXPECT_EQ(input->segments[1].global_mask[8], 0xFC00);
  auto writes = md::Gemma2ChunkWrites(p, *state, 4351, 3);
  ASSERT_TRUE(writes);
  EXPECT_EQ(writes->size(), 78);  // 26 local planes split; 26 globals append.
  EXPECT_EQ((*writes)[0].offset, state->tensors[0].offset + 4351 * 2048);
  EXPECT_EQ((*writes)[0].bytes, 2048);
  EXPECT_EQ((*writes)[1].offset, state->tensors[0].offset);
  EXPECT_EQ((*writes)[1].bytes, 4096);
  for (const auto& range : *writes) EXPECT_LE(range.offset + range.bytes, state->bytes);
  auto duplicate = segments;
  duplicate[1].slot = duplicate[0].slot;
  EXPECT_FALSE(md::Gemma2Chunk(p, *state, duplicate));
  auto edited = segments;
  edited[0].n_past = 8191;
  EXPECT_FALSE(md::Gemma2HostInputBytes(p, *state, edited));
  const std::vector<std::int32_t> invalid{-1};
  edited[0] = {0, 0, invalid};
  EXPECT_FALSE(md::Gemma2Chunk(p, *state, edited));
  EXPECT_FALSE(md::Gemma2ChunkWrites(p, *state, 8191, 2));
  EXPECT_FALSE(md::Gemma2ChunkWrites(p, *state, 0, 0));
}

TEST(Gemma2StateTest, ExplicitWaveRowsPreserveEachOwnersRingAndChunkBound) {
  const auto& p = md::Gemma2_2B();
  auto state = md::Gemma2State(p, 8192, 128);
  ASSERT_TRUE(state);
  const auto bytes = state->bytes;
  const std::vector<std::int32_t> a(128, 2), b(128, 3), oversized(129, 4);
  std::array<md::Gemma2Segment, 2> segments{{{0, 4351, a}, {1, 4608, b}}};
  EXPECT_FALSE(md::Gemma2Chunk(p, *state, segments, true));  // legacy total128
  auto input = md::Gemma2Chunk(p, *state, segments, true, 256, 256);
  auto funded = md::Gemma2HostInputBytes(p, *state, segments, true, 256, 256);
  ASSERT_TRUE(input && funded);
  EXPECT_EQ(input->tokens.size(), 256U);
  EXPECT_EQ(input->segments[1].first_row, 128U);
  EXPECT_EQ(input->positions[128], 4608);
  EXPECT_EQ(state->max_rows, 128U);
  EXPECT_EQ(state->local_cells, 4352U);
  EXPECT_EQ(state->bytes, bytes);
  for (const auto& segment : input->segments) {
    EXPECT_EQ(segment.rows, 128U);
    EXPECT_EQ(segment.local_n_kv, 4352U);
    EXPECT_EQ(segment.local_mask.size(), 128U * 4352U);
  }
  for (auto bad : {127U, 129U, 2049U, 4097U}) {
    EXPECT_FALSE(md::Gemma2Chunk(p, *state, segments, true, 256, bad));
    EXPECT_FALSE(md::Gemma2HostInputBytes(p, *state, segments, true, 256, bad));
  }
  segments[0].tokens = oversized;
  EXPECT_FALSE(md::Gemma2Chunk(p, *state, segments, true, 256, 256));
  EXPECT_FALSE(md::Gemma2ChunkWrites(p, *state, 4351, 129));
  segments[0].tokens = a;
  segments[1].slot = 0;
  EXPECT_FALSE(md::Gemma2Chunk(p, *state, segments, true, 256, 256));
}

}  // namespace
