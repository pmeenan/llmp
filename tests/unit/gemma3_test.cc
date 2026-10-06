// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma3.h"

#include <gtest/gtest.h>

#include <algorithm>
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
#include "gemma3_fixture.h"

namespace {
namespace md = jitllm::model;
namespace json = jitllm::base::json;
using jitllm::test_support::gemma3::Fixture;
using jitllm::test_support::gemma3::Get;
using jitllm::test_support::gemma3::Resources;
using jitllm::test_support::gemma3::Role;

TEST(Gemma3FoundationTest, ApprovedProfileMatchesActualMetadataAndAllTensorStorage) {
  const auto& p = md::Gemma3_4BQat();
  const auto fixture = Fixture();
  const auto metadata = Get(fixture.root(), "metadata");
  const auto number = [&](std::string_view key) { return *Get(metadata, key).float64(); };
  EXPECT_EQ(Get(metadata, "general.architecture").string(), "gemma3");
  EXPECT_EQ(number("gemma3.block_count"), p.layers);
  EXPECT_EQ(number("gemma3.embedding_length"), p.width);
  EXPECT_EQ(number("gemma3.feed_forward_length"), p.ffn);
  EXPECT_EQ(number("gemma3.attention.head_count"), p.heads);
  EXPECT_EQ(number("gemma3.attention.head_count_kv"), p.kv_heads);
  EXPECT_EQ(number("gemma3.attention.key_length"), p.key_dim);
  EXPECT_EQ(number("gemma3.attention.value_length"), p.value_dim);
  EXPECT_EQ(number("gemma3.context_length"), p.context);
  EXPECT_EQ(number("gemma3.attention.sliding_window"), p.window);
  EXPECT_EQ(number("vocab_count"), p.vocab);
  EXPECT_EQ(number("gemma3.rope.freq_base"), p.rope_base);
  EXPECT_EQ(number("gemma3.rope.scaling.factor"), p.rope_scale);
  EXPECT_EQ(Get(metadata, "gemma3.rope.scaling.type").string(), p.rope_scaling);
  EXPECT_FLOAT_EQ(static_cast<float>(number("gemma3.attention.layer_norm_rms_epsilon")), p.rms_eps);
  const auto resources = Resources();
  ASSERT_EQ(resources.size(), 444);
  const auto bound = md::BindGemma3(p, "gemma3", resources);
  ASSERT_TRUE(bound) << *jitllm::test_support::Failed(bound);
  EXPECT_EQ(bound->token_embd.type, "Q8_0");
  EXPECT_EQ(bound->token_embd.readable, 713205760);
  EXPECT_EQ(bound->output, bound->token_embd);
  ASSERT_EQ(bound->layers.size(), 34);
  for (const auto& layer : bound->layers) {
    EXPECT_NE(layer.k.index, layer.v.index);
    EXPECT_EQ(layer.k.ne, layer.v.ne);
    EXPECT_EQ(layer.q.type, "Q4_0");
    EXPECT_EQ(layer.q_norm.ne, (std::vector<std::uint64_t>{256}));
    EXPECT_EQ(layer.k_norm.ne, (std::vector<std::uint64_t>{256}));
  }
}

TEST(Gemma3FoundationTest, RejectsIncompleteDuplicateAndUnusedRolesAndIndependentHead) {
  const auto& p = md::Gemma3_4BQat();
  auto resources = Resources();
  resources.pop_back();
  EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
  resources = Resources();
  resources.back().roles = resources.front().roles;
  EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
  resources = Resources();
  resources.back().roles.push_back("unread.weight");
  EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
  resources = Resources();
  resources.back().roles.clear();
  EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
  resources = Resources();
  Role(resources, "output_norm.weight").roles.push_back("output.weight");
  EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
  resources = Resources();
  Role(resources, "token_embd.weight").roles.push_back("output.weight");
  const auto alias = md::BindGemma3(p, "gemma3", resources);
  ASSERT_TRUE(alias) << *jitllm::test_support::Failed(alias);
  EXPECT_EQ(alias->output, alias->token_embd);
}

TEST(Gemma3FoundationTest, RejectsWrongTypesDimensionsShortStorageAndUnapprovedProfiles) {
  const auto& p = md::Gemma3_4BQat();
  for (const auto role :
       {"token_embd.weight", "blk.0.attn_q.weight", "blk.33.post_ffw_norm.weight"}) {
    auto resources = Resources();
    --Role(resources, role).readable;
    EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
    resources = Resources();
    Role(resources, role).type = "F16";
    EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
    for (const auto dim : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
      resources = Resources();
      Role(resources, role).ne[0] = dim;
      EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
    }
    resources = Resources();
    Role(resources, role).ne.push_back(1);
    EXPECT_FALSE(md::BindGemma3(p, "gemma3", resources));
  }
  const auto resources = Resources();
  auto edited = p;
  --edited.vocab;
  EXPECT_FALSE(md::BindGemma3(edited, "gemma3", resources));
  edited = p;
  edited.rope_scale = 1;
  EXPECT_FALSE(md::BindGemma3(edited, "gemma3", resources));
  EXPECT_FALSE(md::BindGemma3(p, "gemma4", resources));
}

TEST(Gemma3StateTest, ScheduleRepresentationsAndInitializedFootprintsAreChecked) {
  const auto& p = md::Gemma3_4BQat();
  const std::vector<std::uint32_t> globals{5, 11, 17, 23, 29};
  auto state = md::Gemma3State(p, 4096, 16);
  ASSERT_TRUE(state) << *jitllm::test_support::Failed(state);
  EXPECT_EQ(state->global_cells, 4096);
  EXPECT_EQ(state->local_cells, 1280);
  ASSERT_EQ(state->tensors.size(), 68);
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
  EXPECT_TRUE(p.local(33));
  auto empty = md::Gemma3UsedState(p, *state, 0);
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->empty());
  auto initialized = md::Gemma3UsedState(p, *state, 1281);
  ASSERT_TRUE(initialized);
  ASSERT_EQ(initialized->size(), 68);
  for (std::size_t i = 0; i < initialized->size(); ++i) {
    EXPECT_EQ((*initialized)[i].offset, state->tensors[i].offset);
    EXPECT_EQ((*initialized)[i].bytes,
              std::uint64_t{p.local(static_cast<std::uint32_t>(i / 2)) ? 1280U : 1536U} * 2048);
    EXPECT_LE((*initialized)[i].bytes, state->tensors[i].bytes);
  }
  auto malformed = *state;
  ++malformed.tensors[0].offset;
  EXPECT_FALSE(malformed.Representations(p));
  EXPECT_FALSE(md::Gemma3UsedState(p, malformed, 1));
  EXPECT_FALSE(md::Gemma3State(p, 0, 1));
  EXPECT_FALSE(md::Gemma3State(p, p.context + 1, 1));
  EXPECT_FALSE(md::Gemma3State(p, 16, 17));
  EXPECT_FALSE(md::Gemma3UsedState(p, *state, 4097));
  EXPECT_FALSE(md::Gemma3UsedState(p, *state, 1, 255));
}

TEST(Gemma3StateTest, RaggedInputsKeepIndependentCausalityAndSplitRingWrites) {
  const auto& p = md::Gemma3_4BQat();
  auto state = md::Gemma3State(p, 4096, 16);
  ASSERT_TRUE(state);
  const std::vector<std::int32_t> a{1, 2, 3}, b{4};
  const std::vector<md::Gemma3Segment> segments{{3, 1279, a}, {1, 7, b}};
  auto envelope = md::Gemma3HostInputBytes(p, *state, segments, true);
  auto input = md::Gemma3Chunk(p, *state, segments, true);
  ASSERT_TRUE(envelope);
  ASSERT_TRUE(input) << *jitllm::test_support::Failed(input);
  EXPECT_EQ(input->positions, (std::vector<std::int32_t>{1279, 1280, 1281, 7}));
  EXPECT_EQ(input->out_ids, (std::vector<std::int32_t>{0, 1, 2, 3}));
  ASSERT_EQ(input->segments.size(), 2);
  const auto& first = input->segments[0];
  EXPECT_EQ(first.local_cells, (std::vector<std::int64_t>{1279, 0, 1}));
  EXPECT_EQ(first.global_cells, (std::vector<std::int64_t>{1279, 1280, 1281}));
  EXPECT_EQ(input->segments[1].first_row, 3);
  // Writing all rows first must not expose future wrapped rows to query0.
  EXPECT_EQ(first.local_mask[0], 0xFC00);
  EXPECT_EQ(first.local_mask[1279], 0);
  EXPECT_EQ(first.local_mask[255], 0xFC00);
  EXPECT_EQ(first.local_mask[256], 0);
  EXPECT_EQ(first.local_mask[1280], 0);
  EXPECT_EQ(first.global_mask[1280], 0xFC00);
  EXPECT_EQ(input->segments[1].global_mask[7], 0);
  EXPECT_EQ(input->segments[1].global_mask[8], 0xFC00);
  auto writes = md::Gemma3ChunkWrites(p, *state, 1279, 3);
  ASSERT_TRUE(writes);
  EXPECT_EQ(writes->size(), 126);  // 58 local planes split; ten globals append.
  EXPECT_EQ((*writes)[0].offset, state->tensors[0].offset + 1279 * 2048);
  EXPECT_EQ((*writes)[0].bytes, 2048);
  EXPECT_EQ((*writes)[1].offset, state->tensors[0].offset);
  EXPECT_EQ((*writes)[1].bytes, 4096);
  for (const auto& range : *writes) EXPECT_LE(range.offset + range.bytes, state->bytes);
  auto duplicate = segments;
  duplicate[1].slot = duplicate[0].slot;
  EXPECT_FALSE(md::Gemma3Chunk(p, *state, duplicate));
  auto edited = segments;
  edited[0].n_past = 4095;
  EXPECT_FALSE(md::Gemma3HostInputBytes(p, *state, edited));
  const std::vector<std::int32_t> invalid{-1};
  edited[0] = {0, 0, invalid};
  EXPECT_FALSE(md::Gemma3Chunk(p, *state, edited));
  EXPECT_FALSE(md::Gemma3ChunkWrites(p, *state, 4095, 2));
  EXPECT_FALSE(md::Gemma3ChunkWrites(p, *state, 0, 0));
}

TEST(Gemma3StateTest, PublicBindingIdentityIsRecheckedBeforeGraphUse) {
  const auto& p = md::Gemma3_4BQat();
  auto binding = md::BindGemma3(p, "gemma3", Resources());
  ASSERT_TRUE(binding);
  EXPECT_TRUE(md::CheckGemma3Binding(p, *binding));
  auto changed = *binding;
  changed.layers[0].k.index = changed.layers[0].v.index;
  EXPECT_FALSE(md::CheckGemma3Binding(p, changed));
  changed = *binding;
  changed.layers[0].q.index = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(md::CheckGemma3Binding(p, changed));
  changed = *binding;
  changed.layers[33].ffn_post_norm.ne[0] = std::numeric_limits<std::uint64_t>::max();
  EXPECT_FALSE(md::CheckGemma3Binding(p, changed));
  changed = *binding;
  --changed.layers[0].q.readable;
  EXPECT_FALSE(md::CheckGemma3Binding(p, changed));
  changed = *binding;
  changed.output.index = changed.output_norm.index;
  EXPECT_FALSE(md::CheckGemma3Binding(p, changed));
  changed = *binding;
  changed.layers.pop_back();
  EXPECT_FALSE(md::CheckGemma3Binding(p, changed));
}
}  // namespace
