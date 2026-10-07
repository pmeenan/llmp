// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma2.h"

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
#include "gemma2_fixture.h"

namespace {
namespace md = jitllm::model;
namespace json = jitllm::base::json;
using jitllm::test_support::gemma2::Fixture;
using jitllm::test_support::gemma2::Get;
using jitllm::test_support::gemma2::Resources;
using jitllm::test_support::gemma2::Role;

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
  ASSERT_TRUE(bound) << *jitllm::test_support::Failed(bound);
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
  ASSERT_TRUE(alias) << *jitllm::test_support::Failed(alias);
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
}  // namespace
