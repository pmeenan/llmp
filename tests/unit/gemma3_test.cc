// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma3.h"

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
#include "gemma3_fixture.h"

namespace {
namespace md = llmp::model;
namespace json = llmp::base::json;
using llmp::test_support::gemma3::Fixture;
using llmp::test_support::gemma3::Get;
using llmp::test_support::gemma3::Resources;
using llmp::test_support::gemma3::Role;

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
  ASSERT_TRUE(bound) << *llmp::test_support::Failed(bound);
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
  ASSERT_TRUE(alias) << *llmp::test_support::Failed(alias);
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

TEST(Gemma3FoundationTest, PublicBindingChecksEveryMutableDescriptorAndResourceIdentity) {
  const auto& p = md::Gemma3_4BQat();
  auto binding = md::BindGemma3(p, "gemma3", Resources());
  ASSERT_TRUE(binding);
  std::vector<md::Gemma3Tensor*> tensors{&binding->token_embd, &binding->output_norm};
  for (auto& layer : binding->layers)
    for (auto* tensor : {&layer.attn_norm, &layer.q, &layer.k, &layer.v, &layer.out, &layer.q_norm,
                         &layer.k_norm, &layer.attn_post_norm, &layer.ffn_norm, &layer.gate,
                         &layer.up, &layer.down, &layer.ffn_post_norm})
      tensors.push_back(tensor);
  ASSERT_EQ(tensors.size(), 444);
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    SCOPED_TRACE(i);
    auto& tensor = *tensors[i];
    const auto original = tensor;
    const auto refuses = [&] {
      // Also test embedding mutations past the tied-head equality guard.
      binding->output = binding->token_embd;
      EXPECT_FALSE(md::CheckGemma3Binding(p, *binding));
      tensor = original;
      binding->output = binding->token_embd;
    };
    tensor.type = "F16";
    refuses();
    tensor.type = "unrecognized";
    refuses();
    tensor.ne[0] = std::numeric_limits<std::uint64_t>::max();
    refuses();
    tensor.ne.push_back(1);
    refuses();
    --tensor.readable;
    refuses();
    tensor.index = 444;
    refuses();
    tensor.index = i == 0 ? binding->output_norm.index : binding->token_embd.index;
    refuses();
  }
  EXPECT_TRUE(md::CheckGemma3Binding(p, *binding));
}

TEST(Gemma3FoundationTest, PublicBindingAllowsPermutedIndicesAndExcessReadableStorage) {
  const auto& p = md::Gemma3_4BQat();
  auto resources = Resources();
  std::ranges::reverse(resources);
  for (auto& resource : resources) resource.readable += 4096;
  const auto binding = md::BindGemma3(p, "gemma3", resources);
  ASSERT_TRUE(binding);
  EXPECT_TRUE(md::CheckGemma3Binding(p, *binding));
  auto changed_profile = p;
  --changed_profile.vocab;
  EXPECT_FALSE(md::CheckGemma3Binding(changed_profile, *binding));
  auto changed = *binding;
  --changed.output.readable;
  EXPECT_FALSE(md::CheckGemma3Binding(p, changed));
}

TEST(Gemma3FoundationTest, PublicBindingIdentityIsRecheckedBeforeGraphUse) {
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
