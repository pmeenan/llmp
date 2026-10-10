// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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
namespace md = llmp::model;
namespace json = llmp::base::json;
json::Value Get(json::Value value, std::string_view key) {
  auto found = value.find(key);
  llmp::base::Check(found.has_value(), "Gemma4 fixture has a missing key");
  return *found;
}
json::Document Fixture(std::uint32_t size) {
  const auto* dir = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  llmp::base::Check(dir != nullptr, "Gemma4 test fixture directory is missing");
  std::ifstream file(std::string(dir) + "/gemma4/gemma4_" + std::to_string(size) + "b.json");
  llmp::base::Check(file.good(), "Gemma4 fixture is missing");
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = json::Parse(text);
  llmp::base::Check(doc.has_value(), "Gemma4 fixture is not JSON");
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
  const auto doc = llmp::artifact::json::Parse(text);
  llmp::base::Check(doc.has_value(), "Gemma4 test representation is not JSON");
  const auto repr = llmp::artifact::ParseRepresentation(doc->root());
  llmp::base::Check(repr.has_value(), "Gemma4 test representation is invalid");
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
    const auto types = llmp::artifact::GgmlTypes();
    const auto type = std::ranges::find_if(types, [id](const auto& x) { return x.id == id; });
    llmp::base::Check(type != types.end(), "Gemma4 fixture GGML type is unknown");
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
  llmp::base::Check(found != resources.end(), "Gemma4 test role is missing");
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
    ASSERT_TRUE(bound) << *llmp::test_support::Failed(bound);
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
}  // namespace
