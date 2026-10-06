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

namespace {
namespace md = jitllm::model;
namespace json = jitllm::base::json;
json::Value Get(json::Value value, std::string_view key) {
  const auto found = value.find(key);
  jitllm::base::Check(found.has_value(), "Gemma3 fixture key is missing");
  return *found;
}
json::Document Fixture() {
  const auto* dir = std::getenv("JITLLM_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  jitllm::base::Check(dir != nullptr, "Gemma3 test data directory is missing");
  std::ifstream file(std::string(dir) + "/gemma3/gemma3_4b_qat.json");
  jitllm::base::Check(file.good(), "Gemma3 fixture is missing");
  const std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = json::Parse(text);
  jitllm::base::Check(doc.has_value(), "Gemma3 fixture is invalid JSON");
  return std::move(*doc);
}
std::vector<md::Gemma3Resource> Resources() {
  const auto fixture = Fixture();
  const auto tensors = Get(fixture.root(), "tensors");
  std::vector<md::Gemma3Resource> resources;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto tensor = tensors.at(i);
    md::Gemma3Resource resource;
    resource.roles = {std::string(Get(tensor, "name").string())};
    resource.type = Get(tensor, "type").string();
    const auto ne = Get(tensor, "ne");
    for (std::size_t dim = 0; dim < ne.size(); ++dim) {
      resource.ne.push_back(static_cast<std::uint64_t>(*ne.at(dim).int64()));
    }
    std::string representation = "{\"family\":\"ggml\",\"type\":\"" + resource.type + "\",\"ne\":[";
    for (const auto dim : resource.ne) {
      if (representation.back() != '[') {
        representation += ',';
      }
      representation += std::to_string(dim);
    }
    representation += "]}";
    const auto doc = jitllm::artifact::json::Parse(representation);
    jitllm::base::Check(doc.has_value(), "Gemma3 representation is invalid JSON");
    const auto parsed = jitllm::artifact::ParseRepresentation(doc->root());
    jitllm::base::Check(parsed.has_value(), "Gemma3 representation is invalid");
    jitllm::base::Check(parsed->bytes == static_cast<std::uint64_t>(*Get(tensor, "bytes").int64()),
                        "Gemma3 recorded bytes differ from native type helpers");
    jitllm::base::Check(parsed->bytes == parsed->readable,
                        "approved Gemma3 rows unexpectedly require padding");
    const auto* traits = jitllm::artifact::FindGgmlType(resource.type);
    jitllm::base::Check(traits != nullptr && static_cast<std::int64_t>(traits->id) ==
                                                 *Get(tensor, "ggml_type").int64(),
                        "Gemma3 recorded type ID differs from native helpers");
    resource.readable = parsed->readable;
    resources.push_back(std::move(resource));
  }
  return resources;
}
md::Gemma3Resource& Role(std::vector<md::Gemma3Resource>& resources, std::string_view role) {
  const auto found = std::ranges::find_if(resources, [role](const auto& resource) {
    return std::ranges::contains(resource.roles, role);
  });
  jitllm::base::Check(found != resources.end(), "Gemma3 role is missing");
  return *found;
}

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
}  // namespace
