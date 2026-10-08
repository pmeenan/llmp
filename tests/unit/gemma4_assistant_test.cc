// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "model/gemma4_assistant.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "artifact/representation.h"
#include "base/check.h"
#include "base/json.h"

namespace md = llmp::model;
namespace json = llmp::base::json;
namespace {
json::Document Fixture(unsigned size, bool assistant = true) {
  const auto* root = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  llmp::base::Check(root != nullptr, "missing assistant fixture directory");
  const auto suffix = assistant ? "/gemma4-assistant/assistant" : "/gemma4/gemma4_";
  std::ifstream file(std::string(root) + suffix + std::to_string(size) +
                     (assistant ? ".json" : "b.json"));
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = json::Parse(text);
  llmp::base::Check(doc.has_value(), "invalid assistant fixture");
  return std::move(*doc);
}
json::Value Get(json::Value v, std::string_view key) {
  auto found = v.find(key);
  llmp::base::Check(found.has_value(), "missing assistant fixture key");
  return *found;
}
std::vector<md::Gemma4Resource> Resources(unsigned size, bool assistant = true) {
  const auto doc = Fixture(size, assistant);
  const auto tensors = Get(doc.root(), "tensors");
  std::vector<md::Gemma4Resource> result;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto t = tensors.at(i);
    md::Gemma4Resource r;
    r.roles.emplace_back(Get(t, "name").string());
    if (assistant)
      r.type = Get(t, "type").string();
    else {
      const auto id = *Get(t, "ggml_type").int64();
      const auto types = llmp::artifact::GgmlTypes();
      const auto found =
          std::ranges::find_if(types, [id](const auto& type) { return type.id == id; });
      llmp::base::Check(found != types.end(), "unknown fixture type");
      r.type = found->name;
    }
    const auto ne = Get(t, "ne");
    for (std::size_t j = 0; j < ne.size(); ++j)
      r.ne.push_back(static_cast<std::uint64_t>(*ne.at(j).int64()));
    if (r.ne.size() == 3) {
      r.expert_array = true;
      r.count = static_cast<std::uint32_t>(r.ne.back());
      r.ne.pop_back();
    }
    const auto* type = llmp::artifact::FindGgmlType(r.type);
    r.readable = r.ne[0] / type->block_elements * type->block_bytes;
    for (std::size_t j = 1; j < r.ne.size(); ++j) r.readable *= r.ne[j];
    if (type->block_elements > 1 && r.ne[0] % 512 != 0)
      r.readable += (512 - r.ne[0] % 512) / type->block_elements * type->block_bytes;
    result.push_back(std::move(r));
  }
  return result;
}
void Unsigned(std::vector<std::byte>& bytes, std::uint64_t value, unsigned width) {
  for (unsigned i = 0; i < width; ++i) bytes.push_back(static_cast<std::byte>(value >> (i * 8)));
}
void String(std::vector<std::byte>& bytes, std::string_view text) {
  Unsigned(bytes, text.size(), 8);
  for (char c : text) bytes.push_back(static_cast<std::byte>(c));
}
std::vector<std::byte> Metadata(unsigned size, std::string_view changed = {},
                                std::int64_t value = 0, std::span<const std::int64_t> array = {}) {
  const auto doc = Fixture(size);
  const auto metadata = Get(doc.root(), "metadata");
  std::vector<std::byte> body;
  std::uint64_t count = 0;
  for (std::size_t i = 0; i < metadata.size(); ++i) {
    const auto key = metadata.key(i);
    if (key != "general.architecture" && !key.starts_with("gemma4-assistant.")) continue;
    ++count;
    String(body, key);
    const auto v = metadata.member(i);
    if (key == changed && !array.empty()) {
      Unsigned(body, 9, 4);
      Unsigned(body, 4, 4);
      Unsigned(body, array.size(), 8);
      for (const auto item : array) Unsigned(body, static_cast<std::uint64_t>(item), 4);
    } else if (key == changed) {
      Unsigned(body, 4, 4);
      Unsigned(body, static_cast<std::uint64_t>(value), 4);
    } else if (v.is_string()) {
      Unsigned(body, 8, 4);
      String(body, v.string());
    } else if (v.is_array()) {
      Unsigned(body, 9, 4);
      Unsigned(body, v.at(0).is_bool() ? 7 : 4, 4);
      Unsigned(body, v.size(), 8);
      for (std::size_t j = 0; j < v.size(); ++j)
        Unsigned(
            body,
            v.at(j).is_bool() ? v.at(j).boolean() : static_cast<std::uint64_t>(*v.at(j).int64()),
            v.at(j).is_bool() ? 1 : 4);
    } else if (v.is_integer()) {
      Unsigned(body, 4, 4);
      Unsigned(body, static_cast<std::uint64_t>(*v.int64()), 4);
    } else {
      Unsigned(body, 6, 4);
      Unsigned(body, std::bit_cast<std::uint32_t>(static_cast<float>(*v.float64())), 4);
    }
  }
  if (!changed.empty() && !metadata.find(changed)) {
    ++count;
    String(body, changed);
    Unsigned(body, 4, 4);
    Unsigned(body, static_cast<std::uint64_t>(value), 4);
  }
  std::vector<std::byte> bytes;
  Unsigned(bytes, 0x46554747, 4);
  Unsigned(bytes, 3, 4);
  Unsigned(bytes, 0, 8);
  Unsigned(bytes, count, 8);
  bytes.insert(bytes.end(), body.begin(), body.end());
  return bytes;
}
const md::Gemma4AssistantProfile& Profile(unsigned size) {
  return size == 26 ? md::Gemma4Assistant26() : md::Gemma4Assistant31();
}
}  // namespace
TEST(Gemma4Assistant, ExactPinnedTensorContractsAndArbitraryOrdering) {
  for (unsigned size : {26U, 31U}) {
    auto resources = Resources(size);
    ASSERT_EQ(resources.size(), 49U);
    auto binding = md::BindGemma4Assistant(Profile(size), "gemma4-assistant", resources);
    ASSERT_TRUE(binding);
    EXPECT_TRUE(md::CheckGemma4AssistantBinding(Profile(size), *binding));
    EXPECT_EQ(binding->head, binding->embedding);
    EXPECT_EQ(binding->layers[3].q.ne[1], 512U * Profile(size).heads);
    std::ranges::reverse(resources);
    binding = md::BindGemma4Assistant(Profile(size), "gemma4-assistant", resources);
    ASSERT_TRUE(binding);
    EXPECT_TRUE(md::CheckGemma4AssistantBinding(Profile(size), *binding));
  }
}
TEST(Gemma4Assistant, RejectsUnknownMissingDuplicateExtraAndUnsupportedStorage) {
  const auto original = Resources(26);
  const auto& p = Profile(26);
  EXPECT_FALSE(md::BindGemma4Assistant(p, "gemma4", original));
  EXPECT_FALSE(md::BindGemma4Assistant(Profile(31), "gemma4-assistant", original));
  for (unsigned mutation = 0; mutation < 11; ++mutation) {
    auto r = original;
    switch (mutation) {
      case 0:
        r.pop_back();
        break;
      case 1:
        r.push_back(r[0]);
        break;
      case 2:
        r[1].roles = r[0].roles;
        break;
      case 3:
        r[0].type = "F16";
        break;
      case 4:
        r[0].ne[0] = UINT64_MAX;
        break;
      case 5:
        r[0].ne.resize(100, 1);
        break;
      case 6:
        r[0].readable = 1;
        break;
      case 7:
        r[0].expert_array = true;
        break;
      case 8:
        r[0].group_offset = UINT64_MAX;
        break;
      case 10:
        r[0].readable = UINT64_MAX;
        break;
      case 9:
        r[0].roles = {"masked_embd.centroids.weight"};
        break;
    }
    EXPECT_FALSE(md::BindGemma4Assistant(p, "gemma4-assistant", r)) << mutation;
  }
}
TEST(Gemma4Assistant, MutableDescriptorsCannotChangeIdentityTiesShapeOrStorage) {
  const auto bound = md::BindGemma4Assistant(Profile(26), "gemma4-assistant", Resources(26));
  ASSERT_TRUE(bound);
  for (unsigned i = 0; i < 9; ++i) {
    auto b = *bound;
    switch (i) {
      case 0:
        b.layers[0].q.index = UINT32_MAX;
        break;
      case 1:
        b.layers[0].q.index = b.layers[1].q.index;
        break;
      case 2:
        b.head.index = b.layers[0].q.index;
        break;
      case 3:
        b.layers[0].q.ne.resize(100, 1);
        break;
      case 4:
        b.layers[0].q.readable = 1;
        break;
      case 5:
        b.layers[0].q.ne[1] /= 2;
        break;
      case 6:
        b.rope_freqs.type = "Q8_0";
        break;
      case 7:
        b.embedding.ne.resize(10000, 1);
        b.head = b.embedding;
        break;
      case 8:
        b.embedding.type.assign(10000, 'x');
        b.head = b.embedding;
        break;
    }
    EXPECT_FALSE(md::CheckGemma4AssistantBinding(Profile(26), b)) << i;
  }
}
TEST(Gemma4Assistant, KeptMetadataAuthenticatesSharedKvAndAllArchitectureSemantics) {
  for (unsigned size : {26U, 31U}) {
    const auto& p = Profile(size);
    EXPECT_TRUE(md::CheckGemma4AssistantMetadata(p, Metadata(size)));
    EXPECT_FALSE(md::CheckGemma4AssistantMetadata(Profile(size == 26 ? 31 : 26), Metadata(size)));
    for (const auto* suffix :
         {"block_count", "nextn_predict_layers", "embedding_length_out", "attention.key_length",
          "attention.key_length_swa", "rope.dimension_count", "attention.shared_kv_layers",
          "attention.sliding_window", "rope.freq_base"})
      EXPECT_FALSE(md::CheckGemma4AssistantMetadata(
          p, Metadata(size, std::string("gemma4-assistant.") + suffix, 9)))
          << suffix;
    for (const auto* suffix :
         {"final_logit_softcapping", "use_ordered_embeddings", "rope.scaling.factor",
          "rope.scale_linear", "rope.scaling.attn_factor", "rope.scaling.alpha",
          "rope.scaling.original_context_length", "rope.scaling.finetuned"})
      EXPECT_FALSE(md::CheckGemma4AssistantMetadata(
          p, Metadata(size, std::string("gemma4-assistant.") + suffix, 2)))
          << suffix;
    for (const auto* suffix : {"attention.head_count_kv", "attention.sliding_window_pattern"}) {
      const auto key = std::string("gemma4-assistant.") + suffix;
      EXPECT_FALSE(md::CheckGemma4AssistantMetadata(p, Metadata(size, key, 1)));
      const std::array<std::int64_t, 4> wrong{1, 1, 0, 1};
      EXPECT_FALSE(md::CheckGemma4AssistantMetadata(p, Metadata(size, key, 0, wrong)));
    }
    const std::array<std::int64_t, 4> reordered_kv{p.global_kv_heads, p.local_kv_heads,
                                                   p.local_kv_heads, p.local_kv_heads};
    EXPECT_FALSE(md::CheckGemma4AssistantMetadata(
        p, Metadata(size, "gemma4-assistant.attention.head_count_kv", 0, reordered_kv)));
    EXPECT_FALSE(md::CheckGemma4AssistantMetadata(
        p, Metadata(size, "gemma4-assistant.attention.causal", 0)));
    EXPECT_FALSE(md::CheckGemma4AssistantMetadata(
        p, Metadata(size, "gemma4-assistant.rope.scaling.type", 1)));
    auto bytes = Metadata(size);
    bytes.pop_back();
    EXPECT_FALSE(md::CheckGemma4AssistantMetadata(p, bytes));
  }
}
TEST(Gemma4Assistant, TargetProfilesAndFullBindingsDefineReadOnlyLocalGlobalBorrowing) {
  for (unsigned size : {26U, 31U}) {
    const auto& p = Profile(size);
    const auto& target = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
    const auto b = md::BindGemma4Assistant(p, "gemma4-assistant", Resources(size));
    const auto tb = md::BindGemma4(target, "gemma4", Resources(size, false));
    ASSERT_TRUE(b);
    ASSERT_TRUE(tb);
    auto pair = md::CheckGemma4AssistantTarget(p, *b, target, *tb);
    ASSERT_TRUE(pair);
    for (unsigned i = 0; i < 4; ++i) {
      EXPECT_EQ(pair->layers[i].target_layer,
                i == 3 ? p.global_target_layer : p.local_target_layer);
      EXPECT_EQ(pair->layers[i].local, i != 3);
      EXPECT_EQ(pair->layers[i].rope_dims, i == 3 ? 512U : 256U);
    }
    auto wrong = target;
    ++wrong.local_kv_heads;
    EXPECT_FALSE(md::CheckGemma4AssistantTarget(p, *b, wrong, *tb));
    auto wrong_binding = *tb;
    wrong_binding.layers.back().k.ne[0] = UINT64_MAX;
    EXPECT_FALSE(md::CheckGemma4AssistantTarget(p, *b, target, wrong_binding));
  }
}
TEST(Gemma4Assistant, CanonicalVocabularyPreservesRawScoresAndOnlyTwoExactKindExceptions) {
  std::vector<std::string> tokens(262144);
  std::vector<std::pair<std::string, std::string>> merges(514906);
  std::vector<double> scores(262144);
  std::vector<std::int64_t> target_types(262144, 1), assistant_types = target_types;
  tokens[1] = "<eos>";
  tokens[258884] = "<|video|>";
  target_types[258884] = 3;
  assistant_types[1] = 3;
  md::Gemma4AssistantVocabulary target{tokens, merges, scores, target_types};
  md::Gemma4AssistantVocabulary assistant{tokens, merges, scores, assistant_types};
  EXPECT_TRUE(md::CheckGemma4AssistantVocabulary(target, assistant));
  assistant.types = target_types;
  EXPECT_TRUE(md::CheckGemma4AssistantVocabulary(target, assistant));
  assistant.types = assistant_types;
  assistant_types[2] = 3;
  EXPECT_FALSE(md::CheckGemma4AssistantVocabulary(target, assistant));
  assistant_types[2] = 1;
  tokens[1] = "wrong-eos";
  EXPECT_FALSE(md::CheckGemma4AssistantVocabulary(target, assistant));
  tokens[1] = "<eos>";
  auto changed_scores = scores;
  changed_scores[10] = -0.;
  assistant.scores = changed_scores;
  EXPECT_FALSE(md::CheckGemma4AssistantVocabulary(target, assistant));
  changed_scores[10] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(md::CheckGemma4AssistantVocabulary(target, assistant));
  assistant.scores = scores;
  assistant.tokens = std::span(tokens).first(262143);
  EXPECT_FALSE(md::CheckGemma4AssistantVocabulary(target, assistant));
  assistant.tokens = tokens;
  auto changed_merges = merges;
  changed_merges[0] = {"a", "b"};
  assistant.merges = changed_merges;
  EXPECT_FALSE(md::CheckGemma4AssistantVocabulary(target, assistant));
}
