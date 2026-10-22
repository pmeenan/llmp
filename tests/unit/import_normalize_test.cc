// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "base/check.h"
#include "base/sha256.h"
#include "components/step.h"
#include "dsv4_gguf_normalization_fixture.h"
#include "expected_error.h"
#include "gemma2_fixture.h"
#include "gemma3_fixture.h"
#include "gemma4_assistant_fixture.h"
#include "gemma4_fixture.h"
#include "import/normalize.h"
#include "import/readers/gguf.h"
#include "import/readers/safetensors.h"
#include "import_fixtures.h"
#include "import_target_fixture.h"
#include "model/state_contract.h"
#include "qwen2_exl3_normalization_fixture.h"
#include "qwen2_gguf_normalization_fixture.h"
#include "qwen38_gguf_normalization_fixture.h"
#include "spec/mappings/qwen2.h"
#include "spec/mappings/qwen38.h"

namespace llmp::import {
namespace {
using format::Encoding;
namespace nd = normalization_detail;
namespace json = base::json;
constexpr NormalizationLimits kFunded{std::uint64_t{2} << 30U, std::uint64_t{1} << 45U};

// Pinned libstdc++ node allocations used by mapping.cc and step.cc, including
// links/alignment rather than just value_type. This test target must compile
// before the normalizer's storage ceiling is accepted on a new ABI.
template <class Value>
constexpr bool kNodeFits = sizeof(std::_Rb_tree_node<Value>) <= 256;
static_assert(kNodeFits<std::pair<const std::string_view, const spec::SourceTensor*>>);
static_assert(kNodeFits<std::pair<const std::string_view, const spec::ParamValue*>>);
static_assert(kNodeFits<std::pair<const std::pair<std::size_t, std::uint32_t>, std::string>>);
static_assert(kNodeFits<std::pair<const std::string_view, const spec::SlotRule*>>);
static_assert(
    kNodeFits<std::pair<const std::tuple<std::string_view, std::string_view, format::PartRole>,
                        std::string>>);
static_assert(kNodeFits<std::pair<const std::string, std::span<const spec::SourceTensor>>>);
static_assert(kNodeFits<std::string_view>);
// Exact field mirrors of the private records in components/step.cc. Their
// source identity is part of the normalizer's bounded implementation profile.
struct RoleFormatAbi {
  format::TensorFormat format;
  std::vector<std::pair<std::string, format::PartRole>> parts;
  std::string stage;
  bool experts = false;
};
struct ContractTensorAbi {
  std::uint32_t layer = 0;
  std::uint16_t index = 0;
  model::StateTensorContract contract;
  const components::Instance* instance = nullptr;
};
static_assert(sizeof(std::optional<RoleFormatAbi>) <= 256);
static_assert(sizeof(ContractTensorAbi) <= 256);
static_assert(sizeof(components::RoleShape) <= 256);
static_assert(sizeof(std::pair<std::string, format::PartRole>) <= 256);
static_assert(sizeof(SourceFile) <= 256 && sizeof(Error) <= 64);

void Ordered(Checkpoint& s) { std::ranges::sort(s.metadata, {}, &MetaEntry::key); }
void Value(Checkpoint& s, std::string key, MetaValue value) {
  const auto old = std::ranges::find(s.metadata, key, &MetaEntry::key);
  if (old == s.metadata.end())
    s.metadata.push_back({std::move(key), std::move(value)});
  else
    old->value = std::move(value);
  Ordered(s);
}
void AddTensor(Checkpoint& s, std::string name, std::initializer_list<std::uint64_t> ne,
               Encoding encoding = Encoding::kF32) {
  const auto shape = format::ShapeOf(std::span(ne.begin(), ne.size()));
  base::Check(shape.has_value(), "test shape");
  const auto& info = format::Info(encoding);
  format::TensorFormat f{
      .encoding = encoding,
      .layout = info.quantized ? format::Layout::kGgmlBlocks : format::Layout::kDense,
      .shape = *shape};
  const auto parts = format::PartsOf(f);
  base::Check(parts.has_value(), "test format");
  const auto offset = (s.files[0].bytes + 31) / 32 * 32;
  s.tensors.push_back({.name = std::move(name),
                       .format = f,
                       .extent = {.offset = offset, .bytes = (*parts)[0].bytes},
                       .dtype = info.name,
                       .dims = std::vector<std::uint64_t>(ne)});
  s.files[0].bytes = offset + (*parts)[0].bytes;
}
void AddTensor(Checkpoint& s, std::string name, std::span<const std::uint64_t> ne,
               Encoding encoding = Encoding::kF32) {
  const auto shape = format::ShapeOf(ne);
  base::Check(shape.has_value(), "test shape");
  const auto& info = format::Info(encoding);
  format::TensorFormat f{
      .encoding = encoding,
      .layout = info.quantized ? format::Layout::kGgmlBlocks : format::Layout::kDense,
      .shape = *shape};
  const auto parts = format::PartsOf(f);
  base::Check(parts.has_value(), "test format");
  const auto offset = (s.files[0].bytes + 31) / 32 * 32;
  s.tensors.push_back({.name = std::move(name),
                       .format = f,
                       .extent = {.offset = offset, .bytes = (*parts)[0].bytes},
                       .dtype = info.name,
                       .dims = std::vector<std::uint64_t>(ne.begin(), ne.end())});
  s.files[0].bytes = offset + (*parts)[0].bytes;
}

Checkpoint AssistantCheckpoint(std::uint32_t size) {
  const auto* dir = std::getenv("LLMP_TEST_DATA");
  base::Check(dir != nullptr, "missing assistant fixtures");
  std::ifstream file(std::string(dir) + "/gemma4-assistant/assistant" + std::to_string(size) +
                     ".json");
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = base::json::Parse(text);
  base::Check(doc.has_value(), "invalid assistant fixtures");
  Checkpoint s{.container = "gguf",
               .architecture = "gemma4-assistant",
               .files = {{.name = "mtp-gemma-4-" + std::to_string(size) +
                                  (size == 26 ? "B-A4B-it.gguf" : "B-it.gguf"),
                          .bytes = 0}}};
  const auto meta = test_support::gemma4::Get(doc->root(), "metadata");
  for (std::size_t i = 0; i < meta.size(); ++i) {
    const auto key = meta.key(i);
    if (key != "general.architecture" && !key.starts_with("gemma4-assistant.")) continue;
    const auto val = meta.member(i);
    if (val.is_bool()) {
      Value(s, std::string(key), val.boolean());
    } else if (val.is_number()) {
      if (val.is_integer()) {
        Value(s, std::string(key), *val.int64());
      } else {
        Value(s, std::string(key), *val.float64());
      }
    } else if (val.is_string()) {
      Value(s, std::string(key), std::string(val.string()));
    } else if (val.is_array()) {
      if (val.size() > 0 && val.at(0).is_bool()) {
        std::vector<std::int64_t> list;
        for (std::size_t j = 0; j < val.size(); ++j) list.push_back(val.at(j).boolean() ? 1 : 0);
        Value(s, std::string(key), list);
      } else if (val.size() > 0 && val.at(0).is_number() && val.at(0).is_integer()) {
        std::vector<std::int64_t> list;
        for (std::size_t j = 0; j < val.size(); ++j) list.push_back(*val.at(j).int64());
        Value(s, std::string(key), list);
      }
    }
  }
  Ordered(s);
  const auto tensors = test_support::gemma4::Get(doc->root(), "tensors");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto t = tensors.at(i);
    const std::string name = std::string(test_support::gemma4::Get(t, "name").string());
    const std::string type_name = std::string(test_support::gemma4::Get(t, "type").string());
    const auto ne = test_support::gemma4::Get(t, "ne");
    std::vector<std::uint64_t> dims;
    for (std::size_t j = 0; j < ne.size(); ++j)
      dims.push_back(static_cast<std::uint64_t>(*ne.at(j).int64()));
    Encoding enc = Encoding::kF32;
    if (type_name == "Q8_0") enc = Encoding::kQ8_0;
    AddTensor(s, name, dims, enc);
  }
  return s;
}
Checkpoint Tiny(bool gemma3 = false, bool untied = false, std::uint32_t layers = 2) {
  const std::string arch = gemma3 ? "gemma3" : "gemma2";
  Checkpoint s{
      .container = "gguf", .architecture = arch, .files = {{.name = "memory", .bytes = 0}}};
  Value(s, "general.architecture", arch);
  for (const auto& [key, value] :
       std::array<std::pair<std::string_view, std::int64_t>, 10>{{{"block_count", layers},
                                                                  {"context_length", 128},
                                                                  {"embedding_length", 8},
                                                                  {"feed_forward_length", 16},
                                                                  {"attention.head_count", 2},
                                                                  {"attention.head_count_kv", 1},
                                                                  {"attention.key_length", 4},
                                                                  {"attention.value_length", 4},
                                                                  {"attention.sliding_window", 64},
                                                                  {"rope.dimension_count", 4}}})
    Value(s, arch + "." + std::string(key), value);
  Value(s, arch + ".attention.layer_norm_rms_epsilon", 1e-6);
  if (!gemma3) {
    Value(s, arch + ".attn_logit_softcapping", 50.0);
    Value(s, arch + ".final_logit_softcapping", 30.0);
  }
  AddTensor(s, "token_embd.weight", {8, 10});
  AddTensor(s, "output_norm.weight", {8});
  if (untied) AddTensor(s, "output.weight", {8, 10});
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    const auto name = [&](std::string_view key) {
      return std::format("blk.{}.{}.weight", layer, key);
    };
    for (const auto key : {"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"})
      AddTensor(s, name(key), {8});
    AddTensor(s, name("attn_q"), {8, 8});
    AddTensor(s, name("attn_k"), {8, 4});
    AddTensor(s, name("attn_v"), {8, 4});
    AddTensor(s, name("attn_output"), {8, 8});
    AddTensor(s, name("ffn_gate"), {8, 16});
    AddTensor(s, name("ffn_up"), {8, 16});
    AddTensor(s, name("ffn_down"), {16, 8});
    if (gemma3) {
      AddTensor(s, name("attn_q_norm"), {4});
      AddTensor(s, name("attn_k_norm"), {4});
    }
  }
  return s;
}

Checkpoint TinyGemmaHf(bool gemma3, bool conditional = false, std::uint32_t layers = 2) {
  auto input = Tiny(gemma3, false, layers);
  const auto* gguf = FindSourceMapping("gguf", input.architecture);
  const std::string architecture = gemma3 ? conditional ? "gemma3" : "gemma3_text" : "gemma2";
  const auto* native = FindSourceMapping("safetensors", architecture);
  base::Check(native != nullptr, "native Gemma fixture table");
  auto values = std::move(input.metadata);
  input.metadata.clear();
  input.container = "safetensors";
  input.architecture = architecture;
  Value(input, "model_type", architecture);
  for (std::size_t i = 0; i < gguf->architecture->params.size(); ++i) {
    const auto& before = gguf->architecture->params[i];
    const auto& after = native->architecture->params[i];
    const std::string before_key = spec::CanonicalKey(*gguf->architecture, before.key);
    const std::string after_key = spec::CanonicalKey(*native->architecture, after.key);
    if (before.key.empty() || after.key.empty() || after_key.ends_with("query_pre_attn_scalar"))
      continue;
    const auto found = std::ranges::find(values, before_key, &MetaEntry::key);
    if (found != values.end()) Value(input, after_key, found->value);
  }
  const std::string prefix = conditional ? "text_config." : "";
  Value(input, prefix + "query_pre_attn_scalar", gemma3 ? 168.0 : 144.0);
  Value(input, prefix + "num_hidden_layers", std::int64_t{layers});
  Value(input, prefix + "hidden_activation", std::string("gelu_pytorch_tanh"));
  Value(input, prefix + "attention_bias", false);
  Value(input, prefix + "vocab_size", std::int64_t{10});
  if (conditional) Value(input, "text_config.model_type", std::string("gemma3_text"));
  if (gemma3) {
    Value(input, prefix + "attn_logit_softcapping", nullptr);
    Value(input, prefix + "final_logit_softcapping", nullptr);
  }
  for (auto& tensor : input.tensors) {
    const auto match = std::ranges::find_if(native->tensors, [&](const auto& rule) {
      return nd::Match(rule.canonical, tensor.name, layers).has_value();
    });
    base::Check(match != native->tensors.end(), "Gemma fixture tensor alias");
    const std::string digits(nd::Match(match->canonical, tensor.name, layers)->digits);
    tensor.name = std::string(native->architecture->tensor_prefix) + std::string(match->source);
    if (const auto at = tensor.name.find(spec::kLayerField); at != std::string::npos)
      tensor.name.replace(at, spec::kLayerField.size(), digits);
    std::ranges::reverse(tensor.dims);
    tensor.dtype = "F32";
  }
  return input;
}

TEST(ImportNormalizeTest, GemmaNativeHfUsesExplicitQueryScalarAndFoldsNormsAtImport) {
  for (const auto [gemma3, conditional, layers] :
       {std::tuple{false, false, 46U}, std::tuple{true, false, 62U}, std::tuple{true, true, 62U}}) {
    SCOPED_TRACE(std::format("gemma3={} conditional={} layers={}", gemma3, conditional, layers));
    auto input = TinyGemmaHf(gemma3, conditional, layers);
    const auto original = input.tensors;
    auto normalized = Normalize(std::move(input), kFunded);
    ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
    const auto& stage = normalized->spec().stages.front();
    EXPECT_EQ(stage.folds, std::vector<std::string>{"norm.plus_one"});
    EXPECT_EQ(stage.layers.size(), layers);
    EXPECT_DOUBLE_EQ(std::get<double>(stage.layers.front().components.front().Find("scale")->value),
                     1.0 / std::sqrt(gemma3 ? 168.0 : 144.0));
    for (const auto& layer : stage.layers)
      for (const auto& component : layer.components)
        EXPECT_EQ(component.Find("norm_zero_centered"), nullptr);
    EXPECT_EQ(stage.heads.final_norm->Find("norm_zero_centered"), nullptr);
    ASSERT_TRUE(normalized->is_indexed());
    ASSERT_EQ(normalized->parts().members().size(), original.size());
    std::vector<bool> seen(original.size(), false);
    for (const auto& member : normalized->parts().members()) {
      ASSERT_LT(member.source, original.size());
      EXPECT_FALSE(seen[member.source]);
      seen[member.source] = true;
      EXPECT_EQ(member.extent, original[member.source].extent);
      ASSERT_LT(member.output, normalized->parts().outputs().size());
      const auto& output = normalized->parts().outputs()[member.output];
      if (output.name.contains("norm")) {
        EXPECT_EQ(output.format.encoding, Encoding::kF32);
        EXPECT_TRUE(std::ranges::any_of(normalized->planned_folds(), [&](const auto& fold) {
          return fold.fold == "norm.plus_one" && std::ranges::contains(fold.outputs, member.output);
        }));
      } else {
        EXPECT_EQ(output.format, original[member.source].format);
      }
    }
  }
}

TEST(ImportNormalizeTest, GemmaNativeHfReadsNestedConfigAndBf16Safetensors) {
  for (const bool gemma3 : {false, true}) {
    SCOPED_TRACE(gemma3);
    auto source = TinyGemmaHf(gemma3, gemma3);
    std::vector<fixtures::Entry> tensors;
    for (const auto& tensor : source.tensors)
      tensors.push_back(fixtures::E(tensor.name, "BF16", tensor.dims));
    std::string text = "{";
    bool first = true;
    for (const auto& entry : source.metadata) {
      if (entry.key == "model_type" && gemma3) continue;
      if (!first) text += ',';
      first = false;
      json::AppendQuoted(gemma3 ? std::string_view(entry.key).substr(12) : entry.key, text);
      text += ':';
      if (const auto* string_value = std::get_if<std::string>(&entry.value))
        json::AppendQuoted(*string_value, text);
      else if (const auto* integer_value = std::get_if<std::int64_t>(&entry.value))
        text += std::to_string(*integer_value);
      else if (const auto* real_value = std::get_if<double>(&entry.value))
        text += std::format("{}", *real_value);
      else if (const auto* boolean_value = std::get_if<bool>(&entry.value))
        text += *boolean_value ? "true" : "false";
      else
        text += "null";
    }
    text += '}';
    if (gemma3) text = R"({"model_type":"gemma3","text_config":)" + text + '}';
    auto checkpoint = ParseSafetensors(fixtures::Shard(tensors), text);
    ASSERT_TRUE(checkpoint) << test_support::Failed(checkpoint, &Error::ToString).value_or("");
    auto normalized = Normalize(std::move(*checkpoint), kFunded);
    ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
    for (const auto& tensor : normalized->canonical().tensors) {
      EXPECT_EQ(tensor.format.encoding,
                tensor.name.contains("norm") ? Encoding::kF32 : Encoding::kBF16);
    }
    EXPECT_EQ(normalized->spec().stages.front().layers.size(), 2);
  }
}

TEST(ImportNormalizeTest, GemmaGgufExplicitScaleUnlocksAmbiguousLegacyDepths) {
  for (const bool gemma3 : {false, true}) {
    auto input = Tiny(gemma3, false, gemma3 ? 62 : 46);
    EXPECT_FALSE(Normalize(input, kFunded));
    Value(input, input.architecture + ".attention.scale", 0.071234);
    auto normalized = Normalize(std::move(input), kFunded);
    ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
    EXPECT_DOUBLE_EQ(std::get<double>(normalized->spec()
                                          .stages.front()
                                          .layers.front()
                                          .components.front()
                                          .Find("scale")
                                          ->value),
                     0.071234);
  }
}

TEST(ImportNormalizeTest, GemmaNativeHfRefusesMissingOrUnsupportedMath) {
  for (const bool gemma3 : {false, true}) {
    const std::string prefix = gemma3 ? "text_config." : "";
    auto valid = TinyGemmaHf(gemma3, gemma3);
    const auto baseline = Normalize(valid, kFunded);
    ASSERT_TRUE(baseline) << test_support::Failed(baseline, &Error::ToString).value_or("");
    for (const MetaValue& value :
         {MetaValue{std::int64_t{0}}, MetaValue{-1.0}, MetaValue{true},
          MetaValue{std::numeric_limits<double>::infinity()}, MetaValue{nullptr}}) {
      auto input = valid;
      Value(input, prefix + "query_pre_attn_scalar", value);
      EXPECT_FALSE(Normalize(std::move(input), kFunded));
    }
    auto input = valid;
    std::erase_if(input.metadata,
                  [&](const auto& value) { return value.key == prefix + "query_pre_attn_scalar"; });
    EXPECT_FALSE(Normalize(std::move(input), kFunded));
    for (const auto& [key, value] : std::vector<std::pair<std::string, MetaValue>>{
             {"attention_bias", true},
             {"hidden_activation", std::string("silu")},
             {"rope_scaling.rope_type", std::string("yarn")},
             {"use_bidirectional_attention", true}}) {
      input = valid;
      Value(input, prefix + key, value);
      EXPECT_FALSE(Normalize(std::move(input), kFunded));
    }
  }
}

TEST(ImportNormalizeTest, GemmaNativeTyingFlagCannotInventAnUntiedHead) {
  for (const bool gemma3 : {false, true}) {
    auto input = TinyGemmaHf(gemma3, gemma3);
    const std::string key = gemma3 ? "text_config.tie_word_embeddings" : "tie_word_embeddings";
    Value(input, key, false);
    EXPECT_FALSE(Normalize(input, kFunded));
    Value(input, key, true);
    auto tied = Normalize(input, kFunded);
    ASSERT_TRUE(tied) << test_support::Failed(tied, &Error::ToString).value_or("");
    if (gemma3) {
      Value(input, "tie_word_embeddings", false);
      EXPECT_FALSE(Normalize(input, kFunded));
    }
    Value(input, key, false);
    AddTensor(input, gemma3 ? "language_model.lm_head.weight" : "lm_head.weight", {8, 10});
    input.tensors.back().dtype = "F32";
    std::ranges::reverse(input.tensors.back().dims);
    auto untied = Normalize(std::move(input), kFunded);
    ASSERT_TRUE(untied) << test_support::Failed(untied, &Error::ToString).value_or("");
  }
}

TEST(ImportNormalizeTest, GemmaConditionalTextImportExplicitlyExcludesVisionComponents) {
  auto input = TinyGemmaHf(true, true);
  const auto count = input.tensors.size();
  AddTensor(input, "vision_tower.vision_model.embeddings.patch_embedding.weight", {2, 2});
  AddTensor(input, "multi_modal_projector.mm_input_projection_weight", {2, 2});
  for (std::size_t i = count; i < input.tensors.size(); ++i)
    std::ranges::reverse(input.tensors[i].dims);
  Value(input, "vision_config.hidden_size", std::int64_t{2});
  auto normalized = Normalize(input, kFunded);
  ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
  ASSERT_TRUE(normalized->is_indexed());
  EXPECT_EQ(normalized->parts().members().size(), count);
  ASSERT_EQ(normalized->parts().dispositions().size(), count + 2);
  for (std::size_t i = count; i < count + 2; ++i) {
    EXPECT_EQ(normalized->parts().dispositions()[i].kind, SourceDispositionKind::kAuxiliary);
    EXPECT_FALSE(std::ranges::contains(normalized->parts().members(), i, &IndexedMember::source));
  }
  Value(input, "text_config.model_type", std::string("wrong"));
  EXPECT_FALSE(Normalize(std::move(input), kFunded));
}

Checkpoint TinyGemma4(bool moe) {
  Checkpoint s{
      .container = "gguf", .architecture = "gemma4", .files = {{.name = "memory", .bytes = 0}}};
  Value(s, "general.architecture", std::string("gemma4"));
  for (const auto& [key, value] :
       std::array<std::pair<std::string_view, std::int64_t>, 13>{{{"block_count", 6},
                                                                  {"context_length", 128},
                                                                  {"embedding_length", 8},
                                                                  {"feed_forward_length", 16},
                                                                  {"attention.head_count", 2},
                                                                  {"attention.head_count_kv", 1},
                                                                  {"attention.key_length", 4},
                                                                  {"attention.value_length", 4},
                                                                  {"attention.key_length_swa", 4},
                                                                  {"attention.value_length_swa", 4},
                                                                  {"attention.sliding_window", 64},
                                                                  {"rope.dimension_count", 4},
                                                                  {"rope.dimension_count_swa", 4}}})
    Value(s, "gemma4." + std::string(key), value);
  Value(s, "gemma4.attention.layer_norm_rms_epsilon", 1e-6);
  Value(s, "gemma4.rope.freq_base", 1000000.0);
  Value(s, "gemma4.rope.freq_base_swa", 10000.0);
  Value(s, "gemma4.attention.sliding_window_pattern", std::vector<std::int64_t>{1, 1, 1, 1, 1, 0});
  Value(s, "gemma4.attention.head_count_kv", std::vector<std::int64_t>(6, 1));
  if (moe) {
    Value(s, "gemma4.expert_count", std::int64_t{4});
    Value(s, "gemma4.expert_used_count", std::int64_t{2});
    Value(s, "gemma4.expert_feed_forward_length", std::int64_t{8});
  }
  AddTensor(s, "token_embd.weight", {8, 10});
  AddTensor(s, "output_norm.weight", {8});
  AddTensor(s, "rope_freqs.weight", {2});
  for (int layer = 0; layer < 6; ++layer) {
    const auto name = [&](std::string_view key) { return std::format("blk.{}.{}", layer, key); };
    for (const auto key : {"attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
                           "post_ffw_norm.weight"})
      AddTensor(s, name(key), {8});
    for (const auto key : {"attn_q_norm.weight", "attn_k_norm.weight"})
      AddTensor(s, name(key), {4});
    AddTensor(s, name("attn_q.weight"), {8, 8});
    AddTensor(s, name("attn_k.weight"), {8, 4});
    if (layer != 5) AddTensor(s, name("attn_v.weight"), {8, 4});
    AddTensor(s, name("attn_output.weight"), {8, 8});
    AddTensor(s, name("ffn_gate.weight"), {8, 16});
    AddTensor(s, name("ffn_up.weight"), {8, 16});
    AddTensor(s, name("ffn_down.weight"), {16, 8});
    AddTensor(s, name("layer_output_scale.weight"), {1});
    if (moe) {
      for (const auto key : {"post_ffw_norm_1.weight", "pre_ffw_norm_2.weight",
                             "post_ffw_norm_2.weight", "ffn_gate_inp.scale"})
        AddTensor(s, name(key), {8});
      AddTensor(s, name("ffn_gate_inp.weight"), {8, 4});
      AddTensor(s, name("ffn_gate_up_exps.weight"), {8, 16, 4});
      s.tensors.back().experts = true;
      AddTensor(s, name("ffn_down_exps.weight"), {8, 8, 4});
      s.tensors.back().experts = true;
      AddTensor(s, name("ffn_down_exps.scale"), {4});
    }
  }
  return s;
}

// A real, bounded GGUF serialization for the copy-byte test, not a model
// payload. Selected lists use GGUF integer arrays; every tensor is F32.
std::vector<std::byte> Gguf(const Checkpoint& s) {
  std::vector<std::byte> file;
  const auto add = [&](std::uint64_t n, std::size_t bytes) { fixtures::Append(file, n, bytes); };
  const auto string = [&](std::string_view text) {
    add(text.size(), 8);
    fixtures::AppendText(file, text);
  };
  add(0x46554747, 4);
  add(3, 4);
  add(s.tensors.size(), 8);
  add(s.metadata.size(), 8);
  for (const auto& e : s.metadata) {
    string(e.key);
    if (const auto* integer = std::get_if<std::int64_t>(&e.value)) {
      add(11, 4);
      add(std::bit_cast<std::uint64_t>(*integer), 8);
    } else if (const auto* real = std::get_if<double>(&e.value)) {
      add(12, 4);
      add(std::bit_cast<std::uint64_t>(*real), 8);
    } else if (const auto* list = std::get_if<std::vector<std::int64_t>>(&e.value)) {
      add(9, 4);
      add(11, 4);
      add(list->size(), 8);
      for (const auto item : *list) add(std::bit_cast<std::uint64_t>(item), 8);
    } else {
      add(8, 4);
      string(std::get<std::string>(e.value));
    }
  }
  for (const auto& t : s.tensors) {
    string(t.name);
    add(t.dims.size(), 4);
    for (auto d : t.dims) add(d, 8);
    add(format::Info(t.format.encoding).ggml->type, 4);
    add(t.extent.offset, 8);
  }
  file.resize((file.size() + 31) / 32 * 32);
  const auto begin = file.size();
  base::Check(s.files[0].bytes < (1U << 20U), "test payload ceiling");
  file.resize(begin + static_cast<std::size_t>(s.files[0].bytes));
  for (std::size_t i = begin; i < file.size(); ++i) file[i] = static_cast<std::byte>(i % 251);
  return file;
}

TEST(ImportNormalizeTest, CompleteOwnedPlansAndTiedHeadDoNotInventPayloads) {
  for (const bool gemma3 : {false, true})
    for (const bool untied : {false, true}) {
      auto input = Tiny(gemma3, untied);
      Value(input, "tokenizer.ggml.tokens", std::vector<std::string>{"a", "b"});
      const auto expected =
          spec::Synthesize(*FindSourceMapping(input.container, input.architecture)->architecture,
                           SourceOf(input), ValuesOf(input));
      ASSERT_TRUE(expected) << test_support::Failed(expected).value_or("");
      const auto count = input.tensors.size();
      auto plan = test_support::NormalizeWithFixtureTarget(std::move(input), kFunded);
      ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
      EXPECT_EQ(plan->spec(), *expected);
      EXPECT_EQ(plan->routes().size(), count);
      EXPECT_EQ(plan->metadata().size(), plan->source().metadata.size());
      EXPECT_EQ(plan->spec().stages[0].heads.output->weights[0].tensor,
                untied ? "output.weight" : "token_embd.weight");
      ASSERT_EQ(plan->applied_folds().size(), 1U);
      EXPECT_EQ(plan->applied_folds()[0].fold, "norm.plus_one");
      EXPECT_EQ(plan->applied_folds()[0].container, "gguf");
      EXPECT_EQ(plan->spec().stages[0].folds, std::vector<std::string>{"norm.plus_one"});
      EXPECT_EQ(plan->normalization_version(), 1U);
      EXPECT_EQ(plan->architecture_version(), 1U);
      auto moved = std::move(*plan);
      for (const auto& t : moved.source().tensors)
        EXPECT_EQ(t.dtype.data(), format::Info(t.format.encoding).name.data());
    }
}

TEST(ImportNormalizeTest, RealGgufCopyCursorsCoverEveryOriginalByteWithoutRefolding) {
  const auto file = Gguf(Tiny());
  auto parsed = ParseGguf(file);
  ASSERT_TRUE(parsed) << test_support::Failed(parsed, &Error::ToString).value_or("");
  auto plan = test_support::NormalizeWithFixtureTarget(std::move(*parsed), kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < plan->routes().size(); ++i) {
    const auto& r = plan->routes()[i];
    auto cursor = plan->CursorFor(i);
    ASSERT_TRUE(cursor);
    EXPECT_FALSE(cursor->Next(0));
    std::vector<std::byte> copied(static_cast<std::size_t>(r.extent.bytes));
    std::uint64_t at = 0;
    while (true) {
      const auto next = cursor->Next(13);
      ASSERT_TRUE(next);
      if (!*next) break;
      const auto& c = **next;
      EXPECT_EQ(c.file, 0U);
      EXPECT_EQ(c.output_offset, at);
      EXPECT_EQ(c.source_offset, r.extent.offset + at);
      std::copy_n(file.data() + c.source_offset, static_cast<std::size_t>(c.bytes),
                  copied.data() + at);
      at += c.bytes;
    }
    EXPECT_EQ(at, r.extent.bytes);
    total += at;
    EXPECT_TRUE(std::ranges::equal(
        copied, std::span(file).subspan(static_cast<std::size_t>(r.extent.offset),
                                        static_cast<std::size_t>(r.extent.bytes))));
    EXPECT_FALSE(plan->source().Read(r.extent, 0, std::span(copied).first(1)));
  }
  EXPECT_EQ(total, plan->cost().payload_bytes);
  EXPECT_FALSE(plan->CursorFor(plan->routes().size()));
}

TEST(ImportNormalizeTest, AliasesHaveOneDestinationAndTableLifetimeIsNotBorrowed) {
  auto input = Tiny();
  input.tensors[0].name = "words.weight";
  const std::array aliases{TensorAlias{"words.weight", "token_embd.weight"}};
  const std::array keys{KeyAlias{"config.width", "gemma2.embedding_length"}};
  auto table = *FindSourceMapping("gguf", "gemma2");
  table.tensors = aliases;
  table.keys = keys;
  auto width = std::ranges::find(input.metadata, "gemma2.embedding_length", &MetaEntry::key);
  ASSERT_NE(width, input.metadata.end());
  width->key = "config.width";
  Ordered(input);
  ASSERT_TRUE(ValidateSourceMapping(table));
  auto plan = nd::Build(std::move(input), table, kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  EXPECT_EQ(plan->canonical().tensors[0].name, "token_embd.weight");
  EXPECT_EQ(plan->spec().stages[0].stream.width, 8U);
  auto expired = [] {
    auto source = Tiny();
    source.tensors[0].name = "words.weight";
    const std::string original = "words.weight", canonical = "token_embd.weight";
    const std::string prefix = "custom.", reason = "local compiled annotation";
    const std::array local_alias{TensorAlias{original, canonical}};
    const std::array local_aux{AuxiliaryNamespace{prefix, reason}};
    auto local_table = *FindSourceMapping("gguf", "gemma2");
    local_table.tensors = local_alias;
    local_table.auxiliary = local_aux;
    Value(source, "custom.origin", std::string("kept"));
    return nd::Build(std::move(source), local_table, kFunded);
  }();
  ASSERT_TRUE(expired) << test_support::Failed(expired, &Error::ToString).value_or("");
  EXPECT_EQ(expired->canonical().tensors[0].name, "token_embd.weight");
  ASSERT_EQ(expired->auxiliary_namespaces().size(), 2U);
  EXPECT_EQ(expired->auxiliary_namespaces()[0].reason, "local compiled annotation");
  auto duplicate = Tiny();
  AddTensor(duplicate, "words.weight", {8, 10});
  EXPECT_FALSE(nd::Build(std::move(duplicate), table, kFunded));
  auto key_duplicate = Tiny();
  Value(key_duplicate, "config.width", std::int64_t{8});
  EXPECT_FALSE(nd::Build(std::move(key_duplicate), table, kFunded));
}

TEST(ImportNormalizeTest, InvalidTablesCannotBroadenInputAuthority) {
  auto table = *FindSourceMapping("gguf", "gemma2");
  const std::array malformed{TensorAlias{"blk.{layer}.{layer}", "token_embd.weight"}};
  table.tensors = malformed;
  EXPECT_FALSE(ValidateSourceMapping(table));
  const std::array overlap{TensorAlias{"blk.{layer}.attn_q.weight", "blk.{layer}.attn_k.weight"}};
  table.tensors = overlap;
  EXPECT_FALSE(ValidateSourceMapping(table));
  table.tensors = {};
  const std::array aux{AuxiliaryNamespace{"gemma2.", "must not swallow model keys"}};
  table.auxiliary = aux;
  EXPECT_FALSE(ValidateSourceMapping(table));
  table = *FindSourceMapping("gguf", "gemma2");
  table.route = static_cast<RouteOperation>(255);
  EXPECT_FALSE(ValidateSourceMapping(table));
  table = *FindSourceMapping("gguf", "gemma2");
  table.requirements = {};
  EXPECT_FALSE(ValidateSourceMapping(table));
}

TEST(ImportNormalizeTest, PresentBadOptionalKeysNeverBecomeDefaults) {
  for (const MetaValue& value : std::vector<MetaValue>{
           true, std::string("wrong"), std::vector<std::int64_t>{4},
           std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    auto source = Tiny();
    Value(source, "gemma2.rope.freq_base", value);
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(source), kFunded));
  }
  auto source = Tiny();
  Value(source, "gemma2.rope.freq_base", std::int64_t{10000});
  Value(source, "gemma2.rope.freq_base_swa", 1000.0);
  EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(std::move(source),
                                                       kFunded));  // two distinct fallback keys
}

TEST(ImportNormalizeTest, UnknownMissingDuplicateAndNoncanonicalNamesRefuse) {
  for (const auto mode : {0, 1, 2, 3, 4, 5}) {
    auto s = Tiny();
    if (mode == 0) s.tensors.pop_back();
    if (mode == 1) s.tensors[2].name = "blk.00.attn_norm.weight";
    if (mode == 2) s.tensors[2].name = "blk.2.attn_norm.weight";
    if (mode == 3) s.tensors[2].name = s.tensors[3].name;
    if (mode == 4) s.tensors[2].name = "vision.patch.weight";
    if (mode == 5) Value(s, "gemma2.unhandled_policy", std::int64_t{1});
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded)) << mode;
  }
  auto s = Tiny();
  Value(s, "vocab_count", std::int64_t{10});
  EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
}

TEST(ImportNormalizeTest, RefusesUnimplementedDomainsAndAlternateQueryScale) {
  for (const auto container : {"safetensors", "mlx", "exl3", "diffusers"}) {
    auto s = Tiny();
    s.container = container;
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  }
  for (const bool g3 : {false, true}) {
    auto s = Tiny(g3);
    Value(s, s.architecture + ".block_count", std::int64_t{g3 ? 62 : 46});
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  }
  for (const int mode : {0, 1, 2, 3}) {
    auto s = Tiny();
    if (mode == 0) s.role = "target";
    if (mode == 1) s.quantization = "invented";
    if (mode == 2) s.components.emplace_back();
    if (mode == 3) s.skipped.push_back({"file", "outside"});
    if (mode == 3)
      EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
    else
      EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  }
}

TEST(ImportNormalizeTest, GeometryAndPhysicalAliasesCannotHidePayloadOverlap) {
  for (const int mode : {0, 1, 2, 3, 4, 5, 6}) {
    auto s = Tiny();
    auto& t = s.tensors[1];
    if (mode == 0) ++t.extent.bytes;
    if (mode == 1) t.extent.offset = UINT64_MAX;
    if (mode == 2) t.extent.file = 1;
    if (mode == 3) t.extent.offset = s.tensors[0].extent.offset;
    if (mode == 4) t.dtype = "bf16";
    if (mode == 5) t.dims[0] = 9;
    if (mode == 6) t.experts = true;
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded)) << mode;
  }
  auto s = Tiny();
  s.files[0].path = "/inventoried/a";
  s.files[0].device = 1;
  s.files[0].inode = 3;
  s.files.push_back(s.files[0]);
  s.files[1].path = "/inventoried/alias";
  s.files[1].name = "alias";
  Value(s, "split.no", std::int64_t{0});
  Value(s, "split.count", std::int64_t{2});
  s.tensors[1].extent.file = 1;
  s.tensors[1].extent.offset = 0;
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(s, kFunded));
  s.files[1].inode = 4;
  EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(
      std::move(s), kFunded));  // distinct physical file, same offset
}

TEST(ImportNormalizeTest, FormatAndComponentChecksGoBeyondNameCoverage) {
  auto s = Tiny();
  auto& last = s.tensors.back();
  ++last.format.shape.ne[1];
  ++last.dims[1];
  last.extent.bytes = (*format::PartsOf(last.format))[0].bytes;
  s.files[0].bytes = last.extent.offset + last.extent.bytes;
  ASSERT_TRUE(NormalizationBound(s));  // geometry/name coverage alone is insufficient
  const auto malformed = test_support::NormalizeWithFixtureTarget(std::move(s), kFunded);
  ASSERT_FALSE(malformed);
  EXPECT_TRUE(
      malformed.error().reason.starts_with("normalized tensors do not satisfy their components"));
  s = Tiny();
  Value(s, "gemma2.attention.head_count", std::int64_t{3});
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  s = Tiny();
  Value(s, "gemma2.attention.sliding_window_pattern", std::int64_t{3});
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  s = Tiny(true);
  Value(s, "gemma3.rope.scaling.type", std::string("yarn"));
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
}

TEST(ImportNormalizeTest, ExactReservationAndWorkLimitAreCheckedBeforeConstruction) {
  auto s = Tiny();
  const auto cost = NormalizationBound(s);
  ASSERT_TRUE(cost);
  EXPECT_FALSE(
      test_support::NormalizeWithFixtureTarget(s, {cost->metadata_bytes - 1, cost->work_units}));
  EXPECT_FALSE(
      test_support::NormalizeWithFixtureTarget(s, {cost->metadata_bytes, cost->work_units - 1}));
  EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(std::move(s),
                                                       {cost->metadata_bytes, cost->work_units}));
  s = Tiny();
  Value(s, "gemma2.block_count", std::int64_t{1025});
  EXPECT_FALSE(NormalizationBound(s));
  s = Tiny();
  Value(s, "gemma2.embedding_length", std::int64_t{65537});
  EXPECT_FALSE(NormalizationBound(s));
}

TEST(ImportNormalizeTest, StructuralKeysAndAuxiliaryIndicesStayDistinct) {
  for (const auto key : {"general.architecture", "general.alignment", "split.no", "split.count",
                         "split.tensors.count"}) {
    auto s = Tiny();
    Value(s, key, true);
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  }
  auto s = Tiny();
  Value(s, "general.alignment", std::int64_t{3});
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  s = Tiny();
  Value(s, "general.description", std::string("unaltered provenance"));
  Value(s, "tokenizer.ggml.tokens", std::vector<std::string>{"a", "b"});
  auto p = test_support::NormalizeWithFixtureTarget(std::move(s), kFunded);
  ASSERT_TRUE(p);
  std::size_t auxiliary = 0;
  for (const auto& use : p->metadata())
    if (use.classification == MetadataClass::kAuxiliary) {
      ++auxiliary;
      ASSERT_LT(use.destination, p->auxiliary_namespaces().size());
      EXPECT_TRUE(p->source().metadata[use.source].key.starts_with(
          p->auxiliary_namespaces()[use.destination].prefix));
    }
  EXPECT_EQ(auxiliary, 2U);
}

TEST(ImportNormalizeTest, QuantizationProvenanceIsExplicitAuxiliaryMetadata) {
  auto s = Tiny();
  Value(s, "quantize.imatrix.chunks_count", std::int64_t{128});
  Value(s, "quantize.imatrix.dataset", std::string("/training_dir/calibration_datav3.txt"));
  Value(s, "quantize.imatrix.entries_count", std::int64_t{182});
  Value(s, "quantize.imatrix.file", std::string("/models_out/gemma.imatrix"));
  auto plan = test_support::NormalizeWithFixtureTarget(s, kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  std::size_t provenance = 0;
  for (const auto& use : plan->metadata()) {
    if (!plan->source().metadata[use.source].key.starts_with("quantize.imatrix.")) continue;
    ++provenance;
    EXPECT_EQ(use.classification, MetadataClass::kAuxiliary);
    ASSERT_LT(use.destination, plan->auxiliary_namespaces().size());
    EXPECT_EQ(plan->auxiliary_namespaces()[use.destination].reason,
              "source quantization calibration provenance");
  }
  EXPECT_EQ(provenance, 4U);
  Value(s, "quantize.other", std::int64_t{1});
  EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
}

TEST(ImportNormalizeTest, Gemma4DenseAndExpertPlansPreserveEverySourcePart) {
  for (const bool moe : {false, true}) {
    auto s = TinyGemma4(moe);
    auto expected = spec::Synthesize(*FindSourceMapping("gguf", "gemma4")->architecture,
                                     SourceOf(s), ValuesOf(s));
    ASSERT_TRUE(expected) << test_support::Failed(expected).value_or("");
    const auto bytes = Gguf(s);
    auto input = ParseGguf(bytes);
    ASSERT_TRUE(input) << test_support::Failed(input, &Error::ToString).value_or("");
    const auto bound = NormalizationBound(*input);
    ASSERT_TRUE(bound) << test_support::Failed(bound, &Error::ToString).value_or("");
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(
        *input, {bound->metadata_bytes - 1, bound->work_units}));
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(
        *input, {bound->metadata_bytes, bound->work_units - 1}));
    auto plan = test_support::NormalizeWithFixtureTarget(
        std::move(*input), {bound->metadata_bytes, bound->work_units});
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    EXPECT_EQ(plan->spec(), *expected);
    EXPECT_TRUE(plan->applied_folds().empty());
    EXPECT_TRUE(plan->spec().stages[0].folds.empty());
    EXPECT_EQ(plan->spec().stages[0].heads.output->weights[0].tensor, "token_embd.weight");
    ASSERT_EQ(plan->routes().size(), s.tensors.size());
    std::size_t experts = 0, rope = 0;
    for (std::size_t i = 0; i < plan->routes().size(); ++i) {
      const auto& t = plan->canonical().tensors[i];
      EXPECT_EQ(t.name, s.tensors[i].name);
      EXPECT_EQ(t.format, s.tensors[i].format);
      EXPECT_EQ(t.experts, s.tensors[i].experts);
      experts += t.experts;
      rope += t.name == "rope_freqs.weight";
      auto cursor = plan->CursorFor(i);
      ASSERT_TRUE(cursor);
      std::vector<std::byte> output(static_cast<std::size_t>(plan->routes()[i].extent.bytes));
      std::uint64_t copied = 0;
      while (true) {
        const auto part = cursor->Next(37);
        ASSERT_TRUE(part);
        if (!*part) break;
        const auto& copy = **part;
        EXPECT_EQ(copy.output_offset, copied);
        const auto source_offset = plan->routes()[i].extent.offset + copied;
        EXPECT_EQ(copy.source_offset, source_offset);
        std::copy_n(bytes.data() + copy.source_offset, static_cast<std::size_t>(copy.bytes),
                    output.data() + copy.output_offset);
        copied += copy.bytes;
      }
      EXPECT_EQ(copied, s.tensors[i].extent.bytes);
      EXPECT_TRUE(
          std::equal(output.begin(), output.end(),
                     bytes.begin() + static_cast<std::ptrdiff_t>(plan->routes()[i].extent.offset)));
    }
    EXPECT_EQ(experts, moe ? 12U : 0U);
    EXPECT_EQ(rope, 1U);
    for (const auto& layer : plan->spec().stages[0].layers) {
      ASSERT_EQ(layer.components.size(), 3U);
      EXPECT_EQ(layer.components[1].kind, moe ? "ffn.moe" : "ffn.glu");
    }
  }
}

TEST(ImportNormalizeTest, Gemma4LayerListsGatesAndExpertGeometryRefuseAmbiguity) {
  for (const int mode : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) {
    auto s = TinyGemma4(true);
    if (mode == 0) Value(s, "gemma4.attention.head_count_kv", std::vector<std::int64_t>(5, 1));
    if (mode == 1) Value(s, "gemma4.attention.head_count_kv", std::vector<double>(6, 1.0));
    if (mode == 2)
      Value(s, "gemma4.attention.sliding_window_pattern", std::vector<std::int64_t>(6, 1));
    if (mode == 3) Value(s, "gemma4.expert_count", true);
    if (mode == 4) Value(s, "gemma4.expert_count", std::int64_t{0});
    if (mode == 5) Value(s, "gemma4.expert_count", std::int64_t{-1});
    if (mode == 6) AddTensor(s, "blk.5.attn_v.weight", {8, 4});
    if (mode == 7) Value(s, "gemma4.attention.shared_kv_layers", std::int64_t{1});
    if (mode == 8) Value(s, "gemma4.embedding_length_per_layer_input", std::int64_t{1});
    auto expert = std::ranges::find(s.tensors, "blk.0.ffn_gate_up_exps.weight", &Tensor::name);
    ASSERT_NE(expert, s.tensors.end());
    if (mode == 9) expert->experts = false;
    if (mode == 10) expert->name = "blk.0.ffn_gate_exps.weight";
    if (mode == 11) {
      expert->format.shape.ne[2] = 3;
      expert->dims[2] = 3;
      expert->extent.bytes = (*format::PartsOf(expert->format))[0].bytes;
    }
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded)) << mode;
  }
  auto s = TinyGemma4(false);
  s.architecture = "gemma4-assistant";
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  s = TinyGemma4(false);
  Value(s, "gemma4.feed_forward_length", std::vector<std::int64_t>(6, 16));
  EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(std::move(s), kFunded));
  const auto table = *FindSourceMapping("gguf", "gemma4");
  const std::array aliases{TensorAlias{"dense.{layer}.weight", "blk.{layer}.ffn_gate.weight"}};
  auto aliased = table;
  aliased.tensors = aliases;
  ASSERT_TRUE(ValidateSourceMapping(aliased));
  s = TinyGemma4(false);
  auto gate = std::ranges::find(s.tensors, "blk.0.ffn_gate.weight", &Tensor::name);
  ASSERT_NE(gate, s.tensors.end());
  gate->name = "dense.0.weight";
  EXPECT_TRUE(nd::Build(s, aliased, kFunded));
  AddTensor(s, "blk.0.ffn_gate.weight", {8, 16});
  EXPECT_FALSE(nd::Build(std::move(s), aliased, kFunded));
}

TEST(ImportNormalizeTest, Gemma4CompleteDescriptorInventoriesUseTheSameMapping) {
  for (const std::uint32_t size : {26U, 31U}) {
    const auto doc = test_support::gemma4::Fixture(size);
    const auto get = [](json::Value v, std::string_view key) { return *v.find(key); };
    // The checked fixture contains actual names, shapes and encodings but no
    // source offsets. Give those descriptors disjoint synthetic extents; the
    // models test separately authenticates actual file extents and v0 parity.
    Checkpoint s{
        .container = "gguf", .architecture = "gemma4", .files = {{.name = "memory", .bytes = 0}}};
    Value(s, "general.architecture", std::string("gemma4"));
    const auto metadata = get(doc.root(), "metadata");
    for (std::size_t i = 0; i < metadata.size(); ++i) {
      const auto key = metadata.key(i);
      const auto value = metadata.member(i);
      if (key == "vocab_count") continue;
      if (value.is_array()) {
        std::vector<std::int64_t> list;
        for (std::size_t j = 0; j < value.size(); ++j) {
          const auto item = value.at(j);
          list.push_back(item.is_bool() ? (item.boolean() ? 1 : 0) : *item.int64());
        }
        Value(s, std::string(key), std::move(list));
      } else if (value.is_integer())
        Value(s, std::string(key), *value.int64());
      else
        Value(s, std::string(key), *value.float64());
    }
    const auto tensors = get(doc.root(), "tensors");
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto t = tensors.at(i);
      const auto ne = get(t, "ne");
      std::vector<std::uint64_t> dims;
      for (std::size_t d = 0; d < ne.size(); ++d)
        dims.push_back(static_cast<std::uint64_t>(*ne.at(d).int64()));
      const auto type = static_cast<std::uint32_t>(*get(t, "ggml_type").int64());
      const auto encoding = std::ranges::find_if(
          format::Encodings(), [&](const auto& e) { return e.ggml && e.ggml->type == type; });
      ASSERT_NE(encoding, format::Encodings().end());
      format::TensorFormat f{
          .encoding = encoding->id,
          .layout = encoding->quantized ? format::Layout::kGgmlBlocks : format::Layout::kDense,
          .shape = *format::ShapeOf(dims)};
      const auto parts = format::PartsOf(f);
      ASSERT_TRUE(parts);
      const auto offset = (s.files[0].bytes + 31) / 32 * 32;
      s.tensors.push_back({.name = std::string(get(t, "name").string()),
                           .format = f,
                           .experts = ne.size() == 3,
                           .extent = {.offset = offset, .bytes = (*parts)[0].bytes},
                           .dtype = encoding->name,
                           .dims = std::move(dims)});
      s.files[0].bytes = offset + (*parts)[0].bytes;
    }
    const auto expected = spec::Synthesize(*FindSourceMapping("gguf", "gemma4")->architecture,
                                           SourceOf(s), ValuesOf(s));
    ASSERT_TRUE(expected) << test_support::Failed(expected).value_or("");
    auto p = test_support::NormalizeWithFixtureTarget(std::move(s), kFunded);
    ASSERT_TRUE(p) << test_support::Failed(p, &Error::ToString).value_or("");
    EXPECT_EQ(p->spec(), *expected);
    EXPECT_EQ(p->routes().size(), size == 26 ? 658U : 833U);
    EXPECT_TRUE(p->applied_folds().empty());
  }
}

TEST(ImportNormalizeTest, ApprovedCompleteDescriptorInventoriesMatchExistingSpecs) {
  for (const bool g3 : {false, true}) {
    const auto doc = g3 ? test_support::gemma3::Fixture() : test_support::gemma2::Fixture();
    const auto get = [](json::Value v, std::string_view key) { return *v.find(key); };
    Checkpoint s{.container = "gguf",
                 .architecture = g3 ? "gemma3" : "gemma2",
                 .files = {{.name = "memory",
                            .bytes = static_cast<std::uint64_t>(
                                *get(get(doc.root(), "source"), "full_bytes").int64())}}};
    const auto metadata = get(doc.root(), "metadata");
    for (std::size_t i = 0; i < metadata.size(); ++i) {
      const auto key = metadata.key(i);
      const auto value = metadata.member(i);
      if (key == "vocab_count") continue;  // fixture summary, not raw GGUF metadata
      if (value.is_string())
        Value(s, std::string(key), std::string(value.string()));
      else if (value.is_integer())
        Value(s, std::string(key), *value.int64());
      else
        Value(s, std::string(key), *value.float64());
    }
    const auto tensors = get(doc.root(), "tensors");
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto t = tensors.at(i);
      const auto ne = get(t, "ne");
      std::vector<std::uint64_t> dims;
      for (std::size_t d = 0; d < ne.size(); ++d)
        dims.push_back(static_cast<std::uint64_t>(*ne.at(d).int64()));
      const auto type = static_cast<std::uint32_t>(*get(t, "ggml_type").int64());
      const auto encoding = std::ranges::find_if(
          format::Encodings(), [&](const auto& e) { return e.ggml && e.ggml->type == type; });
      ASSERT_NE(encoding, format::Encodings().end());
      s.tensors.push_back(
          {.name = std::string(get(t, "name").string()),
           .format = {.encoding = encoding->id,
                      .layout = encoding->quantized ? format::Layout::kGgmlBlocks
                                                    : format::Layout::kDense,
                      .shape = *format::ShapeOf(dims)},
           .extent = {.offset =
                          static_cast<std::uint64_t>(*get(doc.root(), "data_offset").int64()) +
                          static_cast<std::uint64_t>(*get(t, "relative_offset").int64()),
                      .bytes = static_cast<std::uint64_t>(*get(t, "bytes").int64())},
           .dtype = encoding->name,
           .dims = std::move(dims)});
    }
    const auto old_source = SourceOf(s);
    const auto* table = FindSourceMapping(s.container, s.architecture);
    const auto expected = spec::Synthesize(*table->architecture, old_source, ValuesOf(s));
    ASSERT_TRUE(expected) << test_support::Failed(expected).value_or("");
    auto p = test_support::NormalizeWithFixtureTarget(std::move(s), kFunded);
    ASSERT_TRUE(p) << test_support::Failed(p, &Error::ToString).value_or("");
    EXPECT_EQ(p->spec(), *expected);
    ASSERT_EQ(p->routes().size(), g3 ? 444U : 288U);
    for (std::size_t i = 0; i < p->routes().size(); ++i) {
      EXPECT_EQ(p->canonical().tensors[i].name, old_source.tensors[i].name);
      EXPECT_EQ(p->canonical().tensors[i].format, old_source.tensors[i].format);
      EXPECT_EQ(p->routes()[i].extent, p->source().tensors[i].extent);
    }
  }
}
TEST(ImportNormalizeQwenTest, OwnsCompleteEncodedTargetAndAllSemanticLists) {
  auto input = test_support::Qwen38GgufCheckpoint();
  auto bound = NormalizationBound(input);
  ASSERT_TRUE(bound) << test_support::Failed(bound, &Error::ToString).value_or("");
  EXPECT_LT(bound->metadata_bytes, std::uint64_t{512} << 20U);
  EXPECT_EQ(bound->layers, 48U);
  auto plan =
      test_support::NormalizeWithFixtureTarget(input, {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  ASSERT_EQ(plan->routes().size(), 1224U);
  ASSERT_EQ(plan->spec().stages.size(), 1U);
  EXPECT_EQ(plan->spec().stages[0].side.size(), 1U);
  EXPECT_TRUE(plan->spec().stages[0].stream.mixing);
  EXPECT_EQ(plan->spec().stages[0].layers.size(), 48U);
  auto historical =
      spec::Synthesize(spec::mappings::Qwen38Mapping(), plan->canonical(), plan->values());
  ASSERT_TRUE(historical) << test_support::Failed(historical).value_or("");
  EXPECT_EQ(plan->spec(), *historical);  // same existing GGUF interpretation/policies
  std::set<Encoding> encodings;
  std::uint64_t bytes = 0;
  for (std::size_t i = 0; i < input.tensors.size(); ++i) {
    EXPECT_EQ(plan->canonical().tensors[i].name, input.tensors[i].name);
    EXPECT_EQ(plan->canonical().tensors[i].format, input.tensors[i].format);
    EXPECT_EQ(plan->routes()[i].source_tensor, i);
    EXPECT_EQ(plan->routes()[i].extent, input.tensors[i].extent);
    auto cursor = plan->CursorFor(i);
    ASSERT_TRUE(cursor);
    auto next = cursor->Next(65536);
    ASSERT_TRUE(next && *next);
    EXPECT_EQ((*next)->source_offset, input.tensors[i].extent.offset);
    EXPECT_EQ((*next)->bytes, std::min<std::uint64_t>(65536, input.tensors[i].extent.bytes));
    bytes += input.tensors[i].extent.bytes;
    encodings.insert(input.tensors[i].format.encoding);
  }
  EXPECT_EQ(encodings.size(), 7U);
  EXPECT_EQ(bytes, bound->payload_bytes);
  EXPECT_EQ(plan->applied_folds().size(), 3U);
  auto contracts = components::StageContracts(plan->spec(), 0);
  ASSERT_TRUE(contracts) << test_support::Failed(contracts).value_or("");
  std::size_t count = 0;
  for (const auto& [layer, state] : *contracts) {
    EXPECT_LT(layer, 48U);
    count += state.tensors.size();
  }
  EXPECT_EQ(count, 121U);
  // Caller's metadata can disappear; both normalized values and synthesized
  // side params own their lists, with no borrowed source views.
  input.metadata.clear();
  input.tensors.clear();
  auto moved = std::move(*plan);
  const auto value =
      std::ranges::find(moved.values(), "qwen4exp.ple.head_offsets", &spec::SourceValue::key);
  ASSERT_NE(value, moved.values().end());
  const auto& offsets = std::get<std::vector<std::int64_t>>(value->value);
  ASSERT_EQ(offsets.size(), 16U);
  EXPECT_EQ(offsets.back(), 300001275);
  EXPECT_TRUE(components::ValidateSpec(moved.spec(), moved.canonical().tensors));
}

TEST(ImportNormalizeQwenTest, PreservesDifferentValidSourceHashConstants) {
  auto input = test_support::Qwen38GgufCheckpoint();
  const std::vector<std::int64_t> multipliers{-7, 11, 19};
  const std::vector<std::int64_t> offsets(16, 17), vocab(16, 101);
  Value(input, "qwen4exp.ple.layer_multipliers", multipliers);
  Value(input, "qwen4exp.ple.head_offsets", offsets);
  Value(input, "qwen4exp.ple.head_vocab_sizes", vocab);
  auto plan = test_support::NormalizeWithFixtureTarget(std::move(input), kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  for (const auto& [key, expected] :
       std::array{std::pair{"qwen4exp.ple.layer_multipliers", multipliers},
                  std::pair{"qwen4exp.ple.head_offsets", offsets},
                  std::pair{"qwen4exp.ple.head_vocab_sizes", vocab}}) {
    const auto value = std::ranges::find(plan->values(), key, &spec::SourceValue::key);
    ASSERT_NE(value, plan->values().end());
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(value->value), expected);
  }
}

TEST(ImportNormalizeQwenTest, DifferentValidContextIsNotACheckpointIdentityGate) {
  auto input = test_support::Qwen38GgufCheckpoint();
  Value(input, "qwen4exp.context_length", std::int64_t{131071});
  Value(input, "qwen4exp.ple.image_token_id", std::int64_t{17});
  const auto plan = test_support::NormalizeWithFixtureTarget(input, kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  EXPECT_EQ(plan->spec().stages.front().context, 131071U);
}

TEST(ImportNormalizeQwenTest, ProfilePreflightHonorsExistingSourceAliases) {
  auto input = test_support::Qwen38GgufCheckpoint();
  const std::array keys{KeyAlias{"config.width", "qwen4exp.embedding_length"},
                        KeyAlias{"config.offsets", "qwen4exp.ple.head_offsets"}};
  const std::array tensors{TensorAlias{"words.weight", "token_embd.weight"},
                           TensorAlias{"ngrams.weight", "per_layer_token_embd.weight"}};
  for (const auto& alias : keys) {
    auto found = std::ranges::find(input.metadata, alias.canonical, &MetaEntry::key);
    ASSERT_NE(found, input.metadata.end());
    found->key = alias.source;
  }
  Ordered(input);
  for (const auto& alias : tensors) {
    auto found = std::ranges::find(input.tensors, alias.canonical, &Tensor::name);
    ASSERT_NE(found, input.tensors.end());
    found->name = alias.source;
  }
  auto table = *FindSourceMapping("gguf", "qwen4exp");
  table.keys = keys;
  table.tensors = tensors;
  ASSERT_TRUE(ValidateSourceMapping(table));
  auto plan = nd::Build(std::move(input), table, kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  EXPECT_EQ(plan->routes().size(), 1224U);
  EXPECT_EQ(plan->spec().stages[0].stream.width, 2560U);
  EXPECT_TRUE(components::ValidateSpec(plan->spec(), plan->canonical().tensors));
}

TEST(ImportNormalizeQwenTest, GgufLayerIntervalAgreesWithItsExplicitFlags) {
  auto source = test_support::Qwen38GgufCheckpoint();
  Value(source, "qwen4exp.full_attention_interval", std::int64_t{3});
  std::vector<std::int64_t> flags(48);
  for (std::size_t layer = 0; layer < flags.size(); ++layer)
    flags[layer] = (layer + 1) % 3 == 0 ? 4 : 0;
  Value(source, "qwen4exp.attention.compress_ratios", flags);
  auto cost = NormalizationBound(source);
  ASSERT_TRUE(cost) << cost.error().ToString();
  flags.front() = 4;
  Value(source, "qwen4exp.attention.compress_ratios", flags);
  EXPECT_FALSE(NormalizationBound(source));
}

TEST(ImportNormalizeQwenTest, RejectsUnfundedConstructionAndInvalidStructure) {
  const auto input = test_support::Qwen38GgufCheckpoint();
  const auto bound = NormalizationBound(input);
  ASSERT_TRUE(bound);
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(
      input, {bound->metadata_bytes - 1, bound->work_units}));
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(
      input, {bound->metadata_bytes, bound->work_units - 1}));
  for (int mode = 0; mode < 13; ++mode) {
    SCOPED_TRACE(mode);
    auto bad = input;
    switch (mode) {
      case 0:
        Value(bad, "qwen4exp.ple.layers", std::vector<std::int64_t>{1, 1});
        break;
      case 1:
        Value(bad, "qwen4exp.attention.compress_ratios", std::vector<std::int64_t>(47, 0));
        break;
      case 2:
        Value(bad, "qwen4exp.rope.dimension_sections", std::vector<std::int64_t>{11, 11, 11, 0});
        break;
      case 3:
        Value(bad, "qwen4exp.ple.head_offsets", std::vector<std::int64_t>(16, 320001536));
        break;
      case 4:
        Value(bad, "llmp.qwen4exp.indexer", std::string("fused"));
        break;
      case 5:
        Value(bad, "qwen4exp.attention.value_length", std::int64_t{128});
        break;
      case 6:
        Value(bad, "qwen4exp.unknown_structure", std::int64_t{1});
        break;
      case 7:
        bad.metadata.push_back(bad.metadata.back());
        break;
      case 8:
        std::erase_if(bad.tensors,
                      [](const auto& t) { return t.name == "blk.0.ffn_gate_inp.weight"; });
        break;
      case 9:
        bad.tensors.push_back(bad.tensors.front());
        break;
      case 10:
        bad.tensors.front().format.encoding = Encoding::kF16;
        break;
      case 11:
        bad.tensors.front().dims[0] += 1;
        break;
      case 12:
        Value(bad, "qwen4exp.ssm.inner_size", std::int64_t{6145});
        break;
    }
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(bad), kFunded));
  }
}

TEST(ImportNormalizeDsv4Test, CompleteCapturedTargetKeepsAllEncodingsAndLayerSemantics) {
  auto input = test_support::Dsv4GgufCheckpoint();
  auto cost = NormalizationBound(input);
  ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
  EXPECT_EQ(cost->layers, 43U);
  EXPECT_LT(cost->metadata_bytes, std::uint64_t{1} << 30U);
  auto plan =
      test_support::NormalizeWithFixtureTarget(input, {cost->metadata_bytes, cost->work_units});
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  ASSERT_EQ(plan->routes().size(), 1328U);
  ASSERT_EQ(plan->spec().stages.size(), 1U);
  EXPECT_EQ(plan->spec().stages[0].side.size(), 3U);
  EXPECT_EQ(plan->spec().stages[0].layers.size(), 43U);
  // The spec the approved artifact stores (runtime/approvals.cc), which
  // the retired profile's synthesis made.
  EXPECT_EQ(base::ToHex(plan->spec().Digest()),
            "ddaaa1d38bd6a501b9ef891bc0f639a70f41b19ff85a85674be5d77470e7a8e8");
  std::set<Encoding> encodings;
  for (std::size_t i = 0; i != input.tensors.size(); ++i) {
    EXPECT_EQ(plan->routes()[i].extent, input.tensors[i].extent);
    EXPECT_EQ(plan->canonical().tensors[i].format, input.tensors[i].format);
    encodings.insert(input.tensors[i].format.encoding);
  }
  EXPECT_EQ(encodings.size(), 10U);
  EXPECT_TRUE(plan->applied_folds().empty());
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(
      input, {cost->metadata_bytes - 1, cost->work_units}));
}
TEST(ImportNormalizeDsv4Test, RefusesIncompleteListsCountsAndUnclassifiedSourceSemantics) {
  for (unsigned mutation = 0; mutation != 7; ++mutation) {
    auto input = test_support::Dsv4GgufCheckpoint();
    if (mutation == 0) Value(input, "deepseek4.swiglu_clamp_exp", std::vector<double>(42, 10));
    if (mutation == 1)
      Value(input, "deepseek4.swiglu_clamp_shexp",
            std::vector<double>(43, std::numeric_limits<double>::infinity()));
    if (mutation == 2)
      Value(input, "deepseek4.attention.compress_ratios", std::vector<std::int64_t>(46, 4));
    if (mutation == 3) Value(input, "deepseek4.hash_layer_count", std::int64_t{44});
    if (mutation == 4) Value(input, "deepseek4.attention.value_length", std::int64_t{256});
    if (mutation == 5) Value(input, "deepseek4.new_semantics", std::int64_t{1});
    if (mutation == 6) input.tensors.pop_back();
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(input), kFunded)) << mutation;
  }
}

// antirez's community GGUFs (the IQ2_XXS artifact): the layer-type list may
// stop at the 43 blocks (its trailing MTP layers' entries absent) but must be
// the profile's prefix; the shared expert's clamps may be absent, when the
// mapping clamps it as the routed experts; the MTP depth and the vocabulary
// size are exact auxiliary keys, never a prefix that admits a near miss.
TEST(ImportNormalizeDsv4Test, CommunityListsAuxiliaryKeysAndTheClampFallback) {
  const auto normalized = [](Checkpoint input) {
    return test_support::NormalizeWithFixtureTarget(std::move(input), kFunded);
  };
  const auto ratios = [](std::size_t count) {
    std::vector<std::int64_t> out;
    for (std::size_t il = 0; il < count; ++il)
      out.push_back(il < 2 || il >= 43 ? 0 : (il % 2 == 0 ? 4 : 128));
    return out;
  };
  const auto erase = [](Checkpoint& input, std::string_view key) {
    std::erase_if(input.metadata, [&](const MetaEntry& e) { return e.key == key; });
  };
  {  // The 43 blocks' entries alone: accepted at the list's minimum size.
    auto input = test_support::Dsv4GgufCheckpoint();
    Value(input, "deepseek4.attention.compress_ratios", ratios(43));
    const auto plan = normalized(std::move(input));
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    EXPECT_EQ(plan->spec().stages[0].layers.size(), 43U);
  }
  {  // Below the minimum: refused.
    auto input = test_support::Dsv4GgufCheckpoint();
    Value(input, "deepseek4.attention.compress_ratios", ratios(42));
    EXPECT_FALSE(normalized(std::move(input)));
  }
  {  // Longer than the bounded architecture list: refused.
    auto input = test_support::Dsv4GgufCheckpoint();
    Value(input, "deepseek4.attention.compress_ratios", std::vector<std::int64_t>(1025, 0));
    EXPECT_FALSE(normalized(std::move(input)));
  }
  {  // Trailing type labels are provenance for layers not imported.
    auto input = test_support::Dsv4GgufCheckpoint();
    auto changed = ratios(44);
    changed.back() = 4;
    Value(input, "deepseek4.attention.compress_ratios", changed);
    EXPECT_TRUE(normalized(std::move(input)));
  }
  {  // A prefix that is not the profile's: refused.
    auto input = test_support::Dsv4GgufCheckpoint();
    auto mismatched = ratios(43);
    mismatched[2] = 128;
    Value(input, "deepseek4.attention.compress_ratios", mismatched);
    EXPECT_FALSE(normalized(std::move(input)));
  }
  {  // The shared expert's clamps absent: the routed experts' clamp it.
    auto input = test_support::Dsv4GgufCheckpoint();
    erase(input, "deepseek4.swiglu_clamp_shexp");
    Value(input, "deepseek4.swiglu_clamp_exp", std::vector<double>(43, 9.5));
    const auto plan = normalized(std::move(input));
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    std::size_t clamped = 0;
    for (const auto& layer : plan->spec().stages[0].layers)
      for (const auto& c : layer.components)
        if (const auto* shared = c.Find("shared_limit"); shared != nullptr) {
          const auto* routed = c.Find("limit");
          ASSERT_NE(routed, nullptr);
          EXPECT_EQ(shared->value, routed->value);
          EXPECT_EQ(shared->value, spec::ParamValue(9.5));
          ++clamped;
        }
    EXPECT_EQ(clamped, 43U);
  }
  {  // Present but malformed: refused, not replaced by the fallback.
    auto input = test_support::Dsv4GgufCheckpoint();
    Value(input, "deepseek4.swiglu_clamp_shexp", std::vector<double>(42, 10));
    EXPECT_FALSE(normalized(std::move(input)));
  }
  {  // The exact auxiliary keys: accepted.
    auto input = test_support::Dsv4GgufCheckpoint();
    Value(input, "deepseek4.nextn_predict_layers", std::int64_t{1});
    Value(input, "deepseek4.attention.compress_ratios", ratios(44));
    Value(input, "deepseek4.vocab_size", std::int64_t{129280});
    const auto plan = normalized(std::move(input));
    EXPECT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  }
  {  // A vocabulary size the token table does not state: refused.
    auto input = test_support::Dsv4GgufCheckpoint();
    Value(input, "deepseek4.vocab_size", std::int64_t{129281});
    EXPECT_FALSE(normalized(std::move(input)));
  }
  // The community GGUF's encodings: an F16 compressor APE table, Q2_K down
  // and IQ2_XXS gate experts, each kept as the source holds it.
  for (const auto& [name, encoding] :
       {std::pair<std::string_view, Encoding>{"attn_compressor_ape.weight", Encoding::kF16},
        std::pair<std::string_view, Encoding>{"ffn_down_exps.weight", Encoding::kQ2_K},
        std::pair<std::string_view, Encoding>{"ffn_gate_exps.weight", Encoding::kIQ2_XXS}}) {
    SCOPED_TRACE(name);
    auto input = test_support::Dsv4GgufCheckpoint();
    const auto at = std::ranges::find_if(input.tensors,
                                         [&](const Tensor& t) { return t.name.ends_with(name); });
    ASSERT_NE(at, input.tensors.end());
    const auto index = static_cast<std::size_t>(at - input.tensors.begin());
    at->format.encoding = encoding;
    at->dtype = format::Info(encoding).name;  // as the GGUF names its type
    at->format.layout =
        format::Info(encoding).quantized ? format::Layout::kGgmlBlocks : format::Layout::kDense;
    const auto parts = format::PartsOf(at->format);
    ASSERT_TRUE(parts);
    ASSERT_LE((*parts)[0].bytes, at->extent.bytes);  // within its old extent
    at->extent.bytes = (*parts)[0].bytes;
    const auto plan = normalized(std::move(input));
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    EXPECT_EQ(plan->canonical().tensors[index].format.encoding, encoding);
  }
  for (const std::string_view near :
       {"deepseek4.nextn_predict_layers_extra", "deepseek4.vocab_size.v2", "deepseek4.vocab"}) {
    // A near miss of an exact key stays unclassified: refused.
    auto input = test_support::Dsv4GgufCheckpoint();
    Value(input, std::string(near), std::int64_t{1});
    EXPECT_FALSE(normalized(std::move(input))) << near;
  }
}

TEST(ImportNormalizeDsv4Test, ExtraLayerLabelsAreBoundedAndExplicitDepthCannotDisagree) {
  auto input = test_support::Dsv4GgufCheckpoint();
  ASSERT_TRUE(NormalizationBound(input));  // captured 43-block, 46-label source
  auto labels =
      std::get<std::vector<std::int64_t>>(*input.FindValue("deepseek4.attention.compress_ratios"));
  Value(input, "deepseek4.nextn_predict_layers", std::int64_t{2});
  EXPECT_FALSE(NormalizationBound(input));
  Value(input, "deepseek4.nextn_predict_layers", std::int64_t{3});
  EXPECT_TRUE(NormalizationBound(input));
  labels.back() = 7;
  Value(input, "deepseek4.attention.compress_ratios", labels);
  EXPECT_FALSE(NormalizationBound(input));
}

TEST(ImportNormalizeQwen2Test, CompleteCapturedTargetMatchesV0SpecWithOmittedOrExplicitKeyLength) {
  for (const bool explicit_keys : {false, true}) {
    auto input = test_support::Qwen2GgufCheckpoint();
    if (explicit_keys) {
      Value(input, "qwen2.attention.key_length", std::int64_t{64});
      Value(input, "qwen2.attention.value_length", std::int64_t{64});
      Value(input, "qwen2.rope.dimension_count", std::int64_t{64});
      Value(input, "qwen2.rope.scaling.type", std::string("none"));
    }
    auto cost = NormalizationBound(input);
    ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
    EXPECT_EQ(cost->layers, 24U);
    EXPECT_EQ(cost->payload_bytes, 1260477952ULL);
    auto plan =
        test_support::NormalizeWithFixtureTarget(input, {cost->metadata_bytes, cost->work_units});
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    ASSERT_EQ(plan->routes().size(), 291U);
    ASSERT_EQ(plan->spec().stages.size(), 1U);
    EXPECT_EQ(plan->spec().stages[0].layers.size(), 24U);
    EXPECT_EQ(plan->spec().stages[0].heads.output->weights[0].tensor, "output.weight");
    // The spec the approved artifact stores (runtime/approvals.cc), with or
    // without the optional keys.
    EXPECT_EQ(base::ToHex(plan->spec().Digest()),
              "d673370e16b0b37fe06a16305abcf032842a8d792afa6b9e30f77e10027f5403");
    EXPECT_TRUE(plan->applied_folds().empty());
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(
        input, {cost->metadata_bytes - 1, cost->work_units}));
  }
  auto tied = test_support::Qwen2GgufCheckpoint(false);
  auto plan = test_support::NormalizeWithFixtureTarget(std::move(tied), kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  ASSERT_EQ(plan->routes().size(), 290U);
  EXPECT_EQ(plan->spec().stages[0].heads.output->weights[0].tensor, "token_embd.weight");
}

TEST(ImportNormalizeQwen2Test, RefusesProfileShapeEncodingAndUnclassifiedMetadataDrift) {
  for (unsigned mutation = 0; mutation != 9; ++mutation) {
    auto input = test_support::Qwen2GgufCheckpoint();
    if (mutation == 0) Value(input, "qwen2.block_count", std::int64_t{28});
    if (mutation == 1) Value(input, "qwen2.attention.key_length", std::int64_t{128});
    if (mutation == 2) Value(input, "qwen2.rope.scaling.type", std::string("yarn"));
    if (mutation == 3) Value(input, "qwen2.rope.freq_base", 10000.0);
    if (mutation == 4) Value(input, "qwen2.unknown_semantics", std::int64_t{1});
    if (mutation == 5) Value(input, "llmp.qwen2.exl3", true);
    if (mutation == 6) input.tensors.front().format.encoding = Encoding::kBF16;
    if (mutation == 7) input.tensors.back().dims[0] = 1536;
    if (mutation == 8) input.tensors.pop_back();
    if (mutation == 3)
      EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(std::move(input), kFunded));
    else
      EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(input), kFunded)) << mutation;
  }
}

TEST(ImportNormalizeQwen2Exl3Test, CompleteCaptured40And45TargetsMatchV0SpecAndPreserveParts) {
  for (const bool rate_4_5 : {false, true}) {
    auto input = test_support::Qwen2Exl3Checkpoint(rate_4_5);
    auto cost = NormalizationBound(input);
    ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
    EXPECT_EQ(cost->layers, 24U);
    EXPECT_EQ(cost->payload_bytes, rate_4_5 ? 611173284ULL : 588866468ULL);
    auto plan =
        test_support::NormalizeWithFixtureTarget(input, {cost->metadata_bytes, cost->work_units});
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    ASSERT_EQ(plan->routes().size(), 798U);
    ASSERT_EQ(plan->spec().stages.size(), 1U);
    EXPECT_EQ(plan->spec().stages[0].layers.size(), 24U);
    // The spec both approved EXL3 artifacts store (runtime/approvals.cc),
    // identical to the tied GGUF spec above.
    EXPECT_EQ(base::ToHex(plan->spec().Digest()),
              "b2cf6f15206be2712e20fe080df11f7edcbf8aaeda23f830755f767f11433d6a");
    EXPECT_TRUE(plan->applied_folds().empty());
    EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(
        input, {cost->metadata_bytes - 1, cost->work_units}));
  }
}

TEST(ImportNormalizeQwen2Exl3Test, RefusesMalformedPartRolesRatesCodebooksAndMetadataDrift) {
  for (unsigned mutation = 0; mutation != 10; ++mutation) {
    auto input = test_support::Qwen2Exl3Checkpoint();
    if (mutation == 0) input.quantization = "gguf";
    if (mutation == 1) Value(input, "architectures", std::vector<std::string>{"LlamaForCausalLM"});
    if (mutation == 2) Value(input, "tie_word_embeddings", false);
    if (mutation == 3) Value(input, "use_sliding_window", true);
    if (mutation == 4) Value(input, "rms_norm_eps", 1e-5);
    if (mutation == 5) input.tensors[2].format.params.k_bits = 7;
    if (mutation == 6) input.tensors[2].format.params.codebook = format::Exl3Codebook::kMul1;
    if (mutation == 7) input.tensors[5].dims = {1};
    if (mutation == 8) input.tensors[3].format.params.k_bits = 4;  // suh disagrees with trellis
    if (mutation == 9) input.tensors.pop_back();
    if (mutation == 1 || mutation == 2 || mutation == 4)
      EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(std::move(input), kFunded)) << mutation;
    else
      EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(input), kFunded)) << mutation;
  }
}

TEST(ImportNormalizeQwen2Exl3Test, ReaderCodebooksAndRatesPreserveFormatsAndRequireExactParts) {
  using format::Exl3Codebook;
  using format::PartRole;
  for (const auto codebook : {Exl3Codebook::kMcg, Exl3Codebook::kMul1, Exl3Codebook::k3Inst}) {
    for (const auto bits : {1U, 3U, 7U}) {
      SCOPED_TRACE(static_cast<unsigned>(codebook));
      SCOPED_TRACE(bits);
      auto input = test_support::Qwen2Exl3Checkpoint();
      if (codebook == Exl3Codebook::k3Inst)
        std::erase_if(input.tensors,
                      [](const auto& tensor) { return tensor.part == PartRole::kCodebookMarker; });
      std::uint64_t offset = 0;
      for (auto& tensor : input.tensors) {
        if (tensor.format.encoding == Encoding::kExl3Trellis) {
          tensor.format.params.codebook = codebook;
          tensor.format.params.k_bits = static_cast<std::uint8_t>(bits);
          if (tensor.part == PartRole::kCodes) tensor.dims[2] = 16ULL * bits;
          if (tensor.part == PartRole::kCodebookMarker && codebook == Exl3Codebook::kMul1)
            tensor.name.replace(tensor.name.size() - 4, 4, ".mul1");
        }
        tensor.extent.offset = offset;
        tensor.extent.bytes = format::PartsOf(tensor.format)->Find(tensor.part)->bytes;
        offset += tensor.extent.bytes;
      }
      input.files[0].bytes = offset;
      auto plan = test_support::NormalizeWithFixtureTarget(input, kFunded);
      ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
      for (const auto& tensor : plan->canonical().tensors)
        if (tensor.format.encoding == Encoding::kExl3Trellis) {
          EXPECT_EQ(tensor.format.params.codebook, codebook);
          EXPECT_EQ(tensor.format.params.k_bits, bits);
          // The legacy canonical marker spelling is a binding identity only.
          if (tensor.part == PartRole::kCodebookMarker) EXPECT_TRUE(tensor.name.ends_with(".mcg"));
        }
      if (codebook == Exl3Codebook::k3Inst) {
        const auto trellis = std::ranges::find(input.tensors, "lm_head.trellis", &Tensor::name);
        ASSERT_NE(trellis, input.tensors.end());
        auto marker = *trellis;
        marker.name = "lm_head.mcg";
        marker.part = PartRole::kCodebookMarker;
        marker.dtype = "I32";
        marker.dims.clear();
        marker.extent = {.offset = offset, .bytes = 4};
        input.tensors.push_back(std::move(marker));
        input.files[0].bytes += 4;
      } else {
        std::erase_if(input.tensors, [](const auto& tensor) {
          return tensor.name == "lm_head.mcg" || tensor.name == "lm_head.mul1";
        });
      }
      EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(input), kFunded));
    }
  }
}

TEST(ImportNormalizeDrafterTest, ApprovedGgufDrafterMappingsValidateAndRejectMalformedRoles) {
  const auto* gemma_asst = FindSourceMapping("gguf", "gemma4-assistant");
  ASSERT_NE(gemma_asst, nullptr);
  EXPECT_TRUE(ValidateSourceMapping(*gemma_asst));
  EXPECT_EQ(gemma_asst->architecture->role, spec::StageRole::kDrafter);
  EXPECT_EQ(gemma_asst->architecture->stage, "assistant");
  EXPECT_FALSE(gemma_asst->architecture->reads.empty());

  const auto* dspark = FindSourceMapping("gguf", "dflash");
  ASSERT_NE(dspark, nullptr);
  EXPECT_TRUE(ValidateSourceMapping(*dspark));
  EXPECT_EQ(dspark->architecture->role, spec::StageRole::kDrafter);
  EXPECT_EQ(dspark->architecture->stage, "dspark");
  EXPECT_FALSE(dspark->architecture->reads.empty());
  EXPECT_TRUE(dspark->extended_components);

  auto bad_gemma = *gemma_asst;
  auto bad_m = *gemma_asst->architecture;
  bad_m.stage = "target";
  bad_gemma.architecture = &bad_m;
  EXPECT_FALSE(ValidateSourceMapping(bad_gemma));

  bad_m = *gemma_asst->architecture;
  bad_m.role = spec::StageRole::kTarget;
  bad_gemma.architecture = &bad_m;
  EXPECT_FALSE(ValidateSourceMapping(bad_gemma));

  bad_m = *gemma_asst->architecture;
  bad_m.stage = "";
  bad_gemma.architecture = &bad_m;
  EXPECT_FALSE(ValidateSourceMapping(bad_gemma));
}

TEST(ImportNormalizeDrafterTest, NormalizesGemma4AssistantCheckpointsIntoComposedSpecs) {
  for (std::uint32_t size : {26U, 31U}) {
    auto cp = AssistantCheckpoint(size);
    const auto count = cp.tensors.size();
    EXPECT_EQ(count, 49U);

    auto cost = NormalizationBound(cp);
    ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
    auto plan = test_support::NormalizeWithFixtureTarget(std::move(cp),
                                                         {cost->metadata_bytes, cost->work_units});
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");

    // Drafter spec is composed: stage 0 is target, stage 1 is drafter.
    ASSERT_EQ(plan->spec().stages.size(), 2U);
    EXPECT_EQ(plan->spec().stages[0].role, spec::StageRole::kTarget);
    EXPECT_EQ(plan->spec().stages[0].artifact, 0U);
    EXPECT_EQ(plan->spec().stages[1].role, spec::StageRole::kDrafter);
    EXPECT_EQ(plan->spec().stages[1].artifact, 1U);

    // Target width matches embedding_length_out (2816 for 26B, 5376 for 31B).
    const std::uint32_t expected_target_width = size == 26 ? 2816 : 5376;
    EXPECT_EQ(plan->spec().stages[0].stream.width, expected_target_width);
    EXPECT_EQ(plan->spec().stages[1].stream.width, 1024U);

    // embedding.table reads target stage.
    ASSERT_TRUE(plan->spec().stages[1].embedding.has_value());
    const auto table_w = std::ranges::find(plan->spec().stages[1].embedding->weights, "table",
                                           &spec::WeightBinding::role);
    ASSERT_NE(table_w, plan->spec().stages[1].embedding->weights.end());
    EXPECT_EQ(table_w->stage, "target");

    // Routes cover all 49 drafter tensors.
    EXPECT_EQ(plan->routes().size(), 49U);
  }
}

TEST(ImportNormalizeDrafterTest, RejectsMissingDrafterKeysAndMismatchedParentWidth) {
  auto cp = AssistantCheckpoint(26);
  // Remove embedding_length_out
  auto it =
      std::ranges::find(cp.metadata, "gemma4-assistant.embedding_length_out", &MetaEntry::key);
  ASSERT_NE(it, cp.metadata.end());
  cp.metadata.erase(it);
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(cp), kFunded));

  // Invalid embedding_length_out (neither 2816 nor 5376)
  cp = AssistantCheckpoint(26);
  Value(cp, "gemma4-assistant.embedding_length_out", std::int64_t{1024});
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(cp), kFunded));
}

TEST(ImportNormalizeSafetensorsTest, Qwen2SafetensorsMappingValidatesAndGgufSynthesizesNon05B) {
  const auto* qwen2_sf = FindSourceMapping("safetensors", "qwen2");
  ASSERT_NE(qwen2_sf, nullptr);
  EXPECT_TRUE(ValidateSourceMapping(*qwen2_sf));
  for (const auto arch : {"llama", "mistral"}) {
    for (const auto container : {"gguf", "safetensors", "exl3"}) {
      const auto* table = FindSourceMapping(container, arch);
      ASSERT_NE(table, nullptr) << container << " " << arch;
      EXPECT_TRUE(ValidateSourceMapping(*table)) << container << " " << arch;
    }
  }

  spec::Source source;
  source.architecture = "qwen2";
  source.container = "gguf";
  const std::vector<std::string> names = {
      "token_embd.weight",     "blk.0.attn_norm.weight", "blk.0.attn_q.weight",
      "blk.0.attn_k.weight",   "blk.0.attn_v.weight",    "blk.0.attn_q.bias",
      "blk.0.attn_k.bias",     "blk.0.attn_v.bias",      "blk.0.attn_output.weight",
      "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight",  "blk.0.ffn_up.weight",
      "blk.0.ffn_down.weight", "output_norm.weight",     "output.weight",
  };
  std::vector<spec::SourceTensor> tensors;
  for (const auto& name : names) {
    tensors.push_back({.name = name});
  }
  source.tensors = tensors;

  const std::vector<spec::SourceValue> values = {
      {.key = "qwen2.block_count", .value = std::int64_t{1}},
      {.key = "qwen2.context_length", .value = std::int64_t{32768}},
      {.key = "qwen2.embedding_length", .value = std::int64_t{3584}},
      {.key = "qwen2.feed_forward_length", .value = std::int64_t{18944}},
      {.key = "qwen2.attention.head_count", .value = std::int64_t{28}},
      {.key = "qwen2.attention.head_count_kv", .value = std::int64_t{4}},
      {.key = "qwen2.attention.key_length", .value = std::int64_t{128}},
      {.key = "qwen2.attention.value_length", .value = std::int64_t{128}},
      {.key = "qwen2.rope.scaling.type", .value = std::string("none")},
      {.key = "qwen2.rope.freq_base", .value = 1000000.0},
      {.key = "qwen2.attention.layer_norm_rms_epsilon", .value = static_cast<double>(1e-6F)},
  };
  auto spec = spec::Synthesize(spec::mappings::Qwen2GgufMapping(), source, values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  EXPECT_EQ(spec->stages[0].stream.width, 3584U);
  EXPECT_EQ(spec->stages[0].layers.size(), 1U);
}

TEST(ImportNormalizeSafetensorsTest, ConditionalMetadataAliasesUseTheDeclaredPrimary) {
  auto source = ParseSafetensors(fixtures::Shard(fixtures::TinyQwen2HfEntries()),
                                 fixtures::TinyQwen2HfConfig());
  ASSERT_TRUE(source);
  auto table = *FindSourceMapping("safetensors", "qwen2");
  std::vector<spec::KeyAlias> aliases(table.keys.begin(), table.keys.end());
  aliases.push_back({"legacy_rms_eps", "attention.layer_norm_rms_epsilon", "rms_norm_eps"});
  table.keys = aliases;
  std::vector<spec::AuxiliaryNamespace> auxiliary(table.auxiliary.begin(), table.auxiliary.end());
  auxiliary.push_back({"legacy_rms_eps", "provenance when the primary mathematical scalar exists"});
  table.auxiliary = auxiliary;
  ASSERT_TRUE(ValidateSourceMapping(table));
  Value(*source, "legacy_rms_eps", std::vector<std::int64_t>{7, 8});
  auto preferred = nd::Build(*source, table, kFunded);
  ASSERT_TRUE(preferred) << preferred.error().ToString();
  EXPECT_DOUBLE_EQ(
      std::get<double>(
          preferred->spec().stages.front().layers.front().components.front().Find("eps")->value),
      1e-6);
  std::erase_if(source->metadata, [](const auto& entry) { return entry.key == "rms_norm_eps"; });
  EXPECT_FALSE(nd::Build(*source, table, kFunded));
  Value(*source, "legacy_rms_eps", 0.000002);
  auto fallback = nd::Build(*source, table, kFunded);
  ASSERT_TRUE(fallback) << fallback.error().ToString();
  EXPECT_DOUBLE_EQ(
      std::get<double>(
          fallback->spec().stages.front().layers.front().components.front().Find("eps")->value),
      2e-6);
  aliases.back().unless_present = "legacy_rms_eps";
  EXPECT_FALSE(ValidateSourceMapping(table));
}

TEST(ImportNormalizeSafetensorsTest, SharedScalarMayPrecedeItsIndexedRowShards) {
  auto entries = fixtures::TinyQwen2HfEntries();
  std::erase_if(entries,
                [](const auto& entry) { return entry.name == "model.embed_tokens.weight"; });
  for (auto& entry : entries) {
    for (auto& dim : entry.shape)
      if (dim == 4 || dim == 8 || dim == 16) dim *= 2;
    if (entry.bytes) entry.bytes = *format::BlockBytes(Encoding::kF16);
    for (auto dim : entry.shape) entry.bytes *= dim;
  }
  entries.insert(entries.begin(), fixtures::E("model.embed_tokens.weight_scale_2", "F32", {}));
  for (const auto shard : {0, 1}) {
    const auto prefix = std::format("model.embed_tokens.shard_{}.", shard);
    entries.push_back(fixtures::E(prefix + "weight", "U8", {5, 8}));
    entries.push_back(fixtures::E(prefix + "weight_scale", "F8_E4M3", {5, 1}));
  }
  auto config = fixtures::TinyQwen2HfConfig();
  for (const auto& [from, to] : std::array<std::pair<std::string_view, std::string_view>, 3>{
           {{"\"hidden_size\":8", "\"hidden_size\":16"},
            {"\"intermediate_size\":16", "\"intermediate_size\":32"},
            {"\"head_dim\":4", "\"head_dim\":8"}}})
    config.replace(config.find(from), from.size(), to);
  auto source = ParseSafetensors(fixtures::Shard(entries), config);
  ASSERT_TRUE(source) << source.error().ToString();
  std::erase_if(source->metadata, [](const auto& entry) { return entry.key == "vocab_size"; });
  // ParseSafetensors is the raw in-memory parser; describe the reader's
  // paired NVFP4 parts explicitly before exercising generic normalization.
  for (auto& tensor : source->tensors) {
    if (!tensor.name.starts_with("model.embed_tokens.")) continue;
    tensor.format = {.encoding = Encoding::kNvfp4,
                     .params = {.tensor_scale = true},
                     .layout = format::Layout::kSplit,
                     .shape = {.ne = {16, 5, 1, 1}, .rank = 2}};
    tensor.part = tensor.name.ends_with(".weight_scale_2") ? format::PartRole::kTensorScale
                  : tensor.name.ends_with(".weight_scale") ? format::PartRole::kBlockScales
                                                           : format::PartRole::kCodes;
  }
  auto table = *FindSourceMapping("safetensors", "qwen2");
  std::vector<spec::TensorAlias> aliases(table.tensors.begin(), table.tensors.end());
  aliases.push_back({"model.embed_tokens.shard_{shard}.weight", "token_embd.weight"});
  table.tensors = aliases;
  auto plan = nd::Build(std::move(*source), table, kFunded);
  ASSERT_TRUE(plan) << plan.error().ToString();
  auto valid = components::ValidateSpec(plan->spec(), plan->canonical().tensors);
  EXPECT_TRUE(valid) << valid.error();
  const auto embedding = std::ranges::find(
      plan->canonical().tensors, std::string("token_embd.weight"), &spec::SourceTensor::name);
  ASSERT_NE(embedding, plan->canonical().tensors.end());
  EXPECT_EQ(embedding->format.shape.ne[1], 10U);
}

TEST(ImportNormalizeSafetensorsTest, RealisticTransformersConfigNormalizesWithEitherDtypeSpelling) {
  const auto entries = fixtures::TinyQwen2HfEntries();
  for (const auto dtype : {"dtype", "torch_dtype"}) {
    const auto config = fixtures::TinyQwen2HfConfig(dtype);
    auto input = ParseSafetensors(fixtures::Shard(entries), config);
    ASSERT_TRUE(input) << test_support::Failed(input, &Error::ToString).value_or("");
    ASSERT_EQ(input->skipped_tensors.size(), 1U);
    auto bound = NormalizationBound(*input);
    ASSERT_TRUE(bound) << test_support::Failed(bound, &Error::ToString).value_or("");
    auto plan = test_support::NormalizeWithFixtureTarget(
        std::move(*input), {bound->metadata_bytes, bound->work_units});
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    EXPECT_EQ(plan->spec().stages[0].stream.width, 8U);
    EXPECT_EQ(plan->canonical().tensors.size(), 15U);
    ASSERT_TRUE(plan->is_indexed());
    EXPECT_EQ(plan->parts().members().size(), 15U);
    std::set<std::uint32_t> sources;
    for (const auto& member : plan->parts().members()) {
      ASSERT_LT(member.source, plan->source().tensors.size());
      EXPECT_TRUE(sources.insert(member.source).second);
      EXPECT_EQ(member.extent, plan->source().tensors[member.source].extent);
    }
    auto malformed = config;
    const auto null_at = malformed.find("\"rope_scaling\":null");
    ASSERT_NE(null_at, std::string::npos);
    malformed.replace(null_at, std::string_view("\"rope_scaling\":null").size(),
                      "\"rope_scaling\":[true,1]");
    auto unsafe = ParseSafetensors(fixtures::Shard(entries), malformed);
    ASSERT_TRUE(unsafe);
    EXPECT_FALSE(NormalizationBound(*unsafe));
  }
}

TEST(ImportNormalizeSafetensorsTest, EmptyFallbackRoleMatchesAbsentButRequiredEmptyStillRefuses) {
  // Tied: the head falls back to the token table.
  auto config = fixtures::TinyQwen2HfConfig();
  const std::string_view untied = R"("tie_word_embeddings":false)";
  const auto at = config.find(untied);
  ASSERT_NE(at, std::string::npos);
  config.replace(at, untied.size(), R"("tie_word_embeddings":true)");
  auto entries = fixtures::TinyQwen2HfEntries();
  std::erase_if(entries, [](const auto& entry) { return entry.name == "lm_head.weight"; });
  auto absent = ParseSafetensors(fixtures::Shard(entries), config);
  ASSERT_TRUE(absent);
  auto expected = test_support::NormalizeWithFixtureTarget(std::move(*absent), kFunded);
  ASSERT_TRUE(expected) << test_support::Failed(expected, &Error::ToString).value_or("");
  entries.push_back(fixtures::E("lm_head.weight", "F16", {0}));
  auto empty = ParseSafetensors(fixtures::Shard(entries), config);
  ASSERT_TRUE(empty);
  auto plan = test_support::NormalizeWithFixtureTarget(std::move(*empty), kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  EXPECT_EQ(plan->spec(), expected->spec());
  EXPECT_EQ(plan->spec().stages[0].heads.output->weights[0].tensor, "token_embd.weight");
  for (auto& entry : entries)
    if (entry.name == "model.norm.weight") entry = fixtures::E(entry.name, "F16", {0});
  auto required = ParseSafetensors(fixtures::Shard(entries), config);
  ASSERT_TRUE(required);
  EXPECT_TRUE(NormalizationBound(*required));
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(*required), kFunded));
}

TEST(ImportNormalizeSafetensorsTest, UntiedFlagWithoutAHeadRefusesForQwen2AndExl3) {
  // Qwen2 safetensors: an untied checkpoint without lm_head would otherwise
  // fall back to the token table, tying what config.json says is untied.
  auto entries = fixtures::TinyQwen2HfEntries();
  std::erase_if(entries, [](const auto& entry) { return entry.name == "lm_head.weight"; });
  auto untied = ParseSafetensors(fixtures::Shard(entries), fixtures::TinyQwen2HfConfig());
  ASSERT_TRUE(untied);
  EXPECT_FALSE(NormalizationBound(*untied));
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(*untied), kFunded));
  // The flag is read as a flag, whatever the table.
  auto config = fixtures::TinyQwen2HfConfig();
  const std::string_view flag = R"("tie_word_embeddings":false)";
  config.replace(config.find(flag), flag.size(), R"("tie_word_embeddings":"no")");
  auto text = ParseSafetensors(fixtures::Shard(fixtures::TinyQwen2HfEntries()), config);
  ASSERT_TRUE(text);
  EXPECT_FALSE(NormalizationBound(*text));
  // EXL3: the head's trellis must be there when untied.
  auto exl3 = test_support::Qwen2Exl3Checkpoint();
  Value(exl3, "tie_word_embeddings", false);
  std::erase_if(exl3.tensors, [](const Tensor& t) { return t.name.starts_with("lm_head."); });
  EXPECT_FALSE(NormalizationBound(exl3));
}

TEST(ImportNormalizeTest, SemanticNamespaceIsTheArchitecturesOwn) {
  // Another architecture's keys are not this mapping's math (llama.cpp reads
  // only "<general.architecture>."): retained as provenance.
  auto input = Tiny();
  Value(input, "llama.rope.freq_base", 500000.0);
  Value(input, "gemma3.attention.scale", 0.5);
  auto plan = test_support::NormalizeWithFixtureTarget(input, kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  for (const std::string_view key : {"llama.rope.freq_base", "gemma3.attention.scale"}) {
    const auto entry = std::ranges::find(plan->source().metadata, key, &MetaEntry::key);
    ASSERT_NE(entry, plan->source().metadata.end()) << key;
    EXPECT_EQ(plan->metadata()[static_cast<std::size_t>(entry - plan->source().metadata.begin())]
                  .classification,
              MetadataClass::kAuxiliary)
        << key;
  }
  // Its own unconsumed keys refuse, for every registered GGUF table.
  for (const std::string_view arch : {"gemma4-assistant", "dflash", "qwen4exp", "deepseek4"}) {
    const auto* table = FindSourceMapping("gguf", arch);
    ASSERT_NE(table, nullptr) << arch;
    auto other = Tiny();
    other.architecture = std::string(arch);
    Value(other, "general.architecture", std::string(arch));
    Value(other, std::string(arch) + ".unknown_semantics", std::int64_t{1});
    EXPECT_EQ(test_support::Failed(NormalizationBound(other), &Error::reason),
              "unconsumed semantic metadata changes unsupported mapping semantics")
        << arch;
  }
}

TEST(ImportNormalizeTest, UnreadProvenanceIsRetainedWhileUnconsumedSemanticsRefuse) {
  auto input = Tiny();
  Value(input, "new_converter.note", std::string("provenance"));
  auto plan = test_support::NormalizeWithFixtureTarget(input, kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  const auto entry =
      std::ranges::find(plan->source().metadata, "new_converter.note", &MetaEntry::key);
  ASSERT_NE(entry, plan->source().metadata.end());
  const auto use =
      plan->metadata()[static_cast<std::size_t>(entry - plan->source().metadata.begin())];
  EXPECT_EQ(use.classification, MetadataClass::kAuxiliary);
  EXPECT_TRUE(plan->auxiliary_namespaces()[use.destination].prefix.empty());
  Value(input, "gemma2.rope.scaling.new_algorithm", true);
  EXPECT_FALSE(NormalizationBound(input));
}

TEST(ImportNormalizeTest, EmptyTensorRecordsCannotHideRequiredWeights) {
  auto input = Tiny();
  input.skipped_tensors.push_back({.name = "unused.empty", .reason = "zero-element tensor"});
  EXPECT_TRUE(test_support::NormalizeWithFixtureTarget(input, kFunded));
  std::erase_if(input.tensors,
                [](const auto& tensor) { return tensor.name == "output_norm.weight"; });
  input.skipped_tensors.push_back({.name = "output_norm.weight", .reason = "zero-element tensor"});
  // Preflight bounds the inventory; synthesis owns actual role requiredness.
  EXPECT_TRUE(NormalizationBound(input));
  EXPECT_FALSE(test_support::NormalizeWithFixtureTarget(std::move(input), kFunded));
}

TEST(ImportNormalizeQwen2Exl3Test, DenseFormatsAndShardsAreStructuralAndNominalDtypeIsProvenance) {
  auto input = test_support::Qwen2Exl3Checkpoint();
  input.files.push_back(input.files.front());
  input.files.back().name = "second.safetensors";
  // Metadata-only memory fixture; the extra file must have a distinct physical identity.
  input.files.front().path = "/tmp/llmp-exl3-first";
  input.files.front().inode = 1;
  input.files.back().path = "/tmp/llmp-exl3-second";
  input.files.back().inode = 2;
  input.tensors.back().extent.file = 1;
  Value(input, "torch_dtype", std::string("float16"));
  Value(input, "bos_token_id", std::int64_t{1});
  auto plan = test_support::NormalizeWithFixtureTarget(input, kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  EXPECT_EQ(plan->spec().stages[0].stream.width, 896U);
}

TEST(ImportNormalizeDrafterTest, ActualTargetIsRequiredAndValidatedBeforePlanning) {
  auto source = AssistantCheckpoint(26);
  const auto missing = llmp::import::Normalize(source, kFunded);
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().reason.find("--target"), std::string::npos);
  auto target = test_support::GemmaImportTarget(26);
  EXPECT_TRUE(llmp::import::Normalize(source, kFunded, &target));
  target.model.stages.front().layers.resize(24);
  std::erase_if(target.source.tensors, [](const spec::SourceTensor& tensor) {
    if (!tensor.name.starts_with("blk.")) return false;
    const auto dot = tensor.name.find('.', 4);
    std::uint32_t layer = 0;
    for (const char c : tensor.name.substr(4, dot - 4))
      layer = layer * 10 + static_cast<std::uint32_t>(c - '0');
    return layer >= 24;
  });
  auto smaller = llmp::import::Normalize(source, kFunded, &target);
  ASSERT_TRUE(smaller) << smaller.error().ToString();
  EXPECT_EQ(smaller->spec().stages.front().layers.size(), 24U);
  target.source.architecture = "different";
  EXPECT_FALSE(llmp::import::Normalize(source, kFunded, &target));
  target = test_support::GemmaImportTarget(31);
  EXPECT_FALSE(llmp::import::Normalize(source, kFunded, &target));
}

TEST(ImportNormalizeTest, OversizedMetadataListsAreRetainedForExplicitValidation) {
  Checkpoint source;
  source.metadata.push_back({"test.list", std::vector<std::int64_t>(spec::kMaxList + 1, 7)});
  const auto values = ValuesOf(source);
  ASSERT_EQ(values.size(), 1U);
  EXPECT_EQ(std::get<std::vector<std::int64_t>>(values.front().value).size(), spec::kMaxList + 1);
}

TEST(ImportNormalizeTest, LlamaMappingSynthesizesLlama3FrequencyScaledRope) {
  auto source = SourceOf(test_support::Qwen2GgufCheckpoint());
  source.architecture = "llama";
  std::erase_if(source.tensors, [](const spec::SourceTensor& t) {
    return t.name.ends_with(".attn_q.bias") || t.name.ends_with(".attn_k.bias") ||
           t.name.ends_with(".attn_v.bias");
  });
  const std::vector<spec::SourceValue> values = {
      {"llama.block_count", std::int64_t{24}},
      {"llama.context_length", std::int64_t{131072}},
      {"llama.embedding_length", std::int64_t{896}},
      {"llama.attention.layer_norm_rms_epsilon", 1e-6},
      {"llama.attention.head_count", std::int64_t{14}},
      {"llama.attention.head_count_kv", std::int64_t{2}},
      {"llama.attention.key_length", std::int64_t{64}},
      {"llama.attention.value_length", std::int64_t{64}},
      {"llama.rope.dimension_count", std::int64_t{64}},
      {"llama.rope.freq_base", 500000.0},
      {"llama.rope.scaling.type", std::string("llama3")},
      {"llama.rope.scaling.factor", 32.0},
      {"llama.rope.scaling.low_freq_factor", 1.0},
      {"llama.rope.scaling.high_freq_factor", 4.0},
      {"llama.rope.scaling.original_context_length", std::int64_t{8192}},
      {"llama.feed_forward_length", std::int64_t{4864}},
  };
  auto spec = spec::Synthesize(spec::mappings::LlamaGgufMapping(), source, values);
  ASSERT_TRUE(spec) << spec.error();
  const auto& attn = spec->stages.front().layers.front().components.front();
  ASSERT_NE(attn.Find("llama3_rope"), nullptr);
  EXPECT_TRUE(std::get<bool>(attn.Find("llama3_rope")->value));
  ASSERT_NE(attn.Find("rope_factor"), nullptr);
  EXPECT_DOUBLE_EQ(std::get<double>(attn.Find("rope_factor")->value), 32.0);
  ASSERT_NE(attn.Find("low_freq_factor"), nullptr);
  EXPECT_DOUBLE_EQ(std::get<double>(attn.Find("low_freq_factor")->value), 1.0);
  ASSERT_NE(attn.Find("high_freq_factor"), nullptr);
  EXPECT_DOUBLE_EQ(std::get<double>(attn.Find("high_freq_factor")->value), 4.0);
  ASSERT_NE(attn.Find("yarn_context"), nullptr);
  EXPECT_EQ(std::get<std::int64_t>(attn.Find("yarn_context")->value), 8192);
}

TEST(ImportNormalizeTest, Gemma4NestedNativeMathFlagsFailClosed) {
  auto native = TinyGemma4(true);
  native.container = "safetensors";
  for (auto& tensor : native.tensors) {
    tensor.dtype = "F32";
    std::ranges::reverse(tensor.dims);
  }
  std::erase_if(native.metadata,
                [](const auto& entry) { return entry.key == "general.architecture"; });
  Value(native, "model_type", std::string("gemma4"));
  Value(native, "gemma4.layer_types",
        std::vector<std::string>{"sliding_attention", "sliding_attention", "sliding_attention",
                                 "sliding_attention", "sliding_attention", "full_attention"});
  Value(native, "text_config.use_bidirectional_attention", std::string("vision"));
  Value(native, "text_config.enable_moe_block", true);
  Value(native, "text_config.rope_parameters.full_attention.rope_type",
        std::string("proportional"));
  Value(native, "text_config.rope_parameters.sliding_attention.rope_type", std::string("default"));
  const auto valid = NormalizationBound(native);
  ASSERT_TRUE(valid) << test_support::Failed(valid, &Error::ToString).value_or("");
  for (const auto& [key, value] : std::array<std::pair<std::string_view, MetaValue>, 9>{
           {{"text_config.use_bidirectional_attention", true},
            {"text_config.use_bidirectional_attention", std::string("all")},
            {"text_config.rope_parameters.full_attention.rope_type", std::string("linear")},
            {"text_config.rope_parameters.full_attention.unknown_math", 1.0},
            {"text_config.attention_bias", true},
            {"text_config.hidden_activation", std::string("relu")},
            {"text_config.use_double_wide_mlp", true},
            {"text_config.enable_moe_block", false},
            {"text_config.attention_k_eq_v", false}}}) {
    SCOPED_TRACE(key);
    auto changed = native;
    Value(changed, std::string(key), value);
    EXPECT_FALSE(NormalizationBound(changed));
  }
  Value(native, "text_config.use_bidirectional_attention", false);
  EXPECT_TRUE(NormalizationBound(native));
  auto dense = TinyGemma4(false);
  dense.container = "safetensors";
  for (auto& tensor : dense.tensors) {
    tensor.dtype = "F32";
    std::ranges::reverse(tensor.dims);
  }
  std::erase_if(dense.metadata,
                [](const auto& entry) { return entry.key == "general.architecture"; });
  Value(dense, "model_type", std::string("gemma4"));
  Value(dense, "gemma4.layer_types",
        std::vector<std::string>{"sliding_attention", "sliding_attention", "sliding_attention",
                                 "sliding_attention", "sliding_attention", "full_attention"});
  Value(dense, "text_config.enable_moe_block", false);
  const auto dense_valid = NormalizationBound(dense);
  ASSERT_TRUE(dense_valid) << test_support::Failed(dense_valid, &Error::ToString).value_or("");
  Value(dense, "text_config.enable_moe_block", true);
  EXPECT_FALSE(NormalizationBound(dense));
}

}  // namespace
}  // namespace llmp::import
