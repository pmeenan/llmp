// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
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
#include "components/step.h"
#include "dsv4_gguf_normalization_fixture.h"
#include "expected_error.h"
#include "gemma2_fixture.h"
#include "gemma3_fixture.h"
#include "gemma4_fixture.h"
#include "import/normalize.h"
#include "import/readers/gguf.h"
#include "import_fixtures.h"
#include "model/dsv4.h"
#include "qwen38_gguf_normalization_fixture.h"
#include "spec/mappings/dsv4.h"
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
Checkpoint Tiny(bool gemma3 = false, bool untied = false) {
  const std::string arch = gemma3 ? "gemma3" : "gemma2";
  Checkpoint s{
      .container = "gguf", .architecture = arch, .files = {{.name = "memory", .bytes = 0}}};
  Value(s, "general.architecture", arch);
  for (const auto& [key, value] :
       std::array<std::pair<std::string_view, std::int64_t>, 10>{{{"block_count", 2},
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
  for (int layer = 0; layer < 2; ++layer) {
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
      auto plan = Normalize(std::move(input), kFunded);
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
  auto plan = Normalize(std::move(*parsed), kFunded);
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
  ASSERT_EQ(expired->auxiliary_namespaces().size(), 1U);
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
    EXPECT_FALSE(Normalize(std::move(source), kFunded));
  }
  auto source = Tiny();
  Value(source, "gemma2.rope.freq_base", std::int64_t{10000});
  Value(source, "gemma2.rope.freq_base_swa", 1000.0);
  EXPECT_TRUE(Normalize(std::move(source), kFunded));  // two distinct fallback keys
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
    EXPECT_FALSE(Normalize(std::move(s), kFunded)) << mode;
  }
  auto s = Tiny();
  Value(s, "vocab_count", std::int64_t{10});
  EXPECT_FALSE(Normalize(std::move(s), kFunded));
}

TEST(ImportNormalizeTest, RefusesUnimplementedDomainsAndAlternateQueryScale) {
  for (const auto container : {"safetensors", "mlx", "exl3", "diffusers"}) {
    auto s = Tiny();
    s.container = container;
    EXPECT_FALSE(Normalize(std::move(s), kFunded));
  }
  for (const bool g3 : {false, true}) {
    auto s = Tiny(g3);
    Value(s, s.architecture + ".block_count", std::int64_t{g3 ? 62 : 46});
    EXPECT_FALSE(Normalize(std::move(s), kFunded));
  }
  for (const int mode : {0, 1, 2, 3}) {
    auto s = Tiny();
    if (mode == 0) s.role = "target";
    if (mode == 1) s.quantization = "invented";
    if (mode == 2) s.components.emplace_back();
    if (mode == 3) s.skipped.push_back({"file", "outside"});
    EXPECT_FALSE(Normalize(std::move(s), kFunded));
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
    EXPECT_FALSE(Normalize(std::move(s), kFunded)) << mode;
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
  EXPECT_FALSE(Normalize(s, kFunded));
  s.files[1].inode = 4;
  EXPECT_TRUE(Normalize(std::move(s), kFunded));  // distinct physical file, same offset
}

TEST(ImportNormalizeTest, FormatAndComponentChecksGoBeyondNameCoverage) {
  auto s = Tiny();
  auto& last = s.tensors.back();
  ++last.format.shape.ne[1];
  ++last.dims[1];
  last.extent.bytes = (*format::PartsOf(last.format))[0].bytes;
  s.files[0].bytes = last.extent.offset + last.extent.bytes;
  ASSERT_TRUE(NormalizationBound(s));  // geometry/name coverage alone is insufficient
  const auto malformed = Normalize(std::move(s), kFunded);
  EXPECT_EQ(test_support::Failed(malformed, &Error::reason),
            "normalized tensors do not satisfy their components");
  s = Tiny();
  Value(s, "gemma2.attention.head_count", std::int64_t{3});
  EXPECT_FALSE(Normalize(std::move(s), kFunded));
  s = Tiny();
  Value(s, "gemma2.attention.sliding_window_pattern", std::int64_t{3});
  EXPECT_FALSE(Normalize(std::move(s), kFunded));
  s = Tiny(true);
  Value(s, "gemma3.rope.scaling.type", std::string("yarn"));
  EXPECT_FALSE(Normalize(std::move(s), kFunded));
}

TEST(ImportNormalizeTest, ExactReservationAndWorkLimitAreCheckedBeforeConstruction) {
  auto s = Tiny();
  const auto cost = NormalizationBound(s);
  ASSERT_TRUE(cost);
  EXPECT_FALSE(Normalize(s, {cost->metadata_bytes - 1, cost->work_units}));
  EXPECT_FALSE(Normalize(s, {cost->metadata_bytes, cost->work_units - 1}));
  EXPECT_TRUE(Normalize(std::move(s), {cost->metadata_bytes, cost->work_units}));
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
    EXPECT_FALSE(Normalize(std::move(s), kFunded));
  }
  auto s = Tiny();
  Value(s, "general.alignment", std::int64_t{3});
  EXPECT_FALSE(Normalize(std::move(s), kFunded));
  s = Tiny();
  Value(s, "general.description", std::string("unaltered provenance"));
  Value(s, "tokenizer.ggml.tokens", std::vector<std::string>{"a", "b"});
  auto p = Normalize(std::move(s), kFunded);
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
  auto plan = Normalize(s, kFunded);
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
  EXPECT_FALSE(Normalize(std::move(s), kFunded));
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
    EXPECT_FALSE(Normalize(*input, {bound->metadata_bytes - 1, bound->work_units}));
    EXPECT_FALSE(Normalize(*input, {bound->metadata_bytes, bound->work_units - 1}));
    auto plan = Normalize(std::move(*input), {bound->metadata_bytes, bound->work_units});
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
    EXPECT_FALSE(Normalize(std::move(s), kFunded)) << mode;
  }
  auto s = TinyGemma4(false);
  s.architecture = "gemma4-assistant";
  EXPECT_FALSE(Normalize(std::move(s), kFunded));
  s = TinyGemma4(false);
  Value(s, "gemma4.feed_forward_length", std::vector<std::int64_t>(6, 16));
  EXPECT_TRUE(Normalize(std::move(s), kFunded));
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
    auto p = Normalize(std::move(s), kFunded);
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
    auto p = Normalize(std::move(s), kFunded);
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
  auto plan = Normalize(input, {bound->metadata_bytes, bound->work_units});
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
  auto plan = Normalize(std::move(input), kFunded);
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

TEST(ImportNormalizeQwenTest, RejectsUnfundedConstructionAndClosedProfileMutations) {
  const auto input = test_support::Qwen38GgufCheckpoint();
  const auto bound = NormalizationBound(input);
  ASSERT_TRUE(bound);
  EXPECT_FALSE(Normalize(input, {bound->metadata_bytes - 1, bound->work_units}));
  EXPECT_FALSE(Normalize(input, {bound->metadata_bytes, bound->work_units - 1}));
  for (int mode = 0; mode < 12; ++mode) {
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
    }
    EXPECT_FALSE(Normalize(std::move(bad), kFunded));
  }
}

TEST(ImportNormalizeDsv4Test, CompleteCapturedTargetKeepsAllEncodingsAndLayerSemantics) {
  auto input = test_support::Dsv4GgufCheckpoint();
  auto cost = NormalizationBound(input);
  ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
  EXPECT_EQ(cost->layers, 43U);
  EXPECT_LT(cost->metadata_bytes, std::uint64_t{1} << 30U);
  auto plan = Normalize(input, {cost->metadata_bytes, cost->work_units});
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  ASSERT_EQ(plan->routes().size(), 1328U);
  ASSERT_EQ(plan->spec().stages.size(), 1U);
  EXPECT_EQ(plan->spec().stages[0].side.size(), 3U);
  EXPECT_EQ(plan->spec().stages[0].layers.size(), 43U);
  auto values = spec::mappings::Dsv4V0Values(model::Dsv4Flash());
  ASSERT_TRUE(values);
  auto expected = spec::Synthesize(spec::mappings::Dsv4Mapping(), plan->canonical(), *values);
  ASSERT_TRUE(expected) << test_support::Failed(expected).value_or("");
  EXPECT_EQ(plan->spec(), *expected);
  std::set<Encoding> encodings;
  for (std::size_t i = 0; i != input.tensors.size(); ++i) {
    EXPECT_EQ(plan->routes()[i].extent, input.tensors[i].extent);
    EXPECT_EQ(plan->canonical().tensors[i].format, input.tensors[i].format);
    encodings.insert(input.tensors[i].format.encoding);
  }
  EXPECT_EQ(encodings.size(), 10U);
  EXPECT_TRUE(plan->applied_folds().empty());
  EXPECT_FALSE(Normalize(input, {cost->metadata_bytes - 1, cost->work_units}));
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
    EXPECT_FALSE(Normalize(std::move(input), kFunded)) << mutation;
  }
}

}  // namespace
}  // namespace llmp::import
