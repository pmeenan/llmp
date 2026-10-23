// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <utility>
#include <variant>

#include "artifact_writer.h"
#include "base/check.h"
#include "base/json.h"
#include "cli/cli.h"
#include "components/step.h"
#include "dsv4_gguf_normalization_fixture.h"
#include "expected_error.h"
#include "format/tensor_format.h"
#include "image_family.h"
#include "import/archive_input.h"
#include "import/image_pipeline.h"
#include "import/normalize.h"
#include "import/reader.h"
#include "import/readers/gguf.h"
#include "import/readers/safetensors.h"
#include "import/sampling_defaults.h"
#include "import/stream.h"
#include "import/writer.h"
#include "import_fixtures.h"
#include "part_publication_fixture.h"
#include "qwen2_exl3_normalization_fixture.h"
#include "qwen2_gguf_normalization_fixture.h"
#include "qwen38_gguf_normalization_fixture.h"
#include "runtime/component_admission.h"
#include "runtime/model_settings.h"
#include "spec/image_spatial.h"
#include "spec/stored.h"

namespace llmp::import {
namespace {
using format::Encoding;
constexpr NormalizationLimits kFunded{std::uint64_t{2} << 30U, std::uint64_t{1} << 45U};
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

class ImportWriterTest : public fixtures::ScratchTest {
 protected:
  std::expected<NormalizedPlan, Error> Plan(bool moe = false) {
    Write(dir_ / "source/model.gguf", Gguf(moe ? TinyGemma4(true) : Tiny()));
    auto input = ReadGguf(dir_ / "source/model.gguf");
    if (!input) return std::unexpected(input.error());
    return Normalize(std::move(*input), kFunded);
  }
  WriteOptions Options(std::string_view store = "store") {
    return {.store = dir_ / store, .build = "test", .metadata_bytes = std::uint64_t{1} << 30U};
  }
};
TEST_F(ImportWriterTest, BuildProvenanceDoesNotChangeIdentityOrWeakenReceipts) {
  auto plan = Plan();
  ASSERT_TRUE(plan);
  auto opts = Options();
  opts.build = "unknown";
  auto first = WriteArtifact(*plan, opts);
  ASSERT_TRUE(first) << test_support::Failed(first, &Error::ToString).value_or("");
  const auto manifest = dir_ / "store" / first->id / "manifest.json";
  opts.build = "40bf23e3.dirty";
  auto again = WriteArtifact(*plan, opts);
  ASSERT_TRUE(again) << test_support::Failed(again, &Error::ToString).value_or("");
  EXPECT_EQ(again->id, first->id);
  EXPECT_TRUE(again->reused);
  auto opened = artifact::Artifact::Open(again->path);
  ASSERT_TRUE(opened);
  EXPECT_EQ(opened->converter().build, "unknown");
  opts.store = dir_ / "other";
  auto other = WriteArtifact(*plan, opts);
  ASSERT_TRUE(other);
  EXPECT_EQ(other->id, first->id);
  opened = artifact::Artifact::Open(other->path);
  ASSERT_TRUE(opened);
  EXPECT_EQ(opened->converter().build, opts.build);
  std::vector<std::optional<base::Sha256Digest>> wrong(plan->source().files.size(),
                                                       base::Sha256Digest{});
  opts.expected = wrong;
  EXPECT_FALSE(WriteArtifact(*plan, opts));
}

TEST_F(ImportWriterTest, ImportPreservesF16NormsAndDefersExecutionSupportToPlanning) {
  auto checkpoint = Tiny();
  auto& norm = checkpoint.tensors[1];
  ASSERT_EQ(norm.name, "output_norm.weight");
  norm.format.encoding = Encoding::kF16;
  norm.extent.bytes = format::PartsOf(norm.format)->Find(norm.part)->bytes;
  Write(dir_ / "source/fp16.gguf", Gguf(checkpoint));
  auto source = ReadGguf(dir_ / "source/fp16.gguf");
  ASSERT_TRUE(source);
  auto plan = Normalize(std::move(*source), kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  EXPECT_TRUE(components::ValidateSpec(plan->spec(), plan->canonical().tensors,
                                       components::ValidationPurpose::kImport));
  EXPECT_TRUE(components::ValidateSpec(plan->spec(), plan->canonical().tensors));
  auto published = WriteArtifact(*plan, Options());
  ASSERT_TRUE(published) << test_support::Failed(published, &Error::ToString).value_or("");
  auto opened = artifact::Artifact::Open(published->path);
  ASSERT_TRUE(opened);
  auto stored = spec::SourceOf(*opened);
  ASSERT_TRUE(stored);
  const auto tensor =
      std::ranges::find(stored->tensors, "output_norm.weight", &spec::SourceTensor::name);
  ASSERT_NE(tensor, stored->tensors.end());
  EXPECT_EQ(tensor->format.encoding, Encoding::kF16);
  auto admitted = runtime::ReadComponentAdmission(*opened);
  ASSERT_TRUE(admitted) << test_support::Failed(admitted).value_or("");
}

TEST_F(ImportWriterTest, HfSamplingDefaultsRoundTripFromAuthenticatedGenerationConfig) {
  for (const bool sample : {false, true}) {
    const auto source = dir_ / (sample ? "sample-source" : "greedy-source");
    auto config = fixtures::TinyQwen2HfConfig();
    config.pop_back();
    config += R"(,"temperature":0.4,"top_p":0.8,"top_k":12})";
    Write(source / "config.json", config);
    Write(source / "model.safetensors", fixtures::Shard(fixtures::TinyQwen2HfEntries()));
    const std::string generation =
        std::string(R"({"do_sample":)") + (sample ? "true" : "false") +
        R"(,"temperature":1.25,"top_p":0.92,"top_k":2147483647,"min_p":0.07})";
    Write(source / "generation_config.json", generation);
    auto input = ReadCheckpoint(source);
    ASSERT_TRUE(input) << test_support::Failed(input, &Error::ToString).value_or("");
    EXPECT_FALSE(ReadSamplingDefaults(*input, {}));
    auto bound = NormalizationBound(*input);
    ASSERT_TRUE(bound);
    auto plan = Normalize(std::move(*input), {bound->metadata_bytes, bound->work_units});
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    const auto& defaults = plan->spec().stages.front().sampling;
    EXPECT_EQ(defaults.temperature, sample ? 1.25 : 0.0);
    EXPECT_EQ(defaults.top_p, 0.92);
    EXPECT_EQ(defaults.top_k, 2147483647);
    EXPECT_EQ(defaults.min_p, 0.07);
    auto published = WriteArtifact(*plan, Options(sample ? "sample-store" : "greedy-store"));
    ASSERT_TRUE(published) << test_support::Failed(published, &Error::ToString).value_or("");
    const auto opened = artifact::Artifact::Open(published->path);
    ASSERT_TRUE(opened);
    const auto stored = spec::StoredSpec(*opened);
    ASSERT_TRUE(stored);
    EXPECT_EQ(stored->stages.front().sampling, defaults);
    const auto facts = runtime::ArtifactFactsOf(*opened, nullptr);
    ASSERT_TRUE(facts) << test_support::Failed(facts).value_or("");
    EXPECT_EQ(facts->temperature, defaults.temperature);
    EXPECT_EQ(facts->top_p, defaults.top_p);
    EXPECT_EQ(facts->top_k, defaults.top_k);
    EXPECT_EQ(facts->min_p, defaults.min_p);
  }
}

TEST_F(ImportWriterTest, HfFp8BlockWeightsPublishWithStoredFormatAndSpec) {
  const auto source = dir_ / "fp8-source";
  auto entries = fixtures::TinyQwen2HfEntries();
  for (auto& entry : entries) {
    if (entry.name == "unused.empty") continue;
    if (entry.name == "model.layers.0.self_attn.q_proj.weight") {
      entry = fixtures::E(entry.name, "F8_E4M3", {64, 64});
    } else {
      for (auto& dim : entry.shape) {
        if (dim == 10) continue;
        if (dim == 4) dim = 32;
        if (dim == 8) dim = 64;
        if (dim == 16) dim = 128;
      }
      entry.bytes = fixtures::DtypeBytes(entry.dtype);
      for (const auto dim : entry.shape) entry.bytes *= dim;
    }
  }
  entries.push_back(
      fixtures::E("model.layers.0.self_attn.q_proj.weight_scale_inv", "F32", {22, 22}));
  auto config = fixtures::TinyQwen2HfConfig();
  for (const auto& [from, to] : std::array<std::pair<std::string_view, std::string_view>, 3>{
           {{"\"hidden_size\":8", "\"hidden_size\":64"},
            {"\"intermediate_size\":16", "\"intermediate_size\":128"},
            {"\"head_dim\":4", "\"head_dim\":32"}}}) {
    const auto at = config.find(from);
    ASSERT_NE(at, std::string::npos);
    config.replace(at, from.size(), to);
  }
  const auto end = config.rfind('}');
  ASSERT_NE(end, std::string::npos);
  config.insert(end, R"(,"quantization_config":{"quant_method":"fp8","weight_block_size":[3,3]})");
  Write(source / "config.json", config);
  const auto shard = fixtures::Shard(entries);
  Write(source / "model.safetensors", shard);
  auto input = ReadCheckpoint(source);
  ASSERT_TRUE(input) << test_support::Failed(input, &Error::ToString).value_or("");
  auto bound = NormalizationBound(*input);
  ASSERT_TRUE(bound);
  auto plan = Normalize(std::move(*input), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  EXPECT_NE(std::ranges::find(plan->source().tensors, "model.layers.0.self_attn.q_proj.weight",
                              &Tensor::name),
            plan->source().tensors.end());
  EXPECT_NE(std::ranges::find(plan->source().tensors,
                              "model.layers.0.self_attn.q_proj.weight_scale_inv", &Tensor::name),
            plan->source().tensors.end());
  auto written = WriteArtifact(*plan, Options("fp8-store"));
  ASSERT_TRUE(written) << test_support::Failed(written, &Error::ToString).value_or("");
  const auto opened = artifact::Artifact::Open(written->path);
  ASSERT_TRUE(opened);
  const auto stored = spec::StoredSpec(*opened);
  ASSERT_TRUE(stored) << test_support::Failed(stored).value_or("");
  EXPECT_EQ(*stored, plan->spec());
  const auto artifact_source = spec::SourceOf(*opened);
  ASSERT_TRUE(artifact_source) << test_support::Failed(artifact_source).value_or("");
  const auto weight =
      std::ranges::find(artifact_source->tensors, "blk.0.attn_q.weight", &spec::SourceTensor::name);
  const auto scales = std::ranges::find(artifact_source->tensors, "blk.0.attn_q.weight_scale_inv",
                                        &spec::SourceTensor::name);
  ASSERT_NE(weight, artifact_source->tensors.end());
  ASSERT_NE(scales, artifact_source->tensors.end());
  EXPECT_EQ(weight->format.encoding, Encoding::kFp8Scaled);
  EXPECT_EQ(weight->format.params.block_rows, 3U);
  EXPECT_EQ(weight->format.params.block_cols, 3U);
  EXPECT_EQ(scales->format, weight->format);
  EXPECT_EQ(scales->part, format::PartRole::kBlockScales);
  EXPECT_TRUE(components::ValidateSpec(*stored, artifact_source->tensors));
}

TEST_F(ImportWriterTest, NestedSamplingDefaultsFillUnsetOuterValuesBeforeGenerationOverrides) {
  for (const auto [sample, override] :
       {std::pair{true, false}, std::pair{false, false}, std::pair{false, true}}) {
    const auto source = dir_ / std::format("source-{}-{}", sample, override);
    auto config = fixtures::TinyQwen2HfConfig();
    config.pop_back();
    config +=
        std::string(
            R"(,"temperature":1.2,"text_config":{"temperature":0.5,"top_p":0.81,"top_k":13,"min_p":0.12,"do_sample":)") +
        (sample ? "true}}" : "false}}");
    Write(source / "config.json", config);
    Write(source / "model.safetensors", fixtures::Shard(fixtures::TinyQwen2HfEntries()));
    if (override)
      Write(source / "generation_config.json",
            R"({"temperature":0.9,"top_p":0.72,"do_sample":true})");
    auto input = ReadCheckpoint(source);
    ASSERT_TRUE(input) << test_support::Failed(input, &Error::ToString).value_or("");
    const auto cost = SamplingDefaultsBound(*input);
    ASSERT_TRUE(cost);
    const auto defaults = ReadSamplingDefaults(*input, *cost, "text_config.");
    ASSERT_TRUE(defaults) << test_support::Failed(defaults, &Error::ToString).value_or("");
    EXPECT_EQ(defaults->temperature, override ? 0.9 : sample ? 1.2 : 0.0);
    EXPECT_EQ(defaults->top_p, override ? 0.72 : 0.81);
    EXPECT_EQ(defaults->top_k, 13);
    EXPECT_EQ(defaults->min_p, 0.12);
  }
}

TEST_F(ImportWriterTest, GgufSamplingDefaultsRoundTripWithoutKeptMetadataReparsing) {
  auto checkpoint = Tiny();
  Value(checkpoint, "general.sampling.temp", 0.75);
  Value(checkpoint, "general.sampling.top_p", 0.875);
  Value(checkpoint, "general.sampling.top_k", std::int64_t{32});
  Value(checkpoint, "general.sampling.min_p", 0.125);
  Write(dir_ / "source/sampling.gguf", Gguf(checkpoint));
  auto source = ReadGguf(dir_ / "source/sampling.gguf");
  ASSERT_TRUE(source);
  auto bound = NormalizationBound(*source);
  ASSERT_TRUE(bound);
  auto plan = Normalize(std::move(*source), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  const spec::SamplingDefaults expected{
      .temperature = 0.75, .top_p = 0.875, .top_k = 32, .min_p = 0.125};
  EXPECT_EQ(plan->spec().stages.front().sampling, expected);
  auto published = WriteArtifact(*plan, Options());
  ASSERT_TRUE(published) << test_support::Failed(published, &Error::ToString).value_or("");
  const auto opened = artifact::Artifact::Open(published->path);
  ASSERT_TRUE(opened);
  const auto stored = spec::StoredSpec(*opened);
  ASSERT_TRUE(stored);
  EXPECT_EQ(stored->stages.front().sampling, expected);
}

TEST_F(ImportWriterTest, SamplingDefaultsRefuseChangedRetainedDocumentBeforeInterpretation) {
  const auto source = dir_ / "source";
  Write(source / "config.json", fixtures::TinyQwen2HfConfig());
  Write(source / "model.safetensors", fixtures::Shard(fixtures::TinyQwen2HfEntries()));
  Write(source / "generation_config.json", R"({"temperature":0.5})");
  auto input = ReadCheckpoint(source);
  ASSERT_TRUE(input);
  const auto cost = SamplingDefaultsBound(*input);
  ASSERT_TRUE(cost);
  Write(source / "generation_config.json", R"({"temperature":1.5})");
  const auto changed = ReadSamplingDefaults(*input, *cost);
  ASSERT_FALSE(changed);
  EXPECT_EQ(changed.error().rule, Rule::kIo);
  EXPECT_FALSE(std::filesystem::exists(dir_ / "store"));
}

TEST_F(ImportWriterTest, HfPublicationKeepsTokenizerDocumentsAndUsesTheirChatTemplate) {
  const auto source = dir_ / "source";
  Write(source / "config.json", fixtures::TinyQwen2HfConfig());
  Write(source / "model.safetensors", fixtures::Shard(fixtures::TinyQwen2HfEntries()));
  constexpr std::string_view regex =
      R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
  std::string quoted;
  base::json::AppendQuoted(regex, quoted);
  const std::string tokenizer =
      R"({"added_tokens":[],"decoder":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","behavior":"Isolated","pattern":{"Regex":)" +
      quoted + R"(}},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]}})";
  Write(source / "tokenizer.json", tokenizer);
  Write(source / "tokenizer_config.json",
        R"({"chat_template":[{"name":"default","template":"config template"}]})");
  Write(source / "special_tokens_map.json", "{}");
  Write(source / "generation_config.json", R"({"eos_token_id":2})");
  Write(source / "tokenizer.model", "preserved model bytes");
  Write(source / "chat_template.json", R"({"chat_template":"JSON template"})");
  Write(source / "chat_template.jinja", "Jinja template");
  for (const auto expected : {"Jinja template", "JSON template", "config template"}) {
    auto checkpoint = ReadSafetensors(source);
    ASSERT_TRUE(checkpoint) << test_support::Failed(checkpoint, &Error::ToString).value_or("");
    auto plan = Normalize(std::move(*checkpoint), kFunded);
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    auto written = WriteArtifact(*plan, Options());
    ASSERT_TRUE(written) << test_support::Failed(written, &Error::ToString).value_or("");
    auto artifact = artifact::Artifact::Open(written->path);
    ASSERT_TRUE(artifact);
    EXPECT_EQ(artifact->ReadMetadata("tokenizer.json"), tokenizer);
    EXPECT_EQ(artifact->ReadMetadata("tokenizer.model"), "preserved model bytes");
    EXPECT_EQ(artifact->ReadMetadata("generation_config.json"), R"({"eos_token_id":2})");
    auto assets = runtime::ReadChatAssets(*artifact, {});
    ASSERT_TRUE(assets) << test_support::Failed(assets).value_or("");
    EXPECT_EQ(assets->from, "tokenizer.json");
    ASSERT_TRUE(assets->chat_template);
    EXPECT_EQ(*assets->chat_template, expected);
    std::filesystem::remove(source / (std::string(expected) == "Jinja template"
                                          ? "chat_template.jinja"
                                          : "chat_template.json"));
  }
  std::filesystem::remove(source / "tokenizer.json");
  auto checkpoint = ReadSafetensors(source);
  ASSERT_TRUE(checkpoint);
  auto plan = Normalize(std::move(*checkpoint), kFunded);
  ASSERT_TRUE(plan);
  auto written = WriteArtifact(*plan, Options());
  ASSERT_TRUE(written);
  auto artifact = artifact::Artifact::Open(written->path);
  ASSERT_TRUE(artifact);
  auto assets = runtime::ReadChatAssets(*artifact, {});
  ASSERT_FALSE(assets);
  EXPECT_NE(assets.error().find("tokenizer.model:"), std::string::npos);
}

TEST_F(ImportWriterTest, ReaderAcceptsLegacyNamedBuildManifestIdentity) {
  auto plan = Plan();
  ASSERT_TRUE(plan);
  auto opts = Options();
  opts.build = "40bf23e3.dirty";
  auto published = WriteArtifact(*plan, opts);
  ASSERT_TRUE(published);
  auto opened = artifact::Artifact::Open(published->path);
  ASSERT_TRUE(opened) << test_support::Failed(opened, &artifact::Error::ToString).value_or("");
  EXPECT_EQ(opened->converter().build, opts.build);
  std::ifstream in(published->path / "manifest.json", std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), {});
  const auto raw = base::ToHex(base::Sha256().Update(text).Finish());
  ASSERT_NE(raw, published->id);
  const auto legacy = opts.store / raw;
  std::filesystem::rename(published->path, legacy);
  EXPECT_FALSE(artifact::Artifact::Open(legacy));
}

TEST_F(ImportWriterTest, ConvertedNormReceiptsPreserveSourceBytesAndStoredSpec) {
  auto plan = Plan();
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  ASSERT_EQ(plan->applied_folds().size(), 1U);
  EXPECT_EQ(plan->applied_folds()[0].fold, "norm.plus_one");
  auto published = WriteArtifact(*plan, Options());
  ASSERT_TRUE(published) << test_support::Failed(published, &Error::ToString).value_or("");
  auto artifact = artifact::Artifact::Open(published->path);
  ASSERT_TRUE(artifact) << test_support::Failed(artifact, &artifact::Error::ToString).value_or("");
  EXPECT_TRUE(artifact->folds().empty());
  ASSERT_EQ(artifact->source_folds().size(), 1U);
  EXPECT_EQ(artifact->source_folds()[0].name, "norm.plus_one");
  EXPECT_EQ(
      artifact->source_folds()[0].tensors,
      (std::vector<std::string>{
          "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight", "blk.0.post_attention_norm.weight",
          "blk.0.post_ffw_norm.weight", "blk.1.attn_norm.weight", "blk.1.ffn_norm.weight",
          "blk.1.post_attention_norm.weight", "blk.1.post_ffw_norm.weight", "output_norm.weight"}));
  const auto stored = spec::StoredSpec(*artifact);
  ASSERT_TRUE(stored);
  EXPECT_EQ(*stored, plan->spec());
  const auto source = spec::SourceOf(*artifact);
  ASSERT_TRUE(source);
  EXPECT_TRUE(source->folds.empty());
  EXPECT_TRUE(components::ValidateFoldReceipts(*stored, 0, *source));
  EXPECT_TRUE(runtime::ReadComponentAdmission(*artifact));
  ASSERT_EQ(artifact->sources().size(), plan->source().files.size());
  EXPECT_EQ(artifact->sources()[0].name, plan->source().files[0].name);
  EXPECT_EQ(artifact->sources()[0].bytes.value(), plan->source().files[0].bytes);
}

TEST_F(ImportWriterTest, DeterministicPublicationAndEveryPayloadByte) {
  for (const bool moe : {false, true}) {
    auto plan = Plan(moe);
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    auto first = WriteArtifact(*plan, Options(moe ? "moe" : "dense"));
    ASSERT_TRUE(first) << test_support::Failed(first, &Error::ToString).value_or("");
    auto again = WriteArtifact(*plan, Options(moe ? "moe" : "dense"));
    ASSERT_TRUE(again) << test_support::Failed(again, &Error::ToString).value_or("");
    EXPECT_TRUE(again->reused);
    EXPECT_EQ(first->id, again->id);
    auto separate = WriteArtifact(*plan, Options(moe ? "moe-other" : "dense-other"));
    ASSERT_TRUE(separate);
    EXPECT_EQ(first->id, separate->id);
    auto artifact = artifact::Artifact::Open(first->path);
    ASSERT_TRUE(artifact);
    auto stored = spec::StoredSpec(*artifact);
    ASSERT_TRUE(stored);
    EXPECT_EQ(*stored, plan->spec());
    EXPECT_TRUE(artifact->folds().empty());
    std::uint64_t verified = 0;
    const auto compare = [&](std::string_view name, std::uint64_t source_delta,
                             const artifact::Placement& place) {
      const auto& tensors = plan->canonical().tensors;
      const auto tensor = std::ranges::find(tensors, name, &spec::SourceTensor::name);
      ASSERT_NE(tensor, tensors.end());
      const auto i = static_cast<std::size_t>(tensor - tensors.begin());
      std::vector<std::byte> expected(static_cast<std::size_t>(place.bytes.value()));
      ASSERT_TRUE(plan->source().Read(plan->routes()[i].extent, source_delta, expected));
      const auto& group = artifact->groups()[place.group];
      const auto& shard = artifact->shards()[group.shard];
      auto file = OpenCheckpointFile(first->path / shard.path, 0);
      ASSERT_TRUE(file);
      std::vector<std::byte> actual(expected.size());
      ASSERT_TRUE(ReadExactly(
          file->fd.get(), shard.data_offset.value() + group.offset.value() + place.offset.value(),
          actual, 0));
      EXPECT_EQ(expected, actual);
      verified += actual.size();
    };
    for (std::size_t i = 0; i < artifact->resources().size(); ++i) {
      auto place = artifact->ResourcePlacement(static_cast<std::uint32_t>(i));
      ASSERT_TRUE(place);
      compare(artifact->resources()[i].name, 0, *place);
    }
    for (std::size_t i = 0; i < artifact->expert_arrays().size(); ++i) {
      const auto& a = artifact->expert_arrays()[i];
      for (std::uint32_t e = 0; e < a.count; ++e) {
        auto place = artifact->SlicePlacement(static_cast<std::uint32_t>(i), e);
        ASSERT_TRUE(place);
        compare(a.name, e * a.slice_bytes.value(), *place);
      }
    }
    EXPECT_EQ(verified, plan->cost().payload_bytes);
  }
}
TEST_F(ImportWriterTest, ChangedParsedHeaderAndWrongArchiveHashNeverPublish) {
  auto plan = Plan();
  ASSERT_TRUE(plan);
  const auto& file = plan->source().files.front();
  auto bytes = Gguf(Tiny());
  bytes[32] ^= std::byte{1};
  Write(file.path, bytes);  // same inode and length; receipt must detect it
  auto bad = WriteArtifact(*plan, Options());
  EXPECT_FALSE(bad);
  EXPECT_FALSE(std::filesystem::exists(Options().store / "manifest.json"));
  for (const auto& entry : std::filesystem::directory_iterator(Options().store))
    EXPECT_EQ(entry.path().filename(), ".staging");
  plan = Plan();
  ASSERT_TRUE(plan);
  std::array<std::optional<base::Sha256Digest>, 1> hashes = {base::Sha256Digest{}};
  auto options = Options();
  options.expected = hashes;
  EXPECT_FALSE(WriteArtifact(*plan, options));
  for (const auto& entry : std::filesystem::directory_iterator(options.store))
    EXPECT_EQ(entry.path().filename(), ".staging");
}
TEST_F(ImportWriterTest, DamagedExistingArtifactRefusedAndUnfundedPlanHasNoSideEffects) {
  auto plan = Plan();
  ASSERT_TRUE(plan);
  auto options = Options();
  options.metadata_bytes = 0;
  EXPECT_FALSE(WriteArtifact(*plan, options));
  EXPECT_FALSE(std::filesystem::exists(options.store));
  auto result = WriteArtifact(*plan, Options());
  ASSERT_TRUE(result);
  Write(result->path / "data/00000.safetensors", "damaged");
  EXPECT_FALSE(WriteArtifact(*plan, Options()));
}
TEST_F(ImportWriterTest, MissingReceiptAndChangedFileIdentityAreRefused) {
  Write(dir_ / "source/model.gguf", Gguf(Tiny()));
  auto input = ReadGguf(dir_ / "source/model.gguf");
  ASSERT_TRUE(input);
  input->files.front().parsed = {};
  auto plan = Normalize(std::move(*input), kFunded);
  ASSERT_TRUE(plan);
  EXPECT_FALSE(WriteArtifact(*plan, Options()));
  plan = Plan();
  ASSERT_TRUE(plan);
  std::filesystem::rename(dir_ / "source/model.gguf", dir_ / "source/old.gguf");
  Write(dir_ / "source/model.gguf", Gguf(Tiny()));
  EXPECT_FALSE(WriteArtifact(*plan, Options()));
}
TEST_F(ImportWriterTest, StreamIsBoundedAndDetectsConcurrentMutationAndSinkFailure) {
  std::string payload((3U << 20U) + 17, 'a');
  Write(dir_ / "payload", payload);
  auto opened = OpenCheckpointFile(dir_ / "payload", 0);
  ASSERT_TRUE(opened);
  SourceFile source{
      .name = "payload",
      .path = opened->path,
      .bytes = opened->bytes,
      .device = opened->device,
      .inode = opened->inode,
      .parsed = {7, base::Sha256().Update(std::string_view(payload).substr(0, 7)).Finish()}};
  std::uint64_t copied = 0;
  auto digest = StreamSource(source, 0, [&](auto at, auto bytes) -> std::expected<void, Error> {
    EXPECT_EQ(at, copied);
    EXPECT_LE(bytes.size(), 1U << 20U);
    copied += bytes.size();
    return {};
  });
  ASSERT_TRUE(digest);
  EXPECT_EQ(copied, payload.size());
  EXPECT_EQ(*digest, base::Sha256().Update(payload).Finish());
  EXPECT_FALSE(StreamSource(source, 0, [](auto, auto) -> std::expected<void, Error> {
    return std::unexpected(Error{.rule = Rule::kIo, .reason = "test sink failure"});
  }));
  EXPECT_FALSE(StreamSource(source, 0, [&](auto at, auto) -> std::expected<void, Error> {
    if (at == 0) {
      std::ofstream changed(source.path, std::ios::binary | std::ios::app);
      changed << 'b';
    }
    return {};
  }));
}

std::string ArchiveManifestFiles(
    const std::vector<std::pair<std::string, std::vector<std::byte>>>& files,
    std::string_view absent = {}) {
  std::string entries, members;
  for (const auto& [name, payload] : files) {
    if (!entries.empty()) {
      entries += ',';
      members += ',';
    }
    entries += std::format(
        R"({{"path":"{}","size":{},"git_oid":null,"lfs":true,"sha256":"{}","present":{}}})", name,
        payload.size(), base::ToHex(base::Sha256().Update(payload).Finish()), name != absent);
    members += '"' + name + '"';
  }
  return std::format(
      R"({{"format":"llmp-archive-checkpoint","format_version":1,"origin":"hf","repo":"owner/model","commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","hub":{{}},"files":[{}],"variants":[{{"id":"model","files":[{}]}}]}})",
      entries, members);
}
std::string ArchiveManifest(std::span<const std::byte> payload) {
  return ArchiveManifestFiles({{"model.gguf", {payload.begin(), payload.end()}}});
}
TEST_F(ImportWriterTest, ArchivedCliPublishesVerifiedArtifactAndIndependentEvent) {
  const auto root = dir_ / "archive";
  const auto revision = root / "hf/owner/model@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const auto bytes = Gguf(Tiny());
  Write(root / "llmp-archive-root.json", R"({"format":"llmp-archive-root","format_version":1})");
  Write(revision / "model.gguf", bytes);
  Write(revision / "llmp-archive.json", ArchiveManifest(bytes));
  // An independently active checkpoint writer does not serialize import events.
  Write(root / ".llmp-writer/owner.json", "active-checkpoint-writer");
  const std::string requested = revision.string(), store = (dir_ / "store").string();
  const std::array<std::string_view, 4> args{"import", requested, "--store", store};
  std::FILE* output = std::tmpfile();
  std::FILE* error = std::tmpfile();
  ASSERT_NE(output, nullptr);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(cli::Run(args, output, error), cli::kExitOk);
  std::fclose(output);
  std::fclose(error);
  std::size_t count = 0;
  for (const auto& item : std::filesystem::directory_iterator(root / "catalog/events")) {
    ASSERT_EQ(item.path().extension(), ".json");
    auto file = OpenCheckpointFile(item.path(), 0);
    ASSERT_TRUE(file);
    auto text = ReadDocument(*file, 65536, 0);
    ASSERT_TRUE(text);
    const auto json = base::json::Parse(*text);
    ASSERT_TRUE(json);
    EXPECT_EQ(json->root().find("kind")->string(), "import");
    EXPECT_EQ(json->root().find("body")->find("variant")->string(), "model");
    ++count;
  }
  EXPECT_EQ(count, 1U);
}
TEST_F(ImportWriterTest, StagedCopiesVerifyButDoNotCreateAnArchiveCatalog) {
  auto plan = Plan();
  ASSERT_TRUE(plan);
  const auto bytes = Gguf(Tiny());
  Write(dir_ / "source/llmp-archive.json", ArchiveManifest(bytes));
  auto archive = FindArchiveInput(dir_ / "source/model.gguf");
  ASSERT_TRUE(archive && *archive);
  EXPECT_TRUE((**archive).root.empty());
  auto identities = ArchiveIdentities(**archive, plan->source());
  ASSERT_TRUE(identities);
  auto options = Options();
  options.expected = *identities;
  auto written = WriteArtifact(*plan, options);
  ASSERT_TRUE(written);
  auto artifact = artifact::Artifact::Open(written->path);
  ASSERT_TRUE(artifact);
  EXPECT_TRUE(RecordArchiveImport(**archive, plan->source(), *artifact));
  EXPECT_FALSE(std::filesystem::exists(dir_ / "source/catalog"));
}
TEST_F(ImportWriterTest, ArchiveVariantSubsetsAndMixedContainersNeverPublishOrRecord) {
  const auto root = dir_ / "archive";
  const auto revision = root / "hf/owner/model@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const auto bytes = Gguf(Tiny());
  Write(root / "llmp-archive-root.json", R"({"format":"llmp-archive-root","format_version":1})");
  Write(revision / "model.gguf", bytes);
  for (const std::string other : {"other.gguf", "other.safetensors"}) {
    for (const bool present : {false, true}) {
      if (present) Write(revision / other, bytes);
      Write(revision / "llmp-archive.json",
            ArchiveManifestFiles({{"model.gguf", bytes}, {other, bytes}}, present ? "" : other));
      // Both explicit-file and directory selection must reject the subset.
      for (const auto& selected : {revision, revision / "model.gguf"}) {
        const std::string requested = selected.string(), store = (dir_ / "store").string();
        const std::array<std::string_view, 4> args{"import", requested, "--store", store};
        std::FILE* output = std::tmpfile();
        std::FILE* error = std::tmpfile();
        ASSERT_NE(output, nullptr);
        ASSERT_NE(error, nullptr);
        EXPECT_EQ(cli::Run(args, output, error), cli::kExitFailure) << other << present;
        std::fclose(output);
        std::fclose(error);
        EXPECT_FALSE(std::filesystem::exists(dir_ / "store"));
        EXPECT_FALSE(std::filesystem::exists(root / "catalog"));
      }
    }
  }
}
TEST_F(ImportWriterTest, CompleteArchivedSplitPublishesToNewStoreHierarchy) {
  auto first = Tiny(), second = Tiny();
  const auto total = static_cast<std::int64_t>(first.tensors.size());
  first.tensors.erase(first.tensors.begin() + total / 2, first.tensors.end());
  second.tensors.erase(second.tensors.begin(), second.tensors.begin() + total / 2);
  for (auto* part : {&first, &second}) {
    Value(*part, "split.count", std::int64_t{2});
    Value(*part, "split.no", std::int64_t{part == &first ? 0 : 1});
    Value(*part, "split.tensors.count", total);
  }
  const auto root = dir_ / "archive";
  const auto revision = root / "hf/owner/model@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const std::vector<std::pair<std::string, std::vector<std::byte>>> files{
      {"model-00001-of-00002.gguf", Gguf(first)}, {"model-00002-of-00002.gguf", Gguf(second)}};
  Write(root / "llmp-archive-root.json", R"({"format":"llmp-archive-root","format_version":1})");
  for (const auto& [name, bytes] : files) Write(revision / name, bytes);
  Write(revision / "llmp-archive.json", ArchiveManifestFiles(files));
  const std::string requested = revision.string(), store = (dir_ / "new/deep/store").string();
  const std::array<std::string_view, 4> args{"import", requested, "--store", store};
  std::FILE* output = std::tmpfile();
  std::FILE* error = std::tmpfile();
  ASSERT_NE(output, nullptr);
  ASSERT_NE(error, nullptr);
  ASSERT_EQ(cli::Run(args, output, error), cli::kExitOk);
  std::fclose(output);
  std::fclose(error);
  std::size_t artifacts = 0;
  for (const auto& item : std::filesystem::directory_iterator(store)) {
    if (item.path().filename() == ".staging") continue;
    auto artifact = artifact::Artifact::Open(item.path());
    ASSERT_TRUE(artifact);
    EXPECT_EQ(artifact->sources().size(), 2U);
    ++artifacts;
  }
  EXPECT_EQ(artifacts, 1U);
  EXPECT_FALSE(std::filesystem::is_empty(root / "catalog/events"));
}
TEST_F(ImportWriterTest, DirectoryCreationRefusesMissingArchiveParentsAndBlockingFiles) {
  EXPECT_FALSE(CreateImportDirectory(dir_ / "missing/catalog", false));
  EXPECT_FALSE(std::filesystem::exists(dir_ / "missing"));
  Write(dir_ / "blocking", "regular file");
  EXPECT_FALSE(CreateImportDirectory(dir_ / "blocking/store", true));
  auto plan = Plan();
  ASSERT_TRUE(plan);
  EXPECT_FALSE(WriteArtifact(*plan, Options("blocking/store")));
  EXPECT_TRUE(CreateImportDirectory(dir_ / "nested/child", true));
  EXPECT_TRUE(CreateImportDirectory(dir_ / "nested/child", true));
  EXPECT_TRUE(std::filesystem::is_directory(dir_ / "nested/child"));
}
TEST_F(ImportWriterTest, ArchiveManifestRefusalsDoNotFallBackToUnverifiedImport) {
  const auto bytes = Gguf(Tiny());
  Write(dir_ / "source/model.gguf", bytes);
  const std::string valid = ArchiveManifest(bytes);
  for (const auto& [old_text, new_text] : std::vector<std::pair<std::string, std::string>>{
           {"\"format_version\":1", "\"format_version\":2"},
           {"\"path\":\"model.gguf\"", "\"path\":\"../model.gguf\""},
           {"\"size\":" + std::to_string(bytes.size()), "\"size\":true"},
           {"\"present\":true", "\"present\":1"},
           {"\"files\":[\"model.gguf\"]", "\"files\":[\"missing.gguf\"]"}}) {
    auto text = valid;
    auto at = text.find(old_text);
    ASSERT_NE(at, std::string::npos);
    text.replace(at, old_text.size(), new_text);
    Write(dir_ / "source/llmp-archive.json", text);
    EXPECT_FALSE(FindArchiveInput(dir_ / "source/model.gguf"));
  }
  Write(dir_ / "source/llmp-archive.json", valid);
  auto archive = FindArchiveInput(dir_ / "source/model.gguf");
  ASSERT_TRUE(archive && *archive);
  auto plan = Plan();
  ASSERT_TRUE(plan);
  auto& entry = (**archive).files.at("model.gguf");
  entry.present = false;
  EXPECT_FALSE(ArchiveIdentities(**archive, plan->source()));
  entry.present = true;
  ++entry.bytes;
  EXPECT_FALSE(ArchiveIdentities(**archive, plan->source()));
}
// A safetensors variant's weights are the shards its index names: the
// directory's other .safetensors files (ModelOpt's amax calibration files)
// are never read and never required; an index's shard left unread is.
TEST_F(ImportWriterTest, IndexedSafetensorsVariantsIgnoreUnindexedSidecars) {
  const auto revision = dir_ / "staged";
  const std::string index =
      R"({"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors",)"
      R"("b":"model-00002-of-00002.safetensors"}})";
  std::vector<std::pair<std::string, std::vector<std::byte>>> files;
  for (const std::string name :
       {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json", "amax.safetensors", "config.json"}) {
    std::vector<std::byte> bytes(16 + name.size(), std::byte{7});
    if (name.ends_with(".index.json"))
      bytes =
          std::vector<std::byte>(reinterpret_cast<const std::byte*>(index.data()),
                                 reinterpret_cast<const std::byte*>(index.data()) + index.size());
    Write(revision / name, bytes);
    files.emplace_back(name, bytes);
  }
  std::ranges::sort(files);  // the manifest lists paths sorted
  Write(revision / "llmp-archive.json", ArchiveManifestFiles(files));
  auto archive = FindArchiveInput(revision);
  ASSERT_TRUE(archive && *archive);
  const auto source_file = [&](const std::string& name) {
    const auto bytes = std::filesystem::file_size(revision / name);
    return SourceFile{.name = name, .path = revision / name, .bytes = bytes, .parsed = {bytes, {}}};
  };
  Checkpoint read{.container = "safetensors"};
  for (const std::string name :
       {"config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json"})
    read.files.push_back(source_file(name));
  EXPECT_TRUE(ArchiveIdentities(**archive, read));
  // Without its index, the sidecar is a weight member the import lacks.
  Checkpoint unindexed = read;
  unindexed.files.pop_back();
  EXPECT_FALSE(ArchiveIdentities(**archive, unindexed));
  // A shard the index names, unread, is still a subset.
  Checkpoint subset = read;
  subset.files.erase(subset.files.begin() + 2);
  EXPECT_FALSE(ArchiveIdentities(**archive, subset));
}
// A component's import event: every source the checkpoint holds as the
// archive has it, and the only other the caller's named input (its list).
TEST_F(ImportWriterTest, ComponentEventsAdmitOnlyTheCallersNamedSource) {
  const auto root = dir_ / "archive";
  const auto revision = root / "hf/owner/model@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  Write(root / "llmp-archive-root.json", R"({"format":"llmp-archive-root","format_version":1})");
  std::vector<std::pair<std::string, std::vector<std::byte>>> files;
  for (const std::string name : {"config.json", "model.safetensors"}) {
    std::vector<std::byte> bytes(8 + name.size(), std::byte{3});
    Write(revision / name, bytes);
    files.emplace_back(name, bytes);
  }
  Write(revision / "llmp-archive.json", ArchiveManifestFiles(files));
  auto archive = FindArchiveInput(revision);
  ASSERT_TRUE(archive && *archive);
  ASSERT_FALSE((**archive).root.empty());
  Checkpoint source{.container = "safetensors"};
  test::V1Plan plan;
  plan.container = "safetensors";
  plan.sources.clear();
  for (const auto& [name, bytes] : files) {
    source.files.push_back({.name = name,
                            .path = revision / name,
                            .bytes = bytes.size(),
                            .parsed = {bytes.size(), {}}});
    plan.sources.emplace_back(
        name, std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  }
  plan.sources.emplace_back("ids.txt", "1\n2\n");
  plan.groups = {
      test::V1Group{.kind = artifact::GroupKind::kTable,
                    .resources = {{.name = "token_embd.weight",
                                   .format = {.encoding = format::Encoding::kBF16,
                                              .layout = format::Layout::kDense,
                                              .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto published = artifact::Artifact::Open(test::Publish(dir_ / "artifact", test::Render(plan)));
  ASSERT_TRUE(published) << published.error().ToString();
  EXPECT_FALSE(RecordArchiveComponentImport(**archive, source, *published));
  EXPECT_FALSE(RecordArchiveComponentImport(**archive, source, *published, "other.txt"));
  EXPECT_TRUE(RecordArchiveComponentImport(**archive, source, *published, "ids.txt"));
}
TEST_F(ImportWriterTest, ArchivedLinksOutAndUnmanifestedRevisionsRefuse) {
  const auto root = dir_ / "archive";
  Write(root / "llmp-archive-root.json", R"({"format":"llmp-archive-root","format_version":1})");
  const auto bytes = Gguf(Tiny());
  Write(root / "hf/owner/model/model.gguf", bytes);
  EXPECT_FALSE(FindArchiveInput(root / "hf/owner/model/model.gguf"));
  Write(dir_ / "outside/model.gguf", bytes);
  const auto revision = dir_ / "stage";
  Write(revision / "llmp-archive.json", ArchiveManifest(bytes));
  std::filesystem::create_symlink(dir_ / "outside/model.gguf", revision / "model.gguf");
  auto archive = FindArchiveInput(revision / "model.gguf");
  ASSERT_TRUE(archive && *archive);
  auto input = ReadGguf(revision / "model.gguf");
  ASSERT_TRUE(input);
  EXPECT_FALSE(ArchiveIdentities(**archive, *input));
}
TEST_F(ImportWriterTest, CliRequiresExplicitStoreAndValidFunding) {
  std::FILE* output = std::tmpfile();
  std::FILE* error = std::tmpfile();
  ASSERT_NE(output, nullptr);
  ASSERT_NE(error, nullptr);
  const std::array<std::string_view, 2> missing{"import", "model.gguf"};
  EXPECT_EQ(cli::Run(missing, output, error), cli::kExitUsage);
  const std::array<std::string_view, 6> invalid{"import", "model.gguf",     "--store",
                                                "store",  "--metadata-mib", "0"};
  EXPECT_EQ(cli::Run(invalid, output, error), cli::kExitUsage);
  std::fclose(output);
  std::fclose(error);
}
TEST_F(ImportWriterTest, CliTakesTheMtpComponentAndItsDraftHeadOnlyTogether) {
  std::FILE* output = std::tmpfile();
  std::FILE* error = std::tmpfile();
  ASSERT_NE(output, nullptr);
  ASSERT_NE(error, nullptr);
  const std::array<std::string_view, 6> head_alone{"import", "model.gguf",   "--store",
                                                   "store",  "--draft-head", "ids.txt"};
  EXPECT_EQ(cli::Run(head_alone, output, error), cli::kExitUsage);
  const std::array<std::string_view, 6> other{"import", "model.gguf",  "--store",
                                              "store",  "--component", "vision"};
  EXPECT_EQ(cli::Run(other, output, error), cli::kExitUsage);
  const std::array<std::string_view, 8> twice{"import",      "model.gguf", "--store",     "store",
                                              "--component", "mtp",        "--component", "mtp"};
  EXPECT_EQ(cli::Run(twice, output, error), cli::kExitUsage);
  // Another checkpoint's drafter: refused, nothing published.
  Write(dir_ / "source/model.gguf", Gguf(Tiny()));
  const std::string requested = (dir_ / "source/model.gguf").string();
  const std::string store = (dir_ / "never-created").string();
  const std::array<std::string_view, 6> gguf{"import", requested,     "--store",
                                             store,    "--component", "mtp"};
  EXPECT_EQ(cli::Run(gguf, output, error), cli::kExitFailure);
  EXPECT_EQ(std::ftell(output), 0);
  EXPECT_FALSE(std::filesystem::exists(dir_ / "never-created"));
  std::fclose(output);
  std::fclose(error);
}
TEST_F(ImportWriterTest, CliRefusesIncompleteModelOptWithoutPublishingOrFallingBack) {
  Write(dir_ / "source/model.safetensors", fixtures::Shard({fixtures::E("weight", "F32", {4, 4})}));
  Write(
      dir_ / "source/config.json",
      R"({"model_type":"qwen3_8_flash_next","quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{}}})");
  auto input = ReadCheckpoint(dir_ / "source");
  ASSERT_TRUE(input) << test_support::Failed(input, &Error::ToString).value_or("");
  EXPECT_EQ(input->container, "safetensors");
  EXPECT_EQ(input->architecture, "qwen3_8_flash_next");
  EXPECT_EQ(input->quantization, "modelopt MIXED_PRECISION");
  const std::string store = (dir_ / "never-created").string();
  for (const auto& path : {dir_ / "source", dir_ / "source/model.safetensors"}) {
    const std::string requested = path.string();
    const std::array<std::string_view, 6> args{"import", requested,        "--store",
                                               store,    "--metadata-mib", "512"};
    std::FILE* output = std::tmpfile();
    std::FILE* error = std::tmpfile();
    ASSERT_NE(output, nullptr);
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(cli::Run(args, output, error), cli::kExitFailure);
    EXPECT_EQ(std::ftell(output), 0);
    std::rewind(error);
    std::array<char, 1024> text{};
    const auto count = std::fread(text.data(), 1, text.size(), error);
    EXPECT_GT(count, 0U);
    EXPECT_NE(std::string_view(text.data(), count).find("llmp import:"), std::string_view::npos);
    std::fclose(output);
    std::fclose(error);
    EXPECT_FALSE(std::filesystem::exists(store));
  }
}

TEST_F(ImportWriterTest, SafetensorsAndConfigReceiptsDescribeExactlyParsedBytes) {
  const auto shard = fixtures::Shard({fixtures::E("weight", "F32", {4, 4})});
  const std::string config = R"({"model_type":"test"})";
  Write(dir_ / "source/model.safetensors", shard);
  Write(dir_ / "source/config.json", config);
  auto checkpoint = ReadSafetensors(dir_ / "source");
  ASSERT_TRUE(checkpoint) << test_support::Failed(checkpoint, &Error::ToString).value_or("");
  for (const auto& file : checkpoint->files) {
    EXPECT_GT(file.parsed.bytes, 0U);
    EXPECT_TRUE(StreamSource(file, 0, [](auto, auto) -> std::expected<void, Error> { return {}; }));
    if (file.name == "config.json") {
      EXPECT_EQ(file.parsed.bytes, config.size());
      EXPECT_EQ(file.parsed.sha256, base::Sha256().Update(config).Finish());
    } else {
      EXPECT_LT(file.parsed.bytes, file.bytes);
      EXPECT_EQ(file.parsed.sha256,
                base::Sha256().Update(std::span(shard).first(file.parsed.bytes)).Finish());
    }
  }
}

// The source adapter remains closed to full pinned ModelOpt geometry. These
// private tiny products exercise only the shared publisher and component spec.
struct PartCase {
  NormalizedPlan plan;
  std::map<std::string, std::vector<std::byte>> expected;
};
class ImportPartWriterTest : public ImportWriterTest {
 protected:
  // `poison`: an undefined block scale (format::ScaleCode) written into
  // one scale byte of every source part of that encoding.
  std::expected<PartCase, Error> Parts(
      std::optional<std::pair<Encoding, std::uint8_t>> poison = std::nullopt) {
    auto tiny = TinyGemma4(true);
    auto old_tensors = std::move(tiny.tensors);
    tiny.tensors.clear();
    tiny.files[0].bytes = 0;
    for (const auto key : {"embedding_length", "expert_feed_forward_length"})
      Value(tiny, "gemma4." + std::string(key), std::int64_t{32});
    Value(tiny, "gemma4.feed_forward_length", std::int64_t{64});
    for (const auto key :
         {"attention.key_length", "attention.value_length", "attention.key_length_swa",
          "attention.value_length_swa", "rope.dimension_count", "rope.dimension_count_swa"})
      Value(tiny, "gemma4." + std::string(key), std::int64_t{16});
    for (const auto& old : old_tensors) {
      auto dims = old.dims;
      for (std::size_t d = 0; d < dims.size(); ++d) {
        const bool expert_count = (old.experts && d == 2) ||
                                  (old.name.ends_with("ffn_gate_inp.weight") && d == 1) ||
                                  old.name.ends_with("ffn_down_exps.scale");
        if (expert_count) continue;
        if (dims[d] == 8)
          dims[d] = 32;
        else if (dims[d] == 16)
          dims[d] = 64;
        else if (dims[d] == 4)
          dims[d] = 16;
        else if (old.name == "rope_freqs.weight")
          dims[d] = 8;
      }
      auto format = old.format;
      format.shape = *format::ShapeOf(dims);
      const auto bytes = (*format::PartsOf(format))[0].bytes;
      const auto at = (tiny.files[0].bytes + 31) / 32 * 32;
      tiny.tensors.push_back({.name = old.name,
                              .format = format,
                              .experts = old.experts,
                              .extent = {.offset = at, .bytes = bytes},
                              .dtype = "F32",
                              .dims = dims});
      tiny.files[0].bytes = at + bytes;
    }
    Write(dir_ / "template/model.gguf", Gguf(tiny));
    auto parsed = ReadGguf(dir_ / "template/model.gguf");
    if (!parsed) return std::unexpected(parsed.error());
    auto normalized = Normalize(std::move(*parsed), kFunded);
    if (!normalized) return std::unexpected(normalized.error());
    spec::Source canonical = normalized->canonical();
    canonical.tensors.clear();
    auto spec = normalized->spec();
    Checkpoint source{
        .container = "safetensors", .architecture = "gemma4", .files = {{.name = "weights.bin"}}};
    std::vector<std::byte> bytes((1U << 20U) - 3, std::byte{0});
    const std::string prefix = "part-v1\n";
    std::copy(std::as_bytes(std::span(prefix)).begin(), std::as_bytes(std::span(prefix)).end(),
              bytes.begin());
    std::vector<TensorAssembly> outputs;
    std::vector<SourceDisposition> dispositions;
    std::vector<NormalizedPartBinding> bindings;
    std::vector<PlannedPartFold> folds{{"gdn.value_heads_tiled", {}}};
    std::map<std::string, std::vector<std::byte>> expected;
    std::map<std::string, std::vector<std::pair<std::string, format::PartRole>>> names;
    unsigned seed = 1;
    const auto input = [&](const std::string& name, const format::TensorFormat& format,
                           format::PartRole part, SourceDisposition disposition) {
      const auto extent = format::PartsOf(format)->Find(part)->bytes;
      const auto index = static_cast<std::uint32_t>(source.tensors.size());
      source.tensors.push_back({.name = name,
                                .format = format,
                                .part = part,
                                .extent = {.offset = bytes.size(), .bytes = extent}});
      dispositions.push_back(disposition);
      std::vector<std::byte> payload(static_cast<std::size_t>(extent));
      for (std::size_t i = 0; i < payload.size(); ++i)
        payload[i] = static_cast<std::byte>((seed * 13 + i * 29) % 251);
      ++seed;
      // Block scales hold finite scales, as a checkpoint's (format::ScaleCode).
      const auto scales = format::ScaleBytesOf(format, part);
      if (scales.code != format::ScaleCode::kNone) {
        for (std::byte& b : payload)
          b = static_cast<std::byte>(std::to_integer<unsigned>(b) %
                                     (scales.code == format::ScaleCode::kUe4m3 ? 0x7FU : 0xFFU));
        if (poison && poison->first == format.encoding && payload.size() > 3)
          payload[3] = std::byte{poison->second};
      }
      bytes.insert(bytes.end(), payload.begin(), payload.end());
      bytes.insert(bytes.end(), 7, std::byte{0xB5});
      return std::pair(index, std::move(payload));
    };
    // Put the permuted BF16 weight first physically so the 1 MiB source stream
    // splits it after exactly 3 bytes, independently of numeric output order.
    auto logical = normalized->canonical().tensors;
    // A genuine GDN fragment governs the permutation receipt. The surrounding
    // Gemma-style expert fixture still exercises stacking/gather/scatter paths.
    spec::ComponentRef gdn{.kind = "attention.gdn",
                           .version = 1,
                           .params = {{"a_log", true},
                                      {"conv", std::int64_t{2}},
                                      {"eps", 1e-6},
                                      {"head_dim", std::int64_t{8}},
                                      {"k_heads", std::int64_t{2}},
                                      {"v_heads", std::int64_t{4}}}};
    const auto add_gdn = [&](std::string role, std::initializer_list<std::uint64_t> shape) {
      const auto name = "gdn." + role;
      logical.push_back(
          {.name = name,
           .format = {.encoding = Encoding::kF32,
                      .shape = *format::ShapeOf(std::span(shape.begin(), shape.size()))}});
      gdn.weights.push_back({.role = std::move(role), .tensor = name});
    };
    add_gdn("qkv", {32, 64});
    add_gdn("gate", {32, 32});
    add_gdn("alpha", {32, 4});
    add_gdn("beta", {32, 4});
    add_gdn("a", {4});
    add_gdn("dt_bias", {4});
    add_gdn("conv", {2, 64});
    add_gdn("norm", {8});
    add_gdn("output", {32, 32});
    spec.stages[0].layers[0].components.push_back(std::move(gdn));
    const auto query = std::ranges::find(logical, "gdn.gate", &spec::SourceTensor::name);
    if (query == logical.end()) return std::unexpected(Error{.reason = "missing fixture query"});
    std::rotate(logical.begin(), query, query + 1);
    for (const auto& old : logical) {
      auto format = old.format;
      const bool permute = old.name.starts_with("gdn.") && old.name != "gdn.norm";
      const bool columns =
          old.name == "gdn.a" || old.name == "gdn.dt_bias" || old.name == "gdn.output";
      const bool scalar = old.name == "gdn.a" || old.name == "gdn.dt_bias" ||
                          old.name == "gdn.alpha" || old.name == "gdn.beta";
      const bool qk_prefix = old.name == "gdn.qkv" || old.name == "gdn.conv";
      const bool quant_experts = old.name == "blk.0.ffn_gate_up_exps.weight";
      const bool quant_dense = old.name == "blk.0.ffn_gate.weight";
      const bool quant_mx = old.name == "blk.0.ffn_up.weight" || old.name == "gdn.qkv";
      if (old.name == "gdn.gate") format.encoding = Encoding::kBF16;
      if (quant_experts || quant_dense) {
        format.encoding = Encoding::kNvfp4;
        format.layout = format::Layout::kSplit;
        format.params = {.tensor_scale = quant_dense};
      }
      if (quant_mx) {
        format.encoding = Encoding::kMxfp8;
        format.layout = format::Layout::kSplit;
        format.params = {};
      }
      TensorAssembly assembly{.name = old.name, .format = format};
      if (permute) {
        auto permutation =
            HeadPermutationPlan::Make(format, columns ? HeadAxis::kColumns : HeadAxis::kRows, 2, 4,
                                      scalar ? 1 : 8, qk_prefix ? 32 : 0);
        if (!permutation) return std::unexpected(permutation.error());
        assembly.permutation = *permutation;
        folds[0].outputs.push_back(static_cast<std::uint32_t>(outputs.size()));
      }
      const auto format_parts = format::PartsOf(format);
      if (!format_parts) return std::unexpected(Error{.reason = "fixture part geometry"});
      for (const auto& part : format_parts->parts()) {
        const auto name = (quant_experts || quant_dense || quant_mx)
                              ? old.name + "." + std::string(format::PartRoleName(part.role))
                              : old.name;
        bindings.push_back({static_cast<std::uint32_t>(outputs.size()), part.role});
        canonical.tensors.push_back(
            {.name = name, .format = format, .part = part.role, .experts = old.experts});
        names[old.name].push_back({name, part.role});
        PartAssembly member{.part = part.role,
                            .kind = quant_dense && part.role == format::PartRole::kTensorScale
                                        ? PartAssemblyKind::kSharedTensorScale
                                        : PartAssemblyKind::kCopy};
        auto& wanted = expected[name];
        if (quant_experts) {
          member.kind = PartAssemblyKind::kMatrices;
          auto slice = format;
          slice.shape.rank = 2;
          slice.shape.ne[2] = 1;
          std::array<std::vector<std::byte>, 4> payloads;
          member.sources.resize(4);
          // Physical source order is the reverse of the declared numeric gather.
          for (std::uint32_t e = 4; e > 0; --e) {
            auto [index, payload] =
                input(name + ".expert" + std::to_string(e - 1), slice, part.role, {});
            member.sources[e - 1] = index;
            payloads[e - 1] = std::move(payload);
          }
          for (const auto& payload : payloads)
            wanted.insert(wanted.end(), payload.begin(), payload.end());
        } else if (quant_dense && part.role != format::PartRole::kTensorScale) {
          // The one shared tensor scale belongs to a two-shard row gather;
          // whole single-source copies use kCopy for every part.
          member.kind = PartAssemblyKind::kRows;
          auto slice = format;
          slice.shape.ne[1] /= 2;
          std::array<std::vector<std::byte>, 2> payloads;
          member.sources.resize(2);
          for (std::uint32_t shard = 2; shard > 0; --shard) {
            auto [index, payload] =
                input(name + ".row" + std::to_string(shard - 1), slice, part.role, {});
            member.sources[shard - 1] = index;
            payloads[shard - 1] = std::move(payload);
          }
          for (const auto& payload : payloads)
            wanted.insert(wanted.end(), payload.begin(), payload.end());
        } else {
          auto [index, payload] = input(name, format, part.role, {});
          member.sources.push_back(index);
          wanted = std::move(payload);
          if (permute) {
            const auto original = wanted;
            // Independent oracle: groups [0,1] [2,3] become [0,2,1,3].
            // Preserve Q/K prefix rows; column permutations repeat per row.
            const auto rows = columns ? (format.shape.rank == 1 ? 1U : format.shape.ne[1]) : 1U;
            const auto row_bytes = wanted.size() / rows;
            const auto prefix_bytes = qk_prefix ? wanted.size() / 2 : 0;
            const auto head_bytes = (row_bytes - prefix_bytes) / 4;
            for (std::size_t row = 0; row < rows; ++row)
              for (std::size_t destination = 0; destination < 4; ++destination) {
                constexpr std::array<std::size_t, 4> order{0, 2, 1, 3};
                const auto begin = row * row_bytes + prefix_bytes;
                std::copy_n(
                    original.begin() +
                        static_cast<std::ptrdiff_t>(begin + order[destination] * head_bytes),
                    head_bytes,
                    wanted.begin() + static_cast<std::ptrdiff_t>(begin + destination * head_bytes));
              }
          }
        }
        assembly.parts.push_back(std::move(member));
      }
      outputs.push_back(std::move(assembly));
    }
    const auto rebind = [&](spec::ComponentRef& component) {
      std::vector<spec::WeightBinding> weights;
      for (const auto& binding : component.weights)
        for (const auto& [name, part] : names.at(binding.tensor)) {
          auto changed = binding;
          changed.tensor = name;
          changed.part = part;
          weights.push_back(std::move(changed));
        }
      component.weights = std::move(weights);
    };
    for (auto& stage : spec.stages) {
      stage.folds.push_back("gdn.value_heads_tiled");
      if (stage.embedding) rebind(*stage.embedding);
      if (stage.stream.mixing) rebind(*stage.stream.mixing);
      for (auto& side : stage.side) rebind(side.source);
      for (auto& layer : stage.layers)
        for (auto& component : layer.components) rebind(component);
      if (stage.heads.final_norm) rebind(*stage.heads.final_norm);
      if (stage.heads.output) rebind(*stage.heads.output);
      for (auto& component : stage.heads.mtp) rebind(component);
    }
    spec::Canonicalize(spec);
    std::array<SemanticReceipt, 3> semantic{};
    for (std::size_t i = 0; i < semantic.size(); ++i) {
      const auto elements = i == 2 ? 66U : 2U;
      const std::array<std::uint64_t, 1> ne{elements};
      const format::TensorFormat format{.encoding = Encoding::kF32, .shape = *format::ShapeOf(ne)};
      auto [index, payload] =
          input("semantic." + std::to_string(i), format, format::PartRole::kData,
                {.kind = SourceDispositionKind::kSemantic, .rule = 1});
      semantic[i] = {index, source.tensors[index].extent, base::Sha256().Update(payload).Finish()};
    }
    Write(dir_ / "parts/weights.bin", bytes);
    auto opened = OpenCheckpointFile(dir_ / "parts/weights.bin", 0);
    if (!opened) return std::unexpected(opened.error());
    source.files[0] = {.name = "weights.bin",
                       .path = opened->path,
                       .bytes = opened->bytes,
                       .device = opened->device,
                       .inode = opened->inode,
                       .parsed = {prefix.size(), base::Sha256().Update(prefix).Finish()}};
    auto cost = IndexedPartPlan::Bound(source, outputs);
    if (!cost) return std::unexpected(cost.error());
    auto parts = IndexedPartPlan::Make(std::move(source), outputs, dispositions, *cost);
    if (!parts) return std::unexpected(parts.error());
    return PartCase{test_support::PartPublicationFixture::Make(
                        std::move(*parts), std::move(canonical), std::move(spec),
                        std::move(bindings), semantic, std::move(folds)),
                    std::move(expected)};
  }
};

TEST_F(ImportPartWriterTest, PermutationMultipartAndWholeExpertScatterAreExact) {
  auto fixture = Parts();
  ASSERT_TRUE(fixture) << test_support::Failed(fixture, &Error::ToString).value_or("");
  auto result = WriteArtifact(fixture->plan, Options());
  ASSERT_TRUE(result) << test_support::Failed(result, &Error::ToString).value_or("");
  auto opened = artifact::Artifact::Open(result->path);
  ASSERT_TRUE(opened);
  auto stored = spec::StoredSpec(*opened);
  ASSERT_TRUE(stored);
  EXPECT_EQ(*stored, fixture->plan.spec());
  ASSERT_EQ(opened->folds().size(), 1U);
  EXPECT_EQ(opened->folds()[0].name, "gdn.value_heads_tiled");
  EXPECT_EQ(opened->folds()[0].tensors,
            (std::vector<std::string>{"gdn.a", "gdn.alpha", "gdn.beta", "gdn.conv", "gdn.dt_bias",
                                      "gdn.gate", "gdn.output", "gdn.qkv.block_scales",
                                      "gdn.qkv.codes"}));
  std::size_t total = 0;
  const auto compare = [&](std::string_view name, std::uint64_t offset,
                           const artifact::Placement& place) {
    const auto& group = opened->groups()[place.group];
    const auto& shard = opened->shards()[group.shard];
    auto file = OpenCheckpointFile(result->path / shard.path, 0);
    ASSERT_TRUE(file);
    std::vector<std::byte> actual(static_cast<std::size_t>(place.bytes.value()));
    ASSERT_TRUE(ReadExactly(file->fd.get(),
                            shard.data_offset.value() + group.offset.value() + place.offset.value(),
                            actual, 0));
    const auto& expected = fixture->expected.at(std::string(name));
    ASSERT_LE(offset + actual.size(), expected.size());
    EXPECT_TRUE(std::ranges::equal(
        actual, std::span(expected).subspan(static_cast<std::size_t>(offset), actual.size())))
        << name;
    total += actual.size();
  };
  for (std::size_t i = 0; i < opened->resources().size(); ++i) {
    auto placement = opened->ResourcePlacement(static_cast<std::uint32_t>(i));
    ASSERT_TRUE(placement);
    compare(opened->resources()[i].name, 0, *placement);
  }
  for (std::size_t i = 0; i < opened->expert_arrays().size(); ++i)
    for (std::uint32_t expert = 0; expert < opened->expert_arrays()[i].count; ++expert) {
      auto placement = opened->SlicePlacement(static_cast<std::uint32_t>(i), expert);
      ASSERT_TRUE(placement);
      compare(opened->expert_arrays()[i].name,
              expert * opened->expert_arrays()[i].slice_bytes.value(), *placement);
    }
  std::size_t expected_bytes = 0;
  for (const auto& [name, payload] : fixture->expected) expected_bytes += payload.size();
  EXPECT_EQ(total, expected_bytes);
  auto again = WriteArtifact(fixture->plan, Options());
  ASSERT_TRUE(again);
  EXPECT_TRUE(again->reused);
  EXPECT_EQ(again->id, result->id);
}
TEST_F(ImportPartWriterTest, SemanticMutationAndUnfundedOrMalformedOwnersCannotPublish) {
  for (const int mutation : {0, 1, 2, 3, 4, 5, 6}) {
    SCOPED_TRACE(mutation);
    auto fixture = Parts();
    ASSERT_TRUE(fixture) << test_support::Failed(fixture, &Error::ToString).value_or("");
    auto options = Options("bad" + std::to_string(mutation));
    if (mutation == 0) options.metadata_bytes = 0;
    if (mutation == 1) test_support::PartPublicationFixture::WrongBinding(fixture->plan);
    if (mutation == 2) test_support::PartPublicationFixture::MissingSemantic(fixture->plan);
    if (mutation == 3) test_support::PartPublicationFixture::WrongSpec(fixture->plan);
    if (mutation == 5) test_support::PartPublicationFixture::WrongFoldStage(fixture->plan);
    if (mutation == 6) test_support::PartPublicationFixture::DuplicateFoldOutput(fixture->plan);
    if (mutation == 4) {
      const auto& source = fixture->plan.parts().source();
      const auto& receipt = fixture->plan.semantic()[0];
      std::fstream file(source.files[receipt.extent.file].path,
                        std::ios::in | std::ios::out | std::ios::binary);
      ASSERT_TRUE(file);
      file.seekg(static_cast<std::streamoff>(receipt.extent.offset));
      const auto original = file.get();
      ASSERT_NE(original, std::char_traits<char>::eof());
      file.seekp(static_cast<std::streamoff>(receipt.extent.offset));
      file.put(static_cast<char>(original ^ 1));
      file.close();
    }
    const auto result = WriteArtifact(fixture->plan, options);
    EXPECT_FALSE(result);
    if (mutation != 4) EXPECT_FALSE(std::filesystem::exists(options.store));
    if (std::filesystem::exists(options.store))
      for (const auto& entry : std::filesystem::directory_iterator(options.store))
        EXPECT_EQ(entry.path().filename(), ".staging");
  }
}
// A block scale that readers decode apart (NVFP4's UE4M3 past 0x7E, an
// E8M0 0xFF: format::ScaleCode) is refused as it streams; nothing publishes.
TEST_F(ImportPartWriterTest, UndefinedBlockScalesCannotPublish) {
  for (const auto& [encoding, byte] :
       std::vector<std::pair<Encoding, std::uint8_t>>{{Encoding::kNvfp4, 0x7F},
                                                      {Encoding::kNvfp4, 0x80},
                                                      {Encoding::kNvfp4, 0xFF},
                                                      {Encoding::kMxfp8, 0xFF}}) {
    SCOPED_TRACE(std::format("{} {:#04x}", format::Info(encoding).name, byte));
    auto fixture = Parts(std::pair(encoding, byte));
    ASSERT_TRUE(fixture) << test_support::Failed(fixture, &Error::ToString).value_or("");
    auto options = Options(std::format("scale{}{}", static_cast<int>(encoding), byte));
    const auto result = WriteArtifact(fixture->plan, options);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().rule, Rule::kQuantization) << result.error().ToString();
    EXPECT_EQ(result.error().kind, ItemKind::kByte);
    if (std::filesystem::exists(options.store))
      for (const auto& entry : std::filesystem::directory_iterator(options.store))
        EXPECT_EQ(entry.path().filename(), ".staging");
  }
  // The finite extremes publish.
  for (const auto& [encoding, byte] : std::vector<std::pair<Encoding, std::uint8_t>>{
           {Encoding::kNvfp4, 0x7E}, {Encoding::kMxfp8, 0xFE}, {Encoding::kNvfp4, 0x00}}) {
    SCOPED_TRACE(std::format("{} {:#04x}", format::Info(encoding).name, byte));
    auto fixture = Parts(std::pair(encoding, byte));
    ASSERT_TRUE(fixture);
    const auto result = WriteArtifact(
        fixture->plan, Options(std::format("fine{}{}", static_cast<int>(encoding), byte)));
    EXPECT_TRUE(result) << test_support::Failed(result, &Error::ToString).value_or("");
  }
}
TEST_F(ImportWriterTest, QwenSideTablePlacementAndCompleteExpertCoverageAreFundedBeforeEffects) {
  auto input = test_support::Qwen38GgufCheckpoint();
  const auto cost = NormalizationBound(input);
  ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
  auto plan = Normalize(std::move(input), {cost->metadata_bytes, cost->work_units});
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  constexpr std::uint64_t funding = std::uint64_t{2} << 30U;
  const auto layout = publication_detail::InspectPlacement(*plan, funding);
  ASSERT_TRUE(layout) << test_support::Failed(layout, &Error::ToString).value_or("");
  EXPECT_EQ(layout->copies, 1224U);
  EXPECT_EQ(layout->destinations, 1224U + 48U * 3U * 511U);
  EXPECT_EQ(layout->tables, 2U);      // resident embedding and independently paged side table
  EXPECT_EQ(layout->row_tables, 2U);  // embedding plus side.ngram, unchanged encoding
  EXPECT_EQ(layout->globals, 0U);     // residual.hc mixing has no weights
  EXPECT_EQ(layout->layers, 48U);
  EXPECT_EQ(layout->heads, 1U);
  EXPECT_EQ(layout->expert_groups, 48U * 512U);
  EXPECT_FALSE(publication_detail::InspectPlacement(*plan, layout->metadata_bytes - 1));
  const auto destination = dir_ / "qwen-unfunded";
  auto written = WriteArtifact(
      *plan, {.store = destination, .build = "test", .metadata_bytes = layout->metadata_bytes - 1});
  EXPECT_FALSE(written);
  EXPECT_FALSE(std::filesystem::exists(destination));
  // With placement funded the synthetic source still cannot publish: it has
  // no parsed file receipt. The writer refuses before creating a destination.
  written = WriteArtifact(
      *plan, {.store = destination, .build = "test", .metadata_bytes = layout->metadata_bytes});
  ASSERT_FALSE(written);
  EXPECT_EQ(written.error().reason, "a parsed source lacks a valid identity receipt");
  EXPECT_FALSE(std::filesystem::exists(destination));
}

TEST_F(ImportWriterTest, Dsv4HashTablesHaveExclusiveGroupsAndCompleteExpertSlices) {
  auto input = test_support::Dsv4GgufCheckpoint();
  auto cost = NormalizationBound(input);
  ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
  auto plan = Normalize(std::move(input), {cost->metadata_bytes, cost->work_units});
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  auto layout = publication_detail::InspectPlacement(*plan, std::uint64_t{2} << 30U);
  ASSERT_TRUE(layout) << test_support::Failed(layout, &Error::ToString).value_or("");
  EXPECT_EQ(layout->copies, 1328U);
  EXPECT_EQ(layout->destinations, 1328U + 43U * 3U * 255U);
  EXPECT_EQ(layout->tables, 4U);
  EXPECT_EQ(layout->row_tables, 4U);
  EXPECT_EQ(layout->globals, 0U);
  EXPECT_EQ(layout->layers, 43U);
  EXPECT_EQ(layout->heads, 1U);
  EXPECT_EQ(layout->expert_groups, 43U * 256U);
  EXPECT_FALSE(publication_detail::InspectPlacement(*plan, layout->metadata_bytes - 1));
  const auto destination = dir_ / "dsv4-unfunded";
  EXPECT_FALSE(WriteArtifact(
      *plan,
      {.store = destination, .build = "test", .metadata_bytes = layout->metadata_bytes - 1}));
  EXPECT_FALSE(std::filesystem::exists(destination));
}

TEST_F(ImportWriterTest, PlacementAcceptsTheReaderShardCountBeyond256Files) {
  auto input = Tiny();
  for (std::size_t i = 0; i < 300; ++i) {
    SourceFile file{.name = std::format("shard-{}.gguf", i),
                    .path = std::format("/tmp/llmp-import-shard-{}.gguf", i),
                    .bytes = i == 0 ? input.files[0].bytes : 0,
                    .inode = i + 1};
    if (i == 0)
      input.files[0] = std::move(file);
    else
      input.files.push_back(std::move(file));
  }
  Value(input, "split.no", std::int64_t{0});
  Value(input, "split.count", std::int64_t{300});
  auto plan = Normalize(input, kFunded);
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  auto placement = publication_detail::InspectPlacement(*plan, std::uint64_t{1} << 30U);
  ASSERT_TRUE(placement) << test_support::Failed(placement, &Error::ToString).value_or("");
  EXPECT_EQ(plan->source().files.size(), 300U);
  input.files.resize(kMaxSourceFiles + 1);
  EXPECT_FALSE(NormalizationBound(input));
}

TEST_F(ImportWriterTest, Qwen2PlacementAndCompleteLayerCoverageAreFundedBeforeEffects) {
  auto input = test_support::Qwen2GgufCheckpoint();
  auto cost = NormalizationBound(input);
  ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
  auto plan = Normalize(std::move(input), {cost->metadata_bytes, cost->work_units});
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  auto layout = publication_detail::InspectPlacement(*plan, std::uint64_t{2} << 30U);
  ASSERT_TRUE(layout) << test_support::Failed(layout, &Error::ToString).value_or("");
  EXPECT_EQ(layout->copies, 291U);
  EXPECT_EQ(layout->destinations, 291U);
  EXPECT_EQ(layout->tables, 1U);
  EXPECT_EQ(layout->row_tables, 1U);
  EXPECT_EQ(layout->globals, 0U);
  EXPECT_EQ(layout->layers, 24U);
  EXPECT_EQ(layout->heads, 1U);
  EXPECT_EQ(layout->expert_groups, 0U);
  EXPECT_FALSE(publication_detail::InspectPlacement(*plan, layout->metadata_bytes - 1));
  const auto destination = dir_ / "qwen2-unfunded";
  EXPECT_FALSE(WriteArtifact(
      *plan,
      {.store = destination, .build = "test", .metadata_bytes = layout->metadata_bytes - 1}));
  EXPECT_FALSE(std::filesystem::exists(destination));
}

TEST_F(ImportWriterTest, Qwen2Exl3PlacementAndMultipartCopyCoverageAreFundedBeforeEffects) {
  for (const bool rate_4_5 : {false, true}) {
    auto input = test_support::Qwen2Exl3Checkpoint(rate_4_5);
    auto cost = NormalizationBound(input);
    ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
    auto plan = Normalize(std::move(input), {cost->metadata_bytes, cost->work_units});
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    auto layout = publication_detail::InspectPlacement(*plan, std::uint64_t{2} << 30U);
    ASSERT_TRUE(layout) << test_support::Failed(layout, &Error::ToString).value_or("");
    EXPECT_EQ(layout->copies, 798U);
    EXPECT_EQ(layout->destinations, 798U);
    EXPECT_EQ(layout->tables, 1U);
    EXPECT_EQ(layout->row_tables, 1U);
    EXPECT_EQ(layout->globals, 0U);
    EXPECT_EQ(layout->layers, 24U);
    EXPECT_EQ(layout->heads, 1U);
    EXPECT_EQ(layout->expert_groups, 0U);
    EXPECT_FALSE(publication_detail::InspectPlacement(*plan, layout->metadata_bytes - 1));
    const auto destination = dir_ / (rate_4_5 ? "qwen2-exl3-45" : "qwen2-exl3-40");
    EXPECT_FALSE(WriteArtifact(
        *plan,
        {.store = destination, .build = "test", .metadata_bytes = layout->metadata_bytes - 1}));
    EXPECT_FALSE(std::filesystem::exists(destination));
  }
}

TEST_F(ImportWriterTest, CompositionPublicationAndReuse) {
  auto plan = Plan();
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  auto written = WriteArtifact(*plan, Options());
  ASSERT_TRUE(written) << test_support::Failed(written, &Error::ToString).value_or("");
  const std::string art_id = written->id;

  const std::string meta_bytes = R"({"beta_start": 0.00085, "beta_end": 0.012})";
  Write(dir_ / "source/scheduler_config.json", meta_bytes);
  auto meta_file = OpenCheckpointFile(dir_ / "source/scheduler_config.json", 0);
  ASSERT_TRUE(meta_file);

  CompositionWriteOptions opts{
      .store = dir_ / "store",
      .build = "test",
      .architecture = "QwenImage21Pipeline",
      .components =
          {
              {.role = "text_encoder",
               .artifact = art_id,
               .architecture = "Qwen2_5_VLForConditionalGeneration"},
              {.role = "transformer",
               .artifact = art_id,
               .architecture = "QwenImageTransformer2DModel"},
          },
      .files =
          {
              {.name = "scheduler/scheduler_config.json",
               .path = dir_ / "source/scheduler_config.json",
               .bytes = static_cast<std::uint64_t>(meta_bytes.size()),
               .device = meta_file->device,
               .inode = meta_file->inode},
          },
  };

  auto comp = WriteComposition(opts);
  ASSERT_TRUE(comp) << test_support::Failed(comp, &Error::ToString).value_or("");
  EXPECT_FALSE(comp->reused);
  EXPECT_FALSE(comp->id.empty());

  auto opened = artifact::OpenComposition(comp->path, comp->id);
  ASSERT_TRUE(opened) << opened.error().ToString();
  EXPECT_EQ(opened->architecture(), "QwenImage21Pipeline");
  EXPECT_EQ(opened->components().size(), 2U);
  EXPECT_EQ(opened->components()[0].role, "text_encoder");
  EXPECT_EQ(opened->components()[1].role, "transformer");
  auto meta = opened->Metadata("scheduler/scheduler_config.json");
  ASSERT_TRUE(meta.has_value());
  EXPECT_EQ(*meta, meta_bytes);

  // A composition keeps the spec it is given as its spec.json (D-113): the
  // image family's, whose stages count its three components; refused
  // where a stage names a component it does not have.
  {
    const auto& family = test_support::QwenImage().spec;
    auto two = opts;
    two.store = dir_ / "spec-store";
    two.spec = family;
    EXPECT_FALSE(WriteComposition(two));
    auto three = two;
    three.components.push_back(
        {.role = "vae", .artifact = art_id, .architecture = "AutoencoderKLQwenImage21"});
    auto kept = WriteComposition(three);
    ASSERT_TRUE(kept) << test_support::Failed(kept, &Error::ToString).value_or("");
    EXPECT_NE(kept->id, comp->id);
    auto reopened = artifact::OpenComposition(kept->path, kept->id);
    ASSERT_TRUE(reopened) << reopened.error().ToString();
    ASSERT_TRUE(reopened->spec_json().has_value());
    EXPECT_EQ(*reopened->spec_json(), spec::ToJson(family));
  }

  // Re-publishing detects reuse
  auto again = WriteComposition(opts);
  ASSERT_TRUE(again) << test_support::Failed(again, &Error::ToString).value_or("");
  EXPECT_TRUE(again->reused);
  EXPECT_EQ(again->id, comp->id);

  // Refusal: unsorted components
  auto unsorted_opts = opts;
  std::swap(unsorted_opts.components[0], unsorted_opts.components[1]);
  EXPECT_FALSE(WriteComposition(unsorted_opts));

  // Refusal: expected archive identity mismatch
  auto bad_expected_opts = opts;
  const std::array<std::optional<base::Sha256Digest>, 1> bad_digest{base::Sha256Digest{}};
  bad_expected_opts.expected = bad_digest;
  EXPECT_FALSE(WriteComposition(bad_expected_opts));
}

TEST_F(ImportWriterTest, CompositionKeepsScopedDuplicatesAndAuthenticatesFileIdentity) {
  std::vector<SourceFile> files;
  for (const auto role : {"tokenizer", "tokenizer_2"}) {
    const auto path = dir_ / "source" / role / "tokenizer.json";
    Write(path, std::string(role));
    auto opened = OpenCheckpointFile(path, files.size());
    ASSERT_TRUE(opened);
    files.push_back({.name = std::string(role) + "/tokenizer.json",
                     .path = opened->path,
                     .bytes = opened->bytes,
                     .device = opened->device,
                     .inode = opened->inode,
                     .parsed = {opened->bytes, base::Sha256().Update(role).Finish()}});
  }
  CompositionWriteOptions options{.store = dir_ / "store",
                                  .build = "test",
                                  .architecture = "Pipeline",
                                  .components = {{.role = "text_encoder",
                                                  .artifact = std::string(64, 'a'),
                                                  .architecture = "Text"}},
                                  .files = files};
  auto written = WriteComposition(options);
  ASSERT_TRUE(written) << test_support::Failed(written, &Error::ToString).value_or("");
  auto opened = artifact::OpenComposition(written->path);
  ASSERT_TRUE(opened) << test_support::Failed(opened, &artifact::Error::ToString).value_or("");
  EXPECT_EQ(opened->format_version(), 2U);
  EXPECT_EQ(opened->Metadata("tokenizer/tokenizer.json"), "tokenizer");
  EXPECT_EQ(opened->Metadata("tokenizer_2/tokenizer.json"), "tokenizer_2");
  options.build = "another-commit";
  auto reused = WriteComposition(options);
  ASSERT_TRUE(reused);
  EXPECT_EQ(reused->id, written->id);
  EXPECT_TRUE(reused->reused);
  auto unsafe = files;
  unsafe[0].name = "../tokenizer.json";
  EXPECT_FALSE(ValidateCompositionFiles(unsafe));
  unsafe = files;
  unsafe[0].name = "tokenizer/nested/tokenizer.json";
  EXPECT_FALSE(ValidateCompositionFiles(unsafe));
  unsafe = files;
  unsafe.push_back(unsafe.front());
  EXPECT_FALSE(ValidateCompositionFiles(unsafe));
  // A parsed file can also change in place without changing its inode or size.
  Write(files[0].path, "different");
  auto changed = WriteComposition(options);
  ASSERT_FALSE(changed);
  EXPECT_EQ(changed.error().reason, "parsed source bytes changed before streaming");
  Write(files[0].path, "tokenizer");
  // Replacement of identical length is a different file, even before hashing.
  const auto replacement = dir_ / "replacement";
  Write(replacement, "different");
  std::filesystem::rename(replacement, files[0].path);
  EXPECT_FALSE(WriteComposition(options));
}

TEST_F(ImportWriterTest, PlainUtf8SourceNamesAreJsonQuotedAndUnsafeNamesFailAtRead) {
  Write(dir_ / "source/model \"café\".gguf", Gguf(Tiny()));
  auto input = ReadGguf(dir_ / "source/model \"café\".gguf");
  ASSERT_TRUE(input);
  auto plan = Normalize(std::move(*input), kFunded);
  ASSERT_TRUE(plan);
  auto written = WriteArtifact(*plan, Options());
  ASSERT_TRUE(written) << test_support::Failed(written, &Error::ToString).value_or("");
  auto opened = artifact::Artifact::Open(written->path);
  ASSERT_TRUE(opened) << test_support::Failed(opened, &artifact::Error::ToString).value_or("");
  EXPECT_EQ(opened->sources()[0].name, "model \"café\".gguf");
  Write(dir_ / "source/bad\nname.gguf", Gguf(Tiny()));
  EXPECT_FALSE(ReadGguf(dir_ / "source/bad\nname.gguf"));
}

TEST_F(ImportWriterTest, ImageComponentMetadataPreservesF16AndBf16VaeWeights) {
  const auto& family = test_support::QwenImage();
  const auto tensors = model::VaeDecoderTensors(family.profile.vae, family.mapping->names.vae);
  auto built = spec::ImageDecodeSpec(family.profile.vae, 64, 64, tensors);
  ASSERT_TRUE(built) << test_support::Failed(built).value_or("");
  for (const auto encoding : {Encoding::kF16, Encoding::kBF16}) {
    Checkpoint component{
        .role = "vae", .container = "safetensors", .architecture = "AutoencoderKLQwenImage21"};
    for (const auto& tensor : tensors) {
      format::Shape shape;
      shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
      for (std::size_t d = 0; d < tensor.shape.size(); ++d)
        shape.ne[d] = tensor.shape[tensor.shape.size() - d - 1];
      const auto format = ir::Dense(encoding, shape);
      component.tensors.push_back(
          {.name = tensor.name,
           .format = format,
           .extent = {.bytes = format::PartsOf(format)->Find(format::PartRole::kData)->bytes},
           .dtype = encoding == Encoding::kF16 ? "F16" : "BF16",
           .dims = tensor.shape});
    }
    auto planned = image_pipeline_detail::PlanComponent(component, tensors, {}, built->spec, {});
    ASSERT_TRUE(planned) << test_support::Failed(planned, &Error::ToString).value_or("");
    EXPECT_EQ(planned->canonical.tensors.size(), tensors.size());
    for (const auto& tensor : planned->canonical.tensors)
      EXPECT_EQ(tensor.format.encoding, encoding);
    EXPECT_EQ(planned->spec, built->spec);
  }
}

TEST_F(ImportWriterTest, ImageComponentRefusesTensorsItsMappingNeitherReadsNorLeavesOut) {
  const auto& family = test_support::QwenImage();
  const auto tensors = model::VaeDecoderTensors(family.profile.vae, family.mapping->names.vae);
  auto built = spec::ImageDecodeSpec(family.profile.vae, 64, 64, tensors);
  ASSERT_TRUE(built) << test_support::Failed(built).value_or("");
  Checkpoint component{
      .role = "vae", .container = "safetensors", .architecture = "AutoencoderKLQwenImage21"};
  const auto add = [&](std::string name, const std::vector<std::uint64_t>& dims) {
    format::Shape shape;
    shape.rank = static_cast<std::uint8_t>(dims.size());
    for (std::size_t d = 0; d < dims.size(); ++d) shape.ne[d] = dims[dims.size() - d - 1];
    const auto format = ir::Dense(Encoding::kBF16, shape);
    component.tensors.push_back(
        {.name = std::move(name),
         .format = format,
         .extent = {.bytes = format::PartsOf(format)->Find(format::PartRole::kData)->bytes},
         .dtype = "BF16",
         .dims = dims});
  };
  for (const auto& tensor : tensors) add(tensor.name, tensor.shape);
  const auto rule = std::ranges::find(family.mapping->components, std::string_view("vae"),
                                      &spec::ImageComponentRule::role);
  ASSERT_NE(rule, family.mapping->components.end());
  const auto plan = [&] {
    return image_pipeline_detail::PlanComponent(component, tensors, rule->ignored_tensors,
                                                built->spec, {});
  };
  ASSERT_TRUE(plan());
  // What the mapping leaves out by name: the encoder, its quant_conv, the up
  // blocks' temporal convolutions (any block, by its index).
  for (const auto* name : {"encoder.conv_in.weight", "quant_conv.bias",
                           "decoder.up_blocks.2.upsampler.time_conv.weight"})
    add(name, {4});
  auto planned = plan();
  ASSERT_TRUE(planned) << test_support::Failed(planned, &Error::ToString).value_or("");
  EXPECT_EQ(planned->canonical.tensors.size(), tensors.size());
  // A variant's extra norm or bias, or a near miss of an ignored name, is
  // refused rather than dropped.
  const auto base = component;
  for (const auto* name :
       {"decoder.norm_out.bias", "decoder.up_blocks.x.upsampler.time_conv.weight",
        "decoder.up_blocks.02.upsampler.time_conv.weight", "quant_conv", "encoderx.weight"}) {
    component = base;
    add(name, {4});
    EXPECT_EQ(test_support::Failed(plan(), &Error::rule), Rule::kContainer) << name;
  }
  EXPECT_TRUE(spec::IgnoredImageTensor("a.{layer}.b", "a.17.b"));
  EXPECT_FALSE(spec::IgnoredImageTensor("a.{layer}.b", "a..b"));
  EXPECT_FALSE(spec::IgnoredImageTensor("a.{layer}.b", "a.17.bc"));
  EXPECT_TRUE(spec::IgnoredImageTensor("a.{layer}.", "a.3.c.d"));
}

TEST_F(ImportWriterTest, ImagePipelineValidationAndBound) {
  Checkpoint invalid_container{.container = "gguf", .architecture = "QwenImage21Pipeline"};
  EXPECT_FALSE(ImagePipelineBound(invalid_container));
  EXPECT_FALSE(PlanImagePipeline(invalid_container));

  Checkpoint invalid_arch{.container = "diffusers", .architecture = "FluxPipeline"};
  EXPECT_FALSE(ImagePipelineBound(invalid_arch));
  EXPECT_FALSE(PlanImagePipeline(invalid_arch));

  Checkpoint missing_comp{.container = "diffusers", .architecture = "QwenImage21Pipeline"};
  missing_comp.components.push_back({.role = "text_encoder"});
  EXPECT_FALSE(PlanImagePipeline(missing_comp));

  // Bound calculation sums tensor bytes
  Checkpoint pipeline{.container = "diffusers", .architecture = "QwenImage21Pipeline"};
  Checkpoint comp{.role = "text_encoder"};
  comp.tensors.push_back({.name = "t1", .extent = {.bytes = 100}});
  comp.tensors.push_back({.name = "t2", .extent = {.bytes = 200}});
  pipeline.components.push_back(comp);
  auto bound = ImagePipelineBound(pipeline);
  ASSERT_TRUE(bound);
  EXPECT_EQ(bound->payload_bytes, 300U);
  EXPECT_EQ(bound->work_units, 2U);
  EXPECT_EQ(bound->metadata_bytes, 512U << 20U);
}

TEST_F(ImportWriterTest, DiffusersArchiveEventRecorded) {
  const auto root = dir_ / "archive";
  const auto revision = root / "hf/owner/model@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const auto bytes = Gguf(Tiny());
  Write(root / "llmp-archive-root.json", R"({"format":"llmp-archive-root","format_version":1})");
  const std::string model_index_json = R"({"_class_name":"QwenImage21Pipeline"})";
  Write(revision / "model_index.json", model_index_json);
  Write(revision / "text_encoder/model.safetensors", bytes);

  const std::string text_safetensors = "text_encoder/model.safetensors";
  const std::vector<std::pair<std::string, std::vector<std::byte>>> archive_files = {
      {"model_index.json",
       {reinterpret_cast<const std::byte*>(model_index_json.data()),
        reinterpret_cast<const std::byte*>(model_index_json.data()) + model_index_json.size()}},
      {text_safetensors, {bytes.begin(), bytes.end()}},
  };
  Write(revision / "llmp-archive.json", ArchiveManifestFiles(archive_files));

  Checkpoint source{.container = "diffusers", .architecture = "QwenImage21Pipeline"};
  source.files.push_back({.name = "model_index.json",
                          .path = revision / "model_index.json",
                          .bytes = model_index_json.size()});
  Checkpoint comp{.role = "text_encoder"};
  comp.files.push_back(
      {.name = text_safetensors, .path = revision / text_safetensors, .bytes = bytes.size()});
  source.components.push_back(comp);

  auto archive = FindArchiveInput(revision / "model_index.json");
  ASSERT_TRUE(archive && *archive);
  auto identities = ArchiveIdentities(**archive, source);
  ASSERT_TRUE(identities) << test_support::Failed(identities, &Error::ToString).value_or("");
  EXPECT_EQ(identities->size(), 1U);

  auto source_opened = OpenCheckpointFile(source.files[0].path, 0);
  ASSERT_TRUE(source_opened);
  source.files[0].device = source_opened->device;
  source.files[0].inode = source_opened->inode;
  CompositionWriteOptions opts{
      .store = dir_ / "store",
      .build = "test",
      .architecture = "QwenImage21Pipeline",
      .components =
          {
              {.role = "text_encoder", .artifact = std::string(64, 'a'), .architecture = "Arch"},
          },
      .files = source.files,
  };
  auto comp_result = WriteComposition(opts);
  ASSERT_TRUE(comp_result) << test_support::Failed(comp_result, &Error::ToString).value_or("");
  auto opened = artifact::OpenComposition(comp_result->path);
  ASSERT_TRUE(opened);

  EXPECT_TRUE(RecordArchiveImport(**archive, source, *opened));

  std::size_t count = 0;
  for (const auto& item : std::filesystem::directory_iterator(root / "catalog/events")) {
    ASSERT_EQ(item.path().extension(), ".json");
    auto file = OpenCheckpointFile(item.path(), 0);
    ASSERT_TRUE(file);
    auto text = ReadDocument(*file, 65536, 0);
    ASSERT_TRUE(text);
    const auto json = base::json::Parse(*text);
    ASSERT_TRUE(json);
    EXPECT_EQ(json->root().find("kind")->string(), "import");
    EXPECT_EQ(json->root().find("body")->find("artifact")->find("format")->string(),
              "llmp-composition");
    EXPECT_EQ(json->root().find("body")->find("variant")->string(), "model");
    ++count;
  }
  EXPECT_EQ(count, 1U);
}

// A drafter's publication with a gathered head and a
// token map generated from its whole authenticated list file: the private
// tiny product exercises only the shared publisher.
class ImportDrafterWriterTest : public ImportWriterTest {
 protected:
  static constexpr std::string_view kOptions = R"({"draft_head":"ids.txt"})";
  std::vector<std::byte> weights_;
  std::expected<NormalizedPlan, Error> Drafter(std::string architecture = "qwen4exp-mtp",
                                               std::string container = "safetensors",
                                               bool executable = false) {
    weights_.resize(256);
    for (std::size_t i = 0; i < weights_.size(); ++i)
      weights_[i] = static_cast<std::byte>((i * 37 + 11) & 0xFF);
    Write(dir_ / "drafter/weights.bin", weights_);
    Write(dir_ / "drafter/ids.txt", "1\n2\n5\n");
    Checkpoint source{.container = container, .architecture = architecture};
    for (const auto name : {"weights.bin", "ids.txt"}) {
      auto opened = OpenCheckpointFile(dir_ / "drafter" / name, 0);
      if (!opened) return std::unexpected(opened.error());
      const bool list = std::string_view(name) == "ids.txt";
      const auto parsed = list ? opened->bytes : 16;
      source.files.push_back(
          {.name = name,
           .path = opened->path,
           .bytes = opened->bytes,
           .device = opened->device,
           .inode = opened->inode,
           .parsed = {parsed,
                      list ? base::Sha256().Update("1\n2\n5\n").Finish()
                           : base::Sha256().Update(std::span(weights_).first(16)).Finish()}});
    }
    if (executable) {
      const auto config = std::format(R"({{"model_type":"{}"}})", architecture);
      Write(dir_ / "drafter/config.json", config);
      auto opened = OpenCheckpointFile(dir_ / "drafter/config.json", 2);
      if (!opened) return std::unexpected(opened.error());
      source.files.push_back({.name = "config.json",
                              .path = opened->path,
                              .bytes = opened->bytes,
                              .device = opened->device,
                              .inode = opened->inode,
                              .parsed = {opened->bytes, base::Sha256().Update(config).Finish()}});
    }
    const auto bf16 = [](std::uint64_t rows) {
      return format::TensorFormat{.encoding = Encoding::kBF16,
                                  .shape = {.ne = {4, rows, 1, 1}, .rank = 2}};
    };
    const auto local_format = executable
                                  ? format::TensorFormat{.encoding = Encoding::kBF16,
                                                         .shape = {.ne = {4, 1, 1, 1}, .rank = 1}}
                                  : bf16(4);
    const auto local_bytes = executable ? 8U : 32U;
    // A 10-row table of 4 BF16 values a row at 0, local matrix/norm at 128.
    source.tensors.push_back(
        {.name = "fc.weight",
         .format = local_format,
         .extent = {.file = 0, .offset = 128, .bytes = local_bytes},
         .dtype = "BF16",
         .dims = executable ? std::vector<std::uint64_t>{4} : std::vector<std::uint64_t>{4, 4}});
    source.tensors.push_back({.name = "table.rows.1",
                              .format = bf16(2),
                              .extent = {.file = 0, .offset = 8, .bytes = 16},
                              .dtype = "BF16",
                              .dims = {2, 4}});
    source.tensors.push_back({.name = "table.rows.5",
                              .format = bf16(1),
                              .extent = {.file = 0, .offset = 40, .bytes = 8},
                              .dtype = "BF16",
                              .dims = {1, 4}});
    std::vector<TensorAssembly> outputs;
    outputs.push_back({.name = "fc.weight",
                       .format = local_format,
                       .parts = {{.part = format::PartRole::kData, .sources = {0}}}});
    outputs.push_back({.name = "draft_output.weight",
                       .format = bf16(3),
                       .parts = {{.part = format::PartRole::kData,
                                  .kind = PartAssemblyKind::kRows,
                                  .sources = {1, 2}}}});
    const format::TensorFormat projection_format{.encoding = Encoding::kBF16,
                                                 .shape = {.ne = {8, 4, 1, 1}, .rank = 2}};
    if (executable) {
      source.tensors.push_back({.name = "projection.weight",
                                .format = projection_format,
                                .extent = {.file = 0, .offset = 160, .bytes = 64},
                                .dtype = "BF16",
                                .dims = {4, 8}});
      outputs.push_back({.name = "projection.weight",
                         .format = projection_format,
                         .parts = {{.part = format::PartRole::kData, .sources = {3}}}});
    }
    std::vector<SourceDisposition> dispositions(source.tensors.size());
    auto cost = IndexedPartPlan::Bound(source, outputs);
    if (!cost) return std::unexpected(cost.error());
    auto parts = IndexedPartPlan::Make(std::move(source), outputs, dispositions, *cost);
    if (!parts) return std::unexpected(parts.error());
    spec::Source canonical{.architecture = "qwen4exp-mtp", .container = container};
    canonical.tensors = {
        {.name = "fc.weight", .format = local_format},
        {.name = "draft_output.weight", .format = bf16(3)},
        {.name = "draft_output.ids",
         .format = {.encoding = Encoding::kI32, .shape = {.ne = {1, 3, 1, 1}, .rank = 2}}}};
    spec::ModelSpec model;
    model.stages.resize(2);
    model.stages[0].name = "target";
    model.stages[1] = {.name = "mtp", .role = spec::StageRole::kDrafter, .artifact = 1};
    model.stages[1].layers.resize(1);
    model.stages[1].layers[0].components.push_back(
        {.kind = "x.y", .weights = {{.role = "w", .tensor = "fc.weight"}}});
    model.stages[1].heads.output =
        spec::ComponentRef{.kind = "head.linear",
                           .weights = {{.role = "ids", .tensor = "draft_output.ids"},
                                       {.role = "weight", .tensor = "draft_output.weight"}}};
    if (executable) {
      canonical.tensors.insert(canonical.tensors.end() - 1,
                               {.name = "projection.weight", .format = projection_format});
      model.stages[0].embedding = spec::ComponentRef{
          .kind = "embedding.token", .weights = {{.role = "table", .tensor = "token_embd.weight"}}};
      model.stages[0].heads.output = spec::ComponentRef{
          .kind = "head.linear", .weights = {{.role = "weight", .tensor = "token_embd.weight"}}};
      model.stages[0].layers = {
          {{spec::ComponentRef{.kind = "norm.rms",
                               .params = {{"eps", 0.00001}},
                               .weights = {{.role = "weight", .tensor = "target.norm"}}}}}};
      model.stages[0].taps = {{.name = "features", .layer = 0}};
      model.stages[1].drafting = spec::MtpDraft{};
      model.stages[1].embedding = spec::ComponentRef{
          .kind = "embedding.projected",
          .params = {{"scale", 1.0},
                     {"feature", std::int64_t{4}},
                     {"tap", std::string("features")}},
          .weights = {{.role = "table", .tensor = "token_embd.weight", .stage = "target"},
                      {.role = "projection", .tensor = "projection.weight"}}};
      model.stages[1].reads = {
          {.stage = "target", .kind = spec::ReadKind::kTap, .name = "features"}};
      for (auto& stage : model.stages) {
        stage.stream.width = 4;
        stage.context = 128;
      }
      model.stages[1].layers[0].components[0] = {
          .kind = "norm.rms",
          .params = {{"eps", 0.00001}},
          .weights = {{.role = "weight", .tensor = "fc.weight"}}};
      model.stages[1].heads.output->params = {{"selected", true}, {"rows", std::int64_t{3}}};
      spec::Canonicalize(model);
    }
    std::vector<GeneratedPart> generated{{.tensor = executable ? 3U : 2U, .file = 1, .bytes = {}}};
    for (const int id : {1, 2, 5})
      for (int b = 0; b < 4; ++b)
        generated[0].bytes.push_back(static_cast<std::byte>(b == 0 ? id : 0));
    std::vector<NormalizedPartBinding> bindings{{0, format::PartRole::kData},
                                                {1, format::PartRole::kData}};
    if (executable) bindings.push_back({2, format::PartRole::kData});
    return test_support::PartPublicationFixture::MakeDrafter(
        std::move(*parts), std::move(canonical), std::move(model), std::move(bindings),
        std::move(generated), std::string(kOptions));
  }
  std::vector<std::byte> Read(const artifact::Artifact& a, const std::filesystem::path& root,
                              std::string_view name) {
    const auto resource = a.FindResource(name);
    EXPECT_TRUE(resource) << name;
    if (!resource) return {};
    auto place = a.ResourcePlacement(*resource);
    EXPECT_TRUE(place);
    if (!place) return {};
    const auto& group = a.groups()[place->group];
    const auto& shard = a.shards()[group.shard];
    auto file = OpenCheckpointFile(root / shard.path, 0);
    EXPECT_TRUE(file);
    std::vector<std::byte> out(static_cast<std::size_t>(place->bytes.value()));
    EXPECT_TRUE(ReadExactly(
        file->fd.get(), shard.data_offset.value() + group.offset.value() + place->offset.value(),
        out, 0));
    return out;
  }
};

TEST_F(ImportDrafterWriterTest, GathersRowsAndWritesItsAuthenticatedTokenMap) {
  auto plan = Drafter();
  ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
  auto options = Options();
  options.options = std::string(kOptions);
  auto result = WriteArtifact(*plan, options);
  ASSERT_TRUE(result) << test_support::Failed(result, &Error::ToString).value_or("");
  auto opened = artifact::Artifact::Open(result->path);
  ASSERT_TRUE(opened);
  EXPECT_TRUE(opened->spec_json().has_value());
  EXPECT_EQ(opened->model().architecture, "qwen4exp-mtp");
  EXPECT_EQ(opened->converter().options_sha256, base::Sha256().Update(kOptions).Finish());
  const auto list = std::ranges::find(opened->sources(), "ids.txt", &artifact::Source::name);
  ASSERT_NE(list, opened->sources().end());
  EXPECT_EQ(list->sha256, base::Sha256().Update("1\n2\n5\n").Finish());
  const auto ids = Read(*opened, result->path, "draft_output.ids");
  EXPECT_EQ(ids, plan->generated()[0].bytes);
  std::vector<std::byte> rows;
  for (const std::ptrdiff_t row : {std::ptrdiff_t{1}, std::ptrdiff_t{2}, std::ptrdiff_t{5}})
    rows.insert(rows.end(), weights_.begin() + row * 8, weights_.begin() + row * 8 + 8);
  EXPECT_EQ(Read(*opened, result->path, "draft_output.weight"), rows);
  EXPECT_TRUE(std::ranges::equal(Read(*opened, result->path, "fc.weight"),
                                 std::span(weights_).subspan(128, 32)));
  auto source = spec::SourceOf(*opened);
  ASSERT_TRUE(source);
  EXPECT_EQ(source->tensors.size(), 3U);
  // Without the plan's options the identity would differ: refused.
  EXPECT_FALSE(WriteArtifact(*plan, Options("no-options")));
  EXPECT_FALSE(std::filesystem::exists(dir_ / "no-options" / result->id));
}

TEST_F(ImportDrafterWriterTest, NativeComponentsPublishTheirCanonicalRoleAndKeepSourceIdentity) {
  for (const std::string container : {"safetensors", "mlx"}) {
    const std::string source_architecture =
        container == "safetensors" ? "qwen3_8_flash_next" : "qwen4_exp";
    SCOPED_TRACE(container);
    auto plan = Drafter(source_architecture, container, true);
    ASSERT_TRUE(plan) << plan.error().ToString();
    EXPECT_EQ(plan->source().architecture, source_architecture);
    auto options = Options(container);
    options.options = std::string(kOptions);
    // Publication and reuse both perform the writer's deep payload validation.
    auto result = WriteArtifact(*plan, options);
    ASSERT_TRUE(result) << result.error().ToString();
    auto reused = WriteArtifact(*plan, options);
    ASSERT_TRUE(reused) << reused.error().ToString();
    EXPECT_TRUE(reused->reused);
    auto opened = artifact::Artifact::Open(result->path, {.expected_id = result->id});
    ASSERT_TRUE(opened) << opened.error().ToString();
    EXPECT_EQ(opened->model().architecture, "qwen4exp-mtp");
    EXPECT_EQ(opened->model().container, container);
    const auto config = std::format(R"({{"model_type":"{}"}})", source_architecture);
    EXPECT_EQ(opened->ReadMetadata("config.json"), config);
    const auto receipt =
        std::ranges::find(opened->sources(), "config.json", &artifact::Source::name);
    ASSERT_NE(receipt, opened->sources().end());
    EXPECT_EQ(receipt->sha256, base::Sha256().Update(config).Finish());
    const auto stored = spec::StoredSpec(*opened);
    ASSERT_TRUE(stored) << stored.error();
    EXPECT_EQ(*stored, plan->spec());
    const auto admitted = runtime::ReadComponentDrafterAdmission(*opened);
    ASSERT_TRUE(admitted) << admitted.error();

    spec::ModelSpec target_spec;
    target_spec.stages.push_back(plan->spec().stages.front());
    spec::Canonicalize(target_spec);
    for (const std::string& architecture :
         {std::string("qwen4exp"), source_architecture, std::string("deepseek4"),
          std::string("unregistered-target")}) {
      test::V1Plan target_plan;
      target_plan.architecture = architecture;
      target_plan.container = architecture == "deepseek4" ? "gguf" : container;
      target_plan.spec = spec::ToJson(target_spec);
      target_plan.groups = {test::V1Group{
          .kind = artifact::GroupKind::kTable,
          .resources = {
              {.name = "token_embd.weight",
               .format = {.encoding = Encoding::kBF16, .shape = {.ne = {4, 10, 1, 1}, .rank = 2}}},
              {.name = "target.norm",
               .format = {.encoding = Encoding::kBF16,
                          .shape = {.ne = {4, 1, 1, 1}, .rank = 1}}}}}};
      auto target_artifact = artifact::Artifact::Open(
          test::Publish(dir_ / (container + architecture), test::Render(target_plan)));
      ASSERT_TRUE(target_artifact) << target_artifact.error().ToString();
      const auto facts = runtime::ArtifactFactsOf(*target_artifact, &*opened);
      ASSERT_TRUE(facts) << facts.error();
      EXPECT_TRUE(facts->composed_spec) << facts->composed_spec_error;
      EXPECT_TRUE(facts->composed_spec_error.empty());

      // A compatible graph is admitted regardless of the provenance name;
      // a real feature-width mismatch still prevents composition.
      auto wider = target_spec;
      wider.stages[0].stream.width = 5;
      spec::Canonicalize(wider);
      target_plan.spec = spec::ToJson(wider);
      for (auto& resource : target_plan.groups[0].resources) resource.format.shape.ne[0] = 5;
      auto incompatible = artifact::Artifact::Open(
          test::Publish(dir_ / (container + architecture + "-wide"), test::Render(target_plan)));
      ASSERT_TRUE(incompatible) << incompatible.error().ToString();
      const auto refused = runtime::ArtifactFactsOf(*incompatible, &*opened);
      ASSERT_TRUE(refused) << refused.error();
      EXPECT_FALSE(refused->composed_spec);
      EXPECT_FALSE(refused->composed_spec_error.empty());
    }
  }
}

TEST_F(ImportDrafterWriterTest, AChangedListOrMisfitTokenMapCannotPublish) {
  {
    auto plan = Drafter();
    ASSERT_TRUE(plan);
    // Same size, other IDs: the list no longer matches its parsed receipt.
    std::fstream file(dir_ / "drafter/ids.txt", std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file);
    file.seekp(0);
    file.write("3", 1);
    file.close();
    auto options = Options("changed");
    options.options = std::string(kOptions);
    EXPECT_FALSE(WriteArtifact(*plan, options));
    auto store = std::filesystem::directory_iterator(dir_ / "changed");
    EXPECT_TRUE(std::ranges::none_of(
        store, [](const auto& entry) { return entry.path().filename().string().size() == 64; }));
  }
  {
    auto plan = Drafter();
    ASSERT_TRUE(plan);
    test_support::PartPublicationFixture::ResizeGenerated(*plan, 8);
    auto options = Options("misfit");
    options.options = std::string(kOptions);
    EXPECT_FALSE(WriteArtifact(*plan, options));
    EXPECT_FALSE(std::filesystem::exists(dir_ / "misfit"));
  }
}

void ExpectMode(const std::filesystem::path& path, unsigned mode) {
  std::error_code error;
  const auto status = std::filesystem::status(path, error);
  ASSERT_FALSE(error) << path << ": " << error.message();
  EXPECT_EQ(static_cast<unsigned>(status.permissions() & std::filesystem::perms::mask), mode)
      << path;
}
void ExpectPrivateTree(const std::filesystem::path& path) {
  ExpectMode(path, 0700);
  for (const auto& entry : std::filesystem::recursive_directory_iterator(path))
    ExpectMode(entry.path(), entry.is_directory() ? 0700 : 0600);
}
std::map<std::filesystem::path, unsigned> MakeExistingTreePublic(
    const std::filesystem::path& path) {
  std::map<std::filesystem::path, unsigned> modes{{path, 0755}};
  for (const auto& entry : std::filesystem::recursive_directory_iterator(path))
    modes.emplace(entry.path(), entry.is_directory() ? 0755 : 0644);
  for (const auto& [name, mode] : modes) EXPECT_EQ(::chmod(name.c_str(), mode), 0);
  return modes;
}

TEST_F(ImportWriterTest, PublicationIsPrivateAndPreservesExistingModes) {
  auto plan = Plan();
  ASSERT_TRUE(plan);
  std::string identity;
  // Both extremes: never expose files under a permissive umask, and restore
  // owner access only on our newly created paths under a restrictive one.
  for (const mode_t mask : {mode_t{0000}, mode_t{0777}}) {
    const auto options =
        Options(mask == 0 ? "permissive/nested/store" : "restrictive/nested/store");
    const auto previous = ::umask(mask);
    auto written = WriteArtifact(*plan, options);
    (void)::umask(previous);
    ASSERT_TRUE(written) << test_support::Failed(written, &Error::ToString).value_or("");
    EXPECT_FALSE(written->reused);
    if (identity.empty()) identity = written->id;
    EXPECT_EQ(written->id, identity);
    ExpectPrivateTree(options.store.parent_path().parent_path());
    auto opened = artifact::Artifact::Open(written->path, {.expected_id = identity});
    ASSERT_TRUE(opened);
    auto stored = spec::StoredSpec(*opened);
    ASSERT_TRUE(stored);
    EXPECT_EQ(*stored, plan->spec());
  }
  ExpectMode(dir_, 0750);
  ExpectMode(dir_ / "source/model.gguf", 0644);

  const auto options = Options("permissive/nested/store");
  const auto modes = MakeExistingTreePublic(options.store);
  const auto previous = ::umask(0000);
  auto reused = WriteArtifact(*plan, options);  // includes full existing-payload verification
  (void)::umask(previous);
  ASSERT_TRUE(reused);
  EXPECT_TRUE(reused->reused);
  EXPECT_EQ(reused->id, identity);
  for (const auto& [path, mode] : modes) ExpectMode(path, mode);

  const auto staging = options.store / ".staging";
  const auto orphan = staging / "job-orphan123";
  ASSERT_TRUE(std::filesystem::create_directory(orphan));
  auto other_plan = Plan(true);
  ASSERT_TRUE(other_plan);
  auto other = WriteArtifact(*other_plan, options);
  ASSERT_TRUE(other);
  EXPECT_FALSE(std::filesystem::exists(orphan));
  ExpectPrivateTree(other->path);
  for (const auto& [path, mode] : modes) ExpectMode(path, mode);

  auto invalid = options;
  const std::array<std::optional<base::Sha256Digest>, 1> wrong{base::Sha256Digest{}};
  invalid.expected = wrong;
  EXPECT_FALSE(WriteArtifact(*other_plan, invalid));
  for (const auto& entry : std::filesystem::directory_iterator(staging))
    EXPECT_EQ(entry.path().filename(), "native-import.lock");
}

TEST_F(ImportWriterTest, CompositionIsPrivateAndPreservesExistingModes) {
  auto plan = Plan();
  ASSERT_TRUE(plan);
  auto component = WriteArtifact(*plan, Options("components"));
  ASSERT_TRUE(component);
  const auto metadata = dir_ / "source/config.json";
  Write(metadata, "{\"fixture\":true}");
  auto source = OpenCheckpointFile(metadata, 0);
  ASSERT_TRUE(source);
  CompositionWriteOptions options{
      .store = {},
      .build = "test",
      .architecture = "fixture",
      .components = {{.role = "target", .artifact = component->id, .architecture = "Gemma2"}},
      .files = {{.name = "tokenizer/config.json",
                 .path = metadata,
                 .bytes = source->bytes,
                 .device = source->device,
                 .inode = source->inode}},
      .spec = plan->spec()};
  std::string identity;
  for (const mode_t mask : {mode_t{0000}, mode_t{0777}}) {
    options.store = dir_ / (mask == 0 ? "permissive/nested/store" : "restrictive/nested/store");
    const auto previous = ::umask(mask);
    auto written = WriteComposition(options);
    (void)::umask(previous);
    ASSERT_TRUE(written) << test_support::Failed(written, &Error::ToString).value_or("");
    EXPECT_FALSE(written->reused);
    if (identity.empty()) identity = written->id;
    EXPECT_EQ(written->id, identity);
    ExpectPrivateTree(options.store.parent_path().parent_path());
    auto opened = artifact::OpenComposition(written->path, identity);
    ASSERT_TRUE(opened);
    EXPECT_EQ(opened->Metadata("tokenizer/config.json"), "{\"fixture\":true}");
    ASSERT_TRUE(opened->spec_json());
    EXPECT_EQ(*opened->spec_json(), spec::ToJson(plan->spec()));
  }
  ExpectMode(dir_, 0750);
  ExpectMode(metadata, 0644);
  options.store = dir_ / "permissive/nested/store";
  const auto modes = MakeExistingTreePublic(options.store);
  const auto previous = ::umask(0000);
  auto reused = WriteComposition(options);
  (void)::umask(previous);
  ASSERT_TRUE(reused);
  EXPECT_TRUE(reused->reused);
  EXPECT_EQ(reused->id, identity);
  for (const auto& [path, mode] : modes) ExpectMode(path, mode);

  const std::array<std::optional<base::Sha256Digest>, 1> wrong{base::Sha256Digest{}};
  options.expected = wrong;
  EXPECT_FALSE(WriteComposition(options));
  for (const auto& entry : std::filesystem::directory_iterator(options.store / ".staging"))
    EXPECT_EQ(entry.path().filename(), "native-import.lock");
}

}  // namespace
}  // namespace llmp::import
