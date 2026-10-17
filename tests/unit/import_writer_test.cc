// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <map>
#include <variant>

#include "artifact_writer.h"
#include "base/check.h"
#include "base/json.h"
#include "cli/cli.h"
#include "components/step.h"
#include "dsv4_gguf_normalization_fixture.h"
#include "expected_error.h"
#include "image_family.h"
#include "import/archive_input.h"
#include "import/image_pipeline.h"
#include "import/modelopt_qwen38.h"
#include "import/normalize.h"
#include "import/reader.h"
#include "import/readers/gguf.h"
#include "import/readers/safetensors.h"
#include "import/stream.h"
#include "import/writer.h"
#include "import_fixtures.h"
#include "part_publication_fixture.h"
#include "qwen2_exl3_normalization_fixture.h"
#include "qwen2_gguf_normalization_fixture.h"
#include "qwen38_gguf_normalization_fixture.h"
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
    EXPECT_NE(
        std::string_view(text.data(), count).find("outside closed ModelOpt target source domain"),
        std::string_view::npos);
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
  ModelOptTargetPlan plan;
  std::map<std::string, std::vector<std::byte>> expected;
};
class ImportPartWriterTest : public ImportWriterTest {
 protected:
  std::expected<PartCase, Error> Parts() {
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
      bytes.insert(bytes.end(), payload.begin(), payload.end());
      bytes.insert(bytes.end(), 7, std::byte{0xB5});
      return std::pair(index, std::move(payload));
    };
    // Put the permuted BF16 weight first physically so the 1 MiB source stream
    // splits it after exactly 3 bytes, independently of numeric output order.
    auto logical = normalized->canonical().tensors;
    const auto query = std::ranges::find(logical, "blk.0.attn_q.weight", &spec::SourceTensor::name);
    if (query == logical.end()) return std::unexpected(Error{.reason = "missing fixture query"});
    std::rotate(logical.begin(), query, query + 1);
    for (const auto& old : logical) {
      auto format = old.format;
      const bool permute = old.name == "blk.0.attn_q.weight";
      const bool quant_experts = old.name == "blk.0.ffn_gate_up_exps.weight";
      const bool quant_dense = old.name == "blk.0.ffn_gate.weight";
      const bool quant_mx = old.name == "blk.0.ffn_up.weight";
      if (permute) format.encoding = Encoding::kBF16;
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
      if (permute || quant_mx) {
        auto permutation =
            HeadPermutationPlan::Make(format, HeadAxis::kRows, 2, 4, format.shape.ne[1] / 4);
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
          if (permute || quant_mx) {
            const auto original = wanted;
            const auto head_bytes = wanted.size() / 4;
            for (std::size_t destination = 0; destination < 4; ++destination) {
              const auto source_head = (destination % 2) * 2 + destination / 2;
              std::copy_n(original.begin() + static_cast<std::ptrdiff_t>(source_head * head_bytes),
                          head_bytes,
                          wanted.begin() + static_cast<std::ptrdiff_t>(destination * head_bytes));
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
            (std::vector<std::string>{"blk.0.attn_q.weight", "blk.0.ffn_up.weight.block_scales",
                                      "blk.0.ffn_up.weight.codes"}));
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
               .bytes = static_cast<std::uint64_t>(meta_bytes.size())},
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
  auto meta = opened->Metadata("scheduler_config.json");
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

// A drafter's publication (no spec of its own) with a gathered head and a
// token map generated from its whole authenticated list file: the private
// tiny product exercises only the shared publisher.
class ImportDrafterWriterTest : public ImportWriterTest {
 protected:
  static constexpr std::string_view kOptions = R"({"draft_head":"ids.txt"})";
  std::vector<std::byte> weights_;
  std::expected<ModelOptDrafterPlan, Error> Drafter() {
    weights_.resize(256);
    for (std::size_t i = 0; i < weights_.size(); ++i)
      weights_[i] = static_cast<std::byte>((i * 37 + 11) & 0xFF);
    Write(dir_ / "drafter/weights.bin", weights_);
    Write(dir_ / "drafter/ids.txt", "1\n2\n5\n");
    Checkpoint source{.container = "safetensors", .architecture = "qwen4exp-mtp"};
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
    const auto bf16 = [](std::uint64_t rows) {
      return format::TensorFormat{.encoding = Encoding::kBF16,
                                  .shape = {.ne = {4, rows, 1, 1}, .rank = 2}};
    };
    // A 10-row table of 4 BF16 values a row at 0, a 4x4 matrix at 128.
    source.tensors.push_back({.name = "fc.weight",
                              .format = bf16(4),
                              .extent = {.file = 0, .offset = 128, .bytes = 32},
                              .dtype = "BF16",
                              .dims = {4, 4}});
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
                       .format = bf16(4),
                       .parts = {{.part = format::PartRole::kData, .sources = {0}}}});
    outputs.push_back({.name = "draft_output.weight",
                       .format = bf16(3),
                       .parts = {{.part = format::PartRole::kData,
                                  .kind = PartAssemblyKind::kRows,
                                  .sources = {1, 2}}}});
    std::vector<SourceDisposition> dispositions(source.tensors.size());
    auto cost = IndexedPartPlan::Bound(source, outputs);
    if (!cost) return std::unexpected(cost.error());
    auto parts = IndexedPartPlan::Make(std::move(source), outputs, dispositions, *cost);
    if (!parts) return std::unexpected(parts.error());
    spec::Source canonical{.architecture = "qwen4exp-mtp", .container = "safetensors"};
    canonical.tensors = {
        {.name = "fc.weight", .format = bf16(4)},
        {.name = "draft_output.weight", .format = bf16(3)},
        {.name = "draft_output.ids",
         .format = {.encoding = Encoding::kI32, .shape = {.ne = {1, 3, 1, 1}, .rank = 2}}}};
    spec::ModelSpec model;
    model.stages.resize(2);
    model.stages[0].name = "target";
    model.stages[1] = {.name = "mtp", .role = spec::StageRole::kDrafter, .artifact = 1};
    model.stages[1].layers.resize(1);
    model.stages[1].layers[0].components.push_back(
        {.kind = "x", .weights = {{.role = "w", .tensor = "fc.weight"}}});
    model.stages[1].heads.output =
        spec::ComponentRef{.kind = "head.linear",
                           .weights = {{.role = "ids", .tensor = "draft_output.ids"},
                                       {.role = "weight", .tensor = "draft_output.weight"}}};
    std::vector<GeneratedPart> generated{{.tensor = 2, .file = 1, .bytes = {}}};
    for (const int id : {1, 2, 5})
      for (int b = 0; b < 4; ++b)
        generated[0].bytes.push_back(static_cast<std::byte>(b == 0 ? id : 0));
    return test_support::PartPublicationFixture::MakeDrafter(
        std::move(*parts), std::move(canonical), std::move(model),
        {{0, format::PartRole::kData}, {1, format::PartRole::kData}}, std::move(generated),
        std::string(kOptions));
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
  EXPECT_FALSE(opened->spec_json().has_value());
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

TEST_F(ImportWriterTest, StorePermissionsAndOrphanedStagingJobsCleanup) {
  auto plan = Plan(false);
  ASSERT_TRUE(plan);
  auto result = WriteArtifact(*plan, Options());
  ASSERT_TRUE(result);

  std::error_code ec;
  auto dest_status = std::filesystem::status(result->path, ec);
  ASSERT_FALSE(ec);
  EXPECT_EQ(static_cast<unsigned>(dest_status.permissions() & std::filesystem::perms::mask), 0755U);

  auto data_status = std::filesystem::status(result->path / "data", ec);
  ASSERT_FALSE(ec);
  EXPECT_EQ(static_cast<unsigned>(data_status.permissions() & std::filesystem::perms::mask), 0755U);

  auto meta_status = std::filesystem::status(result->path / "meta", ec);
  ASSERT_FALSE(ec);
  EXPECT_EQ(static_cast<unsigned>(meta_status.permissions() & std::filesystem::perms::mask), 0755U);

  auto manifest_status = std::filesystem::status(result->path / "manifest.json", ec);
  ASSERT_FALSE(ec);
  EXPECT_EQ(static_cast<unsigned>(manifest_status.permissions() & std::filesystem::perms::mask),
            0644U);

  const auto staging = dir_ / "store" / ".staging";
  const auto orphan = staging / "job-orphan123";
  std::filesystem::create_directory(orphan, ec);
  ASSERT_FALSE(ec);
  EXPECT_TRUE(std::filesystem::exists(orphan));

  auto plan2 = Plan(true);
  ASSERT_TRUE(plan2);
  auto result2 = WriteArtifact(*plan2, Options());
  ASSERT_TRUE(result2);
  EXPECT_FALSE(std::filesystem::exists(orphan));
}

}  // namespace
}  // namespace llmp::import
