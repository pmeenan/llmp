// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <map>
#include <variant>

#include "base/check.h"
#include "base/json.h"
#include "cli/cli.h"
#include "components/step.h"
#include "expected_error.h"
#include "import/archive_input.h"
#include "import/normalize.h"
#include "import/readers/gguf.h"
#include "import/readers/safetensors.h"
#include "import/stream.h"
#include "import/writer.h"
#include "import_fixtures.h"
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
}  // namespace
}  // namespace llmp::import
