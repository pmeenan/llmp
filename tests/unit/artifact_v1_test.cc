// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The v1 prepared-artifact format (docs/artifact-format.md, "Version 1";
// D-113) through the reader (artifact/artifact.h, composition.h): valid
// artifacts written by the test-only writer (tests/support/artifact_writer.h)
// with every kind of format, part, fold, relayout, spec and expert array;
// the formats' JSON both ways; each rule broken one at a time with the rule
// and reason it is refused under; seeded mutations of the documents (ASan in
// check:full); and v1 compositions.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "artifact/error.h"
#include "artifact/format_json.h"
#include "artifact/json.h"
#include "artifact/layout.h"
#include "artifact_writer.h"
#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace {

namespace fs = std::filesystem;
namespace artifact = llmp::artifact;
namespace format = llmp::format;
namespace json = llmp::artifact::json;
using artifact::Artifact;
using artifact::Error;
using artifact::GroupKind;
using artifact::Rule;
using format::Encoding;
using format::Layout;
using format::PartRole;
using format::TensorFormat;
using llmp::base::Bytes;
using llmp::test::Json;
using llmp::test::V1Array;
using llmp::test::V1Files;
using llmp::test::V1Group;
using llmp::test::V1Plan;
using llmp::test::V1Resource;

// ---------------------------------------------------------------- helpers

std::string Env(const char* name) {
  const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value == nullptr ? std::string() : std::string(value);
}

fs::path Scratch(std::string_view name) {
  const std::string base = Env("LLMP_TEST_SCRATCH");
  const fs::path root = base.empty() ? fs::path(::testing::TempDir()) : fs::path(base);
  const fs::path dir = root / "artifact-v1" / name;
  std::error_code error;
  fs::remove_all(dir, error);
  fs::create_directories(dir, error);
  return dir;
}

void WriteFile(const fs::path& path, std::string_view data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::string Hex(std::string_view bytes) {
  return llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish());
}

std::string Describe(const std::expected<Artifact, Error>& opened) {
  return opened ? std::string("ok") : opened.error().ToString();
}

TensorFormat Dense(Encoding e, std::initializer_list<std::uint64_t> ne) {
  TensorFormat f{.encoding = e, .layout = Layout::kDense};
  f.shape = *format::ShapeOf(std::vector<std::uint64_t>(ne));
  return f;
}

TensorFormat Blocks(Encoding e, std::initializer_list<std::uint64_t> ne) {
  TensorFormat f = Dense(e, ne);
  f.layout = e == Encoding::kNvfp4 ? Layout::kGgmlNvfp4Blocks : Layout::kGgmlBlocks;
  return f;
}

TensorFormat Mxfp8Split(std::uint64_t k, std::uint64_t n) {
  TensorFormat f = Dense(Encoding::kMxfp8, {k, n});
  f.layout = Layout::kSplit;
  return f;
}

// gate and up as one stacked NVFP4 matrix in CUTLASS's scale atoms, with
// a tensor scale per stacked matrix.
TensorFormat Nvfp4Atoms(std::uint64_t k, std::uint64_t n) {
  TensorFormat f = Dense(Encoding::kNvfp4, {k, n});
  f.layout = Layout::kScaleAtoms;
  f.params.tensor_scale = true;
  f.stacked = 2;
  return f;
}

TensorFormat Trellis(std::uint64_t k, std::uint64_t n) {
  TensorFormat f = Dense(Encoding::kExl3Trellis, {k, n});
  f.layout = Layout::kExl3Tiles;
  f.params.k_bits = 4;
  f.params.codebook = format::Exl3Codebook::kMcg;
  return f;
}

TensorFormat Affine(std::uint64_t k, std::uint64_t n) {
  TensorFormat f = Dense(Encoding::kAffine, {k, n});
  f.layout = Layout::kSplit;
  f.params.bits = 4;
  f.params.group = 32;
  f.params.scale = format::AffineScale::kBF16;
  return f;
}

constexpr std::string_view kSpecJson = R"({"pipeline":"llm","stages":[],"version":1})"
                                       "\n";

// Every kind of v1 entry: a row table with a tied alias, dense and GGML
// block tensors (one with GGML's over-read), an MXFP8 pair and an affine
// triple in split parts, an EXL3 trellis's four parts, an expert layer of
// NVFP4 arrays in CUTLASS's scale atoms with their tensor scale as a
// resource, and a head group in a second shard; one fold, one relayout,
// kept metadata and a spec.
V1Plan FullPlan() {
  V1Plan p;
  p.architecture = "toy";
  p.container = "safetensors";
  p.expert_count = 4;
  p.meta = {{"tokenizer.json", R"({"model":{}})"}};
  p.sources = {{"model.safetensors", "the checkpoint"}};
  p.spec = std::string(kSpecJson);
  // The stacked NVFP4 tensor's scale part over its four experts.
  TensorFormat experts = Nvfp4Atoms(64, 256);
  experts.shape.ne[2] = 4;
  experts.shape.rank = 3;
  p.groups = {
      V1Group{.kind = GroupKind::kTable,
              .resources = {{.name = "token_embd.weight",
                             .aliases = {"output.weight"},
                             .format = Blocks(Encoding::kQ8_0, {64, 100}),
                             .rows = true}}},
      V1Group{.kind = GroupKind::kLayer,
              .layer = 0,
              .resources =
                  {
                      {.name = "blk.0.attn_norm.weight", .format = Dense(Encoding::kF32, {64})},
                      {.name = "blk.0.attn_q.weight", .format = Blocks(Encoding::kQ4_K, {256, 8})},
                      {.name = "blk.0.attn_k.weight",
                       .format = Mxfp8Split(64, 32),
                       .part = PartRole::kCodes},
                      {.name = "blk.0.attn_k.weight_scale",
                       .format = Mxfp8Split(64, 32),
                       .part = PartRole::kBlockScales},
                      {.name = "blk.0.attn_v.weight",
                       .format = Affine(64, 16),
                       .part = PartRole::kCodes},
                      {.name = "blk.0.attn_v.scales",
                       .format = Affine(64, 16),
                       .part = PartRole::kBlockScales},
                      {.name = "blk.0.attn_v.biases",
                       .format = Affine(64, 16),
                       .part = PartRole::kBiases},
                      {.name = "blk.0.ffn_down.trellis",
                       .format = Trellis(128, 64),
                       .part = PartRole::kCodes},
                      {.name = "blk.0.ffn_down.suh",
                       .format = Trellis(128, 64),
                       .part = PartRole::kInputSigns},
                      {.name = "blk.0.ffn_down.svh",
                       .format = Trellis(128, 64),
                       .part = PartRole::kOutputSigns},
                      {.name = "blk.0.ffn_down.mcg",
                       .format = Trellis(128, 64),
                       .part = PartRole::kCodebookMarker},
                  }},
      V1Group{.kind = GroupKind::kLayer,
              .layer = 1,
              .resources = {{.name = "blk.1.ffn_gate_up_exps.weight_scale_2",
                             .format = experts,
                             .part = PartRole::kTensorScale}}},
      V1Group{.kind = GroupKind::kExpert,
              .layer = 1,
              .arrays = {{.name = "blk.1.ffn_gate_up_exps.codes",
                          .format = Nvfp4Atoms(64, 256),
                          .part = PartRole::kCodes},
                         {.name = "blk.1.ffn_gate_up_exps.scales",
                          .format = Nvfp4Atoms(64, 256),
                          .part = PartRole::kBlockScales}}},
      V1Group{.kind = GroupKind::kHead,
              .resources = {{.name = "output_norm.weight", .format = Dense(Encoding::kF32, {64})}},
              .new_shard = true},
  };
  p.folds = {{"norm.plus_one", {"blk.0.attn_norm.weight", "output_norm.weight"}}};
  p.relayouts = {
      {Layout::kSplit, {"blk.1.ffn_gate_up_exps.codes", "blk.1.ffn_gate_up_exps.scales"}}};
  return p;
}

// A resource's (or expert array's) index entry in a rendered index.
Json& Entry(V1Files& files, std::string_view list, std::string_view name) {
  for (Json& e : files.index[list].array()) {
    std::string n;
    e["name"].Write(n);
    if (n == "\"" + std::string(name) + "\"") {
      return e;
    }
  }
  ADD_FAILURE() << "no entry " << name;
  static Json none;
  return none;
}

// One manifest transformation of `kind` (the first).
Json& Transformation(V1Files& files, std::string_view kind) {
  for (Json& t : files.manifest["transformations"].array()) {
    std::string k;
    t["kind"].Write(k);
    if (k == "\"" + std::string(kind) + "\"") {
      return t;
    }
  }
  ADD_FAILURE() << "no transformation " << kind;
  static Json none;
  return none;
}

void Replace(std::string& text, std::string_view from, std::string_view to) {
  const auto at = text.find(from);
  ASSERT_NE(at, std::string::npos) << from;
  text.replace(at, from.size(), to);
}

// ------------------------------------------------------------------ valid

TEST(ArtifactV1Test, OpensEveryKindOfEntry) {
  const fs::path store = Scratch("full");
  const V1Plan plan = FullPlan();
  const fs::path root = Publish(store, Render(plan));
  const auto opened = Artifact::Open(root);
  ASSERT_TRUE(opened.has_value()) << Describe(opened);
  const Artifact& a = *opened;
  EXPECT_EQ(a.format_version(), 1U);
  EXPECT_EQ(a.model().architecture, "toy");
  EXPECT_EQ(a.model().container, "safetensors");
  EXPECT_EQ(a.model().expert_count, 4U);
  EXPECT_EQ(a.converter().name, "llmp-import-test");
  EXPECT_EQ(a.converter().build, "test");
  EXPECT_EQ(llmp::base::ToHex(a.converter().options_sha256), Hex(""));

  // The spec: the bytes Open hashed, whose digest the manifest lists.
  ASSERT_TRUE(a.spec_json().has_value());
  EXPECT_EQ(*a.spec_json(), kSpecJson);
  const auto spec_file = std::ranges::find(a.files(), "spec.json", &artifact::ListedFile::path);
  ASSERT_NE(spec_file, a.files().end());
  EXPECT_EQ(spec_file->role, artifact::FileRole::kSpec);
  EXPECT_EQ(llmp::base::ToHex(spec_file->sha256), Hex(kSpecJson));

  ASSERT_EQ(a.folds().size(), 1U);
  EXPECT_EQ(a.folds()[0].name, "norm.plus_one");
  EXPECT_EQ(a.folds()[0].tensors,
            (std::vector<std::string>{"blk.0.attn_norm.weight", "output_norm.weight"}));
  ASSERT_EQ(a.relayouts().size(), 1U);
  EXPECT_EQ(a.relayouts()[0].from, Layout::kSplit);

  // Formats and parts as written; the tied alias finds its resource.
  const auto embd = a.FindResource("output.weight");
  ASSERT_TRUE(embd.has_value());
  EXPECT_EQ(a.resources()[*embd].name, "token_embd.weight");
  EXPECT_EQ(a.resources()[*embd].format, Blocks(Encoding::kQ8_0, {64, 100}));
  const auto scales = a.FindResource("blk.0.attn_k.weight_scale");
  ASSERT_TRUE(scales.has_value());
  EXPECT_EQ(a.resources()[*scales].format, Mxfp8Split(64, 32));
  EXPECT_EQ(a.resources()[*scales].part, PartRole::kBlockScales);
  EXPECT_EQ(a.resources()[*scales].bytes.value(), 32U * 2);  // two E8M0 scales a row
  const auto marker = a.FindResource("blk.0.ffn_down.mcg");
  ASSERT_TRUE(marker.has_value());
  EXPECT_EQ(a.resources()[*marker].bytes.value(), 4U);
  // Q4_K at K = 256 reserves GGML's over-read to the next 512 values.
  const auto q = a.FindResource("blk.0.attn_q.weight");
  ASSERT_TRUE(q.has_value());
  EXPECT_EQ(a.resources()[*q].bytes.value(), 8U * 144);
  EXPECT_EQ(a.resources()[*q].readable.value(), (8U * 144) + 144);

  const auto codes = a.FindExpertArray("blk.1.ffn_gate_up_exps.codes");
  ASSERT_TRUE(codes.has_value());
  const artifact::ExpertArray& arr = a.expert_arrays()[*codes];
  EXPECT_EQ(arr.count, 4U);
  EXPECT_EQ(arr.format, Nvfp4Atoms(64, 256));
  EXPECT_EQ(arr.part, PartRole::kCodes);
  for (std::uint32_t e = 0; e < arr.count; ++e) {
    const auto placed = a.SlicePlacement(*codes, e);
    ASSERT_TRUE(placed.has_value());
    EXPECT_EQ(placed->group, arr.first_group + e);
    EXPECT_EQ(placed->bytes.value(), 64U / 2 * 256);
  }

  // The row table: Q8_0 rows of 2 blocks of 34 bytes.
  const auto rows = a.Rows(*embd);
  ASSERT_TRUE(rows.has_value());
  EXPECT_EQ(rows->rows, 100U);
  EXPECT_EQ(rows->row_bytes, 68U);
  const std::vector<std::uint64_t> want = {0, 99};
  EXPECT_TRUE(a.RowChunks(*embd, want).has_value());
  const std::vector<std::uint64_t> past = {100};
  EXPECT_FALSE(a.RowChunks(*embd, past).has_value());

  // Two shards, every group's chunks readable through the read plan.
  EXPECT_EQ(a.shards().size(), 2U);
  std::vector<artifact::ChunkKey> all;
  for (std::uint32_t g = 0; g < a.groups().size(); ++g) {
    for (std::uint32_t k = 0; k < a.groups()[g].chunks; ++k) {
      all.push_back({.group = g, .chunk = k});
    }
  }
  const auto runs = a.PlanReads(all, {});
  ASSERT_TRUE(runs.has_value());
  EXPECT_EQ(runs->size(), 2U);  // one run per shard

  const auto kept = a.ReadMetadata("tokenizer.json");
  ASSERT_TRUE(kept.has_value());
  EXPECT_EQ(*kept, R"({"model":{}})");
}

// What a v1 reader accepts beyond v0: parts sized by their format alone,
// without a closure rule (an EXL3 trellis without its sign vectors; which
// parts make one tensor is the spec's binding), expert slices of any
// encoding, and no spec, folds or kept metadata at all.
TEST(ArtifactV1Test, AcceptsWhatIsDescribedCorrectly) {
  const fs::path store = Scratch("permissive");
  V1Plan p;
  p.expert_count = 2;
  p.groups = {
      V1Group{.kind = GroupKind::kLayer,
              .layer = 0,
              .resources = {{.name = "blk.0.ffn_down.trellis",
                             .format = Trellis(128, 64),
                             .part = PartRole::kCodes}}},
      V1Group{.kind = GroupKind::kExpert,
              .layer = 0,
              .arrays = {{.name = "blk.0.ffn_down_exps.weight",
                          .format = Mxfp8Split(64, 32),
                          .part = PartRole::kCodes}}},
  };
  const auto opened = Artifact::Open(Publish(store, Render(p)));
  ASSERT_TRUE(opened.has_value()) << Describe(opened);
  EXPECT_FALSE(opened->spec_json().has_value());
  EXPECT_TRUE(opened->folds().empty());
  EXPECT_EQ(opened->converter().name, "llmp-import-test");
  EXPECT_EQ(opened->converter().version, "1");
  EXPECT_EQ(opened->converter().build, "test");
  EXPECT_TRUE(opened->converter().mapping_architecture.empty());
  EXPECT_EQ(opened->converter().mapping_version, 0U);

  V1Files with_mapping = Render(p);
  with_mapping.manifest["importer"]["mapping_architecture"] = "qwen2";
  with_mapping.manifest["importer"]["mapping_version"] = 1;
  const fs::path mapped_root = Publish(store, with_mapping);
  const auto mapped = Artifact::Open(mapped_root);
  ASSERT_TRUE(mapped.has_value()) << Describe(mapped);
  EXPECT_EQ(mapped->converter().mapping_architecture, "qwen2");
  EXPECT_EQ(mapped->converter().mapping_version, 1U);

  // Raw manifest SHA-256 (when importer.build != "unknown") is not accepted as the artifact ID.
  std::ifstream in(mapped_root / "manifest.json", std::ios::binary);
  const std::string raw_manifest((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  const auto raw_sha_open = Artifact::Open(mapped_root, {.expected_id = Hex(raw_manifest)});
  ASSERT_FALSE(raw_sha_open.has_value());
  EXPECT_EQ(raw_sha_open.error().rule, Rule::kIdentity);
}

TEST(ArtifactV1Test, FormatsRoundTripThroughTheirJson) {
  TensorFormat scalar = Dense(Encoding::kI32, {});
  TensorFormat four = Dense(Encoding::kBF16, {8, 4, 3, 2});
  TensorFormat zeros = Affine(64, 8);
  zeros.params.zero_point = true;
  zeros.params.scale = format::AffineScale::kF16;
  TensorFormat three = Trellis(32, 16);
  three.params.codebook = format::Exl3Codebook::k3Inst;
  three.params.k_bits = 8;
  TensorFormat interleaved = Dense(Encoding::kNvfp4, {64, 10});
  interleaved.layout = Layout::kRowInterleaved;
  TensorFormat fp8 = Dense(Encoding::kFp8Scaled, {5, 10, 2});
  fp8.layout = Layout::kSplit;
  fp8.stacked = 2;
  fp8.params.block_rows = 3;
  fp8.params.block_cols = 2;
  for (const TensorFormat& f :
       {scalar, four, zeros, three, interleaved, fp8, Nvfp4Atoms(64, 256), Mxfp8Split(64, 32),
        Blocks(Encoding::kNvfp4, {128, 2}), Blocks(Encoding::kIQ2_XXS, {256, 4}),
        Dense(Encoding::kE8M0, {7})}) {
    std::string text;
    artifact::AppendFormatJson(f, text);
    // A document is canonical with its one trailing newline.
    const std::string document = text + "\n";
    const auto doc = json::Parse(document);
    ASSERT_TRUE(doc.has_value()) << text;
    EXPECT_TRUE(doc->canonical()) << text;
    const auto back = artifact::ParseFormat(doc->root());
    ASSERT_TRUE(back.has_value()) << text << ": " << back.error().ToString();
    EXPECT_EQ(*back, f) << text;
  }
  std::string text;
  artifact::AppendFormatJson(Nvfp4Atoms(64, 256), text);
  EXPECT_EQ(text, R"({"encoding":"nvfp4","layout":"scale_atoms","ne":[64,256],)"
                  R"("params":{"tensor_scale":true},"stacked":2})");
  text.clear();
  artifact::AppendFormatJson(fp8, text);
  EXPECT_EQ(text, R"({"encoding":"fp8_scaled","layout":"split","ne":[5,10,2],)"
                  R"("params":{"block_cols":2,"block_rows":3},"stacked":2})");
}

// The container entry: a dense tensor keeps its dtype and shape (outermost
// first); every other part is bytes.
TEST(ArtifactV1Test, ContainerViews) {
  const auto dense = artifact::ContainerViewOf(Dense(Encoding::kBF16, {8, 4}), PartRole::kData, 64);
  EXPECT_EQ(dense.dtype, "BF16");
  EXPECT_EQ(dense.shape, (std::vector<std::uint64_t>{4, 8}));
  const auto e8m0 = artifact::ContainerViewOf(Dense(Encoding::kE8M0, {8}), PartRole::kData, 8);
  EXPECT_EQ(e8m0.dtype, "U8");
  const auto codes = artifact::ContainerViewOf(Mxfp8Split(64, 32), PartRole::kCodes, 2048);
  EXPECT_EQ(codes.dtype, "U8");
  EXPECT_EQ(codes.shape, (std::vector<std::uint64_t>{2048}));
}

// ---------------------------------------------------------------- refusals

struct Case {
  using Edit = std::function<void(V1Files&, llmp::test::PublishHooks&)>;
  Case(std::string_view n, Edit e, Rule r, std::string_view why,
       std::optional<std::uint64_t> at = std::nullopt)
      : name(n), edit(std::move(e)), rule(r), reason(why), item(at) {}

  std::string_view name;
  Edit edit;
  Rule rule;
  std::string_view reason;  // empty: any
  std::optional<std::uint64_t> item;
};

std::string Raw(const TensorFormat& f) {
  std::string text;
  artifact::AppendFormatJson(f, text);
  return text;
}

std::vector<Case> Cases() {
  using H = llmp::test::PublishHooks;
  return {
      // Versions: v1's format with a later or earlier version names it.
      {"future-version", [](V1Files& f, H&) { f.manifest["format_version"] = 2; },
       Rule::kUnsupportedVersion, "format_version is not one this build reads", 2},
      {"v1-format-version-0", [](V1Files& f, H&) { f.manifest["format_version"] = 0; },
       Rule::kUnsupportedVersion, "format_version is not one this build reads", 0},
      {"v0-format-version-1", [](V1Files& f, H&) { f.manifest["format"] = "jitllm-artifact"; },
       Rule::kFormat, "not a llmpalooza artifact"},
      {"unknown-format", [](V1Files& f, H&) { f.manifest["format"] = "llmp-artefact"; },
       Rule::kFormat, "not a llmpalooza artifact"},
      {"spark-v0-profile",
       [](V1Files& f, H&) {
         f.manifest["layout"]["profile"] = "spark-v0";
         f.manifest["layout"]["profile_version"] = 0;
       },
       Rule::kUnsupportedProfile, "layout is not spark v1; re-import required"},
      {"index-v0-header", [](V1Files& f, H&) { f.index["format"] = "jitllm-index"; },
       Rule::kUnsupportedVersion, "index header differs from the profile"},
      {"index-version-2", [](V1Files& f, H&) { f.index["format_version"] = 2; },
       Rule::kUnsupportedVersion, "index header differs from the profile"},
      // The manifest's v1 keys.
      {"converter-not-importer",
       [](V1Files& f, H&) {
         f.manifest["converter"] = f.manifest["importer"];
         f.manifest.Erase("importer");
       },
       Rule::kSchema, "manifest keys"},
      {"unknown-manifest-key", [](V1Files& f, H&) { f.manifest["spec"] = Json::Object{}; },
       Rule::kSchema, "manifest keys"},
      {"model-representation",
       [](V1Files& f, H&) { f.manifest["model"]["representation"] = Json::Array{"ggml"}; },
       Rule::kSchema, "model keys"},
      {"model-container-pattern", [](V1Files& f, H&) { f.manifest["model"]["container"] = "GGUF"; },
       Rule::kSchema, "model.container"},
      // A container this build does not know: its converter's folds would
      // be unknown, so it is refused, not read as another.
      {"model-container-unknown", [](V1Files& f, H&) { f.manifest["model"]["container"] = "onnx"; },
       Rule::kSchema, "model.container"},
      {"importer-options-not-hex",
       [](V1Files& f, H&) { f.manifest["importer"]["options_sha256"] = "00"; }, Rule::kSchema,
       "importer.options_sha256"},
      {"importer-build-missing", [](V1Files& f, H&) { f.manifest["importer"].Erase("build"); },
       Rule::kSchema, "importer keys"},
      {"importer-mapping-half-present",
       [](V1Files& f, H&) { f.manifest["importer"]["mapping_architecture"] = "qwen2"; },
       Rule::kSchema, "importer keys"},
      {"importer-mapping-version-zero",
       [](V1Files& f, H&) {
         f.manifest["importer"]["mapping_architecture"] = "qwen2";
         f.manifest["importer"]["mapping_version"] = 0;
       },
       Rule::kSchema, "importer.mapping_version"},
      {"importer-mapping-arch-invalid",
       [](V1Files& f, H&) {
         f.manifest["importer"]["mapping_architecture"] = "-bad";
         f.manifest["importer"]["mapping_version"] = 1;
       },
       Rule::kSchema, "importer.mapping_architecture"},
      // Folds and relayouts.
      {"fold-name-pattern",
       [](V1Files& f, H&) { Transformation(f, "fold")["fold"] = "Norm.plus_one"; }, Rule::kSchema,
       "transformation.fold"},
      {"fold-tensors-unsorted",
       [](V1Files& f, H&) {
         Transformation(f, "fold")["tensors"] =
             Json::Array{"output_norm.weight", "blk.0.attn_norm.weight"};
       },
       Rule::kCanonical, "a transformation's tensors not sorted or duplicated"},
      {"fold-tensors-empty",
       [](V1Files& f, H&) { Transformation(f, "fold")["tensors"] = Json::Array{}; }, Rule::kSchema,
       "transformation.tensors"},
      {"fold-names-an-alias",
       [](V1Files& f, H&) { Transformation(f, "fold")["tensors"] = Json::Array{"output.weight"}; },
       Rule::kSchema, "a fold or relayout names no resource or expert array"},
      {"fold-names-nothing",
       [](V1Files& f, H&) { Transformation(f, "fold")["tensors"] = Json::Array{"nothing"}; },
       Rule::kSchema, "a fold or relayout names no resource or expert array"},
      {"fold-twice",
       [](V1Files& f, H&) {
         Json again = Transformation(f, "fold");
         again["tensors"] = Json::Array{"token_embd.weight"};
         f.manifest["transformations"].array().push_back(again);
         auto& list = f.manifest["transformations"].array();
         std::ranges::sort(list, {}, [](const Json& j) {
           std::string out;
           j.Write(out);
           return out;
         });
       },
       Rule::kSchema, "a fold recorded twice"},
      {"relayout-from-unknown",
       [](V1Files& f, H&) { Transformation(f, "relayout")["from"] = "swizzled"; }, Rule::kSchema,
       "transformation.from is not a layout"},
      {"relayout-from-own-layout",
       [](V1Files& f, H&) { Transformation(f, "relayout")["from"] = "scale_atoms"; }, Rule::kRepr,
       "a relayout from its own layout, or one its format has not"},
      {"relayout-from-a-layout-that-cannot-hold-it",
       [](V1Files& f, H&) {
         Transformation(f, "relayout")["tensors"] = Json::Array{"blk.0.attn_q.weight"};
       },
       Rule::kRepr, "a relayout from its own layout, or one its format has not"},
      {"relayout-twice",
       [](V1Files& f, H&) {
         Json again = Transformation(f, "relayout");
         again["from"] = "ggml_nvfp4_blocks";
         f.manifest["transformations"].array().push_back(again);
         auto& list = f.manifest["transformations"].array();
         std::ranges::sort(list, {}, [](const Json& j) {
           std::string out;
           j.Write(out);
           return out;
         });
       },
       Rule::kSchema, "a tensor relaid out from two layouts"},
      {"unknown-transformation",
       [](V1Files& f, H&) {
         f.manifest["transformations"].array().push_back(
             Json::Object{{"kind", "requantize"}, {"tensors", Json::Array{"x"}}});
       },
       Rule::kSchema, "unknown transformation kind"},
      // The spec document.
      {"spec-path",
       [](V1Files& f, H& h) {
         h.manifest = [](std::string& t) {
           Replace(t, R"("path":"spec.json")", R"("path":"meta/spec.json")");
         };
         (void)f;
       },
       Rule::kPath, "a listed path does not fit its role"},
      {"spec-not-canonical",
       [](V1Files& f, H&) {
         f.spec = R"({"version": 1})"
                  "\n";
       },
       Rule::kCanonical, "spec.json is not in canonical form"},
      {"spec-not-an-object", [](V1Files& f, H&) { f.spec = "[1]\n"; }, Rule::kSchema,
       "spec.json is not an object"},
      {"spec-not-json", [](V1Files& f, H&) { f.spec = "{\"a\":}\n"; }, Rule::kJson, ""},
      {"spec-too-deep",
       [](V1Files& f, H&) {
         f.spec = R"({"a":[[[[[[[[1]]]]]]]]})"
                  "\n";
       },
       Rule::kJson, ""},
      {"spec-over-its-cap",
       [](V1Files& f, H&) {
         f.spec = R"({"a":")" + std::string(artifact::kMaxSpecBytes, 'x') + "\"}\n";
       },
       Rule::kFileSize, "document larger than its cap"},
      // Index entries.
      {"resource-repr-not-format",
       [](V1Files& f, H&) {
         Json& e = Entry(f, "resources", "blk.0.attn_norm.weight");
         e["repr"] = e["format"];
         e.Erase("format");
       },
       Rule::kSchema, "resource keys"},
      {"unknown-encoding",
       [](V1Files& f, H& h) {
         h.index = [](std::string& t) {
           Replace(t, R"("encoding":"q4_k")", R"("encoding":"q4_kk")");
         };
         (void)f;
       },
       Rule::kRepr, "an unknown encoding"},
      {"unknown-layout",
       [](V1Files& f, H& h) {
         h.index = [](std::string& t) {
           Replace(t, R"("layout":"exl3_tiles")", R"("layout":"tiles")");
         };
         (void)f;
       },
       Rule::kRepr, "an unknown layout"},
      {"layout-cannot-hold-encoding",
       [](V1Files& f, H&) {
         TensorFormat dense_q4 = Blocks(Encoding::kQ4_K, {256, 8});
         dense_q4.layout = Layout::kDense;
         Entry(f, "resources", "blk.0.attn_q.weight")["format"] = Json::Raw{Raw(dense_q4)};
       },
       Rule::kRepr, "layout does not hold the encoding"},
      {"rows-not-whole-blocks",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.0.attn_q.weight")["format"] = Json::Raw{
             R"({"encoding":"q4_k","layout":"ggml_blocks","ne":[255,8],"params":{},"stacked":1})"};
       },
       Rule::kRepr, "rows not whole blocks"},
      {"param-at-default",
       [](V1Files& f, H& h) {
         h.index = [](std::string& t) {
           Replace(t, R"("params":{"tensor_scale":true})", R"("params":{"tensor_scale":false})");
         };
         (void)f;
       },
       Rule::kCanonical, "a format parameter written at its default"},
      {"param-zero",
       [](V1Files& f, H& h) {
         h.index = [](std::string& t) { Replace(t, R"("k_bits":4)", R"("k_bits":0)"); };
         (void)f;
       },
       Rule::kCanonical, "a format parameter written at its default"},
      {"param-unknown",
       [](V1Files& f, H& h) {
         h.index = [](std::string& t) {
           Replace(t, R"("params":{"tensor_scale":true})", R"("params":{"swizzle":true})");
         };
         (void)f;
       },
       Rule::kSchema, "format.params keys"},
      {"param-the-encoding-does-not-take",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.0.attn_norm.weight")["format"] = Json::Raw{
             R"({"encoding":"f32","layout":"dense","ne":[64],"params":{"bits":4},"stacked":1})"};
       },
       Rule::kRepr, "bad encoding parameters"},
      {"fp8-param-other-encoding",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.0.attn_norm.weight")["format"] = Json::Raw{
             R"({"encoding":"f32","layout":"dense","ne":[64],"params":{"block_cols":2},"stacked":1})"};
       },
       Rule::kRepr, "bad encoding parameters"},
      {"unknown-codebook",
       [](V1Files& f, H& h) {
         h.index = [](std::string& t) { Replace(t, R"("codebook":"mcg")", R"("codebook":"xyz")"); };
         (void)f;
       },
       Rule::kRepr, "an unknown EXL3 codebook"},
      {"five-dimensions",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.0.attn_norm.weight")["format"] = Json::Raw{
             R"({"encoding":"f32","layout":"dense","ne":[64,1,1,1,1],"params":{},"stacked":1})"};
       },
       Rule::kSchema, "format.ne"},
      {"zero-dimension",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.0.attn_norm.weight")["format"] =
             Json::Raw{R"({"encoding":"f32","layout":"dense","ne":[0],"params":{},"stacked":1})"};
       },
       Rule::kSchema, "format.ne"},
      {"size-overflow",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.0.attn_norm.weight")["format"] = Json::Raw{
             R"({"encoding":"f32","layout":"dense","ne":[4611686018427387904,4],"params":{},"stacked":1})"};
       },
       Rule::kRepr, "size beyond 64 bits"},
      {"stacked-zero",
       [](V1Files& f, H& h) {
         h.index = [](std::string& t) { Replace(t, R"("stacked":2)", R"("stacked":0)"); };
         (void)f;
       },
       Rule::kSchema, "format.stacked"},
      {"stacked-not-dividing-rows",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.0.attn_norm.weight")["format"] = Json::Raw{
             R"({"encoding":"f32","layout":"dense","ne":[64,3],"params":{},"stacked":2})"};
       },
       Rule::kRepr, "bad shape"},
      {"unknown-part",
       [](V1Files& f, H&) { Entry(f, "resources", "blk.0.attn_norm.weight")["part"] = "weights"; },
       Rule::kRepr, "an unknown part role"},
      {"part-the-format-lacks",
       [](V1Files& f, H&) { Entry(f, "resources", "blk.0.attn_norm.weight")["part"] = "codes"; },
       Rule::kRepr, "a part its format does not have"},
      {"marker-of-a-3inst-trellis",
       [](V1Files& f, H& h) {
         h.index = [](std::string& t) {
           const std::string from = R"("codebook":"mcg")";
           // Only the marker's entry: the last trellis format in the index.
           const auto at = t.rfind(from);
           ASSERT_NE(at, std::string::npos);
           t.replace(at, from.size(), R"("codebook":"3inst")");
         };
         (void)f;
       },
       Rule::kRepr, "a part its format does not have"},
      {"bytes-differ",
       [](V1Files& f, H&) { Entry(f, "resources", "blk.0.attn_norm.weight")["bytes"] = 255; },
       Rule::kRepr, "bytes differ from what the format implies"},
      {"over-read-left-out",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.0.attn_q.weight")["readable_bytes"] = 8 * 144;
       },
       Rule::kRepr, "readable_bytes differ from what the format implies"},
      {"row-table-of-scale-atoms",
       [](V1Files& f, H&) {
         Entry(f, "resources", "blk.1.ffn_gate_up_exps.weight_scale_2")["access"] = "rows";
       },
       Rule::kRepr, "a row table must be one matrix of addressable rows"},
      {"expert-slice-of-three-dimensions",
       [](V1Files& f, H&) {
         TensorFormat three = Nvfp4Atoms(64, 256);
         three.shape.ne[2] = 2;
         three.shape.rank = 3;
         Entry(f, "expert_arrays", "blk.1.ffn_gate_up_exps.codes")["format"] =
             Json::Raw{Raw(three)};
       },
       Rule::kExpertArray, "a slice must be one matrix of at most two dimensions"},
      {"slice-bytes-differ",
       [](V1Files& f, H&) {
         Entry(f, "expert_arrays", "blk.1.ffn_gate_up_exps.codes")["slice_bytes"] = 4096;
       },
       Rule::kRepr, "slice sizes disagree with the format"},
      // A format of the same size but another dtype: the shard header the
      // index implies is not the one written.
      {"header-follows-the-format",
       [](V1Files& f, H&) {
         Entry(f, "resources", "output_norm.weight")["format"] =
             Json::Raw{Raw(Dense(Encoding::kI32, {64}))};
       },
       Rule::kContainer, "the shard header disagrees with the index"},
  };
}

TEST(ArtifactV1Test, RefusesEachRuleWithItsReason) {
  const fs::path store = Scratch("cases");
  const V1Plan plan = FullPlan();
  for (const Case& c : Cases()) {
    SCOPED_TRACE(c.name);
    V1Files files = Render(plan);
    llmp::test::PublishHooks hooks;
    c.edit(files, hooks);
    const auto opened = Artifact::Open(Publish(store, files, hooks));
    ASSERT_FALSE(opened.has_value());
    EXPECT_EQ(artifact::RuleName(opened.error().rule), artifact::RuleName(c.rule))
        << opened.error().ToString();
    if (!c.reason.empty()) {
      EXPECT_EQ(opened.error().reason, c.reason) << opened.error().ToString();
    }
    if (c.item) {
      EXPECT_EQ(opened.error().item, *c.item);
    }
  }
}

// The files around the documents: spec.json's digest, presence and listing.
TEST(ArtifactV1Test, SpecFileMustBeTheListedOne) {
  const fs::path store = Scratch("spec-files");
  const V1Files files = Render(FullPlan());
  {
    const fs::path root = Publish(store, files);
    std::string changed(kSpecJson);
    changed[2] = 'g';  // same size, other bytes
    WriteFile(root / "spec.json", changed);
    const auto opened = Artifact::Open(root);
    ASSERT_FALSE(opened.has_value());
    EXPECT_EQ(opened.error().rule, Rule::kHash);
    EXPECT_EQ(opened.error().reason, "spec.json differs from its manifest digest");
  }
  {
    const fs::path root = Publish(store, files);
    fs::remove(root / "spec.json");
    const auto opened = Artifact::Open(root);
    ASSERT_FALSE(opened.has_value());
    EXPECT_EQ(opened.error().rule, Rule::kFileSet);
  }
  {
    V1Files unlisted = files;
    unlisted.spec.reset();
    const fs::path root = Publish(store, unlisted);
    WriteFile(root / "spec.json", kSpecJson);
    const auto opened = Artifact::Open(root);
    ASSERT_FALSE(opened.has_value());
    EXPECT_EQ(opened.error().rule, Rule::kFileSet);
  }
}

// Seeded mutations of the three documents: whatever they do, the reader
// refuses with a rule or accepts, never crashing or reading out of bounds
// (ASan in check:full), and an accepted one has every placement inside its
// group.
TEST(ArtifactV1Test, MutatedDocumentsFailClosed) {
  const fs::path store = Scratch("fuzz");
  const V1Files files = Render(FullPlan());
  std::string manifest;
  {
    // The published manifest, its file list filled in.
    std::ifstream in(Publish(store, files) / "manifest.json", std::ios::binary);
    manifest.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  const std::string index = files.index.Text();
  std::mt19937_64 random(20261009);
  const std::string alphabet = R"({}[]",:0123456789-ae truflsn#.~\)";
  const auto mutate = [&](std::string text) {
    const int edits = 1 + static_cast<int>(random() % 3);
    for (int e = 0; e < edits && !text.empty(); ++e) {
      const std::size_t at = random() % text.size();
      switch (random() % 5) {
        case 0:
          text[at] = alphabet[random() % alphabet.size()];
          break;
        case 1:
          text.erase(at, 1 + (random() % 8));
          break;
        case 2:
          text.insert(at, 1, alphabet[random() % alphabet.size()]);
          break;
        case 3: {  // duplicate a span
          const std::size_t length = std::min<std::size_t>(1 + (random() % 64), text.size() - at);
          text.insert(at, text.substr(at, length));
          break;
        }
        default: {  // a digit run made large
          if (text[at] >= '0' && text[at] <= '9') {
            text.insert(at, "9999999999");
          }
          break;
        }
      }
    }
    return text;
  };
  std::size_t accepted = 0;
  std::size_t refused = 0;
  for (int n = 0; n < 600; ++n) {
    llmp::test::PublishHooks hooks;
    V1Files edited = files;
    switch (n % 3) {
      case 0: {
        const std::string text = mutate(manifest);
        hooks.manifest = [text](std::string& t) { t = text; };
        break;
      }
      case 1: {
        const std::string text = mutate(index);
        hooks.index = [text](std::string& t) { t = text; };
        break;
      }
      default:
        edited.spec = mutate(std::string(kSpecJson));
        break;
    }
    const auto opened = Artifact::Open(Publish(store, edited, hooks));
    if (!opened) {
      ++refused;
      continue;
    }
    ++accepted;
    for (std::uint32_t r = 0; r < opened->resources().size(); ++r) {
      const auto placed = opened->ResourcePlacement(r);
      ASSERT_TRUE(placed.has_value());
      EXPECT_LE(placed->offset.value() + placed->readable.value(),
                opened->groups()[placed->group].stored.value());
    }
  }
  EXPECT_GT(refused, 0U);
  RecordProperty("accepted", static_cast<int>(accepted));
}

// --------------------------------------------------------------- compositions

TEST(CompositionV1Test, OpensWithASpecAndNoKeptFiles) {
  const fs::path store = Scratch("composition");
  llmp::test::CompositionPlan plan;
  plan.architecture = "gemma4";
  plan.components = {{"assistant", std::string(64, 'b'), "gemma4-assistant"},
                     {"target", std::string(64, 'a'), "gemma4"}};
  plan.spec = std::string(kSpecJson);
  const fs::path root = Publish(store, Render(plan));
  const auto opened = artifact::OpenComposition(root);
  ASSERT_TRUE(opened.has_value()) << opened.error().ToString();
  EXPECT_EQ(opened->format_version(), 1U);
  EXPECT_EQ(opened->components().size(), 2U);
  EXPECT_TRUE(opened->sources().empty());
  ASSERT_TRUE(opened->spec_json().has_value());
  EXPECT_EQ(*opened->spec_json(), kSpecJson);
  EXPECT_EQ(opened->converter().build, "test");
}

TEST(CompositionV1Test, KeepsAPipelinesFilesWithoutASpec) {
  const fs::path store = Scratch("composition-kept");
  llmp::test::CompositionPlan plan;
  plan.components = {{"transformer", std::string(64, 'c'), "QwenImage21Transformer2DModel"},
                     {"vae", std::string(64, 'd'), "AutoencoderKLQwenImage21"}};
  plan.meta = {{"model_index.json", R"({"_class_name":"QwenImage21Pipeline"})"},
               {"scheduler_config.json", R"({"shift":1})"}};
  const auto opened = artifact::OpenComposition(Publish(store, Render(plan)));
  ASSERT_TRUE(opened.has_value()) << opened.error().ToString();
  EXPECT_FALSE(opened->spec_json().has_value());
  EXPECT_TRUE(opened->Metadata("scheduler_config.json").has_value());
}

TEST(CompositionV1Test, Refusals) {
  const fs::path store = Scratch("composition-refusals");
  llmp::test::CompositionPlan plan;
  plan.components = {{"target", std::string(64, 'a'), "gemma4"}};
  plan.meta = {{"model_index.json", "{}"}};
  plan.spec = std::string(kSpecJson);
  const auto refused =
      [&](const std::function<void(llmp::test::CompositionFiles&, llmp::test::PublishHooks&)>& edit,
          Rule rule, std::string_view reason, bool replace_spec = false) {
        llmp::test::CompositionFiles files = Render(plan);
        llmp::test::PublishHooks hooks;
        edit(files, hooks);
        const fs::path root = Publish(store, files, hooks);
        if (replace_spec) {
          std::string changed(kSpecJson);
          changed[2] = 'g';
          WriteFile(root / "spec.json", changed);
        }
        const auto opened = artifact::OpenComposition(root);
        ASSERT_FALSE(opened.has_value());
        EXPECT_EQ(artifact::RuleName(opened.error().rule), artifact::RuleName(rule))
            << opened.error().ToString();
        if (!reason.empty()) {
          EXPECT_EQ(opened.error().reason, reason);
        }
      };
  refused([](auto& f, auto&) { f.manifest["format_version"] = 3; }, Rule::kUnsupportedVersion,
          "format_version is not one this build reads");
  refused([](auto& f, auto&) { f.manifest["format"] = "jitllm-composition"; }, Rule::kFormat,
          "not a llmpalooza composition");
  refused(
      [](auto& f, auto&) {
        f.manifest["converter"] = f.manifest["importer"];
        f.manifest.Erase("importer");
      },
      Rule::kSchema, "composition keys");
  // The importer member is an artifact's (the shared reader), without the
  // architecture mapping an artifact may name.
  refused(
      [](auto& f, auto&) {
        f.manifest["importer"]["mapping_architecture"] = "gemma4";
        f.manifest["importer"]["mapping_version"] = 1;
      },
      Rule::kSchema, "importer keys");
  refused([](auto& f, auto&) { f.manifest["importer"]["options_sha256"] = "00"; }, Rule::kSchema,
          "importer.options_sha256");
  refused([](auto& f, auto&) { f.manifest["importer"].Erase("build"); }, Rule::kSchema,
          "importer keys");
  refused([](auto&, auto&) {}, Rule::kHash, "spec.json differs from its manifest digest", true);
  refused([](auto& f, auto&) { f.spec = "[]\n"; }, Rule::kSchema, "spec.json is not an object");
  refused(
      [](auto&, auto& h) {
        h.manifest = [](std::string& t) {
          Replace(t, R"("path":"spec.json","role":"spec")",
                  R"("path":"spec.json","role":"source-metadata")");
        };
      },
      Rule::kSchema, "");
  refused(
      [](auto& f, auto&) {
        f.keep_files = true;
        f.manifest["files"] = Json::Array{};
      },
      Rule::kFileSet, "files differ from the manifest's list");
}

}  // namespace
