// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// v0's representation descriptors as tensor formats (artifact/format_map.h,
// D-108): the size rule (format::PartsOf) gives exactly the bytes and
// readable range ParseRepresentation implies, for every GGML type and
// shape, every EXL3 case and every dtype; pairings assemble and are refused
// as they should; and every resource and expert array of the reader's
// corpus, and of the installed real artifacts where a host has them (a
// Spark), maps to a format of exactly its recorded size, with the Qwen3.8
// import's multi-part tensors (MXFP8, CUTLASS experts, the n-gram table)
// assembling.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/format_map.h"
#include "artifact/json.h"
#include "artifact/representation.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace {

namespace ar = llmp::artifact;
namespace format = llmp::format;
namespace fs = std::filesystem;
using format::Encoding;
using format::FormatError;
using format::Layout;
using format::PartRole;

constexpr std::uint64_t kSaturated = std::numeric_limits<std::uint64_t>::max();

std::string Env(const char* name) {
  const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value == nullptr ? std::string() : std::string(value);
}

// A representation parsed from its JSON (which must outlive nothing: its
// text fields point into static tables).
std::optional<ar::Representation> Parse(const std::string& text) {
  const auto doc = ar::json::Parse(text);
  if (!doc) {
    return std::nullopt;
  }
  auto rep = ar::ParseRepresentation(doc->root());
  if (!rep) {
    return std::nullopt;
  }
  return *rep;
}

std::string List(const std::vector<std::uint64_t>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    out += (i == 0 ? "" : ",") + std::to_string(values[i]);
  }
  return out + "]";
}

// The resource's format, and its part's extent, agree with what v0 records.
void ExpectSized(const ar::Representation& rep, std::uint64_t bytes, std::uint64_t readable,
                 const std::string& what) {
  const auto f = ar::FormatOf(rep);
  ASSERT_TRUE(f.has_value()) << what << ": " << format::FormatErrorName(f.error());
  const auto parts = format::PartsOf(*f);
  ASSERT_TRUE(parts.has_value()) << what;
  // A trellis is its format's codes; any other resource is its own data.
  const auto part =
      parts->Find(f->encoding == Encoding::kExl3Trellis ? PartRole::kCodes : PartRole::kData);
  ASSERT_TRUE(part.has_value()) << what;
  EXPECT_EQ(part->bytes, bytes) << what;
  EXPECT_EQ(part->readable, readable) << what;
}

TEST(ArtifactFormatMapTest, GgmlTypesSizeAsTheV0Rule) {
  const std::vector<std::vector<std::uint64_t>> shapes = {
      {256},
      {512, 3},
      {640, 2560},
      {2560, 640, 4},
      {1024, 7, 3, 2},
      {7168, 1},
      {160, 37},
      {4096, 1},
      {96, 5},
      {32},
      {1ULL << 40, 1ULL << 30},
      {1ULL << 62, 1ULL << 20},
  };
  std::size_t compared = 0;
  for (const ar::GgmlType& type : ar::GgmlTypes()) {
    SCOPED_TRACE(std::string(type.name));
    const auto encoding = ar::EncodingOfGgmlType(type.name);
    if (type.name == "Q8_1") {
      // v0 records gguf-py's 40 bytes; GGML's block at the pin is 36.
      EXPECT_FALSE(encoding.has_value());
      EXPECT_EQ(type.block_bytes, 40U);
      EXPECT_EQ(format::Info(Encoding::kQ8_1).ggml->block_bytes, 36U);
      continue;
    }
    ASSERT_TRUE(encoding.has_value());
    const format::EncodingInfo& info = format::Info(*encoding);
    ASSERT_TRUE(info.ggml.has_value());
    EXPECT_EQ(info.ggml->type, type.id);
    EXPECT_EQ(info.ggml->block_elements, type.block_elements);
    EXPECT_EQ(info.ggml->block_bytes, type.block_bytes);
    for (const auto& ne : shapes) {
      const std::string json = R"({"family":"ggml","ne":)" + List(ne) + R"(,"type":")" +
                               std::string(type.name) + R"("})";
      const auto rep = Parse(json);
      format::TensorFormat f{.encoding = *encoding,
                             .layout = !info.quantized                 ? Layout::kDense
                                       : *encoding == Encoding::kNvfp4 ? Layout::kGgmlNvfp4Blocks
                                                                       : Layout::kGgmlBlocks,
                             .shape = *format::ShapeOf(ne)};
      if (!rep) {
        // v0 refuses a row that is not whole blocks; so does the format.
        EXPECT_EQ(format::PartsOf(f).error(), FormatError::kBlock) << json;
        continue;
      }
      if (rep->bytes == kSaturated || rep->readable == kSaturated) {
        EXPECT_EQ(format::PartsOf(f).error(), FormatError::kOverflow) << json;
        EXPECT_EQ(ar::FormatOf(*rep).error(), FormatError::kOverflow) << json;
        continue;
      }
      EXPECT_EQ(ar::FormatOf(*rep), f) << json;
      ExpectSized(*rep, rep->bytes, rep->readable, json);
      EXPECT_EQ(ar::V0PartRole(*rep), PartRole::kData);
      ++compared;
    }
  }
  EXPECT_GT(compared, 200U);
}

TEST(ArtifactFormatMapTest, DtypesAreDenseInGgmlOrder) {
  for (const ar::Dtype& dtype : ar::Dtypes()) {
    SCOPED_TRACE(std::string(dtype.name));
    const auto encoding = ar::EncodingOfDtype(dtype.name);
    ASSERT_TRUE(encoding.has_value());
    EXPECT_EQ(format::Info(*encoding).element_bits, 8 * dtype.bytes);
    for (const auto& shape : std::vector<std::vector<std::uint64_t>>{
             {}, {7}, {12288, 80}, {3, 4, 5}, {2, 3, 4, 5, 6, 7}}) {
      const std::string json = R"({"dtype":")" + std::string(dtype.name) +
                               R"(","family":"plain","shape":)" + List(shape) + "}";
      const auto rep = Parse(json);
      ASSERT_TRUE(rep.has_value()) << json;
      ExpectSized(*rep, rep->bytes, rep->readable, json);
      const auto f = ar::FormatOf(*rep);
      if (shape.size() == 2) {
        EXPECT_EQ(f->shape.ne[0], shape[1]);
        EXPECT_EQ(f->shape.ne[1], shape[0]);
      }
      if (shape.size() == 6) {  // [2, 3, 4, 5, 6, 7]: the outer three fold
        EXPECT_EQ(f->shape.rank, 4);
        EXPECT_EQ(f->shape.ne[3], 2U * 3 * 4);
      }
    }
  }
}

std::string Trellis(int k, std::uint64_t in, std::uint64_t out) {
  return R"({"codebook":"mcg","dtype":"I16","family":"exl3","in_features":)" + std::to_string(in) +
         R"(,"k_bits":)" + std::to_string(k) + R"(,"out_features":)" + std::to_string(out) +
         R"(,"role":"trellis","shape":[)" + std::to_string(in / 16) + "," +
         std::to_string(out / 16) + "," + std::to_string(16 * k) + "]}";
}

std::string Side(std::string_view role, std::uint64_t n) {
  return R"({"dtype":"F16","family":"exl3","role":")" + std::string(role) + R"(","shape":[)" +
         std::to_string(n) + "]}";
}

TEST(ArtifactFormatMapTest, Exl3PartsSizeAndAssembleAsTheV0Rule) {
  const auto mcg = Parse(R"({"dtype":"I32","family":"exl3","role":"mcg","shape":[]})");
  ASSERT_TRUE(mcg.has_value());
  EXPECT_EQ(ar::V0PartRole(*mcg), PartRole::kCodebookMarker);
  for (const int k : {4, 5, 6, 8}) {
    for (const auto& [in, out] : std::vector<std::pair<std::uint64_t, std::uint64_t>>{
             {896, 896}, {896, 4864}, {4864, 896}, {896, 128}}) {
      SCOPED_TRACE(std::to_string(k) + " " + std::to_string(in) + "x" + std::to_string(out));
      const auto trellis = Parse(Trellis(k, in, out));
      const auto suh = Parse(Side("suh", in));
      const auto svh = Parse(Side("svh", out));
      ASSERT_TRUE(trellis && suh && svh);
      EXPECT_EQ(ar::V0PartRole(*trellis), PartRole::kCodes);
      ExpectSized(*trellis, trellis->bytes, trellis->readable, "trellis");
      const auto f = ar::FormatOf(*trellis);
      EXPECT_EQ(f->encoding, Encoding::kExl3Trellis);
      EXPECT_EQ(f->params.k_bits, k);
      EXPECT_EQ(f->shape.ne[0], in);
      EXPECT_EQ(f->shape.ne[1], out);
      const std::vector<ar::V0Part> parts = {{PartRole::kCodes, &*trellis},
                                             {PartRole::kInputSigns, &*suh},
                                             {PartRole::kOutputSigns, &*svh},
                                             {PartRole::kCodebookMarker, &*mcg}};
      EXPECT_EQ(ar::Assemble(Encoding::kExl3Trellis, Layout::kExl3Tiles, parts), f);
      // A side vector of the other length, or none, is refused.
      const std::vector<ar::V0Part> swapped = {{PartRole::kCodes, &*trellis},
                                               {PartRole::kInputSigns, &*svh},
                                               {PartRole::kOutputSigns, &*suh},
                                               {PartRole::kCodebookMarker, &*mcg}};
      if (in != out) {
        EXPECT_EQ(ar::Assemble(Encoding::kExl3Trellis, Layout::kExl3Tiles, swapped).error(),
                  FormatError::kParts);
      }
      EXPECT_EQ(ar::Assemble(Encoding::kExl3Trellis, Layout::kExl3Tiles, std::span(parts).first(3))
                    .error(),
                FormatError::kParts);
    }
  }
}

ar::Representation Plain(std::string_view dtype, const std::vector<std::uint64_t>& shape) {
  const auto rep = Parse(R"({"dtype":")" + std::string(dtype) + R"(","family":"plain","shape":)" +
                         List(shape) + "}");
  EXPECT_TRUE(rep.has_value());
  return rep.value_or(ar::Representation{});
}

ar::Representation Ggml(std::string_view type, const std::vector<std::uint64_t>& ne) {
  const auto rep =
      Parse(R"({"family":"ggml","ne":)" + List(ne) + R"(,"type":")" + std::string(type) + R"("})");
  EXPECT_TRUE(rep.has_value());
  return rep.value_or(ar::Representation{});
}

TEST(ArtifactFormatMapTest, PairingsAssembleOrAreRefused) {
  // An MXFP8 linear: F8_E4M3 [N, K] codes and U8 [N, K / 32] scales.
  const ar::Representation codes = Plain("F8_E4M3", {640, 2560});
  const ar::Representation scales = Plain("U8", {640, 80});
  const std::vector<ar::V0Part> mx = {{PartRole::kCodes, &codes},
                                      {PartRole::kBlockScales, &scales}};
  const auto linear = ar::Assemble(Encoding::kMxfp8, Layout::kSplit, mx);
  ASSERT_TRUE(linear.has_value());
  EXPECT_EQ(linear->shape.ne[0], 2560U);
  EXPECT_EQ(linear->shape.ne[1], 640U);
  const ar::Representation wrong_scales = Plain("U8", {640, 40});
  EXPECT_EQ(ar::Assemble(Encoding::kMxfp8, Layout::kSplit,
                         std::vector<ar::V0Part>{{PartRole::kCodes, &codes},
                                                 {PartRole::kBlockScales, &wrong_scales}})
                .error(),
            FormatError::kParts);
  EXPECT_EQ(
      ar::Assemble(Encoding::kMxfp8, Layout::kSplit,
                   std::vector<ar::V0Part>{{PartRole::kCodes, &codes}, {PartRole::kCodes, &codes}})
          .error(),
      FormatError::kParts);
  // CUTLASS experts: GGML I8 code rows and scale atoms, gate and up stacked.
  const ar::Representation gu_codes = Ggml("I8", {1280, 1280});
  const ar::Representation gu_scales = Ggml("I8", {512, 400});
  const auto gate_up = ar::Assemble(
      Encoding::kNvfp4, Layout::kScaleAtoms,
      std::vector<ar::V0Part>{{PartRole::kCodes, &gu_codes}, {PartRole::kBlockScales, &gu_scales}},
      2);
  ASSERT_TRUE(gate_up.has_value());
  EXPECT_EQ(gate_up->shape.ne[0], 2560U);
  EXPECT_EQ(gate_up->shape.ne[1], 1280U);
  EXPECT_EQ(gate_up->stacked, 2);
  // The n-gram table: 90-byte rows of 160 values, and its F32 scale.
  const ar::Representation table = Plain("U8", {37, 90});
  const ar::Representation table_scale = Plain("F32", {1});
  const auto rows = ar::Assemble(
      Encoding::kNvfp4, Layout::kRowInterleaved,
      std::vector<ar::V0Part>{{PartRole::kData, &table}, {PartRole::kTensorScale, &table_scale}});
  ASSERT_TRUE(rows.has_value());
  EXPECT_EQ(rows->shape.ne[0], 160U);
  EXPECT_TRUE(rows->params.tensor_scale);
  const ar::Representation i32_scale = Plain("I32", {1});
  EXPECT_EQ(ar::Assemble(Encoding::kNvfp4, Layout::kRowInterleaved,
                         std::vector<ar::V0Part>{{PartRole::kData, &table},
                                                 {PartRole::kTensorScale, &i32_scale}})
                .error(),
            FormatError::kParts);
  // GGML's NVFP4 blocks with per-expert scales (one here: a slice's).
  const ar::Representation nv = Ggml("NVFP4", {2560, 640});
  const auto blocks = ar::Assemble(
      Encoding::kNvfp4, Layout::kGgmlNvfp4Blocks,
      std::vector<ar::V0Part>{{PartRole::kData, &nv}, {PartRole::kTensorScale, &table_scale}});
  ASSERT_TRUE(blocks.has_value());
  EXPECT_EQ(blocks->layout, Layout::kGgmlNvfp4Blocks);
  // The wrong encoding or layout for a resource.
  EXPECT_EQ(ar::Assemble(Encoding::kQ8_0, Layout::kGgmlBlocks,
                         std::vector<ar::V0Part>{{PartRole::kData, &nv}})
                .error(),
            FormatError::kLayout);
  EXPECT_EQ(ar::Assemble(
                Encoding::kMxfp8, Layout::kSplit,
                std::vector<ar::V0Part>{{PartRole::kCodes, &nv}, {PartRole::kBlockScales, &scales}})
                .error(),
            FormatError::kParts);
}

// Every resource and expert array of an opened artifact maps to a format
// of exactly its recorded size and readable range.
std::size_t ExpectArtifactSized(const ar::Artifact& a, const std::string& what) {
  std::size_t checked = 0;
  for (const ar::Resource& r : a.resources()) {
    ExpectSized(r.repr, r.bytes.value(), r.readable.value(), what + " " + r.name);
    ++checked;
  }
  for (const ar::ExpertArray& x : a.expert_arrays()) {
    ExpectSized(x.repr, x.slice_bytes.value(), x.readable.value(), what + " " + x.name);
    ++checked;
  }
  return checked;
}

fs::path Corpus() {
  const std::string configured = Env("LLMP_TEST_ARTIFACT_CORPUS");
  return configured.empty() ? fs::path("artifact-corpus") : fs::path(configured);
}

fs::path OnlyChild(const fs::path& dir) {
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(dir, error)) {
    return entry.path();
  }
  return {};
}

TEST(ArtifactFormatMapTest, TheCorpusArtifactsMapToTheirSizes) {
  for (const char* name : {"tiny", "exl3", "moe"}) {
    const fs::path root = OnlyChild(Corpus() / "golden" / name);
    ASSERT_FALSE(root.empty()) << "no corpus at " << Corpus();
    const auto opened = ar::Artifact::Open(root);
    ASSERT_TRUE(opened.has_value()) << name;
    EXPECT_GT(ExpectArtifactSized(*opened, name), 0U);
  }
}

// The installed v0 fixtures and M3 artifacts, where a host has them.
TEST(ArtifactFormatMapTest, RealArtifactsMapToTheirSizes) {
  const std::string home = Env("HOME");
  if (home.empty()) {
    GTEST_SKIP() << "no HOME";
  }
  std::size_t opened = 0;
  for (const char* store :
       {".local/share/llmp/artifact-layout-20260922/installed", ".local/share/llmp/m3-artifacts"}) {
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(fs::path(home) / store, error)) {
      const auto a = ar::Artifact::Open(entry.path());
      if (!a) {
        continue;  // a composition, or not an artifact (artifact_test's concern)
      }
      ++opened;
      ExpectArtifactSized(*a, entry.path().filename().string());
    }
  }
  if (opened == 0) {
    GTEST_SKIP() << "no installed artifacts (they are on the Sparks)";
  }
}

// The Qwen3.8 ModelOpt import's multi-part tensors assemble: every MXFP8
// linear's codes and scales, the CUTLASS expert arrays, the n-gram table.
TEST(ArtifactFormatMapTest, Qwen38ImportAssembles) {
  const fs::path root = fs::path(Env("HOME")) / ".local/share/llmp/m3-artifacts" /
                        "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";
  std::error_code error;
  if (!fs::is_directory(root, error)) {
    GTEST_SKIP() << "no Qwen3.8 import at " << root;
  }
  const auto opened = ar::Artifact::Open(root);
  ASSERT_TRUE(opened.has_value());
  const ar::Artifact& a = *opened;
  std::size_t linears = 0;
  for (const ar::Resource& r : a.resources()) {
    if (r.repr.type != "F8_E4M3" || !r.name.ends_with(".weight")) {
      continue;
    }
    const auto scales = a.FindResource(r.name + "_scale");
    ASSERT_TRUE(scales.has_value()) << r.name;
    const std::vector<ar::V0Part> parts = {{PartRole::kCodes, &r.repr},
                                           {PartRole::kBlockScales, &a.resources()[*scales].repr}};
    const auto f = ar::Assemble(Encoding::kMxfp8, Layout::kSplit, parts);
    ASSERT_TRUE(f.has_value()) << r.name << ": " << format::FormatErrorName(f.error());
    ++linears;
  }
  EXPECT_GT(linears, 100U);
  std::size_t experts = 0;
  for (const ar::ExpertArray& x : a.expert_arrays()) {
    if (!x.name.ends_with("_exps.codes")) {
      continue;
    }
    const std::string base = x.name.substr(0, x.name.size() - std::string_view("codes").size());
    const auto scales = a.FindExpertArray(base + "scales");
    ASSERT_TRUE(scales.has_value()) << x.name;
    const std::vector<ar::V0Part> parts = {
        {PartRole::kCodes, &x.repr}, {PartRole::kBlockScales, &a.expert_arrays()[*scales].repr}};
    const bool gate_up = x.name.find("gate_up") != std::string::npos;
    const auto f = ar::Assemble(Encoding::kNvfp4, Layout::kScaleAtoms, parts, gate_up ? 2 : 1);
    ASSERT_TRUE(f.has_value()) << x.name << ": " << format::FormatErrorName(f.error());
    ++experts;
  }
  EXPECT_GT(experts, 0U);
  const auto table = a.FindResource("per_layer_token_embd.weight");
  const auto table_scale = a.FindResource("per_layer_token_embd.weight_scale_2");
  ASSERT_TRUE(table && table_scale);
  const std::vector<ar::V0Part> parts = {
      {PartRole::kData, &a.resources()[*table].repr},
      {PartRole::kTensorScale, &a.resources()[*table_scale].repr}};
  const auto f = ar::Assemble(Encoding::kNvfp4, Layout::kRowInterleaved, parts);
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(f->shape.ne[0], 160U);
  EXPECT_EQ(f->shape.ne[1], 320001536U);
}

}  // namespace
