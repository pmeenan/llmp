// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The importer's MLX, EXL3 and diffusers readers (import/readers/mlx.h,
// exl3.h, diffusers.h) on checkpoints built here: each quantized weight's
// parts paired into the format its scheme gives (MLX's affine at any
// width and its fp modes, per-module overrides, expert arrays; EXL3's
// trellis with each codebook), what stays dense, a refusal for each rule,
// every truncation and random corruptions; a pipeline read as its
// components and kept files; and which reader claims which directory.

#include <gtest/gtest.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/reader.h"
#include "import/readers/diffusers.h"
#include "import/readers/exl3.h"
#include "import/readers/mlx.h"
#include "import/readers/safetensors.h"
#include "import_fixtures.h"

namespace llmp::import {
namespace {

using format::Encoding;
using format::Layout;
using format::PartRole;
using namespace fixtures;  // NOLINT(google-build-using-namespace): the tests' builders
using test_support::Failed;

const Tensor& Named(const Checkpoint& c, std::string_view name) {
  const Tensor* t = c.FindTensor(name);
  EXPECT_NE(t, nullptr) << name;
  static const Tensor kNone{};
  return t != nullptr ? *t : kNone;
}

// ---------------------------------------------------------------- MLX

constexpr std::string_view kMlxConfig =
    R"({"model_type":"qwen3","quantization":{"group_size":32,"bits":4},
        "quantization_config":{"group_size":32,"bits":4}})";

// An affine module of K values a row: U32 codes, BF16 scales and biases.
std::vector<Entry> AffineModule(const std::string& m, std::vector<std::uint64_t> outer,
                                std::uint64_t k, std::uint32_t bits, std::uint32_t group,
                                std::string_view scale = "BF16") {
  std::vector<std::uint64_t> codes = outer;
  codes.push_back(k * bits / 32);
  std::vector<std::uint64_t> scales = outer;
  scales.push_back(k / group);
  return {E(m + ".weight", "U32", codes), E(m + ".scales", std::string(scale), scales),
          E(m + ".biases", std::string(scale), scales)};
}

std::vector<Entry> Join(std::initializer_list<std::vector<Entry>> parts) {
  std::vector<Entry> out;
  for (const auto& p : parts) {
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}

TEST(ImportMlxTest, PairsAffineWeightsAndExpertArrays) {
  const auto shard =
      Shard(Join({AffineModule("m.q_proj", {4}, 64, 4, 32),
                  AffineModule("m.switch_mlp.gate_proj", {3, 4}, 64, 4, 32),
                  {E("m.norm.weight", "BF16", {64}), E("m.gate.weight", "BF16", {8, 64}),
                   E("m.switch_mlp.table", "BF16", {2, 3, 4}), E("lone.scales", "BF16", {2}),
                   E("m.conv1d.bias", "F32", {3})}}),
            R"({"format":"mlx"})");
  const auto c = ParseMlx(shard, kMlxConfig);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "mlx");
  EXPECT_EQ(c->architecture, "qwen3");
  EXPECT_EQ(c->quantization, "mlx affine");
  ExpectConsistent(*c);
  format::TensorFormat q{
      .encoding = Encoding::kAffine,
      .params = {.bits = 4, .group = 32, .scale = format::AffineScale::kBF16, .zero_point = false},
      .layout = Layout::kSplit,
      .shape = {.ne = {64, 4, 1, 1}, .rank = 2}};
  for (const auto& [name, part] : {std::pair{"m.q_proj.weight", PartRole::kCodes},
                                   std::pair{"m.q_proj.scales", PartRole::kBlockScales},
                                   std::pair{"m.q_proj.biases", PartRole::kBiases}}) {
    EXPECT_EQ(Named(*c, name).format, q) << name;
    EXPECT_EQ(Named(*c, name).part, part) << name;
    EXPECT_FALSE(Named(*c, name).experts) << name;
  }
  EXPECT_EQ(Named(*c, "m.q_proj.weight").dtype, "U32");
  q.shape = {.ne = {64, 4, 3, 1}, .rank = 3};
  EXPECT_EQ(Named(*c, "m.switch_mlp.gate_proj.weight").format, q);
  EXPECT_TRUE(Named(*c, "m.switch_mlp.gate_proj.biases").experts);
  // Dense: norms, an unquantized linear, stray scales; a dense expert array.
  EXPECT_EQ(Named(*c, "m.gate.weight").format.encoding, Encoding::kBF16);
  EXPECT_EQ(Named(*c, "lone.scales").part, PartRole::kData);
  EXPECT_TRUE(Named(*c, "m.switch_mlp.table").experts);
  // The config's quantization is not metadata; the rest is.
  EXPECT_EQ(c->FindValue("quantization.bits"), nullptr);
  EXPECT_EQ(c->FindValue("quantization_config.bits"), nullptr);
}

TEST(ImportMlxTest, OverridesAndFpModes) {
  // Module overrides: 8 bits at group 64 with F16 scales; false where no
  // scales are; true for the default.
  const std::string config =
      R"({"model_type":"t","quantization":{"group_size":32,"bits":4,
          "a.o_proj":{"group_size":64,"bits":8},"a.gate":false,"a.k_proj":true}})";
  const auto shard = Shard(Join({AffineModule("a.o_proj", {2}, 128, 8, 64, "F16"),
                                 AffineModule("a.k_proj", {2}, 64, 4, 32),
                                 {E("a.gate.weight", "BF16", {2, 8})}}));
  const auto c = ParseMlx(shard, config);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  ExpectConsistent(*c);
  const format::TensorFormat& o = Named(*c, "a.o_proj.weight").format;
  EXPECT_EQ(o.params.bits, 8);
  EXPECT_EQ(o.params.group, 64U);
  EXPECT_EQ(o.params.scale, format::AffineScale::kF16);
  EXPECT_EQ(o.shape.ne[0], 128U);
  EXPECT_EQ(Named(*c, "a.k_proj.weight").format.params.bits, 4);
  // Every width the encoding takes, 3, 5 and 6 bits included.
  for (const std::uint32_t bits : {2U, 3U, 5U, 6U, 8U}) {
    const auto c3 = ParseMlx(
        Shard(AffineModule("w", {2}, 128, bits, 32)),
        std::format(R"({{"model_type":"t","quantization":{{"group_size":32,"bits":{}}}}})", bits));
    ASSERT_TRUE(c3.has_value()) << bits << ": " << c3.error().ToString();
    EXPECT_EQ(std::uint32_t{Named(*c3, "w.weight").format.params.bits}, bits);
  }
  // MLX's fp modes: codes and E8M0 or E4M3 scales, no biases (one beside
  // them stays dense).
  const auto fp = [](std::string_view mode, std::uint64_t k, std::uint32_t bits,
                     std::uint32_t group, std::string_view scale) {
    return ParseMlx(Shard({E("w.weight", "U32", {2, k * bits / 32}),
                           E("w.scales", std::string(scale), {2, k / group}),
                           E("w.biases", "BF16", {2, k / group})}),
                    std::format(R"({{"model_type":"t","quantization":{{"mode":"{}"}}}})", mode));
  };
  const auto mxfp4 = fp("mxfp4", 64, 4, 32, "U8");
  ASSERT_TRUE(mxfp4.has_value()) << mxfp4.error().ToString();
  EXPECT_EQ(mxfp4->quantization, "mlx mxfp4");
  EXPECT_EQ(Named(*mxfp4, "w.weight").format,
            (format::TensorFormat{.encoding = Encoding::kMxfp4,
                                  .layout = Layout::kSplit,
                                  .shape = {.ne = {64, 2, 1, 1}, .rank = 2}}));
  EXPECT_EQ(Named(*mxfp4, "w.biases").part, PartRole::kData);
  const auto nvfp4 = fp("nvfp4", 64, 4, 16, "F8_E4M3");
  ASSERT_TRUE(nvfp4.has_value()) << nvfp4.error().ToString();
  EXPECT_EQ(Named(*nvfp4, "w.scales").format.encoding, Encoding::kNvfp4);
  EXPECT_FALSE(Named(*nvfp4, "w.scales").format.params.tensor_scale);
  const auto mxfp8 = fp("mxfp8", 64, 8, 32, "F8_E8M0");
  ASSERT_TRUE(mxfp8.has_value()) << mxfp8.error().ToString();
  EXPECT_EQ(Named(*mxfp8, "w.weight").format.encoding, Encoding::kMxfp8);
  // A mode's own scale format only, and its own bits and group.
  EXPECT_EQ(RuleOf(fp("nvfp4", 64, 4, 16, "F8_E8M0")), Rule::kQuantization);
  EXPECT_EQ(RuleOf(fp("mxfp4", 64, 4, 32, "BF16")), Rule::kQuantization);
  EXPECT_EQ(RuleOf(ParseMlx(Shard(AffineModule("w", {2}, 64, 4, 32)),
                            R"({"model_type":"t","quantization":{"mode":"mxfp4","bits":8}})")),
            Rule::kConfig);
  EXPECT_EQ(RuleOf(ParseMlx(Shard(AffineModule("w", {2}, 64, 4, 32)),
                            R"({"model_type":"t","quantization":{"mode":"int4"}})")),
            Rule::kUnsupported);
}

TEST(ImportMlxTest, RefusesWhatItCannotDescribe) {
  const auto rule = [](const std::vector<Entry>& entries, std::string_view config = kMlxConfig) {
    return RuleOf(ParseMlx(Shard(entries), config));
  };
  const auto good = AffineModule("w", {2}, 64, 4, 32);
  EXPECT_EQ(rule(good), std::nullopt);
  // A module the config leaves unquantized, or with no scheme at all.
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"group_size":32,"bits":4,"w":false}})"),
            Rule::kQuantization);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{}})"), Rule::kConfig);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"bits":4}})"), Rule::kConfig);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"w":{"bits":4}}})"), Rule::kConfig);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"group_size":32,"bits":4,"w":1}})"),
            Rule::kConfig);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"group_size":32,"bits":9}})"),
            Rule::kConfig);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"group_size":0,"bits":4}})"),
            Rule::kConfig);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"group_size":32,"bits":"4"}})"),
            Rule::kConfig);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":[]})"), Rule::kConfig);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"group_size":32,"bits":4,"":true}})"),
            Rule::kConfig);
  // The scheme's bits or group against the parts' sizes.
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"group_size":32,"bits":8}})"),
            Rule::kQuantization);
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization":{"group_size":12,"bits":4}})"),
            Rule::kQuantization);
  // Parts of other dtypes, shapes or ranks; biases missing or unlike the scales.
  EXPECT_EQ(rule({E("w.weight", "F16", {2, 16}), good[1], good[2]}), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], E("w.scales", "U8", {2, 2}), E("w.biases", "U8", {2, 2})}),
            Rule::kQuantization);
  EXPECT_EQ(rule({good[0], good[1]}), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], good[1], E("w.biases", "F16", {2, 2})}), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], good[1], E("w.biases", "BF16", {1, 4})}), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], E("w.scales", "BF16", {2, 3}), E("w.biases", "BF16", {2, 3})}),
            Rule::kQuantization);
  EXPECT_EQ(rule({good[0], E("w.scales", "BF16", {3, 2}), E("w.biases", "BF16", {3, 2})}),
            Rule::kQuantization);
  EXPECT_EQ(
      rule({E("w.weight", "U32", {16}), E("w.scales", "BF16", {2}), E("w.biases", "BF16", {2})}),
      Rule::kQuantization);
  // Codes of any unsigned width are the same bytes.
  EXPECT_EQ(rule({E("w.weight", "U8", {2, 32}), good[1], good[2]}), std::nullopt);
}

TEST(ImportMlxTest, EveryPrefixAndCorruptionIsRefusedOrConsistent) {
  const auto shard = Shard(Join(
      {AffineModule("m.q", {2}, 64, 4, 32), AffineModule("m.switch_mlp.up", {2, 2}, 64, 4, 32)}));
  ASSERT_TRUE(ParseMlx(shard, kMlxConfig).has_value());
  for (std::size_t n = 0; n < shard.size(); ++n) {
    EXPECT_EQ(RuleOf(ParseMlx(View(shard).first(n), kMlxConfig)), Rule::kContainer) << n;
  }
  Corrupt(shard, [](const std::vector<std::byte>& bytes) { return ParseMlx(bytes, kMlxConfig); });
}

// An unquantized MLX checkpoint is known by its shards' format: the
// safetensors reader hands it to MLX's (container "mlx", expert arrays).
TEST(ImportMlxTest, KnowsUnquantizedCheckpointsByTheirShards) {
  const std::vector<Entry> entries = {E("m.conv1d.weight", "BF16", {8, 4, 1}),
                                      E("m.switch_mlp.up_proj.weight", "BF16", {2, 4, 8})};
  const auto c = ParseSafetensors(Shard(entries, R"({"format":"mlx"})"), R"({"model_type":"t"})");
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "mlx");
  EXPECT_TRUE(c->quantization.empty());
  // Dims as stored: MLX's channels-last [channels, kernel, 1].
  EXPECT_EQ(Named(*c, "m.conv1d.weight").dims, (std::vector<std::uint64_t>{8, 4, 1}));
  EXPECT_TRUE(Named(*c, "m.switch_mlp.up_proj.weight").experts);
  // Another writer's format, or none, is safetensors'.
  for (const std::string_view metadata : {R"({"format":"pt"})", R"({"other":"mlx"})", ""}) {
    const auto pt = ParseSafetensors(Shard(entries, metadata), R"({"model_type":"t"})");
    ASSERT_TRUE(pt.has_value()) << pt.error().ToString();
    EXPECT_EQ(pt->container, "safetensors") << metadata;
  }
}

// ---------------------------------------------------------------- EXL3

constexpr std::string_view kExl3Config =
    R"({"model_type":"qwen2","quantization_config":{"quant_method":"exl3","bits":4.0,
        "codebook":"mcg"}})";

// An EXL3 linear of K inputs and N outputs at k bits a weight.
std::vector<Entry> Linear(const std::string& m, std::uint64_t k, std::uint64_t n,
                          std::uint64_t bits, std::string_view marker = "mcg") {
  std::vector<Entry> out = {E(m + ".trellis", "I16", {k / 16, n / 16, 16 * bits}),
                            E(m + ".suh", "F16", {k}), E(m + ".svh", "F16", {n})};
  if (!marker.empty()) {
    out.push_back(E(m + "." + std::string(marker), "I32", {}));
  }
  return out;
}

TEST(ImportExl3Test, PairsTrellisesWithEachCodebook) {
  const auto shard =
      Shard(Join({Linear("l.q_proj", 32, 48, 4),
                  Linear("l.k_proj", 32, 16, 5, "mul1"),
                  Linear("lm_head", 16, 64, 8, ""),
                  {E("l.q_proj.bias", "F16", {48}), E("embed.weight", "BF16", {10, 32}),
                   E("orphan.suh", "F16", {8})}}));
  const auto c = ParseExl3(shard, kExl3Config);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "exl3");
  EXPECT_EQ(c->quantization, "exl3");
  ExpectConsistent(*c);
  const format::TensorFormat q{.encoding = Encoding::kExl3Trellis,
                               .params = {.k_bits = 4, .codebook = format::Exl3Codebook::kMcg},
                               .layout = Layout::kExl3Tiles,
                               .shape = {.ne = {32, 48, 1, 1}, .rank = 2}};
  for (const auto& [name, part] : {std::pair{"l.q_proj.trellis", PartRole::kCodes},
                                   std::pair{"l.q_proj.suh", PartRole::kInputSigns},
                                   std::pair{"l.q_proj.svh", PartRole::kOutputSigns},
                                   std::pair{"l.q_proj.mcg", PartRole::kCodebookMarker}}) {
    EXPECT_EQ(Named(*c, name).format, q) << name;
    EXPECT_EQ(Named(*c, name).part, part) << name;
  }
  EXPECT_EQ(Named(*c, "l.k_proj.trellis").format.params.codebook, format::Exl3Codebook::kMul1);
  EXPECT_EQ(Named(*c, "l.k_proj.mul1").part, PartRole::kCodebookMarker);
  const format::TensorFormat& head = Named(*c, "lm_head.trellis").format;
  EXPECT_EQ(head.params.codebook, format::Exl3Codebook::k3Inst);
  EXPECT_EQ(head.params.k_bits, 8);
  EXPECT_EQ(Named(*c, "l.q_proj.bias").format.encoding, Encoding::kF16);
  EXPECT_EQ(Named(*c, "orphan.suh").part, PartRole::kData);
  EXPECT_EQ(c->FindValue("quantization_config.bits"), nullptr);
}

TEST(ImportExl3Test, RefusesWhatItCannotDescribe) {
  const auto rule = [](const std::vector<Entry>& entries, std::string_view config = kExl3Config) {
    return RuleOf(ParseExl3(Shard(entries), config));
  };
  const auto good = Linear("w", 32, 16, 4);
  EXPECT_EQ(rule(good), std::nullopt);
  // Rates with no format: above 8 bits, half-integer.
  EXPECT_EQ(rule(Linear("w", 32, 16, 9)), Rule::kUnsupported);
  EXPECT_EQ(rule({E("w.trellis", "I16", {2, 1, 40}), good[1], good[2]}), Rule::kUnsupported);
  // A trellis of another dtype or rank; its signs missing, packed, or not
  // F16 of its sides; both markers; a marker of other than 4 bytes.
  EXPECT_EQ(rule({E("w.trellis", "U16", {2, 1, 64}), good[1], good[2]}), Rule::kQuantization);
  EXPECT_EQ(rule({E("w.trellis", "I16", {2, 64}), good[1], good[2]}), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], good[1]}), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], E("w.su", "I16", {2}), E("w.sv", "I16", {1})}), Rule::kUnsupported);
  EXPECT_EQ(rule({good[0], E("w.suh", "F32", {32}), good[2]}), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], E("w.suh", "F16", {48}), good[2]}), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], good[1], E("w.svh", "F16", {32})}), Rule::kQuantization);
  EXPECT_EQ(rule(Join({good, {E("w.mul1", "I32", {})}})), Rule::kQuantization);
  EXPECT_EQ(rule({good[0], good[1], good[2], E("w.mcg", "I64", {})}), Rule::kQuantization);
  // Unpacked signs beside packed ones: upstream reads the unpacked.
  EXPECT_EQ(rule(Join({good, {E("w.su", "I16", {2})}})), std::nullopt);
  // Sign vectors of any shape holding their side's values.
  EXPECT_EQ(rule({good[0], E("w.suh", "F16", {2, 16}), good[2], good[3]}), std::nullopt);
  // Not an EXL3 config.
  EXPECT_EQ(rule(good, R"({"model_type":"t"})"), Rule::kConfig);
  // EXL3's and MLX's quantization both: ambiguous.
  EXPECT_EQ(rule(good, R"({"model_type":"t","quantization_config":{"quant_method":"exl3"},
                          "quantization":{"group_size":32,"bits":4}})"),
            Rule::kConfig);
  // The safetensors reader leaves EXL3 to its own reader.
  EXPECT_EQ(RuleOf(ParseSafetensors(Shard(good), kExl3Config)), Rule::kUnsupported);
}

TEST(ImportExl3Test, EveryPrefixAndCorruptionIsRefusedOrConsistent) {
  const auto shard = Shard(Join({Linear("a", 32, 16, 4), Linear("b", 16, 32, 2, "")}));
  ASSERT_TRUE(ParseExl3(shard, kExl3Config).has_value());
  for (std::size_t n = 0; n < shard.size(); ++n) {
    EXPECT_EQ(RuleOf(ParseExl3(View(shard).first(n), kExl3Config)), Rule::kContainer) << n;
  }
  Corrupt(shard, [](const std::vector<std::byte>& bytes) { return ParseExl3(bytes, kExl3Config); });
}

// ---------------------------------------------------------------- diffusers

TEST(ImportDiffusersTest, ParsesModelIndex) {
  const auto index = ParseModelIndex(R"({
    "_class_name": "QwenImagePipeline", "_diffusers_version": "0.37.0",
    "vae": ["diffusers", "AutoencoderKL"], "text_encoder": ["transformers", "Qwen3VL"],
    "safety_checker": [null, null], "requires_safety_checker": false,
    "pair": [1, 2], "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]})");
  ASSERT_TRUE(index.has_value()) << index.error().ToString();
  EXPECT_EQ(index->architecture, "QwenImagePipeline");
  EXPECT_EQ(index->roles, (std::vector<std::string>{"scheduler", "text_encoder", "vae"}));
  const auto vae = std::ranges::find(index->metadata, "vae", &MetaEntry::key);
  ASSERT_NE(vae, index->metadata.end());
  EXPECT_EQ(std::get<std::vector<std::string>>(vae->value),
            (std::vector<std::string>{"diffusers", "AutoencoderKL"}));
  const auto rule = [](std::string_view text) {
    return Failed(ParseModelIndex(text), &Error::rule);
  };
  EXPECT_EQ(rule("{"), Rule::kConfig);
  EXPECT_EQ(rule("[]"), Rule::kConfig);
  EXPECT_EQ(rule(R"({"vae":["d","V"]})"), Rule::kConfig);
  EXPECT_EQ(rule(R"({"_class_name":""})"), Rule::kConfig);
  // Component names that are not roles (or paths).
  for (const std::string_view key : {"../vae", "Vae", "v/ae", "9vae", "vae.x"}) {
    EXPECT_EQ(rule(std::format(R"({{"_class_name":"P","{}":["d","V"]}})", key)), Rule::kConfig)
        << key;
  }
  EXPECT_EQ(rule(std::format(R"({{"_class_name":"P","{}":["d","V"]}})", std::string(65, 'a'))),
            Rule::kConfig);
  std::string many = R"({"_class_name":"P")";
  for (std::size_t i = 0; i <= kMaxPipelineEntries; ++i) {
    many += std::format(R"(,"c{}":["d","V"])", i);
  }
  EXPECT_EQ(rule(many + "}"), Rule::kBounds);
  EXPECT_EQ(rule(std::string(kMaxModelIndexBytes + 1, ' ')), Rule::kBounds);
}

using ImportDiffusersFilesTest = ScratchTest;

TEST_F(ImportDiffusersFilesTest, ReadsComponentsAndKeptFiles) {
  const std::filesystem::path p = dir_ / "pipeline";
  Write(p / "model_index.json", R"({"_class_name":"P","_diffusers_version":"0.37.0",
      "text_encoder":["transformers","T"],"transformer":["diffusers","D"],"vae":["diffusers","V"],
      "scheduler":["diffusers","S"],"tokenizer":["transformers","K"],"absent":[null,null]})");
  Write(p / "README.md", std::string_view("not kept"));
  // A transformers component, sharded.
  Write(p / "text_encoder" / "config.json", R"({"model_type":"qwen3_vl"})");
  Write(p / "text_encoder" / "model-1.safetensors", Shard({E("a", "BF16", {4})}));
  Write(p / "text_encoder" / "model-2.safetensors", Shard({E("b", "BF16", {2})}));
  Write(p / "text_encoder" / "model.safetensors.index.json",
        Index({{"a", "model-1.safetensors"}, {"b", "model-2.safetensors"}}));
  // Diffusers components, sharded and not.
  Write(p / "transformer" / "config.json", R"({"_class_name":"D","num_layers":2})");
  Write(p / "transformer" / "diffusion_pytorch_model-1.safetensors", Shard({E("x", "BF16", {4})}));
  Write(p / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        Index({{"x", "diffusion_pytorch_model-1.safetensors"}}));
  Write(p / "vae" / "config.json", R"({"_class_name":"V"})");
  Write(p / "vae" / "diffusion_pytorch_model.safetensors", Shard({E("y", "F32", {3})}));
  // Files kept as they are, a subdirectory left out.
  Write(p / "scheduler" / "scheduler_config.json", std::string_view("{}"));
  Write(p / "tokenizer" / "vocab.json", std::string_view("{}"));
  Write(p / "tokenizer" / "merges.txt", std::string_view("a b"));
  Write(p / "tokenizer" / "sub" / "x.json", std::string_view("{}"));

  const auto c = ReadCheckpoint(p, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "diffusers");
  EXPECT_EQ(c->architecture, "P");
  EXPECT_TRUE(c->tensors.empty());
  ASSERT_NE(c->FindValue("_diffusers_version"), nullptr);
  ASSERT_EQ(c->components.size(), 3U);
  EXPECT_EQ(c->components[0].role, "text_encoder");
  EXPECT_EQ(c->components[0].architecture, "qwen3_vl");
  EXPECT_EQ(c->components[0].container, "safetensors");
  EXPECT_EQ(c->components[0].tensors.size(), 2U);
  EXPECT_EQ(c->components[1].role, "transformer");
  EXPECT_EQ(c->components[1].architecture, "D");
  ASSERT_NE(c->components[1].FindValue("num_layers"), nullptr);
  EXPECT_EQ(c->components[1].files[0].name, "diffusion_pytorch_model-1.safetensors");
  EXPECT_EQ(c->components[2].role, "vae");
  EXPECT_EQ(c->components[2].files[0].name, "diffusion_pytorch_model.safetensors");
  std::vector<std::string> files;
  for (const SourceFile& f : c->files) {
    files.push_back(f.name);
  }
  EXPECT_EQ(files, (std::vector<std::string>{"model_index.json", "scheduler/scheduler_config.json",
                                             "tokenizer/merges.txt", "tokenizer/vocab.json"}));
  EXPECT_EQ(c->open.size(), c->files.size());
  EXPECT_EQ(c->files[2].bytes, 3U);
  // A component read alone is an ordinary safetensors checkpoint.
  const auto vae = ReadCheckpoint(p / "vae", ::geteuid());
  ASSERT_TRUE(vae.has_value()) << vae.error().ToString();
  EXPECT_EQ(vae->architecture, "V");
}

TEST_F(ImportDiffusersFilesTest, RefusesPipelinesItCannotDescribe) {
  const std::filesystem::path p = dir_ / "pipeline";
  const auto rule = [&] { return RuleOf(ReadDiffusers(p, ::geteuid())); };
  Write(p / "model_index.json", R"({"_class_name":"P","vae":["d","V"],"scheduler":["d","S"]})");
  Write(p / "vae" / "config.json", R"({"_class_name":"V"})");
  Write(p / "vae" / "diffusion_pytorch_model.safetensors", Shard({E("y", "F32", {3})}));
  Write(p / "scheduler" / "scheduler_config.json", std::string_view("{}"));
  ASSERT_EQ(rule(), std::nullopt);
  // Weights under both names.
  Write(p / "vae" / "model.safetensors", Shard({E("y", "F32", {3})}));
  EXPECT_EQ(rule(), Rule::kShards);
  std::filesystem::remove(p / "vae" / "model.safetensors");
  // A component's directory missing, or others could change it.
  std::filesystem::rename(p / "scheduler", p / "s");
  EXPECT_EQ(rule(), Rule::kFileType);
  std::filesystem::rename(p / "s", p / "scheduler");
  ::chmod((p / "scheduler").c_str(), 0777);
  EXPECT_EQ(rule(), Rule::kUntrusted);
  ::chmod((p / "scheduler").c_str(), 0755);
  ::chmod((p / "scheduler" / "scheduler_config.json").c_str(), 0666);
  EXPECT_EQ(rule(), Rule::kUntrusted);
  ::chmod((p / "scheduler" / "scheduler_config.json").c_str(), 0644);
  // A kept file whose name has a control character.
  Write(p / "scheduler" / "a\x01.json", std::string_view("{}"));
  EXPECT_EQ(rule(), Rule::kFileType);
  std::filesystem::remove(p / "scheduler" / "a\x01.json");
  // A component's own refusal is the pipeline's.
  Write(p / "vae" / "config.json", R"({"nothing":1})");
  EXPECT_EQ(rule(), Rule::kConfig);
  Write(p / "vae" / "config.json", R"({"_class_name":"V"})");
  // No component with weights; no model_index.json.
  Write(p / "model_index.json", R"({"_class_name":"P","scheduler":["d","S"]})");
  EXPECT_EQ(rule(), Rule::kConfig);
  std::filesystem::remove(p / "model_index.json");
  EXPECT_EQ(rule(), Rule::kFileType);
  // A component directory with its own model_index.json is not read as a
  // pipeline.
  Write(p / "model_index.json", R"({"_class_name":"P","vae":["d","V"]})");
  Write(p / "vae" / "model_index.json", R"({"_class_name":"Q","vae":["d","V"]})");
  const auto c = ReadDiffusers(p, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->components.at(0).container, "safetensors");
}

TEST_F(ImportDiffusersFilesTest, BoundsTheDirectoriesItLists) {
  const std::filesystem::path p = dir_ / "pipeline";
  Write(p / "model_index.json", R"({"_class_name":"P","vae":["d","V"],"tokenizer":["t","T"]})");
  Write(p / "vae" / "config.json", R"({"_class_name":"V"})");
  Write(p / "vae" / "diffusion_pytorch_model.safetensors", Shard({E("y", "F32", {3})}));
  // Kept files: at most kMaxPipelineFiles (each stays open).
  for (std::size_t i = 0; i < kMaxPipelineFiles; ++i) {
    Write(p / "tokenizer" / std::format("f{}", i), std::string_view("{}"));
  }
  const auto c = ReadDiffusers(p, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->files.size(), 1 + kMaxPipelineFiles);
  Write(p / "tokenizer" / "one_more", std::string_view("{}"));
  EXPECT_EQ(RuleOf(ReadDiffusers(p, ::geteuid())), Rule::kBounds);
  std::filesystem::remove(p / "tokenizer" / "one_more");
  // Entries listed: at most kMaxPipelineDirectoryEntries, directories too.
  std::error_code error;
  for (std::size_t i = kMaxPipelineFiles; i < kMaxPipelineDirectoryEntries; ++i) {
    std::filesystem::create_directory(p / "tokenizer" / std::format("d{}", i), error);
  }
  EXPECT_TRUE(ReadDiffusers(p, ::geteuid()).has_value());
  std::filesystem::create_directory(p / "tokenizer" / "one_more", error);
  EXPECT_EQ(RuleOf(ReadDiffusers(p, ::geteuid())), Rule::kBounds);
}

TEST_F(ImportDiffusersFilesTest, ComponentsStayInsideThePipelineAndNameThemselves) {
  const std::filesystem::path p = dir_ / "pipeline";
  Write(p / "model_index.json", R"({"_class_name":"P","unet":["d","U"],"vae":["d","V"]})");
  Write(p / "unet" / "config.json", R"({"_class_name":"U"})");
  Write(p / "unet" / "diffusion_pytorch_model.safetensors", Shard({E("u", "F32", {3})}));
  Write(dir_ / "elsewhere" / "config.json", R"({"_class_name":"V"})");
  Write(dir_ / "elsewhere" / "diffusion_pytorch_model.safetensors", Shard({E("y", "F32", {3})}));
  std::error_code error;
  // A component directory linked out of the pipeline: an escape.
  std::filesystem::create_directory_symlink("../elsewhere", p / "vae", error);
  ASSERT_FALSE(error) << error.message();
  auto c = ReadDiffusers(p, ::geteuid());
  ASSERT_FALSE(c.has_value());
  EXPECT_EQ(c.error().rule, Rule::kUntrusted);
  EXPECT_EQ(c.error().component, 1U);  // vae: roles unet, vae
  EXPECT_TRUE(c.error().ToString().contains("(component 1)")) << c.error().ToString();
  // Linked to the pipeline itself: not beneath it either.
  std::filesystem::remove(p / "vae");
  std::filesystem::create_directory_symlink(".", p / "vae", error);
  EXPECT_EQ(RuleOf(ReadDiffusers(p, ::geteuid())), Rule::kUntrusted);
  // Linked inside the pipeline: read.
  std::filesystem::remove(p / "vae");
  std::filesystem::rename(dir_ / "elsewhere", p / "vae_real");
  std::filesystem::create_directory_symlink("vae_real", p / "vae", error);
  c = ReadDiffusers(p, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->components.at(1).architecture, "V");
  // An error inside a component names it, beside its own file and item.
  Write(p / "vae_real" / "diffusion_pytorch_model.safetensors", RawShard("{", 0));
  c = ReadDiffusers(p, ::geteuid());
  EXPECT_EQ(RuleOf(c), Rule::kContainer);
  EXPECT_EQ(Failed(c, &Error::component), 1U);
  EXPECT_EQ(Failed(c, &Error::file), 0U);
}

TEST_F(ImportDiffusersFilesTest, ReadsAVariantWhenItIsTheOnlyWeights) {
  const std::filesystem::path p = dir_ / "pipeline";
  Write(p / "model_index.json", R"({"_class_name":"P","vae":["d","V"],"text_encoder":["t","T"]})");
  Write(p / "vae" / "config.json", R"({"_class_name":"V"})");
  Write(p / "vae" / "diffusion_pytorch_model.fp16.safetensors", Shard({E("y", "F16", {3})}));
  // transformers' sharded variant: the index's name ends .<v>.json, its
  // shards' names are the index's.
  Write(p / "text_encoder" / "config.json", R"({"model_type":"t5"})");
  Write(p / "text_encoder" / "model.fp16-00001-of-00002.safetensors", Shard({E("a", "F16", {2})}));
  Write(p / "text_encoder" / "model.fp16-00002-of-00002.safetensors", Shard({E("b", "F16", {2})}));
  Write(p / "text_encoder" / "model.safetensors.index.fp16.json",
        Index({{"a", "model.fp16-00001-of-00002.safetensors"},
               {"b", "model.fp16-00002-of-00002.safetensors"}}));
  auto c = ReadDiffusers(p, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->components.at(0).role, "text_encoder");
  EXPECT_EQ(c->components.at(0).tensors.size(), 2U);
  EXPECT_EQ(c->components.at(0).files.back().name, "model.safetensors.index.fp16.json");
  EXPECT_EQ(c->components.at(1).files.at(0).name, "diffusion_pytorch_model.fp16.safetensors");
  // Two variants and no default: ambiguous.
  Write(p / "vae" / "diffusion_pytorch_model.bf16.safetensors", Shard({E("y", "BF16", {3})}));
  c = ReadDiffusers(p, ::geteuid());
  EXPECT_EQ(RuleOf(c), Rule::kShards);
  EXPECT_EQ(Failed(c, &Error::component), 1U);
  // With the default beside them, the default (as diffusers loads it).
  Write(p / "vae" / "diffusion_pytorch_model.safetensors", Shard({E("y", "F32", {3})}));
  c = ReadDiffusers(p, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->components.at(1).files.at(0).name, "diffusion_pytorch_model.safetensors");
  EXPECT_EQ(c->components.at(1).tensors.at(0).dtype, "F32");
  // A variant of each base: ambiguous too; nothing but non-variants: none.
  std::filesystem::remove(p / "vae" / "diffusion_pytorch_model.safetensors");
  std::filesystem::remove(p / "vae" / "diffusion_pytorch_model.bf16.safetensors");
  Write(p / "vae" / "model.fp16.safetensors", Shard({E("y", "F16", {3})}));
  EXPECT_EQ(RuleOf(ReadDiffusers(p, ::geteuid())), Rule::kShards);
  std::filesystem::remove(p / "vae" / "model.fp16.safetensors");
  std::filesystem::remove(p / "vae" / "diffusion_pytorch_model.fp16.safetensors");
  Write(p / "vae" / "diffusion_pytorch_model.fp-16.safetensors", Shard({E("y", "F16", {3})}));
  Write(p / "vae" / "diffusion_pytorch_model.bin", std::string_view("x"));
  EXPECT_EQ(RuleOf(ReadDiffusers(p, ::geteuid())), Rule::kFileType);
}

TEST_F(ImportDiffusersFilesTest, BoundsTheBytesItKeeps) {
  const std::filesystem::path p = dir_ / "pipeline";
  Write(p / "model_index.json", R"({"_class_name":"P","vae":["d","V"],"tokenizer":["t","T"]})");
  Write(p / "vae" / "config.json", R"({"_class_name":"V"})");
  Write(p / "vae" / "diffusion_pytorch_model.safetensors", Shard({E("y", "F32", {3})}));
  // Kept files are opened, not read: sparse files of their sizes.
  const auto sized = [&](const std::string& name, std::uint64_t bytes) {
    Write(p / "tokenizer" / name, std::string_view(""));
    std::filesystem::resize_file(p / "tokenizer" / name, bytes);
  };
  sized("a", kMaxPipelineFileBytes);
  sized("b", kMaxPipelineFilesBytes - kMaxPipelineFileBytes);
  ASSERT_TRUE(ReadDiffusers(p, ::geteuid()).has_value());
  sized("c", 1);
  EXPECT_EQ(RuleOf(ReadDiffusers(p, ::geteuid())), Rule::kBounds);
  std::filesystem::remove(p / "tokenizer" / "c");
  sized("a", kMaxPipelineFileBytes + 1);
  sized("b", 0);
  EXPECT_EQ(RuleOf(ReadDiffusers(p, ::geteuid())), Rule::kBounds);
}

// ---------------------------------------------------------------- claims

using ImportClaimsTest = ScratchTest;

TEST_F(ImportClaimsTest, EachDirectoryGoesToItsReader) {
  const auto container = [&](const std::filesystem::path& p) {
    const auto c = ReadCheckpoint(p, ::geteuid());
    EXPECT_TRUE(c.has_value()) << p << ": " << c.error().ToString();
    return c ? c->container : std::string();
  };
  Write(dir_ / "mlx" / "config.json", std::string(kMlxConfig));
  Write(dir_ / "mlx" / "model.safetensors", Shard(AffineModule("w", {2}, 64, 4, 32)));
  EXPECT_EQ(container(dir_ / "mlx"), "mlx");
  EXPECT_EQ(RuleOf(ReadSafetensors(dir_ / "mlx", ::geteuid())), Rule::kUnsupported);
  Write(dir_ / "exl3" / "config.json", std::string(kExl3Config));
  Write(dir_ / "exl3" / "model.safetensors", Shard(Linear("w", 32, 16, 4)));
  EXPECT_EQ(container(dir_ / "exl3"), "exl3");
  Write(dir_ / "plain" / "config.json", std::string_view(R"({"model_type":"t"})"));
  Write(dir_ / "plain" / "model.safetensors", Shard({E("a", "F32", {2})}));
  EXPECT_EQ(container(dir_ / "plain"), "safetensors");
  // An MLX or EXL3 config.json that cannot be read is the safetensors
  // reader's to refuse (no reader claims on what it could not check).
  ::chmod((dir_ / "mlx" / "config.json").c_str(), 0666);
  EXPECT_EQ(RuleOf(ReadCheckpoint(dir_ / "mlx", ::geteuid())), Rule::kUntrusted);
  // A reader called directly on another's checkpoint.
  EXPECT_EQ(RuleOf(ReadExl3(dir_ / "plain", ::geteuid())), Rule::kConfig);
  const auto unquantized = ReadMlx(dir_ / "plain", ::geteuid());
  ASSERT_TRUE(unquantized.has_value()) << unquantized.error().ToString();
  EXPECT_EQ(unquantized->container, "mlx");
  EXPECT_TRUE(unquantized->quantization.empty());
}

// An unquantized MLX directory (its shards say "mlx") is MLX's; shards of
// MLX's format beside another's are refused.
TEST_F(ImportClaimsTest, ShardFormatsMakeADirectoryMlx) {
  const std::filesystem::path p = dir_ / "mlx";
  Write(p / "config.json", std::string_view(R"({"model_type":"t"})"));
  Write(p / "model-1.safetensors", Shard({E("a", "BF16", {4})}, R"({"format":"mlx"})"));
  Write(p / "model-2.safetensors", Shard({E("b", "BF16", {4})}, R"({"format":"mlx"})"));
  Write(p / "model.safetensors.index.json",
        Index({{"a", "model-1.safetensors"}, {"b", "model-2.safetensors"}}));
  const auto c = ReadCheckpoint(p, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "mlx");
  Write(p / "model-2.safetensors", Shard({E("b", "BF16", {4})}, R"({"format":"pt"})"));
  EXPECT_EQ(RuleOf(ReadCheckpoint(p, ::geteuid())), Rule::kShards);
  Write(p / "model-2.safetensors", Shard({E("b", "BF16", {4})}));
  EXPECT_TRUE(ReadCheckpoint(p, ::geteuid()).has_value());
}

// A checkpoint keeps every shard open: more of them than the usual soft
// open-file limit (1,024, systemd's for a service) still read, the soft
// limit raised as far as the hard limit allows.
TEST_F(ImportClaimsTest, KeepsMoreShardsOpenThanTheUsualSoftLimit) {
  rlimit limit{};
  ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0);
  constexpr std::size_t kShardCount = 1100;
  if (limit.rlim_max != RLIM_INFINITY && limit.rlim_max < kShardCount + 256) {
    GTEST_SKIP() << "an open-file hard limit of " << limit.rlim_max;
  }
  struct Restore {
    rlimit saved;
    ~Restore() { ::setrlimit(RLIMIT_NOFILE, &saved); }
  } restore{limit};
  rlimit low = limit;
  low.rlim_cur = 1024;
  ASSERT_EQ(::setrlimit(RLIMIT_NOFILE, &low), 0);
  const std::filesystem::path p = dir_ / "many";
  Write(p / "config.json", std::string_view(R"({"model_type":"t"})"));
  std::map<std::string, std::string> index;
  for (std::size_t i = 0; i < kShardCount; ++i) {
    const std::string shard = std::format("model-{:05}.safetensors", i);
    const std::string tensor = std::format("t{}", i);
    Write(p / shard, Shard({E(tensor, "U8", {1})}));
    index.emplace(tensor, shard);
  }
  Write(p / "model.safetensors.index.json", Index(index));
  const auto c = ReadCheckpoint(p, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->open.size(), kShardCount + 2);
}

}  // namespace
}  // namespace llmp::import
