// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/step.h"
#include "import/mlx_qwen38.h"
#include "qwen38_fixture.h"
#include "spec/mappings/qwen38.h"

namespace llmp::import {
namespace {
namespace fx = llmp::test_support::qwen38;
using format::Encoding;
using format::PartRole;

TEST(ImportMlxNormalizeTest, NativeSplitPartsValidateAcrossCompleteTarget) {
  auto f = fx::NativeMlx();
  EXPECT_EQ(f.source.architecture, "qwen4exp");
  EXPECT_EQ(f.source.container, "mlx");
  EXPECT_EQ(f.source.folds, std::vector<std::string>{"gdn.value_heads_tiled"});
  EXPECT_EQ(f.source.tensors.size(), 2274U);

  // Verify inventory hash computation
  std::vector<const spec::SourceTensor*> sorted;
  sorted.reserve(f.source.tensors.size());
  for (const auto& t : f.source.tensors) sorted.push_back(&t);
  std::ranges::sort(sorted, {}, [](const auto* tensor) { return tensor->name; });
  base::Sha256 hash;
  for (const auto* tensor : sorted) {
    hash.Update(tensor->name)
        .Update("\n")
        .Update(format::Info(tensor->format.encoding).name)
        .Update("\n");
    for (std::uint8_t i = 0; i < tensor->format.shape.rank; ++i)
      hash.Update(std::to_string(tensor->format.shape.ne[i])).Update(",");
    hash.Update("\n");
  }
  const auto hex = base::ToHex(hash.Finish());
  EXPECT_EQ(hex, "8cabbbeebf6a6786dbc52ac0471b5dd34d360f4a9c000de7a806bb9234f57dd5");

  const auto& mapping = spec::mappings::Qwen38NativeMapping();
  EXPECT_TRUE(mapping.folds.empty());
  EXPECT_TRUE(mapping.policies.empty());
  EXPECT_TRUE(mapping.v0_assembly.empty());

  auto spec = spec::Synthesize(mapping, f.source, f.values);
  ASSERT_TRUE(spec) << spec.error();
  auto valid = components::ValidateSpec(*spec, f.source.tensors);
  ASSERT_TRUE(valid) << valid.error();
  ASSERT_EQ(spec->stages.size(), 1U);
  const auto& stage = spec->stages[0];
  ASSERT_EQ(stage.layers.size(), 48U);
  EXPECT_EQ(stage.folds, std::vector<std::string>{"gdn.value_heads_tiled"});

  // Embedding and output head must NOT be tied.
  ASSERT_TRUE(stage.embedding);
  ASSERT_TRUE(stage.heads.output);
  EXPECT_NE(stage.embedding->weights[0].tensor, stage.heads.output->weights[0].tensor);

  unsigned linears = 0, attention = 0, moes = 0;
  for (const auto& layer : stage.layers) {
    for (const auto& c : layer.components) {
      if (c.kind == "attention.gdn") {
        ++linears;
        ASSERT_NE(c.Find("a_log"), nullptr);
        EXPECT_EQ(std::get<bool>(c.Find("a_log")->value), true);
      }
      if (c.kind == "attention.qsa") ++attention;
      if (c.kind == "ffn.moe") {
        ++moes;
        ASSERT_NE(c.Find("projection_scales"), nullptr);
        EXPECT_EQ(std::get<bool>(c.Find("projection_scales")->value), false);
        for (const auto role : {"gate_experts", "up_experts", "down_experts"}) {
          EXPECT_EQ(std::ranges::count_if(c.weights, [&](const auto& w) { return w.role == role; }),
                    3);
          EXPECT_EQ(std::ranges::count_if(c.weights,
                                          [&](const auto& w) {
                                            return w.role == role && w.part == PartRole::kBiases &&
                                                   w.experts;
                                          }),
                    1);
          EXPECT_EQ(std::ranges::count_if(c.weights,
                                          [&](const auto& w) {
                                            return w.role == role &&
                                                   w.part == PartRole::kBlockScales && w.experts;
                                          }),
                    1);
          EXPECT_EQ(std::ranges::count_if(c.weights,
                                          [&](const auto& w) {
                                            return w.role == role && w.part == PartRole::kCodes &&
                                                   w.experts;
                                          }),
                    1);
        }
      }
    }
  }
  EXPECT_EQ(linears, 36U);
  EXPECT_EQ(attention, 12U);
  EXPECT_EQ(moes, 48U);
  ASSERT_EQ(stage.side.size(), 1U);
  EXPECT_EQ(stage.side[0].source.weights.size(), 3U);
}

TEST(ImportMlxNormalizeTest, NativeMappingRejectsMissingPartAndAppliedRawFolds) {
  auto f = fx::NativeMlx();
  auto before = f.source.tensors;
  std::erase_if(f.source.tensors,
                [](const auto& t) { return t.name == "blk.0.ffn_gate_exps.weight_bias"; });
  auto missing = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), f.source, f.values);
  ASSERT_TRUE(missing);
  EXPECT_FALSE(components::ValidateSpec(*missing, f.source.tensors));

  f.source.tensors = std::move(before);
  for (const auto fold : {"norm.plus_one", "ssm.a_neg_exp"}) {
    f.source.folds = {"gdn.value_heads_tiled", fold};
    auto contradictory =
        spec::Synthesize(spec::mappings::Qwen38NativeMapping(), f.source, f.values);
    ASSERT_TRUE(contradictory);
    EXPECT_FALSE(components::ValidateSpec(*contradictory, f.source.tensors));
  }
  f.source.folds.clear();
  auto unpermuted = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), f.source, f.values);
  ASSERT_TRUE(unpermuted);
  EXPECT_FALSE(components::ValidateSpec(*unpermuted, f.source.tensors));
}

TEST(ImportMlxNormalizeTest, UnknownSourceRefusalKeepsSourceOwnership) {
  Checkpoint source;
  source.container = "safetensors";
  source.architecture = "qwen4_exp";
  source.quantization = "mlx affine";
  source.tensors.push_back({.name = "owned source"});
  const auto* owned = source.tensors.data();
  EXPECT_FALSE(MlxTargetBound(source));
  EXPECT_FALSE(NormalizeMlxTarget(std::move(source), {}));
  EXPECT_EQ(source.tensors.data(), owned);
  ASSERT_EQ(source.tensors.size(), 1U);
  EXPECT_EQ(source.tensors[0].name, "owned source");
}

}  // namespace
}  // namespace llmp::import
