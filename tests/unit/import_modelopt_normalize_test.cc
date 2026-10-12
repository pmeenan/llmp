// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "components/step.h"
#include "import/modelopt_qwen38.h"
#include "qwen38_fixture.h"
#include "spec/mappings/qwen38.h"

namespace llmp::import {
namespace {
namespace fx = llmp::test_support::qwen38;
using format::Encoding;
using format::PartRole;
struct NativeFixture {
  spec::Source source{};
  std::vector<spec::SourceValue> values{};
};
NativeFixture Native() {
  const auto& p = model::Qwen38Flash();
  auto resources = fx::Resources(p, fx::Form::kGgmlExperts);
  auto old = fx::SourceOf("qwen4exp", "safetensors", std::string(fx::kModelOptConverter), resources,
                          spec::mappings::Qwen38Mapping());
  NativeFixture result;
  result.source.architecture = "qwen4exp";
  result.source.container = "safetensors";
  result.source.folds = {"gdn.value_heads_tiled"};
  for (auto tensor : old.tensors) {
    const bool table = tensor.name.starts_with("per_layer_token_embd.");
    const bool experts = tensor.name.find("_exps.") != std::string::npos;
    if ((table && tensor.part == PartRole::kTensorScale) ||
        (experts && tensor.name.ends_with(".weight_scale_2")))
      continue;
    if (table || tensor.experts) {
      tensor.format.encoding = Encoding::kNvfp4;
      tensor.format.layout = format::Layout::kSplit;
      tensor.format.params = {.tensor_scale = true};
      tensor.format.stacked = 1;
      const auto name = tensor.name;
      for (const auto part : {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale}) {
        tensor.part = part;
        tensor.name =
            part == PartRole::kCodes
                ? name
                : name.substr(0, name.size() - 7) +
                      (part == PartRole::kBlockScales ? ".weight_scale" : ".weight_scale_2");
        result.source.tensors.push_back(tensor);
      }
    } else {
      if (tensor.format.encoding == Encoding::kF32 &&
          tensor.format.layout == format::Layout::kDense)
        tensor.format.encoding = Encoding::kBF16;
      result.source.tensors.push_back(std::move(tensor));
    }
  }
  model::Qwen38Binding binding;
  binding.layers.resize(48);
  binding.format = model::Qwen38Format::kModelOpt;
  binding.experts = model::Qwen38Experts::kGgml;
  std::vector<std::int64_t> offsets, vocab;
  for (std::int64_t i = 0; i < 16; ++i) {
    offsets.push_back(i * 20000096);
    vocab.push_back(20000096 - i * 8);
  }
  auto hash = model::CheckQwen38PleHash(p, std::vector<std::int64_t>{3, 5, 7}, offsets, vocab,
                                        fx::kTableRows);
  EXPECT_TRUE(hash);
  auto values = spec::mappings::Qwen38V0Values(p, binding, *hash);
  EXPECT_TRUE(values);
  result.values = std::move(*values);
  for (auto& v : result.values)
    if (v.key == "llmp.qwen4exp.expert_scales") v.value = false;
  return result;
}
TEST(ImportModelOptNormalizeTest, NativeSplitPartsValidateAcrossCompleteTarget) {
  auto f = Native();
  const auto& mapping = spec::mappings::Qwen38NativeMapping();
  EXPECT_TRUE(mapping.folds.empty());
  EXPECT_TRUE(mapping.policies.empty());
  EXPECT_TRUE(mapping.v0_assembly.empty());
  EXPECT_TRUE(std::ranges::none_of(mapping.tensors, [](const auto& rule) {
    return rule.pattern.ends_with(".codes") || rule.pattern.ends_with(".scales");
  }));
  for (const auto& rule : mapping.tensors)
    if (rule.pattern == "per_layer_token_embd.weight" ||
        (rule.experts && rule.pattern.ends_with(".weight"))) {
      EXPECT_EQ(rule.part, PartRole::kCodes);
      EXPECT_FALSE(rule.any_part);
    }
  auto spec = spec::Synthesize(mapping, f.source, f.values);
  ASSERT_TRUE(spec) << spec.error();
  auto valid = components::ValidateSpec(*spec, f.source.tensors);
  ASSERT_TRUE(valid) << valid.error();
  ASSERT_EQ(spec->stages.size(), 1U);
  const auto& stage = spec->stages[0];
  ASSERT_EQ(stage.layers.size(), 48U);
  EXPECT_EQ(stage.folds, std::vector<std::string>{"gdn.value_heads_tiled"});
  unsigned linears = 0, attention = 0, moes = 0;
  for (const auto& layer : stage.layers)
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
                                            return w.role == role &&
                                                   w.part == PartRole::kTensorScale && w.experts;
                                          }),
                    1);
        }
        EXPECT_TRUE(std::ranges::none_of(c.weights, [](const auto& w) {
          return w.role == "gate_scale" || w.role == "up_scale" || w.role == "down_scale";
        }));
      }
    }
  EXPECT_EQ(linears, 36U);
  EXPECT_EQ(attention, 12U);
  EXPECT_EQ(moes, 48U);
  ASSERT_EQ(stage.side.size(), 1U);
  EXPECT_EQ(stage.side[0].source.weights.size(), 3U);
}
TEST(ImportModelOptNormalizeTest, NativeMappingRejectsMissingPartAndAppliedRawFolds) {
  auto f = Native();
  auto before = f.source.tensors;
  std::erase_if(f.source.tensors,
                [](const auto& t) { return t.name == "blk.0.ffn_gate_exps.weight_scale_2"; });
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
TEST(ImportModelOptNormalizeTest, ExistingConvertedMappingKeepsItsFoldAndPolicyContract) {
  const auto& old = spec::mappings::Qwen38Mapping();
  EXPECT_FALSE(old.folds.empty());
  EXPECT_FALSE(old.policies.empty());
  EXPECT_FALSE(old.v0_assembly.empty());
  EXPECT_TRUE(std::ranges::none_of(old.params, [](const auto& p) {
    return p.param == "a_log" || p.param == "norm_zero_centered";
  }));
  EXPECT_TRUE(std::ranges::any_of(old.tensors, [](const auto& t) {
    return t.role == "gate_scale" && t.part == PartRole::kData;
  }));
}
TEST(ImportModelOptNormalizeTest, UnknownSourceRefusalKeepsSourceOwnership) {
  Checkpoint source;
  source.container = "mlx";
  source.architecture = "qwen3_8_flash_next";
  source.tensors.push_back({.name = "owned source"});
  const auto* owned = source.tensors.data();
  EXPECT_FALSE(ModelOptTargetBound(source));
  EXPECT_FALSE(NormalizeModelOptTarget(std::move(source), {}));
  EXPECT_EQ(source.tensors.data(), owned);
  ASSERT_EQ(source.tensors.size(), 1U);
  EXPECT_EQ(source.tensors[0].name, "owned source");
}
}  // namespace
}  // namespace llmp::import
