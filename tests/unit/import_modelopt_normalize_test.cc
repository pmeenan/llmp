// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/step.h"
#include "import/normalize.h"
#include "qwen38_fixture.h"
#include "spec/mappings/qwen38.h"

namespace llmp::import {
namespace {
using format::Encoding;
using format::PartRole;
struct NativeFixture {
  spec::Source source{};
  std::vector<spec::SourceValue> values{};
};
// The native target as Normalize publishes it.
NativeFixture Native() {
  auto form = test_support::qwen38::NativeModelOpt();
  return {std::move(form.source), std::move(form.values)};
}
const spec::Param* ParamOf(const spec::ComponentRef& c, std::string_view name) {
  return c.Find(name);
}
TEST(ImportModelOptNormalizeTest, NativeRelaidTargetValidatesAcrossCompleteTarget) {
  auto f = Native();
  const auto& mapping = spec::mappings::Qwen38NativeMapping();
  ASSERT_FALSE(mapping.folds.empty());
  EXPECT_TRUE(
      std::ranges::all_of(mapping.folds, [](const spec::FoldRule& fold) { return fold.imported; }));
  // Activation scales select the native rule; structural patterns default
  // on in either form and arithmetic changes still require separate evidence.
  ASSERT_EQ(mapping.policies.size(), 2U);
  EXPECT_EQ(mapping.policies[0].gate, "llmp.qwen4exp.expert_activation_scales");
  EXPECT_EQ(mapping.policies[0].text, spec::mappings::Qwen38Mapping().policies[0].text);
  EXPECT_TRUE(mapping.policies[1].gate.empty());
  EXPECT_EQ(mapping.policies[1].text, spec::mappings::Qwen38Mapping().policies.back().text);
  EXPECT_FALSE(mapping.policies[1].text.contains("enable moe_experts\n"));
  // The experts' tables in the scale-atom layout: gate-up stacked.
  std::size_t tables = 0, vectors = 0;
  for (const auto& t : f.source.tensors) {
    if (t.experts) {
      ++tables;
      EXPECT_EQ(t.format.layout, format::Layout::kScaleAtoms) << t.name;
      EXPECT_EQ(t.format.stacked, t.name.contains("gate_up") ? 2U : 1U) << t.name;
      EXPECT_FALSE(t.format.params.tensor_scale) << t.name;
    }
    if (t.name.ends_with("_exps.weight_scale_2")) {
      ++vectors;
      EXPECT_EQ(t.format.encoding, Encoding::kF32);
      EXPECT_EQ(t.format.shape.ne[0], 512U);
    }
  }
  EXPECT_EQ(tables, 48U * 4);
  EXPECT_EQ(vectors, 48U * 3);
  // Its zero-centered norms and A folded into F32 by role.
  EXPECT_EQ(f.source.folds,
            (std::vector<std::string>{"gdn.value_heads_tiled", "norm.plus_one", "ssm.a_neg_exp"}));
  for (const auto& t : f.source.tensors)
    if (t.name.ends_with("ssm_a") || t.name.ends_with("attn_q_norm.weight") ||
        t.name.ends_with("hc_attn_norm.weight") || t.name == "output_hc_norm.weight")
      EXPECT_EQ(t.format.encoding, Encoding::kF32) << t.name;
    else if (t.name.ends_with("ssm_norm.weight") || t.name.ends_with("attn_norm.weight"))
      EXPECT_EQ(t.format.encoding, Encoding::kBF16) << t.name;
  EXPECT_EQ(f.source.tensors.size(), 1789U);
  auto spec = spec::Synthesize(mapping, f.source, f.values);
  ASSERT_TRUE(spec) << spec.error();
  auto valid = components::ValidateSpec(*spec, f.source.tensors);
  ASSERT_TRUE(valid) << valid.error();
  ASSERT_EQ(spec->stages.size(), 1U);
  const auto& stage = spec->stages[0];
  ASSERT_EQ(stage.layers.size(), 48U);
  EXPECT_EQ(stage.folds, f.source.folds);
  unsigned linears = 0, attention = 0, moes = 0;
  for (const auto& layer : stage.layers)
    for (const auto& c : layer.components) {
      // The folds left no derived parameter to compute.
      EXPECT_EQ(ParamOf(c, "a_log"), nullptr) << c.kind;
      EXPECT_EQ(ParamOf(c, "norm_zero_centered"), nullptr) << c.kind;
      if (c.kind == "attention.gdn") ++linears;
      if (c.kind == "attention.qsa") ++attention;
      if (c.kind == "ffn.moe") {
        ++moes;
        EXPECT_EQ(std::get<bool>(ParamOf(c, "projection_scales")->value), false);
        for (const auto scale : {"gate_scale", "up_scale", "down_scale"})
          EXPECT_EQ(std::get<bool>(ParamOf(c, scale)->value), true);
        EXPECT_EQ(std::get<bool>(ParamOf(c, "stacked")->value), true);
        for (const auto& [role, count] :
             std::array<std::pair<std::string_view, int>, 5>{{{"gate_up_experts", 2},
                                                              {"down_experts", 2},
                                                              {"gate_scale", 1},
                                                              {"up_scale", 1},
                                                              {"down_scale", 1}}})
          EXPECT_EQ(std::ranges::count_if(c.weights, [&](const auto& w) { return w.role == role; }),
                    count)
              << role;
      }
    }
  EXPECT_EQ(linears, 36U);
  EXPECT_EQ(attention, 12U);
  EXPECT_EQ(moes, 48U);
  // The n-gram table row-interleaved: its rows and its tensor scale.
  ASSERT_EQ(stage.side.size(), 1U);
  EXPECT_EQ(stage.side[0].source.weights.size(), 2U);
  // Its policy: the W4A4 form; without activation scales, the ladder's.
  auto rule = spec::DefaultRule(mapping, f.values, {.context = 8192, .slots = 1});
  ASSERT_TRUE(rule && *rule != nullptr);
  EXPECT_EQ(*rule, &mapping.policies[0]);
  EXPECT_FALSE((**rule).text.contains("candidate_numerics "));
  auto w4a16 = f.values;
  for (auto& v : w4a16)
    if (v.key == "llmp.qwen4exp.expert_activation_scales") v.value = false;
  rule = spec::DefaultRule(mapping, w4a16, {.context = 8192, .slots = 1});
  ASSERT_TRUE(rule && *rule != nullptr);
  EXPECT_EQ(*rule, &mapping.policies[1]);
  EXPECT_FALSE((**rule).text.contains("candidate_numerics "));
}
TEST(ImportModelOptNormalizeTest, NativeMappingRejectsMissingPartAndReadsItsFolds) {
  auto f = Native();
  auto before = f.source.tensors;
  std::erase_if(f.source.tensors,
                [](const auto& t) { return t.name == "blk.0.ffn_down_exps.weight_scale"; });
  auto missing = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), f.source, f.values);
  if (missing) EXPECT_FALSE(components::ValidateSpec(*missing, f.source.tensors));
  f.source.tensors = std::move(before);
  // Without a fold, its flag returns (the checkpoint's convention, as an
  // import that could not fold publishes it).
  for (const auto& [fold, flag] : std::array<std::pair<std::string_view, std::string_view>, 2>{
           {{"norm.plus_one", "norm_zero_centered"}, {"ssm.a_neg_exp", "a_log"}}}) {
    f.source.folds = {"gdn.value_heads_tiled", std::string(fold)};
    auto spec = spec::Synthesize(spec::mappings::Qwen38NativeMapping(), f.source, f.values);
    ASSERT_TRUE(spec) << spec.error();
    // The flag the fold clears is back on: the other one's.
    const std::string_view back = flag == "a_log" ? "norm_zero_centered" : "a_log";
    bool found = false;
    for (const auto& layer : spec->stages[0].layers)
      for (const auto& c : layer.components) {
        EXPECT_EQ(ParamOf(c, flag), nullptr);
        if (const auto* p = ParamOf(c, back)) found = found || std::get<bool>(p->value);
      }
    EXPECT_TRUE(found) << fold;
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
  EXPECT_TRUE(std::ranges::none_of(old.params, [](const auto& p) {
    return p.param == "a_log" || p.param == "norm_zero_centered";
  }));
  EXPECT_TRUE(std::ranges::any_of(old.tensors, [](const auto& t) {
    return t.role == "gate_scale" && t.part == PartRole::kData;
  }));
}
TEST(ImportModelOptNormalizeTest, RegistryRecognitionIsUniqueAndRequiresAllSourceFields) {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 2> kIndexed = {{
      {"safetensors", "modelopt nvfp4+mxfp8"},
      {"mlx", "mlx affine"},
  }};
  for (const auto& [container, quantization] : kIndexed) {
    Checkpoint source;
    source.container = std::string(container);
    source.architecture = "qwen4_exp";
    source.quantization = std::string(quantization);
    EXPECT_NE(FindImportMapping(source.container, source.architecture), nullptr);
    source.quantization += "-unknown";
    EXPECT_FALSE(NormalizationBound(source));
    source.quantization = std::string(quantization);
    source.container += "-unknown";
    EXPECT_EQ(FindImportMapping(source.container, source.architecture), nullptr);
    source.container = std::string(container);
    source.architecture += "-unknown";
    EXPECT_EQ(FindImportMapping(source.container, source.architecture), nullptr);
  }
}

TEST(ImportModelOptNormalizeTest, UnknownSourceRefusalKeepsSourceOwnership) {
  Checkpoint source;
  source.container = "mlx";
  source.architecture = "qwen3_8_flash_next";
  source.tensors.push_back({.name = "owned source"});
  const auto* owned = source.tensors.data();
  EXPECT_FALSE(NormalizationBound(source));
  EXPECT_FALSE(Normalize(std::move(source), {}));
  EXPECT_EQ(source.tensors.data(), owned);
  ASSERT_EQ(source.tensors.size(), 1U);
  EXPECT_EQ(source.tensors[0].name, "owned source");
}
}  // namespace
}  // namespace llmp::import
