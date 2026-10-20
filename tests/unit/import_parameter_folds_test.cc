// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/component.h"
#include "import/parameter_folds.h"
#include "spec/spec.h"

namespace llmp::import {
namespace {

using format::Encoding;
using format::PartRole;

// Every derived role names a flag its component takes and a catalog fold.
TEST(ImportParameterFoldsTest, DerivedRolesNameTheirFlagAndACatalogFold) {
  std::size_t derived = 0;
  for (const components::ComponentDef* def : components::Catalog())
    for (const auto& role : def->roles) {
      if (role.derived.fold.empty() && role.derived.flag.empty()) continue;
      ++derived;
      SCOPED_TRACE(std::string(def->kind) + "." + std::string(role.name));
      EXPECT_NE(spec::FindFold(role.derived.fold), nullptr);
      EXPECT_TRUE(std::ranges::any_of(def->params, [&](const auto& p) {
        return p.name == role.derived.flag && p.type == components::ParamType::kBool;
      }));
    }
  EXPECT_GE(derived, 12U);
}

TensorAssembly Dense(std::string name, std::uint64_t n) {
  return {.name = std::move(name),
          .format = {.encoding = Encoding::kBF16,
                     .layout = format::Layout::kDense,
                     .shape = {.ne = {n, 1, 1, 1}, .rank = 1}},
          .parts = {{.part = PartRole::kData, .kind = PartAssemblyKind::kCopy, .sources = {0}}}};
}

spec::ComponentRef Ref(std::string kind, std::vector<spec::Param> params,
                       std::vector<std::pair<std::string, std::string>> weights) {
  spec::ComponentRef c{.kind = std::move(kind), .version = 1, .params = std::move(params)};
  for (auto& [role, tensor] : weights)
    c.weights.push_back({.role = std::move(role), .tensor = std::move(tensor)});
  return c;
}

// One layer: a gated delta rule (its A logarithmic) and a hyper-connection
// mix (its norm zero-centered).
struct Case {
  spec::Stage stage;
  std::vector<TensorAssembly> outputs;
};
Case Make(bool a_log, bool zero_centered) {
  Case c;
  c.stage.layers.push_back({.components = {
                                Ref("attention.gdn", {{"a_log", a_log}},
                                    {{"a", "blk.0.ssm_a"}, {"norm", "blk.0.ssm_norm.weight"}}),
                                Ref("hc.mix", {{"norm_zero_centered", zero_centered}},
                                    {{"norm", "blk.0.hc_attn_norm.weight"}}),
                            }});
  c.stage.heads.final_norm =
      Ref("hc.mix", {{"norm_zero_centered", zero_centered}}, {{"norm", "output_hc_norm.weight"}});
  c.outputs = {Dense("blk.0.ssm_a", 48), Dense("blk.0.ssm_norm.weight", 128),
               Dense("blk.0.hc_attn_norm.weight", 10240), Dense("output_hc_norm.weight", 10240)};
  return c;
}

TEST(ImportParameterFoldsTest, FoldsWhatTheFlagsDeriveByRole) {
  auto c = Make(true, true);
  auto applied = FoldDerivedParameters(c.stage, c.outputs);
  ASSERT_TRUE(applied) << applied.error().ToString();
  ASSERT_EQ(applied->size(), 2U);
  EXPECT_EQ((*applied)[0].fold, "norm.plus_one");
  EXPECT_EQ((*applied)[0].outputs, (std::vector<std::uint32_t>{2, 3}));
  EXPECT_EQ((*applied)[1].fold, "ssm.a_neg_exp");
  EXPECT_EQ((*applied)[1].outputs, (std::vector<std::uint32_t>{0}));
  EXPECT_EQ(c.outputs[0].fold, ScalarFold::kANegExp);
  EXPECT_EQ(c.outputs[0].format.encoding, Encoding::kF32);
  // The gated delta rule's own norm derives nothing: unchanged.
  EXPECT_FALSE(c.outputs[1].fold);
  EXPECT_EQ(c.outputs[1].format.encoding, Encoding::kBF16);
  for (const std::size_t o : {2U, 3U}) {
    EXPECT_EQ(c.outputs[o].fold, ScalarFold::kNormPlusOne);
    EXPECT_EQ(c.outputs[o].format.encoding, Encoding::kF32);
  }
}

TEST(ImportParameterFoldsTest, AFoldCoversAllItsTensorsOrNone) {
  {  // flags off: nothing derived, nothing folded
    auto c = Make(false, false);
    auto applied = FoldDerivedParameters(c.stage, c.outputs);
    ASSERT_TRUE(applied);
    EXPECT_TRUE(applied->empty());
  }
  {  // one governed norm not a foldable copy: no norm fold, A still
    auto c = Make(true, true);
    c.outputs[3].format.encoding = Encoding::kI32;
    auto applied = FoldDerivedParameters(c.stage, c.outputs);
    ASSERT_TRUE(applied);
    ASSERT_EQ(applied->size(), 1U);
    EXPECT_EQ(applied->front().fold, "ssm.a_neg_exp");
    EXPECT_FALSE(c.outputs[2].fold);
    EXPECT_EQ(c.outputs[2].format.encoding, Encoding::kBF16);
  }
  {  // a governed norm bound by another stage's artifact
    auto c = Make(false, true);
    c.stage.heads.final_norm->weights[0].stage = "target";
    auto applied = FoldDerivedParameters(c.stage, c.outputs);
    ASSERT_TRUE(applied);
    EXPECT_TRUE(applied->empty());
  }
  {  // a governed tensor also read by a role that does not derive it
    auto c = Make(true, false);
    c.stage.layers[0].components[1].weights[0].tensor = "blk.0.ssm_a";
    auto applied = FoldDerivedParameters(c.stage, c.outputs);
    ASSERT_TRUE(applied);
    EXPECT_TRUE(applied->empty());
    EXPECT_FALSE(c.outputs[0].fold);
  }
  {  // a fold the stage already carries never applies twice
    auto c = Make(true, true);
    c.stage.folds = {"norm.plus_one"};
    auto applied = FoldDerivedParameters(c.stage, c.outputs);
    ASSERT_TRUE(applied);
    ASSERT_EQ(applied->size(), 1U);
    EXPECT_EQ(applied->front().fold, "ssm.a_neg_exp");
  }
  {  // a governed tensor wider than a per-channel vector: no norm fold
    auto c = Make(false, true);
    c.outputs[3].format.shape.ne[0] = (std::uint64_t{1} << 18U) + 1;  // over 1 MiB in F32
    auto applied = FoldDerivedParameters(c.stage, c.outputs);
    ASSERT_TRUE(applied);
    EXPECT_TRUE(applied->empty());
    EXPECT_EQ(c.outputs[2].format.encoding, Encoding::kBF16);
  }
  {  // a matrix, however small
    auto c = Make(false, true);
    c.outputs[2].format.shape = {.ne = {4, 4, 1, 1}, .rank = 2};
    auto applied = FoldDerivedParameters(c.stage, c.outputs);
    ASSERT_TRUE(applied);
    EXPECT_TRUE(applied->empty());
  }
  {  // a component the catalog lacks
    auto c = Make(true, true);
    c.stage.layers[0].components[0].kind = "attention.unknown";
    EXPECT_FALSE(FoldDerivedParameters(c.stage, c.outputs));
  }
}

}  // namespace
}  // namespace llmp::import
