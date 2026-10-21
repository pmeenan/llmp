// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_TESTS_UNIT_COMPONENT_CATALOG_CASES_H_
#define LLMP_TESTS_UNIT_COMPONENT_CATALOG_CASES_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/spatial.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "kernels/reference/exec.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "spec/spec.h"

namespace llmp::components::test_catalog {

namespace ref = llmp::kernels::reference;
using format::Encoding;
using spec::ComponentRef;
using spec::Param;
using spec::ParamValue;

inline constexpr std::array<Encoding, 5> kProductEncodings{
    Encoding::kF32, Encoding::kF16, Encoding::kBF16, Encoding::kQ8_0, Encoding::kQ4_0,
};

inline ComponentRef MakeRef(std::string kind, std::uint16_t version, std::vector<Param> params,
                            std::vector<spec::WeightBinding> weights = {}) {
  std::ranges::sort(params, {}, &Param::name);
  std::ranges::sort(weights, {}, &spec::WeightBinding::role);
  return {
      .kind = std::move(kind),
      .version = version,
      .params = std::move(params),
      .weights = std::move(weights),
  };
}

inline Param P(std::string name, ParamValue v) {
  return {.name = std::move(name), .value = std::move(v)};
}
inline ParamValue I(std::int64_t v) { return ParamValue(v); }
inline ParamValue R(double v) { return ParamValue(v); }
inline ParamValue N(std::string v) { return ParamValue(std::move(v)); }
inline ParamValue B(bool v) { return ParamValue(v); }
inline ParamValue L(std::vector<std::int64_t> v) { return ParamValue(std::move(v)); }

inline planner::Segment Seg(std::uint8_t slot, std::uint32_t rows, std::uint32_t past,
                            std::uint32_t read, std::uint32_t outputs, bool greedy = false) {
  return {
      .slot = slot,
      .rows = rows,
      .past = past,
      .read_cells = read,
      .outputs = outputs,
      .greedy = greedy,
  };
}

inline const std::vector<std::string>& GdnFolds() {
  static const std::vector<std::string> folds{"gdn.value_heads_tiled", "norm.plus_one",
                                              "ssm.a_neg_exp"};
  return folds;
}

inline double Gap(std::span<const float> got, std::span<const double> want) {
  EXPECT_EQ(got.size(), want.size());
  double worst = 0.0;
  double scale = 1.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
      if (static_cast<double>(got[i]) == want[i]) continue;
      return std::numeric_limits<double>::infinity();
    }
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
    scale = std::max(scale, std::fabs(want[i]));
  }
  return worst / scale;
}

struct CatalogTokenCase {
  std::string label{};
  ComponentRef ref{};
  StageFacts stage{};
  Site site = Site::kLayer;
  ComponentInputs inputs{};
  std::uint64_t table_rows = 40;
  std::vector<std::uint64_t> cells{};
  planner::StepShape shape{};
  std::vector<std::int32_t> tokens{};
  std::vector<format::TensorFormat> formats{};
  std::map<std::uint16_t, std::pair<float, float>> ranges{};
  bool random_state = false;
  double tolerance = 2e-2;
  double state_tolerance = 2e-2;
};

struct CatalogTokenRecipe {
  std::string_view kind{};
  std::uint16_t version = 1;
  std::vector<CatalogTokenCase> cases{};
};

struct CatalogSpatialCase {
  std::string label{};
  ComponentRef ref{};
  SpatialOwner owner{0, {2, 2, 2, 1}};
  std::vector<SpatialInput> inputs{};
  std::vector<SpatialLink> links{};
  std::vector<BoundTensor> tensors{};
  double tolerance = 2e-2;
};

struct CatalogSpatialRecipe {
  std::string_view kind{};
  std::uint16_t version = 1;
  std::vector<CatalogSpatialCase> cases{};
};

inline format::TensorFormat MakeRoleFormat(const RoleShape& r, Encoding enc,
                                           std::uint64_t free_extent) {
  format::Shape sh;
  sh.rank = r.rank;
  for (std::uint8_t d = 0; d < r.rank; ++d) {
    sh.ne[d] = r.ne[d] == 0 ? free_extent : r.ne[d];
  }
  format::TensorFormat f = ir::Dense(enc, sh);
  f.shape.rank = r.rank;
  if (format::Info(enc).quantized) {
    f.layout = format::Layout::kGgmlBlocks;
  }
  return f;
}

inline bool RoleAcceptsEncoding(const ComponentDef& def, const RoleShape& r, Encoding enc,
                                std::uint64_t free_extent) {
  const auto& role = def.roles[r.role];
  const std::uint64_t ne0 = r.ne[0] == 0 ? free_extent : r.ne[0];
  if (format::Info(enc).quantized && (r.rank < 2 || ne0 % 32 != 0)) {
    return false;
  }
  if (r.rank == 3 && !format::Info(enc).quantized) {
    return false;
  }
  if (role.encodings.empty()) {
    return std::ranges::contains(kProductEncodings, enc);
  }
  return std::ranges::contains(role.encodings, enc);
}

inline void AppendFormatCases(CatalogTokenCase proto, std::vector<CatalogTokenCase>* out) {
  auto in = MakeInstance(proto.ref, proto.stage, proto.site, 0, 1);
  ASSERT_TRUE(in.has_value()) << proto.ref.kind << ": " << in.error();
  const auto shapes = in->def->shapes(*in);
  ASSERT_TRUE(shapes.has_value()) << proto.ref.kind << ": " << shapes.error();

  const format::TensorFormat dummy = ir::Dense(Encoding::kF32, ir::ShapeOf({1}));
  proto.formats.assign(in->def->roles.size(), dummy);
  for (const RoleShape& r : *shapes) {
    const auto& role = in->def->roles[r.role];
    const Encoding base_enc = r.rank == 3              ? Encoding::kQ8_0
                              : role.encodings.empty() ? Encoding::kF32
                                                       : role.encodings.front();
    proto.formats[r.role] = MakeRoleFormat(r, base_enc, proto.table_rows);
  }
  if (shapes->empty()) {
    proto.label = "f32";
    out->push_back(std::move(proto));
    return;
  }

  constexpr std::array<Encoding, 6> kAllEncodings{
      Encoding::kF32,  Encoding::kF16,  Encoding::kBF16,
      Encoding::kQ8_0, Encoding::kQ4_0, Encoding::kI32,
  };
  for (Encoding enc : kAllEncodings) {
    CatalogTokenCase c = proto;
    bool any = false;
    for (const RoleShape& r : *shapes) {
      if (RoleAcceptsEncoding(*in->def, r, enc, c.table_rows)) {
        c.formats[r.role] = MakeRoleFormat(r, enc, c.table_rows);
        any = true;
      }
    }
    if (!any) continue;
    const bool any_quant = std::ranges::any_of(c.formats, [](const format::TensorFormat& f) {
      return f.encoding == Encoding::kQ8_0 || f.encoding == Encoding::kQ4_0;
    });
    if (any_quant) {
      c.tolerance = std::max(c.tolerance, 0.25);
      c.state_tolerance = std::max(c.state_tolerance, 0.25);
    } else if (enc == Encoding::kF16 || enc == Encoding::kBF16) {
      c.tolerance = std::max(c.tolerance, 5e-2);
      c.state_tolerance = std::max(c.state_tolerance, 5e-2);
    }
    c.label = std::string(format::Info(enc).name);
    out->push_back(std::move(c));
  }
}

inline std::vector<CatalogTokenRecipe> CatalogTokenRecipes() {
  std::vector<CatalogTokenRecipe> recipes;

  auto add =
      [&](std::string_view kind, std::uint16_t version, Site site, StageFacts stage,
          std::vector<Param> params, ComponentInputs inputs, std::vector<std::uint64_t> cells,
          planner::Segment seg, std::vector<std::int32_t> tokens = {},
          std::map<std::uint16_t, std::pair<float, float>> ranges = {}, bool random_state = false,
          planner::Phase phase = planner::Phase::kPrefill, std::uint64_t table_rows = 40) {
        CatalogTokenCase proto{
            .label = {},
            .ref = MakeRef(std::string(kind), version, std::move(params)),
            .stage = stage,
            .site = site,
            .inputs = inputs,
            .table_rows = table_rows,
            .cells = std::move(cells),
            .shape = {.phase = phase, .segments = {seg}},
            .tokens = std::move(tokens),
            .formats = {},
            .ranges = std::move(ranges),
            .random_state = random_state,
        };
        CatalogTokenRecipe recipe{.kind = kind, .version = version, .cases = {}};
        AppendFormatCases(std::move(proto), &recipe.cases);
        recipes.push_back(std::move(recipe));
      };

  // 1. attention.borrowed v1
  add("attention.borrowed", 1, Site::kLayer, {.width = 64, .context = 4096},
      {P("eps", R(1e-6)), P("heads", I(4)), P("key_dim", I(64)), P("kv_heads", I(2)),
       P("placement", N("sandwich")), P("qk_norm", B(true)), P("rope_base", R(10000.0)),
       P("rope_dims", I(64)), P("rope_freqs", B(false)), P("rope_scale", R(1.0)),
       P("rotation", N("neox")), P("scale", R(1.0 / std::sqrt(64.0))), P("target_layer", I(3)),
       P("value_dim", I(64)), P("window", I(8))},
      {}, {256, 256}, Seg(0, 1, 5, 256, 1), {}, {}, true, planner::Phase::kDecode);

  // 2. attention.dense v1
  add("attention.dense", 1, Site::kLayer, {.width = 64, .context = 4096},
      {P("eps", R(1e-6)), P("heads", I(4)), P("key_dim", I(64)), P("kv_heads", I(2)),
       P("placement", N("sandwich")), P("qk_norm", B(true)), P("rope_base", R(10000.0)),
       P("rope_dims", I(64)), P("rope_scale", R(1.0)), P("rotation", N("neox")),
       P("scale", R(1.0 / std::sqrt(64.0))), P("softcap", R(0.0)), P("value_dim", I(64)),
       P("window", I(8))},
      {}, {256, 256}, Seg(0, 3, 0, 256, 3));

  // 3. attention.gdn v1
  add("attention.gdn", 1, Site::kLayer,
      {.width = 64, .context = 4096, .streams = 4, .folds = GdnFolds()},
      {P("conv", I(4)), P("eps", R(1e-6)), P("head_dim", I(16)), P("k_heads", I(2)),
       P("v_heads", I(4))},
      {}, {1, 1}, Seg(0, 3, 0, 256, 3), {},
      {{4, {-0.5F, 0.5F}}, {5, {-1.2F, 0.5F}}, {6, {-0.6F, 0.6F}}});

  // 4. attention.mqa v1 (head_dim = 256, heads = 8, o_groups = 4 -> per = 512)
  add("attention.mqa", 1, Site::kLayer, {.width = 64, .context = 4096, .streams = 1},
      {P("cache", N("ring")), P("eps", R(1e-6)), P("head_dim", I(256)), P("heads", I(8)),
       P("o_groups", I(4)), P("o_rank", I(64)), P("q_rank", I(64)), P("rope_base", R(10000.0)),
       P("rope_dims", I(64)), P("window", I(6))},
      {}, {256}, Seg(0, 3, 0, 256, 3));

  // 5. attention.qsa v1
  add("attention.qsa", 1, Site::kLayer,
      {.width = 64, .context = 4096, .streams = 4, .folds = GdnFolds()},
      {P("budget", I(256)), P("eps", R(1e-6)), P("head_dim", I(64)), P("heads", I(4)),
       P("indexer", N("fused")), P("indexer_dim", I(64)), P("indexer_heads", I(2)),
       P("kv_heads", I(2)), P("ratio", I(4)), P("rope_base", R(10000.0)), P("rope_dims", I(64)),
       P("rope_sections", L({11, 11, 10, 0}))},
      {}, {256, 256, 256, 64}, Seg(0, 4, 0, 256, 4));

  // 6. concat.spatial v1
  add("concat.spatial", 1, Site::kLayer, {.width = 64, .context = 256}, {P("axis", I(2))},
      {.side = 64}, {}, Seg(0, 2, 0, 256, 2));

  // 7. condition.rows v1
  add("condition.rows", 1, Site::kLayer, {.width = 64, .context = 256},
      {P("prefix_rows", I(1)), P("quarter", I(0)), P("quarters", I(2))}, {.side = 128}, {},
      Seg(0, 2, 0, 256, 2));

  // 8. embedding.inject v1
  add("embedding.inject", 1, Site::kEmbedding, {.width = 64, .context = 4096},
      {P("cells", I(32)), P("eps", R(1e-6)), P("tap", N("features")), P("taps", I(3))}, {}, {},
      Seg(0, 3, 0, 256, 3), {7, 1, 3}, {}, false, planner::Phase::kDraft);

  // 9. embedding.mtp v1
  add("embedding.mtp", 1, Site::kEmbedding,
      {.width = 64, .context = 4096, .streams = 4, .folds = GdnFolds()},
      {P("eps", R(1e-6)), P("tap", N("streams"))}, {}, {9}, Seg(0, 3, 0, 256, 3), {4, 0, 9});

  // 11. embedding.projected v1
  add("embedding.projected", 1, Site::kEmbedding, {.width = 64, .context = 4096},
      {P("feature", I(64)), P("scale", R(8.0)), P("tap", N("features"))}, {}, {16},
      Seg(0, 3, 0, 256, 3), {4, 0, 9});

  // 12. embedding.token v1
  add("embedding.token", 1, Site::kEmbedding, {.width = 64, .context = 4096}, {P("scale", R(8.0))},
      {}, {}, Seg(0, 3, 0, 256, 3), {4, 0, 9});

  // 13. ffn.glu v1
  add("ffn.glu", 1, Site::kLayer, {.width = 64, .context = 4096},
      {P("activation", N("silu")), P("eps", R(1e-6)), P("hidden", I(96)),
       P("placement", N("sandwich"))},
      {}, {}, Seg(0, 3, 0, 256, 3));

  // 14. ffn.moe v1
  add("ffn.moe", 1, Site::kLayer, {.width = 64, .context = 4096},
      {P("activation", N("gelu_tanh")), P("eps", R(1e-6)), P("expert_hidden", I(32)),
       P("expert_scales", B(true)), P("experts", I(8)), P("hidden", I(64)), P("normalize", B(true)),
       P("placement", N("parallel")), P("router", N("softmax_topk")),
       P("router_input", N("rms_scaled")), P("stacked", B(true)), P("used", I(3))},
      {}, {}, Seg(0, 3, 0, 256, 3), {}, {{12, {0.5F, 0.5F}}});

  // 15. hc.combine v1
  add("hc.combine", 1, Site::kLayer,
      {.width = 32, .context = 4096, .streams = 4, .folds = GdnFolds()}, {},
      {.residual = true, .carry = 128}, {}, Seg(0, 3, 0, 256, 3));

  // 16. hc.mix v1
  add("hc.mix", 1, Site::kLayer, {.width = 32, .context = 4096, .streams = 4, .folds = GdnFolds()},
      {P("eps", R(1e-6)), P("rank", I(8))}, {.residual = true}, {}, Seg(0, 3, 0, 256, 3));

  // 17. head.linear v1
  add("head.linear", 1, Site::kOutput, {.width = 64, .context = 4096}, {}, {}, {},
      Seg(0, 3, 0, 256, 1));

  // 18. norm.adaptive v1
  add("norm.adaptive", 1, Site::kLayer, {.width = 64, .context = 256},
      {P("eps", R(1e-6)), P("side", N("scale"))}, {.side = 64}, {}, Seg(0, 3, 0, 256, 3));

  // 19. norm.channel v1
  add("norm.channel", 1, Site::kLayer, {.width = 64, .context = 256}, {P("silu", B(false))}, {}, {},
      Seg(0, 3, 0, 256, 3));

  // 20. norm.rms v1
  add("norm.rms", 1, Site::kFinalNorm, {.width = 64, .context = 4096}, {P("eps", R(1e-6))}, {}, {},
      Seg(0, 3, 0, 256, 3));

  // 21. projection.spatial v1
  add("projection.spatial", 1, Site::kLayer, {.width = 64, .context = 256},
      {P("in_channels", I(64)), P("out_channels", I(64)), P("silu", B(true))}, {}, {},
      Seg(0, 3, 0, 256, 3));

  // 22. residual.gated v1
  add("residual.gated", 1, Site::kLayer, {.width = 64, .context = 256}, {P("side", N("gate"))},
      {.carry = 64, .side = 64}, {}, Seg(0, 3, 0, 256, 3));

  // 23. residual.hc v1
  add("residual.hc", 1, Site::kMixing,
      {.width = 32, .context = 4096, .streams = 4, .folds = GdnFolds()}, {P("mix", N("sigmoid"))},
      {}, {}, Seg(0, 3, 0, 256, 3));

  // 24. residual.ple v1
  add("residual.ple", 1, Site::kLayer,
      {.width = 32, .context = 4096, .streams = 4, .folds = GdnFolds()},
      {P("conv", I(4)), P("dilation", I(3)), P("eps", R(1e-6)), P("input", N("ngram"))},
      {.residual = true, .side = 64}, {1}, Seg(0, 3, 0, 256, 3), {}, {{5, {-0.6F, 0.6F}}}, false,
      planner::Phase::kPrefill, 64);

  // 25. residual.scaled v1
  add("residual.scaled", 1, Site::kLayer, {.width = 64, .context = 256}, {},
      {.carry = 64, .side = 1}, {}, Seg(0, 3, 0, 256, 3));

  // 26. residual.spatial v1
  add("residual.spatial", 1, Site::kLayer, {.width = 64, .context = 256}, {}, {.carry = 64}, {},
      Seg(0, 3, 0, 256, 3));

  // 27. scale.learned v1
  add("scale.learned", 1, Site::kLayer, {.width = 64, .context = 4096}, {}, {}, {},
      Seg(0, 3, 0, 256, 3));

  // 28. side.hash v1
  add("side.hash", 1, Site::kSide, {.width = 64, .context = 4096},
      {P("experts", I(8)), P("used", I(3))}, {}, {}, Seg(0, 3, 0, 256, 0), {}, {}, false,
      planner::Phase::kPrefill, 12);

  // 29. side.ngram v1
  add("side.ngram", 1, Site::kSide,
      {.width = 32, .context = 4096, .streams = 4, .folds = GdnFolds()},
      {P("eos", I(0)), P("heads_per_ngram", I(1)), P("multipliers", L({3, 5, 7})), P("ngram", I(3)),
       P("offsets", L({0, 40})), P("row", I(32)), P("vocab", L({40, 40}))},
      {}, {}, Seg(0, 3, 0, 256, 3), {}, {}, false, planner::Phase::kPrefill, 80);

  // 30. slice.spatial v1
  add("slice.spatial", 1, Site::kLayer, {.width = 64, .context = 256},
      {P("input_rows", I(3)), P("rows", I(2)), P("start", I(1))}, {}, {}, Seg(0, 3, 0, 256, 3));

  return recipes;
}

inline std::vector<CatalogSpatialRecipe> CatalogSpatialRecipes() {
  std::vector<CatalogSpatialRecipe> recipes;

  // 31. affine.channel v1
  recipes.push_back({
      .kind = "affine.channel",
      .version = 1,
      .cases = {CatalogSpatialCase{
          .label = "f32",
          .ref = MakeRef("affine.channel", 1, {}),
          .owner = SpatialOwner{0, {2, 2, 2, 1}},
          .inputs = {SpatialInput{0, 0, {1, 2, 2, 1}}},
          .links = {SpatialLink{.input = 0, .auxiliary_port = 0}},
          .tensors = {},
      }},
  });

  // 32. attention.spatial v1
  recipes.push_back({
      .kind = "attention.spatial",
      .version = 1,
      .cases = {CatalogSpatialCase{
          .label = "f32",
          .ref = MakeRef("attention.spatial", 1, {P("heads", I(1)), P("kv_heads", I(1))}),
          .owner = SpatialOwner{0, {2, 2, 192, 1}},
          .inputs = {},
          .links = {},
          .tensors = {},
      }},
  });

  // 33. conv.spatial v1
  recipes.push_back({
      .kind = "conv.spatial",
      .version = 1,
      .cases = {CatalogSpatialCase{
          .label = "f32",
          .ref =
              MakeRef("conv.spatial", 1,
                      {P("in_channels", I(2)), P("kernel", I(3)), P("out_channels", I(3)),
                       P("padding", I(1))},
                      {{.role = "kernel", .tensor = "kernel"}, {.role = "bias", .tensor = "bias"}}),
          .owner = SpatialOwner{0, {2, 2, 2, 1}},
          .inputs = {},
          .links = {},
          .tensors = {BoundTensor{"kernel", ir::Dense(Encoding::kF32, ir::ShapeOf({3, 3, 2, 3}))},
                      BoundTensor{"bias", ir::Dense(Encoding::kF32, ir::ShapeOf({3}))}},
      }},
  });

  // 34. reshape.spatial v1
  recipes.push_back({
      .kind = "reshape.spatial",
      .version = 1,
      .cases = {CatalogSpatialCase{
          .label = "f32",
          .ref = MakeRef("reshape.spatial", 1, {P("height", I(2)), P("width", I(2))}),
          .owner = SpatialOwner{0, {1, 4, 2, 1}},
          .inputs = {},
          .links = {},
          .tensors = {},
      }},
  });

  // 35. residual.upsample v1
  recipes.push_back({
      .kind = "residual.upsample",
      .version = 1,
      .cases = {CatalogSpatialCase{
          .label = "f32",
          .ref = MakeRef("residual.upsample", 1, {P("factor_t", I(1))}),
          .owner = SpatialOwner{0, {4, 4, 2, 1}},
          .inputs = {SpatialInput{0, 0, {2, 2, 8, 1}}},
          .links = {SpatialLink{.input = 0, .auxiliary_port = 0}},
          .tensors = {},
      }},
  });

  // 36. upsample.nearest v1
  recipes.push_back({
      .kind = "upsample.nearest",
      .version = 1,
      .cases = {CatalogSpatialCase{
          .label = "f32",
          .ref = MakeRef("upsample.nearest", 1, {P("factor", I(2))}),
          .owner = SpatialOwner{0, {2, 2, 2, 1}},
          .inputs = {},
          .links = {},
          .tensors = {},
      }},
  });

  return recipes;
}

inline void RunCatalogTokenCpuCase(const CatalogTokenCase& tc) {
  auto in = MakeInstance(tc.ref, tc.stage, tc.site, 0, 1);
  ASSERT_TRUE(in.has_value()) << tc.ref.kind << ": " << in.error();
  const auto shapes = in->def->shapes(*in);
  ASSERT_TRUE(shapes.has_value()) << tc.ref.kind << ": " << shapes.error();

  std::mt19937 rng(77);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  RefIO io{.instance = &*in};
  io.weights.resize(in->def->roles.size());
  io.weight_shapes.resize(in->def->roles.size());
  std::vector<std::vector<std::byte>> encoded(in->def->roles.size());
  for (const RoleShape& r : *shapes) {
    const auto& f = tc.formats[r.role];
    std::vector<float> v(ir::Elements(f.shape));
    const bool norm = f.shape.rank == 1;
    const auto range = tc.ranges.find(r.role);
    for (float& x : v) {
      x = range != tc.ranges.end() ? range->second.first + (range->second.second * (1.0F + u(rng)))
          : norm                   ? 1.0F + (0.25F * u(rng))
                                   : 0.3F * u(rng);
    }
    const auto parts = format::PartsOf(f);
    ASSERT_TRUE(parts.has_value());
    encoded[r.role].resize((*parts)[0].bytes);
    if (f.encoding == Encoding::kI32) {
      io.weights[r.role].resize(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) {
        const auto id = static_cast<std::int32_t>((i * 5) % 8);
        std::memcpy(encoded[r.role].data() + (i * 4), &id, 4);
        io.weights[r.role][i] = static_cast<float>(id);
      }
      io.weight_shapes[r.role] = f.shape;
      continue;
    }
    const std::uint64_t matrices = format::Matrices(f);
    const std::size_t per = v.size() / matrices;
    io.weights[r.role].resize(v.size());
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::EncodeRows(f, m, 0, std::span(v).subspan(m * per, per), encoded[r.role]));
    }
    const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[r.role]};
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::DecodeRows(f, std::span(&part, 1), m, 0,
                                  f.shape.ne[1] * f.shape.ne[2] / matrices,
                                  std::span(io.weights[r.role]).subspan(m * per, per)));
    }
    io.weight_shapes[r.role] = f.shape;
  }

  const model::StateContract contract =
      in->def->state != nullptr ? in->def->state(*in) : model::StateContract{};
  io.state_cells.assign(tc.cells.begin(), tc.cells.end());

  auto graph = BuildComponent(*in, tc.shape, tc.formats, tc.cells, tc.inputs);
  ASSERT_TRUE(graph.has_value()) << tc.ref.kind << " (" << tc.label << "): " << graph.error();
  auto exec = ref::Executor::Create(graph->step.graph);
  ASSERT_TRUE(exec.has_value()) << tc.ref.kind << " (" << tc.label << "): " << exec.error().detail;

  for (const RoleShape& r : *shapes) {
    const ir::TensorId leaf = graph->weights[r.role];
    if (!leaf.valid()) continue;
    const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[r.role]};
    ASSERT_TRUE(exec->Bind(leaf, std::span(&part, 1)));
  }

  std::map<std::uint8_t, std::vector<std::vector<double>>> ref_state;
  for (const StepState& st : graph->step.state) {
    const std::uint8_t slot = tc.shape.segments.at(st.ref.segment).slot;
    auto& start = ref_state[slot];
    if (start.empty()) start.resize(contract.tensors.size());
    std::vector<float> values(ir::Elements(graph->step.graph.tensor(st.tensor).shape()), 0.0F);
    if (tc.random_state) {
      for (float& x : values) x = 0.25F * u(rng);
    }
    ASSERT_TRUE(exec->BindValues(st.tensor, values));
    auto held = exec->Values(st.tensor);
    ASSERT_TRUE(held.has_value());
    start[st.ref.tensor].assign(held->begin(), held->end());
  }
  for (const planner::Segment& s : tc.shape.segments) {
    io.segments.push_back(
        {.rows = s.rows, .past = s.past, .read_cells = s.read_cells, .outputs = s.outputs});
    auto& st = ref_state[s.slot];
    if (st.empty()) st.resize(contract.tensors.size());
    io.state.push_back(st);
  }

  for (const StepInput& input : graph->step.inputs) {
    const auto random = [&](std::vector<double>& into, float scale) {
      std::vector<float> f(ir::Elements(graph->step.graph.tensor(input.tensor).shape()));
      for (float& x : f) x = scale * u(rng);
      EXPECT_TRUE(exec->BindValues(input.tensor, f));
      into.assign(f.begin(), f.end());
    };
    switch (input.kind) {
      case InputKind::kStream:
        random(io.stream, 1.0F);
        continue;
      case InputKind::kResidual:
        random(io.residual, 1.0F);
        continue;
      case InputKind::kCarry:
        random(input.tensor == graph->carry2_in ? io.carry2 : io.carry, 1.0F);
        continue;
      case InputKind::kSide:
        random(io.side, 1.0F);
        continue;
      case InputKind::kFeatures:
        random(io.features, 1.0F);
        continue;
      case InputKind::kSideRows: {
        std::uniform_int_distribution<std::int64_t> pick(
            0, static_cast<std::int64_t>(tc.table_rows) - 1);
        std::vector<std::int64_t> rows(
            ir::Elements(graph->step.graph.tensor(input.tensor).shape()));
        for (auto& r : rows) r = pick(rng);
        ASSERT_TRUE(exec->BindInts(input.tensor, rows));
        io.side_rows = rows;
        continue;
      }
      default:
        break;
    }
    auto values = InputValues(input, tc.shape, tc.tokens);
    ASSERT_TRUE(values.has_value()) << values.error();
    ASSERT_TRUE(exec->BindInts(input.tensor, *values));
  }

  auto ran = exec->Run();
  ASSERT_TRUE(ran.has_value()) << tc.ref.kind << " (" << tc.label << "): " << ran.error().detail;

  io.tokens = tc.tokens;
  io.greedy = graph->step.greedy.valid();
  ASSERT_NE(in->def->reference, nullptr) << tc.ref.kind;
  auto r = in->def->reference(io);
  ASSERT_TRUE(r.has_value()) << tc.ref.kind << " (" << tc.label << "): " << r.error();

  const auto check = [&](ir::TensorId t, std::span<const double> want, std::string_view what) {
    if (!t.valid()) return;
    auto got = exec->Values(t);
    ASSERT_TRUE(got.has_value());
    EXPECT_LE(Gap(*got, want), tc.tolerance) << tc.ref.kind << " (" << tc.label << ") " << what;
  };
  if (graph->step.logits.valid()) {
    check(graph->step.logits, io.logits, "logits");
  } else if (tc.ref.kind == "side.hash") {
    auto got = exec->Ints(graph->stream_out);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, io.side_ids) << tc.ref.kind;
  } else {
    check(graph->stream_out, io.stream, "stream");
    check(graph->residual_out, io.residual, "residual");
    check(graph->carry_out, io.carry, "carry");
    check(graph->carry2_out, io.carry2, "carry2");
  }
  check(graph->step.features, io.features_out, "features");
  for (const StepState& st : graph->step.state) {
    if (!st.written) continue;
    auto values = exec->Values(st.tensor);
    ASSERT_TRUE(values.has_value());
    EXPECT_LE(Gap(*values, io.state[st.ref.segment][st.ref.tensor]), tc.state_tolerance)
        << tc.ref.kind << " (" << tc.label << ") state " << st.ref.tensor;
  }
}

inline spec::ModelSpec MakeSingleSpatialSpec(const CatalogSpatialCase& sc) {
  spec::Stage stage{.name = "vae", .role = spec::StageRole::kVae, .artifact = 0};
  stage.stream.width = sc.owner.shape.channels;
  spec::Layer layer;
  layer.components.push_back(sc.ref);
  stage.layers.push_back(std::move(layer));
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {std::move(stage)}};
  spec::Canonicalize(model);
  return model;
}

inline SpatialPayment UnboundedSpatialPayment() {
  return {[](std::uint64_t) { return true; }, [](std::uint64_t) {}};
}

inline void RunCatalogSpatialCpuCase(const CatalogSpatialCase& sc) {
  const spec::ModelSpec model = MakeSingleSpatialSpec(sc);
  const std::array<std::span<const BoundTensor>, 1> tables{sc.tensors};
  const std::array<SpatialOwner, 1> owners{sc.owner};
  auto graph = BuildSpatialStep(model, tables, owners, UnboundedSpatialPayment(),
                                {.links = sc.links, .inputs = sc.inputs});
  ASSERT_TRUE(graph.has_value()) << sc.ref.kind << " (" << sc.label << "): " << graph.error();

  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec.has_value()) << sc.ref.kind << " (" << sc.label << "): " << exec.error().detail;

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> u(-0.6F, 0.6F);

  std::vector<std::vector<float>> weights(graph->weights().size());
  std::array<std::span<const float>, 2> roles{};
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const auto& w = graph->weights()[i];
    const auto count = ir::Elements(graph->graph().tensor(w.tensor).shape());
    weights[i].resize(static_cast<std::size_t>(count));
    for (float& x : weights[i]) x = u(rng);
    ASSERT_TRUE(exec->BindValues(w.tensor, weights[i]));
    roles.at(w.ref.role) = weights[i];
  }

  const auto in_count = ir::Elements(graph->graph().tensor(graph->graph().inputs()[0]).shape());
  std::vector<float> input_f(static_cast<std::size_t>(in_count));
  for (float& x : input_f) x = u(rng);
  ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[0], input_f));
  std::vector<double> input_d(input_f.begin(), input_f.end());

  std::vector<double> aux_d;
  std::optional<SpatialShape> aux_shape;
  if (!sc.inputs.empty()) {
    ASSERT_GE(graph->graph().inputs().size(), 2U);
    const auto aux_id = graph->graph().inputs()[1];
    const auto aux_count = ir::Elements(graph->graph().tensor(aux_id).shape());
    std::vector<float> aux_f(static_cast<std::size_t>(aux_count), 0.0F);
    if (sc.ref.kind != "attention.spatial") {
      for (float& x : aux_f) x = u(rng);
    }
    ASSERT_TRUE(exec->BindValues(aux_id, aux_f));
    aux_d.assign(aux_f.begin(), aux_f.end());
    aux_shape = sc.inputs[0].shape;
  }

  auto ran = exec->Run();
  ASSERT_TRUE(ran.has_value()) << sc.ref.kind << " (" << sc.label << "): " << ran.error().detail;

  auto instances = InstancesOf(model, 0);
  ASSERT_TRUE(instances.has_value()) << instances.error();
  const Instance& instance = instances->layers[0][0];
  ASSERT_NE(instance.def->spatial, nullptr);
  ASSERT_NE(instance.def->spatial->reference, nullptr);

  SpatialRefIO oracle{
      .instance = instance,
      .shape = sc.owner.shape,
      .input = input_d,
      .kernel = roles[0],
      .bias = roles[1],
      .output = {},
      .auxiliary = aux_d,
      .auxiliary_shape = aux_shape,
  };
  auto ref_st = RunSpatialReference(oracle);
  ASSERT_TRUE(ref_st.has_value()) << sc.ref.kind << " (" << sc.label << "): " << ref_st.error();

  auto actual = exec->Values(graph->graph().outputs()[0]);
  ASSERT_TRUE(actual.has_value());
  EXPECT_LE(Gap(*actual, oracle.output), sc.tolerance) << sc.ref.kind << " (" << sc.label << ")";
}

}  // namespace llmp::components::test_catalog

#endif  // LLMP_TESTS_UNIT_COMPONENT_CATALOG_CASES_H_
