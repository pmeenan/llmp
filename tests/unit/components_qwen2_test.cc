// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// G5's specs and arithmetic shapes, without loading model payloads. These
// are structural controls, not the FP16 bridge or EXL3 Tier C model gate.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "components/step.h"
#include "execution/registry.h"
#include "families/qwen.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/lower.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen2.h"

namespace {
namespace co = llmp::components;
namespace sp = llmp::spec;
namespace md = llmp::model;
namespace fam = llmp::test_support::families;
namespace ir = llmp::ir;
namespace fmt = llmp::format;
using fmt::Encoding;

sp::Source Source(bool exl3) {
  const auto& p = exl3 ? fam::Qwen25Instruct05BExl3() : fam::Qwen25Instruct05B();
  sp::Source source{.architecture = "qwen2", .container = exl3 ? "exl3" : "gguf"};
  const auto add = [&](std::string name, Encoding encoding, fmt::Shape shape) {
    source.tensors.push_back({.name = std::move(name), .format = ir::Dense(encoding, shape)});
  };
  const auto linear = [&](std::string name, std::uint64_t k, std::uint64_t n, bool head = false) {
    if (!exl3) {
      add(name + ".weight", Encoding::kF16, ir::ShapeOf({k, n}));
      return;
    }
    fmt::TensorFormat format{.encoding = Encoding::kExl3Trellis,
                             .params = {.k_bits = static_cast<std::uint8_t>(head ? 8 : 4),
                                        .codebook = fmt::Exl3Codebook::kMcg},
                             .layout = fmt::Layout::kExl3Tiles,
                             .shape = ir::ShapeOf({k, n})};
    for (const auto& [suffix, role] :
         {std::pair{"trellis", fmt::PartRole::kCodes}, std::pair{"suh", fmt::PartRole::kInputSigns},
          std::pair{"svh", fmt::PartRole::kOutputSigns},
          std::pair{"mcg", fmt::PartRole::kCodebookMarker}}) {
      source.tensors.push_back({.name = name + "." + suffix, .format = format, .part = role});
    }
  };
  const auto norm = exl3 ? Encoding::kBF16 : Encoding::kF32;
  add(exl3 ? "model.embed_tokens.weight" : "token_embd.weight",
      exl3 ? Encoding::kBF16 : Encoding::kF16, ir::ShapeOf({p.width, p.vocab}));
  add(exl3 ? "model.norm.weight" : "output_norm.weight", norm, ir::ShapeOf({p.width}));
  if (exl3) linear("lm_head", p.width, p.vocab, true);
  for (std::uint32_t layer = 0; layer < p.layers; ++layer) {
    const auto b = exl3 ? std::format("model.layers.{}.", layer) : std::format("blk.{}.", layer);
    add(b + (exl3 ? "input_layernorm.weight" : "attn_norm.weight"), norm, ir::ShapeOf({p.width}));
    add(b + (exl3 ? "post_attention_layernorm.weight" : "ffn_norm.weight"), norm,
        ir::ShapeOf({p.width}));
    const std::array<const char*, 3> names =
        exl3 ? std::array{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
             : std::array{"attn_q", "attn_k", "attn_v"};
    for (std::size_t i = 0; i < names.size(); ++i) {
      const auto n = i == 0 ? p.width : p.kv_width();
      linear(b + names[i], p.width, n);
      add(b + names[i] + ".bias", exl3 ? Encoding::kF16 : Encoding::kF32, ir::ShapeOf({n}));
    }
    linear(b + (exl3 ? "self_attn.o_proj" : "attn_output"), p.width, p.width);
    linear(b + (exl3 ? "mlp.gate_proj" : "ffn_gate"), p.width, p.ffn);
    linear(b + (exl3 ? "mlp.up_proj" : "ffn_up"), p.width, p.ffn);
    linear(b + (exl3 ? "mlp.down_proj" : "ffn_down"), p.ffn, p.width);
  }
  return source;
}

TEST(ComponentsQwen2, BothEncodingsUseTheSharedCatalogAndPhysicalShapes) {
  const auto& mapping = sp::mappings::Qwen2Mapping();
  ASSERT_TRUE(sp::ValidateMapping(mapping));
  EXPECT_EQ(sp::FindMapping("qwen2"), &mapping);
  for (const bool exl3 : {false, true}) {
    SCOPED_TRACE(exl3);
    const auto& p = exl3 ? fam::Qwen25Instruct05BExl3() : fam::Qwen25Instruct05B();
    auto values = fam::Qwen2Values(p);
    ASSERT_TRUE(values.has_value()) << values.error();
    const auto source = Source(exl3);
    auto spec = sp::Synthesize(mapping, source, *values);
    ASSERT_TRUE(spec.has_value()) << spec.error();
    const std::array<std::span<const co::BoundTensor>, 1> artifacts{source.tensors};
    ASSERT_TRUE(co::ValidateSpec(*spec, artifacts));
    ASSERT_EQ(spec->stages[0].layers.size(), 24U);
    for (const auto& layer : spec->stages[0].layers) {
      ASSERT_EQ(layer.components.size(), 2U);
      EXPECT_EQ(layer.components[0].kind, "attention.dense");
      EXPECT_EQ(layer.components[1].kind, "ffn.glu");
    }
    auto layout = co::StageLayout(*spec, 0, 2048, 1024);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    ASSERT_EQ(layout->items.size(), 48U);
    for (const auto& item : layout->items) {
      EXPECT_EQ(item.contract.encoding, Encoding::kF16);
      EXPECT_EQ(item.contract.cell_bytes, 256U);
      EXPECT_EQ(item.cells, 2048U);
      EXPECT_EQ(item.contract.rollback, md::Rollback::kTruncate);
    }
    const co::Bindings bindings{.tensors = source.tensors, .state = &*layout};
    for (const std::uint32_t rows : {1U, 32U, 145U, 1023U, 1024U}) {
      SCOPED_TRACE(rows);
      const auto step = co::BuildStep(*spec, 0,
                                      {.segments = {{.slot = 0,
                                                     .rows = rows,
                                                     .past = 0,
                                                     .read_cells = (rows + 255U) / 256U * 256U,
                                                     .outputs = rows}}},
                                      bindings);
      ASSERT_TRUE(step.has_value()) << step.error();
      std::size_t half_products = 0, attention = 0, softmax = 0, norm_scales = 0;
      for (const auto& node : step->graph.nodes()) {
        if (node.key.op == ir::Op::kConvert &&
            step->graph.tensor(node.src[0]).format.encoding == Encoding::kBF16 &&
            step->graph.tensor(node.out[0]).format.encoding == Encoding::kF32) {
          ++norm_scales;
        }
        if (node.key.op == ir::Op::kMatMul &&
            step->graph.tensor(node.out[0]).format.encoding == Encoding::kF16) {
          ++half_products;
          EXPECT_EQ(step->graph.tensor(node.src[1]).format.encoding, Encoding::kF16);
        }
        if (node.key.op == ir::Op::kAttention) {
          ++attention;
          EXPECT_EQ(node.Params<ir::AttnParams>().scale, 0.125F);
        }
        if (node.key.op == ir::Op::kSoftMax) {
          ++softmax;
          EXPECT_EQ(node.Params<ir::SoftMaxParams>().scale, 0.125F);
        }
      }
      EXPECT_EQ(norm_scales, exl3 ? 49U : 0U);    // two norms per layer and final norm
      EXPECT_EQ(half_products, exl3 ? 73U : 0U);  // Q,K,V a layer and the head
      EXPECT_EQ(attention, 24U);
      EXPECT_EQ(softmax, 0U);
      EXPECT_EQ(step->graph.tensor(step->logits).format.encoding, Encoding::kF32);
    }
  }
}

TEST(ComponentsQwen2, RefusesMissingBiasAndTrellisParts) {
  auto source = Source(true);
  const auto values = fam::Qwen2Values(fam::Qwen25Instruct05BExl3());
  ASSERT_TRUE(values.has_value());
  const auto& mapping = sp::mappings::Qwen2Mapping();
  const auto intact = sp::Synthesize(mapping, source, *values);
  ASSERT_TRUE(intact) << (intact ? "" : intact.error());
  for (const auto purpose : {co::ValidationPurpose::kExecution, co::ValidationPurpose::kImport}) {
    const auto valid = co::ValidateSpec(*intact, source.tensors, purpose);
    ASSERT_TRUE(valid) << (valid ? "" : valid.error());
  }
  for (const std::string missing :
       {"model.layers.0.self_attn.q_proj.bias", "model.layers.0.self_attn.q_proj.suh"}) {
    auto damaged = source;
    std::erase_if(damaged.tensors, [&](const auto& t) { return t.name == missing; });
    EXPECT_FALSE(sp::Synthesize(mapping, damaged, *values)) << missing;
  }
  // Marker bindings are optional in the mapping because 3Inst has none.
  // The actual Mcg format still requires its marker at component validation.
  std::erase_if(source.tensors,
                [](const auto& t) { return t.name == "model.layers.0.self_attn.q_proj.mcg"; });
  const auto spec = sp::Synthesize(mapping, source, *values);
  ASSERT_TRUE(spec) << (spec ? "" : spec.error());
  for (const auto purpose : {co::ValidationPurpose::kExecution, co::ValidationPurpose::kImport}) {
    const auto valid = co::ValidateSpec(*spec, source.tensors, purpose);
    ASSERT_FALSE(valid);
    EXPECT_NE(valid.error().find("binds 3 parts of 4"), std::string::npos) << valid.error();
  }
}
TEST(ComponentsQwen2, Fp16ActualShapesSelectAndLowerThroughTheSharedGgmlBackend) {
  namespace kg = llmp::kernels::ggml;
  namespace ex = llmp::execution;
  namespace pl = llmp::planner;
  std::vector<ex::Implementation> implementations;
  for (const auto& d : kg::CapabilityDeclarations()) {
    implementations.push_back({.name = std::string(d.name),
                               .operation = d.operation,
                               .source = "ggml",
                               .revision = "planning",
                               .build = "cpu",
                               .variant = std::string(kg::AcceptsTextOf(d.name)),
                               .capability = *d.capability});
  }
  auto registry = ex::Registry::Create(std::move(implementations));
  ASSERT_TRUE(registry.has_value());
  const auto source = Source(false);
  const auto& mapping = sp::mappings::Qwen2Mapping();
  const auto values = fam::Qwen2Values(fam::Qwen25Instruct05B());
  ASSERT_TRUE(values.has_value());
  const auto spec = sp::Synthesize(mapping, source, *values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const auto state = co::StageLayout(*spec, 0, 2048, 1024);
  ASSERT_TRUE(state.has_value()) << state.error();
  const auto policy = pl::ParsePolicy(mapping.policies.front().text);
  ASSERT_TRUE(policy.has_value()) << policy.error();
  kg::DeviceChoices choices;
  choices.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  const llmp::kernels::Addresses addresses{};
  for (const auto rows : {1U, 32U, 145U}) {
    SCOPED_TRACE(rows);
    const pl::StepShape shape{
        .segments = {{.slot = 0, .rows = rows, .past = 0, .read_cells = 256, .outputs = rows}}};
    auto step = co::BuildStep(*spec, 0, shape, {.tensors = source.tensors, .state = &*state});
    ASSERT_TRUE(step.has_value()) << step.error();
    std::vector<ir::TensorId> staged;
    for (const auto& input : step->inputs) {
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
    }
    auto plan =
        pl::Plan(step->graph.Clone(), *registry, {.device_class = "sm_121", .arch = 121}, *policy,
                 {.staged = staged, .written = &kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    const llmp::kernels::LowerInput input{*plan, *step, shape, addresses, 2048};
    for (const bool masks : {false, true}) {
      auto lowered = kg::LowerStep(input, *registry, choices, {.device_masks = masks});
      ASSERT_TRUE(lowered.has_value()) << lowered.error().detail;
      EXPECT_FALSE(lowered->plan.steps.empty());
    }
  }
}

TEST(ComponentsQwen2, Fp16FloatFusionsAreSelectedOnlyForOneColumn) {
  namespace kg = llmp::kernels::ggml;
  namespace ex = llmp::execution;
  namespace pl = llmp::planner;
  std::vector<ex::Implementation> implementations;
  for (const auto& d : kg::CapabilityDeclarations()) {
    implementations.push_back({.name = std::string(d.name),
                               .operation = d.operation,
                               .source = "ggml",
                               .revision = "planning",
                               .build = "cpu",
                               .variant = std::string(kg::AcceptsTextOf(d.name)),
                               .capability = *d.capability});
  }
  auto registry = ex::Registry::Create(std::move(implementations));
  ASSERT_TRUE(registry.has_value());
  const auto source = Source(false);
  const auto& mapping = sp::mappings::Qwen2Mapping();
  const auto values = fam::Qwen2Values(fam::Qwen25Instruct05B());
  ASSERT_TRUE(values.has_value());
  const auto spec = sp::Synthesize(mapping, source, *values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const auto state = co::StageLayout(*spec, 0, 2048, 1024);
  ASSERT_TRUE(state.has_value()) << state.error();
  auto policy = pl::ParsePolicy(mapping.policies.front().text);
  ASSERT_TRUE(policy.has_value()) << policy.error();
  policy->enabled = {"float_mul_mat_add", "float_mul_mat_glu"};
  pl::Canonicalize(*policy);
  kg::DeviceChoices choices;
  choices.mul_mat =
      [](const ggml_tensor* tensor) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return tensor->ne[1] == 1 ? kg::MulMatPath::kVector : kg::MulMatPath::kCublas;
  };
  choices.vector_fusible = [](const ggml_tensor* tensor) { return tensor->ne[1] == 1; };
  const llmp::kernels::Addresses addresses{};
  for (const auto rows : {1U, 32U}) {
    SCOPED_TRACE(rows);
    const pl::StepShape shape{
        .segments = {{.slot = 0, .rows = rows, .past = 0, .read_cells = 256, .outputs = rows}}};
    auto step = co::BuildStep(*spec, 0, shape, {.tensors = source.tensors, .state = &*state});
    ASSERT_TRUE(step.has_value()) << step.error();
    std::vector<ir::TensorId> staged;
    for (const auto& input : step->inputs) {
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
    }
    auto plan =
        pl::Plan(step->graph.Clone(), *registry, {.device_class = "sm_121", .arch = 121}, *policy,
                 {.staged = staged, .written = &kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    const llmp::kernels::LowerInput input{*plan, *step, shape, addresses, 2048};
    for (const bool masks : {false, true}) {
      auto lowered = kg::LowerStep(input, *registry, choices, {.device_masks = masks});
      ASSERT_TRUE(lowered.has_value()) << lowered.error().detail;
      std::size_t adds = 0, glus = 0;
      for (const auto& selected : lowered->plan.steps) {
        adds += selected.implementation == "ggml.mul_mat_add.mmvf_fused";
        glus += selected.implementation == "ggml.mul_mat_glu.mmvf_fused";
      }
      EXPECT_EQ(adds, rows == 1 ? 119U : 0U);
      EXPECT_EQ(glus, rows == 1 ? 24U : 0U);
      if (rows == 1 && masks) {
        auto no_predicate = choices;
        no_predicate.vector_fusible = {};
        const auto refused = kg::LowerStep(input, *registry, no_predicate, {.device_masks = masks});
        ASSERT_FALSE(refused.has_value());
        EXPECT_NE(refused.error().detail.find("needs the device MMVF predicate"),
                  std::string::npos);
        auto row_invariant = choices;
        row_invariant.row_invariant = true;
        const auto invariant =
            kg::LowerStep(input, *registry, row_invariant, {.device_masks = masks});
        ASSERT_FALSE(invariant.has_value());
        EXPECT_NE(invariant.error().detail.find("row-invariant"), std::string::npos);
      }
    }
  }
}

}  // namespace
