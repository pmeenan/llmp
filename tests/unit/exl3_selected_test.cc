// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Descriptor-only controls; no payload allocation and no device work.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <thread>
#include <vector>

#include "components/step.h"
#include "engine/component_source.h"
#include "expected_error.h"
#include "families/qwen.h"
#include "ir/graph.h"
#include "kernels/decode/selected_source.h"
#include "kernels/exl3/reconstruction_tuning.h"
#include "kernels/exl3/selected_source.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/step_program.h"
#include "kernels/ggml/tensors.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_runtime.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen2.h"

namespace {
namespace ex = llmp::kernels::exl3;
namespace kg = llmp::kernels::ggml;
namespace ir = llmp::ir;
namespace fmt = llmp::format;
using fmt::Encoding;

namespace sp = llmp::spec;
namespace fam = llmp::test_support::families;
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

struct Product {
  std::array<kg::SelectedOperand, 2> inputs;
  kg::SelectedOperand output;
  ir::Node node;
};
Product Make(ggml_context* context, int rows, int k, int n, Encoding output, int bits = 4) {
  Product result;
  auto& weight = result.inputs[0];
  weight.desc.format = {
      .encoding = Encoding::kExl3Trellis,
      .params = {.k_bits = static_cast<std::uint8_t>(bits), .codebook = fmt::Exl3Codebook::kMcg},
      .layout = fmt::Layout::kExl3Tiles,
      .shape = ir::ShapeOf({static_cast<unsigned>(k), static_cast<unsigned>(n)})};
  const auto extents = fmt::PartsOf(weight.desc.format);
  for (const auto& extent : extents->parts()) {
    auto* tensor =
        ggml_new_tensor_1d(context, GGML_TYPE_I8, static_cast<std::int64_t>(extent.readable));
    weight.parts.push_back({extent.role, tensor});
  }
  weight.tensor = weight.parts[0].tensor;
  const auto dense = [&](int width, Encoding encoding) {
    kg::SelectedOperand operand;
    operand.desc.format = ir::Dense(
        encoding, ir::ShapeOf({static_cast<unsigned>(width), static_cast<unsigned>(rows)}));
    operand.desc.nb = ir::NaturalStrides(operand.desc.format);
    operand.tensor = ggml_new_tensor_2d(
        context, encoding == Encoding::kF16 ? GGML_TYPE_F16 : GGML_TYPE_F32, width, rows);
    return operand;
  };
  result.inputs[1] = dense(k, Encoding::kF16);
  result.output = dense(n, output);
  result.node.key = ir::Primitive(ir::Op::kMatMul);
  result.node.nsrc = 2;
  result.node.nout = 1;
  result.node.SetParams(ir::ProductParams{});
  return result;
}

TEST(Exl3SelectedTest, DefaultsCoverNonfixtureWidthsAndDropMissingCapacityOrPins) {
  for (const auto arch : {86U, 121U}) {
    ex::PackedCapacity capacity{.arch = arch, .sms = arch == 86 ? 84 : 48};
    for (auto* table : {&capacity.gemm, &capacity.multi})
      for (auto& output : *table)
        for (auto& rate : output) rate.fill(capacity.sms);
    const auto source = ex::SelectedSource(ex::FrozenCache::k40G, {}, capacity);
    ASSERT_TRUE(source) << (source ? "" : source.error());
    auto arena = kg::TensorArena::Create(32).value();
    auto product = Make(arena.context(), 257, 2048, 6144, Encoding::kF32, 6);
    const auto declarations = (*source)->Implementations();
    const auto gemm = std::ranges::find(declarations, "exl3.pinned.40g.linear.gemm",
                                        &llmp::execution::Implementation::name);
    ASSERT_NE(gemm, declarations.end());
    EXPECT_FALSE(gemm->capability->needs_tuning);
    EXPECT_EQ(gemm->capability->device.min_arch, arch);
    const auto call = (*source)->Lower(*gemm, product.node, product.inputs, product.output);
    ASSERT_TRUE(call) << (call ? "" : call.error().detail);
    const auto fallback = ex::DefaultGemmPlan(capacity, 257, 2048, 6144, 6, ex::Output::kF32);
    ASSERT_TRUE(fallback);
    EXPECT_GE(fallback->shape, 1);
    EXPECT_LE(fallback->shape, ex::kShapes);
    EXPECT_LE(fallback->blocks, capacity.sms);
    auto missing = capacity;
    for (auto* table : {&missing.gemm, &missing.multi})
      for (auto& output : *table)
        for (auto& rate : output) rate.fill(0);
    EXPECT_FALSE(ex::DefaultGemmPlan(missing, 257, 2048, 6144, 6, ex::Output::kF32));
    EXPECT_FALSE(ex::SelectedSource(ex::FrozenCache::k40G, {}, missing));
    EXPECT_FALSE(ex::ReconstructionAlgorithms(arch, 257, 2048, 6144, ex::Output::kF32));
    EXPECT_FALSE(ex::DefaultGemmPlan(capacity, 257, 2049, 6144, 6, ex::Output::kF32));
    if (arch == 121) {
      namespace execution = llmp::execution;
      const auto operand = [](std::uint8_t index, const kg::SelectedOperand& value) {
        const auto& f = value.desc.format;
        return execution::OperandFormat{
            .operand = index,
            .encoding = f.encoding,
            .layout = f.layout,
            .params = f.params,
            .ne = {value.desc.ne(0), value.desc.ne(1), value.desc.ne(2), value.desc.ne(3)}};
      };
      auto registry = execution::Registry::Create(declarations);
      ASSERT_TRUE(registry);
      std::array operands{operand(0, product.inputs[0]), operand(1, product.inputs[1]),
                          operand(execution::kOutputOperand, product.output)};
      execution::CapabilityQuery query{.op = product.node.key,
                                       .operands = operands,
                                       .rows = 257,
                                       .k = 2048,
                                       .n = 6144,
                                       .arch = arch,
                                       .params = product.node.params};
      const auto candidates = execution::Candidates(*registry, query);
      ASSERT_EQ(candidates.size(), 1U);
      EXPECT_EQ(registry->at(candidates.front()).name, gemm->name);
      // Planner queries use Node's full fixed storage; direct capability
      // callers may provide only ProductParams. Both carry the same values.
      ir::ProductParams params{};
      query.params = {};
      EXPECT_EQ(execution::Candidates(*registry, query).size(), 1U);
      query.params = std::as_bytes(std::span(&params, 1));
      EXPECT_EQ(execution::Candidates(*registry, query).size(), 1U);
      query.params = query.params.first(sizeof(params) - 1);
      EXPECT_TRUE(execution::Candidates(*registry, query).empty());
      params = {.accumulate = ir::Accumulate::kF32};
      query.params = std::as_bytes(std::span(&params, 1));
      EXPECT_TRUE(execution::Candidates(*registry, query).empty());
      product.node.SetParams(params);
      query.params = product.node.params;
      EXPECT_TRUE(execution::Candidates(*registry, query).empty());
      params = {.hint = ir::ProductHint::kHadamard};
      query.params = std::as_bytes(std::span(&params, 1));
      EXPECT_TRUE(execution::Candidates(*registry, query).empty());
      product.node.SetParams(params);
      query.params = product.node.params;
      EXPECT_TRUE(execution::Candidates(*registry, query).empty());
      product.node.SetParams(ir::ProductParams{});
      EXPECT_EQ(execution::Candidates(*registry, query).size(), 1U);
      // Multipart alias descriptors cannot carry all four physical-part
      // views through shared lowering, including a zero-offset alias.
      operands[0].memory = execution::OperandMemory{.is_view = true};
      EXPECT_TRUE(execution::Candidates(*registry, query).empty());
      operands[0].memory = execution::OperandMemory{.view_offset = 16};
      EXPECT_TRUE(execution::Candidates(*registry, query).empty());
      operands[0].memory.reset();
      auto pinned = ex::SelectedSource(ex::FrozenCache::k40G);
      ASSERT_TRUE(pinned);
      auto no_defaults = execution::Registry::Create((*pinned)->Implementations());
      ASSERT_TRUE(no_defaults);
      const auto pins = (*pinned)->Tuning(arch);
      query.tuned = pins;
      EXPECT_TRUE(execution::Candidates(*no_defaults, query).empty());
      operands[2].encoding = Encoding::kBF16;
      EXPECT_TRUE(execution::Candidates(*registry, query).empty());

      std::array pair{operand(0, product.inputs[0]), operand(1, product.inputs[0]),
                      operand(2, product.inputs[1]),
                      operand(execution::kOutputOperand, product.output),
                      operand(execution::kOutputOperand + 1, product.output)};
      query.op = ir::Fused(ir::FusedKind::kPreparedProducts);
      query.operands = pair;
      const auto paired = execution::Candidates(*registry, query);
      ASSERT_EQ(paired.size(), 1U);
      EXPECT_EQ(registry->at(paired.front()).name, "exl3.pinned.40g.multi_linear.mgemm");
      for (const auto index : {0U, 1U}) {
        pair[index].memory = execution::OperandMemory{.is_view = true};
        EXPECT_TRUE(execution::Candidates(*registry, query).empty());
        pair[index].memory.reset();
      }
      for (const auto index : {3U, 4U}) {
        pair[index].encoding = Encoding::kF16;
        EXPECT_TRUE(execution::Candidates(*registry, query).empty());
        pair[index].encoding = Encoding::kF32;
      }
    }
  }
}

TEST(Exl3SelectedTest, GemvCapacityCannotBorrowPackedPinsFromAnotherSmCount) {
  ex::GemvCapacity capacity{.arch = 121, .sms = 24};  // Synthetic device facts.
  for (auto& output : capacity.blocks)
    for (auto& rows : output) rows.fill(capacity.sms);
  const auto source = ex::SelectedSource(ex::FrozenCache::k40O, capacity);
  ASSERT_TRUE(source);
  for (const auto& fact : (*source)->Tuning(121))
    EXPECT_TRUE(fact.implementation.ends_with(".linear.gemv"));
  const auto declarations = (*source)->Implementations();
  const auto gemm = std::ranges::find(declarations, "exl3.pinned.40o.linear.gemm",
                                      &llmp::execution::Implementation::name);
  const auto multi = std::ranges::find(declarations, "exl3.pinned.40o.multi_linear.mgemm",
                                       &llmp::execution::Implementation::name);
  ASSERT_NE(gemm, declarations.end());
  ASSERT_NE(multi, declarations.end());
  auto arena = kg::TensorArena::Create(32).value();
  const auto a = Make(arena.context(), 16, 896, 4864, Encoding::kF32);
  const auto b = Make(arena.context(), 16, 896, 4864, Encoding::kF32);
  EXPECT_FALSE((*source)->Lower(*gemm, a.node, a.inputs, a.output));
  auto pair = a.node;
  pair.key = ir::Fused(ir::FusedKind::kPreparedProducts);
  pair.nsrc = 3;
  pair.nout = 2;
  const std::array inputs{a.inputs[0], b.inputs[0], a.inputs[1]};
  const std::array outputs{a.output, b.output};
  EXPECT_FALSE((*source)->Lower(*multi, pair, inputs, outputs));
}

TEST(Exl3SelectedTest, AllCompletePinnedReconstructionsRetainFourPartsAndBoundScratch) {
  auto source = ex::SelectedSource(ex::FrozenCache::k40G);
  ASSERT_TRUE(source.has_value()) << source.error();
  const auto implementations = (*source)->Implementations();
  struct Shape {
    int k, n;
    Encoding output;
  };
  const std::array shapes{Shape{896, 896, Encoding::kF16},  Shape{896, 128, Encoding::kF16},
                          Shape{896, 896, Encoding::kF32},  Shape{896, 4864, Encoding::kF32},
                          Shape{4864, 896, Encoding::kF32}, Shape{896, 151936, Encoding::kF16}};
  for (const int rows : {145, 1023, 1024}) {
    for (const auto shape : shapes) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(shape.n);
      auto arena = kg::TensorArena::Create(8);
      ASSERT_TRUE(arena.has_value()) << arena.error().detail;
      const auto p = Make(arena->context(), rows, shape.k, shape.n, shape.output);
      const auto name = rows == 1024 ? "exl3.pinned.40g.linear.reconstruct_fused"
                                     : "exl3.pinned.40g.linear.reconstruct";
      const auto impl =
          std::ranges::find(implementations, name, &llmp::execution::Implementation::name);
      ASSERT_NE(impl, implementations.end());
      const auto call = (*source)->Lower(*impl, p.node, p.inputs, p.output);
      ASSERT_TRUE(call.has_value()) << call.error().detail;
      EXPECT_EQ((*call)->owner(), source->get());
      const std::uint64_t weight_slice = static_cast<std::uint64_t>(shape.k) *
                                         static_cast<unsigned>(std::min(shape.n, 32768)) * 2U;
      const auto input = static_cast<std::uint64_t>(rows) * static_cast<unsigned>(shape.k) * 2U;
      EXPECT_EQ((*call)->scratch_bytes(), weight_slice + (rows == 1024 ? 0 : input));
      EXPECT_LE((*call)->host_bytes(), 16384U);
      EXPECT_EQ(p.inputs[0].parts.size(), 4U);
    }
  }
}

TEST(Exl3SelectedTest, RefusesMissingDuplicateShortPartsAndWrongNumericalPath) {
  auto source = ex::SelectedSource(ex::FrozenCache::k40G);
  ASSERT_TRUE(source.has_value());
  const auto implementations = (*source)->Implementations();
  const auto impl = std::ranges::find(implementations, "exl3.pinned.40g.linear.reconstruct",
                                      &llmp::execution::Implementation::name);
  ASSERT_NE(impl, implementations.end());
  auto arena = kg::TensorArena::Create(64);
  ASSERT_TRUE(arena.has_value());
  auto p = Make(arena->context(), 145, 896, 896, Encoding::kF16);
  ASSERT_TRUE((*source)->Lower(*impl, p.node, p.inputs, p.output));
  p.inputs[0].parts.pop_back();
  EXPECT_FALSE((*source)->Lower(*impl, p.node, p.inputs, p.output));
  p = Make(arena->context(), 145, 896, 896, Encoding::kF16);
  p.inputs[0].parts[3] = p.inputs[0].parts[0];
  EXPECT_FALSE((*source)->Lower(*impl, p.node, p.inputs, p.output));
  p = Make(arena->context(), 145, 896, 896, Encoding::kF16);
  p.inputs[0].parts[0].tensor = ggml_new_tensor_1d(arena->context(), GGML_TYPE_I8, 16);
  EXPECT_FALSE((*source)->Lower(*impl, p.node, p.inputs, p.output));
  for (const int rows : {144, 1024}) {
    p = Make(arena->context(), rows, 896, 896, Encoding::kF16);
    EXPECT_FALSE((*source)->Lower(*impl, p.node, p.inputs, p.output));
  }
  p = Make(arena->context(), 145, 896, 151936, Encoding::kF32);
  EXPECT_FALSE((*source)->Lower(*impl, p.node, p.inputs, p.output));
  EXPECT_FALSE(ex::SelectedSource(ex::FrozenCache::k40O));
}
TEST(Exl3SelectedTest, AutomaticPairedProductsNeedQualificationAcrossTheCompleteRegistry) {
  namespace pl = llmp::planner;
  namespace execution = llmp::execution;
  namespace de = llmp::kernels::decode;
  const auto hardware = llmp::providers::QueryDeviceFacts(0);
  ASSERT_TRUE(hardware);
  if (hardware->architecture != 1210 || hardware->multiprocessors != 48)
    GTEST_SKIP() << "the historical reconstruction pins require the measured GB10/48SM device";
  const auto pinned = ex::SelectedSource(ex::FrozenCache::k40G);
  const auto decoders = de::DecodeResidentSelectedSource();
  const auto device = ex::DeviceSelectedSource(0);
  ASSERT_TRUE(pinned);
  ASSERT_TRUE(decoders);
  ASSERT_TRUE(device);
  const auto complete = de::CompositeSelectedSource({*pinned, *decoders, *device});
  auto implementations = kg::Implementations();
  for (auto& implementation : complete->Implementations())
    implementations.push_back(std::move(implementation));
  const auto registry = execution::Registry::Create(std::move(implementations));
  ASSERT_TRUE(registry);
  auto device_execution = llmp::providers::cuda::OpenDeviceExecution(0);
  ASSERT_TRUE(device_execution);
  const auto stream = (*device_execution)->CreateStream();
  ASSERT_TRUE(stream);
  auto default_backend = kg::GgmlBackend::Create(0, **device_execution, *stream, nullptr);
  ASSERT_TRUE(default_backend);
  const auto ordinary_backend = (*default_backend)->Backend();
  ASSERT_TRUE(ordinary_backend.device.preferred.empty());
  pl::DeviceFacts facts{.device_class = "sm_121", .arch = 121, .tuned = complete->Tuning(121)};
  for (const auto& implementation : (*pinned)->Implementations())
    facts.preferred.push_back(implementation.name);
  const auto graph = [](std::uint32_t rows, bool direct, bool decode_glu = false,
                        std::uint32_t width = 896) {
    ir::Builder b;
    const fmt::TensorFormat weight{
        .encoding = decode_glu ? Encoding::kMxfp8 : Encoding::kExl3Trellis,
        .params = decode_glu
                      ? fmt::EncodingParams{}
                      : fmt::EncodingParams{.k_bits = 4, .codebook = fmt::Exl3Codebook::kMcg},
        .layout = decode_glu ? fmt::Layout::kSplit : fmt::Layout::kExl3Tiles,
        .shape = ir::ShapeOf({width, 4864})};
    const auto wg = b.Weight(weight, {.stage = 0, .component = 1, .role = 0});
    const auto wu = b.Weight(weight, {.stage = 0, .component = 1, .role = 1});
    const auto x = b.Input(
        ir::Dense(decode_glu ? Encoding::kF32 : Encoding::kF16, ir::ShapeOf({width, rows})));
    if (direct) {
      const auto output = b.Fused(ir::FusedKind::kPreparedProducts, std::array{wg, wu, x});
      b.Output(output[0]);
      b.Output(output[1]);
    } else {
      const auto gate = b.MatMul(wg, x);
      const auto up = b.MatMul(wu, x);
      b.Output(b.Glu(gate, up, {.kind = ir::GluKind::kSwiGlu}));
    }
    return std::move(b).Finish();
  };
  const auto names = [&](const pl::StepPlan& plan) {
    std::vector<std::string> result;
    for (const auto& step : plan.steps) result.push_back(registry->at(step.implementation).name);
    return result;
  };
  // An already available native quantized GLU stays eligible alongside the
  // newly registered decoder GLU; the qualification gate is not a blanket
  // prohibition on automatic fused operations.
  ir::Builder native_glu;
  const fmt::TensorFormat q8{.encoding = Encoding::kQ8_0,
                             .layout = fmt::Layout::kGgmlBlocks,
                             .shape = ir::ShapeOf({896, 4864})};
  const auto qg = native_glu.Weight(q8, {.stage = 0, .component = 1, .role = 0});
  const auto qu = native_glu.Weight(q8, {.stage = 0, .component = 1, .role = 1});
  const auto qx = native_glu.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({896, 1})));
  const auto qgate = native_glu.MatMul(qg, qx);
  const auto qup = native_glu.MatMul(qu, qx);
  native_glu.Output(native_glu.Glu(qgate, qup, {.kind = ir::GluKind::kSwiGlu}));
  auto native_graph = std::move(native_glu).Finish();
  ASSERT_TRUE(native_graph);
  const auto native_plan = pl::Plan(std::move(*native_graph), *registry, facts, {});
  ASSERT_TRUE(native_plan) << native_plan.error().detail;
  EXPECT_EQ(names(*native_plan), std::vector<std::string>{"ggml.mul_mat_glu.mmvq_fused"});
  for (const auto rows : {1U, 145U, 1024U}) {
    SCOPED_TRACE(rows);
    auto primitive = graph(rows, false);
    ASSERT_TRUE(primitive);
    const auto ordinary = pl::Plan(primitive->Clone(), *registry, facts, {});
    ASSERT_TRUE(ordinary) << ordinary.error().detail;
    // The ordinary production backend has no explicit-source preference to
    // hide a later unqualified competitor behind the matcher-approved pin.
    const auto default_plan =
        pl::Plan(primitive->Clone(), *ordinary_backend.registry, ordinary_backend.device, {});
    ASSERT_TRUE(default_plan) << default_plan.error().detail;
    for (const auto& step : default_plan->steps) {
      const auto& implementation = ordinary_backend.registry->at(step.implementation);
      EXPECT_NE(implementation.name, "exl3.device.multi_linear.mgemm");
    }
    EXPECT_EQ(default_plan->steps.size(), rows == 1 ? 2U : 3U);
    if (rows == 1)
      EXPECT_EQ(ordinary_backend.registry->at(default_plan->steps.front().implementation).name,
                "exl3.pinned.40g.multi_linear.mgemm");
    auto default_direct = graph(rows, true);
    ASSERT_TRUE(default_direct);
    const auto default_typed = pl::Plan(std::move(*default_direct), *ordinary_backend.registry,
                                        ordinary_backend.device, {});
    ASSERT_TRUE(default_typed) << default_typed.error().detail;
    ASSERT_EQ(default_typed->steps.size(), 1U);
    EXPECT_EQ(ordinary_backend.registry->at(default_typed->steps.front().implementation).name,
              "exl3.device.multi_linear.mgemm");
    if (rows == 1) {
      EXPECT_EQ(names(*ordinary),
                (std::vector<std::string>{"exl3.pinned.40g.multi_linear.mgemm", "ggml.swiglu"}));
    } else {
      const auto linear = rows == 145 ? "exl3.pinned.40g.linear.reconstruct"
                                      : "exl3.pinned.40g.linear.reconstruct_fused";
      EXPECT_EQ(names(*ordinary), (std::vector<std::string>{linear, linear, "ggml.swiglu"}));
      // Direct typed use and a caller's actual measured preference remain
      // available. These synthetic facts qualify only this test, not a device
      // measurement or a production rank-table entry.
      auto direct = graph(rows, true);
      ASSERT_TRUE(direct);
      const auto typed = pl::Plan(std::move(*direct), *registry, facts, {});
      ASSERT_TRUE(typed) << typed.error().detail;
      EXPECT_EQ(names(*typed), std::vector<std::string>{"exl3.device.multi_linear.mgemm"});
      pl::Policy measured;
      measured.ranks = {{.device_class = "sm_121",
                         .implementation = "exl3.device.multi_linear.mgemm",
                         .rank = 0}};
      const auto qualified = pl::Plan(primitive->Clone(), *registry, facts, measured);
      ASSERT_TRUE(qualified) << qualified.error().detail;
      EXPECT_EQ(names(*qualified),
                (std::vector<std::string>{"exl3.device.multi_linear.mgemm", "ggml.swiglu"}));
      auto unsupported = graph(rows, false, false, 912);  // valid tiles, no packed K=128 grid
      ASSERT_TRUE(unsupported);
      EXPECT_FALSE(pl::Plan(std::move(*unsupported), *registry, facts, measured));
      measured.ranks[0].device_class = "different_device";
      const auto other_device = pl::Plan(primitive->Clone(), *registry, facts, measured);
      ASSERT_TRUE(other_device) << other_device.error().detail;
      EXPECT_EQ(names(*other_device), names(*ordinary));
      measured.ranks[0].device_class = "sm_121";
      measured.ranks[0].implementation = "different_implementation";
      const auto other_impl = pl::Plan(primitive->Clone(), *registry, facts, measured);
      ASSERT_TRUE(other_impl) << other_impl.error().detail;
      EXPECT_EQ(names(*other_impl), names(*ordinary));
    }
    auto decode = graph(rows, false, true);
    ASSERT_TRUE(decode);
    const auto decoded = pl::Plan(std::move(*decode), *registry, facts, {});
    ASSERT_TRUE(decoded) << decoded.error().detail;
    EXPECT_FALSE(std::ranges::any_of(decoded->graph.nodes(), [](const auto& node) {
      return node.key == ir::Fused(ir::FusedKind::kMatMulGlu) ||
             node.key == ir::Fused(ir::FusedKind::kPreparedProducts);
    }));
  }
  // Backend creation borrows Submission even without payload launches. The
  // provider requires a seen-complete, released fence before stream retirement.
  const auto fence = (*device_execution)->Record(*stream);
  ASSERT_TRUE(fence)
      << llmp::test_support::Failed(fence, &llmp::providers::Failure::detail).value_or("");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  auto state = (*device_execution)->Query(*fence);
  while (state && *state == llmp::providers::FenceState::kPending &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    state = (*device_execution)->Query(*fence);
  }
  ASSERT_TRUE(state)
      << llmp::test_support::Failed(state, &llmp::providers::Failure::detail).value_or("");
  ASSERT_EQ(*state, llmp::providers::FenceState::kComplete);
  const auto released = (*device_execution)->Release(*fence);
  ASSERT_TRUE(released)
      << llmp::test_support::Failed(released, &llmp::providers::Failure::detail).value_or("");
  default_backend->reset();
  const auto destroyed = (*device_execution)->DestroyStream(*stream);
  EXPECT_TRUE(destroyed)
      << llmp::test_support::Failed(destroyed, &llmp::providers::Failure::detail).value_or("");
}

TEST(Exl3SelectedTest, SharedGraphKeepsEveryWeightPartAndExactSelectedOperation) {
  namespace co = llmp::components;
  namespace sp = llmp::spec;
  namespace pl = llmp::planner;
  auto native = ex::SelectedSource(ex::FrozenCache::k40G);
  ASSERT_TRUE(native.has_value()) << native.error();
  auto declarations = kg::Implementations();
  for (auto& implementation : (*native)->Implementations())
    declarations.push_back(std::move(implementation));
  auto registry = llmp::execution::Registry::Create(std::move(declarations));
  ASSERT_TRUE(registry.has_value()) << registry.error().detail;
  const auto source = Source(true);
  const auto& mapping = sp::mappings::Qwen2Mapping();
  const auto values = fam::Qwen2Values(fam::Qwen25Instruct05BExl3());
  ASSERT_TRUE(values.has_value());
  const auto spec = sp::Synthesize(mapping, source, *values);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const auto state = co::StageLayout(*spec, 0, 2048, 1024);
  ASSERT_TRUE(state.has_value()) << state.error();
  auto policy = pl::ParsePolicy(mapping.policies.front().text);
  ASSERT_TRUE(policy.has_value()) << policy.error();
  kg::DeviceChoices choices;
  choices.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  const llmp::kernels::Addresses addresses{};
  // This manually assembled backend supplies one exact source, just as
  // GgmlBackend::Create's explicit-source preference does in production.
  std::vector<std::string> preferred;
  for (const auto& implementation : (*native)->Implementations())
    preferred.push_back(implementation.name);
  for (const auto rows : {1U, 3U, 7U, 8U, 32U, 33U, 144U, 145U, 1023U, 1024U}) {
    SCOPED_TRACE(rows);
    const pl::StepShape shape{.phase = (rows <= 8) ? pl::Phase::kScore : pl::Phase::kPrefill,
                              .segments = {{.slot = 0,
                                            .rows = rows,
                                            .past = 0,
                                            .read_cells = (rows + 255U) / 256U * 256U,
                                            .outputs = rows}}};
    auto step = co::BuildStep(*spec, 0, shape, {.tensors = source.tensors, .state = &*state});
    ASSERT_TRUE(step.has_value()) << step.error();
    std::vector<ir::TensorId> staged;
    for (const auto& input : step->inputs)
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
    auto plan = pl::Plan(step->graph.Clone(), *registry,
                         {.device_class = "sm_121",
                          .arch = 121,
                          .tuned = (*native)->Tuning(121),
                          .preferred = preferred},
                         *policy,
                         {.staged = staged,
                          .written = &kg::WrittenBytes,
                          .within = &kg::WithinRoot,
                          .shape = &shape});
    if (rows == 3) {
      // The frozen G cache has no bucket4 fact for this required product.
      // Refuse instead of manufacturing tuning for the natural43 tail.
      ASSERT_FALSE(plan.has_value());
      EXPECT_NE(plan.error().detail.find("mul_mat exl3_trellis[896,896"), std::string::npos);
      continue;
    }
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    const llmp::kernels::LowerInput input{*plan, *step, shape, addresses, 2048};
    auto lowered = kg::LowerStep(input, *registry, choices, {.selected_source = native->get()});
    ASSERT_TRUE(lowered.has_value()) << lowered.error().detail;
    EXPECT_EQ(lowered->selected.size(),
              rows > 32 ? 241U : 217U);  // 169 products, 72 q/k/v bias additions
    std::size_t products = 0, pairs = 0, biases = 0, attention = 0, parts = 0;
    for (const auto& selected : lowered->selected) {
      ASSERT_NE(selected.tensor, nullptr);
      ASSERT_NE(selected.implementation, nullptr);
      ASSERT_NE(selected.call, nullptr);
      EXPECT_EQ(selected.call->owner(), native->get());
      if (selected.implementation->name == "exl3.pinned.40g.bias_add") {
        ++biases;
      } else if (selected.implementation->name == "exl3.pinned.40g.multi_linear.mgemm") {
        ++pairs;
        products += 2;
        std::size_t sources = 0;
        for (const auto* source_tensor : selected.tensor->src) sources += source_tensor != nullptr;
        EXPECT_EQ(sources, 9U);
      } else {
        ++products;
        EXPECT_EQ(selected.implementation->name, rows == 1024
                                                     ? "exl3.pinned.40g.linear.reconstruct_fused"
                                                 : rows > 144 ? "exl3.pinned.40g.linear.reconstruct"
                                                              : "exl3.pinned.40g.linear.gemm");
        std::size_t sources = 0;
        for (const auto* source_tensor : selected.tensor->src) sources += source_tensor != nullptr;
        EXPECT_EQ(sources, 5U);  // four physical parts and the activation
      }
    }
    for (const auto& selected : lowered->plan.steps)
      attention += selected.implementation == "ggml.flash_attn_ext.vec";
    for (const auto& leaf : lowered->leaves) parts += leaf.part.has_value();
    EXPECT_EQ(pairs, rows > 32 ? 0U : 24U);
    EXPECT_EQ(products, 169U);
    EXPECT_EQ(biases, 72U);
    EXPECT_EQ(attention, 24U);
    EXPECT_EQ(parts, 676U);
  }
}

TEST(Exl3SelectedTest, ProfileIdentityAndGemvFactsAreExplicit) {
  ex::GemvCapacity capacity;
  for (auto& output : capacity.blocks)
    for (auto& rows : output) rows = {96, 96};  // synthetic capacity, no device claim
  const auto g = ex::SelectedSource(ex::FrozenCache::k40G);
  const auto o = ex::SelectedSource(ex::FrozenCache::k40O, capacity);
  ASSERT_TRUE(g.has_value());
  ASSERT_TRUE(o.has_value());
  EXPECT_FALSE(ex::SelectedSource(ex::FrozenCache::k40G, capacity));
  EXPECT_FALSE(ex::SelectedSource(ex::FrozenCache::k40O));
  auto invalid = capacity;
  invalid.blocks[0][0][0] = 95;
  EXPECT_FALSE(ex::SelectedSource(ex::FrozenCache::k40O, invalid));
  const auto gfacts = (*g)->Tuning(121);
  const auto ofacts = (*o)->Tuning(121);
  EXPECT_FALSE(std::ranges::any_of(
      gfacts, [](const auto& fact) { return fact.implementation.ends_with("linear.gemv"); }));
  EXPECT_TRUE(std::ranges::any_of(ofacts, [](const auto& fact) {
    return fact.implementation == "exl3.pinned.40o.linear.gemv" && fact.rows == 1 &&
           fact.k == 896 && fact.n == 896 && fact.output_encoding == Encoding::kF16;
  }));
  EXPECT_TRUE((*o)->Tuning(120).empty());
  for (const auto& fact : ofacts)
    if (fact.implementation.ends_with("multi_linear.mgemm")) EXPECT_LE(fact.rows, 32U);
  const auto declarations = (*g)->Implementations();
  const auto at = std::ranges::find(declarations, "exl3.pinned.40g.linear.reconstruct",
                                    &llmp::execution::Implementation::name);
  ASSERT_NE(at, declarations.end());
  auto arena = kg::TensorArena::Create(8);
  ASSERT_TRUE(arena.has_value());
  const auto p = Make(arena->context(), 145, 896, 896, Encoding::kF16);
  EXPECT_FALSE((*o)->Lower(*at, p.node, p.inputs, p.output));
  auto stale = *at;
  stale.variant += "; stale";
  EXPECT_FALSE((*g)->Lower(stale, p.node, p.inputs, p.output));
}

TEST(Exl3SelectedTest, BiasCandidatesRequireTheSamePackedGeometryAsLowering) {
  namespace execution = llmp::execution;
  auto source = ex::SelectedSource(ex::FrozenCache::k40G);
  ASSERT_TRUE(source);
  const auto declarations = (*source)->Implementations();
  const auto bias =
      std::ranges::find(declarations, "exl3.pinned.40g.bias_add", &execution::Implementation::name);
  ASSERT_NE(bias, declarations.end());
  auto registry = execution::Registry::Create(declarations).value();
  auto arena = kg::TensorArena::Create(4).value();
  const auto dense = [&](std::uint64_t rows) {
    kg::SelectedOperand value;
    value.desc.format = ir::Dense(Encoding::kF16, ir::ShapeOf({64, rows}));
    value.desc.nb = ir::NaturalStrides(value.desc.format);
    value.tensor =
        ggml_new_tensor_2d(arena.context(), GGML_TYPE_F16, 64, static_cast<std::int64_t>(rows));
    return value;
  };
  std::array inputs{dense(7), dense(1)};
  auto output = dense(7);
  const std::array descriptors{&inputs[0], &inputs[1], &output};
  std::array<execution::OperandFormat, 3> operands{};
  for (std::size_t index = 0; index < operands.size(); ++index) {
    operands[index] = {
        .operand = index == 2 ? execution::kOutputOperand : static_cast<std::uint8_t>(index),
        .encoding = Encoding::kF16,
        .layout = fmt::Layout::kDense,
        .ne = descriptors[index]->desc.shape().ne,
        .memory = execution::OperandMemory{.nb = descriptors[index]->desc.nb}};
  }
  ir::Node node;
  node.key = ir::Primitive(ir::Op::kAdd);
  node.nsrc = 2;
  node.nout = 1;
  execution::CapabilityQuery query{.op = node.key, .operands = operands, .arch = 121};
  ASSERT_EQ(execution::Candidates(registry, query).size(), 1U);
  ASSERT_TRUE((*source)->Lower(*bias, node, inputs, output));
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
      SCOPED_TRACE(std::format("operand{} stride{}", index, dimension));
      auto& desc = descriptors[index]->desc;
      const auto packed = desc.nb;
      desc.nb[dimension] += 2;
      operands[index].memory->nb = desc.nb;
      EXPECT_TRUE(execution::Candidates(registry, query).empty());
      EXPECT_FALSE((*source)->Lower(*bias, node, inputs, output));
      desc.nb = packed;
      operands[index].memory->nb = packed;
    }
  }
}

TEST(Exl3SelectedTest, UncapturedComponentSelectionIsExplicitAndKeyed) {
  namespace co = llmp::components;
  namespace sp = llmp::spec;
  namespace pl = llmp::planner;
  namespace en = llmp::engine;
  auto native = ex::SelectedSource(ex::FrozenCache::k40G);
  ASSERT_TRUE(native.has_value());
  auto declarations = kg::Implementations();
  for (auto& implementation : (*native)->Implementations())
    declarations.push_back(std::move(implementation));
  auto registry = llmp::execution::Registry::Create(std::move(declarations));
  ASSERT_TRUE(registry.has_value());
  const auto source = Source(true);
  const auto& mapping = sp::mappings::Qwen2Mapping();
  const auto values = fam::Qwen2Values(fam::Qwen25Instruct05BExl3());
  ASSERT_TRUE(values.has_value());
  const auto spec = sp::Synthesize(mapping, source, *values);
  ASSERT_TRUE(spec.has_value());
  const auto state = co::StageLayout(*spec, 0, 512, 145);
  ASSERT_TRUE(state.has_value());
  const pl::StepShape shape{
      .segments = {{.slot = 0, .rows = 145, .past = 0, .read_cells = 256, .outputs = 145}}};
  bool lowered = false;
  const llmp::kernels::StepBackend backend{
      .registry = &*registry,
      .device = {.device_class = "sm_121", .arch = 121, .tuned = (*native)->Tuning(121)},
      .written = &kg::WrittenBytes,
      .within = &kg::WithinRoot,
      .lower = [&](const llmp::kernels::LowerInput&)
          -> std::expected<std::unique_ptr<llmp::kernels::StepProgram>, std::string> {
        lowered = true;
        return std::unexpected("selection reached diagnostic lowering");
      }};
  const auto create = [&](bool require_capture) {
    return en::ComponentStepSource::Create(
        {.spec = &*spec,
         .bindings = {.tensors = source.tensors, .state = &*state},
         .bound_digest = spec->Digest(),
         .mapping = &mapping,
         .values = *values,
         .slots = 1,
         .backend = backend,
         .require_capture = require_capture});
  };
  auto captured = create(true);
  auto uncaptured = create(false);
  ASSERT_TRUE(captured.has_value());
  ASSERT_TRUE(uncaptured.has_value());
  EXPECT_NE((*captured)->ShapeParameters(shape), (*uncaptured)->ShapeParameters(shape));
  EXPECT_FALSE((*captured)->Build(shape, {}));
  EXPECT_FALSE(lowered);
  const auto result = (*uncaptured)->Build(shape, {});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), "selection reached diagnostic lowering");
  EXPECT_TRUE(lowered);
}

// These complete logical problems are the unique shape/rate/output tuples
// in backend-proof-p3/results.json's authenticated approved fixture inventory.
TEST(Exl3SelectedTest, AllProfilesCoverPackedPairsGemvAndReconstructionBoundaries) {
  struct Problem {
    int k, n, bits;
    Encoding output;
  };
  const std::array problems40{
      Problem{896, 896, 4, Encoding::kF16},  Problem{896, 128, 4, Encoding::kF16},
      Problem{896, 896, 4, Encoding::kF32},  Problem{896, 4864, 4, Encoding::kF32},
      Problem{4864, 896, 4, Encoding::kF32}, Problem{896, 151936, 8, Encoding::kF16}};
  const std::array problems45{
      Problem{896, 896, 5, Encoding::kF16},   Problem{896, 896, 6, Encoding::kF16},
      Problem{896, 128, 5, Encoding::kF16},   Problem{896, 128, 6, Encoding::kF16},
      Problem{896, 896, 5, Encoding::kF32},   Problem{896, 896, 6, Encoding::kF32},
      Problem{896, 4864, 4, Encoding::kF32},  Problem{896, 4864, 5, Encoding::kF32},
      Problem{4864, 896, 4, Encoding::kF32},  Problem{4864, 896, 5, Encoding::kF32},
      Problem{896, 151936, 8, Encoding::kF16}};
  ex::GemvCapacity capacity;
  for (auto& output : capacity.blocks)
    for (auto& rows : output) rows = {96, 96};  // synthetic host control only
  for (const auto cache : {ex::FrozenCache::k40G, ex::FrozenCache::k45G, ex::FrozenCache::k40O,
                           ex::FrozenCache::k45O}) {
    const bool optimized = cache == ex::FrozenCache::k40O || cache == ex::FrozenCache::k45O;
    const bool forty = cache == ex::FrozenCache::k40G || cache == ex::FrozenCache::k40O;
    const std::string profile = forty ? (optimized ? "40o" : "40g") : (optimized ? "45o" : "45g");
    auto source = ex::SelectedSource(cache, optimized ? std::optional(capacity) : std::nullopt);
    ASSERT_TRUE(source.has_value());
    const auto declarations = (*source)->Implementations();
    const auto facts = (*source)->Tuning(121);
    const std::span<const Problem> problems =
        forty ? std::span<const Problem>(problems40) : std::span<const Problem>(problems45);
    for (const auto rows : {1, 7, 8, 9, 16, 32, 33, 144, 145, 1023, 1024}) {
      for (const auto problem : problems) {
        SCOPED_TRACE(std::format("{} rows={} k={} n={} bits={}", profile, rows, problem.k,
                                 problem.n, problem.bits));
        auto arena = kg::TensorArena::Create(16);
        ASSERT_TRUE(arena.has_value());
        auto a = Make(arena->context(), rows, problem.k, problem.n, problem.output, problem.bits);
        const bool paired = rows <= 32 && problem.n == 4864;
        const bool gemv = !paired && optimized && problem.bits == 4 && rows <= 8;
        const std::string name = "exl3.pinned." + profile + "." +
                                 (paired         ? "multi_linear.mgemm"
                                  : rows == 1024 ? "linear.reconstruct_fused"
                                  : rows > 144   ? "linear.reconstruct"
                                  : gemv         ? "linear.gemv"
                                                 : "linear.gemm");
        const auto implementation =
            std::ranges::find(declarations, name, &llmp::execution::Implementation::name);
        ASSERT_NE(implementation, declarations.end());
        EXPECT_TRUE(std::ranges::any_of(facts, [&](const auto& fact) {
          return fact.implementation == name && fact.rows == static_cast<unsigned>(rows) &&
                 fact.k == static_cast<unsigned>(problem.k) &&
                 fact.n == static_cast<unsigned>(problem.n) &&
                 fact.output_encoding == problem.output;
        }));
        if (!paired) {
          const auto call = (*source)->Lower(*implementation, a.node, a.inputs, a.output);
          ASSERT_TRUE(call.has_value()) << call.error().detail;
          continue;
        }
        auto b = Make(arena->context(), rows, problem.k, problem.n, problem.output, problem.bits);
        const std::array inputs{a.inputs[0], b.inputs[0], a.inputs[1]};
        const std::array outputs{a.output, b.output};
        a.node.key = ir::Fused(ir::FusedKind::kPreparedProducts);
        a.node.nsrc = 3;
        a.node.nout = 2;
        const auto call = (*source)->Lower(*implementation, a.node, inputs, outputs);
        ASSERT_TRUE(call.has_value()) << call.error().detail;
        const auto input_bytes =
            static_cast<std::uint64_t>(rows) * static_cast<unsigned>(problem.k) * 2;
        EXPECT_EQ((*call)->scratch_bytes(), 256 + 2 * ((input_bytes + 255) / 256 * 256));
        auto missing = inputs;
        missing[1].parts.pop_back();
        EXPECT_FALSE((*source)->Lower(*implementation, a.node, missing, outputs));
      }
    }
  }
}

}  // namespace
