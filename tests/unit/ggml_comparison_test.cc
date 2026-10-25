// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "components/graph_io.h"
#include "components/step.h"
#include "execution/fusion_ranks.h"
#include "ir/builder.h"
#include "kernels/common/comparison_protocol.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/comparison.h"
#include "kernels/ggml/comparison_access.h"
#include "kernels/ggml/comparison_region.h"
#include "kernels/ggml/comparison_schema.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/measured_ranks.h"

namespace {

namespace base = llmp::base;
namespace co = llmp::components;
namespace ir = llmp::ir;
namespace kn = llmp::kernels;
namespace kg = llmp::kernels::ggml;

class ComparisonFunder final : public ir::StorageFunder {
 public:
  bool Charge(std::uint64_t bytes) const override {
    used += bytes;
    return true;
  }
  void Release(std::uint64_t bytes) const override {
    EXPECT_LE(bytes, used);
    used -= bytes;
  }
  mutable std::uint64_t used = 0;
};

// CPU-only authority controls: these descriptors never mint a backend seal.
class ComparisonSelectedSource final : public kn::SelectedSource {
 public:
  std::vector<llmp::execution::Implementation> Implementations() const override { return {}; }
  std::vector<llmp::execution::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view name) const override { return name == "test.selected.pair"; }
  std::optional<base::Sha256Digest> ExecutionIdentity() const override {
    base::Sha256Digest id{};
    id[0] = 1;
    return id;
  }
  std::uint64_t persistent_bytes() const override { return 256; }
  std::expected<std::unique_ptr<kn::SelectedCall>, kn::KernelFailure> Lower(
      const llmp::execution::Implementation&, const ir::Node&, std::span<const kn::SelectedOperand>,
      std::span<const kn::SelectedOperand>) const override {
    return std::unexpected(kn::KernelFailure{.detail = "descriptor-only control"});
  }
  std::expected<std::unique_ptr<kn::SelectedLaunch>, kn::KernelFailure> Launch(
      int, llmp::providers::DeviceExecution&, llmp::providers::StreamId,
      kn::SelectedWorkspace) const override {
    return std::unexpected(kn::KernelFailure{.detail = "descriptor-only control"});
  }
};
class ComparisonSelectedCall final : public kn::SelectedCall {
 public:
  explicit ComparisonSelectedCall(const kn::SelectedSource* source) : SelectedCall(source) {}
  std::uint64_t scratch_bytes() const override { return 256; }
  std::uint64_t host_bytes() const override { return sizeof(*this); }
  std::uint64_t launches() const override { return 2; }
  std::expected<std::unique_ptr<kn::BoundSelectedCall>, kn::KernelFailure> Bind(
      const llmp::execution::Implementation&, kn::SelectedLaunch&,
      kn::SelectedWorkspace) const override {
    return std::unexpected(kn::KernelFailure{.detail = "descriptor-only control"});
  }
};

// Test only the explicit scalar serialization, never backend execution authority.
struct ParameterWords {
  std::vector<std::uint32_t> values;
  void U8(std::uint8_t value) { values.push_back(value); }
  void U32(std::uint32_t value) { values.push_back(value); }
};

TEST(GgmlComparisonProtocolTest,
     WholeNativeParametersSealConstructorScalarsAndRejectReservedTails) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 128 * 1024, .mem_buffer = nullptr, .no_alloc = true}), ggml_free);
  ASSERT_TRUE(context);
  auto* c = context.get();
  auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 1);
  auto* ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, 1);
  auto* weight = ggml_new_tensor_2d(c, GGML_TYPE_BF16, 64, 64);
  auto* route = kg::MoeRoute(c, ids, 2);
  auto* quantized = kg::MoeQuantize(c, x, route);
  auto* product = kg::MoeGemm(c, quantized, route, weight, 64, 0, 4096);
  auto* read_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
  auto* qhead = kg::Dsv4QHead(c, x, ids,
                              {.eps = 1e-5F,
                               .original_context = 4096,
                               .base = 10000.F,
                               .scale = 1.F,
                               .extension = 0.F,
                               .attention = 1.F,
                               .beta_fast = 32.F,
                               .beta_slow = 1.F});
  const std::array<ggml_tensor*, 1> positions{ids};
  const std::array<std::uint32_t, 1> dimensions{64};
  const std::array nodes{kg::ToBf16(c, x),
                         kg::GemmBf16(c, weight, kg::ToBf16(c, x)),
                         route,
                         quantized,
                         product,
                         kg::MoeGemv(c, weight, x, ids, 64, 0, 64, 0, 4096),
                         kg::Mxfp8Quantize(c, x),
                         kg::HcNorm(c, x, weight, 1e-5F, GGML_TYPE_BF16),
                         kg::VecQ(c, weight, quantized, ids, 1, false),
                         kg::Dsv4HcMix(c, x, weight),
                         kg::Dsv4HcPre(c, x, x, x, x, x, 1e-5F, 1e-4F, 10),
                         kg::Dsv4Compress(c, x, x, x, x, read_ids, 4, false),
                         kg::HcPrep(c, x, weight, nullptr, nullptr, nullptr, 1e-5F),
                         qhead,
                         kg::RopeTable(c, positions, dimensions, 10000.F),
                         kg::QkAttention(c, x, weight, x, x, x, x, 1e-5F, 0.125F, GGML_TYPE_BF16)};
  for (auto* node : nodes) {
    ASSERT_NE(node, nullptr);
    const auto op = kg::LlmpOpOf(node);
    SCOPED_TRACE(static_cast<int>(op));
    ASSERT_TRUE(kg::HasWholeNativeCustomScalarSchema(*node));
    EXPECT_FALSE(kg::HasTypedDescriptorSchema(*node));
    const int count = kg::WholeNativeCustomScalarWords(op);
    ASSERT_GE(count, 0);
    ASSERT_LE(count, 8);
    ParameterWords original;
    kg::HashTypedOperationParameters(original, *node);
    ASSERT_EQ(original.values.size(), static_cast<std::size_t>(count + 1));
    auto* tail = reinterpret_cast<std::byte*>(node->op_params) + 32;
    for (int field = 0; field < count; ++field) {
      // Every semantic word, including late offset/mode/RoPE fields, changes
      // identity. Validity of the new value is the native checker's concern.
      tail[4 * field] ^= std::byte{1};
      ParameterWords changed;
      kg::HashTypedOperationParameters(changed, *node);
      EXPECT_NE(changed.values, original.values);
      tail[4 * field] ^= std::byte{1};
    }
    for (int field = count; field < 8; ++field) {
      tail[4 * field] = std::byte{1};
      EXPECT_FALSE(kg::HasWholeNativeCustomScalarSchema(*node));
      tail[4 * field] = std::byte{0};
    }
  }
  // An unknown custom function/tag cannot acquire a schema from zero bytes.
  ggml_tensor unknown{};
  unknown.op = GGML_OP_CUSTOM;
  EXPECT_FALSE(kg::HasWholeNativeCustomScalarSchema(unknown));
  EXPECT_EQ(kg::WholeNativeCustomScalarWords(kg::LlmpOp::kConv1x1), -1);
}

TEST(GgmlComparisonProtocolTest, WholeFillAndTopKKeepTypedScalarsAndReservedTails) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 128 * 1024, .mem_buffer = nullptr, .no_alloc = true}), ggml_free);
  ASSERT_TRUE(context);
  auto* x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 7, 3);
  auto* fill = ggml_fill(context.get(), x, -std::numeric_limits<float>::infinity());
  auto* top = ggml_top_k(context.get(), x, 3);
  for (auto* node : {fill, top}) {
    ASSERT_TRUE(kg::HasWholeNativeFillOrTopKSchema(*node));
    EXPECT_FALSE(kg::HasTypedDescriptorSchema(*node));
    ParameterWords original;
    kg::HashTypedOperationParameters(original, *node);
    EXPECT_EQ(original.values.size(), node == fill ? 1U : 0U);
    if (node == fill) {
      EXPECT_EQ(original.values[0],
                std::bit_cast<std::uint32_t>(-std::numeric_limits<float>::infinity()));
      node->op_params[0] = 0;
      ParameterWords changed;
      kg::HashTypedOperationParameters(changed, *node);
      EXPECT_NE(original.values, changed.values);
    }
    for (std::size_t word = node == fill ? 1 : 0; word < std::size(node->op_params); ++word) {
      node->op_params[word] = 1;
      EXPECT_FALSE(kg::HasWholeNativeFillOrTopKSchema(*node));
      node->op_params[word] = 0;
    }
    node->src[1] = x;
    EXPECT_FALSE(kg::HasWholeNativeFillOrTopKSchema(*node));
    node->src[1] = nullptr;
    node->src[0] = nullptr;
    EXPECT_EQ(kg::HasWholeNativeFillOrTopKSchema(*node), node == fill);
    if (node == fill) {
      // Lower's source-free fill has exactly the same scalar semantics as
      // ggml_fill's unused shape-source form, including 0 and -infinity.
      ParameterWords source_free;
      kg::HashTypedOperationParameters(source_free, *node);
      EXPECT_EQ(source_free.values, (std::vector<std::uint32_t>{0}));
      node->op_params[0] = std::bit_cast<std::int32_t>(-std::numeric_limits<float>::infinity());
      ASSERT_TRUE(kg::HasWholeNativeFillOrTopKSchema(*node));
      ParameterWords hidden;
      kg::HashTypedOperationParameters(hidden, *node);
      EXPECT_EQ(hidden.values, original.values);
    }
  }
}

TEST(GgmlComparisonProtocolTest, WholeReluRequiresItsExactUnaryAndSourceContract) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 128 * 1024, .mem_buffer = nullptr, .no_alloc = true}), ggml_free);
  ASSERT_TRUE(context);
  auto* x = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 7, 4, 17);
  auto* node = ggml_relu(context.get(), x);
  ASSERT_TRUE(kg::HasWholeNativeReluSchema(*node));
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*node));
  ParameterWords parameters;
  kg::HashTypedOperationParameters(parameters, *node);
  EXPECT_EQ(parameters.values,
            (std::vector<std::uint32_t>{static_cast<std::uint32_t>(GGML_UNARY_OP_RELU)}));
  for (std::size_t word = 1; word < std::size(node->op_params); ++word) {
    node->op_params[word] = 1;
    EXPECT_FALSE(kg::HasWholeNativeReluSchema(*node));
    node->op_params[word] = 0;
  }
  for (std::size_t source = 1; source < GGML_MAX_SRC; ++source) {
    node->src[source] = x;
    EXPECT_FALSE(kg::HasWholeNativeReluSchema(*node));
    node->src[source] = nullptr;
  }
  node->src[0] = nullptr;
  EXPECT_FALSE(kg::HasWholeNativeReluSchema(*node));
  node->src[0] = x;
  node->op_params[0] = GGML_UNARY_OP_ABS;
  EXPECT_FALSE(kg::HasWholeNativeReluSchema(*node));
  ParameterWords changed;
  kg::HashTypedOperationParameters(changed, *node);
  EXPECT_NE(changed.values, parameters.values);
  node->op_params[0] = GGML_UNARY_OP_RELU;
  node->op = GGML_OP_FILL;
  EXPECT_FALSE(kg::HasWholeNativeReluSchema(*node));
}

TEST(GgmlComparisonProtocolTest, NativeExpertSlabRequiresEveryExactDeclaredPart) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_I8;
  tensor.op = GGML_OP_NONE;
  tensor.data = reinterpret_cast<void*>(0x100000);
  tensor.ne[0] = 13824;
  tensor.ne[1] = 8;
  tensor.ne[2] = tensor.ne[3] = 1;
  tensor.nb[0] = 1;
  tensor.nb[1] = 13824;
  tensor.nb[2] = tensor.nb[3] = 110592;
  kg::LoweredLeaf leaf{
      .tensor = &tensor,
      .primitive = ir::TensorId{7},
      .part = llmp::format::PartRole::kCodes,
      .comparison_slab = kg::ComparisonExpertSlab{.bindings = {0, 1, 2, 3},
                                                  .roots = {ir::TensorId{7}, ir::TensorId{8}},
                                                  .width = 128,
                                                  .hidden = 64,
                                                  .experts = 8}};
  const std::array<std::uint64_t, 5> offsets{0, 8192, 9216, 13312, 13824};
  std::array<kn::ComparisonBindingRange, 4> ranges{};
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    ranges[i] = {
        .original_root = ir::TensorId{static_cast<std::uint32_t>(7 + i / 2)},
        .part = i % 2 == 0 ? llmp::format::PartRole::kCodes : llmp::format::PartRole::kBlockScales,
        .allocation_class = 2,
        .allocation_bytes = 110592,
        .offset = offsets[i],
        .bytes = 7 * 13824 + offsets[i + 1] - offsets[i],
        .alignment = 256,
        .base = 0x100000};
  }
  ASSERT_TRUE(kg::CheckComparisonExpertSlab(leaf, ranges));
  EXPECT_LT(ranges[0].bytes, ggml_nbytes(&tensor));  // part authority is not inflated to slab
  for (std::size_t part = 0; part < ranges.size(); ++part) {
    for (int mutation = 0; mutation < 7; ++mutation) {
      SCOPED_TRACE(part);
      SCOPED_TRACE(mutation);
      auto changed = ranges;
      auto& r = changed[part];
      if (mutation == 0) r.original_root = ir::TensorId{9};
      if (mutation == 1) r.part = llmp::format::PartRole::kData;
      if (mutation == 2) ++r.offset;
      if (mutation == 3) --r.bytes;
      if (mutation == 4) --r.allocation_bytes;
      if (mutation == 5) r.base += 256;
      if (mutation == 6) ++r.allocation_class;
      EXPECT_FALSE(kg::CheckComparisonExpertSlab(leaf, changed));
    }
  }
  EXPECT_FALSE(kg::CheckComparisonExpertSlab(leaf, std::span(ranges).first(3)));
  const auto saved = *leaf.comparison_slab;
  leaf.comparison_slab->bindings[3] = 2;
  EXPECT_FALSE(kg::CheckComparisonExpertSlab(leaf, ranges));
  leaf.comparison_slab = saved;
  ++tensor.nb[1];
  EXPECT_FALSE(kg::CheckComparisonExpertSlab(leaf, ranges));
  --tensor.nb[1];
  leaf.comparison_slab->experts = UINT64_MAX;
  EXPECT_FALSE(kg::CheckComparisonExpertSlab(leaf, ranges));
  leaf.comparison_slab = saved;
  tensor.data = reinterpret_cast<void*>(UINTPTR_MAX - 255);
  EXPECT_FALSE(kg::CheckComparisonExpertSlab(leaf, ranges));
}

TEST(GgmlComparisonProtocolTest, WholeHostIndicesRequireRegisteredRecipesAndActualCapacity) {
  llmp::planner::StepShape shape{
      .phase = llmp::execution::Phase::kPrefill,
      .segments = {{.slot = 0, .rows = 6, .past = 7, .read_cells = 16, .outputs = 1}}};
  const auto make = [&](bool block, bool overlap) {
    co::StepInput source{
        .tensor = ir::TensorId{0}, .kind = co::InputKind::kHostComputed, .segment = 0};
    source.host = {
        .def = block ? &co::CompressorBlockCellsInput() : &co::CompressorKeptCellsInput(),
        .ratio = block ? 4U : 0U,
        .overlap = overlap};
    source.ratio = source.host.ratio;
    source.overlap = overlap;
    source.cells = {.capacity = block ? 8U : 4U,
                    .ring = !block,
                    .block = block ? 4U : 0U,
                    .positions = block,
                    .recent = !block};
    source.ring = source.cells.capacity;
    kn::InputSlot slot{.bytes = block ? 16U : 32U,
                       .tensor = source.tensor,
                       .segment = 0,
                       .kind = kn::InputKind::kHostComputed};
    slot.host = source.host.def;
    slot.ratio = source.ratio;
    slot.overlap = overlap;
    slot.ring = source.ring;
    slot.cells = {.capacity = source.cells.capacity,
                  .ring = source.cells.ring,
                  .block = source.cells.block,
                  .positions = source.cells.positions,
                  .recent = source.cells.recent};
    return std::pair{source, slot};
  };
  for (const auto scenario : {0, 1, 2}) {
    auto [source, slot] = make(scenario != 2, scenario == 1);
    const auto recipe = kg::ComparisonHostIndexRecipe(source, slot, shape, source.cells.capacity);
    ASSERT_TRUE(recipe);
    EXPECT_EQ(*recipe, scenario == 2 ? "components.compressor.kept_cells.v1"
                                     : "components.compressor.block_cells.v1");
    const auto values = co::InputValues(source, shape, {});
    ASSERT_TRUE(values);
    EXPECT_EQ(*values, scenario == 2 ? (std::vector<std::int64_t>{1, 2, 3, 0})
                                     : (std::vector<std::int64_t>{1, 2}));
    EXPECT_FALSE(kg::ComparisonHostIndexRecipe(source, slot, shape, source.cells.capacity - 1));
    slot.bytes += 8;
    EXPECT_FALSE(kg::ComparisonHostIndexRecipe(source, slot, shape, source.cells.capacity));
  }
  // The dummy block is a real write to the last cache cell, never discarded
  // from the contract just because no visible block completed.
  auto [source, slot] = make(true, true);
  shape.segments[0].past = 2;
  shape.segments[0].rows = 1;
  slot.bytes = 8;
  ASSERT_TRUE(kg::ComparisonHostIndexRecipe(source, slot, shape, 8));
  auto dummy = co::InputValues(source, shape, {});
  ASSERT_TRUE(dummy);
  EXPECT_EQ(*dummy, (std::vector<std::int64_t>{7}));
  shape.segments[0].past = 32;  // completed block/dummy is beyond the same cache
  EXPECT_FALSE(kg::ComparisonHostIndexRecipe(source, slot, shape, 8));
  shape.segments[0].past = 2;
  const auto saved_source = source;
  const auto saved_slot = slot;
  const auto copied_definition = *source.host.def;
  for (unsigned mutation = 0; mutation < 10; ++mutation) {
    SCOPED_TRACE(mutation);
    source = saved_source;
    slot = saved_slot;
    switch (mutation) {
      case 0:
        source.host.def = slot.host = &copied_definition;
        break;
      case 1:
        source.host.def = slot.host = &co::CompressorBlockStartsInput();
        break;
      case 2:
        slot.host = &co::CompressorKeptCellsInput();
        break;
      case 3:
        source.host.ratio = 2;
        break;
      case 4:
        slot.overlap = false;
        break;
      case 5:
        slot.ring = 7;
        break;
      case 6:
        slot.cells.block = 2;
        break;
      case 7:
        slot.commit_save = true;
        break;
      case 8:
        source.segment = slot.segment = 1;
        break;
      case 9:
        slot.tensor = ir::TensorId{1};
        break;
    }
    EXPECT_FALSE(kg::ComparisonHostIndexRecipe(source, slot, shape, 8));
  }
}

TEST(GgmlComparisonProtocolTest, WholeMmfRequiresExactActualSelectionAndCheckedOperands) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 256 * 1024, .mem_buffer = nullptr, .no_alloc = false}), ggml_free);
  ASSERT_TRUE(context);
  auto* weights = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 64, 64);
  auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 64, 4);
  auto* product = ggml_mul_mat(context.get(), weights, input);
  kg::PlanStep step{.operation = ir::Primitive(ir::Op::kMatMul),
                    .implementation = "ggml.mul_mat.mmf",
                    .nodes = {product}};
  kg::ComparisonRemainderCallFacts facts;
  EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  facts.float_selection = kg::MulMatPath::kTensorCore;
  ASSERT_TRUE(kg::CheckComparisonMmf(step, facts));
  for (const auto path : {kg::MulMatPath::kVector, kg::MulMatPath::kCublas}) {
    facts.float_selection = path;
    EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  }
  facts.float_selection = kg::MulMatPath::kTensorCore;
  facts.scratch_bytes = 1;
  EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  facts.scratch_bytes = 0;
  step.implementation = "ggml.mul_mat.mmvf";
  EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  step.implementation = "ggml.mul_mat.mmf";
  step.lane = 1;
  EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  step.lane = 0;
  const auto saved = *product;
  product->src[1] = nullptr;
  EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  *product = saved;
  product->op_params[1] = 1;  // non-semantic reserved word
  EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  *product = saved;
  product->data = input->data;
  EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  *product = saved;
  input->ne[1] = 17;
  EXPECT_FALSE(kg::CheckComparisonMmf(step, facts));
  input->ne[1] = 4;
  ASSERT_TRUE(kg::CheckComparisonMmf(step, facts));
}

TEST(GgmlComparisonProtocolTest, WholeBf16CublasRequiresExactConversionPlanAndWorkspace) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 512 * 1024, .mem_buffer = nullptr, .no_alloc = false}), ggml_free);
  ASSERT_TRUE(context);
  auto* weights = ggml_new_tensor_2d(context.get(), GGML_TYPE_BF16, 256, 256);
  auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 256, 17);
  auto* product = ggml_mul_mat(context.get(), weights, input);
  kg::PlanStep step{.operation = ir::Primitive(ir::Op::kMatMul),
                    .implementation = "ggml.mul_mat.cublas",
                    .nodes = {product}};
  const auto plan = kg::CheckMulMatCublas(product, GGML_TYPE_BF16, true);
  ASSERT_TRUE(plan);
  ASSERT_EQ(plan->input, kg::CublasOperand::kConverted);
  ASSERT_EQ(plan->scratch, 256U * 17U * 2U);
  kg::ComparisonRemainderCallFacts facts;
  facts.has_mul_mat_selection = true;
  facts.float_selection = kg::MulMatPath::kCublas;
  facts.cublas_plan = *plan;
  facts.scratch_bytes = plan->scratch;
  facts.library_workspace_base = 0x100000000ULL;
  facts.library_workspace_bytes = 256;
  facts.cublas_configuration[0] = 1;
  ASSERT_TRUE(kg::CheckComparisonWholeCublas(step, facts));  // descriptor: pool not bound yet
  EXPECT_FALSE(kg::CheckComparisonSgemm(step, facts));       // no local SGEMM widening
  const auto original_facts = facts;
  for (unsigned mutation = 0; mutation < 18; ++mutation) {
    SCOPED_TRACE(mutation);
    facts = original_facts;
    switch (mutation) {
      case 0:
        facts.has_mul_mat_selection = false;
        break;
      case 1:
        facts.float_selection = kg::MulMatPath::kTensorCore;
        break;
      case 2:
        facts.mul_mat_vector = true;
        break;
      case 3:
        facts.cublas_plan.reset();
        break;
      case 4:
        facts.cublas_plan->compute = GGML_TYPE_F32;
        break;
      case 5:
        facts.cublas_plan->f32_output = false;
        break;
      case 6:
        facts.cublas_plan->weights = kg::CublasOperand::kConverted;
        break;
      case 7:
        facts.cublas_plan->input = kg::CublasOperand::kDirect;
        break;
      case 8:
        facts.cublas_plan->gemm = kg::CublasGemm::kSgemm;
        break;
      case 9:
        ++facts.cublas_plan->s11;
        break;
      case 10:
        facts.cublas_plan->alignment /= 2;
        break;
      case 11:
        ++facts.cublas_plan->scratch;
        break;
      case 12:
        --facts.scratch_bytes;
        break;
      case 13:
        facts.cublas_configuration = {};
        break;
      case 14:
        facts.library_workspace_base = reinterpret_cast<std::uintptr_t>(input->data);
        break;
      case 15:
        facts.library_workspace_bytes = 0;
        break;
      case 16:
        facts.pool_faulted = true;
        break;
      case 17:
        facts.pool_lanes = 1;
        break;
    }
    EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  }
  facts = original_facts;
  facts.pool_base = 0x200000000ULL;
  facts.pool_bytes = plan->scratch;
  ASSERT_TRUE(kg::CheckComparisonWholeCublas(step, facts));
  --facts.pool_bytes;
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  facts.pool_bytes = plan->scratch;
  facts.pool_base = reinterpret_cast<std::uintptr_t>(input->data);
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  facts = original_facts;
  const auto saved = *product;
  product->src[1] = nullptr;
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  *product = saved;
  product->op_params[0] = GGML_PREC_F32;
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  *product = saved;
  product->op_params[1] = 1;
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  *product = saved;
  product->data = input->data;
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  *product = saved;
  const auto input_saved = *input;
  input->ne[2] = 2;
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  *input = input_saved;
  weights->type = GGML_TYPE_F16;
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  weights->type = GGML_TYPE_BF16;
  step.lane = 1;
  EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
  step.lane = 0;
  ASSERT_TRUE(kg::CheckComparisonWholeCublas(step, facts));
}

TEST(GgmlComparisonProtocolTest, OmittedWholeInputCannotAcquireNativeStorage) {
  kg::LoweredStep lowered;
  const kg::ComparisonOmittedInput fact{
      .original = ir::TensorId{1}, .planned = ir::TensorId{2}, .producer = ir::NodeId{3}};
  EXPECT_FALSE(kg::CheckComparisonOmittedInput(lowered, fact.original));
  lowered.comparison_omitted_inputs.push_back(fact);
  ASSERT_TRUE(kg::CheckComparisonOmittedInput(lowered, fact.original));
  EXPECT_FALSE(kg::CheckComparisonOmittedInput(lowered, ir::TensorId{0}));
  for (unsigned mutation = 0; mutation < 10; ++mutation) {
    SCOPED_TRACE(mutation);
    switch (mutation) {
      case 0:
        lowered.comparison_omitted_inputs.push_back(fact);
        break;
      case 1:
        lowered.comparison_omitted_inputs[0].planned = {};
        break;
      case 2:
        lowered.comparison_omitted_inputs[0].producer = {};
        break;
      case 3:
        lowered.leaves.push_back({.primitive = fact.original});
        break;
      case 4:
        lowered.inputs.push_back({.tensor = fact.original});
        break;
      case 5:
        lowered.physical_tensors.push_back({.tensor = fact.planned});
        break;
      case 6:
        lowered.physical_tensors.push_back({.root = fact.planned});
        break;
      case 7:
        lowered.physical_tensors.push_back({.producer = fact.producer});
        break;
      case 8: {
        kg::PhysicalRoot root{};
        root.root = fact.planned;
        lowered.physical_roots.push_back(root);
        break;
      }
      case 9:
        lowered.comparison_omitted_inputs.push_back(
            {.original = ir::TensorId{4}, .planned = fact.planned, .producer = ir::NodeId{5}});
        break;
    }
    EXPECT_FALSE(kg::CheckComparisonOmittedInput(lowered, fact.original));
    lowered.comparison_omitted_inputs = {fact};
    lowered.leaves.clear();
    lowered.inputs.clear();
    lowered.physical_tensors.clear();
    lowered.physical_roots.clear();
    EXPECT_TRUE(kg::CheckComparisonOmittedInput(lowered, fact.original));
  }
}

TEST(GgmlComparisonProtocolTest, FloatVectorRowsUsesTheLaunchersExactCheckedDomain) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 512 * 1024, .mem_buffer = nullptr, .no_alloc = false}), ggml_free);
  ASSERT_TRUE(context);
  for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
    SCOPED_TRACE(type);
    for (const int rows : {1, 8, 9}) {
      auto* weights = ggml_new_tensor_2d(context.get(), type, 128, 32);
      auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 128, rows);
      auto* out = ggml_mul_mat(context.get(), weights, input);
      ASSERT_TRUE(kg::CheckMulMat(out));
      EXPECT_EQ(kg::CheckMulMatVecFRows(out).has_value(), rows <= 8);
      const auto saved = *out;
      out->data = input->data;
      EXPECT_FALSE(kg::CheckMulMatVecFRows(out));
      *out = saved;
      out->src[1] = nullptr;
      EXPECT_FALSE(kg::CheckMulMatVecFRows(out));
    }
    auto* odd_weights = ggml_new_tensor_2d(context.get(), type, 127, 32);
    auto* odd_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 127, 1);
    auto* odd_out = ggml_mul_mat(context.get(), odd_weights, odd_input);
    EXPECT_FALSE(kg::CheckMulMatVecFRows(odd_out));
  }
}

TEST(GgmlComparisonProtocolTest, BackwardRopeSealsItsLateOffsetWithoutLocalAuthority) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 1024 * 1024, .mem_buffer = nullptr, .no_alloc = false}), ggml_free);
  ASSERT_TRUE(context);
  auto* input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 512, 64, 1);
  auto* positions = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 1);
  auto* backward = ggml_rope_ext_back(context.get(), input, positions, nullptr, 64, 0, 0, 160000.0F,
                                      1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
  ggml_rope_set_offset(backward, 448);
  ASSERT_TRUE(kg::CheckRopeExt(backward));
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*backward));
  kg::PlanStep step{.operation = ir::Primitive(ir::Op::kRope),
                    .implementation = kg::kRopeExtName,
                    .nodes = {backward}};
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  ParameterWords original;
  kg::HashTypedOperationParameters(original, *backward);
  ASSERT_EQ(original.values.size(), 16U);
  EXPECT_EQ(original.values.back(), 448U);
  ggml_rope_set_offset(backward, 446);
  ASSERT_TRUE(kg::CheckRopeExt(backward));
  ParameterWords changed;
  kg::HashTypedOperationParameters(changed, *backward);
  EXPECT_NE(changed.values, original.values);
  ggml_rope_set_offset(backward, 450);
  EXPECT_FALSE(kg::CheckRopeExt(backward));
  ggml_rope_set_offset(backward, 448);
  const auto saved = backward->src[1];
  backward->src[1] = nullptr;
  EXPECT_FALSE(kg::CheckRopeExt(backward));
  backward->src[1] = saved;
  const auto data = backward->data;
  backward->data = static_cast<std::byte*>(input->data) + 8;
  EXPECT_FALSE(kg::CheckRopeExt(backward));
  backward->data = data;
  EXPECT_TRUE(kg::CheckRopeExt(backward));
}

TEST(GgmlComparisonProtocolTest, WholeCublasClosesCheckedBatchesWithoutLocalAuthority) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 2 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = false}),
      ggml_free);
  ASSERT_TRUE(context);
  for (unsigned form = 0; form < 3; ++form) {
    SCOPED_TRACE(form);
    auto* c = context.get();
    auto* weights =
        form == 2 ? ggml_permute(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 2, 17), 0, 2, 1, 3)
                  : ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 17, form == 1 ? 1 : 2);
    auto* input = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 17, 2);
    auto* product = ggml_mul_mat(c, weights, input);
    kg::PlanStep step{.operation = ir::Primitive(ir::Op::kMatMul),
                      .implementation = "ggml.mul_mat.cublas",
                      .nodes = {product}};
    const auto plan = kg::CheckMulMatCublas(product, GGML_TYPE_F32, false);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->gemm,
              form == 1 ? kg::CublasGemm::kGemmBatchedEx : kg::CublasGemm::kGemmStridedBatchedEx);
    EXPECT_EQ(plan->scratch == 0, form != 1);
    kg::ComparisonRemainderCallFacts facts;
    facts.has_mul_mat_selection = true;
    facts.float_selection = kg::MulMatPath::kCublas;
    facts.cublas_plan = *plan;
    facts.scratch_bytes = plan->scratch;
    facts.library_workspace_base = 0x100000000ULL;
    facts.library_workspace_bytes = 256;
    facts.cublas_configuration[0] = 1;
    ASSERT_TRUE(kg::CheckComparisonWholeCublas(step, facts));
    EXPECT_FALSE(kg::ComparisonSgemmCall(step));
    EXPECT_FALSE(kg::CheckComparisonSgemm(step, facts));
    const auto original = facts;
    for (unsigned mutation = 0; mutation < 17; ++mutation) {
      SCOPED_TRACE(mutation);
      facts = original;
      switch (mutation) {
        case 0:
          facts.float_selection = kg::MulMatPath::kTensorCore;
          break;
        case 1:
          facts.cublas_plan->gemm = kg::CublasGemm::kSgemm;
          break;
        case 2:
          ++facts.cublas_plan->s01;
          break;
        case 3:
          ++facts.cublas_plan->s02;
          break;
        case 4:
          ++facts.cublas_plan->s03;
          break;
        case 5:
          ++facts.cublas_plan->s11;
          break;
        case 6:
          ++facts.cublas_plan->s12;
          break;
        case 7:
          ++facts.cublas_plan->s13;
          break;
        case 8:
          ++facts.cublas_plan->scratch;
          break;
        case 9:
          ++facts.scratch_bytes;
          break;
        case 10:
          facts.cublas_plan->alignment /= 2;
          break;
        case 11:
          facts.cublas_plan->compute = GGML_TYPE_Q8_0;
          break;
        case 12:
          facts.cublas_plan->input = kg::CublasOperand::kConverted;
          break;
        case 13:
          facts.library_workspace_base = reinterpret_cast<std::uintptr_t>(input->data);
          break;
        case 14:
          facts.pool_faulted = true;
          break;
        case 15:
          facts.cublas_configuration = {};
          break;
        case 16:
          facts.cublas_plan->f32_output = true;
          break;
      }
      EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
    }
    facts = original;
    facts.pool_base = 0x200000000ULL;
    facts.pool_bytes = std::max<std::uint64_t>(256, plan->scratch);
    EXPECT_TRUE(kg::CheckComparisonWholeCublas(step, facts));
    facts.pool_base = reinterpret_cast<std::uintptr_t>(input->data);
    EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
    facts = original;
    const auto saved = *product;
    product->data = input->data;
    EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
    *product = saved;
    product->src[1] = nullptr;
    EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
    *product = saved;
    product->op_params[1] = 1;
    EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
    *product = saved;
    const auto old_stride = weights->nb[1];
    weights->nb[1] = sizeof(float);  // invalid cuBLAS leading dimension
    EXPECT_FALSE(kg::CheckComparisonWholeCublas(step, facts));
    weights->nb[1] = old_stride;
    EXPECT_TRUE(kg::CheckComparisonWholeCublas(step, facts));
  }
}

TEST(GgmlComparisonProtocolTest, WholeBf16ProductRequiresActualMethodAndLibraryFacts) {
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 256 * 1024, .mem_buffer = nullptr, .no_alloc = false}), ggml_free);
  ASSERT_TRUE(context);
  auto* c = context.get();
  auto* weights = ggml_new_tensor_2d(c, GGML_TYPE_BF16, 64, 48);
  auto* input = ggml_new_tensor_2d(c, GGML_TYPE_BF16, 64, 4);
  auto* product = kg::GemmBf16(c, weights, input);
  kg::PlanStep step{.operation = ir::Primitive(ir::Op::kMatMul),
                    .implementation = "llmp.gemm.bf16",
                    .nodes = {product}};
  ASSERT_TRUE(kg::ComparisonBf16Call(step));
  kg::ComparisonRemainderCallFacts facts;
  EXPECT_FALSE(kg::CheckComparisonBf16(step, facts));
  facts.bf16_product = kg::ComparisonBf16Product::kVector;
  EXPECT_FALSE(kg::CheckComparisonBf16(step, facts));
  facts.bf16_product = kg::ComparisonBf16Product::kGemmEx;
  EXPECT_FALSE(kg::CheckComparisonBf16(step, facts));
  facts.library_workspace_base = 0x100000000ULL;
  facts.library_workspace_bytes = 256;
  facts.cublas_configuration[0] = 1;
  ASSERT_TRUE(kg::CheckComparisonBf16(step, facts));
  facts.scratch_bytes = 1;
  EXPECT_FALSE(kg::CheckComparisonBf16(step, facts));
  facts.scratch_bytes = 0;
  facts.library_workspace_base = reinterpret_cast<std::uintptr_t>(weights->data);
  EXPECT_FALSE(kg::CheckComparisonBf16(step, facts));
  auto* mode = reinterpret_cast<std::byte*>(product->op_params) + 32;
  *mode = std::byte{4};
  EXPECT_FALSE(kg::ComparisonBf16Call(step));
  *mode = std::byte{3};
  EXPECT_FALSE(kg::ComparisonBf16Call(step));  // pinned mode under the ordinary name
  *mode = std::byte{0};
  weights->type = GGML_TYPE_F16;
  EXPECT_FALSE(kg::ComparisonBf16Call(step));
  weights->type = GGML_TYPE_BF16;
  product->data = weights->data;
  EXPECT_FALSE(kg::ComparisonBf16Call(step));
  auto* single = ggml_new_tensor_2d(c, GGML_TYPE_BF16, 64, 1);
  auto* vector = kg::GemvBf16(c, weights, single);
  step.nodes = {vector};
  facts = {};
  facts.bf16_product = kg::ComparisonBf16Product::kVector;
  ASSERT_TRUE(kg::CheckComparisonBf16(step, facts));
  facts.bf16_product = kg::ComparisonBf16Product::kGemmEx;
  EXPECT_FALSE(kg::CheckComparisonBf16(step, facts));
}

TEST(GgmlComparisonProtocolTest, SelectedPairRequiresEveryPhysicalPartAndOwnedOutput) {
  namespace fmt = llmp::format;
  const auto operation = ir::Fused(ir::FusedKind::kPreparedProducts);
  auto registry = llmp::execution::Registry::Create({{.name = "test.selected.pair",
                                                      .operation = operation,
                                                      .source = "test",
                                                      .revision = "one",
                                                      .build = "cpu",
                                                      .variant = "pair"}});
  ASSERT_TRUE(registry);
  ComparisonSelectedSource source, foreign_source;
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
      ggml_init({.mem_size = 128 * 1024, .mem_buffer = nullptr, .no_alloc = true}), ggml_free);
  ASSERT_TRUE(context);
  kg::LoweredStep lowered;
  lowered.comparison_selected_graph = kg::LoweredStep::SelectedComparisonGraph{};
  ASSERT_FALSE(lowered.comparison_origins);
  auto& relation = *lowered.comparison_selected_graph;
  relation.planned_tensors.resize(5);
  const fmt::TensorFormat weight{.encoding = fmt::Encoding::kExl3Trellis,
                                 .params = {.k_bits = 4, .codebook = fmt::Exl3Codebook::kMcg},
                                 .layout = fmt::Layout::kExl3Tiles,
                                 .shape = ir::ShapeOf({128, 128})};
  std::vector<ggml_tensor*> inputs;
  const auto record = [&](ggml_tensor* tensor, ir::TensorId id, ir::NodeId producer,
                          std::optional<fmt::PartRole> part,
                          std::optional<fmt::TensorFormat> format) {
    lowered.physical_tensors.push_back(
        {.descriptor = tensor,
         .id = static_cast<std::uint32_t>(lowered.physical_tensors.size()),
         .tensor = id,
         .root = id,
         .producer = producer,
         .part = part,
         .ir_format = format});
  };
  for (std::uint32_t i = 0; i < 2; ++i) {
    relation.planned_tensors[i].format = weight;
    const auto parts = fmt::PartsOf(weight);
    ASSERT_TRUE(parts);
    for (const auto& part : parts->parts()) {
      auto* tensor =
          ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, static_cast<std::int64_t>(part.readable));
      inputs.push_back(tensor);
      record(tensor, ir::TensorId{i}, {}, part.role, weight);
    }
  }
  for (std::uint32_t i = 2; i < 5; ++i) {
    auto& desc = relation.planned_tensors[i];
    desc.format =
        ir::Dense(i == 2 ? fmt::Encoding::kF16 : fmt::Encoding::kF32, ir::ShapeOf({128, 1}));
    desc.nb = ir::NaturalStrides(desc.format);
  }
  auto* input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, 128, 1);
  inputs.push_back(input);
  record(input, ir::TensorId{2}, {}, {}, relation.planned_tensors[2].format);
  auto* marker = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 256);
  marker->op = GGML_OP_CUSTOM;
  ASSERT_LE(inputs.size(), GGML_MAX_SRC);
  std::copy(inputs.begin(), inputs.end(), marker->src);
  record(marker, {}, ir::NodeId{0}, {}, {});
  std::array<ggml_tensor*, 2> outputs{};
  for (std::uint32_t i = 0; i < 2; ++i) {
    outputs[i] = ggml_view_2d(context.get(), marker, 128, 1, 512, i * 512);
    record(outputs[i], ir::TensorId{3 + i}, ir::NodeId{0}, {},
           relation.planned_tensors[3 + i].format);
  }
  ir::Node node{};
  node.key = operation;
  node.nsrc = 3;
  node.src[0] = ir::TensorId{0};
  node.src[1] = ir::TensorId{1};
  node.src[2] = ir::TensorId{2};
  node.nout = 2;
  node.out[0] = ir::TensorId{3};
  node.out[1] = ir::TensorId{4};
  relation.planned_nodes.push_back(node);
  lowered.selected.push_back({.tensor = marker,
                              .implementation = &registry->at(0),
                              .call = std::make_unique<ComparisonSelectedCall>(&source)});
  lowered.plan.steps.push_back(
      {.operation = operation, .implementation = "test.selected.pair", .nodes = {marker}});
  std::ranges::sort(lowered.physical_tensors, {}, &kg::PhysicalTensor::descriptor);
  const auto identity = [&] {
    return kg::ComparisonSelectedScheduleIdentity(lowered, *registry, source, 256);
  };
  ASSERT_TRUE(identity());
  EXPECT_FALSE(kg::ComparisonSelectedScheduleIdentity(lowered, *registry, source, 255));
  EXPECT_FALSE(kg::ComparisonSelectedScheduleIdentity(lowered, *registry, foreign_source, 256));
  auto* saved = marker->src[1];
  marker->src[1] = nullptr;
  EXPECT_FALSE(identity());
  marker->src[1] = marker->src[0];
  EXPECT_FALSE(identity());
  marker->src[1] = saved;
  marker->op_params[0] = 1;
  EXPECT_FALSE(identity());
  marker->op_params[0] = 0;
  outputs[1]->view_offs = 0;
  EXPECT_FALSE(identity());
  outputs[1]->view_offs = 512;
  lowered.plan.steps[0].lane = 1;
  EXPECT_FALSE(identity());
  lowered.plan.steps[0].lane = 0;
  EXPECT_TRUE(identity());
}

TEST(GgmlComparisonProtocolTest, DeviceIdentityUsesAllTypedImmutableFacts) {
  const kn::ComparisonDeviceFacts original{.device_class = "sm_121",
                                           .device_model = "GB10",
                                           .arch = 121,
                                           .features = 3,
                                           .multiprocessors = 48};
  const auto identity = kn::ComparisonDeviceIdentity(original);
  EXPECT_EQ(identity, kn::ComparisonDeviceIdentity(original));
  auto changed = original;
  changed.device_class = "sm_120";
  EXPECT_NE(identity, kn::ComparisonDeviceIdentity(changed));
  changed = original;
  changed.device_model = "GB10-rev2";
  EXPECT_NE(identity, kn::ComparisonDeviceIdentity(changed));
  changed = original;
  ++changed.arch;
  EXPECT_NE(identity, kn::ComparisonDeviceIdentity(changed));
  changed = original;
  ++changed.features;
  EXPECT_NE(identity, kn::ComparisonDeviceIdentity(changed));
  changed = original;
  ++changed.multiprocessors;
  EXPECT_NE(identity, kn::ComparisonDeviceIdentity(changed));
}

TEST(GgmlComparisonDispatchTest, LocalMmvqIdentityExcludesOnlyCudaOrdinal) {
  const kg::MmvqDispatchDescriptor actual{.device = 0,
                                          .compute_capability = 121,
                                          .parameter_table = 4,
                                          .warp_size = 32,
                                          .weight_type = GGML_TYPE_Q4_K,
                                          .columns = 2,
                                          .k = 2304,
                                          .padded_k = 2560,
                                          .row_low = 0,
                                          .row_high = 64,
                                          .stride_col_dst = 64,
                                          .rows_per_block = 1,
                                          .nwarps = 4,
                                          .grid_x = 1,
                                          .grid_y = 1,
                                          .grid_z = 1,
                                          .block_x = 128,
                                          .block_y = 1,
                                          .block_z = 1,
                                          .dynamic_shared_bytes = 0,
                                          .static_shared_bytes = 1024,
                                          .small_k = false,
                                          .halve_iters = false,
                                          .immutable_weights = true,
                                          .pdl_enabled = false,
                                          .immutable_prefetch = false,
                                          .streaming_prefetch = true,
                                          .fused = false};
  const auto local = kg::ComparisonLocalMmvqDispatchDigest(actual);
  const auto whole = kg::ComparisonMmvqDispatchDigest(actual);
  auto another_ordinal = actual;
  ++another_ordinal.device;
  EXPECT_EQ(local, kg::ComparisonLocalMmvqDispatchDigest(another_ordinal));
  EXPECT_NE(whole, kg::ComparisonMmvqDispatchDigest(another_ordinal));
  auto another_table = actual;
  ++another_table.parameter_table;
  EXPECT_NE(local, kg::ComparisonLocalMmvqDispatchDigest(another_table));
}

TEST(GgmlComparisonDispatchTest, DefaultRegionAccessAlignmentUsesGgmlTypeSize) {
  EXPECT_EQ(kg::ComparisonRegionDefaultAccessAlignment(GGML_TYPE_F32),
            ggml_type_size(GGML_TYPE_F32));
  EXPECT_EQ(kg::ComparisonRegionDefaultAccessAlignment(GGML_TYPE_Q4_K),
            ggml_type_size(GGML_TYPE_Q4_K));
}

TEST(GgmlComparisonDispatchTest, SplitF16ProductUsesActualGluAliasingAndEdges) {
  alignas(16) std::array<std::uint16_t, 128 * 128> gate_weights{}, up_weights{};
  alignas(16) std::array<float, 128> features{}, gate_values{}, up_values{}, output{};
  const auto tensor = [](ggml_type type, void* data, std::int64_t width, std::int64_t rows = 1) {
    ggml_tensor result{};
    result.type = type;
    result.data = data;
    result.ne[0] = width;
    result.ne[1] = rows;
    result.ne[2] = result.ne[3] = 1;
    result.nb[0] = ggml_type_size(type);
    result.nb[1] = result.nb[0] * static_cast<std::size_t>(width);
    result.nb[2] = result.nb[3] = result.nb[1] * static_cast<std::size_t>(rows);
    return result;
  };
  auto x = tensor(GGML_TYPE_F32, features.data(), 128);
  auto wg = tensor(GGML_TYPE_F16, gate_weights.data(), 128, 128);
  auto wu = tensor(GGML_TYPE_F16, up_weights.data(), 128, 128);
  auto gate = tensor(GGML_TYPE_F32, gate_values.data(), 128);
  auto up = tensor(GGML_TYPE_F32, up_values.data(), 128);
  gate.op = up.op = GGML_OP_MUL_MAT;
  gate.src[0] = &wg;
  up.src[0] = &wu;
  gate.src[1] = up.src[1] = &x;
  auto glu = tensor(GGML_TYPE_F32, output.data(), 128);
  glu.op = GGML_OP_GLU;
  glu.op_params[0] = GGML_GLU_OP_SWIGLU;
  glu.src[0] = &gate;
  glu.src[1] = &up;
  EXPECT_TRUE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  // Split GLU may reuse dead original Features or a product exactly in place.
  glu.data = features.data();
  EXPECT_TRUE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  glu.data = gate_values.data();
  EXPECT_TRUE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  glu.data = gate_values.data() + 1;
  EXPECT_FALSE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  glu.data = output.data();
  glu.src[0] = &up;
  glu.src[1] = &gate;
  EXPECT_FALSE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  glu.src[0] = &gate;
  glu.src[1] = &up;
  glu.op_params[0] = GGML_GLU_OP_GEGLU;
  EXPECT_FALSE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  glu.op_params[0] = GGML_GLU_OP_SWIGLU;
  glu.op_params[3] = 1;
  EXPECT_FALSE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  glu.op_params[3] = 0;
  gate.op_params[0] = GGML_PREC_F32;
  EXPECT_FALSE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  gate.op_params[0] = GGML_PREC_DEFAULT;
  up.src[1] = &gate;
  EXPECT_FALSE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
  up.src[1] = nullptr;
  EXPECT_FALSE(kg::CheckComparisonF16SwiGlu(&gate, &up, &glu));
}

TEST(GgmlComparisonDispatchTest, F16BiasFusionRequiresDefaultPrecisionAndSafeActualRanges) {
  alignas(16) std::array<std::uint16_t, 128 * 128> gate_weights{}, up_weights{};
  alignas(16) std::array<float, 128> features{}, gate_values{}, up_values{}, output{};
  const auto tensor = [](ggml_type type, void* data, std::int64_t width, std::int64_t rows = 1) {
    ggml_tensor result{};
    result.type = type;
    result.data = data;
    result.ne[0] = width;
    result.ne[1] = rows;
    result.ne[2] = result.ne[3] = 1;
    result.nb[0] = ggml_type_size(type);
    result.nb[1] = result.nb[0] * static_cast<std::size_t>(width);
    result.nb[2] = result.nb[3] = result.nb[1] * static_cast<std::size_t>(rows);
    return result;
  };
  auto x = tensor(GGML_TYPE_F32, features.data(), 128);
  auto wg = tensor(GGML_TYPE_F16, gate_weights.data(), 128, 128);
  auto wu = tensor(GGML_TYPE_F16, up_weights.data(), 128, 128);
  auto gate = tensor(GGML_TYPE_F32, gate_values.data(), 128);
  auto up = tensor(GGML_TYPE_F32, up_values.data(), 128);
  gate.op = up.op = GGML_OP_MUL_MAT;
  gate.src[0] = &wg;
  up.src[0] = &wu;
  gate.src[1] = up.src[1] = &x;
  auto add = tensor(GGML_TYPE_F32, output.data(), 128);
  add.op = GGML_OP_ADD;
  add.src[0] = &gate;
  add.src[1] = &up;
  up.op = GGML_OP_NONE;
  EXPECT_TRUE(kg::CheckComparisonMatMulAdd(&gate, &add));
  gate.op_params[0] = GGML_PREC_F32;
  EXPECT_FALSE(kg::CheckComparisonMatMulAdd(&gate, &add));
  gate.op_params[0] = GGML_PREC_DEFAULT;
  wg.type = GGML_TYPE_BF16;
  EXPECT_FALSE(kg::CheckComparisonMatMulAdd(&gate, &add));
  wg.type = GGML_TYPE_F16;
  // The fused kernel may overwrite the bias exactly, never partially.
  add.data = up_values.data();
  EXPECT_TRUE(kg::CheckComparisonMatMulAdd(&gate, &add));
  add.data = up_values.data() + 1;
  EXPECT_FALSE(kg::CheckComparisonMatMulAdd(&gate, &add));
  add.data = features.data();
  EXPECT_FALSE(kg::CheckComparisonMatMulAdd(&gate, &add));
  add.data = output.data();
  add.op_params[0] = 1;
  EXPECT_FALSE(kg::CheckComparisonMatMulAdd(&gate, &add));
  add.op_params[0] = 0;
  gate.src[1] = nullptr;
  EXPECT_FALSE(kg::CheckComparisonMatMulAdd(&gate, &add));
}

TEST(GgmlComparisonDispatchTest, ExplicitAccessInventoryPreservesScopedBinaryDomains) {
  auto* context = ggml_init({.mem_size = 1U << 20U, .mem_buffer = nullptr, .no_alloc = false});
  ASSERT_NE(context, nullptr);
  auto* a = ggml_new_tensor_1d(context, GGML_TYPE_F16, 128);
  auto* b = ggml_new_tensor_1d(context, GGML_TYPE_F16, 128);
  auto* add = ggml_add(context, a, b);
  kg::PlanStep step{
      .operation = ir::Primitive(ir::Op::kAdd), .implementation = "ggml.add", .nodes = {add}};
  const auto whole = kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kWhole);
  ASSERT_TRUE(whole);
  ASSERT_EQ(whole->count, 3U);
  EXPECT_EQ(whole->values[0].tensor, a);
  EXPECT_TRUE(whole->values[2].written);
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  step.operation = ir::Primitive(ir::Op::kMul);
  step.implementation = "ggml.mul";
  auto* mul = ggml_mul(context, a, b);
  step.nodes = {mul};
  EXPECT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kWhole));
  EXPECT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  step.operation = ir::Primitive(ir::Op::kAdd);
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kWhole));
  step.operation = ir::Primitive(ir::Op::kMul);
  step.nodes.push_back(mul);
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kWhole));
  step.nodes = {mul};
  step.implementation = "not-a-concrete-recipe";
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kWhole));
  auto* x = ggml_new_tensor_1d(context, GGML_TYPE_F32, 128);
  auto* gamma = ggml_new_tensor_1d(context, GGML_TYPE_F32, 128);
  auto* norm = ggml_rms_norm(context, x, 1e-6F);
  auto* scaled = ggml_mul(context, norm, gamma);
  step = {.operation = ir::Fused(ir::FusedKind::kRmsNormMul),
          .implementation = "ggml.rms_norm_mul.unfused",
          .nodes = {norm, scaled}};
  const auto ordered = kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal);
  ASSERT_TRUE(ordered);
  ASSERT_EQ(ordered->count, 5U);
  EXPECT_EQ(ordered->values[1].tensor, norm);
  EXPECT_TRUE(ordered->values[1].written);
  EXPECT_EQ(ordered->values[1].phase, 0U);
  EXPECT_EQ(ordered->values[2].tensor, norm);
  EXPECT_TRUE(ordered->values[2].read);
  EXPECT_EQ(ordered->values[2].phase, 1U);
  EXPECT_EQ(ordered->elided_count, 0U);
  step.implementation = "ggml.rms_norm_mul.fused";
  const auto fused = kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal);
  ASSERT_TRUE(fused);
  EXPECT_EQ(fused->count, 3U);
  ASSERT_EQ(fused->elided_count, 1U);
  EXPECT_EQ(fused->elided[0], norm);
  ggml_free(context);
}

TEST(GgmlComparisonDispatchTest, DirectSgemmDescriptorDomainRejectsConversionAndBatches) {
  auto* context = ggml_init({.mem_size = 1U << 20U, .mem_buffer = nullptr, .no_alloc = false});
  ASSERT_NE(context, nullptr);
  auto* weights = ggml_new_tensor_2d(context, GGML_TYPE_F32, 64, 48);
  auto* input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 64, 4);
  auto* output = ggml_mul_mat(context, weights, input);
  kg::PlanStep step{.operation = ir::Primitive(ir::Op::kMatMul),
                    .implementation = kg::kMulMatCublas,
                    .nodes = {output}};
  EXPECT_TRUE(kg::ComparisonSgemmCall(step));
  const auto original_type = weights->type;
  weights->type = GGML_TYPE_F16;
  EXPECT_FALSE(kg::ComparisonSgemmCall(step));
  weights->type = original_type;
  output->op_params[2] = 1;
  EXPECT_FALSE(kg::ComparisonSgemmCall(step));
  output->op_params[2] = 0;
  const auto original_stride = input->nb[1];
  input->nb[1] += sizeof(float);
  EXPECT_FALSE(kg::ComparisonSgemmCall(step));
  input->nb[1] = original_stride;
  auto* batch_weights = ggml_new_tensor_3d(context, GGML_TYPE_F32, 64, 48, 2);
  auto* batch_input = ggml_new_tensor_3d(context, GGML_TYPE_F32, 64, 4, 2);
  step.nodes = {ggml_mul_mat(context, batch_weights, batch_input)};
  EXPECT_FALSE(kg::ComparisonSgemmCall(step));
  step.nodes = {output};
  step.implementation = kg::kMulMatVector;
  EXPECT_FALSE(kg::ComparisonSgemmCall(step));
  ggml_free(context);
}

TEST(GgmlComparisonDispatchTest, GdnGatesAccessesOwnBothPlanesAndRejectUnsafeDescriptors) {
  auto* context = ggml_init({.mem_size = 1U << 20U, .mem_buffer = nullptr, .no_alloc = false});
  ASSERT_NE(context, nullptr);
  auto* alpha = ggml_new_tensor_2d(context, GGML_TYPE_F32, 48, 3);
  auto* beta = ggml_new_tensor_2d(context, GGML_TYPE_F32, 48, 3);
  auto* bias = ggml_new_tensor_1d(context, GGML_TYPE_F32, 48);
  auto* a = ggml_new_tensor_1d(context, GGML_TYPE_F32, 48);
  auto* gates = kg::GdnGates(context, alpha, beta, bias, a);
  ASSERT_NE(gates, nullptr);
  kg::PlanStep step{.operation = ir::Fused(ir::FusedKind::kGdnGates),
                    .implementation = kg::kGdnGatesName,
                    .nodes = {gates}};
  for (const auto scope : {kg::ComparisonAccessScope::kWhole, kg::ComparisonAccessScope::kLocal}) {
    const auto facts = kg::DescribeComparisonAccess(step, scope);
    ASSERT_TRUE(facts);
    ASSERT_EQ(facts->count, 5U);
    for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(facts->values[i].tensor, gates->src[i]);
      EXPECT_TRUE(facts->values[i].read);
      EXPECT_FALSE(facts->values[i].written);
    }
    EXPECT_EQ(facts->values[4].tensor, gates);
    EXPECT_EQ(facts->values[4].bytes, 2U * 48 * 3 * sizeof(float));
    EXPECT_FALSE(facts->values[4].read);
    EXPECT_TRUE(facts->values[4].written);
  }
  const auto saved = gates->data;
  gates->data = alpha->data;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  gates->data = saved;
  gates->src[4] = a;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  gates->src[4] = nullptr;
  auto* words =
      reinterpret_cast<std::int32_t*>(reinterpret_cast<std::byte*>(gates->op_params) + 32);
  words[0] = 1;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  words[0] = 0;
  step.operation = ir::Primitive(ir::Op::kUnary);
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  auto* softplus = ggml_softplus(context, alpha);
  EXPECT_TRUE(kg::HasTypedDescriptorSchema(*softplus));
  softplus->op_params[1] = 1;
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*softplus));
  ggml_free(context);
}

TEST(GgmlComparisonDispatchTest, HcCombineAccessesRetainExactOperandsAndFloat4Alignment) {
  auto* context = ggml_init({.mem_size = 1U << 20U, .mem_buffer = nullptr, .no_alloc = false});
  ASSERT_NE(context, nullptr);
  auto* streams = ggml_new_tensor_3d(context, GGML_TYPE_F32, 64, 4, 2);
  auto* block = ggml_new_tensor_2d(context, GGML_TYPE_F32, 64, 2);
  auto* logits = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 2);
  auto* combine = kg::HcCombine(context, streams, block, logits);
  ASSERT_NE(combine, nullptr);
  kg::PlanStep step{.operation = ir::Fused(ir::FusedKind::kHcCombine),
                    .implementation = kg::kHcCombineName,
                    .nodes = {combine}};
  for (const auto scope : {kg::ComparisonAccessScope::kWhole, kg::ComparisonAccessScope::kLocal}) {
    const auto facts = kg::DescribeComparisonAccess(step, scope);
    ASSERT_TRUE(facts);
    ASSERT_EQ(facts->count, 4U);
    for (std::size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(facts->values[i].tensor, combine->src[i]);
      EXPECT_EQ(facts->values[i].bytes, ggml_nbytes(combine->src[i]));
      EXPECT_EQ(facts->values[i].alignment, i == 2 ? 4U : 16U);
      EXPECT_TRUE(facts->values[i].read);
      EXPECT_FALSE(facts->values[i].written);
    }
    EXPECT_EQ(facts->values[3].tensor, combine);
    EXPECT_EQ(facts->values[3].alignment, 16U);
    EXPECT_FALSE(facts->values[3].read);
    EXPECT_TRUE(facts->values[3].written);
  }
  const auto saved = combine->data;
  combine->data = streams->data;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  combine->data = static_cast<std::byte*>(saved) + sizeof(float);
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  combine->data = saved;
  combine->src[3] = logits;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  combine->src[3] = nullptr;
  auto* words =
      reinterpret_cast<std::int32_t*>(reinterpret_cast<std::byte*>(combine->op_params) + 32);
  words[0] = 1;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  words[0] = 0;
  combine->type = GGML_TYPE_BF16;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  combine->type = GGML_TYPE_F32;
  step.operation = ir::Primitive(ir::Op::kHcPost);
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  ggml_free(context);
}

TEST(GgmlComparisonDispatchTest, HcPostNormAccessesWriteBothOutputsWithoutReadingTheFlat) {
  auto* context = ggml_init({.mem_size = 4U << 20U, .mem_buffer = nullptr, .no_alloc = false});
  ASSERT_NE(context, nullptr);
  auto* y = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4096, 2);
  auto* x = ggml_new_tensor_3d(context, GGML_TYPE_F32, 4096, 4, 2);
  auto* weights = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 2);
  auto* comb = ggml_new_tensor_3d(context, GGML_TYPE_F32, 4, 4, 2);
  auto* post = ggml_dsv4_hc_post(context, y, x, weights, comb);
  auto* flat = ggml_reshape_2d(context, post, 16384, 2);
  auto* norm = kg::Dsv4HcNormF16(context, flat, 1e-6F, GGML_TYPE_F32);
  kg::PlanStep step{.operation = ir::Fused(ir::FusedKind::kHcNormPost),
                    .implementation = kg::kDsv4HcPostNormF16Name,
                    .nodes = {post, norm}};
  for (auto scope : {kg::ComparisonAccessScope::kWhole, kg::ComparisonAccessScope::kLocal}) {
    auto facts = kg::DescribeComparisonAccess(step, scope);
    ASSERT_TRUE(facts);
    ASSERT_EQ(facts->count, 6U);
    for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(facts->values[i].tensor, post->src[i]);
      EXPECT_TRUE(facts->values[i].read);
      EXPECT_FALSE(facts->values[i].written);
    }
    EXPECT_EQ(facts->values[4].tensor, post);
    EXPECT_EQ(facts->values[5].tensor, norm);
    EXPECT_TRUE(facts->values[4].written);
    EXPECT_TRUE(facts->values[5].written);
    EXPECT_FALSE(facts->values[4].read);
    EXPECT_FALSE(facts->values[5].read);
  }
  auto* rounded = kg::Dsv4HcNormF16(context, flat, 1e-6F, GGML_TYPE_F16);
  step.nodes = {post, rounded};
  EXPECT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kWhole));
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  step.nodes = {post, norm};
  auto* saved = norm->data;
  norm->data = post->data;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  norm->data = saved;
  flat->ne[0] = 8192;
  flat->ne[1] = 4;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  flat->ne[0] = 16384;
  flat->ne[1] = 2;
  flat->view_offs = sizeof(float);
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  flat->view_offs = 0;
  auto* words = reinterpret_cast<std::int32_t*>(reinterpret_cast<std::byte*>(norm->op_params) + 32);
  words[1] = 1;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  words[1] = 0;
  step.operation = ir::Primitive(ir::Op::kHcPost);
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  EXPECT_FALSE(kg::HasComparisonAccessRecipe(kg::kDsv4HcPostExpertsNormF16Name));
  step.implementation = kg::kHcPostName;
  step.nodes = {post};
  ASSERT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  post->src[4] = y;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  ggml_free(context);
}

TEST(GgmlComparisonDispatchTest, SpatialAccessesRetainActualRangesAndAliasDomains) {
  auto* context = ggml_init({.mem_size = 4U << 20U, .mem_buffer = nullptr, .no_alloc = false});
  ASSERT_NE(context, nullptr);
  auto* x = ggml_new_tensor_3d(context, GGML_TYPE_F32, 37, 3, 7);
  auto* gamma = ggml_new_tensor_1d(context, GGML_TYPE_F32, 7);
  auto* spatial = kg::SpatialChannelNorm(context, x, gamma, 1e-12F, 2.6457513F, true);
  kg::PlanStep step{.operation = ir::Fused(ir::FusedKind::kSpatialChannelNorm),
                    .implementation = kg::kSpatialChannelNormName,
                    .nodes = {spatial}};
  for (auto scope : {kg::ComparisonAccessScope::kWhole, kg::ComparisonAccessScope::kLocal}) {
    auto facts = kg::DescribeComparisonAccess(step, scope);
    ASSERT_TRUE(facts);
    ASSERT_EQ(facts->count, 3U);
    EXPECT_EQ(facts->values[0].tensor, x);
    EXPECT_EQ(facts->values[0].bytes, ggml_nbytes(x));
    EXPECT_EQ(facts->values[1].tensor, gamma);
    EXPECT_TRUE(facts->values[2].written);
  }
  auto* saved = spatial->data;
  spatial->data = x->data;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  spatial->data = saved;
  spatial->src[2] = gamma;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  spatial->src[2] = nullptr;
  auto* words =
      reinterpret_cast<std::int32_t*>(reinterpret_cast<std::byte*>(spatial->op_params) + 32);
  words[3] = 1;
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*spatial));
  words[3] = 0;
  auto* permute = ggml_permute(context, x, 1, 2, 0, 3);
  EXPECT_TRUE(kg::HasTypedDescriptorSchema(*permute));
  permute->op_params[1] = permute->op_params[0];
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*permute));
  permute->op_params[1] = 2;
  permute->op_params[0] = 2;
  permute->op_params[1] = 1;
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*permute));
  permute->op_params[0] = 1;
  permute->op_params[1] = 2;
  permute->op_params[4] = 1;
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*permute));
  permute->op_params[4] = 0;
  permute->src[1] = gamma;
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*permute));
  permute->src[1] = nullptr;
  auto* unit = ggml_new_tensor_3d(context, GGML_TYPE_F32, 37, 3, 1);
  auto* unit_permute = ggml_permute(context, unit, 1, 2, 0, 3);
  unit_permute->nb[0] = sizeof(float);
  EXPECT_TRUE(kg::HasTypedDescriptorSchema(*unit_permute));
  unit_permute->nb[1] += sizeof(float);
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*unit_permute));
  auto* rows = ggml_cont(context, permute);
  step = {
      .operation = ir::Primitive(ir::Op::kCont), .implementation = "ggml.cont", .nodes = {rows}};
  ASSERT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  auto* l2 = ggml_l2_norm(context, rows, 1e-12F);
  step = {
      .operation = ir::Primitive(ir::Op::kL2Norm), .implementation = "ggml.l2_norm", .nodes = {l2}};
  ASSERT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  saved = l2->data;
  l2->data = rows->data;
  EXPECT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  l2->data = static_cast<std::byte*>(rows->data) + 4;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  l2->data = saved;
  l2->op_params[1] = 1;
  EXPECT_FALSE(kg::HasTypedDescriptorSchema(*l2));
  ggml_free(context);
}

TEST(GgmlComparisonDispatchTest, SectionsRopeAccessesSealPositionsGammaAndPartialCopies) {
  auto* context = ggml_init({.mem_size = 4U << 20U, .mem_buffer = nullptr, .no_alloc = false});
  ASSERT_NE(context, nullptr);
  auto* x = ggml_new_tensor_3d(context, GGML_TYPE_F32, 64, 3, 5);
  auto* gamma = ggml_new_tensor_1d(context, GGML_TYPE_BF16, 64);
  auto* p0 = ggml_new_tensor_1d(context, GGML_TYPE_I32, 5);
  auto* p1 = ggml_new_tensor_1d(context, GGML_TYPE_I32, 5);
  const std::array positions{p0, p1};
  const std::array<std::uint32_t, 2> dims{32, 32};
  auto* fused = kg::NormRopeSections(context, x, gamma, positions, 1e-6F, 10000.0F, false, dims,
                                     GGML_TYPE_F32);
  kg::PlanStep step{.operation = ir::Fused(ir::FusedKind::kRmsNormMulRopeSections),
                    .implementation = kg::kNormRopeSectionsName,
                    .nodes = {fused}};
  auto facts = kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal);
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->count, 5U);
  EXPECT_EQ(facts->values[0].tensor, x);
  EXPECT_EQ(facts->values[1].tensor, gamma);
  EXPECT_EQ(facts->values[2].tensor, p0);
  EXPECT_EQ(facts->values[3].tensor, p1);
  EXPECT_EQ(facts->values[4].tensor, fused);
  EXPECT_TRUE(facts->values[4].written);
  EXPECT_EQ(facts->values[1].alignment, 2U);
  auto* bf16_x = ggml_new_tensor_3d(context, GGML_TYPE_BF16, 64, 3, 5);
  auto* rounded = kg::NormRopeSections(context, bf16_x, gamma, positions, 1e-6F, 10000.0F, false,
                                       dims, GGML_TYPE_BF16);
  ASSERT_TRUE(kg::CheckNormRopeSections(rounded));
  step.nodes = {rounded};
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  step.nodes = {fused};
  auto* saved_position = fused->src[3];
  fused->src[3] = nullptr;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  fused->src[3] = saved_position;
  fused->src[7] = p0;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  fused->src[7] = nullptr;
  const auto saved_data = fused->data;
  fused->data = x->data;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  fused->data = saved_data;
  // The constructor's custom record ends at seven semantic words. Unused
  // section dimensions and the final reserved word are not authority.
  constexpr std::size_t kCustomSemanticOffset = 32;
  auto* words = reinterpret_cast<std::int32_t*>(reinterpret_cast<std::byte*>(fused->op_params) +
                                                kCustomSemanticOffset);
  words[6] = 2;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  words[6] = 0;
  words[7] = 1;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  words[7] = 0;
  ASSERT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));

  auto* rope =
      ggml_rope_ext(context, x, p0, nullptr, 32, 0, 0, 10000.0F, 1.0F, 0.0F, 1.0F, 32.0F, 1.0F);
  ggml_rope_set_offset(rope, 32);
  step = {.operation = ir::Primitive(ir::Op::kRope),
          .implementation = kg::kRopeExtName,
          .nodes = {rope}};
  const auto rotated = kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal);
  ASSERT_TRUE(rotated);
  ASSERT_EQ(rotated->count, 3U);
  EXPECT_EQ(rotated->values[0].bytes, static_cast<std::uint64_t>(ggml_nbytes(x)));
  EXPECT_EQ(rotated->values[2].bytes, static_cast<std::uint64_t>(ggml_nbytes(rope)));
  EXPECT_EQ(rotated->values[2].alignment, 8U);
  const auto rope_data = rope->data;
  rope->data = x->data;
  EXPECT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  rope->data = static_cast<std::byte*>(x->data) + 8;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  rope->data = rope_data;
  rope->op_params[15] = 34;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  rope->op_params[15] = 32;
  rope->op_params[11] = 1;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  rope->op_params[11] = 0;

  auto* frequencies = ggml_new_tensor_1d(context, GGML_TYPE_F32, 32);
  auto* frequency_rope = ggml_rope_ext(context, x, p0, frequencies, 64, GGML_ROPE_TYPE_NEOX, 4096,
                                       10000.0F, 1.0F, 0.0F, 1.0F, 32.0F, 1.0F);
  step.nodes = {frequency_rope};
  const auto frequency_facts =
      kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal);
  ASSERT_TRUE(frequency_facts);
  ASSERT_EQ(frequency_facts->count, 4U);
  EXPECT_EQ(frequency_facts->values[2].tensor, frequencies);

  frequency_rope->src[2] = nullptr;
  frequency_rope->op_params[6] = std::bit_cast<std::int32_t>(2.0F);
  EXPECT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kWhole));
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));

  auto* dst = ggml_new_tensor_1d(context, GGML_TYPE_F32, 64);
  auto* convert = ggml_cpy(context, gamma, dst);
  step = {.operation = ir::Primitive(ir::Op::kConvert),
          .implementation = "ggml.convert",
          .nodes = {convert}};
  const auto widened = kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal);
  ASSERT_TRUE(widened);
  ASSERT_EQ(widened->count, 2U);
  EXPECT_EQ(widened->values[0].tensor, gamma);
  EXPECT_TRUE(widened->values[0].read);
  EXPECT_EQ(widened->values[1].tensor, dst);
  EXPECT_TRUE(widened->values[1].written);
  EXPECT_FALSE(widened->values[1].read);
  const auto* actual_destination = convert->view_src;
  convert->view_src = x;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  convert->view_src = const_cast<ggml_tensor*>(actual_destination);
  convert->src[7] = p0;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  convert->src[7] = nullptr;
  const auto stride = convert->nb[0];
  convert->nb[0] *= 2;
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  convert->nb[0] = stride;
  auto* contiguous = ggml_cont(context, ggml_new_tensor_3d(context, GGML_TYPE_F16, 64, 3, 5));
  step = {.operation = ir::Primitive(ir::Op::kCont),
          .implementation = "ggml.cont",
          .nodes = {contiguous}};
  EXPECT_TRUE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kWhole));
  EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
  ggml_free(context);
}

TEST(GgmlComparisonDispatchTest, SplitAccessesRecordRealDispatchSecondaryWritesAndLiveAliases) {
  auto* context = ggml_init({.mem_size = 4U << 20U, .mem_buffer = nullptr, .no_alloc = false});
  ASSERT_NE(context, nullptr);
  for (const std::int64_t width : {4, 4096, 8193}) {
    auto* x = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, 1);
    auto* branch = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, 1);
    auto* ga = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
    auto* gb = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
    auto* ma = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
    auto* mb = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
    auto* residual = kg::GatedResidualSplit(context, x, branch, ga, gb, 0);
    auto* modulated =
        kg::LayerNormModulateSplit(context, residual, ma, mb, 1e-6F, 0, GGML_TYPE_F32);
    kg::MarkJoined(modulated);
    kg::PlanStep step{.operation = ir::Fused(ir::FusedKind::kGatedResidualModulateSplit),
                      .implementation = kg::kGatedResidualModulateSplitName,
                      .nodes = {residual, modulated}};
    const auto facts = kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal);
    ASSERT_TRUE(facts) << width;
    EXPECT_EQ(facts->count, width > 8192 ? 9U : 8U);
    EXPECT_EQ(facts->values[4].tensor, residual);
    EXPECT_TRUE(facts->values[4].written);
    EXPECT_EQ(facts->values[4].phase, 0U);
    if (width > 8192) {
      EXPECT_EQ(facts->values[5].tensor, residual);
      EXPECT_TRUE(facts->values[5].read);
      EXPECT_EQ(facts->values[5].phase, 1U);
    }
    EXPECT_EQ(facts->values[facts->count - 1].tensor, modulated);
    EXPECT_TRUE(facts->values[facts->count - 1].written);
    EXPECT_EQ(*facts->dispatch, width == 4096 ? 1U : (width > 8192 ? 2U : 0U));
    if (width == 4) {
      void* saved = residual->data;
      residual->data = ma->data;
      EXPECT_FALSE(kg::CheckGatedResidualModulateSplit(residual, modulated));
      EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
      residual->data = saved;
      const auto saved_word = modulated->op_params[15];
      modulated->op_params[15] = 1;
      EXPECT_FALSE(kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal));
      modulated->op_params[15] = saved_word;
    }
    if (width == 4096) {
      auto* backing = ggml_new_tensor_1d(context, GGML_TYPE_F32, width + 1);
      auto* misaligned = ggml_view_1d(context, backing, width, 4);
      modulated->src[1] = misaligned;
      const auto scalar = kg::DescribeComparisonAccess(step, kg::ComparisonAccessScope::kLocal);
      ASSERT_TRUE(scalar);
      EXPECT_EQ(*scalar->dispatch, 0U);
    }
  }
  ggml_free(context);
}

llmp::planner::StepPlan PlanFor(const ir::Graph& graph) {
  llmp::planner::StepPlan plan{.primitive_digest = {},
                               .graph_digest = {},
                               .policy_digest = {},
                               .graph = std::move(graph.Clone().value()),
                               .steps = {},
                               .regions = {},
                               .placement = {},
                               .applied = {},
                               .from = {},
                               .tensors = {},
                               .decomposed = {},
                               .missing_numerical = {},
                               .owner_row_invariant = false,
                               .owner_rows_policy_enabled = false,
                               .products = {},
                               .root_alignments = {},
                               .owner_products = {},
                               .owner_digest = {},
                               .invariant = {},
                               .prologue = {}};
  plan.tensors.resize(graph.tensor_count());
  for (std::uint32_t i = 0; i < graph.tensor_count(); ++i) plan.tensors[i] = ir::TensorId{i};
  return plan;
}

TEST(GgmlComparisonProtocolTest, WholeTopKIndicesRequireTheActualBoundedProducerAndConsumer) {
  namespace ex = llmp::execution;
  auto registry = ex::Registry::Create({{.name = std::string(kg::kTopKName),
                                         .operation = ir::Primitive(ir::Op::kTopK),
                                         .source = "test",
                                         .revision = "one",
                                         .build = "cpu",
                                         .variant = "descriptor-only"},
                                        {.name = std::string(kg::kSetRowsExtName),
                                         .operation = ir::Primitive(ir::Op::kCacheStore),
                                         .source = "test",
                                         .revision = "one",
                                         .build = "cpu",
                                         .variant = "descriptor-only"}});
  ASSERT_TRUE(registry);
  // Both the padded-bitonic and radix row classes, with more than one index row.
  for (const std::uint32_t columns : {7U, 1025U}) {
    for (const bool undersized : {false, true}) {
      SCOPED_TRACE(columns);
      SCOPED_TRACE(undersized);
      const std::uint32_t capacity = columns - (undersized ? 1 : 0);
      ir::Builder b;
      const auto scores =
          b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({columns, 3})));
      const auto indices = b.TopK(scores, 3);
      const auto destination =
          b.Fill(ir::ShapeOf({1, capacity, 3}), -std::numeric_limits<float>::infinity());
      const auto values = b.Fill(ir::ShapeOf({1, 3, 3}), 0.F);
      const auto output = b.CacheStore(destination, values, indices);
      b.Output(output);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph);
      auto plan = PlanFor(*graph);
      kn::Addresses addresses{};
      const kn::LowerInput input(plan, {*graph, graph->inputs(), graph->outputs(), {}}, addresses);
      std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
          ggml_init({.mem_size = 128 * 1024, .mem_buffer = nullptr, .no_alloc = true}), ggml_free);
      ASSERT_TRUE(context);
      auto* c = context.get();
      auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, columns, 3);
      auto* ids = ggml_top_k(c, x, 3);
      auto* dst = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, capacity, 3);
      auto* val = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 3, 3);
      auto* store = ggml_set_rows(c, dst, val, ids);
      x->data = reinterpret_cast<void*>(0x100000);
      ids->data = reinterpret_cast<void*>(0x200000);
      dst->data = store->data = reinterpret_cast<void*>(0x300000);
      val->data = reinterpret_cast<void*>(0x400000);
      kg::LoweredStep lowered;
      const std::array<std::pair<ggml_tensor*, ir::TensorId>, 5> tensors{
          std::pair{x, scores}, {ids, indices}, {dst, destination}, {val, values}, {store, output}};
      for (const auto& [native, typed] : tensors)
        lowered.physical_tensors.push_back(
            {.descriptor = native,
             .id = static_cast<std::uint32_t>(lowered.physical_tensors.size()),
             .tensor = typed,
             .root = graph->root(typed),
             .producer = graph->producer(typed),
             .ir_format = graph->tensor(typed).format});
      std::ranges::sort(lowered.physical_tensors, std::less<const ggml_tensor*>{},
                        &kg::PhysicalTensor::descriptor);
      lowered.plan.steps = {{.operation = ir::Primitive(ir::Op::kTopK),
                             .implementation = kg::kTopKName,
                             .nodes = {ids}},
                            {.operation = ir::Primitive(ir::Op::kCacheStore),
                             .implementation = kg::kSetRowsExtName,
                             .nodes = {store}}};
      const auto checked = [&] {
        return kg::CheckComparisonTopKIndices(input, lowered, *registry, store);
      };
      if (undersized) {
        // All source/physical identities and layouts agree, but TopK can emit a
        // valid source column beyond this actual destination's row capacity.
        EXPECT_FALSE(checked());
        continue;
      }
      ASSERT_TRUE(checked());
      // Oversized producer columns cannot authorize indices for this destination.
      ++x->ne[0];
      EXPECT_FALSE(checked());
      --x->ne[0];
      auto* foreign = ggml_new_tensor_2d(c, GGML_TYPE_I32, 3, 3);
      foreign->data = reinterpret_cast<void*>(0x500000);
      store->src[1] = foreign;
      EXPECT_FALSE(checked());
      store->src[1] = ids;
      ids->view_src = foreign;
      EXPECT_FALSE(checked());
      ids->view_src = nullptr;
      ids->op_params[0] = 1;
      EXPECT_FALSE(checked());
      ids->op_params[0] = 0;
      ids->src[1] = x;
      EXPECT_FALSE(checked());
      ids->src[1] = nullptr;
      auto& identity =
          *std::ranges::find(lowered.physical_tensors, ids, &kg::PhysicalTensor::descriptor);
      const auto original = identity;
      identity.tensor = scores;
      EXPECT_FALSE(checked());
      identity = original;
      identity.root = destination;
      EXPECT_FALSE(checked());
      identity = original;
      identity.producer = graph->producer(destination);
      EXPECT_FALSE(checked());
      identity = original;
      lowered.plan.steps[0].implementation = kg::kSetRowsExtName;
      EXPECT_FALSE(checked());
      lowered.plan.steps[0].implementation = kg::kTopKName;
      std::swap(lowered.plan.steps[0], lowered.plan.steps[1]);
      EXPECT_FALSE(checked());
      std::swap(lowered.plan.steps[0], lowered.plan.steps[1]);
      lowered.plan.steps[0].lane = 1;
      EXPECT_FALSE(checked());
      lowered.plan.steps[0].lane = 0;
      auto* overwrite = ggml_new_tensor_2d(c, GGML_TYPE_I32, 3, 3);
      overwrite->data = ids->data;
      lowered.plan.steps.insert(lowered.plan.steps.begin() + 1,
                                {.operation = ir::Primitive(ir::Op::kFill),
                                 .implementation = "ggml.fill",
                                 .nodes = {overwrite}});
      EXPECT_FALSE(checked());
      // A hidden/in-place writer can expose a disjoint nominal output while
      // retaining the index allocation as an operand. Reads also refuse.
      overwrite->data = reinterpret_cast<void*>(0x600000);
      overwrite->src[0] = ids;
      EXPECT_FALSE(checked());
      overwrite->src[0] = nullptr;
      ASSERT_TRUE(checked());
      lowered.plan.steps.erase(lowered.plan.steps.begin() + 1);
      ASSERT_TRUE(checked());
    }
  }
}

struct Fixture {
  ir::Graph graph;
  llmp::planner::StepPlan plan;
  kn::Addresses addresses;
  std::array<kn::ComparisonBindingRange, 1> ranges{};
  std::shared_ptr<ComparisonFunder> funder = std::make_shared<ComparisonFunder>();
  std::optional<co::OwnedComparisonBoundary> boundary;
  kn::ComparisonRequest request;
  kn::LowerInput input;

  Fixture()
      : graph([] {
          ir::Builder builder;
          const auto value =
              builder.Input(llmp::ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 4})));
          builder.Output(value);
          return std::move(builder).Finish().value();
        }()),
        plan(PlanFor(graph)),
        request{.mode = kn::ComparisonRunMode::kFull, .bindings = ranges},
        input(plan, {graph, graph.inputs(), graph.outputs(), {}}, addresses, std::nullopt,
              &request) {
    addresses.activations = 1;
    ranges[0] = {.original_root = graph.inputs().front(),
                 .allocation_class = 7,
                 .allocation_bytes = 4096,
                 .offset = 0,
                 .bytes = 1024,
                 .alignment = 16,
                 .base = 0x10000};
    const std::array<co::ComparisonExternalBinding, 1> external{
        co::ComparisonExternalBinding{.original_root = graph.inputs().front(),
                                      .allocation_class = ranges[0].allocation_class,
                                      .allocation_bytes = ranges[0].allocation_bytes,
                                      .offset = ranges[0].offset,
                                      .bytes = ranges[0].bytes,
                                      .alignment = ranges[0].alignment}};
    const co::ComparisonBoundaryView view{
        {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
    auto owned = co::MakeComparisonBoundary(view, funder);
    EXPECT_TRUE(owned) << (owned ? "" : owned.error().detail);
    if (owned) {
      boundary.emplace(std::move(*owned));
      request.expected_boundary = &*boundary;
    }
  }
};

TEST(GgmlComparisonTest, WholeStagedInputRelocationKeepsLogicalOwnerAndActualSlot) {
  for (const auto& [owned_bytes, physical_offset] :
       {std::pair<std::uint64_t, std::uint64_t>{272, 0}, {768, 0}, {768, 256}}) {
    SCOPED_TRACE(owned_bytes);
    SCOPED_TRACE(physical_offset);
    ir::Builder builder;
    const auto omitted = builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({17})));
    const auto positions =
        builder.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({68})));
    builder.Output(positions);
    auto source = std::move(builder).Finish().value();
    auto plan = PlanFor(source);
    plan.placement.alignment = 256;
    plan.placement.extent = 1024;
    plan.placement.offsets = {{omitted, 0}, {positions, 256}};
    kn::Addresses addresses{};
    addresses.activations = 0x100000;
    addresses.activation_bytes = 4096;
    addresses.activation_alignment = 256;
    std::array<kn::ComparisonBindingRange, 2> ranges{{{.original_root = omitted,
                                                       .allocation_class = 1,
                                                       .allocation_bytes = 68,
                                                       .bytes = 68,
                                                       .alignment = 256,
                                                       .base = addresses.activations},
                                                      {.original_root = positions,
                                                       .allocation_class = 2,
                                                       .allocation_bytes = 272,
                                                       .bytes = 272,
                                                       .alignment = 256,
                                                       .base = addresses.activations + 256}}};
    std::array<co::ComparisonExternalBinding, 2> external{};
    for (std::size_t i = 0; i < ranges.size(); ++i)
      external[i] = {.original_root = ranges[i].original_root,
                     .allocation_class = ranges[i].allocation_class,
                     .allocation_bytes = ranges[i].allocation_bytes,
                     .bytes = ranges[i].bytes,
                     .alignment = ranges[i].alignment};
    const co::GraphBuildIO io{source, source.inputs(), source.outputs(), {}};
    const co::ComparisonBoundaryView view{.io = io, .external = external};
    auto funder = std::make_shared<ComparisonFunder>();
    auto boundary = co::MakeComparisonBoundary(view, funder);
    ASSERT_TRUE(boundary);
    kn::ComparisonRequest request{.scope = kn::ComparisonProofScope::kWholeProgramNative,
                                  .bindings = ranges,
                                  .expected_boundary = &*boundary};
    const kn::LowerInput input(plan, io, addresses, std::nullopt, &request);
    ASSERT_TRUE(kg::ValidateComparisonBindings(input, ranges));
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({.mem_size = 16384, .mem_buffer = nullptr, .no_alloc = true}), ggml_free);
    ASSERT_TRUE(context);
    auto* tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 68);
    tensor->data = reinterpret_cast<void*>(addresses.activations + physical_offset);
    kg::LoweredStep lowered;
    lowered.leaves.push_back({.tensor = tensor, .primitive = positions});
    lowered.inputs.push_back({.device_address = addresses.activations + physical_offset,
                              .bytes = 272,
                              .tensor = positions,
                              .kind = kn::InputKind::kPositions,
                              .sections = 4});
    lowered.physical_tensors.push_back({.descriptor = tensor,
                                        .id = 0,
                                        .tensor = positions,
                                        .root = positions,
                                        .producer = plan.graph.producer(positions),
                                        .ir_format = plan.graph.tensor(positions).format});
    kg::PhysicalRoot root{};
    root.root = positions;
    root.producer = plan.graph.producer(positions);
    root.operation = ir::Primitive(ir::Op::kInput);
    root.backend_tensor = 0;
    root.offset = physical_offset;
    root.bytes = owned_bytes;
    root.encoding = llmp::format::Encoding::kI32;
    for (std::size_t d = 0; d < 4; ++d) {
      root.ne[d] = static_cast<std::uint64_t>(tensor->ne[d]);
      root.nb[d] = tensor->nb[d];
    }
    lowered.physical_roots.push_back(root);
    lowered.comparison_omitted_inputs.push_back(
        {.original = omitted, .planned = omitted, .producer = plan.graph.producer(omitted)});
    const auto describe = [&] {
      return kg::DescribeComparisonStagedInput(input, lowered, lowered.leaves[0], 1);
    };
    auto fact = describe();
    ASSERT_TRUE(fact);
    EXPECT_EQ(fact->logical_offset, 256);
    EXPECT_EQ(fact->physical_offset, physical_offset);
    EXPECT_EQ(fact->bytes, owned_bytes);
    EXPECT_EQ(fact->input_bytes, 272);
    // The omitted source declaration is retained; only the actual live slice moves.
    EXPECT_EQ(ranges[1].base, addresses.activations + 256);
    EXPECT_TRUE(kg::CheckComparisonOmittedInput(lowered, omitted));
    auto actual = ranges[1];
    actual.base = addresses.activations + physical_offset;
    actual.bytes = owned_bytes;
    actual.allocation_bytes = owned_bytes;
    lowered.leaves[0].comparison_staged = *fact;
    ASSERT_TRUE(kg::CheckComparisonStagedInput(lowered, lowered.leaves[0], actual));
    const auto saved_range = ranges[1];
    const auto saved_actual = actual;
    const auto saved_slot = lowered.inputs[0];
    const auto saved_descriptor = lowered.physical_tensors[0];
    for (unsigned mutation = 0; mutation < 16; ++mutation) {
      SCOPED_TRACE(mutation);
      if (mutation == 0) request.scope = kn::ComparisonProofScope::kTypedAccessClosure;
      if (mutation == 1) request.expected_boundary = nullptr;
      if (mutation == 2) ++ranges[1].base;
      if (mutation == 3) ranges[1].allocation_class = ranges[0].allocation_class;
      if (mutation == 4) ranges[1].allocation_bytes += 256;
      if (mutation == 5) ranges[1].part = llmp::format::PartRole::kData;
      if (mutation == 6) addresses.activation_bytes = 271;
      if (mutation == 7) ++lowered.inputs[0].device_address;
      if (mutation == 8) --lowered.inputs[0].bytes;
      if (mutation == 9) lowered.inputs[0].commit_save = true;
      if (mutation == 10) ++lowered.physical_roots[0].offset;
      if (mutation == 11) lowered.physical_tensors[0].producer = {};
      if (mutation == 12) tensor->type = GGML_TYPE_F32;
      if (mutation == 13) ++tensor->nb[1];
      if (mutation == 14) plan.tensors[positions.v] = omitted;
      if (mutation == 15) lowered.inputs.push_back(saved_slot);
      EXPECT_FALSE(describe());
      request.scope = kn::ComparisonProofScope::kWholeProgramNative;
      request.expected_boundary = &*boundary;
      ranges[1] = saved_range;
      addresses.activation_bytes = 4096;
      lowered.inputs = {saved_slot};
      lowered.physical_roots[0] = root;
      lowered.physical_tensors[0] = saved_descriptor;
      tensor->type = GGML_TYPE_I32;
      tensor->nb[1] = root.nb[1];
      plan.tensors[positions.v] = positions;
    }
    for (unsigned mutation = 0; mutation < 13; ++mutation) {
      SCOPED_TRACE(mutation);
      if (mutation == 0) actual.base += 256;
      if (mutation == 1) actual.allocation_class = 1;
      if (mutation == 2) lowered.inputs[0].tensor = omitted;
      if (mutation == 3) lowered.inputs[0].kind = kn::InputKind::kFeatures;
      if (mutation == 4) lowered.leaves[0].comparison_staged->activation_bytes = 271;
      if (mutation == 5)
        tensor->data = reinterpret_cast<void*>(addresses.activations + physical_offset + 256);
      if (mutation == 6) lowered.physical_roots[0].backend_tensor = 1;
      if (mutation == 7) {
        auto* other = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 17);
        other->data = reinterpret_cast<void*>(addresses.activations + physical_offset + 128);
        lowered.leaves.push_back({.tensor = other, .primitive = omitted});
      }
      if (mutation == 8) ++lowered.physical_roots[0].bytes;
      if (mutation == 9) --lowered.physical_roots[0].bytes;
      if (mutation == 10) lowered.leaves[0].comparison_staged->input_bytes = owned_bytes + 1;
      if (mutation == 11) {
        // A protected input cannot occupy even the auxiliary tail of the root.
        auto* other = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 1);
        other->data =
            reinterpret_cast<void*>(addresses.activations + physical_offset + owned_bytes - 4);
        lowered.leaves.push_back({.tensor = other, .primitive = omitted});
      }
      if (mutation == 12) {
        // The declared span cannot shrink to just the visible input.
        --actual.bytes;
        --actual.allocation_bytes;
        --lowered.leaves[0].comparison_staged->bytes;
      }
      EXPECT_FALSE(kg::CheckComparisonStagedInput(lowered, lowered.leaves[0], actual));
      actual = saved_actual;
      lowered.inputs[0] = saved_slot;
      lowered.leaves.resize(1);
      lowered.leaves[0].comparison_staged = *fact;
      lowered.physical_roots[0] = root;
      tensor->data = reinterpret_cast<void*>(addresses.activations + physical_offset);
    }
    ASSERT_TRUE(describe());
    EXPECT_TRUE(kg::CheckComparisonStagedInput(lowered, lowered.leaves[0], actual));
  }
}

TEST(GgmlComparisonTest, AcceptsOwnedAlignedInputRange) {
  Fixture fixture;
  EXPECT_TRUE(kg::ValidateComparisonBindings(fixture.input, fixture.ranges));
}

TEST(GgmlComparisonTest, RematerializedOutputBindingsRequireTheCompleteOwnedBoundary) {
  for (int mutation = 0; mutation < 9; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder source_builder;
    const auto value =
        source_builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({4, 2})));
    const auto root = mutation == 4 ? value : source_builder.Unary(value, ir::UnaryFn::kSigmoid);
    const auto output = mutation == 3
                            ? source_builder.View(root, ir::ShapeOf({1, 4, 1}), {4, 4, 16, 16}, 0)
                            : source_builder.Reshape(root, ir::ShapeOf({1, 4, 2}));
    source_builder.Output(output);
    ir::TensorId substituted_output;
    if (mutation == 8) {
      const auto private_root = source_builder.Unary(value, ir::UnaryFn::kTanh);
      substituted_output = source_builder.Reshape(private_root, ir::ShapeOf({1, 4, 2}));
    }
    if (mutation == 7) source_builder.Keep(root);
    auto source = std::move(source_builder).Finish();
    ASSERT_TRUE(source);
    ir::Builder planned_builder;
    const auto planned_input =
        planned_builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({4, 2})));
    const auto reshape =
        mutation == 3
            ? planned_builder.View(planned_input, ir::ShapeOf({1, 4, 1}), {4, 4, 16, 16}, 0)
            : planned_builder.Reshape(planned_input, ir::ShapeOf({1, 4, 2}));
    const auto result = planned_builder.Unary(reshape, ir::UnaryFn::kSigmoid);
    planned_builder.Output(result);
    const auto private_result =
        mutation == 8 ? planned_builder.Unary(reshape, ir::UnaryFn::kTanh) : ir::TensorId{};
    if (mutation == 7) planned_builder.Keep(result);
    auto planned = std::move(planned_builder).Finish();
    ASSERT_TRUE(planned);
    auto plan = PlanFor(*planned);
    plan.tensors.assign(source->tensor_count(), ir::TensorId{});
    plan.tensors[value.v] = planned_input;
    plan.tensors[root.v] = mutation == 5   ? planned_input
                           : mutation == 6 ? ir::TensorId{UINT32_MAX - 1}
                                           : ir::TensorId{};
    plan.tensors[output.v] = result;
    if (substituted_output.valid()) plan.tensors[substituted_output.v] = private_result;
    const kn::ComparisonBindingRange range{.original_root = value,
                                           .allocation_class = 1,
                                           .allocation_bytes = 32,
                                           .bytes = 32,
                                           .alignment = 16,
                                           .base = 0x10000};
    const co::ComparisonExternalBinding external{.original_root = value,
                                                 .allocation_class = 1,
                                                 .allocation_bytes = 32,
                                                 .bytes = 32,
                                                 .alignment = 16};
    auto other_source = source->Clone();
    ASSERT_TRUE(other_source);
    const auto& boundary_source = mutation == 2 ? *other_source : *source;
    const co::ComparisonBoundaryView view{
        {boundary_source, boundary_source.inputs(), boundary_source.outputs(), {}},
        {},
        0,
        std::span(&external, 1)};
    auto boundary = co::MakeComparisonBoundary(view, std::make_shared<ComparisonFunder>());
    ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .bindings = std::span(&range, 1),
                                  .expected_boundary = mutation == 1 ? nullptr : &*boundary};
    kn::Addresses addresses;
    addresses.activations = 1;
    const std::array substituted_outputs{substituted_output};
    const auto io_outputs =
        mutation == 8 ? std::span<const ir::TensorId>(substituted_outputs) : source->outputs();
    const kn::LowerInput input(plan, {*source, source->inputs(), io_outputs, {}}, addresses,
                               std::nullopt, &request);
    const auto checked = kg::ValidateComparisonBindings(input, std::span(&range, 1));
    if (mutation == 0)
      EXPECT_TRUE(checked) << (checked ? "" : checked.error());
    else
      EXPECT_FALSE(checked);
  }
}

TEST(GgmlComparisonTest, InstalledGeneratedRecipeTableAndLegacyRowsRemainPresent) {
  const auto legacy = kg::MeasuredFusionComparisons();
  const auto recipes = kg::MeasuredRecipeComparisons();
  EXPECT_EQ(legacy.size(), 12U);
  // Authenticated r151 baseline: retain all fourteen historical records,
  // including their timings/evidence, while admitting validated new rows.
  constexpr std::array<std::string_view, 14> historical_keys{
      "d55b0c79e21f4a135779be677525d5dc196335737a5beb22f4c54dc0145f4eb1",
      "ad06cca150cbf3f1076357cb2ab3bcd36d4dea5cda3919e9785a20879d9da04c",
      "2855914fe765b06721fcc3960b448dd9292ee426658b581c888750d546c25c3c",
      "3102e77d5d520011ef41f744a5df3fb4c2a774ca50e9bfd3d06ade10ea02cf15",
      "0a5b4a15c30da277991c2c4ca8413b7de68c87d4003a911cda3a4a1d1f74e93f",
      "4c3122c72d3ea2ffd47c822a1132ac1e12044fbdd85f318b02e3e34aca3fae14",
      "d91038b54fbcf8d0863953de2e3b3b7f4c3ccd367403b2e929384e86dc1a2518",
      "5a843b9911d530cc2e2adb81d0f54cbb2a2a86a2605b260de55c5b4fae2ce8e7",
      "3ae3c090caca0e5eda3c299635229bdbc7d52111fec263e51c7e2fec76f32620",
      "98892d27be2a95ff391c087429d7dc7162c4c956cd6bce53bdc1588ee4413d46",
      "2a8cbf9cba0bb7036c53585a9360e3c517c77281b11b890757bd93dc9703614d",
      "f976071f8b566851335b97b6f6736ecd96fdba872f74e4f12f370fe741dbe6ab",
      "4c8c0fafa6b61b47e0f6624cd337d6c0e6ed8d749fc8b78b207083d264b203fc",
      "e55a5bb1499432c3147f39f18730844ac8638a137603bdd91ff3fc98406f5452",
  };
  ASSERT_GE(recipes.size(), historical_keys.size());
  std::vector<llmp::execution::RecipeComparison> historical;
  std::vector<base::Sha256Digest> all_keys;
  for (const auto& row : recipes) {
    EXPECT_TRUE(llmp::execution::ValidateRecipeComparison(row));
    const auto key = llmp::execution::RecipeComparisonKeyDigest(row.key);
    all_keys.push_back(key);
    if (std::ranges::contains(historical_keys, base::ToHex(key))) historical.push_back(row);
  }
  std::ranges::sort(all_keys);
  EXPECT_EQ(std::ranges::adjacent_find(all_keys), all_keys.end());
  ASSERT_EQ(historical.size(), historical_keys.size());
  EXPECT_EQ(base::ToHex(llmp::execution::RecipeComparisonsDigest(historical)),
            "d66376222a43c01ab0fa458825062cbc0b4452da21b0f936ba9dde3646f29602");
}

TEST(GgmlComparisonTest, RejectsMissingInputRange) {
  Fixture fixture;
  EXPECT_FALSE(kg::ValidateComparisonBindings(fixture.input, {}));
}

TEST(GgmlComparisonTest, RejectsMisalignedAndOverflowingRanges) {
  Fixture fixture;
  fixture.ranges[0].offset = 1;
  EXPECT_FALSE(kg::ValidateComparisonBindings(fixture.input, fixture.ranges));

  fixture.ranges[0].offset = 0;
  fixture.ranges[0].bytes = UINT64_MAX;
  EXPECT_FALSE(kg::ValidateComparisonBindings(fixture.input, fixture.ranges));
}

TEST(GgmlComparisonTest, RejectsUndersizedRangesInMeasurementAndBoundModes) {
  Fixture fixture;
  fixture.ranges[0].bytes = 1;
  EXPECT_FALSE(kg::ValidateComparisonBindings(fixture.input, fixture.ranges));

  fixture.addresses.activations = 0;
  fixture.ranges[0].base = 0;
  EXPECT_FALSE(kg::ValidateComparisonBindings(fixture.input, fixture.ranges));
}

TEST(GgmlComparisonTest, RejectsDifferentAliasClassForOverlappingRanges) {
  Fixture fixture;
  std::array<kn::ComparisonBindingRange, 2> ranges{fixture.ranges[0], fixture.ranges[0]};
  ranges[1].part = llmp::format::PartRole::kData;
  ranges[1].allocation_class = 8;
  EXPECT_FALSE(kg::ValidateComparisonBindings(fixture.input, ranges));
}

TEST(GgmlComparisonTest, StateAccessFactsComeFromGraphUses) {
  ir::Builder builder;
  const ir::StateRef ref{.stage = 1, .tensor = 3, .layer = 2, .segment = 1};
  const auto state =
      builder.State(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({4, 8})), ref);
  builder.Output(state);
  const auto graph = std::move(builder).Finish().value();
  std::array<llmp::components::StepState, 1> declarations{
      llmp::components::StepState{.tensor = state, .ref = ref}};
  EXPECT_TRUE(llmp::components::CheckStateAccess(graph, declarations));

  declarations[0].read = true;
  EXPECT_FALSE(llmp::components::CheckStateAccess(graph, declarations));
  declarations[0].read = false;
  EXPECT_FALSE(llmp::components::CheckStateAccess(graph, {}));
}

TEST(GgmlComparisonTest, MissingOwnedSourceBoundaryCannotPromoteProof) {
  ir::Builder builder;
  const auto first = builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 4})));
  const auto second = builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 4})));
  builder.Output(first);
  builder.Output(second);
  auto graph = std::move(builder).Finish().value();
  auto plan = PlanFor(graph);
  llmp::kernels::Addresses addresses{};
  std::array<kn::ComparisonBindingRange, 2> ranges{
      kn::ComparisonBindingRange{.original_root = first,
                                 .allocation_class = 7,
                                 .allocation_bytes = 512,
                                 .offset = 0,
                                 .bytes = 256,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = second,
                                 .allocation_class = 7,
                                 .allocation_bytes = 512,
                                 .offset = 0,
                                 .bytes = 256,
                                 .alignment = 16}};
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull, .bindings = ranges};
  kn::LowerInput input(plan, {graph, graph.inputs(), graph.outputs(), {}}, addresses, std::nullopt,
                       &request);
  auto registry = llmp::execution::Registry::Create({});
  ASSERT_TRUE(registry);
  kg::LoweredStep lowered;
  const auto original = kg::DescribeComparison(input, lowered, *registry, nullptr, {});
  EXPECT_EQ(original.state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(original.missing, kn::ComparisonMissing::kSourceIdentity);
  EXPECT_EQ(original.boundary, base::Sha256Digest{});
}

TEST(GgmlComparisonTest, MissingComparisonSelectionIdentityCannotFallBackToSelection) {
  Fixture fixture;
  auto registry = llmp::execution::Registry::Create({});
  ASSERT_TRUE(registry);
  llmp::execution::BackendDeclaration declaration{.recipe = base::Sha256Digest{1},
                                                  .build = base::Sha256Digest{2},
                                                  .selection = base::Sha256Digest{3},
                                                  .comparison_selection = {}};
  kg::LoweredStep lowered;
  const auto seal = kg::DescribeComparison(fixture.input, lowered, *registry, &declaration, {},
                                           fixture.boundary->digest());
  EXPECT_EQ(seal.state, kn::ComparisonProofState::kUnproved);
  EXPECT_EQ(seal.missing, kn::ComparisonMissing::kSourceIdentity);
}

TEST(GgmlComparisonTest, ActualBoundaryUsesSharedCanonicalizer) {
  Fixture fixture;
  ASSERT_TRUE(fixture.boundary);
  auto observed = kg::ObserveComparisonBoundary(fixture.input, fixture.ranges, fixture.funder);
  ASSERT_TRUE(observed) << (observed ? "" : observed.error());
  EXPECT_EQ(observed->combined, fixture.boundary->digest().combined);
  fixture.ranges[0].offset = 16;
  EXPECT_FALSE(kg::ObserveComparisonBoundary(fixture.input, fixture.ranges, fixture.funder));
}

TEST(GgmlComparisonTest, RejectsWeightPartWithoutFormatReadableTail) {
  const auto weight_format = llmp::format::TensorFormat{.encoding = llmp::format::Encoding::kQ4_0,
                                                        .layout = llmp::format::Layout::kGgmlBlocks,
                                                        .shape = ir::ShapeOf({32, 64})};
  const auto parts = llmp::format::PartsOf(weight_format);
  ASSERT_TRUE(parts);
  const auto data = parts->Find(llmp::format::PartRole::kData);
  ASSERT_TRUE(data);
  ASSERT_GT(data->readable, data->bytes);

  ir::Builder builder;
  const auto activation =
      builder.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({32, 1})));
  const auto weight = builder.Weight(weight_format, {.component = 1, .role = 0}, "q4");
  builder.Output(builder.MatMul(weight, activation));
  auto graph = std::move(builder).Finish().value();
  auto plan = PlanFor(graph);
  std::array<llmp::components::StepWeight, 1> weights{
      llmp::components::StepWeight{.tensor = weight,
                                   .ref = {.component = 1, .role = 0},
                                   .parts = {{"q4", llmp::format::PartRole::kData}}}};
  std::array<kn::ComparisonBindingRange, 2> ranges{
      kn::ComparisonBindingRange{.original_root = activation,
                                 .allocation_class = 1,
                                 .allocation_bytes = 4096,
                                 .bytes = 4096,
                                 .alignment = 16},
      kn::ComparisonBindingRange{.original_root = weight,
                                 .part = llmp::format::PartRole::kData,
                                 .allocation_class = 2,
                                 .allocation_bytes = data->bytes,
                                 .bytes = data->bytes,
                                 .alignment = 16}};
  kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull, .bindings = ranges};
  kn::Addresses addresses{};  // measuring mode still validates the declared readable extent
  kn::LowerInput input(plan, {graph, graph.inputs(), graph.outputs(), weights}, addresses,
                       std::nullopt, &request);
  EXPECT_FALSE(kg::ValidateComparisonBindings(input, ranges));
}

TEST(GgmlComparisonTest, RegionPortCapacityCountsRepeatedEdgeOccurrences) {
  ir::Builder builder;
  const auto input =
      builder.Input(llmp::ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 4})));
  const auto weight =
      builder.Weight(llmp::ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16})),
                     {.component = 1, .role = 0}, "shared");
  for (std::size_t i = 0; i < 5; ++i) builder.Output(builder.Mul(input, weight));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  const auto capacity = kg::ComparisonRegionEdgeOccurrenceCapacity(graph->nodes());
  ASSERT_TRUE(capacity);
  // The source input/weight nodes also contribute their output occurrences.
  EXPECT_EQ(*capacity, 17U);
  // A unique-tensor bound would be 14 here and silently miss one repeated
  // input/weight occurrence across the five member nodes.
  EXPECT_GT(*capacity, 2U * graph->tensor_count());
  std::array<ir::Node, 1> malformed{graph->nodes().front()};
  malformed.front().nsrc =
      static_cast<decltype(malformed.front().nsrc)>(malformed.front().src.size() + 1);
  EXPECT_FALSE(kg::ComparisonRegionEdgeOccurrenceCapacity(malformed));
}

}  // namespace
