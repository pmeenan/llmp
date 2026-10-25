// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/comparison.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <tuple>

#include "base/sha256.h"
#include "components/input_values.h"
#include "components/step.h"
#include "kernels/ggml/comparison_access.h"
#include "kernels/ggml/comparison_schema.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/set_rows_group.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "planner/placement.h"

namespace llmp::kernels::ggml {
namespace {

// llmp_ops.cc's stable custom prefix occupies bytes [0, 32); custom semantic
// fields, including epsilon, begin at this pin's kEpsOffset of 32.
constexpr std::size_t kCustomSemanticOffset = 32;

class CanonicalHash {
 public:
  explicit CanonicalHash(std::string_view domain) { Text(domain); }

  void U8(std::uint8_t value) { Add(value); }
  void U16(std::uint16_t value) { Add(value); }
  void U32(std::uint32_t value) { Add(value); }
  void U64(std::uint64_t value) { Add(value); }
  void Text(std::string_view value) {
    U64(value.size());
    hash_.Update(value);
  }
  void Digest(const base::Sha256Digest& value) { hash_.Update(std::as_bytes(std::span(value))); }
  base::Sha256Digest Finish() { return hash_.Finish(); }

 private:
  template <class T>
  void Add(T value) {
    std::array<std::byte, sizeof(T)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::byte>((value >> (i * 8U)) & 0xffU);
    hash_.Update(bytes);
  }
  base::Sha256 hash_;
};

void HashPhysicalBindings(CanonicalHash& hash, const LowerInput& input) {
  std::vector<const ComparisonBindingRange*> sorted;
  sorted.reserve(input.comparison->bindings.size());
  for (const auto& binding : input.comparison->bindings) sorted.push_back(&binding);
  std::ranges::sort(sorted, [](const auto* a, const auto* b) {
    const auto part_key = [](const auto* range) {
      return std::tuple(range->part.has_value(),
                        range->part ? static_cast<std::uint8_t>(*range->part) : 0);
    };
    return std::tuple(a->original_root.v, part_key(a)) <
           std::tuple(b->original_root.v, part_key(b));
  });
  std::vector<std::pair<std::uint32_t, std::uint32_t>> classes;
  hash.U64(sorted.size());
  for (const auto* binding : sorted) {
    auto found = std::ranges::find(classes, binding->allocation_class,
                                   &std::pair<std::uint32_t, std::uint32_t>::first);
    if (found == classes.end()) {
      classes.emplace_back(binding->allocation_class, static_cast<std::uint32_t>(classes.size()));
      found = std::prev(classes.end());
    }
    hash.U32(binding->original_root.v);
    hash.U8(binding->part.has_value() ? 1 : 0);
    if (binding->part) hash.U8(static_cast<std::uint8_t>(*binding->part));
    hash.U32(found->second);
    hash.U64(binding->allocation_bytes);
    hash.U64(binding->offset);
    hash.U64(binding->bytes);
    hash.U64(binding->alignment);
    hash.U8(binding->resource_part_identity.has_value() ? 1 : 0);
    if (binding->resource_part_identity) hash.Digest(*binding->resource_part_identity);
  }
}

bool HashPhysicalDescriptor(CanonicalHash& hash, const LoweredStep& lowered,
                            const PhysicalTensor& physical, bool selected_marker = false) {
  bool complete = true;
  const auto* tensor = physical.descriptor;
  hash.U32(physical.id);
  hash.U32(physical.tensor.v);
  hash.U32(physical.root.v);
  hash.U32(physical.producer.v);
  hash.U8(physical.part.has_value() ? 1 : 0);
  if (physical.part) hash.U8(static_cast<std::uint8_t>(*physical.part));
  hash.U8(physical.ir_format.has_value() ? 1 : 0);
  if (physical.ir_format) {
    HashTensorFormat(hash, *physical.ir_format);
  }
  hash.U32(static_cast<std::uint32_t>(tensor->op));
  hash.U32(static_cast<std::uint32_t>(tensor->type));
  for (const auto ne : tensor->ne) hash.U64(static_cast<std::uint64_t>(ne));
  for (const auto nb : tensor->nb) hash.U64(static_cast<std::uint64_t>(nb));
  hash.U32(GGML_MAX_SRC);
  for (std::uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
    const auto* source = tensor->src[i];
    hash.U8(source ? 1 : 0);
    if (!source) {
      continue;
    }
    const auto* source_record = PhysicalTensorOf(lowered, source);
    hash.U32(source_record ? source_record->id : UINT32_MAX);
    complete = complete && source_record != nullptr;
  }
  hash.U8(tensor->view_src ? 1 : 0);
  const auto* view_record =
      tensor->view_src ? PhysicalTensorOf(lowered, tensor->view_src) : nullptr;
  hash.U32(view_record ? view_record->id : UINT32_MAX);
  if (tensor->view_src && !view_record) complete = false;
  hash.U64(static_cast<std::uint64_t>(tensor->view_offs));
  hash.U8(static_cast<std::uint8_t>(tensor->op));
  if (selected_marker)
    hash.Text("llmp.selected.owned-marker.v1");
  else
    HashTypedOperationParameters(hash, *tensor);
  return complete;
}

void HashQuantStoreLaunch(CanonicalHash& hash, const QuantStoreLaunchRecipe& recipe) {
  hash.Text("llmp.ggml.kv_quant_store.launch.v1");
  hash.U32(recipe.member_count);
  hash.U32(recipe.grid_x);
  hash.U32(recipe.grid_y);
  hash.U32(recipe.block_x);
  hash.U32(recipe.block_y);
  hash.U32(recipe.block_z);
  hash.U32(recipe.dynamic_shared_bytes);
  hash.U64(recipe.pool_bytes);
  for (std::size_t i = 0; i < recipe.member_count; ++i) {
    const auto& member = recipe.members[i];
    for (auto value : member.values_stride) hash.U64(value);
    for (auto value : member.ids_stride) hash.U64(value);
    for (auto value : member.destination_stride) hash.U64(value);
    hash.U32(member.blocks_per_row);
    hash.U32(member.rows);
    hash.U32(member.planes);
    hash.U32(member.ids_planes);
    hash.U32(member.ids_batches);
    hash.U32(member.total_blocks);
    hash.U32(static_cast<std::uint32_t>(member.type));
    hash.U8(member.index_bytes);
  }
}

void HashRopeQuantStoreLaunch(CanonicalHash& hash, const RopeQuantStoreLaunchRecipe& recipe) {
  hash.Text("llmp.ggml.kv_quant_rope_store.launch.v1");
  hash.U64(recipe.x_head_stride);
  hash.U64(recipe.x_token_stride);
  hash.U64(recipe.destination_row_stride);
  hash.U32(recipe.blocks_per_head);
  hash.U32(recipe.blocks_per_row);
  hash.U32(recipe.total_blocks);
  hash.U32(recipe.grid_x);
  hash.U32(recipe.block_x);
  hash.U32(recipe.dynamic_shared_bytes);
  hash.U64(recipe.pool_bytes);
  hash.U32(static_cast<std::uint32_t>(recipe.n_dims));
  hash.U32(std::bit_cast<std::uint32_t>(recipe.freq_scale));
  hash.U32(std::bit_cast<std::uint32_t>(recipe.ext_factor));
  hash.U32(std::bit_cast<std::uint32_t>(recipe.attn_factor));
  hash.U32(std::bit_cast<std::uint32_t>(recipe.theta_scale));
  hash.U32(std::bit_cast<std::uint32_t>(recipe.correction_dims[0]));
  hash.U32(std::bit_cast<std::uint32_t>(recipe.correction_dims[1]));
  hash.U32(static_cast<std::uint32_t>(recipe.type));
  hash.U8(recipe.has_frequency_factors ? 1 : 0);
}

bool HasSchema(const ggml_tensor& tensor) {
  // Typed cases below mirror the pinned GGML constructors and the specific
  // Llmp custom helper needed by Batch 1. No opaque parameter bytes are sealed.
  const bool has_sources =
      std::ranges::any_of(tensor.src, [](const ggml_tensor* source) { return source != nullptr; });
  const auto all_zero = [&](std::size_t first) {
    return std::ranges::all_of(std::span(tensor.op_params).subspan(first),
                               [](std::int32_t field) { return field == 0; });
  };
  if (tensor.op == GGML_OP_VIEW) {
    const auto* source = tensor.src[0];
    if (!source || !tensor.view_src || sizeof(tensor.view_offs) > sizeof(tensor.op_params))
      return false;
    std::size_t encoded_offset = 0;
    std::memcpy(&encoded_offset, tensor.op_params, sizeof(encoded_offset));
    constexpr std::size_t kOffsetWords =
        (sizeof(std::size_t) + sizeof(tensor.op_params[0]) - 1) / sizeof(tensor.op_params[0]);
    const auto source_offset = source->view_src ? source->view_offs : 0;
    if (encoded_offset > std::numeric_limits<std::size_t>::max() - source_offset) return false;
    const auto* expected_root = source->view_src ? source->view_src : source;
    return tensor.view_src == expected_root && source_offset + encoded_offset == tensor.view_offs &&
           all_zero(kOffsetWords);
  }
  if (tensor.op == GGML_OP_PERMUTE) {
    const auto* source = tensor.src[0];
    if (!source || tensor.type != source->type ||
        std::ranges::any_of(std::span(tensor.src).subspan(1),
                            [](const auto* extra) { return extra != nullptr; }))
      return false;
    std::array<bool, GGML_MAX_DIMS> seen{};
    for (std::size_t i = 0; i < GGML_MAX_DIMS; ++i) {
      const auto axis = tensor.op_params[i];
      if (axis < 0 || axis >= GGML_MAX_DIMS || seen[static_cast<std::size_t>(axis)]) return false;
      seen[static_cast<std::size_t>(axis)] = true;
      if (tensor.ne[axis] != source->ne[i] ||
          ((axis == 0 ? tensor.ne[axis] != ggml_blck_size(tensor.type) : tensor.ne[axis] != 1) &&
           tensor.nb[axis] != source->nb[i]))
        return false;
    }
    return all_zero(GGML_MAX_DIMS);
  }
  if ((tensor.op == GGML_OP_NONE && !has_sources) || tensor.op == GGML_OP_RESHAPE ||
      tensor.op == GGML_OP_TRANSPOSE || tensor.op == GGML_OP_CONT || tensor.op == GGML_OP_ADD ||
      tensor.op == GGML_OP_MUL || tensor.op == GGML_OP_REPEAT || tensor.op == GGML_OP_CPY ||
      tensor.op == GGML_OP_GET_ROWS || tensor.op == GGML_OP_SET_ROWS ||
      tensor.op == GGML_OP_DSV4_HC_POST)
    return all_zero(0);
  if (tensor.op == GGML_OP_MUL_MAT)
    return (tensor.op_params[0] == GGML_PREC_DEFAULT || tensor.op_params[0] == GGML_PREC_F32) &&
           tensor.op_params[1] == GGML_HINT_NONE && all_zero(2);
  if (tensor.op == GGML_OP_RMS_NORM || tensor.op == GGML_OP_NORM || tensor.op == GGML_OP_L2_NORM)
    return all_zero(1);
  if (tensor.op == GGML_OP_SCALE) return all_zero(2);
  if (tensor.op == GGML_OP_CONCAT)
    return tensor.op_params[0] >= 0 && tensor.op_params[0] < 4 && all_zero(1);
  if (tensor.op == GGML_OP_ROPE) return all_zero(16);
  if (tensor.op == GGML_OP_UNARY) {
    const auto unary = tensor.op_params[0];
    return (unary == GGML_UNARY_OP_SILU || unary == GGML_UNARY_OP_SIGMOID ||
            unary == GGML_UNARY_OP_TANH || unary == GGML_UNARY_OP_SOFTPLUS) &&
           all_zero(1);
  }
  if (tensor.op == GGML_OP_GLU) {
    if (tensor.op_params[1] != 0) return false;
    if (ggml_get_glu_op(&tensor) != GGML_GLU_OP_SWIGLU &&
        ggml_get_glu_op(&tensor) != GGML_GLU_OP_GEGLU)
      return false;
    return all_zero(2);
  }
  if (tensor.op == GGML_OP_CUSTOM && (LlmpOpOf(&tensor) == LlmpOp::kLayerNormModulateSplit ||
                                      LlmpOpOf(&tensor) == LlmpOp::kGatedResidualSplit)) {
    const bool modulate = LlmpOpOf(&tensor) == LlmpOp::kLayerNormModulateSplit;
    if (!modulate && LlmpOpInt(&tensor, 0) != 0) return false;
    if (modulate && (LlmpOpInt(&tensor, 2) < 0 || LlmpOpInt(&tensor, 2) > 1)) return false;
    for (int field = modulate ? 3 : 2; field < 8; ++field)
      if (LlmpOpInt(&tensor, field) != 0) return false;
    return true;
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kSpatialChannelNorm) {
    for (int field = 3; field < 8; ++field)
      if (LlmpOpInt(&tensor, field) != 0) return false;
    return CheckSpatialChannelNorm(&tensor).has_value();
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kNormRopeSections) {
    for (int field = 7; field < 8; ++field)
      if (LlmpOpInt(&tensor, field) != 0) return false;
    const auto sections = LlmpOpInt(&tensor, 3);
    for (int section = sections; section >= 0 && section < 3; ++section)
      if (LlmpOpInt(&tensor, 4 + section) != 0) return false;
    return CheckNormRopeSections(&tensor).has_value();
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kDsv4HcNormF16) {
    for (std::size_t field = kCustomSemanticOffset + sizeof(float);
         field < sizeof(tensor.op_params); ++field)
      if (reinterpret_cast<const std::byte*>(tensor.op_params)[field] != std::byte{0}) return false;
    return CheckDsv4HcNormF16(&tensor).has_value();
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kHcCombine) {
    for (int field = 0; field < 8; ++field)
      if (LlmpOpInt(&tensor, field) != 0) return false;
    return CheckHcCombine(&tensor).has_value();
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kGdnGates) {
    for (int field = 0; field < 8; ++field)
      if (LlmpOpInt(&tensor, field) != 0) return false;
    return CheckGdnGates(&tensor).has_value();
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kGdnNormGate) {
    for (int field = 2; field < 8; ++field)
      if (LlmpOpInt(&tensor, field) != 0) return false;
    return CheckGdnNormGate(&tensor).has_value();
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kQsaPrep) {
    for (int field = 6; field < 8; ++field)
      if (LlmpOpInt(&tensor, field) != 0) return false;
    return CheckQsaPrep(&tensor).has_value();
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kQuantizeQ8) {
    // QuantizeQ8's schema owns exactly {K, rows} after the operation tag.
    // Authenticate those fields with CheckQuantizeQ8 and require only the
    // remaining reserved tail to be zero.
    constexpr std::size_t kQuantizeQ8Params = 2 * sizeof(std::int32_t);
    for (std::size_t field = kCustomSemanticOffset + kQuantizeQ8Params;
         field < sizeof(tensor.op_params); ++field)
      if (reinterpret_cast<const std::byte*>(tensor.op_params)[field] != std::byte{0}) return false;
    return CheckQuantizeQ8(&tensor).has_value();
  }
  if (tensor.op == GGML_OP_CUSTOM && LlmpOpOf(&tensor) == LlmpOp::kMmvqPrepared) {
    for (std::size_t field = kCustomSemanticOffset; field < sizeof(tensor.op_params); ++field)
      if (reinterpret_cast<const std::byte*>(tensor.op_params)[field] != std::byte{0}) return false;
    return CheckMmvqPrepared(&tensor).has_value();
  }
  return false;
}

// Schema knowledge is not itself an access proof. These GGML operations have
// direct, descriptor-visible operands/results: their complete accesses are
// bounded by the source tensor's format parts and the placed output root.
// Anything with an opaque/custom helper, selected implementation, or route
// preparation remains explicitly unproved until that helper publishes its
// own typed access recipe.
bool IsPrimitiveFloatMmvf(const PlanStep& step, const execution::Registry& registry,
                          const LaunchContext* launch_context,
                          ComparisonMulMatSelector mul_mat_selector) {
  if (!launch_context || !mul_mat_selector || step.operation != ir::Primitive(ir::Op::kMatMul) ||
      step.implementation != kMulMatVector || step.nodes.size() != 1)
    return false;
  const auto registered = registry.Find(step.implementation);
  if (!registered || registry.at(*registered).name != kMulMatVector) return false;
  const ggml_tensor* node = step.nodes[0];
  if (!node || node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr || node->src[1] == nullptr ||
      (node->src[0]->type == GGML_TYPE_F16 && node->op_params[0] != GGML_PREC_DEFAULT) ||
      !CheckMulMat(node))
    return false;
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* input = node->src[1];
  if ((weights->type != GGML_TYPE_F16 && weights->type != GGML_TYPE_F32) ||
      input->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) ||
      !ggml_is_contiguous(input) || !ggml_is_contiguous(node))
    return false;
  const auto selected = mul_mat_selector(*launch_context, node);
  return selected && *selected == MulMatPath::kVector;
}

// These additions apply only to complete native programs; local and
// remainder access domains continue to require their existing typed schemas.
bool HasWholeNativeSchema(const ggml_tensor& tensor) {
  if (HasSchema(tensor) || HasWholeNativeFillOrTopKSchema(tensor) ||
      HasWholeNativeReluSchema(tensor))
    return true;
  const auto zero_tail = [&](std::size_t first) {
    for (std::size_t i = first; i < std::size(tensor.op_params); ++i)
      if (tensor.op_params[i] != 0) return false;
    return true;
  };
  // These parameter records are the pinned native constructors' scalar
  // semantics. Actual execution still requires the registered native checker.
  if (tensor.op == GGML_OP_SSM_CONV || tensor.op == GGML_OP_SUM_ROWS || tensor.op == GGML_OP_DIV ||
      tensor.op == GGML_OP_MUL_MAT_ID || tensor.op == GGML_OP_LIGHTNING_INDEXER)
    return zero_tail(0);
  if (tensor.op == GGML_OP_SOFT_MAX || tensor.op == GGML_OP_CLAMP) return zero_tail(2);
  if (tensor.op == GGML_OP_ARGSORT)
    return (tensor.op_params[0] == GGML_SORT_ORDER_ASC ||
            tensor.op_params[0] == GGML_SORT_ORDER_DESC) &&
           zero_tail(1);
  if (tensor.op == GGML_OP_GATED_DELTA_NET) return zero_tail(1);
  if (tensor.op == GGML_OP_ROPE_BACK) return zero_tail(16);
  if (tensor.op == GGML_OP_FLASH_ATTN_EXT) return zero_tail(7);
  if (tensor.op == GGML_OP_DSV4_HC_COMB || tensor.op == GGML_OP_DSV4_HC_PRE) return zero_tail(2);
  if (tensor.op == GGML_OP_MUL_MAT && tensor.op_params[1] == GGML_HINT_SRC0_IS_HADAMARD)
    return (tensor.op_params[0] == GGML_PREC_DEFAULT || tensor.op_params[0] == GGML_PREC_F32) &&
           zero_tail(2);
  return HasWholeNativeCustomScalarSchema(tensor);
}

bool HasPrimitiveF16MmvfGluClosure(const LoweredStep& lowered, const execution::Registry& registry,
                                   const LaunchContext* launch_context,
                                   ComparisonMulMatSelector mul_mat_selector) {
  if (!launch_context || !mul_mat_selector || lowered.plan.steps.size() != 3) return false;
  std::array<const ggml_tensor*, 2> products{};
  std::size_t product_count = 0;
  const ggml_tensor* glu = nullptr;
  for (const auto& step : lowered.plan.steps) {
    if (step.operation == ir::Primitive(ir::Op::kMatMul)) {
      if (product_count == products.size() ||
          !IsPrimitiveFloatMmvf(step, registry, launch_context, mul_mat_selector))
        return false;
      products[product_count++] = step.nodes[0];
    } else if (step.operation == ir::Primitive(ir::Op::kGlu)) {
      if (glu || step.implementation != kSwiGluName || step.nodes.size() != 1) return false;
      const auto registered = registry.Find(step.implementation);
      if (!registered || registry.at(*registered).name != kSwiGluName) return false;
      glu = step.nodes[0];
    } else {
      // This complete recipe intentionally contains only two products and
      // their checked split SwiGLU. A third product/helper cannot inherit it.
      return false;
    }
  }
  if (product_count != products.size() || !glu || products[0] == products[1] ||
      products[0]->src[1] != products[1]->src[1] ||
      !CheckComparisonF16SwiGlu(products[0], products[1], glu))
    return false;
  std::size_t physical_products = 0, physical_glus = 0;
  for (const auto& physical : lowered.physical_tensors) {
    if (!physical.descriptor) return false;
    if (physical.descriptor->op == GGML_OP_MUL_MAT) {
      ++physical_products;
      if (physical.descriptor != products[0] && physical.descriptor != products[1]) return false;
    } else if (physical.descriptor->op == GGML_OP_GLU) {
      ++physical_glus;
      if (physical.descriptor != glu) return false;
    }
  }
  return physical_products == products.size() && physical_glus == 1;
}

bool HasPrimitiveMmvfAddClosure(const LoweredStep& lowered, const execution::Registry& registry,
                                const LaunchContext* launch_context,
                                ComparisonMulMatSelector mul_mat_selector) {
  if (lowered.plan.steps.size() != 2) return false;
  const PlanStep* product = nullptr;
  const ggml_tensor* add = nullptr;
  for (const auto& step : lowered.plan.steps) {
    if (IsPrimitiveFloatMmvf(step, registry, launch_context, mul_mat_selector)) {
      if (product) return false;
      product = &step;
    } else if (step.implementation == kAddName && step.operation == ir::Primitive(ir::Op::kAdd) &&
               step.nodes.size() == 1 && CheckBinary(step.nodes[0], GGML_OP_ADD)) {
      if (add) return false;
      add = step.nodes[0];
    } else {
      return false;
    }
  }
  return product && add && (add->src[0] == product->nodes[0] || add->src[1] == product->nodes[0]);
}

bool HasSharedQ8PreparationClosure(const LowerInput& input, const LoweredStep& lowered,
                                   const LaunchContext* launch_context,
                                   const ComparisonQ8Dispatch* q8_dispatch) {
  if (!launch_context || !q8_dispatch || !q8_dispatch->complete() || !input.comparison ||
      !lowered.selected.empty() || !lowered.routing.empty())
    return false;
  std::array<ComparisonQ8StepRef, 3> q8_steps{};
  if (lowered.plan.steps.size() > q8_steps.size()) return false;
  for (std::size_t i = 0; i < lowered.plan.steps.size(); ++i)
    q8_steps[i] = {.step = &lowered.plan.steps[i], .ordinal = i};
  const auto typed_span = DescribeComparisonQ8Span(
      std::span<const ComparisonQ8StepRef>(q8_steps).first(lowered.plan.steps.size()),
      *launch_context, *q8_dispatch);
  if (!typed_span) return false;
  if (lowered.plan.steps.size() == 2) {
    std::array<const ggml_tensor*, 2> products{};
    std::array<const ggml_tensor*, 2> weights{};
    for (std::size_t i = 0; i < products.size(); ++i) {
      const PlanStep& step = lowered.plan.steps[i];
      if (step.operation != ir::Primitive(ir::Op::kMatMul) || step.implementation != kMulMatVecQ ||
          step.nodes.size() != 1 || step.nodes[0] == nullptr ||
          step.nodes[0]->op != GGML_OP_MUL_MAT)
        return false;
      products[i] = step.nodes[0];
      weights[i] = products[i]->src[0];
      auto dispatch = q8_dispatch->describe_mmvq(*launch_context, products[i], false);
      auto quantizer = q8_dispatch->describe_quantize_q8(products[i]);
      auto scratch = q8_dispatch->plan_mmvq(*launch_context, products[i]);
      if (!dispatch || !quantizer || !scratch || dispatch->immutable_weights || dispatch->fused ||
          *scratch == 0)
        return false;
      auto typed_call = std::ranges::find_if(
          typed_span->calls, [&](const auto& call) { return call.node == products[i]; });
      if (typed_call == typed_span->calls.end()) return false;
    }
    if (products[0]->src[1] != products[1]->src[1] || weights[0] == weights[1] ||
        !((weights[0]->type == GGML_TYPE_Q4_K && weights[1]->type == GGML_TYPE_Q6_K) ||
          (weights[1]->type == GGML_TYPE_Q4_K && weights[0]->type == GGML_TYPE_Q6_K)))
      return false;
    for (const ggml_tensor* weight : weights)
      if (!ggml_is_contiguous(weight) || weight->ne[0] != typed_span->input->ne[0] ||
          weight->ne[1] <= 0)
        return false;
    std::array<const PhysicalTensor*, 2> outputs{};
    for (std::size_t i = 0; i < products.size(); ++i) {
      outputs[i] = PhysicalTensorOf(lowered, products[i]);
      if (!outputs[i]) return false;
    }
    if (outputs[0]->root == outputs[1]->root) return false;
    const auto disjoint = [](const PhysicalRoot& a, const PhysicalRoot& b) {
      std::uint64_t a_end = 0, b_end = 0;
      return !__builtin_add_overflow(a.offset, a.bytes, &a_end) &&
             !__builtin_add_overflow(b.offset, b.bytes, &b_end) &&
             (a_end <= b.offset || b_end <= a.offset);
    };
    const auto p0_root_it =
        std::ranges::find(lowered.physical_roots, outputs[0]->root, &PhysicalRoot::root);
    const auto p1_root_it =
        std::ranges::find(lowered.physical_roots, outputs[1]->root, &PhysicalRoot::root);
    if (p0_root_it == lowered.physical_roots.end() || p1_root_it == lowered.physical_roots.end() ||
        !disjoint(*p0_root_it, *p1_root_it))
      return false;
    for (const ggml_tensor* weight : weights) {
      const auto* leaf = PhysicalTensorOf(lowered, weight);
      if (!leaf || !weight->data) return false;
      const auto typed_call = std::ranges::find_if(
          typed_span->calls, [&](const auto& call) { return call.weight == weight; });
      if (typed_call == typed_span->calls.end()) return false;
      const auto readable = ComparisonMmvqWeightReadableBytes(*weight, typed_call->mmvq);
      if (!readable) return false;
      const auto address = reinterpret_cast<std::uintptr_t>(weight->data);
      if (address > UINTPTR_MAX - *readable) return false;
      for (const PhysicalRoot* root : {&*p0_root_it, &*p1_root_it}) {
        const auto base = input.addresses.activations + root->offset;
        if (base > UINTPTR_MAX - root->bytes ||
            (base < address + *readable && address < base + root->bytes))
          return false;
      }
    }
    return true;
  }
  if (lowered.plan.steps.size() != 3) return false;

  const PlanStep* prepare_step = nullptr;
  const ggml_tensor* prepare = nullptr;
  const ggml_tensor* common_input = nullptr;
  std::array<const ggml_tensor*, 2> products{};
  std::array<const ggml_tensor*, 2> weights{};
  std::size_t product_count = 0;
  std::size_t ordinary_count = 0;
  std::size_t prepared_count = 0;
  for (const PlanStep& step : lowered.plan.steps) {
    if (step.operation == ir::Primitive(ir::Op::kPrepare)) {
      if (prepare_step || step.implementation != kQuantizeQ8Name || step.nodes.size() != 1 ||
          step.nodes[0] == nullptr || step.nodes[0]->op != GGML_OP_CUSTOM ||
          LlmpOpOf(step.nodes[0]) != LlmpOp::kQuantizeQ8 || !CheckQuantizeQ8(step.nodes[0]))
        return false;
      prepare_step = &step;
      prepare = step.nodes[0];
      continue;
    }
    if (step.operation != ir::Primitive(ir::Op::kMatMul) || step.nodes.size() != 1 ||
        step.nodes[0] == nullptr || product_count == products.size())
      return false;
    const ggml_tensor* node = step.nodes[0];
    const ggml_tensor* weight = nullptr;
    MmvqDispatchDescriptor mmvq{};
    if (step.implementation == kMmvqPreparedName && node->op == GGML_OP_CUSTOM &&
        LlmpOpOf(node) == LlmpOp::kMmvqPrepared) {
      auto original = MmvqPreparedOriginal(node);
      auto dispatch = q8_dispatch->describe_mmvq(*launch_context, node, true);
      if (!original || !q8_dispatch->plan_prepared_mmvq(*launch_context, node) || !dispatch ||
          !dispatch->immutable_weights || dispatch->fused)
        return false;
      mmvq = *dispatch;
      weight = original->src[0];
      if (original->src[1] != node->src[2] || node->src[1] == nullptr ||
          node->src[1]->src[0] != node->src[2] || node->src[2] == nullptr ||
          node->src[2]->type != GGML_TYPE_F32)
        return false;
      common_input = common_input == nullptr ? node->src[2] : common_input;
      if (common_input != node->src[2] || node->src[1] == nullptr) return false;
      ++prepared_count;
    } else if (step.implementation == kMulMatVecQ && node->op == GGML_OP_MUL_MAT &&
               node->src[0] != nullptr && node->src[1] != nullptr) {
      auto dispatch = q8_dispatch->describe_mmvq(*launch_context, node, false);
      auto scratch = q8_dispatch->plan_mmvq(*launch_context, node);
      if (!dispatch || !scratch || dispatch->immutable_weights || dispatch->fused || *scratch == 0)
        return false;
      mmvq = *dispatch;
      weight = node->src[0];
      common_input = common_input == nullptr ? node->src[1] : common_input;
      if (common_input != node->src[1]) return false;
      ++ordinary_count;
    } else {
      return false;
    }
    if ((weight->type != GGML_TYPE_Q4_K && weight->type != GGML_TYPE_Q6_K) ||
        !ggml_is_contiguous(weight) || weight->ne[0] != common_input->ne[0] || weight->ne[1] <= 0 ||
        common_input == nullptr || common_input->type != GGML_TYPE_F32 ||
        common_input->ne[0] <= 0 || common_input->ne[1] < 1 || node->ne[0] != weight->ne[1] ||
        node->ne[1] != common_input->ne[1] || mmvq.row_low != 0 || mmvq.row_high != weight->ne[1] ||
        mmvq.columns != common_input->ne[1] || mmvq.stride_col_dst != node->ne[0])
      return false;
    products[product_count] = node;
    weights[product_count] = weight;
    ++product_count;
  }
  if (!prepare_step || !prepare || !q8_dispatch->describe_quantize_q8(prepare) ||
      product_count != 2 || ordinary_count > 1 || prepared_count == 0 ||
      ordinary_count + prepared_count != 2 || common_input != prepare->src[0] ||
      weights[0] == weights[1] ||
      !((weights[0]->type == GGML_TYPE_Q4_K && weights[1]->type == GGML_TYPE_Q6_K) ||
        (weights[1]->type == GGML_TYPE_Q4_K && weights[0]->type == GGML_TYPE_Q6_K)) ||
      common_input != prepare->src[0])
    return false;
  for (const ggml_tensor* product : products) {
    if (product->op == GGML_OP_CUSTOM) {
      if (product->src[1] != prepare || product->src[2] != prepare->src[0]) return false;
    } else if (product->op != GGML_OP_MUL_MAT || product->src[1] != prepare->src[0]) {
      return false;
    }
  }
  const auto prep_index = static_cast<std::size_t>(prepare_step - lowered.plan.steps.data());
  const auto product_index = [&](const ggml_tensor* node) {
    return static_cast<std::size_t>(std::ranges::find_if(lowered.plan.steps,
                                                         [&](const auto& step) {
                                                           return step.nodes.size() == 1 &&
                                                                  step.nodes[0] == node;
                                                         }) -
                                    lowered.plan.steps.begin());
  };
  if (prep_index >= lowered.plan.steps.size()) return false;
  for (const ggml_tensor* product : products) {
    if (product->op == GGML_OP_CUSTOM && product_index(product) <= prep_index) return false;
  }

  const auto* source_record = PhysicalTensorOf(lowered, prepare->src[0]);
  const auto* prep_record = PhysicalTensorOf(lowered, prepare);
  if (!source_record || !prep_record || source_record->ir_format == std::nullopt ||
      prep_record->ir_format == std::nullopt ||
      source_record->ir_format->encoding != format::Encoding::kF32 ||
      prep_record->ir_format->encoding != format::Encoding::kQ8_1 ||
      source_record->root == prep_record->root)
    return false;
  const auto prep_root_it =
      std::ranges::find(lowered.physical_roots, prep_record->root, &PhysicalRoot::root);
  if (prep_root_it == lowered.physical_roots.end()) return false;
  const PhysicalRoot* prep_root = &*prep_root_it;
  const std::uint64_t expected_q8 =
      static_cast<std::uint64_t>(Q8Bytes(prepare->src[0]->ne[0], prepare->src[0]->ne[1]));
  if (expected_q8 == 0 || prep_root->bytes < expected_q8 ||
      prep_root->bytes < WrittenBytes(input.plan.graph, prep_record->root))
    return false;

  std::array<const PhysicalRoot*, 2> output_roots{};
  for (std::size_t i = 0; i < products.size(); ++i) {
    const auto* output = PhysicalTensorOf(lowered, products[i]);
    if (!output || output->root == prep_record->root) return false;
    const auto output_root =
        std::ranges::find(lowered.physical_roots, output->root, &PhysicalRoot::root);
    if (output_root == lowered.physical_roots.end()) return false;
    output_roots[i] = &*output_root;
  }
  const auto disjoint = [](const PhysicalRoot& a, const PhysicalRoot& b) {
    std::uint64_t a_end = 0, b_end = 0;
    return !__builtin_add_overflow(a.offset, a.bytes, &a_end) &&
           !__builtin_add_overflow(b.offset, b.bytes, &b_end) &&
           (a_end <= b.offset || b_end <= a.offset);
  };
  if (!disjoint(*prep_root, *output_roots[0]) || !disjoint(*prep_root, *output_roots[1]) ||
      !disjoint(*output_roots[0], *output_roots[1]))
    return false;

  // Bindings are checked against every actual lowered leaf by LowerFor;
  // here additionally require two distinct immutable source parts and prove
  // that all writable roots stay outside both readable weight tails.
  std::array<std::pair<std::uintptr_t, std::uint64_t>, 2> weight_ranges{};
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const auto leaf = std::ranges::find_if(lowered.leaves, [&](const LoweredLeaf& item) {
      if (item.tensor != weights[i] || !item.primitive.valid()) return false;
      const auto binding = std::ranges::find_if(input.comparison->bindings, [&](const auto& range) {
        return range.original_root == item.primitive && range.part == format::PartRole::kData;
      });
      if (binding == input.comparison->bindings.end()) return false;
      if (item.part == format::PartRole::kData) return true;
      if (item.part) return false;

      // Ordinary one-part GGML weights are represented by an untagged leaf.
      // Resolve that omission only through the authenticated original root,
      // its declared WeightRef, and the format's single data part.
      const auto declared = std::ranges::find_if(input.io.weights, [&](const auto& candidate) {
        return input.io.graph.root(candidate.tensor) == item.primitive;
      });
      if (declared == input.io.weights.end() || declared->parts.size() != 1 ||
          declared->parts.front().second != format::PartRole::kData)
        return false;
      const auto parts = format::PartsOf(input.io.graph.tensor(item.primitive).format);
      return parts && parts->size() == 1 && parts->parts().front().role == format::PartRole::kData;
    });
    if (leaf == lowered.leaves.end() || !weights[i]->data) return false;
    const std::int64_t padded_k = GGML_PAD(weights[i]->ne[0], 512);
    const std::uint64_t tail =
        static_cast<std::uint64_t>(ggml_row_size(weights[i]->type, padded_k - weights[i]->ne[0]));
    const std::uint64_t extent = static_cast<std::uint64_t>(ggml_nbytes(weights[i]));
    const auto address = reinterpret_cast<std::uintptr_t>(weights[i]->data);
    std::uint64_t readable = 0;
    if (__builtin_add_overflow(extent, tail, &readable) || readable == 0 ||
        address > UINTPTR_MAX - readable)
      return false;
    weight_ranges[i] = {address, readable};
  }
  const auto overlaps = [](std::uintptr_t a, std::uint64_t an, std::uintptr_t b, std::uint64_t bn) {
    return a < b + bn && b < a + an;
  };
  if (overlaps(weight_ranges[0].first, weight_ranges[0].second, weight_ranges[1].first,
               weight_ranges[1].second))
    return false;
  for (const PhysicalRoot* root : output_roots) {
    const auto address = input.addresses.activations + root->offset;
    if (overlaps(address, root->bytes, weight_ranges[0].first, weight_ranges[0].second) ||
        overlaps(address, root->bytes, weight_ranges[1].first, weight_ranges[1].second))
      return false;
  }
  const auto prep_address = input.addresses.activations + prep_root->offset;
  if (overlaps(prep_address, prep_root->bytes, weight_ranges[0].first, weight_ranges[0].second) ||
      overlaps(prep_address, prep_root->bytes, weight_ranges[1].first, weight_ranges[1].second))
    return false;
  return true;
}

bool HasDirectAccessClosure(const LoweredStep& lowered, const execution::Registry& registry,
                            const LowerInput& input, const LaunchContext* launch_context,
                            ComparisonMulMatSelector mul_mat_selector,
                            const ComparisonQ8Dispatch* q8_dispatch,
                            ComparisonRemainderCallFactsFn describe_call_facts) {
  if (!lowered.selected.empty() || !lowered.routing.empty()) return false;
  const bool shared_q8 = HasSharedQ8PreparationClosure(input, lowered, launch_context, q8_dispatch);
  // Indexed stores remain fail-closed without a program-owned generated
  // input. RopeStore is admitted below only for the ordinary F16 fused
  // launcher after DescribeGeneratedStore has authenticated its exact
  // generated index, state/view mapping, and physical topology.
  for (const auto& step : lowered.plan.steps) {
    if ((step.operation == ir::Fused(ir::FusedKind::kRopeStore) &&
         ((step.implementation != kRopeSetRowsFused &&
           step.implementation != kRopeSetRowsQuantName) ||
          !lowered.generated_store_access_closed)) ||
        ((step.implementation == kSetRowsGroupedName ||
          step.implementation == kSetRowsQuantGroupName ||
          step.implementation == kSetRowsQuantName) &&
         !lowered.generated_store_access_closed))
      return false;
  }
  for (const auto& physical : lowered.physical_tensors) {
    if (!physical.descriptor) return false;
    if (physical.descriptor->op == GGML_OP_SET_ROWS && !lowered.generated_store_access_closed)
      return false;
  }
  for (const auto& step : lowered.plan.steps)
    if (HasComparisonAccessRecipe(step.implementation) &&
        !DescribeComparisonAccess(step, ComparisonAccessScope::kWhole))
      return false;
  const auto explicitly_checked = [&](const ggml_tensor* tensor) {
    return std::ranges::any_of(lowered.plan.steps, [&](const auto& step) {
      return std::ranges::contains(step.nodes, const_cast<ggml_tensor*>(tensor)) &&
             DescribeComparisonAccess(step, ComparisonAccessScope::kWhole).has_value();
    });
  };
  const auto direct_float_product = [&](const ggml_tensor* tensor) {
    for (const auto& step : lowered.plan.steps) {
      if (step.nodes.size() != 1 || step.nodes[0] != tensor || !launch_context ||
          !describe_call_facts)
        continue;
      ComparisonRemainderCallFacts facts;
      if (!describe_call_facts(*launch_context, step, facts)) return false;
      if (CheckComparisonSgemm(step, facts))
        return ComparisonLibraryWorkspaceDisjoint(lowered, facts);
      return tensor->src[0] && tensor->src[0]->type == GGML_TYPE_F32 &&
             IsPrimitiveFloatMmvf(step, registry, launch_context, mul_mat_selector) &&
             facts.has_mul_mat_selection && facts.mul_mat_vector && facts.scratch_bytes == 0;
    }
    return false;
  };
  const bool primitive_mmvf_add_closed =
      HasPrimitiveMmvfAddClosure(lowered, registry, launch_context, mul_mat_selector);
  const auto matmul_add_helper = [&](const ggml_tensor* tensor) {
    for (const auto& step : lowered.plan.steps) {
      if (step.implementation != kMulMatAddFused ||
          step.operation != ir::Fused(ir::FusedKind::kMatMulAdd) || step.nodes.size() != 2 ||
          !std::ranges::contains(step.nodes, const_cast<ggml_tensor*>(tensor)))
        continue;
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end() || !launch_context || !mul_mat_selector) return false;
      const auto chain =
          MulMatAddFusionAt(lowered.nodes, static_cast<std::size_t>(first - lowered.nodes.begin()));
      const auto selected = mul_mat_selector(*launch_context, step.nodes[0]);
      return chain && chain->mul_mat == step.nodes[0] && chain->add == step.nodes[1] && selected &&
             *selected == MulMatPath::kVector &&
             CheckComparisonMatMulAdd(step.nodes[0], step.nodes[1]);
    }
    return false;
  };
  for (const auto& step : lowered.plan.steps) {
    if (step.operation == ir::Fused(ir::FusedKind::kRmsNormMulAdd)) {
      if (step.implementation != kGemmaNormAddName || step.nodes.size() != 3 ||
          !CheckComparisonNormAdd(step.nodes[0], step.nodes[1], step.nodes[2]))
        return false;
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end()) return false;
      const auto chain = GemmaNormAddFusionAt(
          lowered.nodes, static_cast<std::size_t>(first - lowered.nodes.begin()));
      if (!chain || chain->norm != step.nodes[0] || chain->mul != step.nodes[1] ||
          chain->out != step.nodes[2])
        return false;
    }
    if (step.operation == ir::Fused(ir::FusedKind::kMatMulAdd) &&
        (step.nodes.size() != 2 || !matmul_add_helper(step.nodes[0]) ||
         !matmul_add_helper(step.nodes[1])))
      return false;
  }
  bool norm_rope_helper_closed = false;
  const bool primitive_mmvf_glu_closed =
      HasPrimitiveF16MmvfGluClosure(lowered, registry, launch_context, mul_mat_selector);
  const auto mmvf_glu_helper = [&](const ggml_tensor* tensor) {
    for (const auto& step : lowered.plan.steps) {
      if ((step.implementation != kMulMatGluFused && step.implementation != kMulMatGeGluFused) ||
          step.operation != ir::Fused(ir::FusedKind::kMatMulGlu) || step.nodes.size() != 3 ||
          !std::ranges::contains(step.nodes, const_cast<ggml_tensor*>(tensor)))
        continue;
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end() ||
          static_cast<std::size_t>(lowered.nodes.end() - first) < step.nodes.size())
        return false;
      const auto index = static_cast<std::size_t>(first - lowered.nodes.begin());
      const auto nodes = MulMatGluFusionAt(lowered.nodes, index);
      if (!nodes || nodes->gate != step.nodes[0] || nodes->up != step.nodes[1] ||
          nodes->glu != step.nodes[2])
        return false;
      if (step.implementation == kMulMatGluFused)
        return CheckMulMatVecGlu(nodes->gate, nodes->up, nodes->glu).has_value();
      return CheckMulMatVecGeGlu(nodes->gate, nodes->up, nodes->glu).has_value();
    }
    return false;
  };
  bool matmul_glu_helper_closed = false;
  const auto rope_store_helper = [&](const ggml_tensor* tensor) {
    for (const auto& step : lowered.plan.steps) {
      const bool ordinary = step.implementation == kRopeSetRowsFused;
      const bool quantized = step.implementation == kRopeSetRowsQuantName;
      if ((!ordinary && !quantized) || step.operation != ir::Fused(ir::FusedKind::kRopeStore) ||
          step.nodes.size() != 2 ||
          !std::ranges::contains(step.nodes, const_cast<ggml_tensor*>(tensor)))
        continue;
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end()) return false;
      const auto index = static_cast<std::size_t>(first - lowered.nodes.begin());
      const auto nodes = ordinary ? RopeSetRowsFusionAt(lowered.nodes, index)
                                  : RopeQuantStoreFusionAt(lowered.nodes, index);
      if (!nodes || nodes->rope != step.nodes[0] || nodes->set_rows != step.nodes[1]) return false;
      return ordinary ? CheckRopeSetRows(nodes->rope, nodes->set_rows).has_value()
                      : CheckRopeSetRowsQuant(nodes->rope, nodes->set_rows).has_value();
    }
    return false;
  };
  bool rope_store_helper_closed = false;
  const auto grouped_store_helper = [&](const ggml_tensor* tensor) {
    for (const auto& step : lowered.plan.steps) {
      const bool ordinary = step.implementation == kSetRowsGroupedName;
      const bool quantized = step.implementation == kSetRowsQuantGroupName;
      const bool operation_matches = step.operation == ir::Fused(ir::FusedKind::kGroupedStores);
      if ((!ordinary && !quantized) || !operation_matches || step.nodes.size() < 2 ||
          step.nodes.size() > kSetRowsGroupMax ||
          !std::ranges::contains(step.nodes, const_cast<ggml_tensor*>(tensor)))
        continue;
      std::array<const ggml_tensor*, kSetRowsGroupMax> nodes{};
      for (std::size_t i = 0; i < step.nodes.size(); ++i) nodes[i] = step.nodes[i];
      const auto view = std::span<const ggml_tensor* const>(nodes).first(step.nodes.size());
      return ordinary ? CheckSetRowsGroup(view).has_value()
                      : CheckSetRowsQuantGroup(view).has_value();
    }
    return false;
  };
  const auto physical_root_covers = [&](const ggml_tensor* tensor) {
    if (!tensor) return false;
    const auto descriptor =
        std::ranges::find(lowered.physical_tensors, tensor, &PhysicalTensor::descriptor);
    if (descriptor == lowered.physical_tensors.end()) return false;
    const auto bytes = static_cast<std::uint64_t>(ggml_nbytes(tensor));
    return std::ranges::any_of(lowered.physical_roots, [&](const auto& root) {
      return root.backend_tensor == descriptor->id && root.bytes >= bytes;
    });
  };
  const auto hc_post_norm_helper = [&](const PlanStep& step) {
    if (step.implementation == kDsv4HcPostNormF16Name) {
      if (step.operation != ir::Fused(ir::FusedKind::kHcNormPost) || step.nodes.size() != 2 ||
          !step.nodes[0] || !step.nodes[1])
        return false;
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end()) return false;
      const auto nodes =
          Dsv4HcPostNormF16At(std::span<ggml_tensor* const>(lowered.nodes),
                              static_cast<std::size_t>(first - lowered.nodes.begin()));
      return nodes && nodes->post == step.nodes[0] && nodes->norm == step.nodes[1] &&
             CheckDsv4HcPostNormF16(nodes->post, nodes->norm).has_value() &&
             physical_root_covers(nodes->post) && physical_root_covers(nodes->norm);
    }
    if (step.implementation == kHcPostName) {
      return step.operation == ir::Primitive(ir::Op::kHcPost) && step.nodes.size() == 1 &&
             step.nodes[0] != nullptr && step.nodes[0]->op == GGML_OP_DSV4_HC_POST &&
             CheckHcPost(step.nodes[0]).has_value() && physical_root_covers(step.nodes[0]);
    }
    return false;
  };
  bool hc_post_norm_helper_closed = false;
  for (const auto& step : lowered.plan.steps) {
    if (step.operation == ir::Fused(ir::FusedKind::kMatMulGlu) &&
        (step.implementation == kMulMatGluFused || step.implementation == kMulMatGeGluFused)) {
      if (step.nodes.size() != 3 || !mmvf_glu_helper(step.nodes[0]) ||
          !mmvf_glu_helper(step.nodes[1]) || !mmvf_glu_helper(step.nodes[2]))
        return false;
      matmul_glu_helper_closed = true;
    }
    if (step.operation == ir::Fused(ir::FusedKind::kRopeStore) &&
        (step.implementation == kRopeSetRowsFused ||
         step.implementation == kRopeSetRowsQuantName)) {
      if (step.nodes.size() != 2 || !rope_store_helper(step.nodes[0]) ||
          !rope_store_helper(step.nodes[1]))
        return false;
      rope_store_helper_closed = true;
    }
    if (step.implementation == kDsv4HcPostExpertsNormF16Name) return false;
    if (step.implementation == kDsv4HcPostNormF16Name || step.implementation == kHcPostName ||
        step.operation == ir::Primitive(ir::Op::kHcPost)) {
      if (!hc_post_norm_helper(step)) return false;
      hc_post_norm_helper_closed = true;
    }
    if (step.implementation != kGemmaNormRopeName) continue;
    if (step.operation != ir::Fused(ir::FusedKind::kRmsNormMulRope) || step.nodes.size() != 3 ||
        !CheckGemmaNormRope(step.nodes[0], step.nodes[1], step.nodes[2]))
      return false;
    norm_rope_helper_closed = true;
  }
  for (const auto& physical : lowered.physical_tensors) {
    if (!physical.descriptor) return false;
    switch (physical.descriptor->op) {
      case GGML_OP_NONE:
      case GGML_OP_VIEW:
      case GGML_OP_RESHAPE:
      case GGML_OP_PERMUTE:
      case GGML_OP_TRANSPOSE:
      case GGML_OP_CONT:
      case GGML_OP_ADD:
      case GGML_OP_MUL:
      case GGML_OP_RMS_NORM:
      case GGML_OP_ROPE:
      case GGML_OP_UNARY:
        if (physical.descriptor->op == GGML_OP_UNARY && !CheckUnary(physical.descriptor))
          return false;
        break;
      case GGML_OP_CPY:
      case GGML_OP_L2_NORM:
      case GGML_OP_NORM:
      case GGML_OP_REPEAT:
      case GGML_OP_CONCAT:
      case GGML_OP_SCALE:
        if (!explicitly_checked(physical.descriptor)) return false;
        break;
      case GGML_OP_DSV4_HC_POST:
        if (!std::ranges::any_of(lowered.plan.steps, [&](const auto& step) {
              return hc_post_norm_helper(step) &&
                     std::ranges::contains(step.nodes, physical.descriptor);
            }))
          return false;
        break;
      case GGML_OP_MUL_MAT:
      case GGML_OP_GLU: {
        const bool in_closed_primitive_glu =
            primitive_mmvf_glu_closed &&
            std::ranges::any_of(lowered.plan.steps, [&](const auto& step) {
              return (step.operation == ir::Primitive(ir::Op::kMatMul) ||
                      step.operation == ir::Primitive(ir::Op::kGlu)) &&
                     std::ranges::contains(step.nodes,
                                           const_cast<ggml_tensor*>(physical.descriptor));
            });
        const bool ordinary_q8_mmvq =
            shared_q8 && std::ranges::any_of(lowered.plan.steps, [&](const auto& step) {
              return step.implementation == kMulMatVecQ && step.nodes.size() == 1 &&
                     step.nodes[0] == physical.descriptor;
            });
        const bool in_primitive_add =
            primitive_mmvf_add_closed &&
            std::ranges::any_of(lowered.plan.steps, [&](const auto& step) {
              return step.operation == ir::Primitive(ir::Op::kMatMul) &&
                     std::ranges::contains(step.nodes,
                                           const_cast<ggml_tensor*>(physical.descriptor));
            });
        if (!mmvf_glu_helper(physical.descriptor) && !matmul_add_helper(physical.descriptor) &&
            !in_closed_primitive_glu && !in_primitive_add && !ordinary_q8_mmvq &&
            !direct_float_product(physical.descriptor))
          return false;
        break;
      }
      case GGML_OP_SET_ROWS:
        if (!rope_store_helper(physical.descriptor) && !grouped_store_helper(physical.descriptor) &&
            !(std::ranges::any_of(lowered.plan.steps,
                                  [&](const auto& step) {
                                    return step.implementation == kSetRowsQuantName &&
                                           step.nodes.size() == 1 &&
                                           step.nodes[0] == physical.descriptor;
                                  }) &&
              CheckSetRowsQuant(physical.descriptor)) &&
            !(std::ranges::any_of(lowered.plan.steps,
                                  [&](const auto& step) {
                                    return step.implementation == kSetRowsName &&
                                           step.nodes.size() == 1 &&
                                           step.nodes[0] == physical.descriptor;
                                  }) &&
              CheckSetRows(physical.descriptor)))
          return false;
        break;
      case GGML_OP_CUSTOM:
        switch (LlmpOpOf(physical.descriptor)) {
          case LlmpOp::kQuantizeQ8:
            if (!shared_q8 || !std::ranges::any_of(lowered.plan.steps, [&](const auto& step) {
                  return step.implementation == kQuantizeQ8Name && step.nodes.size() == 1 &&
                         step.nodes[0] == physical.descriptor;
                }))
              return false;
            break;
          case LlmpOp::kMmvqPrepared:
            if (!shared_q8 || !std::ranges::any_of(lowered.plan.steps, [&](const auto& step) {
                  return step.implementation == kMmvqPreparedName && step.nodes.size() == 1 &&
                         step.nodes[0] == physical.descriptor;
                }))
              return false;
            break;
          case LlmpOp::kGdnNormGate:
            if (!CheckGdnNormGate(physical.descriptor)) return false;
            break;
          case LlmpOp::kQsaPrep:
            // The schema is typed, but its internal prep accesses are not yet
            // represented in the sealed recipe. In particular, a separate
            // closed norm/RoPE helper in this graph cannot authorize this
            // QSA node through a graph-wide success flag.
            return false;
          case LlmpOp::kDsv4HcNormF16:
            if (!CheckDsv4HcNormF16(physical.descriptor) ||
                !std::ranges::any_of(lowered.plan.steps, [&](const auto& step) {
                  return step.implementation == kDsv4HcPostNormF16Name &&
                         hc_post_norm_helper(step) &&
                         std::ranges::contains(step.nodes, physical.descriptor);
                }))
              return false;
            break;
          default:
            if (!explicitly_checked(physical.descriptor)) return false;
            break;
        }
        break;
      default:
        return false;
    }
  }
  if (!norm_rope_helper_closed && std::ranges::any_of(lowered.plan.steps, [](const auto& step) {
        return step.operation == ir::Fused(ir::FusedKind::kRmsNormMulRope);
      }))
    return false;
  if (!matmul_glu_helper_closed && std::ranges::any_of(lowered.plan.steps, [](const auto& step) {
        return step.operation == ir::Fused(ir::FusedKind::kMatMulGlu);
      }))
    return false;
  if (!rope_store_helper_closed && std::ranges::any_of(lowered.plan.steps, [](const auto& step) {
        return step.operation == ir::Fused(ir::FusedKind::kRopeStore);
      }))
    return false;
  if (!hc_post_norm_helper_closed && std::ranges::any_of(lowered.plan.steps, [](const auto& step) {
        return step.implementation == kDsv4HcPostNormF16Name ||
               step.implementation == kDsv4HcPostExpertsNormF16Name ||
               step.implementation == kHcPostName ||
               step.operation == ir::Primitive(ir::Op::kHcPost);
      }))
    return false;
  for (const auto& root : lowered.physical_roots) {
    std::uint64_t end = 0;
    if (__builtin_add_overflow(root.offset, root.bytes, &end) || end > lowered.extent) return false;
  }
  return true;
}

}  // namespace

bool CheckComparisonMatMulAdd(const ggml_tensor* product, const ggml_tensor* add) {
  if (!product || !add || !product->src[0]) return false;
  const auto type = product->src[0]->type;
  return (type == GGML_TYPE_F16 || type == GGML_TYPE_F32) &&
         (type != GGML_TYPE_F16 || product->op_params[0] == GGML_PREC_DEFAULT) &&
         HasTypedDescriptorSchema(*product) && HasTypedDescriptorSchema(*add) &&
         CheckMulMatVecBias(product, add).has_value();
}

bool CheckComparisonNormAdd(const ggml_tensor* norm, const ggml_tensor* mul,
                            const ggml_tensor* add) {
  return norm && mul && add && HasTypedDescriptorSchema(*norm) && HasTypedDescriptorSchema(*mul) &&
         HasTypedDescriptorSchema(*add) && CheckGemmaNormAdd(norm, mul, add).has_value();
}

bool CheckComparisonF16SwiGlu(const ggml_tensor* gate, const ggml_tensor* up,
                              const ggml_tensor* glu) {
  return gate && up && glu && gate != up && gate->op == GGML_OP_MUL_MAT &&
         up->op == GGML_OP_MUL_MAT && gate->src[0] && up->src[0] && gate->src[1] &&
         gate->src[1] == up->src[1] && gate->src[0]->type == GGML_TYPE_F16 &&
         up->src[0]->type == GGML_TYPE_F16 && gate->src[1]->type == GGML_TYPE_F32 &&
         gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 &&
         gate->op_params[0] == GGML_PREC_DEFAULT && up->op_params[0] == GGML_PREC_DEFAULT &&
         glu->src[0] == gate && glu->src[1] == up && HasTypedDescriptorSchema(*glu) &&
         CheckSwiGlu(glu).has_value();
}

std::expected<ComparisonQ8SpanRecipe, std::string_view> DescribeComparisonQ8Span(
    std::span<const ComparisonQ8StepRef> steps, const LaunchContext& launch,
    const ComparisonQ8Dispatch& dispatch) {
  if (!dispatch.complete()) return std::unexpected("Q8 dispatch provider is incomplete");
  ComparisonQ8SpanRecipe recipe;
  std::size_t prepare_count = 0, ordinary_count = 0, prepared_count = 0;
  const PlanStep* prepare_step = nullptr;
  std::size_t prepare_position = 0;
  for (const auto& reference : steps) {
    if (!reference.step) return std::unexpected("Q8 span contains a missing actual call");
    const auto& step = *reference.step;
    if (step.implementation == kQuantizeQ8Name) {
      if (step.operation != ir::Primitive(ir::Op::kPrepare) || step.nodes.size() != 1 ||
          !step.nodes.front() || step.nodes.front()->op != GGML_OP_CUSTOM ||
          LlmpOpOf(step.nodes.front()) != LlmpOp::kQuantizeQ8 ||
          !CheckQuantizeQ8(step.nodes.front()))
        return std::unexpected("Q8 preparation call has an invalid typed descriptor");
      if (++prepare_count != 1 || recipe.call_count == recipe.calls.size())
        return std::unexpected("Q8 span has duplicate or excessive preparation calls");
      auto quantizer = dispatch.describe_quantize_q8(step.nodes.front());
      if (!quantizer) return std::unexpected("Q8 preparation dispatch is unavailable");
      prepare_step = &step;
      prepare_position = reference.ordinal;
      recipe.prepared = step.nodes.front();
      recipe.input = step.nodes.front()->src[0];
      recipe.calls[recipe.call_count++] = {.kind = ComparisonQ8CallKind::kQuantize,
                                           .step = &step,
                                           .step_ordinal = reference.ordinal,
                                           .node = step.nodes.front(),
                                           .input = recipe.input,
                                           .packed = step.nodes.front(),
                                           .quantizer = *quantizer};
      continue;
    }
    const bool ordinary = step.implementation == kMulMatVecQ;
    const bool prepared = step.implementation == kMmvqPreparedName;
    if (!ordinary && !prepared) continue;
    if (step.operation != ir::Primitive(ir::Op::kMatMul) || step.nodes.size() != 1 ||
        !step.nodes.front() || recipe.product_count == recipe.products.size() ||
        recipe.call_count == recipe.calls.size())
      return std::unexpected("MMVQ span has an invalid actual call descriptor");
    const auto* node = step.nodes.front();
    const ggml_tensor* input = nullptr;
    const ggml_tensor* weight = nullptr;
    const ggml_tensor* packed = nullptr;
    MmvqDispatchDescriptor mmvq{};
    Q8QuantizeDispatchDescriptor quantizer{};
    std::uint64_t scratch = 0;
    if (ordinary) {
      if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1])
        return std::unexpected("ordinary MMVQ is not a dense matrix-product descriptor");
      if (!CheckMulMatQ(node))
        return std::unexpected("ordinary MMVQ fails its typed operand and write checker");
      auto described = dispatch.describe_mmvq(launch, node, false);
      auto quantized = dispatch.describe_quantize_q8(node);
      auto planned = dispatch.plan_mmvq(launch, node);
      if (!described || !quantized || !planned || described->immutable_weights ||
          described->fused || *planned == 0 || *planned != quantized->output_bytes)
        return std::unexpected("ordinary MMVQ dispatch or private Q8 draw is incomplete");
      input = node->src[1];
      weight = node->src[0];
      mmvq = *described;
      quantizer = *quantized;
      scratch = *planned;
      ++ordinary_count;
    } else {
      if (node->op != GGML_OP_CUSTOM || LlmpOpOf(node) != LlmpOp::kMmvqPrepared)
        return std::unexpected("prepared MMVQ has an invalid typed custom descriptor");
      auto original = MmvqPreparedOriginal(node);
      auto described = dispatch.describe_mmvq(launch, node, true);
      if (!original || !CheckMmvqPrepared(node))
        return std::unexpected("prepared MMVQ original descriptor is invalid");
      auto quantized = dispatch.describe_quantize_q8(&*original);
      auto planned = dispatch.plan_prepared_mmvq(launch, node);
      if (!original || !described || !quantized || !planned || *planned != 0 ||
          !described->immutable_weights || described->fused || !node->src[1] || !node->src[2] ||
          original->src[0] == nullptr || original->src[1] != node->src[2] ||
          node->src[1]->src[0] != node->src[2])
        return std::unexpected("prepared MMVQ dispatch differs from its actual producer lineage");
      input = original->src[1];
      weight = original->src[0];
      packed = node->src[1];
      mmvq = *described;
      quantizer = *quantized;
      ++prepared_count;
    }
    if (!input || input->type != GGML_TYPE_F32 || !weight || !ggml_is_contiguous(weight) ||
        (weight->type != GGML_TYPE_Q4_K && weight->type != GGML_TYPE_Q6_K) ||
        weight->ne[0] != input->ne[0] || input->ne[2] != 1 || input->ne[3] != 1 ||
        mmvq.weight_type != weight->type || mmvq.k != input->ne[0] ||
        mmvq.columns != input->ne[1] || mmvq.row_high != weight->ne[1] || mmvq.padded_k < mmvq.k ||
        quantizer.k != input->ne[0] || quantizer.rows != input->ne[1] || quantizer.channels != 1 ||
        quantizer.samples != 1 || quantizer.output_bytes == 0)
      return std::unexpected("MMVQ typed dispatch dimensions disagree with its actual operands");
    if (recipe.input && recipe.input != input)
      return std::unexpected("shared MMVQ consumers do not use one authenticated F32 source");
    recipe.input = input;
    recipe.products[recipe.product_count] = node;
    recipe.weights[recipe.product_count] = weight;
    ++recipe.product_count;
    recipe.calls[recipe.call_count++] = {
        .kind = ordinary ? ComparisonQ8CallKind::kMmvq : ComparisonQ8CallKind::kMmvqPrepared,
        .step = &step,
        .step_ordinal = reference.ordinal,
        .node = node,
        .input = input,
        .weight = weight,
        .packed = packed,
        .mmvq = mmvq,
        .quantizer = quantizer,
        .scratch_bytes = scratch};
  }
  if (recipe.product_count != 2 || ordinary_count + prepared_count != 2 || ordinary_count > 2 ||
      prepared_count > 2 || prepare_count > 1 ||
      (prepare_count == 0 && (ordinary_count != 2 || prepared_count != 0)) ||
      (prepare_count == 1 && prepared_count == 0) || recipe.weights[0] == recipe.weights[1] ||
      !((recipe.weights[0]->type == GGML_TYPE_Q4_K && recipe.weights[1]->type == GGML_TYPE_Q6_K) ||
        (recipe.weights[1]->type == GGML_TYPE_Q4_K && recipe.weights[0]->type == GGML_TYPE_Q6_K)))
    return std::unexpected("Q8 span is not the exact supported Q4_K/Q6_K consumer set");
  if (prepare_count != 0) {
    auto* prepared = recipe.prepared;
    auto producer = std::ranges::find_if(recipe.calls, [](const auto& call) {
      return call.kind == ComparisonQ8CallKind::kQuantize;
    });
    if (!prepared || !prepare_step || producer == recipe.calls.end() || !producer->input ||
        producer->input != recipe.input)
      return std::unexpected("Q8 shared producer differs from the MMVQ activation geometry");
    for (std::size_t i = 0; i < recipe.product_count; ++i) {
      const auto* product = recipe.products[i];
      const auto call = std::ranges::find_if(
          recipe.calls, [&](const auto& candidate) { return candidate.node == product; });
      if (call == recipe.calls.end() ||
          call->quantizer.output_bytes != producer->quantizer.output_bytes)
        return std::unexpected("Q8 producer and consumer padding extents disagree");
      if (call->kind == ComparisonQ8CallKind::kMmvqPrepared &&
          (call->step == nullptr || call->step_ordinal <= prepare_position ||
           product->src[1] != prepared || product->src[2] != recipe.input))
        return std::unexpected("prepared MMVQ precedes or differs from its Q8 producer");
      if (call->kind == ComparisonQ8CallKind::kMmvq && product->src[1] != recipe.input)
        return std::unexpected("ordinary MMVQ differs from the shared source input");
    }
  }
  return recipe;
}

base::Sha256Digest ComparisonMmvqDispatchDigest(const MmvqDispatchDescriptor& descriptor) {
  CanonicalHash hash("llmp.comparison.dispatch.mmvq.v1");
  HashComparisonMmvqDispatch(hash, descriptor);
  return hash.Finish();
}

base::Sha256Digest ComparisonLocalMmvqDispatchDigest(const MmvqDispatchDescriptor& descriptor) {
  CanonicalHash hash("llmp.comparison.dispatch.mmvq.local.v1");
  HashComparisonLocalMmvqDispatch(hash, descriptor);
  return hash.Finish();
}

base::Sha256Digest ComparisonQ8QuantizeDispatchDigest(
    const Q8QuantizeDispatchDescriptor& descriptor) {
  CanonicalHash hash("llmp.comparison.dispatch.q8_quantize.v1");
  HashComparisonQ8QuantizeDispatch(hash, descriptor);
  return hash.Finish();
}

std::optional<std::uint64_t> ComparisonMmvqWeightReadableBytes(
    const ggml_tensor& weight, const MmvqDispatchDescriptor& dispatch) {
  if ((weight.type != GGML_TYPE_Q4_K && weight.type != GGML_TYPE_Q6_K) || weight.ne[0] <= 0 ||
      dispatch.k != weight.ne[0] || dispatch.padded_k < dispatch.k)
    return std::nullopt;
  const auto block = ggml_blck_size(weight.type);
  const auto tail_elements = dispatch.padded_k - dispatch.k;
  if (block <= 0 || tail_elements < 0 || tail_elements % block != 0) return std::nullopt;
  const auto logical = static_cast<std::uint64_t>(ggml_nbytes(&weight));
  const auto tail = static_cast<std::uint64_t>(ggml_row_size(weight.type, tail_elements));
  std::uint64_t readable = 0;
  if (logical == 0 || __builtin_add_overflow(logical, tail, &readable)) return std::nullopt;
  return readable;
}

bool ComparisonPoolDisjoint(const LoweredStep& lowered, const Q8MainPoolDescriptor& pool,
                            std::uint64_t bytes) {
  if (bytes == 0) return true;
  if (pool.faulted || pool.lanes != 0 || bytes > pool.bytes || pool.base % 256 != 0) return false;
  const auto base = static_cast<std::uintptr_t>(pool.base);
  std::uintptr_t end = 0;
  if (base == 0 || __builtin_add_overflow(base, static_cast<std::uintptr_t>(bytes), &end))
    return false;
  const auto overlaps = [&](std::uintptr_t other, std::uint64_t other_bytes) {
    std::uintptr_t other_end = 0;
    return other == 0 || other_bytes == 0 ||
           __builtin_add_overflow(other, static_cast<std::uintptr_t>(other_bytes), &other_end) ||
           (base < other_end && other < end);
  };
  for (const auto& root : lowered.physical_roots) {
    const auto descriptor =
        std::ranges::find(lowered.physical_tensors, root.backend_tensor, &PhysicalTensor::id);
    if (descriptor == lowered.physical_tensors.end() || !descriptor->descriptor ||
        !descriptor->descriptor->data ||
        overlaps(reinterpret_cast<std::uintptr_t>(descriptor->descriptor->data), root.bytes))
      return false;
  }
  for (const auto& range : lowered.comparison_bindings) {
    std::uintptr_t begin = 0;
    if (range.base == 0 || range.bytes == 0 ||
        __builtin_add_overflow(range.base, static_cast<std::uintptr_t>(range.offset), &begin) ||
        overlaps(begin, range.bytes))
      return false;
  }
  return true;
}

bool ComparisonSgemmCall(const PlanStep& step) {
  if (step.implementation != kMulMatCublas || step.operation != ir::Primitive(ir::Op::kMatMul) ||
      step.nodes.size() != 1 || !step.nodes[0])
    return false;
  const auto* t = step.nodes[0];
  if (!HasTypedDescriptorSchema(*t) || !t->src[0] || !t->src[1] || t->type != GGML_TYPE_F32 ||
      t->src[0]->type != GGML_TYPE_F32 || t->src[1]->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(t) || !ggml_is_contiguous(t->src[0]) || !ggml_is_contiguous(t->src[1]) ||
      t->ne[2] != 1 || t->ne[3] != 1 || t->src[0]->ne[2] != 1 || t->src[0]->ne[3] != 1 ||
      t->src[1]->ne[2] != 1 || t->src[1]->ne[3] != 1)
    return false;
  const auto p = CheckMulMatCublas(t, GGML_TYPE_F32, false);
  return p && p->compute == GGML_TYPE_F32 && !p->f32_output &&
         p->weights == CublasOperand::kDirect && p->input == CublasOperand::kDirect &&
         p->gemm == CublasGemm::kSgemm && p->scratch == 0;
}

bool CheckComparisonSgemm(const PlanStep& step, const ComparisonRemainderCallFacts& facts) {
  if (!ComparisonSgemmCall(step) || !facts.has_mul_mat_selection || facts.mul_mat_vector ||
      !facts.cublas_plan || facts.scratch_bytes != 0 || facts.library_workspace_base == 0 ||
      facts.library_workspace_base % 256 != 0 || facts.library_workspace_bytes == 0 ||
      std::ranges::all_of(facts.cublas_configuration, [](auto byte) { return byte == 0; }))
    return false;
  const auto* t = step.nodes[0];
  if (!HasTypedDescriptorSchema(*t) || !t->src[0] || !t->src[1] || t->type != GGML_TYPE_F32 ||
      t->src[0]->type != GGML_TYPE_F32 || t->src[1]->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(t) || !ggml_is_contiguous(t->src[0]) || !ggml_is_contiguous(t->src[1]) ||
      t->ne[2] != 1 || t->ne[3] != 1 || t->src[0]->ne[2] != 1 || t->src[0]->ne[3] != 1 ||
      t->src[1]->ne[2] != 1 || t->src[1]->ne[3] != 1)
    return false;
  const auto checked = CheckMulMatCublas(t, GGML_TYPE_F32, false);
  if (!checked) return false;
  const auto& p = *facts.cublas_plan;
  return p.compute == GGML_TYPE_F32 && !p.f32_output && p.weights == CublasOperand::kDirect &&
         p.input == CublasOperand::kDirect && p.gemm == CublasGemm::kSgemm && p.scratch == 0 &&
         p.s01 == checked->s01 && p.s02 == checked->s02 && p.s03 == checked->s03 &&
         p.s11 == checked->s11 && p.s12 == checked->s12 && p.s13 == checked->s13 &&
         p.alignment == checked->alignment &&
         CheckClearOf(t, facts.library_workspace_base, facts.library_workspace_bytes).has_value();
}

bool CheckComparisonWholeCublas(const PlanStep& step, const ComparisonRemainderCallFacts& facts) {
  if (step.implementation != kMulMatCublas || step.operation != ir::Primitive(ir::Op::kMatMul) ||
      step.nodes.size() != 1 || !step.nodes[0] || step.lane != 0 || !facts.has_mul_mat_selection ||
      facts.mul_mat_vector || facts.float_selection != MulMatPath::kCublas || !facts.cublas_plan ||
      facts.pool_faulted || facts.pool_lanes != 0 || !facts.library_workspace_base ||
      facts.library_workspace_base % 256 != 0 || !facts.library_workspace_bytes ||
      std::ranges::all_of(facts.cublas_configuration, [](auto byte) { return byte == 0; }))
    return false;
  std::uint64_t workspace_end = 0;
  if (__builtin_add_overflow(facts.library_workspace_base, facts.library_workspace_bytes,
                             &workspace_end))
    return false;
  const auto* t = step.nodes[0];
  if (!HasTypedDescriptorSchema(*t) || !t->src[0] || !t->src[1]) return false;
  const auto& p = *facts.cublas_plan;
  // Compute type/output precision come from the owned device's actual
  // PlanMulMatCublas callback. Check the complete resulting plan against the
  // operands, including broadcasts, nonpacked inputs and pointer-array scratch.
  // The native selector only preserves the weight type or promotes it to F32.
  if ((p.compute != t->src[0]->type && p.compute != GGML_TYPE_F32) ||
      (t->op_params[0] == GGML_PREC_F32 && p.compute != GGML_TYPE_F32) ||
      (p.compute == GGML_TYPE_F32 && p.f32_output))
    return false;
  const auto checked = CheckMulMatCublas(t, p.compute, p.f32_output);
  if (!checked || p.compute != checked->compute || p.f32_output != checked->f32_output ||
      p.weights != checked->weights || p.input != checked->input || p.gemm != checked->gemm ||
      p.scratch != checked->scratch || facts.scratch_bytes != p.scratch || p.s01 != checked->s01 ||
      p.s02 != checked->s02 || p.s03 != checked->s03 || p.s11 != checked->s11 ||
      p.s12 != checked->s12 || p.s13 != checked->s13 || p.alignment != checked->alignment ||
      !CheckClearOf(t, facts.library_workspace_base, facts.library_workspace_bytes))
    return false;
  // Descriptor construction has no execution pool yet. Its exact scratch is
  // compared with the planned metrics; Bind requires the complete paid pool.
  // An already present pool must be usable now, not deferred to that check.
  if (facts.pool_base == 0 && facts.pool_bytes == 0) return true;
  std::uint64_t pool_end = 0;
  return facts.pool_base != 0 && facts.pool_base % 256 == 0 && facts.pool_bytes >= p.scratch &&
         !__builtin_add_overflow(facts.pool_base, facts.pool_bytes, &pool_end) &&
         CheckClearOf(t, facts.pool_base, facts.pool_bytes).has_value();
}

bool CheckComparisonMmf(const PlanStep& step, const ComparisonRemainderCallFacts& facts) {
  return step.implementation == "ggml.mul_mat.mmf" &&
         step.operation == ir::Primitive(ir::Op::kMatMul) && step.nodes.size() == 1 &&
         step.nodes[0] && HasTypedDescriptorSchema(*step.nodes[0]) &&
         CheckMulMatF(step.nodes[0]).has_value() && step.lane == 0 &&
         facts.float_selection == MulMatPath::kTensorCore && facts.scratch_bytes == 0;
}

bool ComparisonBf16Call(const PlanStep& step) {
  if ((step.implementation != "llmp.gemm.bf16" && step.implementation != kGemmBf16LtName) ||
      step.operation != ir::Primitive(ir::Op::kMatMul) || step.nodes.size() != 1 || !step.nodes[0])
    return false;
  const auto* node = step.nodes[0];
  if (LlmpOpOf(node) != LlmpOp::kGemmBf16 || !HasWholeNativeCustomScalarSchema(*node) ||
      !CheckGemmBf16(node))
    return false;
  const auto mode = LlmpOpInt(node, 0);
  return mode >= 0 && mode <= 3 && ((step.implementation == kGemmBf16LtName) == (mode == 3));
}

bool CheckComparisonBf16(const PlanStep& step, const ComparisonRemainderCallFacts& facts) {
  if (!ComparisonBf16Call(step) || !facts.bf16_product || facts.scratch_bytes != 0 ||
      facts.cublas_plan || step.lane != 0)
    return false;
  const auto* node = step.nodes[0];
  if (*facts.bf16_product == ComparisonBf16Product::kVector) return UsesGemvBf16Vector(node);
  if (*facts.bf16_product != ComparisonBf16Product::kGemmEx || UsesGemvBf16Vector(node) ||
      !facts.library_workspace_base || facts.library_workspace_base % 256 ||
      !facts.library_workspace_bytes ||
      std::ranges::all_of(facts.cublas_configuration, [](auto byte) { return byte == 0; }))
    return false;
  return CheckClearOf(node, facts.library_workspace_base, facts.library_workspace_bytes)
      .has_value();
}

namespace {
void HashComparisonBf16Facts(CanonicalHash& hash, const ComparisonRemainderCallFacts& facts) {
  hash.Text("llmp.cuda.bf16-product.v1");
  hash.U8(static_cast<std::uint8_t>(*facts.bf16_product));
  // The exact descriptor supplies packed K/N/T, output type and vector mode.
  // GemmEx is T,N with alpha=1, beta=0, BF16 operands, COMPUTE_32F and
  // DEFAULT_TENSOR_OP; these fixed constants are versioned by this tag/build.
  if (*facts.bf16_product == ComparisonBf16Product::kGemmEx) {
    hash.Digest(facts.cublas_configuration);
    hash.U64(facts.library_workspace_bytes);
  }
}
}  // namespace

bool ComparisonLibraryWorkspaceDisjoint(const LoweredStep& lowered,
                                        const ComparisonRemainderCallFacts& facts) {
  const auto begin = facts.library_workspace_base;
  std::uint64_t end = 0;
  if (!begin || !facts.library_workspace_bytes ||
      __builtin_add_overflow(begin, facts.library_workspace_bytes, &end))
    return false;
  const auto clear = [&](std::uint64_t address, std::uint64_t bytes) {
    std::uint64_t limit = 0;
    return address && bytes && !__builtin_add_overflow(address, bytes, &limit) &&
           (limit <= begin || end <= address);
  };
  if (facts.pool_faulted || facts.pool_lanes != 0 ||
      (facts.pool_bytes != 0 && !clear(facts.pool_base, facts.pool_bytes)))
    return false;
  for (const auto& physical : lowered.physical_tensors) {
    const auto* t = physical.descriptor;
    if (!t || !t->data || !clear(reinterpret_cast<std::uintptr_t>(t->data), ggml_nbytes(t)))
      return false;
  }
  for (const auto& root : lowered.physical_roots) {
    const auto found =
        std::ranges::find(lowered.physical_tensors, root.backend_tensor, &PhysicalTensor::id);
    const auto* tensor = found == lowered.physical_tensors.end() ? nullptr : &*found;
    if (!tensor || !tensor->descriptor ||
        !clear(reinterpret_cast<std::uintptr_t>(tensor->descriptor->data), root.bytes))
      return false;
  }
  for (const auto& binding : lowered.comparison_bindings) {
    std::uint64_t address = 0;
    if (__builtin_add_overflow(binding.base, binding.offset, &address) ||
        !clear(address, binding.bytes))
      return false;
  }
  return true;
}

std::optional<base::Sha256Digest> ComparisonProductScheduleIdentity(
    const LoweredStep& lowered, const LaunchContext& context,
    ComparisonRemainderCallFactsFn describe) {
  if (!describe) return {};
  CanonicalHash hash("llmp.ggml.product-schedule.v3");
  const auto included = [](const PlanStep& step) {
    return step.implementation == kMulMatCublas || step.implementation == "llmp.gemm.bf16" ||
           step.implementation == kGemmBf16LtName || step.implementation == "ggml.mul_mat.mmf";
  };
  hash.U64(static_cast<std::uint64_t>(std::ranges::count_if(lowered.plan.steps, included)));
  for (const auto& step : lowered.plan.steps) {
    if (!included(step)) continue;
    ComparisonRemainderCallFacts facts;
    if (!describe(context, step, facts)) return {};
    hash.Text(step.implementation);
    if (step.implementation == "ggml.mul_mat.mmf") {
      if (!CheckComparisonMmf(step, facts)) return {};
      hash.U8(static_cast<std::uint8_t>(*facts.float_selection));
      hash.U64(facts.scratch_bytes);
      for (const auto* tensor : {step.nodes[0], step.nodes[0]->src[0], step.nodes[0]->src[1]}) {
        const auto* descriptor = PhysicalTensorOf(lowered, tensor);
        if (!descriptor || !HashPhysicalDescriptor(hash, lowered, *descriptor)) return {};
      }
    } else if (step.implementation == kMulMatCublas) {
      const bool whole_cublas =
          lowered.comparison &&
          lowered.comparison->scope == ComparisonProofScope::kWholeProgramNative &&
          CheckComparisonWholeCublas(step, facts);
      if ((!CheckComparisonSgemm(step, facts) && !whole_cublas) ||
          !ComparisonLibraryWorkspaceDisjoint(lowered, facts))
        return {};
      HashComparisonCublasFacts(hash, facts);
      for (const auto* tensor : {step.nodes[0], step.nodes[0]->src[0], step.nodes[0]->src[1]}) {
        const auto* descriptor = PhysicalTensorOf(lowered, tensor);
        if (!descriptor || !HashPhysicalDescriptor(hash, lowered, *descriptor)) return {};
      }
    } else {
      if (!CheckComparisonBf16(step, facts) ||
          (*facts.bf16_product == ComparisonBf16Product::kGemmEx &&
           !ComparisonLibraryWorkspaceDisjoint(lowered, facts)))
        return {};
      HashComparisonBf16Facts(hash, facts);
    }
  }
  return hash.Finish();
}

bool HasTypedDescriptorSchema(const ggml_tensor& tensor) { return HasSchema(tensor); }

std::expected<components::ComparisonBoundaryDigest, std::string> ObserveComparisonBoundary(
    const LowerInput& input, std::span<const ComparisonBindingRange> bindings,
    std::shared_ptr<const ir::StorageFunder> funder) {
  if (!input.comparison || !input.comparison->expected_boundary)
    return std::unexpected("comparison request has no owned source boundary");
  const auto& expected = *input.comparison->expected_boundary;
  if (expected.source_identity() == 0 || expected.source_identity() != input.io.graph.identity() ||
      expected.source_digest() != input.io.graph.Digest())
    return std::unexpected("comparison source identity or whole-graph subject changed");

  std::span<const components::StepState> state;
  std::uint16_t active_stage = input.comparison->active_stage;
  if (input.token) {
    state = input.token->step.state;
    active_stage = input.token->shape.stage;
    const auto same_state = [](const components::StepState& a, const components::StepState& b) {
      return a.tensor == b.tensor && a.ref == b.ref && a.item == b.item &&
             a.borrowed == b.borrowed && a.tap == b.tap && a.source == b.source &&
             a.read == b.read && a.written == b.written;
    };
    if (input.comparison->active_stage != active_stage ||
        input.comparison->state.size() != state.size() ||
        !std::ranges::equal(input.comparison->state, state, same_state))
      return std::unexpected("comparison state/stage differs from the actual token step");
  } else if (!input.comparison->state.empty()) {
    return std::unexpected("comparison state has no authenticated token step");
  }

  std::vector<components::ComparisonExternalBinding> external;
  external.reserve(bindings.size());
  for (const auto& binding : bindings) {
    external.push_back({.original_root = binding.original_root,
                        .part = binding.part,
                        .allocation_class = binding.allocation_class,
                        .allocation_bytes = binding.allocation_bytes,
                        .offset = binding.offset,
                        .bytes = binding.bytes,
                        .alignment = binding.alignment,
                        .resource_part_identity = binding.resource_part_identity,
                        .row_storage = binding.row_storage});
  }
  std::uint64_t unstored_bytes = 0;
  constexpr std::uint64_t kUnstoredVectorEnvelope = 64;
  if (__builtin_mul_overflow(
          static_cast<std::uint64_t>(expected.unstored_inputs().size()),
          static_cast<std::uint64_t>(sizeof(components::ComparisonUnstoredInput)),
          &unstored_bytes) ||
      (!expected.unstored_inputs().empty() &&
       __builtin_add_overflow(unstored_bytes, kUnstoredVectorEnvelope, &unstored_bytes)))
    return std::unexpected("comparison unstored-input observation inventory overflows");
  auto unstored_loan = ir::AcquireStorageLoan(funder, unstored_bytes);
  if (!unstored_loan)
    return std::unexpected("comparison unstored-input observation admission refused");
  std::vector<components::ComparisonUnstoredInput> unstored;
  unstored.reserve(expected.unstored_inputs().size());
  for (const auto& unstored_input : expected.unstored_inputs())
    unstored.push_back(
        {.original_input = unstored_input.original_input, .reason = unstored_input.reason});
  const components::ComparisonBoundaryView view{input.io, state, active_stage, external, unstored};
  auto observed = components::MakeComparisonBoundary(view, std::move(funder));
  if (!observed)
    return std::unexpected(std::string("comparison boundary observation failed: ") +
                           observed.error().detail);
  const auto& actual = observed->digest();
  if (actual.version != expected.digest().version || actual.logical != expected.digest().logical ||
      actual.external != expected.digest().external ||
      actual.combined != expected.digest().combined)
    return std::unexpected("comparison actual boundary differs from the owned source boundary");
  if (auto mapped = components::ValidateMappedComparisonBoundary(expected, input.plan.graph,
                                                                 input.plan.tensors);
      !mapped)
    return std::unexpected(std::string("comparison planned map differs from source: ") +
                           mapped.error().detail);
  if (auto mapped = components::ValidateMappedComparisonBoundary(*observed, input.plan.graph,
                                                                 input.plan.tensors);
      !mapped)
    return std::unexpected(std::string("comparison planned map differs from observation: ") +
                           mapped.error().detail);
  return actual;
}

bool CheckComparisonStagedInput(const LoweredStep& lowered, const LoweredLeaf& leaf,
                                const ComparisonBindingRange& actual) {
  if (!leaf.comparison_staged || leaf.comparison_slab || leaf.part || !leaf.tensor ||
      !leaf.tensor->data || leaf.tensor->op != GGML_OP_NONE || leaf.tensor->view_src ||
      actual.part || actual.row_storage || actual.resource_part_identity)
    return false;
  const auto& fact = *leaf.comparison_staged;
  if (!fact.planned.valid() || !fact.producer.valid() || fact.activation_base == 0 ||
      fact.activation_bytes == 0 || fact.bytes == 0 || fact.input_bytes == 0 ||
      fact.input_bytes > fact.bytes || fact.alignment == 0 ||
      !std::has_single_bit(fact.alignment) ||
      (fact.logical_offset == fact.physical_offset && fact.bytes == fact.input_bytes) ||
      fact.activation_base % fact.alignment || fact.logical_offset % fact.alignment ||
      fact.physical_offset % fact.alignment || actual.original_root != leaf.primitive ||
      actual.allocation_class != fact.allocation_class || actual.offset != 0 ||
      actual.bytes != fact.bytes || actual.allocation_bytes != fact.bytes ||
      actual.alignment != fact.alignment || leaf.tensor->type != fact.type ||
      ggml_nbytes(leaf.tensor) != fact.input_bytes ||
      std::ranges::count(lowered.leaves, leaf.primitive, &LoweredLeaf::primitive) != 1 ||
      std::ranges::count(lowered.leaves, leaf.tensor, &LoweredLeaf::tensor) != 1 ||
      std::ranges::count(lowered.inputs, leaf.primitive, &InputSlot::tensor) != 1)
    return false;
  std::uint64_t arena_end = 0, logical_end = 0, physical_end = 0, address = 0;
  if (__builtin_add_overflow(fact.activation_base, fact.activation_bytes, &arena_end) ||
      __builtin_add_overflow(fact.logical_offset, fact.input_bytes, &logical_end) ||
      __builtin_add_overflow(fact.physical_offset, fact.bytes, &physical_end) ||
      logical_end > fact.activation_bytes || physical_end > fact.activation_bytes ||
      __builtin_add_overflow(fact.activation_base, fact.physical_offset, &address) ||
      actual.base != address || reinterpret_cast<std::uintptr_t>(leaf.tensor->data) != address)
    return false;
  const auto slot = std::ranges::find(lowered.inputs, leaf.primitive, &InputSlot::tensor);
  if (slot->device_address != address || slot->bytes != fact.input_bytes ||
      slot->segment != fact.segment || slot->kind != fact.kind || slot->commit_save)
    return false;
  for (const auto& other : lowered.inputs) {
    if (&other == &*slot) continue;
    std::uint64_t end = 0;
    if (other.device_address == 0 || other.bytes == 0 ||
        __builtin_add_overflow(other.device_address, other.bytes, &end) ||
        (address < end && other.device_address < address + fact.bytes))
      return false;
  }
  const auto* descriptor = PhysicalTensorOf(lowered, leaf.tensor);
  if (!descriptor || descriptor->tensor != fact.planned || descriptor->root != fact.planned ||
      descriptor->producer != fact.producer || descriptor->part ||
      std::ranges::count(lowered.physical_roots, fact.planned, &PhysicalRoot::root) != 1)
    return false;
  const auto root = std::ranges::find(lowered.physical_roots, fact.planned, &PhysicalRoot::root);
  if (root->producer != fact.producer || root->operation != ir::Primitive(ir::Op::kInput) ||
      root->backend_tensor != descriptor->id || root->offset != fact.physical_offset ||
      root->bytes != fact.bytes || root->offset > fact.activation_bytes ||
      root->bytes > fact.activation_bytes - root->offset)
    return false;
  for (std::size_t d = 0; d < 4; ++d)
    if (leaf.tensor->ne[d] <= 0 || root->ne[d] != static_cast<std::uint64_t>(leaf.tensor->ne[d]) ||
        root->nb[d] != leaf.tensor->nb[d])
      return false;
  // Staged values exist together before execution. Relocation cannot alias
  // another staged input or protected native leaf, even if the IR lifetimes differ.
  for (const auto& other : lowered.leaves) {
    if (other.tensor == leaf.tensor) continue;
    if (!other.tensor || !other.tensor->data) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(other.tensor->data);
    std::uint64_t end = 0;
    if (__builtin_add_overflow(static_cast<std::uint64_t>(begin),
                               static_cast<std::uint64_t>(ggml_nbytes(other.tensor)), &end) ||
        (address < end && begin < address + fact.bytes))
      return false;
  }
  return true;
}

std::optional<ComparisonStagedInput> DescribeComparisonStagedInput(const LowerInput& input,
                                                                   const LoweredStep& lowered,
                                                                   const LoweredLeaf& leaf,
                                                                   std::size_t binding) {
  if (!input.comparison || input.comparison->scope != ComparisonProofScope::kWholeProgramNative ||
      !input.comparison->expected_boundary || input.addresses.measuring() || lowered.measuring ||
      binding >= input.comparison->bindings.size() || leaf.part || leaf.comparison_slab ||
      !leaf.tensor || !leaf.tensor->data)
    return {};
  const auto& range = input.comparison->bindings[binding];
  const auto& source = input.io.graph;
  const auto& planned = input.plan.graph;
  if (input.comparison->expected_boundary->source_identity() != source.identity() ||
      input.comparison->expected_boundary->source_digest() != source.Digest())
    return {};
  if (range.part || range.row_storage || range.resource_part_identity || range.offset != 0 ||
      range.bytes != range.allocation_bytes || range.original_root != leaf.primitive ||
      !leaf.primitive.valid() || leaf.primitive.v >= source.tensor_count() ||
      leaf.primitive.v >= input.plan.tensors.size() ||
      !std::ranges::contains(input.io.inputs, leaf.primitive) ||
      !std::ranges::contains(source.inputs(), leaf.primitive) ||
      source.root(leaf.primitive) != leaf.primitive ||
      source.tensor(leaf.primitive).role != ir::Role::kInput ||
      !std::holds_alternative<std::monostate>(source.tensor(leaf.primitive).binding) ||
      source.node(source.producer(leaf.primitive)).key != ir::Primitive(ir::Op::kInput))
    return {};
  if (input.token) {
    const auto typed =
        std::ranges::find(input.token->step.inputs, leaf.primitive, &components::StepInput::tensor);
    if (typed == input.token->step.inputs.end() ||
        typed->kind == components::InputKind::kCommitRows)
      return {};
  }
  const auto mapped = input.plan.tensors[leaf.primitive.v];
  if (!mapped.valid() || mapped.v >= planned.tensor_count() || planned.root(mapped) != mapped ||
      !std::ranges::contains(planned.inputs(), mapped))
    return {};
  const auto producer = planned.producer(mapped);
  const auto& original = source.tensor(leaf.primitive);
  const auto& current = planned.tensor(mapped);
  if (!producer.valid() || producer.v >= planned.nodes().size() ||
      planned.node(producer).key != ir::Primitive(ir::Op::kInput) ||
      current.role != ir::Role::kInput || current.format != original.format ||
      current.nb != original.nb || current.segment != original.segment ||
      current.binding != original.binding || current.view_offset != 0 || original.view_offset != 0)
    return {};
  const auto type = GgmlTypeOf(format::Info(current.format.encoding).name);
  if (!type || *type != leaf.tensor->type || current.format.layout != format::Layout::kDense)
    return {};
  const auto logical = std::ranges::find(input.plan.placement.offsets, mapped,
                                         &std::pair<ir::TensorId, std::uint64_t>::first);
  if (logical == input.plan.placement.offsets.end() ||
      input.plan.placement.alignment != range.alignment ||
      input.addresses.activation_alignment < range.alignment)
    return {};
  std::uint64_t logical_address = 0;
  if (__builtin_add_overflow(input.addresses.activations, logical->second, &logical_address) ||
      logical_address != range.base)
    return {};
  // A logical input declared aliased with another resource cannot be moved
  // independently. The full original boundary remains the authority here.
  for (std::size_t i = 0; i < input.comparison->bindings.size(); ++i) {
    if (i == binding) continue;
    const auto& other = input.comparison->bindings[i];
    if (other.original_root == range.original_root ||
        other.allocation_class == range.allocation_class)
      return {};
  }
  const auto slot = std::ranges::find(lowered.inputs, leaf.primitive, &InputSlot::tensor);
  const auto* descriptor = PhysicalTensorOf(lowered, leaf.tensor);
  if (slot == lowered.inputs.end() || !descriptor || !descriptor->ir_format ||
      range.bytes != ggml_nbytes(leaf.tensor) || *descriptor->ir_format != current.format ||
      reinterpret_cast<std::uintptr_t>(leaf.tensor->data) < input.addresses.activations)
    return {};
  for (std::size_t d = 0; d < 4; ++d)
    if (leaf.tensor->ne[d] <= 0 ||
        static_cast<std::uint64_t>(leaf.tensor->ne[d]) != current.ne(d) ||
        leaf.tensor->nb[d] != current.nb[d])
      return {};
  const auto physical_root = std::ranges::find(lowered.physical_roots, mapped, &PhysicalRoot::root);
  if (physical_root == lowered.physical_roots.end()) return {};
  ComparisonStagedInput fact{
      .planned = mapped,
      .producer = producer,
      .activation_base = input.addresses.activations,
      .activation_bytes = input.addresses.activation_bytes,
      .logical_offset = logical->second,
      .physical_offset =
          reinterpret_cast<std::uintptr_t>(leaf.tensor->data) - input.addresses.activations,
      .bytes = physical_root->bytes,
      .input_bytes = range.bytes,
      .alignment = range.alignment,
      .allocation_class = range.allocation_class,
      .type = leaf.tensor->type,
      .segment = current.segment,
      .kind = slot->kind};
  auto checked_leaf = leaf;
  checked_leaf.comparison_staged = fact;
  auto actual = range;
  actual.base = reinterpret_cast<std::uintptr_t>(leaf.tensor->data);
  actual.bytes = fact.bytes;
  actual.allocation_bytes = fact.bytes;
  if (!CheckComparisonStagedInput(lowered, checked_leaf, actual)) return {};
  return fact;
}

bool CheckComparisonExpertSlab(const LoweredLeaf& leaf,
                               std::span<const ComparisonBindingRange> ranges) {
  if (!leaf.comparison_slab || !leaf.tensor || !leaf.tensor->data ||
      leaf.part != format::PartRole::kCodes)
    return false;
  const auto& s = *leaf.comparison_slab;
  std::uint64_t product = 0, total = 0;
  if (!s.width || s.width % 128 || !s.hidden || s.hidden % 64 || !s.experts ||
      !s.roots[0].valid() || !s.roots[1].valid() || s.roots[0] == s.roots[1] ||
      leaf.primitive != s.roots[0] || __builtin_mul_overflow(s.width, s.hidden, &product) ||
      product > INT64_MAX / 4)
    return false;
  const moe::ExpertLayout layout{.ffn = s.hidden, .width = s.width};
  const auto pitch = layout.bytes();
  if (__builtin_mul_overflow(pitch, s.experts, &total) || total > INT64_MAX) return false;
  const auto* t = leaf.tensor;
  if (t->op != GGML_OP_NONE || t->type != GGML_TYPE_I8 || t->view_src || t->view_offs ||
      !std::cmp_equal(t->ne[0], pitch) || !std::cmp_equal(t->ne[1], s.experts) || t->ne[2] != 1 ||
      t->ne[3] != 1 || t->nb[0] != 1 || t->nb[1] != pitch || t->nb[2] != total || t->nb[3] != total)
    return false;
  const std::array offsets{layout.gate_up_codes(), layout.gate_up_scales(), layout.down_codes(),
                           layout.down_scales(), pitch};
  const auto address = reinterpret_cast<std::uintptr_t>(t->data);
  std::uint64_t end = 0;
  if (__builtin_add_overflow(address, total, &end)) return false;
  std::optional<std::uint32_t> allocation;
  for (std::size_t i = 0; i < s.bindings.size(); ++i) {
    if (s.bindings[i] >= ranges.size()) return false;
    for (std::size_t j = 0; j < i; ++j)
      if (s.bindings[i] == s.bindings[j]) return false;
    const auto& r = ranges[s.bindings[i]];
    const auto role = i % 2 == 0 ? format::PartRole::kCodes : format::PartRole::kBlockScales;
    const auto span = (s.experts - 1) * pitch + offsets[i + 1] - offsets[i];
    if (r.original_root != s.roots[i / 2] || r.part != role || r.row_storage || r.base != address ||
        r.allocation_bytes != total || r.offset != offsets[i] || r.bytes != span ||
        r.alignment == 0 || !std::has_single_bit(r.alignment) ||
        (address + r.offset) % r.alignment != 0 ||
        (allocation && *allocation != r.allocation_class))
      return false;
    allocation = r.allocation_class;
  }
  return std::ranges::count(ranges, *allocation, &ComparisonBindingRange::allocation_class) == 4;
}

std::optional<ComparisonExpertSlab> DescribeComparisonExpertSlab(const LowerInput& input,
                                                                 const LoweredStep& lowered,
                                                                 const LoweredLeaf& leaf) {
  if (!input.comparison || input.comparison->scope != ComparisonProofScope::kWholeProgramNative ||
      input.addresses.measuring() || !input.addresses.expert_part || !leaf.primitive.valid() ||
      leaf.primitive.v >= input.io.graph.tensor_count() || leaf.part != format::PartRole::kCodes)
    return {};
  const auto& graph = input.io.graph;
  const auto gu =
      std::ranges::find(input.io.weights, leaf.primitive, &components::StepWeight::tensor);
  if (gu == input.io.weights.end() || !gu->ref.expert_array) return {};
  const auto& gf = graph.tensor(gu->tensor).format;
  const auto native = [](const format::TensorFormat& f) {
    return f.encoding == format::Encoding::kNvfp4 && f.layout == format::Layout::kScaleAtoms &&
           f.shape.rank == 3 && f.shape.ne[3] == 1;
  };
  std::uint64_t geometry = 0;
  if (!native(gf) || gf.stacked != 2 || !gf.shape.ne[0] || gf.shape.ne[0] % 128 ||
      !gf.shape.ne[1] || gf.shape.ne[1] % 128 || !gf.shape.ne[2] ||
      __builtin_mul_overflow(gf.shape.ne[0], gf.shape.ne[1] / 2, &geometry) ||
      geometry > INT64_MAX / 4 || graph.root(gu->tensor) != gu->tensor ||
      graph.tensor(gu->tensor).binding != ir::Binding{gu->ref})
    return {};
  const components::StepWeight* down = nullptr;
  for (const auto& w : input.io.weights) {
    if (w.tensor == gu->tensor || !w.ref.expert_array || w.ref.stage != gu->ref.stage ||
        w.ref.layer != gu->ref.layer || w.ref.component != gu->ref.component)
      continue;
    const auto& f = graph.tensor(w.tensor).format;
    if (!native(f) || f.stacked != 1 || f.shape.ne[0] != gf.shape.ne[1] / 2 ||
        f.shape.ne[1] != gf.shape.ne[0] || f.shape.ne[2] != gf.shape.ne[2])
      continue;
    if (down || graph.root(w.tensor) != w.tensor ||
        graph.tensor(w.tensor).binding != ir::Binding{w.ref})
      return {};
    down = &w;
  }
  if (!down || gu->tensor.v >= input.plan.tensors.size() ||
      down->tensor.v >= input.plan.tensors.size())
    return {};
  const auto mapped_gu = input.plan.tensors[gu->tensor.v];
  const auto mapped_down = input.plan.tensors[down->tensor.v];
  // The selected MoeExperts capability expands into registered auxiliary
  // calls. Their operation is MatMulId, while their owned descriptor producer
  // still names the exact planned MoeExperts node that selected those calls.
  const bool used = std::ranges::any_of(lowered.plan.steps, [&](const PlanStep& step) {
    if (step.operation != ir::Primitive(ir::Op::kMatMulId) || step.nodes.size() != 1 ||
        step.lane != 0 || !step.nodes[0])
      return false;
    const auto* t = step.nodes[0];
    const auto* record = PhysicalTensorOf(lowered, t);
    if (!record || !record->producer.valid() ||
        record->producer.v >= input.plan.graph.nodes().size())
      return false;
    const auto& parent = input.plan.graph.node(record->producer);
    if (parent.key != ir::Fused(ir::FusedKind::kMoeExperts) || parent.nsrc < 2 ||
        parent.src[0] != mapped_gu || parent.src[1] != mapped_down ||
        !std::ranges::any_of(
            input.plan.steps,
            [&](const planner::Selected& selected) {
              return std::ranges::contains(selected.nodes, record->producer);
            }) ||
        !HasWholeNativeCustomScalarSchema(*t))
      return false;
    if (step.implementation == "llmp.moe.gemv")
      return LlmpOpOf(t) == LlmpOp::kMoeGemv && t->src[0] == leaf.tensor &&
             CheckMoeGemv(t).has_value();
    if (step.implementation == "llmp.moe.gemm.cutlass")
      return LlmpOpOf(t) == LlmpOp::kMoeGemm && t->src[2] == leaf.tensor &&
             CheckMoeGemm(t).has_value();
    return false;
  });
  if (!used) return {};
  ComparisonExpertSlab result{.roots = {gu->tensor, down->tensor},
                              .width = gf.shape.ne[0],
                              .hidden = gf.shape.ne[1] / 2,
                              .experts = gf.shape.ne[2]};
  const std::array weights{&*gu, down};
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& w = *weights[i / 2];
    const auto role = i % 2 == 0 ? format::PartRole::kCodes : format::PartRole::kBlockScales;
    auto one = graph.tensor(w.tensor).format;
    one.shape.rank = 2;
    one.shape.ne[2] = one.shape.ne[3] = 1;
    const auto parts = format::PartsOf(one);
    if (!parts || parts->size() != 2 || w.parts.size() != 2) return {};
    const auto named = std::ranges::find(w.parts, role, [](const auto& p) { return p.second; });
    if (named == w.parts.end() ||
        std::ranges::count_if(w.parts, [&](const auto& p) { return p.second == role; }) != 1)
      return {};
    const auto binding = input.addresses.expert_part(w.ref.stage, named->first);
    const auto extent = std::ranges::find(parts->parts(), role, &format::PartExtent::role);
    if (!binding || extent == parts->parts().end() || binding->format != one ||
        binding->role != role)
      return {};
    std::size_t found = input.comparison->bindings.size();
    for (std::size_t j = 0; j < input.comparison->bindings.size(); ++j) {
      const auto& r = input.comparison->bindings[j];
      if (r.original_root != w.tensor || r.part != role) continue;
      if (found != input.comparison->bindings.size()) return {};
      found = j;
      std::uint64_t address = 0, bytes = 0;
      if (__builtin_add_overflow(r.base, r.offset, &address) || binding->address != address ||
          __builtin_mul_overflow(result.experts - 1, binding->pitch, &bytes) ||
          __builtin_add_overflow(bytes, extent->readable, &bytes) || binding->bytes != bytes ||
          r.bytes != bytes ||
          binding->pitch != moe::ExpertLayout{.ffn = result.hidden, .width = result.width}.bytes())
        return {};
    }
    if (found >= input.comparison->bindings.size() || found > UINT32_MAX) return {};
    result.bindings[i] = static_cast<std::uint32_t>(found);
  }
  auto checked = leaf;
  checked.comparison_slab = result;
  return CheckComparisonExpertSlab(checked, input.comparison->bindings) ? std::optional(result)
                                                                        : std::nullopt;
}

bool CheckComparisonTopKIndices(const LowerInput& input, const LoweredStep& lowered,
                                const execution::Registry& registry, const ggml_tensor* store) {
  if (!CheckSetRowsExt(store) || !lowered.plan.regions.empty()) return false;
  const auto* ids = store->src[1];
  if (!ids || ids->op != GGML_OP_TOP_K || ids->view_src || ids->view_offs != 0 ||
      !HasWholeNativeFillOrTopKSchema(*ids) || !CheckTopK(ids) || ids->src[0]->ne[0] > store->ne[1])
    return false;
  const auto& graph = input.plan.graph;
  const auto* output = PhysicalTensorOf(lowered, store);
  if (!output || !output->tensor.valid() || output->tensor.v >= graph.tensor_count() ||
      !output->producer.valid() || output->producer.v >= graph.nodes().size())
    return false;
  const auto& write = graph.node(output->producer);
  if (write.key != ir::Primitive(ir::Op::kCacheStore) || write.nsrc != 3 || write.nout != 1 ||
      write.out[0] != output->tensor)
    return false;
  const auto matches = [&](const ggml_tensor* actual, ir::TensorId expected) {
    if (!actual || !expected.valid() || expected.v >= graph.tensor_count()) return false;
    const auto* record = PhysicalTensorOf(lowered, actual);
    const auto& typed = graph.tensor(expected);
    const auto type = GgmlTypeOf(format::Info(typed.format.encoding).name);
    if (!record || record->tensor != expected || record->root != graph.root(expected) ||
        record->producer != graph.producer(expected) || record->part || !record->ir_format ||
        *record->ir_format != typed.format || !type || *type != actual->type)
      return false;
    for (std::size_t axis = 0; axis < 4; ++axis)
      if (!std::cmp_equal(actual->ne[axis], typed.ne(axis)) ||
          (typed.ne(axis) > 1 && actual->nb[axis] != typed.nb[axis]))
        return false;
    return true;
  };
  if (!matches(store, write.out[0]) || !matches(store->src[2], write.src[0]) ||
      !matches(store->src[0], write.src[1]) || !matches(ids, write.src[2]))
    return false;
  const auto& top = graph.node(graph.producer(write.src[2]));
  if (top.key != ir::Primitive(ir::Op::kTopK) || top.nsrc != 1 || top.nout != 1 ||
      top.out[0] != write.src[2] || !matches(ids->src[0], top.src[0]) ||
      !std::cmp_equal(top.Params<ir::TopKParams>().k, ids->ne[0]))
    return false;
  std::size_t producer = lowered.plan.steps.size(), consumer = lowered.plan.steps.size();
  for (std::size_t i = 0; i < lowered.plan.steps.size(); ++i) {
    const auto& call = lowered.plan.steps[i];
    const bool produces = std::ranges::contains(call.nodes, ids);
    const bool consumes = std::ranges::contains(call.nodes, store);
    if (!produces && !consumes) continue;
    const auto name = produces ? kTopKName : kSetRowsExtName;
    const auto operation =
        produces ? ir::Primitive(ir::Op::kTopK) : ir::Primitive(ir::Op::kCacheStore);
    const auto registered = registry.Find(name);
    if (produces == consumes || !registered || registry.at(*registered).operation != operation ||
        call.implementation != name || call.operation != operation || call.lane != 0 ||
        call.nodes.size() != 1)
      return false;
    auto& position = produces ? producer : consumer;
    if (position != lowered.plan.steps.size()) return false;
    position = i;
  }
  if (producer >= consumer || consumer == lowered.plan.steps.size()) return false;
  // A TopK result must still hold those indices when the store consumes it.
  // Conservatively treat every intervening output AND data-bearing operand
  // as a write: custom/in-place calls may write through an operand while
  // their nominal output lives elsewhere. This single-use proof needs no
  // intervening reader and grants no general per-operation access authority.
  const auto begin = reinterpret_cast<std::uintptr_t>(ids->data);
  std::uintptr_t end = 0;
  if (__builtin_add_overflow(begin, ggml_nbytes(ids), &end)) return false;
  const auto disjoint = [&](const ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return true;
    const auto other = reinterpret_cast<std::uintptr_t>(tensor->data);
    std::uintptr_t other_end = 0;
    return !__builtin_add_overflow(other, ggml_nbytes(tensor), &other_end) &&
           !(begin < other_end && other < end);
  };
  for (std::size_t i = producer + 1; i < consumer; ++i) {
    for (const auto* node : lowered.plan.steps[i].nodes) {
      if (!node || !node->data || !disjoint(node)) return false;
      for (const auto* operand : node->src)
        if (!disjoint(operand)) return false;
    }
  }
  // At the pinned no-CUB implementation, radix gather writes only col<ncols;
  // bitonic selection orders padded sentinels behind all real indices. Ties,
  // infinities and NaNs change ordering, never this integer range guarantee.
  return true;
}

std::optional<std::string_view> ComparisonHostIndexRecipe(const components::StepInput& source,
                                                          const InputSlot& slot,
                                                          const planner::StepShape& shape,
                                                          std::uint64_t destination_capacity) {
  const auto* def = source.host.def;
  std::string_view identity;
  if (def == &components::CompressorBlockCellsInput())
    identity = "components.compressor.block_cells.v1";
  else if (def == &components::CompressorKeptCellsInput())
    identity = "components.compressor.kept_cells.v1";
  else
    return {};
  if (source.kind != components::InputKind::kHostComputed ||
      slot.kind != InputKind::kHostComputed || slot.host != def || source.tensor != slot.tensor ||
      source.segment != slot.segment || source.segment >= shape.segments.size() ||
      source.host.ratio != source.ratio || source.host.overlap != source.overlap ||
      source.ratio != slot.ratio || source.overlap != slot.overlap || source.ring != slot.ring ||
      source.cells.capacity != slot.cells.capacity || source.cells.ring != slot.cells.ring ||
      source.cells.frozen != slot.cells.frozen || source.cells.fixed != slot.cells.fixed ||
      source.cells.block != slot.cells.block || source.cells.per_row != slot.cells.per_row ||
      source.cells.first_row != slot.cells.first_row ||
      source.cells.positions != slot.cells.positions || source.cells.recent != slot.cells.recent ||
      source.read != slot.read || source.sections != slot.sections || source.visible_capacity ||
      !source.what.empty() || slot.commit_save || slot.outputs || slot.visible_capacity ||
      slot.window || slot.mask_rows || slot.count || slot.ahead || slot.cells.capacity == 0 ||
      slot.cells.capacity > destination_capacity || slot.bytes == 0 ||
      slot.bytes % sizeof(std::int64_t) != 0 || !def->validate(source, shape))
    return {};
  struct Bounds {
    std::uint64_t capacity;
    std::uint64_t count;
  } bounds{slot.cells.capacity, slot.bytes / sizeof(std::int64_t)};
  components::InputValueSink values(&bounds, [](void* p, std::size_t index, std::int64_t value) {
    const auto& b = *static_cast<Bounds*>(p);
    return index < b.count && value >= 0 && static_cast<std::uint64_t>(value) < b.capacity;
  });
  if (!components::EmitInputValues(source, shape, {}, values) || values.size() != bounds.count)
    return {};
  return identity;
}

// Whole-program inputs are supplied by ordinary canonical staging before each
// invocation. Check its owned recipe against every actual indexed destination;
// arbitrary device indices and commit-save generators are not this protocol.
bool HashWholeIndexedPreparation(CanonicalHash& hash, const LowerInput& input,
                                 const LoweredStep& lowered, const execution::Registry& registry) {
  hash.Text("llmp.whole.normal-index-preparation.v3");
  std::size_t count = 0;
  for (const auto& physical : lowered.physical_tensors)
    if (physical.descriptor && physical.descriptor->op == GGML_OP_SET_ROWS) ++count;
  hash.U64(count);
  for (std::uint32_t id = 0; id < lowered.physical_tensors.size(); ++id) {
    const auto physical = std::ranges::find(lowered.physical_tensors, id, &PhysicalTensor::id);
    if (physical == lowered.physical_tensors.end()) return false;
    const auto* node = physical->descriptor;
    if (!node || node->op != GGML_OP_SET_ROWS) continue;
    if (!node->src[1] || !node->src[2] || node->ne[1] <= 0) return false;
    const auto* ids = node->src[1];
    if (ids->op == GGML_OP_TOP_K) {
      if (!CheckComparisonTopKIndices(input, lowered, registry, node)) return false;
      hash.U32(physical->id);
      hash.U8(2);  // Checked device TopK, distinct from ordinary staged indices.
      hash.Text("ggml.top_k.valid-column-indices.v1");
      hash.U64(static_cast<std::uint64_t>(ids->src[0]->ne[0]));
      hash.U64(static_cast<std::uint64_t>(node->ne[1]));
      for (const auto* actual :
           std::array<const ggml_tensor*, 5>{ids->src[0], ids, node->src[0], node->src[2], node}) {
        const auto* record = PhysicalTensorOf(lowered, actual);
        if (!record || !HashPhysicalDescriptor(hash, lowered, *record)) return false;
      }
      continue;
    }
    if (!input.token) return false;
    if (ids->type != GGML_TYPE_I64 || !ggml_is_contiguous(ids) || ids->view_src || !ids->data ||
        ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1)
      return false;
    const InputSlot* slot = nullptr;
    for (const auto& candidate : lowered.inputs) {
      if ((candidate.kind != InputKind::kStoreCells &&
           candidate.kind != InputKind::kHostComputed) ||
          candidate.device_address != reinterpret_cast<std::uintptr_t>(ids->data))
        continue;
      if (slot) return false;
      slot = &candidate;
    }
    if (!slot || slot->commit_save || slot->bytes != ggml_nbytes(ids) ||
        slot->cells.capacity == 0 || slot->cells.capacity > static_cast<std::uint64_t>(node->ne[1]))
      return false;
    const components::StepInput* source = nullptr;
    for (const auto& candidate : input.token->step.inputs) {
      if (candidate.tensor != slot->tensor) continue;
      if (source) return false;
      source = &candidate;
    }
    const auto host_recipe =
        source && source->kind == components::InputKind::kHostComputed
            ? ComparisonHostIndexRecipe(*source, *slot, input.token->shape,
                                        static_cast<std::uint64_t>(node->ne[1]))
            : std::nullopt;
    if (source && source->kind == components::InputKind::kHostComputed && !host_recipe)
      return false;
    if (!source ||
        (!host_recipe &&
         (slot->kind != InputKind::kStoreCells ||
          (source->kind != components::InputKind::kStoreCells &&
           source->kind != components::InputKind::kOutputCells) ||
          source->segment != slot->segment || source->cells.capacity != slot->cells.capacity ||
          source->cells.ring != slot->cells.ring || source->cells.frozen != slot->cells.frozen ||
          source->cells.fixed != slot->cells.fixed || source->cells.block != slot->cells.block ||
          source->cells.per_row != slot->cells.per_row ||
          source->cells.first_row != slot->cells.first_row ||
          source->cells.positions != slot->cells.positions ||
          source->cells.recent != slot->cells.recent || source->read != slot->read ||
          source->sections != slot->sections || source->host.def || source->host.ratio ||
          source->host.overlap || source->ratio || source->overlap || source->ring ||
          source->visible_capacity || !source->what.empty())))
      return false;
    if (input.token->step.graph.identity() != input.io.graph.identity() ||
        !source->tensor.valid() || source->tensor.v >= input.plan.tensors.size() ||
        std::ranges::find(input.io.inputs, source->tensor) == input.io.inputs.end())
      return false;
    const auto mapped = input.plan.tensors[source->tensor.v];
    const auto* actual = PhysicalTensorOf(lowered, ids);
    if (!mapped.valid() || !actual || actual->tensor != mapped ||
        actual->root != input.plan.graph.root(mapped) ||
        input.io.graph.root(source->tensor) != source->tensor ||
        input.io.graph.tensor(source->tensor).view_offset != 0 ||
        input.plan.graph.tensor(mapped).view_offset != 0)
      return false;
    const bool output_cells = source->kind == components::InputKind::kOutputCells;
    if ((!output_cells && slot->outputs != 0) ||
        (output_cells &&
         (source->segment >= input.token->shape.segments.size() || slot->outputs == 0 ||
          slot->outputs != input.token->shape.segments[source->segment].outputs)))
      return false;
    const auto& desc = input.io.graph.tensor(source->tensor);
    if (desc.format.encoding != format::Encoding::kI64 || !std::cmp_equal(desc.ne(0), ids->ne[0]) ||
        desc.ne(1) != 1 || desc.ne(2) != 1 || desc.ne(3) != 1)
      return false;
    struct Bounds {
      std::uint64_t capacity;
    } bounds{slot->cells.capacity};
    components::InputValueSink values(&bounds, [](void* p, std::size_t, std::int64_t value) {
      return value >= 0 && static_cast<std::uint64_t>(value) < static_cast<Bounds*>(p)->capacity;
    });
    if (!host_recipe && (!components::EmitInputValues(*source, input.token->shape, {}, values) ||
                         values.size() != static_cast<std::uint64_t>(ggml_nelements(ids))))
      return false;
    hash.U32(physical->id);
    hash.U8(host_recipe.has_value() ? 1 : 0);
    if (host_recipe) {
      hash.Text(*host_recipe);
      hash.U32(source->ratio);
      hash.U8(source->overlap);
      hash.U64(source->ring);
    }
    hash.U32(source->tensor.v);
    hash.U8(static_cast<std::uint8_t>(source->kind));
    hash.U8(source->segment);
    hash.U64(slot->bytes);
    hash.U64(source->cells.capacity);
    hash.U8(source->cells.ring);
    hash.U8(source->cells.frozen);
    hash.U8(source->cells.fixed);
    hash.U32(source->cells.block);
    hash.U8(source->cells.per_row);
    hash.U32(source->cells.first_row);
    hash.U8(source->cells.positions);
    hash.U8(source->cells.recent);
    hash.U32(source->cells.cell_pad);
    hash.U64(source->read);
    hash.U8(source->sections);
    hash.U32(slot->outputs);
    hash.U32(input.token->shape.stage);
    hash.U8(static_cast<std::uint8_t>(input.token->shape.phase));
    // Destination capacity is sealed separately for every actual consumer.
    hash.U64(static_cast<std::uint64_t>(node->ne[1]));
  }
  return true;
}

std::optional<base::Sha256Digest> ComparisonSelectedScheduleIdentity(
    const LoweredStep& lowered, const execution::Registry& registry, const SelectedSource& source,
    std::uint64_t pool_bytes) {
  const auto identity = source.ExecutionIdentity();
  if (!identity || (!lowered.comparison_origins && !lowered.comparison_selected_graph) ||
      !lowered.routing.empty() || !lowered.plan.regions.empty() ||
      std::ranges::all_of(*identity, [](std::uint8_t byte) { return byte == 0; }))
    return {};
  const auto& planned_nodes = lowered.comparison_selected_graph
                                  ? lowered.comparison_selected_graph->planned_nodes
                                  : lowered.comparison_origins->planned_nodes;
  const auto& planned_tensors = lowered.comparison_selected_graph
                                    ? lowered.comparison_selected_graph->planned_tensors
                                    : lowered.comparison_origins->planned_tensors;
  CanonicalHash hash("llmp.selected.resident-whole.v1");
  hash.Digest(*identity);
  hash.U64(source.persistent_bytes());
  hash.U64(lowered.selected.size());
  std::size_t seen = 0;
  for (const auto& step : lowered.plan.steps) {
    const auto selected =
        step.nodes.size() == 1
            ? std::ranges::find(lowered.selected, step.nodes[0], &SelectedNode::tensor)
            : lowered.selected.end();
    if (selected == lowered.selected.end()) continue;
    ++seen;
    const auto at = registry.Find(step.implementation);
    const auto* marker = selected->tensor;
    const auto* record = PhysicalTensorOf(lowered, marker);
    if (!at || selected->implementation != &registry.at(*at) ||
        registry.at(*at).operation != step.operation || !source.Owns(step.implementation) ||
        !selected->call || selected->call->owner() != &source || selected->call->route() ||
        selected->call->route_descriptor_bytes() != 0 ||
        selected->call->scratch_bytes() > pool_bytes || step.lane != 0 || !marker ||
        marker->op != GGML_OP_CUSTOM || marker->view_src || marker->view_offs != 0 ||
        !ggml_is_contiguous(marker) ||
        std::ranges::any_of(marker->op_params, [](std::int32_t word) { return word != 0; }) ||
        !record || !record->producer.valid() || record->producer.v >= planned_nodes.size() ||
        std::ranges::count(lowered.selected, marker, &SelectedNode::tensor) != 1 ||
        std::ranges::count_if(lowered.plan.steps, [&](const PlanStep& call) {
          return std::ranges::contains(call.nodes, marker);
        }) != 1)
      return {};
    const auto& node = planned_nodes[record->producer.v];
    // Selected GetRows can depend on host-validated side-row/paged slot maps,
    // which are additional preparation authority, not marker operands.
    if (node.key == ir::Primitive(ir::Op::kGetRows) || node.key != step.operation ||
        node.nout == 0 || node.nout > 2 ||
        (node.nout == 2 && node.key != ir::Fused(ir::FusedKind::kPreparedProducts)))
      return {};
    hash.Digest(registry.identity(*at));
    hash.U32(record->producer.v);
    hash.U32(record->id);
    if (!HashPhysicalDescriptor(hash, lowered, *record, true)) return {};
    hash.U64(selected->call->scratch_bytes());
    hash.U64(selected->call->host_bytes());
    hash.U64(selected->call->launches());
    hash.U8(registry.at(*at).capability && registry.at(*at).capability->capturable ? 1 : 0);
    // MakeSelected flattens physical parts in format order, never just a
    // representative logical weight. Authenticate exactly that complete list.
    std::size_t dependency = 0;
    for (const auto id : node.sources()) {
      if (!id.valid() || id.v >= planned_tensors.size()) return {};
      const auto& expected = planned_tensors[id.v];
      const auto parts = format::PartsOf(expected.format);
      if (!parts) return {};
      for (const auto& part : parts->parts()) {
        if (dependency >= GGML_MAX_SRC || !marker->src[dependency]) return {};
        const auto* actual = PhysicalTensorOf(lowered, marker->src[dependency++]);
        if (!actual || actual->tensor != id || !actual->ir_format ||
            *actual->ir_format != expected.format ||
            (parts->size() > 1 ? actual->part != part.role
                               : actual->part && actual->part != part.role))
          return {};
        hash.U32(actual->id);
        if (!HashPhysicalDescriptor(hash, lowered, *actual)) return {};
        hash.U8(static_cast<std::uint8_t>(part.role));
        hash.U64(part.readable);
      }
    }
    for (; dependency < GGML_MAX_SRC; ++dependency)
      if (marker->src[dependency]) return {};
    std::uint64_t output_bytes = 0;
    for (std::size_t i = 0; i < node.nout; ++i) {
      const auto id = node.out[i];
      if (!id.valid() || id.v >= planned_tensors.size()) return {};
      const auto& expected = planned_tensors[id.v];
      const PhysicalTensor* output = nullptr;
      for (const auto& candidate : lowered.physical_tensors) {
        if (candidate.tensor != id || candidate.producer != record->producer) continue;
        if (output) return {};
        output = &candidate;
      }
      if (!output || !output->descriptor || !output->ir_format ||
          *output->ir_format != expected.format || !ir::IsContiguous(expected))
        return {};
      const auto* actual = output->descriptor;
      for (std::size_t d = 0; d < 4; ++d)
        if (actual->ne[d] < 1 || static_cast<std::uint64_t>(actual->ne[d]) != expected.ne(d) ||
            (expected.ne(d) > 1 && actual->nb[d] != expected.nb[d]))
          return {};
      if (actual->type != marker->type || !ggml_is_contiguous(actual) ||
          (node.nout == 1 ? actual != marker
                          : actual->op != GGML_OP_VIEW || actual->src[0] != marker ||
                                actual->view_src != marker || actual->view_offs != output_bytes))
        return {};
      if (__builtin_add_overflow(output_bytes, static_cast<std::uint64_t>(ggml_nbytes(actual)),
                                 &output_bytes))
        return {};
      hash.U32(output->id);
      if (actual != marker && !HashPhysicalDescriptor(hash, lowered, *output)) return {};
    }
    if (output_bytes != ggml_nbytes(marker)) return {};
  }
  if (seen != lowered.selected.size()) return {};
  return hash.Finish();
}

ComparisonRecipeSeal DescribeComparison(
    const LowerInput& input, const LoweredStep& lowered, const execution::Registry& registry,
    const execution::BackendDeclaration* declaration, ComparisonMetrics metrics,
    std::optional<components::ComparisonBoundaryDigest> observed_boundary,
    const LaunchContext* launch_context, ComparisonMulMatSelector mul_mat_selector,
    const ComparisonQ8Dispatch* q8_dispatch, ComparisonRemainderCallFactsFn describe_call_facts,
    ComparisonNativeCallChecker check_native_call, const SelectedSource* selected_source) {
  ComparisonRecipeSeal seal;
  if (!input.comparison) return seal;
  seal.mode = input.comparison->mode;
  seal.scope = input.comparison->scope;
  const bool native_whole = seal.scope == ComparisonProofScope::kWholeProgramNative;
  if (native_whole && std::ranges::any_of(lowered.selected, [&](const SelectedNode& selected) {
        const auto at =
            selected.implementation ? registry.Find(selected.implementation->name) : std::nullopt;
        return at && selected.implementation == &registry.at(*at) && registry.at(*at).capability &&
               !registry.at(*at).capability->capturable;
      }))
    seal.execution_method = ComparisonExecutionMethod::kOrdinaryOnly;
  seal.subject = input.io.graph.Digest();
  if (observed_boundary)
    seal.boundary = observed_boundary->combined;
  else
    seal.missing = ComparisonMissing::kBoundary;
  seal.planned_graph = input.plan.graph.Digest();
  seal.policy = input.plan.policy_digest;
  seal.registry = registry.SelectionDigest();
  seal.backend_recipe = declaration ? declaration->recipe : base::Sha256Digest{};
  seal.backend_build = declaration ? declaration->build : base::Sha256Digest{};
  seal.backend_selection = declaration ? declaration->comparison_selection : base::Sha256Digest{};
  seal.activation_bytes = lowered.extent;
  seal.pool_bytes = metrics.pool_bytes;
  seal.private_pool_bytes = metrics.private_pool_bytes;
  seal.lanes = metrics.lanes;
  seal.lane_scratch_bytes = metrics.lane_scratch_bytes;

  CanonicalHash physical(native_whole ? "llmp.comparison.whole-native.physical.v2"
                                      : "llmp.comparison.physical.v2");
  physical.U8(static_cast<std::uint8_t>(seal.mode));
  if (native_whole) physical.U8(static_cast<std::uint8_t>(seal.execution_method));
  physical.Digest(seal.subject);
  physical.Digest(seal.planned_graph);
  physical.U64(lowered.comparison_omitted_inputs.size());
  for (const auto& omitted : lowered.comparison_omitted_inputs) {
    const auto& source = input.io.graph;
    const auto& planned = input.plan.graph;
    if (!native_whole || !CheckComparisonOmittedInput(lowered, omitted.original) ||
        omitted.original.v >= source.tensor_count() ||
        omitted.original.v >= input.plan.tensors.size() ||
        input.plan.tensors[omitted.original.v] != omitted.planned ||
        omitted.planned.v >= planned.tensor_count() ||
        !std::ranges::contains(source.inputs(), omitted.original) ||
        !std::ranges::contains(planned.inputs(), omitted.planned) ||
        source.root(omitted.original) != omitted.original ||
        planned.root(omitted.planned) != omitted.planned ||
        source.tensor(omitted.original).role != ir::Role::kInput ||
        planned.tensor(omitted.planned).role != ir::Role::kInput ||
        source.node(source.producer(omitted.original)).key != ir::Primitive(ir::Op::kInput) ||
        planned.producer(omitted.planned) != omitted.producer ||
        planned.node(omitted.producer).key != ir::Primitive(ir::Op::kInput))
      seal.missing = ComparisonMissing::kPhysicalClosure;
    physical.U32(omitted.original.v);
    physical.U32(omitted.planned.v);
    physical.U32(omitted.producer.v);
  }

  const auto staged_count = std::ranges::count_if(
      lowered.leaves, [](const LoweredLeaf& leaf) { return leaf.comparison_staged.has_value(); });
  physical.U64(static_cast<std::uint64_t>(staged_count));
  for (const auto& leaf : lowered.leaves) {
    if (!leaf.comparison_staged) continue;
    const auto& fact = *leaf.comparison_staged;
    const auto actual = std::ranges::find(lowered.comparison_bindings, leaf.primitive,
                                          &ComparisonBindingRange::original_root);
    if (!native_whole || actual == lowered.comparison_bindings.end() ||
        !CheckComparisonStagedInput(lowered, leaf, *actual))
      seal.missing = ComparisonMissing::kPhysicalClosure;
    physical.Text("ggml.staged-input.physical-placement.v2");
    physical.U32(leaf.primitive.v);
    physical.U32(fact.planned.v);
    physical.U32(fact.producer.v);
    physical.U64(fact.logical_offset);
    physical.U64(fact.physical_offset);
    physical.U64(fact.bytes);
    physical.U64(fact.input_bytes);
    physical.U64(fact.alignment);
    physical.U32(static_cast<std::uint32_t>(fact.type));
    physical.U8(fact.segment);
    physical.U8(static_cast<std::uint8_t>(fact.kind));
  }

  physical.Digest(seal.policy);
  physical.Digest(seal.registry);
  physical.Digest(seal.backend_recipe);
  physical.Digest(seal.backend_build);
  physical.Digest(seal.backend_selection);
  physical.U64(input.addresses.activation_bytes);
  physical.U64(input.addresses.activation_alignment);
  physical.U64(seal.activation_bytes);
  physical.U64(seal.pool_bytes);
  physical.U64(seal.private_pool_bytes);
  physical.U64(seal.lanes);
  physical.U64(seal.lane_scratch_bytes);
  HashPhysicalBindings(physical, input);
  if (native_whole && !HashWholeIndexedPreparation(physical, input, lowered, registry))
    seal.missing = ComparisonMissing::kPhysicalClosure;
  if (lowered.generated_store) {
    const auto& generated = *lowered.generated_store;
    // This is a fixed, audited value recipe, never a hash of addresses or one
    // current payload. The owning launch enforces the H2D copy before every
    // eager execution and captures the same copy in replayable graphs.
    physical.Text("llmp.store_cells.generated_input.v1");
    physical.Text("components.EmitInputValues/StoreCells");
    physical.Text("providers.CopyAsync/HostToDevice/before-consumer");
    physical.U32(generated.slot.tensor.v);
    physical.U64(generated.slot.bytes);
    physical.U64(generated.slot.cells.capacity);
    physical.U32(generated.segment.rows);
    physical.U32(generated.stage);
    physical.U8(static_cast<std::uint8_t>(generated.phase));
    physical.U8(generated.segment.slot);
    physical.U32(generated.segment.outputs);
    physical.U8(generated.row_invariant ? 1 : 0);
    physical.U32(generated.read_align);
  }
  physical.Digest(input.plan.owner_digest);
  physical.U8(input.plan.owner_row_invariant ? 1 : 0);
  physical.U8(input.plan.owner_rows_policy_enabled ? 1 : 0);
  physical.U8(input.plan.products.compact_experts ? 1 : 0);
  physical.U32(input.plan.products.compact_expert_min_rows);
  physical.U8(input.plan.products.d2r_experts ? 1 : 0);
  physical.U8(input.plan.products.dense_pair ? 1 : 0);
  physical.U8(input.plan.products.pair_experts ? 1 : 0);
  physical.U8(input.plan.products.pair_glu ? 1 : 0);
  physical.U8(input.plan.products.pair_glu_q8 ? 1 : 0);
  physical.U8(input.plan.products.vector_floats ? 1 : 0);
  physical.U64(input.plan.owner_products.size());
  for (const auto& owner : input.plan.owner_products) {
    physical.U32(owner.primitive.v);
    physical.U32(owner.result.v);
    physical.U8(owner.slot);
    physical.U8(owner.segment);
    physical.U64(owner.columns);
  }
  physical.U64(input.plan.applied.size());
  for (const auto& applied : input.plan.applied) {
    physical.Text(applied.pattern);
    physical.U64(applied.replaced.size());
    for (const auto node : applied.replaced) physical.U32(node.v);
    physical.U64(applied.made.size());
    for (const auto node : applied.made) physical.U32(node.v);
    physical.U8(applied.numerical_change ? 1 : 0);
    if (applied.numerical_change) physical.U8(static_cast<std::uint8_t>(*applied.numerical_change));
    physical.Digest(applied.numerical_evidence);
    physical.Digest(applied.numerical_source);
    physical.Digest(applied.numerical_semantic);
    physical.U8(applied.numerical_candidate ? 1 : 0);
    physical.U8(applied.numerical_incumbent ? 1 : 0);
    physical.U64(applied.numerical_proofs.size());
    for (const auto& proof : applied.numerical_proofs) {
      physical.U32(proof.site.v);
      physical.U8(proof.weight_operand);
      physical.U8(static_cast<std::uint8_t>(proof.qualification.change));
      physical.Digest(proof.qualification.evidence);
      physical.Digest(proof.qualification.source);
      physical.Digest(proof.qualification.semantic);
      physical.U16(proof.qualification.stage);
      physical.U16(proof.qualification.component);
      physical.U8(static_cast<std::uint8_t>(proof.qualification.anchor.op));
      physical.U8(static_cast<std::uint8_t>(proof.qualification.anchor.fused));
      physical.U32(proof.qualification.arch);
      physical.U64(proof.qualification.min_rows);
      physical.U64(proof.qualification.max_rows);
      physical.U64(proof.qualification.max_k);
      physical.U64(proof.qualification.max_n);
      physical.U8(proof.qualification.exact_shape ? 1 : 0);
      physical.U8(proof.qualification.exact_dimensions ? 1 : 0);
      physical.U8(proof.qualification.incumbent_recipe ? 1 : 0);
      physical.U16(proof.qualification.weight_role);
      physical.U64(proof.actual.rows);
      physical.U64(proof.actual.k);
      physical.U64(proof.actual.n);
      physical.U8(proof.candidate ? 1 : 0);
      physical.U8(proof.grouped_projection ? 1 : 0);
      physical.U16(proof.observed.active_stage);
      HashTensorFormat(physical, proof.observed.operand_format);
      HashTensorFormat(physical, proof.observed.leaf_format);
      HashTensorFormat(physical, proof.observed.input_format);
      for (const auto stride : proof.observed.operand_strides) physical.U64(stride);
      for (const auto stride : proof.observed.leaf_strides) physical.U64(stride);
      for (const auto stride : proof.observed.input_strides) physical.U64(stride);
      physical.U8(proof.observed.weight.has_value() ? 1 : 0);
      if (proof.observed.weight) {
        physical.U16(proof.observed.weight->stage);
        physical.U16(proof.observed.weight->component);
        physical.U16(proof.observed.weight->role);
        physical.U32(proof.observed.weight->layer);
        physical.U8(proof.observed.weight->expert_array ? 1 : 0);
      }
      physical.Digest(proof.observed.bound_model);
      physical.Digest(proof.observed.model_spec);
    }
  }
  physical.U64(lowered.plan.steps.size());
  bool selected_verified = lowered.selected.empty();
  if (!lowered.selected.empty()) {
    const auto selected = native_whole && selected_source
                              ? ComparisonSelectedScheduleIdentity(
                                    lowered, registry, *selected_source, metrics.pool_bytes)
                              : std::nullopt;
    if (!selected)
      seal.missing = ComparisonMissing::kSelectedCall;
    else {
      selected_verified = true;
      physical.Digest(*selected);
    }
  }
  if (!lowered.routing.empty()) seal.missing = ComparisonMissing::kPhysicalClosure;
  for (const auto& step : lowered.plan.steps) {
    const bool selected_call =
        selected_verified && !lowered.selected.empty() && step.nodes.size() == 1 &&
        std::ranges::contains(lowered.selected, step.nodes[0], &SelectedNode::tensor);
    if (native_whole && !selected_call) {
      ComparisonRemainderCallFacts facts;
      if (!check_native_call || !check_native_call(registry, step) || !launch_context ||
          !describe_call_facts || !describe_call_facts(*launch_context, step, facts)) {
        seal.missing = ComparisonMissing::kPhysicalClosure;
      } else {
        physical.Text("llmp.native-checked-call.v2");
        physical.U64(facts.scratch_bytes);
        physical.U8(facts.float_selection.has_value() ? 1 : 0);
        if (facts.float_selection) physical.U8(static_cast<std::uint8_t>(*facts.float_selection));
        physical.U8(facts.quantized_selection.has_value() ? 1 : 0);
        if (facts.quantized_selection)
          physical.U8(static_cast<std::uint8_t>(*facts.quantized_selection));
        physical.U8(facts.has_attention_plan ? 1 : 0);
        if (facts.has_attention_plan) {
          physical.U32(static_cast<std::uint32_t>(facts.attention_columns_per_block));
          physical.U32(static_cast<std::uint32_t>(facts.attention_parallel_blocks));
          physical.U64(facts.attention_scratch_bytes);
          physical.U8(facts.attention_mask_prepass ? 1 : 0);
        }
        // The existing selector callback authenticates these direct products.
        // Other dynamic product families need their own existing sealed facts.
        const bool vector_call = step.implementation == kMulMatVector ||
                                 step.implementation == kMulMatGluFused ||
                                 step.implementation == kMulMatAddFused;
        const bool quantized_call = step.implementation == kMulMatVecQ ||
                                    step.implementation == kMmvqPreparedName ||
                                    step.implementation == "ggml.mul_mat_id.mmvq" ||
                                    step.implementation == "ggml.mul_mat_glu.mmvq_fused" ||
                                    step.implementation == "ggml.mul_mat_geglu.mmvq_fused";
        const bool quantized_tile = step.implementation == "ggml.mul_mat.mmq";
        if ((quantized_call && facts.quantized_selection != QuantMulMatPath::kVector) ||
            (quantized_tile && facts.quantized_selection != QuantMulMatPath::kTile))
          seal.missing = ComparisonMissing::kPhysicalClosure;
        const bool matrix_call = step.implementation == "ggml.mul_mat.mmf";
        if (matrix_call && !CheckComparisonMmf(step, facts))
          seal.missing = ComparisonMissing::kPhysicalClosure;
        const bool compiled_product =
            (step.implementation == kMxfp8MulMatVecName || step.implementation == kMxfp8GemmName ||
             step.implementation == "ggml.mul_mat.fwht" || step.implementation == "llmp.vecq" ||
             step.implementation == "llmp.mul_mat.mmvf_rows");
        if ((step.implementation == "llmp.vecq" ||
             step.implementation == "llmp.mul_mat.mmvf_rows") &&
            facts.scratch_bytes != 0)
          seal.missing = ComparisonMissing::kPhysicalClosure;
        const bool bf16_product = ComparisonBf16Call(step);
        if (bf16_product) {
          if (!CheckComparisonBf16(step, facts) ||
              (*facts.bf16_product == ComparisonBf16Product::kGemmEx &&
               !ComparisonLibraryWorkspaceDisjoint(lowered, facts)))
            seal.missing = ComparisonMissing::kPhysicalClosure;
          else
            HashComparisonBf16Facts(physical, facts);
        }
        if ((vector_call &&
             (!facts.has_mul_mat_selection || !facts.mul_mat_vector || facts.scratch_bytes != 0)) ||
            (step.operation == ir::Primitive(ir::Op::kMatMul) &&
             step.implementation != kMulMatCublas && step.implementation != kMulMatVector &&
             !compiled_product && !bf16_product && !quantized_tile && !matrix_call &&
             !((step.implementation == kMulMatVecQ || step.implementation == kMmvqPreparedName) &&
               facts.quantized_selection == QuantMulMatPath::kVector)))
          seal.missing = ComparisonMissing::kPhysicalClosure;
      }
    }
    if (step.implementation == kMulMatCublas) {
      ComparisonRemainderCallFacts facts;
      if (!launch_context || !describe_call_facts ||
          !describe_call_facts(*launch_context, step, facts) ||
          (!CheckComparisonSgemm(step, facts) &&
           !(native_whole && CheckComparisonWholeCublas(step, facts) &&
             facts.scratch_bytes <= metrics.pool_bytes)) ||
          !ComparisonLibraryWorkspaceDisjoint(lowered, facts))
        seal.missing = ComparisonMissing::kPhysicalClosure;
      else
        HashComparisonCublasFacts(physical, facts);
    }
    const auto explicit_access = DescribeComparisonAccess(step, ComparisonAccessScope::kWhole);
    if (explicit_access && explicit_access->dispatch) {
      physical.Text("llmp.ggml.explicit-dispatch.v1");
      physical.Text(step.implementation);
      physical.U32(*explicit_access->dispatch);
      physical.U64(explicit_access->count);
      for (const auto& access : std::span(explicit_access->values).first(explicit_access->count)) {
        const auto* descriptor = PhysicalTensorOf(lowered, access.tensor);
        if (!descriptor) {
          seal.missing = ComparisonMissing::kPhysicalClosure;
          continue;
        }
        physical.U32(descriptor->id);
        physical.U64(access.bytes);
        physical.U64(access.alignment);
        physical.U8(access.phase);
        physical.U8(access.read ? 1 : 0);
        physical.U8(access.written ? 1 : 0);
      }
    }
    physical.U8(static_cast<std::uint8_t>(step.operation.op));
    physical.U8(static_cast<std::uint8_t>(step.operation.fused));
    physical.Text(step.implementation);
    if (const auto index = registry.Find(step.implementation))
      physical.Digest(registry.identity(*index));
    else
      seal.missing = ComparisonMissing::kSourceIdentity;
    physical.U8(step.lane);
    physical.U64(step.nodes.size());
    seal.scheduled_calls += step.nodes.size();
    for (const auto* node : step.nodes) {
      const auto* descriptor = PhysicalTensorOf(lowered, node);
      if (!descriptor) {
        seal.missing = ComparisonMissing::kPhysicalClosure;
        continue;
      }
      physical.U32(descriptor->id);
      const bool native_q8_vector =
          native_whole &&
          (step.implementation == kMulMatVecQ || step.implementation == kMmvqPreparedName) &&
          node->src[0] && node->src[0]->type == GGML_TYPE_Q8_0;
      if (native_q8_vector) continue;
      if (step.implementation == kQuantizeQ8Name) {
        auto quantizer = q8_dispatch && q8_dispatch->complete()
                             ? q8_dispatch->describe_quantize_q8(node)
                             : std::unexpected(KernelFailure{});
        if (!quantizer)
          seal.missing = ComparisonMissing::kPhysicalClosure;
        else
          HashComparisonQ8QuantizeDispatch(physical, *quantizer);
      } else if (step.implementation == kMmvqPreparedName) {
        if (!launch_context) {
          seal.missing = ComparisonMissing::kPhysicalClosure;
        } else {
          auto dispatch = q8_dispatch && q8_dispatch->complete()
                              ? q8_dispatch->describe_mmvq(*launch_context, node, true)
                              : std::unexpected(KernelFailure{});
          if (!dispatch)
            seal.missing = ComparisonMissing::kPhysicalClosure;
          else
            HashComparisonMmvqDispatch(physical, *dispatch);
        }
      } else if (step.implementation == kMulMatVecQ) {
        if (!launch_context) {
          seal.missing = ComparisonMissing::kPhysicalClosure;
        } else {
          auto dispatch = q8_dispatch && q8_dispatch->complete()
                              ? q8_dispatch->describe_mmvq(*launch_context, node, false)
                              : std::unexpected(KernelFailure{});
          auto quantizer = q8_dispatch && q8_dispatch->complete()
                               ? q8_dispatch->describe_quantize_q8(node)
                               : std::unexpected(KernelFailure{});
          auto scratch = q8_dispatch && q8_dispatch->complete()
                             ? q8_dispatch->plan_mmvq(*launch_context, node)
                             : std::unexpected(KernelFailure{});
          if (!dispatch || !quantizer || !scratch) {
            seal.missing = ComparisonMissing::kPhysicalClosure;
          } else {
            HashComparisonMmvqDispatch(physical, *dispatch);
            HashComparisonQ8QuantizeDispatch(physical, *quantizer);
            physical.U64(*scratch);
          }
        }
      }
    }
    if (step.implementation == kSetRowsQuantName || step.implementation == kSetRowsQuantGroupName) {
      std::array<const ggml_tensor*, kSetRowsQuantGroupMax> nodes{};
      if (step.nodes.empty() || step.nodes.size() > nodes.size()) {
        seal.missing = ComparisonMissing::kPhysicalClosure;
      } else {
        for (std::size_t i = 0; i < step.nodes.size(); ++i) nodes[i] = step.nodes[i];
        auto recipe = DescribeSetRowsQuantLaunch(
            std::span<const ggml_tensor* const>(nodes).first(step.nodes.size()));
        if (!recipe)
          seal.missing = ComparisonMissing::kPhysicalClosure;
        else
          HashQuantStoreLaunch(physical, *recipe);
      }
    } else if (step.implementation == kRopeSetRowsQuantName) {
      if (step.nodes.size() != 2) {
        seal.missing = ComparisonMissing::kPhysicalClosure;
      } else {
        auto recipe = DescribeRopeSetRowsQuantLaunch(step.nodes[0], step.nodes[1]);
        if (!recipe)
          seal.missing = ComparisonMissing::kPhysicalClosure;
        else
          HashRopeQuantStoreLaunch(physical, *recipe);
      }
    }
  }
  physical.U64(lowered.physical_tensors.size());
  for (std::uint32_t id = 0; id < lowered.physical_tensors.size(); ++id) {
    const auto descriptor = std::ranges::find(lowered.physical_tensors, id, &PhysicalTensor::id);
    if (descriptor == lowered.physical_tensors.end()) {
      seal.missing = ComparisonMissing::kPhysicalClosure;
      continue;
    }
    const bool selected_marker =
        native_whole && selected_verified &&
        std::ranges::contains(lowered.selected, descriptor->descriptor, &SelectedNode::tensor);
    if (!HashPhysicalDescriptor(physical, lowered, *descriptor, selected_marker))
      seal.missing = ComparisonMissing::kPhysicalClosure;
    if (!selected_marker && !(native_whole ? HasWholeNativeSchema(*descriptor->descriptor)
                                           : HasSchema(*descriptor->descriptor)))
      seal.missing = ComparisonMissing::kOperationSchema;
  }
  physical.U64(lowered.plan.regions.size());
  for (const auto& region : lowered.plan.regions) {
    physical.U32(region.first);
    physical.U32(region.last);
  }
  physical.U64(lowered.prologue.size());
  for (const auto step : lowered.prologue) physical.U64(step);
  physical.U64(lowered.physical_roots.size());
  for (const auto& root : lowered.physical_roots) {
    physical.U32(root.root.v);
    physical.U32(root.producer.v);
    physical.U8(static_cast<std::uint8_t>(root.operation.op));
    physical.U8(static_cast<std::uint8_t>(root.operation.fused));
    physical.U16(root.component);
    physical.U32(root.backend_tensor);
    physical.U64(root.offset);
    physical.U64(root.bytes);
    for (const auto ne : root.ne) physical.U64(ne);
    for (const auto nb : root.nb) physical.U64(nb);
    physical.U8(static_cast<std::uint8_t>(root.encoding));
  }
  const auto hash_access_tensor = [&](const ggml_tensor* tensor) {
    const auto* descriptor = PhysicalTensorOf(lowered, tensor);
    if (!descriptor || !tensor) {
      seal.missing = ComparisonMissing::kPhysicalClosure;
      return;
    }
    physical.U32(descriptor->id);
    physical.U64(static_cast<std::uint64_t>(ggml_nbytes(tensor)));
  };
  for (const auto& step : lowered.plan.steps) {
    if (step.implementation == kDsv4HcPostNormF16Name) {
      physical.Text("llmp.ggml.dsv4.hc_post_norm_f16.access.v1");
      if (step.nodes.size() != 2) {
        seal.missing = ComparisonMissing::kPhysicalClosure;
        continue;
      }
      const auto* post = step.nodes[0];
      const auto* norm = step.nodes[1];
      physical.U32(1024);                                     // block x
      physical.U32(1);                                        // block y
      physical.U32(1);                                        // block z
      physical.U64(static_cast<std::uint64_t>(post->ne[2]));  // grid x: tokens
      physical.U32(128);                                      // dynamic shared bytes
      physical.U64(0);                                        // launch pool bytes
      physical.U64(4);                                        // complete HC source regions
      for (std::size_t i = 0; i < 4; ++i) hash_access_tensor(post->src[i]);
      physical.U64(2);  // both post and normalized outputs
      hash_access_tensor(post);
      hash_access_tensor(norm);
    } else if (step.implementation == kHcPostName) {
      physical.Text("llmp.ggml.dsv4.hc_post.access.v1");
      if (step.nodes.size() != 1 || !step.nodes[0]) {
        seal.missing = ComparisonMissing::kPhysicalClosure;
        continue;
      }
      const auto* post = step.nodes[0];
      std::uint64_t elements = 0;
      std::uint64_t blocks = 0;
      if (post->ne[0] <= 0 || post->ne[1] <= 0 || post->ne[2] <= 0 || !post->src[1] ||
          !post->src[2] || !post->src[3] ||
          __builtin_mul_overflow(static_cast<std::uint64_t>(post->ne[0]),
                                 static_cast<std::uint64_t>(post->src[1]->ne[1]), &elements) ||
          __builtin_mul_overflow(elements, static_cast<std::uint64_t>(post->ne[2]), &elements) ||
          __builtin_add_overflow(elements, std::uint64_t{255}, &blocks)) {
        seal.missing = ComparisonMissing::kPhysicalClosure;
        continue;
      }
      blocks /= 256;
      physical.U32(256);  // block x
      physical.U32(1);
      physical.U32(1);
      physical.U64(blocks);  // grid x: ceil(width * streams * tokens / 256)
      physical.U32(0);       // dynamic shared bytes
      physical.U64(0);       // launch pool bytes
      physical.U64(4);       // complete HC source regions
      for (std::size_t i = 0; i < 4; ++i) hash_access_tensor(post->src[i]);
      physical.U64(1);
      hash_access_tensor(post);
    }
  }
  seal.physical_recipe = physical.Finish();

  if (!input.token && !input.comparison->state.empty()) {
    seal.missing = ComparisonMissing::kBoundary;
  } else if (!declaration) {
    seal.missing = ComparisonMissing::kSourceIdentity;
  } else if (std::ranges::all_of(declaration->comparison_selection,
                                 [](std::uint8_t byte) { return byte == 0; })) {
    seal.missing = ComparisonMissing::kSourceIdentity;
  } else if (observed_boundary && seal.missing == ComparisonMissing::kBoundary &&
             ValidateComparisonBindings(input, input.comparison->bindings)) {
    seal.missing = ComparisonMissing::kNone;
    if (!lowered.plan.regions.empty()) {
      for (std::size_t i = 0; i < input.comparison->bindings.size(); ++i) {
        const auto& a = input.comparison->bindings[i];
        for (std::size_t j = 0; j < i; ++j) {
          const auto& b = input.comparison->bindings[j];
          std::uint64_t a_begin = a.offset, b_begin = b.offset, a_end = 0, b_end = 0;
          if (__builtin_add_overflow(a_begin, a.bytes, &a_end) ||
              __builtin_add_overflow(b_begin, b.bytes, &b_end)) {
            seal.missing = ComparisonMissing::kBindings;
            break;
          }
          if (!input.addresses.measuring() && (__builtin_add_overflow(a.base, a.offset, &a_begin) ||
                                               __builtin_add_overflow(b.base, b.offset, &b_begin) ||
                                               __builtin_add_overflow(a_begin, a.bytes, &a_end) ||
                                               __builtin_add_overflow(b_begin, b.bytes, &b_end))) {
            seal.missing = ComparisonMissing::kBindings;
            break;
          }
          const bool declared_alias = a.allocation_class == b.allocation_class;
          const bool overlaps =
              (input.addresses.measuring() ? declared_alias : a_begin < b_end && b_begin < a_end) &&
              a_begin < b_end && b_begin < a_end;
          if (a.original_root != b.original_root && overlaps) {
            seal.missing = ComparisonMissing::kBindings;
            break;
          }
        }
        if (seal.missing != ComparisonMissing::kNone) break;
      }
    }
  } else if (observed_boundary && seal.missing == ComparisonMissing::kBoundary) {
    seal.missing = ComparisonMissing::kBindings;
  }
  // Do not infer access closure from an operation schema or a successful
  // lower. All direct operand descriptors, format-readable tails, and placed
  // write extents must be represented before a seal may be promoted.
  if (!native_whole && seal.missing == ComparisonMissing::kNone &&
      !HasDirectAccessClosure(lowered, registry, input, launch_context, mul_mat_selector,
                              q8_dispatch, describe_call_facts)) {
    seal.missing = ComparisonMissing::kPhysicalClosure;
  }
  const bool has_hc_post_recipe = std::ranges::any_of(lowered.plan.steps, [](const auto& step) {
    return step.implementation == kDsv4HcPostNormF16Name || step.implementation == kHcPostName;
  });
  if (!native_whole && seal.missing == ComparisonMissing::kNone && has_hc_post_recipe &&
      (metrics.pool_bytes != 0 || metrics.private_pool_bytes != 0 || metrics.lanes != 0 ||
       metrics.lane_scratch_bytes != 0 || !lowered.plan.regions.empty() ||
       !lowered.prologue.empty()))
    seal.missing = ComparisonMissing::kPhysicalClosure;
  return seal;
}

std::expected<void, std::string> ValidateComparisonBindings(
    const LowerInput& input, std::span<const ComparisonBindingRange> bindings) {
  const auto& graph = input.io.graph;
  const auto valid_tensor = [&](ir::TensorId id) {
    return id.valid() && id.v < graph.tensor_count();
  };
  if (input.plan.tensors.size() != graph.tensor_count())
    return std::unexpected("comparison original-to-planned map has the wrong size");
  const auto* expected_boundary = input.comparison ? input.comparison->expected_boundary : nullptr;
  if (expected_boundary && (expected_boundary->source_identity() != graph.identity() ||
                            expected_boundary->source_digest() != graph.Digest()))
    return std::unexpected("comparison boundary belongs to another source graph");
  const auto mapped = [&](ir::TensorId id) {
    return valid_tensor(id) ? input.plan.tensors[id.v] : ir::TensorId{};
  };
  const auto state =
      input.comparison ? input.comparison->state : std::span<const components::StepState>{};
  std::size_t boundary_count = input.io.inputs.size();
  if (__builtin_add_overflow(boundary_count, input.io.outputs.size(), &boundary_count) ||
      __builtin_add_overflow(boundary_count, graph.keep().size(), &boundary_count) ||
      __builtin_add_overflow(boundary_count, input.io.weights.size(), &boundary_count) ||
      __builtin_add_overflow(boundary_count, state.size(), &boundary_count))
    return std::unexpected("comparison boundary tensor count overflows");
  const auto boundary_at = [&](std::size_t index) {
    if (index < input.io.inputs.size()) return input.io.inputs[index];
    index -= input.io.inputs.size();
    if (index < input.io.outputs.size()) return input.io.outputs[index];
    index -= input.io.outputs.size();
    if (index < graph.keep().size()) return graph.keep()[index];
    index -= graph.keep().size();
    if (index < input.io.weights.size()) return input.io.weights[index].tensor;
    index -= input.io.weights.size();
    return state[index].tensor;
  };
  bool shared_mapping_verified = false;
  for (std::size_t i = 0; i < boundary_count; ++i) {
    const auto original_id = boundary_at(i);
    if (!valid_tensor(original_id))
      return std::unexpected("comparison boundary has an invalid tensor ID");
    const auto planned_id = mapped(original_id);
    if (!planned_id.valid() || planned_id.v >= input.plan.graph.tensor_count())
      return std::unexpected("comparison boundary tensor is absent from the planned map");
    const auto& original = graph.tensor(original_id);
    const auto& planned = input.plan.graph.tensor(planned_id);
    if (original.role != planned.role || original.format != planned.format ||
        original.nb != planned.nb || original.segment != planned.segment ||
        original.binding != planned.binding || original.view_offset != planned.view_offset)
      return std::unexpected("comparison boundary tensor changed its typed port contract");
    const auto original_root = graph.root(original_id);
    const auto planned_root = input.plan.graph.root(planned_id);
    const auto mapped_root = mapped(original_root);
    if (!mapped_root.valid()) {
      // The shared owned-boundary validator alone authenticates a removed
      // private activation root rematerialized as a complete output value.
      // Unowned requests retain the strict original-root requirement.
      if (!expected_boundary)
        return std::unexpected("comparison boundary tensor changed its root/view alias contract");
      if (!shared_mapping_verified) {
        if (!std::ranges::equal(input.io.inputs, graph.inputs()) ||
            !std::ranges::equal(input.io.outputs, graph.outputs()))
          return std::unexpected(
              "comparison rematerialized outputs need the graph's exact ordered interface");
        const auto checked = components::ValidateMappedComparisonBoundary(
            *expected_boundary, input.plan.graph, input.plan.tensors);
        if (!checked) return std::unexpected(checked.error().detail);
        shared_mapping_verified = true;
      }
    } else if (mapped_root.v >= input.plan.graph.tensor_count() ||
               input.plan.graph.root(mapped_root) != planned_root) {
      return std::unexpected("comparison boundary tensor changed its root/view alias contract");
    }
  }
  for (std::size_t i = 0; i < boundary_count; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      const auto first = boundary_at(i), second = boundary_at(j);
      const bool original_alias = graph.root(first) == graph.root(second);
      const bool planned_alias =
          input.plan.graph.root(mapped(first)) == input.plan.graph.root(mapped(second));
      if (original_alias != planned_alias)
        return std::unexpected("comparison planned map changed boundary root aliasing");
    }
  }
  const auto covers = [&](ir::TensorId root, std::optional<format::PartRole> part) {
    return std::ranges::any_of(bindings, [&](const ComparisonBindingRange& candidate) {
      return candidate.original_root == root && candidate.part == part;
    });
  };
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const auto& range = bindings[i];
    if (!range.original_root.valid() || range.original_root.v >= graph.tensor_count())
      return std::unexpected("comparison binding names an invalid original tensor");
    if (graph.root(range.original_root) != range.original_root)
      return std::unexpected("comparison binding must name an original root");
    if (range.allocation_bytes == 0 || range.bytes == 0 || range.alignment == 0 ||
        !std::has_single_bit(range.alignment) || range.offset % range.alignment != 0)
      return std::unexpected("comparison binding has an empty or misaligned allocation range");
    std::uint64_t end = 0;
    if (__builtin_add_overflow(range.offset, range.bytes, &end) || end > range.allocation_bytes)
      return std::unexpected("comparison binding range overflows its allocation or address");
    const bool concrete = !input.addresses.measuring();
    if (concrete) {
      std::uint64_t address = 0;
      if (__builtin_add_overflow(range.base, range.offset, &address) ||
          __builtin_add_overflow(address, range.bytes, &end))
        return std::unexpected("comparison binding range overflows its allocation or address");
      if (address % range.alignment != 0)
        return std::unexpected("comparison binding address violates its declared alignment");
    }
    if (std::holds_alternative<ir::WeightRef>(graph.tensor(range.original_root).binding) &&
        !range.part)
      return std::unexpected("comparison weight binding is missing its semantic part");
    if (!std::holds_alternative<ir::WeightRef>(graph.tensor(range.original_root).binding) &&
        range.row_storage)
      return std::unexpected("comparison non-weight binding carries row-table facts");
    const auto& original = graph.tensor(range.original_root);
    const auto parts = format::PartsOf(original.format);
    if (!parts) return std::unexpected("comparison binding has an invalid original format");
    std::uint64_t required = 0;
    if (std::holds_alternative<ir::WeightRef>(original.binding)) {
      const auto extent = parts->Find(*range.part);
      if (!extent) return std::unexpected("comparison binding names a part absent from its format");
      required = extent->readable;
      if (range.row_storage) {
        const auto& ref = std::get<ir::WeightRef>(original.binding);
        if (!input.addresses.row_storage || !input.addresses.weight_rows ||
            !format::RowBytes(original.format, *range.part) ||
            input.addresses.row_storage(ref, *range.part) != range.row_storage ||
            input.addresses.weight_rows(ref) != range.row_storage->slot_capacity ||
            !components::ValidComparisonRowStorage(*range.row_storage, ref, *range.part,
                                                   original.format, range.bytes))
          return std::unexpected("comparison row-table range lacks matching owner provenance");
        const auto actual = input.addresses.weight_range
                                ? input.addresses.weight_range(ref, *range.part)
                                : std::nullopt;
        const auto address =
            input.addresses.weight ? input.addresses.weight(ref, *range.part) : std::uint64_t{0};
        if (!actual || actual->base != range.base || actual->bytes != range.bytes ||
            actual->alignment != range.alignment || address != range.base)
          return std::unexpected("comparison row-table range differs from its actual owner slot");
        required = range.bytes;
      }
    } else {
      if (range.part)
        return std::unexpected("comparison non-weight binding names a semantic weight part");
      const auto root_bytes = planner::RootBytes(original);
      if (!root_bytes) return std::unexpected(root_bytes.error());
      required = *root_bytes;
      if (const auto data = parts->Find(format::PartRole::kData))
        required = std::max(required, data->readable);
      required = std::max(required, WrittenBytes(graph, range.original_root));
    }
    if (range.bytes < required)
      return std::unexpected(
          "comparison binding range is smaller than its readable/written extent");
    for (std::size_t j = 0; j < i; ++j) {
      if (bindings[j].original_root == range.original_root && bindings[j].part == range.part)
        return std::unexpected("comparison binding repeats an original root and part");
      const auto& other = bindings[j];
      if (other.allocation_class == range.allocation_class &&
          ((concrete && other.base != range.base) ||
           other.allocation_bytes != range.allocation_bytes))
        return std::unexpected("comparison alias class has inconsistent allocation ownership");
      std::uint64_t this_begin = range.offset, this_end = end;
      std::uint64_t other_begin = other.offset, other_end = 0;
      if (__builtin_add_overflow(other_begin, other.bytes, &other_end))
        return std::unexpected("comparison binding overlap interval overflows");
      if (concrete && (__builtin_add_overflow(range.base, range.offset, &this_begin) ||
                       __builtin_add_overflow(this_begin, range.bytes, &this_end) ||
                       __builtin_add_overflow(other.base, other.offset, &other_begin) ||
                       __builtin_add_overflow(other_begin, other.bytes, &other_end)))
        return std::unexpected("comparison binding overlap interval overflows");
      const bool overlaps = this_begin < other_end && other_begin < this_end;
      if (concrete && overlaps && other.allocation_class != range.allocation_class)
        return std::unexpected("comparison overlapping ranges disagree on backing alias class");
    }
  }
  for (const auto input_id : input.io.inputs) {
    const auto root = graph.root(input_id);
    const bool unstored =
        expected_boundary &&
        std::ranges::count(expected_boundary->unstored_inputs(), input_id,
                           &components::ComparisonBoundaryUnstoredInput::original_input) == 1;
    if (unstored) {
      if (!input.token || std::ranges::count(input.token->step.inputs, input_id,
                                             &components::StepInput::tensor) != 1)
        return std::unexpected("typed unstored input has no unique token input contract");
      const auto token_input =
          std::ranges::find(input.token->step.inputs, input_id, &components::StepInput::tensor);
      if (token_input == input.token->step.inputs.end() ||
          token_input->kind != components::InputKind::kCellPositions)
        return std::unexpected("typed unstored input is not cell positions");
      const auto planned_id = mapped(input_id);
      if (!planned_id.valid() || planned_id.v >= input.plan.graph.tensor_count() ||
          !std::ranges::contains(input.plan.graph.inputs(), planned_id))
        return std::unexpected("typed unstored input is not a planned graph input");
      std::size_t readers = 0;
      for (const auto& node : input.plan.graph.nodes()) {
        for (std::size_t source = 0; source < node.nsrc; ++source) {
          if (node.source(source) != planned_id) continue;
          ++readers;
          if (node.key != ir::Primitive(ir::Op::kMaskBuild) || source != 1)
            return std::unexpected("typed unstored input has a non-mask reader");
        }
      }
      if (readers == 0 || covers(root, std::nullopt))
        return std::unexpected("typed unstored input is backed or has no mask reader");
      continue;
    }
    if (!covers(root, std::nullopt))
      return std::unexpected("comparison binding is missing an original graph input root");
  }
  for (std::size_t weight_index = 0; weight_index < input.io.weights.size(); ++weight_index) {
    const auto& weight = input.io.weights[weight_index];
    for (std::size_t prior = 0; prior < weight_index; ++prior)
      if (input.io.weights[prior].ref == weight.ref)
        return std::unexpected("comparison repeats a complete weight binding identity");
    const auto root = graph.root(weight.tensor);
    const auto* weight_ref = std::get_if<ir::WeightRef>(&graph.tensor(root).binding);
    if (!weight_ref || *weight_ref != weight.ref)
      return std::unexpected(
          "comparison weight declaration differs from its original graph binding");
    const auto format_parts = format::PartsOf(graph.tensor(root).format);
    if (!format_parts)
      return std::unexpected("comparison weight has an invalid original tensor format");
    for (std::size_t part_index = 0; part_index < weight.parts.size(); ++part_index)
      for (std::size_t prior = 0; prior < part_index; ++prior)
        if (weight.parts[prior].first == weight.parts[part_index].first ||
            weight.parts[prior].second == weight.parts[part_index].second)
          return std::unexpected("comparison weight repeats a part name or semantic role");
    if (weight.parts.empty() || weight.parts.size() != format_parts->size())
      return std::unexpected("comparison weight parts differ from its declared tensor format");
    if (!weight.parts.empty()) {
      for (const auto& extent : format_parts->parts()) {
        if (std::ranges::none_of(weight.parts,
                                 [&](const auto& part) { return part.second == extent.role; }))
          return std::unexpected("comparison weight omits a format part");
      }
    }
    if (weight.parts.empty()) {
      if (!covers(root, std::nullopt))
        return std::unexpected("comparison binding is missing a weight root");
    } else {
      for (const auto& [name, part] : weight.parts) {
        (void)name;
        if (!covers(root, part))
          return std::unexpected("comparison binding is missing a weight part");
        const auto row_bytes = format::RowBytes(graph.tensor(root).format, part);
        const auto range = std::ranges::find_if(bindings, [&](const ComparisonBindingRange& item) {
          return item.original_root == root && item.part == part;
        });
        const auto rows = input.addresses.weight_rows ? input.addresses.weight_rows(weight.ref)
                                                      : std::uint64_t{0};
        const auto owner_fact = input.addresses.row_storage
                                    ? input.addresses.row_storage(weight.ref, part)
                                    : std::nullopt;
        if (row_bytes && (rows != 0 || owner_fact) &&
            (rows == 0 || !owner_fact || range == bindings.end() || !range->row_storage ||
             range->row_storage != owner_fact || range->row_storage->slot_capacity != rows))
          return std::unexpected("row-addressable source is missing its exact typed owner fact");
        if (range != bindings.end() && range->row_storage && !row_bytes)
          return std::unexpected("row-table facts were attached to a non-row-addressable part");
      }
    }
  }
  for (const auto& weight : input.io.weights) {
    const auto root = graph.root(weight.tensor);
    const auto& tensor_format = graph.tensor(root).format;
    for (const auto& [name, part] : weight.parts) {
      (void)name;
      const auto row_bytes = format::RowBytes(tensor_format, part);
      const auto fact = input.addresses.row_storage ? input.addresses.row_storage(weight.ref, part)
                                                    : std::nullopt;
      if (!row_bytes || !fact) continue;
      if (!input.token || weight.ref.stage != input.token->shape.stage)
        return std::unexpected("row-table GetRows requires the actual same-stage token step");
      const auto mapped_weight = mapped(root);
      if (!mapped_weight.valid() || mapped_weight.v >= input.plan.graph.tensor_count())
        return std::unexpected("row-table weight is absent from the planned graph");
      const auto planned_weight = input.plan.graph.root(mapped_weight);
      std::size_t uses = 0;
      for (std::size_t node_index = 0; node_index < input.plan.graph.nodes().size(); ++node_index) {
        const ir::NodeId node_id{static_cast<std::uint32_t>(node_index)};
        const auto& node = input.plan.graph.node(node_id);
        for (std::size_t source_index = 0; source_index < node.nsrc; ++source_index) {
          if (input.plan.graph.root(node.source(source_index)) != planned_weight) continue;
          if (node.source(source_index) != planned_weight ||
              node.key != ir::Primitive(ir::Op::kGetRows) || source_index != 0 || node.nsrc != 2)
            return std::unexpected("short row-table slot is used outside direct GetRows");
          ++uses;
          const auto planned_indices = input.plan.graph.root(node.source(1));
          if (node.source(1) != planned_indices)
            return std::unexpected("GetRows index is a view rather than the typed side input");
          std::optional<ir::TensorId> original_indices;
          for (const auto original_input : input.io.inputs) {
            if (!original_input.valid() || original_input.v >= input.plan.tensors.size()) continue;
            const auto candidate = input.plan.tensors[original_input.v];
            if (candidate.valid() && candidate.v < input.plan.graph.tensor_count() &&
                input.plan.graph.root(candidate) == planned_indices) {
              if (original_indices) {
                original_indices.reset();
                break;
              }
              original_indices = original_input;
            }
          }
          if (!original_indices || !input.token ||
              std::ranges::count(input.token->step.inputs, *original_indices,
                                 &components::StepInput::tensor) != 1)
            return std::unexpected("GetRows index is not one original typed side input");
          const auto side = std::ranges::find(input.token->step.inputs, *original_indices,
                                              &components::StepInput::tensor);
          if (side == input.token->step.inputs.end() ||
              side->kind != components::InputKind::kSideRows ||
              side->segment != ir::kJoinedSegment || side->component != weight.ref.component ||
              !original_indices->valid() ||
              graph.tensor(*original_indices).format.encoding != format::Encoding::kI32 ||
              graph.root(*original_indices) != *original_indices)
            return std::unexpected(
                "GetRows indices lack the exact joined same-component I32 input");
        }
      }
      if (uses == 0)
        return std::unexpected("row-table slots have no authenticated direct GetRows consumer");
    }
  }
  if (input.comparison) {
    for (std::size_t state_index = 0; state_index < input.comparison->state.size(); ++state_index) {
      const auto& state_fact = input.comparison->state[state_index];
      if (!state_fact.tensor.valid() || state_fact.tensor.v >= graph.tensor_count() ||
          graph.root(state_fact.tensor) != state_fact.tensor)
        return std::unexpected("comparison state declaration has an invalid original root");
      for (std::size_t prior = 0; prior < state_index; ++prior)
        if (input.comparison->state[prior].ref == state_fact.ref)
          return std::unexpected("comparison repeats a complete state binding identity");
      if (!covers(state_fact.tensor, std::nullopt))
        return std::unexpected("comparison binding is missing a state root");
    }
    if (input.token && input.comparison->state.size() != input.token->step.state.size())
      return std::unexpected("comparison state declarations differ from the token step");
    if (input.token) {
      if (&input.io.graph != &input.token->step.graph)
        return std::unexpected("comparison original graph is not the token step graph");
      for (const auto& requested : input.comparison->state) {
        if (std::ranges::count(input.token->step.state, requested.ref,
                               &components::StepState::ref) != 1)
          return std::unexpected("comparison state declarations differ from the token step");
        const auto at =
            std::ranges::find(input.token->step.state, requested.ref, &components::StepState::ref);
        if (at == input.token->step.state.end())
          return std::unexpected("comparison state declarations differ from the token step");
        const auto& declared = *at;
        if (requested.tensor != declared.tensor || requested.ref != declared.ref ||
            requested.item != declared.item || requested.borrowed != declared.borrowed ||
            requested.tap != declared.tap || requested.source != declared.source ||
            requested.read != declared.read || requested.written != declared.written)
          return std::unexpected("comparison state declarations differ from the token step");
      }
      if (auto checked = components::CheckStateAccess(graph, input.comparison->state); !checked)
        return std::unexpected(checked.error());
      if (auto checked = components::CheckStateWrites(graph, input.comparison->state, {},
                                                      input.comparison->active_stage);
          !checked)
        return std::unexpected(checked.error());
      if (auto checked =
              components::CheckStateWrites(input.plan.graph, input.comparison->state,
                                           input.plan.tensors, input.comparison->active_stage);
          !checked)
        return std::unexpected(checked.error());
    }
    for (std::uint32_t i = 0; i < graph.tensor_count(); ++i) {
      const ir::TensorId id{i};
      const auto* state_ref = std::get_if<ir::StateRef>(&graph.tensor(id).binding);
      if (!state_ref) continue;
      const bool declared = std::ranges::any_of(input.comparison->state, [&](const auto& state) {
        return state.tensor == id && state.ref == *state_ref;
      });
      if (!declared)
        return std::unexpected("comparison state declaration omits an original graph state root");
    }
  }
  return {};
}

}  // namespace llmp::kernels::ggml
