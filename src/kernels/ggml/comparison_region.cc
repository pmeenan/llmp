// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/comparison_region.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>

#include "ggml.h"
#include "kernels/ggml/comparison.h"
#include "kernels/ggml/comparison_access.h"
#include "kernels/ggml/comparison_schema.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/set_rows_group.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

namespace llmp::kernels::ggml {
namespace {

constexpr std::uint32_t kUnset = UINT32_MAX;
constexpr std::size_t kRemainderAccessCapacity = 3 * kSetRowsGroupMax + 8;

bool IsPacked(const ggml_tensor* tensor) {
  if (!tensor) return false;
  bool packed = tensor->nb[0] == ggml_type_size(tensor->type);
  for (int i = 1; i < GGML_MAX_DIMS; ++i)
    packed =
        packed && tensor->nb[i] == tensor->nb[i - 1] * static_cast<std::size_t>(tensor->ne[i - 1]);
  return packed;
}

bool HasExactFlattenedStorageView(const ggml_tensor* view, const ggml_tensor* source) {
  if (!view || !source) return false;
  const ggml_tensor* source_root = source->view_src ? source->view_src : source;
  const auto source_offset = source->view_src ? source->view_offs : std::size_t{0};
  if (source->view_src) {
    std::uint64_t end = 0;
    if (__builtin_add_overflow(static_cast<std::uint64_t>(source_offset),
                               static_cast<std::uint64_t>(ggml_nbytes(source)), &end) ||
        end > ggml_nbytes(source_root))
      return false;
  }
  return view->view_src == source_root && view->view_offs == source_offset;
}

bool ExactNormGateZView(const ggml_tensor* node, const ggml_tensor* z, const ggml_tensor* parent) {
  if (!node || !z || !parent || node->op != GGML_OP_CUSTOM || z->op != GGML_OP_RESHAPE ||
      node->src[0] == nullptr || node->src[1] == nullptr || node->src[2] != z ||
      z->src[0] != parent || node->src[0]->type != GGML_TYPE_F32 || parent->type != GGML_TYPE_F32 ||
      z->type != GGML_TYPE_F32 || !CheckGdnNormGate(node) || !ggml_is_contiguous(node->src[0]) ||
      !ggml_is_contiguous(parent) || !ggml_is_contiguous(z))
    return false;
  const auto d = node->src[0]->ne[0];
  const auto heads = node->src[0]->ne[1];
  const auto rows = node->src[0]->ne[2];
  std::int64_t flattened = 0;
  if (d <= 0 || heads <= 0 || rows <= 0 || __builtin_mul_overflow(d, heads, &flattened) ||
      z->ne[0] != flattened || z->ne[1] != rows || z->ne[2] != 1 || z->ne[3] != 1 ||
      ggml_nelements(z) != ggml_nelements(parent) || ggml_nbytes(z) != ggml_nbytes(parent) ||
      z->data != parent->data || !HasExactFlattenedStorageView(z, parent))
    return false;
  const auto* root = parent->view_src ? parent->view_src : parent;
  const auto offset = parent->view_src ? parent->view_offs : std::size_t{0};
  std::uintptr_t expected = 0;
  std::uint64_t end = 0;
  return root->type == GGML_TYPE_F32 &&
         !__builtin_add_overflow(reinterpret_cast<std::uintptr_t>(root->data),
                                 static_cast<std::uintptr_t>(offset), &expected) &&
         expected == reinterpret_cast<std::uintptr_t>(z->data) &&
         !__builtin_add_overflow(static_cast<std::uint64_t>(offset),
                                 static_cast<std::uint64_t>(ggml_nbytes(z)), &end) &&
         end <= ggml_nbytes(root);
}

// Native source reshapes are descriptor-only edges. Keep each selected
// helper's operand meaning explicit while sharing the existing paid-view path.
std::optional<std::size_t> AuxiliarySourceOrdinal(const ggml_tensor* node, const ggml_tensor* view,
                                                  const ggml_tensor* parent) {
  if (ExactNormGateZView(node, view, parent)) return 2;
  if (!node || !view || !parent || node->op != GGML_OP_CUSTOM ||
      LlmpOpOf(node) != LlmpOp::kHcCombine || node->src[1] != view || view->op != GGML_OP_RESHAPE ||
      view->src[0] != parent || !CheckHcCombine(node) || !HasTypedDescriptorSchema(*view) ||
      parent->type != GGML_TYPE_F32 || view->type != GGML_TYPE_F32 ||
      parent->ne[0] != node->ne[0] || parent->ne[1] != node->ne[2] || parent->ne[2] != 1 ||
      parent->ne[3] != 1 || !ggml_is_contiguous(parent) || !ggml_is_contiguous(view) ||
      ggml_nbytes(parent) != ggml_nbytes(view) || parent->data != view->data ||
      !HasExactFlattenedStorageView(view, parent) || !HasNoExtraSources(*view, 1))
    return {};
  return 1;
}

bool IsQ8Implementation(std::string_view implementation) {
  return implementation == kQuantizeQ8Name || implementation == kMulMatVecQ ||
         implementation == kMmvqPreparedName;
}

bool HasIdentity(const base::Sha256Digest& digest) {
  return std::ranges::any_of(digest, [](std::uint8_t value) { return value != 0; });
}

using LocalHash = CanonicalHash;

// Narrow unquantized MMVF products. Dispatch/scratch are independently
// authenticated against the current launch before a local certificate closes.
bool FloatMmvfCall(const PlanStep& step) {
  if (step.implementation != kMulMatVector || step.operation != ir::Primitive(ir::Op::kMatMul) ||
      step.nodes.size() != 1 || !step.nodes.front())
    return false;
  const auto* node = step.nodes.front();
  return node->type == GGML_TYPE_F32 && node->src[0] && node->src[1] &&
         node->src[1]->type == GGML_TYPE_F32 &&
         (node->src[0]->type == GGML_TYPE_F32 ||
          (node->src[0]->type == GGML_TYPE_F16 && HasTypedDescriptorSchema(*node) &&
           node->op_params[0] == GGML_PREC_DEFAULT && ggml_is_contiguous(node->src[0]) &&
           ggml_is_contiguous(node->src[1]) && ggml_is_contiguous(node))) &&
         CheckMulMat(node).has_value();
}

// These are the actual MMVF launch operands: the intermediate products of a
// fused call are descriptor helpers, not separate writes.
bool F16MatMulGluCall(const PlanStep& step) {
  if (step.operation != ir::Fused(ir::FusedKind::kMatMulGlu) ||
      step.implementation != kMulMatGluFused || step.nodes.size() != 3)
    return false;
  const auto* gate = step.nodes[0];
  const auto* up = step.nodes[1];
  const auto* glu = step.nodes[2];
  return gate && up && glu && HasTypedDescriptorSchema(*gate) && HasTypedDescriptorSchema(*up) &&
         HasTypedDescriptorSchema(*glu) && gate->src[0] && up->src[0] && gate->src[1] &&
         gate->src[0]->type == GGML_TYPE_F16 && up->src[0]->type == GGML_TYPE_F16 &&
         gate->src[1]->type == GGML_TYPE_F32 && glu->type == GGML_TYPE_F32 &&
         gate->op_params[0] == GGML_PREC_DEFAULT && up->op_params[0] == GGML_PREC_DEFAULT &&
         ggml_is_contiguous(gate->src[0]) && ggml_is_contiguous(up->src[0]) &&
         ggml_is_contiguous(gate->src[1]) && ggml_is_contiguous(glu) &&
         CheckMulMatVecGlu(gate, up, glu).has_value();
}

bool SplitSwiGluCall(const PlanStep& step) {
  if (step.operation != ir::Primitive(ir::Op::kGlu) || step.implementation != kSwiGluName ||
      step.nodes.size() != 1 || !step.nodes[0])
    return false;
  const auto* node = step.nodes[0];
  return CheckComparisonF16SwiGlu(node->src[0], node->src[1], node);
}

bool NormAddCall(const PlanStep& step) {
  return step.operation == ir::Fused(ir::FusedKind::kRmsNormMulAdd) &&
         step.implementation == kGemmaNormAddName && step.nodes.size() == 3 &&
         CheckComparisonNormAdd(step.nodes[0], step.nodes[1], step.nodes[2]);
}

bool MatMulAddCall(const PlanStep& step) {
  return step.operation == ir::Fused(ir::FusedKind::kMatMulAdd) &&
         step.implementation == kMulMatAddFused && step.nodes.size() == 2 &&
         CheckComparisonMatMulAdd(step.nodes[0], step.nodes[1]);
}

bool RegionCallClosed(const PlanStep& step) {
  if (HasComparisonAccessRecipe(step.implementation))
    return DescribeComparisonAccess(step, ComparisonAccessScope::kLocal).has_value();
  if (FloatMmvfCall(step) || F16MatMulGluCall(step) || SplitSwiGluCall(step) || NormAddCall(step) ||
      MatMulAddCall(step))
    return true;
  if (IsQ8Implementation(step.implementation)) return true;
  if (step.operation == ir::Fused(ir::FusedKind::kNormGate)) {
    if (step.nodes.size() != 1 || step.implementation != kGdnNormGateName) return false;
    const auto* node = step.nodes.front();
    return node && node->type == GGML_TYPE_F32 && node->src[0] &&
           node->src[0]->type == GGML_TYPE_F32 && node->src[1] &&
           node->src[1]->type == GGML_TYPE_F32 && node->src[2] &&
           node->src[2]->type == GGML_TYPE_F32 && CheckGdnNormGate(node).has_value();
  }
  if (step.operation == ir::Fused(ir::FusedKind::kRmsNormMulRope)) {
    return step.implementation == kGemmaNormRopeName && step.nodes.size() == 3 &&
           CheckGemmaNormRope(step.nodes[0], step.nodes[1], step.nodes[2]).has_value();
  }
  if (step.nodes.size() == 1 && step.operation == ir::Primitive(ir::Op::kRope))
    return CheckRope(step.nodes[0]).has_value();
  if (step.operation.op == ir::Op::kRoute || step.operation.op == ir::Op::kCacheStore ||
      step.operation.op == ir::Op::kHcPost || step.operation.op == ir::Op::kMatMulId ||
      step.operation.op == ir::Op::kGatedDeltaNet)
    return false;
  // A whole-program descriptor does not imply that an arbitrary call is
  // independently sealable as a local recipe.
  return false;
}

bool DescriptorFirstDomainCall(const PlanStep& step) {
  if (HasComparisonAccessRecipe(step.implementation))
    return DescribeComparisonAccess(step, ComparisonAccessScope::kLocal).has_value();
  if (FloatMmvfCall(step) || ComparisonSgemmCall(step) || F16MatMulGluCall(step) ||
      SplitSwiGluCall(step) || NormAddCall(step) || MatMulAddCall(step))
    return true;
  if (IsQ8Implementation(step.implementation)) return true;
  if (step.operation == ir::Fused(ir::FusedKind::kNormGate))
    return step.nodes.size() == 1 && step.implementation == kGdnNormGateName;
  if (step.operation == ir::Fused(ir::FusedKind::kRmsNormMulRope))
    return step.implementation == kGemmaNormRopeName && step.nodes.size() == 3 &&
           CheckGemmaNormRope(step.nodes[0], step.nodes[1], step.nodes[2]).has_value();
  if (step.nodes.size() != 1) return false;
  if (step.operation == ir::Primitive(ir::Op::kRope))
    return step.implementation == kRopeName && step.nodes.front() &&
           step.nodes.front()->type == GGML_TYPE_F32 && CheckRope(step.nodes.front()).has_value();
  return false;
}

const ComparisonQ8CallRecipe* Q8CallForStep(const ComparisonQ8SpanRecipe* recipe,
                                            const PlanStep& step) {
  if (!recipe) return nullptr;
  const auto calls = std::span(recipe->calls).first(recipe->call_count);
  const auto found =
      std::ranges::find_if(calls, [&](const auto& call) { return call.step == &step; });
  return found == calls.end() ? nullptr : &*found;
}

bool AddDescriptorOrigins(const LoweredStep::ComparisonOrigins& relation,
                          const LoweredStep& lowered, const ggml_tensor* node,
                          std::span<std::uint8_t> membership, std::vector<ir::NodeId>& expansion,
                          bool& intersects, bool expand) {
  const auto* physical = PhysicalTensorOf(lowered, node);
  if (!physical || !physical->producer.valid() ||
      physical->producer.v >= relation.offsets.size() - 1)
    return false;
  const auto producer = physical->producer.v;
  const auto first = relation.offsets[producer];
  const auto end = relation.offsets[producer + 1];
  if (first >= end || end > relation.source_nodes.size()) return false;
  for (auto offset = first; offset < end; ++offset) {
    const auto source = relation.source_nodes[static_cast<std::size_t>(offset)];
    if (!source.valid() || source.v >= membership.size()) return false;
    if (membership[source.v] == 1 || membership[source.v] == 2 || membership[source.v] == 3)
      intersects = true;
    if (!expand) continue;
    if (membership[source.v] == 1) membership[source.v] = 3;
    if (membership[source.v] == 0) {
      membership[source.v] = 2;
      expansion.push_back(source);
    }
  }
  return true;
}

bool AddCallOrigins(const LoweredStep::ComparisonOrigins& relation, const LoweredStep& lowered,
                    const PlanStep& step, std::span<std::uint8_t> membership,
                    std::vector<ir::NodeId>& expansion, bool& intersects, bool expand) {
  intersects = false;
  if (step.nodes.empty()) return false;
  for (const auto* node : step.nodes)
    if (!AddDescriptorOrigins(relation, lowered, node, membership, expansion, intersects, expand))
      return false;
  return true;
}

bool ExactSavedCallChain(const LoweredStep& lowered, const SavedLeaf& save,
                         const ggml_tensor* candidate) {
  const auto source_record =
      std::ranges::find(lowered.physical_tensors, save.source, &PhysicalTensor::tensor);
  const auto rows_record =
      std::ranges::find(lowered.physical_tensors, save.rows, &PhysicalTensor::tensor);
  if (!candidate || !save.tensor || !save.store || !save.source.valid() || !save.rows.valid() ||
      save.store->op != GGML_OP_SET_ROWS || save.store->src[0] == nullptr ||
      save.store->src[1] == nullptr || save.store->src[2] != save.tensor ||
      save.store->view_src != save.tensor || save.store->view_offs != 0)
    return false;

  if (source_record == lowered.physical_tensors.end() || !source_record->descriptor ||
      rows_record == lowered.physical_tensors.end() ||
      rows_record->descriptor != save.store->src[1] ||
      std::ranges::count(lowered.physical_tensors, save.source, &PhysicalTensor::tensor) != 1 ||
      std::ranges::count(lowered.physical_tensors, save.rows, &PhysicalTensor::tensor) != 1)
    return false;

  const auto* source_tensor = source_record->descriptor;
  const auto* values = save.store->src[0];
  if (values->op != GGML_OP_RESHAPE || !values->src[0] ||
      std::ranges::find(lowered.nodes, values) == lowered.nodes.end())
    return false;
  const bool generated_copy = save.packed && save.packed != source_tensor &&
                              save.packed->op == GGML_OP_CONT && save.scratch_bytes != 0;
  if (IsPacked(source_tensor)) {
    if (save.packed != source_tensor || values->src[0] != source_tensor) return false;
  } else if (!generated_copy || save.packed->src[0] != source_tensor ||
             save.packed->view_src != nullptr || values->src[0] != save.packed ||
             std::ranges::find(lowered.nodes, save.packed) == lowered.nodes.end()) {
    return false;
  }
  if (!HasExactFlattenedStorageView(values, save.packed)) return false;
  if (std::ranges::find(lowered.nodes, save.store) == lowered.nodes.end()) return false;
  return candidate == save.store || candidate == values ||
         (generated_copy && candidate == save.packed);
}

bool ExactSavedStoreCall(const LoweredStep& lowered, const PlanStep& step) {
  if (step.nodes.size() != 1 || !step.nodes.front()) return false;
  const auto* store = step.nodes.front();
  return std::ranges::any_of(lowered.saves, [&](const auto& save) {
    return save.store == store && ExactSavedCallChain(lowered, save, store);
  });
}

bool ExactSavedAuxiliaryCall(const LoweredStep& lowered, const PlanStep& step) {
  if (step.nodes.size() != 1 || !step.nodes.front()) return false;
  const auto* node = step.nodes.front();
  for (const auto& save : lowered.saves) {
    if (ExactSavedCallChain(lowered, save, node) && node != save.store) return true;
  }
  return false;
}

enum class SavedCallKind : std::uint8_t { kNone, kPackedCopy, kValuesView, kStore };

struct SavedCall {
  const SavedLeaf* leaf = nullptr;
  SavedCallKind kind = SavedCallKind::kNone;
};

SavedCall FindSavedCall(const LoweredStep& lowered, const PlanStep& step) {
  if (step.nodes.size() != 1 || !step.nodes.front()) return {};
  const auto* node = step.nodes.front();
  for (const auto& save : lowered.saves) {
    if (!ExactSavedCallChain(lowered, save, node)) continue;
    if (node == save.store) return {.leaf = &save, .kind = SavedCallKind::kStore};
    const auto source =
        std::ranges::find(lowered.physical_tensors, save.source, &PhysicalTensor::tensor);
    if (node == save.packed && save.packed->op == GGML_OP_CONT &&
        source != lowered.physical_tensors.end() && source->descriptor != save.packed &&
        save.scratch_bytes != 0)
      return {.leaf = &save, .kind = SavedCallKind::kPackedCopy};
    if (node == save.store->src[0]) return {.leaf = &save, .kind = SavedCallKind::kValuesView};
  }
  return {};
}

bool SavedMappingCurrent(const LoweredStep& lowered, const SavedLeaf& save) {
  if (!lowered.comparison_origins || !save.original_source.valid() || !save.original_rows.valid())
    return false;
  const auto& origins = *lowered.comparison_origins;
  return save.original_source.v < origins.original_to_planned.size() &&
         save.original_rows.v < origins.original_to_planned.size() &&
         origins.original_to_planned[save.original_source.v] == save.source &&
         origins.original_to_planned[save.original_rows.v] == save.rows;
}

bool SavedOwnerRangeCurrent(const SavedLeaf& save) {
  if (!save.owner_range || !save.owner_range_current || !save.tensor || !save.tensor->data)
    return false;
  const auto& range = *save.owner_range;
  const auto required = static_cast<std::uint64_t>(ggml_nbytes(save.tensor));
  return range.base != 0 && range.bytes >= required && range.bytes != 0 && range.alignment != 0 &&
         std::has_single_bit(range.alignment) &&
         reinterpret_cast<std::uintptr_t>(save.tensor->data) == range.base &&
         range.base <= UINT64_MAX - range.bytes && range.base % range.alignment == 0;
}

const components::ComparisonRegionTensor* RegionTensor(
    const components::OwnedComparisonRegionBoundary& boundary, std::uint32_t canonical) {
  const auto tensors = boundary.tensors();
  const auto found =
      std::ranges::find(tensors, canonical, &components::ComparisonRegionTensor::canonical);
  return found == tensors.end() ? nullptr : &*found;
}

const components::ComparisonRegionRoot* RegionRoot(
    const components::OwnedComparisonRegionBoundary& boundary, std::uint32_t canonical) {
  const auto roots = boundary.roots();
  const auto found =
      std::ranges::find(roots, canonical, &components::ComparisonRegionRoot::canonical);
  if (found == roots.end() ||
      std::ranges::count(roots, canonical, &components::ComparisonRegionRoot::canonical) != 1)
    return nullptr;
  return &*found;
}

const ComparisonBindingRange* BindingForExternal(
    const LoweredStep& lowered, const components::OwnedComparisonRegionBoundary& boundary,
    const components::ComparisonRegionExternal& external) {
  const auto* root = RegionRoot(boundary, external.canonical_root);
  if (!root || !root->original_root.valid()) return nullptr;
  if (!lowered.comparison_origins) return nullptr;
  const auto& ranges = lowered.comparison_origins->original_bindings;
  const auto found = std::ranges::find_if(ranges, [&](const ComparisonBindingRange& range) {
    return range.original_root == root->original_root && range.part == external.part;
  });
  if (found == ranges.end() ||
      std::ranges::count_if(ranges, [&](const ComparisonBindingRange& range) {
        return range.original_root == root->original_root && range.part == external.part;
      }) != 1)
    return nullptr;
  if (found->allocation_bytes != external.allocation_bytes || found->offset != external.offset ||
      found->bytes != external.bytes || found->alignment != external.alignment ||
      found->resource_part_identity != external.resource_part_identity ||
      found->row_storage != external.row_storage)
    return nullptr;
  return &*found;
}

bool RegionBindingsCurrent(const LoweredStep& lowered,
                           const components::OwnedComparisonRegionBoundary& boundary) {
  if (!lowered.comparison_origins || !lowered.comparison_origins->source_boundary_verified ||
      boundary.active_stage() != lowered.comparison_origins->active_stage)
    return false;
  const auto roots = boundary.roots();
  const auto external = boundary.external();
  if (!lowered.comparison_origins) return false;
  const auto& ranges = lowered.comparison_origins->original_bindings;
  const auto& physical_ranges = lowered.comparison_bindings;
  if (ranges.size() != physical_ranges.size()) return false;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const auto& source = ranges[i];
    const auto& physical = physical_ranges[i];
    if (source.original_root != physical.original_root ||
        source.allocation_class != physical.allocation_class ||
        source.allocation_bytes != physical.allocation_bytes || source.offset != physical.offset ||
        source.bytes != physical.bytes || source.alignment != physical.alignment ||
        source.base != physical.base ||
        source.resource_part_identity != physical.resource_part_identity ||
        source.row_storage != physical.row_storage)
      return false;
  }
  for (const auto& fact : external) {
    if (!RegionRoot(boundary, fact.canonical_root) || !BindingForExternal(lowered, boundary, fact))
      return false;
  }
  for (const auto& range : ranges) {
    const bool root_is_local = std::ranges::any_of(
        roots, [&](const auto& root) { return root.original_root == range.original_root; });
    if (!root_is_local) continue;
    std::size_t matches = 0;
    for (const auto& fact : external) {
      const auto* root = RegionRoot(boundary, fact.canonical_root);
      if (root && root->original_root == range.original_root && fact.part == range.part) ++matches;
    }
    if (matches != 1) return false;
  }
  for (std::size_t i = 0; i < external.size(); ++i) {
    const auto* first = BindingForExternal(lowered, boundary, external[i]);
    if (!first) return false;
    for (std::size_t j = 0; j < i; ++j) {
      const auto* prior = BindingForExternal(lowered, boundary, external[j]);
      if (!prior ||
          ((external[i].canonical_allocation_class == external[j].canonical_allocation_class) !=
           (first->allocation_class == prior->allocation_class)))
        return false;
    }
  }
  return true;
}

struct RegionAccessSpec {
  const ggml_tensor* tensor = nullptr;
  const ggml_tensor* metadata_owner = nullptr;
  const ggml_tensor* semantic_parent = nullptr;
  bool read = false;
  bool written = false;
  std::uint64_t alignment = 0;
  std::uint64_t bytes = 0;
  bool pool = false;
  bool authenticated_state_write = false;
  std::uint8_t phase = 0;
};

struct RegionAccessList {
  std::array<RegionAccessSpec, kRemainderAccessCapacity> values{};
  std::size_t size = 0;
};

bool AddRegionAccess(RegionAccessList& list, const ggml_tensor* tensor, bool read, bool written,
                     std::uint64_t bytes = 0, std::uint64_t alignment = 0,
                     bool authenticated_state_write = false) {
  if (!tensor || list.size == list.values.size()) return false;
  if (alignment == 0) alignment = ComparisonRegionDefaultAccessAlignment(tensor->type);
  if (alignment == 0) return false;
  list.values[list.size++] = {.tensor = tensor,
                              .read = read,
                              .written = written,
                              .alignment = alignment,
                              .bytes = bytes,
                              .authenticated_state_write = authenticated_state_write};
  return true;
}

bool AddNormGateZRead(RegionAccessList& list, const ggml_tensor* node) {
  if (!node || !node->src[2] || !node->src[2]->src[0] ||
      !ExactNormGateZView(node, node->src[2], node->src[2]->src[0]) ||
      list.size == list.values.size())
    return false;
  const auto alignment = ComparisonRegionDefaultAccessAlignment(node->src[2]->type);
  if (alignment == 0) return false;
  list.values[list.size++] = {.tensor = node->src[2],
                              .metadata_owner = node,
                              .semantic_parent = node->src[2]->src[0],
                              .read = true,
                              .alignment = alignment};
  return true;
}

bool AddRegionPoolAccess(RegionAccessList& list, std::uint64_t bytes, bool read, bool written) {
  if (bytes == 0 || list.size == list.values.size()) return false;
  list.values[list.size++] = {.tensor = nullptr,
                              .read = read,
                              .written = written,
                              .alignment = 256,
                              .bytes = bytes,
                              .pool = true};
  return true;
}

bool DescribeRegionAccesses(const PlanStep& step, RegionAccessList& accesses,
                            const ComparisonQ8SpanRecipe* q8_recipe = nullptr) {
  if (HasComparisonAccessRecipe(step.implementation)) {
    const auto described = DescribeComparisonAccess(step, ComparisonAccessScope::kLocal);
    if (!described) return false;
    for (const auto& fact : std::span(described->values).first(described->count)) {
      if (!AddRegionAccess(accesses, fact.tensor, fact.read, fact.written, fact.bytes,
                           fact.alignment))
        return false;
      accesses.values[accesses.size - 1].phase = fact.phase;
      if (step.implementation == kHcCombineName && step.nodes.size() == 1 &&
          fact.tensor == step.nodes[0]->src[1] && fact.tensor->op == GGML_OP_RESHAPE) {
        if (AuxiliarySourceOrdinal(step.nodes[0], fact.tensor, fact.tensor->src[0]) != 1)
          return false;
        accesses.values[accesses.size - 1].metadata_owner = step.nodes[0];
        accesses.values[accesses.size - 1].semantic_parent = fact.tensor->src[0];
      }
    }
    return true;
  }

  if (ComparisonSgemmCall(step)) {
    const auto* node = step.nodes.front();
    return AddRegionAccess(accesses, node->src[0], true, false, 0, 4) &&
           AddRegionAccess(accesses, node->src[1], true, false, 0, 4) &&
           AddRegionAccess(accesses, node, false, true, 0, 4);
  }
  if (FloatMmvfCall(step)) {
    const auto* node = step.nodes.front();
    return AddRegionAccess(accesses, node->src[0], true, false, 0,
                           2 * ggml_type_size(node->src[0]->type)) &&
           AddRegionAccess(accesses, node->src[1], true, false, 0, 8) &&
           AddRegionAccess(accesses, node, false, true, 0, 8);
  }
  if (F16MatMulGluCall(step)) {
    const auto* gate = step.nodes[0];
    const auto* up = step.nodes[1];
    return AddRegionAccess(accesses, gate->src[0], true, false, 0, 4) &&
           AddRegionAccess(accesses, up->src[0], true, false, 0, 4) &&
           AddRegionAccess(accesses, up->src[1], true, false, 0, 8) &&
           AddRegionAccess(accesses, step.nodes[2], false, true, 0, 8);
  }
  if (SplitSwiGluCall(step)) {
    const auto* node = step.nodes[0];
    return AddRegionAccess(accesses, node->src[0], true, false) &&
           AddRegionAccess(accesses, node->src[1], true, false) &&
           AddRegionAccess(accesses, node, false, true);
  }
  if (NormAddCall(step)) {
    const auto* norm = step.nodes[0];
    const auto* mul = step.nodes[1];
    const auto* add = step.nodes[2];
    const auto* weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    const auto* residual = add->src[0] == mul ? add->src[1] : add->src[0];
    return AddRegionAccess(accesses, norm->src[0], true, false) &&
           AddRegionAccess(accesses, weight, true, false) &&
           AddRegionAccess(accesses, residual, true, false) &&
           AddRegionAccess(accesses, add, false, true);
  }
  if (MatMulAddCall(step)) {
    const auto* product = step.nodes[0];
    const auto* add = step.nodes[1];
    const auto* bias = add->src[0] == product ? add->src[1] : add->src[0];
    return AddRegionAccess(accesses, product->src[0], true, false, 0,
                           2 * ggml_type_size(product->src[0]->type)) &&
           AddRegionAccess(accesses, product->src[1], true, false, 0, 8) &&
           AddRegionAccess(accesses, bias, true, false) &&
           AddRegionAccess(accesses, add, false, true, 0, 8);
  }
  if (q8_recipe) {
    const auto call =
        std::ranges::find_if(std::span(q8_recipe->calls).first(q8_recipe->call_count),
                             [&](const auto& candidate) { return candidate.step == &step; });
    if (call != std::span(q8_recipe->calls).first(q8_recipe->call_count).end()) {
      if (call->kind == ComparisonQ8CallKind::kQuantize)
        return AddRegionAccess(accesses, call->input, true, false, 0, alignof(float)) &&
               AddRegionAccess(accesses, call->node, false, true, call->quantizer.output_bytes,
                               alignof(float));
      if (!call->weight || call->weight->ne[1] <= 0 || call->weight->ne[2] <= 0 ||
          call->weight->ne[3] <= 0)
        return false;
      const auto readable = ComparisonMmvqWeightReadableBytes(*call->weight, call->mmvq);
      if (!readable) return false;
      if (call->kind == ComparisonQ8CallKind::kMmvqPrepared)
        return AddRegionAccess(accesses, call->packed, true, false, call->quantizer.output_bytes,
                               alignof(float)) &&
               AddRegionAccess(accesses, call->weight, true, false, *readable, 16) &&
               AddRegionAccess(accesses, call->node, false, true, 0, alignof(float));
      return AddRegionAccess(accesses, call->input, true, false, 0, alignof(float)) &&
             AddRegionPoolAccess(accesses, call->scratch_bytes, false, true) &&
             AddRegionPoolAccess(accesses, call->scratch_bytes, true, false) &&
             AddRegionAccess(accesses, call->weight, true, false, *readable, 16) &&
             AddRegionAccess(accesses, call->node, false, true, 0, alignof(float));
    }
  }
  if (step.operation == ir::Fused(ir::FusedKind::kNormGate) && step.nodes.size() == 1 &&
      step.implementation == kGdnNormGateName) {
    const auto* node = step.nodes.front();
    if (!node || node->type != GGML_TYPE_F32 || !node->src[0] ||
        node->src[0]->type != GGML_TYPE_F32 || !node->src[1] ||
        node->src[1]->type != GGML_TYPE_F32 || !node->src[2] ||
        node->src[2]->type != GGML_TYPE_F32 || !CheckGdnNormGate(node))
      return false;
    return AddRegionAccess(accesses, node->src[0], true, false) &&
           AddRegionAccess(accesses, node->src[1], true, false) &&
           AddNormGateZRead(accesses, node) && AddRegionAccess(accesses, node, false, true);
  }
  if (step.operation == ir::Fused(ir::FusedKind::kRmsNormMulRope) &&
      step.implementation == kGemmaNormRopeName && step.nodes.size() == 3) {
    const auto* norm = step.nodes[0];
    const auto* mul = step.nodes[1];
    const auto* rope = step.nodes[2];
    if (!CheckGemmaNormRope(norm, mul, rope)) return false;
    const auto* weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if (!AddRegionAccess(accesses, norm->src[0], true, false) ||
        !AddRegionAccess(accesses, weight, true, false))
      return false;
    // The fused launch elides norm and mul; only RoPE's metadata
    // operands are read in addition to the original source and gamma.
    for (std::size_t i = 1; i < GGML_MAX_SRC; ++i)
      if (rope->src[i] && !AddRegionAccess(accesses, rope->src[i], true, false)) return false;
    return AddRegionAccess(accesses, rope, false, true);
  }
  if (step.nodes.size() != 1 || !step.nodes.front()) return false;
  const auto* node = step.nodes.front();
  if (step.operation == ir::Primitive(ir::Op::kRope) && step.implementation == kRopeName) {
    if (node->type != GGML_TYPE_F32 || !node->src[0] || node->src[0]->type != GGML_TYPE_F32 ||
        !CheckRope(node))
      return false;
    for (const auto* source : node->src)
      if (source && !AddRegionAccess(accesses, source, true, false)) return false;
    return AddRegionAccess(accesses, node, false, true);
  }

  return false;
}

const PhysicalRoot* PhysicalRootFor(const LoweredStep& lowered, ir::TensorId root) {
  const auto& roots = lowered.physical_roots;
  const auto found = std::ranges::find(roots, root, &PhysicalRoot::root);
  if (found == roots.end() || std::ranges::count(roots, root, &PhysicalRoot::root) != 1)
    return nullptr;
  return &*found;
}

const PhysicalTensor* PhysicalTensorForId(const LoweredStep& lowered, std::uint32_t id) {
  const auto& tensors = lowered.physical_tensors;
  const auto found = std::ranges::find(tensors, id, &PhysicalTensor::id);
  if (found == tensors.end() || std::ranges::count(tensors, id, &PhysicalTensor::id) != 1)
    return nullptr;
  return &*found;
}

struct GdnGatesStorage {
  const PhysicalTensor* parent = nullptr;
  std::array<const PhysicalTensor*, 2> outputs{};
  const PhysicalRoot* allocation = nullptr;
};

std::optional<GdnGatesStorage> CheckedGdnGatesStorage(const LoweredStep& lowered,
                                                      const ggml_tensor* descriptor) {
  const auto* parent = descriptor;
  if (parent && parent->op == GGML_OP_VIEW) parent = parent->src[0];
  if (!parent || parent->op != GGML_OP_CUSTOM || LlmpOpOf(parent) != LlmpOp::kGdnGates ||
      !lowered.comparison_origins)
    return {};
  const auto& relation = *lowered.comparison_origins;
  const auto* owner = PhysicalTensorOf(lowered, parent);
  if (!owner || owner->tensor.valid() || owner->root.valid() || !owner->producer.valid() ||
      owner->producer.v >= relation.planned_nodes.size())
    return {};
  const auto& node = relation.planned_nodes[owner->producer.v];
  if (node.key != ir::Fused(ir::FusedKind::kGdnGates) || node.outputs().size() != 2 ||
      node.sources().size() != 4)
    return {};
  const auto call = std::ranges::find_if(lowered.plan.steps, [&](const PlanStep& step) {
    return step.implementation == kGdnGatesName && step.operation == node.key &&
           step.nodes.size() == 1 && step.nodes[0] == parent;
  });
  if (call == lowered.plan.steps.end() ||
      !DescribeComparisonAccess(*call, ComparisonAccessScope::kLocal))
    return {};
  const auto matches_planned = [&](const PhysicalTensor& physical, ir::TensorId expected) {
    if (physical.tensor != expected || !expected.valid() ||
        expected.v >= relation.planned_tensors.size() || !physical.descriptor ||
        !physical.ir_format || *physical.ir_format != relation.planned_tensors[expected.v].format)
      return false;
    auto root = expected;
    for (std::size_t depth = 0; depth <= relation.planned_tensors.size(); ++depth) {
      if (!root.valid() || root.v >= relation.planned_tensors.size()) return false;
      const auto next = relation.planned_tensors[root.v].view_of;
      if (!next.valid()) break;
      if (depth == relation.planned_tensors.size()) return false;
      root = next;
    }
    if (physical.root != root) return false;
    const auto& shape = relation.planned_tensors[expected.v].format.shape;
    for (std::size_t axis = 0; axis < GGML_MAX_DIMS; ++axis)
      if (physical.descriptor->ne[axis] < 1 ||
          static_cast<std::uint64_t>(physical.descriptor->ne[axis]) !=
              (axis < shape.rank ? shape.ne[axis] : 1))
        return false;
    return true;
  };
  for (std::size_t i = 0; i < 4; ++i) {
    const auto* input = PhysicalTensorOf(lowered, parent->src[i]);
    if (!input || !matches_planned(*input, node.sources()[i])) return {};
  }
  GdnGatesStorage storage{.parent = owner};
  const auto plane = static_cast<std::uint64_t>(parent->ne[0]) *
                     static_cast<std::uint64_t>(parent->ne[1]) * sizeof(float);
  for (std::size_t i = 0; i < 2; ++i) {
    if (std::ranges::count(lowered.physical_tensors, node.outputs()[i], &PhysicalTensor::tensor) !=
        1)
      return {};
    const auto found =
        std::ranges::find(lowered.physical_tensors, node.outputs()[i], &PhysicalTensor::tensor);
    const auto* view = found->descriptor;
    if (!view || !matches_planned(*found, node.outputs()[i]) ||
        found->producer != owner->producer || !found->root.valid() || view->op != GGML_OP_VIEW ||
        view->type != GGML_TYPE_F32 || !HasTypedDescriptorSchema(*view) || view->src[0] != parent ||
        view->view_src != parent || view->view_offs != i * plane || view->ne[0] != 1 ||
        view->ne[1] != parent->ne[0] || view->ne[2] != parent->ne[1] || view->ne[3] != 1 ||
        view->nb[0] != 4 || view->nb[1] != 4 ||
        view->nb[2] != static_cast<std::uint64_t>(parent->ne[0]) * 4 || view->nb[3] != plane ||
        ggml_nbytes(view) != plane ||
        reinterpret_cast<std::uintptr_t>(view->data) !=
            reinterpret_cast<std::uintptr_t>(parent->data) + i * plane ||
        !HasNoExtraSources(*view, 1))
      return {};
    storage.outputs[i] = &*found;
  }
  storage.allocation = PhysicalRootFor(lowered, storage.outputs[0]->root);
  if (!storage.allocation || storage.allocation->backend_tensor != storage.outputs[0]->id ||
      storage.allocation->producer != owner->producer || storage.allocation->bytes != 2 * plane ||
      storage.allocation->offset > lowered.extent ||
      storage.allocation->bytes > lowered.extent - storage.allocation->offset ||
      ggml_nbytes(parent) != 2 * plane)
    return {};
  if (descriptor != parent && descriptor != storage.outputs[0]->descriptor &&
      descriptor != storage.outputs[1]->descriptor)
    return {};
  return storage;
}

std::uint64_t ComparisonSemanticViewOffset(const LoweredStep& lowered, const ggml_tensor* tensor) {
  if (const auto storage = CheckedGdnGatesStorage(lowered, tensor)) {
    const auto* physical = PhysicalTensorOf(lowered, tensor);
    if (physical && physical->tensor.valid() &&
        physical->tensor.v < lowered.comparison_origins->planned_tensors.size())
      return lowered.comparison_origins->planned_tensors[physical->tensor.v].view_offset;
  }
  return tensor->view_offs;
}

const ComparisonBindingRange* BindingForPhysicalRoot(const LoweredStep& lowered,
                                                     const LoweredStep::ComparisonOrigins& relation,
                                                     ir::TensorId planned_root,
                                                     std::optional<format::PartRole> part) {
  const auto& ranges = lowered.comparison_bindings;
  const auto found = std::ranges::find_if(ranges, [&](const ComparisonBindingRange& range) {
    return range.original_root.valid() &&
           range.original_root.v < relation.original_to_planned.size() &&
           relation.original_to_planned[range.original_root.v] == planned_root &&
           range.part == part;
  });
  if (found == ranges.end() ||
      std::ranges::count_if(ranges, [&](const ComparisonBindingRange& range) {
        return range.original_root.valid() &&
               range.original_root.v < relation.original_to_planned.size() &&
               relation.original_to_planned[range.original_root.v] == planned_root &&
               range.part == part;
      }) != 1)
    return nullptr;
  return &*found;
}

bool ReadyTokenValueKind(components::InputKind kind);

bool ResolveRegionAccess(const LoweredStep& lowered, LoweredStep::ComparisonOrigins& relation,
                         const RegionAccessSpec& spec, bool require_local_ids,
                         std::uint32_t& physical_count, ComparisonRegionAccessFact& result,
                         const Q8MainPoolDescriptor* pool = nullptr) {
  if (spec.pool) {
    if (!pool || pool->faulted || (!spec.read && !spec.written) || spec.bytes == 0 ||
        spec.alignment != 256 || pool->lanes != 0 || spec.bytes > pool->bytes ||
        pool->base % 256 != 0)
      return false;
    const auto address = static_cast<std::uintptr_t>(pool->base);
    std::uintptr_t end = 0;
    if (address == 0 ||
        __builtin_add_overflow(address, static_cast<std::uintptr_t>(spec.bytes), &end))
      return false;
    result.address = address;
    result.physical_id = kUnset;
    result.bytes = spec.bytes;
    result.required_alignment = 256;
    result.domain = ComparisonRegionAccessDomain::kPool;
    result.read = spec.read;
    result.written = spec.written;
    return true;
  }
  if (!spec.tensor || !spec.tensor->data || (!spec.read && !spec.written) || spec.alignment == 0)
    return false;
  const auto* descriptor = PhysicalTensorOf(lowered, spec.tensor);
  if (!descriptor || !descriptor->descriptor || descriptor->descriptor != spec.tensor) return false;
  const auto gates_storage = CheckedGdnGatesStorage(lowered, spec.tensor);
  const PhysicalTensor* physical =
      gates_storage && descriptor == gates_storage->parent ? gates_storage->outputs[0] : descriptor;
  if (spec.metadata_owner || spec.semantic_parent) {
    const auto ordinal =
        AuxiliarySourceOrdinal(spec.metadata_owner, spec.tensor, spec.semantic_parent);
    if (!ordinal || descriptor->tensor.valid() || descriptor->root.valid() ||
        (*ordinal == 1 && descriptor->producer.valid()))
      return false;
    physical = PhysicalTensorOf(lowered, spec.semantic_parent);
    if (!physical || !physical->descriptor || physical->descriptor != spec.semantic_parent ||
        !physical->tensor.valid() || physical->tensor.v >= relation.planned_tensors.size() ||
        !physical->root.valid() || physical->root.v >= relation.planned_tensors.size())
      return false;
    const auto* owner = PhysicalTensorOf(lowered, spec.metadata_owner);
    if (!owner || !owner->producer.valid() || owner->producer.v >= relation.planned_nodes.size() ||
        relation.planned_nodes[owner->producer.v].sources().size() <= *ordinal ||
        relation.planned_nodes[owner->producer.v].sources()[*ordinal] != physical->tensor ||
        relation.planned_nodes[owner->producer.v].key !=
            ir::Fused(*ordinal == 1 ? ir::FusedKind::kHcCombine : ir::FusedKind::kNormGate))
      return false;
  }
  if (!physical->tensor.valid() || physical->tensor.v >= relation.planned_tensors.size() ||
      !physical->root.valid() || physical->root.v >= relation.planned_tensors.size())
    return false;
  auto cursor = physical->tensor;
  for (std::size_t steps = 0; steps <= relation.planned_tensors.size(); ++steps) {
    if (!cursor.valid() || cursor.v >= relation.planned_tensors.size()) return false;
    const auto parent = relation.planned_tensors[cursor.v].view_of;
    if (!parent.valid()) break;
    cursor = parent;
    if (steps == relation.planned_tensors.size()) return false;
  }
  if (cursor != physical->root) return false;
  const auto logical_bytes = static_cast<std::uint64_t>(ggml_nbytes(spec.tensor));
  const auto bytes = spec.bytes == 0 ? logical_bytes : spec.bytes;
  if (bytes < logical_bytes) return false;
  const auto address = reinterpret_cast<std::uintptr_t>(spec.tensor->data);
  std::uintptr_t access_end = 0;
  if (bytes == 0 || address % spec.alignment != 0 ||
      __builtin_add_overflow(address, static_cast<std::uintptr_t>(bytes), &access_end))
    return false;
  if (descriptor->id >= relation.physical_local.size()) return false;
  std::uint32_t local_physical = kUnset;
  if (require_local_ids) {
    if (relation.physical_local[descriptor->id] == kUnset)
      relation.physical_local[descriptor->id] = physical_count++;
    local_physical = relation.physical_local[descriptor->id];
  }
  const auto local_tensor = relation.tensor_local[physical->tensor.v];
  const auto local_root = relation.tensor_local[physical->root.v];
  if (require_local_ids && (local_tensor == kUnset || local_root == kUnset)) return false;

  const auto* placed =
      gates_storage ? gates_storage->allocation : PhysicalRootFor(lowered, physical->root);
  std::uint64_t root_offset = 0, domain_offset = 0;
  ComparisonRegionAccessDomain domain = ComparisonRegionAccessDomain::kActivation;
  std::uint32_t allocation_class = 0;
  if (placed) {
    const auto* root_tensor = PhysicalTensorForId(lowered, placed->backend_tensor);
    if (!root_tensor || !root_tensor->descriptor || !root_tensor->descriptor->data ||
        (!gates_storage && root_tensor->root != physical->root) ||
        placed->offset > lowered.extent || placed->bytes > lowered.extent - placed->offset)
      return false;
    const auto root_address = reinterpret_cast<std::uintptr_t>(root_tensor->descriptor->data);
    if (address < root_address) return false;
    root_offset = static_cast<std::uint64_t>(address - root_address);
    if (root_offset > placed->bytes || bytes > placed->bytes - root_offset ||
        __builtin_add_overflow(placed->offset, root_offset, &domain_offset))
      return false;
  } else {
    const auto* binding = BindingForPhysicalRoot(lowered, relation, physical->root, physical->part);
    if (!binding || binding->base == 0 || binding->allocation_bytes == 0 ||
        binding->offset > binding->allocation_bytes ||
        binding->bytes > binding->allocation_bytes - binding->offset)
      return false;
    std::uintptr_t bound_begin = 0;
    if (__builtin_add_overflow(binding->base, static_cast<std::uintptr_t>(binding->offset),
                               &bound_begin) ||
        address < bound_begin)
      return false;
    root_offset = static_cast<std::uint64_t>(address - bound_begin);
    if (root_offset > binding->bytes || bytes > binding->bytes - root_offset ||
        __builtin_add_overflow(binding->offset, root_offset, &domain_offset))
      return false;
    domain = ComparisonRegionAccessDomain::kExternal;
    allocation_class = binding->allocation_class;
  }

  const auto role = relation.planned_tensors[physical->root.v].role;
  if (spec.written && (role == ir::Role::kWeight || role == ir::Role::kConstant ||
                       (role == ir::Role::kState && !spec.authenticated_state_write)))
    return false;
  if (spec.written) {
    for (const auto& binding : lowered.comparison_bindings) {
      if (!binding.original_root.valid() ||
          binding.original_root.v >= relation.original_to_planned.size())
        return false;
      const auto root = relation.original_to_planned[binding.original_root.v];
      if (!root.valid() || root.v >= relation.planned_tensors.size()) return false;
      const auto bound_role = relation.planned_tensors[root.v].role;
      if (bound_role == ir::Role::kState && spec.authenticated_state_write &&
          root == physical->root)
        continue;
      if (bound_role != ir::Role::kWeight && bound_role != ir::Role::kConstant &&
          bound_role != ir::Role::kState)
        continue;
      std::uintptr_t immutable_begin = 0, immutable_end = 0;
      if (binding.base == 0 ||
          __builtin_add_overflow(binding.base, static_cast<std::uintptr_t>(binding.offset),
                                 &immutable_begin) ||
          __builtin_add_overflow(immutable_begin, static_cast<std::uintptr_t>(binding.bytes),
                                 &immutable_end))
        return false;
      if (address < immutable_end && immutable_begin < access_end) return false;
    }
    for (const auto& input : relation.input_facts) {
      if (!input.token_input || (input.token_kind && ReadyTokenValueKind(*input.token_kind)) ||
          (input.unstored_verified &&
           input.unstored_reason ==
               components::ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks))
        continue;
      if (!input.original_tensor.valid() ||
          std::ranges::count(relation.input_facts, input.original_tensor,
                             &ComparisonInputFact::original_tensor) != 1)
        return false;
      const auto mapped = input.original_tensor.v < relation.original_to_planned.size()
                              ? relation.original_to_planned[input.original_tensor.v]
                              : ir::TensorId{};
      if (!mapped.valid() || mapped.v >= relation.planned_tensors.size() ||
          relation.planned_tensors[mapped.v].role != ir::Role::kInput)
        return false;
      const auto binding = std::ranges::find(lowered.comparison_bindings, input.original_tensor,
                                             &ComparisonBindingRange::original_root);
      if (binding == lowered.comparison_bindings.end() ||
          std::ranges::count(lowered.comparison_bindings, input.original_tensor,
                             &ComparisonBindingRange::original_root) != 1 ||
          binding->base == 0 || binding->bytes == 0 ||
          binding->offset > binding->allocation_bytes ||
          binding->bytes > binding->allocation_bytes - binding->offset)
        return false;
      std::uintptr_t protected_begin = 0;
      if (__builtin_add_overflow(binding->base, static_cast<std::uintptr_t>(binding->offset),
                                 &protected_begin))
        return false;
      const auto overlap = IntervalsOverlap(address, bytes, protected_begin, binding->bytes);
      if (!overlap || *overlap) return false;
    }
  }

  result.address = address;
  result.physical_id = descriptor->id;
  result.physical_local = local_physical;
  result.tensor_local = local_tensor;
  result.root_local = local_root;
  result.allocation_class = allocation_class;
  result.bytes = bytes;
  result.required_alignment = spec.alignment;
  result.root_offset = root_offset;
  result.domain_offset = domain_offset;
  result.domain = domain;
  result.read = spec.read;
  result.written = spec.written;
  return true;
}

bool IsValueRole(ir::Role role) {
  return role == ir::Role::kInput || role == ir::Role::kActivation || role == ir::Role::kOutput;
}

bool ReadyTokenValueKind(components::InputKind kind) {
  switch (kind) {
    case components::InputKind::kStream:
    case components::InputKind::kTokens:
    case components::InputKind::kPositions:
    case components::InputKind::kOutputRows:
    case components::InputKind::kFeatures:
    case components::InputKind::kSideRows:
    case components::InputKind::kResidual:
    case components::InputKind::kCarry:
    case components::InputKind::kSide:
      return true;
    case components::InputKind::kStoreCells:
    case components::InputKind::kCellPositions:
    case components::InputKind::kOutputCells:
    case components::InputKind::kVisibleCells:
    case components::InputKind::kBlockPositions:
    case components::InputKind::kCommitRows:
    case components::InputKind::kHostComputed:
    case components::InputKind::kHadamard:
      return false;
  }
  return false;
}

bool ProductionRegionEffectsClosed(const LoweredStep& lowered,
                                   const components::OwnedComparisonRegionBoundary& boundary,
                                   const LoweredStep::ComparisonOrigins& relation) {
  // A direct source input is a ready crossing only when the original token
  // contract says the caller stages ordinary values. Missing classifications
  // stay unknown; absence from LoweredStep::inputs is not evidence.
  for (const auto& port : boundary.ports()) {
    if (port.kind != components::ComparisonRegionPortKind::kInput) continue;
    const auto* root = RegionRoot(boundary, port.canonical_root);
    if (!root || !root->original_root.valid()) return false;
    const auto input = std::ranges::find(relation.input_facts, root->original_root,
                                         &ComparisonInputFact::original_tensor);
    if (input == relation.input_facts.end()) {
      // A boundary crossing may be produced inside the source graph. Prove
      // that fact through the retained original-to-planned map and the actual
      // producer's complete source-origin set; absence from the external
      // input inventory alone is not evidence.
      if (root->original_root.v >= relation.original_to_planned.size()) return false;
      const auto mapped = relation.original_to_planned[root->original_root.v];
      if (!mapped.valid() || mapped.v >= relation.planned_tensors.size()) return false;
      const auto& descriptor = relation.planned_tensors[mapped.v];
      if (descriptor.role == ir::Role::kWeight || descriptor.role == ir::Role::kConstant ||
          descriptor.role == ir::Role::kState) {
        bool authenticated = false;
        for (const auto& binding : relation.original_bindings)
          if (binding.original_root == root->original_root) authenticated = true;
        if (!authenticated) return false;
        continue;
      }
      if (descriptor.role != ir::Role::kActivation && descriptor.role != ir::Role::kOutput)
        return false;
      std::optional<std::uint32_t> producer;
      for (std::uint32_t i = 0; i < relation.planned_nodes.size(); ++i) {
        const auto outputs = relation.planned_nodes[i].outputs();
        if (std::ranges::find(outputs, mapped) == outputs.end()) continue;
        if (producer) return false;
        producer = i;
      }
      if (!producer || *producer + 1 >= relation.offsets.size() ||
          relation.offsets[*producer] == relation.offsets[*producer + 1])
        return false;
      for (auto origin = relation.offsets[*producer]; origin < relation.offsets[*producer + 1];
           ++origin) {
        const auto node = relation.source_nodes[static_cast<std::size_t>(origin)];
        if (!node.valid() || node.v >= relation.membership.size()) return false;
      }
      continue;
    }
    if (std::ranges::count(relation.input_facts, root->original_root,
                           &ComparisonInputFact::original_tensor) != 1)
      return false;
    if (input->token_input && (!input->token_kind || !ReadyTokenValueKind(*input->token_kind)))
      return false;
  }

  // The typed body access inventory is the authority for aliases. Every
  // state interval is protected even for reads; a body Value or private
  // scratch extent may not borrow state, immutable storage, generated input
  // storage, or a retained save buffer by address alias.
  for (const auto& state : boundary.state()) {
    const auto* root = RegionRoot(boundary, state.canonical_root);
    if (!root || !root->original_root.valid() || root->role != ir::Role::kState) return false;
    bool found = false;
    for (const auto& binding : lowered.comparison_bindings) {
      if (binding.original_root != root->original_root) continue;
      std::uintptr_t begin = 0;
      if (binding.base == 0 || binding.bytes == 0 || binding.part.has_value() ||
          binding.offset > binding.allocation_bytes ||
          binding.bytes > binding.allocation_bytes - binding.offset ||
          __builtin_add_overflow(static_cast<std::uintptr_t>(binding.base),
                                 static_cast<std::uintptr_t>(binding.offset), &begin))
        return false;
      if (!ValidAddressRange(begin, binding.bytes)) return false;
      found = true;
      for (const auto& access : relation.accesses) {
        const auto overlap = IntervalsOverlap(access.address, access.bytes, begin, binding.bytes);
        if (!overlap || *overlap) return false;
      }
    }
    if (!found) return false;
  }
  for (const auto& access : relation.accesses) {
    bool value_or_scratch = access.domain == ComparisonRegionAccessDomain::kPool;
    if (access.domain != ComparisonRegionAccessDomain::kPool) {
      const auto root = std::ranges::find(relation.tensor_local, access.root_local);
      if (access.root_local == kUnset || root == relation.tensor_local.end()) return false;
      const auto root_index = static_cast<std::size_t>(root - relation.tensor_local.begin());
      if (root_index >= relation.planned_tensors.size()) return false;
      value_or_scratch = IsValueRole(relation.planned_tensors[root_index].role);
    }
    for (const auto& binding : lowered.comparison_bindings) {
      if (!binding.original_root.valid() ||
          binding.original_root.v >= relation.original_to_planned.size())
        return false;
      const auto mapped = relation.original_to_planned[binding.original_root.v];
      if (!mapped.valid() || mapped.v >= relation.planned_tensors.size()) return false;
      const auto role = relation.planned_tensors[mapped.v].role;
      const auto input_fact = std::ranges::find(relation.input_facts, binding.original_root,
                                                &ComparisonInputFact::original_tensor);
      const bool protected_special_input =
          value_or_scratch && input_fact != relation.input_facts.end() && input_fact->token_input &&
          (!input_fact->token_kind || !ReadyTokenValueKind(*input_fact->token_kind)) &&
          !(input_fact->unstored_verified &&
            input_fact->unstored_reason ==
                components::ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks);
      const bool protected_state = role == ir::Role::kState;
      const bool protected_immutable =
          value_or_scratch && (role == ir::Role::kWeight || role == ir::Role::kConstant);
      if (!protected_state && !protected_immutable && !protected_special_input) continue;
      std::uintptr_t begin = 0;
      std::uintptr_t end = 0;
      if (binding.base == 0 || binding.offset > binding.allocation_bytes ||
          binding.bytes > binding.allocation_bytes - binding.offset ||
          __builtin_add_overflow(static_cast<std::uintptr_t>(binding.base),
                                 static_cast<std::uintptr_t>(binding.offset), &begin) ||
          binding.bytes == 0 ||
          __builtin_add_overflow(begin, static_cast<std::uintptr_t>(binding.bytes), &end))
        return false;
      if (!ValidAddressRange(begin, binding.bytes)) return false;
      const auto overlap = IntervalsOverlap(access.address, access.bytes, begin, binding.bytes);
      if (!overlap || *overlap) return false;
    }
    // Token-only StepInputs such as CommitRows may not appear in InputSlot or
    // SavedLeaf. Their retained original binding still protects the actual
    // address interval from local Value and private-scratch aliases.
    for (const auto& input : relation.input_facts) {
      if (!input.token_input || (input.token_kind && ReadyTokenValueKind(*input.token_kind)))
        continue;
      if (input.unstored_verified &&
          input.unstored_reason ==
              components::ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks)
        continue;
      if (!input.original_tensor.valid() ||
          std::ranges::count(relation.input_facts, input.original_tensor,
                             &ComparisonInputFact::original_tensor) != 1)
        return false;
      const auto mapped = input.original_tensor.v < relation.original_to_planned.size()
                              ? relation.original_to_planned[input.original_tensor.v]
                              : ir::TensorId{};
      if (!mapped.valid() || mapped.v >= relation.planned_tensors.size()) return false;
      const auto role = relation.planned_tensors[mapped.v].role;
      const auto matches = std::ranges::count_if(
          lowered.comparison_bindings, [&](const ComparisonBindingRange& binding) {
            return binding.original_root == input.original_tensor;
          });
      if (matches != 1) return false;
      const auto binding = std::ranges::find(lowered.comparison_bindings, input.original_tensor,
                                             &ComparisonBindingRange::original_root);
      if (binding == lowered.comparison_bindings.end() || role != ir::Role::kInput) return false;
      std::uintptr_t begin = 0;
      if (binding->base == 0 || binding->bytes == 0 ||
          binding->offset > binding->allocation_bytes ||
          binding->bytes > binding->allocation_bytes - binding->offset ||
          __builtin_add_overflow(static_cast<std::uintptr_t>(binding->base),
                                 static_cast<std::uintptr_t>(binding->offset), &begin) ||
          !ValidAddressRange(begin, binding->bytes))
        return false;
      const auto overlap = IntervalsOverlap(access.address, access.bytes, begin, binding->bytes);
      if (!overlap || *overlap) return false;
    }
    for (const auto& save : lowered.saves) {
      if (!save.tensor || !save.tensor->data) return false;
      const auto save_bytes = static_cast<std::uint64_t>(ggml_nbytes(save.tensor));
      const auto save_begin = reinterpret_cast<std::uintptr_t>(save.tensor->data);
      if (!ValidAddressRange(save_begin, save_bytes)) return false;
      const auto overlap = IntervalsOverlap(access.address, access.bytes, save_begin, save_bytes);
      if (!overlap || *overlap) return false;
    }
    if (lowered.generated_store) {
      const auto& slot = lowered.generated_store->slot;
      if (!ValidAddressRange(static_cast<std::uintptr_t>(slot.device_address), slot.bytes))
        return false;
      const auto overlap =
          IntervalsOverlap(access.address, access.bytes,
                           static_cast<std::uintptr_t>(slot.device_address), slot.bytes);
      if (!overlap || *overlap) return false;
    }
  }
  return true;
}

void HashRelativeDelta(LocalHash& hash, std::uint64_t left, std::uint64_t right) {
  hash.U8(left >= right ? 1 : 0);
  hash.U64(left >= right ? left - right : right - left);
}

std::optional<std::uint32_t> CanonicalActualComponent(
    std::span<const components::ComparisonRegionNode> boundary_nodes,
    const LoweredStep::ComparisonOrigins& relation, std::uint32_t planned_node) {
  if (relation.offsets.empty() || planned_node >= relation.offsets.size() - 1 ||
      planned_node >= relation.planned_nodes.size())
    return std::nullopt;
  const auto actual_component = relation.planned_nodes[planned_node].component;
  const auto first = relation.offsets[planned_node];
  const auto end = relation.offsets[planned_node + 1];
  if (first > end || end > relation.source_nodes.size()) return std::nullopt;
  for (auto offset = first; offset < end; ++offset) {
    const auto source = relation.source_nodes[static_cast<std::size_t>(offset)];
    const auto member =
        std::ranges::find(boundary_nodes, source, &components::ComparisonRegionNode::original_node);
    if (member == boundary_nodes.end() || member->component != actual_component) continue;
    const auto index = static_cast<std::size_t>(member - boundary_nodes.begin());
    const auto canonical = components::CanonicalComparisonComponent(boundary_nodes, index);
    if (canonical != UINT32_MAX) return canonical;
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::uint32_t> CanonicalLocalSegment(const LoweredStep::ComparisonOrigins& relation,
                                                   std::uint32_t local_tensor) {
  const auto current = std::ranges::find(relation.tensor_local, local_tensor);
  if (current == relation.tensor_local.end()) return std::nullopt;
  const auto current_index = static_cast<std::size_t>(current - relation.tensor_local.begin());
  if (current_index >= relation.planned_tensors.size()) return std::nullopt;
  const auto segment = relation.planned_tensors[current_index].segment;
  if (segment == ir::kJoinedSegment) return UINT32_MAX;
  for (std::uint32_t prior_local = 0; prior_local < local_tensor; ++prior_local) {
    const auto prior = std::ranges::find(relation.tensor_local, prior_local);
    if (prior == relation.tensor_local.end()) return std::nullopt;
    const auto prior_index = static_cast<std::size_t>(prior - relation.tensor_local.begin());
    if (prior_index >= relation.planned_tensors.size()) return std::nullopt;
    if (relation.planned_tensors[prior_index].segment == segment) return prior_local;
  }
  return local_tensor;
}

}  // namespace

std::uint64_t ComparisonRegionDefaultAccessAlignment(ggml_type type) {
  return static_cast<std::uint64_t>(ggml_type_size(type));
}

namespace {

bool NonzeroDigest(const base::Sha256Digest& value) {
  return std::ranges::any_of(value, [](std::uint8_t byte) { return byte != 0; });
}

bool SameOrigins(const LoweredStep::ComparisonOrigins& a, ir::NodeId a_node,
                 const LoweredStep::ComparisonOrigins& b, ir::NodeId b_node) {
  if (!a_node.valid() || !b_node.valid() || a_node.v + 1 >= a.offsets.size() ||
      b_node.v + 1 >= b.offsets.size())
    return false;
  const auto a_first = a.offsets[a_node.v], a_end = a.offsets[a_node.v + 1];
  const auto b_first = b.offsets[b_node.v], b_end = b.offsets[b_node.v + 1];
  if (a_end < a_first || b_end < b_first || a_end - a_first != b_end - b_first ||
      a_end > a.source_nodes.size() || b_end > b.source_nodes.size())
    return false;
  if (a_end == a_first) return false;
  return std::ranges::equal(std::span(a.source_nodes).subspan(a_first, a_end - a_first),
                            std::span(b.source_nodes).subspan(b_first, b_end - b_first));
}

bool SameMappedTensor(const LoweredStep& a, const PhysicalTensor& ap,
                      const LoweredStep::ComparisonOrigins& ar, const LoweredStep& b,
                      const PhysicalTensor& bp, const LoweredStep::ComparisonOrigins& br) {
  if (ap.tensor.valid() != bp.tensor.valid() || ap.root.valid() != bp.root.valid()) return false;
  const auto same_original_map = [](ir::TensorId a_id, const auto& a_relation, ir::TensorId b_id,
                                    const auto& b_relation) {
    if (!a_id.valid() || !b_id.valid() || a_id.v >= a_relation.planned_tensors.size() ||
        b_id.v >= b_relation.planned_tensors.size())
      return false;
    const auto& at = a_relation.planned_tensors[a_id.v];
    const auto& bt = b_relation.planned_tensors[b_id.v];
    if (at.role != bt.role || at.format != bt.format || at.nb != bt.nb ||
        at.view_offset != bt.view_offset || at.segment != bt.segment || at.binding != bt.binding)
      return false;
    std::size_t a_count = 0, b_count = 0;
    for (std::size_t i = 0; i < a_relation.original_to_planned.size(); ++i)
      if (a_relation.original_to_planned[i] == a_id) {
        if (i >= b_relation.original_to_planned.size() || b_relation.original_to_planned[i] != b_id)
          return false;
        ++a_count;
      }
    for (const auto mapped : b_relation.original_to_planned)
      if (mapped == b_id) ++b_count;
    return a_count != 0 && a_count == b_count;
  };
  if (ap.tensor.valid() && !same_original_map(ap.tensor, ar, bp.tensor, br)) return false;
  if (ap.root.valid() && !same_original_map(ap.root, ar, bp.root, br)) return false;
  if (ap.producer.valid() != bp.producer.valid()) return false;
  if (ap.producer.valid() && !SameOrigins(ar, ap.producer, br, bp.producer)) return false;
  if (ap.part != bp.part || ap.ir_format != bp.ir_format) return false;
  (void)a;
  (void)b;
  return true;
}

const ComparisonRegionValuePort* BoundaryPortForActualTensor(const LoweredStep& lowered,
                                                             const ComparisonRegionAnalysis& region,
                                                             const ggml_tensor* tensor) {
  const auto* physical = PhysicalTensorOf(lowered, tensor);
  if (!physical || !physical->tensor.valid() || !lowered.comparison_origins) return nullptr;
  const auto& relation = *lowered.comparison_origins;
  for (const auto& port : region.value_ports) {
    if (!port.source_tensor.valid() || port.source_tensor.v >= relation.original_to_planned.size())
      continue;
    if (relation.original_to_planned[port.source_tensor.v] == physical->tensor) return &port;
  }
  return nullptr;
}

bool SameActualTensor(const LoweredStep& a, const ComparisonRegionAnalysis& a_region,
                      const ggml_tensor* at, const LoweredStep::ComparisonOrigins& ar,
                      const LoweredStep& b, const ComparisonRegionAnalysis& b_region,
                      const ggml_tensor* bt, const LoweredStep::ComparisonOrigins& br,
                      bool source_edge = false) {
  const auto* a_port = source_edge ? BoundaryPortForActualTensor(a, a_region, at) : nullptr;
  const auto* b_port = source_edge ? BoundaryPortForActualTensor(b, b_region, bt) : nullptr;
  if (a_port || b_port) {
    // A closed crossing is a logical cut. Do not recurse into a different
    // internal value or implementation on the far side of that cut.
    const bool same =
        a_port && b_port && a_port->kind == b_port->kind && a_port->ordinal == b_port->ordinal &&
        a_port->source_tensor == b_port->source_tensor && a_port->role == b_port->role &&
        a_port->format == b_port->format && a_port->bytes == b_port->bytes &&
        a_port->root_offset == b_port->root_offset && a_port->read == b_port->read &&
        a_port->written == b_port->written;
    return same;
  }
  if (!at || !bt || at->op != bt->op || at->type != bt->type || at->view_offs != bt->view_offs)
    return false;
  for (int i = 0; i < GGML_MAX_DIMS; ++i)
    if (at->ne[i] != bt->ne[i] || at->nb[i] != bt->nb[i]) return false;
  if (at->op == GGML_OP_CUSTOM && bt->op == GGML_OP_CUSTOM && LlmpOpOf(at) == LlmpOp::kCausalMask &&
      LlmpOpOf(bt) == LlmpOp::kCausalMask) {
    if (!CheckCausalMask(at) || !CheckCausalMask(bt)) return false;
    // CheckCausalMask validates the complete bounded kernel contract, including
    // the reserved word. Compare only its declared eight-word parameter record.
    for (int field = 0; field < 8; ++field)
      if (LlmpOpInt(at, field) != LlmpOpInt(bt, field)) return false;
  } else if (at->op == GGML_OP_PERMUTE && bt->op == GGML_OP_PERMUTE) {
    std::array<bool, GGML_MAX_DIMS> axes{};
    for (std::size_t axis = 0; axis < GGML_MAX_DIMS; ++axis) {
      const auto a_axis = at->op_params[axis];
      const auto b_axis = bt->op_params[axis];
      if (a_axis < 0 || a_axis >= GGML_MAX_DIMS || a_axis != b_axis) return false;
      const auto axis_index = static_cast<std::size_t>(a_axis);
      if (axes[axis_index]) return false;
      axes[axis_index] = true;
    }
    if (!std::ranges::all_of(std::span(at->op_params).subspan(GGML_MAX_DIMS),
                             [](std::int32_t word) { return word == 0; }) ||
        !std::ranges::all_of(std::span(bt->op_params).subspan(GGML_MAX_DIMS),
                             [](std::int32_t word) { return word == 0; }))
      return false;
  } else if (HasTypedDescriptorSchema(*at) && HasTypedDescriptorSchema(*bt)) {
    LocalHash a_params("llmp.comparison.remainder.typed-params.v1");
    LocalHash b_params("llmp.comparison.remainder.typed-params.v1");
    HashTypedOperationParameters(a_params, *at);
    HashTypedOperationParameters(b_params, *bt);
    if (a_params.Finish() != b_params.Finish()) return false;
  } else if (at->op == GGML_OP_FLASH_ATTN_EXT && bt->op == at->op) {
    // CheckFlashAttnVec validates this pinned op's complete parameter record;
    // byte equality then compares its checked typed fields exactly.
    if (std::memcmp(at->op_params, bt->op_params, sizeof(at->op_params)) != 0) return false;
  } else if (at->op == GGML_OP_GET_ROWS && bt->op == at->op) {
    // GetRows has no parameter payload; its checked descriptor and actual
    // vectorized-versus-scalar address predicates are the complete dispatch
    // facts for this finite outside-call schema.
    if (!CheckGetRows(at) || !CheckGetRows(bt) || GetRowsVectorized(at) != GetRowsVectorized(bt) ||
        reinterpret_cast<std::uintptr_t>(at->src[0]->data) % 16 !=
            reinterpret_cast<std::uintptr_t>(bt->src[0]->data) % 16 ||
        reinterpret_cast<std::uintptr_t>(at->data) % 16 !=
            reinterpret_cast<std::uintptr_t>(bt->data) % 16)
      return false;
  } else {
    return false;
  }
  for (int i = 0; i < GGML_MAX_SRC; ++i) {
    const auto* as = at->src[i];
    const auto* bs = bt->src[i];
    if ((as == nullptr) != (bs == nullptr)) return false;
    if (!as) continue;
    const auto* ap = PhysicalTensorOf(a, as);
    const auto* bp = PhysicalTensorOf(b, bs);
    const bool a_cut = BoundaryPortForActualTensor(a, a_region, as) != nullptr;
    const bool b_cut = BoundaryPortForActualTensor(b, b_region, bs) != nullptr;
    if (!ap || !bp) return false;
    if (!SameActualTensor(a, a_region, as, ar, b, b_region, bs, br, true)) return false;
    if (a_cut != b_cut || (!a_cut && !SameMappedTensor(a, *ap, ar, b, *bp, br))) return false;
  }
  if ((at->view_src == nullptr) != (bt->view_src == nullptr)) return false;
  if (at->view_src) {
    // A metadata view may sit directly on a changed local result. Compare
    // that storage edge through the authenticated crossing port; comparing
    // its producer origins would look through the cut into each arm's body.
    if (!SameActualTensor(a, a_region, at->view_src, ar, b, b_region, bt->view_src, br, true))
      return false;
  }
  const auto* ap = PhysicalTensorOf(a, at);
  const auto* bp = PhysicalTensorOf(b, bt);
  if (!ap || !bp || !SameMappedTensor(a, *ap, ar, b, *bp, br)) return false;
  return true;
}

bool SameSavedLeaf(const LoweredStep& a, const SavedLeaf& x,
                   const ComparisonRegionAnalysis& a_region, const LoweredStep& b,
                   const SavedLeaf& y, const ComparisonRegionAnalysis& b_region) {
  if (!SavedMappingCurrent(a, x) || !SavedMappingCurrent(b, y) || !SavedOwnerRangeCurrent(x) ||
      !SavedOwnerRangeCurrent(y) || x.original_source != y.original_source ||
      x.original_rows != y.original_rows || x.layer != y.layer || x.component != y.component ||
      x.segment != y.segment || x.what != y.what || !x.tensor || !y.tensor ||
      *x.owner_range != *y.owner_range || x.tensor->type != y.tensor->type ||
      ggml_nbytes(x.tensor) != ggml_nbytes(y.tensor) || !x.packed || !y.packed ||
      ((x.packed->op == GGML_OP_CONT) != (y.packed->op == GGML_OP_CONT)))
    return false;
  for (int i = 0; i < GGML_MAX_DIMS; ++i)
    if (x.tensor->ne[i] != y.tensor->ne[i] || x.tensor->nb[i] != y.tensor->nb[i]) return false;
  const auto descriptor_for = [](const LoweredStep& lowered,
                                 ir::TensorId id) -> const ggml_tensor* {
    const auto found = std::ranges::find(lowered.physical_tensors, id, &PhysicalTensor::tensor);
    if (found == lowered.physical_tensors.end() || !found->descriptor ||
        std::ranges::count(lowered.physical_tensors, id, &PhysicalTensor::tensor) != 1)
      return nullptr;
    return found->descriptor;
  };
  const auto* ax = descriptor_for(a, x.source);
  const auto* bx = descriptor_for(b, y.source);
  const auto* ar = descriptor_for(a, x.rows);
  const auto* br = descriptor_for(b, y.rows);
  if (!ax || !bx || !ar || !br ||
      !SameActualTensor(a, a_region, ax, *a.comparison_origins, b, b_region, bx,
                        *b.comparison_origins, true) ||
      !SameActualTensor(a, a_region, ar, *a.comparison_origins, b, b_region, br,
                        *b.comparison_origins, true))
    return false;
  if (x.packed->type != y.packed->type) return false;
  for (int i = 0; i < GGML_MAX_DIMS; ++i)
    if (x.packed->ne[i] != y.packed->ne[i] || x.packed->nb[i] != y.packed->nb[i]) return false;
  return true;
}

bool ExistingExplicitRemainderRecipe(const PlanStep& step) {
  return step.implementation == kRmsNormMulFused || step.implementation == kRmsNormMulUnfused ||
         step.implementation == kAddName || step.implementation == kMulName ||
         step.implementation == kRmsNormName || step.implementation == kScaleName;
}

bool SupportedRemainderCall(const PlanStep& step, const execution::Registry& registry,
                            const ComparisonRemainderCallFacts& facts) {
  if (step.lane != 0 || step.nodes.empty() || step.nodes.size() > kSetRowsGroupMax ||
      !registry.Find(step.implementation))
    return false;
  if (ExistingExplicitRemainderRecipe(step))
    return DescribeComparisonAccess(step, ComparisonAccessScope::kWhole).has_value();
  if (ComparisonSgemmCall(step)) return CheckComparisonSgemm(step, facts);
  if (step.implementation == kGetRowsName && step.nodes.size() == 1 &&
      step.operation == ir::Primitive(ir::Op::kGetRows)) {
    const auto* node = step.nodes.front();
    return CheckGetRows(node).has_value() && node->src[0] && node->src[0]->data && node->data;
  }
  if (step.implementation == kCausalMaskName && step.nodes.size() == 1 &&
      step.operation == ir::Primitive(ir::Op::kMaskBuild))
    return CheckCausalMask(step.nodes.front()).has_value();
  if (step.implementation == kMulMatVector && step.nodes.size() == 1 &&
      facts.has_mul_mat_selection && facts.mul_mat_vector &&
      step.operation == ir::Primitive(ir::Op::kMatMul)) {
    const auto* node = step.nodes.front();
    return CheckMulMat(node).has_value();
  }
  if (step.implementation == kRopeName && step.nodes.size() == 1 &&
      step.operation == ir::Primitive(ir::Op::kRope))
    return CheckRope(step.nodes.front()).has_value();
  if (step.implementation == kSetRowsName && step.nodes.size() == 1 &&
      step.operation == ir::Primitive(ir::Op::kCacheStore))
    return CheckSetRows(step.nodes.front()).has_value();
  if (step.implementation == kSetRowsGroupedName &&
      step.operation == ir::Fused(ir::FusedKind::kGroupedStores) && step.nodes.size() >= 2 &&
      step.nodes.size() <= kSetRowsGroupMax) {
    std::array<const ggml_tensor*, kSetRowsGroupMax> nodes{};
    for (std::size_t i = 0; i < step.nodes.size(); ++i) nodes[i] = step.nodes[i];
    return CheckSetRowsGroup(std::span<const ggml_tensor* const>(nodes).first(step.nodes.size()))
        .has_value();
  }
  if (step.implementation == kRopeSetRowsFused && step.nodes.size() == 2 &&
      step.operation == ir::Fused(ir::FusedKind::kRopeStore))
    return CheckRopeSetRows(step.nodes[0], step.nodes[1]).has_value();
  if (step.implementation == kContName && step.nodes.size() == 1 &&
      step.operation == ir::Primitive(ir::Op::kCont))
    return CheckCont(step.nodes.front()).has_value();
  if (step.implementation == "ggml.flash_attn_ext.vec" && step.nodes.size() == 1 &&
      facts.has_attention_plan && step.operation == ir::Primitive(ir::Op::kAttention)) {
    return CheckFlashAttnVec(step.nodes.front()).has_value();
  }
  return false;
}

bool DescribeRemainderAccesses(const PlanStep& step, const ComparisonRemainderCallFacts& facts,
                               RegionAccessList& accesses) {
  if (ExistingExplicitRemainderRecipe(step)) {
    const auto described = DescribeComparisonAccess(step, ComparisonAccessScope::kWhole);
    if (!described) return false;
    for (const auto& fact : std::span(described->values).first(described->count)) {
      if (!AddRegionAccess(accesses, fact.tensor, fact.read, fact.written, fact.bytes,
                           fact.alignment))
        return false;
      accesses.values[accesses.size - 1].phase = fact.phase;
    }
    return true;
  }
  if (step.nodes.size() == 1 && step.nodes.front()) {
    const auto* node = step.nodes.front();
    if (step.implementation == kGetRowsName && step.operation == ir::Primitive(ir::Op::kGetRows) &&
        CheckGetRows(node))
      return AddRegionAccess(accesses, node->src[0], true, false) &&
             AddRegionAccess(accesses, node->src[1], true, false) &&
             AddRegionAccess(accesses, node, false, true);
    if (step.implementation == kCausalMaskName &&
        step.operation == ir::Primitive(ir::Op::kMaskBuild) && CheckCausalMask(node))
      return AddRegionAccess(accesses, node->src[0], true, false) &&
             AddRegionAccess(accesses, node, false, true);
    if (ComparisonSgemmCall(step) && CheckComparisonSgemm(step, facts))
      return AddRegionAccess(accesses, node->src[0], true, false, 0, 4) &&
             AddRegionAccess(accesses, node->src[1], true, false, 0, 4) &&
             AddRegionAccess(accesses, node, false, true, 0, 4);
    if (step.implementation == kMulMatVector && facts.has_mul_mat_selection &&
        facts.mul_mat_vector && step.operation == ir::Primitive(ir::Op::kMatMul) &&
        CheckMulMat(node))
      return AddRegionAccess(accesses, node->src[0], true, false) &&
             AddRegionAccess(accesses, node->src[1], true, false) &&
             AddRegionAccess(accesses, node, false, true);
    if (step.implementation == kRopeName && step.operation == ir::Primitive(ir::Op::kRope) &&
        CheckRope(node)) {
      for (const auto* source : node->src)
        if (source && !AddRegionAccess(accesses, source, true, false)) return false;
      return AddRegionAccess(accesses, node, false, true);
    }
    if (step.implementation == kSetRowsName &&
        step.operation == ir::Primitive(ir::Op::kCacheStore) && CheckSetRows(node))
      return AddRegionAccess(accesses, node->src[0], true, false) &&
             AddRegionAccess(accesses, node->src[1], true, false) &&
             AddRegionAccess(accesses, node->src[2], false, true, 0, 0, true);
    if (step.implementation == kContName && step.operation == ir::Primitive(ir::Op::kCont) &&
        CheckCont(node))
      return AddRegionAccess(accesses, node->src[0], true, false) &&
             AddRegionAccess(accesses, node, false, true);
    if (step.implementation == "ggml.flash_attn_ext.vec" && facts.has_attention_plan &&
        step.operation == ir::Primitive(ir::Op::kAttention) && CheckFlashAttnVec(node)) {
      if (facts.scratch_bytes != facts.attention_scratch_bytes) return false;
      for (const auto* source : node->src)
        if (source && !AddRegionAccess(accesses, source, true, false)) return false;
      if (facts.scratch_bytes != 0 &&
          !AddRegionPoolAccess(accesses, facts.scratch_bytes, true, true))
        return false;
      return AddRegionAccess(accesses, node, false, true);
    }
  }
  if (step.implementation == kSetRowsGroupedName &&
      step.operation == ir::Fused(ir::FusedKind::kGroupedStores) && step.nodes.size() >= 2 &&
      step.nodes.size() <= kSetRowsGroupMax) {
    std::array<const ggml_tensor*, kSetRowsGroupMax> nodes{};
    for (std::size_t i = 0; i < step.nodes.size(); ++i) nodes[i] = step.nodes[i];
    if (!CheckSetRowsGroup(std::span<const ggml_tensor* const>(nodes).first(step.nodes.size())))
      return false;
    for (const auto* node : step.nodes)
      if (!AddRegionAccess(accesses, node->src[0], true, false) ||
          !AddRegionAccess(accesses, node->src[1], true, false) ||
          !AddRegionAccess(accesses, node->src[2], false, true, 0, 0, true))
        return false;
    return true;
  }
  if (step.implementation == kRopeSetRowsFused && step.nodes.size() == 2 &&
      step.operation == ir::Fused(ir::FusedKind::kRopeStore) &&
      CheckRopeSetRows(step.nodes[0], step.nodes[1])) {
    const auto* rope = step.nodes[0];
    const auto* store = step.nodes[1];
    for (const auto* source : rope->src)
      if (source && !AddRegionAccess(accesses, source, true, false)) return false;
    return AddRegionAccess(accesses, store->src[1], true, false) &&
           AddRegionAccess(accesses, store->src[2], false, true, 0, 0, true);
  }
  return false;
}

bool SameCall(const LoweredStep& a, const execution::Registry& ar, std::size_t ai,
              const LaunchContext* ac, const LoweredStep& b, const execution::Registry& br,
              std::size_t bi, const LaunchContext* bc, const ComparisonRegionAnalysis& a_region,
              const ComparisonRegionAnalysis& b_region,
              ComparisonRemainderCallFactsFn describe_call_facts) {
  const auto mismatch = [](const char*) { return false; };
  if (!ac || !bc || !describe_call_facts) return mismatch("missing facts context");
  const auto& as = a.plan.steps[ai];
  const auto& bs = b.plan.steps[bi];
  const auto a_saved = FindSavedCall(a, as);
  const auto b_saved = FindSavedCall(b, bs);
  if (a_saved.leaf || b_saved.leaf) {
    if (!a_saved.leaf || !b_saved.leaf || a_saved.kind != b_saved.kind || as.lane != 0 ||
        bs.lane != 0 || as.nodes.size() != 1 || bs.nodes.size() != 1 ||
        as.operation != bs.operation || as.implementation != bs.implementation ||
        !SameSavedLeaf(a, *a_saved.leaf, a_region, b, *b_saved.leaf, b_region))
      return mismatch("authenticated saved helper identity or source mapping");
    const auto a_impl = ar.Find(as.implementation);
    const auto b_impl = br.Find(bs.implementation);
    if (!a_impl || !b_impl || ar.identity(*a_impl) != br.identity(*b_impl))
      return mismatch("saved helper implementation identity");
    if ((std::ranges::contains(a.prologue, ai)) != (std::ranges::contains(b.prologue, bi)))
      return mismatch("saved helper prologue membership");
    const auto same_descriptor = [](const ggml_tensor* x, const ggml_tensor* y) {
      if (!x || !y || x->op != y->op || x->type != y->type || x->view_offs != y->view_offs)
        return false;
      for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension)
        if (x->ne[dimension] != y->ne[dimension] || x->nb[dimension] != y->nb[dimension])
          return false;
      return true;
    };
    if (!same_descriptor(as.nodes.front(), bs.nodes.front()))
      return mismatch("saved helper physical descriptor");
    if (a_saved.kind == SavedCallKind::kStore &&
        (as.operation != ir::Primitive(ir::Op::kCacheStore) ||
         as.implementation != kSetRowsExtName || !CheckSetRowsExt(as.nodes.front()) ||
         !CheckSetRowsExt(bs.nodes.front()) ||
         !same_descriptor(as.nodes.front()->src[0], bs.nodes.front()->src[0])))
      return mismatch("saved F32 SET_ROWS_EXT recipe");
    if (a_saved.kind == SavedCallKind::kPackedCopy &&
        (as.operation != ir::Primitive(ir::Op::kCont) || as.implementation != kContName ||
         !CheckCont(as.nodes.front()) || !CheckCont(bs.nodes.front())))
      return mismatch("saved CONT recipe");
    return true;
  }
  ComparisonRemainderCallFacts a_facts{}, b_facts{};
  if (!describe_call_facts(*ac, as, a_facts) || !describe_call_facts(*bc, bs, b_facts))
    return mismatch("describe call facts");
  if (as.operation != bs.operation || as.implementation != bs.implementation ||
      as.nodes.size() != bs.nodes.size() || as.lane != bs.lane ||
      !SupportedRemainderCall(as, ar, a_facts) || !SupportedRemainderCall(bs, br, b_facts))
    return mismatch("call eligibility or schema");
  const auto a_impl = ar.Find(as.implementation);
  const auto b_impl = br.Find(bs.implementation);
  if (!a_impl || !b_impl || ar.identity(*a_impl) != br.identity(*b_impl))
    return mismatch("registry implementation identity");
  if (a_facts.scratch_bytes != b_facts.scratch_bytes || a_facts.pool_bytes != b_facts.pool_bytes ||
      a_facts.pool_lanes != b_facts.pool_lanes || a_facts.pool_faulted != b_facts.pool_faulted ||
      a_facts.has_mul_mat_selection != b_facts.has_mul_mat_selection ||
      a_facts.mul_mat_vector != b_facts.mul_mat_vector ||
      a_facts.has_attention_plan != b_facts.has_attention_plan ||
      a_facts.attention_columns_per_block != b_facts.attention_columns_per_block ||
      a_facts.attention_parallel_blocks != b_facts.attention_parallel_blocks ||
      a_facts.attention_scratch_bytes != b_facts.attention_scratch_bytes ||
      a_facts.attention_mask_prepass != b_facts.attention_mask_prepass)
    return mismatch("dispatch/scratch facts");
  if (a_facts.cublas_plan.has_value() != b_facts.cublas_plan.has_value())
    return mismatch("library call kind");
  if (a_facts.cublas_plan) {
    if (!CheckComparisonSgemm(as, a_facts) || !CheckComparisonSgemm(bs, b_facts) ||
        !ComparisonLibraryWorkspaceDisjoint(a, a_facts) ||
        !ComparisonLibraryWorkspaceDisjoint(b, b_facts))
      return mismatch("library workspace/plan");
    LocalHash ah("remainder.cublas"), bh("remainder.cublas");
    HashComparisonCublasFacts(ah, a_facts);
    HashComparisonCublasFacts(bh, b_facts);
    if (ah.Finish() != bh.Finish()) return mismatch("library configuration/plan");
  }
  if ((std::ranges::contains(a.prologue, ai)) != std::ranges::contains(b.prologue, bi))
    return mismatch("prologue membership");
  const auto& ao = *a.comparison_origins;
  const auto& bo = *b.comparison_origins;
  for (std::size_t i = 0; i < as.nodes.size(); ++i) {
    const auto* ap = PhysicalTensorOf(a, as.nodes[i]);
    const auto* bp = PhysicalTensorOf(b, bs.nodes[i]);
    if (!ap || !bp || !ap->producer.valid() || !bp->producer.valid() ||
        ap->producer.v >= ao.offsets.size() - 1 || bp->producer.v >= bo.offsets.size() - 1 ||
        ao.offsets[ap->producer.v] >= ao.offsets[ap->producer.v + 1] ||
        bo.offsets[bp->producer.v] >= bo.offsets[bp->producer.v + 1] ||
        !SameOrigins(ao, ap->producer, bo, bp->producer))
      return mismatch("node source origins");
    if (!SameActualTensor(a, a_region, as.nodes[i], ao, b, b_region, bs.nodes[i], bo))
      return mismatch("node descriptor or source/value lineage");
    if (as.implementation == kGetRowsName) {
      const auto* an = as.nodes[i];
      const auto* bn = bs.nodes[i];
      if (GetRowsVectorized(an) != GetRowsVectorized(bn) ||
          reinterpret_cast<std::uintptr_t>(an->src[0]->data) % 16 !=
              reinterpret_cast<std::uintptr_t>(bn->src[0]->data) % 16 ||
          reinterpret_cast<std::uintptr_t>(an->data) % 16 !=
              reinterpret_cast<std::uintptr_t>(bn->data) % 16)
        return mismatch("GetRows alignment facts");
    }
  }
  return true;
}

bool ResolveSavedCopyAccess(const LoweredStep& lowered, const SavedLeaf& save,
                            const ggml_tensor* accessed, bool read, bool written,
                            ComparisonRegionAccessFact& result) {
  if (!accessed || !save.packed || save.packed->op != GGML_OP_CONT || !save.scratch_bytes ||
      !HasExactFlattenedStorageView(save.store->src[0], save.packed))
    return false;
  const auto* physical = PhysicalTensorOf(lowered, save.packed);
  if (!physical || !physical->descriptor || physical->id >= lowered.physical_tensors.size() ||
      accessed->data != save.packed->data || !accessed->data)
    return false;
  const auto found = std::ranges::find_if(lowered.physical_roots, [&](const PhysicalRoot& root) {
    return root.backend_tensor == physical->id && !root.root.valid() && !root.producer.valid() &&
           root.operation == ir::Primitive(ir::Op::kCont) && root.component == save.component &&
           root.offset == save.scratch_offset && root.bytes == save.scratch_bytes;
  });
  if (found == lowered.physical_roots.end() ||
      std::ranges::count_if(
          lowered.physical_roots,
          [&](const PhysicalRoot& root) { return root.backend_tensor == physical->id; }) != 1 ||
      found->offset > lowered.extent || found->bytes > lowered.extent - found->offset ||
      found->bytes != ggml_nbytes(save.packed) ||
      reinterpret_cast<std::uintptr_t>(accessed->data) % 256 != 0)
    return false;
  for (std::size_t dimension = 0; dimension < static_cast<std::size_t>(GGML_MAX_DIMS); ++dimension)
    if (found->ne[dimension] != static_cast<std::uint64_t>(save.packed->ne[dimension]) ||
        found->nb[dimension] != save.packed->nb[dimension])
      return false;
  result.address = reinterpret_cast<std::uintptr_t>(accessed->data);
  result.physical_id = physical->id;
  result.bytes = static_cast<std::uint64_t>(ggml_nbytes(accessed));
  result.required_alignment = 256;
  result.domain = ComparisonRegionAccessDomain::kActivation;
  result.read = read;
  result.written = written;
  return result.bytes != 0 && ValidAddressRange(result.address, result.bytes);
}

bool ResolveSavedDestinationAccess(const SavedLeaf& save, ComparisonRegionAccessFact& result) {
  if (!SavedOwnerRangeCurrent(save)) return false;
  result.address = static_cast<std::uintptr_t>(save.owner_range->base);
  result.bytes = static_cast<std::uint64_t>(ggml_nbytes(save.tensor));
  result.required_alignment = save.owner_range->alignment;
  result.domain = ComparisonRegionAccessDomain::kExternal;
  result.written = true;
  result.saved_destination = true;
  return ValidAddressRange(result.address, result.bytes);
}

bool CollectSavedCallAccesses(
    const LoweredStep& lowered, const PlanStep& step, const SavedCall& saved,
    std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& facts, std::size_t& count) {
  const auto& save = *saved.leaf;
  if (!SavedMappingCurrent(lowered, save) || !SavedOwnerRangeCurrent(save)) return false;
  const auto resolve_ordinary = [&](const ggml_tensor* tensor, bool read, bool written,
                                    ComparisonRegionAccessFact& fact) {
    if (!tensor) return false;
    RegionAccessSpec spec{.tensor = tensor,
                          .read = read,
                          .written = written,
                          .alignment = ComparisonRegionDefaultAccessAlignment(tensor->type)};
    std::uint32_t ignored = 0;
    return ResolveRegionAccess(lowered, *lowered.comparison_origins, spec, false, ignored, fact);
  };
  count = 0;
  const auto append = [&](const ComparisonRegionAccessFact& fact) {
    if (count == facts.size()) return false;
    facts[count++] = fact;
    return true;
  };
  const auto* node = step.nodes.front();
  if (saved.kind == SavedCallKind::kPackedCopy) {
    if (step.operation != ir::Primitive(ir::Op::kCont) || step.implementation != kContName ||
        node != save.packed || !CheckCont(node))
      return false;
    ComparisonRegionAccessFact source{}, destination{};
    return resolve_ordinary(node->src[0], true, false, source) &&
           ResolveSavedCopyAccess(lowered, save, node, false, true, destination) &&
           append(source) && append(destination);
  }
  if (saved.kind == SavedCallKind::kStore) {
    if (step.operation != ir::Primitive(ir::Op::kCacheStore) ||
        step.implementation != kSetRowsExtName || node != save.store || !CheckSetRowsExt(node))
      return false;
    ComparisonRegionAccessFact values{}, rows{}, destination{};
    const auto source_record =
        std::ranges::find(lowered.physical_tensors, save.source, &PhysicalTensor::tensor);
    if (source_record == lowered.physical_tensors.end() || !source_record->descriptor) return false;
    const bool generated_copy = save.packed->op == GGML_OP_CONT &&
                                save.packed != source_record->descriptor && save.scratch_bytes != 0;
    // SET_ROWS src[0] is a generated reshape in either case. Resolve the
    // exact retained packed source or CONT, never that unmapped helper tensor.
    const bool values_ok =
        generated_copy ? ResolveSavedCopyAccess(lowered, save, save.packed, true, false, values)
                       : resolve_ordinary(source_record->descriptor, true, false, values);
    return values_ok && resolve_ordinary(node->src[1], true, false, rows) &&
           ResolveSavedDestinationAccess(save, destination) && append(values) && append(rows) &&
           append(destination);
  }
  // The reshape is a metadata-only view. It is part of the exact retained
  // SET_ROWS chain but has no independent device memory access.
  if (saved.kind == SavedCallKind::kValuesView && node == save.store->src[0]) return true;
  return false;
}

bool RemainderPoolDisjoint(const LoweredStep& lowered, const Q8MainPoolDescriptor& pool,
                           std::uint64_t bytes) {
  if (!ComparisonPoolDisjoint(lowered, pool, bytes)) return false;
  if (bytes == 0) return true;
  const auto begin = static_cast<std::uintptr_t>(pool.base);
  std::uintptr_t end = 0;
  if (begin == 0 || __builtin_add_overflow(begin, static_cast<std::uintptr_t>(bytes), &end))
    return false;
  for (const auto& save : lowered.saves) {
    if (!SavedOwnerRangeCurrent(save)) return false;
    const auto save_begin = static_cast<std::uintptr_t>(save.owner_range->base);
    std::uintptr_t save_end = 0;
    if (__builtin_add_overflow(save_begin, static_cast<std::uintptr_t>(save.owner_range->bytes),
                               &save_end) ||
        (begin < save_end && save_begin < end))
      return false;
  }
  return true;
}

bool CollectCallAccesses(const LoweredStep& lowered, std::size_t call, const LaunchContext* context,
                         ComparisonRemainderCallFactsFn describe_facts, bool region_call,
                         RegionAccessList& specs,
                         std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& facts,
                         std::size_t& count) {
  if (call >= lowered.plan.steps.size()) return false;
  const auto& step = lowered.plan.steps[call];
  const auto saved = FindSavedCall(lowered, step);
  if (saved.leaf) {
    if (region_call) return false;
    count = 0;
    if (!CollectSavedCallAccesses(lowered, step, saved, facts, count)) return false;
    for (std::size_t i = 0; i < count; ++i)
      facts[i].call_ordinal = static_cast<std::uint32_t>(call);
    return true;
  }
  ComparisonRemainderCallFacts call_facts{};
  if (region_call) {
    if (!DescribeRegionAccesses(step, specs)) {
      return false;
    }
  } else {
    if (!context || !describe_facts || !describe_facts(*context, step, call_facts)) {
      return false;
    }
    if (!DescribeRemainderAccesses(step, call_facts, specs)) {
      return false;
    }
  }
  if (specs.size > facts.size()) {
    return false;
  }
  std::optional<Q8MainPoolDescriptor> pool;
  if (call_facts.cublas_plan && !ComparisonLibraryWorkspaceDisjoint(lowered, call_facts))
    return false;
  if (call_facts.scratch_bytes != 0) {
    pool = Q8MainPoolDescriptor{.base = call_facts.pool_base,
                                .bytes = call_facts.pool_bytes,
                                .lanes = call_facts.pool_lanes,
                                .faulted = call_facts.pool_faulted};
    if (!context || !RemainderPoolDisjoint(lowered, *pool, call_facts.scratch_bytes)) {
      return false;
    }
  }
  std::uint32_t ignored_physical_count = 0;
  for (std::size_t i = 0; i < specs.size; ++i) {
    if (!ResolveRegionAccess(lowered, *lowered.comparison_origins, specs.values[i], false,
                             ignored_physical_count, facts[i], pool ? &*pool : nullptr)) {
      return false;
    }
    facts[i].call_ordinal = static_cast<std::uint32_t>(call);
    facts[i].subcall_ordinal = specs.values[i].phase;
  }
  count = specs.size;
  return true;
}

bool SameAccessDescription(const ComparisonRegionAccessFact& a,
                           const ComparisonRegionAccessFact& b) {
  return a.bytes == b.bytes && a.required_alignment == b.required_alignment &&
         a.domain == b.domain && a.read == b.read && a.written == b.written &&
         a.subcall_ordinal == b.subcall_ordinal && a.saved_destination == b.saved_destination;
}

bool SameOverlapRelation(std::uintptr_t a0, std::uint64_t an, std::uintptr_t b0, std::uint64_t bn,
                         std::uintptr_t c0, std::uint64_t cn, std::uintptr_t d0, std::uint64_t dn) {
  const auto left = IntervalsOverlap(a0, an, b0, bn);
  const auto right = IntervalsOverlap(c0, cn, d0, dn);
  if (!left || !right || *left != *right) return false;
  if (!*left) return true;
  const bool left_after = a0 >= b0;
  const bool right_after = c0 >= d0;
  const auto left_delta = left_after ? a0 - b0 : b0 - a0;
  const auto right_delta = right_after ? c0 - d0 : d0 - c0;
  return left_after == right_after && left_delta == right_delta;
}

template <std::size_t N>
bool SameWithinCallTopology(const std::array<ComparisonRegionAccessFact, N>& a, std::size_t a_count,
                            const std::array<ComparisonRegionAccessFact, N>& b,
                            std::size_t b_count) {
  if (a_count != b_count) return false;
  for (std::size_t i = 0; i < a_count; ++i) {
    if (!SameAccessDescription(a[i], b[i])) return false;
    for (std::size_t j = 0; j < i; ++j)
      if (!SameOverlapRelation(a[i].address, a[i].bytes, a[j].address, a[j].bytes, b[i].address,
                               b[i].bytes, b[j].address, b[j].bytes))
        return false;
  }
  return true;
}

bool SamePortTopology(const ComparisonRegionAnalysis& a, const ComparisonRegionAnalysis& b) {
  if (a.value_ports.size() != b.value_ports.size()) return false;
  for (std::size_t i = 0; i < a.value_ports.size(); ++i) {
    const auto& x = a.value_ports[i];
    const auto& y = b.value_ports[i];
    if (x.kind != y.kind || x.ordinal != y.ordinal || x.source_tensor != y.source_tensor ||
        x.role != y.role || x.format != y.format || x.bytes != y.bytes ||
        x.root_offset != y.root_offset || x.read != y.read || x.written != y.written ||
        x.address == 0 || y.address == 0)
      return false;
  }
  return true;
}

bool OutsideCallAt(const LoweredStep& lowered, const ComparisonRegionAnalysis& region,
                   std::size_t ordinal, std::size_t& call) {
  std::size_t seen = 0;
  for (std::size_t i = 0; i < lowered.plan.steps.size(); ++i) {
    if (i >= region.first && i < region.end) continue;
    if (seen++ == ordinal) {
      call = i;
      return true;
    }
  }
  return false;
}

bool CollectActualCallAccesses(
    const LoweredStep& lowered, const ComparisonRegionAnalysis& region, std::size_t call,
    const LaunchContext* context, ComparisonRemainderCallFactsFn describe_facts,
    std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& facts, std::size_t& count) {
  const bool in_region = call >= region.first && call < region.end;
  RegionAccessList specs{};
  return CollectCallAccesses(lowered, call, context, describe_facts, in_region, specs, facts,
                             count);
}

bool IsMetadataViewOfSource(const ggml_tensor* view, const ggml_tensor* source) {
  if (!view || !source || !view->data || !source->data ||
      (view->op != GGML_OP_VIEW && view->op != GGML_OP_RESHAPE && view->op != GGML_OP_PERMUTE &&
       view->op != GGML_OP_TRANSPOSE) ||
      view->src[0] != source)
    return false;
  const auto* root = source->view_src ? source->view_src : source;
  const std::uint64_t source_offset = source->view_src ? source->view_offs : 0;
  const std::uint64_t view_offset = view->view_offs;
  const std::uint64_t source_bytes = ggml_nbytes(source);
  const std::uint64_t view_bytes = ggml_nbytes(view);
  const std::uint64_t root_bytes = ggml_nbytes(root);
  if (view->view_src != root || view_offset < source_offset || view_bytes == 0 ||
      source_offset > root_bytes || source_bytes > root_bytes - source_offset ||
      view_offset > root_bytes || view_bytes > root_bytes - view_offset ||
      view_offset - source_offset > source_bytes ||
      view_bytes > source_bytes - (view_offset - source_offset))
    return false;
  std::uintptr_t expected_data = 0;
  return !__builtin_add_overflow(reinterpret_cast<std::uintptr_t>(source->data),
                                 static_cast<std::uintptr_t>(view_offset - source_offset),
                                 &expected_data) &&
         expected_data == reinterpret_cast<std::uintptr_t>(view->data);
}

const PhysicalTensor* CheckedCopyDestination(const LoweredStep& lowered,
                                             const PhysicalTensor& result) {
  const auto* copy = result.descriptor;
  if (!copy || copy->op != GGML_OP_CPY || !result.tensor.valid() || result.root != result.tensor ||
      !result.producer.valid())
    return nullptr;
  const auto step = std::ranges::find_if(lowered.plan.steps, [&](const PlanStep& call) {
    return call.implementation == "ggml.convert" && call.nodes.size() == 1 && call.nodes[0] == copy;
  });
  if (step == lowered.plan.steps.end() ||
      !DescribeComparisonAccess(*step, ComparisonAccessScope::kWhole))
    return nullptr;
  const auto* destination = PhysicalTensorOf(lowered, copy->src[1]);
  const auto* storage = PhysicalRootFor(lowered, result.root);
  if (!destination || destination->tensor != result.tensor || destination->root != result.root ||
      destination->producer != result.producer || destination->ir_format != result.ir_format ||
      destination->id == result.id || !storage || storage->backend_tensor != result.id ||
      storage->producer != result.producer)
    return nullptr;
  return destination;
}

const PhysicalTensor* LifetimeStorageRoot(const LoweredStep& lowered, ir::TensorId root) {
  const auto count = std::ranges::count(lowered.physical_tensors, root, &PhysicalTensor::tensor);
  if (count == 1) {
    const auto found = std::ranges::find(lowered.physical_tensors, root, &PhysicalTensor::tensor);
    return found->descriptor ? &*found : nullptr;
  }
  if (count != 2) return nullptr;
  // The only additional identity is the checked CPY result view paired with
  // its actual destination. Other duplicate root descriptors remain refused.
  for (const auto& candidate : lowered.physical_tensors) {
    if (candidate.tensor != root) continue;
    if (const auto* destination = CheckedCopyDestination(lowered, candidate)) return destination;
  }
  return nullptr;
}

std::optional<std::uint32_t> DefinitionIdFor(const LoweredStep& lowered,
                                             std::uint32_t physical_id) {
  const PhysicalTensor* current = PhysicalTensorForId(lowered, physical_id);
  if (!current || !current->descriptor) return std::nullopt;
  for (std::size_t depth = 0; depth <= lowered.physical_tensors.size(); ++depth) {
    const auto* descriptor = current->descriptor;
    if (descriptor->op == GGML_OP_CPY) {
      const auto* destination = CheckedCopyDestination(lowered, *current);
      return destination ? std::optional{destination->id} : std::nullopt;
    }
    if (descriptor->op != GGML_OP_VIEW && descriptor->op != GGML_OP_RESHAPE &&
        descriptor->op != GGML_OP_PERMUTE && descriptor->op != GGML_OP_TRANSPOSE)
      return current->id;
    const auto* source = descriptor->src[0];
    if (!IsMetadataViewOfSource(descriptor, source)) return std::nullopt;
    const auto* source_physical = PhysicalTensorOf(lowered, source);
    if (!source_physical || source_physical->descriptor != source) return std::nullopt;
    current = PhysicalTensorForId(lowered, source_physical->id);
    if (!current || !current->descriptor) return std::nullopt;
  }
  return std::nullopt;
}

bool IsAuthenticatedExternalDefinition(const LoweredStep& lowered, std::uint32_t physical_id) {
  const auto* physical = PhysicalTensorForId(lowered, physical_id);
  if (!physical || !physical->tensor.valid() || !lowered.comparison_origins) return false;
  const auto& relation = *lowered.comparison_origins;
  if (physical->tensor.v >= relation.planned_tensors.size()) return false;
  const auto role = relation.planned_tensors[physical->tensor.v].role;
  if (role != ir::Role::kInput && role != ir::Role::kWeight && role != ir::Role::kConstant &&
      role != ir::Role::kState)
    return false;
  std::size_t matches = 0;
  for (std::size_t original = 0; original < relation.original_to_planned.size(); ++original) {
    if (relation.original_to_planned[original] != physical->tensor) continue;
    ++matches;
    const ir::TensorId original_id{static_cast<std::uint32_t>(original)};
    if (std::ranges::count(relation.original_bindings, original_id,
                           &ComparisonBindingRange::original_root) != 1)
      return false;
  }
  return matches == 1;
}

bool RecordLifetimeAccess(const LoweredStep& lowered, std::size_t call,
                          const ComparisonRegionAccessFact& access) {
  auto& relation = *lowered.comparison_origins;
  if (access.domain == ComparisonRegionAccessDomain::kPool) return access.physical_id == kUnset;
  if (access.saved_destination) {
    if (access.domain != ComparisonRegionAccessDomain::kExternal || !access.written ||
        access.read || access.physical_id != kUnset ||
        !ValidAddressRange(access.address, access.bytes))
      return false;
    const auto saved = FindSavedCall(lowered, lowered.plan.steps[call]);
    if (!saved.leaf || !SavedOwnerRangeCurrent(*saved.leaf) ||
        access.address != saved.leaf->owner_range->base ||
        access.bytes != ggml_nbytes(saved.leaf->tensor) ||
        reinterpret_cast<std::uintptr_t>(saved.leaf->store->data) != access.address)
      return false;
    const auto* physical = PhysicalTensorOf(lowered, saved.leaf->store);
    const auto definition = physical ? DefinitionIdFor(lowered, physical->id) : std::nullopt;
    if (!physical || !definition || physical->id >= relation.lifetimes.size() ||
        *definition != physical->id)
      return false;
    auto& lifetime = relation.lifetimes[physical->id];
    if ((lifetime.address != 0 && lifetime.address != access.address) ||
        (lifetime.definition_id != kUnset && lifetime.definition_id != *definition))
      return false;
    lifetime.address = access.address;
    lifetime.storage_id = physical->id;
    lifetime.definition_id = *definition;
    lifetime.bytes = std::max(lifetime.bytes, access.bytes);
    lifetime.external_source = false;
    if (std::pair{call, access.subcall_ordinal} <
        std::pair{lifetime.first_write_call, lifetime.first_write_subcall}) {
      lifetime.first_write_call = call;
      lifetime.first_write_subcall = access.subcall_ordinal;
    }
    return true;
  }
  const auto* physical = PhysicalTensorForId(lowered, access.physical_id);
  const auto definition = DefinitionIdFor(lowered, access.physical_id);
  if (access.physical_id >= relation.lifetimes.size() || !physical || !physical->descriptor ||
      !definition || access.address == 0 || access.bytes == 0)
    return false;
  std::uint32_t storage_id = physical->id;
  std::uint64_t storage_offset = 0;
  const auto gates_storage = CheckedGdnGatesStorage(lowered, physical->descriptor);
  if (gates_storage || physical->root.valid()) {
    const auto* root =
        gates_storage ? gates_storage->outputs[0] : LifetimeStorageRoot(lowered, physical->root);
    if (!root) return false;
    storage_id = root->id;
    const auto root_address = reinterpret_cast<std::uintptr_t>(root->descriptor->data);
    if (root_address == 0 || access.address < root_address) return false;
    storage_offset = static_cast<std::uint64_t>(access.address - root_address);
  }
  auto& lifetime = relation.lifetimes[access.physical_id];
  if ((lifetime.address != 0 && lifetime.address != access.address) ||
      (lifetime.storage_id != kUnset && lifetime.storage_id != storage_id) ||
      (lifetime.definition_id != kUnset && lifetime.definition_id != *definition) ||
      (lifetime.address != 0 && lifetime.storage_offset != storage_offset))
    return false;
  lifetime.address = access.address;
  lifetime.storage_id = storage_id;
  lifetime.definition_id = *definition;
  lifetime.external_source = IsAuthenticatedExternalDefinition(lowered, *definition);
  lifetime.storage_offset = storage_offset;
  lifetime.bytes = std::max(lifetime.bytes, access.bytes);
  if (access.read) {
    if (std::pair{call, access.subcall_ordinal} <
        std::pair{lifetime.first_read_call, lifetime.first_read_subcall}) {
      lifetime.first_read_call = call;
      lifetime.first_read_subcall = access.subcall_ordinal;
    }
    lifetime.has_read = true;
    if (std::pair{call, access.subcall_ordinal} >
        std::pair{lifetime.last_read_call, lifetime.last_read_subcall}) {
      lifetime.last_read_call = call;
      lifetime.last_read_subcall = access.subcall_ordinal;
    }
  }
  if (access.written && !lifetime.external_source &&
      std::pair{call, access.subcall_ordinal} <
          std::pair{lifetime.first_write_call, lifetime.first_write_subcall}) {
    lifetime.first_write_call = call;
    lifetime.first_write_subcall = access.subcall_ordinal;
  }
  return true;
}

bool MarkKeptLifetimes(const LoweredStep& lowered) {
  auto& relation = *lowered.comparison_origins;
  const auto end = lowered.plan.steps.size();
  for (const auto* tensor : lowered.keep) {
    const auto* physical = PhysicalTensorOf(lowered, tensor);
    if (!physical || !PhysicalTensorForId(lowered, physical->id) ||
        physical->id >= relation.lifetimes.size())
      return false;
    auto& lifetime = relation.lifetimes[physical->id];
    const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
    if (address == 0 || (lifetime.address != 0 && lifetime.address != address)) return false;
    if (lifetime.storage_id == kUnset || lifetime.definition_id == kUnset) {
      const auto* descriptor = PhysicalTensorForId(lowered, physical->id);
      if (!descriptor) return false;
      const auto definition = DefinitionIdFor(lowered, physical->id);
      if (!definition) return false;
      lifetime.definition_id = *definition;
      lifetime.storage_id = physical->id;
      const auto gates_storage = CheckedGdnGatesStorage(lowered, descriptor->descriptor);
      if (gates_storage || descriptor->root.valid()) {
        const auto* root = gates_storage ? gates_storage->outputs[0]
                                         : LifetimeStorageRoot(lowered, descriptor->root);
        if (!root) return false;
        lifetime.storage_id = root->id;
        const auto root_address = reinterpret_cast<std::uintptr_t>(root->descriptor->data);
        if (root_address == 0 || address < root_address) return false;
        lifetime.storage_offset = static_cast<std::uint64_t>(address - root_address);
      }
    }
    if (std::pair{end, std::uint8_t{0}} <
        std::pair{lifetime.first_read_call, lifetime.first_read_subcall}) {
      lifetime.first_read_call = end;
      lifetime.first_read_subcall = 0;
    }
    lifetime.address = reinterpret_cast<std::uintptr_t>(tensor->data);
    lifetime.bytes = std::max(lifetime.bytes, static_cast<std::uint64_t>(ggml_nbytes(tensor)));
    lifetime.has_read = true;
    lifetime.kept = true;
    lifetime.last_read_call = end;
    lifetime.last_read_subcall = 0;
  }
  return true;
}

bool FinalizeRemainderLifetimes(const LoweredStep& lowered) {
  auto& lifetimes = lowered.comparison_origins->lifetimes;
  for (std::size_t id = 0; id < lifetimes.size(); ++id) {
    auto& value = lifetimes[id];
    if (!value.has_read || value.definition_id == kUnset || value.definition_id >= lifetimes.size())
      continue;
    const auto& definition = lifetimes[value.definition_id];
    value.external_source =
        IsAuthenticatedExternalDefinition(lowered, static_cast<std::uint32_t>(value.definition_id));
    if (!value.external_source && definition.first_write_call != SIZE_MAX && value.has_read &&
        std::pair{definition.first_write_call, definition.first_write_subcall} >=
            std::pair{value.first_read_call, value.first_read_subcall})
      return false;
    if (definition.first_write_call != SIZE_MAX &&
        std::pair{definition.first_write_call, definition.first_write_subcall} <
            std::pair{value.first_write_call, value.first_write_subcall}) {
      value.first_write_call = definition.first_write_call;
      value.first_write_subcall = definition.first_write_subcall;
    } else if (definition.first_write_call == SIZE_MAX && !value.external_source) {
      return false;
    }
  }
  return true;
}

bool BuildRemainderLifetimes(const LoweredStep& lowered, const ComparisonRegionAnalysis& region,
                             const LaunchContext* context,
                             ComparisonRemainderCallFactsFn describe_facts) {
  if (!lowered.comparison_origins) return false;
  auto& lifetimes = lowered.comparison_origins->lifetimes;
  if (lifetimes.size() < lowered.physical_tensors.size()) return false;
  std::ranges::fill(lifetimes, ComparisonLifetimeFact{});
  for (std::size_t call = 0; call < lowered.plan.steps.size(); ++call) {
    std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity> facts{};
    std::size_t count = 0;
    if (!CollectActualCallAccesses(lowered, region, call, context, describe_facts, facts, count)) {
      return false;
    }
    for (std::size_t i = 0; i < count; ++i)
      if (!RecordLifetimeAccess(lowered, call, facts[i])) {
        return false;
      }
  }
  if (!MarkKeptLifetimes(lowered)) {
    return false;
  }
  if (!FinalizeRemainderLifetimes(lowered)) {
    return false;
  }
  return true;
}

bool LiveAtEvent(const ComparisonLifetimeFact& lifetime, std::size_t call, std::uint8_t subcall) {
  if (!lifetime.has_read || lifetime.address == 0 || lifetime.bytes == 0) return false;
  const auto event = std::pair{call, subcall};
  const bool born = lifetime.first_write_call == SIZE_MAX ||
                    std::pair{lifetime.first_write_call, lifetime.first_write_subcall} <= event;
  const bool alive =
      lifetime.kept || std::pair{lifetime.last_read_call, lifetime.last_read_subcall} >= event;
  return born && alive;
}

bool LiveAfterEvent(const ComparisonLifetimeFact& lifetime, std::size_t call,
                    std::uint8_t subcall) {
  const auto event = std::pair{call, subcall};
  return LiveAtEvent(lifetime, call, subcall) &&
         (lifetime.kept || std::pair{lifetime.last_read_call, lifetime.last_read_subcall} > event);
}

bool ComparisonWritePreservesLifetimeInternal(std::uintptr_t write_address,
                                              std::uint64_t write_bytes, std::size_t call,
                                              std::uint8_t subcall,
                                              const ComparisonLifetimeFact& value) {
  if (!ValidAddressRange(write_address, write_bytes)) return false;
  if (!value.has_read) return true;
  if (!ValidAddressRange(value.address, value.bytes)) return false;
  if (!LiveAfterEvent(value, call, subcall)) return true;
  const auto overlap = IntervalsOverlap(write_address, write_bytes, value.address, value.bytes);
  return overlap.has_value() && !*overlap;
}

bool CheckRemainderWriteLiveness(const LoweredStep& lowered, const ComparisonRegionAnalysis& region,
                                 const LaunchContext* context,
                                 ComparisonRemainderCallFactsFn describe_facts,
                                 std::string_view* refusal_reason) {
  auto& relation = *lowered.comparison_origins;
  for (std::size_t call = 0; call < lowered.plan.steps.size(); ++call) {
    std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity> facts{};
    std::size_t count = 0;
    if (!CollectActualCallAccesses(lowered, region, call, context, describe_facts, facts, count))
      return false;
    for (std::size_t i = 0; i < count; ++i) {
      const auto& write = facts[i];
      if (!write.written || write.domain == ComparisonRegionAccessDomain::kPool) continue;
      if (write.saved_destination) {
        if (write.physical_id != kUnset ||
            write.domain != ComparisonRegionAccessDomain::kExternal || write.read ||
            !ValidAddressRange(write.address, write.bytes) ||
            !ExactSavedStoreCall(lowered, lowered.plan.steps[call]))
          return false;
        const auto save_call = FindSavedCall(lowered, lowered.plan.steps[call]);
        if (!save_call.leaf || !SavedOwnerRangeCurrent(*save_call.leaf) ||
            write.address != save_call.leaf->owner_range->base ||
            write.bytes != ggml_nbytes(save_call.leaf->tensor))
          return false;
        const auto* save_physical = PhysicalTensorOf(lowered, save_call.leaf->store);
        if (!save_physical) return false;
        // The destination write is authorized for its exact SavedLeaf only.
        // Its range must still be disjoint from every different live value,
        // protected input/state/immutable binding, generated index buffer,
        // and other save allocation.
        for (const auto& other : lowered.physical_tensors) {
          if (other.id >= relation.lifetimes.size()) return false;
          const auto& lifetime = relation.lifetimes[other.id];
          if (lifetime.definition_id == save_physical->id) continue;
          if (!ComparisonWritePreservesLifetimeInternal(write.address, write.bytes, call,
                                                        write.subcall_ordinal, lifetime)) {
            if (refusal_reason)
              *refusal_reason = "saved destination overlaps an activation value that remains live";
            return false;
          }
        }
        for (const auto& binding : lowered.comparison_bindings) {
          if (!binding.original_root.valid() ||
              binding.original_root.v >= relation.original_to_planned.size() ||
              !relation.original_to_planned[binding.original_root.v].valid())
            return false;
          const auto planned = relation.original_to_planned[binding.original_root.v];
          if (planned.v >= relation.planned_tensors.size()) return false;
          const auto role = relation.planned_tensors[planned.v].role;
          const auto input = std::ranges::find(relation.input_facts, binding.original_root,
                                               &ComparisonInputFact::original_tensor);
          const bool protected_binding =
              role == ir::Role::kWeight || role == ir::Role::kConstant ||
              role == ir::Role::kState ||
              (role == ir::Role::kInput && input != relation.input_facts.end() &&
               input->token_input &&
               (!input->token_kind || !ReadyTokenValueKind(*input->token_kind)));
          if (!protected_binding) continue;
          std::uintptr_t begin = 0;
          if (binding.base == 0 || binding.bytes == 0 ||
              binding.offset > binding.allocation_bytes ||
              binding.bytes > binding.allocation_bytes - binding.offset ||
              __builtin_add_overflow(static_cast<std::uintptr_t>(binding.base),
                                     static_cast<std::uintptr_t>(binding.offset), &begin))
            return false;
          const auto overlap =
              IntervalsOverlap(static_cast<std::uintptr_t>(save_call.leaf->owner_range->base),
                               save_call.leaf->owner_range->bytes, begin, binding.bytes);
          if (!overlap || *overlap) {
            if (refusal_reason)
              *refusal_reason = "saved destination overlaps a protected immutable or state binding";
            return false;
          }
        }
        if (lowered.generated_store) {
          const auto& slot = lowered.generated_store->slot;
          const auto overlap =
              IntervalsOverlap(static_cast<std::uintptr_t>(save_call.leaf->owner_range->base),
                               save_call.leaf->owner_range->bytes,
                               static_cast<std::uintptr_t>(slot.device_address), slot.bytes);
          if (!overlap || *overlap) return false;
        }
        for (const auto& other_save : lowered.saves) {
          if (&other_save == save_call.leaf) continue;
          if (!SavedOwnerRangeCurrent(other_save)) return false;
          const auto overlap =
              IntervalsOverlap(static_cast<std::uintptr_t>(save_call.leaf->owner_range->base),
                               save_call.leaf->owner_range->bytes,
                               static_cast<std::uintptr_t>(other_save.owner_range->base),
                               other_save.owner_range->bytes);
          if (!overlap || *overlap) {
            if (refusal_reason)
              *refusal_reason = "saved destination overlaps another saved allocation";
            return false;
          }
        }
        continue;
      }
      if (write.physical_id >= relation.lifetimes.size()) return false;
      const auto* written_tensor = PhysicalTensorForId(lowered, write.physical_id);
      if (!written_tensor || !written_tensor->descriptor) return false;
      for (const auto& save : lowered.saves) {
        if (!SavedOwnerRangeCurrent(save)) return false;
        const auto save_address = static_cast<std::uintptr_t>(save.owner_range->base);
        const auto save_bytes = static_cast<std::uint64_t>(ggml_nbytes(save.tensor));
        if (!ValidAddressRange(save_address, save_bytes)) return false;
        const auto overlap = IntervalsOverlap(write.address, write.bytes, save_address, save_bytes);
        if (!overlap) return false;
        if (*overlap) return false;
      }
      if (lowered.generated_store) {
        const auto& slot = lowered.generated_store->slot;
        const auto slot_address = static_cast<std::uintptr_t>(slot.device_address);
        if (!ValidAddressRange(slot_address, slot.bytes)) return false;
        const auto overlap = IntervalsOverlap(write.address, write.bytes, slot_address, slot.bytes);
        if (!overlap || *overlap) return false;
      }
      const auto& write_lifetime = relation.lifetimes[write.physical_id];
      if (write_lifetime.definition_id == kUnset) return false;
      bool authorized_state_store = false;
      if (written_tensor->root.valid() &&
          written_tensor->root.v < relation.planned_tensors.size() &&
          relation.planned_tensors[written_tensor->root.v].role == ir::Role::kState &&
          write_lifetime.external_source) {
        // Only checked SetRows/GroupedStores/RopeStore call schemas can emit
        // an authenticated state-write access. Their state target and view
        // range were already checked by ResolveRegionAccess.
        authorized_state_store = true;
      }
      for (const auto& other : lowered.physical_tensors) {
        if (other.id >= relation.lifetimes.size()) return false;
        if (other.id == write.physical_id) continue;
        const auto& lifetime = relation.lifetimes[other.id];
        if (authorized_state_store && lifetime.external_source &&
            lifetime.storage_id == write_lifetime.storage_id &&
            lifetime.definition_id == write_lifetime.definition_id)
          continue;
        if (lifetime.definition_id == write_lifetime.definition_id &&
            write_lifetime.first_write_call == call &&
            write_lifetime.first_write_subcall == write.subcall_ordinal)
          continue;
        if (!ComparisonWritePreservesLifetimeInternal(write.address, write.bytes, call,
                                                      write.subcall_ordinal, lifetime))
          return false;
      }
    }
  }
  return true;
}

std::optional<std::uint32_t> PhysicalIdForPort(const LoweredStep& lowered,
                                               const ComparisonRegionValuePort& port) {
  if (!lowered.comparison_origins || !port.source_tensor.valid() || port.address == 0 ||
      port.source_tensor.v >= lowered.comparison_origins->original_to_planned.size())
    return std::nullopt;
  const auto planned = lowered.comparison_origins->original_to_planned[port.source_tensor.v];
  if (!planned.valid()) return std::nullopt;
  std::optional<std::uint32_t> id;
  for (const auto& physical : lowered.physical_tensors) {
    if (physical.tensor != planned || !physical.descriptor ||
        reinterpret_cast<std::uintptr_t>(physical.descriptor->data) != port.address)
      continue;
    if (id) return std::nullopt;
    id = physical.id;
  }
  return id;
}

bool SameLiveCallPortTopology(
    const LoweredStep& a, const ComparisonRegionAnalysis& ar,
    const std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& aa, std::size_t an,
    std::size_t acall, const LoweredStep& b, const ComparisonRegionAnalysis& br,
    const std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& ba, std::size_t bn,
    std::size_t bcall) {
  if (an != bn || ar.value_ports.size() != br.value_ports.size()) return false;
  for (std::size_t access = 0; access < an; ++access) {
    for (std::size_t port = 0; port < ar.value_ports.size(); ++port) {
      const auto aid = PhysicalIdForPort(a, ar.value_ports[port]);
      const auto bid = PhysicalIdForPort(b, br.value_ports[port]);
      if (!aid || !bid || *aid >= a.comparison_origins->lifetimes.size() ||
          *bid >= b.comparison_origins->lifetimes.size())
        return false;
      const bool a_live =
          LiveAtEvent(a.comparison_origins->lifetimes[*aid], acall, aa[access].subcall_ordinal);
      const bool b_live =
          LiveAtEvent(b.comparison_origins->lifetimes[*bid], bcall, ba[access].subcall_ordinal);
      if (a_live != b_live) return false;
      if (a_live &&
          !SameOverlapRelation(aa[access].address, aa[access].bytes, ar.value_ports[port].address,
                               ar.value_ports[port].bytes, ba[access].address, ba[access].bytes,
                               br.value_ports[port].address, br.value_ports[port].bytes))
        return false;
    }
  }
  return true;
}

bool SameLiveCrossCallTopology(
    const LoweredStep& a,
    const std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& a_early,
    std::size_t a_early_count, std::size_t a_early_call,
    const std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& a_late,
    std::size_t a_late_count, std::size_t a_late_call, const LoweredStep& b,
    const std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& b_early,
    std::size_t b_early_count, std::size_t b_early_call,
    const std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity>& b_late,
    std::size_t b_late_count, std::size_t b_late_call) {
  if (a_early_count != b_early_count || a_late_count != b_late_count) return false;
  for (std::size_t i = 0; i < a_early_count; ++i) {
    if (a_early[i].physical_id == kUnset) {
      if (b_early[i].physical_id != kUnset) return false;
      continue;
    }
    if (b_early[i].physical_id == kUnset ||
        a_early[i].physical_id >= a.comparison_origins->lifetimes.size() ||
        b_early[i].physical_id >= b.comparison_origins->lifetimes.size())
      return false;
    const auto& a0 = a.comparison_origins->lifetimes[a_early[i].physical_id];
    const auto& b0 = b.comparison_origins->lifetimes[b_early[i].physical_id];
    for (std::size_t j = 0; j < a_late_count; ++j) {
      if (a_late[j].physical_id == kUnset) {
        if (b_late[j].physical_id != kUnset) return false;
        continue;
      }
      if (b_late[j].physical_id == kUnset ||
          a_late[j].physical_id >= a.comparison_origins->lifetimes.size() ||
          b_late[j].physical_id >= b.comparison_origins->lifetimes.size())
        return false;
      const auto& a1 = a.comparison_origins->lifetimes[a_late[j].physical_id];
      const auto& b1 = b.comparison_origins->lifetimes[b_late[j].physical_id];
      const bool a_live = LiveAtEvent(a0, a_late_call, a_late[j].subcall_ordinal) &&
                          LiveAtEvent(a1, a_late_call, a_late[j].subcall_ordinal);
      const bool b_live = LiveAtEvent(b0, b_late_call, b_late[j].subcall_ordinal) &&
                          LiveAtEvent(b1, b_late_call, b_late[j].subcall_ordinal);
      if (a_live != b_live) return false;
      if (a_live && !SameOverlapRelation(a_early[i].address, a_early[i].bytes, a_late[j].address,
                                         a_late[j].bytes, b_early[i].address, b_early[i].bytes,
                                         b_late[j].address, b_late[j].bytes))
        return false;
    }
  }
  (void)a_early_call;
  (void)b_early_call;
  return true;
}

bool SameComparisonAccessTopology(const LoweredStep& a, const ComparisonRegionAnalysis& ar,
                                  const LaunchContext* ac, const LoweredStep& b,
                                  const ComparisonRegionAnalysis& br, const LaunchContext* bc,
                                  ComparisonRemainderCallFactsFn describe_facts,
                                  std::string_view* refusal_reason) {
  std::string_view liveness_reason;
  if (!SamePortTopology(ar, br) || !BuildRemainderLifetimes(a, ar, ac, describe_facts) ||
      !BuildRemainderLifetimes(b, br, bc, describe_facts) ||
      !CheckRemainderWriteLiveness(a, ar, ac, describe_facts, &liveness_reason) ||
      !CheckRemainderWriteLiveness(b, br, bc, describe_facts, &liveness_reason)) {
    if (refusal_reason && !liveness_reason.empty()) *refusal_reason = liveness_reason;
    return false;
  }
  const auto a_outside_count = a.plan.steps.size() - (ar.end - ar.first);
  const auto b_outside_count = b.plan.steps.size() - (br.end - br.first);
  if (a_outside_count != b_outside_count) return false;

  for (std::size_t i = 0; i < a_outside_count; ++i) {
    std::size_t ai = 0, bi = 0;
    std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity> aa{}, ba{};
    std::size_t an = 0, bn = 0;
    if (!OutsideCallAt(a, ar, i, ai) || !OutsideCallAt(b, br, i, bi) ||
        !CollectActualCallAccesses(a, ar, ai, ac, describe_facts, aa, an) ||
        !CollectActualCallAccesses(b, br, bi, bc, describe_facts, ba, bn) ||
        !SameWithinCallTopology(aa, an, ba, bn)) {
      return false;
    }
    for (std::size_t j = 0; j < an; ++j)
      if (!SameAccessDescription(aa[j], ba[j])) {
        return false;
      }
    if (!SameLiveCallPortTopology(a, ar, aa, an, ai, b, br, ba, bn, bi)) {
      return false;
    }
    for (std::size_t j = 0; j < i; ++j) {
      std::size_t aj = 0, bj = 0;
      std::array<ComparisonRegionAccessFact, kRemainderAccessCapacity> ja{}, jb{};
      std::size_t jan = 0, jbn = 0;
      if (!OutsideCallAt(a, ar, j, aj) || !OutsideCallAt(b, br, j, bj) ||
          !CollectActualCallAccesses(a, ar, aj, ac, describe_facts, ja, jan) ||
          !CollectActualCallAccesses(b, br, bj, bc, describe_facts, jb, jbn) ||
          !SameLiveCrossCallTopology(a, ja, jan, aj, aa, an, ai, b, jb, jbn, bj, ba, bn, bi)) {
        return false;
      }
    }
  }
  return true;
}

bool SameInputSlot(const InputSlot& a, const InputSlot& b) {
  return a.bytes == b.bytes && a.tensor == b.tensor && a.segment == b.segment && a.kind == b.kind &&
         a.cells == b.cells && a.read == b.read && a.window == b.window &&
         a.mask_rows == b.mask_rows && a.outputs == b.outputs && a.sections == b.sections &&
         a.count == b.count && a.host == b.host && a.ratio == b.ratio && a.overlap == b.overlap &&
         a.ring == b.ring && a.ahead == b.ahead && a.commit_save == b.commit_save &&
         a.component == b.component && a.visible_capacity == b.visible_capacity &&
         a.invariant == b.invariant;
}

bool SameProgramSetup(const LoweredStep& a, const LoweredStep& b) {
  if (a.inputs.size() != b.inputs.size() || a.outputs.size() != b.outputs.size() ||
      a.saves.size() != b.saves.size() || !a.routing.empty() || !b.routing.empty() ||
      !a.plan.regions.empty() || !b.plan.regions.empty() || a.prologue.size() != b.prologue.size())
    return false;
  for (std::size_t i = 0; i < a.inputs.size(); ++i)
    if (!SameInputSlot(a.inputs[i], b.inputs[i])) return false;
  for (std::size_t i = 0; i < a.outputs.size(); ++i)
    if (a.outputs[i].bytes != b.outputs[i].bytes || a.outputs[i].tensor != b.outputs[i].tensor ||
        a.outputs[i].segment != b.outputs[i].segment ||
        a.outputs[i].features != b.outputs[i].features)
      return false;
  for (std::size_t i = 0; i < a.saves.size(); ++i) {
    const auto& x = a.saves[i];
    const auto& y = b.saves[i];
    if (!SavedMappingCurrent(a, x) || !SavedMappingCurrent(b, y) || !SavedOwnerRangeCurrent(x) ||
        !SavedOwnerRangeCurrent(y) || x.original_source != y.original_source ||
        x.original_rows != y.original_rows || x.layer != y.layer || x.component != y.component ||
        x.segment != y.segment || x.what != y.what || !x.tensor || !y.tensor ||
        *x.owner_range != *y.owner_range || ggml_nbytes(x.tensor) != ggml_nbytes(y.tensor) ||
        x.tensor->data != y.tensor->data)
      return false;
  }
  if (a.generated_store.has_value() != b.generated_store.has_value()) return false;
  if (a.generated_store) {
    const auto& x = *a.generated_store;
    const auto& y = *b.generated_store;
    if (!SameInputSlot(x.slot, y.slot) || x.stage != y.stage || x.phase != y.phase ||
        x.segment != y.segment || x.row_invariant != y.row_invariant ||
        x.read_align != y.read_align || !a.generated_store_access_closed ||
        !b.generated_store_access_closed)
      return false;
  }
  return true;
}

bool SameExternalBindingContracts(const LoweredStep::ComparisonOrigins& a,
                                  const LoweredStep::ComparisonOrigins& b) {
  if (a.original_bindings.size() != b.original_bindings.size() ||
      a.input_facts.size() != b.input_facts.size())
    return false;
  for (std::size_t i = 0; i < a.input_facts.size(); ++i) {
    const auto& x = a.input_facts[i];
    const auto& y = b.input_facts[i];
    if (x.original_tensor != y.original_tensor || x.token_input != y.token_input ||
        x.token_kind != y.token_kind || x.token_segment != y.token_segment ||
        x.token_component != y.token_component || x.token_read != y.token_read ||
        x.unstored_reason != y.unstored_reason || x.unstored_verified != y.unstored_verified)
      return false;
  }
  const auto is_source_input = [](const LoweredStep::ComparisonOrigins& origins,
                                  ir::TensorId tensor) {
    return std::ranges::any_of(origins.input_facts, [&](const ComparisonInputFact& input) {
      return input.original_tensor == tensor;
    });
  };
  for (std::size_t i = 0; i < a.original_bindings.size(); ++i) {
    const auto& x = a.original_bindings[i];
    const auto& y = b.original_bindings[i];
    if (x.original_root != y.original_root || x.part != y.part || x.bytes != y.bytes ||
        x.alignment != y.alignment || x.resource_part_identity != y.resource_part_identity ||
        x.row_storage != y.row_storage)
      return false;
    // Source inputs may be placed at different addresses in the two actual
    // activation arenas. Their accessed intervals and live alias relations are
    // compared below from the real lowered descriptors. Immutable weights,
    // constants, and state retain exact resource/range identity.
    const bool input = is_source_input(a, x.original_root);
    if (input != is_source_input(b, y.original_root)) return false;
    if (!input &&
        (x.base != y.base || x.allocation_bytes != y.allocation_bytes || x.offset != y.offset))
      return false;
    for (std::size_t j = 0; j < i; ++j) {
      const auto& xp = a.original_bindings[j];
      const auto& yp = b.original_bindings[j];
      const bool prior_input = is_source_input(a, xp.original_root);
      if (!input && !prior_input &&
          ((x.allocation_class == xp.allocation_class) !=
           (y.allocation_class == yp.allocation_class)))
        return false;
    }
  }
  return true;
}

bool AppendCallOrigins(const LoweredStep& source, std::size_t call,
                       std::vector<ir::NodeId>& destination,
                       const components::OwnedComparisonRegionBoundary& boundary) {
  if (!source.comparison_origins || call >= source.plan.steps.size()) return false;
  const auto& relation = *source.comparison_origins;
  const auto saved = FindSavedCall(source, source.plan.steps[call]);
  if (saved.leaf) {
    if (!SavedMappingCurrent(source, *saved.leaf)) return false;
    bool has_expandable_origin = false;
    for (const auto planned : {saved.leaf->source, saved.leaf->rows}) {
      const auto physical =
          std::ranges::find(source.physical_tensors, planned, &PhysicalTensor::tensor);
      if (physical == source.physical_tensors.end() || !physical->producer.valid() ||
          std::ranges::count(source.physical_tensors, planned, &PhysicalTensor::tensor) != 1 ||
          physical->producer.v + 1 >= relation.offsets.size())
        return false;
      const auto first = relation.offsets[physical->producer.v];
      const auto end = relation.offsets[physical->producer.v + 1];
      if (first >= end || end > relation.source_nodes.size()) return false;
      for (auto offset = first; offset < end; ++offset) {
        const auto origin = relation.source_nodes[static_cast<std::size_t>(offset)];
        if (!origin.valid()) return false;
        if (std::ranges::any_of(boundary.nodes(),
                                [&](const auto& member) { return member.original_node == origin; }))
          continue;
        has_expandable_origin = true;
        if (std::ranges::find(destination, origin) == destination.end()) {
          if (destination.size() == destination.capacity()) return false;
          destination.push_back(origin);
        }
      }
    }
    return has_expandable_origin;
  }
  bool has_expandable_origin = false;
  for (const auto* descriptor : source.plan.steps[call].nodes) {
    const auto* physical = PhysicalTensorOf(source, descriptor);
    if (!physical || !physical->producer.valid() ||
        physical->producer.v + 1 >= relation.offsets.size())
      return false;
    const auto first = relation.offsets[physical->producer.v];
    const auto end = relation.offsets[physical->producer.v + 1];
    if (end < first || end > relation.source_nodes.size()) return false;
    for (auto offset = first; offset < end; ++offset) {
      const auto origin = relation.source_nodes[offset];
      if (!origin.valid()) return false;
      if (std::ranges::any_of(boundary.nodes(),
                              [&](const auto& member) { return member.original_node == origin; }))
        continue;
      has_expandable_origin = true;
      if (std::ranges::find(destination, origin) == destination.end()) {
        if (destination.size() == destination.capacity()) return false;
        destination.push_back(origin);
      }
    }
  }
  return has_expandable_origin;
}

bool AppendExpansion(std::span<const ir::NodeId> additions, std::vector<ir::NodeId>& destination,
                     const components::OwnedComparisonRegionBoundary& boundary) {
  for (const auto origin : additions) {
    if (!origin.valid()) return false;
    if (std::ranges::any_of(boundary.nodes(),
                            [&](const auto& member) { return member.original_node == origin; }))
      continue;
    if (std::ranges::find(destination, origin) == destination.end()) {
      if (destination.size() == destination.capacity()) return false;
      destination.push_back(origin);
    }
  }
  return true;
}

bool SameEnvironment(const kernels::ComparisonRecipeSeal& a,
                     const kernels::ComparisonRecipeSeal& b) {
  return a.mode == kernels::ComparisonRunMode::kFull && b.mode == a.mode &&
         a.subject == b.subject && a.boundary == b.boundary && a.policy == b.policy &&
         a.registry == b.registry && a.backend_recipe == b.backend_recipe &&
         a.backend_build == b.backend_build && a.backend_selection == b.backend_selection &&
         NonzeroDigest(a.policy) && NonzeroDigest(a.registry) && NonzeroDigest(a.backend_recipe) &&
         NonzeroDigest(a.backend_build) && NonzeroDigest(a.backend_selection);
}

bool AppendAnalysisExpansion(const ComparisonRegionAnalysis& analysis,
                             LoweredStep::ComparisonOrigins& destination,
                             const components::OwnedComparisonRegionBoundary& boundary) {
  if (analysis.status != ComparisonRegionStatus::kExpand) return true;
  return AppendExpansion(analysis.required_members, destination.expansion, boundary);
}

bool AppendOutsideSuffix(const LoweredStep& source, const ComparisonRegionAnalysis& region,
                         std::size_t begin, std::vector<ir::NodeId>& destination,
                         const components::OwnedComparisonRegionBoundary& boundary) {
  for (std::size_t call = begin; call < source.plan.steps.size(); ++call) {
    if (call >= region.first && call < region.end) {
      call = region.end - 1;
      continue;
    }
    if (!AppendCallOrigins(source, call, destination, boundary)) return false;
  }
  return true;
}

enum class OutsideSequenceStatus : std::uint8_t { kSame, kExpand, kUnproved };

OutsideSequenceStatus SameOutsideCallSequences(
    const LoweredStep& current, const execution::Registry& current_registry,
    const LaunchContext* current_context, const LoweredStep& candidate,
    const execution::Registry& candidate_registry, const LaunchContext* candidate_context,
    const ComparisonRegionAnalysis& current_region,
    const ComparisonRegionAnalysis& candidate_region,
    const components::OwnedComparisonRegionBoundary& boundary,
    ComparisonRemainderCallFactsFn describe_call_facts, std::vector<ir::NodeId>& expansion) {
  std::size_t ci = 0, pi = 0;
  while (true) {
    if (ci == current_region.first) ci = current_region.end;
    if (pi == candidate_region.first) pi = candidate_region.end;
    if (ci == current.plan.steps.size() || pi == candidate.plan.steps.size()) {
      if (ci == current.plan.steps.size() && pi == candidate.plan.steps.size())
        return OutsideSequenceStatus::kSame;
      const auto before = expansion.size();
      if (!AppendOutsideSuffix(current, current_region, ci, expansion, boundary) ||
          !AppendOutsideSuffix(candidate, candidate_region, pi, expansion, boundary))
        return OutsideSequenceStatus::kUnproved;
      return expansion.size() > before ? OutsideSequenceStatus::kExpand
                                       : OutsideSequenceStatus::kUnproved;
    }
    if (!SameCall(current, current_registry, ci, current_context, candidate, candidate_registry, pi,
                  candidate_context, current_region, candidate_region, describe_call_facts)) {
      const auto before = expansion.size();
      // Once ordered correspondence diverges, every remaining outside call is
      // part of the changed closure. Appending only the first mismatching pair
      // could incorrectly hide a shifted or missing suffix.
      if (!AppendOutsideSuffix(current, current_region, ci, expansion, boundary) ||
          !AppendOutsideSuffix(candidate, candidate_region, pi, expansion, boundary))
        return OutsideSequenceStatus::kUnproved;
      return expansion.size() > before ? OutsideSequenceStatus::kExpand
                                       : OutsideSequenceStatus::kUnproved;
    }
    ++ci;
    ++pi;
  }
}

}  // namespace

bool ComparisonWritePreservesLifetime(std::uintptr_t write_address, std::uint64_t write_bytes,
                                      std::size_t call, std::uint8_t subcall,
                                      const ComparisonLifetimeFact& value) {
  return ComparisonWritePreservesLifetimeInternal(write_address, write_bytes, call, subcall, value);
}

ComparisonRegionAnalysis AnalyzeComparisonRegion(
    const LoweredStep& lowered, const components::OwnedComparisonRegionBoundary& boundary,
    const execution::Registry& registry, const kernels::ComparisonRecipeSeal& whole_recipe,
    ComparisonRegionPhase phase, const LaunchContext* launch_context,
    const ComparisonQ8Dispatch* q8_dispatch, ComparisonRegionPurpose purpose,
    ComparisonRemainderCallFactsFn describe_call_facts) {
  ComparisonRegionAnalysis result;
  if (lowered.whole_checkpoint_only) {
    result.reason = "checkpoint-only whole comparison has no local region authority";
    return result;
  }
  if (std::ranges::any_of(lowered.leaves, [](const LoweredLeaf& leaf) {
        return leaf.comparison_staged.has_value();
      })) {
    result.reason = "whole-program staged-input relocation grants no local region authority";
    return result;
  }
  const bool production = purpose != ComparisonRegionPurpose::kExecutableComparison;
  const bool certificate_only = purpose == ComparisonRegionPurpose::kProductionCertificate;
  const auto mmvf_selection_current = [&](const PlanStep& step) {
    if (step.implementation == kDsv4HcPostNormF16Name) {
      if (step.nodes.size() != 2 || !DescribeComparisonAccess(step, ComparisonAccessScope::kLocal))
        return false;
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end()) return false;
      const auto chain = Dsv4HcPostNormF16At(
          lowered.nodes, static_cast<std::size_t>(first - lowered.nodes.begin()));
      return chain && chain->post == step.nodes[0] && chain->norm == step.nodes[1];
    }
    if (NormAddCall(step)) {
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end()) return false;
      const auto chain = GemmaNormAddFusionAt(
          lowered.nodes, static_cast<std::size_t>(first - lowered.nodes.begin()));
      return chain && chain->norm == step.nodes[0] && chain->mul == step.nodes[1] &&
             chain->out == step.nodes[2];
    }
    if (MatMulAddCall(step)) {
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end()) return false;
      const auto chain =
          MulMatAddFusionAt(lowered.nodes, static_cast<std::size_t>(first - lowered.nodes.begin()));
      if (!chain || chain->mul_mat != step.nodes[0] || chain->add != step.nodes[1]) return false;
    }
    if (ComparisonSgemmCall(step)) {
      ComparisonRemainderCallFacts facts;
      return launch_context && describe_call_facts &&
             describe_call_facts(*launch_context, step, facts) &&
             CheckComparisonSgemm(step, facts) &&
             ComparisonLibraryWorkspaceDisjoint(lowered, facts);
    }
    if (!FloatMmvfCall(step) && !F16MatMulGluCall(step) && !MatMulAddCall(step)) return true;
    if (F16MatMulGluCall(step)) {
      const auto first = std::ranges::find(lowered.nodes, step.nodes[0]);
      if (first == lowered.nodes.end()) return false;
      const auto triple =
          MulMatGluFusionAt(lowered.nodes, static_cast<std::size_t>(first - lowered.nodes.begin()));
      if (!triple || triple->gate != step.nodes[0] || triple->up != step.nodes[1] ||
          triple->glu != step.nodes[2])
        return false;
    }
    ComparisonRemainderCallFacts facts;
    return launch_context && describe_call_facts &&
           describe_call_facts(*launch_context, step, facts) && facts.has_mul_mat_selection &&
           facts.mul_mat_vector && facts.scratch_bytes == 0;
  };
  if (!lowered.comparison_origins || !lowered.comparison ||
      whole_recipe.mode != ComparisonRunMode::kFull || !HasIdentity(whole_recipe.policy) ||
      !HasIdentity(whole_recipe.registry) || !HasIdentity(whole_recipe.backend_recipe) ||
      !HasIdentity(whole_recipe.backend_build) || !HasIdentity(whole_recipe.backend_selection) ||
      !HasIdentity(lowered.comparison_origins->backend_device) ||
      !lowered.comparison_origins->descriptor_bindings_verified ||
      (phase == ComparisonRegionPhase::kBound && !lowered.comparison_origins->bindings_verified) ||
      !lowered.routing.empty() || !lowered.plan.regions.empty() ||
      (!production && (lowered.generated_store || !boundary.state().empty())) ||
      !RegionBindingsCurrent(lowered, boundary)) {
    result.reason = "region descriptor or current binding facts are incomplete";
    return result;
  }
  auto& relation = *lowered.comparison_origins;
  if (boundary.source_identity() == 0 || boundary.source_identity() != relation.source_identity ||
      relation.planned_identity == 0 ||
      relation.offsets.size() != relation.planned_nodes.size() + 1 || relation.membership.empty() ||
      relation.original_to_planned.empty()) {
    result.reason = "region boundary or retained source-to-call provenance is stale";
    return result;
  }
  std::ranges::fill(relation.membership, 0);
  relation.expansion.clear();
  relation.value_ports.clear();
  relation.accesses.clear();
  struct ClearAccessAddresses {
    std::vector<ComparisonRegionAccessFact>& accesses;
    ~ClearAccessAddresses() {
      for (auto& access : accesses) access.address = 0;
    }
  } clear_access_addresses{relation.accesses};
  relation.node_local.assign(relation.node_local.size(), kUnset);
  relation.tensor_local.assign(relation.tensor_local.size(), kUnset);
  relation.physical_local.assign(relation.physical_local.size(), kUnset);
  if (relation.projection_serial == std::numeric_limits<std::uint64_t>::max()) {
    result.reason = "region projection serial exhausted";
    return result;
  }
  ++relation.projection_serial;

  const auto members = boundary.nodes();
  if (members.empty()) {
    result.reason = "empty common source region";
    return result;
  }
  for (const auto& member : members) {
    if (!member.original_node.valid() || member.original_node.v >= relation.membership.size() ||
        relation.membership[member.original_node.v] != 0) {
      result.reason = "source boundary contains an invalid or repeated original member";
      return result;
    }
    relation.membership[member.original_node.v] = 1;
  }

  // Applicability describes the actual local call subsequence. It never
  // authorizes replay: executable projection and remainder equality still
  // close a contiguous interval. Consult immutable source members here,
  // never the mutable expansion marks accumulated below.
  const auto includes_call = [&](std::size_t index) {
    if (!certificate_only) return true;
    for (const auto* node : lowered.plan.steps[index].nodes) {
      const auto* physical = PhysicalTensorOf(lowered, node);
      if (!physical || !physical->producer.valid() ||
          physical->producer.v >= relation.offsets.size() - 1)
        return false;  // the complete provenance scan below refuses it
      const auto begin = relation.offsets[physical->producer.v];
      const auto limit = relation.offsets[physical->producer.v + 1];
      if (begin >= limit || limit > relation.source_nodes.size()) return false;
      for (auto offset = begin; offset < limit; ++offset)
        if (std::ranges::any_of(members, [&](const auto& member) {
              return member.original_node == relation.source_nodes[offset];
            }))
          return true;
    }
    return false;
  };

  std::size_t first = lowered.plan.steps.size();
  std::size_t end = 0;
  std::vector<ir::NodeId>& expansion = relation.expansion;
  for (std::size_t i = 0; i < lowered.plan.steps.size(); ++i) {
    bool intersects = false;
    if (!AddCallOrigins(relation, lowered, lowered.plan.steps[i], relation.membership, expansion,
                        intersects, false)) {
      if (production && (ExactSavedStoreCall(lowered, lowered.plan.steps[i]) ||
                         ExactSavedAuxiliaryCall(lowered, lowered.plan.steps[i])))
        continue;
      result.reason = "actual call has no authenticated complete source-origin set";
      return result;
    }
    if (intersects) {
      first = std::min(first, i);
      end = i + 1;
    }
  }
  relation.expansion.clear();
  if (first == lowered.plan.steps.size()) {
    result.reason = "no actual backend call consumes the requested source region";
    return result;
  }
  // Add every selected call's complete origins. Mixed local/outside origin
  // sets still require monotone source closure; authenticated disjoint calls
  // can be skipped only by the applicability certificate.
  std::ranges::fill(relation.membership, 0);
  for (const auto& member : members) relation.membership[member.original_node.v] = 1;
  for (std::size_t i = first; i < end; ++i) {
    if (!includes_call(i)) continue;
    bool intersects = false;
    if (lowered.plan.steps[i].lane != 0 ||
        !AddCallOrigins(relation, lowered, lowered.plan.steps[i], relation.membership, expansion,
                        intersects, true)) {
      result.reason = "region call span is lane-bound or has incomplete provenance";
      return result;
    }
    if (!mmvf_selection_current(lowered.plan.steps[i])) {
      result.reason = "F32 MMVF region selection or zero-scratch recipe changed";
      return result;
    }
    if ((phase == ComparisonRegionPhase::kDescriptor &&
         !DescriptorFirstDomainCall(lowered.plan.steps[i])) ||
        !RegionCallClosed(lowered.plan.steps[i])) {
      result.reason = "actual region call lacks a supported typed access recipe";
      return result;
    }
  }
  std::uint64_t local_call_count = 0;
  for (std::size_t i = first; i < end; ++i)
    if (includes_call(i)) ++local_call_count;
  std::array<ComparisonQ8StepRef, 3> q8_step_refs{};
  std::size_t q8_step_count = 0;
  const ggml_tensor* q8_seed_input = nullptr;
  const ggml_tensor* q8_seed_prepared = nullptr;
  ComparisonQ8SpanRecipe q8_recipe{};
  std::optional<Q8MainPoolDescriptor> q8_pool;
  bool has_q8_recipe = false;
  bool touches_q8 = false;
  for (std::size_t i = first; i < end; ++i) {
    if (!includes_call(i)) continue;
    const auto& step = lowered.plan.steps[i];
    if (!IsQ8Implementation(step.implementation)) continue;
    touches_q8 = true;
    if (step.nodes.size() != 1 || !step.nodes.front()) {
      result.reason = "actual Q8 call has no unique tensor descriptor";
      return result;
    }
    const auto* node = step.nodes.front();
    const ggml_tensor* source = nullptr;
    if (step.implementation == kQuantizeQ8Name) {
      source = node->src[0];
      q8_seed_prepared = node;
    } else if (step.implementation == kMulMatVecQ)
      source = node->src[1];
    else if (step.implementation == kMmvqPreparedName) {
      source = node->src[2];
      q8_seed_prepared = node->src[1];
    }
    if (!source || (q8_seed_input && q8_seed_input != source)) {
      result.reason = "actual Q8 region calls do not share one source input";
      return result;
    }
    q8_seed_input = source;
  }
  if (touches_q8) {
    if (local_call_count != end - first) {
      result.reason = "noncontiguous Q8 applicability requires its complete shared call interval";
      return result;
    }
    if (!launch_context || !q8_dispatch || !q8_dispatch->complete() ||
        !q8_dispatch->has_main_pool() || !lowered.plan.regions.empty()) {
      result.reason = "Q8 region requires its current healthy backend launch and dispatch provider";
      return result;
    }
    auto current_pool = q8_dispatch->describe_main_pool(*launch_context);
    if (!current_pool || current_pool->faulted || current_pool->lanes != 0 ||
        current_pool->base == 0 || current_pool->base % 256 != 0) {
      result.reason = "Q8 region has no healthy aligned current main-pool descriptor";
      return result;
    }
    q8_pool = *current_pool;
    for (std::size_t i = 0; i < lowered.plan.steps.size(); ++i) {
      const auto& step = lowered.plan.steps[i];
      if (!IsQ8Implementation(step.implementation) || step.nodes.size() != 1 || !step.nodes.front())
        continue;
      const auto* node = step.nodes.front();
      const ggml_tensor* source = nullptr;
      if (step.implementation == kQuantizeQ8Name)
        source = node->src[0];
      else if (step.implementation == kMulMatVecQ)
        source = node->src[1];
      else
        source = node->src[2];
      if (source != q8_seed_input) continue;
      if (q8_seed_prepared && step.implementation == kMmvqPreparedName &&
          node->src[1] != q8_seed_prepared)
        continue;
      if (q8_step_count == q8_step_refs.size()) {
        result.reason = "shared Q8 source has too many actual consumers";
        return result;
      }
      q8_step_refs[q8_step_count++] = {.step = &step, .ordinal = i};
      if (step.implementation == kQuantizeQ8Name) {
        if (q8_seed_prepared && q8_seed_prepared != node) {
          result.reason = "shared Q8 source has multiple preparation outputs";
          return result;
        }
        q8_seed_prepared = node;
      }
    }
    // If a prepared Q8 consumer was the seed, its exact producer must be
    // retained too; likewise a producer seed must not omit any consumer.
    if (q8_seed_prepared) {
      const auto producer = std::ranges::find_if(lowered.plan.steps, [&](const auto& step) {
        return step.implementation == kQuantizeQ8Name && step.nodes.size() == 1 &&
               step.nodes.front() == q8_seed_prepared;
      });
      if (producer == lowered.plan.steps.end()) {
        result.reason = "prepared Q8 consumer has no actual retained producer";
        return result;
      }
    }
    // A shared prepared output is a real graph activation. Every physical
    // descriptor consumer must be one of the exact prepared MMVQ calls found
    // above; an unknown helper cannot inherit this producer's recipe.
    if (q8_seed_prepared) {
      for (const auto& physical : lowered.physical_tensors) {
        const auto* node = physical.descriptor;
        if (!node) continue;
        for (const auto* source : node->src) {
          if (source != q8_seed_prepared) continue;
          bool found_consumer = false;
          for (std::size_t i = 0; i < q8_step_count; ++i) {
            const auto* step = q8_step_refs[i].step;
            if (step && step->implementation == kMmvqPreparedName && step->nodes.size() == 1 &&
                step->nodes.front() == node && node->src[1] == q8_seed_prepared &&
                node->src[2] == q8_seed_input)
              found_consumer = true;
          }
          if (!found_consumer) {
            result.reason = "prepared Q8 output has an unsupported actual consumer";
            return result;
          }
        }
      }
    }
    const auto described_q8 = DescribeComparisonQ8Span(
        std::span<const ComparisonQ8StepRef>(q8_step_refs).first(q8_step_count), *launch_context,
        *q8_dispatch);
    if (!described_q8) {
      result.reason = described_q8.error();
      return result;
    }
    q8_recipe = *described_q8;
    has_q8_recipe = true;
    for (std::size_t i = first; i < end; ++i) {
      if (!includes_call(i)) continue;
      const auto& step = lowered.plan.steps[i];
      if (IsQ8Implementation(step.implementation) && !Q8CallForStep(&q8_recipe, step)) {
        result.reason = "actual Q8 call in the body is outside the authenticated shared recipe";
        return result;
      }
    }
    // Add the shared producer and every consumer's complete original origins
    // before accepting the span; the next projection must close the same
    // retained program over their full contiguous interval.
    for (std::size_t i = 0; i < q8_step_count; ++i) {
      bool intersects = false;
      if (!AddCallOrigins(relation, lowered, *q8_step_refs[i].step, relation.membership, expansion,
                          intersects, true)) {
        result.reason = "shared Q8 call has incomplete retained source provenance";
        return result;
      }
    }
    if (q8_recipe.prepared) {
      for (const auto& physical : lowered.physical_tensors) {
        const auto* node = physical.descriptor;
        if (!node) continue;
        for (const auto* source : node->src) {
          if (source != q8_recipe.prepared) continue;
          bool found_consumer = false;
          for (std::size_t i = 0; i < q8_recipe.call_count; ++i) {
            const auto& call = q8_recipe.calls[i];
            if (call.kind == ComparisonQ8CallKind::kMmvqPrepared && call.node == node)
              found_consumer = true;
          }
          if (!found_consumer) {
            result.reason = "Q8 preparation descriptor has an unclosed physical consumer";
            return result;
          }
        }
      }
    }
    std::uint64_t local_scratch = 0;
    for (std::size_t i = 0; i < q8_recipe.call_count; ++i)
      local_scratch = std::max(local_scratch, q8_recipe.calls[i].scratch_bytes);
    if (local_scratch != 0 && !ComparisonPoolDisjoint(lowered, *q8_pool, local_scratch)) {
      result.reason = "Q8 private pool aliases a local activation or external readable range";
      return result;
    }
  }
  // Metadata transforms consume no launch, but remain authenticated members
  // of the source region. Follow selected real-call inputs and the actual
  // mapped output/Keep endpoints through the same bounded typed view proof.
  const auto walk_metadata = [&](auto&& visit) -> bool {
    const auto walk_chain = [&](const ggml_tensor* current, std::string_view implementation,
                                const ggml_tensor* metadata_owner) {
      for (std::size_t depth = 0;
           current && (current->op == GGML_OP_PERMUTE || current->op == GGML_OP_RESHAPE); ++depth) {
        const auto* record = PhysicalTensorOf(lowered, current);
        if (record && std::ranges::any_of(boundary.ports(), [&](const auto& port) {
              return port.kind == components::ComparisonRegionPortKind::kInput &&
                     port.original_tensor.valid() &&
                     port.original_tensor.v < relation.original_to_planned.size() &&
                     relation.original_to_planned[port.original_tensor.v] == record->tensor;
            }))
          break;
        if (depth >= lowered.physical_tensors.size() || !HasTypedDescriptorSchema(*current) ||
            !IsMetadataViewOfSource(current, current->src[0]))
          return false;
        // Only genuine planned reshapes enter this metadata closure.
        // Fused auxiliary views retain their existing checked owner-edge
        // authority below, including the HC post-output flat.
        if (current->op == GGML_OP_RESHAPE && record && !record->tensor.valid() &&
            (implementation == kDsv4HcPostNormF16Name || implementation == kGdnNormGateName ||
             (implementation == kHcCombineName &&
              AuxiliarySourceOrdinal(metadata_owner, current, current->src[0]) == 1)))
          break;
        const auto* input = PhysicalTensorOf(lowered, current->src[0]);
        if (!record || !input || !record->producer.valid() ||
            record->producer.v >= relation.planned_nodes.size() ||
            record->id >= relation.physical_local.size())
          return false;
        const auto& node = relation.planned_nodes[record->producer.v];
        if ((node.key != ir::Primitive(ir::Op::kPermute) &&
             node.key != ir::Primitive(ir::Op::kReshape)) ||
            node.nsrc != 1 || node.nout != 1 || node.sources()[0] != input->tensor ||
            node.outputs()[0] != record->tensor || !record->root.valid() ||
            record->root != input->root || record->descriptor != current ||
            input->descriptor != current->src[0])
          return false;
        if (current->type != current->src[0]->type) return false;
        if (node.key == ir::Primitive(ir::Op::kPermute)) {
          const auto axes = node.Params<ir::PermuteParams>();
          for (std::size_t axis = 0; axis < GGML_MAX_DIMS; ++axis)
            if (current->op != GGML_OP_PERMUTE || current->op_params[axis] != axes.axes[axis])
              return false;
        } else {
          const auto matches_shape = [&](const ggml_tensor* actual, ir::TensorId tensor) {
            if (!tensor.valid() || tensor.v >= relation.planned_tensors.size()) return false;
            const auto& shape = relation.planned_tensors[tensor.v].format.shape;
            for (std::size_t axis = 0; axis < GGML_MAX_DIMS; ++axis) {
              const auto extent = axis < shape.rank ? shape.ne[axis] : 1;
              if (actual->ne[axis] < 1 || static_cast<std::uint64_t>(actual->ne[axis]) != extent)
                return false;
            }
            return true;
          };
          if (!matches_shape(current, record->tensor) ||
              !matches_shape(current->src[0], input->tensor))
            return false;
          if (current->op != GGML_OP_RESHAPE || current->view_offs != current->src[0]->view_offs ||
              !ggml_is_contiguous(current) || !ggml_is_contiguous(current->src[0]) ||
              ggml_nelements(current) != ggml_nelements(current->src[0]) ||
              !HasNoExtraSources(*current, 1))
            return false;
        }
        if (!visit(*record)) return false;
        current = current->src[0];
      }
      return true;
    };
    for (std::size_t call = first; call < end; ++call) {
      if (!includes_call(call)) continue;
      const auto& step = lowered.plan.steps[call];
      for (const auto* actual : step.nodes)
        for (const auto* source : actual->src)
          if (!walk_chain(source, step.implementation, actual)) return false;
    }
    for (const auto& port : boundary.ports()) {
      if (port.kind != components::ComparisonRegionPortKind::kOutput &&
          port.kind != components::ComparisonRegionPortKind::kKeep)
        continue;
      if (!port.original_tensor.valid() ||
          port.original_tensor.v >= relation.original_to_planned.size())
        return false;
      const auto mapped = relation.original_to_planned[port.original_tensor.v];
      if (!mapped.valid()) return false;
      const PhysicalTensor* endpoint = nullptr;
      for (const auto& physical : lowered.physical_tensors) {
        if (physical.tensor != mapped || !physical.descriptor) continue;
        if (endpoint) return false;
        endpoint = &physical;
      }
      if (!endpoint || !walk_chain(endpoint->descriptor, {}, nullptr)) return false;
    }
    return true;
  };

  if (!walk_metadata([&](const PhysicalTensor& record) {
        bool intersects = false;
        return AddDescriptorOrigins(relation, lowered, record.descriptor, relation.membership,
                                    expansion, intersects, true);
      })) {
    result.reason = "actual metadata dependency has incomplete typed provenance";
    return result;
  }
  bool missing_seed = false;
  for (const auto& member : members)
    if (relation.membership[member.original_node.v] == 1) missing_seed = true;
  if (missing_seed) {
    result.reason = "one or more source members have no actual call in the projected span";
    return result;
  }
  std::size_t expansion_count = 0;
  for (std::uint32_t i = 0; i < relation.membership.size(); ++i)
    if (relation.membership[i] == 2) ++expansion_count;
  if (expansion_count != 0) {
    relation.expansion.clear();
    for (std::uint32_t i = 0; i < relation.membership.size(); ++i)
      if (relation.membership[i] == 2) relation.expansion.push_back(ir::NodeId{i});
    result.status = ComparisonRegionStatus::kExpand;
    result.required_members = relation.expansion;
    result.reason = "contiguous actual calls add authenticated source members";
    return result;
  }
  result.first = first;
  result.end = end;

  if (!production) {
    // Executable comparison replays this prefix and must prove its supported
    // side-effect-free schedule. Production certificates never execute it;
    // normal full-program state and input contracts remain owned by Bind.
    for (std::size_t i = 0; i < first; ++i) {
      RegionAccessList accesses;
      if ((phase == ComparisonRegionPhase::kDescriptor &&
           !DescriptorFirstDomainCall(lowered.plan.steps[i])) ||
          lowered.plan.steps[i].lane != 0 || !mmvf_selection_current(lowered.plan.steps[i]) ||
          !DescribeRegionAccesses(lowered.plan.steps[i], accesses)) {
        result.reason = "actual prefix is not a proved side-effect-free activation schedule";
        return result;
      }
      for (std::size_t access_index = 0; access_index < accesses.size; ++access_index) {
        ComparisonRegionAccessFact fact;
        std::uint32_t ignored_physical_count = 0;
        if (!ResolveRegionAccess(lowered, relation, accesses.values[access_index], false,
                                 ignored_physical_count, fact, q8_pool ? &*q8_pool : nullptr)) {
          result.reason = "actual prefix access exceeds its bound activation or mutable input";
          return result;
        }
      }
    }
  }

  // Rebuild exact crossing value facts from the owned canonical boundary and
  // actual bound descriptors. No logical placement offset becomes an address.
  std::uint32_t node_count = 0, tensor_count = 0, physical_count = 0;
  const auto ensure_tensor = [&](auto&& self, std::uint32_t id) -> bool {
    if (id >= relation.tensor_local.size()) return false;
    if (relation.tensor_local[id] != kUnset) return true;
    const auto& descriptor = relation.planned_tensors[id];
    if (descriptor.view_of.valid() && !self(self, descriptor.view_of.v)) return false;
    relation.tensor_local[id] = tensor_count++;
    return true;
  };
  const auto source_ports = boundary.ports();
  for (std::size_t port_index = 0; port_index < source_ports.size(); ++port_index) {
    const auto& port = source_ports[port_index];
    if (port.kind != components::ComparisonRegionPortKind::kInput &&
        port.kind != components::ComparisonRegionPortKind::kOutput &&
        port.kind != components::ComparisonRegionPortKind::kKeep)
      continue;
    if (!port.original_tensor.valid() ||
        port.original_tensor.v >= relation.original_to_planned.size()) {
      result.reason = "boundary value port has no authenticated original tensor";
      return result;
    }
    const auto planned = relation.original_to_planned[port.original_tensor.v];
    if (!planned.valid() || planned.v >= relation.planned_tensors.size() ||
        !ensure_tensor(ensure_tensor, planned.v)) {
      result.reason = "boundary value port was removed without a directional tensor mapping";
      return result;
    }
  }
  for (std::size_t port_index = 0; port_index < source_ports.size(); ++port_index) {
    const auto& port = source_ports[port_index];
    if (port.kind == components::ComparisonRegionPortKind::kStateRead ||
        port.kind == components::ComparisonRegionPortKind::kStateWrite) {
      result.reason = production ? "production certificate body directly reads or writes state"
                                 : "stateful region execution is not in the first region domain";
      return result;
    }
    const bool is_input = port.kind == components::ComparisonRegionPortKind::kInput;
    const bool is_output = port.kind == components::ComparisonRegionPortKind::kOutput ||
                           port.kind == components::ComparisonRegionPortKind::kKeep;
    if (!is_input && !is_output) continue;
    if (!port.original_tensor.valid() ||
        port.original_tensor.v >= relation.original_to_planned.size()) {
      result.reason = "boundary value port has no authenticated original tensor";
      return result;
    }
    const auto planned = relation.original_to_planned[port.original_tensor.v];
    if (!planned.valid() || planned.v >= relation.planned_tensors.size()) {
      result.reason = "boundary value port was removed without a directional tensor mapping";
      return result;
    }
    const PhysicalTensor* physical = nullptr;
    for (const auto& candidate : lowered.physical_tensors) {
      if (candidate.tensor != planned || !candidate.descriptor || !candidate.ir_format) continue;
      if (physical) {
        result.reason = "boundary value maps to multiple physical descriptors";
        return result;
      }
      physical = &candidate;
    }
    const auto* tensor = RegionTensor(boundary, port.canonical_tensor);
    const auto gates_storage =
        physical ? CheckedGdnGatesStorage(lowered, physical->descriptor) : std::nullopt;
    if (!physical || !physical->descriptor->data || !tensor ||
        *physical->ir_format != tensor->format ||
        (!gates_storage && physical->descriptor->view_offs != tensor->view_offset)) {
      result.reason = "actual value port differs from its typed source descriptor";
      return result;
    }
    const auto actual_root = relation.planned_tensors[planned.v].view_of.valid()
                                 ? relation.planned_tensors[planned.v].view_of
                                 : planned;
    if (physical->root != actual_root) {
      result.reason = "actual value port root differs from its authenticated mapped root";
      return result;
    }
    const auto bytes = static_cast<std::uint64_t>(ggml_nbytes(physical->descriptor));
    if (bytes == 0 || relation.value_ports.size() == relation.value_ports.capacity()) {
      result.reason = "actual value port inventory exceeds its paid descriptor bound";
      return result;
    }
    relation.value_ports.push_back(
        {.kind = is_input ? ComparisonRegionValueKind::kInput : ComparisonRegionValueKind::kOutput,
         .ordinal = static_cast<std::uint32_t>(port_index),
         .source_tensor = port.original_tensor,
         .local_tensor = relation.tensor_local[planned.v],
         .role = tensor->role,
         .format = tensor->format,
         .address = reinterpret_cast<std::uintptr_t>(physical->descriptor->data),
         .bytes = bytes,
         .root_offset = ComparisonSemanticViewOffset(lowered, physical->descriptor),
         .read = is_input,
         .written = is_output});
  }
  if (relation.value_ports.empty()) {
    result.reason = "closed region has no actual crossing values";
    return result;
  }

  // Canonicalize the planned call/tensor subgraph in first-use order. Global
  // source and plan IDs are lookup keys only and never enter these digests.
  for (std::size_t i = first; i < end; ++i) {
    if (!includes_call(i)) continue;
    for (const auto* actual : lowered.plan.steps[i].nodes) {
      const auto* physical = PhysicalTensorOf(lowered, actual);
      if (!physical || !physical->producer.valid() ||
          physical->producer.v >= relation.node_local.size()) {
        result.reason = "local plan node has no authenticated actual producer";
        return result;
      }
      if (physical->id >= relation.physical_local.size()) {
        result.reason = "actual descriptor is outside its paid local-ID inventory";
        return result;
      }
      const auto planned_node = physical->producer.v;
      if (relation.node_local[planned_node] == kUnset)
        relation.node_local[planned_node] = node_count++;
      const auto& node = relation.planned_nodes[planned_node];
      for (const auto input : node.sources())
        if (!input.valid() || !ensure_tensor(ensure_tensor, input.v)) {
          result.reason = "local plan source tensor mapping is incomplete";
          return result;
        }
      for (const auto output : node.outputs())
        if (!output.valid() || !ensure_tensor(ensure_tensor, output.v)) {
          result.reason = "local plan output tensor mapping is incomplete";
          return result;
        }
      if (relation.physical_local[physical->id] == kUnset)
        relation.physical_local[physical->id] = physical_count++;
      if (const auto storage = CheckedGdnGatesStorage(lowered, actual)) {
        for (const auto* output : storage->outputs) {
          if (output->id >= relation.physical_local.size()) {
            result.reason = "GDN output view has no paid descriptor ID";
            return result;
          }
          if (relation.physical_local[output->id] == kUnset)
            relation.physical_local[output->id] = physical_count++;
        }
      }
    }
  }
  if (!walk_metadata([&](const PhysicalTensor& physical) {
        if (relation.physical_local[physical.id] != kUnset) return true;
        const auto& node = relation.planned_nodes[physical.producer.v];
        if (relation.node_local[physical.producer.v] == kUnset)
          relation.node_local[physical.producer.v] = node_count++;
        for (const auto input : node.sources())
          if (!input.valid() || !ensure_tensor(ensure_tensor, input.v)) return false;
        for (const auto output : node.outputs())
          if (!output.valid() || !ensure_tensor(ensure_tensor, output.v)) return false;
        relation.physical_local[physical.id] = physical_count++;
        return true;
      })) {
    result.reason = "actual metadata dependency exceeds its paid canonical inventory";
    return result;
  }
  std::uint32_t local_call_ordinal = 0;
  for (std::size_t i = first; i < end; ++i) {
    if (!includes_call(i)) continue;
    RegionAccessList accesses;
    if (!DescribeRegionAccesses(lowered.plan.steps[i], accesses,
                                has_q8_recipe ? &q8_recipe : nullptr)) {
      result.reason = "actual region call has no closed typed read/write schedule";
      return result;
    }
    for (std::size_t access_index = 0; access_index < accesses.size; ++access_index) {
      if (relation.accesses.size() == relation.accesses.capacity()) {
        result.reason = "actual region access inventory exceeds its paid edge bound";
        return result;
      }
      ComparisonRegionAccessFact fact;
      if (!ResolveRegionAccess(lowered, relation, accesses.values[access_index], true,
                               physical_count, fact, q8_pool ? &*q8_pool : nullptr)) {
        result.reason = "actual region read/write exceeds its authenticated storage extent";
        return result;
      }
      fact.call_ordinal = local_call_ordinal;
      fact.subcall_ordinal = accesses.values[access_index].phase;
      relation.accesses.push_back(fact);
    }
    ++local_call_ordinal;
  }
  if (production && !ProductionRegionEffectsClosed(lowered, boundary, relation)) {
    result.reason =
        "production certificate has an unauthenticated input or protected-storage alias";
    return result;
  }

  LocalHash planned_hash("llmp.comparison.region.planned.v2");
  planned_hash.Digest(boundary.digest().logical);
  planned_hash.U64(local_call_count);
  const auto boundary_nodes = boundary.nodes();
  for (std::size_t i = first; i < end; ++i) {
    if (!includes_call(i)) continue;
    const auto& step = lowered.plan.steps[i];
    planned_hash.U8(static_cast<std::uint8_t>(step.operation.op));
    planned_hash.U8(static_cast<std::uint8_t>(step.operation.fused));
    planned_hash.U64(step.nodes.size());
    for (const auto* actual : step.nodes) {
      const auto* physical = PhysicalTensorOf(lowered, actual);
      const auto& node = relation.planned_nodes[physical->producer.v];
      const auto component =
          CanonicalActualComponent(boundary_nodes, relation, physical->producer.v);
      if (!component) {
        result.reason = "actual planned component has no matching authenticated source class";
        return result;
      }
      planned_hash.U32(relation.node_local[physical->producer.v]);
      planned_hash.U8(static_cast<std::uint8_t>(node.key.op));
      planned_hash.U8(static_cast<std::uint8_t>(node.key.fused));
      planned_hash.U32(*component);
      planned_hash.U8(static_cast<std::uint8_t>(node.batch));
      planned_hash.U8(node.nsrc);
      for (const auto source : node.sources()) planned_hash.U32(relation.tensor_local[source.v]);
      planned_hash.U8(node.nout);
      for (const auto output : node.outputs()) planned_hash.U32(relation.tensor_local[output.v]);
      planned_hash.Bytes(node.params);
    }
  }
  planned_hash.Text("llmp.comparison.metadata.v1");
  std::uint64_t metadata_node_count = 0;
  for (std::size_t index = 0; index < relation.node_local.size(); ++index)
    if (relation.node_local[index] != kUnset &&
        (relation.planned_nodes[index].key == ir::Primitive(ir::Op::kPermute) ||
         relation.planned_nodes[index].key == ir::Primitive(ir::Op::kReshape)))
      ++metadata_node_count;
  planned_hash.U64(metadata_node_count);
  for (std::uint32_t local = 0; local < node_count; ++local) {
    const auto found = std::ranges::find(relation.node_local, local);
    const auto index = static_cast<std::size_t>(found - relation.node_local.begin());
    const auto& node = relation.planned_nodes[index];
    if (node.key != ir::Primitive(ir::Op::kPermute) && node.key != ir::Primitive(ir::Op::kReshape))
      continue;
    planned_hash.U32(local);
    planned_hash.U8(static_cast<std::uint8_t>(node.key.op));
    const auto component =
        CanonicalActualComponent(boundary_nodes, relation, static_cast<std::uint32_t>(index));
    if (!component) {
      result.reason = "metadata source component is unauthenticated";
      return result;
    }
    planned_hash.U32(*component);
    for (const auto input : node.sources()) planned_hash.U32(relation.tensor_local[input.v]);
    for (const auto output : node.outputs()) planned_hash.U32(relation.tensor_local[output.v]);
    planned_hash.Bytes(node.params);
  }
  planned_hash.U32(tensor_count);
  for (std::uint32_t local = 0; local < tensor_count; ++local) {
    const auto found = std::ranges::find(relation.tensor_local, local);
    if (found == relation.tensor_local.end()) {
      result.reason = "local tensor canonicalization has a missing ordinal";
      return result;
    }
    const auto tensor_index = static_cast<std::size_t>(found - relation.tensor_local.begin());
    const auto& tensor = relation.planned_tensors[tensor_index];
    const auto canonical_segment = CanonicalLocalSegment(relation, local);
    if (!canonical_segment) {
      result.reason = "local tensor segment has no canonical equivalence class";
      return result;
    }
    planned_hash.U8(static_cast<std::uint8_t>(components::CanonicalComparisonRole(tensor.role)));
    HashTensorFormat(planned_hash, tensor.format);
    for (const auto stride : tensor.nb) planned_hash.U64(stride);
    planned_hash.U8(tensor.view_of.valid() ? 1 : 0);
    if (tensor.view_of.valid()) planned_hash.U32(relation.tensor_local[tensor.view_of.v]);
    planned_hash.U64(tensor.view_offset);
    planned_hash.U32(*canonical_segment);
    planned_hash.U8(static_cast<std::uint8_t>(tensor.binding.index()));
  }

  LocalHash physical_hash("llmp.comparison.region.physical.v4");
  physical_hash.Digest(boundary.digest().external);
  physical_hash.U64(local_call_count);
  std::uint64_t region_pool_draw = 0;
  if (has_q8_recipe)
    for (std::size_t i = 0; i < q8_recipe.call_count; ++i)
      region_pool_draw = std::max(region_pool_draw, q8_recipe.calls[i].scratch_bytes);
  for (std::size_t i = first; i < end; ++i) {
    if (!includes_call(i)) continue;
    const auto& step = lowered.plan.steps[i];
    physical_hash.U8(static_cast<std::uint8_t>(step.operation.op));
    physical_hash.U8(static_cast<std::uint8_t>(step.operation.fused));
    physical_hash.Text(step.implementation);
    const auto implementation = registry.Find(step.implementation);
    if (!implementation) {
      result.reason = "region implementation identity is absent from the current registry";
      return result;
    }
    physical_hash.Digest(registry.identity(*implementation));
    const auto* q8_call = Q8CallForStep(has_q8_recipe ? &q8_recipe : nullptr, step);
    if (IsQ8Implementation(step.implementation) && !q8_call) {
      result.reason = "Q8 physical call is missing its typed shared-dispatch descriptor";
      return result;
    }
    const bool unfused_rms_norm_mul = step.operation == ir::Fused(ir::FusedKind::kRmsNormMul) &&
                                      step.implementation == kRmsNormMulUnfused;
    const std::uint64_t launches = q8_call && q8_call->kind == ComparisonQ8CallKind::kMmvq
                                       ? 2
                                       : (unfused_rms_norm_mul ? 2 : 1);
    physical_hash.U64(launches);
    physical_hash.U64(q8_call ? q8_call->scratch_bytes : 0);
    if (q8_call) {
      if (q8_call->kind != ComparisonQ8CallKind::kQuantize)
        physical_hash.Digest(ComparisonLocalMmvqDispatchDigest(q8_call->mmvq));
      physical_hash.Digest(ComparisonQ8QuantizeDispatchDigest(q8_call->quantizer));
    }
    const auto explicit_access = DescribeComparisonAccess(step, ComparisonAccessScope::kLocal);
    if (explicit_access && explicit_access->dispatch) {
      physical_hash.Text("llmp.ggml.explicit-dispatch.v1");
      physical_hash.U32(*explicit_access->dispatch);
    }
    if (step.implementation == kGdnGatesName && step.nodes.size() == 1) {
      const auto storage = CheckedGdnGatesStorage(lowered, step.nodes[0]);
      if (!storage) {
        result.reason = "GDN packed outputs lack their exact owned storage relation";
        return result;
      }
      physical_hash.Text("llmp.gdn.gates.packed-output-views.v1");
      physical_hash.U64(storage->outputs.size());
      for (const auto* output : storage->outputs) {
        const auto* view = output->descriptor;
        physical_hash.U32(relation.physical_local[output->id]);
        physical_hash.U32(relation.tensor_local[output->tensor.v]);
        physical_hash.U32(relation.tensor_local[output->root.v]);
        physical_hash.U32(static_cast<std::uint32_t>(view->op));
        physical_hash.U32(static_cast<std::uint32_t>(view->type));
        HashTypedOperationParameters(physical_hash, *view);
        for (const auto extent : view->ne) physical_hash.U64(static_cast<std::uint64_t>(extent));
        for (const auto stride : view->nb) physical_hash.U64(stride);
        physical_hash.U64(view->view_offs);
        physical_hash.U32(relation.physical_local[storage->parent->id]);
      }
    }
    physical_hash.U64(step.nodes.size());
    for (std::size_t actual_index = 0; actual_index < step.nodes.size(); ++actual_index) {
      const auto* actual = step.nodes[actual_index];
      const auto* physical = PhysicalTensorOf(lowered, actual);
      if (!physical || !physical->descriptor || !HasTypedDescriptorSchema(*actual)) {
        result.reason = "region physical operation has no supported typed parameter schema";
        return result;
      }
      physical_hash.U32(relation.physical_local[physical->id]);
      physical_hash.U32(static_cast<std::uint32_t>(actual->op));
      physical_hash.U32(static_cast<std::uint32_t>(actual->type));
      HashTypedOperationParameters(physical_hash, *actual);
      for (const auto extent : actual->ne) physical_hash.U64(static_cast<std::uint64_t>(extent));
      for (const auto stride : actual->nb) physical_hash.U64(static_cast<std::uint64_t>(stride));
      physical_hash.U64(static_cast<std::uint64_t>(actual->view_offs));
      physical_hash.U32(GGML_MAX_SRC);
      for (std::uint32_t source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
        const auto* source = actual->src[source_index];
        physical_hash.U8(source ? 1 : 0);
        if (!source) continue;
        const auto* source_record = PhysicalTensorOf(lowered, source);
        if (!source_record || (source_record->root.valid() &&
                               source_record->root.v >= relation.tensor_local.size())) {
          result.reason = "actual GGML source has no local descriptor record";
          return result;
        }
        if (!source_record->root.valid() && step.implementation == kDsv4HcPostNormF16Name &&
            actual_index == 1 && source_index == 0) {
          const auto* post_record = PhysicalTensorOf(lowered, step.nodes[0]);
          if (!post_record || source != actual->src[0] || source->op != GGML_OP_RESHAPE ||
              source->src[0] != step.nodes[0] || source->view_src != step.nodes[0] ||
              source->view_offs != 0 || source_record->tensor.valid() ||
              source_record->producer.valid() || post_record->producer != physical->producer ||
              !post_record->tensor.valid() || !post_record->root.valid() ||
              source_record->id >= relation.physical_local.size() ||
              !HasTypedDescriptorSchema(*source) ||
              !IsMetadataViewOfSource(source, step.nodes[0])) {
            result.reason = "HC auxiliary flat is not the exact owned post-output reshape";
            return result;
          }
          if (relation.physical_local[source_record->id] == kUnset)
            relation.physical_local[source_record->id] = physical_count++;
        }
        const auto source_physical_local = source_record->id < relation.physical_local.size()
                                               ? relation.physical_local[source_record->id]
                                               : kUnset;
        bool internal_descriptor_edge = false;
        bool metadata_view_edge = false;
        const PhysicalTensor* edge_record = source_record;
        const auto auxiliary_source = AuxiliarySourceOrdinal(actual, source, source->src[0]);
        if (!source_record->root.valid() && actual_index == 0 && auxiliary_source &&
            source_index == *auxiliary_source &&
            ((step.operation == ir::Fused(ir::FusedKind::kNormGate) &&
              step.implementation == kGdnNormGateName) ||
             (step.operation == ir::Fused(ir::FusedKind::kHcCombine) &&
              step.implementation == kHcCombineName))) {
          const auto* owner = PhysicalTensorOf(lowered, actual);
          edge_record = PhysicalTensorOf(lowered, source->src[0]);
          if (!owner || !owner->producer.valid() || !edge_record || !edge_record->tensor.valid() ||
              !edge_record->root.valid() || edge_record->tensor.v >= relation.tensor_local.size() ||
              edge_record->root.v >= relation.tensor_local.size() ||
              owner->producer.v >= relation.planned_nodes.size() ||
              relation.planned_nodes[owner->producer.v].sources().size() <= *auxiliary_source ||
              relation.planned_nodes[owner->producer.v].sources()[*auxiliary_source] !=
                  edge_record->tensor ||
              relation.planned_nodes[owner->producer.v].key != step.operation ||
              (*auxiliary_source == 1 && source_record->producer.valid())) {
            result.reason = "auxiliary reshape is not the exact selected planned source view";
            return result;
          }
          metadata_view_edge = true;
        }
        if (!source_record->root.valid() && step.implementation == kDsv4HcPostNormF16Name &&
            actual_index == 1 && source_index == 0) {
          edge_record = PhysicalTensorOf(lowered, step.nodes[0]);
          metadata_view_edge = true;
        }
        const auto source_root_local =
            edge_record->root.valid() ? relation.tensor_local[edge_record->root.v] : kUnset;
        if (edge_record->root.valid() && source_root_local == kUnset) {
          result.reason = "actual GGML source root is outside the local tensor closure";
          return result;
        }
        if (!edge_record->root.valid() && !metadata_view_edge) {
          const bool checked_norm_rope =
              step.operation == ir::Fused(ir::FusedKind::kRmsNormMulRope) &&
              step.implementation == kGemmaNormRopeName && step.nodes.size() == 3 &&
              CheckGemmaNormRope(step.nodes[0], step.nodes[1], step.nodes[2]).has_value();
          const bool checked_explicit_elision =
              explicit_access &&
              std::ranges::contains(
                  std::span(explicit_access->elided).first(explicit_access->elided_count), source);
          if (!checked_explicit_elision && !checked_norm_rope && !F16MatMulGluCall(step) &&
              !NormAddCall(step) && !MatMulAddCall(step)) {
            result.reason = "only checked fused recipes may reference an elided descriptor";
            return result;
          }
          // Some supported fused calls retain descriptors for an elided
          // intermediate (for example RMSNorm -> Mul). Such an edge is local
          // only when it names the exact earlier descriptor in this same
          // authenticated call, with the same producer and paid local ID.
          // It has no IR tensor or allocation root, so never invent one.
          for (std::size_t earlier = 0; earlier < actual_index; ++earlier) {
            const auto* prior = step.nodes[earlier];
            if (prior != source) continue;
            const auto* prior_record = PhysicalTensorOf(lowered, prior);
            if (prior_record && prior_record->id == source_record->id &&
                prior_record->producer == source_record->producer &&
                source_physical_local != kUnset) {
              internal_descriptor_edge = true;
              break;
            }
          }
          if (!internal_descriptor_edge) {
            result.reason = "actual GGML source is not an earlier descriptor in this closed call";
            return result;
          }
        }
        const auto source_producer_local =
            edge_record->producer.valid() && edge_record->producer.v < relation.node_local.size()
                ? relation.node_local[edge_record->producer.v]
                : kUnset;
        std::uint32_t crossing_port = kUnset;
        if (!internal_descriptor_edge && edge_record->producer.valid() &&
            source_producer_local == kUnset) {
          for (const auto& port : relation.value_ports) {
            if (port.kind == ComparisonRegionValueKind::kInput &&
                port.address == reinterpret_cast<std::uintptr_t>(source->data) &&
                port.bytes == static_cast<std::uint64_t>(ggml_nbytes(source)) &&
                port.root_offset == ComparisonSemanticViewOffset(lowered, source) &&
                (port.local_tensor == source_root_local ||
                 (edge_record->tensor.valid() &&
                  edge_record->tensor.v < relation.tensor_local.size() &&
                  port.local_tensor == relation.tensor_local[edge_record->tensor.v]))) {
              // Equivalent source-boundary ports may repeat the same tensor
              // across members. The canonical global boundary ordinal picks
              // the first such port while the port table below retains all
              // of them; pointer identity never resolves the semantic port.
              crossing_port =
                  crossing_port == kUnset ? port.ordinal : std::min(crossing_port, port.ordinal);
            }
          }
          if (crossing_port == kUnset) {
            result.reason = "actual GGML source producer is neither local nor a proved input port";
            return result;
          }
        }
        physical_hash.U32(source_producer_local);
        physical_hash.U8(metadata_view_edge ? 2 : (internal_descriptor_edge ? 1 : 0));
        if (internal_descriptor_edge || metadata_view_edge)
          physical_hash.U32(source_physical_local);
        if (metadata_view_edge) physical_hash.U32(relation.tensor_local[edge_record->tensor.v]);
        physical_hash.U32(source_root_local);
        physical_hash.U32(crossing_port);
        physical_hash.U32(static_cast<std::uint32_t>(source->type));
        for (const auto extent : source->ne) physical_hash.U64(static_cast<std::uint64_t>(extent));
        for (const auto stride : source->nb) physical_hash.U64(static_cast<std::uint64_t>(stride));
        physical_hash.U64(static_cast<std::uint64_t>(source->view_offs));
      }
      physical_hash.U8(actual->view_src ? 1 : 0);
      if (actual->view_src) {
        const auto* root = PhysicalTensorOf(lowered, actual->view_src);
        if (!root) {
          result.reason = "actual view root has no local descriptor record";
          return result;
        }
        const auto view_producer_local =
            root->producer.valid() && root->producer.v < relation.node_local.size()
                ? relation.node_local[root->producer.v]
                : kUnset;
        std::uint32_t crossing_port = kUnset;
        if (root->producer.valid() && view_producer_local == kUnset) {
          const auto root_local = root->root.valid() && root->root.v < relation.tensor_local.size()
                                      ? relation.tensor_local[root->root.v]
                                      : kUnset;
          for (const auto& port : relation.value_ports) {
            if (port.kind == ComparisonRegionValueKind::kInput &&
                port.address == reinterpret_cast<std::uintptr_t>(actual->view_src->data) &&
                port.bytes == static_cast<std::uint64_t>(ggml_nbytes(actual->view_src)) &&
                port.root_offset == ComparisonSemanticViewOffset(lowered, actual->view_src) &&
                (port.local_tensor == root_local ||
                 (root->tensor.valid() && root->tensor.v < relation.tensor_local.size() &&
                  port.local_tensor == relation.tensor_local[root->tensor.v]))) {
              crossing_port =
                  crossing_port == kUnset ? port.ordinal : std::min(crossing_port, port.ordinal);
            }
          }
          if (crossing_port == kUnset) {
            result.reason = "actual view root is neither local nor a proved input port";
            return result;
          }
        }
        physical_hash.U32(view_producer_local);
        physical_hash.U32(crossing_port);
      }
    }
  }
  physical_hash.Text("llmp.comparison.metadata.v1");
  const auto inside_metadata = [&](const PhysicalTensor* record) {
    return record && record->descriptor &&
           (record->descriptor->op == GGML_OP_PERMUTE ||
            record->descriptor->op == GGML_OP_RESHAPE) &&
           record->id < relation.physical_local.size() &&
           relation.physical_local[record->id] != kUnset && record->producer.valid() &&
           record->producer.v < relation.node_local.size() &&
           relation.node_local[record->producer.v] != kUnset && record->tensor.valid() &&
           record->root.valid() &&
           (relation.planned_nodes[record->producer.v].key == ir::Primitive(ir::Op::kPermute) ||
            relation.planned_nodes[record->producer.v].key == ir::Primitive(ir::Op::kReshape));
  };
  std::uint64_t metadata_descriptor_count = 0;
  for (const auto& record : lowered.physical_tensors)
    if (inside_metadata(&record)) ++metadata_descriptor_count;
  physical_hash.U64(metadata_descriptor_count);
  for (std::uint32_t local = 0; local < physical_count; ++local) {
    const auto found = std::ranges::find(relation.physical_local, local);
    if (found == relation.physical_local.end()) {
      result.reason = "metadata physical ordinal is missing";
      return result;
    }
    const auto id = static_cast<std::uint32_t>(found - relation.physical_local.begin());
    const auto* record = PhysicalTensorForId(lowered, id);
    if (!inside_metadata(record)) continue;
    const auto* actual = record->descriptor;
    const auto* source = PhysicalTensorOf(lowered, actual->src[0]);
    if (!source || !source->tensor.valid() || !source->root.valid() ||
        source->tensor.v >= relation.tensor_local.size() ||
        source->root.v >= relation.tensor_local.size() ||
        relation.tensor_local[source->tensor.v] == kUnset ||
        relation.tensor_local[source->root.v] == kUnset) {
      result.reason = "metadata source is outside the authenticated tensor closure";
      return result;
    }
    if ((!source->producer.valid() || source->producer.v >= relation.node_local.size() ||
         relation.node_local[source->producer.v] == kUnset) &&
        !std::ranges::any_of(relation.value_ports, [&](const auto& port) {
          return port.kind == ComparisonRegionValueKind::kInput &&
                 port.local_tensor == relation.tensor_local[source->tensor.v] &&
                 port.address == reinterpret_cast<std::uintptr_t>(source->descriptor->data) &&
                 port.bytes == ggml_nbytes(source->descriptor) &&
                 port.root_offset == ComparisonSemanticViewOffset(lowered, source->descriptor);
        })) {
      result.reason = "metadata source is neither local nor a proved input port";
      return result;
    }
    physical_hash.U32(local);
    physical_hash.U32(relation.node_local[record->producer.v]);
    physical_hash.U32(static_cast<std::uint32_t>(actual->op));
    physical_hash.U32(static_cast<std::uint32_t>(actual->type));
    HashTypedOperationParameters(physical_hash, *actual);
    for (const auto extent : actual->ne) physical_hash.U64(static_cast<std::uint64_t>(extent));
    for (const auto stride : actual->nb) physical_hash.U64(static_cast<std::uint64_t>(stride));
    physical_hash.U64(actual->view_offs);
    physical_hash.U32(relation.tensor_local[record->tensor.v]);
    physical_hash.U32(relation.tensor_local[record->root.v]);
    physical_hash.U32(relation.tensor_local[source->tensor.v]);
    physical_hash.U32(relation.tensor_local[source->root.v]);
  }
  physical_hash.U64(region_pool_draw);
  for (const auto& port : relation.value_ports) {
    physical_hash.U8(static_cast<std::uint8_t>(port.kind));
    physical_hash.U32(port.ordinal);
    physical_hash.U32(port.local_tensor);
    physical_hash.U8(static_cast<std::uint8_t>(components::CanonicalComparisonRole(port.role)));
    HashTensorFormat(physical_hash, port.format);
    physical_hash.U64(port.bytes);
    physical_hash.U64(port.root_offset);
  }

  // The typed dispatch walk above is the access authority for this domain.
  // Seal its exact ordered read/write inventory and address-independent alias
  // relations; never serialize a device pointer or arena-wide placement.
  physical_hash.U64(relation.accesses.size());
  for (std::size_t i = 0; i < relation.accesses.size(); ++i) {
    const auto& access = relation.accesses[i];
    physical_hash.U32(access.call_ordinal);
    physical_hash.U8(access.subcall_ordinal);
    physical_hash.U32(access.physical_local);
    physical_hash.U32(access.tensor_local);
    physical_hash.U32(access.root_local);
    if (access.domain == ComparisonRegionAccessDomain::kPool) {
      physical_hash.U8(0xff);
    } else {
      const auto root = std::ranges::find(relation.tensor_local, access.root_local);
      if (access.root_local == kUnset || root == relation.tensor_local.end()) {
        result.reason = "typed access has no authenticated semantic storage root";
        return result;
      }
      physical_hash.U8(static_cast<std::uint8_t>(components::CanonicalComparisonRole(
          relation.planned_tensors[static_cast<std::size_t>(root - relation.tensor_local.begin())]
              .role)));
    }
    physical_hash.U64(access.bytes);
    physical_hash.U64(access.required_alignment);
    physical_hash.U64(access.root_offset);
    physical_hash.U8(access.read ? 1 : 0);
    physical_hash.U8(access.written ? 1 : 0);
    std::uintptr_t access_end = 0;
    if (access.address == 0 || access.bytes == 0 ||
        __builtin_add_overflow(access.address, static_cast<std::uintptr_t>(access.bytes),
                               &access_end)) {
      result.reason = "typed access interval overflowed before recipe sealing";
      return result;
    }
    for (std::size_t j = 0; j < i; ++j) {
      const auto& prior = relation.accesses[j];
      std::uintptr_t prior_end = 0;
      if (prior.address == 0 || prior.bytes == 0 ||
          __builtin_add_overflow(prior.address, static_cast<std::uintptr_t>(prior.bytes),
                                 &prior_end)) {
        result.reason = "prior typed access interval overflowed before recipe sealing";
        return result;
      }
      const bool overlaps = access.address < prior_end && prior.address < access_end;
      physical_hash.U8(overlaps ? 1 : 0);
      if (overlaps)
        HashRelativeDelta(physical_hash, static_cast<std::uint64_t>(access.address),
                          static_cast<std::uint64_t>(prior.address));
    }
  }

  result.seal.version = 3;
  result.seal.state = ComparisonProofState::kBoundVerified;
  result.seal.missing = ComparisonMissing::kNone;
  result.seal.subject = boundary.digest().logical;
  result.seal.boundary = boundary.digest().combined;
  result.seal.planned_region = planned_hash.Finish();
  result.seal.physical_recipe = physical_hash.Finish();
  result.seal.policy = whole_recipe.policy;
  result.seal.registry = whole_recipe.registry;
  result.seal.backend_recipe = whole_recipe.backend_recipe;
  result.seal.backend_build = whole_recipe.backend_build;
  result.seal.backend_selection = whole_recipe.backend_selection;
  result.seal.backend_device = relation.backend_device;
  result.seal.state = phase == ComparisonRegionPhase::kDescriptor
                          ? ComparisonProofState::kDescriptorVerified
                          : ComparisonProofState::kBoundVerified;
  // Repetition consumes the same prepared values again. Legal dead-input
  // reuse remains executable once, but cannot authorize repeated invocations.
  result.inputs_preserved = phase == ComparisonRegionPhase::kBound;
  for (const auto& port : relation.value_ports) {
    if (port.kind != ComparisonRegionValueKind::kInput) continue;
    for (const auto& access : relation.accesses) {
      if (!access.written) continue;
      const auto overlap = IntervalsOverlap(access.address, access.bytes, port.address, port.bytes);
      if (!overlap || *overlap) result.inputs_preserved = false;
    }
  }
  result.status = ComparisonRegionStatus::kClosed;
  result.value_ports = relation.value_ports;
  return result;
}

ComparisonRemainderResult CompareComparisonRemainder(
    const LoweredStep& current, const execution::Registry& current_registry,
    const kernels::ComparisonRecipeSeal& current_recipe, const LaunchContext* current_context,
    const LoweredStep& candidate, const execution::Registry& candidate_registry,
    const kernels::ComparisonRecipeSeal& candidate_recipe, const LaunchContext* candidate_context,
    const components::OwnedComparisonRegionBoundary& boundary,
    kernels::ComparisonCertificatePhase phase, ComparisonRemainderCallFactsFn describe_call_facts) {
  struct ClearLifetimeAddresses {
    const LoweredStep& current;
    const LoweredStep& candidate;
    ~ClearLifetimeAddresses() {
      const auto clear_value_ports = [](const LoweredStep& lowered) {
        if (!lowered.comparison_origins) return;
        for (auto& port : lowered.comparison_origins->value_ports) port.address = 0;
      };
      if (current.comparison_origins) {
        std::ranges::fill(current.comparison_origins->lifetimes, ComparisonLifetimeFact{});
        clear_value_ports(current);
      }
      if (&current != &candidate && candidate.comparison_origins) {
        std::ranges::fill(candidate.comparison_origins->lifetimes, ComparisonLifetimeFact{});
        clear_value_ports(candidate);
      }
    }
  } clear_lifetime_addresses{current, candidate};
  if (!current.comparison_origins || !candidate.comparison_origins || !current.comparison ||
      !candidate.comparison || !current_context || !candidate_context ||
      !SameEnvironment(current_recipe, candidate_recipe))
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = "remainder programs lack one authenticated source and backend environment"};
  const auto& current_origins = *current.comparison_origins;
  const auto& candidate_origins = *candidate.comparison_origins;
  if (current_origins.source_identity == 0 ||
      current_origins.source_identity != boundary.source_identity() ||
      current_origins.source_identity != candidate_origins.source_identity ||
      current_origins.planned_identity == 0 || candidate_origins.planned_identity == 0 ||
      !current_origins.source_boundary_verified || !candidate_origins.source_boundary_verified ||
      !current_origins.descriptor_bindings_verified ||
      !candidate_origins.descriptor_bindings_verified ||
      current_origins.backend_device != candidate_origins.backend_device ||
      current_origins.active_stage != candidate_origins.active_stage ||
      !SameExternalBindingContracts(current_origins, candidate_origins) ||
      current_recipe.subject != candidate_recipe.subject ||
      current_recipe.boundary != candidate_recipe.boundary || current.routing.size() != 0 ||
      candidate.routing.size() != 0 || !current.plan.regions.empty() ||
      !candidate.plan.regions.empty() || current.selected.size() != 0 ||
      candidate.selected.size() != 0 ||
      (phase == kernels::ComparisonCertificatePhase::kBound &&
       (!current_origins.bindings_verified || !candidate_origins.bindings_verified)))
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = "remainder source, phase, or retained execution facts are incomplete"};
  if (!SameProgramSetup(current, candidate))
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = "remainder programs have different public slots or execution setup"};

  const auto phase_for_region = phase == kernels::ComparisonCertificatePhase::kDescriptor
                                    ? ComparisonRegionPhase::kDescriptor
                                    : ComparisonRegionPhase::kBound;
  auto& expansion = current.comparison_origins->expansion;
  expansion.clear();
  const auto current_region = AnalyzeComparisonRegion(
      current, boundary, current_registry, current_recipe, phase_for_region, current_context,
      nullptr, ComparisonRegionPurpose::kProductionRemainder, describe_call_facts);
  if (current_region.status == ComparisonRegionStatus::kUnproved)
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = current_region.reason};
  if (!AppendAnalysisExpansion(current_region, *current.comparison_origins, boundary))
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = "current remainder expansion exceeded its paid source inventory"};

  const bool same_program = &current == &candidate;
  if (same_program && current_context != candidate_context)
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = "one retained program was presented with two launch contexts"};
  ComparisonRegionAnalysis candidate_region;
  if (same_program) {
    // Reanalyzing the same object would invalidate its borrowed expansion and
    // value-port spans. One authenticated analysis is sufficient for the
    // reflexive comparison; the actual outside calls are still checked below.
    candidate_region = current_region;
  } else {
    candidate_region =
        AnalyzeComparisonRegion(candidate, boundary, candidate_registry, candidate_recipe,
                                phase_for_region, candidate_context, nullptr,
                                ComparisonRegionPurpose::kProductionRemainder, describe_call_facts);
    if (candidate_region.status == ComparisonRegionStatus::kUnproved)
      return {.status = kernels::ComparisonRemainderStatus::kUnproved,
              .reason = candidate_region.reason};
    if (!AppendAnalysisExpansion(candidate_region, *current.comparison_origins, boundary))
      return {.status = kernels::ComparisonRemainderStatus::kUnproved,
              .reason = "candidate remainder expansion exceeded the current paid source inventory"};
  }
  if (current_region.status == ComparisonRegionStatus::kExpand ||
      candidate_region.status == ComparisonRegionStatus::kExpand)
    return {.status = kernels::ComparisonRemainderStatus::kExpand, .required_members = expansion};

  if (current_region.first > current.plan.steps.size() ||
      current_region.end > current.plan.steps.size() ||
      candidate_region.first > candidate.plan.steps.size() ||
      candidate_region.end > candidate.plan.steps.size() ||
      current_region.first >= current_region.end || candidate_region.first >= candidate_region.end)
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = "closed remainder region has an invalid actual call span"};
  if (current_recipe.lanes != candidate_recipe.lanes || current_recipe.lanes != 0 ||
      current_recipe.lane_scratch_bytes != candidate_recipe.lane_scratch_bytes ||
      current_recipe.pool_bytes != candidate_recipe.pool_bytes ||
      current_recipe.private_pool_bytes != candidate_recipe.private_pool_bytes)
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = "remainder programs differ in lane or shared scratch setup"};

  const auto outside = SameOutsideCallSequences(
      current, current_registry, current_context, candidate, candidate_registry, candidate_context,
      current_region, candidate_region, boundary, describe_call_facts, expansion);
  if (outside == OutsideSequenceStatus::kExpand)
    return {.status = kernels::ComparisonRemainderStatus::kExpand, .required_members = expansion};
  if (outside == OutsideSequenceStatus::kUnproved)
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = "outside calls are not covered by the finite typed remainder schema"};
  std::string_view access_refusal;
  if (!SameComparisonAccessTopology(current, current_region, current_context, candidate,
                                    candidate_region, candidate_context, describe_call_facts,
                                    &access_refusal))
    return {.status = kernels::ComparisonRemainderStatus::kUnproved,
            .reason = access_refusal.empty()
                          ? "body and remainder actual access or interference relations differ"
                          : access_refusal};
  return {.status = kernels::ComparisonRemainderStatus::kSame};
}

}  // namespace llmp::kernels::ggml
