// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/comparison_pair.h"

#include <algorithm>
#include <bit>
#include <functional>
#include <limits>
#include <new>
#include <numeric>
#include <ranges>
#include <utility>

#include "base/sha256.h"
#include "components/comparison_boundary.h"
#include "components/graph_io.h"
#include "execution/fusion_ranks.h"
#include "format/tensor_format.h"
#include "planner/placement.h"

namespace llmp::engine {
namespace {

constexpr std::uint64_t kAllocatorEnvelope = 2 * sizeof(void*) + alignof(std::max_align_t);

bool AddBytes(std::uint64_t amount, std::uint64_t& total) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - total) return false;
  total += amount;
  return true;
}

template <class T>
bool VectorBytes(const std::vector<T>& values, std::uint64_t& total) {
  if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(T)) return false;
  const auto bytes = static_cast<std::uint64_t>(values.capacity()) * sizeof(T);
  return AddBytes(bytes, total) && (values.capacity() == 0 || AddBytes(kAllocatorEnvelope, total));
}

bool StringBytes(const std::string& value, std::uint64_t& total) {
  if (value.capacity() >= std::numeric_limits<std::uint64_t>::max() ||
      !AddBytes(static_cast<std::uint64_t>(value.capacity()) + 1, total))
    return false;
  // Cover a heap-backed payload and its allocator metadata conservatively;
  // short strings are overcharged, never undercounted.
  return AddBytes(kAllocatorEnvelope, total);
}

ComparisonPairFailure Failure(ComparisonPairFailure::Code code, std::string_view detail) {
  return {code, detail};
}

std::optional<const components::StepInput*> FindTypedInput(const components::StepGraph& source,
                                                           ir::TensorId root);

template <class T>
void HashFact(base::Sha256& hash, const T& value) {
  hash.Update(std::as_bytes(std::span(&value, 1)));
}

void HashText(base::Sha256& hash, std::string_view value) {
  const auto size = static_cast<std::uint64_t>(value.size());
  HashFact(hash, size);
  hash.Update(value);
}

base::Sha256Digest SourceMetadataDigest(const components::StepGraph& source,
                                        std::uint16_t active_stage) {
  base::Sha256 hash;
  hash.Update("llmp-comparison-source-bindings-v1");
  HashFact(hash, active_stage);
  const auto inputs = static_cast<std::uint64_t>(source.inputs.size());
  const auto weights = static_cast<std::uint64_t>(source.weights.size());
  const auto states = static_cast<std::uint64_t>(source.state.size());
  const auto saves = static_cast<std::uint64_t>(source.saves.size());
  HashFact(hash, inputs);
  for (const auto& input : source.inputs) {
    HashFact(hash, input.tensor.v);
    HashFact(hash, input.kind);
    HashFact(hash, input.segment);
    HashFact(hash, input.cells.capacity);
    HashFact(hash, input.cells.ring);
    HashFact(hash, input.cells.frozen);
    HashFact(hash, input.cells.fixed);
    HashFact(hash, input.cells.block);
    HashFact(hash, input.cells.per_row);
    HashFact(hash, input.cells.first_row);
    HashFact(hash, input.cells.positions);
    HashFact(hash, input.cells.recent);
    HashFact(hash, input.cells.cell_pad);
    HashFact(hash, input.read);
    HashFact(hash, input.sections);
    HashFact(hash, input.component);
    HashFact(hash, input.layer);
    HashText(hash, input.what);
    const auto host_def = reinterpret_cast<std::uintptr_t>(input.host.def);
    HashFact(hash, host_def);
    HashFact(hash, input.host.ratio);
    HashFact(hash, input.host.overlap);
    HashFact(hash, input.ratio);
    HashFact(hash, input.overlap);
    HashFact(hash, input.ring);
    HashFact(hash, input.visible_capacity);
  }
  HashFact(hash, weights);
  for (const auto& weight : source.weights) {
    HashFact(hash, weight.tensor.v);
    HashFact(hash, weight.ref.stage);
    HashFact(hash, weight.ref.component);
    HashFact(hash, weight.ref.role);
    HashFact(hash, weight.ref.layer);
    HashFact(hash, weight.ref.expert_array);
    const auto parts = static_cast<std::uint64_t>(weight.parts.size());
    HashFact(hash, parts);
    for (const auto& [name, role] : weight.parts) {
      HashText(hash, name);
      HashFact(hash, role);
    }
  }
  HashFact(hash, states);
  for (const auto& state : source.state) {
    HashFact(hash, state.tensor.v);
    HashFact(hash, state.ref.stage);
    HashFact(hash, state.ref.tensor);
    HashFact(hash, state.ref.layer);
    HashFact(hash, state.ref.segment);
    HashFact(hash, state.item);
    HashFact(hash, state.borrowed);
    HashFact(hash, state.tap);
    HashFact(hash, state.source);
    HashFact(hash, state.read);
    HashFact(hash, state.written);
  }
  HashFact(hash, saves);
  for (const auto& save : source.saves) {
    HashFact(hash, save.layer);
    HashFact(hash, save.component);
    HashText(hash, save.what);
    HashFact(hash, save.segment);
    HashFact(hash, save.tensor.v);
    HashFact(hash, save.rows.v);
  }
  const auto owner_products = static_cast<std::uint64_t>(source.owner_products.size());
  HashFact(hash, owner_products);
  for (const auto product : source.owner_products) HashFact(hash, product.v);
  HashFact(hash, source.logits.v);
  HashFact(hash, source.greedy.v);
  HashFact(hash, source.features.v);
  return hash.Finish();
}

bool ExactSourceInventory(const components::StepGraph& source, const components::GraphBuildIO& io) {
  if (source.inputs.size() != io.inputs.size()) return false;
  for (const auto& input : source.inputs)
    if (!input.tensor.valid() || input.tensor.v >= source.graph.tensor_count()) return false;
  for (const auto input : io.inputs) {
    if (!input.valid() || input.v >= source.graph.tensor_count()) return false;
    if (!FindTypedInput(source, source.graph.root(input))) return false;
    if (std::ranges::count(source.inputs, source.graph.root(input),
                           [&](const components::StepInput& typed) {
                             return source.graph.root(typed.tensor);
                           }) != 1)
      return false;
  }
  // Validate the complete ID list before any root lookup. A valid early item
  // must not cause graph access through a malformed later item.
  for (const auto& weight : source.weights)
    if (!weight.tensor.valid() || weight.tensor.v >= source.graph.tensor_count()) return false;
  for (std::size_t i = 0; i < source.weights.size(); ++i) {
    const auto root = source.graph.root(source.weights[i].tensor);
    if (std::ranges::count(source.weights, root, [&](const components::StepWeight& weight) {
          return source.graph.root(weight.tensor);
        }) != 1)
      return false;
    for (std::size_t j = 0; j < i; ++j)
      if (source.graph.root(source.weights[j].tensor) == root) return false;
  }
  for (std::size_t i = 0; i < source.state.size(); ++i) {
    if (!source.state[i].tensor.valid() || source.state[i].tensor.v >= source.graph.tensor_count())
      return false;
    const auto root = source.graph.root(source.state[i].tensor);
    for (std::size_t j = 0; j < i; ++j)
      if (source.graph.root(source.state[j].tensor) == root) return false;
  }
  for (ir::TensorId id{0}; id.v < source.graph.tensor_count(); ++id.v) {
    if (source.graph.root(id) != id) continue;
    const auto& binding = source.graph.tensor(id).binding;
    if (const auto* weight_ref = std::get_if<ir::WeightRef>(&binding)) {
      if (std::ranges::count(source.weights, id, &components::StepWeight::tensor) != 1)
        return false;
      const auto found = std::ranges::find(source.weights, id, &components::StepWeight::tensor);
      if (found == source.weights.end() || found->ref != *weight_ref) return false;
    } else if (const auto* state_ref = std::get_if<ir::StateRef>(&binding)) {
      if (std::ranges::count(source.state, id, &components::StepState::tensor) != 1) return false;
      const auto found = std::ranges::find(source.state, id, &components::StepState::tensor);
      if (found == source.state.end() || found->ref != *state_ref) return false;
    }
  }
  for (const auto& input : source.inputs)
    if (!std::ranges::contains(io.inputs, source.graph.root(input.tensor))) return false;
  return true;
}

// Ordinary graph ports are already typed by their actual IR descriptors. They
// do not acquire token semantics from an invented StepGraph input inventory.
bool SourceInventoryCurrent(const components::GraphBuildIO& io,
                            const components::StepGraph* token) {
  if (token) return &token->graph == &io.graph && ExactSourceInventory(*token, io);
  if (!std::ranges::equal(io.inputs, io.graph.inputs()) ||
      !std::ranges::equal(io.outputs, io.graph.outputs()))
    return false;
  for (const auto input : io.inputs)
    if (!input.valid() || input.v >= io.graph.tensor_count() || io.graph.root(input) != input ||
        io.graph.tensor(input).role != ir::Role::kInput ||
        std::ranges::count(io.inputs, input) != 1)
      return false;
  for (const auto& weight : io.weights)
    if (!weight.tensor.valid() || weight.tensor.v >= io.graph.tensor_count() ||
        io.graph.root(weight.tensor) != weight.tensor ||
        std::ranges::count(io.weights, weight.tensor, &components::StepWeight::tensor) != 1)
      return false;
  for (ir::TensorId id{0}; id.v < io.graph.tensor_count(); ++id.v) {
    if (io.graph.root(id) != id) continue;
    const auto& tensor = io.graph.tensor(id);
    if (std::holds_alternative<ir::StateRef>(tensor.binding) || tensor.role == ir::Role::kState)
      return false;
    if (const auto* weight = std::get_if<ir::WeightRef>(&tensor.binding)) {
      const auto found = std::ranges::find(io.weights, id, &components::StepWeight::tensor);
      if (found == io.weights.end() || found->ref != *weight) return false;
    }
  }
  return true;
}

base::Sha256Digest SourceInventoryDigest(const components::GraphBuildIO& io,
                                         const components::StepGraph* token,
                                         std::uint16_t active_stage) {
  if (token) return SourceMetadataDigest(*token, active_stage);
  base::Sha256 hash;
  hash.Update("llmp-comparison-component-ports-v1");
  HashFact(hash, active_stage);
  HashFact(hash, static_cast<std::uint64_t>(io.inputs.size()));
  for (const auto input : io.inputs) HashFact(hash, input.v);
  HashFact(hash, static_cast<std::uint64_t>(io.outputs.size()));
  for (const auto output : io.outputs) HashFact(hash, output.v);
  HashFact(hash, static_cast<std::uint64_t>(io.weights.size()));
  for (const auto& weight : io.weights) {
    HashFact(hash, weight.tensor.v);
    HashFact(hash, weight.ref.stage);
    HashFact(hash, weight.ref.component);
    HashFact(hash, weight.ref.role);
    HashFact(hash, weight.ref.layer);
    HashFact(hash, weight.ref.expert_array);
    HashFact(hash, static_cast<std::uint64_t>(weight.parts.size()));
    for (const auto& [name, role] : weight.parts) {
      HashText(hash, name);
      HashFact(hash, role);
    }
  }
  return hash.Finish();
}

bool AddAllocationBytes(std::uint64_t amount, std::uint64_t& total) {
  return amount <= std::numeric_limits<std::uint64_t>::max() - total && (total += amount, true);
}

bool ValidBoundedRange(const kernels::BoundedAddress& range, std::uint64_t required) {
  std::uint64_t end = 0;
  if (range.base == 0 || range.bytes < required || range.alignment == 0 ||
      !std::has_single_bit(range.alignment) || range.base % range.alignment != 0 ||
      __builtin_add_overflow(range.base, range.bytes, &end))
    return false;
  if (!range.allocation) return true;
  const auto& allocation = *range.allocation;
  std::uint64_t allocation_end = 0;
  return allocation.base != 0 && allocation.bytes != 0 && allocation.base % range.alignment == 0 &&
         !__builtin_add_overflow(allocation.base, allocation.bytes, &allocation_end) &&
         range.base >= allocation.base && end <= allocation_end;
}

std::optional<std::uint64_t> RequiredBytes(const ir::TensorDesc& tensor,
                                           const planner::RootBytesOf& written,
                                           const ir::Graph& graph, ir::TensorId id) {
  auto storage = planner::RootBytes(tensor);
  auto parts = format::PartsOf(tensor.format);
  if (!storage || !parts) return std::nullopt;
  const auto data = parts->Find(format::PartRole::kData);
  if (!data) return std::nullopt;
  std::uint64_t bytes = std::max(*storage, data->readable);
  if (written) bytes = std::max(bytes, written(graph, id));
  return bytes;
}

std::optional<const components::StepInput*> FindTypedInput(const components::StepGraph& source,
                                                           ir::TensorId root) {
  const components::StepInput* found = nullptr;
  for (const auto& input : source.inputs) {
    if (source.graph.root(input.tensor) != root) continue;
    if (found != nullptr) return std::nullopt;
    found = &input;
  }
  if (found == nullptr) return std::nullopt;
  return found;
}

bool ExternalMatches(const components::ComparisonExternalBinding& external,
                     const kernels::ComparisonBindingRange& range) {
  return external.original_root == range.original_root && external.part == range.part &&
         external.allocation_class == range.allocation_class &&
         external.allocation_bytes == range.allocation_bytes && external.offset == range.offset &&
         external.bytes == range.bytes && external.alignment == range.alignment &&
         external.resource_part_identity == range.resource_part_identity &&
         external.row_storage == range.row_storage;
}

// A row-bounded physical slot is useful to comparison only when every
// planned read is still the ordinary, typed side-row gather the engine
// stages. The source inventory supplies semantic authority; the planned graph
// supplies the actual consumers after any planner rewrite. This is an
// allocation-free preflight, so unsupported proof domains fall back before a
// range owner is published while ordinary lowering remains unaffected.
bool SupportedRowTableConsumers(const components::StepGraph& source, const planner::StepPlan& plan,
                                ir::TensorId source_weight, const ir::WeightRef& ref,
                                std::uint16_t active_stage) {
  if (ref.stage != active_stage || !source_weight.valid() ||
      source_weight.v >= source.graph.tensor_count() || source_weight.v >= plan.tensors.size())
    return false;
  const ir::TensorId mapped_weight = plan.tensors[source_weight.v];
  if (!mapped_weight.valid() || mapped_weight.v >= plan.graph.tensor_count()) return false;
  const ir::TensorId planned_weight = plan.graph.root(mapped_weight);
  bool saw_get_rows = false;
  for (const auto& node : plan.graph.nodes()) {
    for (std::size_t operand = 0; operand < node.nsrc; ++operand) {
      const ir::TensorId source_id = node.source(operand);
      if (!source_id.valid() || source_id.v >= plan.graph.tensor_count() ||
          plan.graph.root(source_id) != planned_weight)
        continue;
      if (node.key != ir::Primitive(ir::Op::kGetRows) || node.nsrc != 2 || operand != 0 ||
          node.source(0) != planned_weight)
        return false;

      const ir::TensorId ids_source = node.source(1);
      if (!ids_source.valid() || ids_source.v >= plan.graph.tensor_count()) return false;
      const ir::TensorId planned_ids = plan.graph.root(ids_source);
      const components::StepInput* mapped_input = nullptr;
      for (const auto& input : source.inputs) {
        if (!input.tensor.valid() || input.tensor.v >= source.graph.tensor_count() ||
            input.tensor.v >= plan.tensors.size())
          continue;
        const auto mapped_ids = plan.tensors[input.tensor.v];
        if (!mapped_ids.valid() || mapped_ids.v >= plan.graph.tensor_count() ||
            plan.graph.root(mapped_ids) != planned_ids)
          continue;
        if (mapped_input != nullptr) return false;
        mapped_input = &input;
      }
      if (mapped_input == nullptr ||
          source.graph.root(mapped_input->tensor) != mapped_input->tensor ||
          mapped_input->kind != components::InputKind::kSideRows ||
          mapped_input->segment != ir::kJoinedSegment || mapped_input->component != ref.component ||
          source.graph.tensor(mapped_input->tensor).format.encoding != format::Encoding::kI32 ||
          source.graph.tensor(mapped_input->tensor).format.layout != format::Layout::kDense ||
          ids_source != planned_ids)
        return false;
      saw_get_rows = true;
    }
  }
  return saw_get_rows;
}

bool NormalizeAllocationClasses(std::vector<kernels::ComparisonBindingRange>& ranges) {
  std::uint32_t next_class = 0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto& current = ranges[i];
    std::uint64_t current_end = 0;
    if (current.base == 0 || current.allocation_bytes == 0 ||
        __builtin_add_overflow(current.base, current.allocation_bytes, &current_end))
      return false;
    bool aliased = false;
    for (std::size_t j = 0; j < i; ++j) {
      const auto& prior = ranges[j];
      std::uint64_t prior_end = 0;
      if (prior.base == 0 || prior.allocation_bytes == 0 ||
          __builtin_add_overflow(prior.base, prior.allocation_bytes, &prior_end))
        return false;
      const bool overlaps = current.base < prior_end && prior.base < current_end;
      if (!overlaps) continue;
      if (current.base != prior.base || current_end != prior_end) return false;
      current.allocation_class = prior.allocation_class;
      aliased = true;
      break;
    }
    if (!aliased) {
      if (next_class == std::numeric_limits<std::uint32_t>::max()) return false;
      current.allocation_class = next_class++;
    }
  }
  return true;
}

ComparisonPairFailure::Code PlanFailureCode(planner::PlanError error) {
  if (error == planner::PlanError::kAdmission) return ComparisonPairFailure::Code::kAdmission;
  if (error == planner::PlanError::kStaleComparisonSite ||
      error == planner::PlanError::kAmbiguousComparisonSite ||
      error == planner::PlanError::kMissingComparisonSite)
    return ComparisonPairFailure::Code::kStale;
  return ComparisonPairFailure::Code::kInvalid;
}

bool SamePorts(std::span<const kernels::ComparisonRegionValuePort> a,
               std::span<const kernels::ComparisonRegionValuePort> b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto& x = a[i];
    const auto& y = b[i];
    if (x.kind != y.kind || x.ordinal != y.ordinal || x.source_tensor != y.source_tensor ||
        x.role != y.role || x.format != y.format || x.bytes != y.bytes ||
        x.root_offset != y.root_offset || x.read != y.read || x.written != y.written)
      return false;
  }
  return true;
}

bool MatchesBoundaryPorts(const components::OwnedComparisonRegionBoundary& boundary,
                          std::span<const kernels::ComparisonRegionValuePort> values) {
  std::size_t expected_count = 0;
  for (const auto& port : boundary.ports()) {
    if (port.kind == components::ComparisonRegionPortKind::kStateRead ||
        port.kind == components::ComparisonRegionPortKind::kStateWrite)
      return false;
    ++expected_count;
  }
  if (values.size() != expected_count || values.size() != boundary.ports().size()) return false;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto& value = values[i];
    if (value.ordinal != i) return false;
    if (value.ordinal >= boundary.ports().size() || value.address == 0 || value.bytes == 0 ||
        value.bytes > std::numeric_limits<std::uint64_t>::max() - value.address)
      return false;
    const auto& expected = boundary.ports()[value.ordinal];
    const auto expected_kind = expected.kind == components::ComparisonRegionPortKind::kInput
                                   ? kernels::ComparisonRegionValueKind::kInput
                                   : kernels::ComparisonRegionValueKind::kOutput;
    if (expected.kind == components::ComparisonRegionPortKind::kStateRead ||
        expected.kind == components::ComparisonRegionPortKind::kStateWrite ||
        value.kind != expected_kind || value.source_tensor != expected.original_tensor ||
        !expected.original_tensor.valid() || expected.canonical_tensor >= boundary.tensors().size())
      return false;
    const auto& tensor = boundary.tensors()[expected.canonical_tensor];
    if (value.role != tensor.role || value.format != tensor.format ||
        value.bytes != tensor.storage_bytes || value.root_offset != expected.view_offset)
      return false;
    if (value.kind == kernels::ComparisonRegionValueKind::kInput && (!value.read || value.written))
      return false;
    if (value.kind == kernels::ComparisonRegionValueKind::kOutput && (value.read || !value.written))
      return false;
  }
  return true;
}

bool SameDigest(const base::Sha256Digest& a, const base::Sha256Digest& b) {
  return std::ranges::equal(a, b);
}

template <class Project, class Clear, class Rebuild, class Accept>
std::expected<bool, ComparisonPairFailure> CloseCommonBoundary(
    std::optional<components::OwnedComparisonRegionBoundary>& boundary,
    std::span<std::uint8_t> membership, std::vector<ir::NodeId>& additions, Project&& project,
    Clear&& clear, Rebuild&& rebuild, Accept&& accept) {
  while (true) {
    if (!boundary)
      return std::unexpected(
          Failure(ComparisonPairFailure::Code::kNotReady, "comparison boundary is absent"));
    auto results = std::invoke(project, *boundary);
    auto& left = results.first;
    auto& right = results.second;
    if (left.status == kernels::ComparisonRegionStatus::kUnproved ||
        right.status == kernels::ComparisonRegionStatus::kUnproved)
      return false;
    if (left.status != kernels::ComparisonRegionStatus::kExpand &&
        right.status != kernels::ComparisonRegionStatus::kExpand) {
      std::invoke(accept, std::move(left), std::move(right));
      return true;
    }
    const auto merged = MergeComparisonExpansions(
        left.status == kernels::ComparisonRegionStatus::kExpand ? left.required_members
                                                                : std::span<const ir::NodeId>{},
        right.status == kernels::ComparisonRegionStatus::kExpand ? right.required_members
                                                                 : std::span<const ir::NodeId>{},
        membership, additions);
    if (merged == ComparisonExpansionMerge::kOutOfRange)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                     "region expansion escaped the source graph"));
    if (merged == ComparisonExpansionMerge::kCapacity)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                     "comparison expansion exceeded its paid source bound"));
    if (merged == ComparisonExpansionMerge::kNoProgress)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kNoProgress,
                                     "region expansion requested no new source members"));
    std::invoke(clear);
    boundary.reset();
    auto rebuilt = std::invoke(rebuild);
    if (!rebuilt) return std::unexpected(rebuilt.error());
  }
}

bool HasOnlyOrdinaryRestagedInputs(const components::StepGraph& source, bool ready_boundary) {
  return std::ranges::all_of(source.inputs, [ready_boundary](const components::StepInput& input) {
    switch (input.kind) {
      case components::InputKind::kStream:
      case components::InputKind::kResidual:
      case components::InputKind::kCarry:
      case components::InputKind::kSide:
        return true;
      case components::InputKind::kPositions:
      case components::InputKind::kFeatures:
        return ready_boundary;
      default:
        return false;
    }
  });
}

bool HasExactReadyValueSlots(const components::StepGraph& source,
                             std::span<const kernels::InputSlot> slots) {
  const auto exact_kind = [](components::InputKind kind) {
    return kind == components::InputKind::kFeatures || kind == components::InputKind::kPositions;
  };
  const auto exact_slot_kind = [](kernels::InputKind kind) {
    return kind == kernels::InputKind::kFeatures || kind == kernels::InputKind::kPositions;
  };
  if (std::ranges::none_of(source.inputs,
                           [&](const auto& input) { return exact_kind(input.kind); }) &&
      std::ranges::none_of(slots, [&](const auto& slot) { return exact_slot_kind(slot.kind); }))
    return true;
  const auto graph_inputs = source.graph.inputs();
  if (graph_inputs.size() != source.inputs.size()) return false;
  for (const auto& input : source.inputs) {
    if (!input.tensor.valid() || input.tensor.v >= source.graph.tensor_count() ||
        source.graph.root(input.tensor) != input.tensor ||
        std::ranges::count(graph_inputs, input.tensor) != 1 ||
        std::ranges::count(source.inputs, input.tensor, &components::StepInput::tensor) != 1)
      return false;
    if (!exact_kind(input.kind)) continue;
    const bool positions = input.kind == components::InputKind::kPositions;
    const auto expected_kind =
        positions ? kernels::InputKind::kPositions : kernels::InputKind::kFeatures;
    const auto& tensor = source.graph.tensor(input.tensor);
    const auto bytes = ir::StorageBytes(tensor);
    if (input.segment != tensor.segment || tensor.role != ir::Role::kInput || tensor.is_view() ||
        !std::holds_alternative<std::monostate>(tensor.binding) ||
        tensor.format.encoding != (positions ? format::Encoding::kI32 : format::Encoding::kF32) ||
        (positions && tensor.format.shape.rank > 1) ||
        tensor.format.layout != format::Layout::kDense || !bytes)
      return false;
    std::size_t matching_slots = 0;
    for (const auto& slot : slots) {
      if (slot.tensor != input.tensor) continue;
      ++matching_slots;
      if (slot.kind != expected_kind || slot.segment != input.segment || slot.bytes != *bytes)
        return false;
    }
    if (matching_slots != 1) return false;
  }
  for (const auto& slot : slots) {
    if (slot.tensor.v >= source.graph.tensor_count() ||
        std::ranges::count(source.inputs, slot.tensor, &components::StepInput::tensor) != 1 ||
        std::ranges::count(slots, slot.tensor, &kernels::InputSlot::tensor) != 1)
      return false;
    if (!exact_slot_kind(slot.kind)) continue;
    const auto typed =
        std::ranges::find(source.inputs, slot.tensor, &components::StepInput::tensor);
    if (typed == source.inputs.end() ||
        (slot.kind == kernels::InputKind::kFeatures &&
         typed->kind != components::InputKind::kFeatures) ||
        (slot.kind == kernels::InputKind::kPositions &&
         typed->kind != components::InputKind::kPositions))
      return false;
  }
  return true;
}

bool HasExactComponentInputInventory(const components::StepGraph& source) {
  const auto graph_inputs = source.graph.inputs();
  if (graph_inputs.size() != source.inputs.size()) return false;
  for (const auto& input : source.inputs) {
    switch (input.kind) {
      case components::InputKind::kStream:
      case components::InputKind::kResidual:
      case components::InputKind::kCarry:
      case components::InputKind::kSide:
        break;
      default:
        return false;
    }
    if (!input.tensor.valid() || input.tensor.v >= source.graph.tensor_count() ||
        std::ranges::count(graph_inputs, input.tensor) != 1)
      return false;
  }
  for (const auto tensor : graph_inputs)
    if (std::ranges::count(source.inputs, tensor, &components::StepInput::tensor) != 1)
      return false;
  return true;
}

bool SamePersistentSeal(const kernels::ComparisonRegionRecipeSeal& a,
                        const kernels::ComparisonRegionRecipeSeal& b) {
  return a.version == b.version && a.missing == b.missing && SameDigest(a.subject, b.subject) &&
         SameDigest(a.boundary, b.boundary) && SameDigest(a.planned_region, b.planned_region) &&
         SameDigest(a.physical_recipe, b.physical_recipe) && SameDigest(a.policy, b.policy) &&
         SameDigest(a.registry, b.registry) && SameDigest(a.backend_recipe, b.backend_recipe) &&
         SameDigest(a.backend_build, b.backend_build) &&
         SameDigest(a.backend_selection, b.backend_selection) &&
         SameDigest(a.backend_device, b.backend_device);
}

bool NonzeroDigest(const base::Sha256Digest& digest) {
  return std::ranges::any_of(digest, [](std::uint8_t byte) { return byte != 0; });
}

execution::RecipeArmIdentity ArmIdentity(const kernels::ComparisonRegionRecipeSeal& seal) {
  return {.planned_region = seal.planned_region,
          .physical_recipe = seal.physical_recipe,
          .backend_recipe = seal.backend_recipe,
          .backend_build = seal.backend_build,
          .backend_selection = seal.backend_selection};
}

}  // namespace

std::optional<base::Sha256Digest> ComparisonPreparationDomain(
    ComparisonRecipePreparation preparation, execution::Phase phase) noexcept {
  if (preparation == ComparisonRecipePreparation::kWholeProgramRestoredStateSingleInvocation ||
      preparation == ComparisonRecipePreparation::kWholeProgramRestoredStateOrdinaryInvocation) {
    if (static_cast<std::uint8_t>(phase) > static_cast<std::uint8_t>(execution::Phase::kCommit))
      return {};
    base::Sha256 hash;
    hash.Update("llmp-whole-program-preparation-domain-v1;");
    const std::array<std::byte, 1> value{static_cast<std::byte>(phase)};
    hash.Update(value);
    hash.Update(
        "mode:full;invocations:1;state:complete-identical-image-restored-before-each-arm;"
        "saves:complete-identical-image-restored-before-each-arm;"
        "inputs:original-canonical-typed-restage-before-each-invocation;");
    hash.Update(preparation ==
                        ComparisonRecipePreparation::kWholeProgramRestoredStateOrdinaryInvocation
                    ? "execution:ordinary-Run;capture:forbidden;"
                    : "capture:entire-ordinary-program;");
    hash.Update(
        "completion:owned-fence-before-observation;"
        "production:no-state-reset;");
    return hash.Finish();
  }
  if ((preparation !=
           ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody &&
       preparation != ComparisonRecipePreparation::kStatelessReadyBoundaryOperands) ||
      static_cast<std::uint8_t>(phase) > static_cast<std::uint8_t>(execution::Phase::kCommit))
    return {};
  base::Sha256 hash;
  const bool ready_boundary =
      preparation == ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  // The legacy full-region protocol bytes are evidence-stable. The ready
  // protocol has a separate explicit applicability clause: runtime programs
  // never execute the measurement prefix, while the production certificate
  // proves the boundary is already supplied by ordinary execution.
  hash.Update(ready_boundary ? "llmp-ready-boundary-preparation-domain-v2"
                             : "llmp-region-preparation-domain-v1");
  hash.Update(ready_boundary ? "protocol:2;" : "protocol:1;");
  const std::array<std::byte, 1> phase_byte{
      static_cast<std::byte>(static_cast<std::uint8_t>(phase))};
  hash.Update(phase_byte);
  hash.Update("mode:full;");
  hash.Update(ready_boundary ? "measurement-state:none;" : "state:none;");
  hash.Update(ready_boundary ? "measurement-inputs:original-restaged-each-frame;"
                             : "inputs:original-restaged-each-frame;");
  hash.Update(ready_boundary ? "measurement-prefix:pure-once-outside-capture;"
                             : "prefix:pure-once-outside-capture;");
  hash.Update(ready_boundary ? "captured-full-body;" : "body:entire-local-region-captured;");
  hash.Update(ready_boundary
                  ? "applicability:stateless-ready-boundary-values-and-immutable-operands;"
                  : "retained-preparation:none;");
  return hash.Finish();
}

std::optional<execution::RecipeComparisonKey> RecipeComparisonKeyForReadyCertificates(
    const components::OwnedComparisonRegionBoundary& boundary, ir::OpKey discovery_operation,
    execution::Phase phase, const base::Sha256Digest& selection_environment,
    const kernels::ComparisonRegionRecipeSeal& incumbent,
    const kernels::ComparisonRegionRecipeSeal& alternative) noexcept {
  const auto valid = [&](const kernels::ComparisonRegionRecipeSeal& seal) {
    return seal.version == 3 && seal.state == kernels::ComparisonProofState::kDescriptorVerified &&
           seal.missing == kernels::ComparisonMissing::kNone &&
           SameDigest(seal.subject, boundary.digest().logical) &&
           SameDigest(seal.boundary, boundary.digest().combined) &&
           SameDigest(seal.registry, selection_environment) && NonzeroDigest(seal.subject) &&
           NonzeroDigest(seal.boundary) && NonzeroDigest(seal.planned_region) &&
           NonzeroDigest(seal.physical_recipe) && NonzeroDigest(seal.policy) &&
           NonzeroDigest(seal.registry) && NonzeroDigest(seal.backend_recipe) &&
           NonzeroDigest(seal.backend_build) && NonzeroDigest(seal.backend_selection) &&
           NonzeroDigest(seal.backend_device);
  };
  if (!valid(incumbent) || !valid(alternative) ||
      !SameDigest(incumbent.subject, alternative.subject) ||
      !SameDigest(incumbent.boundary, alternative.boundary) ||
      !SameDigest(incumbent.policy, alternative.policy) ||
      !SameDigest(incumbent.registry, alternative.registry) ||
      !SameDigest(incumbent.backend_device, alternative.backend_device))
    return {};
  const auto domain = ComparisonPreparationDomain(
      ComparisonRecipePreparation::kStatelessReadyBoundaryOperands, phase);
  if (!domain) return {};
  execution::RecipeComparisonKey key;
  key.version = 1;
  key.region_canonicalizer = components::kComparisonRegionCanonicalizerVersion;
  key.discovery_operation = discovery_operation;
  key.mode = execution::RecipeRunMode::kFull;
  key.subject = boundary.digest().logical;
  key.boundary = boundary.digest().combined;
  key.policy = incumbent.policy;
  key.device = incumbent.backend_device;
  key.selection_environment = selection_environment;
  key.preparation_domain = *domain;
  key.incumbent = ArmIdentity(incumbent);
  key.alternative = ArmIdentity(alternative);
  return key;
}

bool SameWholeProgramRecipe(const kernels::ComparisonRecipeSeal& descriptor,
                            const kernels::ComparisonRecipeSeal& bound) noexcept {
  if (descriptor.state != kernels::ComparisonProofState::kDescriptorVerified ||
      bound.state != kernels::ComparisonProofState::kBoundVerified ||
      descriptor.scope != kernels::ComparisonProofScope::kWholeProgramNative ||
      descriptor.missing != kernels::ComparisonMissing::kNone)
    return false;
  auto normalized = bound;
  normalized.state = descriptor.state;
  return normalized == descriptor;
}

std::optional<execution::RecipeComparisonKey> RecipeComparisonKeyForWholePrograms(
    const components::OwnedComparisonBoundary& full,
    const components::OwnedComparisonRegionBoundary& site_boundary,
    const planner::ComparisonSite& site, execution::Phase phase,
    const base::Sha256Digest& selection_environment, const kernels::ComparisonRecipeSeal& incumbent,
    const kernels::ComparisonRecipeSeal& alternative) noexcept {
  if (full.source_identity() == 0 || full.source_identity() != site_boundary.source_identity() ||
      full.source_identity() != site.selector.source_graph_identity ||
      full.active_stage() != site_boundary.active_stage() || site.accepted_member_origins.empty() ||
      !NonzeroDigest(site.exact_site_identity) || !planner::AuthenticComparisonSite(site))
    return {};
  std::size_t at = 0;
  for (std::size_t group = 0; group < site.accepted_member_origins.size(); ++group) {
    if (site.accepted_member_origins[group].empty()) return {};
    for (const auto original : site.accepted_member_origins[group]) {
      if (at == site_boundary.nodes().size() || site_boundary.nodes()[at].group != group ||
          site_boundary.nodes()[at].original_node != original)
        return {};
      ++at;
    }
  }
  if (at != site_boundary.nodes().size()) return {};
  const auto valid = [&](const kernels::ComparisonRecipeSeal& seal) {
    return seal.version == 4 && seal.mode == kernels::ComparisonRunMode::kFull &&
           seal.scope == kernels::ComparisonProofScope::kWholeProgramNative &&
           (seal.execution_method == kernels::ComparisonExecutionMethod::kCaptured ||
            seal.execution_method == kernels::ComparisonExecutionMethod::kOrdinaryOnly) &&
           seal.state == kernels::ComparisonProofState::kDescriptorVerified &&
           seal.missing == kernels::ComparisonMissing::kNone &&
           SameDigest(seal.subject, full.source_digest()) &&
           SameDigest(seal.boundary, full.digest().combined) &&
           SameDigest(seal.registry, selection_environment) && NonzeroDigest(seal.subject) &&
           NonzeroDigest(seal.boundary) && NonzeroDigest(seal.planned_graph) &&
           NonzeroDigest(seal.physical_recipe) && NonzeroDigest(seal.policy) &&
           NonzeroDigest(seal.registry) && NonzeroDigest(seal.backend_recipe) &&
           NonzeroDigest(seal.backend_build) && NonzeroDigest(seal.backend_selection) &&
           NonzeroDigest(seal.backend_device);
  };
  if (!valid(incumbent) || !valid(alternative) ||
      !SameDigest(incumbent.policy, alternative.policy) ||
      !SameDigest(incumbent.backend_device, alternative.backend_device))
    return {};
  const bool ordinary =
      incumbent.execution_method == kernels::ComparisonExecutionMethod::kOrdinaryOnly ||
      alternative.execution_method == kernels::ComparisonExecutionMethod::kOrdinaryOnly;
  const auto preparation = ComparisonPreparationDomain(
      ordinary ? ComparisonRecipePreparation::kWholeProgramRestoredStateOrdinaryInvocation
               : ComparisonRecipePreparation::kWholeProgramRestoredStateSingleInvocation,
      phase);
  if (!preparation) return {};
  execution::RecipeComparisonKey key;
  key.region_canonicalizer = components::kComparisonRegionCanonicalizerVersion;
  key.discovery_operation = site.registered_result;
  key.mode = execution::RecipeRunMode::kFull;
  // Canonical site identity is already source-occurrence normalized by the
  // owned region serializer; live graph tokens authenticate, never hash.
  base::Sha256 subject;
  if (site.whole_checkpoint.whole_only()) {
    subject.Update("llmp-whole-program-checkpoint-subject-v1");
    subject.Update(std::as_bytes(std::span(site.whole_checkpoint.subject())));
  } else {
    subject.Update("llmp-whole-program-site-subject-v1");
  }
  subject.Update(std::as_bytes(std::span(full.source_digest())));
  subject.Update(std::as_bytes(std::span(site_boundary.digest().logical)));
  HashText(subject, site.selector.pattern);
  HashFact(subject, site.selector.version);
  HashFact(subject, site.selector.kind);
  key.subject = subject.Finish();
  base::Sha256 boundary;
  boundary.Update("llmp-whole-program-site-boundary-v1");
  boundary.Update(std::as_bytes(std::span(full.digest().combined)));
  boundary.Update(std::as_bytes(std::span(site_boundary.digest().combined)));
  key.boundary = boundary.Finish();
  key.policy = incumbent.policy;
  key.device = incumbent.backend_device;
  key.selection_environment = selection_environment;
  key.preparation_domain = *preparation;
  const auto arm = [](const kernels::ComparisonRecipeSeal& seal) -> execution::RecipeArmIdentity {
    return {.planned_region = seal.planned_graph,
            .physical_recipe = seal.physical_recipe,
            .backend_recipe = seal.backend_recipe,
            .backend_build = seal.backend_build,
            .backend_selection = seal.backend_selection};
  };
  key.incumbent = arm(incumbent);
  key.alternative = arm(alternative);
  return key;
}

ComparisonExpansionMerge MergeComparisonExpansions(std::span<const ir::NodeId> incumbent,
                                                   std::span<const ir::NodeId> alternative,
                                                   std::span<std::uint8_t> membership,
                                                   std::vector<ir::NodeId>& additions) {
  const auto start = additions.size();
  const auto append = [&](std::span<const ir::NodeId> requested) {
    for (const auto node : requested) {
      if (!node.valid() || node.v >= membership.size())
        return ComparisonExpansionMerge::kOutOfRange;
      if (membership[node.v]) continue;
      if (additions.size() == additions.capacity()) return ComparisonExpansionMerge::kCapacity;
      membership[node.v] = 1;
      additions.push_back(node);
    }
    return ComparisonExpansionMerge::kProgress;
  };
  const auto first = append(incumbent);
  if (first == ComparisonExpansionMerge::kOutOfRange ||
      first == ComparisonExpansionMerge::kCapacity)
    return first;
  const auto second = append(alternative);
  if (second == ComparisonExpansionMerge::kOutOfRange ||
      second == ComparisonExpansionMerge::kCapacity)
    return second;
  if (additions.size() == start) return ComparisonExpansionMerge::kNoProgress;
  std::ranges::sort(additions, {}, &ir::NodeId::v);
  return ComparisonExpansionMerge::kProgress;
}

std::expected<ComparisonSourceBindings, ComparisonPairFailure> ComparisonSourceBindings::Create(
    std::shared_ptr<const ir::StorageFunder> funder, const components::StepGraph& source,
    std::uint16_t active_stage) {
  return CreateCore(std::move(funder), components::GraphIO(source), &source, active_stage);
}

std::expected<ComparisonSourceBindings, ComparisonPairFailure> ComparisonSourceBindings::Create(
    std::shared_ptr<const ir::StorageFunder> funder, const components::GraphBuildIO& source,
    std::uint16_t active_stage) {
  return CreateCore(std::move(funder), source, nullptr, active_stage);
}

std::expected<ComparisonSourceBindings, ComparisonPairFailure> ComparisonSourceBindings::CreateCore(
    std::shared_ptr<const ir::StorageFunder> funder, const components::GraphBuildIO& io,
    const components::StepGraph* source, std::uint16_t active_stage) {
  if (!funder || io.graph.identity() == 0)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                   "comparison source inventory needs a funded identified graph"));
  if (source && !source->owner_products.empty())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "owner-product source inventories are unsupported"));
  const auto states = source ? std::span<const components::StepState>(source->state)
                             : std::span<const components::StepState>{};
  if (!SourceInventoryCurrent(io, source))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "source graph roots and typed inventory are not one-to-one"));
  std::size_t cell_positions = 0;
  for (const auto input_id : io.inputs) {
    const auto typed = source ? FindTypedInput(*source, input_id) : std::nullopt;
    if (source && (!typed || (*typed)->tensor != input_id))
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "typed step inputs do not exactly cover graph inputs"));
    if (typed && (*typed)->kind == components::InputKind::kCellPositions) {
      ++cell_positions;
      continue;
    }
  }
  for (const auto& typed : source ? std::span<const components::StepInput>(source->inputs)
                                  : std::span<const components::StepInput>{}) {
    if (!std::ranges::contains(io.inputs, typed.tensor))
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "typed step input is absent from GraphIO"));
  }

  std::size_t external_count = io.inputs.size() - cell_positions;
  for (const auto& weight : io.weights) {
    if (weight.tensor.v >= io.graph.tensor_count() ||
        io.graph.root(weight.tensor) != weight.tensor ||
        !std::holds_alternative<ir::WeightRef>(io.graph.tensor(weight.tensor).binding) ||
        std::get<ir::WeightRef>(io.graph.tensor(weight.tensor).binding) != weight.ref)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "weight metadata differs from the source graph"));
    const auto parts = format::PartsOf(io.graph.tensor(weight.tensor).format);
    if (!parts || parts->size() != weight.parts.size())
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "weight metadata omits or adds format parts"));
    for (const auto& part : parts->parts())
      if (std::ranges::count(weight.parts, part.role,
                             &std::pair<std::string, format::PartRole>::second) != 1)
        return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                       "weight metadata does not exactly name format parts"));
    if (weight.parts.size() > std::numeric_limits<std::size_t>::max() - external_count)
      return std::unexpected(
          Failure(ComparisonPairFailure::Code::kAdmission, "source external count overflows"));
    external_count += weight.parts.size();
  }
  for (const auto& state : states) {
    if (state.tensor.v >= io.graph.tensor_count() || io.graph.root(state.tensor) != state.tensor ||
        !std::holds_alternative<ir::StateRef>(io.graph.tensor(state.tensor).binding) ||
        std::get<ir::StateRef>(io.graph.tensor(state.tensor).binding) != state.ref)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "state metadata differs from the source graph"));
  }
  if (states.size() > std::numeric_limits<std::size_t>::max() - external_count)
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kAdmission, "source external count overflows"));
  external_count += states.size();
  if (external_count > std::numeric_limits<std::uint32_t>::max())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "source allocation-class inventory overflows"));

  std::uint64_t bytes = sizeof(ComparisonSourceBindings);
  const auto add_vector = [&](std::size_t count, std::size_t element_size) {
    if (count > std::numeric_limits<std::uint64_t>::max() / element_size) return false;
    const auto amount = static_cast<std::uint64_t>(count) * element_size;
    return AddAllocationBytes(amount, bytes) &&
           (count == 0 || AddAllocationBytes(kAllocatorEnvelope, bytes));
  };
  if (!add_vector(external_count, sizeof(components::ComparisonExternalBinding)) ||
      !add_vector(cell_positions, sizeof(components::ComparisonUnstoredInput)))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "source boundary inventory payment overflows"));
  auto loan = ir::AcquireStorageLoan(funder, bytes);
  if (!loan)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "source boundary inventory admission refused"));

  ComparisonSourceBindings result(std::move(*loan), std::move(funder), io, source, active_stage);
  result.source_identity_ = io.graph.identity();
  result.source_digest_ = io.graph.Digest();
  result.metadata_digest_ = SourceInventoryDigest(io, source, active_stage);
  result.external_count_ = external_count;
  result.external_.reserve(external_count);
  result.unstored_.reserve(cell_positions);
  for (const auto input_id : io.inputs) {
    const auto typed = source ? FindTypedInput(*source, input_id) : std::nullopt;
    if (typed && (*typed)->kind == components::InputKind::kCellPositions)
      result.unstored_.push_back(
          {.original_input = io.graph.root(input_id),
           .reason = components::ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks});
  }
  return result;
}

std::expected<ComparisonResolvedBindings, ComparisonPairFailure> ComparisonSourceBindings::Resolve(
    const planner::StepPlan& plan, const kernels::Addresses& addresses,
    const planner::RootBytesOf& written) {
  const auto io = source_ ? components::GraphIO(*source_) : io_;
  if (!funder_ || io.graph.identity() != source_identity_ || io.graph.Digest() != source_digest_ ||
      SourceInventoryDigest(io, source_, active_stage_) != metadata_digest_ ||
      !SourceInventoryCurrent(io, source_) || plan.primitive_digest != source_digest_ ||
      plan.tensors.size() != io.graph.tensor_count())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "plan no longer belongs to the retained source graph"));
  if (addresses.measuring() || addresses.activation_bytes == 0 ||
      addresses.activation_alignment == 0 || !std::has_single_bit(addresses.activation_alignment))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "comparison source ranges need real activation bounds"));

  // Reject unsupported row-table proof domains before paying or publishing a
  // per-plan external range vector. The exact ordinary side-row GetRows
  // lowering is still allowed to run through its normal path.
  if (addresses.row_storage) {
    for (const auto& weight : io.weights) {
      if (!weight.tensor.valid() || weight.tensor.v >= io.graph.tensor_count())
        return std::unexpected(
            Failure(ComparisonPairFailure::Code::kStale, "row-table weight metadata changed"));
      const auto root = io.graph.root(weight.tensor);
      const auto parts = format::PartsOf(io.graph.tensor(root).format);
      if (!parts) continue;
      for (const auto& declared : weight.parts) {
        const auto row_bytes = format::RowBytes(io.graph.tensor(root).format, declared.second);
        const auto storage = addresses.row_storage(weight.ref, declared.second);
        if (storage && row_bytes &&
            (!source_ ||
             !SupportedRowTableConsumers(*source_, plan, root, weight.ref, active_stage_)))
          return std::unexpected(
              Failure(ComparisonPairFailure::Code::kUnproved,
                      "row-table consumers are outside the typed side-row GetRows proof domain"));
      }
    }
  }

  std::uint64_t bytes = sizeof(ComparisonResolvedBindings);
  if (external_count_ >
          std::numeric_limits<std::uint64_t>::max() / sizeof(kernels::ComparisonBindingRange) ||
      !AddAllocationBytes(
          static_cast<std::uint64_t>(external_count_) * sizeof(kernels::ComparisonBindingRange),
          bytes) ||
      (external_count_ != 0 && !AddAllocationBytes(kAllocatorEnvelope, bytes)))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "per-plan range inventory payment overflows"));
  auto loan = ir::AcquireStorageLoan(funder_, bytes);
  if (!loan)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "per-plan range inventory admission refused"));
  ComparisonResolvedBindings result{};
  result.loan_ = std::move(*loan);
  result.ranges_.reserve(external_count_);

  const auto append = [&](ir::TensorId root, std::optional<format::PartRole> part,
                          const kernels::BoundedAddress& bound, std::uint64_t required,
                          std::optional<components::ComparisonRowStorage> row_storage = {}) {
    if (result.ranges_.size() >= external_count_ || !ValidBoundedRange(bound, required))
      return false;
    // Row-slot custody retains its exact independent slot allocation. A
    // backing declaration cannot enlarge that separately authenticated domain.
    if (row_storage && bound.allocation &&
        (bound.bytes != required || bound.allocation->base != bound.base ||
         bound.allocation->bytes != bound.bytes))
      return false;
    if (bound.content_identity && !part) return false;
    if (bound.content_identity &&
        !std::ranges::any_of(*bound.content_identity, [](auto b) { return b != 0; }))
      return false;
    if (row_storage && bound.content_identity &&
        *bound.content_identity != row_storage->content_identity)
      return false;
    const auto identity = row_storage
                              ? std::optional<base::Sha256Digest>(row_storage->content_identity)
                              : bound.content_identity;
    result.ranges_.push_back(
        {.original_root = root,
         .part = part,
         .allocation_bytes = bound.allocation ? bound.allocation->bytes : bound.bytes,
         .offset = bound.allocation ? bound.base - bound.allocation->base : 0,
         .bytes = bound.allocation ? bound.bytes : required,
         .alignment = bound.alignment,
         .base = bound.allocation ? bound.allocation->base : bound.base,
         .resource_part_identity = identity,
         .row_storage = std::move(row_storage)});
    return true;
  };
  const auto input_range = [&](ir::TensorId source_input, const components::StepInput* typed)
      -> std::optional<std::pair<kernels::BoundedAddress, std::uint64_t>> {
    const auto& original = io.graph.tensor(source_input);
    if (typed && typed->kind == components::InputKind::kCommitRows) {
      if (!addresses.save || !addresses.save_range) return std::nullopt;
      const auto bound =
          addresses.save_range(typed->layer, typed->component, typed->what, typed->segment);
      const auto base = addresses.save(typed->layer, typed->component, typed->what, typed->segment);
      if (!bound || base == 0 || bound->base != base) return std::nullopt;
      const auto required = RequiredBytes(original, {}, io.graph, source_input);
      if (!required || !ValidBoundedRange(*bound, *required)) return std::nullopt;
      return std::pair{*bound, *required};
    }
    if (typed && typed->kind == components::InputKind::kCellPositions) return std::nullopt;
    if (source_input.v >= plan.tensors.size()) return std::nullopt;
    const auto mapped = plan.tensors[source_input.v];
    if (!mapped.valid() || mapped.v >= plan.graph.tensor_count()) return std::nullopt;
    const auto planned_root = plan.graph.root(mapped);
    if (!planned_root.valid() || planned_root.v >= plan.graph.tensor_count()) return std::nullopt;
    const auto offset = std::ranges::find(plan.placement.offsets, planned_root,
                                          &std::pair<ir::TensorId, std::uint64_t>::first);
    if (offset == plan.placement.offsets.end() || plan.placement.alignment == 0 ||
        !std::has_single_bit(plan.placement.alignment) ||
        addresses.activation_alignment < plan.placement.alignment ||
        offset->second % plan.placement.alignment != 0)
      return std::nullopt;
    auto required = RequiredBytes(original, {}, io.graph, source_input);
    auto planned_required =
        RequiredBytes(plan.graph.tensor(planned_root), written, plan.graph, planned_root);
    if (!required || !planned_required) return std::nullopt;
    // The source interface is the original staged value. Native auxiliary
    // storage belongs to the backend's physical root, not this logical input.
    const auto physical_required = std::max(*required, *planned_required);
    std::uint64_t end = 0, base = 0, arena_end = 0;
    if (__builtin_add_overflow(offset->second, physical_required, &end) ||
        end > addresses.activation_bytes ||
        __builtin_add_overflow(addresses.activations, offset->second, &base) ||
        __builtin_add_overflow(addresses.activations, addresses.activation_bytes, &arena_end))
      return std::nullopt;
    const auto alignment = plan.placement.alignment;
    if (base == 0 || alignment == 0 || !std::has_single_bit(alignment) || base % alignment != 0)
      return std::nullopt;
    std::uint64_t physical_end = 0;
    if (__builtin_add_overflow(base, *required, &physical_end) || physical_end > arena_end)
      return std::nullopt;
    return std::pair{
        kernels::BoundedAddress{.base = base, .bytes = *required, .alignment = alignment},
        *required};
  };

  for (const auto input_id : io.inputs) {
    const auto root = io.graph.root(input_id);
    const auto typed = source_ ? FindTypedInput(*source_, input_id) : std::nullopt;
    if (source_ && !typed)
      return std::unexpected(
          Failure(ComparisonPairFailure::Code::kUnproved, "source input kind inventory changed"));
    if (typed && (*typed)->kind == components::InputKind::kCellPositions) continue;
    const auto bound = input_range(input_id, typed ? *typed : nullptr);
    if (!bound)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "source input has no authenticated bounded placement"));
    const auto required = RequiredBytes(io.graph.tensor(root), {}, io.graph, root);
    if (!required || !append(root, std::nullopt, bound->first, bound->second))
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "source input range is shorter than its required extent"));
  }
  for (const auto& weight : io.weights) {
    if (!addresses.weight || !addresses.weight_range)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "source weight has no authenticated bounded range"));
    const auto root = io.graph.root(weight.tensor);
    const auto parts = format::PartsOf(io.graph.tensor(root).format);
    if (!parts)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "source weight format has no part inventory"));
    for (const auto& declared : weight.parts) {
      const auto part = parts->Find(declared.second);
      if (!part)
        return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                       "source weight part is not in its format"));
      const auto bound = addresses.weight_range(weight.ref, declared.second);
      const auto base = addresses.weight(weight.ref, declared.second);
      std::optional<components::ComparisonRowStorage> row_storage;
      if (addresses.row_storage) row_storage = addresses.row_storage(weight.ref, declared.second);
      const auto capacity = addresses.weight_rows ? addresses.weight_rows(weight.ref) : 0;
      std::uint64_t required = part->readable;
      const auto row_bytes = format::RowBytes(io.graph.tensor(root).format, declared.second);
      if (row_storage || (row_bytes && capacity != 0)) {
        if (!row_bytes || !row_storage || capacity == 0 || row_storage->slot_capacity != capacity ||
            !components::ValidComparisonRowStorage(*row_storage, weight.ref, declared.second,
                                                   io.graph.tensor(root).format,
                                                   bound ? bound->bytes : 0))
          return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                         "row-table owner facts do not authenticate its slots"));
        std::uint64_t slot_bytes = 0;
        if (__builtin_mul_overflow(capacity, row_storage->row_bytes, &slot_bytes))
          return std::unexpected(
              Failure(ComparisonPairFailure::Code::kUnproved, "row-table slot extent overflows"));
        required = slot_bytes;
      }
      if (!bound || base == 0 || bound->base != base ||
          !append(root, declared.second, *bound, required, std::move(row_storage)))
        return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                       "source weight part lacks its required bounded extent"));
    }
  }
  for (const auto& state : source_ ? std::span<const components::StepState>(source_->state)
                                   : std::span<const components::StepState>{}) {
    if (!addresses.state || !addresses.state_range)
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "source state has no authenticated bounded range"));
    const auto root = io.graph.root(state.tensor);
    const auto& descriptor = io.graph.tensor(root);
    auto required = RequiredBytes(descriptor, {}, io.graph, root);
    const auto bound = addresses.state_range(state.ref);
    const auto base = addresses.state(state.ref);
    if (!required || !bound || base == 0 || bound->base != base ||
        !append(root, std::nullopt, *bound, *required))
      return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                     "source state lacks its required bounded extent"));
  }
  if (result.ranges_.size() != external_count_ || !NormalizeAllocationClasses(result.ranges_))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "source ranges alias ambiguously or exceed their inventory"));

  if (!resolved_) {
    external_.clear();
    for (const auto& range : result.ranges_) {
      external_.push_back({.original_root = range.original_root,
                           .part = range.part,
                           .allocation_class = range.allocation_class,
                           .allocation_bytes = range.allocation_bytes,
                           .offset = range.offset,
                           .bytes = range.bytes,
                           .alignment = range.alignment,
                           .resource_part_identity = range.resource_part_identity,
                           .row_storage = range.row_storage});
    }
    resolved_ = true;
  } else {
    if (external_.size() != result.ranges_.size())
      return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                     "source external inventory changed between plans"));
    for (std::size_t i = 0; i < external_.size(); ++i)
      if (!ExternalMatches(external_[i], result.ranges_[i]))
        return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                       "plan changes source slice bounds or alias relationships"));
  }
  return std::move(result);
}

std::optional<components::ComparisonBoundaryView> ComparisonSourceBindings::boundary_view() const {
  const auto io = source_ ? components::GraphIO(*source_) : io_;
  if (!resolved_ || !loan_ || !funder_ || io.graph.identity() != source_identity_ ||
      io.graph.Digest() != source_digest_ ||
      SourceInventoryDigest(io, source_, active_stage_) != metadata_digest_ ||
      !SourceInventoryCurrent(io, source_))
    return std::nullopt;
  return components::ComparisonBoundaryView{
      io,
      source_ ? std::span<const components::StepState>(source_->state)
              : std::span<const components::StepState>{},
      active_stage_, external_, unstored_};
}

ComparisonPair::ComparisonPair(ir::StorageLoan package_loan,
                               std::shared_ptr<const ir::StorageFunder> funder,
                               components::StepGraph&& source, planner::StepShape&& shape,
                               std::vector<components::ComparisonExternalBinding>&& external,
                               std::uint32_t context, std::uint16_t active_stage,
                               ComparisonInputDomain input_domain)
    : package_loan_(std::move(package_loan)),
      funder_(std::move(funder)),
      source_(std::move(source)),
      shape_(std::move(shape)),
      external_(std::move(external)),
      context_(context),
      active_stage_(active_stage),
      input_domain_(input_domain) {}

std::expected<std::unique_ptr<ComparisonPair>, ComparisonPairFailure> ComparisonPair::Create(
    std::shared_ptr<const ir::StorageFunder> funder, components::StepGraph&& source,
    planner::StepShape&& shape, std::vector<components::ComparisonExternalBinding>&& external,
    std::uint32_t context, std::uint16_t active_stage, ComparisonInputDomain input_domain) {
  if (!funder || source.graph.identity() == 0)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                   "comparison pair needs a funded, identified source graph"));
  if (shape.stage != active_stage)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                   "comparison stage differs from the supplied step shape"));
  if (input_domain == ComparisonInputDomain::kComponentPorts &&
      !HasExactComponentInputInventory(source))
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kUnproved,
                "component-port lowering requires an exact ordinary graph-input inventory"));
  std::uint64_t bytes = sizeof(ComparisonPair) + kAllocatorEnvelope;
  if (!VectorBytes(source.inputs, bytes) || !VectorBytes(source.weights, bytes) ||
      !VectorBytes(source.state, bytes) || !VectorBytes(source.saves, bytes) ||
      !VectorBytes(source.owner_products, bytes) || !VectorBytes(shape.segments, bytes) ||
      !VectorBytes(external, bytes))
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kAdmission, "comparison package inventory overflow"));
  for (const auto& input : source.inputs)
    if (!StringBytes(input.what, bytes))
      return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                     "comparison input-name inventory overflow"));
  for (const auto& weight : source.weights) {
    if (!VectorBytes(weight.parts, bytes))
      return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                     "comparison weight-part inventory overflow"));
    for (const auto& [name, role] : weight.parts) {
      (void)role;
      if (!StringBytes(name, bytes))
        return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                       "comparison weight-part name inventory overflow"));
    }
  }
  for (const auto& save : source.saves)
    if (!StringBytes(save.what, bytes))
      return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                     "comparison save-name inventory overflow"));
  const auto nodes = source.graph.nodes().size();
  if (nodes >= std::numeric_limits<std::uint32_t>::max() ||
      nodes > std::numeric_limits<std::uint64_t>::max() / sizeof(ir::NodeId) ||
      nodes >= std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint32_t) ||
      !AddBytes(static_cast<std::uint64_t>(nodes) * sizeof(ir::NodeId), bytes) ||
      !AddBytes(static_cast<std::uint64_t>(nodes) * sizeof(ir::NodeId), bytes) ||
      !AddBytes(static_cast<std::uint64_t>(nodes + 1) * sizeof(std::uint32_t), bytes) ||
      !AddBytes(static_cast<std::uint64_t>(nodes) * sizeof(std::uint8_t), bytes) ||
      !AddBytes(4 * kAllocatorEnvelope, bytes))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "comparison workspace inventory overflow"));
  auto loan = ir::AcquireStorageLoan(funder, bytes);
  if (!loan)
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kAdmission, "comparison package admission refused"));
  auto pair = std::unique_ptr<ComparisonPair>(new (std::nothrow) ComparisonPair(
      std::move(*loan), std::move(funder), std::move(source), std::move(shape), std::move(external),
      context, active_stage, input_domain));
  if (!pair)
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kAdmission, "comparison package allocation failed"));
  pair->candidate_offsets_.reserve(nodes + 1);
  pair->candidate_members_.reserve(nodes);
  pair->additions_.reserve(nodes);
  pair->membership_.assign(nodes, 0);
  return pair;
}

components::ComparisonBoundaryView ComparisonPair::SourceBoundaryView() const {
  return {components::GraphIO(source_), source_.state, active_stage_, external_};
}

std::expected<void, ComparisonPairFailure> ComparisonPair::Plan(
    const planner::ComparisonSiteSelector& selector, const execution::Registry& registry,
    const planner::DeviceFacts& device, const planner::Policy& policy,
    const planner::PlanOptions& options) {
  if (plan_attempted_ || inventory_ || alternative_plan_ || closed_)
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kInvalid, "comparison package was already planned"));
  plan_attempted_ = true;
  if (!source_.state.empty() || !source_.saves.empty() || !source_.owner_products.empty())
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kUnproved,
                "state, commit saves, and owner products are outside this pair domain"));
  // AssignLanes returns before consulting on_stream when there are no lane tags.
  if (!options.lanes.empty())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "lanes and routed selection are outside this pair domain"));
  auto inventory = planner::BoundComparison(SourceBoundaryView(), selector, funder_, registry,
                                            device, policy, options, &shape_);
  if (!inventory)
    return std::unexpected(
        Failure(PlanFailureCode(inventory.error().error), "incumbent comparison planning failed"));
  auto alternative = planner::MakeAlternative(*inventory);
  if (!alternative)
    return std::unexpected(Failure(PlanFailureCode(alternative.error().error),
                                   "comparison alternative construction failed"));
  auto planned = planner::PlanAlternative(*inventory, std::move(*alternative), registry, device,
                                          policy, options, &shape_);
  if (!planned)
    return std::unexpected(
        Failure(PlanFailureCode(planned.error().error), "alternative planning failed"));
  inventory_.emplace(std::move(*inventory));
  alternative_plan_.emplace(std::move(*planned));
  registry_digest_ = registry.SelectionDigest();
  auto seed = BuildSeedBoundary();
  if (!seed) {
    alternative_plan_.reset();
    inventory_.reset();
    return seed;
  }
  requests_[0] = {.mode = kernels::ComparisonRunMode::kFull,
                  .state = source_.state,
                  .active_stage = active_stage_,
                  .expected_boundary = &inventory_->source_boundary};
  requests_[1] = {.mode = kernels::ComparisonRunMode::kFull,
                  .state = source_.state,
                  .active_stage = active_stage_,
                  .expected_boundary = &inventory_->source_boundary};
  return {};
}

std::expected<void, ComparisonPairFailure> ComparisonPair::InstallBindings(
    std::array<std::span<const kernels::ComparisonBindingRange>, 2> bindings) {
  if (!inventory_ || closed_ || lowering_started_ || bindings_installed_)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                   "actual bindings must be installed once after planning"));
  std::uint64_t additional = 0;
  for (const auto ranges : bindings) {
    if (ranges.size() >
        std::numeric_limits<std::uint64_t>::max() / sizeof(kernels::ComparisonBindingRange))
      return std::unexpected(
          Failure(ComparisonPairFailure::Code::kAdmission, "actual binding inventory overflows"));
    if (!AddBytes(
            static_cast<std::uint64_t>(ranges.size()) * sizeof(kernels::ComparisonBindingRange),
            additional) ||
        (ranges.size() != 0 && !AddBytes(kAllocatorEnvelope, additional)))
      return std::unexpected(
          Failure(ComparisonPairFailure::Code::kAdmission, "actual binding inventory overflows"));
  }
  if (!package_loan_.Grow(additional))
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kAdmission, "actual binding admission was refused"));
  std::array<std::vector<kernels::ComparisonBindingRange>, 2> next;
  for (std::size_t arm = 0; arm < bindings.size(); ++arm) {
    next[arm].reserve(bindings[arm].size());
    next[arm].insert(next[arm].end(), bindings[arm].begin(), bindings[arm].end());
    if (next[arm].capacity() > bindings[arm].size())
      return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                     "actual binding capacity exceeded its paid inventory"));
  }
  for (std::size_t arm = 0; arm < bindings_.size(); ++arm) {
    bindings_[arm].swap(next[arm]);
    requests_[arm].bindings = bindings_[arm];
  }
  bindings_installed_ = true;
  return {};
}

std::expected<kernels::LowerInput, ComparisonPairFailure> ComparisonPair::LoweringInput(
    ComparisonPairArm arm, const kernels::Addresses& addresses) {
  if (!inventory_ || !alternative_plan_ || closed_ || !bindings_installed_)
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kNotReady,
                "comparison plans and actual bindings are required for lowering"));
  const auto* owned_plan = plan(arm);
  if (!owned_plan)
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kInvalid, "unknown comparison arm"));
  if (!owned_plan->final_origins ||
      owned_plan->final_node_origins.source_graph_identity != source_.graph.identity() ||
      owned_plan->final_node_origins.planned_graph_identity != owned_plan->plan.graph.identity())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "comparison plan lost its authenticated final lineage"));
  auto& capture = Capture(arm);
  capture.seed = seed_boundary_ ? &*seed_boundary_ : nullptr;
  if (capture.seed == nullptr)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kNotReady,
                                   "comparison seed boundary was not prepared"));
  // The capture's DAG and final map are selected below from the actual plan
  // owners. The incumbent DAG is the comparison checkpoint's planned owner;
  // the alternative retains its own transformed DAG.
  if (arm == ComparisonPairArm::kIncumbent) {
    capture.origins = &inventory_->incumbent.final_origins.value();
  } else {
    capture.origins = &alternative_plan_->final_origins.value();
  }
  capture.final_nodes = &owned_plan->final_node_origins;
  lowering_started_ = true;
  if (input_domain_ == ComparisonInputDomain::kComponentPorts)
    return kernels::LowerInput(owned_plan->plan, components::GraphIO(source_), addresses,
                               std::nullopt, &Request(arm), &capture);
  return kernels::LowerInput(owned_plan->plan, source_, shape_, addresses, context_, {},
                             &Request(arm), &capture);
}

std::expected<void, ComparisonPairFailure> ComparisonPair::InstallPrograms(
    std::unique_ptr<kernels::StepProgram> incumbent,
    std::unique_ptr<kernels::StepProgram> alternative) {
  if (!inventory_ || !alternative_plan_ || !bindings_installed_ || !lowering_started_ ||
      !incumbent || !alternative || incumbent_.program || alternative_.program || closed_ ||
      !incumbent->bound() || !alternative->bound())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                   "comparison programs are incomplete or already installed"));
  const auto authentic = [&](const kernels::StepProgram& program,
                             const planner::OwnedComparisonPlan& owned_plan) {
    const auto recipe = program.comparison_recipe();
    const auto identity = program.comparison_plan_identity();
    if (!recipe || recipe->mode != kernels::ComparisonRunMode::kFull || !identity) return false;
    return SameDigest(recipe->subject, inventory_->source_boundary.source_digest()) &&
           SameDigest(recipe->planned_graph, owned_plan.plan.graph.Digest()) &&
           SameDigest(recipe->policy, owned_plan.plan.policy_digest) &&
           SameDigest(recipe->registry, registry_digest_) &&
           identity->source_graph_identity == source_.graph.identity() &&
           identity->planned_graph_identity == owned_plan.plan.graph.identity() &&
           owned_plan.final_node_origins.source_graph_identity == identity->source_graph_identity &&
           owned_plan.final_node_origins.planned_graph_identity == identity->planned_graph_identity;
  };
  if (!authentic(*incumbent, inventory_->incumbent) || !authentic(*alternative, *alternative_plan_))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "bound programs do not match their retained plans"));
  incumbent_.resources = incumbent->RetainedResources();
  alternative_.resources = alternative->RetainedResources();
  incumbent_.program = std::move(incumbent);
  alternative_.program = std::move(alternative);
  return {};
}

std::expected<void, ComparisonPairFailure> ComparisonPair::InstallDescriptorPrograms(
    std::unique_ptr<kernels::StepProgram> incumbent,
    std::unique_ptr<kernels::StepProgram> alternative) {
  if (!inventory_ || !alternative_plan_ || !bindings_installed_ || !lowering_started_ ||
      !incumbent || !alternative || incumbent_.program || alternative_.program || closed_ ||
      descriptor_programs_installed_ || incumbent->bound() || alternative->bound())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                   "descriptor programs are incomplete or already installed"));
  const auto authentic = [&](const kernels::StepProgram& program,
                             const planner::OwnedComparisonPlan& owned_plan) {
    const auto recipe = program.comparison_recipe();
    const auto identity = program.comparison_plan_identity();
    if (!recipe || recipe->mode != kernels::ComparisonRunMode::kFull || !identity) return false;
    return SameDigest(recipe->subject, inventory_->source_boundary.source_digest()) &&
           SameDigest(recipe->planned_graph, owned_plan.plan.graph.Digest()) &&
           SameDigest(recipe->policy, owned_plan.plan.policy_digest) &&
           SameDigest(recipe->registry, registry_digest_) &&
           identity->source_graph_identity == source_.graph.identity() &&
           identity->planned_graph_identity == owned_plan.plan.graph.identity() &&
           owned_plan.final_node_origins.source_graph_identity == identity->source_graph_identity &&
           owned_plan.final_node_origins.planned_graph_identity == identity->planned_graph_identity;
  };
  if (!authentic(*incumbent, inventory_->incumbent) || !authentic(*alternative, *alternative_plan_))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "unbound programs do not match their retained plans"));
  incumbent_.resources = incumbent->RetainedResources();
  alternative_.resources = alternative->RetainedResources();
  incumbent_.program = std::move(incumbent);
  alternative_.program = std::move(alternative);
  descriptor_programs_installed_ = true;
  return {};
}

std::expected<void, ComparisonPairFailure> ComparisonPair::BindPrograms(
    kernels::StepLaunch& incumbent_launch, kernels::StepLaunch& alternative_launch) {
  if (!descriptor_programs_installed_ || !bindings_installed_ || !lowering_started_ || closed_ ||
      measurement_bind_attempted_ || measurement_bind_succeeded_ || !descriptor_key_ ||
      !descriptor_seals_[0] || !descriptor_seals_[1] || !boundary_ || !incumbent_.program ||
      !alternative_.program || incumbent_.projection || alternative_.projection ||
      (descriptor_preparation_ == ComparisonRecipePreparation::kStatelessReadyBoundaryOperands &&
       (!descriptor_certificate_seals_[0] || !descriptor_certificate_seals_[1])) ||
      incumbent_.recording || alternative_.recording || incumbent_.program->bound() ||
      alternative_.program->bound())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kNotReady,
                                   "a successful unbound descriptor pair is required"));

  auto valid = ValidatePlans();
  if (!valid) return std::unexpected(valid.error());
  const auto& key = *descriptor_key_;
  const auto preparation_domain =
      ComparisonPreparationDomain(descriptor_preparation_, shape_.phase);
  const auto& incumbent_seal = *descriptor_seals_[0];
  const auto& alternative_seal = *descriptor_seals_[1];
  const auto valid_seal = [&](const kernels::ComparisonRegionRecipeSeal& seal,
                              const planner::OwnedComparisonPlan& owned_plan) {
    return seal.version == 3 && seal.state == kernels::ComparisonProofState::kDescriptorVerified &&
           seal.missing == kernels::ComparisonMissing::kNone &&
           SameDigest(seal.subject, boundary_->digest().logical) &&
           SameDigest(seal.boundary, boundary_->digest().combined) &&
           SameDigest(seal.policy, owned_plan.plan.policy_digest) &&
           SameDigest(seal.registry, registry_digest_) && NonzeroDigest(seal.backend_device);
  };
  const auto authentic = [&](const kernels::StepProgram& program,
                             const planner::OwnedComparisonPlan& owned_plan,
                             const kernels::ComparisonRegionRecipeSeal& seal) {
    const auto recipe = program.comparison_recipe();
    const auto identity = program.comparison_plan_identity();
    return recipe && recipe->mode == kernels::ComparisonRunMode::kFull && identity &&
           SameDigest(recipe->subject, inventory_->source_boundary.source_digest()) &&
           SameDigest(recipe->planned_graph, owned_plan.plan.graph.Digest()) &&
           SameDigest(recipe->policy, owned_plan.plan.policy_digest) &&
           SameDigest(recipe->registry, registry_digest_) &&
           identity->source_graph_identity == source_.graph.identity() &&
           identity->planned_graph_identity == owned_plan.plan.graph.identity() &&
           owned_plan.final_node_origins.source_graph_identity == identity->source_graph_identity &&
           owned_plan.final_node_origins.planned_graph_identity ==
               identity->planned_graph_identity &&
           key.policy == seal.policy && key.selection_environment == seal.registry;
  };
  if (!preparation_domain || boundary_->source_identity() != source_.graph.identity() ||
      key.version != 1 ||
      key.region_canonicalizer != components::kComparisonRegionCanonicalizerVersion ||
      key.discovery_operation != inventory_->site.registered_result ||
      key.mode != execution::RecipeRunMode::kFull ||
      !SameDigest(key.subject, boundary_->digest().logical) ||
      !SameDigest(key.boundary, boundary_->digest().combined) ||
      !SameDigest(key.policy, incumbent_seal.policy) ||
      !SameDigest(key.policy, alternative_seal.policy) ||
      !SameDigest(key.device, incumbent_seal.backend_device) ||
      !SameDigest(key.device, alternative_seal.backend_device) ||
      !SameDigest(key.selection_environment, registry_digest_) ||
      !SameDigest(key.preparation_domain, *preparation_domain) ||
      key.incumbent != ArmIdentity(incumbent_seal) ||
      key.alternative != ArmIdentity(alternative_seal) ||
      !valid_seal(incumbent_seal, inventory_->incumbent) ||
      !valid_seal(alternative_seal, *alternative_plan_) ||
      !authentic(*incumbent_.program, inventory_->incumbent, incumbent_seal) ||
      !authentic(*alternative_.program, *alternative_plan_, alternative_seal))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "saved descriptor key or live plan identity changed"));

  // A failed first or second Bind is terminal. The owners remain together;
  // no synthetic unbind or retry can restore the descriptor contract.
  measurement_bind_attempted_ = true;
  if ((incumbent_.resources && incumbent_.resources.launch() != &incumbent_launch) ||
      (alternative_.resources && alternative_.resources.launch() != &alternative_launch))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                   "measurement launch differs from retained program resources"));
  const auto bind_one = [](Arm& arm, kernels::StepLaunch& launch) {
    auto result = arm.program->Bind(launch);
    if (!result) return false;
    if (!arm.program->bound()) return false;
    auto retained = arm.program->RetainedResources();
    if (retained) {
      if (retained.launch() != &launch) return false;
      arm.resources = std::move(retained);
    }
    return true;
  };
  if (!bind_one(incumbent_, incumbent_launch))
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kUnproved, "incumbent measurement Bind failed"));
  if (!bind_one(alternative_, alternative_launch))
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kUnproved, "alternative measurement Bind failed"));

  if (descriptor_preparation_ == ComparisonRecipePreparation::kStatelessReadyBoundaryOperands) {
    for (const auto arm : {ComparisonPairArm::kIncumbent, ComparisonPairArm::kAlternative}) {
      const auto index = arm == ComparisonPairArm::kIncumbent ? 0U : 1U;
      auto& program = *GetArm(arm).program;
      const auto cert = program.ProjectComparisonRegionCertificate(
          *boundary_, kernels::ComparisonCertificatePhase::kBound);
      if (cert.status != kernels::ComparisonRegionStatus::kClosed ||
          cert.seal.state != kernels::ComparisonProofState::kBoundVerified ||
          cert.seal.missing != kernels::ComparisonMissing::kNone ||
          !descriptor_certificate_seals_[index] ||
          !SamePersistentSeal(*descriptor_certificate_seals_[index], cert.seal))
        return std::unexpected(
            Failure(ComparisonPairFailure::Code::kUnproved,
                    "bound applicability certificate differs from its descriptor"));
      bound_certificate_seals_[index].emplace(cert.seal);
    }
  }
  const auto key_arm_matches = [&](std::size_t index,
                                   const kernels::ComparisonRegionRecipeSeal& seal) {
    const auto expected = ArmIdentity(seal);
    return index == 0 ? key.incumbent == expected : key.alternative == expected;
  };
  if (!key_arm_matches(0, incumbent_seal) || !key_arm_matches(1, alternative_seal) ||
      !authentic(*incumbent_.program, inventory_->incumbent, incumbent_seal) ||
      !authentic(*alternative_.program, *alternative_plan_, alternative_seal) ||
      !incumbent_.program->bound() || !alternative_.program->bound())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "measurement Bind changed the retained live plan identity"));
  measurement_bind_succeeded_ = true;
  return {};
}

class ComparisonPair::SelectedRecipeProgram final : public kernels::StepProgram {
 public:
  SelectedRecipeProgram(ir::StorageLoan owner_loan,
                        components::OwnedComparisonRegionBoundary&& boundary,
                        std::unique_ptr<kernels::StepProgram> program,
                        kernels::ComparisonRegionRecipeSeal descriptor_seal,
                        kernels::ComparisonPlanIdentity plan_identity,
                        ComparisonRecipePreparation preparation,
                        std::optional<kernels::ComparisonRegionRecipeSeal> certificate_seal)
      : owner_loan_(std::move(owner_loan)),
        boundary_(std::move(boundary)),
        program_(std::move(program)),
        descriptor_seal_(descriptor_seal),
        plan_identity_(plan_identity),
        preparation_(preparation),
        certificate_seal_(std::move(certificate_seal)),
        retained_metadata_bytes_(owner_loan_.bytes()) {
    const auto resources = program_->RetainedResources();
    resources_adopted_ = !resources || HoldResources(resources);
  }

  bool resources_adopted() const { return resources_adopted_; }

  void ConstructionPaid() override {
    program_->ConstructionPaid();
    boundary_.ConstructionPaid();
    owner_loan_.Reset();
    kernels::StepProgram::ConstructionPaid();
  }
  std::uint64_t scratch_bytes() const override { return program_->scratch_bytes(); }
  std::uint64_t pool_bytes() const override { return program_->pool_bytes(); }
  std::uint64_t private_pool_bytes() const override { return program_->private_pool_bytes(); }
  std::uint64_t host_bytes() const override {
    return program_->host_bytes() + retained_metadata_bytes_;
  }
  std::uint64_t launches() const override { return program_->launches(); }
  std::span<const kernels::InputSlot> inputs() const override { return program_->inputs(); }
  std::span<const kernels::OutputSlot> outputs() const override { return program_->outputs(); }
  std::span<const kernels::TensorUse> tensors() const override { return program_->tensors(); }
  const execution::BoundPlan* bound() const override {
    return promoted_ ? program_->bound() : nullptr;
  }
  std::optional<kernels::ComparisonRecipeSeal> comparison_recipe() const override {
    return program_->comparison_recipe();
  }
  std::optional<kernels::ComparisonPlanIdentity> comparison_plan_identity() const override {
    const auto identity = program_->comparison_plan_identity();
    if (!identity || identity->source_graph_identity != plan_identity_.source_graph_identity ||
        identity->planned_graph_identity != plan_identity_.planned_graph_identity)
      return {};
    return identity;
  }
  kernels::ComparisonRegionResult ProjectComparisonRegion(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    if (!promoted_)
      return {.status = kernels::ComparisonRegionStatus::kUnproved,
              .reason = "selected recipe is not bound and revalidated"};
    return program_->ProjectComparisonRegion(boundary);
  }
  kernels::ComparisonRegionDescriptorResult ProjectComparisonRegionDescriptor(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    if (promoted_)
      return {.status = kernels::ComparisonRegionStatus::kUnproved,
              .reason = "selected recipe is already bound"};
    return program_->ProjectComparisonRegionDescriptor(boundary);
  }
  kernels::ComparisonRegionDescriptorResult ProjectComparisonRegionCertificate(
      const components::OwnedComparisonRegionBoundary& boundary,
      kernels::ComparisonCertificatePhase phase) const override {
    if (preparation_ != ComparisonRecipePreparation::kStatelessReadyBoundaryOperands ||
        !promoted_ || phase != kernels::ComparisonCertificatePhase::kBound)
      return {.status = kernels::ComparisonRegionStatus::kUnproved,
              .reason = "selected recipe has no current ready-boundary certificate"};
    return program_->ProjectComparisonRegionCertificate(boundary, phase);
  }
  bool ValidateComparisonProjection(
      const kernels::ComparisonRegionProjection& projection) const override {
    return promoted_ && program_->ValidateComparisonProjection(projection);
  }
  std::expected<void, kernels::StepFailure> RunComparisonPrefix(
      const kernels::ComparisonRegionProjection& projection, kernels::StepLaunch& launch,
      providers::NativeStream stream) const override {
    if (!promoted_)
      return std::unexpected(kernels::StepFailure{.detail = "selected recipe is not bound"});
    return program_->RunComparisonPrefix(projection, launch, stream);
  }
  std::expected<void, kernels::StepFailure> RunComparisonRegion(
      const kernels::ComparisonRegionProjection& projection, kernels::StepLaunch& launch,
      providers::NativeStream stream) const override {
    if (!promoted_)
      return std::unexpected(kernels::StepFailure{.detail = "selected recipe is not bound"});
    return program_->RunComparisonRegion(projection, launch, stream);
  }
  void ReportPlan(const kernels::PlanReportSink& sink) const override {
    program_->ReportPlan(sink);
  }
  void StreamPlanReport(const kernels::PlanReportSink& sink) const override {
    program_->StreamPlanReport(sink);
  }
  kernels::PlanCoverage plan_coverage() const override { return program_->plan_coverage(); }
  bool capturable() const override { return program_->capturable(); }
  std::optional<std::span<const kernels::RoutedWeight>> weight_reads() const override {
    return program_->weight_reads();
  }
  kernels::RoutedProgram* routed() override { return promoted_ ? program_->routed() : nullptr; }
  const kernels::RoutedProgram* routed() const override {
    return promoted_ ? program_->routed() : nullptr;
  }
  std::expected<void, kernels::StepFailure> Bind(kernels::StepLaunch& launch) override {
    promoted_ = false;
    auto bound = program_->Bind(launch);
    if (!bound) return std::unexpected(std::move(bound.error()));
    const auto identity = program_->comparison_plan_identity();
    if (!identity || identity->source_graph_identity != plan_identity_.source_graph_identity ||
        identity->planned_graph_identity != plan_identity_.planned_graph_identity ||
        !boundary_.source_identity() ||
        identity->source_graph_identity != boundary_.source_identity())
      return std::unexpected(
          kernels::StepFailure{.detail = "selected recipe live plan identity changed"});
    if (preparation_ == ComparisonRecipePreparation::kStatelessReadyBoundaryOperands) {
      const auto certificate = program_->ProjectComparisonRegionCertificate(
          boundary_, kernels::ComparisonCertificatePhase::kBound);
      if (!certificate_seal_ || certificate.status != kernels::ComparisonRegionStatus::kClosed ||
          certificate.seal.state != kernels::ComparisonProofState::kBoundVerified ||
          certificate.seal.missing != kernels::ComparisonMissing::kNone ||
          !SamePersistentSeal(*certificate_seal_, certificate.seal))
        return std::unexpected(kernels::StepFailure{
            .detail = "selected recipe applicability certificate changed before Bind"});
    }
    auto projection = program_->ProjectComparisonRegion(boundary_);
    if (projection.status != kernels::ComparisonRegionStatus::kClosed || !projection.projection ||
        projection.projection->seal().state != kernels::ComparisonProofState::kBoundVerified ||
        projection.projection->seal().missing != kernels::ComparisonMissing::kNone ||
        !SamePersistentSeal(descriptor_seal_, projection.projection->seal()) ||
        (certificate_seal_ &&
         !SamePersistentSeal(*certificate_seal_, projection.projection->seal())) ||
        !program_->ValidateComparisonProjection(*projection.projection) ||
        !MatchesBoundaryPorts(boundary_, projection.value_ports))
      return std::unexpected(
          kernels::StepFailure{.detail = "selected recipe changed before bound promotion"});
    promoted_ = true;
    return {};
  }
  std::expected<void, kernels::StepFailure> TryUpdateGeneratedInputs(
      const planner::StepShape& shape) override {
    if (!promoted_)
      return std::unexpected(kernels::StepFailure{.detail = "selected recipe is not bound"});
    return program_->TryUpdateGeneratedInputs(shape);
  }
  std::expected<void, kernels::StepFailure> Position(std::span<const std::uint32_t> past) override {
    if (!promoted_)
      return std::unexpected(kernels::StepFailure{.detail = "selected recipe is not bound"});
    return program_->Position(past);
  }
  std::expected<void, kernels::StepFailure> Run(kernels::StepLaunch& launch,
                                                providers::NativeStream stream) const override {
    if (!promoted_)
      return std::unexpected(kernels::StepFailure{.detail = "selected recipe is not bound"});
    return program_->Run(launch, stream);
  }
  bool warm() const override { return program_->warm(); }
  std::expected<void, kernels::StepFailure> RunWarm(kernels::StepLaunch& launch,
                                                    providers::NativeStream stream) const override {
    if (!promoted_)
      return std::unexpected(kernels::StepFailure{.detail = "selected recipe is not bound"});
    return program_->RunWarm(launch, stream);
  }
  std::expected<std::unique_ptr<kernels::Recorded>, kernels::StepFailure> Record(
      kernels::StepLaunch& launch, providers::NativeStream stream,
      const kernels::StepLaunch::Body& body) const override {
    if (!promoted_)
      return std::unexpected(kernels::StepFailure{.detail = "selected recipe is not bound"});
    return program_->Record(launch, stream, body);
  }

 private:
  // The selected program dies before the boundary; its externally paid loan
  // outlives both. The source/plan package and losing arm are released at
  // selection, because lowering has snapshotted their required facts.
  ir::StorageLoan owner_loan_;
  components::OwnedComparisonRegionBoundary boundary_;
  std::unique_ptr<kernels::StepProgram> program_;
  kernels::ComparisonRegionRecipeSeal descriptor_seal_{};
  kernels::ComparisonPlanIdentity plan_identity_{};
  ComparisonRecipePreparation preparation_ =
      ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody;
  std::optional<kernels::ComparisonRegionRecipeSeal> certificate_seal_;
  const std::uint64_t retained_metadata_bytes_;
  bool resources_adopted_ = false;
  bool promoted_ = false;
};

std::expected<std::optional<execution::RecipeComparisonKey>, ComparisonPairFailure>
DescribeRecipeComparison(ComparisonPair& pair, const execution::Registry& registry) {
  return DescribeRecipeComparison(
      pair, registry,
      ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody);
}

std::expected<std::optional<execution::RecipeComparisonKey>, ComparisonPairFailure>
DescribeRecipeComparison(ComparisonPair& pair, const execution::Registry& registry,
                         ComparisonRecipePreparation preparation) {
  if (!pair.descriptor_programs_installed_ || !pair.inventory_ || !pair.alternative_plan_ ||
      !pair.incumbent_.program || !pair.alternative_.program || pair.closed_ ||
      pair.measurement_bind_attempted_)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kNotReady,
                                   "descriptor-ready comparison programs are required"));
  pair.descriptor_key_.reset();
  pair.descriptor_preparation_ = preparation;
  pair.descriptor_seals_ = {};
  pair.descriptor_certificate_seals_ = {};
  pair.bound_certificate_seals_ = {};
  auto valid = pair.ValidatePlans();
  if (!valid) return std::unexpected(valid.error());
  if (registry.SelectionDigest() != pair.registry_digest_)
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kStale, "comparison registry changed after planning"));
  auto seeded = pair.BuildSeedBoundary();
  if (!seeded) return std::unexpected(seeded.error());

  const auto live_matches = [&](ComparisonPairArm arm) {
    const auto* owned_plan = pair.plan(arm);
    const auto identity = pair.GetArm(arm).program->comparison_plan_identity();
    return owned_plan && identity && owned_plan->final_origins &&
           identity->source_graph_identity == pair.source_.graph.identity() &&
           identity->planned_graph_identity == owned_plan->plan.graph.identity() &&
           owned_plan->final_node_origins.source_graph_identity ==
               identity->source_graph_identity &&
           owned_plan->final_node_origins.planned_graph_identity ==
               identity->planned_graph_identity;
  };
  if (!live_matches(ComparisonPairArm::kIncumbent) ||
      !live_matches(ComparisonPairArm::kAlternative))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "comparison live plan identity changed before projection"));

  const bool ready_boundary =
      preparation == ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  if (!ready_boundary &&
      preparation !=
          ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody)
    return {};
  auto closure = CloseCommonBoundary(
      pair.boundary_, pair.membership_, pair.additions_,
      [&](const components::OwnedComparisonRegionBoundary& boundary) {
        if (!ready_boundary) {
          auto left = pair.incumbent_.program->ProjectComparisonRegionDescriptor(boundary);
          auto right = pair.alternative_.program->ProjectComparisonRegionDescriptor(boundary);
          return std::pair(std::move(left), std::move(right));
        }

        // Descriptor and applicability analyzers may reuse their own borrowed
        // expansion buffers. Copy each exact requested member into this
        // already-paid source-sized scratch before asking the same program for
        // its second proof. The common closure loop then unions both arms.
        auto& requested = pair.candidate_members_;
        requested.clear();
        bool any_expand = false;
        bool any_unproved = false;
        const auto append = [&](std::span<const ir::NodeId> members) {
          for (const auto node : members) {
            if (!node.valid() || node.v >= pair.source_.graph.nodes().size() ||
                (std::ranges::find(requested, node) == requested.end() &&
                 requested.size() == requested.capacity()))
              return false;
            if (std::ranges::find(requested, node) == requested.end()) requested.push_back(node);
          }
          return true;
        };
        const auto project_arm = [&](ComparisonPairArm arm, std::size_t index) {
          auto& program = *pair.GetArm(arm).program;
          auto executable = program.ProjectComparisonRegionDescriptor(boundary);
          if (executable.status == kernels::ComparisonRegionStatus::kUnproved) {
            any_unproved = true;
            return;
          }
          if (executable.status == kernels::ComparisonRegionStatus::kExpand) {
            any_expand = true;
            if (!append(executable.required_members)) any_unproved = true;
          } else {
            pair.descriptor_seals_[index].emplace(executable.seal);
          }
          auto applicability = program.ProjectComparisonRegionCertificate(
              boundary, kernels::ComparisonCertificatePhase::kDescriptor);
          if (applicability.status == kernels::ComparisonRegionStatus::kUnproved) {
            any_unproved = true;
            return;
          }
          if (applicability.status == kernels::ComparisonRegionStatus::kExpand) {
            any_expand = true;
            if (!append(applicability.required_members)) any_unproved = true;
          } else {
            pair.descriptor_certificate_seals_[index].emplace(applicability.seal);
          }
        };
        project_arm(ComparisonPairArm::kIncumbent, 0);
        project_arm(ComparisonPairArm::kAlternative, 1);
        if (any_unproved) {
          return std::pair(
              kernels::ComparisonRegionDescriptorResult{
                  .status = kernels::ComparisonRegionStatus::kUnproved},
              kernels::ComparisonRegionDescriptorResult{
                  .status = kernels::ComparisonRegionStatus::kUnproved});
        }
        if (any_expand) {
          const auto span = std::span<const ir::NodeId>(requested);
          return std::pair(
              kernels::ComparisonRegionDescriptorResult{
                  .status = kernels::ComparisonRegionStatus::kExpand, .required_members = span},
              kernels::ComparisonRegionDescriptorResult{
                  .status = kernels::ComparisonRegionStatus::kExpand, .required_members = span});
        }
        return std::pair(
            kernels::ComparisonRegionDescriptorResult{
                .status = kernels::ComparisonRegionStatus::kClosed,
                .seal = *pair.descriptor_seals_[0]},
            kernels::ComparisonRegionDescriptorResult{
                .status = kernels::ComparisonRegionStatus::kClosed,
                .seal = *pair.descriptor_seals_[1]});
      },
      [&] {
        pair.descriptor_seals_ = {};
        pair.descriptor_certificate_seals_ = {};
      },
      [&] { return pair.BuildCandidateBoundary(); },
      [&](kernels::ComparisonRegionDescriptorResult&& left,
          kernels::ComparisonRegionDescriptorResult&& right) {
        if (!ready_boundary && left.status == kernels::ComparisonRegionStatus::kClosed)
          pair.descriptor_seals_[0].emplace(left.seal);
        if (!ready_boundary && right.status == kernels::ComparisonRegionStatus::kClosed)
          pair.descriptor_seals_[1].emplace(right.seal);
      });
  if (!live_matches(ComparisonPairArm::kIncumbent) ||
      !live_matches(ComparisonPairArm::kAlternative))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "comparison live plan identity changed during projection"));
  if (!closure) return std::unexpected(closure.error());
  if (!*closure || !pair.descriptor_seals_[0] || !pair.descriptor_seals_[1] ||
      (ready_boundary &&
       (!pair.descriptor_certificate_seals_[0] || !pair.descriptor_certificate_seals_[1])))
    return {};

  const auto& left = *pair.descriptor_seals_[0];
  const auto& right = *pair.descriptor_seals_[1];
  const auto& left_certificate = ready_boundary ? *pair.descriptor_certificate_seals_[0] : left;
  const auto& right_certificate = ready_boundary ? *pair.descriptor_certificate_seals_[1] : right;
  const auto valid_local = [&](const kernels::ComparisonRegionRecipeSeal& seal,
                               const planner::OwnedComparisonPlan& owned_plan) {
    return seal.version == 3 && seal.state == kernels::ComparisonProofState::kDescriptorVerified &&
           seal.missing == kernels::ComparisonMissing::kNone &&
           SameDigest(seal.subject, pair.boundary_->digest().logical) &&
           SameDigest(seal.boundary, pair.boundary_->digest().combined) &&
           SameDigest(seal.policy, owned_plan.plan.policy_digest) &&
           SameDigest(seal.registry, pair.registry_digest_) && NonzeroDigest(seal.planned_region) &&
           NonzeroDigest(seal.physical_recipe) && NonzeroDigest(seal.policy) &&
           NonzeroDigest(seal.registry) && NonzeroDigest(seal.backend_recipe) &&
           NonzeroDigest(seal.backend_build) && NonzeroDigest(seal.backend_selection) &&
           NonzeroDigest(seal.backend_device);
  };
  if (!valid_local(left, pair.inventory_->incumbent) ||
      !valid_local(right, *pair.alternative_plan_) ||
      (ready_boundary &&
       (!valid_local(left_certificate, pair.inventory_->incumbent) ||
        !valid_local(right_certificate, *pair.alternative_plan_) ||
        !SamePersistentSeal(left, left_certificate) ||
        !SamePersistentSeal(right, right_certificate) ||
        left_certificate.state != kernels::ComparisonProofState::kDescriptorVerified ||
        right_certificate.state != kernels::ComparisonProofState::kDescriptorVerified ||
        !SameDigest(left.subject, left_certificate.subject) ||
        !SameDigest(right.subject, right_certificate.subject) ||
        !SameDigest(left.boundary, left_certificate.boundary) ||
        !SameDigest(right.boundary, right_certificate.boundary) ||
        !SameDigest(left.policy, left_certificate.policy) ||
        !SameDigest(right.policy, right_certificate.policy) ||
        !SameDigest(left.registry, left_certificate.registry) ||
        !SameDigest(right.registry, right_certificate.registry) ||
        !SameDigest(left.backend_device, left_certificate.backend_device) ||
        !SameDigest(right.backend_device, right_certificate.backend_device))) ||
      pair.boundary_->source_identity() != pair.source_.graph.identity() ||
      !SameDigest(left.subject, right.subject) || !SameDigest(left.boundary, right.boundary) ||
      !SameDigest(left.policy, right.policy) || !SameDigest(left.registry, right.registry) ||
      !SameDigest(left.backend_device, right.backend_device))
    return {};

  if (!pair.source_.state.empty() || !pair.source_.saves.empty() ||
      !pair.source_.owner_products.empty() ||
      !HasOnlyOrdinaryRestagedInputs(pair.source_, ready_boundary) ||
      (ready_boundary &&
       (!HasExactReadyValueSlots(pair.source_, pair.incumbent_.program->inputs()) ||
        !HasExactReadyValueSlots(pair.source_, pair.alternative_.program->inputs()))) ||
      std::ranges::any_of(pair.incumbent_.program->inputs(),
                          [ready_boundary](const kernels::InputSlot& slot) {
                            return slot.kind == kernels::InputKind::kStoreCells ||
                                   (!ready_boundary &&
                                    slot.kind == kernels::InputKind::kFeatures) ||
                                   slot.kind == kernels::InputKind::kHostComputed ||
                                   slot.kind == kernels::InputKind::kHadamard ||
                                   slot.kind == kernels::InputKind::kVisibleCells;
                          }) ||
      std::ranges::any_of(
          pair.alternative_.program->inputs(), [ready_boundary](const kernels::InputSlot& slot) {
            return slot.kind == kernels::InputKind::kStoreCells ||
                   (!ready_boundary && slot.kind == kernels::InputKind::kFeatures) ||
                   slot.kind == kernels::InputKind::kHostComputed ||
                   slot.kind == kernels::InputKind::kHadamard ||
                   slot.kind == kernels::InputKind::kVisibleCells;
          }))
    return {};
  if (ready_boundary) {
    auto key = RecipeComparisonKeyForReadyCertificates(
        *pair.boundary_, pair.inventory_->site.registered_result, pair.shape_.phase,
        pair.registry_digest_, left_certificate, right_certificate);
    if (!key) return {};
    pair.descriptor_key_.emplace(*key);
    return key;
  }
  const auto preparation_domain = ComparisonPreparationDomain(preparation, pair.shape_.phase);
  if (!preparation_domain) return {};
  execution::RecipeComparisonKey key;
  key.version = 1;
  key.region_canonicalizer = components::kComparisonRegionCanonicalizerVersion;
  key.discovery_operation = pair.inventory_->site.registered_result;
  key.mode = execution::RecipeRunMode::kFull;
  key.subject = pair.boundary_->digest().logical;
  key.boundary = pair.boundary_->digest().combined;
  key.policy = left.policy;
  key.device = left.backend_device;
  key.selection_environment = pair.registry_digest_;
  key.preparation_domain = *preparation_domain;
  key.incumbent = ArmIdentity(left);
  key.alternative = ArmIdentity(right);
  pair.descriptor_key_.emplace(key);
  return std::optional<execution::RecipeComparisonKey>{key};
}

std::expected<std::unique_ptr<kernels::StepProgram>, ComparisonPairFailure> SelectRecipeComparison(
    std::unique_ptr<ComparisonPair> pair, const execution::Registry& registry,
    ComparisonRecipePreparation preparation) {
  if (!pair)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kNotReady,
                                   "descriptor-ready comparison programs are required"));
  const auto fallback =
      [&]() -> std::expected<std::unique_ptr<kernels::StepProgram>, ComparisonPairFailure> {
    auto incumbent = std::move(pair->incumbent_.program);
    pair->alternative_.program.reset();
    pair->alternative_.projection.reset();
    pair->incumbent_.projection.reset();
    return incumbent;
  };
  auto described = DescribeRecipeComparison(*pair, registry, preparation);
  if (!described) return std::unexpected(described.error());
  if (!*described) return fallback();
  const auto& key = **described;
  const auto* measured = registry.FindRecipeComparison(key);
  if (!measured || measured->kind != execution::RecipeComparisonKind::kLatency) return fallback();
  const auto preference =
      execution::ClassifyRecipeLatency(measured->incumbent_ns, measured->alternative_ns);
  if (preference == execution::RecipePreference::kNeutral) return fallback();
  const auto selected_arm = preference == execution::RecipePreference::kAlternative
                                ? ComparisonPairArm::kAlternative
                                : ComparisonPairArm::kIncumbent;
  auto& selected_owner = pair->GetArm(selected_arm);
  auto& loser =
      pair->GetArm(selected_arm == ComparisonPairArm::kIncumbent ? ComparisonPairArm::kAlternative
                                                                 : ComparisonPairArm::kIncumbent);
  const auto* selected_plan = pair->plan(selected_arm);
  const auto live_identity = selected_owner.program->comparison_plan_identity();
  const auto selected_seal =
      *pair->descriptor_seals_[selected_arm == ComparisonPairArm::kIncumbent ? 0 : 1];
  const auto selected_certificate =
      pair->descriptor_preparation_ == ComparisonRecipePreparation::kStatelessReadyBoundaryOperands
          ? pair->descriptor_certificate_seals_[selected_arm == ComparisonPairArm::kIncumbent ? 0
                                                                                              : 1]
          : std::optional<kernels::ComparisonRegionRecipeSeal>{};
  if (!selected_plan || !selected_plan->final_origins || !live_identity || !pair->boundary_)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "selected recipe lost its retained provenance"));
  std::uint64_t retained_bytes = pair->boundary_->storage_bytes();
  if (!AddBytes(sizeof(ComparisonPair::SelectedRecipeProgram), retained_bytes) ||
      !AddBytes(kAllocatorEnvelope, retained_bytes))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "selected recipe owner inventory overflows"));
  std::uint64_t reported_host_bytes = retained_bytes;
  if (!AddBytes(selected_owner.program->host_bytes(), reported_host_bytes))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "selected recipe host footprint overflows"));
  (void)reported_host_bytes;
  auto owner_loan = ir::AcquireStorageLoan(pair->funder_, retained_bytes);
  if (!owner_loan)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "selected recipe owner admission was refused"));
  components::OwnedComparisonRegionBoundary boundary = std::move(*pair->boundary_);
  pair->boundary_.reset();
  // The new owner has admitted the same capacities before the boundary's
  // construction loan is retired. Cache its byte count so ConstructionPaid
  // can release the loan without changing the reported retained footprint.
  boundary.ConstructionPaid();
  auto selected_program = std::move(selected_owner.program);
  loser.program.reset();
  auto* owner = new (std::nothrow) ComparisonPair::SelectedRecipeProgram(
      std::move(*owner_loan), std::move(boundary), std::move(selected_program), selected_seal,
      *live_identity, pair->descriptor_preparation_, selected_certificate);
  if (owner == nullptr)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kAdmission,
                                   "selected recipe owner allocation failed"));
  if (!owner->resources_adopted()) {
    delete owner;
    return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                   "selected recipe resource ownership could not be retained"));
  }
  pair.reset();
  return std::unique_ptr<kernels::StepProgram>(owner);
}

const planner::OwnedComparisonPlan* ComparisonPair::plan(ComparisonPairArm arm) const {
  if (arm == ComparisonPairArm::kIncumbent) return inventory_ ? &inventory_->incumbent : nullptr;
  if (arm == ComparisonPairArm::kAlternative)
    return alternative_plan_ ? &*alternative_plan_ : nullptr;
  return nullptr;
}

ComparisonPair::Arm& ComparisonPair::GetArm(ComparisonPairArm arm) {
  return arm == ComparisonPairArm::kIncumbent ? incumbent_ : alternative_;
}
const ComparisonPair::Arm& ComparisonPair::GetArm(ComparisonPairArm arm) const {
  return arm == ComparisonPairArm::kIncumbent ? incumbent_ : alternative_;
}
kernels::ComparisonRequest& ComparisonPair::Request(ComparisonPairArm arm) {
  return requests_[arm == ComparisonPairArm::kIncumbent ? 0 : 1];
}
kernels::ComparisonRegionCapture& ComparisonPair::Capture(ComparisonPairArm arm) {
  return captures_[arm == ComparisonPairArm::kIncumbent ? 0 : 1];
}

std::expected<void, ComparisonPairFailure> ComparisonPair::BuildSeedBoundary() {
  if (!inventory_ || inventory_->site.accepted_member_origins.empty())
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kInvalid, "comparison site has no accepted members"));
  auto& offsets = candidate_offsets_;
  auto& members = candidate_members_;
  offsets.clear();
  members.clear();
  std::fill(membership_.begin(), membership_.end(), 0);
  offsets.push_back(0);
  for (const auto& group : inventory_->site.accepted_member_origins) {
    if (group.empty())
      return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                     "comparison site has an empty member lineage"));
    for (const auto node : group) {
      if (!node.valid() || node.v >= source_.graph.nodes().size())
        return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                       "comparison site lineage is outside its source graph"));
      if (membership_[node.v])
        return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                       "comparison site repeats a source member"));
      membership_[node.v] = 1;
      members.push_back(node);
    }
    offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }
  // The seed is rebuilt before bound projection as well as during descriptor
  // closure. Preserve the additions already incorporated into the candidate
  // boundary in the membership bitmap; otherwise a legitimate repeated query
  // can be mistaken for progress and append duplicate members.
  for (const auto node : additions_) {
    if (!node.valid() || node.v >= source_.graph.nodes().size())
      return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                     "comparison expansion is outside its source graph"));
    if (membership_[node.v])
      return std::unexpected(Failure(ComparisonPairFailure::Code::kInvalid,
                                     "comparison expansion repeats a seed member"));
    membership_[node.v] = 1;
  }
  if (!seed_boundary_) {
    auto seed =
        components::MakeComparisonRegionBoundary({SourceBoundaryView(), offsets, members}, funder_);
    if (!seed)
      return std::unexpected(
          Failure(seed.error().code == components::ComparisonBoundaryError::kAdmission
                      ? ComparisonPairFailure::Code::kAdmission
                      : ComparisonPairFailure::Code::kInvalid,
                  "comparison seed boundary construction failed"));
    seed_boundary_.emplace(std::move(*seed));
    return BuildCandidateBoundary();
  }
  return {};
}

std::expected<void, ComparisonPairFailure> ComparisonPair::BuildCandidateBoundary() {
  const auto seed_groups = inventory_->site.accepted_member_origins.size();
  const auto seed_count = seed_groups + additions_.size();
  if (seed_count >= std::numeric_limits<std::uint32_t>::max())
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kAdmission, "comparison region group count overflow"));
  auto& offsets = candidate_offsets_;
  auto& members = candidate_members_;
  offsets.clear();
  members.clear();
  offsets.push_back(0);
  for (const auto& group : inventory_->site.accepted_member_origins) {
    for (const auto node : group) members.push_back(node);
    offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }
  for (const auto node : additions_) {
    members.push_back(node);
    offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }
  auto made =
      components::MakeComparisonRegionBoundary({SourceBoundaryView(), offsets, members}, funder_);
  if (!made)
    return std::unexpected(
        Failure(made.error().code == components::ComparisonBoundaryError::kAdmission
                    ? ComparisonPairFailure::Code::kAdmission
                    : ComparisonPairFailure::Code::kInvalid,
                "comparison region boundary construction failed"));
  boundary_.emplace(std::move(*made));
  return {};
}

std::expected<void, ComparisonPairFailure> ComparisonPair::ValidatePlans() {
  if (!inventory_ || !alternative_plan_ || source_.graph.identity() == 0)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                   "comparison plans have no authenticated source"));
  if (!boundary_ || !boundary_->state().empty())
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "the current region contains state effects"));
  for (const auto arm : {ComparisonPairArm::kIncumbent, ComparisonPairArm::kAlternative}) {
    auto* p = arm == ComparisonPairArm::kIncumbent ? &inventory_->incumbent : &*alternative_plan_;
    if (!p || !p->final_origins ||
        p->final_node_origins.source_graph_identity != source_.graph.identity() ||
        p->final_node_origins.planned_graph_identity != p->plan.graph.identity() ||
        p->final_node_origins.node_origins.values.size() != p->plan.graph.nodes().size())
      return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                     "comparison final lineage does not match its plan"));
    for (const auto set : p->final_node_origins.node_origins.values) {
      const auto count = p->final_origins->OriginCount(set);
      if (!count || *count == 0)
        return std::unexpected(Failure(ComparisonPairFailure::Code::kStale,
                                       "comparison plan has unresolved node lineage"));
    }
  }
  return {};
}

std::expected<std::unique_ptr<ComparisonPair>, ComparisonPairFailure> CloseComparisonPair(
    std::unique_ptr<ComparisonPair> pair) {
  if (!pair || !pair->incumbent_.program || !pair->alternative_.program || pair->closed_)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kNotReady,
                                   "both actual comparison programs must be installed"));
  if (pair->descriptor_programs_installed_ &&
      (!pair->measurement_bind_attempted_ || !pair->measurement_bind_succeeded_ ||
       !pair->descriptor_key_))
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kNotReady,
                "descriptor programs must complete the measurement Bind bridge"));
  if (pair->measurement_bind_attempted_ && !pair->measurement_bind_succeeded_)
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kNotReady, "a failed measurement Bind is terminal"));
  auto valid = pair->ValidatePlans();
  if (!valid) return std::unexpected(valid.error());
  if (!pair->source_.state.empty())
    return std::unexpected(
        Failure(ComparisonPairFailure::Code::kUnproved,
                "stateful comparison pairs are outside this coordinator domain"));
  auto seeded = pair->BuildSeedBoundary();
  if (!seeded) return std::unexpected(seeded.error());

  auto closure = CloseCommonBoundary(
      pair->boundary_, pair->membership_, pair->additions_,
      [&](const components::OwnedComparisonRegionBoundary& boundary) {
        auto left = pair->incumbent_.program->ProjectComparisonRegion(boundary);
        auto right = pair->alternative_.program->ProjectComparisonRegion(boundary);
        return std::pair(std::move(left), std::move(right));
      },
      [&] {
        pair->incumbent_.projection.reset();
        pair->alternative_.projection.reset();
        pair->incumbent_.ports = {};
        pair->alternative_.ports = {};
      },
      [&] { return pair->BuildCandidateBoundary(); },
      [&](kernels::ComparisonRegionResult&& left, kernels::ComparisonRegionResult&& right) {
        if (left.status == kernels::ComparisonRegionStatus::kClosed && left.projection) {
          pair->incumbent_.projection.emplace(std::move(*left.projection));
          pair->incumbent_.ports = left.value_ports;
        }
        if (right.status == kernels::ComparisonRegionStatus::kClosed && right.projection) {
          pair->alternative_.projection.emplace(std::move(*right.projection));
          pair->alternative_.ports = right.value_ports;
        }
      });
  if (!closure) return std::unexpected(closure.error());
  if (!*closure || !pair->incumbent_.projection || !pair->alternative_.projection)
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "both bound arms must prove this local region"));

  const auto& left_seal = pair->incumbent_.projection->seal();
  const auto& right_seal = pair->alternative_.projection->seal();
  if (left_seal.state != kernels::ComparisonProofState::kBoundVerified ||
      right_seal.state != kernels::ComparisonProofState::kBoundVerified ||
      left_seal.missing != kernels::ComparisonMissing::kNone ||
      right_seal.missing != kernels::ComparisonMissing::kNone ||
      !SameDigest(left_seal.subject, right_seal.subject) ||
      pair->boundary_->source_identity() != pair->source_.graph.identity() ||
      !SameDigest(left_seal.subject, pair->boundary_->digest().logical) ||
      !SameDigest(left_seal.boundary, right_seal.boundary) ||
      !SameDigest(left_seal.policy, pair->inventory_->incumbent.plan.policy_digest) ||
      !SameDigest(right_seal.policy, pair->alternative_plan_->plan.policy_digest) ||
      !SameDigest(left_seal.policy, right_seal.policy) ||
      !SameDigest(left_seal.registry, pair->registry_digest_) ||
      !SameDigest(right_seal.registry, pair->registry_digest_) ||
      !SameDigest(left_seal.registry, right_seal.registry) ||
      !SameDigest(left_seal.backend_selection, right_seal.backend_selection) ||
      !SameDigest(left_seal.boundary, pair->boundary_->digest().combined) ||
      (pair->descriptor_programs_installed_ &&
       (!pair->descriptor_seals_[0] || !pair->descriptor_seals_[1] ||
        !SamePersistentSeal(left_seal, *pair->descriptor_seals_[0]) ||
        !SamePersistentSeal(right_seal, *pair->descriptor_seals_[1]))) ||
      (pair->descriptor_preparation_ ==
           ComparisonRecipePreparation::kStatelessReadyBoundaryOperands &&
       (!pair->descriptor_certificate_seals_[0] || !pair->descriptor_certificate_seals_[1] ||
        !pair->bound_certificate_seals_[0] || !pair->bound_certificate_seals_[1] ||
        pair->bound_certificate_seals_[0]->state != kernels::ComparisonProofState::kBoundVerified ||
        pair->bound_certificate_seals_[1]->state != kernels::ComparisonProofState::kBoundVerified ||
        !SamePersistentSeal(*pair->descriptor_certificate_seals_[0],
                            *pair->bound_certificate_seals_[0]) ||
        !SamePersistentSeal(*pair->descriptor_certificate_seals_[1],
                            *pair->bound_certificate_seals_[1]) ||
        !SamePersistentSeal(left_seal, *pair->bound_certificate_seals_[0]) ||
        !SamePersistentSeal(right_seal, *pair->bound_certificate_seals_[1]))) ||
      !pair->incumbent_.program->ValidateComparisonProjection(*pair->incumbent_.projection) ||
      !pair->alternative_.program->ValidateComparisonProjection(*pair->alternative_.projection) ||
      !MatchesBoundaryPorts(*pair->boundary_, pair->incumbent_.ports) ||
      !MatchesBoundaryPorts(*pair->boundary_, pair->alternative_.ports) ||
      !SamePorts(pair->incumbent_.ports, pair->alternative_.ports))
    return std::unexpected(Failure(ComparisonPairFailure::Code::kUnproved,
                                   "closed arms do not share the same bound semantic region"));
  pair->closed_ = true;
  return pair;
}

std::expected<void, kernels::StepFailure> ComparisonPair::CaptureRegion(
    ComparisonPairArm arm, kernels::StepLaunch& launch, providers::NativeStream stream,
    std::uint32_t repetitions) {
  auto& a = GetArm(arm);
  if (repetitions == 0 || !closed_ || !a.program || !a.projection || a.recording ||
      (a.resources && a.resources.launch() != &launch) ||
      !a.program->ValidateComparisonProjection(*a.projection))
    return std::unexpected(kernels::StepFailure{});
  if (repetitions > 1 && !a.projection->inputs_preserved())
    return std::unexpected(kernels::StepFailure{});
  auto callback = [this, arm, &launch, stream,
                   repetitions]() -> std::expected<void, kernels::StepFailure> {
    const auto& current = GetArm(arm);
    for (std::uint32_t repetition = 0; repetition < repetitions; ++repetition) {
      if (!closed_ || !current.program || !current.projection ||
          (current.resources && current.resources.launch() != &launch) ||
          !current.program->ValidateComparisonProjection(*current.projection))
        return std::unexpected(kernels::StepFailure{});
      auto run = current.program->RunComparisonRegion(*current.projection, launch, stream);
      if (!run) return std::unexpected(std::move(run.error()));
    }
    return {};
  };
  kernels::StepLaunch::Body body = std::ref(callback);
  auto recorded = a.program->Record(launch, stream, body);
  if (!recorded || !*recorded || (*recorded)->owner() != &launch)
    return std::unexpected(recorded ? kernels::StepFailure{} : std::move(recorded.error()));
  a.recording = std::move(*recorded);
  a.recording_repetitions = repetitions;
  return {};
}

std::expected<void, kernels::StepFailure> ComparisonPair::PrepareFrame(
    ComparisonPairArm arm, kernels::StepLaunch& launch, providers::NativeStream stream) const {
  const auto& a = GetArm(arm);
  if (!closed_ || !a.program || !a.projection || !a.recording || a.recording_repetitions == 0 ||
      a.recording->owner() != &launch || (a.resources && a.resources.launch() != &launch) ||
      !a.program->ValidateComparisonProjection(*a.projection))
    return std::unexpected(kernels::StepFailure{});
  return a.program->RunComparisonPrefix(*a.projection, launch, stream);
}

std::expected<void, kernels::StepFailure> ComparisonPair::ReplayCapturedRegion(
    ComparisonPairArm arm, kernels::StepLaunch& launch, providers::NativeStream stream) const {
  const auto& a = GetArm(arm);
  if (!closed_ || !a.program || !a.projection || !a.recording || a.recording_repetitions == 0 ||
      a.recording->owner() != &launch || (a.resources && a.resources.launch() != &launch) ||
      !a.program->ValidateComparisonProjection(*a.projection))
    return std::unexpected(kernels::StepFailure{});
  return launch.Launch(stream, *a.recording);
}

std::expected<void, kernels::StepFailure> ComparisonPair::RunFrame(
    ComparisonPairArm arm, kernels::StepLaunch& launch, providers::NativeStream stream) const {
  auto prefix = PrepareFrame(arm, launch, stream);
  if (!prefix) return std::unexpected(std::move(prefix.error()));
  return ReplayCapturedRegion(arm, launch, stream);
}

std::optional<CapturedRegionInventory> ComparisonPair::captured_region_inventory(
    ComparisonPairArm arm, const kernels::StepLaunch& launch) const {
  const auto& a = GetArm(arm);
  if (!closed_ || !a.program || !a.projection || !a.recording || a.recording_repetitions == 0 ||
      a.recording->owner() != &launch || (a.resources && a.resources.launch() != &launch) ||
      !a.program->ValidateComparisonProjection(*a.projection))
    return std::nullopt;
  return CapturedRegionInventory{.recorded_nodes = a.recording->nodes(),
                                 .repetitions = a.recording_repetitions};
}

std::expected<double, kernels::StepFailure> ComparisonPair::captured_region_duration_ns(
    ComparisonPairArm arm, const kernels::StepLaunch& launch) const {
  const auto& a = GetArm(arm);
  if (!closed_ || !a.program || !a.projection || !a.recording || a.recording_repetitions == 0 ||
      a.recording->owner() != &launch || (a.resources && a.resources.launch() != &launch) ||
      !a.program->ValidateComparisonProjection(*a.projection))
    return std::unexpected(kernels::StepFailure{});
  return a.recording->gpu_duration_ns();
}

std::span<const kernels::ComparisonRegionValuePort> ComparisonPair::value_ports(
    ComparisonPairArm arm) const {
  const auto& a = GetArm(arm);
  if (!closed_ || !a.projection || !a.program ||
      !a.program->ValidateComparisonProjection(*a.projection))
    return {};
  return a.ports;
}

std::span<const kernels::InputSlot> ComparisonPair::input_slots(ComparisonPairArm arm) const {
  const auto& a = GetArm(arm);
  if (!closed_ || !a.projection || !a.program ||
      !a.program->ValidateComparisonProjection(*a.projection))
    return {};
  return a.program->inputs();
}

}  // namespace llmp::engine
