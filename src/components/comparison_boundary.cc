// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "components/comparison_boundary.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <tuple>
#include <utility>

#include "components/step.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "ir/verify.h"

namespace llmp::components {

static_assert(kComparisonFrameworkComponent == kFrameworkComponent);

OwnedComparisonBoundary::OwnedComparisonBoundary(OwnedComparisonBoundary&& other) noexcept
    : loan_(std::move(other.loan_)),
      digest_(other.digest_),
      source_digest_(other.source_digest_),
      source_identity_(std::exchange(other.source_identity_, 0)),
      source_tensor_count_(std::exchange(other.source_tensor_count_, 0)),
      source_input_count_(std::exchange(other.source_input_count_, 0)),
      source_output_count_(std::exchange(other.source_output_count_, 0)),
      source_keep_count_(std::exchange(other.source_keep_count_, 0)),
      active_stage_(std::exchange(other.active_stage_, 0)),
      roots_(std::move(other.roots_)),
      ports_(std::move(other.ports_)),
      weight_parts_(std::move(other.weight_parts_)),
      state_(std::move(other.state_)),
      state_facts_(std::move(other.state_facts_)),
      external_(std::move(other.external_)),
      unstored_inputs_(std::move(other.unstored_inputs_)) {
  other.digest_ = {};
  other.source_digest_ = {};
}

namespace {

using Digest = base::Sha256Digest;

class Hash {
 public:
  explicit Hash(std::string_view domain) { Text(domain); }
  void U8(std::uint8_t value) { Add(value); }
  void U16(std::uint16_t value) { Add(value); }
  void U32(std::uint32_t value) { Add(value); }
  void U64(std::uint64_t value) { Add(value); }
  void Bytes(std::span<const std::byte> bytes) {
    U64(bytes.size());
    hash_.Update(bytes);
  }
  void Text(std::string_view value) {
    U64(value.size());
    hash_.Update(value);
  }
  void DigestValue(const Digest& value) { hash_.Update(std::as_bytes(std::span(value))); }
  Digest Finish() { return hash_.Finish(); }

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

ComparisonBoundaryError Invalid(std::string_view detail,
                                ir::DiagnosticAdmission* diagnostic = nullptr) {
  auto text = ir::FormatDiagnostic(diagnostic, "{}", detail);
  if (!text) return {.code = ComparisonBoundaryError::kAdmission, .detail = {}};
  return {.code = ComparisonBoundaryError::kInvalid, .detail = std::move(*text)};
}

ComparisonBoundaryError Admission(std::string_view) {
  return {.code = ComparisonBoundaryError::kAdmission, .detail = {}};
}

bool Add(std::uint64_t& total, std::uint64_t amount) {
  return !__builtin_add_overflow(total, amount, &total);
}

bool Product(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
  return !__builtin_mul_overflow(a, b, &out);
}

constexpr std::uint64_t StateValidationDiagnosticBytes() {
  constexpr std::uint64_t kStateAccessText =
      (std::max)({sizeof("state access declaration does not name a mapped graph root") - 1,
                  sizeof("state access declaration differs from its graph leaf binding") - 1,
                  sizeof("state read/write flags differ from graph access facts") - 1,
                  sizeof("state access declarations omit an original graph state leaf") - 1,
                  sizeof("a store into a leaf that is no state") - 1});
  constexpr std::uint64_t kBorrowedStateText = sizeof("a store into stage 's borrowed  (layer )") -
                                               1 + std::numeric_limits<std::uint16_t>::digits10 +
                                               1 + std::numeric_limits<std::uint16_t>::digits10 +
                                               1 + std::numeric_limits<std::uint32_t>::digits10 + 1;
  constexpr std::uint64_t kCrossStageText =
      sizeof("stage 's step stores into stage 's state, not a tap it writes") - 1 +
      2 * (std::numeric_limits<std::uint16_t>::digits10 + 1);
  constexpr std::uint64_t kTextBytes =
      (std::max)({kStateAccessText, kBorrowedStateText, kCrossStageText});
  return kTextBytes + 1 + 2 * sizeof(void*) + alignof(std::max_align_t);
}

bool Valid(const ir::Graph& graph, ir::TensorId id) {
  return id.valid() && id.v < graph.tensor_count();
}

std::optional<std::size_t> StateStoreTarget(const ir::Node& node) {
  if (node.key == ir::Primitive(ir::Op::kCacheStore)) return 0;
  if (node.key == ir::Fused(ir::FusedKind::kRopeStore)) return 2;
  return std::nullopt;
}

auto WeightKey(const ir::WeightRef& ref) {
  return std::tuple(ref.stage, ref.component, ref.role, ref.layer, ref.expert_array);
}

auto StateKey(const ir::StateRef& ref) {
  return std::tuple(ref.stage, ref.tensor, ref.layer, ref.segment);
}

void HashBinding(Hash& hash, const ir::Binding& binding) {
  hash.U8(static_cast<std::uint8_t>(binding.index()));
  if (const auto* weight = std::get_if<ir::WeightRef>(&binding)) {
    hash.U16(weight->stage);
    hash.U16(weight->component);
    hash.U16(weight->role);
    hash.U32(weight->layer);
    hash.U8(weight->expert_array ? 1 : 0);
  } else if (const auto* state = std::get_if<ir::StateRef>(&binding)) {
    hash.U16(state->stage);
    hash.U16(state->tensor);
    hash.U32(state->layer);
    hash.U8(state->segment);
  }
}

Digest ConstantContent(const ir::Graph& graph, ir::TensorId root) {
  const auto& node = graph.node(graph.producer(root));
  base::Sha256 hash;
  hash.Update(std::span(node.params).first(ir::kConstantDigestBytes));
  return hash.Finish();
}

void HashFormat(Hash& hash, const format::TensorFormat& value) {
  hash.U8(static_cast<std::uint8_t>(value.encoding));
  hash.U8(static_cast<std::uint8_t>(value.layout));
  hash.U8(value.stacked);
  hash.U8(value.shape.rank);
  for (const auto dimension : value.shape.ne) hash.U64(dimension);
  hash.U8(value.params.k_bits);
  hash.U8(static_cast<std::uint8_t>(value.params.codebook));
  hash.U8(value.params.tensor_scale ? 1 : 0);
  hash.U8(value.params.bits);
  hash.U32(value.params.group);
  hash.U8(static_cast<std::uint8_t>(value.params.scale));
  hash.U8(value.params.zero_point ? 1 : 0);
  hash.U32(value.params.block_rows);
  hash.U32(value.params.block_cols);
}

void HashRoot(Hash& hash, const ComparisonBoundaryRoot& root) {
  hash.U32(root.canonical);
  hash.U8(static_cast<std::uint8_t>(root.role));
  HashFormat(hash, root.format);
  for (const auto stride : root.strides) hash.U64(stride);
  hash.U64(root.storage_bytes);
  hash.U8(root.segment);
  HashBinding(hash, root.binding);
  if (root.role == ir::Role::kConstant) hash.DigestValue(root.constant_content);
}

struct CanonicalLabel {
  std::uint32_t original = 0;
  std::uint32_t canonical = 0;
};

struct CanonicalSizeLabel {
  std::size_t original = 0;
  std::uint64_t canonical = 0;
};

class CanonicalLabels {
 public:
  explicit CanonicalLabels(std::uint16_t active_stage) : active_stage_(active_stage) {}

  void Reserve(std::size_t stages, std::size_t components, std::size_t roles, std::size_t layers,
               std::size_t state_tensors, std::size_t segments, std::size_t state_items) {
    stages_.reserve(stages);
    components_.reserve(components);
    roles_.reserve(roles);
    layers_.reserve(layers);
    state_tensors_.reserve(state_tensors);
    segments_.reserve(segments);
    state_items_.reserve(state_items);
    Label(stages_, active_stage_);
  }

  std::uint32_t Stage(std::uint32_t value) { return Label(stages_, value); }
  std::uint32_t Component(std::uint32_t value) {
    return value == kFrameworkComponent ? kFrameworkComponent : Label(components_, value);
  }
  void SeedComponent(std::uint32_t value, std::uint32_t canonical) {
    if (value == kFrameworkComponent) return;
    const auto found = std::ranges::find(components_, value, &CanonicalLabel::original);
    if (found == components_.end()) components_.push_back({value, canonical});
  }
  std::uint32_t Role(std::uint32_t value) { return Label(roles_, value); }
  std::uint32_t Layer(std::uint32_t value) { return Label(layers_, value); }
  std::uint32_t StateTensor(std::uint32_t value) { return Label(state_tensors_, value); }
  std::uint32_t Segment(std::uint32_t value) {
    return value == ir::kJoinedSegment ? ir::kJoinedSegment : Label(segments_, value);
  }
  std::uint64_t StateItem(std::size_t value) {
    const auto found = std::ranges::find(state_items_, value, &CanonicalSizeLabel::original);
    if (found != state_items_.end()) return found->canonical;
    const auto canonical = static_cast<std::uint64_t>(state_items_.size());
    state_items_.push_back({value, canonical});
    return canonical;
  }

 private:
  static std::uint32_t Label(std::vector<CanonicalLabel>& labels, std::uint32_t value) {
    const auto found = std::ranges::find(labels, value, &CanonicalLabel::original);
    if (found != labels.end()) return found->canonical;
    const auto canonical = static_cast<std::uint32_t>(labels.size());
    labels.push_back({value, canonical});
    return canonical;
  }

  std::vector<CanonicalLabel> stages_;
  std::vector<CanonicalLabel> components_;
  std::vector<CanonicalLabel> roles_;
  std::vector<CanonicalLabel> layers_;
  std::vector<CanonicalLabel> state_tensors_;
  std::vector<CanonicalLabel> segments_;
  std::vector<CanonicalSizeLabel> state_items_;
  std::uint16_t active_stage_ = 0;
};

void HashCanonicalBinding(Hash& hash, const ir::Binding& binding, CanonicalLabels& labels) {
  hash.U8(static_cast<std::uint8_t>(binding.index()));
  if (const auto* weight = std::get_if<ir::WeightRef>(&binding)) {
    hash.U32(labels.Stage(weight->stage));
    hash.U32(labels.Component(weight->component));
    hash.U32(labels.Role(weight->role));
    hash.U32(labels.Layer(weight->layer));
    hash.U8(weight->expert_array ? 1 : 0);
  } else if (const auto* state = std::get_if<ir::StateRef>(&binding)) {
    hash.U32(labels.Stage(state->stage));
    hash.U32(labels.StateTensor(state->tensor));
    hash.U32(labels.Layer(state->layer));
    hash.U32(labels.Segment(state->segment));
  }
}

bool HashStorageParts(Hash& hash, const format::TensorFormat& value) {
  const auto parts = format::PartsOf(value);
  if (!parts) return false;
  hash.U8(static_cast<std::uint8_t>(parts->size()));
  std::array<format::PartExtent, format::PartList::kMax> ordered{};
  std::ranges::copy(parts->parts(), ordered.begin());
  std::ranges::sort(std::span(ordered).first(parts->size()), {}, &format::PartExtent::role);
  for (const auto& part : std::span(ordered).first(parts->size())) {
    hash.U8(static_cast<std::uint8_t>(part.role));
    hash.U64(part.bytes);
    hash.U64(part.readable);
  }
  return true;
}

bool HashCanonicalRoot(Hash& hash, const ComparisonRegionRoot& root, CanonicalLabels& labels) {
  hash.U32(root.canonical);
  hash.U8(static_cast<std::uint8_t>(CanonicalComparisonRole(root.role)));
  HashFormat(hash, root.format);
  for (const auto stride : root.strides) hash.U64(stride);
  hash.U64(root.storage_bytes);
  if (!HashStorageParts(hash, root.format)) return false;
  hash.U32(labels.Segment(root.segment));
  HashCanonicalBinding(hash, root.binding, labels);
  if (root.role == ir::Role::kConstant) hash.DigestValue(root.constant_content);
  return true;
}

void HashCanonicalState(Hash& hash, const ComparisonRegionState& state, CanonicalLabels& labels) {
  const auto& facts = state.facts;
  hash.U32(state.canonical_root);
  hash.U32(labels.Stage(facts.ref.stage));
  hash.U32(labels.StateTensor(facts.ref.tensor));
  hash.U32(labels.Layer(facts.ref.layer));
  hash.U32(labels.Segment(facts.ref.segment));
  hash.U64(labels.StateItem(facts.item));
  hash.U8(facts.borrowed ? 1 : 0);
  hash.U8(facts.tap ? 1 : 0);
  hash.U32(labels.Stage(facts.source));
  hash.U8(facts.read ? 1 : 0);
  hash.U8(facts.written ? 1 : 0);
}

struct BuildFacts {
  std::vector<std::uint32_t> canonical_by_tensor;
  std::vector<ir::TensorId> canonical_roots;
  std::vector<const StepWeight*> ordered_weights;
  std::vector<const StepState*> ordered_states;
  std::vector<const ComparisonExternalBinding*> ordered_external;
};

std::expected<void, ComparisonBoundaryError> ValidateView(
    const ComparisonBoundaryView& view, ir::DiagnosticAdmission* diagnostic = nullptr) {
  const auto& graph = view.io.graph;
  if (!std::ranges::equal(view.io.inputs, graph.inputs()) ||
      !std::ranges::equal(view.io.outputs, graph.outputs()))
    return std::unexpected(
        Invalid("typed IO ports differ from the original graph interface", diagnostic));
  const auto check_port =
      [&](std::span<const ir::TensorId> ports,
          std::string_view name) -> std::expected<void, ComparisonBoundaryError> {
    (void)name;
    for (const auto id : ports) {
      if (!Valid(graph, id))
        return std::unexpected(Invalid("interface port has invalid tensor ID", diagnostic));
      if (!ir::StorageBytes(graph.tensor(id)))
        return std::unexpected(Invalid("interface port has invalid storage extent", diagnostic));
    }
    return {};
  };
  if (auto checked = check_port(view.io.inputs, "input port"); !checked) return checked;
  if (auto checked = check_port(view.io.outputs, "output port"); !checked) return checked;
  if (auto checked = check_port(graph.keep(), "keep port"); !checked) return checked;

  for (std::size_t weight_index = 0; weight_index < view.io.weights.size(); ++weight_index) {
    const auto& weight = view.io.weights[weight_index];
    if (!Valid(graph, weight.tensor))
      return std::unexpected(Invalid("weight has invalid tensor ID", diagnostic));
    const auto root = graph.root(weight.tensor);
    const auto* ref = std::get_if<ir::WeightRef>(&graph.tensor(root).binding);
    if (!ref || *ref != weight.ref)
      return std::unexpected(
          Invalid("weight declaration differs from its original graph binding", diagnostic));
    for (std::size_t prior = 0; prior < weight_index; ++prior)
      if (view.io.weights[prior].ref == weight.ref)
        return std::unexpected(Invalid("duplicate complete weight identity", diagnostic));
    const auto parts = format::PartsOf(graph.tensor(root).format);
    if (!parts) return std::unexpected(Invalid("weight format is invalid", diagnostic));
    if (weight.parts.empty() || weight.parts.size() != parts->size())
      return std::unexpected(
          Invalid("weight part declarations differ from format inventory", diagnostic));
    for (std::size_t part_index = 0; part_index < weight.parts.size(); ++part_index) {
      const auto& [name, role] = weight.parts[part_index];
      for (std::size_t prior = 0; prior < part_index; ++prior)
        if (weight.parts[prior].first == name || weight.parts[prior].second == role)
          return std::unexpected(Invalid("weight repeats part name or semantic role", diagnostic));
    }
    if (!weight.parts.empty()) {
      for (const auto& part : parts->parts())
        if (std::ranges::none_of(weight.parts,
                                 [&](const auto& item) { return item.second == part.role; }))
          return std::unexpected(
              Invalid("weight part declaration omits a format part", diagnostic));
    }
  }
  for (std::uint32_t i = 0; i < graph.tensor_count(); ++i) {
    const ir::TensorId id{i};
    if (graph.root(id) != id) continue;
    const auto& binding = graph.tensor(id).binding;
    if (const auto* weight = std::get_if<ir::WeightRef>(&binding)) {
      if (std::ranges::count(view.io.weights, *weight, &StepWeight::ref) != 1)
        return std::unexpected(
            Invalid("graph contains an undeclared or duplicate weight root", diagnostic));
    }
  }

  for (std::size_t state_index = 0; state_index < view.state.size(); ++state_index) {
    const auto& state = view.state[state_index];
    if (!Valid(graph, state.tensor) || graph.root(state.tensor) != state.tensor)
      return std::unexpected(Invalid("state declaration has invalid original root", diagnostic));
    const auto* ref = std::get_if<ir::StateRef>(&graph.tensor(state.tensor).binding);
    bool duplicate = false;
    for (std::size_t prior = 0; prior < state_index; ++prior)
      duplicate |= view.state[prior].ref == state.ref;
    if (!ref || *ref != state.ref || duplicate)
      return std::unexpected(
          Invalid("state declaration differs from graph or repeats identity", diagnostic));
  }
  for (std::uint32_t i = 0; i < graph.tensor_count(); ++i) {
    const ir::TensorId id{i};
    if (graph.root(id) != id) continue;
    const auto* ref = std::get_if<ir::StateRef>(&graph.tensor(id).binding);
    if (ref && std::ranges::count(view.state, *ref, &StepState::ref) != 1)
      return std::unexpected(
          Invalid("graph contains an undeclared or duplicate state root", diagnostic));
  }
  if (auto checked = CheckStateAccess(graph, view.state); !checked)
    return std::unexpected(
        Invalid(std::string("state declaration failed its access/write proof: ") + checked.error(),
                diagnostic));
  if (auto checked = CheckStateWrites(graph, view.state, {}, view.active_stage); !checked)
    return std::unexpected(
        Invalid(std::string("state declaration failed its access/write proof: ") + checked.error(),
                diagnostic));

  const auto external_count = [&](ir::TensorId root, std::optional<format::PartRole> part) {
    return std::ranges::count_if(view.external, [&](const ComparisonExternalBinding& candidate) {
      return candidate.original_root == root && candidate.part == part;
    });
  };
  const auto unstored_count = [&](ir::TensorId input) {
    return std::ranges::count(view.unstored_inputs, input,
                              &ComparisonUnstoredInput::original_input);
  };
  for (std::size_t i = 0; i < view.unstored_inputs.size(); ++i) {
    const auto& declaration = view.unstored_inputs[i];
    if (!Valid(graph, declaration.original_input) ||
        graph.root(declaration.original_input) != declaration.original_input ||
        !std::ranges::contains(view.io.inputs, declaration.original_input) ||
        std::ranges::any_of(view.io.outputs,
                            [&](ir::TensorId output) {
                              return graph.root(output) == declaration.original_input;
                            }) ||
        std::ranges::any_of(graph.keep(), [&](ir::TensorId kept) {
          return graph.root(kept) == declaration.original_input;
        }))
      return std::unexpected(
          Invalid("unstored declaration is not an exact nonobservable original input", diagnostic));
    if (declaration.reason != ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks)
      return std::unexpected(Invalid("unstored input reason is unsupported", diagnostic));
    if (unstored_count(declaration.original_input) != 1 ||
        external_count(declaration.original_input, std::nullopt) != 0)
      return std::unexpected(
          Invalid("unstored declaration is duplicate or also externally backed", diagnostic));
  }
  for (const auto input : view.io.inputs) {
    const auto root = view.io.graph.root(input);
    const bool backed = external_count(root, std::nullopt) == 1;
    const bool unstored = unstored_count(input) == 1;
    if (backed == unstored || external_count(root, std::nullopt) > 1 || unstored_count(input) > 1)
      return std::unexpected(Invalid(
          "each original input needs exactly one external or typed unstored fact", diagnostic));
  }
  for (const auto& weight : view.io.weights) {
    const auto root = graph.root(weight.tensor);
    const auto parts = format::PartsOf(graph.tensor(root).format);
    if (!parts) return std::unexpected(Invalid("weight format is invalid", diagnostic));
    if (weight.parts.empty()) {
      if (external_count(root, parts->parts().front().role) != 1)
        return std::unexpected(
            Invalid("external facts omit or duplicate a weight part", diagnostic));
    } else {
      for (const auto& [name, part] : weight.parts) {
        (void)name;
        if (external_count(root, part) != 1)
          return std::unexpected(
              Invalid("external facts omit or duplicate a weight part", diagnostic));
      }
    }
  }
  for (const auto& state : view.state)
    if (external_count(state.tensor, std::nullopt) != 1)
      return std::unexpected(Invalid("external facts omit or duplicate a state root", diagnostic));

  for (std::size_t external_index = 0; external_index < view.external.size(); ++external_index) {
    const auto& external = view.external[external_index];
    if (!Valid(graph, external.original_root) ||
        graph.root(external.original_root) != external.original_root)
      return std::unexpected(
          Invalid("external binding names an invalid original root", diagnostic));
    if (external.allocation_bytes == 0 || external.bytes == 0 || external.alignment == 0 ||
        !std::has_single_bit(external.alignment) || external.offset % external.alignment != 0)
      return std::unexpected(Invalid("external binding has empty or misaligned range", diagnostic));
    std::uint64_t end = 0;
    if (__builtin_add_overflow(external.offset, external.bytes, &end) ||
        end > external.allocation_bytes)
      return std::unexpected(Invalid("external binding range exceeds allocation", diagnostic));
    const auto parts = format::PartsOf(graph.tensor(external.original_root).format);
    if (!parts)
      return std::unexpected(Invalid("external binding root format is invalid", diagnostic));
    std::uint64_t required = 0;
    if (const auto* weight =
            std::get_if<ir::WeightRef>(&graph.tensor(external.original_root).binding)) {
      if (!external.part)
        return std::unexpected(Invalid("external weight part is missing", diagnostic));
      const auto part = parts->Find(*external.part);
      if (!part)
        return std::unexpected(Invalid("external part is absent from weight format", diagnostic));
      required = part->readable;
      if (external.row_storage) {
        if (!ValidComparisonRowStorage(*external.row_storage, *weight, *external.part,
                                       graph.tensor(external.original_root).format,
                                       external.bytes) ||
            external.resource_part_identity != external.row_storage->content_identity)
          return std::unexpected(
              Invalid("row-table facts do not match the source or slot range", diagnostic));
        required = external.bytes;
      }
    } else {
      if (external.part || external.row_storage)
        return std::unexpected(Invalid("non-weight external binding has a part", diagnostic));
      const auto extent = ir::StorageBytes(graph.tensor(external.original_root));
      if (!extent)
        return std::unexpected(Invalid("external binding has invalid storage extent", diagnostic));
      required = *extent;
      if (const auto data = parts->Find(format::PartRole::kData))
        required = std::max(required, data->readable);
    }
    if (external.bytes < required)
      return std::unexpected(
          Invalid("external binding omits format-readable storage extent", diagnostic));
    for (std::size_t prior = 0; prior < external_index; ++prior)
      if (view.external[prior].original_root == external.original_root &&
          view.external[prior].part == external.part)
        return std::unexpected(Invalid("duplicate external root/part binding", diagnostic));
      else if (view.external[prior].allocation_class == external.allocation_class &&
               view.external[prior].allocation_bytes != external.allocation_bytes)
        return std::unexpected(
            Invalid("external alias class has inconsistent allocation extent", diagnostic));
    bool declared = std::ranges::any_of(view.io.inputs, [&](ir::TensorId id) {
      return graph.root(id) == external.original_root && !external.part;
    });
    for (const auto& weight : view.io.weights) {
      if (graph.root(weight.tensor) != external.original_root) continue;
      const auto weight_parts = format::PartsOf(graph.tensor(external.original_root).format);
      if (weight_parts && weight.parts.empty() && weight_parts->size() == 1 &&
          external.part == weight_parts->parts().front().role)
        declared = true;
      if (std::ranges::any_of(weight.parts, [&](const auto& item) {
            return external.part && item.second == *external.part;
          }))
        declared = true;
    }
    for (const auto& state : view.state)
      if (state.tensor == external.original_root && !external.part) declared = true;
    if (!declared)
      return std::unexpected(
          Invalid("external binding is not part of the original interface", diagnostic));
  }

  return {};
}

std::expected<BuildFacts, ComparisonBoundaryError> MakeBuildFacts(
    const ComparisonBoundaryView& view, ir::DiagnosticAdmission* diagnostic = nullptr) {
  if (auto checked = ValidateView(view, diagnostic); !checked)
    return std::unexpected(std::move(checked.error()));
  const auto& graph = view.io.graph;
  BuildFacts facts;
  facts.canonical_by_tensor.resize(graph.tensor_count(), UINT32_MAX);
  facts.canonical_roots.reserve(graph.tensor_count());
  const auto add_root = [&](ir::TensorId id) {
    const auto root = graph.root(id);
    if (facts.canonical_by_tensor[root.v] == UINT32_MAX) {
      facts.canonical_by_tensor[root.v] = static_cast<std::uint32_t>(facts.canonical_roots.size());
      facts.canonical_roots.push_back(root);
    }
  };
  for (const auto id : view.io.inputs) add_root(id);
  for (const auto id : view.io.outputs) add_root(id);
  for (const auto id : graph.keep()) add_root(id);
  facts.ordered_weights.reserve(view.io.weights.size());
  for (const auto& weight : view.io.weights) facts.ordered_weights.push_back(&weight);
  std::ranges::sort(facts.ordered_weights, [](const StepWeight* a, const StepWeight* b) {
    return WeightKey(a->ref) < WeightKey(b->ref);
  });
  for (const auto* weight : facts.ordered_weights) add_root(weight->tensor);
  facts.ordered_states.reserve(view.state.size());
  for (const auto& state : view.state) facts.ordered_states.push_back(&state);
  std::ranges::sort(facts.ordered_states, [](const StepState* a, const StepState* b) {
    return StateKey(a->ref) < StateKey(b->ref);
  });
  for (const auto* state : facts.ordered_states) add_root(state->tensor);

  facts.ordered_external.reserve(view.external.size());
  for (const auto& binding : view.external) facts.ordered_external.push_back(&binding);
  const auto external_key = [&](const ComparisonExternalBinding* binding) {
    const auto canonical = facts.canonical_by_tensor[graph.root(binding->original_root).v];
    const bool has_part = binding->part.has_value();
    const auto part = has_part ? static_cast<std::uint8_t>(*binding->part) : 0;
    return std::tuple(canonical, has_part, part);
  };
  std::ranges::sort(facts.ordered_external, [&](const auto* a, const auto* b) {
    return external_key(a) < external_key(b);
  });
  return facts;
}

std::expected<std::uint64_t, ComparisonBoundaryError> RetainedBound(
    const ComparisonBoundaryView& view, std::uint64_t roots_count, std::uint64_t weight_part_count,
    ir::DiagnosticAdmission* diagnostic = nullptr) {
  std::uint64_t total = sizeof(OwnedComparisonBoundary);
  std::uint64_t amount = 0;
  constexpr std::uint64_t kAllocatorEnvelope = 64;
  constexpr std::uint64_t kOwnedVectorCount = 7;
  if (!Product(kOwnedVectorCount, kAllocatorEnvelope, amount) || !Add(total, amount))
    return std::unexpected(
        Invalid("comparison boundary vector allocation bound overflows", diagnostic));
  std::uint64_t port_count = view.io.inputs.size();
  if (!Add(port_count, view.io.outputs.size()) || !Add(port_count, view.io.graph.keep().size()) ||
      !Product(roots_count, sizeof(ComparisonBoundaryRoot), amount) || !Add(total, amount) ||
      !Product(port_count, sizeof(ComparisonBoundaryPort), amount) || !Add(total, amount) ||
      !Product(weight_part_count, sizeof(ComparisonBoundaryWeightPart), amount) ||
      !Add(total, amount) || !Product(view.state.size(), sizeof(ComparisonBoundaryState), amount) ||
      !Add(total, amount) || !Product(view.state.size(), sizeof(StepState), amount) ||
      !Add(total, amount) ||
      !Product(view.external.size(), sizeof(ComparisonBoundaryExternal), amount) ||
      !Add(total, amount) ||
      !Product(view.unstored_inputs.size(), sizeof(ComparisonBoundaryUnstoredInput), amount) ||
      !Add(total, amount))
    return std::unexpected(
        Invalid("comparison boundary retained storage bound overflows", diagnostic));
  for (const auto& weight : view.io.weights)
    for (const auto& [name, role] : weight.parts) {
      (void)role;
      std::uint64_t string_bound = name.size();
      if (!Add(string_bound, 1) || !Add(string_bound, kAllocatorEnvelope) ||
          !Add(total, string_bound))
        return std::unexpected(Invalid("comparison boundary string bound overflows", diagnostic));
    }
  return total;
}

std::expected<std::uint64_t, ComparisonBoundaryError> ScratchBound(
    const ComparisonBoundaryView& view, ir::DiagnosticAdmission* diagnostic = nullptr) {
  std::uint64_t total = 0, amount = 0;
  constexpr std::uint64_t kAllocatorEnvelope = 64;
  constexpr std::uint64_t kScratchVectorCount = 6;
  const std::uint64_t graph_count = view.io.graph.tensor_count();
  const std::uint64_t node_count = view.io.graph.nodes().size();
  std::uint64_t part_count = 0, string_bytes = 0, part_copy_bytes = 0;
  for (const auto& weight : view.io.weights) {
    const std::uint64_t weight_parts = weight.parts.empty() ? 1 : weight.parts.size();
    if (!Add(part_count, weight_parts))
      return std::unexpected(
          Invalid("comparison boundary scratch part count overflows", diagnostic));
    std::uint64_t weight_string_bytes = 0;
    for (const auto& [name, role] : weight.parts) {
      (void)role;
      std::uint64_t copied_name_bytes = name.size();
      if (!Add(copied_name_bytes, 1) || !Add(copied_name_bytes, kAllocatorEnvelope) ||
          !Add(weight_string_bytes, copied_name_bytes))
        return std::unexpected(
            Invalid("comparison boundary scratch string count overflows", diagnostic));
    }
    string_bytes = std::max(string_bytes, weight_string_bytes);
    if (!weight.parts.empty() &&
        !Product(weight.parts.size(), sizeof(std::pair<std::string, format::PartRole>), amount))
      return std::unexpected(
          Invalid("comparison boundary sorted-part scratch overflows", diagnostic));
    part_copy_bytes = std::max(part_copy_bytes, amount);
  }
  std::uint64_t port_count = view.io.inputs.size();
  if (!Add(port_count, view.io.outputs.size()) || !Add(port_count, view.io.graph.keep().size()) ||
      !Add(port_count, view.io.weights.size()) || !Add(port_count, view.state.size()) ||
      !Add(port_count, view.external.size()))
    return std::unexpected(Invalid("comparison boundary scratch count overflows", diagnostic));
  std::uint64_t scratch_records = port_count;
  if (!Add(scratch_records, part_count) ||
      !Product(graph_count, 3 * sizeof(std::uint32_t), total) ||
      !Product(node_count, sizeof(std::uint32_t), amount) || !Add(total, amount) ||
      !Product(scratch_records, sizeof(std::uint32_t) + sizeof(void*), amount) ||
      !Add(total, amount) ||
      !Product(view.external.size(), sizeof(std::pair<std::uint32_t, std::uint32_t>), amount) ||
      !Add(total, amount) || !Add(total, part_copy_bytes) || !Add(total, string_bytes) ||
      !Product(kScratchVectorCount, kAllocatorEnvelope, amount) || !Add(total, amount))
    return std::unexpected(Invalid("comparison boundary scratch bound overflows", diagnostic));
  return total;
}

void HashPort(Hash& hash, const ComparisonBoundaryPort& port) {
  hash.U8(static_cast<std::uint8_t>(port.kind));
  hash.U32(port.ordinal);
  hash.U32(port.canonical_root);
  hash.U8(static_cast<std::uint8_t>(port.role));
  HashFormat(hash, port.format);
  for (const auto stride : port.strides) hash.U64(stride);
  hash.U64(port.root_offset);
  hash.U64(port.storage_bytes);
  hash.U8(port.segment);
  HashBinding(hash, port.binding);
}

void HashState(Hash& hash, const ComparisonBoundaryState& state) {
  const auto& facts = state.facts;
  hash.U32(state.canonical_root);
  hash.U16(facts.ref.stage);
  hash.U16(facts.ref.tensor);
  hash.U32(facts.ref.layer);
  hash.U8(facts.ref.segment);
  hash.U64(facts.item);
  hash.U8(facts.borrowed ? 1 : 0);
  hash.U8(facts.tap ? 1 : 0);
  hash.U16(facts.source);
  hash.U8(facts.read ? 1 : 0);
  hash.U8(facts.written ? 1 : 0);
}

void HashRowStorage(Hash& hash, const ComparisonRowStorage& storage) {
  hash.U8(0xD3);
  hash.U16(storage.ref.stage);
  hash.U16(storage.ref.component);
  hash.U16(storage.ref.role);
  hash.U32(storage.ref.layer);
  hash.U8(storage.ref.expert_array ? 1 : 0);
  hash.U8(static_cast<std::uint8_t>(storage.part));
  HashFormat(hash, storage.logical_format);
  hash.DigestValue(storage.content_identity);
  hash.U32(storage.resource);
  hash.U32(storage.group);
  hash.U32(storage.shard);
  hash.U64(storage.group_bytes);
  hash.U64(storage.resource_offset);
  hash.U64(storage.resource_bytes);
  hash.U64(storage.logical_rows);
  hash.U64(storage.slot_capacity);
  hash.U64(storage.row_bytes);
}

ComparisonBoundaryDigest MakeDigest(const OwnedComparisonBoundary& boundary,
                                    std::uint16_t active_stage) {
  Hash logical("llmp.comparison.boundary.logical.v2");
  logical.U16(active_stage);
  logical.U64(boundary.roots().size());
  for (const auto& root : boundary.roots()) HashRoot(logical, root);
  logical.U64(boundary.ports().size());
  for (const auto& port : boundary.ports()) HashPort(logical, port);
  logical.U64(boundary.unstored_inputs().size());
  for (const auto& input : boundary.unstored_inputs()) {
    logical.U32(input.canonical_root);
    logical.U8(static_cast<std::uint8_t>(input.reason));
  }
  logical.U64(boundary.weight_parts().size());
  for (const auto& part : boundary.weight_parts()) {
    logical.U32(part.canonical_root);
    logical.U16(part.ref.stage);
    logical.U16(part.ref.component);
    logical.U16(part.ref.role);
    logical.U32(part.ref.layer);
    logical.U8(part.ref.expert_array ? 1 : 0);
    logical.Text(part.name);
    logical.U8(static_cast<std::uint8_t>(part.part));
  }
  logical.U64(boundary.state().size());
  for (const auto& state : boundary.state()) HashState(logical, state);

  Hash external("llmp.comparison.boundary.external.v2");
  external.U64(boundary.external().size());
  for (const auto& binding : boundary.external()) {
    external.U32(binding.canonical_root);
    external.U8(binding.part.has_value() ? 1 : 0);
    if (binding.part) external.U8(static_cast<std::uint8_t>(*binding.part));
    external.U32(binding.canonical_allocation_class);
    external.U64(binding.allocation_bytes);
    external.U64(binding.offset);
    external.U64(binding.bytes);
    external.U64(binding.alignment);
    external.U8(binding.resource_part_identity.has_value() ? 1 : 0);
    if (binding.resource_part_identity) external.DigestValue(*binding.resource_part_identity);
    external.U8(binding.row_storage.has_value() ? 1 : 0);
    if (binding.row_storage) HashRowStorage(external, *binding.row_storage);
  }

  ComparisonBoundaryDigest result;
  result.logical = logical.Finish();
  result.external = external.Finish();
  Hash combined("llmp.comparison.boundary.combined.v2");
  combined.U32(result.version);
  combined.DigestValue(result.logical);
  combined.DigestValue(result.external);
  result.combined = combined.Finish();
  return result;
}

}  // namespace

bool ValidComparisonRowStorage(const ComparisonRowStorage& storage, const ir::WeightRef& ref,
                               format::PartRole part, const format::TensorFormat& logical_format,
                               std::uint64_t physical_bytes) noexcept {
  const auto row_bytes = format::RowBytes(logical_format, part);
  std::uint64_t source_end = 0, expected_source_bytes = 0, expected_physical_bytes = 0;
  const bool digest_present =
      std::ranges::any_of(storage.content_identity, [](std::uint8_t byte) { return byte != 0; });
  return digest_present && storage.ref == ref && storage.part == part &&
         storage.logical_format == logical_format && logical_format.shape.rank == 2 && row_bytes &&
         storage.row_bytes == *row_bytes && storage.logical_rows == logical_format.shape.ne[1] &&
         storage.logical_rows != 0 && storage.slot_capacity != 0 &&
         storage.slot_capacity <= storage.logical_rows &&
         storage.slot_capacity <=
             static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) &&
         storage.group_bytes != 0 && storage.resource_bytes != 0 &&
         !__builtin_add_overflow(storage.resource_offset, storage.resource_bytes, &source_end) &&
         source_end <= storage.group_bytes &&
         !__builtin_mul_overflow(storage.logical_rows, storage.row_bytes, &expected_source_bytes) &&
         expected_source_bytes == storage.resource_bytes &&
         !__builtin_mul_overflow(storage.slot_capacity, storage.row_bytes,
                                 &expected_physical_bytes) &&
         expected_physical_bytes == physical_bytes;
}

std::expected<std::uint64_t, ComparisonBoundaryError> ComparisonBoundaryStorageBound(
    const ComparisonBoundaryView& view, ir::DiagnosticAdmission* diagnostic) {
  if (auto checked = ValidateView(view, diagnostic); !checked)
    return std::unexpected(std::move(checked.error()));
  const auto& graph = view.io.graph;
  std::uint64_t roots_count = view.io.inputs.size();
  if (!Add(roots_count, view.io.outputs.size()) || !Add(roots_count, graph.keep().size()) ||
      !Add(roots_count, view.io.weights.size()) || !Add(roots_count, view.state.size()))
    return std::unexpected(Invalid("comparison boundary root count overflows", diagnostic));
  std::uint64_t part_count = 0;
  for (const auto& weight : view.io.weights) {
    const auto parts = format::PartsOf(graph.tensor(graph.root(weight.tensor)).format);
    if (!parts || !Add(part_count, parts->size()))
      return std::unexpected(
          Invalid("comparison boundary weight-part count overflows", diagnostic));
  }
  return RetainedBound(view, roots_count, part_count, diagnostic);
}

std::expected<OwnedComparisonBoundary, ComparisonBoundaryError> MakeComparisonBoundary(
    const ComparisonBoundaryView& view, std::shared_ptr<const ir::StorageFunder> funder,
    ir::DiagnosticAdmission* diagnostic) {
  auto bound = ComparisonBoundaryStorageBound(view, diagnostic);
  if (!bound) return std::unexpected(std::move(bound.error()));
  auto retained_loan = ir::AcquireStorageLoan(funder, *bound);
  if (!retained_loan)
    return std::unexpected(Admission("comparison boundary storage admission refused"));
  auto scratch_bytes = ScratchBound(view, diagnostic);
  if (!scratch_bytes) return std::unexpected(std::move(scratch_bytes.error()));
  auto scratch_loan = ir::AcquireStorageLoan(funder, *scratch_bytes);
  if (!scratch_loan)
    return std::unexpected(Admission("comparison boundary scratch admission refused"));

  auto facts_result = MakeBuildFacts(view, diagnostic);
  if (!facts_result) return std::unexpected(std::move(facts_result.error()));
  auto facts = std::move(*facts_result);
  OwnedComparisonBoundary result;
  result.loan_ = std::move(*retained_loan);
  result.source_identity_ = view.io.graph.identity();
  result.source_digest_ = view.io.graph.Digest();
  result.source_tensor_count_ = static_cast<std::uint32_t>(view.io.graph.tensor_count());
  result.source_input_count_ = static_cast<std::uint32_t>(view.io.inputs.size());
  result.source_output_count_ = static_cast<std::uint32_t>(view.io.outputs.size());
  result.source_keep_count_ = static_cast<std::uint32_t>(view.io.graph.keep().size());
  result.active_stage_ = view.active_stage;
  result.roots_.reserve(facts.canonical_roots.size());
  for (std::uint32_t canonical = 0; canonical < facts.canonical_roots.size(); ++canonical) {
    const auto root_id = facts.canonical_roots[canonical];
    const auto& desc = view.io.graph.tensor(root_id);
    const auto bytes = ir::StorageBytes(desc);
    if (!bytes)
      return std::unexpected(Invalid("graph tensor has invalid storage extent", diagnostic));
    result.roots_.push_back({.canonical = canonical,
                             .original_root = root_id,
                             .role = desc.role,
                             .format = desc.format,
                             .strides = desc.nb,
                             .storage_bytes = *bytes,
                             .constant_content = desc.role == ir::Role::kConstant
                                                     ? ConstantContent(view.io.graph, root_id)
                                                     : Digest{},
                             .binding = desc.binding,
                             .segment = desc.segment});
  }
  const auto append_ports =
      [&](std::span<const ir::TensorId> ids,
          ComparisonPortKind kind) -> std::expected<void, ComparisonBoundaryError> {
    for (std::uint32_t ordinal = 0; ordinal < ids.size(); ++ordinal) {
      const auto id = ids[ordinal];
      const auto& desc = view.io.graph.tensor(id);
      const auto storage = ir::StorageBytes(desc);
      if (!storage)
        return std::unexpected(Invalid("boundary tensor has invalid storage extent", diagnostic));
      result.ports_.push_back(
          {.kind = kind,
           .ordinal = ordinal,
           .canonical_root = facts.canonical_by_tensor[view.io.graph.root(id).v],
           .original_tensor = id,
           .role = desc.role,
           .format = desc.format,
           .strides = desc.nb,
           .root_offset = desc.view_offset,
           .storage_bytes = *storage,
           .binding = desc.binding,
           .segment = desc.segment});
    }
    return {};
  };
  std::uint64_t port_capacity = view.io.inputs.size();
  if (!Add(port_capacity, view.io.outputs.size()) ||
      !Add(port_capacity, view.io.graph.keep().size()))
    return std::unexpected(Invalid("comparison boundary port count overflows", diagnostic));
  result.ports_.reserve(static_cast<std::size_t>(port_capacity));
  if (auto appended = append_ports(view.io.inputs, ComparisonPortKind::kInput); !appended)
    return std::unexpected(std::move(appended.error()));
  if (auto appended = append_ports(view.io.outputs, ComparisonPortKind::kOutput); !appended)
    return std::unexpected(std::move(appended.error()));
  if (auto appended = append_ports(view.io.graph.keep(), ComparisonPortKind::kKeep); !appended)
    return std::unexpected(std::move(appended.error()));

  std::uint64_t part_capacity = 0;
  for (const auto& weight : view.io.weights) {
    const auto parts =
        format::PartsOf(view.io.graph.tensor(view.io.graph.root(weight.tensor)).format);
    if (!parts || !Add(part_capacity, parts->size()))
      return std::unexpected(
          Invalid("comparison boundary weight part count overflows", diagnostic));
  }
  result.weight_parts_.reserve(static_cast<std::size_t>(part_capacity));
  for (const auto* weight : facts.ordered_weights) {
    const auto root = view.io.graph.root(weight->tensor);
    const auto parts = format::PartsOf(view.io.graph.tensor(root).format);
    if (!parts)
      return std::unexpected(Invalid("weight format lost its part inventory", diagnostic));
    if (weight->parts.empty()) {
      result.weight_parts_.push_back({.canonical_root = facts.canonical_by_tensor[root.v],
                                      .original_root = root,
                                      .ref = weight->ref,
                                      .name = {},
                                      .part = parts->parts().front().role});
      continue;
    }
    auto sorted_parts = weight->parts;
    std::ranges::sort(sorted_parts, [](const auto& a, const auto& b) {
      return std::tuple(a.second, a.first) < std::tuple(b.second, b.first);
    });
    for (const auto& [name, part] : sorted_parts)
      result.weight_parts_.push_back({.canonical_root = facts.canonical_by_tensor[root.v],
                                      .original_root = root,
                                      .ref = weight->ref,
                                      .name = name,
                                      .part = part});
  }
  result.state_.reserve(facts.ordered_states.size());
  result.state_facts_.reserve(facts.ordered_states.size());
  for (const auto* state : facts.ordered_states) {
    const auto root = view.io.graph.root(state->tensor);
    result.state_.push_back({.canonical_root = facts.canonical_by_tensor[root.v], .facts = *state});
    result.state_facts_.push_back(*state);
  }
  std::vector<std::pair<std::uint32_t, std::uint32_t>> classes;
  classes.reserve(facts.ordered_external.size());
  result.external_.reserve(facts.ordered_external.size());
  for (const auto* external : facts.ordered_external) {
    const auto root = view.io.graph.root(external->original_root);
    auto at = std::ranges::find(classes, external->allocation_class,
                                &std::pair<std::uint32_t, std::uint32_t>::first);
    if (at == classes.end()) {
      classes.emplace_back(external->allocation_class, static_cast<std::uint32_t>(classes.size()));
      at = std::prev(classes.end());
    }
    result.external_.push_back({.canonical_root = facts.canonical_by_tensor[root.v],
                                .part = external->part,
                                .canonical_allocation_class = at->second,
                                .allocation_bytes = external->allocation_bytes,
                                .offset = external->offset,
                                .bytes = external->bytes,
                                .alignment = external->alignment,
                                .resource_part_identity = external->resource_part_identity,
                                .row_storage = external->row_storage});
  }
  result.unstored_inputs_.reserve(view.unstored_inputs.size());
  for (const auto input : view.io.inputs) {
    const auto found =
        std::ranges::find(view.unstored_inputs, input, &ComparisonUnstoredInput::original_input);
    if (found == view.unstored_inputs.end()) continue;
    const auto root = view.io.graph.root(input);
    result.unstored_inputs_.push_back({.canonical_root = facts.canonical_by_tensor[root.v],
                                       .original_input = input,
                                       .reason = found->reason});
  }
  result.digest_ = MakeDigest(result, view.active_stage);
  return result;
}

std::expected<OwnedComparisonRegionBoundary, ComparisonBoundaryError> MakeComparisonRegionBoundary(
    const ComparisonRegionBoundaryView& view, std::shared_ptr<const ir::StorageFunder> funder,
    ir::DiagnosticAdmission* diagnostic) {
  const auto& graph = view.source.io.graph;
  const auto invalid = [&](std::string_view message) {
    return std::unexpected(Invalid(message, diagnostic));
  };
  const auto admission = [] {
    return std::unexpected(Admission("comparison region admission refused"));
  };
  if (!funder) return admission();
  if (graph.identity() == 0 || view.member_offsets.size() < 2 || view.member_offsets.front() != 0 ||
      view.member_offsets.back() != view.members.size())
    return invalid("comparison region source or member offsets are invalid");
  auto state_validation_loan = ir::AcquireStorageLoan(funder, StateValidationDiagnosticBytes());
  if (!state_validation_loan) return admission();
  if (auto checked = ValidateView(view.source, diagnostic); !checked)
    return std::unexpected(std::move(checked.error()));
  const std::size_t group_count = view.member_offsets.size() - 1;
  for (std::size_t g = 0; g < group_count; ++g)
    if (view.member_offsets[g] >= view.member_offsets[g + 1])
      return invalid("comparison region contains an empty or unordered member group");
  const std::uint64_t member_count = view.members.size();
  std::uint64_t edge_bound = 0;
  if (!Product(member_count, ir::kMaxSources + ir::kMaxOutputs, edge_bound))
    return invalid("comparison region edge inventory overflows");
  std::uint64_t port_bound = 0;
  if (!Product(edge_bound, 2, port_bound))
    return invalid("comparison region port inventory overflows");
  std::uint64_t parts_bound = 0;
  for (const auto& weight : view.source.io.weights) {
    if (weight.tensor.valid() && weight.tensor.v < graph.tensor_count()) {
      const auto parts = format::PartsOf(graph.tensor(graph.root(weight.tensor)).format);
      if (!parts || !Add(parts_bound, parts->size()))
        return invalid("comparison region weight part inventory overflows");
    }
  }
  std::uint64_t retained = 0, scratch = 0;
  const auto capacity = [](std::uint64_t count, std::uint64_t size, std::uint64_t& total) {
    constexpr std::uint64_t env = 2 * sizeof(void*) + alignof(std::max_align_t);
    std::uint64_t bytes = 0;
    return Product(count, size, bytes) && Add(bytes, count == 0 ? 0 : env) && Add(total, bytes);
  };
  if (!capacity(member_count, sizeof(ComparisonRegionNode), retained) ||
      !capacity(edge_bound, sizeof(ComparisonRegionTensor), retained) ||
      !capacity(edge_bound, sizeof(ComparisonRegionRoot), retained) ||
      !capacity(port_bound, sizeof(ComparisonRegionPort), retained) ||
      !capacity(parts_bound, sizeof(ComparisonRegionWeightPart), retained) ||
      !capacity(view.source.state.size(), sizeof(ComparisonRegionState), retained) ||
      !capacity(view.source.external.size(), sizeof(ComparisonRegionExternal), retained) ||
      !capacity(group_count + 1, sizeof(std::uint32_t), scratch) ||
      !capacity(graph.tensor_count(), sizeof(std::uint32_t) * 2, scratch) ||
      !capacity(graph.nodes().size(), sizeof(std::uint8_t), scratch) ||
      !capacity(view.source.external.size(), sizeof(std::pair<std::uint32_t, std::uint32_t>),
                scratch))
    return invalid("comparison region storage inventory overflows");
  std::uint64_t binding_records = 0;
  if (!Add(binding_records, edge_bound) || !Add(binding_records, parts_bound) ||
      !Add(binding_records, view.source.state.size()))
    return invalid("comparison region canonical-label inventory overflows");
  std::uint64_t stage_labels = 1;
  if (!Add(stage_labels, binding_records) || !Add(stage_labels, view.source.state.size()))
    return invalid("comparison region stage-label inventory overflows");
  std::uint64_t component_labels = member_count;
  if (!Add(component_labels, binding_records))
    return invalid("comparison region component-label inventory overflows");
  std::uint64_t state_tensor_labels = binding_records;
  if (!Add(state_tensor_labels, view.source.state.size()))
    return invalid("comparison region state-label inventory overflows");
  std::uint64_t segment_labels = edge_bound;
  if (!Add(segment_labels, view.source.state.size()))
    return invalid("comparison region segment-label inventory overflows");
  const auto label_capacity = [&](std::uint64_t count) {
    return capacity(count, sizeof(CanonicalLabel), scratch);
  };
  if (!label_capacity(stage_labels) || !label_capacity(component_labels) ||
      !label_capacity(binding_records) || !label_capacity(binding_records) ||
      !label_capacity(state_tensor_labels) || !label_capacity(segment_labels) ||
      !capacity(view.source.state.size(), sizeof(CanonicalSizeLabel), scratch))
    return invalid("comparison region canonical-label storage overflows");
  if (!capacity(member_count, sizeof(std::uint32_t), scratch))
    return invalid("comparison region component-label cache overflows");
  // Weight-part labels are copied into owned storage; charge their exact
  // NUL-terminated payload and allocation envelope before construction.
  for (const auto& weight : view.source.io.weights)
    for (const auto& [name, _] : weight.parts) {
      std::uint64_t bytes = 0;
      if (!Product(1, name.size() + 1 + 2 * sizeof(void*) + alignof(std::max_align_t), bytes) ||
          !Add(retained, bytes))
        return invalid("comparison region weight-name inventory overflows");
    }
  auto retained_loan = ir::AcquireStorageLoan(funder, retained);
  if (!retained_loan) return admission();
  auto scratch_loan = ir::AcquireStorageLoan(funder, scratch);
  if (!scratch_loan) return admission();
  std::vector<std::uint32_t> canonical_components;
  canonical_components.reserve(member_count);
  std::vector<std::uint32_t> tensor_ids(graph.tensor_count(), UINT32_MAX);
  std::vector<std::uint32_t> root_ids(graph.tensor_count(), UINT32_MAX);
  std::vector<std::uint8_t> included(graph.nodes().size(), 0);
  for (const auto id : view.members) {
    if (!id.valid() || id.v >= graph.nodes().size() || included[id.v] != 0)
      return invalid("comparison region contains an invalid or repeated source member");
    included[id.v] = 1;
  }

  OwnedComparisonRegionBoundary result;
  result.loan_ = std::move(*retained_loan);
  result.source_identity_ = graph.identity();
  result.active_stage_ = view.source.active_stage;
  result.nodes_.reserve(static_cast<std::size_t>(member_count));
  result.tensors_.reserve(static_cast<std::size_t>(edge_bound));
  result.roots_.reserve(static_cast<std::size_t>(edge_bound));
  result.ports_.reserve(static_cast<std::size_t>(port_bound));
  result.weight_parts_.reserve(static_cast<std::size_t>(parts_bound));
  result.state_.reserve(view.source.state.size());
  result.external_.reserve(view.source.external.size());

  const auto map_tensor = [&](ir::TensorId id) -> std::optional<std::uint32_t> {
    if (!Valid(graph, id)) return std::nullopt;
    if (tensor_ids[id.v] != UINT32_MAX) return tensor_ids[id.v];
    if (result.tensors_.size() >= UINT32_MAX) return std::nullopt;
    const ir::TensorDesc& desc = graph.tensor(id);
    const ir::TensorId root = graph.root(id);
    if (!Valid(graph, root)) return std::nullopt;
    if (root_ids[root.v] == UINT32_MAX) {
      const auto storage = ir::StorageBytes(graph.tensor(root));
      if (!storage || result.roots_.size() >= UINT32_MAX) return std::nullopt;
      const auto canonical = static_cast<std::uint32_t>(result.roots_.size());
      root_ids[root.v] = canonical;
      result.roots_.push_back({.canonical = canonical,
                               .original_root = root,
                               .role = graph.tensor(root).role,
                               .format = graph.tensor(root).format,
                               .strides = graph.tensor(root).nb,
                               .storage_bytes = *storage,
                               .constant_content = graph.tensor(root).role == ir::Role::kConstant
                                                       ? ConstantContent(graph, root)
                                                       : Digest{},
                               .binding = graph.tensor(root).binding,
                               .segment = graph.tensor(root).segment});
    }
    const auto storage = ir::StorageBytes(desc);
    if (!storage) return std::nullopt;
    const auto canonical = static_cast<std::uint32_t>(result.tensors_.size());
    tensor_ids[id.v] = canonical;
    result.tensors_.push_back({.canonical = canonical,
                               .canonical_root = root_ids[root.v],
                               .role = desc.role,
                               .format = desc.format,
                               .strides = desc.nb,
                               .view_offset = desc.view_offset,
                               .binding = desc.binding,
                               .segment = desc.segment,
                               .storage_bytes = *storage});
    return canonical;
  };

  for (std::uint32_t group = 0; group < group_count; ++group) {
    for (std::uint32_t at = view.member_offsets[group]; at < view.member_offsets[group + 1]; ++at) {
      const ir::NodeId node_id = view.members[at];
      const auto& node = graph.node(node_id);
      ComparisonRegionNode record{.canonical = static_cast<std::uint32_t>(result.nodes_.size()),
                                  .group = group,
                                  .original_node = node_id,
                                  .key = node.key,
                                  .component = node.component,
                                  .batch = node.batch,
                                  .params = node.params,
                                  .nsrc = node.nsrc,
                                  .nout = node.nout};
      for (std::size_t i = 0; i < node.nsrc; ++i) {
        const auto tensor = node.source(i);
        if (!tensor.valid()) {
          record.sources[i] = UINT32_MAX;
          continue;
        }
        const auto local = map_tensor(tensor);
        if (!local) return invalid("comparison region source tensor is invalid");
        record.sources[i] = *local;
      }
      for (std::size_t i = 0; i < node.nout; ++i) {
        const auto local = map_tensor(node.outputs()[i]);
        if (!local) return invalid("comparison region output tensor is invalid");
        record.outputs[i] = *local;
      }
      result.nodes_.push_back(record);
    }
  }
  const auto add_port = [&](ComparisonRegionPortKind kind, std::uint32_t member,
                            std::uint16_t ordinal, ir::TensorId tensor) {
    const auto local = map_tensor(tensor);
    const auto root = graph.root(tensor);
    if (!local || !Valid(graph, root) || root_ids[root.v] == UINT32_MAX) return false;
    result.ports_.push_back({.kind = kind,
                             .member = member,
                             .ordinal = ordinal,
                             .canonical_tensor = *local,
                             .canonical_root = root_ids[root.v],
                             .original_tensor = tensor,
                             .view_offset = graph.tensor(tensor).view_offset});
    return true;
  };
  for (std::uint32_t member = 0; member < result.nodes_.size(); ++member) {
    const ir::NodeId source_member = view.members[member];
    const auto& node = graph.node(source_member);
    for (std::size_t i = 0; i < node.nsrc; ++i) {
      const auto tensor = node.source(i);
      if (!tensor.valid()) continue;
      const auto& desc = graph.tensor(tensor);
      const bool state =
          desc.role == ir::Role::kState || std::holds_alternative<ir::StateRef>(desc.binding);
      const ir::NodeId producer = graph.producer(tensor);
      const bool produced_inside =
          producer.valid() && producer.v < included.size() && included[producer.v] != 0;
      if ((state || !produced_inside) &&
          !add_port(state ? ComparisonRegionPortKind::kStateRead : ComparisonRegionPortKind::kInput,
                    member, static_cast<std::uint16_t>(i), tensor))
        return invalid("comparison region input port is not representable");
    }
    for (std::size_t i = 0; i < node.nout; ++i) {
      const auto tensor = node.outputs()[i];
      bool crossing = std::ranges::find(graph.outputs(), tensor) != graph.outputs().end() ||
                      std::ranges::find(graph.keep(), tensor) != graph.keep().end();
      for (std::uint32_t consumer = 0; !crossing && consumer < graph.nodes().size(); ++consumer) {
        if (included[consumer] != 0) continue;
        const auto& candidate = graph.node(ir::NodeId{consumer});
        crossing = std::ranges::find(candidate.sources(), tensor) != candidate.sources().end();
      }
      const auto& desc = graph.tensor(tensor);
      const auto target = StateStoreTarget(node);
      const bool state_root =
          desc.role == ir::Role::kState || std::holds_alternative<ir::StateRef>(desc.binding);
      const bool state_write = state_root && target && *target < node.nsrc &&
                               Valid(graph, node.source(*target)) &&
                               graph.root(node.source(*target)) == graph.root(tensor);
      if (crossing || state_write) {
        const auto kind = state_write ? ComparisonRegionPortKind::kStateWrite
                          : std::ranges::find(graph.keep(), tensor) != graph.keep().end()
                              ? ComparisonRegionPortKind::kKeep
                              : ComparisonRegionPortKind::kOutput;
        if (!add_port(kind, member, static_cast<std::uint16_t>(i), tensor))
          return invalid("comparison region output port is not representable");
      }
    }
  }

  for (const auto& weight : view.source.io.weights) {
    if (!weight.tensor.valid() || weight.tensor.v >= graph.tensor_count()) continue;
    const auto root = graph.root(weight.tensor);
    if (!Valid(graph, root) || root_ids[root.v] == UINT32_MAX) continue;
    const auto parts = format::PartsOf(graph.tensor(root).format);
    if (!parts) return invalid("comparison region weight parts are invalid");
    if (weight.parts.empty()) {
      result.weight_parts_.push_back({.canonical_root = root_ids[root.v],
                                      .ref = weight.ref,
                                      .name = {},
                                      .part = parts->parts().front().role});
    } else {
      for (const auto& [name, part] : weight.parts)
        result.weight_parts_.push_back(
            {.canonical_root = root_ids[root.v], .ref = weight.ref, .name = name, .part = part});
    }
  }
  for (const auto& state : view.source.state) {
    if (!state.tensor.valid() || state.tensor.v >= graph.tensor_count()) continue;
    const auto root = graph.root(state.tensor);
    if (Valid(graph, root) && root_ids[root.v] != UINT32_MAX)
      result.state_.push_back({.canonical_root = root_ids[root.v], .facts = state});
  }
  std::ranges::sort(result.weight_parts_, [](const auto& a, const auto& b) {
    return std::tie(a.canonical_root, a.part) < std::tie(b.canonical_root, b.part);
  });
  std::ranges::sort(result.state_, [](const auto& a, const auto& b) {
    return std::tie(a.canonical_root, a.facts.read, a.facts.written, a.facts.borrowed,
                    a.facts.tap) <
           std::tie(b.canonical_root, b.facts.read, b.facts.written, b.facts.borrowed, b.facts.tap);
  });
  std::vector<std::pair<std::uint32_t, std::uint32_t>> allocation_classes;
  allocation_classes.reserve(view.source.external.size());
  for (const auto& external : view.source.external) {
    if (!Valid(graph, external.original_root))
      return invalid("comparison region external binding root is invalid");
    const auto root = graph.root(external.original_root);
    if (root_ids[root.v] == UINT32_MAX) continue;
    std::uint64_t end = 0;
    if (external.allocation_bytes == 0 || external.bytes == 0 || external.alignment == 0 ||
        !std::has_single_bit(external.alignment) || external.offset % external.alignment != 0 ||
        __builtin_add_overflow(external.offset, external.bytes, &end) ||
        end > external.allocation_bytes)
      return invalid("comparison region external range is invalid");
    result.external_.push_back({.canonical_root = root_ids[root.v],
                                .part = external.part,
                                .canonical_allocation_class = external.allocation_class,
                                .allocation_bytes = external.allocation_bytes,
                                .offset = external.offset,
                                .bytes = external.bytes,
                                .alignment = external.alignment,
                                .resource_part_identity = external.resource_part_identity,
                                .row_storage = external.row_storage});
  }
  std::ranges::sort(result.external_, [](const auto& a, const auto& b) {
    const auto a_part = a.part ? static_cast<std::uint8_t>(*a.part) : UINT8_MAX;
    const auto b_part = b.part ? static_cast<std::uint8_t>(*b.part) : UINT8_MAX;
    return std::tie(a.canonical_root, a_part, a.offset, a.bytes, a.allocation_bytes, a.alignment,
                    a.resource_part_identity) < std::tie(b.canonical_root, b_part, b.offset,
                                                         b.bytes, b.allocation_bytes, b.alignment,
                                                         b.resource_part_identity);
  });
  for (std::size_t i = 1; i < result.external_.size(); ++i)
    if (result.external_[i - 1].canonical_root == result.external_[i].canonical_root &&
        result.external_[i - 1].part == result.external_[i].part)
      return invalid("comparison region external binding duplicates a root part");
  allocation_classes.clear();
  for (auto& external : result.external_) {
    auto at = std::ranges::find(allocation_classes, external.canonical_allocation_class,
                                &std::pair<std::uint32_t, std::uint32_t>::first);
    if (at == allocation_classes.end()) {
      allocation_classes.emplace_back(external.canonical_allocation_class,
                                      static_cast<std::uint32_t>(allocation_classes.size()));
      at = std::prev(allocation_classes.end());
    }
    external.canonical_allocation_class = at->second;
  }

  CanonicalLabels labels(view.source.active_stage);
  labels.Reserve(static_cast<std::size_t>(stage_labels), static_cast<std::size_t>(component_labels),
                 static_cast<std::size_t>(binding_records),
                 static_cast<std::size_t>(binding_records),
                 static_cast<std::size_t>(state_tensor_labels),
                 static_cast<std::size_t>(segment_labels), view.source.state.size());
  const auto node_view = std::span<const ComparisonRegionNode>(result.nodes_);
  for (std::size_t n = 0; n < node_view.size(); ++n) {
    const auto component = CanonicalComparisonComponent(node_view, n);
    if (component == UINT32_MAX) return invalid("comparison region component label is invalid");
    canonical_components.push_back(component);
    labels.SeedComponent(node_view[n].component, component);
  }
  Hash logical("llmp.comparison.region.logical.v2");
  logical.U32(kComparisonRegionCanonicalizerVersion);
  logical.U32(labels.Stage(view.source.active_stage));
  logical.U64(group_count);
  for (std::uint32_t g = 0; g < group_count; ++g) {
    logical.U64(view.member_offsets[g + 1] - view.member_offsets[g]);
    for (std::uint32_t n = 0; n < result.nodes_.size(); ++n) {
      const auto& node = result.nodes_[n];
      if (node.group != g) continue;
      logical.U32(node.canonical);
      logical.U8(static_cast<std::uint8_t>(node.key.op));
      logical.U8(static_cast<std::uint8_t>(node.key.fused));
      logical.U32(canonical_components[n]);
      logical.U8(static_cast<std::uint8_t>(node.batch));
      logical.Bytes(node.params);
      logical.U8(node.nsrc);
      for (std::size_t i = 0; i < node.nsrc; ++i) logical.U32(node.sources[i]);
      logical.U8(node.nout);
      for (std::size_t i = 0; i < node.nout; ++i) logical.U32(node.outputs[i]);
    }
  }
  logical.U64(result.roots_.size());
  for (const auto& root : result.roots_)
    if (!HashCanonicalRoot(logical, root, labels))
      return invalid("comparison region root storage parts are invalid");
  logical.U64(result.tensors_.size());
  for (const auto& tensor : result.tensors_) {
    logical.U32(tensor.canonical);
    logical.U32(tensor.canonical_root);
    logical.U64(tensor.view_offset);
    HashFormat(logical, tensor.format);
    logical.U8(static_cast<std::uint8_t>(CanonicalComparisonRole(tensor.role)));
    for (const auto stride : tensor.strides) logical.U64(stride);
    HashCanonicalBinding(logical, tensor.binding, labels);
    logical.U32(labels.Segment(tensor.segment));
    logical.U64(tensor.storage_bytes);
  }
  logical.U64(result.ports_.size());
  for (const auto& port : result.ports_) {
    logical.U8(static_cast<std::uint8_t>(port.kind));
    logical.U32(port.member);
    logical.U16(port.ordinal);
    logical.U32(port.canonical_tensor);
    logical.U32(port.canonical_root);
    logical.U64(port.view_offset);
  }
  logical.U64(result.weight_parts_.size());
  for (const auto& part : result.weight_parts_) {
    logical.U32(part.canonical_root);
    HashCanonicalBinding(logical, ir::Binding{part.ref}, labels);
    logical.U8(static_cast<std::uint8_t>(part.part));
  }
  logical.U64(result.state_.size());
  for (const auto& state : result.state_) HashCanonicalState(logical, state, labels);
  std::uint64_t region_unstored_count = 0;
  for (const auto& input : view.source.unstored_inputs) {
    if (!Valid(graph, input.original_input))
      return invalid("comparison region unstored input is invalid");
    if (root_ids[graph.root(input.original_input).v] != UINT32_MAX) ++region_unstored_count;
  }
  if (region_unstored_count != 0) {
    logical.U8(0xA7);  // versioned extension tag; absent preserves old local keys
    logical.U64(region_unstored_count);
    for (const auto& input : view.source.unstored_inputs) {
      const auto root = graph.root(input.original_input);
      if (root_ids[root.v] == UINT32_MAX) continue;
      logical.U32(root_ids[root.v]);
      logical.U8(static_cast<std::uint8_t>(input.reason));
    }
  }
  // Parent allocation facts authenticate live bindings; portable storage is
  // derived from every canonical root, not just source-external allocations.
  Hash external_hash("llmp.comparison.region.storage-parts.v2");
  external_hash.U64(result.roots_.size());
  for (const auto& root : result.roots_) {
    external_hash.U32(root.canonical);
    external_hash.U64(root.storage_bytes);
    if (!HashStorageParts(external_hash, root.format))
      return invalid("comparison region root storage parts are invalid");
  }
  result.digest_.version = kComparisonRegionCanonicalizerVersion;
  result.digest_.logical = logical.Finish();
  result.digest_.external = external_hash.Finish();
  Hash combined("llmp.comparison.region.combined.v2");
  combined.U32(kComparisonRegionCanonicalizerVersion);
  combined.DigestValue(result.digest_.logical);
  combined.DigestValue(result.digest_.external);
  result.digest_.combined = combined.Finish();
  return result;
}

std::expected<void, ComparisonBoundaryError> ValidateMappedComparisonBoundary(
    const OwnedComparisonBoundary& source, const ir::Graph& planned,
    std::span<const ir::TensorId> original_to_planned, ir::DiagnosticAdmission* diagnostic) {
  if (source.source_identity() == 0 || source.source_tensor_count() == 0 ||
      source.digest().combined == Digest{})
    return std::unexpected(Invalid("comparison boundary is empty or moved from", diagnostic));
  if (original_to_planned.size() != source.source_tensor_count())
    return std::unexpected(
        Invalid("original-to-planned map has the wrong source domain size", diagnostic));
  if (planned.inputs().size() != source.source_input_count() ||
      planned.outputs().size() != source.source_output_count() ||
      planned.keep().size() != source.source_keep_count())
    return std::unexpected(
        Invalid("planned graph changed the complete ordered interface", diagnostic));
  const auto mapping = [&](ir::TensorId original) -> ir::TensorId {
    if (!original.valid() || original.v >= original_to_planned.size()) return {};
    return original_to_planned[original.v];
  };
  // A private activation root may disappear when a full-span output view is
  // rematerialized by a fusion. Authenticate the unchanged observable value,
  // without assigning the removed intermediate a different tensor identity.
  const auto resolved_root = [&](const ComparisonBoundaryRoot& root) -> ir::TensorId {
    const auto mapped = mapping(root.original_root);
    if (mapped.valid()) return mapped;
    if (root.role != ir::Role::kActivation ||
        !std::holds_alternative<std::monostate>(root.binding) ||
        root.format.layout != format::Layout::kDense ||
        std::ranges::any_of(source.external(), [&](const auto& external) {
          return external.canonical_root == root.canonical;
        }))
      return {};
    ir::TensorId replacement;
    for (const auto& port : source.ports()) {
      if (port.canonical_root != root.canonical) continue;
      if (port.kind != ComparisonPortKind::kOutput || port.root_offset != 0 ||
          port.storage_bytes != root.storage_bytes ||
          port.format.layout != format::Layout::kDense ||
          port.format.encoding != root.format.encoding)
        return {};
      const auto id = mapping(port.original_tensor);
      if (!id.valid() || id.v >= planned.tensor_count() ||
          port.ordinal >= planned.outputs().size() || planned.outputs()[port.ordinal] != id)
        return {};
      const auto& actual = planned.tensor(id);
      const auto bytes = ir::StorageBytes(actual);
      if (port.role != actual.role || port.format != actual.format || port.strides != actual.nb ||
          port.segment != actual.segment || port.binding != actual.binding ||
          actual.view_offset != 0 || !bytes || *bytes != port.storage_bytes)
        return {};
      const auto candidate = planned.root(id);
      const auto& storage = planned.tensor(candidate);
      const auto storage_bytes = ir::StorageBytes(storage);
      if ((storage.role != ir::Role::kActivation && storage.role != ir::Role::kOutput) ||
          storage.binding != root.binding || storage.segment != root.segment ||
          storage.format.layout != root.format.layout ||
          storage.format.encoding != root.format.encoding || !storage_bytes ||
          *storage_bytes != root.storage_bytes || (replacement.valid() && replacement != candidate))
        return {};
      replacement = candidate;
    }
    return replacement;
  };
  for (const auto& root : source.roots()) {
    const auto mapped_root = resolved_root(root);
    if (!mapped_root.valid() || mapped_root.v >= planned.tensor_count())
      return std::unexpected(Invalid("required canonical root mapping is absent", diagnostic));
    const auto& actual = planned.tensor(planned.root(mapped_root));
    if (mapping(root.original_root).valid() &&
        (root.role != actual.role || root.format != actual.format || root.strides != actual.nb ||
         root.segment != actual.segment || root.binding != actual.binding))
      return std::unexpected(
          Invalid("planned canonical root differs from source contract", diagnostic));
    if (root.role == ir::Role::kConstant &&
        root.constant_content != ConstantContent(planned, planned.root(mapped_root)))
      return std::unexpected(
          Invalid("planned constant content differs from source contract", diagnostic));
  }
  for (const auto& port : source.ports()) {
    const auto original_id = port.original_tensor;
    if (!original_id.valid() || original_id.v >= original_to_planned.size())
      return std::unexpected(Invalid("boundary record has invalid original tensor ID", diagnostic));
    const auto planned_id = mapping(original_id);
    if (!planned_id.valid() || planned_id.v >= planned.tensor_count())
      return std::unexpected(
          Invalid("required boundary mapping is absent from planned graph", diagnostic));
    const auto& actual = planned.tensor(planned_id);
    const std::span<const ir::TensorId> planned_ports = [&] {
      switch (port.kind) {
        case ComparisonPortKind::kInput:
          return planned.inputs();
        case ComparisonPortKind::kOutput:
          return planned.outputs();
        case ComparisonPortKind::kKeep:
          return planned.keep();
      }
      return std::span<const ir::TensorId>{};
    }();
    if (port.ordinal >= planned_ports.size() || planned_ports[port.ordinal] != planned_id)
      return std::unexpected(
          Invalid("planned graph changed ordered input/output/keep interface", diagnostic));
    if (port.role != actual.role || port.format != actual.format || port.strides != actual.nb ||
        port.segment != actual.segment || port.binding != actual.binding ||
        port.root_offset != actual.view_offset)
      return std::unexpected(
          Invalid("planned boundary port differs from source contract", diagnostic));
    const auto storage = ir::StorageBytes(actual);
    if (!storage || *storage != port.storage_bytes)
      return std::unexpected(
          Invalid("planned boundary view extent differs from source contract", diagnostic));
    const auto root =
        std::ranges::find(source.roots(), port.canonical_root, &ComparisonBoundaryRoot::canonical);
    if (root == source.roots().end())
      return std::unexpected(Invalid("boundary port names an unknown canonical root", diagnostic));
    const auto mapped_root = resolved_root(*root);
    if (!mapped_root.valid() || mapped_root.v >= planned.tensor_count() ||
        planned.root(mapped_root) != planned.root(planned_id))
      return std::unexpected(Invalid("planned boundary changed root/view aliasing", diagnostic));
  }
  for (const auto& unstored : source.unstored_inputs()) {
    const auto original_id = unstored.original_input;
    const auto planned_id = mapping(original_id);
    if (unstored.reason != ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks ||
        !planned_id.valid() || planned_id.v >= planned.tensor_count() ||
        !std::ranges::contains(planned.inputs(), planned_id) ||
        std::ranges::contains(planned.outputs(), planned_id) ||
        std::ranges::contains(planned.keep(), planned_id))
      return std::unexpected(
          Invalid("planned unstored input changed its typed interface role", diagnostic));
    std::size_t readers = 0;
    for (const auto& node : planned.nodes()) {
      for (std::size_t source_index = 0; source_index < node.nsrc; ++source_index) {
        if (node.source(source_index) != planned_id) continue;
        ++readers;
        if (node.key != ir::Primitive(ir::Op::kMaskBuild) || source_index != 1)
          return std::unexpected(
              Invalid("planned unstored input has a non-mask reader", diagnostic));
      }
    }
    if (readers == 0)
      return std::unexpected(Invalid("planned unstored input has no mask reader", diagnostic));
  }
  for (std::size_t i = 0; i < source.ports().size(); ++i)
    for (std::size_t j = 0; j < i; ++j) {
      const auto a = mapping(source.ports()[i].original_tensor);
      const auto b = mapping(source.ports()[j].original_tensor);
      const bool same_source = source.ports()[i].canonical_root == source.ports()[j].canonical_root;
      const bool same_planned = a.valid() && b.valid() && a.v < planned.tensor_count() &&
                                b.v < planned.tensor_count() && planned.root(a) == planned.root(b);
      if (same_source != same_planned)
        return std::unexpected(
            Invalid("planned graph changed boundary root alias partition", diagnostic));
    }
  for (std::size_t i = 0; i < source.roots().size(); ++i)
    for (std::size_t j = 0; j < i; ++j) {
      const auto a = resolved_root(source.roots()[i]);
      const auto b = resolved_root(source.roots()[j]);
      if (!a.valid() || !b.valid() || a.v >= planned.tensor_count() ||
          b.v >= planned.tensor_count())
        return std::unexpected(Invalid("required canonical root mapping is absent", diagnostic));
      const bool same_source = source.roots()[i].canonical == source.roots()[j].canonical;
      const bool same_planned = planned.root(a) == planned.root(b);
      if (same_source != same_planned)
        return std::unexpected(
            Invalid("planned graph changed canonical root alias partition", diagnostic));
    }
  for (const auto& weight : source.weight_parts()) {
    const auto mapped_id = mapping(weight.original_root);
    const auto root = std::ranges::find(source.roots(), weight.canonical_root,
                                        &ComparisonBoundaryRoot::canonical);
    if (root == source.roots().end())
      return std::unexpected(Invalid("weight part names unknown canonical root", diagnostic));
    if (root->original_root != weight.original_root)
      return std::unexpected(
          Invalid("weight part root does not match canonical source root", diagnostic));
    if (!mapped_id.valid() || mapped_id.v >= planned.tensor_count())
      return std::unexpected(
          Invalid("required weight mapping is absent from planned graph", diagnostic));
    if (planned.root(mapped_id) != mapped_id)
      return std::unexpected(
          Invalid("planned weight mapping is a view, not its complete root", diagnostic));
    const auto& actual = planned.tensor(planned.root(mapped_id));
    const auto* ref = std::get_if<ir::WeightRef>(&actual.binding);
    if (!ref || *ref != weight.ref || actual.role != root->role || actual.format != root->format ||
        actual.nb != root->strides || actual.segment != root->segment)
      return std::unexpected(
          Invalid("planned weight root differs from source contract", diagnostic));
  }
  for (const auto& state : source.state()) {
    const auto root =
        std::ranges::find(source.roots(), state.canonical_root, &ComparisonBoundaryRoot::canonical);
    if (root == source.roots().end())
      return std::unexpected(Invalid("state names unknown canonical root", diagnostic));
    const auto mapped_id = mapping(state.facts.tensor);
    if (!mapped_id.valid() || mapped_id.v >= planned.tensor_count())
      return std::unexpected(
          Invalid("required state mapping is absent from planned graph", diagnostic));
    if (planned.root(mapped_id) != mapped_id)
      return std::unexpected(
          Invalid("planned state mapping is a view, not its complete root", diagnostic));
    const auto& actual = planned.tensor(planned.root(mapped_id));
    const auto* ref = std::get_if<ir::StateRef>(&actual.binding);
    if (!ref || *ref != state.facts.ref || actual.role != root->role ||
        actual.format != root->format || actual.nb != root->strides ||
        actual.segment != root->segment)
      return std::unexpected(
          Invalid("planned state root differs from source contract", diagnostic));
  }
  std::size_t planned_state_roots = 0;
  for (std::uint32_t tensor = 0; tensor < planned.tensor_count(); ++tensor) {
    const auto id = ir::TensorId{tensor};
    if (planned.root(id) != id || !std::holds_alternative<ir::StateRef>(planned.tensor(id).binding))
      continue;
    ++planned_state_roots;
    const auto mapped_source = std::ranges::find_if(source.state(), [&](const auto& state) {
      const auto mapped_id = mapping(state.facts.tensor);
      return mapped_id.valid() && mapped_id.v < planned.tensor_count() &&
             planned.root(mapped_id) == id;
    });
    if (mapped_source == source.state().end())
      return std::unexpected(
          Invalid("planned graph added an unauthenticated state root", diagnostic));
  }
  if (planned_state_roots != source.state().size())
    return std::unexpected(
        Invalid("planned graph changed the complete state root inventory", diagnostic));
  if (!source.state_facts().empty())
    if (auto checked = CheckStateWrites(planned, source.state_facts(), original_to_planned,
                                        source.active_stage());
        !checked)
      return std::unexpected(Invalid(
          std::string("state declaration failed its access/write proof: ") + checked.error(),
          diagnostic));
  if (!source.state_facts().empty())
    if (auto checked = CheckMappedStateAccess(planned, source.state_facts(), original_to_planned);
        !checked)
      return std::unexpected(Invalid(
          std::string("state declaration failed its access/write proof: ") + checked.error(),
          diagnostic));
  return {};
}

}  // namespace llmp::components
