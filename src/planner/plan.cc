// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/plan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/json_stream.h"
#include "base/sha256.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"
#include "planner/alternative.h"
#include "planner/lanes.h"
#include "planner/pattern.h"
#include "planner/placement.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::planner {
namespace {

using ir::NodeId;
using ir::TensorId;

// Decompositions may emit composite ops and fused kinds of their own; a
// plan takes at most this many rounds of them.
constexpr int kMaxPasses = 4;

std::unexpected<PlanFailure> Failed(PlanError error, std::string detail) {
  return std::unexpected(PlanFailure{.error = error, .detail = std::move(detail)});
}

PlanError ErrorOf(const ir::VerifyError& error, PlanError ordinary) {
  return error.error == ir::GraphError::kAdmission ? PlanError::kAdmission : ordinary;
}

// Comparison-only phase inventories use the real graph dimensions at the
// point the corresponding containers are about to be created.  The envelope
// is the pinned libstdc++ allocator allowance used by IR funding.
constexpr std::uint64_t kVectorEnvelope = ir::kAdmissionHeapBlockOverhead;

bool AddExact(std::uint64_t count, std::uint64_t stride, std::uint64_t& total) {
  return ir::AddVectorAdmissionBytes(count, stride, total);
}

bool AddGeometric(std::uint64_t count, std::uint64_t stride, std::uint64_t& total) {
  // Current and prospective doubled arrays may coexist while vector grows.
  if (count != 0) {
    if (stride == 0 || stride > std::numeric_limits<std::uint64_t>::max() / 3 ||
        total > std::numeric_limits<std::uint64_t>::max() - 2 * kVectorEnvelope ||
        count > (std::numeric_limits<std::uint64_t>::max() - total - 2 * kVectorEnvelope) /
                    (3 * stride))
      return false;
    total += 3 * count * stride + 2 * kVectorEnvelope;
  }
  return true;
}

bool AddBits(std::uint64_t count, std::uint64_t& total) {
  constexpr std::uint64_t word_bits = std::numeric_limits<unsigned long>::digits;
  if (count > std::numeric_limits<std::uint64_t>::max() - (word_bits - 1)) return false;
  const auto words = (count + word_bits - 1) / word_bits;
  return AddExact(words, sizeof(unsigned long), total);
}

bool AddGrowingAllocations(std::uint64_t elements, std::uint64_t allocations, std::uint64_t stride,
                           std::uint64_t& total) {
  if (elements == 0) return true;
  if (stride == 0 || stride > std::numeric_limits<std::uint64_t>::max() / 3 ||
      elements > std::numeric_limits<std::uint64_t>::max() / (3 * stride))
    return false;
  const std::uint64_t payload = 3 * elements * stride;
  if (allocations > std::numeric_limits<std::uint64_t>::max() / (2 * kVectorEnvelope)) return false;
  const std::uint64_t overhead = 2 * allocations * kVectorEnvelope;
  if (payload > std::numeric_limits<std::uint64_t>::max() - overhead ||
      total > std::numeric_limits<std::uint64_t>::max() - payload - overhead)
    return false;
  total += payload + overhead;
  return true;
}

std::optional<std::uint64_t> SelectionMetadataBytes(const ir::Graph& graph,
                                                    std::size_t registry_size) {
  const std::uint64_t n = graph.nodes().size();
  const std::uint64_t t = graph.tensor_count();
  std::uint64_t bytes = 0;
  // Select owns missing/automatic/taken bitmaps, the reader index used by
  // MatchContext, and up to one Selected record per launching node. A batch
  // candidate may retain each member and its single-candidate result.
  std::uint64_t edges = 0;
  for (const auto& node : graph.nodes()) {
    if (node.nsrc > std::numeric_limits<std::uint64_t>::max() - edges) return std::nullopt;
    edges += node.nsrc;
  }
  if (!AddExact(t, sizeof(std::vector<ir::NodeId>), bytes) ||
      !AddGrowingAllocations(edges, std::min(edges, t), sizeof(ir::NodeId), bytes) ||
      !AddBits(t, bytes) || !AddBits(t, bytes) || !AddBits(n, bytes) || !AddBits(n, bytes) ||
      !AddBits(n, bytes) || !AddGeometric(n, sizeof(Selected), bytes) ||
      !AddGrowingAllocations(n, n, sizeof(ir::NodeId), bytes) ||
      !AddGrowingAllocations(n, n, sizeof(std::size_t), bytes) ||
      !AddGrowingAllocations(n, n, sizeof(std::string_view), bytes))
    return std::nullopt;

  // GroupedStoresMatch keeps one candidate index vector per participating
  // member. Bound its real node-by-registry product before BatchAt runs.
  std::uint64_t per_node = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(registry_size), std::uint64_t{3},
                             &per_node) ||
      __builtin_mul_overflow(per_node, static_cast<std::uint64_t>(sizeof(std::size_t)),
                             &per_node) ||
      (per_node != 0 && n > (std::numeric_limits<std::uint64_t>::max() - bytes) / per_node))
    return std::nullopt;
  bytes += n * per_node;
  if (n != 0 && (n > std::numeric_limits<std::uint64_t>::max() / (2 * kVectorEnvelope) ||
                 2 * n * kVectorEnvelope > std::numeric_limits<std::uint64_t>::max() - bytes))
    return std::nullopt;
  if (n != 0) bytes += 2 * n * kVectorEnvelope;
  return bytes;
}

std::optional<std::uint64_t> DecompositionRecordsBytes(const std::vector<std::string>& names,
                                                       const ir::Graph& graph,
                                                       const std::vector<bool>& missing) {
  std::uint64_t new_count = 0;
  for (std::size_t i = 0; i < missing.size(); ++i) {
    if (!missing[i] || !ir::HasDecomposition(graph.nodes()[i].key)) continue;
    ++new_count;
  }
  constexpr std::uint64_t envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  if (names.size() > std::numeric_limits<std::uint64_t>::max() - new_count) return std::nullopt;
  const std::uint64_t total_count = names.size() + new_count;
  const std::uint64_t old_capacity = names.capacity();
  if (old_capacity > (std::numeric_limits<std::uint64_t>::max() - envelope) / sizeof(std::string) ||
      total_count > std::numeric_limits<std::uint64_t>::max() / sizeof(std::string))
    return std::nullopt;
  const std::uint64_t old_bytes =
      old_capacity == 0 ? 0 : old_capacity * sizeof(std::string) + envelope;
  const std::uint64_t new_capacity = std::max(old_capacity, total_count);
  if (new_capacity == 0) return std::uint64_t{0};
  if (new_capacity > (std::numeric_limits<std::uint64_t>::max() - envelope) / sizeof(std::string))
    return std::nullopt;
  const std::uint64_t new_bytes = new_capacity * sizeof(std::string) + envelope;
  // During explicit reserve, both vector allocations coexist. Afterwards the
  // new vector is retained through selection; string payloads are separately
  // charged by the diagnostic formatter that constructs each name.
  if (total_count > old_capacity) {
    if (old_bytes > std::numeric_limits<std::uint64_t>::max() - new_bytes) return std::nullopt;
    return old_bytes + new_bytes;
  }
  return new_bytes;
}

std::optional<std::uint64_t> PrologueMetadataBytes(const ir::Graph& graph, std::size_t step_count,
                                                   std::size_t invariant_count) {
  std::uint64_t bytes = 0;
  const std::uint64_t tensors = graph.tensor_count();
  const std::uint64_t nodes = graph.nodes().size();
  const std::uint64_t steps = step_count;
  const std::uint64_t invariants = invariant_count;
  // PrologueOf's fixed, demoted, written, prologue, and kept bitmaps, plus
  // its dynamic step/preserved results. Plan also owns the planned-input and
  // invariant result vectors while PrologueOf runs.
  if (!AddBits(tensors, bytes) || !AddBits(nodes, bytes) || !AddBits(tensors, bytes) ||
      !AddBits(steps, bytes) || !AddBits(invariants, bytes) ||
      !AddGeometric(steps, sizeof(std::uint32_t), bytes) ||
      !AddGeometric(tensors, sizeof(TensorId), bytes) ||
      !AddGeometric(invariants, sizeof(TensorId), bytes) ||
      !AddGeometric(invariants, sizeof(TensorId), bytes))
    return std::nullopt;
  return bytes;
}

std::string Describe(const ir::Graph& graph, NodeId id) {
  const ir::Node& n = graph.node(id);
  std::string text = std::format("node {} ({}", id.v, ir::OpKeyName(n.key));
  for (const TensorId s : n.sources()) {
    if (s.valid()) {
      const ir::TensorDesc& t = graph.tensor(s);
      text += std::format(" {}[{},{},{},{}]", format::Info(t.format.encoding).name, t.ne(0),
                          t.ne(1), t.ne(2), t.ne(3));
    }
  }
  return text + ")";
}

std::expected<std::string, ir::GraphError> PaidDescribeNoImplementation(
    ir::DiagnosticAdmission* admission, const ir::Graph& graph, NodeId id) {
  const ir::Node& node = graph.node(id);
  constexpr std::string_view prefix = "no implementation computes node {} (";
  constexpr std::string_view source_format = " {}[{}, {}, {}, {}]";
  constexpr std::string_view suffix = ") on this device";
  std::array<char, sizeof("no implementation computes node {} ({}) on this device") +
                       ir::kMaxSources * sizeof(" {}[{}, {}, {}, {}]")>
      format_buffer{};
  std::size_t format_size = 0;
  const auto append_format = [&](std::string_view part) {
    std::copy(part.begin(), part.end(),
              format_buffer.begin() + static_cast<std::ptrdiff_t>(format_size));
    format_size += part.size();
  };
  append_format(prefix);
  append_format("{}");
  std::array<ir::DiagnosticValue, 2 + ir::kMaxSources * 5> values{};
  std::size_t value_count = 0;
  values[value_count++] = ir::DiagnosticArg(id.v);
  values[value_count++] = ir::DiagnosticArg(node.key);
  for (const TensorId source : node.sources()) {
    if (!source.valid()) continue;
    append_format(source_format);
    const auto& tensor = graph.tensor(source);
    values[value_count++] = ir::DiagnosticArg(format::Info(tensor.format.encoding).name);
    for (const auto dim : tensor.shape().ne) values[value_count++] = ir::DiagnosticArg(dim);
  }
  append_format(suffix);
  return ir::MakeDiagnostic(admission, std::string_view(format_buffer.data(), format_size),
                            std::span<const ir::DiagnosticValue>(values).first(value_count));
}

// `graph` with flagged nodes decomposed, dropped or repeated. The returned
// maps retain their direction; `seed_to_input` is a copy ancestry only.
std::expected<PlanSeed, ir::VerifyError> CopyExpandImpl(
    const ir::Graph& graph, const std::vector<bool>& expand,
    const std::vector<bool>* drop = nullptr, const std::vector<NodeId>* repeats = nullptr,
    std::span<const ComparisonOriginSetId> origin_sets = {}, ComparisonOriginDag* origins = nullptr,
    PlannerMetadataAdmission* metadata_admission = nullptr) {
  const bool track_origins = origins != nullptr;
  if (expand.size() != graph.nodes().size() ||
      (drop != nullptr && drop->size() != graph.nodes().size()) ||
      (repeats != nullptr && repeats->size() != graph.nodes().size()))
    return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
  if (repeats != nullptr)
    for (std::uint32_t i = 0; i < repeats->size(); ++i) {
      const NodeId target = (*repeats)[i];
      if (target.valid() &&
          (target.v >= i || graph.node(target).nout != graph.node(NodeId{i}).nout ||
           (drop != nullptr && (*drop)[target.v])))
        return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
    }
  if (track_origins != !origin_sets.empty() ||
      (track_origins && origin_sets.size() != graph.nodes().size()))
    return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
  ir::Builder b;
  b.SetDiagnosticAdmission(metadata_admission);
  if (auto funder = graph.storage_funder(); funder && !b.SetStorageFunder(std::move(funder)))
    return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
  ir::StorageLoan metadata_loan;
  std::vector<TensorId> map;
  std::vector<NodeId> from;
  std::vector<ComparisonOriginSetId> expanded_origin_sets;
  ir::StorageLoan source_origins_loan;
  std::vector<ComparisonOriginSetId> source_origins;
  std::vector<NodeId> repeat_roots;
  const auto funder = graph.storage_funder();
  const auto reserve_metadata = [&](std::size_t from_capacity,
                                    std::size_t origin_capacity) -> bool {
    const auto grow = [](std::size_t current, std::size_t requested) {
      if (requested <= current) return current;
      if (current == 0 || current > std::numeric_limits<std::size_t>::max() / 2) return requested;
      return std::max(requested, current * 2);
    };
    from_capacity = grow(from.capacity(), from_capacity);
    origin_capacity = track_origins ? grow(expanded_origin_sets.capacity(), origin_capacity) : 0;
    if (!funder) {
      map.reserve(graph.tensor_count());
      from.reserve(from_capacity);
      if (track_origins) expanded_origin_sets.reserve(origin_capacity);
      return true;
    }
    std::uint64_t bytes = 0;
    if (!ir::AddVectorAdmissionBytes(graph.tensor_count(), sizeof(TensorId), bytes) ||
        !ir::AddVectorAdmissionBytes(from_capacity, sizeof(NodeId), bytes) ||
        (track_origins &&
         !ir::AddVectorAdmissionBytes(origin_capacity, sizeof(ComparisonOriginSetId), bytes)) ||
        (track_origins && repeats != nullptr &&
         !ir::AddVectorAdmissionBytes(graph.nodes().size(), sizeof(NodeId), bytes)))
      return false;
    if (!metadata_loan.GrowTo(funder, bytes)) return false;
    map.reserve(graph.tensor_count());
    from.reserve(from_capacity);
    if (track_origins) expanded_origin_sets.reserve(origin_capacity);
    if (track_origins && repeats != nullptr) {
      repeat_roots.reserve(graph.nodes().size());
      repeat_roots.resize(graph.nodes().size());
    }
    return true;
  };
  if (track_origins && !funder)
    return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
  if (!reserve_metadata(0, 0))
    return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
  map.resize(graph.tensor_count(), TensorId{});
  if (track_origins && std::ranges::any_of(origin_sets, [&](ComparisonOriginSetId id) {
        return id >= origins->entries().size();
      }))
    return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
  if (track_origins) {
    const std::uint64_t count = graph.nodes().size();
    if (count != 0) {
      const auto source_bytes = ir::VectorAdmissionBytes(count, sizeof(ComparisonOriginSetId));
      if (!source_bytes || !source_origins_loan.GrowTo(funder, *source_bytes))
        return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
      source_origins.reserve(graph.nodes().size());
    }
    source_origins.assign(origin_sets.begin(), origin_sets.end());
  }
  RewriteContext rc(graph, b, map);
  std::array<TensorId, ir::kMaxSources> sources{};
  std::array<format::Encoding, ir::kMaxOutputs> encodings{};
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const ir::Node& node = graph.nodes()[i];
    if (drop != nullptr && (*drop)[i]) continue;
    if (repeats != nullptr && (*repeats)[i].valid()) {
      // A repeat: the earlier node's outputs stand for its own.
      const NodeId repeated_id = (*repeats)[i];
      const NodeId earlier_id = track_origins ? repeat_roots[repeated_id.v] : repeated_id;
      const ir::Node& earlier = graph.node(repeated_id);
      for (std::size_t k = 0; k < node.nout; ++k) rc.Set(node.out[k], rc.Map(earlier.out[k]));
      if (track_origins && source_origins[earlier_id.v] != source_origins[i]) {
        const std::array members = {source_origins[earlier_id.v], source_origins[i]};
        auto joined = origins->AddUnion(members, metadata_admission);
        if (!joined)
          return std::unexpected(
              ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
        source_origins[earlier_id.v] = *joined;
      }
      if (track_origins) repeat_roots[i] = earlier_id;
      continue;
    }
    if (track_origins && repeats != nullptr) repeat_roots[i] = NodeId{i};
    if (!expand[i]) {
      rc.Copy(node);
    } else {
      b.SetComponent(node.component);
      if (node.batch == ir::Batch::kPerOwner && node.nout > 0) {
        b.SetOwner(graph.tensor(node.out[0]).segment);
      } else {
        b.SetJoined();
      }
      for (std::size_t j = 0; j < node.nsrc; ++j) sources[j] = rc.Map(node.src[j]);
      for (std::size_t j = 0; j < node.nout; ++j)
        encodings[j] = graph.tensor(node.out[j]).format.encoding;
      const ir::Outputs outs = ir::Decompose(
          b, node.key, node.params, std::span<const TensorId>(sources).first(node.nsrc),
          std::span<const format::Encoding>(encodings).first(node.nout));
      for (std::size_t k = 0; k < node.nout && k < outs.n; ++k) {
        rc.Set(node.out.at(k), outs[k]);
      }
    }
    if (!b.ok()) {
      auto taken_error = b.TakeError();
      ir::VerifyError e = std::move(*taken_error);
      e.node = i;
      return std::unexpected(std::move(e));
    }
    const std::size_t next_node_count = b.node_count();
    if (next_node_count > from.capacity()) {
      if (!reserve_metadata(next_node_count, expanded_origin_sets.capacity()))
        return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
    }
    from.resize(next_node_count, NodeId{i});
  }
  if (track_origins) {
    if (from.size() > expanded_origin_sets.capacity() &&
        !reserve_metadata(from.capacity(), from.size()))
      return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
    expanded_origin_sets.resize(from.size());
    for (std::size_t i = 0; i < from.size(); ++i) {
      if (!from[i].valid() || from[i].v >= source_origins.size())
        return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
      expanded_origin_sets[i] = source_origins[from[i].v];
    }
  }
  for (const TensorId o : graph.outputs()) {
    b.Output(rc.Map(o));
  }
  for (const TensorId k : graph.keep()) {
    b.Keep(rc.Map(k));
  }
  auto finished = std::move(b).Finish();
  if (!finished) return std::unexpected(std::move(finished.error()));
  return PlanSeed(std::move(*finished), std::move(metadata_loan), std::move(map), std::move(from),
                  std::move(expanded_origin_sets));
}

// A routing implementation that declares a pitched output
// (Capability::pitched_output: GGML's selected top-k writes IDs at the full
// sort's expert pitch) leaves its IDs in that pitch. Direct products and
// Cont understand it. Byte-layout operations and exposed values need the
// IR's compact IDs instead: make that copy an explicit, placed step, shared
// by those readers only. Other implementations retain their own layout
// contract and do not acquire a copy here.
std::expected<std::optional<ir::Graph>, ir::VerifyError> CompactRouteReaders(
    const ir::Graph& graph, std::span<const Selected> steps, const execution::Registry& registry,
    std::vector<TensorId>& map, std::vector<NodeId>& from,
    PlannerMetadataAdmission* metadata_admission = nullptr) {
  if (metadata_admission != nullptr) {
    std::uint64_t bytes = 0;
    const std::uint64_t tensors = graph.tensor_count();
    const std::uint64_t nodes = graph.nodes().size();
    if (nodes > std::numeric_limits<std::uint64_t>::max() - tensors || !AddBits(tensors, bytes) ||
        !AddBits(tensors, bytes) || !AddExact(tensors, sizeof(TensorId), bytes) ||
        !AddExact(tensors, sizeof(TensorId), bytes) ||
        !AddGeometric(nodes + tensors, sizeof(NodeId), bytes) ||
        !AddExact(ir::kMaxSources, sizeof(std::pair<TensorId, TensorId>), bytes))
      return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
    if (auto paid = metadata_admission->Require(PlannerMetadataPhase::kAdopt, bytes); !paid)
      return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
  }
  std::vector<bool> pitched(graph.tensor_count(), false);
  for (const auto& step : steps) {
    const auto& capability = registry.at(step.implementation).capability;
    if (!capability || !capability->pitched_output) continue;
    for (const auto id : step.nodes) {
      const auto& node = graph.node(id);
      if (node.key == ir::Primitive(ir::Op::kRoute) && node.nout > 0 && node.nsrc > 0 &&
          node.out[0].valid() && node.src[0].valid()) {
        const auto& ids_desc = graph.tensor(node.out[0]);
        const auto& gate_desc = graph.tensor(node.src[0]);
        if (ids_desc.shape().ne[0] < gate_desc.shape().ne[0]) {
          pitched[node.out[0].v] = true;
        }
      }
    }
  }
  const auto byte_layout = [](const ir::Node& node) {
    return node.key == ir::Primitive(ir::Op::kView) ||
           node.key == ir::Primitive(ir::Op::kReshape) ||
           node.key == ir::Primitive(ir::Op::kPermute);
  };
  std::vector<bool> needed(graph.tensor_count(), false);
  for (const auto& node : graph.nodes())
    for (std::size_t i = 0; i < node.nsrc; ++i)
      if (node.src[i].valid() && pitched[node.src[i].v] && byte_layout(node))
        needed[node.src[i].v] = true;
  for (const auto t : graph.outputs()) needed[t.v] = pitched[t.v];
  for (const auto t : graph.keep()) needed[t.v] = pitched[t.v];
  if (std::ranges::none_of(needed, [](bool value) { return value; })) return std::nullopt;

  ir::Builder b;
  b.SetDiagnosticAdmission(metadata_admission);
  if (auto funder = graph.storage_funder(); funder && !b.SetStorageFunder(std::move(funder)))
    return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
  map.assign(graph.tensor_count(), TensorId{});
  from.clear();
  RewriteContext rc(graph, b, map);
  std::vector<TensorId> compact(graph.tensor_count());
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const auto& node = graph.nodes()[i];
    std::vector<std::pair<TensorId, TensorId>> restore;
    for (std::size_t j = 0; j < node.nsrc; ++j) {
      const auto t = node.src[j];
      if (t.valid() && needed[t.v] && byte_layout(node)) {
        if (std::ranges::none_of(restore, [t](const auto& entry) { return entry.first == t; }))
          restore.emplace_back(t, rc.Map(t));
        rc.Set(t, compact[t.v]);
      }
    }
    rc.Copy(node);
    for (const auto& [t, original] : restore) rc.Set(t, original);
    for (const auto t : node.outputs())
      if (needed[t.v]) compact[t.v] = b.Cont(rc.Map(t));
    if (!b.ok()) return std::unexpected(std::move(*b.TakeError()));
    from.resize(b.node_count(), NodeId{i});
  }
  for (const auto t : graph.outputs()) b.Output(needed[t.v] ? compact[t.v] : rc.Map(t));
  for (const auto t : graph.keep()) b.Keep(needed[t.v] ? compact[t.v] : rc.Map(t));
  for (std::size_t i = 0; i < needed.size(); ++i)
    if (needed[i]) map[i] = compact[i];
  auto result = std::move(b).Finish();
  if (!result) return std::unexpected(std::move(result.error()));
  return std::optional<ir::Graph>(std::move(*result));
}

// A node whose run matters beyond its outputs: a leaf, a write in place (a
// cache store, a fused node's aliased output) or communication.
bool Effect(const ir::Node& n) {
  switch (n.key.op) {
    case ir::Op::kInput:
    case ir::Op::kWeight:
    case ir::Op::kState:
    case ir::Op::kCacheStore:
    case ir::Op::kAllReduce:
    case ir::Op::kAllGather:
    case ir::Op::kAllToAll:
    case ir::Op::kSend:
    case ir::Op::kRecv:
      return true;
    case ir::Op::kFused:
      return ir::WritesInPlace(n);
    default:
      return false;
  }
}

// The roots `graph` writes in place, by tensor: a cache store's target, a
// fused node's aliased source, a state, and every root a node with an
// Effect (but a weight or input leaf) reads or makes. A value read from
// one may differ between two reads, so it is neither another node's
// repeat nor a prologue's.
std::vector<bool> WrittenRoots(const ir::Graph& graph) {
  std::vector<bool> written(graph.tensor_count(), false);
  const auto mark = [&](TensorId t) {
    if (t.valid()) written.at(graph.root(t).v) = true;
  };
  for (const ir::Node& n : graph.nodes()) {
    if (!Effect(n) || n.key.op == ir::Op::kInput || n.key.op == ir::Op::kWeight) continue;
    for (const TensorId t : n.sources()) mark(t);
    for (const TensorId t : n.outputs()) mark(t);
    ir::ForEachWrite(graph, n, mark);
  }
  return written;
}

// The nodes of `graph` nothing needs: none of their outputs is the
// graph's output or kept, or read by a node that is needed, and they have
// no Effect. A rewrite leaves such nodes where its fused node reads their
// sources directly (one pass, readers after their producers). Empty when
// every node is needed.
std::vector<bool> Unread(const ir::Graph& graph) {
  std::vector<bool> needed(graph.tensor_count(), false);
  for (const auto list : {graph.outputs(), graph.keep()})
    for (const TensorId t : list) {
      needed.at(t.v) = true;
      needed.at(graph.root(t).v) = true;
    }
  std::vector<bool> unread(graph.nodes().size(), false);
  bool any = false;
  for (std::uint32_t i = static_cast<std::uint32_t>(graph.nodes().size()); i-- > 0;) {
    const ir::Node& n = graph.nodes()[i];
    if (!Effect(n) && std::ranges::none_of(n.outputs(), [&](TensorId o) {
          return needed.at(o.v) || needed.at(graph.root(o).v);
        })) {
      unread[i] = any = true;
      continue;
    }
    for (const TensorId src : n.sources())
      if (src.valid()) {
        needed.at(src.v) = true;
        needed.at(graph.root(src).v) = true;
      }
  }
  if (!any) unread.clear();
  return unread;
}

// Applied::made follows every rewrite round and subsequent graph adoption.
// Keep that authority through final selection, where an eligible incumbent
// must not admit an unqualified competitor for the same automatic operation.
std::vector<bool> AutomaticNodes(const ir::Graph& graph, std::span<const Applied> applied) {
  std::vector<bool> automatic(graph.nodes().size(), false);
  for (const auto& rewrite : applied)
    for (const auto node : rewrite.made) automatic.at(node.v) = true;
  return automatic;
}

// For each node of `graph` that repeats an earlier one, the earlier one:
// the same operation and parameters over the same sources (as repeats
// resolve), with outputs of the same roles, formats, strides and segments,
// whichever component emitted it (two blocks joining the same rows). Only
// nodes with sources and no Effect whose outputs are roots the graph
// neither outputs nor keeps (nor an Effect reads) repeat. Empty when none
// does.
std::vector<NodeId> Repeats(const ir::Graph& graph, const std::vector<bool>& automatic) {
  std::vector<TensorId> canon(graph.tensor_count());
  for (std::uint32_t t = 0; t < canon.size(); ++t) canon[t] = TensorId{t};
  // Outputs and kept tensors are never another's; a node reading or
  // making a root written in place repeats none (WrittenRoots).
  std::vector<bool> external(graph.tensor_count(), false);
  for (const auto list : {graph.outputs(), graph.keep()})
    for (const TensorId t : list) external.at(t.v) = true;
  const std::vector<bool> written = WrittenRoots(graph);
  const auto eligible = [&](const ir::Node& n) {
    return n.nsrc > 0 && n.nout > 0 && !Effect(n) &&
           std::ranges::none_of(
               n.sources(), [&](TensorId t) { return t.valid() && written.at(graph.root(t).v); }) &&
           std::ranges::none_of(n.outputs(), [&](TensorId o) {
             return external.at(o.v) || graph.tensor(o).is_view() || written.at(graph.root(o).v);
           });
  };
  const auto same = [&](const ir::Node& a, const ir::Node& b) {
    if (a.key != b.key || a.params != b.params || a.nsrc != b.nsrc || a.nout != b.nout ||
        a.batch != b.batch)
      return false;
    for (std::size_t k = 0; k < a.nsrc; ++k) {
      const TensorId x = a.src[k], y = b.src[k];
      if (x.valid() != y.valid() || (x.valid() && canon[x.v] != canon[y.v])) return false;
    }
    for (std::size_t k = 0; k < a.nout; ++k) {
      const ir::TensorDesc& x = graph.tensor(a.out[k]);
      const ir::TensorDesc& y = graph.tensor(b.out[k]);
      if (x.role != y.role || x.format != y.format || x.nb != y.nb || x.segment != y.segment)
        return false;
    }
    return true;
  };
  // Candidates by operation and first source.
  std::map<std::pair<std::uint64_t, std::uint32_t>, std::vector<std::uint32_t>> seen;
  std::vector<NodeId> repeats;
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const ir::Node& n = graph.nodes()[i];
    if (!eligible(n)) continue;
    const TensorId first = n.src[0];
    const std::pair<std::uint64_t, std::uint32_t> key{
        (std::uint64_t{static_cast<std::uint16_t>(n.key.op)} << 16U) |
            static_cast<std::uint16_t>(n.key.fused),
        first.valid() ? canon[first.v].v : TensorId::kNone};
    auto& bucket = seen[key];
    const auto at = std::ranges::find_if(bucket, [&](std::uint32_t j) {
      // Explicit and automatic operations may select different implementations.
      // Joining them would erase one caller's qualification contract.
      return automatic.at(j) == automatic.at(i) && same(graph.nodes()[j], n);
    });
    if (at == bucket.end()) {
      bucket.push_back(i);
      continue;
    }
    if (repeats.empty()) repeats.assign(graph.nodes().size(), NodeId{});
    repeats[i] = NodeId{*at};
    for (std::size_t k = 0; k < n.nout; ++k) canon[n.out[k].v] = graph.nodes()[*at].out[k];
  }
  return repeats;
}

struct Selection {
  std::vector<Selected> steps;
  std::vector<bool> missing;  // by node: no implementation
  std::optional<PlanFailure> comparison_failure;
  bool complete = true;
};

Selected One(const execution::Registry& registry, NodeId id, std::size_t implementation) {
  return Selected{.nodes = {id},
                  .implementation = implementation,
                  .group = registry.at(implementation).capability->group};
}

Selection Select(const ir::Graph& graph, const SelectionView& view,
                 std::span<const OwnerProductScope> scopes, std::span<const Applied> applied,
                 std::span<const TensorId> source_to_graph = {},
                 std::span<const ComparisonOriginSetId> origin_sets = {},
                 ComparisonOriginDag* origins = nullptr,
                 ComparisonCaptureSession* comparison_capture = nullptr) {
  Selection out;
  out.missing.assign(graph.nodes().size(), false);
  const auto automatic = AutomaticNodes(graph, applied);
  MatchContext context(graph, view);
  const execution::Registry& registry = view.registry();
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const NodeId id{i};
    const ir::Node& node = graph.node(id);
    if (LaunchesNothing(node.key) || context.taken(id)) {
      continue;
    }
    bool batched = false;
    for (const Pattern& p : Patterns()) {
      auto m = BatchAt(p, context, id);
      if (!m || std::ranges::any_of(m->nodes, [&](NodeId member) {
            return std::ranges::any_of(
                scopes, [&](const auto& scope) { return graph.producer(scope.result) == member; });
          })) {
        continue;
      }
      // Each member keeps its own selection, for a run of one (lanes.h).
      Selected step{.nodes = m->nodes,
                    .implementation = m->implementation,
                    .group = registry.at(m->implementation).capability->group,
                    .batch = p.name};
      bool singles = true;
      for (const NodeId member : m->nodes) {
        const auto& own_node = graph.node(member);
        const auto own =
            view.FinalCandidates(graph, own_node, own_node.key, automatic.at(member.v));
        if (own.empty()) {
          singles = false;
          break;
        }
        step.singles.push_back(own.front());
        step.single_groups.push_back(registry.at(own.front()).capability->group);
      }
      if (!singles) {
        continue;
      }
      if (comparison_capture != nullptr && origins != nullptr &&
          comparison_capture->ExcludesBatch(p.name, p.version, m->nodes, origin_sets, graph,
                                            source_to_graph))
        continue;
      context.Take(m->nodes);
      out.steps.push_back(std::move(step));
      batched = true;
      break;
    }
    if (batched) {
      continue;
    }
    const auto candidates = view.FinalCandidates(graph, node, node.key, automatic.at(i));
    if (candidates.empty()) {
      out.missing[i] = true;
      out.complete = false;
      continue;
    }
    out.steps.push_back(One(registry, id, candidates.front()));
  }
  return out;
}

// The steps of `steps` that read only `invariant` (planned input ids),
// weights and each other's results (a plan's prologue), and the tensors
// the other steps read from them (kept in place between runs). A step
// with any node that reads anything else, or has an Effect (a state, a
// store, communication), is not the prologue's, and what it computes is
// not invariant; nor is anything a root written in place holds.
struct Prologue {
  std::vector<std::uint32_t> steps;
  std::vector<TensorId> preserved;
  // Of `invariant`, by position, those it holds fixed (the rest, a root
  // written in place, are copied every run).
  std::vector<bool> kept;
};
Prologue PrologueOf(const ir::Graph& graph, std::span<const Selected> steps,
                    std::span<const TensorId> invariant, const RootWithin& within) {
  std::vector<bool> fixed(graph.tensor_count(), false);
  std::vector<bool> demoted(graph.nodes().size(), false);
  // A node with an Effect (a store, a state, communication) is never the
  // prologue's, nor is what a root written in place holds.
  const std::vector<bool> written = WrittenRoots(graph);
  const auto effect = [](const ir::Node& n) { return Effect(n) && n.key.op != ir::Op::kWeight; };
  std::vector<bool> prologue(steps.size(), false);
  for (bool changed = true; changed;) {
    changed = false;
    std::ranges::fill(fixed, false);
    for (const TensorId t : invariant)
      if (!written.at(graph.root(t).v)) fixed.at(t.v) = true;
    for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
      const ir::Node& n = graph.nodes()[i];
      // An input only where declared (above); a weight always.
      if (n.key == ir::Primitive(ir::Op::kInput)) continue;
      bool all = !demoted[i] && !effect(n);
      for (const TensorId src : n.sources())
        if (src.valid() && (!fixed.at(src.v) || written.at(graph.root(src).v))) all = false;
      if (all)
        for (const TensorId o : n.outputs())
          if (!written.at(graph.root(o).v) && !demoted[i]) fixed.at(o.v) = true;
    }
    // A root placed inside another (RootWithin: a blob's later output, a
    // preparation in its norm's place) shares its bytes: both fixed or
    // neither, or one's run would write the other's kept bytes.
    if (within) {
      for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
        for (const TensorId o : graph.nodes()[i].outputs()) {
          const TensorId root = graph.root(o);
          const TensorId outer = within(graph, root);
          if (!outer.valid() || fixed.at(root.v) == fixed.at(outer.v)) continue;
          const TensorId kept = fixed.at(root.v) ? root : outer;
          const NodeId producer = graph.producer(kept);
          if (producer.valid() && !demoted[producer.v]) {
            demoted[producer.v] = true;
            changed = true;
          }
        }
      }
      if (changed) continue;
    }
    for (std::size_t s = 0; s < steps.size(); ++s) {
      const bool whole = std::ranges::all_of(steps[s].nodes, [&](NodeId id) {
        return std::ranges::all_of(graph.node(id).outputs(),
                                   [&](TensorId o) { return fixed.at(o.v); });
      });
      prologue[s] = whole && !steps[s].nodes.empty();
      if (whole) continue;
      // A step that runs anyway: what it computes is not the prologue's.
      for (const NodeId id : steps[s].nodes) {
        if (!demoted[id.v] && std::ranges::any_of(graph.node(id).outputs(),
                                                  [&](TensorId o) { return fixed.at(o.v); })) {
          demoted[id.v] = true;
          changed = true;
        }
      }
    }
  }
  Prologue out;
  for (std::size_t s = 0; s < steps.size(); ++s) {
    if (prologue[s]) {
      out.steps.push_back(static_cast<std::uint32_t>(s));
      continue;
    }
    for (const NodeId id : steps[s].nodes)
      for (const TensorId src : graph.node(id).sources())
        if (src.valid() && fixed.at(src.v) && !std::ranges::contains(out.preserved, src))
          out.preserved.push_back(src);
  }
  // The step's outputs and kept tensors computed by the prologue stay too.
  for (const auto list : {graph.outputs(), graph.keep()})
    for (const TensorId t : list)
      if (fixed.at(t.v) && !std::ranges::contains(out.preserved, t)) out.preserved.push_back(t);
  // And the root a kept one lies in.
  if (within) {
    for (std::size_t k = 0; k < out.preserved.size(); ++k) {
      const TensorId outer = within(graph, graph.root(out.preserved[k]));
      if (outer.valid() && !std::ranges::contains(out.preserved, outer))
        out.preserved.push_back(outer);
    }
  }
  for (const TensorId t : invariant) out.kept.push_back(fixed.at(t.v));
  return out;
}

}  // namespace

std::expected<PlanSeed, ir::VerifyError> CopyExpand(
    const ir::Graph& graph, const std::vector<bool>& expand, const std::vector<bool>* drop,
    const std::vector<ir::NodeId>* repeats, std::span<const ComparisonOriginSetId> origin_sets,
    ComparisonOriginDag* origins, PlannerMetadataAdmission* metadata_admission) {
  return CopyExpandImpl(graph, expand, drop, repeats, origin_sets, origins, metadata_admission);
}

std::string_view PlanErrorName(PlanError error) {
  switch (error) {
    case PlanError::kNoImplementation:
      return "no implementation";
    case PlanError::kPattern:
      return "pattern";
    case PlanError::kPlacement:
      return "placement";
    case PlanError::kBounds:
      return "bounds";
    case PlanError::kPolicy:
      return "policy";
    case PlanError::kAdmission:
      return "admission";
    case PlanError::kStaleComparisonSite:
      return "stale comparison site";
    case PlanError::kAmbiguousComparisonSite:
      return "ambiguous comparison site";
    case PlanError::kMissingComparisonSite:
      return "missing comparison site";
  }
  return "unknown";
}

std::expected<StepPlan, PlanFailure> Plan(ir::Graph primitive, const execution::Registry& registry,
                                          const DeviceFacts& device, const Policy& policy,
                                          const PlanOptions& options) {
  const auto failed = [&](PlanError error,
                          std::string_view detail) -> std::unexpected<PlanFailure> {
    if (error == PlanError::kAdmission)
      return std::unexpected(PlanFailure{.error = error, .detail = {}});
    if (options.metadata_admission != nullptr) {
      auto owned = ir::FormatDiagnostic(options.metadata_admission, "{}", detail);
      if (!owned) return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
      return std::unexpected(PlanFailure{.error = error, .detail = std::move(*owned)});
    }
    return std::unexpected(PlanFailure{.error = error, .detail = std::string(detail)});
  };
  const auto failed_diagnostic = [&](PlanError error, std::string_view format,
                                     const auto&... args) -> std::unexpected<PlanFailure> {
    if (error == PlanError::kAdmission)
      return std::unexpected(PlanFailure{.error = error, .detail = {}});
    if (options.metadata_admission != nullptr) {
      auto owned = ir::FormatDiagnostic(options.metadata_admission, format, args...);
      if (!owned) return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
      return std::unexpected(PlanFailure{.error = error, .detail = std::move(*owned)});
    }
    return std::unexpected(PlanFailure{
        .error = error, .detail = std::vformat(format, std::make_format_args(args...))});
  };
  const bool track_comparison_origins = options.comparison_capture != nullptr;
  if (track_comparison_origins && options.metadata_admission == nullptr)
    return failed(PlanError::kAdmission, "comparison lineage requires a metadata admission owner");
  if (track_comparison_origins &&
      (options.comparison_origins == nullptr ||
       options.comparison_node_origins.size() != primitive.nodes().size() ||
       options.owner_products != nullptr))
    return failed(PlanError::kStaleComparisonSite,
                  "comparison lineage or owner-product source mapping is unavailable");
  if (options.metadata_admission != nullptr) {
    const auto input_bytes = ComparisonPlanInputMetadataBytes(primitive, registry, policy, options);
    if (!input_bytes) return std::unexpected(std::move(input_bytes.error()));
    if (auto paid = options.metadata_admission->Require(PlannerMetadataPhase::kInput, *input_bytes);
        !paid)
      return std::unexpected(std::move(paid.error()));
  }
  const auto primitive_digest = primitive.Digest();
  if (auto valid = Validate(policy, options.metadata_admission); !valid) {
    if (options.metadata_admission != nullptr && valid.error().empty())
      return failed(PlanError::kAdmission, {});
    return std::unexpected(
        PlanFailure{.error = PlanError::kPolicy, .detail = std::move(valid.error())});
  }
  if (options.activation_alignment != 0 &&
      (options.activation_alignment & (options.activation_alignment - 1)) != 0)
    return failed(PlanError::kPlacement, "the activation base alignment is not a power of two");
  if (options.alignment_staged) {
    for (const auto t : *options.alignment_staged) {
      if (!t.valid() || t.v >= primitive.tensor_count() || primitive.tensor(t).is_view() ||
          !std::ranges::contains(primitive.inputs(), t) ||
          (options.staged && !std::ranges::contains(*options.staged, t)))
        return failed(PlanError::kPlacement, "an aligned input is not actually staged");
    }
  }
  SelectionView view(
      registry, device, policy, options.row_invariant, options.capturable, options.state_spans,
      options.root_alignment,
      {.base = options.activation_alignment,
       .offset = options.alignment,
       .staged = options.alignment_staged ? options.alignment_staged : options.staged,
       .within = options.within},
      options.fusion_comparisons, options.metadata_admission);
  std::optional<OwnerRewritten> owners;
  std::vector<OwnerProductScope> scopes;
  std::vector<NodeId> preserve;
  if (options.owner_products != nullptr) {
    if (options.shape != nullptr) {
      for (const auto& p : options.owner_products->products()) {
        if (p.owners.size() != options.shape->segments.size())
          return failed(PlanError::kPattern, "owner product and step partitions differ");
        for (std::size_t i = 0; i < p.owners.size(); ++i)
          if (p.owners[i].slot != options.shape->segments[i].slot ||
              p.owners[i].rows != options.shape->segments[i].rows)
            return failed(PlanError::kPattern, "owner product and step rows differ");
      }
    }
    auto split = SplitOwnerProducts(primitive, *options.owner_products, options.metadata_admission);
    if (!split) return failed(ErrorOf(split.error(), PlanError::kPattern), split.error().detail);
    owners = std::move(*split);
    if (auto forwarded = view.Forward(primitive, owners->rewrite.graph, owners->rewrite.tensors);
        !forwarded)
      return failed(ErrorOf(forwarded.error(), PlanError::kPattern), forwarded.error().detail);
    scopes = owners->scopes;
    for (const auto& scope : scopes)
      preserve.push_back(owners->rewrite.graph.producer(scope.result));
  }
  const ir::Graph& pattern_input = owners ? owners->rewrite.graph : primitive;
  auto rewritten = ApplyPatterns(
      pattern_input, view, options.shape, options.state_spans, preserve,
      track_comparison_origins ? options.comparison_node_origins
                               : std::span<const ComparisonOriginSetId>{},
      track_comparison_origins ? options.comparison_origins : nullptr,
      track_comparison_origins ? options.comparison_capture : nullptr,
      track_comparison_origins ? options.comparison_source_to_graph : std::span<const TensorId>{},
      track_comparison_origins ? options.comparison_graph_node_to_source
                               : std::span<const NodeId>{},
      options.metadata_admission);
  if (!rewritten) {
    return failed_diagnostic(ErrorOf(rewritten.error(), PlanError::kPattern),
                             "a rewrite at node {}: {}", rewritten.error().node,
                             std::string_view(rewritten.error().detail));
  }
  // Observations do not participate in matching, qualification, selection or
  // keys. Attach the actual source context only after all rewrites completed.
  for (auto& applied : rewritten->applied)
    for (auto& proof : applied.numerical_proofs) {
      proof.observed.bound_model = options.observed_bound_model;
      proof.observed.model_spec = options.observed_model_spec;
    }
  if (auto forwarded = view.Forward(pattern_input, rewritten->graph, rewritten->tensors);
      !forwarded)
    return failed(ErrorOf(forwarded.error(), PlanError::kPattern), forwarded.error().detail);
  for (auto& scope : scopes) scope.result = rewritten->tensors.at(scope.result.v);
  if (owners) {
    for (auto& a : rewritten->applied) {
      for (auto& n : a.replaced) n = owners->rewrite.from.at(n.v);
      for (auto& proof : a.numerical_proofs) proof.site = owners->rewrite.from.at(proof.site.v);
    }
    for (auto& d : rewritten->dropped) d.anchor = owners->rewrite.from.at(d.anchor.v);
    std::size_t max_from = 0;
    for (const auto& nid : rewritten->from) {
      if (nid.v >= max_from) max_from = nid.v + 1;
    }
    if (options.metadata_admission != nullptr) {
      std::uint64_t made_outputs = 0;
      for (const Applied& a : owners->rewrite.applied) {
        if (a.made.size() != 0 &&
            rewritten->from.size() >
                (std::numeric_limits<std::uint64_t>::max() - made_outputs) / a.made.size())
          return failed(PlanError::kAdmission, "owner rewrite composition inventory overflows");
        made_outputs += static_cast<std::uint64_t>(a.made.size()) * rewritten->from.size();
      }
      std::uint64_t bytes = 0;
      if (!AddExact(max_from, sizeof(std::vector<NodeId>), bytes) ||
          !AddGeometric(rewritten->from.size(), sizeof(NodeId), bytes) ||
          !AddExact(rewritten->from.size(), kVectorEnvelope, bytes) ||
          !AddGeometric(made_outputs, sizeof(NodeId), bytes) ||
          !AddExact(owners->rewrite.tensors.size(), sizeof(TensorId), bytes) ||
          !AddGeometric(rewritten->applied.size() + owners->rewrite.applied.size(), sizeof(Applied),
                        bytes))
        return failed(PlanError::kAdmission, "owner rewrite composition inventory overflows");
      if (auto paid = options.metadata_admission->Require(PlannerMetadataPhase::kAdopt, bytes);
          !paid)
        return std::unexpected(std::move(paid.error()));
    }
    std::vector<std::vector<NodeId>> by_old(max_from);
    for (std::uint32_t i = 0; i < rewritten->from.size(); ++i) {
      if (rewritten->from[i].v < max_from) {
        by_old[rewritten->from[i].v].push_back(NodeId{i});
      }
    }
    for (auto& a : owners->rewrite.applied) {
      std::vector<NodeId> made;
      for (const NodeId old_id : a.made) {
        if (old_id.v < by_old.size()) {
          made.insert(made.end(), by_old[old_id.v].begin(), by_old[old_id.v].end());
        }
      }
      std::ranges::sort(made, {}, &NodeId::v);
      a.made = std::move(made);
    }
    for (auto& n : rewritten->from) n = owners->rewrite.from.at(n.v);
    std::vector<TensorId> composed;
    for (auto t : owners->rewrite.tensors)
      composed.push_back(t.valid() ? rewritten->tensors.at(t.v) : t);
    rewritten->tensors = std::move(composed);
    rewritten->applied.insert(rewritten->applied.begin(),
                              std::make_move_iterator(owners->rewrite.applied.begin()),
                              std::make_move_iterator(owners->rewrite.applied.end()));
  }
  ir::Graph graph = std::move(rewritten->graph);
  std::vector<TensorId> tensors = std::move(rewritten->tensors);
  std::vector<NodeId> from = std::move(rewritten->from);
  std::vector<Applied> applied = std::move(rewritten->applied);
  OwnedRewriteOrigins current_origins = std::move(rewritten->origin_sets);
  // `next` for `graph`, its tensors (`map`) and nodes (`back`) carried
  // through every map onto the primitive graph.
  const auto adopt =
      [&](ir::Graph next, std::span<const TensorId> map, std::span<const NodeId> back,
          OwnedRewriteOrigins next_origins = {}) -> std::expected<void, ir::VerifyError> {
    if (options.metadata_admission != nullptr) {
      std::size_t max_back = 0;
      for (const NodeId old : back) {
        if (!old.valid() || old.v >= from.size())
          return std::unexpected(
              ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
        max_back = std::max(max_back, static_cast<std::size_t>(old.v) + 1);
      }
      std::uint64_t made_outputs = 0;
      for (const Applied& a : applied) {
        if (a.made.size() != 0 &&
            back.size() >
                (std::numeric_limits<std::uint64_t>::max() - made_outputs) / a.made.size())
          return std::unexpected(
              ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
        made_outputs += static_cast<std::uint64_t>(a.made.size()) * back.size();
      }
      std::uint64_t bytes = 0;
      if (!AddExact(back.size(), sizeof(NodeId), bytes) ||
          !AddExact(max_back, sizeof(std::vector<NodeId>), bytes) ||
          !AddGeometric(back.size(), sizeof(NodeId), bytes) ||
          !AddExact(back.size(), kVectorEnvelope, bytes) ||
          !AddGeometric(made_outputs, sizeof(NodeId), bytes))
        return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
      if (auto paid = options.metadata_admission->Require(PlannerMetadataPhase::kAdopt, bytes);
          !paid)
        return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
    }
    if (options.comparison_capture &&
        !options.comparison_capture->AdvanceCheckpointMap(graph.identity(), next.identity(), map))
      return std::unexpected(ir::VerifyError{.error = ir::GraphError::kAdmission, .detail = {}});
    if (auto forwarded = view.Forward(graph, next, map); !forwarded)
      return std::unexpected(std::move(forwarded.error()));
    for (auto& scope : scopes) scope.result = map[scope.result.v];
    for (TensorId& t : tensors) {
      t = t.valid() ? map[t.v] : t;
    }
    std::vector<NodeId> composed(back.size());
    for (std::size_t j = 0; j < back.size(); ++j) {
      composed[j] = from.at(back[j].v);
    }
    std::size_t max_back = 0;
    for (const auto& nid : back) {
      if (nid.v >= max_back) max_back = nid.v + 1;
    }
    std::vector<std::vector<NodeId>> by_old(max_back);
    for (std::uint32_t j = 0; j < back.size(); ++j) {
      if (back[j].v < max_back) by_old[back[j].v].push_back(NodeId{j});
    }
    for (Applied& a : applied) {
      std::vector<NodeId> made;
      for (const NodeId old_id : a.made) {
        if (old_id.v < by_old.size()) {
          made.insert(made.end(), by_old[old_id.v].begin(), by_old[old_id.v].end());
        }
      }
      std::ranges::sort(made, {}, &NodeId::v);
      a.made = std::move(made);
    }
    from = std::move(composed);
    graph = std::move(next);
    if (track_comparison_origins) current_origins = std::move(next_origins);
    return {};
  };
  // A node computed again (the same rows joined twice) runs once.
  if (const auto repeats = Repeats(graph, AutomaticNodes(graph, applied)); !repeats.empty()) {
    auto joined = CopyExpand(
        graph, std::vector<bool>(graph.nodes().size(), false), nullptr, &repeats,
        track_comparison_origins ? std::span<const ComparisonOriginSetId>(current_origins.values)
                                 : std::span<const ComparisonOriginSetId>{},
        track_comparison_origins ? options.comparison_origins : nullptr,
        options.metadata_admission);
    if (!joined) {
      return failed_diagnostic(ErrorOf(joined.error(), PlanError::kPattern),
                               "joining repeated nodes at node {}: {}", joined.error().node,
                               std::string_view(joined.error().detail));
    }
    if (auto adopted = adopt(std::move(joined->graph), joined->input_to_seed, joined->seed_to_input,
                             OwnedRewriteOrigins{std::move(joined->metadata_loan),
                                                 std::move(joined->seed_origin_sets)});
        !adopted)
      return failed(ErrorOf(adopted.error(), PlanError::kPattern), adopted.error().detail);
  }
  // What the rewrites left unread is not run (a split pattern's fused node
  // reading the vectors another node joined for the replaced readers).
  if (const auto unread = Unread(graph); !unread.empty()) {
    auto pruned = CopyExpand(
        graph, std::vector<bool>(graph.nodes().size(), false), &unread, nullptr,
        track_comparison_origins ? std::span<const ComparisonOriginSetId>(current_origins.values)
                                 : std::span<const ComparisonOriginSetId>{},
        track_comparison_origins ? options.comparison_origins : nullptr,
        options.metadata_admission);
    if (!pruned) {
      return failed_diagnostic(ErrorOf(pruned.error(), PlanError::kPattern),
                               "pruning unread nodes at node {}: {}", pruned.error().node,
                               std::string_view(pruned.error().detail));
    }
    if (auto adopted = adopt(std::move(pruned->graph), pruned->input_to_seed, pruned->seed_to_input,
                             OwnedRewriteOrigins{std::move(pruned->metadata_loan),
                                                 std::move(pruned->seed_origin_sets)});
        !adopted)
      return failed(ErrorOf(adopted.error(), PlanError::kPattern), adopted.error().detail);
  }
  std::vector<std::string> decomposed;
  Selection selection;
  for (int pass = 0;; ++pass) {
    std::vector<TensorId> comparison_source_to_current;
    std::vector<NodeId> comparison_current_to_source;
    std::span<const TensorId> comparison_tensor_map;
    std::span<const NodeId> comparison_node_map;
    if (track_comparison_origins) {
      const std::uint64_t tensor_count = options.comparison_source_to_graph.empty()
                                             ? tensors.size()
                                             : options.comparison_source_to_graph.size();
      const std::uint64_t node_count = from.size();
      if (tensor_count > (std::numeric_limits<std::uint64_t>::max() - 2 * kVectorEnvelope) /
                             sizeof(TensorId) ||
          node_count > (std::numeric_limits<std::uint64_t>::max() -
                        tensor_count * sizeof(TensorId) - 2 * kVectorEnvelope) /
                           sizeof(NodeId))
        return failed(PlanError::kAdmission, "comparison directional map inventory overflows");
      const auto map_bytes = tensor_count * sizeof(TensorId) + node_count * sizeof(NodeId) +
                             (tensor_count == 0 ? 0 : kVectorEnvelope) +
                             (node_count == 0 ? 0 : kVectorEnvelope);
      if (auto paid =
              options.metadata_admission->Require(PlannerMetadataPhase::kComparisonMap, map_bytes);
          !paid)
        return std::unexpected(std::move(paid.error()));
      comparison_source_to_current.reserve(static_cast<std::size_t>(tensor_count));
      for (std::size_t source_index = 0; source_index < tensor_count; ++source_index) {
        const TensorId initial = options.comparison_source_to_graph.empty()
                                     ? TensorId{static_cast<std::uint32_t>(source_index)}
                                     : options.comparison_source_to_graph[source_index];
        if (!initial.valid()) {
          comparison_source_to_current.push_back(TensorId{});
        } else if (initial.v >= tensors.size()) {
          return failed(PlanError::kStaleComparisonSite,
                        "comparison source tensor map is out of range");
        } else {
          comparison_source_to_current.push_back(tensors[initial.v]);
        }
      }
      comparison_current_to_source.reserve(static_cast<std::size_t>(node_count));
      for (const NodeId initial : from) {
        if (!initial.valid())
          return failed(PlanError::kStaleComparisonSite,
                        "comparison source node map is out of range");
        if (options.comparison_graph_node_to_source.empty()) {
          comparison_current_to_source.push_back(initial);
        } else if (initial.v >= options.comparison_graph_node_to_source.size() ||
                   !options.comparison_graph_node_to_source[initial.v].valid()) {
          return failed(PlanError::kStaleComparisonSite,
                        "comparison source node map is out of range");
        } else {
          comparison_current_to_source.push_back(
              options.comparison_graph_node_to_source[initial.v]);
        }
      }
      comparison_tensor_map = comparison_source_to_current;
      comparison_node_map = comparison_current_to_source;
    }
    if (options.metadata_admission != nullptr) {
      const auto bytes = SelectionMetadataBytes(graph, registry.size());
      if (!bytes) return failed(PlanError::kAdmission, "comparison selection inventory overflows");
      if (auto paid = options.metadata_admission->Require(PlannerMetadataPhase::kSelection, *bytes);
          !paid)
        return std::unexpected(std::move(paid.error()));
    }
    const ComparisonCaptureMark decision_mark =
        options.comparison_capture != nullptr ? options.comparison_capture->DecisionJournalMark()
                                              : ComparisonCaptureMark{};
    selection = Select(
        graph, view, scopes, applied,
        track_comparison_origins ? comparison_tensor_map : std::span<const TensorId>{},
        track_comparison_origins ? std::span<const ComparisonOriginSetId>(current_origins.values)
                                 : std::span<const ComparisonOriginSetId>{},
        track_comparison_origins ? options.comparison_origins : nullptr,
        track_comparison_origins ? options.comparison_capture : nullptr);
    if (options.comparison_capture != nullptr && options.comparison_capture->record_refused())
      return failed(PlanError::kAdmission, "comparison decision journal was refused");
    if (selection.comparison_failure)
      return std::unexpected(std::move(*selection.comparison_failure));
    if (selection.complete) {
      std::vector<TensorId> map;
      std::vector<NodeId> back;
      auto compact = CompactRouteReaders(graph, selection.steps, registry, map, back,
                                         options.metadata_admission);
      if (!compact)
        return failed_diagnostic(ErrorOf(compact.error(), PlanError::kPattern),
                                 "materializing routing IDs: {}",
                                 std::string_view(compact.error().detail));
      if (*compact) {
        if (options.comparison_capture != nullptr)
          options.comparison_capture->RollbackDecisionJournal(decision_mark);
        OwnedRewriteOrigins compact_origins;
        if (track_comparison_origins) {
          constexpr std::uint64_t envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
          const std::uint64_t count = back.size();
          if (count > (std::numeric_limits<std::uint64_t>::max() - envelope) /
                          sizeof(ComparisonOriginSetId))
            return failed(PlanError::kAdmission, "route lineage capacity overflows");
          auto loan =
              ir::AcquireStorageLoan(graph.storage_funder(), count * sizeof(ComparisonOriginSetId) +
                                                                 (count == 0 ? 0 : envelope));
          if (!loan) return failed(PlanError::kAdmission, "route lineage payment was refused");
          std::vector<ComparisonOriginSetId> next;
          next.reserve(back.size());
          for (const NodeId node : back) {
            if (!node.valid() || node.v >= current_origins.values.size())
              return failed(PlanError::kStaleComparisonSite, "route lost node lineage");
            next.push_back(current_origins.values[node.v]);
          }
          compact_origins = OwnedRewriteOrigins{std::move(*loan), std::move(next)};
        }
        if (auto adopted = adopt(std::move(**compact), map, back, std::move(compact_origins));
            !adopted)
          return failed(ErrorOf(adopted.error(), PlanError::kPattern), adopted.error().detail);
        continue;
      }
      // Selection can be retried after route-reader compaction. Publish
      // comparison tickets and capture checkpoints only for the final stable
      // selection, never for a tentative pass that is about to be discarded.
      if (track_comparison_origins && options.comparison_capture != nullptr &&
          options.comparison_origins != nullptr) {
        std::optional<ComparisonCheckpoint> tentative_checkpoint;
        std::size_t accepted_batch_candidates = 0;
        for (const Selected& selected : selection.steps) {
          if (!selected.batch.empty()) {
            const Pattern* pattern = FindPattern(selected.batch);
            if (pattern == nullptr || pattern->kind != PatternKind::kBatch)
              return failed(PlanError::kStaleComparisonSite,
                            "final selected batch is not a registered pattern");
            auto captured = options.comparison_capture->CaptureBatch(
                pattern->name, pattern->version, selected.nodes, current_origins.values, graph,
                comparison_tensor_map, comparison_node_map, options.metadata_admission, applied,
                options.comparison_graph_node_to_source);
            if (!captured) return std::unexpected(std::move(captured.error()));
            if (*captured) {
              ++accepted_batch_candidates;
              if (!tentative_checkpoint) tentative_checkpoint.emplace(std::move(**captured));
            }
            auto recorded = options.comparison_capture->RecordDiscovery(
                ComparisonSiteKind::kBatch, pattern->name, pattern->version, pattern->result,
                selected.nodes, current_origins.values, graph, comparison_tensor_map,
                options.metadata_admission);
            if (!recorded) return std::unexpected(std::move(recorded.error()));
            continue;
          }
          for (const NodeId node_id : selected.nodes) {
            const auto& node = graph.node(node_id);
            ComparisonSiteKind kind{};
            std::string_view pattern_name;
            std::uint16_t version = 1;
            if (node.key.op == ir::Op::kFused) {
              kind = ComparisonSiteKind::kFusedNode;
              pattern_name = ir::FusedKindName(node.key.fused);
            } else {
              const Pattern* composite = nullptr;
              for (const Pattern& candidate : Patterns())
                if (candidate.kind == PatternKind::kComposite &&
                    std::ranges::contains(candidate.composites, node.key)) {
                  composite = &candidate;
                  break;
                }
              if (composite == nullptr) continue;
              kind = ComparisonSiteKind::kCompositeNode;
              pattern_name = composite->name;
              version = composite->version;
            }
            const std::array one{node_id};
            auto recorded = options.comparison_capture->RecordDiscovery(
                kind, pattern_name, version, node.key, one, current_origins.values, graph,
                comparison_tensor_map, options.metadata_admission);
            if (!recorded) return std::unexpected(std::move(recorded.error()));
          }
        }
        if (options.comparison_capture->record_refused())
          return failed(PlanError::kAdmission, "comparison discovery ticket was refused");
        options.comparison_capture->CommitRound(std::move(tentative_checkpoint),
                                                accepted_batch_candidates);
      }
      break;
    }
    // An incomplete selection is also tentative: primitive fallback changes
    // the graph and the selector will run again in the next planning pass.
    // Keep comparison exclusions from this discarded pass out of the
    // committed trial journal, while leaving any admission refusal sticky.
    if (options.comparison_capture != nullptr)
      options.comparison_capture->RollbackDecisionJournal(decision_mark);
    // The primitive-fallback rule: a node no implementation computes is
    // replaced by its decomposition, where it has one.
    std::size_t new_decomposed = 0;
    for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
      if (!selection.missing[i]) {
        continue;
      }
      const ir::Node& n = graph.nodes()[i];
      if (!ir::HasDecomposition(n.key)) {
        if (options.metadata_admission != nullptr) {
          auto owned = PaidDescribeNoImplementation(options.metadata_admission, graph, NodeId{i});
          if (!owned)
            return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
          return std::unexpected(
              PlanFailure{.error = PlanError::kNoImplementation, .detail = std::move(*owned)});
        }
        return failed(PlanError::kNoImplementation,
                      std::format("no implementation computes {} on this device",
                                  Describe(graph, NodeId{i})));
      }
      ++new_decomposed;
    }
    if (options.metadata_admission != nullptr) {
      auto base = SelectionMetadataBytes(graph, registry.size());
      auto records_bytes = DecompositionRecordsBytes(decomposed, graph, selection.missing);
      if (!base || !records_bytes)
        return failed(PlanError::kAdmission, "decomposition-record inventory overflows");
      if (auto paid = options.metadata_admission->Require(PlannerMetadataPhase::kSelection, *base);
          !paid)
        return std::unexpected(std::move(paid.error()));
      if (auto paid = options.metadata_admission->Require(
              PlannerMetadataPhase::kDecompositionRecords, *records_bytes);
          !paid)
        return std::unexpected(std::move(paid.error()));
      if (decomposed.size() + new_decomposed > decomposed.capacity())
        decomposed.reserve(decomposed.size() + new_decomposed);
    }
    for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
      if (!selection.missing[i]) continue;
      const ir::Node& n = graph.nodes()[i];
      if (!ir::HasDecomposition(n.key)) continue;
      if (options.metadata_admission != nullptr) {
        auto name = ir::FormatDiagnostic(options.metadata_admission, "{}", n.key);
        if (!name)
          return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
        decomposed.push_back(std::move(*name));
      } else {
        decomposed.push_back(ir::OpKeyName(n.key));
      }
    }
    if (pass + 1 == kMaxPasses) {
      return failed(PlanError::kBounds, "decompositions did not reach implemented operations");
    }
    auto expanded = CopyExpand(graph, selection.missing, nullptr, nullptr,
                               track_comparison_origins
                                   ? std::span<const ComparisonOriginSetId>(current_origins.values)
                                   : std::span<const ComparisonOriginSetId>{},
                               track_comparison_origins ? options.comparison_origins : nullptr,
                               options.metadata_admission);
    if (!expanded) {
      return failed_diagnostic(ErrorOf(expanded.error(), PlanError::kPattern),
                               "a decomposition at node {}: {}", expanded.error().node,
                               std::string_view(expanded.error().detail));
    }
    if (auto adopted =
            adopt(std::move(expanded->graph), expanded->input_to_seed, expanded->seed_to_input,
                  OwnedRewriteOrigins{std::move(expanded->metadata_loan),
                                      std::move(expanded->seed_origin_sets)});
        !adopted)
      return failed(ErrorOf(adopted.error(), PlanError::kPattern), adopted.error().detail);
  }
  for (const auto& scope : scopes) {
    if (!scope.result.valid() ||
        graph.node(graph.producer(scope.result)).key != ir::Primitive(ir::Op::kMatMul) ||
        graph.tensor(scope.result).ne(1) != scope.columns)
      return failed(PlanError::kPattern, "an owner product lost its primitive scope");
  }
  std::vector<Selected> steps = std::move(selection.steps);
  // Lane tags follow the primitive nodes into what stands for them.
  if (options.metadata_admission != nullptr) {
    const auto lane_bytes = LaneMetadataBytes(steps, options.lanes, graph.nodes().size());
    if (!lane_bytes) return failed(PlanError::kAdmission, lane_bytes.error());
    if (auto paid = options.metadata_admission->Require(PlannerMetadataPhase::kLanes, *lane_bytes);
        !paid)
      return std::unexpected(std::move(paid.error()));
  }
  LaneTags tags;
  if (!options.lanes.empty()) {
    std::uint32_t max_node = 0;
    for (const auto& [node, _] : options.lanes) {
      if (node.valid()) {
        max_node = std::max(max_node, node.v);
      }
    }
    std::vector<LaneTag> lane_lut(max_node + 1);
    std::vector<bool> has_lane(max_node + 1, false);
    for (const auto& [node, tag] : options.lanes) {
      if (node.valid()) {
        lane_lut[node.v] = tag;
        has_lane[node.v] = true;
      }
    }
    for (std::uint32_t j = 0; j < from.size(); ++j) {
      const NodeId fn = from[j];
      if (fn.valid() && fn.v < lane_lut.size() && has_lane[fn.v]) {
        tags.emplace_back(NodeId{j}, lane_lut[fn.v]);
      }
    }
  }
  std::vector<Region> regions = AssignLanes(steps, tags, registry, options.on_stream);
  if (options.metadata_admission != nullptr) {
    std::uint64_t placement_bytes = 0;
    auto placement_workspace = PlacementMetadataBytes(graph);
    const std::uint64_t staged_count =
        options.staged ? options.staged->size() : graph.inputs().size();
    const std::uint64_t prologue_invariants = regions.empty() ? options.invariant.size() : 0;
    const auto prologue_workspace = PrologueMetadataBytes(graph, steps.size(), prologue_invariants);
    if (!placement_workspace || !prologue_workspace ||
        !AddGeometric(staged_count, sizeof(TensorId), placement_bytes) ||
        *placement_workspace > std::numeric_limits<std::uint64_t>::max() - placement_bytes ||
        *prologue_workspace >
            std::numeric_limits<std::uint64_t>::max() - placement_bytes - *placement_workspace)
      return failed(PlanError::kAdmission, "comparison placement inventory overflows");
    placement_bytes += *placement_workspace + *prologue_workspace;
    if (auto paid =
            options.metadata_admission->Require(PlannerMetadataPhase::kPlacement, placement_bytes);
        !paid)
      return std::unexpected(std::move(paid.error()));
  }
  std::vector<TensorId> staged;
  if (options.staged) {
    for (const TensorId t : *options.staged) {
      if (!t.valid() || t.v >= tensors.size() || !tensors[t.v].valid()) {
        return failed(PlanError::kPlacement, "a staged input the plan does not hold");
      }
      staged.push_back(tensors[t.v]);
    }
  } else {
    staged.assign(graph.inputs().begin(), graph.inputs().end());
  }
  // The prologue (PlanOptions::invariant), without lanes.
  Prologue prologue;
  std::vector<TensorId> invariant;
  if (!options.invariant.empty() && regions.empty()) {
    std::vector<TensorId> planned;
    for (const TensorId t : options.invariant) {
      if (!t.valid() || t.v >= tensors.size() || !tensors[t.v].valid() ||
          !std::ranges::contains(graph.inputs(), tensors[t.v]))
        return failed(PlanError::kPlacement, "an invariant input the plan does not hold");
      planned.push_back(tensors[t.v]);
    }
    prologue = PrologueOf(graph, steps, planned, options.within);
    // Only what the prologue holds fixed is kept between runs (an input a
    // store or communication reads is copied every run).
    for (std::size_t k = 0; k < planned.size(); ++k)
      if (prologue.kept[k]) invariant.push_back(options.invariant[k]);
  }
  auto placement = PlaceActivations(graph, steps, regions, staged, options.alignment,
                                    options.written, options.within, prologue.preserved);
  if (!placement) {
    return failed(PlanError::kPlacement, placement.error());
  }
  auto alignments = view.Alignments(graph);
  if (!alignments.empty()) {
    std::uint32_t max_tensor = 0;
    for (const auto& [tid, _] : placement->offsets) {
      if (tid.valid()) {
        max_tensor = std::max(max_tensor, tid.v);
      }
    }
    std::vector<std::uint64_t> offset_lut(max_tensor + 1, 0);
    std::vector<bool> has_offset(max_tensor + 1, false);
    for (const auto& [tid, off] : placement->offsets) {
      if (tid.valid()) {
        offset_lut[tid.v] = off;
        has_offset[tid.v] = true;
      }
    }
    for (const auto& fact : alignments) {
      if (fact.root.valid() && fact.root.v < has_offset.size() && has_offset[fact.root.v]) {
        const std::uint64_t off = offset_lut[fact.root.v];
        if (off % fact.alignment != 0 || (options.activation_alignment != 0 &&
                                          options.activation_alignment % fact.alignment != 0)) {
          return failed(PlanError::kPlacement,
                        "an activation's placement breaks its alignment promise");
        }
      }
    }
  }
  if (options.comparison_capture != nullptr && track_comparison_origins) {
    auto finalized = options.comparison_capture->FinalizeDiscovery(
        graph, tensors, current_origins.values, options.metadata_admission);
    if (!finalized) return std::unexpected(std::move(finalized.error()));
  }
  if (options.comparison_final_lineage != nullptr) {
    if (!track_comparison_origins || options.comparison_origins == nullptr ||
        options.comparison_final_lineage->source_graph_identity == 0 || graph.identity() == 0 ||
        current_origins.values.size() != graph.nodes().size())
      return failed(PlanError::kStaleComparisonSite,
                    "final comparison lineage does not describe this planned graph");
    for (const ComparisonOriginSetId set : current_origins.values) {
      const auto count = options.comparison_origins->OriginCount(set);
      if (!count || *count == 0)
        return failed(PlanError::kStaleComparisonSite,
                      "final comparison lineage contains an invalid origin set");
    }
    options.comparison_final_lineage->planned_graph_identity = graph.identity();
    options.comparison_final_lineage->node_origins = std::move(current_origins);
  }
  return StepPlan{
      .primitive_digest = primitive_digest,
      .graph_digest = graph.Digest(),
      .policy_digest = Digest(policy),
      .graph = std::move(graph),
      .steps = std::move(steps),
      .regions = std::move(regions),
      .placement = std::move(*placement),
      .applied = std::move(applied),
      .from = std::move(from),
      .tensors = std::move(tensors),
      .decomposed = std::move(decomposed),
      .missing_numerical = std::move(rewritten->missing_numerical),
      .dropped_rewrites = std::move(rewritten->dropped),
      .owner_row_invariant =
          !Disabled(policy, "owner_row_invariant") && options.shape != nullptr &&
          options.shape->segments.size() > 1 &&
          std::ranges::all_of(options.shape->segments, [](const auto& s) { return s.rows == 1; }),
      .owner_rows_policy_enabled = !Disabled(policy, "owner_row_invariant"),
      .products =
          [&] {
            auto products = policy.products;
            if (options.shape == nullptr ||
                options.shape->rows() < products.compact_expert_min_rows)
              products.compact_experts = false;
            return products;
          }(),
      .root_alignments = std::move(alignments),
      .owner_products = std::move(scopes),
      .owner_digest =
          options.owner_products ? options.owner_products->digest() : base::Sha256Digest{},
      .invariant = std::move(invariant),
      .prologue = std::move(prologue.steps)};
}

std::expected<StepPlan, PlanFailure> Plan(std::expected<ir::Graph, ir::VerifyError> primitive,
                                          const execution::Registry& registry,
                                          const DeviceFacts& device, const Policy& policy,
                                          const PlanOptions& options) {
  if (!primitive) {
    return Failed(ErrorOf(primitive.error(), PlanError::kPattern), primitive.error().detail);
  }
  return Plan(std::move(*primitive), registry, device, policy, options);
}

std::expected<std::uint64_t, std::string> PlanReportInventory::Layout(
    const StepPlan& plan, const execution::Registry& registry, std::byte* storage) {
  std::uint64_t at = 0;
  bool overflow = false;
  // The same checked layout is measured before payment and filled afterwards.
  // Placement construction starts every typed object's lifetime in the buffer.
  const auto array = [&]<class T>(std::size_t count) -> std::span<T> {
    static_assert(std::is_trivially_destructible_v<T>);
    static_assert(alignof(T) <= alignof(std::max_align_t));
    std::uint64_t aligned = 0, bytes = 0, end = 0;
    if (__builtin_add_overflow(at, static_cast<std::uint64_t>(alignof(T) - 1), &aligned) ||
        __builtin_mul_overflow(static_cast<std::uint64_t>(count),
                               static_cast<std::uint64_t>(sizeof(T)), &bytes)) {
      overflow = true;
      return {};
    }
    aligned &= ~(static_cast<std::uint64_t>(alignof(T)) - 1);
    if (__builtin_add_overflow(aligned, bytes, &end) ||
        end > std::numeric_limits<std::size_t>::max()) {
      overflow = true;
      return {};
    }
    at = end;
    if (!storage || overflow || count == 0) return {};
    auto* data = reinterpret_cast<T*>(storage + aligned);
    for (std::size_t i = 0; i < count; ++i) std::construct_at(data + i);
    return {data, count};
  };
  const auto text = [&](std::string_view source) -> std::string_view {
    auto bytes = array.template operator()<char>(source.size());
    if (!storage || source.empty() || overflow) return {};
    std::memcpy(bytes.data(), source.data(), source.size());
    return {bytes.data(), bytes.size()};
  };
  primitive_ = plan.primitive_digest;
  graph_ = plan.graph_digest;
  policy_ = plan.policy_digest;
  registry_ = registry.Digest();
  activation_bytes_ = plan.placement.extent;
  // Registry entries are indexed directly by selections. Their immutable
  // names and hashes are copied once, regardless of the graph's step count.
  implementations_ = array.template operator()<Implementation>(registry.size());
  for (std::size_t i = 0; i < registry.size(); ++i) {
    const auto& impl = registry.at(i);
    const auto name = text(impl.name);
    if (storage && !overflow)
      implementations_[i] = {
          name, registry.identity(i),
          impl.capability ? static_cast<unsigned>(impl.capability->rung) : 0,
          impl.capability ? static_cast<unsigned>(impl.capability->expansion) : 0};
  }
  selections_ = array.template operator()<Selection>(plan.steps.size());
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    const auto& selected = plan.steps[i];
    const auto group = text(selected.group);
    auto nodes = array.template operator()<Node>(selected.nodes.size());
    if (storage && !overflow)
      selections_[i] = {selected.implementation, selected.lane, group, nodes};
    for (std::size_t j = 0; j < selected.nodes.size(); ++j) {
      const auto id = selected.nodes[j];
      const auto& node = plan.graph.node(id);
      auto sources = array.template operator()<ir::TensorId>(node.nsrc);
      auto outputs = array.template operator()<ir::TensorId>(node.nout);
      if (!storage || overflow) continue;
      std::ranges::copy(node.sources(), sources.begin());
      std::ranges::copy(node.outputs(), outputs.begin());
      nodes[j] = {.id = id,
                  .key = node.key,
                  .component = node.component,
                  .sources = sources,
                  .outputs = outputs};
      if (node.key.op == ir::Op::kMatMul || node.key.op == ir::Op::kMatMulId ||
          node.key == ir::Fused(ir::FusedKind::kGroupedMatMul))
        nodes[j].arithmetic = node.Params<ir::ProductParams>().arithmetic;
    }
  }
  patterns_ = array.template operator()<PatternRecord>(plan.applied.size());
  for (std::size_t i = 0; i < plan.applied.size(); ++i) {
    const auto& applied = plan.applied[i];
    const auto name = text(applied.pattern);
    auto proofs = array.template operator()<NumericalProof>(applied.numerical_proofs.size());
    if (storage && !overflow) {
      std::ranges::copy(applied.numerical_proofs, proofs.begin());
      patterns_[i] = {name,
                      applied.replaced.size(),
                      applied.made.size(),
                      applied.numerical_change,
                      applied.numerical_evidence,
                      applied.numerical_source,
                      applied.numerical_semantic,
                      applied.numerical_candidate,
                      applied.numerical_incumbent,
                      proofs};
    }
  }
  fallbacks_ = array.template operator()<std::string_view>(plan.decomposed.size());
  for (std::size_t i = 0; i < plan.decomposed.size(); ++i) {
    const auto name = text(plan.decomposed[i]);
    if (storage && !overflow) fallbacks_[i] = name;
  }
  activations_ = array.template operator()<Activation>(plan.placement.offsets.size());
  if (storage && !overflow) {
    for (std::size_t i = 0; i < plan.placement.offsets.size(); ++i) {
      const auto [root, offset] = plan.placement.offsets[i];
      const auto& tensor = plan.graph.tensor(root);
      auto& target = activations_[i];
      target.root = root;
      target.producer = plan.graph.producer(root);
      target.operation = plan.graph.node(target.producer).key;
      target.component = plan.graph.node(target.producer).component;
      target.offset = offset;
      target.bytes = RootBytes(tensor).value_or(0);
      for (std::size_t axis = 0; axis < 4; ++axis) {
        target.shape[axis] = tensor.ne(axis);
        target.strides[axis] = tensor.nb[axis];
      }
    }
  }
  missing_ =
      array.template operator()<MissingNumericalQualification>(plan.missing_numerical.size());
  if (storage && !overflow) std::ranges::copy(plan.missing_numerical, missing_.begin());
  if (overflow) return std::unexpected("the plan report resource layout overflows");
  return at;
}

std::expected<std::uint64_t, std::string> PlanReportInventory::StorageBytes(
    const StepPlan& plan, const execution::Registry& registry) {
  PlanReportInventory report;
  return report.Layout(plan, registry, nullptr);
}

std::expected<PlanReportInventory, std::string> PlanReportInventory::Capture(
    const StepPlan& plan, const execution::Registry& registry, std::uint64_t bytes) {
  // Reject a stale measurement before allocating or filling anything.
  auto measured = StorageBytes(plan, registry);
  if (!measured) return std::unexpected(std::move(measured.error()));
  if (*measured != bytes) return std::unexpected("the paid plan report layout changed");
  PlanReportInventory report;
  report.storage_.reset(new (std::nothrow) std::byte[static_cast<std::size_t>(bytes)]);
  if (!report.storage_) return std::unexpected("allocating the paid plan report inventory failed");
  report.storage_bytes_ = bytes;
  auto filled = report.Layout(plan, registry, report.storage_.get());
  if (!filled) return std::unexpected(std::move(filled.error()));
  return report;
}

namespace {
void StreamTensorIds(const base::json::Stream& out, std::span<const ir::TensorId> ids) {
  out.Raw("[");
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) out.Raw(",");
    if (ids[i].valid())
      out.Number(ids[i].v);
    else
      out.Raw("null");
  }
  out.Raw("]");
}
}  // namespace

namespace {
void StreamNumericalProofs(const base::json::Stream& out, std::span<const NumericalProof> proofs) {
  if (proofs.empty()) return;
  out.Raw(R"(,"numerical_recipients":[)");
  bool first = true;
  for (const auto& proof : proofs) {
    if (!first) out.Raw(",");
    first = false;
    const auto& grant = proof.qualification;
    out.Raw(R"({"primitive_site":)");
    out.Number(proof.site.v);
    out.Raw(R"(,"weight_operand":)");
    out.Number(proof.weight_operand);
    out.Raw(R"(,"stage":)");
    out.Number(grant.stage);
    out.Raw(R"(,"component":)");
    out.Number(grant.component);
    out.Raw(R"(,"weight_role":)");
    out.Number(grant.weight_role);
    out.Raw(R"(,"rows":)");
    out.Number(proof.actual.rows);
    out.Raw(R"(,"k":)");
    out.Number(proof.actual.k);
    out.Raw(R"(,"n":)");
    out.Number(proof.actual.n);
    out.Raw(R"(,"qualified_envelope":[)");
    out.Number(grant.min_rows);
    out.Raw(",");
    out.Number(grant.max_rows);
    out.Raw(",");
    out.Number(grant.max_k);
    out.Raw(",");
    out.Number(grant.max_n);
    out.Raw("]");
    if (proof.grouped_projection) {
      out.Raw(R"(,"grouped_projection":true,"qualified_leaf_envelope":[)");
      out.Number(grant.min_rows);
      out.Raw(",");
      out.Number(grant.max_rows);
      out.Raw(",");
      out.Number(proof.observed.leaf_format.shape.ne[0]);
      out.Raw(",");
      out.Number(proof.observed.leaf_format.shape.ne[1]);
      out.Raw("]");
    }
    out.Raw(R"(,"exact_shape":)");
    out.Boolean(grant.exact_shape);
    out.Raw(R"(,"exact_dimensions":)");
    out.Boolean(grant.exact_dimensions);
    out.Raw(R"(,"status":)");
    out.Quoted(proof.candidate          ? "UNQUALIFIED_CANDIDATE"
               : grant.incumbent_recipe ? "ACCEPTED_INCUMBENT_RECIPE"
                                        : "MEASURED_SAME_QUALITY");
    out.Raw(R"(,"change":)");
    out.Quoted(NumericalChangeName(grant.change));
    out.Raw(R"(,"evidence":")");
    out.Hex(grant.evidence);
    out.Raw(R"(","source":")");
    out.Hex(grant.source);
    out.Raw(R"(","semantic":")");
    out.Hex(grant.semantic);
    const auto& observed = proof.observed;
    out.Raw(R"(","observed":{"status":)");
    out.Quoted(!observed.weight                      ? "NO_WEIGHT_REF"
               : observed.active_stage == UINT16_MAX ? "ACTIVE_STAGE_UNAVAILABLE"
               : observed.weight->stage != observed.active_stage ||
                       observed.weight->component != grant.component
                   ? "FOREIGN_STAGE_OR_COMPONENT"
                   : "BOUND_WEIGHT_REF");
    out.Raw(R"(,"source_status":)");
    out.Quoted(observed.bound_model != base::Sha256Digest{} &&
                       observed.model_spec != base::Sha256Digest{}
                   ? "BOUND_MODEL_IDENTITY"
                   : "UNAVAILABLE");
    out.Raw(R"(,"active_stage":)");
    out.Number(observed.active_stage);
    out.Raw(R"(,"bound_model":")");
    out.Hex(observed.bound_model);
    out.Raw(R"(","model_spec":")");
    out.Hex(observed.model_spec);
    out.Raw("\"");
    if (observed.weight) {
      out.Raw(R"(,"stage":)");
      out.Number(observed.weight->stage);
      out.Raw(R"(,"component":)");
      out.Number(observed.weight->component);
      out.Raw(R"(,"weight_role":)");
      out.Number(observed.weight->role);
      out.Raw(R"(,"layer":)");
      out.Number(observed.weight->layer);
      out.Raw(R"(,"expert_array":)");
      out.Boolean(observed.weight->expert_array);
      const auto tensor = [&](std::string_view name, const format::TensorFormat& tensor_format,
                              const std::array<std::uint64_t, 4>& strides) {
        out.Raw(",");
        out.Quoted(name);
        out.Raw(R"(:{"encoding":)");
        out.Quoted(format::Info(tensor_format.encoding).name);
        out.Raw(R"(,"layout":)");
        out.Quoted(format::Info(tensor_format.layout).name);
        out.Raw(R"(,"rank":)");
        out.Number(tensor_format.shape.rank);
        out.Raw(R"(,"stacked":)");
        out.Number(tensor_format.stacked);
        out.Raw(R"(,"encoding_params":[)");
        out.Number(tensor_format.params.k_bits);
        out.Raw(",");
        out.Number(static_cast<std::uint8_t>(tensor_format.params.codebook));
        out.Raw(",");
        out.Boolean(tensor_format.params.tensor_scale);
        out.Raw(",");
        out.Number(tensor_format.params.bits);
        out.Raw(",");
        out.Number(tensor_format.params.group);
        out.Raw(",");
        out.Number(static_cast<std::uint8_t>(tensor_format.params.scale));
        out.Raw(",");
        out.Boolean(tensor_format.params.zero_point);
        out.Raw(",");
        out.Number(tensor_format.params.block_rows);
        out.Raw(",");
        out.Number(tensor_format.params.block_cols);
        out.Raw("]");
        out.Raw(R"(,"shape":[)");
        for (std::size_t k = 0; k < 4; ++k) {
          if (k) out.Raw(",");
          out.Number(tensor_format.shape.ne[k]);
        }
        out.Raw(R"(],"strides":[)");
        for (std::size_t k = 0; k < 4; ++k) {
          if (k) out.Raw(",");
          out.Number(strides[k]);
        }
        out.Raw("]}");
      };
      tensor("weight_operand", observed.operand_format, observed.operand_strides);
      tensor("weight_leaf", observed.leaf_format, observed.leaf_strides);
      tensor("input", observed.input_format, observed.input_strides);
    }
    out.Raw("}}");
  }
  out.Raw("]");
}
}  // namespace

void PlanReportInventory::Stream(const PlanReportSink& sink) const {
  if (!sink) return;
  const base::json::Stream out(sink);
  out.Raw(R"({"kind":"plan","version":1,"primitive":")");
  out.Hex(primitive_);
  out.Raw(R"(","graph":")");
  out.Hex(graph_);
  out.Raw(R"(","policy":")");
  out.Hex(policy_);
  out.Raw(R"(","registry":")");
  out.Hex(registry_);
  out.Raw(R"(","steps":)");
  out.Number(selections_.size());
  out.Raw(R"(,"activation_bytes":)");
  out.Number(activation_bytes_);
  out.Raw(R"(,"missing_numerical_qualifications":)");
  out.Number(missing_.size());
  out.Raw("}\n");
  const auto op = [&](ir::OpKey key) {
    out.Raw(ir::OpName(key.op));
    if (key.op == ir::Op::kFused) {
      out.Raw(".");
      out.Raw(ir::FusedKindName(key.fused));
    }
  };
  for (std::size_t i = 0; i < selections_.size(); ++i) {
    const auto& step = selections_[i];
    const auto& impl = implementations_[step.implementation];
    out.Raw(R"({"kind":"selection","step":)");
    out.Number(i);
    out.Raw(R"(,"implementation":)");
    out.Quoted(impl.name);
    out.Raw(R"(,"identity":")");
    out.Hex(impl.identity);
    out.Raw(R"(","lane":)");
    out.Number(step.lane);
    out.Raw(R"(,"rung":)");
    out.Number(impl.rung);
    out.Raw(R"(,"expansion_rule":)");
    out.Number(impl.expansion);
    out.Raw(R"(,"group":)");
    out.Quoted(step.group);
    out.Raw(R"(,"nodes":[)");
    for (std::size_t j = 0; j < step.nodes.size(); ++j) {
      const auto& node = step.nodes[j];
      if (j != 0) out.Raw(",");
      out.Raw(R"({"id":)");
      out.Number(node.id.v);
      out.Raw(R"(,"op":")");
      op(node.key);
      out.Raw("\"");
      if (node.arithmetic) {
        out.Raw(R"(,"product_arithmetic":)");
        out.Quoted(ir::ProductArithmeticName(*node.arithmetic));
      }
      out.Raw(R"(,"component":)");
      out.Number(node.component);
      out.Raw(R"(,"sources":)");
      StreamTensorIds(out, node.sources);
      out.Raw(R"(,"outputs":)");
      StreamTensorIds(out, node.outputs);
      out.Raw("}");
    }
    out.Raw("]}\n");
  }
  for (const auto& pattern : patterns_) {
    out.Raw(R"({"kind":"pattern","name":)");
    out.Quoted(pattern.name);
    out.Raw(R"(,"replaced":)");
    out.Number(pattern.replaced);
    out.Raw(R"(,"made":)");
    out.Number(pattern.made);
    if (pattern.numerical_change) {
      out.Raw(R"(,"numerical_status":)");
      out.Quoted(pattern.numerical_candidate   ? "UNQUALIFIED_CANDIDATE"
                 : pattern.numerical_incumbent ? "ACCEPTED_INCUMBENT_RECIPE"
                                               : "MEASURED_SAME_QUALITY");
      out.Raw(R"(,"numerical_change":)");
      out.Quoted(NumericalChangeName(*pattern.numerical_change));
      out.Raw(R"(,"numerical_evidence":")");
      out.Hex(pattern.evidence);
      out.Raw(R"(","numerical_source":")");
      out.Hex(pattern.source);
      out.Raw(R"(","numerical_semantic":")");
      out.Hex(pattern.semantic);
      out.Raw("\"");
    }
    StreamNumericalProofs(out, pattern.numerical_proofs);
    out.Raw("}\n");
  }
  for (const auto key : fallbacks_) {
    out.Raw(R"({"kind":"fallback","op":)");
    out.Quoted(key);
    out.Raw(R"(,"reason":"no_eligible_implementation"})");
    out.Raw("\n");
  }
  for (const auto& root : activations_) {
    out.Raw(R"({"kind":"activation","root":)");
    out.Number(root.root.v);
    out.Raw(R"(,"producer":)");
    out.Number(root.producer.v);
    out.Raw(R"(,"op":")");
    op(root.operation);
    out.Raw(R"(","component":)");
    out.Number(root.component);
    out.Raw(R"(,"offset":)");
    out.Number(root.offset);
    out.Raw(R"(,"logical_bytes":)");
    out.Number(root.bytes);
    out.Raw(R"(,"shape":[)");
    for (std::size_t d = 0; d < 4; ++d) {
      if (d != 0) out.Raw(",");
      out.Number(root.shape[d]);
    }
    out.Raw(R"(],"strides":[)");
    for (std::size_t d = 0; d < 4; ++d) {
      if (d != 0) out.Raw(",");
      out.Number(root.strides[d]);
    }
    out.Raw("]}\n");
  }
  for (const auto& missing : missing_) {
    out.Raw(R"({"kind":"missing_numerical_qualification","change":)");
    out.Quoted(NumericalChangeName(missing.change));
    out.Raw(R"(,"device_arch":)");
    out.Number(missing.device_arch);
    out.Raw(R"(,"reason":"no matching numerical evidence")");
    out.Raw(R"(,"component":)");
    out.Number(missing.component);
    out.Raw(R"(,"anchor":")");
    op(missing.anchor);
    out.Raw("\"}\n");
  }
}

void StreamPlanReport(const StepPlan& plan, const execution::Registry& registry,
                      const PlanReportSink& sink) {
  if (!sink) return;
  const base::json::Stream out(sink);
  out.Raw(R"({"kind":"plan","version":1,"primitive":")");
  out.Hex(plan.primitive_digest);
  out.Raw(R"(","graph":")");
  out.Hex(plan.graph_digest);
  out.Raw(R"(","policy":")");
  out.Hex(plan.policy_digest);
  out.Raw(R"(","registry":")");
  out.Hex(registry.Digest());
  out.Raw(R"(","steps":)");
  out.Number(plan.steps.size());
  out.Raw(R"(,"activation_bytes":)");
  out.Number(plan.placement.extent);
  out.Raw(R"(,"missing_numerical_qualifications":)");
  out.Number(plan.missing_numerical.size());
  out.Raw("}\n");
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    const auto& step = plan.steps[i];
    const auto& impl = registry.at(step.implementation);
    out.Raw(R"({"kind":"selection","step":)");
    out.Number(i);
    out.Raw(R"(,"implementation":)");
    out.Quoted(impl.name);
    out.Raw(R"(,"identity":")");
    out.Hex(registry.identity(step.implementation));
    out.Raw(R"(","lane":)");
    out.Number(step.lane);
    out.Raw(R"(,"rung":)");
    out.Number(impl.capability ? static_cast<unsigned>(impl.capability->rung) : 0);
    out.Raw(R"(,"expansion_rule":)");
    out.Number(impl.capability ? static_cast<unsigned>(impl.capability->expansion) : 0);
    out.Raw(R"(,"group":)");
    out.Quoted(step.group);
    out.Raw(R"(,"nodes":[)");
    for (std::size_t j = 0; j < step.nodes.size(); ++j) {
      const auto id = step.nodes[j];
      if (j != 0) out.Raw(",");
      out.Raw(R"({"id":)");
      out.Number(id.v);
      out.Raw(R"(,"op":")");
      const auto key = plan.graph.node(id).key;
      out.Raw(ir::OpName(key.op));
      if (key.op == ir::Op::kFused) {
        out.Raw(".");
        out.Raw(ir::FusedKindName(key.fused));
      }
      out.Raw("\"");
      if (key.op == ir::Op::kMatMul || key.op == ir::Op::kMatMulId ||
          key == ir::Fused(ir::FusedKind::kGroupedMatMul)) {
        const auto params = plan.graph.node(id).Params<ir::ProductParams>();
        out.Raw(R"(,"product_arithmetic":)");
        out.Quoted(ir::ProductArithmeticName(params.arithmetic));
      }
      const auto& node = plan.graph.node(id);
      out.Raw(R"(,"component":)");
      out.Number(node.component);
      out.Raw(R"(,"sources":)");
      StreamTensorIds(out, node.sources());
      out.Raw(R"(,"outputs":)");
      StreamTensorIds(out, node.outputs());
      out.Raw("}");
    }
    out.Raw("]}\n");
  }
  for (const auto& applied : plan.applied) {
    out.Raw(R"({"kind":"pattern","name":)");
    out.Quoted(applied.pattern);
    out.Raw(R"(,"replaced":)");
    out.Number(applied.replaced.size());
    out.Raw(R"(,"made":)");
    out.Number(applied.made.size());
    if (applied.numerical_change) {
      out.Raw(R"(,"numerical_status":)");
      out.Quoted(applied.numerical_candidate   ? "UNQUALIFIED_CANDIDATE"
                 : applied.numerical_incumbent ? "ACCEPTED_INCUMBENT_RECIPE"
                                               : "MEASURED_SAME_QUALITY");
      out.Raw(R"(,"numerical_change":)");
      out.Quoted(NumericalChangeName(*applied.numerical_change));
      out.Raw(R"(,"numerical_evidence":")");
      out.Hex(applied.numerical_evidence);
      out.Raw(R"(","numerical_source":")");
      out.Hex(applied.numerical_source);
      out.Raw(R"(","numerical_semantic":")");
      out.Hex(applied.numerical_semantic);
      out.Raw("\"");
    }
    StreamNumericalProofs(out, applied.numerical_proofs);
    out.Raw("}\n");
  }
  for (const auto& key : plan.decomposed) {
    out.Raw(R"({"kind":"fallback","op":)");
    out.Quoted(key);
    out.Raw(R"(,"reason":"no_eligible_implementation"})");
    out.Raw("\n");
  }
  for (const auto& [root, offset] : plan.placement.offsets) {
    const auto& t = plan.graph.tensor(root);
    const auto bytes = RootBytes(t);
    out.Raw(R"({"kind":"activation","root":)");
    out.Number(root.v);
    const auto producer = plan.graph.producer(root);
    const auto& node = plan.graph.node(producer);
    out.Raw(R"(,"producer":)");
    out.Number(producer.v);
    out.Raw(R"(,"op":")");
    out.Raw(ir::OpName(node.key.op));
    if (node.key.op == ir::Op::kFused) {
      out.Raw(".");
      out.Raw(ir::FusedKindName(node.key.fused));
    }
    out.Raw(R"(","component":)");
    out.Number(node.component);
    out.Raw(R"(,"offset":)");
    out.Number(offset);
    out.Raw(R"(,"logical_bytes":)");
    out.Number(bytes.value_or(0));
    out.Raw(R"(,"shape":[)");
    for (std::size_t d = 0; d < 4; ++d) {
      if (d != 0) out.Raw(",");
      out.Number(t.ne(d));
    }
    out.Raw(R"(],"strides":[)");
    for (std::size_t d = 0; d < 4; ++d) {
      if (d != 0) out.Raw(",");
      out.Number(t.nb[d]);
    }
    out.Raw("]}\n");
  }
  for (const auto& missing : plan.missing_numerical) {
    out.Raw(R"({"kind":"missing_numerical_qualification","change":)");
    out.Quoted(NumericalChangeName(missing.change));
    out.Raw(R"(,"device_arch":)");
    out.Number(missing.device_arch);
    out.Raw(R"(,"reason":"no matching numerical evidence")");
    out.Raw(R"(,"component":)");
    out.Number(missing.component);
    out.Raw(R"(,"anchor":")");
    out.Raw(ir::OpName(missing.anchor.op));
    if (missing.anchor.op == ir::Op::kFused) {
      out.Raw(".");
      out.Raw(ir::FusedKindName(missing.anchor.fused));
    }
    out.Raw("\"}\n");
  }
}

void ReportPlan(const StepPlan& plan, const execution::Registry& registry,
                const PlanReportSink& sink) {
  if (!sink) return;
  std::string record;
  StreamPlanReport(plan, registry, [&](std::string_view fragment) {
    auto newline = fragment.find('\n');
    while (newline != std::string_view::npos) {
      record.append(fragment.substr(0, newline));
      sink(record);
      record.clear();
      fragment.remove_prefix(newline + 1);
      newline = fragment.find('\n');
    }
    record.append(fragment);
  });
}

PlanKey KeyOf(const base::Sha256Digest& spec, const StepShape& shape, const Policy& policy,
              std::uint32_t partition) {
  return PlanKey{.spec = spec,
                 .shape = shape,
                 .patterns = PatternsDigest(policy),
                 .policy = Digest(policy),
                 .partition = partition};
}

PlanKey KeyOf(const base::Sha256Digest& spec, const StepShape& shape,
              const base::Sha256Digest& policy_digest, const base::Sha256Digest& patterns_digest,
              std::uint32_t partition) {
  return PlanKey{.spec = spec,
                 .shape = shape,
                 .patterns = patterns_digest,
                 .policy = policy_digest,
                 .partition = partition};
}

}  // namespace llmp::planner
