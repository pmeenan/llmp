// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/alternative.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "ir/decompose.h"

namespace llmp::planner {
struct ComparisonCheckpointAuthorityAccess {
  static void RecaptureAfterDecisions(ComparisonCaptureSession& session) {
    session.recapture_after_decisions_ = true;
  }
  template <class... Args>
  static GraphAlternative MakeAlternative(Args&&... args) {
    return GraphAlternative(std::forward<Args>(args)...);
  }
  static WholeComparisonCheckpoint Mint(std::uint64_t source, std::uint64_t checkpoint,
                                        const base::Sha256Digest& subject) {
    WholeComparisonCheckpoint result;
    result.source_identity_ = source;
    result.checkpoint_identity_ = checkpoint;
    result.subject_ = subject;
    return result;
  }
};
namespace {

std::uint64_t NextComparisonLedgerIdentity() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  while (value != 0 && value != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  }
  return 0;
}

std::unexpected<PlanFailure> Failure(PlanError error, const char* detail,
                                     ir::DiagnosticAdmission* admission = nullptr) {
  if (error == PlanError::kAdmission)
    return std::unexpected(PlanFailure{.error = error, .detail = {}});
  auto owned = ir::FormatDiagnostic(admission, "{}", std::string_view(detail));
  if (!owned) return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  return std::unexpected(PlanFailure{.error = error, .detail = std::move(*owned)});
}

// Copy the authentic earlier transaction, not a reconstructed match. All
// nested storage is prepaid; the returned vectors die before their loan.
std::expected<std::vector<Applied>, PlanFailure> CopyAppliedHistory(
    std::span<const Applied> history, std::span<const ir::NodeId> to_source,
    std::size_t current_nodes, std::size_t source_nodes, ir::StorageLoan& loan,
    ir::DiagnosticAdmission* diagnostic) {
  constexpr std::uint64_t envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  std::uint64_t bytes = 0;
  const auto add = [&](std::uint64_t count, std::uint64_t stride) {
    if (count == 0) return true;
    std::uint64_t payload = 0;
    if (__builtin_mul_overflow(count, stride, &payload) ||
        __builtin_add_overflow(payload, envelope, &payload) ||
        __builtin_add_overflow(bytes, payload, &bytes))
      return false;
    return true;
  };
  const auto mapped = [&](ir::NodeId id) {
    if (!id.valid()) return ir::NodeId{};
    if (!to_source.empty()) {
      if (id.v >= to_source.size()) return ir::NodeId{};
      id = to_source[id.v];
    }
    return id.valid() && id.v < source_nodes ? id : ir::NodeId{};
  };
  if (!add(history.size(), sizeof(Applied)))
    return Failure(PlanError::kAdmission, "checkpoint rewrite inventory overflows", diagnostic);
  for (const auto& a : history) {
    if (!FindPattern(a.pattern))
      return Failure(PlanError::kStaleComparisonSite, "checkpoint rewrite is not registered",
                     diagnostic);
    for (const auto id : a.replaced)
      if (!mapped(id).valid())
        return Failure(PlanError::kStaleComparisonSite, "checkpoint rewrite source is out of range",
                       diagnostic);
    for (const auto id : a.made)
      if (!id.valid() || id.v >= current_nodes)
        return Failure(PlanError::kStaleComparisonSite, "checkpoint rewrite result is out of range",
                       diagnostic);
    for (const auto& proof : a.numerical_proofs)
      if (!mapped(proof.site).valid())
        return Failure(PlanError::kStaleComparisonSite,
                       "checkpoint numerical source is out of range", diagnostic);
    if (!add(a.replaced.size(), sizeof(ir::NodeId)) || !add(a.made.size(), sizeof(ir::NodeId)) ||
        !add(a.numerical_proofs.size(), sizeof(NumericalProof)))
      return Failure(PlanError::kAdmission, "checkpoint rewrite inventory overflows", diagnostic);
  }
  if (bytes != 0 && !loan.Grow(bytes))
    return Failure(PlanError::kAdmission, "checkpoint rewrite storage was refused", diagnostic);
  std::vector<Applied> copied;
  copied.reserve(history.size());
  for (const auto& a : history) {
    copied.push_back(a);
    for (auto& id : copied.back().replaced) id = mapped(id);
    for (auto& proof : copied.back().numerical_proofs) proof.site = mapped(proof.site);
  }
  return copied;
}

std::unexpected<PlanFailure> LedgerFailure(PlanError error, const char* detail,
                                           ir::DiagnosticAdmission* diagnostic) {
  if (diagnostic == nullptr) return std::unexpected(PlanFailure{.error = error, .detail = {}});
  return Failure(error, detail, diagnostic);
}

bool CapacityBytes(std::uint64_t entries, std::uint64_t children, std::uint64_t source_node_count,
                   std::uint64_t& bytes) {
  constexpr std::uint64_t kVectorEnvelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  constexpr std::uint64_t kEntryStride = sizeof(ComparisonOriginEntry) + sizeof(std::uint8_t);
  if (entries > std::numeric_limits<std::uint64_t>::max() / kEntryStride) return false;
  bytes = entries * kEntryStride;
  if (entries != 0) {
    if (2 * kVectorEnvelope > std::numeric_limits<std::uint64_t>::max() - bytes) return false;
    bytes += 2 * kVectorEnvelope;
  }
  if (children > std::numeric_limits<std::uint64_t>::max() / sizeof(ComparisonOriginSetId))
    return false;
  const auto child_bytes = children * sizeof(ComparisonOriginSetId);
  if (child_bytes > std::numeric_limits<std::uint64_t>::max() - bytes) return false;
  bytes += child_bytes;
  if (children != 0) {
    if (kVectorEnvelope > std::numeric_limits<std::uint64_t>::max() - bytes) return false;
    bytes += kVectorEnvelope;
  }
  if (source_node_count > std::numeric_limits<std::uint64_t>::max() - bytes) return false;
  bytes += source_node_count;
  if (source_node_count != 0) {
    if (kVectorEnvelope > std::numeric_limits<std::uint64_t>::max() - bytes) return false;
    bytes += kVectorEnvelope;
  }
  return true;
}

bool AddBytes(std::uint64_t count, std::uint64_t element, std::uint64_t& total) {
  constexpr std::uint64_t kEnvelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  if (count > (std::numeric_limits<std::uint64_t>::max() - total) / element) return false;
  total += count * element;
  if (count != 0) {
    if (kEnvelope > std::numeric_limits<std::uint64_t>::max() - total) return false;
    total += kEnvelope;
  }
  return true;
}

bool AddText(std::string_view text, std::uint64_t& total) {
  constexpr std::uint64_t kEnvelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  const std::uint64_t bytes = static_cast<std::uint64_t>(text.size()) + 1;
  if (total > std::numeric_limits<std::uint64_t>::max() - kEnvelope ||
      bytes > std::numeric_limits<std::uint64_t>::max() - total - kEnvelope)
    return false;
  total += bytes + kEnvelope;
  return true;
}

bool AddOriginSets(std::span<const ComparisonOriginSet> sets, std::uint64_t& total) {
  if (!AddBytes(sets.size(), sizeof(ComparisonOriginSet), total)) return false;
  for (const auto& set : sets)
    if (!AddBytes(set.size(), sizeof(ir::NodeId), total)) return false;
  return true;
}

std::optional<std::uint64_t> SiteStorageBytes(const ComparisonSiteSelector& selector,
                                              std::string_view canonical_pattern) {
  std::uint64_t bytes = sizeof(ComparisonSite) + sizeof(ComparisonSiteSelector);
  if (!AddText(canonical_pattern, bytes) || !AddOriginSets(selector.member_origins, bytes) ||
      !AddOriginSets(selector.member_origins, bytes) ||
      !AddBytes(selector.member_owner_segments.size(), sizeof(std::uint8_t), bytes) ||
      !AddBytes(selector.member_owner_segments.size(), sizeof(std::uint8_t), bytes) ||
      !AddBytes(selector.boundary_ports.size(), sizeof(ComparisonSitePortRef), bytes) ||
      !AddBytes(selector.boundary_ports.size(), sizeof(ComparisonSitePortRef), bytes))
    return std::nullopt;
  return bytes;
}

std::optional<std::uint64_t> SiteSupplementBytes(const ComparisonSite& site) {
  std::uint64_t bytes = 0;
  if (!AddBytes(site.complete_member_owner_segments.size(), sizeof(std::uint8_t), bytes) ||
      !AddBytes(site.complete_boundary_ports.size(), sizeof(ComparisonSitePortRef), bytes))
    return std::nullopt;
  return bytes;
}

template <class UInt>
void HashInteger(base::Sha256& hash, UInt value) {
  static_assert(std::is_unsigned_v<UInt>);
  std::array<std::byte, sizeof(UInt)> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<std::byte>((value >> (i * 8U)) & static_cast<UInt>(0xff));
  hash.Update(bytes);
}

bool HasCheckpointMatch(const ComparisonSiteSelector& selector) {
  return std::ranges::any_of(selector.checkpoint_match, [](auto b) { return b != 0; });
}

// Graph::Digest names compute nodes in declaration order while leaf references
// use input positions or semantic weight/state bindings. Match references use
// that same compute order, so independent graph instances and leaf renumbering
// retain the same subject. Raw IDs remain lookup keys in this exact graph only.
std::uint32_t ComputeOrdinal(const ir::Graph& graph, ir::NodeId member) {
  std::uint32_t ordinal = 0;
  for (std::uint32_t n = 0; n < graph.nodes().size(); ++n) {
    const auto op = graph.node(ir::NodeId{n}).key.op;
    const bool leaf = op == ir::Op::kInput || op == ir::Op::kWeight || op == ir::Op::kState;
    if (n == member.v) return leaf ? UINT32_MAX : ordinal;
    if (!leaf) ++ordinal;
  }
  return UINT32_MAX;
}

base::Sha256Digest CheckpointMatchIdentity(const ir::Graph& graph,
                                           std::span<const ir::NodeId> members) {
  base::Sha256 hash;
  hash.Update("llmp.comparison.checkpoint-match.v1\n");
  const auto digest = graph.Digest();
  hash.Update(std::as_bytes(std::span(digest)));
  HashInteger(hash, static_cast<std::uint64_t>(members.size()));
  for (const auto member : members) HashInteger(hash, ComputeOrdinal(graph, member));
  return hash.Finish();
}

base::Sha256Digest WholeCheckpointSubject(const ir::Graph& source,
                                          const ComparisonCheckpoint& checkpoint,
                                          const ComparisonSite& site) {
  base::Sha256 hash;
  hash.Update("llmp.comparison.whole-checkpoint.v1\n");
  const auto graph = checkpoint.graph.Digest();
  hash.Update(std::as_bytes(std::span(graph)));
  hash.Update(std::as_bytes(std::span(site.selector.checkpoint_match)));
  HashInteger(hash, static_cast<std::uint64_t>(checkpoint.prior_applied().size()));
  for (const auto& prior : checkpoint.prior_applied()) {
    const auto* pattern = FindPattern(prior.pattern);
    HashInteger(hash, static_cast<std::uint64_t>(prior.pattern.size()));
    hash.Update(prior.pattern);
    HashInteger(hash, pattern ? pattern->version : std::uint16_t{0});
    HashInteger(hash, static_cast<std::uint64_t>(prior.replaced.size()));
    for (const auto n : prior.replaced) HashInteger(hash, ComputeOrdinal(source, n));
    HashInteger(hash, static_cast<std::uint64_t>(prior.made.size()));
    for (const auto n : prior.made) HashInteger(hash, ComputeOrdinal(checkpoint.graph, n));
    HashInteger(hash, static_cast<std::uint8_t>(prior.numerical_change.has_value()));
    if (prior.numerical_change)
      HashInteger(hash, static_cast<std::uint8_t>(*prior.numerical_change));
    HashInteger(hash, static_cast<std::uint8_t>(prior.numerical_candidate));
    HashInteger(hash, static_cast<std::uint8_t>(prior.numerical_incumbent));
    hash.Update(std::as_bytes(std::span(prior.numerical_evidence)));
    hash.Update(std::as_bytes(std::span(prior.numerical_source)));
    hash.Update(std::as_bytes(std::span(prior.numerical_semantic)));
    HashInteger(hash, static_cast<std::uint64_t>(prior.numerical_proofs.size()));
    for (const auto& proof : prior.numerical_proofs) {
      HashInteger(hash, ComputeOrdinal(source, proof.site));
      const auto& q = proof.qualification;
      for (const auto scalar : std::array<std::uint64_t, 23>{proof.weight_operand,
                                                             proof.candidate,
                                                             proof.grouped_projection,
                                                             static_cast<unsigned>(q.change),
                                                             q.stage,
                                                             q.component,
                                                             static_cast<unsigned>(q.anchor.op),
                                                             static_cast<unsigned>(q.anchor.fused),
                                                             q.arch,
                                                             q.min_rows,
                                                             q.max_rows,
                                                             q.max_k,
                                                             q.max_n,
                                                             q.exact_shape,
                                                             q.exact_dimensions,
                                                             q.incumbent_recipe,
                                                             q.weight_role,
                                                             proof.actual.rows,
                                                             proof.actual.k,
                                                             proof.actual.n,
                                                             proof.observed.active_stage,
                                                             proof.observed.weight.has_value(),
                                                             0})
        HashInteger(hash, scalar);
      hash.Update(std::as_bytes(std::span(q.evidence)));
      hash.Update(std::as_bytes(std::span(q.source)));
      hash.Update(std::as_bytes(std::span(q.semantic)));
      const auto& observed = proof.observed;
      hash.Update(std::as_bytes(std::span(observed.bound_model)));
      hash.Update(std::as_bytes(std::span(observed.model_spec)));
      if (observed.weight) {
        const auto& w = *observed.weight;
        for (const auto scalar :
             std::array<std::uint64_t, 5>{w.stage, w.component, w.role, w.layer, w.expert_array})
          HashInteger(hash, scalar);
      }
      const auto tensor = [&](const format::TensorFormat& f,
                              const std::array<std::uint64_t, 4>& strides) {
        for (const auto scalar : std::array<std::uint64_t, 13>{
                 static_cast<unsigned>(f.encoding), static_cast<unsigned>(f.layout), f.stacked,
                 f.shape.rank, f.params.k_bits, static_cast<unsigned>(f.params.codebook),
                 f.params.tensor_scale, f.params.bits, f.params.group,
                 static_cast<unsigned>(f.params.scale), f.params.zero_point, f.params.block_rows,
                 f.params.block_cols})
          HashInteger(hash, scalar);
        for (const auto dimension : f.shape.ne) HashInteger(hash, dimension);
        for (const auto stride : strides) HashInteger(hash, stride);
      };
      tensor(observed.operand_format, observed.operand_strides);
      tensor(observed.leaf_format, observed.leaf_strides);
      tensor(observed.input_format, observed.input_strides);
    }
  }
  return hash.Finish();
}

base::Sha256Digest ExactSiteIdentity(const ComparisonSite& site) {
  base::Sha256 hash;
  hash.Update("llmp.comparison.exact-site.v1\n");
  HashInteger(hash, static_cast<std::uint8_t>(site.whole_checkpoint.whole_only()));
  hash.Update(std::as_bytes(std::span(site.whole_checkpoint.subject())));
  hash.Update(std::as_bytes(std::span(site.selector.checkpoint_match)));
  HashInteger(hash, site.selector.source_graph_identity);
  HashInteger(hash, static_cast<std::uint8_t>(site.selector.kind));
  HashInteger(hash, site.selector.version);
  HashInteger(hash, static_cast<std::uint8_t>(site.registered_result.op));
  HashInteger(hash, static_cast<std::uint8_t>(site.registered_result.fused));
  hash.Update(site.selector.pattern);
  HashInteger(hash, static_cast<std::uint64_t>(site.accepted_member_origins.size()));
  for (const auto& member : site.accepted_member_origins) {
    HashInteger(hash, static_cast<std::uint64_t>(member.size()));
    for (const auto origin : member) HashInteger(hash, origin.v);
  }
  HashInteger(hash, static_cast<std::uint64_t>(site.accepted_member_owner_segments.size()));
  for (const auto owner : site.accepted_member_owner_segments) HashInteger(hash, owner);
  HashInteger(hash, static_cast<std::uint64_t>(site.accepted_boundary_ports.size()));
  for (const auto& port : site.accepted_boundary_ports) {
    HashInteger(hash, port.member);
    HashInteger(hash, static_cast<std::uint8_t>(port.kind));
    HashInteger(hash, port.ordinal);
    HashInteger(hash, port.original_tensor.v);
    HashInteger(hash, port.original_root.v);
    HashInteger(hash, port.view_offset);
  }
  HashInteger(hash, static_cast<std::uint64_t>(site.complete_member_owner_segments.size()));
  for (const auto owner : site.complete_member_owner_segments) HashInteger(hash, owner);
  HashInteger(hash, static_cast<std::uint64_t>(site.complete_boundary_ports.size()));
  for (const auto& port : site.complete_boundary_ports) {
    HashInteger(hash, port.member);
    HashInteger(hash, static_cast<std::uint8_t>(port.kind));
    HashInteger(hash, port.ordinal);
    HashInteger(hash, port.original_tensor.v);
    HashInteger(hash, port.original_root.v);
    HashInteger(hash, port.view_offset);
  }
  return hash.Finish();
}

void HashText(base::Sha256& hash, std::string_view text) {
  HashInteger(hash, static_cast<std::uint64_t>(text.size()));
  hash.Update(text);
}

base::Sha256Digest DeviceFactsDigest(const DeviceFacts& device) {
  base::Sha256 hash;
  hash.Update("llmp.comparison.device-facts.v1\n");
  HashText(hash, device.device_class);
  HashInteger(hash, device.arch);
  HashInteger(hash, device.features);
  HashText(hash, device.device_model);
  HashInteger(hash, device.multiprocessors);
  HashInteger(hash, static_cast<std::uint64_t>(device.tuned.size()));
  for (const auto& tuned : device.tuned) {
    HashText(hash, tuned.implementation);
    HashInteger(hash, tuned.rows);
    HashInteger(hash, tuned.k);
    HashInteger(hash, tuned.n);
    HashInteger(hash, static_cast<std::uint8_t>(tuned.output_encoding.has_value()));
    if (tuned.output_encoding) HashInteger(hash, static_cast<std::uint8_t>(*tuned.output_encoding));
    HashInteger(hash, tuned.weight_bits);
    HashInteger(hash, static_cast<std::uint8_t>(tuned.all_rows));
  }
  HashInteger(hash, static_cast<std::uint64_t>(device.preferred.size()));
  for (const auto& preferred : device.preferred) HashText(hash, preferred);
  return hash.Finish();
}

base::Sha256Digest ComparisonShapeDigest(const StepShape* shape) {
  base::Sha256 hash;
  hash.Update("llmp.comparison.step-shape.v1\n");
  HashInteger(hash, static_cast<std::uint8_t>(shape != nullptr));
  if (shape == nullptr) return hash.Finish();
  HashInteger(hash, shape->stage);
  HashInteger(hash, static_cast<std::uint8_t>(shape->phase));
  HashInteger(hash, static_cast<std::uint8_t>(shape->row_invariant));
  HashInteger(hash, shape->read_align);
  HashInteger(hash, static_cast<std::uint64_t>(shape->segments.size()));
  for (const auto& segment : shape->segments) {
    HashInteger(hash, segment.slot);
    HashInteger(hash, segment.rows);
    HashInteger(hash, segment.past);
    HashInteger(hash, segment.read_cells);
    HashInteger(hash, segment.outputs);
    HashInteger(hash, segment.group);
    HashInteger(hash, static_cast<std::uint8_t>(segment.want_head));
    HashInteger(hash, static_cast<std::uint8_t>(segment.greedy));
    HashInteger(hash, static_cast<std::uint8_t>(segment.origin_phase.has_value()));
    if (segment.origin_phase) HashInteger(hash, static_cast<std::uint8_t>(*segment.origin_phase));
  }
  return hash.Finish();
}

base::Sha256Digest ComparisonOptionsDigest(const PlanOptions& options) {
  base::Sha256 hash;
  hash.Update("llmp.comparison.plan-options.v1\n");
  HashInteger(hash, static_cast<std::uint8_t>(options.fusion_comparisons));
  HashInteger(hash, static_cast<std::uint8_t>(options.row_invariant));
  HashInteger(hash, static_cast<std::uint8_t>(options.capturable));
  HashInteger(hash, options.alignment);
  HashInteger(hash, options.activation_alignment);
  HashInteger(hash, static_cast<std::uint8_t>(options.staged.has_value()));
  if (options.staged) {
    HashInteger(hash, options.staged->size());
    for (const auto value : *options.staged) HashInteger(hash, value.v);
  }
  HashInteger(hash, options.lanes.size());
  for (const auto& [node, lane] : options.lanes) {
    HashInteger(hash, node.v);
    HashInteger(hash, lane.lane);
    HashInteger(hash, lane.region);
  }
  HashInteger(hash, static_cast<std::uint8_t>(options.alignment_staged.has_value()));
  if (options.alignment_staged) {
    HashInteger(hash, options.alignment_staged->size());
    for (const auto value : *options.alignment_staged) HashInteger(hash, value.v);
  }
  HashInteger(hash, options.invariant.size());
  for (const auto value : options.invariant) HashInteger(hash, value.v);
  // The noncopyable transaction environment binds immutable callback targets
  // and captures by object identity and synchronous lifetime.
  return hash.Finish();
}

std::optional<std::uint64_t> DecisionRecordBytes(const ComparisonSite& site) {
  constexpr std::uint64_t envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  std::uint64_t bytes = sizeof(ComparisonDecisionRecord);
  const auto add_vector = [&](std::uint64_t count, std::uint64_t size) {
    if (!AddBytes(count, size, bytes)) return false;
    return count == 0 || AddBytes(1, envelope, bytes);
  };
  if (!add_vector(site.selector.member_origins.size(), sizeof(ComparisonOriginSet)))
    return std::nullopt;
  for (const auto& origins : site.selector.member_origins)
    if (!add_vector(origins.size(), sizeof(ir::NodeId))) return std::nullopt;
  if (!add_vector(site.selector.member_owner_segments.size(), sizeof(std::uint8_t)) ||
      !add_vector(site.selector.boundary_ports.size(), sizeof(ComparisonSitePortRef)) ||
      !add_vector(site.complete_member_owner_segments.size(), sizeof(std::uint8_t)) ||
      !add_vector(site.complete_boundary_ports.size(), sizeof(ComparisonSitePortRef)))
    return std::nullopt;
  return bytes;
}

std::expected<ComparisonDecisionRecord, PlanFailure> CopyDecisionRecord(
    const ComparisonSite& site, const std::shared_ptr<const ir::StorageFunder>& funder,
    std::string_view registered_pattern, ir::DiagnosticAdmission* diagnostic) {
  const auto bytes = DecisionRecordBytes(site);
  if (!bytes)
    return LedgerFailure(PlanError::kAdmission, "comparison decision record overflows", diagnostic);
  auto loan = ir::AcquireStorageLoan(funder, *bytes);
  if (!loan)
    return LedgerFailure(PlanError::kAdmission, "comparison decision record payment was refused",
                         diagnostic);
  ComparisonDecisionRecord record;
  record.metadata_loan = std::move(*loan);
  record.kind = site.selector.kind;
  record.pattern = registered_pattern;
  record.version = site.selector.version;
  record.registered_result = site.registered_result;
  record.site_identity = ExactSiteIdentity(site);
  record.member_origins.reserve(site.selector.member_origins.size());
  for (const auto& member : site.selector.member_origins) {
    record.member_origins.emplace_back();
    record.member_origins.back().reserve(member.size());
    record.member_origins.back().insert(record.member_origins.back().end(), member.begin(),
                                        member.end());
  }
  record.member_owner_segments.reserve(site.selector.member_owner_segments.size());
  record.member_owner_segments.insert(record.member_owner_segments.end(),
                                      site.selector.member_owner_segments.begin(),
                                      site.selector.member_owner_segments.end());
  record.boundary_ports.reserve(site.selector.boundary_ports.size());
  record.boundary_ports.insert(record.boundary_ports.end(), site.selector.boundary_ports.begin(),
                               site.selector.boundary_ports.end());
  record.complete_member_owner_segments.reserve(site.complete_member_owner_segments.size());
  record.complete_member_owner_segments.insert(record.complete_member_owner_segments.end(),
                                               site.complete_member_owner_segments.begin(),
                                               site.complete_member_owner_segments.end());
  record.complete_boundary_ports.reserve(site.complete_boundary_ports.size());
  record.complete_boundary_ports.insert(record.complete_boundary_ports.end(),
                                        site.complete_boundary_ports.begin(),
                                        site.complete_boundary_ports.end());
  return record;
}

bool InSet(std::span<const ir::NodeId> nodes, ir::NodeId node) {
  return std::ranges::find(nodes, node) != nodes.end();
}

// State effects are observable even without an SSA consumer or declared result.
bool ObservedSiteOutput(const ir::Graph& graph, std::span<const ir::NodeId> members,
                        ir::TensorId tensor) {
  if (std::ranges::find(graph.outputs(), tensor) != graph.outputs().end() ||
      std::ranges::find(graph.keep(), tensor) != graph.keep().end() ||
      graph.tensor(tensor).role == ir::Role::kState ||
      std::holds_alternative<ir::StateRef>(graph.tensor(tensor).binding))
    return true;
  for (std::uint32_t consumer = 0; consumer < graph.nodes().size(); ++consumer) {
    if (InSet(members, ir::NodeId{consumer})) continue;
    const auto& candidate = graph.node(ir::NodeId{consumer});
    if (std::ranges::find(candidate.sources(), tensor) != candidate.sources().end()) return true;
  }
  return false;
}

std::optional<ir::TensorId> UniqueSourceForCheckpointTensor(const ir::Graph& source,
                                                            const ComparisonCheckpoint& checkpoint,
                                                            ir::TensorId tensor) {
  if (!tensor.valid() || tensor.v >= checkpoint.graph.tensor_count() ||
      checkpoint.source_to_checkpoint().size() != source.tensor_count())
    return std::nullopt;
  ir::TensorId found{};
  for (std::uint32_t i = 0; i < checkpoint.source_to_checkpoint().size(); ++i) {
    if (checkpoint.source_to_checkpoint()[i] != tensor) continue;
    if (found.valid()) return std::nullopt;
    found = ir::TensorId{i};
  }
  if (!found.valid()) return std::nullopt;
  const auto root = source.root(found);
  if (!root.valid() || root.v >= checkpoint.source_to_checkpoint().size() ||
      checkpoint.source_to_checkpoint()[root.v] != checkpoint.graph.root(tensor) ||
      source.tensor(found).view_offset != checkpoint.graph.tensor(tensor).view_offset)
    return std::nullopt;
  return found;
}

std::expected<void, PlanFailure> DeriveCompleteSiteIdentity(const ir::Graph& source,
                                                            ComparisonCheckpoint& checkpoint,
                                                            ComparisonSite& site,
                                                            ir::StorageLoan& site_loan,
                                                            ir::DiagnosticAdmission* diagnostic) {
  const auto fail = [diagnostic](PlanError error, const char* detail) {
    return Failure(error, detail, diagnostic);
  };
  if (site.selector.member_origins.empty() ||
      checkpoint.checkpoint_node_origins().size() != checkpoint.graph.nodes().size())
    return fail(PlanError::kStaleComparisonSite,
                "the complete comparison region is not a captured site");

  std::uint64_t max_ports = 0;
  std::uint64_t scratch_bytes = 0;
  if (!AddBytes(site.selector.member_origins.size(), sizeof(ir::NodeId), scratch_bytes))
    return fail(PlanError::kAdmission, "comparison member-map inventory overflows");
  auto scratch_loan = ir::AcquireStorageLoan(site_loan.funder(), scratch_bytes);
  if (!scratch_loan)
    return fail(PlanError::kAdmission, "comparison member-map admission was refused");
  // Paid scratch must be destroyed before its loan is returned.
  std::vector<ir::NodeId> matched_nodes;
  matched_nodes.reserve(site.selector.member_origins.size());
  for (const auto& expected : site.selector.member_origins) {
    ir::NodeId found{};
    if (HasCheckpointMatch(site.selector)) {
      if (matched_nodes.size() >= checkpoint.captured_members().size())
        return fail(PlanError::kStaleComparisonSite, "whole captured member list is incomplete");
      found = checkpoint.captured_members()[matched_nodes.size()];
      if (!found.valid() || found.v >= checkpoint.graph.nodes().size() ||
          !checkpoint.origins.Equals(checkpoint.checkpoint_node_origins()[found.v], expected))
        return fail(PlanError::kStaleComparisonSite, "whole captured member lineage changed");
    }
    for (std::uint32_t n = 0; n < checkpoint.graph.nodes().size(); ++n) {
      if (HasCheckpointMatch(site.selector)) break;
      if (!checkpoint.origins.Equals(checkpoint.checkpoint_node_origins()[n], expected)) continue;
      if (found.valid())
        return fail(PlanError::kAmbiguousComparisonSite,
                    "a captured member has more than one exact checkpoint node");
      found = ir::NodeId{n};
    }
    if (!found.valid())
      return fail(PlanError::kStaleComparisonSite,
                  "a captured member has no exact checkpoint node");
    const auto& node = checkpoint.graph.node(found);
    const std::uint64_t node_ports = static_cast<std::uint64_t>(node.nsrc) + node.nout;
    if (node_ports > std::numeric_limits<std::uint64_t>::max() - max_ports)
      return fail(PlanError::kAdmission, "comparison boundary port inventory overflows");
    max_ports += node_ports;
    matched_nodes.push_back(found);
  }

  std::uint64_t retained_bytes = 0;
  if (!AddBytes(matched_nodes.size(), sizeof(std::uint8_t), retained_bytes) ||
      !AddBytes(max_ports, sizeof(ComparisonSitePortRef), retained_bytes) ||
      !site_loan.Grow(retained_bytes))
    return fail(PlanError::kAdmission, "complete comparison site payment was refused");
  site.complete_member_owner_segments.reserve(matched_nodes.size());
  site.complete_boundary_ports.reserve(static_cast<std::size_t>(max_ports));

  for (const auto member : matched_nodes) {
    const auto& node = checkpoint.graph.node(member);
    const auto segment =
        node.nout == 0 ? ir::kJoinedSegment : checkpoint.graph.tensor(node.outputs()[0]).segment;
    for (const auto origin :
         site.selector.member_origins[site.complete_member_owner_segments.size()]) {
      if (!origin.valid() || origin.v >= source.nodes().size())
        return fail(PlanError::kStaleComparisonSite, "a site owner origin is out of range");
      const auto& source_node = source.node(origin);
      const auto source_segment = source_node.nout == 0
                                      ? ir::kJoinedSegment
                                      : source.tensor(source_node.outputs()[0]).segment;
      // Owner facts describe this checkpoint member, not every ancestor of
      // a prior rewrite. Its exact current owner is replayed with its ports.
      if (site.selector.member_origins[site.complete_member_owner_segments.size()].size() == 1 &&
          source_segment != segment)
        return fail(PlanError::kStaleComparisonSite,
                    "a captured member has inconsistent owner segments");
    }
    site.complete_member_owner_segments.push_back(segment);
  }

  const auto append_port = [&](std::size_t member_index, ComparisonSitePortKind kind,
                               std::size_t ordinal,
                               ir::TensorId checkpoint_tensor) -> std::expected<void, PlanFailure> {
    if (ordinal > std::numeric_limits<std::uint16_t>::max())
      return fail(PlanError::kStaleComparisonSite,
                  "a comparison boundary port ordinal is not representable");
    const auto original =
        HasCheckpointMatch(site.selector)
            ? std::optional(checkpoint_tensor)
            : UniqueSourceForCheckpointTensor(source, checkpoint, checkpoint_tensor);
    if (!original)
      return fail(PlanError::kStaleComparisonSite,
                  "a checkpoint boundary tensor has no unique source mapping");
    site.complete_boundary_ports.push_back(ComparisonSitePortRef{
        .member = static_cast<std::uint32_t>(member_index),
        .kind = kind,
        .ordinal = static_cast<std::uint16_t>(ordinal),
        .original_tensor = *original,
        .original_root = HasCheckpointMatch(site.selector) ? checkpoint.graph.root(*original)
                                                           : source.root(*original),
        .view_offset = HasCheckpointMatch(site.selector)
                           ? checkpoint.graph.tensor(*original).view_offset
                           : source.tensor(*original).view_offset});
    return {};
  };

  // Input boundaries first, in member/source-port order.
  for (std::size_t member_index = 0; member_index < matched_nodes.size(); ++member_index) {
    const auto& node = checkpoint.graph.node(matched_nodes[member_index]);
    for (std::size_t ordinal = 0; ordinal < node.nsrc; ++ordinal) {
      const auto tensor = node.source(ordinal);
      if (!tensor.valid()) continue;  // absent optional GGML/IR source slot
      if (InSet(matched_nodes, checkpoint.graph.producer(tensor))) continue;
      auto added = append_port(member_index, ComparisonSitePortKind::kInput, ordinal, tensor);
      if (!added) return std::unexpected(std::move(added.error()));
    }
  }
  // Outputs crossing the captured region, including declared graph results.
  for (std::size_t member_index = 0; member_index < matched_nodes.size(); ++member_index) {
    const auto& node = checkpoint.graph.node(matched_nodes[member_index]);
    for (std::size_t ordinal = 0; ordinal < node.nout; ++ordinal) {
      const auto tensor = node.outputs()[ordinal];
      if (!ObservedSiteOutput(checkpoint.graph, matched_nodes, tensor)) continue;
      auto added = append_port(member_index, ComparisonSitePortKind::kOutput, ordinal, tensor);
      if (!added) return std::unexpected(std::move(added.error()));
    }
  }
  const bool has_input = std::ranges::any_of(site.complete_boundary_ports, [](const auto& port) {
    return port.kind == ComparisonSitePortKind::kInput;
  });
  const bool has_output = std::ranges::any_of(site.complete_boundary_ports, [](const auto& port) {
    return port.kind == ComparisonSitePortKind::kOutput;
  });
  if ((site.selector.kind == ComparisonSiteKind::kRewrite && (!has_input || !has_output)) ||
      (site.selector.kind == ComparisonSiteKind::kBatch && site.complete_boundary_ports.empty()))
    return fail(PlanError::kStaleComparisonSite,
                "the captured site has no complete boundary ports");
  if (HasCheckpointMatch(site.selector)) {
    if (!checkpoint.history_matches_graph() ||
        CheckpointMatchIdentity(checkpoint.graph, matched_nodes) != site.selector.checkpoint_match)
      return fail(PlanError::kStaleComparisonSite, "whole checkpoint match changed after capture");
    site.whole_checkpoint =
        ComparisonCheckpointAuthorityAccess::Mint(source.identity(), checkpoint.graph.identity(),
                                                  WholeCheckpointSubject(source, checkpoint, site));
  }
  site.exact_site_identity = ExactSiteIdentity(site);
  return {};
}

std::expected<void, PlanFailure> DeriveCompleteSourceNodeIdentity(
    const ir::Graph& source, ir::NodeId source_node_id, ComparisonSite& site,
    ir::StorageLoan& site_loan, ir::DiagnosticAdmission* diagnostic) {
  const auto fail = [diagnostic](PlanError error, const char* detail) {
    return Failure(error, detail, diagnostic);
  };
  if ((site.selector.kind != ComparisonSiteKind::kFusedNode &&
       site.selector.kind != ComparisonSiteKind::kCompositeNode) ||
      site.selector.member_origins.size() != 1 || site.selector.member_origins[0].size() != 1 ||
      site.selector.member_origins[0][0] != source_node_id || !source_node_id.valid() ||
      source_node_id.v >= source.nodes().size())
    return fail(PlanError::kStaleComparisonSite,
                "the source-node site does not name one authenticated graph node");

  const auto& node = source.node(source_node_id);
  const auto segment =
      node.nout == 0 ? ir::kJoinedSegment : source.tensor(node.outputs()[0]).segment;
  if (!site.selector.member_owner_segments.empty() &&
      (site.selector.member_owner_segments.size() != 1 ||
       site.selector.member_owner_segments[0] != segment))
    return fail(PlanError::kStaleComparisonSite,
                "the source-node owner differs from the selected site");

  std::uint64_t port_count = 0;
  for (const auto tensor : node.sources())
    if (tensor.valid()) ++port_count;
  for (std::size_t ordinal = 0; ordinal < node.nout; ++ordinal) {
    const auto tensor = node.outputs()[ordinal];
    bool outside = std::ranges::contains(source.outputs(), tensor) ||
                   std::ranges::contains(source.keep(), tensor);
    for (std::uint32_t consumer = 0; !outside && consumer < source.nodes().size(); ++consumer) {
      if (consumer == source_node_id.v) continue;
      outside = std::ranges::contains(source.node(ir::NodeId{consumer}).sources(), tensor);
    }
    if (outside) ++port_count;
  }
  std::uint64_t added = 0;
  if (!AddBytes(1, sizeof(std::uint8_t), added) ||
      !AddBytes(port_count, sizeof(ComparisonSitePortRef), added) || !site_loan.Grow(added))
    return fail(PlanError::kAdmission, "complete source-node site payment was refused");

  site.complete_member_owner_segments.reserve(1);
  site.complete_boundary_ports.reserve(static_cast<std::size_t>(port_count));
  site.complete_member_owner_segments.push_back(segment);
  const auto add_port = [&](ComparisonSitePortKind kind, std::size_t ordinal,
                            ir::TensorId tensor) -> std::expected<void, PlanFailure> {
    if (ordinal > std::numeric_limits<std::uint16_t>::max() || !tensor.valid() ||
        tensor.v >= source.tensor_count())
      return fail(PlanError::kStaleComparisonSite,
                  "a source-node boundary port is not representable");
    site.complete_boundary_ports.push_back(
        ComparisonSitePortRef{.member = 0,
                              .kind = kind,
                              .ordinal = static_cast<std::uint16_t>(ordinal),
                              .original_tensor = tensor,
                              .original_root = source.root(tensor),
                              .view_offset = source.tensor(tensor).view_offset});
    return {};
  };
  for (std::size_t ordinal = 0; ordinal < node.nsrc; ++ordinal) {
    const auto tensor = node.source(ordinal);
    if (!tensor.valid()) continue;
    auto added_port = add_port(ComparisonSitePortKind::kInput, ordinal, tensor);
    if (!added_port) return added_port;
  }
  for (std::size_t ordinal = 0; ordinal < node.nout; ++ordinal) {
    const auto tensor = node.outputs()[ordinal];
    bool outside = std::ranges::contains(source.outputs(), tensor) ||
                   std::ranges::contains(source.keep(), tensor);
    for (std::uint32_t consumer = 0; !outside && consumer < source.nodes().size(); ++consumer) {
      if (consumer == source_node_id.v) continue;
      outside = std::ranges::contains(source.node(ir::NodeId{consumer}).sources(), tensor);
    }
    if (!outside) continue;
    auto added_port = add_port(ComparisonSitePortKind::kOutput, ordinal, tensor);
    if (!added_port) return added_port;
  }
  if (site.complete_boundary_ports.empty())
    return fail(PlanError::kStaleComparisonSite,
                "the source-node site has no externally visible boundary port");
  site.exact_site_identity = ExactSiteIdentity(site);
  return {};
}

bool ValidPortRefs(const ir::Graph& graph, std::span<const ir::NodeId> members,
                   std::span<const ComparisonSitePortRef> ports) {
  for (const auto& port : ports) {
    if (port.member >= members.size()) return false;
    const auto node_id = members[port.member];
    if (!node_id.valid() || node_id.v >= graph.nodes().size()) return false;
    const auto& node = graph.node(node_id);
    ir::TensorId tensor{};
    if (port.kind == ComparisonSitePortKind::kInput) {
      if (port.ordinal >= node.nsrc) return false;
      tensor = node.source(port.ordinal);
    } else {
      if (port.ordinal >= node.nout) return false;
      tensor = node.outputs()[port.ordinal];
    }
    if (tensor != port.original_tensor || !tensor.valid() || tensor.v >= graph.tensor_count())
      return false;
    const auto& desc = graph.tensor(tensor);
    if (graph.root(tensor) != port.original_root || desc.view_offset != port.view_offset)
      return false;
  }
  return true;
}

bool ValidOriginPortRefs(const ir::Graph& graph, std::span<const ComparisonOriginSet> members,
                         std::span<const ComparisonSitePortRef> ports) {
  for (const auto& port : ports) {
    if (port.member >= members.size() || !port.original_tensor.valid() ||
        port.original_tensor.v >= graph.tensor_count() || !port.original_root.valid() ||
        port.original_root.v >= graph.tensor_count())
      return false;
    if (graph.root(port.original_tensor) != port.original_root ||
        graph.tensor(port.original_tensor).view_offset != port.view_offset)
      return false;
    for (const auto origin : members[port.member])
      if (!origin.valid() || origin.v >= graph.nodes().size()) return false;
    // A grouped member's ports belong to the current fused node. A rewrite
    // may expose upstream inputs through shared layout nodes outside its
    // original lineage. Exact incidence and ordinals are authenticated by
    // SitePortsMatchCurrent when the checkpoint is captured.
    if (members[port.member].size() != 1) continue;
    const auto& node = graph.node(members[port.member].front());
    const auto tensors =
        port.kind == ComparisonSitePortKind::kInput ? node.sources() : node.outputs();
    if (port.ordinal >= tensors.size() || tensors[port.ordinal] != port.original_tensor)
      return false;
  }
  return true;
}

ir::TensorId MappedComparisonTensor(ir::TensorId source, std::span<const ir::TensorId> current,
                                    std::span<const ir::TensorId> source_to_checkpoint = {}) {
  if (!source_to_checkpoint.empty()) {
    if (!source.valid() || source.v >= source_to_checkpoint.size()) return {};
    source = source_to_checkpoint[source.v];
  }
  return source.valid() && source.v < current.size() ? current[source.v] : ir::TensorId{};
}

bool SitePortsMatchCurrent(std::span<const std::uint8_t> expected_owner_segments,
                           std::span<const ComparisonSitePortRef> ports, const ir::Graph& graph,
                           std::span<const ir::NodeId> members,
                           std::span<const ir::TensorId> source_to_graph,
                           bool require_complete_boundary = false,
                           std::span<const ir::TensorId> source_to_checkpoint = {}) {
  if ((!expected_owner_segments.empty() && expected_owner_segments.size() != members.size()))
    return false;
  for (const auto node_id : members)
    if (!node_id.valid() || node_id.v >= graph.nodes().size()) return false;
  for (std::size_t i = 0; i < expected_owner_segments.size(); ++i) {
    const auto node_id = members[i];
    if (!node_id.valid() || node_id.v >= graph.nodes().size()) return false;
    const auto& node = graph.node(node_id);
    const std::uint8_t segment =
        node.nout == 0 ? ir::kJoinedSegment : graph.tensor(node.out[0]).segment;
    if (segment != expected_owner_segments[i]) return false;
  }
  if (!require_complete_boundary) {
    for (const auto& port : ports) {
      if (port.member >= members.size()) return false;
      const auto tensor =
          MappedComparisonTensor(port.original_tensor, source_to_graph, source_to_checkpoint);
      const auto root =
          MappedComparisonTensor(port.original_root, source_to_graph, source_to_checkpoint);
      if (!tensor.valid() || tensor.v >= graph.tensor_count() || !root.valid() ||
          root.v >= graph.tensor_count() || graph.root(tensor) != root ||
          graph.tensor(tensor).view_offset != port.view_offset)
        return false;
      const auto& node = graph.node(members[port.member]);
      const ir::TensorId actual =
          port.kind == ComparisonSitePortKind::kInput
              ? (port.ordinal < node.nsrc ? node.source(port.ordinal) : ir::TensorId{})
              : (port.ordinal < node.nout ? node.out[port.ordinal] : ir::TensorId{});
      if (actual != tensor) return false;
    }
    return true;
  }
  std::size_t expected_index = 0;
  const auto matches = [&](std::size_t member, ComparisonSitePortKind kind, std::size_t ordinal,
                           ir::TensorId actual) {
    if (expected_index >= ports.size() || ordinal > std::numeric_limits<std::uint16_t>::max())
      return false;
    const auto& port = ports[expected_index++];
    if (port.member != member || port.kind != kind || port.ordinal != ordinal) return false;
    const auto mapped_tensor =
        MappedComparisonTensor(port.original_tensor, source_to_graph, source_to_checkpoint);
    const auto mapped_root =
        MappedComparisonTensor(port.original_root, source_to_graph, source_to_checkpoint);
    return mapped_tensor.valid() && mapped_tensor.v < graph.tensor_count() && mapped_root.valid() &&
           mapped_root.v < graph.tensor_count() && actual == mapped_tensor &&
           graph.root(mapped_tensor) == mapped_root &&
           graph.tensor(mapped_tensor).view_offset == port.view_offset;
  };

  // Re-enumerate the complete input boundary in the same member/source order
  // used by DeriveCompleteSiteIdentity. This detects both missing and added
  // boundary ports, not only mutation of the ports captured earlier.
  for (std::size_t member_index = 0; member_index < members.size(); ++member_index) {
    const auto& node = graph.node(members[member_index]);
    for (std::size_t ordinal = 0; ordinal < node.nsrc; ++ordinal) {
      const auto tensor = node.source(ordinal);
      if (!tensor.valid() || InSet(members, graph.producer(tensor))) continue;
      if (!matches(member_index, ComparisonSitePortKind::kInput, ordinal, tensor)) return false;
    }
  }
  // Then enumerate observed outputs, including graph outputs, kept tensors,
  // and values with consumers outside the exact matched region.
  for (std::size_t member_index = 0; member_index < members.size(); ++member_index) {
    const auto& node = graph.node(members[member_index]);
    for (std::size_t ordinal = 0; ordinal < node.nout; ++ordinal) {
      const auto tensor = node.out[ordinal];
      if (ObservedSiteOutput(graph, members, tensor) &&
          !matches(member_index, ComparisonSitePortKind::kOutput, ordinal, tensor))
        return false;
    }
  }
  return expected_index == ports.size();
}

bool SitePortsMatchCurrent(const ComparisonSiteSelector& selector, const ir::Graph& graph,
                           std::span<const ir::NodeId> members,
                           std::span<const ir::TensorId> source_to_graph) {
  return SitePortsMatchCurrent(selector.member_owner_segments, selector.boundary_ports, graph,
                               members, source_to_graph);
}

bool FusedRegionMatchesCurrent(ComparisonSiteKind kind,
                               std::span<const ComparisonOriginSet> member_origins,
                               std::span<const std::uint8_t> owner_segments,
                               std::span<const ComparisonSitePortRef> boundary_ports,
                               std::span<const ir::NodeId> matched_nodes,
                               std::span<const ComparisonOriginSetId> node_origins,
                               ComparisonOriginDag& origins, const ir::Graph& graph,
                               std::span<const ir::TensorId> source_to_graph,
                               std::span<const ir::TensorId> source_to_checkpoint = {}) {
  if ((kind != ComparisonSiteKind::kFusedNode && kind != ComparisonSiteKind::kCompositeNode) ||
      member_origins.size() != 1 || member_origins.front().empty() || boundary_ports.empty() ||
      owner_segments.size() != 1 || matched_nodes.empty())
    return false;
  for (const auto node_id : matched_nodes) {
    if (!node_id.valid() || node_id.v >= graph.nodes().size() || node_id.v >= node_origins.size())
      return false;
    const auto segment = graph.node(node_id).nout == 0
                             ? ir::kJoinedSegment
                             : graph.tensor(graph.node(node_id).outputs()[0]).segment;
    if (segment != owner_segments[0]) return false;
    for (const auto origin : member_origins.front())
      if (!origins.Contains(node_origins[node_id.v], origin)) return false;
  }
  if (!origins.EqualsUnionOfNodes(matched_nodes, node_origins, member_origins.front()))
    return false;

  std::size_t expected = 0;
  const auto match_port = [&](ComparisonSitePortKind kind, ir::TensorId actual) {
    if (expected >= boundary_ports.size()) return false;
    const auto& port = boundary_ports[expected++];
    if (port.kind != kind) return false;
    const auto tensor =
        MappedComparisonTensor(port.original_tensor, source_to_graph, source_to_checkpoint);
    const auto root =
        MappedComparisonTensor(port.original_root, source_to_graph, source_to_checkpoint);
    return tensor.valid() && tensor.v < graph.tensor_count() && root.valid() &&
           root.v < graph.tensor_count() && actual == tensor && graph.root(tensor) == root &&
           graph.tensor(tensor).view_offset == port.view_offset;
  };
  for (const auto member : matched_nodes) {
    const auto& node = graph.node(member);
    for (std::size_t ordinal = 0; ordinal < node.nsrc; ++ordinal) {
      const auto tensor = node.source(ordinal);
      if (tensor.valid() && !InSet(matched_nodes, graph.producer(tensor)) &&
          !match_port(ComparisonSitePortKind::kInput, tensor))
        return false;
    }
  }
  for (const auto member : matched_nodes) {
    const auto& node = graph.node(member);
    for (std::size_t ordinal = 0; ordinal < node.nout; ++ordinal) {
      const auto tensor = node.outputs()[ordinal];
      bool outside = std::ranges::find(graph.outputs(), tensor) != graph.outputs().end() ||
                     std::ranges::find(graph.keep(), tensor) != graph.keep().end();
      for (std::uint32_t consumer = 0; !outside && consumer < graph.nodes().size(); ++consumer) {
        if (InSet(matched_nodes, ir::NodeId{consumer})) continue;
        const auto& candidate = graph.node(ir::NodeId{consumer});
        outside = std::ranges::find(candidate.sources(), tensor) != candidate.sources().end();
      }
      if (outside && !match_port(ComparisonSitePortKind::kOutput, tensor)) return false;
    }
  }
  return expected == boundary_ports.size();
}

bool DecisionRegionMatchesCurrent(const ComparisonDecisionRecord& decision,
                                  std::span<const ir::NodeId> matched_nodes,
                                  std::span<const ComparisonOriginSetId> node_origins,
                                  ComparisonOriginDag& origins, const ir::Graph& graph,
                                  std::span<const ir::TensorId> source_to_graph,
                                  std::span<const ir::TensorId> source_to_checkpoint = {}) {
  return FusedRegionMatchesCurrent(decision.kind, decision.member_origins,
                                   decision.complete_member_owner_segments,
                                   decision.complete_boundary_ports, matched_nodes, node_origins,
                                   origins, graph, source_to_graph, source_to_checkpoint);
}

}  // namespace

std::expected<void, PlanFailure> ComparisonCaptureSession::PrepareDecisionJournal(
    std::shared_ptr<const ir::StorageFunder> funder) {
  if (decision_journal_prepared_)
    return std::unexpected(PlanFailure{.error = PlanError::kStaleComparisonSite, .detail = {}});
  decision_journal_prepared_ = true;
  if (decisions_.empty()) return {};
  std::uint64_t candidates = 0;
  for (const Pattern& candidate : Patterns())
    if (candidate.kind == PatternKind::kRewrite || candidate.kind == PatternKind::kBatch)
      ++candidates;
  if (candidates != 0 && decisions_.size() > std::numeric_limits<std::uint64_t>::max() / candidates)
    return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  const auto entries = static_cast<std::uint64_t>(decisions_.size()) * candidates;
  constexpr std::uint64_t envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  std::uint64_t bytes = 0;
  if (!AddBytes(entries, sizeof(ComparisonDecisionExclusion), bytes) ||
      (entries != 0 && envelope > std::numeric_limits<std::uint64_t>::max() - bytes))
    return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  if (entries != 0) bytes += envelope;
  if (!funder) return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  auto loan = ir::AcquireStorageLoan(std::move(funder), bytes);
  if (!loan) return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  decision_journal_loan_ = std::move(*loan);
  decision_exclusions_.reserve(static_cast<std::size_t>(entries));
  return {};
}

bool ComparisonCaptureSession::RecordDecisionExclusion(const ComparisonDecisionRecord& decision,
                                                       std::string_view pattern,
                                                       std::uint16_t version) {
  const auto prior = std::ranges::find_if(decision_exclusions_, [&](const auto& entry) {
    return entry.prior_site_identity == decision.site_identity &&
           entry.competing_pattern == pattern && entry.competing_version == version;
  });
  if (prior != decision_exclusions_.end()) return true;
  if (!decision_journal_prepared_ ||
      decision_exclusions_.size() == decision_exclusions_.capacity()) {
    record_refused_ = true;
    return false;
  }
  decision_exclusions_.push_back(
      ComparisonDecisionExclusion{.prior_site_identity = decision.site_identity,
                                  .competing_pattern = pattern,
                                  .competing_version = version});
  return true;
}

namespace {

std::optional<std::uint64_t> MappingBytes(std::uint64_t tensors, std::uint64_t nodes) {
  std::uint64_t bytes = 0;
  if (!AddBytes(tensors, sizeof(ir::TensorId), bytes) ||
      !AddBytes(nodes, sizeof(ir::NodeId), bytes) ||
      !AddBytes(nodes, sizeof(ComparisonOriginSetId), bytes))
    return std::nullopt;
  return bytes;
}

bool SelectedNodeThroughSeed(const StepPlan& plan, std::span<const ir::NodeId> seed_to_source,
                             ir::NodeId source_node, ir::OpKey key) {
  for (std::uint32_t i = 0; i < plan.graph.nodes().size(); ++i) {
    if (i >= plan.from.size() || !plan.from[i].valid() || plan.from[i].v >= seed_to_source.size() ||
        seed_to_source[plan.from[i].v] != source_node || plan.graph.node(ir::NodeId{i}).key != key)
      continue;
    for (const auto& step : plan.steps)
      if (std::ranges::contains(step.nodes, ir::NodeId{i})) return true;
  }
  return false;
}

bool SelectedOriginalThroughAlternative(const StepPlan& plan,
                                        std::span<const ir::NodeId> alternative_to_source,
                                        ir::NodeId source_node, ir::OpKey key) {
  for (std::uint32_t i = 0; i < plan.graph.nodes().size(); ++i) {
    if (i >= plan.from.size() || !plan.from[i].valid() ||
        plan.from[i].v >= alternative_to_source.size() ||
        alternative_to_source[plan.from[i].v] != source_node || plan.graph.node({i}).key != key)
      continue;
    for (const auto& step : plan.steps)
      if (std::ranges::contains(step.nodes, ir::NodeId{i})) return true;
  }
  return false;
}

bool SelectedBatch(const StepPlan& plan, const ComparisonSiteSelector& selector,
                   std::string_view pattern, ComparisonOriginDag& origins,
                   std::span<const ComparisonOriginSetId> node_origins) {
  for (const auto& member : selector.member_origins) {
    bool found = false;
    for (std::uint32_t i = 0; i < plan.graph.nodes().size(); ++i) {
      if (i >= node_origins.size() || !origins.Equals(node_origins[i], member)) continue;
      for (const auto& step : plan.steps)
        if (step.batch == pattern && std::ranges::contains(step.nodes, ir::NodeId{i})) found = true;
    }
    if (!found) return false;
  }
  return !selector.member_origins.empty();
}

bool BoundaryMatches(const components::OwnedComparisonBoundary& boundary, const ir::Graph& graph,
                     std::span<const ir::TensorId> mapping,
                     ir::DiagnosticAdmission* diagnostic = nullptr) {
  return static_cast<bool>(
      components::ValidateMappedComparisonBoundary(boundary, graph, mapping, diagnostic));
}

template <class T>
bool AddGeometricCapacity(std::uint64_t count, std::uint64_t& total) {
  constexpr std::uint64_t kEnvelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  if (count > (std::numeric_limits<std::uint64_t>::max() - total) / (3 * sizeof(T))) return false;
  total += count * 3 * sizeof(T);
  if (count != 0) {
    if (2 * kEnvelope > std::numeric_limits<std::uint64_t>::max() - total) return false;
    total += 2 * kEnvelope;
  }
  return true;
}

PlanOptions BorrowedOptions(const PlanOptions& source) {
  PlanOptions out;
  out.fusion_comparisons = source.fusion_comparisons;
  out.row_invariant = source.row_invariant;
  out.capturable = source.capturable;
  out.staged = source.staged;
  out.lanes = source.lanes;
  out.alignment = source.alignment;
  out.shape = source.shape;
  out.activation_alignment = source.activation_alignment;
  out.alignment_staged = source.alignment_staged;
  out.invariant = source.invariant;
  out.owner_products = source.owner_products;
  out.observed_bound_model = source.observed_bound_model;
  out.observed_model_spec = source.observed_model_spec;
  out.comparison_capture = source.comparison_capture;
  out.comparison_origins = source.comparison_origins;
  out.comparison_node_origins = source.comparison_node_origins;
  out.comparison_source_to_graph = source.comparison_source_to_graph;
  out.comparison_graph_node_to_source = source.comparison_graph_node_to_source;
  out.metadata_admission = source.metadata_admission;
  const PlanOptions* borrowed = &source;
  if (source.on_stream)
    out.on_stream = [borrowed](std::string_view group) { return borrowed->on_stream(group); };
  if (source.state_spans)
    out.state_spans = [borrowed](std::span<const StateCellSlice> slices) {
      return borrowed->state_spans(slices);
    };
  if (source.root_alignment)
    out.root_alignment = [borrowed](const ir::Graph& graph, ir::TensorId root) {
      return borrowed->root_alignment(graph, root);
    };
  if (source.written)
    out.written = [borrowed](const ir::Graph& graph, ir::TensorId root) {
      return borrowed->written(graph, root);
    };
  if (source.within)
    out.within = [borrowed](const ir::Graph& graph, ir::TensorId root) {
      return borrowed->within(graph, root);
    };
  return out;
}

std::expected<void, PlanFailure> MapComparisonPlanInputs(
    PlanOptions& mapped, std::span<const ir::TensorId> source_to_target,
    std::span<const ir::NodeId> target_to_source, PlannerMetadataAdmission& admission) {
  const auto map_tensor_list = [&](std::optional<std::vector<ir::TensorId>>& values) {
    if (!values) return true;
    for (auto& value : *values) {
      if (!value.valid() || value.v >= source_to_target.size() ||
          !source_to_target[value.v].valid())
        return false;
      value = source_to_target[value.v];
    }
    return true;
  };
  const auto map_invariant = [&] {
    for (auto& value : mapped.invariant) {
      if (!value.valid() || value.v >= source_to_target.size() ||
          !source_to_target[value.v].valid())
        return false;
      value = source_to_target[value.v];
    }
    return true;
  };
  if (!map_tensor_list(mapped.staged) || !map_tensor_list(mapped.alignment_staged) ||
      !map_invariant())
    return std::unexpected(PlanFailure{.error = PlanError::kStaleComparisonSite, .detail = {}});
  if (mapped.lanes.empty()) return {};
  constexpr std::uint64_t envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  const auto count = static_cast<std::uint64_t>(target_to_source.size());
  if (count > (std::numeric_limits<std::uint64_t>::max() - envelope) / sizeof(LaneTags::value_type))
    return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  const auto bytes = count * sizeof(LaneTags::value_type) + envelope;
  if (auto paid = admission.Require(PlannerMetadataPhase::kComparisonMappedInputs, bytes); !paid)
    return std::unexpected(std::move(paid.error()));
  std::vector<LaneTags::value_type> lanes;
  lanes.reserve(target_to_source.size());
  for (std::uint32_t target = 0; target < target_to_source.size(); ++target) {
    const auto source = target_to_source[target];
    if (!source.valid())
      return std::unexpected(PlanFailure{.error = PlanError::kStaleComparisonSite, .detail = {}});
    // Plan builds its lane lookup in declaration order, so duplicate source
    // tags are last-wins. Preserve that behavior when a source node fans out
    // into several seed nodes.
    bool found = false;
    LaneTag tag{};
    for (const auto& entry : mapped.lanes) {
      if (entry.first == source) {
        tag = entry.second;
        found = true;
      }
    }
    if (found) lanes.emplace_back(ir::NodeId{target}, tag);
  }
  mapped.lanes = std::move(lanes);
  return {};
}

struct ValidatedSiteNode {
  ir::NodeId node{};
  const Pattern* pattern = nullptr;
};

std::expected<ValidatedSiteNode, PlanFailure> RequiredSiteNode(
    const ir::Graph& graph, const ComparisonSiteSelector& requested,
    ir::DiagnosticAdmission* diagnostic) {
  const auto failure = [diagnostic](PlanError error, const char* detail) {
    return Failure(error, detail, diagnostic);
  };
  if ((requested.kind == ComparisonSiteKind::kFusedNode ||
       requested.kind == ComparisonSiteKind::kCompositeNode) &&
      (requested.member_origins.size() != 1 || requested.member_origins[0].size() != 1 ||
       (!requested.member_owner_segments.empty() && requested.member_owner_segments.size() != 1)))
    return failure(PlanError::kStaleComparisonSite,
                   "the node comparison selector must name one exact source node");
  if (requested.member_origins.empty() || requested.member_origins[0].empty())
    return failure(PlanError::kStaleComparisonSite, "the comparison selector has no source node");
  const ir::NodeId node_id = requested.member_origins[0][0];
  if (!node_id.valid() || node_id.v >= graph.nodes().size())
    return failure(PlanError::kStaleComparisonSite, "the requested source node is out of range");
  const auto& node = graph.node(node_id);
  const Pattern* pattern = nullptr;
  if (requested.kind == ComparisonSiteKind::kRewrite) {
    if (requested.member_origins.empty())
      return failure(PlanError::kStaleComparisonSite, "a rewrite selector has no members");
    for (const auto& member : requested.member_origins) {
      if (member.empty())
        return failure(PlanError::kStaleComparisonSite,
                       "rewrite members must have nonempty original lineage");
      for (std::size_t i = 0; i < member.size(); ++i)
        if (!member[i].valid() || member[i].v >= graph.nodes().size() ||
            (i != 0 && member[i - 1].v >= member[i].v))
          return failure(PlanError::kStaleComparisonSite,
                         "rewrite lineage must be sorted unique source node IDs");
    }
    if (!requested.member_owner_segments.empty() &&
        requested.member_owner_segments.size() != requested.member_origins.size())
      return failure(PlanError::kStaleComparisonSite,
                     "rewrite owner segments do not match its members");
    for (const auto& candidate : Patterns())
      if (candidate.kind == PatternKind::kRewrite &&
          (requested.pattern.empty() || candidate.name == requested.pattern)) {
        if (pattern != nullptr)
          return failure(PlanError::kAmbiguousComparisonSite,
                         "the rewrite selector names more than one pattern");
        pattern = &candidate;
      }
    if (pattern == nullptr || (requested.version != 0 && requested.version != pattern->version))
      return failure(PlanError::kStaleComparisonSite,
                     "the selector pattern/version is not a registered rewrite");
    if (!HasCheckpointMatch(requested) &&
        !ValidOriginPortRefs(graph, requested.member_origins, requested.boundary_ports))
      return failure(PlanError::kStaleComparisonSite,
                     "a requested site boundary port does not match its source member");
    for (std::size_t i = 0; i < requested.member_owner_segments.size(); ++i) {
      if (requested.member_origins[i].size() != 1) continue;
      const auto& member_node = graph.node(requested.member_origins[i][0]);
      const std::uint8_t segment =
          member_node.nout == 0 ? ir::kJoinedSegment : graph.tensor(member_node.out[0]).segment;
      if (segment != requested.member_owner_segments[i])
        return failure(PlanError::kStaleComparisonSite,
                       "a requested owner segment differs from its source member");
    }
    return ValidatedSiteNode{.node = requested.member_origins.front().front(), .pattern = pattern};
  } else if (requested.kind == ComparisonSiteKind::kBatch) {
    if (requested.member_origins.empty())
      return failure(PlanError::kStaleComparisonSite, "a batch selector has no members");
    for (const auto& member : requested.member_origins)
      if (member.size() != 1 || !member[0].valid() || member[0].v >= graph.nodes().size())
        return failure(PlanError::kStaleComparisonSite,
                       "batch members must name exact original source nodes");
    if (!requested.member_owner_segments.empty() &&
        requested.member_owner_segments.size() != requested.member_origins.size())
      return failure(PlanError::kStaleComparisonSite,
                     "batch owner segments do not match its members");
    for (const auto& candidate : Patterns())
      if (candidate.kind == PatternKind::kBatch &&
          (requested.pattern.empty() || candidate.name == requested.pattern)) {
        if (pattern != nullptr)
          return failure(PlanError::kAmbiguousComparisonSite,
                         "the batch selector names more than one pattern");
        pattern = &candidate;
      }
    if (pattern == nullptr || (requested.version != 0 && requested.version != pattern->version))
      return failure(PlanError::kStaleComparisonSite,
                     "the selector pattern/version is not a registered batch");
    if (!ValidOriginPortRefs(graph, requested.member_origins, requested.boundary_ports))
      return failure(PlanError::kStaleComparisonSite,
                     "a requested site boundary port does not match its batch member");
    for (std::size_t i = 0; i < requested.member_owner_segments.size(); ++i) {
      const auto& member_node = graph.node(requested.member_origins[i][0]);
      const std::uint8_t segment =
          member_node.nout == 0 ? ir::kJoinedSegment : graph.tensor(member_node.out[0]).segment;
      if (segment != requested.member_owner_segments[i])
        return failure(PlanError::kStaleComparisonSite,
                       "a requested owner segment differs from its source member");
    }
    return ValidatedSiteNode{.node = requested.member_origins.front().front(), .pattern = pattern};
  } else if (requested.kind == ComparisonSiteKind::kFusedNode) {
    if (node.key.fused == ir::FusedKind::kNone)
      return failure(PlanError::kStaleComparisonSite, "the requested node is not fused");
    for (const auto& candidate : Patterns())
      if (candidate.kind == PatternKind::kRewrite && candidate.AcceptsResult(node.key)) {
        if (pattern != nullptr)
          return failure(PlanError::kAmbiguousComparisonSite,
                         "more than one rewrite names the requested fused node");
        pattern = &candidate;
      }
  } else if (requested.kind == ComparisonSiteKind::kCompositeNode) {
    if (node.key.fused != ir::FusedKind::kNone || !ir::HasDecomposition(node.key))
      return failure(PlanError::kStaleComparisonSite, "the requested node is not composite");
    for (const auto& candidate : Patterns())
      if (candidate.kind == PatternKind::kComposite &&
          std::ranges::contains(candidate.composites, node.key)) {
        if (pattern != nullptr)
          return failure(PlanError::kAmbiguousComparisonSite,
                         "more than one composite pattern names the requested node");
        pattern = &candidate;
      }
  } else {
    return failure(PlanError::kStaleComparisonSite,
                   "this comparison entry point currently accepts node-level sites");
  }
  if (pattern == nullptr || (!requested.pattern.empty() && requested.pattern != pattern->name) ||
      (requested.version != 0 && requested.version != pattern->version))
    return failure(PlanError::kStaleComparisonSite,
                   "the selector pattern/version does not match the source node");
  const std::array members{node_id};
  if (!ValidPortRefs(graph, members, requested.boundary_ports))
    return failure(PlanError::kStaleComparisonSite,
                   "a requested site boundary port does not match the source node");
  if (!requested.member_owner_segments.empty()) {
    const std::uint8_t segment =
        node.nout == 0 ? ir::kJoinedSegment : graph.tensor(node.out[0]).segment;
    if (segment != requested.member_owner_segments[0])
      return failure(PlanError::kStaleComparisonSite,
                     "the requested owner segment differs from the source node");
  }
  return ValidatedSiteNode{.node = node_id, .pattern = pattern};
}

struct ChildRange {
  bool aliases = false;
  std::size_t offset = 0;
};

std::optional<ChildRange> ChildAlias(std::span<const ComparisonOriginSetId> input,
                                     std::span<const ComparisonOriginSetId> owned) {
  if (input.empty() || owned.empty()) return ChildRange{};
  const auto begin = reinterpret_cast<std::uintptr_t>(owned.data());
  if (owned.size_bytes() > std::numeric_limits<std::uintptr_t>::max() - begin) return std::nullopt;
  const auto end = begin + owned.size_bytes();
  const auto first = reinterpret_cast<std::uintptr_t>(input.data());
  const auto bytes = input.size_bytes();
  if (first > std::numeric_limits<std::uintptr_t>::max() - bytes) return std::nullopt;
  const auto last = first + bytes;
  const bool overlaps = first < end && last > begin;
  if (!overlaps) return ChildRange{};
  if (first < begin || last > end || (first - begin) % sizeof(ComparisonOriginSetId) != 0)
    return std::nullopt;
  return ChildRange{
      .aliases = true,
      .offset = static_cast<std::size_t>((first - begin) / sizeof(ComparisonOriginSetId))};
}

bool BorrowsEntryStorage(const void* data, std::size_t bytes,
                         std::span<const ComparisonOriginEntry> entries) {
  if (bytes == 0 || entries.empty()) return false;
  const auto begin = reinterpret_cast<std::uintptr_t>(entries.data());
  const auto entry_bytes = entries.size_bytes();
  const auto first = reinterpret_cast<std::uintptr_t>(data);
  if (entry_bytes > std::numeric_limits<std::uintptr_t>::max() - begin ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - first)
    return true;
  return first < begin + entry_bytes && first + bytes > begin;
}

}  // namespace

std::expected<ComparisonDecisionLedger, PlanFailure> ComparisonDecisionLedger::Create(
    const ComparisonPlanningEnvironment& environment,
    std::shared_ptr<const ir::StorageFunder> funder, ir::DiagnosticAdmission* diagnostic) {
  PlannerMetadataAdmission local_diagnostic(funder);
  auto* effective_diagnostic = diagnostic != nullptr ? diagnostic : &local_diagnostic;
  const auto& source = environment.source;
  const auto& registry = environment.registry;
  const auto& device = environment.device;
  const auto& policy = environment.policy;
  const auto* shape = environment.shape_override != nullptr ? environment.shape_override
                                                            : environment.options.shape;
  if (!funder || source.io.graph.identity() == 0)
    return LedgerFailure(PlanError::kAdmission, "decision ledger requires a paid original source",
                         diagnostic);
  auto boundary = components::MakeComparisonBoundary(source, funder, effective_diagnostic);
  if (!boundary)
    return LedgerFailure(boundary.error().code == components::ComparisonBoundaryError::kAdmission
                             ? PlanError::kAdmission
                             : PlanError::kStaleComparisonSite,
                         boundary.error().detail.c_str(), diagnostic);
  auto loan = ir::AcquireStorageLoan(funder, 0);
  if (!loan)
    return LedgerFailure(PlanError::kAdmission, "decision ledger owner was refused", diagnostic);
  ComparisonDecisionLedger ledger(std::move(funder));
  ledger.records_loan_ = std::move(*loan);
  ledger.environment_ = &environment;
  ledger.ledger_identity_ = NextComparisonLedgerIdentity();
  if (ledger.ledger_identity_ == 0)
    return LedgerFailure(PlanError::kAdmission, "decision ledger identity space is exhausted",
                         diagnostic);
  ledger.source_identity_ = source.io.graph.identity();
  ledger.source_graph_digest_ = source.io.graph.Digest();
  ledger.source_boundary_digest_ = boundary->digest().combined;
  ledger.registry_digest_ = registry.SelectionDigest();
  ledger.policy_digest_ = Digest(policy);
  ledger.device_digest_ = DeviceFactsDigest(device);
  ledger.shape_digest_ = ComparisonShapeDigest(shape);
  ledger.options_digest_ = ComparisonOptionsDigest(environment.options);
  return ledger;
}

std::expected<void, PlanFailure> ComparisonDecisionLedger::Validate(
    const ComparisonPlanningEnvironment& environment, ir::DiagnosticAdmission* diagnostic) const {
  PlannerMetadataAdmission local_diagnostic(funder_);
  auto* effective_diagnostic = diagnostic != nullptr ? diagnostic : &local_diagnostic;
  if (environment_ != &environment || &environment.source != &environment_->source ||
      &environment.registry != &environment_->registry ||
      &environment.device != &environment_->device ||
      &environment.policy != &environment_->policy ||
      &environment.options != &environment_->options ||
      environment.shape_override != environment_->shape_override)
    return LedgerFailure(PlanError::kStaleComparisonSite,
                         "decision ledger was used with another planning environment", diagnostic);
  const auto& source = environment.source;
  const auto& registry = environment.registry;
  const auto& device = environment.device;
  const auto& policy = environment.policy;
  const auto* shape = environment.shape_override != nullptr ? environment.shape_override
                                                            : environment.options.shape;
  if (source.io.graph.identity() != source_identity_ ||
      source.io.graph.Digest() != source_graph_digest_ ||
      registry.SelectionDigest() != registry_digest_ || Digest(policy) != policy_digest_ ||
      DeviceFactsDigest(device) != device_digest_ ||
      ComparisonShapeDigest(shape) != shape_digest_ ||
      ComparisonOptionsDigest(environment.options) != options_digest_)
    return LedgerFailure(PlanError::kStaleComparisonSite,
                         "decision ledger source or planning facts changed", diagnostic);
  auto boundary = components::MakeComparisonBoundary(source, funder_, effective_diagnostic);
  if (!boundary)
    return LedgerFailure(boundary.error().code == components::ComparisonBoundaryError::kAdmission
                             ? PlanError::kAdmission
                             : PlanError::kStaleComparisonSite,
                         boundary.error().detail.c_str(), diagnostic);
  if (boundary->digest().combined != source_boundary_digest_)
    return LedgerFailure(PlanError::kStaleComparisonSite,
                         "decision ledger external source boundary changed", diagnostic);
  return {};
}

std::expected<void, PlanFailure> ComparisonDecisionLedger::ValidateCompletedAlternative(
    const ComparisonPlanningEnvironment& environment, const ComparisonTrial& trial,
    const OwnedComparisonPlan& plan, ir::DiagnosticAdmission* diagnostic) const {
  auto valid = Validate(environment, diagnostic);
  if (!valid) return std::unexpected(std::move(valid.error()));
  if (trial.source_boundary.source_identity() != source_identity_ ||
      trial.ledger_identity != ledger_identity_ || trial.ledger_generation != generation_ ||
      !plan.alternative_owner || plan.alternative_owner->ledger_identity != ledger_identity_ ||
      plan.alternative_owner->ledger_generation != generation_ ||
      trial.source_boundary.source_digest() != source_graph_digest_ ||
      trial.source_boundary.digest().combined != source_boundary_digest_ ||
      plan.plan.primitive_digest != source_graph_digest_ ||
      plan.plan.policy_digest != policy_digest_)
    return LedgerFailure(PlanError::kStaleComparisonSite,
                         "only a completed alternative for this original source can be committed",
                         diagnostic);
  const auto& site = plan.alternative_owner->site;
  const Pattern* registered = FindPattern(site.selector.pattern);
  if (site.selector.source_graph_identity != source_identity_ || registered == nullptr ||
      registered->version != site.selector.version ||
      registered->result != site.registered_result || site.selector.member_origins.empty() ||
      !AuthenticComparisonSite(site) ||
      site.complete_member_owner_segments.size() != site.selector.member_origins.size() ||
      site.complete_boundary_ports.empty() ||
      site.exact_site_identity != trial.site.exact_site_identity ||
      !AuthenticComparisonSite(trial.site) || !trial.checkpoint.history_matches_graph() ||
      plan.final_node_origins.source_graph_identity != source_identity_ ||
      plan.final_node_origins.planned_graph_identity != plan.plan.graph.identity() ||
      plan.final_node_origins.node_origins.size() != plan.plan.graph.nodes().size() ||
      plan.plan.graph_digest != plan.plan.graph.Digest())
    return LedgerFailure(
        PlanError::kStaleComparisonSite,
        "accepted alternative site no longer matches its registered trial identity", diagnostic);
  for (const auto& members : site.selector.member_origins) {
    if (members.empty())
      return LedgerFailure(PlanError::kStaleComparisonSite,
                           "accepted alternative contains an empty original member", diagnostic);
    for (std::size_t i = 0; i < members.size(); ++i)
      if (!members[i].valid() || members[i].v >= trial.checkpoint.origins.source_node_count() ||
          (i != 0 && members[i - 1].v >= members[i].v))
        return LedgerFailure(PlanError::kStaleComparisonSite,
                             "accepted alternative original members are invalid or unordered",
                             diagnostic);
  }
  for (const auto& prior : records_)
    if (prior.site_identity == site.exact_site_identity)
      return LedgerFailure(PlanError::kStaleComparisonSite,
                           "the exact accepted site already has a committed directive", diagnostic);
  const auto& journal = plan.alternative_owner->prior_decision_exclusions;
  for (std::size_t i = 0; i < journal.size(); ++i) {
    const auto& exclusion = journal[i];
    const auto prior = std::ranges::find_if(records_, [&](const auto& record) {
      return record.site_identity == exclusion.prior_site_identity;
    });
    const Pattern* competing = FindPattern(exclusion.competing_pattern);
    if (prior == records_.end() || competing == nullptr ||
        competing->version != exclusion.competing_version ||
        (competing->kind != PatternKind::kRewrite && competing->kind != PatternKind::kBatch))
      return LedgerFailure(PlanError::kStaleComparisonSite,
                           "the alternative exclusion journal is not bound to committed directives",
                           diagnostic);
    for (std::size_t j = 0; j < i; ++j)
      if (journal[j].prior_site_identity == exclusion.prior_site_identity &&
          journal[j].competing_pattern == exclusion.competing_pattern &&
          journal[j].competing_version == exclusion.competing_version)
        return LedgerFailure(PlanError::kStaleComparisonSite,
                             "the alternative exclusion journal contains a duplicate", diagnostic);
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max())
    return LedgerFailure(PlanError::kAdmission, "decision ledger generation is exhausted",
                         diagnostic);

  return {};
}

std::expected<void, PlanFailure> ComparisonDecisionLedger::ValidateTerminalWholeAlternative(
    const ComparisonPlanningEnvironment& environment, const ComparisonTrial& trial,
    const OwnedComparisonPlan& plan, ir::DiagnosticAdmission* diagnostic) const {
  auto valid = ValidateCompletedAlternative(environment, trial, plan, diagnostic);
  if (!valid) return valid;
  if (!trial.site.whole_checkpoint.belongs_to(source_identity_,
                                              trial.checkpoint.graph.identity()) ||
      trial.site.selector.checkpoint_match !=
          CheckpointMatchIdentity(trial.checkpoint.graph, trial.checkpoint.captured_members()) ||
      trial.site.whole_checkpoint.subject() !=
          WholeCheckpointSubject(environment.source.io.graph, trial.checkpoint, trial.site) ||
      plan.plan.whole_checkpoint.subject() != trial.site.whole_checkpoint.subject())
    return LedgerFailure(PlanError::kStaleComparisonSite,
                         "terminal whole alternative lacks its exact checkpoint authority",
                         diagnostic);
  return {};
}

std::expected<void, PlanFailure> ComparisonDecisionLedger::CommitAcceptedAlternative(
    const ComparisonPlanningEnvironment& environment, const ComparisonTrial& trial,
    const OwnedComparisonPlan& plan, ir::DiagnosticAdmission* diagnostic) {
  auto valid = ValidateCompletedAlternative(environment, trial, plan, diagnostic);
  if (!valid) return valid;
  const auto& site = plan.alternative_owner->site;
  if (site.whole_checkpoint.whole_only())
    return LedgerFailure(PlanError::kStaleComparisonSite,
                         "whole checkpoint authority cannot become a local decision record",
                         diagnostic);
  const Pattern* registered = FindPattern(site.selector.pattern);
  // A local fallback diagnostic owner only covers the synchronous boundary
  // validation above. A public failure returned from this call must not
  // borrow that local owner, so keep ledger-operation errors allocation-free
  // unless the caller supplied a longer-lived diagnostic admission.
  auto pending = CopyDecisionRecord(site, funder_, registered->name, diagnostic);
  if (!pending) return std::unexpected(std::move(pending.error()));
  if (records_.size() == records_.max_size())
    return LedgerFailure(PlanError::kAdmission, "decision ledger record count overflows",
                         diagnostic);
  if (records_.size() == records_.capacity()) {
    const std::size_t old_capacity = records_.capacity();
    const std::size_t next_capacity =
        old_capacity == 0
            ? 1
            : (old_capacity > records_.max_size() / 2 ? records_.max_size() : old_capacity * 2);
    constexpr std::uint64_t envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
    if (next_capacity >
        (std::numeric_limits<std::uint64_t>::max() - envelope) / sizeof(ComparisonDecisionRecord))
      return LedgerFailure(PlanError::kAdmission, "decision ledger capacity overflows", diagnostic);
    const auto bytes =
        static_cast<std::uint64_t>(next_capacity) * sizeof(ComparisonDecisionRecord) + envelope;
    if (!records_loan_.Grow(bytes))
      return LedgerFailure(PlanError::kAdmission, "decision ledger growth was refused", diagnostic);
    records_.reserve(next_capacity);
  }
  records_.push_back(std::move(*pending));
  ++generation_;
  return {};
}

std::expected<std::uint64_t, PlanFailure> ComparisonPlanInputMetadataBytes(
    const ir::Graph&, const execution::Registry&, const Policy& policy,
    const PlanOptions& options) {
  // SelectionView copies the device's measured-rank subset through a
  // push_back loop and appends at most one string_view per built-in pattern.
  // Charge geometric growth (old plus new buffers) before constructing it.
  // ApplyPatternTable separately acquires its identity-map loan immediately
  // before those temporary maps are reserved.
  constexpr std::uint64_t kEnvelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  std::uint64_t bytes = sizeof(StepPlan) + sizeof(PlanOptions) + sizeof(SelectionView);
  if (!AddGeometricCapacity<execution::MeasuredRank>(policy.ranks.size(), bytes) ||
      !AddGeometricCapacity<std::string_view>(Patterns().size(), bytes) ||
      !AddBytes(options.lanes.size(), sizeof(LaneTags::value_type), bytes) ||
      !AddBytes(options.invariant.size(), sizeof(ir::TensorId), bytes) ||
      !AddBytes(options.staged ? options.staged->size() : 0, sizeof(ir::TensorId), bytes) ||
      !AddBytes(options.alignment_staged ? options.alignment_staged->size() : 0,
                sizeof(ir::TensorId), bytes))
    return Failure(PlanError::kAdmission, "comparison input metadata inventory overflows");
  // Pointer-capturing forwarding adapters are fixed-size std::function
  // targets. Cover their possible allocation envelopes conservatively.
  constexpr std::uint64_t kForwarders = 5;
  if (kForwarders * kEnvelope > std::numeric_limits<std::uint64_t>::max() - bytes)
    return Failure(PlanError::kAdmission, "comparison callback inventory overflows");
  bytes += kForwarders * kEnvelope;
  return bytes;
}

std::expected<ComparisonOriginDag, PlanFailure> ComparisonOriginDag::Create(
    std::shared_ptr<const ir::StorageFunder> funder, std::uint32_t source_node_count,
    std::uint64_t entry_capacity, std::uint64_t child_capacity,
    ir::DiagnosticAdmission* diagnostic) {
  if (!funder)
    return Failure(PlanError::kAdmission, "comparison lineage requires a funder", diagnostic);
  ComparisonOriginDag dag;
  auto empty = ir::AcquireStorageLoan(funder, 0);
  if (!empty)
    return Failure(PlanError::kAdmission, "comparison lineage owner was refused", diagnostic);
  dag.loan_ = std::move(*empty);
  dag.source_node_count_ = source_node_count;
  auto reserved = dag.Reserve(entry_capacity, child_capacity, diagnostic);
  if (!reserved) return std::unexpected(std::move(reserved.error()));
  return dag;
}

std::expected<ComparisonOriginDag, PlanFailure> ComparisonOriginDag::Clone(
    std::shared_ptr<const ir::StorageFunder> funder, ir::DiagnosticAdmission* diagnostic) const {
  auto cloned =
      Create(std::move(funder), source_node_count_, entries_.size(), children_.size(), diagnostic);
  if (!cloned) return std::unexpected(std::move(cloned.error()));
  cloned->entries_.assign(entries_.begin(), entries_.end());
  cloned->children_.assign(children_.begin(), children_.end());
  cloned->reachable_.resize(entries_.size());
  return cloned;
}

std::expected<void, PlanFailure> ComparisonOriginDag::Reserve(std::uint64_t entries,
                                                              std::uint64_t children,
                                                              ir::DiagnosticAdmission* diagnostic) {
  if (entries <= entries_.capacity() && children <= children_.capacity() &&
      entries <= reachable_.capacity() && source_node_count_ <= source_membership_.capacity())
    return {};
  if (entries > std::numeric_limits<std::size_t>::max() ||
      children > std::numeric_limits<std::size_t>::max())
    return Failure(PlanError::kAdmission, "comparison lineage capacity overflows size_t",
                   diagnostic);
  const auto grow = [](std::uint64_t current, std::uint64_t requested) {
    if (requested <= current) return current;
    if (current == 0 || current > std::numeric_limits<std::uint64_t>::max() / 2) return requested;
    return std::max(requested, current * 2);
  };
  const auto next_entries = grow(entries_.capacity(), entries);
  const auto next_children = grow(children_.capacity(), children);
  if (next_entries > std::numeric_limits<std::size_t>::max() ||
      next_children > std::numeric_limits<std::size_t>::max())
    return Failure(PlanError::kAdmission, "comparison lineage growth exceeds size_t", diagnostic);
  std::uint64_t bytes = 0;
  if (!CapacityBytes(next_entries, next_children, source_node_count_, bytes))
    return Failure(PlanError::kAdmission, "comparison lineage capacity overflows byte count",
                   diagnostic);
  const auto funder = loan_.funder();
  auto next = ir::AcquireStorageLoan(funder, bytes);
  if (!next)
    return Failure(PlanError::kAdmission, "comparison lineage growth was refused", diagnostic);
  entries_.reserve(static_cast<std::size_t>(next_entries));
  children_.reserve(static_cast<std::size_t>(next_children));
  reachable_.reserve(static_cast<std::size_t>(next_entries));
  source_membership_.reserve(source_node_count_);
  source_membership_.resize(source_node_count_);
  // The pinned libstdc++ reserve implementation allocates exactly the
  // requested capacity; CapacityBytes charges those arrays plus allocator
  // envelopes before this point.
  // Both new arrays are installed before the old loan is refunded.
  loan_ = std::move(*next);
  return {};
}

std::expected<ComparisonOriginSetId, PlanFailure> ComparisonOriginDag::AddSingleton(
    ir::NodeId origin, ir::DiagnosticAdmission* diagnostic) {
  if (!origin.valid() || origin.v >= source_node_count_)
    return Failure(PlanError::kStaleComparisonSite, "lineage origin is outside the source graph",
                   diagnostic);
  if (entries_.size() >= std::numeric_limits<ComparisonOriginSetId>::max())
    return Failure(PlanError::kAdmission, "comparison lineage id space exhausted", diagnostic);
  auto reserved = Reserve(entries_.size() + 1, children_.size(), diagnostic);
  if (!reserved) return std::unexpected(std::move(reserved.error()));
  const auto id = static_cast<ComparisonOriginSetId>(entries_.size());
  entries_.push_back(ComparisonOriginEntry{.singleton = true, .origin = origin});
  reachable_.resize(entries_.size());
  return id;
}

std::expected<ComparisonOriginSetId, PlanFailure> ComparisonOriginDag::AddUnion(
    std::span<const ComparisonOriginSetId> members, ir::DiagnosticAdmission* diagnostic) {
  if (members.empty())
    return Failure(PlanError::kStaleComparisonSite, "empty lineage union", diagnostic);
  if (BorrowsEntryStorage(members.data(), members.size_bytes(), entries_))
    return Failure(PlanError::kStaleComparisonSite, "lineage members borrow entry storage",
                   diagnostic);
  const auto alias = ChildAlias(members, children_);
  if (!alias)
    return Failure(PlanError::kStaleComparisonSite, "malformed lineage edge self-borrow",
                   diagnostic);
  if (entries_.size() >= std::numeric_limits<ComparisonOriginSetId>::max() ||
      children_.size() > std::numeric_limits<std::uint32_t>::max() ||
      members.size() > std::numeric_limits<std::uint32_t>::max() - children_.size())
    return Failure(PlanError::kAdmission, "comparison lineage id space exhausted", diagnostic);
  for (const auto member : members)
    if (member >= entries_.size())
      return Failure(PlanError::kStaleComparisonSite, "lineage union references an unknown set",
                     diagnostic);
  auto reserved = Reserve(entries_.size() + 1, children_.size() + members.size(), diagnostic);
  if (!reserved) return std::unexpected(std::move(reserved.error()));
  const auto id = static_cast<ComparisonOriginSetId>(entries_.size());
  const auto first = static_cast<std::uint32_t>(children_.size());
  if (alias->aliases) {
    for (std::size_t i = 0; i < members.size(); ++i)
      children_.push_back(children_[alias->offset + i]);
  } else {
    children_.insert(children_.end(), members.begin(), members.end());
  }
  entries_.push_back(ComparisonOriginEntry{
      .first_child = first, .child_count = static_cast<std::uint32_t>(members.size())});
  reachable_.resize(entries_.size());
  return id;
}

std::expected<ComparisonOriginSetId, PlanFailure> ComparisonOriginDag::AddUnion(
    std::span<const ir::NodeId> member_nodes, std::span<const ComparisonOriginSetId> node_origins,
    ir::DiagnosticAdmission* diagnostic) {
  if (member_nodes.empty())
    return Failure(PlanError::kStaleComparisonSite, "empty lineage union", diagnostic);
  // A caller can pass a span over the `origin` fields of entries(). Reserve
  // may move that storage. Reject any such borrow before changing capacity;
  // an arbitrary strided view is not safely rebased by this API.
  if (BorrowsEntryStorage(member_nodes.data(), member_nodes.size_bytes(), entries_) ||
      BorrowsEntryStorage(node_origins.data(), node_origins.size_bytes(), entries_))
    return Failure(PlanError::kStaleComparisonSite, "lineage inputs borrow entry storage",
                   diagnostic);
  const auto alias = ChildAlias(node_origins, children_);
  if (!alias)
    return Failure(PlanError::kStaleComparisonSite, "malformed lineage node-map self-borrow",
                   diagnostic);
  if (entries_.size() >= std::numeric_limits<ComparisonOriginSetId>::max() ||
      children_.size() > std::numeric_limits<std::uint32_t>::max() ||
      member_nodes.size() > std::numeric_limits<std::uint32_t>::max() - children_.size())
    return Failure(PlanError::kAdmission, "comparison lineage id space exhausted", diagnostic);
  for (const auto node : member_nodes)
    if (!node.valid() || node.v >= node_origins.size() || node_origins[node.v] >= entries_.size())
      return Failure(PlanError::kStaleComparisonSite, "lineage union references an unknown node",
                     diagnostic);
  auto reserved = Reserve(entries_.size() + 1, children_.size() + member_nodes.size(), diagnostic);
  if (!reserved) return std::unexpected(std::move(reserved.error()));
  const auto id = static_cast<ComparisonOriginSetId>(entries_.size());
  const auto first = static_cast<std::uint32_t>(children_.size());
  for (const auto node : member_nodes)
    children_.push_back(alias->aliases ? children_[alias->offset + node.v] : node_origins[node.v]);
  entries_.push_back(ComparisonOriginEntry{
      .first_child = first, .child_count = static_cast<std::uint32_t>(member_nodes.size())});
  reachable_.resize(entries_.size());
  return id;
}

namespace {
bool MarkOrigins(ComparisonOriginSetId set, std::span<const ComparisonOriginEntry> entries,
                 std::span<const ComparisonOriginSetId> children, std::span<std::uint8_t> reachable,
                 std::span<std::uint8_t> source_membership, std::uint32_t source_node_count,
                 bool clear_membership = true) {
  if (set >= entries.size() || reachable.size() != entries.size() ||
      source_membership.size() != source_node_count)
    return false;
  std::ranges::fill(reachable, 0);
  if (clear_membership) std::ranges::fill(source_membership, 0);
  reachable[set] = 1;
  for (std::size_t cursor = static_cast<std::size_t>(set) + 1; cursor-- > 0;) {
    if (reachable[cursor] == 0) continue;
    const auto& entry = entries[cursor];
    if (entry.singleton) {
      if (!entry.origin.valid() || entry.origin.v >= source_node_count) return false;
      source_membership[entry.origin.v] = 1;
      continue;
    }
    const std::uint64_t end = static_cast<std::uint64_t>(entry.first_child) + entry.child_count;
    if (end > children.size()) return false;
    for (std::uint32_t j = 0; j < entry.child_count; ++j) {
      const auto child = children[entry.first_child + j];
      if (child >= cursor) return false;
      reachable[child] = 1;
    }
  }
  return true;
}
}  // namespace

bool ComparisonOriginDag::Contains(ComparisonOriginSetId set, ir::NodeId origin) {
  return origin.valid() && origin.v < source_node_count_ &&
         MarkOrigins(set, entries_, children_, reachable_, source_membership_,
                     source_node_count_) &&
         source_membership_[origin.v] != 0;
}

std::optional<std::size_t> ComparisonOriginDag::OriginCount(ComparisonOriginSetId set) {
  if (!MarkOrigins(set, entries_, children_, reachable_, source_membership_, source_node_count_))
    return std::nullopt;
  return static_cast<std::size_t>(std::ranges::count(source_membership_, std::uint8_t{1}));
}

bool ComparisonOriginDag::CopyOrigins(ComparisonOriginSetId set, std::span<ir::NodeId> output,
                                      std::size_t& written) {
  written = 0;
  if (!MarkOrigins(set, entries_, children_, reachable_, source_membership_, source_node_count_))
    return false;
  const auto count =
      static_cast<std::size_t>(std::ranges::count(source_membership_, std::uint8_t{1}));
  if (output.size() < count) return false;
  for (std::uint32_t i = 0; i < source_node_count_; ++i)
    if (source_membership_[i] != 0) output[written++] = ir::NodeId{i};
  return true;
}

bool ComparisonOriginDag::Equals(ComparisonOriginSetId set, const ComparisonOriginSet& origins) {
  if (set >= entries_.size()) return false;
  for (std::size_t i = 0; i < origins.size(); ++i) {
    if (!origins[i].valid() || origins[i].v >= source_node_count_ ||
        (i != 0 && origins[i - 1].v >= origins[i].v))
      return false;
  }
  if (!MarkOrigins(set, entries_, children_, reachable_, source_membership_, source_node_count_))
    return false;
  std::size_t found = 0;
  std::size_t requested = 0;
  for (std::uint32_t n = 0; n < source_node_count_; ++n) {
    const bool actual = source_membership_[n] != 0;
    if (actual) ++found;
    const bool expected = requested < origins.size() && origins[requested].v == n;
    if (expected) ++requested;
    if (actual != expected) return false;
  }
  return found == origins.size() && requested == origins.size();
}

bool ComparisonOriginDag::EqualsUnionOfNodes(std::span<const ir::NodeId> nodes,
                                             std::span<const ComparisonOriginSetId> node_origins,
                                             const ComparisonOriginSet& origins) {
  if (nodes.empty() || origins.size() > source_node_count_) return false;
  for (std::size_t i = 0; i < origins.size(); ++i)
    if (!origins[i].valid() || origins[i].v >= source_node_count_ ||
        (i != 0 && origins[i - 1].v >= origins[i].v))
      return false;
  std::ranges::fill(source_membership_, 0);
  for (const ir::NodeId node : nodes) {
    if (!node.valid() || node.v >= node_origins.size() ||
        !MarkOrigins(node_origins[node.v], entries_, children_, reachable_, source_membership_,
                     source_node_count_, false))
      return false;
  }
  std::size_t found = 0;
  std::size_t requested = 0;
  for (std::uint32_t n = 0; n < source_node_count_; ++n) {
    const bool actual = source_membership_[n] != 0;
    if (actual) ++found;
    const bool expected = requested < origins.size() && origins[requested].v == n;
    if (expected) ++requested;
    if (actual != expected) return false;
  }
  return found == origins.size() && requested == origins.size();
}

bool ComparisonCaptureSession::ExcludesRewrite(const Pattern& pattern,
                                               std::span<const ir::NodeId> matched_nodes,
                                               std::span<const ComparisonOriginSetId> node_origins,
                                               const ir::Graph& graph,
                                               std::span<const ir::TensorId> source_to_graph) {
  // Ledger directives are independent of the current-site exclusion mode.
  // In particular, a fused original member can decompose into several nodes,
  // so its owner count must not be compared with the rewrite's match count.
  if (pattern.kind == PatternKind::kRewrite) {
    for (const auto& decision : decisions_) {
      bool same_members = false;
      const bool fused_record = decision.kind == ComparisonSiteKind::kFusedNode ||
                                decision.kind == ComparisonSiteKind::kCompositeNode;
      if (decision.complete_boundary_ports.empty()) continue;
      if (fused_record) {
        same_members = DecisionRegionMatchesCurrent(decision, matched_nodes, node_origins, origins_,
                                                    graph, source_to_graph, PriorTensorMap());
      } else if (decision.kind == ComparisonSiteKind::kRewrite &&
                 decision.complete_member_owner_segments.size() == matched_nodes.size()) {
        same_members = decision.member_origins.size() == matched_nodes.size();
        for (std::size_t i = 0; same_members && i < matched_nodes.size(); ++i) {
          const auto node = matched_nodes[i];
          same_members = node.valid() && node.v < node_origins.size() &&
                         origins_.Equals(node_origins[node.v], decision.member_origins[i]);
        }
      }
      if (!same_members ||
          (!fused_record &&
           !SitePortsMatchCurrent(decision.complete_member_owner_segments,
                                  decision.complete_boundary_ports, graph, matched_nodes,
                                  source_to_graph, true, PriorTensorMap())))
        continue;
      RecordDecisionExclusion(decision, pattern.name, pattern.version);
      // The decision record itself is the authenticated exclusion. Do not
      // relabel it as a current-site competing rewrite: the two sites may
      // share pattern/version, while exact identity belongs to the prior site.
      return true;
    }
  }

  if (!exclude_site_ ||
      (selector_.kind != ComparisonSiteKind::kFusedNode &&
       selector_.kind != ComparisonSiteKind::kCompositeNode &&
       selector_.kind != ComparisonSiteKind::kRewrite) ||
      selector_.member_origins.empty())
    return false;
  const bool accepted_pattern =
      selector_.pattern == pattern.name && selector_.version == pattern.version;
  if (accepted_pattern) {
    if (selector_.kind == ComparisonSiteKind::kRewrite) {
      if (selector_.member_origins.size() != matched_nodes.size()) return false;
      for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
        const auto node = matched_nodes[i];
        if (!node.valid() || node.v >= node_origins.size() ||
            !origins_.Equals(node_origins[node.v], selector_.member_origins[i]))
          return false;
      }
      if (alternative_site_ && alternative_site_->whole_checkpoint.whole_only()) {
        if (!whole_checkpoint_ ||
            whole_checkpoint_->captured_members().size() != matched_nodes.size())
          return false;
        const auto same_tensor = [&](ir::TensorId old, ir::TensorId current) {
          if (!old.valid() || !current.valid()) return old.valid() == current.valid();
          if (old.v >= whole_checkpoint_->graph.tensor_count() || current.v >= graph.tensor_count())
            return false;
          const auto& a = whole_checkpoint_->graph.tensor(old);
          const auto& b = graph.tensor(current);
          return a.role == b.role && a.format == b.format && a.nb == b.nb &&
                 a.segment == b.segment && a.view_offset == b.view_offset &&
                 a.is_view() == b.is_view() && a.binding == b.binding;
        };
        for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
          const auto& a = whole_checkpoint_->graph.node(whole_checkpoint_->captured_members()[i]);
          const auto& b = graph.node(matched_nodes[i]);
          if (a.key != b.key || a.params != b.params || a.nsrc != b.nsrc || a.nout != b.nout)
            return false;
          for (std::size_t j = 0; j < a.nsrc; ++j)
            if (!same_tensor(a.source(j), b.source(j))) return false;
          for (std::size_t j = 0; j < a.nout; ++j)
            if (!same_tensor(a.out[j], b.out[j])) return false;
        }
        const bool exact =
            alternative_site_->exact_site_identity == ExactSiteIdentity(*alternative_site_) &&
            SitePortsMatchCurrent(alternative_site_->complete_member_owner_segments,
                                  alternative_site_->complete_boundary_ports, graph, matched_nodes,
                                  source_to_graph, true);
        whole_site_excluded_ |= exact;
        return exact;
      }
      return SitePortsMatchCurrent(selector_, graph, matched_nodes, source_to_graph);
    }
    if (selector_.kind != ComparisonSiteKind::kFusedNode &&
        selector_.kind != ComparisonSiteKind::kCompositeNode)
      return false;
    if (alternative_site_ == nullptr ||
        alternative_site_->exact_site_identity != ExactSiteIdentity(*alternative_site_) ||
        alternative_site_->selector.source_graph_identity != selector_.source_graph_identity ||
        alternative_site_->selector.kind != selector_.kind ||
        alternative_site_->selector.pattern != selector_.pattern ||
        alternative_site_->selector.version != selector_.version ||
        alternative_site_->selector.member_origins != selector_.member_origins)
      return false;
    return FusedRegionMatchesCurrent(alternative_site_->selector.kind,
                                     alternative_site_->selector.member_origins,
                                     alternative_site_->complete_member_owner_segments,
                                     alternative_site_->complete_boundary_ports, matched_nodes,
                                     node_origins, origins_, graph, source_to_graph);
  }

  if (alternative_site_ == nullptr ||
      alternative_mode_ != AlternativeMode::kDecomposeAcceptedRegion ||
      pattern.kind != PatternKind::kRewrite ||
      (pattern.name == selector_.pattern && pattern.version == selector_.version) ||
      alternative_site_->exact_site_identity != ExactSiteIdentity(*alternative_site_))
    return false;
  if (selector_.kind != ComparisonSiteKind::kRewrite ||
      selector_.member_origins.size() != matched_nodes.size() ||
      alternative_site_->complete_member_owner_segments.size() != matched_nodes.size() ||
      alternative_site_->complete_boundary_ports.empty())
    return false;
  for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
    const auto node = matched_nodes[i];
    if (!node.valid() || node.v >= node_origins.size() ||
        !origins_.Equals(node_origins[node.v], selector_.member_origins[i]))
      return false;
  }
  if (!SitePortsMatchCurrent(alternative_site_->complete_member_owner_segments,
                             alternative_site_->complete_boundary_ports, graph, matched_nodes,
                             source_to_graph, true))
    return false;

  const auto already_recorded =
      std::ranges::find_if(alternative_site_->excluded_competing_rewrites, [&](const auto& record) {
        return record.pattern == pattern.name && record.version == pattern.version &&
               record.site_identity == alternative_site_->exact_site_identity;
      });
  if (already_recorded != alternative_site_->excluded_competing_rewrites.end()) return true;
  if (alternative_site_->excluded_competing_rewrites.size() ==
      alternative_site_->excluded_competing_rewrites.capacity()) {
    // Capacity is prepaid from the immutable registered table before planning.
    // Overflow means the audit record could not be retained, so fail closed.
    record_refused_ = true;
    return false;
  }
  alternative_site_->excluded_competing_rewrites.push_back(
      ComparisonSite::ExcludedRewrite{.pattern = pattern.name,
                                      .version = pattern.version,
                                      .site_identity = alternative_site_->exact_site_identity});
  return true;
}

std::expected<std::optional<ComparisonCheckpoint>, PlanFailure>
ComparisonCaptureSession::CaptureRewrite(
    std::string_view pattern, std::uint16_t version, std::span<const ir::NodeId> matched_nodes,
    std::span<const ComparisonOriginSetId> node_origins, const ir::Graph& graph,
    std::span<const ir::TensorId> source_to_graph, std::span<const ir::NodeId> graph_node_to_source,
    ir::DiagnosticAdmission* diagnostic, std::span<const Applied> prior_applied,
    std::span<const ir::NodeId> prior_to_source) {
  if (selector_.kind != ComparisonSiteKind::kRewrite || selector_.pattern != pattern ||
      selector_.version != version || selector_.member_origins.size() != matched_nodes.size())
    return std::optional<ComparisonCheckpoint>{};
  if (node_origins.size() != graph.nodes().size() ||
      graph_node_to_source.size() != graph.nodes().size())
    return Failure(PlanError::kStaleComparisonSite, "accepted rewrite lineage is incomplete",
                   diagnostic);
  for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
    const auto node = matched_nodes[i];
    if (!node.valid() || node.v >= node_origins.size() ||
        !origins_.Equals(node_origins[node.v], selector_.member_origins[i]))
      return std::optional<ComparisonCheckpoint>{};
  }
  if (!SitePortsMatchCurrent(selector_, graph, matched_nodes, source_to_graph))
    return std::optional<ComparisonCheckpoint>{};
  if (HasCheckpointMatch(selector_) && !recapture_after_decisions_ &&
      CheckpointMatchIdentity(graph, matched_nodes) != selector_.checkpoint_match)
    return std::optional<ComparisonCheckpoint>{};
  if (!funder_)
    return Failure(PlanError::kAdmission, "comparison capture has no storage funder", diagnostic);
  const auto bytes = MappingBytes(source_to_graph.size(), graph.nodes().size());
  if (!bytes) return Failure(PlanError::kAdmission, "rewrite checkpoint maps overflow", diagnostic);
  auto loan = ir::AcquireStorageLoan(funder_, *bytes);
  if (!loan)
    return Failure(PlanError::kAdmission, "rewrite checkpoint maps were refused", diagnostic);
  auto graph_copy = graph.Clone(funder_, diagnostic);
  if (!graph_copy)
    return Failure(graph_copy.error().error == ir::GraphError::kAdmission ? PlanError::kAdmission
                                                                          : PlanError::kPattern,
                   graph_copy.error().detail.c_str(), diagnostic);
  auto dag_copy = origins_.Clone(funder_, diagnostic);
  if (!dag_copy) return std::unexpected(std::move(dag_copy.error()));
  std::vector<ir::TensorId> tensor_map;
  std::vector<ir::NodeId> node_map;
  std::vector<ComparisonOriginSetId> node_origins_copy;
  tensor_map.reserve(source_to_graph.size());
  node_map.reserve(graph_node_to_source.size());
  node_origins_copy.reserve(node_origins.size());
  tensor_map.insert(tensor_map.end(), source_to_graph.begin(), source_to_graph.end());
  node_map.insert(node_map.end(), graph_node_to_source.begin(), graph_node_to_source.end());
  node_origins_copy.insert(node_origins_copy.end(), node_origins.begin(), node_origins.end());
  std::uint64_t member_bytes = 0;
  if (!AddBytes(matched_nodes.size(), sizeof(ir::NodeId), member_bytes) ||
      !loan->Grow(member_bytes))
    return Failure(PlanError::kAdmission, "captured member payment was refused", diagnostic);
  std::vector<ir::NodeId> captured_members(matched_nodes.begin(), matched_nodes.end());
  auto history = CopyAppliedHistory(prior_applied, prior_to_source, graph.nodes().size(),
                                    origins_.source_node_count(), *loan, diagnostic);
  if (!history) return std::unexpected(std::move(history.error()));
  std::optional<ComparisonCheckpoint> captured(
      std::in_place, std::move(*loan), std::move(*graph_copy), std::move(tensor_map),
      std::move(node_map), std::move(node_origins_copy), std::move(*dag_copy));
  captured->prior_applied_ = std::move(*history);
  captured->captured_members_ = std::move(captured_members);
  if (HasCheckpointMatch(selector_)) {
    std::uint64_t map_bytes = 0;
    if (!AddBytes(graph.tensor_count(), sizeof(ir::TensorId), map_bytes) ||
        !captured->maps_loan.Grow(map_bytes))
      return Failure(PlanError::kAdmission, "whole checkpoint output map payment was refused",
                     diagnostic);
    captured->checkpoint_to_current_.reserve(graph.tensor_count());
    for (std::uint32_t t = 0; t < graph.tensor_count(); ++t)
      captured->checkpoint_to_current_.push_back(ir::TensorId{t});
    captured->current_graph_identity_ = graph.identity();
  }
  return captured;
}

std::expected<std::optional<ComparisonCheckpoint>, PlanFailure>
ComparisonCaptureSession::CaptureBatch(
    std::string_view pattern, std::uint16_t version, std::span<const ir::NodeId> matched_nodes,
    std::span<const ComparisonOriginSetId> node_origins, const ir::Graph& graph,
    std::span<const ir::TensorId> source_to_graph, std::span<const ir::NodeId> graph_node_to_source,
    ir::DiagnosticAdmission* diagnostic, std::span<const Applied> prior_applied,
    std::span<const ir::NodeId> prior_to_source) {
  if (selector_.kind != ComparisonSiteKind::kBatch || selector_.pattern != pattern ||
      selector_.version != version || selector_.member_origins.size() != matched_nodes.size() ||
      (!selector_.member_owner_segments.empty() &&
       selector_.member_owner_segments.size() != matched_nodes.size()))
    return std::optional<ComparisonCheckpoint>{};
  if (node_origins.size() != graph.nodes().size() ||
      graph_node_to_source.size() != graph.nodes().size())
    return Failure(PlanError::kStaleComparisonSite, "accepted batch lineage is incomplete",
                   diagnostic);
  for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
    const auto node = matched_nodes[i];
    if (!node.valid() || node.v >= node_origins.size() ||
        !origins_.Equals(node_origins[node.v], selector_.member_origins[i]))
      return std::optional<ComparisonCheckpoint>{};
    if (!selector_.member_owner_segments.empty()) {
      const auto& member = graph.node(node);
      const auto segment =
          member.nout == 0 ? ir::kJoinedSegment : graph.tensor(member.out[0]).segment;
      if (segment != selector_.member_owner_segments[i])
        return std::optional<ComparisonCheckpoint>{};
    }
  }
  if (!funder_)
    return Failure(PlanError::kAdmission, "comparison capture has no storage funder", diagnostic);
  const auto bytes = MappingBytes(source_to_graph.size(), graph.nodes().size());
  if (!bytes) return Failure(PlanError::kAdmission, "batch checkpoint maps overflow", diagnostic);
  auto loan = ir::AcquireStorageLoan(funder_, *bytes);
  if (!loan)
    return Failure(PlanError::kAdmission, "batch checkpoint maps were refused", diagnostic);
  auto graph_copy = graph.Clone(funder_, diagnostic);
  if (!graph_copy)
    return Failure(graph_copy.error().error == ir::GraphError::kAdmission ? PlanError::kAdmission
                                                                          : PlanError::kPattern,
                   graph_copy.error().detail.c_str(), diagnostic);
  auto dag_copy = origins_.Clone(funder_, diagnostic);
  if (!dag_copy) return std::unexpected(std::move(dag_copy.error()));
  std::vector<ir::TensorId> tensor_map;
  std::vector<ir::NodeId> node_map;
  std::vector<ComparisonOriginSetId> node_origins_copy;
  tensor_map.reserve(source_to_graph.size());
  node_map.reserve(graph_node_to_source.size());
  node_origins_copy.reserve(node_origins.size());
  tensor_map.insert(tensor_map.end(), source_to_graph.begin(), source_to_graph.end());
  node_map.insert(node_map.end(), graph_node_to_source.begin(), graph_node_to_source.end());
  node_origins_copy.insert(node_origins_copy.end(), node_origins.begin(), node_origins.end());
  auto history = CopyAppliedHistory(prior_applied, prior_to_source, graph.nodes().size(),
                                    origins_.source_node_count(), *loan, diagnostic);
  if (!history) return std::unexpected(std::move(history.error()));
  std::optional<ComparisonCheckpoint> captured(
      std::in_place, std::move(*loan), std::move(*graph_copy), std::move(tensor_map),
      std::move(node_map), std::move(node_origins_copy), std::move(*dag_copy));
  captured->prior_applied_ = std::move(*history);
  return captured;
}

std::expected<void, PlanFailure> ComparisonCaptureSession::RecordDiscovery(
    ComparisonSiteKind kind, std::string_view pattern, std::uint16_t version, ir::OpKey operation,
    std::span<const ir::NodeId> matched_nodes, std::span<const ComparisonOriginSetId> node_origins,
    const ir::Graph& graph, std::span<const ir::TensorId> source_to_graph,
    ir::DiagnosticAdmission* diagnostic) {
  if (discovery_ == nullptr) return {};
  return discovery_->Record(kind, pattern, version, operation, matched_nodes, node_origins, graph,
                            source_to_graph, diagnostic);
}

std::expected<void, PlanFailure> ComparisonCaptureSession::FinalizeDiscovery(
    const ir::Graph& planned_graph, std::span<const ir::TensorId> source_to_planned,
    std::span<const ComparisonOriginSetId> planned_node_origins,
    ir::DiagnosticAdmission* diagnostic) {
  if (discovery_ == nullptr) return {};
  return discovery_->Finalize(planned_graph, source_to_planned, planned_node_origins, diagnostic);
}

bool ComparisonCaptureSession::AdvanceCheckpointMap(std::uint64_t previous_graph,
                                                    std::uint64_t next_graph,
                                                    std::span<const ir::TensorId> tensors) {
  if (!checkpoint_ || checkpoint_->checkpoint_to_current_.empty()) return true;
  if (!previous_graph || !next_graph || checkpoint_->current_graph_identity_ != previous_graph) {
    record_refused_ = true;
    return false;
  }
  for (auto& tensor : checkpoint_->checkpoint_to_current_) {
    if (!tensor.valid()) continue;
    if (tensor.v >= tensors.size()) {
      record_refused_ = true;
      return false;
    }
    tensor = tensors[tensor.v];
  }
  checkpoint_->current_graph_identity_ = next_graph;
  return true;
}

void ComparisonCaptureSession::CommitRound(std::optional<ComparisonCheckpoint> checkpoint,
                                           std::size_t candidates, std::uint64_t previous_graph,
                                           std::uint64_t next_graph,
                                           std::span<const ir::TensorId> tensors) {
  if (checkpoint && !checkpoint_) checkpoint_.emplace(std::move(*checkpoint));
  if (previous_graph || next_graph) AdvanceCheckpointMap(previous_graph, next_graph, tensors);
  accepted_candidates_ += candidates;
}

std::expected<ComparisonCheckpoint, PlanFailure> ComparisonCaptureSession::TakeCheckpoint(
    ir::DiagnosticAdmission* diagnostic) {
  if (accepted_candidates_ == 0 || !checkpoint_)
    return Failure(PlanError::kMissingComparisonSite,
                   "the requested site was not accepted by the incumbent plan", diagnostic);
  if (accepted_candidates_ != 1)
    return Failure(PlanError::kAmbiguousComparisonSite,
                   "the selector matches more than one accepted rewrite site", diagnostic);
  return std::move(*checkpoint_);
}

bool ComparisonCaptureSession::ExcludesBatch(std::string_view pattern, std::uint16_t version,
                                             std::span<const ir::NodeId> matched_nodes,
                                             std::span<const ComparisonOriginSetId> node_origins,
                                             const ir::Graph& graph,
                                             std::span<const ir::TensorId> source_to_graph) {
  for (const auto& decision : decisions_) {
    if (decision.kind != ComparisonSiteKind::kBatch || decision.pattern != pattern ||
        decision.version != version || decision.member_origins.size() != matched_nodes.size() ||
        decision.complete_member_owner_segments.size() != matched_nodes.size() ||
        decision.complete_boundary_ports.empty())
      continue;
    bool same_members = true;
    for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
      const auto node = matched_nodes[i];
      if (!node.valid() || node.v >= node_origins.size() ||
          !origins_.Equals(node_origins[node.v], decision.member_origins[i])) {
        same_members = false;
        break;
      }
    }
    if (same_members &&
        SitePortsMatchCurrent(decision.complete_member_owner_segments,
                              decision.complete_boundary_ports, graph, matched_nodes,
                              source_to_graph, true, PriorTensorMap())) {
      RecordDecisionExclusion(decision, pattern, version);
      return true;
    }
  }
  if (!exclude_site_ || selector_.kind != ComparisonSiteKind::kBatch ||
      selector_.pattern != pattern || selector_.version != version ||
      selector_.member_origins.size() != matched_nodes.size() ||
      (!selector_.member_owner_segments.empty() &&
       selector_.member_owner_segments.size() != matched_nodes.size()))
    return false;
  for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
    const ir::NodeId node = matched_nodes[i];
    if (node.v >= graph.nodes().size() ||
        !origins_.Equals(node_origins[node.v], selector_.member_origins[i]))
      return false;
    if (!selector_.member_owner_segments.empty()) {
      const auto segment = graph.node(node).nout == 0
                               ? ir::kJoinedSegment
                               : graph.tensor(graph.node(node).out[0]).segment;
      if (segment != selector_.member_owner_segments[i]) return false;
    }
  }
  return true;
}

std::expected<ComparisonDiscoverySession, PlanFailure> ComparisonDiscoverySession::Create(
    const ir::Graph& source, std::shared_ptr<const ir::StorageFunder> funder,
    std::span<const ir::OpKey> operation_filter, bool harness_all,
    ir::DiagnosticAdmission* diagnostic) {
  if (!funder || source.identity() == 0)
    return Failure(PlanError::kAdmission, "comparison discovery requires a live source and funder",
                   diagnostic);
  const bool enabled = harness_all || !operation_filter.empty();
  if (!enabled) return ComparisonDiscoverySession(source.identity(), std::move(funder), false);
  if (source.nodes().size() > std::numeric_limits<std::uint32_t>::max())
    return Failure(PlanError::kAdmission, "comparison discovery source exceeds lineage IDs",
                   diagnostic);
  auto loan = ir::AcquireStorageLoan(funder, 0);
  if (!loan)
    return Failure(PlanError::kAdmission, "comparison discovery owner was refused", diagnostic);
  auto dag = ComparisonOriginDag::Create(funder, static_cast<std::uint32_t>(source.nodes().size()),
                                         0, 0, diagnostic);
  if (!dag) return std::unexpected(std::move(dag.error()));
  ComparisonDiscoverySession result(source.identity(), funder, true, std::move(*dag));
  result.loan_ = std::move(*loan);
  result.operation_filter_ = operation_filter;
  result.harness_all_ = harness_all;
  std::uint64_t initial_bytes = 0;
  if (!AddBytes(source.tensor_count(), sizeof(ir::TensorId), initial_bytes) ||
      !AddBytes(source.nodes().size(), sizeof(ir::NodeId), initial_bytes) ||
      !AddBytes(source.nodes().size(), sizeof(ComparisonOriginSetId), initial_bytes))
    return Failure(PlanError::kAdmission, "comparison discovery seed-map admission refused",
                   diagnostic);
  auto seed_loan = ir::AcquireStorageLoan(funder, initial_bytes);
  if (!seed_loan)
    return Failure(PlanError::kAdmission, "comparison discovery seed-map admission refused",
                   diagnostic);
  result.seed_loan_ = std::move(*seed_loan);
  result.initial_source_to_graph_.reserve(source.tensor_count());
  result.initial_graph_node_to_source_.reserve(source.nodes().size());
  result.initial_node_origins_.reserve(source.nodes().size());
  for (std::uint32_t tensor = 0; tensor < source.tensor_count(); ++tensor)
    result.initial_source_to_graph_.push_back(ir::TensorId{tensor});
  for (std::uint32_t node = 0; node < source.nodes().size(); ++node) {
    result.initial_graph_node_to_source_.push_back(ir::NodeId{node});
    auto singleton = result.origins_.AddSingleton(ir::NodeId{node}, diagnostic);
    if (!singleton) return std::unexpected(std::move(singleton.error()));
    result.initial_node_origins_.push_back(*singleton);
  }
  result.source_graph_ = &source;
  return result;
}

bool ComparisonDiscoverySession::Interested(ir::OpKey operation) const {
  if (!enabled_) return false;
  return harness_all_ || std::ranges::find(operation_filter_, operation) != operation_filter_.end();
}

std::expected<void, PlanFailure> ComparisonDiscoverySession::Record(
    ComparisonSiteKind kind, std::string_view pattern, std::uint16_t version, ir::OpKey operation,
    std::span<const ir::NodeId> matched_nodes, std::span<const ComparisonOriginSetId> node_origins,
    const ir::Graph& graph, std::span<const ir::TensorId> source_to_graph,
    ir::DiagnosticAdmission* diagnostic) {
  if (!Interested(operation)) return {};
  const auto stale = [&](const char* detail) {
    return Failure(PlanError::kStaleComparisonSite, detail, diagnostic);
  };
  if (source_graph_ == nullptr || source_graph_->identity() != source_identity_ ||
      matched_nodes.empty() || node_origins.size() != graph.nodes().size() ||
      source_to_graph.size() != source_graph_->tensor_count())
    return stale("comparison discovery lost its source or accepted-site mapping");
  for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
    const auto node = matched_nodes[i];
    if (!node.valid() || node.v >= graph.nodes().size())
      return stale("comparison discovery accepted an invalid graph member");
    const auto prior_nodes = matched_nodes.first(i);
    if (std::ranges::find(prior_nodes, node) != prior_nodes.end())
      return stale("comparison discovery accepted a repeated graph member");
  }

  std::uint64_t origin_count = 0;
  for (const auto node : matched_nodes) {
    const auto count = origins_.OriginCount(node_origins[node.v]);
    if (!count || *count == 0 || *count > std::numeric_limits<std::uint32_t>::max() - origin_count)
      return stale("comparison discovery member lineage is invalid");
    origin_count += *count;
  }
  std::uint64_t port_count = 0;
  const auto is_member = [&](ir::NodeId candidate) {
    return std::ranges::find(matched_nodes, candidate) != matched_nodes.end();
  };
  const auto crosses = [&](ir::TensorId tensor) {
    return ObservedSiteOutput(graph, matched_nodes, tensor);
  };
  for (const auto member : matched_nodes) {
    const auto& node = graph.node(member);
    for (const auto tensor : node.sources()) {
      if (tensor.valid() && !is_member(graph.producer(tensor))) ++port_count;
    }
    for (const auto tensor : node.outputs()) {
      if (crosses(tensor)) ++port_count;
    }
  }
  if (matched_nodes.size() > UINT32_MAX || port_count > UINT32_MAX)
    return Failure(PlanError::kAdmission, "comparison discovery ticket exceeds compact IDs",
                   diagnostic);

  std::uint64_t bytes = 0;
  if (!AddBytes(1, sizeof(ComparisonDiscoveryTicket), bytes) ||
      !AddBytes(matched_nodes.size() + 1, sizeof(std::uint32_t), bytes) ||
      !AddBytes(origin_count, sizeof(ir::NodeId), bytes) ||
      !AddBytes(matched_nodes.size(), sizeof(std::uint8_t), bytes) ||
      !AddBytes(port_count, sizeof(ComparisonSitePortRef), bytes))
    return Failure(PlanError::kAdmission, "comparison discovery ticket inventory overflows",
                   diagnostic);
  std::uint64_t ticket_capacity = tickets_.capacity();
  if (tickets_.size() == ticket_capacity) {
    const auto next = ticket_capacity == 0 ? std::uint64_t{1} : ticket_capacity * 2;
    if (next < ticket_capacity || !AddBytes(next, sizeof(ComparisonDiscoveryTicket), bytes))
      return Failure(PlanError::kAdmission, "comparison discovery ticket capacity overflows",
                     diagnostic);
    ticket_capacity = next;
  }
  if (!loan_.Grow(bytes))
    return Failure(PlanError::kAdmission, "comparison discovery ticket payment was refused",
                   diagnostic);

  ComparisonDiscoveryTicket ticket;
  ticket.checkpoint_match = CheckpointMatchIdentity(graph, matched_nodes);
  ticket.kind = kind;
  ticket.pattern = pattern;
  ticket.version = version;
  ticket.registered_operation = operation;
  ticket.ordinal = next_ordinal_;
  ticket.generation = generation_;
  ticket.member_offsets.reserve(matched_nodes.size() + 1);
  ticket.member_origins.reserve(static_cast<std::size_t>(origin_count));
  ticket.owner_segments.reserve(matched_nodes.size());
  ticket.boundary_ports.reserve(static_cast<std::size_t>(port_count));
  ticket.member_offsets.push_back(0);
  for (const auto node_id : matched_nodes) {
    const auto count = origins_.OriginCount(node_origins[node_id.v]);
    if (!count) return stale("comparison discovery lineage changed while recording");
    const std::size_t old = ticket.member_origins.size();
    ticket.member_origins.resize(old + *count);
    std::size_t written = 0;
    if (!origins_.CopyOrigins(node_origins[node_id.v],
                              std::span(ticket.member_origins).subspan(old, *count), written) ||
        written != *count)
      return stale("comparison discovery could not retain member lineage");
    ticket.member_offsets.push_back(static_cast<std::uint32_t>(ticket.member_origins.size()));
    const auto& current_node = graph.node(node_id);
    const auto segment =
        current_node.nout == 0 ? ir::kJoinedSegment : graph.tensor(current_node.out[0]).segment;
    for (std::size_t i = old; i < ticket.member_origins.size(); ++i) {
      const auto source_id = ticket.member_origins[i];
      if (!source_id.valid() || source_id.v >= source_graph_->nodes().size())
        return stale("comparison discovery source member is out of range");
      const auto& source_node = source_graph_->node(source_id);
      const auto source_segment = source_node.nout == 0
                                      ? ir::kJoinedSegment
                                      : source_graph_->tensor(source_node.out[0]).segment;
      // A fused member can retain joined views and per-owner arithmetic in
      // one lineage. Keep singleton source ownership exact; grouped ownership
      // is authenticated against the actual accepted checkpoint member.
      if (*count == 1 && source_segment != segment)
        return stale("comparison discovery member owner facts disagree");
    }
    ticket.owner_segments.push_back(segment);
  }
  const auto source_tensor = [&](ir::TensorId current) -> std::optional<ir::TensorId> {
    ir::TensorId found{};
    for (std::uint32_t i = 0; i < source_to_graph.size(); ++i) {
      if (source_to_graph[i] != current) continue;
      if (found.valid()) return std::nullopt;
      found = ir::TensorId{i};
    }
    if (!found.valid() || found.v >= source_graph_->tensor_count() || !current.valid() ||
        current.v >= graph.tensor_count())
      return std::nullopt;
    const auto source_root = source_graph_->root(found);
    const auto current_root = graph.root(current);
    if (!source_root.valid() || source_root.v >= source_to_graph.size() ||
        source_to_graph[source_root.v] != current_root ||
        source_graph_->tensor(found).view_offset != graph.tensor(current).view_offset)
      return std::nullopt;
    return found;
  };
  const auto append_port = [&](std::uint32_t member, ComparisonSitePortKind port_kind,
                               std::size_t ordinal, ir::TensorId tensor) {
    if (ordinal > UINT16_MAX) {
      ticket.boundary_representable = false;
      return true;
    }
    const auto original = source_tensor(tensor);
    if (!original) {
      ticket.boundary_representable = false;
      return true;
    }
    ticket.boundary_ports.push_back({.member = member,
                                     .kind = port_kind,
                                     .ordinal = static_cast<std::uint16_t>(ordinal),
                                     .original_tensor = *original,
                                     .original_root = source_graph_->root(*original),
                                     .view_offset = source_graph_->tensor(*original).view_offset});
    return true;
  };
  for (std::uint32_t member = 0; member < matched_nodes.size(); ++member) {
    const auto& node = graph.node(matched_nodes[member]);
    for (std::size_t ordinal = 0; ordinal < node.nsrc; ++ordinal) {
      const auto tensor = node.source(ordinal);
      if (tensor.valid() && !is_member(graph.producer(tensor)) &&
          !append_port(member, ComparisonSitePortKind::kInput, ordinal, tensor))
        return stale("comparison discovery input boundary could not be retained");
    }
  }
  for (std::uint32_t member = 0; member < matched_nodes.size(); ++member) {
    const auto& node = graph.node(matched_nodes[member]);
    for (std::size_t ordinal = 0; ordinal < node.nout; ++ordinal) {
      const auto tensor = node.out[ordinal];
      if (crosses(tensor))
        if (!append_port(member, ComparisonSitePortKind::kOutput, ordinal, tensor))
          return stale("comparison discovery output boundary could not be retained");
    }
  }
  ticket.boundary_representable =
      ticket.boundary_representable && ticket.boundary_ports.size() == port_count;
  base::Sha256 hash;
  hash.Update("llmp.comparison.discovery-occurrence.v1\n");
  HashInteger(hash, source_identity_);
  HashInteger(hash, ticket.ordinal);
  HashInteger(hash, ticket.generation);
  HashInteger(hash, static_cast<std::uint8_t>(kind));
  HashInteger(hash, version);
  HashInteger(hash, static_cast<std::uint8_t>(operation.op));
  HashInteger(hash, static_cast<std::uint8_t>(operation.fused));
  HashInteger(hash, static_cast<std::uint8_t>(ticket.boundary_representable));
  hash.Update(pattern);
  for (const auto offset : ticket.member_offsets) HashInteger(hash, offset);
  for (const auto owner : ticket.owner_segments) HashInteger(hash, owner);
  for (const auto& port : ticket.boundary_ports) {
    HashInteger(hash, port.member);
    HashInteger(hash, static_cast<std::uint8_t>(port.kind));
    HashInteger(hash, port.ordinal);
    HashInteger(hash, port.view_offset);
  }
  ticket.occurrence_identity = hash.Finish();
  if (tickets_.capacity() < ticket_capacity)
    tickets_.reserve(static_cast<std::size_t>(ticket_capacity));
  tickets_.push_back(std::move(ticket));
  ++next_ordinal_;
  ++generation_;
  return {};
}

std::expected<void, PlanFailure> ComparisonDiscoverySession::Finalize(
    const ir::Graph& planned_graph, std::span<const ir::TensorId> source_to_planned,
    std::span<const ComparisonOriginSetId> planned_node_origins,
    ir::DiagnosticAdmission* diagnostic) {
  if (!enabled_) return {};
  if (source_graph_ == nullptr || source_graph_->identity() != source_identity_ ||
      source_to_planned.size() != source_graph_->tensor_count() || planned_graph.identity() == 0 ||
      planned_node_origins.size() != planned_graph.nodes().size())
    return Failure(PlanError::kStaleComparisonSite, "comparison discovery final mapping is stale",
                   diagnostic);
  for (const ir::TensorId mapped : source_to_planned)
    if (mapped.valid() && mapped.v >= planned_graph.tensor_count())
      return Failure(PlanError::kStaleComparisonSite,
                     "comparison discovery final tensor map is invalid", diagnostic);
  for (const ComparisonOriginSetId set : planned_node_origins) {
    const auto count = origins_.OriginCount(set);
    if (!count || *count == 0)
      return Failure(PlanError::kStaleComparisonSite,
                     "comparison discovery final node lineage is invalid", diagnostic);
  }
  std::uint64_t bytes = 0;
  if (!AddBytes(source_to_planned.size(), sizeof(ir::TensorId), bytes) ||
      !AddBytes(planned_node_origins.size(), sizeof(ComparisonOriginSetId), bytes) ||
      !loan_.Grow(bytes))
    return Failure(PlanError::kAdmission, "comparison discovery final lineage payment refused",
                   diagnostic);
  source_to_planned_.reserve(source_to_planned.size());
  planned_node_origins_.reserve(planned_node_origins.size());
  source_to_planned_.insert(source_to_planned_.end(), source_to_planned.begin(),
                            source_to_planned.end());
  planned_node_origins_.insert(planned_node_origins_.end(), planned_node_origins.begin(),
                               planned_node_origins.end());
  std::vector<ir::TensorId>().swap(initial_source_to_graph_);
  std::vector<ir::NodeId>().swap(initial_graph_node_to_source_);
  std::vector<ComparisonOriginSetId>().swap(initial_node_origins_);
  seed_loan_.Reset();
  source_graph_ = nullptr;
  operation_filter_ = {};
  harness_all_ = false;
  return {};
}

PlannerMetadataAdmission::PlannerMetadataAdmission(
    std::shared_ptr<const ir::StorageFunder> funder) {
  auto loan = ir::AcquireStorageLoan(funder, 0);
  if (loan)
    loan_ = std::move(*loan);
  else
    refused_ = true;
}

std::expected<void, PlanFailure> PlannerMetadataAdmission::Require(PlannerMetadataPhase phase,
                                                                   std::uint64_t bytes) {
  const auto denied = [&]() -> std::expected<void, PlanFailure> {
    refused_ = true;
    return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  };
  const auto index = static_cast<std::size_t>(phase);
  if (refused_ || index >= phases_.size()) {
    return denied();
  }
  if (bytes <= phases_[index]) return {};
  std::uint64_t required = 0;
  for (std::size_t i = 0; i < phases_.size(); ++i) {
    const auto amount = i == index ? bytes : phases_[i];
    if (amount > std::numeric_limits<std::uint64_t>::max() - required) {
      return denied();
    }
    required += amount;
  }
  const auto current = loan_.bytes();
  // A live phase may be lowered after its trial objects have been destroyed.
  // The shared loan remains at its high-water mark; only grow when the new
  // simultaneous-live requirement exceeds that retained payment.
  if (required > current && !loan_.Grow(required - current)) {
    return denied();
  }
  phases_[index] = bytes;
  return {};
}

std::expected<void, PlanFailure> PlannerMetadataAdmission::SetMatchDetailLive(std::uint64_t bytes) {
  const auto index = static_cast<std::size_t>(PlannerMetadataPhase::kMatchDetail);
  if (refused_ || index >= phases_.size())
    return std::unexpected(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  if (bytes >= phases_[index]) {
    return Require(PlannerMetadataPhase::kMatchDetail, bytes);
  }
  // Lowering represents objects already destroyed by the caller. Do not
  // refund the high-water loan and do not alter cumulative diagnostic bytes.
  phases_[index] = bytes;
  return {};
}

bool PlannerMetadataAdmission::TryChargeDiagnostic(std::uint64_t bytes) {
  if (refused_ || bytes > std::numeric_limits<std::uint64_t>::max() - diagnostic_bytes_) {
    refused_ = true;
    return false;
  }
  const std::uint64_t total = diagnostic_bytes_ + bytes;
  if (!Require(PlannerMetadataPhase::kDiagnostic, total)) return false;
  diagnostic_bytes_ = total;
  return true;
}

std::expected<DiscoveredComparisonPlan, OwnedComparisonFailure> DiscoverComparisonPlan(
    const components::ComparisonBoundaryView& source, std::span<const ir::OpKey> operation_filter,
    bool harness_all, std::shared_ptr<const ir::StorageFunder> funder,
    const execution::Registry& registry, const DeviceFacts& device, const Policy& policy,
    const PlanOptions& options) {
  PlannerMetadataAdmission metadata(funder);
  const auto wrap_failure =
      [&metadata](PlanFailure&& failure) -> std::unexpected<OwnedComparisonFailure> {
    return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(failure)));
  };
  if (!funder || metadata.refused())
    return wrap_failure(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  auto discovery_result = ComparisonDiscoverySession::Create(
      source.io.graph, funder, operation_filter, harness_all, &metadata);
  if (!discovery_result) return wrap_failure(std::move(discovery_result.error()));
  ComparisonDiscoverySession discovery = std::move(*discovery_result);
  auto graph = source.io.graph.Clone(funder, &metadata);
  if (!graph)
    return wrap_failure(PlanFailure{.error = graph.error().error == ir::GraphError::kAdmission
                                                 ? PlanError::kAdmission
                                                 : PlanError::kPattern,
                                    .detail = std::move(graph.error().detail)});
  PlanOptions planned_options = BorrowedOptions(options);
  planned_options.metadata_admission = &metadata;
  ComparisonSiteSelector no_selector;
  std::optional<ComparisonCaptureSession> capture;
  if (discovery.enabled()) {
    capture.emplace(no_selector, discovery.origins(), funder, false, &discovery);
    planned_options.comparison_capture = &*capture;
    planned_options.comparison_origins = &discovery.origins();
    planned_options.comparison_node_origins = discovery.initial_node_origins();
    planned_options.comparison_source_to_graph = discovery.initial_source_to_graph();
    planned_options.comparison_graph_node_to_source = discovery.initial_graph_node_to_source();
  }
  auto planned = Plan(std::move(*graph), registry, device, policy, planned_options);
  if (!planned) return wrap_failure(std::move(planned.error()));
  if (discovery.enabled() &&
      (discovery.source_to_planned().size() != planned->tensors.size() ||
       discovery.planned_node_origins().size() != planned->graph.nodes().size()))
    return wrap_failure(PlanFailure{.error = PlanError::kStaleComparisonSite, .detail = {}});
  OwnedComparisonPlan incumbent(std::move(metadata), std::move(*planned));
  return DiscoveredComparisonPlan(std::move(discovery), std::move(incumbent));
}

static std::expected<ComparisonInventory, OwnedComparisonFailure> BoundComparisonCore(
    const components::ComparisonBoundaryView& source, const ComparisonSiteSelector& requested,
    std::shared_ptr<const ir::StorageFunder> funder, const execution::Registry& registry,
    const DeviceFacts& device, const Policy& policy, const PlanOptions& options,
    const StepShape* shape_override, const ComparisonDecisionLedger* decisions,
    const ComparisonPlanningEnvironment* environment, bool whole_checkpoint,
    bool recapture_after_decisions = false) {
  PlannerMetadataAdmission metadata(funder);
  const auto fail = [&metadata](PlanError error,
                                const char* detail) -> std::unexpected<OwnedComparisonFailure> {
    PlanFailure failure{.error = error, .detail = {}};
    if (error != PlanError::kAdmission) {
      auto text = ir::FormatDiagnostic(&metadata, "{}", std::string_view(detail));
      if (!text) {
        failure.error = PlanError::kAdmission;
      } else {
        failure.detail = std::move(*text);
      }
    }
    return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(failure)));
  };
  const auto wrap = [&metadata](PlanFailure&& failure) -> std::unexpected<OwnedComparisonFailure> {
    return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(failure)));
  };
  if (whole_checkpoint != HasCheckpointMatch(requested) ||
      (whole_checkpoint && requested.kind != ComparisonSiteKind::kRewrite))
    return fail(PlanError::kStaleComparisonSite,
                "checkpoint-only selectors require explicit whole-program comparison");
  if (!funder || metadata.refused() || source.io.graph.identity() == 0 ||
      requested.source_graph_identity != source.io.graph.identity())
    return fail(funder ? PlanError::kStaleComparisonSite : PlanError::kAdmission,
                "the comparison selector does not name this live source graph");
  if (decisions != nullptr) {
    if (environment == nullptr || &source != &environment->source ||
        &registry != &environment->registry || &device != &environment->device ||
        &policy != &environment->policy || &options != &environment->options ||
        shape_override != environment->shape_override)
      return fail(PlanError::kStaleComparisonSite,
                  "committed comparison planning requires its original environment");
    auto valid = decisions->Validate(*environment, &metadata);
    if (!valid) return wrap(std::move(valid.error()));
  }
  auto source_node = RequiredSiteNode(source.io.graph, requested, &metadata);
  if (!source_node) return wrap(std::move(source_node.error()));
  auto boundary = components::MakeComparisonBoundary(source, funder, &metadata);
  if (!boundary) {
    const auto& error = boundary.error();
    return wrap(PlanFailure{.error = error.code == components::ComparisonBoundaryError::kAdmission
                                         ? PlanError::kAdmission
                                         : PlanError::kPattern,
                            .detail = std::move(boundary.error().detail)});
  }
  if (boundary->source_identity() != requested.source_graph_identity)
    return fail(PlanError::kStaleComparisonSite,
                "the comparison boundary authenticated another source graph");

  const auto site_bytes = SiteStorageBytes(requested, source_node->pattern->name);
  if (!site_bytes) return fail(PlanError::kAdmission, "comparison site metadata overflows");
  auto site_loan = ir::AcquireStorageLoan(funder, *site_bytes);
  if (!site_loan) return fail(PlanError::kAdmission, "comparison site metadata was refused");
  ComparisonSiteSelector canonical;
  canonical.source_graph_identity = requested.source_graph_identity;
  canonical.kind = requested.kind;
  canonical.pattern = source_node->pattern->name;
  canonical.version = source_node->pattern->version;
  canonical.member_origins = requested.member_origins;
  canonical.member_owner_segments = requested.member_owner_segments;
  canonical.boundary_ports = requested.boundary_ports;
  canonical.checkpoint_match = requested.checkpoint_match;
  ComparisonSite site{.selector = {},
                      .accepted_member_origins = canonical.member_origins,
                      .accepted_member_owner_segments = canonical.member_owner_segments,
                      .accepted_boundary_ports = canonical.boundary_ports,
                      .complete_member_owner_segments = {},
                      .complete_boundary_ports = {},
                      .registered_result = source_node->pattern->result,
                      .exact_site_identity = {},
                      .excluded_competing_rewrites = {}};
  // Move the canonical selector into its final owner after the accepted
  // records have been copied.  This keeps the two-copy inventory in
  // SiteStorageBytes true for the entire synchronous planning call.
  site.selector = std::move(canonical);

  const auto source_nodes = source.io.graph.nodes().size();
  if (source_nodes > std::numeric_limits<std::uint32_t>::max())
    return fail(PlanError::kAdmission, "comparison source node count exceeds lineage IDs");
  const auto map_bytes = MappingBytes(source.io.graph.tensor_count(), source_nodes);
  if (!map_bytes) return fail(PlanError::kAdmission, "comparison checkpoint map overflows");
  auto map_loan = ir::AcquireStorageLoan(funder, *map_bytes);
  if (!map_loan) return fail(PlanError::kAdmission, "comparison checkpoint maps were refused");
  auto checkpoint_graph = source.io.graph.Clone(funder, &metadata);
  if (!checkpoint_graph)
    return wrap(PlanFailure{.error = checkpoint_graph.error().error == ir::GraphError::kAdmission
                                         ? PlanError::kAdmission
                                         : PlanError::kPattern,
                            .detail = std::move(checkpoint_graph.error().detail)});
  auto origins = ComparisonOriginDag::Create(funder, static_cast<std::uint32_t>(source_nodes), 0, 0,
                                             &metadata);
  if (!origins) return wrap(std::move(origins.error()));
  std::vector<ir::TensorId> source_to_checkpoint;
  std::vector<ComparisonOriginSetId> checkpoint_node_origins;
  source_to_checkpoint.reserve(source.io.graph.tensor_count());
  checkpoint_node_origins.reserve(source_nodes);
  for (std::uint32_t t = 0; t < source.io.graph.tensor_count(); ++t)
    source_to_checkpoint.push_back(ir::TensorId{t});
  for (std::uint32_t n = 0; n < source_nodes; ++n) {
    auto singleton = origins->AddSingleton(ir::NodeId{n}, &metadata);
    if (!singleton) return wrap(std::move(singleton.error()));
    checkpoint_node_origins.push_back(*singleton);
  }
  if (!BoundaryMatches(*boundary, *checkpoint_graph, source_to_checkpoint, &metadata))
    return fail(PlanError::kStaleComparisonSite,
                "the original source boundary does not map to its owned checkpoint");
  std::vector<ir::NodeId> checkpoint_node_to_source;
  checkpoint_node_to_source.reserve(source_nodes);
  for (std::uint32_t n = 0; n < source_nodes; ++n)
    checkpoint_node_to_source.push_back(ir::NodeId{n});
  std::optional<ComparisonCheckpoint> checkpoint;
  checkpoint.emplace(std::move(*map_loan), std::move(*checkpoint_graph),
                     std::move(source_to_checkpoint), std::move(checkpoint_node_to_source),
                     std::move(checkpoint_node_origins), std::move(*origins));

  auto incumbent_graph = source.io.graph.Clone(funder, &metadata);
  if (!incumbent_graph)
    return wrap(PlanFailure{.error = incumbent_graph.error().error == ir::GraphError::kAdmission
                                         ? PlanError::kAdmission
                                         : PlanError::kPattern,
                            .detail = std::move(incumbent_graph.error().detail)});
  auto planning_origins = checkpoint->origins.Clone(funder, &metadata);
  if (!planning_origins) return wrap(std::move(planning_origins.error()));
  constexpr std::uint64_t kBitWordBits = std::numeric_limits<unsigned long>::digits;
  constexpr std::uint64_t kVectorEnvelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  if (source_nodes > std::numeric_limits<std::uint64_t>::max() - (kBitWordBits - 1))
    return fail(PlanError::kAdmission, "committed expansion map overflows");
  const auto expand_words = (source_nodes + kBitWordBits - 1) / kBitWordBits;
  if (expand_words >
      (std::numeric_limits<std::uint64_t>::max() - kVectorEnvelope) / sizeof(unsigned long))
    return fail(PlanError::kAdmission, "committed expansion map overflows");
  const auto expand_bytes =
      expand_words * sizeof(unsigned long) + (source_nodes == 0 ? 0 : kVectorEnvelope);
  auto expand_loan = ir::AcquireStorageLoan(funder, expand_bytes);
  if (!expand_loan) return fail(PlanError::kAdmission, "committed expansion map was refused");
  std::vector<bool> expand(source_nodes, false);
  if (decisions != nullptr) {
    for (const auto& decision : decisions->records()) {
      if (decision.kind != ComparisonSiteKind::kFusedNode &&
          decision.kind != ComparisonSiteKind::kCompositeNode)
        continue;
      const Pattern* registered = FindPattern(decision.pattern);
      if (registered == nullptr || registered->version != decision.version ||
          registered->result != decision.registered_result || decision.member_origins.size() != 1)
        return fail(PlanError::kStaleComparisonSite,
                    "a committed fused directive no longer matches its registration");
      const auto& members = decision.member_origins.front();
      if (members.empty())
        return fail(PlanError::kStaleComparisonSite,
                    "a committed fused directive has no original members");
      for (const auto member : members) {
        if (!member.valid() || member.v >= source_nodes ||
            source.io.graph.node(member).key != decision.registered_result)
          return fail(PlanError::kStaleComparisonSite,
                      "a committed fused directive no longer names its original node");
        expand[member.v] = true;
      }
    }
  }
  auto seed = CopyExpand(*incumbent_graph, expand, nullptr, nullptr,
                         checkpoint->checkpoint_node_origins(), &*planning_origins, &metadata);
  if (!seed)
    return wrap(PlanFailure{.error = seed.error().error == ir::GraphError::kAdmission
                                         ? PlanError::kAdmission
                                         : PlanError::kPattern,
                            .detail = std::move(seed.error().detail)});
  if (requested.kind == ComparisonSiteKind::kFusedNode ||
      requested.kind == ComparisonSiteKind::kCompositeNode) {
    const auto saved_map_bytes =
        MappingBytes(seed->input_to_seed.size(), seed->seed_to_input.size());
    if (!saved_map_bytes)
      return fail(PlanError::kAdmission, "committed checkpoint mapping overflows");
    auto saved_map_loan = ir::AcquireStorageLoan(funder, *saved_map_bytes);
    if (!saved_map_loan)
      return fail(PlanError::kAdmission, "committed checkpoint mapping was refused");
    auto saved_graph = seed->graph.Clone(funder, &metadata);
    if (!saved_graph)
      return wrap(PlanFailure{.error = saved_graph.error().error == ir::GraphError::kAdmission
                                           ? PlanError::kAdmission
                                           : PlanError::kPattern,
                              .detail = std::move(saved_graph.error().detail)});
    auto saved_origins = planning_origins->Clone(funder, &metadata);
    if (!saved_origins) return wrap(std::move(saved_origins.error()));
    std::vector<ir::TensorId> saved_tensor_map(seed->input_to_seed.begin(),
                                               seed->input_to_seed.end());
    std::vector<ir::NodeId> saved_node_map(seed->seed_to_input.begin(), seed->seed_to_input.end());
    std::vector<ComparisonOriginSetId> saved_node_origins(seed->seed_origin_sets.begin(),
                                                          seed->seed_origin_sets.end());
    checkpoint.reset();
    checkpoint.emplace(std::move(*saved_map_loan), std::move(*saved_graph),
                       std::move(saved_tensor_map), std::move(saved_node_map),
                       std::move(saved_node_origins), std::move(*saved_origins));
  }
  const auto incumbent_input =
      ComparisonPlanInputMetadataBytes(seed->graph, registry, policy, options);
  if (!incumbent_input) return wrap(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  if (auto paid = metadata.Require(PlannerMetadataPhase::kInput, *incumbent_input); !paid)
    return wrap(std::move(paid.error()));
  PlanOptions incumbent_options = BorrowedOptions(options);
  if (shape_override) incumbent_options.shape = shape_override;
  incumbent_options.comparison_source_to_graph = seed->input_to_seed;
  incumbent_options.comparison_graph_node_to_source = seed->seed_to_input;
  if (auto mapped = MapComparisonPlanInputs(incumbent_options, seed->input_to_seed,
                                            seed->seed_to_input, metadata);
      !mapped)
    return wrap(std::move(mapped.error()));
  ComparisonFinalNodeOrigins final_lineage(source.io.graph.identity());
  std::optional<ComparisonCaptureSession> capture;
  capture.emplace(
      site.selector, *planning_origins, funder, false, nullptr,
      decisions == nullptr ? std::span<const ComparisonDecisionRecord>{} : decisions->records());
  if (auto journal = capture->PrepareDecisionJournal(funder); !journal)
    return wrap(std::move(journal.error()));
  if (recapture_after_decisions) {
    if (!whole_checkpoint || decisions == nullptr || decisions->generation() == 0)
      return fail(PlanError::kStaleComparisonSite,
                  "fresh whole capture requires validated prior decisions");
    ComparisonCheckpointAuthorityAccess::RecaptureAfterDecisions(*capture);
  }
  incumbent_options.comparison_capture = &*capture;
  incumbent_options.comparison_origins = &*planning_origins;
  incumbent_options.comparison_node_origins = seed->seed_origin_sets;
  incumbent_options.metadata_admission = &metadata;
  incumbent_options.comparison_final_lineage = &final_lineage;
  auto incumbent_plan = Plan(std::move(seed->graph), registry, device, policy, incumbent_options);
  if (!incumbent_plan) return wrap(std::move(incumbent_plan.error()));
  if (capture->record_refused())
    return fail(PlanError::kAdmission, "comparison decision exclusion journal was refused");
  constexpr std::uint64_t vector_envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  const std::uint64_t source_tensor_count = seed->input_to_seed.size();
  if (source_tensor_count >
      (std::numeric_limits<std::uint64_t>::max() - vector_envelope) / sizeof(ir::TensorId))
    return fail(PlanError::kAdmission, "composed comparison source map overflows");
  const std::uint64_t composed_map_bytes =
      source_tensor_count * sizeof(ir::TensorId) + (source_tensor_count == 0 ? 0 : vector_envelope);
  if (auto paid = metadata.Require(PlannerMetadataPhase::kAdopt, composed_map_bytes); !paid)
    return wrap(std::move(paid.error()));
  std::vector<ir::TensorId> source_to_planned;
  source_to_planned.reserve(static_cast<std::size_t>(source_tensor_count));
  for (const auto seed_tensor : seed->input_to_seed) {
    if (!seed_tensor.valid()) {
      source_to_planned.push_back(ir::TensorId{});
      continue;
    }
    if (seed_tensor.v >= incumbent_plan->tensors.size())
      return fail(PlanError::kStaleComparisonSite, "normal planning lost a source tensor mapping");
    source_to_planned.push_back(incumbent_plan->tensors[seed_tensor.v]);
  }
  if (!BoundaryMatches(*boundary, incumbent_plan->graph, source_to_planned, &metadata))
    return fail(PlanError::kStaleComparisonSite,
                "normal planning changed the declared source boundary");
  if (requested.kind == ComparisonSiteKind::kRewrite ||
      requested.kind == ComparisonSiteKind::kBatch) {
    auto captured = capture->TakeCheckpoint(&metadata);
    if (!captured) return wrap(std::move(captured.error()));
    checkpoint.emplace(std::move(*captured));
    if (recapture_after_decisions)
      site.selector.checkpoint_match =
          CheckpointMatchIdentity(checkpoint->graph, checkpoint->captured_members());
  }
  if (requested.kind == ComparisonSiteKind::kRewrite ||
      requested.kind == ComparisonSiteKind::kBatch) {
    auto complete =
        DeriveCompleteSiteIdentity(source.io.graph, *checkpoint, site, *site_loan, &metadata);
    if (!complete) return wrap(std::move(complete.error()));
  } else if (requested.kind == ComparisonSiteKind::kFusedNode ||
             requested.kind == ComparisonSiteKind::kCompositeNode) {
    auto complete = DeriveCompleteSourceNodeIdentity(source.io.graph, source_node->node, site,
                                                     *site_loan, &metadata);
    if (!complete) return wrap(std::move(complete.error()));
  }
  const auto& source_node_desc = source.io.graph.node(source_node->node);
  if (whole_checkpoint) {
    bool selected = false;
    for (const auto& port : site.complete_boundary_ports) {
      if (port.kind != ComparisonSitePortKind::kOutput) continue;
      auto output =
          checkpoint->mapped_output(port.original_tensor, incumbent_plan->graph.identity());
      if (!output.valid() || output.v >= incumbent_plan->graph.tensor_count()) continue;
      auto producer = incumbent_plan->graph.producer(output);
      // Follow the actual output's alias chain, never another same-origin node.
      for (std::size_t depth = 0; depth < incumbent_plan->graph.nodes().size(); ++depth) {
        if (!producer.valid() || producer.v >= incumbent_plan->graph.nodes().size()) break;
        const auto& node = incumbent_plan->graph.node(producer);
        if (source_node->pattern->AcceptsResult(node.key)) {
          selected = std::ranges::any_of(incumbent_plan->steps, [&](const auto& step) {
            return std::ranges::contains(step.nodes, producer);
          });
          break;
        }
        if (node.nsrc != 1 || node.nout != 1 || node.out[0] != output ||
            (node.key != ir::Primitive(ir::Op::kReshape) &&
             node.key != ir::Primitive(ir::Op::kView) &&
             node.key != ir::Primitive(ir::Op::kPermute)) ||
            incumbent_plan->graph.root(output) != incumbent_plan->graph.root(node.source(0)))
          break;
        output = node.source(0);
        producer = incumbent_plan->graph.producer(output);
      }
      if (selected) break;
    }
    if (!selected)
      return fail(PlanError::kMissingComparisonSite,
                  "the whole checkpoint output has no selected registered primary");
  }
  if (requested.kind == ComparisonSiteKind::kRewrite && !whole_checkpoint) {
    const auto selected_rewrite = [&](const Applied& applied) {
      if (applied.pattern != source_node->pattern->name) return false;
      const bool singleton_members = std::ranges::all_of(
          requested.member_origins, [](const ComparisonOriginSet& set) { return set.size() == 1; });
      if (singleton_members) {
        if (applied.replaced.size() != requested.member_origins.size()) return false;
        for (std::size_t i = 0; i < applied.replaced.size(); ++i) {
          const auto seed_node = applied.replaced[i];
          if (!seed_node.valid() || seed_node.v >= seed->seed_to_input.size() ||
              seed->seed_to_input[seed_node.v] != requested.member_origins[i][0])
            return false;
        }
      } else if (std::ranges::count_if(incumbent_plan->applied, [&](const Applied& candidate) {
                   return candidate.pattern == source_node->pattern->name;
                 }) != 1) {
        // Applied::replaced is one-to-one attribution, never a rewrite
        // preimage. Without an exact nested applied identity, refuse rather
        // than credit another same-pattern rewrite.
        return false;
      }
      for (const auto made : applied.made) {
        if (!made.valid() || made.v >= incumbent_plan->graph.nodes().size() ||
            !source_node->pattern->AcceptsResult(incumbent_plan->graph.node(made).key))
          continue;
        for (const auto& step : incumbent_plan->steps)
          if (std::ranges::contains(step.nodes, made)) return true;
      }
      // A later round may replace an emitted node, or fold its preparation
      // into an existing result. Follow the exact accepted output, never an
      // arbitrary same-kind node elsewhere in the final graph.
      for (const auto& port : site.complete_boundary_ports) {
        if (port.kind != ComparisonSitePortKind::kOutput ||
            port.member >= site.accepted_member_origins.size() || !port.original_tensor.valid() ||
            port.original_tensor.v >= source_to_planned.size())
          continue;
        const auto original_producer = source.io.graph.producer(port.original_tensor);
        if (!std::ranges::contains(site.accepted_member_origins[port.member], original_producer))
          continue;
        const auto output = source_to_planned[port.original_tensor.v];
        if (!output.valid() || output.v >= incumbent_plan->graph.tensor_count()) continue;
        const auto producer = incumbent_plan->graph.producer(output);
        if (!producer.valid() || producer.v >= incumbent_plan->graph.nodes().size()) continue;
        const auto& result = incumbent_plan->graph.node(producer);
        if (!source_node->pattern->AcceptsResult(result.key) ||
            !std::ranges::contains(result.outputs(), output))
          continue;
        const bool descendant =
            producer.v < final_lineage.node_origins.size() &&
            planning_origins->Contains(final_lineage.node_origins.values[producer.v],
                                       original_producer);
        // The sole existing-result recipe maps a Prepare/Convert directly
        // to the exact HC fold output authenticated by its captured matcher.
        const bool existing =
            source_node->pattern->existing_result && result.key == source_node->pattern->result;
        if (!descendant && !existing) continue;
        for (const auto& step : incumbent_plan->steps)
          if (std::ranges::contains(step.nodes, producer)) return true;
      }
      return false;
    };
    if (!std::ranges::any_of(incumbent_plan->applied, selected_rewrite))
      return fail(PlanError::kMissingComparisonSite,
                  "the accepted rewrite is not selected in the incumbent plan");
  } else if (requested.kind == ComparisonSiteKind::kBatch) {
    if (!SelectedBatch(*incumbent_plan, site.selector, source_node->pattern->name,
                       *planning_origins, final_lineage.node_origins.values))
      return fail(PlanError::kMissingComparisonSite,
                  "the accepted batch is not selected in the incumbent plan");
  } else if (requested.kind != ComparisonSiteKind::kRewrite &&
             !SelectedNodeThroughSeed(*incumbent_plan, seed->seed_to_input, source_node->node,
                                      source_node_desc.key)) {
    return fail(PlanError::kMissingComparisonSite,
                "the incumbent plan does not select the requested source node");
  }
  // BoundComparison is a public inventory result. Publish its directional
  // maps in the original source namespace, just like PlanAlternative does;
  // callers must not have to infer whether these vectors describe the
  // expanded seed graph.
  const std::uint64_t final_node_count = incumbent_plan->from.size();
  constexpr std::uint64_t map_envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  if (final_node_count >
      (std::numeric_limits<std::uint64_t>::max() - map_envelope) / sizeof(ir::NodeId))
    return fail(PlanError::kAdmission, "incumbent source mapping overflows");
  const std::uint64_t published_map_bytes =
      final_node_count * sizeof(ir::NodeId) + (final_node_count == 0 ? 0 : map_envelope);
  if (auto paid = metadata.Require(PlannerMetadataPhase::kAdopt, published_map_bytes); !paid)
    return wrap(std::move(paid.error()));
  for (auto& node : incumbent_plan->from) {
    if (!node.valid() || node.v >= seed->seed_to_input.size())
      return fail(PlanError::kStaleComparisonSite, "incumbent source node map is out of range");
    node = seed->seed_to_input[node.v];
    if (!node.valid())
      return fail(PlanError::kStaleComparisonSite, "incumbent source node was elided");
  }
  for (auto& applied : incumbent_plan->applied)
    for (auto& node : applied.replaced) {
      if (!node.valid() || node.v >= seed->seed_to_input.size())
        return fail(PlanError::kStaleComparisonSite, "incumbent rewrite node map is out of range");
      node = seed->seed_to_input[node.v];
      if (!node.valid())
        return fail(PlanError::kStaleComparisonSite, "incumbent rewrite node was elided");
    }
  for (auto& applied : incumbent_plan->applied)
    for (auto& proof : applied.numerical_proofs) {
      if (!proof.site.valid() || proof.site.v >= seed->seed_to_input.size())
        return fail(PlanError::kStaleComparisonSite,
                    "incumbent numerical node map is out of range");
      proof.site = seed->seed_to_input[proof.site.v];
      if (!proof.site.valid())
        return fail(PlanError::kStaleComparisonSite, "incumbent numerical site was elided");
    }
  incumbent_plan->whole_checkpoint = site.whole_checkpoint;
  incumbent_plan->tensors = std::move(source_to_planned);
  incumbent_plan->primitive_digest = source.io.graph.Digest();
  capture.reset();
  OwnedComparisonPlan incumbent(std::move(metadata), std::move(*incumbent_plan), {},
                                std::optional<ComparisonOriginDag>(std::move(*planning_origins)),
                                std::move(final_lineage));
  return ComparisonInventory(std::move(*site_loan), std::move(site), std::move(*checkpoint),
                             std::move(incumbent), std::move(*boundary));
}

std::expected<ComparisonInventory, OwnedComparisonFailure> BoundComparison(
    const components::ComparisonBoundaryView& source, const ComparisonSiteSelector& requested,
    std::shared_ptr<const ir::StorageFunder> funder, const execution::Registry& registry,
    const DeviceFacts& device, const Policy& policy, const PlanOptions& options,
    const StepShape* shape_override, const ComparisonDecisionLedger* decisions,
    const ComparisonPlanningEnvironment* environment) {
  return BoundComparisonCore(source, requested, std::move(funder), registry, device, policy,
                             options, shape_override, decisions, environment, false);
}

std::expected<ComparisonInventory, OwnedComparisonFailure> BoundWholeComparison(
    const components::ComparisonBoundaryView& source, const ComparisonSiteSelector& requested,
    std::shared_ptr<const ir::StorageFunder> funder, const execution::Registry& registry,
    const DeviceFacts& device, const Policy& policy, const PlanOptions& options,
    const StepShape* shape_override) {
  return BoundComparisonCore(source, requested, std::move(funder), registry, device, policy,
                             options, shape_override, nullptr, nullptr, true);
}

std::expected<ComparisonTrial, OwnedComparisonFailure> CaptureComparisonTrial(
    const ComparisonPlanningEnvironment& environment, const ComparisonDecisionLedger& decisions,
    const ComparisonSiteSelector& requested, std::shared_ptr<const ir::StorageFunder> funder) {
  auto inventory =
      BoundComparison(environment.source, requested, std::move(funder), environment.registry,
                      environment.device, environment.policy, environment.options,
                      environment.shape_override, &decisions, &environment);
  if (!inventory) return std::unexpected(std::move(inventory.error()));
  // The replay incumbent is validation scratch only. Retain the accepted site,
  // exact pre-match checkpoint, and authenticated source boundary.
  return ComparisonTrial(std::move(inventory->metadata_loan), std::move(inventory->site),
                         std::move(inventory->checkpoint), std::move(inventory->source_boundary),
                         decisions.ledger_identity(), decisions.generation());
}

std::expected<ComparisonTrial, OwnedComparisonFailure> CaptureWholeComparisonTrial(
    const ComparisonPlanningEnvironment& environment, const ComparisonDecisionLedger& decisions,
    const ComparisonSiteSelector& requested, std::shared_ptr<const ir::StorageFunder> funder) {
  // Authenticate the original discovery ticket before interpreting its exact
  // source groups in a changed cumulative context. Both passes use the same
  // payer; the first owner is released before the current capture is built.
  // Generation zero retains the single-pass path.
  if (decisions.generation() != 0) {
    auto valid = decisions.Validate(environment);
    if (!valid) {
      PlannerMetadataAdmission metadata(funder);
      return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(valid.error())));
    }
    auto original = BoundWholeComparison(
        environment.source, requested, funder, environment.registry, environment.device,
        environment.policy, environment.options, environment.shape_override);
    if (!original) return std::unexpected(std::move(original.error()));
  }
  auto inventory = BoundComparisonCore(environment.source, requested, std::move(funder),
                                       environment.registry, environment.device, environment.policy,
                                       environment.options, environment.shape_override, &decisions,
                                       &environment, true, decisions.generation() != 0);
  if (!inventory) return std::unexpected(std::move(inventory.error()));
  // The replay incumbent is validation scratch only. Retain the accepted site,
  // exact pre-match checkpoint, and authenticated source boundary.
  return ComparisonTrial(std::move(inventory->metadata_loan), std::move(inventory->site),
                         std::move(inventory->checkpoint), std::move(inventory->source_boundary),
                         decisions.ledger_identity(), decisions.generation());
}

bool AuthenticComparisonSite(const ComparisonSite& site) {
  const auto ports_equal = std::ranges::equal(
      site.selector.boundary_ports, site.accepted_boundary_ports, [](const auto& a, const auto& b) {
        return a.member == b.member && a.kind == b.kind && a.ordinal == b.ordinal &&
               a.original_tensor == b.original_tensor && a.original_root == b.original_root &&
               a.view_offset == b.view_offset;
      });
  return site.selector.source_graph_identity != 0 &&
         site.selector.member_origins == site.accepted_member_origins &&
         site.selector.member_owner_segments == site.accepted_member_owner_segments &&
         ports_equal && site.exact_site_identity == ExactSiteIdentity(site) &&
         (site.whole_checkpoint.whole_only() == HasCheckpointMatch(site.selector));
}

namespace {
std::expected<GraphAlternative, OwnedComparisonFailure> MakeAlternativeCore(
    const ComparisonCheckpoint& checkpoint, const ComparisonSite& accepted_site,
    const components::OwnedComparisonBoundary& source_boundary) {
  const auto funder = checkpoint.graph.storage_funder();
  PlannerMetadataAdmission metadata(funder);
  const auto fail = [&metadata](PlanError error,
                                const char* detail) -> std::unexpected<OwnedComparisonFailure> {
    PlanFailure failure{.error = error, .detail = {}};
    if (error != PlanError::kAdmission) {
      auto text = ir::FormatDiagnostic(&metadata, "{}", std::string_view(detail));
      if (!text)
        failure.error = PlanError::kAdmission;
      else
        failure.detail = std::move(*text);
    }
    return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(failure)));
  };
  const auto wrap = [&metadata](PlanFailure&& failure) -> std::unexpected<OwnedComparisonFailure> {
    return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(failure)));
  };
  if (!funder || metadata.refused())
    return fail(PlanError::kAdmission, "alternative graph has no paid owner");
  if (accepted_site.selector.source_graph_identity == 0 ||
      accepted_site.selector.source_graph_identity != source_boundary.source_identity() ||
      (accepted_site.selector.kind == ComparisonSiteKind::kRewrite &&
       !AuthenticComparisonSite(accepted_site)))
    return fail(PlanError::kStaleComparisonSite,
                "the comparison inventory no longer authenticates its original graph");
  if (accepted_site.whole_checkpoint.whole_only() &&
      (!accepted_site.whole_checkpoint.belongs_to(source_boundary.source_identity(),
                                                  checkpoint.graph.identity()) ||
       CheckpointMatchIdentity(checkpoint.graph, checkpoint.captured_members()) !=
           accepted_site.selector.checkpoint_match))
    return fail(PlanError::kStaleComparisonSite,
                "whole checkpoint authority has a different graph");
  if (!checkpoint.history_matches_graph())
    return fail(PlanError::kStaleComparisonSite,
                "the checkpoint rewrite history has a different graph owner");
  const Pattern* accepted_pattern = FindPattern(accepted_site.selector.pattern);
  if (accepted_pattern == nullptr || accepted_pattern->version != accepted_site.selector.version ||
      accepted_pattern->result != accepted_site.registered_result)
    return fail(PlanError::kStaleComparisonSite,
                "the accepted result no longer matches its registered pattern identity");
  const std::uint64_t exclusion_capacity =
      accepted_site.selector.kind == ComparisonSiteKind::kRewrite ? [&] {
        const auto patterns = Patterns();
        std::uint64_t count = 0;
        for (std::size_t i = 0; i < patterns.size(); ++i) {
          const auto& candidate = patterns[i];
          if (candidate.kind != PatternKind::kRewrite ||
              (candidate.name == accepted_pattern->name &&
               candidate.version == accepted_pattern->version))
            continue;
          const bool duplicate = std::ranges::any_of(patterns.first(i), [&](const Pattern& prior) {
            return prior.kind == PatternKind::kRewrite && prior.name == candidate.name &&
                   prior.version == candidate.version;
          });
          if (!duplicate) ++count;
        }
        return count;
      }()
                                                                  : 0;
  std::uint64_t exclusion_bytes = 0;
  if (!AddBytes(exclusion_capacity, sizeof(ComparisonSite::ExcludedRewrite), exclusion_bytes))
    return fail(PlanError::kAdmission, "comparison exclusion-record inventory overflows");
  auto exclusion_loan = ir::AcquireStorageLoan(funder, exclusion_bytes);
  if (!exclusion_loan)
    return fail(PlanError::kAdmission, "comparison exclusion-record payment was refused");
  auto graph = checkpoint.graph.Clone(funder, &metadata);
  if (!graph)
    return wrap(PlanFailure{.error = graph.error().error == ir::GraphError::kAdmission
                                         ? PlanError::kAdmission
                                         : PlanError::kPattern,
                            .detail = std::move(graph.error().detail)});
  auto origins = checkpoint.origins.Clone(funder, &metadata);
  if (!origins) return wrap(std::move(origins.error()));
  constexpr std::uint64_t kEnvelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  const std::uint64_t bits = checkpoint.graph.nodes().size();
  constexpr std::uint64_t kBitWordBits = std::numeric_limits<unsigned long>::digits;
  if (bits > std::numeric_limits<std::uint64_t>::max() - (kBitWordBits - 1))
    return fail(PlanError::kAdmission, "alternative selection map size overflows");
  const std::uint64_t words = (bits + kBitWordBits - 1) / kBitWordBits;
  if (words > (std::numeric_limits<std::uint64_t>::max() - kEnvelope) / sizeof(unsigned long))
    return fail(PlanError::kAdmission, "alternative selection map size overflows");
  // Pinned libstdc++ vector<bool> stores rounded unsigned-long words.
  const std::uint64_t expand_bytes = words * sizeof(unsigned long) + (bits == 0 ? 0 : kEnvelope);
  auto expand_loan = ir::AcquireStorageLoan(funder, expand_bytes);
  if (!expand_loan) return fail(PlanError::kAdmission, "alternative selection map was refused");
  std::vector<bool> expand(checkpoint.graph.nodes().size(), false);
  std::size_t expanded_members = 0;
  if (accepted_site.selector.kind == ComparisonSiteKind::kFusedNode ||
      accepted_site.selector.kind == ComparisonSiteKind::kCompositeNode) {
    for (std::size_t member = 0; member < accepted_site.selector.member_origins.size(); ++member) {
      bool found_member = false;
      for (std::uint32_t n = 0; n < checkpoint.graph.nodes().size(); ++n) {
        if (n < checkpoint.checkpoint_node_origins().size() &&
            origins->Equals(checkpoint.checkpoint_node_origins()[n],
                            accepted_site.selector.member_origins[member])) {
          expand[n] = true;
          found_member = true;
        }
      }
      if (!found_member)
        return fail(PlanError::kStaleComparisonSite,
                    "a selected member has no node in the captured checkpoint");
      ++expanded_members;
    }
  } else {
    // Rewrite and batch checkpoints are already the exact pre-match graph;
    // their alternative replays the ordinary continuation with only that
    // accepted site excluded.
    expanded_members = accepted_site.selector.member_origins.size();
  }
  if (expanded_members == 0)
    return fail(PlanError::kStaleComparisonSite, "the captured comparison site is empty");
  auto seed = CopyExpand(*graph, expand, nullptr, nullptr, checkpoint.checkpoint_node_origins(),
                         &*origins, &metadata);
  if (!seed)
    return wrap(PlanFailure{.error = seed.error().error == ir::GraphError::kAdmission
                                         ? PlanError::kAdmission
                                         : PlanError::kPattern,
                            .detail = std::move(seed.error().detail)});
  const auto copy_bytes = SiteStorageBytes(accepted_site.selector, accepted_site.selector.pattern);
  const auto supplement_bytes = SiteSupplementBytes(accepted_site);
  if (!copy_bytes || !supplement_bytes ||
      *supplement_bytes > std::numeric_limits<std::uint64_t>::max() - *copy_bytes ||
      !seed->metadata_loan.Grow(*copy_bytes + *supplement_bytes))
    return fail(PlanError::kAdmission, "alternative site metadata payment was refused");
  ComparisonSite site{
      .whole_checkpoint = accepted_site.whole_checkpoint,
      .selector = accepted_site.selector,
      .accepted_member_origins = accepted_site.accepted_member_origins,
      .accepted_member_owner_segments = accepted_site.accepted_member_owner_segments,
      .accepted_boundary_ports = accepted_site.accepted_boundary_ports,
      .complete_member_owner_segments = accepted_site.complete_member_owner_segments,
      .complete_boundary_ports = accepted_site.complete_boundary_ports,
      .registered_result = accepted_site.registered_result,
      .exact_site_identity = accepted_site.exact_site_identity,
      .excluded_competing_rewrites = {}};
  site.excluded_competing_rewrites.reserve(static_cast<std::size_t>(exclusion_capacity));
  constexpr std::uint64_t vector_envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
  const std::uint64_t tensor_count = checkpoint.source_to_checkpoint().size();
  const std::uint64_t node_count = seed->seed_to_input.size();
  if (tensor_count >
          (std::numeric_limits<std::uint64_t>::max() - vector_envelope) / sizeof(ir::TensorId) ||
      node_count >
          (std::numeric_limits<std::uint64_t>::max() - vector_envelope) / sizeof(ir::NodeId))
    return fail(PlanError::kAdmission, "alternative source maps overflow");
  std::uint64_t added_map_bytes =
      tensor_count * sizeof(ir::TensorId) + (tensor_count == 0 ? 0 : vector_envelope);
  const std::uint64_t added_node_bytes =
      node_count * sizeof(ir::NodeId) + (node_count == 0 ? 0 : vector_envelope);
  if (added_node_bytes > std::numeric_limits<std::uint64_t>::max() - added_map_bytes ||
      added_map_bytes + added_node_bytes > std::numeric_limits<std::uint64_t>::max() ||
      !seed->metadata_loan.Grow(added_map_bytes + added_node_bytes))
    return fail(PlanError::kAdmission, "alternative source-map payment was refused");
  std::vector<ir::TensorId> source_to_alternative;
  source_to_alternative.reserve(static_cast<std::size_t>(tensor_count));
  for (std::size_t source_tensor = 0; source_tensor < checkpoint.source_to_checkpoint().size();
       ++source_tensor) {
    const auto checkpoint_tensor = checkpoint.source_to_checkpoint()[source_tensor];
    if (!checkpoint_tensor.valid()) {
      source_to_alternative.push_back(ir::TensorId{});
      continue;
    }
    if (checkpoint_tensor.v >= seed->input_to_seed.size())
      return fail(PlanError::kStaleComparisonSite,
                  "checkpoint expansion lost a source-tensor mapping");
    source_to_alternative.push_back(seed->input_to_seed[checkpoint_tensor.v]);
  }
  std::vector<ir::NodeId> alternative_to_source;
  alternative_to_source.reserve(seed->seed_to_input.size());
  for (const auto checkpoint_node : seed->seed_to_input) {
    if (!checkpoint_node.valid() ||
        checkpoint_node.v >= checkpoint.checkpoint_node_to_source().size())
      return fail(PlanError::kStaleComparisonSite,
                  "checkpoint expansion lost a source-node attribution");
    alternative_to_source.push_back(checkpoint.checkpoint_node_to_source()[checkpoint_node.v]);
  }
  return ComparisonCheckpointAuthorityAccess::MakeAlternative(
      std::move(seed->metadata_loan), std::move(*exclusion_loan), std::move(seed->graph),
      std::move(source_to_alternative), std::move(alternative_to_source),
      std::move(seed->seed_origin_sets), std::move(*origins), std::vector<Applied>{},
      std::move(site), std::move(seed->seed_to_input), std::move(seed->input_to_seed));
}

}  // namespace

std::expected<GraphAlternative, OwnedComparisonFailure> MakeAlternative(
    const ComparisonInventory& inventory) {
  return MakeAlternativeCore(inventory.checkpoint, inventory.site, inventory.source_boundary);
}

std::expected<GraphAlternative, OwnedComparisonFailure> MakeAlternative(
    const ComparisonTrial& trial) {
  auto alternative = MakeAlternativeCore(trial.checkpoint, trial.site, trial.source_boundary);
  if (alternative) {
    alternative->ledger_identity = trial.ledger_identity;
    alternative->ledger_generation = trial.ledger_generation;
  }
  return alternative;
}

namespace {
std::expected<OwnedComparisonPlan, OwnedComparisonFailure> PlanAlternativeCore(
    const ComparisonCheckpoint& checkpoint, const ComparisonSite& accepted_site,
    const components::OwnedComparisonBoundary& source_boundary, GraphAlternative alternative,
    const execution::Registry& registry, const DeviceFacts& device, const Policy& policy,
    const PlanOptions& options, const StepShape* shape_override,
    std::span<const ComparisonDecisionRecord> decisions) {
  auto metadata_owner = alternative.graph.storage_funder();
  PlannerMetadataAdmission metadata(metadata_owner);
  const auto fail = [&metadata](PlanError error,
                                const char* detail) -> std::unexpected<OwnedComparisonFailure> {
    PlanFailure failure{.error = error, .detail = {}};
    if (error != PlanError::kAdmission) {
      auto text = ir::FormatDiagnostic(&metadata, "{}", std::string_view(detail));
      if (!text)
        failure.error = PlanError::kAdmission;
      else
        failure.detail = std::move(*text);
    }
    return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(failure)));
  };
  const auto wrap = [&metadata](PlanFailure&& failure) -> std::unexpected<OwnedComparisonFailure> {
    return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(failure)));
  };
  if (source_boundary.source_identity() == 0 ||
      alternative.site.selector.source_graph_identity != source_boundary.source_identity())
    return fail(PlanError::kStaleComparisonSite,
                "the alternative does not belong to the authenticated source graph");
  if (!checkpoint.history_matches_graph() || !alternative.checkpoint_map_matches_graph())
    return fail(PlanError::kStaleComparisonSite,
                "the continuation rewrite history has a different graph owner");
  if (checkpoint.source_to_checkpoint().size() != source_boundary.source_tensor_count())
    return fail(PlanError::kStaleComparisonSite,
                "the trial checkpoint tensor map no longer matches its source boundary");
  if (checkpoint.checkpoint_node_origins().size() != checkpoint.graph.nodes().size() ||
      checkpoint.checkpoint_node_to_source().size() != checkpoint.graph.nodes().size())
    return fail(PlanError::kStaleComparisonSite,
                "the trial checkpoint node maps no longer match its graph");
  if (accepted_site.selector.source_graph_identity != source_boundary.source_identity() ||
      !AuthenticComparisonSite(accepted_site) ||
      accepted_site.exact_site_identity != alternative.site.exact_site_identity)
    return fail(PlanError::kStaleComparisonSite,
                "the accepted site no longer matches its alternative owner");
  for (const auto& members : accepted_site.selector.member_origins)
    for (const auto member : members)
      if (!member.valid() || member.v >= checkpoint.origins.source_node_count())
        return fail(PlanError::kStaleComparisonSite,
                    "the accepted site contains an invalid original member");
  if (!AuthenticComparisonSite(alternative.site))
    return fail(PlanError::kStaleComparisonSite,
                "the alternative exact-site identity changed after construction");
  if (!metadata_owner || metadata.refused())
    return fail(PlanError::kAdmission, "alternative graph has no paid owner");
  const auto alternative_input =
      ComparisonPlanInputMetadataBytes(alternative.graph, registry, policy, options);
  if (!alternative_input) return wrap(PlanFailure{.error = PlanError::kAdmission, .detail = {}});
  if (auto paid = metadata.Require(PlannerMetadataPhase::kInput, *alternative_input); !paid)
    return wrap(std::move(paid.error()));
  std::optional<ComparisonCaptureSession> capture;
  if (alternative.site.selector.kind == ComparisonSiteKind::kRewrite) {
    capture.emplace(alternative.site, alternative.origins,
                    ComparisonCaptureSession::AlternativeMode::kDecomposeAcceptedRegion, decisions);
  } else {
    capture.emplace(alternative.site, alternative.origins,
                    ComparisonCaptureSession::AlternativeMode::kExactAcceptedPattern, decisions);
  }
  if (alternative.site.whole_checkpoint.whole_only()) capture->UseWholeCheckpoint(checkpoint);
  if (auto journal = capture->PrepareDecisionJournal(metadata_owner); !journal)
    return wrap(std::move(journal.error()));
  PlanOptions alternative_options = BorrowedOptions(options);
  if (shape_override) alternative_options.shape = shape_override;
  ComparisonFinalNodeOrigins final_lineage(source_boundary.source_identity());
  alternative_options.comparison_capture = &*capture;
  alternative_options.comparison_origins = &alternative.origins;
  alternative_options.comparison_node_origins = alternative.alternative_node_origins;
  alternative_options.comparison_source_to_graph = alternative.source_to_alternative;
  alternative_options.comparison_graph_node_to_source = alternative.alternative_to_source;
  if (auto mapped = MapComparisonPlanInputs(alternative_options, alternative.source_to_alternative,
                                            alternative.alternative_to_source, metadata);
      !mapped)
    return wrap(std::move(mapped.error()));
  if (alternative.site.whole_checkpoint.whole_only()) {
    if (alternative.checkpoint_tensors().size() != checkpoint.graph.tensor_count())
      return fail(PlanError::kStaleComparisonSite, "whole checkpoint tensor map is incomplete");
    // Capture/exclusion uses checkpoint ports; runtime input mapping above
    // continues to use original source IDs and the ordinary boundary.
    alternative_options.comparison_source_to_graph = alternative.checkpoint_tensors();
  }
  alternative_options.metadata_admission = &metadata;
  alternative_options.comparison_final_lineage = &final_lineage;
  if (alternative.checkpoint_nodes().size() != alternative.graph.nodes().size())
    return fail(PlanError::kStaleComparisonSite, "alternative checkpoint node map is incomplete");
  auto planned = Plan(std::move(alternative.graph), registry, device, policy, alternative_options);
  if (!planned) return wrap(std::move(planned.error()));
  if (capture->record_refused())
    return fail(PlanError::kAdmission, "comparison rewrite exclusion record could not be retained");
  if (alternative.site.whole_checkpoint.whole_only() && !capture->whole_site_excluded())
    return fail(PlanError::kMissingComparisonSite,
                "the exact whole checkpoint choice was not excluded in its alternative");
  for (std::size_t i = 0; i < alternative.site.excluded_competing_rewrites.size(); ++i) {
    const auto& record = alternative.site.excluded_competing_rewrites[i];
    const Pattern* registered = FindPattern(record.pattern);
    if (registered == nullptr || registered->version != record.version ||
        registered->kind != PatternKind::kRewrite ||
        (record.pattern == alternative.site.selector.pattern &&
         record.version == alternative.site.selector.version) ||
        record.site_identity != alternative.site.exact_site_identity)
      return fail(PlanError::kStaleComparisonSite,
                  "an excluded rewrite record no longer matches its registered exact site");
    for (std::size_t j = 0; j < i; ++j)
      if (alternative.site.excluded_competing_rewrites[j].pattern == record.pattern &&
          alternative.site.excluded_competing_rewrites[j].version == record.version)
        return fail(PlanError::kStaleComparisonSite,
                    "the alternative contains a duplicate competing rewrite record");
  }
  const auto selected_alternative_source = [&](ir::NodeId original, ir::OpKey key) {
    return SelectedOriginalThroughAlternative(*planned, alternative.alternative_to_source, original,
                                              key);
  };
  bool own_fusion_remains = false;
  if (alternative.site.selector.kind == ComparisonSiteKind::kFusedNode ||
      alternative.site.selector.kind == ComparisonSiteKind::kCompositeNode) {
    if (alternative.site.selector.member_origins.size() != 1 ||
        alternative.site.selector.member_origins.front().size() != 1)
      return fail(PlanError::kStaleComparisonSite,
                  "the selected original fused node is not uniquely attributable");
    own_fusion_remains =
        selected_alternative_source(alternative.site.selector.member_origins.front().front(),
                                    alternative.site.registered_result);
  }
  for (const auto& decision : decisions) {
    if (decision.kind != ComparisonSiteKind::kFusedNode &&
        decision.kind != ComparisonSiteKind::kCompositeNode)
      continue;
    for (const auto& members : decision.member_origins) {
      if (members.size() != 1)
        return fail(PlanError::kStaleComparisonSite,
                    "a committed fused directive is not a singleton original node");
      own_fusion_remains = own_fusion_remains ||
                           selected_alternative_source(members.front(), decision.registered_result);
    }
  }
  for (auto& tensor : alternative.source_to_alternative)
    if (tensor.valid()) {
      if (tensor.v >= planned->tensors.size())
        return fail(PlanError::kStaleComparisonSite, "the source tensor map is out of range");
      tensor = planned->tensors[tensor.v];
    }
  if (!BoundaryMatches(source_boundary, planned->graph, alternative.source_to_alternative,
                       &metadata))
    return fail(PlanError::kStaleComparisonSite,
                "the alternative plan changed the declared source boundary");
  // The continuation's Applied records cannot describe unchanged fusions
  // already present in its input checkpoint. Carry the actual prior records
  // through the exact seed/current maps, excluding every newly rewritten node.
  auto inherited = CopyAppliedHistory(
      checkpoint.prior_applied(), {}, checkpoint.graph.nodes().size(),
      checkpoint.origins.source_node_count(), alternative.metadata_loan, &metadata);
  if (!inherited) return wrap(std::move(inherited.error()));
  for (std::size_t a = 0; a < inherited->size(); ++a) {
    for (auto& proof : (*inherited)[a].numerical_proofs) {
      proof.observed.bound_model = options.observed_bound_model;
      proof.observed.model_spec = options.observed_model_spec;
    }
    auto& made = (*inherited)[a].made;
    made.clear();
    for (const auto old : checkpoint.prior_applied()[a].made) {
      const auto& original = checkpoint.graph.node(old);
      ir::NodeId surviving{};
      for (std::uint32_t n = 0; n < planned->from.size(); ++n) {
        const auto seed_node = planned->from[n];
        if (!seed_node.valid() || seed_node.v >= alternative.checkpoint_nodes().size())
          return fail(PlanError::kStaleComparisonSite, "continuation rewrite map is out of range");
        if (alternative.checkpoint_nodes()[seed_node.v] != old) continue;
        const auto& current = planned->graph.node(ir::NodeId{n});
        if (current.key != original.key || current.params != original.params ||
            std::ranges::any_of(planned->applied, [&](const Applied& applied) {
              return std::ranges::contains(applied.made, ir::NodeId{n});
            }))
          continue;
        if (surviving.valid())
          return fail(PlanError::kStaleComparisonSite, "checkpoint rewrite result is ambiguous");
        surviving = ir::NodeId{n};
      }
      if (surviving.valid()) made.push_back(surviving);
    }
  }
  for (auto& source : planned->from) {
    if (!source.valid() || source.v >= alternative.alternative_to_source.size())
      return fail(PlanError::kStaleComparisonSite, "the planned node map is out of range");
    source = alternative.alternative_to_source[source.v];
  }
  for (auto& applied : planned->applied) {
    for (auto& source : applied.replaced) {
      if (!source.valid() || source.v >= alternative.alternative_to_source.size())
        return fail(PlanError::kStaleComparisonSite, "a rewrite preimage map is out of range");
      source = alternative.alternative_to_source[source.v];
    }
    for (auto& proof : applied.numerical_proofs) {
      if (!proof.site.valid() || proof.site.v >= alternative.alternative_to_source.size())
        return fail(PlanError::kStaleComparisonSite, "a numerical proof site map is out of range");
      proof.site = alternative.alternative_to_source[proof.site.v];
    }
  }
  if (!inherited->empty()) {
    constexpr std::uint64_t envelope = 2 * sizeof(void*) + alignof(std::max_align_t);
    std::uint64_t count = 0, bytes = 0;
    if (__builtin_add_overflow(static_cast<std::uint64_t>(inherited->size()),
                               static_cast<std::uint64_t>(planned->applied.size()), &count) ||
        __builtin_mul_overflow(count, std::uint64_t{sizeof(Applied)}, &bytes) ||
        __builtin_add_overflow(bytes, envelope, &bytes) || !alternative.metadata_loan.Grow(bytes))
      return fail(PlanError::kAdmission, "continued rewrite inventory was refused");
    std::vector<Applied> complete;
    complete.reserve(static_cast<std::size_t>(count));
    for (auto& applied : *inherited)
      if (!applied.made.empty()) complete.push_back(std::move(applied));
    for (auto& applied : planned->applied) complete.push_back(std::move(applied));
    planned->applied = std::move(complete);
  }
  planned->tensors = std::move(alternative.source_to_alternative);
  planned->whole_checkpoint = accepted_site.whole_checkpoint;
  planned->primitive_digest = source_boundary.source_digest();
  if (own_fusion_remains)
    return fail(PlanError::kMissingComparisonSite,
                "the tested fusion remained selected in its own alternative");
  auto journal = capture->TakeDecisionJournal();
  alternative.decision_journal_loan = std::move(journal.loan);
  alternative.prior_decision_exclusions = std::move(journal.records);
  // The capture session borrows the alternative's DAG. End that borrow before
  // transferring the exact mutated DAG beside the final node-origin map.
  capture.reset();
  ComparisonOriginDag final_origins = std::move(alternative.origins);
  return OwnedComparisonPlan(std::move(metadata), std::move(*planned),
                             std::optional<GraphAlternative>(std::move(alternative)),
                             std::optional<ComparisonOriginDag>(std::move(final_origins)),
                             std::move(final_lineage));
}

}  // namespace

std::expected<OwnedComparisonPlan, OwnedComparisonFailure> PlanAlternative(
    const ComparisonInventory& inventory, GraphAlternative alternative,
    const execution::Registry& registry, const DeviceFacts& device, const Policy& policy,
    const PlanOptions& options, const StepShape* shape_override) {
  if (alternative.ledger_identity != 0 || alternative.ledger_generation != 0) {
    PlannerMetadataAdmission metadata(alternative.graph.storage_funder());
    return std::unexpected(OwnedComparisonFailure(
        std::move(metadata), PlanFailure{.error = PlanError::kStaleComparisonSite, .detail = {}}));
  }
  return PlanAlternativeCore(inventory.checkpoint, inventory.site, inventory.source_boundary,
                             std::move(alternative), registry, device, policy, options,
                             shape_override, {});
}

std::expected<OwnedComparisonPlan, OwnedComparisonFailure> PlanAlternative(
    const ComparisonPlanningEnvironment& environment, const ComparisonTrial& trial,
    GraphAlternative alternative, const ComparisonDecisionLedger& decisions) {
  auto metadata = PlannerMetadataAdmission(alternative.graph.storage_funder());
  auto valid = decisions.Validate(environment, &metadata);
  if (!valid || trial.ledger_identity != decisions.ledger_identity() ||
      trial.ledger_generation != decisions.generation() ||
      alternative.ledger_identity != trial.ledger_identity ||
      alternative.ledger_generation != trial.ledger_generation ||
      trial.source_boundary.source_identity() != decisions.source_identity() ||
      trial.source_boundary.source_digest() != environment.source.io.graph.Digest() ||
      trial.source_boundary.digest().combined != decisions.source_boundary_digest()) {
    PlanFailure failure = valid
                              ? PlanFailure{.error = PlanError::kStaleComparisonSite, .detail = {}}
                              : std::move(valid.error());
    if (valid) {
      auto text =
          ir::FormatDiagnostic(&metadata, "{}", std::string_view("comparison trial is stale"));
      if (text)
        failure.detail = std::move(*text);
      else
        failure.error = PlanError::kAdmission;
    }
    return std::unexpected(OwnedComparisonFailure(std::move(metadata), std::move(failure)));
  }
  const auto* shape = environment.shape_override != nullptr ? environment.shape_override
                                                            : environment.options.shape;
  return PlanAlternativeCore(trial.checkpoint, trial.site, trial.source_boundary,
                             std::move(alternative), environment.registry, environment.device,
                             environment.policy, environment.options, shape, decisions.records());
}

}  // namespace llmp::planner
