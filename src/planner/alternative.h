// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Owned, exact-site alternatives for validation-only plan comparisons.

#ifndef LLMP_PLANNER_ALTERNATIVE_H_
#define LLMP_PLANNER_ALTERNATIVE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/comparison_boundary.h"
#include "execution/registry.h"
#include "ir/graph.h"
#include "planner/plan.h"

namespace llmp::planner {

struct Pattern;

class PlannerMetadataAdmission;

enum class ComparisonSiteKind : std::uint8_t {
  kFusedNode,
  kCompositeNode,
  kRewrite,
  kBatch,
};

enum class ComparisonSitePortKind : std::uint8_t { kInput, kOutput };

// A caller-supplied site discriminator. Tensor IDs are checked against the
// authenticated graph and are lookup keys only. Selector/accepted ports
// always name the original source. Complete ports normally do too; ONLY a
// site with opaque whole_checkpoint authority uses its sealed checkpoint's
// tensor/root namespace. Such records cannot enter any local projection.
// The member and ordinal preserve the site-local port order.
struct ComparisonSitePortRef {
  std::uint32_t member = 0;
  ComparisonSitePortKind kind = ComparisonSitePortKind::kInput;
  std::uint16_t ordinal = 0;
  ir::TensorId original_tensor{};
  ir::TensorId original_root{};
  std::uint64_t view_offset = 0;
};

// One member's exact source-node lineage. The ordered outer vector preserves
// member identity for batches; the inner vector is the member's full origin
// set after earlier rewrites/decompositions.
using ComparisonOriginSet = std::vector<ir::NodeId>;
struct ComparisonOriginEntry {
  bool singleton = false;
  ir::NodeId origin{};
  std::uint32_t first_child = 0;
  std::uint32_t child_count = 0;
};

// Persistent exact set membership. Per-node records store an ID; unions add
// only their actual member edges, never a copied origin list per descendant.
class ComparisonOriginDag {
 public:
  ComparisonOriginDag() = default;
  ComparisonOriginDag(ComparisonOriginDag&& other) noexcept
      : loan_(std::move(other.loan_)),
        entries_(std::move(other.entries_)),
        children_(std::move(other.children_)),
        reachable_(std::move(other.reachable_)),
        source_membership_(std::move(other.source_membership_)),
        source_node_count_(std::exchange(other.source_node_count_, 0)) {}
  ComparisonOriginDag& operator=(ComparisonOriginDag&&) noexcept = delete;
  ComparisonOriginDag(const ComparisonOriginDag&) = delete;
  ComparisonOriginDag& operator=(const ComparisonOriginDag&) = delete;

  static std::expected<ComparisonOriginDag, PlanFailure> Create(
      std::shared_ptr<const ir::StorageFunder> funder, std::uint32_t source_node_count,
      std::uint64_t entry_capacity = 0, std::uint64_t child_capacity = 0,
      ir::DiagnosticAdmission* diagnostic = nullptr);
  std::expected<ComparisonOriginDag, PlanFailure> Clone(
      std::shared_ptr<const ir::StorageFunder> funder,
      ir::DiagnosticAdmission* diagnostic = nullptr) const;
  std::expected<ComparisonOriginSetId, PlanFailure> AddSingleton(
      ir::NodeId origin, ir::DiagnosticAdmission* diagnostic = nullptr);
  std::expected<ComparisonOriginSetId, PlanFailure> AddUnion(
      std::span<const ComparisonOriginSetId> members,
      ir::DiagnosticAdmission* diagnostic = nullptr);
  std::expected<ComparisonOriginSetId, PlanFailure> AddUnion(
      std::span<const ir::NodeId> member_nodes, std::span<const ComparisonOriginSetId> node_origins,
      ir::DiagnosticAdmission* diagnostic = nullptr);
  bool Contains(ComparisonOriginSetId set, ir::NodeId origin);
  bool Equals(ComparisonOriginSetId set, const ComparisonOriginSet& origins);
  std::optional<std::size_t> OriginCount(ComparisonOriginSetId set);
  bool CopyOrigins(ComparisonOriginSetId set, std::span<ir::NodeId> output, std::size_t& written);
  bool EqualsUnionOfNodes(std::span<const ir::NodeId> nodes,
                          std::span<const ComparisonOriginSetId> node_origins,
                          const ComparisonOriginSet& origins);
  std::uint64_t storage_bytes() const { return loan_.bytes(); }
  std::uint32_t source_node_count() const { return source_node_count_; }
  std::span<const ComparisonOriginEntry> entries() const { return entries_; }
  std::span<const ComparisonOriginSetId> children() const { return children_; }

 private:
  std::expected<void, PlanFailure> Reserve(std::uint64_t entries, std::uint64_t children,
                                           ir::DiagnosticAdmission* diagnostic = nullptr);
  // Declared first so vector storage dies before its admission loan.
  ir::StorageLoan loan_{};
  std::vector<ComparisonOriginEntry> entries_{};
  std::vector<ComparisonOriginSetId> children_{};
  // Paid reusable query workspace, private to this synchronous capture
  // session. Queries are non-const so callers cannot share mutable scratch
  // through an immutable DAG reference.
  std::vector<std::uint8_t> reachable_{};
  std::vector<std::uint8_t> source_membership_{};
  std::uint32_t source_node_count_ = 0;
};

// A successful tracked Plan may hand off its already-paid final per-node
// lineage without copying it. The source identity authenticates the request;
// the planned identity authenticates the graph whose nodes the IDs describe.
struct ComparisonFinalNodeOrigins {
  explicit ComparisonFinalNodeOrigins(std::uint64_t source_identity_in = 0)
      : source_graph_identity(source_identity_in) {}
  ComparisonFinalNodeOrigins(ComparisonFinalNodeOrigins&&) noexcept = default;
  ComparisonFinalNodeOrigins& operator=(ComparisonFinalNodeOrigins&&) noexcept = delete;
  ComparisonFinalNodeOrigins(const ComparisonFinalNodeOrigins&) = delete;
  ComparisonFinalNodeOrigins& operator=(const ComparisonFinalNodeOrigins&) = delete;
  std::uint64_t source_graph_identity = 0;
  std::uint64_t planned_graph_identity = 0;
  OwnedRewriteOrigins node_origins{};
};

// The shared copy/decompose seam used by ordinary fallback and comparison
// arms. `input_to_seed` is directional; `seed_to_input` names a source for
// copies only and is never treated as a rewrite preimage.
struct PlanSeed {
  PlanSeed(ir::Graph graph_in, ir::StorageLoan loan_in, std::vector<ir::TensorId> map_in,
           std::vector<ir::NodeId> from_in, std::vector<ComparisonOriginSetId> origins_in = {})
      : graph(std::move(graph_in)),
        metadata_loan(std::move(loan_in)),
        input_to_seed(std::move(map_in)),
        seed_to_input(std::move(from_in)),
        seed_origin_sets(std::move(origins_in)) {}
  PlanSeed(PlanSeed&&) noexcept = default;
  PlanSeed& operator=(PlanSeed&&) noexcept = delete;
  PlanSeed(const PlanSeed&) = delete;
  PlanSeed& operator=(const PlanSeed&) = delete;
  ir::Graph graph;
  // Owns the directional maps and optional lineage vector until they die.
  ir::StorageLoan metadata_loan;
  std::vector<ir::TensorId> input_to_seed;
  std::vector<ir::NodeId> seed_to_input;
  std::vector<ComparisonOriginSetId> seed_origin_sets;
};

std::expected<PlanSeed, ir::VerifyError> CopyExpand(
    const ir::Graph& graph, const std::vector<bool>& expand,
    const std::vector<bool>* drop = nullptr, const std::vector<ir::NodeId>* repeats = nullptr,
    std::span<const ComparisonOriginSetId> origin_sets = {}, ComparisonOriginDag* origins = nullptr,
    PlannerMetadataAdmission* metadata_admission = nullptr);

struct ComparisonSiteSelector {
  std::uint64_t source_graph_identity = 0;
  ComparisonSiteKind kind = ComparisonSiteKind::kFusedNode;
  std::string pattern;
  std::uint16_t version = 0;
  std::vector<ComparisonOriginSet> member_origins;
  // Optional disambiguators. If present, owner segments correspond one to
  // one with members; ports are ordered site-boundary inputs then outputs.
  std::vector<std::uint8_t> member_owner_segments;
  std::vector<ComparisonSitePortRef> boundary_ports;
  // Nonzero only for explicit BoundWholeComparison. Canonical checkpoint
  // match identity from discovery; never interprets generated ports as source IDs.
  base::Sha256Digest checkpoint_match{};
};

struct ComparisonSite {
  WholeComparisonCheckpoint whole_checkpoint{};
  ComparisonSiteSelector selector;
  std::vector<ComparisonOriginSet> accepted_member_origins;
  std::vector<std::uint8_t> accepted_member_owner_segments;
  std::vector<ComparisonSitePortRef> accepted_boundary_ports;
  // Derived from the exact pre-match checkpoint and its directional source
  // map. Unlike the caller's optional disambiguators above, these are the
  // complete region boundary used to authenticate overlapping routes. A
  // whole_checkpoint site keeps current checkpoint IDs here, never false
  // original-source aliases; its opaque authority seals that namespace.
  std::vector<std::uint8_t> complete_member_owner_segments;
  std::vector<ComparisonSitePortRef> complete_boundary_ports;
  ir::OpKey registered_result{};
  base::Sha256Digest exact_site_identity{};
  struct ExcludedRewrite {
    std::string_view pattern;
    std::uint16_t version = 0;
    base::Sha256Digest site_identity{};
  };
  // Each record names an actually matched and authorized competing rewrite;
  // its site_identity points to this parent's exact authenticated site.
  std::vector<ExcludedRewrite> excluded_competing_rewrites;
};

// Immutable original-source directive retained after an accepted alternative.
// It contains source NodeIds and complete ordered site facts only; no trial
// graph IDs, DAG set handles, winner labels, or borrowed trial owners.
struct ComparisonDecisionRecord {
  ComparisonDecisionRecord() = default;
  ComparisonDecisionRecord(ComparisonDecisionRecord&&) noexcept = default;
  ComparisonDecisionRecord& operator=(ComparisonDecisionRecord&&) noexcept = delete;
  ComparisonDecisionRecord(const ComparisonDecisionRecord&) = delete;
  ComparisonDecisionRecord& operator=(const ComparisonDecisionRecord&) = delete;

  ir::StorageLoan metadata_loan;
  ComparisonSiteKind kind = ComparisonSiteKind::kRewrite;
  std::string_view pattern{};
  std::uint16_t version = 0;
  ir::OpKey registered_result{};
  base::Sha256Digest site_identity{};
  std::vector<ComparisonOriginSet> member_origins;
  std::vector<std::uint8_t> member_owner_segments;
  std::vector<ComparisonSitePortRef> boundary_ports;
  std::vector<std::uint8_t> complete_member_owner_segments;
  std::vector<ComparisonSitePortRef> complete_boundary_ports;
};

struct ComparisonDecisionExclusion {
  base::Sha256Digest prior_site_identity{};
  std::string_view competing_pattern;  // registered static name
  std::uint16_t competing_version = 0;
};

struct ComparisonDecisionJournal {
  ir::StorageLoan loan;
  std::vector<ComparisonDecisionExclusion> records;
};

struct ComparisonCaptureMark {
  std::size_t decision_exclusions = 0;
  std::size_t current_site_exclusions = 0;
  bool whole_site_excluded = false;
};

// The graph and directional maps at the immutable input to an accepted
// transformation. The payment owner precedes all mapping storage.
struct ComparisonCheckpoint {
  ComparisonCheckpoint(ir::StorageLoan maps_loan_in, ir::Graph graph_in,
                       std::vector<ir::TensorId> source_to_checkpoint_in,
                       std::vector<ir::NodeId> checkpoint_node_to_source_in,
                       std::vector<ComparisonOriginSetId> checkpoint_node_origins_in,
                       ComparisonOriginDag origins_in)
      : maps_loan(std::move(maps_loan_in)),
        graph(std::move(graph_in)),
        source_to_checkpoint_(std::move(source_to_checkpoint_in)),
        checkpoint_node_to_source_(std::move(checkpoint_node_to_source_in)),
        checkpoint_node_origins_(std::move(checkpoint_node_origins_in)),
        origins(std::move(origins_in)),
        history_graph_identity_(graph.identity()) {}
  ComparisonCheckpoint(ComparisonCheckpoint&&) noexcept = default;
  ComparisonCheckpoint& operator=(ComparisonCheckpoint&&) noexcept = delete;
  ComparisonCheckpoint(const ComparisonCheckpoint&) = delete;
  ir::StorageLoan maps_loan;
  ir::Graph graph;

 private:
  std::vector<ir::TensorId> source_to_checkpoint_;
  std::vector<ir::NodeId> checkpoint_node_to_source_;
  std::vector<ComparisonOriginSetId> checkpoint_node_origins_;

 public:
  std::span<const ir::TensorId> source_to_checkpoint() const { return source_to_checkpoint_; }
  std::span<const ir::NodeId> checkpoint_node_to_source() const {
    return checkpoint_node_to_source_;
  }
  std::span<const ComparisonOriginSetId> checkpoint_node_origins() const {
    return checkpoint_node_origins_;
  }
  ComparisonOriginDag origins;
  // Replaced/proof IDs are original-source IDs; made IDs belong to graph.
  // Captured only from the successful earlier rewrite transaction.
  std::span<const Applied> prior_applied() const { return prior_applied_; }
  std::span<const ir::NodeId> captured_members() const { return captured_members_; }
  bool history_matches_graph() const { return graph.identity() == history_graph_identity_; }
  ir::TensorId mapped_output(ir::TensorId checkpoint_tensor, std::uint64_t final_graph) const {
    return final_graph == current_graph_identity_ && checkpoint_tensor.valid() &&
                   checkpoint_tensor.v < checkpoint_to_current_.size()
               ? checkpoint_to_current_[checkpoint_tensor.v]
               : ir::TensorId{};
  }

 private:
  friend class ComparisonCaptureSession;
  std::vector<Applied> prior_applied_;
  std::vector<ir::NodeId> captured_members_;
  std::vector<ir::TensorId> checkpoint_to_current_;
  std::uint64_t current_graph_identity_ = 0;
  std::uint64_t history_graph_identity_ = 0;
};

struct ComparisonSiteSelector;
class ComparisonDiscoverySession;

struct ComparisonDiscoveryTicket {
  ComparisonSiteKind kind = ComparisonSiteKind::kFusedNode;
  std::string_view pattern;  // registered static name; never caller-owned
  std::uint16_t version = 0;
  ir::OpKey registered_operation{};
  std::uint64_t ordinal = 0;
  std::uint64_t generation = 0;
  std::vector<std::uint32_t> member_offsets;
  std::vector<ir::NodeId> member_origins;
  std::vector<std::uint8_t> owner_segments;
  // A committed ticket can remain useful for lineage even when an exact
  // original boundary port has no unique source preimage after a merge.
  // Such a ticket cannot authorize local materialization. An explicit
  // BoundWholeComparison can authenticate its checkpoint_match instead.
  bool boundary_representable = true;
  base::Sha256Digest checkpoint_match{};
  std::vector<ComparisonSitePortRef> boundary_ports;
  base::Sha256Digest occurrence_identity{};
};

// An opt-in, non-capturing committed trace. The borrowed operation filter is
// used only during the synchronous Plan call. The returned owner retains the
// final lineage and mapping needed to resolve tickets after later rewrites.
class ComparisonDiscoverySession {
 public:
  static std::expected<ComparisonDiscoverySession, PlanFailure> Create(
      const ir::Graph& source, std::shared_ptr<const ir::StorageFunder> funder,
      std::span<const ir::OpKey> operation_filter, bool harness_all,
      ir::DiagnosticAdmission* diagnostic = nullptr);
  ComparisonDiscoverySession(ComparisonDiscoverySession&&) noexcept = default;
  ComparisonDiscoverySession& operator=(ComparisonDiscoverySession&&) noexcept = delete;
  ComparisonDiscoverySession(const ComparisonDiscoverySession&) = delete;
  ComparisonDiscoverySession& operator=(const ComparisonDiscoverySession&) = delete;
  ComparisonOriginDag& origins() { return origins_; }
  const ComparisonOriginDag& origins() const { return origins_; }
  std::uint64_t source_identity() const { return source_identity_; }
  std::span<const ComparisonDiscoveryTicket> tickets() const { return tickets_; }
  std::span<const ir::TensorId> source_to_planned() const { return source_to_planned_; }
  std::span<const ComparisonOriginSetId> planned_node_origins() const {
    return planned_node_origins_;
  }
  // Seed maps are borrowed by Plan during the synchronous discovery call and
  // are released at Finalize; finalized results retain only source-to-final.
  std::span<const ir::TensorId> initial_source_to_graph() const { return initial_source_to_graph_; }
  std::span<const ir::NodeId> initial_graph_node_to_source() const {
    return initial_graph_node_to_source_;
  }
  std::span<const ComparisonOriginSetId> initial_node_origins() const {
    return initial_node_origins_;
  }
  bool enabled() const { return enabled_; }
  std::expected<void, PlanFailure> Record(ComparisonSiteKind kind, std::string_view pattern,
                                          std::uint16_t version, ir::OpKey operation,
                                          std::span<const ir::NodeId> matched_nodes,
                                          std::span<const ComparisonOriginSetId> node_origins,
                                          const ir::Graph& graph,
                                          std::span<const ir::TensorId> source_to_graph,
                                          ir::DiagnosticAdmission* diagnostic = nullptr);
  std::expected<void, PlanFailure> Finalize(
      const ir::Graph& planned_graph, std::span<const ir::TensorId> source_to_planned,
      std::span<const ComparisonOriginSetId> planned_node_origins,
      ir::DiagnosticAdmission* diagnostic = nullptr);

 private:
  explicit ComparisonDiscoverySession(std::uint64_t source_identity,
                                      std::shared_ptr<const ir::StorageFunder> funder, bool enabled)
      : source_identity_(source_identity), funder_(std::move(funder)), enabled_(enabled) {}
  ComparisonDiscoverySession(std::uint64_t source_identity,
                             std::shared_ptr<const ir::StorageFunder> funder, bool enabled,
                             ComparisonOriginDag origins)
      : origins_(std::move(origins)),
        source_identity_(source_identity),
        funder_(std::move(funder)),
        enabled_(enabled) {}
  bool Interested(ir::OpKey operation) const;
  ir::StorageLoan loan_{};
  ir::StorageLoan seed_loan_{};
  ComparisonOriginDag origins_{};
  std::uint64_t source_identity_ = 0;
  const ir::Graph* source_graph_ = nullptr;  // synchronous only; cleared by Finalize
  std::shared_ptr<const ir::StorageFunder> funder_;
  bool enabled_ = false;
  std::span<const ir::OpKey> operation_filter_{};  // borrowed only during planning
  bool harness_all_ = false;
  std::uint64_t next_ordinal_ = 0;
  std::uint64_t generation_ = 0;
  std::vector<ComparisonDiscoveryTicket> tickets_;
  std::vector<ir::TensorId> initial_source_to_graph_;
  std::vector<ir::NodeId> initial_graph_node_to_source_;
  std::vector<ComparisonOriginSetId> initial_node_origins_;
  std::vector<ir::TensorId> source_to_planned_;
  std::vector<ComparisonOriginSetId> planned_node_origins_;
};

class ComparisonCaptureSession {
 public:
  // The alternative mode suppresses registered rewrite candidates only when
  // their complete matched region equals the authenticated accepted site.
  enum class AlternativeMode : std::uint8_t { kExactAcceptedPattern, kDecomposeAcceptedRegion };
  ComparisonCaptureSession(const ComparisonSiteSelector& selector, ComparisonOriginDag& origins,
                           std::shared_ptr<const ir::StorageFunder> funder = {},
                           bool exclude_site = true,
                           ComparisonDiscoverySession* discovery = nullptr,
                           std::span<const ComparisonDecisionRecord> decisions = {})
      : selector_(selector),
        origins_(origins),
        funder_(std::move(funder)),
        exclude_site_(exclude_site),
        discovery_(discovery),
        decisions_(decisions) {}
  ComparisonCaptureSession(ComparisonSite& site, ComparisonOriginDag& origins, AlternativeMode mode,
                           std::span<const ComparisonDecisionRecord> decisions = {})
      : selector_(site.selector),
        origins_(origins),
        exclude_site_(true),
        alternative_site_(&site),
        alternative_mode_(mode),
        decisions_(decisions) {}
  void UseWholeCheckpoint(const ComparisonCheckpoint& checkpoint) {
    whole_checkpoint_ = &checkpoint;
  }
  bool ExcludesRewrite(const Pattern& pattern, std::span<const ir::NodeId> matched_nodes,
                       std::span<const ComparisonOriginSetId> node_origins, const ir::Graph& graph,
                       std::span<const ir::TensorId> source_to_graph);
  std::expected<std::optional<ComparisonCheckpoint>, PlanFailure> CaptureRewrite(
      std::string_view pattern, std::uint16_t version, std::span<const ir::NodeId> matched_nodes,
      std::span<const ComparisonOriginSetId> node_origins, const ir::Graph& graph,
      std::span<const ir::TensorId> source_to_graph,
      std::span<const ir::NodeId> graph_node_to_source,
      ir::DiagnosticAdmission* diagnostic = nullptr, std::span<const Applied> prior_applied = {},
      std::span<const ir::NodeId> prior_to_source = {});
  std::expected<std::optional<ComparisonCheckpoint>, PlanFailure> CaptureBatch(
      std::string_view pattern, std::uint16_t version, std::span<const ir::NodeId> matched_nodes,
      std::span<const ComparisonOriginSetId> node_origins, const ir::Graph& graph,
      std::span<const ir::TensorId> source_to_graph,
      std::span<const ir::NodeId> graph_node_to_source,
      ir::DiagnosticAdmission* diagnostic = nullptr, std::span<const Applied> prior_applied = {},
      std::span<const ir::NodeId> prior_to_source = {});
  void CommitRound(std::optional<ComparisonCheckpoint> checkpoint, std::size_t candidates,
                   std::uint64_t previous_graph = 0, std::uint64_t next_graph = 0,
                   std::span<const ir::TensorId> tensors = {});
  bool AdvanceCheckpointMap(std::uint64_t previous_graph, std::uint64_t next_graph,
                            std::span<const ir::TensorId> tensors);
  std::expected<ComparisonCheckpoint, PlanFailure> TakeCheckpoint(
      ir::DiagnosticAdmission* diagnostic = nullptr);
  bool record_refused() const { return record_refused_; }
  bool whole_site_excluded() const { return whole_site_excluded_; }
  ComparisonCaptureMark DecisionJournalMark() const {
    return ComparisonCaptureMark{
        .decision_exclusions = decision_exclusions_.size(),
        .current_site_exclusions = alternative_site_ != nullptr
                                       ? alternative_site_->excluded_competing_rewrites.size()
                                       : 0,
        .whole_site_excluded = whole_site_excluded_};
  }
  void RollbackDecisionJournal(ComparisonCaptureMark mark) {
    whole_site_excluded_ = mark.whole_site_excluded;
    if (mark.decision_exclusions <= decision_exclusions_.size())
      decision_exclusions_.resize(mark.decision_exclusions);
    if (alternative_site_ != nullptr &&
        mark.current_site_exclusions <= alternative_site_->excluded_competing_rewrites.size())
      alternative_site_->excluded_competing_rewrites.resize(mark.current_site_exclusions);
  }
  std::span<const ComparisonDecisionExclusion> decision_exclusions() const {
    return decision_exclusions_;
  }
  std::expected<void, PlanFailure> PrepareDecisionJournal(
      std::shared_ptr<const ir::StorageFunder> funder);
  ComparisonDecisionJournal TakeDecisionJournal() {
    return ComparisonDecisionJournal{std::move(decision_journal_loan_),
                                     std::move(decision_exclusions_)};
  }
  std::expected<void, PlanFailure> RecordDiscovery(
      ComparisonSiteKind kind, std::string_view pattern, std::uint16_t version, ir::OpKey operation,
      std::span<const ir::NodeId> matched_nodes,
      std::span<const ComparisonOriginSetId> node_origins, const ir::Graph& graph,
      std::span<const ir::TensorId> source_to_graph, ir::DiagnosticAdmission* diagnostic = nullptr);
  std::expected<void, PlanFailure> FinalizeDiscovery(
      const ir::Graph& planned_graph, std::span<const ir::TensorId> source_to_planned,
      std::span<const ComparisonOriginSetId> planned_node_origins,
      ir::DiagnosticAdmission* diagnostic = nullptr);
  bool ExcludesBatch(std::string_view pattern, std::uint16_t version,
                     std::span<const ir::NodeId> matched_nodes,
                     std::span<const ComparisonOriginSetId> node_origins, const ir::Graph& graph,
                     std::span<const ir::TensorId> source_to_graph = {});

 private:
  bool RecordDecisionExclusion(const ComparisonDecisionRecord& decision, std::string_view pattern,
                               std::uint16_t version);
  const ComparisonSiteSelector& selector_;
  ComparisonOriginDag& origins_;
  std::shared_ptr<const ir::StorageFunder> funder_;
  std::span<const ir::TensorId> PriorTensorMap() const {
    return whole_checkpoint_
               ? std::span<const ir::TensorId>(whole_checkpoint_->source_to_checkpoint())
               : std::span<const ir::TensorId>{};
  }
  friend struct ComparisonCheckpointAuthorityAccess;
  bool recapture_after_decisions_ = false;
  const ComparisonCheckpoint* whole_checkpoint_ = nullptr;  // synchronous alternative borrow
  bool exclude_site_ = true;
  ComparisonDiscoverySession* discovery_ = nullptr;
  ComparisonSite* alternative_site_ = nullptr;
  AlternativeMode alternative_mode_ = AlternativeMode::kExactAcceptedPattern;
  std::span<const ComparisonDecisionRecord> decisions_{};  // borrowed for this synchronous Plan
  ir::StorageLoan decision_journal_loan_{};
  std::vector<ComparisonDecisionExclusion> decision_exclusions_;
  bool decision_journal_prepared_ = false;
  bool record_refused_ = false;
  bool whole_site_excluded_ = false;
  std::optional<ComparisonCheckpoint> checkpoint_;
  std::size_t accepted_candidates_ = 0;
};

enum class PlannerMetadataPhase : std::uint8_t {
  kInput,
  kMatch,
  kMatchDetail,
  kRetry,
  kRewrite,
  kRewriteCompose,
  kAdopt,
  kOwnerSplit,
  kForward,
  kComparisonMap,
  kComparisonMappedInputs,
  kSelection,
  kDecompositionRecords,
  kLanes,
  kPlacement,
  kDiagnostic,
  kCount,
};

// Comparison-only planner metadata admission. Each category records its
// maximum live envelope over the synchronous call; the retained high-water
// loan remains with the returned plan.
class PlannerMetadataAdmission : public ir::DiagnosticAdmission {
 public:
  explicit PlannerMetadataAdmission(std::shared_ptr<const ir::StorageFunder> funder);
  PlannerMetadataAdmission(PlannerMetadataAdmission&&) noexcept = default;
  PlannerMetadataAdmission& operator=(PlannerMetadataAdmission&&) noexcept = delete;
  PlannerMetadataAdmission(const PlannerMetadataAdmission&) = delete;
  PlannerMetadataAdmission& operator=(const PlannerMetadataAdmission&) = delete;

  std::expected<void, PlanFailure> Require(PlannerMetadataPhase phase, std::uint64_t bytes);
  std::expected<void, PlanFailure> SetMatchDetailLive(std::uint64_t bytes);
  bool TryChargeDiagnostic(std::uint64_t bytes) override;
  std::uint64_t storage_bytes() const { return loan_.bytes(); }
  bool refused() const { return refused_; }
  void RefuseAdmission() { refused_ = true; }

 private:
  ir::StorageLoan loan_{};
  std::array<std::uint64_t, static_cast<std::size_t>(PlannerMetadataPhase::kCount)> phases_{};
  std::uint64_t diagnostic_bytes_ = 0;
  bool refused_ = false;
};

// The payment owner must outlive its diagnostic string. Members are ordered so
// `detail` is destroyed before `metadata`; move assignment is forbidden for
// the same reason.
struct OwnedComparisonFailure {
  OwnedComparisonFailure(PlannerMetadataAdmission metadata_in, PlanFailure failure_in)
      : metadata(std::move(metadata_in)),
        error(failure_in.error),
        detail(std::move(failure_in.detail)) {}
  OwnedComparisonFailure(OwnedComparisonFailure&&) noexcept = default;
  OwnedComparisonFailure& operator=(OwnedComparisonFailure&&) noexcept = delete;
  OwnedComparisonFailure(const OwnedComparisonFailure&) = delete;
  OwnedComparisonFailure& operator=(const OwnedComparisonFailure&) = delete;
  PlannerMetadataAdmission metadata;
  PlanError error;
  std::string detail;
};

std::expected<std::uint64_t, PlanFailure> ComparisonPlanInputMetadataBytes(
    const ir::Graph& graph, const execution::Registry& registry, const Policy& policy,
    const PlanOptions& options);

// The graph and tensor mapping captured immediately before the selected
// transformation. A loan owns the mapping's complete allocation envelope;
// the loan is declared first so mapping storage is destroyed before refund.
// A complete independently planned graph derived from one exact site. Maps
// are directional and never imply that a many-to-one output has a unique
// inverse.
struct GraphAlternative {
 private:
  friend struct ComparisonCheckpointAuthorityAccess;
  GraphAlternative(ir::StorageLoan metadata_loan_in, ir::StorageLoan exclusion_records_loan_in,
                   ir::Graph graph_in, std::vector<ir::TensorId> source_to_alternative_in,
                   std::vector<ir::NodeId> alternative_to_source_in,
                   std::vector<ComparisonOriginSetId> alternative_node_origins_in,
                   ComparisonOriginDag origins_in, std::vector<Applied> applied_in,
                   ComparisonSite site_in, std::vector<ir::NodeId> checkpoint_nodes_in,
                   std::vector<ir::TensorId> checkpoint_tensors_in = {},
                   std::uint64_t ledger_identity_in = 0, std::uint64_t ledger_generation_in = 0)
      : metadata_loan(std::move(metadata_loan_in)),
        exclusion_records_loan(std::move(exclusion_records_loan_in)),
        graph(std::move(graph_in)),
        source_to_alternative(std::move(source_to_alternative_in)),
        alternative_to_source(std::move(alternative_to_source_in)),
        alternative_node_origins(std::move(alternative_node_origins_in)),
        origins(std::move(origins_in)),
        applied(std::move(applied_in)),
        site(std::move(site_in)),
        ledger_identity(ledger_identity_in),
        ledger_generation(ledger_generation_in),
        checkpoint_tensors_(std::move(checkpoint_tensors_in)),
        checkpoint_nodes_(std::move(checkpoint_nodes_in)),
        seed_graph_identity_(graph.identity()) {}

 public:
  GraphAlternative(GraphAlternative&&) noexcept = default;
  GraphAlternative& operator=(GraphAlternative&&) noexcept = delete;
  GraphAlternative(const GraphAlternative&) = delete;
  ir::StorageLoan metadata_loan;
  // Pays the pre-reserved, registry-bounded exclusion audit records below.
  // Declared before every vector that uses it so records die before refund.
  ir::StorageLoan exclusion_records_loan;
  ir::Graph graph;
  std::vector<ir::TensorId> source_to_alternative;
  std::vector<ir::NodeId> alternative_to_source;
  std::vector<ComparisonOriginSetId> alternative_node_origins;
  ComparisonOriginDag origins;
  std::vector<Applied> applied;
  ComparisonSite site;
  // Capture-session payment precedes the exact actual-prior-site journal.
  ir::StorageLoan decision_journal_loan;
  std::vector<ComparisonDecisionExclusion> prior_decision_exclusions;
  std::uint64_t ledger_identity = 0;
  std::uint64_t ledger_generation = 0;
  // Exact CopyExpand reverse map; unlike source lineage this identifies each
  // unchanged checkpoint node separately. The seed's retained loan pays it.
  std::span<const ir::TensorId> checkpoint_tensors() const { return checkpoint_tensors_; }
  std::span<const ir::NodeId> checkpoint_nodes() const { return checkpoint_nodes_; }
  bool checkpoint_map_matches_graph() const { return graph.identity() == seed_graph_identity_; }

 private:
  std::vector<ir::TensorId> checkpoint_tensors_;
  std::vector<ir::NodeId> checkpoint_nodes_;
  std::uint64_t seed_graph_identity_ = 0;
};

struct OwnedComparisonPlan {
  OwnedComparisonPlan(
      PlannerMetadataAdmission metadata_in, StepPlan plan_in,
      std::optional<GraphAlternative> alternative_owner_in = {},
      std::optional<ComparisonOriginDag> final_origins_in = {},
      ComparisonFinalNodeOrigins final_node_origins_in = ComparisonFinalNodeOrigins{})
      : metadata(std::move(metadata_in)),
        alternative_owner(std::move(alternative_owner_in)),
        final_origins(std::move(final_origins_in)),
        final_node_origins(std::move(final_node_origins_in)),
        plan(std::move(plan_in)) {}
  OwnedComparisonPlan(OwnedComparisonPlan&&) noexcept = default;
  OwnedComparisonPlan& operator=(OwnedComparisonPlan&&) noexcept = delete;
  OwnedComparisonPlan(const OwnedComparisonPlan&) = delete;
  OwnedComparisonPlan& operator=(const OwnedComparisonPlan&) = delete;
  const StepPlan& view() const { return plan; }
  PlannerMetadataAdmission metadata;
  // The source mapping/DAG loan can own vectors moved into `plan`; retain its
  // payment owner until after the StepPlan vectors die.
  std::optional<GraphAlternative> alternative_owner;
  // Pair the final DAG with the exact final-node map; the map is directional
  // source provenance and never reconstructed from StepPlan::from.
  std::optional<ComparisonOriginDag> final_origins;
  ComparisonFinalNodeOrigins final_node_origins;
  StepPlan plan;
};

// Owns the normal-policy incumbent and the checkpoint needed to construct
// one site's alternative. It retains the original source identity and the
// payment owner for all comparison-only metadata.
struct DiscoveredComparisonPlan {
  DiscoveredComparisonPlan(ComparisonDiscoverySession discovery_in,
                           OwnedComparisonPlan incumbent_in)
      : discovery(std::move(discovery_in)), incumbent(std::move(incumbent_in)) {}
  DiscoveredComparisonPlan(DiscoveredComparisonPlan&&) noexcept = default;
  DiscoveredComparisonPlan& operator=(DiscoveredComparisonPlan&&) noexcept = delete;
  DiscoveredComparisonPlan(const DiscoveredComparisonPlan&) = delete;
  ComparisonDiscoverySession discovery;
  OwnedComparisonPlan incumbent;
};

struct ComparisonInventory {
  ComparisonInventory(ir::StorageLoan metadata_loan_in, ComparisonSite site_in,
                      ComparisonCheckpoint checkpoint_in, OwnedComparisonPlan incumbent_in,
                      components::OwnedComparisonBoundary source_boundary_in)
      : metadata_loan(std::move(metadata_loan_in)),
        site(std::move(site_in)),
        checkpoint(std::move(checkpoint_in)),
        incumbent(std::move(incumbent_in)),
        source_boundary(std::move(source_boundary_in)) {}
  ComparisonInventory(ComparisonInventory&&) noexcept = default;
  ComparisonInventory& operator=(ComparisonInventory&&) noexcept = delete;
  ComparisonInventory(const ComparisonInventory&) = delete;
  ir::StorageLoan metadata_loan;
  ComparisonSite site;
  ComparisonCheckpoint checkpoint;
  OwnedComparisonPlan incumbent;
  components::OwnedComparisonBoundary source_boundary;
};

struct ComparisonTrial {
  ComparisonTrial(ir::StorageLoan metadata_loan_in, ComparisonSite site_in,
                  ComparisonCheckpoint checkpoint_in,
                  components::OwnedComparisonBoundary source_boundary_in,
                  std::uint64_t ledger_identity_in = 0, std::uint64_t ledger_generation_in = 0)
      : metadata_loan(std::move(metadata_loan_in)),
        site(std::move(site_in)),
        checkpoint(std::move(checkpoint_in)),
        source_boundary(std::move(source_boundary_in)),
        ledger_identity(ledger_identity_in),
        ledger_generation(ledger_generation_in) {}
  ComparisonTrial(ComparisonTrial&&) noexcept = default;
  ComparisonTrial& operator=(ComparisonTrial&&) noexcept = delete;
  ComparisonTrial(const ComparisonTrial&) = delete;
  ir::StorageLoan metadata_loan;
  ComparisonSite site;
  ComparisonCheckpoint checkpoint;
  components::OwnedComparisonBoundary source_boundary;
  std::uint64_t ledger_identity = 0;
  std::uint64_t ledger_generation = 0;
};

// A single synchronous planning transaction. Callback targets and every
// object they reference are borrowed and immutable until the transaction ends.
// This identity prevents a ledger from being replayed with substituted query
// callbacks or planning facts.
class ComparisonPlanningEnvironment {
 public:
  ComparisonPlanningEnvironment(const components::ComparisonBoundaryView& source_in,
                                const execution::Registry& registry_in,
                                const DeviceFacts& device_in, const Policy& policy_in,
                                const PlanOptions& options_in,
                                const StepShape* shape_override_in = nullptr)
      : source(source_in),
        registry(registry_in),
        device(device_in),
        policy(policy_in),
        options(options_in),
        shape_override(shape_override_in) {}
  ComparisonPlanningEnvironment(const ComparisonPlanningEnvironment&) = delete;
  ComparisonPlanningEnvironment& operator=(const ComparisonPlanningEnvironment&) = delete;
  ComparisonPlanningEnvironment(ComparisonPlanningEnvironment&&) = delete;
  ComparisonPlanningEnvironment& operator=(ComparisonPlanningEnvironment&&) = delete;

  const components::ComparisonBoundaryView& source;
  const execution::Registry& registry;
  const DeviceFacts& device;
  const Policy& policy;
  const PlanOptions& options;
  const StepShape* shape_override;
};

// Paid, append-only construction restrictions for alternatives of one
// immutable source/environment. The registry/device/shape digests are
// canonical facts, not caller labels.
class ComparisonDecisionLedger {
 public:
  static std::expected<ComparisonDecisionLedger, PlanFailure> Create(
      const ComparisonPlanningEnvironment& environment,
      std::shared_ptr<const ir::StorageFunder> funder,
      ir::DiagnosticAdmission* diagnostic = nullptr);
  ComparisonDecisionLedger(ComparisonDecisionLedger&&) noexcept = default;
  ComparisonDecisionLedger& operator=(ComparisonDecisionLedger&&) noexcept = delete;
  ComparisonDecisionLedger(const ComparisonDecisionLedger&) = delete;
  std::span<const ComparisonDecisionRecord> records() const { return records_; }
  std::uint64_t source_identity() const { return source_identity_; }
  std::uint64_t ledger_identity() const { return ledger_identity_; }
  std::uint64_t generation() const { return generation_; }
  const base::Sha256Digest& source_boundary_digest() const { return source_boundary_digest_; }
  std::expected<void, PlanFailure> Validate(const ComparisonPlanningEnvironment& environment,
                                            ir::DiagnosticAdmission* diagnostic = nullptr) const;
  // The same completed-trial checks used by commit, without minting a reusable
  // local directive. A whole-program decision terminates cumulative selection.
  std::expected<void, PlanFailure> ValidateTerminalWholeAlternative(
      const ComparisonPlanningEnvironment& environment, const ComparisonTrial& trial,
      const OwnedComparisonPlan& plan, ir::DiagnosticAdmission* diagnostic = nullptr) const;
  std::expected<void, PlanFailure> CommitAcceptedAlternative(
      const ComparisonPlanningEnvironment& environment, const ComparisonTrial& trial,
      const OwnedComparisonPlan& plan, ir::DiagnosticAdmission* diagnostic = nullptr);

 private:
  std::expected<void, PlanFailure> ValidateCompletedAlternative(
      const ComparisonPlanningEnvironment&, const ComparisonTrial&, const OwnedComparisonPlan&,
      ir::DiagnosticAdmission*) const;
  explicit ComparisonDecisionLedger(std::shared_ptr<const ir::StorageFunder> funder)
      : funder_(std::move(funder)) {}
  ir::StorageLoan records_loan_{};
  std::shared_ptr<const ir::StorageFunder> funder_{};
  const ComparisonPlanningEnvironment* environment_ = nullptr;
  std::uint64_t source_identity_ = 0;
  std::uint64_t ledger_identity_ = 0;
  std::uint64_t generation_ = 0;
  base::Sha256Digest source_graph_digest_{};
  base::Sha256Digest source_boundary_digest_{};
  base::Sha256Digest registry_digest_{};
  base::Sha256Digest policy_digest_{};
  base::Sha256Digest device_digest_{};
  base::Sha256Digest shape_digest_{};
  base::Sha256Digest options_digest_{};
  std::vector<ComparisonDecisionRecord> records_{};
};

// Creates a normal plan plus its opt-in committed-ticket trace. An empty
// filter is a fast-path no-trace request unless harness_all is explicit.
std::expected<DiscoveredComparisonPlan, OwnedComparisonFailure> DiscoverComparisonPlan(
    const ir::Graph& source_graph, std::span<const ir::OpKey> operation_filter, bool harness_all,
    std::shared_ptr<const ir::StorageFunder> funder, const execution::Registry& registry,
    const DeviceFacts& device, const Policy& policy, const PlanOptions& options = {});
std::expected<DiscoveredComparisonPlan, OwnedComparisonFailure> DiscoverComparisonPlan(
    const components::ComparisonBoundaryView& source, std::span<const ir::OpKey> operation_filter,
    bool harness_all, std::shared_ptr<const ir::StorageFunder> funder,
    const execution::Registry& registry, const DeviceFacts& device, const Policy& policy,
    const PlanOptions& options = {});

// The caller supplies an actual comparison funder. Capture is opt-in; the
// normal Plan path remains unchanged and allocates no checkpoint.
std::expected<ComparisonInventory, OwnedComparisonFailure> BoundComparison(
    const components::ComparisonBoundaryView& source, const ComparisonSiteSelector& requested,
    std::shared_ptr<const ir::StorageFunder> funder, const execution::Registry& registry,
    const DeviceFacts& device, const Policy& policy, const PlanOptions& options = {},
    const StepShape* shape_override = nullptr, const ComparisonDecisionLedger* decisions = nullptr,
    const ComparisonPlanningEnvironment* environment = nullptr);

// Explicit whole-program comparison of a genuinely accepted checkpoint-only
// site. The selector must retain the discovery checkpoint_match. Generated
// values remain in the checkpoint namespace and cannot authorize local cuts.
std::expected<ComparisonInventory, OwnedComparisonFailure> BoundWholeComparison(
    const components::ComparisonBoundaryView& source, const ComparisonSiteSelector& requested,
    std::shared_ptr<const ir::StorageFunder> funder, const execution::Registry& registry,
    const DeviceFacts& device, const Policy& policy, const PlanOptions& options = {},
    const StepShape* shape_override = nullptr);

std::expected<ComparisonTrial, OwnedComparisonFailure> CaptureComparisonTrial(
    const ComparisonPlanningEnvironment& environment, const ComparisonDecisionLedger& decisions,
    const ComparisonSiteSelector& requested, std::shared_ptr<const ir::StorageFunder> funder);

std::expected<ComparisonTrial, OwnedComparisonFailure> CaptureWholeComparisonTrial(
    const ComparisonPlanningEnvironment& environment, const ComparisonDecisionLedger& decisions,
    const ComparisonSiteSelector& requested, std::shared_ptr<const ir::StorageFunder> funder);

// Checks all mutable site facts against its minted live exact identity.
bool AuthenticComparisonSite(const ComparisonSite& site);

std::expected<GraphAlternative, OwnedComparisonFailure> MakeAlternative(
    const ComparisonInventory& inventory);
std::expected<GraphAlternative, OwnedComparisonFailure> MakeAlternative(
    const ComparisonTrial& trial);

std::expected<OwnedComparisonPlan, OwnedComparisonFailure> PlanAlternative(
    const ComparisonInventory& inventory, GraphAlternative alternative,
    const execution::Registry& registry, const DeviceFacts& device, const Policy& policy,
    const PlanOptions& options = {}, const StepShape* shape_override = nullptr);
std::expected<OwnedComparisonPlan, OwnedComparisonFailure> PlanAlternative(
    const ComparisonPlanningEnvironment& environment, const ComparisonTrial& trial,
    GraphAlternative alternative, const ComparisonDecisionLedger& decisions);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_ALTERNATIVE_H_
