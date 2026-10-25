// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Planning one step (D-053, D-107; docs/engine-components.md#the-graph-ir):
// from a step's primitive graph (components/step.h BuildStep), the
// registry, the device's facts and the alias's policy to a StepPlan that a
// lowering (kernels/common/step_program.h) turns into a program:
//
//   1. the enabled rewrite patterns applied (pattern.h);
//   2. every launching node selected (select.h): its first candidate, or
//      with a batch pattern a run of nodes under one implementation. A
//      node without a candidate whose op has a primitive decomposition
//      (every fused kind and composite op, ir/decompose.h) is replaced by
//      it and the graph selected again (the primitive-fallback rule,
//      docs/portability.md); one without either refuses the plan;
//   3. lanes (lanes.h), where the caller tags nodes;
//   4. the activations placed (placement.h).
//
// Deterministic: one graph, registry, device and policy give one plan.
// The planned graph, its steps and placement name the planned graph's
// tensors and nodes; `tensors` and `from` relate them to the primitive
// graph's (the step's inputs, weights and state are the primitive's ids).
//
// KeyOf makes the plan cache's key (step_shape.h PlanKey): the spec's
// digest, the step's shape, the enabled patterns and the policy, and the
// partition.

#ifndef LLMP_PLANNER_PLAN_H_
#define LLMP_PLANNER_PLAN_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "ir/graph.h"
#include "planner/lanes.h"
#include "planner/owner_products.h"
#include "planner/pattern.h"
#include "planner/placement.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::planner {

class ComparisonCaptureSession;
class ComparisonOriginDag;
class PlannerMetadataAdmission;
struct ComparisonFinalNodeOrigins;
using ComparisonOriginSetId = std::uint32_t;

enum class PlanError : std::uint8_t {
  kNoImplementation,  // a node no implementation computes and nothing decomposes
  kPattern,           // a rewrite or decomposition the IR refused (a planner bug)
  kPlacement,         // the activations could not be placed
  kBounds,            // past a bound (the decomposition passes)
  kPolicy,            // the policy breaks its own rules
  kAdmission,         // comparison-owned construction payment was refused
  kStaleComparisonSite,
  kAmbiguousComparisonSite,
  kMissingComparisonSite,
};
std::string_view PlanErrorName(PlanError error);

struct PlanFailure {
  PlanError error = PlanError::kNoImplementation;
  std::string detail;
};

struct PlanOptions {
  SelectionView::FusionComparisons fusion_comparisons = SelectionView::FusionComparisons::enabled;
  // A D-092 verify's plan: every implementation row-invariant (also the
  // policy's row_invariant).
  bool row_invariant = false;
  // Every implementation recordable for replay (a captured step).
  bool capturable = false;
  // The inputs the step copies into its activation region (the primitive
  // graph's ids); none given: every input leaf.
  std::optional<std::vector<ir::TensorId>> staged{};
  // Lanes for the primitive graph's nodes (a node a rewrite replaced
  // passes its tag to what replaced it), and the implementations that stay
  // on the stream.
  LaneTags lanes{};
  std::function<bool(std::string_view)> on_stream{};
  std::uint64_t alignment = 256;
  // What the lowering's implementations write beyond a root's own bytes
  // (placement.h RootBytesOf), and the roots it writes inside another's.
  RootBytesOf written{};
  RootWithin within{};
  // The step's shape, for the patterns whose forms follow its outputs
  // (owner_attention's one-row form only for a step returning every
  // owner's one row); outlives the call. Null: those forms are not taken.
  const StepShape* shape = nullptr;
  StateSpans state_spans{};
  // Explicit immutable allocation facts, queried against the current graph.
  // Component sources seal physical state bindings; missing facts decompose.
  RootAlignment root_alignment{};
  // Physical base guarantee for the activation reservation. Zero is unknown.
  std::uint64_t activation_alignment = 0;
  // When a lowering rebinds some listed inputs externally (commit rows),
  // only this subset receives arena guarantees. Absent: the staged set.
  // This changes no placement/accounting; it describes physical binding.
  std::optional<std::vector<ir::TensorId>> alignment_staged{};
  // Inputs (the primitive graph's ids) whose values the caller keeps the
  // same between consecutive runs of the plan: the steps that read only
  // them, weights and each other's results form the plan's prologue
  // (StepPlan::prologue), whose results the rest reads stay in place
  // between runs, so a run after a run of the same plan may skip the
  // prologue and these inputs' copies (kernels/common/step_program.h
  // RunWarm). Ignored with lanes (the regions run in their own order).
  // KeyOf does not hold it: the image's stages (the only callers) key their
  // plans by graph and stage, so each graph has one declaration. A caller
  // whose plans share a PlanKey (an LLM's steps) must add it to the key
  // before it declares invariant inputs.
  std::vector<ir::TensorId> invariant{};
  // Proved physical owner columns, expanded before width-sensitive patterns.
  // Null keeps existing planning unchanged. The proof outlives this call.
  const OwnerProducts* owner_products = nullptr;
  // Report-only identities supplied by the actual component source, never
  // inspected by selection/qualification or parsed as policy permission.
  base::Sha256Digest observed_bound_model{}, observed_model_spec{};
  // Internal, synchronous comparison capture. Public callers use the owned
  // APIs in alternative.h; ordinary plans leave this null.
  ComparisonCaptureSession* comparison_capture = nullptr;
  ComparisonOriginDag* comparison_origins = nullptr;
  std::span<const ComparisonOriginSetId> comparison_node_origins{};
  // Directional original-source maps for an owned alternative seed. Ordinary
  // and incumbent plans leave these empty, in which case capture uses identity.
  std::span<const ir::TensorId> comparison_source_to_graph{};
  std::span<const ir::NodeId> comparison_graph_node_to_source{};
  // Comparison-only retained metadata admission. Ordinary planning leaves
  // this null and keeps its existing allocation behavior.
  PlannerMetadataAdmission* metadata_admission = nullptr;
  // Optional comparison-only sink populated by a successful tracked Plan.
  // The wrapper supplies the authenticated original source identity; Plan
  // moves its existing paid final lineage vector into this owner at return.
  ComparisonFinalNodeOrigins* comparison_final_lineage = nullptr;
};

// Opaque authority minted only by an authenticated whole-comparison capture.
// It never grants a local region projection or a caller-declared access fact.
class WholeComparisonCheckpoint {
 public:
  bool whole_only() const { return source_identity_ != 0; }
  const base::Sha256Digest& subject() const { return subject_; }
  bool source_matches(std::uint64_t source) const {
    return whole_only() && source_identity_ == source;
  }
  bool belongs_to(std::uint64_t source, std::uint64_t checkpoint) const {
    return whole_only() && source == source_identity_ && checkpoint == checkpoint_identity_;
  }

 private:
  friend struct ComparisonCheckpointAuthorityAccess;
  std::uint64_t source_identity_ = 0, checkpoint_identity_ = 0;
  base::Sha256Digest subject_{};
};

struct StepPlan {
  WholeComparisonCheckpoint whole_checkpoint{};
  base::Sha256Digest primitive_digest{};
  // Computed when planning finishes; report capture never hashes the graph.
  base::Sha256Digest graph_digest{};
  base::Sha256Digest policy_digest{};
  ir::Graph graph;               // the planned graph
  std::vector<Selected> steps;   // in node order
  std::vector<Region> regions;   // lanes' spans of steps
  Placement placement;           // the planned graph's roots
  std::vector<Applied> applied;  // rewrites, over the primitive graph's nodes
  // The planned graph's node i stands for the primitive graph's from[i];
  // the primitive graph's tensor i is the planned graph's tensors[i] (none
  // where a rewrite elided it).
  std::vector<ir::NodeId> from;
  std::vector<ir::TensorId> tensors;
  // Primitive decompositions taken for want of an implementation (their
  // keys' names), in order.
  std::vector<std::string> decomposed;
  std::vector<MissingNumericalQualification> missing_numerical{};
  // Matches whose rewrite failed, left to the primitives (pattern.h
  // ApplyPatterns), anchored at the primitive graph's nodes.
  std::vector<DroppedRewrite> dropped_rewrites{};
  // Several one-row owners retain each owner's scalar product arithmetic
  // (the policy's owner_row_invariant option). Multirow input products retain
  // ordinary batched arithmetic; a selected-output basis may be proved below.
  // This is independent of an exact verify's
  // explicit row_invariant requirement on implementation selection.
  bool owner_row_invariant = false;
  // The enabled policy, independent of the all-input-rows-one fast path.
  // Lowering may prove a narrower selected-output basis for individual ops.
  bool owner_rows_policy_enabled = false;
  ProductPolicy products{};  // independent choices, included in the policy digest
  // Promises used by selection, authenticated again at final placement/binding.
  std::vector<RootAlignmentFact> root_alignments{};
  // Exact products retained through rewrites; lowering authenticates recipes.
  std::vector<OwnerProductScope> owner_products{};
  base::Sha256Digest owner_digest{};
  // PlanOptions::invariant's (the primitive graph's ids) and the prologue:
  // the steps that read only them, weights and each other's results, in
  // order (none without invariant inputs).
  std::vector<ir::TensorId> invariant{};
  std::vector<std::uint32_t> prologue{};
};

// On-demand JSON-lines diagnostics of the actual plan, in execution order.
// Includes identities and shapes, never tensor names, constants, input values,
// addresses or prompt/state contents. The caller owns sink storage and admission.
using PlanReportSink = std::function<void(std::string_view)>;
// Allocation-free serialization; a record may span many fragments and each ends
// with a newline. The borrowed sink consumes every fragment synchronously. Count
// and fill the same stream to fund exactly its inventory, without a global cap.
void StreamPlanReport(const StepPlan& plan, const execution::Registry& registry,
                      const PlanReportSink& sink);
// Convenience wrapper: assembles each complete record, without its newline,
// before calling the sink. May allocate; serving uses StreamPlanReport instead.
void ReportPlan(const StepPlan& plan, const execution::Registry& registry,
                const PlanReportSink& sink);

// Owned, compact report facts. Capture copies no tensors/parameters/payloads;
// every string and identity survives the temporary graph and registry. One
// allocation holds exact arrays and a string pool, with no growth capacities.
// The caller pays StorageBytes plus its wrapper BEFORE Capture, and retains
// that payment until destruction or transfer to the installed plan's charge.
class PlanReportInventory {
 public:
  static std::expected<std::uint64_t, std::string> StorageBytes(
      const StepPlan& plan, const execution::Registry& registry);
  static std::expected<PlanReportInventory, std::string> Capture(
      const StepPlan& plan, const execution::Registry& registry, std::uint64_t bytes);
  void Stream(const PlanReportSink& sink) const;
  std::uint64_t storage_bytes() const { return storage_bytes_; }

 private:
  std::expected<std::uint64_t, std::string> Layout(const StepPlan& plan,
                                                   const execution::Registry& registry,
                                                   std::byte* storage);
  struct Implementation {
    std::string_view name;
    base::Sha256Digest identity{};
    unsigned rung = 0, expansion = 0;
  };
  struct Node {
    ir::NodeId id;
    ir::OpKey key;
    std::optional<ir::ProductArithmetic> arithmetic{};
    std::uint16_t component = 0;
    std::span<ir::TensorId> sources{}, outputs{};
  };
  struct Selection {
    std::size_t implementation = 0;
    std::uint32_t lane = 0;
    std::string_view group;
    std::span<Node> nodes;
  };
  struct PatternRecord {
    std::string_view name;
    std::uint64_t replaced = 0, made = 0;
    std::optional<NumericalChange> numerical_change{};
    base::Sha256Digest evidence{}, source{}, semantic{};
    bool numerical_candidate = false;
    bool numerical_incumbent = false;
    std::span<NumericalProof> numerical_proofs{};
  };
  struct Activation {
    ir::TensorId root;
    ir::NodeId producer;
    ir::OpKey operation;
    std::uint16_t component = 0;
    std::uint64_t offset = 0, bytes = 0;
    std::array<std::uint64_t, 4> shape{}, strides{};
  };
  base::Sha256Digest primitive_{}, graph_{}, policy_{}, registry_{};
  std::uint64_t activation_bytes_ = 0, storage_bytes_ = 0;
  std::unique_ptr<std::byte[]> storage_;
  std::span<Implementation> implementations_;
  std::span<Selection> selections_;
  std::span<PatternRecord> patterns_;
  std::span<std::string_view> fallbacks_;
  std::span<Activation> activations_;
  std::span<MissingNumericalQualification> missing_;
};

std::expected<StepPlan, PlanFailure> Plan(ir::Graph primitive, const execution::Registry& registry,
                                          const DeviceFacts& device, const Policy& policy,
                                          const PlanOptions& options = {});
std::expected<StepPlan, PlanFailure> Plan(std::expected<ir::Graph, ir::VerifyError> primitive,
                                          const execution::Registry& registry,
                                          const DeviceFacts& device, const Policy& policy,
                                          const PlanOptions& options = {});

// A plan's key: `spec` the model's bound digest (spec.h BoundDigest).
PlanKey KeyOf(const base::Sha256Digest& spec = {}, const StepShape& shape = {},
              const Policy& policy = {}, std::uint32_t partition = 0);

PlanKey KeyOf(const base::Sha256Digest& spec, const StepShape& shape,
              const base::Sha256Digest& policy_digest,
              const base::Sha256Digest& patterns_digest = {}, std::uint32_t partition = 0);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_PLAN_H_
