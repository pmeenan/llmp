// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Lowering a planned IR step onto the GGML module (D-107;
// docs/engine-components.md, "The planner, as built"): the IR is the source
// of truth, and a ggml_cgraph exists only here, as the private target that
// lets the module's planning, placement checks, binding and capture run
// unchanged. Private to kernels/ggml: the engine sees the step program
// (step_program.h) and never a GGML type.
//
// LowerStep, from a StepPlan (planner/plan.h) and the primitive step it was
// planned from:
//   1. makes a GGML tensor for each tensor of the planned graph in a
//      TensorArena, in node order: leaves for the step's inputs, weights and
//      state, GGML's own builder call for each node, and for a fused node
//      the GGML node sequence its implementation covers (the primitive
//      nodes the rewrite replaced: GGML's fused implementations run over
//      them, in the selected recipe), and for a composite op an
//      implementation of its own runs (planner/pattern.h route_reduce) the
//      GGML node sequence that implementation's gate takes (Gemma's fused
//      routing and scaled reduce). Node order is the planned graph's,
//      which keeps the builders' (GraphOrder) order the components emit, so
//      the graph is the one today's builders make for the same step;
//   2. makes each mask (kMaskBuild) on the device (llmp.causal.mask, whose
//      producer derives the cells' positions, so the step stages none; a
//      frozen read's under its frozen policy) or, with device masks off, a
//      host-built input (step_program.h kMask);
//   3. authenticates physical materializations and reads from the selected
//      recipes, retaining proof-only dependencies in the IR; a descriptor-only
//      provisional binding makes group routing checkable without submitting work;
//   4. emits the selected implementations' explicit launch recipes, resolving
//      only declared groups. It never runs the independent native PlanGraph;
//   5. places the real schedule, including paired products and extra reads,
//      then authenticates final addresses, typed operands and protected owner
//      recipes. Provisional addresses grant no execution authority;
//   6. gives the input slots (tokens, positions, the head's rows, each
//      owner's store cells, a target's tap cells and host masks, a
//      drafter's features, which the engine copies in on the device), the
//      output slots (each owner's logits rows, or its greedy token; a
//      drafter's next features after them, marked as features) and every
//      tensor use for the coverage check.
// Every profile builds it (no launch); kernels/ggml/step_program.h makes a
// program of the result on CUDA builds.

#ifndef LLMP_KERNELS_GGML_LOWER_H_
#define LLMP_KERNELS_GGML_LOWER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "execution/registry.h"
#include "format/encoding.h"
#include "ggml.h"
#include "ir/graph.h"
#include "kernels/common/lower.h"
#include "kernels/common/routed_program.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/selected_operation.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

// Independent of numerical execution declarations. The backend prepays this
// immutable callback owner before allocating it; each native build keeps it
// alive until retained-plan accounting takes over or native teardown completes.
using ConstructionFunder = kernels::ConstructionFunder;
struct ConstructionPayment {
  ConstructionPayment() = default;
  ConstructionPayment(const ConstructionPayment&) = delete;
  ConstructionPayment& operator=(const ConstructionPayment&) = delete;
  std::shared_ptr<const ConstructionFunder> funder;
  std::uint64_t bytes = 0;
  ~ConstructionPayment() {
    if (bytes != 0) funder->release(bytes);
  }
};
struct LowerConstructionInventory {
  std::size_t descriptors = 0;
  std::uint64_t bytes = 0;
  std::uint64_t comparison_bytes = 0;
};
// Allocation-free checked inventory of the actual lowering descriptor bound.
// Both backend admission and TensorArena sizing consume this same operation
// inventory. It is a bound, not a measured memory peak. Standalone LowerStep
// callers must supply their own enclosing prepayment; production uses Lower.
std::expected<LowerConstructionInventory, KernelFailure> LowerConstructionOf(
    const LowerInput& input);

// Checked upper bound on the source graph's input/output edge occurrences.
// Region value ports are derived from these occurrences, not unique tensors.
std::optional<std::uint64_t> ComparisonRegionEdgeOccurrenceCapacity(
    std::span<const ir::Node> source_nodes) noexcept;

// Checked upper bound for typed local access facts. Most operations contribute
// their source/output edge occurrences; an ordinary MMVQ may add the private
// Q8 write/read pair beyond those IR edges.
std::optional<std::uint64_t> ComparisonRegionAccessCapacity(
    std::span<const ir::Node> planned_nodes) noexcept;

using kernels::LowerOptions;

// A leaf of the lowered graph and the primitive step's tensor it stands for
// (an input, a host-built mask's kMaskBuild output, a weight, a state).
// A checked native ExpertLayout slab, composed of four truthful part ranges.
// Populated only by whole-comparison construction; no residency authority.
struct ComparisonExpertSlab {
  std::array<std::uint32_t, 4> bindings{};
  std::array<ir::TensorId, 2> roots{};
  std::uint64_t width = 0, hidden = 0, experts = 0;
};

// An owned physical-placement fact for an ordinary staged Input. The original
// logical binding remains in the source boundary; this records its actual slice.
struct ComparisonStagedInput {
  ir::TensorId planned{};
  ir::NodeId producer{};
  std::uint64_t activation_base = 0, activation_bytes = 0;
  // bytes covers the full paid root, including native auxiliary storage;
  // input_bytes is exactly the visible descriptor and staged upload.
  std::uint64_t logical_offset = 0, physical_offset = 0, bytes = 0, input_bytes = 0, alignment = 0;
  std::uint32_t allocation_class = 0;
  ggml_type type = GGML_TYPE_COUNT;
  std::uint8_t segment = ir::kJoinedSegment;
  InputKind kind = InputKind::kTokens;
};

struct LoweredLeaf {
  ggml_tensor* tensor = nullptr;
  ir::TensorId primitive{};
  // A weight of several parts (MXFP8's codes and scales, NVFP4 rows'
  // data and tensor scale, a CUTLASS expert slab: its gate-and-up codes):
  // the part this leaf is.
  std::optional<format::PartRole> part{};
  std::optional<ComparisonExpertSlab> comparison_slab{};
  std::optional<ComparisonStagedInput> comparison_staged{};
};

// A verify's save's buffer (components/step.h CommitSave): F32 [width,
// rows], bound at Addresses::save's.
struct SavedLeaf {
  ggml_tensor* tensor = nullptr;
  // Exact planned operands and the generated SET_ROWS descriptor, retained
  // while CommitSave is still available during construction.
  ir::TensorId source{};
  ir::TensorId rows{};
  // Directional source-graph identities. `source` and `rows` above are the
  // final planned IDs and may differ between an incumbent and a rewrite.
  ir::TensorId original_source{};
  ir::TensorId original_rows{};
  ggml_tensor* store = nullptr;
  // A nonpacked source is copied into this exact CONT output before SET_ROWS.
  // When generated, its retained slot is in the activation tail, after the
  // planner-owned placement; it has no fabricated IR tensor identity.
  ggml_tensor* packed = nullptr;
  std::uint64_t scratch_offset = 0;
  std::uint64_t scratch_bytes = 0;
  // Optional exact owner interval captured with the source CommitSave tuple.
  // A missing or stale interval refuses comparison projection, not lowering.
  std::optional<BoundedAddress> owner_range{};
  bool owner_range_current = false;
  std::uint32_t layer = 0;
  std::uint16_t component = 0;
  std::string what;
  std::uint8_t segment = 0;
};

struct ComparisonInputFact {
  ir::TensorId original_tensor{};
  bool token_input = false;
  std::optional<components::InputKind> token_kind{};
  std::uint8_t token_segment = ir::kJoinedSegment;
  std::uint16_t token_component = 0;
  std::uint64_t token_read = 0;
  std::optional<components::ComparisonUnstoredInputReason> unstored_reason{};
  bool unstored_verified = false;
};

// Native whole-program input whose exact planned Input producer is absent
// from the backend's authenticated Needed closure. The source binding remains
// present; this record is never a local-cut or caller-supplied exemption.
struct ComparisonOmittedInput {
  ir::TensorId original{}, planned{};
  ir::NodeId producer{};
};

struct LoweredRouting {
  kernels::RoutedLayout layout;
  std::vector<std::size_t> members;
  std::vector<kernels::RoutedCopy> uploads;
  std::vector<std::pair<std::size_t, std::uint64_t>> descriptors;  // selected index, pinned offset
  ggml_tensor* preparation = nullptr;
  // The preparation's actual activation-tail placement, relative to the
  // common activation base. Never a process address in a report.
  std::uint64_t activation_offset = 0, activation_bytes = 0;
  std::size_t cut = 0;
  base::Sha256Digest graph_identity{};
};
// Actual arena descriptor identity. The pointer is an internal lookup key,
// never serialized; id is its stable construction ordinal. Unmapped auxiliary
// descriptors retain their owning planned producer without inventing an IR value.
struct PhysicalTensor {
  const ggml_tensor* descriptor = nullptr;
  std::uint32_t id = 0;
  ir::TensorId tensor{}, root{};
  ir::NodeId producer{};
  std::optional<format::PartRole> part{};
  // The planned value's semantic format, when this descriptor maps to one.
  // Backend-only descriptors retain their actual GGML dtype/shape/strides.
  std::optional<format::TensorFormat> ir_format{};
};
struct PhysicalRoot {
  ir::TensorId root;
  ir::NodeId producer;
  ir::OpKey operation;
  std::uint16_t component = 0;
  std::uint32_t backend_tensor = UINT32_MAX;
  std::uint64_t offset = 0, bytes = 0;
  std::array<std::uint64_t, 4> ne{}, nb{};
  format::Encoding encoding = format::Encoding::kF32;
};
enum class ComparisonRegionAccessDomain : std::uint8_t { kActivation, kExternal, kPool };
struct ComparisonRegionAccessFact {
  // The address is transient overlap evidence and is cleared before projection
  // returns. It is never included in a recipe digest or exposed to callers.
  std::uintptr_t address = 0;
  std::uint32_t call_ordinal = 0;
  std::uint8_t subcall_ordinal = 0;
  std::uint32_t physical_local = UINT32_MAX;
  // Actual lowered descriptor ID for synchronous remainder-liveness checks.
  // This is transient and is never serialized into a recipe key.
  std::uint32_t physical_id = UINT32_MAX;
  std::uint32_t tensor_local = UINT32_MAX;
  std::uint32_t root_local = UINT32_MAX;
  std::uint32_t allocation_class = 0;
  std::uint64_t bytes = 0;
  std::uint64_t required_alignment = 0;
  std::uint64_t root_offset = 0;
  std::uint64_t domain_offset = 0;
  ComparisonRegionAccessDomain domain = ComparisonRegionAccessDomain::kActivation;
  bool read = false;
  bool written = false;
  // The exact destination of an authenticated CommitSave SET_ROWS call has
  // an owner range but no planned IR tensor or PhysicalTensor root.
  bool saved_destination = false;
};
struct ComparisonLifetimeFact {
  std::uintptr_t address = 0;
  // storage_id describes allocation identity for overlap checks; definition_id
  // describes the logical value materialized by an actual kernel call. Pure
  // metadata views inherit a definition, materializing ops do not.
  std::uint32_t storage_id = UINT32_MAX;
  std::uint32_t definition_id = UINT32_MAX;
  std::uint64_t storage_offset = 0;
  std::uint64_t bytes = 0;
  std::size_t first_write_call = SIZE_MAX;
  std::uint8_t first_write_subcall = 0;
  std::size_t first_read_call = SIZE_MAX;
  std::uint8_t first_read_subcall = 0;
  std::size_t last_read_call = 0;
  std::uint8_t last_read_subcall = 0;
  bool has_read = false;
  bool external_source = false;
  bool kept = false;
};
struct LoweredStep {
  bool whole_checkpoint_only = false;
  std::optional<TensorArena> arena;  // owns every tensor below
  std::vector<ggml_tensor*> nodes;   // computed nodes and views, in execution order
  std::vector<ggml_tensor*> keep;    // the step's outputs and kept tensors
  std::vector<LoweredLeaf> leaves;
  std::vector<SavedLeaf> saves;
  std::vector<SelectedNode> selected;  // exact selected calls, borrowing this arena
  GraphPlan plan;                      // explicit lowered schedule, lanes and regions
  std::vector<InputSlot> inputs;
  std::vector<OutputSlot> outputs;
  std::vector<TensorUse> tensors;
  std::vector<PhysicalRoot> physical_roots;
  // Sorted by internal descriptor pointer for allocation-free logarithmic
  // ID lookup while lazily reporting actual concrete nodes/source edges.
  std::vector<PhysicalTensor> physical_tensors;
  std::size_t descriptor_bound = 0;  // canonical prepaid construction inventory
  std::uint64_t extent = 0;          // the activations' (the IR placement's)
  std::uint64_t logical_extent = 0;  // conservative semantic-IR placement, for diagnostics
  // Sum of omitted logical root sizes, not a peak-memory saving.
  std::uint64_t unmaterialized_logical_bytes = 0;
  std::uint64_t unmaterialized_roots = 0;
  bool measuring = false;  // placeless: never bound to run
  // The plan's prologue (planner/plan.h StepPlan::prologue) as `plan`'s
  // step indices, in order: what a warm run skips.
  std::vector<std::size_t> prologue{};
  // Payment outlives the routed vectors (members are destroyed in reverse).
  std::shared_ptr<const void> routing_payment{};
  std::vector<LoweredRouting> routing{};
  base::Sha256Digest routing_identity{};
  std::uint64_t routed_pinned_bytes = 0;
  std::optional<std::vector<kernels::RoutedWeight>> weight_reads{};
  std::optional<kernels::ComparisonRecipeSeal> comparison{};
  // Full directional provenance captured synchronously from the planner.
  // Each planned-node ordinal owns the complete original source-node set at
  // offsets[i..i+1]; actual calls resolve through their physical producers.
  // The paid construction envelope owns all vectors for their lifetime.
  struct ComparisonOrigins {
    std::uint64_t source_identity = 0;
    std::uint64_t planned_identity = 0;
    std::vector<std::uint64_t> offsets;
    std::vector<ir::NodeId> source_nodes;
    std::vector<std::uint8_t> membership;
    std::vector<ir::NodeId> expansion;
    // The final IR is temporary to LowerInput. Copy exact typed facts once so
    // repeated region expansion can canonically reproject without replanning.
    std::vector<ir::Node> planned_nodes;
    std::vector<ir::TensorDesc> planned_tensors;
    std::vector<ir::TensorId> original_to_planned;
    // Minimal source input provenance. A token input without a classified
    // StepInput remains explicitly unknown; spatial GraphIO inputs carry no
    // token staging semantics.
    std::vector<ComparisonInputFact> input_facts;
    // Exact semantic request ranges before LowerFor normalizes a single-part
    // weight's physical leaf part. These authenticate the source boundary;
    // comparison_bindings below remain the physical-leaf view used at Bind.
    std::vector<kernels::ComparisonBindingRange> original_bindings;
    std::vector<std::uint32_t> node_local;
    std::vector<std::uint32_t> tensor_local;
    std::vector<std::uint32_t> physical_local;
    std::vector<kernels::ComparisonRegionValuePort> value_ports;
    std::vector<ComparisonRegionAccessFact> accesses;
    // Temporary descriptor-indexed storage proof table. It is populated and
    // cleared synchronously by remainder analysis, with one prepaid entry per
    // actual physical descriptor.
    std::vector<ComparisonLifetimeFact> lifetimes;
    base::Sha256Digest backend_device{};
    std::uint64_t projection_serial = 0;
    std::uint16_t active_stage = 0;
    bool source_boundary_verified = false;
    bool descriptor_bindings_verified = false;
    bool bindings_verified = false;
  };
  mutable std::optional<ComparisonOrigins> comparison_origins{};
  // Native whole selected-call identity needs the exact lowered IR even when
  // no local-region provenance was requested. Node parameters and tensor
  // descriptors are value-owned; this grants no origin or local-cut proof.
  struct SelectedComparisonGraph {
    std::vector<ir::Node> planned_nodes;
    std::vector<ir::TensorDesc> planned_tensors;
  };
  std::optional<SelectedComparisonGraph> comparison_selected_graph{};
  struct GeneratedStoreRecipe {
    InputSlot slot{};
    std::uint16_t stage = 0;
    planner::Phase phase = planner::Phase::kPrefill;
    planner::Segment segment{};
    bool row_invariant = false;
    std::uint32_t read_align = 0;
  };
  std::optional<GeneratedStoreRecipe> generated_store{};
  bool generated_store_access_closed = false;
  // Sanitized copies only: base is zeroed before the request expires.
  std::vector<kernels::ComparisonBindingRange> comparison_bindings{};
  std::vector<ComparisonOmittedInput> comparison_omitted_inputs{};
  bool comparison_bindable = false;
};

// Recheck an immutable backend omission fact against the finalized native
// inventory. Construction separately joins its exact source/planned Input IDs.
bool CheckComparisonOmittedInput(const LoweredStep&, ir::TensorId original);

const PhysicalTensor* PhysicalTensorOf(const LoweredStep& step, const ggml_tensor* tensor);
struct PhysicalReportMetrics {
  std::uint64_t pool_bytes = 0, lanes = 0, lane_scratch_bytes = 0, launches = 0;
};
// The same allocation-free lazy serializer is built in every profile.
void StreamPhysicalPlanReport(const LoweredStep& step, const execution::Registry& registry,
                              const kernels::PlanReportSink& sink,
                              const PhysicalReportMetrics& metrics = {});

// The bytes the module writes for a root of `graph` beyond the tensor's
// own (planner/placement.h RootBytesOf): a prepared Q8_1 input, which
// QuantizeQ8 writes with each row padded to MMVQ's row step (llmp_ops.h
// Q8Bytes); 0 for any other.
std::uint64_t WrittenBytes(const ir::Graph& graph, ir::TensorId root);

// The root the module writes a root inside (placement.h RootWithin): a
// kernel's outputs laid out in one blob at its first output's place
// (llmp.hc.prep, llmp.hc.mix_bf16's prepared and rounded rows,
// llmp.moe.router, llmp.gdn.gates), or none.
ir::TensorId WithinRoot(const ir::Graph& graph, ir::TensorId root);

// Backend indexing limits (RE-037), checked before creating any GGML tensor.
// A wide graph remains valid IR and may be accepted by another backend.
std::expected<void, KernelFailure> CheckIrIndexBounds(const ir::Graph& graph);

// Steps 1 to 6 above, with `device` the device's own choices (executor.h
// DeviceChoicesOf on CUDA builds; a model of it in tests) and `registry` the
// one the plan selected from.
std::expected<LoweredStep, KernelFailure> LowerStep(const LowerInput& input,
                                                    const execution::Registry& registry,
                                                    const DeviceChoices& device,
                                                    const LowerOptions& options = {});

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_LOWER_H_
