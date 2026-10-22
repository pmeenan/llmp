// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The planner's patterns (D-107; docs/engine-components.md#the-graph-ir):
// one fixed, ordered table. A rewrite pattern matches a subgraph of a
// step's primitive graph and replaces it, through the IR builder, with
// what an implementation computes in one launch: a fused node of op set
// v1 (fused.rms_norm_mul, ...), or primitives in another form (a shared
// input preparation). A batch pattern leaves its nodes in the graph and
// gives a run of them one implementation (grouped KV stores). A pattern
// matches only where the selection view has a candidate for its result,
// so a graph no pattern rewrites stays primitive and correct (D-053's
// primitive-fallback rule, docs/portability.md).
//
// Matching is by dataflow, not by component: nodes match whatever
// component emitted them, and a match's intermediates are read by its own
// nodes only (once each, never an output or kept tensor of the step) and,
// for a fused result, computed in F32 as the fused kind's decomposition
// computes them (a fusion never drops a rounding the graph had). A
// rewrite emits its replacement at the place of the match's last node, so
// every source is made before it and every reader after it; a match is
// refused where that would move a read past a write to the same root, or
// a write past another node's read or write of it (cache stores write
// state, or activations they target, in node order: ir/graph.h).
//
// The table's order is priority: ApplyPatterns applies each enabled
// rewrite pattern over the whole graph, trying it at every node not yet
// taken whose operation anchors it, in node order, before the next pattern
// (so a GLU's products fuse with it before a later pattern can share their
// input). A match takes its nodes. Batch patterns are the selection's
// (plan.h), over the rewritten graph. The table, with what each reproduces
// of today's production plans (kernels/ggml/graph_plan.h DeviceChoices).
// A label says when a pattern applies: "qualified" only where its
// NumericalChange is qualified for the site; "opt-in" (Pattern::opt_in) only
// when selected explicitly (rounded_indexer, owner_row_invariant); no label
// or "default" whenever it matches and an implementation exists:
//
//   spatial_channel_norm fused.spatial_channel_norm: a spatial tensor's
//                      channels L2-normed through channel rows, scaled, times
//                      a weight, SiLU or not, and back (the spatial
//                      components' channel norm)
//   split_modulate     default: fused.layer_norm_modulate_split: a layer norm
//                      times one plus conditioning rows that repeat one
//                      vector over the first rows and another over the rest
//                      (the image's text and latent rows), in BF16 where
//                      only products of BF16 weights read it
//   split_gated_residual default: fused.gated_residual_split: carry plus a
//                      branch times tanh of such conditioning rows
//   bf16_glu           qualified: an MLP's gate and up products of BF16 weights
//                      into BF16 from their input converted once, the
//                      SwiGLU into BF16, the down product from it
//   (round 1, over round 0's results:)
//   split_gated_residual_modulate default: a split residual and the split
//                      modulation of its sum, one pass
//                      (fused.gated_residual_modulate_split); the option
//                      bf16_residual keeps a sum only later residuals carry
//                      in BF16
//   qk_attention       qualified: the image's attention: BF16 Q/K/V products,
//                      the key's sectioned norm and rotation in BF16, a
//                      packed mask and a rotation table each made once a
//                      step, and fused.qk_attention (the query's norm and
//                      rotation inside the attention)
//   bf16_branch        qualified: a product of a BF16 weight only a split
//                      residual adds, into BF16
//   rms_norm_mul_rope  fused.rms_norm_mul_rope: an RMS norm, its weight and
//                      the rotation of the result (fuse_norm_rope; QK norms)
//   rms_norm_mul_rope_sections fused.rms_norm_mul_rope_sections: an RMS
//                      norm, its weight (with its widening) and one to three
//                      rotations of consecutive sections covering the head,
//                      each by its own positions (with their packed copies):
//                      the multi-axis rotation. Opt-in: a
//                      model whose policy disables rms_norm_mul_rope keeps
//                      its primitive rotations; the image enables it
//                      (its mapping's policy, spec/mappings/qwen_image.cc)
//   rms_norm_mul_add   fused.rms_norm_mul_add: an RMS norm, its weight and a
//                      residual add of its shape, one after another (or
//                      the residual's gather between the mul and the add)
//                      (fuse_norm_add)
//   mul_mat_glu        fused.mul_mat_glu: gate and up products of one input
//                      and the GLU over them (fuse_quant_glu)
//   rope_store         fused.rope_store: a rotation stored whole as cache
//                      rows (fuse_rope_store)
//   rms_norm_mul       fused.rms_norm_mul (fuse_norms)
//   owner_attention    fused.owner_attention: every layer's per-owner
//                      attention of a step's joined owners as owner-root
//                      nodes over their own caches, with the owners'
//                      masks joined once (the Gemma builders' owner
//                      decode, owner and flexible owner prefill, bounded
//                      roots and four-root carriers; fattn_owner.h)
//   paired_glu_products two products of one input retain two F32 results
//                      for one GLU, using a native paired-product implementation
//   prepared_inputs    two or more products of one F32 input share one
//                      Q8_1 preparation (kPrepare), each product reading it
//                      (the Gemma builders' shared_q8, shared_q8.h)
//   grouped_stores     batch: consecutive cache stores in one launch, a
//                      capability of fused.grouped_stores with a batch
//                      (group_set_rows, set_rows_group.h)
//   route_reduce       composite: routing (kRoute) and the experts' reduce
//                      (kExpertReduce) each run as an implementation of
//                      the composite op where one takes it (GGML's fused
//                      Gemma routing and scaled reduce: fuse_gemma_route,
//                      fuse_gemma_reduce); disabled, as their
//                      decompositions, llama.cpp's primitives
//
// and the options of owner_attention's forms and of the step's builder,
// each on unless a policy disables it:
//
//   owner_bounded_roots  owners' caches read at their own widths where an
//                        implementation takes that (bounded_roots)
//   owner_padded_reads   narrower owners' caches padded with zeros to the
//                        widest (the builders' common reads); with neither,
//                        owner attention joins only owners of equal reads
//   owner_chained_masks  the owners' masks joined one after another (the
//                        Gemma 2 and 3 builders); disabled, in pairs, the
//                        pairs then joined (Gemma 4's owner transform, at
//                        most four owners)
//   rounded_indexer      opt-in bounded CSA indexer recipe; the component
//                        source requires an owned backend and explicit domain
//   layout_sinking       default: the step's builder removes its layout round
//                        trips before planning (ir/layout.h SinkLayouts,
//                        components/spatial.h sink_layouts): values followed
//                        as views of their storage, materialized only where
//                        a reader needs their layout
//   narrow_final         a step with a head narrows its last row-mixing
//                        component to the output rows (components/step.h
//                        BuildStep; the runners' frontier_head); disabled,
//                        every row runs to the final norm and the head
//                        gathers its rows there

#ifndef LLMP_PLANNER_PATTERN_H_
#define LLMP_PLANNER_PATTERN_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/state_span.h"
#include "planner/step_shape.h"

namespace llmp::planner {

// Every candidate lookup made while matching a rewrite or batch has the
// same qualification contract. This borrowed view cannot silently perform an
// ordinary direct-operation lookup; it adds no tuning/name inventory copy.
class RewriteSelectionView {
 public:
  explicit RewriteSelectionView(const SelectionView& view) : view_(view) {}
  const execution::Registry& registry() const { return view_.registry(); }
  const Policy& policy() const { return view_.policy(); }
  // Describe the incumbent before rewriting. Replacement lookups below still
  // retain automatic-fusion qualification and every final selection constraint.
  ir::ProductArithmetic IncumbentArithmetic(const ir::Graph& graph, const ir::Node& node) const {
    return view_.IncumbentArithmetic(graph, node);
  }
  std::optional<NumericalQualification> Qualification(NumericalChange change,
                                                      const ir::Graph& graph,
                                                      const ir::Node& node) const {
    return view_.Qualification(change, graph, node);
  }
  bool NumericallyQualified(NumericalChange change, const ir::Graph& graph,
                            const ir::Node& node) const {
    return view_.NumericallyQualified(change, graph, node);
  }
  std::vector<std::size_t> Candidates(ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
                                      std::span<const ir::TensorDesc* const> outputs,
                                      std::span<const std::byte> params = {},
                                      bool state_span = false) const {
    return view_.Candidates(key, sources, outputs, params, state_span, true);
  }
  std::vector<std::size_t> Candidates(const ir::Graph& graph, const ir::Node& node,
                                      ir::OpKey key) const {
    return view_.Candidates(graph, node, key, true);
  }
  std::vector<std::size_t> Candidates(const ir::Graph& graph, const ir::Node& node) const {
    return Candidates(graph, node, node.key);
  }

 private:
  const SelectionView& view_;
};

// A graph's readers, as patterns ask about them.
class MatchContext {
 public:
  // `shape`: the step's, where the planner was given it (patterns whose
  // forms follow the step's outputs need it; null: none).
  MatchContext(const ir::Graph& graph, const SelectionView& view, const StepShape* shape = nullptr,
               StateSpans state_spans = {});

  const ir::Graph& graph() const { return graph_; }
  const RewriteSelectionView& view() const { return view_; }
  const StepShape* shape() const { return shape_; }
  const StateSpans& state_spans() const { return state_spans_; }
  std::uint32_t max_decode_rows() const { return view_.policy().max_decode_rows; }
  const ir::Node& node(ir::NodeId id) const { return graph_.node(id); }
  const ir::TensorDesc& tensor(ir::TensorId id) const { return graph_.tensor(id); }
  const ir::Node& producer(ir::TensorId id) const { return graph_.node(graph_.producer(id)); }
  // The nodes reading `t`, an entry per source position, in node order.
  std::span<const ir::NodeId> readers(ir::TensorId t) const;
  // Whether the step returns or keeps `t` (or a view of it).
  bool external(ir::TensorId t) const;
  // Whether the step returns or keeps `t` itself (not another view of its
  // root): a view another one is kept beside may be elided.
  bool kept(ir::TensorId t) const;
  // Whether a match already took the node.
  bool taken(ir::NodeId id) const { return taken_.at(id.v); }
  void Take(std::span<const ir::NodeId> nodes);
  // The node reading `t` when exactly one source of one node does and the
  // step does not return or keep it.
  std::optional<ir::NodeId> OnlyReader(ir::TensorId t) const;

 private:
  const ir::Graph& graph_;
  RewriteSelectionView view_;
  const StepShape* shape_ = nullptr;
  StateSpans state_spans_{};
  std::vector<std::vector<ir::NodeId>> readers_;  // by tensor
  std::vector<bool> external_;                    // by root tensor
  std::vector<bool> kept_;                        // by tensor: returned or kept itself
  std::vector<bool> taken_;                       // by node
};

// What a pattern's match function found for its rewrite beyond a fused
// node's sources and parameters (owner_attention's layers); each pattern
// knows its own.
struct MatchDetail {
  MatchDetail() = default;
  MatchDetail(const MatchDetail&) = default;
  MatchDetail& operator=(const MatchDetail&) = default;
  MatchDetail(MatchDetail&&) = default;
  MatchDetail& operator=(MatchDetail&&) = default;
  virtual ~MatchDetail() = default;
};

struct Match {
  // The primitive nodes it replaces (a rewrite) or joins (a batch), in
  // node order.
  std::vector<ir::NodeId> nodes{};
  // A rewrite's result: its sources (the original graph's ids) and
  // parameters; a batch's implementation.
  std::vector<ir::TensorId> sources{};
  ir::ParamBytes params{};
  std::size_t implementation = 0;
  std::shared_ptr<const MatchDetail> detail{};
  std::optional<NumericalChange> numerical_change{};
  std::optional<NumericalChange> unqualified_numerical{};
  // Every product whose arithmetic changes, in this round's input graph.
  // Empty means the pattern anchor. Equivalent preparation sharing leaves
  // numerical_change absent and needs no changed-arithmetic authorization.
  std::vector<ir::NodeId> numerical_sites{};
  std::vector<ir::NodeId> unqualified_sites{};
};

// Rebuilding a graph with matches replaced: the original graph's tensors
// as the new graph has them, and per match a few tensors of its own (a
// prepared input).
class RewriteContext {
 public:
  RewriteContext(const ir::Graph& original, ir::Builder& builder, std::vector<ir::TensorId>& map,
                 const SelectionView* view = nullptr)
      : original_(original), builder_(builder), map_(map), view_(view) {}
  const ir::Graph& original() const { return original_; }
  ir::Builder& builder() { return builder_; }
  ir::TensorId Map(ir::TensorId original) const {
    return original.valid() ? map_.at(original.v) : ir::TensorId{};
  }
  void Set(ir::TensorId original, ir::TensorId made) { map_.at(original.v) = made; }
  // A node of the original graph made again over the mapped sources.
  ir::Outputs Copy(const ir::Node& node);
  // A match's own tensors, kept across its nodes' rewrites.
  std::vector<ir::TensorId>& local() { return local_; }
  void Reset() { local_.clear(); }
  // Tensors a round's rewrites share, by a key their pattern gives (one
  // prepared mask for every attention of a step that reads it).
  std::map<std::string, ir::TensorId>& shared() { return shared_; }

 private:
  const ir::Graph& original_;
  ir::Builder& builder_;
  std::vector<ir::TensorId>& map_;
  const SelectionView* view_ = nullptr;
  std::vector<ir::TensorId> local_{};
  std::map<std::string, ir::TensorId> shared_{};
};

// kComposite: the selection may take an implementation of its composite
// ops (`composites`) only while it is enabled; disabled, they decompose.
// kOption: a switch another pattern (`parent`) or the step's builder
// reads; it rewrites nothing itself.
enum class PatternKind : std::uint8_t { kRewrite, kBatch, kComposite, kOption };

struct Pattern {
  std::string_view name{};
  std::uint16_t version = 1;
  PatternKind kind = PatternKind::kRewrite;
  // A rewrite's result (the fused kind it makes, or the primitive it
  // adds); a batch's capability key.
  ir::OpKey result{};
  std::span<const ir::OpKey> composites{};
  std::string_view parent{};  // an option's: the pattern it shapes, or empty
  std::span<const ir::Op> anchors{};
  std::optional<Match> (*match)(const MatchContext& context, ir::NodeId anchor) = nullptr;
  // Its match function checks what the table checks for a fused result
  // (F32 intermediates, the order of reads and writes) itself: a rewrite
  // that emits at several of its nodes' places (owner_attention).
  bool own_checks = false;
  // A rewrite's: what replaces the match's node `index` (each of its nodes
  // in turn, in node order), mapping the original outputs it stands for.
  void (*rewrite)(RewriteContext& context, const Match& match, std::size_t index) = nullptr;
  // Off unless explicitly selected: a numerical representation whose
  // same-quality domain is qualified separately, or a caller requirement.
  // Structural fusions and proved incumbent preparation sharing default on.
  bool opt_in = false;
  std::optional<NumericalChange> numerical_change{};
  // The rewrite's round: round 0 matches the step's primitive graph, round
  // 1 round 0's result (a pattern over another's fused nodes).
  std::uint8_t round = 0;
};

// The table, in its fixed order.
std::span<const Pattern> Patterns();
const Pattern* FindPattern(std::string_view name);

// Whether the selection may take an implementation of composite op `key`
// under `policy`: false where a disabled composite pattern names it.
bool CompositeImplemented(const Policy& policy, ir::OpKey key);

// The enabled patterns' names and versions in the table's order: a plan
// key's pattern input (step_shape.h PlanKey::patterns).
base::Sha256Digest PatternsDigest(const Policy& policy);

// A rewrite applied: the pattern, the original nodes it replaced and the
// nodes it made.
// Observed graph facts, never a qualification or evidence grant. Bound model
// identity is spec::BoundDigest (whole actual artifact composition); model_spec
// is the actual semantic ModelSpec digest. Zero means unavailable, independent
// of a candidate's deliberately zero grant identities.
struct NumericalObservation {
  std::uint16_t active_stage = UINT16_MAX;
  std::optional<ir::WeightRef> weight{};
  format::TensorFormat operand_format{}, leaf_format{}, input_format{};
  std::array<std::uint64_t, 4> operand_strides{}, leaf_strides{}, input_strides{};
  base::Sha256Digest bound_model{}, model_spec{};
};
struct NumericalProof {
  ir::NodeId site{};  // original primitive graph ID, composed across rounds
  // For a compound site, distinguish its two original product weight operands.
  std::uint8_t weight_operand = 0;
  NumericalQualification qualification{};
  QueryShape actual{};
  bool candidate = false;
  bool grouped_projection = false;
  NumericalObservation observed{};
};
struct Applied {
  std::string_view pattern{};
  std::vector<ir::NodeId> replaced{};
  std::vector<ir::NodeId> made{};
  std::optional<NumericalChange> numerical_change{};
  base::Sha256Digest numerical_evidence{}, numerical_source{}, numerical_semantic{};
  bool numerical_candidate = false;
  bool numerical_incumbent = false;
  // One authenticated proof (or explicit candidate) per changed recipient.
  std::vector<NumericalProof> numerical_proofs{};
};
struct MissingNumericalQualification {
  NumericalChange change{};
  std::uint32_t device_arch = 0;
  std::uint16_t component = 0;
  ir::OpKey anchor{};
  bool operator==(const MissingNumericalQualification&) const = default;
};

// A match whose rewrite the builder refused (an emission's shape or format
// rule): dropped, so its nodes stay primitive, and the round retried
// without it. `anchor` is the node the pattern matched at, of the graph its
// round rewrote (composed onto the primitive graph by Plan).
struct DroppedRewrite {
  std::string_view pattern{};
  ir::NodeId anchor{};
  std::string detail{};
};

struct Rewritten {
  ir::Graph graph;
  std::vector<Applied> applied;
  // The new graph's node i stands for the original's from[i] (a rewrite's
  // node: the original node at whose place it was made).
  std::vector<ir::NodeId> from;
  // The original graph's tensor i in the new graph, or none (an elided
  // intermediate).
  std::vector<ir::TensorId> tensors;
  std::vector<MissingNumericalQualification> missing_numerical{};
  std::vector<DroppedRewrite> dropped{};
};

// The enabled rewrite patterns applied over `graph` (see the header), for
// a step of `shape` where it is given. A match whose rewrite fails is
// dropped (Rewritten::dropped) and its round applied again without it, so
// a bad fusion leaves its nodes to the primitives instead of failing the
// step; an error no match made fails.
std::expected<Rewritten, ir::VerifyError> ApplyPatterns(const ir::Graph& graph,
                                                        const SelectionView& view,
                                                        const StepShape* shape = nullptr,
                                                        StateSpans state_spans = {},
                                                        std::span<const ir::NodeId> preserve = {});
// ApplyPatterns with `table`'s rewrites in place of the built-in table's
// (a test's own patterns).
std::expected<Rewritten, ir::VerifyError> ApplyPatternTable(
    std::span<const Pattern> table, const ir::Graph& graph, const SelectionView& view,
    const StepShape* shape = nullptr, StateSpans state_spans = {},
    std::span<const ir::NodeId> preserve = {});

// The batch a batch pattern makes at `anchor` of `graph`, where it matches
// and is enabled.
std::optional<Match> BatchAt(const Pattern& pattern, const MatchContext& context,
                             ir::NodeId anchor);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_PATTERN_H_
