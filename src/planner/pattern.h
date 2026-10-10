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
// of today's production plans (kernels/ggml/graph_plan.h DeviceChoices):
//
//   rms_norm_mul_rope  fused.rms_norm_mul_rope: an RMS norm, its weight and
//                      the rotation of the result (fuse_norm_rope; QK norms)
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
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::planner {

// A graph's readers, as patterns ask about them.
class MatchContext {
 public:
  // `shape`: the step's, where the planner was given it (patterns whose
  // forms follow the step's outputs need it; null: none).
  MatchContext(const ir::Graph& graph, const SelectionView& view, const StepShape* shape = nullptr);

  const ir::Graph& graph() const { return graph_; }
  const SelectionView& view() const { return view_; }
  const StepShape* shape() const { return shape_; }
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
  const SelectionView& view_;
  const StepShape* shape_ = nullptr;
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
};

// Rebuilding a graph with matches replaced: the original graph's tensors
// as the new graph has them, and per match a few tensors of its own (a
// prepared input).
class RewriteContext {
 public:
  RewriteContext(const ir::Graph& original, ir::Builder& builder, std::vector<ir::TensorId>& map)
      : original_(original), builder_(builder), map_(map) {}
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

 private:
  const ir::Graph& original_;
  ir::Builder& builder_;
  std::vector<ir::TensorId>& map_;
  std::vector<ir::TensorId> local_{};
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
  // Off unless the policy enables it (policy.h `enable`): a pattern whose
  // result rounds where the primitive graph does not (a fast form's BF16
  // and MXFP8 activations), which a model's policy chooses.
  bool opt_in = false;
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
struct Applied {
  std::string_view pattern{};
  std::vector<ir::NodeId> replaced{};
  std::vector<ir::NodeId> made{};
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
};

// The enabled rewrite patterns applied over `graph` (see the header), for
// a step of `shape` where it is given.
std::expected<Rewritten, ir::VerifyError> ApplyPatterns(const ir::Graph& graph,
                                                        const SelectionView& view,
                                                        const StepShape* shape = nullptr);

// The batch a batch pattern makes at `anchor` of `graph`, where it matches
// and is enabled.
std::optional<Match> BatchAt(const Pattern& pattern, const MatchContext& context,
                             ir::NodeId anchor);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_PATTERN_H_
