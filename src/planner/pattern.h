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
//                      residual add of its shape (fuse_norm_add)
//   mul_mat_glu        fused.mul_mat_glu: gate and up products of one input
//                      and the GLU over them (fuse_quant_glu)
//   rope_store         fused.rope_store: a rotation stored whole as cache
//                      rows (fuse_rope_store)
//   rms_norm_mul       fused.rms_norm_mul (fuse_norms)
//   prepared_inputs    two or more products of one F32 input share one
//                      Q8_1 preparation (kPrepare), each product reading it
//                      (the Gemma builders' shared_q8, shared_q8.h)
//   grouped_stores     batch: consecutive cache stores in one launch, a
//                      capability of fused.grouped_stores with a batch
//                      (group_set_rows, set_rows_group.h)

#ifndef LLMP_PLANNER_PATTERN_H_
#define LLMP_PLANNER_PATTERN_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
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

namespace llmp::planner {

// A graph's readers, as patterns ask about them.
class MatchContext {
 public:
  MatchContext(const ir::Graph& graph, const SelectionView& view);

  const ir::Graph& graph() const { return graph_; }
  const SelectionView& view() const { return view_; }
  const ir::Node& node(ir::NodeId id) const { return graph_.node(id); }
  const ir::TensorDesc& tensor(ir::TensorId id) const { return graph_.tensor(id); }
  const ir::Node& producer(ir::TensorId id) const { return graph_.node(graph_.producer(id)); }
  // The nodes reading `t`, an entry per source position, in node order.
  std::span<const ir::NodeId> readers(ir::TensorId t) const;
  // Whether the step returns or keeps `t` (or a view of it).
  bool external(ir::TensorId t) const;
  // Whether a match already took the node.
  bool taken(ir::NodeId id) const { return taken_.at(id.v); }
  void Take(std::span<const ir::NodeId> nodes);
  // The node reading `t` when exactly one source of one node does and the
  // step does not return or keep it.
  std::optional<ir::NodeId> OnlyReader(ir::TensorId t) const;

 private:
  const ir::Graph& graph_;
  const SelectionView& view_;
  std::vector<std::vector<ir::NodeId>> readers_;  // by tensor
  std::vector<bool> external_;                    // by root tensor
  std::vector<bool> taken_;                       // by node
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
  std::array<ir::TensorId, 4>& local() { return local_; }
  void Reset() { local_ = {}; }

 private:
  const ir::Graph& original_;
  ir::Builder& builder_;
  std::vector<ir::TensorId>& map_;
  std::array<ir::TensorId, 4> local_{};
};

enum class PatternKind : std::uint8_t { kRewrite, kBatch };

struct Pattern {
  std::string_view name{};
  std::uint16_t version = 1;
  PatternKind kind = PatternKind::kRewrite;
  // A rewrite's result (the fused kind it makes, or the primitive it
  // adds); a batch's capability key.
  ir::OpKey result{};
  std::span<const ir::Op> anchors{};
  std::optional<Match> (*match)(const MatchContext& context, ir::NodeId anchor) = nullptr;
  // A rewrite's: what replaces the match's node `index` (each of its nodes
  // in turn, in node order), mapping the original outputs it stands for.
  void (*rewrite)(RewriteContext& context, const Match& match, std::size_t index) = nullptr;
};

// The table, in its fixed order.
std::span<const Pattern> Patterns();
const Pattern* FindPattern(std::string_view name);

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

// The enabled rewrite patterns applied over `graph` (see the header).
std::expected<Rewritten, ir::VerifyError> ApplyPatterns(const ir::Graph& graph,
                                                        const SelectionView& view);

// The batch a batch pattern makes at `anchor` of `graph`, where it matches
// and is enabled.
std::optional<Match> BatchAt(const Pattern& pattern, const MatchContext& context,
                             ir::NodeId anchor);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_PATTERN_H_
