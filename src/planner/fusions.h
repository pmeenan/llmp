// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The pattern table's fusions of hyper-connections, Gated DeltaNet, QSA,
// routed experts and product inputs (slice G3b; pattern.h lists them in
// the table): each a match function over a step's graph and its rewrite.
// Their results are op set v1's fused kinds (ir/op.h kHcPrep to
// kMatMulIdGlu), defined by their decompositions, or primitives in another
// form (a shared input preparation). Private to the planner: pattern.cc's
// table names them.
//
// They match by dataflow, as every pattern does; each checks what its own
// rewrite needs (Pattern::own_checks): its intermediates read only inside
// the match, its leaves made where its replacement reads them, and no read
// moved past a write of the same root.

#ifndef LLMP_PLANNER_FUSIONS_H_
#define LLMP_PLANNER_FUSIONS_H_

#include <cstddef>
#include <optional>

#include "ir/graph.h"
#include "planner/pattern.h"

namespace llmp::planner::fusions {

struct GroupedProduct {
  ir::NodeId product{}, input_permute{}, output_permute{}, assembly{};
  ir::TensorId leaf{}, weight{}, input{};
};

std::optional<GroupedProduct> GroupedProductOf(const ir::Graph& graph, ir::NodeId product);

// Round 0 (over the step's primitive graph).
std::optional<Match> GroupedVecQMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> RoundedIndexMaskMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> StateBankAttentionMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> RmsNormRopeTailMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> InverseRopeGroupedMatMulMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> SoftmaxPoolCompressMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> SinkhornHcPreMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> HcPostFlatNormMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> HcPrepMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> HcMixMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> HcCombineMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> GdnConvMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> GdnHistoryMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> GdnGatesMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> GdnStepMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> GdnNormGateMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> InterleavedNormRopeMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> MeanPoolNormRopeStoreMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> BlockSparseAttentionMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> PreparedHeadGateMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> MoeRouterMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> MoeExpertsMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> MoeGluMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> MoeCombineMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> VecQGluMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> Mxfp8InputsMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> TrellisProductsMatch(const MatchContext& c, ir::NodeId anchor);
// Round 1 (over round 0's result).
std::optional<Match> HcFoldInputsMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> Bf16InputsMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> Bf16RowsMatch(const MatchContext& c, ir::NodeId anchor);
std::optional<Match> VecQInputsMatch(const MatchContext& c, ir::NodeId anchor);

// The rewrite every one of them takes: the replacement each match planned
// for its nodes' places (Emissions, fusions.cc).
void Rewrite(RewriteContext& c, const Match& m, std::size_t index);

}  // namespace llmp::planner::fusions

#endif  // LLMP_PLANNER_FUSIONS_H_
