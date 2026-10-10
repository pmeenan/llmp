// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Building a graph (graph.h): typed helpers for every op of op set v1 and
// every fused kind, each making its outputs from its sources by op.h's
// rules (infer.h). Components emit their fragments through it, and
// rewrites (decompose.h, the planner's patterns) copy and replace nodes
// through Emit.
//
// A builder never aborts on a malformed request: the first error is kept,
// every later call returns invalid ids, and Finish() reports it. Finish()
// verifies the whole graph (verify.h) before handing it over.
//
// Context applies to the nodes made after it is set: the component index
// (SetComponent), and whether nodes are joined or one owner's
// (SetOwner/SetJoined), which sets their computed outputs' segment.

#ifndef LLMP_IR_BUILDER_H_
#define LLMP_IR_BUILDER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"

namespace llmp::ir {

// A node's outputs.
struct Outputs {
  std::array<TensorId, kMaxOutputs> ids{};
  std::uint8_t n = 0;
  TensorId operator[](std::size_t i) const { return i < n ? ids.at(i) : TensorId{}; }
  std::span<const TensorId> span() const { return std::span(ids).first(n); }
};

using ParamBytes = std::array<std::byte, kMaxParamBytes>;

template <typename P>
ParamBytes ToBytes(const P& p) {
  Node n;
  n.SetParams(p);
  return n.params;
}

class Builder {
 public:
  explicit Builder(std::size_t max_nodes = Graph::kMaxNodes);
  Builder(Builder&&) noexcept = default;
  Builder& operator=(Builder&&) noexcept = default;
  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;
  ~Builder() = default;

  // ---- context and state
  void SetComponent(std::uint16_t component) { component_ = component; }
  void SetOwner(std::uint8_t segment) {
    batch_ = Batch::kPerOwner;
    segment_ = segment;
  }
  void SetJoined() {
    batch_ = Batch::kJoined;
    segment_ = kJoinedSegment;
  }
  bool ok() const { return !error_.has_value(); }
  const std::optional<VerifyError>& error() const { return error_; }
  // A tensor made so far (a valid id of this builder), as a copy: the
  // builder's tensors move whenever it makes another, so it hands out no
  // reference into them (one kept across a later call would dangle).
  TensorDesc tensor(TensorId id) const { return graph_.tensor(id); }
  std::size_t node_count() const { return graph_.nodes_.size(); }
  std::size_t tensor_count() const { return graph_.tensors_.size(); }

  // ---- the generic form: a node of `key` over `sources`, `requested` as
  // infer.h's Infer takes it (a leaf's request also gives its binding,
  // segment and role; a result's request may give another float encoding).
  // `name` names the first output.
  Outputs Emit(OpKey key, const ParamBytes& params, std::span<const TensorId> sources,
               std::span<const TensorDesc> requested = {}, std::string_view name = {});

  // ---- leaves
  TensorId Input(const format::TensorFormat& format, std::string_view name = {});
  TensorId Constant(const format::TensorFormat& format, std::string_view name = {});
  TensorId Weight(const format::TensorFormat& format, const WeightRef& ref,
                  std::string_view name = {});
  // Its segment is the StateRef's.
  TensorId State(const format::TensorFormat& format, const StateRef& ref,
                 std::string_view name = {});

  // ---- structure
  // A view of a's bytes at `offset` from a's own start, with byte strides.
  TensorId View(TensorId a, const format::Shape& shape,
                const std::array<std::uint64_t, format::kMaxDims>& nb, std::uint64_t offset);
  TensorId Reshape(TensorId a, const format::Shape& shape);
  TensorId Permute(TensorId a, std::array<std::uint8_t, 4> axes);
  TensorId Transpose(TensorId a);  // ne[0] and ne[1] swapped
  TensorId Cont(TensorId a);
  TensorId Cont(TensorId a, const format::Shape& shape);
  TensorId Concat(TensorId a, TensorId b, std::uint8_t dim);
  TensorId Repeat(TensorId a, const format::Shape& shape);
  TensorId Convert(TensorId a, format::Encoding encoding);
  TensorId GetRows(TensorId a, TensorId ids);

  // ---- products
  TensorId MatMul(TensorId a, TensorId b, const ProductParams& params = {});
  TensorId MatMulId(TensorId as, TensorId b, TensorId ids, const ProductParams& params = {});
  TensorId Prepare(TensorId a, format::Encoding encoding,
                   format::Layout layout = format::Layout::kGgmlBlocks);
  Outputs ExpertSort(TensorId ids, std::uint32_t experts);  // order, offsets

  // ---- attention and masks
  TensorId Attention(TensorId q, TensorId k, TensorId v, TensorId mask, TensorId sinks,
                     const AttnParams& params);
  TensorId MaskBuild(TensorId q_pos, TensorId cell_pos, std::uint64_t rows,
                     const MaskParams& params, format::Encoding encoding = format::Encoding::kF16);
  TensorId CacheStore(TensorId cache, TensorId src, TensorId idx);
  TensorId IndexScore(TensorId q, TensorId k, TensorId w, TensorId mask);
  TensorId RoundedIndexScore(TensorId q, TensorId k, TensorId w, TensorId counts,
                             const RoundedIndexScoreParams& params = {});
  TensorId PrefixOverwrite(TensorId values, TensorId counts, const PrefixOverwriteParams& params);
  TensorId Pool(TensorId x, TensorId scores, const PoolParams& params);

  // ---- norms and rotations
  TensorId RmsNorm(TensorId x, float eps);
  TensorId L2Norm(TensorId x, float eps);
  TensorId LayerNorm(TensorId x, float eps);
  TensorId ChannelNorm(TensorId x, float eps);
  TensorId Rope(TensorId x, TensorId pos, TensorId freq, const RopeParams& params);

  // ---- elementwise
  TensorId Add(TensorId a, TensorId b);
  TensorId Sub(TensorId a, TensorId b);
  TensorId Mul(TensorId a, TensorId b);
  TensorId Div(TensorId a, TensorId b);
  TensorId Scale(TensorId x, float scale, float bias = 0.0f);
  TensorId Unary(TensorId x, UnaryFn fn);
  TensorId Clamp(TensorId x, float min, float max);
  TensorId Fill(const format::Shape& shape, float value,
                format::Encoding encoding = format::Encoding::kF32);
  TensorId Glu(TensorId gate, TensorId up,
               const GluParams& params);  // up absent: halves
  TensorId Softcap(TensorId x, float cap);

  // ---- reductions and selection
  TensorId SumRows(TensorId x);
  TensorId SoftMax(TensorId x, TensorId mask, TensorId sinks, const SoftMaxParams& params);
  TensorId Argmax(TensorId x, ArgmaxNan nan = ArgmaxNan::kIgnore);
  TensorId Argsort(TensorId x, bool descending);
  TensorId TopK(TensorId x, std::uint32_t k);
  TensorId VisibleTopK(TensorId scores, TensorId counts, std::uint32_t k);
  TensorId SelectedMask(TensorId raw, TensorId ids, const SelectedMaskParams& params);

  // ---- routing
  Outputs Route(TensorId logits, TensorId bias, TensorId table, TensorId tokens,
                const RouteParams& params);  // ids, weights
  TensorId ExpertReduce(TensorId x, TensorId w, TensorId ids, TensorId scale);

  // ---- recurrences and convolutions
  TensorId SsmConv(TensorId sx, TensorId c);
  Outputs GatedDeltaNet(TensorId q, TensorId k, TensorId v, TensorId g, TensorId beta,
                        TensorId state);  // attention, state
  TensorId Conv2d(TensorId kernel, TensorId x, TensorId bias, const Conv2dParams& params);
  TensorId Upsample(TensorId x, std::uint32_t factor);

  // ---- hyper-connections
  TensorId HcPre(TensorId x, TensorId w, const HcParams& params);
  TensorId HcPost(TensorId y, TensorId x, TensorId post, TensorId comb, const HcParams& params);
  TensorId HcMix(TensorId mixes, TensorId scale, TensorId base, const HcParams& params);

  // ---- sampling and collectives
  TensorId Sample(TensorId logits, TensorId keys, const SampleParams& params);
  TensorId AllReduce(TensorId x, const CollectiveParams& params);
  // `shape`: x's with ne[dim] times the group's members.
  TensorId AllGather(TensorId x, const format::Shape& shape, const CollectiveParams& params);
  TensorId AllToAll(TensorId x, const CollectiveParams& params);
  void Send(TensorId x, const CollectiveParams& params);
  TensorId Recv(const format::TensorFormat& format, const CollectiveParams& params);

  // ---- fused kinds (op.h lists their sources and parameters). Outputs
  // that are computed floats may be recorded in another float encoding.
  template <typename P>
  Outputs Fused(FusedKind kind, std::span<const TensorId> sources, const P& params,
                std::span<const format::Encoding> encodings = {}) {
    return FusedBytes(kind, sources, ToBytes(params), encodings);
  }
  Outputs Fused(FusedKind kind, std::span<const TensorId> sources,
                std::span<const format::Encoding> encodings = {}) {
    return FusedBytes(kind, sources, ParamBytes{}, encodings);
  }
  Outputs FusedBytes(FusedKind kind, std::span<const TensorId> sources, const ParamBytes& params,
                     std::span<const format::Encoding> encodings);

  // ---- the step's interface
  void Output(TensorId id);
  void Keep(TensorId id);

  // Refuses the graph (a decomposition's or a component's own check); the
  // first error is kept.
  void Refuse(GraphError error, std::string detail);

  // The graph, verified; the builder is spent.
  std::expected<Graph, VerifyError> Finish() &&;

 private:
  TensorId One(OpKey key, const ParamBytes& params, std::span<const TensorId> sources,
               std::span<const TensorDesc> requested = {});
  TensorId Leaf(Op op, Role role, const format::TensorFormat& format, Binding binding,
                std::uint8_t segment, std::string_view name);
  bool Known(TensorId id) const { return id.valid() && id.v < graph_.tensors_.size(); }

  Graph graph_;
  std::size_t max_nodes_;
  std::uint16_t component_ = 0;
  Batch batch_ = Batch::kJoined;
  std::uint8_t segment_ = kJoinedSegment;
  std::optional<VerifyError> error_;
};

}  // namespace llmp::ir

#endif  // LLMP_IR_BUILDER_H_
