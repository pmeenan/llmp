// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/builder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "ir/infer.h"
#include "ir/op.h"
#include "ir/verify.h"

namespace llmp::ir {
namespace {

template <class T>
std::size_t GrowthCapacity(const std::vector<T>& values, std::size_t required) {
  if (required <= values.capacity()) return values.capacity();
  const std::size_t doubled = values.capacity() > std::numeric_limits<std::size_t>::max() / 2
                                  ? std::numeric_limits<std::size_t>::max()
                                  : values.capacity() * 2;
  return std::max(required, std::max<std::size_t>(4, doubled));
}

bool IsLeafOp(Op op) { return op == Op::kInput || op == Op::kWeight || op == Op::kState; }

bool AddBytes(std::uint64_t& total, std::uint64_t count, std::uint64_t size) {
  if (size != 0 && count > (std::numeric_limits<std::uint64_t>::max() - total) / size) return false;
  total += count * size;
  return true;
}

template <typename P>
ParamBytes B(const P& p) {
  return ToBytes(p);
}

}  // namespace

Builder::Builder(std::size_t max_nodes) : max_nodes_(std::min(max_nodes, Graph::kMaxNodes)) {}

void Builder::Refuse(GraphError error, std::string_view detail) {
  if (!error_) {
    if (diagnostic_ != nullptr) {
      RefuseDiagnostic(error, "{}", std::array{DiagnosticArg(detail)});
      return;
    }
    error_ = VerifyError{.error = error,
                         .node = static_cast<std::uint32_t>(graph_.nodes_.size()),
                         .detail = std::string(detail)};
  }
}

void Builder::RefuseAdmission() {
  if (!error_) {
    error_ = VerifyError{.error = GraphError::kAdmission,
                         .node = static_cast<std::uint32_t>(graph_.nodes_.size()),
                         .detail = {}};
  }
}

void Builder::RefuseDiagnostic(GraphError error, std::string_view format,
                               std::span<const DiagnosticValue> values) {
  if (error_) return;
  auto detail = MakeDiagnostic(diagnostic_, format, values);
  if (!detail) {
    if (diagnostic_ != nullptr) {
      RefuseAdmission();
    } else {
      Refuse(GraphError::kAdmission,
             "comparison builder storage funder must be installed on an empty builder");
    }
    return;
  }
  error_ = VerifyError{.error = error,
                       .node = static_cast<std::uint32_t>(graph_.nodes_.size()),
                       .detail = std::move(*detail)};
}

bool Builder::SetStorageFunder(std::shared_ptr<const StorageFunder> funder) {
  if (!ok() || !funder || funder_ || graph_.nodes_.size() != 0 || graph_.tensors_.size() != 0 ||
      !graph_.inputs_.empty() || !graph_.outputs_.empty() || !graph_.keep_.empty() ||
      !graph_.names_.empty() || graph_.nodes_.capacity() != 0 || graph_.tensors_.capacity() != 0 ||
      graph_.producers_.capacity() != 0 || graph_.inputs_.capacity() != 0 ||
      graph_.outputs_.capacity() != 0 || graph_.keep_.capacity() != 0 ||
      graph_.names_.capacity() != 0) {
    if (diagnostic_ != nullptr) {
      RefuseAdmission();
    } else {
      Refuse(GraphError::kAdmission,
             "comparison builder storage funder must be installed on an empty builder");
    }
    return false;
  }
  funder_ = std::move(funder);
  graph_.storage_loan_ = StorageLoan(funder_, 0);
  return true;
}

std::optional<StorageLoan> Builder::FundScratch(std::uint64_t bytes) {
  if (!ok() || !funder_) return std::nullopt;
  auto loan = AcquireStorageLoan(funder_, bytes);
  if (!loan) RefuseAdmission();
  return loan;
}

bool Builder::Admit(std::size_t nodes, std::size_t tensors, std::size_t producers,
                    std::size_t inputs, std::size_t outputs, std::size_t keeps, std::size_t names,
                    std::uint64_t copied_name_bytes) {
  if (!ok()) return false;
  if (!funder_) return true;
  BuilderStorageRequest request{.node_capacity = GrowthCapacity(graph_.nodes_, nodes),
                                .tensor_capacity = GrowthCapacity(graph_.tensors_, tensors),
                                .producer_capacity = GrowthCapacity(graph_.producers_, producers),
                                .input_capacity = GrowthCapacity(graph_.inputs_, inputs),
                                .output_capacity = GrowthCapacity(graph_.outputs_, outputs),
                                .keep_capacity = GrowthCapacity(graph_.keep_, keeps),
                                .name_capacity = GrowthCapacity(graph_.names_, names),
                                .owned_name_count = names,
                                .copied_name_bytes = copied_name_bytes};
  const bool unchanged = request.node_capacity == graph_.nodes_.capacity() &&
                         request.tensor_capacity == graph_.tensors_.capacity() &&
                         request.producer_capacity == graph_.producers_.capacity() &&
                         request.input_capacity == graph_.inputs_.capacity() &&
                         request.output_capacity == graph_.outputs_.capacity() &&
                         request.keep_capacity == graph_.keep_.capacity() &&
                         request.name_capacity == graph_.names_.capacity() &&
                         request.copied_name_bytes == copied_name_bytes_;
  if (unchanged) return true;
  std::uint64_t bytes = 0;
  constexpr std::uint64_t kVectorAllocationOverhead = 2 * sizeof(void*) + alignof(std::max_align_t);
  if (!AddBytes(bytes, request.node_capacity, sizeof(Node)) ||
      !AddBytes(bytes, request.tensor_capacity, sizeof(TensorDesc)) ||
      !AddBytes(bytes, request.producer_capacity, sizeof(NodeId)) ||
      !AddBytes(bytes, request.input_capacity, sizeof(TensorId)) ||
      !AddBytes(bytes, request.output_capacity, sizeof(TensorId)) ||
      !AddBytes(bytes, request.keep_capacity, sizeof(TensorId)) ||
      !AddBytes(bytes, request.name_capacity, sizeof(std::string)) ||
      !AddBytes(bytes, request.node_capacity != 0, kVectorAllocationOverhead) ||
      !AddBytes(bytes, request.tensor_capacity != 0, kVectorAllocationOverhead) ||
      !AddBytes(bytes, request.producer_capacity != 0, kVectorAllocationOverhead) ||
      !AddBytes(bytes, request.input_capacity != 0, kVectorAllocationOverhead) ||
      !AddBytes(bytes, request.output_capacity != 0, kVectorAllocationOverhead) ||
      !AddBytes(bytes, request.keep_capacity != 0, kVectorAllocationOverhead) ||
      !AddBytes(bytes, request.name_capacity != 0, kVectorAllocationOverhead) ||
      !AddBytes(bytes, request.copied_name_bytes, 1) ||
      !AddBytes(bytes, request.owned_name_count, kVectorAllocationOverhead)) {
    RefuseAdmission();
    return false;
  }
  auto loan = AcquireStorageLoan(funder_, bytes);
  if (!loan) {
    RefuseAdmission();
    return false;
  }
  graph_.nodes_.reserve(request.node_capacity);
  graph_.tensors_.reserve(request.tensor_capacity);
  graph_.producers_.reserve(request.producer_capacity);
  graph_.inputs_.reserve(request.input_capacity);
  graph_.outputs_.reserve(request.output_capacity);
  graph_.keep_.reserve(request.keep_capacity);
  graph_.names_.reserve(request.name_capacity);
  graph_.storage_loan_ = std::move(*loan);
  copied_name_bytes_ = copied_name_bytes;
  return true;
}

Outputs Builder::Emit(OpKey key, const ParamBytes& params, std::span<const TensorId> sources,
                      std::span<const TensorDesc> requested, std::string_view name) {
  if (!ok()) {
    return {};
  }
  if (graph_.nodes_.size() >= max_nodes_ ||
      graph_.tensors_.size() + kMaxOutputs > Graph::kMaxTensors) {
    RefuseDiagnostic(GraphError::kBounds,
                     DiagFormat("past {} nodes or {} tensors", max_nodes_, Graph::kMaxTensors));
    return {};
  }
  if (sources.size() > kMaxSources) {
    RefuseDiagnostic(GraphError::kArity,
                     DiagFormat("{} sources, past {}", sources.size(), kMaxSources));
    return {};
  }
  for (const TensorId s : sources) {
    if (s.valid() && !Known(s)) {
      RefuseDiagnostic(GraphError::kCycle,
                       DiagFormat("{}'s source {} was not made here", key, s.v));
      return {};
    }
  }
  auto inferred = Infer(key, params, sources, graph_.tensors_, requested, funder_, diagnostic_);
  if (!inferred) {
    VerifyError e = std::move(inferred.error());
    e.node = static_cast<std::uint32_t>(graph_.nodes_.size());
    error_ = std::move(e);
    return {};
  }
  Node node;
  node.key = key;
  node.params = params;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    node.src.at(i) = sources[i];
  }
  node.nsrc = static_cast<std::uint8_t>(sources.size());
  node.component = component_;
  node.batch = batch_;
  const NodeId id{static_cast<std::uint32_t>(graph_.nodes_.size())};
  Outputs outs;
  const std::size_t input_add = key.op == Op::kInput ? 1 : 0;
  std::uint64_t copied = copied_name_bytes_;
  if (!name.empty()) {
    if (name.size() == std::numeric_limits<std::uint64_t>::max() ||
        name.size() + 1 > std::numeric_limits<std::uint64_t>::max() - copied) {
      RefuseAdmission();
      return {};
    }
    copied += name.size() + 1;  // includes each owned string's trailing NUL
  }
  if (!Admit(graph_.nodes_.size() + 1, graph_.tensors_.size() + inferred->nout,
             graph_.producers_.size() + inferred->nout, graph_.inputs_.size() + input_add,
             graph_.outputs_.size(), graph_.keep_.size(),
             graph_.names_.size() + static_cast<std::size_t>(!name.empty()), copied))
    return {};
  for (std::size_t i = 0; i < inferred->nout; ++i) {
    TensorDesc t = inferred->out.at(i);
    const TensorDesc* r = i < requested.size() ? &requested[i] : nullptr;
    if (r != nullptr && inferred->float_result.at(i) && key.op == Op::kFused &&
        PreparedRows(r->format)) {
      t.format.encoding = r->format.encoding;
      t.format.layout = r->format.layout;
      if (!format::Validate(t.format)) {
        RefuseDiagnostic(GraphError::kFormat,
                         DiagFormat("{}'s output {} is not valid as prepared rows", key, i));
        return {};
      }
      t.nb = NaturalStrides(t.format);
    } else if (r != nullptr && inferred->float_result.at(i) &&
               r->format.encoding != t.format.encoding) {
      if (!ResultEncoding(r->format.encoding) || r->format.layout != format::Layout::kDense) {
        RefuseDiagnostic(GraphError::kFormat,
                         DiagFormat("{}'s output {} is recorded as F32, F16 or BF16", key, i));
        return {};
      }
      t.format.encoding = r->format.encoding;
      t.nb = NaturalStrides(t.format);
    }
    if (!IsLeafOp(key.op) && !t.view_of.valid()) {
      t.role = Role::kActivation;
      t.segment = segment_;
    }
    if (i == 0 && !name.empty()) {
      t.name = static_cast<std::uint32_t>(graph_.names_.size());
      graph_.names_.emplace_back(name);
    } else {
      t.name = kNoName;
    }
    const TensorId tid{static_cast<std::uint32_t>(graph_.tensors_.size())};
    graph_.tensors_.push_back(t);
    graph_.producers_.push_back(id);
    node.out.at(i) = tid;
    outs.ids.at(i) = tid;
  }
  node.nout = inferred->nout;
  // Leaves compute nothing: joined. A node whose outputs are all views (a
  // view, a store) is its outputs' owner's, or joined.
  if (IsLeafOp(key.op)) {
    node.batch = Batch::kJoined;
  } else if (node.nout > 0) {
    bool views = true;
    bool owned = true;
    for (std::size_t i = 0; i < node.nout; ++i) {
      const TensorDesc& t = graph_.tensors_.at(node.out.at(i).v);
      views = views && t.view_of.valid();
      owned = owned && t.segment != kJoinedSegment;
    }
    if (views) {
      node.batch = owned ? Batch::kPerOwner : Batch::kJoined;
    }
  }
  outs.n = inferred->nout;
  graph_.nodes_.push_back(node);
  if (key.op == Op::kInput) {
    graph_.inputs_.push_back(outs[0]);
  }
  return outs;
}

TensorId Builder::One(OpKey key, const ParamBytes& params, std::span<const TensorId> sources,
                      std::span<const TensorDesc> requested) {
  return Emit(key, params, sources, requested)[0];
}

TensorId Builder::Leaf(Op op, Role role, const format::TensorFormat& format, Binding binding,
                       std::uint8_t segment, std::string_view name, const ParamBytes& params) {
  TensorDesc r;
  r.role = role;
  r.format = format;
  r.format.shape =
      ShapeOf({format.shape.ne[0], format.shape.ne[1], format.shape.ne[2], format.shape.ne[3]});
  r.binding = binding;
  r.segment = segment;
  return Emit(Primitive(op), params, {}, std::span(&r, 1), name)[0];
}

TensorId Builder::Input(const format::TensorFormat& format, std::string_view name,
                        const InputParams& fact) {
  ParamBytes params{};
  std::memcpy(params.data(), &fact, sizeof(fact));
  return Leaf(Op::kInput, Role::kInput, format, {}, segment_, name, params);
}

TensorId Builder::Constant(const format::TensorFormat& format, std::string_view name,
                           std::span<const std::byte> content) {
  ParamBytes params{};
  if (!content.empty()) {
    base::Sha256 hash;
    hash.Update(content);
    const base::Sha256Digest digest = hash.Finish();
    static_assert(sizeof(digest) <= kConstantDigestBytes);
    std::memcpy(params.data(), digest.data(), sizeof(digest));
  }
  return Leaf(Op::kInput, Role::kConstant, format, {}, segment_, name, params);
}

TensorId Builder::Weight(const format::TensorFormat& format, const WeightRef& ref,
                         std::string_view name) {
  return Leaf(Op::kWeight, Role::kWeight, format, ref, kJoinedSegment, name);
}

TensorId Builder::State(const format::TensorFormat& format, const StateRef& ref,
                        std::string_view name) {
  return Leaf(Op::kState, Role::kState, format, ref, ref.segment, name);
}

TensorId Builder::View(TensorId a, const format::Shape& shape,
                       const std::array<std::uint64_t, format::kMaxDims>& nb,
                       std::uint64_t offset) {
  if (!ok() || !Known(a)) {
    Refuse(GraphError::kArity, "a view of an unknown tensor");
    return {};
  }
  TensorDesc r;
  r.format.shape = ShapeOf({shape.ne[0], shape.ne[1], shape.ne[2], shape.ne[3]});
  r.nb = nb;
  r.view_offset = tensor(a).view_offset + offset;
  return One(Primitive(Op::kView), ParamBytes{}, std::array{a}, std::span(&r, 1));
}

TensorId Builder::Reshape(TensorId a, const format::Shape& shape) {
  TensorDesc r;
  r.format.shape = ShapeOf({shape.ne[0], shape.ne[1], shape.ne[2], shape.ne[3]});
  return One(Primitive(Op::kReshape), ParamBytes{}, std::array{a}, std::span(&r, 1));
}

TensorId Builder::Permute(TensorId a, std::array<std::uint8_t, 4> axes) {
  return One(Primitive(Op::kPermute), B(PermuteParams{.axes = axes}), std::array{a});
}

TensorId Builder::Transpose(TensorId a) { return Permute(a, {1, 0, 2, 3}); }

TensorId Builder::Cont(TensorId a) {
  return One(Primitive(Op::kCont), ParamBytes{}, std::array{a});
}

TensorId Builder::Cont(TensorId a, const format::Shape& shape) {
  TensorDesc r;
  r.format.shape = ShapeOf({shape.ne[0], shape.ne[1], shape.ne[2], shape.ne[3]});
  return One(Primitive(Op::kCont), ParamBytes{}, std::array{a}, std::span(&r, 1));
}

TensorId Builder::Concat(TensorId a, TensorId b, std::uint8_t dim) {
  return One(Primitive(Op::kConcat), B(ConcatParams{.dim = dim}), std::array{a, b});
}

TensorId Builder::Repeat(TensorId a, const format::Shape& shape) {
  TensorDesc r;
  r.format.shape = shape;
  return One(Primitive(Op::kRepeat), ParamBytes{}, std::array{a}, std::span(&r, 1));
}

TensorId Builder::Convert(TensorId a, format::Encoding encoding) {
  TensorDesc r;
  r.format.encoding = encoding;
  r.format.layout = format::Layout::kDense;
  return One(Primitive(Op::kConvert), ParamBytes{}, std::array{a}, std::span(&r, 1));
}

TensorId Builder::GetRows(TensorId a, TensorId ids) {
  return One(Primitive(Op::kGetRows), ParamBytes{}, std::array{a, ids});
}

TensorId Builder::MatMul(TensorId a, TensorId b, const ProductParams& params) {
  return One(Primitive(Op::kMatMul), B(params), std::array{a, b});
}

TensorId Builder::MatMulId(TensorId as, TensorId b, TensorId ids, const ProductParams& params) {
  return One(Primitive(Op::kMatMulId), B(params), std::array{as, b, ids});
}

TensorId Builder::Prepare(TensorId a, format::Encoding encoding, format::Layout layout) {
  TensorDesc r;
  r.format.encoding = encoding;
  r.format.layout = layout;
  return One(Primitive(Op::kPrepare), ParamBytes{}, std::array{a}, std::span(&r, 1));
}

Outputs Builder::ExpertSort(TensorId ids, std::uint32_t experts) {
  return Emit(Primitive(Op::kExpertSort), B(ExpertSortParams{.experts = experts}), std::array{ids});
}

TensorId Builder::Attention(TensorId q, TensorId k, TensorId v, TensorId mask, TensorId sinks,
                            const AttnParams& params) {
  if (sinks.valid()) {
    return One(Primitive(Op::kAttention), B(params), std::array{q, k, v, mask, sinks});
  }
  if (mask.valid()) {
    return One(Primitive(Op::kAttention), B(params), std::array{q, k, v, mask});
  }
  return One(Primitive(Op::kAttention), B(params), std::array{q, k, v});
}

TensorId Builder::MaskBuild(TensorId q_pos, TensorId cell_pos, std::uint64_t rows,
                            const MaskParams& params, format::Encoding encoding) {
  if (!ok() || !Known(cell_pos)) {
    Refuse(GraphError::kArity, "a mask over unknown positions");
    return {};
  }
  TensorDesc r;
  r.format = Dense(encoding, ShapeOf({tensor(cell_pos).ne(0), rows}));
  return One(Primitive(Op::kMaskBuild), B(params), std::array{q_pos, cell_pos}, std::span(&r, 1));
}

TensorId Builder::CacheStore(TensorId cache, TensorId src, TensorId idx) {
  return One(Primitive(Op::kCacheStore), ParamBytes{}, std::array{cache, src, idx});
}

TensorId Builder::IndexScore(TensorId q, TensorId k, TensorId w, TensorId mask) {
  if (mask.valid()) {
    return One(Primitive(Op::kIndexScore), ParamBytes{}, std::array{q, k, w, mask});
  }
  return One(Primitive(Op::kIndexScore), ParamBytes{}, std::array{q, k, w});
}

TensorId Builder::RoundedIndexScore(TensorId q, TensorId k, TensorId w, TensorId counts,
                                    const RoundedIndexScoreParams& params) {
  return One(Primitive(Op::kRoundedIndexScore), B(params), std::array{q, k, w, counts});
}

TensorId Builder::PrefixOverwrite(TensorId values, TensorId counts,
                                  const PrefixOverwriteParams& params) {
  return One(Primitive(Op::kPrefixOverwrite), B(params), std::array{values, counts});
}

TensorId Builder::Pool(TensorId x, TensorId scores, const PoolParams& params) {
  if (scores.valid()) {
    return One(Primitive(Op::kPool), B(params), std::array{x, scores});
  }
  return One(Primitive(Op::kPool), B(params), std::array{x});
}

TensorId Builder::RmsNorm(TensorId x, float eps) {
  return One(Primitive(Op::kRmsNorm), B(NormParams{.eps = eps}), std::array{x});
}

TensorId Builder::L2Norm(TensorId x, float eps) {
  return One(Primitive(Op::kL2Norm), B(NormParams{.eps = eps}), std::array{x});
}

TensorId Builder::LayerNorm(TensorId x, float eps) {
  return One(Primitive(Op::kLayerNorm), B(NormParams{.eps = eps}), std::array{x});
}

TensorId Builder::ChannelNorm(TensorId x, float eps) {
  return One(Primitive(Op::kChannelNorm), B(NormParams{.eps = eps}), std::array{x});
}

TensorId Builder::Rope(TensorId x, TensorId pos, TensorId freq, const RopeParams& params) {
  if (freq.valid()) {
    return One(Primitive(Op::kRope), B(params), std::array{x, pos, freq});
  }
  return One(Primitive(Op::kRope), B(params), std::array{x, pos});
}

TensorId Builder::Add(TensorId a, TensorId b) {
  return One(Primitive(Op::kAdd), ParamBytes{}, std::array{a, b});
}

TensorId Builder::Sub(TensorId a, TensorId b) {
  return One(Primitive(Op::kSub), ParamBytes{}, std::array{a, b});
}

TensorId Builder::Mul(TensorId a, TensorId b) {
  return One(Primitive(Op::kMul), ParamBytes{}, std::array{a, b});
}

TensorId Builder::Select(TensorId condition, TensorId yes, TensorId no) {
  return One(Primitive(Op::kSelect), ParamBytes{}, std::array{condition, yes, no});
}

TensorId Builder::Div(TensorId a, TensorId b) {
  return One(Primitive(Op::kDiv), ParamBytes{}, std::array{a, b});
}

TensorId Builder::Scale(TensorId x, float scale, float bias) {
  return One(Primitive(Op::kScale), B(ScaleParams{.scale = scale, .bias = bias}), std::array{x});
}

TensorId Builder::Unary(TensorId x, UnaryFn fn) {
  return One(Primitive(Op::kUnary), B(UnaryParams{.fn = fn}), std::array{x});
}

TensorId Builder::Clamp(TensorId x, float min, float max) {
  return One(Primitive(Op::kClamp), B(ClampParams{.min = min, .max = max}), std::array{x});
}

TensorId Builder::Fill(const format::Shape& shape, float value, format::Encoding encoding) {
  TensorDesc r;
  r.format = Dense(encoding, shape);
  return One(Primitive(Op::kFill), B(FillParams{.value = value}), {}, std::span(&r, 1));
}

TensorId Builder::Glu(TensorId gate, TensorId up, const GluParams& params) {
  if (up.valid()) {
    return One(Primitive(Op::kGlu), B(params), std::array{gate, up});
  }
  return One(Primitive(Op::kGlu), B(params), std::array{gate});
}

TensorId Builder::Softcap(TensorId x, float cap) {
  return One(Primitive(Op::kSoftcap), B(SoftcapParams{.cap = cap}), std::array{x});
}

TensorId Builder::SumRows(TensorId x) {
  return One(Primitive(Op::kSumRows), ParamBytes{}, std::array{x});
}

TensorId Builder::SoftMax(TensorId x, TensorId mask, TensorId sinks, const SoftMaxParams& params) {
  if (sinks.valid()) {
    return One(Primitive(Op::kSoftMax), B(params), std::array{x, mask, sinks});
  }
  if (mask.valid()) {
    return One(Primitive(Op::kSoftMax), B(params), std::array{x, mask});
  }
  return One(Primitive(Op::kSoftMax), B(params), std::array{x});
}

TensorId Builder::Argmax(TensorId x, ArgmaxNan nan) {
  return One(Primitive(Op::kArgmax), B(ArgmaxParams{.nan = nan}), std::array{x});
}

TensorId Builder::Argsort(TensorId x, bool descending) {
  return One(Primitive(Op::kArgsort), B(ArgsortParams{.descending = descending}), std::array{x});
}

TensorId Builder::TopK(TensorId x, std::uint32_t k) {
  return One(Primitive(Op::kTopK), B(TopKParams{.k = k}), std::array{x});
}
TensorId Builder::VisibleTopK(TensorId scores, TensorId counts, std::uint32_t k) {
  return One(Primitive(Op::kVisibleTopK), B(TopKParams{.k = k}), std::array{scores, counts});
}
TensorId Builder::SelectedMask(TensorId raw, TensorId ids, const SelectedMaskParams& params) {
  return One(Primitive(Op::kSelectedMask), B(params), std::array{raw, ids});
}

Outputs Builder::Route(TensorId logits, TensorId bias, TensorId table, TensorId tokens,
                       const RouteParams& params) {
  std::array<TensorId, 4> src{logits, bias, table, tokens};
  std::size_t n = 4;
  while (n > 1 && !src.at(n - 1).valid()) {
    --n;
  }
  return Emit(Primitive(Op::kRoute), B(params), std::span(src).first(n));
}

TensorId Builder::ExpertReduce(TensorId x, TensorId w, TensorId ids, TensorId scale) {
  std::array<TensorId, 4> src{x, w, ids, scale};
  std::size_t n = 4;
  while (n > 2 && !src.at(n - 1).valid()) {
    --n;
  }
  return One(Primitive(Op::kExpertReduce), ParamBytes{}, std::span(src).first(n));
}

TensorId Builder::SsmConv(TensorId sx, TensorId c) {
  return One(Primitive(Op::kSsmConv), ParamBytes{}, std::array{sx, c});
}

Outputs Builder::GatedDeltaNet(TensorId q, TensorId k, TensorId v, TensorId g, TensorId beta,
                               TensorId state) {
  return Emit(Primitive(Op::kGatedDeltaNet), ParamBytes{}, std::array{q, k, v, g, beta, state});
}

TensorId Builder::Conv2d(TensorId kernel, TensorId x, TensorId bias, const Conv2dParams& params) {
  if (bias.valid()) {
    return One(Primitive(Op::kConv2d), B(params), std::array{kernel, x, bias});
  }
  return One(Primitive(Op::kConv2d), B(params), std::array{kernel, x});
}

TensorId Builder::Upsample(TensorId x, std::uint32_t factor) {
  return One(Primitive(Op::kUpsample), B(UpsampleParams{.factor = factor}), std::array{x});
}

TensorId Builder::HcPre(TensorId x, TensorId w, const HcParams& params) {
  return One(Primitive(Op::kHcPre), B(params), std::array{x, w});
}

TensorId Builder::HcPost(TensorId y, TensorId x, TensorId post, TensorId comb,
                         const HcParams& params) {
  if (comb.valid()) {
    return One(Primitive(Op::kHcPost), B(params), std::array{y, x, post, comb});
  }
  return One(Primitive(Op::kHcPost), B(params), std::array{y, x, post});
}

TensorId Builder::HcMix(TensorId mixes, TensorId scale, TensorId base, const HcParams& params) {
  return One(Primitive(Op::kHcMix), B(params), std::array{mixes, scale, base});
}

TensorId Builder::Sample(TensorId logits, TensorId keys, const SampleParams& params) {
  if (keys.valid()) {
    return One(Primitive(Op::kSample), B(params), std::array{logits, keys});
  }
  return One(Primitive(Op::kSample), B(params), std::array{logits});
}

TensorId Builder::AllReduce(TensorId x, const CollectiveParams& params) {
  return One(Primitive(Op::kAllReduce), B(params), std::array{x});
}

TensorId Builder::AllGather(TensorId x, const format::Shape& shape,
                            const CollectiveParams& params) {
  TensorDesc r;
  r.format.shape = shape;
  return One(Primitive(Op::kAllGather), B(params), std::array{x}, std::span(&r, 1));
}

TensorId Builder::AllToAll(TensorId x, const CollectiveParams& params) {
  return One(Primitive(Op::kAllToAll), B(params), std::array{x});
}

void Builder::Send(TensorId x, const CollectiveParams& params) {
  Emit(Primitive(Op::kSend), B(params), std::array{x});
}

TensorId Builder::Recv(const format::TensorFormat& format, const CollectiveParams& params) {
  TensorDesc r;
  r.format = format;
  return One(Primitive(Op::kRecv), B(params), {}, std::span(&r, 1));
}

Outputs Builder::FusedBytes(FusedKind kind, std::span<const TensorId> sources,
                            const ParamBytes& params, std::span<const format::Encoding> encodings) {
  std::array<TensorDesc, kMaxOutputs> req{};
  const std::size_t n = std::min(encodings.size(), kMaxOutputs);
  for (std::size_t i = 0; i < n; ++i) {
    req.at(i).format.encoding = encodings[i];
  }
  return Emit(ir::Fused(kind), params, sources, std::span(req).first(n));
}

void Builder::Output(TensorId id) {
  if (!ok()) {
    return;
  }
  if (!Known(id)) {
    Refuse(GraphError::kArity, "an unknown output");
    return;
  }
  if (!Admit(graph_.nodes_.size(), graph_.tensors_.size(), graph_.producers_.size(),
             graph_.inputs_.size(), graph_.outputs_.size() + 1, graph_.keep_.size(),
             graph_.names_.size(), copied_name_bytes_))
    return;
  TensorDesc& t = graph_.tensors_.at(id.v);
  if (t.role == Role::kActivation) {
    t.role = Role::kOutput;
  }
  graph_.outputs_.push_back(id);
}

void Builder::Keep(TensorId id) {
  if (!ok()) {
    return;
  }
  if (!Known(id)) {
    Refuse(GraphError::kArity, "an unknown kept tensor");
    return;
  }
  if (!Admit(graph_.nodes_.size(), graph_.tensors_.size(), graph_.producers_.size(),
             graph_.inputs_.size(), graph_.outputs_.size(), graph_.keep_.size() + 1,
             graph_.names_.size(), copied_name_bytes_))
    return;
  graph_.keep_.push_back(id);
}

std::expected<Graph, VerifyError> Builder::Finish() && {
  if (error_) {
    return std::unexpected(std::move(*error_));
  }
  Graph g = std::move(graph_);
  if (auto v = Verify(g, funder_, diagnostic_); !v) {
    return std::unexpected(std::move(v.error()));
  }
  StorageLoan digest_loan;
  if (funder_) {
    auto bound = g.DigestScratchBound();
    if (!bound) return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
    auto loan = AcquireStorageLoan(funder_, *bound);
    if (!loan) return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
    digest_loan = std::move(*loan);
  }
  g.digest_ = g.ComputeDigest();
  g.identity_ = Graph::NextIdentity();
  return g;
}

}  // namespace llmp::ir
