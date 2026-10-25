// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/graph.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/op.h"
#include "ir/verify.h"

namespace llmp::ir {

std::string_view RoleName(Role role) {
  switch (role) {
    case Role::kWeight:
      return "weight";
    case Role::kState:
      return "state";
    case Role::kActivation:
      return "activation";
    case Role::kInput:
      return "input";
    case Role::kOutput:
      return "output";
    case Role::kConstant:
      return "constant";
  }
  return "unknown";
}

std::uint64_t UnitBytes(const format::TensorFormat& format) {
  const auto unit = format::AddressUnitOf(format);
  return unit ? unit->bytes : 0;
}

bool PreparedRows(const format::TensorFormat& format) {
  return format.encoding == format::Encoding::kMxfp8 &&
         format.layout == format::Layout::kScaleAtoms;
}

std::uint64_t UnitElements(const format::TensorFormat& format) {
  const auto unit = format::AddressUnitOf(format);
  return unit ? unit->elements : 0;
}

std::expected<std::uint64_t, std::string> StorageBytes(const TensorDesc& tensor) {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t unit = UnitBytes(tensor.format);
  const std::uint64_t per = UnitElements(tensor.format);
  if (unit == 0 || per == 0) {
    const auto parts = format::PartsOf(tensor.format);
    if (!parts) {
      return std::unexpected(std::string("a tensor's format has no size"));
    }
    std::uint64_t bytes = 0;
    for (const format::PartExtent& p : parts->parts()) {
      if (p.bytes > kMax - bytes) {
        return std::unexpected(std::string("a tensor's size overflows"));
      }
      bytes += p.bytes;
    }
    return bytes;
  }
  std::uint64_t bytes = 0;
  if (__builtin_mul_overflow(tensor.ne(0) / per, tensor.nb[0], &bytes)) {
    return std::unexpected(std::string("a tensor's size overflows"));
  }
  for (std::size_t d = 1; d < format::kMaxDims; ++d) {
    std::uint64_t reach = 0;
    if (tensor.ne(d) == 0 || __builtin_mul_overflow(tensor.ne(d) - 1, tensor.nb.at(d), &reach) ||
        reach > kMax - bytes) {
      return std::unexpected(std::string("a tensor's size overflows"));
    }
    bytes += reach;
  }
  return bytes;
}

std::array<std::uint64_t, format::kMaxDims> NaturalStrides(const format::TensorFormat& format) {
  std::array<std::uint64_t, format::kMaxDims> nb{};
  const std::uint64_t unit = UnitBytes(format);
  const std::uint64_t per = UnitElements(format);
  if (unit == 0 || per == 0) {
    return nb;
  }
  const format::Shape& s = format.shape;
  nb[0] = unit;
  nb[1] = unit * (s.ne[0] / per);
  nb[2] = nb[1] * s.ne[1];
  nb[3] = nb[2] * s.ne[2];
  return nb;
}

bool IsContiguous(const TensorDesc& tensor) {
  const auto unit = UnitBytes(tensor.format);
  const auto per = UnitElements(tensor.format);
  if (!unit || !per || tensor.ne(0) == 0 || tensor.ne(0) % per) return false;
  // A single unit never steps along axis zero. Other singleton axes likewise
  // have no observable stride: reshaping them cannot change any addressed byte.
  if (tensor.ne(0) != per && tensor.nb[0] != unit) return false;
  std::uint64_t next = 0;
  if (__builtin_mul_overflow(unit, tensor.ne(0) / per, &next)) return false;
  for (std::size_t axis = 1; axis < format::kMaxDims; ++axis) {
    if (tensor.ne(axis) == 0 || (tensor.ne(axis) != 1 && tensor.nb[axis] != next) ||
        __builtin_mul_overflow(next, tensor.ne(axis), &next))
      return false;
  }
  return true;
}

bool SameViewSource(const TensorDesc& was, const TensorDesc& now) {
  return was.format.shape == now.format.shape && was.format.encoding == now.format.encoding &&
         was.format.layout == now.format.layout && was.format.params == now.format.params &&
         was.nb == now.nb && was.view_offset == now.view_offset;
}

std::uint64_t Elements(const format::Shape& shape) {
  return shape.ne[0] * shape.ne[1] * shape.ne[2] * shape.ne[3];
}

format::TensorFormat Dense(format::Encoding encoding, const format::Shape& shape) {
  return {.encoding = encoding, .layout = format::Layout::kDense, .shape = shape};
}

format::Shape ShapeOf(std::initializer_list<std::uint64_t> ne) {
  format::Shape s;
  std::size_t i = 0;
  for (const std::uint64_t n : ne) {
    if (i < format::kMaxDims) {
      s.ne.at(i) = n;
    }
    ++i;
  }
  // The canonical rank: the last dimension past one, plus one.
  s.rank = 0;
  for (std::size_t d = 0; d < format::kMaxDims; ++d) {
    if (s.ne.at(d) != 1) {
      s.rank = static_cast<std::uint8_t>(d + 1);
    }
  }
  return s;
}

TensorId Graph::root(TensorId id) const {
  const TensorDesc& t = tensor(id);
  return t.view_of.valid() ? t.view_of : id;
}

std::string_view Graph::name(TensorId id) const {
  const std::uint32_t n = tensor(id).name;
  return n < names_.size() ? std::string_view(names_[n]) : std::string_view();
}

namespace {

class Hasher {
 public:
  void U8(std::uint8_t v) { Bytes(std::as_bytes(std::span(&v, 1))); }
  void U16(std::uint16_t v) {
    U8(static_cast<std::uint8_t>(v));
    U8(static_cast<std::uint8_t>(v >> 8U));
  }
  void U32(std::uint32_t v) {
    U16(static_cast<std::uint16_t>(v));
    U16(static_cast<std::uint16_t>(v >> 16U));
  }
  void U64(std::uint64_t v) {
    U32(static_cast<std::uint32_t>(v));
    U32(static_cast<std::uint32_t>(v >> 32U));
  }
  void Bytes(std::span<const std::byte> b) { sha_.Update(b); }
  base::Sha256Digest Finish() { return sha_.Finish(); }

 private:
  base::Sha256 sha_;
};

bool IsLeaf(Op op) { return op == Op::kInput || op == Op::kWeight || op == Op::kState; }

void Format(Hasher& h, const format::TensorFormat& f) {
  h.U16(static_cast<std::uint16_t>(f.encoding));
  h.U8(f.params.k_bits);
  h.U8(static_cast<std::uint8_t>(f.params.codebook));
  h.U8(f.params.tensor_scale ? 1 : 0);
  if (f.encoding == format::Encoding::kAffine) {  // other formats hash as before
    h.U8(f.params.bits);
    h.U32(f.params.group);
    h.U8(static_cast<std::uint8_t>(f.params.scale));
    h.U8(f.params.zero_point ? 1 : 0);
  }
  if (f.encoding == format::Encoding::kFp8Scaled) {
    h.U32(f.params.block_rows);
    h.U32(f.params.block_cols);
  }
  h.U16(static_cast<std::uint16_t>(f.layout));
  for (const std::uint64_t n : f.shape.ne) {
    h.U64(n);
  }
  h.U8(f.shape.rank);
  h.U8(f.stacked);
}

}  // namespace

Graph::Graph(Graph&& other) noexcept
    : storage_loan_(std::move(other.storage_loan_)),
      nodes_(std::move(other.nodes_)),
      tensors_(std::move(other.tensors_)),
      producers_(std::move(other.producers_)),
      inputs_(std::move(other.inputs_)),
      outputs_(std::move(other.outputs_)),
      keep_(std::move(other.keep_)),
      names_(std::move(other.names_)),
      digest_(other.digest_),
      identity_(std::exchange(other.identity_, 0)) {
  other.digest_ = {};
}

Graph& Graph::operator=(Graph&& other) noexcept {
  if (this != &other) {
    Graph incoming(std::move(other));
    std::swap(storage_loan_, incoming.storage_loan_);
    std::swap(nodes_, incoming.nodes_);
    std::swap(tensors_, incoming.tensors_);
    std::swap(producers_, incoming.producers_);
    std::swap(inputs_, incoming.inputs_);
    std::swap(outputs_, incoming.outputs_);
    std::swap(keep_, incoming.keep_);
    std::swap(names_, incoming.names_);
    std::swap(digest_, incoming.digest_);
    std::swap(identity_, incoming.identity_);
  }
  return *this;
}

std::uint64_t Graph::NextIdentity() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

std::expected<Graph, VerifyError> Graph::Clone(std::shared_ptr<const StorageFunder> funder,
                                               DiagnosticAdmission* diagnostic) const {
  const auto error = [diagnostic](GraphError code,
                                  std::string_view detail) -> std::unexpected<VerifyError> {
    if (code == GraphError::kAdmission)
      return std::unexpected(VerifyError{.error = code, .detail = {}});
    auto text = FormatDiagnostic(diagnostic, "{}", detail);
    if (!text) return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
    return std::unexpected(VerifyError{.error = code, .detail = std::move(*text)});
  };
  if (identity_ == 0) return error(GraphError::kBounds, "cannot clone a moved-from graph");
  if (!funder) funder = storage_loan_.funder();
  Graph g;
  std::uint64_t name_bytes = 0;
  for (const std::string& name : names_) {
    const std::uint64_t bytes = name.size() + 1;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - name_bytes)
      return error(GraphError::kAdmission, "comparison graph clone inventory overflowed");
    name_bytes += bytes;
  }
  std::uint64_t bytes = 0;
  constexpr std::uint64_t kVectorAllocationOverhead = 2 * sizeof(void*) + alignof(std::max_align_t);
  auto add = [&](std::uint64_t count, std::uint64_t size) {
    if (size != 0 && count > (std::numeric_limits<std::uint64_t>::max() - bytes) / size)
      return false;
    bytes += count * size;
    return true;
  };
  if (!add(nodes_.size(), sizeof(Node)) || !add(nodes_.size() != 0, kVectorAllocationOverhead) ||
      !add(tensors_.size(), sizeof(TensorDesc)) ||
      !add(tensors_.size() != 0, kVectorAllocationOverhead) ||
      !add(producers_.size(), sizeof(NodeId)) ||
      !add(producers_.size() != 0, kVectorAllocationOverhead) ||
      !add(inputs_.size(), sizeof(TensorId)) ||
      !add(inputs_.size() != 0, kVectorAllocationOverhead) ||
      !add(outputs_.size(), sizeof(TensorId)) ||
      !add(outputs_.size() != 0, kVectorAllocationOverhead) ||
      !add(keep_.size(), sizeof(TensorId)) || !add(keep_.size() != 0, kVectorAllocationOverhead) ||
      !add(names_.size(), sizeof(std::string)) ||
      !add(names_.size() != 0, kVectorAllocationOverhead) || !add(name_bytes, 1) ||
      !add(names_.size(), kVectorAllocationOverhead))
    return error(GraphError::kAdmission, "comparison graph clone inventory overflowed");
  if (funder) {
    auto loan = AcquireStorageLoan(funder, bytes);
    if (!loan)
      return error(GraphError::kAdmission, "comparison graph clone storage was not funded");
    g.storage_loan_ = std::move(*loan);
  }
  g.nodes_.reserve(nodes_.size());
  g.nodes_.insert(g.nodes_.end(), nodes_.begin(), nodes_.end());
  g.tensors_.reserve(tensors_.size());
  g.tensors_.insert(g.tensors_.end(), tensors_.begin(), tensors_.end());
  g.producers_.reserve(producers_.size());
  g.producers_.insert(g.producers_.end(), producers_.begin(), producers_.end());
  g.inputs_.reserve(inputs_.size());
  g.inputs_.insert(g.inputs_.end(), inputs_.begin(), inputs_.end());
  g.outputs_.reserve(outputs_.size());
  g.outputs_.insert(g.outputs_.end(), outputs_.begin(), outputs_.end());
  g.keep_.reserve(keep_.size());
  g.keep_.insert(g.keep_.end(), keep_.begin(), keep_.end());
  g.names_.reserve(names_.size());
  g.names_.insert(g.names_.end(), names_.begin(), names_.end());
  g.digest_ = digest_;
  g.identity_ = NextIdentity();
  return std::move(g);
}

base::Sha256Digest Graph::Digest() const { return digest_; }

std::expected<std::uint64_t, VerifyError> Graph::DigestScratchBound() const {
  const std::uint64_t tensors = tensors_.size();
  const std::uint64_t nodes = nodes_.size();
  const std::uint64_t leaves = static_cast<std::uint64_t>(
      std::ranges::count_if(nodes_, [](const Node& node) { return IsLeaf(node.key.op); }));
  std::uint64_t bytes = 0;
  auto add = [&](std::uint64_t count, std::uint64_t size) {
    if (size != 0 && count > (std::numeric_limits<std::uint64_t>::max() - bytes) / size)
      return false;
    bytes += count * size;
    return true;
  };
  constexpr std::uint64_t kVectorAllocationOverhead = 2 * sizeof(void*) + alignof(std::max_align_t);
  if (!add(tensors, sizeof(std::uint32_t)) || !add(tensors != 0, kVectorAllocationOverhead) ||
      !add(nodes, sizeof(std::uint32_t)) || !add(nodes != 0, kVectorAllocationOverhead) ||
      !add(leaves, sizeof(base::Sha256Digest)) || !add(leaves != 0, kVectorAllocationOverhead))
    return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
  return bytes;
}

base::Sha256Digest Graph::ComputeDigest() const {
  Hasher h;
  h.Bytes(std::as_bytes(std::span(std::string_view("llmp.ir.graph.v2"))));
  h.U32(kOpSetVersion);

  const std::size_t leaf_count = static_cast<std::size_t>(
      std::ranges::count_if(nodes_, [](const Node& node) { return IsLeaf(node.key.op); }));

  // Inputs are named by their position among the inputs.
  std::vector<std::uint32_t> input_index(tensors_.size(), ~std::uint32_t{0});
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    input_index.at(inputs_[i].v) = static_cast<std::uint32_t>(i);
  }
  std::vector<std::uint32_t> compute_index(nodes_.size(), ~std::uint32_t{0});
  std::uint32_t computed = 0;
  for (std::size_t n = 0; n < nodes_.size(); ++n) {
    if (!IsLeaf(nodes_[n].key.op)) {
      compute_index[n] = computed++;
    }
  }
  const auto ref = [&](TensorId id) {
    if (!id.valid()) {
      h.U8(0);  // absent
      return;
    }
    const NodeId p = producers_.at(id.v);
    const Node& node = nodes_.at(p.v);
    const TensorDesc& t = tensors_.at(id.v);
    if (IsLeaf(node.key.op)) {
      if (const auto* w = std::get_if<WeightRef>(&t.binding)) {
        h.U8(1);
        h.U16(w->stage);
        h.U16(w->component);
        h.U16(w->role);
        h.U32(w->layer);
        h.U8(w->expert_array ? 1 : 0);
      } else if (const auto* s = std::get_if<StateRef>(&t.binding)) {
        h.U8(2);
        h.U16(s->stage);
        h.U16(s->tensor);
        h.U32(s->layer);
        h.U8(s->segment);
      } else {
        h.U8(3);
        h.U32(input_index.at(id.v));
      }
      return;
    }
    h.U8(4);
    h.U32(compute_index.at(p.v));
    std::uint8_t slot = 0;
    for (std::uint8_t i = 0; i < node.nout; ++i) {
      if (node.out.at(i) == id) {
        slot = i;
      }
    }
    h.U8(slot);
  };
  const auto desc = [&](const TensorDesc& t) {
    h.U8(static_cast<std::uint8_t>(t.role));
    Format(h, t.format);
    for (const std::uint64_t b : t.nb) {
      h.U64(b);
    }
    h.U8(t.segment);
    if (t.view_of.valid()) {
      h.U8(1);
      ref(t.view_of);
      h.U64(t.view_offset);
    } else {
      h.U8(0);
    }
  };

  // Leaves, by what they bind, in a canonical order: their descriptions
  // under their references, sorted by those bytes (so declaring them in
  // another order digests the same).
  std::vector<base::Sha256Digest> leaves;
  leaves.reserve(leaf_count);
  for (const Node& node : nodes_) {
    if (!IsLeaf(node.key.op)) {
      continue;
    }
    Hasher leaf;
    const TensorDesc& t = tensors_.at(node.out[0].v);
    leaf.U16(static_cast<std::uint16_t>(node.key.op));
    if (const auto* w = std::get_if<WeightRef>(&t.binding)) {
      leaf.U8(1);
      leaf.U16(w->stage);
      leaf.U16(w->component);
      leaf.U16(w->role);
      leaf.U32(w->layer);
      leaf.U8(w->expert_array ? 1 : 0);
    } else if (const auto* s = std::get_if<StateRef>(&t.binding)) {
      leaf.U8(2);
      leaf.U16(s->stage);
      leaf.U16(s->tensor);
      leaf.U32(s->layer);
      leaf.U8(s->segment);
    } else {
      leaf.U8(3);
      leaf.U32(input_index.at(node.out[0].v));
    }
    leaf.U8(static_cast<std::uint8_t>(t.role));
    Format(leaf, t.format);
    for (const std::uint64_t b : t.nb) {
      leaf.U64(b);
    }
    leaf.U8(t.segment);
    // A plan constant's content digest, where it has one (its parameters
    // are otherwise zero, and then add nothing: the digests of graphs
    // without one are unchanged).
    if (t.role == Role::kConstant &&
        std::ranges::any_of(node.params, [](std::byte b) { return b != std::byte{0}; })) {
      leaf.Bytes(std::as_bytes(std::span(node.params).first(kConstantDigestBytes)));
    }
    // A step input's fact (InputParams), likewise only where it has one:
    // what a pass may read it as, which its name never decides.
    if (t.role == Role::kInput &&
        std::ranges::any_of(node.params, [](std::byte b) { return b != std::byte{0}; })) {
      leaf.U8(0x1F);
      leaf.Bytes(std::as_bytes(std::span(node.params).first(sizeof(InputParams))));
    }
    leaves.push_back(leaf.Finish());
  }
  std::ranges::sort(leaves);
  h.U64(leaves.size());
  for (const base::Sha256Digest& d : leaves) {
    h.Bytes(std::as_bytes(std::span(d)));
  }

  // Every other node in order.
  h.U64(computed);
  for (const Node& node : nodes_) {
    if (IsLeaf(node.key.op)) {
      continue;
    }
    h.U16(static_cast<std::uint16_t>(node.key.op));
    h.U16(static_cast<std::uint16_t>(node.key.fused));
    h.U8(static_cast<std::uint8_t>(node.batch));
    h.U16(node.component);
    h.Bytes(std::as_bytes(std::span(node.params)));
    h.U8(node.nsrc);
    for (const TensorId s : node.sources()) {
      ref(s);
    }
    h.U8(node.nout);
    for (const TensorId o : node.outputs()) {
      desc(tensors_.at(o.v));
    }
  }
  h.U64(outputs_.size());
  for (const TensorId o : outputs_) {
    ref(o);
  }
  h.U64(keep_.size());
  for (const TensorId k : keep_) {
    ref(k);
  }
  return h.Finish();
}

}  // namespace llmp::ir
