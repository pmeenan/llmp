// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/verify.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "ir/infer.h"
#include "ir/op.h"

namespace llmp::ir {

namespace {
constexpr std::uint64_t kDiagnosticAllocatorEnvelope =
    2 * sizeof(void*) + alignof(std::max_align_t);

std::size_t NumberText(std::uint64_t value, bool negative, char* out) {
  const auto result = std::to_chars(out + (negative ? 1 : 0), out + 32, value);
  if (result.ec != std::errc{}) return 0;
  if (negative) out[0] = '-';
  return static_cast<std::size_t>(result.ptr - out);
}

std::size_t ValueSize(const DiagnosticValue& value) {
  char digits[32];
  switch (value.kind) {
    case DiagnosticValue::Kind::kText:
      return value.text.size();
    case DiagnosticValue::Kind::kUnsigned:
      return NumberText(value.number, false, digits);
    case DiagnosticValue::Kind::kSigned: {
      const auto signed_value = value.signed_number;
      const bool negative = signed_value < 0;
      const std::uint64_t magnitude = negative ? static_cast<std::uint64_t>(-(signed_value + 1)) + 1
                                               : static_cast<std::uint64_t>(signed_value);
      return NumberText(magnitude, negative, digits);
    }
    case DiagnosticValue::Kind::kShape: {
      std::size_t size = 2 + 3 * 2;
      for (const auto dim : value.shape.ne) {
        char part[32];
        size += NumberText(dim, false, part);
      }
      return size;
    }
    case DiagnosticValue::Kind::kOpKey:
    case DiagnosticValue::Kind::kOutputName: {
      const std::size_t op_size =
          OpName(value.op_key.op).size() + (value.op_key.fused == FusedKind::kNone
                                                ? 0
                                                : 1 + FusedKindName(value.op_key.fused).size());
      if (value.kind == DiagnosticValue::Kind::kOpKey) return op_size;
      return op_size + sizeof("'s output ") - 1 + NumberText(value.output_index, false, digits);
    }
  }
  return 0;
}

char* WriteNumber(std::uint64_t value, bool negative, char* out) {
  char scratch[32];
  char* const digits = scratch + (negative ? 1 : 0);
  const auto result = std::to_chars(digits, scratch + sizeof(scratch), value);
  if (result.ec != std::errc{}) return out;
  char* end = result.ptr;
  if (negative) scratch[0] = '-';
  const auto length = static_cast<std::size_t>(end - scratch);
  std::copy_n(scratch, length, out);
  return out + length;
}

char* WriteValue(const DiagnosticValue& value, char* out) {
  switch (value.kind) {
    case DiagnosticValue::Kind::kText:
      std::copy(value.text.begin(), value.text.end(), out);
      return out + value.text.size();
    case DiagnosticValue::Kind::kUnsigned:
      return WriteNumber(value.number, false, out);
    case DiagnosticValue::Kind::kSigned: {
      const auto signed_value = value.signed_number;
      const bool negative = signed_value < 0;
      const std::uint64_t magnitude = negative ? static_cast<std::uint64_t>(-(signed_value + 1)) + 1
                                               : static_cast<std::uint64_t>(signed_value);
      return WriteNumber(magnitude, negative, out);
    }
    case DiagnosticValue::Kind::kShape: {
      *out++ = '[';
      for (std::size_t i = 0; i < value.shape.ne.size(); ++i) {
        if (i != 0) {
          *out++ = ',';
          *out++ = ' ';
        }
        out = WriteNumber(value.shape.ne[i], false, out);
      }
      *out++ = ']';
      return out;
    }
    case DiagnosticValue::Kind::kOpKey:
    case DiagnosticValue::Kind::kOutputName: {
      const auto op = OpName(value.op_key.op);
      std::copy(op.begin(), op.end(), out);
      out += op.size();
      if (value.op_key.fused != FusedKind::kNone) {
        *out++ = '.';
        const auto fused = FusedKindName(value.op_key.fused);
        std::copy(fused.begin(), fused.end(), out);
        out += fused.size();
      }
      if (value.kind == DiagnosticValue::Kind::kOutputName) {
        constexpr std::string_view label = "'s output ";
        std::copy(label.begin(), label.end(), out);
        out += label.size();
        out = WriteNumber(value.output_index, false, out);
      }
      return out;
    }
  }
  return out;
}
}  // namespace

std::expected<std::string, GraphError> MakeDiagnostic(DiagnosticAdmission* admission,
                                                      std::string_view format,
                                                      std::span<const DiagnosticValue> values) {
  std::uint64_t length = 0;
  std::size_t value_index = 0;
  for (std::size_t i = 0; i < format.size();) {
    if (format[i] == '{' && i + 1 < format.size() && format[i + 1] == '}') {
      if (value_index == values.size()) return std::unexpected(GraphError::kParams);
      const std::size_t part = ValueSize(values[value_index++]);
      if (part > std::numeric_limits<std::uint64_t>::max() - length)
        return std::unexpected(GraphError::kAdmission);
      length += part;
      i += 2;
      continue;
    }
    if (format[i] == '{' || format[i] == '}') return std::unexpected(GraphError::kParams);
    if (length == std::numeric_limits<std::uint64_t>::max())
      return std::unexpected(GraphError::kAdmission);
    ++length;
    ++i;
  }
  if (value_index != values.size() || length > std::numeric_limits<std::size_t>::max() ||
      length > std::numeric_limits<std::uint64_t>::max() - 1 - kDiagnosticAllocatorEnvelope)
    return std::unexpected(GraphError::kAdmission);
  if (admission != nullptr &&
      !admission->TryChargeDiagnostic(length + 1 + kDiagnosticAllocatorEnvelope))
    return std::unexpected(GraphError::kAdmission);
  const std::string probe;
  if (length > probe.max_size()) return std::unexpected(GraphError::kAdmission);
  std::string result(static_cast<std::size_t>(length), '\0');
  char* out = result.data();
  value_index = 0;
  for (std::size_t i = 0; i < format.size();) {
    if (format[i] == '{') {
      out = WriteValue(values[value_index++], out);
      i += 2;
    } else {
      *out++ = format[i++];
    }
  }
  return result;
}

std::string_view GraphErrorName(GraphError error) {
  switch (error) {
    case GraphError::kShape:
      return "shape";
    case GraphError::kFormat:
      return "format";
    case GraphError::kArity:
      return "arity";
    case GraphError::kBounds:
      return "bounds";
    case GraphError::kCycle:
      return "cycle";
    case GraphError::kParams:
      return "params";
    case GraphError::kBinding:
      return "binding";
    case GraphError::kLimit:
      return "limit";
    case GraphError::kAdmission:
      return "admission";
  }
  return "unknown";
}

namespace {

std::unexpected<VerifyError> Err(GraphError e, std::size_t node, std::string detail) {
  return std::unexpected(VerifyError{
      .error = e, .node = static_cast<std::uint32_t>(node), .detail = std::move(detail)});
}

template <class T>
struct IsDiagnosticMessage : std::false_type {};
template <std::size_t N>
struct IsDiagnosticMessage<DiagnosticMessage<N>> : std::true_type {};

template <class Detail>
std::unexpected<VerifyError> PaidErr(DiagnosticAdmission* admission, GraphError error,
                                     std::size_t node, const Detail& detail) {
  std::expected<std::string, GraphError> text = [&] {
    if constexpr (IsDiagnosticMessage<Detail>::value) {
      return MakeDiagnostic(admission, detail);
    } else {
      return FormatDiagnostic(admission, "{}", detail);
    }
  }();
  if (!text)
    return std::unexpected(VerifyError{
        .error = GraphError::kAdmission, .node = static_cast<std::uint32_t>(node), .detail = {}});
  return Err(error, node, std::move(*text));
}

std::uint8_t MinRank(const format::Shape& s) {
  std::uint8_t rank = 0;
  for (std::size_t d = 0; d < format::kMaxDims; ++d) {
    if (s.ne.at(d) != 1) {
      rank = static_cast<std::uint8_t>(d + 1);
    }
  }
  return rank;
}

bool SameExceptEncoding(const format::TensorFormat& a, const format::TensorFormat& b) {
  return a.layout == b.layout && a.params == b.params && a.shape.ne == b.shape.ne &&
         a.stacked == b.stacked;
}

bool IsLeaf(Op op) { return op == Op::kInput || op == Op::kWeight || op == Op::kState; }

bool AddBound(std::uint64_t& total, std::uint64_t count, std::uint64_t size) {
  if (size != 0 && count > (std::numeric_limits<std::uint64_t>::max() - total) / size) return false;
  total += count * size;
  return true;
}

constexpr std::uint64_t TreeNodeBound(std::uint64_t value_bytes) {
  // libstdc++'s _Rb_tree_node_base has three links and a color byte. Two
  // pointer widths and max alignment conservatively cover allocator/node
  // bookkeeping around the value object.
  return value_bytes + 5 * sizeof(void*) + alignof(std::max_align_t);
}

constexpr std::uint64_t kVectorAllocationOverhead = 2 * sizeof(void*) + alignof(std::max_align_t);

}  // namespace

std::expected<std::uint64_t, VerifyError> VerifyScratchBound(const Graph& graph) {
  const std::uint64_t nodes = graph.nodes().size();
  const std::uint64_t tensors = graph.tensor_count();
  const std::uint64_t outputs = graph.outputs().size();  // duplicates still occupy the input range
  std::uint64_t bytes = 0;
  const std::uint64_t bit_words = (tensors + std::numeric_limits<unsigned long>::digits - 1) /
                                  std::numeric_limits<unsigned long>::digits;
  if (!AddBound(bytes, tensors, sizeof(TensorDesc)) ||
      !AddBound(bytes, bit_words, sizeof(unsigned long)) ||
      !AddBound(bytes, tensors, sizeof(TensorId)) ||
      !AddBound(bytes, kMaxOutputs, sizeof(TensorDesc)) ||
      !AddBound(bytes, 4, kVectorAllocationOverhead) ||
      !AddBound(bytes, outputs, TreeNodeBound(sizeof(TensorId))) ||
      !AddBound(bytes, nodes, TreeNodeBound(sizeof(WeightRef))) ||
      !AddBound(bytes, nodes, TreeNodeBound(sizeof(StateRef)))) {
    return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
  }
  return bytes;
}

std::expected<void, VerifyError> Verify(const Graph& graph,
                                        std::shared_ptr<const StorageFunder> funder,
                                        DiagnosticAdmission* diagnostic) {
  if (!funder) funder = graph.storage_funder();
  StorageLoan scratch_loan;
  if (funder) {
    auto bound = VerifyScratchBound(graph);
    if (!bound) return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
    auto loan = AcquireStorageLoan(funder, *bound);
    if (!loan) return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
    scratch_loan = std::move(*loan);
  }
  const auto Err = [diagnostic](GraphError error, std::size_t node, const auto& detail) {
    return PaidErr(diagnostic, error, node, detail);
  };
  const std::span<const Node> nodes = graph.nodes();
  const std::size_t tensors = graph.tensor_count();
  if (nodes.size() > Graph::kMaxNodes || tensors > Graph::kMaxTensors) {
    return Err(GraphError::kBounds, 0,
               DiagFormat("{} nodes and {} tensors, past {} and {}", nodes.size(), tensors,
                          Graph::kMaxNodes, Graph::kMaxTensors));
  }
  std::vector<TensorDesc> descs;
  descs.reserve(tensors);
  for (std::size_t i = 0; i < tensors; ++i) {
    descs.push_back(graph.tensor(TensorId{static_cast<std::uint32_t>(i)}));
  }
  std::set<TensorId> outputs(graph.outputs().begin(), graph.outputs().end());
  std::vector<bool> made(tensors, false);
  std::set<WeightRef> weights;
  std::set<StateRef> states;
  std::vector<TensorId> inputs;
  std::vector<TensorDesc> requested;
  inputs.reserve(tensors);
  requested.reserve(kMaxOutputs);

  for (std::size_t n = 0; n < nodes.size(); ++n) {
    const Node& node = nodes[n];
    const OpKey op = node.key;
    if (node.nsrc > kMaxSources || node.nout > kMaxOutputs) {
      return Err(GraphError::kArity, n, DiagFormat("{}: too many sources or outputs", op));
    }
    for (const TensorId s : node.sources()) {
      if (!s.valid()) {
        continue;
      }
      if (s.v >= tensors || !made.at(s.v)) {
        return Err(GraphError::kCycle, n,
                   DiagFormat("{}: source {} is not made by an earlier node", op, s.v));
      }
    }
    requested.clear();
    for (const TensorId o : node.outputs()) {
      if (!o.valid() || o.v >= tensors || graph.producer(o).v != n || made.at(o.v)) {
        return Err(GraphError::kCycle, n, DiagFormat("{}: output {} is not this node's", op, o.v));
      }
      made.at(o.v) = true;
      requested.push_back(descs.at(o.v));
    }
    auto inferred =
        Infer(node.key, node.params, node.sources(), descs, requested, funder, diagnostic);
    if (!inferred) {
      VerifyError e = std::move(inferred.error());
      e.node = static_cast<std::uint32_t>(n);
      return std::unexpected(std::move(e));
    }
    if (inferred->nout != node.nout) {
      return Err(GraphError::kArity, n,
                 DiagFormat("{} makes {} outputs, not {}", op, inferred->nout, node.nout));
    }
    std::uint8_t owner = kJoinedSegment;
    for (std::size_t i = 0; i < node.nout; ++i) {
      const TensorId id = node.out.at(i);
      const TensorDesc& rec = descs.at(id.v);
      const TensorDesc& inf = inferred->out.at(i);
      const DiagnosticOutputName which{node.key, i};
      if (rec.format.shape.rank != MinRank(rec.format.shape)) {
        return Err(GraphError::kShape, n, DiagFormat("{} has a rank that is not canonical", which));
      }
      // A fused output recorded prepared (graph.h PreparedRows).
      const bool prepared = node.key.op == Op::kFused && inferred->float_result.at(i) &&
                            PreparedRows(rec.format) && rec.format.shape.ne == inf.format.shape.ne;
      if (prepared) {
        if (rec.nb != NaturalStrides(rec.format) || rec.view_of.valid()) {
          return Err(GraphError::kShape, n,
                     DiagFormat("{}'s prepared rows are not a root of their own", which));
        }
      } else if (!SameExceptEncoding(rec.format, inf.format)) {
        return Err(GraphError::kShape, n,
                   DiagFormat("{} is [{}, {}, {}, {}], where the op makes [{}, {}, {}, {}]", which,
                              rec.ne(0), rec.ne(1), rec.ne(2), rec.ne(3), inf.ne(0), inf.ne(1),
                              inf.ne(2), inf.ne(3)));
      }
      if (prepared) {
        // checked above
      } else if (rec.format.encoding != inf.format.encoding) {
        if (!inferred->float_result.at(i) || !ResultEncoding(rec.format.encoding)) {
          return Err(GraphError::kFormat, n,
                     DiagFormat("{} has another encoding than the op makes", which));
        }
        if (rec.nb != NaturalStrides(rec.format)) {
          return Err(GraphError::kShape, n, DiagFormat("{}'s strides are not packed", which));
        }
      } else if (rec.nb != inf.nb) {
        return Err(GraphError::kShape, n,
                   DiagFormat("{}'s strides are not what the op makes", which));
      }
      if (rec.view_of != inf.view_of || rec.view_offset != inf.view_offset) {
        return Err(GraphError::kShape, n, DiagFormat("{}'s view is not what the op makes", which));
      }
      // Roles and bindings.
      const bool leaf = IsLeaf(node.key.op);
      if (leaf) {
        const bool role_ok =
            (node.key.op == Op::kInput &&
             (rec.role == Role::kInput || rec.role == Role::kConstant) &&
             std::holds_alternative<std::monostate>(rec.binding)) ||
            (node.key.op == Op::kWeight && rec.role == Role::kWeight &&
             std::holds_alternative<WeightRef>(rec.binding) && rec.segment == kJoinedSegment) ||
            (node.key.op == Op::kState && rec.role == Role::kState &&
             std::holds_alternative<StateRef>(rec.binding) &&
             std::get_if<StateRef>(&rec.binding)->segment == rec.segment);
        if (!role_ok) {
          return Err(GraphError::kBinding, n,
                     DiagFormat("{}: a leaf's role, binding or segment", which));
        }
        if (const auto* w = std::get_if<WeightRef>(&rec.binding);
            w != nullptr && !weights.insert(*w).second) {
          return Err(GraphError::kBinding, n, DiagFormat("{}: a weight bound twice", which));
        }
        if (const auto* s = std::get_if<StateRef>(&rec.binding);
            s != nullptr && !states.insert(*s).second) {
          return Err(GraphError::kBinding, n, DiagFormat("{}: a state tensor bound twice", which));
        }
        if (node.key.op == Op::kInput) {
          inputs.push_back(id);
        }
      } else {
        if (!std::holds_alternative<std::monostate>(rec.binding)) {
          return Err(GraphError::kBinding, n, DiagFormat("{}: only leaves are bound", which));
        }
        const bool output =
            rec.role == Role::kOutput && inf.role == Role::kActivation && outputs.contains(id);
        if (rec.role != inf.role && !output) {
          return Err(
              GraphError::kBinding, n,
              DiagFormat("{} has role {}, not {}", which, RoleName(rec.role), RoleName(inf.role)));
        }
        if (inf.view_of.valid()) {
          if (rec.segment != inf.segment) {
            return Err(GraphError::kBinding, n,
                       DiagFormat("{}: a view's segment is its root's", which));
          }
        } else if (node.batch == Batch::kJoined) {
          if (rec.segment != kJoinedSegment) {
            return Err(GraphError::kBinding, n,
                       DiagFormat("{}: a joined node's result is joined", which));
          }
        } else if (rec.segment == kJoinedSegment) {
          return Err(GraphError::kBinding, n,
                     DiagFormat("{}: a per-owner node's result is its owner's", which));
        }
      }
      if (node.batch == Batch::kPerOwner) {
        if (rec.segment == kJoinedSegment || (owner != kJoinedSegment && owner != rec.segment)) {
          return Err(GraphError::kBinding, n,
                     DiagFormat("{}: a per-owner node is one owner's", which));
        }
        owner = rec.segment;
      }
    }
    if (node.batch == Batch::kPerOwner) {
      if (owner == kJoinedSegment) {
        return Err(GraphError::kBinding, n,
                   DiagFormat("{}: a per-owner node needs an owner's output", op));
      }
      for (const TensorId s : node.sources()) {
        if (s.valid() && descs.at(s.v).segment != kJoinedSegment &&
            descs.at(s.v).segment != owner) {
          return Err(
              GraphError::kBinding, n,
              DiagFormat("{}: reads segment {} for owner {}", op, descs.at(s.v).segment, owner));
        }
      }
    }
  }
  for (std::size_t i = 0; i < tensors; ++i) {
    if (!made.at(i)) {
      return Err(GraphError::kCycle, nodes.size(), DiagFormat("tensor {} is made by no node", i));
    }
  }
  if (inputs.size() != graph.inputs().size() ||
      !std::equal(inputs.begin(), inputs.end(), graph.inputs().begin())) {
    return Err(GraphError::kBinding, nodes.size(), "the inputs are not the input leaves in order");
  }
  for (const TensorId o : graph.outputs()) {
    if (!o.valid() || o.v >= tensors) {
      return Err(GraphError::kArity, nodes.size(), "an output that is not a tensor");
    }
  }
  for (const TensorId k : graph.keep()) {
    if (!k.valid() || k.v >= tensors) {
      return Err(GraphError::kArity, nodes.size(), "a kept tensor that is not a tensor");
    }
  }
  return {};
}

}  // namespace llmp::ir
