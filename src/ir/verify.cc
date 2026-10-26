// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/verify.h"

#include <algorithm>
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
  }
  return "unknown";
}

namespace {

std::unexpected<VerifyError> Err(GraphError e, std::size_t node, std::string detail) {
  return std::unexpected(VerifyError{
      .error = e, .node = static_cast<std::uint32_t>(node), .detail = std::move(detail)});
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

}  // namespace

std::expected<void, VerifyError> Verify(const Graph& graph) {
  const std::span<const Node> nodes = graph.nodes();
  const std::size_t tensors = graph.tensor_count();
  if (nodes.size() > Graph::kMaxNodes || tensors > Graph::kMaxTensors) {
    return Err(GraphError::kBounds, 0,
               std::format("{} nodes and {} tensors, past {} and {}", nodes.size(), tensors,
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

  for (std::size_t n = 0; n < nodes.size(); ++n) {
    const Node& node = nodes[n];
    const std::string op = OpKeyName(node.key);
    if (node.nsrc > kMaxSources || node.nout > kMaxOutputs) {
      return Err(GraphError::kArity, n, std::format("{}: too many sources or outputs", op));
    }
    for (const TensorId s : node.sources()) {
      if (!s.valid()) {
        continue;
      }
      if (s.v >= tensors || !made.at(s.v)) {
        return Err(GraphError::kCycle, n,
                   std::format("{}: source {} is not made by an earlier node", op, s.v));
      }
    }
    requested.clear();
    for (const TensorId o : node.outputs()) {
      if (!o.valid() || o.v >= tensors || graph.producer(o).v != n || made.at(o.v)) {
        return Err(GraphError::kCycle, n, std::format("{}: output {} is not this node's", op, o.v));
      }
      made.at(o.v) = true;
      requested.push_back(descs.at(o.v));
    }
    auto inferred = Infer(node.key, node.params, node.sources(), descs, requested);
    if (!inferred) {
      VerifyError e = std::move(inferred.error());
      e.node = static_cast<std::uint32_t>(n);
      return std::unexpected(e);
    }
    if (inferred->nout != node.nout) {
      return Err(GraphError::kArity, n,
                 std::format("{} makes {} outputs, not {}", op, inferred->nout, node.nout));
    }
    std::uint8_t owner = kJoinedSegment;
    for (std::size_t i = 0; i < node.nout; ++i) {
      const TensorId id = node.out.at(i);
      const TensorDesc& rec = descs.at(id.v);
      const TensorDesc& inf = inferred->out.at(i);
      const std::string which = std::format("{}'s output {}", op, i);
      if (rec.format.shape.rank != MinRank(rec.format.shape)) {
        return Err(GraphError::kShape, n, which + " has a rank that is not canonical");
      }
      // A fused output recorded prepared (graph.h PreparedRows).
      const bool prepared = node.key.op == Op::kFused && inferred->float_result.at(i) &&
                            PreparedRows(rec.format) && rec.format.shape.ne == inf.format.shape.ne;
      if (prepared) {
        if (rec.nb != NaturalStrides(rec.format) || rec.view_of.valid()) {
          return Err(GraphError::kShape, n, which + "'s prepared rows are not a root of their own");
        }
      } else if (!SameExceptEncoding(rec.format, inf.format)) {
        return Err(GraphError::kShape, n,
                   std::format("{} is [{}, {}, {}, {}], where the op makes [{}, {}, {}, {}]", which,
                               rec.ne(0), rec.ne(1), rec.ne(2), rec.ne(3), inf.ne(0), inf.ne(1),
                               inf.ne(2), inf.ne(3)));
      }
      if (prepared) {
        // checked above
      } else if (rec.format.encoding != inf.format.encoding) {
        if (!inferred->float_result.at(i) || !ResultEncoding(rec.format.encoding)) {
          return Err(GraphError::kFormat, n, which + " has another encoding than the op makes");
        }
        if (rec.nb != NaturalStrides(rec.format)) {
          return Err(GraphError::kShape, n, which + "'s strides are not packed");
        }
      } else if (rec.nb != inf.nb) {
        return Err(GraphError::kShape, n, which + "'s strides are not what the op makes");
      }
      if (rec.view_of != inf.view_of || rec.view_offset != inf.view_offset) {
        return Err(GraphError::kShape, n, which + "'s view is not what the op makes");
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
          return Err(GraphError::kBinding, n, which + ": a leaf's role, binding or segment");
        }
        if (const auto* w = std::get_if<WeightRef>(&rec.binding);
            w != nullptr && !weights.insert(*w).second) {
          return Err(GraphError::kBinding, n, which + ": a weight bound twice");
        }
        if (const auto* s = std::get_if<StateRef>(&rec.binding);
            s != nullptr && !states.insert(*s).second) {
          return Err(GraphError::kBinding, n, which + ": a state tensor bound twice");
        }
        if (node.key.op == Op::kInput) {
          inputs.push_back(id);
        }
      } else {
        if (!std::holds_alternative<std::monostate>(rec.binding)) {
          return Err(GraphError::kBinding, n, which + ": only leaves are bound");
        }
        const bool output =
            rec.role == Role::kOutput && inf.role == Role::kActivation && outputs.contains(id);
        if (rec.role != inf.role && !output) {
          return Err(
              GraphError::kBinding, n,
              std::format("{} has role {}, not {}", which, RoleName(rec.role), RoleName(inf.role)));
        }
        if (inf.view_of.valid()) {
          if (rec.segment != inf.segment) {
            return Err(GraphError::kBinding, n, which + ": a view's segment is its root's");
          }
        } else if (node.batch == Batch::kJoined) {
          if (rec.segment != kJoinedSegment) {
            return Err(GraphError::kBinding, n, which + ": a joined node's result is joined");
          }
        } else if (rec.segment == kJoinedSegment) {
          return Err(GraphError::kBinding, n, which + ": a per-owner node's result is its owner's");
        }
      }
      if (node.batch == Batch::kPerOwner) {
        if (rec.segment == kJoinedSegment || (owner != kJoinedSegment && owner != rec.segment)) {
          return Err(GraphError::kBinding, n, which + ": a per-owner node is one owner's");
        }
        owner = rec.segment;
      }
    }
    if (node.batch == Batch::kPerOwner) {
      if (owner == kJoinedSegment) {
        return Err(GraphError::kBinding, n, op + ": a per-owner node needs an owner's output");
      }
      for (const TensorId s : node.sources()) {
        if (s.valid() && descs.at(s.v).segment != kJoinedSegment &&
            descs.at(s.v).segment != owner) {
          return Err(
              GraphError::kBinding, n,
              std::format("{}: reads segment {} for owner {}", op, descs.at(s.v).segment, owner));
        }
      }
    }
  }
  for (std::size_t i = 0; i < tensors; ++i) {
    if (!made.at(i)) {
      return Err(GraphError::kCycle, nodes.size(), std::format("tensor {} is made by no node", i));
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
