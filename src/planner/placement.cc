// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/placement.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "format/tensor_format.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/lanes.h"
#include "planner/select.h"

namespace llmp::planner {
namespace {

std::unexpected<std::string> Error(std::string detail) {
  return std::unexpected(std::move(detail));
}

constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();

bool Placed(const ir::TensorDesc& t) {
  return !t.is_view() && (t.role == ir::Role::kActivation || t.role == ir::Role::kOutput);
}

struct Life {
  ir::TensorId tensor{};
  std::int64_t first = 0;
  std::int64_t last = 0;
  std::uint64_t bytes = 0;
  std::uint64_t offset = 0;
  bool placed = false;
};

struct PlacedSpan {
  std::uint64_t offset = 0;
  std::uint64_t end = 0;
  std::int64_t first = 0;
  std::int64_t last = 0;
};

constexpr std::uint64_t kVectorEnvelope = ir::kAdmissionHeapBlockOverhead;

bool AddExact(std::uint64_t count, std::uint64_t stride, std::uint64_t& total) {
  return ir::AddVectorAdmissionBytes(count, stride, total);
}

bool AddGeometric(std::uint64_t count, std::uint64_t stride, std::uint64_t& total) {
  if (count == 0) return true;
  if (stride == 0 || stride > kMax / 3 || total > kMax - 2 * kVectorEnvelope ||
      count > (kMax - total - 2 * kVectorEnvelope) / (3 * stride))
    return false;
  total += count * 3 * stride + 2 * kVectorEnvelope;
  return true;
}

bool AddBits(std::uint64_t count, std::uint64_t& total) {
  constexpr std::uint64_t kWordBits = std::numeric_limits<unsigned long>::digits;
  if (count > kMax - (kWordBits - 1)) return false;
  return AddExact((count + kWordBits - 1) / kWordBits, sizeof(unsigned long), total);
}

}  // namespace

std::expected<std::uint64_t, std::string> RootBytes(const ir::TensorDesc& tensor) {
  return ir::StorageBytes(tensor);
}

std::expected<std::uint64_t, std::string> PlacementMetadataBytes(const ir::Graph& graph) {
  const std::uint64_t tensors = graph.tensor_count();
  const std::uint64_t nodes = graph.nodes().size();
  std::uint64_t bytes = 0;
  // PlaceActivations: geometric lives growth; exact index/inside/root_of,
  // order, placed and result offsets; rounded stepped bitmap; stable_sort's
  // full index buffer. Alignments() may grow geometrically and the final
  // alignment audit owns a tensor-indexed offset and bitmap.
  if (!AddGeometric(tensors, sizeof(Life), bytes) ||
      !AddExact(tensors, sizeof(std::size_t), bytes) ||
      !AddExact(tensors, sizeof(ir::TensorId), bytes) ||
      !AddExact(tensors, sizeof(ir::TensorId), bytes) || !AddBits(nodes, bytes) ||
      !AddExact(tensors, sizeof(ir::TensorId), bytes) ||
      !AddExact(tensors, sizeof(std::size_t), bytes) ||
      !AddExact(tensors, sizeof(std::size_t), bytes) ||
      !AddExact(tensors, sizeof(PlacedSpan), bytes) ||
      !AddExact(tensors, sizeof(std::size_t), bytes) ||
      !AddExact(tensors, sizeof(std::pair<ir::TensorId, std::uint64_t>), bytes) ||
      !AddGeometric(tensors, sizeof(RootAlignmentFact), bytes) ||
      !AddExact(tensors, sizeof(std::uint64_t), bytes) || !AddBits(tensors, bytes))
    return Error("comparison placement metadata inventory overflows");
  return bytes;
}

std::expected<Placement, std::string> PlaceActivations(
    const ir::Graph& graph, std::span<const Selected> steps, std::span<const Region> regions,
    std::span<const ir::TensorId> staged, std::uint64_t alignment, const RootBytesOf& written,
    const RootWithin& within, std::span<const ir::TensorId> preserved,
    const PhysicalStorage* physical) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Error("the alignment is not a power of two");
  }
  if (physical && (physical->materialized.size() != graph.tensor_count() ||
                   physical->reads.size() != graph.nodes().size()))
    return Error("physical storage inventory differs from the graph");
  if (physical)
    for (const auto& reads : physical->reads)
      for (const auto t : reads)
        if (!t.valid() || t.v >= graph.tensor_count() || !physical->materialized[graph.root(t).v])
          return Error("a physical read names an absent materialization");
  const auto sources = [&](ir::NodeId id) -> std::span<const ir::TensorId> {
    return physical ? std::span<const ir::TensorId>(physical->reads[id.v])
                    : graph.node(id).sources();
  };
  std::vector<Life> lives;
  std::vector<std::size_t> index(graph.tensor_count(), ~std::size_t{0});
  // A root inside another's place stands for it.
  std::vector<ir::TensorId> inside(graph.tensor_count(), ir::TensorId{});
  const auto track = [&](ir::TensorId t, std::int64_t first) -> std::expected<void, std::string> {
    if (physical && !physical->materialized[t.v]) return {};
    if (index.at(t.v) != ~std::size_t{0}) {
      return {};
    }
    if (within) {
      if (const ir::TensorId owner = within(graph, t); owner.valid() && owner != t) {
        inside.at(t.v) = owner;
        return {};
      }
    }
    auto bytes = RootBytes(graph.tensor(t));
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
    if (written) {
      bytes = std::max(*bytes, written(graph, t));
    }
    if (*bytes > kMax - (alignment - 1)) {
      return Error("an activation's rounded size overflows");
    }
    index.at(t.v) = lives.size();
    lives.push_back({.tensor = t,
                     .first = first,
                     .last = first,
                     .bytes = (*bytes + alignment - 1) / alignment * alignment});
    return {};
  };
  for (const ir::TensorId input : staged) {
    const ir::TensorDesc& t = graph.tensor(input);
    if (t.is_view() || (t.role != ir::Role::kInput && t.role != ir::Role::kConstant)) {
      return Error("a staged tensor is not an input leaf");
    }
    if (auto r = track(input, -1); !r) {
      return std::unexpected(r.error());
    }
  }
  std::vector<bool> stepped(graph.nodes().size(), false);
  for (std::size_t s = 0; s < steps.size(); ++s) {
    for (const ir::NodeId id : steps[s].nodes) {
      stepped.at(id.v) = true;
      for (const ir::TensorId o : graph.node(id).outputs()) {
        if (Placed(graph.tensor(o))) {
          if (auto r = track(o, static_cast<std::int64_t>(s)); !r) {
            return std::unexpected(r.error());
          }
        }
      }
    }
  }
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const ir::Node& n = graph.nodes()[i];
    if (!LaunchesNothing(n.key) && !stepped[i]) {
      return Error(std::format("node {} ({}) is computed by no step", i, ir::OpKeyName(n.key)));
    }
  }
  std::vector<ir::TensorId> root_of;
  root_of.reserve(graph.tensor_count());
  for (std::uint32_t i = 0; i < graph.tensor_count(); ++i) {
    root_of.push_back(graph.root(ir::TensorId{i}));
  }
  const auto holder = [&](ir::TensorId t) {
    const ir::TensorId r = root_of[t.v];
    return inside.at(r.v).valid() ? inside.at(r.v) : r;
  };
  const auto read = [&](ir::TensorId t, std::int64_t s) {
    const std::size_t at = index.at(holder(t).v);
    if (at != ~std::size_t{0}) {
      lives[at].last = std::max(lives[at].last, s);
    }
  };
  for (std::size_t s = 0; s < steps.size(); ++s) {
    const auto step = static_cast<std::int64_t>(s);
    for (const ir::NodeId id : steps[s].nodes) {
      for (const ir::TensorId src : sources(id)) {
        if (!src.valid()) {
          continue;
        }
        read(src, step);
        // A prepared input keeps its source live with it.
        const ir::Node& made = graph.node(graph.producer(root_of[src.v]));
        if (made.key == ir::Primitive(ir::Op::kPrepare)) {
          read(made.src[0], step);
        }
      }
    }
  }
  for (const Region& region : regions) {
    const auto first = static_cast<std::int64_t>(region.first);
    const auto last = static_cast<std::int64_t>(region.last);
    const auto widen = [&](ir::TensorId t) {
      const std::size_t at = index.at(holder(t).v);
      if (at != ~std::size_t{0}) {
        Life& life = lives[at];
        life.first = life.first < 0 ? life.first : std::min(life.first, first);
        life.last = std::max(life.last, last);
      }
    };
    for (std::size_t s = region.first; s <= region.last && s < steps.size(); ++s) {
      for (const ir::NodeId id : steps[s].nodes) {
        const ir::Node& n = graph.node(id);
        for (const ir::TensorId o : n.outputs()) {
          widen(o);
        }
        for (const ir::TensorId src : sources(id)) {
          if (src.valid()) {
            widen(src);
          }
        }
      }
    }
  }
  const auto end = static_cast<std::int64_t>(steps.size());
  const auto to_end = [&](ir::TensorId t) {
    const std::size_t at = index.at(holder(t).v);
    if (at != ~std::size_t{0}) {
      lives[at].last = end;
    }
  };
  for (std::size_t i = graph.nodes().size(); i-- > 0;) {
    const ir::Node& n = graph.nodes()[i];
    if (!LaunchesNothing(n.key)) {
      for (const ir::TensorId o : n.outputs()) {
        to_end(o);
      }
      break;
    }
  }
  for (const auto list : {graph.outputs(), graph.keep()}) {
    for (const ir::TensorId t : list) {
      to_end(t);
    }
  }
  for (const ir::TensorId t : preserved) {
    to_end(t);
    if (const std::size_t at = index.at(holder(t).v); at != ~std::size_t{0}) lives[at].first = -1;
  }
  std::vector<std::size_t> order(lives.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(
      order, [&](std::size_t a, std::size_t b) { return lives[a].bytes > lives[b].bytes; });
  Placement placement{.alignment = alignment};
  std::vector<PlacedSpan> placed;
  placed.reserve(lives.size());
  for (const std::size_t i : order) {
    Life& life = lives[i];
    std::uint64_t offset = 0;
    for (const auto& b : placed) {
      if (b.offset >= offset + life.bytes) {
        break;
      }
      if (b.first <= life.last && life.first <= b.last) {
        offset = std::max(offset, b.end);
      }
    }
    if (offset > kMax - life.bytes) {
      return Error("the activation region overflows");
    }
    life.offset = offset;
    life.placed = true;
    placement.extent = std::max(placement.extent, offset + life.bytes);
    PlacedSpan span{offset, offset + life.bytes, life.first, life.last};
    auto it = std::ranges::lower_bound(placed, span.offset, {}, &PlacedSpan::offset);
    placed.insert(it, span);
  }
  placement.offsets.reserve(lives.size());
  for (const Life& life : lives) {
    placement.offsets.emplace_back(life.tensor, life.offset);
  }
  return placement;
}

}  // namespace llmp::planner
