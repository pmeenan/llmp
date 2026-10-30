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

}  // namespace

std::expected<std::uint64_t, std::string> RootBytes(const ir::TensorDesc& tensor) {
  const std::uint64_t unit = ir::UnitBytes(tensor.format);
  const std::uint64_t per = ir::UnitElements(tensor.format);
  if (unit == 0 || per == 0) {
    const auto parts = format::PartsOf(tensor.format);
    if (!parts) {
      return Error("a tensor's format has no size");
    }
    std::uint64_t bytes = 0;
    for (const format::PartExtent& p : parts->parts()) {
      if (p.bytes > kMax - bytes) {
        return Error("a tensor's size overflows");
      }
      bytes += p.bytes;
    }
    return bytes;
  }
  std::uint64_t bytes = 0;
  if (__builtin_mul_overflow(tensor.ne(0) / per, tensor.nb[0], &bytes)) {
    return Error("a tensor's size overflows");
  }
  for (std::size_t d = 1; d < format::kMaxDims; ++d) {
    std::uint64_t reach = 0;
    if (tensor.ne(d) == 0 || __builtin_mul_overflow(tensor.ne(d) - 1, tensor.nb.at(d), &reach) ||
        reach > kMax - bytes) {
      return Error("a tensor's size overflows");
    }
    bytes += reach;
  }
  return bytes;
}

std::expected<Fit, std::string> FitIntervals(std::span<const Interval> intervals) {
  Fit fit;
  fit.offsets.reserve(intervals.size());
  if (intervals.empty()) return fit;
  const auto place = [&](const Interval& v,
                         std::uint64_t offset) -> std::expected<void, std::string> {
    if (offset > kMax - v.bytes) return Error("the activation region overflows");
    fit.offsets.push_back(offset);
    fit.extent = std::max(fit.extent, offset + v.bytes);
    return {};
  };
  // Short lifetimes (a plan without lanes: a step or a few, a mean of 8 to
  // 9 in Qwen3.8's) meet few placed intervals each, found by their first
  // steps; long ones (lanes' regions: means of 60 to 200) meet many, and
  // sorting those costs more than the scan, which walks every placed
  // interval below the result (measured on Qwen3.8's plans, F16).
  constexpr std::int64_t kShort = 32;
  std::int64_t low = intervals.front().first, high = intervals.front().last;
  std::uint64_t steps = 0;
  bool ordered = true;
  for (const Interval& v : intervals) {
    low = std::min(low, v.first);
    high = std::max(high, v.last);
    ordered = ordered && v.first <= v.last;
    // Unsigned: a span of the whole int64 range must not overflow.
    if (ordered) steps += static_cast<std::uint64_t>(v.last) - static_cast<std::uint64_t>(v.first);
  }
  const std::uint64_t span = static_cast<std::uint64_t>(high) - static_cast<std::uint64_t>(low);
  const auto count = static_cast<std::uint64_t>(intervals.size());
  if (!ordered || steps > 16 * count || span > 4 * count + 1024) {
    // The scan: every placed interval in offset order, past each whose
    // steps meet this one's, until one starts beyond the candidate's end.
    struct Placed {
      std::uint64_t offset = 0, end = 0;
      std::int64_t first = 0, last = 0;
    };
    std::vector<Placed> placed;
    placed.reserve(intervals.size());
    for (const Interval& v : intervals) {
      std::uint64_t offset = 0;
      for (const Placed& b : placed) {
        if (b.offset >= offset + v.bytes) break;
        if (b.first <= v.last && v.first <= b.last) offset = std::max(offset, b.end);
      }
      if (auto placed_at = place(v, offset); !placed_at) return std::unexpected(placed_at.error());
      const Placed at{offset, offset + v.bytes, v.first, v.last};
      placed.insert(std::ranges::lower_bound(placed, at.offset, {}, &Placed::offset), at);
    }
    return fit;
  }
  // By first step those of at most kShort steps (one starting before this
  // one's first by more ends before it), the longer in one list: together
  // every placed interval whose steps meet this one's, swept in offset
  // order to the same lowest free offset the scan finds.
  const auto position = [&](std::int64_t step) {
    return static_cast<std::size_t>(static_cast<std::uint64_t>(step) -
                                    static_cast<std::uint64_t>(low));
  };
  std::vector<std::vector<std::uint32_t>> starting(static_cast<std::size_t>(span) + 1);
  std::vector<std::uint32_t> longer, met;
  std::vector<std::uint64_t> ends;
  ends.reserve(intervals.size());
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    const Interval& v = intervals[i];
    const auto meets = [&](std::uint32_t j) {
      return intervals[j].first <= v.last && v.first <= intervals[j].last;
    };
    met.clear();
    for (const std::uint32_t j : longer)
      if (meets(j)) met.push_back(j);
    const std::int64_t from = v.first - low > kShort ? v.first - kShort : low;
    for (std::size_t q = position(from); q <= position(v.last); ++q)
      for (const std::uint32_t j : starting[q])
        if (meets(j)) met.push_back(j);
    std::ranges::sort(met, {}, [&](std::uint32_t j) { return fit.offsets[j]; });
    std::uint64_t offset = 0;
    for (const std::uint32_t j : met) {
      if (fit.offsets[j] >= offset + v.bytes) break;
      offset = std::max(offset, ends[j]);
    }
    if (auto placed_at = place(v, offset); !placed_at) return std::unexpected(placed_at.error());
    ends.push_back(offset + v.bytes);
    const auto id = static_cast<std::uint32_t>(i);
    if (v.last - v.first > kShort)
      longer.push_back(id);
    else
      starting[position(v.first)].push_back(id);
  }
  return fit;
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
  struct Life {
    ir::TensorId tensor{};
    std::int64_t first = 0;  // the step computing it; -1 for a staged input
    std::int64_t last = 0;   // the last step reading it
    std::uint64_t bytes = 0;
    std::uint64_t offset = 0;
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
  std::vector<Interval> intervals;
  intervals.reserve(order.size());
  for (const std::size_t i : order)
    intervals.push_back({.first = lives[i].first, .last = lives[i].last, .bytes = lives[i].bytes});
  auto fit = FitIntervals(intervals);
  if (!fit) return std::unexpected(std::move(fit.error()));
  for (std::size_t k = 0; k < order.size(); ++k) lives[order[k]].offset = fit->offsets[k];
  placement.extent = fit->extent;
  placement.offsets.reserve(lives.size());
  for (const Life& life : lives) {
    placement.offsets.emplace_back(life.tensor, life.offset);
  }
  return placement;
}

}  // namespace llmp::planner
