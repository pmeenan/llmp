// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "components/step.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "base/check.h"
#include "components/component.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"
#include "model/batching.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "spec/spec.h"

namespace llmp::components {
namespace {

std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}

std::uint64_t Pad(std::uint64_t value, std::uint64_t align) {
  return (value + align - 1) / align * align;
}

std::string Where(const spec::Stage& stage, Site site, std::uint32_t layer) {
  switch (site) {
    case Site::kEmbedding:
      return std::format("stage {} embedding", stage.name);
    case Site::kLayer:
      return std::format("stage {} layer {}", stage.name, layer);
    case Site::kFinalNorm:
      return std::format("stage {} final norm", stage.name);
    case Site::kOutput:
      return std::format("stage {} output", stage.name);
  }
  return std::format("stage {}", stage.name);
}

// A state tensor of `cells` cells as its contract stores them: dense
// values, or a block encoding's whole blocks in GGML's layout (the cell's
// bytes model::CellBytes gave the contract).
format::TensorFormat CellFormat(const model::StateTensorContract& c, std::uint64_t cells) {
  const format::EncodingInfo& info = format::Info(c.encoding);
  if (info.ggml.has_value() && info.quantized) {
    const std::uint64_t values = c.cell_bytes / info.ggml->block_bytes * info.ggml->block_elements;
    return {.encoding = c.encoding,
            .layout = format::Layout::kGgmlBlocks,
            .shape = ir::ShapeOf({values, cells})};
  }
  return ir::Dense(c.encoding, ir::ShapeOf({c.cell_bytes / (info.element_bits / 8), cells}));
}

// ------------------------------------------------------------ the builder

// Joins a stage's fragments into one graph: the shared step context, the
// weight leaves (once per bound tensor) and the state leaves.
class StepBuilder {
 public:
  StepBuilder(std::uint16_t stage, const planner::StepShape& shape)
      : stage_(stage), ctx_(b_, Segments(shape), Greedy(shape)) {}

  static std::vector<SegmentRows> Segments(const planner::StepShape& shape) {
    std::vector<SegmentRows> out;
    std::uint32_t first = 0;
    for (const planner::Segment& s : shape.segments) {
      out.push_back({.slot = s.slot,
                     .first_row = first,
                     .rows = s.rows,
                     .past = s.past,
                     .read_cells = s.read_cells,
                     .outputs = s.outputs});
      first += s.rows;
    }
    return out;
  }
  static bool Greedy(const planner::StepShape& shape) {
    return !shape.segments.empty() &&
           std::ranges::all_of(shape.segments, [](const auto& s) { return s.greedy; });
  }

  ir::Builder& b() { return b_; }
  StepContext& ctx() { return ctx_; }
  void Begin(const Instance& in) {
    b_.SetComponent(in.index);
    ctx_.SetComponent(in.index);
  }

  // A weight leaf of `format` for (instance, role), shared by every role
  // bound to the same tensor (`key`: its first part's name; empty: never
  // shared).
  ir::TensorId Weight(const Instance& in, std::uint16_t role, const format::TensorFormat& format,
                      std::vector<std::pair<std::string, format::PartRole>> parts) {
    const std::string key = parts.empty() ? std::string{} : parts.front().first;
    if (!key.empty()) {
      if (const auto at = by_name_.find(key); at != by_name_.end()) return at->second;
    }
    const ir::WeightRef ref{.stage = stage_,
                            .component = in.index,
                            .role = role,
                            .layer = in.layer,
                            .expert_array = false};
    const std::string name =
        key.empty() ? std::format("{}.{}", in.def->kind, in.def->roles[role].name) : key;
    const ir::TensorId id = b_.Weight(format, ref, name);
    if (!key.empty()) by_name_.emplace(key, id);
    weights_.push_back({.tensor = id, .ref = ref, .parts = std::move(parts)});
    return id;
  }

  // State leaves for an instance's contract tensors from `first` in its
  // layer's contract: by segment, then tensor.
  std::vector<ir::TensorId> State(const Instance& in, const model::StateContract& contract,
                                  std::uint16_t first, std::span<const std::uint64_t> cells,
                                  std::span<const std::size_t> items) {
    std::vector<ir::TensorId> out;
    for (std::size_t s = 0; s < ctx_.segments().size(); ++s) {
      for (std::size_t t = 0; t < contract.tensors.size(); ++t) {
        const model::StateTensorContract& c = contract.tensors[t];
        const ir::StateRef ref{.stage = stage_,
                               .tensor = static_cast<std::uint16_t>(first + t),
                               .layer = in.layer,
                               .segment = static_cast<std::uint8_t>(s)};
        const ir::TensorId id = b_.State(CellFormat(c, cells[t]), ref,
                                         std::format("blk.{}.{}.{}", in.layer, c.name, s));
        out.push_back(id);
        state_.push_back({.tensor = id, .ref = ref, .item = items.empty() ? 0 : items[t]});
      }
    }
    return out;
  }

  std::expected<StepGraph, std::string> Finish(ir::TensorId logits, ir::TensorId greedy) && {
    auto inputs = ctx_.TakeInputs();
    auto graph = std::move(b_).Finish();
    if (!graph) {
      return Refused(std::format("the step's graph does not verify at node {}: {} ({})",
                                 graph.error().node, graph.error().detail,
                                 ir::GraphErrorName(graph.error().error)));
    }
    return StepGraph{.graph = std::move(*graph),
                     .inputs = std::move(inputs),
                     .weights = std::move(weights_),
                     .state = std::move(state_),
                     .logits = logits,
                     .greedy = greedy};
  }

 private:
  std::uint16_t stage_;
  ir::Builder b_;
  StepContext ctx_;
  std::map<std::string, ir::TensorId> by_name_;
  std::vector<StepWeight> weights_;
  std::vector<StepState> state_;
};

std::expected<void, std::string> CheckShape(const planner::StepShape& shape) {
  if (shape.segments.empty() || shape.segments.size() > kMaxSegments) {
    return Refused(std::format("a step of {} segments", shape.segments.size()));
  }
  const bool greedy = shape.segments.front().greedy;
  std::set<std::uint8_t> slots;
  for (const planner::Segment& s : shape.segments) {
    if (s.rows == 0 || s.outputs > s.rows || !slots.insert(s.slot).second) {
      return Refused("an empty or repeated segment, or more outputs than rows");
    }
    if (s.greedy != greedy || (greedy && s.outputs != 1)) {
      return Refused("greedy segments are all of them, each with one output");
    }
    if (s.past > std::numeric_limits<std::uint32_t>::max() - s.rows) {
      return Refused("a segment's positions overflow");
    }
  }
  return {};
}

// Whether a segment's state reads cover its rows: a per-position tensor's
// cells hold every position; a ring that wraps keeps every row's window
// once all the step's rows are stored (rows + window - 1 cells: a step's
// stores precede its attention, so a smaller ring would overwrite cells
// its first rows still see); and each tensor reads through the rows' end
// (or all of a ring).
std::expected<void, std::string> CheckCells(const SegmentRows& s, const CellKey& cells,
                                            std::uint32_t window) {
  const std::uint64_t end = std::uint64_t{s.past} + s.rows;
  if (!cells.ring && end > cells.capacity) {
    return Refused(
        std::format("positions through {} past a cache of {} cells", end, cells.capacity));
  }
  if (cells.ring && s.rows > cells.capacity) {
    return Refused(std::format("{} rows in a ring of {} cells", s.rows, cells.capacity));
  }
  if (cells.ring && end > cells.capacity && std::uint64_t{s.rows} + window > cells.capacity + 1) {
    return Refused(std::format("{} rows and a window of {} in a ring of {} cells", s.rows, window,
                               cells.capacity));
  }
  if (std::min<std::uint64_t>(cells.capacity, s.read_cells) < std::min(end, cells.capacity)) {
    return Refused(
        std::format("{} read cells do not cover positions through {}", s.read_cells, end));
  }
  return {};
}

// The instance's role leaves from its bindings and the bound tensors.
struct RoleFormat {
  format::TensorFormat format;
  std::vector<std::pair<std::string, format::PartRole>> parts;
};
std::expected<std::vector<std::optional<RoleFormat>>, std::string> RoleFormats(
    const Instance& in, std::span<const BoundTensor> tensors) {
  std::vector<std::optional<RoleFormat>> out(in.def->roles.size());
  for (const spec::WeightBinding& w : in.ref->weights) {
    const auto role = RoleIndex(*in.def, w.role);
    if (!role) return Refused(std::format("{} has no weight role {}", in.def->kind, w.role));
    const auto t = std::ranges::find_if(tensors, [&](const BoundTensor& b) {
      return b.name == w.tensor && b.part == w.part && b.experts == w.experts;
    });
    if (t == tensors.end()) {
      return Refused(std::format("{} role {} binds {}, which the artifact does not have",
                                 in.def->kind, w.role, w.tensor));
    }
    auto& slot = out[*role];
    if (!slot) {
      slot = RoleFormat{.format = t->format, .parts = {}};
    } else if (slot->format != t->format) {
      return Refused(
          std::format("{} role {}'s parts disagree on its format", in.def->kind, w.role));
    }
    slot->parts.emplace_back(w.tensor, w.part);
  }
  return out;
}

std::expected<void, std::string> CheckRole(const Instance& in, const RoleShape& r,
                                           const RoleFormat& f) {
  const WeightRole& role = in.def->roles[r.role];
  const auto what = [&] { return std::format("{} role {}", in.def->kind, role.name); };
  if (!format::Validate(f.format)) return Refused(what() + " has an invalid format");
  if (!role.encodings.empty() &&
      (std::ranges::find(role.encodings, f.format.encoding) == role.encodings.end() ||
       f.format.layout != format::Layout::kDense)) {
    return Refused(std::format("{} is {}, which it does not take", what(),
                               format::Info(f.format.encoding).name));
  }
  // Its parts: every part its format stores, each bound once.
  const auto parts = format::PartsOf(f.format);
  if (!parts) return Refused(what() + " has a format without a size");
  if (parts->size() != f.parts.size()) {
    return Refused(std::format("{} binds {} parts of {}", what(), f.parts.size(), parts->size()));
  }
  for (std::size_t i = 0; i < f.parts.size(); ++i) {
    const format::PartRole part = f.parts[i].second;
    if (!parts->Find(part)) return Refused(std::format("{} binds a part it has not", what()));
    for (std::size_t j = 0; j < i; ++j) {
      if (f.parts[j].second == part) return Refused(what() + " binds a part twice");
    }
  }
  const format::Shape& s = f.format.shape;
  bool same = s.rank == r.rank;
  for (std::size_t d = 0; same && d < r.rank; ++d) {
    same = r.ne.at(d) == 0 || r.ne.at(d) == s.ne.at(d);
  }
  if (!same) {
    return Refused(std::format("{} is [{}, {}, {}, {}], not the shape its parameters imply", what(),
                               s.ne[0], s.ne[1], s.ne[2], s.ne[3]));
  }
  return {};
}

model::Batching Tighter(model::Batching a, const model::Batching& b) {
  const auto least = [](std::uint32_t x, std::uint32_t y) {
    if (x == 0) return y;
    if (y == 0) return x;
    return std::min(x, y);
  };
  a.max_segments = std::min(a.max_segments, b.max_segments);
  a.max_segment_rows = least(a.max_segment_rows, b.max_segment_rows);
  a.max_step_rows = least(a.max_step_rows, b.max_step_rows);
  a.group_rows = least(a.group_rows, b.group_rows);
  a.group_segments = least(a.group_segments, b.group_segments);
  a.invariant_group_segments = least(a.invariant_group_segments, b.invariant_group_segments);
  a.joined_read_align = std::max(a.joined_read_align, b.joined_read_align);
  a.common_read = a.common_read || b.common_read;
  a.join_prefill = a.join_prefill && b.join_prefill;
  a.equal_prefill = a.equal_prefill || b.equal_prefill;
  a.mixed_heads = a.mixed_heads && b.mixed_heads;
  return a;
}

// Every instance of a stage, in ordinal order.
std::vector<const Instance*> All(const StageInstances& s) {
  std::vector<const Instance*> out;
  if (s.embedding) out.push_back(&*s.embedding);
  for (const auto& layer : s.layers) {
    for (const Instance& in : layer) out.push_back(&in);
  }
  if (s.final_norm) out.push_back(&*s.final_norm);
  if (s.output) out.push_back(&*s.output);
  return out;
}

}  // namespace

// --------------------------------------------------------------- instances

std::expected<Instance, std::string> MakeInstance(const spec::ComponentRef& ref,
                                                  const StageFacts& stage, Site site,
                                                  std::uint32_t layer, std::uint16_t index) {
  const ComponentDef* def = Find(ref.kind, ref.version);
  if (def == nullptr) {
    return Refused(std::format("no component {} version {} in the catalog", ref.kind, ref.version));
  }
  auto params = Params::Of(def->params, ref);
  if (!params) return std::unexpected(params.error());
  Instance in{.def = def,
              .ref = &ref,
              .params = std::move(*params),
              .stage = stage,
              .site = site,
              .layer = layer,
              .index = index};
  if (auto shapes = def->shapes(in); !shapes) {
    return Refused(std::format("{}: {}", ref.kind, shapes.error()));
  }
  return in;
}

std::expected<StageInstances, std::string> InstancesOf(const spec::ModelSpec& spec,
                                                       std::size_t stage) {
  if (stage >= spec.stages.size()) return Refused("no such stage");
  const spec::Stage& s = spec.stages[stage];
  if (s.stream.streams != 1 || s.stream.mixing.has_value()) {
    return Refused(std::format("stage {}: no stream-mixing component in the catalog", s.name));
  }
  if (!s.side.empty() || !s.heads.mtp.empty()) {
    return Refused(std::format("stage {}: no side-input or MTP component in the catalog", s.name));
  }
  if (s.stream.width == 0 || s.stream.width > kMaxStreamWidth) {
    return Refused(std::format("stage {}: a stream width of {}", s.name, s.stream.width));
  }
  if (s.layers.size() > kMaxStageLayers) {
    return Refused(std::format("stage {}: {} layers", s.name, s.layers.size()));
  }
  const StageFacts facts{.width = s.stream.width, .context = s.context, .softcap = s.heads.softcap};
  StageInstances out;
  std::uint32_t index = 1;
  const auto make = [&](const spec::ComponentRef& ref, Site site, std::uint32_t layer,
                        std::uint16_t at) -> std::expected<Instance, std::string> {
    auto in = MakeInstance(ref, facts, site, layer, at);
    if (!in) return Refused(std::format("{}: {}", Where(s, site, layer), in.error()));
    return in;
  };
  if (s.embedding) {
    auto in = make(*s.embedding, Site::kEmbedding, 0, 0);
    if (!in) return std::unexpected(in.error());
    out.embedding = std::move(*in);
  }
  for (std::uint32_t l = 0; l < s.layers.size(); ++l) {
    auto& layer = out.layers.emplace_back();
    for (const spec::ComponentRef& ref : s.layers[l].components) {
      if (index >= spec::kMaxComponents) return Refused("too many components");
      auto in = make(ref, Site::kLayer, l, static_cast<std::uint16_t>(index++));
      if (!in) return std::unexpected(in.error());
      layer.push_back(std::move(*in));
    }
  }
  if (s.heads.final_norm) {
    auto in = make(*s.heads.final_norm, Site::kFinalNorm, 0, static_cast<std::uint16_t>(index++));
    if (!in) return std::unexpected(in.error());
    out.final_norm = std::move(*in);
  }
  if (s.heads.output) {
    auto in = make(*s.heads.output, Site::kOutput, 0, static_cast<std::uint16_t>(index++));
    if (!in) return std::unexpected(in.error());
    out.output = std::move(*in);
  }
  return out;
}

// --------------------------------------------------------------- validation

std::expected<void, std::string> ValidateSpec(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors) {
  if (auto valid = spec::Validate(spec); !valid) return valid;
  std::vector<StageInstances> stages;
  for (std::size_t si = 0; si < spec.stages.size(); ++si) {
    const spec::Stage& s = spec.stages[si];
    auto instances = InstancesOf(spec, si);
    if (!instances) return std::unexpected(instances.error());
    if (s.artifact >= tensors.size()) {
      return Refused(std::format("stage {}: no artifact {} bound", s.name, s.artifact));
    }
    for (const spec::FeatureTap& tap : s.taps) {
      if (tap.point == spec::TapPoint::kStreams) {
        return Refused(
            std::format("stage {}: tap {} reads streams of a single stream", s.name, tap.name));
      }
    }
    for (const Instance* in : All(*instances)) {
      const std::string where = Where(s, in->site, in->layer);
      auto formats = RoleFormats(*in, tensors[s.artifact]);
      if (!formats) return Refused(std::format("{}: {}", where, formats.error()));
      const auto shapes = in->def->shapes(*in);
      std::vector<bool> used(in->def->roles.size(), false);
      for (const RoleShape& r : *shapes) {
        used[r.role] = true;
        const auto& f = (*formats)[r.role];
        if (!f) {
          return Refused(std::format("{}: {} role {} is unbound", where, in->def->kind,
                                     in->def->roles[r.role].name));
        }
        if (auto checked = CheckRole(*in, r, *f); !checked) {
          return Refused(std::format("{}: {}", where, checked.error()));
        }
      }
      for (std::size_t r = 0; r < used.size(); ++r) {
        if (!used[r] && (*formats)[r]) {
          return Refused(std::format("{}: {} binds role {}, which its parameters do not use", where,
                                     in->def->kind, in->def->roles[r].name));
        }
      }
    }
    stages.push_back(std::move(*instances));
  }
  // A read of another stage's state names one of its state tensors.
  for (const spec::Stage& s : spec.stages) {
    for (const spec::Read& read : s.reads) {
      if (read.kind != spec::ReadKind::kState) continue;
      const auto source = std::ranges::find(spec.stages, read.stage, &spec::Stage::name);
      bool found = false;
      if (source != spec.stages.end()) {
        const auto at = static_cast<std::size_t>(source - spec.stages.begin());
        for (const Instance* in : All(stages[at])) {
          if (in->def->state == nullptr) continue;
          const model::StateContract contract = in->def->state(*in);
          for (const auto& t : contract.tensors) found = found || t.name == read.name;
        }
      }
      if (!found) {
        return Refused(std::format("stage {} reads state {} of stage {}, which has none of it",
                                   s.name, read.name, read.stage));
      }
    }
  }
  return {};
}

std::expected<void, std::string> ValidateSpec(const spec::ModelSpec& spec,
                                              std::span<const BoundTensor> tensors) {
  const std::array<std::span<const BoundTensor>, 1> one{tensors};
  return ValidateSpec(spec, one);
}

// --------------------------------------------------------------- state

std::expected<std::vector<std::pair<std::uint32_t, model::StateContract>>, std::string>
StageContracts(const spec::ModelSpec& spec, std::size_t stage) {
  auto instances = InstancesOf(spec, stage);
  if (!instances) return std::unexpected(instances.error());
  std::vector<std::pair<std::uint32_t, model::StateContract>> out;
  for (std::uint32_t l = 0; l < instances->layers.size(); ++l) {
    model::StateContract layer;
    for (const Instance& in : instances->layers[l]) {
      if (in.def->state == nullptr) continue;
      model::StateContract contract = in.def->state(in);
      for (auto& t : contract.tensors) layer.tensors.push_back(std::move(t));
    }
    if (!layer.tensors.empty()) out.emplace_back(l, std::move(layer));
  }
  return out;
}

std::expected<model::StateLayoutDerived, std::string> StageLayout(const spec::ModelSpec& spec,
                                                                  std::size_t stage,
                                                                  std::uint32_t context,
                                                                  std::uint32_t max_rows,
                                                                  std::uint64_t tensor_align) {
  auto contracts = StageContracts(spec, stage);
  if (!contracts) return std::unexpected(contracts.error());
  return model::Layout(*contracts, context, max_rows, tensor_align);
}

std::expected<model::Batching, std::string> StageBatching(const spec::ModelSpec& spec,
                                                          std::size_t stage) {
  auto instances = InstancesOf(spec, stage);
  if (!instances) return std::unexpected(instances.error());
  // The primitive graph joins any rows of up to kMaxSegments owners: its
  // products join rows, attention runs per owner over its own cache.
  model::Batching out{.max_segments = static_cast<std::uint32_t>(kMaxSegments),
                      .join_prefill = true,
                      .mixed_heads = true};
  for (const Instance* in : All(*instances)) {
    if (in->def->batching != nullptr) out = Tighter(out, in->def->batching(*in));
  }
  return out;
}

// ------------------------------------------------------------ step context

StepContext::StepContext(ir::Builder& b, std::vector<SegmentRows> segments, bool greedy)
    : b_(b), segments_(std::move(segments)), greedy_(greedy) {
  for (const SegmentRows& s : segments_) {
    rows_ += s.rows;
    outputs_ += s.outputs;
  }
  segment_positions_.resize(segments_.size());
}

void StepContext::SetComponent(std::uint16_t component) { component_ = component; }

ir::TensorId StepContext::Input(InputKind kind, std::uint8_t segment,
                                const format::TensorFormat& format, const CellKey& cells,
                                std::uint64_t read, std::string_view name) {
  b_.SetComponent(kFrameworkComponent);
  if (segment == ir::kJoinedSegment) {
    b_.SetJoined();
  } else {
    b_.SetOwner(segment);
  }
  const ir::TensorId id = b_.Input(format, name);
  b_.SetJoined();
  b_.SetComponent(component_);
  inputs_.push_back({.tensor = id, .kind = kind, .segment = segment, .cells = cells, .read = read});
  return id;
}

ir::TensorId StepContext::Tokens() {
  if (!tokens_.valid()) {
    tokens_ = Input(InputKind::kTokens, ir::kJoinedSegment,
                    ir::Dense(format::Encoding::kI32, ir::ShapeOf({rows_})), {}, 0, "tokens");
  }
  return tokens_;
}

ir::TensorId StepContext::Positions() {
  if (!positions_.valid()) {
    positions_ = Input(InputKind::kPositions, ir::kJoinedSegment,
                       ir::Dense(format::Encoding::kI32, ir::ShapeOf({rows_})), {}, 0, "positions");
  }
  return positions_;
}

ir::TensorId StepContext::OutputRows() {
  if (!output_rows_.valid()) {
    output_rows_ =
        Input(InputKind::kOutputRows, ir::kJoinedSegment,
              ir::Dense(format::Encoding::kI32, ir::ShapeOf({outputs_})), {}, 0, "out_ids");
  }
  return output_rows_;
}

ir::TensorId StepContext::StoreCells(std::size_t segment, const CellKey& cells) {
  const auto key = std::pair{segment, cells};
  if (const auto at = store_cells_.find(key); at != store_cells_.end()) return at->second;
  const ir::TensorId id =
      Input(InputKind::kStoreCells, static_cast<std::uint8_t>(segment),
            ir::Dense(format::Encoding::kI64, ir::ShapeOf({segments_.at(segment).rows})), cells, 0,
            std::format("cells.{}.{}{}", segment, cells.capacity, cells.ring ? ".ring" : ""));
  store_cells_.emplace(key, id);
  return id;
}

std::uint64_t StepContext::ReadCells(std::size_t segment, const CellKey& cells) const {
  return std::min<std::uint64_t>(cells.capacity, segments_.at(segment).read_cells);
}

std::expected<ir::TensorId, std::string> StepContext::Mask(std::size_t segment,
                                                           const CellKey& cells,
                                                           std::uint32_t window) const {
  const auto at = masks_.find({segment, cells, window});
  if (at == masks_.end()) return Refused("a mask its component did not declare");
  return at->second;
}

void StepContext::DeclareMasks(std::span<const std::pair<CellKey, std::uint32_t>> masks) {
  // Causal first, then by window; cells within.
  std::vector<std::pair<CellKey, std::uint32_t>> order(masks.begin(), masks.end());
  std::ranges::sort(order, [](const auto& a, const auto& b) {
    return std::tie(a.second, a.first) < std::tie(b.second, b.first);
  });
  order.erase(std::ranges::unique(order).begin(), order.end());
  for (std::size_t s = 0; s < segments_.size(); ++s) {
    const SegmentRows& seg = segments_[s];
    for (const auto& [cells, window] : order) {
      if (masks_.contains({s, cells, window})) continue;
      const std::uint64_t read = ReadCells(s, cells);
      const auto key = std::tuple{s, cells, read};
      ir::TensorId held{};
      if (const auto at = cell_positions_.find(key); at != cell_positions_.end()) {
        held = at->second;
      } else {
        held = Input(InputKind::kCellPositions, static_cast<std::uint8_t>(s),
                     ir::Dense(format::Encoding::kI32, ir::ShapeOf({read})), cells, read,
                     std::format("held.{}.{}{}", s, cells.capacity, cells.ring ? ".ring" : ""));
        cell_positions_.emplace(key, held);
      }
      b_.SetComponent(kFrameworkComponent);
      if (!segment_positions_[s].valid()) {
        const ir::TensorId all = Positions();
        b_.SetComponent(kFrameworkComponent);
        const format::Shape shape = ir::ShapeOf({seg.rows});
        segment_positions_[s] =
            b_.View(all, shape, {4, 4ULL * seg.rows, 4ULL * seg.rows, 4ULL * seg.rows},
                    std::uint64_t{seg.first_row} * 4);
      }
      b_.SetOwner(static_cast<std::uint8_t>(s));
      const ir::MaskParams params{
          .kind = window == 0 ? ir::MaskKind::kCausal : ir::MaskKind::kWindow, .window = window};
      const ir::TensorId mask =
          b_.MaskBuild(segment_positions_[s], held, Pad(seg.rows, kMaskRowAlign), params,
                       format::Encoding::kF16);
      b_.SetJoined();
      b_.SetComponent(component_);
      masks_.emplace(std::tuple{s, cells, window}, mask);
    }
  }
}

// --------------------------------------------------------------- the step

namespace {

// The cells of a layer's contract tensor in the stage layout.
std::optional<std::size_t> ItemOf(const model::StateLayoutDerived& layout, std::uint32_t layer,
                                  std::uint32_t tensor) {
  for (std::size_t i = 0; i < layout.items.size(); ++i) {
    if (layout.items[i].layer == layer && layout.items[i].tensor == tensor) return i;
  }
  return std::nullopt;
}

CellKey KeyOf(const model::StateTensorContract& c, std::uint64_t cells) {
  return {.capacity = cells, .ring = c.cells == model::CellRule::kRing};
}

}  // namespace

std::optional<std::pair<std::size_t, std::size_t>> NarrowPoint(const StageInstances& instances) {
  if (!instances.output || (instances.final_norm && !instances.final_norm->def->row_local) ||
      !instances.output->def->row_local) {
    return std::nullopt;
  }
  for (std::size_t l = instances.layers.size(); l-- > 0;) {
    for (std::size_t c = instances.layers[l].size(); c-- > 0;) {
      const ComponentDef& def = *instances.layers[l][c].def;
      if (def.row_local) continue;
      if (def.narrows) return std::pair{l, c};
      return std::nullopt;  // the last component that mixes rows cannot narrow
    }
  }
  return std::nullopt;
}

std::expected<StepGraph, std::string> BuildStep(const spec::ModelSpec& spec, std::uint16_t stage,
                                                const planner::StepShape& shape,
                                                const Bindings& bindings) {
  if (shape.stage != stage) return Refused("a shape of another stage");
  if (auto checked = CheckShape(shape); !checked) return std::unexpected(checked.error());
  auto instances = InstancesOf(spec, stage);
  if (!instances) return std::unexpected(instances.error());
  if (!instances->embedding) return Refused("a stage without an embedding");
  StepBuilder sb(stage, shape);
  StepContext& ctx = sb.ctx();
  const bool head = ctx.outputs() > 0;
  if (head && !instances->output) return Refused("outputs from a stage without an output head");

  // Each layer's stateful instances: their contracts, first tensor and
  // layout items.
  struct Stateful {
    model::StateContract contract;
    std::uint16_t first = 0;
    std::vector<std::uint64_t> cells;
    std::vector<std::size_t> items;
  };
  std::vector<std::vector<std::optional<Stateful>>> stateful(instances->layers.size());
  std::vector<std::pair<CellKey, std::uint32_t>> masks;
  for (std::size_t l = 0; l < instances->layers.size(); ++l) {
    std::uint16_t first = 0;
    for (const Instance& in : instances->layers[l]) {
      auto& slot = stateful[l].emplace_back();
      if (in.def->state == nullptr) continue;
      if (bindings.state == nullptr) return Refused("a stateful stage without its state layout");
      Stateful st{.contract = in.def->state(in), .first = first, .cells = {}, .items = {}};
      for (std::size_t t = 0; t < st.contract.tensors.size(); ++t) {
        const auto item = ItemOf(*bindings.state, in.layer, static_cast<std::uint32_t>(first + t));
        if (!item || bindings.state->items[*item].contract != st.contract.tensors[t]) {
          return Refused(std::format("layer {}: the state layout is not this spec's", in.layer));
        }
        st.items.push_back(*item);
        st.cells.push_back(bindings.state->items[*item].cells);
        const CellKey key = KeyOf(st.contract.tensors[t], st.cells.back());
        for (const SegmentRows& seg : ctx.segments()) {
          if (auto c = CheckCells(seg, key, st.contract.tensors[t].window_or_ratio); !c) {
            return Refused(std::format("layer {} state {}: {}", in.layer,
                                       st.contract.tensors[t].name, c.error()));
          }
        }
      }
      if (in.def->masks != nullptr) {
        for (const auto& [tensor, window] : in.def->masks(in)) {
          masks.emplace_back(KeyOf(st.contract.tensors.at(tensor), st.cells.at(tensor)), window);
        }
      }
      first = static_cast<std::uint16_t>(first + st.contract.tensors.size());
      slot = std::move(st);
    }
  }
  ctx.DeclareMasks(masks);

  const auto emit = [&](const Instance& in, ir::TensorId stream,
                        std::span<const ir::TensorId> state, std::size_t tensors, bool needed,
                        bool narrow = false,
                        bool narrowed = false) -> std::expected<FragmentOut, std::string> {
    auto formats = RoleFormats(in, bindings.tensors);
    if (!formats) return std::unexpected(formats.error());
    const auto shapes = in.def->shapes(in);
    std::vector<ir::TensorId> leaves(in.def->roles.size());
    std::vector<bool> used(in.def->roles.size(), false);
    for (const RoleShape& r : *shapes) {
      const auto& f = (*formats)[r.role];
      if (!f) return Refused(std::format("role {} is unbound", in.def->roles[r.role].name));
      if (auto checked = CheckRole(in, r, *f); !checked) return std::unexpected(checked.error());
      used[r.role] = true;
    }
    sb.Begin(in);
    Fragment f{.b = sb.b(),
               .instance = in,
               .step = ctx,
               .stream = stream,
               .weight = [&](std::uint16_t role) -> ir::TensorId {
                 base::Check(role < used.size() && used[role], "a fragment uses its roles");
                 if (!leaves[role].valid()) {
                   const auto& rf = *(*formats)[role];
                   leaves[role] = sb.Weight(in, role, rf.format, rf.parts);
                 }
                 return leaves[role];
               },
               .state = state,
               .state_tensors = tensors,
               .stream_needed = needed,
               .narrow = narrow,
               .narrowed = narrowed};
    auto out = in.def->emit(f);
    if (!out) return std::unexpected(out.error());
    sb.b().SetJoined();
    return out;
  };

  auto embedded = emit(*instances->embedding, {}, {}, 0, true);
  if (!embedded) return Refused(std::format("embedding: {}", embedded.error()));
  ir::TensorId stream = embedded->stream;

  // Which components' output streams something reads: from the head back.
  std::vector<std::vector<bool>> needed(instances->layers.size());
  bool need = head;
  for (std::size_t l = instances->layers.size(); l-- > 0;) {
    needed[l].resize(instances->layers[l].size());
    for (std::size_t c = instances->layers[l].size(); c-- > 0;) {
      needed[l][c] = need;
      if (stateful[l][c]) need = true;  // its writes read its input
    }
  }
  // The head reads only its output rows: the stage's last component that
  // mixes rows narrows its output to them when it can (the Gemma builders'
  // narrow_final), and what follows it, row-local, runs at their width.
  const std::optional<std::pair<std::size_t, std::size_t>> narrow_at =
      head ? NarrowPoint(*instances) : std::nullopt;
  bool narrowed = false;
  for (std::size_t l = 0; l < instances->layers.size(); ++l) {
    for (std::size_t c = 0; c < instances->layers[l].size(); ++c) {
      const Instance& in = instances->layers[l][c];
      const auto& st = stateful[l][c];
      if (!needed[l][c] && !st) continue;
      std::vector<ir::TensorId> state;
      if (st) state = sb.State(in, st->contract, st->first, st->cells, st->items);
      const bool narrow = narrow_at == std::pair{l, c};
      auto out = emit(in, stream, state, st ? st->contract.tensors.size() : 0, needed[l][c], narrow,
                      narrowed);
      if (!out) {
        return Refused(std::format("layer {} {}: {}", l, in.def->kind, out.error()));
      }
      stream = out->stream;
      narrowed = narrowed || narrow;
    }
  }
  ir::TensorId logits{};
  ir::TensorId greedy{};
  if (head) {
    if (instances->final_norm) {
      auto out = emit(*instances->final_norm, stream, {}, 0, true, false, narrowed);
      if (!out) return Refused(std::format("final norm: {}", out.error()));
      stream = out->stream;
    }
    auto out = emit(*instances->output, stream, {}, 0, true, false, narrowed);
    if (!out) return Refused(std::format("output: {}", out.error()));
    logits = out->logits;
    greedy = out->greedy;
    sb.b().Output(logits);
    if (greedy.valid()) sb.b().Output(greedy);
  }
  return std::move(sb).Finish(logits, greedy);
}

std::expected<ComponentGraph, std::string> BuildComponent(
    const Instance& in, const planner::StepShape& shape,
    std::span<const format::TensorFormat> formats, std::span<const std::uint64_t> cells) {
  if (auto checked = CheckShape(shape); !checked) return std::unexpected(checked.error());
  const auto shapes = in.def->shapes(in);
  if (!shapes) return std::unexpected(shapes.error());
  if (formats.size() != in.def->roles.size()) return Refused("a format for every role");
  std::vector<bool> used(in.def->roles.size(), false);
  for (const RoleShape& r : *shapes) {
    const RoleFormat f{
        .format = formats[r.role],
        .parts = {{std::format("{}", in.def->roles[r.role].name), format::PartRole::kData}}};
    if (auto checked = CheckRole(in, r, f); !checked) return std::unexpected(checked.error());
    used[r.role] = true;
  }
  StepBuilder sb(shape.stage, shape);
  StepContext& ctx = sb.ctx();
  model::StateContract contract;
  if (in.def->state != nullptr) contract = in.def->state(in);
  if (cells.size() != contract.tensors.size()) return Refused("cells for every state tensor");
  std::vector<std::pair<CellKey, std::uint32_t>> masks;
  for (std::size_t t = 0; t < contract.tensors.size(); ++t) {
    for (const SegmentRows& seg : ctx.segments()) {
      if (auto c = CheckCells(seg, KeyOf(contract.tensors[t], cells[t]),
                              contract.tensors[t].window_or_ratio);
          !c) {
        return std::unexpected(c.error());
      }
    }
  }
  if (in.def->masks != nullptr) {
    for (const auto& [tensor, window] : in.def->masks(in)) {
      masks.emplace_back(KeyOf(contract.tensors.at(tensor), cells[tensor]), window);
    }
  }
  ctx.DeclareMasks(masks);
  ir::TensorId stream_in{};
  ir::TensorId stream_out{};
  std::vector<ir::TensorId> weights(in.def->roles.size());
  sb.Begin(in);
  if (in.site != Site::kEmbedding) {
    const format::TensorFormat f =
        ir::Dense(format::Encoding::kF32, ir::ShapeOf({in.stage.width, ctx.rows()}));
    sb.b().SetComponent(kFrameworkComponent);
    stream_in = sb.b().Input(f, "stream");
    sb.b().SetComponent(in.index);
  }
  std::vector<ir::TensorId> state;
  if (!contract.tensors.empty()) state = sb.State(in, contract, 0, cells, {});
  Fragment f{.b = sb.b(),
             .instance = in,
             .step = ctx,
             .stream = stream_in,
             .weight = [&](std::uint16_t role) -> ir::TensorId {
               base::Check(role < used.size() && used[role], "a fragment uses its roles");
               if (!weights[role].valid()) {
                 weights[role] =
                     sb.Weight(in, role, formats[role],
                               {{std::string(in.def->roles[role].name), format::PartRole::kData}});
               }
               return weights[role];
             },
             .state = state,
             .state_tensors = contract.tensors.size(),
             // A stateful component's state-only step: no outputs.
             .stream_needed = contract.tensors.empty() || ctx.outputs() > 0};
  auto emitted = in.def->emit(f);
  if (!emitted) return std::unexpected(emitted.error());
  sb.b().SetJoined();
  if (emitted->logits.valid()) {
    sb.b().Output(emitted->logits);
    if (emitted->greedy.valid()) sb.b().Output(emitted->greedy);
  } else if (f.stream_needed) {
    stream_out = emitted->stream;
    sb.b().Output(stream_out);
  }
  auto step = std::move(sb).Finish(emitted->logits, emitted->greedy);
  if (!step) return std::unexpected(step.error());
  if (stream_in.valid()) {
    step->inputs.insert(step->inputs.begin(),
                        StepInput{.tensor = stream_in, .kind = InputKind::kStream});
  }
  return ComponentGraph{.step = std::move(*step),
                        .stream_in = stream_in,
                        .stream_out = stream_out,
                        .weights = std::move(weights)};
}

std::expected<std::vector<std::int64_t>, std::string> InputValues(
    const StepInput& input, const planner::StepShape& shape, std::span<const std::int32_t> tokens) {
  std::vector<std::int64_t> out;
  std::uint32_t rows = 0;
  for (const planner::Segment& s : shape.segments) rows += s.rows;
  switch (input.kind) {
    case InputKind::kStream:
      return Refused("a stream's values are the caller's");
    case InputKind::kTokens:
      if (tokens.size() != rows) return Refused("a token per row");
      out.assign(tokens.begin(), tokens.end());
      return out;
    case InputKind::kPositions:
      for (const planner::Segment& s : shape.segments) {
        for (std::uint32_t i = 0; i < s.rows; ++i) out.push_back(std::int64_t{s.past} + i);
      }
      return out;
    case InputKind::kOutputRows: {
      std::int64_t first = 0;
      for (const planner::Segment& s : shape.segments) {
        for (std::uint32_t i = s.rows - s.outputs; i < s.rows; ++i) out.push_back(first + i);
        first += s.rows;
      }
      return out;
    }
    case InputKind::kStoreCells:
    case InputKind::kCellPositions:
      break;
  }
  if (input.segment >= shape.segments.size() || input.cells.capacity == 0) {
    return Refused("an input of no segment of the shape");
  }
  const planner::Segment& s = shape.segments[input.segment];
  const std::uint64_t cap = input.cells.capacity;
  if (input.kind == InputKind::kStoreCells) {
    for (std::uint32_t i = 0; i < s.rows; ++i) {
      const std::uint64_t p = std::uint64_t{s.past} + i;
      out.push_back(static_cast<std::int64_t>(input.cells.ring ? p % cap : p));
    }
    return out;
  }
  const std::uint64_t end = std::uint64_t{s.past} + s.rows;
  for (std::uint64_t c = 0; c < input.read; ++c) {
    if (c >= end) {
      out.push_back(-1);
    } else if (input.cells.ring) {
      out.push_back(static_cast<std::int64_t>(c + (((end - 1 - c) / cap) * cap)));
    } else {
      out.push_back(static_cast<std::int64_t>(c));
    }
  }
  return out;
}

}  // namespace llmp::components
