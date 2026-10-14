// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "components/step.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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

// A compressor input of every segment's rows (InputKind::kPositionPhases).
constexpr std::size_t kJoinedCompressor = ~std::size_t{0};

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
    case Site::kSide:
      return std::format("stage {} side input", stage.name);
    case Site::kMixing:
      return std::format("stage {} stream mixing", stage.name);
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

// The stage of a spec by name.
std::optional<std::uint16_t> StageIndex(const spec::ModelSpec& spec, std::string_view name) {
  const auto at = std::ranges::find(spec.stages, name, &spec::Stage::name);
  if (at == spec.stages.end()) return std::nullopt;
  return static_cast<std::uint16_t>(at - spec.stages.begin());
}

// The source a node stores into, if it is a store: a cache store's cache,
// a fused RoPE store's.
std::optional<std::size_t> StoreTarget(const ir::Node& n) {
  if (n.key == ir::Primitive(ir::Op::kCacheStore)) return 0;
  if (n.key == ir::Fused(ir::FusedKind::kRopeStore)) return 2;
  return std::nullopt;
}

// Which state leaves the step's nodes read and which its stores write.
void MarkAccess(StepGraph& step) {
  const ir::Graph& g = step.graph;
  for (const ir::Node& n : g.nodes()) {
    const auto target = StoreTarget(n);
    for (std::size_t j = 0; j < n.nsrc; ++j) {
      if (!n.src[j].valid()) continue;
      const ir::TensorId root = g.root(n.src[j]);
      const auto st = std::ranges::find(step.state, root, &StepState::tensor);
      if (st == step.state.end()) continue;
      if (target == j) {
        st->written = true;
      } else {
        st->read = true;
      }
    }
  }
}

// ------------------------------------------------------------ the builder

// Joins a stage's fragments into one graph: the shared step context, the
// weight leaves (once per bound tensor) and the state leaves.
class StepBuilder {
 public:
  StepBuilder(std::uint16_t stage, const planner::StepShape& shape)
      : stage_(stage), ctx_(b_, Segments(shape), Greedy(shape)) {
    ctx_.SetPhase(shape.phase);
  }

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
  std::uint16_t stage() const { return stage_; }
  void Begin(const Instance& in) {
    b_.SetComponent(in.index);
    ctx_.SetComponent(in.index);
  }

  // A weight leaf of `format` for (instance, role), shared by every role
  // bound to the same tensor of the same stage's artifact (`key`: its
  // first part's name; empty: never shared). `owner`, for a tensor of
  // another stage's artifact, is that stage's WeightRef of it.
  ir::TensorId Weight(const Instance& in, std::uint16_t role, const format::TensorFormat& format,
                      std::vector<std::pair<std::string, format::PartRole>> parts,
                      std::optional<ir::WeightRef> owner = std::nullopt, bool experts = false) {
    const ir::WeightRef ref = owner.value_or(ir::WeightRef{.stage = stage_,
                                                           .component = in.index,
                                                           .role = role,
                                                           .layer = in.layer,
                                                           .expert_array = experts});
    const std::string name =
        parts.empty() ? std::string{} : std::format("{}/{}", ref.stage, parts.front().first);
    if (!name.empty()) {
      if (const auto at = by_name_.find(name); at != by_name_.end()) return at->second;
    }
    const std::string label = parts.empty()
                                  ? std::format("{}.{}", in.def->kind, in.def->roles[role].name)
                                  : parts.front().first;
    const ir::TensorId id = b_.Weight(format, ref, label);
    if (!name.empty()) by_name_.emplace(name, id);
    weights_.push_back({.tensor = id, .ref = ref, .parts = std::move(parts)});
    return id;
  }

  // A state leaf of `contract` tensor `t` (its index `index` in its layer's
  // contract) of `stage`'s layer `layer`, for segment `s`.
  ir::TensorId StateLeaf(std::uint16_t stage, std::uint32_t layer, std::uint16_t index,
                         const model::StateTensorContract& c, std::size_t s, std::uint64_t cells,
                         std::size_t item, bool tap, std::uint16_t source = 0) {
    const ir::StateRef ref{
        .stage = stage, .tensor = index, .layer = layer, .segment = static_cast<std::uint8_t>(s)};
    const ir::TensorId id =
        b_.State(CellFormat(c, cells), ref,
                 stage == stage_ ? std::format("blk.{}.{}.{}", layer, c.name, s)
                                 : std::format("stage{}.blk.{}.{}.{}", stage, layer, c.name, s));
    state_.push_back({.tensor = id,
                      .ref = ref,
                      .item = item,
                      .borrowed = c.borrowed_from_target,
                      .tap = tap,
                      .source = c.borrowed_from_target ? source : stage});
    return id;
  }

  // State leaves for an instance's contract tensors from `first` in its
  // layer's contract: by segment, then tensor.
  std::vector<ir::TensorId> State(const Instance& in, const model::StateContract& contract,
                                  std::uint16_t first, std::span<const std::uint64_t> cells,
                                  std::span<const std::size_t> items,
                                  std::span<const std::uint16_t> sources = {}) {
    std::vector<ir::TensorId> out;
    for (std::size_t s = 0; s < ctx_.segments().size(); ++s) {
      for (std::size_t t = 0; t < contract.tensors.size(); ++t) {
        out.push_back(StateLeaf(stage_, in.layer, static_cast<std::uint16_t>(first + t),
                                contract.tensors[t], s, cells[t], items.empty() ? 0 : items[t],
                                false, sources.empty() ? 0 : sources[t]));
      }
    }
    return out;
  }

  std::expected<StepGraph, std::string> Finish(ir::TensorId logits, ir::TensorId greedy,
                                               ir::TensorId features) && {
    auto inputs = ctx_.TakeInputs();
    auto saves = ctx_.TakeSaves();
    for (const CommitSave& save : saves) b_.Output(save.tensor);
    auto graph = std::move(b_).Finish();
    if (!graph) {
      return Refused(std::format("the step's graph does not verify at node {}: {} ({})",
                                 graph.error().node, graph.error().detail,
                                 ir::GraphErrorName(graph.error().error)));
    }
    OwnerProducts owners;
    const auto requests = ctx_.TakeOwnerProducts();
    if (!requests.empty()) {
      auto proved = ProveOwnerProducts(*graph, inputs, ctx_.segments(), requests);
      if (!proved) return std::unexpected(proved.error());
      owners = std::move(*proved);
    }
    StepGraph step{.graph = std::move(*graph),
                   .inputs = std::move(inputs),
                   .weights = std::move(weights_),
                   .state = std::move(state_),
                   .logits = logits,
                   .greedy = greedy,
                   .features = features,
                   .saves = std::move(saves),
                   .owner_products = std::move(owners)};
    MarkAccess(step);
    if (auto ok = CheckStateWrites(step, stage_); !ok) return std::unexpected(ok.error());
    return step;
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
  if (shape.read_align == 0 || shape.read_align % 256 != 0) {
    return Refused(std::format("a read alignment of {}", shape.read_align));
  }
  const bool greedy = shape.segments.front().greedy;
  std::set<std::uint8_t> slots;
  for (const planner::Segment& s : shape.segments) {
    if (s.rows == 0 || s.outputs > s.rows || !slots.insert(s.slot).second) {
      return Refused("an empty or repeated segment, or more outputs than rows");
    }
    if (s.greedy != greedy || (greedy && s.outputs == 0)) {
      return Refused("greedy segments are all of them, each with an output");
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
// (or all of a ring). A frozen read (another stage's tensor) holds the
// positions before the segment's: it reads through them, and a ring of
// it holds every row's window but the row's own position.
std::expected<void, std::string> CheckCells(const SegmentRows& s, const CellKey& cells,
                                            std::uint32_t window) {
  // Fixed state and block keys hold no positions to cover; a per-row
  // tensor holds the segment's rows after its first; a recent ring only
  // the rows it keeps (its components store those, and check what a
  // window over it needs).
  if (cells.fixed || cells.block != 0 || cells.recent) return {};
  if (cells.per_row) {
    if (std::uint64_t{cells.first_row} + s.rows > cells.capacity) {
      return Refused(
          std::format("{} rows past a per-row tensor of {} cells", s.rows, cells.capacity));
    }
    return {};
  }
  const std::uint64_t end = std::uint64_t{s.past} + (cells.frozen ? 0 : s.rows);
  if (cells.frozen) {
    if (!cells.ring && end > cells.capacity) {
      return Refused(
          std::format("positions through {} past a cache of {} cells", end, cells.capacity));
    }
    // A ring that has wrapped holds its last `capacity` positions: every
    // row's window but its own position must be among them.
    if (cells.ring && end > cells.capacity && std::uint64_t{window} > cells.capacity + 1) {
      return Refused(std::format("a window of {} in a ring of {} cells", window, cells.capacity));
    }
  } else {
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
  }
  if (std::min<std::uint64_t>(cells.capacity, s.read_cells) < std::min(end, cells.capacity)) {
    return Refused(
        std::format("{} read cells do not cover positions through {}", s.read_cells, end));
  }
  return {};
}

// Where a stage's bindings find a tensor: its own artifact's tensors, or
// for a binding that names another stage, that stage's artifact's.
struct TensorTables {
  std::span<const BoundTensor> own{};
  std::map<std::string, std::span<const BoundTensor>, std::less<>> stages{};
};

// The instance's role leaves from its bindings and the bound tensors.
struct RoleFormat {
  format::TensorFormat format;
  std::vector<std::pair<std::string, format::PartRole>> parts;
  std::string stage;     // the binding's (empty: its own stage's)
  bool experts = false;  // an expert array's (its WeightRef's expert_array)
};
std::expected<std::vector<std::optional<RoleFormat>>, std::string> RoleFormats(
    const Instance& in, const TensorTables& tables) {
  std::vector<std::optional<RoleFormat>> out(in.def->roles.size());
  for (const spec::WeightBinding& w : in.ref->weights) {
    const auto role = RoleIndex(*in.def, w.role);
    if (!role) return Refused(std::format("{} has no weight role {}", in.def->kind, w.role));
    std::span<const BoundTensor> tensors = tables.own;
    if (!w.stage.empty()) {
      const auto at = tables.stages.find(w.stage);
      if (at == tables.stages.end()) {
        return Refused(std::format("{} role {} binds {} of stage {}, whose tensors are not bound",
                                   in.def->kind, w.role, w.tensor, w.stage));
      }
      tensors = at->second;
    }
    const auto t = std::ranges::find_if(tensors, [&](const BoundTensor& b) {
      return b.name == w.tensor && b.part == w.part && b.experts == w.experts;
    });
    if (t == tensors.end()) {
      return Refused(std::format("{} role {} binds {}, which the artifact does not have",
                                 in.def->kind, w.role, w.tensor));
    }
    auto& slot = out[*role];
    if (!slot) {
      slot = RoleFormat{.format = t->format, .parts = {}, .stage = w.stage, .experts = w.experts};
    } else if (slot->format != t->format || slot->stage != w.stage) {
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
  if (in.def->kind == "head.linear" && role.name == "weight" &&
      static_cast<std::uint64_t>(in.params.Int("rows")) > s.ne[1]) {
    return Refused("head.linear rows exceed the weight's vocabulary");
  }
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

std::expected<std::vector<std::optional<RoleFormat>>, std::string> CheckedRoleFormats(
    const Instance& in, const TensorTables& tables) {
  auto formats = RoleFormats(in, tables);
  if (!formats) return std::unexpected(formats.error());
  const auto shapes = in.def->shapes(in);
  if (!shapes) return std::unexpected(shapes.error());
  std::vector<bool> used(in.def->roles.size(), false);
  for (const RoleShape& r : *shapes) {
    used[r.role] = true;
    const auto& f = (*formats)[r.role];
    if (!f)
      return Refused(
          std::format("{} role {} is unbound", in.def->kind, in.def->roles[r.role].name));
    if (auto checked = CheckRole(in, r, *f); !checked) return std::unexpected(checked.error());
  }
  for (std::size_t r = 0; r < used.size(); ++r)
    if (!used[r] && (*formats)[r])
      return Refused(std::format("{} binds role {}, which its parameters do not use", in.def->kind,
                                 in.def->roles[r].name));
  return formats;
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
  for (const Instance& in : s.side) out.push_back(&in);
  if (s.mixing) out.push_back(&*s.mixing);
  for (const auto& layer : s.layers) {
    for (const Instance& in : layer) out.push_back(&in);
  }
  if (s.final_norm) out.push_back(&*s.final_norm);
  if (s.output) out.push_back(&*s.output);
  return out;
}

// A stage's contract tensors with where they sit: (layer, index in the
// layer's contract, the tensor).
struct ContractTensor {
  std::uint32_t layer = 0;
  std::uint16_t index = 0;
  model::StateTensorContract contract;
  const Instance* instance = nullptr;  // its component's
};
std::vector<ContractTensor> ContractTensors(const StageInstances& s) {
  std::vector<ContractTensor> out;
  for (std::uint32_t l = 0; l < s.layers.size(); ++l) {
    std::uint16_t index = 0;
    const auto add = [&](const Instance& in) {
      if (in.def->state == nullptr) return;
      for (auto& t : in.def->state(in).tensors) {
        out.push_back({.layer = l, .index = index++, .contract = std::move(t), .instance = &in});
      }
    };
    if (l == 0 && s.embedding) add(*s.embedding);
    for (const Instance& in : s.layers[l]) add(in);
  }
  return out;
}

// The tables of a spec's stages' bound tensors (`tensors[artifact]`).
TensorTables TablesOf(const spec::ModelSpec& spec, std::size_t stage,
                      std::span<const std::span<const BoundTensor>> tensors,
                      std::span<const BoundTensor> own) {
  TensorTables t{.own = own, .stages = {}};
  for (std::size_t i = 0; i < spec.stages.size(); ++i) {
    if (i == stage || spec.stages[i].artifact >= tensors.size()) continue;
    t.stages.emplace(spec.stages[i].name, tensors[spec.stages[i].artifact]);
  }
  return t;
}

// The stage a drafter's read of `kind` and `name` reads (each name of one
// stage: CheckReads refuses another).
std::optional<std::uint16_t> ReadSource(const spec::ModelSpec& spec, const spec::Stage& s,
                                        spec::ReadKind kind, std::string_view name) {
  for (const spec::Read& r : s.reads) {
    if (r.kind == kind && r.name == name) return StageIndex(spec, r.stage);
  }
  return std::nullopt;
}

// The bytes of one F32 cell of a tap at `point` of `source`: its width, or
// at its streams its streams times its width.
std::uint64_t TapCellBytes(const spec::Stage& source, spec::TapPoint point) {
  // A layer's output, its streams' mean (the stage's width) or its streams.
  const std::uint64_t streams = point == spec::TapPoint::kStreams ? source.stream.streams : 1;
  return streams * source.stream.width * 4;
}

// The reads of a stage linked to its state contracts (ValidateSpec's rule).
std::expected<void, std::string> CheckReads(const spec::ModelSpec& spec,
                                            std::span<const StageInstances> stages,
                                            std::size_t si) {
  const spec::Stage& s = spec.stages[si];
  const std::vector<ContractTensor> mine = ContractTensors(stages[si]);
  const bool reads = std::ranges::any_of(mine, [](const ContractTensor& t) {
    return t.contract.borrowed_from_target || t.contract.target_writes;
  });
  if ((reads || !s.reads.empty()) && s.role != spec::StageRole::kDrafter) {
    return Refused(std::format("stage {} reads another stage, but is no drafter", s.name));
  }
  if (!reads && s.reads.empty()) return {};
  // A drafter reads stages before it: its target (the spec's first stage,
  // of the target role), or in a cascade the drafter before it.
  if (si == 0 || spec.stages.front().role != spec::StageRole::kTarget) {
    return Refused(std::format("stage {} reads a target the spec does not begin with", s.name));
  }
  for (const spec::Read& r : s.reads) {
    const auto source = StageIndex(spec, r.stage);
    if (!source || *source >= si) {
      return Refused(
          std::format("stage {} reads stage {}, which is not before it", s.name, r.stage));
    }
    // A name of one stage: a contract tensor names what it reads, not where.
    for (const spec::Read& o : s.reads) {
      if (&o != &r && o.kind == r.kind && o.name == r.name) {
        return Refused(std::format("stage {} reads {} {} of two stages", s.name,
                                   spec::ReadKindName(r.kind), r.name));
      }
    }
  }
  for (const ContractTensor& t : mine) {
    const model::StateTensorContract& c = t.contract;
    const std::string at = std::format("stage {} layer {} state {}", s.name, t.layer, c.name);
    if (c.borrowed_from_target) {
      const auto source = ReadSource(spec, s, spec::ReadKind::kState, c.name);
      if (!source) return Refused(at + ": borrows another stage's state without reading it");
      const std::vector<ContractTensor> theirs = ContractTensors(stages[*source]);
      const auto found = std::ranges::find_if(theirs, [&](const ContractTensor& x) {
        return x.layer == c.target_layer && x.contract.name == c.name;
      });
      if (found == theirs.end()) {
        return Refused(std::format("{}: stage {} has no {} in layer {}", at,
                                   spec.stages[*source].name, c.name, c.target_layer));
      }
      const model::StateTensorContract& o = found->contract;
      if (o.borrowed_from_target || o.kind != c.kind || o.encoding != c.encoding ||
          o.cell_bytes != c.cell_bytes || o.cells != c.cells ||
          o.window_or_ratio != c.window_or_ratio) {
        return Refused(at + ": not the tensor of that name of the stage it reads");
      }
    }
    // State its target writes by this drafter's injection needs no tap of
    // its name (the injection reads the taps its embedding names).
    if (c.target_writes && t.instance != nullptr && t.instance->def->inject != nullptr &&
        t.instance->site == Site::kLayer) {
      if (!stages[si].embedding || stages[si].embedding->def->inject == nullptr) {
        return Refused(at + ": injected state of a drafter whose embedding injects nothing");
      }
      continue;
    }
    if (c.target_writes) {
      const auto source = ReadSource(spec, s, spec::ReadKind::kTap, c.name);
      if (!source) return Refused(at + ": another stage writes it without a tap it reads");
      const spec::Stage& from = spec.stages[*source];
      const auto tap = std::ranges::find(from.taps, c.name, &spec::FeatureTap::name);
      if (tap == from.taps.end() || c.encoding != format::Encoding::kF32 ||
          c.cell_bytes != TapCellBytes(from, tap->point)) {
        return Refused(std::format(
            "{}: not a tap of stage {} in F32 of its width (or its streams' at its streams)", at,
            from.name));
      }
    }
  }
  // The taps an injection reads: each a tap read of one stage, which a
  // layer's tap point gives (not the final norm's: an injection runs
  // whether or not the target's step has a head).
  std::vector<std::string> injected;
  if (stages[si].embedding && stages[si].embedding->def->inject != nullptr) {
    const Instance& e = *stages[si].embedding;
    if (e.def->inject_reads == nullptr || e.def->inject_rows == nullptr) {
      return Refused(std::format("stage {}: an injection without its reads or rows", s.name));
    }
    injected = e.def->inject_reads(e);
    std::optional<std::string> from;
    for (const std::string& name : injected) {
      const auto source = ReadSource(spec, s, spec::ReadKind::kTap, name);
      if (!source) {
        return Refused(
            std::format("stage {} injects from tap {}, which it does not read", s.name, name));
      }
      const spec::Stage& target = spec.stages[*source];
      if (from && *from != target.name) {
        return Refused(std::format("stage {} injects from two stages' taps", s.name));
      }
      from = target.name;
      const auto tap = std::ranges::find(target.taps, name, &spec::FeatureTap::name);
      if (tap == target.taps.end() || tap->point == spec::TapPoint::kPostFinalNorm) {
        return Refused(std::format("stage {} injects from tap {}, no layer's tap of stage {}",
                                   s.name, name, target.name));
      }
    }
    const bool writes = std::ranges::any_of(mine, [](const ContractTensor& t) {
      return t.contract.target_writes && t.instance != nullptr &&
             t.instance->def->inject != nullptr;
    });
    if (!writes) return Refused(std::format("stage {} injects into no state", s.name));
  }
  // Every read used: a state read by a borrowed tensor, a tap read by a
  // tensor another stage writes or by an injection.
  for (const spec::Read& r : s.reads) {
    const bool used = std::ranges::any_of(mine,
                                          [&](const ContractTensor& t) {
                                            return t.contract.name == r.name &&
                                                   (r.kind == spec::ReadKind::kState
                                                        ? t.contract.borrowed_from_target
                                                        : t.contract.target_writes);
                                          }) ||
                      (r.kind == spec::ReadKind::kTap && std::ranges::contains(injected, r.name));
    if (!used) {
      return Refused(std::format("stage {} reads {} {} of stage {}, which nothing of it uses",
                                 s.name, spec::ReadKindName(r.kind), r.name, r.stage));
    }
  }
  return {};
}

}  // namespace

// --------------------------------------------------------------- compressors

std::vector<float> HadamardValues(std::uint64_t n) {
  std::vector<float> out(n * n, 0.0F);
  if (n == 0 || (n & (n - 1)) != 0) return out;
  const float value = 1.0F / std::sqrt(static_cast<float>(n));
  for (std::uint64_t i = 0; i < n; ++i) {
    for (std::uint64_t j = 0; j < n; ++j) {
      out[(i * n) + j] = std::popcount(i & j) % 2 == 0 ? value : -value;
    }
  }
  return out;
}

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
  if (s.stream.mixing.has_value() != (s.stream.streams > 1)) {
    return Refused(std::format(
        "stage {}: a stream-mixing component exactly when it has several streams", s.name));
  }
  if (!s.heads.mtp.empty()) {
    return Refused(std::format("stage {}: MTP heads (a drafter stage instead)", s.name));
  }
  if (s.stream.width == 0 || s.stream.width > kMaxStreamWidth) {
    return Refused(std::format("stage {}: a stream width of {}", s.name, s.stream.width));
  }
  if (s.layers.size() > kMaxStageLayers) {
    return Refused(std::format("stage {}: {} layers", s.name, s.layers.size()));
  }
  const StageFacts facts{.width = s.stream.width,
                         .context = s.context,
                         .softcap = s.heads.softcap,
                         .streams = s.stream.streams,
                         .folds = s.folds};
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
  for (const spec::SideInput& side : s.side) {
    if (index >= spec::kMaxComponents) return Refused("too many components");
    auto in = make(side.source, Site::kSide, 0, static_cast<std::uint16_t>(index++));
    if (!in) return std::unexpected(in.error());
    if (!in->def->side_source) {
      return Refused(
          std::format("{}: {} makes no side input", Where(s, Site::kSide, 0), side.source.kind));
    }
    out.side.push_back(std::move(*in));
  }
  if (s.stream.mixing) {
    auto in = make(*s.stream.mixing, Site::kMixing, 0, static_cast<std::uint16_t>(index++));
    if (!in) return std::unexpected(in.error());
    if (!in->def->mixing) {
      return Refused(std::format("{}: {} mixes no streams", Where(s, Site::kMixing, 0),
                                 s.stream.mixing->kind));
    }
    out.mixing = std::move(*in);
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
  // State lives on a layer's components and the embedding's (layer 0's).
  for (const Instance* in : All(out)) {
    if (in->def->state != nullptr && in->site != Site::kLayer &&
        (in->site != Site::kEmbedding || out.layers.empty())) {
      return Refused(std::format("{}: state off a layer", Where(s, in->site, in->layer)));
    }
    if ((in->def->side_source && in->site != Site::kSide) ||
        (in->def->mixing && in->site != Site::kMixing)) {
      return Refused(
          std::format("{}: {} out of its place", Where(s, in->site, in->layer), in->def->kind));
    }
  }
  // Each side input read by one component of each layer it lists, and by
  // no other.
  for (const spec::SideInput& side : s.side) {
    for (std::uint32_t l = 0; l < out.layers.size(); ++l) {
      const auto readers = std::ranges::count_if(out.layers[l], [&](const Instance& in) {
        return in.def->side_input != nullptr && in.def->side_input(in) == side.name;
      });
      const bool listed = std::ranges::contains(side.layers, l);
      if (readers != (listed ? 1 : 0)) {
        return Refused(std::format("{}: side input {} read {} times", Where(s, Site::kLayer, l),
                                   side.name, readers));
      }
    }
  }
  for (const Instance* in : All(out)) {
    if (in->def->side_input == nullptr) continue;
    // Stateless image compositions bind dynamic operands by typed owner-local
    // graph edges, not token side-table declarations. Their spatial builder
    // validates every required auxiliary/carry before allocating the graph.
    if (spec.pipeline == spec::PipelineKind::kImage && s.side.empty() && in->def->spatial) continue;
    const std::string_view name = in->def->side_input(*in);
    if (!name.empty() && !std::ranges::contains(s.side, name, &spec::SideInput::name)) {
      return Refused(std::format("{}: no side input {}", Where(s, in->site, in->layer), name));
    }
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
    const TensorTables tables = TablesOf(spec, si, tensors, tensors[s.artifact]);
    for (const Instance* in : All(*instances)) {
      const std::string where = Where(s, in->site, in->layer);
      auto formats = CheckedRoleFormats(*in, tables);
      if (!formats) return Refused(std::format("{}: {}", where, formats.error()));
    }
    stages.push_back(std::move(*instances));
  }
  // A read of another stage's state names one of its state tensors.
  for (const spec::Stage& s : spec.stages) {
    for (const spec::Read& read : s.reads) {
      if (read.kind != spec::ReadKind::kState) continue;
      const auto source = StageIndex(spec, read.stage);
      bool found = false;
      if (source) {
        for (const ContractTensor& t : ContractTensors(stages[*source])) {
          found = found || t.contract.name == read.name;
        }
      }
      if (!found) {
        return Refused(std::format("stage {} reads state {} of stage {}, which has none of it",
                                   s.name, read.name, read.stage));
      }
    }
  }
  // Each stage's reads linked to its contracts.
  for (std::size_t si = 0; si < spec.stages.size(); ++si) {
    if (auto ok = CheckReads(spec, stages, si); !ok) return ok;
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
    const auto add = [&](const Instance& in) {
      if (in.def->state == nullptr) return;
      model::StateContract contract = in.def->state(in);
      for (auto& t : contract.tensors) layer.tensors.push_back(std::move(t));
    };
    if (l == 0 && instances->embedding) add(*instances->embedding);
    for (const Instance& in : instances->layers[l]) add(in);
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
  // Cached and newly emitted framework inputs must leave the caller's
  // component and row ownership identical. Otherwise a first lazy input
  // silently makes the remainder of an owner's fragment joined.
  const auto context = b_.context();
  b_.SetComponent(kFrameworkComponent);
  if (segment == ir::kJoinedSegment) {
    b_.SetJoined();
  } else {
    b_.SetOwner(segment);
  }
  const ir::TensorId id = b_.Input(format, name);
  b_.SetContext(context);
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

ir::TensorId StepContext::Features(std::uint64_t width) {
  if (!features_.valid()) {
    features_ =
        Input(InputKind::kFeatures, ir::kJoinedSegment,
              ir::Dense(format::Encoding::kF32, ir::ShapeOf({width, rows_})), {}, 0, "features");
  }
  return features_;
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

ir::TensorId StepContext::OutputCells(std::size_t segment, const CellKey& cells) {
  const auto key = std::pair{segment, cells};
  if (const auto at = output_cells_.find(key); at != output_cells_.end()) return at->second;
  const ir::TensorId id =
      Input(InputKind::kOutputCells, static_cast<std::uint8_t>(segment),
            ir::Dense(format::Encoding::kI64, ir::ShapeOf({segments_.at(segment).outputs})), cells,
            0, std::format("tap_cells.{}.{}", segment, cells.capacity));
  output_cells_.emplace(key, id);
  return id;
}

ir::TensorId StepContext::BlockCells(std::size_t segment, const CellKey& cells,
                                     std::uint64_t blocks) {
  const auto key = std::tuple{segment, cells, blocks};
  if (const auto at = block_cells_.find(key); at != block_cells_.end()) return at->second;
  const ir::TensorId id = Input(InputKind::kStoreCells, static_cast<std::uint8_t>(segment),
                                ir::Dense(format::Encoding::kI64, ir::ShapeOf({blocks})), cells,
                                blocks, std::format("block_cells.{}.{}", segment, cells.block));
  block_cells_.emplace(key, id);
  return id;
}

std::uint64_t StepContext::ReadCells(std::size_t segment, const CellKey& cells) const {
  const SegmentRows& seg = segments_.at(segment);
  const std::uint64_t read = seg.read_cells;
  // A positions-basis block tensor's: the blocks its positions complete,
  // at the state contract's cell padding.
  if (cells.block != 0 && cells.positions) {
    const std::uint64_t end = std::uint64_t{seg.past} + (cells.frozen ? 0 : seg.rows);
    return std::min(cells.capacity, CompressedReadCells(end, cells.block, cells.cell_pad));
  }
  // A block tensor's cells are blocks of the read cells.
  if (cells.block != 0)
    return std::min<std::uint64_t>(cells.capacity, Pad(read, cells.block) / cells.block);
  return std::min<std::uint64_t>(cells.capacity, read);
}

std::expected<ir::TensorId, std::string> StepContext::Mask(std::size_t segment,
                                                           const CellKey& cells,
                                                           std::uint32_t window, bool ahead,
                                                           bool exact) const {
  const auto at = masks_.find(
      {segment, MaskKey{.cells = cells, .window = window, .ahead = ahead, .exact = exact}});
  if (at == masks_.end()) return Refused("a mask its component did not declare");
  return at->second;
}

ir::TensorId StepContext::HeldPositions(std::size_t segment, const CellKey& cells) {
  const std::uint64_t read = ReadCells(segment, cells);
  const auto key = std::tuple{segment, cells, read};
  if (const auto at = cell_positions_.find(key); at != cell_positions_.end()) return at->second;
  const ir::TensorId held =
      Input(InputKind::kCellPositions, static_cast<std::uint8_t>(segment),
            ir::Dense(format::Encoding::kI32, ir::ShapeOf({read})), cells, read,
            std::format("held.{}.{}{}{}{}", segment, cells.capacity, cells.ring ? ".ring" : "",
                        cells.frozen ? ".frozen" : "",
                        cells.block != 0 ? std::format(".block{}", cells.block) : std::string()));
  cell_positions_.emplace(key, held);
  return held;
}

std::expected<ir::TensorId, std::string> StepContext::VisibleCells(std::size_t segment,
                                                                   const CellKey& cells,
                                                                   std::uint64_t read,
                                                                   std::uint64_t visible_capacity) {
  if (segment >= segments_.size()) return Refused("visible counts of no segment");
  const auto& seg = segments_[segment];
  StepInput input{.kind = InputKind::kVisibleCells,
                  .segment = 0,
                  .cells = cells,
                  .read = read,
                  .visible_capacity = visible_capacity};
  planner::StepShape shape;
  shape.segments.push_back({.slot = seg.slot, .rows = seg.rows, .past = seg.past});
  if (auto values = InputValues(input, shape, {}); !values) return std::unexpected(values.error());
  const auto key = std::tuple{segment, cells, read, visible_capacity};
  if (const auto at = visible_cells_.find(key); at != visible_cells_.end()) return at->second;
  const auto id = Input(InputKind::kVisibleCells, static_cast<std::uint8_t>(segment),
                        ir::Dense(format::Encoding::kI32, ir::ShapeOf({seg.rows})), cells, read,
                        std::format("visible.{}.{}.{}", segment, read, visible_capacity));
  inputs_.back().visible_capacity = visible_capacity;
  visible_cells_.emplace(key, id);
  return id;
}

ir::TensorId StepContext::SegmentPositions(std::size_t segment) {
  if (!segment_positions_.at(segment).valid()) {
    const SegmentRows& seg = segments_.at(segment);
    b_.SetComponent(kFrameworkComponent);
    const ir::TensorId all = Positions();
    b_.SetComponent(kFrameworkComponent);
    const format::Shape shape = ir::ShapeOf({seg.rows});
    segment_positions_[segment] =
        b_.View(all, shape, {4, 4ULL * seg.rows, 4ULL * seg.rows, 4ULL * seg.rows},
                std::uint64_t{seg.first_row} * 4);
    b_.SetComponent(component_);
  }
  return segment_positions_[segment];
}

ir::TensorId StepContext::Positions(std::uint8_t sections) {
  if (sections <= 1) return Positions();
  if (const auto at = sectioned_.find(sections); at != sectioned_.end()) return at->second;
  const ir::TensorId id =
      Input(InputKind::kPositions, ir::kJoinedSegment,
            ir::Dense(format::Encoding::kI32, ir::ShapeOf({std::uint64_t{sections} * rows_})), {},
            0, std::format("positions.{}", sections));
  inputs_.back().sections = sections;
  sectioned_.emplace(sections, id);
  return id;
}

ir::TensorId StepContext::StateRow() {
  if (!state_row_.valid()) {
    state_row_ = Input(InputKind::kStoreCells, ir::kJoinedSegment,
                       ir::Dense(format::Encoding::kI64, ir::ShapeOf({1})),
                       {.capacity = 1, .fixed = true}, 0, "state_row");
  }
  return state_row_;
}

ir::TensorId StepContext::BlockPositions(std::size_t segment, const CellKey& cells,
                                         std::uint64_t blocks, std::uint8_t sections) {
  const auto key = std::tuple{segment, cells, blocks, sections};
  if (const auto at = blocks_.find(key); at != blocks_.end()) return at->second;
  const ir::TensorId id =
      Input(InputKind::kBlockPositions, static_cast<std::uint8_t>(segment),
            ir::Dense(format::Encoding::kI32, ir::ShapeOf({std::uint64_t{sections} * blocks})),
            cells, blocks, std::format("block_pos.{}.{}", segment, cells.block));
  inputs_.back().sections = sections;
  blocks_.emplace(key, id);
  return id;
}

ir::TensorId StepContext::SideRows(std::uint16_t component, std::uint64_t count) {
  if (const auto at = side_rows_.find(component); at != side_rows_.end()) return at->second;
  const ir::TensorId id = Input(InputKind::kSideRows, ir::kJoinedSegment,
                                ir::Dense(format::Encoding::kI32, ir::ShapeOf({count * rows_})), {},
                                count, std::format("side_rows.{}", component));
  inputs_.back().component = component;
  side_rows_.emplace(component, id);
  return id;
}

std::expected<std::uint64_t, std::string> CommitSaveBound(const spec::ModelSpec& model,
                                                          std::uint32_t owners) {
  if (owners == 0 || owners > kMaxSegments) return Refused("invalid commit-save owner envelope");
  std::uint64_t total = 0;
  const auto add = [&](const spec::ComponentRef& ref) -> bool {
    const auto* def = Find(ref.kind, ref.version);
    return def != nullptr &&
           !__builtin_add_overflow(total, std::uint64_t{def->max_commit_saves} * owners, &total);
  };
  const auto optional = [&](const std::optional<spec::ComponentRef>& ref) {
    return !ref || add(*ref);
  };
  for (const auto& stage : model.stages) {
    if (!optional(stage.embedding) || !optional(stage.stream.mixing) ||
        !optional(stage.heads.final_norm) || !optional(stage.heads.output))
      return Refused("unknown component or overflowing commit-save bound");
    for (const auto& side : stage.side)
      if (!add(side.source)) return Refused("unknown side component or commit-save overflow");
    for (const auto& layer : stage.layers)
      for (const auto& ref : layer.components)
        if (!add(ref)) return Refused("unknown layer component or commit-save overflow");
    for (const auto& ref : stage.heads.mtp)
      if (!add(ref)) return Refused("unknown head component or commit-save overflow");
  }
  return total;
}

void StepContext::Save(const Instance& in, std::string_view what, ir::TensorId tensor,
                       std::uint8_t segment) {
  const std::uint64_t limit =
      in.def ? std::uint64_t{in.def->max_commit_saves} * segments_.size() : 0;
  const auto count = std::ranges::count_if(saves_, [&](const CommitSave& save) {
    return save.layer == in.layer && save.component == in.index;
  });
  if (std::cmp_greater_equal(count, limit) ||
      (segment != ir::kJoinedSegment && segment >= segments_.size())) {
    b_.Refuse(ir::GraphError::kBounds, "component exceeds its declared commit-save bound");
    return;
  }

  // The rows of the save's buffer: the segment's (the step's, joined).
  ir::TensorId rows = save_rows_[segment];
  if (!rows.valid()) {
    const std::uint32_t n = segment == ir::kJoinedSegment ? rows_ : segments_.at(segment).rows;
    rows = Input(
        InputKind::kStoreCells, segment, ir::Dense(format::Encoding::kI64, ir::ShapeOf({n})),
        {.capacity = n, .per_row = true, .first_row = 0}, 0, std::format("save_rows.{}", segment));
    save_rows_[segment] = rows;
    // Lowerings store saves after the IR launches; the indices stay live
    // through those stores even though no IR node reads them.
    b_.Keep(rows);
    // The save's explicitly named owner also applies to subsequent emission.
    if (segment != ir::kJoinedSegment) b_.SetOwner(segment);
  }
  saves_.push_back({.layer = in.layer,
                    .component = in.index,
                    .what = std::string(what),
                    .segment = segment,
                    .tensor = tensor,
                    .rows = rows});
}

ir::TensorId StepContext::Saved(const Instance& in, std::string_view what, std::size_t segment,
                                std::uint64_t width) {
  const ir::TensorId id =
      Input(InputKind::kCommitRows, static_cast<std::uint8_t>(segment),
            ir::Dense(format::Encoding::kF32, ir::ShapeOf({width, segments_.at(segment).rows})), {},
            0, std::format("saved.{}.{}.{}", in.layer, what, segment));
  inputs_.back().component = in.index;
  inputs_.back().layer = in.layer;
  inputs_.back().what = std::string(what);
  return id;
}

void StepContext::DeclareMasks(std::span<const MaskKey> masks) {
  // Causal first, then by window (each window's plain before its kind's);
  // cells within.
  std::vector<MaskKey> order(masks.begin(), masks.end());
  std::ranges::sort(order, [](const MaskKey& a, const MaskKey& b) {
    return std::tie(a.window, a.ahead, a.cells, a.exact) <
           std::tie(b.window, b.ahead, b.cells, b.exact);
  });
  order.erase(std::ranges::unique(order).begin(), order.end());
  for (std::size_t s = 0; s < segments_.size(); ++s) {
    const SegmentRows& seg = segments_[s];
    for (const MaskKey& key : order) {
      if (masks_.contains({s, key})) continue;
      const ir::TensorId held = HeldPositions(s, key.cells);
      const ir::TensorId positions = SegmentPositions(s);
      b_.SetComponent(kFrameworkComponent);
      b_.SetOwner(static_cast<std::uint8_t>(s));
      ir::MaskParams params{.kind = key.window == 0 ? ir::MaskKind::kCausal : ir::MaskKind::kWindow,
                            .exact_rows = key.exact,
                            .window = key.window};
      if (key.ahead) params.kind = ir::MaskKind::kAhead;
      const ir::TensorId mask =
          b_.MaskBuild(positions, held, key.exact ? seg.rows : Pad(seg.rows, kMaskRowAlign), params,
                       format::Encoding::kF16);
      b_.SetJoined();
      b_.SetComponent(component_);
      masks_.emplace(std::pair{s, key}, mask);
    }
  }
}

std::uint64_t StepContext::Blocks(std::size_t segment, std::uint32_t ratio, bool overlap) const {
  const SegmentRows& seg = segments_.at(segment);
  return CompressorBlocks(seg.rows, seg.past, ratio, overlap);
}

std::uint64_t StepContext::Kept(std::size_t segment, std::uint64_t capacity) const {
  return std::min<std::uint64_t>(segments_.at(segment).rows, capacity);
}

ir::TensorId StepContext::Compressor(InputKind kind, std::size_t segment, const CellKey& cells,
                                     std::uint32_t ratio, bool overlap, std::uint64_t count,
                                     format::Encoding encoding, std::string_view name) {
  const auto key = std::tuple{kind, segment, cells, ratio, overlap};
  if (const auto at = compressor_.find(key); at != compressor_.end()) return at->second;
  const ir::TensorId id = Input(
      kind, segment == kJoinedCompressor ? ir::kJoinedSegment : static_cast<std::uint8_t>(segment),
      ir::Dense(encoding, ir::ShapeOf({count})), cells, 0, name);
  inputs_.back().ratio = ratio;
  inputs_.back().overlap = overlap;
  inputs_.back().ring = cells.capacity;
  compressor_.emplace(key, id);
  return id;
}

ir::TensorId StepContext::PositionPhases(std::uint32_t ratio) {
  return Compressor(InputKind::kPositionPhases, kJoinedCompressor, {}, ratio, false, rows_,
                    format::Encoding::kI32, std::format("phases.{}", ratio));
}

ir::TensorId StepContext::KeptRows(std::size_t segment, const CellKey& ring) {
  return Compressor(InputKind::kKeptRows, segment, ring, 0, false, Kept(segment, ring.capacity),
                    format::Encoding::kI32, std::format("kept_rows.{}.{}", segment, ring.capacity));
}

ir::TensorId StepContext::KeptCells(std::size_t segment, const CellKey& ring) {
  return Compressor(InputKind::kKeptCells, segment, ring, 0, false, Kept(segment, ring.capacity),
                    format::Encoding::kI64,
                    std::format("kept_cells.{}.{}", segment, ring.capacity));
}

ir::TensorId StepContext::PoolReads(std::size_t segment, const CellKey& ring, std::uint32_t ratio,
                                    bool overlap) {
  const std::uint64_t reads =
      std::uint64_t{ratio} * Blocks(segment, ratio, overlap) * (overlap ? 2 : 1);
  return Compressor(InputKind::kPoolReads, segment, ring, ratio, overlap, reads,
                    format::Encoding::kI32,
                    std::format("pool_reads.{}.{}.{}", segment, ratio, ring.capacity));
}

ir::TensorId StepContext::BlockCells(std::size_t segment, const CellKey& cache, std::uint32_t ratio,
                                     bool overlap) {
  return Compressor(InputKind::kBlockCells, segment, cache, ratio, overlap,
                    Blocks(segment, ratio, overlap), format::Encoding::kI64,
                    std::format("block_cells.{}.{}.{}", segment, ratio, cache.capacity));
}

ir::TensorId StepContext::BlockStarts(std::size_t segment, std::uint32_t ratio, bool overlap) {
  return Compressor(InputKind::kBlockStarts, segment, {}, ratio, overlap,
                    Blocks(segment, ratio, overlap), format::Encoding::kI32,
                    std::format("block_starts.{}.{}", segment, ratio));
}

ir::TensorId StepContext::Hadamard(std::uint64_t n) {
  if (const auto at = hadamard_.find(n); at != hadamard_.end()) return at->second;
  b_.SetComponent(kFrameworkComponent);
  b_.SetJoined();
  const ir::TensorId id = b_.Constant(ir::Dense(format::Encoding::kF32, ir::ShapeOf({n, n})),
                                      std::format("hadamard.{}", n));
  b_.SetComponent(component_);
  inputs_.push_back({.tensor = id, .kind = InputKind::kHadamard, .read = n});
  hadamard_.emplace(n, id);
  return id;
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
  CellKey key{.capacity = cells, .frozen = c.borrowed_from_target};
  switch (c.cells) {
    case model::CellRule::kRing:
      key.ring = true;
      break;
    case model::CellRule::kPositionRing:
      key.ring = true;
      key.recent = true;
      break;
    case model::CellRule::kFixed:
      key.fixed = true;
      break;
    case model::CellRule::kPerBlock:
      key.block = c.window_or_ratio;
      key.positions = c.basis == model::BlockBasis::kPositions;
      key.cell_pad = c.cell_pad;
      break;
    case model::CellRule::kPerRow:
      key.per_row = true;
      key.first_row = c.window_or_ratio;
      break;
    case model::CellRule::kPerPosition:
      break;
  }
  return key;
}

// The layout of stage `stage` from the bindings, if given.
const model::StateLayoutDerived* LayoutOf(const Bindings& bindings, std::size_t stage) {
  return stage < bindings.stages.size() ? bindings.stages[stage] : nullptr;
}

// Stage `owner`'s WeightRef of its tensor `name` (`part`): its first
// component binding it of its own artifact.
std::expected<ir::WeightRef, std::string> OwnerRef(const spec::ModelSpec& spec, std::uint16_t owner,
                                                   std::string_view name, format::PartRole part) {
  auto instances = InstancesOf(spec, owner);
  if (!instances) return std::unexpected(instances.error());
  for (const Instance* in : All(*instances)) {
    for (const spec::WeightBinding& w : in->ref->weights) {
      if (w.tensor == name && w.part == part && w.stage.empty()) {
        const auto role = RoleIndex(*in->def, w.role);
        if (!role) break;
        return ir::WeightRef{.stage = owner,
                             .component = in->index,
                             .role = *role,
                             .layer = in->layer,
                             .expert_array = w.experts};
      }
    }
  }
  return Refused(std::format("stage {} binds no {} of its own", spec.stages.at(owner).name, name));
}

std::expected<ir::WeightRef, std::string> ResolvedWeightRef(const spec::ModelSpec& spec,
                                                            std::uint16_t at, const Instance& in,
                                                            std::uint16_t role,
                                                            const RoleFormat& format) {
  if (!format.stage.empty()) {
    const auto owner = StageIndex(spec, format.stage);
    if (!owner) return Refused(std::format("no stage {}", format.stage));
    return OwnerRef(spec, *owner, format.parts.front().first, format.parts.front().second);
  }
  return ir::WeightRef{.stage = at,
                       .component = in.index,
                       .role = role,
                       .layer = in.layer,
                       .expert_array = format.experts};
}

// A streams tap's value of the streams `x` [width, streams, rows]: them as
// [width · streams, rows] (kStreams), or their mean (kStreamMean: each
// row's streams added in order onto the first's view, then scaled).
ir::TensorId TapOfStreams(ir::Builder& b, ir::TensorId x, spec::TapPoint point) {
  const ir::TensorDesc d = b.tensor(x);
  if (point == spec::TapPoint::kStreams) {
    return b.Reshape(x, ir::ShapeOf({d.ne(0) * d.ne(1), d.ne(2)}));
  }
  const format::Shape one = ir::ShapeOf({d.ne(0), d.ne(2)});
  const std::array<std::uint64_t, 4> nb{d.nb[0], d.nb[2], d.nb[2] * d.ne(2), d.nb[2] * d.ne(2)};
  ir::TensorId sum = b.View(x, one, nb, 0);
  for (std::uint64_t st = 1; st < d.ne(1); ++st) {
    const ir::TensorId next = b.View(x, one, nb, st * d.nb[1]);
    sum = b.Add(sum, next);
  }
  return b.Scale(sum, 1.0F / static_cast<float>(d.ne(1)));
}

// The taps of `stage` other stages read: (reader, its tensor, the tap).
struct TapWrite {
  std::uint16_t drafter = 0;
  ContractTensor tensor;
  spec::FeatureTap tap;
};
std::expected<std::vector<TapWrite>, std::string> TapWrites(const spec::ModelSpec& spec,
                                                            std::uint16_t stage) {
  std::vector<TapWrite> out;
  const spec::Stage& source = spec.stages[stage];
  for (std::size_t d = 0; d < spec.stages.size(); ++d) {
    if (d == stage) continue;
    const spec::Stage& s = spec.stages[d];
    std::optional<StageInstances> instances;
    for (const spec::Read& r : s.reads) {
      if (r.kind != spec::ReadKind::kTap || r.stage != source.name) continue;
      const auto tap = std::ranges::find(source.taps, r.name, &spec::FeatureTap::name);
      if (tap == source.taps.end()) {
        return Refused(std::format("stage {} reads tap {}, which stage {} has not", s.name, r.name,
                                   source.name));
      }
      if (!instances) {
        auto made = InstancesOf(spec, d);
        if (!made) return std::unexpected(made.error());
        instances = std::move(*made);
      }
      for (ContractTensor& t : ContractTensors(*instances)) {
        // Injected state is the injection's (Injections), not a tap's.
        const bool injected = t.instance != nullptr && t.instance->def->inject != nullptr;
        if (t.contract.target_writes && t.contract.name == r.name && !injected) {
          t.instance = nullptr;  // `instances` is this loop's
          out.push_back(
              {.drafter = static_cast<std::uint16_t>(d), .tensor = std::move(t), .tap = *tap});
        }
      }
    }
  }
  return out;
}

// The drafters that inject into their state from `stage`'s taps (their
// embedding's ComponentDef::inject): their instances and the taps their
// injection reads, in its order.
struct Injection {
  std::uint16_t drafter = 0;
  StageInstances instances;
  std::vector<spec::FeatureTap> taps;
};
std::expected<std::vector<Injection>, std::string> Injections(const spec::ModelSpec& spec,
                                                              std::uint16_t stage) {
  std::vector<Injection> out;
  const spec::Stage& source = spec.stages[stage];
  for (std::size_t d = 0; d < spec.stages.size(); ++d) {
    if (d == stage) continue;
    const spec::Stage& s = spec.stages[d];
    const bool reads = std::ranges::any_of(s.reads, [&](const spec::Read& r) {
      return r.kind == spec::ReadKind::kTap && r.stage == source.name;
    });
    if (!reads) continue;
    auto instances = InstancesOf(spec, d);
    if (!instances) return std::unexpected(instances.error());
    if (!instances->embedding || instances->embedding->def->inject == nullptr) continue;
    const Instance& e = *instances->embedding;
    if (e.def->inject_reads == nullptr) return Refused("an injection without its reads");
    Injection made{.drafter = static_cast<std::uint16_t>(d), .instances = {}, .taps = {}};
    for (const std::string& name : e.def->inject_reads(e)) {
      const bool mine = std::ranges::any_of(s.reads, [&](const spec::Read& r) {
        return r.kind == spec::ReadKind::kTap && r.stage == source.name && r.name == name;
      });
      if (!mine) continue;  // another stage's (CheckReads refuses two)
      const auto tap = std::ranges::find(source.taps, name, &spec::FeatureTap::name);
      if (tap == source.taps.end() || tap->point == spec::TapPoint::kPostFinalNorm) {
        return Refused(std::format("stage {} injects from tap {}, no layer's tap of stage {}",
                                   s.name, name, source.name));
      }
      made.taps.push_back(*tap);
    }
    if (made.taps.empty()) continue;
    made.instances = std::move(*instances);
    out.push_back(std::move(made));
  }
  return out;
}

}  // namespace

std::expected<void, std::string> CheckStateWrites(const ir::Graph& g,
                                                  std::span<const StepState> state,
                                                  std::span<const ir::TensorId> tensors,
                                                  std::uint16_t stage) {
  const auto mapped = [&](ir::TensorId t) {
    return tensors.empty() ? t : (t.v < tensors.size() ? tensors[t.v] : ir::TensorId{});
  };
  for (const ir::Node& n : g.nodes()) {
    const auto target = StoreTarget(n);
    if (!target) continue;
    const ir::TensorId root = g.root(n.src[*target]);
    const ir::Role role = g.tensor(root).role;
    // A store into an activation is the step's own (a mask built by stores).
    if (role == ir::Role::kActivation || role == ir::Role::kOutput) continue;
    const auto st =
        std::ranges::find_if(state, [&](const StepState& s) { return mapped(s.tensor) == root; });
    if (st == state.end()) return Refused("a store into a leaf that is no state");
    if (st->borrowed) {
      return Refused(std::format("a store into stage {}'s borrowed {} (layer {})", st->ref.stage,
                                 st->ref.tensor, st->ref.layer));
    }
    if (st->ref.stage != stage && !st->tap) {
      return Refused(
          std::format("stage {}'s step stores into stage {}'s state, not a tap it writes", stage,
                      st->ref.stage));
    }
  }
  return {};
}

std::expected<void, std::string> CheckStateWrites(const StepGraph& step, std::uint16_t stage) {
  return CheckStateWrites(step.graph, step.state, {}, stage);
}

std::optional<std::pair<std::size_t, std::size_t>> NarrowPoint(const StageInstances& instances) {
  if (!instances.output || (instances.final_norm && !instances.final_norm->def->row_local) ||
      !instances.output->def->row_local ||
      (instances.output->def->gathers != nullptr &&
       !instances.output->def->gathers(*instances.output))) {
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
                                                const Bindings& bindings,
                                                const StepOptions& options) {
  if (shape.stage != stage) return Refused("a shape of another stage");
  if (auto checked = CheckShape(shape); !checked) return std::unexpected(checked.error());
  auto instances = InstancesOf(spec, stage);
  if (!instances) return std::unexpected(instances.error());
  if (!instances->embedding) return Refused("a stage without an embedding");
  const spec::Stage& self = spec.stages[stage];
  if (options.rounded_indexer) {
    if (stage != 0 || spec.stages.size() != 1 || self.role != spec::StageRole::kTarget ||
        !self.reads.empty() || !self.taps.empty() || bindings.state == nullptr ||
        !bindings.state->borrowed.empty() || bindings.state->context < 4 ||
        bindings.state->context > 4096 || shape.segments.size() != 1 || shape.row_invariant ||
        (shape.phase != planner::Phase::kPrefill && shape.phase != planner::Phase::kDecode &&
         shape.phase != planner::Phase::kScore))
      return Refused("rounded indexer requires one ordinary bounded target");
    const auto& own = shape.segments.front();
    if (own.slot != 0 || own.rows > 8 || own.past > bindings.state->context ||
        own.rows > bindings.state->context - own.past || own.read_cells == 0 ||
        own.read_cells > bindings.state->context)
      return Refused("rounded indexer owner exceeds its admitted context or rows");
    bool indexed = false;
    for (const auto& layer : instances->layers) {
      for (const auto& in : layer) {
        if (in.def->kind != "attention.mqa") continue;
        if (in.params.Name("cache") == "injected")
          return Refused("rounded indexer does not admit injected attention");
        if (!in.params.Flag("indexer")) continue;
        indexed = true;
        if (in.params.Int("ratio") != 4 || !in.params.Flag("overlap") ||
            in.params.Int("index_dim") != 128 || in.params.Int("indexer_heads") != 64 ||
            in.params.Int("top_k") <= 0 || in.params.Int("top_k") > 512)
          return Refused("rounded indexer requires the bounded ratio-four CSA geometry");
      }
    }
    if (!indexed) return Refused("rounded indexer policy names no indexed attention");
  }
  StepBuilder sb(stage, shape);
  StepContext& ctx = sb.ctx();
  ctx.EnableOwnerProducts(options.owner_products);
  if (options.rounded_indexer) ctx.SetRoundedIndexerContext(bindings.state->context);
  const bool head = ctx.outputs() > 0;
  if (head && !instances->output) return Refused("outputs from a stage without an output head");
  TensorTables tables{.own = bindings.tensors, .stages = {}};
  for (std::size_t i = 0; i < spec.stages.size(); ++i) {
    if (i != stage && spec.stages[i].artifact < bindings.artifacts.size()) {
      tables.stages.emplace(spec.stages[i].name, bindings.artifacts[spec.stages[i].artifact]);
    }
  }
  // The taps other stages read of this one.
  auto taps = TapWrites(spec, stage);
  if (!taps) return std::unexpected(taps.error());

  // Each layer's stateful instances: their contracts, first tensor and
  // layout items (a borrowed tensor's that of the stage it reads).
  struct Stateful {
    model::StateContract contract;
    std::uint16_t first = 0;
    std::vector<std::uint64_t> cells;
    std::vector<std::size_t> items;
    std::vector<std::uint16_t> sources;
  };
  std::vector<std::vector<std::optional<Stateful>>> stateful(instances->layers.size());
  std::vector<StepContext::MaskKey> masks;
  for (std::size_t l = 0; l < instances->layers.size(); ++l) {
    // The embedding's state comes first in layer 0's contract (a tap
    // another stage writes; the stage's own steps neither read nor write
    // it).
    std::uint16_t first = 0;
    if (l == 0 && instances->embedding && instances->embedding->def->state != nullptr) {
      first = static_cast<std::uint16_t>(
          instances->embedding->def->state(*instances->embedding).tensors.size());
    }
    for (const Instance& in : instances->layers[l]) {
      auto& slot = stateful[l].emplace_back();
      if (in.def->state == nullptr) continue;
      if (bindings.state == nullptr) return Refused("a stateful stage without its state layout");
      Stateful st{
          .contract = in.def->state(in), .first = first, .cells = {}, .items = {}, .sources = {}};
      for (std::size_t t = 0; t < st.contract.tensors.size(); ++t) {
        const model::StateTensorContract& c = st.contract.tensors[t];
        const auto index = static_cast<std::uint32_t>(first + t);
        if (c.borrowed_from_target) {
          // The tensor it reads of the stage it reads: its cells.
          const auto borrowed = std::ranges::find_if(bindings.state->borrowed, [&](const auto& b) {
            return b.layer == in.layer && b.tensor == index;
          });
          if (borrowed == bindings.state->borrowed.end() || borrowed->contract != c) {
            return Refused(std::format("layer {}: the state layout is not this spec's", in.layer));
          }
          const auto source = ReadSource(spec, self, spec::ReadKind::kState, c.name);
          const model::StateLayoutDerived* target = source ? LayoutOf(bindings, *source) : nullptr;
          if (target == nullptr) {
            return Refused(std::format(
                "layer {} borrows {} without the layout of the stage it reads", in.layer, c.name));
          }
          const auto* item = model::BorrowedItem(*target, *borrowed);
          if (item == nullptr) {
            return Refused(std::format("layer {}: the stage it reads has no {} in layer {}",
                                       in.layer, c.name, c.target_layer));
          }
          st.items.push_back(static_cast<std::size_t>(item - target->items.data()));
          st.cells.push_back(item->cells);
          st.sources.push_back(*source);
        } else {
          const auto item = ItemOf(*bindings.state, in.layer, index);
          if (!item || bindings.state->items[*item].contract != c) {
            return Refused(std::format("layer {}: the state layout is not this spec's", in.layer));
          }
          st.items.push_back(*item);
          st.cells.push_back(bindings.state->items[*item].cells);
          st.sources.push_back(stage);
        }
        const CellKey key = KeyOf(c, st.cells.back());
        for (const SegmentRows& seg : ctx.segments()) {
          if (auto ok = CheckCells(seg, key, c.window_or_ratio); !ok) {
            return Refused(std::format("layer {} state {}: {}", in.layer, c.name, ok.error()));
          }
        }
      }
      if (in.def->masks != nullptr) {
        for (const MaskUse& m : in.def->masks(in)) {
          if (options.rounded_indexer &&
              st.contract.tensors.at(m.tensor).kind == model::StateKind::kCompressed) {
            continue;
          }
          masks.push_back({.cells = KeyOf(st.contract.tensors.at(m.tensor), st.cells.at(m.tensor)),
                           .window = m.window,
                           .ahead = m.ahead,
                           .exact = in.def->exact_masks});
        }
      }
      first = static_cast<std::uint16_t>(first + st.contract.tensors.size());
      slot = std::move(st);
    }
  }
  ctx.DeclareMasks(masks);

  std::map<std::string, ir::TensorId, std::less<>> sides;
  struct Flow {
    ir::TensorId stream{};
    ir::TensorId residual{};
    ir::TensorId carry{};
    ir::TensorId carry2{};
  };
  // A component's fragment (`hook`: its emit, or a drafter's inject), with
  // the weights of `at`'s artifact (`at_tables`): this stage's, or an
  // injecting drafter's.
  using Hook = std::expected<FragmentOut, std::string> (*)(Fragment&);
  bool carry_needed = true;
  std::function<ir::TensorId(std::string_view)> side_lookup = [&](std::string_view name) {
    const auto at = sides.find(name);
    return at == sides.end() ? ir::TensorId{} : at->second;
  };
  const auto emit_at =
      [&](std::uint16_t at, const TensorTables& at_tables, Hook hook, const Instance& in,
          const Flow& flow, std::span<const ir::TensorId> state, std::size_t tensors, bool needed,
          bool narrow, bool narrowed, const std::function<ir::TensorId(std::string_view)>& side)
      -> std::expected<FragmentOut, std::string> {
    auto formats = CheckedRoleFormats(in, at_tables);
    if (!formats) return std::unexpected(formats.error());
    std::vector<ir::TensorId> leaves(in.def->roles.size());
    std::vector<bool> used(in.def->roles.size(), false);
    std::vector<std::optional<ir::WeightRef>> owners(in.def->roles.size());
    for (std::size_t role = 0; role != formats->size(); ++role) {
      if (!(*formats)[role]) continue;
      used[role] = true;
      auto ref =
          ResolvedWeightRef(spec, at, in, static_cast<std::uint16_t>(role), *(*formats)[role]);
      if (!ref) return std::unexpected(ref.error());
      owners[role] = *ref;
    }
    sb.Begin(in);
    Fragment f{.b = sb.b(),
               .instance = in,
               .step = ctx,
               .stream = flow.stream,
               .weight = [&](std::uint16_t role) -> ir::TensorId {
                 base::Check(role < used.size() && used[role], "a fragment uses its roles");
                 if (!leaves[role].valid()) {
                   const auto& rf = *(*formats)[role];
                   leaves[role] =
                       sb.Weight(in, role, rf.format, rf.parts, owners[role], rf.experts);
                 }
                 return leaves[role];
               },
               .state = state,
               .state_tensors = tensors,
               .stream_needed = needed,
               .narrow = narrow,
               .narrowed = narrowed,
               .residual = flow.residual,
               .carry = flow.carry,
               .carry2 = flow.carry2,
               .side = side,
               .carry_needed = carry_needed};
    auto out = hook(f);
    if (!out) return std::unexpected(out.error());
    sb.b().SetJoined();
    return out;
  };
  const auto emit = [&](const Instance& in, const Flow& flow, std::span<const ir::TensorId> state,
                        std::size_t tensors, bool needed, bool narrow = false,
                        bool narrowed = false) {
    return emit_at(stage, tables, in.def->emit, in, flow, state, tensors, needed, narrow, narrowed,
                   side_lookup);
  };
  // A component's outputs into the flow.
  const auto follow = [](Flow& flow, const FragmentOut& out) {
    if (out.stream.valid()) flow.stream = out.stream;
    if (out.residual.valid()) flow.residual = out.residual;
    if (out.carry.valid()) flow.carry = out.carry;
    if (out.carry2.valid()) flow.carry2 = out.carry2;
  };
  // The drafters this stage's steps inject into, and the layers whose
  // streams their taps read.
  auto injections = Injections(spec, stage);
  if (!injections) return std::unexpected(injections.error());
  // A tapped layer's streams or stream, by tap name (TapPoint), kept until
  // the injection reads them; a stream mean is made when first read.
  std::map<std::string, ir::TensorId, std::less<>> captured;

  Flow flow;
  auto embedded = emit(*instances->embedding, flow, {}, 0, true);
  if (!embedded) return Refused(std::format("embedding: {}", embedded.error()));
  follow(flow, *embedded);
  // The side inputs, then the streams made from the embedding (unless it
  // made them itself: an MTP drafter's).
  for (std::size_t i = 0; i < instances->side.size(); ++i) {
    auto made = emit(instances->side[i], {}, {}, 0, true);
    if (!made) return Refused(std::format("side input {}: {}", self.side[i].name, made.error()));
    sides.emplace(self.side[i].name, made->stream);
  }
  if (instances->mixing && !flow.residual.valid()) {
    auto mixed = emit(*instances->mixing, flow, {}, 0, true);
    if (!mixed) return Refused(std::format("stream mixing: {}", mixed.error()));
    if (!mixed->residual.valid()) return Refused("stream mixing made no streams");
    follow(flow, *mixed);
  }

  // A layer's taps (its output or its streams, every row) are written
  // after its last component, a final norm's after the final norm.
  const auto layer_tap = [&](std::size_t l) {
    return std::ranges::any_of(*taps,
                               [&](const TapWrite& t) {
                                 return t.tap.point != spec::TapPoint::kPostFinalNorm &&
                                        t.tap.layer == l;
                               }) ||
           std::ranges::any_of(*injections, [&](const Injection& i) {
             return std::ranges::contains(i.taps, l, &spec::FeatureTap::layer);
           });
  };
  // Which components' output streams something reads: from the head back.
  std::vector<std::vector<bool>> needed(instances->layers.size());
  bool need = head;
  for (std::size_t l = instances->layers.size(); l-- > 0;) {
    need = need || layer_tap(l);
    needed[l].resize(instances->layers[l].size());
    for (std::size_t c = instances->layers[l].size(); c-- > 0;) {
      needed[l][c] = need;
      if (stateful[l][c]) need = true;  // its writes read its input
    }
  }
  // Writes `tap`'s tensor in its reader's stage: every row of `source`
  // [width, rows] at the rows' cells, or (`outputs`) the output rows of
  // `source` [width, outputs] at their positions' cells.
  const auto write_tap = [&](const TapWrite& tap, ir::TensorId source,
                             bool outputs) -> std::expected<void, std::string> {
    const model::StateTensorContract& c = tap.tensor.contract;
    const model::StateLayoutDerived* layout = LayoutOf(bindings, tap.drafter);
    if (layout == nullptr) {
      return Refused(std::format("a tap stage {} reads without its layout", tap.drafter));
    }
    const auto item = ItemOf(*layout, tap.tensor.layer, tap.tensor.index);
    if (!item || layout->items[*item].contract != c) {
      return Refused(std::format("stage {}'s layout is not this spec's", tap.drafter));
    }
    const CellKey key = KeyOf(c, layout->items[*item].cells);
    ir::Builder& b = sb.b();
    b.SetComponent(kFrameworkComponent);
    ctx.SetComponent(kFrameworkComponent);
    const std::uint64_t width = b.tensor(source).ne(0);
    if (width * 4 != c.cell_bytes) return Refused(std::format("tap {} of another width", c.name));
    const auto nb = b.tensor(source).nb;
    std::uint64_t first = 0;
    for (std::size_t s = 0; s < ctx.segments().size(); ++s) {
      const SegmentRows& seg = ctx.segments()[s];
      const std::uint32_t rows = outputs ? seg.outputs : seg.rows;
      // A segment without outputs (a prefill chunk beside another owner's
      // head rows) has no feature to tap.
      if (rows == 0) continue;
      if (auto ok = CheckCells(seg, key, 0); !ok && !outputs) {
        return Refused(std::format("tap {}: {}", c.name, ok.error()));
      }
      if (outputs && seg.outputs > key.capacity) {
        return Refused(std::format("{} outputs past the {} cells of tap {}", seg.outputs,
                                   key.capacity, c.name));
      }
      const ir::TensorId leaf = sb.StateLeaf(tap.drafter, tap.tensor.layer, tap.tensor.index, c, s,
                                             key.capacity, *item, true);
      // One segment's rows are the whole tensor; several, a view of each.
      const ir::TensorId mine =
          ctx.segments().size() == 1 && !outputs
              ? source
              : b.View(source, ir::ShapeOf({width, rows}),
                       {nb[0], nb[1], nb[1] * rows, nb[1] * rows}, first * nb[1]);
      b.SetOwner(static_cast<std::uint8_t>(s));
      b.CacheStore(leaf, mine, outputs ? ctx.OutputCells(s, key) : ctx.StoreCells(s, key));
      b.SetJoined();
      first += rows;
    }
    return {};
  };
  // The head reads only its output rows: the stage's last component that
  // mixes rows narrows its output to them when it can (the Gemma builders'
  // narrow_final), and what follows it, row-local, runs at their width;
  // unless the options keep every row (a policy without narrow_final).
  const std::optional<std::pair<std::size_t, std::size_t>> narrow_at =
      head && options.narrow ? NarrowPoint(*instances) : std::nullopt;
  bool narrowed = false;
  for (std::size_t l = 0; l < instances->layers.size(); ++l) {
    flow.carry = {};  // a layer's own
    flow.carry2 = {};
    for (std::size_t c = 0; c < instances->layers[l].size(); ++c) {
      const Instance& in = instances->layers[l][c];
      const auto& st = stateful[l][c];
      if (!needed[l][c] && !st) continue;
      std::vector<ir::TensorId> state;
      if (st) state = sb.State(in, st->contract, st->first, st->cells, st->items, st->sources);
      const bool narrow = narrow_at == std::pair{l, c};
      // Whether a later component of the layer runs to read its carry.
      carry_needed = std::any_of(needed[l].begin() + static_cast<std::ptrdiff_t>(c) + 1,
                                 needed[l].end(), [](bool b) { return b; });
      auto out = emit(in, flow, state, st ? st->contract.tensors.size() : 0, needed[l][c], narrow,
                      narrowed);
      if (!out) {
        return Refused(std::format("layer {} {}: {}", l, in.def->kind, out.error()));
      }
      follow(flow, *out);
      narrowed = narrowed || narrow;
    }
    for (const TapWrite& tap : *taps) {
      if (tap.tap.point == spec::TapPoint::kPostFinalNorm || tap.tap.layer != l) continue;
      ir::TensorId source = flow.stream;
      if (tap.tap.point == spec::TapPoint::kStreams ||
          tap.tap.point == spec::TapPoint::kStreamMean) {
        if (!flow.residual.valid()) return Refused("a streams tap of a stage of one stream");
        sb.b().SetComponent(kFrameworkComponent);
        source = TapOfStreams(sb.b(), flow.residual, tap.tap.point);
      }
      if (narrowed) return Refused("a layer's tap after the step narrowed its rows");
      if (auto ok = write_tap(tap, source, false); !ok) return std::unexpected(ok.error());
    }
    // The layer's taps an injection reads, kept for it.
    for (const Injection& inj : *injections) {
      for (const spec::FeatureTap& tap : inj.taps) {
        if (tap.layer != l || captured.contains(tap.name)) continue;
        if (narrowed) return Refused("an injection's tap after the step narrowed its rows");
        const bool streams = tap.point != spec::TapPoint::kLayerOut;
        if (streams && !flow.residual.valid()) {
          return Refused("a streams tap of a stage of one stream");
        }
        captured.emplace(tap.name, streams ? flow.residual : flow.stream);
      }
    }
  }
  carry_needed = true;
  ir::TensorId logits{};
  ir::TensorId greedy{};
  ir::TensorId features{};
  if (head) {
    if (instances->final_norm) {
      auto out = emit(*instances->final_norm, flow, {}, 0, true, false, narrowed);
      if (!out) return Refused(std::format("final norm: {}", out.error()));
      follow(flow, *out);
      narrowed = narrowed || out->narrowed;
      features = out->features;
    }
    auto out = emit(*instances->output, flow, {}, 0, true, false, narrowed);
    if (!out) return Refused(std::format("output: {}", out.error()));
    logits = out->logits;
    greedy = out->greedy;
    if (out->features.valid()) features = out->features;
    sb.b().Output(logits);
    if (greedy.valid()) sb.b().Output(greedy);
    if (features.valid()) sb.b().Output(features);
    // The final norm's taps: its output rows, stored at their positions'
    // cells of each reader's tensor.
    ir::TensorId rows{};
    for (const TapWrite& tap : *taps) {
      if (tap.tap.point != spec::TapPoint::kPostFinalNorm) continue;
      if (!instances->final_norm) return Refused("a final norm's tap of a stage without one");
      // The output rows in segment order: the stream itself when the step
      // narrowed it to them (its rows are no longer the step's rows).
      if (!rows.valid()) {
        sb.b().SetComponent(kFrameworkComponent);
        ctx.SetComponent(kFrameworkComponent);
        rows = narrowed ? flow.stream : sb.b().GetRows(flow.stream, ctx.OutputRows());
      }
      if (auto ok = write_tap(tap, rows, true); !ok) return std::unexpected(ok.error());
    }
  }
  // Each injecting drafter's injection, after the stage's own work (as
  // llama.cpp's target graph runs its drafter's part last): its embedding's
  // from the taps, then each of its layers' injecting components'.
  for (const Injection& inj : *injections) {
    const std::uint16_t d = inj.drafter;
    const spec::Stage& drafter = spec.stages[d];
    const model::StateLayoutDerived* layout = LayoutOf(bindings, d);
    if (layout == nullptr) {
      return Refused(std::format("an injection into stage {} without its layout", drafter.name));
    }
    if (drafter.artifact >= bindings.artifacts.size()) {
      return Refused(std::format("an injection into stage {} without its tensors", drafter.name));
    }
    TensorTables at_tables{.own = bindings.artifacts[drafter.artifact], .stages = {}};
    for (std::size_t i = 0; i < spec.stages.size(); ++i) {
      if (i != d && spec.stages[i].artifact < bindings.artifacts.size()) {
        at_tables.stages.emplace(spec.stages[i].name, bindings.artifacts[spec.stages[i].artifact]);
      }
    }
    // A tap's value, made when the injection first reads it: a layer's
    // output or streams as kept, its streams' mean (each row's streams
    // summed in order, then divided).
    std::map<std::string, ir::TensorId, std::less<>> values;
    std::string failure;
    const std::function<ir::TensorId(std::string_view)> tap_value =
        [&](std::string_view name) -> ir::TensorId {
      if (const auto at = values.find(name); at != values.end()) return at->second;
      const auto tap = std::ranges::find(inj.taps, name, &spec::FeatureTap::name);
      const auto kept = captured.find(name);
      if (tap == inj.taps.end() || kept == captured.end()) {
        failure = std::format("an injection's tap {} was not kept", name);
        return {};
      }
      ir::Builder& b = sb.b();
      b.SetComponent(kFrameworkComponent);
      const ir::TensorId value = tap->point == spec::TapPoint::kLayerOut
                                     ? kept->second
                                     : TapOfStreams(b, kept->second, tap->point);
      values.emplace(std::string(name), value);
      return value;
    };
    const Instance& e = *inj.instances.embedding;
    auto input =
        emit_at(d, at_tables, e.def->inject, e, Flow{}, {}, 0, true, false, false, tap_value);
    if (!failure.empty()) return Refused(failure);
    if (!input) return Refused(std::format("stage {} injection: {}", drafter.name, input.error()));
    if (!input->stream.valid()) {
      return Refused(std::format("stage {}'s injection made no input", drafter.name));
    }
    const std::uint32_t kept_rows = e.def->inject_rows(e);
    for (std::size_t l = 0; l < inj.instances.layers.size(); ++l) {
      std::uint16_t first = 0;
      if (l == 0 && e.def->state != nullptr) {
        first = static_cast<std::uint16_t>(e.def->state(e).tensors.size());
      }
      for (const Instance& in : inj.instances.layers[l]) {
        const model::StateContract contract =
            in.def->state != nullptr ? in.def->state(in) : model::StateContract{};
        // Its injection writes the state its target writes (none: it has no
        // injection, as a kind of component may inject in one form only).
        const bool injects =
            in.def->inject != nullptr &&
            std::ranges::any_of(contract.tensors, &model::StateTensorContract::target_writes);
        if (injects) {
          std::vector<ir::TensorId> leaves(ctx.segments().size() * contract.tensors.size());
          for (std::size_t t = 0; t < contract.tensors.size(); ++t) {
            const model::StateTensorContract& c = contract.tensors[t];
            const auto index = static_cast<std::uint16_t>(first + t);
            const auto item = ItemOf(*layout, in.layer, index);
            if (!c.target_writes || !item || layout->items[*item].contract != c) {
              return Refused(std::format("stage {} layer {}: its injected state is not this spec's",
                                         drafter.name, in.layer));
            }
            const std::uint64_t cells = layout->items[*item].cells;
            if (c.cells == model::CellRule::kPositionRing && cells < kept_rows) {
              return Refused(std::format("stage {} layer {}: a ring of {} cells for {} rows",
                                         drafter.name, in.layer, cells, kept_rows));
            }
            for (std::size_t sg = 0; sg < ctx.segments().size(); ++sg) {
              leaves[(sg * contract.tensors.size()) + t] =
                  sb.StateLeaf(d, in.layer, index, c, sg, cells, *item, true);
            }
          }
          const Flow injected{.stream = input->stream};
          auto out = emit_at(d, at_tables, in.def->inject, in, injected, leaves,
                             contract.tensors.size(), true, false, false, tap_value);
          if (!out) {
            return Refused(std::format("stage {} layer {} {} injection: {}", drafter.name, l,
                                       in.def->kind, out.error()));
          }
        }
        first = static_cast<std::uint16_t>(first + contract.tensors.size());
      }
    }
  }
  auto step = std::move(sb).Finish(logits, greedy, features);
  if (!step) return step;
  // A verify writes none of its stage's state that rolls back by commit:
  // a rejected row must leave no trace in it (its commit writes it).
  if (shape.phase == planner::Phase::kVerify) {
    for (const StepState& st : step->state) {
      if (!st.written || st.ref.stage != stage || st.borrowed || bindings.state == nullptr ||
          st.item >= bindings.state->items.size()) {
        continue;
      }
      const model::StateTensorContract& c = bindings.state->items[st.item].contract;
      if (c.rollback == model::Rollback::kCommit) {
        return Refused(std::format("a verify writes layer {} state {}, which its commit writes",
                                   st.ref.layer, c.name));
      }
    }
  }
  return step;
}

std::expected<StepGraph, std::string> BuildCommit(const spec::ModelSpec& spec, std::uint16_t stage,
                                                  const planner::StepShape& kept,
                                                  const Bindings& bindings) {
  if (kept.stage != stage) return Refused("a shape of another stage");
  if (auto checked = CheckShape(kept); !checked) return std::unexpected(checked.error());
  if (kept.outputs() != 0) return Refused("a commit returns no outputs");
  auto instances = InstancesOf(spec, stage);
  if (!instances) return std::unexpected(instances.error());
  if (bindings.state == nullptr) return Refused("a commit without its state layout");
  StepBuilder sb(stage, kept);
  StepContext& ctx = sb.ctx();
  bool any = false;
  for (std::size_t l = 0; l < instances->layers.size(); ++l) {
    std::uint16_t first = 0;
    if (l == 0 && instances->embedding && instances->embedding->def->state != nullptr) {
      first = static_cast<std::uint16_t>(
          instances->embedding->def->state(*instances->embedding).tensors.size());
    }
    for (const Instance& in : instances->layers[l]) {
      if (in.def->state == nullptr) continue;
      const model::StateContract contract = in.def->state(in);
      if (in.def->commit != nullptr) {
        std::vector<std::uint64_t> cells;
        std::vector<std::size_t> items;
        for (std::size_t t = 0; t < contract.tensors.size(); ++t) {
          const auto item =
              ItemOf(*bindings.state, in.layer, static_cast<std::uint32_t>(first + t));
          if (!item || bindings.state->items[*item].contract != contract.tensors[t]) {
            return Refused(std::format("layer {}: the state layout is not this spec's", in.layer));
          }
          items.push_back(*item);
          cells.push_back(bindings.state->items[*item].cells);
        }
        const std::vector<ir::TensorId> state = sb.State(in, contract, first, cells, items);
        sb.Begin(in);
        Fragment f{.b = sb.b(),
                   .instance = in,
                   .step = ctx,
                   .stream = {},
                   .weight = [](std::uint16_t) -> ir::TensorId {
                     base::Check(false, "a commit reads no weights");
                     return {};
                   },
                   .state = state,
                   .state_tensors = contract.tensors.size(),
                   .stream_needed = false,
                   .narrow = false,
                   .narrowed = false,
                   .residual = {},
                   .carry = {},
                   .side = {}};
        if (auto ok = in.def->commit(f); !ok) {
          return Refused(std::format("layer {} {}: {}", l, in.def->kind, ok.error()));
        }
        sb.b().SetJoined();
        any = true;
      }
      first = static_cast<std::uint16_t>(first + contract.tensors.size());
    }
  }
  if (!any) return Refused("a stage with nothing to commit");
  return std::move(sb).Finish({}, {}, {});
}

std::expected<ComponentGraph, std::string> BuildComponent(
    const Instance& in, const planner::StepShape& shape,
    std::span<const format::TensorFormat> formats, std::span<const std::uint64_t> cells,
    const ComponentInputs& inputs) {
  if (auto checked = CheckShape(shape); !checked) return std::unexpected(checked.error());
  const auto shapes = in.def->shapes(in);
  if (!shapes) return std::unexpected(shapes.error());
  // A format for every role it uses (roles past `formats`, which a later
  // form of the component added, unused).
  if (formats.size() > in.def->roles.size()) return Refused("a format for no role");
  std::vector<bool> used(in.def->roles.size(), false);
  for (const RoleShape& r : *shapes) {
    if (r.role >= formats.size()) return Refused("a format for every role it uses");
    const RoleFormat f{
        .format = formats[r.role],
        .parts = {{std::format("{}", in.def->roles[r.role].name), format::PartRole::kData}},
        .stage = {}};
    if (auto checked = CheckRole(in, r, f); !checked) return std::unexpected(checked.error());
    used[r.role] = true;
  }
  StepBuilder sb(shape.stage, shape);
  StepContext& ctx = sb.ctx();
  ctx.EnableOwnerProducts(inputs.owner_products);
  model::StateContract contract;
  if (in.def->state != nullptr) contract = in.def->state(in);
  if (cells.size() != contract.tensors.size()) return Refused("cells for every state tensor");
  std::vector<StepContext::MaskKey> masks;
  for (std::size_t t = 0; t < contract.tensors.size(); ++t) {
    // A tap another stage writes: the instance's own steps never touch it.
    if (contract.tensors[t].target_writes) continue;
    for (const SegmentRows& seg : ctx.segments()) {
      if (auto c = CheckCells(seg, KeyOf(contract.tensors[t], cells[t]),
                              contract.tensors[t].window_or_ratio);
          !c) {
        return std::unexpected(c.error());
      }
    }
  }
  if (in.def->masks != nullptr) {
    for (const MaskUse& m : in.def->masks(in)) {
      masks.push_back({.cells = KeyOf(contract.tensors.at(m.tensor), cells[m.tensor]),
                       .window = m.window,
                       .ahead = m.ahead,
                       .exact = in.def->exact_masks});
    }
  }
  ctx.DeclareMasks(masks);
  struct Ids {
    ir::TensorId stream_in, stream_out, residual_in, residual_out, carry_in, carry_out, carry2_in,
        carry2_out, side_in;
  } out{};
  std::vector<ir::TensorId> weights(in.def->roles.size());
  sb.Begin(in);
  const auto input = [&](std::uint64_t width, std::uint64_t streams, std::string_view name,
                         format::Encoding encoding = format::Encoding::kF32) {
    const format::TensorFormat f =
        ir::Dense(encoding, streams > 1 ? ir::ShapeOf({width, streams, ctx.rows()})
                                        : ir::ShapeOf({width, ctx.rows()}));
    sb.b().SetComponent(kFrameworkComponent);
    const ir::TensorId id = sb.b().Input(f, name);
    sb.b().SetComponent(in.index);
    return id;
  };
  if (in.site != Site::kEmbedding && in.site != Site::kSide) {
    out.stream_in = input(in.stage.width, 1, "stream");
  }
  if (inputs.residual) out.residual_in = input(in.stage.width, in.stage.streams, "residual");
  if (inputs.carry != 0) out.carry_in = input(inputs.carry, 1, "carry");
  if (inputs.carry2 != 0) out.carry2_in = input(in.stage.streams, in.stage.streams, "carry2");
  if (inputs.side != 0) {
    out.side_in = input(inputs.side, 1, "side",
                        inputs.side_ids ? format::Encoding::kI32 : format::Encoding::kF32);
  }
  std::vector<ir::TensorId> state;
  if (!contract.tensors.empty()) state = sb.State(in, contract, 0, cells, {});
  Fragment f{.b = sb.b(),
             .instance = in,
             .step = ctx,
             .stream = out.stream_in,
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
             .stream_needed = contract.tensors.empty() || ctx.outputs() > 0,
             .residual = out.residual_in,
             .carry = out.carry_in,
             .carry2 = out.carry2_in,
             .side = [&](std::string_view) -> ir::TensorId { return out.side_in; }};
  auto emitted = in.def->emit(f);
  if (!emitted) return std::unexpected(emitted.error());
  sb.b().SetJoined();
  if (emitted->logits.valid()) {
    sb.b().Output(emitted->logits);
    if (emitted->greedy.valid()) sb.b().Output(emitted->greedy);
    if (emitted->features.valid()) sb.b().Output(emitted->features);
  } else if (f.stream_needed) {
    if (emitted->features.valid()) sb.b().Output(emitted->features);
    if (emitted->stream.valid() && emitted->stream != out.stream_in) {
      out.stream_out = emitted->stream;
      sb.b().Output(out.stream_out);
    }
    if (emitted->residual.valid()) {
      out.residual_out = emitted->residual;
      sb.b().Output(out.residual_out);
    }
    if (emitted->carry.valid()) {
      out.carry_out = emitted->carry;
      sb.b().Output(out.carry_out);
    }
    if (emitted->carry2.valid()) {
      out.carry2_out = emitted->carry2;
      sb.b().Output(out.carry2_out);
    }
  }
  auto step = std::move(sb).Finish(emitted->logits, emitted->greedy, emitted->features);
  if (!step) return std::unexpected(step.error());
  std::vector<StepInput> lone;
  for (const auto& [id, kind] :
       {std::pair{out.stream_in, InputKind::kStream},
        std::pair{out.residual_in, InputKind::kResidual},
        std::pair{out.carry_in, InputKind::kCarry}, std::pair{out.carry2_in, InputKind::kCarry},
        std::pair{out.side_in, InputKind::kSide}}) {
    if (id.valid()) lone.push_back(StepInput{.tensor = id, .kind = kind});
  }
  step->inputs.insert(step->inputs.begin(), lone.begin(), lone.end());
  return ComponentGraph{.step = std::move(*step),
                        .stream_in = out.stream_in,
                        .stream_out = out.stream_out,
                        .residual_in = out.residual_in,
                        .residual_out = out.residual_out,
                        .carry_in = out.carry_in,
                        .carry_out = out.carry_out,
                        .carry2_in = out.carry2_in,
                        .carry2_out = out.carry2_out,
                        .side_in = out.side_in,
                        .weights = std::move(weights)};
}

std::expected<std::vector<std::int64_t>, std::string> InputValues(
    const StepInput& input, const planner::StepShape& shape, std::span<const std::int32_t> tokens) {
  std::vector<std::int64_t> out;
  std::uint32_t rows = 0;
  for (const planner::Segment& s : shape.segments) rows += s.rows;
  switch (input.kind) {
    case InputKind::kStream:
    case InputKind::kResidual:
    case InputKind::kCarry:
    case InputKind::kSide:
      return Refused("a lone component's stream, streams, carry or side input are the caller's");
    case InputKind::kFeatures:
      return Refused("a feature's values are the caller's");
    case InputKind::kSideRows:
      return Refused("a side table's rows are the caller's (its source's lookup)");
    case InputKind::kCommitRows:
      return Refused("a verify's saved rows are the caller's");
    case InputKind::kVisibleCells: {
      const auto& c = input.cells;
      constexpr auto max = static_cast<std::uint64_t>(INT32_MAX);
      if (input.segment >= shape.segments.size() || c.capacity == 0 || c.capacity > max ||
          c.block == 0 || c.block > max || !c.positions || c.ring || c.frozen || c.fixed ||
          c.per_row || c.first_row != 0 || c.recent || c.cell_pad == 0 ||
          input.visible_capacity == 0 || input.visible_capacity > c.capacity || input.read == 0 ||
          input.read > c.capacity || input.sections != 1 || input.ratio != 0 || input.overlap ||
          input.ring != 0) {
        return Refused(
            "visible counts require zero-based positions blocks and bounded real/scored cells");
      }
      const auto& segment = shape.segments[input.segment];
      if (segment.rows == 0) return Refused("visible counts of no rows");
      for (std::uint32_t row = 0; row < segment.rows; ++row) {
        out.push_back(
            static_cast<std::int64_t>(std::min((std::uint64_t{segment.past} + row + 1) / c.block,
                                               std::min(input.visible_capacity, input.read))));
      }
      return out;
    }
    case InputKind::kTokens:
      if (tokens.size() != rows) return Refused("a token per row");
      out.assign(tokens.begin(), tokens.end());
      return out;
    case InputKind::kPositions:
      for (std::uint8_t sec = 0; sec < std::max<std::uint8_t>(input.sections, 1); ++sec) {
        for (const planner::Segment& s : shape.segments) {
          for (std::uint32_t i = 0; i < s.rows; ++i) out.push_back(std::int64_t{s.past} + i);
        }
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
      // A fixed tensor's one row.
      if (input.cells.fixed) return std::vector<std::int64_t>{0};
      // A cell a row of the whole step (a verify's joined saves).
      if (input.cells.per_row && input.segment == ir::kJoinedSegment) {
        for (std::uint32_t i = 0; i < rows; ++i)
          out.push_back(std::int64_t{input.cells.first_row} + i);
        return out;
      }
      break;
    case InputKind::kHadamard:
      return Refused("a Hadamard matrix's values are floats (HadamardValues)");
    case InputKind::kPositionPhases:
      if (input.ratio == 0) return Refused("phases of no ratio");
      for (const planner::Segment& s : shape.segments) {
        for (std::uint32_t i = 0; i < s.rows; ++i) {
          out.push_back(static_cast<std::int64_t>((std::uint64_t{s.past} + i) % input.ratio));
        }
      }
      return out;
    case InputKind::kCellPositions:
    case InputKind::kOutputCells:
    case InputKind::kBlockPositions:
    case InputKind::kKeptRows:
    case InputKind::kKeptCells:
    case InputKind::kPoolReads:
    case InputKind::kBlockCells:
    case InputKind::kBlockStarts:
      break;
  }
  if (input.kind == InputKind::kBlockStarts || input.kind == InputKind::kPoolReads ||
      input.kind == InputKind::kBlockCells) {
    if (input.segment >= shape.segments.size()) return Refused("an input of no segment");
    const planner::Segment& s = shape.segments[input.segment];
    // kBlockStarts names no cache: any cell count past the blocks plans it.
    const std::uint64_t cache = input.kind == InputKind::kBlockCells
                                    ? input.cells.capacity
                                    : std::numeric_limits<std::uint32_t>::max();
    const std::uint64_t ring = input.kind == InputKind::kPoolReads
                                   ? input.cells.capacity
                                   : std::uint64_t{input.ratio} * (input.overlap ? 2 : 1);
    auto plan = PlanCompressor(s.rows, s.past, input.ratio, input.overlap, ring, cache);
    if (!plan) return std::unexpected(plan.error());
    switch (input.kind) {
      case InputKind::kPoolReads:
        return std::move(plan->reads);
      case InputKind::kBlockCells:
        return std::move(plan->cells);
      default:
        return std::move(plan->starts);
    }
  }
  if (input.segment >= shape.segments.size() || input.cells.capacity == 0) {
    return Refused("an input of no segment of the shape");
  }
  const planner::Segment& s = shape.segments[input.segment];
  const std::uint64_t cap = input.cells.capacity;
  if (input.kind == InputKind::kBlockPositions) {
    if (input.cells.block == 0) return Refused("block positions of a tensor of no blocks");
    for (std::uint8_t sec = 0; sec < std::max<std::uint8_t>(input.sections, 1); ++sec) {
      for (std::uint64_t b = 0; b < input.read; ++b) {
        out.push_back(static_cast<std::int64_t>(b * input.cells.block));
      }
    }
    return out;
  }
  if (input.kind == InputKind::kKeptRows || input.kind == InputKind::kKeptCells) {
    // A recent ring's last min(rows, capacity) rows, in row order.
    const std::uint64_t kept = std::min<std::uint64_t>(s.rows, cap);
    for (std::uint64_t i = s.rows - kept; i < s.rows; ++i) {
      out.push_back(input.kind == InputKind::kKeptRows
                        ? static_cast<std::int64_t>(i)
                        : static_cast<std::int64_t>((std::uint64_t{s.past} + i) % cap));
    }
    return out;
  }
  if (input.kind == InputKind::kStoreCells && input.cells.block != 0) {
    // A block tensor's first `read` blocks, each at its own cell.
    if (input.read > cap) return Refused("blocks past a block tensor's cells");
    for (std::uint64_t b = 0; b < input.read; ++b) out.push_back(static_cast<std::int64_t>(b));
    return out;
  }
  if (input.kind == InputKind::kStoreCells || input.kind == InputKind::kOutputCells) {
    if (input.cells.block != 0) return Refused("a block tensor's cells are not a row's");
    const std::uint32_t from = input.kind == InputKind::kStoreCells ? 0 : s.rows - s.outputs;
    for (std::uint32_t i = from; i < s.rows; ++i) {
      if (input.cells.per_row) {
        out.push_back(std::int64_t{input.cells.first_row} +
                      (input.kind == InputKind::kStoreCells ? i : i - from));
        continue;
      }
      const std::uint64_t p = std::uint64_t{s.past} + i;
      out.push_back(static_cast<std::int64_t>(input.cells.ring ? p % cap : p));
    }
    return out;
  }
  // A frozen read's cells hold the positions before the segment's.
  const std::uint64_t end = std::uint64_t{s.past} + (input.cells.frozen ? 0 : s.rows);
  for (std::uint64_t c = 0; c < input.read; ++c) {
    if (input.cells.block != 0) {
      // A block's cell holds its last position once the block is complete.
      const std::uint64_t last = (c * input.cells.block) + input.cells.block - 1;
      out.push_back(last < end ? static_cast<std::int64_t>(last) : -1);
    } else if (c >= end) {
      out.push_back(-1);
    } else if (input.cells.ring) {
      out.push_back(static_cast<std::int64_t>(c + (((end - 1 - c) / cap) * cap)));
    } else {
      out.push_back(static_cast<std::int64_t>(c));
    }
  }
  return out;
}

std::expected<void, std::string> VisitWeightBindings(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors,
    const WeightVisitor& visitor) {
  if (!visitor) return Refused("weight inventory needs a visitor");
  if (auto checked = ValidateSpec(spec, tensors); !checked) return checked;
  for (std::uint16_t stage = 0; stage != spec.stages.size(); ++stage) {
    auto instances = InstancesOf(spec, stage);
    if (!instances) return std::unexpected(instances.error());
    const auto tables = TablesOf(spec, stage, tensors, tensors[spec.stages[stage].artifact]);
    for (const auto* in : All(*instances)) {
      auto formats = CheckedRoleFormats(*in, tables);
      if (!formats) return std::unexpected(formats.error());
      for (std::size_t role = 0; role != formats->size(); ++role) {
        const auto& f = (*formats)[role];
        if (!f) continue;
        auto ref = ResolvedWeightRef(spec, stage, *in, static_cast<std::uint16_t>(role), *f);
        if (!ref) return std::unexpected(ref.error());
        const bool rows = in->def->roles[role].rows_on_demand;
        if (rows && !in->def->side_count) return Refused("row-paged role has no row count");
        const auto selected_rows =
            in->def->kind == "head.linear" && in->def->roles[role].name == "weight"
                ? static_cast<std::uint64_t>(in->params.Int("rows"))
                : 0;
        const WeightBindingView binding{
            .ref = *ref,
            .artifact = spec.stages[ref->stage].artifact,
            .format = f->format,
            .parts = f->parts,
            .rows_per_token = rows ? in->def->side_count(*in) : 0,
            .head_rows = in->site == Site::kOutput && in->def->roles[role].name == "weight"
                             ? (selected_rows ? selected_rows : f->format.shape.ne[1])
                             : 0,
            .use_stage = stage,
            .use_component = in->index,
            .use_role = static_cast<std::uint16_t>(role)};
        if (auto result = visitor(binding); !result) return result;
      }
    }
  }
  return {};
}

}  // namespace llmp::components
