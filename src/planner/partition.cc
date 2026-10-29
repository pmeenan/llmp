// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/partition.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/check.h"
#include "components/component.h"
#include "components/step.h"
#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/op_traits.h"
#include "model/state_contract.h"
#include "spec/spec.h"

namespace llmp::planner {
namespace {

using components::Instance;
using components::SplitKind;
using components::SplitOption;
using ir::IsLeaf;
using ir::Op;
using ir::TensorId;

constexpr double kInf = std::numeric_limits<double>::infinity();
// Existing exact planner work contract, shared by work derivation and search.
inline constexpr std::uint64_t kMaxPartitionWork = std::uint64_t{1} << 24U;

std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}

std::vector<const Instance*> All(const components::StageInstances& s) {
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

void Add(Traffic& a, const Traffic& b) {
  a.bytes += b.bytes;
  a.seconds += b.seconds;
}

// ------------------------------------------------------------------ topology

// A validated topology by slot (its devices' order).
class Net {
 public:
  explicit Net(const Topology& t) : n_(t.devices.size()), links_(n_ * n_), topology_(t) {
    for (const Device& d : t.devices) {
      ids_.push_back(d.id);
      memory_.push_back(d.memory);
    }
    for (const Link& l : t.links) {
      const std::size_t a = *Slot(l.a);
      const std::size_t b = *Slot(l.b);
      links_[(a * n_) + b] = l;
      links_[(b * n_) + a] = l;
    }
  }
  std::size_t size() const { return n_; }
  std::optional<std::size_t> Slot(std::uint32_t id) const {
    for (std::size_t i = 0; i < n_; ++i) {
      if (ids_[i] == id) return i;
    }
    return std::nullopt;
  }
  std::uint32_t Id(std::size_t slot) const { return ids_[slot]; }
  std::uint64_t Memory(std::size_t slot) const { return memory_[slot]; }
  bool Execution() const { return topology_.cost_model == CostModel::kExecution; }
  const std::optional<DevicePerformance>& Performance(std::size_t slot) const {
    return topology_.devices[slot].performance;
  }
  const std::optional<Link>& LinkOf(std::size_t a, std::size_t b) const {
    return links_[(a * n_) + b];
  }
  // A transfer of `bytes` from a to b: bandwidth plus latency; none without
  // a link.
  double Seconds(std::size_t a, std::size_t b, double bytes) const {
    if (a == b) return 0.0;
    const auto& l = LinkOf(a, b);
    if (!l) return kInf;
    return (bytes / l->bytes_per_s) + l->latency_s;
  }

 private:
  std::size_t n_;
  std::vector<std::uint32_t> ids_;
  std::vector<std::uint64_t> memory_;
  std::vector<std::optional<Link>> links_;
  const Topology& topology_;
};

// A ring of `steps` steps over `group` in its order, each member sending
// 1/g of `bytes` to the next: as slow as its slowest link.
Traffic Ring(const Net& net, std::span<const std::size_t> group, std::uint64_t bytes,
             std::uint32_t steps) {
  const std::size_t g = group.size();
  if (g < 2 || steps == 0) return {};
  double worst = 0.0;
  for (std::size_t i = 0; i < g; ++i) {
    worst = std::max(worst, net.Seconds(group[i], group[(i + 1) % g],
                                        static_cast<double>(bytes) / static_cast<double>(g)));
  }
  return {.bytes = bytes * steps, .seconds = worst * steps};
}

// A collective joining a tensor of `bytes` over `group`.
Traffic Collective(const Net& net, Op op, std::span<const std::size_t> group, std::uint64_t bytes) {
  const auto g = static_cast<std::uint32_t>(group.size());
  if (g < 2) return {};
  switch (op) {
    case Op::kAllReduce:
      return Ring(net, group, bytes, 2 * (g - 1));
    case Op::kAllGather:
    case Op::kAllToAll:
      return Ring(net, group, bytes, g - 1);
    default:
      return {};
  }
}

// The holder a tensor of `bytes` goes to `to` from: the cheapest link,
// the first among equals; none if no holder links to it.
std::optional<std::size_t> BestHolder(const Net& net, std::span<const std::size_t> holders,
                                      std::size_t to, double bytes) {
  std::optional<std::size_t> best;
  double cost = kInf;
  for (const std::size_t h : holders) {
    const double c = net.Seconds(h, to, bytes);
    if (c < cost) {
      cost = c;
      best = h;
    }
  }
  return best;
}

// A tensor of `bytes` held by `holders` sent to each of `targets` in order
// (each receiver then a holder); seconds infinite when one is unreachable.
Traffic Relay(const Net& net, std::vector<std::size_t> holders,
              std::span<const std::size_t> targets, std::uint64_t bytes) {
  Traffic t;
  for (const std::size_t to : targets) {
    if (std::ranges::find(holders, to) != holders.end()) continue;
    const auto from = BestHolder(net, holders, to, static_cast<double>(bytes));
    if (!from) return {.bytes = t.bytes, .seconds = kInf};
    Add(t, {.bytes = bytes, .seconds = net.Seconds(*from, to, static_cast<double>(bytes))});
    holders.push_back(to);
  }
  return t;
}

double Weighted(const Workload& w, const Traffic& prefill, const Traffic& decode) {
  return (w.decode_share * decode.seconds) + ((1.0 - w.decode_share) * prefill.seconds);
}

// --------------------------------------------------------------- the model

std::expected<std::uint64_t, std::string> FormatBytes(const format::TensorFormat& f) {
  const auto parts = format::PartsOf(f);
  if (!parts) return Refused("a weight format without a size");
  std::uint64_t n = 0;
  for (const format::PartExtent& p : parts->parts()) {
    if (p.bytes > std::numeric_limits<std::uint64_t>::max() - n)
      return Refused("weight byte count overflows");
    n += p.bytes;
  }
  return n;
}

format::TensorFormat Local(format::TensorFormat f, std::uint8_t dim, std::uint32_t parts) {
  f.shape.ne.at(dim) /= parts;
  return f;
}

struct WeightUse {
  std::string name{};
  std::uint16_t role = 0;
  format::TensorFormat format{};
  std::uint64_t bytes = 0;
};

// Work in an actual local primitive graph: bytes include source reads and
// output writes. Structural aliases do no work. Product units are FLOPs;
// other operations use typed output-element rates, including their parameters.
struct WorkTerm {
  ir::OpKey op{};
  std::array<std::byte, ir::kMaxParamBytes> params{};
  format::Encoding encoding = format::Encoding::kF32;
  ir::Accumulate accumulate = ir::Accumulate::kDefault;
  double units = 0;
  double bytes = 0;
};
using RankWork = std::vector<std::vector<WorkTerm>>;
struct ComponentWork {
  std::array<RankWork, 2> phase{};
};

// What the partitioner needs of an instance.
struct Info {
  const Instance* in = nullptr;
  std::vector<format::Shape> shapes{};  // by role; rank 0 where unused
  std::vector<WeightUse> weights{};     // the roles in use
  model::StateContract contract{};
  std::vector<std::uint64_t> state{};   // per contract tensor, bytes per slot
  std::vector<std::uint64_t> values{};  // per contract tensor, values per cell
  std::uint16_t state_first = 0;        // in its layer's contract
  // It runs after the step's narrowing (components::NarrowPoint), so with a
  // head it works on the step's output rows, not its rows.
  bool narrowed = false;
  std::uint64_t side_bytes = 0;  // a side source's output bytes per row
  std::map<std::pair<std::uint32_t, std::uint8_t>, ComponentWork> work{};
};

// With `state`, its state's bytes for the workload too.
std::expected<Info, std::string> InfoOf(const StageModel& m, const Instance& in, const Workload& w,
                                        std::uint16_t state_first, bool state) {
  Info info{.in = &in, .state_first = state_first};
  if (in.def->splits.size() > kWhole) return Refused("too many split options for a placement");
  const auto shapes = in.def->shapes(in);
  if (!shapes) return Refused(std::format("{}: {}", in.def->kind, shapes.error()));
  info.shapes.assign(in.def->roles.size(), format::Shape{.ne = {1, 1, 1, 1}, .rank = 0});
  if (in.index >= m.roles.size()) {
    return Refused(std::format("{}: no bound weights", in.def->kind));
  }
  const auto& bound = m.roles[in.index];
  for (const components::RoleShape& r : *shapes) {
    if (r.role >= bound.size() || bound[r.role].name.empty()) {
      return Refused(
          std::format("{}: role {} is unbound", in.def->kind, in.def->roles[r.role].name));
    }
    const RoleTensor& t = bound[r.role];
    auto bytes = FormatBytes(t.format);
    if (!bytes) return std::unexpected(bytes.error());
    info.shapes[r.role] = t.format.shape;
    info.weights.push_back({.name = t.name, .role = r.role, .format = t.format, .bytes = *bytes});
  }
  if (in.def->side_source) {
    if (!in.def->row_local) return Refused("a side source must be row-local for partitioning");
    std::vector<format::TensorFormat> formats(in.def->roles.size());
    for (const auto& weight : info.weights) formats[weight.role] = weight.format;
    const StepShape one{.stage = m.stage, .segments = {{.rows = 1, .outputs = 1}}};
    auto fragment = components::BuildComponent(in, one, formats, {});
    if (!fragment) return Refused(std::format("{}: {}", in.def->kind, fragment.error()));
    if (!fragment->stream_out.valid()) return Refused("a side source without an output");
    auto bytes = FormatBytes(fragment->step.graph.tensor(fragment->stream_out).format);
    if (!bytes) return std::unexpected(bytes.error());
    info.side_bytes = *bytes;
  }
  if (in.def->state != nullptr) info.contract = in.def->state(in);
  if (in.def->state != nullptr && state) {
    const std::array per{std::pair{in.layer, info.contract}};
    const std::uint32_t context = std::max(w.context, 1U);
    const std::uint32_t rows = std::min(std::max({w.prefill_rows, w.decode_rows, 1U}), context);
    auto layout = model::Layout(per, context, rows);
    if (!layout) return Refused(std::format("{}: {}", in.def->kind, layout.error()));
    info.state.assign(info.contract.tensors.size(), 0);
    for (const auto& item : layout->items) info.state.at(item.tensor) = item.bytes;
    for (const auto& c : info.contract.tensors) {
      const std::uint64_t el = format::Info(c.encoding).element_bits / 8;
      info.values.push_back(el == 0 ? 0 : c.cell_bytes / el);
    }
  }
  return info;
}

const WeightUse* WeightOf(const Info& info, std::uint16_t role) {
  for (const WeightUse& w : info.weights) {
    if (w.role == role) return &w;
  }
  return nullptr;
}

// Whether `o` splits the instance `g` ways.
bool Fits(const Info& info, const SplitOption& o, std::uint32_t g) {
  if (g < 2) return false;
  if (o.units != nullptr) {
    const std::uint32_t u = o.units(*info.in);
    if (u == 0 || u % g != 0) return false;
  }
  for (const components::RoleSplit& rs : o.weights) {
    const WeightUse* w = WeightOf(info, rs.role);
    if (w == nullptr) continue;
    if (rs.dim >= format::kMaxDims || w->format.shape.ne.at(rs.dim) % g != 0) return false;
    const format::TensorFormat local = Local(w->format, rs.dim, g);
    if (!format::Validate(local) || !format::PartsOf(local)) return false;
  }
  for (const components::StateSplit& ss : o.state) {
    if (ss.tensor >= info.values.size() || ss.dim != 0 || info.values[ss.tensor] == 0 ||
        info.values[ss.tensor] % g != 0) {
      return false;
    }
  }
  return true;
}

// A part of a tensor a device holds: whole (parts 1) or part `part` of
// `parts` along `dim`.
struct Held {
  std::string name{};
  std::uint8_t dim = 0;
  std::uint32_t parts = 1;
  std::uint32_t part = 0;
  std::uint64_t bytes = 0;
};

// What an instance places on the group's rank-`rank` device: its weights
// and its state bytes (for `slots` slots).
bool Hold(const Info& info, const SplitOption* o, std::uint32_t g, std::uint32_t rank,
          std::uint32_t slots, std::vector<Held>& held, std::uint64_t& state) {
  for (const WeightUse& w : info.weights) {
    std::optional<std::uint8_t> dim;
    if (o != nullptr) {
      for (const components::RoleSplit& rs : o->weights) {
        if (rs.role == w.role) dim = rs.dim;
      }
    }
    if (!dim) {
      held.push_back({.name = w.name, .bytes = w.bytes});
      continue;
    }
    const auto bytes = FormatBytes(Local(w.format, *dim, g));
    if (!bytes) return false;
    held.push_back({.name = w.name, .dim = *dim, .parts = g, .part = rank, .bytes = *bytes});
  }
  for (std::size_t t = 0; t < info.state.size(); ++t) {
    bool split = false;
    if (o != nullptr) {
      for (const components::StateSplit& ss : o->state) split = split || ss.tensor == t;
    }
    const auto bytes = split ? info.state[t] / g : info.state[t];
    if (slots != 0 && bytes > (std::numeric_limits<std::uint64_t>::max() - state) / slots)
      return false;
    state += bytes * slots;
  }
  return true;
}

// A device's bytes: each tensor once (whole, or its distinct parts), and
// its state.
std::optional<std::uint64_t> Total(std::span<const Held> held, std::uint64_t state) {
  std::map<std::string,
           std::pair<std::uint64_t, std::map<std::tuple<std::uint8_t, std::uint32_t, std::uint32_t>,
                                             std::uint64_t>>>
      by;
  for (const Held& h : held) {
    auto& [whole, parts] = by[h.name];
    if (h.parts == 1) {
      whole = h.bytes;
    } else {
      parts[{h.dim, h.parts, h.part}] = h.bytes;
    }
  }
  std::uint64_t n = state;
  for (const auto& [name, v] : by) {
    if (v.first != 0) {
      if (v.first > std::numeric_limits<std::uint64_t>::max() - n) return std::nullopt;
      n += v.first;
    } else {
      for (const auto& [key, bytes] : v.second) {
        if (bytes > std::numeric_limits<std::uint64_t>::max() - n) return std::nullopt;
        n += bytes;
      }
    }
  }
  return n;
}

// The stream's bytes per step of `rows` rows (F32).
std::uint64_t StreamBytes(const StageModel& m, std::uint32_t rows) {
  const auto all = All(m.instances);
  const std::uint64_t width = all.empty() ? 0 : all.front()->stage.width;
  const std::uint64_t streams =
      all.empty() ? 1 : std::max<std::uint32_t>(1, all.front()->stage.streams);
  return width * streams * rows * 4;
}

// A pipeline boundary after the final row-mixing layer carries narrowed rows.
std::uint64_t BoundaryBytes(const StageModel& model, std::uint32_t first_layer, std::uint32_t rows,
                            std::uint32_t outputs) {
  const auto narrow = components::NarrowPoint(model.instances);
  return StreamBytes(model, narrow && first_layer > narrow->first ? outputs : rows);
}

// The modelled traffic of an option's collective over `group`.
Traffic OptionTraffic(const Net& net, const Info& info, const SplitOption& o,
                      std::span<const std::size_t> group, std::uint32_t rows,
                      std::uint32_t outputs) {
  if (o.bytes_per_step == nullptr) return {};
  const components::SplitStep step{.instance = info.in,
                                   .weights = info.shapes,
                                   .rows = info.narrowed && outputs > 0 ? outputs : rows,
                                   .outputs = outputs,
                                   .ways = static_cast<std::uint32_t>(group.size())};
  return Collective(net, o.collective, group, o.bytes_per_step(step));
}

double ExecutionSeconds(const Net& net, const Info& info, std::uint8_t choice,
                        std::span<const std::size_t> group, std::size_t phase) {
  if (!net.Execution()) return 0;
  const auto at = info.work.find({static_cast<std::uint32_t>(group.size()), choice});
  if (at == info.work.end()) return kInf;
  double bottleneck = 0;
  for (std::size_t rank = 0; rank < group.size(); ++rank) {
    const auto& performance = net.Performance(group[rank]);
    if (!performance) return kInf;
    double seconds = 0;
    for (const auto& term : at->second.phase[phase][rank]) {
      const auto rate = std::ranges::find_if(performance->rates, [&](const auto& r) {
        return r.op == term.op && r.params == term.params && r.encoding == term.encoding &&
               r.accumulate == term.accumulate;
      });
      if (rate == performance->rates.end()) return kInf;
      seconds +=
          std::max(term.bytes / performance->memory_bytes_per_s, term.units / rate->units_per_s);
    }
    bottleneck = std::max(bottleneck, seconds);
  }
  return bottleneck;
}

double ComponentCost(const Net& net, const Workload& w, const Info& info, std::uint8_t choice,
                     std::span<const std::size_t> group) {
  double cost = 0;
  if (choice != kWhole) {
    const auto& o = info.in->def->splits[choice];
    cost = Weighted(w, OptionTraffic(net, info, o, group, w.prefill_rows, 1),
                    OptionTraffic(net, info, o, group, w.decode_rows, w.decode_rows));
  }
  // Avoid 0 * infinity for an unused phase, without excusing missing facts in
  // phases that the workload actually asks to price.
  if (w.decode_share < 1)
    cost += (1 - w.decode_share) * ExecutionSeconds(net, info, choice, group, 0);
  if (w.decode_share > 0) cost += w.decode_share * ExecutionSeconds(net, info, choice, group, 1);
  return cost;
}

// A layer component's choice on a group: its first fitting option of least
// cost, or whole.
std::uint8_t LayerChoice(const Net& net, const Workload& w, const Info& info,
                         std::span<const std::size_t> group) {
  const auto g = static_cast<std::uint32_t>(group.size());
  std::uint8_t best = kWhole;
  double cost = kInf;
  for (std::size_t i = 0; i < info.in->def->splits.size(); ++i) {
    const SplitOption& o = info.in->def->splits[i];
    if (!Fits(info, o, g)) continue;
    const double c = ComponentCost(net, w, info, static_cast<std::uint8_t>(i), group);
    if (c < cost) {
      cost = c;
      best = static_cast<std::uint8_t>(i);
    }
  }
  return best;
}

const SplitOption* OptionOf(const Info& info, std::uint8_t choice) {
  return choice == kWhole ? nullptr : &info.in->def->splits[choice];
}

// Every instance's Info, by ordinal.
struct Infos {
  std::vector<Info> all{};
  std::optional<std::size_t> embedding{};
  std::vector<std::size_t> side{};
  std::optional<std::size_t> mixing{};
  std::optional<std::size_t> final_norm{};
  std::optional<std::size_t> output{};
  std::vector<std::vector<std::size_t>> layers{};
};

std::expected<Infos, std::string> InfosOf(const StageModel& m, const Workload& w,
                                          bool state = true) {
  Infos out;
  const auto& s = m.instances;
  if (m.side_layers.size() != s.side.size()) return Refused("reader layers for every side source");
  for (const auto& readers : m.side_layers)
    for (const auto layer : readers)
      if (layer >= s.layers.size()) return Refused("a side reader outside the stage's layers");
  const auto add = [&](const Instance& in,
                       std::uint16_t first) -> std::expected<std::size_t, std::string> {
    auto info = InfoOf(m, in, w, first, state);
    if (!info) return std::unexpected(info.error());
    out.all.push_back(std::move(*info));
    return out.all.size() - 1;
  };
  if (s.embedding) {
    auto i = add(*s.embedding, 0);
    if (!i) return std::unexpected(i.error());
    out.embedding = *i;
  }
  for (const Instance& in : s.side) {
    auto i = add(in, 0);
    if (!i) return std::unexpected(i.error());
    out.side.push_back(*i);
  }
  if (s.mixing) {
    auto i = add(*s.mixing, 0);
    if (!i) return std::unexpected(i.error());
    out.mixing = *i;
  }
  for (const auto& layer : s.layers) {
    auto& ids = out.layers.emplace_back();
    std::uint16_t first = 0;
    for (const Instance& in : layer) {
      auto i = add(in, first);
      if (!i) return std::unexpected(i.error());
      ids.push_back(*i);
      first = static_cast<std::uint16_t>(first + out.all[*i].contract.tensors.size());
    }
  }
  if (s.final_norm) {
    auto i = add(*s.final_norm, 0);
    if (!i) return std::unexpected(i.error());
    out.final_norm = *i;
  }
  if (s.output) {
    auto i = add(*s.output, 0);
    if (!i) return std::unexpected(i.error());
    out.output = *i;
  }
  // What follows the narrowing component runs at the output rows.
  if (const auto at = components::NarrowPoint(s)) {
    for (std::size_t l = at->first; l < out.layers.size(); ++l) {
      for (std::size_t c = l == at->first ? at->second + 1 : 0; c < out.layers[l].size(); ++c) {
        out.all[out.layers[l][c]].narrowed = true;
      }
    }
    if (out.final_norm) out.all[*out.final_norm].narrowed = true;
    if (out.output) out.all[*out.output].narrowed = true;
  }
  std::map<std::string, format::TensorFormat> formats;
  for (const auto& info : out.all) {
    for (const auto& weight : info.weights) {
      const auto [at, fresh] = formats.emplace(weight.name, weight.format);
      if (!fresh && at->second != weight.format)
        return Refused("one weight identity has conflicting formats");
    }
  }
  return out;
}

// The stage's components outside the layers that run with the first
// stage (the embedding, side sources and stream mixing) or the last
// (the final norm and the head).
std::vector<std::size_t> Extras(const Infos& infos, bool first, bool last) {
  std::vector<std::size_t> out;
  if (first) {
    if (infos.embedding) out.push_back(*infos.embedding);
    out.insert(out.end(), infos.side.begin(), infos.side.end());
    if (infos.mixing) out.push_back(*infos.mixing);
  }
  if (last && infos.final_norm) out.push_back(*infos.final_norm);
  if (last && infos.output) out.push_back(*infos.output);
  return out;
}

std::vector<std::size_t> SortedSlots(std::uint32_t mask) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; mask != 0; ++i, mask >>= 1U) {
    if ((mask & 1U) != 0) out.push_back(i);
  }
  return out;
}

ComponentPlacement PlacementOf(const Info& info, std::uint8_t choice, const Net& net,
                               std::span<const std::size_t> group) {
  ComponentPlacement p{
      .component = info.in->index,
      .def = info.in->def,
      .split = choice == kWhole ? SplitKind::kNone : info.in->def->splits[choice].kind,
      .option = choice,
      .state_first = info.state_first,
      .devices = {}};
  for (const std::size_t s : group) p.devices.push_back(net.Id(s));
  return p;
}

// Derive work from actual primitive graphs. Source-backed models use BuildStep,
// including real narrowing/head gathers. Framework dependencies such as masks
// are charged with every consuming component: a conservative no-reuse bound,
// rather than assuming physical fusion or cross-component sharing.
std::expected<std::vector<WorkTerm>, std::string> WorkOf(const ir::Graph& graph,
                                                         std::uint16_t component) {
  std::vector<bool> framework(graph.nodes().size(), false);
  const auto visit = [&](auto&& self, TensorId id) -> void {
    if (!id.valid()) return;
    const auto producer = graph.producer(id);
    const auto& node = graph.node(producer);
    if (node.component != components::kFrameworkComponent || framework[producer.v]) return;
    framework[producer.v] = true;
    for (auto src : node.sources()) self(self, src);
  };
  for (const auto& node : graph.nodes()) {
    if (node.component == component)
      for (auto src : node.sources()) visit(visit, src);
  }
  std::vector<WorkTerm> out;
  for (std::size_t ni = 0; ni < graph.nodes().size(); ++ni) {
    const auto& node = graph.nodes()[ni];
    if (node.component != component && !framework[ni]) continue;
    const Op op = node.key.op;
    if (op == Op::kInput || op == Op::kWeight || op == Op::kState || op == Op::kView ||
        op == Op::kReshape || op == Op::kPermute || op == Op::kAllReduce || op == Op::kAllGather ||
        op == Op::kAllToAll || op == Op::kSend || op == Op::kRecv)
      continue;
    if (node.nout == 0) return Refused("execution work for an operation without output");
    WorkTerm term{.op = node.key,
                  .params = node.params,
                  .encoding = graph.tensor(node.out[0]).format.encoding};
    const bool product = op == Op::kMatMul || op == Op::kMatMulId;
    if (product) {
      term.encoding = graph.tensor(node.src[0]).format.encoding;
      term.accumulate = node.Params<ir::ProductParams>().accumulate;
    } else if (op == Op::kAttention) {
      term.encoding = graph.tensor(node.src[1]).format.encoding;
      term.accumulate = node.Params<ir::AttnParams>().accumulate;
    }
    for (auto id : node.outputs()) {
      const auto& tensor = graph.tensor(id);
      term.units += static_cast<double>(ir::Elements(tensor.shape()));
      auto bytes = FormatBytes(tensor.format);
      if (!bytes) return std::unexpected(bytes.error());
      term.bytes += static_cast<double>(*bytes);
    }
    for (std::size_t si = 0; si < node.nsrc; ++si) {
      const auto id = node.src[si];
      if (!id.valid() || (op == Op::kCacheStore && si == 0)) continue;
      const auto& tensor = graph.tensor(id);
      auto bytes = FormatBytes(tensor.format);
      if (!bytes) return std::unexpected(bytes.error());
      double reads = static_cast<double>(*bytes);
      if (op == Op::kGetRows && si == 0) {
        // Only selected rows are read; repeated ids may reuse them, but reuse
        // is not assumed. Batch products, in contrast, reuse a matrix once.
        auto row = tensor.format;
        row.shape.ne[1] = row.shape.ne[2] = row.shape.ne[3] = 1;
        auto row_bytes = FormatBytes(row);
        if (!row_bytes) return std::unexpected(row_bytes.error());
        reads = (term.units / static_cast<double>(tensor.ne(0))) * static_cast<double>(*row_bytes);
      }
      if (op == Op::kMatMulId && si == 0) {
        const auto experts = tensor.ne(2);
        const auto& ids = graph.tensor(node.src[2]);
        // Worst possible distinct local experts, with no balance assumption.
        // EP executes every routed slot, including foreign slots mapped to its
        // first expert. The local graph retains the full routed output shape.
        reads *= static_cast<double>(std::min(experts, ir::Elements(ids.shape()))) /
                 static_cast<double>(experts);
      }
      term.bytes += reads;
    }
    if (op == Op::kCacheStore) {
      // The result aliases the cache: only the submitted values are written.
      auto written = graph.tensor(node.src[0]).format;
      written.shape = graph.tensor(node.src[1]).format.shape;
      auto bytes = FormatBytes(written);
      if (!bytes) return std::unexpected(bytes.error());
      term.bytes = static_cast<double>(*bytes);
      for (std::size_t si = 1; si < node.nsrc; ++si) {
        if (!node.src[si].valid()) continue;
        auto read = FormatBytes(graph.tensor(node.src[si]).format);
        if (!read) return std::unexpected(read.error());
        term.bytes += static_cast<double>(*read);
      }
      term.units = static_cast<double>(ir::Elements(graph.tensor(node.src[1]).shape()));
    }
    if (product) term.units *= 2 * static_cast<double>(graph.tensor(node.src[0]).ne(0));
    if (op == Op::kAttention) {
      const auto& q = graph.tensor(node.src[0]);
      const auto& k = graph.tensor(node.src[1]);
      const auto& v = graph.tensor(node.src[2]);
      term.units = 2 * static_cast<double>(q.ne(0) + v.ne(0)) * static_cast<double>(k.ne(1)) *
                   static_cast<double>(q.ne(1)) * static_cast<double>(q.ne(2)) *
                   static_cast<double>(q.ne(3));
    }
    if (!std::isfinite(term.bytes) || !std::isfinite(term.units))
      return Refused("execution work overflow");
    out.push_back(term);
  }
  return out;
}

std::expected<void, std::string> PrepareWork(const StageModel& model, Infos& infos,
                                             const Topology& topology, const Workload& workload) {
  if (topology.cost_model != CostModel::kExecution) return {};
  if (workload.prefill_rows == 0 || workload.decode_rows == 0 || workload.context == 0 ||
      workload.slots == 0)
    return Refused("execution work needs positive rows, context and retained slots");
  const auto max_rows = std::max(workload.prefill_rows, workload.decode_rows);
  if (max_rows > workload.context) return Refused("execution rows past the described context");
  if (workload.context > std::numeric_limits<std::uint32_t>::max() - 255)
    return Refused("execution context overflows aligned read cells");
  std::vector<model::StateLayoutDerived> layouts;
  std::vector<const model::StateLayoutDerived*> layout_ptrs;
  if (model.source) {
    for (std::size_t stage = 0; stage < model.source->stages.size(); ++stage) {
      if (model.source->stages[stage].artifact != model.source->stages[model.stage].artifact)
        return Refused("execution work needs the other stage's artifact bindings");
      auto layout = components::StageLayout(*model.source, stage, workload.context, max_rows);
      if (!layout) return std::unexpected(layout.error());
      layouts.push_back(std::move(*layout));
    }
    for (const auto& layout : layouts) layout_ptrs.push_back(&layout);
  }
  std::uint64_t graph_work = 0;
  std::array<std::optional<ir::Graph>, 2> source;
  for (std::size_t phase = 0; phase < 2; ++phase) {
    if ((phase == 0 && workload.decode_share == 1) || (phase == 1 && workload.decode_share == 0))
      continue;
    const auto rows = phase == 0 ? workload.prefill_rows : workload.decode_rows;
    const auto outputs = phase == 0 ? 1U : rows;
    const StepShape shape{.stage = model.stage,
                          .phase = phase == 0 ? Phase::kPrefill : Phase::kDecode,
                          .segments = {{.rows = rows,
                                        .past = workload.context - rows,
                                        .read_cells = ((workload.context + 255) / 256) * 256,
                                        .outputs = outputs}}};
    if (model.source) {
      auto step = components::BuildStep(
          *model.source, model.stage, shape,
          {.tensors = model.tensors, .state = layout_ptrs[model.stage], .stages = layout_ptrs});
      if (!step) return Refused(std::format("execution work: {}", step.error()));
      source[phase] = std::move(step->graph);
    }
    for (auto& info : infos.all) {
      std::optional<ir::Graph> fragment;
      if (!model.source) {
        std::vector<format::TensorFormat> formats(info.in->def->roles.size());
        for (const auto& weight : info.weights) formats[weight.role] = weight.format;
        const std::array contract{std::pair{info.in->layer, info.contract}};
        auto layout = model::Layout(contract, workload.context, max_rows);
        if (!layout) return std::unexpected(layout.error());
        std::vector<std::uint64_t> cells(info.contract.tensors.size());
        for (const auto& item : layout->items) cells[item.tensor] = item.cells;
        auto built = components::BuildComponent(*info.in, shape, formats, cells);
        if (!built)
          return Refused(
              std::format("execution fragment {}: {}", info.in->def->kind, built.error()));
        fragment = std::move(built->step.graph);
      }
      const auto& graph = model.source ? *source[phase] : *fragment;
      for (std::uint32_t ways = 1; ways <= topology.devices.size(); ++ways) {
        std::vector<std::uint8_t> choices{kWhole};
        for (std::size_t oi = 0; oi < info.in->def->splits.size(); ++oi)
          if (Fits(info, info.in->def->splits[oi], ways))
            choices.push_back(static_cast<std::uint8_t>(oi));
        for (auto choice : choices) {
          // Synthetic rank ids carry no device-rate claim. Partition determines
          // the actual shapes/replicas/EP auxiliary operations, once per option.
          Assignment assignment{.stage = model.stage};
          std::vector<std::uint32_t> group;
          for (std::uint32_t rank = 0; rank < ways; ++rank) {
            assignment.topology.devices.push_back({.id = rank});
            group.push_back(rank);
            for (std::uint32_t other = 0; other < rank; ++other)
              assignment.topology.links.push_back({.a = other, .b = rank, .bytes_per_s = 1});
          }
          assignment.stages.push_back(group);
          for (const auto& placed : infos.all) {
            auto p = PlacementOf(placed, placed.in->index == info.in->index ? choice : kWhole,
                                 Net(assignment.topology), SortedSlots((1U << ways) - 1));
            if (!model.source) p.state_first = 0;
            assignment.per_layer.push_back({std::move(p)});
          }
          std::map<std::string, ir::WeightRef> first;
          for (const auto& placed : infos.all) {
            for (const auto& weight : placed.weights) {
              ir::WeightRef ref{.stage = model.stage,
                                .component = placed.in->index,
                                .role = weight.role,
                                .layer = placed.in->layer};
              const auto [at, fresh] = first.emplace(weight.name, ref);
              if (!fresh)
                assignment.aliases.push_back(
                    {.component = placed.in->index, .leaf = at->second, .role = weight.role});
            }
          }
          const auto upper = static_cast<std::uint64_t>(graph.nodes().size()) * ways;
          if (upper > kMaxPartitionWork - graph_work)
            return Refused("execution graph work is past the exact planner's work bound");
          graph_work += upper;
          auto partition = Partition(graph, assignment);
          if (!partition)
            return Refused(
                std::format("execution partition {}: {}", info.in->def->kind, partition.error()));
          auto& work = info.work[{ways, choice}].phase[phase];
          work.resize(ways);
          for (const auto& device : partition->devices) {
            auto terms = WorkOf(device.graph, info.in->index);
            if (!terms) return std::unexpected(terms.error());
            work[device.device] = std::move(*terms);
          }
        }
      }
    }
  }
  // All candidate work must be priceable; missing rates must not make an
  // unknown candidate silently lose against a known one.
  for (const auto& device : topology.devices) {
    for (const auto& info : infos.all)
      for (const auto& [key, work] : info.work)
        for (const auto& ranks : work.phase)
          for (const auto& terms : ranks)
            for (const auto& term : terms) {
              const auto& rates = device.performance->rates;
              if (!std::ranges::any_of(rates, [&](const auto& rate) {
                    return rate.op == term.op && rate.params == term.params &&
                           rate.encoding == term.encoding && rate.accumulate == term.accumulate;
                  }))
                return Refused(std::format("device {} lacks described rate for {} ({})", device.id,
                                           ir::OpKeyName(term.op),
                                           static_cast<int>(term.encoding)));
            }
  }
  return {};
}

// One interval's cheapest jointly feasible choices. Layer choices cannot be
// finalized before capacity is checked: a cheaper whole replica can be too big
// where a split fits. The same choice search serves configured pipelines and
// the global DP. A cost lower bound prunes only provably no-better suffixes.
struct RangeChoice {
  std::vector<std::size_t> ids{};
  std::vector<std::uint8_t> choices{};
  std::vector<std::uint64_t> memory{};
  double cost = kInf;
};
std::expected<RangeChoice, std::string> ChooseRange(const Net& net, const Workload& workload,
                                                    const Infos& infos,
                                                    std::span<const std::size_t> group,
                                                    std::uint32_t begin, std::uint32_t end,
                                                    std::uint64_t& work) {
  RangeChoice best;
  std::set<std::size_t> layers;
  for (auto layer = begin; layer < end; ++layer) {
    best.ids.insert(best.ids.end(), infos.layers[layer].begin(), infos.layers[layer].end());
    layers.insert(infos.layers[layer].begin(), infos.layers[layer].end());
  }
  const auto extras = Extras(infos, begin == 0, end == infos.layers.size());
  best.ids.insert(best.ids.end(), extras.begin(), extras.end());
  const auto ways = static_cast<std::uint32_t>(group.size());
  std::vector<std::vector<std::pair<std::uint8_t, double>>> options;
  for (auto id : best.ids) {
    const auto& info = infos.all[id];
    auto& choices = options.emplace_back();
    if (!net.Execution() && layers.contains(id)) {
      const auto choice = LayerChoice(net, workload, info, group);
      choices.emplace_back(choice, ComponentCost(net, workload, info, choice, group));
    } else {
      choices.emplace_back(kWhole, ComponentCost(net, workload, info, kWhole, group));
      for (std::size_t option = 0; option < info.in->def->splits.size(); ++option) {
        if (!Fits(info, info.in->def->splits[option], ways)) continue;
        const auto choice = static_cast<std::uint8_t>(option);
        choices.emplace_back(choice, ComponentCost(net, workload, info, choice, group));
      }
    }
    std::ranges::stable_sort(choices, {}, &std::pair<std::uint8_t, double>::second);
  }
  std::vector<double> suffix(best.ids.size() + 1, 0);
  for (std::size_t i = best.ids.size(); i > 0; --i)
    suffix[i - 1] = suffix[i] + options[i - 1].front().second;
  std::vector<std::uint8_t> chosen(best.ids.size());
  bool overflow = false;
  const auto visit = [&](auto&& self, std::size_t at, double cost) -> void {
    if (++work > kMaxPartitionWork) {
      overflow = true;
      return;
    }
    if (!std::isfinite(cost + suffix[at]) || cost + suffix[at] >= best.cost) return;
    if (at < best.ids.size()) {
      for (const auto& [choice, seconds] : options[at]) {
        chosen[at] = choice;
        self(self, at + 1, cost + seconds);
        if (overflow) return;
      }
      return;
    }
    std::vector<std::uint64_t> memory;
    for (std::uint32_t rank = 0; rank < ways; ++rank) {
      std::vector<Held> held;
      std::uint64_t state = 0;
      for (std::size_t k = 0; k < best.ids.size(); ++k) {
        if (++work > kMaxPartitionWork) {
          overflow = true;
          return;
        }
        const auto& info = infos.all[best.ids[k]];
        if (!Hold(info, OptionOf(info, chosen[k]), ways, rank, workload.slots, held, state)) return;
      }
      const auto bytes = Total(held, state);
      if (!bytes || *bytes > net.Memory(group[rank])) return;
      memory.push_back(*bytes);
    }
    best.cost = cost;
    best.choices = chosen;
    best.memory = std::move(memory);
  };
  visit(visit, 0, 0);
  if (overflow) return Refused("the topology is past the exact search's work bound");
  return best;
}

}  // namespace

// ------------------------------------------------------------------ public

std::expected<void, std::string> Validate(const Topology& t) {
  if (t.devices.empty()) return Refused("a topology without devices");
  if (t.devices.size() > kMaxDevices) {
    return Refused(std::format("{} devices, past {}", t.devices.size(), kMaxDevices));
  }
  std::set<std::uint32_t> ids;
  for (const Device& d : t.devices) {
    if (!ids.insert(d.id).second) return Refused(std::format("device {} twice", d.id));
  }
  if (t.cost_model != CostModel::kTraffic && t.cost_model != CostModel::kExecution)
    return Refused("unknown partition cost model");
  for (const auto& d : t.devices) {
    if (!d.performance) {
      if (t.cost_model == CostModel::kExecution)
        return Refused(std::format("device {} has no described execution performance", d.id));
      continue;
    }
    const auto& p = *d.performance;
    if (!(p.memory_bytes_per_s > 0) || !std::isfinite(p.memory_bytes_per_s) || p.provenance.empty())
      return Refused(
          std::format("device {} needs finite positive memory bandwidth and provenance", d.id));
    for (std::size_t i = 0; i < p.rates.size(); ++i) {
      const auto& r = p.rates[i];
      if (!(r.units_per_s > 0) || !std::isfinite(r.units_per_s))
        return Refused(std::format("device {} needs finite positive operation rates", d.id));
      if (ir::OpName(r.op.op) == "unknown" ||
          (r.op.op == Op::kFused
               ? (r.op.fused == ir::FusedKind::kNone || ir::FusedKindName(r.op.fused) == "unknown")
               : r.op.fused != ir::FusedKind::kNone) ||
          static_cast<std::size_t>(r.encoding) >= format::Encodings().size() ||
          (r.accumulate != ir::Accumulate::kDefault && r.accumulate != ir::Accumulate::kF32))
        return Refused(std::format("device {} has an invalid typed operation rate", d.id));
      for (std::size_t j = 0; j < i; ++j) {
        const auto& before = p.rates[j];
        if (before.op == r.op && before.params == r.params && before.encoding == r.encoding &&
            before.accumulate == r.accumulate)
          return Refused(std::format("device {} repeats an operation rate", d.id));
      }
    }
  }
  std::set<std::pair<std::uint32_t, std::uint32_t>> pairs;
  for (const Link& l : t.links) {
    if (!ids.contains(l.a) || !ids.contains(l.b) || l.a == l.b) {
      return Refused(std::format("a link between {} and {}", l.a, l.b));
    }
    if (!pairs.insert(std::minmax(l.a, l.b)).second) {
      return Refused(std::format("two links between {} and {}", l.a, l.b));
    }
    if (!(l.bytes_per_s > 0.0) || !std::isfinite(l.bytes_per_s) || !(l.latency_s >= 0.0) ||
        !std::isfinite(l.latency_s)) {
      return Refused(std::format("the link between {} and {}: a bandwidth above zero", l.a, l.b));
    }
  }
  return {};
}

const ComponentPlacement* Assignment::Find(std::uint16_t component) const {
  if (embedding && embedding->component == component) return &*embedding;
  for (const ComponentPlacement& p : side) {
    if (p.component == component) return &p;
  }
  if (mixing && mixing->component == component) return &*mixing;
  for (const auto& layer : per_layer) {
    for (const ComponentPlacement& p : layer) {
      if (p.component == component) return &p;
    }
  }
  if (final_norm && final_norm->component == component) return &*final_norm;
  if (output && output->component == component) return &*output;
  return nullptr;
}

std::expected<StageModel, std::string> ModelOf(const spec::ModelSpec& spec, std::uint16_t stage,
                                               std::span<const components::BoundTensor> tensors) {
  auto instances = components::InstancesOf(spec, stage);
  if (!instances) return std::unexpected(instances.error());
  StageModel m{.stage = stage,
               .instances = std::move(*instances),
               .roles = {},
               .side_layers = {},
               .source = &spec,
               .tensors = tensors};
  for (const auto& side : spec.stages[stage].side) m.side_layers.push_back(side.layers);
  for (const Instance* in : All(m.instances)) {
    if (m.roles.size() <= in->index) m.roles.resize(in->index + 1);
    auto& roles = m.roles[in->index];
    roles.resize(in->def->roles.size());
    for (const spec::WeightBinding& b : in->ref->weights) {
      const auto role = components::RoleIndex(*in->def, b.role);
      if (!role) return Refused(std::format("{} has no weight role {}", in->def->kind, b.role));
      if (!roles[*role].name.empty()) continue;  // a later part of the same role
      const auto t = std::ranges::find_if(tensors, [&](const components::BoundTensor& x) {
        return x.name == b.tensor && x.part == b.part && x.experts == b.experts;
      });
      if (t == tensors.end()) {
        return Refused(
            std::format("{} binds {}, which the artifact does not have", in->def->kind, b.tensor));
      }
      roles[*role] = {.name = b.tensor, .format = t->format};
    }
  }
  return m;
}

Traffic ModelTraffic(const StageModel& model, const Assignment& a, std::uint32_t rows,
                     std::uint32_t outputs) {
  Traffic t;
  if (!Validate(a.topology)) return {.bytes = 0, .seconds = kInf};
  const Net net(a.topology);
  auto infos = InfosOf(model, Workload{}, false);
  if (!infos) return {.bytes = 0, .seconds = kInf};
  const auto slots = [&](const std::vector<std::uint32_t>& ids) {
    std::vector<std::size_t> out;
    for (const std::uint32_t id : ids) out.push_back(net.Slot(id).value_or(0));
    return out;
  };
  for (const Info& info : infos->all) {
    const ComponentPlacement* p = a.Find(info.in->index);
    if (p == nullptr || p->option == kWhole) continue;
    Add(t, OptionTraffic(net, info, info.in->def->splits[p->option], slots(p->devices), rows,
                         outputs));
  }
  for (std::size_t s = 1; s < a.stages.size(); ++s) {
    const auto from = slots(a.stages[s - 1]);
    const auto to = slots(a.stages[s]);
    const auto begin = std::ranges::find(a.stage_of_layer, static_cast<std::uint32_t>(s));
    if (begin == a.stage_of_layer.end()) return {.bytes = t.bytes, .seconds = kInf};
    Add(t, Relay(net, from, to,
                 BoundaryBytes(model, static_cast<std::uint32_t>(begin - a.stage_of_layer.begin()),
                               rows, outputs)));
  }
  // Side outputs are immutable: a group receives each one only if it has
  // a reader, and later groups can relay it from any previous recipient.
  for (std::size_t i = 0; i < infos->side.size(); ++i) {
    const auto* source = a.Find(infos->all[infos->side[i]].in->index);
    if (source == nullptr) return {.bytes = 0, .seconds = kInf};
    auto holders = slots(source->devices);
    for (std::size_t s = 0; s < a.stages.size(); ++s) {
      const bool reads = std::ranges::any_of(model.side_layers[i], [&](std::uint32_t l) {
        return l < a.stage_of_layer.size() && a.stage_of_layer[l] == s;
      });
      if (!reads) continue;
      const auto targets = slots(a.stages[s]);
      Add(t, Relay(net, holders, targets, infos->all[infos->side[i]].side_bytes * rows));
      for (const auto target : targets)
        if (!std::ranges::contains(holders, target)) holders.push_back(target);
    }
  }
  return t;
}

std::expected<Assignment, std::string> AssignPipeline(const StageModel& model,
                                                      const Topology& topology,
                                                      const Workload& workload,
                                                      const Pipeline& pipeline) {
  if (auto v = Validate(topology); !v) return std::unexpected(v.error());
  if (!(workload.decode_share >= 0.0 && workload.decode_share <= 1.0)) {
    return Refused("a decode share outside [0, 1]");
  }
  const Net net(topology);
  auto infos = InfosOf(model, workload);
  if (!infos) return std::unexpected(infos.error());
  if (auto prepared = PrepareWork(model, *infos, topology, workload); !prepared)
    return std::unexpected(prepared.error());
  const auto layers = static_cast<std::uint32_t>(infos->layers.size());
  const std::size_t stages = pipeline.stages.size();
  if (stages == 0 || pipeline.first_layer.size() != stages || pipeline.first_layer[0] != 0) {
    return Refused("a pipeline of stages, the first from layer 0");
  }
  std::vector<std::vector<std::size_t>> groups;
  std::set<std::size_t> used;
  for (std::size_t s = 0; s < stages; ++s) {
    if (pipeline.stages[s].empty()) return Refused(std::format("stage {} has no devices", s));
    if (s > 0 && (pipeline.first_layer[s] <= pipeline.first_layer[s - 1] ||
                  pipeline.first_layer[s] >= layers)) {
      return Refused(std::format("stage {} starts at layer {}", s, pipeline.first_layer[s]));
    }
    auto& g = groups.emplace_back();
    for (const std::uint32_t id : pipeline.stages[s]) {
      const auto slot = net.Slot(id);
      if (!slot) return Refused(std::format("no device {}", id));
      if (!used.insert(*slot).second) return Refused(std::format("device {} in two places", id));
      g.push_back(*slot);
    }
  }
  if (layers == 0 && stages != 1) return Refused("a stage without layers in one place");

  Assignment a{.topology = topology, .stage = model.stage};
  a.stages = pipeline.stages;
  a.stage_of_layer.resize(layers);
  std::uint64_t work = 0;
  for (std::size_t s = 0; s < stages; ++s) {
    const auto begin = pipeline.first_layer[s];
    const auto end = s + 1 < stages ? pipeline.first_layer[s + 1] : layers;
    auto range = ChooseRange(net, workload, *infos, groups[s], begin, end, work);
    if (!range) return std::unexpected(range.error());
    if (!std::isfinite(range->cost)) return Refused("no component choices fit the devices' memory");
    for (auto layer = begin; layer < end; ++layer) {
      a.stage_of_layer[layer] = static_cast<std::uint32_t>(s);
      a.per_layer.emplace_back();
    }
    for (std::size_t k = 0; k < range->ids.size(); ++k) {
      const auto id = range->ids[k];
      const auto& info = infos->all[id];
      auto placement = PlacementOf(info, range->choices[k], net, groups[s]);
      if (infos->embedding == id) a.embedding = placement;
      if (std::ranges::contains(infos->side, id)) a.side.push_back(placement);
      if (infos->mixing == id) a.mixing = placement;
      if (infos->final_norm == id) a.final_norm = placement;
      if (infos->output == id) a.output = placement;
      for (auto layer = begin; layer < end; ++layer)
        if (std::ranges::contains(infos->layers[layer], id))
          a.per_layer[layer].push_back(placement);
    }
    for (std::size_t rank = 0; rank < groups[s].size(); ++rank)
      a.memory.emplace_back(net.Id(groups[s][rank]), range->memory[rank]);
  }
  std::ranges::sort(a.memory, {}, [&](const auto& value) { return *net.Slot(value.first); });
  // A tensor first bound by one component and read by another: the later
  // one's alias of the first's leaf.
  std::map<std::string, ir::WeightRef> first;
  for (const Info& info : infos->all) {
    for (const WeightUse& w : info.weights) {
      const ir::WeightRef ref{.stage = a.stage,
                              .component = info.in->index,
                              .role = w.role,
                              .layer = info.in->layer,
                              .expert_array = false};
      const auto [at, fresh] = first.emplace(w.name, ref);
      if (!fresh) {
        a.aliases.push_back({.component = info.in->index, .leaf = at->second, .role = w.role});
      }
    }
  }
  a.prefill = ModelTraffic(model, a, workload.prefill_rows, 1);
  a.decode = ModelTraffic(model, a, workload.decode_rows, workload.decode_rows);
  if (!std::isfinite(a.prefill.seconds) || !std::isfinite(a.decode.seconds)) {
    return Refused("a group or stage boundary without the links it needs");
  }
  a.modelled_seconds_per_step = Weighted(workload, a.prefill, a.decode);
  if (net.Execution()) {
    if (workload.decode_share < 1) a.prefill_execution_seconds = 0;
    if (workload.decode_share > 0) a.decode_execution_seconds = 0;
    for (const auto& info : infos->all) {
      const auto* placement = a.Find(info.in->index);
      std::vector<std::size_t> group;
      for (const auto id : placement->devices) group.push_back(*net.Slot(id));
      if (workload.decode_share < 1)
        *a.prefill_execution_seconds += ExecutionSeconds(net, info, placement->option, group, 0);
      if (workload.decode_share > 0)
        *a.decode_execution_seconds += ExecutionSeconds(net, info, placement->option, group, 1);
    }
    a.modelled_seconds_per_step +=
        (1 - workload.decode_share) * a.prefill_execution_seconds.value_or(0) +
        workload.decode_share * a.decode_execution_seconds.value_or(0);
  }
  a.modelled_bytes_per_step =
      (workload.decode_share * static_cast<double>(a.decode.bytes)) +
      ((1.0 - workload.decode_share) * static_cast<double>(a.prefill.bytes));
  return a;
}

// ------------------------------------------------------------- the search

namespace {

// The search's value from a state on: cost, then devices it adds, then
// stages.
struct Value {
  double cost = kInf;
  std::uint32_t devices = 0;
  std::uint32_t stages = 0;
  std::uint32_t group = 0;  // the next stage's devices
  std::uint32_t end = 0;    // and the layer after it
};
bool Better(const Value& a, const Value& b) {
  const double eps = 1e-12 * std::max({1.0, std::fabs(a.cost), std::fabs(b.cost)});
  if (std::isinf(b.cost)) return std::isfinite(a.cost);
  if (a.cost < b.cost - eps) return true;
  if (a.cost > b.cost + eps) return false;
  return std::tie(a.devices, a.stages) < std::tie(b.devices, b.stages);
}

class Search {
 public:
  Search(const StageModel& model, const Topology& topology, const Workload& w, const Infos& infos)
      : model_(model),
        net_(topology),
        w_(w),
        infos_(infos),
        layers_(static_cast<std::uint32_t>(infos.layers.size())) {
    // Twins: same memory and the same link to every other device (an
    // equivalence: any permutation inside a class maps the topology onto
    // itself).
    const std::size_t n = net_.size();
    std::vector<int> cls(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
      if (cls[i] >= 0) continue;
      cls[i] = static_cast<int>(classes_.size());
      classes_.push_back({i});
      for (std::size_t j = i + 1; j < n; ++j) {
        if (cls[j] >= 0 || net_.Memory(i) != net_.Memory(j) ||
            net_.Performance(i) != net_.Performance(j))
          continue;
        bool twin = true;
        for (std::size_t k = 0; k < n && twin; ++k) {
          if (k == i || k == j) continue;
          const auto& a = net_.LinkOf(i, k);
          const auto& b = net_.LinkOf(j, k);
          twin = a.has_value() == b.has_value() &&
                 (!a || (a->bytes_per_s == b->bytes_per_s && a->latency_s == b->latency_s));
        }
        if (twin) {
          cls[j] = cls[i];
          classes_.back().push_back(j);
        }
      }
    }
    for (const int c : cls) class_of_.push_back(static_cast<std::size_t>(c));
  }

  std::expected<Value, std::string> Run() {
    // The groups a stage may take: every class's lowest members, any count.
    std::uint64_t groups = 1;
    for (const auto& c : classes_) groups *= c.size() + 1;
    if (groups - 1 > kMaxGroups) {
      return Refused(std::format("{} candidate device groups, past the exact search's {}",
                                 groups - 1, kMaxGroups));
    }
    const Value v = F(0, 0, 0, std::vector<std::uint32_t>(infos_.side.size()));
    if (overflow_) return Refused("the topology is past the exact search's bound");
    if (std::isinf(v.cost)) return Refused("no placement fits the devices' memory");
    return v;
  }
  // The pipeline the search chose.
  Pipeline Chosen() {
    Pipeline p;
    std::uint32_t i = 0;
    std::uint32_t prev = 0;
    std::uint32_t used = 0;
    std::vector<std::uint32_t> side(infos_.side.size());
    while (i < layers_) {
      const Value v = memo_.at(Key(i, prev, used, side));
      for (std::size_t k = 0; k < side.size(); ++k)
        if (i == 0 || Reads(k, i, v.end)) side[k] |= v.group;
      std::vector<std::uint32_t> ids;
      for (const std::size_t s : Ordered(v.group)) ids.push_back(net_.Id(s));
      p.stages.push_back(std::move(ids));
      p.first_layer.push_back(i);
      prev = v.group;
      used |= v.group;
      i = v.end;
    }
    return p;
  }

 private:
  static constexpr std::size_t kMaxStates = std::size_t{1} << 21U;
  // The search's work (a group's layers evaluated, a candidate group or a
  // stage end tried): bounded too, so a topology past it is refused
  // promptly rather than searched for hours.
  static constexpr std::uint64_t kMaxWork = kMaxPartitionWork;
  // The candidate groups (twelve devices without twins; any count of
  // devices in a few twin classes).
  static constexpr std::uint64_t kMaxGroups = std::uint64_t{1} << 12U;

  bool Spend(std::uint64_t units) {
    work_ += units;
    if (work_ > kMaxWork) overflow_ = true;
    return !overflow_;
  }

  // A group's devices in ring order: by twin class, then slot. A twin swap
  // maps a group's order onto its image's, so taking each class's members
  // lowest first loses no cost (a ring in plain slot order would not be
  // invariant, and the pruning would miss cheaper groups).
  std::vector<std::size_t> Ordered(std::uint32_t mask) const {
    std::vector<std::size_t> out = SortedSlots(mask);
    std::ranges::stable_sort(out, {}, [&](std::size_t s) { return class_of_[s]; });
    return out;
  }

  struct StateKey {
    std::uint64_t pipeline = 0;
    std::vector<std::uint32_t> side{};
    bool operator==(const StateKey&) const = default;
  };
  struct StateHash {
    std::size_t operator()(const StateKey& key) const {
      auto hash = std::hash<std::uint64_t>{}(key.pipeline);
      for (const auto holders : key.side) hash = (hash * 31) + holders;
      return hash;
    }
  };
  static StateKey Key(std::uint32_t i, std::uint32_t prev, std::uint32_t used,
                      const std::vector<std::uint32_t>& side) {
    return {(std::uint64_t{i} << 40U) | (std::uint64_t{prev} << 20U) | used, side};
  }
  bool Reads(std::size_t side, std::uint32_t begin, std::uint32_t end) const {
    return std::ranges::any_of(model_.side_layers[side],
                               [&](std::uint32_t layer) { return layer >= begin && layer < end; });
  }

  struct Group {
    std::vector<std::size_t> slots{};
    std::map<std::pair<std::uint32_t, std::uint32_t>, RangeChoice> ranges{};
  };
  Group& GroupOf(std::uint32_t mask) {
    if (const auto at = groups_.find(mask); at != groups_.end()) return at->second;
    return groups_.emplace(mask, Group{.slots = Ordered(mask)}).first->second;
  }
  const RangeChoice* RangeOf(Group& group, std::uint32_t begin, std::uint32_t end) {
    const auto key = std::pair{begin, end};
    if (const auto at = group.ranges.find(key); at != group.ranges.end()) return &at->second;
    auto choice = ChooseRange(net_, w_, infos_, group.slots, begin, end, work_);
    if (!choice) {
      overflow_ = true;
      return nullptr;
    }
    return &group.ranges.emplace(key, std::move(*choice)).first->second;
  }

  double Boundary(std::uint32_t from, std::uint32_t to, std::uint32_t first_layer) {
    const auto key = std::tuple{from, to, first_layer};
    if (const auto at = boundaries_.find(key); at != boundaries_.end()) return at->second;
    const auto a = Ordered(from);
    const auto b = Ordered(to);
    const double c = Weighted(
        w_, Relay(net_, a, b, BoundaryBytes(model_, first_layer, w_.prefill_rows, 1)),
        Relay(net_, a, b, BoundaryBytes(model_, first_layer, w_.decode_rows, w_.decode_rows)));
    boundaries_.emplace(key, c);
    return c;
  }

  // Groups of unused devices, each class's members taken lowest first.
  std::vector<std::uint32_t> Candidates(std::uint32_t used) const {
    std::vector<std::vector<std::size_t>> free;
    for (const auto& c : classes_) {
      auto& f = free.emplace_back();
      for (const std::size_t s : c) {
        if ((used & (1U << s)) == 0) f.push_back(s);
      }
    }
    std::vector<std::uint32_t> out;
    std::vector<std::size_t> take(free.size(), 0);
    while (true) {
      std::uint32_t mask = 0;
      for (std::size_t c = 0; c < free.size(); ++c) {
        for (std::size_t k = 0; k < take[c]; ++k) mask |= 1U << free[c][k];
      }
      if (mask != 0) out.push_back(mask);
      std::size_t c = 0;
      while (c < free.size() && ++take[c] > free[c].size()) take[c++] = 0;
      if (c == free.size()) break;
    }
    std::ranges::sort(out, [](std::uint32_t a, std::uint32_t b) {
      return std::pair(std::popcount(a), a) < std::pair(std::popcount(b), b);
    });
    return out;
  }

  Value F(std::uint32_t i, std::uint32_t prev, std::uint32_t used,
          const std::vector<std::uint32_t>& side) {
    const StateKey key = Key(i, prev, used, side);
    if (const auto at = memo_.find(key); at != memo_.end()) return at->second;
    if (!Spend(1 + side.size()) || memo_.size() >= kMaxStates) {
      overflow_ = true;
      return {};
    }
    Value best;
    for (const std::uint32_t mask : Candidates(used)) {
      if (!Spend(1)) return {};
      const auto ways = static_cast<std::uint64_t>(std::popcount(mask));
      if (!groups_.contains(mask) && !Spend(std::uint64_t{layers_} * ways)) return {};
      Group& g = GroupOf(mask);
      const double boundary = i == 0 ? 0.0 : Boundary(prev, mask, i);
      if (std::isinf(boundary)) continue;
      double cost = boundary;
      auto next_side = side;
      if (i == 0) std::ranges::fill(next_side, mask);
      for (std::uint32_t j = i + 1; j <= layers_; ++j) {
        if (!Spend(1)) return {};
        if (!Spend(side.size())) return {};
        for (std::size_t k = 0; k < side.size(); ++k) {
          if ((next_side[k] & mask) != 0 || !Reads(k, j - 1, j)) continue;
          const auto holders = Ordered(next_side[k]);
          const auto bytes = infos_.all[infos_.side[k]].side_bytes;
          cost += Weighted(w_, Relay(net_, holders, g.slots, bytes * w_.prefill_rows),
                           Relay(net_, holders, g.slots, bytes * w_.decode_rows));
          next_side[k] |= mask;
        }
        const bool last = j == layers_;
        const auto* range = RangeOf(g, i, j);
        if (!range) return {};
        if (!std::isfinite(range->cost)) continue;
        Value v{.cost = cost + range->cost,
                .devices = static_cast<std::uint32_t>(std::popcount(mask)),
                .stages = 1,
                .group = mask,
                .end = j};
        if (!last) {
          const Value rest = F(j, mask, used | mask, next_side);
          if (overflow_) return {};
          if (std::isinf(rest.cost)) continue;
          v.cost += rest.cost;
          v.devices += rest.devices;
          v.stages += rest.stages;
        }
        if (std::isinf(v.cost)) continue;
        if (Better(v, best)) best = v;
      }
    }
    memo_.emplace(key, best);
    return best;
  }

  const StageModel& model_;
  Net net_;
  const Workload& w_;
  const Infos& infos_;
  std::uint32_t layers_;
  std::vector<std::vector<std::size_t>> classes_;
  std::vector<std::size_t> class_of_;  // by slot
  std::unordered_map<std::uint32_t, Group> groups_;
  std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, double> boundaries_;
  std::unordered_map<StateKey, Value, StateHash> memo_;
  std::uint64_t work_ = 0;
  bool overflow_ = false;
};

}  // namespace

std::expected<Assignment, std::string> Assign(const StageModel& model, const Topology& topology,
                                              const Workload& workload) {
  if (auto v = Validate(topology); !v) return std::unexpected(v.error());
  if (!(workload.decode_share >= 0.0 && workload.decode_share <= 1.0)) {
    return Refused("a decode share outside [0, 1]");
  }
  auto infos = InfosOf(model, workload);
  if (!infos) return std::unexpected(infos.error());
  if (auto prepared = PrepareWork(model, *infos, topology, workload); !prepared)
    return std::unexpected(prepared.error());
  if (infos->layers.empty()) return Refused("a stage without layers");
  Search search(model, topology, workload, *infos);
  auto best = search.Run();
  if (!best) return std::unexpected(best.error());
  return AssignPipeline(model, topology, workload, search.Chosen());
}

std::expected<Assignment, std::string> Assign(const spec::ModelSpec& spec,
                                              std::span<const components::BoundTensor> tensors,
                                              const Topology& topology, const Workload& workload) {
  if (auto v = components::ValidateSpec(spec, tensors); !v) return std::unexpected(v.error());
  auto model = ModelOf(spec, 0, tensors);
  if (!model) return std::unexpected(model.error());
  return Assign(*model, topology, workload);
}

// ---------------------------------------------------------------- partition

namespace {

// How a tensor lies over a group's devices: whole on each, split along a
// dim (part r on rank r), partial sums (the whole is their sum), or an
// expert parallel group's routed slots, valid only where the slot's expert
// is the device's own.
enum class LayKind : std::uint8_t { kFull, kSplit, kPartial, kExperts };
struct Lay {
  LayKind kind = LayKind::kFull;
  std::uint8_t dim = 0;
  std::uint32_t experts = 0;  // kExperts': the experts' count
  bool operator==(const Lay&) const = default;
};
constexpr Lay kFull{};
constexpr Lay kPartial{.kind = LayKind::kPartial};
Lay SplitAt(std::uint8_t dim) { return {.kind = LayKind::kSplit, .dim = dim}; }

bool IsCollective(Op op) {
  return op == Op::kAllReduce || op == Op::kAllGather || op == Op::kAllToAll || op == Op::kSend ||
         op == Op::kRecv;
}
std::uint64_t DenseBytes(const ir::TensorDesc& t) {
  return ir::Elements(t.shape()) * (format::Info(t.format.encoding).element_bits / 8);
}

format::Shape LocalShape(format::Shape shape, const Lay& lay, std::uint32_t g) {
  if (lay.kind == LayKind::kSplit) shape.ne.at(lay.dim) /= g;
  return shape;
}

// A view of a root split along `d` into `g` parts, on one part: its local
// description and the dim it splits along, when each part's elements are
// one rank's (the split's dim covered whole by one of the view's dims, the
// view's finer dims inside one step of it and its coarser dims whole
// multiples of the split's span).
std::optional<std::pair<ir::TensorDesc, std::uint8_t>> LocalView(const ir::TensorDesc& root,
                                                                 std::uint8_t d, std::uint32_t g,
                                                                 const ir::TensorDesc& view) {
  if (root.format.layout != format::Layout::kDense) return std::nullopt;
  const std::uint64_t el = ir::UnitBytes(root.format);
  const auto& n = root.shape().ne;
  if (el == 0 || n.at(d) % g != 0) return std::nullopt;
  std::uint64_t span = 1;
  for (std::size_t k = 0; k <= d; ++k) span *= n.at(k);
  const std::uint64_t block = span / g;
  const auto& m = view.shape().ne;
  std::array<std::uint64_t, format::kMaxDims> s{};
  for (std::size_t k = 0; k < format::kMaxDims; ++k) {
    if (view.nb.at(k) % el != 0) return std::nullopt;
    s.at(k) = view.nb.at(k) / el;
  }
  if (view.view_offset % el != 0) return std::nullopt;
  const std::uint64_t o = view.view_offset / el;
  std::optional<std::size_t> j;
  for (std::size_t k = 0; k < format::kMaxDims; ++k) {
    if (m.at(k) > 1 && s.at(k) * m.at(k) == span && m.at(k) % g == 0) {
      j = k;
      break;
    }
  }
  if (!j) return std::nullopt;
  std::uint64_t inner = o % span;
  for (std::size_t k = 0; k < format::kMaxDims; ++k) {
    if (k == *j || m.at(k) <= 1) continue;
    if (s.at(k) >= span) {
      if (s.at(k) % span != 0) return std::nullopt;
    } else {
      inner += s.at(k) * (m.at(k) - 1);
    }
  }
  if (inner >= s.at(*j)) return std::nullopt;
  ir::TensorDesc local = view;
  local.format.shape.ne.at(*j) /= g;
  for (std::size_t k = 0; k < format::kMaxDims; ++k) {
    const bool outer = s.at(k) >= span && s.at(k) % span == 0;
    local.nb.at(k) = (outer ? s.at(k) / g : s.at(k)) * el;
  }
  local.view_offset = (((o / span) * block) + (o % span)) * el;
  return std::pair{local, static_cast<std::uint8_t>(*j)};
}

// The dim of `to` a split of `from` along `d` into `g` parts becomes when
// reshaped (the same flattened order): one whose prefix of extents ends
// where the split dim's does, and divides.
std::optional<std::uint8_t> ReshapedDim(const format::Shape& from, std::uint8_t d,
                                        const format::Shape& to, std::uint32_t g) {
  if (from.ne.at(d) % g != 0) return std::nullopt;
  std::uint64_t span = 1;
  for (std::size_t k = 0; k <= d; ++k) span *= from.ne.at(k);
  std::uint64_t prefix = 1;
  for (std::size_t e = 0; e < format::kMaxDims; ++e) {
    prefix *= to.ne.at(e);
    if (prefix > span) return std::nullopt;
    if (prefix == span && to.ne.at(e) > 1) {
      if (to.ne.at(e) % g != 0) return std::nullopt;
      return static_cast<std::uint8_t>(e);
    }
  }
  return std::nullopt;
}

// What a node on a split group makes: its output's lay, each source's
// lay as it reads it, and a view's local description.
struct Rule {
  Lay out{};
  std::array<Lay, ir::kMaxSources> want{};
  std::optional<ir::TensorDesc> desc{};
};

class Partitioner {
 public:
  Partitioner(const ir::Graph& g, const Assignment& a)
      : g_(g),
        a_(a),
        net_(a.topology),
        b_(net_.size()),
        leaves_(net_.size()),
        constants_(net_.size()),
        placed_(g.tensor_count()),
        replicable_(g.tensor_count(), false),
        stateful_(g.tensor_count(), false),
        holders_(g.tensor_count()) {}

  std::expected<Partitioned, std::string> Run();

 private:
  struct Where {
    std::vector<std::size_t> slots{};
    Lay lay{};
    std::vector<TensorId> local{};  // by rank
  };
  struct Place {
    const ComponentPlacement* p = nullptr;
    std::vector<std::size_t> slots{};
    const SplitOption* option = nullptr;
  };
  struct LeafAt {
    TensorId local{};
    Lay lay{};
    std::uint32_t parts = 1;
    std::uint32_t part = 0;
  };

  const ir::Node& Producer(TensorId t) const { return g_.node(g_.producer(t)); }
  std::expected<const Place*, std::string> PlaceOf(std::uint16_t component);
  std::uint32_t GroupId(const std::vector<std::size_t>& slots);
  void Context(ir::Builder& b, const ir::Node& node) const;
  std::expected<void, std::string> Ok(std::size_t slot, const ir::Node& node) const;

  Lay Current(TensorId src, const ir::Node& node, const Place& place) const;
  std::optional<Rule> RuleOf(const ir::Node& node, const std::array<Lay, ir::kMaxSources>& cur,
                             const Place& place) const;

  std::expected<void, std::string> Emit(const ir::Node& node, const Place& place);
  std::expected<TensorId, std::string> Materialize(TensorId src, std::size_t slot,
                                                   std::uint32_t rank, const Place& place,
                                                   const Lay& want);
  std::expected<TensorId, std::string> Leaf(TensorId src, std::size_t slot, const Lay& want,
                                            std::uint32_t parts, std::uint32_t part);
  std::expected<TensorId, std::string> FullOn(TensorId t, std::size_t slot);
  std::expected<std::vector<TensorId>, std::string> MakeFull(TensorId t);
  std::expected<TensorId, std::string> SliceOf(TensorId t, std::size_t slot, std::uint8_t dim,
                                               std::uint32_t part, std::uint32_t parts);
  std::expected<TensorId, std::string> LocalIds(TensorId ids, std::size_t slot, std::uint32_t rank,
                                                std::uint32_t parts, std::uint32_t experts,
                                                std::uint16_t component);
  std::expected<std::pair<TensorId, TensorId>, std::string> ExpertMask(
      TensorId ids, std::size_t slot, std::uint32_t rank, std::uint32_t parts,
      std::uint32_t experts, std::uint16_t component);

  const ir::Graph& g_;
  const Assignment& a_;
  Net net_;
  std::vector<ir::Builder> b_;
  std::vector<std::vector<LeafBinding>> leaves_;
  std::vector<std::vector<ConstantValues>> constants_;
  std::vector<std::optional<Where>> placed_;
  std::vector<bool> replicable_;
  std::vector<bool> stateful_;  // its root is a state leaf
  // The devices holding a tensor whole, in the order they came to.
  std::vector<std::vector<std::size_t>> holders_;
  std::map<std::pair<std::uint32_t, std::size_t>, TensorId> full_;
  std::map<std::pair<std::uint32_t, std::size_t>, LeafAt> leaf_;
  std::map<std::tuple<std::uint32_t, std::size_t, std::uint8_t, std::uint32_t, std::uint32_t>,
           TensorId>
      slices_;
  std::map<std::tuple<std::uint32_t, std::size_t, std::uint32_t, std::uint32_t>, TensorId> ids_;
  std::map<std::uint16_t, Place> places_;
  std::map<std::vector<std::size_t>, std::uint32_t> groups_;
  Traffic traffic_;
};

std::expected<const Partitioner::Place*, std::string> Partitioner::PlaceOf(
    std::uint16_t component) {
  if (const auto at = places_.find(component); at != places_.end()) return &at->second;
  const ComponentPlacement* p = a_.Find(component);
  if (p == nullptr || p->def == nullptr) {
    return Refused(std::format("component {} has no placement", component));
  }
  Place place{.p = p, .slots = {}, .option = nullptr};
  for (const std::uint32_t id : p->devices) {
    const auto slot = net_.Slot(id);
    if (!slot) return Refused(std::format("component {} placed on no device {}", component, id));
    place.slots.push_back(*slot);
  }
  if (place.slots.empty()) return Refused(std::format("component {} on no devices", component));
  if (p->option != kWhole && place.slots.size() > 1) {
    if (p->option >= p->def->splits.size()) {
      return Refused(std::format("component {}: no split {}", component, p->option));
    }
    place.option = &p->def->splits[p->option];
  }
  return &places_.emplace(component, std::move(place)).first->second;
}

std::uint32_t Partitioner::GroupId(const std::vector<std::size_t>& slots) {
  return groups_.emplace(slots, static_cast<std::uint32_t>(groups_.size())).first->second;
}

void Partitioner::Context(ir::Builder& b, const ir::Node& node) const {
  b.SetComponent(node.component);
  if (node.batch == ir::Batch::kPerOwner && node.nout > 0) {
    b.SetOwner(g_.tensor(node.out[0]).segment);
  } else {
    b.SetJoined();
  }
}

std::expected<void, std::string> Partitioner::Ok(std::size_t slot, const ir::Node& node) const {
  if (b_[slot].ok()) return {};
  return Refused(std::format("device {}: {} does not verify: {}", net_.Id(slot),
                             ir::OpKeyName(node.key), b_[slot].error()->detail));
}

Lay Partitioner::Current(TensorId src, const ir::Node& node, const Place& place) const {
  const ir::Node& prod = Producer(src);
  const ir::TensorDesc& t = g_.tensor(src);
  if (prod.key.op == Op::kWeight && place.option != nullptr) {
    const auto& ref = std::get<ir::WeightRef>(t.binding);
    std::optional<std::uint16_t> role;
    if (ref.component == node.component) {
      role = ref.role;
    } else {
      for (const WeightAlias& alias : a_.aliases) {
        if (alias.component == node.component && alias.leaf == ref) role = alias.role;
      }
    }
    if (role) {
      for (const components::RoleSplit& rs : place.option->weights) {
        if (rs.role == *role) return SplitAt(rs.dim);
      }
    }
    return kFull;
  }
  if (prod.key.op == Op::kState && place.option != nullptr) {
    const auto& ref = std::get<ir::StateRef>(t.binding);
    if (ref.tensor >= place.p->state_first) {
      for (const components::StateSplit& ss : place.option->state) {
        if (ss.tensor == ref.tensor - place.p->state_first) return SplitAt(ss.dim);
      }
    }
    return kFull;
  }
  if (IsLeaf(prod.key.op) || replicable_[src.v]) return kFull;
  const auto& w = placed_[src.v];
  if (w && w->slots == place.slots) return w->lay;
  return kFull;
}

std::optional<Rule> Partitioner::RuleOf(const ir::Node& node,
                                        const std::array<Lay, ir::kMaxSources>& cur,
                                        const Place& place) const {
  const auto g = static_cast<std::uint32_t>(place.slots.size());
  const auto* traits = ir::Traits(node.key);
  if (!traits || traits->partition == ir::PartitionRule::kWhole) return std::nullopt;
  const auto rule = traits->partition;
  using ir::PartitionRule;
  const auto shape = [&](std::size_t i) -> const format::Shape& {
    return g_.tensor(node.src.at(i)).shape();
  };
  const auto present = [&](std::size_t i) { return i < node.nsrc && node.src.at(i).valid(); };
  const auto split = [](const Lay& l, std::uint8_t d) {
    return l.kind == LayKind::kSplit && l.dim == d;
  };
  // A broadcast operand of `m` along a dim of `n` split `g` ways, read at
  // index i % m: whole where each part's indices map alike (m divides a
  // part), split where it is not broadcast, else it cannot follow.
  const auto broadcast = [&](std::uint64_t m, std::uint64_t n,
                             std::uint8_t d) -> std::optional<Lay> {
    if (m == 1 || (n / g) % m == 0) return kFull;
    if (m == n) return SplitAt(d);
    return std::nullopt;
  };
  Rule r;
  r.want = cur;
  const Lay& x = cur[0];
  // Two operands, b broadcasting over a.
  const auto binary = [&]() -> std::optional<Rule> {
    Lay a = cur[0];
    Lay b = cur[1];
    if (a.kind == LayKind::kExperts || b.kind == LayKind::kExperts) {
      if ((a.kind != LayKind::kExperts && a.kind != LayKind::kFull) ||
          (b.kind != LayKind::kExperts && b.kind != LayKind::kFull)) {
        return std::nullopt;
      }
      r.out = a.kind == LayKind::kExperts ? a : b;
      return r;
    }
    // Partial sums stay partial through a sum of two, or a product with a
    // whole factor; anything else reads them reduced.
    const bool sum = rule == PartitionRule::kAdditive;
    if (a.kind == LayKind::kPartial && b.kind == LayKind::kPartial && sum) {
      r.out = kPartial;
      return r;
    }
    if (a.kind == LayKind::kPartial && b.kind == LayKind::kFull &&
        (rule == PartitionRule::kMultiply || rule == PartitionRule::kDivide)) {
      r.out = kPartial;
      return r;
    }
    if (b.kind == LayKind::kPartial && a.kind == LayKind::kFull &&
        rule == PartitionRule::kMultiply) {
      r.out = kPartial;
      return r;
    }
    if (a.kind == LayKind::kPartial) a = r.want[0] = kFull;
    if (b.kind == LayKind::kPartial) b = r.want[1] = kFull;
    if (a.kind == LayKind::kFull && b.kind == LayKind::kFull) {
      r.out = kFull;
      return r;
    }
    if (a.kind == LayKind::kSplit) {
      const std::uint8_t d = a.dim;
      if (b.kind == LayKind::kSplit) {
        if (b.dim != d) return std::nullopt;
      } else if (shape(1).ne.at(d) == 1) {
        r.want[1] = kFull;
      } else if (shape(1).ne.at(d) == shape(0).ne.at(d)) {
        r.want[1] = SplitAt(d);
      } else {
        return std::nullopt;
      }
      r.out = SplitAt(d);
      return r;
    }
    // a whole, b split: a sliced alike where b does not broadcast.
    if (shape(0).ne.at(b.dim) != shape(1).ne.at(b.dim)) return std::nullopt;
    r.want[0] = SplitAt(b.dim);
    r.out = SplitAt(b.dim);
    return r;
  };
  switch (rule) {
    case PartitionRule::kGlu:
      if (present(1)) return binary();
      [[fallthrough]];
    case PartitionRule::kScale:
    case PartitionRule::kUnary:
    case PartitionRule::kRows:
    case PartitionRule::kChannelNorm:
    case PartitionRule::kPrepare:
    case PartitionRule::kCont: {
      // Over each row: a split of the rows' values cannot pass.
      const bool rows = rule == PartitionRule::kRows || rule == PartitionRule::kPrepare ||
                        rule == PartitionRule::kGlu;
      const format::Shape& to = g_.tensor(node.out[0]).shape();
      if (x.kind == LayKind::kSplit) {
        if ((rows && x.dim == 0) || (rule == PartitionRule::kChannelNorm && x.dim == 1))
          return std::nullopt;
        if (rule == PartitionRule::kCont && to.ne != shape(0).ne) {
          const auto e = ReshapedDim(shape(0), x.dim, to, g);
          if (!e) return std::nullopt;
          r.out = SplitAt(*e);
          return r;
        }
        r.out = x;
        return r;
      }
      if (x.kind == LayKind::kPartial) {
        if (rule == PartitionRule::kScale && node.Params<ir::ScaleParams>().bias == 0.0F) {
          r.out = kPartial;
          return r;
        }
        r.want[0] = kFull;
        r.out = kFull;
        return r;
      }
      if (x.kind == LayKind::kExperts && rule != PartitionRule::kPrepare &&
          rule != PartitionRule::kCont && rule != PartitionRule::kChannelNorm) {
        r.out = x;
        return r;
      }
      return std::nullopt;
    }
    case PartitionRule::kAdditive:
    case PartitionRule::kMultiply:
    case PartitionRule::kDivide:
      return binary();
    case PartitionRule::kView: {
      if (x.kind == LayKind::kFull || x.kind == LayKind::kPartial) {
        r.out = x;
        return r;
      }
      if (x.kind != LayKind::kSplit) return std::nullopt;
      const TensorId root = g_.root(node.src[0]);
      const Lay rl = Current(root, node, place);
      if (rl.kind != LayKind::kSplit) return std::nullopt;
      auto local = LocalView(g_.tensor(root), rl.dim, g, g_.tensor(node.out[0]));
      if (!local) return std::nullopt;
      r.out = SplitAt(local->second);
      r.desc = local->first;
      return r;
    }
    case PartitionRule::kReshape: {
      if (x.kind == LayKind::kFull || x.kind == LayKind::kPartial) {
        r.out = x;
        return r;
      }
      if (x.kind != LayKind::kSplit) return std::nullopt;
      const auto e = ReshapedDim(shape(0), x.dim, g_.tensor(node.out[0]).shape(), g);
      if (!e) return std::nullopt;
      r.out = SplitAt(*e);
      return r;
    }
    case PartitionRule::kPermute:
      if (x.kind == LayKind::kExperts) return std::nullopt;
      r.out = x;
      if (x.kind == LayKind::kSplit) {
        r.out = SplitAt(node.Params<ir::PermuteParams>().axes.at(x.dim));
      }
      return r;
    case PartitionRule::kConcat: {
      const std::uint8_t c = node.Params<ir::ConcatParams>().dim;
      const Lay& s = cur[0].kind == LayKind::kSplit ? cur[0] : cur[1];
      if (s.kind != LayKind::kSplit || s.dim == c) return std::nullopt;
      for (std::size_t i = 0; i < 2; ++i) {
        if (cur.at(i) != s && cur.at(i).kind != LayKind::kFull) return std::nullopt;
        r.want.at(i) = s;
      }
      r.out = s;
      return r;
    }
    case PartitionRule::kMatMul: {
      // out[n, m] = Σ_k a[k, n] b[k, m].
      const Lay& a = cur[0];
      const Lay& b = cur[1];
      const bool full_a = a.kind == LayKind::kFull;
      const bool full_b = b.kind == LayKind::kFull;
      if (split(a, 1) && full_b) {
        r.out = SplitAt(0);
      } else if (split(a, 0) && (split(b, 0) || full_b)) {
        r.want[1] = SplitAt(0);
        r.out = kPartial;
      } else if (full_a && split(b, 0)) {
        r.want[0] = SplitAt(0);
        r.out = kPartial;
      } else if (full_a && split(b, 1)) {
        r.out = b;
      } else if (full_a && b.kind == LayKind::kSplit && b.dim >= 2 && shape(0).ne.at(b.dim) == 1) {
        r.out = b;
      } else if (a.kind == LayKind::kSplit && a.dim >= 2 && a == b) {
        r.out = a;
      } else {
        return std::nullopt;
      }
      return r;
    }
    case PartitionRule::kMatMulId: {
      const Lay& as = cur[0];
      const Lay& b = cur[1];
      if (cur[2].kind != LayKind::kFull) return std::nullopt;
      if (split(as, 2)) {
        // Experts placed whole: each device its own experts' slots.
        const auto experts = static_cast<std::uint32_t>(shape(0).ne[2]);
        if (b.kind != LayKind::kFull && !(b.kind == LayKind::kExperts && b.experts == experts)) {
          return std::nullopt;
        }
        r.out = {.kind = LayKind::kExperts, .dim = 0, .experts = experts};
      } else if (split(as, 1) && b.kind == LayKind::kFull) {
        r.out = SplitAt(0);
      } else if (split(as, 0) && (split(b, 0) || b.kind == LayKind::kFull)) {
        r.want[1] = SplitAt(0);
        r.out = kPartial;
      } else if (as.kind == LayKind::kFull && split(b, 0)) {
        r.want[0] = SplitAt(0);
        r.out = kPartial;
      } else {
        return std::nullopt;
      }
      return r;
    }
    case PartitionRule::kExpertReduce:
      for (std::size_t i = 1; i < 4; ++i) {
        if (present(i) && cur.at(i).kind != LayKind::kFull) return std::nullopt;
      }
      if (x.kind == LayKind::kExperts && !present(2)) return std::nullopt;
      if (x.kind == LayKind::kExperts || x.kind == LayKind::kPartial) {
        r.out = kPartial;
      } else if (split(x, 0)) {
        r.out = x;
      } else {
        return std::nullopt;
      }
      return r;
    case PartitionRule::kGetRows:
      if (split(x, 0) && cur[1].kind == LayKind::kFull) {
        r.out = x;
        return r;
      }
      return std::nullopt;
    case PartitionRule::kCacheStore:
      if (x.kind != LayKind::kSplit || x.dim == 1 || cur[2].kind != LayKind::kFull) {
        return std::nullopt;
      }
      if (cur[1] != x && cur[1].kind != LayKind::kFull) return std::nullopt;
      r.want[1] = x;
      r.out = x;
      return r;
    case PartitionRule::kAttention: {
      // By heads: a query head's KV head is on its device (whole KV heads).
      // ALiBi's slopes follow the global head index: not split.
      if (node.Params<ir::AttnParams>().max_bias > 0.0F) return std::nullopt;
      for (std::size_t i = 0; i < 3; ++i) {
        if (!split(cur.at(i), 2)) return std::nullopt;
      }
      if (present(3)) {
        if (cur[3].kind != LayKind::kFull) return std::nullopt;
        const auto m = broadcast(shape(3).ne[2], shape(0).ne[2], 2);
        if (!m) return std::nullopt;
        r.want[3] = *m;
      }
      if (present(4)) {
        if (cur[4].kind != LayKind::kFull) return std::nullopt;
        r.want[4] = SplitAt(0);
      }
      r.out = SplitAt(1);
      return r;
    }
    case PartitionRule::kSoftMax:
      if (x.kind != LayKind::kSplit || x.dim < 2) return std::nullopt;
      if (x.dim == 2 && node.Params<ir::SoftMaxParams>().max_bias > 0.0F) return std::nullopt;
      if (present(1)) {
        if (cur[1].kind != LayKind::kFull) return std::nullopt;
        const auto m = broadcast(shape(1).ne.at(x.dim), shape(0).ne.at(x.dim), x.dim);
        if (!m) return std::nullopt;
        r.want[1] = *m;
      }
      if (present(2)) {
        if (cur[2].kind != LayKind::kFull) return std::nullopt;
        if (x.dim == 2) r.want[2] = SplitAt(0);
      }
      r.out = x;
      return r;
    case PartitionRule::kArgmax:
      if (split(x, 1)) {
        r.out = SplitAt(0);
        return r;
      }
      return std::nullopt;
    case PartitionRule::kRowReduce:
      if (x.kind == LayKind::kSplit && x.dim != 0) {
        r.out = x;
        return r;
      }
      return std::nullopt;
    case PartitionRule::kRope:
      if (split(x, 1) || split(x, 3)) {
        for (std::size_t i = 1; i < 3; ++i) {
          if (present(i) && cur.at(i).kind != LayKind::kFull) return std::nullopt;
        }
        r.out = x;
        return r;
      }
      return std::nullopt;
    case PartitionRule::kWhole:
      return std::nullopt;
  }
  base::Fatal("invalid declared partition rule");
}

std::expected<TensorId, std::string> Partitioner::Leaf(TensorId src, std::size_t slot,
                                                       const Lay& want, std::uint32_t parts,
                                                       std::uint32_t part) {
  const ir::Node& prod = Producer(src);
  const ir::TensorDesc& t = g_.tensor(src);
  const bool split = want.kind == LayKind::kSplit && parts > 1;
  const Lay lay = split ? want : kFull;
  if (!split) {
    parts = 1;
    part = 0;
  }
  ir::Builder& b = b_[slot];
  if (const auto at = leaf_.find({src.v, slot}); at != leaf_.end()) {
    const LeafAt& have = at->second;
    if (have.lay == lay && have.parts == parts && have.part == part) return have.local;
    if (have.lay == kFull && split && prod.key.op == Op::kWeight) {
      // A part of a weight the device holds whole: a view of it.
      const auto key = std::tuple{src.v, slot, lay.dim, part, parts};
      if (const auto s = slices_.find(key); s != slices_.end()) return s->second;
      const ir::TensorDesc whole = b.tensor(have.local);
      const format::Shape shape = LocalShape(whole.shape(), lay, parts);
      std::uint64_t step = shape.ne.at(lay.dim);
      if (lay.dim == 0) step /= ir::UnitElements(whole.format);
      b.SetComponent(prod.component);
      b.SetJoined();
      const TensorId v = b.View(have.local, shape, whole.nb, part * step * whole.nb.at(lay.dim));
      if (auto ok = Ok(slot, prod); !ok) return std::unexpected(ok.error());
      slices_.emplace(key, v);
      return v;
    }
    return Refused(std::format("device {} needs {} in two forms", net_.Id(slot), g_.name(src)));
  }
  ir::TensorDesc r = t;
  if (split) {
    if (t.shape().ne.at(lay.dim) % parts != 0) {
      return Refused(
          std::format("{} does not split {} ways along {}", g_.name(src), parts, lay.dim));
    }
    r.format = Local(t.format, lay.dim, parts);
  }
  b.SetComponent(prod.component);
  b.SetJoined();
  const TensorId local = b.Emit(prod.key, prod.params, {}, std::span(&r, 1), g_.name(src))[0];
  if (auto ok = Ok(slot, prod); !ok) return std::unexpected(ok.error());
  leaf_.emplace(std::pair{src.v, slot},
                LeafAt{.local = local, .lay = lay, .parts = parts, .part = part});
  leaves_[slot].push_back(
      {.local = local, .source = src, .slice = {.dim = lay.dim, .part = part, .parts = parts}});
  return local;
}

std::expected<std::vector<TensorId>, std::string> Partitioner::MakeFull(TensorId t) {
  Where& w = *placed_[t.v];
  const ir::TensorDesc& desc = g_.tensor(t);
  const ir::Node& prod = Producer(t);
  if (w.lay.kind == LayKind::kFull) return w.local;
  if (!stateful_[t.v] && !w.slots.empty() && full_.contains({t.v, w.slots.front()})) {
    std::vector<TensorId> out;
    out.reserve(w.slots.size());
    for (const std::size_t s : w.slots) out.push_back(full_.at({t.v, s}));
    return out;
  }
  if (w.lay.kind == LayKind::kExperts) {
    return Refused(std::format("{}: an expert-local value read whole", g_.name(t)));
  }
  const std::uint32_t group = GroupId(w.slots);
  const bool gather = w.lay.kind == LayKind::kSplit;
  std::vector<TensorId> out;
  for (std::size_t r = 0; r < w.slots.size(); ++r) {
    ir::Builder& b = b_[w.slots[r]];
    b.SetComponent(prod.component);
    b.SetJoined();
    out.push_back(gather ? b.AllGather(w.local[r], desc.shape(), {.group = group, .dim = w.lay.dim})
                         : b.AllReduce(w.local[r], {.group = group}));
    if (auto ok = Ok(w.slots[r], prod); !ok) return std::unexpected(ok.error());
  }
  Add(traffic_,
      Collective(net_, gather ? Op::kAllGather : Op::kAllReduce, w.slots, DenseBytes(desc)));
  if (!gather) {
    // Reduced once: its later readers read the sums.
    w.lay = kFull;
    w.local = out;
  }
  if (!stateful_[t.v]) {
    for (std::size_t r = 0; r < w.slots.size(); ++r) full_[{t.v, w.slots[r]}] = out[r];
  }
  return out;
}

std::expected<TensorId, std::string> Partitioner::FullOn(TensorId t, std::size_t slot) {
  if (!stateful_[t.v]) {
    if (const auto at = full_.find({t.v, slot}); at != full_.end()) return at->second;
  }
  const ir::Node& prod = Producer(t);
  if (IsLeaf(prod.key.op)) return Leaf(t, slot, kFull, 1, 0);
  if (replicable_[t.v]) {
    // Made from inputs alone: made again on each device that reads it.
    std::array<TensorId, ir::kMaxSources> sources{};
    for (std::size_t i = 0; i < prod.nsrc; ++i) {
      if (!prod.src.at(i).valid()) continue;
      auto s = FullOn(prod.src.at(i), slot);
      if (!s) return s;
      sources.at(i) = *s;
    }
    std::vector<ir::TensorDesc> requested;
    for (const TensorId o : prod.outputs()) requested.push_back(g_.tensor(o));
    ir::Builder& b = b_[slot];
    Context(b, prod);
    const ir::Outputs outs = b.Emit(prod.key, prod.params, std::span(sources).first(prod.nsrc),
                                    requested, prod.nout > 0 ? g_.name(prod.out[0]) : "");
    if (auto ok = Ok(slot, prod); !ok) return std::unexpected(ok.error());
    for (std::size_t i = 0; i < prod.nout; ++i) full_[{prod.out.at(i).v, slot}] = outs[i];
    return full_.at({t.v, slot});
  }
  if (!placed_[t.v]) return Refused(std::format("{} is read before it is made", g_.name(t)));
  const std::vector<std::size_t> slots = placed_[t.v]->slots;
  const auto rank = std::ranges::find(slots, slot);
  auto whole = MakeFull(t);
  if (!whole) return std::unexpected(whole.error());
  if (rank != slots.end()) return (*whole)[static_cast<std::size_t>(rank - slots.begin())];
  if (stateful_[t.v]) return Refused(std::format("{}: state read on another device", g_.name(t)));
  // A send from the holder with the cheapest link.
  auto& holders = holders_[t.v];
  if (holders.empty()) holders = slots;
  const ir::TensorDesc& desc = g_.tensor(t);
  const std::uint64_t bytes = DenseBytes(desc);
  const auto from = BestHolder(net_, holders, slot, static_cast<double>(bytes));
  if (!from) {
    return Refused(std::format("no link brings {} to device {}", g_.name(t), net_.Id(slot)));
  }
  auto x = FullOn(t, *from);
  if (!x) return x;
  const std::vector<std::size_t> pair{std::min(*from, slot), std::max(*from, slot)};
  const std::uint32_t group = GroupId(pair);
  const auto peer = [&](std::size_t s) { return s == pair[0] ? 0U : 1U; };
  ir::Builder& src = b_[*from];
  src.SetComponent(prod.component);
  src.SetJoined();
  src.Send(*x, {.group = group, .peer = peer(slot)});
  if (auto ok = Ok(*from, prod); !ok) return std::unexpected(ok.error());
  ir::Builder& dst = b_[slot];
  dst.SetComponent(prod.component);
  dst.SetJoined();
  const TensorId got = dst.Recv(ir::Dense(desc.format.encoding, desc.shape()),
                                {.group = group, .peer = peer(*from)});
  if (auto ok = Ok(slot, prod); !ok) return std::unexpected(ok.error());
  Add(traffic_, {.bytes = bytes, .seconds = net_.Seconds(*from, slot, static_cast<double>(bytes))});
  full_[{t.v, slot}] = got;
  holders.push_back(slot);
  return got;
}

std::expected<TensorId, std::string> Partitioner::SliceOf(TensorId t, std::size_t slot,
                                                          std::uint8_t dim, std::uint32_t part,
                                                          std::uint32_t parts) {
  const auto key = std::tuple{t.v, slot, dim, part, parts};
  if (!stateful_[t.v]) {
    if (const auto at = slices_.find(key); at != slices_.end()) return at->second;
  }
  auto x = FullOn(t, slot);
  if (!x) return x;
  ir::Builder& b = b_[slot];
  const ir::TensorDesc whole = b.tensor(*x);
  if (whole.format.layout != format::Layout::kDense || whole.shape().ne.at(dim) % parts != 0) {
    return Refused(std::format("{} does not split {} ways along {}", g_.name(t), parts, dim));
  }
  const format::Shape shape = LocalShape(whole.shape(), SplitAt(dim), parts);
  const ir::Node& prod = Producer(t);
  b.SetComponent(prod.component);
  b.SetJoined();
  const TensorId v =
      b.Cont(b.View(*x, shape, whole.nb, part * shape.ne.at(dim) * whole.nb.at(dim)));
  if (auto ok = Ok(slot, prod); !ok) return std::unexpected(ok.error());
  if (!stateful_[t.v]) slices_.emplace(key, v);
  return v;
}

std::expected<TensorId, std::string> Partitioner::Materialize(TensorId src, std::size_t slot,
                                                              std::uint32_t rank,
                                                              const Place& place, const Lay& want) {
  const auto parts = static_cast<std::uint32_t>(place.slots.size());
  const Op op = Producer(src).key.op;
  if (op == Op::kWeight || op == Op::kState) return Leaf(src, slot, want, parts, rank);
  if (want.kind == LayKind::kFull) return FullOn(src, slot);
  const auto& w = placed_[src.v];
  if (w && w->slots == place.slots && w->lay == want) return w->local[rank];
  if (want.kind != LayKind::kSplit) {
    return Refused(std::format("{} is not where its reader needs it", g_.name(src)));
  }
  return SliceOf(src, slot, want.dim, rank, parts);
}

std::expected<TensorId, std::string> Partitioner::LocalIds(TensorId ids, std::size_t slot,
                                                           std::uint32_t rank, std::uint32_t parts,
                                                           std::uint32_t experts,
                                                           std::uint16_t component) {
  const auto key = std::tuple{ids.v, slot, rank, parts};
  if (const auto at = ids_.find(key); at != ids_.end()) return at->second;
  auto whole = FullOn(ids, slot);
  if (!whole) return whole;
  ir::Builder& b = b_[slot];
  b.SetComponent(component);
  b.SetJoined();
  const ir::TensorDesc d = b.tensor(*whole);
  const std::uint64_t used = d.ne(0);
  const std::uint64_t tokens = d.ne(1);
  const TensorId dense = ir::IsContiguous(d) ? *whole : b.Cont(*whole);
  const TensorId flat = b.Reshape(dense, ir::ShapeOf({used * tokens}));
  // Each expert's index among the device's own; another's, 0 (its slot is
  // masked out of the sum).
  ConstantValues values{.local = {}, .ints = {}, .values = {}};
  const std::uint32_t own = experts / parts;
  for (std::uint32_t e = 0; e < experts; ++e) {
    values.ints.push_back(static_cast<std::int32_t>(e / own == rank ? e - (rank * own) : 0));
  }
  // Each rank's table digests by its values (ir Builder::Constant).
  const TensorId table = b.Constant(ir::Dense(format::Encoding::kI32, ir::ShapeOf({1, experts})),
                                    "expert_ids", std::as_bytes(std::span(values.ints)));
  values.local = table;
  constants_[slot].push_back(std::move(values));
  const TensorId local = b.Reshape(b.GetRows(table, flat), ir::ShapeOf({used, tokens}));
  if (!b.ok()) return Refused(std::format("expert ids: {}", b.error()->detail));
  ids_.emplace(key, local);
  return local;
}

std::expected<std::pair<TensorId, TensorId>, std::string> Partitioner::ExpertMask(
    TensorId ids, std::size_t slot, std::uint32_t rank, std::uint32_t parts, std::uint32_t experts,
    std::uint16_t component) {
  auto original = FullOn(ids, slot);
  if (!original) return std::unexpected(original.error());
  ir::Builder& b = b_[slot];
  b.SetComponent(component);
  b.SetJoined();
  ConstantValues values;
  const auto own = experts / parts;
  for (std::uint32_t e = 0; e < experts; ++e) {
    values.ints.push_back(static_cast<std::int32_t>(e / own == rank ? 1 : 0));
  }
  const auto table = b.Constant(ir::Dense(format::Encoding::kI32, ir::ShapeOf({experts})),
                                "expert_ownership", std::as_bytes(std::span(values.ints)));
  values.local = table;
  constants_[slot].push_back(std::move(values));
  const auto tokens = g_.tensor(ids).ne(1);
  const auto rows =
      b.Repeat(b.Reshape(table, ir::ShapeOf({1, experts, 1})), ir::ShapeOf({1, experts, tokens}));
  const auto selected = b.GetRows(rows, *original);
  if (!b.ok()) return Refused(std::format("expert ownership: {}", b.error()->detail));
  return std::pair{table, selected};
}

std::expected<void, std::string> Partitioner::Emit(const ir::Node& node, const Place& place) {
  const auto g = static_cast<std::uint32_t>(place.slots.size());
  std::array<Lay, ir::kMaxSources> cur{};
  for (std::size_t i = 0; i < node.nsrc; ++i) {
    if (node.src.at(i).valid()) cur.at(i) = Current(node.src.at(i), node, place);
  }
  bool whole = place.option == nullptr ||
               std::ranges::all_of(std::span(cur).first(node.nsrc),
                                   [](const Lay& l) { return l.kind == LayKind::kFull; });
  std::optional<Rule> rule;
  if (!whole) rule = RuleOf(node, cur, place);
  const bool fallback = !whole && !rule;
  if (fallback) {
    // Its sources gathered, it runs whole on each device of the group.
    for (std::size_t i = 0; i < node.nsrc; ++i) {
      if (cur.at(i).kind == LayKind::kExperts) {
        return Refused(
            std::format("{}: an expert-local value it cannot take", ir::OpKeyName(node.key)));
      }
    }
    if (node.key.op == Op::kCacheStore && cur[0].kind == LayKind::kSplit) {
      return Refused(std::format("a store into a split cache its rows cannot follow ({})",
                                 g_.name(node.src[0])));
    }
  }
  const bool experts_ids =
      rule && rule->out.kind == LayKind::kExperts && node.key.op == Op::kMatMulId;
  const bool experts_mask =
      rule && node.key.op == Op::kExpertReduce && cur[0].kind == LayKind::kExperts;
  std::vector<std::vector<TensorId>> locals(node.nout);
  for (std::uint32_t rank = 0; rank < g; ++rank) {
    const std::size_t slot = place.slots[rank];
    std::array<TensorId, ir::kMaxSources> sources{};
    for (std::size_t i = 0; i < node.nsrc; ++i) {
      const TensorId src = node.src.at(i);
      if (!src.valid()) continue;
      std::expected<TensorId, std::string> s;
      if (rule) {
        s = Materialize(src, slot, rank, place, rule->want.at(i));
      } else if (node.key.op == Op::kView) {
        // Its description is its root's: the root whole (a source's whole
        // copy made by a collective or a receive is a root of its own).
        s = FullOn(g_.root(src), slot);
      } else {
        s = FullOn(src, slot);
      }
      if (!s) return std::unexpected(s.error());
      sources.at(i) = *s;
    }
    std::vector<ir::TensorDesc> requested;
    for (const TensorId o : node.outputs()) {
      ir::TensorDesc d = g_.tensor(o);
      if (rule && rule->desc) {
        d = *rule->desc;
      } else if (rule) {
        d.format.shape = LocalShape(d.format.shape, rule->out, g);
      }
      requested.push_back(d);
    }
    std::size_t nsrc = node.nsrc;
    if (experts_ids) {
      auto ids = LocalIds(node.src[2], slot, rank, g, rule->out.experts, node.component);
      if (!ids) return std::unexpected(ids.error());
      sources[2] = *ids;
    }
    if (experts_mask) {
      auto mask = ExpertMask(node.src[2], slot, rank, g, cur[0].experts, node.component);
      if (!mask) return std::unexpected(mask.error());
      auto& builder = b_[slot];
      Context(builder, node);
      const auto zero = [&](TensorId value) {
        return builder.Fill(ir::ShapeOf({1}), 0, builder.tensor(value).format.encoding);
      };
      // Literal selection, before any reduction arithmetic. Selecting products
      // alone would still allow 0 * Inf/NaN in a foreign routing weight/scale.
      // Original expert ids remain authoritative for the reduction and scales.
      sources[0] = builder.Select(mask->second, sources[0], zero(sources[0]));
      const auto weights = builder.tensor(sources[1]);
      const auto weight_mask = builder.Reshape(mask->second, weights.shape());
      sources[1] = builder.Select(weight_mask, sources[1], zero(sources[1]));
      if (node.nsrc > 3 && node.src[3].valid()) {
        const auto scale = builder.tensor(sources[3]);
        const auto& ids = g_.tensor(node.src[2]);
        const bool gathered = scale.ne(0) == 1 && scale.ne(1) == ids.ne(0) &&
                              scale.ne(2) == ids.ne(1) && scale.ne(3) == 1;
        const auto one = builder.Fill(ir::ShapeOf({1}), 1, scale.format.encoding);
        sources[3] = builder.Select(gathered ? mask->second : mask->first, sources[3], one);
      }
    }
    ir::Builder& b = b_[slot];
    Context(b, node);
    const ir::Outputs outs = b.Emit(node.key, node.params, std::span(sources).first(nsrc),
                                    requested, node.nout > 0 ? g_.name(node.out[0]) : "");
    if (auto ok = Ok(slot, node); !ok) return std::unexpected(ok.error());
    for (std::size_t i = 0; i < node.nout; ++i) locals[i].push_back(outs[i]);
  }
  const Lay out = rule ? rule->out : kFull;
  for (std::size_t i = 0; i < node.nout; ++i) {
    placed_[node.out.at(i).v] = Where{.slots = place.slots, .lay = out, .local = locals[i]};
  }
  return {};
}

std::expected<Partitioned, std::string> Partitioner::Run() {
  for (const ir::Node& node : g_.nodes()) {
    const Op op = node.key.op;
    if (op == Op::kFused) {
      return Refused("a fused node: partition the primitive graph, before patterns");
    }
    if (IsCollective(op)) return Refused("a collective in the graph to partition");
    bool rep = op == Op::kInput;
    if (!IsLeaf(op)) {
      rep = true;
      for (const TensorId s : node.sources()) rep = rep && (!s.valid() || replicable_[s.v]);
    }
    for (const TensorId o : node.outputs()) {
      replicable_[o.v] = rep;
      stateful_[o.v] = Producer(g_.root(o)).key.op == Op::kState;
    }
  }
  for (const ir::Node& node : g_.nodes()) {
    if (IsLeaf(node.key.op) || node.nout == 0 || replicable_[node.out[0].v]) continue;
    auto place = PlaceOf(node.component);
    if (!place) return std::unexpected(place.error());
    if (auto e = Emit(node, **place); !e) return std::unexpected(e.error());
  }
  Partitioned out;
  const auto home = [&](TensorId t) -> std::size_t {
    return placed_[t.v] ? placed_[t.v]->slots.front() : 0;
  };
  for (const TensorId t : g_.outputs()) {
    const std::size_t slot = home(t);
    auto local = FullOn(t, slot);
    if (!local) return std::unexpected(local.error());
    b_[slot].Output(*local);
    out.outputs.push_back({.source = t, .device = net_.Id(slot), .local = *local});
  }
  for (const TensorId t : g_.keep()) {
    const std::size_t slot = home(t);
    auto local = FullOn(t, slot);
    if (!local) return std::unexpected(local.error());
    b_[slot].Keep(*local);
    out.kept.push_back({.source = t, .device = net_.Id(slot), .local = *local});
  }
  for (std::size_t slot = 0; slot < net_.size(); ++slot) {
    if (b_[slot].node_count() == 0) continue;
    auto graph = std::move(b_[slot]).Finish();
    if (!graph) {
      return Refused(std::format("device {}'s graph does not verify at node {}: {}", net_.Id(slot),
                                 graph.error().node, graph.error().detail));
    }
    out.devices.push_back({.device = net_.Id(slot),
                           .graph = std::move(*graph),
                           .leaves = std::move(leaves_[slot]),
                           .constants = std::move(constants_[slot])});
  }
  for (const auto& [slots, id] : groups_) {
    CollectiveGroup group{.group = id, .devices = {}};
    for (const std::size_t s : slots) group.devices.push_back(net_.Id(s));
    out.groups.push_back(std::move(group));
  }
  std::ranges::sort(out.groups, {}, &CollectiveGroup::group);
  out.traffic = traffic_;
  return out;
}

}  // namespace

std::expected<Partitioned, std::string> Partition(const ir::Graph& graph,
                                                  const Assignment& assignment) {
  if (auto v = Validate(assignment.topology); !v) return std::unexpected(v.error());
  Partitioner p(graph, assignment);
  return p.Run();
}

}  // namespace llmp::planner
