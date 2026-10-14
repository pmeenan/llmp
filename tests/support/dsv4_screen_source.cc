// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "dsv4_screen_source.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <set>

#include "ir/builder.h"
#include "planner/pattern.h"
#include "planner/plan.h"

namespace llmp::test_support::dsv4_screen {
namespace co = components;
namespace pl = planner;
namespace {
using format::Encoding;
using ir::NodeId;
using ir::Op;
using ir::TensorId;
using Error = std::unexpected<std::string>;
const ir::Node* Made(const ir::Graph& g, TensorId t, Op op) {
  if (!t.valid()) return nullptr;
  const auto& n = g.node(g.producer(t));
  return n.key == ir::Primitive(op) && n.nout == 1 && n.out[0] == t ? &n : nullptr;
}
struct Selection {
  NodeId score, top, concat;
  TensorId query, key, weights, raw;
  co::CellKey cells;
  std::uint32_t read = 0, k = 0, raw_cells = 0;
  std::set<std::uint32_t> erased;
};
std::expected<Selection, std::string> Match(const ir::Graph& g, const ir::Node& cat,
                                            const co::StepGraph& step,
                                            const model::StateLayoutDerived& layout) {
  const auto* add = Made(g, cat.src[1], Op::kAdd);
  const auto* rows = add ? Made(g, add->src[0], Op::kView) : nullptr;
  const auto* store = rows ? Made(g, rows->src[0], Op::kCacheStore) : nullptr;
  const auto* all4 = store ? Made(g, store->src[0], Op::kView) : nullptr;
  const auto* zeros = store ? Made(g, store->src[1], Op::kFill) : nullptr;
  const auto* best3 = store ? Made(g, store->src[2], Op::kView) : nullptr;
  const auto* all = all4 ? Made(g, all4->src[0], Op::kFill) : nullptr;
  const auto* cont = best3 ? Made(g, best3->src[0], Op::kCont) : nullptr;
  const auto* top = cont ? Made(g, cont->src[0], Op::kTopK) : nullptr;
  const auto* score = top ? Made(g, top->src[0], Op::kIndexScore) : nullptr;
  if (!score || !all || !zeros || score->source(3) != add->src[1] ||
      cat.Params<ir::ConcatParams>().dim != 0 || cat.batch != ir::Batch::kPerOwner ||
      zeros->Params<ir::FillParams>().value != 0.0F ||
      all->Params<ir::FillParams>().value != -std::numeric_limits<float>::infinity())
    return Error("not the complete canonical mqa@1 selected mask");
  const auto& kd = g.tensor(score->src[1]);
  const auto& qd = g.tensor(score->src[0]);
  const auto& wd = g.tensor(score->src[2]);
  const auto& raw = g.tensor(cat.src[0]);
  const auto C = kd.ne(2), T = qd.ne(2);
  const auto K = top->Params<ir::TopKParams>().k;
  if (qd.shape().ne != std::array<std::uint64_t, 4>{128, 64, T, 1} ||
      kd.shape().ne != std::array<std::uint64_t, 4>{128, 1, C, 1} ||
      wd.shape().ne != std::array<std::uint64_t, 4>{64, T, 1, 1} || raw.ne(1) != T ||
      raw.ne(2) != 1 || raw.ne(3) != 1 || raw.format.encoding != Encoding::kF16 ||
      K != std::min<std::uint64_t>(512, C) || C > UINT32_MAX || raw.ne(0) > UINT32_MAX ||
      g.tensor(all->out[0]).shape().ne != std::array<std::uint64_t, 4>{C, T, 1, 1} ||
      g.tensor(zeros->out[0]).shape().ne != std::array<std::uint64_t, 4>{1, K, T, 1})
    return Error("the private CSA-4 recipe has unexpected dimensions");
  // Authenticate the view identities and all scalar/formula parameters by
  // independently emitting precisely the old formula over its boundaries.
  ir::Builder expected;
  const auto sc = expected.Input(g.tensor(score->out[0]).format);
  const auto cm = expected.Input(g.tensor(add->src[1]).format);
  const auto rr = expected.Input(raw.format);
  const auto z = expected.Fill(ir::ShapeOf({1, K, T, 1}), 0.0F, Encoding::kF16);
  const auto tk = expected.TopK(sc, K);
  const auto packed = expected.Cont(tk);
  const auto pd = expected.tensor(packed);
  const auto ids = expected.View(packed, ir::ShapeOf({K, T, 1, 1}), pd.nb, 0);
  const auto f =
      expected.Fill(ir::ShapeOf({C, T}), -std::numeric_limits<float>::infinity(), Encoding::kF16);
  const auto fd = expected.tensor(f);
  const auto f4 =
      expected.View(f, ir::ShapeOf({1, C, T, 1}), {fd.nb[0], fd.nb[0], fd.nb[1], fd.nb[2]}, 0);
  const auto st = expected.CacheStore(f4, z, ids);
  const auto sd = expected.tensor(st);
  const auto v =
      expected.View(st, ir::ShapeOf({C, T, 1, 1}), {sd.nb[0], sd.nb[2], sd.nb[3], sd.nb[3]}, 0);
  const auto a = expected.Add(v, cm);
  const auto result = expected.Concat(rr, a, 0);
  expected.Output(result);
  auto formula = std::move(expected).Finish();
  if (!formula) return Error("private canonical mask template failed");
  const std::array pairs{std::pair{z, zeros},    std::pair{tk, top}, std::pair{packed, cont},
                         std::pair{ids, best3},  std::pair{f, all},  std::pair{f4, all4},
                         std::pair{st, store},   std::pair{v, rows}, std::pair{a, add},
                         std::pair{result, &cat}};
  std::map<std::uint32_t, TensorId> bindings{
      {sc.v, score->out[0]}, {cm.v, add->src[1]}, {rr.v, cat.src[0]}};
  Selection match{.score = g.producer(score->out[0]),
                  .top = g.producer(top->out[0]),
                  .concat = g.producer(cat.out[0]),
                  .query = score->src[0],
                  .key = score->src[1],
                  .weights = score->src[2],
                  .raw = cat.src[0],
                  .cells = {},
                  .read = static_cast<std::uint32_t>(C),
                  .k = K,
                  .raw_cells = static_cast<std::uint32_t>(raw.ne(0)),
                  .erased = {}};
  for (const auto& [et, actual] : pairs) {
    const auto& en = formula->node(formula->producer(et));
    const auto& ed = formula->tensor(et);
    const auto& ad = g.tensor(actual->out[0]);
    if (en.key != actual->key || en.params != actual->params || en.nsrc != actual->nsrc ||
        ed.format != ad.format || ed.nb != ad.nb || ed.view_offset != ad.view_offset ||
        actual->component != cat.component)
      return Error("private selected-mask formula/alias mismatch");
    for (std::size_t i = 0; i < en.nsrc; ++i)
      if (!bindings.contains(en.src[i].v) || bindings.at(en.src[i].v) != actual->src[i])
        return Error("private selected-mask operand ordering mismatch");
    bindings[et.v] = actual->out[0];
    match.erased.insert(g.producer(actual->out[0]).v);
  }
  match.erased.insert(match.score.v);
  const auto root = g.root(match.key);
  const auto at = std::ranges::find(step.state, root, &co::StepState::tensor);
  if (at == step.state.end() || at->borrowed || at->tap || at->ref.stage != 0 ||
      at->ref.segment != 0)
    return Error("indexer keys lack their sole owned state binding");
  const auto& item = layout.items.at(at->item);
  const auto& contract = item.contract;
  if (contract.cells != model::CellRule::kPerBlock || contract.window_or_ratio != 4 ||
      contract.basis != model::BlockBasis::kPositions || !contract.dummy_last ||
      contract.partial_blocks || contract.encoding != Encoding::kF16 || contract.cell_bytes != 256)
    return Error("indexer keys are not the completed CSA-4 cell contract");
  match.cells = {
      .capacity = item.cells, .block = 4, .positions = true, .cell_pad = contract.cell_pad};
  // No external observer, alias or write of an eliminated node may survive.
  // The mask output is the single intentional replacement boundary.
  for (const auto& n : g.nodes()) {
    if (match.erased.contains(g.producer(n.out[0]).v)) continue;
    for (const auto input : n.sources()) {
      if (!input.valid() || input == cat.out[0]) continue;
      if (match.erased.contains(g.producer(input).v) ||
          match.erased.contains(g.producer(g.root(input)).v))
        return Error("old score/selection/mask has an outside observer");
    }
  }
  for (const auto list : {g.outputs(), g.keep()})
    for (const auto t : list)
      if (t != cat.out[0] && (match.erased.contains(g.producer(t).v) ||
                              match.erased.contains(g.producer(g.root(t)).v)))
        return Error("old score/selection retained for diagnostics");
  return match;
}

// Authenticate the ordinary HCA-128 boundary independently of CSA's selected
// mask. The two KV banks, both mask position inputs and the layer's sink must
// be the actual owned bindings; a similar-shaped unrelated concat is refused.
bool HcaMask(const ir::Graph& g, const ir::Node& n) {
  const auto* cat = Made(g, n.source(3), Op::kConcat);
  const auto* rows = cat ? Made(g, cat->src[1], Op::kView) : nullptr;
  return rows && Made(g, rows->src[0], Op::kMaskBuild);
}
std::expected<std::uint32_t, std::string> HcaBound(const co::StepGraph& step, const ir::Node& n,
                                                   const pl::StepShape& shape,
                                                   const model::StateLayoutDerived& layout) {
  const auto& g = step.graph;
  const auto& seg = shape.segments.front();
  const auto* mask_cat = Made(g, n.source(3), Op::kConcat);
  const auto* kp = Made(g, n.source(1), Op::kPermute);
  const auto* vp = Made(g, n.source(2), Op::kPermute);
  const auto* bank = kp ? Made(g, kp->src[0], Op::kConcat) : nullptr;
  if (!mask_cat || !kp || !vp || !bank || vp->src[0] != kp->src[0] ||
      mask_cat->Params<ir::ConcatParams>().dim != 0 || bank->Params<ir::ConcatParams>().dim != 2 ||
      kp->Params<ir::PermuteParams>().axes != std::array<std::uint8_t, 4>{0, 2, 1, 3} ||
      vp->params != kp->params || n.batch != ir::Batch::kPerOwner)
    return Error("private HCA attention lacks its canonical two-bank operands");
  std::array<const co::StepState*, 2> states{};
  std::array<const co::StepInput*, 2> held{};
  std::array<const ir::Node*, 2> masks{};
  std::array<std::uint64_t, 2> reads{};
  for (std::size_t i = 0; i < 2; ++i) {
    const auto* kv = Made(g, bank->src[i], Op::kView);
    const auto* rows = Made(g, mask_cat->src[i], Op::kView);
    masks[i] = rows ? Made(g, rows->src[0], Op::kMaskBuild) : nullptr;
    if (!kv || !rows || !masks[i] || !Made(g, kv->src[0], Op::kState))
      return Error("private HCA boundary is not a direct owned state/mask view");
    const auto st = std::ranges::find(step.state, kv->src[0], &co::StepState::tensor);
    if (st == step.state.end() || st->borrowed || st->tap || st->ref.stage != 0 ||
        st->ref.segment != 0 || st->item >= layout.items.size())
      return Error("private HCA state binding is not sole owned target state");
    states[i] = &*st;
    const auto& item = layout.items[st->item];
    const auto& c = item.contract;
    const bool raw = i == 0;
    if (item.layer != st->ref.layer || item.tensor != st->ref.tensor ||
        c.encoding != Encoding::kF16 || c.cell_bytes != 1024 ||
        (raw ? c.kind != model::StateKind::kRing || c.cells != model::CellRule::kRing ||
                   c.window_or_ratio != 128 || !c.ring_reads_whole
             : c.kind != model::StateKind::kCompressed || c.cells != model::CellRule::kPerBlock ||
                   c.window_or_ratio != 128 || c.basis != model::BlockBasis::kPositions ||
                   !c.dummy_last || c.partial_blocks))
      return Error("private HCA state is not the fixed ring/HCA-128 contract");
    const auto& root = g.tensor(kv->src[0]);
    const auto& view = g.tensor(kv->out[0]);
    const auto* binding = std::get_if<ir::StateRef>(&root.binding);
    if (!binding || *binding != st->ref || root.segment != 0)
      return Error("private HCA state descriptor disagrees with its owned binding");
    reads[i] = view.ne(2);
    const auto want_read =
        raw ? std::min<std::uint64_t>(item.cells, seg.read_cells)
            : std::min(item.cells, co::CompressedReadCells(std::uint64_t{seg.past} + seg.rows, 128,
                                                           c.cell_pad));
    if (root.format.encoding != Encoding::kF16 ||
        root.shape().ne != std::array<std::uint64_t, 4>{512, item.cells, 1, 1} ||
        !ir::IsContiguous(root) || reads[i] != want_read || reads[i] == 0 ||
        view.shape().ne != std::array<std::uint64_t, 4>{512, 1, reads[i], 1} ||
        view.view_offset != 0 ||
        view.nb != std::array<std::uint64_t, 4>{2, 1024, 1024, 1024 * item.cells})
      return Error("private HCA cache view differs from its actual read bound");
    const auto input = std::ranges::find(step.inputs, masks[i]->src[1], &co::StepInput::tensor);
    const co::CellKey key =
        raw ? co::CellKey{.capacity = item.cells, .ring = true}
            : co::CellKey{
                  .capacity = item.cells, .block = 128, .positions = true, .cell_pad = c.cell_pad};
    const ir::MaskParams mp{.kind = raw ? ir::MaskKind::kWindow : ir::MaskKind::kCausal,
                            .window = raw ? 128U : 0U};
    const auto& full = g.tensor(masks[i]->out[0]);
    const auto& sliced = g.tensor(rows->out[0]);
    if (input == step.inputs.end() || input->kind != co::InputKind::kCellPositions ||
        input->segment != 0 || input->cells != key || input->read != reads[i] ||
        masks[i]->Params<ir::MaskParams>().kind != mp.kind ||
        masks[i]->Params<ir::MaskParams>().window != mp.window ||
        masks[i]->Params<ir::MaskParams>().block != 0 ||
        masks[i]->Params<ir::MaskParams>().exact_rows ||
        masks[i]->Params<ir::MaskParams>().reserved != std::array<std::uint8_t, 2>{} ||
        full.format.encoding != Encoding::kF16 || !ir::IsContiguous(full) ||
        full.ne(0) != reads[i] || full.ne(1) < seg.rows || full.ne(2) != 1 || full.ne(3) != 1 ||
        sliced.shape().ne != std::array<std::uint64_t, 4>{reads[i], seg.rows, 1, 1} ||
        sliced.view_offset != 0 ||
        sliced.nb != std::array<std::uint64_t, 4>{2, 2 * reads[i], 2 * reads[i] * seg.rows,
                                                  2 * reads[i] * seg.rows})
      return Error("private HCA mask lacks its exact owned cell-position input");
    held[i] = &*input;
    auto positions = co::InputValues(*held[i], shape, {});
    if (!positions) return Error(positions.error());
    std::set<std::int64_t> unique;
    for (const auto position : *positions)
      if (position >= 0 && !unique.insert(position).second)
        return Error("private HCA mask repeats a visible position");
  }
  const auto query_positions =
      std::ranges::find(step.inputs, g.root(masks[0]->src[0]), &co::StepInput::tensor);
  const auto sink = std::ranges::find(step.weights, n.source(4), &co::StepWeight::tensor);
  if (states[0]->ref.layer != states[1]->ref.layer || states[0]->tensor == states[1]->tensor ||
      masks[0]->src[0] != masks[1]->src[0] || query_positions == step.inputs.end() ||
      query_positions->kind != co::InputKind::kPositions ||
      query_positions->segment != ir::kJoinedSegment || query_positions->sections != 1 ||
      sink == step.weights.end() || sink->ref.stage != 0 ||
      sink->ref.layer != states[0]->ref.layer || sink->ref.expert_array ||
      sink->parts.size() != 1 ||
      sink->parts.front() != std::pair{std::format("blk.{}.attn_sinks.weight", sink->ref.layer),
                                       format::PartRole::kData})
    return Error("private HCA banks, query positions or sinks belong to another layer/owner");
  const auto* query_view = Made(g, masks[0]->src[0], Op::kView);
  const auto& query_desc = g.tensor(masks[0]->src[0]);
  if (!query_view || query_view->src[0] != query_positions->tensor || query_desc.view_offset != 0 ||
      query_desc.format.encoding != Encoding::kI32 ||
      query_desc.shape().ne != std::array<std::uint64_t, 4>{seg.rows, 1, 1, 1} ||
      !ir::IsContiguous(query_desc))
    return Error("private HCA query positions are not the exact sole-owner rows");
  for (const auto& writer : g.nodes())
    if (writer.key == ir::Primitive(Op::kCacheStore))
      for (const auto t : {held[0]->tensor, held[1]->tensor, query_positions->tensor})
        if (g.root(writer.src[0]) == g.root(t))
          return Error("private HCA positions have an intervening writer");
  const auto& q = g.tensor(n.src[0]);
  const auto& k = g.tensor(n.src[1]);
  const auto& v = g.tensor(n.src[2]);
  const auto& sink_desc = g.tensor(n.src[4]);
  const auto* sink_binding = std::get_if<ir::WeightRef>(&sink_desc.binding);
  if (!sink_binding || *sink_binding != sink->ref)
    return Error("private HCA sink descriptor disagrees with its layer binding");
  const auto params = n.Params<ir::AttnParams>();
  const auto bound = std::min<std::uint64_t>(128, reads[0]) + reads[1];
  if (bound > INT32_MAX || params.n_kv_max != 0 || params.sparse_any || !params.sinks ||
      params.scale != 1.0F / std::sqrt(512.0F) || params.accumulate != ir::Accumulate::kF32 ||
      params.softcap != 0 || params.max_bias != 0 || q.format.encoding != Encoding::kF32 ||
      q.shape().ne != std::array<std::uint64_t, 4>{512, seg.rows, 64, 1} ||
      k.shape().ne != std::array<std::uint64_t, 4>{512, reads[0] + reads[1], 1, 1} ||
      k.format.encoding != Encoding::kF16 || v.format != k.format || v.nb != k.nb ||
      sink_desc.format.encoding != Encoding::kF32 ||
      sink_desc.shape().ne != std::array<std::uint64_t, 4>{64, 1, 1, 1})
    return Error("private HCA attention differs from its fixed numerical recipe");
  return static_cast<std::uint32_t>(bound);
}
}  // namespace

std::expected<co::StepGraph, std::string> RoundedRecipe(co::StepGraph step,
                                                        const pl::StepShape& shape,
                                                        const model::StateLayoutDerived& layout) {
  if (shape.stage != 0 || shape.segments.size() != 1 || shape.segments[0].slot != 0 ||
      shape.phase == pl::Phase::kVerify || shape.phase == pl::Phase::kCommit ||
      shape.row_invariant || !step.saves.empty() || !step.owner_products.products().empty())
    return Error("private rounded recipe only accepts ordinary sole-owner target steps");
  const auto& seg = shape.segments[0];
  const auto& g = step.graph;
  std::vector<Selection> selections;
  std::size_t original_scores = 0;
  for (const auto& n : g.nodes()) {
    if (n.key.op == Op::kIndexScore) ++original_scores;
    if (n.key.op != Op::kConcat || n.Params<ir::ConcatParams>().dim != 0) continue;
    const auto* added = Made(g, n.src[1], Op::kAdd);
    if (!added || !Made(g, added->src[0], Op::kView)) continue;
    const auto* store = Made(g, g.node(g.producer(added->src[0])).src[0], Op::kCacheStore);
    if (!store) continue;
    auto m = Match(g, n, step, layout);
    if (!m) return Error(m.error());
    selections.push_back(std::move(*m));
  }
  if (selections.empty() || selections.size() != original_scores)
    return Error("not every original indexer has one complete closed selected mask");
  ir::Builder b;
  std::vector<TensorId> map(g.tensor_count());
  pl::RewriteContext rewrite(g, b, map);
  co::StepContext context(b,
                          {{.slot = 0,
                            .first_row = 0,
                            .rows = seg.rows,
                            .past = seg.past,
                            .read_cells = seg.read_cells,
                            .outputs = seg.outputs}},
                          false);
  std::vector<TensorId> counts;
  std::map<std::uint32_t, std::size_t> scores, tops, concats;
  std::set<std::uint32_t> erase;
  for (std::size_t i = 0; i < selections.size(); ++i) {
    const auto& m = selections[i];
    auto count = context.VisibleCells(0, m.cells, m.read, layout.context / 4);
    if (!count) return Error(count.error());
    counts.push_back(*count);
    scores[m.score.v] = i;
    tops[m.top.v] = i;
    concats[m.concat.v] = i;
    for (const auto n : m.erased)
      if (!erase.insert(n).second) return Error("overlapping private mask recipes");
  }
  for (std::uint32_t i = 0; i < g.nodes().size(); ++i) {
    const auto& n = g.node(NodeId{i});
    b.SetComponent(n.component);
    if (n.batch == ir::Batch::kJoined)
      b.SetJoined();
    else
      b.SetOwner(g.tensor(n.out[0]).segment);
    if (scores.contains(i)) {
      const auto j = scores.at(i);
      const auto& m = selections[j];
      rewrite.Set(n.out[0], b.RoundedIndexScore(rewrite.Map(m.query), rewrite.Map(m.key),
                                                rewrite.Map(m.weights), counts[j]));
    } else if (tops.contains(i)) {
      const auto j = tops.at(i);
      const auto& m = selections[j];
      rewrite.Set(n.out[0], b.VisibleTopK(rewrite.Map(g.node(m.score).out[0]), counts[j], m.k));
    } else if (concats.contains(i)) {
      const auto& m = selections[concats.at(i)];
      rewrite.Set(n.out[0], b.SelectedMask(rewrite.Map(m.raw), rewrite.Map(g.node(m.top).out[0]),
                                           {.raw_cells = m.raw_cells, .cells = m.read}));
    } else if (n.key == ir::Primitive(Op::kAttention) && n.source(3).valid() &&
               concats.contains(g.producer(n.src[3]).v)) {
      // Only the already authenticated CSA selected-mask boundary. The
      // legacy sparse-any recipe changes the reduction order even when
      // the complete dense mask has identical values (DS23 layer-2 proof).
      // HCA has its independently authenticated ordinary-sparse recipe below.
      const auto& m = selections[concats.at(g.producer(n.src[3]).v)];
      const auto* view = Made(g, m.raw, Op::kView);
      const auto* mask = view ? Made(g, view->src[0], Op::kMaskBuild) : nullptr;
      if (!mask) return Error("private CSA attention lacks its raw window mask");
      const auto mp = mask->Params<ir::MaskParams>();
      const auto& raw = g.tensor(m.raw);
      const auto& original = g.tensor(mask->out[0]);
      const auto held = std::ranges::find(step.inputs, mask->src[1], &co::StepInput::tensor);
      if (mp.kind != ir::MaskKind::kWindow || mp.window == 0 || raw.view_offset != 0 ||
          raw.nb[0] != original.nb[0] || raw.nb[1] != original.nb[1] ||
          raw.ne(0) != original.ne(0) || raw.ne(1) != seg.rows || original.ne(1) < seg.rows ||
          held == step.inputs.end() || held->kind != co::InputKind::kCellPositions ||
          held->segment != 0 || !held->cells.ring || held->cells.block != 0 ||
          held->read != m.raw_cells)
        return Error("private CSA attention lacks its owned ring positions");
      auto positions = co::InputValues(*held, shape, {});
      if (!positions) return Error(positions.error());
      std::set<std::int64_t> visible;
      for (const auto position : *positions)
        if (position >= 0 && !visible.insert(position).second)
          return Error("private CSA window repeats a visible cell position");
      for (const auto& writer : g.nodes())
        if (writer.key == ir::Primitive(Op::kCacheStore) &&
            g.root(writer.src[0]) == g.root(held->tensor))
          return Error("private CSA window positions have an intervening writer");
      auto params = n.Params<ir::AttnParams>();
      const auto bound = std::min<std::uint64_t>(mp.window, m.raw_cells) + m.k;
      const auto& q = g.tensor(n.src[0]);
      const auto& k = g.tensor(n.src[1]);
      const auto& v = g.tensor(n.src[2]);
      if (bound > UINT32_MAX || params.n_kv_max != bound || !params.sinks ||
          params.accumulate != ir::Accumulate::kF32 || params.softcap != 0 ||
          params.max_bias != 0 || q.ne(0) != 512 || q.ne(1) != seg.rows || q.ne(2) != 64 ||
          q.ne(3) != 1 ||
          k.shape().ne !=
              std::array<std::uint64_t, 4>{512, std::uint64_t{m.raw_cells} + m.read, 1, 1} ||
          v.shape() != k.shape() || k.format.encoding != Encoding::kF16 ||
          v.format.encoding != Encoding::kF16)
        return Error("private CSA attention differs from its proved visible-cell bound");
      params.sparse_any = true;
      // Existing state-bank capability refuses sparse-any. Its ordinary
      // Concat fallback preserves both exact K/V values and the selected
      // wide sparse primitive until bank recipe preservation is qualified.
      rewrite.Set(n.out[0],
                  b.Attention(rewrite.Map(n.src[0]), rewrite.Map(n.src[1]), rewrite.Map(n.src[2]),
                              rewrite.Map(n.src[3]), rewrite.Map(n.source(4)), params));
    } else if (n.key == ir::Primitive(Op::kAttention) && HcaMask(g, n)) {
      auto bound = HcaBound(step, n, shape, layout);
      if (!bound) return Error(bound.error());
      auto params = n.Params<ir::AttnParams>();
      params.n_kv_max = *bound;
      params.sparse_any = true;
      // The production state-bank path deliberately refuses sparse-any:
      // retain the explicit Concat and ordinary MMA recipe for this screen.
      rewrite.Set(n.out[0],
                  b.Attention(rewrite.Map(n.src[0]), rewrite.Map(n.src[1]), rewrite.Map(n.src[2]),
                              rewrite.Map(n.src[3]), rewrite.Map(n.source(4)), params));
    } else if (n.key == ir::Primitive(Op::kAttention) && Made(g, n.source(3), Op::kMaskBuild)) {
      // Plain window, independent of the authenticated CSA recipe above.
      // Unmatched compressed attention retains its existing parameters.
      const auto& mask = g.node(g.producer(n.src[3]));
      const auto mp = mask.Params<ir::MaskParams>();
      const auto held = std::ranges::find(step.inputs, mask.src[1], &co::StepInput::tensor);
      if (mp.kind != ir::MaskKind::kWindow || mp.window == 0 || held == step.inputs.end() ||
          held->kind != co::InputKind::kCellPositions || held->segment != 0 || !held->cells.ring ||
          held->cells.block != 0 || held->read != g.tensor(n.src[1]).ne(1))
        return Error("private window attention lacks its owned ring positions");
      auto positions = co::InputValues(*held, shape, {});
      if (!positions) return Error(positions.error());
      // A window of W integer positions contains at most W distinct cells.
      // Authenticate the actual staged positions, including any ring wrap.
      std::set<std::int64_t> visible;
      for (const auto position : *positions)
        if (position >= 0 && !visible.insert(position).second)
          return Error("private window attention repeats a visible cell position");
      for (const auto& writer : g.nodes())
        if (writer.key == ir::Primitive(Op::kCacheStore) &&
            g.root(writer.src[0]) == g.root(held->tensor))
          return Error("private window positions have an intervening writer");
      auto params = n.Params<ir::AttnParams>();
      params.n_kv_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(mp.window, held->read));
      params.sparse_any = true;
      rewrite.Set(n.out[0],
                  b.Attention(rewrite.Map(n.src[0]), rewrite.Map(n.src[1]), rewrite.Map(n.src[2]),
                              rewrite.Map(n.src[3]), rewrite.Map(n.source(4)), params));
    } else if (!erase.contains(i)) {
      rewrite.Copy(n);
    }
  }
  for (const auto t : g.outputs()) b.Output(rewrite.Map(t));
  for (const auto t : g.keep()) b.Keep(rewrite.Map(t));
  auto made = std::move(b).Finish();
  if (!made) return Error(made.error().detail);
  for (auto& x : step.inputs) x.tensor = rewrite.Map(x.tensor);
  for (auto& x : step.weights) x.tensor = rewrite.Map(x.tensor);
  for (auto& x : step.state) x.tensor = rewrite.Map(x.tensor);
  step.logits = rewrite.Map(step.logits);
  step.greedy = rewrite.Map(step.greedy);
  step.features = rewrite.Map(step.features);
  for (const auto& x : context.TakeInputs()) step.inputs.push_back(x);
  step.graph = std::move(*made);
  if (auto checked = co::CheckStateWrites(step, 0); !checked) return Error(checked.error());
  return step;
}

std::expected<void, std::string> SelectHcaRecipe(pl::StepPlan& plan, const co::StepGraph& step,
                                                 const pl::StepShape& shape,
                                                 const model::StateLayoutDerived& layout,
                                                 const execution::Registry& registry,
                                                 const pl::DeviceFacts& device) {
  if (shape.stage != 0 || shape.segments.size() != 1 || shape.segments.front().slot != 0 ||
      shape.row_invariant || shape.phase == pl::Phase::kVerify || shape.phase == pl::Phase::kCommit)
    return Error("private HCA selection requires its sole-owner ordinary step");
  const auto ordinary = registry.Find("ggml.flash_attn_ext.mma");
  if (!ordinary || !registry.at(*ordinary).capability)
    return Error("private HCA ordinary MMA implementation is unavailable");
  const pl::Policy policy;
  const pl::SelectionView selection(registry, device, policy, false, true);
  std::size_t count = 0;
  for (const auto& n : step.graph.nodes()) {
    if (n.key != ir::Primitive(Op::kAttention) || !HcaMask(step.graph, n)) continue;
    auto original = n;
    auto params = n.Params<ir::AttnParams>();
    if (!params.sparse_any || params.n_kv_max == 0)
      return Error("private HCA selected recipe lost its sparse parameters");
    params.sparse_any = false;
    params.n_kv_max = 0;
    original.SetParams(params);
    const auto bound = HcaBound(step, original, shape, layout);
    if (!bound || n.Params<ir::AttnParams>().n_kv_max != *bound)
      return Error("private HCA selected recipe lost its authenticated boundary");
    if (n.out[0].v >= plan.tensors.size() || !plan.tensors[n.out[0].v].valid())
      return Error("private HCA planned output was erased");
    const auto mapped = plan.tensors[n.out[0].v];
    const auto id = plan.graph.producer(mapped);
    const auto& planned = plan.graph.node(id);
    if (planned.key != n.key || planned.params != n.params || planned.nsrc != n.nsrc ||
        planned.nout != 1 || planned.out[0] != mapped)
      return Error("private HCA plan changed the attention operation");
    for (std::size_t i = 0; i < n.nsrc; ++i)
      if (n.src[i].v >= plan.tensors.size() || planned.src[i] != plan.tensors[n.src[i].v])
        return Error("private HCA plan changed an authenticated operand");
    auto selected = std::ranges::find_if(
        plan.steps, [&](const auto& x) { return x.nodes.size() == 1 && x.nodes.front() == id; });
    if (selected == plan.steps.end() || !selected->batch.empty() || !selected->singles.empty() ||
        (registry.at(selected->implementation).name != "llmp.flash_attn_ext.mma_wide" &&
         selected->implementation != *ordinary) ||
        selected->group != registry.at(*ordinary).capability->group ||
        !std::ranges::contains(selection.Candidates(plan.graph, planned), *ordinary))
      return Error("private HCA plan has no eligible exact ordinary MMA step");
    // Legacy excludes its HCA mask mode from wide-query dispatch. This
    // private per-operation choice preserves CSA/window choices, placement,
    // lane and operands, and the normal strict lowering comparison still runs.
    selected->implementation = *ordinary;
    ++count;
  }
  if (count == 0) return Error("private HCA plan has no authenticated attention");
  return {};
}

std::expected<std::unique_ptr<Source>, std::string> Source::Create(
    engine::ComponentSourceOptions options, std::vector<pl::StepShape> script) {
  if (!options.spec || options.spec->stages.size() != 1 || options.stage != 0 || script.empty() ||
      !options.policy || !options.bindings.state || options.slots != 1)
    return Error("private screen needs one explicit target, policy and script");
  for (const auto& s : script)
    if (s.stage != 0 || s.segments.size() != 1 || s.segments[0].slot != 0 || s.row_invariant ||
        s.segments[0].greedy || s.segments[0].rows == 0 || s.segments[0].rows > 512 ||
        s.phase == pl::Phase::kVerify || s.phase == pl::Phase::kCommit)
      return Error("private screen shape outside its ordinary one-owner script");
  auto ordinary = engine::ComponentStepSource::Create(options);
  if (!ordinary) return Error(ordinary.error());
  auto out = std::unique_ptr<Source>(new Source);
  out->options_ = std::move(options);
  out->ordinary_ = std::move(*ordinary);
  out->script_ = std::move(script);
  return out;
}
pl::Policy ServedPolicy(const pl::Policy& policy, std::uint32_t rows) {
  auto result = policy;
  // dsv4_graph.cc Builder::Fused uses the whole solo step's rows, not the
  // narrowed final layer's rows; VecQRows() is kVecQTokens == 8. Larger
  // steps keep HcPre's ordinary RMS/product/mix corridor. The generic
  // fused capability is unchanged: this reproduces the served selection.
  if (rows > 8) std::erase(result.enabled, "dsv4_hc_pre");
  return result;
}
pl::PlanKey Source::KeyBase() const {
  auto key = ordinary_->KeyBase();
  base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(key.policy)));
  hash.Update("private-dsv4-rounded-visible-full-mask-window-hca-ordinary-sparse-screen-v3");
  key.policy = hash.Finish();
  return key;
}
std::vector<std::uint64_t> Source::ShapeParameters(const pl::StepShape& shape) const {
  return ordinary_->ShapeParameters(shape);
}

planner::StateSpans Source::StateSpans(const planner::StepShape& shape,
                                       const kernels::Addresses& addresses) const {
  return [this, &shape, &addresses](std::span<const planner::StateCellSlice> slices)
             -> std::optional<planner::StateSpanFact> {
    if (slices.empty()) return std::nullopt;
    const auto owner = slices.front().ref.segment;
    if (owner >= shape.segments.size()) return std::nullopt;
    std::vector<model::StateSlice> requested;
    for (const auto& slice : slices) {
      if (slice.ref.stage != options_.stage || slice.ref.segment != owner) return std::nullopt;
      requested.push_back({.layer = slice.ref.layer,
                           .tensor = slice.ref.tensor,
                           .first = slice.first,
                           .cells = slice.cells});
    }
    const auto& layout = *options_.bindings.state;
    auto bank = model::ContiguousStateSpan(layout, requested);
    if (!bank) return std::nullopt;
    // Addresses are of whole items. Prove one stage/slot base, not merely
    // adjacency of two unrelated allocations. Missing measurement addresses
    // are replaced together from this same immutable region by the lowering.
    std::optional<std::uint64_t> base;
    bool missing = false;
    for (const auto& slice : slices) {
      const auto item = std::ranges::find_if(layout.items, [&](const auto& item) {
        return item.layer == slice.ref.layer && item.tensor == slice.ref.tensor;
      });
      if (item == layout.items.end()) return std::nullopt;
      const auto address = addresses.state ? addresses.state(slice.ref) : 0;
      if (address == 0) {
        missing = true;
        continue;
      }
      if (address < item->offset) return std::nullopt;
      const auto candidate = address - item->offset;
      std::uint64_t end = 0;
      if (__builtin_add_overflow(candidate, layout.bytes, &end) || (base && *base != candidate))
        return std::nullopt;
      base = candidate;
    }
    if (missing && (!addresses.measuring() || base)) return std::nullopt;
    return planner::StateSpanFact{.bank = *bank, .region_bytes = layout.bytes};
  };
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Source::Build(
    const planner::StepShape& shape, const kernels::Addresses& addresses) {
  if (std::ranges::find_if(script_, [&](const auto& admitted) {
        return admitted == shape && admitted.segments.size() == shape.segments.size() &&
               std::equal(admitted.segments.begin(), admitted.segments.end(),
                          shape.segments.begin(),
                          [](const auto& a, const auto& b) { return a.past == b.past; });
      }) == script_.end())
    return Error("private source rejected a shape outside its owned script");
  if (shape.stage != options_.stage) {
    return std::unexpected(
        std::format("a step of stage {} from stage {}'s source", shape.stage, options_.stage));
  }
  // The narrowed final layer is the policy's (planner/pattern.h
  // narrow_final).
  auto step =
      shape.phase == planner::Phase::kCommit
          ? components::BuildCommit(*options_.spec, options_.stage, shape, options_.bindings)
          : components::BuildStep(*options_.spec, options_.stage, shape, options_.bindings,
                                  {.narrow = !planner::Disabled(*options_.policy, "narrow_final")});
  if (!step) return std::unexpected(std::move(step.error()));
  step = RoundedRecipe(std::move(*step), shape, *options_.bindings.state);
  if (!step) return Error(step.error());
  // Keep the existing placement set. Cell positions are synthesized by
  // masks; commit rows retain their legacy reserved place but are bound to
  // external saved rows, so receive no arena alignment guarantee.
  std::vector<ir::TensorId> staged, aligned_staged;
  for (const components::StepInput& input : step->inputs) {
    if (input.kind != components::InputKind::kCellPositions) staged.push_back(input.tensor);
    if (input.kind != components::InputKind::kCellPositions &&
        input.kind != components::InputKind::kCommitRows)
      aligned_staged.push_back(input.tensor);
  }
  const auto power_of_two = [](std::uint64_t n) { return n != 0 && (n & (n - 1)) == 0; };
  if (addresses.activation_alignment != 0 && !power_of_two(addresses.activation_alignment))
    return std::unexpected("the activation reservation alignment is not a power of two");
  if (!addresses.measuring() && addresses.activation_alignment != 0 &&
      addresses.activations % addresses.activation_alignment != 0)
    return std::unexpected("the activation base breaks its reservation alignment promise");
  std::vector<std::pair<ir::TensorDesc, std::uint64_t>> state_facts;
  if (addresses.state_alignment) {
    for (const auto& node : step->graph.nodes()) {
      if (node.key.op != ir::Op::kState) continue;
      const auto& desc = step->graph.tensor(node.out[0]);
      const auto* ref = std::get_if<ir::StateRef>(&desc.binding);
      if (ref == nullptr) continue;
      const auto alignment = addresses.state_alignment(*ref);
      if (alignment == 0) continue;
      if (!power_of_two(alignment))
        return std::unexpected("a state reservation alignment is not a power of two");
      const auto address = addresses.state ? addresses.state(*ref) : 0;
      if (address != 0 && address % alignment != 0)
        return std::unexpected("a state base breaks its reservation alignment promise");
      state_facts.emplace_back(desc, alignment);
    }
  }
  planner::RootAlignment root_alignment;
  if (!state_facts.empty()) {
    root_alignment = [facts = std::move(state_facts)](const ir::Graph& graph,
                                                      ir::TensorId root) -> std::uint64_t {
      if (!root.valid() || root.v >= graph.tensor_count()) return 0;
      const auto& desc = graph.tensor(root);
      if (desc.is_view() || graph.node(graph.producer(root)).key.op != ir::Op::kState) return 0;
      for (const auto& [saved, alignment] : facts) {
        if (desc.binding == saved.binding && desc.format == saved.format && desc.nb == saved.nb &&
            desc.role == saved.role && desc.segment == saved.segment)
          return alignment;
      }
      return 0;
    };
  }
  const auto state_spans = StateSpans(shape, addresses);
  const auto policy = ServedPolicy(*options_.policy, shape.segments.front().rows);
  auto plan = planner::Plan(step->graph.Clone(), *options_.backend.registry,
                            options_.backend.device, policy,
                            {.row_invariant = shape.row_invariant,
                             .capturable = true,
                             .staged = std::move(staged),
                             .on_stream = options_.backend.on_stream,
                             .written = options_.backend.written,
                             .within = options_.backend.within,
                             .shape = &shape,
                             .state_spans = state_spans,
                             .root_alignment = std::move(root_alignment),
                             .activation_alignment = addresses.activation_alignment,
                             .alignment_staged = std::move(aligned_staged)});
  if (!plan) {
    return std::unexpected(std::format("planning a step ({}): {}",
                                       planner::PlanErrorName(plan.error().error),
                                       plan.error().detail));
  }
  if (auto recipe = SelectHcaRecipe(*plan, *step, shape, *options_.bindings.state,
                                    *options_.backend.registry, options_.backend.device);
      !recipe)
    return Error(recipe.error());
  std::size_t needed_masks = 0, fused_masks = 0;
  for (const auto& n : step->graph.nodes()) needed_masks += n.key.op == ir::Op::kSelectedMask;
  for (const auto& n : plan->graph.nodes())
    fused_masks += n.key == ir::Fused(ir::FusedKind::kDsv4LidMask);
  if (needed_masks == 0 || needed_masks != fused_masks)
    return Error("private model screen did not select every native full-mask recipe");
  if (!addresses.measuring()) selected_masks_ += fused_masks;
  const kernels::LowerInput input{
      *plan, *step, shape, addresses, options_.bindings.state->context, state_spans};
  return options_.backend.lower(input);
}

}  // namespace llmp::test_support::dsv4_screen
