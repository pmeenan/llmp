// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The component catalog (components/component.h, step.h): the catalog
// and its parameters; ValidateSpec on Gemma 2's and Gemma 3's specs from
// their tensor tables and its refusals; each component's fragment, run by
// the reference executor over several steps (prefill, decode, a ring that
// wraps; several owners), against its own scalar reference; BuildStep on
// the Gemma specs; the stage layout against today's Gemma state layouts;
// and the step's input values against today's host masks.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "component_catalog_cases.h"
#include "components/component.h"
#include "components/moe.h"
#include "components/state_encoding.h"
#include "components/step.h"
#include "families/gemma.h"
#include "format/encoding.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/reference/exec.h"
#include "model/host_mask.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/spec.h"
#include "spec_fixtures.h"
#include "state_layouts.h"

namespace llmp::components {
namespace {

namespace ref = llmp::kernels::reference;
namespace fam = llmp::test_support::families;
using format::Encoding;
using spec::ComponentRef;
using spec::Param;
using spec::ParamValue;

TEST(InputValueSinkTest, TypedCapacityAndNarrowingRefuseWithoutWideningOrLosingPadding) {
  InputValueSink count;
  count.push_back(7);
  count.push_back(-3);
  ASSERT_TRUE(count.Finish());
  EXPECT_EQ(count.size(), 2U);
  std::array<std::int32_t, 3> stored{99, 99, 99};
  InputValueSink padded(std::as_writable_bytes(std::span(stored)), sizeof(std::int32_t), true);
  padded.push_back(7);
  padded.push_back(-3);
  ASSERT_TRUE(padded.Finish());
  EXPECT_EQ(stored, (std::array<std::int32_t, 3>{7, -3, 0}));
  InputValueSink exact(std::as_writable_bytes(std::span(stored)), sizeof(std::int32_t));
  exact.push_back(7);
  EXPECT_FALSE(exact.Finish());
  InputValueSink narrow(std::as_writable_bytes(std::span(stored)), sizeof(std::int32_t), true);
  narrow.push_back(std::int64_t{INT32_MAX} + 1);
  EXPECT_FALSE(narrow.Finish());
  EXPECT_EQ(stored, (std::array<std::int32_t, 3>{7, -3, 0}));
}

TEST(InputValueSinkTest, CanonicalCompressorCountAndFillPreserveIndependentOverlapOrder) {
  const std::array<std::int64_t, 16> expected{16, 16, 16, 16, 8,  9,  10, 11,
                                              8,  9,  10, 11, 12, 13, 14, 15};
  const planner::StepShape shape{.segments = {{.rows = 8, .past = 0}}};
  const StepInput pool_in{.kind = InputKind::kHostComputed,
                          .segment = 0,
                          .cells = {.capacity = 8, .ring = true, .recent = true},
                          .host = {.def = &CompressorPoolReadsInput(), .ratio = 4, .overlap = true},
                          .ratio = 4,
                          .overlap = true,
                          .ring = 8};
  const StepInput cell_in{
      .kind = InputKind::kHostComputed,
      .segment = 0,
      .cells = {.capacity = 16, .block = 4, .positions = true},
      .host = {.def = &CompressorBlockCellsInput(), .ratio = 4, .overlap = true},
      .ratio = 4,
      .overlap = true,
      .ring = 16};
  const StepInput start_in{
      .kind = InputKind::kHostComputed,
      .segment = 0,
      .host = {.def = &CompressorBlockStartsInput(), .ratio = 4, .overlap = true},
      .ratio = 4,
      .overlap = true};
  InputValueSink count;
  ASSERT_TRUE(CompressorPoolReadsInput().emit(pool_in, shape, count));
  ASSERT_EQ(count.size(), expected.size());
  std::array<std::int64_t, 16> reads{};
  std::array<std::int64_t, 2> cells{}, starts{};
  InputValueSink read_sink(std::as_writable_bytes(std::span(reads)), sizeof(std::int64_t));
  InputValueSink cell_sink(std::as_writable_bytes(std::span(cells)), sizeof(std::int64_t));
  InputValueSink start_sink(std::as_writable_bytes(std::span(starts)), sizeof(std::int64_t));
  ASSERT_TRUE(CompressorPoolReadsInput().emit(pool_in, shape, read_sink));
  ASSERT_TRUE(CompressorBlockCellsInput().emit(cell_in, shape, cell_sink));
  ASSERT_TRUE(CompressorBlockStartsInput().emit(start_in, shape, start_sink));
  EXPECT_EQ(reads, expected);
  EXPECT_EQ(cells, (std::array<std::int64_t, 2>{0, 1}));
  EXPECT_EQ(starts, (std::array<std::int64_t, 2>{0, 4}));
  std::array<std::int64_t, 15> short_storage{};
  InputValueSink short_sink(std::as_writable_bytes(std::span(short_storage)), sizeof(std::int64_t));
  EXPECT_FALSE(CompressorPoolReadsInput().emit(pool_in, shape, short_sink));
  StepInput small_cache = cell_in;
  small_cache.cells.capacity = 1;
  InputValueSink unused;
  EXPECT_FALSE(CompressorBlockCellsInput().emit(small_cache, shape, unused));
}

constexpr std::uint64_t kExtent = 2ULL << 20U;

ComponentRef C(std::string kind, std::vector<Param> params) {
  std::ranges::sort(params, {}, &Param::name);
  return {.kind = std::move(kind), .version = 1, .params = std::move(params), .weights = {}};
}
Param P(std::string name, ParamValue v) { return {.name = std::move(name), .value = std::move(v)}; }
ParamValue I(std::int64_t v) { return ParamValue(v); }
ParamValue R(double v) { return ParamValue(v); }
ParamValue N(std::string v) { return ParamValue(std::move(v)); }

spec::ModelSpec Gemma4Spec(std::uint32_t size);
spec::ModelSpec Gemma2Spec() {
  auto values = fam::Gemma2Values(fam::Gemma2_2B());
  auto s = spec::Synthesize(spec::mappings::Gemma2Mapping(), test_support::Gemma2Source(), *values);
  EXPECT_TRUE(s.has_value()) << s.error();
  return *s;
}
spec::ModelSpec Gemma3Spec() {
  auto values = fam::Gemma3Values(fam::Gemma3_4BQat());
  auto s = spec::Synthesize(spec::mappings::Gemma3Mapping(), test_support::Gemma3Source(), *values);
  EXPECT_TRUE(s.has_value()) << s.error();
  return *s;
}

// ---------------------------------------------------------------- catalog

TEST(ComponentsTest, CatalogIsSortedUniqueAndHasReferences) {
  std::vector<std::pair<std::string_view, std::uint16_t>> kinds;
  for (const ComponentDef* def : Catalog()) {
    kinds.emplace_back(def->kind, def->version);
    if (def->spatial != nullptr) {
      EXPECT_EQ(def->emit, nullptr) << def->kind;
      EXPECT_EQ(def->reference, nullptr) << def->kind;
    } else {
      EXPECT_NE(def->emit, nullptr) << def->kind;
      EXPECT_NE(def->reference, nullptr) << def->kind;
    }
    EXPECT_NE(def->shapes, nullptr) << def->kind;
    EXPECT_EQ(Find(def->kind, def->version), def);
    EXPECT_EQ(def->state != nullptr, def->max_state_tensors != 0) << def->kind;
  }
  EXPECT_TRUE(std::ranges::is_sorted(kinds));
  EXPECT_EQ(std::ranges::adjacent_find(kinds), kinds.end());
  EXPECT_EQ(Find("attention.dense", 2), nullptr);
  EXPECT_EQ(Find("attention.sparse", 1), nullptr);
  EXPECT_EQ(RoleIndex(*Find("ffn.glu", 1), "down"), 3);
  EXPECT_FALSE(RoleIndex(*Find("ffn.glu", 1), "bias").has_value());
}

TEST(ComponentsTest, ParamsAreTypedRangedAndDefaulted) {
  const ComponentDef& def = *Find("attention.dense", 1);
  ComponentRef r = C("attention.dense",
                     {P("placement", N("sandwich")), P("eps", R(1e-6)), P("heads", I(8)),
                      P("kv_heads", I(4)), P("key_dim", I(256)), P("value_dim", I(256)),
                      P("scale", R(0.0625)), P("window", I(0)), P("rotation", N("neox")),
                      P("rope_dims", I(256)), P("rope_base", I(10000)), P("rope_scale", R(1.0))});
  auto p = Params::Of(def.params, r);
  ASSERT_TRUE(p.has_value()) << p.error();
  EXPECT_EQ(p->Int("heads"), 8);
  EXPECT_EQ(p->Real("rope_base"), 10000.0);  // an integer is a real too
  EXPECT_EQ(p->Real("softcap"), 0.0);        // absent: its default
  EXPECT_FALSE(p->Flag("qk_norm"));
  EXPECT_FALSE(p->Given("softcap"));
  EXPECT_TRUE(p->Given("heads"));
  const auto with = [&](Param changed) {
    ComponentRef c = r;
    std::erase_if(c.params, [&](const Param& q) { return q.name == changed.name; });
    c.params.push_back(std::move(changed));
    std::ranges::sort(c.params, {}, &Param::name);
    return Params::Of(def.params, c);
  };
  EXPECT_FALSE(with(P("heads", I(0))));          // out of range
  EXPECT_FALSE(with(P("heads", R(8.0))));        // a real, not an integer
  EXPECT_FALSE(with(P("rotation", N("yarn"))));  // not a choice
  EXPECT_FALSE(with(P("qk_norm", I(1))));        // not a flag
  EXPECT_FALSE(with(P("extra", I(1))));          // unknown
  ComponentRef missing = r;
  std::erase_if(missing.params, [](const Param& q) { return q.name == "eps"; });
  EXPECT_FALSE(Params::Of(def.params, missing));  // required
  // A valid parameter set may still be an invalid instance.
  const StageFacts facts{.width = 64, .context = 64};
  ComponentRef bad = C("attention.dense", r.params);
  std::erase_if(bad.params, [](const Param& q) { return q.name == "kv_heads"; });
  bad.params.push_back(P("kv_heads", I(3)));
  std::ranges::sort(bad.params, {}, &Param::name);
  EXPECT_FALSE(MakeInstance(bad, facts, Site::kLayer, 0, 1));  // 8 heads over 3
}

// ------------------------------------------------------------- validation

TEST(ComponentsTest, GemmaSpecsValidateAgainstTheirTensorTables) {
  const auto g2 = test_support::Gemma2Source();
  const auto g3 = test_support::Gemma3Source();
  const auto v2 = ValidateSpec(Gemma2Spec(), g2.tensors);
  EXPECT_TRUE(v2.has_value()) << v2.error();
  const auto v3 = ValidateSpec(Gemma3Spec(), g3.tensors);
  EXPECT_TRUE(v3.has_value()) << v3.error();
  const auto g2_masks = StageIndexedMasks(Gemma2Spec(), 0);
  const auto g3_masks = StageIndexedMasks(Gemma3Spec(), 0);
  ASSERT_TRUE(g2_masks) << g2_masks.error();
  ASSERT_TRUE(g3_masks) << g3_masks.error();
  EXPECT_FALSE(*g2_masks);
  EXPECT_FALSE(*g3_masks);
  // Each against the other's tensors: shapes and roles differ.
  EXPECT_FALSE(ValidateSpec(Gemma2Spec(), g3.tensors));
}

TEST(ComponentsTest, ValidateSpecRefusesWhatTheCatalogCannotRun) {
  const auto tensors = test_support::Gemma2Source().tensors;
  const spec::ModelSpec good = Gemma2Spec();
  const auto refused = [&](const spec::ModelSpec& s, std::span<const BoundTensor> t,
                           std::string_view why) {
    const auto v = ValidateSpec(s, t);
    EXPECT_FALSE(v.has_value()) << why;
    if (!v) SCOPED_TRACE(v.error());
  };
  const auto layer0 = [](spec::ModelSpec& s) -> ComponentRef& {
    return s.stages[0].layers[0].components[0];
  };
  {
    spec::ModelSpec s = good;
    layer0(s).kind = "attention.sparse";
    refused(s, tensors, "unknown kind");
  }
  {
    spec::ModelSpec s = good;
    layer0(s).version = 2;
    refused(s, tensors, "unknown version");
  }
  {
    spec::ModelSpec s = good;
    for (Param& p : layer0(s).params) {
      if (p.name == "heads") p.value = I(5000);
    }
    refused(s, tensors, "out-of-range parameter");
  }
  {
    spec::ModelSpec s = good;
    std::erase_if(layer0(s).weights, [](const spec::WeightBinding& w) { return w.role == "k"; });
    refused(s, tensors, "unbound role");
  }
  {
    spec::ModelSpec s = good;
    layer0(s).weights.push_back({.role = "q_norm", .tensor = "blk.0.attn_norm.weight"});
    std::ranges::sort(layer0(s).weights, {}, &spec::WeightBinding::role);
    refused(s, tensors, "a role its parameters do not use");
  }
  {
    // The attention norm bound to a Q8_0 matrix: wrong encoding and shape.
    spec::ModelSpec s = good;
    for (spec::WeightBinding& w : layer0(s).weights) {
      if (w.role == "norm") w.tensor = "blk.0.attn_q.weight";
    }
    refused(s, tensors, "wrongly encoded role");
  }
  {
    // A Q8_0 tensor bound where another Q8_0 matrix of another shape belongs.
    spec::ModelSpec s = good;
    for (spec::WeightBinding& w : layer0(s).weights) {
      if (w.role == "k") w.tensor = "blk.0.attn_q.weight";
    }
    refused(s, tensors, "shape mismatch");
  }
  {
    spec::ModelSpec s = good;
    for (spec::WeightBinding& w : layer0(s).weights) {
      if (w.role == "k") w.tensor = "blk.0.attn_k.missing";
    }
    refused(s, tensors, "a tensor the artifact does not have");
  }
  {
    spec::ModelSpec s = good;
    s.stages[0].taps.push_back({.name = "deep", .layer = 99, .point = spec::TapPoint::kLayerOut});
    refused(s, tensors, "dangling tap");
  }
  {
    spec::ModelSpec s = good;
    s.stages[0].taps.push_back({.name = "hc", .layer = 1, .point = spec::TapPoint::kStreams});
    refused(s, tensors, "streams tap of a single stream");
  }
  {
    spec::ModelSpec s = good;
    s.stages[0].stream.width = kMaxStreamWidth * 2;
    refused(s, tensors, "width past the catalog's bound");
  }
  {
    // A drafter stage reading state its target does not have; and one
    // reading state none of its components borrows.
    spec::ModelSpec s = good;
    spec::Stage drafter = s.stages[0];
    drafter.name = "assistant";
    drafter.role = spec::StageRole::kDrafter;
    drafter.reads = {{.stage = "target", .kind = spec::ReadKind::kState, .name = "kv"}};
    s.stages.push_back(drafter);
    s.stages[0].name = "target";
    spec::Canonicalize(s);
    refused(s, tensors, "a read of state the stage has not");
    s.stages[1].reads[0].name = "k";
    refused(s, tensors, "a read of state nothing of the stage borrows");
    s.stages[1].reads.clear();
    const auto ok = ValidateSpec(s, tensors);
    EXPECT_TRUE(ok.has_value()) << ok.error();
  }
}

// ---------------------------------------------------------- state layout

// Gemma's state in closed form (the M3 layouts, retired with their
// runners): a K and a V tensor per layer, a global one a cell a position,
// a local one a ring of pad(min(context, window + rows), 256) cells, each
// on its own extent.
struct GemmaStateRef {
  struct Tensor {
    std::uint32_t layer = 0, cells = 0;
    std::uint64_t offset = 0, bytes = 0;
  };
  std::vector<Tensor> tensors;
  std::uint64_t bytes = 0;
};
std::uint64_t PadUp(std::uint64_t value, std::uint64_t to) { return (value + to - 1) / to * to; }
template <typename Profile>
GemmaStateRef GemmaState(const Profile& p, std::uint32_t context, std::uint32_t rows) {
  GemmaStateRef ref;
  const auto global_cells = static_cast<std::uint32_t>(PadUp(context, 256));
  const auto local_cells =
      static_cast<std::uint32_t>(PadUp(std::min<std::uint64_t>(context, p.window + rows), 256));
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    for (const bool value : {false, true}) {
      const auto width = (value ? p.value_dim : p.key_dim) * p.kv_heads;
      const auto cells = p.local(il) ? local_cells : global_cells;
      const auto bytes = std::uint64_t{width} * cells * 2;
      ref.tensors.push_back({il, cells, ref.bytes, bytes});
      ref.bytes += PadUp(bytes, kExtent);
    }
  }
  return ref;
}

TEST(ComponentsTest, StageLayoutsAreTheGemmaLayouts) {
  for (const auto& [context, rows] : {std::pair{4096U, 128U}, {8192U, 512U}, {8192U, 16U}}) {
    const auto want = GemmaState(fam::Gemma2_2B(), context, rows);
    const auto got = StageLayoutUniform(Gemma2Spec(), 0, context, rows, kExtent);
    ASSERT_TRUE(got.has_value()) << got.error();
    ASSERT_EQ(got->items.size(), want.tensors.size());
    for (std::size_t i = 0; i < want.tensors.size(); ++i) {
      EXPECT_EQ(got->items[i].layer, want.tensors[i].layer);
      EXPECT_EQ(got->items[i].offset, want.tensors[i].offset);
      EXPECT_EQ(got->items[i].cells, want.tensors[i].cells);
      EXPECT_EQ(got->items[i].bytes, want.tensors[i].bytes);
    }
    EXPECT_EQ(got->bytes, want.bytes);
  }
  for (const auto& [context, rows] : {std::pair{8192U, 512U}, {131072U, 2048U}}) {
    const auto want = GemmaState(fam::Gemma3_4BQat(), context, rows);
    const auto got = StageLayoutUniform(Gemma3Spec(), 0, context, rows, kExtent);
    ASSERT_TRUE(got.has_value()) << got.error();
    ASSERT_EQ(got->items.size(), want.tensors.size());
    for (std::size_t i = 0; i < want.tensors.size(); ++i) {
      EXPECT_EQ(got->items[i].offset, want.tensors[i].offset);
      EXPECT_EQ(got->items[i].cells, want.tensors[i].cells);
    }
    EXPECT_EQ(got->bytes, want.bytes);
  }
  const auto batching = StageBatching(Gemma2Spec(), 0);
  ASSERT_TRUE(batching.has_value());
  EXPECT_EQ(batching->max_segments, kMaxSegments);
  EXPECT_TRUE(batching->join_prefill);
  EXPECT_TRUE(batching->mixed_heads);
  EXPECT_FALSE(batching->equal_prefill);
}

// ----------------------------------------------------- fragments vs references

struct Step {
  planner::StepShape shape;
  std::vector<std::int32_t> tokens{};  // an embedding's
};

planner::Segment Seg(std::uint8_t slot, std::uint32_t rows, std::uint32_t past, std::uint32_t read,
                     std::uint32_t outputs, bool greedy = false) {
  return {.slot = slot,
          .rows = rows,
          .past = past,
          .read_cells = read,
          .outputs = outputs,
          .greedy = greedy};
}

double Gap(std::span<const float> got, std::span<const double> want) {
  EXPECT_EQ(got.size(), want.size());
  double worst = 0.0;
  double scale = 1.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
    scale = std::max(scale, std::fabs(want[i]));
  }
  return worst / scale;
}

// Runs `in`'s fragment (the reference executor) and its reference over
// `steps`, carrying the state, and compares outputs and state within
// `tolerance` of the largest value (`state_tolerance` for the state, when
// given). Weights are random in `formats`, both
// sides reading the same decoded values.
void Compare(const Instance& in, std::span<const format::TensorFormat> formats,
             std::span<const std::uint64_t> cells, std::span<const Step> steps, double tolerance,
             double state_tolerance = 0.0,
             const std::map<std::uint16_t, std::pair<float, float>>& ranges = {},
             bool random_state = false) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  const auto shapes = in.def->shapes(in);
  ASSERT_TRUE(shapes.has_value());
  // Random weights by role, encoded in their formats; both sides read the
  // decoded values.
  RefIO io{.instance = &in};
  io.store_row = &ref::StoreStateRow;
  io.weights.resize(in.def->roles.size());
  io.weight_shapes.resize(in.def->roles.size());
  std::vector<std::vector<std::byte>> encoded(in.def->roles.size());
  for (const RoleShape& r : *shapes) {
    const auto& f = formats[r.role];
    std::vector<float> v(ir::Elements(f.shape));
    const bool norm = f.shape.rank == 1;
    // A role's own value range (`ranges`: a rotation's factors, a scale).
    const auto range = ranges.find(r.role);
    for (float& x : v) {
      x = range != ranges.end() ? range->second.first + (range->second.second * (1.0F + u(rng)))
          : norm                ? 1.0F + (0.25F * u(rng))
                                : 0.3F * u(rng);
    }
    const auto parts = format::PartsOf(f);
    ASSERT_TRUE(parts.has_value());
    encoded[r.role].resize((*parts)[0].bytes);
    // Matrix by matrix (an expert array's experts).
    const std::uint64_t matrices = format::Matrices(f);
    const std::size_t per = v.size() / matrices;
    io.weights[r.role].resize(v.size());
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::EncodeRows(f, m, 0, std::span(v).subspan(m * per, per), encoded[r.role]));
    }
    const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[r.role]};
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::DecodeRows(f, std::span(&part, 1), m, 0, f.shape.ne[1],
                                  std::span(io.weights[r.role]).subspan(m * per, per)));
    }
    io.weight_shapes[r.role] = f.shape;
  }
  const model::StateContract contract =
      in.def->state != nullptr ? in.def->state(in) : model::StateContract{};
  io.state_cells.assign(cells.begin(), cells.end());
  std::map<std::pair<std::uint8_t, std::uint16_t>, std::vector<std::byte>>
      carried;                                                         // (slot, tensor)
  std::map<std::uint8_t, std::vector<std::vector<double>>> ref_state;  // by slot
  for (std::size_t si = 0; si < steps.size(); ++si) {
    SCOPED_TRACE(std::format("step {}", si));
    const Step& step = steps[si];
    auto graph = BuildComponent(in, step.shape, formats, cells);
    ASSERT_TRUE(graph.has_value()) << graph.error();
    auto exec = ref::Executor::Create(graph->step.graph);
    ASSERT_TRUE(exec.has_value()) << exec.error().detail;
    // Weights.
    for (const RoleShape& r : *shapes) {
      const ir::TensorId leaf = graph->weights[r.role];
      if (!leaf.valid()) continue;  // not used in this step
      const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[r.role]};
      ASSERT_TRUE(exec->Bind(leaf, std::span(&part, 1)));
    }
    // State: zeros (or random values: `random_state`, a borrowed cache's),
    // or what the last step left (by slot).
    io.state.clear();
    io.segments.clear();
    for (const StepState& st : graph->step.state) {
      const std::uint8_t slot = step.shape.segments.at(st.ref.segment).slot;
      const auto at = carried.find({slot, st.ref.tensor});
      auto& start = ref_state[slot];
      if (start.empty()) start.resize(contract.tensors.size());
      if (at == carried.end()) {
        std::vector<float> values(ir::Elements(graph->step.graph.tensor(st.tensor).shape()), 0.0F);
        if (random_state) {
          for (float& x : values) x = u(rng);
        }
        ASSERT_TRUE(exec->BindValues(st.tensor, values));
        // The reference starts from what the graph holds (the values
        // stored in the state's encoding).
        auto held = exec->Values(st.tensor);
        ASSERT_TRUE(held.has_value());
        start[st.ref.tensor].assign(held->begin(), held->end());
      } else {
        const ref::PartView part{.role = format::PartRole::kData, .bytes = at->second};
        ASSERT_TRUE(exec->Bind(st.tensor, std::span(&part, 1)));
      }
    }
    for (const planner::Segment& s : step.shape.segments) {
      io.segments.push_back(
          {.rows = s.rows, .past = s.past, .read_cells = s.read_cells, .outputs = s.outputs});
      auto& st = ref_state[s.slot];
      if (st.empty()) st.resize(contract.tensors.size());
      io.state.push_back(st);
    }
    // Inputs.
    std::vector<double> stream(in.stage.width * step.shape.rows());
    for (double& x : stream) x = 2.0 * u(rng);
    io.features.clear();
    for (const StepInput& input : graph->step.inputs) {
      if (input.kind == InputKind::kStream) {
        std::vector<float> f(stream.begin(), stream.end());
        ASSERT_TRUE(exec->BindValues(input.tensor, f));
        // The reference reads what the graph reads: the F32 values.
        for (std::size_t i = 0; i < stream.size(); ++i) stream[i] = f[i];
        continue;
      }
      if (input.kind == InputKind::kFeatures) {
        std::vector<float> f(ir::Elements(graph->step.graph.tensor(input.tensor).shape()));
        for (float& x : f) x = 2.0F * u(rng);
        ASSERT_TRUE(exec->BindValues(input.tensor, f));
        io.features.assign(f.begin(), f.end());
        continue;
      }
      auto values = InputValues(input, step.shape, step.tokens);
      ASSERT_TRUE(values.has_value()) << values.error();
      ASSERT_TRUE(exec->BindInts(input.tensor, *values));
    }
    auto ran = exec->Run();
    ASSERT_TRUE(ran.has_value()) << ran.error().detail;
    // The reference.
    io.stream = stream;
    io.tokens = step.tokens;
    io.greedy = graph->step.greedy.valid();
    io.greedy_nan = in.def->kind == "head.linear" && in.params.Flag("native_argmax") &&
                            (step.shape.phase == planner::Phase::kVerify ||
                             step.shape.phase == planner::Phase::kDraft)
                        ? ir::ArgmaxNan::kIgnore
                        : ir::ArgmaxNan::kFirstColumn;
    auto r = in.def->reference(io);
    ASSERT_TRUE(r.has_value()) << r.error();
    if (graph->step.logits.valid()) {
      auto logits = exec->Values(graph->step.logits);
      ASSERT_TRUE(logits.has_value());
      EXPECT_LE(Gap(*logits, io.logits), tolerance);
      if (graph->step.greedy.valid()) {
        auto tokens = exec->Ints(graph->step.greedy);
        ASSERT_TRUE(tokens.has_value());
        ASSERT_EQ(tokens->size(), io.tokens_out.size());
        for (std::size_t i = 0; i < tokens->size(); ++i) EXPECT_EQ((*tokens)[i], io.tokens_out[i]);
      }
      if (graph->step.features.valid()) {
        auto features = exec->Values(graph->step.features);
        ASSERT_TRUE(features.has_value());
        EXPECT_LE(Gap(*features, io.features_out), tolerance);
      }
    } else if (graph->stream_out.valid()) {
      auto out = exec->Values(graph->stream_out);
      ASSERT_TRUE(out.has_value());
      EXPECT_LE(Gap(*out, io.stream), tolerance);
    }
    for (const StepState& st : graph->step.state) {
      const std::uint8_t slot = step.shape.segments.at(st.ref.segment).slot;
      auto values = exec->Values(st.tensor);
      ASSERT_TRUE(values.has_value());
      EXPECT_LE(Gap(*values, io.state[st.ref.segment][st.ref.tensor]),
                state_tolerance > 0.0 ? state_tolerance : tolerance)
          << "state " << st.ref.tensor << " of slot " << int{slot};
      const auto bytes = exec->Bytes(st.tensor);
      carried[{slot, st.ref.tensor}] = std::vector<std::byte>(bytes.begin(), bytes.end());
    }
    for (std::size_t s = 0; s < step.shape.segments.size(); ++s) {
      ref_state[step.shape.segments[s].slot] = io.state[s];
    }
  }
}

format::TensorFormat Matrix(Encoding e, std::uint64_t k, std::uint64_t n) {
  format::TensorFormat f = ir::Dense(e, ir::ShapeOf({k, n}));
  if (format::Info(e).quantized) f.layout = format::Layout::kGgmlBlocks;
  return f;
}
format::TensorFormat Vector(std::uint64_t n) { return ir::Dense(Encoding::kF32, ir::ShapeOf({n})); }

constexpr std::uint64_t kWidth = 64;
// Attention's tolerances. Both sides store K and V in F16, the executor
// from F32-rounded results and the reference from doubles, so a value
// within rounding of an F16 midpoint may land one F16 step apart (2^-11
// of the largest value at most); the outputs that read it move by less.
constexpr double kF16StateTolerance = 1.0 / 1024;
constexpr double kOutputTolerance = 1e-4;
const StageFacts kFacts{.width = kWidth, .context = 4096};

ComponentRef Attention(std::int64_t window, double softcap, bool qk_norm, std::string rotation,
                       std::string placement) {
  return C(
      "attention.dense",
      {P("placement", N(std::move(placement))), P("eps", R(1e-6)), P("heads", I(4)),
       P("kv_heads", I(2)), P("key_dim", I(32)), P("value_dim", I(32)),
       P("scale", R(1.0 / std::sqrt(32.0))), P("softcap", R(softcap)), P("window", I(window)),
       P("rotation", N(std::move(rotation))), P("rope_dims", I(32)), P("rope_base", R(10000.0)),
       P("rope_scale", R(window == 0 ? 0.125 : 1.0)), P("qk_norm", ParamValue(qk_norm))});
}

std::vector<format::TensorFormat> AttentionFormats(Encoding e) {
  return {Vector(kWidth),
          Matrix(e, kWidth, 128),
          Matrix(e, kWidth, 64),
          Matrix(e, kWidth, 64),
          Matrix(e, 128, kWidth),
          Vector(kWidth),
          Vector(32),
          Vector(32),
          Vector(16)};
}

// Two owners: prefills of 5 and 3 rows, a decode each, a longer chunk,
// then one that wraps a small ring.
std::vector<Step> AttentionSteps(std::uint32_t read) {
  return {
      {.shape = {.segments = {Seg(0, 5, 0, read, 5), Seg(3, 3, 0, read, 3)}}},
      {.shape = {.segments = {Seg(0, 1, 5, read, 1), Seg(3, 1, 3, read, 1)}}},
      {.shape = {.segments = {Seg(0, 12, 6, read, 12)}}},
      {.shape = {.segments = {Seg(3, 2, 4, read, 2), Seg(0, 8, 18, read, 8)}}},
  };
}

TEST(ComponentsTest, AttentionFragmentsMatchTheirReference) {
  struct Variant {
    std::int64_t window;
    double softcap;
    bool qk_norm;
    std::string rotation;
    std::string placement;
    Encoding weights;
  };
  for (const Variant& v : {Variant{8, 50.0, false, "neox", "sandwich", Encoding::kQ8_0},
                           Variant{0, 50.0, false, "neox", "sandwich", Encoding::kQ8_0},
                           Variant{8, 0.0, true, "neox", "sandwich", Encoding::kQ4_0},
                           Variant{0, 0.0, true, "normal", "pre", Encoding::kF32}}) {
    SCOPED_TRACE(std::format("window {} softcap {} qk_norm {} {} {}", v.window, v.softcap,
                             v.qk_norm, v.rotation, v.placement));
    const ComponentRef r = Attention(v.window, v.softcap, v.qk_norm, v.rotation, v.placement);
    auto in = MakeInstance(r, kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value()) << in.error();
    // A ring of window + 12 rows; a global cache of 32 cells.
    const std::uint64_t cells = v.window != 0 ? 20 : 32;
    const std::array<std::uint64_t, 2> both{cells, cells};
    Compare(*in, AttentionFormats(v.weights), both, AttentionSteps(32), kOutputTolerance,
            kF16StateTolerance);
  }
}

TEST(ComponentsTest, ZeroCenteredDenseNormsMatchTheirReferencesWithoutChangingSourceWeights) {
  for (const auto encoding : {Encoding::kF32, Encoding::kBF16}) {
    for (const bool qk : {false, true}) {
      auto component = Attention(16, qk ? 0.0 : 50.0, qk, "neox", "sandwich");
      component.params.push_back(P("norm_zero_centered", ParamValue(true)));
      std::ranges::sort(component.params, {}, &Param::name);
      auto in = MakeInstance(component, kFacts, Site::kLayer, 0, 1);
      ASSERT_TRUE(in) << in.error();
      auto formats = AttentionFormats(Encoding::kF32);
      for (const auto role : {0U, 5U, 6U, 7U}) formats[role].encoding = encoding;
      // The final eight-row chunk wraps; retain every row's 16-position window.
      const std::array<std::uint64_t, 2> cells{24, 24};
      Compare(*in, formats, cells, AttentionSteps(32), kOutputTolerance, kF16StateTolerance);
    }
    const std::vector<Step> steps{
        {.shape = {.segments = {Seg(0, 3, 0, 256, 1), Seg(1, 2, 7, 256, 2)}}}};
    auto ffn = MakeInstance(C("ffn.glu", {P("placement", N("sandwich")), P("eps", R(1e-6)),
                                          P("activation", N("gelu_tanh")), P("hidden", I(96)),
                                          P("norm_zero_centered", ParamValue(true))}),
                            kFacts, Site::kLayer, 0, 2);
    ASSERT_TRUE(ffn) << ffn.error();
    auto formats = std::vector<format::TensorFormat>{
        Vector(kWidth), Matrix(Encoding::kF32, kWidth, 96), Matrix(Encoding::kF32, kWidth, 96),
        Matrix(Encoding::kF32, 96, kWidth), Vector(kWidth)};
    formats[0].encoding = formats[4].encoding = encoding;
    Compare(*ffn, formats, {}, steps, 1e-5);
    auto norm =
        MakeInstance(C("norm.rms", {P("eps", R(1e-6)), P("norm_zero_centered", ParamValue(true))}),
                     kFacts, Site::kFinalNorm, 0, 3);
    ASSERT_TRUE(norm) << norm.error();
    auto format = Vector(kWidth);
    format.encoding = encoding;
    Compare(*norm, std::span(&format, 1), {}, steps, 1e-5);
  }
}

TEST(ComponentsTest, ZeroCenteredFinalNormAddsOneExactlyOnceAndRefusesFoldContradiction) {
  auto component = C("norm.rms", {P("eps", R(0.0)), P("norm_zero_centered", ParamValue(true))});
  auto instance = MakeInstance(component, {.width = 2, .context = 16}, Site::kFinalNorm, 0, 0);
  ASSERT_TRUE(instance) << instance.error();
  RefIO io{.instance = &*instance};
  io.stream = {3.0, 4.0};
  io.weights = {{0.0F, 0.0F}};
  io.segments = {{.rows = 1}};
  ASSERT_TRUE(instance->def->reference(io));
  EXPECT_DOUBLE_EQ(io.stream[0], 3.0 / std::sqrt(12.5));
  EXPECT_DOUBLE_EQ(io.stream[1], 4.0 / std::sqrt(12.5));
  EXPECT_EQ(io.weights[0], (std::vector<float>{0.0F, 0.0F}));
  const std::array<std::string, 1> folds{"norm.plus_one"};
  auto facts = kFacts;
  facts.folds = folds;
  EXPECT_FALSE(MakeInstance(component, facts, Site::kFinalNorm, 0, 0));
}

// The bias is before rotation (including K's stored values); scaling
// scores and materializing them have the same mathematical reference.
TEST(ComponentsTest, BiasedAttentionAndExplicitScoresMatchTheirReference) {
  for (const bool explicit_scores : {false, true}) {
    auto r = Attention(0, 0.0, false, "neox", "pre");
    r.params.push_back(P("qkv_bias", true));
    r.params.push_back(P("scale_scores", true));
    r.params.push_back(P("explicit_scores", explicit_scores));
    std::ranges::sort(r.params, {}, &Param::name);
    auto in = MakeInstance(r, kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value()) << in.error();
    auto formats = AttentionFormats(Encoding::kF16);
    formats.push_back(Vector(128));
    formats.push_back(Vector(64));
    formats.push_back(Vector(64));
    const std::array<std::uint64_t, 2> cells{32, 32};
    Compare(*in, formats, cells, AttentionSteps(32), kOutputTolerance, kF16StateTolerance,
            {{9, {-0.25F, 0.25F}}, {10, {-0.5F, 0.5F}}, {11, {-0.75F, 0.75F}}});
  }
}

TEST(ComponentsTest, ARingTooSmallForItsRowsAndWindowIsRefused) {
  const ComponentRef r = Attention(8, 0.0, false, "neox", "pre");
  auto in = MakeInstance(r, kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in.has_value());
  const std::array<std::uint64_t, 2> cells{20, 20};
  const auto formats = AttentionFormats(Encoding::kF32);
  const auto build = [&](planner::Segment s) {
    return BuildComponent(*in, {.segments = {s}}, formats, cells);
  };
  // 14 rows after 10 positions store position 23 in cell 3 before the
  // first row (position 10, window 8) reads position 3 there: refused.
  EXPECT_FALSE(build(Seg(0, 14, 10, 32, 1)));
  // 13 rows overwrite only position 2, which no row sees; 14 rows that do
  // not wrap overwrite nothing.
  EXPECT_TRUE(build(Seg(0, 13, 10, 32, 1)));
  EXPECT_TRUE(build(Seg(0, 14, 0, 32, 1)));
  // The scalar reference refuses the same step.
  RefIO io{.instance = &*in};
  io.segments = {{.rows = 14, .past = 10, .read_cells = 32, .outputs = 1}};
  io.state_cells = {20, 20};
  io.state = {{std::vector<double>(20 * 64), std::vector<double>(20 * 64)}};
  io.stream.assign(kWidth * 14, 0.0);
  io.weights.resize(in->def->roles.size());
  const auto shapes = in->def->shapes(*in);
  ASSERT_TRUE(shapes.has_value());
  for (const RoleShape& s : *shapes) io.weights[s.role].assign(s.ne[0] * s.ne[1], 0.0F);
  EXPECT_FALSE(in->def->reference(io));
}

TEST(ComponentsTest, AttentionStateOnlyStepsWriteTheSameState) {
  const ComponentRef r = Attention(8, 50.0, false, "neox", "sandwich");
  auto in = MakeInstance(r, kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in.has_value());
  const std::array<std::uint64_t, 2> cells{20, 20};
  // No outputs: only K and V are stored; the following step reads them.
  const std::vector<Step> steps{
      {.shape = {.segments = {Seg(0, 6, 0, 32, 0)}}},
      {.shape = {.segments = {Seg(0, 2, 6, 32, 2)}}},
  };
  Compare(*in, AttentionFormats(Encoding::kQ8_0), cells, steps, kOutputTolerance,
          kF16StateTolerance);
  auto state_only = BuildComponent(*in, steps[0].shape, AttentionFormats(Encoding::kQ8_0), cells);
  ASSERT_TRUE(state_only.has_value());
  EXPECT_FALSE(state_only->stream_out.valid());
  std::size_t attention = 0;
  for (const ir::Node& n : state_only->step.graph.nodes())
    attention += n.key.op == ir::Op::kAttention;
  EXPECT_EQ(attention, 0U);
  EXPECT_FALSE(state_only->weights[1].valid());  // no query product
}

TEST(ComponentsTest, FeedForwardNormEmbeddingAndHeadMatchTheirReferences) {
  const std::vector<Step> rows{
      {.shape = {.segments = {Seg(0, 3, 0, 256, 1), Seg(1, 2, 7, 256, 2)}},
       .tokens = {4, 0, 19, 7, 3}},
  };
  for (const auto& [activation, placement, e] :
       {std::tuple{"gelu_tanh", "sandwich", Encoding::kQ8_0},
        std::tuple{"silu", "pre", Encoding::kF16}}) {
    auto in = MakeInstance(C("ffn.glu", {P("placement", N(placement)), P("eps", R(1e-6)),
                                         P("activation", N(activation)), P("hidden", I(96))}),
                           kFacts, Site::kLayer, 0, 2);
    ASSERT_TRUE(in.has_value()) << in.error();
    const std::vector<format::TensorFormat> f{Vector(kWidth), Matrix(e, kWidth, 96),
                                              Matrix(e, kWidth, 96), Matrix(e, 96, kWidth),
                                              Vector(kWidth)};
    Compare(*in, f, {}, rows, 1e-5);
  }
  auto norm = MakeInstance(C("norm.rms", {P("eps", R(1e-6))}), kFacts, Site::kFinalNorm, 0, 3);
  ASSERT_TRUE(norm.has_value());
  const std::vector<format::TensorFormat> nf{Vector(kWidth)};
  Compare(*norm, nf, {}, rows, 1e-5);
  auto emb = MakeInstance(C("embedding.token", {P("scale", R(std::sqrt(64.0)))}), kFacts,
                          Site::kEmbedding, 0, 0);
  ASSERT_TRUE(emb.has_value());
  const std::vector<format::TensorFormat> ef{Matrix(Encoding::kQ8_0, kWidth, 20)};
  Compare(*emb, ef, {}, rows, 1e-5);
  for (const std::optional<double> softcap :
       std::array<std::optional<double>, 2>{30.0, std::nullopt}) {
    StageFacts facts = kFacts;
    facts.softcap = softcap;
    auto head = MakeInstance(C("head.linear", {}), facts, Site::kOutput, 0, 4);
    ASSERT_TRUE(head.has_value());
    // The weight, a projection and (unused) a selected head's ids.
    const std::vector<format::TensorFormat> hf{Matrix(Encoding::kQ8_0, kWidth, 40),
                                               Matrix(Encoding::kQ8_0, kWidth, 32), Vector(1)};
    Compare(*head, hf, {}, rows, 1e-5);
    // Greedy: one output a segment, each row's highest logit.
    const std::vector<Step> greedy{
        {.shape = {.segments = {Seg(0, 3, 0, 256, 1, true), Seg(1, 2, 7, 256, 1, true)}}}};
    Compare(*head, hf, {}, greedy, 1e-5);
  }
}

// The F32 component normalizes channels independently for every pixel/owner.
TEST(ComponentsTest, ChannelNormMatchesScalarAcrossOwnersAndSourceEncodings) {
  const std::vector<Step> rows{
      {.shape = {.segments = {Seg(0, 3, 0, 256, 3), Seg(1, 2, 7, 256, 2)}}},
      {.shape = {.segments = {Seg(1, 1, 9, 256, 1)}}},
  };
  for (const bool silu : {false, true}) {
    const auto component = C("norm.channel", {P("silu", ParamValue(silu))});
    auto in = MakeInstance(component, kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value()) << in.error();
    EXPECT_TRUE(in->def->row_local);
    EXPECT_EQ(in->def->state, nullptr);
    for (const auto encoding : {Encoding::kF32, Encoding::kBF16}) {
      const std::array formats{ir::Dense(encoding, ir::ShapeOf({kWidth}))};
      Compare(*in, formats, {}, rows, 1e-5);
    }
  }
}

TEST(ComponentsTest, AdaptiveNormAndGatedResidualComposeWithTheOriginalCarry) {
  const StageFacts facts{.width = 4, .context = 256};
  const auto norm_ref = C("norm.adaptive", {P("side", N("scale"))});
  const auto residual_ref = C("residual.gated", {P("side", N("gate"))});
  auto norm = MakeInstance(norm_ref, facts, Site::kLayer, 0, 1);
  auto residual = MakeInstance(residual_ref, facts, Site::kLayer, 0, 2);
  ASSERT_TRUE(norm.has_value()) << norm.error();
  ASSERT_TRUE(residual.has_value()) << residual.error();
  const planner::StepShape shape{.segments = {Seg(3, 2, 0, 256, 2), Seg(7, 1, 0, 256, 1)}};
  auto ng = BuildComponent(*norm, shape, {}, {}, {.side = 4});
  auto rg = BuildComponent(*residual, shape, {}, {}, {.carry = 4, .side = 4});
  ASSERT_TRUE(ng.has_value()) << ng.error();
  ASSERT_TRUE(rg.has_value()) << rg.error();
  EXPECT_EQ(ng->carry_out, ng->stream_in);
  EXPECT_TRUE(ng->step.state.empty());
  EXPECT_TRUE(rg->step.state.empty());
  for (const auto* graph : {&ng->step.graph, &rg->step.graph}) {
    EXPECT_EQ(
        std::ranges::count_if(
            graph->nodes(), [](const auto& n) { return n.key == ir::Primitive(ir::Op::kConvert); }),
        0);
  }
  const std::array input{0.F, 0.F, 0.F, 0.F, 2.F, 2.F, 2.F, 2.F, -3.F, 1.F, 2.F, 4.F};
  const std::array scales{0.F, 2.F, -1.F, .5F, -1.F, -1.F, -1.F, -1.F, 0.F, -1.F, 2.F, -.5F};
  const std::array gates{0.F, 1.F, -1.F, 20.F, -20.F, 0.F, .25F, -.25F, 1.F, -1.F, 0.F, 20.F};
  auto ne = ref::Executor::Create(ng->step.graph);
  ASSERT_TRUE(ne.has_value());
  ASSERT_TRUE(ne->BindValues(ng->stream_in, input));
  ASSERT_TRUE(ne->BindValues(ng->side_in, scales));
  ASSERT_TRUE(ne->Run());
  auto normalized = ne->Values(ng->stream_out);
  auto saved = ne->Values(ng->carry_out);
  ASSERT_TRUE(normalized.has_value());
  ASSERT_TRUE(saved.has_value());
  EXPECT_EQ(*saved, (std::vector<float>(input.begin(), input.end())));
  RefIO io{.instance = &*norm};
  io.segments = {{.rows = 2}, {.rows = 1}};
  io.stream.assign(input.begin(), input.end());
  io.side.assign(scales.begin(), scales.end());
  ASSERT_TRUE(norm->def->reference(io));
  EXPECT_LT(Gap(*normalized, io.stream), 1e-6);
  EXPECT_EQ(io.carry, (std::vector<double>(input.begin(), input.end())));
  // A nontrivial intervening branch consumes the normalized stream. The
  // residual must still read the original, not that stream or the branch.
  std::vector<float> branch(normalized->size());
  for (std::size_t i = 0; i < branch.size(); ++i) {
    branch[i] = 2.F * (*normalized)[i] + .5F;
    io.stream[i] = 2.0 * io.stream[i] + .5;
  }
  auto re = ref::Executor::Create(rg->step.graph);
  ASSERT_TRUE(re.has_value());
  ASSERT_TRUE(re->BindValues(rg->stream_in, branch));
  ASSERT_TRUE(re->BindValues(rg->carry_in, *saved));
  ASSERT_TRUE(re->BindValues(rg->side_in, gates));
  ASSERT_TRUE(re->Run());
  auto out = re->Values(rg->stream_out);
  ASSERT_TRUE(out.has_value());
  io.instance = &*residual;
  io.side.assign(gates.begin(), gates.end());
  ASSERT_TRUE(residual->def->reference(io));
  EXPECT_LT(Gap(*out, io.stream), 1e-6);
  EXPECT_EQ((*out)[0], input[0]);
  EXPECT_EQ((*out)[5], input[5]);
  EXPECT_EQ((*out)[10], input[10]);
  EXPECT_NEAR((*out)[3], .5F, 1e-6);
  EXPECT_NEAR((*out)[4], 1.5F, 1e-6);
}

TEST(ComponentsTest, AdaptiveBlocksRefuseMissingAndMismatchedOperands) {
  const StageFacts facts{.width = 4, .context = 256};
  const planner::StepShape shape{.segments = {Seg(0, 2, 0, 256, 2), Seg(1, 1, 0, 256, 1)}};
  for (const std::string kind : {"norm.adaptive", "residual.gated"}) {
    const auto component = C(kind, {P("side", N("condition"))});
    auto in = MakeInstance(component, facts, Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value());
    EXPECT_FALSE(BuildComponent(*in, shape, {}, {}));
    EXPECT_FALSE(BuildComponent(*in, shape, {}, {}, {.carry = 4, .side = 1}));
    EXPECT_FALSE(BuildComponent(*in, shape, {}, {}, {.carry = 4, .side = 8}));
    EXPECT_FALSE(BuildComponent(*in, shape, {}, {}, {.carry = 4, .side = 4, .side_ids = true}));
    if (kind == "residual.gated") {
      EXPECT_FALSE(BuildComponent(*in, shape, {}, {}, {.side = 4}));
      EXPECT_FALSE(BuildComponent(*in, shape, {}, {}, {.carry = 1, .side = 4}));
    }
    EXPECT_FALSE(MakeInstance(C(kind, {}), facts, Site::kLayer, 0, 1));
    EXPECT_FALSE(MakeInstance(C(kind, {P("side", N(""))}), facts, Site::kLayer, 0, 1));
    EXPECT_FALSE(MakeInstance(component, {.width = 4, .streams = 2}, Site::kLayer, 0, 1));
    EXPECT_FALSE(MakeInstance(component, facts, Site::kFinalNorm, 0, 1));
    RefIO io{.instance = &*in};
    io.segments = {{.rows = 3}};
    io.stream.resize(12);
    io.carry.resize(12);
    io.side.resize(4);  // A single conditioning row must not silently broadcast.
    const auto before = io.stream;
    EXPECT_FALSE(in->def->reference(io));
    EXPECT_EQ(io.stream, before);
    io.side.resize(12);
    io.carry.resize(4);
    if (kind == "residual.gated") EXPECT_FALSE(in->def->reference(io));
  }
  EXPECT_FALSE(MakeInstance(C("norm.adaptive", {P("side", N("s")), P("eps", R(0))}), facts,
                            Site::kLayer, 0, 1));
}

TEST(ComponentsTest, AdaptiveBlocksValidateExactOwnerRowsBeforeEmission) {
  const StageFacts facts{.width = 4, .context = 256};
  for (const std::string kind : {"norm.adaptive", "residual.gated"}) {
    const auto component = C(kind, {P("side", N("s"))});
    auto in = MakeInstance(component, facts, Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value());
    // Malformed side row count, malformed carry count, or a foreign owner.
    for (int mode = 0; mode < 3; ++mode) {
      if (kind == "norm.adaptive" && mode == 1) continue;
      ir::Builder b;
      StepContext ctx(b, {{.slot = 3, .rows = 2}, {.slot = 7, .first_row = 2, .rows = 1}}, false);
      const auto stream = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({4, 3})), "stream");
      const auto carry =
          b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({4, mode == 1 ? 1U : 3U})), "carry");
      if (mode == 2) b.SetOwner(0);
      const auto side =
          b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({4, mode == 0 ? 1U : 3U})), "side");
      b.SetJoined();
      Fragment f{.b = b,
                 .instance = *in,
                 .step = ctx,
                 .stream = stream,
                 .weight = {},
                 .state = {},
                 .carry = carry,
                 .side = [&](std::string_view) { return side; }};
      const auto nodes = b.node_count();
      const auto tensors = b.tensor_count();
      EXPECT_FALSE(in->def->emit(f));
      EXPECT_EQ(b.node_count(), nodes);
      EXPECT_EQ(b.tensor_count(), tensors);
    }
  }
}

TEST(ComponentsTest, ChannelNormUsesClampedL2AndF32Output) {
  const StageFacts facts{.width = 2, .context = 256};
  const auto component = C("norm.channel", {});
  auto in = MakeInstance(component, facts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in.has_value()) << in.error();
  const planner::StepShape shape{.segments = {Seg(0, 2, 0, 256, 2), Seg(1, 1, 0, 256, 1)}};
  const std::array formats{Vector(2)};
  auto graph = BuildComponent(*in, shape, formats, {});
  ASSERT_TRUE(graph.has_value()) << graph.error();
  EXPECT_TRUE(graph->step.state.empty());
  EXPECT_EQ(graph->step.graph.tensor(graph->stream_out).format.encoding, Encoding::kF32);
  EXPECT_EQ(
      std::ranges::count_if(graph->step.graph.nodes(),
                            [](const auto& n) { return n.key == ir::Primitive(ir::Op::kL2Norm); }),
      1);
  EXPECT_EQ(
      std::ranges::count_if(graph->step.graph.nodes(),
                            [](const auto& n) { return n.key == ir::Primitive(ir::Op::kConvert); }),
      0);
  auto exec = ref::Executor::Create(graph->step.graph);
  ASSERT_TRUE(exec.has_value()) << exec.error().detail;
  const std::array input{0.0F, 0.0F, 3e-14F, 4e-14F, 3.0F, 4.0F};
  const std::array gamma{2.0F, -1.0F};
  ASSERT_TRUE(exec->BindValues(graph->stream_in, input));
  ASSERT_TRUE(exec->BindValues(graph->weights[0], gamma));
  ASSERT_TRUE(exec->Run());
  auto out = exec->Values(graph->stream_out);
  ASSERT_TRUE(out.has_value());
  const double scale = static_cast<float>(std::sqrt(2.0));
  const std::array expected{0.0, 0.0, 0.06 * scale, -0.04 * scale, 1.2 * scale, -0.8 * scale};
  ASSERT_EQ(out->size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR((*out)[i], expected[i], 2e-7) << i;
  RefIO io{.instance = &*in};
  io.segments = {{.rows = 2}, {.rows = 1}};
  io.stream.assign(input.begin(), input.end());
  io.weights.resize(1);
  io.weights[0].assign(gamma.begin(), gamma.end());
  ASSERT_TRUE(in->def->reference(io));
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(io.stream[i], expected[i], 2e-7) << i;
  io.weights[0].pop_back();
  EXPECT_FALSE(in->def->reference(io));
}

TEST(ComponentsTest, ChannelNormRefusesInvalidParametersAndWeightShapes) {
  const StageFacts facts{.width = 2, .context = 256};
  EXPECT_FALSE(MakeInstance(C("norm.channel", {P("eps", R(0.0))}), facts, Site::kLayer, 0, 1));
  EXPECT_FALSE(MakeInstance(C("norm.channel", {P("eps", R(-1.0))}), facts, Site::kLayer, 0, 1));
  EXPECT_FALSE(MakeInstance(C("norm.channel", {P("silu", I(1))}), facts, Site::kLayer, 0, 1));
  const auto component = C("norm.channel", {});
  auto in = MakeInstance(component, facts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in.has_value());
  const planner::StepShape shape{.segments = {Seg(0, 1, 0, 256, 1)}};
  EXPECT_FALSE(BuildComponent(*in, shape, std::array{Vector(3)}, {}));
  const auto f16 = ir::Dense(Encoding::kF16, ir::ShapeOf({2}));
  const auto graph = BuildComponent(*in, shape, std::array{f16}, {});
  ASSERT_TRUE(graph) << graph.error();
  EXPECT_EQ(graph->step.graph.tensor(graph->weights.at(0)).format, f16);
  auto wide = facts;
  wide.width = kMaxStreamWidth + 1;
  EXPECT_FALSE(MakeInstance(C("norm.channel", {}), wide, Site::kLayer, 0, 1));
  wide = facts;
  wide.streams = 2;
  EXPECT_FALSE(MakeInstance(C("norm.channel", {}), wide, Site::kLayer, 0, 1));
}

TEST(ComponentsTest, ChannelNormBuildsInACompleteStatelessSpec) {
  spec::Stage stage{.name = "target", .context = 256};
  stage.stream.width = 2;
  stage.embedding = C("embedding.token", {});
  stage.embedding->weights = {{.role = "table", .tensor = "tokens"}};
  stage.heads.final_norm = C("norm.channel", {});
  stage.heads.final_norm->weights = {{.role = "weight", .tensor = "gamma"}};
  stage.heads.output = C("head.linear", {});
  stage.heads.output->weights = {{.role = "weight", .tensor = "tokens"}};
  spec::ModelSpec model_spec{.stages = {stage}};
  std::vector<spec::SourceTensor> tensors{
      {.name = "gamma", .format = Vector(2)},
      {.name = "tokens", .format = Matrix(Encoding::kF32, 2, 4)},
  };
  const std::array<std::span<const spec::SourceTensor>, 1> tables{tensors};
  ASSERT_TRUE(ValidateSpec(model_spec, tables));
  const auto layout = StageLayoutUniform(model_spec, 0, 256, 3, kExtent);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  EXPECT_TRUE(layout->items.empty());
  const planner::StepShape shape{.segments = {Seg(0, 2, 0, 256, 1), Seg(1, 1, 0, 256, 1)}};
  auto step = BuildStep(model_spec, 0, shape, {.tensors = tensors, .state = &*layout});
  ASSERT_TRUE(step.has_value()) << step.error();
  EXPECT_EQ(step->graph.tensor(step->logits).format.encoding, Encoding::kF32);
  EXPECT_EQ(step->graph.tensor(step->logits).shape().ne[1], 2U);
  EXPECT_TRUE(step->state.empty());
  const auto digest = model_spec.Digest();
  model_spec.stages[0].heads.final_norm->params = {P("silu", ParamValue(true))};
  EXPECT_NE(model_spec.Digest(), digest);
  ASSERT_TRUE(ValidateSpec(model_spec, tables));
  tensors[0].format = Vector(3);
  EXPECT_FALSE(ValidateSpec(model_spec, tables));
}

// Alias-wide conversion is paid and validated before changing any component.
TEST(ComponentsTest, Q8CacheOverrideIsPaidAtomicAndChangesStateIdentity) {
  for (auto model_spec : {Gemma2Spec(), Gemma3Spec()}) {
    const auto original = model_spec;
    const auto old_digest = model_spec.Digest();
    ASSERT_TRUE(ApplyKvCacheEncoding(model_spec, Encoding::kF16, 0));
    EXPECT_EQ(model_spec.Digest(), old_digest);
    const auto bytes = KvCacheEncodingBytes(model_spec, Encoding::kQ8_0);
    ASSERT_TRUE(bytes) << bytes.error();
    ASSERT_GT(*bytes, 0U);
    EXPECT_FALSE(ApplyKvCacheEncoding(model_spec, Encoding::kQ8_0, *bytes - 1));
    EXPECT_EQ(model_spec.Digest(), old_digest);
    ASSERT_TRUE(ApplyKvCacheEncoding(model_spec, Encoding::kQ8_0, *bytes));
    const auto digest = model_spec.Digest();
    EXPECT_NE(digest, old_digest);
    const auto before = StageLayoutUniform(original, 0, 4096, 128, kExtent);
    const auto after = StageLayoutUniform(model_spec, 0, 4096, 128, kExtent);
    ASSERT_TRUE(before) << before.error();
    ASSERT_TRUE(after) << after.error();
    ASSERT_EQ(before->items.size(), after->items.size());
    for (std::size_t i = 0; i < after->items.size(); ++i) {
      const auto& a = after->items[i];
      const auto& b = before->items[i];
      EXPECT_EQ(a.contract.encoding, Encoding::kQ8_0);
      EXPECT_EQ(a.contract.cell_bytes * 64, b.contract.cell_bytes * 34);
      EXPECT_EQ(a.cells, b.cells);
      EXPECT_EQ(a.bytes, a.cells * a.contract.cell_bytes);
      EXPECT_EQ(a.offset % kExtent, 0U);
    }
    EXPECT_NE(model::KeptLayoutId(old_digest, std::span(&*before, 1)),
              model::KeptLayoutId(digest, std::span(&*after, 1)));
    EXPECT_FALSE(ApplyKvCacheEncoding(model_spec, Encoding::kF16, 0));
    EXPECT_EQ(model_spec.Digest(), digest);
  }
  auto unsupported = Gemma2Spec();
  for (auto& component : unsupported.stages[0].layers.back().components) {
    if (component.kind != "attention.dense") continue;
    for (auto& p : component.params)
      if (p.name == "key_dim" || p.name == "value_dim") p.value = I(512);
  }
  const auto digest = unsupported.Digest();
  const auto bytes = KvCacheEncodingBytes(unsupported, Encoding::kQ8_0);
  ASSERT_TRUE(bytes);
  EXPECT_FALSE(ApplyKvCacheEncoding(unsupported, Encoding::kQ8_0, *bytes));
  EXPECT_EQ(unsupported.Digest(), digest);  // late refusal did not rewrite earlier layers
  unsupported.stages[0].layers.back().components.front().kind = "attention.mqa";
  const auto specialized = unsupported.Digest();
  EXPECT_FALSE(ApplyKvCacheEncoding(unsupported, Encoding::kQ8_0, *bytes));
  EXPECT_EQ(unsupported.Digest(), specialized);
}

TEST(ComponentsTest, Q8CacheReferenceUsesBlockStateAcrossRingAndOwners) {
  auto component = Attention(8, 0.0, false, "neox", "pre");
  for (auto& p : component.params) {
    if (p.name == "key_dim" || p.name == "value_dim" || p.name == "rope_dims") p.value = I(64);
    if (p.name == "rope_scale") p.value = R(1e-30);
  }
  component.params.push_back(P("kv_encoding", N("q8_0")));
  component.params.push_back(P("qkv_bias", true));
  std::ranges::sort(component.params, {}, &Param::name);
  const auto instance = MakeInstance(component, kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(instance) << instance.error();
  auto formats = AttentionFormats(Encoding::kF32);
  formats[1] = Matrix(Encoding::kF32, kWidth, 256);
  formats[2] = formats[3] = Matrix(Encoding::kF32, kWidth, 128);
  formats[4] = Matrix(Encoding::kF32, 256, kWidth);
  formats.push_back(Vector(256));
  formats.push_back(Vector(128));
  formats.push_back(Vector(128));
  const std::array<std::uint64_t, 2> cells{20, 20};
  // Exact nonzero bias-driven blocks isolate encode/store/read and ring
  // geometry from projection-rounding differences between the two references.
  Compare(*instance, formats, cells, AttentionSteps(32), kOutputTolerance, kF16StateTolerance,
          {{1, {0.0F, 0.0F}},
           {2, {0.0F, 0.0F}},
           {3, {0.0F, 0.0F}},
           {9, {0.125F, 0.0F}},
           {10, {0.25F, 0.0F}},
           {11, {0.5F, 0.0F}}});
}

TEST(ComponentsTest, Q8CacheViewsPreserveWholeHeadBlocksAndRefuseUnsupportedGeometry) {
  for (const std::int64_t dim : {64, 256}) {
    auto component = Attention(0, 0.0, false, "neox", "pre");
    for (auto& p : component.params)
      if (p.name == "key_dim" || p.name == "value_dim" || p.name == "rope_dims") p.value = I(dim);
    component.params.push_back(P("kv_encoding", N("q8_0")));
    std::ranges::sort(component.params, {}, &Param::name);
    auto instance = MakeInstance(component, kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(instance) << instance.error();
    auto formats = AttentionFormats(Encoding::kF16);
    const auto d = static_cast<std::uint64_t>(dim);
    formats[1] = Matrix(Encoding::kF16, kWidth, 4 * d);
    formats[2] = Matrix(Encoding::kF16, kWidth, 2 * d);
    formats[3] = formats[2];
    formats[4] = Matrix(Encoding::kF16, 4 * d, kWidth);
    const std::array<std::uint64_t, 2> cells{256, 256};
    const auto graph = BuildComponent(*instance, AttentionSteps(256)[0].shape, formats, cells);
    ASSERT_TRUE(graph) << graph.error();
    std::size_t stores = 0, attention = 0;
    for (const auto& node : graph->step.graph.nodes()) {
      if (node.key.op == ir::Op::kCacheStore) {
        ++stores;
        EXPECT_EQ(graph->step.graph.tensor(node.src[1]).format.encoding, Encoding::kF32);
      }
      if (node.key.op != ir::Op::kAttention) continue;
      ++attention;
      for (const auto input : {node.src[1], node.src[2]}) {
        const auto& tensor = graph->step.graph.tensor(input);
        EXPECT_EQ(tensor.format.encoding, Encoding::kQ8_0);
        EXPECT_EQ(tensor.nb[0], 34U);
        EXPECT_EQ(tensor.nb[1], 2 * d / 32 * 34);  // complete cell
        EXPECT_EQ(tensor.nb[2], d / 32 * 34);      // one KV head
      }
    }
    EXPECT_EQ(stores, 4U);
    EXPECT_EQ(attention, 2U);
    RefIO reference{.instance = &*instance};
    EXPECT_FALSE(instance->def->reference(reference));  // no codec, before any state mutation
    auto explicit_scores = component;
    explicit_scores.params.push_back(P("explicit_scores", true));
    std::ranges::sort(explicit_scores.params, {}, &Param::name);
    EXPECT_FALSE(MakeInstance(explicit_scores, kFacts, Site::kLayer, 0, 1));
    auto borrowed = component;
    borrowed.kind = "attention.borrowed";
    EXPECT_FALSE(MakeInstance(borrowed, kFacts, Site::kLayer, 0, 1));
    if (dim == 64) {
      auto capped = component;
      for (auto& p : capped.params)
        if (p.name == "softcap") p.value = R(50.0);
      EXPECT_FALSE(MakeInstance(capped, kFacts, Site::kLayer, 0, 1));
    }
    for (auto& p : component.params)
      if (p.name == "value_dim") p.value = I(dim == 64 ? 256 : 512);
    EXPECT_FALSE(MakeInstance(component, kFacts, Site::kLayer, 0, 1));
  }
}

// ---------------------------------------------------------------- the step

TEST(ComponentsTest, HeadReferencesHonorTheCallsGreedyNanPolicy) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<std::vector<float>, 3> values{std::vector<float>{nan, 1, 3, 3},
                                                 std::vector<float>{nan, nan, nan, nan},
                                                 std::vector<float>{1, 3, 3, 2}};
  const std::array<std::int64_t, 4> ids{11, 21, 31, 41};
  for (bool selected : {false, true}) {
    for (bool native : {false, true}) {
      auto in = MakeInstance(C("head.linear", {P("native_argmax", ParamValue(native)),
                                               P("selected", ParamValue(selected))}),
                             {.width = 1, .context = 8}, Site::kOutput, 0, 0);
      ASSERT_TRUE(in.has_value());
      std::vector<format::TensorFormat> formats(in->def->roles.size(), Vector(1));
      formats[0] = Matrix(Encoding::kF32, 1, 4);
      formats[2] = ir::Dense(Encoding::kI32, ir::ShapeOf({1, 4}));
      for (planner::Phase phase :
           {planner::Phase::kDecode, planner::Phase::kVerify, planner::Phase::kDraft}) {
        const planner::StepShape shape{.phase = phase, .segments = {Seg(0, 1, 0, 1, 1, true)}};
        const bool ignore = native && phase != planner::Phase::kDecode;
        for (std::size_t fixture = 0; fixture < values.size(); ++fixture) {
          SCOPED_TRACE(std::format("selected {} native {} phase {} fixture {}", selected, native,
                                   static_cast<int>(phase), fixture));
          auto graph = BuildComponent(*in, shape, formats, {});
          ASSERT_TRUE(graph.has_value()) << graph.error();
          auto exec = ref::Executor::Create(graph->step.graph);
          ASSERT_TRUE(exec.has_value());
          ASSERT_TRUE(exec->BindValues(graph->weights[0], values[fixture]));
          if (selected) ASSERT_TRUE(exec->BindInts(graph->weights[2], ids));
          for (const StepInput& input : graph->step.inputs) {
            if (input.kind == InputKind::kStream) {
              const std::array<float, 1> stream{1};
              ASSERT_TRUE(exec->BindValues(input.tensor, stream));
            } else {
              const auto staged = InputValues(input, shape, {});
              ASSERT_TRUE(staged.has_value());
              ASSERT_TRUE(exec->BindInts(input.tensor, *staged));
            }
          }
          ASSERT_TRUE(exec->Run());
          RefIO io{.instance = &*in};
          io.segments = {{.rows = 1, .outputs = 1}};
          io.stream = {1};
          io.weights.resize(in->def->roles.size());
          io.weights[0] = values[fixture];
          if (selected) io.weights[2].assign(ids.begin(), ids.end());
          io.greedy = true;
          io.greedy_nan = ignore ? ir::ArgmaxNan::kIgnore : ir::ArgmaxNan::kFirstColumn;
          ASSERT_TRUE(in->def->reference(io));
          const std::size_t best = fixture == 0 ? (ignore ? 2 : 0) : fixture == 1 ? 0 : 1;
          const std::int64_t expected = selected ? ids[best] : static_cast<std::int64_t>(best);
          ASSERT_EQ(io.tokens_out.size(), 1U);
          EXPECT_EQ(io.tokens_out[0], expected);
          const auto tokens = exec->Ints(graph->step.greedy);
          ASSERT_TRUE(tokens.has_value());
          EXPECT_EQ(*tokens, (std::vector<std::int64_t>{expected}));
        }
      }
    }
  }
}

std::size_t Count(const ir::Graph& g, ir::Op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(g.nodes(), [op](const ir::Node& n) { return n.key.op == op; }));
}

// A framework input has its own declared owner, but may be requested
// while a component emits either joined or per-owner computation.
TEST(ComponentsTest, LazyFrameworkInputsPreserveCallerContextOnMissAndHit) {
  for (const bool joined : {false, true}) {
    ir::Builder b;
    StepContext ctx(b, {{.slot = 0, .rows = 8}, {.slot = 1, .first_row = 8, .rows = 8}}, false);
    ctx.SetComponent(7);
    b.SetComponent(7);
    if (joined)
      b.SetJoined();
    else
      b.SetOwner(1);
    const auto context = b.context();
    const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 16})));
    const CellKey ring{.capacity = 8, .ring = true, .recent = true};
    std::array<ir::TensorId, 2> reads{}, phases{}, outputs{};
    for (std::size_t i = 0; i < 2; ++i) {
      // Cold HostInput used to reset this owner's following computation;
      // its cache hit did not. CompressorPhasesInput has a joined input declaration.
      reads[i] =
          ctx.HostInput({.def = &CompressorPoolReadsInput(), .ratio = 4, .overlap = true}, 1, ring);
      EXPECT_EQ(b.context(), context);
      phases[i] = ctx.HostInput({.def = &CompressorPhasesInput(), .ratio = 4});
      EXPECT_EQ(b.context(), context);
      outputs[i] = b.Scale(x, 2.0F);
      b.Output(outputs[i]);
    }
    EXPECT_EQ(reads[0], reads[1]);
    EXPECT_EQ(phases[0], phases[1]);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    EXPECT_EQ(graph->tensor(reads[0]).segment, 1);
    EXPECT_EQ(graph->tensor(phases[0]).segment, ir::kJoinedSegment);
    EXPECT_EQ(graph->node(graph->producer(reads[0])).component, kFrameworkComponent);
    for (const auto output : outputs) {
      const auto& node = graph->node(graph->producer(output));
      EXPECT_EQ(node.component, context.component);
      EXPECT_EQ(node.batch, context.batch);
      EXPECT_EQ(graph->tensor(output).segment, context.segment);
    }
  }
}

TEST(ComponentsTest, AheadAndExactMasksAreIndependentForFrozenAndOwnedCells) {
  ir::Builder b;
  StepContext ctx(b, {{.slot = 0, .rows = 3, .past = 7, .read_cells = 16}}, false);
  std::vector<StepContext::MaskKey> declarations;
  for (bool frozen : {false, true}) {
    for (bool ahead : {false, true}) {
      for (bool exact : {false, true}) {
        declarations.push_back({.cells = {.capacity = 32, .frozen = frozen},
                                .window = 4,
                                .ahead = ahead,
                                .exact = exact});
      }
    }
  }
  ctx.DeclareMasks(declarations);
  std::vector<ir::TensorId> masks;
  for (const auto& key : declarations) {
    const auto mask = ctx.Mask(0, key.cells, key.window, key.ahead, key.exact);
    ASSERT_TRUE(mask.has_value()) << mask.error();
    EXPECT_EQ(b.tensor(*mask).ne(1), key.exact ? 3U : kMaskRowAlign);
    masks.push_back(*mask);
  }
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;
  std::ranges::sort(masks);
  EXPECT_EQ(std::ranges::unique(masks).begin(), masks.end());
  for (std::size_t i = 0; i < declarations.size(); ++i) {
    const auto mask =
        ctx.Mask(0, declarations[i].cells, 4, declarations[i].ahead, declarations[i].exact);
    const auto params = g->node(g->producer(*mask)).Params<ir::MaskParams>();
    EXPECT_EQ(params.kind, declarations[i].ahead ? ir::MaskKind::kAhead : ir::MaskKind::kWindow);
    EXPECT_EQ(params.exact_rows, declarations[i].exact);
  }
}

TEST(ComponentsTest, DeclaredCommitRootsBoundEachOwnerBeforeSaveMetadataGrows) {
  spec::ModelSpec model;
  model.stages.emplace_back();
  model.stages.front().layers.emplace_back();
  auto& refs = model.stages.front().layers.front().components;
  refs.push_back({.kind = "attention.gdn"});
  refs.push_back({.kind = "residual.ple"});
  auto bound = CommitSaveBound(model, 4);
  ASSERT_TRUE(bound);
  EXPECT_EQ(*bound, 20U);
  EXPECT_FALSE(CommitSaveBound(model, 0));
  EXPECT_FALSE(CommitSaveBound(model, UINT32_MAX));
  refs.push_back({.kind = "absent.component"});
  EXPECT_FALSE(CommitSaveBound(model, 4));

  ir::Builder b;
  StepContext ctx(b, {{.slot = 0, .rows = 1}}, false);
  const Instance saver{.def = Find("residual.ple", 1), .params = {}, .stage = {}, .index = 2};
  const auto value = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 1})));
  ctx.Save(saver, "one", value);
  EXPECT_FALSE(b.error());
  ctx.Save(saver, "unpaid", value);
  ASSERT_TRUE(b.error());
  EXPECT_EQ(b.error()->error, ir::GraphError::kBounds);
  EXPECT_EQ(ctx.TakeSaves().size(), 1U);
}

TEST(ComponentsTest, SeparateOwnerSavesShareOneComponentDeclaration) {
  ir::Builder b;
  StepContext ctx(b, {{.slot = 0, .rows = 1}, {.slot = 1, .first_row = 1, .rows = 1}}, false);
  const Instance first{.def = Find("residual.ple", 1), .params = {}, .stage = {}, .index = 2};
  const Instance second{.def = Find("residual.ple", 1), .params = {}, .stage = {}, .index = 3};
  const auto value = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 1})));
  ctx.Save(first, "owner0", value, 0);
  ctx.Save(first, "owner1", value, 1);
  ctx.Save(second, "other-component", value, 0);
  ASSERT_FALSE(b.error());
  ctx.Save(first, "third-root", value, 0);
  ASSERT_TRUE(b.error());
  EXPECT_EQ(ctx.TakeSaves().size(), 3U);
}

TEST(ComponentsTest, SaveIndicesStayLiveUntilTheLoweringsStores) {
  ir::Builder b;
  StepContext ctx(b, {{.slot = 0, .rows = 3}, {.slot = 1, .first_row = 3, .rows = 2}}, false);
  const Instance saver{
      .def = Find("residual.ple", 1), .params = {}, .stage = {}, .layer = 1, .index = 2};
  const auto joined = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 5})));
  const auto own = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 3})));
  ctx.Save(saver, "joined", joined);
  ctx.Save(saver, "own", own, 0);
  const auto saves = ctx.TakeSaves();
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;
  ASSERT_EQ(saves.size(), 2U);
  for (const auto& save : saves) EXPECT_TRUE(std::ranges::contains(g->keep(), save.rows));
}

TEST(ComponentsTest, HeadPrefixesRefuseRowsPastTheirWeightAndMarkovPrefixes) {
  auto head = MakeInstance(C("head.linear", {P("rows", I(41))}), kFacts, Site::kOutput, 0, 4);
  ASSERT_TRUE(head.has_value());
  const std::vector<format::TensorFormat> formats{Matrix(Encoding::kF32, kWidth, 40), Vector(1),
                                                  Vector(1), Vector(1), Vector(1)};
  EXPECT_FALSE(BuildComponent(*head, {.segments = {Seg(0, 3, 0, 256, 1)}}, formats, {}));
  EXPECT_FALSE(MakeInstance(C("head.linear", {P("gather", ParamValue(false)),
                                              P("markov", ParamValue(true)), P("rows", I(20))}),
                            kFacts, Site::kOutput, 0, 4));
}

TEST(ComponentsTest, BuildStepMakesTheGemmaStepsFromTheirSpecs) {
  struct Model {
    spec::ModelSpec spec;
    spec::Source source;
    std::uint32_t layers;
    std::uint32_t resources;
  };
  for (const Model& m : {Model{Gemma2Spec(), test_support::Gemma2Source(), 26, 288},
                         Model{Gemma3Spec(), test_support::Gemma3Source(), 34, 444}}) {
    SCOPED_TRACE(m.source.architecture);
    const auto layout = StageLayoutUniform(m.spec, 0, 8192, 16, kExtent);
    ASSERT_TRUE(layout.has_value());
    const Bindings bindings{.tensors = m.source.tensors, .state = &*layout};
    // Two owners (3 rows after 4351 positions, 1 row after 7).
    const planner::StepShape two{.segments = {Seg(3, 3, 4351, 4608, 1), Seg(1, 1, 7, 256, 1)}};
    auto step = BuildStep(m.spec, 0, two, bindings);
    ASSERT_TRUE(step.has_value()) << step.error();
    auto indexed_policy = BuildStep(m.spec, 0, two, bindings, {.rounded_index_mask = true});
    ASSERT_TRUE(indexed_policy) << indexed_policy.error();
    EXPECT_EQ(indexed_policy->graph.Digest(), step->graph.Digest());
    const ir::Graph& g = step->graph;
    EXPECT_EQ(Count(g, ir::Op::kAttention), m.layers * 2);
    EXPECT_EQ(Count(g, ir::Op::kCacheStore), m.layers * 4);
    EXPECT_EQ(Count(g, ir::Op::kMaskBuild), 4U);  // causal and window, per owner
    EXPECT_EQ(Count(g, ir::Op::kGlu), m.layers);
    EXPECT_EQ(step->weights.size(), m.resources);  // the tied head reads the embedding's leaf
    EXPECT_EQ(step->state.size(), std::size_t{m.layers} * 2 * 2);
    ASSERT_TRUE(step->logits.valid());
    EXPECT_EQ(g.tensor(step->logits).ne(1), 2U);
    EXPECT_FALSE(step->greedy.valid());
    // State-only: the last layer stores its K and V and nothing reads them.
    const planner::StepShape cut{.segments = {Seg(0, 16, 0, 256, 0)}};
    auto state_only = BuildStep(m.spec, 0, cut, bindings);
    ASSERT_TRUE(state_only.has_value()) << state_only.error();
    EXPECT_EQ(Count(state_only->graph, ir::Op::kAttention), m.layers - 1);
    EXPECT_EQ(Count(state_only->graph, ir::Op::kCacheStore), m.layers * 2);
    EXPECT_EQ(Count(state_only->graph, ir::Op::kGlu), m.layers - 1);
    EXPECT_FALSE(state_only->logits.valid());
    // Greedy owners.
    const planner::StepShape greedy{
        .segments = {Seg(0, 1, 9, 256, 1, true), Seg(1, 1, 3, 256, 1, true)}};
    auto picked = BuildStep(m.spec, 0, greedy, bindings);
    ASSERT_TRUE(picked.has_value()) << picked.error();
    ASSERT_TRUE(picked->greedy.valid());
    EXPECT_EQ(picked->graph.tensor(picked->greedy).ne(0), 2U);
    // Mixed head consumers share the same logits; only the owner asking for
    // device tokens reads Argmax. Lowering must preserve the individual ports.
    auto mixed = BuildStep(
        m.spec, 0, {.segments = {Seg(0, 1, 9, 256, 1, true), Seg(1, 1, 3, 256, 1)}}, bindings);
    ASSERT_TRUE(mixed) << mixed.error();
    ASSERT_TRUE(mixed->logits.valid());
    ASSERT_TRUE(mixed->greedy.valid());
    EXPECT_EQ(mixed->graph.tensor(mixed->logits).ne(1), 2U);
    EXPECT_EQ(mixed->graph.tensor(mixed->greedy).ne(0), 2U);
    // Refusals: reads that miss the rows or a stage mismatch.
    EXPECT_FALSE(BuildStep(m.spec, 0, {.segments = {Seg(0, 4, 300, 256, 1)}}, bindings));
    EXPECT_FALSE(BuildStep(m.spec, 0, {.stage = 1, .segments = {Seg(0, 1, 0, 256, 1)}}, bindings));
    EXPECT_FALSE(BuildStep(m.spec, 0, two, {.tensors = m.source.tensors, .state = nullptr}));
  }
}

// A Gemma family's step inputs for owners at their pasts, against the
// closed forms: positions, the cells a store writes, and each host mask.
struct GemmaOwner {
  std::uint8_t slot;
  std::uint32_t rows, past;
};
void ExpectGemmaCellsAndHostMasks(const spec::ModelSpec& spec, const spec::Source& source,
                                  std::uint32_t window, std::uint32_t context,
                                  std::uint32_t max_rows, std::span<const GemmaOwner> owners) {
  const auto layout = StageLayoutUniform(spec, 0, context, max_rows, kExtent);
  ASSERT_TRUE(layout.has_value());
  const auto global_cells = static_cast<std::uint32_t>(PadUp(context, 256));
  const auto local_cells =
      static_cast<std::uint32_t>(PadUp(std::min<std::uint64_t>(context, window + max_rows), 256));
  const auto read_cells = [](std::uint32_t end, std::uint32_t cells) {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(cells, PadUp(end, 256)));
  };
  std::size_t total = 0;
  for (const GemmaOwner& o : owners) total += o.rows;
  std::vector<std::int32_t> tokens(total, 5);
  planner::StepShape shape;
  std::vector<std::int32_t> positions;
  for (const GemmaOwner& o : owners) {
    const std::uint32_t end = o.past + o.rows;
    shape.segments.push_back(
        Seg(o.slot, o.rows, o.past,
            std::max(read_cells(end, global_cells), read_cells(end, local_cells)), 1));
    for (std::uint32_t i = 0; i < o.rows; ++i) {
      positions.push_back(static_cast<std::int32_t>(o.past + i));
    }
  }
  auto step = BuildStep(spec, 0, shape, {.tensors = source.tensors, .state = &*layout});
  ASSERT_TRUE(step.has_value()) << step.error();
  std::size_t checked = 0;
  for (const StepInput& input : step->inputs) {
    auto values = InputValues(input, shape, tokens);
    ASSERT_TRUE(values.has_value()) << values.error();
    if (input.kind == InputKind::kPositions) {
      EXPECT_TRUE(std::ranges::equal(*values, positions));
      ++checked;
    }
    if (input.kind == InputKind::kStoreCells) {
      // A global cache stores a position in its own cell, a ring in the
      // position modulo its cells.
      const GemmaOwner& o = owners[input.segment];
      const bool ring = input.cells.ring;
      ASSERT_EQ(values->size(), o.rows);
      for (std::uint32_t i = 0; i < o.rows; ++i) {
        const std::int64_t pos = o.past + i;
        EXPECT_EQ((*values)[i], ring ? pos % local_cells : pos) << i;
      }
      ++checked;
    }
    if (input.kind == InputKind::kCellPositions) {
      // The host mask, row by row, from the cells' positions and the mask's
      // definition (ir/op.h kMaskBuild), against its closed form: a global
      // cell is visible when it holds a position up to the row's, a ring's
      // when the latest position it holds after the segment is within the
      // window of the row's.
      const GemmaOwner& o = owners[input.segment];
      const bool ring = input.cells.ring;
      const std::uint64_t end = o.past + o.rows;
      const std::uint64_t width =
          read_cells(static_cast<std::uint32_t>(end), ring ? local_cells : global_cells);
      ASSERT_EQ(values->size(), width);
      for (std::uint32_t i = 0; i < o.rows; ++i) {
        const std::int64_t pos = o.past + i;
        for (std::uint64_t c = 0; c < width; ++c) {
          const std::int64_t held = (*values)[c];
          bool seen = held >= 0 && held <= pos;
          if (ring) seen = seen && pos - held < window;
          bool want = c <= static_cast<std::uint64_t>(pos);
          if (ring) {
            const std::uint64_t latest =
                c + (c < end ? ((end - 1 - c) / local_cells) * local_cells : 0);
            want = c < end && static_cast<std::int64_t>(latest) <= pos &&
                   pos - static_cast<std::int64_t>(latest) < window;
          }
          ASSERT_EQ(seen, want) << i << " " << c;
        }
      }
      ++checked;
    }
  }
  EXPECT_EQ(checked, 1U + 2U * owners.size() + 2U * owners.size());
}

TEST(ComponentsTest, InputValuesGiveTheGemmaCellsAndHostMasks) {
  // Two owners, one wrapping its ring (4608 local cells).
  const std::array<GemmaOwner, 2> owners{GemmaOwner{2, 400, 4500}, GemmaOwner{0, 3, 9}};
  ExpectGemmaCellsAndHostMasks(Gemma2Spec(), test_support::Gemma2Source(), fam::Gemma2_2B().window,
                               8192, 512, owners);
}

// Gemma 4's, with its 1,024-cell window: the owner at 1,500 wraps a ring of
// 1,536 cells (pad(1,024 + 512, 256)) with 400 rows.
TEST(ComponentsTest, InputValuesGiveGemma4CellsAndHostMasks) {
  const std::array<GemmaOwner, 2> owners{GemmaOwner{2, 400, 1500}, GemmaOwner{0, 3, 9}};
  for (const std::uint32_t size : {26U, 31U}) {
    SCOPED_TRACE(size);
    ExpectGemmaCellsAndHostMasks(Gemma4Spec(size), test_support::Gemma4Source(size),
                                 (size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B()).window,
                                 8192, 512, owners);
  }
}

// ------------------------------------------------------------------ Gemma 4

format::TensorFormat Matrix3(Encoding e, std::uint64_t k, std::uint64_t n, std::uint64_t m) {
  format::TensorFormat f = ir::Dense(e, ir::ShapeOf({k, n, m}));
  if (format::Info(e).quantized) f.layout = format::Layout::kGgmlBlocks;
  return f;
}

ComponentRef Gemma4Attention(std::string kind, std::int64_t window, bool tie, bool freqs) {
  std::vector<Param> p{P("placement", N("sandwich")),
                       P("eps", R(1e-6)),
                       P("heads", I(4)),
                       P("kv_heads", I(2)),
                       P("key_dim", I(32)),
                       P("value_dim", I(32)),
                       P("scale", R(1.0)),
                       P("window", I(window)),
                       P("rotation", N("neox")),
                       P("rope_dims", I(32)),
                       P("rope_base", R(window == 0 ? 1e6 : 1e4)),
                       P("rope_scale", R(1.0)),
                       P("qk_norm", ParamValue(true)),
                       P("rope_freqs", ParamValue(freqs))};
  if (kind == "attention.dense") {
    p.push_back(P("kv_tie", ParamValue(tie)));
    p.push_back(P("v_norm", ParamValue(true)));
  } else {
    p.push_back(P("target_layer", I(3)));
  }
  return C(std::move(kind), std::move(p));
}

std::size_t CountIn(const ir::Graph& g, ir::Op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(g.nodes(), [op](const ir::Node& n) { return n.key.op == op; }));
}

// Gemma 4's attention: V tied to K's product and RMS-normed, queries
// unscaled, a global layer's proportional rotation (each pair's angle
// divided by its factor, 1 to 4 here), a local layer's ring.
TEST(ComponentsTest, Gemma4AttentionVariantsMatchTheirReference) {
  constexpr std::uint16_t kRopeFreqs = 8;
  for (const auto& [window, tie, freqs] :
       {std::tuple{std::int64_t{8}, false, false}, std::tuple{std::int64_t{0}, true, true},
        std::tuple{std::int64_t{0}, false, true}}) {
    SCOPED_TRACE(std::format("window {} tie {} freqs {}", window, tie, freqs));
    auto in = MakeInstance(Gemma4Attention("attention.dense", window, tie, freqs), kFacts,
                           Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value()) << in.error();
    const std::uint64_t cells = window != 0 ? 20 : 32;
    const std::array<std::uint64_t, 2> both{cells, cells};
    Compare(*in, AttentionFormats(Encoding::kQ8_0), both, AttentionSteps(32), kOutputTolerance,
            kF16StateTolerance, {{kRopeFreqs, {1.0F, 1.5F}}});
    // A tied layer has no V product, and no query scale node at scale 1.
    auto g =
        BuildComponent(*in, AttentionSteps(32)[0].shape, AttentionFormats(Encoding::kQ8_0), both);
    ASSERT_TRUE(g.has_value()) << g.error();
    EXPECT_EQ(CountIn(g->step.graph, ir::Op::kMatMul), tie ? 3U : 4U);
    EXPECT_EQ(CountIn(g->step.graph, ir::Op::kScale), 0U);
    EXPECT_EQ(g->weights[3].valid(), !tie);  // the V role
  }
  // A tied V of another width than K's is no instance.
  ComponentRef bad = Gemma4Attention("attention.dense", 0, true, false);
  for (Param& p : bad.params) {
    if (p.name == "value_dim") p.value = I(16);
  }
  EXPECT_FALSE(MakeInstance(bad, kFacts, Site::kLayer, 0, 1));
}

// The assistant's attention: queries over its target's cache, read frozen
// (only positions before each row's; nothing stored), a ring and a
// per-position cache, owners at their own positions.
TEST(ComponentsTest, BorrowedAttentionReadsItsTargetsCacheFrozen) {
  constexpr std::uint16_t kRopeFreqs = 8;
  for (const auto& [window, freqs] :
       {std::pair{std::int64_t{8}, false}, std::pair{std::int64_t{0}, true}}) {
    SCOPED_TRACE(std::format("window {}", window));
    auto in = MakeInstance(Gemma4Attention("attention.borrowed", window, false, freqs), kFacts,
                           Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value()) << in.error();
    const auto contract = in->def->state(*in);
    ASSERT_EQ(contract.tensors.size(), 2U);
    for (const auto& t : contract.tensors) {
      EXPECT_TRUE(t.borrowed_from_target);
      EXPECT_EQ(t.target_layer, 3U);
    }
    const std::uint64_t cells = window != 0 ? 20 : 64;
    const std::array<std::uint64_t, 2> both{cells, cells};
    // One query an owner, each at its own position; a ring read whole and
    // wrapped (past 30 in 20 cells).
    const std::vector<Step> steps{
        {.shape = {.segments = {Seg(0, 1, 5, 32, 1), Seg(2, 1, 17, 32, 1)}}},
        {.shape = {.segments = {Seg(2, 1, 30, 64, 1)}}},
        {.shape = {.segments = {Seg(0, 1, 9, 32, 1)}}},
    };
    Compare(*in, AttentionFormats(Encoding::kQ8_0), both, steps, kOutputTolerance,
            kF16StateTolerance, {{kRopeFreqs, {1.0F, 1.5F}}}, /*random_state=*/true);
    auto g = BuildComponent(*in, steps[0].shape, AttentionFormats(Encoding::kQ8_0), both);
    ASSERT_TRUE(g.has_value()) << g.error();
    EXPECT_EQ(CountIn(g->step.graph, ir::Op::kCacheStore), 0U);
    EXPECT_EQ(CountIn(g->step.graph, ir::Op::kAttention), 2U);
    // Its masks see the positions before each row's: the frozen cells.
    std::size_t frozen = 0;
    for (const StepInput& input : g->step.inputs) {
      if (input.kind != InputKind::kCellPositions) continue;
      EXPECT_TRUE(input.cells.frozen);
      auto held = InputValues(input, steps[0].shape, {});
      ASSERT_TRUE(held.has_value());
      const std::uint32_t past = steps[0].shape.segments[input.segment].past;
      for (std::size_t c = 0; c < held->size(); ++c) {
        EXPECT_EQ((*held)[c] >= 0, c < past) << c;
      }
      ++frozen;
    }
    EXPECT_EQ(frozen, 2U);
  }
}

TEST(ComponentsTest, Gemma4BlocksMatchTheirReferences) {
  const std::vector<Step> rows{
      {.shape = {.segments = {Seg(0, 3, 0, 256, 3), Seg(1, 2, 7, 256, 2)}},
       .tokens = {4, 0, 19, 7, 3}},
  };
  auto scale = MakeInstance(C("scale.learned", {}), kFacts, Site::kLayer, 0, 2);
  ASSERT_TRUE(scale.has_value()) << scale.error();
  // A one-value role of rank 1, as an artifact describes it.
  format::TensorFormat one = Vector(1);
  one.shape.rank = 1;
  const std::vector<format::TensorFormat> sf{one};
  Compare(*scale, sf, {}, rows, 1e-6);
  // The assistant's input: its target's table rows (48 wide) beside a
  // feature, projected to the stream; its tap (target_writes) untouched.
  auto projected = MakeInstance(
      C("embedding.projected",
        {P("scale", R(std::sqrt(48.0))), P("feature", I(48)), P("tap", N("features"))}),
      kFacts, Site::kEmbedding, 0, 0);
  ASSERT_TRUE(projected.has_value()) << projected.error();
  const auto tap = projected->def->state(*projected);
  ASSERT_EQ(tap.tensors.size(), 1U);
  EXPECT_TRUE(tap.tensors[0].target_writes);
  EXPECT_EQ(tap.tensors[0].cells, model::CellRule::kPositionRing);
  EXPECT_EQ(tap.tensors[0].fixed_cells, 16U);
  EXPECT_EQ(tap.tensors[0].cell_bytes, 48U * 4);
  const std::vector<format::TensorFormat> pf{Matrix(Encoding::kQ8_0, 64, 20),
                                             Matrix(Encoding::kQ8_0, 96, kWidth)};
  // The table's rows are 64 wide in its format but 48 in its role: refused.
  const std::array<std::uint64_t, 1> tap_cells{16};
  EXPECT_FALSE(BuildComponent(*projected, rows[0].shape, pf, tap_cells));
  const std::vector<format::TensorFormat> pf48{Matrix(Encoding::kF16, 48, 20),
                                               Matrix(Encoding::kQ8_0, 96, kWidth)};
  Compare(*projected, pf48, tap_cells, rows, 1e-5);
  // A drafter's head: every row an output, and its rows projected to 48.
  auto head = MakeInstance(C("head.linear", {P("gather", ParamValue(false)), P("features", I(48))}),
                           kFacts, Site::kOutput, 0, 4);
  ASSERT_TRUE(head.has_value()) << head.error();
  const std::vector<format::TensorFormat> hf{Matrix(Encoding::kQ8_0, kWidth, 40),
                                             Matrix(Encoding::kQ8_0, kWidth, 48), Vector(1)};
  Compare(*head, hf, {}, rows, 1e-5);
  // Not every row an output: refused.
  EXPECT_FALSE(BuildComponent(*head, {.segments = {Seg(0, 3, 0, 256, 1)}}, hf, {}));
}

// ffn.moe's roles, in order (moe.cc's).
std::vector<format::TensorFormat> MoeFormats(Encoding e, bool stacked) {
  constexpr std::uint64_t kHidden = 64;
  constexpr std::uint64_t kExperts = 8;
  constexpr std::uint64_t kExpertHidden = 32;
  return {Vector(kWidth), Matrix(e, kWidth, kHidden), Matrix(e, kWidth, kHidden),
          Matrix(e, kHidden, kWidth), Vector(kWidth), Matrix(Encoding::kF32, kWidth, kExperts),
          Vector(kWidth), Vector(kWidth),
          Matrix3(e, kWidth, stacked ? 2 * kExpertHidden : kExpertHidden, kExperts),
          Matrix3(e, kWidth, kExpertHidden, kExperts), Matrix3(e, kWidth, kExpertHidden, kExperts),
          Matrix3(e, kExpertHidden, kWidth, kExperts), Vector(kExperts), Vector(kWidth),
          Vector(kWidth),
          // Qwen3.8's roles, unused here: each product's expert scales,
          // the shared expert's gate.
          Vector(kExperts), Vector(kExperts), Vector(kExperts), Vector(kWidth)};
}

ComponentRef Moe(bool stacked, bool scales, bool normalize) {
  return C("ffn.moe", {P("placement", N("parallel")), P("eps", R(1e-6)),
                       P("activation", N("gelu_tanh")), P("hidden", I(64)), P("experts", I(8)),
                       P("used", I(3)), P("expert_hidden", I(32)), P("router", N("softmax_topk")),
                       P("router_input", N("rms_scaled")), P("normalize", ParamValue(normalize)),
                       P("expert_scales", ParamValue(scales)), P("stacked", ParamValue(stacked))});
}

// The routed experts beside a dense GLU, its route and reduce shapes as
// parameters: stacked or separate gate-up arrays, with or without expert
// scales and renormalized weights.
TEST(ComponentsTest, MoeFragmentMatchesItsReference) {
  constexpr std::uint16_t kExpertScale = 12;
  const std::vector<Step> rows{
      {.shape = {.segments = {Seg(0, 3, 0, 256, 1), Seg(1, 2, 7, 256, 2)}}},
      {.shape = {.segments = {Seg(0, 1, 3, 256, 1)}}},
  };
  for (const auto& [stacked, scales, normalize, e] :
       {std::tuple{true, true, true, Encoding::kQ8_0},
        std::tuple{false, false, false, Encoding::kF32}}) {
    SCOPED_TRACE(std::format("stacked {} scales {} normalize {}", stacked, scales, normalize));
    auto in = MakeInstance(Moe(stacked, scales, normalize), kFacts, Site::kLayer, 0, 2);
    ASSERT_TRUE(in.has_value()) << in.error();
    Compare(*in, MoeFormats(e, stacked), {}, rows, 1e-4, 0.0, {{kExpertScale, {0.5F, 0.5F}}});
    auto g = BuildComponent(*in, rows[0].shape, MoeFormats(e, stacked), {});
    ASSERT_TRUE(g.has_value()) << g.error();
    const ir::Graph& graph = g->step.graph;
    EXPECT_EQ(CountIn(graph, ir::Op::kRoute), 1U);
    EXPECT_EQ(CountIn(graph, ir::Op::kExpertReduce), 1U);
    EXPECT_EQ(CountIn(graph, ir::Op::kMatMulId), stacked ? 2U : 3U);
    EXPECT_EQ(CountIn(graph, ir::Op::kGlu), 2U);
  }
  {
    ComponentRef sigmoid_moe = Moe(true, true, true);
    for (Param& p : sigmoid_moe.params) {
      if (p.name == "router") p.value = N("sigmoid_topk");
    }
    auto in = MakeInstance(sigmoid_moe, kFacts, Site::kLayer, 0, 2);
    ASSERT_TRUE(in.has_value()) << in.error();
    Compare(*in, MoeFormats(Encoding::kF32, true), {}, rows, 1e-4, 0.0,
            {{kExpertScale, {0.5F, 0.5F}}});
    auto g = BuildComponent(*in, rows[0].shape, MoeFormats(Encoding::kF32, true), {});
    ASSERT_TRUE(g.has_value()) << g.error();
    const ir::Graph& graph = g->step.graph;
    EXPECT_EQ(CountIn(graph, ir::Op::kRoute), 1U);
  }
  // More experts used than there are: no instance.
  ComponentRef bad = Moe(true, true, true);
  for (Param& p : bad.params) {
    if (p.name == "used") p.value = I(9);
  }
  EXPECT_FALSE(MakeInstance(bad, kFacts, Site::kLayer, 0, 2));
  // Its split: experts whole per device (expert parallelism).
  const ComponentDef& def = *Find("ffn.moe", 1);
  ASSERT_EQ(def.splits.size(), 1U);
  EXPECT_EQ(def.splits[0].kind, SplitKind::kExperts);
  auto in = MakeInstance(Moe(true, true, true), kFacts, Site::kLayer, 0, 2);
  ASSERT_TRUE(in.has_value());
  EXPECT_EQ(def.splits[0].units(*in), 8U);
}

TEST(ComponentsTest, ParallelClampedMoeMatchesItsReferenceWithDistinctLimits) {
  auto component = Moe(false, false, true);
  for (auto& param : component.params)
    if (param.name == "activation") param.value = N("silu_clamp");
  component.params.push_back(P("limit", R(0.2)));
  component.params.push_back(P("shared_limit", R(0.05)));
  auto in = MakeInstance(component, kFacts, Site::kLayer, 0, 2);
  ASSERT_TRUE(in) << in.error();
  const std::vector<Step> rows{
      {.shape = {.segments = {Seg(0, 3, 0, 256, 1), Seg(1, 2, 7, 256, 2)}}},
      {.shape = {.segments = {Seg(0, 1, 3, 256, 1)}}},
  };
  Compare(*in, MoeFormats(Encoding::kF32, false), {}, rows, 1e-4);
  auto graph = BuildComponent(*in, rows[0].shape, MoeFormats(Encoding::kF32, false), {});
  ASSERT_TRUE(graph) << graph.error();
  std::vector<float> limits;
  for (const auto& node : graph->step.graph.nodes())
    if (node.key.op == ir::Op::kGlu) limits.push_back(node.Params<ir::GluParams>().limit);
  ASSERT_EQ(limits.size(), 2U);
  EXPECT_FLOAT_EQ(limits[0], 0.2F);
  EXPECT_FLOAT_EQ(limits[1], 0.05F);
}

ComponentRef MoePre(bool stacked, bool scales, bool normalize) {
  return C("ffn.moe", {P("placement", N("pre")), P("eps", R(1e-6)), P("activation", N("gelu_tanh")),
                       P("hidden", I(0)), P("experts", I(8)), P("used", I(3)),
                       P("expert_hidden", I(32)), P("router", N("softmax_topk")),
                       P("router_input", N("rms_scaled")), P("normalize", ParamValue(normalize)),
                       P("expert_scales", ParamValue(scales)), P("stacked", ParamValue(stacked))});
}

TEST(ComponentsTest, MoePreFragmentMatchesItsReference) {
  constexpr std::uint16_t kExpertScale = 12;
  const std::vector<Step> rows{
      {.shape = {.segments = {Seg(0, 3, 0, 256, 1), Seg(1, 2, 7, 256, 2)}}},
      {.shape = {.segments = {Seg(0, 1, 3, 256, 1)}}},
  };
  auto in = MakeInstance(MoePre(true, true, true), kFacts, Site::kLayer, 0, 2);
  ASSERT_TRUE(in.has_value()) << in.error();
  Compare(*in, MoeFormats(Encoding::kF32, true), {}, rows, 1e-4, 0.0,
          {{kExpertScale, {0.5F, 0.5F}}});
  auto g = BuildComponent(*in, rows[0].shape, MoeFormats(Encoding::kF32, true), {});
  ASSERT_TRUE(g.has_value()) << g.error();
  EXPECT_EQ(CountIn(g->step.graph, ir::Op::kAdd), 1U);
}

// Each piece alone, through the reference executor against its own
// reference: the router, the routed experts with their reduce, the GLU.
TEST(ComponentsTest, MoePiecesMatchTheirReferences) {
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  constexpr std::uint64_t kRows = 5;
  constexpr std::uint64_t kHidden = 24;
  constexpr std::uint32_t kExperts = 6;
  const auto random = [&](std::size_t n, float lo, float span) {
    std::vector<float> v(n);
    for (float& x : v) x = lo + (span * u(rng));
    return v;
  };
  const auto f32 = [](std::initializer_list<std::uint64_t> ne) {
    return ir::Dense(Encoding::kF32, ir::ShapeOf(ne));
  };
  const std::vector<float> x = random(kWidth * kRows, 0.0F, 2.0F);
  const std::vector<double> xd(x.begin(), x.end());
  const std::vector<float> router = random(kWidth * kExperts, 0.0F, 0.5F);
  const std::vector<float> rscale = random(kWidth, 1.0F, 0.25F);
  const std::vector<float> gate_up = random(kWidth * 2 * kHidden * kExperts, 0.0F, 0.3F);
  const std::vector<float> down = random(kHidden * kWidth * kExperts, 0.0F, 0.3F);
  const std::vector<float> escale = random(kExperts, 1.0F, 0.5F);
  const moe::RouterParams rp{.experts = kExperts, .used = 2, .normalize = true, .eps = 1e-6F};
  const moe::ExpertParams ep{
      .width = kWidth, .hidden = kHidden, .stacked = true, .kind = ir::GluKind::kGeGluTanh};
  ir::Builder b;
  const ir::TensorId in = b.Input(f32({kWidth, kRows}), "x");
  const auto weight = [&](const format::TensorFormat& f, std::uint16_t role) {
    return b.Weight(f, {.stage = 0, .component = 1, .role = role, .layer = 0}, "w");
  };
  const ir::TensorId wr = weight(f32({kWidth, kExperts}), 0);
  const ir::TensorId ws = weight(f32({kWidth}), 1);
  const ir::TensorId wgu = weight(f32({kWidth, 2 * kHidden, kExperts}), 2);
  const ir::TensorId wd = weight(f32({kHidden, kWidth, kExperts}), 3);
  const ir::TensorId we = weight(f32({kExperts}), 4);
  const ir::TensorId wg = weight(f32({kWidth, kHidden}), 5);
  const ir::TensorId wu = weight(f32({kWidth, kHidden}), 6);
  const ir::TensorId wdd = weight(f32({kHidden, kWidth}), 7);
  const moe::Routed routed = moe::EmitRouter(b, in, wr, ws, rp);
  const ir::TensorId experts = moe::EmitExperts(b, in, routed, wgu, {}, wd, we, ep);
  const ir::TensorId glu = moe::EmitGlu(b, in, wg, wu, wdd, ir::GluKind::kSwiGlu);
  for (const ir::TensorId t : {routed.ids, routed.weights, experts, glu}) b.Output(t);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph.has_value()) << graph.error().detail;
  auto exec = ref::Executor::Create(*graph);
  ASSERT_TRUE(exec.has_value()) << exec.error().detail;
  const std::vector<float> g = random(kWidth * kHidden, 0.0F, 0.3F);
  const std::vector<float> up = random(kWidth * kHidden, 0.0F, 0.3F);
  const std::vector<float> dd = random(kHidden * kWidth, 0.0F, 0.3F);
  for (const auto& [t, v] :
       std::vector<std::pair<ir::TensorId, const std::vector<float>*>>{{in, &x},
                                                                       {wr, &router},
                                                                       {ws, &rscale},
                                                                       {wgu, &gate_up},
                                                                       {wd, &down},
                                                                       {we, &escale},
                                                                       {wg, &g},
                                                                       {wu, &up},
                                                                       {wdd, &dd}}) {
    ASSERT_TRUE(exec->BindValues(t, *v));
  }
  ASSERT_TRUE(exec->Run());
  const moe::RoutedRef want = moe::RouterReference(xd, kWidth, router, rscale, rp);
  auto ids = exec->Ints(routed.ids);
  auto weights = exec->Values(routed.weights);
  ASSERT_TRUE(ids.has_value() && weights.has_value());
  ASSERT_EQ(ids->size(), want.ids.size());
  for (std::size_t i = 0; i < ids->size(); ++i)
    EXPECT_EQ((*ids)[i], static_cast<std::int64_t>(want.ids[i])) << i;
  EXPECT_LE(Gap(*weights, want.weights), 1e-6);
  auto got = exec->Values(experts);
  ASSERT_TRUE(got.has_value());
  EXPECT_LE(Gap(*got, moe::ExpertsReference(xd, want, gate_up, {}, down, escale, ep)), 1e-5);
  auto dense = exec->Values(glu);
  ASSERT_TRUE(dense.has_value());
  EXPECT_LE(Gap(*dense, moe::GluReference(xd, g, up, dd, kWidth, kHidden, ir::GluKind::kSwiGlu)),
            1e-5);
}

spec::ModelSpec Gemma4Spec(std::uint32_t size) {
  const auto& p = size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B();
  auto values = fam::Gemma4Values(p);
  EXPECT_TRUE(values.has_value());
  auto s =
      spec::Synthesize(spec::mappings::Gemma4Mapping(), test_support::Gemma4Source(size), *values);
  EXPECT_TRUE(s.has_value()) << s.error();
  return *s;
}

spec::ModelSpec Gemma4WithAssistant(std::uint32_t size) {
  const auto& p = size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B();
  const auto& a = size == 26 ? fam::Gemma4Assistant26() : fam::Gemma4Assistant31();
  auto values = fam::Gemma4AssistantValues(a, p);
  EXPECT_TRUE(values.has_value()) << values.error();
  auto s = spec::ComposeDrafter(Gemma4Spec(size), spec::mappings::Gemma4Mapping(),
                                spec::mappings::Gemma4AssistantMapping(),
                                test_support::Gemma4AssistantSource(size), *values);
  EXPECT_TRUE(s.has_value()) << s.error();
  return *s;
}

TEST(ComponentsTest, Gemma4SpecsValidateAndLayTheGemma4State) {
  for (const std::uint32_t size : {26U, 31U}) {
    SCOPED_TRACE(size);
    const auto& p = size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B();
    const spec::ModelSpec s = Gemma4Spec(size);
    const auto source = test_support::Gemma4Source(size);
    const auto valid = ValidateSpec(s, source.tensors);
    ASSERT_TRUE(valid.has_value()) << valid.error();
    for (const auto& [context, rows] : {std::pair{4096U, 16U}, {262144U, 512U}}) {
      // Gemma 4's state in closed form (the M3 layout, retired with its
      // runner): a K and a V tensor per layer at the layer's own head width
      // and KV heads, a global one a cell a position, a local one a ring of
      // pad(min(context, window + rows), 256) cells, each on its own extent.
      const auto global_cells = static_cast<std::uint32_t>(PadUp(context, 256));
      const auto local_cells =
          static_cast<std::uint32_t>(PadUp(std::min<std::uint64_t>(context, p.window + rows), 256));
      const auto got = StageLayoutUniform(s, 0, context, rows, kExtent);
      ASSERT_TRUE(got.has_value()) << got.error();
      ASSERT_EQ(got->items.size(), std::size_t{p.layers} * 2);
      std::uint64_t at = 0;
      for (std::uint32_t il = 0; il < p.layers; ++il) {
        for (std::uint32_t kv = 0; kv < 2; ++kv) {
          const auto& item = got->items[std::size_t{il} * 2 + kv];
          const auto cells = p.local(il) ? local_cells : global_cells;
          const auto bytes = std::uint64_t{p.head_dim(il)} * p.kv_heads(il) * cells * 2;
          EXPECT_EQ(item.layer, il);
          EXPECT_EQ(item.offset, at);
          EXPECT_EQ(item.cells, cells);
          EXPECT_EQ(item.bytes, bytes);
          at += PadUp(bytes, kExtent);
        }
      }
      EXPECT_EQ(got->bytes, at);
    }
    // Every layer's attention, feed-forward block and output scale.
    const auto layout = StageLayoutUniform(s, 0, 4096, 16, kExtent);
    ASSERT_TRUE(layout.has_value());
    const Bindings bindings{.tensors = source.tensors, .state = &*layout};
    const planner::StepShape two{.segments = {Seg(3, 3, 1500, 1792, 1), Seg(1, 1, 7, 256, 1)}};
    auto step = BuildStep(s, 0, two, bindings);
    ASSERT_TRUE(step.has_value()) << step.error();
    const ir::Graph& g = step->graph;
    EXPECT_EQ(CountIn(g, ir::Op::kAttention), p.layers * 2);
    EXPECT_EQ(CountIn(g, ir::Op::kCacheStore), p.layers * 4);
    EXPECT_EQ(CountIn(g, ir::Op::kRoute), p.experts != 0 ? p.layers : 0U);
    EXPECT_EQ(CountIn(g, ir::Op::kGlu), p.layers * (p.experts != 0 ? 2 : 1));
    // The global layers' rotations read the one shared factor table.
    std::size_t factored = 0;
    for (const ir::Node& n : g.nodes()) {
      if (n.key.op == ir::Op::kRope && n.nsrc > 2 && n.src[2].valid()) ++factored;
    }
    EXPECT_EQ(factored, (p.layers / 6) * 2);
    const auto tables = std::ranges::count_if(step->weights, [](const StepWeight& w) {
      return w.parts.front().first == "rope_freqs.weight";
    });
    EXPECT_EQ(tables, 1);
  }
}

// The assistant composed with its target: its reads link to its
// contracts, its step reads the target's cache frozen and its table, and
// the target's step writes the tap it reads.
TEST(ComponentsTest, Gemma4AssistantStageReadsItsTarget) {
  const spec::ModelSpec s = Gemma4WithAssistant(26);
  const auto& p = fam::Gemma4_26BA4B();
  const auto target = test_support::Gemma4Source(26);
  const auto assistant = test_support::Gemma4AssistantSource(26);
  const std::array<std::span<const BoundTensor>, 2> artifacts{target.tensors, assistant.tensors};
  const auto valid = ValidateSpec(s, artifacts);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  ASSERT_EQ(s.stages.size(), 2U);
  EXPECT_EQ(s.stages[1].role, spec::StageRole::kDrafter);
  ASSERT_EQ(s.stages[0].taps.size(), 1U);
  EXPECT_EQ(s.stages[0].taps[0].point, spec::TapPoint::kPostFinalNorm);
  const auto target_layout = StageLayoutUniform(s, 0, 4096, 16, kExtent);
  const auto drafter_layout = StageLayoutUniform(s, 1, 4096, 16, kExtent);
  ASSERT_TRUE(target_layout.has_value() && drafter_layout.has_value());
  // Its own bytes are the tap's ring; its K and V are the target's last
  // local (28) and global (29) layers'.
  ASSERT_EQ(drafter_layout->items.size(), 1U);
  EXPECT_TRUE(drafter_layout->items[0].contract.target_writes);
  ASSERT_EQ(drafter_layout->borrowed.size(), 8U);
  for (const auto& b : drafter_layout->borrowed) {
    const auto* item = model::BorrowedItem(*target_layout, b);
    ASSERT_NE(item, nullptr);
    EXPECT_EQ(item->layer, b.layer == 3 ? 29U : 28U);
  }
  const std::array<const model::StateLayoutDerived*, 2> layouts{&*target_layout, &*drafter_layout};
  // The target's steps narrow after its last attention (its MoE block and
  // output scale are row-local); the drafter's head reads every row.
  const auto target_instances = InstancesOf(s, 0);
  const auto drafter_instances = InstancesOf(s, 1);
  ASSERT_TRUE(target_instances.has_value() && drafter_instances.has_value());
  EXPECT_EQ(NarrowPoint(*target_instances), (std::pair<std::size_t, std::size_t>{p.layers - 1, 0}));
  EXPECT_FALSE(NarrowPoint(*drafter_instances).has_value());
  // A draft step: one row an owner after its target's 300 and 1100
  // positions.
  const planner::StepShape draft{.stage = 1,
                                 .segments = {Seg(0, 1, 300, 512, 1), Seg(4, 1, 1100, 1280, 1)}};
  auto step = BuildStep(s, 1, draft,
                        {.tensors = assistant.tensors,
                         .state = &*drafter_layout,
                         .stages = layouts,
                         .artifacts = artifacts});
  ASSERT_TRUE(step.has_value()) << step.error();
  EXPECT_EQ(CountIn(step->graph, ir::Op::kCacheStore), 0U);
  auto indexed_policy = BuildStep(s, 1, draft,
                                  {.tensors = assistant.tensors,
                                   .state = &*drafter_layout,
                                   .stages = layouts,
                                   .artifacts = artifacts},
                                  {.rounded_index_mask = true});
  ASSERT_TRUE(indexed_policy) << indexed_policy.error();
  EXPECT_EQ(indexed_policy->graph.Digest(), step->graph.Digest());
  EXPECT_EQ(CountIn(step->graph, ir::Op::kAttention), 8U);
  ASSERT_TRUE(step->features.valid());
  EXPECT_EQ(step->graph.tensor(step->features).ne(0), p.width);
  EXPECT_TRUE(std::ranges::all_of(step->state, [](const StepState& st) { return st.borrowed; }));
  EXPECT_EQ(step->state.size(), 16U);
  // Its input rows are the target's table: the target's WeightRef.
  const auto table =
      std::ranges::find_if(step->weights, [](const StepWeight& w) { return w.ref.stage == 0; });
  ASSERT_NE(table, step->weights.end());
  EXPECT_EQ(table->parts.front().first, "token_embd.weight");
  EXPECT_EQ(table->ref.component, 0U);
  EXPECT_EQ(std::ranges::count_if(
                step->inputs, [](const StepInput& in) { return in.kind == InputKind::kFeatures; }),
            1);
  // The masks see the target's positions before each draft row's: today's
  // assistant masks (cells below the prefix; a ring's window).
  for (const StepInput& in : step->inputs) {
    if (in.kind != InputKind::kCellPositions) continue;
    auto held = InputValues(in, draft, {});
    ASSERT_TRUE(held.has_value());
    const std::uint32_t prefix = draft.segments[in.segment].past;
    for (std::uint64_t c = 0; c < held->size(); ++c) {
      bool visible = c < prefix;
      if (in.cells.ring && visible) {
        const auto h = c + ((prefix - 1 - c) / in.cells.capacity * in.cells.capacity);
        visible = prefix - h < p.window;
        EXPECT_EQ((*held)[c], static_cast<std::int64_t>(h));
      }
      EXPECT_EQ((*held)[c] >= 0, c < prefix) << c;
    }
  }
  // The target's step writes the tap into the drafter's ring at each output
  // row's position; without the drafter's layout it is refused.
  const planner::StepShape verify{.segments = {Seg(0, 4, 300, 512, 4), Seg(4, 1, 1100, 1280, 1)}};
  auto target_step = BuildStep(
      s, 0, verify, {.tensors = target.tensors, .state = &*target_layout, .stages = layouts});
  ASSERT_TRUE(target_step.has_value()) << target_step.error();
  std::size_t taps = 0;
  for (const StepState& st : target_step->state) {
    if (st.ref.stage != 1) continue;
    EXPECT_TRUE(st.tap);
    ++taps;
  }
  EXPECT_EQ(taps, 2U);
  for (const StepInput& in : target_step->inputs) {
    if (in.kind != InputKind::kOutputCells) continue;
    auto cells = InputValues(in, verify, std::vector<std::int32_t>(5, 1));
    ASSERT_TRUE(cells.has_value());
    const planner::Segment& seg = verify.segments[in.segment];
    ASSERT_EQ(cells->size(), seg.outputs);
    for (std::uint32_t i = 0; i < seg.outputs; ++i) {
      EXPECT_EQ((*cells)[i],
                static_cast<std::int64_t>((seg.past + seg.rows - seg.outputs + i) % 16));
    }
  }
  const std::array<const model::StateLayoutDerived*, 1> target_only{&*target_layout};
  EXPECT_FALSE(BuildStep(
      s, 0, verify, {.tensors = target.tensors, .state = &*target_layout, .stages = target_only}));
  // A state-only target step writes no tap.
  auto cut = BuildStep(s, 0, {.segments = {Seg(0, 16, 0, 256, 0)}},
                       {.tensors = target.tensors, .state = &*target_layout, .stages = layouts});
  ASSERT_TRUE(cut.has_value()) << cut.error();
  EXPECT_TRUE(std::ranges::none_of(cut->state, [](const StepState& st) { return st.tap; }));
  // A prefill chunk without outputs beside another owner's decode: only the
  // owner with an output row writes its tap.
  const planner::StepShape mixed{.segments = {Seg(0, 16, 0, 256, 0), Seg(4, 1, 1100, 1280, 1)}};
  auto joined = BuildStep(s, 0, mixed,
                          {.tensors = target.tensors, .state = &*target_layout, .stages = layouts});
  ASSERT_TRUE(joined.has_value()) << joined.error();
  std::vector<std::uint8_t> tapped;
  for (const StepState& st : joined->state) {
    if (st.tap) tapped.push_back(st.ref.segment);
  }
  EXPECT_EQ(tapped, std::vector<std::uint8_t>{1});
}

// A small target with a tap, and its drafter: the target's last attention
// narrows its step to the output rows (NarrowPoint), so the final norm's
// output the tap stores holds those rows only. Its head is the identity,
// so each output's logits are the final norm's row the tap must hold.
TEST(ComponentsTest, ANarrowedTargetStepTapsItsOutputRows) {
  constexpr std::uint64_t kW = 64;
  const auto w = [](std::string role, std::string tensor, std::string stage = {}) {
    return spec::WeightBinding{.role = std::move(role),
                               .tensor = std::move(tensor),
                               .part = format::PartRole::kData,
                               .experts = false,
                               .stage = std::move(stage)};
  };
  const auto ffn = [&](const std::string& prefix) {
    ComponentRef c = C("ffn.glu", {P("placement", N("pre")), P("eps", R(1e-6)),
                                   P("activation", N("gelu_tanh")), P("hidden", I(96))});
    c.weights = {w("down", prefix + "down"), w("gate", prefix + "gate"), w("norm", prefix + "norm"),
                 w("up", prefix + "up")};
    return c;
  };
  spec::ModelSpec s;
  spec::Stage target{.name = "target", .context = 4096};
  target.embedding = C("embedding.token", {P("scale", R(8.0))});
  target.embedding->weights = {w("table", "token_embd")};
  target.stream.width = kW;
  ComponentRef attention =
      C("attention.dense",
        {P("placement", N("pre")), P("eps", R(1e-6)), P("heads", I(4)), P("kv_heads", I(2)),
         P("key_dim", I(16)), P("value_dim", I(16)), P("scale", R(0.25)), P("softcap", R(0.0)),
         P("window", I(0)), P("rotation", N("neox")), P("rope_dims", I(16)),
         P("rope_base", R(10000.0)), P("rope_scale", R(1.0)), P("qk_norm", ParamValue(false))});
  attention.weights = {w("k", "attn_k"), w("norm", "attn_norm"), w("output", "attn_output"),
                       w("q", "attn_q"), w("v", "attn_v")};
  target.layers = {{.components = {attention, ffn("ffn_")}}};
  target.taps = {{.name = "features", .layer = 0, .point = spec::TapPoint::kPostFinalNorm}};
  target.heads.final_norm = C("norm.rms", {P("eps", R(1e-6))});
  target.heads.final_norm->weights = {w("weight", "output_norm")};
  target.heads.output = C("head.linear", {});
  target.heads.output->weights = {w("weight", "output")};
  spec::Stage drafter{.name = "assistant", .role = spec::StageRole::kDrafter, .artifact = 1};
  drafter.embedding = C("embedding.projected", {P("scale", R(8.0)), P("feature", I(kW)),
                                                P("tap", N("features")), P("tap_cells", I(16))});
  drafter.embedding->weights = {w("projection", "a.pre"), w("table", "token_embd", "target")};
  drafter.stream.width = kW;
  drafter.layers = {{.components = {ffn("a.ffn_")}}};
  drafter.heads.final_norm = C("norm.rms", {P("eps", R(1e-6))});
  drafter.heads.final_norm->weights = {w("weight", "a.output_norm")};
  drafter.heads.output = C("head.linear", {P("gather", ParamValue(false)), P("features", I(kW))});
  drafter.heads.output->weights = {w("projection", "a.post"), w("weight", "a.output")};
  drafter.reads = {{.stage = "target", .kind = spec::ReadKind::kTap, .name = "features"}};
  s.stages = {target, drafter};
  spec::Canonicalize(s);

  const auto dense = [](std::initializer_list<std::uint64_t> ne) {
    return ir::Dense(Encoding::kF32, ir::ShapeOf(ne));
  };
  const auto t = [&](std::string name, format::TensorFormat f) {
    return BoundTensor{.name = std::move(name), .format = f};
  };
  const std::vector<BoundTensor> mine{
      t("token_embd", dense({kW, 40})), t("attn_norm", dense({kW})),
      t("attn_q", dense({kW, 64})),     t("attn_k", dense({kW, 32})),
      t("attn_v", dense({kW, 32})),     t("attn_output", dense({64, kW})),
      t("ffn_norm", dense({kW})),       t("ffn_gate", dense({kW, 96})),
      t("ffn_up", dense({kW, 96})),     t("ffn_down", dense({96, kW})),
      t("output_norm", dense({kW})),    t("output", dense({kW, kW}))};
  const std::vector<BoundTensor> theirs{
      t("a.pre", dense({2 * kW, kW})),  t("a.ffn_norm", dense({kW})),
      t("a.ffn_gate", dense({kW, 96})), t("a.ffn_up", dense({kW, 96})),
      t("a.ffn_down", dense({96, kW})), t("a.output_norm", dense({kW})),
      t("a.output", dense({kW, 40})),   t("a.post", dense({kW, kW}))};
  const std::array<std::span<const BoundTensor>, 2> artifacts{mine, theirs};
  const auto valid = ValidateSpec(s, artifacts);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  // The target narrows after its attention; the drafter's head reads every
  // row, so its stage does not.
  const auto target_instances = InstancesOf(s, 0);
  const auto drafter_instances = InstancesOf(s, 1);
  ASSERT_TRUE(target_instances.has_value() && drafter_instances.has_value());
  EXPECT_EQ(NarrowPoint(*target_instances), (std::pair<std::size_t, std::size_t>{0, 0}));
  EXPECT_FALSE(NarrowPoint(*drafter_instances).has_value());

  const auto target_layout = StageLayoutUniform(s, 0, 4096, 16, kExtent);
  const auto drafter_layout = StageLayoutUniform(s, 1, 4096, 16, kExtent);
  ASSERT_TRUE(target_layout.has_value()) << target_layout.error();
  ASSERT_TRUE(drafter_layout.has_value()) << drafter_layout.error();
  const std::array<const model::StateLayoutDerived*, 2> layouts{&*target_layout, &*drafter_layout};
  // Three owners: two outputs of four rows, none of three (a prefill
  // chunk), one of two after five positions: outputs at rows 2, 3 and 8.
  const planner::StepShape shape{
      .segments = {Seg(0, 4, 0, 256, 2), Seg(1, 3, 0, 256, 0), Seg(2, 2, 5, 256, 1)}};
  const std::vector<std::int32_t> tokens{4, 0, 19, 7, 3, 11, 30, 2, 9};
  auto step =
      BuildStep(s, 0, shape, {.tensors = mine, .state = &*target_layout, .stages = layouts});
  ASSERT_TRUE(step.has_value()) << step.error();
  auto exec = ref::Executor::Create(step->graph);
  ASSERT_TRUE(exec.has_value()) << exec.error().detail;
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  for (const StepWeight& sw : step->weights) {
    const auto& name = sw.parts.front().first;
    const auto bound = std::ranges::find(mine, name, &BoundTensor::name);
    ASSERT_NE(bound, mine.end()) << name;
    std::vector<float> v(ir::Elements(bound->format.shape));
    if (name == "output") {
      for (std::uint64_t i = 0; i < kW; ++i) v[(i * kW) + i] = 1.0F;  // the identity
    } else {
      const bool norm = bound->format.shape.rank == 1;
      for (float& x : v) x = norm ? 1.0F + (0.25F * u(rng)) : 0.3F * u(rng);
    }
    ASSERT_TRUE(exec->BindValues(sw.tensor, v)) << name;
  }
  for (const StepState& st : step->state) {
    std::vector<float> zeros(ir::Elements(step->graph.tensor(st.tensor).shape()), 0.0F);
    ASSERT_TRUE(exec->BindValues(st.tensor, zeros));
  }
  for (const StepInput& in : step->inputs) {
    auto values = InputValues(in, shape, tokens);
    ASSERT_TRUE(values.has_value()) << values.error();
    ASSERT_TRUE(exec->BindInts(in.tensor, *values));
  }
  auto ran = exec->Run();
  ASSERT_TRUE(ran.has_value()) << ran.error().detail;
  auto logits = exec->Values(step->logits);
  ASSERT_TRUE(logits.has_value());
  ASSERT_EQ(logits->size(), kW * 3);
  // Each owner's tap: its outputs' rows at their positions' cells, the
  // other cells untouched.
  const std::map<std::uint8_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> want{
      {0, {{2, 0}, {3, 1}}}, {2, {{6, 2}}}};  // segment: (cell, output)
  std::size_t taps = 0;
  for (const StepState& st : step->state) {
    if (!st.tap) continue;
    ++taps;
    const auto cells = want.find(st.ref.segment);
    ASSERT_NE(cells, want.end()) << int{st.ref.segment};
    auto held = exec->Values(st.tensor);
    ASSERT_TRUE(held.has_value());
    ASSERT_EQ(held->size(), kW * 16);
    for (std::uint64_t c = 0; c < 16; ++c) {
      const auto at =
          std::ranges::find(cells->second, c, &std::pair<std::uint64_t, std::uint64_t>::first);
      double scale = 0.0;
      for (std::uint64_t i = 0; i < kW; ++i) {
        const float got = (*held)[(c * kW) + i];
        if (at == cells->second.end()) {
          EXPECT_EQ(got, 0.0F) << "segment " << int{st.ref.segment} << " cell " << c;
          continue;
        }
        const float row = (*logits)[(at->second * kW) + i];
        scale = std::max(scale, std::fabs(double{row}));
        EXPECT_NEAR(got, row, 1e-6 * (1.0 + std::fabs(double{row})))
            << "segment " << int{st.ref.segment} << " cell " << c << " value " << i;
      }
      if (at != cells->second.end()) EXPECT_GT(scale, 0.0);
    }
  }
  EXPECT_EQ(taps, 2U);
}

TEST(ComponentsTest, ReadsAndContractsLinkBothWays) {
  const spec::ModelSpec good = Gemma4WithAssistant(26);
  const auto target = test_support::Gemma4Source(26);
  const auto assistant = test_support::Gemma4AssistantSource(26);
  const std::array<std::span<const BoundTensor>, 2> artifacts{target.tensors, assistant.tensors};
  const auto refused = [&](spec::ModelSpec s, std::string_view why) {
    spec::Canonicalize(s);
    const auto v = ValidateSpec(s, artifacts);
    EXPECT_FALSE(v.has_value()) << why;
  };
  {
    spec::ModelSpec s = good;
    std::erase_if(s.stages[1].reads, [](const spec::Read& r) { return r.name == "v"; });
    refused(s, "a borrowed tensor without its read");
  }
  {
    spec::ModelSpec s = good;
    std::erase_if(s.stages[1].reads,
                  [](const spec::Read& r) { return r.kind == spec::ReadKind::kTap; });
    refused(s, "a tap the target writes without its read");
  }
  {
    spec::ModelSpec s = good;
    s.stages[1].reads.push_back(
        {.stage = "target", .kind = spec::ReadKind::kTap, .name = "features2"});
    s.stages[0].taps.push_back(
        {.name = "features2", .layer = 0, .point = spec::TapPoint::kPostFinalNorm});
    refused(s, "a read nothing uses");
  }
  {
    // A borrowed layer past its target's.
    spec::ModelSpec s = good;
    for (spec::Param& q : s.stages[1].layers[0].components[0].params) {
      if (q.name == "target_layer") q.value = I(31);
    }
    refused(s, "a target layer the target has not");
  }
  {
    // The local layers reading the target's global layer: another cell rule.
    spec::ModelSpec s = good;
    for (spec::Param& q : s.stages[1].layers[0].components[0].params) {
      if (q.name == "target_layer") q.value = I(29);
    }
    refused(s, "a borrowed tensor of another kind than its target's");
  }
  {
    // The target's table bound in the drafter's own artifact: another shape.
    spec::ModelSpec s = good;
    for (spec::WeightBinding& w : s.stages[1].embedding->weights) w.stage.clear();
    refused(s, "the target's table in the drafter's artifact");
  }
  {
    spec::ModelSpec s = good;
    s.stages[1].role = spec::StageRole::kEncoder;
    refused(s, "an encoder that reads");
  }
}

// A store into a borrowed tensor, or into another stage's tensor that is
// not a tap, is refused whichever component made it.
TEST(ComponentsTest, StoresIntoBorrowedOrForeignStateAreRefused) {
  for (const auto& [borrowed, foreign, tap] :
       {std::tuple{true, false, false}, std::tuple{false, true, false},
        std::tuple{false, true, true}, std::tuple{false, false, false}}) {
    ir::Builder b;
    const auto cache_format = ir::Dense(Encoding::kF16, ir::ShapeOf({8, 16}));
    const ir::StateRef ref{.stage = static_cast<std::uint16_t>(foreign ? 1 : 0),
                           .tensor = 0,
                           .layer = 0,
                           .segment = 0};
    const ir::TensorId cache = b.State(cache_format, ref, "cache");
    const ir::TensorId rows = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 2})), "rows");
    const ir::TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({2})), "idx");
    b.Keep(b.CacheStore(cache, rows, idx));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph.has_value()) << graph.error().detail;
    StepGraph step{
        .graph = std::move(*graph),
        .inputs = {},
        .weights = {},
        .state = {{.tensor = cache, .ref = ref, .item = 0, .borrowed = borrowed, .tap = tap}}};
    const bool allowed = !borrowed && (!foreign || tap);
    EXPECT_EQ(CheckStateWrites(step, 0).has_value(), allowed)
        << borrowed << " " << foreign << " " << tap;
  }
}

TEST(ComponentsTest, ApplyKvCacheEncodingLeavesNonAttentionComponentsUnchanged) {
  spec::ModelSpec spec = Gemma2Spec();
  const auto glu_before = spec.stages[0].layers[0].components[1].params;
  auto bytes = KvCacheEncodingBytes(spec, format::Encoding::kQ8_0);
  ASSERT_TRUE(bytes.has_value()) << bytes.error();
  auto applied = ApplyKvCacheEncoding(spec, format::Encoding::kQ8_0, *bytes);
  ASSERT_TRUE(applied.has_value()) << applied.error();
  EXPECT_EQ(spec.stages[0].layers[0].components[1].params, glu_before);
  EXPECT_EQ(spec.stages[0].layers[0].components[1].Find("kv_encoding"), nullptr);

  // A spec with only non-attention components cannot compute KV cache encoding bytes.
  spec::ModelSpec no_attn = spec;
  for (auto& stage : no_attn.stages) {
    for (auto& layer : stage.layers) {
      std::erase_if(layer.components,
                    [](const spec::ComponentRef& c) { return c.kind == "attention.dense"; });
    }
  }
  auto no_attn_bytes = KvCacheEncodingBytes(no_attn, format::Encoding::kQ8_0);
  EXPECT_FALSE(no_attn_bytes.has_value());
}

TEST(ComponentsTest, DenseAttentionComposableOptionsMatchTheirReference) {
  constexpr std::uint16_t kRopeFreqs = 8;
  std::vector<format::TensorFormat> formats{
      Vector(kWidth),
      Matrix(Encoding::kF32, kWidth, 256),
      Matrix(Encoding::kF32, kWidth, 64),
      Matrix(Encoding::kF32, kWidth, 64),
      Matrix(Encoding::kF32, 128, kWidth),
      Vector(kWidth),
      Vector(32),
      Vector(32),
      Vector(8),
      Vector(256),
      Vector(64),
      Vector(64),
      Vector(4),
  };
  for (const bool stateless : {false, true}) {
    for (const std::int64_t window : {std::int64_t{0}, std::int64_t{8}}) {
      SCOPED_TRACE(std::format("stateless {} window {}", stateless, window));
      ComponentRef r = C("attention.dense", {P("placement", N("sandwich")),
                                             P("eps", R(1e-6)),
                                             P("heads", I(4)),
                                             P("kv_heads", I(2)),
                                             P("key_dim", I(32)),
                                             P("value_dim", I(32)),
                                             P("scale", R(1.0 / std::sqrt(32.0))),
                                             P("softcap", R(30.0)),
                                             P("window", I(window)),
                                             P("rotation", N("normal")),
                                             P("rope_dims", I(16)),
                                             P("rope_offset", I(16)),
                                             P("rope_base", R(10000.0)),
                                             P("rope_scale", R(0.25)),
                                             P("yarn", ParamValue(true)),
                                             P("rope_factor", R(4.0)),
                                             P("yarn_context", I(64)),
                                             P("beta_fast", R(32.0)),
                                             P("beta_slow", R(1.0)),
                                             P("qk_norm", ParamValue(true)),
                                             P("rope_freqs", ParamValue(true)),
                                             P("sinks", ParamValue(true)),
                                             P("output_gate", ParamValue(true)),
                                             P("stateless", ParamValue(stateless))});
      auto in = MakeInstance(r, kFacts, Site::kLayer, 0, 1);
      ASSERT_TRUE(in.has_value()) << in.error();
      EXPECT_EQ(in->def->state(*in).tensors.empty(), stateless);
      EXPECT_EQ(in->def->masks(*in).empty(), stateless);
      const std::uint64_t cells = window != 0 ? 20 : 32;
      const std::vector<std::uint64_t> state_cells =
          stateless ? std::vector<std::uint64_t>{} : std::vector<std::uint64_t>{cells, cells};
      Compare(*in, formats, state_cells, AttentionSteps(32), kOutputTolerance, kF16StateTolerance,
              {{kRopeFreqs, {1.0F, 1.5F}}});
    }
  }
}

TEST(ComponentsTest, EveryCatalogComponentHasAReferenceAndIsTested) {
  const auto token_recipes = test_catalog::CatalogTokenRecipes();
  const auto spatial_recipes = test_catalog::CatalogSpatialRecipes();
  for (const ComponentDef* def : Catalog()) {
    SCOPED_TRACE(std::format("{} v{}", def->kind, def->version));
    const bool has_ref = def->reference != nullptr ||
                         (def->spatial != nullptr && def->spatial->reference != nullptr);
    ASSERT_TRUE(has_ref) << def->kind << " v" << def->version
                         << " is missing a scalar CPU reference function";

    const auto tok_it =
        std::ranges::find_if(token_recipes, [&](const test_catalog::CatalogTokenRecipe& r) {
          return r.kind == def->kind && r.version == def->version;
        });
    const auto spa_it =
        std::ranges::find_if(spatial_recipes, [&](const test_catalog::CatalogSpatialRecipe& r) {
          return r.kind == def->kind && r.version == def->version;
        });
    ASSERT_TRUE(tok_it != token_recipes.end() || spa_it != spatial_recipes.end())
        << def->kind << " v" << def->version
        << " is not covered by CatalogTokenRecipes() or CatalogSpatialRecipes()";

    if (tok_it != token_recipes.end()) {
      ASSERT_FALSE(tok_it->cases.empty());
      for (const auto& tc : tok_it->cases) {
        SCOPED_TRACE(tc.label);
        test_catalog::RunCatalogTokenCpuCase(tc);
      }
    }
    if (spa_it != spatial_recipes.end()) {
      ASSERT_FALSE(spa_it->cases.empty());
      for (const auto& sc : spa_it->cases) {
        SCOPED_TRACE(sc.label);
        test_catalog::RunCatalogSpatialCpuCase(sc);
      }
    }
  }
}

}  // namespace
}  // namespace llmp::components
