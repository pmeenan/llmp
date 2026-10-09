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
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/reference/exec.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/host_mask.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/spec.h"
#include "spec_fixtures.h"

namespace llmp::components {
namespace {

namespace ref = llmp::kernels::reference;
using format::Encoding;
using spec::ComponentRef;
using spec::Param;
using spec::ParamValue;

constexpr std::uint64_t kExtent = 2ULL << 20U;

ComponentRef C(std::string kind, std::vector<Param> params) {
  std::ranges::sort(params, {}, &Param::name);
  return {.kind = std::move(kind), .version = 1, .params = std::move(params), .weights = {}};
}
Param P(std::string name, ParamValue v) { return {.name = std::move(name), .value = std::move(v)}; }
ParamValue I(std::int64_t v) { return ParamValue(v); }
ParamValue R(double v) { return ParamValue(v); }
ParamValue N(std::string v) { return ParamValue(std::move(v)); }

spec::ModelSpec Gemma2Spec() {
  auto values = spec::mappings::Gemma2V0Values(model::Gemma2_2B());
  auto s = spec::Synthesize(spec::mappings::Gemma2Mapping(), test_support::Gemma2Source(), *values);
  EXPECT_TRUE(s.has_value()) << s.error();
  return *s;
}
spec::ModelSpec Gemma3Spec() {
  auto values = spec::mappings::Gemma3V0Values(model::Gemma3_4BQat());
  auto s = spec::Synthesize(spec::mappings::Gemma3Mapping(), test_support::Gemma3Source(), *values);
  EXPECT_TRUE(s.has_value()) << s.error();
  return *s;
}

// ---------------------------------------------------------------- catalog

TEST(ComponentsTest, CatalogHoldsTheGemmaKindsInOrder) {
  std::vector<std::string_view> kinds;
  for (const ComponentDef* def : Catalog()) {
    kinds.push_back(def->kind);
    EXPECT_NE(def->emit, nullptr) << def->kind;
    EXPECT_NE(def->reference, nullptr) << def->kind;
    EXPECT_NE(def->shapes, nullptr) << def->kind;
    EXPECT_EQ(Find(def->kind, def->version), def);
  }
  EXPECT_TRUE(std::ranges::is_sorted(kinds));
  EXPECT_EQ(kinds, (std::vector<std::string_view>{"attention.dense", "embedding.token", "ffn.glu",
                                                  "head.linear", "norm.rms"}));
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
    // A drafter stage reading state its target does not have.
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
    const auto ok = ValidateSpec(s, tensors);
    EXPECT_TRUE(ok.has_value()) << ok.error();
  }
}

// ---------------------------------------------------------- state layout

TEST(ComponentsTest, StageLayoutsAreTodaysGemmaLayouts) {
  for (const auto& [context, rows] : {std::pair{4096U, 128U}, {8192U, 512U}, {8192U, 16U}}) {
    const auto want = model::Gemma2State(model::Gemma2_2B(), context, rows);
    ASSERT_TRUE(want.has_value());
    const auto got = StageLayout(Gemma2Spec(), 0, context, rows, kExtent);
    ASSERT_TRUE(got.has_value()) << got.error();
    ASSERT_EQ(got->items.size(), want->tensors.size());
    for (std::size_t i = 0; i < want->tensors.size(); ++i) {
      EXPECT_EQ(got->items[i].layer, want->tensors[i].layer);
      EXPECT_EQ(got->items[i].offset, want->tensors[i].offset);
      EXPECT_EQ(got->items[i].cells, want->tensors[i].cells);
      EXPECT_EQ(got->items[i].bytes, want->tensors[i].bytes);
    }
    EXPECT_EQ(got->bytes, want->bytes);
  }
  for (const auto& [context, rows] : {std::pair{8192U, 512U}, {131072U, 2048U}}) {
    const auto want = model::Gemma3State(model::Gemma3_4BQat(), context, rows);
    ASSERT_TRUE(want.has_value());
    const auto got = StageLayout(Gemma3Spec(), 0, context, rows, kExtent);
    ASSERT_TRUE(got.has_value()) << got.error();
    ASSERT_EQ(got->items.size(), want->tensors.size());
    for (std::size_t i = 0; i < want->tensors.size(); ++i) {
      EXPECT_EQ(got->items[i].offset, want->tensors[i].offset);
      EXPECT_EQ(got->items[i].cells, want->tensors[i].cells);
    }
    EXPECT_EQ(got->bytes, want->bytes);
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
             double state_tolerance = 0.0) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  const auto shapes = in.def->shapes(in);
  ASSERT_TRUE(shapes.has_value());
  // Random weights by role, encoded in their formats; both sides read the
  // decoded values.
  RefIO io{.instance = &in};
  io.weights.resize(in.def->roles.size());
  io.weight_shapes.resize(in.def->roles.size());
  std::vector<std::vector<std::byte>> encoded(in.def->roles.size());
  for (const RoleShape& r : *shapes) {
    const auto& f = formats[r.role];
    std::vector<float> v(ir::Elements(f.shape));
    const bool norm = f.shape.rank == 1;
    for (float& x : v) x = norm ? 1.0F + (0.25F * u(rng)) : 0.3F * u(rng);
    const auto parts = format::PartsOf(f);
    ASSERT_TRUE(parts.has_value());
    encoded[r.role].resize((*parts)[0].bytes);
    ASSERT_TRUE(ref::EncodeRows(f, 0, 0, v, encoded[r.role]));
    const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[r.role]};
    io.weights[r.role].resize(v.size());
    ASSERT_TRUE(ref::DecodeRows(f, std::span(&part, 1), 0, 0, f.shape.ne[1], io.weights[r.role]));
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
    // State: zeros, or what the last step left (by slot).
    io.state.clear();
    io.segments.clear();
    for (const StepState& st : graph->step.state) {
      const std::uint8_t slot = step.shape.segments.at(st.ref.segment).slot;
      const auto at = carried.find({slot, st.ref.tensor});
      if (at == carried.end()) {
        const std::vector<float> zeros(ir::Elements(graph->step.graph.tensor(st.tensor).shape()),
                                       0.0F);
        ASSERT_TRUE(exec->BindValues(st.tensor, zeros));
      } else {
        const ref::PartView part{.role = format::PartRole::kData, .bytes = at->second};
        ASSERT_TRUE(exec->Bind(st.tensor, std::span(&part, 1)));
      }
    }
    for (const planner::Segment& s : step.shape.segments) {
      io.segments.push_back(
          {.rows = s.rows, .past = s.past, .read_cells = s.read_cells, .outputs = s.outputs});
      auto& st = ref_state[s.slot];
      if (st.empty()) {
        for (std::size_t t = 0; t < contract.tensors.size(); ++t) {
          const auto& c = contract.tensors[t];
          st.emplace_back(cells[t] * (c.cell_bytes / 2), 0.0);
        }
      }
      io.state.push_back(st);
    }
    // Inputs.
    std::vector<double> stream(in.stage.width * step.shape.rows());
    for (double& x : stream) x = 2.0 * u(rng);
    for (const StepInput& input : graph->step.inputs) {
      if (input.kind == InputKind::kStream) {
        std::vector<float> f(stream.begin(), stream.end());
        ASSERT_TRUE(exec->BindValues(input.tensor, f));
        // The reference reads what the graph reads: the F32 values.
        for (std::size_t i = 0; i < stream.size(); ++i) stream[i] = f[i];
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
          Vector(32)};
}

// Two owners: prefills of 5 and 3 rows, a decode each, a longer chunk
// that fills the 20-cell ring, then one that wraps it.
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
  io.weights.resize(8);
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
    const std::vector<format::TensorFormat> hf{Matrix(Encoding::kQ8_0, kWidth, 40)};
    Compare(*head, hf, {}, rows, 1e-5);
    // Greedy: one output a segment, each row's highest logit.
    const std::vector<Step> greedy{
        {.shape = {.segments = {Seg(0, 3, 0, 256, 1, true), Seg(1, 2, 7, 256, 1, true)}}}};
    Compare(*head, hf, {}, greedy, 1e-5);
  }
}

// ---------------------------------------------------------------- the step

std::size_t Count(const ir::Graph& g, ir::Op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(g.nodes(), [op](const ir::Node& n) { return n.key.op == op; }));
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
    const auto layout = StageLayout(m.spec, 0, 8192, 16, kExtent);
    ASSERT_TRUE(layout.has_value());
    const Bindings bindings{.tensors = m.source.tensors, .state = &*layout};
    // Two owners (3 rows after 4351 positions, 1 row after 7).
    const planner::StepShape two{.segments = {Seg(3, 3, 4351, 4608, 1), Seg(1, 1, 7, 256, 1)}};
    auto step = BuildStep(m.spec, 0, two, bindings);
    ASSERT_TRUE(step.has_value()) << step.error();
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
    // Refusals: reads that miss the rows, mixed greedy, a stage mismatch.
    EXPECT_FALSE(BuildStep(m.spec, 0, {.segments = {Seg(0, 4, 300, 256, 1)}}, bindings));
    EXPECT_FALSE(BuildStep(
        m.spec, 0, {.segments = {Seg(0, 1, 9, 256, 1, true), Seg(1, 1, 3, 256, 1)}}, bindings));
    EXPECT_FALSE(BuildStep(m.spec, 0, {.stage = 1, .segments = {Seg(0, 1, 0, 256, 1)}}, bindings));
    EXPECT_FALSE(BuildStep(m.spec, 0, two, {.tensors = m.source.tensors, .state = nullptr}));
  }
}

TEST(ComponentsTest, InputValuesReproduceTodaysCellsAndHostMasks) {
  const auto& p = model::Gemma2_2B();
  const auto state = model::Gemma2State(p, 8192, 512);
  ASSERT_TRUE(state.has_value());
  const auto spec = Gemma2Spec();
  const auto layout = StageLayout(spec, 0, 8192, 512, kExtent);
  ASSERT_TRUE(layout.has_value());
  // Two owners, one wrapping its ring (4608 local cells).
  std::vector<std::int32_t> tokens(400 + 3, 5);
  const std::array segments{
      model::Gemma2Segment{.slot = 2, .n_past = 4500, .tokens = std::span(tokens).first(400)},
      model::Gemma2Segment{.slot = 0, .n_past = 9, .tokens = std::span(tokens).last(3)}};
  const auto chunk = model::Gemma2Chunk(p, *state, segments, true);
  ASSERT_TRUE(chunk.has_value()) << chunk.error();
  planner::StepShape shape;
  for (const auto& s : chunk->segments) {
    shape.segments.push_back(Seg(static_cast<std::uint8_t>(s.slot), s.rows, s.n_past,
                                 std::max(s.global_n_kv, s.local_n_kv), 1));
  }
  const auto source = test_support::Gemma2Source();
  auto step = BuildStep(spec, 0, shape, {.tensors = source.tensors, .state = &*layout});
  ASSERT_TRUE(step.has_value()) << step.error();
  std::size_t checked = 0;
  for (const StepInput& input : step->inputs) {
    auto values = InputValues(input, shape, tokens);
    ASSERT_TRUE(values.has_value()) << values.error();
    if (input.kind == InputKind::kPositions) {
      EXPECT_TRUE(std::ranges::equal(*values, chunk->positions));
      ++checked;
    }
    if (input.kind == InputKind::kStoreCells) {
      const auto& seg = chunk->segments[input.segment];
      EXPECT_TRUE(
          std::ranges::equal(*values, input.cells.ring ? seg.local_cells : seg.global_cells));
      ++checked;
    }
    if (input.kind == InputKind::kCellPositions) {
      // Today's host mask, row by row, from the cells' positions and the
      // mask's definition (ir/op.h kMaskBuild).
      const auto& seg = chunk->segments[input.segment];
      const bool ring = input.cells.ring;
      const auto& want = ring ? seg.local_mask : seg.global_mask;
      const std::uint64_t width = ring ? seg.local_n_kv : seg.global_n_kv;
      ASSERT_EQ(values->size(), width);
      for (std::uint32_t i = 0; i < seg.rows; ++i) {
        const std::int64_t pos = seg.n_past + i;
        for (std::uint64_t c = 0; c < width; ++c) {
          const std::int64_t held = (*values)[c];
          bool seen = held >= 0 && held <= pos;
          if (ring) seen = seen && pos - held < p.window;
          ASSERT_EQ(want[(i * width) + c], seen ? 0 : 0xFC00) << i << " " << c;
        }
      }
      ++checked;
    }
  }
  EXPECT_EQ(checked, 1U + 4U + 4U);
}

}  // namespace
}  // namespace llmp::components
