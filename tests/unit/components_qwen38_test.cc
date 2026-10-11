// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen3.8 components (components/hc.cc, gdn.cc, qsa.cc, ple.cc, the
// MoE's hyper-connection form) and the framework they need (several
// streams, side inputs, layer taps, drafter cascades, verify saves and
// commits, state access flags), on the CPU:
//
// - each component's fragment, run by the reference executor over several
//   steps (prefill, decode, several owners) against its own scalar
//   reference, with the streams, carry and side input it reads;
// - a verify's saves and their commit against a plain step of the kept
//   rows (Gated DeltaNet and the n-gram block's fixed state);
// - the n-gram rows' hash against model::Qwen38PleRows;
// - the qwen4exp and qwen4exp-mtp mappings over every import's tensor
//   table (tests/unit/qwen38_fixture.h): one spec structure, the forms each
//   import's tensors take, the folds, the stage layout against today's
//   Qwen38State and the drafter's against Qwen38MtpStateOf; and steps
//   built from them (the target writing its drafter's streams every row,
//   state-only included; the drafter's pass);
// - reads widened: a drafter reading a layer's streams, a cascade (a
//   drafter reading the drafter before it), and the refusals.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/reference/exec.h"
#include "model/qwen38.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "qwen38_fixture.h"
#include "spec/mapping.h"
#include "spec/mappings/qwen38.h"
#include "spec/spec.h"

namespace llmp::components {
namespace {

namespace ref = llmp::kernels::reference;
namespace fx = llmp::test_support::qwen38;
using format::Encoding;
using spec::ComponentRef;
using spec::Param;
using spec::ParamValue;

ComponentRef C(std::string kind, std::vector<Param> params) {
  std::ranges::sort(params, {}, &Param::name);
  return {.kind = std::move(kind), .version = 1, .params = std::move(params), .weights = {}};
}
Param P(std::string name, ParamValue v) { return {.name = std::move(name), .value = std::move(v)}; }
ParamValue I(std::int64_t v) { return ParamValue(v); }
ParamValue R(double v) { return ParamValue(v); }
ParamValue N(std::string v) { return ParamValue(std::move(v)); }
ParamValue B(bool v) { return ParamValue(v); }
ParamValue L(std::vector<std::int64_t> v) { return ParamValue(std::move(v)); }

planner::Segment Seg(std::uint8_t slot, std::uint32_t rows, std::uint32_t past, std::uint32_t read,
                     std::uint32_t outputs, bool greedy = false) {
  return {.slot = slot,
          .rows = rows,
          .past = past,
          .read_cells = read,
          .outputs = outputs,
          .greedy = greedy};
}

format::TensorFormat Matrix(Encoding e, std::uint64_t k, std::uint64_t n, std::uint64_t m = 1) {
  format::TensorFormat f = ir::Dense(e, m > 1 ? ir::ShapeOf({k, n, m}) : ir::ShapeOf({k, n}));
  if (format::Info(e).quantized) f.layout = format::Layout::kGgmlBlocks;
  return f;
}
format::TensorFormat Vector(std::uint64_t n) { return ir::Dense(Encoding::kF32, ir::ShapeOf({n})); }

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

const std::vector<std::string>& Folds() {
  static const std::vector<std::string> folds{"gdn.value_heads_tiled", "norm.plus_one",
                                              "ssm.a_neg_exp"};
  return folds;
}

struct Step {
  planner::StepShape shape;
  std::vector<std::int32_t> tokens{};
};

// What a lone component reads beside its stream, and the side table's rows
// for a side source.
struct Lone {
  ComponentInputs inputs{};
  std::uint64_t table_rows = 0;  // a side source's (kSideRows values below it)
};

// Runs `in`'s fragment (the reference executor) and its reference over
// `steps`, carrying the state by slot, and compares every output (stream,
// streams, carry, logits, features) and the state within `tolerance` of
// the largest value (`state_tolerance` for the state). Weights are random
// in `formats` (a role's own range in `ranges`), both sides reading the
// decoded values.
void Compare(const Instance& in, std::span<const format::TensorFormat> formats,
             std::span<const std::uint64_t> cells, std::span<const Step> steps, double tolerance,
             double state_tolerance, const Lone& lone = {},
             const std::map<std::uint16_t, std::pair<float, float>>& ranges = {}) {
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  const auto shapes = in.def->shapes(in);
  ASSERT_TRUE(shapes.has_value()) << shapes.error();
  RefIO io{.instance = &in};
  io.weights.resize(in.def->roles.size());
  io.weight_shapes.resize(in.def->roles.size());
  std::vector<std::vector<std::byte>> encoded(in.def->roles.size());
  for (const RoleShape& r : *shapes) {
    const auto& f = formats[r.role];
    std::vector<float> v(ir::Elements(f.shape));
    const bool norm = f.shape.rank == 1;
    const auto range = ranges.find(r.role);
    for (float& x : v) {
      x = range != ranges.end() ? range->second.first + (range->second.second * (1.0F + u(rng)))
          : norm                ? 1.0F + (0.25F * u(rng))
                                : 0.3F * u(rng);
    }
    const auto parts = format::PartsOf(f);
    ASSERT_TRUE(parts.has_value());
    encoded[r.role].resize((*parts)[0].bytes);
    if (f.encoding == Encoding::kI32) {
      // Integers (a selected head's ids): ascending.
      io.weights[r.role].resize(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) {
        const auto id = static_cast<std::int32_t>(3 + (5 * i));
        std::memcpy(encoded[r.role].data() + (i * 4), &id, 4);
        io.weights[r.role][i] = static_cast<float>(id);
      }
      io.weight_shapes[r.role] = f.shape;
      continue;
    }
    const std::uint64_t matrices = format::Matrices(f);
    const std::size_t per = v.size() / matrices;
    io.weights[r.role].resize(v.size());
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::EncodeRows(f, m, 0, std::span(v).subspan(m * per, per), encoded[r.role]));
    }
    const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[r.role]};
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::DecodeRows(f, std::span(&part, 1), m, 0,
                                  f.shape.ne[1] * f.shape.ne[2] / matrices,
                                  std::span(io.weights[r.role]).subspan(m * per, per)));
    }
    io.weight_shapes[r.role] = f.shape;
  }
  const model::StateContract contract =
      in.def->state != nullptr ? in.def->state(in) : model::StateContract{};
  io.state_cells.assign(cells.begin(), cells.end());
  std::map<std::pair<std::uint8_t, std::uint16_t>, std::vector<std::byte>> carried;
  std::map<std::uint8_t, std::vector<std::vector<double>>> ref_state;
  for (std::size_t si = 0; si < steps.size(); ++si) {
    SCOPED_TRACE(std::format("step {}", si));
    const Step& step = steps[si];
    auto graph = BuildComponent(in, step.shape, formats, cells, lone.inputs);
    ASSERT_TRUE(graph.has_value()) << graph.error();
    auto exec = ref::Executor::Create(graph->step.graph);
    ASSERT_TRUE(exec.has_value()) << exec.error().detail;
    for (const RoleShape& r : *shapes) {
      const ir::TensorId leaf = graph->weights[r.role];
      if (!leaf.valid()) continue;
      const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[r.role]};
      ASSERT_TRUE(exec->Bind(leaf, std::span(&part, 1)));
    }
    io.state.clear();
    io.segments.clear();
    for (const StepState& st : graph->step.state) {
      const std::uint8_t slot = step.shape.segments.at(st.ref.segment).slot;
      const auto at = carried.find({slot, st.ref.tensor});
      auto& start = ref_state[slot];
      if (start.empty()) start.resize(contract.tensors.size());
      if (at == carried.end()) {
        std::vector<float> values(ir::Elements(graph->step.graph.tensor(st.tensor).shape()), 0.0F);
        ASSERT_TRUE(exec->BindValues(st.tensor, values));
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
    // Inputs: the caller's random, the framework's from the shape.
    io.stream.clear();
    io.residual.clear();
    io.carry.clear();
    io.side.clear();
    io.side_rows.clear();
    io.features.clear();
    for (const StepInput& input : graph->step.inputs) {
      const auto random = [&](std::vector<double>& into, float scale) {
        std::vector<float> f(ir::Elements(graph->step.graph.tensor(input.tensor).shape()));
        for (float& x : f) x = scale * u(rng);
        EXPECT_TRUE(exec->BindValues(input.tensor, f));
        into.assign(f.begin(), f.end());
      };
      switch (input.kind) {
        case InputKind::kStream:
          random(io.stream, 2.0F);
          continue;
        case InputKind::kResidual:
          random(io.residual, 2.0F);
          continue;
        case InputKind::kCarry:
          random(io.carry, 1.0F);
          continue;
        case InputKind::kSide:
          random(io.side, 1.0F);
          continue;
        case InputKind::kFeatures:
          random(io.features, 2.0F);
          continue;
        case InputKind::kSideRows: {
          std::uniform_int_distribution<std::int64_t> pick(
              0, static_cast<std::int64_t>(lone.table_rows) - 1);
          std::vector<std::int64_t> rows(
              ir::Elements(graph->step.graph.tensor(input.tensor).shape()));
          for (auto& r : rows) r = pick(rng);
          ASSERT_TRUE(exec->BindInts(input.tensor, rows));
          io.side_rows = rows;
          continue;
        }
        default:
          break;
      }
      auto values = InputValues(input, step.shape, step.tokens);
      ASSERT_TRUE(values.has_value()) << values.error();
      ASSERT_TRUE(exec->BindInts(input.tensor, *values));
    }
    auto ran = exec->Run();
    ASSERT_TRUE(ran.has_value()) << ran.error().detail;
    io.tokens = step.tokens;
    io.greedy = graph->step.greedy.valid();
    io.greedy_nan = in.def->kind == "head.linear" && in.params.Flag("native_argmax") &&
                            (step.shape.phase == planner::Phase::kVerify ||
                             step.shape.phase == planner::Phase::kDraft)
                        ? ir::ArgmaxNan::kIgnore
                        : ir::ArgmaxNan::kFirstColumn;
    auto r = in.def->reference(io);
    ASSERT_TRUE(r.has_value()) << r.error();
    const auto check = [&](ir::TensorId t, std::span<const double> want, std::string_view what) {
      if (!t.valid()) return;
      auto got = exec->Values(t);
      ASSERT_TRUE(got.has_value());
      EXPECT_LE(Gap(*got, want), tolerance) << what;
    };
    if (graph->step.logits.valid()) {
      check(graph->step.logits, io.logits, "logits");
      if (graph->step.greedy.valid()) {
        auto tokens = exec->Ints(graph->step.greedy);
        ASSERT_TRUE(tokens.has_value());
        ASSERT_EQ(tokens->size(), io.tokens_out.size());
        for (std::size_t i = 0; i < tokens->size(); ++i) EXPECT_EQ((*tokens)[i], io.tokens_out[i]);
      }
    } else {
      check(graph->stream_out, io.stream, "stream");
      check(graph->residual_out, io.residual, "streams");
      check(graph->carry_out, io.carry, "carry");
    }
    check(graph->step.features, io.features_out, "features");
    for (const StepState& st : graph->step.state) {
      const std::uint8_t slot = step.shape.segments.at(st.ref.segment).slot;
      auto values = exec->Values(st.tensor);
      ASSERT_TRUE(values.has_value());
      EXPECT_LE(Gap(*values, io.state[st.ref.segment][st.ref.tensor]), state_tolerance)
          << "state " << st.ref.tensor << " of slot " << int{slot};
      const auto bytes = exec->Bytes(st.tensor);
      carried[{slot, st.ref.tensor}] = std::vector<std::byte>(bytes.begin(), bytes.end());
    }
    for (std::size_t s = 0; s < step.shape.segments.size(); ++s) {
      ref_state[step.shape.segments[s].slot] = io.state[s];
    }
  }
}

std::size_t Count(const ir::Graph& g, ir::Op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(g.nodes(), [op](const ir::Node& n) { return n.key.op == op; }));
}

constexpr double kTolerance = 1e-4;
constexpr double kF16State = 1.0 / 1024;

// ---------------------------------------------------------------- catalog

TEST(ComponentsQwen38Test, TheCatalogHoldsTheQwenKinds) {
  for (const char* kind : {"residual.hc", "hc.mix", "hc.combine", "attention.gdn", "attention.qsa",
                           "side.ngram", "residual.ple", "embedding.mtp"}) {
    EXPECT_NE(Find(kind, 1), nullptr) << kind;
  }
  const auto all = Catalog();
  EXPECT_TRUE(std::ranges::is_sorted(all, {}, [](const ComponentDef* d) { return d->kind; }));
  EXPECT_TRUE(Find("residual.hc", 1)->mixing);
  EXPECT_TRUE(Find("side.ngram", 1)->side_source);
  EXPECT_TRUE(Find("side.ngram", 1)->roles[0].rows_on_demand);
  EXPECT_NE(Find("attention.gdn", 1)->commit, nullptr);
  EXPECT_NE(Find("residual.ple", 1)->commit, nullptr);
  // The source bounds hold Qwen3.8's ModelOpt checkpoint (301,730 tensors).
  EXPECT_GE(spec::kMaxSourceTensors, 301730U);
}

// ---------------------------------------------------------------- hyper-connections

constexpr std::uint64_t kHcWidth = 32;
constexpr std::uint32_t kStreams = 4;
const StageFacts kHcFacts{
    .width = kHcWidth, .context = 4096, .streams = kStreams, .folds = Folds()};

std::vector<Step> RowSteps() {
  return {
      {.shape = {.segments = {Seg(0, 3, 0, 256, 1), Seg(1, 2, 7, 256, 2)}},
       .tokens = {4, 0, 9, 7, 3}},
      {.shape = {.segments = {Seg(0, 1, 3, 256, 1)}}, .tokens = {5}},
  };
}

TEST(ComponentsQwen38Test, HyperConnectionsMatchTheirReferences) {
  const std::uint64_t hw = kHcWidth * kStreams;
  auto mixing =
      MakeInstance(C("residual.hc", {P("mix", N("sigmoid"))}), kHcFacts, Site::kMixing, 0, 1);
  ASSERT_TRUE(mixing.has_value()) << mixing.error();
  Compare(*mixing, {}, {}, RowSteps(), kTolerance, kTolerance);
  // A block's mix: the mixed stream, and its carry for the combine.
  auto mix =
      MakeInstance(C("hc.mix", {P("eps", R(1e-6)), P("rank", I(8))}), kHcFacts, Site::kLayer, 0, 2);
  ASSERT_TRUE(mix.has_value()) << mix.error();
  const std::vector<format::TensorFormat> mf{Vector(hw), Matrix(Encoding::kF16, hw, 8),
                                             Matrix(Encoding::kF16, 8, hw)};
  Compare(*mix, mf, {}, RowSteps(), kTolerance, kTolerance, {.inputs = {.residual = true}});
  // The head's: the output rows' streams gathered first, and returned.
  auto head = MakeInstance(C("hc.mix", {P("eps", R(1e-6)), P("rank", I(8)), P("inject", B(false)),
                                        P("features", B(true))}),
                           kHcFacts, Site::kFinalNorm, 0, 3);
  ASSERT_TRUE(head.has_value()) << head.error();
  Compare(*head, mf, {}, RowSteps(), kTolerance, kTolerance, {.inputs = {.residual = true}});
  // The combine.
  auto combine = MakeInstance(C("hc.combine", {}), kHcFacts, Site::kLayer, 0, 4);
  ASSERT_TRUE(combine.has_value()) << combine.error();
  const std::vector<format::TensorFormat> cf{Matrix(Encoding::kF32, hw, kStreams)};
  Compare(*combine, cf, {}, RowSteps(), kTolerance, kTolerance,
          {.inputs = {.residual = true, .carry = hw}});
  // Refused: in a stage of one stream; a head's mix that injects.
  StageFacts one = kHcFacts;
  one.streams = 1;
  EXPECT_FALSE(MakeInstance(C("hc.combine", {}), one, Site::kLayer, 0, 4));
  EXPECT_FALSE(MakeInstance(C("hc.mix", {P("eps", R(1e-6)), P("rank", I(8))}), kHcFacts,
                            Site::kFinalNorm, 0, 3));
}

// ---------------------------------------------------------------- Gated DeltaNet

constexpr std::uint64_t kWidth = 64;
const StageFacts kFacts{.width = kWidth, .context = 4096, .streams = kStreams, .folds = Folds()};

ComponentRef Gdn() {
  return C("attention.gdn", {P("eps", R(1e-6)), P("k_heads", I(2)), P("v_heads", I(4)),
                             P("head_dim", I(16)), P("conv", I(4))});
}
// qkv, gate, beta, alpha, dt_bias, a, conv, norm, output.
std::vector<format::TensorFormat> GdnFormats(Encoding e) {
  const std::uint64_t ch = (2 * 2 * 16) + (4 * 16);
  return {Matrix(e, kWidth, ch),
          Matrix(e, kWidth, 64),
          Matrix(Encoding::kF32, kWidth, 4),
          Matrix(Encoding::kF32, kWidth, 4),
          Vector(4),
          Vector(4),
          Matrix(Encoding::kF32, 4, ch),
          Vector(16),
          Matrix(e, 64, kWidth)};
}
// A's decay negative (-exp(A_log), folded), dt_bias small, the kernel small.
const std::map<std::uint16_t, std::pair<float, float>> kGdnRanges{
    {5, {-1.2F, 0.5F}}, {4, {-0.5F, 0.5F}}, {6, {-0.6F, 0.6F}}};

TEST(ComponentsQwen38Test, GatedDeltaNetMatchesItsReference) {
  auto in = MakeInstance(Gdn(), kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in.has_value()) << in.error();
  const std::array<std::uint64_t, 2> cells{1, 1};
  const std::vector<Step> steps{
      {.shape = {.segments = {Seg(0, 5, 0, 256, 5), Seg(3, 3, 0, 256, 3)}}},
      {.shape = {.segments = {Seg(0, 1, 5, 256, 1), Seg(3, 1, 3, 256, 1)}}},
      {.shape = {.segments = {Seg(0, 2, 6, 256, 2)}}},
      // State-only, then a row that reads it.
      {.shape = {.segments = {Seg(3, 4, 4, 256, 0)}}},
      {.shape = {.segments = {Seg(3, 1, 8, 256, 1)}}},
  };
  for (const Encoding e : {Encoding::kF32, Encoding::kQ8_0}) {
    SCOPED_TRACE(format::Info(e).name);
    Compare(*in, GdnFormats(e), cells, steps, kTolerance, kTolerance, {}, kGdnRanges);
  }
  // Without the folds its arithmetic assumes, refused.
  StageFacts bare = kFacts;
  bare.folds = {};
  EXPECT_FALSE(MakeInstance(Gdn(), bare, Site::kLayer, 0, 1));
  // Its contract: fixed state a verify never writes.
  const auto contract = in->def->state(*in);
  ASSERT_EQ(contract.tensors.size(), 2U);
  for (const auto& t : contract.tensors) {
    EXPECT_EQ(t.cells, model::CellRule::kFixed);
    EXPECT_EQ(t.rollback, model::Rollback::kCommit);
  }
}

// Runs a step of `graph` from `state` (by StateRef tensor and segment),
// returning the state after it and the step's saves' values.
struct Ran {
  std::map<std::pair<std::uint16_t, std::uint8_t>, std::vector<std::byte>> state;
  std::map<std::tuple<std::uint16_t, std::string, std::uint8_t>, std::vector<float>> saves;
};
Ran RunStep(const StepGraph& step, const planner::StepShape& shape,
            const std::map<std::pair<std::uint16_t, std::uint8_t>, std::vector<std::byte>>& state,
            const std::vector<std::vector<std::byte>>& weights,
            const std::vector<ir::TensorId>& leaves, const std::vector<float>& stream,
            const std::map<std::tuple<std::uint16_t, std::string, std::uint8_t>,
                           std::vector<float>>& saved = {},
            const std::map<InputKind, std::vector<float>>& lone = {}) {
  Ran out;
  auto exec = ref::Executor::Create(step.graph);
  EXPECT_TRUE(exec.has_value());
  if (!exec) return out;
  for (std::size_t r = 0; r < leaves.size(); ++r) {
    if (!leaves[r].valid()) continue;
    const ref::PartView part{.role = format::PartRole::kData, .bytes = weights[r]};
    EXPECT_TRUE(exec->Bind(leaves[r], std::span(&part, 1)));
  }
  for (const StepState& st : step.state) {
    const auto at = state.find({st.ref.tensor, st.ref.segment});
    if (at == state.end()) {
      std::vector<float> zeros(ir::Elements(step.graph.tensor(st.tensor).shape()), 0.0F);
      EXPECT_TRUE(exec->BindValues(st.tensor, zeros));
    } else {
      const ref::PartView part{.role = format::PartRole::kData, .bytes = at->second};
      EXPECT_TRUE(exec->Bind(st.tensor, std::span(&part, 1)));
    }
  }
  for (const StepInput& input : step.inputs) {
    if (input.kind == InputKind::kStream) {
      EXPECT_TRUE(exec->BindValues(input.tensor, stream));
      continue;
    }
    // A lone component's streams and side input.
    if (const auto at = lone.find(input.kind); at != lone.end()) {
      EXPECT_TRUE(exec->BindValues(input.tensor, at->second));
      continue;
    }
    if (input.kind == InputKind::kCommitRows) {
      const auto at = saved.find({input.component, input.what, input.segment});
      EXPECT_NE(at, saved.end()) << input.what;
      if (at == saved.end()) return out;
      EXPECT_TRUE(exec->BindValues(input.tensor, at->second));
      continue;
    }
    auto values = InputValues(input, shape, {});
    EXPECT_TRUE(values.has_value()) << (values ? "" : values.error());
    EXPECT_TRUE(exec->BindInts(input.tensor, *values));
  }
  const auto ran = exec->Run();
  EXPECT_TRUE(ran.has_value()) << (ran ? "" : ran.error().detail);
  for (const StepState& st : step.state) {
    const auto bytes = exec->Bytes(st.tensor);
    out.state[{st.ref.tensor, st.ref.segment}] = std::vector<std::byte>(bytes.begin(), bytes.end());
  }
  for (const CommitSave& save : step.saves) {
    auto values = exec->Values(save.tensor);
    EXPECT_TRUE(values.has_value());
    out.saves[{save.component, save.what, save.segment}] = *values;
  }
  return out;
}

TEST(ComponentsQwen38Test, AVerifysCommitLeavesTheStateItsKeptRowsWould) {
  // A one-layer stage of a hyper-connected Gated DeltaNet (the n-gram
  // block's commit: TheNgramBlocksCommitLeavesTheStateItsKeptRowsWould),
  // two owners: a prefill, then a verify of 4 and 3 rows
  // whose saves commit the first 2 and 3; the state equals a plain step of
  // those rows bit for bit (the reference executor's arithmetic is a row's
  // whatever the other rows).
  auto gdn = MakeInstance(Gdn(), kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(gdn.has_value());
  const auto formats = GdnFormats(Encoding::kF32);
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  std::vector<std::vector<std::byte>> weights(formats.size());
  for (std::size_t r = 0; r < formats.size(); ++r) {
    std::vector<float> v(ir::Elements(formats[r].shape));
    for (float& x : v) x = r == 5 ? -0.5F - (0.4F * u(rng)) : 0.3F * u(rng);
    weights[r].resize(v.size() * 4);
    ASSERT_TRUE(ref::EncodeRows(formats[r], 0, 0, v, weights[r]));
  }
  const std::array<std::uint64_t, 2> cells{1, 1};
  const auto build = [&](const planner::StepShape& shape) {
    auto g = BuildComponent(*gdn, shape, formats, cells);
    EXPECT_TRUE(g.has_value()) << (g ? "" : g.error());
    return std::move(*g);
  };
  const auto random_rows = [&](std::uint32_t rows) {
    std::vector<float> s(kWidth * rows);
    for (float& x : s) x = 2.0F * u(rng);
    return s;
  };
  const planner::StepShape prefill{.segments = {Seg(0, 6, 0, 256, 1), Seg(1, 3, 0, 256, 1)}};
  const auto first = build(prefill);
  const std::vector<float> prompt = random_rows(9);
  const Ran start = RunStep(first.step, prefill, {}, weights, first.weights, prompt);
  // The verify: rows past the prompt; no fixed state written, rows saved.
  const planner::StepShape verify{.phase = planner::Phase::kVerify,
                                  .segments = {Seg(0, 4, 6, 256, 4), Seg(1, 3, 3, 256, 3)}};
  const auto checked = build(verify);
  for (const StepState& st : checked.step.state) {
    EXPECT_TRUE(st.read);
    EXPECT_FALSE(st.written);
  }
  EXPECT_EQ(checked.step.saves.size(), 2U * 4U);  // qkv, conv, gate and beta a segment
  const std::vector<float> rows = random_rows(7);
  const Ran verified = RunStep(checked.step, verify, start.state, weights, checked.weights, rows);
  EXPECT_EQ(verified.state, start.state);
  // Its commit of the first 2 and all 3 rows, from the saves of those rows.
  spec::ModelSpec spec{.stages = {{.name = "target",
                                   .embedding = C("embedding.token", {}),
                                   .stream = {.streams = kStreams,
                                              .width = kWidth,
                                              .mixing = C("residual.hc", {P("mix", N("sigmoid"))})},
                                   .layers = {{.components = {Gdn()}}},
                                   .folds = Folds()}}};
  const auto layout = StageLayout(spec, 0, 64, 8);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  const planner::StepShape kept{.segments = {Seg(0, 2, 6, 256, 0), Seg(1, 3, 3, 256, 0)}};
  auto commit = BuildCommit(spec, 0, kept, {.state = &*layout});
  ASSERT_TRUE(commit.has_value()) << commit.error();
  EXPECT_EQ(Count(commit->graph, ir::Op::kGatedDeltaNet), 2U);
  for (const StepState& st : commit->state) EXPECT_TRUE(st.written);
  // The saves' first kept rows, by (the spec's component, save, owner).
  const auto instances = InstancesOf(spec, 0);
  ASSERT_TRUE(instances.has_value());
  const std::uint16_t component = instances->layers[0][0].index;
  std::map<std::tuple<std::uint16_t, std::string, std::uint8_t>, std::vector<float>> saved;
  for (const auto& [key, values] : verified.saves) {
    const auto& [lone, what, segment] = key;
    EXPECT_EQ(lone, gdn->index);
    const std::uint64_t width = what == "gate" || what == "beta" ? 4 : (2 * 2 * 16) + (4 * 16);
    const std::uint32_t keep = kept.segments.at(segment).rows;
    saved[{component, what, segment}] = std::vector<float>(
        values.begin(), values.begin() + static_cast<std::ptrdiff_t>(width * keep));
  }
  const Ran committed = RunStep(*commit, kept, start.state, weights, {}, {}, saved);
  // A plain step of the kept rows.
  const planner::StepShape plain{.segments = {Seg(0, 2, 6, 256, 2), Seg(1, 3, 3, 256, 3)}};
  const auto direct = build(plain);
  std::vector<float> kept_rows(rows.begin(), rows.begin() + (2 * kWidth));
  kept_rows.insert(kept_rows.end(), rows.begin() + (4 * kWidth), rows.end());
  const Ran want = RunStep(direct.step, plain, start.state, weights, direct.weights, kept_rows);
  EXPECT_EQ(committed.state, want.state);
  // A stage with nothing to commit is refused.
  spec::ModelSpec none = spec;
  none.stages[0].layers = {{.components = {C("hc.combine", {})}}};
  const auto plain_layout = StageLayout(none, 0, 64, 8);
  ASSERT_TRUE(plain_layout.has_value());
  EXPECT_FALSE(BuildCommit(none, 0, kept, {.state = &*plain_layout}));
}

// ---------------------------------------------------------------- QSA

ComponentRef Qsa(std::string indexer, std::int64_t budget = 8) {
  return C("attention.qsa",
           {P("eps", R(1e-6)), P("heads", I(4)), P("kv_heads", I(2)), P("head_dim", I(32)),
            P("rope_dims", I(16)), P("rope_base", R(10000.0)), P("rope_sections", L({3, 3, 2, 0})),
            P("indexer_heads", I(2)), P("indexer_dim", I(32)), P("ratio", I(4)),
            P("budget", I(budget)), P("indexer", N(std::move(indexer)))});
}
// q, k, v, output, q_norm, k_norm, indexer_qk, indexer_q, indexer_k, its norms.
std::vector<format::TensorFormat> QsaFormats(Encoding e) {
  return {Matrix(e, kWidth, 256),
          Matrix(e, kWidth, 64),
          Matrix(e, kWidth, 64),
          Matrix(e, 128, kWidth),
          Vector(32),
          Vector(32),
          Matrix(e, kWidth, 96),
          Matrix(e, kWidth, 64),
          Matrix(e, kWidth, 32),
          Vector(32),
          Vector(32)};
}

TEST(ComponentsQwen38Test, QsaAttentionMatchesItsReferenceWithAndWithoutSelection) {
  for (const char* indexer : {"fused", "split"}) {
    SCOPED_TRACE(indexer);
    auto in = MakeInstance(Qsa(indexer), kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value()) << in.error();
    // 64 cells (16 blocks of 4); a budget of 8 keeps 11 cells: reads of 8
    // cells keep every one, of 64 select.
    const std::array<std::uint64_t, 4> cells{64, 64, 64, 16};
    const std::vector<Step> steps{
        {.shape = {.segments = {Seg(0, 5, 0, 8, 5), Seg(3, 3, 0, 8, 3)}}},
        {.shape = {.segments = {Seg(0, 23, 5, 64, 23)}}},
        {.shape = {.segments = {Seg(0, 1, 28, 64, 1), Seg(3, 6, 3, 64, 6)}}},
        {.shape = {.segments = {Seg(3, 30, 9, 64, 0)}}},
        {.shape = {.segments = {Seg(3, 2, 39, 64, 2)}}},
    };
    Compare(*in, QsaFormats(Encoding::kF32), cells, steps, kTolerance, kF16State);
    // A selecting step's attention promises its kept cells; every step
    // keeps the block keys' cache (its complete blocks' pooled keys) and a
    // selecting one reads it.
    auto g = BuildComponent(*in, steps[2].shape, QsaFormats(Encoding::kF32), cells);
    ASSERT_TRUE(g.has_value()) << g.error();
    std::size_t sparse = 0;
    for (const ir::Node& n : g->step.graph.nodes()) {
      if (n.key.op != ir::Op::kAttention) continue;
      const auto p = n.Params<ir::AttnParams>();
      sparse += p.sparse_any && p.n_kv_max == 11 ? 1 : 0;
    }
    EXPECT_EQ(sparse, 2U);
    EXPECT_EQ(Count(g->step.graph, ir::Op::kTopK), 2U);
    for (const StepState& st : g->step.state) {
      EXPECT_TRUE(st.written);
      if (st.ref.tensor == 3) EXPECT_TRUE(st.read);
    }
    // The selection's mask is built by a store into an activation: the
    // step's own, not state.
    EXPECT_TRUE(CheckStateWrites(g->step, 0));
  }
}

// ---------------------------------------------------------------- the MoE

ComponentRef QwenMoe(bool stacked, bool scales) {
  return C(
      "ffn.moe",
      {P("placement", N("none")), P("eps", R(1e-6)), P("activation", N("silu")), P("hidden", I(24)),
       P("experts", I(8)), P("used", I(3)), P("expert_hidden", I(16)),
       P("router", N("softmax_topk")), P("router_input", N("stream")), P("normalize", B(true)),
       P("stacked", B(stacked)), P("projection_scales", B(scales)), P("shared_gate", B(true))});
}

TEST(ComponentsQwen38Test, TheMoeInsideHyperConnectionsMatchesItsReference) {
  for (const auto& [stacked, scales] :
       {std::pair{false, true}, std::pair{true, true}, std::pair{false, false}}) {
    SCOPED_TRACE(std::format("stacked {} scales {}", stacked, scales));
    auto in = MakeInstance(QwenMoe(stacked, scales), kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value()) << in.error();
    std::vector<format::TensorFormat> f(19, Vector(kWidth));
    f[1] = Matrix(Encoding::kF16, kWidth, 24);     // the shared expert's gate
    f[2] = Matrix(Encoding::kF16, kWidth, 24);     // up
    f[3] = Matrix(Encoding::kF16, 24, kWidth);     // down
    f[5] = Matrix(Encoding::kF32, kWidth, 8);      // router
    f[8] = Matrix(Encoding::kF32, kWidth, 32, 8);  // gate_up experts
    f[9] = Matrix(Encoding::kF32, kWidth, 16, 8);  // gate experts
    f[10] = Matrix(Encoding::kF32, kWidth, 16, 8);
    f[11] = Matrix(Encoding::kF32, 16, kWidth, 8);
    for (const std::size_t r : {15U, 16U, 17U}) f[r] = Vector(8);
    f[18] = Vector(kWidth);  // shared gate
    const std::map<std::uint16_t, std::pair<float, float>> ranges{
        {15, {0.5F, 0.5F}}, {16, {0.5F, 0.5F}}, {17, {0.5F, 0.5F}}};
    Compare(*in, f, {}, RowSteps(), 1e-5, 1e-5, {}, ranges);
  }
}

// ---------------------------------------------------------------- the n-gram embedding

TEST(ComponentsQwen38Test, TheNgramRowsAreLlamaCppsHash) {
  const auto& p = model::Qwen38Flash();
  // A 64-bit multiplier past 2^63 is a negative integer (GGUF's I64).
  std::vector<std::int64_t> mult{static_cast<std::int64_t>(0x9E3779B97F4A7C15ULL), 7,
                                 1099511628211LL};
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> vocab;
  for (std::int64_t h = 0; h < 16; ++h) {
    offsets.push_back(h * 20000096);
    vocab.push_back(20000096 - (h * 8));
  }
  const auto hash = model::CheckQwen38PleHash(p, mult, offsets, vocab, fx::kTableRows);
  ASSERT_TRUE(hash.has_value()) << hash.error();
  auto in =
      MakeInstance(C("side.ngram", {P("ngram", I(3)), P("heads_per_ngram", I(8)), P("row", I(160)),
                                    P("eos", I(p.ple_eos)), P("multipliers", L(mult)),
                                    P("offsets", L(offsets)), P("vocab", L(vocab))}),
                   kFacts, Site::kSide, 0, 1);
  ASSERT_TRUE(in.has_value()) << in.error();
  ASSERT_EQ(in->def->side_count(*in), 16U);
  const std::vector<std::int32_t> history{17, p.ple_eos, 5, 248000, 3, 3, 9};
  std::vector<std::int32_t> got(16);
  for (std::uint32_t pos = 0; pos < history.size(); ++pos) {
    ASSERT_TRUE(in->def->side_rows(*in, history, pos, got));
    EXPECT_EQ(got, model::Qwen38PleRows(p, *hash, history, pos)) << pos;
  }
  EXPECT_FALSE(in->def->side_rows(*in, history, 7, got));
}

TEST(ComponentsQwen38Test, TheNgramSideInputAndItsBlockMatchTheirReferences) {
  // The table's lookup (rows of 32 values: whole Q8_0 blocks).
  const std::vector<format::TensorFormat> table{Matrix(Encoding::kQ8_0, 32, 80)};
  auto ngram32 =
      MakeInstance(C("side.ngram", {P("ngram", I(3)), P("heads_per_ngram", I(1)), P("row", I(32)),
                                    P("eos", I(0)), P("multipliers", L({3, 5, 7})),
                                    P("offsets", L({0, 40})), P("vocab", L({40, 40}))}),
                   kHcFacts, Site::kSide, 0, 1);
  ASSERT_TRUE(ngram32.has_value());
  Compare(*ngram32, table, {}, RowSteps(), kTolerance, kTolerance, {.table_rows = 80});
  // Heads past the table are refused when built.
  const std::vector<format::TensorFormat> small{Matrix(Encoding::kQ8_0, 32, 60)};
  EXPECT_FALSE(BuildComponent(*ngram32, RowSteps()[0].shape, small, {}));
  // The block: its key against the streams, the gated value, the
  // convolution over its history; two owners, carried.
  auto ple = MakeInstance(C("residual.ple", {P("eps", R(1e-6)), P("input", N("ngram")),
                                             P("conv", I(4)), P("dilation", I(3))}),
                          kHcFacts, Site::kLayer, 1, 2);
  ASSERT_TRUE(ple.has_value()) << ple.error();
  const std::uint64_t hw = kHcWidth * kStreams;
  const std::vector<format::TensorFormat> pf{Matrix(Encoding::kF16, 64, hw),
                                             Matrix(Encoding::kF16, 64, kHcWidth),
                                             Vector(hw),
                                             Vector(hw),
                                             Vector(hw),
                                             Matrix(Encoding::kF32, 4, hw)};
  const std::vector<Step> steps{
      {.shape = {.segments = {Seg(0, 5, 0, 256, 5), Seg(3, 2, 0, 256, 2)}}},
      {.shape = {.segments = {Seg(0, 1, 5, 256, 1)}}},
      {.shape = {.segments = {Seg(3, 11, 2, 256, 11), Seg(0, 1, 6, 256, 1)}}},
  };
  const std::array<std::uint64_t, 1> cells{1};
  Compare(*ple, pf, cells, steps, kTolerance, kTolerance,
          {.inputs = {.residual = true, .side = 64}}, {{5, {-0.6F, 0.6F}}});
}

TEST(ComponentsQwen38Test, TheNgramBlocksCommitLeavesTheStateItsKeptRowsWould) {
  // The n-gram block's convolution history (dilated: 9 rows) over two
  // owners: a prefill, then a verify of 4 and 3 rows whose saves commit
  // the first 1 and all 3; the history equals a plain step of those rows'
  // bit for bit, and the verify leaves it as it was.
  const ComponentRef ple_ref = C("residual.ple", {P("eps", R(1e-6)), P("input", N("ngram")),
                                                  P("conv", I(4)), P("dilation", I(3))});
  auto ple = MakeInstance(ple_ref, kHcFacts, Site::kLayer, 0, 2);
  ASSERT_TRUE(ple.has_value()) << ple.error();
  const std::uint64_t hw = kHcWidth * kStreams;
  constexpr std::uint64_t kSideWidth = 64;
  const std::vector<format::TensorFormat> pf{Matrix(Encoding::kF32, kSideWidth, hw),
                                             Matrix(Encoding::kF32, kSideWidth, kHcWidth),
                                             Vector(hw),
                                             Vector(hw),
                                             Vector(hw),
                                             Matrix(Encoding::kF32, 4, hw)};
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  std::vector<std::vector<std::byte>> weights(pf.size());
  for (std::size_t r = 0; r < pf.size(); ++r) {
    std::vector<float> v(ir::Elements(pf[r].shape));
    for (float& x : v) x = pf[r].shape.rank == 1 ? 1.0F + (0.25F * u(rng)) : 0.3F * u(rng);
    weights[r].resize(v.size() * 4);
    ASSERT_TRUE(ref::EncodeRows(pf[r], 0, 0, v, weights[r]));
  }
  const std::array<std::uint64_t, 1> cells{1};
  const ComponentInputs inputs{.residual = true, .side = kSideWidth};
  const auto build = [&](const planner::StepShape& shape) {
    auto g = BuildComponent(*ple, shape, pf, cells, inputs);
    EXPECT_TRUE(g.has_value()) << (g ? "" : g.error());
    return std::move(*g);
  };
  const auto random = [&](std::uint64_t width, std::uint32_t rows) {
    std::vector<float> s(width * rows);
    for (float& x : s) x = 2.0F * u(rng);
    return s;
  };
  const auto lone = [](std::vector<float> residual, std::vector<float> side) {
    return std::map<InputKind, std::vector<float>>{{InputKind::kResidual, std::move(residual)},
                                                   {InputKind::kSide, std::move(side)}};
  };
  // Prefills of 12 and 5 rows (past the history and within it).
  const planner::StepShape prefill{.segments = {Seg(0, 12, 0, 256, 1), Seg(1, 5, 0, 256, 1)}};
  const auto first = build(prefill);
  // (The block returns its stream as it is: an input bound, never read.)
  const Ran start = RunStep(first.step, prefill, {}, weights, first.weights, random(kHcWidth, 17),
                            {}, lone(random(hw, 17), random(kSideWidth, 17)));
  // The verify: no history written, the rows' convolution inputs saved.
  const planner::StepShape verify{.phase = planner::Phase::kVerify,
                                  .segments = {Seg(0, 4, 12, 256, 4), Seg(1, 3, 5, 256, 3)}};
  const auto checked = build(verify);
  for (const StepState& st : checked.step.state) {
    EXPECT_TRUE(st.read);
    EXPECT_FALSE(st.written);
  }
  ASSERT_EQ(checked.step.saves.size(), 2U);
  const std::vector<float> residual = random(hw, 7);
  const std::vector<float> side = random(kSideWidth, 7);
  const Ran verified = RunStep(checked.step, verify, start.state, weights, checked.weights,
                               random(kHcWidth, 7), {}, lone(residual, side));
  EXPECT_EQ(verified.state, start.state);
  // Its commit, from the saves of the kept rows.
  spec::ModelSpec spec{
      .stages = {
          {.name = "target",
           .embedding = C("embedding.token", {}),
           .stream = {.streams = kStreams,
                      .width = kHcWidth,
                      .mixing = C("residual.hc", {P("mix", N("sigmoid"))})},
           .side = {{.name = "ngram",
                     .source = C("side.ngram",
                                 {P("ngram", I(3)), P("heads_per_ngram", I(1)), P("row", I(32)),
                                  P("eos", I(0)), P("multipliers", L({3, 5, 7})),
                                  P("offsets", L({0, 40})), P("vocab", L({40, 40}))}),
                     .layers = {0}}},
           .layers = {{.components = {ple_ref}}},
           .folds = Folds()}}};
  const auto layout = StageLayout(spec, 0, 64, 8);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  const planner::StepShape kept{.segments = {Seg(0, 1, 12, 256, 0), Seg(1, 3, 5, 256, 0)}};
  auto commit = BuildCommit(spec, 0, kept, {.state = &*layout});
  ASSERT_TRUE(commit.has_value()) << commit.error();
  for (const StepState& st : commit->state) EXPECT_TRUE(st.written);
  const auto instances = InstancesOf(spec, 0);
  ASSERT_TRUE(instances.has_value());
  const std::uint16_t component = instances->layers[0][0].index;
  std::map<std::tuple<std::uint16_t, std::string, std::uint8_t>, std::vector<float>> saved;
  for (const auto& [key, values] : verified.saves) {
    const auto& [index, what, segment] = key;
    EXPECT_EQ(index, ple->index);
    EXPECT_EQ(what, "ple");
    const std::uint32_t keep = kept.segments.at(segment).rows;
    saved[{component, what, segment}] =
        std::vector<float>(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(hw * keep));
  }
  const Ran committed = RunStep(*commit, kept, start.state, weights, {}, {}, saved);
  // A plain step of the kept rows: owner 0's first, owner 1's three.
  const planner::StepShape plain{.segments = {Seg(0, 1, 12, 256, 1), Seg(1, 3, 5, 256, 3)}};
  const auto direct = build(plain);
  const auto rows_of = [&](const std::vector<float>& all, std::uint64_t width) {
    std::vector<float> out(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(width));
    out.insert(out.end(), all.begin() + static_cast<std::ptrdiff_t>(4 * width), all.end());
    return out;
  };
  const Ran want =
      RunStep(direct.step, plain, start.state, weights, direct.weights, random(kHcWidth, 4), {},
              lone(rows_of(residual, hw), rows_of(side, kSideWidth)));
  EXPECT_EQ(committed.state, want.state);
  EXPECT_NE(committed.state, start.state);
}

TEST(ComponentsQwen38Test, TheMtpDraftersInputMatchesItsReference) {
  auto in = MakeInstance(C("embedding.mtp", {P("eps", R(1e-6)), P("tap", N("streams"))}), kHcFacts,
                         Site::kEmbedding, 0, 0);
  ASSERT_TRUE(in.has_value()) << in.error();
  const std::vector<format::TensorFormat> f{Matrix(Encoding::kF16, kHcWidth, 40), Vector(kHcWidth),
                                            Matrix(Encoding::kF16, kHcWidth, kHcWidth),
                                            Vector(kHcWidth * kStreams),
                                            Matrix(Encoding::kF16, kHcWidth, kHcWidth)};
  const std::array<std::uint64_t, 1> cells{9};
  Compare(*in, f, cells, RowSteps(), kTolerance, kTolerance);
  // Its tap: every row's streams, a cell a chunk row, written by its target.
  const auto contract = in->def->state(*in);
  ASSERT_EQ(contract.tensors.size(), 1U);
  EXPECT_EQ(contract.tensors[0].cells, model::CellRule::kPerRow);
  EXPECT_TRUE(contract.tensors[0].target_writes);
  EXPECT_EQ(contract.tensors[0].cell_bytes, kHcWidth * kStreams * 4);
}

TEST(ComponentsQwen38Test, ASelectedHeadsTokensAreItsRowsIds) {
  auto head = MakeInstance(C("head.linear", {P("selected", B(true))}), kFacts, Site::kOutput, 0, 4);
  ASSERT_TRUE(head.has_value()) << head.error();
  const std::vector<format::TensorFormat> hf{Matrix(Encoding::kF32, kWidth, 12), Vector(1),
                                             ir::Dense(Encoding::kI32, ir::ShapeOf({1, 12}))};
  const std::vector<Step> greedy{
      {.shape = {.segments = {Seg(0, 3, 0, 256, 1, true), Seg(1, 2, 7, 256, 1, true)}}}};
  Compare(*head, hf, {}, greedy, 1e-5, 1e-5);
}

// ---------------------------------------------------------------- specs

struct Imports {
  spec::Source modelopt_cutlass, modelopt_ggml, gguf, mtp;
};
const Imports& Sources() {
  static const Imports imports = [] {
    const auto& p = model::Qwen38Flash();
    const auto& target = spec::mappings::Qwen38Mapping();
    const auto& mtp = spec::mappings::Qwen38MtpMapping();
    return Imports{
        .modelopt_cutlass =
            fx::SourceOf("qwen4exp", "safetensors", std::string(fx::kModelOptConverter),
                         fx::Resources(p, fx::Form::kCutlass), target),
        .modelopt_ggml =
            fx::SourceOf("qwen4exp", "safetensors", std::string(fx::kModelOptConverter),
                         fx::Resources(p, fx::Form::kGgmlExperts), target),
        .gguf = fx::SourceOf("qwen4exp", "gguf", std::string(fx::kGgufConverter),
                             fx::Resources(p, fx::Form::kGguf), target),
        .mtp = fx::SourceOf("qwen4exp-mtp", "safetensors", std::string(fx::kModelOptConverter),
                            fx::MtpResources(p), mtp)};
  }();
  return imports;
}

model::Qwen38PleHash Hash() {
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> vocab;
  for (std::int64_t h = 0; h < 16; ++h) {
    offsets.push_back(h * 20000096);
    vocab.push_back(20000096 - (h * 8));
  }
  return *model::CheckQwen38PleHash(model::Qwen38Flash(), std::vector<std::int64_t>{3, 5, 7},
                                    offsets, vocab, fx::kTableRows);
}

model::Qwen38Binding BindingOf(fx::Form form) {
  const auto& p = model::Qwen38Flash();
  model::Qwen38Binding b;
  b.layers.resize(p.layers);
  b.format = form == fx::Form::kGguf ? model::Qwen38Format::kGguf : model::Qwen38Format::kModelOpt;
  b.experts =
      form == fx::Form::kCutlass ? model::Qwen38Experts::kCutlass : model::Qwen38Experts::kGgml;
  return b;
}

std::expected<spec::ModelSpec, std::string> TargetSpec(const spec::Source& source, fx::Form form) {
  auto values = spec::mappings::Qwen38V0Values(model::Qwen38Flash(), BindingOf(form), Hash());
  if (!values) return std::unexpected(values.error());
  return spec::Synthesize(spec::mappings::Qwen38Mapping(), source, *values);
}

std::expected<spec::ModelSpec, std::string> WithMtp(const spec::ModelSpec& target) {
  model::Qwen38MtpBinding mb;
  auto values = spec::mappings::Qwen38MtpV0Values(model::Qwen38Flash(), mb);
  if (!values) return std::unexpected(values.error());
  return spec::ComposeDrafter(target, spec::mappings::Qwen38Mapping(),
                              spec::mappings::Qwen38MtpMapping(), Sources().mtp, *values);
}

TEST(ComponentsQwen38Test, RepeatableMtpContractsCoexistWithTheOriginalLayout) {
  auto target = TargetSpec(Sources().modelopt_cutlass, fx::Form::kCutlass);
  ASSERT_TRUE(target);
  auto composed = WithMtp(*target);
  ASSERT_TRUE(composed);
  auto original = StageLayout(*composed, 1, 8192, 512);
  ASSERT_TRUE(original);
  auto old_bounds = model::DraftBounds(*original);
  ASSERT_TRUE(old_bounds);
  EXPECT_FALSE(old_bounds->has_value());
  auto& drafter = composed->stages[1];
  drafter.embedding->version = 2;
  drafter.embedding->params.push_back(P("token_offset", I(1)));
  std::ranges::sort(drafter.embedding->params, {}, &Param::name);
  // Declaring only the input's policy cannot qualify an arbitrary cache.
  EXPECT_FALSE(StageLayout(*composed, 1, 8192, 512));
  for (auto& layer : drafter.layers)
    for (auto& component : layer.components)
      if (component.kind == "attention.qsa") component.version = 2;
  auto replay = StageLayout(*composed, 1, 8192, 512);
  ASSERT_TRUE(replay) << (replay ? "" : replay.error());
  auto bounds = model::DraftBounds(*replay);
  ASSERT_TRUE(bounds);
  ASSERT_TRUE(bounds->has_value());
  EXPECT_EQ((*bounds)->relation, model::DraftRelation::kTokenAfterCache);
  EXPECT_EQ((*bounds)->max_rows, 512U);
  EXPECT_EQ((*bounds)->carry, 1U);
  EXPECT_NE(model::KeptLayoutId({}, std::span(&*original, 1)),
            model::KeptLayoutId({}, std::span(&*replay, 1)));
  ASSERT_NE(Find("embedding.mtp", 1), nullptr);
  ASSERT_NE(Find("attention.qsa", 1), nullptr);
  EXPECT_NE(Find("embedding.mtp", 2), nullptr);
  EXPECT_NE(Find("attention.qsa", 2), nullptr);
  EXPECT_EQ(replay->items.size(), original->items.size() + 1);
}

TEST(ComponentsQwen38Test, ServingReplayMappingDeclaresDurableShiftWithoutChangingV1) {
  auto target = TargetSpec(Sources().modelopt_cutlass, fx::Form::kCutlass);
  ASSERT_TRUE(target);
  model::Qwen38MtpBinding binding;
  auto values = spec::mappings::Qwen38MtpV0Values(model::Qwen38Flash(), binding);
  ASSERT_TRUE(values);
  const auto& original = spec::mappings::Qwen38MtpMapping();
  const auto& replay = spec::mappings::Qwen38MtpReplayMapping();
  EXPECT_EQ(original.version, 1U);
  EXPECT_EQ(replay.version, 2U);
  auto made = spec::ComposeDrafter(*target, spec::mappings::Qwen38Mapping(), replay, Sources().mtp,
                                   *values);
  ASSERT_TRUE(made) << made.error();
  auto layout = StageLayout(*made, 1, 512, 16);
  ASSERT_TRUE(layout) << layout.error();
  auto bounds = model::DraftBounds(*layout);
  ASSERT_TRUE(bounds);
  ASSERT_TRUE(bounds->has_value());
  EXPECT_EQ((*bounds)->relation, model::DraftRelation::kTokenAfterCache);
  EXPECT_EQ((*bounds)->carry, 1U);
  auto old = WithMtp(*target);
  ASSERT_TRUE(old);
  auto old_layout = StageLayout(*old, 1, 512, 16);
  ASSERT_TRUE(old_layout);
  EXPECT_NE(model::KeptLayoutId({}, std::span(&*old_layout, 1)),
            model::KeptLayoutId({}, std::span(&*layout, 1)));
}

// Each layer's component kinds.
std::vector<std::vector<std::string>> Kinds(const spec::Stage& s) {
  std::vector<std::vector<std::string>> out;
  for (const spec::Layer& l : s.layers) {
    auto& kinds = out.emplace_back();
    for (const spec::ComponentRef& c : l.components) kinds.push_back(c.kind);
  }
  return out;
}

TEST(ComponentsQwen38Test, EveryImportsTensorsMakeOneSpecStructure) {
  const auto& p = model::Qwen38Flash();
  struct Case {
    const spec::Source* source;
    fx::Form form;
  };
  std::optional<std::vector<std::vector<std::string>>> kinds;
  for (const Case& c : {Case{&Sources().modelopt_cutlass, fx::Form::kCutlass},
                        Case{&Sources().modelopt_ggml, fx::Form::kGgmlExperts},
                        Case{&Sources().gguf, fx::Form::kGguf}}) {
    SCOPED_TRACE(static_cast<int>(c.form));
    auto spec = TargetSpec(*c.source, c.form);
    ASSERT_TRUE(spec.has_value()) << spec.error();
    const auto valid = ValidateSpec(*spec, c.source->tensors);
    ASSERT_TRUE(valid.has_value()) << valid.error();
    const spec::Stage& s = spec->stages.front();
    EXPECT_EQ(s.stream.streams, p.hc);
    EXPECT_EQ(s.stream.width, p.width);
    ASSERT_TRUE(s.stream.mixing.has_value());
    ASSERT_EQ(s.side.size(), 1U);
    EXPECT_EQ(s.side[0].layers, std::vector<std::uint32_t>{p.ple_layer});
    ASSERT_EQ(s.layers.size(), p.layers);
    // Every import's folds: (1 + w) norms, A, tiled value heads.
    EXPECT_EQ(s.folds, (std::vector<std::string>{"gdn.value_heads_tiled", "norm.plus_one",
                                                 "ssm.a_neg_exp"}));
    if (!kinds) {
      kinds = Kinds(s);
    } else {
      EXPECT_EQ(Kinds(s), *kinds);
    }
    // The forms: the ModelOpt import's MXFP8 matrices (codes and scales
    // bound), a GGUF's whole.
    const auto& gdn = s.layers[0].components[1];
    ASSERT_EQ(gdn.kind, "attention.gdn");
    const auto qkv_parts = std::ranges::count_if(
        gdn.weights, [](const spec::WeightBinding& w) { return w.role == "qkv"; });
    EXPECT_EQ(qkv_parts, c.form == fx::Form::kGguf ? 1 : 2);
    const auto& qsa = s.layers[3].components[1];
    ASSERT_EQ(qsa.kind, "attention.qsa");
    EXPECT_EQ(std::get<std::string>(qsa.Find("indexer")->value),
              c.form == fx::Form::kGguf ? "split" : "fused");
    const auto& moe = s.layers[0].components[4];
    ASSERT_EQ(moe.kind, "ffn.moe");
    EXPECT_EQ(std::get<bool>(moe.Find("stacked")->value), c.form == fx::Form::kCutlass);
    EXPECT_EQ(std::get<bool>(moe.Find("projection_scales")->value), c.form != fx::Form::kGguf);
  }
  ASSERT_TRUE(kinds.has_value());
  // Layer 1 (the n-gram layer) first reads the side input; a QSA layer every fourth.
  EXPECT_EQ((*kinds)[1].front(), "residual.ple");
  EXPECT_EQ((*kinds)[0], (std::vector<std::string>{"hc.mix", "attention.gdn", "hc.combine",
                                                   "hc.mix", "ffn.moe", "hc.combine"}));
  EXPECT_EQ((*kinds)[3][1], "attention.qsa");
  // Without the converter's folds (a checkpoint read as it is), the gated
  // delta rule refuses the spec.
  spec::Source bare = Sources().modelopt_cutlass;
  bare.converter.clear();
  auto unfolded = TargetSpec(bare, fx::Form::kCutlass);
  ASSERT_TRUE(unfolded.has_value()) << unfolded.error();
  EXPECT_TRUE(unfolded->stages.front().folds.empty());
  EXPECT_FALSE(ValidateSpec(*unfolded, bare.tensors));
}

TEST(ComponentsQwen38Test, TheModelOptFoldsFollowTheImportersPartOfItsIdentity) {
  // The ModelOpt importer is one '+'-joined part of the converter's
  // identity: the targets' and the MTP drafter's (followed by "+mtp"); a
  // look-alike part, or its name inside another part, is not it.
  constexpr std::string_view kStem = "artifact-layout/import_m3.py m3-1+layout-a0d1980a9eddd1ad";
  const std::vector<std::string> folds{"gdn.value_heads_tiled", "norm.plus_one", "ssm.a_neg_exp"};
  const auto sorted = [](std::vector<std::string> v) {
    std::ranges::sort(v);
    return v;
  };
  const auto target_folds = [&](std::string converter) {
    spec::Source s = Sources().modelopt_cutlass;
    s.converter = std::move(converter);
    auto made = TargetSpec(s, fx::Form::kCutlass);
    EXPECT_TRUE(made.has_value()) << made.error();
    return made ? sorted(made->stages.front().folds) : std::vector<std::string>{"refused"};
  };
  EXPECT_EQ(target_folds(std::format("{}+modelopt_qwen38-031ed8750c9da55d", kStem)), folds);
  EXPECT_EQ(target_folds(std::format("{}+modelopt_qwen38-bd8e3954615dbe97+mtp", kStem)), folds);
  for (const std::string_view other :
       {"+xmodelopt_qwen38-031ed8750c9da55d", "-modelopt_qwen38-031ed8750c9da55d",
        "+mtp-modelopt_qwen38-bd8e3954615dbe97", "+modelopt_qwen38", "+modelopt_qwen3-031ed875"}) {
    EXPECT_TRUE(target_folds(std::format("{}{}", kStem, other)).empty()) << other;
  }
  // The drafter's own import: its folds are its own stage's, the target's
  // (none) unchanged.
  spec::Source bare = Sources().modelopt_cutlass;
  bare.converter.clear();
  auto target = TargetSpec(bare, fx::Form::kCutlass);
  ASSERT_TRUE(target.has_value()) << target.error();
  ASSERT_TRUE(target->stages.front().folds.empty());
  model::Qwen38MtpBinding mb;
  auto values = spec::mappings::Qwen38MtpV0Values(model::Qwen38Flash(), mb);
  ASSERT_TRUE(values.has_value()) << values.error();
  for (const auto& [converter, want] :
       {std::pair{std::format("{}+modelopt_qwen38-bd8e3954615dbe97+mtp", kStem), folds},
        std::pair{std::format("{}+xmodelopt_qwen38-bd8e3954615dbe97+mtp", kStem),
                  std::vector<std::string>{}}}) {
    spec::Source mtp = Sources().mtp;
    mtp.converter = converter;
    auto composed = spec::ComposeDrafter(*target, spec::mappings::Qwen38Mapping(),
                                         spec::mappings::Qwen38MtpMapping(), mtp, *values);
    ASSERT_TRUE(composed.has_value()) << composed.error();
    ASSERT_EQ(composed->stages.size(), 2U);
    EXPECT_TRUE(composed->stages[0].folds.empty()) << converter;
    EXPECT_EQ(sorted(composed->stages[1].folds), want) << converter;
  }
}

TEST(ComponentsQwen38Test, TheStageLayoutsHoldTodaysQwenState) {
  const auto& p = model::Qwen38Flash();
  auto spec = TargetSpec(Sources().modelopt_cutlass, fx::Form::kCutlass);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  auto composed = WithMtp(*spec);
  ASSERT_TRUE(composed.has_value()) << composed.error();
  // Each stage lists its own artifact's folds: the MTP import's are the
  // target's.
  ASSERT_EQ(composed->stages.size(), 2U);
  EXPECT_EQ(composed->stages[0].folds, Folds());
  EXPECT_EQ(composed->stages[1].folds, Folds());
  for (const auto& [context, rows] : {std::pair{8192U, 512U}, {262144U, 2048U}}) {
    auto want = model::Qwen38State(p, context, rows, false);
    ASSERT_TRUE(want.has_value());
    auto got = StageLayout(*composed, 0, context, rows);
    ASSERT_TRUE(got.has_value()) << got.error();
    // Today's tensors, each the same cells and bytes (the n-gram layer's
    // convolution history first in its layer here, last today).
    std::multiset<std::tuple<std::uint32_t, std::uint64_t, std::uint64_t>> mine;
    std::multiset<std::tuple<std::uint32_t, std::uint64_t, std::uint64_t>> theirs;
    for (const auto& item : got->items) mine.emplace(item.layer, item.cells, item.bytes);
    for (const auto& t : want->tensors) theirs.emplace(t.layer, t.ne1, t.bytes);
    EXPECT_EQ(mine, theirs);
    EXPECT_EQ(got->bytes, want->bytes);
    // The drafter's: its caches and the streams its target writes.
    auto mtp = model::Qwen38MtpStateOf(p, *want);
    ASSERT_TRUE(mtp.has_value());
    auto drafter = StageLayout(*composed, 1, context, rows);
    ASSERT_TRUE(drafter.has_value()) << drafter.error();
    EXPECT_EQ(drafter->bytes, mtp->bytes);
    const auto streams = std::ranges::find(drafter->items, std::string("streams"),
                                           [](const auto& i) { return i.contract.name; });
    ASSERT_NE(streams, drafter->items.end());
    EXPECT_EQ(streams->cells, mtp->hidden_rows);
    // Each tensor's cells and 256-byte-padded extent: K, V, the indexer's
    // keys, its block keys and the streams (here the streams first, today
    // last).
    const auto pad = [](std::uint64_t b) { return (b + 255) / 256 * 256; };
    std::multiset<std::pair<std::uint64_t, std::uint64_t>> drafter_mine;
    for (const auto& item : drafter->items) drafter_mine.emplace(item.cells, pad(item.bytes));
    const std::uint64_t blocks =
        (std::uint64_t{mtp->cells} + p.indexer_ratio - 1) / p.indexer_ratio;
    const std::multiset<std::pair<std::uint64_t, std::uint64_t>> drafter_theirs{
        {mtp->cells, mtp->v - mtp->k},
        {mtp->cells, mtp->indexer - mtp->v},
        {mtp->cells, mtp->blocks - mtp->indexer},
        {blocks, mtp->hidden - mtp->blocks},
        {mtp->hidden_rows, mtp->bytes - mtp->hidden}};
    EXPECT_EQ(drafter_mine, drafter_theirs);
  }
}

TEST(ComponentsQwen38Test, TheTargetWritesItsDraftersStreamsEveryRow) {
  auto target = TargetSpec(Sources().modelopt_cutlass, fx::Form::kCutlass);
  ASSERT_TRUE(target.has_value()) << target.error();
  auto spec = WithMtp(*target);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  const std::array<std::span<const BoundTensor>, 2> artifacts{Sources().modelopt_cutlass.tensors,
                                                              Sources().mtp.tensors};
  const auto valid = ValidateSpec(*spec, artifacts);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  // The target's tap: its last layer's streams.
  ASSERT_EQ(spec->stages[0].taps.size(), 1U);
  EXPECT_EQ(spec->stages[0].taps[0].point, spec::TapPoint::kStreams);
  EXPECT_EQ(spec->stages[0].taps[0].layer, model::Qwen38Flash().layers - 1);
  const auto t_layout = StageLayout(*spec, 0, 8192, 64);
  const auto d_layout = StageLayout(*spec, 1, 8192, 64);
  ASSERT_TRUE(t_layout.has_value() && d_layout.has_value());
  const std::array<const model::StateLayoutDerived*, 2> layouts{&*t_layout, &*d_layout};
  const Bindings target_bindings{.tensors = Sources().modelopt_cutlass.tensors,
                                 .state = &*t_layout,
                                 .stages = layouts,
                                 .artifacts = artifacts};
  const auto streams_stores = [](const StepGraph& step) {
    std::size_t n = 0;
    for (const StepState& st : step.state) n += st.tap && st.written ? 1 : 0;
    return n;
  };
  for (const auto& [shape, outputs] :
       {std::pair{planner::StepShape{.segments = {Seg(0, 7, 0, 256, 1), Seg(2, 3, 40, 256, 1)}},
                  true},
        std::pair{planner::StepShape{.segments = {Seg(0, 16, 0, 256, 0)}}, false}}) {
    auto step = BuildStep(*spec, 0, shape, target_bindings);
    ASSERT_TRUE(step.has_value()) << step.error();
    // Every owner's rows at their cells (the rows before the chunk first).
    EXPECT_EQ(streams_stores(*step), shape.segments.size());
    EXPECT_EQ(step->logits.valid(), outputs);
    for (const StepInput& in : step->inputs) {
      if (in.kind != InputKind::kStoreCells || !in.cells.per_row) continue;
      const auto cells = InputValues(in, shape, {});
      ASSERT_TRUE(cells.has_value());
      EXPECT_EQ(cells->front(), 1);
      EXPECT_EQ(cells->size(), shape.segments.at(in.segment).rows);
    }
    EXPECT_EQ(Count(step->graph, ir::Op::kGatedDeltaNet), 36U * shape.segments.size());
    // (State-only, the last layer runs whole: its streams are tapped.)
    EXPECT_EQ(Count(step->graph, ir::Op::kAttention), 12U * shape.segments.size());
    // The n-gram rows: one input of 16 a row.
    const auto side = std::ranges::count_if(
        step->inputs, [](const StepInput& in) { return in.kind == InputKind::kSideRows; });
    EXPECT_EQ(side, 1);
  }
  // The drafter's pass: its token's row beside a feature, its own layer, the
  // head's mix returning its streams as the next feature.
  const Bindings drafter_bindings{.tensors = Sources().mtp.tensors,
                                  .state = &*d_layout,
                                  .stages = layouts,
                                  .artifacts = artifacts};
  const planner::StepShape pass{
      .stage = 1, .phase = planner::Phase::kDraft, .segments = {Seg(0, 3, 40, 256, 1, true)}};
  auto draft = BuildStep(*spec, 1, pass, drafter_bindings);
  ASSERT_TRUE(draft.has_value()) << draft.error();
  ASSERT_TRUE(draft->features.valid());
  EXPECT_EQ(draft->graph.tensor(draft->features).ne(0), model::Qwen38Flash().hc_width());
  ASSERT_TRUE(draft->greedy.valid());
  // It never writes its own streams tap (its target does).
  for (const StepState& st : draft->state) {
    if (st.ref.tensor == 0) EXPECT_FALSE(st.written);
  }
  // A drafter's state-only catch-up pass writes its caches only: K, V,
  // the indexer's keys and its blocks' pooled keys.
  auto catch_up =
      BuildStep(*spec, 1, {.stage = 1, .segments = {Seg(0, 2, 40, 256, 0)}}, drafter_bindings);
  ASSERT_TRUE(catch_up.has_value()) << catch_up.error();
  EXPECT_EQ(Count(catch_up->graph, ir::Op::kCacheStore), 4U);
}

// ---------------------------------------------------------------- reads

TEST(ComponentsQwen38Test, ReadsReachAnyEarlierStageAndAreRefusedOtherwise) {
  auto target = TargetSpec(Sources().modelopt_cutlass, fx::Form::kCutlass);
  ASSERT_TRUE(target.has_value()) << target.error();
  auto spec = WithMtp(*target);
  ASSERT_TRUE(spec.has_value()) << spec.error();
  // A cascade: a second drafter reading the first's streams after its layer.
  spec::ModelSpec cascade = *spec;
  spec::Stage second = cascade.stages[1];
  second.name = "mtp2";
  second.artifact = 1;
  second.reads = {{.stage = "mtp", .kind = spec::ReadKind::kTap, .name = "streams"}};
  cascade.stages[1].taps = {{.name = "streams", .layer = 0, .point = spec::TapPoint::kStreams}};
  cascade.stages.push_back(second);
  const std::array<std::span<const BoundTensor>, 2> artifacts{Sources().modelopt_cutlass.tensors,
                                                              Sources().mtp.tensors};
  ASSERT_TRUE(spec::Validate(cascade).has_value());
  const auto valid = ValidateSpec(cascade, artifacts);
  ASSERT_TRUE(valid.has_value()) << valid.error();
  // The first drafter's step writes the second's streams, every row.
  const auto t = StageLayout(cascade, 0, 4096, 16);
  const auto d1 = StageLayout(cascade, 1, 4096, 16);
  const auto d2 = StageLayout(cascade, 2, 4096, 16);
  ASSERT_TRUE(t && d1 && d2);
  const std::array<const model::StateLayoutDerived*, 3> layouts{&*t, &*d1, &*d2};
  auto step = BuildStep(
      cascade, 1, {.stage = 1, .segments = {Seg(0, 2, 40, 256, 1)}},
      {.tensors = Sources().mtp.tensors, .state = &*d1, .stages = layouts, .artifacts = artifacts});
  ASSERT_TRUE(step.has_value()) << step.error();
  const auto taps = std::ranges::count_if(
      step->state, [](const StepState& st) { return st.tap && st.written && st.ref.stage == 2; });
  EXPECT_EQ(taps, 1);
  // Refused: reading a later stage; one name of two stages; a tap of
  // another width.
  spec::ModelSpec later = cascade;
  later.stages[1].reads.push_back(
      {.stage = "mtp2", .kind = spec::ReadKind::kTap, .name = "streams"});
  std::ranges::sort(later.stages[1].reads, {},
                    [](const spec::Read& r) { return std::tuple(r.stage, r.kind, r.name); });
  later.stages[2].taps = {{.name = "streams", .layer = 0, .point = spec::TapPoint::kStreams}};
  EXPECT_FALSE(ValidateSpec(later, artifacts));
  spec::ModelSpec twice = cascade;
  twice.stages[2].reads.push_back(
      {.stage = "target", .kind = spec::ReadKind::kTap, .name = "streams"});
  std::ranges::sort(twice.stages[2].reads, {},
                    [](const spec::Read& r) { return std::tuple(r.stage, r.kind, r.name); });
  EXPECT_FALSE(ValidateSpec(twice, artifacts));
  spec::ModelSpec narrow = *spec;
  narrow.stages[0].taps[0].point = spec::TapPoint::kLayerOut;
  EXPECT_FALSE(ValidateSpec(narrow, artifacts));
}

}  // namespace
}  // namespace llmp::components
