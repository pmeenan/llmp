// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The DeepSeek V4 components (components/hc.cc's Sinkhorn mix, mqa.cc,
// moe.cc's DeepSeek routers, its clamped SwiGLU and side.hash, blocks.cc's
// Markov head, inject.cc) and the framework they need (a second carry,
// compressors' inputs, positions-basis block reads, recent rings, a block
// drafter's mask, the injection a target's steps run into its drafter's
// state), on the CPU:
//
// - each component's fragment, run by the reference executor over several
//   steps (prefill, decode, several owners, state-only) against its own
//   scalar reference, with the streams, carries and side input it reads;
// - the deepseek4 and dflash mappings over the fixture's tensor tables
//   (tests/unit/dsv4_fixture.h) and the profiles' values
//   (tests/support/families/dsv4.h): the spec's structure, the layouts
//   against the M3 runners' state in closed form, steps of every kind
//   built;
// - a small target and drafter through BuildStep on the reference
//   executor: the injection writes the drafter's ring from the target's
//   streams' means as an independent computation of it does.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/rotation_reference.h"
#include "components/step.h"
#include "dsv4_fixture.h"
#include "families/dsv4.h"
#include "format/encoding.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/reference/exec.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/dsv4.h"
#include "spec/spec.h"
#include "state_layouts.h"

namespace llmp::components {
namespace {

namespace ref = llmp::kernels::reference;
namespace fam = llmp::test_support::families;
namespace fx = llmp::test_support::dsv4;
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
format::TensorFormat Vector(std::uint64_t n) {
  format::TensorFormat f = ir::Dense(Encoding::kF32, ir::ShapeOf({n}));
  f.shape.rank = 1;  // a vector of one value too, as a GGUF's [1] is
  return f;
}

double Gap(std::span<const float> got, std::span<const double> want) {
  EXPECT_EQ(got.size(), want.size());
  double worst = 0.0;
  double scale = 1.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
      if (static_cast<double>(got[i]) == want[i]) continue;  // matching masked infinities
      return std::numeric_limits<double>::infinity();
    }
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
    scale = std::max(scale, std::fabs(want[i]));
  }
  return worst / scale;
}

struct Step {
  planner::StepShape shape;
  std::vector<std::int32_t> tokens{};
};

// What a lone component reads beside its stream.
struct Lone {
  ComponentInputs inputs{};
  std::uint64_t experts = 0;  // a hash router's side ids are below it
};

// Runs `in`'s fragment (the reference executor) and its reference over
// `steps`, carrying the state by slot, and compares every output and the
// state within `tolerance` of the largest value (`state_tolerance` for the
// state). Weights are random in `formats` (a role's own range in
// `ranges`), both sides reading the decoded values.
void Compare(const Instance& in, std::span<const format::TensorFormat> formats,
             std::span<const std::uint64_t> cells, std::span<const Step> steps, double tolerance,
             double state_tolerance, const Lone& lone = {},
             const std::map<std::uint16_t, std::pair<float, float>>& ranges = {},
             const std::map<std::uint16_t, float>& first_state_values = {}) {
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
        if (const auto initial = first_state_values.find(st.ref.tensor);
            initial != first_state_values.end())
          values[0] = initial->second;
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
    io.stream.clear();
    io.residual.clear();
    io.carry.clear();
    io.carry2.clear();
    io.side.clear();
    io.side_ids.clear();
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
          if (input.tensor == graph->carry2_in) {
            random(io.carry2, 1.0F);
          } else {
            random(io.carry, 1.0F);
          }
          continue;
        case InputKind::kSide: {
          if (!lone.inputs.side_ids) {
            random(io.side, 1.0F);
            continue;
          }
          // Each row's distinct experts.
          const ir::TensorDesc d = graph->step.graph.tensor(input.tensor);
          std::vector<std::int64_t> ids;
          for (std::uint64_t r = 0; r < d.ne(1); ++r) {
            std::vector<std::int64_t> all(lone.experts);
            std::iota(all.begin(), all.end(), 0);
            std::ranges::shuffle(all, rng);
            ids.insert(ids.end(), all.begin(), all.begin() + static_cast<std::ptrdiff_t>(d.ne(0)));
          }
          ASSERT_TRUE(exec->BindInts(input.tensor, ids));
          io.side_ids = ids;
          continue;
        }
        case InputKind::kHadamard: {
          ASSERT_TRUE(exec->BindValues(input.tensor, HadamardValues(input.read)));
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
      check(graph->carry2_out, io.carry2, "carry2");
    }
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

constexpr double kTolerance = 1e-4;
constexpr double kF16State = 1.0 / 1024;

// ---------------------------------------------------------------- catalog

TEST(ComponentsDsv4Test, TheCatalogHoldsTheDeepSeekKinds) {
  for (const char* kind : {"attention.mqa", "side.hash", "embedding.inject"}) {
    EXPECT_NE(Find(kind, 1), nullptr) << kind;
  }
  const auto all = Catalog();
  EXPECT_TRUE(std::ranges::is_sorted(all, {}, [](const ComponentDef* d) { return d->kind; }));
  EXPECT_TRUE(Find("side.hash", 1)->side_source);
  EXPECT_TRUE(Find("side.hash", 1)->roles[0].rows_on_demand);
  EXPECT_NE(Find("embedding.inject", 1)->inject, nullptr);
  EXPECT_NE(Find("attention.mqa", 1)->inject, nullptr);
}

// ---------------------------------------------------------------- hyper-connections

constexpr std::uint64_t kHcWidth = 32;
constexpr std::uint32_t kStreams = 4;
const StageFacts kHcFacts{.width = kHcWidth, .context = 4096, .streams = kStreams};

std::vector<Step> RowSteps() {
  return {
      {.shape = {.segments = {Seg(0, 3, 0, 256, 1), Seg(1, 2, 7, 256, 2)}},
       .tokens = {4, 0, 9, 7, 3}},
      {.shape = {.segments = {Seg(0, 1, 3, 256, 1)}}, .tokens = {5}},
  };
}

TEST(ComponentsDsv4Test, SinkhornHyperConnectionsMatchTheirReferences) {
  const std::uint64_t hw = kHcWidth * kStreams;
  const std::uint64_t mixes = (2 + kStreams) * kStreams;
  auto mixing =
      MakeInstance(C("residual.hc", {P("mix", N("sinkhorn"))}), kHcFacts, Site::kMixing, 0, 1);
  ASSERT_TRUE(mixing.has_value()) << mixing.error();
  Compare(*mixing, {}, {}, RowSteps(), kTolerance, kTolerance);
  // A block's mix: its input, its post weights and its combination.
  const auto mix_ref = [](bool head, std::vector<Param> more) {
    std::vector<Param> p{P("eps", R(1e-6)), P("hc_eps", R(1e-6)), P("mix", N("sinkhorn"))};
    if (!head) p.push_back(P("iterations", I(5)));
    for (Param& m : more) p.push_back(std::move(m));
    return C("hc.mix", std::move(p));
  };
  auto mix = MakeInstance(mix_ref(false, {}), kHcFacts, Site::kLayer, 0, 2);
  ASSERT_TRUE(mix.has_value()) << mix.error();
  // norm, down, up (unused), fn, base, scale, post_norm.
  const std::vector<format::TensorFormat> mf{
      Vector(hw),    Vector(1), Vector(1),       Matrix(Encoding::kF32, hw, mixes),
      Vector(mixes), Vector(3), Vector(kHcWidth)};
  Compare(*mix, mf, {}, RowSteps(), kTolerance, kTolerance, {.inputs = {.residual = true}});
  // The head's: the output rows' streams gathered, or every row's.
  const std::vector<format::TensorFormat> hf{
      Vector(hw),       Vector(1), Vector(1),       Matrix(Encoding::kF32, hw, kStreams),
      Vector(kStreams), Vector(1), Vector(kHcWidth)};
  for (const bool gather : {true, false}) {
    SCOPED_TRACE(gather ? "gathered" : "every row");
    auto head = MakeInstance(mix_ref(true, {P("inject", B(false)), P("gather", B(gather))}),
                             kHcFacts, Site::kFinalNorm, 0, 3);
    ASSERT_TRUE(head.has_value()) << head.error();
    std::vector<Step> steps = RowSteps();
    if (!gather) {
      for (Step& s : steps) {
        for (planner::Segment& seg : s.shape.segments) seg.outputs = seg.rows;
      }
    }
    Compare(*head, hf, {}, steps, kTolerance, kTolerance, {.inputs = {.residual = true}});
  }
  // The combine, from the mix's carries.
  auto combine =
      MakeInstance(C("hc.combine", {P("mix", N("sinkhorn"))}), kHcFacts, Site::kLayer, 0, 4);
  ASSERT_TRUE(combine.has_value()) << combine.error();
  Compare(*combine, std::vector<format::TensorFormat>{Vector(1)}, {}, RowSteps(), kTolerance,
          kTolerance, {.inputs = {.residual = true, .carry = kStreams, .carry2 = 1}});
  // Refused: a Sinkhorn mix with a gate rank or no iterations, a sigmoid
  // mix without its rank, a gather off a head.
  EXPECT_FALSE(MakeInstance(mix_ref(false, {P("rank", I(8))}), kHcFacts, Site::kLayer, 0, 2));
  EXPECT_FALSE(
      MakeInstance(C("hc.mix", {P("eps", R(1e-6)), P("hc_eps", R(1e-6)), P("mix", N("sinkhorn"))}),
                   kHcFacts, Site::kLayer, 0, 2));
  EXPECT_FALSE(MakeInstance(C("hc.mix", {P("eps", R(1e-6))}), kHcFacts, Site::kLayer, 0, 2));
  EXPECT_FALSE(MakeInstance(mix_ref(false, {P("gather", B(false))}), kHcFacts, Site::kLayer, 0, 2));
}

// ---------------------------------------------------------------- attention.mqa

constexpr std::uint64_t kWidth = 64;
const StageFacts kFacts{.width = kWidth, .context = 4096, .streams = 1};

constexpr std::uint64_t kHeads = 4;
constexpr std::uint64_t kHead = 32;
constexpr std::uint64_t kQRank = 32;
constexpr std::uint64_t kORank = 16;
constexpr std::uint64_t kGroups = 2;
constexpr std::uint64_t kWindow = 6;
constexpr std::uint64_t kIndexHeads = 2;
constexpr std::uint64_t kIndexDim = 16;

ComponentRef Mqa(std::string cache, std::int64_t ratio, std::vector<Param> more = {}) {
  std::vector<Param> p{
      P("eps", R(1e-6)),         P("heads", I(kHeads)),   P("head_dim", I(kHead)),
      P("rope_dims", I(8)),      P("q_rank", I(kQRank)),  P("o_rank", I(kORank)),
      P("o_groups", I(kGroups)), P("window", I(kWindow)), P("cache", N(std::move(cache)))};
  if (ratio != 0) {
    p.push_back(P("ratio", I(ratio)));
    p.push_back(P("rope_base", R(1000.0)));
    p.push_back(P("rope_factor", R(4.0)));
    p.push_back(P("yarn_context", I(64)));
    p.push_back(P("beta_fast", R(32.0)));
    p.push_back(P("beta_slow", R(1.0)));
  } else {
    p.push_back(P("rope_base", R(10000.0)));
  }
  for (Param& m : more) p.push_back(std::move(m));
  return C("attention.mqa", std::move(p));
}

// Every role's format (those an instance does not use: placeholders).
std::vector<format::TensorFormat> MqaFormats(Encoding e, std::uint64_t coff) {
  const std::uint64_t hd = kHeads * kHead;
  return {Matrix(e, kWidth, kQRank),
          Vector(kQRank),
          Matrix(e, kQRank, hd),
          Matrix(e, kWidth, kHead),
          Vector(kHead),
          Vector(kHeads),
          Matrix(e, hd / kGroups, kORank * kGroups),
          Matrix(e, kORank * kGroups, kWidth),
          Matrix(Encoding::kF32, kWidth, coff * kHead),
          Matrix(Encoding::kF32, kWidth, coff * kHead),
          Matrix(Encoding::kF32, coff * kHead, coff == 2 ? 4 : 8),
          Vector(kHead),
          Matrix(e, kQRank, kIndexHeads * kIndexDim),
          Matrix(Encoding::kF32, kWidth, kIndexHeads),
          Matrix(Encoding::kF32, kWidth, 2 * kIndexDim),
          Matrix(Encoding::kF32, kWidth, 2 * kIndexDim),
          Matrix(Encoding::kF32, 2 * kIndexDim, 4),
          Vector(kIndexDim)};
}

// Prefill of two owners past their compressors' blocks, a continuation, a
// state-only chunk, then decodes until the indexer selects.
std::vector<Step> MqaSteps(std::uint32_t read) {
  std::vector<Step> steps{
      {.shape = {.segments = {Seg(0, 9, 0, read, 1), Seg(1, 5, 0, read, 2)}}},
      {.shape = {.segments = {Seg(0, 7, 9, read, 7), Seg(1, 1, 5, read, 1)}}},
      {.shape = {.segments = {Seg(1, 6, 6, read, 0)}}},
  };
  for (std::uint32_t p = 16; p < 23; ++p) {
    steps.push_back({.shape = {.segments = {Seg(0, 1, p, read, 1)}}});
  }
  steps.push_back({.shape = {.segments = {Seg(1, 3, 12, read, 3), Seg(0, 2, 23, read, 1)}}});
  return steps;
}

TEST(ComponentsDsv4Test, WindowAttentionMatchesItsReference) {
  // The ring (read whole) and the full cache (a cell a position).
  for (const std::string cache : {"ring", "full"}) {
    SCOPED_TRACE(cache);
    auto in = MakeInstance(Mqa(cache, 0), kFacts, Site::kLayer, 0, 1);
    ASSERT_TRUE(in.has_value()) << in.error();
    const bool ring = cache == "ring";
    const std::array<std::uint64_t, 1> cells{ring ? 16U : 64U};
    for (const Encoding e : {Encoding::kF32, Encoding::kQ8_0}) {
      SCOPED_TRACE(format::Info(e).name);
      Compare(*in, MqaFormats(e, 1), cells, MqaSteps(ring ? 16 : 64), kTolerance, kF16State);
    }
  }
}

// The first main compressor creates its read input; the indexer reuses
// it. Both complete pooling assemblies must retain their owner's metadata.
TEST(ComponentsDsv4Test, FirstAndCachedCompressorAssembliesKeepTheirOwner) {
  auto in = MakeInstance(
      Mqa("ring", 4,
          {P("overlap", B(true)), P("indexer", B(true)), P("indexer_heads", I(kIndexHeads)),
           P("index_dim", I(kIndexDim)), P("top_k", I(3))}),
      kFacts, Site::kLayer, 2, 18);
  ASSERT_TRUE(in) << (in ? "" : in.error());
  const std::array<std::uint64_t, 7> cells{16, 32, 8, 8, 32, 8, 8};
  auto built =
      BuildComponent(*in, MqaSteps(16).front().shape, MqaFormats(Encoding::kF32, 2), cells);
  ASSERT_TRUE(built) << (built ? "" : built.error());
  const auto& graph = built->step.graph;
  std::size_t pools = 0;
  for (const auto& pool : graph.nodes()) {
    if (pool.key != ir::Primitive(ir::Op::kPool)) continue;
    ++pools;
    ASSERT_EQ(pool.batch, ir::Batch::kPerOwner);
    const auto owner = graph.tensor(pool.out[0]).segment;
    ASSERT_LT(owner, 2U);
    std::set<ir::TensorId> seen;
    const auto check = [&](ir::TensorId t, auto&& self) -> void {
      if (!t.valid() || !seen.insert(t).second) return;
      const auto& n = graph.node(graph.producer(t));
      if (n.key == ir::Primitive(ir::Op::kInput)) return;
      EXPECT_EQ(n.component, pool.component);
      EXPECT_EQ(n.batch, pool.batch) << ir::OpKeyName(n.key);
      EXPECT_EQ(graph.tensor(t).segment, owner) << ir::OpKeyName(n.key);
      // Explicit compressor boundary: [saved ring | current projections].
      // The projections themselves are intentionally joined products.
      if (n.key == ir::Primitive(ir::Op::kConcat) &&
          graph.tensor(graph.root(n.src[0])).role == ir::Role::kState)
        return;
      for (const auto source : n.sources()) self(source, self);
    };
    for (const auto source : pool.sources()) check(source, check);
  }
  EXPECT_EQ(pools, 4U);  // main + indexer, each of two owners
}

TEST(ComponentsDsv4Test, CompressedSparseAttentionMatchesItsReference) {
  // CSA: overlapping blocks of 4 and the lightning indexer's best 3.
  auto in = MakeInstance(
      Mqa("ring", 4,
          {P("overlap", B(true)), P("indexer", B(true)), P("indexer_heads", I(kIndexHeads)),
           P("index_dim", I(kIndexDim)), P("top_k", I(3))}),
      kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in.has_value()) << in.error();
  // kv, compressed, its ring kv and score, index, its ring kv and score.
  const std::array<std::uint64_t, 7> cells{16, 32, 8, 8, 32, 8, 8};
  Compare(*in, MqaFormats(Encoding::kF32, 2), cells, MqaSteps(16), kTolerance, kF16State);
  auto wide_raw_alignment = MqaSteps(16);
  for (auto& step : wide_raw_alignment) step.shape.read_align = 2048;
  Compare(*in, MqaFormats(Encoding::kF32, 2), cells, wide_raw_alignment, kTolerance, kF16State);
  // Its contract: today's DeepSeek V4 tensors (state_contract_test's).
  const auto contract = in->def->state(*in);
  ASSERT_EQ(contract.tensors.size(), 7U);
  EXPECT_EQ(contract.tensors[1].cells, model::CellRule::kPerBlock);
  EXPECT_TRUE(contract.tensors[1].dummy_last);
  EXPECT_EQ(contract.tensors[2].cells, model::CellRule::kPositionRing);
  EXPECT_EQ(contract.tensors[2].fixed_cells, 8U);
  EXPECT_EQ(contract.tensors[2].cell_bytes, 2U * kHead * 4U);
  EXPECT_EQ(contract.tensors[5].cell_bytes, 2U * kIndexDim * 4U);
}

TEST(ComponentsDsv4Test, OverlapDummyRowsDoNotReadMaskedStateScores) {
  auto in = MakeInstance(
      Mqa("ring", 4,
          {P("overlap", B(true)), P("indexer", B(true)), P("indexer_heads", I(kIndexHeads)),
           P("index_dim", I(kIndexDim)), P("top_k", I(3))}),
      kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in) << in.error();
  const std::array<std::uint64_t, 7> cells{16, 32, 8, 8, 32, 8, 8};
  const std::array<Step, 2> steps{Step{.shape = {.segments = {Seg(0, 4, 0, 16, 1)}}},
                                  Step{.shape = {.segments = {Seg(0, 1, 4, 16, 1)}}}};
  const float neg_inf = -std::numeric_limits<float>::infinity();
  Compare(*in, MqaFormats(Encoding::kF32, 2), cells, steps, kTolerance, kF16State, {}, {},
          {{3, neg_inf}, {6, neg_inf}});
}

TEST(ComponentsDsv4Test, HeavilyCompressedAttentionMatchesItsReference) {
  // HCA: blocks of 8 without overlap, every block row seen.
  auto in = MakeInstance(Mqa("ring", 8), kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in.has_value()) << in.error();
  const std::array<std::uint64_t, 4> cells{16, 16, 8, 8};
  Compare(*in, MqaFormats(Encoding::kF32, 1), cells, MqaSteps(16), kTolerance, kF16State);
}

TEST(ComponentsDsv4Test, ABlockDraftersAttentionMatchesItsReference) {
  // Window-only over its injected ring, its block's rows seeing each other.
  auto in = MakeInstance(Mqa("injected", 0, {P("cells", I(32))}), kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(in.has_value()) << in.error();
  const std::array<std::uint64_t, 1> cells{32};
  const std::vector<Step> steps{
      {.shape = {.phase = planner::Phase::kDraft,
                 .segments = {Seg(0, 3, 0, 32, 3), Seg(1, 3, 20, 32, 3)}}},
      {.shape = {.phase = planner::Phase::kDraft, .segments = {Seg(0, 3, 40, 32, 3)}}},
  };
  Compare(*in, MqaFormats(Encoding::kF32, 1), cells, steps, kTolerance, kF16State);
  const auto contract = in->def->state(*in);
  ASSERT_EQ(contract.tensors.size(), 1U);
  EXPECT_TRUE(contract.tensors[0].target_writes);
  EXPECT_EQ(contract.tensors[0].kind, model::StateKind::kDrafterRing);
  // Refused: a ring no wider than its window; an injected ring with a
  // compressor; an indexer without overlap.
  EXPECT_FALSE(MakeInstance(Mqa("injected", 0, {P("cells", I(6))}), kFacts, Site::kLayer, 0, 1));
  EXPECT_FALSE(MakeInstance(Mqa("injected", 4, {P("cells", I(32))}), kFacts, Site::kLayer, 0, 1));
  EXPECT_FALSE(MakeInstance(Mqa("ring", 4,
                                {P("indexer", B(true)), P("indexer_heads", I(2)),
                                 P("index_dim", I(16)), P("top_k", I(3))}),
                            kFacts, Site::kLayer, 0, 1));
}

// ---------------------------------------------------------------- routing

constexpr std::uint64_t kExperts = 8;

ComponentRef Moe(std::string router, std::vector<Param> more = {}) {
  std::vector<Param> p{P("placement", N("none")),
                       P("eps", R(1e-6)),
                       P("activation", N("silu_clamp")),
                       P("hidden", I(32)),
                       P("experts", I(kExperts)),
                       P("used", I(2)),
                       P("expert_hidden", I(32)),
                       P("router", N(std::move(router))),
                       P("router_input", N("stream")),
                       P("normalize", B(true)),
                       P("route_scale", R(1.5)),
                       P("router_f32", B(true)),
                       P("limit", R(0.4)),
                       P("shared_limit", R(0.5))};
  for (Param& m : more) p.push_back(std::move(m));
  return C("ffn.moe", std::move(p));
}

std::vector<format::TensorFormat> MoeFormats() {
  // norm, gate, up, down, dense_post_norm, router, router_scale,
  // routed_norm, gate_up_experts, gate_experts, up_experts, down_experts,
  // expert_scale, routed_post_norm, post_norm, gate_scale, up_scale,
  // down_scale, shared_gate, router_bias.
  return {Vector(kWidth),
          Matrix(Encoding::kF32, kWidth, 32),
          Matrix(Encoding::kF32, kWidth, 32),
          Matrix(Encoding::kF32, 32, kWidth),
          Vector(kWidth),
          Matrix(Encoding::kF32, kWidth, kExperts),
          Vector(kWidth),
          Vector(kWidth),
          Matrix(Encoding::kF32, kWidth, 64, kExperts),
          Matrix(Encoding::kF32, kWidth, 32, kExperts),
          Matrix(Encoding::kF32, kWidth, 32, kExperts),
          Matrix(Encoding::kF32, 32, kWidth, kExperts),
          Vector(kExperts),
          Vector(kWidth),
          Vector(kWidth),
          Vector(kExperts),
          Vector(kExperts),
          Vector(kExperts),
          Vector(kWidth),
          Vector(kExperts)};
}

TEST(ComponentsDsv4Test, DeepSeekRoutersAndClampedExpertsMatchTheirReferences) {
  // sqrt(softplus) chosen with a bias; the experts' and shared GLUs clamped.
  auto biased = MakeInstance(Moe("sqrt_softplus"), kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(biased.has_value()) << biased.error();
  Compare(*biased, MoeFormats(), {}, RowSteps(), kTolerance, kTolerance, {}, {{19, {-0.2F, 0.2F}}});
  // By a token-hash side input's ids.
  auto hashed = MakeInstance(Moe("hash", {P("hash", N("hash_0"))}), kFacts, Site::kLayer, 0, 1);
  ASSERT_TRUE(hashed.has_value()) << hashed.error();
  EXPECT_EQ(hashed->def->side_input(*hashed), "hash_0");
  Compare(*hashed, MoeFormats(), {}, RowSteps(), kTolerance, kTolerance,
          {.inputs = {.side = 2, .side_ids = true}, .experts = kExperts});
  // Refused: a hash router without its side input, a clamp without its
  // limits, a DeepSeek router over Gemma's scaled norm.
  EXPECT_FALSE(MakeInstance(Moe("hash"), kFacts, Site::kLayer, 0, 1));
  EXPECT_FALSE(MakeInstance(
      C("ffn.moe",
        {P("placement", N("none")), P("eps", R(1e-6)), P("activation", N("silu_clamp")),
         P("hidden", I(32)), P("experts", I(kExperts)), P("used", I(2)), P("expert_hidden", I(32)),
         P("router", N("softmax_topk")), P("router_input", N("stream"))}),
      kFacts, Site::kLayer, 0, 1));
  EXPECT_FALSE(MakeInstance(
      C("ffn.moe",
        {P("placement", N("none")), P("eps", R(1e-6)), P("activation", N("silu")),
         P("hidden", I(32)), P("experts", I(kExperts)), P("used", I(2)), P("expert_hidden", I(32)),
         P("router", N("sqrt_softplus")), P("router_input", N("rms_scaled"))}),
      kFacts, Site::kLayer, 0, 1));
}

TEST(ComponentsDsv4Test, ATokenHashSideInputReadsItsTokensRows) {
  auto in = MakeInstance(C("side.hash", {P("experts", I(kExperts)), P("used", I(3))}), kFacts,
                         Site::kSide, 0, 1);
  ASSERT_TRUE(in.has_value()) << in.error();
  // Its rows: each row's token.
  const std::vector<std::int32_t> history{5, 9, 2, 7};
  std::array<std::int32_t, 1> row{};
  ASSERT_TRUE(in->def->side_rows(*in, history, 2, row));
  EXPECT_EQ(row[0], 2);
  EXPECT_FALSE(in->def->side_rows(*in, history, 4, row));
  // Its step: the rows' ids, the table's.
  constexpr std::uint64_t kVocab = 12;
  const format::TensorFormat table = ir::Dense(Encoding::kI32, ir::ShapeOf({3, kVocab}));
  const planner::StepShape shape{.segments = {Seg(0, 3, 0, 256, 0), Seg(1, 1, 5, 256, 0)}};
  auto graph = BuildComponent(*in, shape, std::array{table}, {});
  ASSERT_TRUE(graph.has_value()) << graph.error();
  auto exec = ref::Executor::Create(graph->step.graph);
  ASSERT_TRUE(exec.has_value());
  std::vector<std::byte> bytes(3 * kVocab * 4);
  RefIO io{.instance = &*in};
  io.weights.assign(1, std::vector<float>(3 * kVocab));
  for (std::uint64_t i = 0; i < 3 * kVocab; ++i) {
    const auto id = static_cast<std::int32_t>((i * 5) % kExperts);
    std::memcpy(bytes.data() + (i * 4), &id, 4);
    io.weights[0][i] = static_cast<float>(id);
  }
  const ref::PartView part{.role = format::PartRole::kData, .bytes = bytes};
  ASSERT_TRUE(exec->Bind(graph->weights[0], std::span(&part, 1)));
  const std::vector<std::int64_t> tokens{4, 11, 0, 7};
  for (const StepInput& input : graph->step.inputs) {
    ASSERT_EQ(input.kind, InputKind::kSideRows);
    ASSERT_TRUE(exec->BindInts(input.tensor, tokens));
  }
  ASSERT_TRUE(exec->Run());
  io.side_rows = tokens;
  io.segments = {{.rows = 3}, {.rows = 1}};
  ASSERT_TRUE(in->def->reference(io));
  auto got = exec->Ints(graph->stream_out);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(*got, io.side_ids);
}

// ---------------------------------------------------------------- the drafter's ends

TEST(ComponentsDsv4Test, AMarkovHeadAndAnInjectedEmbeddingMatchTheirReferences) {
  constexpr std::uint64_t kVocab = 40;
  auto head = MakeInstance(C("head.linear", {P("gather", B(false)), P("markov", B(true))}), kFacts,
                           Site::kOutput, 0, 5);
  ASSERT_TRUE(head.has_value()) << head.error();
  // weight, projection, ids (unused), markov_w1, markov_w2.
  const std::vector<format::TensorFormat> hf{Matrix(Encoding::kF32, kWidth, kVocab), Vector(1),
                                             Vector(1), Matrix(Encoding::kF16, 8, kVocab),
                                             Matrix(Encoding::kF16, 8, kVocab)};
  const std::vector<Step> blocks{
      {.shape = {.phase = planner::Phase::kDraft,
                 .segments = {Seg(0, 3, 9, 256, 3, true), Seg(1, 3, 2, 256, 3, true)}},
       .tokens = {7, 1, 1, 30, 1, 1}},
      {.shape = {.phase = planner::Phase::kDraft, .segments = {Seg(0, 2, 12, 256, 2)}},
       .tokens = {3, 1}},
  };
  Compare(*head, hf, {}, blocks, kTolerance, kTolerance);
  EXPECT_FALSE(MakeInstance(C("head.linear", {P("markov", B(true))}), kFacts, Site::kOutput, 0, 5));
  auto embedding = MakeInstance(C("embedding.inject", {P("eps", R(1e-6)), P("tap", N("features")),
                                                       P("taps", I(3)), P("cells", I(32))}),
                                kFacts, Site::kEmbedding, 0, 0);
  ASSERT_TRUE(embedding.has_value()) << embedding.error();
  EXPECT_EQ(embedding->def->inject_reads(*embedding),
            (std::vector<std::string>{"features_0", "features_1", "features_2"}));
  const std::vector<format::TensorFormat> ef{Matrix(Encoding::kF16, kWidth, kVocab),
                                             Matrix(Encoding::kF32, 3 * kWidth, kWidth),
                                             Vector(kWidth)};
  Compare(*embedding, ef, {}, blocks, kTolerance, kTolerance);
}

// ---------------------------------------------------------------- compressor inputs

TEST(ComponentsDsv4Test, VisibleCellCountsUseOwnerAndRealCapacityWithoutPadding) {
  ir::Builder b;
  StepContext context(
      b, {{.slot = 2, .rows = 3, .past = 2}, {.slot = 7, .first_row = 3, .rows = 2, .past = 11}},
      false);
  const CellKey cells{.capacity = 256, .block = 4, .positions = true};
  const auto first = context.VisibleCells(0, cells, 16, 3);
  const auto second = context.VisibleCells(1, cells, 16, 3);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(context.VisibleCells(0, cells, 16, 3), first);
  EXPECT_NE(context.VisibleCells(0, cells, 2, 3), first);
  auto inputs = context.TakeInputs();
  ASSERT_EQ(inputs.size(), 3U);
  const planner::StepShape shape{.segments = {Seg(2, 3, 2, 256, 0), Seg(7, 2, 11, 256, 0)}};
  EXPECT_EQ(InputValues(inputs[0], shape, {}).value(), (std::vector<std::int64_t>{0, 1, 1}));
  EXPECT_EQ(InputValues(inputs[1], shape, {}).value(), (std::vector<std::int64_t>{3, 3}));
  auto shifted = shape;
  shifted.segments[0].past = UINT32_MAX;
  EXPECT_EQ(InputValues(inputs[0], shifted, {}).value(), (std::vector<std::int64_t>{3, 3, 3}));
  EXPECT_EQ(InputValues(inputs[2], shifted, {}).value(), (std::vector<std::int64_t>{2, 2, 2}));
  for (int bad = 0; bad < 12; ++bad) {
    auto input = inputs[0];
    switch (bad) {
      case 0:
        input.segment = ir::kJoinedSegment;
        break;
      case 1:
        input.cells.block = 0;
        break;
      case 2:
        input.cells.positions = false;
        break;
      case 3:
        input.cells.ring = true;
        break;
      case 4:
        input.cells.frozen = true;
        break;
      case 5:
        input.cells.first_row = 1;
        break;
      case 6:
        input.visible_capacity = 0;
        break;
      case 7:
        input.visible_capacity = 257;
        break;
      case 8:
        input.read = 257;
        break;
      case 9:
        input.read = 0;
        break;
      case 10:
        input.ratio = 1;
        break;
      case 11:
        input.sections = 2;
        break;
    }
    EXPECT_FALSE(InputValues(input, shape, {})) << bad;
  }
  EXPECT_FALSE(context.VisibleCells(2, cells, 16, 3));
  EXPECT_FALSE(context.VisibleCells(0, cells, 257, 3));
}

// ---------------------------------------------------------------- mappings

struct Composed {
  spec::Source target;
  spec::Source drafter;
  spec::ModelSpec spec;
};

// The target's spec, and composed with its drafter.
std::optional<Composed> Compose(const fam::Dsv4Profile& p, const fam::DsparkProfile& d,
                                const fx::Types& types = {}) {
  Composed out;
  out.target = fx::SourceOf("deepseek4", fx::Resources(p, types), spec::mappings::Dsv4Mapping());
  auto values = fam::Dsv4Values(p);
  EXPECT_TRUE(values.has_value()) << (values ? "" : values.error());
  if (!values) return std::nullopt;
  auto target = spec::Synthesize(spec::mappings::Dsv4Mapping(), out.target, *values);
  EXPECT_TRUE(target.has_value()) << (target ? "" : target.error());
  if (!target) return std::nullopt;
  out.drafter =
      fx::SourceOf("dflash", fx::DrafterResources(d, types), spec::mappings::DsparkMapping());
  auto drafter_values = fam::DsparkValues(d);
  EXPECT_TRUE(drafter_values.has_value()) << (drafter_values ? "" : drafter_values.error());
  if (!drafter_values) return std::nullopt;
  auto spec = spec::ComposeDrafter(*target, spec::mappings::Dsv4Mapping(),
                                   spec::mappings::DsparkMapping(), out.drafter, *drafter_values);
  EXPECT_TRUE(spec.has_value()) << (spec ? "" : spec.error());
  if (!spec) return std::nullopt;
  out.spec = std::move(*spec);
  return out;
}

const spec::Param* ParamOf(const spec::ComponentRef& c, std::string_view name) {
  const auto at = std::ranges::find(c.params, name, &spec::Param::name);
  return at == c.params.end() ? nullptr : &*at;
}

TEST(ComponentsDsv4Test, TheMappingsDescribeDeepSeekV4FlashAndItsDrafter) {
  const fam::Dsv4Profile& p = fam::Dsv4Flash();
  const fam::DsparkProfile& d = fam::DsparkDeepSeekV4Flash();
  auto made = Compose(p, d);
  ASSERT_TRUE(made.has_value());
  const spec::ModelSpec& spec = made->spec;
  ASSERT_EQ(spec.stages.size(), 2U);
  const spec::Stage& target = spec.stages[0];
  ASSERT_EQ(target.layers.size(), p.layers);
  EXPECT_EQ(target.stream.streams, 4U);
  EXPECT_TRUE(target.folds.empty());
  // A token-hash side input a hash-routed layer, read there alone.
  ASSERT_EQ(target.side.size(), p.hash_layers);
  for (std::uint32_t il = 0; il < p.hash_layers; ++il) {
    EXPECT_EQ(target.side[il].name, std::format("hash_{}", il));
    EXPECT_EQ(target.side[il].layers, std::vector<std::uint32_t>{il});
    EXPECT_EQ(target.side[il].source.kind, "side.hash");
  }
  // Every layer: its mixes and combines around its attention by its ratio
  // and its MoE by its routing.
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    SCOPED_TRACE(std::format("layer {}", il));
    const auto& c = target.layers[il].components;
    ASSERT_EQ(c.size(), 6U);
    EXPECT_EQ(c[0].kind, "hc.mix");
    EXPECT_EQ(c[1].kind, "attention.mqa");
    EXPECT_EQ(c[2].kind, "hc.combine");
    EXPECT_EQ(c[4].kind, "ffn.moe");
    const spec::Param* ratio = ParamOf(c[1], "ratio");
    const std::uint32_t want = p.compress_ratios[il];
    EXPECT_EQ(ratio == nullptr ? 0 : std::get<std::int64_t>(ratio->value), want);
    EXPECT_EQ(ParamOf(c[1], "indexer") != nullptr, want == fam::kDsv4CsaRatio);
    EXPECT_EQ(ParamOf(c[1], "rope_factor") != nullptr, want != 0);
    const spec::Param* router = ParamOf(c[4], "router");
    ASSERT_NE(router, nullptr);
    EXPECT_EQ(std::get<std::string>(router->value), il < p.hash_layers ? "hash" : "sqrt_softplus");
  }
  // The drafter's taps: the streams' means entering layers 41 and 42 and
  // leaving the last.
  ASSERT_EQ(target.taps.size(), 3U);
  for (std::uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(target.taps[i].name, std::format("features_{}", i));
    EXPECT_EQ(target.taps[i].point, spec::TapPoint::kStreamMean);
    EXPECT_EQ(target.taps[i].layer, d.target_layers[i] - 1);
  }
  const spec::Stage& drafter = spec.stages[1];
  EXPECT_EQ(drafter.role, spec::StageRole::kDrafter);
  ASSERT_EQ(drafter.layers.size(), d.blocks.layers);
  EXPECT_EQ(drafter.embedding->kind, "embedding.inject");
  EXPECT_EQ(drafter.reads.size(), 3U);
  const std::array<std::span<const BoundTensor>, 2> tensors{made->target.tensors,
                                                            made->drafter.tensors};
  ASSERT_TRUE(ValidateSpec(spec, tensors));
  // The layouts are the M3 runners' state (in closed form): the target's
  // ring and full windows (Dsv4State), the drafter's ring (DsparkState).
  for (const auto& [context, rows] : {std::pair{4096U, 512U}, {65536U, 2048U}}) {
    for (const auto window : {fam::Dsv4Window::kRing, fam::Dsv4Window::kFull}) {
      spec::ModelSpec form = spec;
      if (window == fam::Dsv4Window::kFull) {
        for (spec::Layer& l : form.stages[0].layers) {
          for (spec::ComponentRef& c : l.components) {
            for (spec::Param& q : c.params) {
              if (c.kind == "attention.mqa" && q.name == "cache") q.value = std::string("full");
            }
          }
        }
      }
      auto want = fam::Dsv4State(p, context, rows, window);
      ASSERT_TRUE(want.has_value());
      auto got = StageLayoutUniform(form, 0, context, rows, 256);
      ASSERT_TRUE(got.has_value()) << got.error();
      ASSERT_EQ(got->items.size(), want->tensors.size());
      for (std::size_t i = 0; i < got->items.size(); ++i) {
        EXPECT_EQ(got->items[i].offset, want->tensors[i].offset) << i;
        EXPECT_EQ(got->items[i].bytes, want->tensors[i].bytes) << i;
      }
      EXPECT_EQ(got->bytes, want->bytes);
    }
  }
  auto ring = fam::DsparkState(d, 16);
  ASSERT_TRUE(ring.has_value());
  auto drafted = StageLayoutUniform(spec, 1, 65536, 16, 256);
  ASSERT_TRUE(drafted.has_value()) << drafted.error();
  ASSERT_EQ(drafted->items.size(), ring->offsets.size());
  for (std::size_t i = 0; i < ring->offsets.size(); ++i) {
    EXPECT_EQ(drafted->items[i].offset, ring->offsets[i]);
  }
  EXPECT_EQ(drafted->bytes, ring->bytes);
}

// The serving target is not the composed spec above: with no taps or
// injection, its final stateless FFN is dead on intermediate prompt chunks.
TEST(ComponentsDsv4Test, SoloNoHeadPrefillKeepsStateWithoutFinalExpertProducts) {
  const auto& profile = fam::Dsv4Flash();
  auto source = fx::SourceOf("deepseek4", fx::Resources(profile), spec::mappings::Dsv4Mapping());
  auto values = fam::Dsv4Values(profile);
  ASSERT_TRUE(values.has_value()) << (values ? "" : values.error());
  auto target_spec = spec::Synthesize(spec::mappings::Dsv4Mapping(), source, *values);
  ASSERT_TRUE(target_spec.has_value()) << (target_spec ? "" : target_spec.error());
  ASSERT_EQ(target_spec->stages.size(), 1U);
  ASSERT_TRUE(target_spec->stages[0].taps.empty());
  auto layout = StageLayout(*target_spec, 0, 3072, 512);
  ASSERT_TRUE(layout.has_value()) << (layout ? "" : layout.error());
  const Bindings bindings{.tensors = source.tensors, .state = &*layout};
  for (const std::uint32_t outputs : {0U, 1U}) {
    SCOPED_TRACE(outputs);
    auto segment = Seg(0, 512, 0, 512, outputs);
    segment.want_head = outputs != 0;
    const planner::StepShape shape{.phase = planner::Phase::kPrefill, .segments = {segment}};
    auto step = BuildStep(*target_spec, 0, shape, bindings);
    ASSERT_TRUE(step.has_value()) << (step ? "" : step.error());
    std::vector<std::uint32_t> arrays(profile.layers, 0);
    std::uint32_t products = 0, hash_inputs = 0;
    for (const auto& weight : step->weights) {
      if (!weight.ref.expert_array) continue;
      ASSERT_LT(weight.ref.layer, arrays.size());
      ++arrays[weight.ref.layer];
    }
    for (const auto& node : step->graph.nodes()) {
      if (node.key.op != ir::Op::kMatMulId) continue;
      ++products;
      EXPECT_EQ(step->graph.tensor(node.src[0]).shape().ne[2], profile.experts);
      EXPECT_EQ(step->graph.tensor(node.src[2]).shape().ne[0], profile.experts_used);
    }
    for (const auto& input : step->inputs)
      if (input.kind == InputKind::kSideRows) ++hash_inputs;
    for (std::uint32_t layer = 0; layer < profile.layers; ++layer)
      EXPECT_EQ(arrays[layer], outputs != 0 || layer + 1 < profile.layers ? 3U : 0U) << layer;
    EXPECT_EQ(products, 3U * (profile.layers - (outputs == 0 ? 1U : 0U)));
    EXPECT_EQ(hash_inputs, profile.hash_layers);
    EXPECT_EQ(step->logits.valid(), outputs != 0);
    EXPECT_TRUE(CheckStateWrites(*step, 0));
    // Every declared persistent target state tensor is still written, including
    // the last layer's window/compressor stores; no fake head retains dead work.
    std::vector<bool> written(layout->items.size(), false);
    for (const auto& state : step->state) {
      ASSERT_EQ(state.ref.stage, 0U);
      ASSERT_LT(state.item, written.size());
      written[state.item] = written[state.item] || state.written;
    }
    EXPECT_TRUE(std::ranges::all_of(written, [](bool value) { return value; }));
  }
}

// Every step kind of the composed spec builds: the target's prefill (its
// injection), joined decodes and a verify, a state-only chunk; the
// drafter's draft blocks.
TEST(ComponentsDsv4Test, EveryStepKindBuilds) {
  const fam::Dsv4Profile& p = fam::Dsv4Flash();
  const fam::DsparkProfile& d = fam::DsparkDeepSeekV4Flash();
  auto made = Compose(p, d);
  ASSERT_TRUE(made.has_value());
  constexpr std::uint32_t kContext = 65536;
  constexpr std::uint32_t kRows = 512;
  auto target = StageLayout(made->spec, 0, kContext, kRows);
  auto drafter = StageLayout(made->spec, 1, kContext, 16);
  ASSERT_TRUE(target && drafter);
  const std::array<const model::StateLayoutDerived*, 2> layouts{&*target, &*drafter};
  const std::array<std::span<const BoundTensor>, 2> tensors{made->target.tensors,
                                                            made->drafter.tensors};
  const Bindings target_bindings{
      .tensors = made->target.tensors, .state = &*target, .stages = layouts, .artifacts = tensors};
  const Bindings drafter_bindings{.tensors = made->drafter.tensors,
                                  .state = &*drafter,
                                  .stages = layouts,
                                  .artifacts = tensors};
  for (const auto& [name, shape] : std::vector<std::pair<std::string, planner::StepShape>>{
           {"prefill", {.segments = {Seg(0, 300, 0, 768, 1)}}},
           {"state-only", {.segments = {Seg(0, 512, 300, 768, 0)}}},
           {"decodes",
            {.phase = planner::Phase::kDecode,
             .segments = {Seg(0, 1, 812, 768, 1, true), Seg(1, 1, 5000, 768, 1, true)}}},
           {"verify",
            {.phase = planner::Phase::kVerify,
             .segments = {Seg(0, 4, 813, 768, 4), Seg(1, 4, 5001, 768, 4)}}},
       }) {
    SCOPED_TRACE(name);
    auto step = BuildStep(made->spec, 0, shape, target_bindings);
    ASSERT_TRUE(step.has_value()) << step.error();
    // The injection: each drafter layer's ring written once a segment.
    std::size_t injected = 0;
    for (const StepState& st : step->state) {
      if (st.ref.stage == 1) {
        EXPECT_TRUE(st.tap);
        EXPECT_TRUE(st.written);
        EXPECT_FALSE(st.read);
        ++injected;
      }
    }
    EXPECT_EQ(injected, d.blocks.layers * shape.segments.size());
    ASSERT_TRUE(CheckStateWrites(*step, 0));
  }
  // The drafter's blocks, one owner and two joined.
  for (const auto& shape : {planner::StepShape{.stage = 1,
                                               .phase = planner::Phase::kDraft,
                                               .segments = {Seg(0, 5, 812, 256, 5, true)}},
                            planner::StepShape{.stage = 1,
                                               .phase = planner::Phase::kDraft,
                                               .segments = {Seg(0, 3, 812, 256, 3, true),
                                                            Seg(1, 3, 40, 256, 3, true)}}}) {
    auto step = BuildStep(made->spec, 1, shape, drafter_bindings);
    ASSERT_TRUE(step.has_value()) << step.error();
    ASSERT_TRUE(step->greedy.valid());
    EXPECT_EQ(step->graph.tensor(step->greedy).ne(0), shape.rows());
  }
  // Refused: a target step without the drafter's layout (its injection's).
  const Bindings bare{.tensors = made->target.tensors, .state = &*target, .artifacts = tensors};
  EXPECT_FALSE(BuildStep(made->spec, 0, {.segments = {Seg(0, 1, 0, 768, 1)}}, bare));
}

// ---------------------------------------------------------------- the injection, run

// A small DeepSeek V4 (four layers: window and CSA hash-routed, HCA, CSA)
// and its drafter.
fam::Dsv4Profile Tiny() {
  fam::Dsv4Profile p = fam::Dsv4Flash();
  p.name = "tiny";
  p.layers = 4;
  p.width = 64;
  p.heads = 4;
  p.head_dim = 32;
  p.rope_dims = 8;
  p.q_lora = 32;
  p.o_lora = 16;
  p.o_groups = 2;
  p.window = 6;
  p.experts = 8;
  p.experts_used = 2;
  p.expert_ffn = 32;
  p.hash_layers = 2;
  p.sinkhorn_iterations = 3;
  p.indexer_heads = 2;
  p.indexer_head_dim = 16;
  p.indexer_top_k = 3;
  p.vocab = 40;
  p.yarn_original_context = 64;
  p.compress_rope_base = 1000.0F;
  p.rope_scale = 4.0F;
  p.swiglu_limit = 0.4F;
  p.swiglu_limit_shared = 0.5F;
  p.compress_ratios = {0, fam::kDsv4CsaRatio, fam::kDsv4HcaRatio, fam::kDsv4CsaRatio};
  return p;
}
TEST(ComponentsDsv4Test, PromptRowsFollowChangedWindowAndCompressedResources) {
  for (const auto window : {512U, 4096U}) {
    auto profile = Tiny();
    profile.window = window;
    auto source = fx::SourceOf("deepseek4", fx::Resources(profile, fx::kF16Types),
                               spec::mappings::Dsv4Mapping());
    auto values = fam::Dsv4Values(profile);
    ASSERT_TRUE(values) << values.error();
    auto model = spec::Synthesize(spec::mappings::Dsv4Mapping(), source, *values);
    ASSERT_TRUE(model) << model.error();
    for (const auto context : {257U, 65535U, 131073U}) {
      for (const bool rounded : {false, true}) {
        SCOPED_TRACE(std::format("window={} context={} rounded={}", window, context, rounded));
        const auto fits = [&](std::uint64_t rows) {
          const auto pad = [](std::uint64_t n, std::uint64_t a) { return (n + a - 1) / a * a; };
          const auto raw = std::min(pad(context, 256),
                                    pad(std::min<std::uint64_t>(context, window + rows), 256));
          const auto padded_rows = pad(rows, 32);
          constexpr auto limit = std::uint64_t{std::numeric_limits<std::int32_t>::max()};
          if (raw * padded_rows * 2 > limit) return false;
          for (const auto ratio : {fam::kDsv4CsaRatio, fam::kDsv4HcaRatio}) {
            const auto compressed = pad(std::max<std::uint64_t>(1, context / ratio), 256);
            if (compressed * padded_rows * 2 > limit || (raw + compressed) * rows * 2 > limit)
              return false;
            if (!rounded && ratio == fam::kDsv4CsaRatio && compressed * rows * 4 > limit)
              return false;
          }
          return true;
        };
        std::uint32_t expected = context;
        while (!fits(expected)) --expected;
        const auto bound = StagePrefillRows(*model, 0, context, context, rounded);
        ASSERT_TRUE(bound) << bound.error();
        EXPECT_EQ(*bound, expected);
        EXPECT_TRUE(fits(*bound));
        if (*bound < context) EXPECT_FALSE(fits(*bound + 1));
        EXPECT_EQ(*StagePrefillRows(*model, 0, context, expected - 1, rounded), expected - 1);
      }
    }
  }
}

TEST(ComponentsDsv4Test, IndexerPolicyKeepsMasksForUnselectedCompressedAttention) {
  auto profile = Tiny();
  profile.indexer_heads = 64;
  profile.indexer_head_dim = 128;
  auto source = fx::SourceOf("deepseek4", fx::Resources(profile, fx::kF16Types),
                             spec::mappings::Dsv4Mapping());
  auto values = fam::Dsv4Values(profile);
  ASSERT_TRUE(values) << values.error();
  auto spec = spec::Synthesize(spec::mappings::Dsv4Mapping(), source, *values);
  ASSERT_TRUE(spec) << spec.error();
  const auto indexed = StageIndexedMasks(*spec, 0);
  ASSERT_TRUE(indexed) << indexed.error();
  EXPECT_TRUE(*indexed);
  EXPECT_FALSE(StageIndexedMasks(*spec, spec->stages.size()));
  auto layout = StageLayout(*spec, 0, 257, 8);
  ASSERT_TRUE(layout) << layout.error();
  const Bindings bindings{.tensors = source.tensors, .state = &*layout};
  const planner::StepShape shape{.segments = {Seg(0, 4, 2, 256, 4)}};
  auto ordinary = BuildStep(*spec, 0, shape, bindings);
  ASSERT_TRUE(ordinary) << ordinary.error();
  auto selected = BuildStep(*spec, 0, shape, bindings, {.rounded_index_mask = true});
  ASSERT_TRUE(selected) << selected.error();
  const auto count = [](const ir::Graph& graph, ir::Op op) {
    return std::ranges::count_if(graph.nodes(),
                                 [op](const auto& node) { return node.key == ir::Primitive(op); });
  };
  EXPECT_EQ(count(selected->graph, ir::Op::kAttention), 4);
  EXPECT_EQ(count(selected->graph, ir::Op::kSelectedMask), 2);
  EXPECT_EQ(count(selected->graph, ir::Op::kMaskBuild),
            count(ordinary->graph, ir::Op::kMaskBuild) - 1);
  // The HCA's unselected compressed bank still reads its causal mask;
  // only the two CSA layers receive a mask from indexer selection.
  int causal_compressed = 0;
  for (const auto& node : selected->graph.nodes()) {
    if (node.key != ir::Primitive(ir::Op::kAttention)) continue;
    const auto& mask = selected->graph.node(selected->graph.producer(node.src[3]));
    if (mask.key != ir::Primitive(ir::Op::kConcat)) continue;
    ++causal_compressed;
  }
  EXPECT_EQ(causal_compressed, 1);
}

TEST(ComponentsDsv4Test, RoundedIndexerUsesCompletedCountsAndOneFullSelectedMask) {
  auto profile = Tiny();
  profile.layers = 1;
  profile.hash_layers = 0;
  profile.compress_ratios = {4};
  profile.indexer_heads = 64;
  profile.indexer_head_dim = 128;
  auto source = fx::SourceOf("deepseek4", fx::Resources(profile, fx::kF16Types),
                             spec::mappings::Dsv4Mapping());
  auto values = fam::Dsv4Values(profile);
  ASSERT_TRUE(values) << values.error();
  auto spec = spec::Synthesize(spec::mappings::Dsv4Mapping(), source, *values);
  ASSERT_TRUE(spec) << spec.error();
  constexpr std::uint32_t kContext = 257;  // 64 complete blocks; padding is not capacity.
  auto layout = StageLayout(*spec, 0, kContext, 8);
  ASSERT_TRUE(layout) << layout.error();
  const Bindings bindings{.tensors = source.tensors, .state = &*layout};
  planner::StepShape shape{.segments = {Seg(0, 4, 2, 256, 4)}};
  auto ordinary = BuildStep(*spec, 0, shape, bindings);
  ASSERT_TRUE(ordinary) << ordinary.error();
  auto rounded = BuildStep(*spec, 0, shape, bindings, {.rounded_indexer = true});
  ASSERT_TRUE(rounded) << rounded.error();
  auto after = BuildStep(*spec, 0, shape, bindings);
  ASSERT_TRUE(after) << after.error();
  EXPECT_EQ(ordinary->graph.Digest(), after->graph.Digest());
  const auto count = [](const ir::Graph& graph, ir::Op op) {
    return std::ranges::count_if(graph.nodes(),
                                 [op](const auto& n) { return n.key == ir::Primitive(op); });
  };
  EXPECT_EQ(count(ordinary->graph, ir::Op::kIndexScore), 1);
  EXPECT_EQ(count(ordinary->graph, ir::Op::kRoundedIndexScore), 0);
  EXPECT_EQ(count(rounded->graph, ir::Op::kIndexScore), 0);
  EXPECT_EQ(count(rounded->graph, ir::Op::kRoundedIndexScore), 1);
  EXPECT_EQ(count(rounded->graph, ir::Op::kVisibleTopK), 1);
  EXPECT_EQ(count(rounded->graph, ir::Op::kSelectedMask), 1);
  const auto& graph = rounded->graph;
  for (const auto& attention : graph.nodes()) {
    if (attention.key != ir::Primitive(ir::Op::kAttention)) continue;
    const auto& mask = graph.node(graph.producer(attention.src[3]));
    ASSERT_EQ(mask.key, ir::Primitive(ir::Op::kSelectedMask));
    const auto& selected = graph.node(graph.producer(mask.src[1]));
    ASSERT_EQ(selected.key, ir::Primitive(ir::Op::kVisibleTopK));
    const auto& score = graph.node(graph.producer(selected.src[0]));
    ASSERT_EQ(score.key, ir::Primitive(ir::Op::kRoundedIndexScore));
    EXPECT_EQ(score.src[3], selected.src[1]);
    const auto at = std::ranges::find(rounded->inputs, score.src[3], &StepInput::tensor);
    ASSERT_NE(at, rounded->inputs.end());
    EXPECT_EQ(at->kind, InputKind::kVisibleCells);
    EXPECT_EQ(at->visible_capacity, kContext / 4);
    EXPECT_GT(at->read, at->visible_capacity);
    EXPECT_EQ(InputValues(*at, shape, {}), (std::vector<std::int64_t>{0, 1, 1, 1}));
    auto end = shape;
    end.segments.front().past = 253;
    EXPECT_EQ(InputValues(*at, end, {}), (std::vector<std::int64_t>{63, 63, 64, 64}));
    const auto raw = graph.tensor(mask.src[0]).ne(0);
    const auto compressed = graph.tensor(score.out[0]).ne(0);
    EXPECT_EQ(graph.tensor(attention.src[3]).ne(0), raw + compressed);
    EXPECT_EQ(graph.tensor(attention.src[1]).ne(1), raw + compressed);
  }
  for (auto phase : {planner::Phase::kVerify, planner::Phase::kCommit, planner::Phase::kDraft}) {
    auto unsupported = shape;
    unsupported.phase = phase;
    EXPECT_FALSE(BuildStep(*spec, 0, unsupported, bindings, {.rounded_indexer = true}));
  }
  auto joined = shape;
  joined.segments.push_back(Seg(1, 1, 0, 256, 1));
  EXPECT_FALSE(BuildStep(*spec, 0, joined, bindings, {.rounded_indexer = true}));
  auto beyond = shape;
  beyond.segments.front().past = kContext - 1;
  EXPECT_FALSE(BuildStep(*spec, 0, beyond, bindings, {.rounded_indexer = true}));
}

fam::DsparkProfile TinyDrafter() {
  fam::DsparkProfile d;
  d.name = "tiny-dspark";
  d.blocks = Tiny();
  d.blocks.layers = 1;
  d.blocks.hash_layers = 0;
  d.blocks.compress_ratios = {0};
  d.block_size = 3;
  d.target_layers = {2, 3, 4};
  d.mask_token = 39;
  d.markov_rank = 8;
  d.ring = 256;
  return d;
}

TEST(ComponentsDsv4Test, TapLayerBoundsAreCheckedBeforeEnteringAdjustment) {
  const auto profile = Tiny();
  const auto target_source = fx::SourceOf("deepseek4", fx::Resources(profile, fx::kF16Types),
                                          spec::mappings::Dsv4Mapping());
  auto target_values = fam::Dsv4Values(profile);
  ASSERT_TRUE(target_values) << target_values.error();
  auto target = spec::Synthesize(spec::mappings::Dsv4Mapping(), target_source, *target_values);
  ASSERT_TRUE(target) << target.error();
  const auto drafter_profile = TinyDrafter();
  const auto source = fx::SourceOf("dflash", fx::DrafterResources(drafter_profile, fx::kF16Types),
                                   spec::mappings::DsparkMapping());
  auto values = fam::DsparkValues(drafter_profile);
  ASSERT_TRUE(values) << values.error();
  auto layers = std::ranges::find(*values, "dflash.target_layers", &spec::SourceValue::key);
  ASSERT_NE(layers, values->end());
  auto& selected = std::get<std::vector<std::int64_t>>(layers->value);
  auto mapping = spec::mappings::DsparkMapping();
  std::vector<spec::ReadRule> reads(mapping.reads.begin(), mapping.reads.end());
  mapping.reads = reads;
  const auto count = static_cast<std::int64_t>(target->stages.front().layers.size());
  for (const bool entering : {false, true}) {
    reads.front().entering = entering;
    for (const auto entry :
         {std::numeric_limits<std::int64_t>::min(), std::int64_t{-1}, std::int64_t{0},
          std::int64_t{1}, count - 1, count, count + 1, std::numeric_limits<std::int64_t>::max()}) {
      SCOPED_TRACE(std::format("entering={} entry={}", entering, entry));
      selected.front() = entry;
      auto composed =
          spec::ComposeDrafter(*target, spec::mappings::Dsv4Mapping(), mapping, source, *values);
      const bool valid = entering ? entry > 0 && entry <= count : entry >= 0 && entry < count;
      ASSERT_EQ(composed.has_value(), valid);
      if (!valid) {
        EXPECT_NE(composed.error().find("names no layer"), std::string::npos);
        continue;
      }
      const auto& taps = composed->stages.front().taps;
      const auto tap = std::ranges::find(taps, "features_0", &spec::FeatureTap::name);
      ASSERT_NE(tap, taps.end());
      EXPECT_EQ(tap->layer, static_cast<std::uint32_t>(entering ? entry - 1 : entry));
    }
  }
}

// Binds each of a step's weights to random values in its format (the
// decoded values kept by name), and its inputs to the shape's.
struct Bound {
  std::map<std::string, std::vector<float>> weights;  // "<stage>/<name>"
};
void BindWeights(ref::Executor& exec, const StepGraph& step,
                 std::span<const std::span<const BoundTensor>> tensors, const spec::ModelSpec& spec,
                 std::mt19937& rng, Bound& bound) {
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  for (const StepWeight& w : step.weights) {
    const std::string key = std::format("{}/{}", w.ref.stage, w.parts.front().first);
    const auto artifact = spec.stages.at(w.ref.stage).artifact;
    const auto found =
        std::ranges::find(tensors[artifact], w.parts.front().first, &BoundTensor::name);
    ASSERT_NE(found, tensors[artifact].end()) << key;
    const format::TensorFormat& f = found->format;
    std::vector<float> v(ir::Elements(f.shape));
    const std::string& name = w.parts.front().first;
    if (f.encoding == Encoding::kI32) {
      // A token-hash table: each token's distinct experts.
      std::vector<std::byte> bytes(v.size() * 4);
      for (std::uint64_t t = 0; t < f.shape.ne[1]; ++t) {
        for (std::uint64_t i = 0; i < f.shape.ne[0]; ++i) {
          const auto id = static_cast<std::int32_t>((t + (i * 3)) % 8);
          std::memcpy(bytes.data() + (((t * f.shape.ne[0]) + i) * 4), &id, 4);
        }
      }
      const ref::PartView part{.role = format::PartRole::kData, .bytes = bytes};
      ASSERT_TRUE(exec.Bind(w.tensor, std::span(&part, 1)));
      continue;
    }
    const bool norm = name.find("norm") != std::string::npos || name.ends_with("scale.weight");
    for (float& x : v) x = norm ? 1.0F + (0.25F * u(rng)) : 0.3F * u(rng);
    const auto parts = format::PartsOf(f);
    ASSERT_TRUE(parts.has_value());
    std::vector<std::byte> bytes((*parts)[0].bytes);
    const std::uint64_t matrices = format::Matrices(f);
    const std::size_t per = v.size() / matrices;
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::EncodeRows(f, m, 0, std::span(v).subspan(m * per, per), bytes));
    }
    const ref::PartView part{.role = format::PartRole::kData, .bytes = bytes};
    std::vector<float> decoded(v.size());
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::DecodeRows(f, std::span(&part, 1), m, 0,
                                  f.shape.ne[1] * f.shape.ne[2] / matrices,
                                  std::span(decoded).subspan(m * per, per)));
    }
    ASSERT_TRUE(exec.Bind(w.tensor, std::span(&part, 1)));
    bound.weights[key] = std::move(decoded);
  }
}

TEST(ComponentsDsv4Test, ATargetsStepInjectsItsFeaturesIntoItsDraftersRing) {
  const fam::Dsv4Profile p = Tiny();
  const fam::DsparkProfile d = TinyDrafter();
  auto made = Compose(p, d, fx::kF16Types);
  ASSERT_TRUE(made.has_value());
  constexpr std::uint32_t kContext = 4096;
  auto target = StageLayout(made->spec, 0, kContext, 320);
  auto drafter = StageLayout(made->spec, 1, kContext, 320);
  ASSERT_TRUE(target && drafter);
  const std::array<const model::StateLayoutDerived*, 2> layouts{&*target, &*drafter};
  const std::array<std::span<const BoundTensor>, 2> tensors{made->target.tensors,
                                                            made->drafter.tensors};
  const Bindings bindings{
      .tensors = made->target.tensors, .state = &*target, .stages = layouts, .artifacts = tensors};
  // Two owners: one injecting its last 256 of 300 rows, one all of its 5.
  const auto read = static_cast<std::uint32_t>(target->items.front().cells);
  const planner::StepShape shape{.segments = {Seg(0, 300, 0, read, 1), Seg(1, 5, 2, read, 2)}};
  auto step = BuildStep(made->spec, 0, shape, bindings);
  ASSERT_TRUE(step.has_value()) << step.error();
  auto exec = ref::Executor::Create(step->graph);
  ASSERT_TRUE(exec.has_value()) << exec.error().detail;
  std::mt19937 rng(11);
  Bound bound;
  BindWeights(*exec, *step, tensors, made->spec, rng, bound);
  for (const StepState& st : step->state) {
    std::vector<float> zeros(ir::Elements(step->graph.tensor(st.tensor).shape()), 0.0F);
    ASSERT_TRUE(exec->BindValues(st.tensor, zeros));
  }
  std::vector<std::int32_t> tokens(shape.rows());
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = static_cast<std::int32_t>((i * 7) % p.vocab);
  for (const StepInput& input : step->inputs) {
    if (input.kind == InputKind::kHadamard) {
      ASSERT_TRUE(exec->BindValues(input.tensor, HadamardValues(input.read)));
      continue;
    }
    if (input.kind == InputKind::kSideRows) {
      // A token-hash side input's rows: the tokens.
      ASSERT_TRUE(
          exec->BindInts(input.tensor, std::vector<std::int64_t>(tokens.begin(), tokens.end())));
      continue;
    }
    auto values = InputValues(input, shape, tokens);
    ASSERT_TRUE(values.has_value()) << values.error();
    ASSERT_TRUE(exec->BindInts(input.tensor, *values));
  }
  auto ran = exec->Run();
  ASSERT_TRUE(ran.has_value()) << ran.error().detail;
  // The streams after each layer: its second combine's.
  std::vector<std::vector<float>> streams;
  for (const ir::Node& n : step->graph.nodes()) {
    if (n.key == ir::Primitive(ir::Op::kHcPost)) {
      auto v = exec->Values(n.out[0]);
      ASSERT_TRUE(v.has_value());
      streams.push_back(std::move(*v));
    }
  }
  ASSERT_EQ(streams.size(), 2U * p.layers);
  // The expected ring rows: each segment's last rows' features (the means
  // of the streams entering layers 2, 3 and leaving the last), projected,
  // normed, through the drafter layer's kv and its norm, rotated.
  const std::uint64_t w = p.width;
  const std::uint64_t hc = p.hc;
  const auto weight = [&](std::uint16_t stage, std::string_view name) -> const std::vector<float>& {
    return bound.weights.at(std::format("{}/{}", stage, name));
  };
  std::uint64_t first = 0;
  for (std::size_t s = 0; s < shape.segments.size(); ++s) {
    const planner::Segment& seg = shape.segments[s];
    const std::uint64_t kept = std::min<std::uint64_t>(seg.rows, d.ring);
    std::vector<double> features;
    for (std::uint64_t r = first + seg.rows - kept; r < first + seg.rows; ++r) {
      for (const std::uint32_t layer : d.target_layers) {
        const std::vector<float>& x = streams[(2 * (layer - 1)) + 1];
        for (std::uint64_t i = 0; i < w; ++i) {
          double mean = 0.0;
          for (std::uint64_t h = 0; h < hc; ++h) mean += x[(((r * hc) + h) * w) + i];
          features.push_back(
              static_cast<double>(static_cast<float>(mean / static_cast<double>(hc))));
        }
      }
    }
    std::vector<double> inp = defs::Product(weight(1, "fc.weight"), 3 * w, w, features);
    defs::RmsRows(inp, w, 1e-6, weight(1, "enc.output_norm.weight"));
    std::vector<double> kv = defs::Product(weight(1, "blk.0.attn_kv.weight"), w, p.head_dim, inp);
    defs::RmsRows(kv, p.head_dim, 1e-6, weight(1, "blk.0.attn_kv_a_norm.weight"));
    // The ring's cells after the step.
    const StepState* ring = nullptr;
    for (const StepState& st : step->state) {
      if (st.ref.stage == 1 && st.ref.segment == s) ring = &st;
    }
    ASSERT_NE(ring, nullptr);
    auto cells = exec->Values(ring->tensor);
    ASSERT_TRUE(cells.has_value());
    std::vector<double> want;
    std::vector<float> got;
    for (std::uint64_t k = 0; k < kept; ++k) {
      const std::uint64_t pos = seg.past + seg.rows - kept + k;
      std::span<double> row = std::span(kv).subspan(k * p.head_dim, p.head_dim);
      defs::RotateHeads(row, 1, p.head_dim,
                        {.mode = ir::RopeMode::kNormal,
                         .dims = p.rope_dims,
                         .offset = static_cast<std::uint32_t>(p.head_dim - p.rope_dims),
                         .base = static_cast<float>(p.rope_base)},
                        static_cast<std::int64_t>(pos));
      const std::uint64_t cell = pos % d.ring;
      for (std::uint64_t j = 0; j < p.head_dim; ++j) {
        want.push_back(row[j]);
        got.push_back((*cells)[(cell * p.head_dim) + j]);
      }
    }
    EXPECT_LE(Gap(got, want), 2.0 / 1024) << "segment " << s;
    first += seg.rows;
  }
}

}  // namespace
}  // namespace llmp::components
