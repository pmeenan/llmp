// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Partitioning (planner/partition.h): a small Gemma-like spec's steps and a
// test-only routed-expert component, partitioned by tensor, pipeline and
// expert parallelism and mixes of them over three to eight simulated devices
// (the reference executor's RunDevices), give the unpartitioned graph's
// logits, greedy tokens and state; the partitions' collectives carry what
// the traffic model counts; Assign's choices for described topologies (one
// device when it fits, the fewest pipeline stages on uniform links, tensor
// parallelism inside fast-linked pairs and a pipeline across the slow link,
// expert parallelism for routed experts, the two Sparks' link) match the
// optima derived by hand and, on random topologies with twins, a
// brute-force enumeration; infeasible memory and a search past its bound
// are refused. Attention whose mask or ALiBi cannot follow a head split
// runs whole and still agrees.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/step.h"
#include "families/gemma.h"
#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/reference/exec.h"
#include "planner/partition.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/spec.h"
#include "spec_fixtures.h"

namespace llmp::planner {
namespace {

namespace ref = llmp::kernels::reference;
namespace fam = llmp::test_support::families;
using components::BoundTensor;
using components::SplitKind;
using format::Encoding;
using spec::ComponentRef;
using spec::Param;
using spec::ParamValue;

// ----------------------------------------------------------- a tiny Gemma

constexpr std::uint64_t kWidth = 32;
constexpr std::uint64_t kHeads = 24;
constexpr std::uint64_t kKvHeads = 12;  // GQA: two query heads a KV head
constexpr std::uint64_t kDim = 4;
constexpr std::uint64_t kHidden = 48;
constexpr std::uint64_t kVocab = 48;
constexpr std::uint32_t kLayers = 6;
constexpr std::uint32_t kContext = 64;
constexpr std::uint32_t kMaxRows = 8;
constexpr std::uint32_t kRead = 256;

Param P(std::string name, ParamValue v) { return {.name = std::move(name), .value = std::move(v)}; }
template <typename T>
ParamValue I(T v) {
  return ParamValue(static_cast<std::int64_t>(v));
}
ParamValue R(double v) { return ParamValue(v); }
ParamValue N(std::string v) { return ParamValue(std::move(v)); }

ComponentRef C(std::string kind, std::vector<Param> params,
               std::vector<std::pair<std::string, std::string>> weights) {
  ComponentRef c{.kind = std::move(kind), .version = 1, .params = std::move(params), .weights = {}};
  for (auto& [role, tensor] : weights) {
    c.weights.push_back({.role = role, .tensor = tensor, .part = format::PartRole::kData});
  }
  return c;
}

format::TensorFormat Format(Encoding e, std::initializer_list<std::uint64_t> ne) {
  format::TensorFormat f = ir::Dense(e, ir::ShapeOf(ne));
  if (format::Info(e).quantized) f.layout = format::Layout::kGgmlBlocks;
  return f;
}

struct Tiny {
  spec::ModelSpec spec{};
  std::vector<BoundTensor> tensors{};
};

// Six Gemma 2-like layers (sliding and global, sandwich norms, Q and K
// norms, softcaps), F32 weights unless the feed-forward blocks' encoding is
// given; the head tied to the embedding or its own.
Tiny TinyGemma(bool tied, Encoding ffn = Encoding::kF32, std::uint64_t hidden = kHidden,
               std::uint64_t dim = kDim, Encoding kv = Encoding::kF16) {
  Tiny t;
  const auto bind = [&](std::string name, std::initializer_list<std::uint64_t> ne,
                        Encoding e = Encoding::kF32) {
    t.tensors.push_back({.name = name, .format = Format(e, ne)});
    return name;
  };
  spec::Stage s{
      .name = "target", .role = spec::StageRole::kTarget, .artifact = 0, .context = kContext};
  s.embedding = C("embedding.token", {P("scale", R(2.0))},
                  {{"table", bind("token_embd.weight", {kWidth, kVocab})}});
  s.stream = {.streams = 1, .width = kWidth, .mixing = std::nullopt};
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    const std::string p = std::format("blk.{}.", l);
    spec::Layer layer;
    layer.components.push_back(
        C("attention.dense",
          {P("placement", N("sandwich")), P("eps", R(1e-6)), P("heads", I(kHeads)),
           P("kv_heads", I(kKvHeads)), P("key_dim", I(dim)), P("value_dim", I(dim)),
           P("scale", R(0.5)), P("softcap", R(kv == Encoding::kQ8_0 ? 0.0 : 30.0)),
           P("kv_encoding", N(kv == Encoding::kQ8_0 ? "q8_0" : "f16")),
           P("window", I(l % 2 == 0 ? 4 : 0)), P("rotation", N("neox")), P("rope_dims", I(dim)),
           P("rope_base", R(10000.0)), P("rope_scale", R(1.0)), P("qk_norm", ParamValue(true))},
          {{"norm", bind(p + "attn_norm.weight", {kWidth})},
           {"q", bind(p + "attn_q.weight", {kWidth, kHeads * dim})},
           {"k", bind(p + "attn_k.weight", {kWidth, kKvHeads * dim})},
           {"v", bind(p + "attn_v.weight", {kWidth, kKvHeads * dim})},
           {"output", bind(p + "attn_output.weight", {kHeads * dim, kWidth})},
           {"post_norm", bind(p + "post_attention_norm.weight", {kWidth})},
           {"q_norm", bind(p + "attn_q_norm.weight", {dim})},
           {"k_norm", bind(p + "attn_k_norm.weight", {dim})}}));
    layer.components.push_back(
        C("ffn.glu",
          {P("placement", N("sandwich")), P("eps", R(1e-6)), P("activation", N("gelu_tanh")),
           P("hidden", I(static_cast<std::int64_t>(hidden)))},
          {{"norm", bind(p + "ffn_norm.weight", {kWidth})},
           {"gate", bind(p + "ffn_gate.weight", {kWidth, hidden}, ffn)},
           {"up", bind(p + "ffn_up.weight", {kWidth, hidden}, ffn)},
           {"down", bind(p + "ffn_down.weight", {hidden, kWidth}, ffn)},
           {"post_norm", bind(p + "post_ffw_norm.weight", {kWidth})}}));
    s.layers.push_back(std::move(layer));
  }
  s.heads.final_norm =
      C("norm.rms", {P("eps", R(1e-6))}, {{"weight", bind("output_norm.weight", {kWidth})}});
  s.heads.output = C("head.linear", {},
                     {{"weight", tied ? std::string("token_embd.weight")
                                      : bind("output.weight", {kWidth, kVocab})}});
  s.heads.softcap = 20.0;
  t.spec.pipeline = spec::PipelineKind::kLlm;
  t.spec.stages.push_back(std::move(s));
  spec::Canonicalize(t.spec);
  return t;
}

// Two small hyper-connected layers consume the same side-table lookup.
Tiny TinyHcSide(bool sparse = false) {
  Tiny t;
  const auto bind = [&](std::string name, std::initializer_list<std::uint64_t> ne) {
    t.tensors.push_back({.name = name, .format = Format(Encoding::kF32, ne)});
    return name;
  };
  spec::Stage stage{.name = "target", .context = kContext};
  stage.embedding = C("embedding.token", {}, {{"table", bind("tokens", {4, 8})}});
  stage.stream = {
      .streams = 2, .width = 4, .mixing = C("residual.hc", {P("mix", N("sigmoid"))}, {})};
  stage.side = {{.name = "ngram",
                 .source = C("side.ngram",
                             {P("ngram", I(2)), P("heads_per_ngram", I(1)), P("row", I(2)),
                              P("eos", I(0)), P("multipliers", std::vector<std::int64_t>{3, 5}),
                              P("offsets", std::vector<std::int64_t>{0}),
                              P("vocab", std::vector<std::int64_t>{8})},
                             {{"table", bind("side", {2, 8})}}),
                 .layers = {0, 1}}};
  if (sparse) {
    stage.side[0].layers = {0, 2};
    auto alternate = stage.side[0];
    alternate.name = "alternate";
    alternate.layers = {1, 3};
    alternate.source.weights[0].tensor = bind("side.alternate", {2, 8});
    stage.side.push_back(std::move(alternate));
  }
  for (std::uint32_t l = 0; l < (sparse ? 4U : 2U); ++l) {
    const auto prefix = std::format("layer.{}.", l);
    stage.layers.push_back(
        {.components = {
             C("residual.ple",
               {P("eps", R(1e-6)), P("input", N(sparse && l % 2 == 1 ? "alternate" : "ngram")),
                P("conv", I(2)), P("dilation", I(1))},
               {{"key", bind(prefix + "key", {2, 8})},
                {"value", bind(prefix + "value", {2, 4})},
                {"norm_key", bind(prefix + "norm_key", {8})},
                {"norm_query", bind(prefix + "norm_query", {8})},
                {"norm_conv", bind(prefix + "norm_conv", {8})},
                {"conv", bind(prefix + "conv", {2, 8})}})}});
  }
  stage.heads.final_norm = C("hc.mix", {P("eps", R(1e-6)), P("rank", I(2)), P("inject", false)},
                             {{"norm", bind("head.norm", {8})},
                              {"down", bind("head.down", {8, 2})},
                              {"up", bind("head.up", {2, 8})}});
  stage.heads.output = C("head.linear", {}, {{"weight", bind("head.output", {4, 8})}});
  t.spec.stages.push_back(std::move(stage));
  spec::Canonicalize(t.spec);
  return t;
}

planner::Segment Seg(std::uint8_t slot, std::uint32_t rows, std::uint32_t past,
                     std::uint32_t outputs, bool greedy = false) {
  return {.slot = slot,
          .rows = rows,
          .past = past,
          .read_cells = kRead,
          .outputs = outputs,
          .greedy = greedy};
}

// ------------------------------------------------- a test routed-expert FFN

// moe.test, version 1 (tests only): a pre-normed routed feed-forward block
// from the op set's routing ops: softmax top-`used` of `experts`, SwiGLU
// experts as expert arrays, their weighted sum (with `scaled`, each slot's
// learned expert scale gathered before it, as ffn.moe gathers Gemma 4's),
// the residual add. Splits:
// experts placed whole (expert parallelism) first, the hidden columns
// inside every expert second.
namespace moe {

constexpr std::array kParams{
    components::ParamSpec{
        .name = "eps", .type = components::ParamType::kReal, .least = 0.0, .most = 1.0},
    components::ParamSpec{
        .name = "experts", .type = components::ParamType::kInt, .least = 1, .most = 1024},
    components::ParamSpec{
        .name = "hidden", .type = components::ParamType::kInt, .least = 1, .most = 1 << 20},
    components::ParamSpec{
        .name = "used", .type = components::ParamType::kInt, .least = 1, .most = 64},
    components::ParamSpec{
        .name = "scaled", .type = components::ParamType::kBool, .required = false},
};
enum : std::uint16_t { kNorm, kRouter, kGate, kUp, kDown, kScale };
constexpr std::array<Encoding, 1> kF32Only{Encoding::kF32};
constexpr std::array kRoles{components::WeightRole{.name = "norm", .encodings = kF32Only},
                            components::WeightRole{.name = "router"},
                            components::WeightRole{.name = "gate"},
                            components::WeightRole{.name = "up"},
                            components::WeightRole{.name = "down"},
                            components::WeightRole{.name = "scale", .encodings = kF32Only}};

std::uint64_t E(const components::Instance& in) {
  return static_cast<std::uint64_t>(in.params.Int("experts"));
}
std::uint64_t F(const components::Instance& in) {
  return static_cast<std::uint64_t>(in.params.Int("hidden"));
}

components::RoleShape Role(std::uint16_t role, std::initializer_list<std::uint64_t> ne) {
  components::RoleShape r{.role = role};
  std::size_t i = 0;
  for (const std::uint64_t n : ne) r.ne.at(i++) = n;
  r.rank = static_cast<std::uint8_t>(ne.size());
  return r;
}

std::expected<std::vector<components::RoleShape>, std::string> Shapes(
    const components::Instance& in) {
  const std::uint64_t w = in.stage.width;
  std::vector<components::RoleShape> out{
      Role(kNorm, {w}), Role(kRouter, {w, E(in)}), Role(kGate, {w, F(in), E(in)}),
      Role(kUp, {w, F(in), E(in)}), Role(kDown, {F(in), w, E(in)})};
  if (in.params.Flag("scaled")) out.push_back(Role(kScale, {E(in)}));
  return out;
}

std::uint64_t Stream(const components::SplitStep& s) {
  return s.instance->stage.width * s.rows * 4;
}

constexpr std::array kExpertRoles{components::RoleSplit{.role = kGate, .dim = 2},
                                  components::RoleSplit{.role = kUp, .dim = 2},
                                  components::RoleSplit{.role = kDown, .dim = 2}};
constexpr std::array kColumnRoles{components::RoleSplit{.role = kGate, .dim = 1},
                                  components::RoleSplit{.role = kUp, .dim = 1},
                                  components::RoleSplit{.role = kDown, .dim = 0}};
constexpr std::array kSplits{
    components::SplitOption{
        .kind = SplitKind::kExperts,
        .collective = ir::Op::kAllReduce,
        .units = [](const components::Instance& in) { return static_cast<std::uint32_t>(E(in)); },
        .bytes_per_step = Stream,
        .weights = kExpertRoles,
        .state = {}},
    components::SplitOption{.kind = SplitKind::kColumns,
                            .collective = ir::Op::kAllReduce,
                            .units = nullptr,
                            .bytes_per_step = Stream,
                            .weights = kColumnRoles,
                            .state = {}}};

std::expected<components::FragmentOut, std::string> Emit(components::Fragment& f) {
  ir::Builder& b = f.b;
  const auto& in = f.instance;
  const std::uint64_t rows = f.step.rows();
  const ir::TensorId x = f.stream;
  const ir::TensorId h =
      b.Mul(b.RmsNorm(x, static_cast<float>(in.params.Real("eps"))), f.Weight(kNorm));
  const ir::TensorId logits = b.MatMul(f.Weight(kRouter), h);
  const ir::Outputs routed = b.Route(logits, {}, {}, {},
                                     {.kind = ir::RouteKind::kSoftmaxTopK,
                                      .normalize = true,
                                      .experts = static_cast<std::uint32_t>(E(in)),
                                      .used = static_cast<std::uint32_t>(in.params.Int("used")),
                                      .scale = 1.0F});
  const ir::TensorId hb = b.Reshape(h, ir::ShapeOf({in.stage.width, 1, rows}));
  const ir::TensorId gate = b.MatMulId(f.Weight(kGate), hb, routed[0]);
  const ir::TensorId up = b.MatMulId(f.Weight(kUp), hb, routed[0]);
  const ir::TensorId act = b.Glu(gate, up, {.kind = ir::GluKind::kSwiGlu});
  const ir::TensorId down = b.MatMulId(f.Weight(kDown), act, routed[0]);
  ir::TensorId scale{};
  if (in.params.Flag("scaled")) {
    const ir::TensorId table = b.Repeat(b.Reshape(f.Weight(kScale), ir::ShapeOf({1, E(in), 1})),
                                        ir::ShapeOf({1, E(in), rows}));
    scale = b.GetRows(table, routed[0]);  // [1, used, rows]
  }
  const ir::TensorId sum = b.ExpertReduce(down, routed[1], routed[0], scale);
  return components::FragmentOut{.stream = b.Add(sum, x)};
}

const components::ComponentDef& Def() {
  static const components::ComponentDef def{
      .kind = "moe.test",
      .version = 1,
      .params = kParams,
      .roles = kRoles,
      .shapes = Shapes,
      .splits = kSplits,
      .emit = Emit,
  };
  return def;
}

constexpr std::uint64_t kExperts = 12;
constexpr std::uint64_t kExpertHidden = 48;

const ComponentRef& Ref(bool scaled = false) {
  static const ComponentRef r = C("moe.test",
                                  {P("eps", R(1e-6)), P("experts", I(kExperts)),
                                   P("hidden", I(kExpertHidden)), P("used", I(3))},
                                  {});
  static const ComponentRef with_scale =
      C("moe.test",
        {P("eps", R(1e-6)), P("experts", I(kExperts)), P("hidden", I(kExpertHidden)),
         P("used", I(3)), P("scaled", ParamValue(true))},
        {});
  return scaled ? with_scale : r;
}

components::Instance Instance(bool scaled = false) {
  auto params = components::Params::Of(Def().params, Ref(scaled));
  EXPECT_TRUE(params.has_value()) << params.error();
  return {.def = &Def(),
          .ref = &Ref(scaled),
          .params = *params,
          .stage = {.width = kWidth},
          .site = components::Site::kLayer,
          .layer = 0,
          .index = 1};
}

// Every role's (the scale's used only `scaled`).
std::vector<format::TensorFormat> Formats() {
  return {Format(Encoding::kF32, {kWidth}),
          Format(Encoding::kF32, {kWidth, kExperts}),
          Format(Encoding::kF32, {kWidth, kExpertHidden, kExperts}),
          Format(Encoding::kF32, {kWidth, kExpertHidden, kExperts}),
          Format(Encoding::kF32, {kExpertHidden, kWidth, kExperts}),
          Format(Encoding::kF32, {kExperts})};
}

// One layer of the routed block, as a stage model.
StageModel Model(bool scaled = false) {
  StageModel m;
  m.instances.layers.push_back({Instance(scaled)});
  m.roles.resize(2);
  const std::array<const char*, 6> names{"norm", "router", "gate", "up", "down", "scale"};
  const auto formats = Formats();
  const std::size_t roles = scaled ? names.size() : std::size_t{kScale};
  for (std::size_t r = 0; r < roles; ++r) {
    m.roles[1].push_back({.name = names.at(r), .format = formats[r]});
  }
  return m;
}

}  // namespace moe

// ---------------------------------------------------------------- topologies

// `n` devices of `memory` bytes, every pair linked alike.
Topology Uniform(std::size_t n, std::uint64_t memory, double bytes_per_s = 50e9,
                 double latency_s = 2e-6) {
  Topology t;
  for (std::uint32_t i = 0; i < n; ++i) t.devices.push_back({.id = i, .memory = memory});
  for (std::uint32_t a = 0; a < n; ++a) {
    for (std::uint32_t b = a + 1; b < n; ++b) {
      t.links.push_back({.a = a, .b = b, .bytes_per_s = bytes_per_s, .latency_s = latency_s});
    }
  }
  return t;
}

// Two nodes of two devices: a fast link inside a node, a slow one between.
Topology TwoNodes(std::uint64_t memory) {
  Topology t;
  for (std::uint32_t i = 0; i < 4; ++i) t.devices.push_back({.id = i, .memory = memory});
  for (std::uint32_t a = 0; a < 4; ++a) {
    for (std::uint32_t b = a + 1; b < 4; ++b) {
      const bool inside = a / 2 == b / 2;
      t.links.push_back({.a = a,
                         .b = b,
                         .bytes_per_s = inside ? 100e9 : 10e9,
                         .latency_s = inside ? 1e-6 : 10e-6});
    }
  }
  return t;
}

constexpr std::uint64_t kAmple = std::uint64_t{1} << 40U;

Workload Mix() {
  return {
      .prefill_rows = 16, .decode_rows = 1, .decode_share = 0.5, .context = kContext, .slots = 1};
}

// ------------------------------------------------------------- equivalence

double Gap(std::span<const float> got, std::span<const float> want) {
  EXPECT_EQ(got.size(), want.size());
  double worst = 0.0;
  double scale = 1.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
    scale = std::max(scale, std::fabs(static_cast<double>(want[i])));
  }
  return worst / scale;
}

// The index of (i0, i1, i2, i3) in logical order.
std::size_t Ix(const format::Shape& s, std::uint64_t i0, std::uint64_t i1, std::uint64_t i2,
               std::uint64_t i3) {
  return static_cast<std::size_t>(i0 + (s.ne[0] * (i1 + (s.ne[1] * (i2 + (s.ne[2] * i3))))));
}

// A slice's values of a whole tensor's, or (back) the whole's from a slice.
template <typename T>
std::vector<T> Part(std::span<const T> whole, const format::Shape& shape, const Slice& slice) {
  if (slice.parts == 1) return {whole.begin(), whole.end()};
  format::Shape local = shape;
  local.ne.at(slice.dim) /= slice.parts;
  std::vector<T> out;
  for (std::uint64_t i3 = 0; i3 < local.ne[3]; ++i3) {
    for (std::uint64_t i2 = 0; i2 < local.ne[2]; ++i2) {
      for (std::uint64_t i1 = 0; i1 < local.ne[1]; ++i1) {
        for (std::uint64_t i0 = 0; i0 < local.ne[0]; ++i0) {
          std::array<std::uint64_t, 4> i{i0, i1, i2, i3};
          i.at(slice.dim) += slice.part * local.ne.at(slice.dim);
          out.push_back(whole[Ix(shape, i[0], i[1], i[2], i[3])]);
        }
      }
    }
  }
  return out;
}
void Unpart(std::span<const float> part, const format::Shape& shape, const Slice& slice,
            std::vector<float>& whole, std::vector<int>& covered) {
  format::Shape local = shape;
  local.ne.at(slice.dim) /= slice.parts;
  std::size_t k = 0;
  for (std::uint64_t i3 = 0; i3 < local.ne[3]; ++i3) {
    for (std::uint64_t i2 = 0; i2 < local.ne[2]; ++i2) {
      for (std::uint64_t i1 = 0; i1 < local.ne[1]; ++i1) {
        for (std::uint64_t i0 = 0; i0 < local.ne[0]; ++i0) {
          std::array<std::uint64_t, 4> i{i0, i1, i2, i3};
          i.at(slice.dim) += slice.part * local.ne.at(slice.dim);
          const std::size_t at = Ix(shape, i[0], i[1], i[2], i[3]);
          whole.at(at) = part[k++];
          ++covered.at(at);
        }
      }
    }
  }
}

// A step's leaves' values, by the primitive graph's leaf.
struct Leaves {
  std::map<std::uint32_t, std::vector<float>> floats{};
  std::map<std::uint32_t, std::vector<std::int64_t>> ints{};
};

void Bind(ref::Executor& e, ir::TensorId local, const Leaves& leaves, ir::TensorId source,
          const format::Shape& shape, const Slice& slice) {
  if (const auto at = leaves.ints.find(source.v); at != leaves.ints.end()) {
    const auto v = Part<std::int64_t>(at->second, shape, slice);
    auto bound = e.BindInts(local, v);
    ASSERT_TRUE(bound.has_value()) << bound.error().detail;
    return;
  }
  const auto at = leaves.floats.find(source.v);
  ASSERT_NE(at, leaves.floats.end()) << "leaf " << source.v;
  const auto v = Part<float>(at->second, shape, slice);
  auto bound = e.BindValues(local, v);
  ASSERT_TRUE(bound.has_value()) << bound.error().detail;
}

struct Check {
  std::vector<ir::TensorId> results{};  // compared values
  std::vector<ir::TensorId> tokens{};   // compared exactly (greedy)
  std::vector<ir::TensorId> state{};    // state leaves, compared after the step
  std::uint32_t rows = 0;
  std::uint32_t outputs = 0;
  double tolerance = 0.0;
  bool traffic = true;  // the collectives held against the model
};

// Runs `graph` whole and partitioned under `a` over the reference
// executor's simulated devices; compares results, tokens and state, and the
// partition's collectives with the traffic model. `after` receives the whole
// run's state leaves' values.
void Equivalent(const ir::Graph& graph, const Leaves& leaves, const Assignment& a,
                const StageModel& model, const Check& check,
                std::map<std::uint32_t, std::vector<float>>* after = nullptr) {
  // Whole.
  auto whole = ref::Executor::Create(graph);
  ASSERT_TRUE(whole.has_value()) << whole.error().detail;
  for (const ir::Node& node : graph.nodes()) {
    if (node.key.op != ir::Op::kInput && node.key.op != ir::Op::kWeight &&
        node.key.op != ir::Op::kState) {
      continue;
    }
    Bind(*whole, node.out[0], leaves, node.out[0], graph.tensor(node.out[0]).shape(), {});
  }
  auto ran = whole->Run();
  ASSERT_TRUE(ran.has_value()) << ran.error().detail;

  // Partitioned.
  auto parts = Partition(graph, a);
  ASSERT_TRUE(parts.has_value()) << parts.error();
  std::vector<ref::Executor> execs;
  std::map<std::uint32_t, std::size_t> index;
  for (const DeviceGraph& d : parts->devices) {
    auto e = ref::Executor::Create(d.graph);
    ASSERT_TRUE(e.has_value()) << "device " << d.device << ": " << e.error().detail;
    for (const LeafBinding& lb : d.leaves) {
      Bind(*e, lb.local, leaves, lb.source, graph.tensor(lb.source).shape(), lb.slice);
    }
    for (const ConstantValues& c : d.constants) {
      if (!c.ints.empty()) {
        const std::vector<std::int64_t> i64(c.ints.begin(), c.ints.end());
        ASSERT_TRUE(e->BindInts(c.local, i64));
      } else {
        ASSERT_TRUE(e->BindValues(c.local, c.values));
      }
    }
    index[d.device] = execs.size();
    execs.push_back(std::move(*e));
  }
  std::vector<ref::Executor*> devices;
  for (auto& e : execs) devices.push_back(&e);
  std::vector<ref::DeviceGroup> groups;
  for (const CollectiveGroup& g : parts->groups) {
    ref::DeviceGroup group{.group = g.group, .devices = {}};
    for (const std::uint32_t id : g.devices) {
      group.devices.push_back(static_cast<std::uint32_t>(index.at(id)));
    }
    groups.push_back(std::move(group));
  }
  auto all = ref::RunDevices(devices, groups);
  ASSERT_TRUE(all.has_value()) << all.error().detail;

  const auto placed = [&](ir::TensorId t) -> const PlacedTensor* {
    for (const PlacedTensor& p : parts->outputs) {
      if (p.source == t) return &p;
    }
    return nullptr;
  };
  for (const ir::TensorId t : check.results) {
    const PlacedTensor* p = placed(t);
    ASSERT_NE(p, nullptr);
    auto got = execs[index.at(p->device)].Values(p->local);
    auto want = whole->Values(t);
    ASSERT_TRUE(got.has_value() && want.has_value());
    EXPECT_LE(Gap(*got, *want), check.tolerance) << graph.name(t);
  }
  for (const ir::TensorId t : check.tokens) {
    const PlacedTensor* p = placed(t);
    ASSERT_NE(p, nullptr);
    auto got = execs[index.at(p->device)].Ints(p->local);
    auto want = whole->Ints(t);
    ASSERT_TRUE(got.has_value() && want.has_value());
    EXPECT_EQ(*got, *want);
  }
  for (const ir::TensorId s : check.state) {
    const format::Shape& shape = graph.tensor(s).shape();
    auto want = whole->Values(s);
    ASSERT_TRUE(want.has_value());
    std::vector<float> got(want->size(), 0.0F);
    std::vector<int> covered(want->size(), 0);
    std::size_t copies = 0;
    for (const DeviceGraph& d : parts->devices) {
      for (const LeafBinding& lb : d.leaves) {
        if (lb.source != s) continue;
        auto local = execs[index.at(d.device)].Values(lb.local);
        ASSERT_TRUE(local.has_value());
        if (lb.slice.parts == 1) {
          EXPECT_LE(Gap(*local, *want), check.tolerance) << graph.name(s);
          ++copies;
        } else {
          Unpart(*local, shape, lb.slice, got, covered);
        }
      }
    }
    if (copies == 0) {
      EXPECT_TRUE(std::ranges::all_of(covered, [](int c) { return c == 1; })) << graph.name(s);
      EXPECT_LE(Gap(got, *want), check.tolerance) << graph.name(s);
    }
    if (after != nullptr) (*after)[s.v] = *want;
  }
  // The collectives carry what the model counts.
  if (!check.traffic) return;
  const Traffic modelled = ModelTraffic(model, a, check.rows, check.outputs);
  EXPECT_EQ(parts->traffic.bytes, modelled.bytes);
  EXPECT_NEAR(parts->traffic.seconds, modelled.seconds, 1e-12 + (1e-9 * modelled.seconds));
}

// The tiny Gemma's weights, random and fixed by name.
std::map<std::string, std::vector<float>> Weights(const Tiny& t) {
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  std::map<std::string, std::vector<float>> out;
  for (const BoundTensor& b : t.tensors) {
    std::vector<float> v(ir::Elements(b.format.shape));
    const bool norm = b.format.shape.rank == 1;
    for (float& x : v) x = norm ? 1.0F + (0.25F * u(rng)) : 0.3F * u(rng);
    out[b.name] = std::move(v);
  }
  return out;
}

// A prefill step of two owners, then a greedy decode step of both, whole
// and partitioned under `a`, state carried between them.
void TinySteps(const Tiny& t, const Assignment& a, double tolerance) {
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model.has_value()) << model.error();
  const auto layout = components::StageLayout(t.spec, 0, kContext, kMaxRows);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  const auto weights = Weights(t);
  std::map<ir::StateRef, std::vector<float>> state;
  std::mt19937 rng(5);
  const std::array<StepShape, 2> shapes{
      StepShape{.segments = {Seg(0, 5, 0, 1), Seg(1, 3, 0, 1)}},
      StepShape{.segments = {Seg(0, 1, 5, 1, true), Seg(1, 1, 3, 1, true)}}};
  for (std::size_t si = 0; si < shapes.size(); ++si) {
    SCOPED_TRACE(std::format("step {}", si));
    const StepShape& shape = shapes[si];
    auto step = components::BuildStep(t.spec, 0, shape, {.tensors = t.tensors, .state = &*layout});
    ASSERT_TRUE(step.has_value()) << step.error();
    std::vector<std::int32_t> tokens;
    for (std::uint32_t i = 0; i < shape.rows(); ++i) {
      tokens.push_back(static_cast<std::int32_t>(rng() % kVocab));
    }
    Leaves leaves;
    for (const components::StepInput& in : step->inputs) {
      auto v = components::InputValues(in, shape, tokens);
      ASSERT_TRUE(v.has_value()) << v.error();
      leaves.ints[in.tensor.v] = *v;
    }
    for (const components::StepWeight& w : step->weights) {
      leaves.floats[w.tensor.v] = weights.at(w.parts.front().first);
    }
    Check check{.results = {step->logits},
                .tokens = {},
                .state = {},
                .rows = shape.rows(),
                .outputs = shape.outputs(),
                .tolerance = tolerance};
    if (step->greedy.valid()) check.tokens.push_back(step->greedy);
    for (const components::StepState& s : step->state) {
      const auto at = state.find(s.ref);
      leaves.floats[s.tensor.v] =
          at != state.end()
              ? at->second
              : std::vector<float>(ir::Elements(step->graph.tensor(s.tensor).shape()));
      check.state.push_back(s.tensor);
    }
    std::map<std::uint32_t, std::vector<float>> after;
    Equivalent(step->graph, leaves, a, *model, check, &after);
    for (const components::StepState& s : step->state) state[s.ref] = after.at(s.tensor.v);
  }
}

// The tiny Gemma's assignment for a pipeline on uniform devices; with
// `split_head`, its output head split over its stage as well.
Assignment TinyAssignment(const Tiny& t, std::size_t devices, const Pipeline& pipeline,
                          bool split_head = false) {
  auto model = ModelOf(t.spec, 0, t.tensors);
  EXPECT_TRUE(model.has_value());
  auto a = AssignPipeline(*model, Uniform(devices, kAmple), Mix(), pipeline);
  EXPECT_TRUE(a.has_value()) << a.error();
  if (split_head && a->output->devices.size() > 1) {
    a->output->option = 0;
    a->output->split = SplitKind::kColumns;
  }
  return *a;
}

// Tensor parallelism splits within whole values: a step's partial sums
// differ from the whole product's in rounding only, which the F16 caches
// may round a step apart.
constexpr double kTpTolerance = 2e-3;

TEST(PartitionTest, TinyGemmaComponentsSplitAsDeclared) {
  const Tiny t = TinyGemma(false);
  ASSERT_TRUE(components::ValidateSpec(t.spec, t.tensors));
  const Assignment a = TinyAssignment(t, 4, {.stages = {{0, 1, 2, 3}}, .first_layer = {0}});
  ASSERT_EQ(a.per_layer.size(), kLayers);
  for (const auto& layer : a.per_layer) {
    ASSERT_EQ(layer.size(), 2U);
    EXPECT_EQ(layer[0].split, SplitKind::kHeads);
    EXPECT_EQ(layer[1].split, SplitKind::kColumns);
    EXPECT_EQ(layer[0].devices, (std::vector<std::uint32_t>{0, 1, 2, 3}));
  }
  // The embedding, final norm and head fit whole on each device.
  EXPECT_EQ(a.embedding->split, SplitKind::kNone);
  EXPECT_EQ(a.output->split, SplitKind::kNone);
  EXPECT_EQ(a.stage_of_layer, std::vector<std::uint32_t>(kLayers, 0));
}

TEST(PartitionTest, TensorParallelOverFourDevices) {
  const Tiny t = TinyGemma(false);
  TinySteps(t, TinyAssignment(t, 4, {.stages = {{0, 1, 2, 3}}, .first_layer = {0}}), kTpTolerance);
}

TEST(PartitionTest, TensorParallelWithTheHeadSplitByVocabulary) {
  for (const bool tied : {false, true}) {
    SCOPED_TRACE(tied ? "tied" : "untied");
    const Tiny t = TinyGemma(tied);
    TinySteps(t, TinyAssignment(t, 4, {.stages = {{0, 1, 2, 3}}, .first_layer = {0}}, true),
              kTpTolerance);
  }
}

TEST(PartitionTest, TensorParallelOverOddAndUnevenGroups) {
  // Three ways: four KV heads (eight query heads) a device, odd parts of
  // every split. Eight ways: twelve KV heads do not divide, so attention
  // runs whole on every device beside the split feed-forward blocks and
  // head.
  const Tiny t = TinyGemma(true);
  for (const std::size_t n : {std::size_t{3}, std::size_t{8}}) {
    SCOPED_TRACE(std::format("{} devices", n));
    std::vector<std::uint32_t> all;
    for (std::uint32_t i = 0; i < n; ++i) all.push_back(i);
    const Assignment a = TinyAssignment(t, n, {.stages = {all}, .first_layer = {0}}, true);
    for (const auto& layer : a.per_layer) {
      EXPECT_EQ(layer[0].split, n == 3 ? SplitKind::kHeads : SplitKind::kNone);
      EXPECT_EQ(layer[1].split, SplitKind::kColumns);
    }
    TinySteps(t, a, kTpTolerance);
  }
}

// Attention by heads where its mask or ALiBi cannot follow the split: a
// mask broadcast over heads is read whole where each part's heads read it
// alike, split where it is per head, and otherwise (like ALiBi, whose
// slopes follow the global head) gathers the heads and runs whole.
TEST(PartitionTest, AttentionMasksAndAlibiUnderAHeadSplit) {
  const Tiny t = TinyGemma(false);
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model.has_value());
  const Assignment a = TinyAssignment(t, 4, {.stages = {{0, 1, 2, 3}}, .first_layer = {0}});
  const std::uint16_t c = a.per_layer[0][0].component;
  ASSERT_EQ(a.per_layer[0][0].split, SplitKind::kHeads);
  constexpr std::uint64_t kRows = 5;
  struct Case {
    std::uint64_t mask_heads;
    float max_bias;
  };
  for (const Case k : {Case{1, 0.0F}, Case{kHeads, 0.0F}, Case{2, 0.0F}, Case{8, 0.0F},
                       Case{1, 8.0F}, Case{kHeads, 8.0F}}) {
    SCOPED_TRACE(std::format("mask heads {}, max_bias {}", k.mask_heads, k.max_bias));
    ir::Builder b;
    b.SetComponent(c);
    const auto f32 = [](std::initializer_list<std::uint64_t> ne) {
      return ir::Dense(Encoding::kF32, ir::ShapeOf(ne));
    };
    const auto weight = [&](std::uint16_t role, std::initializer_list<std::uint64_t> ne) {
      return b.Weight(f32(ne), {.stage = 0, .component = c, .role = role, .layer = 0});
    };
    const ir::TensorId x = b.Input(f32({kWidth, kRows}));
    const ir::TensorId mask = b.Input(f32({kRows, kRows, k.mask_heads}));
    const auto heads = [&](std::uint16_t role, std::uint64_t n) {
      const ir::TensorId p = b.MatMul(weight(role, {kWidth, n * kDim}), x);
      return b.Permute(b.Reshape(p, ir::ShapeOf({kDim, n, kRows})), {0, 2, 1, 3});
    };
    const ir::TensorId q = heads(1, kHeads);
    const ir::TensorId kk = heads(2, kKvHeads);
    const ir::TensorId v = heads(3, kKvHeads);
    const ir::TensorId att =
        b.Attention(q, kk, v, mask, {},
                    {.scale = 0.5F, .max_bias = k.max_bias, .accumulate = ir::Accumulate::kF32});
    const ir::TensorId out = b.MatMul(weight(4, {kHeads * kDim, kWidth}),
                                      b.Reshape(att, ir::ShapeOf({kHeads * kDim, kRows})));
    b.Output(out);
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g.has_value()) << g.error().detail;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    Leaves leaves;
    for (const ir::Node& node : g->nodes()) {
      if (node.key.op != ir::Op::kInput && node.key.op != ir::Op::kWeight) continue;
      std::vector<float> values(ir::Elements(g->tensor(node.out[0]).shape()));
      for (float& value : values) value = u(rng);
      leaves.floats[node.out[0].v] = std::move(values);
    }
    Equivalent(
        *g, leaves, a, *model,
        {.results = {out}, .rows = kRows, .outputs = kRows, .tolerance = 1e-5, .traffic = false});
  }
}

TEST(PartitionTest, PipelineOverFourDevicesIsExact) {
  const Tiny t = TinyGemma(true);
  TinySteps(t, TinyAssignment(t, 4, {.stages = {{0}, {1}, {2}, {3}}, .first_layer = {0, 2, 4, 5}}),
            0.0);
}

TEST(PartitionTest, TensorTimesPipelineOverFourDevices) {
  for (const bool tied : {false, true}) {
    SCOPED_TRACE(tied ? "tied" : "untied");
    const Tiny t = TinyGemma(tied);
    TinySteps(t, TinyAssignment(t, 4, {.stages = {{0, 1}, {2, 3}}, .first_layer = {0, 3}}, true),
              kTpTolerance);
  }
}

TEST(PartitionTest, MixesOverSixDevices) {
  const Tiny t = TinyGemma(true);
  const std::array<Pipeline, 4> pipelines{
      Pipeline{.stages = {{0, 1, 2, 3, 4, 5}}, .first_layer = {0}},
      Pipeline{.stages = {{0, 1, 2}, {3, 4, 5}}, .first_layer = {0, 3}},
      Pipeline{.stages = {{0, 1}, {2, 3}, {4, 5}}, .first_layer = {0, 2, 4}},
      Pipeline{.stages = {{0}, {1}, {2}, {3}, {4}, {5}}, .first_layer = {0, 1, 2, 3, 4, 5}}};
  for (std::size_t i = 0; i < pipelines.size(); ++i) {
    SCOPED_TRACE(std::format("pipeline {}", i));
    TinySteps(t, TinyAssignment(t, 6, pipelines[i], i == 1), i == 3 ? 0.0 : kTpTolerance);
  }
}

TEST(PartitionTest, AStreamCrossesTheSlowLinkOnce) {
  // A stage on one node's devices hands its stream to the other node's: one
  // send over the slow link, one over the fast link inside the node.
  const Tiny t = TinyGemma(false);
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model.has_value());
  auto a = AssignPipeline(*model, TwoNodes(kAmple), Mix(),
                          {.stages = {{0, 1}, {2, 3}}, .first_layer = {0, 3}});
  ASSERT_TRUE(a.has_value()) << a.error();
  TinySteps(t, *a, kTpTolerance);
  // Decode: twelve two-device all-reduces of one row and the boundary.
  const std::uint64_t row = kWidth * 4;
  EXPECT_EQ(a->decode.bytes, (12 * 2 * row) + (2 * row));
  EXPECT_NEAR(
      a->decode.seconds,
      (12 * 2 * ((row / 2.0 / 100e9) + 1e-6)) + ((row / 10e9) + 10e-6) + ((row / 100e9) + 1e-6),
      1e-15);
}

// ---------------------------------------------------------- expert parallel

void MoeStep(const Assignment& a, double tolerance, bool scaled = false) {
  const components::Instance in = moe::Instance(scaled);
  const StepShape shape{.segments = {Seg(0, 5, 0, 1), Seg(1, 3, 0, 1)}};
  auto graph = components::BuildComponent(in, shape, moe::Formats(), {});
  ASSERT_TRUE(graph.has_value()) << graph.error();
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  Leaves leaves;
  const auto formats = moe::Formats();
  for (std::size_t r = 0; r < formats.size(); ++r) {
    if (!graph->weights[r].valid()) continue;
    std::vector<float> v(ir::Elements(formats[r].shape));
    for (float& x : v) {
      x = r == moe::kNorm || r == moe::kScale ? 1.0F + (0.25F * u(rng)) : 0.3F * u(rng);
    }
    leaves.floats[graph->weights[r].v] = std::move(v);
  }
  std::vector<float> stream(kWidth * shape.rows());
  for (float& x : stream) x = 2.0F * u(rng);
  leaves.floats[graph->stream_in.v] = stream;
  const StageModel model = moe::Model(scaled);
  Equivalent(
      graph->step.graph, leaves, a, model,
      {.results = {graph->stream_out}, .rows = shape.rows(), .outputs = 0, .tolerance = tolerance});
}

TEST(PartitionTest, ExpertParallelOverFourAndSixDevices) {
  const StageModel model = moe::Model();
  for (const std::uint32_t n : {4U, 6U}) {
    SCOPED_TRACE(std::format("{} devices", n));
    std::vector<std::uint32_t> all;
    for (std::uint32_t i = 0; i < n; ++i) all.push_back(i);
    auto a =
        AssignPipeline(model, Uniform(n, kAmple), Mix(), {.stages = {all}, .first_layer = {0}});
    ASSERT_TRUE(a.has_value()) << a.error();
    ASSERT_EQ(a->per_layer[0][0].split, SplitKind::kExperts);  // the first option
    MoeStep(*a, 1e-5);
    // The same block split inside each expert instead.
    a->per_layer[0][0].option = 1;
    a->per_layer[0][0].split = SplitKind::kColumns;
    MoeStep(*a, 1e-5);
  }
}

// With each slot's learned scale gathered before the reduce (ffn.moe's
// Gemma 4 form, op.h kExpertReduce's [1, U, T] scale): each device's
// foreign products/weights/scales selected before the reduction.
TEST(PartitionTest, ExpertParallelWithGatheredScales) {
  const StageModel model = moe::Model(true);
  for (const std::uint32_t n : {4U, 6U}) {
    SCOPED_TRACE(std::format("{} devices", n));
    std::vector<std::uint32_t> all;
    for (std::uint32_t i = 0; i < n; ++i) all.push_back(i);
    auto a =
        AssignPipeline(model, Uniform(n, kAmple), Mix(), {.stages = {all}, .first_layer = {0}});
    ASSERT_TRUE(a.has_value()) << a.error();
    ASSERT_EQ(a->per_layer[0][0].split, SplitKind::kExperts);
    MoeStep(*a, 1e-5, true);
    a->per_layer[0][0].option = 1;
    a->per_layer[0][0].split = SplitKind::kColumns;
    MoeStep(*a, 1e-5, true);
  }
}

TEST(PartitionTest, ExpertOwnershipSelectsForeignNonfiniteContributionsToLiteralZero) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (bool gathered : {false, true}) {
    for (const auto& values :
         std::array{std::array{nan, 3.0F, 2.0F, 1.0F, 12.0F}, std::array{inf, 3.0F, inf, 1.0F, inf},
                    std::array{inf, 3.0F, 2.0F, inf, inf}, std::array{0.0F, inf, 2.0F, 1.0F, inf},
                    std::array{0.0F, nan, 2.0F, 1.0F, nan}}) {
      SCOPED_TRACE(std::format("gathered {}, foreign {}, own {}, weight {}, scale {}", gathered,
                               values[0], values[1], values[2], values[3]));
      ir::Builder builder;
      builder.SetComponent(1);
      const auto matrix = builder.Weight(Format(Encoding::kF32, {1, 1, moe::kExperts}),
                                         {.component = 1, .role = moe::kGate});
      const auto input = builder.Input(Format(Encoding::kF32, {1, 1, 1}));
      const auto ids = builder.Input(Format(Encoding::kI32, {1, 1}));
      const auto weights = builder.Input(Format(Encoding::kF32, {1, 1}));
      const auto scales = builder.Input(gathered ? Format(Encoding::kF32, {1, 1, 1})
                                                 : Format(Encoding::kF32, {moe::kExperts}));
      const auto products = builder.MatMulId(matrix, input, ids);
      const auto result = builder.ExpertReduce(products, weights, ids, scales);
      builder.Output(result);
      auto graph = std::move(builder).Finish();
      ASSERT_TRUE(graph) << graph.error().detail;
      Assignment assignment{
          .topology = Uniform(2, kAmple), .stages = {{0, 1}}, .stage_of_layer = {0}};
      assignment.per_layer.push_back({ComponentPlacement{.component = 1,
                                                         .def = &moe::Def(),
                                                         .split = SplitKind::kExperts,
                                                         .option = 0,
                                                         .devices = {0, 1}}});
      auto partition = Partition(*graph, assignment);
      ASSERT_TRUE(partition) << partition.error();
      Leaves leaves;
      leaves.floats[matrix.v] = std::vector<float>(moe::kExperts, 1.0F);
      leaves.floats[matrix.v][0] = values[0];  // foreign rank's fallback expert
      leaves.floats[matrix.v][7] = values[1];  // selected expert belongs to rank 1
      leaves.floats[input.v] = {2};
      leaves.ints[ids.v] = {7};
      leaves.floats[weights.v] = {values[2]};
      leaves.floats[scales.v] = std::vector<float>(gathered ? 1 : moe::kExperts, values[3]);
      std::vector<ref::Executor> executors;
      for (const auto& device : partition->devices) {
        auto executor = ref::Executor::Create(device.graph);
        ASSERT_TRUE(executor) << executor.error().detail;
        for (const auto& leaf : device.leaves)
          Bind(*executor, leaf.local, leaves, leaf.source, graph->tensor(leaf.source).shape(),
               leaf.slice);
        for (const auto& constant : device.constants) {
          if (!constant.ints.empty()) {
            EXPECT_EQ(std::as_bytes(std::span(constant.ints)).size(),
                      constant.ints.size() * sizeof(std::int32_t));
            EXPECT_EQ(constant.ints.size(), moe::kExperts);
            const std::vector<std::int64_t> i64(constant.ints.begin(), constant.ints.end());
            ASSERT_TRUE(executor->BindInts(constant.local, i64));
          } else {
            ASSERT_TRUE(executor->BindValues(constant.local, constant.values));
          }
        }
        executors.push_back(std::move(*executor));
      }
      std::vector<ref::Executor*> devices;
      for (auto& executor : executors) devices.push_back(&executor);
      std::vector<ref::DeviceGroup> groups;
      for (const auto& group : partition->groups)
        groups.push_back({.group = group.group, .devices = group.devices});
      auto run = ref::RunDevices(devices, groups);
      ASSERT_TRUE(run) << run.error().detail;
      ASSERT_EQ(partition->outputs.size(), 1U);
      const auto& output = partition->outputs[0];
      auto got = executors[output.device].Values(output.local);
      ASSERT_TRUE(got);
      ASSERT_EQ(got->size(), 1U);
      if (std::isnan(values[4]))
        EXPECT_TRUE(std::isnan((*got)[0]));
      else
        EXPECT_EQ((*got)[0], values[4]);
      // Every neutralized product/weight on the foreign rank is literal zero;
      // every neutralized scale is one, even when its source is nonfinite.
      std::size_t selections = 0;
      for (const auto& node : partition->devices[0].graph.nodes()) {
        if (node.key.op != ir::Op::kSelect) continue;
        auto selected = executors[0].Values(node.out[0]);
        ASSERT_TRUE(selected);
        const auto foreign_values = selections == 2 && !gathered
                                        ? std::span(*selected).last(moe::kExperts / 2)
                                        : std::span(*selected);
        EXPECT_TRUE(std::ranges::all_of(
            foreign_values, [&](float value) { return value == (selections == 2 ? 1.0F : 0.0F); }));
        if (selections == 2 && !gathered) {
          const auto owned_values = std::span(*selected).first(moe::kExperts / 2);
          EXPECT_TRUE(std::ranges::all_of(owned_values, [&](float value) {
            return std::isnan(values[3]) ? std::isnan(value) : value == values[3];
          }));
        }
        ++selections;
      }
      EXPECT_EQ(selections, 3U);
    }
  }
}

// ------------------------------------------------------------ the choices

// The tiny Gemma's bytes, from its shapes: a layer's on one device and on
// each of two (Q, K, V, the output product and the feed-forward products
// halved; norms whole; F16 K and V of 256 cells halved), the embedding and
// head tables, and the final norm.
constexpr std::uint64_t kAttention = 4 * kWidth * ((2 * kHeads * kDim) + (2 * kKvHeads * kDim));
constexpr std::uint64_t kAttentionNorms = 4 * ((2 * kWidth) + (2 * kDim));
constexpr std::uint64_t kFfn = 4 * 3 * kWidth * kHidden;
constexpr std::uint64_t kFfnNorms = 4 * 2 * kWidth;
constexpr std::uint64_t kKv = 2 * (kKvHeads * kDim) * 256 * 2;
constexpr std::uint64_t kLayer1 = kAttention + kAttentionNorms + kFfn + kFfnNorms + kKv;
constexpr std::uint64_t kLayer2 =
    (kAttention / 2) + kAttentionNorms + (kFfn / 2) + kFfnNorms + (kKv / 2);
constexpr std::uint64_t kTable = 4 * kWidth * kVocab;
constexpr std::uint64_t kFinalNorm = 4 * kWidth;

TEST(PartitionTest, TheModelsBytesAreTheDerivedOnes) {
  const Tiny t = TinyGemma(false);
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model.has_value());
  auto one =
      AssignPipeline(*model, Uniform(1, kAmple), Mix(), {.stages = {{0}}, .first_layer = {0}});
  ASSERT_TRUE(one.has_value()) << one.error();
  EXPECT_EQ(one->memory.at(0).second, (kLayers * kLayer1) + (2 * kTable) + kFinalNorm);
  auto two =
      AssignPipeline(*model, Uniform(2, kAmple), Mix(), {.stages = {{0, 1}}, .first_layer = {0}});
  ASSERT_TRUE(two.has_value()) << two.error();
  EXPECT_EQ(two->memory.at(0).second, (kLayers * kLayer2) + (2 * kTable) + kFinalNorm);
}

TEST(PartitionTest, AModelThatFitsStaysOnOneDevice) {
  const Tiny t = TinyGemma(true);
  auto a = Assign(t.spec, t.tensors, Uniform(4, kAmple), Mix());
  ASSERT_TRUE(a.has_value()) << a.error();
  EXPECT_EQ(a->stages, (std::vector<std::vector<std::uint32_t>>{{0}}));
  EXPECT_EQ(a->modelled_seconds_per_step, 0.0);
  EXPECT_EQ(a->modelled_bytes_per_step, 0.0);
}

TEST(PartitionTest, UniformLinksTakeTheFewestPipelineStages) {
  // Room for three layers a device: two stages of three (one boundary)
  // beat any tensor split (two all-reduces a layer).
  const Tiny t = TinyGemma(false);
  {
    const std::uint64_t m = (3 * kLayer1) + kTable + kFinalNorm + 64;
    ASSERT_LT(m, (4 * kLayer1));
    auto a = Assign(t.spec, t.tensors, Uniform(4, m), Mix());
    ASSERT_TRUE(a.has_value()) << a.error();
    EXPECT_EQ(a->stages, (std::vector<std::vector<std::uint32_t>>{{0}, {1}}));
    EXPECT_EQ(a->stage_of_layer, (std::vector<std::uint32_t>{0, 0, 0, 1, 1, 1}));
    EXPECT_EQ(a->decode.bytes, kWidth * 4);  // one row across one boundary
  }
  {
    // Room for two: three stages of two, two boundaries, rather than two
    // stages of three layers split over two devices each.
    const std::uint64_t m = (2 * kLayer1) + kTable + kFinalNorm + 64;
    ASSERT_LT(m, 3 * kLayer1);
    auto a = Assign(t.spec, t.tensors, Uniform(4, m), Mix());
    ASSERT_TRUE(a.has_value()) << a.error();
    EXPECT_EQ(a->stages, (std::vector<std::vector<std::uint32_t>>{{0}, {1}, {2}}));
    EXPECT_EQ(a->stage_of_layer, (std::vector<std::uint32_t>{0, 0, 1, 1, 2, 2}));
  }
}

TEST(PartitionTest, FastPairsSplitTensorsAndTheSlowLinkCutsThePipeline) {
  // A device holds one layer but not two, so six layers on four devices
  // need tensor splits; a pair holds three layers split in halves, and four
  // ways do not fit beside the whole tables. Tensor parallelism inside each
  // node and the pipeline cut across the slow link beats pairing devices
  // across it (all-reduces over the slow link).
  const Tiny t = TinyGemma(false);
  const std::uint64_t m = (3 * kLayer2) + kTable + kFinalNorm + 64;
  ASSERT_LT(m, 2 * kLayer1);
  ASSERT_GE(m, kLayer1 + kTable);
  auto a = Assign(t.spec, t.tensors, TwoNodes(m), Mix());
  ASSERT_TRUE(a.has_value()) << a.error();
  EXPECT_EQ(a->stages, (std::vector<std::vector<std::uint32_t>>{{0, 1}, {2, 3}}));
  EXPECT_EQ(a->stage_of_layer, (std::vector<std::uint32_t>{0, 0, 0, 1, 1, 1}));
  for (const auto& layer : a->per_layer) {
    EXPECT_EQ(layer[0].split, SplitKind::kHeads);
    EXPECT_EQ(layer[1].split, SplitKind::kColumns);
  }
  // The alternatives the search rejected cost more.
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model.has_value());
  auto across = AssignPipeline(*model, TwoNodes(m), Mix(),
                               {.stages = {{0, 2}, {1, 3}}, .first_layer = {0, 3}});
  ASSERT_TRUE(across.has_value()) << across.error();
  EXPECT_GT(across->modelled_seconds_per_step, a->modelled_seconds_per_step);
  EXPECT_FALSE(
      AssignPipeline(*model, TwoNodes(m), Mix(), {.stages = {{0, 1, 2, 3}}, .first_layer = {0}}));
}

TEST(PartitionTest, MemoryNoPlacementHoldsIsRefused) {
  const Tiny t = TinyGemma(false);
  auto a = Assign(t.spec, t.tensors, Uniform(4, kLayer2 / 2), Mix());
  EXPECT_FALSE(a.has_value());
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model.has_value());
  auto p = AssignPipeline(*model, Uniform(2, kLayer1), Mix(),
                          {.stages = {{0}, {1}}, .first_layer = {0, 3}});
  ASSERT_FALSE(p.has_value());
  EXPECT_NE(p.error().find("memory"), std::string::npos) << p.error();
}

TEST(PartitionTest, RoutedExpertsArePlacedWhole) {
  // Each device holds a quarter of the experts but not a third: four ways,
  // experts placed whole (both splits move one all-reduce; expert
  // parallelism is the first option).
  const StageModel model = moe::Model();
  const std::uint64_t arrays = 3 * 4 * kWidth * moe::kExpertHidden * moe::kExperts;
  const std::uint64_t rest = (4 * kWidth) + (4 * kWidth * moe::kExperts);
  const std::uint64_t m = (arrays / 4) + rest + 64;
  ASSERT_LT(m, (arrays / 3) + rest);
  auto a = Assign(model, Uniform(4, m), Mix());
  ASSERT_TRUE(a.has_value()) << a.error();
  EXPECT_EQ(a->stages, (std::vector<std::vector<std::uint32_t>>{{0, 1, 2, 3}}));
  EXPECT_EQ(a->per_layer[0][0].split, SplitKind::kExperts);
}

TEST(PartitionTest, SplitsKeepWholeBlocksOfTheirEncodings) {
  // Q8_0 down products of 96 hidden columns: halves of 48 are not whole
  // 32-value blocks, thirds are.
  const Tiny t = TinyGemma(false, Encoding::kQ8_0, 96);
  ASSERT_TRUE(components::ValidateSpec(t.spec, t.tensors));
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model.has_value());
  auto two =
      AssignPipeline(*model, Uniform(2, kAmple), Mix(), {.stages = {{0, 1}}, .first_layer = {0}});
  ASSERT_TRUE(two.has_value()) << two.error();
  EXPECT_EQ(two->per_layer[0][1].split, SplitKind::kNone);
  auto three = AssignPipeline(*model, Uniform(3, kAmple), Mix(),
                              {.stages = {{0, 1, 2}}, .first_layer = {0}});
  ASSERT_TRUE(three.has_value()) << three.error();
  EXPECT_EQ(three->per_layer[0][1].split, SplitKind::kColumns);
}

TEST(PartitionTest, GemmaTwoOnTwoSparks) {
  // The two Sparks' direct link as measured (docs/environment.md: NCCL
  // SendRecv 22.35 GB/s; 1.4 µs one-way latency) and Gemma 2 2B: it fits
  // one; given less than it needs on each, a pipeline of two.
  auto values = fam::Gemma2Values(fam::Gemma2_2B());
  ASSERT_TRUE(values.has_value());
  const spec::Source source = test_support::Gemma2Source();
  auto s = spec::Synthesize(spec::mappings::Gemma2Mapping(), source, *values);
  ASSERT_TRUE(s.has_value()) << s.error();
  const auto sparks = [](std::uint64_t memory) {
    return Topology{.devices = {{.id = 0, .memory = memory}, {.id = 1, .memory = memory}},
                    .links = {{.a = 0, .b = 1, .bytes_per_s = 22.35e9, .latency_s = 1.4e-6}}};
  };
  const Workload w{
      .prefill_rows = 512, .decode_rows = 1, .decode_share = 0.9, .context = 8192, .slots = 2};
  auto one = Assign(*s, source.tensors, sparks(std::uint64_t{100} << 30U), w);
  ASSERT_TRUE(one.has_value()) << one.error();
  EXPECT_EQ(one->stages, (std::vector<std::vector<std::uint32_t>>{{0}}));
  const std::uint64_t whole = one->memory.at(0).second;
  auto two = Assign(*s, source.tensors, sparks(whole * 7 / 10), w);
  ASSERT_TRUE(two.has_value()) << two.error();
  EXPECT_EQ(two->stages, (std::vector<std::vector<std::uint32_t>>{{0}, {1}}));
  EXPECT_EQ(two->decode.bytes, 2304U * 4);  // one row of Gemma 2 2B's stream
}

// ------------------------------------------------- the search, by brute force

// Devices whose links are fast or slow at random, some made twins of
// another (equal memory, equal links to every other device).
Topology RandomTopology(std::size_t n, std::uint64_t memory, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::vector<bool>> fast(n, std::vector<bool>(n, false));
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = a + 1; b < n; ++b) fast[a][b] = fast[b][a] = (rng() % 2) == 0;
  }
  for (std::size_t twins = 0; twins < 2; ++twins) {
    const std::size_t a = rng() % n;
    const std::size_t b = rng() % n;
    if (a == b) continue;
    for (std::size_t k = 0; k < n; ++k) {
      if (k != a && k != b) fast[b][k] = fast[k][b] = fast[a][k];
    }
  }
  Topology t;
  for (std::uint32_t i = 0; i < n; ++i) t.devices.push_back({.id = i, .memory = memory});
  for (std::uint32_t a = 0; a < n; ++a) {
    for (std::uint32_t b = a + 1; b < n; ++b) {
      t.links.push_back({.a = a,
                         .b = b,
                         .bytes_per_s = fast[a][b] ? 100e9 : 10e9,
                         .latency_s = fast[a][b] ? 1e-6 : 10e-6});
    }
  }
  return t;
}

// The search's ring order of a group (ids = slots here): by twin class (a
// class numbered by its lowest device), then id.
std::vector<std::uint32_t> RingOrder(const Topology& t, std::vector<std::uint32_t> ids) {
  const std::size_t n = t.devices.size();
  const auto link = [&](std::size_t a, std::size_t b) -> std::optional<std::pair<double, double>> {
    for (const Link& l : t.links) {
      if ((l.a == a && l.b == b) || (l.a == b && l.b == a)) {
        return std::pair{l.bytes_per_s, l.latency_s};
      }
    }
    return std::nullopt;
  };
  std::vector<std::size_t> cls(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (cls[i] != n) continue;
    cls[i] = i;
    for (std::size_t j = i + 1; j < n; ++j) {
      if (cls[j] != n || t.devices[i].memory != t.devices[j].memory ||
          t.devices[i].performance != t.devices[j].performance)
        continue;
      bool twin = true;
      for (std::size_t k = 0; k < n; ++k) {
        if (k != i && k != j) twin = twin && link(i, k) == link(j, k);
      }
      if (twin) cls[j] = i;
    }
  }
  std::ranges::sort(ids, {}, [&](std::uint32_t id) { return std::pair(cls[id], id); });
  return ids;
}

struct Best {
  double cost = std::numeric_limits<double>::infinity();
  std::size_t devices = 0;
  std::size_t stages = 0;
};

// Every pipeline over the topology's devices, each stage's group in ring
// order, evaluated by AssignPipeline: the least cost, then devices, then
// stages.
Best Enumerate(const StageModel& model, const Topology& t, const Workload& w,
               std::uint32_t layers) {
  Best best;
  const auto n = static_cast<std::uint32_t>(t.devices.size());
  Pipeline p;
  std::size_t used_devices = 0;
  const auto visit = [&](auto&& self, std::uint32_t first, std::uint32_t used) -> void {
    for (std::uint32_t mask = 1; mask < (1U << n); ++mask) {
      if ((mask & used) != 0) continue;
      std::vector<std::uint32_t> ids;
      for (std::uint32_t i = 0; i < n; ++i) {
        if ((mask & (1U << i)) != 0) ids.push_back(i);
      }
      p.stages.push_back(RingOrder(t, ids));
      p.first_layer.push_back(first);
      used_devices += ids.size();
      for (std::uint32_t end = first + 1; end <= layers; ++end) {
        if (end < layers) {
          self(self, end, used | mask);
          continue;
        }
        auto a = AssignPipeline(model, t, w, p);
        if (!a) continue;
        const double c = a->modelled_seconds_per_step;
        const double eps = 1e-12 * std::max({1.0, c, std::isinf(best.cost) ? 1.0 : best.cost});
        if (c < best.cost - eps ||
            (c <= best.cost + eps &&
             std::pair(used_devices, p.stages.size()) < std::pair(best.devices, best.stages))) {
          best = {.cost = c, .devices = used_devices, .stages = p.stages.size()};
        }
      }
      used_devices -= ids.size();
      p.stages.pop_back();
      p.first_layer.pop_back();
    }
  };
  visit(visit, 0, 0);
  return best;
}

void ExpectSearchExact(const StageModel& model, const Topology& t, const Workload& w,
                       std::uint32_t layers) {
  const Best want = Enumerate(model, t, w, layers);
  auto got = Assign(model, t, w);
  ASSERT_EQ(got.has_value(), std::isfinite(want.cost)) << (got ? "" : got.error());
  if (!got) return;
  std::size_t devices = 0;
  for (const auto& stage : got->stages) devices += stage.size();
  EXPECT_NEAR(got->modelled_seconds_per_step, want.cost, 1e-12 * std::max(1.0, want.cost));
  EXPECT_EQ(devices, want.devices);
  EXPECT_EQ(got->stages.size(), want.stages);
}

TEST(PartitionTest, TheSearchIsExactOverRandomTopologiesWithTwins) {
  // The routed block's one layer on six devices with room for a quarter of
  // its experts (four or six ways): every group's ring, twins included.
  {
    const StageModel model = moe::Model();
    const std::uint64_t arrays = 3 * 4 * kWidth * moe::kExpertHidden * moe::kExperts;
    const std::uint64_t rest = (4 * kWidth) + (4 * kWidth * moe::kExperts);
    const std::uint64_t m = (arrays / 4) + rest + 64;
    for (std::uint32_t seed = 0; seed < 64; ++seed) {
      SCOPED_TRACE(std::format("routed block, seed {}", seed));
      ExpectSearchExact(model, RandomTopology(6, m, seed), Mix(), 1);
    }
  }
  // The tiny Gemma's six layers on four devices, a device holding one
  // layer but not two: pipelines, tensor splits and their boundaries.
  {
    const Tiny t = TinyGemma(false);
    auto model = ModelOf(t.spec, 0, t.tensors);
    ASSERT_TRUE(model.has_value());
    const std::uint64_t m = (3 * kLayer2) + kTable + kFinalNorm + 64;
    for (std::uint32_t seed = 0; seed < 4; ++seed) {
      SCOPED_TRACE(std::format("tiny Gemma, seed {}", seed));
      ExpectSearchExact(*model, RandomTopology(4, m, seed), Mix(), kLayers);
    }
  }
}

TEST(PartitionTest, SparseSideHolderSearchMatchesEnumeration) {
  const Tiny t = TinyHcSide(true);
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model) << model.error();
  for (const std::uint64_t memory : {600U, 900U, 1300U})
    for (std::uint32_t seed = 0; seed < 4; ++seed) {
      SCOPED_TRACE(std::format("memory {} seed {}", memory, seed));
      ExpectSearchExact(*model, RandomTopology(4, memory, seed), Mix(), 4);
    }
}

TEST(PartitionTest, ASearchPastItsBoundIsRefusedPromptly) {
  // Sixteen devices without twins (every link its own bandwidth): 65,535
  // candidate groups, refused at once; AssignPipeline still evaluates a
  // configured placement on them.
  const Tiny t = TinyGemma(false);
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model.has_value());
  Topology topo;
  for (std::uint32_t i = 0; i < kMaxDevices; ++i)
    topo.devices.push_back({.id = i, .memory = kAmple});
  double bw = 10e9;
  for (std::uint32_t a = 0; a < kMaxDevices; ++a) {
    for (std::uint32_t b = a + 1; b < kMaxDevices; ++b) {
      topo.links.push_back({.a = a, .b = b, .bytes_per_s = bw, .latency_s = 1e-6});
      bw += 1e8;
    }
  }
  auto a = Assign(*model, topo, Mix());
  ASSERT_FALSE(a.has_value());
  EXPECT_NE(a.error().find("past the exact search"), std::string::npos) << a.error();
  EXPECT_TRUE(
      AssignPipeline(*model, topo, Mix(), {.stages = {{0, 1}, {2, 3}}, .first_layer = {0, 3}}));
}

// ------------------------------------------------------------- refusals

TEST(PartitionTest, TopologiesAndGraphsItCannotTakeAreRefused) {
  EXPECT_FALSE(Validate(Topology{}));
  EXPECT_FALSE(Validate({.devices = {{.id = 1}, {.id = 1}}, .links = {}}));
  EXPECT_FALSE(Validate({.devices = {{.id = 0}, {.id = 1}},
                         .links = {{.a = 0, .b = 2, .bytes_per_s = 1.0, .latency_s = 0.0}}}));
  EXPECT_FALSE(Validate({.devices = {{.id = 0}, {.id = 1}},
                         .links = {{.a = 0, .b = 1, .bytes_per_s = 0.0, .latency_s = 0.0}}}));
  EXPECT_FALSE(Validate({.devices = {{.id = 0}, {.id = 1}},
                         .links = {{.a = 0, .b = 1, .bytes_per_s = 1.0, .latency_s = 0.0},
                                   {.a = 1, .b = 0, .bytes_per_s = 1.0, .latency_s = 0.0}}}));
  // A fused node, and a component the assignment does not place.
  const Tiny t = TinyGemma(false);
  const Assignment a = TinyAssignment(t, 2, {.stages = {{0, 1}}, .first_layer = {0}});
  {
    ir::Builder b;
    const auto f = ir::Dense(Encoding::kF32, ir::ShapeOf({kWidth, 2}));
    const ir::TensorId x = b.Input(f);
    const ir::TensorId w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({kWidth})));
    b.SetComponent(1);
    b.Output(
        b.Fused(ir::FusedKind::kRmsNormMul, std::array{x, w}, ir::NormParams{.eps = 1e-6F})[0]);
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g.has_value());
    EXPECT_FALSE(Partition(*g, a));
  }
  {
    ir::Builder b;
    const ir::TensorId x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({kWidth, 2})));
    const ir::TensorId w = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({kWidth})),
                                    {.stage = 0, .component = 999, .role = 0, .layer = 0});
    b.SetComponent(999);
    b.Output(b.Mul(x, w));
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g.has_value());
    EXPECT_FALSE(Partition(*g, a));
  }
}

TEST(PartitionTest, HyperConnectionsAndSideSourcesHavePlacementsAndPayTheirWeights) {
  const Tiny t = TinyHcSide();
  ASSERT_TRUE(components::ValidateSpec(t.spec, t.tensors));
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model) << model.error();
  ASSERT_EQ(model->instances.side.size(), 1U);
  ASSERT_TRUE(model->instances.mixing);
  const auto side = model->instances.side[0].index;
  const auto mixing = model->instances.mixing->index;
  ASSERT_EQ(model->roles[side][0].name, "side");
  const auto layout = components::StageLayout(t.spec, 0, kContext, kMaxRows);
  ASSERT_TRUE(layout) << layout.error();
  const StepShape shape{.segments = {Seg(0, 2, 0, 2)}};
  auto step = components::BuildStep(t.spec, 0, shape, {.tensors = t.tensors, .state = &*layout});
  ASSERT_TRUE(step) << step.error();
  Leaves leaves;
  const std::array<std::int32_t, 2> tokens{1, 2};
  for (const auto& input : step->inputs) {
    if (input.kind == components::InputKind::kSideRows) {
      leaves.ints[input.tensor.v] = {3, 4};
    } else {
      auto values = components::InputValues(input, shape, tokens);
      ASSERT_TRUE(values) << values.error();
      leaves.ints[input.tensor.v] = *values;
    }
  }
  const auto weights = Weights(t);
  for (const auto& weight : step->weights)
    leaves.floats[weight.tensor.v] = weights.at(weight.parts.front().first);
  Check check{.results = {step->logits}, .rows = 2, .outputs = 2, .tolerance = 1e-5};
  for (const auto& state : step->state) {
    leaves.floats[state.tensor.v].resize(ir::Elements(step->graph.tensor(state.tensor).shape()));
    check.state.push_back(state.tensor);
  }
  std::uint64_t bytes = 0;
  for (const auto& tensor : t.tensors) bytes += 4 * ir::Elements(tensor.format.shape);
  // Each layer owns one fixed history row of two four-wide streams.
  bytes += 2 * 2 * 4 * 4;
  for (const Pipeline& pipeline : {Pipeline{.stages = {{0, 1}}, .first_layer = {0}},
                                   Pipeline{.stages = {{0}, {1}}, .first_layer = {0, 1}}}) {
    auto assignment = AssignPipeline(*model, Uniform(2, kAmple), Mix(), pipeline);
    ASSERT_TRUE(assignment) << assignment.error();
    ASSERT_NE(assignment->Find(side), nullptr);
    ASSERT_NE(assignment->Find(mixing), nullptr);
    EXPECT_EQ(assignment->Find(side)->devices, pipeline.stages[0]);
    EXPECT_EQ(assignment->Find(mixing)->devices, pipeline.stages[0]);
    Equivalent(step->graph, leaves, *assignment, *model, check);
    if (pipeline.stages.size() == 1) {
      for (const auto& [device, memory] : assignment->memory) EXPECT_EQ(memory, bytes) << device;
    } else {
      EXPECT_EQ(ModelTraffic(*model, *assignment, 2, 2).bytes, 80U);
    }
  }
  auto exact = Assign(*model, Uniform(1, bytes), Mix());
  ASSERT_TRUE(exact) << exact.error();
  EXPECT_EQ(exact->memory[0].second, bytes);
  // Without the side table this budget fits; it must now be refused.
  EXPECT_FALSE(Assign(*model, Uniform(1, bytes - 1), Mix()));
}

TEST(PartitionTest, SideTableSharedWithReaderCountsOnceAtTheExactBudget) {
  Tiny t = TinyHcSide();
  auto& key = t.spec.stages[0].layers[0].components[0].weights;
  const auto bound = std::ranges::find(key, "key", &spec::WeightBinding::role);
  ASSERT_NE(bound, key.end());
  bound->tensor = "side";
  spec::Canonicalize(t.spec);
  ASSERT_TRUE(components::ValidateSpec(t.spec, t.tensors));
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model) << model.error();
  const Pipeline pipeline{.stages = {{0}}, .first_layer = {0}};
  auto ample = AssignPipeline(*model, Uniform(1, kAmple), Mix(), pipeline);
  ASSERT_TRUE(ample) << ample.error();
  const auto bytes = ample->memory[0].second;
  auto explicit_exact = AssignPipeline(*model, Uniform(1, bytes), Mix(), pipeline);
  ASSERT_TRUE(explicit_exact) << explicit_exact.error();
  auto searched = Assign(*model, Uniform(1, bytes), Mix());
  ASSERT_TRUE(searched) << searched.error();
  EXPECT_EQ(searched->memory[0].second, bytes);
  EXPECT_FALSE(Assign(*model, Uniform(1, bytes - 1), Mix()));
}

TEST(PartitionTest, TiedEmbeddingHeadAndLayerNormsUseTheExactBudget) {
  Tiny t = TinyGemma(true);
  for (auto& layer : t.spec.stages[0].layers) {
    auto& weights = layer.components[0].weights;
    const auto norm = std::ranges::find(weights, "norm", &spec::WeightBinding::role);
    ASSERT_NE(norm, weights.end());
    norm->tensor = "output_norm.weight";
  }
  spec::Canonicalize(t.spec);
  ASSERT_TRUE(components::ValidateSpec(t.spec, t.tensors));
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model) << model.error();
  const Pipeline pipeline{.stages = {{0}}, .first_layer = {0}};
  auto ample = AssignPipeline(*model, Uniform(1, kAmple), Mix(), pipeline);
  ASSERT_TRUE(ample) << ample.error();
  const auto bytes = ample->memory[0].second;
  auto explicit_exact = AssignPipeline(*model, Uniform(1, bytes), Mix(), pipeline);
  ASSERT_TRUE(explicit_exact) << explicit_exact.error();
  auto searched = Assign(*model, Uniform(1, bytes), Mix());
  ASSERT_TRUE(searched) << searched.error();
  EXPECT_EQ(searched->memory[0].second, bytes);
  EXPECT_FALSE(Assign(*model, Uniform(1, bytes - 1), Mix()));
}

TEST(PartitionTest, SparseSideReadersTransferOnlyOnceToEachReadingGroup) {
  const Tiny t = TinyHcSide(true);
  ASSERT_TRUE(components::ValidateSpec(t.spec, t.tensors));
  auto model = ModelOf(t.spec, 0, t.tensors);
  ASSERT_TRUE(model) << model.error();
  auto topology = Uniform(4, kAmple);
  // The last reader relays from its earlier reader rather than the source.
  for (auto& link : topology.links)
    if (link.a == 1 && link.b == 3) link.bytes_per_s *= 10;
  const Pipeline pipeline{.stages = {{0}, {1}, {2}, {3}}, .first_layer = {0, 1, 2, 3}};
  auto assignment = AssignPipeline(*model, topology, Mix(), pipeline);
  ASSERT_TRUE(assignment) << assignment.error();
  const auto layout = components::StageLayout(t.spec, 0, kContext, kMaxRows);
  ASSERT_TRUE(layout) << layout.error();
  const StepShape shape{.segments = {Seg(0, 2, 0, 2)}};
  auto step = components::BuildStep(t.spec, 0, shape, {.tensors = t.tensors, .state = &*layout});
  ASSERT_TRUE(step) << step.error();
  Leaves leaves;
  const std::array<std::int32_t, 2> tokens{1, 2};
  for (const auto& input : step->inputs) {
    if (input.kind == components::InputKind::kSideRows) {
      leaves.ints[input.tensor.v] = {3, 4};
    } else {
      auto values = components::InputValues(input, shape, tokens);
      ASSERT_TRUE(values) << values.error();
      leaves.ints[input.tensor.v] = *values;
    }
  }
  const auto weights = Weights(t);
  for (const auto& weight : step->weights)
    leaves.floats[weight.tensor.v] = weights.at(weight.parts.front().first);
  Check check{.results = {step->logits}, .rows = 2, .outputs = 2, .tolerance = 1e-5};
  for (const auto& state : step->state) {
    leaves.floats[state.tensor.v].resize(ir::Elements(step->graph.tensor(state.tensor).shape()));
    check.state.push_back(state.tensor);
  }
  Equivalent(step->graph, leaves, *assignment, *model, check);
  EXPECT_EQ(ModelTraffic(*model, *assignment, 2, 2).bytes, 240U);
  // Missing reader metadata must refuse rather than silently omit traffic.
  auto incomplete = *model;
  incomplete.side_layers.clear();
  EXPECT_FALSE(Assign(incomplete, topology, Mix()));
}

TEST(PartitionTest, StreamBytesScalesWithStreamsCount) {
  Tiny t1 = TinyGemma(false);
  auto model1 = ModelOf(t1.spec, 0, t1.tensors);
  ASSERT_TRUE(model1.has_value());
  const Pipeline pipeline{.stages = {{0}, {1}}, .first_layer = {0, 3}};
  auto a1 = AssignPipeline(*model1, Uniform(2, kAmple), Mix(), pipeline);
  ASSERT_TRUE(a1.has_value()) << a1.error();
  const Traffic traffic1 = ModelTraffic(*model1, *a1, 4, 4);

  StageModel model2 = *model1;
  ASSERT_TRUE(model2.instances.embedding.has_value());
  model2.instances.embedding->stage.streams = 2;
  auto a2 = AssignPipeline(model2, Uniform(2, kAmple), Mix(), pipeline);
  ASSERT_TRUE(a2.has_value()) << a2.error();
  const Traffic traffic2 = ModelTraffic(model2, *a2, 4, 4);

  EXPECT_GT(traffic1.bytes, 0U);
  EXPECT_EQ(traffic2.bytes, 2 * traffic1.bytes);
}

TEST(PartitionTest, MakeFullDoesNotEmitDuplicateAllGather) {
  const Tiny t = TinyGemma(false);
  const Assignment a = TinyAssignment(t, 2, {.stages = {{0, 1}}, .first_layer = {0}});
  const std::uint16_t c = a.per_layer[0][1].component;
  ASSERT_EQ(a.per_layer[0][1].split, SplitKind::kColumns);
  ir::Builder b;
  b.SetComponent(c);
  const auto f32 = [](std::initializer_list<std::uint64_t> ne) {
    return ir::Dense(Encoding::kF32, ir::ShapeOf(ne));
  };
  const auto weight = [&](std::uint16_t role, std::initializer_list<std::uint64_t> ne) {
    return b.Weight(f32(ne), {.stage = 0, .component = c, .role = role, .layer = 0});
  };
  const ir::TensorId x = b.Input(f32({kWidth, 2}));
  const ir::TensorId p = b.MatMul(weight(1, {kWidth, kHidden}), x);
  const ir::TensorId norm1 = b.RmsNorm(p, 1e-6F);
  const ir::TensorId norm2 = b.RmsNorm(p, 1e-5F);
  b.Output(norm1);
  b.Output(norm2);
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;
  auto parts = Partition(*g, a);
  ASSERT_TRUE(parts.has_value()) << parts.error();
  ASSERT_EQ(parts->devices.size(), 2U);
  for (const DeviceGraph& d : parts->devices) {
    const auto gathers = std::ranges::count_if(
        d.graph.nodes(), [](const ir::Node& n) { return n.key.op == ir::Op::kAllGather; });
    EXPECT_EQ(gathers, 1U);
  }
}

// Synthetic described rates below are test inputs, not measurements of any GPU.
namespace cost {
constexpr std::array kRoles{components::WeightRole{.name = "matrix"}};
constexpr std::array kWeightSplit{components::RoleSplit{.role = 0, .dim = 1}};
constexpr std::array kSplits{components::SplitOption{.kind = SplitKind::kColumns,
                                                     .collective = ir::Op::kAllGather,
                                                     .bytes_per_step =
                                                         [](const components::SplitStep& step) {
                                                           return step.instance->stage.width *
                                                                  step.rows * 4;
                                                         },
                                                     .weights = kWeightSplit}};
const components::ComponentDef& Def() {
  static const components::ComponentDef def{
      .kind = "cost.test",
      .roles = kRoles,
      .shapes = [](const components::Instance& in)
          -> std::expected<std::vector<components::RoleShape>, std::string> {
        return std::vector{moe::Role(0, {in.stage.width, in.stage.width})};
      },
      .splits = kSplits,
      .emit = [](components::Fragment& f) -> std::expected<components::FragmentOut, std::string> {
        return components::FragmentOut{.stream = f.b.MatMul(f.Weight(0), f.stream)};
      }};
  return def;
}
StageModel Model(std::uint32_t layers = 1) {
  StageModel model;
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    const auto index = static_cast<std::uint16_t>(layer);
    model.instances.layers.push_back({components::Instance{.def = &Def(),
                                                           .params = {},
                                                           .stage = {.width = kWidth},
                                                           .site = components::Site::kLayer,
                                                           .layer = layer,
                                                           .index = index}});
    model.roles.push_back({RoleTensor{.name = std::format("matrix.{}", layer),
                                      .format = Format(Encoding::kF32, {kWidth, kWidth})}});
  }
  return model;
}
Topology Described(std::size_t devices = 2, double bandwidth = 1e12, double rate = 1e6,
                   double link_bandwidth = 1e12) {
  auto topology = Uniform(devices, kAmple, link_bandwidth, 0);
  topology.cost_model = CostModel::kExecution;
  for (auto& device : topology.devices) {
    device.performance =
        DevicePerformance{.memory_bytes_per_s = bandwidth,
                          .rates = {{.op = ir::Primitive(ir::Op::kMatMul), .units_per_s = rate}},
                          .provenance = "synthetic planner test input; not a hardware measurement"};
  }
  return topology;
}
Workload Work() {
  return {.prefill_rows = 8, .decode_rows = 1, .decode_share = 0.5, .context = 64, .slots = 1};
}
}  // namespace cost

TEST(PartitionTest, ExecutionCostsUseActualSplitShapesAndReportedObjective) {
  const auto model = cost::Model();
  const auto topology = cost::Described();
  const auto workload = cost::Work();
  const Pipeline whole{.stages = {{0}}, .first_layer = {0}};
  const Pipeline split{.stages = {{0, 1}}, .first_layer = {0}};
  auto one = AssignPipeline(model, topology, workload, whole);
  auto two = AssignPipeline(model, topology, workload, split);
  ASSERT_TRUE(one) << one.error();
  ASSERT_TRUE(two) << two.error();
  ASSERT_TRUE(one->prefill_execution_seconds);
  ASSERT_TRUE(two->prefill_execution_seconds);
  // 2*K*N*M FLOPs; local N halves, while every rank reads all input rows.
  EXPECT_DOUBLE_EQ(*one->prefill_execution_seconds, 2.0 * 32 * 32 * 8 / 1e6);
  EXPECT_DOUBLE_EQ(*two->prefill_execution_seconds, 2.0 * 32 * 16 * 8 / 1e6);
  EXPECT_DOUBLE_EQ(*two->decode_execution_seconds, 2.0 * 32 * 16 / 1e6);
  EXPECT_NEAR(two->modelled_seconds_per_step,
              0.5 * (*two->prefill_execution_seconds + two->prefill.seconds) +
                  0.5 * (*two->decode_execution_seconds + two->decode.seconds),
              1e-15);
  auto chosen = Assign(model, topology, workload);
  ASSERT_TRUE(chosen) << chosen.error();
  EXPECT_EQ(chosen->stages, split.stages);
  EXPECT_DOUBLE_EQ(chosen->modelled_seconds_per_step, two->modelled_seconds_per_step);
}

TEST(PartitionTest, ExecutionBandwidthIncludesReusedWeightsAndPerStepReadsWrites) {
  const auto model = cost::Model();
  const auto topology = cost::Described(2, 1e3, 1e12);
  auto one = AssignPipeline(model, topology, cost::Work(), {.stages = {{0}}, .first_layer = {0}});
  auto two =
      AssignPipeline(model, topology, cost::Work(), {.stages = {{0, 1}}, .first_layer = {0}});
  ASSERT_TRUE(one) << one.error();
  ASSERT_TRUE(two) << two.error();
  EXPECT_DOUBLE_EQ(*one->prefill_execution_seconds, (4096.0 + 256 * 8) / 1e3);
  EXPECT_DOUBLE_EQ(*one->decode_execution_seconds, (4096.0 + 256) / 1e3);
  EXPECT_DOUBLE_EQ(*two->prefill_execution_seconds, (2048.0 + 192 * 8) / 1e3);
}

TEST(PartitionTest, ExecutionWholeAndSplitCompeteWithJointCapacityFeasibility) {
  const auto model = cost::Model();
  auto topology = cost::Described(2, 1e12, 1e6, 1);
  const Pipeline both{.stages = {{0, 1}}, .first_layer = {0}};
  auto whole = AssignPipeline(model, topology, cost::Work(), both);
  ASSERT_TRUE(whole) << whole.error();
  EXPECT_EQ(whole->per_layer[0][0].option, kWhole);
  // Communication makes whole cheaper, but only split fits this exact budget.
  for (auto& device : topology.devices) device.memory = 2048;
  auto split = AssignPipeline(model, topology, cost::Work(), both);
  ASSERT_TRUE(split) << split.error();
  EXPECT_EQ(split->per_layer[0][0].option, 0);
  EXPECT_EQ(split->memory[0].second, 2048);
  EXPECT_EQ(split->memory[1].second, 2048);
  auto searched = Assign(model, topology, cost::Work());
  ASSERT_TRUE(searched) << searched.error();
  EXPECT_DOUBLE_EQ(searched->modelled_seconds_per_step, split->modelled_seconds_per_step);
}

TEST(PartitionTest, ExecutionFactsPreventFalseTopologyTwinsAndWeightBothPhases) {
  const auto model = cost::Model();
  auto topology = cost::Described(2, 1e3, 1e9, 1e-9);
  topology.devices[1].performance->memory_bytes_per_s = 1e9;
  topology.devices[1].performance->rates[0].units_per_s = 1e3;
  auto workload = cost::Work();
  workload.decode_share = 1;
  auto decode = Assign(model, topology, workload);
  ASSERT_TRUE(decode) << decode.error();
  EXPECT_EQ(decode->stages, (std::vector<std::vector<std::uint32_t>>{{1}}));
  workload.decode_share = 0;
  auto prefill = Assign(model, topology, workload);
  ASSERT_TRUE(prefill) << prefill.error();
  EXPECT_EQ(prefill->stages, (std::vector<std::vector<std::uint32_t>>{{0}}));
  workload.decode_share = 0.5;
  auto mix = Assign(model, topology, workload);
  ASSERT_TRUE(mix) << mix.error();
  EXPECT_EQ(mix->stages, prefill->stages);
  EXPECT_NEAR(mix->modelled_seconds_per_step,
              0.5 * ((4096.0 + 256 * 8) / 1e3 + (4096.0 + 256) / 1e3), 1e-12);
}

TEST(PartitionTest, UnknownExecutionFactsAreExplicitAndTypedRatesAreRequired) {
  const auto model = cost::Model();
  auto topology = Uniform(1, kAmple);
  auto traffic = Assign(model, topology, cost::Work());
  ASSERT_TRUE(traffic) << traffic.error();
  EXPECT_FALSE(traffic->prefill_execution_seconds);
  EXPECT_FALSE(traffic->decode_execution_seconds);
  topology.cost_model = CostModel::kExecution;
  EXPECT_FALSE(Validate(topology));
  EXPECT_FALSE(Assign(model, topology, cost::Work()));
  topology = cost::Described(1);
  topology.devices[0].performance->rates.clear();
  auto missing = Assign(model, topology, cost::Work());
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().find("lacks described rate"), std::string::npos);
  topology = cost::Described(1);
  topology.devices[0].performance->rates[0].accumulate = ir::Accumulate::kF32;
  EXPECT_FALSE(Assign(model, topology, cost::Work()));
  topology = cost::Described(1);
  topology.devices[0].performance->rates[0].encoding = Encoding::kBF16;
  EXPECT_FALSE(Assign(model, topology, cost::Work()));
  topology = cost::Described(1);
  topology.devices[0].performance->rates[0].params[0] = std::byte{1};
  EXPECT_FALSE(Assign(model, topology, cost::Work()));
}

TEST(PartitionTest, DescribedPerformanceRejectsInvalidMetricsAndProvenance) {
  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    auto topology = cost::Described(1);
    topology.devices[0].performance->memory_bytes_per_s = invalid;
    EXPECT_FALSE(Validate(topology));
    topology = cost::Described(1);
    topology.devices[0].performance->rates[0].units_per_s = invalid;
    EXPECT_FALSE(Validate(topology));
  }
  auto topology = cost::Described(1);
  topology.devices[0].performance->provenance.clear();
  EXPECT_FALSE(Validate(topology));
  topology = cost::Described(1);
  topology.devices[0].performance->rates.push_back(topology.devices[0].performance->rates.front());
  EXPECT_FALSE(Validate(topology));
}

TEST(PartitionTest, ExecutionSearchMatchesConfiguredPipelineEnumeration) {
  const auto model = cost::Model(3);
  auto topology = cost::Described(3, 1e5, 1e7, 2e4);
  topology.devices[1].performance->rates[0].units_per_s *= 2;
  topology.devices[2].performance->memory_bytes_per_s *= 3;
  for (auto& device : topology.devices) device.memory = 6144;
  ExpectSearchExact(model, topology, cost::Work(), 3);
}

TEST(PartitionTest, ExecutionReadsStateAndCapacityRetainsEverySlot) {
  const components::ComponentDef definition{
      .kind = "cost.state.test",
      .shapes = [](const components::Instance&)
          -> std::expected<std::vector<components::RoleShape>, std::string> {
        return std::vector<components::RoleShape>{};
      },
      .state =
          [](const components::Instance&) {
            return model::StateContract{.tensors = {{.name = "fixed",
                                                     .kind = model::StateKind::kFeatures,
                                                     .encoding = Encoding::kF32,
                                                     .cell_bytes = 128,
                                                     .cells = model::CellRule::kFixed,
                                                     .fixed_cells = 1,
                                                     .cell_pad = 1,
                                                     .rollback = model::Rollback::kSnapshot}}};
          },
      .max_state_tensors = 1,
      .emit = [](components::Fragment& fragment)
          -> std::expected<components::FragmentOut, std::string> {
        return components::FragmentOut{.stream =
                                           fragment.b.Add(fragment.stream, fragment.State(0, 0))};
      }};
  StageModel model;
  model.instances.layers.push_back({components::Instance{.def = &definition,
                                                         .params = {},
                                                         .stage = {.width = kWidth},
                                                         .site = components::Site::kLayer}});
  model.roles.resize(1);
  auto topology = cost::Described(1, 1e3, 1e30);
  topology.devices[0].performance->rates[0].op = ir::Primitive(ir::Op::kAdd);
  auto workload = cost::Work();
  workload.slots = 3;
  auto assignment = Assign(model, topology, workload);
  ASSERT_TRUE(assignment) << assignment.error();
  EXPECT_EQ(assignment->memory[0].second, 3 * 128U);
  EXPECT_DOUBLE_EQ(*assignment->prefill_execution_seconds, (128.0 + 256 * 8) / 1e3);
  EXPECT_DOUBLE_EQ(*assignment->decode_execution_seconds, (128.0 + 256) / 1e3);
}

// Describe the actual synthetic graph's typed operations, including the EP
// ownership/select auxiliaries. These are test rates, not measured capabilities.
void AddSyntheticRates(Topology& topology, const ir::Graph& graph) {
  for (const auto& node : graph.nodes()) {
    if (node.nout == 0) continue;
    OperationRate rate{.op = node.key,
                       .params = node.params,
                       .encoding = graph.tensor(node.out[0]).format.encoding,
                       .units_per_s = 1e30};
    if (node.key.op == ir::Op::kMatMul || node.key.op == ir::Op::kMatMulId) {
      rate.encoding = graph.tensor(node.src[0]).format.encoding;
      rate.accumulate = node.Params<ir::ProductParams>().accumulate;
      if (node.key.op == ir::Op::kMatMulId) rate.units_per_s = 1e3;
    } else if (node.key.op == ir::Op::kAttention) {
      rate.encoding = graph.tensor(node.src[1]).format.encoding;
      rate.accumulate = node.Params<ir::AttnParams>().accumulate;
    }
    for (auto& device : topology.devices) {
      auto& rates = device.performance->rates;
      if (std::ranges::none_of(rates, [&](const auto& existing) {
            return existing.op == rate.op && existing.params == rate.params &&
                   existing.encoding == rate.encoding && existing.accumulate == rate.accumulate;
          }))
        rates.push_back(rate);
    }
  }
}

TEST(PartitionTest, ExpertParallelCostKeepsEveryRoutedSlotOnEveryRank) {
  auto definition = moe::Def();
  definition.splits = definition.splits.first(1);  // whole-expert placement only
  auto model = moe::Model(true);
  model.instances.layers[0][0].def = &definition;
  auto topology = cost::Described(2, 1e30, 1e30);
  for (auto& device : topology.devices) device.performance->rates.clear();
  auto workload = cost::Work();
  workload.decode_share = 1;
  const StepShape shape{.phase = Phase::kDecode,
                        .segments = {{.rows = 1, .past = 63, .read_cells = 256, .outputs = 1}}};
  auto fragment =
      components::BuildComponent(model.instances.layers[0][0], shape, moe::Formats(), {});
  ASSERT_TRUE(fragment) << fragment.error();
  for (std::uint32_t ways : {1U, 2U}) {
    Assignment assignment{.topology = Uniform(ways, kAmple)};
    std::vector<std::uint32_t> group;
    for (std::uint32_t rank = 0; rank < ways; ++rank) group.push_back(rank);
    assignment.per_layer.push_back(
        {ComponentPlacement{.component = 1,
                            .def = &definition,
                            .split = ways == 1 ? SplitKind::kNone : SplitKind::kExperts,
                            .option = ways == 1 ? kWhole : std::uint8_t{0},
                            .devices = group}});
    auto partition = Partition(fragment->step.graph, assignment);
    ASSERT_TRUE(partition) << partition.error();
    for (const auto& device : partition->devices) AddSyntheticRates(topology, device.graph);
  }
  auto one = AssignPipeline(model, topology, workload, {.stages = {{0}}, .first_layer = {0}});
  ASSERT_TRUE(one) << one.error();
  // Determine exact EP capacity independently using the traffic-only inventory.
  auto capacity =
      AssignPipeline(model, Uniform(2, kAmple), workload, {.stages = {{0, 1}}, .first_layer = {0}});
  ASSERT_TRUE(capacity) << capacity.error();
  for (auto& device : topology.devices) device.memory = capacity->memory[0].second;
  auto experts =
      AssignPipeline(model, topology, workload, {.stages = {{0, 1}}, .first_layer = {0}});
  ASSERT_TRUE(experts) << experts.error();
  ASSERT_EQ(experts->per_layer[0][0].split, SplitKind::kExperts);
  const double routed = 3.0 * 2 * kWidth * moe::kExpertHidden * 3 / 1e3;
  EXPECT_NEAR(*one->decode_execution_seconds, routed, 1e-12);
  EXPECT_NEAR(*experts->decode_execution_seconds, routed, 1e-12);
}

TEST(PartitionTest, ExecutionSourceWorkIncludesFrameworkMasksAndActualNarrowing) {
  const Tiny tiny = TinyGemma(true);
  auto model = ModelOf(tiny.spec, 0, tiny.tensors);
  ASSERT_TRUE(model) << model.error();
  auto workload = cost::Work();
  auto topology = cost::Described(1, 1e3, 1e30);
  topology.devices[0].performance->rates.clear();
  auto layout = components::StageLayout(tiny.spec, 0, workload.context, workload.prefill_rows);
  ASSERT_TRUE(layout) << layout.error();
  for (const auto rows : {workload.prefill_rows, workload.decode_rows}) {
    const bool prefill = rows == workload.prefill_rows;
    const StepShape shape{.phase = prefill ? Phase::kPrefill : Phase::kDecode,
                          .segments = {{.rows = rows,
                                        .past = workload.context - rows,
                                        .read_cells = 256,
                                        .outputs = prefill ? 1U : rows}}};
    auto step =
        components::BuildStep(tiny.spec, 0, shape, {.tensors = tiny.tensors, .state = &*layout});
    ASSERT_TRUE(step) << step.error();
    AddSyntheticRates(topology, step->graph);
    for (const auto& node : step->graph.nodes()) {
      if (node.component == model->instances.output->index && node.key.op == ir::Op::kMatMul)
        EXPECT_EQ(step->graph.tensor(node.out[0]).ne(1), 1U);
    }
  }
  auto assignment = Assign(*model, topology, workload);
  ASSERT_TRUE(assignment) << assignment.error();
  ASSERT_TRUE(assignment->prefill_execution_seconds);
  EXPECT_GT(*assignment->prefill_execution_seconds, *assignment->decode_execution_seconds);
  // Mask construction must be priced, even though its nodes are framework
  // dependencies rather than members of the attention's component ordinal.
  std::erase_if(topology.devices[0].performance->rates,
                [](const auto& rate) { return rate.op.op == ir::Op::kMaskBuild; });
  auto missing = Assign(*model, topology, workload);
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().find("mask_build"), std::string::npos);
}

TEST(PartitionTest, ExecutionCostsPreserveTheActualQuantizedCacheLayout) {
  std::array<double, 2> decode{};
  std::size_t encoding_index = 0;
  for (const auto encoding : {Encoding::kF16, Encoding::kQ8_0}) {
    SCOPED_TRACE(static_cast<unsigned>(encoding));
    // D64 is a supported quantized attention domain. The producer submits F32
    // rows, while the cache stores either dense F16 or GGML Q8_0 blocks.
    auto tiny = TinyGemma(true, Encoding::kF32, kHidden, 64, encoding);
    // Match all arithmetic across the representation comparison.
    for (auto& layer : tiny.spec.stages[0].layers)
      for (auto& param : layer.components[0].params)
        if (param.name == "softcap") param.value = R(0.0);
    auto model = ModelOf(tiny.spec, 0, tiny.tensors);
    ASSERT_TRUE(model) << model.error();
    const auto workload = cost::Work();
    auto topology = cost::Described(1, 1e6, 1e30);
    topology.devices[0].performance->rates.clear();
    auto layout = components::StageLayout(tiny.spec, 0, workload.context, workload.prefill_rows);
    ASSERT_TRUE(layout) << layout.error();
    std::size_t stores = 0;
    for (const auto rows : {workload.prefill_rows, workload.decode_rows}) {
      const bool prefill = rows == workload.prefill_rows;
      const StepShape shape{.phase = prefill ? Phase::kPrefill : Phase::kDecode,
                            .segments = {{.rows = rows,
                                          .past = workload.context - rows,
                                          .read_cells = 256,
                                          .outputs = prefill ? 1U : rows}}};
      auto step =
          components::BuildStep(tiny.spec, 0, shape, {.tensors = tiny.tensors, .state = &*layout});
      ASSERT_TRUE(step) << step.error();
      AddSyntheticRates(topology, step->graph);
      for (const auto& node : step->graph.nodes()) {
        if (node.key != ir::Primitive(ir::Op::kCacheStore)) continue;
        const auto& destination = step->graph.tensor(node.src[0]).format;
        const auto& submitted = step->graph.tensor(node.src[1]).format;
        EXPECT_EQ(destination.encoding, encoding);
        EXPECT_EQ(submitted.encoding, Encoding::kF32);
        auto written = destination;
        written.shape = submitted.shape;
        auto parts = format::PartsOf(written);
        ASSERT_TRUE(parts);
        if (encoding == Encoding::kQ8_0) {
          EXPECT_EQ(written.layout, format::Layout::kGgmlBlocks);
          EXPECT_EQ(parts->parts()[0].bytes, ir::Elements(submitted.shape) / 32 * 34);
        }
        ++stores;
      }
    }
    EXPECT_GT(stores, 0U);
    auto assignment = Assign(*model, topology, workload);
    ASSERT_TRUE(assignment) << assignment.error();
    ASSERT_TRUE(assignment->prefill_execution_seconds);
    ASSERT_TRUE(assignment->decode_execution_seconds);
    decode.at(encoding_index++) = *assignment->decode_execution_seconds;
  }
  EXPECT_GT(decode[0], decode[1]);  // Both cache reads and stores use their real representations.
}

TEST(PartitionTest, CapacityInventoryCannotWrapOrAliasConflictingFormats) {
  auto model = cost::Model(5);
  constexpr std::uint64_t width = std::uint64_t{1} << 30;
  for (auto& layer : model.instances.layers) layer[0].stage.width = width;
  for (auto& roles : model.roles) roles[0].format = Format(Encoding::kF32, {width, width});
  // Each tensor fits the byte type and each is distinct; their sum does not.
  EXPECT_FALSE(Assign(model, Uniform(1, std::numeric_limits<std::uint64_t>::max()), cost::Work()));
  model.instances.layers.resize(3);
  model.roles.resize(3);
  auto feasible =
      Assign(model, Uniform(1, std::numeric_limits<std::uint64_t>::max()), cost::Work());
  ASSERT_TRUE(feasible) << feasible.error();
  EXPECT_EQ(feasible->memory[0].second, 3 * (std::uint64_t{1} << 62));
  model = cost::Model(2);
  model.roles[1][0].name = model.roles[0][0].name;
  model.roles[1][0].format.encoding = Encoding::kF16;
  auto conflict = Assign(model, Uniform(1, kAmple), cost::Work());
  ASSERT_FALSE(conflict);
  EXPECT_NE(conflict.error().find("conflicting formats"), std::string::npos);
}

}  // namespace
}  // namespace llmp::planner
