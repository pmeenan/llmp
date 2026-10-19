// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Qwen3.8's n-gram embedding (PLE; llama.cpp's qwen4exp build_ple and
// llm_graph_input_ple, kernels/ggml/qwen38_graph.cc's Ple), and its MTP
// drafter's input.
//
// side.ngram, version 1: a side input (spec.h SideInput): for each step row
// `heads` rows of a table, named by a hash of the row's token and the
// ngram - 1 before it (llama.cpp's set_input: an EOS, or the sequence's
// start, cuts every token at or before it, which then reads as EOS; head
// g of the n-gram size n reads row (mixed mod vocab[g]) + offset[g], where
// mixed is the XOR of each context token times its multiplier), joined
// into one row of heads · row values. The table is read by rows on demand
// (WeightRole::rows_on_demand, engine/ple_rows.h); the rows are a step
// input the engine computes (InputKind::kSideRows, ComponentDef::
// side_rows). The hash's constants are its parameters (a GGUF
// checkpoint's metadata; the ModelOpt artifact keeps them as tensors too).
//
// residual.ple, version 1: the n-gram layer's block, which reads the side
// input and adds to every stream (before the layer's first mix): a key
// (its own product of the side input, each stream RMS-normed) against the
// streams (each RMS-normed), a gate sigmoid(sgn(s) · sqrt(max(|s|, 1e-6)))
// of each stream's s = key · query / sqrt(width), the value product times
// the gate, normed per stream, then a depthwise causal convolution over the
// last (conv - 1) · ngram rows (dilated by the n-gram size) and SiLU; the
// gated value and the convolution added to the streams. Its state is the
// convolution's history, fixed, which a verify never writes (it saves the
// convolution's input rows for the commit instead).
//
// embedding.mtp, version 1: an MTP drafter's input (Qwen3.8's
// Qwen3_8FlashNextMultiTokenPredictor, qwen38_graph.cc's MtpPass): each
// row's next token's row of its target's token table, RMS-normed and
// projected (fc_embd), added to every stream of the feature it reads (its
// target's streams before the head's mix, or its own last pass's: a step
// input the engine binds, InputKind::kFeatures), the feature RMS-normed
// over all its streams together and each stream projected (fc_hidden). Its
// state is the tap its target's steps write: every row's streams, a cell a
// chunk row (CellRule::kPerRow, the rows before the chunk first), from
// which the engine binds a pass's feature.
//
// The fragments emit what the unfused form records, in GGML's graph order.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/step.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"

namespace llmp::components::defs {
namespace {

// ---------------------------------------------------------------- side.ngram

constexpr double kInt64 = 9.2e18;
constexpr std::array kNgramParams{
    // The n-gram size (each row's token and the n - 1 before it).
    ParamSpec{.name = "ngram", .type = ParamType::kInt, .least = 2, .most = 16},
    // Heads a size from 2 to ngram (rows a step row reads: (ngram - 1) · this).
    ParamSpec{.name = "heads_per_ngram", .type = ParamType::kInt, .least = 1, .most = 64},
    // Values a table row holds.
    ParamSpec{.name = "row", .type = ParamType::kInt, .least = 1, .most = 1 << 16},
    // The token that cuts the context (EOS).
    ParamSpec{.name = "eos", .type = ParamType::kInt, .least = 0, .most = kInt64},
    // The hash: a multiplier a context place, an offset and a vocabulary a head.
    ParamSpec{.name = "multipliers", .type = ParamType::kInts, .least = -kInt64, .most = kInt64},
    ParamSpec{.name = "offsets", .type = ParamType::kInts, .least = 0, .most = kInt64},
    ParamSpec{.name = "vocab", .type = ParamType::kInts, .least = 1, .most = kInt64},
};
constexpr std::array kNgramRoles{WeightRole{.name = "table", .rows_on_demand = true}};

std::uint64_t NgramHeads(const Instance& in) {
  return static_cast<std::uint64_t>(in.params.Int("ngram") - 1) *
         static_cast<std::uint64_t>(in.params.Int("heads_per_ngram"));
}

std::expected<std::vector<RoleShape>, std::string> NgramShapes(const Instance& in) {
  const std::uint64_t heads = NgramHeads(in);
  if (in.params.Ints("multipliers").size() != static_cast<std::size_t>(in.params.Int("ngram")) ||
      in.params.Ints("offsets").size() != heads || in.params.Ints("vocab").size() != heads) {
    return std::unexpected("a multiplier a context place, an offset and a vocabulary a head");
  }
  // A row's heads are each row's width (side input heads · row).
  return std::vector<RoleShape>{Role(0, {static_cast<std::uint64_t>(in.params.Int("row")), 0})};
}

std::expected<FragmentOut, std::string> NgramEmit(Fragment& f) {
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  const std::uint64_t heads = NgramHeads(in);
  const ir::TensorId table = f.Weight(0);
  // Every head's rows inside the table (I32 rows, as the lookup takes).
  const std::uint64_t rows = Desc(b, table).ne(1);
  const auto offsets = in.params.Ints("offsets");
  const auto vocab = in.params.Ints("vocab");
  for (std::size_t g = 0; g < offsets.size(); ++g) {
    const auto end = static_cast<std::uint64_t>(offsets[g]) + static_cast<std::uint64_t>(vocab[g]);
    if (end > rows || end > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
      return std::unexpected("an n-gram head's rows past the table");
    }
  }
  const ir::TensorId picked = b.GetRows(table, f.step.SideRows(in.index, heads));
  return FragmentOut{
      .stream = b.Reshape(picked, Shape({heads * static_cast<std::uint64_t>(in.params.Int("row")),
                                         f.step.rows()}))};
}

std::expected<void, std::string> NgramRows(const Instance& in,
                                           std::span<const std::int32_t> history,
                                           std::uint32_t position, std::span<std::int32_t> out) {
  const auto n_gram = static_cast<std::uint32_t>(in.params.Int("ngram"));
  const auto per = static_cast<std::uint32_t>(in.params.Int("heads_per_ngram"));
  const auto eos = static_cast<std::uint64_t>(in.params.Int("eos"));
  const auto mult = in.params.Ints("multipliers");
  const auto offsets = in.params.Ints("offsets");
  const auto vocab = in.params.Ints("vocab");
  if (position >= history.size() || out.size() != NgramHeads(in)) {
    return std::unexpected("n-gram rows past the history, or of another count");
  }
  std::vector<std::uint64_t> ctx(n_gram);
  ctx[0] = static_cast<std::uint64_t>(history[position]);
  bool cut = false;
  for (std::uint32_t s = 1; s < n_gram; ++s) {
    // The predecessor s positions back; missing (before the sequence) or
    // an EOS cuts everything at or before it, which then reads as EOS.
    const bool missing = s > position;
    const std::int64_t t = missing ? -1 : history[position - s];
    cut = cut || t < 0 || static_cast<std::uint64_t>(t) == eos;
    ctx[s] = cut ? eos : static_cast<std::uint64_t>(t);
  }
  for (std::uint32_t n = 2; n <= n_gram; ++n) {
    std::uint64_t mixed = ctx[0] * static_cast<std::uint64_t>(mult[0]);
    for (std::uint32_t j = 1; j < n; ++j) mixed ^= ctx[j] * static_cast<std::uint64_t>(mult[j]);
    for (std::uint32_t g = 0; g < per; ++g) {
      const std::uint32_t head = ((n - 2) * per) + g;
      out[head] = static_cast<std::int32_t>((mixed % static_cast<std::uint64_t>(vocab[head])) +
                                            static_cast<std::uint64_t>(offsets[head]));
    }
  }
  return {};
}

std::uint64_t NgramCount(const Instance& in) { return NgramHeads(in); }

std::expected<void, std::string> NgramReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t heads = NgramHeads(in);
  const auto row = static_cast<std::uint64_t>(in.params.Int("row"));
  const std::span<const float> table = io.weights.at(0);
  const std::uint64_t rows = io.rows();
  if (io.side_rows.size() != heads * rows) return std::unexpected("ngram reference: its rows");
  io.stream.assign(heads * row * rows, 0.0);
  for (std::size_t i = 0; i < io.side_rows.size(); ++i) {
    const auto r = static_cast<std::uint64_t>(io.side_rows[i]);
    if (io.side_rows[i] < 0 || (r + 1) * row > table.size()) {
      return std::unexpected("ngram reference: a row past the table");
    }
    for (std::uint64_t j = 0; j < row; ++j)
      io.stream[(i * row) + j] = static_cast<double>(table[(r * row) + j]);
  }
  return {};
}

// --------------------------------------------------------------- residual.ple

constexpr std::array kPleParams{
    kNormZeroCentered,
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    // The side input it reads (spec.h SideInput; its width is the key and
    // value products' input's).
    ParamSpec{.name = "input", .type = ParamType::kName},
    // The convolution's kernel and its dilation (the n-gram size).
    ParamSpec{.name = "conv", .type = ParamType::kInt, .least = 2, .most = 64},
    ParamSpec{.name = "dilation", .type = ParamType::kInt, .least = 1, .most = 64},
};
enum : std::uint16_t { kKey, kValue, kNormKey, kNormQuery, kNormConv, kConvW };
constexpr std::array kPleRoles{
    WeightRole{.name = "key"},
    WeightRole{.name = "value"},
    WeightRole{.name = "norm_key", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "norm_query", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "norm_conv", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "conv", .encodings = kNormEncodings},
};

struct PleDims {
  std::uint64_t width, hc, conv, dil;
  float eps;
  std::uint64_t hw() const { return width * hc; }
  std::uint64_t history() const { return (conv - 1) * dil; }
};
PleDims PleDimsOf(const Instance& in) {
  return {.width = in.stage.width,
          .hc = in.stage.streams,
          .conv = static_cast<std::uint64_t>(in.params.Int("conv")),
          .dil = static_cast<std::uint64_t>(in.params.Int("dilation")),
          .eps = F32(in.params.Real("eps"))};
}

std::string_view PleInput(const Instance& in) { return in.params.Name("input"); }

std::expected<std::vector<RoleShape>, std::string> PleShapes(const Instance& in) {
  if (NormOffsetContradictsFold(in))
    return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
  const PleDims d = PleDimsOf(in);
  if (d.hc < 2) return std::unexpected("an n-gram block in a stage of one stream");
  return std::vector<RoleShape>{Role(kKey, {0, d.hw()}),   Role(kValue, {0, d.width}),
                                Role(kNormKey, {d.hw()}),  Role(kNormQuery, {d.hw()}),
                                Role(kNormConv, {d.hw()}), Role(kConvW, {d.conv, d.hw()})};
}

model::StateContract PleState(const Instance& in) {
  const PleDims d = PleDimsOf(in);
  return {.tensors = {{.name = "ple_conv",
                       .kind = model::StateKind::kConvolution,
                       .encoding = format::Encoding::kF32,
                       .cell_bytes = d.history() * d.hw() * 4,
                       .cells = model::CellRule::kFixed,
                       .window_or_ratio = 1,
                       .rollback = model::Rollback::kCommit}}};
}

std::expected<FragmentOut, std::string> PleEmit(Fragment& f) {
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  const PleDims d = PleDimsOf(in);
  F32Parameters<kPleRoles.size()> parameters(f);
  const ir::TensorId emb = f.Side(PleInput(in));
  if (!emb.valid() || !f.residual.valid())
    return std::unexpected("an n-gram block without its input");
  const std::uint64_t rows = f.step.rows();
  const std::uint64_t hw = d.hw();
  const std::uint64_t hist = d.history();
  const auto segments = f.step.segments();
  const bool one = segments.size() == 1;
  const bool verify = f.step.phase() == planner::Phase::kVerify;
  // Each stream RMS-normed, then the streams as one row times the weight.
  const auto grouped = [&](ir::TensorId x, std::uint16_t role) {
    ir::TensorId t = b.RmsNorm(b.Reshape(x, Shape({d.width, d.hc, rows})), d.eps);
    t = b.Reshape(t, Shape({hw, rows}));
    t = b.Mul(t, parameters.Norm(role));
    return b.Reshape(t, Shape({d.width, d.hc, rows}));
  };
  // The convolution's histories first (their stores read them first).
  std::vector<ir::TensorId> history(segments.size());
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b.SetOwner(static_cast<std::uint8_t>(s));
    history[s] = b.Reshape(f.State(s, 0), Shape({hist, hw, 1}));
    b.SetJoined();
  }
  const ir::TensorId value = b.MatMul(f.Weight(kValue), emb);
  const ir::TensorId v3 =
      b.Repeat(b.Reshape(value, Shape({d.width, 1, rows})), Shape({d.width, d.hc, rows}));
  const ir::TensorId key = grouped(b.MatMul(f.Weight(kKey), emb), kNormKey);
  const ir::TensorId query = grouped(f.residual, kNormQuery);
  const ir::TensorId s =
      b.Scale(b.SumRows(b.Mul(key, query)), 1.0F / std::sqrt(static_cast<float>(d.width)));
  const ir::TensorId sign = b.Unary(s, ir::UnaryFn::kSgn);
  const ir::TensorId magnitude =
      b.Unary(b.Clamp(b.Unary(s, ir::UnaryFn::kAbs), 1e-6F, std::numeric_limits<float>::infinity()),
              ir::UnaryFn::kSqrt);
  const ir::TensorId gate = b.Unary(b.Mul(sign, magnitude), ir::UnaryFn::kSigmoid);
  const ir::TensorId gated = b.Mul(v3, gate);
  ir::TensorId normalized = grouped(b.Reshape(gated, Shape({hw, rows})), kNormConv);
  normalized = b.Reshape(normalized, Shape({hw, rows}));
  const ir::TensorId x = b.Reshape(normalized, Shape({hw, rows, 1}));
  // Each owner's convolution over its history and rows.
  std::vector<ir::TensorId> outs;
  for (std::size_t si = 0; si < segments.size(); ++si) {
    b.SetOwner(static_cast<std::uint8_t>(si));
    const std::uint64_t n = segments[si].rows;
    ir::TensorId mine = x;
    if (!one) {
      const ir::TensorDesc xt = Desc(b, x);
      mine = b.View(x, Shape({hw, n, 1}), {xt.nb[0], xt.nb[1], xt.nb[1] * n, xt.nb[1] * n},
                    std::uint64_t{segments[si].first_row} * xt.nb[1]);
    }
    if (verify) {
      f.step.Save(in, "ple", b.Reshape(mine, Shape({hw, n})), static_cast<std::uint8_t>(si));
    }
    const ir::TensorId padded = b.Concat(history[si], b.Cont(b.Transpose(mine)), 0);
    const ir::TensorDesc pt = Desc(b, padded);
    if (!verify) {
      const ir::TensorId tail =
          b.View(padded, Shape({hist, hw, 1}), {pt.nb[0], pt.nb[1], pt.nb[2], pt.nb[2]},
                 (pt.ne(0) - hist) * pt.nb[0]);
      const ir::TensorId kept = b.Reshape(b.Cont(tail), Shape({hist * hw, 1}));
      b.CacheStore(f.State(si, 0), kept, f.step.StateRow());
    }
    ir::TensorId sum{};
    const ir::TensorDesc cw = Desc(b, parameters.Get(kConvW));
    for (std::uint64_t k = 0; k < d.conv; ++k) {
      const std::uint64_t start = hist - ((d.conv - 1 - k) * d.dil);
      const ir::TensorId shifted = b.Cont(b.Transpose(b.View(
          padded, Shape({n, hw, 1}), {pt.nb[0], pt.nb[1], pt.nb[2], pt.nb[2]}, start * pt.nb[0])));
      const ir::TensorId wk = b.Reshape(
          b.Cont(b.View(parameters.Get(kConvW), Shape({1, hw}),
                        {cw.nb[0], cw.nb[1], cw.nb[1] * hw, cw.nb[1] * hw}, k * cw.nb[0])),
          Shape({hw}));
      const ir::TensorId term = b.Mul(shifted, wk);
      sum = sum.valid() ? b.Add(sum, term) : term;
    }
    outs.push_back(sum);
    b.SetJoined();
  }
  ir::TensorId conv = outs.front();
  for (std::size_t si = 1; si < outs.size(); ++si) conv = b.Concat(conv, outs[si], 1);
  conv = b.Reshape(b.Cont(b.Unary(conv, ir::UnaryFn::kSilu)), Shape({d.width, d.hc, rows}));
  return FragmentOut{.stream = f.stream, .residual = b.Add(f.residual, b.Add(gated, conv))};
}

// The commit of a verify's kept rows: the history, then their saved
// convolution inputs.
std::expected<void, std::string> PleCommit(Fragment& f) {
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  const PleDims d = PleDimsOf(in);
  const std::uint64_t hw = d.hw();
  const std::uint64_t hist = d.history();
  const auto segments = f.step.segments();
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b.SetOwner(static_cast<std::uint8_t>(s));
    const ir::TensorId history = b.Reshape(f.State(s, 0), Shape({hist, hw, 1}));
    const ir::TensorId rows =
        b.Reshape(f.step.Saved(in, "ple", s, hw), Shape({hw, segments[s].rows, 1}));
    const ir::TensorId padded = b.Concat(history, b.Cont(b.Transpose(rows)), 0);
    const ir::TensorDesc pt = Desc(b, padded);
    const ir::TensorId tail =
        b.View(padded, Shape({hist, hw, 1}), {pt.nb[0], pt.nb[1], pt.nb[2], pt.nb[2]},
               (pt.ne(0) - hist) * pt.nb[0]);
    const ir::TensorId kept = b.Reshape(b.Cont(tail), Shape({hist * hw, 1}));
    b.CacheStore(f.State(s, 0), kept, f.step.StateRow());
    b.SetJoined();
  }
  return {};
}

std::expected<void, std::string> PleReference(RefIO& io) {
  const Instance& in = *io.instance;
  const PleDims d = PleDimsOf(in);
  const std::uint64_t hw = d.hw();
  const std::uint64_t hist = d.history();
  const std::uint64_t rows = io.rows();
  const std::uint64_t side = rows == 0 ? 0 : io.side.size() / rows;
  if (side == 0 || io.side.size() != side * rows || io.residual.size() != hw * rows) {
    return std::unexpected("ple reference: side input or streams");
  }
  std::array<std::vector<float>, kPleRoles.size()> adjusted;
  if (in.params.Flag("norm_zero_centered")) {
    for (const auto role : {kNormKey, kNormQuery, kNormConv}) {
      adjusted[role] = io.weights.at(role);
      for (float& value : adjusted[role]) value += 1.0F;
    }
  }
  const auto weight = [&](std::uint16_t role) {
    return std::span<const float>(adjusted[role].empty() ? io.weights.at(role) : adjusted[role]);
  };
  const auto grouped = [&](std::vector<double> x, std::uint16_t role) {
    RmsRows(x, d.width, d.eps, {});
    for (std::size_t i = 0; i < x.size(); ++i) x[i] *= static_cast<double>(weight(role)[i % hw]);
    return x;
  };
  const std::vector<double> value = Product(weight(kValue), side, d.width, io.side);
  const std::vector<double> key = grouped(Product(weight(kKey), side, hw, io.side), kNormKey);
  const std::vector<double> query = grouped(io.residual, kNormQuery);
  std::vector<double> gated(hw * rows);
  for (std::uint64_t t = 0; t < rows; ++t) {
    for (std::uint64_t s = 0; s < d.hc; ++s) {
      double dot = 0.0;
      for (std::uint64_t i = 0; i < d.width; ++i) {
        const std::size_t at = (t * hw) + (s * d.width) + i;
        dot += key[at] * query[at];
      }
      dot /= std::sqrt(static_cast<double>(d.width));
      const double sgn = dot > 0.0 ? 1.0 : (dot < 0.0 ? -1.0 : 0.0);
      const double gate =
          1.0 / (1.0 + std::exp(-(sgn * std::sqrt(std::max(std::fabs(dot), 1e-6)))));
      for (std::uint64_t i = 0; i < d.width; ++i) {
        gated[(t * hw) + (s * d.width) + i] = value[(t * d.width) + i] * gate;
      }
    }
  }
  const std::vector<double> normalized = grouped(gated, kNormConv);
  const auto conv_w = weight(kConvW);  // [conv, hw]: tap k of channel c at c · conv + k
  std::uint64_t first = 0;
  for (std::size_t si = 0; si < io.segments.size(); ++si) {
    const RefSegment& seg = io.segments[si];
    auto& history = io.state[si].at(0);  // [hist, hw], time fastest
    const std::uint64_t span = hist + seg.rows;
    std::vector<double> padded(span * hw);
    for (std::uint64_t c = 0; c < hw; ++c) {
      for (std::uint64_t j = 0; j < hist; ++j) padded[(c * span) + j] = history[(c * hist) + j];
      for (std::uint64_t t = 0; t < seg.rows; ++t)
        padded[(c * span) + hist + t] = normalized[((first + t) * hw) + c];
    }
    for (std::uint64_t c = 0; c < hw; ++c) {
      for (std::uint64_t j = 0; j < hist; ++j) {
        history[(c * hist) + j] = Stored(format::Encoding::kF32, padded[(c * span) + seg.rows + j]);
      }
    }
    for (std::uint64_t t = 0; t < seg.rows; ++t) {
      for (std::uint64_t c = 0; c < hw; ++c) {
        double sum = 0.0;
        for (std::uint64_t k = 0; k < d.conv; ++k) {
          const std::uint64_t start = hist - ((d.conv - 1 - k) * d.dil);
          sum += padded[(c * span) + start + t] * static_cast<double>(conv_w[(c * d.conv) + k]);
        }
        const std::size_t at = ((first + t) * hw) + c;
        io.residual[at] += gated[at] + Silu(sum);
      }
    }
    first += seg.rows;
  }
  return {};
}

// ------------------------------------------------------------- embedding.mtp

constexpr std::array kMtpParams{
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    // The tap of its target it reads (a spec read of that name).
    ParamSpec{.name = "tap", .type = ParamType::kName},
    kNormZeroCentered,
};
// Version 2 declares the cache/token relation and repeatable progress;
// version 1 retains its original layout and orchestration contract.
constexpr std::array kMtpRepeatParams{
    kMtpParams[0],
    kMtpParams[1],
    kNormZeroCentered,
    ParamSpec{.name = "token_offset", .type = ParamType::kInt, .least = 0, .most = 1},
};

enum : std::uint16_t { kTable, kNormEmbd, kFcEmbd, kNormHidden, kFcHidden };
constexpr std::array kMtpRoles{
    WeightRole{.name = "table"},
    WeightRole{.name = "norm_embd", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "fc_embd"},
    WeightRole{.name = "norm_hidden", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "fc_hidden"},
};

std::expected<std::vector<RoleShape>, std::string> MtpShapes(const Instance& in) {
  if (NormOffsetContradictsFold(in))
    return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  if (hc < 2) return std::unexpected("an MTP drafter's input in a stage of one stream");
  return std::vector<RoleShape>{Role(kTable, {w, 0}), Role(kNormEmbd, {w}), Role(kFcEmbd, {w, w}),
                                Role(kNormHidden, {w * hc}), Role(kFcHidden, {w, w})};
}

model::StateContract MtpState(const Instance& in) {
  return {.tensors = {{.name = std::string(in.params.Name("tap")),
                       .kind = model::StateKind::kFeatures,
                       .encoding = format::Encoding::kF32,
                       .cell_bytes = in.stage.width * in.stage.streams * 4,
                       .cells = model::CellRule::kPerRow,
                       .window_or_ratio = 1,
                       .rollback = model::Rollback::kSnapshot,
                       .target_writes = true}}};
}

model::StateContract MtpRepeatState(const Instance& in) {
  auto state = MtpState(in);
  state.tensors.push_back(model::DraftProgressControl(
      "progress", in.params.Int("token_offset") == 0 ? model::DraftRelation::kTokenAtCache
                                                     : model::DraftRelation::kTokenAfterCache));
  return state;
}

std::expected<FragmentOut, std::string> MtpEmit(Fragment& f) {
  F32Parameters<kMtpRoles.size()> parameters(f);
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const std::uint64_t rows = f.step.rows();
  const float eps = F32(in.params.Real("eps"));
  const ir::TensorId table = f.Weight(kTable);
  const ir::TensorId tokens = f.step.Tokens();
  ir::TensorId e = b.GetRows(table, tokens);
  const ir::TensorId fc_embd = f.Weight(kFcEmbd);
  e = b.RmsNorm(e, eps);
  e = b.Mul(e, parameters.Norm(kNormEmbd));
  e = b.MatMul(fc_embd, e);
  const ir::TensorId hidden = f.step.Features(w * hc);
  ir::TensorId h = b.RmsNorm(hidden, eps);
  h = b.Mul(h, parameters.Norm(kNormHidden));
  const ir::TensorId fc_hidden = f.Weight(kFcHidden);
  h = b.Reshape(h, Shape({w, hc * rows}));
  h = b.MatMul(fc_hidden, h);
  f.step.PreserveOwnerProduct(h);
  h = b.Reshape(h, Shape({w, hc, rows}));
  e = b.Reshape(e, Shape({w, 1, rows}));
  const ir::TensorId residual = b.Add(h, e);
  return FragmentOut{.residual = residual};
}

std::expected<void, std::string> MtpReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const std::uint64_t rows = io.rows();
  const double eps = F32(in.params.Real("eps"));
  if (io.tokens.size() != rows || io.features.size() != w * hc * rows) {
    return std::unexpected("mtp embedding reference: tokens or features");
  }
  std::array<std::vector<float>, kMtpRoles.size()> adjusted;
  if (in.params.Flag("norm_zero_centered")) {
    for (const auto role : {kNormEmbd, kNormHidden}) {
      adjusted[role] = io.weights.at(role);
      for (float& value : adjusted[role]) value += 1.0F;
    }
  }
  const auto weight = [&](std::uint16_t role) {
    return std::span<const float>(adjusted[role].empty() ? io.weights.at(role) : adjusted[role]);
  };
  std::vector<double> e(w * rows);
  const std::uint64_t vocab = weight(kTable).size() / w;
  for (std::uint64_t r = 0; r < rows; ++r) {
    const auto t = static_cast<std::uint64_t>(io.tokens[r]);
    if (io.tokens[r] < 0 || t >= vocab) return std::unexpected("mtp embedding reference: a token");
    for (std::uint64_t i = 0; i < w; ++i)
      e[(r * w) + i] = static_cast<double>(weight(kTable)[(t * w) + i]);
  }
  RmsRows(e, w, eps, weight(kNormEmbd));
  e = Product(weight(kFcEmbd), w, w, e);
  std::vector<double> h = io.features;
  RmsRows(h, w * hc, eps, weight(kNormHidden));
  h = Product(weight(kFcHidden), w, w, h);  // each stream's row
  io.residual.assign(w * hc * rows, 0.0);
  for (std::uint64_t r = 0; r < rows; ++r) {
    for (std::uint64_t s = 0; s < hc; ++s) {
      for (std::uint64_t i = 0; i < w; ++i) {
        const std::size_t at = (((r * hc) + s) * w) + i;
        io.residual[at] = h[at] + e[(r * w) + i];
      }
    }
  }
  return {};
}

}  // namespace

const ComponentDef& SideNgram() {
  static const ComponentDef def{
      .kind = "side.ngram",
      .version = 1,
      .row_local = true,
      .side_source = true,
      .params = kNgramParams,
      .roles = kNgramRoles,
      .shapes = NgramShapes,

      .emit = NgramEmit,
      .reference = NgramReference,
      .side_rows = NgramRows,
      .side_count = NgramCount,
  };
  return def;
}

const ComponentDef& ResidualPle() {
  static const ComponentDef def{
      .kind = "residual.ple",
      .version = 1,
      .side_input = PleInput,
      .params = kPleParams,
      .roles = kPleRoles,
      .shapes = PleShapes,
      .state = PleState,
      .max_state_tensors = 1,

      .emit = PleEmit,
      .reference = PleReference,
      .commit = PleCommit,
      .max_commit_saves = 1,
  };
  return def;
}

const ComponentDef& EmbeddingMtp() {
  static const ComponentDef def{
      .kind = "embedding.mtp",
      .version = 1,
      .row_local = true,
      .params = kMtpParams,
      .roles = kMtpRoles,
      .shapes = MtpShapes,
      .state = MtpState,
      .max_state_tensors = 1,

      .emit = MtpEmit,
      .reference = MtpReference,
  };
  return def;
}

const ComponentDef& EmbeddingMtpRepeat() {
  static const ComponentDef def = [] {
    ComponentDef value = EmbeddingMtp();
    value.version = 2;
    value.params = kMtpRepeatParams;
    value.state = MtpRepeatState;
    value.max_state_tensors = 2;
    return value;
  }();
  return def;
}

}  // namespace llmp::components::defs
