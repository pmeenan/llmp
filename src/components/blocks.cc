// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// norm.rms, version 1: rms_norm(x) · w over the stream (a final norm).
// scale.learned, version 1: the stream times one learned value (Gemma 4's
// layer output scale), after the layer's residual add.
// embedding.token, version 1: rows of a token table, scaled (Gemma's
// sqrt(width)).
// embedding.projected, version 1: a drafter's input (Gemma 4's
// assistant): rows of a token table (its target's, bound by stage),
// scaled, beside a feature of the same width (the target's tap, or the
// drafter's own last feature: a step input the engine binds), projected
// to the stream. Its state is the tap the target's steps write for it
// (target_writes): each output row's feature, a ring of `tap_cells` cells
// by position, from which the engine binds a unit's first feature.
// head.linear, version 1: the output rows' product with the output matrix
// (tied or not; the rows gathered here unless the stream was narrowed to
// them), an optional final softcap applied as today's builders do (scale
// by 1 / cap, tanh, scale by cap), and on a greedy step each row's highest
// logit (the host-greedy flavor); a drafter's head reads every row
// (`gather` false: nothing to gather or narrow) and projects its rows back
// to its target's width (`features`: its next feature); it splits by the
// vocabulary. A block drafter's head (`markov`: DSpark's, llama.cpp's
// dflash.cpp build_dspark_markov_head) biases each of a segment's rows'
// logits by markov_w2 · markov_w1[previous] (the previous row's highest
// biased logit's token, the segment's first token for its first row), and
// a greedy step's tokens are each row's highest biased logit (llmp.argmax's
// flavor, which ignores NaN).

#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"

namespace llmp::components::defs {
namespace {

// ------------------------------------------------------------------ norm.rms

constexpr std::array kNormParams{
    kNormZeroCentered,
    ParamSpec{.name = "channels",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 65536,
              .fallback = 0},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
};
constexpr std::array kNormRoles{
    WeightRole{.name = "weight", .encodings = kNormEncodings, .derived = kZeroCenteredNorm}};

std::uint64_t NormChannels(const Instance& in) {
  const auto explicit_width = in.params.Int("channels");
  return explicit_width == 0 ? in.stage.width : static_cast<std::uint64_t>(explicit_width);
}

std::expected<std::vector<RoleShape>, std::string> NormShapes(const Instance& in) {
  if (NormOffsetContradictsFold(in))
    return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
  const auto width = NormChannels(in);
  if (width == 0 || width > kMaxStreamWidth)
    return std::unexpected("norm channels exceed stream width bound");
  return std::vector<RoleShape>{Role(0, {width})};
}

std::expected<FragmentOut, std::string> NormEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  if (f.b.tensor(f.stream).ne(0) != NormChannels(f.instance))
    return std::unexpected("norm stream width does not match configured channels");
  const float eps = F32(f.instance.params.Real("eps"));
  const ir::TensorId normed = f.b.RmsNorm(f.stream, eps);
  ir::TensorId weight = NormScale(f.b, f.Weight(0));
  if (f.instance.params.Flag("norm_zero_centered")) weight = f.b.Scale(weight, 1.0F, 1.0F);
  return FragmentOut{.stream = f.b.Mul(normed, weight)};
}

std::expected<void, std::string> NormReference(RefIO& io) {
  const Instance& in = *io.instance;
  const auto width = NormChannels(in);
  if (width == 0 || io.stream.size() != width * io.rows() || io.weights.at(0).size() != width) {
    return std::unexpected("norm reference: a stream of another shape");
  }
  auto weight = io.weights.at(0);
  if (in.params.Flag("norm_zero_centered"))
    for (float& value : weight) value += 1.0F;
  RmsRows(io.stream, width, F32(in.params.Real("eps")), weight);
  return {};
}

// ------------------------------------------------------------- scale.learned

constexpr std::array kScaleRoles{WeightRole{.name = "weight", .encodings = kF32Only}};

std::expected<std::vector<RoleShape>, std::string> ScaleShapes(const Instance&) {
  return std::vector<RoleShape>{Role(0, {1})};
}

std::expected<FragmentOut, std::string> ScaleEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  return FragmentOut{.stream = f.b.Mul(f.stream, f.Weight(0))};
}

std::expected<void, std::string> ScaleReference(RefIO& io) {
  const Instance& in = *io.instance;
  if (io.stream.size() != in.stage.width * io.rows() || io.weights.at(0).size() != 1) {
    return std::unexpected("scale reference: a stream or weight of another shape");
  }
  const auto w = static_cast<double>(io.weights[0][0]);
  for (double& x : io.stream) x *= w;
  return {};
}

// ----------------------------------------------------------- embedding.token

constexpr std::array kEmbeddingParams{
    ParamSpec{.name = "scale",
              .type = ParamType::kReal,
              .required = false,
              .least = 1e-30,
              .most = 1e6,
              .fallback = 1.0},
};
constexpr std::array kEmbeddingRoles{WeightRole{.name = "table"}};

std::expected<std::vector<RoleShape>, std::string> EmbeddingShapes(const Instance& in) {
  return std::vector<RoleShape>{Role(0, {in.stage.width, 0})};
}

std::expected<FragmentOut, std::string> EmbeddingEmit(Fragment& f) {
  const ir::TensorId table = f.Weight(0);
  const ir::TensorId tokens =
      f.stream.valid() ? f.b.Reshape(f.stream, Shape({f.step.rows()})) : f.step.Tokens();
  const ir::TensorId rows = f.b.GetRows(table, tokens);
  // A scale of 1 is no node (Qwen3.8's rows are the stream as they are).
  const float scale = F32(f.instance.params.Real("scale"));
  return FragmentOut{.stream = scale == 1.0F ? rows : f.b.Scale(rows, scale)};
}

// Each token's row of `table` (`width` values a row) times `scale`.
std::expected<std::vector<double>, std::string> TableRows(std::span<const float> table,
                                                          std::uint64_t width,
                                                          std::span<const std::int32_t> tokens,
                                                          double scale) {
  const std::uint64_t vocab = table.size() / width;
  std::vector<double> out(width * tokens.size(), 0.0);
  for (std::size_t r = 0; r < tokens.size(); ++r) {
    const auto t = static_cast<std::uint64_t>(tokens[r]);
    if (tokens[r] < 0 || t >= vocab) return std::unexpected("embedding reference: a token");
    for (std::uint64_t i = 0; i < width; ++i) {
      out[(r * width) + i] = static_cast<double>(table[(t * width) + i]) * scale;
    }
  }
  return out;
}

std::expected<void, std::string> EmbeddingReference(RefIO& io) {
  const Instance& in = *io.instance;
  if (io.tokens.size() != io.rows()) return std::unexpected("embedding reference: tokens");
  auto rows = TableRows(io.weights.at(0), in.stage.width, io.tokens, F32(in.params.Real("scale")));
  if (!rows) return std::unexpected(rows.error());
  io.stream = std::move(*rows);
  return {};
}

// ------------------------------------------------------- embedding.projected

constexpr std::array kProjectedParams{
    ParamSpec{.name = "scale", .type = ParamType::kReal, .least = 1e-30, .most = 1e6},
    // The feature's width: the token table's and the tap's (its target's).
    ParamSpec{.name = "feature", .type = ParamType::kInt, .least = 1, .most = 1 << 16},
    // The tap of its target it reads (a spec read of that name).
    ParamSpec{.name = "tap", .type = ParamType::kName},
    // The tap's cells: positions in a ring, enough for a verify's rows.
    ParamSpec{.name = "tap_cells",
              .type = ParamType::kInt,
              .required = false,
              .least = 1,
              .most = 4096,
              .fallback = 16},
};
enum : std::uint16_t { kTable, kProjection };
constexpr std::array kProjectedRoles{WeightRole{.name = "table"}, WeightRole{.name = "projection"}};

std::uint64_t FeatureWidth(const Instance& in) {
  return static_cast<std::uint64_t>(in.params.Int("feature"));
}

std::expected<std::vector<RoleShape>, std::string> ProjectedShapes(const Instance& in) {
  const std::uint64_t f = FeatureWidth(in);
  return std::vector<RoleShape>{Role(kTable, {f, 0}), Role(kProjection, {2 * f, in.stage.width})};
}

model::StateContract ProjectedState(const Instance& in) {
  const std::uint64_t f = FeatureWidth(in);
  return {.tensors = {{.name = std::string(in.params.Name("tap")),
                       .kind = model::StateKind::kFeatures,
                       .encoding = format::Encoding::kF32,
                       .cell_bytes = f * 4,
                       .cells = model::CellRule::kPositionRing,
                       .fixed_cells = static_cast<std::uint32_t>(in.params.Int("tap_cells")),
                       // A rejected row's cell is never read: a unit reads
                       // only its committed last position's.
                       .rollback = model::Rollback::kTruncate,
                       .target_writes = true}}};
}

std::expected<FragmentOut, std::string> ProjectedEmit(Fragment& f) {
  ir::Builder& b = f.b;
  const ir::TensorId table = f.Weight(kTable);
  const ir::TensorId tokens = f.step.Tokens();
  const ir::TensorId x = b.Scale(b.GetRows(table, tokens), F32(f.instance.params.Real("scale")));
  const ir::TensorId features = f.step.Features(FeatureWidth(f.instance));
  const ir::TensorId projection = f.Weight(kProjection);
  return FragmentOut{.stream = b.MatMul(projection, b.Concat(x, features, 0))};
}

std::expected<void, std::string> ProjectedReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t f = FeatureWidth(in);
  const std::uint64_t rows = io.rows();
  if (io.tokens.size() != rows || io.features.size() != f * rows) {
    return std::unexpected("projected embedding reference: tokens or features");
  }
  auto x = TableRows(io.weights.at(kTable), f, io.tokens, F32(in.params.Real("scale")));
  if (!x) return std::unexpected(x.error());
  std::vector<double> joined(2 * f * rows);
  for (std::uint64_t r = 0; r < rows; ++r) {
    for (std::uint64_t i = 0; i < f; ++i) {
      joined[(r * 2 * f) + i] = (*x)[(r * f) + i];
      joined[(r * 2 * f) + f + i] = io.features[(r * f) + i];
    }
  }
  io.stream = Product(io.weights.at(kProjection), 2 * f, in.stage.width, joined);
  return {};
}

// --------------------------------------------------------------- head.linear

constexpr std::array kHeadParams{
    // The output rows gathered (each segment's last `outputs`); false: every
    // row is an output (a drafter's).
    ParamSpec{.name = "gather",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 1.0},
    // Its rows projected to this width as well (a drafter's next feature,
    // role projection); 0: none.
    ParamSpec{.name = "features",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 1 << 16,
              .fallback = 0},
    // A selected head (an MTP drafter's draft vocabulary): its weight's
    // rows are some tokens' (role ids, I32 [1, rows]: each row's token,
    // strictly ascending), and a greedy row's token is its best row's id.
    ParamSpec{.name = "selected",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0.0},
    // A block drafter's Markov head (roles markov_w1, markov_w2: [rank,
    // vocabulary] each), over a head of every row.
    ParamSpec{.name = "markov",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0.0},
    // A verify's and a draft pass's tokens as GGML's argmax picks them (the
    // lowest index of the maximum, NaNs ignored), not as a sampled row's
    // host-greedy pick.
    ParamSpec{.name = "native_argmax",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0.0},
    // The weight's first rows alone (an MTP drafter's draft vocabulary: the
    // lowest token IDs, or a selected head's first); 0: every row.
    ParamSpec{.name = "rows",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 1 << 24,
              .fallback = 0.0},
};
enum : std::uint16_t { kHeadWeight, kHeadProjection, kHeadIds, kMarkovW1, kMarkovW2 };
constexpr std::array<format::Encoding, 1> kI32Only{format::Encoding::kI32};
constexpr std::array kHeadRoles{WeightRole{.name = "weight"}, WeightRole{.name = "projection"},
                                WeightRole{.name = "ids", .encodings = kI32Only},
                                WeightRole{.name = "markov_w1"}, WeightRole{.name = "markov_w2"}};

// By the vocabulary: the output matrix by its rows, the logits gathered
// (each output row's logits, F32) where something needs them whole.
constexpr std::array kVocabRoles{RoleSplit{.role = kHeadWeight, .dim = 1}};
constexpr std::array kHeadSplits{
    SplitOption{.kind = SplitKind::kColumns,
                .collective = ir::Op::kAllGather,
                .units = nullptr,
                .bytes_per_step = [](const SplitStep& s) -> std::uint64_t {
                  return s.weights.empty() ? 0 : s.weights[0].ne[1] * s.outputs * 4;
                },
                .weights = kVocabRoles,
                .state = {}}};

std::uint64_t HeadFeatures(const Instance& in) {
  return static_cast<std::uint64_t>(in.params.Int("features"));
}

std::expected<std::vector<RoleShape>, std::string> HeadShapes(const Instance& in) {
  if (in.stage.softcap.has_value() && !(*in.stage.softcap > 0.0)) {
    return std::unexpected("a final softcap above zero");
  }
  std::vector<RoleShape> out{Role(kHeadWeight, {in.stage.width, 0})};
  if (HeadFeatures(in) != 0) {
    out.push_back(Role(kHeadProjection, {in.stage.width, HeadFeatures(in)}));
  }
  if (in.params.Flag("selected")) out.push_back(Role(kHeadIds, {1, 0}));
  if (in.params.Flag("markov")) {
    if (in.params.Flag("gather") || in.params.Flag("selected") || in.params.Int("rows") != 0 ||
        in.stage.softcap.has_value()) {
      return std::unexpected(
          "a Markov head reads every row and its full vocabulary, unselected and uncapped");
    }
    out.push_back(Role(kMarkovW1, {0, 0}));
    out.push_back(Role(kMarkovW2, {0, 0}));
  }
  return out;
}

// The Markov head's bias of each segment's rows (in their order): row i's
// logits plus w2 · w1[prev], prev the segment's first token, then the
// previous row's highest biased logit.
ir::TensorId Markov(Fragment& f, ir::TensorId logits) {
  ir::Builder& b = f.b;
  const ir::TensorDesc ld = Desc(b, logits);
  const format::Shape column = Shape({ld.ne(0), 1});
  ir::TensorId joined{};
  for (std::size_t s = 0; s < f.step.segments().size(); ++s) {
    const SegmentRows& seg = f.step.segments()[s];
    b.SetOwner(static_cast<std::uint8_t>(s));
    ir::TensorId prev =
        b.View(f.step.Tokens(), Shape({1}), {4, 4, 4, 4}, std::uint64_t{seg.first_row} * 4);
    ir::TensorId col{};
    ir::TensorId biased{};
    for (std::uint32_t i = 0; i < seg.rows; ++i) {
      const ir::TensorId base = b.View(logits, column, ViewStrides(ld.nb[0], ld.nb[1], 0, column),
                                       (std::uint64_t{seg.first_row} + i) * ld.nb[1]);
      if (i > 0) prev = b.Argmax(col);
      const ir::TensorId row = b.GetRows(f.Weight(kMarkovW1), prev);
      const ir::TensorId bias = b.MatMul(f.Weight(kMarkovW2), row);
      col = b.Add(base, bias);
      biased = biased.valid() ? b.Concat(biased, col, 1) : col;
    }
    b.SetJoined();
    joined = joined.valid() ? b.Concat(joined, biased, 1) : biased;
  }
  return joined;
}

bool HeadGathers(const Instance& in) { return in.params.Flag("gather"); }

std::expected<FragmentOut, std::string> HeadEmit(Fragment& f) {
  ir::Builder& b = f.b;
  const bool gather = f.instance.params.Flag("gather");
  if (!gather && f.step.outputs() != f.step.rows()) {
    return std::unexpected("a head of every row in a step whose rows are not all outputs");
  }
  // Its rows, unless it reads every row or what came before narrowed the
  // stream to them.
  const ir::TensorId rows =
      gather && !f.narrowed ? b.GetRows(f.stream, f.step.OutputRows()) : f.stream;
  // A prefix of the weight's rows.
  ir::TensorId weight = f.Weight(kHeadWeight);
  const auto prefix = static_cast<std::uint64_t>(f.instance.params.Int("rows"));
  const ir::TensorDesc wd = Desc(b, weight);
  if (prefix > wd.ne(1)) return std::unexpected("head rows exceed the weight's vocabulary");
  if (prefix != 0 && prefix < wd.ne(1)) {
    const format::Shape shape = Shape({wd.ne(0), prefix});
    weight = b.View(weight, shape, ViewStrides(wd.nb[0], wd.nb[1], 0, shape), 0);
  }
  ir::TensorId logits = b.MatMul(weight, rows);
  if (f.instance.stage.softcap.has_value()) {
    const float cap = F32(*f.instance.stage.softcap);
    logits = b.Scale(logits, 1.0F / cap);
    logits = b.Unary(logits, ir::UnaryFn::kTanh);
    logits = b.Scale(logits, cap);
  }
  if (f.instance.params.Flag("markov")) {
    const ir::TensorId biased = Markov(f, logits);
    FragmentOut out{.stream = f.stream, .logits = biased};
    if (f.step.greedy()) out.greedy = b.Argmax(biased, ir::ArgmaxNan::kIgnore);
    return out;
  }
  FragmentOut out{.stream = f.stream, .logits = logits};
  if (HeadFeatures(f.instance) != 0) out.features = b.MatMul(f.Weight(kHeadProjection), rows);
  if (f.step.greedy()) {
    // A verify's or a draft's tokens as GGML's argmax computes them where
    // the head says so (native_argmax); else as the host's greedy pick (a
    // NaN row's first column).
    const bool native =
        f.instance.params.Flag("native_argmax") &&
        (f.step.phase() == planner::Phase::kVerify || f.step.phase() == planner::Phase::kDraft);
    out.greedy = b.Argmax(logits, native ? ir::ArgmaxNan::kIgnore : ir::ArgmaxNan::kFirstColumn);
    // A selected head's rows to their tokens.
    if (f.instance.params.Flag("selected")) {
      out.greedy =
          b.Reshape(b.GetRows(f.Weight(kHeadIds), out.greedy), Shape({b.tensor(out.greedy).ne(0)}));
    }
  }
  return out;
}

std::expected<void, std::string> HeadReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t width = in.stage.width;
  std::span<const float> w = io.weights.at(kHeadWeight);
  std::uint64_t vocab = w.size() / width;
  // A prefix of the weight's rows.
  const auto prefix = static_cast<std::uint64_t>(in.params.Int("rows"));
  if (prefix > vocab) return std::unexpected("head rows exceed the weight's vocabulary");
  if (prefix != 0 && prefix < vocab) {
    vocab = prefix;
    w = w.first(vocab * width);
  }
  if (io.stream.size() != width * io.rows()) {
    return std::unexpected("head reference: a stream of another shape");
  }
  // Each segment's last `outputs` rows, in row order (every row when it
  // does not gather).
  const bool gather = in.params.Flag("gather");
  std::vector<double> selected;
  std::uint64_t first = 0;
  for (const RefSegment& s : io.segments) {
    if (!gather && s.outputs != s.rows) {
      return std::unexpected("head reference: rows that are not all outputs");
    }
    for (std::uint64_t i = s.rows - s.outputs; i < s.rows; ++i) {
      const auto row = std::span(io.stream).subspan((first + i) * width, width);
      selected.insert(selected.end(), row.begin(), row.end());
    }
    first += s.rows;
  }
  io.logits = Product(w, width, vocab, selected);
  if (in.stage.softcap.has_value()) {
    const double cap = *in.stage.softcap;
    for (double& l : io.logits) l = cap * std::tanh(l / cap);
  }
  const auto pick = [](std::span<const double> row, ir::ArgmaxNan nan) {
    std::size_t best = 0;
    bool found = false;
    for (std::size_t i = 0; i < row.size(); ++i) {
      if (nan == ir::ArgmaxNan::kIgnore) {
        if (!std::isnan(row[i]) && (!found || row[i] > row[best])) {
          best = i;
          found = true;
        }
      } else if (row[i] > row[best]) {
        best = i;
      }
    }
    return best;
  };
  if (in.params.Flag("markov")) {
    // Each segment's chain: its first token, then each row's best.
    const std::span<const float> w1 = io.weights.at(kMarkovW1);
    const std::span<const float> w2 = io.weights.at(kMarkovW2);
    const std::uint64_t rank = w1.size() / vocab;
    io.tokens_out.clear();
    std::uint64_t r = 0;
    for (const RefSegment& s : io.segments) {
      std::int64_t prev = io.tokens.at(r);
      for (std::uint32_t i = 0; i < s.rows; ++i, ++r) {
        if (prev < 0 || static_cast<std::uint64_t>(prev) >= vocab) {
          return std::unexpected("head reference: a Markov token past the vocabulary");
        }
        double* row = &io.logits[r * vocab];
        for (std::uint64_t v = 0; v < vocab; ++v) {
          double bias = 0.0;
          for (std::uint64_t k = 0; k < rank; ++k) {
            bias += static_cast<double>(w2[(v * rank) + k]) *
                    static_cast<double>(w1[(static_cast<std::uint64_t>(prev) * rank) + k]);
          }
          row[v] += bias;
        }
        const std::size_t best = pick({row, vocab}, ir::ArgmaxNan::kIgnore);
        prev = static_cast<std::int64_t>(best);
        if (io.greedy) io.tokens_out.push_back(static_cast<std::int32_t>(best));
      }
    }
    io.features_out.clear();
    return {};
  }
  io.features_out.clear();
  if (HeadFeatures(in) != 0) {
    io.features_out = Product(io.weights.at(kHeadProjection), width, HeadFeatures(in), selected);
  }
  io.tokens_out.clear();
  if (io.greedy) {
    const bool mapped = in.params.Flag("selected");
    for (std::size_t r = 0; r < io.logits.size() / vocab; ++r) {
      const std::size_t best = pick(std::span(io.logits).subspan(r * vocab, vocab), io.greedy_nan);
      io.tokens_out.push_back(mapped ? static_cast<std::int32_t>(io.weights.at(kHeadIds).at(best))
                                     : static_cast<std::int32_t>(best));
    }
  }
  return {};
}

}  // namespace

const ComponentDef& NormRms() {
  static const ComponentDef def{
      .kind = "norm.rms",
      .version = 1,
      .row_local = true,
      .params = kNormParams,
      .roles = kNormRoles,
      .shapes = NormShapes,

      .emit = NormEmit,
      .reference = NormReference,
  };
  return def;
}

const ComponentDef& ScaleLearned() {
  static const ComponentDef def{
      .kind = "scale.learned",
      .version = 1,
      .row_local = true,
      .roles = kScaleRoles,
      .shapes = ScaleShapes,

      .emit = ScaleEmit,
      .reference = ScaleReference,
  };
  return def;
}

const ComponentDef& EmbeddingToken() {
  static const ComponentDef def{
      .kind = "embedding.token",
      .version = 1,
      .row_local = true,
      .params = kEmbeddingParams,
      .roles = kEmbeddingRoles,
      .shapes = EmbeddingShapes,

      .emit = EmbeddingEmit,
      .reference = EmbeddingReference,
  };
  return def;
}

const ComponentDef& EmbeddingProjected() {
  static const ComponentDef def{
      .kind = "embedding.projected",
      .version = 1,
      .params = kProjectedParams,
      .roles = kProjectedRoles,
      .shapes = ProjectedShapes,
      .state = ProjectedState,
      .max_state_tensors = 1,

      .emit = ProjectedEmit,
      .reference = ProjectedReference,
  };
  return def;
}

const ComponentDef& HeadLinear() {
  static const ComponentDef def{
      .kind = "head.linear",
      .version = 1,
      .row_local = true,
      .params = kHeadParams,
      .roles = kHeadRoles,
      .shapes = HeadShapes,

      .splits = kHeadSplits,

      .gathers = HeadGathers,
      .emit = HeadEmit,
      .reference = HeadReference,
  };
  return def;
}

}  // namespace llmp::components::defs
