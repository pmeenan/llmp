// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// norm.rms, version 1: rms_norm(x) · w over the stream (a final norm).
// embedding.token, version 1: rows of a token table, scaled (Gemma's
// sqrt(width)). head.linear, version 1: the output rows' product with the
// output matrix (tied or not), an optional final softcap applied as
// today's builders do (scale by 1 / cap, tanh, scale by cap), and on a
// greedy step each row's highest logit (the host-greedy flavor).

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
#include "ir/graph.h"
#include "ir/op.h"

namespace llmp::components::defs {
namespace {

constexpr std::array kJoined{
    BatchClass{.op = ir::Op::kRmsNorm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kGetRows, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kScale, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kArgmax, .batch = ir::Batch::kJoined},
};

// ------------------------------------------------------------------ norm.rms

constexpr std::array kNormParams{
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
};
constexpr std::array kNormRoles{WeightRole{.name = "weight", .encodings = kF32Only}};
constexpr std::array<std::string_view, 2> kNormPatterns{"rms_norm_mul", "rms_norm_mul_add"};

std::expected<std::vector<RoleShape>, std::string> NormShapes(const Instance& in) {
  return std::vector<RoleShape>{Role(0, {in.stage.width})};
}

std::expected<FragmentOut, std::string> NormEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const float eps = F32(f.instance.params.Real("eps"));
  return FragmentOut{.stream = f.b.Mul(f.b.RmsNorm(f.stream, eps), f.Weight(0))};
}

std::expected<void, std::string> NormReference(RefIO& io) {
  const Instance& in = *io.instance;
  if (io.stream.size() != in.stage.width * io.rows()) {
    return std::unexpected("norm reference: a stream of another shape");
  }
  RmsRows(io.stream, in.stage.width, F32(in.params.Real("eps")), io.weights.at(0));
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
  const ir::TensorId rows = f.b.GetRows(f.Weight(0), f.step.Tokens());
  return FragmentOut{.stream = f.b.Scale(rows, F32(f.instance.params.Real("scale")))};
}

std::expected<void, std::string> EmbeddingReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t width = in.stage.width;
  const std::span<const float> table = io.weights.at(0);
  const std::uint64_t vocab = table.size() / width;
  if (io.tokens.size() != io.rows()) return std::unexpected("embedding reference: tokens");
  const double scale = F32(in.params.Real("scale"));
  io.stream.assign(width * io.tokens.size(), 0.0);
  for (std::size_t r = 0; r < io.tokens.size(); ++r) {
    const auto t = static_cast<std::uint64_t>(io.tokens[r]);
    if (io.tokens[r] < 0 || t >= vocab) return std::unexpected("embedding reference: a token");
    for (std::uint64_t i = 0; i < width; ++i) {
      io.stream[(r * width) + i] = static_cast<double>(table[(t * width) + i]) * scale;
    }
  }
  return {};
}

// --------------------------------------------------------------- head.linear

constexpr std::array kHeadRoles{WeightRole{.name = "weight"}};
constexpr std::array<std::string_view, 1> kHeadPatterns{"prepared_products"};

std::expected<std::vector<RoleShape>, std::string> HeadShapes(const Instance& in) {
  if (in.stage.softcap.has_value() && !(*in.stage.softcap > 0.0)) {
    return std::unexpected("a final softcap above zero");
  }
  return std::vector<RoleShape>{Role(0, {in.stage.width, 0})};
}

std::expected<FragmentOut, std::string> HeadEmit(Fragment& f) {
  ir::Builder& b = f.b;
  const ir::TensorId rows = b.GetRows(f.stream, f.step.OutputRows());
  ir::TensorId logits = b.MatMul(f.Weight(0), rows);
  if (f.instance.stage.softcap.has_value()) {
    const float cap = F32(*f.instance.stage.softcap);
    logits = b.Scale(logits, 1.0F / cap);
    logits = b.Unary(logits, ir::UnaryFn::kTanh);
    logits = b.Scale(logits, cap);
  }
  FragmentOut out{.stream = f.stream, .logits = logits};
  if (f.step.greedy()) out.greedy = b.Argmax(logits, ir::ArgmaxNan::kFirstColumn);
  return out;
}

std::expected<void, std::string> HeadReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t width = in.stage.width;
  const std::span<const float> w = io.weights.at(0);
  const std::uint64_t vocab = w.size() / width;
  if (io.stream.size() != width * io.rows()) {
    return std::unexpected("head reference: a stream of another shape");
  }
  // Each segment's last `outputs` rows, in row order.
  std::vector<double> selected;
  std::uint64_t first = 0;
  for (const RefSegment& s : io.segments) {
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
  io.tokens_out.clear();
  if (io.greedy) {
    for (std::size_t r = 0; r < io.logits.size() / vocab; ++r) {
      std::size_t best = 0;
      for (std::size_t j = 1; j < vocab; ++j) {
        if (io.logits[(r * vocab) + j] > io.logits[(r * vocab) + best]) best = j;
      }
      io.tokens_out.push_back(static_cast<std::int32_t>(best));
    }
  }
  return {};
}

}  // namespace

const ComponentDef& NormRms() {
  static const ComponentDef def{
      .kind = "norm.rms",
      .version = 1,
      .params = kNormParams,
      .roles = kNormRoles,
      .shapes = NormShapes,
      .batch_classes = std::span(kJoined).first(2),
      .patterns = kNormPatterns,
      .emit = NormEmit,
      .reference = NormReference,
  };
  return def;
}

const ComponentDef& EmbeddingToken() {
  static const ComponentDef def{
      .kind = "embedding.token",
      .version = 1,
      .params = kEmbeddingParams,
      .roles = kEmbeddingRoles,
      .shapes = EmbeddingShapes,
      .batch_classes = std::span(kJoined).subspan(2, 2),
      .emit = EmbeddingEmit,
      .reference = EmbeddingReference,
  };
  return def;
}

const ComponentDef& HeadLinear() {
  static const ComponentDef def{
      .kind = "head.linear",
      .version = 1,
      .roles = kHeadRoles,
      .shapes = HeadShapes,
      .batch_classes = std::span(kJoined).subspan(2),
      .patterns = kHeadPatterns,
      .emit = HeadEmit,
      .reference = HeadReference,
  };
  return def;
}

}  // namespace llmp::components::defs
