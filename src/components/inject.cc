// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// embedding.inject, version 1: a block drafter's input (DeepSeek V4's
// DSpark: llama.cpp's dflash.cpp graph_dsv4, kernels/ggml/dsv4_graph.cc's
// DraftTrunk and Inject). In its own steps, its rows of a token table (its
// target's, bound by stage), scaled: a draft block's anchor and mask
// tokens. In its target's steps, its injection (ComponentDef::inject): the
// target's features at every committed row, the `taps` taps it reads
// ("<tap>_0" and on: the mean of the target's streams entering each of
// the target layers it names) joined per row in order, each segment's last
// rows (as many as its rings hold, `cells`) projected by `fc` and
// RMS-normed with its weight: the input from which each of its attention
// layers writes its ring (attention.mqa's injection).
//
// The fragments emit what dsv4_graph.cc's reference form records: the
// injection after the target's head, the features joined as each tap is
// made (llama.cpp's layer_inp extraction), then the projection.

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
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

constexpr std::array kParams{
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "scale",
              .type = ParamType::kReal,
              .required = false,
              .least = 1e-30,
              .most = 1e6,
              .fallback = 1.0},
    // The taps its injection reads: "<tap>_0" to "<tap>_<taps - 1>".
    ParamSpec{.name = "tap", .type = ParamType::kName},
    ParamSpec{.name = "taps", .type = ParamType::kInt, .least = 1, .most = 16},
    // The rows of each segment it injects: its rings' cells.
    ParamSpec{.name = "cells", .type = ParamType::kInt, .least = 1, .most = 1 << 24},
};
enum : std::uint16_t { kTable, kFc, kNorm };
constexpr std::array kRoles{WeightRole{.name = "table"}, WeightRole{.name = "fc"},
                            WeightRole{.name = "norm", .encodings = kF32Only}};
constexpr std::array kBatch{
    BatchClass{.op = ir::Op::kGetRows, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kRmsNorm, .batch = ir::Batch::kJoined},
};
constexpr std::array<std::string_view, 2> kPatterns{"rms_norm_mul", "vecq_inputs"};

std::uint64_t Taps(const Instance& in) { return static_cast<std::uint64_t>(in.params.Int("taps")); }

std::vector<std::string> Reads(const Instance& in) {
  std::vector<std::string> out;
  for (std::uint64_t i = 0; i < Taps(in); ++i) {
    out.push_back(std::format("{}_{}", in.params.Name("tap"), i));
  }
  return out;
}

std::uint32_t Rows(const Instance& in) {
  return static_cast<std::uint32_t>(in.params.Int("cells"));
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  const std::uint64_t w = in.stage.width;
  return std::vector<RoleShape>{Role(kTable, {w, 0}), Role(kFc, {w * Taps(in), w}),
                                Role(kNorm, {w})};
}

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  const ir::TensorId rows = f.b.GetRows(f.Weight(kTable), f.step.Tokens());
  const float scale = F32(f.instance.params.Real("scale"));
  return FragmentOut{.stream = scale == 1.0F ? rows : f.b.Scale(rows, scale)};
}

std::expected<FragmentOut, std::string> Inject(Fragment& f) {
  ir::Builder& b = f.b;
  const std::vector<std::string> taps = Reads(f.instance);
  // The features: each tap as it is made, joined per row.
  ir::TensorId features{};
  for (const std::string& name : taps) {
    const ir::TensorId tap = f.Side(name);
    if (!tap.valid())
      return std::unexpected(std::format("the injection's tap {} is not made", name));
    features = features.valid() ? b.Concat(features, tap, 0) : tap;
  }
  // Each segment's last rows, joined (all of one segment's: the features).
  const auto segments = f.step.segments();
  const std::uint64_t cells = Rows(f.instance);
  ir::TensorId x{};
  if (std::ranges::all_of(segments, [&](const SegmentRows& seg) { return seg.rows <= cells; })) {
    x = features;
  } else {
    const ir::TensorDesc fd = Desc(b, features);
    for (const SegmentRows& seg : segments) {
      const std::uint64_t kept = std::min<std::uint64_t>(seg.rows, cells);
      const format::Shape shape = Shape({fd.ne(0), kept});
      const ir::TensorId one = b.View(features, shape, ViewStrides(fd.nb[0], fd.nb[1], 0, shape),
                                      (std::uint64_t{seg.first_row} + seg.rows - kept) * fd.nb[1]);
      x = x.valid() ? b.Concat(x, one, 1) : one;
    }
  }
  const float eps = F32(f.instance.params.Real("eps"));
  return FragmentOut{.stream = b.Mul(b.RmsNorm(b.MatMul(f.Weight(kFc), x), eps), f.Weight(kNorm))};
}

std::expected<void, std::string> Reference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t w = in.stage.width;
  if (io.tokens.size() != io.rows()) return std::unexpected("inject embedding reference: tokens");
  const std::span<const float> table = io.weights.at(kTable);
  const double scale = F32(in.params.Real("scale"));
  io.stream.assign(w * io.rows(), 0.0);
  for (std::size_t r = 0; r < io.tokens.size(); ++r) {
    const auto t = static_cast<std::uint64_t>(io.tokens[r]);
    if (io.tokens[r] < 0 || (t + 1) * w > table.size()) {
      return std::unexpected("inject embedding reference: a token");
    }
    for (std::uint64_t i = 0; i < w; ++i) {
      io.stream[(r * w) + i] = static_cast<double>(table[(t * w) + i]) * scale;
    }
  }
  return {};
}

}  // namespace

const ComponentDef& EmbeddingInject() {
  static const ComponentDef def{
      .kind = "embedding.inject",
      .version = 1,
      .row_local = true,
      .params = kParams,
      .roles = kRoles,
      .shapes = Shapes,
      .batch_classes = kBatch,
      .patterns = kPatterns,
      .emit = Emit,
      .reference = Reference,
      .inject = Inject,
      .inject_reads = Reads,
      .inject_rows = Rows,
  };
  return def;
}

}  // namespace llmp::components::defs
