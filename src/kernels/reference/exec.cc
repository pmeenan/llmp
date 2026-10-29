// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "kernels/reference/exec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "execution/sampling.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"
#include "kernels/reference/decode.h"

namespace llmp::kernels::reference {

std::string_view ExecErrorName(ExecError error) {
  switch (error) {
    case ExecError::kGraph:
      return "graph";
    case ExecError::kUnbound:
      return "unbound";
    case ExecError::kBinding:
      return "binding";
    case ExecError::kUnsupported:
      return "unsupported";
    case ExecError::kIndex:
      return "index";
    case ExecError::kNumeric:
      return "numeric";
    case ExecError::kCollective:
      return "collective";
  }
  return "unknown";
}

namespace {

using format::Encoding;
using ir::IsLeaf;
using ir::Op;
using ir::TensorDesc;
using ir::TensorId;
using Ne = std::array<std::uint64_t, format::kMaxDims>;
using Vec = std::vector<double>;

std::unexpected<ExecFailure> Fail(ExecError e, std::string detail, std::uint32_t node = 0) {
  return std::unexpected(ExecFailure{.error = e, .node = node, .detail = std::move(detail)});
}

#define LLMP_TRY(expr)                         \
  if (auto llmp_try_ = (expr); !llmp_try_) {   \
    return std::unexpected(llmp_try_.error()); \
  }

#define LLMP_LOAD(var, id)                        \
  auto var##_loaded = Load(id);                   \
  if (!var##_loaded) {                            \
    return std::unexpected(var##_loaded.error()); \
  }                                               \
  const Vec& var = *var##_loaded;

constexpr std::size_t Ix(const Ne& ne, std::uint64_t i0, std::uint64_t i1, std::uint64_t i2,
                         std::uint64_t i3) {
  return static_cast<std::size_t>(i0 + (ne[0] * (i1 + (ne[1] * (i2 + (ne[2] * i3))))));
}

std::size_t Count(const Ne& ne) { return static_cast<std::size_t>(ne[0] * ne[1] * ne[2] * ne[3]); }

template <typename T>
T Raw(const std::byte* p) {
  T v{};
  std::memcpy(&v, p, sizeof(T));
  return v;
}

template <typename T>
void Put(std::byte* p, T v) {
  std::memcpy(p, &v, sizeof(T));
}

double ReadElement(Encoding e, const std::byte* p) {
  switch (e) {
    case Encoding::kF32:
      return Raw<float>(p);
    case Encoding::kF16:
      return F16ToF32(Raw<std::uint16_t>(p));
    case Encoding::kBF16:
      return Bf16ToF32(Raw<std::uint16_t>(p));
    case Encoding::kF64:
      return Raw<double>(p);
    case Encoding::kI8:
      return Raw<std::int8_t>(p);
    case Encoding::kI16:
      return Raw<std::int16_t>(p);
    case Encoding::kI32:
      return Raw<std::int32_t>(p);
    case Encoding::kI64:
      return static_cast<double>(Raw<std::int64_t>(p));
    case Encoding::kU8:
    case Encoding::kBool:
      return Raw<std::uint8_t>(p);
    case Encoding::kU16:
      return Raw<std::uint16_t>(p);
    case Encoding::kU32:
      return Raw<std::uint32_t>(p);
    case Encoding::kU64:
      return static_cast<double>(Raw<std::uint64_t>(p));
    case Encoding::kF8E4M3:
      return E4m3ToF32(Raw<std::uint8_t>(p));
    case Encoding::kF8E5M2:
      return E5m2ToF32(Raw<std::uint8_t>(p));
    case Encoding::kE8M0:
      return E8m0ToF32(Raw<std::uint8_t>(p));
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

template <typename T>
T Saturate(double v) {
  if (std::isnan(v)) {
    return T{0};
  }
  const double lo = static_cast<double>(std::numeric_limits<T>::lowest());
  const double hi = static_cast<double>(std::numeric_limits<T>::max());
  return static_cast<T>(std::clamp(std::nearbyint(v), lo, hi));
}

// False for an encoding the reference does not write element by element.
bool WriteElement(Encoding e, double v, std::byte* p) {
  switch (e) {
    case Encoding::kF32:
      Put(p, static_cast<float>(v));
      return true;
    case Encoding::kF16:
      Put(p, F32ToF16(static_cast<float>(v)));
      return true;
    case Encoding::kBF16:
      Put(p, F32ToBf16(static_cast<float>(v)));
      return true;
    case Encoding::kF64:
      Put(p, v);
      return true;
    case Encoding::kI8:
      Put(p, Saturate<std::int8_t>(v));
      return true;
    case Encoding::kI16:
      Put(p, Saturate<std::int16_t>(v));
      return true;
    case Encoding::kI32:
      Put(p, Saturate<std::int32_t>(v));
      return true;
    case Encoding::kI64:
      Put(p, Saturate<std::int64_t>(v));
      return true;
    case Encoding::kU8:
      Put(p, Saturate<std::uint8_t>(v));
      return true;
    case Encoding::kBool:
      Put(p, static_cast<std::uint8_t>(v != 0.0 ? 1 : 0));
      return true;
    case Encoding::kU16:
      Put(p, Saturate<std::uint16_t>(v));
      return true;
    case Encoding::kU32:
      Put(p, Saturate<std::uint32_t>(v));
      return true;
    case Encoding::kU64:
      Put(p, Saturate<std::uint64_t>(v));
      return true;
    default:
      return false;
  }
}

bool IsCollective(Op op) {
  return op == Op::kAllReduce || op == Op::kAllGather || op == Op::kAllToAll || op == Op::kSend ||
         op == Op::kRecv;
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double Silu(double x) { return x * Sigmoid(x); }
double Softplus(double x) { return x > 30.0 ? x : std::log1p(std::exp(x)); }
double GeluTanh(double x) {
  constexpr double kA = 0.044715;
  const double c = std::sqrt(2.0 / std::numbers::pi);
  return 0.5 * x * (1.0 + std::tanh(c * x * (1.0 + (kA * x * x))));
}
double Gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::numbers::sqrt2)); }

double Apply(ir::UnaryFn fn, double x) {
  switch (fn) {
    case ir::UnaryFn::kSilu:
      return Silu(x);
    case ir::UnaryFn::kGeluErf:
      return Gelu(x);
    case ir::UnaryFn::kGeluTanh:
      return GeluTanh(x);
    case ir::UnaryFn::kSigmoid:
      return Sigmoid(x);
    case ir::UnaryFn::kTanh:
      return std::tanh(x);
    case ir::UnaryFn::kRelu:
      return x > 0.0 ? x : 0.0;
    case ir::UnaryFn::kExp:
      return std::exp(x);
    case ir::UnaryFn::kSoftplus:
      return Softplus(x);
    case ir::UnaryFn::kSqrt:
      return std::sqrt(x);
    case ir::UnaryFn::kSqr:
      return x * x;
    case ir::UnaryFn::kNeg:
      return -x;
    case ir::UnaryFn::kAbs:
      return std::fabs(x);
    case ir::UnaryFn::kSgn:
      return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
  }
  return x;
}

double Glu(const ir::GluParams& p, double g, double u) {
  switch (p.kind) {
    case ir::GluKind::kSwiGlu:
      return Silu(g) * u;
    case ir::GluKind::kGeGluErf:
      return Gelu(g) * u;
    case ir::GluKind::kGeGluTanh:
      return GeluTanh(g) * u;
    case ir::GluKind::kSwiGluClamp: {
      const double lim = p.limit;
      return Silu(std::min(g, lim)) * std::clamp(u, -lim, lim);
    }
  }
  return 0.0;
}

// GGML's ALiBi slope for head h of n (soft_max and flash attention).
double Slope(float max_bias, std::uint64_t h, std::uint64_t n) {
  if (max_bias <= 0.0f) {
    return 1.0;
  }
  const auto log2 = static_cast<std::uint64_t>(1)
                    << static_cast<unsigned>(std::floor(std::log2(static_cast<double>(n))));
  const double m0 = std::pow(2.0, -static_cast<double>(max_bias) / static_cast<double>(log2));
  const double m1 = std::pow(2.0, -static_cast<double>(max_bias) / 2.0 / static_cast<double>(log2));
  return h < log2 ? std::pow(m0, static_cast<double>(h + 1))
                  : std::pow(m1, static_cast<double>((2 * (h - log2)) + 1));
}

// A row's softmax with an optional sink in the denominator; a row with
// nothing finite (and no sink) is zero.
void SoftmaxRow(std::span<double> s, std::optional<double> sink) {
  double max = -std::numeric_limits<double>::infinity();
  for (const double v : s) {
    max = std::max(max, v);
  }
  if (sink) {
    max = std::max(max, *sink);
  }
  if (!std::isfinite(max)) {
    std::ranges::fill(s, 0.0);
    return;
  }
  double sum = sink ? std::exp(*sink - max) : 0.0;
  for (double& v : s) {
    v = std::exp(v - max);
    sum += v;
  }
  for (double& v : s) {
    v = sum > 0.0 ? v / sum : 0.0;
  }
}

// Order of a row's values, highest (or lowest) first, the lower index first
// among equals, NaN last.
std::vector<std::size_t> Order(std::span<const double> row, bool descending) {
  std::vector<std::size_t> idx(row.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::ranges::stable_sort(idx, [&](std::size_t a, std::size_t b) {
    const double x = row[a];
    const double y = row[b];
    if (std::isnan(x) || std::isnan(y)) {
      return !std::isnan(x) && std::isnan(y);
    }
    return descending ? x > y : x < y;
  });
  return idx;
}

// A value rounded to F32 (round to nearest even), a step of op.h's F32
// rotation angle.
float RoundF32(double v) { return static_cast<float>(v); }

// GGML's YaRN correction dims (ggml.c ggml_rope_yarn_corr_dims), in float.
std::array<float, 2> CorrDims(std::uint32_t dims, std::uint32_t ctx, float base, float fast,
                              float slow) {
  const auto dim = [&](float rot) {
    return static_cast<float>(dims) *
           std::log(static_cast<float>(ctx) / (rot * 2.0f * std::numbers::pi_v<float>)) /
           (2.0f * std::log(base));
  };
  const float start = std::floor(dim(fast));
  const float end = std::ceil(dim(slow));
  return {std::max(0.0f, start), std::min(static_cast<float>(dims) - 1.0f, end)};
}

}  // namespace

// ---------------------------------------------------------------- lifecycle

std::expected<Executor, ExecFailure> Executor::Create(const ir::Graph& graph) {
  if (auto v = ir::Verify(graph); !v) {
    return Fail(ExecError::kGraph,
                std::format("{}: {}", ir::GraphErrorName(v.error().error), v.error().detail),
                v.error().node);
  }
  Executor e;
  e.source_ = &graph;
  auto x = ir::ExpandWithMap(graph, false);
  if (!x) {
    return Fail(
        ExecError::kGraph,
        std::format("decomposing: {}: {}", ir::GraphErrorName(x.error().error), x.error().detail),
        x.error().node);
  }
  e.map_ = std::move(x->map);
  e.expanded_.emplace(std::move(x->graph));
  e.roots_.resize(e.expanded_->tensor_count());
  return e;
}

std::expected<TensorId, ExecFailure> Executor::Mapped(TensorId id) const {
  if (!id.valid() || id.v >= map_.size() || !map_.at(id.v).valid()) {
    return Fail(ExecError::kBinding, std::format("tensor {} is not the graph's", id.v));
  }
  return map_.at(id.v);
}

Executor::Root& Executor::RootOf(TensorId id) { return roots_.at(graph().root(id).v); }
const Executor::Root& Executor::RootOf(TensorId id) const { return roots_.at(graph().root(id).v); }

std::expected<void, ExecFailure> Executor::Bind(TensorId leaf, std::span<const PartView> parts) {
  auto m = Mapped(leaf);
  if (!m) {
    return std::unexpected(m.error());
  }
  if (!IsLeaf(graph().node(graph().producer(*m)).key.op)) {
    return Fail(ExecError::kBinding, std::format("tensor {} is not a leaf", leaf.v));
  }
  const TensorDesc& t = graph().tensor(*m);
  auto list = format::PartsOf(t.format);
  if (!list || list->size() != parts.size()) {
    return Fail(ExecError::kBinding,
                std::format("tensor {} takes {} parts", leaf.v, list ? list->size() : 0));
  }
  Root& r = roots_.at(m->v);
  r.parts.clear();
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const format::PartExtent& want = (*list)[i];
    if (parts[i].role != want.role || parts[i].bytes.size() < want.bytes) {
      return Fail(ExecError::kBinding,
                  std::format("tensor {}'s part {} is {} of {} bytes, not {} of {}", leaf.v, i,
                              format::PartRoleName(parts[i].role), parts[i].bytes.size(),
                              format::PartRoleName(want.role), want.bytes));
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(want.readable));
    std::ranges::copy(parts[i].bytes.first(static_cast<std::size_t>(want.bytes)), bytes.begin());
    r.parts.push_back(std::move(bytes));
  }
  r.bound = true;
  return {};
}

std::expected<void, ExecFailure> Executor::BindValues(TensorId leaf,
                                                      std::span<const float> values) {
  auto m = Mapped(leaf);
  if (!m) {
    return std::unexpected(m.error());
  }
  if (!IsLeaf(graph().node(graph().producer(*m)).key.op)) {
    return Fail(ExecError::kBinding, std::format("tensor {} is not a leaf", leaf.v));
  }
  const TensorDesc& t = graph().tensor(*m);
  auto list = format::PartsOf(t.format);
  if (!list || list->size() != 1 || ir::UnitBytes(t.format) == 0) {
    return Fail(ExecError::kBinding, std::format("tensor {} is not one part", leaf.v));
  }
  if (values.size() != ir::Elements(t.shape())) {
    return Fail(ExecError::kBinding, std::format("tensor {} has {} values, not {}", leaf.v,
                                                 ir::Elements(t.shape()), values.size()));
  }
  Root& r = roots_.at(m->v);
  r.parts.assign(1, std::vector<std::byte>(static_cast<std::size_t>((*list)[0].readable)));
  r.bound = true;
  Vec v(values.begin(), values.end());
  return Store(*m, v);
}

std::expected<void, ExecFailure> Executor::BindInts(TensorId leaf,
                                                    std::span<const std::int64_t> values) {
  auto m = Mapped(leaf);
  if (!m) {
    return std::unexpected(m.error());
  }
  if (!IsLeaf(graph().node(graph().producer(*m)).key.op)) {
    return Fail(ExecError::kBinding, std::format("tensor {} is not a leaf", leaf.v));
  }
  const TensorDesc& t = graph().tensor(*m);
  if (t.format.layout != format::Layout::kDense || values.size() != ir::Elements(t.shape())) {
    return Fail(ExecError::kBinding,
                std::format("tensor {} takes {} dense values", leaf.v, ir::Elements(t.shape())));
  }
  auto list = format::PartsOf(t.format);
  if (!list) {
    return Fail(ExecError::kBinding, "no size");
  }
  Root& r = roots_.at(m->v);
  r.parts.assign(1, std::vector<std::byte>(static_cast<std::size_t>((*list)[0].readable)));
  r.bound = true;
  const std::uint64_t unit = ir::UnitBytes(t.format);
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::byte* p = r.parts[0].data() + (i * unit);
    switch (t.format.encoding) {
      case Encoding::kI32:
        Put(p, static_cast<std::int32_t>(values[i]));
        break;
      case Encoding::kI64:
        Put(p, values[i]);
        break;
      case Encoding::kU64:
        Put(p, static_cast<std::uint64_t>(values[i]));
        break;
      default:
        if (!WriteElement(t.format.encoding, static_cast<double>(values[i]), p)) {
          return Fail(ExecError::kUnsupported, "an integer leaf of another encoding");
        }
    }
  }
  return {};
}

std::expected<void, ExecFailure> Executor::Prepare() {
  const ir::Graph& g = graph();
  for (std::size_t n = 0; n < g.nodes().size(); ++n) {
    const ir::Node& node = g.nodes()[n];
    for (const TensorId o : node.outputs()) {
      const TensorDesc& t = g.tensor(o);
      if (t.view_of.valid()) {
        continue;
      }
      Root& r = roots_.at(o.v);
      if (IsLeaf(node.key.op)) {
        if (!r.bound) {
          return Fail(ExecError::kUnbound, std::format("leaf {} ({}) has no bytes", o.v, g.name(o)),
                      static_cast<std::uint32_t>(n));
        }
        continue;
      }
      auto list = format::PartsOf(t.format);
      if (!list) {
        return Fail(ExecError::kUnsupported, "a result without a size",
                    static_cast<std::uint32_t>(n));
      }
      r.parts.resize(list->size());
      for (std::size_t i = 0; i < list->size(); ++i) {
        r.parts[i].assign(static_cast<std::size_t>((*list)[i].readable), std::byte{0});
      }
    }
  }
  return {};
}

std::expected<void, ExecFailure> Executor::Run() {
  Executor* self = this;
  return RunDevices(std::span(&self, 1), {});
}

std::expected<std::size_t, ExecFailure> Executor::Advance() {
  const auto nodes = graph().nodes();
  while (next_ < nodes.size()) {
    const ir::Node& node = nodes[next_];
    if (IsCollective(node.key.op)) {
      return next_;
    }
    if (auto r = Exec(node); !r) {
      ExecFailure f = r.error();
      f.node = static_cast<std::uint32_t>(next_);
      f.detail = std::format("{}: {}", ir::OpKeyName(node.key), f.detail);
      return std::unexpected(f);
    }
    ++next_;
  }
  return nodes.size();
}

std::expected<std::vector<float>, ExecFailure> Executor::Values(TensorId id) const {
  auto m = Mapped(id);
  if (!m) {
    return std::unexpected(m.error());
  }
  auto v = Load(*m);
  if (!v) {
    return std::unexpected(v.error());
  }
  std::vector<float> out(v->size());
  for (std::size_t i = 0; i < v->size(); ++i) {
    out[i] = static_cast<float>((*v)[i]);
  }
  return out;
}

std::expected<std::vector<std::int64_t>, ExecFailure> Executor::Ints(TensorId id) const {
  auto m = Mapped(id);
  if (!m) {
    return std::unexpected(m.error());
  }
  return LoadInts(*m);
}

std::expected<std::vector<std::int64_t>, ExecFailure> Executor::LoadInts(TensorId id) const {
  const TensorDesc& t = graph().tensor(id);
  const Root& r = RootOf(id);
  if (t.format.layout != format::Layout::kDense || r.parts.empty()) {
    return Fail(ExecError::kBinding, "not a dense tensor with bytes");
  }
  const Ne& ne = t.shape().ne;
  std::vector<std::int64_t> out;
  out.reserve(Count(ne));
  for (std::uint64_t i3 = 0; i3 < ne[3]; ++i3) {
    for (std::uint64_t i2 = 0; i2 < ne[2]; ++i2) {
      for (std::uint64_t i1 = 0; i1 < ne[1]; ++i1) {
        for (std::uint64_t i0 = 0; i0 < ne[0]; ++i0) {
          const std::uint64_t off =
              t.view_offset + (i0 * t.nb[0]) + (i1 * t.nb[1]) + (i2 * t.nb[2]) + (i3 * t.nb[3]);
          const std::byte* p = r.parts[0].data() + off;
          switch (t.format.encoding) {
            case Encoding::kI64:
              out.push_back(Raw<std::int64_t>(p));
              break;
            case Encoding::kU64:
              out.push_back(static_cast<std::int64_t>(Raw<std::uint64_t>(p)));
              break;
            default:
              out.push_back(static_cast<std::int64_t>(ReadElement(t.format.encoding, p)));
          }
        }
      }
    }
  }
  return out;
}

std::span<const std::byte> Executor::Bytes(TensorId id, format::PartRole role) const {
  auto m = Mapped(id);
  if (!m) {
    return {};
  }
  const TensorDesc& t = graph().tensor(graph().root(*m));
  auto list = format::PartsOf(t.format);
  const Root& r = RootOf(*m);
  if (!list) {
    return {};
  }
  for (std::size_t i = 0; i < list->size() && i < r.parts.size(); ++i) {
    if ((*list)[i].role == role) {
      return std::span(r.parts[i]).first(static_cast<std::size_t>((*list)[i].bytes));
    }
  }
  return {};
}

// ---------------------------------------------------------------- values

std::expected<Vec, ExecFailure> Executor::Load(TensorId id,
                                               std::optional<std::uint64_t> matrix) const {
  const TensorDesc& t = graph().tensor(id);
  const Root& r = RootOf(id);
  if (r.parts.empty()) {
    return Fail(ExecError::kUnbound, std::format("tensor {} has no bytes", id.v));
  }
  const Ne& ne = t.shape().ne;
  // One matrix (i2, of a tensor of one i3) or every one.
  if (matrix && (ne[3] != 1 || *matrix >= ne[2])) {
    return Fail(ExecError::kIndex, std::format("matrix {} of tensor {}", *matrix, id.v));
  }
  const std::uint64_t m0 = matrix.value_or(0);
  const std::uint64_t m1 = matrix ? *matrix + 1 : ne[2];
  Vec out;
  out.reserve(matrix ? ne[0] * ne[1] : Count(ne));
  const std::uint64_t unit = ir::UnitBytes(t.format);
  if (unit == 0) {
    // A format in parts: a root, decoded whole.
    auto list = format::PartsOf(t.format);
    if (!list) {
      return Fail(ExecError::kUnsupported, "a format without a size");
    }
    std::vector<PartView> views;
    for (std::size_t i = 0; i < list->size(); ++i) {
      views.push_back({.role = (*list)[i].role, .bytes = r.parts.at(i)});
    }
    const std::uint64_t rows = ne[1];
    std::vector<float> f(static_cast<std::size_t>(ne[0] * rows));
    const std::uint64_t end = matrix ? m0 + 1 : format::Matrices(t.format);
    for (std::uint64_t m = m0; m < end; ++m) {
      if (auto d = DecodeRows(t.format, views, m, 0, rows, f); !d) {
        return Fail(ExecError::kUnsupported,
                    std::format("decoding {}: {}", format::Info(t.format.encoding).name,
                                DecodeErrorName(d.error())));
      }
      out.insert(out.end(), f.begin(), f.end());
    }
    return out;
  }
  const std::byte* data = r.parts.at(0).data();
  if (t.format.layout == format::Layout::kDense) {
    const Encoding e = t.format.encoding;
    for (std::uint64_t i3 = 0; i3 < ne[3]; ++i3) {
      for (std::uint64_t i2 = m0; i2 < m1; ++i2) {
        for (std::uint64_t i1 = 0; i1 < ne[1]; ++i1) {
          const std::uint64_t row =
              t.view_offset + (i1 * t.nb[1]) + (i2 * t.nb[2]) + (i3 * t.nb[3]);
          for (std::uint64_t i0 = 0; i0 < ne[0]; ++i0) {
            out.push_back(ReadElement(e, data + row + (i0 * t.nb[0])));
          }
        }
      }
    }
    return out;
  }
  // A block layout, row by row.
  format::TensorFormat rowf = t.format;
  rowf.shape = ir::ShapeOf({ne[0]});
  rowf.stacked = 1;
  const std::uint64_t row_bytes = (ne[0] / ir::UnitElements(t.format)) * unit;
  std::vector<float> f(static_cast<std::size_t>(ne[0]));
  for (std::uint64_t i3 = 0; i3 < ne[3]; ++i3) {
    for (std::uint64_t i2 = m0; i2 < m1; ++i2) {
      for (std::uint64_t i1 = 0; i1 < ne[1]; ++i1) {
        const std::uint64_t at = t.view_offset + (i1 * t.nb[1]) + (i2 * t.nb[2]) + (i3 * t.nb[3]);
        const PartView part{.role = format::PartRole::kData,
                            .bytes = std::span(data + at, static_cast<std::size_t>(row_bytes))};
        if (auto d = DecodeRows(rowf, std::span(&part, 1), 0, 0, 1, f); !d) {
          return Fail(ExecError::kUnsupported,
                      std::format("decoding {}: {}", format::Info(t.format.encoding).name,
                                  DecodeErrorName(d.error())));
        }
        out.insert(out.end(), f.begin(), f.end());
      }
    }
  }
  return out;
}

std::expected<void, ExecFailure> Executor::Store(TensorId id, std::span<const double> values) {
  const TensorDesc& t = graph().tensor(id);
  Root& r = RootOf(id);
  const Ne& ne = t.shape().ne;
  const std::uint64_t unit = ir::UnitBytes(t.format);
  if (unit == 0 || r.parts.empty()) {
    return Fail(ExecError::kUnsupported, std::format("tensor {} cannot be written", id.v));
  }
  if (values.size() != Count(ne)) {
    return Fail(ExecError::kBinding,
                std::format("{} values for tensor {} of {}", values.size(), id.v, Count(ne)));
  }
  std::byte* data = r.parts[0].data();
  std::size_t k = 0;
  if (t.format.layout == format::Layout::kDense) {
    const Encoding e = t.format.encoding;
    for (std::uint64_t i3 = 0; i3 < ne[3]; ++i3) {
      for (std::uint64_t i2 = 0; i2 < ne[2]; ++i2) {
        for (std::uint64_t i1 = 0; i1 < ne[1]; ++i1) {
          const std::uint64_t row =
              t.view_offset + (i1 * t.nb[1]) + (i2 * t.nb[2]) + (i3 * t.nb[3]);
          for (std::uint64_t i0 = 0; i0 < ne[0]; ++i0) {
            if (!WriteElement(e, values[k++], data + row + (i0 * t.nb[0]))) {
              return Fail(ExecError::kUnsupported, std::format("writing {}", format::Info(e).name));
            }
          }
        }
      }
    }
    return {};
  }
  format::TensorFormat rowf = t.format;
  rowf.shape = ir::ShapeOf({ne[0]});
  rowf.stacked = 1;
  const std::uint64_t row_bytes = (ne[0] / ir::UnitElements(t.format)) * unit;
  std::vector<float> f(static_cast<std::size_t>(ne[0]));
  for (std::uint64_t i3 = 0; i3 < ne[3]; ++i3) {
    for (std::uint64_t i2 = 0; i2 < ne[2]; ++i2) {
      for (std::uint64_t i1 = 0; i1 < ne[1]; ++i1) {
        const std::uint64_t at = t.view_offset + (i1 * t.nb[1]) + (i2 * t.nb[2]) + (i3 * t.nb[3]);
        for (float& v : f) {
          v = static_cast<float>(values[k++]);
        }
        if (auto e = EncodeRows(rowf, 0, 0, f,
                                std::span(data + at, static_cast<std::size_t>(row_bytes)));
            !e) {
          return Fail(ExecError::kUnsupported,
                      std::format("encoding {}: {}", format::Info(t.format.encoding).name,
                                  DecodeErrorName(e.error())));
        }
      }
    }
  }
  return {};
}

std::expected<void, ExecFailure> Executor::StoreRow(TensorId id, std::uint64_t i1, std::uint64_t i2,
                                                    std::uint64_t i3, std::span<const double> row) {
  const TensorDesc& t = graph().tensor(id);
  Root& r = RootOf(id);
  const std::uint64_t unit = ir::UnitBytes(t.format);
  if (unit == 0 || r.parts.empty() || row.size() != t.ne(0)) {
    return Fail(ExecError::kUnsupported, std::format("a row of tensor {} cannot be written", id.v));
  }
  std::byte* at =
      r.parts[0].data() + t.view_offset + (i1 * t.nb[1]) + (i2 * t.nb[2]) + (i3 * t.nb[3]);
  if (t.format.layout == format::Layout::kDense) {
    for (std::size_t i = 0; i < row.size(); ++i) {
      if (!WriteElement(t.format.encoding, row[i], at + (i * t.nb[0]))) {
        return Fail(ExecError::kUnsupported,
                    std::format("writing {}", format::Info(t.format.encoding).name));
      }
    }
    return {};
  }
  format::TensorFormat rowf = t.format;
  rowf.shape = ir::ShapeOf({t.ne(0)});
  rowf.stacked = 1;
  std::vector<float> f(row.size());
  for (std::size_t i = 0; i < row.size(); ++i) {
    f[i] = static_cast<float>(row[i]);
  }
  const std::uint64_t bytes = (t.ne(0) / ir::UnitElements(t.format)) * unit;
  if (auto e = EncodeRows(rowf, 0, 0, f, std::span(at, static_cast<std::size_t>(bytes))); !e) {
    return Fail(ExecError::kUnsupported,
                std::format("encoding {}: {}", format::Info(t.format.encoding).name,
                            DecodeErrorName(e.error())));
  }
  return {};
}

// ---------------------------------------------------------------- ops

std::expected<void, ExecFailure> Executor::Exec(const ir::Node& node) {
  const ir::Graph& g = graph();
  const auto ne = [&](std::size_t i) -> const Ne& { return g.tensor(node.src.at(i)).shape().ne; };
  const auto has = [&](std::size_t i) { return node.source(i).valid(); };
  const auto out_ne = [&](std::size_t i = 0) -> const Ne& {
    return g.tensor(node.out.at(i)).shape().ne;
  };
  const TensorId out = node.nout > 0 ? node.out[0] : TensorId{};
  switch (node.key.op) {
    case Op::kInput:
    case Op::kWeight:
    case Op::kState:
    case Op::kView:
    case Op::kReshape:
    case Op::kPermute:
      return {};

    case Op::kCont: {
      const TensorDesc& src_desc = g.tensor(node.src[0]);
      const TensorDesc& dst_desc = g.tensor(out);
      const Root& src_root = RootOf(node.src[0]);
      Root& dst_root = RootOf(out);
      if (src_desc.format.layout == format::Layout::kDense &&
          dst_desc.format.layout == format::Layout::kDense &&
          src_desc.format.encoding == dst_desc.format.encoding) {
        if (src_root.parts.empty()) {
          return Fail(ExecError::kUnbound, std::format("tensor {} has no bytes", node.src[0].v));
        }
        const std::byte* src_data = src_root.parts[0].data();
        std::byte* dst_data = dst_root.parts[0].data() + dst_desc.view_offset;
        const std::uint64_t unit = dst_desc.nb[0];
        const Ne& an = ne(0);
        std::uint64_t k = 0;
        for (std::uint64_t i3 = 0; i3 < an[3]; ++i3) {
          for (std::uint64_t i2 = 0; i2 < an[2]; ++i2) {
            for (std::uint64_t i1 = 0; i1 < an[1]; ++i1) {
              for (std::uint64_t i0 = 0; i0 < an[0]; ++i0) {
                const std::uint64_t s_off = src_desc.view_offset + (i0 * src_desc.nb[0]) +
                                            (i1 * src_desc.nb[1]) + (i2 * src_desc.nb[2]) +
                                            (i3 * src_desc.nb[3]);
                std::memcpy(dst_data + (k++ * unit), src_data + s_off,
                            static_cast<std::size_t>(unit));
              }
            }
          }
        }
        return {};
      }
      LLMP_LOAD(a, node.src[0]);
      return Store(out, a);
    }
    case Op::kConvert:
    case Op::kPrepare: {
      LLMP_LOAD(a, node.src[0]);
      return Store(out, a);
    }
    case Op::kConcat: {
      const TensorDesc& a_desc = g.tensor(node.src[0]);
      const TensorDesc& b_desc = g.tensor(node.src[1]);
      const TensorDesc& dst_desc = g.tensor(out);
      const Root& a_root = RootOf(node.src[0]);
      const Root& b_root = RootOf(node.src[1]);
      Root& dst_root = RootOf(out);
      const auto dim = node.Params<ir::ConcatParams>().dim;
      const Ne& an = ne(0);
      const Ne& bn = ne(1);
      const Ne& on = out_ne();
      if (a_desc.format.layout == format::Layout::kDense &&
          b_desc.format.layout == format::Layout::kDense &&
          dst_desc.format.layout == format::Layout::kDense &&
          a_desc.format.encoding == dst_desc.format.encoding &&
          b_desc.format.encoding == dst_desc.format.encoding) {
        if (a_root.parts.empty()) {
          return Fail(ExecError::kUnbound, std::format("tensor {} has no bytes", node.src[0].v));
        }
        if (b_root.parts.empty()) {
          return Fail(ExecError::kUnbound, std::format("tensor {} has no bytes", node.src[1].v));
        }
        const std::byte* a_data = a_root.parts[0].data();
        const std::byte* b_data = b_root.parts[0].data();
        std::byte* dst_data = dst_root.parts[0].data();
        const std::uint64_t unit = dst_desc.nb[0];
        for (std::uint64_t i3 = 0; i3 < on[3]; ++i3) {
          for (std::uint64_t i2 = 0; i2 < on[2]; ++i2) {
            for (std::uint64_t i1 = 0; i1 < on[1]; ++i1) {
              for (std::uint64_t i0 = 0; i0 < on[0]; ++i0) {
                std::array<std::uint64_t, 4> i{i0, i1, i2, i3};
                const bool first = i.at(dim) < an.at(dim);
                if (!first) {
                  i.at(dim) -= an.at(dim);
                }
                const TensorDesc& s_desc = first ? a_desc : b_desc;
                const std::byte* s_data = first ? a_data : b_data;
                const std::uint64_t s_off = s_desc.view_offset + (i[0] * s_desc.nb[0]) +
                                            (i[1] * s_desc.nb[1]) + (i[2] * s_desc.nb[2]) +
                                            (i[3] * s_desc.nb[3]);
                const std::uint64_t d_off = dst_desc.view_offset + (i0 * dst_desc.nb[0]) +
                                            (i1 * dst_desc.nb[1]) + (i2 * dst_desc.nb[2]) +
                                            (i3 * dst_desc.nb[3]);
                std::memcpy(dst_data + d_off, s_data + s_off, static_cast<std::size_t>(unit));
              }
            }
          }
        }
        return {};
      }
      LLMP_LOAD(a, node.src[0]);
      LLMP_LOAD(b, node.src[1]);
      Vec r(Count(on));
      for (std::uint64_t i3 = 0; i3 < on[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < on[2]; ++i2) {
          for (std::uint64_t i1 = 0; i1 < on[1]; ++i1) {
            for (std::uint64_t i0 = 0; i0 < on[0]; ++i0) {
              std::array<std::uint64_t, 4> i{i0, i1, i2, i3};
              const bool first = i.at(dim) < an.at(dim);
              if (!first) {
                i.at(dim) -= an.at(dim);
              }
              r[Ix(on, i0, i1, i2, i3)] =
                  first ? a[Ix(an, i[0], i[1], i[2], i[3])] : b[Ix(bn, i[0], i[1], i[2], i[3])];
            }
          }
        }
      }
      return Store(out, r);
    }
    case Op::kRepeat: {
      LLMP_LOAD(a, node.src[0]);
      const Ne& an = ne(0);
      const Ne& on = out_ne();
      Vec r(Count(on));
      for (std::uint64_t i3 = 0; i3 < on[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < on[2]; ++i2) {
          for (std::uint64_t i1 = 0; i1 < on[1]; ++i1) {
            for (std::uint64_t i0 = 0; i0 < on[0]; ++i0) {
              r[Ix(on, i0, i1, i2, i3)] = a[Ix(an, i0 % an[0], i1 % an[1], i2 % an[2], i3 % an[3])];
            }
          }
        }
      }
      return Store(out, r);
    }
    case Op::kGetRows: {
      LLMP_LOAD(a, node.src[0]);
      LLMP_LOAD(ids, node.src[1]);
      const Ne& an = ne(0);
      const Ne& in = ne(1);
      const Ne& on = out_ne();
      Vec r(Count(on));
      for (std::uint64_t k = 0; k < in[2]; ++k) {
        for (std::uint64_t j = 0; j < in[1]; ++j) {
          for (std::uint64_t i = 0; i < in[0]; ++i) {
            const double id = ids[Ix(in, i, j, k, 0)];
            if (!(id >= 0.0 && id < static_cast<double>(an[1]))) {
              return Fail(ExecError::kIndex, std::format("row {} of {}", id, an[1]));
            }
            const auto row = static_cast<std::uint64_t>(id);
            for (std::uint64_t c = 0; c < an[0]; ++c) {
              r[Ix(on, c, i, j, k)] = a[Ix(an, c, row, j, k)];
            }
          }
        }
      }
      return Store(out, r);
    }

    case Op::kMatMul: {
      LLMP_LOAD(a, node.src[0]);
      LLMP_LOAD(b, node.src[1]);
      const Ne& an = ne(0);
      const Ne& bn = ne(1);
      const Ne& on = out_ne();
      const std::uint64_t r2 = bn[2] / an[2];
      const std::uint64_t r3 = bn[3] / an[3];
      Vec r(Count(on));
      for (std::uint64_t i3 = 0; i3 < bn[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < bn[2]; ++i2) {
          for (std::uint64_t m = 0; m < bn[1]; ++m) {
            const double* bx = &b[Ix(bn, 0, m, i2, i3)];
            for (std::uint64_t n = 0; n < an[1]; ++n) {
              const double* ax = &a[Ix(an, 0, n, i2 / r2, i3 / r3)];
              double s = 0.0;
              for (std::uint64_t k = 0; k < an[0]; ++k) {
                s += ax[k] * bx[k];
              }
              r[Ix(on, n, m, i2, i3)] = s;
            }
          }
        }
      }
      return Store(out, r);
    }
    case Op::kMatMulId: {
      LLMP_LOAD(b, node.src[1]);
      LLMP_LOAD(ids, node.src[2]);
      const Ne& an = ne(0);
      const Ne& bn = ne(1);
      const Ne& in = ne(2);
      const Ne& on = out_ne();
      // Only the selected matrices are decoded (an expert array's few).
      std::map<std::uint64_t, Vec> matrices;
      Vec r(Count(on));
      for (std::uint64_t t = 0; t < in[1]; ++t) {
        for (std::uint64_t u = 0; u < in[0]; ++u) {
          const double id = ids[Ix(in, u, t, 0, 0)];
          if (!(id >= 0.0 && id < static_cast<double>(an[2]))) {
            return Fail(ExecError::kIndex, std::format("expert {} of {}", id, an[2]));
          }
          const auto e = static_cast<std::uint64_t>(id);
          auto held = matrices.find(e);
          if (held == matrices.end()) {
            auto one = Load(node.src[0], e);
            if (!one) return std::unexpected(one.error());
            held = matrices.emplace(e, std::move(*one)).first;
          }
          const Vec& as = held->second;
          const double* bx = &b[Ix(bn, 0, u % bn[1], t, 0)];
          for (std::uint64_t n = 0; n < an[1]; ++n) {
            const double* ax = &as[Ix(an, 0, n, 0, 0)];
            double s = 0.0;
            for (std::uint64_t k = 0; k < an[0]; ++k) {
              s += ax[k] * bx[k];
            }
            r[Ix(on, n, u, t, 0)] = s;
          }
        }
      }
      return Store(out, r);
    }
    case Op::kExpertSort: {
      LLMP_LOAD(ids, node.src[0]);
      const auto experts = node.Params<ir::ExpertSortParams>().experts;
      std::vector<std::vector<std::uint64_t>> by(experts);
      for (std::size_t i = 0; i < ids.size(); ++i) {
        const double id = ids[i];
        if (!(id >= 0.0 && id < static_cast<double>(experts))) {
          return Fail(ExecError::kIndex, std::format("expert {} of {}", id, experts));
        }
        by[static_cast<std::size_t>(id)].push_back(i);  // i = t · U + u
      }
      Vec order;
      Vec offsets;
      for (const auto& list : by) {
        offsets.push_back(static_cast<double>(order.size()));
        for (const std::uint64_t i : list) {
          order.push_back(static_cast<double>(i));
        }
      }
      offsets.push_back(static_cast<double>(order.size()));
      LLMP_TRY(Store(node.out[0], order));
      return Store(node.out[1], offsets);
    }

    case Op::kAttention: {
      const auto p = node.Params<ir::AttnParams>();
      LLMP_LOAD(q, node.src[0]);
      LLMP_LOAD(k, node.src[1]);
      LLMP_LOAD(v, node.src[2]);
      Vec mask;
      Vec sinks;
      if (has(3)) {
        LLMP_LOAD(m, node.src[3]);
        mask = m;
      }
      if (has(4)) {
        LLMP_LOAD(s, node.src[4]);
        sinks = s;
      }
      const Ne& qn = ne(0);
      const Ne& kn = ne(1);
      const Ne& vn = ne(2);
      const Ne mn = has(3) ? ne(3) : Ne{1, 1, 1, 1};
      const Ne& on = out_ne();
      const std::uint64_t heads = qn[2];
      const std::uint64_t group = heads / kn[2];
      Vec r(Count(on));
      std::vector<double> s(static_cast<std::size_t>(kn[1]));
      for (std::uint64_t b = 0; b < qn[3]; ++b) {
        for (std::uint64_t h = 0; h < heads; ++h) {
          const std::uint64_t kh = h / group;
          const double slope = Slope(p.max_bias, h, heads);
          for (std::uint64_t iq = 0; iq < qn[1]; ++iq) {
            for (std::uint64_t c = 0; c < kn[1]; ++c) {
              double dot = 0.0;
              for (std::uint64_t d = 0; d < qn[0]; ++d) {
                dot += q[Ix(qn, d, iq, h, b)] * k[Ix(kn, d, c, kh, b)];
              }
              double score = dot * static_cast<double>(p.scale);
              if (p.softcap > 0.0f) {
                score = static_cast<double>(p.softcap) * std::tanh(score / p.softcap);
              }
              if (!mask.empty()) {
                score += slope * mask[Ix(mn, c, iq, h % mn[2], b % mn[3])];
              }
              s[c] = score;
            }
            std::optional<double> sink;
            if (!sinks.empty()) {
              sink = sinks[h];
            }
            SoftmaxRow(s, sink);
            for (std::uint64_t d = 0; d < vn[0]; ++d) {
              double acc = 0.0;
              for (std::uint64_t c = 0; c < kn[1]; ++c) {
                if (s[c] != 0.0) {
                  acc += s[c] * v[Ix(vn, d, c, kh, b)];
                }
              }
              r[Ix(on, d, h, iq, b)] = acc;
            }
          }
        }
      }
      return Store(out, r);
    }
    case Op::kMaskBuild: {
      const auto p = node.Params<ir::MaskParams>();
      LLMP_LOAD(qp, node.src[0]);
      LLMP_LOAD(cp, node.src[1]);
      const Ne& on = out_ne();
      const double inf = std::numeric_limits<double>::infinity();
      Vec r(Count(on), -inf);
      for (std::uint64_t row = 0; row < qp.size(); ++row) {
        const auto pos = static_cast<std::int64_t>(qp[row]);
        for (std::uint64_t c = 0; c < cp.size(); ++c) {
          const auto cell = static_cast<std::int64_t>(cp[c]);
          bool seen = cell >= 0 && cell <= pos;
          if (p.kind == ir::MaskKind::kWindow) {
            seen = seen && pos - cell < static_cast<std::int64_t>(p.window);
          } else if (p.kind == ir::MaskKind::kBlock) {
            const auto block = static_cast<std::int64_t>(p.block);
            seen = cell >= 0 && pos >= 0 && cell < ((pos / block) + 1) * block;
          } else if (p.kind == ir::MaskKind::kAhead) {
            seen = cell >= 0 && (cell > pos || p.window == 0 ||
                                 pos - cell < static_cast<std::int64_t>(p.window));
          }
          r[Ix(on, c, row, 0, 0)] = seen ? 0.0 : -inf;
        }
      }
      return Store(out, r);
    }
    case Op::kCacheStore: {
      // Only the stored rows are written; the cache's other bytes stay.
      LLMP_LOAD(src, node.src[1]);
      LLMP_LOAD(idx, node.src[2]);
      const Ne& cn = out_ne();
      const Ne& sn = ne(1);
      const Ne& in = ne(2);
      for (std::uint64_t i3 = 0; i3 < sn[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < sn[2]; ++i2) {
          for (std::uint64_t i1 = 0; i1 < sn[1]; ++i1) {
            const double at = idx[Ix(in, i1, i2 % in[1], i3 % in[2], 0)];
            if (!(at >= 0.0 && at < static_cast<double>(cn[1]))) {
              return Fail(ExecError::kIndex, std::format("cell {} of {}", at, cn[1]));
            }
            const std::span<const double> row(&src[Ix(sn, 0, i1, i2, i3)],
                                              static_cast<std::size_t>(sn[0]));
            LLMP_TRY(StoreRow(out, static_cast<std::uint64_t>(at), i2, i3, row));
          }
        }
      }
      return {};
    }
    case Op::kRoundedIndexScore: {
      LLMP_LOAD(source_q, node.src[0]);
      Vec q = source_q;
      LLMP_LOAD(k, node.src[1]);
      LLMP_LOAD(w, node.src[2]);
      const auto counts = LoadInts(node.src[3]);
      if (!counts) return std::unexpected(counts.error());
      const Ne& qn = ne(0);
      const Ne& kn = ne(1);
      const Ne& wn = ne(2);
      Vec result(Count(out_ne()), -std::numeric_limits<double>::infinity());
      // Independent wider formula, not the graph decomposition or a model of
      // one device's MMA schedule. Half rounding is an explicit input rule.
      for (double& value : q) value = F16ToF32(F32ToF16(static_cast<float>(value)));
      for (std::uint64_t row = 0; row < qn[2]; ++row) {
        const auto visible = static_cast<std::uint64_t>(
            std::clamp((*counts)[row], std::int64_t{0}, static_cast<std::int64_t>(kn[2])));
        for (std::uint64_t cell = 0; cell < visible; ++cell) {
          double score = 0;
          for (std::uint64_t head = 0; head < qn[1]; ++head) {
            double dot = 0;
            for (std::uint64_t channel = 0; channel < qn[0]; ++channel)
              dot += q[Ix(qn, channel, head, row, 0)] * k[Ix(kn, channel, 0, cell, 0)];
            score += (dot > 0 ? dot : 0.0) * w[Ix(wn, head, row, 0, 0)];
          }
          result[row * kn[2] + cell] = score;
        }
      }
      return Store(out, result);
    }
    case Op::kSelect: {
      const auto& yes = graph().tensor(node.src[1]);
      const auto& no = graph().tensor(node.src[2]);
      const auto& condition = graph().tensor(node.src[0]);
      const auto& dest = graph().tensor(out);
      const auto& predicate_bytes = RootOf(node.src[0]).parts[0];
      auto& target = RootOf(out).parts[0];
      const auto unit = dest.nb[0];
      const auto offset = [](const TensorDesc& d, const Ne& ix) {
        std::uint64_t pos = d.view_offset;
        for (std::size_t axis = 0; axis < 4; ++axis) pos += (ix[axis] % d.ne(axis)) * d.nb[axis];
        return pos;
      };
      for (std::uint64_t i = 0; i < Count(dest.shape().ne); ++i) {
        std::uint64_t remaining = i;
        Ne ix{};
        for (std::size_t axis = 0; axis < 4; ++axis) {
          ix[axis] = remaining % dest.ne(axis);
          remaining /= dest.ne(axis);
        }
        const bool choose = Raw<std::int32_t>(predicate_bytes.data() + offset(condition, ix)) != 0;
        const auto source_id = node.src[choose ? 1 : 2];
        const auto& source_desc = choose ? yes : no;
        const auto& source = RootOf(source_id).parts[0];
        std::memcpy(target.data() + offset(dest, ix), source.data() + offset(source_desc, ix),
                    unit);
      }
      return {};
    }
    case Op::kPrefixOverwrite: {
      const auto counts = LoadInts(node.src[1]);
      if (!counts) return std::unexpected(counts.error());
      const auto p = node.Params<ir::PrefixOverwriteParams>();
      const TensorDesc& values = graph().tensor(node.src[0]);
      const TensorDesc& dest = graph().tensor(out);
      const auto& source = RootOf(node.src[0]).parts[0];
      auto& target = RootOf(out).parts[0];
      for (std::uint64_t row = 0; row < values.ne(1); ++row) {
        const auto visible = static_cast<std::uint64_t>(
            std::clamp((*counts)[row], std::int64_t{0}, static_cast<std::int64_t>(values.ne(0))));
        for (std::uint64_t cell = 0; cell < values.ne(0); ++cell) {
          const auto bits = cell < visible
                                ? Raw<std::uint32_t>(source.data() + values.view_offset +
                                                     row * values.nb[1] + cell * values.nb[0])
                                : p.fill_bits;
          Put(target.data() + dest.view_offset + row * dest.nb[1] + cell * dest.nb[0], bits);
        }
      }
      return {};
    }
    case Op::kIndexScore: {
      LLMP_LOAD(q, node.src[0]);
      LLMP_LOAD(k, node.src[1]);
      LLMP_LOAD(w, node.src[2]);
      Vec mask;
      if (has(3)) {
        LLMP_LOAD(m, node.src[3]);
        mask = m;
      }
      const Ne& qn = ne(0);
      const Ne& kn = ne(1);
      const Ne& wn = ne(2);
      const Ne mn = has(3) ? ne(3) : Ne{1, 1, 1, 1};
      const Ne& on = out_ne();
      Vec r(Count(on));
      for (std::uint64_t s = 0; s < qn[3]; ++s) {
        for (std::uint64_t t = 0; t < qn[2]; ++t) {
          for (std::uint64_t c = 0; c < kn[2]; ++c) {
            double score = 0.0;
            for (std::uint64_t h = 0; h < qn[1]; ++h) {
              double dot = 0.0;
              for (std::uint64_t d = 0; d < qn[0]; ++d) {
                dot += q[Ix(qn, d, h, t, s)] * k[Ix(kn, d, 0, c, s)];
              }
              score += (dot > 0.0 ? dot : 0.0) * w[Ix(wn, h, t, 0, s)];  // NaN to 0, as relu
            }
            if (!mask.empty()) {
              score += mask[Ix(mn, c, t, 0, s % mn[3])];
            }
            r[Ix(on, c, t, 0, s)] = score;
          }
        }
      }
      return Store(out, r);
    }
    case Op::kPool: {
      const auto p = node.Params<ir::PoolParams>();
      LLMP_LOAD(x, node.src[0]);
      Vec scores;
      if (has(1)) {
        LLMP_LOAD(sc, node.src[1]);
        scores = sc;
      }
      const Ne& xn = ne(0);
      const Ne& on = out_ne();
      Vec r(Count(on));
      std::vector<double> w(p.ratio);
      for (std::uint64_t i3 = 0; i3 < xn[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < xn[2]; ++i2) {
          for (std::uint64_t n = 0; n < on[1]; ++n) {
            for (std::uint64_t c = 0; c < xn[0]; ++c) {
              for (std::uint64_t j = 0; j < p.ratio; ++j) {
                w[j] = scores.empty() ? 0.0 : scores[Ix(xn, c, (n * p.ratio) + j, i2, i3)];
              }
              if (p.kind == ir::PoolKind::kSoftmax) {
                SoftmaxRow(w, std::nullopt);
              }
              double sum = 0.0;
              for (std::uint64_t j = 0; j < p.ratio; ++j) {
                const double val = x[Ix(xn, c, (n * p.ratio) + j, i2, i3)];
                sum += p.kind == ir::PoolKind::kSoftmax ? w[j] * val : val;
              }
              r[Ix(on, c, n, i2, i3)] =
                  p.kind == ir::PoolKind::kMean ? sum / static_cast<double>(p.ratio) : sum;
            }
          }
        }
      }
      return Store(out, r);
    }

    case Op::kRmsNorm:
    case Op::kL2Norm:
    case Op::kLayerNorm: {
      LLMP_LOAD(x, node.src[0]);
      const double eps = node.Params<ir::NormParams>().eps;
      const Ne& xn = ne(0);
      Vec r(x.size());
      for (std::size_t row = 0; row < x.size() / xn[0]; ++row) {
        const std::span<const double> v(&x[row * xn[0]], static_cast<std::size_t>(xn[0]));
        const auto n = static_cast<double>(xn[0]);
        double mean = 0.0;
        if (node.key.op == Op::kLayerNorm) {
          for (const double e : v) {
            mean += e;
          }
          mean /= n;
        }
        double sq = 0.0;
        for (const double e : v) {
          sq += (e - mean) * (e - mean);
        }
        double scale = 0.0;
        if (node.key.op == Op::kL2Norm) {
          scale = 1.0 / std::max(std::sqrt(sq), eps);
        } else {
          scale = 1.0 / std::sqrt((sq / n) + eps);
        }
        for (std::size_t i = 0; i < v.size(); ++i) {
          r[(row * xn[0]) + i] = (v[i] - mean) * scale;
        }
      }
      return Store(out, r);
    }
    case Op::kChannelNorm: {
      LLMP_LOAD(x, node.src[0]);
      const double eps = node.Params<ir::NormParams>().eps;
      const Ne& xn = ne(0);
      Vec r(x.size());
      for (std::uint64_t i3 = 0; i3 < xn[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < xn[2]; ++i2) {
          for (std::uint64_t i0 = 0; i0 < xn[0]; ++i0) {
            double sq = 0.0;
            for (std::uint64_t c = 0; c < xn[1]; ++c) {
              const double e = x[Ix(xn, i0, c, i2, i3)];
              sq += e * e;
            }
            const double scale = 1.0 / std::max(std::sqrt(sq), eps);
            for (std::uint64_t c = 0; c < xn[1]; ++c) {
              r[Ix(xn, i0, c, i2, i3)] = x[Ix(xn, i0, c, i2, i3)] * scale;
            }
          }
        }
      }
      return Store(out, r);
    }
    case Op::kRope: {
      const auto p = node.Params<ir::RopeParams>();
      LLMP_LOAD(x, node.src[0]);
      LLMP_LOAD(pos, node.src[1]);
      Vec freq;
      if (has(2)) {
        LLMP_LOAD(f, node.src[2]);
        freq = f;
      }
      const Ne& xn = ne(0);
      const std::uint64_t t_count = xn[2];
      Vec r = x;
      // op.h's F32 angle: GGML's theta_scale = powf(base, -2.0f / dims),
      // each later step rounded to F32 as rope.cu forms it.
      const float theta_scale = RoundF32(std::pow(
          static_cast<double>(p.base), static_cast<double>(-2.0f / static_cast<float>(p.dims))));
      const bool llama3 = p.llama3;
      const std::array<float, 2> corr =
          (!llama3 && p.ext_factor > 0.0f)
              ? CorrDims(p.dims, p.original_context, p.base, p.beta_fast, p.beta_slow)
              : std::array<float, 2>{0.0f, 0.0f};
      const std::uint64_t pairs = p.rotated_dims != UINT32_MAX      ? p.rotated_dims / 2
                                  : p.mode == ir::RopeMode::kVision ? p.dims
                                                                    : p.dims / 2;
      const auto& sec = p.sections;
      const std::uint64_t sect_dims = std::uint64_t{sec[0]} + sec[1] + sec[2] + sec[3];
      for (std::uint64_t i3 = 0; i3 < xn[3]; ++i3) {
        for (std::uint64_t t = 0; t < t_count; ++t) {
          for (std::uint64_t h = 0; h < xn[1]; ++h) {
            for (std::uint64_t pr = 0; pr < pairs; ++pr) {
              std::uint64_t i0 = 0;
              std::uint64_t i1 = 0;
              switch (p.mode) {
                case ir::RopeMode::kNormal:
                  i0 = p.offset + (2 * pr);
                  i1 = i0 + 1;
                  break;
                case ir::RopeMode::kVision:
                  i0 = pr;
                  i1 = pr + p.dims;
                  break;
                default:
                  i0 = p.offset + pr;
                  i1 = i0 + (p.dims / 2);
                  break;
              }
              double c = 0.0;
              double s = 0.0;
              if (p.table) {
                c = pos[Ix({2, pairs, t_count, 1}, 0, pr, t, 0)];
                s = pos[Ix({2, pairs, t_count, 1}, 1, pr, t, 0)];
              } else {
                std::uint64_t section = 0;
                double exponent = static_cast<double>(pr);
                if (p.mode == ir::RopeMode::kInterleavedMulti) {
                  const std::uint64_t sector = pr % sect_dims;
                  if (sector % 3 == 1 && sector < 3ULL * sec[1]) {
                    section = 1;
                  } else if (sector % 3 == 2 && sector < 3ULL * sec[2]) {
                    section = 2;
                  } else if (sector % 3 == 0 && sector < 3ULL * sec[0]) {
                    section = 0;
                  } else {
                    section = 3;
                  }
                } else if (p.mode == ir::RopeMode::kMulti) {
                  const std::uint64_t sector = pr % sect_dims;
                  if (sector < sec[0]) {
                    section = 0;
                  } else if (sector < std::uint64_t{sec[0]} + sec[1]) {
                    section = 1;
                  } else if (sector < std::uint64_t{sec[0]} + sec[1] + sec[2]) {
                    section = 2;
                  } else {
                    section = 3;
                  }
                } else if (p.mode == ir::RopeMode::kVision) {
                  const std::uint64_t sector = pr % (std::uint64_t{sec[0]} + sec[1]);
                  section = sector < sec[0] ? 0 : 1;
                  exponent = static_cast<double>(sector < sec[0] ? sector : sector - sec[0]);
                }
                const auto position =
                    static_cast<float>(pos[static_cast<std::size_t>((section * t_count) + t)]);
                float theta = 0.0f;
                float mscale = p.attn_factor;
                if (llama3) {
                  const float base_theta = RoundF32(std::pow(
                      static_cast<double>(p.base), (2.0 * exponent) / static_cast<double>(p.dims)));
                  const float freq_val = 1.0f / base_theta;
                  const float old_ctx = static_cast<float>(p.original_context);
                  const float low_freq_wavelen = old_ctx / p.beta_slow;
                  const float high_freq_wavelen = old_ctx / p.beta_fast;
                  const float wavelen = 6.28318530717958647692f / freq_val;
                  float scaled_freq = freq_val;
                  if (wavelen > low_freq_wavelen) {
                    scaled_freq = freq_val * p.scale;
                  } else if (wavelen > high_freq_wavelen && p.beta_fast > p.beta_slow) {
                    const float smooth =
                        (old_ctx / wavelen - p.beta_slow) / (p.beta_fast - p.beta_slow);
                    scaled_freq = ((1.0f - smooth) * freq_val * p.scale) + (smooth * freq_val);
                  }
                  theta = position * scaled_freq;
                  if (!freq.empty()) {
                    theta /= static_cast<float>(freq[static_cast<std::size_t>(pr)]);
                  }
                  mscale = p.attn_factor != 0.0f ? p.attn_factor : 1.0f;
                } else {
                  const float power =
                      RoundF32(std::pow(static_cast<double>(theta_scale), exponent));
                  float extrap = position * power;
                  if (!freq.empty()) {
                    extrap /= static_cast<float>(freq[static_cast<std::size_t>(pr)]);
                  }
                  const float interp = p.scale * extrap;
                  theta = interp;
                  if (p.ext_factor > 0.0f) {
                    const float y =
                        (static_cast<float>(pr) - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
                    const float ramp = (1.0f - std::min(1.0f, std::max(0.0f, y))) * p.ext_factor;
                    const float kept = interp * (1.0f - ramp);
                    const float mixed = extrap * ramp;
                    theta = kept + mixed;
                    // Two statements: no contraction into a fused multiply-add.
                    const float magnitude = 0.1f * std::log(1.0f / p.scale);
                    mscale *= 1.0f + magnitude;
                  }
                }
                c = std::cos(static_cast<double>(theta)) * mscale;
                s = std::sin(static_cast<double>(theta)) * mscale;
              }
              if (p.inverse) {
                s = -s;
              }
              const double a = x[Ix(xn, i0, h, t, i3)];
              const double b = x[Ix(xn, i1, h, t, i3)];
              r[Ix(xn, i0, h, t, i3)] = (a * c) - (b * s);
              r[Ix(xn, i1, h, t, i3)] = (a * s) + (b * c);
            }
          }
        }
      }
      return Store(out, r);
    }

    case Op::kAdd:
    case Op::kSub:
    case Op::kMul:
    case Op::kDiv: {
      LLMP_LOAD(a, node.src[0]);
      LLMP_LOAD(b, node.src[1]);
      const Ne& an = ne(0);
      const Ne& bn = ne(1);
      Vec r(a.size());
      for (std::uint64_t i3 = 0; i3 < an[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < an[2]; ++i2) {
          for (std::uint64_t i1 = 0; i1 < an[1]; ++i1) {
            for (std::uint64_t i0 = 0; i0 < an[0]; ++i0) {
              const double x = a[Ix(an, i0, i1, i2, i3)];
              const double y = b[Ix(bn, i0 % bn[0], i1 % bn[1], i2 % bn[2], i3 % bn[3])];
              double z = 0.0;
              switch (node.key.op) {
                case Op::kAdd:
                  z = x + y;
                  break;
                case Op::kSub:
                  z = x - y;
                  break;
                case Op::kMul:
                  z = x * y;
                  break;
                default:
                  z = x / y;
                  break;
              }
              r[Ix(an, i0, i1, i2, i3)] = z;
            }
          }
        }
      }
      return Store(out, r);
    }
    case Op::kScale:
    case Op::kUnary:
    case Op::kClamp:
    case Op::kSoftcap: {
      LLMP_LOAD(x, node.src[0]);
      Vec r(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) {
        const double v = x[i];
        switch (node.key.op) {
          case Op::kScale: {
            const auto p = node.Params<ir::ScaleParams>();
            r[i] = (v * p.scale) + p.bias;
            break;
          }
          case Op::kUnary:
            r[i] = Apply(node.Params<ir::UnaryParams>().fn, v);
            break;
          case Op::kClamp: {
            const auto p = node.Params<ir::ClampParams>();
            r[i] = std::isnan(v)
                       ? v
                       : std::clamp(v, static_cast<double>(p.min), static_cast<double>(p.max));
            break;
          }
          default: {
            const double cap = node.Params<ir::SoftcapParams>().cap;
            r[i] = cap * std::tanh(v / cap);
            break;
          }
        }
      }
      return Store(out, r);
    }
    case Op::kFill: {
      const Vec r(Count(out_ne()), node.Params<ir::FillParams>().value);
      return Store(out, r);
    }
    case Op::kGlu: {
      const auto p = node.Params<ir::GluParams>();
      LLMP_LOAD(a, node.src[0]);
      const Ne& on = out_ne();
      Vec r(Count(on));
      if (has(1)) {
        LLMP_LOAD(b, node.src[1]);
        for (std::size_t i = 0; i < r.size(); ++i) {
          r[i] = Glu(p, a[i], b[i]);
        }
        return Store(out, r);
      }
      const std::uint64_t half = on[0];
      for (std::size_t row = 0; row < r.size() / half; ++row) {
        for (std::uint64_t i = 0; i < half; ++i) {
          const double first = a[(row * 2 * half) + i];
          const double second = a[(row * 2 * half) + half + i];
          r[(row * half) + i] = p.swapped ? Glu(p, second, first) : Glu(p, first, second);
        }
      }
      return Store(out, r);
    }

    case Op::kSumRows: {
      LLMP_LOAD(x, node.src[0]);
      const std::uint64_t n = ne(0)[0];
      Vec r(x.size() / n);
      for (std::size_t row = 0; row < r.size(); ++row) {
        double s = 0.0;
        for (std::uint64_t i = 0; i < n; ++i) {
          s += x[(row * n) + i];
        }
        r[row] = s;
      }
      return Store(out, r);
    }
    case Op::kSoftMax: {
      const auto p = node.Params<ir::SoftMaxParams>();
      LLMP_LOAD(x, node.src[0]);
      Vec mask;
      Vec sinks;
      if (has(1)) {
        LLMP_LOAD(m, node.src[1]);
        mask = m;
      }
      if (has(2)) {
        LLMP_LOAD(s, node.src[2]);
        sinks = s;
      }
      const Ne& xn = ne(0);
      const Ne mn = has(1) ? ne(1) : Ne{1, 1, 1, 1};
      Vec r(x.size());
      for (std::uint64_t i3 = 0; i3 < xn[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < xn[2]; ++i2) {
          const double slope = Slope(p.max_bias, i2, xn[2]);
          for (std::uint64_t i1 = 0; i1 < xn[1]; ++i1) {
            std::span<double> row(&r[Ix(xn, 0, i1, i2, i3)], static_cast<std::size_t>(xn[0]));
            for (std::uint64_t i0 = 0; i0 < xn[0]; ++i0) {
              double v = x[Ix(xn, i0, i1, i2, i3)] * static_cast<double>(p.scale);
              if (!mask.empty()) {
                v += slope * mask[Ix(mn, i0, i1, i2 % mn[2], i3 % mn[3])];
              }
              row[i0] = v;
            }
            std::optional<double> sink;
            if (!sinks.empty()) {
              sink = sinks[i2];
            }
            SoftmaxRow(row, sink);
          }
        }
      }
      return Store(out, r);
    }
    case Op::kArgmax: {
      LLMP_LOAD(x, node.src[0]);
      const bool first_column = node.Params<ir::ArgmaxParams>().nan == ir::ArgmaxNan::kFirstColumn;
      const std::uint64_t n = ne(0)[0];
      Vec r(x.size() / n);
      for (std::size_t row = 0; row < r.size(); ++row) {
        std::size_t best = 0;
        bool found = false;
        if (first_column) {
          // std::max_element: only a value above the incumbent replaces it,
          // and nothing compares above a NaN.
          for (std::uint64_t i = 1; i < n; ++i) {
            if (x[(row * n) + best] < x[(row * n) + i]) {
              best = static_cast<std::size_t>(i);
            }
          }
        }
        for (std::uint64_t i = 0; i < n && !first_column; ++i) {
          const double v = x[(row * n) + i];
          if (!std::isnan(v) && (!found || v > x[(row * n) + best])) {
            best = static_cast<std::size_t>(i);
            found = true;
          }
        }
        r[row] = static_cast<double>(best);
      }
      return Store(out, r);
    }
    case Op::kArgsort:
    case Op::kTopK: {
      LLMP_LOAD(x, node.src[0]);
      const std::uint64_t n = ne(0)[0];
      const bool topk = node.key.op == Op::kTopK;
      const bool desc = topk || node.Params<ir::ArgsortParams>().descending;
      const std::uint64_t keep = topk ? node.Params<ir::TopKParams>().k : n;
      Vec r;
      r.reserve(x.size() / n * keep);
      for (std::size_t row = 0; row < x.size() / n; ++row) {
        const auto order = Order(std::span(&x[row * n], static_cast<std::size_t>(n)), desc);
        for (std::uint64_t i = 0; i < keep; ++i) {
          r.push_back(static_cast<double>(order[i]));
        }
      }
      return Store(out, r);
    }

    case Op::kVisibleTopK: {
      LLMP_LOAD(scores, node.src[0]);
      const auto counts = LoadInts(node.src[1]);
      if (!counts) return std::unexpected(counts.error());
      const std::uint64_t cells = ne(0)[0];
      const std::uint64_t top = node.Params<ir::TopKParams>().k;
      Vec selected(static_cast<std::size_t>(top * ne(0)[1]), -1);
      for (std::uint64_t row = 0; row < ne(0)[1]; ++row) {
        const auto visible = static_cast<std::uint64_t>(
            std::clamp((*counts)[row], std::int64_t{0}, static_cast<std::int64_t>(cells)));
        auto order = Order(std::span(scores).subspan(row * cells, visible), true);
        order.resize(static_cast<std::size_t>(std::min(top, visible)));
        std::ranges::sort(order);
        for (std::size_t i = 0; i < order.size(); ++i) {
          selected[row * top + i] = static_cast<double>(order[i]);
        }
      }
      return Store(out, selected);
    }
    case Op::kSelectedMask: {
      const auto ids = LoadInts(node.src[1]);
      if (!ids) return std::unexpected(ids.error());
      const auto p = node.Params<ir::SelectedMaskParams>();
      const TensorDesc& raw = graph().tensor(node.src[0]);
      const TensorDesc& dest = graph().tensor(out);
      const auto& source = RootOf(node.src[0]).parts[0];
      auto& target = RootOf(out).parts[0];
      for (std::uint64_t row = 0; row < raw.ne(1); ++row) {
        for (std::uint64_t cell = 0; cell < dest.ne(0); ++cell) {
          const auto bits = cell < raw.ne(0)
                                ? Raw<std::uint16_t>(source.data() + raw.view_offset +
                                                     row * raw.nb[1] + cell * raw.nb[0])
                                : std::uint16_t{0xfc00};
          Put(target.data() + dest.view_offset + row * dest.nb[1] + cell * dest.nb[0], bits);
        }
        for (std::uint64_t i = 0; i < ne(1)[0]; ++i) {
          const auto id = (*ids)[row * ne(1)[0] + i];
          if (id >= 0 && static_cast<std::uint64_t>(id) < p.cells) {
            Put(target.data() + dest.view_offset + row * dest.nb[1] +
                    (p.raw_cells + static_cast<std::uint64_t>(id)) * dest.nb[0],
                std::uint16_t{0});
          }
        }
      }
      return {};
    }

    case Op::kRoute: {
      const auto p = node.Params<ir::RouteParams>();
      LLMP_LOAD(logits, node.src[0]);
      Vec bias;
      if (has(1)) {
        LLMP_LOAD(b, node.src[1]);
        bias = b;
      }
      const std::uint64_t e = p.experts;
      const std::uint64_t u = p.used;
      const std::uint64_t t_count = ne(0)[1];
      Vec ids(u * t_count);
      Vec weights(u * t_count);
      Vec table;
      Vec tokens;
      // Without tokens, a hash's table is each row's ids already.
      const bool looked_up = p.kind == ir::RouteKind::kHash && !has(3);
      if (p.kind == ir::RouteKind::kHash) {
        LLMP_LOAD(tb, node.src[2]);
        table = tb;
        if (!looked_up) {
          LLMP_LOAD(tk, node.src[3]);
          tokens = tk;
        }
      }
      std::vector<double> probs(static_cast<std::size_t>(e));
      for (std::uint64_t t = 0; t < t_count; ++t) {
        for (std::uint64_t i = 0; i < e; ++i) {
          probs[i] = logits[(t * e) + i];
        }
        switch (p.kind) {
          case ir::RouteKind::kSoftmaxTopK:
            SoftmaxRow(probs, std::nullopt);
            break;
          case ir::RouteKind::kSigmoidTopK:
            for (double& v : probs) {
              v = Sigmoid(v);
            }
            break;
          case ir::RouteKind::kSqrtSoftplusBias:
          case ir::RouteKind::kHash:
            for (double& v : probs) {
              v = std::sqrt(Softplus(v));
            }
            break;
        }
        std::vector<std::size_t> chosen;
        if (p.kind == ir::RouteKind::kHash) {
          // The table's column for the row's token, or the row's own.
          std::uint64_t column = t;
          if (!looked_up) {
            const double token = tokens[t];
            const std::uint64_t vocab = ne(2)[1];
            if (!(token >= 0.0 && token < static_cast<double>(vocab))) {
              return Fail(ExecError::kIndex, std::format("token {} of {}", token, vocab));
            }
            column = static_cast<std::uint64_t>(token);
          }
          for (std::uint64_t i = 0; i < u; ++i) {
            const double id = table[(column * u) + i];
            if (!(id >= 0.0 && id < static_cast<double>(e))) {
              return Fail(ExecError::kIndex, std::format("expert {} of {}", id, e));
            }
            chosen.push_back(static_cast<std::size_t>(id));
          }
        } else {
          std::vector<double> selection = probs;
          if (!bias.empty()) {
            for (std::uint64_t i = 0; i < e; ++i) {
              selection[i] += bias[i];
            }
          }
          const auto order = Order(selection, true);
          chosen.assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(u));
        }
        double sum = 0.0;
        for (std::uint64_t i = 0; i < u; ++i) {
          sum += probs[chosen[i]];
        }
        for (std::uint64_t i = 0; i < u; ++i) {
          double w = probs[chosen[i]];
          if (p.normalize) {
            w /= std::max(sum, 0x1p-14);
          }
          ids[(t * u) + i] = static_cast<double>(chosen[i]);
          weights[(t * u) + i] = w * static_cast<double>(p.scale);
        }
      }
      LLMP_TRY(Store(node.out[0], ids));
      return Store(node.out[1], weights);
    }
    case Op::kExpertReduce: {
      LLMP_LOAD(x, node.src[0]);
      LLMP_LOAD(w, node.src[1]);
      Vec ids;
      Vec scale;
      if (has(2)) {
        LLMP_LOAD(i, node.src[2]);
        ids = i;
      }
      if (has(3)) {
        LLMP_LOAD(s, node.src[3]);
        scale = s;
      }
      const Ne& xn = ne(0);
      // A scale gathered per slot [1, U, T] (op.h), else one per expert.
      const bool gathered =
          has(3) && ne(3)[0] == 1 && ne(3)[1] == xn[1] && ne(3)[2] == xn[2] && ne(3)[3] == 1;
      Vec r(xn[0] * xn[2]);
      for (std::uint64_t t = 0; t < xn[2]; ++t) {
        for (std::uint64_t u = 0; u < xn[1]; ++u) {
          double factor = 1.0;
          if (gathered) {
            factor = scale[(t * xn[1]) + u];
          } else if (!scale.empty()) {
            const double id = ids[(t * xn[1]) + u];
            if (!(id >= 0.0 && id < static_cast<double>(scale.size()))) {
              return Fail(ExecError::kIndex, std::format("expert {} of {}", id, scale.size()));
            }
            factor = scale[static_cast<std::size_t>(id)];
          }
          const double weight = w[(t * xn[1]) + u];
          for (std::uint64_t i = 0; i < xn[0]; ++i) {
            r[(t * xn[0]) + i] += (x[Ix(xn, i, u, t, 0)] * factor) * weight;
          }
        }
      }
      return Store(out, r);
    }

    case Op::kSsmConv: {
      LLMP_LOAD(sx, node.src[0]);
      LLMP_LOAD(c, node.src[1]);
      const Ne& xn = ne(0);
      const Ne& cn = ne(1);
      const Ne& on = out_ne();
      Vec r(Count(on));
      for (std::uint64_t s = 0; s < on[2]; ++s) {
        for (std::uint64_t t = 0; t < on[1]; ++t) {
          for (std::uint64_t ch = 0; ch < on[0]; ++ch) {
            double sum = 0.0;
            for (std::uint64_t j = 0; j < cn[0]; ++j) {
              sum += sx[Ix(xn, t + j, ch, s, 0)] * c[Ix(cn, j, ch, 0, 0)];
            }
            r[Ix(on, ch, t, s, 0)] = sum;
          }
        }
      }
      return Store(out, r);
    }
    case Op::kGatedDeltaNet: {
      LLMP_LOAD(q, node.src[0]);
      LLMP_LOAD(k, node.src[1]);
      LLMP_LOAD(v, node.src[2]);
      LLMP_LOAD(gt, node.src[3]);
      LLMP_LOAD(beta, node.src[4]);
      LLMP_LOAD(state, node.src[5]);
      const Ne& qn = ne(0);
      const Ne& vn = ne(2);
      const Ne& gn = ne(3);
      const Ne& bn = ne(4);
      const Ne& sn = ne(5);
      const std::uint64_t sk = qn[0];
      const std::uint64_t sv = vn[0];
      const std::uint64_t heads = vn[1];
      const std::uint64_t tokens = vn[2];
      const std::uint64_t seqs = vn[3];
      const std::uint64_t seq_group = seqs / qn[3];
      const double scale = 1.0 / std::sqrt(static_cast<double>(sk));
      Vec attn(Count(out_ne(0)));
      Vec s_out(Count(out_ne(1)));
      std::vector<double> st(static_cast<std::size_t>(sk * sv));
      for (std::uint64_t b = 0; b < seqs; ++b) {
        for (std::uint64_t h = 0; h < heads; ++h) {
          const std::uint64_t qh = h % qn[1];
          const std::uint64_t qb = b / seq_group;
          for (std::uint64_t c = 0; c < sv; ++c) {
            for (std::uint64_t i = 0; i < sk; ++i) {
              st[(c * sk) + i] = state[Ix(sn, i, c, h, b)];
            }
          }
          for (std::uint64_t t = 0; t < tokens; ++t) {
            const double bt = beta[Ix(bn, 0, h, t, b)];
            for (std::uint64_t c = 0; c < sv; ++c) {
              double kv = 0.0;
              for (std::uint64_t i = 0; i < sk; ++i) {
                const double d = std::exp(gt[Ix(gn, gn[0] == 1 ? 0 : i, h, t, b)]);
                st[(c * sk) + i] *= d;
                kv += st[(c * sk) + i] * k[Ix(qn, i, qh, t, qb)];
              }
              const double delta = bt * (v[Ix(vn, c, h, t, b)] - kv);
              double a = 0.0;
              for (std::uint64_t i = 0; i < sk; ++i) {
                st[(c * sk) + i] += k[Ix(qn, i, qh, t, qb)] * delta;
                a += st[(c * sk) + i] * q[Ix(qn, i, qh, t, qb)];
              }
              attn[Ix(out_ne(0), c, h, t, b)] = a * scale;
            }
          }
          for (std::uint64_t c = 0; c < sv; ++c) {
            for (std::uint64_t i = 0; i < sk; ++i) {
              s_out[Ix(out_ne(1), i, c, h, b)] = st[(c * sk) + i];
            }
          }
        }
      }
      LLMP_TRY(Store(node.out[0], attn));
      return Store(node.out[1], s_out);
    }
    case Op::kConv2d: {
      const auto p = node.Params<ir::Conv2dParams>();
      LLMP_LOAD(w, node.src[0]);
      LLMP_LOAD(x, node.src[1]);
      Vec bias;
      if (has(2)) {
        LLMP_LOAD(b, node.src[2]);
        bias = b;
      }
      const Ne& wn = ne(0);
      const Ne& xn = ne(1);
      const Ne& on = out_ne();
      Vec r(Count(on));
      for (std::uint64_t n = 0; n < on[3]; ++n) {
        for (std::uint64_t oc = 0; oc < on[2]; ++oc) {
          for (std::uint64_t oy = 0; oy < on[1]; ++oy) {
            for (std::uint64_t ox = 0; ox < on[0]; ++ox) {
              double s = bias.empty() ? 0.0 : bias[oc];
              for (std::uint64_t ic = 0; ic < wn[2]; ++ic) {
                for (std::uint64_t ky = 0; ky < wn[1]; ++ky) {
                  const auto iy = static_cast<std::int64_t>((oy * p.stride) + (ky * p.dilation)) -
                                  static_cast<std::int64_t>(p.padding);
                  if (iy < 0 || iy >= static_cast<std::int64_t>(xn[1])) {
                    continue;
                  }
                  for (std::uint64_t kx = 0; kx < wn[0]; ++kx) {
                    const auto ix = static_cast<std::int64_t>((ox * p.stride) + (kx * p.dilation)) -
                                    static_cast<std::int64_t>(p.padding);
                    if (ix < 0 || ix >= static_cast<std::int64_t>(xn[0])) {
                      continue;
                    }
                    s += w[Ix(wn, kx, ky, ic, oc)] * x[Ix(xn, static_cast<std::uint64_t>(ix),
                                                          static_cast<std::uint64_t>(iy), ic, n)];
                  }
                }
              }
              r[Ix(on, ox, oy, oc, n)] = s;
            }
          }
        }
      }
      return Store(out, r);
    }
    case Op::kUpsample: {
      const std::uint64_t f = node.Params<ir::UpsampleParams>().factor;
      LLMP_LOAD(x, node.src[0]);
      const Ne& xn = ne(0);
      const Ne& on = out_ne();
      Vec r(Count(on));
      for (std::uint64_t i3 = 0; i3 < on[3]; ++i3) {
        for (std::uint64_t i2 = 0; i2 < on[2]; ++i2) {
          for (std::uint64_t i1 = 0; i1 < on[1]; ++i1) {
            for (std::uint64_t i0 = 0; i0 < on[0]; ++i0) {
              r[Ix(on, i0, i1, i2, i3)] = x[Ix(xn, i0 / f, i1 / f, i2, i3)];
            }
          }
        }
      }
      return Store(out, r);
    }

    case Op::kHcPre: {
      const auto p = node.Params<ir::HcParams>();
      LLMP_LOAD(x, node.src[0]);
      LLMP_LOAD(w, node.src[1]);
      const Ne& xn = ne(0);
      const std::uint64_t width = xn[0];
      const std::uint64_t hc = xn[1];
      Vec r(width * xn[2]);
      for (std::uint64_t t = 0; t < xn[2]; ++t) {
        for (std::uint64_t i = 0; i < width; ++i) {
          double s = 0.0;
          for (std::uint64_t h = 0; h < hc; ++h) {
            const double val = x[Ix(xn, i, h, t, 0)];
            s += p.fold == ir::HcFold::kWeights ? val * w[(t * hc) + h]
                                                : val * Sigmoid(w[Ix(xn, i, h, t, 0)]);
          }
          r[(t * width) + i] = p.scale == 1.0f ? s : s * p.scale;
        }
      }
      return Store(out, r);
    }
    case Op::kHcPost: {
      LLMP_LOAD(y, node.src[0]);
      LLMP_LOAD(x, node.src[1]);
      LLMP_LOAD(post, node.src[2]);
      Vec comb;
      if (has(3)) {
        LLMP_LOAD(cb, node.src[3]);
        comb = cb;
      }
      const Ne& xn = ne(1);
      const std::uint64_t width = xn[0];
      const std::uint64_t hc = xn[1];
      Vec r(Count(xn));
      for (std::uint64_t t = 0; t < xn[2]; ++t) {
        for (std::uint64_t d = 0; d < hc; ++d) {
          for (std::uint64_t i = 0; i < width; ++i) {
            double s = y[(t * width) + i] * post[(t * hc) + d];
            if (comb.empty()) {
              s += x[Ix(xn, i, d, t, 0)];
            } else {
              for (std::uint64_t src = 0; src < hc; ++src) {
                s += x[Ix(xn, i, src, t, 0)] * comb[(t * hc * hc) + d + (hc * src)];
              }
            }
            r[Ix(xn, i, d, t, 0)] = s;
          }
        }
      }
      return Store(out, r);
    }
    case Op::kHcMix: {
      const auto p = node.Params<ir::HcParams>();
      LLMP_LOAD(mixes, node.src[0]);
      LLMP_LOAD(scale, node.src[1]);
      LLMP_LOAD(base, node.src[2]);
      const std::uint64_t hc = p.streams;
      const std::uint64_t m = ne(0)[0];
      const std::uint64_t t_count = ne(0)[1];
      const double eps = p.eps;
      if (p.mix == ir::HcMixKind::kSigmoid) {
        const std::uint64_t off = p.mix_offset;
        const std::uint64_t sc_idx = p.scale_index;
        Vec r(hc * 2 * t_count);
        for (std::uint64_t t = 0; t < t_count; ++t) {
          for (std::uint64_t d = 0; d < hc; ++d) {
            const double pre =
                Sigmoid((mixes[(t * m) + off + d] * scale[sc_idx]) + base[off + d]) + eps;
            const double post = 2.0 * Sigmoid((mixes[(t * m) + off + hc + d] * scale[sc_idx + 1]) +
                                              base[off + hc + d]);
            r[(t * 2 * hc) + d] = pre;
            r[(t * 2 * hc) + hc + d] = post;
          }
        }
        return Store(out, r);
      }
      const std::uint64_t off = p.mix_offset != 0 ? p.mix_offset : (m == hc * hc ? 0 : 2 * hc);
      const std::uint64_t sc_idx = p.scale_index != 0 ? p.scale_index : (ne(1)[0] == 1 ? 0 : 2);
      Vec r(hc * hc * t_count);
      std::vector<double> c(static_cast<std::size_t>(hc * hc));  // c[dst + hc · src]
      for (std::uint64_t t = 0; t < t_count; ++t) {
        for (std::uint64_t src = 0; src < hc; ++src) {
          std::span<double> col(&c[hc * src], static_cast<std::size_t>(hc));
          for (std::uint64_t dst = 0; dst < hc; ++dst) {
            const std::uint64_t at = off + dst + (hc * src);
            col[dst] = (mixes[(t * m) + at] * scale[sc_idx]) + base[at];
          }
          SoftmaxRow(col, std::nullopt);
          for (double& v : col) {
            v += eps;
          }
        }
        const auto cols = [&] {
          for (std::uint64_t dst = 0; dst < hc; ++dst) {
            double s = eps;
            for (std::uint64_t src = 0; src < hc; ++src) {
              s += c[dst + (hc * src)];
            }
            for (std::uint64_t src = 0; src < hc; ++src) {
              c[dst + (hc * src)] /= s;
            }
          }
        };
        const auto rows = [&] {
          for (std::uint64_t src = 0; src < hc; ++src) {
            double s = eps;
            for (std::uint64_t dst = 0; dst < hc; ++dst) {
              s += c[dst + (hc * src)];
            }
            for (std::uint64_t dst = 0; dst < hc; ++dst) {
              c[dst + (hc * src)] /= s;
            }
          }
        };
        cols();
        for (std::uint32_t i = 1; i < p.iterations; ++i) {
          rows();
          cols();
        }
        std::ranges::copy(c, r.begin() + static_cast<std::ptrdiff_t>(t * hc * hc));
      }
      return Store(out, r);
    }

    case Op::kSample: {
      const auto p = node.Params<ir::SampleParams>();
      LLMP_LOAD(logits, node.src[0]);
      const std::uint64_t vocab = ne(0)[0];
      const std::uint64_t rows = ne(0)[1];
      Vec r(rows);
      std::vector<float> row(static_cast<std::size_t>(vocab));
      std::vector<execution::SamplingCandidate> scratch;
      // Keys are 64-bit: read exactly, not through double.
      std::vector<std::int64_t> keyv;
      if (has(1)) {
        auto kv = LoadInts(node.src[1]);
        if (!kv) {
          return std::unexpected(kv.error());
        }
        keyv = std::move(*kv);
      }
      for (std::uint64_t i = 0; i < rows; ++i) {
        for (std::uint64_t j = 0; j < vocab; ++j) {
          row[j] = static_cast<float>(logits[(i * vocab) + j]);
        }
        std::expected<std::int32_t, execution::SamplingError> tok =
            std::unexpected(execution::SamplingError::kNoLogits);
        if (p.kind == ir::SampleKind::kGreedy) {
          tok = execution::Greedy(row);
        } else {
          const execution::SamplingParams sp{
              .temperature = p.temperature, .top_k = p.top_k, .top_p = p.top_p, .min_p = p.min_p};
          const execution::SamplingKey key{
              .seed = static_cast<std::uint64_t>(keyv[i * 3]),
              .stream = static_cast<std::uint64_t>(keyv[(i * 3) + 1]),
              .position = static_cast<std::uint64_t>(keyv[(i * 3) + 2])};
          tok = execution::Sample(row, sp, key, scratch);
        }
        if (!tok) {
          return Fail(ExecError::kNumeric, std::format("sampling row {}: {}", i,
                                                       execution::SamplingErrorName(tok.error())));
        }
        r[i] = *tok;
      }
      return Store(out, r);
    }
    case Op::kAllReduce:
    case Op::kAllGather:
    case Op::kAllToAll:
    case Op::kSend:
    case Op::kRecv:
    case Op::kFused:
      break;
  }
  return Fail(ExecError::kUnsupported, "not run here");
}

// ---------------------------------------------------------------- devices

std::expected<void, ExecFailure> RunDevices(std::span<Executor* const> devices,
                                            std::span<const DeviceGroup> groups) {
  for (Executor* e : devices) {
    if (auto p = e->Prepare(); !p) {
      return p;
    }
    e->next_ = 0;
  }
  // The group a device's collective names, and the device's rank in it.
  const auto member = [&](std::uint32_t group, std::size_t device)
      -> std::expected<std::pair<std::vector<std::uint32_t>, std::size_t>, ExecFailure> {
    for (const DeviceGroup& g : groups) {
      if (g.group == group) {
        for (std::size_t r = 0; r < g.devices.size(); ++r) {
          if (g.devices[r] == device) {
            return std::pair(g.devices, r);
          }
        }
        return Fail(ExecError::kCollective,
                    std::format("device {} is not in group {}", device, group));
      }
    }
    if (devices.size() == 1) {
      return std::pair(std::vector<std::uint32_t>{0}, std::size_t{0});
    }
    return Fail(ExecError::kCollective, std::format("no group {}", group));
  };
  // Sends in flight: (group, from rank, to rank) -> values in order.
  std::map<std::tuple<std::uint32_t, std::size_t, std::size_t>, std::deque<Vec>> queued;
  std::vector<std::size_t> at(devices.size());
  std::vector<bool> done(devices.size(), false);
  // At a group collective, until its group completes it.
  std::vector<bool> waiting(devices.size(), false);
  while (true) {
    bool progress = false;
    for (std::size_t d = 0; d < devices.size(); ++d) {
      if (done[d] || waiting[d]) {
        continue;
      }
      Executor& e = *devices[d];
      while (true) {
        auto stop = e.Advance();
        if (!stop) {
          return std::unexpected(stop.error());
        }
        if (*stop >= e.graph().nodes().size()) {
          done[d] = true;
          progress = true;
          break;
        }
        at[d] = *stop;
        const ir::Node& node = e.graph().nodes()[*stop];
        const auto p = node.Params<ir::CollectiveParams>();
        if (node.key.op != Op::kSend && node.key.op != Op::kRecv) {
          waiting[d] = true;
          break;  // waits for its group
        }
        auto m = member(p.group, d);
        if (!m) {
          return std::unexpected(m.error());
        }
        if (p.peer >= m->first.size()) {
          return Fail(ExecError::kCollective, std::format("peer {} of {}", p.peer, m->first.size()),
                      static_cast<std::uint32_t>(*stop));
        }
        if (node.key.op == Op::kSend) {
          auto v = e.Load(node.src[0]);
          if (!v) {
            return std::unexpected(v.error());
          }
          queued[{p.group, m->second, p.peer}].push_back(std::move(*v));
        } else {
          auto& q = queued[{p.group, p.peer, m->second}];
          if (q.empty()) {
            break;  // waits for its peer
          }
          if (q.front().size() != Count(e.graph().tensor(node.out[0]).shape().ne)) {
            return Fail(ExecError::kCollective, "a receive of another size than its send",
                        static_cast<std::uint32_t>(*stop));
          }
          if (auto s = e.Store(node.out[0], q.front()); !s) {
            return s;
          }
          q.pop_front();
        }
        ++e.next_;
        progress = true;
      }
    }
    if (std::ranges::all_of(done, [](bool b) { return b; })) {
      return {};
    }
    // Group collectives whose members have all arrived.
    for (std::size_t d = 0; d < devices.size(); ++d) {
      if (!waiting[d]) {
        continue;
      }
      const ir::Node& node = devices[d]->graph().nodes()[at[d]];
      const auto p = node.Params<ir::CollectiveParams>();
      auto m = member(p.group, d);
      if (!m) {
        return std::unexpected(m.error());
      }
      const std::vector<std::uint32_t>& ranks = m->first;
      bool ready = true;
      for (const std::uint32_t r : ranks) {
        if (r >= devices.size() || done[r]) {
          return Fail(ExecError::kCollective,
                      std::format("group {}'s device {} ended before its collective", p.group, r));
        }
        if (!waiting[r]) {
          ready = false;
          break;
        }
        const ir::Node& other = devices[r]->graph().nodes()[at[r]];
        ready = ready && other.key.op == node.key.op &&
                other.Params<ir::CollectiveParams>().group == p.group;
      }
      if (!ready) {
        continue;
      }
      std::vector<Vec> in;
      std::vector<Ne> shapes;
      for (const std::uint32_t r : ranks) {
        Executor& e = *devices[r];
        const ir::Node& n = e.graph().nodes()[at[r]];
        auto v = e.Load(n.src[0]);
        if (!v) {
          return std::unexpected(v.error());
        }
        in.push_back(std::move(*v));
        shapes.push_back(e.graph().tensor(n.src[0]).shape().ne);
      }
      const std::size_t g = ranks.size();
      for (std::size_t rank = 0; rank < g; ++rank) {
        Executor& e = *devices[ranks[rank]];
        const ir::Node& n = e.graph().nodes()[at[ranks[rank]]];
        const Ne& on = e.graph().tensor(n.out[0]).shape().ne;
        const std::uint8_t dim = n.Params<ir::CollectiveParams>().dim;
        Vec r(Count(on));
        if (n.key.op == Op::kAllReduce) {
          for (std::size_t j = 0; j < g; ++j) {
            if (in[j].size() != r.size()) {
              return Fail(ExecError::kCollective, "all-reduce over different sizes");
            }
            for (std::size_t i = 0; i < r.size(); ++i) {
              r[i] += in[j][i];
            }
          }
        } else {
          // Gather (each member's whole x) or all-to-all (part `rank` of each).
          const bool gather = n.key.op == Op::kAllGather;
          const Ne& xn = shapes[rank];
          if (!gather && xn.at(dim) % g != 0) {
            return Fail(ExecError::kCollective, "all-to-all over a dim the group does not divide");
          }
          const std::uint64_t part = gather ? xn.at(dim) : xn.at(dim) / g;
          if (on.at(dim) != part * g) {
            return Fail(ExecError::kCollective,
                        std::format("a gather of {} members into {}", g, on.at(dim)));
          }
          for (std::uint64_t i3 = 0; i3 < on[3]; ++i3) {
            for (std::uint64_t i2 = 0; i2 < on[2]; ++i2) {
              for (std::uint64_t i1 = 0; i1 < on[1]; ++i1) {
                for (std::uint64_t i0 = 0; i0 < on[0]; ++i0) {
                  std::array<std::uint64_t, 4> i{i0, i1, i2, i3};
                  const std::uint64_t from = i.at(dim) / part;
                  i.at(dim) = (i.at(dim) % part) + (gather ? 0 : rank * part);
                  if (shapes[from] != xn) {
                    return Fail(ExecError::kCollective, "members of different shapes");
                  }
                  r[Ix(on, i0, i1, i2, i3)] = in[from][Ix(xn, i[0], i[1], i[2], i[3])];
                }
              }
            }
          }
        }
        if (auto s = e.Store(n.out[0], r); !s) {
          return s;
        }
      }
      for (const std::uint32_t r : ranks) {
        ++devices[r]->next_;
        waiting[r] = false;
      }
      progress = true;
    }
    if (!progress) {
      return Fail(ExecError::kCollective, "the devices wait on each other");
    }
  }
}

}  // namespace llmp::kernels::reference
