// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/infer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/op_traits.h"
#include "ir/verify.h"

namespace llmp::ir {

bool ResultEncoding(format::Encoding encoding) {
  return encoding == format::Encoding::kF32 || encoding == format::Encoding::kF16 ||
         encoding == format::Encoding::kBF16;
}

namespace {

using format::Encoding;
using format::Shape;
using Out = std::expected<Inferred, VerifyError>;

std::unexpected<VerifyError> Err(GraphError error, std::string detail) {
  return std::unexpected(VerifyError{.error = error, .node = 0, .detail = std::move(detail)});
}

std::string Text(const Shape& s) {
  return std::format("[{}, {}, {}, {}]", s.ne[0], s.ne[1], s.ne[2], s.ne[3]);
}

Shape S(std::uint64_t a, std::uint64_t b = 1, std::uint64_t c = 1, std::uint64_t d = 1) {
  return ShapeOf({a, b, c, d});
}

bool Integer(Encoding e) {
  switch (e) {
    case Encoding::kI8:
    case Encoding::kI16:
    case Encoding::kI32:
    case Encoding::kI64:
    case Encoding::kU8:
    case Encoding::kU16:
    case Encoding::kU32:
    case Encoding::kU64:
    case Encoding::kBool:
      return true;
    default:
      return false;
  }
}

// Whether a dense format of this encoding is one a result may be cast to.
bool DenseEncoding(Encoding e) {
  const format::EncodingInfo& info = format::Info(e);
  return !info.quantized && info.block_elements == 1 && e != Encoding::kE8M0;
}

TensorDesc Fresh(const Shape& shape, Encoding encoding) {
  TensorDesc t;
  t.role = Role::kActivation;
  t.format = Dense(encoding, shape);
  t.nb = NaturalStrides(t.format);
  return t;
}

// The float encoding a computed result of `a` takes: a's when it is a
// result encoding, else F32.
Encoding FloatOf(const TensorDesc& a) {
  return ResultEncoding(a.format.encoding) ? a.format.encoding : Encoding::kF32;
}

class Ctx {
 public:
  Ctx(OpKey key, std::span<const std::byte, kMaxParamBytes> params,
      std::span<const TensorId> sources, std::span<const TensorDesc> tensors,
      std::span<const TensorDesc> requested)
      : key_(key), params_(params), sources_(sources), tensors_(tensors), requested_(requested) {}

  std::size_t count() const { return sources_.size(); }
  // Source i, or null when absent.
  const TensorDesc* src(std::size_t i) const {
    if (i >= sources_.size() || !sources_[i].valid()) {
      return nullptr;
    }
    return &tensors_[sources_[i].v];
  }
  TensorId id(std::size_t i) const { return sources_[i]; }
  TensorId root(std::size_t i) const {
    const TensorDesc& t = tensors_[sources_[i].v];
    return t.view_of.valid() ? t.view_of : sources_[i];
  }
  const TensorDesc& root_desc(std::size_t i) const { return tensors_[root(i).v]; }
  const TensorDesc* req(std::size_t i) const {
    return i < requested_.size() ? &requested_[i] : nullptr;
  }
  template <typename P>
  P params() const {
    P p{};
    std::memcpy(&p, params_.data(), sizeof(P));
    return p;
  }
  std::uint8_t byte(std::size_t i) const { return static_cast<std::uint8_t>(params_[i]); }
  std::span<const std::byte, kMaxParamBytes> raw() const { return params_; }
  OpKey key() const { return key_; }
  std::span<const TensorId> sources() const { return sources_; }
  std::span<const TensorDesc> tensors() const { return tensors_; }

 private:
  OpKey key_;
  std::span<const std::byte, kMaxParamBytes> params_;
  std::span<const TensorId> sources_;
  std::span<const TensorDesc> tensors_;
  std::span<const TensorDesc> requested_;
};

// Sources: `required` positions must be present, the rest up to `total`
// may be absent, none past `total`.
std::expected<void, VerifyError> Arity(const Ctx& c, std::size_t required, std::size_t total) {
  if (c.count() < required || c.count() > total) {
    return Err(GraphError::kArity, std::format("{} takes {} to {} sources, not {}",
                                               OpKeyName(c.key()), required, total, c.count()));
  }
  for (std::size_t i = 0; i < required; ++i) {
    if (c.src(i) == nullptr) {
      return Err(GraphError::kArity,
                 std::format("{}'s source {} is required", OpKeyName(c.key()), i));
    }
  }
  return {};
}

// The bytes a parameter struct of `size` leaves zero, and reserved bytes.
std::expected<void, VerifyError> Tail(const Ctx& c, std::size_t size) {
  for (std::size_t i = size; i < kMaxParamBytes; ++i) {
    if (c.byte(i) != 0) {
      return Err(GraphError::kParams, std::format("{}'s parameter byte {} is set past its {} bytes",
                                                  OpKeyName(c.key()), i, size));
    }
  }
  return {};
}

std::expected<void, VerifyError> Zero(const Ctx& c, std::size_t offset, std::size_t count) {
  for (std::size_t i = offset; i < offset + count; ++i) {
    if (c.byte(i) != 0) {
      return Err(GraphError::kParams,
                 std::format("{}'s reserved parameter byte {} is set", OpKeyName(c.key()), i));
    }
  }
  return {};
}

std::expected<void, VerifyError> Bool(const Ctx& c, std::size_t offset) {
  if (c.byte(offset) > 1) {
    return Err(GraphError::kParams,
               std::format("{}'s flag at byte {} is not 0 or 1", OpKeyName(c.key()), offset));
  }
  return {};
}

std::unexpected<VerifyError> Fail(GraphError e, const Ctx& c, const std::string& what) {
  return Err(e, std::format("{}: {}", OpKeyName(c.key()), what));
}

#define LLMP_TRY(expr)                         \
  if (auto llmp_try_ = (expr); !llmp_try_) {   \
    return std::unexpected(llmp_try_.error()); \
  }

std::expected<void, VerifyError> Readable(const Ctx& c, std::size_t i) {
  const TensorDesc* t = c.src(i);
  if (t == nullptr) return {};
  if (const auto valid = format::Validate(t->format); !valid) {
    return Fail(GraphError::kFormat, c,
                std::format("source {} has an invalid format ({} {} [{}, {}, {}, {}]: {})", i,
                            format::Info(t->format.encoding).name,
                            format::Info(t->format.layout).name, t->ne(0), t->ne(1), t->ne(2),
                            t->ne(3), format::FormatErrorName(valid.error())));
  }
  return {};
}

std::expected<void, VerifyError> FloatSrc(const Ctx& c, std::size_t i) {
  LLMP_TRY(Readable(c, i));
  const TensorDesc* t = c.src(i);
  if (t != nullptr && Integer(t->format.encoding)) {
    return Fail(GraphError::kFormat, c, std::format("source {} is an integer, not values", i));
  }
  return {};
}

std::expected<void, VerifyError> IndexSrc(const Ctx& c, std::size_t i, bool i64 = false) {
  const TensorDesc* t = c.src(i);
  if (t == nullptr) {
    return {};
  }
  const Encoding e = t->format.encoding;
  if (t->format.layout != format::Layout::kDense ||
      !(e == Encoding::kI32 || (i64 && e == Encoding::kI64))) {
    return Fail(GraphError::kFormat, c,
                std::format("source {} must be I32{}", i, i64 ? " or I64" : ""));
  }
  return {};
}

std::expected<void, VerifyError> Want(bool ok, const Ctx& c, const std::string& what) {
  if (!ok) {
    return Fail(GraphError::kShape, c, what);
  }
  return {};
}

bool Broadcasts(const Shape& a, const Shape& b) {
  for (std::size_t d = 0; d < format::kMaxDims; ++d) {
    if (b.ne.at(d) == 0 || a.ne.at(d) % b.ne.at(d) != 0) {
      return false;
    }
  }
  return true;
}

Inferred One(const TensorDesc& t, bool float_result) {
  Inferred r;
  r.out[0] = t;
  r.nout = 1;
  r.float_result[0] = float_result;
  return r;
}

Inferred FloatResult(const Shape& shape, Encoding encoding) {
  return One(Fresh(shape, encoding), true);
}

// A view of source i's root with `format`, strides and a root-relative
// offset, inheriting its role and segment.
TensorDesc ViewOf(const Ctx& c, std::size_t i, const format::TensorFormat& f,
                  const std::array<std::uint64_t, format::kMaxDims>& nb, std::uint64_t offset) {
  const TensorDesc& s = *c.src(i);
  TensorDesc t;
  t.role = s.role == Role::kOutput ? Role::kActivation : s.role;
  t.format = f;
  t.nb = nb;
  t.view_of = c.root(i);
  t.view_offset = offset;
  t.segment = s.segment;
  return t;
}

// Whether a view of `root`'s bytes stays inside them.
std::expected<void, VerifyError> InRoot(const Ctx& c, const TensorDesc& view,
                                        const TensorDesc& root) {
  const std::uint64_t unit = UnitBytes(root.format);
  const std::uint64_t per = UnitElements(root.format);
  if (unit == 0 || per == 0) {
    return Fail(GraphError::kFormat, c, "a format stored in parts has no views");
  }
  if (view.format.encoding != root.format.encoding || view.format.layout != root.format.layout ||
      view.format.params != root.format.params) {
    return Fail(GraphError::kFormat, c, "a view's encoding and layout are its root's");
  }
  if ((view.ne(0) != per && view.nb[0] != unit) || view.ne(0) % per != 0) {
    return Fail(GraphError::kShape, c,
                std::format("a view steps by one unit ({} bytes) over whole units", unit));
  }
  // It starts at a unit and steps by whole units along every axis it has
  // more than one of (a singleton axis's stride is never taken): a kernel
  // reads whole aligned elements or blocks, never a unit's middle.
  bool aligned = view.view_offset % unit == 0;
  for (std::size_t d = 1; d < format::kMaxDims; ++d) {
    aligned = aligned && (view.ne(d) <= 1 || view.nb.at(d) % unit == 0);
  }
  if (!aligned) {
    return Fail(
        GraphError::kShape, c,
        std::format("a view at offset {} with strides [{}, {}, {}, {}] is not at whole "
                    "units ({} bytes)",
                    view.view_offset, view.nb[0], view.nb[1], view.nb[2], view.nb[3], unit));
  }
  const auto parts = format::PartsOf(root.format);
  if (!parts) {
    return Fail(GraphError::kFormat, c, "the root's format has no size");
  }
  const std::uint64_t bytes = (*parts)[0].bytes;
  // The last byte it covers: its last unit's.
  std::uint64_t last = view.view_offset;
  bool overflow = false;
  for (std::size_t d = 0; d < format::kMaxDims; ++d) {
    std::uint64_t n = view.format.shape.ne.at(d);
    if (d == 0) {
      n /= per;
    }
    std::uint64_t step = 0;
    overflow |= __builtin_mul_overflow(n - 1, view.nb.at(d), &step);
    overflow |= __builtin_add_overflow(last, step, &last);
  }
  overflow |= __builtin_add_overflow(last, unit, &last);
  if (overflow || last > bytes) {
    return Err(GraphError::kBounds,
               std::format("{}: a view {} at offset {} with strides [{}, {}, {}, {}] ends past its "
                           "root's {} bytes",
                           OpKeyName(c.key()), Text(view.format.shape), view.view_offset,
                           view.nb[0], view.nb[1], view.nb[2], view.nb[3], bytes));
  }
  return {};
}

// A store's target: state, a result of this step or an input (bound each
// step from the caller's buffer; production stores target state or a
// result, and tests store into inputs to probe write hazards), never a
// weight or a plan constant (bound once, shared, read-only).
std::expected<void, VerifyError> Writable(const Ctx& c, std::size_t i) {
  const Role role = c.root_desc(i).role;
  if (role == Role::kWeight || role == Role::kConstant) {
    return Fail(GraphError::kBinding, c,
                std::format("source {} is a {}'s bytes, which nothing writes", i, RoleName(role)));
  }
  return {};
}

// ---------------------------------------------------------------- per op

bool Finite(float v) { return std::isfinite(v); }

// A product's accumulation and hint (op.h): known values, reserved bytes
// zero, and a Hadamard matrix square.
std::expected<void, VerifyError> CheckProduct(const Ctx& c, const TensorDesc& a) {
  const auto p = c.params<ProductParams>();
  LLMP_TRY(Zero(c, 2, 2));
  LLMP_TRY(Zero(c, 5, 3));
  if (static_cast<std::uint8_t>(p.arithmetic) >
      static_cast<std::uint8_t>(ProductArithmetic::kBlock32Q8SumF32))
    return Fail(GraphError::kParams, c, "an unknown product arithmetic recipe");
  if (static_cast<std::uint8_t>(p.accumulate) > static_cast<std::uint8_t>(Accumulate::kF32) ||
      static_cast<std::uint8_t>(p.hint) > static_cast<std::uint8_t>(ProductHint::kHadamard)) {
    return Fail(GraphError::kParams, c, "an unknown accumulation or hint");
  }
  if (p.hint == ProductHint::kHadamard &&
      (c.key().op != Op::kMatMul || a.ne(0) != a.ne(1) || a.ne(2) != 1 || a.ne(3) != 1)) {
    return Fail(GraphError::kParams, c, "a Hadamard hint is a plain product's square a");
  }
  return {};
}

std::expected<void, VerifyError> NormEps(const Ctx& c, float eps) {
  if (!Finite(eps) || eps < 0.0f) {
    return Fail(GraphError::kParams, c, "eps must be finite and not negative");
  }
  return {};
}

std::expected<void, VerifyError> CheckRope(const Ctx& c, const RopeParams& p, std::size_t at) {
  LLMP_TRY(Bool(c, at + 1));
  LLMP_TRY(Bool(c, at + 2));
  LLMP_TRY(Bool(c, at + 3));
  if (static_cast<std::uint8_t>(p.mode) > static_cast<std::uint8_t>(RopeMode::kInterleavedMulti)) {
    return Fail(GraphError::kParams, c, "an unknown rotation mode");
  }
  if (p.dims == 0 || p.dims % 2 != 0) {
    return Fail(GraphError::kParams, c, "rotated dimensions must be even and not zero");
  }
  if (p.table && p.mode != RopeMode::kNormal && p.mode != RopeMode::kNeox) {
    return Fail(GraphError::kParams, c, "a table rotation pairs as kNormal or kNeox");
  }
  if (p.mode == RopeMode::kVision && p.offset != 0) {
    return Fail(GraphError::kParams, c, "the vision rotation has no offset");
  }
  if (p.offset % 2 != 0) {
    return Fail(GraphError::kParams, c, "the first rotated value is even");
  }
  if (p.mode == RopeMode::kVision && std::uint64_t{p.sections[0]} + p.sections[1] == 0) {
    return Fail(GraphError::kParams, c, "the vision rotation's two sections are its first two");
  }
  const bool multi = p.mode == RopeMode::kMulti || p.mode == RopeMode::kInterleavedMulti ||
                     p.mode == RopeMode::kVision;
  std::uint64_t sections = 0;
  for (const std::uint32_t s : p.sections) {
    sections += s;
  }
  if (multi != (sections != 0)) {
    return Fail(GraphError::kParams, c,
                "sections are the multi-section modes', and they need some");
  }
  if (!Finite(p.base) || p.base <= 0.0f || !Finite(p.scale) || p.scale <= 0.0f ||
      !Finite(p.ext_factor) || !Finite(p.attn_factor) || !Finite(p.beta_fast) ||
      !Finite(p.beta_slow)) {
    return Fail(GraphError::kParams, c, "a rotation parameter is out of range");
  }
  if (p.ext_factor != 0.0f && p.original_context == 0) {
    return Fail(GraphError::kParams, c, "YaRN needs the original context");
  }
  if (p.llama3 && (p.table || multi || p.original_context == 0 || p.scale >= 1.0F ||
                   p.ext_factor > 0.0F || p.beta_slow <= 0.0F || p.beta_fast <= p.beta_slow)) {
    return Fail(GraphError::kParams, c,
                "Llama 3 RoPE needs one-dimensional positions, an original context, scale < 1, "
                "and high frequency factor > low frequency factor > 0");
  }
  if (!p.llama3 && p.ext_factor < 0.0F) {
    return Fail(GraphError::kParams, c, "a negative YaRN extension needs Llama 3 RoPE");
  }
  if (p.rotated_dims != UINT32_MAX && (p.rotated_dims > p.dims || p.rotated_dims % 2 != 0 ||
                                       p.table || multi || p.llama3 || p.ext_factor != 0.0F)) {
    return Fail(GraphError::kParams, c,
                "proportional RoPE needs even active dimensions within its one-dimensional "
                "pairing span and no table, Llama 3 scaling or YaRN");
  }
  return {};
}

// kRope's shape rules over (x, pos, freq).
Out RopeOf(const Ctx& c, const RopeParams& p, std::size_t x_at, std::size_t pos_at,
           std::size_t freq_at) {
  LLMP_TRY(FloatSrc(c, x_at));
  const TensorDesc& x = *c.src(x_at);
  const TensorDesc& pos = *c.src(pos_at);
  const std::uint64_t d = x.ne(0);
  const std::uint64_t t = x.ne(2);
  const std::uint64_t span = p.mode == RopeMode::kVision ? 2ULL * p.dims : p.dims;
  LLMP_TRY(Want(std::uint64_t{p.offset} + span <= d, c,
                std::format("rotates {} values from {} of a {}-value head", span, p.offset, d)));
  if (p.table) {
    LLMP_TRY(FloatSrc(c, pos_at));
    LLMP_TRY(Want(
        pos.shape().ne == S(2, p.dims / 2, t).ne, c,
        std::format("a rotation table is [2, {}, {}], not {}", p.dims / 2, t, Text(pos.shape()))));
  } else {
    LLMP_TRY(IndexSrc(c, pos_at));
    const bool four = p.mode == RopeMode::kMulti || p.mode == RopeMode::kInterleavedMulti ||
                      p.mode == RopeMode::kVision;
    LLMP_TRY(Want(pos.shape().ne == S(four ? 4 * t : t).ne, c,
                  std::format("positions are [{}], not {}", four ? 4 * t : t, Text(pos.shape()))));
  }
  if (const TensorDesc* f = c.src(freq_at); f != nullptr) {
    LLMP_TRY(FloatSrc(c, freq_at));
    // One per pair: the vision rotation has dims pairs, the others dims / 2.
    const std::uint64_t pairs = p.mode == RopeMode::kVision ? p.dims : p.dims / 2;
    LLMP_TRY(Want(f->ne(0) >= pairs && Elements(f->shape()) == f->ne(0), c,
                  std::format("frequency factors are a vector of at least {}", pairs)));
  }
  return FloatResult(x.shape(), FloatOf(x));
}

Out Leaf(const Ctx& c) {
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a leaf needs its description");
  }
  // Only a plan constant's parameters hold anything (its content's digest)
  // and a step input's (its InputParams).
  const bool constant = c.key().op == Op::kInput && r->role == Role::kConstant;
  const bool input = c.key().op == Op::kInput && r->role == Role::kInput;
  LLMP_TRY(Tail(c, constant ? kConstantDigestBytes : input ? sizeof(InputParams) : 0));
  if (input) {
    const auto p = c.params<InputParams>();
    if (static_cast<std::uint8_t>(p.fact) > static_cast<std::uint8_t>(kLastInputFact) ||
        (p.fact == InputFact::kNone) != (p.sections == 0) || p.reserved != decltype(p.reserved){}) {
      return Fail(GraphError::kParams, c, "an input's fact is not one the op set has");
    }
  }
  if (!format::Validate(r->format)) {
    return Fail(GraphError::kFormat, c, "a leaf's format is not valid");
  }
  TensorDesc t = *r;
  t.view_of = {};
  t.view_offset = 0;
  t.nb = NaturalStrides(t.format);
  return One(t, false);
}

Out InferView(const Ctx& c) {
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a view needs its description");
  }
  format::TensorFormat f = c.src(0)->format;
  f.shape = r->format.shape;
  f.stacked = 1;
  if (!format::Validate(f)) {
    return Fail(GraphError::kShape, c, "the view's shape is not valid for its format");
  }
  TensorDesc t = ViewOf(c, 0, f, r->nb, r->view_offset);
  LLMP_TRY(InRoot(c, t, c.root_desc(0)));
  return One(t, false);
}

Out InferReshape(const Ctx& c) {
  const TensorDesc& a = *c.src(0);
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a reshape needs its shape");
  }
  LLMP_TRY(Want(IsContiguous(a), c, "reshapes a contiguous tensor"));
  LLMP_TRY(Want(Elements(r->shape()) == Elements(a.shape()), c,
                std::format("{} has another count than {}", Text(r->shape()), Text(a.shape()))));
  format::TensorFormat f = a.format;
  f.shape = r->format.shape;
  f.stacked = 1;
  if (!format::Validate(f)) {
    return Fail(GraphError::kShape, c, "the shape is not valid for its format");
  }
  return One(ViewOf(c, 0, f, NaturalStrides(f), a.view_offset), false);
}

Out InferPermute(const Ctx& c) {
  const auto p = c.params<PermuteParams>();
  std::array<bool, 4> seen{};
  for (const std::uint8_t ax : p.axes) {
    if (ax >= 4 || seen.at(ax)) {
      return Fail(GraphError::kParams, c, "axes are not a permutation of 0 to 3");
    }
    seen.at(ax) = true;
  }
  const TensorDesc& a = *c.src(0);
  if (a.format.layout != format::Layout::kDense && p.axes[0] != 0) {
    return Fail(GraphError::kFormat, c, "a block layout's rows stay rows");
  }
  if (UnitBytes(a.format) == 0) {
    return Fail(GraphError::kFormat, c, "a format stored in parts has no views");
  }
  format::TensorFormat f = a.format;
  std::array<std::uint64_t, 4> ne{};
  std::array<std::uint64_t, 4> nb{};
  for (std::size_t i = 0; i < 4; ++i) {
    ne.at(p.axes.at(i)) = a.ne(i);
    nb.at(p.axes.at(i)) = a.nb.at(i);
  }
  f.shape = ShapeOf({ne[0], ne[1], ne[2], ne[3]});
  f.stacked = 1;
  return One(ViewOf(c, 0, f, nb, a.view_offset), false);
}

Out InferCont(const Ctx& c) {
  const TensorDesc& a = *c.src(0);
  LLMP_TRY(Want(a.format.layout == format::Layout::kDense, c, "copies dense tensors"));
  Shape shape = a.shape();
  if (const TensorDesc* r = c.req(0); r != nullptr) {
    shape = r->shape();
    LLMP_TRY(Want(Elements(shape) == Elements(a.shape()), c,
                  std::format("{} has another count than {}", Text(shape), Text(a.shape()))));
  }
  return One(Fresh(shape, a.format.encoding), false);
}

Out InferConcat(const Ctx& c) {
  const auto p = c.params<ConcatParams>();
  const TensorDesc& a = *c.src(0);
  const TensorDesc& b = *c.src(1);
  if (p.dim >= 4) {
    return Fail(GraphError::kParams, c, "dim past 3");
  }
  if (a.format.layout != b.format.layout || a.format.encoding != b.format.encoding ||
      a.format.params != b.format.params) {
    return Fail(GraphError::kFormat, c, "joins tensors of matching layout and encoding");
  }
  if (a.format.layout == format::Layout::kDense) {
    // Any dimension is valid for dense tensors.
  } else if (format::IsSinglePartBlockOrDense(a.format.layout)) {
    if (p.dim == 0) {
      return Fail(GraphError::kFormat, c, "quantized concat requires a higher dimension");
    }
  } else {
    return Fail(GraphError::kFormat, c, "concat requires a dense or uniform block layout");
  }
  std::array<std::uint64_t, 4> ne = a.shape().ne;
  for (std::size_t d = 0; d < 4; ++d) {
    if (d == p.dim) {
      LLMP_TRY(Want(ne.at(d) <= std::numeric_limits<std::uint64_t>::max() - b.ne(d), c,
                    "a join past 64 bits"));
      ne.at(d) += b.ne(d);
    } else {
      LLMP_TRY(Want(
          a.ne(d) == b.ne(d), c,
          std::format("{} and {} differ beyond dim {}", Text(a.shape()), Text(b.shape()), p.dim)));
    }
  }
  format::TensorFormat f = a.format;
  f.shape = ShapeOf({ne[0], ne[1], ne[2], ne[3]});
  f.stacked = 1;
  TensorDesc r;
  r.role = Role::kActivation;
  r.format = f;
  r.nb = NaturalStrides(f);
  return One(r, false);
}

Out InferRepeat(const Ctx& c) {
  const TensorDesc& a = *c.src(0);
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a repeat needs its shape");
  }
  LLMP_TRY(Want(a.format.layout == format::Layout::kDense, c, "tiles dense tensors"));
  LLMP_TRY(Want(Broadcasts(r->shape(), a.shape()), c,
                std::format("{} is not a multiple of {}", Text(r->shape()), Text(a.shape()))));
  return One(Fresh(ShapeOf({r->ne(0), r->ne(1), r->ne(2), r->ne(3)}), a.format.encoding), false);
}

Out InferConvert(const Ctx& c) {
  const TensorDesc& a = *c.src(0);
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a cast needs its encoding");
  }
  LLMP_TRY(Want(a.format.layout == format::Layout::kDense && DenseEncoding(r->format.encoding) &&
                    r->format.layout == format::Layout::kDense,
                c, "casts a dense tensor to a dense encoding"));
  return One(Fresh(a.shape(), r->format.encoding), false);
}

Out InferGetRows(const Ctx& c) {
  LLMP_TRY(Readable(c, 0));
  LLMP_TRY(IndexSrc(c, 1));
  const TensorDesc& a = *c.src(0);
  const TensorDesc& ids = *c.src(1);
  LLMP_TRY(Want(a.ne(2) == ids.ne(1) && a.ne(3) == ids.ne(2) && ids.ne(3) == 1, c,
                std::format("rows of {} by ids {}", Text(a.shape()), Text(ids.shape()))));
  const Shape out = S(a.ne(0), ids.ne(0), ids.ne(1), ids.ne(2));
  if (a.format.encoding == Encoding::kI32) {
    return One(Fresh(out, Encoding::kI32), false);
  }
  return FloatResult(out, Encoding::kF32);
}

Out InferMatMul(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  const TensorDesc& a = *c.src(0);
  const TensorDesc& b = *c.src(1);
  LLMP_TRY(CheckProduct(c, a));
  LLMP_TRY(Want(a.ne(0) == b.ne(0) && b.ne(2) % a.ne(2) == 0 && b.ne(3) % a.ne(3) == 0, c,
                std::format("{} times {}", Text(a.shape()), Text(b.shape()))));
  return FloatResult(S(a.ne(1), b.ne(1), b.ne(2), b.ne(3)), Encoding::kF32);
}

Out InferMatMulId(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  LLMP_TRY(IndexSrc(c, 2));
  const TensorDesc& as = *c.src(0);
  const TensorDesc& b = *c.src(1);
  const TensorDesc& ids = *c.src(2);
  LLMP_TRY(CheckProduct(c, as));
  LLMP_TRY(Want(as.ne(3) == 1 && b.ne(3) == 1 && ids.ne(2) == 1 && ids.ne(3) == 1 &&
                    ids.ne(1) == b.ne(2) && as.ne(0) == b.ne(0) && ids.ne(0) % b.ne(1) == 0,
                c,
                std::format("experts {} times {} by ids {}", Text(as.shape()), Text(b.shape()),
                            Text(ids.shape()))));
  return FloatResult(S(as.ne(1), ids.ne(0), b.ne(2)), Encoding::kF32);
}

Out InferPrepare(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a preparation needs its format");
  }
  TensorDesc t;
  t.role = Role::kActivation;
  t.format = r->format;
  t.format.shape = c.src(0)->shape();
  t.format.stacked = 1;
  const bool rows = t.format.encoding == Encoding::kMxfp8 &&
                    t.format.layout == format::Layout::kScaleAtoms && t.format.shape.ne[2] == 1 &&
                    t.format.shape.ne[3] == 1;
  if (!format::Validate(t.format) || (UnitBytes(t.format) == 0 && !rows) ||
      Integer(t.format.encoding)) {
    return Fail(GraphError::kFormat, c,
                "prepares into a single-part value format of its shape, or MXFP8 rows");
  }
  t.nb = NaturalStrides(t.format);
  return One(t, false);
}

Out InferExpertSort(const Ctx& c) {
  LLMP_TRY(IndexSrc(c, 0));
  const auto p = c.params<ExpertSortParams>();
  const TensorDesc& ids = *c.src(0);
  if (p.experts == 0) {
    return Fail(GraphError::kParams, c, "no experts");
  }
  LLMP_TRY(Want(ids.ne(2) == 1 && ids.ne(3) == 1, c, "ids are [used, tokens]"));
  Inferred r;
  r.out[0] = Fresh(S(ids.ne(0) * ids.ne(1)), Encoding::kI32);
  r.out[1] = Fresh(S(std::uint64_t{p.experts} + 1), Encoding::kI32);
  r.nout = 2;
  return r;
}

Out InferAttention(const Ctx& c) {
  const auto p = c.params<AttnParams>();
  LLMP_TRY(Bool(c, 16));
  LLMP_TRY(Bool(c, 18));
  LLMP_TRY(Zero(c, 19, 1));
  if (static_cast<std::uint8_t>(p.accumulate) > static_cast<std::uint8_t>(Accumulate::kF32)) {
    return Fail(GraphError::kParams, c, "an unknown accumulation");
  }
  for (std::size_t i = 0; i < 5; ++i) {
    LLMP_TRY(FloatSrc(c, i));
  }
  const TensorDesc& q = *c.src(0);
  const TensorDesc& k = *c.src(1);
  const TensorDesc& v = *c.src(2);
  const TensorDesc* mask = c.src(3);
  const TensorDesc* sinks = c.src(4);
  if (!Finite(p.scale) || !Finite(p.softcap) || p.softcap < 0.0f || !Finite(p.max_bias) ||
      p.max_bias < 0.0f) {
    return Fail(GraphError::kParams, c, "scale, softcap or max_bias out of range");
  }
  if ((p.max_bias > 0.0f || p.window > 0 || p.n_kv_max > 0) && mask == nullptr) {
    return Fail(GraphError::kArity, c, "ALiBi, a window and a sparse bound need the mask");
  }
  if (p.sparse_any && p.n_kv_max == 0) {
    return Fail(GraphError::kParams, c, "sparse_any needs n_kv_max");
  }
  if (p.sinks != (sinks != nullptr)) {
    return Fail(GraphError::kArity, c, "sinks are source 4 exactly when the parameters say");
  }
  LLMP_TRY(Want(
      q.ne(0) == k.ne(0) && k.ne(1) == v.ne(1) && k.ne(2) == v.ne(2) && q.ne(3) == k.ne(3) &&
          q.ne(3) == v.ne(3) && q.ne(2) % k.ne(2) == 0,
      c,
      std::format("q {} over k {} and v {}", Text(q.shape()), Text(k.shape()), Text(v.shape()))));
  if (mask != nullptr) {
    LLMP_TRY(Want(mask->ne(0) == k.ne(1) && mask->ne(1) >= q.ne(1) && q.ne(2) % mask->ne(2) == 0 &&
                      q.ne(3) % mask->ne(3) == 0,
                  c,
                  std::format("a mask {} for q {} over {} cells", Text(mask->shape()),
                              Text(q.shape()), k.ne(1))));
  }
  if (sinks != nullptr) {
    LLMP_TRY(Want(sinks->shape().ne == S(q.ne(2)).ne, c, "a sink per head"));
  }
  return FloatResult(S(v.ne(0), q.ne(2), q.ne(1), q.ne(3)), Encoding::kF32);
}

Out InferMaskBuild(const Ctx& c) {
  LLMP_TRY(IndexSrc(c, 0));
  LLMP_TRY(IndexSrc(c, 1));
  const auto p = c.params<MaskParams>();
  LLMP_TRY(Bool(c, 1));
  LLMP_TRY(Zero(c, 2, 2));
  if (static_cast<std::uint8_t>(p.kind) > static_cast<std::uint8_t>(MaskKind::kAhead) ||
      (p.kind == MaskKind::kWindow && p.window == 0) ||
      ((p.kind == MaskKind::kCausal || p.kind == MaskKind::kBlock) && p.window != 0) ||
      (p.kind == MaskKind::kBlock) != (p.block != 0)) {
    return Fail(GraphError::kParams, c,
                "a window for kWindow (kAhead's optional), a block for kBlock, only");
  }
  const TensorDesc& qp = *c.src(0);
  const TensorDesc& cp = *c.src(1);
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a mask needs its rows and encoding");
  }
  LLMP_TRY(Want(Elements(qp.shape()) == qp.ne(0) && Elements(cp.shape()) == cp.ne(0), c,
                "positions are vectors"));
  const Encoding e = r->format.encoding;
  LLMP_TRY(Want(e == Encoding::kF16 || e == Encoding::kF32, c, "a mask is F16 or F32"));
  LLMP_TRY(Want(
      r->ne(0) == cp.ne(0) && r->ne(1) >= qp.ne(0) && (!p.exact_rows || r->ne(1) == qp.ne(0)) &&
          r->ne(2) == 1 && r->ne(3) == 1,
      c, std::format("a mask {} for {} cells and {} rows", Text(r->shape()), cp.ne(0), qp.ne(0))));
  return One(Fresh(S(cp.ne(0), r->ne(1)), e), false);
}

Out InferCacheStore(const Ctx& c) {
  LLMP_TRY(Readable(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  LLMP_TRY(IndexSrc(c, 2, true));
  LLMP_TRY(Writable(c, 0));
  const TensorDesc& cache = *c.src(0);
  const TensorDesc& src = *c.src(1);
  const TensorDesc& idx = *c.src(2);
  if (UnitBytes(cache.format) == 0 || Integer(cache.format.encoding)) {
    return Fail(GraphError::kFormat, c, "stores into a single-part value format");
  }
  LLMP_TRY(Want(cache.ne(0) == src.ne(0) && cache.ne(2) == src.ne(2) && cache.ne(3) == src.ne(3) &&
                    src.ne(1) == idx.ne(0) && src.ne(2) % idx.ne(1) == 0 &&
                    src.ne(3) % idx.ne(2) == 0 && idx.ne(3) == 1,
                c,
                std::format("rows {} into {} at {}", Text(src.shape()), Text(cache.shape()),
                            Text(idx.shape()))));
  return One(ViewOf(c, 0, cache.format, cache.nb, cache.view_offset), false);
}

Out InferRoundedIndexScore(const Ctx& c) {
  const auto p = c.params<RoundedIndexScoreParams>();
  LLMP_TRY(Zero(c, 1, 3));
  if (p.query != QueryRounding::kF16) return Fail(GraphError::kParams, c, "F16 query rounding");
  const TensorDesc& q = *c.src(0);
  const TensorDesc& k = *c.src(1);
  const TensorDesc& w = *c.src(2);
  const TensorDesc& counts = *c.src(3);
  LLMP_TRY(Want(q.format.encoding == Encoding::kF32 && q.ne(0) > 0 && q.ne(1) > 0 && q.ne(3) == 1 &&
                    k.format.encoding == Encoding::kF16 && k.ne(0) == q.ne(0) && k.ne(1) == 1 &&
                    k.ne(2) <= INT32_MAX && k.ne(3) == 1 && w.format.encoding == Encoding::kF32 &&
                    w.shape().ne == S(q.ne(1), q.ne(2)).ne &&
                    counts.format.encoding == Encoding::kI32 && counts.shape().ne == S(q.ne(2)).ne,
                c, "Q F32 [D,H,T], K F16 [D,1,C], W F32 [H,T], counts I32 [T]"));
  return One(Fresh(S(k.ne(2), q.ne(2)), Encoding::kF32), false);
}

Out InferPrefixOverwrite(const Ctx& c) {
  const TensorDesc& x = *c.src(0);
  const TensorDesc& counts = *c.src(1);
  LLMP_TRY(Want(x.format.encoding == Encoding::kF32 && x.ne(0) <= INT32_MAX && x.ne(2) == 1 &&
                    x.ne(3) == 1 && counts.format.encoding == Encoding::kI32 &&
                    counts.shape().ne == S(x.ne(1)).ne,
                c, "F32 [C,T] values and I32 [T] counts, C representable as I32"));
  return One(Fresh(x.shape(), Encoding::kF32), false);
}

Out InferIndexScore(const Ctx& c) {
  for (std::size_t i = 0; i < 4; ++i) {
    LLMP_TRY(FloatSrc(c, i));
  }
  const TensorDesc& q = *c.src(0);
  const TensorDesc& k = *c.src(1);
  const TensorDesc& w = *c.src(2);
  const TensorDesc* mask = c.src(3);
  LLMP_TRY(
      Want(q.ne(0) == k.ne(0) && k.ne(1) == 1 && k.ne(3) == q.ne(3) && w.ne(0) == q.ne(1) &&
               w.ne(1) == q.ne(2) && w.ne(2) == 1 && w.ne(3) == q.ne(3),
           c, std::format("q {}, k {}, w {}", Text(q.shape()), Text(k.shape()), Text(w.shape()))));
  if (mask != nullptr) {
    LLMP_TRY(Want(mask->ne(0) == k.ne(2) && mask->ne(1) == q.ne(2) && mask->ne(2) == 1 &&
                      q.ne(3) % mask->ne(3) == 0,
                  c, std::format("a mask {}", Text(mask->shape()))));
  }
  return FloatResult(S(k.ne(2), q.ne(2), 1, q.ne(3)), Encoding::kF32);
}

Out InferPool(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  const auto p = c.params<PoolParams>();
  LLMP_TRY(Zero(c, 5, 3));
  if (p.ratio == 0 || static_cast<std::uint8_t>(p.kind) > 1) {
    return Fail(GraphError::kParams, c, "a ratio of at least one, a known kind");
  }
  const TensorDesc& x = *c.src(0);
  const TensorDesc* scores = c.src(1);
  if ((p.kind == PoolKind::kSoftmax) != (scores != nullptr)) {
    return Fail(GraphError::kArity, c, "scores are kSoftmax's, only");
  }
  LLMP_TRY(Want(x.ne(1) % p.ratio == 0, c, "whole runs of rows"));
  if (scores != nullptr) {
    LLMP_TRY(Want(scores->shape().ne == x.shape().ne, c, "scores have x's shape"));
  }
  return FloatResult(S(x.ne(0), x.ne(1) / p.ratio, x.ne(2), x.ne(3)), Encoding::kF32);
}

Out InferNorm(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(NormEps(c, c.params<NormParams>().eps));
  return FloatResult(c.src(0)->shape(), FloatOf(*c.src(0)));
}

Out InferRope(const Ctx& c) {
  const auto p = c.params<RopeParams>();
  LLMP_TRY(CheckRope(c, p, 0));
  return RopeOf(c, p, 0, 1, 2);
}

Out InferSelect(const Ctx& c) {
  LLMP_TRY(IndexSrc(c, 0));
  LLMP_TRY(Readable(c, 1));
  LLMP_TRY(Readable(c, 2));
  const auto& condition = *c.src(0);
  const auto& yes = *c.src(1);
  const auto& no = *c.src(2);
  LLMP_TRY(Want(
      yes.format.layout == format::Layout::kDense && no.format.layout == format::Layout::kDense &&
          yes.format.encoding == no.format.encoding &&
          static_cast<unsigned>(yes.format.encoding) <= static_cast<unsigned>(Encoding::kE8M0),
      c, "select branches need the same dense scalar encoding"));
  LLMP_TRY(Want(Broadcasts(yes.shape(), condition.shape()) && Broadcasts(yes.shape(), no.shape()),
                c, "select predicate and false branch must broadcast over true branch"));
  return One(Fresh(yes.shape(), yes.format.encoding), false);
}

Out InferBinary(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  const TensorDesc& a = *c.src(0);
  const TensorDesc& b = *c.src(1);
  LLMP_TRY(Want(Broadcasts(a.shape(), b.shape()), c,
                std::format("{} does not broadcast over {}", Text(b.shape()), Text(a.shape()))));
  return FloatResult(a.shape(), FloatOf(a));
}

Out InferScale(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  const auto p = c.params<ScaleParams>();
  if (!Finite(p.scale) || !Finite(p.bias)) {
    return Fail(GraphError::kParams, c, "scale and bias must be finite");
  }
  return FloatResult(c.src(0)->shape(), FloatOf(*c.src(0)));
}
Out InferUnary(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  if (static_cast<std::uint8_t>(c.params<UnaryParams>().fn) >
      static_cast<std::uint8_t>(UnaryFn::kSgn)) {
    return Fail(GraphError::kParams, c, "an unknown function");
  }
  return FloatResult(c.src(0)->shape(), FloatOf(*c.src(0)));
}
Out InferClamp(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));

  const auto p = c.params<ClampParams>();
  if (!(p.min <= p.max)) {
    return Fail(GraphError::kParams, c, "min above max");
  }

  return FloatResult(c.src(0)->shape(), FloatOf(*c.src(0)));
}
Out InferSoftcap(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));

  const float cap = c.params<SoftcapParams>().cap;
  if (!Finite(cap) || cap <= 0.0f) {
    return Fail(GraphError::kParams, c, "a cap above zero");
  }

  return FloatResult(c.src(0)->shape(), FloatOf(*c.src(0)));
}

Out InferFill(const Ctx& c) {
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a fill needs its shape");
  }
  LLMP_TRY(Want(ResultEncoding(r->format.encoding), c, "fills F32, F16 or BF16"));
  return One(Fresh(ShapeOf({r->ne(0), r->ne(1), r->ne(2), r->ne(3)}), r->format.encoding), false);
}

Out InferGlu(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  const auto p = c.params<GluParams>();
  LLMP_TRY(Bool(c, 1));
  LLMP_TRY(Zero(c, 2, 2));
  if (static_cast<std::uint8_t>(p.kind) > static_cast<std::uint8_t>(GluKind::kSwiGluClamp) ||
      !Finite(p.limit) || (p.kind == GluKind::kSwiGluClamp) != (p.limit > 0.0f)) {
    return Fail(GraphError::kParams, c, "a known kind; a limit above zero for the clamp only");
  }
  const TensorDesc& g = *c.src(0);
  if (const TensorDesc* u = c.src(1); u != nullptr) {
    LLMP_TRY(Want(u->shape().ne == g.shape().ne && !p.swapped, c,
                  "gate and up of one shape (swapped is the halves')"));
    return FloatResult(g.shape(), FloatOf(g));
  }
  LLMP_TRY(Want(g.ne(0) % 2 == 0, c, "halves of even rows"));
  return FloatResult(S(g.ne(0) / 2, g.ne(1), g.ne(2), g.ne(3)), FloatOf(g));
}

Out InferSumRows(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  const TensorDesc& x = *c.src(0);
  return FloatResult(S(1, x.ne(1), x.ne(2), x.ne(3)), FloatOf(x));
}

Out InferSoftMax(const Ctx& c) {
  for (std::size_t i = 0; i < 3; ++i) {
    LLMP_TRY(FloatSrc(c, i));
  }
  const auto p = c.params<SoftMaxParams>();
  if (!Finite(p.scale) || !Finite(p.max_bias) || p.max_bias < 0.0f) {
    return Fail(GraphError::kParams, c, "scale or max_bias out of range");
  }
  const TensorDesc& x = *c.src(0);
  const TensorDesc* mask = c.src(1);
  if (p.max_bias > 0.0f && mask == nullptr) {
    return Fail(GraphError::kArity, c, "ALiBi needs the mask");
  }
  if (mask != nullptr) {
    LLMP_TRY(Want(mask->ne(0) == x.ne(0) && mask->ne(1) >= x.ne(1) && x.ne(2) % mask->ne(2) == 0 &&
                      x.ne(3) % mask->ne(3) == 0,
                  c, std::format("a mask {} over {}", Text(mask->shape()), Text(x.shape()))));
  }
  if (const TensorDesc* sinks = c.src(2); sinks != nullptr) {
    LLMP_TRY(Want(sinks->shape().ne == S(x.ne(2)).ne, c, "a sink per matrix"));
  }
  return FloatResult(x.shape(), FloatOf(x));
}

Out InferArgmax(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  if (c.byte(0) > static_cast<std::uint8_t>(ArgmaxNan::kFirstColumn)) {
    return Fail(GraphError::kParams, c, "an unknown NaN rule");
  }
  const TensorDesc& x = *c.src(0);
  LLMP_TRY(Want(x.ne(2) == 1 && x.ne(3) == 1, c, "a matrix"));
  return One(Fresh(S(x.ne(1)), Encoding::kI32), false);
}

Out InferArgsort(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(Bool(c, 0));
  return One(Fresh(c.src(0)->shape(), Encoding::kI32), false);
}

Out InferTopK(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  const auto p = c.params<TopKParams>();
  const TensorDesc& x = *c.src(0);
  if (p.k == 0 || p.k > x.ne(0)) {
    return Fail(GraphError::kParams, c, std::format("k {} of {}", p.k, x.ne(0)));
  }
  return One(Fresh(S(p.k, x.ne(1), x.ne(2), x.ne(3)), Encoding::kI32), false);
}

Out InferVisibleTopK(const Ctx& c) {
  const auto p = c.params<TopKParams>();
  const TensorDesc& x = *c.src(0);
  const TensorDesc& counts = *c.src(1);
  if (p.k == 0 || p.k > INT32_MAX) {
    return Fail(GraphError::kParams, c, "a positive I32 selection width");
  }
  LLMP_TRY(Want(x.format.encoding == Encoding::kF32 && x.ne(0) <= INT32_MAX && x.ne(2) == 1 &&
                    x.ne(3) == 1 && counts.format.encoding == Encoding::kI32 &&
                    counts.shape().ne == S(x.ne(1)).ne,
                c, "F32 [C,T] scores and I32 [T] counts, C representable as I32"));
  return One(Fresh(S(p.k, x.ne(1)), Encoding::kI32), false);
}

Out InferSelectedMask(const Ctx& c) {
  const auto p = c.params<SelectedMaskParams>();
  const TensorDesc& raw = *c.src(0);
  const TensorDesc& ids = *c.src(1);
  if (p.raw_cells == 0 || p.cells == 0 || p.cells > INT32_MAX) {
    return Fail(GraphError::kParams, c, "positive raw and compressed widths, I32 cells");
  }
  LLMP_TRY(Want(raw.format.encoding == Encoding::kF16 && raw.ne(0) <= p.raw_cells &&
                    raw.ne(2) == 1 && raw.ne(3) == 1 && ids.format.encoding == Encoding::kI32 &&
                    ids.ne(1) == raw.ne(1) && ids.ne(0) <= INT32_MAX && ids.ne(2) == 1 &&
                    ids.ne(3) == 1,
                c, "F16 [window,T] raw mask and I32 [k,T] IDs"));
  return One(Fresh(S(std::uint64_t{p.raw_cells} + p.cells, raw.ne(1)), Encoding::kF16), false);
}

Out InferRoute(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  LLMP_TRY(IndexSrc(c, 2));
  LLMP_TRY(IndexSrc(c, 3));
  const auto p = c.params<RouteParams>();
  LLMP_TRY(Bool(c, 1));
  LLMP_TRY(Zero(c, 2, 2));
  const TensorDesc& logits = *c.src(0);
  if (static_cast<std::uint8_t>(p.kind) > static_cast<std::uint8_t>(RouteKind::kHash) ||
      p.used == 0 || p.used > p.experts || !Finite(p.scale) || p.scale <= 0.0f) {
    return Fail(GraphError::kParams, c,
                "a known kind, 1 <= used <= experts, a finite scale above zero (an unset "
                "scale is 1)");
  }
  LLMP_TRY(Want(logits.ne(0) == p.experts && logits.ne(2) == 1 && logits.ne(3) == 1, c,
                std::format("logits {} for {} experts", Text(logits.shape()), p.experts)));
  const TensorDesc* bias = c.src(1);
  const TensorDesc* table = c.src(2);
  const TensorDesc* tokens = c.src(3);
  const bool wants_bias = p.kind == RouteKind::kSqrtSoftplusBias;
  const bool may_bias = wants_bias || p.kind == RouteKind::kSigmoidTopK;
  const bool hash = p.kind == RouteKind::kHash;
  if ((bias != nullptr && !may_bias) || (wants_bias && bias == nullptr) ||
      hash != (table != nullptr) || (!hash && tokens != nullptr)) {
    return Fail(GraphError::kArity, c,
                "the kind's sources: a bias, or a table (and the tokens it is read by)");
  }
  if (bias != nullptr) {
    LLMP_TRY(Want(bias->shape().ne == S(p.experts).ne, c, "a bias per expert"));
  }
  if (hash && tokens != nullptr) {
    LLMP_TRY(Want(table->ne(0) == p.used && table->ne(2) == 1 && table->ne(3) == 1 &&
                      tokens->shape().ne == S(logits.ne(1)).ne,
                  c, "a table [used, vocabulary] and a token per row"));
  } else if (hash) {
    LLMP_TRY(Want(
        table->shape().ne == S(p.used, logits.ne(1)).ne && table->format.encoding == Encoding::kI32,
        c, "ids [used, rows] I32 without tokens"));
  }
  Inferred r;
  r.out[0] = Fresh(S(p.used, logits.ne(1)), Encoding::kI32);
  r.out[1] = Fresh(S(p.used, logits.ne(1)), Encoding::kF32);
  r.nout = 2;
  r.float_result[1] = true;
  return r;
}

Out InferExpertReduce(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  LLMP_TRY(IndexSrc(c, 2));
  LLMP_TRY(FloatSrc(c, 3));
  const TensorDesc& x = *c.src(0);
  const TensorDesc& w = *c.src(1);
  const TensorDesc* ids = c.src(2);
  const TensorDesc* scale = c.src(3);
  LLMP_TRY(Want(x.ne(3) == 1 && w.shape().ne == S(x.ne(1), x.ne(2)).ne, c,
                std::format("experts {} with weights {}", Text(x.shape()), Text(w.shape()))));
  const bool gathered = scale != nullptr && scale->ne(0) == 1 && scale->ne(1) == x.ne(1) &&
                        scale->ne(2) == x.ne(2) && scale->ne(3) == 1;
  if ((scale != nullptr) && !gathered && (ids == nullptr)) {
    return Fail(GraphError::kArity, c, "a scale per expert needs the ids");
  }
  if (ids != nullptr) {
    LLMP_TRY(Want(ids->shape().ne == w.shape().ne, c, "ids have the weights' shape"));
  }
  if (scale != nullptr && !gathered) {
    LLMP_TRY(Want(Elements(scale->shape()) == scale->ne(0), c,
                  "a scale vector, or a scale gathered per slot [1, used, rows]"));
  }
  return FloatResult(S(x.ne(0), x.ne(2)), Encoding::kF32);
}

Out InferSsmConv(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  const TensorDesc& sx = *c.src(0);
  const TensorDesc& w = *c.src(1);
  LLMP_TRY(Want(
      sx.ne(3) == 1 && w.ne(2) == 1 && w.ne(3) == 1 && sx.ne(1) == w.ne(1) && sx.ne(0) >= w.ne(0),
      c, std::format("{} over taps {}", Text(sx.shape()), Text(w.shape()))));
  return FloatResult(S(w.ne(1), sx.ne(0) - w.ne(0) + 1, sx.ne(2)), Encoding::kF32);
}

Out InferGatedDeltaNet(const Ctx& c) {
  for (std::size_t i = 0; i < 6; ++i) {
    LLMP_TRY(FloatSrc(c, i));
  }
  const TensorDesc& q = *c.src(0);
  const TensorDesc& k = *c.src(1);
  const TensorDesc& v = *c.src(2);
  const TensorDesc& g = *c.src(3);
  const TensorDesc& beta = *c.src(4);
  const TensorDesc& state = *c.src(5);
  const std::uint64_t sk = q.ne(0);
  const std::uint64_t sv = v.ne(0);
  const std::uint64_t h = v.ne(1);
  const std::uint64_t t = v.ne(2);
  const std::uint64_t b = v.ne(3);
  LLMP_TRY(Want(k.shape().ne == q.shape().ne && q.ne(2) == t && h % q.ne(1) == 0 &&
                    b % q.ne(3) == 0 && (g.ne(0) == 1 || g.ne(0) == sk) && g.ne(1) == h &&
                    g.ne(2) == t && g.ne(3) == b && beta.shape().ne == S(1, h, t, b).ne &&
                    state.shape().ne == S(sk, sv, h, b).ne,
                c,
                std::format("q {}, v {}, g {}, beta {}, state {}", Text(q.shape()), Text(v.shape()),
                            Text(g.shape()), Text(beta.shape()), Text(state.shape()))));
  Inferred r;
  r.out[0] = Fresh(S(sv, h, t, b), Encoding::kF32);
  r.out[1] = Fresh(S(sk, sv, h, b), Encoding::kF32);
  r.nout = 2;
  r.float_result = {true, true, false, false};
  return r;
}

Out InferConv2d(const Ctx& c) {
  for (std::size_t i = 0; i < 3; ++i) {
    LLMP_TRY(FloatSrc(c, i));
  }
  const auto p = c.params<Conv2dParams>();
  const TensorDesc& w = *c.src(0);
  const TensorDesc& x = *c.src(1);
  if (p.stride == 0 || p.dilation == 0) {
    return Fail(GraphError::kParams, c, "stride and dilation at least one");
  }
  LLMP_TRY(Want(w.ne(2) == x.ne(2), c,
                std::format("kernel {} over {}", Text(w.shape()), Text(x.shape()))));
  const auto size = [&](std::uint64_t in, std::uint64_t k) -> std::uint64_t {
    const std::uint64_t span = (std::uint64_t{p.dilation} * (k - 1)) + 1;
    const std::uint64_t padded = in + (2ULL * p.padding);
    return padded < span ? 0 : ((padded - span) / p.stride) + 1;
  };
  const std::uint64_t ow = size(x.ne(0), w.ne(0));
  const std::uint64_t oh = size(x.ne(1), w.ne(1));
  LLMP_TRY(Want(ow > 0 && oh > 0, c, "the kernel is wider than the padded input"));
  if (const TensorDesc* bias = c.src(2); bias != nullptr) {
    LLMP_TRY(Want(bias->shape().ne == S(w.ne(3)).ne, c, "a bias per output channel"));
  }
  return FloatResult(S(ow, oh, w.ne(3), x.ne(3)), Encoding::kF32);
}

Out InferUpsample(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  const auto f = c.params<UpsampleParams>().factor;
  if (f == 0) {
    return Fail(GraphError::kParams, c, "a factor of at least one");
  }
  const TensorDesc& x = *c.src(0);
  LLMP_TRY(Want(x.ne(0) <= std::numeric_limits<std::uint64_t>::max() / f &&
                    x.ne(1) <= std::numeric_limits<std::uint64_t>::max() / f,
                c, "an upsampled shape past 64 bits"));
  return FloatResult(S(x.ne(0) * f, x.ne(1) * f, x.ne(2), x.ne(3)), FloatOf(x));
}

std::expected<void, VerifyError> ValidateHc(const Ctx& c) {
  const auto p = c.params<HcParams>();
  LLMP_TRY(Zero(c, 2, 2));
  if (static_cast<std::uint8_t>(p.mix) > 1 || static_cast<std::uint8_t>(p.fold) > 1 ||
      p.streams == 0 || !Finite(p.eps) || !Finite(p.scale)) {
    return Fail(GraphError::kParams, c, "known kinds, streams, finite eps and scale");
  }
  return {};
}
Out InferHcPre(const Ctx& c) {
  LLMP_TRY(ValidateHc(c));
  const auto p = c.params<HcParams>();
  const std::uint64_t hc = p.streams;

  LLMP_TRY(FloatSrc(c, 0));
  LLMP_TRY(FloatSrc(c, 1));
  const TensorDesc& x = *c.src(0);
  const TensorDesc& w = *c.src(1);
  const std::uint64_t t = x.ne(2);
  LLMP_TRY(Want(x.ne(1) == hc && x.ne(3) == 1, c, "x is [width, streams, tokens]"));
  if (p.fold == HcFold::kWeights) {
    LLMP_TRY(Want(w.shape().ne == S(hc, t).ne, c, "a weight per stream and token"));
  } else {
    LLMP_TRY(Want(w.shape().ne == x.shape().ne, c, "a gate per value"));
  }
  return FloatResult(S(x.ne(0), t), Encoding::kF32);
}
Out InferHcPost(const Ctx& c) {
  LLMP_TRY(ValidateHc(c));
  const auto p = c.params<HcParams>();
  const std::uint64_t hc = p.streams;

  for (std::size_t i = 0; i < 4; ++i) {
    LLMP_TRY(FloatSrc(c, i));
  }
  const TensorDesc& y = *c.src(0);
  const TensorDesc& x = *c.src(1);
  const TensorDesc& post = *c.src(2);
  const std::uint64_t w = y.ne(0);
  const std::uint64_t t = y.ne(1);
  LLMP_TRY(Want(
      y.ne(2) == 1 && y.ne(3) == 1 && x.shape().ne == S(w, hc, t).ne &&
          post.shape().ne == S(hc, t).ne,
      c, std::format("y {}, x {}, post {}", Text(y.shape()), Text(x.shape()), Text(post.shape()))));
  if (const TensorDesc* comb = c.src(3); comb != nullptr) {
    LLMP_TRY(Want(comb->shape().ne == S(hc, hc, t).ne, c, "comb is [hc, hc, tokens]"));
  }
  return FloatResult(S(w, hc, t), Encoding::kF32);
}
Out InferHcMix(const Ctx& c) {
  LLMP_TRY(ValidateHc(c));
  const auto p = c.params<HcParams>();
  const std::uint64_t hc = p.streams;

  for (std::size_t i = 0; i < 3; ++i) {
    LLMP_TRY(FloatSrc(c, i));
  }
  const TensorDesc& mixes = *c.src(0);
  const TensorDesc& scale = *c.src(1);
  const TensorDesc& base = *c.src(2);
  const std::uint64_t m = mixes.ne(0);
  const std::uint64_t t = mixes.ne(1);
  const std::uint64_t off = p.mix == HcMixKind::kSinkhorn
                                ? (p.mix_offset != 0 ? p.mix_offset : (m == hc * hc ? 0 : 2 * hc))
                                : p.mix_offset;
  const std::uint64_t sc_idx =
      p.mix == HcMixKind::kSinkhorn
          ? (p.scale_index != 0 ? p.scale_index : (scale.ne(0) == 1 ? 0 : 2))
          : p.scale_index;
  const std::uint64_t need_m = off + (p.mix == HcMixKind::kSinkhorn ? hc * hc : 2 * hc);
  const std::uint64_t need_sc = sc_idx + (p.mix == HcMixKind::kSinkhorn ? 1 : 2);
  LLMP_TRY(Want(mixes.shape().ne == S(m, t).ne && m >= need_m &&
                    Elements(scale.shape()) == scale.ne(0) && scale.ne(0) >= need_sc &&
                    base.shape().ne == S(m).ne,
                c,
                std::format("mixes {}, scale {}, base {} for {} streams", Text(mixes.shape()),
                            Text(scale.shape()), Text(base.shape()), hc)));
  if (p.mix == HcMixKind::kSinkhorn) {
    if (p.iterations == 0) {
      return Fail(GraphError::kParams, c, "Sinkhorn needs iterations");
    }
    return FloatResult(S(hc, hc, t), Encoding::kF32);
  }
  return FloatResult(S(hc, 2, t), Encoding::kF32);
}

Out InferSample(const Ctx& c) {
  LLMP_TRY(FloatSrc(c, 0));
  const auto p = c.params<SampleParams>();
  LLMP_TRY(Zero(c, 1, 3));
  const TensorDesc& logits = *c.src(0);
  const TensorDesc* keys = c.src(1);
  if (static_cast<std::uint8_t>(p.kind) > 1 || !Finite(p.temperature) || p.temperature < 0.0f ||
      !(p.top_p > 0.0f && p.top_p <= 1.0f) || !(p.min_p >= 0.0f && p.min_p <= 1.0f)) {
    return Fail(GraphError::kParams, c, "sampling parameters out of range");
  }
  if ((p.kind == SampleKind::kSeeded) != (keys != nullptr)) {
    return Fail(GraphError::kArity, c, "keys are the seeded kind's, only");
  }
  LLMP_TRY(Want(logits.ne(2) == 1 && logits.ne(3) == 1, c, "logits are [vocabulary, rows]"));
  if (keys != nullptr) {
    if (keys->format.encoding != Encoding::kI64 || keys->format.layout != format::Layout::kDense) {
      return Fail(GraphError::kFormat, c, "keys are I64");
    }
    LLMP_TRY(Want(keys->shape().ne == S(3, logits.ne(1)).ne, c, "keys are [3, rows]"));
  }
  return One(Fresh(S(logits.ne(1)), Encoding::kI32), false);
}
std::expected<void, VerifyError> ValidateCollective(const Ctx& c) {
  const auto p = c.params<CollectiveParams>();
  LLMP_TRY(Zero(c, 9, 3));
  if (p.dim >= 4) {
    return Fail(GraphError::kParams, c, "dim past 3");
  }
  return {};
}
Out InferRecv(const Ctx& c) {
  LLMP_TRY(ValidateCollective(c));

  const TensorDesc* r = c.req(0);
  if (r == nullptr || r->format.layout != format::Layout::kDense || !format::Validate(r->format)) {
    return Fail(GraphError::kArity, c, "a receive needs its dense format");
  }
  return One(Fresh(r->shape(), r->format.encoding), false);
}
Out InferSend(const Ctx& c) {
  LLMP_TRY(ValidateCollective(c));

  LLMP_TRY(Readable(c, 0));
  const TensorDesc& x = *c.src(0);
  LLMP_TRY(Want(x.format.layout == format::Layout::kDense, c, "moves dense tensors"));
  return Inferred{};
}
Out InferAllGather(const Ctx& c) {
  LLMP_TRY(ValidateCollective(c));
  const auto p = c.params<CollectiveParams>();

  LLMP_TRY(Readable(c, 0));
  const TensorDesc& x = *c.src(0);
  LLMP_TRY(Want(x.format.layout == format::Layout::kDense, c, "moves dense tensors"));

  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a gather needs its shape");
  }
  bool ok = r->ne(p.dim) % x.ne(p.dim) == 0;
  for (std::size_t d = 0; d < 4; ++d) {
    ok = ok && (d == p.dim || r->ne(d) == x.ne(d));
  }
  LLMP_TRY(Want(ok, c, "gathers along dim a multiple of x"));
  return One(Fresh(ShapeOf({r->ne(0), r->ne(1), r->ne(2), r->ne(3)}), x.format.encoding), false);
}
Out InferCollectiveCopy(const Ctx& c) {
  LLMP_TRY(ValidateCollective(c));

  LLMP_TRY(Readable(c, 0));
  const TensorDesc& x = *c.src(0);
  LLMP_TRY(Want(x.format.layout == format::Layout::kDense, c, "moves dense tensors"));

  return One(Fresh(x.shape(), x.format.encoding), false);
}

// The byte offsets InferFused's flag and reserved-byte checks name.
static_assert(offsetof(ProductParams, reserved) == 2 && offsetof(HcParams, reserved) == 2);
static_assert(offsetof(GluParams, swapped) == 1 && offsetof(GluParams, reserved) == 2 &&
              offsetof(MoeExpertsParams, glu) == 0);
static_assert(offsetof(RouteParams, normalize) == 1 && offsetof(RouteParams, reserved) == 2);
static_assert(offsetof(AttnParams, sinks) == 16 && offsetof(AttnParams, sparse_any) == 18 &&
              offsetof(AttnParams, reserved) == 19 && offsetof(OwnerAttnParams, attn) == 0 &&
              offsetof(BlockSparseAttnParams, attn) == 0);
static_assert(offsetof(SpatialNormParams, silu) == 8 && offsetof(SpatialNormParams, reserved) == 9);
static_assert(offsetof(RopeSectionsParams, sections) == 9 &&
              offsetof(RopeSectionsParams, reserved) == 10);

// Parameter layout checks run before any decomposition reads boolean fields.
using Checked = std::expected<void, VerifyError>;
Checked ValidateNone(const Ctx&) { return {}; }
Checked ValidateRope(const Ctx& c) { return CheckRope(c, c.params<RopeParams>(), 0); }
Checked ValidateNormRope(const Ctx& c) { return CheckRope(c, c.params<NormRopeParams>().rope, 4); }
Checked ValidatePoolRope(const Ctx& c) { return CheckRope(c, c.params<BlockPoolParams>().rope, 8); }
Checked ValidateAttention(const Ctx& c) {
  LLMP_TRY(Bool(c, 16));
  LLMP_TRY(Bool(c, 18));
  return Zero(c, 19, 1);
}
Checked ValidateCompressor(const Ctx& c) {
  LLMP_TRY(Bool(c, 4));
  return Zero(c, 5, 3);
}
Checked ValidateHcFold(const Ctx& c) {
  LLMP_TRY(Bool(c, 4));
  LLMP_TRY(Bool(c, 5));
  LLMP_TRY(Zero(c, 6, 2));
  return NormEps(c, c.params<HcFoldParams>().eps);
}
Checked ValidateGdnStep(const Ctx& c) {
  LLMP_TRY(Bool(c, 0));
  return Zero(c, 1, 3);
}
Checked ValidateScaledUnary(const Ctx& c) { return Zero(c, 5, 3); }
Checked ValidateProduct(const Ctx& c) { return CheckProduct(c, *c.src(0)); }
Checked ValidateHcCombine(const Ctx& c) {
  LLMP_TRY(Zero(c, 2, 2));
  const auto p = c.params<HcParams>();
  if (static_cast<std::uint8_t>(p.mix) > 1 || static_cast<std::uint8_t>(p.fold) > 1 ||
      !Finite(p.eps) || !Finite(p.scale)) {
    return Fail(GraphError::kParams, c, "known kinds, finite eps and scale");
  }
  return {};
}
Checked ValidateGdnConv(const Ctx& c) {
  const auto p = c.params<GdnConvParams>();
  if (!(p.eps > 0.0f) || !Finite(p.eps) || !Finite(p.scale)) {
    return Fail(GraphError::kParams, c, "positive finite eps and finite scale");
  }
  return {};
}
Checked ValidateSplitModulate(const Ctx& c) {
  return NormEps(c, c.params<SplitModulateParams>().eps);
}
Checked ValidateFlag1Reserved2(const Ctx& c) {
  LLMP_TRY(Bool(c, 1));
  return Zero(c, 2, 2);
}
Checked ValidateSpatialNorm(const Ctx& c) {
  LLMP_TRY(Bool(c, 8));
  return Zero(c, 9, 3);
}
Checked ValidateRopeSections(const Ctx& c) { return Zero(c, 10, 2); }
Out InferInvalid(const Ctx& c) { return Fail(GraphError::kParams, c, "an op outside op set v1"); }
Out InferBatchKey(const Ctx& c) {
  return Fail(GraphError::kParams, c, "grouped_stores is a batch key, not a node");
}

// A fused node's computed outputs are its decomposition's over stand-ins
// of its sources, so they depend only on its key, its parameters and what
// the stand-ins carry of each source: its format and, for a strided view
// (a permuted or sliced source; L3), its strides, its offset and its
// root's format. A packed source's stand-in is a root of its format.
struct StandIn {
  bool present = false;
  format::TensorFormat format{};
  bool strided = false;
  std::array<std::uint64_t, format::kMaxDims> nb{};
  std::uint64_t view_offset = 0;
  format::TensorFormat root{};
  bool operator==(const StandIn&) const = default;
};

StandIn StandInOf(const Ctx& c, std::size_t i) {
  StandIn s;
  const TensorDesc* t = c.src(i);
  if (t == nullptr) return s;
  s.present = true;
  s.format = t->format;
  if (!t->is_view() || IsContiguous(*t)) return s;
  // A view the stand-in can be: one that steps by one unit along ne[0]
  // (a transposed one stays a packed stand-in, as before).
  const TensorDesc& root = c.root_desc(i);
  const std::uint64_t unit = UnitBytes(root.format);
  const std::uint64_t per = UnitElements(root.format);
  if (unit == 0 || per == 0 || (t->ne(0) != per && t->nb[0] != unit)) return s;
  s.strided = true;
  s.nb = t->nb;
  s.view_offset = t->view_offset;
  s.root = root.format;
  return s;
}

struct FusedQuery {
  OpKey key{};
  ParamBytes params{};
  std::uint8_t count = 0;
  std::array<StandIn, kMaxSources> sources{};
  bool operator==(const FusedQuery&) const = default;
};

struct FusedQueryHash {
  static void Mix(std::uint64_t& h, std::uint64_t v) {
    for (std::size_t i = 0; i < 8; ++i) {
      h ^= (v >> (8 * i)) & 0xFFU;
      h *= 0x100000001b3ULL;
    }
  }
  static void Mix(std::uint64_t& h, const format::TensorFormat& f) {
    Mix(h, static_cast<std::uint64_t>(f.encoding));
    Mix(h, static_cast<std::uint64_t>(f.layout));
    Mix(h, f.stacked);
    for (const std::uint64_t n : f.shape.ne) Mix(h, n);
  }
  std::size_t operator()(const FusedQuery& q) const noexcept {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    Mix(h, static_cast<std::uint64_t>(q.key.op));
    Mix(h, static_cast<std::uint64_t>(q.key.fused));
    for (const std::byte b : q.params) Mix(h, static_cast<std::uint64_t>(b));
    Mix(h, q.count);
    for (std::size_t i = 0; i < q.count; ++i) {
      const StandIn& s = q.sources.at(i);
      Mix(h, s.present ? 1 : 0);
      Mix(h, s.format);
      if (s.strided) {
        for (const std::uint64_t b : s.nb) Mix(h, b);
        Mix(h, s.view_offset);
        Mix(h, s.root);
      }
    }
    return static_cast<std::size_t>(h);
  }
};

using FusedResult = std::expected<Inferred, VerifyError>;

// The decomposition's outputs over the query's stand-ins; an output that
// aliases a source is left for the node's own source (InferFused).
FusedResult DecomposedOutputs(const FusedQuery& q) {
  Builder b;
  std::array<TensorId, kMaxSources> ins{};
  for (std::size_t i = 0; i < q.count; ++i) {
    const StandIn& s = q.sources.at(i);
    if (!s.present) continue;
    if (s.strided) {
      ins.at(i) = b.View(b.Input(s.root), s.format.shape, s.nb, s.view_offset);
    } else {
      ins.at(i) = b.Input(s.format);
    }
  }
  const Outputs outs = Decompose(b, q.key, q.params, std::span(ins).first(q.count), {});
  if (!b.ok()) {
    VerifyError e = *b.error();
    e.detail = std::format("{} (in its decomposition)", e.detail);
    return std::unexpected(e);
  }
  Inferred r;
  r.nout = outs.n;
  for (std::size_t i = 0; i < outs.n; ++i) {
    if (AliasedSource(q.key, i)) continue;
    const TensorDesc t = b.tensor(outs[i]);
    r.out.at(i) = Fresh(t.shape(), t.format.encoding);
    if (PreparedRows(t.format)) {
      // Rows prepared for a tensor-core product (a fold's): their layout.
      r.out.at(i).format.layout = t.format.layout;
      r.out.at(i).nb = NaturalStrides(r.out.at(i).format);
    }
    r.float_result.at(i) = ResultEncoding(t.format.encoding);
  }
  return r;
}

// Each decomposition is built once per query (M2), not at every Emit and
// Verify of every graph a plan makes. The cache is the thread's own, so
// planning on several threads shares nothing, and it holds only what
// DecomposedOutputs returns for its key, so a lookup returns what building
// it would: results do not depend on what was asked before. It is cleared
// when full.
constexpr std::size_t kFusedCacheEntries = 512;

FusedResult CachedDecomposition(const FusedQuery& q) {
  thread_local std::unordered_map<FusedQuery, FusedResult, FusedQueryHash> cache;
  if (const auto found = cache.find(q); found != cache.end()) return found->second;
  // A nested fused kind's inference reenters the cache: no iterator is
  // held across it.
  FusedResult r = DecomposedOutputs(q);
  if (cache.size() >= kFusedCacheEntries) cache.clear();
  cache.emplace(q, r);
  return r;
}

Out InferFused(const Ctx& c) {
  FusedQuery q;
  q.key = c.key();
  std::ranges::copy(c.raw(), q.params.begin());
  q.count = static_cast<std::uint8_t>(c.count());
  for (std::size_t i = 0; i < c.count(); ++i) q.sources.at(i) = StandInOf(c, i);
  FusedResult r = CachedDecomposition(q);
  if (!r) return std::unexpected(std::move(r.error()));
  for (std::size_t i = 0; i < r->nout; ++i) {
    if (const auto alias = AliasedSource(c.key(), i); alias) {
      LLMP_TRY(Writable(c, *alias));
      const TensorDesc& s = *c.src(*alias);
      r->out.at(i) = ViewOf(c, *alias, s.format, s.nb, s.view_offset);
    }
  }
  return r;
}

// Valid-node metadata cannot silently select an invalid/batch handler.
#define LLMP_PRIMITIVE(op, name, params, required, max, infer, ...) \
  static_assert((infer != &InferInvalid) == (Op::op != Op::kFused));
#define LLMP_FUSED(kind, name, params, required, max, infer, validate, layout, partition, \
                   decomposes, handler, ref, ...)                                         \
  static_assert((infer != &InferInvalid && infer != &InferBatchKey) ==                    \
                (ReferencePath::ref != ReferencePath::kNone));
#include "ir/op_traits.inc"
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED

struct InferenceDispatch {
  Out (*infer)(const Ctx&);
  Checked (*validate)(const Ctx&);
};
#define LLMP_DISPATCH(op, name, params, required, max, infer, validate, ...) \
  InferenceDispatch{infer, validate},
#define LLMP_PRIMITIVE(...) LLMP_DISPATCH(__VA_ARGS__)
#define LLMP_FUSED(...)
constexpr auto kPrimitiveInference = std::to_array<InferenceDispatch>({
#include "ir/op_traits.inc"
});
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
#define LLMP_PRIMITIVE(...)
#define LLMP_FUSED(...) LLMP_DISPATCH(__VA_ARGS__)
constexpr auto kFusedInference = std::to_array<InferenceDispatch>({
#include "ir/op_traits.inc"
});
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
#undef LLMP_DISPATCH
static_assert(kPrimitiveInference.size() == kOpCount);
static_assert(kFusedInference.size() == kFusedKindCount);
static_assert([] {
  for (const auto& entry : kPrimitiveInference)
    if (!entry.infer || !entry.validate) return false;
  for (const auto& entry : kFusedInference)
    if (!entry.infer || !entry.validate) return false;
  return true;
}());

}  // namespace

std::expected<Inferred, VerifyError> Infer(OpKey key,
                                           std::span<const std::byte, kMaxParamBytes> params,
                                           std::span<const TensorId> sources,
                                           std::span<const TensorDesc> tensors,
                                           std::span<const TensorDesc> requested) {
  const Ctx c(key, params, sources, tensors, requested);
  if (sources.size() > kMaxSources) {
    return Err(GraphError::kArity, std::format("{} sources, past {}", sources.size(), kMaxSources));
  }
  if (key.op != Op::kFused && key.fused != FusedKind::kNone) {
    return Err(GraphError::kParams, "a fused kind on a primitive op");
  }
  const OpTraits* traits = Traits(key);
  if (!traits) return Fail(GraphError::kParams, c, "an op outside op set v1");
  // Invalid-as-node fused keys keep their parameter error below. Valid keys
  // share static bounds; decompositions refine parameter-dependent arities.
  if (traits->reference != ReferencePath::kNone) {
    LLMP_TRY(Arity(c, traits->required_sources, traits->max_sources));
  }
  const auto& dispatch = key.op == Op::kFused
                             ? kFusedInference[static_cast<std::size_t>(key.fused)]
                             : kPrimitiveInference[static_cast<std::size_t>(key.op)];
  // Leaves validate their input/constant metadata, rather than op parameters.
  if (dispatch.infer != &Leaf) LLMP_TRY(Tail(c, traits->param_bytes));
  LLMP_TRY(dispatch.validate(c));
  Out r = dispatch.infer(c);
  if (!r || dispatch.infer == &Leaf) {
    return r;
  }
  // Every computed output (not a view) is a valid format: no zero extent,
  // no size past 64 bits, whatever shape a parameter or a requested
  // description gave it (a fill's, a repeat's, a copy's, a join's).
  for (std::size_t i = 0; i < r->nout; ++i) {
    const TensorDesc& t = r->out.at(i);
    if (!t.view_of.valid() && !format::Validate(t.format)) {
      return Fail(GraphError::kShape, c,
                  std::format("output {} of {} is not a valid {} tensor", i, Text(t.shape()),
                              format::Info(t.format.encoding).name));
    }
  }
  return r;
}

}  // namespace llmp::ir
