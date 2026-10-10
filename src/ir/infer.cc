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
#include <span>
#include <string>
#include <utility>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
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
  if (t != nullptr && !format::Validate(t->format)) {
    return Fail(GraphError::kFormat, c, std::format("source {} has an invalid format", i));
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
    if (a.ne.at(d) % b.ne.at(d) != 0) {
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
  if (view.nb[0] != unit || view.format.shape.ne[0] % per != 0) {
    return Fail(GraphError::kShape, c,
                std::format("a view's first stride is one unit ({} bytes) over whole units", unit));
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

// A store's target: state or a result of this step, never a weight or a
// plan constant (bound once, shared, read-only).
std::expected<void, VerifyError> Writable(const Ctx& c, std::size_t i) {
  const Role role = c.root_desc(i).role;
  if (role == Role::kWeight || role == Role::kConstant) {
    return Fail(GraphError::kBinding, c,
                std::format("source {} is a {}'s bytes, which nothing writes", i, RoleName(role)));
  }
  return {};
}

// ---------------------------------------------------------------- per op

std::size_t ParamSize(OpKey key) {
  if (key.op == Op::kFused) {
    switch (key.fused) {
      case FusedKind::kRmsNormMul:
      case FusedKind::kRmsNormMulAdd:
      case FusedKind::kHcNorm:
      case FusedKind::kNormGate:
      case FusedKind::kLayerNormModulate:
      case FusedKind::kGatedResidualNorm:
        return sizeof(NormParams);
      case FusedKind::kRmsNormMulRope:
      case FusedKind::kQkPrep:
        return sizeof(NormRopeParams);
      case FusedKind::kRopeStore:
        return sizeof(RopeParams);
      case FusedKind::kMatMulGlu:
      case FusedKind::kMoeGlu:
        return sizeof(GluParams);
      case FusedKind::kOwnerAttention:
        return sizeof(OwnerAttnParams);
      case FusedKind::kHcNormPost:
        return sizeof(HcNormPostParams);
      case FusedKind::kDupUpAdd:
        return sizeof(DupUpParams);
      default:
        return 0;
    }
  }
  switch (key.op) {
    case Op::kMatMul:
    case Op::kMatMulId:
      return sizeof(ProductParams);
    case Op::kPermute:
      return sizeof(PermuteParams);
    case Op::kConcat:
      return sizeof(ConcatParams);
    case Op::kExpertSort:
      return sizeof(ExpertSortParams);
    case Op::kAttention:
      return sizeof(AttnParams);
    case Op::kMaskBuild:
      return sizeof(MaskParams);
    case Op::kPool:
      return sizeof(PoolParams);
    case Op::kRmsNorm:
    case Op::kL2Norm:
    case Op::kLayerNorm:
    case Op::kChannelNorm:
      return sizeof(NormParams);
    case Op::kRope:
      return sizeof(RopeParams);
    case Op::kScale:
      return sizeof(ScaleParams);
    case Op::kUnary:
      return sizeof(UnaryParams);
    case Op::kClamp:
      return sizeof(ClampParams);
    case Op::kFill:
      return sizeof(FillParams);
    case Op::kGlu:
      return sizeof(GluParams);
    case Op::kSoftcap:
      return sizeof(SoftcapParams);
    case Op::kSoftMax:
      return sizeof(SoftMaxParams);
    case Op::kArgmax:
      return sizeof(ArgmaxParams);
    case Op::kArgsort:
      return sizeof(ArgsortParams);
    case Op::kTopK:
      return sizeof(TopKParams);
    case Op::kRoute:
      return sizeof(RouteParams);
    case Op::kHcPre:
    case Op::kHcPost:
    case Op::kHcMix:
      return sizeof(HcParams);
    case Op::kConv2d:
      return sizeof(Conv2dParams);
    case Op::kUpsample:
      return sizeof(UpsampleParams);
    case Op::kSample:
      return sizeof(SampleParams);
    case Op::kAllReduce:
    case Op::kAllGather:
    case Op::kAllToAll:
    case Op::kSend:
    case Op::kRecv:
      return sizeof(CollectiveParams);
    default:
      return 0;
  }
}

bool Finite(float v) { return std::isfinite(v); }

// A product's accumulation and hint (op.h): known values, reserved bytes
// zero, and a Hadamard matrix square.
std::expected<void, VerifyError> CheckProduct(const Ctx& c, const TensorDesc& a) {
  const auto p = c.params<ProductParams>();
  LLMP_TRY(Zero(c, 2, 2));
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
  LLMP_TRY(Zero(c, at + 3, 1));
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
    return Fail(GraphError::kParams, c, "the first rotated value is even (GGML's n_offs)");
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
  LLMP_TRY(Arity(c, 0, 0));
  LLMP_TRY(Tail(c, 0));
  const TensorDesc* r = c.req(0);
  if (r == nullptr) {
    return Fail(GraphError::kArity, c, "a leaf needs its description");
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

Out InferPrimitive(const Ctx& c) {
  const Op op = c.key().op;
  LLMP_TRY(Tail(c, ParamSize(c.key())));
  switch (op) {
    case Op::kInput:
    case Op::kWeight:
    case Op::kState:
      return Leaf(c);

    case Op::kView: {
      LLMP_TRY(Arity(c, 1, 1));
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
    case Op::kReshape: {
      LLMP_TRY(Arity(c, 1, 1));
      const TensorDesc& a = *c.src(0);
      const TensorDesc* r = c.req(0);
      if (r == nullptr) {
        return Fail(GraphError::kArity, c, "a reshape needs its shape");
      }
      LLMP_TRY(Want(IsContiguous(a), c, "reshapes a contiguous tensor"));
      LLMP_TRY(
          Want(Elements(r->shape()) == Elements(a.shape()), c,
               std::format("{} has another count than {}", Text(r->shape()), Text(a.shape()))));
      format::TensorFormat f = a.format;
      f.shape = r->format.shape;
      f.stacked = 1;
      if (!format::Validate(f)) {
        return Fail(GraphError::kShape, c, "the shape is not valid for its format");
      }
      return One(ViewOf(c, 0, f, NaturalStrides(f), a.view_offset), false);
    }
    case Op::kPermute: {
      LLMP_TRY(Arity(c, 1, 1));
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
    case Op::kCont: {
      LLMP_TRY(Arity(c, 1, 1));
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
    case Op::kConcat: {
      LLMP_TRY(Arity(c, 2, 2));
      const auto p = c.params<ConcatParams>();
      const TensorDesc& a = *c.src(0);
      const TensorDesc& b = *c.src(1);
      if (p.dim >= 4) {
        return Fail(GraphError::kParams, c, "dim past 3");
      }
      LLMP_TRY(Want(a.format.layout == format::Layout::kDense &&
                        a.format.layout == b.format.layout &&
                        a.format.encoding == b.format.encoding,
                    c, "joins dense tensors of one encoding"));
      std::array<std::uint64_t, 4> ne = a.shape().ne;
      for (std::size_t d = 0; d < 4; ++d) {
        if (d == p.dim) {
          ne.at(d) += b.ne(d);
        } else {
          LLMP_TRY(Want(a.ne(d) == b.ne(d), c,
                        std::format("{} and {} differ beyond dim {}", Text(a.shape()),
                                    Text(b.shape()), p.dim)));
        }
      }
      return One(Fresh(ShapeOf({ne[0], ne[1], ne[2], ne[3]}), a.format.encoding), false);
    }
    case Op::kRepeat: {
      LLMP_TRY(Arity(c, 1, 1));
      const TensorDesc& a = *c.src(0);
      const TensorDesc* r = c.req(0);
      if (r == nullptr) {
        return Fail(GraphError::kArity, c, "a repeat needs its shape");
      }
      LLMP_TRY(Want(a.format.layout == format::Layout::kDense, c, "tiles dense tensors"));
      LLMP_TRY(Want(Broadcasts(r->shape(), a.shape()), c,
                    std::format("{} is not a multiple of {}", Text(r->shape()), Text(a.shape()))));
      return One(Fresh(ShapeOf({r->ne(0), r->ne(1), r->ne(2), r->ne(3)}), a.format.encoding),
                 false);
    }
    case Op::kConvert: {
      LLMP_TRY(Arity(c, 1, 1));
      const TensorDesc& a = *c.src(0);
      const TensorDesc* r = c.req(0);
      if (r == nullptr) {
        return Fail(GraphError::kArity, c, "a cast needs its encoding");
      }
      LLMP_TRY(Want(a.format.layout == format::Layout::kDense &&
                        DenseEncoding(r->format.encoding) &&
                        r->format.layout == format::Layout::kDense,
                    c, "casts a dense tensor to a dense encoding"));
      return One(Fresh(a.shape(), r->format.encoding), false);
    }
    case Op::kGetRows: {
      LLMP_TRY(Arity(c, 2, 2));
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

    case Op::kMatMul: {
      LLMP_TRY(Arity(c, 2, 2));
      LLMP_TRY(FloatSrc(c, 0));
      LLMP_TRY(FloatSrc(c, 1));
      const TensorDesc& a = *c.src(0);
      const TensorDesc& b = *c.src(1);
      LLMP_TRY(CheckProduct(c, a));
      LLMP_TRY(Want(a.ne(0) == b.ne(0) && b.ne(2) % a.ne(2) == 0 && b.ne(3) % a.ne(3) == 0, c,
                    std::format("{} times {}", Text(a.shape()), Text(b.shape()))));
      return FloatResult(S(a.ne(1), b.ne(1), b.ne(2), b.ne(3)), Encoding::kF32);
    }
    case Op::kMatMulId: {
      LLMP_TRY(Arity(c, 3, 3));
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
    case Op::kPrepare: {
      LLMP_TRY(Arity(c, 1, 1));
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
      if (!format::Validate(t.format) || UnitBytes(t.format) == 0 || Integer(t.format.encoding)) {
        return Fail(GraphError::kFormat, c,
                    "prepares into a single-part value format of its shape");
      }
      t.nb = NaturalStrides(t.format);
      return One(t, false);
    }
    case Op::kExpertSort: {
      LLMP_TRY(Arity(c, 1, 1));
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

    case Op::kAttention: {
      LLMP_TRY(Arity(c, 3, 5));
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
      LLMP_TRY(Want(q.ne(0) == k.ne(0) && k.ne(1) == v.ne(1) && k.ne(2) == v.ne(2) &&
                        q.ne(3) == k.ne(3) && q.ne(3) == v.ne(3) && q.ne(2) % k.ne(2) == 0,
                    c,
                    std::format("q {} over k {} and v {}", Text(q.shape()), Text(k.shape()),
                                Text(v.shape()))));
      if (mask != nullptr) {
        LLMP_TRY(Want(mask->ne(0) == k.ne(1) && mask->ne(1) >= q.ne(1) &&
                          q.ne(2) % mask->ne(2) == 0 && q.ne(3) % mask->ne(3) == 0,
                      c,
                      std::format("a mask {} for q {} over {} cells", Text(mask->shape()),
                                  Text(q.shape()), k.ne(1))));
      }
      if (sinks != nullptr) {
        LLMP_TRY(Want(sinks->shape().ne == S(q.ne(2)).ne, c, "a sink per head"));
      }
      return FloatResult(S(v.ne(0), q.ne(2), q.ne(1), q.ne(3)), Encoding::kF32);
    }
    case Op::kMaskBuild: {
      LLMP_TRY(Arity(c, 2, 2));
      LLMP_TRY(IndexSrc(c, 0));
      LLMP_TRY(IndexSrc(c, 1));
      const auto p = c.params<MaskParams>();
      LLMP_TRY(Zero(c, 1, 3));
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
          r->ne(0) == cp.ne(0) && r->ne(1) >= qp.ne(0) && r->ne(2) == 1 && r->ne(3) == 1, c,
          std::format("a mask {} for {} cells and {} rows", Text(r->shape()), cp.ne(0), qp.ne(0))));
      return One(Fresh(S(cp.ne(0), r->ne(1)), e), false);
    }
    case Op::kCacheStore: {
      LLMP_TRY(Arity(c, 3, 3));
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
      LLMP_TRY(Want(cache.ne(0) == src.ne(0) && cache.ne(2) == src.ne(2) &&
                        cache.ne(3) == src.ne(3) && src.ne(1) == idx.ne(0) &&
                        src.ne(2) % idx.ne(1) == 0 && src.ne(3) % idx.ne(2) == 0 && idx.ne(3) == 1,
                    c,
                    std::format("rows {} into {} at {}", Text(src.shape()), Text(cache.shape()),
                                Text(idx.shape()))));
      return One(ViewOf(c, 0, cache.format, cache.nb, cache.view_offset), false);
    }
    case Op::kIndexScore: {
      LLMP_TRY(Arity(c, 3, 4));
      for (std::size_t i = 0; i < 4; ++i) {
        LLMP_TRY(FloatSrc(c, i));
      }
      const TensorDesc& q = *c.src(0);
      const TensorDesc& k = *c.src(1);
      const TensorDesc& w = *c.src(2);
      const TensorDesc* mask = c.src(3);
      LLMP_TRY(Want(
          q.ne(0) == k.ne(0) && k.ne(1) == 1 && k.ne(3) == q.ne(3) && w.ne(0) == q.ne(1) &&
              w.ne(1) == q.ne(2) && w.ne(2) == 1 && w.ne(3) == q.ne(3),
          c, std::format("q {}, k {}, w {}", Text(q.shape()), Text(k.shape()), Text(w.shape()))));
      if (mask != nullptr) {
        LLMP_TRY(Want(mask->ne(0) == k.ne(2) && mask->ne(1) == q.ne(2) && mask->ne(2) == 1 &&
                          q.ne(3) % mask->ne(3) == 0,
                      c, std::format("a mask {}", Text(mask->shape()))));
      }
      return FloatResult(S(k.ne(2), q.ne(2), 1, q.ne(3)), Encoding::kF32);
    }
    case Op::kPool: {
      LLMP_TRY(Arity(c, 1, 2));
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

    case Op::kRmsNorm:
    case Op::kL2Norm:
    case Op::kLayerNorm:
    case Op::kChannelNorm: {
      LLMP_TRY(Arity(c, 1, 1));
      LLMP_TRY(FloatSrc(c, 0));
      LLMP_TRY(NormEps(c, c.params<NormParams>().eps));
      return FloatResult(c.src(0)->shape(), FloatOf(*c.src(0)));
    }
    case Op::kRope: {
      LLMP_TRY(Arity(c, 2, 3));
      const auto p = c.params<RopeParams>();
      LLMP_TRY(CheckRope(c, p, 0));
      return RopeOf(c, p, 0, 1, 2);
    }

    case Op::kAdd:
    case Op::kSub:
    case Op::kMul:
    case Op::kDiv: {
      LLMP_TRY(Arity(c, 2, 2));
      LLMP_TRY(FloatSrc(c, 0));
      LLMP_TRY(FloatSrc(c, 1));
      const TensorDesc& a = *c.src(0);
      const TensorDesc& b = *c.src(1);
      LLMP_TRY(
          Want(Broadcasts(a.shape(), b.shape()), c,
               std::format("{} does not broadcast over {}", Text(b.shape()), Text(a.shape()))));
      return FloatResult(a.shape(), FloatOf(a));
    }
    case Op::kScale:
    case Op::kUnary:
    case Op::kClamp:
    case Op::kSoftcap: {
      LLMP_TRY(Arity(c, 1, 1));
      LLMP_TRY(FloatSrc(c, 0));
      if (op == Op::kUnary && static_cast<std::uint8_t>(c.params<UnaryParams>().fn) >
                                  static_cast<std::uint8_t>(UnaryFn::kSgn)) {
        return Fail(GraphError::kParams, c, "an unknown function");
      }
      if (op == Op::kSoftcap) {
        const float cap = c.params<SoftcapParams>().cap;
        if (!Finite(cap) || cap <= 0.0f) {
          return Fail(GraphError::kParams, c, "a cap above zero");
        }
      }
      if (op == Op::kClamp) {
        const auto p = c.params<ClampParams>();
        if (!(p.min <= p.max)) {
          return Fail(GraphError::kParams, c, "min above max");
        }
      }
      return FloatResult(c.src(0)->shape(), FloatOf(*c.src(0)));
    }
    case Op::kFill: {
      LLMP_TRY(Arity(c, 0, 0));
      const TensorDesc* r = c.req(0);
      if (r == nullptr) {
        return Fail(GraphError::kArity, c, "a fill needs its shape");
      }
      LLMP_TRY(Want(ResultEncoding(r->format.encoding), c, "fills F32, F16 or BF16"));
      return One(Fresh(ShapeOf({r->ne(0), r->ne(1), r->ne(2), r->ne(3)}), r->format.encoding),
                 false);
    }
    case Op::kGlu: {
      LLMP_TRY(Arity(c, 1, 2));
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

    case Op::kSumRows: {
      LLMP_TRY(Arity(c, 1, 1));
      LLMP_TRY(FloatSrc(c, 0));
      const TensorDesc& x = *c.src(0);
      return FloatResult(S(1, x.ne(1), x.ne(2), x.ne(3)), FloatOf(x));
    }
    case Op::kSoftMax: {
      LLMP_TRY(Arity(c, 1, 3));
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
        LLMP_TRY(Want(mask->ne(0) == x.ne(0) && mask->ne(1) >= x.ne(1) &&
                          x.ne(2) % mask->ne(2) == 0 && x.ne(3) % mask->ne(3) == 0,
                      c, std::format("a mask {} over {}", Text(mask->shape()), Text(x.shape()))));
      }
      if (const TensorDesc* sinks = c.src(2); sinks != nullptr) {
        LLMP_TRY(Want(sinks->shape().ne == S(x.ne(2)).ne, c, "a sink per matrix"));
      }
      return FloatResult(x.shape(), FloatOf(x));
    }
    case Op::kArgmax: {
      LLMP_TRY(Arity(c, 1, 1));
      LLMP_TRY(FloatSrc(c, 0));
      if (c.byte(0) > static_cast<std::uint8_t>(ArgmaxNan::kFirstColumn)) {
        return Fail(GraphError::kParams, c, "an unknown NaN rule");
      }
      const TensorDesc& x = *c.src(0);
      LLMP_TRY(Want(x.ne(2) == 1 && x.ne(3) == 1, c, "a matrix"));
      return One(Fresh(S(x.ne(1)), Encoding::kI32), false);
    }
    case Op::kArgsort: {
      LLMP_TRY(Arity(c, 1, 1));
      LLMP_TRY(FloatSrc(c, 0));
      LLMP_TRY(Bool(c, 0));
      return One(Fresh(c.src(0)->shape(), Encoding::kI32), false);
    }
    case Op::kTopK: {
      LLMP_TRY(Arity(c, 1, 1));
      LLMP_TRY(FloatSrc(c, 0));
      const auto p = c.params<TopKParams>();
      const TensorDesc& x = *c.src(0);
      if (p.k == 0 || p.k > x.ne(0)) {
        return Fail(GraphError::kParams, c, std::format("k {} of {}", p.k, x.ne(0)));
      }
      return One(Fresh(S(p.k, x.ne(1), x.ne(2), x.ne(3)), Encoding::kI32), false);
    }

    case Op::kRoute: {
      LLMP_TRY(Arity(c, 1, 4));
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
        // The ids looked up already (a token-hash side input's).
        LLMP_TRY(Want(table->shape().ne == S(p.used, logits.ne(1)).ne &&
                          table->format.encoding == Encoding::kI32,
                      c, "ids [used, rows] I32 without tokens"));
      }
      Inferred r;
      r.out[0] = Fresh(S(p.used, logits.ne(1)), Encoding::kI32);
      r.out[1] = Fresh(S(p.used, logits.ne(1)), Encoding::kF32);
      r.nout = 2;
      r.float_result[1] = true;
      return r;
    }
    case Op::kExpertReduce: {
      LLMP_TRY(Arity(c, 2, 4));
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
      // A scale gathered per slot [1, U, T], or one per expert [E] (a
      // single expert's [1] is either: the one slot's expert is 0).
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

    case Op::kSsmConv: {
      LLMP_TRY(Arity(c, 2, 2));
      LLMP_TRY(FloatSrc(c, 0));
      LLMP_TRY(FloatSrc(c, 1));
      const TensorDesc& sx = *c.src(0);
      const TensorDesc& w = *c.src(1);
      LLMP_TRY(Want(sx.ne(3) == 1 && w.ne(2) == 1 && w.ne(3) == 1 && sx.ne(1) == w.ne(1) &&
                        sx.ne(0) >= w.ne(0),
                    c, std::format("{} over taps {}", Text(sx.shape()), Text(w.shape()))));
      return FloatResult(S(w.ne(1), sx.ne(0) - w.ne(0) + 1, sx.ne(2)), Encoding::kF32);
    }
    case Op::kGatedDeltaNet: {
      LLMP_TRY(Arity(c, 6, 6));
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
      LLMP_TRY(Want(
          k.shape().ne == q.shape().ne && q.ne(2) == t && h % q.ne(1) == 0 && b % q.ne(3) == 0 &&
              (g.ne(0) == 1 || g.ne(0) == sk) && g.ne(1) == h && g.ne(2) == t && g.ne(3) == b &&
              beta.shape().ne == S(1, h, t, b).ne && state.shape().ne == S(sk, sv, h, b).ne,
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
    case Op::kConv2d: {
      LLMP_TRY(Arity(c, 2, 3));
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
    case Op::kUpsample: {
      LLMP_TRY(Arity(c, 1, 1));
      LLMP_TRY(FloatSrc(c, 0));
      const auto f = c.params<UpsampleParams>().factor;
      if (f == 0) {
        return Fail(GraphError::kParams, c, "a factor of at least one");
      }
      const TensorDesc& x = *c.src(0);
      return FloatResult(S(x.ne(0) * f, x.ne(1) * f, x.ne(2), x.ne(3)), FloatOf(x));
    }

    case Op::kHcPre:
    case Op::kHcPost:
    case Op::kHcMix: {
      const auto p = c.params<HcParams>();
      LLMP_TRY(Zero(c, 2, 2));
      if (static_cast<std::uint8_t>(p.mix) > 1 || static_cast<std::uint8_t>(p.fold) > 1 ||
          p.streams == 0 || !Finite(p.eps) || !Finite(p.scale)) {
        return Fail(GraphError::kParams, c, "known kinds, streams, finite eps and scale");
      }
      const std::uint64_t hc = p.streams;
      if (op == Op::kHcPre) {
        LLMP_TRY(Arity(c, 2, 2));
        LLMP_TRY(FloatSrc(c, 0));
        LLMP_TRY(FloatSrc(c, 1));
        const TensorDesc& x = *c.src(0);
        const TensorDesc& w = *c.src(1);
        const std::uint64_t t = x.ne(2);
        LLMP_TRY(Want(x.ne(1) == hc && x.ne(3) == 1, c, "x is [width, streams, tokens]"));
        if (p.fold == HcFold::kWeights) {
          if (p.scale != 1.0f) {
            return Fail(GraphError::kParams, c, "the weighted fold has no scale");
          }
          LLMP_TRY(Want(w.shape().ne == S(hc, t).ne, c, "a weight per stream and token"));
        } else {
          LLMP_TRY(Want(w.shape().ne == x.shape().ne, c, "a gate per value"));
        }
        return FloatResult(S(x.ne(0), t), Encoding::kF32);
      }
      if (op == Op::kHcPost) {
        LLMP_TRY(Arity(c, 3, 4));
        for (std::size_t i = 0; i < 4; ++i) {
          LLMP_TRY(FloatSrc(c, i));
        }
        const TensorDesc& y = *c.src(0);
        const TensorDesc& x = *c.src(1);
        const TensorDesc& post = *c.src(2);
        const std::uint64_t w = y.ne(0);
        const std::uint64_t t = y.ne(1);
        LLMP_TRY(Want(y.ne(2) == 1 && y.ne(3) == 1 && x.shape().ne == S(w, hc, t).ne &&
                          post.shape().ne == S(hc, t).ne,
                      c,
                      std::format("y {}, x {}, post {}", Text(y.shape()), Text(x.shape()),
                                  Text(post.shape()))));
        if (const TensorDesc* comb = c.src(3); comb != nullptr) {
          LLMP_TRY(Want(comb->shape().ne == S(hc, hc, t).ne, c, "comb is [hc, hc, tokens]"));
        }
        return FloatResult(S(w, hc, t), Encoding::kF32);
      }
      LLMP_TRY(Arity(c, 3, 3));
      for (std::size_t i = 0; i < 3; ++i) {
        LLMP_TRY(FloatSrc(c, i));
      }
      const TensorDesc& mixes = *c.src(0);
      const TensorDesc& scale = *c.src(1);
      const TensorDesc& base = *c.src(2);
      const std::uint64_t m = (2 + hc) * hc;
      const std::uint64_t t = mixes.ne(1);
      LLMP_TRY(Want(mixes.shape().ne == S(m, t).ne && Elements(scale.shape()) == scale.ne(0) &&
                        scale.ne(0) >= 3 && base.shape().ne == S(m).ne,
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

    case Op::kSample: {
      LLMP_TRY(Arity(c, 1, 2));
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
        if (keys->format.encoding != Encoding::kI64 ||
            keys->format.layout != format::Layout::kDense) {
          return Fail(GraphError::kFormat, c, "keys are I64");
        }
        LLMP_TRY(Want(keys->shape().ne == S(3, logits.ne(1)).ne, c, "keys are [3, rows]"));
      }
      return One(Fresh(S(logits.ne(1)), Encoding::kI32), false);
    }
    case Op::kAllReduce:
    case Op::kAllGather:
    case Op::kAllToAll:
    case Op::kSend:
    case Op::kRecv: {
      const auto p = c.params<CollectiveParams>();
      LLMP_TRY(Zero(c, 9, 3));
      if (p.dim >= 4) {
        return Fail(GraphError::kParams, c, "dim past 3");
      }
      if (op == Op::kRecv) {
        LLMP_TRY(Arity(c, 0, 0));
        const TensorDesc* r = c.req(0);
        if (r == nullptr || r->format.layout != format::Layout::kDense ||
            !format::Validate(r->format)) {
          return Fail(GraphError::kArity, c, "a receive needs its dense format");
        }
        return One(Fresh(r->shape(), r->format.encoding), false);
      }
      LLMP_TRY(Arity(c, 1, 1));
      LLMP_TRY(Readable(c, 0));
      const TensorDesc& x = *c.src(0);
      LLMP_TRY(Want(x.format.layout == format::Layout::kDense, c, "moves dense tensors"));
      if (op == Op::kSend) {
        return Inferred{};
      }
      if (op == Op::kAllGather) {
        const TensorDesc* r = c.req(0);
        if (r == nullptr) {
          return Fail(GraphError::kArity, c, "a gather needs its shape");
        }
        bool ok = r->ne(p.dim) % x.ne(p.dim) == 0;
        for (std::size_t d = 0; d < 4; ++d) {
          ok = ok && (d == p.dim || r->ne(d) == x.ne(d));
        }
        LLMP_TRY(Want(ok, c, "gathers along dim a multiple of x"));
        return One(Fresh(ShapeOf({r->ne(0), r->ne(1), r->ne(2), r->ne(3)}), x.format.encoding),
                   false);
      }
      return One(Fresh(x.shape(), x.format.encoding), false);
    }
    case Op::kFused:
      break;
  }
  return Fail(GraphError::kParams, c, "an op outside op set v1");
}

Out InferFused(const Ctx& c) {
  LLMP_TRY(Tail(c, ParamSize(c.key())));
  if (c.key().fused == FusedKind::kNone || static_cast<std::uint16_t>(c.key().fused) >
                                               static_cast<std::uint16_t>(FusedKind::kDupUpAdd)) {
    return Fail(GraphError::kParams, c, "an unknown fused kind");
  }
  if (c.key().fused == FusedKind::kGroupedStores) {
    return Fail(GraphError::kParams, c, "grouped_stores is a batch key, not a node");
  }
  if (c.key().fused == FusedKind::kRopeStore) {
    LLMP_TRY(CheckRope(c, c.params<RopeParams>(), 0));
  }
  if (c.key().fused == FusedKind::kRmsNormMulRope || c.key().fused == FusedKind::kQkPrep) {
    LLMP_TRY(CheckRope(c, c.params<NormRopeParams>().rope, 4));
  }
  // The decomposition over stand-ins of the sources: its outputs are the
  // node's.
  Builder b;
  std::array<TensorId, kMaxSources> ins{};
  for (std::size_t i = 0; i < c.count(); ++i) {
    if (const TensorDesc* s = c.src(i); s != nullptr) {
      TensorDesc t = *s;
      ins.at(i) = b.Input(t.format);
    }
  }
  ParamBytes params{};
  std::ranges::copy(c.raw(), params.begin());
  const Outputs outs = Decompose(b, c.key(), params, std::span(ins).first(c.count()), {});
  if (!b.ok()) {
    VerifyError e = *b.error();
    e.detail = std::format("{} (in its decomposition)", e.detail);
    return std::unexpected(e);
  }
  Inferred r;
  r.nout = outs.n;
  for (std::size_t i = 0; i < outs.n; ++i) {
    if (const auto alias = AliasedSource(c.key(), i); alias) {
      LLMP_TRY(Writable(c, *alias));
      const TensorDesc& s = *c.src(*alias);
      r.out.at(i) = ViewOf(c, *alias, s.format, s.nb, s.view_offset);
      continue;
    }
    const TensorDesc t = b.tensor(outs[i]);
    r.out.at(i) = Fresh(t.shape(), t.format.encoding);
    r.float_result.at(i) = ResultEncoding(t.format.encoding);
  }
  return r;
}

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
  if (key.op == Op::kFused) {
    return InferFused(c);
  }
  if (key.fused != FusedKind::kNone) {
    return Err(GraphError::kParams, "a fused kind on a primitive op");
  }
  return InferPrimitive(c);
}

}  // namespace llmp::ir
