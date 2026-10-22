// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/op.h"
#include "ir/op_traits.h"

namespace llmp::ir {
namespace {

using ir::NodeId;
using ir::Op;
using ir::TensorId;
using Ne = std::array<std::uint64_t, 4>;

ir::OpKey Prim(Op op) { return ir::Primitive(op); }

Ne NeOf(const ir::TensorDesc& t) { return {t.ne(0), t.ne(1), t.ne(2), t.ne(3)}; }
std::uint64_t Count(const Ne& ne) { return ne[0] * ne[1] * ne[2] * ne[3]; }
Ne NaturalOf(const Ne& ne) {
  Ne st{};
  std::uint64_t s = 1;
  for (std::size_t d = 0; d < 4; ++d) {
    st[d] = s;
    s *= ne[d];
  }
  return st;
}
// Whether strides `st` are `ne`'s natural ones on every axis wider than one
// (an extent-one axis's stride steps nothing).
bool NaturalOn(const Ne& ne, const Ne& st) {
  const Ne natural = NaturalOf(ne);
  for (std::size_t d = 0; d < 4; ++d)
    if (ne[d] != 1 && st[d] != natural[d]) return false;
  return true;
}
format::Shape ShapeOfNe(const Ne& ne) { return ir::ShapeOf({ne[0], ne[1], ne[2], ne[3]}); }

// A value as a strided view of a contiguous storage tensor of the new
// graph: element (i0..i3) at storage index offset + Σ i_d · st[d].
struct Strided {
  TensorId storage{};
  Ne ne{};
  Ne st{};
  std::uint64_t offset = 0;
};

// The axes of a compact value (a permutation of its storage's natural
// layout) in storage order, innermost first; extents of one last.
struct Order {
  std::array<std::uint8_t, 4> axes{0, 1, 2, 3};  // storage axis k holds logical axis axes[k]
  bool logical = true;  // the wide axes in logical order: the value's own layout
  bool natural() const { return logical; }
};

using Axes = std::array<std::uint8_t, 4>;

// Extent `ne`'s natural storage order: its wide axes ascending, then the
// extent-one axes.
Axes NaturalAxes(const Ne& ne) {
  Axes a{};
  std::size_t n = 0;
  for (std::uint8_t d = 0; d < 4; ++d)
    if (ne[d] != 1) a[n++] = d;
  for (std::uint8_t d = 0; d < 4; ++d)
    if (ne[d] == 1) a[n++] = d;
  return a;
}
// The wide axes of `ne` in `axes`' order, then its extent-one axes: an
// order asked of a tensor of other extents over the same logical axes.
Axes Restricted(const Axes& axes, const Ne& ne) {
  Axes a{};
  std::size_t n = 0;
  for (const std::uint8_t d : axes)
    if (ne[d] != 1) a[n++] = d;
  for (std::uint8_t d = 0; d < 4; ++d)
    if (ne[d] == 1) a[n++] = d;
  return a;
}

class Sinker {
 public:
  // `through`: elementwise nodes pass their readers' wants to their
  // operands too (SinkLayouts tries both and keeps the cheaper graph).
  Sinker(const ir::Graph& g, bool through)
      : g_(g), through_(through), vals_(g.tensor_count()), wants_(g.tensor_count()) {}

  std::expected<Sunk, ir::VerifyError> Run();

 private:
  // Backward: the storage orders each tensor's readers ask for, by votes
  // (a product's or a norm's input wants its logical layout; a view, an
  // elementwise node or a concatenation passes its own on).
  void Prefer();
  void Want(TensorId t, const Axes& axes, std::uint32_t votes = 1) {
    if (t.valid()) wants_.at(t.v)[Restricted(axes, NeOf(g_.tensor(t)))] += votes;
  }
  std::optional<Axes> Wanted(TensorId t) const {
    const auto& w = wants_.at(t.v);
    std::optional<Axes> best;
    std::uint32_t most = 0;
    for (const auto& [axes, votes] : w)
      if (votes > most) {
        most = votes;
        best = axes;
      }
    return best;
  }

  // A concatenation not made: its operands' values, each from its start
  // along the joined axis.
  struct Joined {
    Ne ne{};
    std::uint8_t axis = 0;
    std::vector<std::pair<std::uint64_t, Strided>> pieces;
  };
  // A value made from small ones, in whatever order its reader wants: a
  // repeat of a small tensor, a concatenation of such, an elementwise
  // function of one, or one with its axes renamed (a permute, a reshape
  // adding or dropping extents of one). A modulation's conditioning rows
  // are one: made where read, they never fix a layout.
  struct Cheap {
    enum class Kind : std::uint8_t { kRepeat, kConcat, kMap, kAxes } kind = Kind::kRepeat;
    Ne ne{};
    TensorId small{};  // kRepeat: packed, of extents small_ne (1 where repeated)
    Ne small_ne{};
    std::uint32_t a = 0, b = 0;  // children
    std::uint8_t axis = 0;       // kConcat
    NodeId node{};               // kMap: the original elementwise node
    Axes from{};                 // kAxes: axis d reads the child's axis from[d] (wide axes)
  };
  struct Val {
    TensorId mapped{};                     // the original's layout, made
    std::optional<Strided> lazy{};         // or as a view of a storage
    std::optional<Joined> joined{};        // or as the pieces of a concatenation
    std::optional<std::uint32_t> cheap{};  // or made where read (cheap_)
    TensorId packed{};                     // a packed copy of the logical value, made
  };
  std::optional<std::uint32_t> CheapOf(const ir::Node& n);
  TensorId EmitCheap(std::uint32_t i, const Axes& order);
  // A view's steps over `ne`'s packed layout, as its axes' indices: where
  // it starts (at), the last index it reaches (used) and, per view axis,
  // the axis it steps (none: an extent-one axis) and by how many.
  struct ViewMap {
    Ne at{};
    Ne used{};
    std::array<std::pair<std::int8_t, std::uint64_t>, 4> steps{};
  };
  static std::optional<ViewMap> MapView(const Ne& ne, const ir::TensorDesc& src,
                                        const ir::TensorDesc& out);
  static Strided Apply(const Strided& v, const ViewMap& m, const Ne& at, const ir::TensorDesc& out);
  std::optional<Joined> JoinedOf(TensorId t);
  bool Join(const ir::Node& n);
  std::optional<Strided> Unjoin(TensorId t);

  std::uint64_t StorageCount(TensorId storage) const {
    return ir::Elements(b_.tensor(storage).shape());
  }
  bool Dense(const ir::TensorDesc& t) const {
    return ir::UnitElements(t.format) == 1 && ir::UnitBytes(t.format) != 0 &&
           t.format.layout == format::Layout::kDense;
  }
  std::optional<Strided> Lift(TensorId t);
  std::optional<Order> CompactOrder(const Strided& v) const;
  TensorId StorageAs(const Strided& v, const Order& o);
  TensorId InOrder(TensorId original, const Order& o);
  std::optional<TensorId> Materialize(TensorId t, bool packed);
  std::optional<TensorId> MaterializeStrided(const Strided& v, const ir::TensorDesc& like,
                                             bool packed);
  std::optional<TensorId> MakeStrided(const Strided& v, const ir::TensorDesc& like, bool packed);
  static std::optional<Strided> Reshaped(const Strided& v, const Ne& to);
  std::optional<Strided> Viewed(const Strided& v, const ir::TensorDesc& src,
                                const ir::TensorDesc& out) const;
  void Context(const ir::Node& n);
  bool Copy(const ir::Node& n);
  bool Elementwise(const ir::Node& n);

  const ir::Graph& g_;
  bool through_ = false;
  ir::Builder b_;
  std::vector<Val> vals_;
  std::map<std::pair<std::uint32_t, std::array<std::uint8_t, 4>>, TensorId> reordered_;
  // Made views of storage values, by (storage, extents, strides, offset,
  // packed): one materialization a value however many readers ask.
  std::map<std::tuple<std::uint32_t, Ne, Ne, std::uint64_t, bool>, TensorId> made_;
  std::vector<NodeId> replaced_;
  std::vector<std::map<Axes, std::uint32_t>> wants_;
  std::vector<Cheap> cheap_;
  std::map<std::pair<std::uint32_t, Axes>, TensorId> cheap_made_;
  bool failed_ = false;
  std::string why_;
};

void Sinker::Prefer() {
  for (const TensorId t : g_.outputs()) Want(t, NaturalAxes(NeOf(g_.tensor(t))));
  for (const TensorId t : g_.keep()) Want(t, NaturalAxes(NeOf(g_.tensor(t))));
  for (std::size_t i = g_.nodes().size(); i-- > 0;) {
    const ir::Node& n = g_.nodes()[i];
    const auto natural = [&] {
      for (const TensorId s : n.sources())
        if (s.valid()) Want(s, NaturalAxes(NeOf(g_.tensor(s))));
    };
    if (n.nout != 1 || n.key.fused != ir::FusedKind::kNone) {
      natural();
      continue;
    }
    const TensorId out = n.out[0];
    const auto wanted = Wanted(out);
    const Ne one = NeOf(g_.tensor(out));
    const auto layout = ir::Traits(n.key)->layout;
    if (layout == ir::LayoutClass::kElementwise || layout == ir::LayoutClass::kConcat) {
      if (wanted && through_) {
        for (const TensorId src : n.sources()) {
          if (!src.valid()) continue;
          const Ne ne = NeOf(g_.tensor(src));
          if (ne == one || n.key.op == Op::kConcat || n.key.op == Op::kRepeat) Want(src, *wanted);
        }
      }
      continue;
    }
    switch (n.key.op) {
      case Op::kPermute: {
        if (!wanted) break;
        const auto p = n.Params<ir::PermuteParams>().axes;
        Axes a{};
        for (std::size_t k = 0; k < 4; ++k)
          for (std::uint8_t d = 0; d < 4; ++d)
            if (p[d] == (*wanted)[k]) a[k] = d;
        Want(n.src[0], a);
        break;
      }
      case Op::kReshape:
      case Op::kCont: {
        if (!wanted) break;
        const Ne from = NeOf(g_.tensor(n.src[0]));
        std::vector<std::uint8_t> wide_out, wide_in;
        for (std::uint8_t d = 0; d < 4; ++d) {
          if (one[d] != 1) wide_out.push_back(d);
          if (from[d] != 1) wide_in.push_back(d);
        }
        bool same = wide_out.size() == wide_in.size();
        for (std::size_t k = 0; same && k < wide_out.size(); ++k)
          same = one[wide_out[k]] == from[wide_in[k]];
        if (same) {
          Axes a = NaturalAxes(from);
          std::size_t m = 0;
          for (const std::uint8_t d : *wanted) {
            if (one[d] == 1) continue;
            const auto at = std::ranges::find(wide_out, d) - wide_out.begin();
            a[m++] = wide_in[static_cast<std::size_t>(at)];
          }
          Want(n.src[0], a);
        } else if (*wanted == NaturalAxes(one)) {
          Want(n.src[0], NaturalAxes(from));
        }
        break;
      }
      case Op::kView:
        break;
      default:
        natural();
        break;
    }
  }
}

std::optional<Strided> Sinker::Lift(TensorId t) {
  Val& v = vals_.at(t.v);
  // A kept concatenation a reader needs whole is made (Unjoin); a cheap
  // value in its logical order.
  if (v.joined) return Unjoin(t);
  if (v.cheap && !v.lazy) {
    const Ne ne = NeOf(g_.tensor(t));
    const TensorId made = EmitCheap(*v.cheap, NaturalAxes(ne));
    if (!made.valid()) return std::nullopt;
    v.lazy = Strided{.storage = made, .ne = ne, .st = NaturalOf(ne), .offset = 0};
  }
  if (v.lazy) return v.lazy;
  if (!v.mapped.valid()) return std::nullopt;
  const ir::TensorDesc& made = b_.tensor(v.mapped);
  if (!Dense(made) || !ir::IsContiguous(made)) return std::nullopt;
  const Ne ne = NeOf(made);
  return Strided{.storage = v.mapped, .ne = ne, .st = NaturalOf(ne), .offset = 0};
}

std::optional<Order> Sinker::CompactOrder(const Strided& v) const {
  if (v.offset != 0 || Count(v.ne) != StorageCount(v.storage)) return std::nullopt;
  std::array<std::uint8_t, 4> axes{};
  std::size_t n = 0;
  std::vector<std::uint8_t> wide;
  for (std::uint8_t d = 0; d < 4; ++d)
    if (v.ne[d] != 1) wide.push_back(d);
  std::ranges::stable_sort(wide, {}, [&](std::uint8_t d) { return v.st[d]; });
  std::uint64_t expect = 1;
  for (const std::uint8_t d : wide) {
    if (v.st[d] != expect) return std::nullopt;
    expect *= v.ne[d];
    axes[n++] = d;
  }
  for (std::uint8_t d = 0; d < 4; ++d)
    if (v.ne[d] == 1) axes[n++] = d;
  return Order{.axes = axes, .logical = std::ranges::is_sorted(wide)};
}

// The storage of compact `v`, shaped in its storage order.
TensorId Sinker::StorageAs(const Strided& v, const Order& o) {
  Ne shape{};
  for (std::size_t k = 0; k < 4; ++k) shape[k] = v.ne[o.axes[k]];
  if (NeOf(b_.tensor(v.storage)) == shape) return v.storage;
  return b_.Reshape(v.storage, ShapeOfNe(shape));
}

// Original tensor `original` (a value of the result's extents, or one
// broadcasting over them) in storage order `o`, packed.
TensorId Sinker::InOrder(TensorId original, const Order& o) {
  const auto key = std::pair{original.v, o.axes};
  if (const auto it = reordered_.find(key); it != reordered_.end()) return it->second;
  auto lifted = Lift(original);
  if (!lifted) {
    const auto made = Materialize(original, true);
    if (!made) {
      failed_ = true;
      return {};
    }
    lifted = Lift(original);
    if (!lifted) {
      failed_ = true;
      return {};
    }
  }
  // The value permuted into the storage order: storage axis k reads its
  // logical axis o.axes[k].
  Strided p = *lifted;
  for (std::size_t k = 0; k < 4; ++k) {
    p.ne[k] = lifted->ne[o.axes[k]];
    p.st[k] = lifted->st[o.axes[k]];
  }
  TensorId out{};
  if (const auto c = CompactOrder(p); c && c->natural()) {
    // Its wide axes already in order: the storage as these extents (a
    // broadcast operand's unit axes where the result's order puts them).
    out = StorageAs(p, *c);
    if (NeOf(b_.tensor(out)) != p.ne) out = b_.Reshape(out, ShapeOfNe(p.ne));
  } else {
    ir::TensorDesc like = g_.tensor(original);
    like.format.shape = ShapeOfNe(p.ne);
    const auto made = MaterializeStrided(p, like, true);
    if (!made) {
      failed_ = true;
      return {};
    }
    out = *made;
  }
  reordered_.emplace(key, out);
  return out;
}

// `v` made a tensor of its logical layout: a view (and a permute) of its
// storage, then a Cont where `packed`.
std::optional<TensorId> Sinker::MaterializeStrided(const Strided& v, const ir::TensorDesc& like,
                                                   bool packed) {
  if (v.offset == 0 && NaturalOn(v.ne, v.st) && Count(v.ne) == StorageCount(v.storage)) {
    if (NeOf(b_.tensor(v.storage)) == v.ne) return v.storage;
    packed = false;  // a reshape of a packed storage is packed
  }
  const auto key = std::tuple{v.storage.v, v.ne, v.st, v.offset, packed};
  if (const auto it = made_.find(key); it != made_.end()) return it->second;
  auto made = MakeStrided(v, like, packed);
  if (made) made_.emplace(key, *made);
  return made;
}

std::optional<TensorId> Sinker::MakeStrided(const Strided& v, const ir::TensorDesc& like,
                                            bool packed) {
  if (v.offset == 0 && NaturalOn(v.ne, v.st) && Count(v.ne) == StorageCount(v.storage))
    return b_.Reshape(v.storage, ShapeOfNe(v.ne));
  const std::uint64_t unit = ir::UnitBytes(like.format);
  // The view's axes: the wide ones by stride (the innermost must step one
  // element: a GGML view's first stride is the element's), then the rest.
  std::vector<std::uint8_t> wide;
  for (std::uint8_t d = 0; d < 4; ++d)
    if (v.ne[d] != 1) wide.push_back(d);
  std::ranges::stable_sort(wide, {}, [&](std::uint8_t d) { return v.st[d]; });
  std::vector<std::uint8_t> order;
  bool lead = !wide.empty() && v.st[wide.front()] != 1;
  if (lead && wide.size() == 4) return std::nullopt;
  std::vector<std::uint8_t> units;
  for (std::uint8_t d = 0; d < 4; ++d)
    if (v.ne[d] == 1) units.push_back(d);
  // A leading extent-one axis carries the element stride where no wide
  // axis steps one element.
  if (lead) {
    order.push_back(units.front());
    units.erase(units.begin());
  }
  order.insert(order.end(), wide.begin(), wide.end());
  order.insert(order.end(), units.begin(), units.end());
  Ne vne{};
  std::array<std::uint64_t, 4> nb{};
  std::uint64_t last = unit;
  for (std::size_t j = 0; j < 4; ++j) {
    const std::uint8_t d = order[j];
    vne[j] = v.ne[d];
    if (j == 0) {
      nb[j] = unit;
    } else if (v.ne[d] == 1) {
      nb[j] = last;
    } else {
      nb[j] = v.st[d] * unit;
    }
    last = nb[j] * vne[j];
  }
  TensorId out = b_.View(v.storage, ShapeOfNe(vne), nb, v.offset * unit);
  // View axis j is logical axis order[j]: a permute puts it there.
  std::array<std::uint8_t, 4> axes{};
  for (std::size_t j = 0; j < 4; ++j) axes[j] = order[j];
  if (axes != std::array<std::uint8_t, 4>{0, 1, 2, 3}) out = b_.Permute(out, axes);
  if (packed) out = b_.Cont(out);
  return out;
}

// Original tensor `t`'s value as a tensor of the new graph: its own where
// it was made; else its view, packed where `packed`.
std::optional<TensorId> Sinker::Materialize(TensorId t, bool packed) {
  if (vals_.at(t.v).joined && !Unjoin(t)) return std::nullopt;
  if (vals_.at(t.v).cheap && !vals_.at(t.v).lazy && !Lift(t)) return std::nullopt;
  Val& v = vals_.at(t.v);
  if (v.mapped.valid() && (!packed || ir::IsContiguous(b_.tensor(v.mapped)))) return v.mapped;
  if (packed && v.packed.valid()) return v.packed;
  if (!v.lazy) {
    if (!v.mapped.valid()) return std::nullopt;
    const TensorId c = b_.Cont(v.mapped);
    v.packed = c;
    return c;
  }
  const auto made = MaterializeStrided(*v.lazy, g_.tensor(t), packed);
  if (!made) return std::nullopt;
  if (packed) {
    v.packed = *made;
  } else if (!v.mapped.valid()) {
    v.mapped = *made;
  }
  return made;
}

// `v` reshaped to `to` without copying, where its strides allow it (the
// extents merged or split lie in runs of packed axes).
std::optional<Strided> Sinker::Reshaped(const Strided& v, const Ne& to) {
  if (Count(v.ne) != Count(to)) return std::nullopt;
  std::vector<std::uint8_t> from_wide, to_wide;
  for (std::uint8_t d = 0; d < 4; ++d) {
    if (v.ne[d] != 1) from_wide.push_back(d);
    if (to[d] != 1) to_wide.push_back(d);
  }
  Strided out{.storage = v.storage, .ne = to, .st = {}, .offset = v.offset};
  std::size_t i = 0, j = 0;
  while (i < from_wide.size() && j < to_wide.size()) {
    // A run of source axes and of target axes with equal products.
    std::size_t i1 = i, j1 = j;
    std::uint64_t pa = v.ne[from_wide[i]], pb = to[to_wide[j]];
    while (pa != pb) {
      if (pa < pb) {
        if (++i1 >= from_wide.size()) return std::nullopt;
        pa *= v.ne[from_wide[i1]];
      } else {
        if (++j1 >= to_wide.size()) return std::nullopt;
        pb *= to[to_wide[j1]];
      }
    }
    for (std::size_t k = i; k < i1; ++k) {
      const std::uint8_t a = from_wide[k], c = from_wide[k + 1];
      if (v.st[c] != v.st[a] * v.ne[a]) return std::nullopt;
    }
    std::uint64_t s = v.st[from_wide[i]];
    for (std::size_t k = j; k <= j1; ++k) {
      out.st[to_wide[k]] = s;
      s *= to[to_wide[k]];
    }
    i = i1 + 1;
    j = j1 + 1;
  }
  if (i != from_wide.size() || j != to_wide.size()) return std::nullopt;
  for (std::uint8_t d = 0; d < 4; ++d)
    if (to[d] == 1) out.st[d] = 1;
  return out;
}

// View `out` of the logical contiguous `src` (held as `v`), as a view of
// v's storage, where each of its strides steps within one of src's axes.
std::optional<Sinker::ViewMap> Sinker::MapView(const Ne& ne, const ir::TensorDesc& src,
                                               const ir::TensorDesc& out) {
  const std::uint64_t unit = ir::UnitBytes(src.format);
  if (unit == 0 || ir::UnitBytes(out.format) != unit) return std::nullopt;
  const std::uint64_t base = src.is_view() ? src.view_offset : 0;
  if (out.view_offset < base || (out.view_offset - base) % unit != 0) return std::nullopt;
  const Ne cs = NaturalOf(ne);
  ViewMap m;
  // The offset as an index of src's axes.
  std::uint64_t rest = (out.view_offset - base) / unit;
  for (std::size_t d = 4; d-- > 0;) {
    m.at[d] = rest / cs[d];
    rest %= cs[d];
    if (m.at[d] >= ne[d]) return std::nullopt;
  }
  m.used = m.at;
  for (std::size_t j = 0; j < 4; ++j) {
    m.steps[j] = {-1, 0};
    if (out.ne(j) == 1) continue;
    if (out.nb.at(j) % unit != 0) return std::nullopt;
    const std::uint64_t e = out.nb.at(j) / unit;
    std::optional<std::size_t> axis;
    for (std::size_t d = 4; d-- > 0;) {
      if (ne[d] != 1 && cs[d] <= e && e % cs[d] == 0) {
        axis = d;
        break;
      }
    }
    if (!axis) return std::nullopt;
    const std::uint64_t f = e / cs[*axis];
    m.used[*axis] += f * (out.ne(j) - 1);
    if (m.used[*axis] >= ne[*axis]) return std::nullopt;
    m.steps[j] = {static_cast<std::int8_t>(*axis), f};
  }
  return m;
}

// `m` over `v`, the view starting at `at` of v's axes.
Strided Sinker::Apply(const Strided& v, const ViewMap& m, const Ne& at, const ir::TensorDesc& out) {
  Strided r{.storage = v.storage, .ne = NeOf(out), .st = {}, .offset = v.offset};
  for (std::size_t d = 0; d < 4; ++d) r.offset += at[d] * v.st[d];
  for (std::size_t j = 0; j < 4; ++j)
    r.st[j] = m.steps[j].first < 0
                  ? 1
                  : v.st[static_cast<std::size_t>(m.steps[j].first)] * m.steps[j].second;
  return r;
}

// View `out` of the logical contiguous `src` (held as `v`), as a view of
// v's storage, where each of its strides steps within one of src's axes.
std::optional<Strided> Sinker::Viewed(const Strided& v, const ir::TensorDesc& src,
                                      const ir::TensorDesc& out) const {
  const auto m = MapView(v.ne, src, out);
  if (!m) return std::nullopt;
  return Apply(v, *m, m->at, out);
}

std::optional<Sinker::Joined> Sinker::JoinedOf(TensorId t) {
  if (vals_.at(t.v).joined) return vals_.at(t.v).joined;
  const auto l = Lift(t);
  if (!l) return std::nullopt;
  return Joined{.ne = l->ne, .axis = 0, .pieces = {{0, *l}}};
}

// A concatenation kept as its pieces (made where a reader needs it whole).
bool Sinker::Join(const ir::Node& n) {
  const auto a = JoinedOf(n.src[0]);
  const auto b = JoinedOf(n.src[1]);
  if (!a || !b) return false;
  const auto dim = n.Params<ir::ConcatParams>().dim;
  if ((a->pieces.size() > 1 && a->axis != dim) || (b->pieces.size() > 1 && b->axis != dim))
    return false;
  Joined j{.ne = NeOf(g_.tensor(n.out[0])), .axis = dim, .pieces = a->pieces};
  for (const auto& [start, piece] : b->pieces) j.pieces.emplace_back(start + a->ne[dim], piece);
  vals_.at(n.out[0].v).joined = std::move(j);
  return true;
}

// A kept concatenation made: its pieces joined in their storage order
// where they share one, else packed and joined; the value then that.
std::optional<Strided> Sinker::Unjoin(TensorId t) {
  Val& v = vals_.at(t.v);
  if (!v.joined) return v.lazy;
  const Joined j = *v.joined;
  v.joined.reset();
  std::optional<Order> shared;
  bool same = true;
  for (const auto& [start, piece] : j.pieces) {
    const auto o = CompactOrder(piece);
    if (!o || (shared && o->axes != shared->axes)) same = false;
    if (o && !shared) shared = o;
  }
  const ir::TensorDesc& like = g_.tensor(t);
  TensorId out{};
  Ne shape{};
  Order order{.axes = NaturalAxes(j.ne), .logical = true};
  if (same && shared && j.ne[j.axis] != 1) {
    order = *shared;
    std::uint8_t k = 0;
    while (k < 4 && order.axes[k] != j.axis) ++k;
    if (k == 4) return std::nullopt;
    for (const auto& [start, piece] : j.pieces) {
      const TensorId part = StorageAs(piece, order);
      out = out.valid() ? b_.Concat(out, part, k) : part;
    }
  } else {
    order = Order{.axes = NaturalAxes(j.ne), .logical = true};
    for (const auto& [start, piece] : j.pieces) {
      ir::TensorDesc part_like = like;
      part_like.format.shape = ShapeOfNe(piece.ne);
      const auto part = MaterializeStrided(piece, part_like, true);
      if (!part) return std::nullopt;
      out = out.valid() ? b_.Concat(out, *part, j.axis) : *part;
    }
  }
  if (!b_.ok()) return std::nullopt;
  for (std::size_t k = 0; k < 4; ++k) shape[k] = j.ne[order.axes[k]];
  if (NeOf(b_.tensor(out)) != shape) return std::nullopt;
  const Ne natural = NaturalOf(shape);
  Strided r{.storage = out, .ne = j.ne, .st = {}, .offset = 0};
  for (std::size_t k = 0; k < 4; ++k) r.st[order.axes[k]] = natural[k];
  v.lazy = r;
  return r;
}

// The node's component and owner (pattern.cc ContextOf's): what the nodes
// made at its place stand for.
void Sinker::Context(const ir::Node& n) {
  b_.SetComponent(n.component);
  if (n.batch == ir::Batch::kPerOwner && n.nout > 0) {
    b_.SetOwner(g_.tensor(n.out[0]).segment);
  } else {
    b_.SetJoined();
  }
}

// `n` made again over its sources' values, each as its original layout
// asked: packed where the original was.
bool Sinker::Copy(const ir::Node& n) {
  Context(n);
  std::vector<TensorId> sources;
  for (const TensorId s : n.sources()) {
    if (!s.valid()) {
      sources.push_back({});
      continue;
    }
    // Packed where the original was; else a view of the value.
    auto made = Materialize(s, ir::IsContiguous(g_.tensor(s)));
    if (!made) return false;
    sources.push_back(*made);
  }
  const TensorId out = n.nout > 0 ? n.out[0] : TensorId{};
  const ir::TensorDesc* od = out.valid() ? &g_.tensor(out) : nullptr;
  TensorId made{};
  switch (n.key.fused == ir::FusedKind::kNone ? n.key.op : Op::kFused) {
    case Op::kPermute:
      made = b_.Permute(sources[0], n.Params<ir::PermuteParams>().axes);
      break;
    case Op::kReshape:
      made = b_.Reshape(sources[0], od->shape());
      break;
    case Op::kCont:
      made = NeOf(*od) == NeOf(b_.tensor(sources[0])) ? b_.Cont(sources[0])
                                                      : b_.Cont(sources[0], od->shape());
      break;
    case Op::kView: {
      // The original view's bytes are its source's: the made source must be
      // laid out as the original was.
      const ir::TensorDesc& original = g_.tensor(n.src[0]);
      const ir::TensorDesc& source = b_.tensor(sources[0]);
      if (source.nb != original.nb || NeOf(source) != NeOf(original)) return false;
      const std::uint64_t base = original.is_view() ? original.view_offset : 0;
      if (od->view_offset < base) return false;
      made = b_.View(sources[0], od->shape(), od->nb, od->view_offset - base);
      break;
    }
    default: {
      std::vector<ir::TensorDesc> requested;
      for (const TensorId o : n.outputs()) requested.push_back(g_.tensor(o));
      const ir::Outputs outs = b_.Emit(n.key, n.params, sources, requested,
                                       out.valid() ? g_.name(out) : std::string_view());
      if (!b_.ok()) return false;
      for (std::size_t i = 0; i < n.nout && i < outs.n; ++i)
        vals_.at(n.out.at(i).v).mapped = outs[i];
      return true;
    }
  }
  if (!b_.ok()) return false;
  vals_.at(out.v).mapped = made;
  return true;
}

// An elementwise node (or a repeat) run in one storage order: the order
// of a full-extent operand that leaves the fewest others to reorder (one
// Cont each), the logical order on a tie (then it is copied as it is).
bool Sinker::Elementwise(const ir::Node& n) {
  const Ne out_ne = NeOf(g_.tensor(n.out[0]));
  std::vector<std::optional<Order>> orders;
  std::vector<bool> full;
  for (std::size_t i = 0; i < n.nsrc; ++i) {
    const TensorId s = n.src[i];
    std::optional<Order> o;
    bool f = false;
    if (s.valid() && vals_.at(s.v).cheap && !vals_.at(s.v).lazy) {
      // Made in whichever order is chosen: no reordering to count.
      orders.push_back(std::nullopt);
      full.push_back(false);
      continue;
    }
    if (s.valid()) {
      if (const auto l = Lift(s)) {
        o = CompactOrder(*l);
        f = l->ne == out_ne;
      } else {
        f = NeOf(g_.tensor(s)) == out_ne;
      }
    }
    orders.push_back(o);
    full.push_back(f);
  }
  std::optional<Order> best;
  if (n.key == Prim(Op::kRepeat)) {
    if (!orders[0] || orders[0]->natural()) return false;
    best = orders[0];
  } else {
    // The order its readers want, as an Order of the result's axes.
    std::optional<Order> wanted;
    if (const auto w = Wanted(n.out[0]))
      wanted = Order{.axes = *w, .logical = *w == NaturalAxes(out_ne)};
    const auto matches = [&](const std::optional<Order>& have, const Order& o) {
      return have && (o.logical ? have->logical : !have->logical && have->axes == o.axes);
    };
    const auto cost = [&](const Order& o) {
      std::size_t c = 0;
      for (std::size_t i = 0; i < n.nsrc; ++i)
        if (full[i] && !matches(orders[i], o)) ++c;
      if (wanted && !matches(wanted, o)) ++c;
      return c;
    };
    std::vector<Order> candidates;
    if (wanted && !wanted->logical) candidates.push_back(*wanted);
    for (std::size_t i = 0; i < n.nsrc; ++i)
      if (full[i] && orders[i] && !orders[i]->natural()) candidates.push_back(*orders[i]);
    // The wanted order first: it wins a tie; the logical one is the last.
    std::size_t least = cost(Order{.axes = NaturalAxes(out_ne), .logical = true});
    const bool want_logical = wanted && wanted->logical;
    for (const Order& o : candidates) {
      const std::size_t c = cost(o);
      if (c < least || (c == least && !want_logical && !best && wanted && matches(wanted, o))) {
        least = c;
        best = o;
      }
    }
    if (!best) return false;
  }
  Context(n);
  std::vector<TensorId> sources;
  for (std::size_t i = 0; i < n.nsrc; ++i) {
    const TensorId s = n.src[i];
    if (!s.valid()) {
      sources.push_back({});
      continue;
    }
    TensorId made{};
    if (vals_.at(s.v).cheap && !vals_.at(s.v).lazy) {
      // Its own extents in the chosen order of the result's axes.
      made = EmitCheap(*vals_.at(s.v).cheap, Restricted(best->axes, NeOf(g_.tensor(s))));
      const Ne sne = NeOf(g_.tensor(s));
      Ne shape{};
      for (std::size_t k = 0; k < 4; ++k) shape[k] = sne[best->axes[k]];
      if (made.valid() && NeOf(b_.tensor(made)) != shape) made = b_.Reshape(made, ShapeOfNe(shape));
    } else {
      made = InOrder(s, *best);
    }
    if (failed_ || !made.valid()) return false;
    sources.push_back(made);
  }
  Ne shape{};
  for (std::size_t k = 0; k < 4; ++k) shape[k] = out_ne[best->axes[k]];
  ir::TensorDesc request = g_.tensor(n.out[0]);
  request.format.shape = ShapeOfNe(shape);
  request.nb = ir::NaturalStrides(request.format);
  request.view_of = {};
  request.view_offset = 0;
  const std::array requested{request};
  const ir::Outputs outs = b_.Emit(n.key, n.params, sources, requested,
                                   n.nout > 0 ? g_.name(n.out[0]) : std::string_view());
  if (!b_.ok() || outs.n != 1) return false;
  const Ne natural = NaturalOf(shape);
  Strided r{.storage = outs[0], .ne = out_ne, .st = {}, .offset = 0};
  for (std::size_t k = 0; k < 4; ++k) r.st[best->axes[k]] = natural[k];
  vals_.at(n.out[0].v).lazy = r;
  return true;
}

// A cheap value for node `n` (see Cheap), where it is one: a repeat of a
// smaller tensor; a concatenation, an elementwise function, a permute or a
// reshape of extents of one of cheap values.
std::optional<std::uint32_t> Sinker::CheapOf(const ir::Node& n) {
  if (n.nout != 1 || n.key.fused != ir::FusedKind::kNone) return std::nullopt;
  const Ne ne = NeOf(g_.tensor(n.out[0]));
  const auto cheap = [&](TensorId t) -> std::optional<std::uint32_t> {
    if (!t.valid()) return std::nullopt;
    const Val& v = vals_.at(t.v);
    if (v.cheap && !v.lazy) return v.cheap;
    return std::nullopt;
  };
  Cheap c{.ne = ne};
  switch (n.key.op) {
    case Op::kRepeat: {
      const TensorId src = n.src[0];
      const Ne sne = NeOf(g_.tensor(src));
      if (sne == ne) return std::nullopt;
      for (std::size_t d = 0; d < 4; ++d)
        if (sne[d] != ne[d] && sne[d] != 1) return std::nullopt;  // repeats of 1 only
      const auto small = Materialize(src, true);
      if (!small) return std::nullopt;
      c.kind = Cheap::Kind::kRepeat;
      c.small = *small;
      c.small_ne = sne;
      break;
    }
    case Op::kConcat: {
      const auto a = cheap(n.src[0]), b = cheap(n.src[1]);
      if (!a || !b) return std::nullopt;
      c.kind = Cheap::Kind::kConcat;
      c.a = *a;
      c.b = *b;
      c.axis = n.Params<ir::ConcatParams>().dim;
      break;
    }
    case Op::kScale:
    case Op::kUnary:
    case Op::kClamp: {
      const auto a = cheap(n.src[0]);
      if (!a || n.nsrc != 1) return std::nullopt;
      c.kind = Cheap::Kind::kMap;
      c.a = *a;
      c.node = g_.producer(n.out[0]);
      break;
    }
    case Op::kPermute: {
      const auto a = cheap(n.src[0]);
      if (!a) return std::nullopt;
      const auto axes = n.Params<ir::PermuteParams>().axes;
      c.kind = Cheap::Kind::kAxes;
      c.a = *a;
      for (std::uint8_t d = 0; d < 4; ++d) c.from[axes[d]] = d;
      break;
    }
    case Op::kReshape:
    case Op::kCont: {
      const auto a = cheap(n.src[0]);
      if (!a) return std::nullopt;
      const Ne from = cheap_.at(*a).ne;
      if (from == ne) return a;  // a Cont of it
      std::vector<std::uint8_t> wide_from, wide_to;
      for (std::uint8_t d = 0; d < 4; ++d) {
        if (from[d] != 1) wide_from.push_back(d);
        if (ne[d] != 1) wide_to.push_back(d);
      }
      if (wide_from.size() != wide_to.size()) return std::nullopt;
      for (std::size_t k = 0; k < wide_to.size(); ++k)
        if (from[wide_from[k]] != ne[wide_to[k]]) return std::nullopt;
      c.kind = Cheap::Kind::kAxes;
      c.a = *a;
      for (std::size_t k = 0; k < wide_to.size(); ++k) c.from[wide_to[k]] = wide_from[k];
      break;
    }
    default:
      return std::nullopt;
  }
  cheap_.push_back(c);
  return static_cast<std::uint32_t>(cheap_.size() - 1);
}

// Cheap value `i` made packed in storage order `order` (its axes,
// innermost first): a tensor of its extents in that order.
TensorId Sinker::EmitCheap(std::uint32_t i, const Axes& order) {
  const auto key = std::pair{i, order};
  if (const auto it = cheap_made_.find(key); it != cheap_made_.end()) return it->second;
  const Cheap c = cheap_.at(i);
  Ne shape{};
  for (std::size_t k = 0; k < 4; ++k) shape[k] = c.ne[order[k]];
  TensorId out{};
  switch (c.kind) {
    case Cheap::Kind::kRepeat: {
      // The small tensor in the order (a permute of extents of one is a
      // reshape; else a small copy), repeated.
      Strided p{.storage = c.small, .ne = {}, .st = {}, .offset = 0};
      const Ne natural = NaturalOf(c.small_ne);
      for (std::size_t k = 0; k < 4; ++k) {
        p.ne[k] = c.small_ne[order[k]];
        p.st[k] = natural[order[k]];
      }
      TensorId small{};
      if (const auto o = CompactOrder(p); o && o->natural()) {
        small = StorageAs(p, *o);
      } else {
        ir::TensorDesc like = b_.tensor(c.small);
        like.format.shape = ShapeOfNe(p.ne);
        const auto made = MaterializeStrided(p, like, true);
        if (!made) return {};
        small = *made;
      }
      if (NeOf(b_.tensor(small)) != p.ne) small = b_.Reshape(small, ShapeOfNe(p.ne));
      out = p.ne == shape ? small : b_.Repeat(small, ShapeOfNe(shape));
      break;
    }
    case Cheap::Kind::kConcat: {
      std::uint8_t k = 0;
      while (k < 4 && order[k] != c.axis) ++k;
      if (k == 4) return {};
      const TensorId a = EmitCheap(c.a, Restricted(order, cheap_.at(c.a).ne));
      const TensorId b = EmitCheap(c.b, Restricted(order, cheap_.at(c.b).ne));
      if (!a.valid() || !b.valid()) return {};
      // Both children laid out with the result's axis order (their own
      // extents): reshaped to it, then joined.
      const auto in_order = [&](TensorId t, const Ne& ne) {
        Ne s{};
        for (std::size_t j = 0; j < 4; ++j) s[j] = ne[order[j]];
        return NeOf(b_.tensor(t)) == s ? t : b_.Reshape(t, ShapeOfNe(s));
      };
      out = b_.Concat(in_order(a, cheap_.at(c.a).ne), in_order(b, cheap_.at(c.b).ne), k);
      break;
    }
    case Cheap::Kind::kMap: {
      const TensorId a = EmitCheap(c.a, order);
      if (!a.valid()) return {};
      const ir::Node& n = g_.node(c.node);
      Context(n);
      ir::TensorDesc request = g_.tensor(n.out[0]);
      request.format.shape = ShapeOfNe(shape);
      request.nb = ir::NaturalStrides(request.format);
      request.view_of = {};
      request.view_offset = 0;
      const std::array<TensorId, 1> src{
          NeOf(b_.tensor(a)) == shape ? a : b_.Reshape(a, ShapeOfNe(shape))};
      const std::array requested{request};
      out = b_.Emit(n.key, n.params, src, requested)[0];
      break;
    }
    case Cheap::Kind::kAxes: {
      // The child in the order its axes have here: the same elements.
      const Ne child = cheap_.at(c.a).ne;
      Axes child_order{};
      std::size_t m = 0;
      for (const std::uint8_t d : order)
        if (c.ne[d] != 1) child_order[m++] = c.from[d];
      for (std::uint8_t d = 0; d < 4; ++d)
        if (child[d] == 1) child_order[m++] = d;
      if (m != 4) return {};
      const TensorId a = EmitCheap(c.a, child_order);
      if (!a.valid()) return {};
      out = NeOf(b_.tensor(a)) == shape ? a : b_.Reshape(a, ShapeOfNe(shape));
      break;
    }
  }
  if (!b_.ok() || !out.valid()) return {};
  cheap_made_.emplace(key, out);
  return out;
}

bool Supported(const ir::Graph& g) {
  for (const ir::Node& n : g.nodes()) {
    if (n.key == Prim(Op::kState) || n.key == Prim(Op::kCacheStore)) return false;
    if (n.key.op == Op::kFused) {
      for (std::size_t i = 0; i < n.nout; ++i)
        if (ir::AliasedSource(n.key, i)) return false;
    }
  }
  return true;
}

std::expected<Sunk, ir::VerifyError> Sinker::Run() {
  Prefer();
  // The step's outputs and kept tensors keep the original's layout: their
  // producers and the views they are are copied.
  // (A packed output is made packed at the end whatever its producer: only
  // an output or kept tensor that is a view keeps its producers' layout.)
  std::vector<bool> external(g_.tensor_count(), false);
  const auto mark = [&](TensorId t) {
    const ir::TensorDesc& d = g_.tensor(t);
    if (d.is_view() || !ir::IsContiguous(d)) external.at(t.v) = true;
  };
  for (const TensorId t : g_.outputs()) mark(t);
  for (const TensorId t : g_.keep()) mark(t);
  for (std::uint32_t i = 0; i < g_.nodes().size(); ++i) {
    const ir::Node& n = g_.nodes()[i];
    const std::size_t before = b_.node_count();
    const bool kept = std::ranges::any_of(n.outputs(), [&](TensorId o) { return external[o.v]; });
    bool done = false;
    if (!kept && n.nout == 1) {
      const TensorId out = n.out[0];
      const ir::TensorDesc& od = g_.tensor(out);
      if (ir::Traits(n.key)->layout == ir::LayoutClass::kElementwise) {
        if (n.key.op == Op::kGlu) {
          done = n.nsrc == 2 && n.src[1].valid() && Dense(od) && Elementwise(n);
        } else if (const auto cheap = Dense(od) ? CheapOf(n) : std::nullopt) {
          vals_.at(out.v).cheap = cheap;
          done = true;
        } else {
          done = Dense(od) && Elementwise(n);
        }
      } else {
        switch (n.key.fused == ir::FusedKind::kNone ? n.key.op : Op::kFused) {
          case Op::kPermute: {
            if (const auto c = CheapOf(n)) {
              vals_.at(out.v).cheap = c;
              done = true;
            } else if (const auto& j = vals_.at(n.src[0].v).joined) {
              const auto axes = n.Params<ir::PermuteParams>().axes;
              Joined p{.ne = {}, .axis = axes[j->axis], .pieces = {}};
              for (std::size_t d = 0; d < 4; ++d) p.ne[axes[d]] = j->ne[d];
              for (const auto& [start, piece] : j->pieces) {
                Strided q = piece;
                for (std::size_t d = 0; d < 4; ++d) {
                  q.ne[axes[d]] = piece.ne[d];
                  q.st[axes[d]] = piece.st[d];
                }
                p.pieces.emplace_back(start, q);
              }
              vals_.at(out.v).joined = std::move(p);
              done = true;
            } else if (const auto l = Lift(n.src[0])) {
              const auto axes = n.Params<ir::PermuteParams>().axes;
              Strided p = *l;
              for (std::size_t d = 0; d < 4; ++d) {
                p.ne[axes[d]] = l->ne[d];
                p.st[axes[d]] = l->st[d];
              }
              vals_.at(out.v).lazy = p;
              done = true;
            }
            break;
          }
          case Op::kReshape:
          case Op::kCont: {
            if (const auto c = CheapOf(n)) {
              vals_.at(out.v).cheap = c;
              done = true;
              break;
            }
            if (const auto& j = vals_.at(n.src[0].v).joined) {
              // Extents of one added or dropped only: the pieces each so.
              const Ne to = NeOf(od);
              std::vector<std::uint8_t> wide_from, wide_to;
              for (std::uint8_t d = 0; d < 4; ++d) {
                if (j->ne[d] != 1) wide_from.push_back(d);
                if (to[d] != 1) wide_to.push_back(d);
              }
              bool same = wide_from.size() == wide_to.size();
              for (std::size_t k = 0; same && k < wide_from.size(); ++k)
                same = j->ne[wide_from[k]] == to[wide_to[k]];
              const auto at = std::ranges::find(wide_from, j->axis) - wide_from.begin();
              if (same && static_cast<std::size_t>(at) < wide_from.size()) {
                const std::uint8_t axis = wide_to[static_cast<std::size_t>(at)];
                Joined r{.ne = to, .axis = axis, .pieces = {}};
                for (const auto& [start, piece] : j->pieces) {
                  Ne shape = to;
                  shape[axis] = piece.ne[j->axis];
                  const auto q = Reshaped(piece, shape);
                  if (!q) {
                    same = false;
                    break;
                  }
                  r.pieces.emplace_back(start, *q);
                }
                if (same) {
                  vals_.at(out.v).joined = std::move(r);
                  done = true;
                  break;
                }
              }
            }
            if (const auto l = Lift(n.src[0])) {
              if (const auto r = Reshaped(*l, NeOf(od))) {
                vals_.at(out.v).lazy = *r;
                done = true;
              }
            }
            break;
          }
          case Op::kView: {
            const ir::TensorDesc& src = g_.tensor(n.src[0]);
            if (ir::IsContiguous(src) && Dense(src) && vals_.at(n.src[0].v).joined) {
              // A view within one piece is a view of it.
              const Joined& j = *vals_.at(n.src[0].v).joined;
              if (const auto m = MapView(j.ne, src, od)) {
                for (const auto& [start, piece] : j.pieces) {
                  if (m->at[j.axis] < start || m->used[j.axis] >= start + piece.ne[j.axis])
                    continue;
                  Ne at = m->at;
                  at[j.axis] -= start;
                  vals_.at(out.v).lazy = Apply(piece, *m, at, od);
                  done = true;
                  break;
                }
              }
              if (done) break;
            }
            if (ir::IsContiguous(src) && Dense(src)) {
              if (const auto l = Lift(n.src[0])) {
                if (const auto r = Viewed(*l, src, od)) {
                  vals_.at(out.v).lazy = *r;
                  done = true;
                }
              }
            }
            break;
          }
          case Op::kConcat:
            if (const auto c = Dense(od) ? CheapOf(n) : std::nullopt) {
              vals_.at(out.v).cheap = c;
              done = true;
            } else {
              done = Dense(od) && Join(n);
            }
            break;
          default:
            break;
        }
      }
      if (failed_ || !b_.ok()) {
        why_ = std::format(
            "node {} ({}): {}", i, ir::OpKeyName(n.key),
            b_.ok() ? std::string("an operand could not be reordered") : b_.error()->detail);
        break;
      }
    }
    if (done) {
      replaced_.push_back(NodeId{i});
    } else if (!Copy(n)) {
      failed_ = true;
      why_ = std::format("node {} ({}) could not be made again{}", i, ir::OpKeyName(n.key),
                         b_.ok() ? "" : ": " + b_.error()->detail);
      break;
    }
    if (!b_.ok()) break;
    (void)before;
  }
  if (failed_ || !b_.ok()) {
    return std::unexpected(ir::VerifyError{
        .error = ir::GraphError::kShape,
        .node = 0,
        .detail = "layout sinking could not rebuild the graph: " +
                  (why_.empty()
                       ? (b_.ok() ? std::string("a value could not be made") : b_.error()->detail)
                       : why_)});
  }
  // The step's outputs and kept tensors, packed: the new graph's own, which
  // the map names (a lowering finds its ports through it).
  for (const TensorId o : g_.outputs()) {
    const auto made = Materialize(o, true);
    if (!made) return std::unexpected(ir::VerifyError{.detail = "an output was not made"});
    vals_.at(o.v).mapped = *made;
    b_.Output(*made);
  }
  for (const TensorId k : g_.keep()) {
    const auto made = Materialize(k, true);
    if (!made) return std::unexpected(ir::VerifyError{.detail = "a kept tensor was not made"});
    vals_.at(k.v).mapped = *made;
    b_.Keep(*made);
  }
  auto graph = std::move(b_).Finish();
  if (!graph) return std::unexpected(graph.error());
  Sunk out{
      .graph = std::move(*graph), .tensors = {}, .rewritten = replaced_.size(), .declined = {}};
  out.tensors.reserve(vals_.size());
  for (const Val& v : vals_) out.tensors.push_back(v.mapped);
  return out;
}

// The graph copied unchanged (what the pass does not handle).
std::expected<Sunk, ir::VerifyError> Unchanged(const ir::Graph& g) {
  ir::Builder b;
  std::vector<TensorId> map(g.tensor_count(), TensorId{});
  for (std::uint32_t i = 0; i < g.nodes().size(); ++i) {
    const ir::Node& n = g.nodes()[i];
    b.SetComponent(n.component);
    if (n.batch == ir::Batch::kPerOwner && n.nout > 0) {
      b.SetOwner(g.tensor(n.out[0]).segment);
    } else {
      b.SetJoined();
    }
    std::vector<TensorId> sources;
    for (const TensorId s : n.sources()) sources.push_back(s.valid() ? map.at(s.v) : TensorId{});
    std::vector<ir::TensorDesc> requested;
    for (const TensorId o : n.outputs()) requested.push_back(g.tensor(o));
    const ir::Outputs outs = b.Emit(n.key, n.params, sources, requested,
                                    n.nout > 0 ? g.name(n.out[0]) : std::string_view());
    if (!b.ok()) {
      ir::VerifyError e = *b.error();
      e.node = i;
      return std::unexpected(std::move(e));
    }
    for (std::size_t k = 0; k < n.nout && k < outs.n; ++k) map.at(n.out.at(k).v) = outs[k];
  }
  for (const TensorId o : g.outputs()) b.Output(map.at(o.v));
  for (const TensorId k : g.keep()) b.Keep(map.at(k.v));
  auto graph = std::move(b).Finish();
  if (!graph) return std::unexpected(graph.error());
  return Sunk{
      .graph = std::move(*graph), .tensors = std::move(map), .rewritten = 0, .declined = {}};
}

// The bytes a graph copies: its Cont and Concat results.
std::uint64_t Copied(const ir::Graph& g) {
  std::uint64_t bytes = 0;
  for (const ir::Node& n : g.nodes()) {
    if (n.key != Prim(Op::kCont) && n.key != Prim(Op::kConcat)) continue;
    const ir::TensorDesc& t = g.tensor(n.out[0]);
    bytes += ir::Elements(t.shape()) * ir::UnitBytes(t.format);
  }
  return bytes;
}

}  // namespace

std::expected<Sunk, ir::VerifyError> SinkLayouts(const ir::Graph& graph) {
  if (!Supported(graph)) return Unchanged(graph);
  // Each reader's wants passed through views only, and through elementwise
  // nodes too: the graph that copies fewer bytes.
  Sinker views(graph, false), through(graph, true);
  auto a = views.Run();
  auto b = through.Run();
  if (!a && !b) {
    // Neither form rebuilds: the graph as it is, the reason kept.
    auto same = Unchanged(graph);
    if (same) same->declined = std::format("{} (node {})", a.error().detail, a.error().node);
    return same;
  }
  auto best = !a                                           ? std::move(b)
              : !b || Copied(a->graph) <= Copied(b->graph) ? std::move(a)
                                                           : std::move(b);
  // The original where neither copies fewer bytes (its components' own
  // forms are the ones their patterns know: the decoder's channel norms).
  if (Copied(best->graph) >= Copied(graph)) return Unchanged(graph);
  return best;
}

}  // namespace llmp::ir
