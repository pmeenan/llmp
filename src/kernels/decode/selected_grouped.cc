// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernels/decode/selected_grouped.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "kernels/common/grouped_worklist.h"
#include "kernels/decode/grouped_binding.h"
#include "kernels/decode/grouped_route.h"
#include "kernels/decode/implementations.h"
#include "kernels/decode/products.h"
#include "kernels/ggml/validate_util.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::decode {
namespace {
namespace kg = ggml;
using Failure = kg::KernelFailure;
constexpr std::string_view kNvfp4Name = "llmp.decode.grouped.nvfp4.split";
constexpr std::string_view kAffineName = "llmp.decode.grouped.affine.split";
// NVFP4 routed experts the importer relaid out into the scale-atom layout,
// decoded on chip (W4A16) where the block-scaled grouped GEMM is not taken.
constexpr std::string_view kNvfp4AtomsName = "llmp.decode.grouped.nvfp4.scale_atoms";
std::unexpected<Failure> Reject(std::string detail) {
  return std::unexpected(Failure{.detail = std::move(detail)});
}
std::uint64_t Address(const ggml_tensor* t) { return reinterpret_cast<std::uintptr_t>(t->data); }
struct Geometry {
  format::TensorFormat format;
  ir::WeightRef weight;
  std::array<kg::SelectedPart, format::PartList::kMax> parts{};
  std::array<std::array<std::int64_t, 4>, format::PartList::kMax> ne{};
  std::array<std::array<std::uint64_t, 4>, format::PartList::kMax> nb{};
  std::size_t count = 0;
  ggml_tensor* ids = nullptr;
  std::array<std::uint64_t, 4> ids_nb{};
  ggml_tensor* x = nullptr;
  ggml_tensor* y = nullptr;
  std::uint64_t used = 0, tokens = 0, lanes = 0, ids_bytes = 0, x_bytes = 0, y_bytes = 0;
};
std::expected<Geometry, Failure> Shape(const ir::Node& n,
                                       std::span<const kg::SelectedOperand> inputs,
                                       std::span<const kg::SelectedOperand> outputs) {
  if (n.key != ir::Primitive(ir::Op::kMatMulId) || n.nsrc != 3 || n.nout != 1 ||
      inputs.size() != 3 || outputs.size() != 1)
    return Reject("grouped selection needs one primitive MatMulId");
  const auto& w = inputs[0];
  const auto& x = inputs[1];
  const auto& ids = inputs[2];
  const auto& y = outputs[0];
  const auto& f = w.desc.format;
  if (!w.expert_weight || !w.expert_weight->expert_array ||
      (f.encoding != format::Encoding::kNvfp4 && f.encoding != format::Encoding::kAffine) ||
      (f.layout != format::Layout::kSplit &&
       !(f.layout == format::Layout::kScaleAtoms && f.encoding == format::Encoding::kNvfp4)) ||
      f.shape.rank != 3 || x.desc.ne(0) != f.shape.ne[0] ||
      y.desc.format.encoding != format::Encoding::kF32 ||
      y.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(y.desc) ||
      x.desc.format.encoding != format::Encoding::kF32 ||
      x.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(x.desc) ||
      ids.desc.format.encoding != format::Encoding::kI32 ||
      ids.desc.format.layout != format::Layout::kDense || ids.desc.nb[0] != sizeof(std::int32_t) ||
      ids.desc.nb[1] % sizeof(std::int32_t) != 0 ||
      ids.desc.nb[1] / sizeof(std::int32_t) > static_cast<std::uint64_t>(INT32_MAX) ||
      ids.desc.ne(0) > ids.desc.nb[1] / sizeof(std::int32_t) ||
      ids.desc.view_offset % sizeof(std::int32_t) != 0 || ids.desc.ne(2) != 1 ||
      ids.desc.ne(3) != 1 || x.desc.ne(3) != 1 || y.desc.ne(0) != f.shape.ne[1] ||
      y.desc.ne(1) != ids.desc.ne(0) || y.desc.ne(2) != ids.desc.ne(1) || y.desc.ne(3) != 1 ||
      x.desc.ne(2) != ids.desc.ne(1) || !x.parts.empty() || !ids.parts.empty() ||
      !y.parts.empty() || w.parts.size() > format::PartList::kMax ||
      !GroupedRouteBound({f.shape.ne[2], ids.desc.ne(0), ids.desc.ne(1), x.desc.ne(1)}))
    return Reject("grouped input lacks exact expert, original slot or F32 geometry");
  auto one = f;
  one.shape.rank = 2;
  one.shape.ne[2] = 1;
  const auto parts = format::PartsOf(one);
  if (!parts || parts->size() != w.parts.size()) return Reject("incomplete grouped parts");
  Geometry g{.format = f,
             .weight = *w.expert_weight,
             .parts = {},
             .ne = {},
             .nb = {},
             .count = w.parts.size(),
             .ids = ids.tensor,
             .x = x.tensor,
             .y = y.tensor,
             .used = ids.desc.ne(0),
             .tokens = ids.desc.ne(1),
             .lanes = x.desc.ne(1)};
  if (!g.ids) return Reject("missing grouped IDs descriptor");
  for (std::size_t d = 0; d != g.ids_nb.size(); ++d) {
    if (g.ids->nb[d] != ids.desc.nb[d]) return Reject("grouped IDs lowering changed its pitch");
    g.ids_nb[d] = ids.desc.nb[d];
  }
  if (__builtin_mul_overflow(g.tokens - 1, g.ids_nb[1], &g.ids_bytes) ||
      __builtin_add_overflow(g.ids_bytes, g.used * sizeof(std::int32_t), &g.ids_bytes))
    return Reject("grouped IDs source extent overflow");
  g.x_bytes = 4;
  g.y_bytes = 4;
  for (auto dimension : {f.shape.ne[0], g.tokens, g.lanes})
    if (__builtin_mul_overflow(g.x_bytes, dimension, &g.x_bytes))
      return Reject("grouped input extent overflow");
  for (auto dimension : {f.shape.ne[1], g.tokens, g.used})
    if (__builtin_mul_overflow(g.y_bytes, dimension, &g.y_bytes))
      return Reject("grouped output extent overflow");
  std::ranges::copy(w.parts, g.parts.begin());
  for (std::size_t p = 0; p != g.count; ++p) {
    if (!g.parts[p].tensor) return Reject("missing grouped part descriptor");
    for (std::size_t d = 0; d != 4; ++d) {
      g.ne[p][d] = g.parts[p].tensor->ne[d];
      g.nb[p][d] = g.parts[p].tensor->nb[d];
    }
  }
  return g;
}
std::expected<std::array<PitchedExpertPart, format::PartList::kMax>, Failure> Parts(
    const Geometry& g) {
  std::array<PitchedExpertPart, format::PartList::kMax> out{};
  for (std::size_t i = 0; i != g.count; ++i) {
    const auto& p = g.parts[i];
    if (!p.tensor || !kg::detail::Current(p.tensor) || !kg::detail::Extent(p.tensor) ||
        p.tensor->type != GGML_TYPE_I8 || p.tensor->ne[0] <= 0 ||
        p.tensor->ne[1] != static_cast<std::int64_t>(g.format.shape.ne[2]) ||
        p.tensor->ne[2] != 1 || p.tensor->ne[3] != 1 || p.tensor->nb[0] != 1 ||
        p.tensor->nb[1] < static_cast<std::uint64_t>(p.tensor->ne[0]))
      return Reject("grouped part binding changed");
    for (std::size_t d = 0; d != 4; ++d)
      if (p.tensor->ne[d] != g.ne[i][d] || p.tensor->nb[d] != g.nb[i][d])
        return Reject("grouped source descriptor changed after lowering");
    out[i] = {p.role, Address(p.tensor), ggml_nbytes(p.tensor), p.tensor->nb[1]};
  }
  for (const auto* t : {g.ids, g.x, g.y})
    if (!t || !kg::detail::Current(t) || !kg::detail::Extent(t) ||
        (t != g.ids && !ggml_is_contiguous(t)) || !Address(t) || Address(t) % 4 ||
        ggml_nbytes(t) > UINT64_MAX - Address(t))
      return Reject("grouped activation binding is not current and packed");
  for (std::size_t d = 0; d != g.ids_nb.size(); ++d)
    if (g.ids->nb[d] != g.ids_nb[d]) return Reject("grouped IDs pitch changed after lowering");
  if (g.ids_bytes > ggml_nbytes(g.ids) || g.ids_bytes > UINT64_MAX - Address(g.ids))
    return Reject("grouped IDs source extent changed");
  if (g.ids->type != GGML_TYPE_I32 || g.ids->ne[0] != static_cast<std::int64_t>(g.used) ||
      g.ids->ne[1] != static_cast<std::int64_t>(g.tokens) || g.ids->ne[2] != 1 ||
      g.ids->ne[3] != 1 || g.x->type != GGML_TYPE_F32 || g.y->type != GGML_TYPE_F32 ||
      g.x->ne[0] != static_cast<std::int64_t>(g.format.shape.ne[0]) ||
      g.x->ne[1] != static_cast<std::int64_t>(g.lanes) ||
      g.x->ne[2] != static_cast<std::int64_t>(g.tokens) || g.x->ne[3] != 1 ||
      g.y->ne[0] != static_cast<std::int64_t>(g.format.shape.ne[1]) ||
      g.y->ne[1] != static_cast<std::int64_t>(g.used) ||
      g.y->ne[2] != static_cast<std::int64_t>(g.tokens) || g.y->ne[3] != 1 ||
      ggml_nbytes(g.x) != g.x_bytes || ggml_nbytes(g.y) != g.y_bytes)
    return Reject("grouped activation extent changed");
  // No output alias with inputs or any consumed expert interval. Parts may be
  // interleaved views; their mutual validation is BindGroupedInto's exact sweep.
  const auto overlaps = [](std::uint64_t a, std::uint64_t n, std::uint64_t b, std::uint64_t m) {
    return a < b + m && b < a + n;
  };
  for (const auto* t : {g.ids, g.x})
    if (overlaps(Address(g.y), ggml_nbytes(g.y), Address(t), ggml_nbytes(t)))
      return Reject("grouped output overlaps an input");
  auto one = g.format;
  one.shape.rank = 2;
  one.shape.ne[2] = 1;
  const auto extents = format::PartsOf(one);
  if (!extents) return Reject("invalid grouped source format");
  for (std::size_t i = 0; i != g.count; ++i) {
    const auto part = std::ranges::find(extents->parts(), out[i].role, &format::PartExtent::role);
    if (part == extents->parts().end() || out[i].bytes > UINT64_MAX - out[i].address)
      return Reject("invalid grouped source part");
    for (std::uint64_t e = 0; e != g.format.shape.ne[2]; ++e) {
      if (e > (UINT64_MAX - out[i].address) / out[i].matrix_pitch)
        return Reject("grouped part pitch overflow");
      const auto address = out[i].address + e * out[i].matrix_pitch;
      if (part->readable > UINT64_MAX - address ||
          overlaps(Address(g.y), ggml_nbytes(g.y), address, part->readable))
        return Reject("grouped output overlaps source bytes");
    }
  }
  return out;
}
struct RoutePlace {
  kg::SelectedRouteBuffers buffers;
  bool assigned = false;
};
class Runtime final : public kg::SelectedLaunch {
 public:
  Runtime(const kg::SelectedSource* owner, providers::NativeStream stream)
      : kg::SelectedLaunch(owner, stream) {}
  bool faulted() const override { return failed; }
  // Whole graphs are deliberately unavailable. The routed program owns and
  // records separate admitted segments instead.
  bool failed = false;
};
class BoundCall final : public kg::BoundSelectedCall {
 public:
  BoundCall(Runtime& r, Geometry g, std::shared_ptr<const RoutePlace> place)
      : runtime_(r), geometry_(std::move(g)), place_(std::move(place)) {}
  std::expected<void, Failure> Run(providers::NativeStream stream) const override {
    const auto& g = geometry_;
    if (runtime_.failed || stream.handle != runtime_.stream().handle || !place_->assigned)
      return Reject("grouped runtime, stream or placed worklist differs");
    if (auto parts = Parts(g); !parts) return std::unexpected(parts.error());
    auto one = g.format;
    one.shape.rank = 2;
    one.shape.ne[2] = 1;
    const auto& b = place_->buffers;
    const auto result = RunGrouped(
        {.format = one,
         .experts = reinterpret_cast<const Operand*>(b.descriptors),
         .experts_count = g.format.shape.ne[2],
         .first = 0,
         .outputs = g.format.shape.ne[1],
         .offsets = reinterpret_cast<const std::int32_t*>(b.offsets),
         .pair_rows = reinterpret_cast<const std::int32_t*>(g.lanes == 1 ? b.tokens : b.slots),
         .pair_out = reinterpret_cast<const std::int32_t*>(b.slots),
         .pairs = g.tokens * g.used,
         .x = static_cast<const float*>(g.x->data),
         .x_rows = g.tokens * g.lanes,
         .x_stride = g.format.shape.ne[0],
         .y = static_cast<float*>(g.y->data),
         .y_rows = g.tokens * g.used,
         .y_stride = g.format.shape.ne[1],
         .rows_hint = g.tokens},
        stream.handle);
    if (!result) {
      runtime_.failed = true;
      return std::unexpected(Failure{.error = kg::KernelError::kUnknown, .detail = result.error()});
    }
    return {};
  }

 private:
  Runtime& runtime_;
  Geometry geometry_;
  std::shared_ptr<const RoutePlace> place_;
};
class Call final : public kg::SelectedCall {
 public:
  Call(const kg::SelectedSource* owner, Geometry geometry, base::Sha256Digest identity)
      : kg::SelectedCall(owner),
        geometry_(std::move(geometry)),
        identity_(identity),
        place_(std::make_shared<RoutePlace>()) {}
  std::uint64_t scratch_bytes() const override { return 0; }
  std::uint64_t host_bytes() const override { return 4096; }
  std::uint64_t launches() const override {
    return 2;
  }  // product plus shared route preparation bound
  std::optional<kg::SelectedRoute> route() const override {
    return kg::SelectedRoute{geometry_.ids, geometry_.weight, geometry_.format};
  }
  std::uint64_t route_descriptor_bytes() const override {
    return geometry_.format.shape.ne[2] * sizeof(Operand);
  }
  std::expected<void, Failure> RouteBuffers(kg::SelectedRouteBuffers buffers) override {
    if (place_->assigned || buffers.offsets % 256 || buffers.tokens % 256 || buffers.slots % 256 ||
        buffers.descriptors % 256 || !buffers.offsets || !buffers.tokens || !buffers.slots ||
        !buffers.descriptors)
      return Reject("grouped route placement is not a fresh aligned range");
    place_->buffers = buffers;
    place_->assigned = true;
    return {};
  }
  std::expected<void, Failure> StageDescriptors(std::span<std::byte> into) const override {
    if (into.size() != route_descriptor_bytes() ||
        reinterpret_cast<std::uintptr_t>(into.data()) % alignof(Operand))
      return Reject("grouped descriptor staging differs");
    const auto parts = Parts(geometry_);
    if (!parts) return std::unexpected(parts.error());
    const auto result =
        BindGroupedInto(geometry_.format, std::span(*parts).first(geometry_.count),
                        {reinterpret_cast<Operand*>(into.data()), geometry_.format.shape.ne[2]});
    if (!result) return Reject("grouped source part geometry is inconsistent");
    return {};
  }
  std::expected<std::unique_ptr<kg::BoundSelectedCall>, Failure> Bind(
      const execution::Implementation& implementation, kg::SelectedLaunch& launch,
      kg::SelectedWorkspace) const override {
    const auto native = decode::Bind(implementation);
    if (launch.owner() != owner() || execution::IdentityOf(implementation) != identity_ ||
        !native || native->tmpl != Template::kGrouped ||
        (native->gate != DecoderId::kNvfp4Split && native->gate != DecoderId::kAffineSplit) ||
        !place_->assigned)
      return Reject("grouped selection lacks its exact routed native declaration");
    if (auto parts = Parts(geometry_); !parts) return std::unexpected(parts.error());
    return std::make_unique<BoundCall>(static_cast<Runtime&>(launch), geometry_, place_);
  }

 private:
  Geometry geometry_;
  base::Sha256Digest identity_;
  std::shared_ptr<RoutePlace> place_;
};
class Source final : public kg::SelectedSource {
 public:
  explicit Source(std::vector<execution::Implementation> implementations)
      : implementations_(std::move(implementations)) {}
  std::vector<execution::Implementation> Implementations() const override {
    return implementations_;
  }
  std::vector<execution::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view name) const override {
    return std::ranges::any_of(implementations_,
                               [&](const auto& impl) { return impl.name == name; });
  }
  std::uint64_t persistent_bytes() const override { return 0; }
  std::optional<base::Sha256Digest> ExecutionIdentity() const override {
    base::Sha256 hash;
    hash.Update("llmp-native-split-routed-group-v2");
    hash.Update(
        "direct-weight-parts;original-route-ids;complete-read-inventory;"
        "pitched-stacked-experts;token-or-original-slot;inverse-output-slot;affine-and-nvfp4");
    for (const auto& impl : implementations_) {
      const auto implementation = execution::IdentityOf(impl);
      hash.Update(std::as_bytes(std::span(implementation)));
    }
    return hash.Finish();
  }
  std::expected<std::unique_ptr<kg::SelectedCall>, Failure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const kg::SelectedOperand> inputs,
      std::span<const kg::SelectedOperand> outputs) const override {
    const auto at =
        std::ranges::find(implementations_, implementation.name, &execution::Implementation::name);
    if (at == implementations_.end() ||
        execution::IdentityOf(implementation) != execution::IdentityOf(*at))
      return Reject("foreign grouped declaration");
    auto geometry = Shape(node, inputs, outputs);
    if (!geometry) return std::unexpected(geometry.error());
    if (implementation.name == kNvfp4Name && geometry->format.encoding != format::Encoding::kNvfp4)
      return Reject("grouped implementation and weight encoding differ");
    if (implementation.name == kAffineName &&
        geometry->format.encoding != format::Encoding::kAffine)
      return Reject("grouped implementation and weight encoding differ");
    return std::make_unique<Call>(this, std::move(*geometry),
                                  execution::IdentityOf(implementation));
  }
  std::expected<std::unique_ptr<kg::SelectedLaunch>, Failure> Launch(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      kg::SelectedWorkspace persistent) const override {
    const auto facts = providers::QueryDeviceFacts(device);
    const auto native = execution.Submission(stream);
    if (!facts || !CompiledFor(facts->architecture / 10) || !native || persistent.bytes)
      return Reject("grouped source device or workspace differs");
    return std::make_unique<Runtime>(this, *native);
  }

 private:
  std::vector<execution::Implementation> implementations_;
};
// ------------------------------------------------------- resident experts

std::uint64_t Align256(std::uint64_t n) { return (n + 255) / 256 * 256; }
// The pool scratch a resident call takes: compact descriptors, offsets and
// the two pair arrays, each 256-byte aligned.
struct ResidentScratch {
  std::uint64_t groups = 0, descriptors = 0, offsets = 0, rows = 0, out = 0, bytes = 0;
};
ResidentScratch ScratchOf(const Geometry& g) {
  ResidentScratch s;
  const std::uint64_t pairs = g.used * g.tokens;
  s.groups = std::min<std::uint64_t>(pairs, g.format.shape.ne[2]);
  s.descriptors = 0;
  s.offsets = Align256(s.groups * sizeof(Operand));
  s.rows = s.offsets + Align256((s.groups + 1) * sizeof(std::int32_t));
  s.out = s.rows + Align256(pairs * sizeof(std::int32_t));
  s.bytes = s.out + Align256(pairs * sizeof(std::int32_t));
  return s;
}
class ResidentRuntime final : public kg::SelectedLaunch {
 public:
  ResidentRuntime(const kg::SelectedSource* owner, providers::NativeStream stream)
      : kg::SelectedLaunch(owner, stream) {}
  bool faulted() const override { return failed; }
  bool capturable() const override { return !failed; }
  bool failed = false;
};
class ResidentBoundCall final : public kg::BoundSelectedCall {
 public:
  ResidentBoundCall(ResidentRuntime& r, Geometry g, ResidentRoute route)
      : runtime_(r), geometry_(std::move(g)), route_(route) {}
  std::expected<void, Failure> Run(providers::NativeStream stream) const override {
    const auto& g = geometry_;
    if (runtime_.failed || stream.handle != runtime_.stream().handle)
      return Reject("resident grouped runtime or stream differs");
    if (auto parts = Parts(g); !parts) return std::unexpected(parts.error());
    const auto fail = [&](std::string detail) -> std::expected<void, Failure> {
      runtime_.failed = true;
      return std::unexpected(Failure{.error = kg::KernelError::kUnknown, .detail = detail});
    };
    if (auto prepared = PrepareResidentRoute(route_, stream.handle); !prepared)
      return fail(prepared.error());
    auto one = g.format;
    one.shape.rank = 2;
    one.shape.ne[2] = 1;
    const auto result = RunGrouped({.format = one,
                                    .experts = route_.descriptors,
                                    .experts_count = route_.groups,
                                    .first = 0,
                                    .outputs = g.format.shape.ne[1],
                                    .offsets = route_.offsets,
                                    .pair_rows = route_.pair_rows,
                                    .pair_out = route_.pair_out,
                                    .pairs = g.tokens * g.used,
                                    .x = static_cast<const float*>(g.x->data),
                                    .x_rows = g.tokens * g.lanes,
                                    .x_stride = g.format.shape.ne[0],
                                    .y = static_cast<float*>(g.y->data),
                                    .y_rows = g.tokens * g.used,
                                    .y_stride = g.format.shape.ne[1],
                                    .rows_hint = g.tokens},
                                   stream.handle);
    if (!result) return fail(result.error());
    return {};
  }

 private:
  ResidentRuntime& runtime_;
  Geometry geometry_;
  ResidentRoute route_;
};
class ResidentCall final : public kg::SelectedCall {
 public:
  ResidentCall(const kg::SelectedSource* owner, Geometry geometry, base::Sha256Digest identity)
      : kg::SelectedCall(owner), geometry_(std::move(geometry)), identity_(identity) {}
  std::uint64_t scratch_bytes() const override { return ScratchOf(geometry_).bytes; }
  // The bound call, and one expert table while binding.
  std::uint64_t host_bytes() const override {
    return 8192 + (geometry_.format.shape.ne[2] * sizeof(Operand));
  }
  std::uint64_t launches() const override { return 2; }
  std::expected<std::unique_ptr<kg::BoundSelectedCall>, Failure> Bind(
      const execution::Implementation& implementation, kg::SelectedLaunch& launch,
      kg::SelectedWorkspace workspace) const override {
    const auto native = decode::Bind(implementation);
    const auto& g = geometry_;
    const auto scratch = ScratchOf(g);
    if (launch.owner() != owner() || execution::IdentityOf(implementation) != identity_ ||
        !native || native->tmpl != Template::kGrouped ||
        (native->gate != DecoderId::kNvfp4Split && native->gate != DecoderId::kAffineSplit &&
         native->gate != DecoderId::kNvfp4Atoms) ||
        // The implementation's decoder is the weight's own (its layout's).
        DecoderOf(g.format) != native->gate || workspace.base == 0 || workspace.base % 256 ||
        workspace.bytes < scratch.bytes)
      return Reject("resident grouped selection lacks its declaration or its scratch");
    const auto parts = Parts(g);
    if (!parts) return std::unexpected(parts.error());
    // Every expert's operand, checked, then expert 0's and each pointer's
    // pitch, which the device preparation moves by.
    const std::uint64_t experts = g.format.shape.ne[2];
    if (experts == 0 || experts > kMaxResidentExperts)
      return Reject("resident grouped experts outside the preparation's bound");
    std::vector<Operand> table(experts);
    if (!BindGroupedInto(g.format, std::span(*parts).first(g.count), table))
      return Reject("resident grouped part geometry is inconsistent");
    const auto at = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p); };
    const auto pitch = [&](auto member) -> std::optional<std::uint64_t> {
      const auto first = at(table[0].*member);
      const std::uint64_t step = experts > 1 ? at(table[1].*member) - first : 0;
      for (std::uint64_t e = 0; e != experts; ++e)
        if (at(table[e].*member) != (first == 0 ? 0 : first + e * step)) return std::nullopt;
      return step;
    };
    const auto data = pitch(&Operand::data);
    const auto scales = pitch(&Operand::scales);
    const auto tensor_scale = pitch(&Operand::tensor_scale);
    const auto biases = pitch(&Operand::biases);
    if (!data || !scales || !tensor_scale || !biases)
      return Reject("resident grouped experts are not at one pitch a part");
    // Every expert's geometry the first's (its pointers checked above).
    const auto same = [](const Operand& a, const Operand& b) {
      return a.k == b.k && a.rows == b.rows && a.rows_per_scale == b.rows_per_scale &&
             a.row_bytes == b.row_bytes && a.scale_row_bytes == b.scale_row_bytes &&
             a.group == b.group && a.bits == b.bits && a.scale_type == b.scale_type;
    };
    for (std::uint64_t e = 0; e != experts; ++e)
      if (!same(table[e], table[0])) return Reject("resident grouped experts differ in geometry");
    ResidentRoute route{
        .first = table[0],
        .data_pitch = *data,
        .scales_pitch = *scales,
        .tensor_scale_pitch = *tensor_scale,
        .biases_pitch = *biases,
        .experts = static_cast<std::uint32_t>(experts),
        .ids = static_cast<const std::int32_t*>(g.ids->data),
        .ids_stride = g.ids_nb[1] / sizeof(std::int32_t),
        .used = static_cast<std::uint32_t>(g.used),
        .tokens = static_cast<std::uint32_t>(g.tokens),
        .per_slot_input = g.lanes != 1,
        .groups = static_cast<std::uint32_t>(scratch.groups),
        .descriptors = reinterpret_cast<Operand*>(workspace.base),
        .offsets = reinterpret_cast<std::int32_t*>(workspace.base + scratch.offsets),
        .pair_rows = reinterpret_cast<std::int32_t*>(workspace.base + scratch.rows),
        .pair_out = reinterpret_cast<std::int32_t*>(workspace.base + scratch.out),
        .y = static_cast<float*>(g.y->data),
        .y_stride = g.format.shape.ne[1],
        .y_outputs = static_cast<std::uint32_t>(g.format.shape.ne[1])};
    if (g.lanes != 1 && g.lanes != g.used)
      return Reject("resident grouped input is neither a row a token nor a row a slot");
    return std::make_unique<ResidentBoundCall>(static_cast<ResidentRuntime&>(launch), g, route);
  }

 private:
  Geometry geometry_;
  base::Sha256Digest identity_;
};
class ResidentSource final : public kg::SelectedSource {
 public:
  explicit ResidentSource(std::vector<execution::Implementation> implementations)
      : implementations_(std::move(implementations)) {}
  std::vector<execution::Implementation> Implementations() const override {
    return implementations_;
  }
  std::vector<execution::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view name) const override {
    return std::ranges::any_of(implementations_,
                               [&](const auto& impl) { return impl.name == name; });
  }
  std::uint64_t persistent_bytes() const override { return 0; }
  std::optional<base::Sha256Digest> ExecutionIdentity() const override {
    base::Sha256 hash;
    hash.Update("llmp-native-split-resident-group-v1");
    hash.Update(
        "direct-weight-parts;device-route-worklist;compact-active-experts;"
        "pitched-stacked-experts;token-or-original-slot;affine-and-nvfp4");
    for (const auto& impl : implementations_) {
      const auto implementation = execution::IdentityOf(impl);
      hash.Update(std::as_bytes(std::span(implementation)));
    }
    return hash.Finish();
  }
  std::expected<std::unique_ptr<kg::SelectedCall>, Failure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const kg::SelectedOperand> inputs,
      std::span<const kg::SelectedOperand> outputs) const override {
    const auto at =
        std::ranges::find(implementations_, implementation.name, &execution::Implementation::name);
    if (at == implementations_.end() ||
        execution::IdentityOf(implementation) != execution::IdentityOf(*at))
      return Reject("foreign resident grouped declaration");
    auto geometry = Shape(node, inputs, outputs);
    if (!geometry) return std::unexpected(geometry.error());
    if ((implementation.name == kNvfp4Name &&
         geometry->format.encoding != format::Encoding::kNvfp4) ||
        (implementation.name == kAffineName &&
         geometry->format.encoding != format::Encoding::kAffine))
      return Reject("resident grouped implementation and weight encoding differ");
    return std::make_unique<ResidentCall>(this, std::move(*geometry),
                                          execution::IdentityOf(implementation));
  }
  std::expected<std::unique_ptr<kg::SelectedLaunch>, Failure> Launch(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      kg::SelectedWorkspace persistent) const override {
    const auto facts = providers::QueryDeviceFacts(device);
    const auto native = execution.Submission(stream);
    if (!facts || !CompiledFor(facts->architecture / 10) || !native || persistent.bytes)
      return Reject("resident grouped source device or workspace differs");
    return std::make_unique<ResidentRuntime>(this, *native);
  }

 private:
  std::vector<execution::Implementation> implementations_;
};
std::vector<execution::Implementation> GroupedImplementations(
    std::optional<format::Encoding> encoding) {
  std::vector<execution::Implementation> matched;
  for (const auto& impl : decode::Implementations()) {
    if ((impl.name == kNvfp4Name || impl.name == kNvfp4AtomsName) &&
        (!encoding || *encoding == format::Encoding::kNvfp4))
      matched.push_back(impl);
    else if (impl.name == kAffineName && (!encoding || *encoding == format::Encoding::kAffine))
      matched.push_back(impl);
  }
  return matched;
}
}  // namespace
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> GroupedSelectedSource(
    std::optional<format::Encoding> encoding) {
  auto implementations = decode::Implementations();
  std::vector<execution::Implementation> matched;
  for (const auto& impl : implementations) {
    if (impl.name == kNvfp4Name && (!encoding || *encoding == format::Encoding::kNvfp4))
      matched.push_back(impl);
    else if (impl.name == kAffineName && (!encoding || *encoding == format::Encoding::kAffine))
      matched.push_back(impl);
  }
  if (matched.empty()) return std::unexpected("native split grouped implementation missing");
  return std::make_shared<Source>(std::move(matched));
}
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string>
ResidentGroupedSelectedSource(std::optional<format::Encoding> encoding) {
  auto matched = GroupedImplementations(encoding);
  if (matched.empty()) return std::unexpected("native split grouped implementation missing");
  return std::make_shared<ResidentSource>(std::move(matched));
}
}  // namespace llmp::kernels::decode
