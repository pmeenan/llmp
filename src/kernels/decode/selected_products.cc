// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/decode/selected_products.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "base/sha256.h"
#include "kernels/decode/implementations.h"
#include "kernels/decode/products.h"
#include "kernels/decode/selected_weight.h"
#include "kernels/ggml/validate_util.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::decode {
namespace {
namespace kg = ggml;
using Failure = kg::KernelFailure;
using Role = format::PartRole;

std::unexpected<Failure> Reject(std::string message) {
  return std::unexpected(Failure{.detail = std::move(message)});
}

std::uint64_t Address(const ggml_tensor* t) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data));
}

using detail::BindWeightParts;
using detail::WeightGeometry;
std::expected<void, Failure> CheckWeight(const kg::SelectedOperand& w, WeightGeometry& out) {
  auto geometry = detail::WeightOf(w);
  if (!geometry) return std::unexpected(geometry.error());
  out = *geometry;
  return {};
}

struct ActivationGeometry {
  ggml_tensor* x = nullptr;
  ggml_tensor* y = nullptr;
  std::uint64_t k = 0, outputs = 0, rows = 0, batches2 = 1, batches3 = 1;
  std::array<std::int64_t, 4> x_ne{}, y_ne{};
  std::array<std::size_t, 4> x_nb{}, y_nb{};
};
struct SingleGeometry : ActivationGeometry {
  WeightGeometry weight;
  Template tmpl = Template::kVector;
};
struct GluGeometry : ActivationGeometry {
  WeightGeometry gate;
  WeightGeometry up;
  ir::GluParams glu{};
};
bool InputRows(const ir::TensorDesc& desc) {
  return desc.nb[0] == 4 && desc.nb[1] % 16 == 0 && desc.nb[1] >= desc.ne(0) * 4 &&
         desc.nb[2] >= desc.nb[1] * desc.ne(1) && desc.nb[3] >= desc.nb[2] * desc.ne(2) &&
         (desc.ne(2) == 1 || desc.nb[2] % 16 == 0) && (desc.ne(3) == 1 || desc.nb[3] % 16 == 0);
}
void Remember(ActivationGeometry& g) {
  std::ranges::copy(g.x->ne, g.x_ne.begin());
  std::ranges::copy(g.y->ne, g.y_ne.begin());
  std::ranges::copy(g.x->nb, g.x_nb.begin());
  std::ranges::copy(g.y->nb, g.y_nb.begin());
}
bool CurrentActivations(const ActivationGeometry& g) {
  return g.x && g.y && kg::detail::Current(g.x) && kg::detail::Current(g.y) &&
         kg::detail::Extent(g.x) && kg::detail::Extent(g.y) && g.x->type == GGML_TYPE_F32 &&
         g.y->type == GGML_TYPE_F32 && std::ranges::equal(g.x->ne, g.x_ne) &&
         std::ranges::equal(g.y->ne, g.y_ne) && std::ranges::equal(g.x->nb, g.x_nb) &&
         std::ranges::equal(g.y->nb, g.y_nb) && Address(g.x) % 16 == 0 && Address(g.y) % 4 == 0;
}

using ProductGeometry = std::variant<SingleGeometry, GluGeometry>;

std::expected<SingleGeometry, Failure> ShapeSingle(Template tmpl, const ir::Node& node,
                                                   std::span<const kg::SelectedOperand> inputs,
                                                   std::span<const kg::SelectedOperand> outputs) {
  if (node.key != ir::Primitive(ir::Op::kMatMul) || node.nsrc != 2 || node.nout != 1 ||
      inputs.size() != 2 || outputs.size() != 1)
    return Reject("selected product requires one two-input MatMul");
  const auto& w = inputs[0];
  const auto& x = inputs[1];
  const auto& y = outputs[0];

  SingleGeometry g;
  g.tmpl = tmpl;
  auto w_ok = CheckWeight(w, g.weight);
  if (!w_ok) return std::unexpected(w_ok.error());

  g.k = g.weight.format.shape.ne[0];
  g.outputs = g.weight.format.shape.ne[1];

  if (x.desc.format.encoding != format::Encoding::kF32 ||
      x.desc.format.layout != format::Layout::kDense || !InputRows(x.desc) || !x.parts.empty() ||
      x.desc.ne(0) != g.k)
    return Reject("selected product input x must be contiguous F32 matching K");

  const std::uint64_t total_x_elems = ir::Elements(x.desc.shape());
  if (total_x_elems % g.k != 0)
    return Reject("selected product input x elements must be multiple of K");
  g.rows = x.desc.ne(1);
  g.batches2 = x.desc.ne(2);
  g.batches3 = x.desc.ne(3);

  if (tmpl == Template::kVector) {
    if (g.rows < 1 || g.rows > kMaxVectorRows)
      return Reject("vector product requires 1 to 16 rows");
  } else {
    if (g.rows < 1 || g.rows > kMaxMatrixRows) return Reject("matrix product rows out of range");
  }

  if (y.desc.format.encoding != format::Encoding::kF32 ||
      y.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(y.desc) ||
      !y.parts.empty() ||
      y.desc.shape().ne != ir::ShapeOf({g.outputs, g.rows, g.batches2, g.batches3}).ne ||
      ir::Elements(y.desc.shape()) != g.outputs * (total_x_elems / g.k))
    return Reject("selected product output y must be contiguous F32 matching N and rows");

  g.x = x.tensor;
  g.y = y.tensor;
  if (!g.x || !g.y || g.x->type != GGML_TYPE_F32 || g.y->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(g.y) ||
      static_cast<std::uint64_t>(ggml_nelements(g.x)) != total_x_elems ||
      static_cast<std::uint64_t>(ggml_nelements(g.y)) != g.outputs * (total_x_elems / g.k))
    return Reject("selected product activation physical geometry differs");

  Remember(g);
  if (!std::ranges::equal(g.x->nb, x.desc.nb) || !std::ranges::equal(g.y->nb, y.desc.nb))
    return Reject("selected product activation strides differ");
  if (x.desc.ne(2) % g.weight.format.shape.ne[2] || x.desc.ne(3) % g.weight.format.shape.ne[3])
    return Reject("selected product weight batches do not broadcast to input");
  return g;
}

std::expected<GluGeometry, Failure> ShapeGlu(const ir::Node& node,
                                             std::span<const kg::SelectedOperand> inputs,
                                             std::span<const kg::SelectedOperand> outputs) {
  if (node.key != ir::Fused(ir::FusedKind::kMatMulGlu) || node.nsrc != 3 || node.nout != 1 ||
      inputs.size() != 3 || outputs.size() != 1)
    return Reject("selected vector_glu requires one three-input MatMulGlu");

  ir::GluParams glu = node.Params<ir::GluParams>();
  const auto& gate_in = glu.swapped ? inputs[1] : inputs[0];
  const auto& up_in = glu.swapped ? inputs[0] : inputs[1];
  glu.swapped = false;

  const auto& x = inputs[2];
  const auto& y = outputs[0];

  GluGeometry g;
  g.glu = glu;

  auto gate_ok = CheckWeight(gate_in, g.gate);
  if (!gate_ok) return std::unexpected(gate_ok.error());
  auto up_ok = CheckWeight(up_in, g.up);
  if (!up_ok) return std::unexpected(up_ok.error());

  if (g.gate.format.shape.ne[0] != g.up.format.shape.ne[0] ||
      g.gate.format.shape.ne != g.up.format.shape.ne)
    return Reject("selected vector_glu gate and up weight shapes must match");

  g.k = g.gate.format.shape.ne[0];
  g.outputs = g.gate.format.shape.ne[1];

  if (x.desc.format.encoding != format::Encoding::kF32 ||
      x.desc.format.layout != format::Layout::kDense || !InputRows(x.desc) || !x.parts.empty() ||
      x.desc.ne(0) != g.k)
    return Reject("selected vector_glu input x must be contiguous F32 matching K");

  const std::uint64_t total_x_elems = ir::Elements(x.desc.shape());
  if (total_x_elems % g.k != 0)
    return Reject("selected vector_glu input x elements must be multiple of K");
  g.rows = x.desc.ne(1);
  g.batches2 = x.desc.ne(2);
  g.batches3 = x.desc.ne(3);

  if (g.rows < 1 || g.rows > kMaxVectorRows)
    return Reject("selected vector_glu requires 1 to 16 rows");

  if (y.desc.format.encoding != format::Encoding::kF32 ||
      y.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(y.desc) ||
      !y.parts.empty() ||
      y.desc.shape().ne != ir::ShapeOf({g.outputs, g.rows, g.batches2, g.batches3}).ne ||
      ir::Elements(y.desc.shape()) != g.outputs * (total_x_elems / g.k))
    return Reject("selected vector_glu output y must be contiguous F32 matching N and rows");

  g.x = x.tensor;
  g.y = y.tensor;
  if (!g.x || !g.y || g.x->type != GGML_TYPE_F32 || g.y->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(g.y) ||
      static_cast<std::uint64_t>(ggml_nelements(g.x)) != total_x_elems ||
      static_cast<std::uint64_t>(ggml_nelements(g.y)) != g.outputs * (total_x_elems / g.k))
    return Reject("selected vector_glu activation physical geometry differs");

  Remember(g);
  if (!std::ranges::equal(g.x->nb, x.desc.nb) || !std::ranges::equal(g.y->nb, y.desc.nb))
    return Reject("selected GLU activation strides differ");
  if (x.desc.ne(2) % g.gate.format.shape.ne[2] || x.desc.ne(3) % g.gate.format.shape.ne[3])
    return Reject("selected GLU weight batches do not broadcast to input");
  return g;
}

std::uint64_t MatrixAt(const WeightGeometry& w, std::uint64_t b2, std::uint64_t b3,
                       std::uint64_t x2, std::uint64_t x3) {
  const auto w2 = w.format.shape.ne[2], w3 = w.format.shape.ne[3];
  return b2 / (x2 / w2) + (b3 / (x3 / w3)) * w2;
}

class Runtime final : public kg::SelectedLaunch {
 public:
  Runtime(const kg::SelectedSource* owner, providers::NativeStream stream)
      : kg::SelectedLaunch(owner, stream) {}
  bool faulted() const override { return failed; }
  bool capturable() const override { return true; }
  bool failed = false;
};

class BoundCall final : public kg::BoundSelectedCall {
 public:
  BoundCall(Runtime& runtime, ProductGeometry geometry)
      : runtime_(runtime), geometry_(std::move(geometry)) {}

  std::expected<void, Failure> Run(providers::NativeStream stream) const override {
    if (runtime_.failed || stream.handle != runtime_.stream().handle)
      return Reject("selected product cannot run on a failed or foreign stream");

    if (!std::visit([](const auto& g) { return CurrentActivations(g); }, geometry_))
      return Reject("selected product activation geometry changed");
    if (std::holds_alternative<SingleGeometry>(geometry_)) {
      const auto& g = std::get<SingleGeometry>(geometry_);
      const auto parts = BindWeightParts(g.weight);
      if (!parts) return std::unexpected(parts.error());
      for (std::uint64_t b3 = 0; b3 < g.batches3; ++b3) {
        for (std::uint64_t b2 = 0; b2 < g.batches2; ++b2) {
          const auto offset = (b3 * g.batches2 + b2) * g.rows;
          Product p{
              .weight = {g.weight.format, std::span(*parts).first(g.weight.count),
                         MatrixAt(g.weight, b2, b3, g.batches2, g.batches3)},
              .first = 0,
              .outputs = g.outputs,
              .x = reinterpret_cast<const float*>(static_cast<const std::byte*>(g.x->data) +
                                                  b2 * g.x_nb[2] + b3 * g.x_nb[3]),
              .rows = g.rows,
              .x_stride = g.x_nb[1] / sizeof(float),
              .y = static_cast<float*>(g.y->data) + offset * g.outputs,
              .y_stride = g.outputs,
          };
          Status status;
          if (g.tmpl == Template::kVector) {
            status = RunVector(p, stream.handle);
          } else if (g.tmpl == Template::kMatrix) {
            status = RunMatrix(p, MatrixPath::kMma, stream.handle);
          } else if (g.tmpl == Template::kMatrixFma) {
            status = RunMatrix(p, MatrixPath::kFma, stream.handle);
          }
          if (!status) {
            runtime_.failed = true;
            return std::unexpected(
                Failure{.error = kg::KernelError::kUnknown, .detail = status.error()});
          }
        }
      }
    } else {
      const auto& g = std::get<GluGeometry>(geometry_);
      const auto gate_parts = BindWeightParts(g.gate);
      if (!gate_parts) return std::unexpected(gate_parts.error());
      const auto up_parts = BindWeightParts(g.up);
      if (!up_parts) return std::unexpected(up_parts.error());
      for (std::uint64_t b3 = 0; b3 < g.batches3; ++b3) {
        for (std::uint64_t b2 = 0; b2 < g.batches2; ++b2) {
          const auto offset = (b3 * g.batches2 + b2) * g.rows;
          GluProduct p{
              .gate = {g.gate.format, std::span(*gate_parts).first(g.gate.count),
                       MatrixAt(g.gate, b2, b3, g.batches2, g.batches3)},
              .gate_first = 0,
              .up = {g.up.format, std::span(*up_parts).first(g.up.count),
                     MatrixAt(g.up, b2, b3, g.batches2, g.batches3)},
              .up_first = 0,
              .outputs = g.outputs,
              .glu = g.glu,
              .x = reinterpret_cast<const float*>(static_cast<const std::byte*>(g.x->data) +
                                                  b2 * g.x_nb[2] + b3 * g.x_nb[3]),
              .rows = g.rows,
              .x_stride = g.x_nb[1] / sizeof(float),
              .y = static_cast<float*>(g.y->data) + offset * g.outputs,
              .y_stride = g.outputs,
          };
          auto status = RunVectorGlu(p, stream.handle);
          if (!status) {
            runtime_.failed = true;
            return std::unexpected(
                Failure{.error = kg::KernelError::kUnknown, .detail = status.error()});
          }
        }
      }
    }
    return {};
  }

 private:
  Runtime& runtime_;
  ProductGeometry geometry_;
};

class Call final : public kg::SelectedCall {
 public:
  Call(const kg::SelectedSource* owner, ProductGeometry geometry, base::Sha256Digest identity)
      : kg::SelectedCall(owner), geometry_(std::move(geometry)), identity_(identity) {}
  std::uint64_t scratch_bytes() const override { return 0; }
  std::uint64_t host_bytes() const override { return sizeof(Call) + sizeof(BoundCall) + 128; }
  std::uint64_t launches() const override {
    return std::visit([](const auto& g) { return g.batches2 * g.batches3; }, geometry_);
  }
  std::expected<std::unique_ptr<kg::BoundSelectedCall>, Failure> Bind(
      const execution::Implementation& implementation, kg::SelectedLaunch& launch,
      kg::SelectedWorkspace) const override {
    if (launch.owner() != owner() || execution::IdentityOf(implementation) != identity_)
      return Reject("selected product source or implementation identity changed");
    const auto bound = decode::Bind(implementation);
    if (!bound) return Reject("selected product declaration has no decoder");

    if (!std::visit([](const auto& g) { return CurrentActivations(g); }, geometry_))
      return Reject("selected product activation geometry changed");
    if (std::holds_alternative<SingleGeometry>(geometry_)) {
      const auto& g = std::get<SingleGeometry>(geometry_);
      if (bound->tmpl != g.tmpl || DecoderOf(g.weight.format) != bound->gate)
        return Reject("selected product implementation template mismatch");
      auto parts = BindWeightParts(g.weight);
      if (!parts) return std::unexpected(parts.error());
      std::array<const ggml_tensor*, 6> all{};
      std::size_t all_count = 0;
      for (std::size_t i = 0; i < g.weight.count; ++i) all[all_count++] = g.weight.parts[i];
      all[all_count++] = g.x;
      all[all_count++] = g.y;
      for (std::size_t i = 0; i < all_count; ++i) {
        const auto* t = all[i];
        if (!kg::detail::Current(t) || Address(t) == 0)
          return Reject("selected product requires current physical bindings");
        const auto bytes = ggml_nbytes(t);
        if (bytes > UINT64_MAX - Address(t))
          return Reject("selected product address range overflows");
        for (std::size_t j = 0; j < i; ++j) {
          const auto* other = all[j];
          if (t == other) continue;
          if (Address(t) < Address(other) + ggml_nbytes(other) &&
              Address(other) < Address(t) + bytes)
            return Reject("selected product operand ranges overlap");
        }
      }
      if (Address(g.x) % 16 != 0 || Address(g.y) % 4 != 0)
        return Reject("selected product input or output is misaligned");
    } else {
      const auto& g = std::get<GluGeometry>(geometry_);
      if (bound->tmpl != Template::kVectorGlu || DecoderOf(g.gate.format) != bound->gate ||
          DecoderOf(g.up.format) != bound->up)
        return Reject("selected vector_glu implementation template mismatch");
      auto gate_parts = BindWeightParts(g.gate);
      if (!gate_parts) return std::unexpected(gate_parts.error());
      auto up_parts = BindWeightParts(g.up);
      if (!up_parts) return std::unexpected(up_parts.error());
      std::array<const ggml_tensor*, 10> all{};
      std::size_t all_count = 0;
      for (std::size_t i = 0; i < g.gate.count; ++i) all[all_count++] = g.gate.parts[i];
      for (std::size_t i = 0; i < g.up.count; ++i) all[all_count++] = g.up.parts[i];
      all[all_count++] = g.x;
      all[all_count++] = g.y;
      for (std::size_t i = 0; i < all_count; ++i) {
        const auto* t = all[i];
        if (!kg::detail::Current(t) || Address(t) == 0)
          return Reject("selected product requires current physical bindings");
        const auto bytes = ggml_nbytes(t);
        if (bytes > UINT64_MAX - Address(t))
          return Reject("selected product address range overflows");
        for (std::size_t j = 0; j < i; ++j) {
          const auto* other = all[j];
          if (t == other) continue;
          if (Address(t) < Address(other) + ggml_nbytes(other) &&
              Address(other) < Address(t) + bytes)
            return Reject("selected product operand ranges overlap");
        }
      }
      if (Address(g.x) % 16 != 0 || Address(g.y) % 4 != 0)
        return Reject("selected product input or output is misaligned");
    }
    return std::make_unique<BoundCall>(static_cast<Runtime&>(launch), geometry_);
  }

 private:
  ProductGeometry geometry_;
  base::Sha256Digest identity_;
};

class Source final : public kg::SelectedSource {
 public:
  explicit Source(std::vector<execution::Implementation> implementations)
      : implementations_(std::move(implementations)) {
    base::Sha256 hash;
    hash.Update("llmp-native-split-products-v1");
    hash.Update(
        "checked-weights;all-decoders;batched-broadcast;vector-vector_glu-matrix-matrix_fma");
    for (const auto& impl : implementations_) {
      const auto implementation = execution::IdentityOf(impl);
      hash.Update(std::as_bytes(std::span(implementation)));
    }
    identity_ = hash.Finish();
  }
  std::vector<execution::Implementation> Implementations() const override {
    return implementations_;
  }
  std::vector<execution::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view name) const override {
    return std::ranges::any_of(implementations_,
                               [&](const auto& impl) { return impl.name == name; });
  }
  std::uint64_t persistent_bytes() const override { return 0; }
  std::optional<base::Sha256Digest> ExecutionIdentity() const override { return identity_; }
  std::expected<std::unique_ptr<kg::SelectedCall>, Failure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const kg::SelectedOperand> inputs,
      std::span<const kg::SelectedOperand> outputs) const override {
    const auto at =
        std::ranges::find(implementations_, implementation.name, &execution::Implementation::name);
    if (at == implementations_.end() ||
        execution::IdentityOf(implementation) != execution::IdentityOf(*at))
      return Reject("foreign product declaration");

    const auto bound = decode::Bind(implementation);
    if (!bound) return Reject("product declaration has no decoder");
    if (bound->tmpl == Template::kVector) {
      auto geom = ShapeSingle(Template::kVector, node, inputs, outputs);
      if (!geom) return std::unexpected(geom.error());
      if (DecoderOf(geom->weight.format) != bound->gate)
        return Reject("product decoder and format differ");
      return std::make_unique<Call>(this, *geom, execution::IdentityOf(implementation));
    } else if (bound->tmpl == Template::kMatrix) {
      auto geom = ShapeSingle(Template::kMatrix, node, inputs, outputs);
      if (!geom) return std::unexpected(geom.error());
      if (DecoderOf(geom->weight.format) != bound->gate)
        return Reject("product decoder and format differ");
      return std::make_unique<Call>(this, *geom, execution::IdentityOf(implementation));
    } else if (bound->tmpl == Template::kMatrixFma) {
      auto geom = ShapeSingle(Template::kMatrixFma, node, inputs, outputs);
      if (!geom) return std::unexpected(geom.error());
      if (DecoderOf(geom->weight.format) != bound->gate)
        return Reject("product decoder and format differ");
      return std::make_unique<Call>(this, *geom, execution::IdentityOf(implementation));
    } else if (bound->tmpl == Template::kVectorGlu) {
      auto geom = ShapeGlu(node, inputs, outputs);
      if (!geom) return std::unexpected(geom.error());
      if (DecoderOf(geom->gate.format) != bound->gate || DecoderOf(geom->up.format) != bound->up)
        return Reject("GLU decoder pair and physical operand formats differ");
      return std::make_unique<Call>(this, *geom, execution::IdentityOf(implementation));
    }
    return Reject("unsupported product implementation name");
  }
  std::expected<std::unique_ptr<kg::SelectedLaunch>, Failure> Launch(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      kg::SelectedWorkspace persistent) const override {
    const auto facts = providers::QueryDeviceFacts(device);
    if (!facts || !CompiledFor(facts->architecture / 10) || persistent.bytes != 0)
      return Reject("selected products device or persistent workspace differs");
    const auto native = execution.Submission(stream);
    if (!native) return Reject(native.error().detail);
    return std::make_unique<Runtime>(this, *native);
  }

 private:
  std::vector<execution::Implementation> implementations_;
  base::Sha256Digest identity_{};
};

}  // namespace

std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> ProductsSelectedSource(
    std::optional<format::Encoding> encoding) {
  auto implementations = decode::Implementations();
  std::vector<execution::Implementation> matched;
  for (const auto& impl : implementations) {
    const auto bound = decode::Bind(impl);
    if (bound &&
        (bound->tmpl == Template::kVector || bound->tmpl == Template::kVectorGlu ||
         bound->tmpl == Template::kMatrix || bound->tmpl == Template::kMatrixFma) &&
        (!encoding || DecoderEncoding(bound->gate) == *encoding)) {
      matched.push_back(impl);
    }
  }
  if (matched.empty())
    return std::unexpected("native split affine products implementations missing");
  return std::make_shared<Source>(std::move(matched));
}

}  // namespace llmp::kernels::decode
