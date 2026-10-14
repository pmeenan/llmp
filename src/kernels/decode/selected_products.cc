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
#include "kernels/ggml/validate_util.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::decode {
namespace {
namespace kg = ggml;
using Failure = kg::KernelFailure;
using Role = format::PartRole;

constexpr std::string_view kVectorAffineName = "llmp.decode.vector.affine.split";
constexpr std::string_view kVectorGluAffineName = "llmp.decode.vector_glu.affine.split";
constexpr std::string_view kMatrixAffineName = "llmp.decode.matrix.affine.split";
constexpr std::string_view kMatrixFmaAffineName = "llmp.decode.matrix_fma.affine.split";

std::unexpected<Failure> Reject(std::string message) {
  return std::unexpected(Failure{.detail = std::move(message)});
}

std::uint64_t Address(const ggml_tensor* t) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data));
}

bool Packed(const ggml_tensor* t, ggml_type type, std::uint64_t width, std::uint64_t rows) {
  return t != nullptr && t->type == type && width > 0 && rows > 0 &&
         static_cast<std::uint64_t>(t->ne[0]) == width &&
         static_cast<std::uint64_t>(t->ne[1]) == rows && t->ne[2] == 1 && t->ne[3] == 1 &&
         ggml_is_contiguous(t);
}

struct WeightGeometry {
  format::TensorFormat format{};
  std::array<ggml_tensor*, 3> parts{};
  std::array<format::PartRole, 3> roles{};
  std::size_t count = 0;
};

std::expected<void, Failure> CheckWeight(const kg::SelectedOperand& w, WeightGeometry& out) {
  const auto& f = w.desc.format;
  const auto parts = format::PartsOf(f);
  if (!parts || f.encoding != format::Encoding::kAffine || f.layout != format::Layout::kSplit ||
      f.shape.rank != 2 || f.stacked != 1 || f.shape.ne[0] >= (1ULL << 30) ||
      f.shape.ne[1] >= (1ULL << 31) || w.parts.size() != parts->size()) {
    return Reject("selected product requires a complete affine split weight");
  }
  const std::uint64_t k = f.shape.ne[0];
  const std::uint64_t n = f.shape.ne[1];
  if (k % DecoderKMultiple(DecoderId::kAffineSplit) != 0) {
    return Reject("selected product weight K must be a multiple of decoder segment");
  }
  out.format = f;
  out.count = parts->size();
  for (std::size_t i = 0; i < out.count; ++i) {
    const auto role = (*parts)[i].role;
    const auto at = std::ranges::find(w.parts, role, &kg::SelectedPart::role);
    if (at == w.parts.end() || std::ranges::count(w.parts, role, &kg::SelectedPart::role) != 1 ||
        at->tensor == nullptr) {
      return Reject("selected product weight part missing or duplicated");
    }
    out.parts[i] = at->tensor;
    out.roles[i] = role;
    if (role == Role::kTensorScale) {
      if (!Packed(at->tensor, GGML_TYPE_F32, 1, 1))
        return Reject("selected product tensor scale physical geometry differs");
    } else {
      const auto row_bytes = format::RowBytes(f, role);
      if (!row_bytes || !Packed(at->tensor, GGML_TYPE_I8, *row_bytes, n))
        return Reject("selected product weight part physical geometry differs");
    }
  }
  return {};
}

std::expected<std::array<DevicePart, 3>, Failure> BindWeightParts(const WeightGeometry& g) {
  for (std::size_t i = 0; i < g.count; ++i) {
    if (g.roles[i] == Role::kTensorScale) {
      if (!Packed(g.parts[i], GGML_TYPE_F32, 1, 1))
        return Reject("selected product part geometry changed after lowering");
    } else {
      const auto row_bytes = format::RowBytes(g.format, g.roles[i]);
      if (!row_bytes || !Packed(g.parts[i], GGML_TYPE_I8, *row_bytes, g.format.shape.ne[1]))
        return Reject("selected product part geometry changed after lowering");
    }
    if (!kg::detail::Current(g.parts[i]) || Address(g.parts[i]) == 0)
      return Reject("selected product part requires current physical binding");
    const auto bytes = ggml_nbytes(g.parts[i]);
    if (bytes > UINT64_MAX - Address(g.parts[i]))
      return Reject("selected product part address overflows");
  }
  std::array<DevicePart, 3> result{};
  for (std::size_t i = 0; i < g.count; ++i) {
    result[i] = {g.roles[i], g.parts[i]->data, ggml_nbytes(g.parts[i])};
  }
  auto operand = OperandOf({g.format, std::span(result).first(g.count), 0});
  if (!operand) return Reject(operand.error());
  return result;
}

struct SingleGeometry {
  WeightGeometry weight;
  ggml_tensor* x = nullptr;
  ggml_tensor* y = nullptr;
  std::uint64_t k = 0;
  std::uint64_t outputs = 0;
  std::uint64_t rows = 0;
  Template tmpl = Template::kVector;
};

struct GluGeometry {
  WeightGeometry gate;
  WeightGeometry up;
  ir::GluParams glu{};
  ggml_tensor* x = nullptr;
  ggml_tensor* y = nullptr;
  std::uint64_t k = 0;
  std::uint64_t outputs = 0;
  std::uint64_t rows = 0;
};

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
      x.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(x.desc) ||
      !x.parts.empty() || x.desc.ne(0) != g.k)
    return Reject("selected product input x must be contiguous F32 matching K");

  const std::uint64_t total_x_elems = ir::Elements(x.desc.shape());
  if (total_x_elems % g.k != 0)
    return Reject("selected product input x elements must be multiple of K");
  g.rows = total_x_elems / g.k;

  if (tmpl == Template::kVector) {
    if (g.rows < 1 || g.rows > kMaxVectorRows)
      return Reject("vector product requires 1 to 16 rows");
  } else {
    if (g.rows < 1 || g.rows > kMaxMatrixRows) return Reject("matrix product rows out of range");
  }

  if (y.desc.format.encoding != format::Encoding::kF32 ||
      y.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(y.desc) ||
      !y.parts.empty() || y.desc.ne(0) != g.outputs ||
      ir::Elements(y.desc.shape()) != g.outputs * g.rows)
    return Reject("selected product output y must be contiguous F32 matching N and rows");

  g.x = x.tensor;
  g.y = y.tensor;
  if (!g.x || !g.y || g.x->type != GGML_TYPE_F32 || g.y->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(g.x) || !ggml_is_contiguous(g.y) ||
      static_cast<std::uint64_t>(ggml_nelements(g.x)) != total_x_elems ||
      static_cast<std::uint64_t>(ggml_nelements(g.y)) != g.outputs * g.rows)
    return Reject("selected product activation physical geometry differs");

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
      g.gate.format.shape.ne[1] != g.up.format.shape.ne[1])
    return Reject("selected vector_glu gate and up weight shapes must match");

  g.k = g.gate.format.shape.ne[0];
  g.outputs = g.gate.format.shape.ne[1];

  if (x.desc.format.encoding != format::Encoding::kF32 ||
      x.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(x.desc) ||
      !x.parts.empty() || x.desc.ne(0) != g.k)
    return Reject("selected vector_glu input x must be contiguous F32 matching K");

  const std::uint64_t total_x_elems = ir::Elements(x.desc.shape());
  if (total_x_elems % g.k != 0)
    return Reject("selected vector_glu input x elements must be multiple of K");
  g.rows = total_x_elems / g.k;

  if (g.rows < 1 || g.rows > kMaxVectorRows)
    return Reject("selected vector_glu requires 1 to 16 rows");

  if (y.desc.format.encoding != format::Encoding::kF32 ||
      y.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(y.desc) ||
      !y.parts.empty() || y.desc.ne(0) != g.outputs ||
      ir::Elements(y.desc.shape()) != g.outputs * g.rows)
    return Reject("selected vector_glu output y must be contiguous F32 matching N and rows");

  g.x = x.tensor;
  g.y = y.tensor;
  if (!g.x || !g.y || g.x->type != GGML_TYPE_F32 || g.y->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(g.x) || !ggml_is_contiguous(g.y) ||
      static_cast<std::uint64_t>(ggml_nelements(g.x)) != total_x_elems ||
      static_cast<std::uint64_t>(ggml_nelements(g.y)) != g.outputs * g.rows)
    return Reject("selected vector_glu activation physical geometry differs");

  return g;
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

    if (std::holds_alternative<SingleGeometry>(geometry_)) {
      const auto& g = std::get<SingleGeometry>(geometry_);
      const auto parts = BindWeightParts(g.weight);
      if (!parts) return std::unexpected(parts.error());
      Product p{
          .weight = {g.weight.format, std::span(*parts).first(g.weight.count), 0},
          .first = 0,
          .outputs = g.outputs,
          .x = static_cast<const float*>(g.x->data),
          .rows = g.rows,
          .x_stride = g.k,
          .y = static_cast<float*>(g.y->data),
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
    } else {
      const auto& g = std::get<GluGeometry>(geometry_);
      const auto gate_parts = BindWeightParts(g.gate);
      if (!gate_parts) return std::unexpected(gate_parts.error());
      const auto up_parts = BindWeightParts(g.up);
      if (!up_parts) return std::unexpected(up_parts.error());
      GluProduct p{
          .gate = {g.gate.format, std::span(*gate_parts).first(g.gate.count), 0},
          .gate_first = 0,
          .up = {g.up.format, std::span(*up_parts).first(g.up.count), 0},
          .up_first = 0,
          .outputs = g.outputs,
          .glu = g.glu,
          .x = static_cast<const float*>(g.x->data),
          .rows = g.rows,
          .x_stride = g.k,
          .y = static_cast<float*>(g.y->data),
          .y_stride = g.outputs,
      };
      auto status = RunVectorGlu(p, stream.handle);
      if (!status) {
        runtime_.failed = true;
        return std::unexpected(
            Failure{.error = kg::KernelError::kUnknown, .detail = status.error()});
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
  std::uint64_t host_bytes() const override { return 2048; }
  std::uint64_t launches() const override { return 1; }
  std::expected<std::unique_ptr<kg::BoundSelectedCall>, Failure> Bind(
      const execution::Implementation& implementation, kg::SelectedLaunch& launch,
      kg::SelectedWorkspace) const override {
    if (launch.owner() != owner() || execution::IdentityOf(implementation) != identity_)
      return Reject("selected product source or implementation identity changed");
    const auto bound = decode::Bind(implementation);
    if (!bound || bound->gate != DecoderId::kAffineSplit)
      return Reject("selected product declaration is not affine split decoder");

    if (std::holds_alternative<SingleGeometry>(geometry_)) {
      const auto& g = std::get<SingleGeometry>(geometry_);
      if (bound->tmpl != g.tmpl) return Reject("selected product implementation template mismatch");
      auto parts = BindWeightParts(g.weight);
      if (!parts) return std::unexpected(parts.error());
      std::array<const ggml_tensor*, 5> all{};
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
      if (bound->tmpl != Template::kVectorGlu)
        return Reject("selected vector_glu implementation template mismatch");
      auto gate_parts = BindWeightParts(g.gate);
      if (!gate_parts) return std::unexpected(gate_parts.error());
      auto up_parts = BindWeightParts(g.up);
      if (!up_parts) return std::unexpected(up_parts.error());
      std::array<const ggml_tensor*, 8> all{};
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
    hash.Update("llmp-native-split-products-v1");
    hash.Update("checked-weights;affine-split;vector-vector_glu-matrix-matrix_fma");
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
      return Reject("foreign product declaration");

    if (implementation.name == kVectorAffineName) {
      auto geom = ShapeSingle(Template::kVector, node, inputs, outputs);
      if (!geom) return std::unexpected(geom.error());
      return std::make_unique<Call>(this, *geom, execution::IdentityOf(implementation));
    } else if (implementation.name == kMatrixAffineName) {
      auto geom = ShapeSingle(Template::kMatrix, node, inputs, outputs);
      if (!geom) return std::unexpected(geom.error());
      return std::make_unique<Call>(this, *geom, execution::IdentityOf(implementation));
    } else if (implementation.name == kMatrixFmaAffineName) {
      auto geom = ShapeSingle(Template::kMatrixFma, node, inputs, outputs);
      if (!geom) return std::unexpected(geom.error());
      return std::make_unique<Call>(this, *geom, execution::IdentityOf(implementation));
    } else if (implementation.name == kVectorGluAffineName) {
      auto geom = ShapeGlu(node, inputs, outputs);
      if (!geom) return std::unexpected(geom.error());
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
};

}  // namespace

std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> ProductsSelectedSource(
    std::optional<format::Encoding> encoding) {
  if (encoding && *encoding != format::Encoding::kAffine) {
    return std::unexpected("products selected source only supports affine encoding");
  }
  auto implementations = decode::Implementations();
  std::vector<execution::Implementation> matched;
  for (const auto& impl : implementations) {
    if (impl.name == kVectorAffineName || impl.name == kVectorGluAffineName ||
        impl.name == kMatrixAffineName || impl.name == kMatrixFmaAffineName) {
      matched.push_back(impl);
    }
  }
  if (matched.empty())
    return std::unexpected("native split affine products implementations missing");
  return std::make_shared<Source>(std::move(matched));
}

}  // namespace llmp::kernels::decode
