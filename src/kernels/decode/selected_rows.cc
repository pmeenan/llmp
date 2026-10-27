// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernels/decode/selected_rows.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "kernels/decode/implementations.h"
#include "kernels/decode/products.h"
#include "kernels/decode/selected_weight.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::decode {
namespace {
using Failure = kernels::KernelFailure;
using Role = format::PartRole;
std::unexpected<Failure> Reject(std::string message) {
  return std::unexpected(Failure{.detail = std::move(message)});
}
bool Packed(const TensorRef& ref, StorageType type, std::uint64_t width, std::uint64_t rows) {
  if (!ref) return false;
  const auto t = ref.facts();
  return t.type == type && width > 0 && rows > 0 && static_cast<std::uint64_t>(t.ne[0]) == width &&
         static_cast<std::uint64_t>(t.ne[1]) == rows && t.ne[2] == 1 && t.ne[3] == 1 &&
         t.contiguous;
}
struct Geometry {
  format::TensorFormat effective;
  std::array<TensorRef, format::PartList::kMax> parts{};
  std::array<format::PartRole, format::PartList::kMax> roles{};
  std::size_t count = 0;
  TensorRef ids{};
  TensorRef out{};
  std::uint64_t rows = 0;
  std::uint64_t batches = 1;
  std::optional<detail::WeightGeometry> resident{};
  std::array<std::int64_t, 4> ids_ne{}, out_ne{};
  std::array<std::uint64_t, 4> ids_nb{}, out_nb{};
};
std::expected<Geometry, Failure> Shape(const ir::Node& node,
                                       std::span<const SelectedOperand> inputs,
                                       std::span<const SelectedOperand> outputs) {
  if (node.key != ir::Primitive(ir::Op::kGetRows) || node.nsrc != 2 || node.nout != 1 ||
      inputs.size() != 2 || outputs.size() != 1)
    return Reject("selected rows require one two-input GetRows");
  const auto& w = inputs[0];
  const auto& ids = inputs[1];
  const auto& out = outputs[0];
  const auto& f = w.desc.format;
  if (!w.paged_weight) {
    auto weight = detail::WeightOf(w);
    if (!weight) return std::unexpected(weight.error());
    if (ids.side_rows_component || ids.desc.format.encoding != format::Encoding::kI32 ||
        ids.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(ids.desc) ||
        ids.desc.ne(1) != f.shape.ne[2] || ids.desc.ne(2) != f.shape.ne[3] || ids.desc.ne(3) != 1 ||
        ids.desc.ne(0) >= (1ULL << 31) || out.desc.format.encoding != format::Encoding::kF32 ||
        out.desc.format.layout != format::Layout::kDense || !ir::IsContiguous(out.desc) ||
        out.desc.shape().ne !=
            ir::ShapeOf({f.shape.ne[0], ids.desc.ne(0), ids.desc.ne(1), ids.desc.ne(2)}).ne ||
        !ids.parts.empty() || !out.parts.empty() || !ids.tensor || !out.tensor)
      return Reject("resident rows require matching batched I32 IDs and F32 output");
    const auto idf = ids.tensor.facts(), outf = out.tensor.facts();
    if (idf.type != StorageType::kI32 || outf.type != StorageType::kF32 || !idf.contiguous ||
        !outf.contiguous ||
        idf.elements != static_cast<std::int64_t>(ir::Elements(ids.desc.shape())) ||
        outf.elements != static_cast<std::int64_t>(ir::Elements(out.desc.shape())))
      return Reject("resident rows require matching batched I32 IDs and F32 output");
    Geometry g;
    g.effective = f;
    g.parts = weight->parts;
    g.roles = weight->roles;
    g.count = weight->count;
    g.ids = ids.tensor;
    g.out = out.tensor;
    g.rows = ids.desc.ne(0);
    g.batches = format::Matrices(f);
    g.resident = *weight;
    g.ids_ne = idf.ne;
    g.ids_nb = idf.nb;
    g.out_ne = outf.ne;
    g.out_nb = outf.nb;
    return g;
  }
  const auto parts = format::PartsOf(f);
  if (!parts || f.layout != format::Layout::kSplit || f.shape.ne[2] != 1 || f.shape.ne[3] != 1 ||
      f.stacked != 1 || f.shape.ne[0] >= (1ULL << 30) || f.shape.ne[1] >= (1ULL << 31) ||
      !w.paged_weight || w.paged_weight->expert_array || !ids.side_rows_component ||
      *ids.side_rows_component != w.paged_weight->component || w.parts.size() != parts->size())
    return Reject("selected rows require a complete split table and its checked side-row input");
  if (ids.desc.format.encoding != format::Encoding::kI32 ||
      ids.desc.format.layout != format::Layout::kDense || ids.desc.shape().rank != 1 ||
      !ir::IsContiguous(ids.desc) || !ids.parts.empty() || !out.parts.empty() ||
      out.desc.format !=
          ir::Dense(format::Encoding::kF32, ir::ShapeOf({f.shape.ne[0], ids.desc.ne(0)})) ||
      !ir::IsContiguous(out.desc) || ids.desc.ne(0) >= (1ULL << 31))
    return Reject("selected rows require complete dense I32 IDs and F32 output rows");
  Geometry g{.effective = f,
             .parts = {},
             .roles = {},
             .count = parts->size(),
             .ids = ids.tensor,
             .out = out.tensor,
             .rows = ids.desc.ne(0)};
  if (!Packed(g.ids, StorageType::kI32, g.rows, 1) ||
      !Packed(g.out, StorageType::kF32, f.shape.ne[0], g.rows))
    return Reject("selected row input or output physical geometry differs");
  std::uint64_t resident = 0;
  for (std::size_t i = 0; i < g.count; ++i) {
    const auto role = (*parts)[i].role;
    const auto at = std::ranges::find(w.parts, role, &SelectedPart::role);
    if (at == w.parts.end() || std::ranges::count(w.parts, role, &SelectedPart::role) != 1 ||
        !at->tensor)
      return Reject("selected row part missing or duplicated");
    g.parts[i] = at->tensor;
    g.roles[i] = role;
    if (i == 0) {
      const auto slots = at->tensor.facts().ne[1];
      if (slots <= 0) return Reject("selected row slots are empty");
      resident = static_cast<std::uint64_t>(slots);
      if (resident > f.shape.ne[1]) return Reject("selected row slots exceed their table");
    }
    if (role == Role::kTensorScale) {
      if (!Packed(at->tensor, StorageType::kF32, 1, 1))
        return Reject("selected row part has inconsistent physical geometry");
    } else {
      const auto row_bytes = format::RowBytes(f, role);
      if (!row_bytes || !Packed(at->tensor, StorageType::kI8, *row_bytes, resident))
        return Reject("selected row part has inconsistent physical geometry");
    }
  }
  g.effective.shape.ne[1] = resident;
  return g;
}
std::expected<std::array<DevicePart, format::PartList::kMax>, Failure> BindParts(
    const Geometry& g) {
  if (g.resident) {
    const auto idf = g.ids.facts(), outf = g.out.facts();
    if (idf.type != StorageType::kI32 || outf.type != StorageType::kF32 || idf.ne != g.ids_ne ||
        idf.nb != g.ids_nb || outf.ne != g.out_ne || outf.nb != g.out_nb)
      return Reject("selected resident row geometry changed after lowering");
  }
  if (!g.resident) {
    for (std::size_t i = 0; i < g.count; ++i) {
      if (g.roles[i] == Role::kTensorScale) {
        if (!Packed(g.parts[i], StorageType::kF32, 1, 1))
          return Reject("selected row geometry changed after lowering");
      } else {
        const auto row_bytes = format::RowBytes(g.effective, g.roles[i]);
        if (!row_bytes ||
            !Packed(g.parts[i], StorageType::kI8, *row_bytes, g.effective.shape.ne[1]))
          return Reject("selected row geometry changed after lowering");
      }
    }
    if (!Packed(g.ids, StorageType::kI32, g.rows, 1) ||
        !Packed(g.out, StorageType::kF32, g.effective.shape.ne[0], g.rows))
      return Reject("selected row geometry changed after lowering");
  }
  std::array<DevicePart, format::PartList::kMax> result{};
  std::array<TensorFacts, 6> all{};
  std::size_t all_count = 0;
  for (std::size_t i = 0; i < g.count; ++i) {
    if (!g.parts[i]) return Reject("selected rows require current physical bindings");
    all[all_count++] = g.parts[i].facts();
  }
  if (!g.ids || !g.out) return Reject("selected rows require current physical bindings");
  all[all_count++] = g.ids.facts();
  all[all_count++] = g.out.facts();
  for (std::size_t i = 0; i < all_count; ++i) {
    const auto& t = all[i];
    if (!t.current || AddressOf(t) == 0)
      return Reject("selected rows require current physical bindings");
    if (t.bytes > UINT64_MAX - AddressOf(t)) return Reject("selected rows address range overflows");
    for (std::size_t j = 0; j < i; ++j) {
      const auto& other = all[j];
      if (AddressOf(t) < AddressOf(other) + other.bytes &&
          AddressOf(other) < AddressOf(t) + t.bytes)
        return Reject("selected rows operand ranges overlap");
    }
  }
  if (AddressOf(all[g.count]) % 4 != 0 || AddressOf(all[g.count + 1]) % 4 != 0)
    return Reject("selected row IDs or output are misaligned");
  for (std::size_t i = 0; i < g.count; ++i) result[i] = {g.roles[i], all[i].data, all[i].bytes};
  if (g.resident) {
    auto checked = detail::BindWeightParts(*g.resident);
    if (!checked) return std::unexpected(checked.error());
    return checked;
  }
  auto operand = OperandOf({g.effective, std::span(result).first(g.count), 0});
  if (!operand) return Reject(operand.error());
  return result;
}
class Runtime final : public SelectedLaunch {
 public:
  Runtime(const SelectedSource* owner, providers::NativeStream stream)
      : SelectedLaunch(owner, stream) {}
  bool faulted() const override { return failed; }
  // RunRows only queues its existing kernel; IDs and part slots remain stable
  // allocations whose contents ModelRunner stages before capture or replay.
  bool capturable() const override { return true; }
  bool failed = false;
};
class BoundCall final : public BoundSelectedCall {
 public:
  BoundCall(Runtime& runtime, Geometry geometry) : runtime_(runtime), geometry_(geometry) {}
  std::expected<void, Failure> Run(providers::NativeStream stream) const override {
    if (runtime_.failed || stream.handle != runtime_.stream().handle)
      return Reject("selected rows cannot run on a failed or foreign stream");
    const auto parts = BindParts(geometry_);
    if (!parts) return std::unexpected(parts.error());
    void* const ids = geometry_.ids.facts().data;
    void* const out = geometry_.out.facts().data;
    for (std::uint64_t matrix = 0; matrix < geometry_.batches; ++matrix) {
      auto ran = RunRows(
          {.weight = {geometry_.effective, std::span(*parts).first(geometry_.count), matrix},
           .ids = static_cast<const std::int32_t*>(ids) + matrix * geometry_.rows,
           .count = geometry_.rows,
           .out =
               static_cast<float*>(out) + matrix * geometry_.rows * geometry_.effective.shape.ne[0],
           .out_stride = geometry_.effective.shape.ne[0],
           .bad = nullptr},
          stream.handle);
      if (!ran) {
        runtime_.failed = true;
        return std::unexpected(Failure{.error = KernelError::kUnknown, .detail = ran.error()});
      }
    }
    return {};
  }

 private:
  Runtime& runtime_;
  Geometry geometry_;
};
class Call final : public SelectedCall {
 public:
  Call(const SelectedSource* owner, Geometry geometry, base::Sha256Digest identity)
      : SelectedCall(owner), geometry_(geometry), identity_(identity) {}
  std::uint64_t scratch_bytes() const override { return 0; }
  std::uint64_t host_bytes() const override { return sizeof(Call) + sizeof(BoundCall) + 128; }
  std::uint64_t launches() const override { return geometry_.batches; }
  std::expected<std::unique_ptr<BoundSelectedCall>, Failure> Bind(
      const execution::Implementation& implementation, SelectedLaunch& launch,
      SelectedWorkspace) const override {
    if (launch.owner() != owner() || execution::IdentityOf(implementation) != identity_)
      return Reject("selected row source or implementation identity changed");
    const auto bound = decode::Bind(implementation);
    if (!bound || bound->tmpl != Template::kRows || DecoderOf(geometry_.effective) != bound->gate)
      return Reject("selected row declaration is not the native split decoder");
    if (auto parts = BindParts(geometry_); !parts) return std::unexpected(parts.error());
    return std::make_unique<BoundCall>(static_cast<Runtime&>(launch), geometry_);
  }

 private:
  Geometry geometry_;
  base::Sha256Digest identity_;
};
class Source final : public SelectedSource {
 public:
  explicit Source(std::vector<execution::Implementation> implementations)
      : implementations_(std::move(implementations)) {
    base::Sha256 hash;
    hash.Update("llmp-native-rows-v1;resident-batches;paged-component-slot-authentication");
    for (const auto& impl : implementations_) {
      const auto id = execution::IdentityOf(impl);
      hash.Update(std::as_bytes(std::span(id)));
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
  std::expected<std::unique_ptr<SelectedCall>, Failure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const SelectedOperand> inputs,
      std::span<const SelectedOperand> outputs) const override {
    const auto at =
        std::ranges::find(implementations_, implementation.name, &execution::Implementation::name);
    if (at == implementations_.end() ||
        execution::IdentityOf(implementation) != execution::IdentityOf(*at))
      return Reject("foreign rows declaration");
    auto geometry = Shape(node, inputs, outputs);
    if (!geometry) return std::unexpected(geometry.error());
    const auto bound = decode::Bind(implementation);
    if (!bound || DecoderOf(geometry->effective) != bound->gate)
      return Reject("rows implementation and weight format differ");
    return std::make_unique<Call>(this, *geometry, execution::IdentityOf(implementation));
  }
  std::expected<std::unique_ptr<SelectedLaunch>, Failure> Launch(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      SelectedWorkspace persistent) const override {
    const auto facts = providers::QueryDeviceFacts(device);
    if (!facts || !CompiledFor(facts->architecture / 10) || persistent.bytes != 0)
      return Reject("selected rows device or persistent workspace differs");
    const auto native = execution.Submission(stream);
    if (!native) return Reject(native.error().detail);
    return std::make_unique<Runtime>(this, *native);
  }

 private:
  std::vector<execution::Implementation> implementations_;
  base::Sha256Digest identity_{};
};
}  // namespace
std::expected<std::shared_ptr<const kernels::SelectedSource>, std::string> RowsSelectedSource(
    std::optional<format::Encoding> encoding) {
  auto implementations = decode::Implementations();
  std::vector<execution::Implementation> matched;
  for (const auto& impl : implementations) {
    const auto bound = decode::Bind(impl);
    if (bound && bound->tmpl == Template::kRows &&
        (!encoding || DecoderEncoding(bound->gate) == *encoding))
      matched.push_back(impl);
  }
  if (matched.empty()) return std::unexpected("native split rows implementation missing");
  return std::make_shared<Source>(std::move(matched));
}
}  // namespace llmp::kernels::decode
