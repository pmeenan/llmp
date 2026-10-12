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

#include "kernels/decode/implementations.h"
#include "kernels/decode/products.h"
#include "kernels/ggml/validate_util.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::decode {
namespace {
namespace kg = ggml;
using Failure = kg::KernelFailure;
using Role = format::PartRole;
constexpr std::string_view kName = "llmp.decode.rows.nvfp4.split";
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
struct Geometry {
  format::TensorFormat effective;
  std::array<ggml_tensor*, 3> parts{};
  std::size_t count = 0;
  ggml_tensor* ids = nullptr;
  ggml_tensor* out = nullptr;
  std::uint64_t rows = 0;
};
std::expected<Geometry, Failure> Shape(const ir::Node& node,
                                       std::span<const kg::SelectedOperand> inputs,
                                       std::span<const kg::SelectedOperand> outputs) {
  if (node.key != ir::Primitive(ir::Op::kGetRows) || node.nsrc != 2 || node.nout != 1 ||
      inputs.size() != 2 || outputs.size() != 1)
    return Reject("selected rows require one two-input GetRows");
  const auto& w = inputs[0];
  const auto& ids = inputs[1];
  const auto& out = outputs[0];
  const auto& f = w.desc.format;
  const auto parts = format::PartsOf(f);
  if (!parts || f.encoding != format::Encoding::kNvfp4 || f.layout != format::Layout::kSplit ||
      f.shape.rank != 2 || f.stacked != 1 || f.shape.ne[0] >= (1ULL << 30) ||
      f.shape.ne[1] >= (1ULL << 31) || !w.paged_weight || w.paged_weight->expert_array ||
      !ids.side_rows_component || *ids.side_rows_component != w.paged_weight->component ||
      w.parts.size() != parts->size())
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
             .count = parts->size(),
             .ids = ids.tensor,
             .out = out.tensor,
             .rows = ids.desc.ne(0)};
  if (!Packed(g.ids, GGML_TYPE_I32, g.rows, 1) ||
      !Packed(g.out, GGML_TYPE_F32, f.shape.ne[0], g.rows))
    return Reject("selected row input or output physical geometry differs");
  constexpr std::array roles{Role::kCodes, Role::kBlockScales, Role::kTensorScale};
  std::uint64_t resident = 0;
  for (std::size_t i = 0; i < g.count; ++i) {
    const auto at = std::ranges::find(w.parts, roles[i], &kg::SelectedPart::role);
    if (at == w.parts.end() ||
        std::ranges::count(w.parts, roles[i], &kg::SelectedPart::role) != 1 ||
        at->tensor == nullptr)
      return Reject("selected row part missing or duplicated");
    g.parts[i] = at->tensor;
    if (i == 0) {
      if (at->tensor->ne[1] <= 0) return Reject("selected row slots are empty");
      resident = static_cast<std::uint64_t>(at->tensor->ne[1]);
      if (resident > f.shape.ne[1]) return Reject("selected row slots exceed their table");
    }
    if (!Packed(at->tensor, i == 2 ? GGML_TYPE_F32 : GGML_TYPE_I8,
                i == 0   ? f.shape.ne[0] / 2
                : i == 1 ? f.shape.ne[0] / 16
                         : 1,
                i == 2 ? 1 : resident))
      return Reject("selected row part has inconsistent physical geometry");
  }
  g.effective.shape.ne[1] = resident;
  return g;
}
std::expected<std::array<DevicePart, 3>, Failure> BindParts(const Geometry& g) {
  constexpr std::array roles{Role::kCodes, Role::kBlockScales, Role::kTensorScale};
  if (!Packed(g.parts[0], GGML_TYPE_I8, g.effective.shape.ne[0] / 2, g.effective.shape.ne[1]) ||
      !Packed(g.parts[1], GGML_TYPE_I8, g.effective.shape.ne[0] / 16, g.effective.shape.ne[1]) ||
      (g.count == 3 && !Packed(g.parts[2], GGML_TYPE_F32, 1, 1)) ||
      !Packed(g.ids, GGML_TYPE_I32, g.rows, 1) ||
      !Packed(g.out, GGML_TYPE_F32, g.effective.shape.ne[0], g.rows))
    return Reject("selected row geometry changed after lowering");
  std::array<DevicePart, 3> result{};
  std::array<const ggml_tensor*, 5> all{g.parts[0], g.parts[1], g.ids, g.out, g.parts[2]};
  const std::size_t count = g.count + 2;
  for (std::size_t i = 0; i < count; ++i) {
    const auto* t = all[i];
    if (!kg::detail::Current(t) || Address(t) == 0)
      return Reject("selected rows require current physical bindings");
    const auto bytes = ggml_nbytes(t);
    if (bytes > UINT64_MAX - Address(t)) return Reject("selected rows address range overflows");
    for (std::size_t j = 0; j < i; ++j) {
      const auto* other = all[j];
      if (Address(t) < Address(other) + ggml_nbytes(other) && Address(other) < Address(t) + bytes)
        return Reject("selected rows operand ranges overlap");
    }
  }
  if (Address(g.ids) % 4 != 0 || Address(g.out) % 4 != 0)
    return Reject("selected row IDs or output are misaligned");
  for (std::size_t i = 0; i < g.count; ++i)
    result[i] = {roles[i], g.parts[i]->data, ggml_nbytes(g.parts[i])};
  auto operand = OperandOf({g.effective, std::span(result).first(g.count), 0});
  if (!operand) return Reject(operand.error());
  return result;
}
class Runtime final : public kg::SelectedLaunch {
 public:
  Runtime(const kg::SelectedSource* owner, providers::NativeStream stream)
      : kg::SelectedLaunch(owner, stream) {}
  bool faulted() const override { return failed; }
  // RunRows only queues its existing kernel; IDs and part slots remain stable
  // allocations whose contents ModelRunner stages before capture or replay.
  bool capturable() const override { return true; }
  bool failed = false;
};
class BoundCall final : public kg::BoundSelectedCall {
 public:
  BoundCall(Runtime& runtime, Geometry geometry) : runtime_(runtime), geometry_(geometry) {}
  std::expected<void, Failure> Run(providers::NativeStream stream) const override {
    if (runtime_.failed || stream.handle != runtime_.stream().handle)
      return Reject("selected rows cannot run on a failed or foreign stream");
    const auto parts = BindParts(geometry_);
    if (!parts) return std::unexpected(parts.error());
    auto ran =
        RunRows({.weight = {geometry_.effective, std::span(*parts).first(geometry_.count), 0},
                 .ids = static_cast<const std::int32_t*>(geometry_.ids->data),
                 .count = geometry_.rows,
                 .out = static_cast<float*>(geometry_.out->data),
                 .out_stride = geometry_.effective.shape.ne[0],
                 .bad = nullptr},
                stream.handle);
    if (!ran) {
      runtime_.failed = true;
      return std::unexpected(Failure{.error = kg::KernelError::kUnknown, .detail = ran.error()});
    }
    return {};
  }

 private:
  Runtime& runtime_;
  Geometry geometry_;
};
class Call final : public kg::SelectedCall {
 public:
  Call(const kg::SelectedSource* owner, Geometry geometry, base::Sha256Digest identity)
      : kg::SelectedCall(owner), geometry_(geometry), identity_(identity) {}
  std::uint64_t scratch_bytes() const override { return 0; }
  std::uint64_t host_bytes() const override {
    static_assert(sizeof(Call) + sizeof(BoundCall) + 128 <= 2048);
    return 2048;
  }
  std::uint64_t launches() const override { return 1; }
  std::expected<std::unique_ptr<kg::BoundSelectedCall>, Failure> Bind(
      const execution::Implementation& implementation, kg::SelectedLaunch& launch,
      kg::SelectedWorkspace) const override {
    if (launch.owner() != owner() || execution::IdentityOf(implementation) != identity_)
      return Reject("selected row source or implementation identity changed");
    const auto bound = decode::Bind(implementation);
    if (!bound || bound->tmpl != Template::kRows || bound->gate != DecoderId::kNvfp4Split)
      return Reject("selected row declaration is not the native split decoder");
    if (auto parts = BindParts(geometry_); !parts) return std::unexpected(parts.error());
    return std::make_unique<BoundCall>(static_cast<Runtime&>(launch), geometry_);
  }

 private:
  Geometry geometry_;
  base::Sha256Digest identity_;
};
class Source final : public kg::SelectedSource {
 public:
  explicit Source(execution::Implementation implementation)
      : implementation_(std::move(implementation)) {}
  std::vector<execution::Implementation> Implementations() const override {
    return {implementation_};
  }
  std::vector<execution::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view name) const override { return name == kName; }
  std::uint64_t persistent_bytes() const override { return 0; }
  std::expected<std::unique_ptr<kg::SelectedCall>, Failure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const kg::SelectedOperand> inputs,
      std::span<const kg::SelectedOperand> outputs) const override {
    if (execution::IdentityOf(implementation) != execution::IdentityOf(implementation_))
      return Reject("selected rows received a foreign declaration");
    auto geometry = Shape(node, inputs, outputs);
    if (!geometry) return std::unexpected(geometry.error());
    return std::make_unique<Call>(this, *geometry, execution::IdentityOf(implementation_));
  }
  std::expected<std::unique_ptr<kg::SelectedLaunch>, Failure> Launch(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      kg::SelectedWorkspace persistent) const override {
    const auto facts = providers::QueryDeviceFacts(device);
    if (!facts || !CompiledFor(facts->architecture / 10) || persistent.bytes != 0)
      return Reject("selected rows device or persistent workspace differs");
    const auto native = execution.Submission(stream);
    if (!native) return Reject(native.error().detail);
    return std::make_unique<Runtime>(this, *native);
  }

 private:
  execution::Implementation implementation_;
};
}  // namespace
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> RowsSelectedSource() {
  auto implementations = decode::Implementations();
  const auto at = std::ranges::find(implementations, kName, &execution::Implementation::name);
  if (at == implementations.end()) return std::unexpected("split NVFP4 rows are unavailable");
  return std::make_shared<Source>(std::move(*at));
}
}  // namespace llmp::kernels::decode
