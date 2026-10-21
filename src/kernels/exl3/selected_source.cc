// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernels/exl3/selected_source.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "kernels/decode/selected_source.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/reconstruction_tuning.h"
#include "kernels/ggml/selected_operation.h"
#include "llmp_exl3_kernels.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::exl3 {
namespace {
namespace kg = ggml;
using kg::SelectedOperand;
using kg::SelectedWorkspace;
using Failure = kg::KernelFailure;

std::unexpected<Failure> Reject(std::string message) {
  return std::unexpected(Failure{.detail = std::move(message)});
}
Failure Convert(const KernelFailure& failure) {
  return {.error = failure.error == KernelError::kUnknown ? kg::KernelError::kUnknown
                                                          : kg::KernelError::kRejected,
          .detail = failure.detail};
}
std::uint64_t Address(const ggml_tensor* tensor) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tensor->data));
}
std::uint64_t Align(std::uint64_t bytes) { return (bytes + 255U) / 256U * 256U; }
constexpr std::uint64_t kPersistent = (kLockBytes + 255U) / 256U * 256U;

class Runtime final : public kg::SelectedLaunch {
 public:
  Runtime(const kg::SelectedSource* owner, providers::NativeStream native, int device,
          providers::DeviceExecution& execution, providers::StreamId stream, std::uint64_t locks,
          int sms)
      : kg::SelectedLaunch(owner, native),
        device_(device),
        execution_(execution),
        stream_(stream),
        locks_(locks),
        sms_(sms) {}
  bool faulted() const override { return launch && launch->faulted(); }
  // Bind runs on the owning device driver, before submission/capture. Lower
  // only builds immutable descriptors and may run concurrently without entering
  // this state. Unused registered sources claim no live lock context/handle.
  std::expected<void, Failure> Ensure() {
    if (launch) return {};
    auto lt = ReconGemm::Create();
    if (!lt) return std::unexpected(Convert(lt.error()));
    auto made = LaunchContext::Create(device_, execution_, stream_, locks_, true);
    if (!made) return std::unexpected(Convert(made.error()));
    if ((*made)->sm_count() != sms_) return Reject("EXL3 source device SM count changed");
    gemm = std::move(*lt);
    launch = std::move(*made);
    return {};
  }
  std::unique_ptr<LaunchContext> launch;
  std::unique_ptr<ReconGemm> gemm;

 private:
  int device_;
  providers::DeviceExecution& execution_;
  providers::StreamId stream_;
  std::uint64_t locks_;
  int sms_;
};

enum class Kind : std::uint8_t { kGemm, kGemv, kMulti, kReconstruct, kFused, kBias };

class Bound final : public kg::BoundSelectedCall {
 public:
  Bound(Runtime& runtime, Kernel kernel, Kind kind)
      : runtime_(runtime), kernel_(kernel), kind_(kind) {}
  std::expected<void, Failure> Run(providers::NativeStream stream) const override {
    if (stream.handle != runtime_.stream().handle)
      return Reject("an EXL3 operation cannot launch on a foreign stream");
    std::expected<void, KernelFailure> ran;
    switch (kind_) {
      case Kind::kGemm:
        ran = kernel_.Run(*runtime_.launch, linear, plan, 0);
        break;
      case Kind::kGemv:
        ran = kernel_.Run(*runtime_.launch, linear, gemv, 0);
        break;
      case Kind::kMulti:
        ran = runtime_.launch->WritePointerTable(multi.trellis_table, multi.written);
        if (ran) {
          ran = runtime_.launch->AfterQueued(kernel_.Run(*runtime_.launch, multi, multi_plan));
        }
        break;
      case Kind::kReconstruct:
      case Kind::kFused:
        ran = kernel_.Run(*runtime_.launch, *runtime_.gemm, recon, algorithms);
        break;
      case Kind::kBias:
        ran = kernel_.Run(*runtime_.launch, bias);
        break;
    }
    if (!ran) return std::unexpected(Convert(ran.error()));
    return {};
  }
  MultiLinearOperands multi{};
  MultiGemmPlan multi_plan{};
  GemvPlan gemv{};
  LinearOperands linear{};
  GemmPlan plan{};
  ReconstructedOperands recon{};
  std::vector<LtAlgorithm> algorithms;
  BiasOperands bias{};

 private:
  Runtime& runtime_;
  Kernel kernel_;
  Kind kind_;
};

class Call final : public kg::SelectedCall {
 public:
  Call(const kg::SelectedSource* owner, Kind kind, const execution::Implementation* native,
       base::Sha256Digest identity)
      : kg::SelectedCall(owner), kind(kind), native(native), identity(identity) {}
  std::uint64_t scratch_bytes() const override {
    if (kind == Kind::kBias) return 0;
    const auto input = Align(static_cast<std::uint64_t>(rows) * static_cast<unsigned>(k) * 2U);
    if (kind == Kind::kMulti) return 256 + input * 2;
    if (kind == Kind::kGemm || kind == Kind::kGemv) return input;
    const auto slice = Align(static_cast<std::uint64_t>(k) *
                             static_cast<unsigned>(std::min(n, kReconstructSliceColumns)) * 2U);
    return slice + (kind == Kind::kFused ? 0 : input);
  }
  std::uint64_t host_bytes() const override {
    // At most 16 reconstruction slices. Cover both descriptors (2 KiB),
    // two pinned-algorithm copies (4 KiB), twice the maximum capacity of
    // both per-run slice vectors (8 KiB), slice indices and allocation
    // headers (under 1 KiB). The checked sum is below this 16 KiB owner.
    static_assert(sizeof(Call) <= 1024 && sizeof(Bound) <= 1024);
    static_assert(sizeof(LtAlgorithm) <= 128 && sizeof(ReconstructOperands) <= 128 &&
                  sizeof(ReconGemmOperands) <= 128);
    static_assert(kMaxDimension <= 16 * kReconstructSliceColumns);
    return 16384;
  }
  std::uint64_t launches() const override {
    if (kind == Kind::kMulti) return 3;  // lazy zero, table upload, cooperative product
    if (kind == Kind::kGemm || kind == Kind::kGemv || kind == Kind::kBias)
      return 2;  // includes lazy lock zeroing
    return 1U +
           2U * static_cast<unsigned>((n + kReconstructSliceColumns - 1) /
                                      kReconstructSliceColumns) +
           (kind == Kind::kFused ? 0U : 2U);
  }
  std::expected<std::unique_ptr<kg::BoundSelectedCall>, Failure> Bind(
      const execution::Implementation& implementation, kg::SelectedLaunch& launch,
      SelectedWorkspace scratch) const override {
    if (launch.owner() != owner()) return Reject("foreign selected-source launch");
    auto& runtime = static_cast<Runtime&>(launch);
    if (auto ready = runtime.Ensure(); !ready) return std::unexpected(ready.error());
    if (scratch.bytes < scratch_bytes() || scratch.base % 256 != 0 ||
        scratch.base + scratch.bytes < scratch.base)
      return Reject("EXL3 selected operation's declared scratch is not covered");
    if (native == nullptr || execution::IdentityOf(implementation) != identity)
      return Reject("a selected EXL3 identity changed after lowering");
    auto kernel = Kernel::Bind(*native);
    if (!kernel) return std::unexpected(Convert(kernel.error()));
    auto bound = std::make_unique<Bound>(runtime, *kernel, kind);
    if (kind == Kind::kBias) {
      bound->bias = {
          .x = Address(x), .bias = Address(bias), .y = Address(y), .rows = rows, .columns = n};
      auto checked = CheckBias(bound->bias);
      if (!checked) return std::unexpected(Convert(checked.error()));
      return bound;
    }
    Weights weights{.trellis = Address(parts[0]),
                    .suh = Address(parts[1]),
                    .svh = Address(parts[2]),
                    .k = k,
                    .n = n,
                    .bits = bits};
    if (auto checked = CheckWeights(weights); !checked)
      return std::unexpected(Convert(checked.error()));
    if (kind == Kind::kMulti) {
      Weights second{.trellis = Address(second_parts[0]),
                     .suh = Address(second_parts[1]),
                     .svh = Address(second_parts[2]),
                     .k = k,
                     .n = n,
                     .bits = bits};
      const auto one_output = static_cast<std::uint64_t>(rows) * static_cast<unsigned>(n) *
                              static_cast<std::uint64_t>(OutputBytes(output));
      if (Address(y) + one_output < Address(y) || Address(second_y) != Address(y) + one_output)
        return Reject("paired EXL3 results are not adjacent in their declared placement");
      bound->multi = {.first = weights,
                      .second = second,
                      .trellis_table = scratch.base,
                      .suh_table = scratch.base + 16,
                      .svh_table = scratch.base + 32,
                      .written = MultiGemmTables(weights, second),
                      .x = Address(x),
                      .a_had = scratch.base + 256,
                      .y = Address(y),
                      .output = output,
                      .m = rows};
      bound->multi_plan = multi_plan;
      auto resident = runtime.launch->MultiGemmCoresident(bits, multi_plan.shape, output);
      if (!resident) return std::unexpected(Convert(resident.error()));
      auto checked = CheckMultiGemm(bound->multi, multi_plan, *resident, runtime.launch->locks());
      if (!checked) return std::unexpected(Convert(checked.error()));
    } else if (kind == Kind::kGemm || kind == Kind::kGemv) {
      bound->linear = {.weights = weights,
                       .x = Address(x),
                       .a_had = scratch.base,
                       .y = Address(y),
                       .output = output,
                       .m = rows};
      bound->plan = plan;
      if (kind == Kind::kGemv) {
        bound->gemv = gemv;
        auto actual = runtime.launch->UpstreamGemv(weights, output, rows);
        if (!actual) return std::unexpected(Convert(actual.error()));
        if (!*actual || (*actual)->config != gemv.config || (*actual)->blocks != gemv.blocks)
          return Reject("EXL3 GEMV selection differs from the queried occupancy facts");
        auto resident = runtime.launch->GemvCoresident(bits, output, rows, gemv.config);
        if (!resident) return std::unexpected(Convert(resident.error()));
        if (auto checked = CheckGemv(bound->linear, gemv, *resident, runtime.launch->locks());
            !checked)
          return std::unexpected(Convert(checked.error()));
      } else {
        auto resident = runtime.launch->GemmCoresident(bits, plan.shape, output);
        if (!resident) return std::unexpected(Convert(resident.error()));
        if (auto checked = CheckGemm(bound->linear, plan, *resident, runtime.launch->locks());
            !checked)
          return std::unexpected(Convert(checked.error()));
      }
    } else {
      const bool fused = kind == Kind::kFused;
      const auto input_bytes =
          fused ? 0 : Align(static_cast<std::uint64_t>(rows) * static_cast<unsigned>(k) * 2U);
      bound->recon = {.weights = weights,
                      .x = Address(x),
                      .xh = fused ? 0 : scratch.base,
                      .w = scratch.base + input_bytes,
                      .y = Address(y),
                      .output = output,
                      .m = rows,
                      .bias = 0};
      bound->algorithms = algorithms;
      auto checked = CheckReconstructedScratch(bound->recon, fused);
      if (!checked) return std::unexpected(Convert(checked.error()));
      // Validate each exact pinned cuBLASLt problem before any work can run.
      int column = 0;
      for (const auto& algorithm : algorithms) {
        const ReconGemmOperands product{
            .w = bound->recon.w,
            .x = fused ? bound->recon.x : bound->recon.xh,
            .y = bound->recon.y + static_cast<std::uint64_t>(column) *
                                      static_cast<std::uint64_t>(OutputBytes(output)),
            .m = rows,
            .k = k,
            .n = algorithm.n,
            .ldc = n,
            .output = output};
        auto accepted = runtime.gemm->Check(product, algorithm, runtime.launch->sm_count());
        if (!accepted) return std::unexpected(Convert(accepted.error()));
        column += algorithm.n;
      }
      if (column != n) return Reject("EXL3 reconstruction does not cover its complete output");
    }
    return bound;
  }
  Kind kind;
  const execution::Implementation* native;
  base::Sha256Digest identity;
  std::array<ggml_tensor*, 4> second_parts{};
  ggml_tensor* second_y = nullptr;
  MultiGemmPlan multi_plan{};
  GemvPlan gemv{};
  int rows = 0, k = 0, n = 0, bits = 0;
  Output output = Output::kF16;
  std::array<ggml_tensor*, 4> parts{};
  ggml_tensor* x = nullptr;
  ggml_tensor* y = nullptr;
  ggml_tensor* bias = nullptr;
  GemmPlan plan{};
  std::vector<LtAlgorithm> algorithms;
};

bool Dense(const SelectedOperand& operand, format::Encoding encoding) {
  return operand.tensor != nullptr && operand.parts.empty() &&
         operand.desc.format.encoding == encoding &&
         operand.desc.format.layout == format::Layout::kDense && ir::IsContiguous(operand.desc) &&
         operand.desc.ne(2) == 1 && operand.desc.ne(3) == 1;
}
constexpr std::array kF16 = {format::Encoding::kF16};
constexpr std::array kOutputs = {format::Encoding::kF16, format::Encoding::kF32};
constexpr std::array kF32 = {format::Encoding::kF32};
constexpr std::array kTrellis = {format::Encoding::kExl3Trellis};
constexpr std::array kTiles = {format::Layout::kExl3Tiles};
constexpr std::array kDense = {format::Layout::kDense};
constexpr std::array<execution::OperandRule, 3> kLinearRules = {{
    {.operand = 0, .encodings = kTrellis, .layouts = kTiles},
    {.operand = 1, .encodings = kF16, .layouts = kDense},
    {.operand = execution::kOutputOperand, .encodings = kOutputs, .layouts = kDense},
}};
constexpr std::array<execution::OperandRule, 5> kPairRules = {{
    {.operand = 0, .encodings = kTrellis, .layouts = kTiles},
    {.operand = 1, .encodings = kTrellis, .layouts = kTiles},
    {.operand = 2, .encodings = kF16, .layouts = kDense},
    {.operand = execution::kOutputOperand, .encodings = kF32, .layouts = kDense},
    {.operand = execution::kOutputOperand + 1, .encodings = kF32, .layouts = kDense},
}};
constexpr std::array<execution::OperandRule, 3> kBiasRules = {{
    {.operand = 0, .encodings = kF16, .layouts = kDense},
    {.operand = 1, .encodings = kF16, .layouts = kDense},
    {.operand = execution::kOutputOperand, .encodings = kF16, .layouts = kDense},
}};
bool AcceptsBias(const execution::CapabilityQuery& query) {
  const auto operand = [&](std::uint8_t id) -> const execution::OperandFormat* {
    const auto at = std::ranges::find(query.operands, id, &execution::OperandFormat::operand);
    return at == query.operands.end() ? nullptr : &*at;
  };
  const auto packed = [](const execution::OperandFormat& value) {
    if (!value.memory) return true;
    std::uint64_t stride = 2;
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
      if (value.ne[dimension] == 0 ||
          (value.ne[dimension] > 1 && value.memory->nb[dimension] != stride) ||
          __builtin_mul_overflow(stride, value.ne[dimension], &stride))
        return false;
    }
    return true;
  };
  const auto* x = operand(0);
  const auto* b = operand(1);
  const auto* y = operand(execution::kOutputOperand);
  return query.arch >= 80 && x && b && y && packed(*x) && packed(*b) && packed(*y) &&
         x->ne == y->ne && x->ne[0] >= 1 && x->ne[0] <= kMaxDimension && x->ne[1] >= 1 &&
         x->ne[1] <= kMaxRows && x->ne[2] == 1 && x->ne[3] == 1 && b->ne[0] == x->ne[0] &&
         b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1 &&
         // CheckBias's (CheckExtent's) element bound, so a planned bias lowers.
         static_cast<std::uint64_t>(x->ne[0]) * static_cast<std::uint64_t>(x->ne[1]) <= INT32_MAX;
}
std::string_view ProfileName(FrozenCache cache) {
  switch (cache) {
    case FrozenCache::k40G:
      return "40g";
    case FrozenCache::k45G:
      return "45g";
    case FrozenCache::k40O:
      return "40o";
    case FrozenCache::k45O:
      return "45o";
  }
  return {};
}
bool Optimized(FrozenCache cache) {
  return cache == FrozenCache::k40O || cache == FrozenCache::k45O;
}
std::expected<std::array<ggml_tensor*, 4>, Failure> PhysicalParts(const SelectedOperand& operand) {
  const auto extents = format::PartsOf(operand.desc.format);
  if (!extents || extents->size() != 4 || operand.parts.size() != 4)
    return Reject("EXL3 requires all four physical parts");
  constexpr std::array roles{format::PartRole::kCodes, format::PartRole::kInputSigns,
                             format::PartRole::kOutputSigns, format::PartRole::kCodebookMarker};
  std::array<ggml_tensor*, 4> result{};
  for (std::size_t i = 0; i < roles.size(); ++i) {
    const auto found = std::ranges::find(operand.parts, roles[i], &kg::SelectedPart::role);
    const auto extent = extents->Find(roles[i]);
    if (found == operand.parts.end() || found->tensor == nullptr || !extent ||
        std::ranges::count(operand.parts, roles[i], &kg::SelectedPart::role) != 1 ||
        ggml_nbytes(found->tensor) != extent->readable)
      return Reject("EXL3 has a missing, duplicate or incorrectly sized part");
    result[i] = found->tensor;
  }
  return result;
}
class Source final : public kg::SelectedSource {
 public:
  Source(FrozenCache cache, std::optional<GemvCapacity> capacity,
         std::optional<PackedCapacity> defaults, bool generic = false)
      : cache_(cache),
        generic_(generic),
        capacity_(capacity),
        defaults_(defaults),
        canonical_(exl3::Implementations()),
        declarations_(canonical_) {
    for (auto& declaration : declarations_) {
      const auto native_name = declaration.name;
      declaration.name =
          generic ? std::format("exl3.device.{}", native_name.substr(5))
                  : std::format("exl3.pinned.{}.{}", ProfileName(cache), native_name.substr(5));
      declaration.variant +=
          generic ? "; actual-device occupancy defaults; no measured pins"
                  : std::format("; frozen profile {}, exact cache keys; ordered shared IR",
                                ProfileName(cache));
      if (defaults_) {
        declaration.variant +=
            std::format("; default occupancy arch{} SMs{}", defaults_->arch, defaults_->sms);
        for (const auto& table : {defaults_->gemm, defaults_->multi})
          for (const auto& output : table)
            for (const auto& rate : output)
              for (const auto count : rate)
                declaration.variant += std::format("; packed={}", count);
      }
      if (capacity_) {
        declaration.variant += std::format("; GEMV arch{} SMs{}", capacity_->arch, capacity_->sms);
        for (const auto& output : capacity_->blocks)
          for (const auto& rows : output)
            for (const auto count : rows)
              declaration.variant += std::format("; occupancy={}", count);
      }
      if (native_name == "exl3.bias_add") {
        declaration.capability =
            execution::Capability{.op = ir::Primitive(ir::Op::kAdd),
                                  .operands = kBiasRules,
                                  .device = {.min_arch = defaults_   ? defaults_->arch
                                                         : capacity_ ? capacity_->arch
                                                                     : 121U,
                                             .max_arch = defaults_   ? defaults_->arch
                                                         : capacity_ ? capacity_->arch
                                                                     : 121U},
                                  .row_invariant = false,
                                  .capturable = false,
                                  .priority = 1,
                                  .accepts = &AcceptsBias};
      } else {
        auto& capability = *declaration.capability;
        // Pair() writes adjacent F32 results; Linear() permits F16 or F32.
        // Selection must reject other encodings before either lowering runs.
        capability.operands = native_name == "exl3.multi_linear.mgemm"
                                  ? std::span<const execution::OperandRule>(kPairRules)
                                  : std::span<const execution::OperandRule>(kLinearRules);
        capability.device.min_arch = 121;
        capability.device.max_arch = 121;
        capability.needs_tuning = true;
        if (native_name.starts_with("exl3.linear.reconstruct"))
          capability.priority = 4;
        else
          capability.shape.max_rows = native_name == "exl3.linear.gemv"          ? 8
                                      : native_name == "exl3.multi_linear.mgemm" ? 32
                                                                                 : 144;
      }
      if (defaults_ &&
          (native_name == "exl3.linear.gemm" || native_name == "exl3.multi_linear.mgemm")) {
        auto& capability = *declaration.capability;
        capability.device = {.min_arch = defaults_->arch, .max_arch = defaults_->arch};
        capability.needs_tuning = false;
        capability.shape.max_rows = kMaxRows;
        // Occupancy proves executable capacity, not a qualified replacement
        // for two separately selected products. Direct paired operations stay
        // available; automatic rewrites need their own measured preference.
        capability.fusion_needs_measurement = native_name == "exl3.multi_linear.mgemm";
      }
      if (native_name == "exl3.linear.gemv" && capacity_) {
        declaration.capability->device = {.min_arch = capacity_->arch, .max_arch = capacity_->arch};
      }
    }
    if (generic_) {
      for (std::size_t i = canonical_.size(); i-- > 0;) {
        if (canonical_[i].name == "exl3.linear.gemm" ||
            canonical_[i].name == "exl3.multi_linear.mgemm" ||
            (canonical_[i].name == "exl3.linear.gemv" && capacity_.has_value()) ||
            canonical_[i].name == "exl3.bias_add")
          continue;
        canonical_.erase(canonical_.begin() + static_cast<std::ptrdiff_t>(i));
        declarations_.erase(declarations_.begin() + static_cast<std::ptrdiff_t>(i));
      }
      for (auto& declaration : declarations_) {
        declaration.capability->priority = declaration.name.ends_with(".linear.gemv") ? 1 : 0;
        if (declaration.name.ends_with(".linear.gemm") ||
            declaration.name.ends_with(".linear.gemv")) {
          declaration.capability->needs_tuning = false;
        }
      }
    }
    base::Sha256 hash;
    hash.Update("llmp-exl3-selected-v1;exact-caches-and-device-capacity;deferred-bind-resources");
    for (const auto& impl : declarations_) {
      const auto id = execution::IdentityOf(impl);
      hash.Update(std::as_bytes(std::span(id)));
    }
    identity_ = hash.Finish();
  }
  std::vector<execution::Implementation> Implementations() const override { return declarations_; }
  std::string_view Name(std::string_view native) const {
    const auto at = std::ranges::find(canonical_, native, &execution::Implementation::name);
    return at == canonical_.end()
               ? std::string_view{}
               : declarations_[static_cast<std::size_t>(at - canonical_.begin())].name;
  }
  std::vector<execution::TunedShape> Tuning(std::uint32_t arch) const override {
    if (generic_) return {};
    if ((defaults_ && arch != defaults_->arch) || (capacity_ && arch != capacity_->arch)) return {};
    const bool pinned_device =
        (!defaults_ || defaults_->sms == 48) && (!capacity_ || capacity_->sms == 48);
    const auto sms = defaults_ ? defaults_->sms : capacity_ ? capacity_->sms : 48;
    auto result = pinned_device ? ReconstructionTuning(arch) : std::vector<execution::TunedShape>{};
    for (auto& fact : result) fact.implementation = Name(fact.implementation);
    if (arch != 121) return result;
    // Physical problems present in the authenticated caches, not model dispatch.
    constexpr std::array shapes{std::pair{896, 896}, std::pair{896, 128}, std::pair{896, 4864},
                                std::pair{4864, 896}, std::pair{896, 151936}};
    const auto add = [&](std::string_view implementation, int rows, int k, int n, Output output,
                         int bits) {
      const execution::TunedShape fact{.implementation = Name(implementation),
                                       .rows = static_cast<unsigned>(rows),
                                       .k = static_cast<unsigned>(k),
                                       .n = static_cast<unsigned>(n),
                                       .output_encoding = output == Output::kF32
                                                              ? format::Encoding::kF32
                                                              : format::Encoding::kF16,
                                       .weight_bits = static_cast<std::uint8_t>(bits)};
      if (!std::ranges::contains(result, fact)) result.push_back(fact);
    };
    for (const auto [k, n] : shapes) {
      for (int rows = 1; rows <= 144; ++rows) {
        for (const auto output : {Output::kF16, Output::kF32}) {
          for (const int bits : {4, 5, 6, 8}) {
            if (FrozenGemmPlan(cache_, arch, sms, rows, k, n, bits, output))
              add("exl3.linear.gemm", rows, k, n, output, bits);
            if (rows <= 32 && FrozenMultiGemmPlan(cache_, arch, sms, rows, k, n, bits, output))
              add("exl3.multi_linear.mgemm", rows, k, n, output, bits);
          }
          if (Optimized(cache_) && capacity_ && rows <= 8 && Gemv(rows, k, n, 4, output))
            add("exl3.linear.gemv", rows, k, n, output, 4);
        }
      }
    }
    return result;
  }
  bool Owns(std::string_view name) const override {
    return std::ranges::contains(declarations_, name, &execution::Implementation::name);
  }
  std::optional<GemvPlan> Gemv(int rows, int k, int n, int bits, Output output) const {
    if (!capacity_ || rows < 1 || rows > 8) return {};
    const auto& counts = capacity_->blocks[output == Output::kF32 ? 1 : 0][rows == 1 ? 0 : 1];
    return UpstreamGemvPlan(UpstreamCcClass(static_cast<int>(capacity_->arch / 10),
                                            static_cast<int>(capacity_->arch % 10)),
                            rows, k, n, bits, kCodebookMcg, counts[0], counts[1]);
  }
  std::uint64_t persistent_bytes() const override { return kPersistent; }
  std::optional<base::Sha256Digest> ExecutionIdentity() const override { return identity_; }
  std::expected<std::unique_ptr<kg::SelectedCall>, Failure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const SelectedOperand> inputs,
      std::span<const SelectedOperand> outputs) const override {
    const auto selected =
        std::ranges::find(declarations_, implementation.name, &execution::Implementation::name);
    if (selected == declarations_.end() ||
        execution::IdentityOf(*selected) != execution::IdentityOf(implementation))
      return Reject("foreign or changed selected EXL3 declaration");
    const auto index = static_cast<std::size_t>(selected - declarations_.begin());
    const auto& name = canonical_[index].name;
    const auto make = [&](Kind kind) {
      return std::make_unique<Call>(this, kind, &canonical_[index],
                                    execution::IdentityOf(implementation));
    };
    if (name == "exl3.multi_linear.mgemm") return Pair(node, inputs, outputs, make(Kind::kMulti));
    if (node.nout != 1 || outputs.size() != 1 || inputs.size() != 2)
      return Reject("EXL3 selected operation has an unsupported identity or arity");
    const auto& out = outputs[0];
    const bool bias = name == "exl3.bias_add";
    if (bias) {
      if (node.key != ir::Primitive(ir::Op::kAdd) || !Dense(inputs[0], format::Encoding::kF16) ||
          !Dense(inputs[1], format::Encoding::kF16) || !Dense(out, format::Encoding::kF16) ||
          inputs[0].desc.shape().ne != out.desc.shape().ne ||
          inputs[1].desc.ne(0) != out.desc.ne(0) || inputs[1].desc.ne(1) != 1 ||
          out.desc.ne(0) > kMaxDimension || out.desc.ne(1) > kMaxRows)
        return Reject("EXL3 bias requires contiguous F16 rows and one matching bias row");
      auto call = make(Kind::kBias);
      call->rows = static_cast<int>(out.desc.ne(1));
      call->n = static_cast<int>(out.desc.ne(0));
      call->x = inputs[0].tensor;
      call->bias = inputs[1].tensor;
      call->y = out.tensor;
      return call;
    }
    const auto& w = inputs[0];
    const auto& x = inputs[1];
    if (node.key != ir::Primitive(ir::Op::kMatMul) || !Dense(x, format::Encoding::kF16) ||
        (!Dense(out, format::Encoding::kF16) && !Dense(out, format::Encoding::kF32)) ||
        w.desc.format.encoding != format::Encoding::kExl3Trellis ||
        w.desc.format.layout != format::Layout::kExl3Tiles ||
        w.desc.format.params.codebook != format::Exl3Codebook::kMcg || w.desc.ne(2) != 1 ||
        w.desc.ne(3) != 1 || !RateCompiled(w.desc.format.params.k_bits) ||
        w.desc.ne(0) != x.desc.ne(0) || w.desc.ne(1) != out.desc.ne(0) ||
        x.desc.ne(1) != out.desc.ne(1) || x.desc.ne(1) > kMaxRows || w.desc.ne(0) > kMaxDimension ||
        w.desc.ne(1) > kMaxDimension)
      return Reject("EXL3 product's logical format or physical shape is unsupported");
    const auto pp = node.Params<ir::ProductParams>();
    if (pp.accumulate != ir::Accumulate::kDefault || pp.hint != ir::ProductHint::kNone)
      return Reject("EXL3 product requires its native accumulation and no hint");
    const Kind kind = name == "exl3.linear.gemv"          ? Kind::kGemv
                      : name == "exl3.linear.gemm"        ? Kind::kGemm
                      : name == "exl3.linear.reconstruct" ? Kind::kReconstruct
                                                          : Kind::kFused;
    auto call = make(kind);
    call->rows = static_cast<int>(x.desc.ne(1));
    call->k = static_cast<int>(w.desc.ne(0));
    call->n = static_cast<int>(w.desc.ne(1));
    call->bits = w.desc.format.params.k_bits;
    call->output = out.desc.format.encoding == format::Encoding::kF32 ? Output::kF32 : Output::kF16;
    call->x = x.tensor;
    call->y = out.tensor;
    const auto parts = PhysicalParts(w);
    if (!parts) return std::unexpected(parts.error());
    call->parts = *parts;
    if (kind == Kind::kGemv) {
      const auto plan = Gemv(call->rows, call->k, call->n, call->bits, call->output);
      if (!plan) return Reject("the frozen O profile does not select this GEMV");
      call->gemv = *plan;
    } else if (kind == Kind::kGemm) {
      const auto arch = defaults_ ? defaults_->arch : capacity_ ? capacity_->arch : 121U;
      const auto sms = defaults_ ? defaults_->sms : capacity_ ? capacity_->sms : 48;
      auto plan = generic_ ? DefaultGemmPlan(*defaults_, call->rows, call->k, call->n, call->bits,
                                             call->output)
                           : FrozenGemmPlan(cache_, arch, sms, call->rows, call->k, call->n,
                                            call->bits, call->output);
      if (!plan && defaults_)
        plan = DefaultGemmPlan(*defaults_, call->rows, call->k, call->n, call->bits, call->output);
      if (!defaults_ && UpstreamPath(call->rows) != LinearPath::kPacked)
        return Reject("EXL3 packed product is outside the approved reference routing");
      if (!plan) return Reject(plan.error());
      call->plan = *plan;
    } else {
      const auto expected =
          kind == Kind::kFused ? LinearPath::kReconstructFused : LinearPath::kReconstruct;
      if (UpstreamPath(call->rows) != expected)
        return Reject(
            "EXL3 selected reconstruction path disagrees with the approved row threshold");
      if ((defaults_ && (defaults_->arch != 121 || defaults_->sms != 48)) ||
          (capacity_ && (capacity_->arch != 121 || capacity_->sms != 48)))
        return Reject("reconstruction requires the measured device; use the packed default");
      auto algorithms = ReconstructionAlgorithms(121, call->rows, call->k, call->n, call->output);
      if (!algorithms) return Reject(algorithms.error());
      call->algorithms = std::move(*algorithms);
    }
    return call;
  }
  std::expected<std::unique_ptr<kg::SelectedLaunch>, Failure> Launch(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      SelectedWorkspace persistent) const override {
    if (persistent.bytes != kPersistent || persistent.base == 0 || persistent.base % 256 != 0 ||
        persistent.base + persistent.bytes < persistent.base)
      return Reject("EXL3 persistent locks lack their declared aligned workspace");
    const auto facts = providers::QueryDeviceFacts(device);
    const auto arch = defaults_ ? defaults_->arch : capacity_ ? capacity_->arch : 121U;
    const auto sms = defaults_ ? defaults_->sms : capacity_ ? capacity_->sms : 48;
    if (!facts || facts->architecture != arch * 10)
      return Reject("EXL3 source device differs from its queried architecture");
    if (defaults_) {
      const auto actual = QueryPackedCapacity(device);
      if (!actual || *actual != *defaults_) return Reject("EXL3 packed occupancy facts changed");
    }
    if (capacity_) {
      const auto actual = QueryGemvCapacity(device);
      if (!actual || *actual != *capacity_) return Reject("EXL3 GEMV occupancy facts changed");
    }
    auto native = execution.Submission(stream);
    if (!native) return Reject(native.error().detail);
    return std::make_unique<Runtime>(this, *native, device, execution, stream, persistent.base,
                                     sms);
  }

 private:
  std::expected<std::unique_ptr<kg::SelectedCall>, Failure> Pair(
      const ir::Node& node, std::span<const SelectedOperand> inputs,
      std::span<const SelectedOperand> outputs, std::unique_ptr<Call> call) const {
    if (node.key != ir::Fused(ir::FusedKind::kPreparedProducts) || node.nsrc != 3 ||
        node.nout != 2 || inputs.size() != 3 || outputs.size() != 2)
      return Reject("paired EXL3 product requires two weights and one input, with two results");
    const auto& w = inputs[0];
    const auto& other = inputs[1];
    const auto& x = inputs[2];
    const auto& first = outputs[0];
    const auto& second = outputs[1];
    if (!Dense(x, format::Encoding::kF16) || !Dense(first, format::Encoding::kF32) ||
        !Dense(second, format::Encoding::kF32) || first.desc.format != second.desc.format ||
        w.desc.format != other.desc.format ||
        w.desc.format.encoding != format::Encoding::kExl3Trellis ||
        w.desc.format.layout != format::Layout::kExl3Tiles ||
        w.desc.format.params.codebook != format::Exl3Codebook::kMcg ||
        !format::Validate(w.desc.format) || w.desc.ne(2) != 1 || w.desc.ne(3) != 1 ||
        !RateCompiled(w.desc.format.params.k_bits) || w.desc.ne(0) != x.desc.ne(0) ||
        w.desc.ne(1) != first.desc.ne(0) || x.desc.ne(1) != first.desc.ne(1) || x.desc.ne(1) < 1 ||
        x.desc.ne(1) > static_cast<std::uint64_t>(defaults_ ? kMaxRows : 32) ||
        w.desc.ne(0) > kMaxDimension || w.desc.ne(1) > kMaxDimension)
      return Reject("paired EXL3 operands violate the native complete-matrix contract");
    const auto first_parts = PhysicalParts(w);
    const auto second_parts = PhysicalParts(other);
    if (!first_parts) return std::unexpected(first_parts.error());
    if (!second_parts) return std::unexpected(second_parts.error());
    call->parts = *first_parts;
    call->second_parts = *second_parts;
    call->x = x.tensor;
    call->y = first.tensor;
    call->second_y = second.tensor;
    call->rows = static_cast<int>(x.desc.ne(1));
    call->k = static_cast<int>(w.desc.ne(0));
    call->n = static_cast<int>(w.desc.ne(1));
    call->bits = w.desc.format.params.k_bits;
    call->output = Output::kF32;
    const auto arch = defaults_ ? defaults_->arch : capacity_ ? capacity_->arch : 121U;
    const auto sms = defaults_ ? defaults_->sms : capacity_ ? capacity_->sms : 48;
    std::expected<MultiGemmPlan, std::string> plan = std::unexpected("no measured multi-GEMM pin");
    if (!generic_)
      plan = FrozenMultiGemmPlan(cache_, arch, sms, call->rows, call->k, call->n, call->bits,
                                 call->output);
    if (!plan && defaults_) {
      plan =
          DefaultMultiGemmPlan(*defaults_, call->rows, call->k, call->n, call->bits, call->output);
    }
    if (!plan) return Reject(plan.error());
    call->multi_plan = *plan;
    return call;
  }
  base::Sha256Digest identity_{};
  FrozenCache cache_;
  bool generic_ = false;
  std::optional<GemvCapacity> capacity_;
  std::optional<PackedCapacity> defaults_;
  std::vector<execution::Implementation> canonical_;
  std::vector<execution::Implementation> declarations_;
};
}  // namespace
std::expected<GemvCapacity, std::string> QueryGemvCapacity(int device) {
  const auto facts = providers::QueryDeviceFacts(device);
  int current = -1, sms = 0;
  if (!facts || facts->architecture < 800 || cudaGetDevice(&current) != cudaSuccess ||
      current != device ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
      sms < 1) {
    // A failed query leaves the runtime's last error set, which the next
    // launch context's Begin would take for its own launch's.
    (void)cudaGetLastError();
    return std::unexpected("GEMV occupancy requires a current CUDA device with compiled kernels");
  }
  GemvCapacity result;
  result.arch = facts->architecture / 10;
  result.sms = sms;
  for (std::size_t output = 0; output < 2; ++output) {
    for (std::size_t rows = 0; rows < 2; ++rows) {
      for (std::size_t config = 0; config < 2; ++config) {
        const void* kernel =
            llmp_exl3::GemvKernel(4, output != 0, static_cast<int>(rows), static_cast<int>(config));
        int per_sm = 0;
        if (kernel == nullptr ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, kernel, GemvThreads(static_cast<int>(config)), 0) != cudaSuccess ||
            per_sm < 1 || per_sm > 32) {
          (void)cudaGetLastError();
          return std::unexpected("the frozen GEMV kernel occupancy query failed");
        }
        result.blocks[output][rows][config] = per_sm * sms;
      }
    }
  }
  return result;
}
std::expected<std::shared_ptr<const kg::SelectedSource>, std::string> SelectedSource(
    FrozenCache cache, std::optional<GemvCapacity> capacity, std::optional<PackedCapacity> defaults,
    bool generic) {
  if (ProfileName(cache).empty() || (!generic && (Optimized(cache) != capacity.has_value())))
    return std::unexpected(
        "the frozen O profile requires queried GEMV capacities; G requires none");
  if (capacity) {
    for (const auto& output : capacity->blocks)
      for (const auto& rows : output)
        for (const auto count : rows)
          if (capacity->arch < 80 || capacity->sms < 1 || count < capacity->sms ||
              count > 32LL * capacity->sms || count % capacity->sms != 0)
            return std::unexpected("invalid GEMV co-resident count");
  }
  if (defaults) {
    if (defaults->arch < 80 || defaults->sms < 1 ||
        (capacity && (capacity->arch != defaults->arch || capacity->sms != defaults->sms)))
      return std::unexpected("inconsistent EXL3 device capacities");
    // Every rate/output must have the universally divisible128-wide tile.
    // Without this, shape-only candidate declaration cannot prove a default.
    for (const auto& table : {defaults->gemm, defaults->multi})
      for (const auto& output : table)
        for (const auto& rate : output) {
          if (rate[0] < 1) return std::unexpected("no universally divisible packed EXL3 default");
          for (const auto count : rate)
            if (count < 0 || count > 32LL * defaults->sms || count % defaults->sms != 0)
              return std::unexpected("invalid packed EXL3 occupancy");
        }
  }
  return std::make_shared<Source>(cache, capacity, defaults, generic);
}
std::expected<std::shared_ptr<const kg::SelectedSource>, std::string> DeviceSelectedSource(
    int device) {
  const auto facts = providers::QueryDeviceFacts(device);
  if (!facts) return std::unexpected(facts.error().text());
  if (!CompiledForDevice(facts->architecture / 10)) return decode::CompositeSelectedSource({});
  auto gemv = QueryGemvCapacity(device);
  if (!gemv) return std::unexpected(gemv.error());
  auto packed = QueryPackedCapacity(device);
  if (!packed) return std::unexpected(packed.error());
  std::vector<std::shared_ptr<const kg::SelectedSource>> sources;
  for (const auto cache :
       {FrozenCache::k40G, FrozenCache::k45G, FrozenCache::k40O, FrozenCache::k45O}) {
    // G profiles retain their measured GEMM-only routing; occupancy facts
    // still identify their actual device and prevent borrowing 48-SM pins.
    sources.push_back(std::make_shared<Source>(cache, *gemv, std::nullopt));
  }
  sources.push_back(std::make_shared<Source>(FrozenCache::k45O, *gemv, *packed, true));
  return decode::CompositeSelectedSource(std::move(sources));
}

}  // namespace llmp::kernels::exl3
