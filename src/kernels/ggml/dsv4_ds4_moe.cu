// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_ds4_moe.h"
#include "kernels/ggml/dsv4_ds4_moe_raw.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {
auto Reject(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}
template <typename T>
T* Pointer(Ds4CacheBuffer b) {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(b.address));
}
std::expected<ds4_moe::Device, KernelFailure> Device(const LaunchContext& launch) {
  const auto& info = ggml_cuda_info();
  if (launch.device() < 0 || launch.device() >= info.device_count)
    return Reject("ds4 MoE has no native device facts");
  const auto& d = info.devices[launch.device()];
  if (d.cc != 1210 || d.warp_size != 32 || d.nsm <= 0)
    return Reject("original ds4 MoE closure initially requires sm_121");
  return ds4_moe::Device{.cc = d.cc, .sm = d.nsm, .warp = d.warp_size, .shared = d.smpbo};
}
bool Disjoint(const LaunchContext& launch, std::span<const Ds4CacheBuffer> buffers) {
  const auto w = launch.workspace();
  std::uint64_t end = 0;
  if (__builtin_add_overflow(w.base, w.size.value(), &end)) return false;
  for (auto b : buffers) {
    if (b.address == 0 || w.size.value() == 0) continue;
    std::uint64_t bend = 0;
    if (__builtin_add_overflow(b.address, b.bytes, &bend) || (w.base < bend && b.address < end))
      return false;
  }
  return true;
}
auto Router(const Ds4Router& d) {
  return ds4_moe::Router{.logits = Pointer<float>(d.logits),
                         .bias = Pointer<const float>(d.bias),
                         .hash = Pointer<const std::int32_t>(d.hash),
                         .tokens = Pointer<const std::int32_t>(d.tokens),
                         .selected = Pointer<std::int32_t>(d.selected),
                         .weights = Pointer<float>(d.weights),
                         .probabilities = Pointer<float>(d.probabilities),
                         .rows = static_cast<int>(d.rows),
                         .hash_rows = d.hash_rows,
                         .select = static_cast<int>(d.select)};
}
std::expected<ds4_moe::Plan, KernelFailure> Plan(const LaunchContext& launch, const Ds4Moe& d) {
  if (auto checked = CheckDs4Moe(d); !checked) return std::unexpected(checked.error());
  const auto device = Device(launch);
  if (!device) return std::unexpected(device.error());
  const std::array operands = {d.input,         d.gate_weights,
                               d.up_weights,    d.down_weights,
                               d.selected,      d.weights,
                               d.compact_ids,   d.compact_weights,
                               d.ids_source,    d.ids_destination,
                               d.expert_bounds, d.work,
                               d.input_quant,   d.down_quant,
                               d.gate,          d.up,
                               d.middle,        d.down,
                               d.sum,           d.producer.storage};
  if (!Disjoint(launch, operands))
    return Reject("ds4 MoE borrowed operands overlap native workspace");
  ds4_moe::Plan plan;
  if (d.tier == Ds4MoeTier::kClassic &&
      !ds4_moe::PlanClassic(*device, static_cast<int>(d.shape.rows),
                            static_cast<int>(d.shape.input), static_cast<int>(d.shape.middle),
                            static_cast<int>(d.shape.output), plan))
    return Reject("original classic MoE has no bounded native stream-K plan");
  return plan;
}
}  // namespace

std::expected<void, KernelFailure> RunDs4Router(LaunchContext& launch, const Ds4Router& d) {
  if (auto checked = CheckDs4Router(d); !checked) return checked;
  if (auto device = Device(launch); !device) return std::unexpected(device.error());
  const std::array operands = {d.logits,   d.bias,    d.hash,         d.tokens,
                               d.selected, d.weights, d.probabilities};
  if (!Disjoint(launch, operands)) return Reject("ds4 router overlaps native workspace");
  return launch.Run(base::Bytes(0), [d](auto& context) {
    CUDA_CHECK(ds4_moe::Select(Router(d), context.stream()));
  });
}
std::expected<void, KernelFailure> RunDs4RouterCooperative(LaunchContext& launch,
                                                           const Ds4RouterCooperative& d) {
  if (auto checked = CheckDs4RouterCooperative(d); !checked) return checked;
  const auto device = Device(launch);
  if (!device) return std::unexpected(device.error());
  if (!ds4_moe::CooperativeFits(device->sm, static_cast<int>(d.router.rows)))
    return Reject("original cooperative router lacks its 48-block resident capacity");
  const auto& r = d.router;
  const std::array operands = {r.logits,  r.bias,          r.hash,  r.tokens,     r.selected,
                               r.weights, r.probabilities, d.input, d.projection, d.partials};
  if (!Disjoint(launch, operands))
    return Reject("ds4 cooperative router overlaps native workspace");
  return launch.Run(base::Bytes(0), [d](auto& context) {
    CUDA_CHECK(ds4_moe::Cooperative(Router(d.router), Pointer<const float>(d.input),
                                    Pointer<const void>(d.projection), Pointer<float>(d.partials),
                                    context.stream()));
  });
}
std::expected<void, KernelFailure> RunDs4SharedSwiglu(LaunchContext& launch,
                                                      const Ds4SharedSwiglu& d) {
  if (auto checked = CheckDs4SharedSwiglu(d); !checked) return checked;
  if (auto device = Device(launch); !device) return std::unexpected(device.error());
  const std::array operands = {d.gate, d.up, d.output};
  if (!Disjoint(launch, operands)) return Reject("ds4 shared SwiGLU overlaps native workspace");
  return launch.Run(base::Bytes(0), [d](auto& context) {
    CUDA_CHECK(ds4_moe::Shared(Pointer<const float>(d.gate), Pointer<const float>(d.up),
                               Pointer<float>(d.output), static_cast<int>(d.rows),
                               static_cast<int>(d.width), context.stream()));
  });
}
std::expected<void, KernelFailure> RunDs4MoeSum(LaunchContext& launch, const Ds4MoeSum& d) {
  if (auto checked = CheckDs4MoeSum(d); !checked) return checked;
  if (auto device = Device(launch); !device) return std::unexpected(device.error());
  const std::array operands = {d.slots, d.output};
  if (!Disjoint(launch, operands)) return Reject("ds4 sum overlaps native workspace");
  return launch.Run(base::Bytes(0), [d](auto& context) {
    CUDA_CHECK(ds4_moe::Sum(Pointer<const float>(d.slots), Pointer<float>(d.output),
                            static_cast<int>(d.rows), static_cast<int>(d.width), context.stream()));
  });
}
std::expected<std::uint64_t, KernelFailure> PlanDs4MoeScratch(const LaunchContext& launch,
                                                              const Ds4Moe& d) {
  const auto plan = Plan(launch, d);
  if (!plan) return std::unexpected(plan.error());
  return plan->fixup;
}
std::expected<void, KernelFailure> RunDs4Moe(LaunchContext& launch, const Ds4Moe& d) {
  const auto plan = Plan(launch, d);
  if (!plan) return std::unexpected(plan.error());
  const auto device = *Device(launch);
  return launch.Run(base::Bytes(plan->fixup), [d, device, plan = *plan](auto& context) {
    ggml_cuda_pool_alloc<char> fixup(context.pool());
    if (plan.fixup != 0) fixup.alloc(static_cast<std::size_t>(plan.fixup));
    if (internal::CudaErrorPending()) return;
    const ds4_moe::Call call{
        .input = Pointer<const float>(d.input),
        .gate_weights = Pointer<const void>(d.gate_weights),
        .up_weights = Pointer<const void>(d.up_weights),
        .down_weights = Pointer<const void>(d.down_weights),
        .selected = Pointer<const std::int32_t>(d.selected),
        .weights = Pointer<const float>(d.weights),
        .compact_ids = Pointer<std::int32_t>(d.compact_ids),
        .compact_weights = Pointer<float>(d.compact_weights),
        .ids_source = Pointer<std::int32_t>(d.ids_source),
        .ids_destination = Pointer<std::int32_t>(d.ids_destination),
        .bounds = Pointer<std::int32_t>(d.expert_bounds),
        .work = Pointer<int>(d.work),
        .input_quant =
            Pointer<void>(d.producer.storage.address != 0 ? d.producer.storage : d.input_quant),
        .down_quant = Pointer<void>(d.down_quant),
        .gate = Pointer<float>(d.gate),
        .up = Pointer<float>(d.up),
        .middle = Pointer<float>(d.middle),
        .down = Pointer<float>(d.down),
        .sum = Pointer<float>(d.sum),
        .rows = static_cast<int>(d.shape.rows),
        .input_width = static_cast<int>(d.shape.input),
        .middle_width = static_cast<int>(d.shape.middle),
        .output_width = static_cast<int>(d.shape.output),
        .selected_stride = static_cast<int>(d.selected_stride),
        .weight_stride = static_cast<int>(d.weight_stride),
        .tier = static_cast<int>(d.tier),
        .produced = d.producer.storage.address != 0};
    CUDA_CHECK(ds4_moe::Moe(device, call, plan, fixup.get(), context.stream()));
  });
}
}  // namespace jitllm::kernels::ggml
