// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Host time per launch of RMSNorm-mul (docs/backend-proof.md, BP-F4 in P1:
// reported, not gated), for each implementation the plan can select
// (kernels/ggml/implementations.h), through four layers:
//   direct   GGML's launchers alone, called in a loop inside one launch
//            context run: upstream's own host cost;
//   run      one launch context run per operation, with the launchers and
//            no operand checks: what the K-C context adds;
//   checked  the ops.h entry point: the operand checks and a run;
//   plan     the kernel a bound plan selects (RmsNormMulKernel::Run).
// Decode's shape: one row of Qwen2.5-0.5B's width (896) in device VMM.
// Each batch submits a fixed number of operations, timing only the
// submission; the stream then drains, untimed, so the device queue never
// fills. Arms alternate batch by batch. Prints a Markdown table of the
// per-operation host time across batches; the report is under
// docs/experiments/launch-overhead/.
//
//   llmp_launch_bench [BATCHES [OPERATIONS_PER_BATCH]]

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <print>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

// GGML's launchers, called directly in the `direct` and `run` arms.
void ggml_cuda_op_rms_norm(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context& ctx, ggml_tensor* dst,
                                 ggml_tensor* mul_tensor);
void ggml_cuda_op_mul(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

namespace {

using Clock = std::chrono::steady_clock;
using llmp::base::Bytes;
using llmp::kernels::ggml::LaunchContext;
using llmp::kernels::ggml::RmsNormMulKernel;
using llmp::kernels::ggml::TensorArena;
using llmp::providers::Access;
using llmp::providers::BackingKind;
using llmp::providers::DeviceExecution;
using llmp::providers::FenceState;
using llmp::providers::StreamId;
using llmp::providers::VmmProvider;

constexpr std::int64_t kWidth = 896;
constexpr int kWarmupBatches = 20;

double Percentile(std::vector<double> samples, double p) {
  if (samples.empty()) {
    return 0;
  }
  std::ranges::sort(samples);
  return samples[static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1))];
}

bool Parse(std::string_view text, int& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size() && value > 0;
}

// Device VMM for `bytes`, mapped read-write; its address, or 0.
std::uint64_t DeviceBuffer(VmmProvider& memory, std::uint64_t bytes) {
  const std::uint64_t granule = memory.Granularity().value();
  if (granule == 0) {
    return 0;
  }
  const std::uint64_t size = (bytes + granule - 1) / granule * granule;
  std::size_t device = memory.Classes().size();
  for (std::size_t i = 0; i < memory.Classes().size(); ++i) {
    if (memory.Classes()[i].kind == BackingKind::kDevice) {
      device = i;
    }
  }
  if (device == memory.Classes().size()) {
    return 0;
  }
  const auto reservation = memory.Reserve(Bytes(size));
  const auto backing = memory.Create(device, Bytes(size));
  if (!reservation || !backing || !memory.Map(*reservation, Bytes(0), *backing) ||
      !memory.SetAccess(*reservation, Bytes(0), Bytes(size), Access::kReadWrite)) {
    return 0;
  }
  const auto range = memory.RangeOf(*reservation);
  return range ? range->base : 0;
}

// Waits for everything queued on the stream so far.
bool Drain(DeviceExecution& execution, StreamId stream) {
  const auto fence = execution.Record(stream);
  if (!fence) {
    return false;
  }
  for (;;) {
    const auto state = execution.Query(*fence);
    if (!state) {
      return false;
    }
    if (*state == FenceState::kComplete) {
      break;
    }
    std::this_thread::yield();
  }
  return execution.Release(*fence).has_value();
}

enum class Layer : std::uint8_t { kDirect, kRun, kChecked, kPlan };
constexpr std::array<Layer, 4> kLayers = {Layer::kDirect, Layer::kRun, Layer::kChecked,
                                          Layer::kPlan};

std::string_view LayerName(Layer layer) {
  switch (layer) {
    case Layer::kDirect:
      return "direct";
    case Layer::kRun:
      return "run";
    case Layer::kChecked:
      return "checked";
    case Layer::kPlan:
      return "plan";
  }
  return "";
}

struct Arm {
  bool fused = false;
  Layer layer = Layer::kDirect;
  std::vector<double> ns_per_operation;
};

}  // namespace

int main(int argc, char** argv) {
  int batches = 500;
  int per_batch = 64;
  if ((argc > 1 && !Parse(argv[1], batches)) || (argc > 2 && !Parse(argv[2], per_batch)) ||
      argc > 3) {
    std::println(stderr, "usage: llmp_launch_bench [BATCHES [OPERATIONS_PER_BATCH]]");
    return 2;
  }
  auto memory = llmp::providers::cuda::OpenDeviceMemory(0);
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
  if (!memory || !execution) {
    std::println(stderr, "no CUDA device");
    return 1;
  }
  const auto stream = (*execution)->CreateStream();
  const std::uint64_t base = DeviceBuffer(**memory, 4 * kWidth * sizeof(float));
  if (!stream || base == 0) {
    std::println(stderr, "no stream or device memory");
    return 1;
  }
  // Contents do not change host time; zeros keep the norm finite.
  void* operands = reinterpret_cast<void*>(base);  // NOLINT(performance-no-int-to-ptr)
  if (cudaMemset(operands, 0, 4 * kWidth * sizeof(float)) != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::println(stderr, "cannot clear the operands");
    return 1;
  }

  auto arena = TensorArena::Create(8).value();
  ggml_context* context = arena.context();
  ggml_tensor* x = ggml_new_tensor_1d(context, GGML_TYPE_F32, kWidth);
  ggml_tensor* w = ggml_new_tensor_1d(context, GGML_TYPE_F32, kWidth);
  TensorArena::Bind(x, base);
  TensorArena::Bind(w, base + (kWidth * sizeof(float)));
  ggml_tensor* norm = ggml_rms_norm(context, x, 1e-6f);
  ggml_tensor* scaled = ggml_mul(context, norm, w);
  TensorArena::Bind(norm, base + (2 * kWidth * sizeof(float)));
  TensorArena::Bind(scaled, base + (3 * kWidth * sizeof(float)));

  auto launch = LaunchContext::Create(0, **execution, *stream, {.base = 0, .size = Bytes(0)});
  const auto registry = llmp::execution::Registry::Create(llmp::kernels::ggml::Implementations());
  if (!launch || !registry) {
    std::println(stderr, "cannot create the launch context or registry");
    return 1;
  }
  std::array<std::optional<RmsNormMulKernel>, 2> kernels;  // unfused, fused
  for (const bool fused : {false, true}) {
    const std::vector<llmp::execution::Choice> choices = {
        {.operation = llmp::execution::Operation::kRmsNormMul,
         .implementation = fused ? "ggml.rms_norm_mul.fused" : "ggml.rms_norm_mul.unfused"}};
    const auto plan = llmp::execution::Plan::Build(*registry, choices);
    if (!plan) {
      std::println(stderr, "no plan: {}", plan.error().detail);
      return 1;
    }
    const auto bound = llmp::execution::Resolve(*registry, *plan);
    if (!bound) {
      std::println(stderr, "the plan does not bind: {}", bound.error().detail);
      return 1;
    }
    auto kernel = RmsNormMulKernel::Bind(bound->at(0));
    if (!kernel) {
      std::println(stderr, "no kernel: {}", kernel.error().detail);
      return 1;
    }
    kernels[fused ? 1 : 0] = *kernel;
  }

  // One batch of `per_batch` operations through one arm; the time per
  // operation in ns, or a negative value on failure.
  const auto batch = [&](bool fused, Layer layer) -> double {
    bool ok = true;
    const auto launchers = [fused, norm, scaled](ggml_backend_cuda_context& ctx) {
      if (fused) {
        ggml_cuda_op_rms_norm_fused(ctx, norm, scaled);
      } else {
        ggml_cuda_op_rms_norm(ctx, norm);
        ggml_cuda_op_mul(ctx, scaled);
      }
    };
    const Clock::time_point start = Clock::now();
    switch (layer) {
      case Layer::kDirect:
        ok = (*launch)
                 ->Run(Bytes(0),
                       [&](ggml_backend_cuda_context& ctx) {
                         for (int i = 0; i < per_batch; ++i) {
                           launchers(ctx);
                         }
                       })
                 .has_value();
        break;
      case Layer::kRun:
        for (int i = 0; i < per_batch && ok; ++i) {
          ok = (*launch)->Run(Bytes(0), launchers).has_value();
        }
        break;
      case Layer::kChecked:
        for (int i = 0; i < per_batch && ok; ++i) {
          ok = (fused ? llmp::kernels::ggml::RmsNormMul(**launch, norm, scaled)
                      : llmp::kernels::ggml::RmsNormThenMul(**launch, norm, scaled))
                   .has_value();
        }
        break;
      case Layer::kPlan:
        for (int i = 0; i < per_batch && ok; ++i) {
          ok = kernels[fused ? 1 : 0]->Run(**launch, norm, scaled).has_value();
        }
        break;
    }
    const auto elapsed = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
    if (!ok || !Drain(**execution, *stream)) {
      return -1.0;
    }
    return elapsed / per_batch;
  };

  std::vector<Arm> arms;
  for (const bool fused : {true, false}) {
    for (const Layer layer : kLayers) {
      arms.push_back({.fused = fused, .layer = layer, .ns_per_operation = {}});
    }
  }
  for (int round = 0; round < kWarmupBatches + batches; ++round) {
    for (Arm& arm : arms) {
      const double ns = batch(arm.fused, arm.layer);
      if (ns < 0) {
        std::println(stderr, "a launch failed");
        return 1;
      }
      if (round >= kWarmupBatches) {
        arm.ns_per_operation.push_back(ns);
      }
    }
  }

  int driver = 0;
  (void)cudaDriverGetVersion(&driver);
  std::println("driver API {}, {} batches of {} operations per arm after {} warm-up batches",
               driver, batches, per_batch, kWarmupBatches);
  std::println("");
  std::println("| Implementation | Kernels | Layer | p10 µs | p50 µs | p90 µs |");
  std::println("| --- | ---: | --- | ---: | ---: | ---: |");
  for (const Arm& arm : arms) {
    std::println("| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} |", arm.fused ? "fused" : "unfused",
                 arm.fused ? 1 : 2, LayerName(arm.layer),
                 Percentile(arm.ns_per_operation, 0.10) / 1000,
                 Percentile(arm.ns_per_operation, 0.50) / 1000,
                 Percentile(arm.ns_per_operation, 0.90) / 1000);
  }
  launch->reset();
  return (*execution)->DestroyStream(*stream) ? 0 : 1;
}
