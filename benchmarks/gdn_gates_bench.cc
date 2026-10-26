// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Fixed 48-head, one-row old/new screen; see experiments/gdn-gates/README.md.
// Build this identical collector against each source arm. No timing in CTest.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <print>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "base/bytes.h"
#include "base/sha256.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {
namespace kg = llmp::kernels::ggml;
constexpr int kHeads = 48, kNodes = 128, kReplays = 128, kWarmups = 4, kSamples = 7;
[[noreturn]] void Fail(std::string_view message) {
  std::println(stderr, "gdn screen failed: {}", message);
  std::abort();  // Supervisor retains process ownership until all work retires.
}
void Check(cudaError_t error) {
  if (error != cudaSuccess) Fail(cudaGetErrorString(error));
}
template <class T, class E>
T Must(std::expected<T, E> result) {
  if (!result) Fail("provider or kernel operation failed");
  if constexpr (!std::is_void_v<T>) return std::move(*result);
}
void Run(bool validation_only) {
  auto execution = Must(llmp::providers::cuda::OpenDeviceExecution(0));
  const auto stream = Must(execution->CreateStream());
  const auto native = Must(execution->Submission(stream));
  const auto cuda_stream = static_cast<cudaStream_t>(native.handle);
  // Four 48-F32 inputs, one two-plane output; no scratch or library workspace.
  constexpr std::size_t kValues = 6 * kHeads;
  void* allocation = nullptr;
  Check(cudaMalloc(&allocation, kValues * sizeof(float)));
  std::array<float, kValues> initial{};
  for (std::size_t h = 0; h < kHeads; ++h) {
    initial[h] = (static_cast<float>(h) - 24.0F) * 0.25F;
    initial[kHeads + h] = (static_cast<float>(h) - 20.0F) * 0.125F;
    initial[2 * kHeads + h] = (static_cast<float>(h % 7) - 3.0F) * 0.1F;
    initial[3 * kHeads + h] = -0.125F * static_cast<float>(h % 5 + 1);
  }
  std::fill(initial.begin() + 4 * kHeads, initial.end(), -12345.0F);
  Check(cudaMemcpy(allocation, initial.data(), sizeof(initial), cudaMemcpyHostToDevice));
  Check(cudaDeviceSynchronize());
  auto arena = Must(kg::TensorArena::Create(8));
  std::array<ggml_tensor*, 4> inputs{};
  const auto base = reinterpret_cast<std::uintptr_t>(allocation);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, kHeads, 1);
    kg::TensorArena::Bind(inputs[i], base + i * kHeads * sizeof(float));
  }
  auto* gates = kg::GdnGates(arena.context(), inputs[0], inputs[1], inputs[2], inputs[3]);
  if (!gates) Fail("constructor refused fixed fixture");
  kg::TensorArena::Bind(gates, base + 4 * kHeads * sizeof(float));
  auto launch = Must(
      kg::LaunchContext::Create(0, *execution, stream, {.base = 0, .size = llmp::base::Bytes(0)}));
  const auto finish = [&] {
    const auto fence = Must(execution->Record(stream));
    Check(cudaStreamSynchronize(cuda_stream));
    if (Must(execution->Query(fence)) != llmp::providers::FenceState::kComplete)
      Fail("provider fence incomplete after stream synchronization");
    Must(execution->Release(fence));
  };
  const auto read = [&] {
    std::array<float, 2 * kHeads> values{};
    Check(cudaMemcpy(values.data(), gates->data, sizeof(values), cudaMemcpyDeviceToHost));
    return values;
  };
  Must(kg::RunGdnGates(*launch, gates));
  finish();
  const auto expected = read();
  for (std::size_t h = 0; h < kHeads; ++h) {
    const double x = static_cast<double>(initial[h] + initial[2 * kHeads + h]);
    const double gate = std::log1p(std::exp(x)) * initial[3 * kHeads + h];
    const double beta = 1.0 / (1.0 + std::exp(-static_cast<double>(initial[kHeads + h])));
    if (!std::isfinite(expected[h]) || !std::isfinite(expected[kHeads + h]) ||
        std::abs(expected[h] - gate) > 2e-5 || std::abs(expected[kHeads + h] - beta) > 2e-5)
      Fail("fixed fixture output mismatch");
  }
  llmp::base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(expected)));
  const auto registry = Must(llmp::execution::Registry::Create(kg::Implementations()));
  int driver = 0, runtime = 0;
  Check(cudaDriverGetVersion(&driver));
  Check(cudaRuntimeGetVersion(&runtime));
  cudaDeviceProp properties{};
  Check(cudaGetDeviceProperties(&properties, 0));
  std::println(
      "{{\"event\":\"identity\",\"heads\":{},\"rows\":1,\"nodes\":{},"
      "\"replays\":{},\"warmups\":{},\"samples\":{},\"device\":\"{}\","
      "\"driver\":{},\"runtime\":{},\"registry_digest\":\"{}\","
      "\"output_sha256\":\"{}\",\"fixture_bytes\":{},\"scratch_bytes\":0}}",
      kHeads, kNodes, kReplays, kWarmups, kSamples, properties.name, driver, runtime,
      llmp::base::ToHex(registry.Digest()), llmp::base::ToHex(hash.Finish()), sizeof(initial));
  {
    auto graph = Must(
        launch->Capture([&](kg::LaunchContext& captured) -> std::expected<void, kg::KernelFailure> {
          for (int i = 0; i < kNodes; ++i) {
            if (auto result = kg::RunGdnGates(captured, gates); !result) return result;
          }
          return {};
        }));
    if (graph.nodes() != kNodes) Fail("capture has the wrong paid kernel count");
    // Poison output before a replay proves the captured graph writes both planes.
    Check(cudaMemset(gates->data, 0x7b, sizeof(expected)));
    Check(cudaDeviceSynchronize());
    Must(launch->Launch(graph));
    finish();
    if (std::bit_cast<std::array<std::uint32_t, 2 * kHeads>>(read()) !=
        std::bit_cast<std::array<std::uint32_t, 2 * kHeads>>(expected))
      Fail("captured output differs from the checked direct launch");
    if (!validation_only) {
      cudaEvent_t start = nullptr, stop = nullptr;
      Check(cudaEventCreate(&start));
      Check(cudaEventCreate(&stop));
      for (int sample = -kWarmups; sample < kSamples; ++sample) {
        Check(cudaEventRecord(start, cuda_stream));
        for (int replay = 0; replay < kReplays; ++replay) Must(launch->Launch(graph));
        Check(cudaEventRecord(stop, cuda_stream));
        Check(cudaEventSynchronize(stop));
        finish();
        float milliseconds = 0;
        Check(cudaEventElapsedTime(&milliseconds, start, stop));
        if (!(milliseconds > 0) || !std::isfinite(milliseconds)) Fail("invalid timer result");
        if (std::bit_cast<std::array<std::uint32_t, 2 * kHeads>>(read()) !=
            std::bit_cast<std::array<std::uint32_t, 2 * kHeads>>(expected))
          Fail("timed output differs from the checked direct launch");
        std::println("{{\"event\":\"sample\",\"sample\":{},\"ns_per_kernel\":{}}}", sample,
                     static_cast<double>(milliseconds) * 1e6 / (kNodes * kReplays));
      }
      Check(cudaEventDestroy(start));
      Check(cudaEventDestroy(stop));
    }
    finish();
  }
  launch.reset();
  Must(execution->DestroyStream(stream));
  Check(cudaFree(allocation));
  std::println("{{\"event\":\"complete\",\"validation_only\":{}}}", validation_only);
}
}  // namespace
int main(int argc, char** argv) {
  const bool validation_only = argc == 2 && std::string_view(argv[1]) == "--validate-only";
  if (argc > 1 && !validation_only) return 2;
  Run(validation_only);
}
