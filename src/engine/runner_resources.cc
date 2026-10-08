// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/runner_resources.h"

#include <format>
#include <utility>

#include "engine/support.h"
#include "kernels/ggml/implementations.h"
#include "providers/device_runtime.h"

namespace llmp::engine {

namespace {

namespace kg = llmp::kernels::ggml;
using base::Bytes;
using support::Error;

}  // namespace

RunnerResources::Status RunnerResources::OpenCublas(std::string name) {
  const providers::DeviceFacts facts =
      providers::QueryDeviceFacts(0).value_or(providers::DeviceFacts{});
  cublas_bytes_ = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts.architecture)).value();
  if (auto r =
          Map(cublas_workspace_, std::move(name), cublas_bytes_, catalog::MemoryClass::kRuntime);
      !r) {
    return r;
  }
  auto cublas =
      kg::CublasHandle::Create(0, node().execution(), node().stream(stream()),
                               {.base = cublas_workspace_.base, .size = Bytes(cublas_bytes_)});
  if (!cublas) {
    return Error(std::format("cuBLAS: {}", cublas.error().detail));
  }
  cublas_ = std::move(*cublas);
  return {};
}

std::expected<std::unique_ptr<kg::LaunchContext>, std::string> RunnerResources::MeasuringContext()
    const {
  auto measure = kg::LaunchContext::Create(0, node().execution(), node().stream(stream()),
                                           {.base = 0, .size = Bytes(0)}, cublas_.get());
  if (!measure) {
    return Error(measure.error().detail);
  }
  return std::move(*measure);
}

RunnerResources::Status RunnerResources::BindLaunch(std::uint64_t pool_bytes) {
  auto launch = kg::LaunchContext::Create(0, node().execution(), node().stream(stream()),
                                          {.base = node().pool().base, .size = Bytes(pool_bytes)},
                                          cublas_.get());
  if (!launch) {
    return Error(launch.error().detail);
  }
  launch_ = std::move(*launch);
  auto registry = execution::Registry::Create(kg::Implementations());
  if (!registry) {
    return Error(registry.error().detail);
  }
  registry_ = std::make_unique<execution::Registry>(std::move(*registry));
  return {};
}

void RunnerResources::Release(std::vector<std::string>& problems) {
  launch_.reset();
  cublas_.reset();
  ReleaseMappings(problems);
}

}  // namespace llmp::engine
