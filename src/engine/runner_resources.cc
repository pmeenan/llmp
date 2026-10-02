// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/runner_resources.h"

#include <algorithm>
#include <format>
#include <utility>

#include "engine/support.h"
#include "kernels/ggml/implementations.h"
#include "providers/device_runtime.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
using base::Bytes;
using support::Error;

}  // namespace

RunnerResources::Status RunnerResources::Map(Mapped& mapped, std::string name, std::uint64_t bytes,
                                             catalog::MemoryClass memory_class) {
  // Retain even a partially mapped attempt through fenced teardown.
  // MapResident can fail after reservation or some backing was acquired.
  mapped_.push_back(&mapped);
  if (auto r = node_.MapResident(mapped, std::move(name), bytes, providers::BackingKind::kDevice,
                                 memory_class, catalog::Recovery::kPinned, owner_);
      !r) {
    return r;
  }
  return {};
}

std::expected<void*, std::string> RunnerResources::Pinned(std::uint64_t bytes) {
  auto allocated = node_.Pinned(bytes, owner_, staging_);
  if (allocated) {
    pinned_bytes_ += std::max<std::uint64_t>(bytes, 256);
  }
  return allocated;
}

std::uint64_t RunnerResources::mapped_bytes() const {
  std::uint64_t bytes = 0;
  for (const Mapped* mapped : mapped_) {
    bytes += mapped->bytes;
  }
  return bytes;
}

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
      kg::CublasHandle::Create(0, node_.execution(), node_.stream(stream_),
                               {.base = cublas_workspace_.base, .size = Bytes(cublas_bytes_)});
  if (!cublas) {
    return Error(std::format("cuBLAS: {}", cublas.error().detail));
  }
  cublas_ = std::move(*cublas);
  return {};
}

std::expected<std::unique_ptr<kg::LaunchContext>, std::string> RunnerResources::MeasuringContext()
    const {
  auto measure = kg::LaunchContext::Create(0, node_.execution(), node_.stream(stream_),
                                           {.base = 0, .size = Bytes(0)}, cublas_.get());
  if (!measure) {
    return Error(measure.error().detail);
  }
  return std::move(*measure);
}

RunnerResources::Status RunnerResources::BindLaunch(std::uint64_t pool_bytes) {
  auto launch = kg::LaunchContext::Create(0, node_.execution(), node_.stream(stream_),
                                          {.base = node_.pool().base, .size = Bytes(pool_bytes)},
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

std::vector<catalog::ExtentId> RunnerResources::extents() const {
  std::vector<catalog::ExtentId> all;
  for (const Mapped* mapped : mapped_) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging_.begin(), staging_.end());
  return all;
}

void RunnerResources::Release(std::vector<std::string>& problems) {
  launch_.reset();
  cublas_.reset();
  for (Mapped* mapped : mapped_) {
    if (!ReleaseMapped(node_.memory(), *mapped)) {
      problems.push_back(std::format("{} could not be released", mapped->name));
    }
  }
  mapped_.clear();
}

}  // namespace jitllm::engine
