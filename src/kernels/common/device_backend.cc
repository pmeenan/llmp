// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/common/device_backend.h"

#include <atomic>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kernels/common/composite_source.h"

namespace llmp::kernels {
namespace {

std::atomic<const KernelSet*> registered{nullptr};

}  // namespace

std::expected<void, std::string> RegisterKernelSet(const KernelSet& set) {
  if (set.capabilities == nullptr || set.sources == nullptr || set.create == nullptr)
    return std::unexpected(std::format("the kernel set {} is incomplete", set.name));
  const KernelSet* expected = nullptr;
  if (registered.compare_exchange_strong(expected, &set, std::memory_order_acq_rel) ||
      expected == &set)
    return {};
  return std::unexpected(std::format("the kernel set {} is registered already; {} is refused",
                                     expected->name, set.name));
}

const KernelSet* RegisteredKernelSet() { return registered.load(std::memory_order_acquire); }

std::expected<std::unique_ptr<DeviceBackend>, std::string> CreateDeviceBackend(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    DeviceBackendOptions options) {
  const auto* set = RegisteredKernelSet();
  if (set == nullptr) return std::unexpected("no kernel set is registered");
  return set->create(device, execution, stream, std::move(options));
}

BackendCapabilities DeviceBackendCapabilities() {
  const auto* set = RegisteredKernelSet();
  return set == nullptr ? BackendCapabilities{} : set->capabilities();
}

std::expected<ComposedSources, std::string> ComposeSelectedSources(
    int device, std::shared_ptr<const SelectedSource> own) {
  const auto* set = RegisteredKernelSet();
  if (set == nullptr) return std::unexpected("no kernel set is registered");
  ComposedSources composed;
  if (own) {
    for (const auto& implementation : own->Implementations())
      composed.preferred.push_back(implementation.name);
  }
  auto build = set->sources(device);
  if (!build) return std::unexpected(build.error());
  std::vector<std::shared_ptr<const SelectedSource>> sources;
  sources.reserve(build->size() + 1);
  if (own) sources.push_back(std::move(own));
  for (auto& source : *build) sources.push_back(std::move(source));
  composed.source = CompositeSelectedSource(std::move(sources));
  return composed;
}

}  // namespace llmp::kernels
