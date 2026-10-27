// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/common/fake_device_backend.h"

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "kernels/common/recording_launch.h"

namespace llmp::kernels {
namespace {

class FakeDeviceBackend final : public DeviceBackend {
 public:
  explicit FakeDeviceBackend(execution::Registry registry) : registry_(std::move(registry)) {
    facts_.device_class = "fake";
  }

  std::uint64_t library_workspace_bytes() const override { return kFakeLibraryWorkspaceBytes; }
  std::uint64_t persistent_bytes() const override { return 0; }
  std::uint64_t lane_pool_reserve() const override { return 0; }
  const execution::Registry& registry() const override { return registry_; }
  const planner::DeviceFacts& device() const override { return facts_; }
  StepBackend Backend() override {
    return {
        .registry = &registry_,
        .device = facts_,
        .lower = [](const LowerInput&) -> std::expected<std::unique_ptr<StepProgram>, std::string> {
          return std::unexpected("the fake backend lowers nothing");
        }};
  }
  std::expected<void, std::string> FundConstruction(std::function<bool(std::uint64_t)> charge,
                                                    std::function<void(std::uint64_t)>) override {
    if (!charge) return std::unexpected("construction needs a funder");
    return {};
  }
  std::expected<void, std::string> Declare(std::span<const execution::TunedShape> tuning,
                                           std::function<bool(std::uint64_t)>,
                                           std::function<void(std::uint64_t)>,
                                           DeclaredDefaults defaults) override {
    // It has no implementation a pin could name.
    if (!tuning.empty() || defaults.image_attention_head_dim != 0)
      return std::unexpected("the fake backend has no implementation to pin");
    return {};
  }
  std::expected<void, std::string> BindWorkspaces(std::span<const std::uint64_t> bases) override {
    if (bases.size() != 1 || bases[0] == 0)
      return std::unexpected("the kernel workspace is not mapped");
    bound_ = true;
    return {};
  }
  void ReleaseWorkspaces() override { bound_ = false; }
  std::expected<std::unique_ptr<StepLaunch>, std::string> Launch(providers::AddressRange,
                                                                 bool) override {
    if (!bound_) return std::unexpected("the kernel workspace is not bound");
    return std::make_unique<RecordingLaunch>();
  }

 private:
  execution::Registry registry_;
  planner::DeviceFacts facts_{};
  bool bound_ = false;
};

BackendCapabilities Capabilities() { return {}; }

std::expected<std::vector<std::shared_ptr<const SelectedSource>>, std::string> Sources(int) {
  return std::vector<std::shared_ptr<const SelectedSource>>{};
}

std::expected<std::unique_ptr<DeviceBackend>, std::string> Create(int, providers::DeviceExecution&,
                                                                  providers::StreamId,
                                                                  DeviceBackendOptions options) {
  if (options.selected || options.lower.selected_source)
    return std::unexpected("the fake backend owns no kernel source");
  auto registry = execution::Registry::Create({});
  if (!registry) return std::unexpected(registry.error().detail);
  return std::make_unique<FakeDeviceBackend>(std::move(*registry));
}

constexpr KernelSet kFake{.name = "the fake kernels",
                          .capabilities = &Capabilities,
                          .sources = &Sources,
                          .create = &Create};

}  // namespace

const KernelSet& FakeKernelSet() { return kFake; }

}  // namespace llmp::kernels
