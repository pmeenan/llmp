// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_COMMON_DEVICE_BACKEND_H_
#define LLMP_KERNELS_COMMON_DEVICE_BACKEND_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "execution/registry.h"
#include "kernels/common/lower.h"
#include "kernels/common/selected_operation.h"
#include "kernels/common/step_program.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace llmp::providers {
using DeviceSlice = AddressRange;
}  // namespace llmp::providers

namespace llmp::kernels {

class StreamLibrary {
 public:
  virtual ~StreamLibrary() = default;
};

std::expected<std::uint64_t, std::string> StreamLibraryWorkspaceBytes(int device);
std::expected<std::unique_ptr<StreamLibrary>, std::string> CreateStreamLibrary(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    providers::DeviceSlice workspace);

struct DeviceBackendOptions {
  LowerOptions lower{};
  std::shared_ptr<const SelectedSource> selected{};
  LaneConfiguration lanes{};
};

class DeviceBackend {
 public:
  static BackendCapabilities Capabilities();
  static std::expected<std::unique_ptr<DeviceBackend>, std::string> Create(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      DeviceBackendOptions options = {});
  virtual ~DeviceBackend() = default;

  virtual std::uint64_t library_workspace_bytes() const = 0;
  virtual std::uint64_t persistent_bytes() const = 0;
  virtual std::uint64_t lane_pool_reserve() const = 0;
  virtual std::uint32_t arch() const = 0;
  virtual const execution::Registry& registry() const = 0;
  virtual StepBackend Backend() = 0;
  virtual std::vector<execution::TunedShape> DefaultImageAttentionTuning(
      std::span<const std::uint64_t> rows, std::uint64_t head_dim) const = 0;
  virtual std::expected<void, std::string> FundConstruction(
      std::function<bool(std::uint64_t)> charge, std::function<void(std::uint64_t)> release) = 0;
  virtual std::expected<void, std::string> Declare(std::span<const execution::TunedShape> tuning,
                                                   std::function<bool(std::uint64_t)> charge,
                                                   std::function<void(std::uint64_t)> release) = 0;
  virtual std::expected<void, std::string> BindWorkspaces(std::span<const std::uint64_t> bases) = 0;
  virtual void ReleaseWorkspaces() = 0;
  virtual std::expected<std::unique_ptr<StepLaunch>, std::string> Launch(
      providers::DeviceSlice pool, bool allow_lanes = true) = 0;
};

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_DEVICE_BACKEND_H_
