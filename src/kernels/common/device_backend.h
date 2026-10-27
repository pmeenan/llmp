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
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "kernels/common/lower.h"
#include "kernels/common/selected_operation.h"
#include "kernels/common/step_program.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace llmp::kernels {

struct DeviceBackendOptions {
  LowerOptions lower{};
  // The caller's own kernel source (its implementations preferred where they
  // apply); the backend owns it together with the build's (ComposeSelectedSources).
  std::shared_ptr<const SelectedSource> selected{};
  LaneConfiguration lanes{};
};

// Pins a backend declares of its own accord (Declare): with a nonzero
// head size, its attention implementation for an image stage's attention at
// that head size, as qualified by the caller's evidence.
struct DeclaredDefaults {
  std::uint64_t image_attention_head_dim = 0;
};

// A device backend: the step backend of one device and stream, with the
// kernel workspaces it binds and the launch it makes. Made only through the
// registered kernel set (CreateDeviceBackend).
class DeviceBackend {
 public:
  virtual ~DeviceBackend() = default;

  virtual std::uint64_t library_workspace_bytes() const = 0;
  virtual std::uint64_t persistent_bytes() const = 0;
  virtual std::uint64_t lane_pool_reserve() const = 0;
  virtual const execution::Registry& registry() const = 0;
  // Physical provider facts, readable before declaration/construction begins.
  virtual const planner::DeviceFacts& device() const = 0;
  virtual StepBackend Backend() = 0;
  virtual std::expected<void, std::string> FundConstruction(
      std::function<bool(std::uint64_t)> charge, std::function<void(std::uint64_t)> release) = 0;
  // Declares `tuning` (the caller's measured pins), plus the backend's own
  // pins `defaults` asks for, which name its implementations without the
  // caller learning their names.
  virtual std::expected<void, std::string> Declare(std::span<const execution::TunedShape> tuning,
                                                   std::function<bool(std::uint64_t)> charge,
                                                   std::function<void(std::uint64_t)> release,
                                                   DeclaredDefaults defaults = {}) = 0;
  virtual std::expected<void, std::string> BindWorkspaces(std::span<const std::uint64_t> bases) = 0;
  virtual void ReleaseWorkspaces() = 0;
  virtual std::expected<std::unique_ptr<StepLaunch>, std::string> Launch(
      providers::AddressRange pool, bool allow_lanes = true) = 0;
};

// What a build links to execute (D-053: implementations are linked, never
// loaded; D-110): its device backend's capabilities and factory, and the
// kernel sources every backend is lent beside the caller's own. A build's
// composition root registers one (kernels/compose for the CUDA build; host
// programs and tests the fake, fake_device_backend.h); the engine and the
// runtime reach a backend only through it and name no kernel library.
struct KernelSet {
  std::string_view name{};
  BackendCapabilities (*capabilities)() = nullptr;
  // The build's kernel sources for `device`, in the order they are composed.
  std::expected<std::vector<std::shared_ptr<const SelectedSource>>, std::string> (*sources)(
      int device) = nullptr;
  std::expected<std::unique_ptr<DeviceBackend>, std::string> (*create)(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      DeviceBackendOptions options) = nullptr;
};

// Registers the process's kernel set; `set` must outlive the process (a
// static). Registering the same set again is a no-op; another is refused.
std::expected<void, std::string> RegisterKernelSet(const KernelSet& set);
// The registered set, or null before one is registered.
const KernelSet* RegisteredKernelSet();

// The registered set's backend on `stream` of `execution`'s device `device`.
std::expected<std::unique_ptr<DeviceBackend>, std::string> CreateDeviceBackend(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    DeviceBackendOptions options = {});
// The registered backend's capabilities (none before one is registered).
BackendCapabilities DeviceBackendCapabilities();

// The kernel source a backend owns: the caller's own first, then the
// registered set's for `device`, as one composite (composite_source.h); and
// the implementations the planner prefers, the caller's own. Refused before
// a set is registered, so no backend silently lacks the build's sources.
struct ComposedSources {
  std::shared_ptr<const SelectedSource> source{};
  std::vector<std::string> preferred{};
};
std::expected<ComposedSources, std::string> ComposeSelectedSources(
    int device, std::shared_ptr<const SelectedSource> own);

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_DEVICE_BACKEND_H_
