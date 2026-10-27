// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runner_kernels.h"

#include <expected>
#include <span>
#include <utility>

#include "catalog/catalog.h"
#include "engine/device_backend.h"

namespace llmp::test_support {

namespace {

engine::Status Error(std::string message) { return std::unexpected(std::move(message)); }

}  // namespace

engine::Status RunnerKernels::Create(kernels::DeviceBackendOptions options) {
  if (backend_) return Error("the runner's backend was made already");
  auto made = engine::CreateDeviceBackend(node(), stream(), std::move(options));
  if (!made) return Error(made.error());
  backend_ = std::move(*made);
  return {};
}

engine::Status RunnerKernels::Bind(std::string_view label) {
  if (!backend_) return Error("the runner has no backend to bind");
  if (bound_ || !mapped_.empty()) return Error("the runner's backend was bound already");
  for (const auto& memory : engine::BackendKernelMemory(*backend_, label)) {
    auto& mapped = mapped_.emplace_back(std::make_unique<engine::Mapped>());
    if (auto r = Map(*mapped, memory.name, memory.bytes, catalog::MemoryClass::kRuntime); !r)
      return r;
    bases_.push_back(mapped->base);
  }
  // Released at Release even when it fails part way, as ModelRunner does.
  bound_ = true;
  if (auto bound = backend_->BindWorkspaces(bases_); !bound) return Error(bound.error());
  return {};
}

engine::Status RunnerKernels::Rebind() {
  if (!bound_) return Error("the runner's backend is not bound");
  backend_->ReleaseWorkspaces();
  if (auto bound = backend_->BindWorkspaces(bases_); !bound) return Error(bound.error());
  return {};
}

void RunnerKernels::Release(std::vector<std::string>& problems) {
  if (bound_) backend_->ReleaseWorkspaces();
  bound_ = false;
  backend_.reset();
  // The mappings' records stay at their addresses (one that could not be
  // released is kept).
  ReleaseMappings(problems);
}

void RunnerKernelConfig(engine::RunnerConfig& config, kernels::DeviceBackend& backend,
                        std::string_view label) {
  config.kernel_memory = engine::BackendKernelMemory(backend, label);
  config.prepare = [&backend](std::span<const std::uint64_t> bases) {
    return backend.BindWorkspaces(bases);
  };
  config.retire = [&backend] { backend.ReleaseWorkspaces(); };
}

}  // namespace llmp::test_support
