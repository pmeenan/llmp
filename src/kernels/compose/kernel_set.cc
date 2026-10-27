// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/compose/kernel_set.h"

#include <expected>
#include <memory>
#include <string>
#include <vector>

#include "kernels/common/device_backend.h"
#include "kernels/decode/selected_source.h"
#include "kernels/exl3/selected_source.h"
#include "kernels/ggml/step_program.h"

namespace llmp::kernels {
namespace {

// The decoders over resident weights, then EXL3's sources for the device:
// lent to every backend after the caller's own, in this order.
std::expected<std::vector<std::shared_ptr<const SelectedSource>>, std::string> BuildSources(
    int device) {
  auto decoders = decode::DecodeResidentSelectedSource();
  if (!decoders) return std::unexpected(decoders.error());
  auto exl3 = exl3::DeviceSelectedSource(device);
  if (!exl3) return std::unexpected(exl3.error());
  return std::vector<std::shared_ptr<const SelectedSource>>{std::move(*decoders), std::move(*exl3)};
}

BackendCapabilities Capabilities() { return ggml::GgmlBackend::Capabilities(); }

constexpr KernelSet kBuild{.name = "the CUDA build's kernels",
                           .capabilities = &Capabilities,
                           .sources = &BuildSources,
                           .create = &ggml::CreateDeviceBackend};

}  // namespace

std::expected<void, std::string> RegisterBuildKernels() { return RegisterKernelSet(kBuild); }

}  // namespace llmp::kernels
