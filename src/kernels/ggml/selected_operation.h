// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Build-time kernel-source adapters inside the shared ordered step program.
// This seam owns no model graph and never chooses a replacement operation:
// the IR planner has already selected the exact implementation. GGML's
// private descriptors retain the global placement and every part dependency.
#ifndef LLMP_KERNELS_GGML_SELECTED_OPERATION_H_
#define LLMP_KERNELS_GGML_SELECTED_OPERATION_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "execution/registry.h"
#include "format/tensor_format.h"
#include "ggml.h"
#include "ir/graph.h"
#include "kernels/ggml/tensors.h"
#include "providers/device_execution.h"

namespace llmp::kernels::ggml {

struct SelectedPart {
  format::PartRole role = format::PartRole::kData;
  ggml_tensor* tensor = nullptr;
};
struct SelectedOperand {
  ir::TensorId id{};
  ir::TensorDesc desc{};
  ggml_tensor* tensor = nullptr;
  std::vector<SelectedPart> parts;
  // Typed evidence from direct, unmodified shared-step leaves. Row adapters
  // use both facts to tie device IDs to the runner's checked host slot map.
  std::optional<std::uint16_t> side_rows_component{};
  std::optional<ir::WeightRef> paged_weight{};
};
struct SelectedWorkspace {
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;
};
class SelectedSource;
class SelectedCall;

// Launch resources outlive every bound call and recording, and are released
// only after the runner proves retirement. Construction must queue no work;
// initialization, if needed, is part of the first Run on the owned stream.
class SelectedLaunch {
 public:
  explicit SelectedLaunch(const SelectedSource* owner, providers::NativeStream stream)
      : owner_(owner), stream_(stream) {}
  virtual ~SelectedLaunch() = default;
  const SelectedSource* owner() const { return owner_; }
  providers::NativeStream stream() const { return stream_; }
  virtual bool faulted() const = 0;
  // Explicit opt-in only when every call queues capture-safe work on the owned
  // stream, with stable borrowed buffers through completed replay. No allocation,
  // host-dependent kernel preparation or synchronization may occur in capture.
  // Captures retain the launch generation; a fault forbids capture and replay.
  virtual bool capturable() const { return false; }

 private:
  const SelectedSource* owner_;
  providers::NativeStream stream_;
};

// Kernel binding checks identity and every physical operand without queueing
// work. Run is ordered with the other selected steps on exactly this stream.
class BoundSelectedCall {
 public:
  virtual ~BoundSelectedCall() = default;
  virtual std::expected<void, KernelFailure> Run(providers::NativeStream stream) const = 0;
};
class SelectedCall {
 public:
  explicit SelectedCall(const SelectedSource* owner) : owner_(owner) {}
  virtual ~SelectedCall() = default;
  const SelectedSource* owner() const { return owner_; }
  virtual std::uint64_t scratch_bytes() const = 0;
  // Includes this descriptor and its eventual bound call, excluding GGML
  // descriptors which the shared arena already counts.
  virtual std::uint64_t host_bytes() const = 0;
  virtual std::uint64_t launches() const = 0;
  virtual std::expected<std::unique_ptr<BoundSelectedCall>, KernelFailure> Bind(
      const execution::Implementation&, SelectedLaunch&, SelectedWorkspace) const = 0;

 private:
  const SelectedSource* owner_;
};

// Optional kernel source lent to a GGML step backend. There is no dynamic
// plugin ABI; implementations and the source are linked into this build.
// Lower must reject an unsupported selected operation, never substitute one.
class SelectedSource {
 public:
  virtual ~SelectedSource() = default;
  virtual std::vector<execution::Implementation> Implementations() const = 0;
  virtual std::vector<execution::TunedShape> Tuning(std::uint32_t arch) const = 0;
  virtual bool Owns(std::string_view implementation) const = 0;
  // Persistent bytes (e.g. cooperative lock slots), separate from the pool
  // scratch shared serially with GGML. Both bounds are 256-byte aligned.
  virtual std::uint64_t persistent_bytes() const = 0;
  virtual std::expected<std::unique_ptr<SelectedCall>, KernelFailure> Lower(
      const execution::Implementation&, const ir::Node&, std::span<const SelectedOperand>,
      std::span<const SelectedOperand> outputs) const = 0;
  std::expected<std::unique_ptr<SelectedCall>, KernelFailure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const SelectedOperand> inputs, const SelectedOperand& output) const {
    return Lower(implementation, node, inputs, std::span(&output, 1));
  }
  virtual std::expected<std::unique_ptr<SelectedLaunch>, KernelFailure> Launch(
      int device, providers::DeviceExecution&, providers::StreamId,
      SelectedWorkspace persistent) const = 0;
};

// A selected call's marker is an ordinary dense output with all physical
// inputs attached. It is never submitted to a third-party graph dispatcher.
struct SelectedNode {
  ggml_tensor* tensor = nullptr;
  const execution::Implementation* implementation = nullptr;
  std::unique_ptr<SelectedCall> call;
};

}  // namespace llmp::kernels::ggml
#endif  // LLMP_KERNELS_GGML_SELECTED_OPERATION_H_
