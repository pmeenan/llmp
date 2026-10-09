// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML module as a kernel source of planned steps (D-107;
// kernels/common/lower.h, step_program.h), CUDA builds only: its registry
// (implementations.h, with capabilities.h's capabilities), the device's
// facts, and a lowering (lower.h) whose programs bind their plan against
// the registry (executor.h BoundGraph, with PlanScratch checked against the
// launch's pool), run it on the launch context's stream, and record it as
// the context's captured graph (launch.h, D-090).
//
// The step launch (GgmlStepLaunch) wraps the K-C launch context over the
// node's pool: a runner makes it at Bind (Launch), and a program binds only
// to the launch of its own backend. Building a program is host work (no
// launch, no device call): the backend's measuring context (a launch
// context without a pool) gives upstream's device choices and each plan's
// pool scratch, so a lookahead may build while the device runs.

#ifndef LLMP_KERNELS_GGML_STEP_PROGRAM_H_
#define LLMP_KERNELS_GGML_STEP_PROGRAM_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <string>

#include "execution/registry.h"
#include "kernels/common/lower.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/lower.h"
#include "providers/device_execution.h"

namespace llmp::kernels::ggml {

class CublasHandle;

// A step launch over a K-C launch context (launch.h): captures are its
// captured graphs, a fault its context's.
class GgmlStepLaunch final : public kernels::StepLaunch {
 public:
  explicit GgmlStepLaunch(std::unique_ptr<LaunchContext> launch) : launch_(std::move(launch)) {}
  LaunchContext& context() const { return *launch_; }

  std::expected<std::unique_ptr<Recorded>, StepFailure> Capture(providers::NativeStream stream,
                                                                const Body& body) override;
  std::expected<void, StepFailure> Launch(providers::NativeStream stream,
                                          const Recorded& recorded) override;
  bool faulted() const override { return launch_->faulted(); }

 private:
  std::unique_ptr<LaunchContext> launch_;
};

struct GgmlBackendOptions {
  LowerOptions lower{};
};

class GgmlBackend {
 public:
  // On `stream` of `execution`'s device `device`, lending `cublas` (null:
  // none; a plan needing it is refused) to its launches, which must
  // outlive the backend.
  static std::expected<std::unique_ptr<GgmlBackend>, std::string> Create(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      CublasHandle* cublas, GgmlBackendOptions options = {});
  GgmlBackend(const GgmlBackend&) = delete;
  GgmlBackend& operator=(const GgmlBackend&) = delete;
  GgmlBackend(GgmlBackend&&) = delete;
  GgmlBackend& operator=(GgmlBackend&&) = delete;
  ~GgmlBackend();

  const execution::Registry& registry() const { return *registry_; }
  // The registry, the device's facts, the cuBLAS products on the stream,
  // the prepared inputs' padded rows and the lowering, for a step source.
  // The backend outlives what it gives.
  kernels::StepBackend Backend();
  // The step launch over `pool` (the node's pool, mapped), made once at the
  // runner's Bind; a program of this backend binds only to it.
  std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> Launch(
      LaunchContext::Workspace pool);

  // A planned step as a program (Backend().lower).
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Lower(
      const kernels::LowerInput& input) const;

 private:
  GgmlBackend(int device, providers::DeviceExecution& execution, providers::StreamId stream,
              CublasHandle* cublas, GgmlBackendOptions options);

  int device_;
  providers::DeviceExecution& execution_;
  providers::StreamId stream_;
  CublasHandle* cublas_;
  GgmlBackendOptions options_;
  std::unique_ptr<execution::Registry> registry_;
  std::unique_ptr<LaunchContext> measuring_;
  planner::DeviceFacts facts_{};
  const GgmlStepLaunch* launch_ = nullptr;  // the one Launch made
};

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_STEP_PROGRAM_H_
