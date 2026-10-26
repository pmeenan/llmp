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

#include <concepts>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "execution/registry.h"
#include "kernels/common/device_backend.h"
#include "kernels/common/lower.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/lower.h"
#include "planner/lanes.h"
#include "providers/device_execution.h"

namespace llmp::kernels::ggml {

class CublasHandle;

// A step launch over a K-C launch context (launch.h): captures are its
// captured graphs, a fault its context's.
class GgmlStepLaunch final : public kernels::StepLaunch {
 public:
  explicit GgmlStepLaunch(std::unique_ptr<LaunchContext> launch,
                          std::unique_ptr<SelectedLaunch> selected = {},
                          std::shared_ptr<const kernels::BackendOwner> owner = {},
                          const GgmlStepLaunch* const* active = nullptr,
                          const std::uint64_t* current_generation = nullptr,
                          std::uint64_t generation = 0, SelectedLaunch* borrowed_selected = nullptr,
                          bool private_eager = false)
      : launch_(std::move(launch)),
        selected_(std::move(selected)),
        owner_(std::move(owner)),
        active_(active),
        current_generation_(current_generation),
        generation_(generation),
        borrowed_selected_(borrowed_selected),
        private_eager_(private_eager) {}
  SelectedLaunch* selected() const { return selected_ ? selected_.get() : borrowed_selected_; }
  LaunchContext& context() const { return *launch_; }

  std::expected<std::unique_ptr<Recorded>, StepFailure> Capture(providers::NativeStream stream,
                                                                const Body& body) override;
  std::expected<void, StepFailure> Launch(providers::NativeStream stream,
                                          const Recorded& recorded) override;
  bool faulted() const override {
    return retired() || launch_->faulted() || (selected() && selected()->faulted());
  }

 private:
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<SelectedLaunch> selected_;
  std::shared_ptr<const kernels::BackendOwner> owner_;
  const GgmlStepLaunch* const* active_ = nullptr;
  const std::uint64_t* current_generation_ = nullptr;
  std::uint64_t generation_ = 0;
  SelectedLaunch* borrowed_selected_ = nullptr;  // retained by the private origin
  bool private_eager_ = false;
};

struct GgmlBackendOptions {
  LowerOptions lower{};
  // Build-time native operation source; owned for the whole backend lifetime.
  std::shared_ptr<const SelectedSource> selected{};
  // Current executor supports either the ordinary stream or all four lanes.
  kernels::LaneConfiguration lanes{};
};

class GgmlBackend {
 public:
  static constexpr kernels::BackendCapabilities Capabilities() {
    return {.owner_lanes = planner::kMaxLanes, .device_tokens = true, .indexed_masks = true};
  }
  // On `stream` of `execution`'s device `device`, lending `cublas` (null:
  // none; a plan needing it is refused) to its launches, which must
  // outlive the backend.
  static std::expected<std::unique_ptr<GgmlBackend>, std::string> Create(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      CublasHandle* cublas, GgmlBackendOptions options = {});
  template <std::same_as<kernels::StreamLibrary> Lib>
  static std::expected<std::unique_ptr<GgmlBackend>, std::string> Create(
      int device, providers::DeviceExecution& execution, providers::StreamId stream, Lib* library,
      GgmlBackendOptions options = {}) {
    return CreateFromLibrary(device, execution, stream, library, std::move(options));
  }
  GgmlBackend(const GgmlBackend&) = delete;
  GgmlBackend& operator=(const GgmlBackend&) = delete;
  GgmlBackend(GgmlBackend&&) = delete;
  GgmlBackend& operator=(GgmlBackend&&) = delete;
  ~GgmlBackend();

  // Required independently of Declare, before producing/lending a backend or
  // launch. Charge/refund retain the real enclosing allowance, never a raw
  // source borrower. Installation prepays the callback owner/control metadata;
  // each Lower separately admits the checked actual native inventory. The
  // incoming callback captures use their caller's prepaid control inventory;
  // installation moves them without allocating another callback payload.
  std::expected<void, std::string> FundConstruction(std::function<bool(std::uint64_t)> charge,
                                                    std::function<void(std::uint64_t)> release);

  // Opt-in immutable configuration for a required shared source. The caller
  // pays counted metadata before it is allocated; the backend retains its
  // charge through all source/program borrowers. No tuning is installed by
  // ordinary Create beyond the source's exact existing pins. Call before
  // plans/launches or lending the setup handle.
  std::expected<void, std::string> Declare(std::span<const execution::TunedShape> tuning,
                                           std::uint64_t cublas_workspace_bytes,
                                           std::function<bool(std::uint64_t)> charge,
                                           std::function<void(std::uint64_t)> release);

  const execution::Registry& registry() const { return *registry_; }
  const planner::DeviceFacts& device() const { return facts_; }
  // The registry, the device's facts, the cuBLAS products on the stream,
  // the prepared inputs' padded rows and the lowering, for a step source.
  // The backend outlives what it gives.
  kernels::StepBackend Backend();
  // Lends `cublas` (a handle on the backend's device and stream, which
  // must live until LaunchReleased) to the launch Launch makes; refused once it is
  // made. Its plans are built the same with or without one (only running a
  // cuBLAS product needs it).
  std::expected<void, std::string> LendCublas(CublasHandle* cublas);
  std::expected<void, std::string> LendLibrary(kernels::StreamLibrary* library);
  // Exact source-owned state requirement, separate from discardable scratch.
  std::uint64_t persistent_bytes() const {
    return options_.selected ? options_.selected->persistent_bytes() : 0;
  }
  // Carved from the launch pool, apart from the measured ordinary scratch.
  std::uint64_t lane_pool_reserve() const {
    return options_.lanes.count * options_.lanes.scratch_bytes_per_lane;
  }
  // The caller owns distinct, cataloged persistent backing for this backend,
  // mapped and charged until completed retirement destroys every launch/call.
  // It must survive scratch eviction/remapping; the node pool cannot supply it.
  std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> Launch(
      LaunchContext::Workspace pool, SelectedWorkspace persistent);
  // The runner destroyed the launch (its Release): nothing binds to it
  // after, and another may be made.
  // Returns the descriptor context's library loan too. Re-lend before reuse.
  void LaunchReleased();

  // A planned step as a program (Backend().lower).
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Lower(
      const kernels::LowerInput& input) const;

 private:
  static std::expected<std::unique_ptr<GgmlBackend>, std::string> CreateFromLibrary(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      kernels::StreamLibrary* library, GgmlBackendOptions options);
  GgmlBackend(int device, providers::DeviceExecution& execution, providers::StreamId stream,
              CublasHandle* cublas, GgmlBackendOptions options);

  friend class GgmlResources;
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> LowerFor(
      const kernels::LowerInput& input, const GgmlStepLaunch* const* launch,
      const std::uint64_t* generation, bool private_resources = false) const;
  std::shared_ptr<const ConstructionFunder> construction_funder_{};
  bool construction_started_ = false;
  struct Declaration;
  std::shared_ptr<Declaration> declaration_{};
  kernels::StepBackend PlainBackend();
  std::expected<void, std::string> PrepareDeclaration();

  int device_;
  providers::DeviceExecution& execution_;
  providers::StreamId stream_;
  CublasHandle* cublas_;
  GgmlBackendOptions options_;
  std::unique_ptr<execution::Registry> registry_;
  std::unique_ptr<LaunchContext> measuring_;
  planner::DeviceFacts facts_{};
  const GgmlStepLaunch* launch_ = nullptr;  // the one Launch made
  std::uint64_t launch_generation_ = 0;
};

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_STEP_PROGRAM_H_
