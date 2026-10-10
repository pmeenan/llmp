// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/step_program.h"

#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "base/bytes.h"
#include "execution/registry.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::ggml {
namespace {

StepFailure Failure(const KernelFailure& f) {
  return {.error = f.error == KernelError::kUnknown ? StepError::kUnknown : StepError::kRejected,
          .detail = f.detail};
}

// The launch context queues on its own stream, never the one a caller
// names: a runner's job on another stream would not fence its work.
std::expected<void, StepFailure> SameStream(const LaunchContext& context,
                                            providers::NativeStream stream) {
  if (stream.handle != context.native_stream().handle) {
    return std::unexpected(
        StepFailure{.detail = "the step's stream is not its backend's (the runner's job stream)"});
  }
  return {};
}

// What a planned step holds on the host besides its arena (engine/planned.h
// kPlanNodeHostBytes: the plan's steps, the activations' places and the
// bound launches, an upper bound a node measured on GB10).
constexpr std::uint64_t kPlanNodeHostBytes = 512;

class GgmlRecorded final : public Recorded {
 public:
  GgmlRecorded(const StepLaunch* owner, CapturedGraph graph)
      : Recorded(owner), graph_(std::move(graph)) {}
  std::size_t nodes() const override { return graph_.nodes(); }
  double capture_seconds() const override { return graph_.capture_seconds(); }
  double instantiate_seconds() const override { return graph_.instantiate_seconds(); }
  const CapturedGraph& graph() const { return graph_; }

 private:
  CapturedGraph graph_;
};

class GgmlStepProgram final : public StepProgram {
 public:
  GgmlStepProgram(const GgmlStepLaunch* const* launch, const execution::Registry& registry,
                  LoweredStep lowered, std::uint64_t pool)
      : launch_(launch), registry_(registry), lowered_(std::move(lowered)), pool_(pool) {
    for (const PlanStep& step : lowered_.plan.steps) nodes_ += step.nodes.size();
  }

  std::uint64_t scratch_bytes() const override { return lowered_.extent; }
  std::uint64_t pool_bytes() const override { return pool_; }
  std::uint64_t host_bytes() const override {
    return lowered_.arena->bytes() + (nodes_ * kPlanNodeHostBytes);
  }
  std::uint64_t launches() const override { return nodes_; }
  std::span<const InputSlot> inputs() const override { return lowered_.inputs; }
  std::span<const OutputSlot> outputs() const override { return lowered_.outputs; }
  std::span<const TensorUse> tensors() const override { return lowered_.tensors; }
  const execution::BoundPlan* bound() const override { return bound_ ? &bound_->bound() : nullptr; }

  // As engine/planned.h BindPlanned: the pool scratch against the launch's
  // pools, then every implementation bound.
  std::expected<void, StepFailure> Bind(StepLaunch& launch) override {
    if (lowered_.measuring) {
      return std::unexpected(StepFailure{.detail = "a measurement does not bind"});
    }
    if (*launch_ == nullptr || &launch != *launch_) {
      return std::unexpected(StepFailure{.detail = "a GGML step binds to its backend's launch"});
    }
    LaunchContext& context = (*launch_)->context();
    GraphPlan& plan = lowered_.plan;
    if (context.lanes() < kMaxLanes && !plan.regions.empty()) {
      // A context without lanes runs everything on its stream; the
      // placement made for lanes only lives longer.
      plan.regions.clear();
      for (PlanStep& step : plan.steps) step.lane = 0;
    }
    auto scratch = PlanScratch(context, plan);
    if (!scratch) return std::unexpected(Failure(scratch.error()));
    if (*scratch > context.scratch_size(0).value()) {
      return std::unexpected(StepFailure{
          .detail = std::format("the step's scratch ({} bytes) exceeds the pool ({} bytes)",
                                *scratch, context.scratch_size(0).value())});
    }
    auto lane = PlanLaneScratch(context, plan);
    if (!lane) return std::unexpected(Failure(lane.error()));
    if (*lane > context.scratch_size(1).value()) {
      return std::unexpected(
          StepFailure{.detail = std::format(
                          "the step's lane scratch ({} bytes) exceeds a lane's pool", *lane)});
    }
    auto bound = BoundGraph::Bind(registry_, plan);
    if (!bound) return std::unexpected(Failure(bound.error()));
    bound_.emplace(std::move(*bound));
    return {};
  }

  std::expected<void, StepFailure> Run(StepLaunch& launch,
                                       providers::NativeStream stream) const override {
    if (!bound_ || &launch != *launch_) {
      return std::unexpected(StepFailure{.detail = "the step is not bound to this launch"});
    }
    if (auto same = SameStream((*launch_)->context(), stream); !same) return same;
    if (auto ran = bound_->Run((*launch_)->context()); !ran) {
      return std::unexpected(Failure(ran.error()));
    }
    return {};
  }

 private:
  const GgmlStepLaunch* const* launch_;
  const execution::Registry& registry_;
  LoweredStep lowered_;
  std::uint64_t pool_ = 0;
  std::uint64_t nodes_ = 0;
  std::optional<BoundGraph> bound_;
};

}  // namespace

std::expected<std::unique_ptr<Recorded>, StepFailure> GgmlStepLaunch::Capture(
    providers::NativeStream stream, const Body& body) {
  if (auto same = SameStream(*launch_, stream); !same) return std::unexpected(same.error());
  auto captured = launch_->Capture([&](LaunchContext&) -> std::expected<void, KernelFailure> {
    auto recorded = body();
    if (!recorded) {
      return std::unexpected(KernelFailure{.error = recorded.error().error == StepError::kUnknown
                                                        ? KernelError::kUnknown
                                                        : KernelError::kRejected,
                                           .detail = recorded.error().detail});
    }
    return {};
  });
  if (!captured) return std::unexpected(Failure(captured.error()));
  return std::make_unique<GgmlRecorded>(this, std::move(*captured));
}

std::expected<void, StepFailure> GgmlStepLaunch::Launch(providers::NativeStream stream,
                                                        const Recorded& recorded) {
  if (auto same = SameStream(*launch_, stream); !same) return same;
  if (recorded.owner() != this) {
    return std::unexpected(StepFailure{.detail = "another launch's recording"});
  }
  if (auto launched = launch_->Launch(static_cast<const GgmlRecorded&>(recorded).graph());
      !launched) {
    return std::unexpected(Failure(launched.error()));
  }
  return {};
}

GgmlBackend::GgmlBackend(int device, providers::DeviceExecution& execution,
                         providers::StreamId stream, CublasHandle* cublas,
                         GgmlBackendOptions options)
    : device_(device), execution_(execution), stream_(stream), cublas_(cublas), options_(options) {}

GgmlBackend::~GgmlBackend() = default;

std::expected<std::unique_ptr<GgmlBackend>, std::string> GgmlBackend::Create(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    CublasHandle* cublas, GgmlBackendOptions options) {
  std::unique_ptr<GgmlBackend> backend(new GgmlBackend(device, execution, stream, cublas, options));
  auto registry = execution::Registry::Create(Implementations());
  if (!registry) return std::unexpected(registry.error().detail);
  backend->registry_ = std::make_unique<execution::Registry>(std::move(*registry));
  auto measuring =
      LaunchContext::Create(device, execution, stream, {.base = 0, .size = base::Bytes(0)}, cublas);
  if (!measuring) return std::unexpected(measuring.error().detail);
  backend->measuring_ = std::move(*measuring);
  const auto facts = providers::QueryDeviceFacts(device);
  if (!facts) return std::unexpected(std::format("the device's facts: {}", facts.error().text()));
  // The provider's architecture is GGML's compute capability (1210 on the
  // GB10); the planner's is CUDA's major · 10 + minor.
  const std::uint32_t arch = facts->architecture / 10;
  backend->facts_ = {.device_class = std::format("sm_{}", arch), .arch = arch};
  return backend;
}

kernels::StepBackend GgmlBackend::Backend() {
  // A plan names a group's member but the lowering may resolve it to
  // another (PlanGraph's), so a member of a group with a cuBLAS member
  // stays on the stream too.
  return {.registry = registry_.get(),
          .device = facts_,
          .on_stream =
              [](std::string_view implementation) {
                const execution::Capability* c = CapabilityOf(implementation);
                return UsesCublas(implementation) ||
                       (c != nullptr && c->group == CapabilityOf("ggml.mul_mat.cublas")->group);
              },
          .written = &WrittenBytes,
          .within = &WithinRoot,
          .lower = [this](const kernels::LowerInput& input) { return Lower(input); }};
}

std::expected<void, std::string> GgmlBackend::LendCublas(CublasHandle* cublas) {
  if (launch_ != nullptr) return std::unexpected("the backend's launch was made already");
  cublas_ = cublas;
  return {};
}

std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> GgmlBackend::Launch(
    LaunchContext::Workspace pool) {
  if (launch_ != nullptr) return std::unexpected("the backend's launch was made already");
  auto context = LaunchContext::Create(device_, execution_, stream_, pool, cublas_);
  if (!context) return std::unexpected(context.error().detail);
  auto launch = std::make_unique<GgmlStepLaunch>(std::move(*context));
  launch_ = launch.get();
  return launch;
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> GgmlBackend::Lower(
    const kernels::LowerInput& input) const {
  auto lowered = LowerStep(input, *registry_, DeviceChoicesOf(*measuring_), options_.lower);
  if (!lowered) return std::unexpected(lowered.error().detail);
  auto pool = PlanScratchOn(*measuring_, lowered->plan, false);
  if (!pool) return std::unexpected(pool.error().detail);
  return std::make_unique<GgmlStepProgram>(&launch_, *registry_, std::move(*lowered), *pool);
}

}  // namespace llmp::kernels::ggml
