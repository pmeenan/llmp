// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/step_program.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "base/bytes.h"
#include "execution/registry.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/cublas.h"
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
  GgmlRecorded(const StepLaunch* owner, std::uint64_t generation, CapturedGraph graph)
      : Recorded(owner), graph_(std::move(graph)), generation_(generation) {}
  std::size_t nodes() const override { return graph_.nodes(); }
  double capture_seconds() const override { return graph_.capture_seconds(); }
  double instantiate_seconds() const override { return graph_.instantiate_seconds(); }
  const CapturedGraph& graph() const { return graph_; }
  std::uint64_t generation() const { return generation_; }

 private:
  CapturedGraph graph_;
  std::uint64_t generation_;
};

class GgmlStepProgram final : public StepProgram {
 public:
  GgmlStepProgram(const GgmlStepLaunch* const* launch, const std::uint64_t* generation,
                  const execution::Registry& registry, LoweredStep lowered, std::uint64_t pool,
                  std::shared_ptr<const kernels::BackendOwner> owner)
      : launch_(launch),
        generation_(generation),
        registry_(registry),
        lowered_(std::move(lowered)),
        pool_(pool),
        owner_(std::move(owner)) {
    for (const PlanStep& step : lowered_.plan.steps) {
      nodes_ += step.nodes.size();
      const auto selected =
          step.nodes.size() == 1
              ? std::ranges::find(lowered_.selected, step.nodes[0], &SelectedNode::tensor)
              : lowered_.selected.end();
      launches_ +=
          selected == lowered_.selected.end() ? step.nodes.size() : selected->call->launches();
    }
    for (const auto& selected : lowered_.selected) selected_host_ += selected.call->host_bytes();
  }

  std::uint64_t scratch_bytes() const override { return lowered_.extent; }
  std::uint64_t pool_bytes() const override { return pool_; }
  std::uint64_t host_bytes() const override {
    return lowered_.arena->bytes() + (nodes_ * kPlanNodeHostBytes) + selected_host_;
  }
  std::uint64_t launches() const override { return launches_; }
  std::span<const InputSlot> inputs() const override { return lowered_.inputs; }
  std::span<const OutputSlot> outputs() const override { return lowered_.outputs; }
  std::span<const TensorUse> tensors() const override { return lowered_.tensors; }
  const execution::BoundPlan* bound() const override { return bound_ ? &bound_->bound() : nullptr; }

  // As engine/planned.h BindPlanned: the pool scratch against the launch's
  // pools, then every implementation bound.
  std::expected<void, StepFailure> Bind(StepLaunch& launch) override {
    if (owner_) {
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    }
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
    auto bound = BoundGraph::Bind(registry_, plan, lowered_.selected, (*launch_)->selected(),
                                  {context.workspace().base, context.scratch_size(0).value()});
    if (!bound) return std::unexpected(Failure(bound.error()));
    bound_.emplace(std::move(*bound));
    bound_generation_ = *generation_;
    return {};
  }

  std::expected<void, StepFailure> Run(StepLaunch& launch,
                                       providers::NativeStream stream) const override {
    if (owner_) {
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    }
    if (!bound_ || &launch != *launch_ || bound_generation_ != *generation_) {
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
  const std::uint64_t* generation_;
  std::uint64_t bound_generation_ = 0;
  const execution::Registry& registry_;
  LoweredStep lowered_;
  std::uint64_t pool_ = 0;
  std::shared_ptr<const kernels::BackendOwner> owner_;
  std::uint64_t nodes_ = 0;
  std::uint64_t launches_ = 0;
  std::uint64_t selected_host_ = 0;
  std::optional<BoundGraph> bound_;
};

}  // namespace

std::expected<std::unique_ptr<Recorded>, StepFailure> GgmlStepLaunch::Capture(
    providers::NativeStream stream, const Body& body) {
  if ((active_ && *active_ != this) || (current_generation_ && *current_generation_ != generation_))
    return std::unexpected(StepFailure{.detail = "the declared launch was retired"});
  if (owner_) {
    if (auto ready = owner_->Ready(); !ready)
      return std::unexpected(StepFailure{.detail = ready.error()});
  }
  if (auto same = SameStream(*launch_, stream); !same) return std::unexpected(same.error());
  if (selected_ && !selected_->capturable())
    return std::unexpected(StepFailure{.detail = "selected native source is not capturable"});
  if (selected_ && selected_->faulted())
    return std::unexpected(StepFailure{.detail = "selected native source has faulted"});
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
  return std::make_unique<GgmlRecorded>(this, generation_, std::move(*captured));
}

std::expected<void, StepFailure> GgmlStepLaunch::Launch(providers::NativeStream stream,
                                                        const Recorded& recorded) {
  if ((active_ && *active_ != this) || (current_generation_ && *current_generation_ != generation_))
    return std::unexpected(StepFailure{.detail = "the declared launch was retired"});
  if (owner_) {
    if (auto ready = owner_->Ready(); !ready)
      return std::unexpected(StepFailure{.detail = ready.error()});
  }
  if (auto same = SameStream(*launch_, stream); !same) return same;
  if (selected_ && selected_->faulted())
    return std::unexpected(StepFailure{.detail = "selected native source has faulted"});
  if (recorded.owner() != this ||
      static_cast<const GgmlRecorded&>(recorded).generation() != generation_) {
    return std::unexpected(StepFailure{.detail = "another launch's recording"});
  }
  if (auto launched = launch_->Launch(static_cast<const GgmlRecorded&>(recorded).graph());
      !launched) {
    return std::unexpected(Failure(launched.error()));
  }
  return {};
}

// Configuration has stable storage and no callback-address identity. The
// backend/resources still obey the existing rule that they outlive borrowers.
struct GgmlBackend::Declaration final : kernels::BackendOwner {
  Declaration(GgmlBackend& backend_source, planner::OwnedDeviceFacts owned_facts,
              std::uint64_t workspace_bytes, std::function<bool(std::uint64_t)> reserve,
              std::function<void(std::uint64_t)> refund)
      : source(backend_source),
        facts(std::move(owned_facts)),
        workspace(workspace_bytes),
        charge(std::move(reserve)),
        release(std::move(refund)) {}
  ~Declaration() override = default;
  const kernels::StepBackend& backend() const override { return table; }
  bool Charge(std::uint64_t n) const override { return charge(n); }
  void Release(std::uint64_t n) const override { release(n); }
  std::expected<execution::BackendDeclaration, std::string> Ready() const override {
    if (!sealed || source.cublas_ == nullptr || source.measuring_->faulted() ||
        (source.launch_ && source.launch_->faulted()))
      return std::unexpected("the declared backend is not prepared or has faulted");
    return *sealed;
  }
  GgmlBackend& source;
  planner::OwnedDeviceFacts facts;
  kernels::StepBackend table{};
  std::uint64_t workspace;
  std::function<bool(std::uint64_t)> charge;
  std::function<void(std::uint64_t)> release;
  std::optional<execution::BackendDeclaration> sealed{};
};

std::expected<void, std::string> GgmlBackend::Declare(std::span<const execution::TunedShape> tuning,
                                                      std::uint64_t workspace,
                                                      std::function<bool(std::uint64_t)> charge,
                                                      std::function<void(std::uint64_t)> release) {
  if (declaration_ || launch_ || cublas_ || options_.selected || !charge || !release ||
      workspace == 0 || workspace % 256 != 0)
    return std::unexpected("the backend cannot declare this configuration");
  // Count/copy admission includes both the owned view and the adapter vector.
  // The input vector itself belongs to the caller and is already funded.
  // Refuse excessive input metadata before the temporary vector allocation.
  std::uint64_t temporary = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(tuning.size()),
                             sizeof(execution::TunedShape), &temporary) ||
      __builtin_add_overflow(temporary, std::uint64_t{512}, &temporary) || !charge(temporary))
    return std::unexpected("the tuning input metadata was not funded");
  // Guards precede the allocations they fund, so every refusal destroys the
  // copied facts before releasing capacity. The retained charge transfers to
  // the shared owner's deleter, which refunds only after complete destruction.
  struct Charge {
    const std::function<void(std::uint64_t)>& refund;
    std::uint64_t bytes;
    ~Charge() {
      if (bytes != 0) refund(bytes);
    }
  } temporary_charge{release, temporary};
  planner::DeviceFacts requested{
      .device_class = facts_.device_class, .arch = facts_.arch, .features = facts_.features};
  requested.tuned.assign(tuning.begin(), tuning.end());
  auto bound = planner::OwnedDeviceFacts::Bound(requested);
  if (!bound) return std::unexpected(bound.error());
  std::uint64_t bytes = 0;
  if (__builtin_add_overflow(*bound, sizeof(Declaration) + 512, &bytes) ||
      __builtin_add_overflow(bytes, temporary, &bytes) || !charge(bytes))
    return std::unexpected("the owned backend metadata was not funded");
  Charge retained_charge{release, bytes};
  auto owned = planner::OwnedDeviceFacts::Create(requested, *bound);
  if (!owned) return std::unexpected(owned.error());
  for (const auto& tune : owned->facts().tuned) {
    if (!registry_->Find(tune.implementation))
      return std::unexpected("the tuning names an implementation outside this backend");
  }
  auto declared = std::shared_ptr<Declaration>(
      new Declaration(*this, std::move(*owned), workspace, charge, release),
      [release, bytes](Declaration* value) {
        delete value;
        release(bytes);
      });
  retained_charge.bytes = 0;
  declared->table = PlainBackend();
  declared->table.device = declared->facts.facts();
  declaration_ = std::move(declared);
  return {};
}

std::expected<void, std::string> GgmlBackend::PrepareDeclaration() {
  if (!declaration_) return {};
  if (!cublas_ || cublas_->device() != device_ || cublas_->stream() != stream_ ||
      cublas_->workspace().size.value() != declaration_->workspace)
    return std::unexpected("the declared cuBLAS workspace/device/stream differs");
  auto actual = cublas_->ConfigurationIdentity();
  if (!actual) return std::unexpected(actual.error().detail);
  base::Sha256 recipe, build, selection;
  recipe.Update("llmp-ggml-declared-recipe-v1");
  build.Update("llmp-ggml-qualified-build-v1");
  selection.Update("llmp-ggml-declared-selection-v1");
  for (std::size_t i = 0; i < registry_->size(); ++i) {
    recipe.Update(std::as_bytes(std::span(registry_->identity(i))));
  }
  // The qualified registry declarations carry effective SDK/target/images,
  // source/module revisions and arithmetic build recipes. Server kept records
  // also retain the executable's existing build/stamp identity.
  for (std::size_t i = 0; i < registry_->size(); ++i) {
    const auto& implementation = registry_->at(i);
    for (const auto text :
         std::array<std::string_view, 2>{implementation.build, implementation.revision}) {
      build.Update(std::format("{}:", text.size()));
      build.Update(text);
    }
  }
  selection.Update(std::as_bytes(std::span(*actual)));
  selection.Update(std::format("{}:{}:{}:{}:{}:{}", facts_.arch, facts_.features,
                               options_.lower.device_masks, options_.lower.frozen_host_masks,
                               options_.lower.index_scratch_limit, declaration_->workspace));
  execution::BackendDeclaration value{
      .recipe = recipe.Finish(), .build = build.Finish(), .selection = selection.Finish()};
  if (declaration_->sealed && *declaration_->sealed != value)
    return std::unexpected("the backend configuration changed after it was sealed");
  declaration_->sealed = value;
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
  auto implementations = Implementations();
  if (options.selected) {
    if (options.lower.selected_source && options.lower.selected_source != options.selected.get())
      return std::unexpected("the selected lowering and owned source differ");
    backend->options_.lower.selected_source = options.selected.get();
    auto extra = options.selected->Implementations();
    for (auto& implementation : extra) implementations.push_back(std::move(implementation));
  } else if (options.lower.selected_source) {
    return std::unexpected("a backend must own its selected source");
  }
  auto registry = execution::Registry::Create(std::move(implementations));
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
  if (options.selected) backend->facts_.tuned = options.selected->Tuning(arch);
  return backend;
}

kernels::StepBackend GgmlBackend::Backend() {
  if (!declaration_) return PlainBackend();
  // The owner supplies the table by const reference. In particular, returning
  // its token does not allocate another uncharged tuning/name vector.
  return {.owner = declaration_};
}

kernels::StepBackend GgmlBackend::PlainBackend() {
  // A plan names a group's member but the lowering may resolve it to
  // another (PlanGraph's), so a member of a group with a cuBLAS member
  // stays on the stream too.
  return {.registry = registry_.get(),
          .device = facts_,
          .on_stream =
              [this](std::string_view implementation) {
                // Native adapters share one stream and one serial scratch pool.
                // Keep the whole mixed step ordered until lane support is proved.
                if (options_.selected) return true;
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
  CublasHandle* previous = cublas_;
  cublas_ = cublas;
  if (auto ready = PrepareDeclaration(); !ready) {
    cublas_ = previous;
    return ready;
  }
  return {};
}

void GgmlBackend::LaunchReleased() {
  launch_ = nullptr;
  // The caller has fenced and drops its handle next. Do not leave a required
  // declaration apparently ready through a borrowed pointer after teardown.
  if (declaration_) cublas_ = nullptr;
}

std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> GgmlBackend::Launch(
    LaunchContext::Workspace pool) {
  if (launch_ != nullptr) return std::unexpected("the backend's launch was made already");
  if (launch_generation_ == std::numeric_limits<std::uint64_t>::max())
    return std::unexpected("the backend launch generation is exhausted");
  std::unique_ptr<SelectedLaunch> selected;
  if (options_.selected) {
    const auto prefix = options_.selected->persistent_bytes();
    if (prefix % 256 != 0 || pool.base % 256 != 0 || prefix > pool.size.value() ||
        pool.base + pool.size.value() < pool.base)
      return std::unexpected("selected-source persistent workspace exceeds the pool");
    auto made = options_.selected->Launch(device_, execution_, stream_, {pool.base, prefix});
    if (!made) return std::unexpected(made.error().detail);
    selected = std::move(*made);
    if (!selected || selected->owner() != options_.selected.get())
      return std::unexpected("selected launch belongs to another source");
    pool = {.base = pool.base + prefix, .size = base::Bytes(pool.size.value() - prefix)};
  }
  auto context = LaunchContext::Create(device_, execution_, stream_, pool, cublas_);
  if (!context) return std::unexpected(context.error().detail);
  if (declaration_) {
    if (auto ready = declaration_->Ready(); !ready) return std::unexpected(ready.error());
  }
  auto launch =
      std::make_unique<GgmlStepLaunch>(std::move(*context), std::move(selected), declaration_,
                                       &launch_, &launch_generation_, launch_generation_ + 1);
  ++launch_generation_;
  launch_ = launch.get();
  return launch;
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> GgmlBackend::Lower(
    const kernels::LowerInput& input) const {
  if (declaration_) {
    if (auto ready = declaration_->Ready(); !ready) return std::unexpected(ready.error());
  }
  auto lowered = LowerStep(input, *registry_, DeviceChoicesOf(*measuring_), options_.lower);
  if (!lowered) return std::unexpected(lowered.error().detail);
  auto pool = PlanScratchOn(*measuring_, lowered->plan, false);
  if (!pool) return std::unexpected(pool.error().detail);
  for (const auto& selected : lowered->selected)
    *pool = std::max(*pool, selected.call->scratch_bytes());
  const std::uint64_t prefix = options_.selected ? options_.selected->persistent_bytes() : 0;
  if (__builtin_add_overflow(*pool, prefix, &*pool))
    return std::unexpected("selected-source workspace extent overflows");
  return std::make_unique<GgmlStepProgram>(&launch_, &launch_generation_, *registry_,
                                           std::move(*lowered), *pool, declaration_);
}

}  // namespace llmp::kernels::ggml
