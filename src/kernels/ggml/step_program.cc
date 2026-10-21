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
#include "base/check.h"
#include "base/json_stream.h"
#include "execution/measured_ranks.h"
#include "execution/registry.h"
#include "kernels/decode/selected_source.h"
#include "kernels/exl3/selected_source.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/plan_step.h"
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

// What a planned step holds on the host besides its arena (the plan's
// steps, the activations' places and the bound launches, an upper bound a
// node measured on GB10).
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

}  // namespace

class GgmlStepProgram final : public StepProgram, public RoutedProgram {
 public:
  GgmlStepProgram(const GgmlStepLaunch* const* launch, const std::uint64_t* generation,
                  const execution::Registry& registry, LoweredStep lowered, std::uint64_t pool,
                  std::uint64_t private_pool, kernels::LaneConfiguration lanes,
                  std::shared_ptr<const kernels::BackendOwner> owner)
      : launch_(launch),
        generation_(generation),
        registry_(registry),
        lowered_(std::move(lowered)),
        pool_(pool),
        private_pool_(private_pool),
        lanes_(lanes),
        owner_(std::move(owner)) {
    for (const PlanStep& step : lowered_.plan.steps) {
      if (const auto at = registry_.Find(step.implementation); at && registry_.at(*at).capability)
        capturable_ = capturable_ && registry_.at(*at).capability->capturable;
      nodes_ += step.nodes.size();
      const auto selected =
          step.nodes.size() == 1
              ? std::ranges::find(lowered_.selected, step.nodes[0], &SelectedNode::tensor)
              : lowered_.selected.end();
      if (step.implementation == kAttentionF32) {
        launches_ += AttentionF32Plan::kLaunchBound;
      } else if (step.implementation == kConv2dTiledF32) {
        // Lowering and physical admission have checked this descriptor already.
        // Each stripe is four actual calls, not one compound launch.
        const auto tile = CheckConv2dTiledF32(step.nodes.front());
        base::Check(tile.has_value(),
                    "an admitted tiled convolution retains its checked descriptor");
        launches_ += tile->launches;
      } else {
        launches_ +=
            selected == lowered_.selected.end() ? step.nodes.size() : selected->call->launches();
      }
    }
    selected_host_ += lowered_.physical_roots.capacity() * sizeof(PhysicalRoot);
    selected_host_ += lowered_.physical_tensors.capacity() * sizeof(PhysicalTensor);
    for (const auto& selected : lowered_.selected) selected_host_ += selected.call->host_bytes();
    if (lowered_.weight_reads) {
      selected_host_ += lowered_.weight_reads->capacity() * sizeof(RoutedWeight);
      for (const auto& w : *lowered_.weight_reads) {
        selected_host_ += w.parts.capacity() * sizeof(decltype(w.parts)::value_type);
        for (const auto& part : w.parts) selected_host_ += part.first.capacity() + 1;
      }
    }
    if (!lowered_.routing.empty()) {
      selected_host_ +=
          lowered_.routing.capacity() * sizeof(LoweredRouting) +
          2 * (lowered_.routing.size() + 1) * sizeof(std::unique_ptr<RoutedRecording>);
      for (const auto& r : lowered_.routing) {
        selected_host_ += r.members.capacity() * sizeof(std::size_t) +
                          r.layout.readback.capacity() * sizeof(RoutedCopy) +
                          r.uploads.capacity() * sizeof(RoutedCopy) +
                          r.descriptors.capacity() * sizeof(decltype(r.descriptors)::value_type);
        // Preparation, complete route readback, all worklist/descriptor uploads,
        // and two opaque segment owners are accounted in the graph plan charge.
        launches_ += 3 + r.uploads.size();
        for (const auto& copy : r.layout.readback) launches_ += copy.rows;
      }
    }
  }

  bool capturable() const override {
    if (!capturable_) return false;
    if (lowered_.selected.empty()) return true;
    if (!bound_ || *launch_ == nullptr || bound_generation_ != *generation_ ||
        !(*launch_)->selected())
      return false;
    return std::ranges::all_of(lowered_.selected, [&](const auto& selected) {
      return selected.call->capturable(*(*launch_)->selected());
    });
  }
  std::uint64_t scratch_bytes() const override { return lowered_.extent; }
  std::uint64_t pool_bytes() const override { return pool_; }
  std::uint64_t private_pool_bytes() const override { return private_pool_; }
  std::uint64_t host_bytes() const override {
    return lowered_.arena->bytes() + (nodes_ * kPlanNodeHostBytes) + selected_host_;
  }
  std::uint64_t launches() const override { return launches_; }
  kernels::PlanCoverage plan_coverage() const override {
    return {.logical_activation_bytes = lowered_.logical_extent,
            .physical_activation_bytes = lowered_.extent,
            .unmaterialized_logical_bytes = lowered_.unmaterialized_logical_bytes,
            .unmaterialized_roots = lowered_.unmaterialized_roots};
  }
  void StreamPlanReport(const kernels::PlanReportSink& sink) const override {
    StreamPhysicalPlanReport(lowered_, registry_, sink,
                             {.pool_bytes = pool_,
                              .lanes = lanes_.count,
                              .lane_scratch_bytes = lanes_.scratch_bytes_per_lane,
                              .launches = launches_});
  }
  std::span<const InputSlot> inputs() const override { return lowered_.inputs; }
  std::span<const OutputSlot> outputs() const override { return lowered_.outputs; }
  std::span<const TensorUse> tensors() const override { return lowered_.tensors; }
  const execution::BoundPlan* bound() const override { return bound_ ? &bound_->bound() : nullptr; }

  // The pool scratch against the launch's pools, then every implementation
  // bound.
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
    if (context.lanes() != lanes_.count ||
        (lanes_.count != 0 && context.scratch_size(1).value() != lanes_.scratch_bytes_per_lane))
      return std::unexpected(
          StepFailure{.detail = "the launch's lane resources differ from its plan"});
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
    auto bound = BoundGraph::BindImpl(registry_, plan, lowered_.selected, (*launch_)->selected(),
                                      {context.workspace().base, context.scratch_size(0).value()},
                                      !lowered_.routing.empty());
    if (!bound) return std::unexpected(Failure(bound.error()));
    bound_.emplace(std::move(*bound));
    bound_generation_ = *generation_;
    return {};
  }

  std::expected<void, StepFailure> Run(StepLaunch& launch,
                                       providers::NativeStream stream) const override {
    if (!lowered_.routing.empty())
      return std::unexpected(StepFailure{.detail = "routed program refuses whole-step Run"});
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

  bool warm() const override {
    return !lowered_.prologue.empty() ||
           std::ranges::any_of(lowered_.inputs, [](const InputSlot& s) { return s.invariant; });
  }
  std::expected<void, StepFailure> RunWarm(StepLaunch& launch,
                                           providers::NativeStream stream) const override {
    if (lowered_.prologue.empty()) return Run(launch, stream);
    if (!lowered_.routing.empty())
      return std::unexpected(StepFailure{.detail = "routed program refuses whole-step Run"});
    if (owner_) {
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    }
    if (!bound_ || &launch != *launch_ || bound_generation_ != *generation_) {
      return std::unexpected(StepFailure{.detail = "the step is not bound to this launch"});
    }
    if (auto same = SameStream((*launch_)->context(), stream); !same) return same;
    if (auto ran = bound_->RunSkipping((*launch_)->context(), lowered_.prologue); !ran) {
      return std::unexpected(Failure(ran.error()));
    }
    return {};
  }

  std::optional<std::span<const RoutedWeight>> weight_reads() const override {
    if (!lowered_.weight_reads) return {};
    return std::span<const RoutedWeight>(*lowered_.weight_reads);
  }
  RoutedProgram* routed() override { return !lowered_.routing.empty() ? this : nullptr; }
  const RoutedProgram* routed() const override {
    return !lowered_.routing.empty() ? this : nullptr;
  }
  std::span<const RoutedWeight> weights() const override { return *lowered_.weight_reads; }
  std::size_t groups() const override { return lowered_.routing.size(); }
  const RoutedLayout& layout(std::size_t group = 0) const override {
    return lowered_.routing.at(group).layout;
  }
  std::span<const std::size_t> members(std::size_t group) const override {
    return lowered_.routing.at(group).members;
  }
  std::uint64_t pinned_bytes() const override { return lowered_.routed_pinned_bytes; }
  const base::Sha256Digest& routing_identity() const override { return lowered_.routing_identity; }
  const base::Sha256Digest& graph_identity() const override {
    return lowered_.routing.front().graph_identity;
  }
  std::expected<std::unique_ptr<Recorded>, StepFailure> Record(
      StepLaunch& launch, providers::NativeStream stream,
      const StepLaunch::Body& body) const override {
    if (!capturable())
      return std::unexpected(StepFailure{.detail = "selected program is not capturable"});
    if (!lowered_.routing.empty())
      return std::unexpected(StepFailure{.detail = "routed program refuses whole-step Record"});
    return launch.Capture(stream, body);
  }
  std::expected<void, StepFailure> Current(StepLaunch& launch,
                                           providers::NativeStream stream) const {
    if (lowered_.routing.empty() || !bound_ || *launch_ == nullptr || &launch != *launch_ ||
        bound_generation_ != *generation_ || launch.faulted())
      return std::unexpected(
          StepFailure{.detail = "routed program has no current healthy binding"});
    if (owner_)
      if (auto ready = owner_->Ready(); !ready)
        return std::unexpected(StepFailure{.detail = ready.error()});
    return SameStream((*launch_)->context(), stream);
  }
  std::expected<void, StepFailure> Access(const RoutedAccess& access) {
    if (!access.Names(this, access.epoch()) || access.segment() > groups())
      return std::unexpected(StepFailure{.detail = "foreign routed segment access"});
    if (access.segment() == 0 && access.epoch() > progress_.epoch()) {
      if (!progress_.Start(access.epoch(), groups()))
        return std::unexpected(StepFailure{.detail = "previous routed segment remains unretired"});
      pinned_ = {};
    }
    if (!progress_.Ready(access.epoch(), access.segment()))
      return std::unexpected(StepFailure{.detail = "routed segment is out of order or unprepared"});
    return {};
  }
  std::expected<void, StepFailure> RunSegment(const RoutedAccess& access, StepLaunch& launch,
                                              providers::NativeStream stream) override {
    if (auto current = Current(launch, stream); !current) return current;
    if (auto valid = Access(access); !valid) return valid;
    const auto segment = access.segment();
    if (!recording_ && !progress_.Submit(access.epoch(), segment))
      return std::unexpected(StepFailure{.detail = "routed segment was already submitted"});
    if (segment != 0) {
      const auto& previous = lowered_.routing[segment - 1];
      if (pinned_.size() != previous.layout.host_bytes)
        return std::unexpected(StepFailure{.detail = "continuation lost its prepared staging"});
      for (const auto& copy : previous.uploads) {
        auto copied = providers::CopyAsync(stream, reinterpret_cast<void*>(copy.device),
                                           pinned_.data() + copy.host_offset, copy.bytes,
                                           providers::CopyKind::kHostToDevice);
        if (!copied.ok())
          return std::unexpected(
              StepFailure{.error = StepError::kUnknown, .detail = copied.text()});
      }
    }
    const auto begin = segment == 0 ? 0 : lowered_.routing[segment - 1].cut;
    const auto end =
        segment == groups() ? lowered_.plan.steps.size() : lowered_.routing[segment].cut;
    auto& context = (*launch_)->context();
    if (auto ran = bound_->RunRange(context, begin, end); !ran)
      return std::unexpected(Failure(ran.error()));
    if (segment < groups())
      if (auto route = RunMoeRoute(context, lowered_.routing[segment].preparation); !route)
        return std::unexpected(Failure(route.error()));
    return {};
  }
  std::expected<void, StepFailure> CompleteSegment(const RoutedAccess& access) override {
    if (!access.Names(this, progress_.epoch()) || recording_ ||
        !progress_.Complete(access.epoch(), access.segment()))
      return std::unexpected(StepFailure{.detail = "routed completion is stale or out of order"});
    pinned_ = {};
    return {};
  }
  std::expected<void, StepFailure> Prepare(const RoutedAccess& access,
                                           std::span<std::byte> pinned) override {
    pinned_ = {};
    const auto group = access.segment();
    if (group >= groups() || !access.Names(this, progress_.epoch()) ||
        !progress_.CanPrepare(access.epoch(), group) || pinned.size() != layout(group).host_bytes ||
        reinterpret_cast<std::uintptr_t>(pinned.data()) % 256)
      return std::unexpected(
          StepFailure{.detail = "routed preparation lost its completed epoch or staging"});
    const auto& r = layout(group);
    const auto E = r.shape.experts, P = r.shape.used * r.shape.tokens;
    const auto ints = [&](std::uint64_t offset, std::uint64_t count) {
      return std::span(reinterpret_cast<std::int32_t*>(pinned.data() + offset), count);
    };
    const auto valid =
        ValidateGroupedRouteInto(r.shape,
                                 {ints(r.ids, P), ints(r.offsets, E + 1), ints(r.position, P),
                                  ints(r.token, P), ints(r.expert, P)},
                                 {ints(r.work_offsets, E + 1),
                                  ints(r.work_tokens, P),
                                  ints(r.work_slots, P),
                                  {reinterpret_cast<std::uint8_t*>(pinned.data() + r.seen), P}});
    if (!valid)
      return std::unexpected(
          StepFailure{.detail = "actual router IDs or route bijection are invalid"});
    for (const auto& [index, offset] : lowered_.routing[group].descriptors) {
      const auto& call = *lowered_.selected[index].call;
      if (auto staged =
              call.StageDescriptors(pinned.subspan(offset, call.route_descriptor_bytes()));
          !staged)
        return std::unexpected(Failure(staged.error()));
    }
    pinned_ = pinned;
    if (!progress_.Prepared(access.epoch(), group))
      return std::unexpected(StepFailure{.detail = "routed preparation order changed"});
    return {};
  }
  class SegmentRecording final : public RoutedRecording {
   public:
    SegmentRecording(GgmlStepProgram& owner, std::uint64_t generation, std::size_t segment,
                     CapturedGraph graph)
        : owner_(owner), generation_(generation), segment_(segment), graph_(std::move(graph)) {}
    const RoutedProgram* owner() const override { return &owner_; }
    std::size_t segment() const override { return segment_; }
    std::size_t nodes() const override { return graph_.nodes(); }
    double capture_seconds() const override { return graph_.capture_seconds(); }
    double instantiate_seconds() const override { return graph_.instantiate_seconds(); }
    std::expected<void, StepFailure> Replay(const RoutedAccess& access, StepLaunch& launch,
                                            providers::NativeStream stream) const override {
      if (auto current = owner_.Current(launch, stream); !current) return current;
      if (owner_.bound_generation_ != generation_ || access.segment() != segment_)
        return std::unexpected(
            StepFailure{.detail = "routed recording binding or ordinal changed"});
      if (auto valid = owner_.Access(access); !valid) return valid;
      if (!owner_.progress_.Submit(access.epoch(), segment_))
        return std::unexpected(StepFailure{.detail = "routed recording already submitted"});
      if (auto ran = (*owner_.launch_)->context().Launch(graph_); !ran)
        return std::unexpected(Failure(ran.error()));
      return {};
    }

   private:
    GgmlStepProgram& owner_;
    std::uint64_t generation_;
    std::size_t segment_;
    CapturedGraph graph_;
  };
  std::expected<std::unique_ptr<RoutedRecording>, StepFailure> RecordSegment(
      const RoutedAccess& access, StepLaunch& launch, providers::NativeStream stream,
      const StepLaunch::Body& body) override {
    if (auto current = Current(launch, stream); !current) return std::unexpected(current.error());
    if (!capturable())
      return std::unexpected(StepFailure{.detail = "routed program is not capturable"});
    if (auto valid = Access(access); !valid) return std::unexpected(valid.error());
    if (recording_) return std::unexpected(StepFailure{.detail = "nested routed recording"});
    recording_ = true;
    auto captured =
        (*launch_)->context().Capture([&](LaunchContext&) -> std::expected<void, KernelFailure> {
          const auto result = body();
          if (!result)
            return std::unexpected(KernelFailure{
                .error = result.error().error == StepError::kUnknown ? KernelError::kUnknown
                                                                     : KernelError::kRejected,
                .detail = result.error().detail});
          return {};
        });
    recording_ = false;
    if (!captured) return std::unexpected(Failure(captured.error()));
    return std::make_unique<SegmentRecording>(*this, bound_generation_, access.segment(),
                                              std::move(*captured));
  }
  std::expected<void, StepFailure> ReplaySegment(const RoutedAccess& access, StepLaunch& launch,
                                                 providers::NativeStream stream,
                                                 const RoutedRecording& recorded) override {
    if (recorded.owner() != this || recorded.segment() != access.segment())
      return std::unexpected(StepFailure{.detail = "foreign routed recording or segment"});
    return recorded.Replay(access, launch, stream);
  }

 private:
  RoutedProgress progress_;
  bool recording_ = false;
  std::span<std::byte> pinned_{};
  const GgmlStepLaunch* const* launch_;
  const std::uint64_t* generation_;
  std::uint64_t bound_generation_ = 0;
  const execution::Registry& registry_;
  LoweredStep lowered_;
  std::uint64_t pool_ = 0, private_pool_ = 0;
  kernels::LaneConfiguration lanes_{};
  std::shared_ptr<const kernels::BackendOwner> owner_;
  std::uint64_t nodes_ = 0;
  std::uint64_t launches_ = 0;
  std::uint64_t selected_host_ = 0;
  bool capturable_ = true;
  std::optional<BoundGraph> bound_;
};

std::expected<std::unique_ptr<Recorded>, StepFailure> GgmlStepLaunch::Capture(
    providers::NativeStream stream, const Body& body) {
  if (private_eager_ || retired())
    return std::unexpected(StepFailure{.detail = "private or retired launch refuses capture"});
  if ((active_ && *active_ != this) || (current_generation_ && *current_generation_ != generation_))
    return std::unexpected(StepFailure{.detail = "the declared launch was retired"});
  if (owner_) {
    if (auto ready = owner_->Ready(); !ready)
      return std::unexpected(StepFailure{.detail = ready.error()});
  }
  if (auto same = SameStream(*launch_, stream); !same) return std::unexpected(same.error());
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
  if (private_eager_ || retired())
    return std::unexpected(StepFailure{.detail = "private or retired launch refuses replay"});
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
// Binding slots are owned by this private instance, never the global backend.
// The origin authenticates/retains the actual runner before borrowing anything.
class GgmlResources final : public kernels::ResourceInstance {
 public:
  GgmlResources(kernels::ResourceOrigin origin, const GgmlBackend& source,
                std::unique_ptr<LaunchContext> context, SelectedLaunch* selected)
      : ResourceInstance(std::move(origin)), source_(source) {
    launch_ = std::make_unique<GgmlStepLaunch>(std::move(context), nullptr, parent_owner(),
                                               &active_, &generation_, generation_, selected, true);
    active_ = launch_.get();
  }
  kernels::StepLaunch& launch() const override { return *launch_; }
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Lower(
      const kernels::LowerInput& input) const override {
    return source_.LowerFor(input, &active_, &generation_, true);
  }

 private:
  const GgmlBackend& source_;
  const std::uint64_t generation_ = 1;
  const GgmlStepLaunch* active_ = nullptr;
  std::unique_ptr<GgmlStepLaunch> launch_;
};

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
        (source.launch_ && source.launch_->faulted()) ||
        (source.options_.selected &&
         (source.options_.selected->ExecutionIdentity() != selected ||
          source.options_.selected->persistent_bytes() != selected_persistent)))
      return std::unexpected("the declared backend is not prepared or has faulted");
    return *sealed;
  }
  std::expected<std::uint64_t, kernels::ResourceFailure> ResourceBytes(
      kernels::ResourceWorkspace region) const override {
    auto host = LaunchContext::PrivateHostBytes({region.base, base::Bytes(region.bytes)});
    if (!host)
      return std::unexpected(
          kernels::ResourceFailure{kernels::ResourceError::kBounds, host.error().detail});
    constexpr std::uint64_t kAllocationMetadataBytes = 64;
    std::uint64_t bytes = 0;
    if (__builtin_add_overflow(*host,
                               std::uint64_t{sizeof(GgmlResources) + sizeof(GgmlStepLaunch) +
                                             2 * kAllocationMetadataBytes},
                               &bytes))
      return std::unexpected(kernels::ResourceFailure{kernels::ResourceError::kBounds,
                                                      "private backend metadata size overflows"});
    return bytes;
  }
  std::expected<kernels::ResourceInstance*, kernels::ResourceFailure> MakeResources(
      kernels::ResourceOrigin origin, kernels::ResourceWorkspace region) const override {
    if (!source.launch_ || origin.parent_launch() != source.launch_)
      return std::unexpected(kernels::ResourceFailure{
          kernels::ResourceError::kOwnership,
          "private resources do not retain this producer's actual parent launch"});
    auto context = LaunchContext::CreatePrivate(source.launch_->context(),
                                                {region.base, base::Bytes(region.bytes)});
    if (!context)
      return std::unexpected(kernels::ResourceFailure{context.error().error == KernelError::kUnknown
                                                          ? kernels::ResourceError::kFault
                                                          : kernels::ResourceError::kProvider,
                                                      context.error().detail});
    return new GgmlResources(std::move(origin), source, std::move(*context),
                             source.launch_->selected());
  }
  GgmlBackend& source;
  planner::OwnedDeviceFacts facts;
  kernels::StepBackend table{};
  std::uint64_t workspace;
  std::function<bool(std::uint64_t)> charge;
  std::function<void(std::uint64_t)> release;
  std::optional<execution::BackendDeclaration> sealed{};
  std::optional<base::Sha256Digest> selected{};
  std::uint64_t selected_persistent = 0;
};

std::expected<void, std::string> GgmlBackend::FundConstruction(
    std::function<bool(std::uint64_t)> charge, std::function<void(std::uint64_t)> release) {
  if (construction_funder_ || construction_started_ || launch_ || !charge || !release)
    return std::unexpected("the backend needs one complete construction funder before use");
  constexpr std::uint64_t bytes = sizeof(ConstructionFunder) + 2 * 64;
  if (!charge(bytes)) return std::unexpected("backend construction funder metadata was not funded");
  construction_funder_ = std::shared_ptr<const ConstructionFunder>(
      new ConstructionFunder{std::move(charge), std::move(release)}, [](ConstructionFunder* owner) {
        auto refund = std::move(owner->release);
        delete owner;
        refund(bytes);
      });
  // Declarations may precede construction funding (the image path). Export
  // the same paid token without changing any numerical/device declaration.
  if (declaration_) declaration_->table.construction_funder = construction_funder_;
  return {};
}

std::expected<void, std::string> GgmlBackend::Declare(std::span<const execution::TunedShape> tuning,
                                                      std::uint64_t workspace,
                                                      std::function<bool(std::uint64_t)> charge,
                                                      std::function<void(std::uint64_t)> release) {
  if (declaration_ || launch_ || cublas_ ||
      (options_.selected && !options_.selected->ExecutionIdentity()) || !charge || !release ||
      workspace == 0 || workspace % 256 != 0)
    return std::unexpected("the backend cannot declare this configuration");
  // Count/copy admission includes both the owned view and the adapter vector.
  // The input vector itself belongs to the caller and is already funded.
  // Refuse excessive input metadata before the temporary vector allocation.
  std::uint64_t temporary = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(tuning.size() + facts_.tuned.size()),
                             sizeof(execution::TunedShape), &temporary) ||
      __builtin_add_overflow(temporary, std::uint64_t{512}, &temporary))
    return std::unexpected("the tuning input metadata was not funded");
  for (const auto& name : facts_.preferred) {
    std::uint64_t bytes = sizeof(std::string) + 64;
    if (__builtin_add_overflow(bytes, static_cast<std::uint64_t>(name.size() + 1), &bytes) ||
        __builtin_add_overflow(temporary, bytes, &temporary))
      return std::unexpected("preferred source metadata overflows");
  }
  if (!charge(temporary)) return std::unexpected("the preferred/tuning metadata was not funded");
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
  planner::DeviceFacts requested{.device_class = facts_.device_class,
                                 .arch = facts_.arch,
                                 .features = facts_.features,
                                 .device_model = facts_.device_model,
                                 .multiprocessors = facts_.multiprocessors,
                                 .preferred = facts_.preferred};
  requested.tuned = facts_.tuned;
  for (const auto& fact : tuning)
    if (!std::ranges::contains(requested.tuned, fact)) requested.tuned.push_back(fact);
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
  declared->selected = options_.selected ? options_.selected->ExecutionIdentity() : std::nullopt;
  declared->selected_persistent = options_.selected ? options_.selected->persistent_bytes() : 0;
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
  selection.Update("llmp-ggml-declared-selection-v2");
  selection.Update(
      std::format("lanes:{}:{};", options_.lanes.count, options_.lanes.scratch_bytes_per_lane));
  selection.Update(std::as_bytes(std::span(execution::MeasuredRanksDigest())));
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
  if (options_.selected) {
    const auto identity = options_.selected->ExecutionIdentity();
    if (!identity || identity != declaration_->selected ||
        options_.selected->persistent_bytes() != declaration_->selected_persistent)
      return std::unexpected("selected adapter identity changed after declaration");
    recipe.Update(std::as_bytes(std::span(*identity)));
    selection.Update(std::as_bytes(std::span(*identity)));
    selection.Update(std::format("selected-persistent:{}", options_.selected->persistent_bytes()));
  }
  for (const auto& name : facts_.preferred) {
    selection.Update(std::format("preferred:{}:", name.size()));
    selection.Update(name);
  }
  const auto& effective = declaration_->facts.facts();
  for (const auto text :
       {std::string_view(effective.device_class), std::string_view(effective.device_model)}) {
    selection.Update(std::format("fact:{}:", text.size()));
    selection.Update(text);
  }
  selection.Update(std::format("SMs:{};", effective.multiprocessors));
  for (const auto& tune : effective.tuned) {
    selection.Update(std::format("tune:{}:", tune.implementation.size()));
    selection.Update(tune.implementation);
    selection.Update(std::format(
        ":{}:{}:{}:{}:{};", tune.rows, tune.k, tune.n,
        tune.output_encoding ? static_cast<int>(*tune.output_encoding) : -1, tune.weight_bits));
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
  const auto lanes = options.lanes;
  if ((lanes.count == 0 && lanes.scratch_bytes_per_lane != 0) ||
      (lanes.count != 0 && (lanes.count != kMaxLanes || lanes.scratch_bytes_per_lane == 0 ||
                            lanes.scratch_bytes_per_lane % 256 != 0 ||
                            lanes.scratch_bytes_per_lane > UINT64_MAX / lanes.count)))
    return std::unexpected("the backend needs zero lanes or four funded aligned lane pools");
  std::unique_ptr<GgmlBackend> backend(new GgmlBackend(device, execution, stream, cublas, options));
  if (options.lower.selected_source && options.lower.selected_source != options.selected.get())
    return std::unexpected("the selected lowering and owned source differ");
  std::vector<std::string> preferred;
  if (options.selected)
    for (const auto& implementation : options.selected->Implementations())
      preferred.push_back(implementation.name);
  auto decoders = decode::DecodeResidentSelectedSource();
  if (!decoders) return std::unexpected(decoders.error());
  auto exl3 = exl3::DeviceSelectedSource(device);
  if (!exl3) return std::unexpected(exl3.error());
  std::vector<std::shared_ptr<const SelectedSource>> sources;
  if (options.selected) sources.push_back(options.selected);
  sources.push_back(*decoders);
  sources.push_back(*exl3);
  options.selected = decode::CompositeSelectedSource(std::move(sources));
  options.lower.selected_source = options.selected.get();
  backend->options_ = options;
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
  std::uint32_t features = 0;
  if (facts->bf16_mma) features |= execution::kBf16Mma;
  if (facts->int8_mma) features |= execution::kInt8Mma;
  if (facts->fp8_mma) features |= execution::kFp8Mma;
  if (facts->fp4_mma) features |= execution::kFp4Mma;
  if (facts->block_scaled_mma) features |= execution::kBlockScaledMma;
  backend->facts_ = {.device_class = std::format("sm_{}", arch),
                     .arch = arch,
                     .features = features,
                     .device_model = facts->name.data(),
                     .multiprocessors = static_cast<std::uint32_t>(facts->multiprocessors)};
  if (options.selected) backend->facts_.tuned = options.selected->Tuning(arch);
  backend->facts_.preferred = std::move(preferred);
  return backend;
}

std::expected<std::unique_ptr<GgmlBackend>, std::string> GgmlBackend::CreateFromLibrary(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    kernels::StreamLibrary* library, GgmlBackendOptions options) {
  return Create(device, execution, stream, static_cast<CublasHandle*>(library), std::move(options));
}

kernels::StepBackend GgmlBackend::Backend() {
  construction_started_ = true;
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
          .lanes = options_.lanes,
          .on_stream =
              [this](std::string_view implementation) {
                // Selected native calls share the backend's stream and serial
                // scratch. Independent GGML calls retain declared lane support.
                if (options_.selected && options_.selected->Owns(implementation)) return true;
                const execution::Capability* c = CapabilityOf(implementation);
                return UsesCublas(implementation) ||
                       (c != nullptr && c->group == CapabilityOf("ggml.mul_mat.cublas")->group);
              },
          .written = &WrittenBytes,
          .within = &WithinRoot,
          .lower = [this](const kernels::LowerInput& input) { return Lower(input); },
          .construction_funder = construction_funder_};
}

std::expected<void, std::string> GgmlBackend::LendCublas(CublasHandle* cublas) {
  construction_started_ = true;
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
    LaunchContext::Workspace pool, SelectedWorkspace persistent) {
  construction_started_ = true;
  if (launch_ != nullptr) return std::unexpected("the backend's launch was made already");
  if (launch_generation_ == std::numeric_limits<std::uint64_t>::max())
    return std::unexpected("the backend launch generation is exhausted");
  std::unique_ptr<SelectedLaunch> selected;
  if (options_.selected) {
    const auto bytes = persistent_bytes();
    if (bytes % 256 != 0 || (bytes && (!persistent.base || persistent.base % 256 != 0)) ||
        bytes > persistent.bytes || persistent.base + persistent.bytes < persistent.base ||
        pool.base + pool.size.value() < pool.base ||
        (bytes && persistent.base < pool.base + pool.size.value() &&
         pool.base < persistent.base + bytes))
      return std::unexpected("selected-source state needs distinct owned persistent backing");
    auto made = options_.selected->Launch(device_, execution_, stream_, {persistent.base, bytes});
    if (!made) return std::unexpected(made.error().detail);
    selected = std::move(*made);
    if (!selected || selected->owner() != options_.selected.get())
      return std::unexpected("selected launch belongs to another source");
  }
  auto context = LaunchContext::Create(device_, execution_, stream_, pool, cublas_);
  if (!context) return std::unexpected(context.error().detail);
  if (options_.lanes.count != 0) {
    auto configured = (*context)->ConfigureLanes(
        options_.lanes.count, base::Bytes(options_.lanes.scratch_bytes_per_lane));
    if (!configured) return std::unexpected(configured.error().detail);
  }
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
  return LowerFor(input, &launch_, &launch_generation_);
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> GgmlBackend::LowerFor(
    const kernels::LowerInput& input, const GgmlStepLaunch* const* launch,
    const std::uint64_t* generation, bool /*private_resources*/) const {
  if (!construction_funder_)
    return std::unexpected("native GGML construction requires a paid backend funder");
  auto inventory = LowerConstructionOf(input);
  if (!inventory) return std::unexpected(inventory.error().detail);
  if (declaration_) {
    if (auto ready = declaration_->Ready(); !ready) return std::unexpected(ready.error());
  }
  // This guard precedes every native allocation (including its own shared
  // control). Declared before lowered/program so all refusal and abandoned
  // result paths destroy native storage before refunding the loan.
  if (!construction_funder_->charge(inventory->bytes))
    return std::unexpected("native GGML construction inventory was not funded");
  auto payment = std::make_shared<ConstructionPayment>();
  payment->funder = construction_funder_;
  payment->bytes = inventory->bytes;
  auto options = options_.lower;
  options.routed_funder = declaration_;
  auto lowered = LowerStep(input, *registry_, DeviceChoicesOf(*measuring_), options);
  if (!lowered) return std::unexpected(lowered.error().detail);
  if (std::ranges::any_of(lowered->plan.steps,
                          [&](const PlanStep& step) { return step.lane > options_.lanes.count; }))
    return std::unexpected("the step requires lanes outside its backend's execution domain");
  if (options_.lanes.count == 0) lowered->plan.regions.clear();
  auto scratch = PlanScratch(*measuring_, lowered->plan);
  if (!scratch) return std::unexpected(scratch.error().detail);
  std::uint64_t pool = *scratch, private_pool = *scratch;
  if (!lowered->plan.regions.empty()) {
    auto lane = PlanLaneScratch(*measuring_, lowered->plan);
    if (!lane) return std::unexpected(lane.error().detail);
    if (*lane > options_.lanes.scratch_bytes_per_lane)
      return std::unexpected("the step exceeds its declared per-lane scratch capacity");
    // This work uses the immutable lane slices carved from the launch pool's
    // tail. The runner adds lane_pool_reserve() beside the measured ordinary
    // scratch; charging it here too would fund the same physical slices twice.
  }
  for (const auto& selected : lowered->selected) {
    pool = std::max(pool, selected.call->scratch_bytes());
    private_pool = std::max(private_pool, selected.call->scratch_bytes());
  }
  auto program =
      std::make_unique<GgmlStepProgram>(launch, generation, *registry_, std::move(*lowered), pool,
                                        private_pool, options_.lanes, declaration_);
  if (!program->HoldConstruction(payment))
    return std::unexpected("native construction payment could not follow its program");
  return program;
}

std::expected<void, std::string> GgmlBackend::LendLibrary(kernels::StreamLibrary* library) {
  return LendCublas(static_cast<CublasHandle*>(library));
}

}  // namespace llmp::kernels::ggml

namespace llmp::kernels {

std::expected<std::uint64_t, std::string> StreamLibraryWorkspaceBytes(int device) {
  const auto facts = providers::QueryDeviceFacts(device);
  if (!facts) return std::unexpected(std::format("the device's facts: {}", facts.error().text()));
  return ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture)).value();
}

std::expected<std::unique_ptr<StreamLibrary>, std::string> CreateStreamLibrary(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    providers::DeviceSlice workspace) {
  auto handle = ggml::CublasHandle::Create(device, execution, stream,
                                           {.base = workspace.base, .size = workspace.size});
  if (!handle) return std::unexpected(std::format("cuBLAS: {}", handle.error().detail));
  return std::unique_ptr<StreamLibrary>(std::move(*handle));
}

BackendCapabilities DeviceBackend::Capabilities() { return ggml::GgmlBackend::Capabilities(); }

namespace {

class GgmlDeviceBackend final : public DeviceBackend {
 public:
  GgmlDeviceBackend(int device, providers::DeviceExecution& execution, providers::StreamId stream,
                    std::uint64_t library_workspace_bytes, std::uint32_t arch,
                    std::unique_ptr<ggml::GgmlBackend> backend)
      : device_(device),
        execution_(execution),
        stream_(stream),
        library_workspace_bytes_(library_workspace_bytes),
        arch_(arch),
        backend_(std::move(backend)) {}

  std::uint64_t library_workspace_bytes() const override { return library_workspace_bytes_; }
  std::uint64_t persistent_bytes() const override { return backend_->persistent_bytes(); }
  std::uint64_t lane_pool_reserve() const override { return backend_->lane_pool_reserve(); }
  std::uint32_t arch() const override { return arch_; }
  const execution::Registry& registry() const override { return backend_->registry(); }
  StepBackend Backend() override { return backend_->Backend(); }

  std::vector<execution::TunedShape> DefaultImageAttentionTuning(
      std::span<const std::uint64_t> rows, std::uint64_t head_dim) const override {
    if (arch() != 121) return {};
    std::vector<execution::TunedShape> tuned;
    tuned.reserve(rows.size());
    for (const auto r : rows) {
      tuned.push_back(execution::TunedShape{.implementation = ggml::kAttentionF32,
                                            .rows = r,
                                            .k = head_dim,
                                            .n = head_dim,
                                            .output_encoding = format::Encoding::kF32});
    }
    return tuned;
  }

  std::expected<void, std::string> FundConstruction(
      std::function<bool(std::uint64_t)> charge,
      std::function<void(std::uint64_t)> release) override {
    return backend_->FundConstruction(std::move(charge), std::move(release));
  }

  std::expected<void, std::string> Declare(std::span<const execution::TunedShape> tuning,
                                           std::function<bool(std::uint64_t)> charge,
                                           std::function<void(std::uint64_t)> release) override {
    return backend_->Declare(tuning, library_workspace_bytes_, std::move(charge),
                             std::move(release));
  }

  std::expected<void, std::string> BindWorkspaces(std::span<const std::uint64_t> bases) override {
    if (bases.size() != 1U + static_cast<std::size_t>(persistent_bytes() != 0)) {
      return std::unexpected("the kernel workspaces are not mapped");
    }
    selected_state_ = {persistent_bytes() ? bases[1] : 0, persistent_bytes()};
    auto cublas = ggml::CublasHandle::Create(
        device_, execution_, stream_,
        {.base = bases[0], .size = base::Bytes(library_workspace_bytes_)});
    if (!cublas) return std::unexpected(std::format("cuBLAS: {}", cublas.error().detail));
    cublas_ = std::move(*cublas);
    return backend_->LendCublas(cublas_.get());
  }

  void ReleaseWorkspaces() override {
    backend_->LaunchReleased();
    selected_state_ = {};
    cublas_.reset();
  }

  std::expected<std::unique_ptr<StepLaunch>, std::string> Launch(providers::DeviceSlice pool,
                                                                 bool allow_lanes = true) override {
    auto made = backend_->Launch({.base = pool.base, .size = pool.size}, selected_state_);
    if (!made) return std::unexpected(made.error());
    if (!allow_lanes && static_cast<ggml::GgmlStepLaunch*>(made->get())->context().lanes() != 0) {
      return std::unexpected("an image launch has concurrent lanes");
    }
    return made;
  }

 private:
  int device_;
  providers::DeviceExecution& execution_;
  providers::StreamId stream_;
  std::uint64_t library_workspace_bytes_;
  std::uint32_t arch_;
  std::unique_ptr<ggml::GgmlBackend> backend_;
  std::unique_ptr<ggml::CublasHandle> cublas_;
  ggml::SelectedWorkspace selected_state_{};
};

}  // namespace

std::expected<std::unique_ptr<DeviceBackend>, std::string> DeviceBackend::Create(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    DeviceBackendOptions options) {
  const auto facts = providers::QueryDeviceFacts(device);
  if (!facts) return std::unexpected(std::format("the device's facts: {}", facts.error().text()));
  const std::uint64_t library_workspace_bytes =
      ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture)).value();
  const std::uint32_t arch = facts->architecture / 10;
  ggml::GgmlBackendOptions ggml_options{
      .lower = std::move(options.lower),
      .selected = std::move(options.selected),
      .lanes = options.lanes,
  };
  auto backend =
      ggml::GgmlBackend::Create(device, execution, stream,
                                static_cast<ggml::CublasHandle*>(nullptr), std::move(ggml_options));
  if (!backend) return std::unexpected(backend.error());
  return std::make_unique<GgmlDeviceBackend>(device, execution, stream, library_workspace_bytes,
                                             arch, std::move(*backend));
}

}  // namespace llmp::kernels
