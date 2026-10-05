// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include "engine/gemma4_assistant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <utility>

#include "engine/support.h"
#include "providers/device_runtime.h"

namespace jitllm::engine {
namespace kg = kernels::ggml;
namespace md = model;
namespace sc = scheduler;
using support::Address;
using support::Error;
using support::Pointer;
using support::Round;
Status CheckGemma4AssistantOutputs(std::uint32_t owners, std::uint32_t target_width,
                                   std::span<const float> heads, std::span<const float> features) {
  if (owners == 0 || owners > kMaxRequestSlots ||
      (target_width != md::Gemma4_26BA4B().width && target_width != md::Gemma4_31B().width) ||
      heads.size() != std::uint64_t{owners} * md::Gemma4_26BA4B().vocab ||
      features.size() != std::uint64_t{owners} * target_width)
    return Error("Gemma assistant completed output shape differs");
  if (!std::ranges::all_of(heads, [](float v) { return std::isfinite(v); }) ||
      !std::ranges::all_of(features, [](float v) { return std::isfinite(v); }))
    return Error("Gemma assistant completed output is nonfinite");
  return {};
}
Gemma4Assistant::Gemma4Assistant(Gemma4Runner& target)
    : target_(target),
      profile_(target.profile_ == md::Gemma4_31B() ? md::Gemma4Assistant31()
                                                   : md::Gemma4Assistant26()),
      resources_(target.node_, target.owner_, target.stream_),
      runs_(target.o_.graphs) {}
Status Gemma4Assistant::Setup(const std::filesystem::path& path) {
  if (auto r = weights_.Open(path); !r) return r;
  auto binding = md::BindGemma4Assistant(profile_, weights_.artifact());
  if (!binding) return Error(binding.error());
  binding_ = std::move(*binding);
  if (auto r =
          md::CheckGemma4AssistantTarget(profile_, binding_, target_.profile_, target_.binding_);
      !r)
    return Error(r.error());
  const std::vector<GroupPlace> place(weights_.artifact().groups().size(), GroupPlace::kDevice);
  if (auto r = weights_.Reserve(target_.node_, place, {}); !r) return r;
  model_.profile = &profile_;
  model_.binding = &binding_;
  model_.target = &target_.model_;
  for (std::uint32_t i = 0; i < weights_.artifact().resources().size(); ++i)
    model_.resources.push_back(
        {weights_.resource_address(i), weights_.artifact().resources()[i].readable.value()});
  auto measuring = target_.resources_.MeasuringContext();
  if (!measuring) return Error(measuring.error());
  auto choices = kg::DeviceChoicesOf(**measuring);
  // Ordinary primitive policy. Do not inherit target's experimental norm,
  // routing, quant writer or row-invariant options across this new width.
  std::uint64_t staging = 0;
  for (std::uint32_t count = 1; count <= target_.o_.slots; ++count) {
    kg::Gemma4AssistantShape shape;
    const auto prefix = target_.layout_.context - 1;
    for (std::uint32_t i = 0; i < count; ++i)
      shape.segments.push_back({i, prefix,
                                std::min(target_.layout_.local_cells, (prefix + 255) / 256 * 256),
                                (prefix + 255) / 256 * 256});
    auto p = PlanGemma4Assistant(model_, shape, choices, 0, 0);
    if (!p) return Error(std::format("measuring Gemma assistant: {}", p.error()));
    auto scratch = kg::PlanScratch(**measuring, (*p)->plan);
    if (!scratch) return Error(scratch.error().detail);
    auto host = Gemma4AssistantSourceBytes((*p)->graph, true);
    if (!host) return Error(host.error());
    activation_bytes_ = std::max(activation_bytes_, (*p)->placement.extent);
    scratch_bytes_ = std::max(scratch_bytes_, *scratch);
    staging = std::max(staging, (*p)->inputs_bytes);
    // Shape/input vectors and borrowed-pointer bookkeeping also live under
    // the caller's host grant before allocation, bounded by 16 owners.
    host_bytes_ = std::max(host_bytes_, *host + count * (sizeof(kg::Gemma4AssistantSegment) +
                                                         sizeof(Gemma4AssistantInput)));
    plan_floor_bytes_ = std::max(plan_floor_bytes_, PlannedHostBytes(**p));
  }
  activation_bytes_ = Round(activation_bytes_ + activation_bytes_ / 4, kPagedExtent);
  scratch_bytes_ = Round(scratch_bytes_ + scratch_bytes_ / 4 + (1U << 20U), kPagedExtent);
  host_bytes_ = Round(host_bytes_ + (1U << 20U), kPagedExtent);
  const auto staging_bytes = Round(staging + staging / 4 + (1U << 20U), kPagedExtent);
  auto staging_host = resources_.Pinned(staging_bytes);
  auto logits =
      resources_.Pinned(std::uint64_t{target_.o_.slots} * target_.profile_.vocab * sizeof(float));
  auto features =
      resources_.Pinned(std::uint64_t{target_.o_.slots} * profile_.target_width * sizeof(float));
  auto factors = resources_.Pinned(target_.profile_.global_rope_dims / 2 * sizeof(float));
  if (!staging_host || !logits || !features || !factors)
    return Error("Gemma assistant pinned inputs/outputs");
  runs_.SetStaging(*staging_host, staging_bytes);
  logits_ = *logits;
  features_ = *features;
  factors_ = *factors;
  feature_stride_ = Round(std::uint64_t{profile_.target_width} * sizeof(float), 256);
  if (auto r = resources_.Map(recurrent_, "Gemma assistant recurrent features",
                              feature_stride_ * target_.o_.slots, catalog::MemoryClass::kLiveState);
      !r)
    return r;
  setup_success_ = true;
  return {};
}
Status Gemma4Assistant::Register() {
  if (!setup_success_ || registered_ || released_)
    return Error("Gemma assistant registration order");
  if (auto r = weights_.Register(target_.node_, target_.owner_); !r) return r;
  if (auto r = target_.node_.scheduler().PinPlaces(weights_.extents()); !r)
    return Error(std::format("Gemma assistant places: {}", sc::ToString(r.error())));
  registered_ = true;
  return {};
}
std::vector<catalog::ExtentId> Gemma4Assistant::Shared() const {
  auto extents = weights_.extents();
  const auto own = resources_.extents();
  extents.insert(extents.end(), own.begin(), own.end());
  return extents;
}
Status Gemma4Assistant::Bind() {
  if (!setup_success_ || !registered_ || bound_ || released_)
    return Error("Gemma assistant bind order");
  runs_.SetLaunch(&target_.resources_.launch());
  account_.Bind([this](std::uint64_t bytes,
                       bool required) { return target_.node_.ChargeHost(bytes, required); },
                [this](std::uint64_t bytes) { target_.node_.UnchargeHost(bytes); });
  plans_.set_account(&account_);
  bound_ = true;
  return {};
}
Status Gemma4Assistant::Factors() {
  if (factors_checked_) return {};
  bool unknown = false;
  auto copied = target_.node_.Job(
      target_.execution_,
      [&](providers::NativeStream native) {
        unknown =
            !providers::CopyAsync(native, factors_,
                                  Pointer(weights_.resource_address(binding_.rope_freqs.index)),
                                  target_.profile_.global_rope_dims / 2 * sizeof(float),
                                  providers::CopyKind::kDeviceToHost)
                 .ok();
        return unknown ? sc::JobResult::kUnknown : sc::JobResult::kQueued;
      },
      "Gemma assistant frequency factors", target_.stream_);
  if (!copied) {
    auto states = target_.States();
    target_.cohort_.CheckFailedJob(target_.node_, target_.stream_, target_.execution_, states);
    if (unknown) target_.cohort_.Fault(states);
    return copied;
  }
  if (auto r = md::CheckGemma4RopeFactors(
          target_.profile_,
          std::span(static_cast<const float*>(factors_), target_.profile_.global_rope_dims / 2));
      !r)
    return Error(r.error());
  factors_checked_ = true;
  return {};
}
std::expected<Gemma4Assistant::Plans::Entry*, std::string> Gemma4Assistant::Planned(
    const kg::Gemma4AssistantShape& shape) {
  if (auto* found = plans_.Find(shape)) return found;
  const auto started = std::chrono::steady_clock::now();
  auto choices = kg::DeviceChoicesOf(target_.resources_.launch());
  auto p = PlanGemma4Assistant(model_, shape, choices, target_.node_.activations().base,
                               target_.node_.activations().bytes);
  if (!p) return Error(p.error());
  if (auto r = BindPlanned(**p, target_.resources_.launch(), target_.resources_.registry(),
                           "Gemma assistant");
      !r)
    return Error(r.error());
  std::vector<const ggml_tensor*> state;
  for (const auto& s : (*p)->graph.segments)
    for (const auto& cache : s.caches) {
      state.push_back(cache.first);
      state.push_back(cache.second);
    }
  CheckCoverage(target_.node_, target_.owner_, (*p)->graph.nodes,
                {.state = state, .inputs = (*p)->graph.inputs}, coverage_);
  if (coverage_.violations != 0)
    return Error("Gemma assistant catalog coverage: " + coverage_.first_violation);
  const auto bytes = PlannedHostBytes(**p), nodes = PlannedNodes(**p);
  return &plans_.Add(shape, std::move(*p), bytes, nodes,
                     support::Seconds(std::chrono::steady_clock::now() - started));
}
Status Gemma4Assistant::Step(std::span<const Gemma4Runner::FrozenBorrow* const> borrows,
                             std::span<const std::int32_t> anchors, bool first,
                             std::span<std::vector<float>* const> logits,
                             std::span<std::vector<float>* const> features) {
  const PlanStep step;
  if (!setup_success_ || !bound_ || released_ || borrows.empty() ||
      borrows.size() > target_.o_.slots || anchors.size() != borrows.size() ||
      logits.size() != borrows.size() || (!features.empty() && features.size() != borrows.size()))
    return Error("Gemma assistant owner/output count differs");
  std::array<bool, kMaxRequestSlots> seen{};
  for (std::size_t i = 0; i < borrows.size(); ++i) {
    if (borrows[i] == nullptr) return Error("Gemma assistant missing frozen borrow");
    const auto& b = *borrows[i];
    if (auto r = target_.CheckBorrow(b); !r) return r;
    if (seen[b.slot_] || anchors[i] < 0 ||
        std::cmp_greater_equal(anchors[i], target_.profile_.vocab) || logits[i] == nullptr ||
        (!features.empty() && features[i] == nullptr) ||
        (first ? anchors[i] != b.anchor_
               : epoch_[b.slot_] != b.epoch_ || prefix_[b.slot_] != b.prefix_))
      return Error("Gemma assistant proposal/feature generation differs");
    seen[b.slot_] = true;
    for (std::size_t j = 0; j < i; ++j)
      if (logits[i] == logits[j] ||
          (!features.empty() &&
           (features[i] == features[j] || features[i] == logits[j] || logits[i] == features[j])))
        return Error("Gemma assistant publication vectors must be independent");
    if (!features.empty() && features[i] == logits[i])
      return Error("Gemma assistant feature/head output aliases");
  }
  if (auto r = target_.CheckPlaces(); !r) return r;
  if (auto r = Factors(); !r) return r;
  struct Grant {
    PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  };
  if (!target_.node_.ChargeHost(host_bytes_, false))
    return Error("Gemma assistant host sources do not fit");
  const Grant grant{target_.node_, host_bytes_};
  kg::Gemma4AssistantShape shape;
  std::vector<Gemma4AssistantInput> input;
  for (std::size_t i = 0; i < borrows.size(); ++i) {
    const auto& b = *borrows[i];
    shape.segments.push_back({b.slot_, b.prefix_,
                              std::min(target_.layout_.local_cells, (b.prefix_ + 255) / 256 * 256),
                              (b.prefix_ + 255) / 256 * 256});
    input.push_back({b.slot_, b.prefix_, anchors[i]});
  }
  auto planned = Planned(shape);
  if (!planned) return Error(planned.error());
  auto& entry = **planned;
  auto& p = *entry.planned;
  auto source = Gemma4AssistantSources(p.graph, input, {}, true, host_bytes_);
  if (!source) return Error(source.error());
  auto copies = runs_.Stage(source->sources, 0);
  if (!copies) return Error(copies.error());
  bool capture = entry.runs[0].CaptureDue(runs_.graphs());
  if (capture && !plans_.ChargeGraph(entry)) capture = false;
  const std::array<RunCopy, 2> outputs{
      {{Address(logits_), Address(p.graph.logits->data),
        std::uint64_t{borrows.size()} * target_.profile_.vocab * sizeof(float)},
       {Address(features_), Address(p.graph.next_features->data),
        std::uint64_t{borrows.size()} * profile_.target_width * sizeof(float)}}};
  bool unknown = false, started = false;
  Status queued;
  RunPath path = RunPath::kEager;
  auto posted = target_.node_.Job(
      target_.execution_,
      [&](providers::NativeStream native) {
        started = true;
        const auto between = [&](void* raw) {
          const providers::NativeStream stream{raw};
          for (std::size_t i = 0; i < borrows.size(); ++i) {
            const auto& b = *borrows[i];
            const auto address = first ? b.feature_ : recurrent_.base + b.slot_ * feature_stride_;
            if (!providers::CopyAsync(stream,
                                      Pointer(Address(p.graph.features->data) +
                                              i * profile_.target_width * sizeof(float)),
                                      Pointer(address), profile_.target_width * sizeof(float),
                                      providers::CopyKind::kDeviceToDevice)
                     .ok())
              return false;
          }
          return true;
        };
        // first vs recurrent changes captured D2D source addresses. Keep these
        // copies outside capture; their stream order precedes captured consumers.
        if (!between(native.handle)) {
          unknown = true;
          queued = Error("Gemma assistant input copy is uncertain");
          return sc::JobResult::kUnknown;
        }
        const auto result =
            runs_.Queue(entry.runs[0], *copies, {}, *p.bound, outputs, capture, stats_, native);
        path = result.path;
        if (!result.result) {
          queued = Error(result.result.error().detail);
          unknown = result.result.error().error == kg::KernelError::kUnknown;
          return unknown         ? sc::JobResult::kUnknown
                 : result.before ? sc::JobResult::kFailed
                                 : sc::JobResult::kNotStarted;
        }
        for (std::size_t i = 0; i < borrows.size(); ++i)
          if (!providers::CopyAsync(
                   native, Pointer(recurrent_.base + borrows[i]->slot_ * feature_stride_),
                   Pointer(Address(p.graph.next_features->data) +
                           i * profile_.target_width * sizeof(float)),
                   profile_.target_width * sizeof(float), providers::CopyKind::kDeviceToDevice)
                   .ok()) {
            unknown = true;
            queued = Error("Gemma assistant recurrent copy is uncertain");
            return sc::JobResult::kUnknown;
          }
        return sc::JobResult::kQueued;
      },
      "Gemma assistant frozen query", target_.stream_);
  if (!posted || !queued || target_.resources_.launch().faulted()) {
    auto states = target_.States();
    target_.cohort_.CheckFailedJob(target_.node_, target_.stream_, target_.execution_, states);
    if (unknown || target_.resources_.launch().faulted()) target_.cohort_.Fault(states);
    if (started || unknown)
      for (const auto* b : borrows) epoch_[b->slot_] = 0;
    return !queued ? queued : !posted ? posted : Error("Gemma assistant launch faulted");
  }
  const auto invalidate = [&] {
    for (const auto* b : borrows) epoch_[b->slot_] = 0;
  };
  if (auto r = CheckGemma4AssistantOutputs(
          static_cast<std::uint32_t>(borrows.size()), profile_.target_width,
          std::span(static_cast<const float*>(logits_), borrows.size() * target_.profile_.vocab),
          std::span(static_cast<const float*>(features_), borrows.size() * profile_.target_width));
      !r) {
    invalidate();
    return r;
  }
  for (const auto* borrow : borrows)
    if (auto r = target_.CheckBorrow(*borrow); !r) {
      invalidate();
      return r;
    }
  for (std::size_t i = 0; i < borrows.size(); ++i) {
    const auto* head = static_cast<const float*>(logits_) + i * target_.profile_.vocab;
    logits[i]->assign(head, head + target_.profile_.vocab);
    if (!features.empty()) {
      const auto* feature = static_cast<const float*>(features_) + i * profile_.target_width;
      features[i]->assign(feature, feature + profile_.target_width);
    }
    epoch_[borrows[i]->slot_] = borrows[i]->epoch_;
    prefix_[borrows[i]->slot_] = borrows[i]->prefix_;
  }
  Count(stats_, path);
  return {};
}
Status Gemma4Assistant::Release() {
  if (released_) return {};
  released_ = true;
  DropPlans();
  std::vector<std::string> problems;
  resources_.Release(problems);
  if (weights_.opened())
    if (auto r = weights_.Release(target_.node_.memory()); !r) problems.push_back(r.error());
  return support::Joined(problems);
}
}  // namespace jitllm::engine
