// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/common/composite_source.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "providers/device_execution.h"

namespace llmp::kernels {
namespace {

// Each child's persistent workspace starts at a 256-byte boundary.
constexpr std::uint64_t kChildAlign = 256;
std::optional<std::uint64_t> AlignedTotal(
    std::span<const std::shared_ptr<const SelectedSource>> sources) {
  std::uint64_t total = 0;
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  for (const auto& source : sources) {
    const auto bytes = source->persistent_bytes();
    if (bytes > max - (kChildAlign - 1)) return {};
    const auto aligned = (bytes + kChildAlign - 1) / kChildAlign * kChildAlign;
    if (aligned > max - total) return {};
    total += aligned;
  }
  return total;
}

class CompositeLaunch final : public SelectedLaunch {
 public:
  CompositeLaunch(const SelectedSource* owner, providers::NativeStream stream,
                  std::vector<std::unique_ptr<SelectedLaunch>> launches)
      : SelectedLaunch(owner, stream), launches_(std::move(launches)) {}

  bool faulted() const override {
    return std::ranges::any_of(launches_, [](const auto& l) { return l->faulted(); });
  }

  bool capturable() const override {
    return std::ranges::all_of(launches_, [](const auto& child) { return child->capturable(); });
  }

  SelectedLaunch& Child(std::size_t index) { return *launches_.at(index); }
  const SelectedLaunch& Child(std::size_t index) const { return *launches_.at(index); }

 private:
  std::vector<std::unique_ptr<SelectedLaunch>> launches_;
};

class CompositeCall final : public SelectedCall {
 public:
  CompositeCall(const SelectedSource* owner, std::size_t child_index,
                std::unique_ptr<SelectedCall> child_call)
      : SelectedCall(owner), child_index_(child_index), child_call_(std::move(child_call)) {}

  std::uint64_t scratch_bytes() const override { return child_call_->scratch_bytes(); }
  std::uint64_t host_bytes() const override { return child_call_->host_bytes(); }
  std::uint64_t launches() const override { return child_call_->launches(); }
  bool capturable(const SelectedLaunch& launch) const override {
    return launch.owner() == owner() &&
           child_call_->capturable(static_cast<const CompositeLaunch&>(launch).Child(child_index_));
  }

  std::optional<SelectedRoute> route() const override { return child_call_->route(); }
  std::uint64_t route_descriptor_bytes() const override {
    return child_call_->route_descriptor_bytes();
  }
  std::expected<void, KernelFailure> RouteBuffers(SelectedRouteBuffers buffers) override {
    return child_call_->RouteBuffers(buffers);
  }
  std::expected<void, KernelFailure> StageDescriptors(std::span<std::byte> into) const override {
    return child_call_->StageDescriptors(into);
  }

  std::expected<std::unique_ptr<BoundSelectedCall>, KernelFailure> Bind(
      const execution::Implementation& impl, SelectedLaunch& launch,
      SelectedWorkspace ws) const override {
    if (launch.owner() != owner()) {
      return std::unexpected(KernelFailure{.detail = "foreign composite launch"});
    }
    auto& comp_launch = static_cast<CompositeLaunch&>(launch);
    return child_call_->Bind(impl, comp_launch.Child(child_index_), ws);
  }

 private:
  std::size_t child_index_;
  std::unique_ptr<SelectedCall> child_call_;
};

class CompositeSource final : public SelectedSource {
 public:
  explicit CompositeSource(std::vector<std::shared_ptr<const SelectedSource>> sources)
      : sources_(std::move(sources)) {
    base::Sha256 hash;
    for (const auto& impl : Implementations()) {
      const auto id = execution::IdentityOf(impl);
      hash.Update(std::as_bytes(std::span(id)));
    }
    declarations_ = hash.Finish();
  }

  std::vector<execution::Implementation> Implementations() const override {
    std::vector<execution::Implementation> result;
    for (const auto& s : sources_) {
      auto impls = s->Implementations();
      for (auto& impl : impls)
        if (!std::ranges::contains(result, impl.name, &execution::Implementation::name))
          result.push_back(std::move(impl));
    }
    return result;
  }

  std::vector<execution::TunedShape> Tuning(std::uint32_t arch) const override {
    std::vector<execution::TunedShape> result;
    for (const auto& s : sources_) {
      auto tuned = s->Tuning(arch);
      for (const auto& fact : tuned)
        if (!std::ranges::contains(result, fact)) result.push_back(fact);
    }
    return result;
  }

  bool Owns(std::string_view implementation) const override {
    return std::ranges::any_of(sources_, [&](const auto& s) { return s->Owns(implementation); });
  }

  std::optional<base::Sha256Digest> ExecutionIdentity() const override {
    base::Sha256 hash;
    hash.Update("llmp-decode-composite-source-v2");
    for (const auto& source : sources_) {
      const auto child = source->ExecutionIdentity();
      if (!child) return {};
      hash.Update(std::as_bytes(std::span(*child)));
    }
    hash.Update(std::as_bytes(std::span(declarations_)));
    return hash.Finish();
  }

  // The children's workspaces, each at a 256-byte boundary (Launch).
  std::uint64_t persistent_bytes() const override {
    // The interface has no error result: an unrepresentable workspace
    // advertises the largest bound, and Launch refuses it before children run.
    return AlignedTotal(sources_).value_or(std::numeric_limits<std::uint64_t>::max());
  }

  std::expected<std::unique_ptr<SelectedCall>, KernelFailure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const SelectedOperand> inputs,
      std::span<const SelectedOperand> outputs) const override {
    for (std::size_t i = 0; i < sources_.size(); ++i) {
      if (sources_[i]->Owns(implementation.name)) {
        auto call = sources_[i]->Lower(implementation, node, inputs, outputs);
        if (!call) return std::unexpected(call.error());
        return std::make_unique<CompositeCall>(this, i, std::move(*call));
      }
    }
    return std::unexpected(
        KernelFailure{.detail = "no child source owns implementation " + implementation.name});
  }

  std::expected<std::unique_ptr<SelectedLaunch>, KernelFailure> Launch(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      SelectedWorkspace persistent) const override {
    const auto needed = AlignedTotal(sources_);
    if (!needed || persistent.bytes < *needed ||
        (*needed > 0 && (persistent.base == 0 || persistent.base % kChildAlign != 0)) ||
        persistent.base + persistent.bytes < persistent.base) {
      return std::unexpected(KernelFailure{
          .detail = "the composite's persistent workspace does not cover its children aligned"});
    }
    std::vector<std::unique_ptr<SelectedLaunch>> launches;
    launches.reserve(sources_.size());
    std::uint64_t offset = 0;
    for (const auto& s : sources_) {
      const auto bytes = s->persistent_bytes();
      SelectedWorkspace child_ws{.base = persistent.base + offset, .bytes = bytes};
      // AlignedTotal proved that each round-up and the complete sum fit.
      offset += (bytes + kChildAlign - 1) / kChildAlign * kChildAlign;
      auto l = s->Launch(device, execution, stream, child_ws);
      if (!l) return std::unexpected(l.error());
      launches.push_back(std::move(*l));
    }
    const auto native = execution.Submission(stream);
    if (!native) return std::unexpected(KernelFailure{.detail = native.error().detail});
    return std::make_unique<CompositeLaunch>(this, *native, std::move(launches));
  }

 private:
  std::vector<std::shared_ptr<const SelectedSource>> sources_;
  base::Sha256Digest declarations_{};
};

}  // namespace

std::shared_ptr<const SelectedSource> CompositeSelectedSource(
    std::vector<std::shared_ptr<const SelectedSource>> sources) {
  return std::make_shared<CompositeSource>(std::move(sources));
}

}  // namespace llmp::kernels
