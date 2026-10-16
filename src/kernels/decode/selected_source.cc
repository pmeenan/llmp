// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/decode/selected_source.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "kernels/decode/selected_grouped.h"
#include "kernels/decode/selected_products.h"
#include "kernels/decode/selected_rows.h"
#include "providers/device_runtime.h"

namespace llmp::kernels::decode {
namespace {
namespace kg = ggml;

class CompositeLaunch final : public kg::SelectedLaunch {
 public:
  CompositeLaunch(const kg::SelectedSource* owner, providers::NativeStream stream,
                  std::vector<std::unique_ptr<kg::SelectedLaunch>> launches)
      : kg::SelectedLaunch(owner, stream), launches_(std::move(launches)) {}

  bool faulted() const override {
    return std::ranges::any_of(launches_, [](const auto& l) { return l->faulted(); });
  }

  bool capturable() const override {
    return std::ranges::all_of(launches_, [](const auto& l) { return l->capturable(); });
  }

  kg::SelectedLaunch& Child(std::size_t index) { return *launches_.at(index); }

 private:
  std::vector<std::unique_ptr<kg::SelectedLaunch>> launches_;
};

class CompositeCall final : public kg::SelectedCall {
 public:
  CompositeCall(const kg::SelectedSource* owner, std::size_t child_index,
                std::unique_ptr<kg::SelectedCall> child_call)
      : kg::SelectedCall(owner), child_index_(child_index), child_call_(std::move(child_call)) {}

  std::uint64_t scratch_bytes() const override { return child_call_->scratch_bytes(); }
  std::uint64_t host_bytes() const override { return child_call_->host_bytes(); }
  std::uint64_t launches() const override { return child_call_->launches(); }

  std::optional<kg::SelectedRoute> route() const override { return child_call_->route(); }
  std::uint64_t route_descriptor_bytes() const override {
    return child_call_->route_descriptor_bytes();
  }
  std::expected<void, kg::KernelFailure> RouteBuffers(kg::SelectedRouteBuffers buffers) override {
    return child_call_->RouteBuffers(buffers);
  }
  std::expected<void, kg::KernelFailure> StageDescriptors(
      std::span<std::byte> into) const override {
    return child_call_->StageDescriptors(into);
  }

  std::expected<std::unique_ptr<kg::BoundSelectedCall>, kg::KernelFailure> Bind(
      const execution::Implementation& impl, kg::SelectedLaunch& launch,
      kg::SelectedWorkspace ws) const override {
    if (launch.owner() != owner()) {
      return std::unexpected(kg::KernelFailure{.detail = "foreign composite launch"});
    }
    auto& comp_launch = static_cast<CompositeLaunch&>(launch);
    return child_call_->Bind(impl, comp_launch.Child(child_index_), ws);
  }

 private:
  std::size_t child_index_;
  std::unique_ptr<kg::SelectedCall> child_call_;
};

class CompositeSource final : public kg::SelectedSource {
 public:
  explicit CompositeSource(std::vector<std::shared_ptr<const kg::SelectedSource>> sources)
      : sources_(std::move(sources)) {}

  std::vector<execution::Implementation> Implementations() const override {
    std::vector<execution::Implementation> result;
    for (const auto& s : sources_) {
      auto impls = s->Implementations();
      result.insert(result.end(), std::make_move_iterator(impls.begin()),
                    std::make_move_iterator(impls.end()));
    }
    return result;
  }

  std::vector<execution::TunedShape> Tuning(std::uint32_t arch) const override {
    std::vector<execution::TunedShape> result;
    for (const auto& s : sources_) {
      auto tuned = s->Tuning(arch);
      result.insert(result.end(), std::make_move_iterator(tuned.begin()),
                    std::make_move_iterator(tuned.end()));
    }
    return result;
  }

  bool Owns(std::string_view implementation) const override {
    return std::ranges::any_of(sources_, [&](const auto& s) { return s->Owns(implementation); });
  }

  std::optional<base::Sha256Digest> ExecutionIdentity() const override {
    base::Sha256 hash;
    hash.Update("llmp-decode-composite-source-v1");
    for (const auto& impl : Implementations()) {
      const auto id = execution::IdentityOf(impl);
      hash.Update(std::as_bytes(std::span(id)));
    }
    return hash.Finish();
  }

  std::uint64_t persistent_bytes() const override {
    std::uint64_t total = 0;
    for (const auto& s : sources_) {
      total += s->persistent_bytes();
    }
    return total;
  }

  std::expected<std::unique_ptr<kg::SelectedCall>, kg::KernelFailure> Lower(
      const execution::Implementation& implementation, const ir::Node& node,
      std::span<const kg::SelectedOperand> inputs,
      std::span<const kg::SelectedOperand> outputs) const override {
    for (std::size_t i = 0; i < sources_.size(); ++i) {
      if (sources_[i]->Owns(implementation.name)) {
        auto call = sources_[i]->Lower(implementation, node, inputs, outputs);
        if (!call) return std::unexpected(call.error());
        return std::make_unique<CompositeCall>(this, i, std::move(*call));
      }
    }
    return std::unexpected(
        kg::KernelFailure{.detail = "no child source owns implementation " + implementation.name});
  }

  std::expected<std::unique_ptr<kg::SelectedLaunch>, kg::KernelFailure> Launch(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      kg::SelectedWorkspace persistent) const override {
    std::vector<std::unique_ptr<kg::SelectedLaunch>> launches;
    launches.reserve(sources_.size());
    std::uint64_t offset = 0;
    for (const auto& s : sources_) {
      const auto bytes = s->persistent_bytes();
      kg::SelectedWorkspace child_ws{.base = persistent.base + offset, .bytes = bytes};
      offset += bytes;
      auto l = s->Launch(device, execution, stream, child_ws);
      if (!l) return std::unexpected(l.error());
      launches.push_back(std::move(*l));
    }
    const auto native = execution.Submission(stream);
    if (!native) return std::unexpected(kg::KernelFailure{.detail = native.error().detail});
    return std::make_unique<CompositeLaunch>(this, *native, std::move(launches));
  }

 private:
  std::vector<std::shared_ptr<const kg::SelectedSource>> sources_;
};

}  // namespace

std::shared_ptr<const ggml::SelectedSource> CompositeSelectedSource(
    std::vector<std::shared_ptr<const ggml::SelectedSource>> sources) {
  return std::make_shared<CompositeSource>(std::move(sources));
}

std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> DecodeSelectedSource(
    std::optional<format::Encoding> encoding) {
  std::vector<std::shared_ptr<const ggml::SelectedSource>> sources;

  auto grouped = GroupedSelectedSource(encoding);
  if (grouped) sources.push_back(std::move(*grouped));

  auto rows = RowsSelectedSource(encoding);
  if (rows) sources.push_back(std::move(*rows));

  auto products = ProductsSelectedSource(encoding);
  if (products) sources.push_back(std::move(*products));

  if (sources.empty()) {
    return std::unexpected("no decode sources available for requested encoding");
  }
  return CompositeSelectedSource(std::move(sources));
}

std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string>
DecodeResidentSelectedSource(std::optional<format::Encoding> encoding) {
  std::vector<std::shared_ptr<const ggml::SelectedSource>> sources;
  if (auto grouped = ResidentGroupedSelectedSource(encoding))
    sources.push_back(std::move(*grouped));
  if (auto rows = RowsSelectedSource(encoding)) sources.push_back(std::move(*rows));
  if (auto products = ProductsSelectedSource(encoding)) sources.push_back(std::move(*products));
  if (sources.empty()) return std::unexpected("no decode sources available for requested encoding");
  return CompositeSelectedSource(std::move(sources));
}

}  // namespace llmp::kernels::decode
