// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/runner_owner.h"

#include <new>

#include "base/check.h"
#include "engine/model_runner.h"

namespace llmp::engine {
namespace {
constexpr std::uint64_t kAllocationMetadataBytes = 64;
}

struct ModelRunner::Lifetime final : kernels::ResourceDependency {
  Lifetime(HostPayment paid, std::shared_ptr<const void> owned, std::unique_ptr<ModelRunner> value)
      : payment(std::move(paid)), dependencies(std::move(owned)), runner(std::move(value)) {
    runner->lifetime_ = this;
    registered.model = runner.get();
  }
  kernels::ResourceKeep Publish() { return Adopt(); }
  void Destroy() override {
    auto paid = std::move(payment);
    delete this;  // complete runner and root/control storage go before refund
    paid.Reset();
  }
  HostPayment payment;
  // Source/config borrowers die with the complete runner before their owners.
  std::shared_ptr<const void> dependencies;
  std::unique_ptr<ModelRunner> runner;
  RetainedModelRoot registered;
};

struct ModelRunner::ResourceLease final : kernels::ResourceDependency {
  ResourceLease(HostPayment paid, ModelRunner& value)
      : payment(std::move(paid)), root(value.lifetime_->Keep()), runner(value) {
    runner.resource_leases_.fetch_add(1, std::memory_order_relaxed);
  }
  ~ResourceLease() override {
    base::Check(runner.resource_leases_.fetch_sub(1, std::memory_order_acq_rel) != 0,
                "a private resource lease is released once");
  }
  bool ready() const override { return !runner.closing_ && !runner.released_; }
  kernels::ResourceKeep Publish() { return Adopt(); }
  void Destroy() override {
    auto paid = std::move(payment);
    delete this;  // root keeps runner valid through the lease destructor
    paid.Reset();
  }
  HostPayment payment;
  kernels::ResourceKeep root;
  ModelRunner& runner;
};

std::expected<RunnerOwner, std::string> RunnerOwner::Adopt(
    PagedNode& node, std::unique_ptr<ModelRunner>& runner,
    std::shared_ptr<const void> dependencies) {
  if (!runner || runner->lifetime_ || &runner->node_ != &node)
    return std::unexpected("a retained runner needs its unique unpublished root");
  auto paid = node.PrepayHost(sizeof(ModelRunner::Lifetime) + kAllocationMetadataBytes);
  if (!paid) return std::unexpected(paid.error());
  auto* root = new (std::nothrow)
      ModelRunner::Lifetime(std::move(*paid), std::move(dependencies), std::move(runner));
  if (!root) return std::unexpected("allocating a paid runner root failed");
  root->registered.keep = root->Keep();
  node.KeepModelRoot(root->registered);
  return RunnerOwner(root->Publish(), root->runner.get());
}

void ModelRunner::RetireRoot() {
  if (lifetime_) node_.RetireModelRoot(lifetime_->registered);
}

RunnerOwner ModelRunner::RetainedOwner() const {
  return lifetime_ ? RunnerOwner(lifetime_->Keep(), lifetime_->runner.get()) : RunnerOwner{};
}

std::expected<kernels::ParentResources, kernels::ResourceFailure> ModelRunner::KeepResources() {
  if (!lifetime_)
    return std::unexpected(kernels::ResourceFailure{
        kernels::ResourceError::kOwnership, "private resources require a retained runner owner"});
  if (released_ || closing_ || !bound_ || !programs_.launch())
    return std::unexpected(
        kernels::ResourceFailure{kernels::ResourceError::kOwnership,
                                 "private resources require a live bound parent runner"});
  auto paid = node_.PrepayHost(sizeof(ResourceLease) + kAllocationMetadataBytes);
  if (!paid)
    return std::unexpected(
        kernels::ResourceFailure{kernels::ResourceError::kCapacity, paid.error()});
  auto* lease = new (std::nothrow) ResourceLease(std::move(*paid), *this);
  if (!lease)
    return std::unexpected(kernels::ResourceFailure{
        kernels::ResourceError::kCapacity, "allocating a paid private resource lease failed"});
  return kernels::ParentResources(programs_.launch(), lease->Publish());
}
}  // namespace llmp::engine
