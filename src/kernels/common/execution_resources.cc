// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/common/execution_resources.h"

#include "kernels/common/lower.h"
#include "kernels/common/step_program.h"

namespace llmp::kernels {
ResourceKeep::ResourceKeep(const ResourceKeep& other) : dependency_(other.dependency_) {
  if (dependency_) dependency_->Add();
}
ResourceKeep& ResourceKeep::operator=(const ResourceKeep& other) {
  ResourceKeep replacement(other);
  std::swap(dependency_, replacement.dependency_);
  return *this;
}
ResourceKeep& ResourceKeep::operator=(ResourceKeep&& other) noexcept {
  ResourceKeep replacement(std::move(other));
  std::swap(dependency_, replacement.dependency_);
  return *this;
}
ResourceKeep::~ResourceKeep() {
  if (dependency_) dependency_->Drop();
}
bool ResourceKeep::ready() const { return dependency_ && dependency_->ready(); }
ResourceKeep ResourceDependency::Keep() {
  Add();
  return ResourceKeep(this);
}

ResourceOrigin::ResourceOrigin(ResourceOrigin&& other) noexcept
    : parent_(std::move(other.parent_)),
      declaration_(other.declaration_),
      retained_(std::move(other.retained_)),
      backing_(std::move(other.backing_)),
      bytes_(std::exchange(other.bytes_, 0)) {}
ResourceOrigin::~ResourceOrigin() {
  if (bytes_ != 0) parent_->Release(bytes_);
}
void ResourceInstance::Drop() {
  if (references_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    auto origin = std::move(origin_);
    delete this;
  }  // The paid instance/control bytes are gone before origin refunds them.
}
std::expected<void, std::string> ResourceInstance::Ready() const {
  if (!origin_.parent_ready()) return std::unexpected("the parent resource owner is draining");
  auto current = origin_.parent()->Ready();
  if (!current) return std::unexpected(current.error());
  if (*current != origin_.declaration())
    return std::unexpected("private execution resources lost their original producer identity");
  if (launch().faulted()) return std::unexpected("private execution resources have faulted");
  return {};
}
std::expected<void, std::string> ExecutionResources::Ready() const {
  if (!instance_) return std::unexpected("private execution resources are absent");
  return instance_->Ready();
}
std::expected<std::unique_ptr<StepProgram>, std::string> ExecutionResources::Lower(
    const LowerInput& input) const {
  if (auto ready = Ready(); !ready) return std::unexpected(ready.error());
  auto program = instance_->Lower(input);
  if (!program) return std::unexpected(program.error());
  if (!*program || !(**program).HoldResources(*this))
    return std::unexpected("private lowered program lacks its unique resource owner");
  return program;
}

std::expected<ExecutionResources, ResourceFailure> BackendOwner::Resources(
    std::shared_ptr<const BackendOwner> self, ResourceWorkspace workspace, ParentResources retained,
    ResourceKeep backing) const {
  if (self.get() != this || !retained.ready() || !backing ||
      (workspace.bytes == 0) != (workspace.base == 0) || workspace.base % 256 != 0 ||
      workspace.bytes % 256 != 0 || workspace.bytes > UINT64_MAX - workspace.base)
    return std::unexpected(
        ResourceFailure{ResourceError::kOwnership,
                        "private execution resources need their actual producer and paid backing"});
  auto declaration = Ready();
  if (!declaration)
    return std::unexpected(ResourceFailure{ResourceError::kFault, declaration.error()});
  auto bytes = ResourceBytes(workspace);
  if (!bytes) return std::unexpected(bytes.error());
  if (*bytes == 0 || !Charge(*bytes))
    return std::unexpected(ResourceFailure{ResourceError::kCapacity,
                                           "private execution resource metadata cannot be paid"});
  ResourceOrigin origin(std::move(self), *declaration, std::move(retained), std::move(backing),
                        *bytes);
  auto instance = MakeResources(std::move(origin), workspace);
  if (!instance) return std::unexpected(instance.error());
  ExecutionResources result(*instance);
  if (!result || result.parent() != this)
    return std::unexpected(ResourceFailure{
        ResourceError::kOwnership, "private execution resources belong to another producer"});
  if (auto ready = result.Ready(); !ready)
    return std::unexpected(ResourceFailure{ResourceError::kFault, ready.error()});
  return result;
}
}  // namespace llmp::kernels
