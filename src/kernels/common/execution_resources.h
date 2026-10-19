// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_COMMON_EXECUTION_RESOURCES_H_
#define LLMP_KERNELS_COMMON_EXECUTION_RESOURCES_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <utility>

#include "execution/registry.h"

namespace llmp::engine {
class ModelRunner;
}

namespace llmp::kernels {

class BackendOwner;
class StepLaunch;
class StepProgram;
struct LowerInput;
class ResourceInstance;

// Intrusive ownership for already paid backing/dependency objects. The producer
// of each concrete object destroys its payload/control before returning its
// original receipt. No new allocation or rights are introduced by copying it.
class ResourceDependency;
class ResourceKeep {
 public:
  ResourceKeep() = default;
  ResourceKeep(const ResourceKeep& other);
  ResourceKeep& operator=(const ResourceKeep& other);
  ResourceKeep(ResourceKeep&& other) noexcept
      : dependency_(std::exchange(other.dependency_, nullptr)) {}
  ResourceKeep& operator=(ResourceKeep&& other) noexcept;
  ~ResourceKeep();
  explicit operator bool() const { return dependency_ != nullptr; }
  bool ready() const;

 private:
  friend class ResourceDependency;
  explicit ResourceKeep(ResourceDependency* dependency) : dependency_(dependency) {}
  ResourceDependency* dependency_ = nullptr;
};

class ResourceDependency {
 public:
  ResourceKeep Keep();
  virtual bool ready() const { return true; }
  ResourceDependency(const ResourceDependency&) = delete;
  ResourceDependency& operator=(const ResourceDependency&) = delete;

 protected:
  ResourceDependency() = default;
  virtual ~ResourceDependency() = default;
  // Adopts the construction reference once, after successful publication.
  ResourceKeep Adopt() { return ResourceKeep(this); }
  virtual void Destroy() = 0;

 private:
  friend class ResourceKeep;
  std::atomic<std::size_t> references_{1};
  void Add() { references_.fetch_add(1, std::memory_order_relaxed); }
  void Drop() {
    if (references_.fetch_sub(1, std::memory_order_acq_rel) == 1) Destroy();
  }
};

// Authenticated by the actual owning runner, never by a caller's opaque keep.
// Retains its complete launch environment and prevents release/unmapping while
// a private view names the original lanes, selected calls or library handle.
class ParentResources {
 public:
  ParentResources() = default;
  StepLaunch* launch() const { return launch_; }
  explicit operator bool() const { return launch_ && keep_; }
  bool ready() const { return launch_ && keep_ && keep_.ready(); }

 private:
  friend class llmp::engine::ModelRunner;
  ParentResources(StepLaunch* launch, ResourceKeep keep)
      : launch_(launch), keep_(std::move(keep)) {}
  StepLaunch* launch_ = nullptr;
  ResourceKeep keep_;
};

enum class ResourceError : std::uint8_t {
  kOwnership,
  kUnsupported,
  kCapacity,
  kBounds,
  kProvider,
  kFault
};
struct ResourceFailure {
  ResourceError error = ResourceError::kUnsupported;
  std::string what;
};

struct ResourceWorkspace {
  std::uint64_t base = 0, bytes = 0;
};

// Only the actual producer can mint this origin. Its original paid receipt
// follows the complete private instance, never a caller-supplied capability.
class ResourceOrigin {
 public:
  ResourceOrigin(const ResourceOrigin&) = delete;
  ResourceOrigin& operator=(const ResourceOrigin&) = delete;
  ResourceOrigin(ResourceOrigin&& other) noexcept;
  ResourceOrigin& operator=(ResourceOrigin&&) = delete;
  ~ResourceOrigin();
  const std::shared_ptr<const BackendOwner>& parent() const { return parent_; }
  const execution::BackendDeclaration& declaration() const { return declaration_; }
  StepLaunch* parent_launch() const { return retained_.launch(); }
  bool parent_ready() const { return retained_.ready(); }
  std::uint64_t host_bytes() const { return bytes_; }

 private:
  friend class BackendOwner;
  ResourceOrigin(std::shared_ptr<const BackendOwner> parent,
                 execution::BackendDeclaration declaration, ParentResources retained,
                 ResourceKeep backing, std::uint64_t bytes)
      : parent_(std::move(parent)),
        declaration_(declaration),
        retained_(std::move(retained)),
        backing_(std::move(backing)),
        bytes_(bytes) {}
  std::shared_ptr<const BackendOwner> parent_;
  execution::BackendDeclaration declaration_;
  ParentResources retained_;
  ResourceKeep backing_;
  std::uint64_t bytes_ = 0;
};

// Ready/lowering/submission use the serialized driver. Copies and destruction
// may cross threads; their retained root keeps driver cleanup on the node.
// Producer-owned private binding and launch. The last intrusive reference moves
// payment out, destroys this complete object/all its allocations, THEN refunds.
// Private programs retain the same instance, including uncertain completion.
class ResourceInstance {
 public:
  ResourceInstance(const ResourceInstance&) = delete;
  ResourceInstance& operator=(const ResourceInstance&) = delete;
  virtual StepLaunch& launch() const = 0;
  virtual std::expected<std::unique_ptr<StepProgram>, std::string> Lower(
      const LowerInput& input) const = 0;
  const BackendOwner* parent() const { return origin_.parent().get(); }
  std::uint64_t host_bytes() const { return origin_.host_bytes(); }
  std::expected<void, std::string> Ready() const;

 protected:
  explicit ResourceInstance(ResourceOrigin origin) : origin_(std::move(origin)) {}
  virtual ~ResourceInstance() = default;
  const std::shared_ptr<const BackendOwner>& parent_owner() const { return origin_.parent(); }

 private:
  friend class ExecutionResources;
  ResourceOrigin origin_;
  std::atomic<std::size_t> references_{1};
  void Add() { references_.fetch_add(1, std::memory_order_relaxed); }
  void Drop();
};

class ExecutionResources {
 public:
  ExecutionResources() = default;
  ExecutionResources(const ExecutionResources& other) : instance_(other.instance_) {
    if (instance_) instance_->Add();
  }
  ExecutionResources& operator=(const ExecutionResources& other) {
    ExecutionResources replacement(other);
    Swap(replacement);
    return *this;
  }
  ExecutionResources(ExecutionResources&& other) noexcept
      : instance_(std::exchange(other.instance_, nullptr)) {}
  ExecutionResources& operator=(ExecutionResources&& other) noexcept {
    ExecutionResources replacement(std::move(other));
    Swap(replacement);
    return *this;
  }
  ~ExecutionResources() {
    if (instance_) instance_->Drop();
  }
  explicit operator bool() const { return instance_ != nullptr; }
  const BackendOwner* parent() const { return instance_ ? instance_->parent() : nullptr; }
  StepLaunch* launch() const { return instance_ ? &instance_->launch() : nullptr; }
  std::uint64_t host_bytes() const { return instance_ ? instance_->host_bytes() : 0; }
  std::expected<void, std::string> Ready() const;
  std::expected<std::unique_ptr<StepProgram>, std::string> Lower(const LowerInput& input) const;

 private:
  friend class BackendOwner;
  friend class StepProgram;
  explicit ExecutionResources(ResourceInstance* instance) : instance_(instance) {}
  void Swap(ExecutionResources& other) noexcept { std::swap(instance_, other.instance_); }
  ResourceInstance* instance_ = nullptr;
};

}  // namespace llmp::kernels
#endif  // LLMP_KERNELS_COMMON_EXECUTION_RESOURCES_H_
