// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_RUNNER_OWNER_H_
#define LLMP_ENGINE_RUNNER_OWNER_H_

#include <expected>
#include <memory>
#include <string>
#include <utility>

#include "kernels/common/execution_resources.h"

namespace llmp::engine {
class ModelRunner;
class PagedNode;

// One paid intrusive lifetime owner. A batch retains the same complete runner,
// including its explicit dependencies and config's launch/handle/payer owners. The outer model
// can go away without destroying resources still named by submitted programs.
class RunnerOwner {
 public:
  RunnerOwner() = default;
  RunnerOwner(const RunnerOwner&) = default;
  RunnerOwner& operator=(const RunnerOwner&) = default;
  RunnerOwner(RunnerOwner&&) noexcept = default;
  RunnerOwner& operator=(RunnerOwner&&) noexcept = default;
  static std::expected<RunnerOwner, std::string> Adopt(
      PagedNode& node, std::unique_ptr<ModelRunner>& runner,
      std::shared_ptr<const void> dependencies = {});
  ModelRunner* get() const { return keep_ ? runner_ : nullptr; }
  ModelRunner* operator->() const { return get(); }
  ModelRunner& operator*() const { return *get(); }
  explicit operator bool() const { return get() != nullptr; }
  void reset() {
    keep_ = {};
    runner_ = nullptr;
  }

 private:
  friend class ModelRunner;
  RunnerOwner(kernels::ResourceKeep keep, ModelRunner* runner)
      : keep_(std::move(keep)), runner_(runner) {}
  kernels::ResourceKeep keep_;
  ModelRunner* runner_ = nullptr;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_RUNNER_OWNER_H_
