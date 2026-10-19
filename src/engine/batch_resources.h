// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_BATCH_RESOURCES_H_
#define LLMP_ENGINE_BATCH_RESOURCES_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <utility>

#include "engine/step_ahead.h"
#include "kernels/common/execution_resources.h"

namespace llmp::engine {
class ModelRunner;
class WaveComposer;
class StepTransaction;
struct PendingWork;
struct SegmentWork;
struct StepRequest;
struct ComposedStep;
using BatchAhead = std::function<StepAhead(const ComposedStep&, std::span<const std::size_t>)>;

// One driver-owned complete prompt batch. Copies retain paid metadata, not
// permission to keep an execution environment open after authenticated Close.
// Actual per-step permissions still belong to the canonical progress ledgers.
class PromptBatch {
 public:
  PromptBatch() = default;
  PromptBatch(const PromptBatch&) = default;
  PromptBatch& operator=(const PromptBatch&) = default;
  PromptBatch(PromptBatch&&) noexcept = default;
  PromptBatch& operator=(PromptBatch&&) noexcept = default;
  static bool DeclaredFor(const ModelRunner& runner);
  static std::expected<PromptBatch, kernels::ResourceFailure> Prepare(
      ModelRunner& runner, const WaveComposer& composer, std::span<const PendingWork> pending,
      std::span<const SegmentWork> work, std::span<const std::span<const std::int32_t>> histories,
      std::span<const std::byte> step_bytes = {}, const BatchAhead& ahead = {});
  explicit operator bool() const { return static_cast<bool>(keep_); }
  // A clean capacity refusal precedes every numeric child; Close itself never
  // asks for new optional storage. Failed/unknown close preserves all owners.
  std::expected<void, std::string> Close();
  bool ready() const;
  StepAhead Ahead(const planner::StepShape& shape) const;
  std::uint64_t paid_bytes() const;

 private:
  friend class StepTransaction;
  friend class ModelRunner;
  struct Data;
  PromptBatch(kernels::ResourceKeep keep, Data* data) : keep_(std::move(keep)), data_(data) {}
  kernels::ResourceKeep keep_;
  Data* data_ = nullptr;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_BATCH_RESOURCES_H_
