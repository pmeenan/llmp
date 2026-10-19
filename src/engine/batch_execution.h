// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_BATCH_EXECUTION_H_
#define LLMP_ENGINE_BATCH_EXECUTION_H_

// Internal driver-owned implementation shared by the batch factory and actual
// step transaction. No caller can construct a batch or publish retirement.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "engine/batch_inventory.h"
#include "engine/batch_resources.h"
#include "engine/model_runner.h"
#include "engine/program_lifecycle.h"
#include "engine/runner_memory.h"

namespace llmp::engine {
class StepTransaction;

struct PromptBatch::Data final : kernels::ResourceDependency, PagedModel {
  struct Backing final : kernels::ResourceDependency {
    Backing(HostPayment paid, PagedNode& node, int owner, std::uint32_t stream)
        : payment(std::move(paid)), memory(node, owner, stream) {}
    kernels::ResourceKeep Publish() { return Adopt(); }
    bool ready() const override { return open; }
    void Destroy() override;
    HostPayment payment, host_inputs_payment;
    std::unique_ptr<std::byte[]> host_inputs;
    Mapped activations, pool;
    RunnerMemory memory;
    bool open = true;
  };
  struct Entry {
    // Host payment precedes the program, every plan descriptor and result.
    HostPayment payment, frame_payment;
    std::unique_ptr<kernels::StepProgram> built_program;
    double build_seconds = 0;
    ModelRunner::Plans::Entry plan;
    ModelRunner::Plans::Entry* cached = nullptr;
    ModelRunner::Plans::Entry& selected() { return cached ? *cached : plan; }
    StepAhead ahead;
    ProgramEnvelope measured;
    std::uint64_t inputs_first = 0;
    // A batch preallocates its transaction/control permission storage before
    // any numeric child. It is consumed exactly once at actual mint/use.
    std::shared_ptr<StepTransaction> transaction;
    std::shared_ptr<ModelRunner::RoutedFlight> route;
    bool consumed = false;
  };
  Data(HostPayment paid, ModelRunner& runner, PromptBatchInventory preview);
  kernels::ResourceKeep Publish() { return Adopt(); }
  void Destroy() override;
  bool ready() const override;
  std::uint32_t stream() const override;
  const catalog::Closure& fence_closure() const override { return empty; }
  std::vector<catalog::ExtentId> managed_extents() const override { return {}; }
  Status Release() override;
  Status Close();
  std::expected<std::size_t, std::string> Find(const ModelRunner& runner,
                                               const planner::StepShape& shape) const;

  bool Matches(std::size_t index, const planner::StepShape& shape) const;

  HostPayment payment;
  RunnerOwner owner;
  PromptBatchInventory inventory;
  kernels::ParentResources retained_parent;
  kernels::ResourceKeep backing_keep;
  Backing* backing = nullptr;
  std::vector<kernels::ExecutionResources> views;
  std::vector<Entry> entries;
  StepRuns runs{false};
  RequestExtension extension;
  RetainedModelRoot registered;
  catalog::Closure empty;
  PagedNode& node;
  const std::uint32_t stream_index;
  std::uint64_t output_first = 0, route_first = 0, route_bytes = 0, pinned_bytes = 0, pending = 0;
  std::byte* pinned = nullptr;
  bool private_resources = true, owns_request = false, unknown = false, closing = false,
       closed = false, published = false;

 private:
  // This callback is inaccessible as a caller-asserted drain override.
  Status RetireAfterFence() override;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_BATCH_EXECUTION_H_
