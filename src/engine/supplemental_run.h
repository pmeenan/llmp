// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_SUPPLEMENTAL_RUN_H_
#define LLMP_ENGINE_SUPPLEMENTAL_RUN_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <utility>

#include "catalog/catalog.h"
#include "engine/execution_contract.h"
#include "kernels/common/execution_resources.h"
#include "scheduler/programs.h"

namespace llmp::engine {

// A paid reusable reservation for one supplemental continuation plus its node
// controls. It authenticates the actual base channel and catalog domain, and
// never grants closure authority. Copies retain payment; an offered operation
// prevents reuse until all its state has been destroyed after proven retirement.
class SupplementalCapacity {
 public:
  SupplementalCapacity() = default;
  explicit operator bool() const { return bool(keep_); }

 private:
  friend class SupplementalRun;
  friend class PagedNode;
  struct Data;
  SupplementalCapacity(kernels::ResourceKeep keep, Data* data)
      : keep_(std::move(keep)), data_(data) {}
  kernels::ResourceKeep keep_;
  Data* data_ = nullptr;
};

// One extra closure under an existing request's base lease. Create and Retire
// run on the same catalog/scheduler thread; the driver offers Program and reads
// done(). These calls never overlap. The node posts only to that scheduler.
// The complete payload owner is lifetime storage, never closure authority.
// Caller-created job captures and payload allocations need their own funding.
class SupplementalRun {
 public:
  static std::expected<std::uint64_t, std::string> Bound(std::uint64_t extents);
  static std::expected<SupplementalCapacity, std::string> Prepare(
      const catalog::Catalog& catalog, catalog::DomainId domain,
      const scheduler::RequestChannel& base, const scheduler::ProgramDone& base_done,
      std::uint64_t max_extents, std::shared_ptr<ExecutionMetadataAllowance> allowance);
  // Checks only reservation ownership/payment; Create still authenticates every
  // actual selected extent and the live base request before any effects.
  static bool Covers(const SupplementalCapacity& capacity, const scheduler::RequestChannel& base,
                     catalog::DomainId domain,
                     const std::shared_ptr<ExecutionMetadataAllowance>& allowance);
  static constexpr std::uint64_t kControlBytes = 4096;
  static std::expected<std::unique_ptr<SupplementalRun>, std::string> Create(
      const catalog::Catalog& catalog, catalog::DomainId domain, scheduler::RequestChannel& base,
      const scheduler::ProgramDone& base_done, const catalog::Closure& selected,
      scheduler::DeviceJob job, std::uint32_t stream, std::shared_ptr<const void> payload,
      std::shared_ptr<ExecutionMetadataAllowance> allowance,
      const SupplementalCapacity& capacity = {});
  SupplementalRun(const SupplementalRun&) = delete;
  SupplementalRun& operator=(const SupplementalRun&) = delete;
  ~SupplementalRun();

  scheduler::ProgramDone& done();
  // Exactly once. Until handed to the scheduler, destroying the returned
  // program proves no effect. Thereafter gone alone is insufficient.
  std::unique_ptr<scheduler::TaskProgram> Program();
  // True only after program destruction AND all its own operations and scoped
  // page-ins retired. Success/failure of the work is separate from retirement.
  // A false/error result keeps the base counter, payload and payment intact.
  std::expected<bool, std::string> Retire(const scheduler::Scheduler& scheduler);

 private:
  struct State;
  class ProgramImpl;
  explicit SupplementalRun(std::shared_ptr<State> state) : state_(std::move(state)) {}
  std::shared_ptr<State> state_;
};

}  // namespace llmp::engine
#endif  // LLMP_ENGINE_SUPPLEMENTAL_RUN_H_
