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
#include "scheduler/programs.h"

namespace llmp::engine {

// One extra closure under an existing request's base lease. Create and Retire
// run on the same catalog/scheduler thread; the driver offers Program and reads
// done(). These calls never overlap. The node posts only to that scheduler.
// The complete payload owner is lifetime storage, never closure authority.
// Caller-created job captures and payload allocations need their own funding.
class SupplementalRun {
 public:
  static std::expected<std::uint64_t, std::string> Bound(std::uint64_t extents);
  static std::expected<std::unique_ptr<SupplementalRun>, std::string> Create(
      const catalog::Catalog& catalog, catalog::DomainId domain, scheduler::RequestChannel& base,
      const scheduler::ProgramDone& base_done, const catalog::Closure& selected,
      scheduler::DeviceJob job, std::uint32_t stream, std::shared_ptr<const void> payload,
      std::shared_ptr<ExecutionMetadataAllowance> allowance);
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
