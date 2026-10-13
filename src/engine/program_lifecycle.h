// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_PROGRAM_LIFECYCLE_H_
#define LLMP_ENGINE_PROGRAM_LIFECYCLE_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>

#include "engine/coverage.h"
#include "engine/plan_account.h"
#include "engine/step_runs.h"

namespace llmp::engine {
class PagedNode;

// Actual measuring-program requirements, before workspace admission. This
// does not pay construction: its caller already owns the graph/compiler grant.
// Input staging excludes device-carried feature ports unless requested.
struct ProgramEnvelope {
  std::uint64_t scratch = 0, pool = 0, host = 0, launches = 0;
  std::uint64_t inputs = 0, outputs = 0;
  static std::expected<ProgramEnvelope, std::string> Measure(const kernels::StepProgram& program,
                                                             bool stage_features = false);
  void Include(const ProgramEnvelope& other);
};

// Common prepared-program resources for token and typed-port pipelines.
// Consumers own the backend/source and cache; their declaration/destruction
// order keeps those owners alive until programs/recordings and this launch go.
// This object neither submits work nor turns a failed Status into retirement.
class ProgramLifecycle {
 public:
  ProgramLifecycle(PagedNode& node, int owner, bool graphs);
  ProgramLifecycle(const ProgramLifecycle&) = delete;
  ProgramLifecycle& operator=(const ProgramLifecycle&) = delete;
  ProgramLifecycle(ProgramLifecycle&&) = delete;
  ProgramLifecycle& operator=(ProgramLifecycle&&) = delete;
  ~ProgramLifecycle();

  std::expected<void, std::string> BindLaunch(std::unique_ptr<kernels::StepLaunch> launch);
  // Caller-specific shape/typed-port checks run after Bind, before coverage.
  // Source/routing authority is checked by the consumer before this call.
  using Validate = std::function<std::expected<void, std::string>(const kernels::StepProgram&)>;
  std::expected<Coverage, std::string> BindPrepared(StepPlanned& planned, const Validate& validate);
  StepRuns& runs() { return runs_; }
  const StepRuns& runs() const { return runs_; }
  kernels::StepLaunch* launch() const { return launch_.get(); }
  PlanAccount& account() { return account_; }
  // Only after positive whole-owner drain, and after all caches/recordings
  // have been cleared. Unknown completion retains this entire owner instead.
  void ReleaseAfterDrain();

 private:
  PagedNode& node_;
  int owner_;
  StepRuns runs_;
  std::unique_ptr<kernels::StepLaunch> launch_;
  PlanAccount account_;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_PROGRAM_LIFECYCLE_H_
