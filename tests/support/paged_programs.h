// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler's task programs (scheduler/programs.h) under the names the
// paged harnesses and their tests use, and one program only tests post:
// LaunchOnlyProgram, which submits a job over a closure it never
// materialized (BP-P2's incomplete closure).

#ifndef LLMP_TESTS_SUPPORT_PAGED_PROGRAMS_H_
#define LLMP_TESTS_SUPPORT_PAGED_PROGRAMS_H_

#include <atomic>
#include <cstdint>
#include <utility>

#include "catalog/catalog.h"
#include "scheduler/programs.h"
#include "scheduler/scheduler.h"

namespace llmp::test_support {

using Done = scheduler::ProgramDone;
using HarnessProgram = scheduler::ReportingProgram;
using scheduler::AcquireProgram;
using scheduler::AcquireReport;
using scheduler::CallProgram;
using scheduler::EvictingProgram;
using scheduler::EvictProgram;
using scheduler::RequestChannel;
using scheduler::RequestProgram;
using scheduler::RunProgram;
using scheduler::SwapProgram;
using scheduler::SwapReport;

// Submits a job over a closure without materializing it first: with any
// extent nonresident the lease, and so the submission, must be refused
// before the job can run (BP-P2's incomplete closure).
class LaunchOnlyProgram final : public HarnessProgram {
 public:
  LaunchOnlyProgram(Done& done, catalog::Closure closure, std::atomic<bool>& ran,
                    std::uint32_t stream = 0)
      : HarnessProgram(done), closure_(std::move(closure)), ran_(ran), stream_(stream) {}
  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (submitted_) {
      return scheduler::Step::Finish(context.TakeFailure() ? scheduler::TaskOutcome::kFailed
                                                           : scheduler::TaskOutcome::kSucceeded);
    }
    std::atomic<bool>& ran = ran_;
    const auto submitted = context.SubmitLaunch(
        closure_, scheduler::LaunchWork{.stream = stream_, .job = [&ran](providers::NativeStream) {
                                          ran.store(true);
                                          return scheduler::JobResult::kQueued;
                                        }});
    if (!submitted) {
      return Fail(submitted.error());
    }
    submitted_ = true;
    return scheduler::Step::Wait();
  }

 private:
  catalog::Closure closure_;
  std::atomic<bool>& ran_;
  std::uint32_t stream_;
  bool submitted_ = false;
};

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_PAGED_PROGRAMS_H_
