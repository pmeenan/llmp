// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The task programs a driver posts to the scheduler (engine/paged_node.h,
// which the runtime and the paged harnesses drive; the fake backend's tests
// check them). Each tells a ProgramDone what became of it; its destructor
// is its last touch of that ProgramDone, so the thread that posted it may
// end the ProgramDone's frame once `gone` is set. Scheduler-owned page-ins
// can outlive a withdrawn AcquireProgram: their source/state owners also
// require a scoped PageInsRetired proof. A driver never returns while a
// program, or a job it queued, may still refer to its frame
// (completion-aware lifetimes, D-048).
//
// AcquireProgram is BP-S3's (docs/backend-proof.md): it makes room for a
// closure under the execution budget B with the memory module's planning
// (memory/materialize.h), evicting the victims it chooses, which are never
// the closure's own extents or the ones it protects, and then materializes
// the closure. With two models in one catalog, the victims are the other
// model's clean weights.
//
// RequestProgram is a request's task (M3's lease per request, scheduler.h):
// it materializes the request's closure and leases it once, then runs each
// step its driver hands it (RequestChannel) as device work under that
// lease, until the driver ends the request. The evicting programs wait for
// a request's lease to be released rather than retrying past it.

#ifndef JITLLM_SCHEDULER_PROGRAMS_H_
#define JITLLM_SCHEDULER_PROGRAMS_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "memory/materialize.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace jitllm::scheduler {

// What a program tells the thread that posted it.
struct ProgramDone {
  std::atomic<int> outcome{-1};
  std::atomic<bool> retired{false};
  std::atomic<int> error{-1};  // a WorkError that ended it
  // The program is destroyed: retired, or never admitted (a refused start
  // goes with its control). Its last touch of `done`: a job it submitted
  // retired first (its lease held until its fence), and one never
  // submitted went with it. Scheduler-owned materialization can outlive a
  // withdrawn task; gone alone does not retire its source/state payloads.
  std::atomic<bool> gone{false};
};

// A program that tells `done` how it ended and when it is destroyed.
class ReportingProgram : public scheduler::TaskProgram {
 public:
  explicit ReportingProgram(ProgramDone& done) : done_(done) {}
  ReportingProgram(const ReportingProgram&) = delete;
  ReportingProgram& operator=(const ReportingProgram&) = delete;
  ReportingProgram(ReportingProgram&&) = delete;
  ReportingProgram& operator=(ReportingProgram&&) = delete;
  ~ReportingProgram() override { done_.gone.store(true); }

  void Finished(scheduler::TaskOutcome outcome) override {
    done_.outcome.store(static_cast<int>(outcome));
  }
  void Retired() override { done_.retired.store(true); }

 protected:
  scheduler::Step Fail(scheduler::WorkError error) {
    done_.error.store(static_cast<int>(error));
    return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
  }

  ProgramDone& done_;
};

// Materializes a closure, then optionally runs one device job over it on
// the device lane's stream `stream`.
class RunProgram final : public ReportingProgram {
 public:
  RunProgram(ProgramDone& done, catalog::Closure closure, scheduler::DeviceJob job,
             std::uint32_t stream = 0)
      : ReportingProgram(done),
        closure_(std::move(closure)),
        job_(std::move(job)),
        stream_(stream) {}

  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (context.TakeFailure()) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
    }
    if (submitted_) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
    }
    const auto ready = context.Materialize(closure_);
    if (!ready) {
      if (ready.error() == scheduler::WorkError::kBusy) {
        return scheduler::Step::Yield();  // no mailbox now: again next turn
      }
      return Fail(ready.error());
    }
    if (*ready == scheduler::Readiness::kWaiting) {
      return scheduler::Step::Wait();
    }
    if (!job_) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
    }
    // A refused launch is left in `work`.
    scheduler::LaunchWork work{.stream = stream_, .job = std::move(job_)};
    const auto submitted = context.SubmitLaunch(closure_, std::move(work));
    if (!submitted) {
      if (submitted.error() == scheduler::WorkError::kBusy) {
        // No mailbox now (other work holds them): the job, untouched,
        // again next turn.
        // NOLINTNEXTLINE(bugprone-use-after-move): SubmitLaunch leaves a refused launch in work
        job_ = std::move(work.job);
        return scheduler::Step::Yield();
      }
      return Fail(submitted.error());
    }
    submitted_ = true;
    return scheduler::Step::Wait();
  }

 private:
  catalog::Closure closure_;
  scheduler::DeviceJob job_;
  std::uint32_t stream_;
  bool submitted_ = false;
};

// Runs a call on the scheduler thread, which alone may change the
// scheduler's records (registering sources anew) or read the catalog once
// the scheduler runs.
class CallProgram final : public ReportingProgram {
 public:
  CallProgram(ProgramDone& done, std::function<std::expected<void, std::string>()> call)
      : ReportingProgram(done), call_(std::move(call)) {}
  scheduler::Step Advance(scheduler::TaskContext& /*context*/) override {
    status_ = call_();
    return scheduler::Step::Finish(status_ ? scheduler::TaskOutcome::kSucceeded
                                           : scheduler::TaskOutcome::kFailed);
  }
  const std::expected<void, std::string>& status() const { return status_; }

 private:
  std::function<std::expected<void, std::string>()> call_;
  std::expected<void, std::string> status_;
};

// Evicts extents in order, up to kWindow at once (each unmap, and
// write-back, in flight together), then waits for them all before the
// next; those not resident are passed over. With a handoff
// (scheduler::EvictOptions) each eviction keeps its managed backing
// parked, and a load this task starts may take it; whatever no load took
// is released when the task finishes.
class EvictingProgram : public ReportingProgram {
 public:
  static constexpr std::size_t kWindow = 256;

  EvictingProgram(ProgramDone& done, std::vector<catalog::ExtentId> extents,
                  scheduler::EvictOptions options, std::vector<catalog::ExtentId> release = {})
      : ReportingProgram(done),
        extents_(std::move(extents)),
        options_(options),
        release_(release.begin(), release.end()) {}

 protected:
  // One round of evictions: nullopt once every one has ended, else the
  // step to return.
  std::optional<scheduler::Step> EvictRound(scheduler::TaskContext& context) {
    std::size_t issued = 0;
    while (next_ < extents_.size() && issued < kWindow) {
      const catalog::ExtentId extent = extents_[next_];
      if (context.catalog().Describe(extent).value().state != catalog::ExtentState::kResident) {
        ++next_;
        continue;
      }
      auto options = options_;
      if (release_.contains(extent)) options.handoff = false;
      const auto evicted = context.Evict(extent, options);
      if (!evicted) {
        if (evicted.error() == scheduler::WorkError::kBusy) {
          // A request's lease holds it: wait for its release (this extent
          // is tried again then). Otherwise no mailbox now, or a job's own
          // lease that ends at its fence: wait for the evictions in
          // flight, or try next turn.
          const auto held = context.AwaitRelease(std::span(&extent, 1));
          if (held && *held == scheduler::Readiness::kWaiting) {
            return scheduler::Step::Wait();
          }
          return issued > 0 ? scheduler::Step::Wait() : scheduler::Step::Yield();
        }
        return Fail(evicted.error());
      }
      ++next_;
      ++evicted_;
      if (*evicted == scheduler::Readiness::kWaiting) {
        ++issued;
      }
    }
    if (issued > 0) {
      return scheduler::Step::Wait();
    }
    return std::nullopt;
  }
  std::uint64_t evicted() const { return evicted_; }

 private:
  std::vector<catalog::ExtentId> extents_;
  scheduler::EvictOptions options_;
  std::set<catalog::ExtentId> release_;
  std::size_t next_ = 0;
  std::uint64_t evicted_ = 0;
};

class EvictProgram final : public EvictingProgram {
 public:
  EvictProgram(ProgramDone& done, std::vector<catalog::ExtentId> extents,
               scheduler::EvictOptions options = {})
      : EvictingProgram(done, std::move(extents), options) {}
  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (context.TakeFailure()) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
    }
    if (auto step = EvictRound(context)) {
      return *step;
    }
    return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
  }
};

// What a swap did, and when (steady clock, on the scheduler thread): read
// by the poster once the Done is gone.
struct SwapReport {
  std::chrono::steady_clock::time_point started;
  std::chrono::steady_clock::time_point evicted;  // every eviction ended (or parked)
  std::chrono::steady_clock::time_point loaded;   // the incoming closure resident
  std::uint64_t evictions = 0;                    // extents resident when it began
  std::uint64_t loads = 0;                        // closure extents nonresident then
  // Requests the node ended before the swap because they held outgoing
  // extents (a swap asked for between a request's steps, paged_node.h).
  std::uint64_t requests_ended = 0;
};

// A full swap (M3): evicts the outgoing extents (write-back first for live
// state at a write-back place), then materializes the incoming closure.
// With a handoff the evicted backing is parked, the incoming loads take it
// instead of creating their own (scheduler.h), and what they leave is
// released when this task finishes, after `loaded`. Nothing is leased.
class SwapProgram final : public EvictingProgram {
 public:
  SwapProgram(ProgramDone& done, std::vector<catalog::ExtentId> out, catalog::Closure in,
              bool handoff, SwapReport& report, std::vector<catalog::ExtentId> release = {},
              catalog::Closure weights = {})
      : EvictingProgram(done, std::move(out), scheduler::EvictOptions{.handoff = handoff},
                        std::move(release)),
        in_(std::move(in)),
        weights_(std::move(weights)),
        report_(report) {}

  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (context.TakeFailure()) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
    }
    if (!started_) {
      // Refuse stale or newly held policy victims before any eviction.
      for (const auto& [extent, generation] : weights_.extents) {
        const auto view = context.catalog().Describe(extent);
        if (!view || view->content_generation != generation ||
            !catalog::Catalog::Evictable(*view) ||
            view->descriptor.memory_class != catalog::MemoryClass::kWeights ||
            view->descriptor.recovery != catalog::Recovery::kFromArtifact)
          return Fail(scheduler::WorkError::kBusy);
      }
      for (const auto& [extent, generation] : in_.extents) {
        const auto view = context.catalog().Describe(extent);
        if (!view || view->content_generation != generation || view->discarded ||
            view->state == catalog::ExtentState::kQuarantined)
          return Fail(scheduler::WorkError::kBusy);
      }
      started_ = true;
      report_.started = std::chrono::steady_clock::now();
    }
    if (!evicted_all_) {
      if (auto step = EvictRound(context)) {
        return *step;
      }
      evicted_all_ = true;
      report_.evicted = std::chrono::steady_clock::now();
      report_.evictions = evicted();
      for (const auto& [extent, generation] : in_.extents) {
        report_.loads +=
            context.catalog().Describe(extent).value().state == catalog::ExtentState::kResident ? 0
                                                                                                : 1;
      }
    }
    const auto ready = context.Materialize(in_);
    if (!ready) {
      if (ready.error() == scheduler::WorkError::kBusy) {
        return scheduler::Step::Yield();
      }
      return Fail(ready.error());
    }
    if (*ready == scheduler::Readiness::kWaiting) {
      return scheduler::Step::Wait();
    }
    report_.loaded = std::chrono::steady_clock::now();
    return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
  }

 private:
  catalog::Closure in_;
  catalog::Closure weights_;
  SwapReport& report_;
  bool started_ = false;
  bool evicted_all_ = false;
};

// Between a request's driver (the thread that drives the node) and its task
// (RequestProgram). The driver writes `job`, `stream` and `end`, then posts
// a SignalRequest for the request; the task reads them only once that
// signal has woken it (the control queue orders the two), and touches `job`
// no more once it has counted the step. The task publishes the rest.
struct RequestChannel {
  scheduler::DeviceJob job;  // the next step's
  std::uint32_t stream = 0;
  bool end = false;                      // end the request instead
  std::atomic<bool> held{false};         // the lease is taken
  std::atomic<std::uint64_t> steps{0};   // steps ended: their fence seen, or refused
  std::atomic<bool> step_failed{false};  // the last step's outcome, set before `steps`
  std::atomic<int> step_error{-1};       // a WorkError that refused it, else -1
  base::WakeFlag reported;               // signalled after each `steps`, for a driver asleep
  // Steps the driver runs under the lease itself (PagedNode's direct
  // steps), not yet seen to end: the lease outlasts them (HoldLease), which
  // shares the count, so it may outlive the channel.
  std::shared_ptr<std::atomic<std::uint32_t>> external =
      std::make_shared<std::atomic<std::uint32_t>>(0);
};

// A request's task: materializes `closure` and holds a lease on it
// (TaskContext::HoldLease) until the driver ends the request, running each
// step as device work under that lease: per step, no closure is walked and
// nothing is leased or released, and the step still ends on its fence. A
// step refused (other than for want of a mailbox, retried) or failed is
// reported and the request goes on. Ending (or finishing any other way:
// failed, cancelled) releases the lease once the step in flight, if any,
// has drained; its extents stay resident.
class RequestProgram final : public ReportingProgram {
 public:
  RequestProgram(ProgramDone& done, catalog::Closure closure, RequestChannel& channel)
      : ReportingProgram(done), closure_(std::move(closure)), channel_(channel) {}

  scheduler::Step Advance(scheduler::TaskContext& context) override {
    while (true) {
      switch (phase_) {
        case Phase::kLeasing: {
          if (context.TakeFailure()) {
            return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);  // a page-in failed
          }
          const auto ready = context.Materialize(closure_);
          if (!ready) {
            if (ready.error() == scheduler::WorkError::kBusy) {
              return scheduler::Step::Yield();
            }
            return Fail(ready.error());
          }
          if (*ready == scheduler::Readiness::kWaiting) {
            return scheduler::Step::Wait();
          }
          // In the same step as the materialization: nothing can have
          // begun evicting it since.
          const auto held = context.HoldLease(closure_, channel_.external);
          if (!held) {
            if (held.error() == scheduler::WorkError::kBusy) {
              return scheduler::Step::Yield();
            }
            return Fail(held.error());
          }
          lease_ = *held;
          phase_ = Phase::kIdle;
          channel_.held.store(true, std::memory_order_release);
          continue;
        }
        case Phase::kIdle: {
          const auto signalled = context.AwaitSignal();
          if (!signalled) {
            return Fail(signalled.error());
          }
          if (*signalled == scheduler::Readiness::kWaiting) {
            return scheduler::Step::Wait();
          }
          if (channel_.end) {
            phase_ = Phase::kEnding;
            const auto ended = context.EndLease(lease_);
            if (!ended) {
              return Fail(ended.error());
            }
            if (*ended == scheduler::Readiness::kWaiting) {
              return scheduler::Step::Wait();
            }
            return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
          }
          phase_ = Phase::kSubmitting;
          continue;
        }
        case Phase::kSubmitting: {
          scheduler::LaunchWork work{.stream = channel_.stream, .job = std::move(channel_.job)};
          const auto submitted = context.SubmitLaunch(lease_, std::move(work));
          if (!submitted) {
            if (submitted.error() == scheduler::WorkError::kBusy) {
              // No mailbox now: the job, untouched (SubmitLaunch leaves a
              // refused launch in work), again next turn.
              // NOLINTNEXTLINE(bugprone-use-after-move)
              channel_.job = std::move(work.job);
              return scheduler::Step::Yield();
            }
            work.job = nullptr;  // gone before the driver hears of it
            Report(true, static_cast<int>(submitted.error()));
            continue;
          }
          phase_ = Phase::kRunning;
          return scheduler::Step::Wait();
        }
        case Phase::kRunning:
          Report(context.TakeFailure(), -1);  // its fence was seen: it touches nothing more
          continue;
        case Phase::kEnding:
          return scheduler::Step::Finish(context.TakeFailure()
                                             ? scheduler::TaskOutcome::kFailed
                                             : scheduler::TaskOutcome::kSucceeded);
      }
    }
  }

 private:
  enum class Phase : std::uint8_t { kLeasing, kIdle, kSubmitting, kRunning, kEnding };
  void Report(bool failed, int error) {
    phase_ = Phase::kIdle;
    channel_.step_failed.store(failed, std::memory_order_relaxed);
    channel_.step_error.store(error, std::memory_order_relaxed);
    channel_.steps.fetch_add(1, std::memory_order_release);
    channel_.reported.Signal();
  }

  catalog::Closure closure_;
  RequestChannel& channel_;
  catalog::LeaseId lease_;
  Phase phase_ = Phase::kLeasing;
};

// What an acquisition did: the victims it evicted, in order, and how many
// of the closure's extents it found nonresident (and so loaded). Written on
// the scheduler thread; read by the poster once the Done is gone.
struct AcquireReport {
  std::vector<catalog::ExtentId> evicted;
  std::uint64_t loaded = 0;
  std::uint64_t plans = 0;  // materialization plans made
};

// Waits only for already existing page-ins of these typed extent IDs. A
// cancelled AcquireProgram can be gone while its independent page-ins still
// drain. This program never materializes, chooses victims or restarts a load.
// Its gone flag alone is not a payload retirement proof if it was cancelled;
// the owner must also recheck Scheduler::PageInsRetired before releasing its
// source/state/host descriptors. Quarantined loads cannot satisfy that proof.
class DrainPageInsProgram final : public ReportingProgram {
 public:
  DrainPageInsProgram(ProgramDone& done, std::span<const catalog::ExtentId> extents)
      : ReportingProgram(done), extents_(extents) {}

  scheduler::Step Advance(scheduler::TaskContext& context) override {
    const auto ready = context.AwaitPageIns(extents_);
    if (!ready) {
      if (ready.error() == scheduler::WorkError::kBusy) return scheduler::Step::Yield();
      return Fail(ready.error());
    }
    return *ready == scheduler::Readiness::kWaiting
               ? scheduler::Step::Wait()
               : scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
  }

 private:
  // Stable storage belongs to the owner until this program is gone AND every
  // scoped page-in has a positive retirement proof.
  std::span<const catalog::ExtentId> extents_;
};

// Makes room for a closure under the budget B, then materializes it. Each
// round plans the closure's materialization in `domain`
// (memory::PlanMaterialization: the victims eligible for the shortfall,
// never the closure's own extents or `protect`), evicts every victim, each
// eviction waited for, and plans again, until the closure fits. Eligible
// means resident and unleased (Catalog::Evictable), and discardable
// scratch comes first: the node protects its shared workspace, which no
// source can restore, even from a closure that omits it. A closure that
// cannot fit (the eligible victims do not cover the shortfall), or whose
// extents are stale or quarantined, fails with kOverBudget or kUnavailable
// and evicts nothing more. Nothing is leased: a job that follows
// materializes the closure again and takes its own lease (RunProgram).
class AcquireProgram final : public ReportingProgram {
 public:
  // Rounds of planning before giving up: each round's victims cover its
  // shortfall, so one round suffices unless an eviction is abandoned.
  static constexpr std::uint64_t kRounds = 8;

  AcquireProgram(ProgramDone& done, catalog::Closure closure, catalog::DomainId domain,
                 base::Bytes budget, AcquireReport& report,
                 std::vector<catalog::ExtentId> protect = {}, bool select_victims = true)
      : ReportingProgram(done),
        closure_(std::move(closure)),
        domain_(domain),
        budget_(budget),
        report_(report),
        protect_(std::move(protect)),
        select_victims_(select_victims) {}

  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (context.TakeFailure()) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
    }
    while (!materializing_) {
      while (next_ < victims_.size()) {
        const catalog::ExtentId extent = victims_[next_];
        const auto view = context.catalog().Describe(extent);
        if (!view || view->state != catalog::ExtentState::kResident) {
          ++next_;  // gone already (an eviction this program waited for)
          continue;
        }
        const auto evicted = context.Evict(extent);
        if (!evicted) {
          if (evicted.error() == scheduler::WorkError::kBusy) {
            return scheduler::Step::Yield();
          }
          return Fail(evicted.error());
        }
        report_.evicted.push_back(extent);
        ++next_;
        if (*evicted == scheduler::Readiness::kWaiting) {
          return scheduler::Step::Wait();
        }
      }
      const memory::MaterializationPlan plan = memory::PlanMaterialization(
          context.catalog(), domain_, budget_, closure_, protect_, select_victims_);
      ++report_.plans;
      if (!plan.feasible) {
        return Fail(plan.victims.sufficient || plan.shortfall.value() == 0
                        ? scheduler::WorkError::kUnavailable
                        : scheduler::WorkError::kOverBudget);
      }
      if (plan.shortfall.value() == 0) {
        materializing_ = true;
        report_.loaded = plan.load.size();
        break;
      }
      if (report_.plans > kRounds) {
        return Fail(scheduler::WorkError::kOverBudget);
      }
      victims_.clear();
      for (const memory::Victim& victim : plan.victims.victims) {
        victims_.push_back(victim.extent);
      }
      next_ = 0;
    }
    const auto ready = context.Materialize(closure_);
    if (!ready) {
      if (ready.error() == scheduler::WorkError::kBusy) {
        return scheduler::Step::Yield();
      }
      return Fail(ready.error());
    }
    if (*ready == scheduler::Readiness::kWaiting) {
      return scheduler::Step::Wait();
    }
    return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
  }

 private:
  catalog::Closure closure_;
  catalog::DomainId domain_;
  base::Bytes budget_;
  AcquireReport& report_;
  std::vector<catalog::ExtentId> protect_;
  bool select_victims_ = true;
  std::vector<catalog::ExtentId> victims_;
  std::size_t next_ = 0;
  bool materializing_ = false;
};

}  // namespace jitllm::scheduler

#endif  // JITLLM_SCHEDULER_PROGRAMS_H_
