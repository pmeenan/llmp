// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <poll.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <expected>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/json.h"
#include "runtime/api_server_internal.h"
#include "scheduler/cohort_policy.h"

namespace llmp::runtime::api {
using server_detail::kListenerTag;
using server_detail::kMaxEmptyPauses;
using server_detail::kStopTag;
using server_detail::kTrimAfterBytes;
using server_detail::kWakeTag;
using server_detail::Readable;
using server_detail::Refusal;
using server_detail::Seconds;
// ---------------------------------------------------------------- driver

void Server::ProcessPendingRelease() {
  std::vector<PendingRelease> releases;
  bool idle = false;
  {
    const std::scoped_lock lock(mutex_);
    if (!ReleaseWaitingLocked()) return;  // none, or their clients all left
    releases.swap(pending_releases_);
    // Idle: nothing runs and nothing queued would start (a request parked
    // until its client reads would not). Queued work is never held back
    // for a release (the starvation bound comes first, D-102): it is
    // answered busy instead.
    idle = running_.empty() && dispatching_ == 0 && queue_.empty() && suspended_requests_ == 0;
  }
  // The backend may still find state held: a continuation parked until its
  // client reads, which may be never; the release does not wait for it.
  auto report = idle ? backend_.ReleaseWorkingSet("http")
                     : std::expected<ReleaseReport, std::string>(std::unexpected("busy"));
  {
    const std::scoped_lock lock(mutex_);
    AnswerReleasesLocked(releases, report);
  }
  WakeIo();
}

void Server::RefusePendingReleases() {
  {
    const std::scoped_lock lock(mutex_);
    if (!ReleaseWaitingLocked()) return;
    std::vector<PendingRelease> releases;
    releases.swap(pending_releases_);
    AnswerReleasesLocked(releases, std::unexpected("busy"));
  }
  WakeIo();
}

void Server::AnswerReleasesLocked(std::span<const PendingRelease> releases,
                                  const std::expected<ReleaseReport, std::string>& report) {
  const bool busy = !report && report.error() == "busy";
  std::string body;
  if (report) {
    std::string quoted_model;
    base::json::AppendQuoted(report->evicted_model, quoted_model);
    body = std::format(
        R"({{"evicted_model":{},"spilled_bytes":{},"reclaimed_bytes":{},"released_backings":{}}})",
        quoted_model, report->spilled_bytes, report->reclaimed_bytes, report->released_backings);
  }
  for (const PendingRelease& release : releases) {
    Channel& ch = *release.channel;
    if (ch.gone) {
      continue;  // its client left while it waited
    }
    if (!report) {
      // Anything other than work in flight is the runtime's own failure
      // to release, reported as such.
      ch.PutError(busy ? server_detail::Refusal(409, "server is busy with active work", "busy")
                       : server_detail::Refusal(500, report.error(), "release_failed"));
    } else {
      ch.PutHead(200, server_detail::kJson, body.size());
      ch.PutBody(body);
      ch.PutEnd();
    }
    if (!ch.dirty) {
      ch.dirty = true;
      dirty_.push_back(ch.connection);
    }
  }
}

void Server::Serve(Pending& pending, int wake_fd, const std::function<bool()>& on_wake) {
  const auto started = Clock::now();
  Channel& channel = *pending.channel;
  std::optional<std::string> recovered;
  bool gone = false;
  {
    const std::scoped_lock lock(mutex_);
    if (dispatching_ > 0) {
      --dispatching_;
    }
    gone = channel.gone;
    if (!gone) {
      running_.push_back(pending.channel);
      // Whatever is queued now waits for this request alone: a paused
      // stream yields its place to it (YieldDueLocked).
      blocked_ = true;
      recovered = BeatLocked(Phase::kStarting, 0);
    }
  }
  if (gone) {
    Log(std::format("request {}: the client left while it was queued", channel.id));
    return;
  }
  Recovered(recovered);
  Stream stream(*this, pending, wake_fd, on_wake, started);
  std::expected<Completion, Error> result;
  if (pending.literal) {
    if (const std::shared_ptr<Yielded> from = std::move(pending.resume); from != nullptr) {
      result = backend_.Resume(*pending.literal, stream, *from);
    } else {
      result = backend_.Complete(*pending.literal, stream);
    }
  } else if (const std::shared_ptr<Yielded> from = std::move(pending.resume); from != nullptr) {
    result = backend_.Resume(pending.request, stream, *from);
  } else {
    result = backend_.Complete(pending.request, stream);
  }
  {
    const std::scoped_lock lock(mutex_);
    blocked_ = false;
  }
  if (result && result->yielded != nullptr) {
    Requeue(pending, stream, std::move(result->yielded));
    Progress(Phase::kIdle, 0);
    return;
  }
  FinishResponse(pending, stream, result, started);
  Progress(Phase::kIdle, 0);
}

bool Server::ReleaseWaitingLocked() {
  // A release whose client left is answered to no one: it neither runs nor
  // keeps a cohort draining for it.
  std::erase_if(pending_releases_, [](const PendingRelease& r) { return r.channel->gone; });
  return !pending_releases_.empty();
}

bool Server::WaitersLocked() const {
  // A pending release is not one: a member that yielded its place to it
  // would keep its state as a continuation, which only leaves the release
  // busy.
  return !queue_.empty() || suspended_requests_ != 0;
}

bool Server::YieldDueLocked(const Channel& channel, Clock::time_point now) const {
  return blocked_ && WaitersLocked() && channel.paused && !channel.gone && !channel.stalled &&
         now - channel.paused_at >= options_.yield_after;
}

void Server::Requeue(Pending& pending, Stream& stream, std::shared_ptr<Yielded> yielded) {
  stream.Carry(pending, true);
  pending.resume = std::move(yielded);
  std::optional<std::string> recovered;
  std::string line;
  {
    const std::scoped_lock lock(mutex_);
    std::erase(running_, pending.channel);
    recovered = BeatLocked(Phase::kFinishing, 0);
    Channel& channel = *pending.channel;
    if (channel.gone || channel.stalled) {
      line = std::format("request {}: {} as it yielded its place", channel.id,
                         channel.gone ? "499, the client left" : "504, the backend stalled");
    } else {
      // Positions and counts only (D-014).
      line = std::format(
          "request {}: its client has read nothing for {:.1f} s while {} requests wait; it "
          "yields its place and continues once its client reads",
          channel.id, Seconds(Clock::now() - channel.paused_at),
          queue_.size() + suspended_requests_);
      if (channel.paused) {
        // Parked until its client reads (Flush, UnparkLocked): taken up
        // while it still could not send, it would only yield again (after
        // its model was swapped back in).
        channel.parked = true;
        parked_.push_back(std::move(pending));
      } else {
        channel.queued = true;
        queue_.push_back(std::move(pending));
        ++enqueued_;
        ready_.Signal();
      }
    }
  }
  Log(line);
  Recovered(recovered);
}

void Server::UnparkLocked(const Channel& channel) {
  if (!channel.parked) {
    return;
  }
  const auto at =
      std::ranges::find_if(parked_, [&](const Pending& p) { return p.channel.get() == &channel; });
  if (at == parked_.end()) {
    return;
  }
  at->channel->parked = false;
  at->channel->queued = true;
  queue_.push_back(std::move(*at));
  parked_.erase(at);
  ++enqueued_;
  ready_.Signal();
}

void Server::Progress(Phase next, double expected) {
  std::optional<std::string> recovered;
  {
    const std::scoped_lock lock(mutex_);
    recovered = BeatLocked(next, expected);
  }
  Recovered(recovered);
}

bool Server::AwaitReaders(const std::function<bool()>& paused, int wake_fd,
                          const std::function<bool()>& on_wake) {
  Phase before = Phase::kDecode;
  {
    const std::scoped_lock lock(mutex_);
    before = watchdog_.health().phase;
  }
  // A pause is not a stall: the watchdog watches nothing until it ends.
  Progress(Phase::kPaused, 0);
  bool stopped = false;
  {
    std::unique_lock lock(mutex_);
    while (paused() && !stopping_) {
      // The runtime's signals are the driver's to read: woken every 100 ms
      // to look, or at once when the I/O thread hands output on or a client
      // leaves.
      (void)drained_.wait_for(lock, std::chrono::milliseconds(100));
      lock.unlock();
      const bool wake = Readable(wake_fd) && on_wake();
      // The driver waits here, so it settles the request memory here too:
      // a body or a parse waiting for it is not held up by a stuck reader.
      memory_->Settle();
      lock.lock();
      if (wake) {
        stopping_ = true;
      }
    }
    stopped = stopping_;
  }
  Progress(before == Phase::kPaused ? Phase::kDecode : before, 0);
  return !stopped;
}

void Server::FinishResponse(Pending& pending, Stream& stream,
                            const std::expected<Completion, Error>& result,
                            Clock::time_point started) {
  Channel& channel = *pending.channel;
  const int status = stream.End(result);
  bool stalled = false;
  std::optional<std::string> recovered;
  {
    const std::scoped_lock lock(mutex_);
    stalled = channel.stalled;
    std::erase(running_, pending.channel);
    // The backend returned: progress, and what follows the response.
    recovered = BeatLocked(Phase::kFinishing, 0);
  }
  Recovered(recovered);
  const Usage& u = stream.usage();
  std::string_view why;
  if (stalled) {
    why = " (the backend made no progress)";
  } else if (status == 499) {
    why = " (the client left)";
  }
  Log(std::format(
      "request {}: {} {}{}, {} prompt tokens ({} cached), {} generated, {:.3f} s "
      "({:.3f} s queued){}",
      channel.id, pending.request.model, status, why, u.prompt_tokens, u.cached_tokens,
      u.completion_tokens, Seconds(Clock::now() - started), Seconds(started - channel.queued_at),
      stream.pauses() == 0
          ? std::string()
          : std::format(", paused {} times for its client to read", stream.pauses())));
  // A large request's many small allocations, freed with it, go back to
  // the system once it is gone (TrimIfOwed).
  trim_owed_ = trim_owed_ || pending.charge.bytes() >= kTrimAfterBytes;
  backend_.AfterResponse();
}

void Server::TrimIfOwed() {
  if (trim_owed_) {
    trim_owed_ = false;
    platform::ReleaseFreeHeap();
  }
}

std::expected<void, std::string> Server::ServeCooperative(Pending first,
                                                          CooperativeBackend& cooperative,
                                                          int wake_fd,
                                                          const std::function<bool()>& on_wake,
                                                          bool substitute) {
  std::vector<std::unique_ptr<Active>> active;
  active.reserve(kMaxActiveRequests);
  std::deque<Pending> suspended;
  using Policy = scheduler::CohortPolicy;
  Policy policy(options_.model_turn, Clock::now(),
                substitute ? Policy::Role::kSubstitute : Policy::Role::kPrimary);
  std::string fatal;
  // The queue as this cohort sees it (CohortPolicy::Look), under mutex_.
  const auto look = [&](std::string_view model) {
    return Policy::Look(
        queue_, model, [](const Pending& p) -> std::string_view { return p.request.model; },
        [&](const Pending& head) {
          return cooperative.Supports({.options = head.request,
                                       .literal = head.literal ? &*head.literal : nullptr,
                                       .resume = head.resume.get()});
        });
  };
  // False means deferred with no effects. The caller still owns the frame
  // and either restores it to the FIFO or uses the serial path alone.
  const auto start = [&](std::unique_ptr<Active>& frame) {
    bool gone = false;
    {
      const std::scoped_lock lock(mutex_);
      if (dispatching_ > 0) {
        --dispatching_;
      }
      gone = frame->pending.channel->gone;
      if (!gone) {
        running_.push_back(frame->pending.channel);
      }
    }
    if (gone) {
      Log(std::format("request {}: the client left while it was queued",
                      frame->pending.channel->id));
      return true;
    }
    Progress(Phase::kStarting, 0);
    auto created = cooperative.Start(frame->request, frame->exchange);
    if (!created) {
      FinishResponse(frame->pending, frame->exchange, std::unexpected(created.error()),
                     frame->started);
      return true;
    }
    if (!*created) {
      const std::scoped_lock lock(mutex_);
      ++dispatching_;
      std::erase(running_, frame->pending.channel);
      return false;
    }
    frame->work = std::move(*created);
    active.push_back(std::move(frame));
    return true;
  };

  auto initial = std::make_unique<Active>(*this, std::move(first), wake_fd, on_wake);
  if (!start(initial)) {
    // A request that cannot join an empty cohort must not wait forever.
    // Start's no-effects contract makes one serial fallback safe.
    // A yielded request's response so far.
    initial->exchange.Carry(initial->pending, initial->pending.resume != nullptr);
    Serve(initial->pending, wake_fd, on_wake);
    Progress(Phase::kIdle, 0);
    return {};
  }
  const auto retire = [&](bool stop) {
    std::erase_if(active, [&](const std::unique_ptr<Active>& frame) {
      // A member yielding its place (Work::Yield) ends at its completed
      // step; it is not cancelled unless the runtime stops or fails.
      const bool yielding = frame->exchange.Yielding();
      const bool go_on = frame->exchange.Continue();
      if ((!go_on && !yielding) || stop || !fatal.empty()) {
        frame->work->Cancel();
      } else if (!frame->work->terminal()) {
        return false;
      }
      Progress(Phase::kFinishing, 0);
      auto retired = cooperative.Retire(*frame->work);
      if (!retired.references_retired) {
        (void)std::fputs("cooperative backend did not retire its references: aborting\n", stderr);
        std::abort();
      }
      if (!fatal.empty()) {
        retired.result = std::unexpected(
            Refusal(500, "the model backend failed; the runtime is stopping", "backend_failed"));
      }
      if (retired.result && retired.result->yielded != nullptr) {
        if (frame->model_paused) {
          frame->exchange.Carry(frame->pending, true);
          frame->pending.resume = std::move(retired.result->yielded);
          {
            const std::scoped_lock lock(mutex_);
            std::erase(running_, frame->pending.channel);
            frame->pending.channel->queued = true;
            ++suspended_requests_;
          }
          suspended.push_back(std::move(frame->pending));
          policy.ContinuationRetired();
          return true;
        }
        Requeue(frame->pending, frame->exchange, std::move(retired.result->yielded));
        return true;
      }
      FinishResponse(frame->pending, frame->exchange, retired.result, frame->started);
      return true;
    });
    TrimIfOwed();
  };
  int empty_pauses = 0;  // consecutive kPaused units with no member paused (M1's race)
  while (!active.empty()) {
    if (Readable(wake_fd) && on_wake()) {
      const std::scoped_lock lock(mutex_);
      stopping_ = true;
    }
    if (!backend_.healthy() && fatal.empty()) {
      fatal = backend_.failure();
      if (fatal.empty()) {
        fatal = "the cooperative backend failed";
      }
    }
    bool stop = false;
    {
      const std::scoped_lock lock(mutex_);
      stop = stopping_;
    }
    // Retire before admitting a replacement. The exchange and parsed
    // inputs survive even after their socket disappeared.
    retire(stop);
    if (!backend_.healthy() && fatal.empty()) {
      fatal = backend_.failure().empty() ? "the cooperative backend failed" : backend_.failure();
    }
    if (active.empty()) {
      break;
    }
    if (stop || !fatal.empty()) {
      continue;
    }

    // Failed or departed starters do not grow active.size(). Bound attempts
    // as well as live works so an endless invalid FIFO cannot starve decode.
    std::size_t attempts = 0;
    std::uint64_t seen = 0;
    bool releasing = false;
    std::expected<Policy::Action, std::string> pass;
    {
      const std::scoped_lock lock(mutex_);
      // Capture arrivals with the queue observation. Taking this later could
      // hide a competing arrival during refill from an all-reader-paused wait;
      // a release that arrives ends such a wait too (the turn's end then
      // bounds the wait, where the release is answered).
      seen = enqueued_;
      releasing = ReleaseWaitingLocked();
      pass = policy.Decide({.point = Policy::Point::kPass,
                            .queue = look(active.front()->pending.request.model),
                            .release_pending = releasing,
                            .now = Clock::now()});
    }
    if (!pass) {
      fatal = pass.error();  // the executor's own fault: the cohort ends
      continue;
    }
    if (pass->release_busy) {
      RefusePendingReleases();  // the turn ended, its members still busy
      releasing = false;
    }
    if (pass->yield_for_switch) {
      for (const auto& frame : active) {
        if (!frame->exchange.Yielding() && frame->work->YieldForSwitch()) {
          frame->model_paused = true;
          frame->exchange.SetYielding();
        }
      }
      policy.SwitchAttempted();
      retire(false);
      if (active.empty()) {
        break;
      }
    }
    for (;;) {
      std::optional<Pending> pending;
      {
        const std::scoped_lock lock(mutex_);
        if (stopping_) {
          break;
        }
        const auto admit = policy.Decide({.point = Policy::Point::kAdmit,
                                          .queue = look(active.front()->pending.request.model),
                                          .active = active.size(),
                                          .capacity = kMaxActiveRequests,
                                          .attempts = attempts,
                                          .release_pending = ReleaseWaitingLocked()});
        if (!admit) {
          fatal = admit.error();
          break;
        }
        if (!admit->admit) {
          break;
        }
        pending.emplace(std::move(queue_.front()));
        queue_.pop_front();
        pending->channel->queued = false;
        ++dispatching_;
      }
      ++attempts;
      auto frame = std::make_unique<Active>(*this, std::move(*pending), wake_fd, on_wake);
      if (!start(frame)) {
        // Restored to the queue's head as it came: its carried response
        // (a yielded request's) goes back with it.
        frame->exchange.Carry(frame->pending, frame->pending.resume != nullptr);
        const std::scoped_lock lock(mutex_);
        if (dispatching_ > 0) {
          --dispatching_;
        }
        frame->pending.channel->queued = true;
        queue_.push_front(std::move(frame->pending));
        policy.Deferred();
        break;
      }
    }
    // Backpressure that blocks others (D-102): the queue waits on this
    // cohort (it must drain for another model, or is full) while a member
    // waits for its client. Such a member yields its place once paused for
    // yield_after (Work::Yield); one that cannot yet goes on.
    std::vector<Active*> yielding;
    Clock::time_point turn_ends;
    {
      const std::scoped_lock lock(mutex_);
      const auto wait = policy.Decide({.point = Policy::Point::kWait,
                                       .active = active.size(),
                                       .capacity = kMaxActiveRequests,
                                       .suspended_elsewhere = suspended_requests_ != 0});
      if (!wait) fatal = wait.error();
      blocked_ = wait && wait->blocks_waiters;
      turn_ends = wait ? wait->wake : Clock::time_point::max();
      const auto now = Clock::now();
      for (const auto& frame : active) {
        if (!frame->exchange.Yielding() && YieldDueLocked(*frame->pending.channel, now)) {
          yielding.push_back(frame.get());
        }
      }
    }
    bool yielded = false;
    for (Active* frame : yielding) {
      if (frame->work->Yield()) {
        frame->exchange.SetYielding();
        yielded = true;
      }
    }
    if (!backend_.healthy() && fatal.empty()) {
      fatal = backend_.failure().empty() ? "the cooperative backend failed" : backend_.failure();
    }
    {
      const std::scoped_lock lock(mutex_);
      stop = stopping_;
    }
    // A new work can already be terminal, and intake may have consumed a
    // shutdown signal. Recheck without starting another admission cycle.
    retire(stop);
    if (active.empty()) {
      break;
    }
    if (stop || !fatal.empty() || yielded) {
      continue;  // a member yielding its place retires first
    }

    // Between completed units: the backend's housekeeping (it leaves every
    // active Work's resources alone).
    backend_.Maintain();
    std::array<CooperativeBackend::Work*, kMaxActiveRequests> work{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      work[i] = active[i]->work.get();
    }
    const std::span<CooperativeBackend::Work* const> wave(work.data(), active.size());
    // Any arrival since this pass observed the head ends a pause's wait,
    // including arrivals during admission, housekeeping or unit selection.
    auto next = cooperative.NextUnit(wave);
    if (!next) {
      fatal = next.error().empty() ? "no cooperative backend unit" : next.error();
      continue;
    }
    if (next->phase == Phase::kPaused) {
      // Every runnable request waits for its client to read (backpressure,
      // D-102): wait until one of them reads or leaves, a request arrives
      // (it may join), one is due to yield its place, or the runtime stops;
      // the next pass then retires, admits, yields or advances.
      std::vector<const Channel*> waiting;
      Clock::time_point yield_due = turn_ends;
      {
        const std::scoped_lock lock(mutex_);
        for (const auto& frame : active) {
          const Channel& ch = *frame->pending.channel;
          if (ch.paused && !ch.gone && !ch.stalled) {
            waiting.push_back(&ch);
            if (blocked_ && WaitersLocked()) {
              yield_due = std::min(yield_due, ch.paused_at + options_.yield_after);
            }
          }
        }
      }
      if (waiting.empty()) {
        // A reader caught up (or left) between the backend's look and this
        // one: only resume. The same with nothing paused, pass after pass,
        // is the backend's fault.
        if (++empty_pauses > kMaxEmptyPauses) {
          fatal = "the cooperative backend keeps pausing while no request waits for its client";
        }
        continue;
      }
      empty_pauses = 0;
      std::ignore = AwaitReaders(
          [this, &waiting, seen, releasing, yield_due] {
            return enqueued_ == seen && pending_releases_.empty() != releasing &&
                   Clock::now() < yield_due &&
                   std::ranges::all_of(
                       waiting,
                       [](const Channel* ch) { return ch->paused && !ch->gone && !ch->stalled; });
          },
          wake_fd, on_wake);
      continue;
    }
    empty_pauses = 0;
    if (next->phase == Phase::kIdle || !std::isfinite(next->expected_seconds) ||
        next->expected_seconds < 0) {
      fatal = "invalid cooperative backend unit";
      continue;
    }
    Progress(next->phase, next->expected_seconds);
    auto advanced = cooperative.Advance(wave);
    // This is an actual unit's return, including failure. Exchange polling
    // and output from its members never renew a hung unit's allowance.
    Progress(Phase::kFinishing, 0);
    if (!advanced) {
      fatal = advanced.error().empty() ? "the cooperative backend unit failed" : advanced.error();
    }
    if (advanced) {
      policy.UnitCompleted(Clock::now(), next->phase == Phase::kSwap ? Policy::Unit::kSwap
                                         : next->advances_request    ? Policy::Unit::kProgress
                                                                     : Policy::Unit::kReadiness);
    }
  }
  if (!suspended.empty()) {
    std::optional<Pending> replacement;
    {
      const std::scoped_lock lock(mutex_);
      // The other model's request may have left while native retirement
      // waited. A fresh same-model request must wait for the originals'
      // held slots, rather than falling back to a serial clear of one; the
      // substitute is the first request for another model, wherever it waits.
      const auto seen = look(suspended.front().request.model);
      const auto end = policy.Decide(
          {.point = Policy::Point::kEnd, .queue = seen, .stopped = stopping_ || !fatal.empty()});
      if (!end) fatal = end.error();
      if (end && end->run_substitute) {
        const auto at = queue_.begin() + static_cast<std::ptrdiff_t>(*seen.other);
        replacement.emplace(std::move(*at));
        replacement->channel->queued = false;
        queue_.erase(at);
        ++dispatching_;
      }
    }
    if (replacement) {
      if (cooperative.Supports({.options = replacement->request,
                                .literal = replacement->literal ? &*replacement->literal : nullptr,
                                .resume = replacement->resume.get()})) {
        auto ran = ServeCooperative(std::move(*replacement), cooperative, wake_fd, on_wake, true);
        if (!ran) {
          fatal = ran.error();
        }
      } else {
        Serve(*replacement, wake_fd, on_wake);
      }
    }
    // Ownership moves only after every native reference retired. The policy
    // chooses ordering, including continuations' precedence over fresh peers.
    {
      const std::scoped_lock lock(mutex_);
      const auto count = suspended.size();
      policy.Restore(queue_, suspended, [](const Pending& pending) -> std::string_view {
        return pending.request.model;
      });
      suspended_requests_ -= count;
      enqueued_ += count;
      ready_.Signal();
    }
  }
  Progress(Phase::kIdle, 0);
  return fatal.empty() ? std::expected<void, std::string>{} : std::unexpected(std::move(fatal));
}

std::expected<void, std::string> Server::Run(int wake_fd, const std::function<bool()>& on_wake) {
  if (listeners_.empty()) {
    return std::unexpected("the chat route is not listening");
  }
  std::expected<void, std::error_code> added =
      loop_.Add(stop_.descriptor(), kStopTag, platform::kReadable);
  added = added ? loop_.Add(io_wake_.descriptor(), kWakeTag, platform::kReadable) : added;
  for (std::size_t i = 0; i < listeners_.size(); ++i) {
    added = added ? loop_.Add(listeners_[i].get(), kListenerTag | i, platform::kReadable) : added;
  }
  if (!added) {
    return std::unexpected(std::format("the event loop: {}", added.error().value()));
  }
  parser_ = std::jthread([this](const std::stop_token& stop) { ParseLoop(stop); });
  io_ = std::jthread([this] { Loop(); });
  CooperativeBackend* const cooperative = backend_.cooperative();
  std::expected<void, std::string> result;
  for (;;) {
    {
      const std::scoped_lock lock(mutex_);
      if (stopping_) {
        break;
      }
    }
    // Idle, the backend's housekeeping runs between waits (Backend::Maintain:
    // retention, the spill budget, memory pressure from outside).
    backend_.Maintain();
    std::array<pollfd, 2> fds{{{.fd = wake_fd, .events = POLLIN, .revents = 0},
                               {.fd = ready_.descriptor(), .events = POLLIN, .revents = 0}}};
    const int n = ::poll(fds.data(), fds.size(), static_cast<int>(kMaintenanceMs));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      result = std::unexpected(std::format("poll: {}", errno));
      break;
    }
    if ((fds[0].revents & POLLIN) != 0 && on_wake()) {
      break;
    }
    if ((fds[1].revents & POLLIN) == 0) {
      continue;
    }
    ready_.Drain();
    ProcessPendingRelease();
    for (;;) {
      ProcessPendingRelease();
      std::optional<Pending> next;
      {
        const std::scoped_lock lock(mutex_);
        if (stopping_ || queue_.empty()) {
          break;
        }
        next.emplace(std::move(queue_.front()));
        next->channel->queued = false;
        queue_.pop_front();
        ++dispatching_;
      }
      if (cooperative != nullptr &&
          cooperative->Supports({.options = next->request,
                                 .literal = next->literal ? &*next->literal : nullptr,
                                 .resume = next->resume.get()})) {
        result = ServeCooperative(std::move(*next), *cooperative, wake_fd, on_wake);
      } else {
        Serve(*next, wake_fd, on_wake);
      }
      next.reset();
      ProcessPendingRelease();
      TrimIfOwed();
      if (!result) {
        break;
      }
      if (!backend_.healthy()) {
        result = std::unexpected(backend_.failure());
        break;
      }
      if (Readable(wake_fd) && on_wake()) {
        const std::scoped_lock lock(mutex_);
        stopping_ = true;
      }
    }
    if (!result) {
      break;
    }
  }
  // Whatever still waits is refused; the I/O thread writes it out.
  std::vector<std::string> lines;
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
    for (Pending& p : parked_) {
      p.channel->parked = false;
      queue_.push_back(std::move(p));
    }
    parked_.clear();
    for (Pending& p : queue_) {
      Channel& channel = *p.channel;
      channel.queued = false;
      channel.keep_alive = false;
      lines.push_back(std::format("request {}: 503, the runtime is stopping", channel.id));
      channel.PutError(Refusal(503, "the runtime is stopping"));
      if (!channel.dirty) {
        channel.dirty = true;
        dirty_.push_back(channel.connection);
      }
    }
    for (const PendingRelease& release : pending_releases_) {
      Channel& channel = *release.channel;
      channel.keep_alive = false;
      lines.push_back(std::format("request {}: 503, the runtime is stopping", channel.id));
      channel.PutError(server_detail::Refusal(503, "the runtime is stopping"));
      if (!channel.dirty) {
        channel.dirty = true;
        dirty_.push_back(channel.connection);
      }
    }
    pending_releases_.clear();
    queue_.clear();
  }
  for (const std::string& line : lines) {
    Log(line);
  }
  WakeIo();
  stop_.Signal();
  io_.join();
  parser_.request_stop();
  parser_.join();
  return result;
}

}  // namespace llmp::runtime::api
