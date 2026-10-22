// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "runtime/api_server_internal.h"

namespace llmp::runtime::api {
using server_detail::kJson;
using server_detail::Readable;
using server_detail::Refusal;
Server::Stream::Stream(Server& server, Pending& pending, int wake_fd,
                       const std::function<bool()>& on_wake, Clock::time_point started,
                       bool cooperative)
    : server_(server),
      pending_(pending),
      channel_(*pending.channel),
      wake_fd_(wake_fd),
      on_wake_(on_wake),
      started_(pending.started.value_or(started)),
      text_(pending.text ? std::move(*pending.text) : OutputText(pending.request.stop)),
      usage_(pending.usage),
      reasoning_(std::move(pending.reasoning)),
      content_(std::move(pending.content)),
      text_charge_(std::move(pending.text_charge)),
      pauses_(pending.pauses),
      cooperative_(cooperative) {
  pending.text.reset();
  if (!pending_.request.stream && server_.options_.deadline_cap) {
    // Set before any work, so a configured deadline ends a long rendering
    // too; Admit then scales it to the request's work, within the cap.
    deadline_ = started_ + *server_.options_.deadline_cap;
  }
}

bool Server::Stream::Admit(const Admission& admission) {
  usage_.prompt_tokens = admission.prompt_tokens;
  floors_ = admission.floors;
  if (!pending_.request.stream && server_.options_.deadline_cap) {
    // A stream has no deadline, nor by default a non-streaming request
    // (D-102); with a cap configured, its work's, at most the cap.
    deadline_ = started_ + ScaledDeadline(server_.options_.stall, *server_.options_.deadline_cap,
                                          floors_, admission.swap_bytes, admission.prompt_tokens,
                                          admission.max_tokens);
  }
  if (pending_.request.stream) {
    const std::scoped_lock lock(server_.mutex_);
    if (!channel_.gone && !channel_.stalled && !channel_.ended) {
      channel_.PutStart(usage_);
      Dirty();
    }
  }
  server_.WakeIo();
  return Check(std::nullopt, 0);
}

bool Server::Stream::Next(Phase phase, std::uint64_t size) {
  return Check(phase, ExpectedSeconds(phase, size, floors_));
}

bool Server::Stream::Reasoning(std::string_view text) {
  Emit(text_.Reasoning(text));
  return Backpressure() && Check(std::nullopt, 0);
}

bool Server::Stream::Content(std::string_view text) {
  Emit(text_.Content(text));
  if (text_.stopped()) {
    Progress(std::nullopt, 0);  // the answer is whole: nothing else to ask
    return false;
  }
  return Backpressure() && Check(std::nullopt, 0);
}

bool Server::Stream::Continue() { return Backpressure() && Check(std::nullopt, 0); }

bool Server::Stream::Paused() const {
  const std::scoped_lock lock(server_.mutex_);
  return PausedLocked();
}

bool Server::Stream::Yielding() const { return yielding_; }

void Server::Stream::SetYielding() { yielding_ = true; }

void Server::Stream::Carry(Pending& pending, bool taken_up) {
  pending.text = std::move(text_);
  pending.usage = usage_;
  pending.reasoning = std::move(reasoning_);
  pending.content = std::move(content_);
  pending.text_charge = std::move(text_charge_);
  pending.pauses = pauses_;
  if (taken_up) {
    pending.started = started_;
  }
}

int Server::Stream::End(const std::expected<Completion, Error>& result) {
  {
    const std::scoped_lock lock(server_.mutex_);
    if (channel_.encoding_error) {
      return channel_.encoding_error->status;
    }
    if (channel_.stalled || channel_.gone) {
      return channel_.stalled ? 504 : 499;
    }
  }
  // A non-streaming body (large literal score arrays among them) is the
  // driver's: made without the shared I/O lock, so another connection can
  // still disconnect or read, and charged to the request memory by its
  // allocation, which follows it to the socket.
  std::optional<std::expected<std::string, Error>> body;
  MemoryCharge body_charge;
  if (!pending_.request.stream && result && !timed_out_ && !stopping_) {
    EmitLocked(text_.Finish());  // non-streaming: only local strings
    usage_.completion_tokens = result->completion_tokens;
    usage_.cached_tokens = result->cached_tokens;
    const Finish finish = result->stopped || text_.stopped() ? Finish::kStop : Finish::kLength;
    // The body's most: what the request memory could hold besides it.
    const std::uint64_t most = server_.memory_->capacity();
    body =
        pending_.literal
            ? LiteralCompletionJson(channel_.id, channel_.created, *pending_.literal, content_,
                                    result->literal, finish, usage_, static_cast<std::size_t>(most))
            : std::expected<std::string, Error>(CompletionJson(
                  channel_.id, channel_.created, pending_.request.model, content_,
                  reasoning_.empty() ? std::nullopt : std::optional<std::string>(reasoning_),
                  finish, usage_));
    std::string().swap(content_);
    std::string().swap(reasoning_);
    text_charge_.Reset();
    // Charged as it goes to the socket (on the driver, the request memory
    // grows for it); one that cannot be held beside what slow readers
    // keep is refused (503), so they cannot retain responses without
    // bound.
    if (*body && !body_charge.Add(*server_.memory_, (*body)->capacity())) {
      body = std::unexpected(
          MemoryRefusal(*server_.memory_, (*body)->capacity(), "the completed response"));
    }
    timed_out_ = timed_out_ || Clock::now() > deadline_;
  }
  int status = 200;
  {
    const std::scoped_lock lock(server_.mutex_);
    if (channel_.stalled || channel_.gone) {
      // The watchdog answered it (a 504), or the client closed the
      // request or stopped reading (499, logged only).
      return channel_.stalled ? 504 : 499;
    }
    if (!result) {
      channel_.PutError(result.error());
      status = result.error().status;
    } else {
      if (pending_.request.stream) {
        EmitLocked(text_.Finish());
      }
      usage_.completion_tokens = result->completion_tokens;
      usage_.cached_tokens = result->cached_tokens;
      if (timed_out_ || stopping_) {
        const Error error =
            timed_out_ ? Refusal(504, std::format("the request passed its {} s deadline",
                                                  std::chrono::duration_cast<std::chrono::seconds>(
                                                      deadline_ - started_)
                                                      .count()))
                       : Refusal(503, "the runtime is stopping");
        if (stopping_) {
          channel_.keep_alive = false;
        }
        channel_.PutError(error);
        status = error.status;
      } else {
        const Finish finish = result->stopped || text_.stopped() ? Finish::kStop : Finish::kLength;
        if (pending_.request.stream) {
          if (!channel_.head_sent) {
            channel_.PutStart();
          }
          channel_.PutFinish(finish, usage_, pending_.request.include_usage);
          if (channel_.encoding_error) {
            status = channel_.encoding_error->status;
          }
        } else if (!body || !*body) {
          const Error error =
              body ? body->error() : Refusal(500, "the response was not made", "internal");
          channel_.PutError(error);
          status = error.status;
        } else {
          channel_.PutHead(200, kJson, (*body)->size());
          channel_.response = std::move(body_charge);
          channel_.body = std::move(**body);
          channel_.PutEnd();
        }
      }
    }
    Dirty();
  }
  server_.WakeIo();
  return status;
}

const Usage& Server::Stream::usage() const { return usage_; }

std::uint64_t Server::Stream::pauses() const { return pauses_; }

void Server::Stream::Dirty() {
  if (!channel_.dirty) {
    channel_.dirty = true;
    server_.dirty_.push_back(channel_.connection);
  }
}

void Server::Stream::EmitLocked(const OutputText::Out& out) {
  if (!pending_.request.stream) {
    reasoning_ += out.reasoning;
    content_ += out.content;
    // The text so far, charged by its allocation as it grows (bounded by
    // the context's tokens). Already held: past the grant it is charged
    // anyway, for the driver to grow it, and later charges wait.
    const std::uint64_t held = reasoning_.capacity() + content_.capacity();
    if (held > text_charge_.bytes()) {
      text_charge_.Force(*server_.memory_, held - text_charge_.bytes());
    }
    return;
  }
  if (channel_.gone || channel_.stalled || channel_.ended ||
      (out.reasoning.empty() && out.content.empty())) {
    return;
  }
  if (!channel_.head_sent) {
    channel_.PutStart();
  }
  if (!out.reasoning.empty()) {
    channel_.PutText(StreamText::kReasoning, out.reasoning);
  }
  if (!out.content.empty()) {
    channel_.PutText(StreamText::kContent, out.content);
  }
  // The unread output, charged as it grows (within the grant: this is
  // the middle of a unit, where the driver does not reclaim). What the
  // request memory cannot take now is still held, charged past the grant
  // for the driver to grow it, and pauses the request like a reader
  // behind.
  bool over = false;
  if (const std::uint64_t held = channel_.out.capacity(); held > channel_.unread.bytes()) {
    const std::uint64_t more = held - channel_.unread.bytes();
    if (!channel_.unread.Add(*server_.memory_, more, true)) {
      channel_.unread.Force(*server_.memory_, more);
      over = true;
    }
  }
  if (over || channel_.out.size() > server_.options_.intake.stream_buffer) {
    // A reader this far behind gets backpressure (D-102): the request
    // pauses at this completed step until the I/O thread has handed the
    // buffer to the socket (Server::Flush clears it), instead of the
    // buffer growing or the client being dropped.
    if (!channel_.paused) {
      channel_.paused = true;
      channel_.paused_at = Clock::now();
      ++pauses_;
    }
  }
  Dirty();
}

bool Server::Stream::PausedLocked() const {
  return channel_.paused && !channel_.gone && !channel_.stalled;
}

bool Server::Stream::Backpressure() {
  if (yielding_) {
    return false;
  }
  if (cooperative_ || !pending_.request.stream) {
    return true;
  }
  {
    const std::scoped_lock lock(server_.mutex_);
    if (!PausedLocked()) {
      return true;
    }
  }
  const bool go_on = server_.AwaitReaders(
      [this] { return PausedLocked() && !server_.YieldDueLocked(channel_, Clock::now()); },
      wake_fd_, on_wake_);
  if (!go_on) {
    stopping_ = true;
    return false;
  }
  const std::scoped_lock lock(server_.mutex_);
  if (PausedLocked() && server_.YieldDueLocked(channel_, Clock::now())) {
    yielding_ = true;
    return false;
  }
  return true;
}

void Server::Stream::Emit(const OutputText::Out& out) {
  if (!pending_.request.stream) {
    EmitLocked(out);  // local text only
    return;
  }
  {
    const std::scoped_lock lock(server_.mutex_);
    EmitLocked(out);
  }
  server_.WakeIo();
}

void Server::Stream::Progress(std::optional<Phase> next, double expected) {
  if (cooperative_) {
    return;  // the driver declares and completes units, not channel polling
  }
  std::optional<std::string> recovered;
  {
    const std::scoped_lock lock(server_.mutex_);
    recovered = server_.BeatLocked(next.value_or(server_.watchdog_.health().phase), expected);
  }
  server_.Recovered(recovered);
}

bool Server::Stream::Check(std::optional<Phase> next, double expected) {
  Progress(next, expected);
  if (!cooperative_) {
    // Between a serial request's steps the driver settles the request
    // memory (a cohort's settles between its units, Backend::Maintain),
    // so bodies and parses waiting for it do not wait for the request.
    server_.memory_->Settle();
  }
  if (timed_out_ || stopping_ || yielding_) {
    return false;
  }
  {
    const std::scoped_lock lock(server_.mutex_);
    stopping_ = stopping_ || server_.stopping_;
    if (stopping_ || channel_.gone || channel_.stalled || channel_.ended) {
      return false;
    }
  }
  if (Clock::now() > deadline_) {
    timed_out_ = true;
    return false;
  }
  if (Readable(wake_fd_) && on_wake_()) {
    stopping_ = true;
    const std::scoped_lock lock(server_.mutex_);
    server_.stopping_ = true;
    return false;
  }
  return true;
}

}  // namespace llmp::runtime::api
