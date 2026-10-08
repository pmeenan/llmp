// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Readiness events for the runtime's network I/O (docs/portability.md): the
// event loop the chat route's I/O thread runs (runtime/api_server.h), the
// wakers other threads post to it, and the runtime's signals delivered as
// readiness on the node's driver thread (runtime/serve_api.cc).
//
// Readiness, not completion: a registration names the descriptor, a tag
// and what it is waited for (readable, writable, the peer's half-close);
// a wait reports each ready registration's tag and what became true, a
// hang-up and an error included whether asked for or not. Registrations
// are level-triggered. One thread waits on a loop; any thread may signal
// a waker.
//
// Linux implements this with epoll, eventfd and signalfd. macOS would
// implement the same interface with kqueue (EVFILT_READ and EVFILT_WRITE,
// EV_EOF for the half-close, EVFILT_USER for a waker, EVFILT_SIGNAL for
// the signals); Windows with an I/O completion port and the AFD poll
// requests that wepoll and mio use for socket readiness, a posted
// completion for a waker, and a console control handler that posts one for
// the signals. Descriptors are ints here, as on both POSIX systems; a
// Windows port gives them a type of their own.

#ifndef LLMP_PLATFORM_EVENT_LOOP_H_
#define LLMP_PLATFORM_EVENT_LOOP_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <system_error>
#include <utility>

namespace llmp::platform {

// What a registration waits for (kReadable, kWritable, kPeerClosed) and
// what a wait reports (any of them, and kHangUp and kError).
inline constexpr std::uint32_t kReadable = 1U << 0U;
inline constexpr std::uint32_t kWritable = 1U << 1U;
inline constexpr std::uint32_t kPeerClosed = 1U << 2U;  // the peer shut its sending side
inline constexpr std::uint32_t kHangUp = 1U << 3U;      // closed both ways
inline constexpr std::uint32_t kError = 1U << 4U;

struct ReadyEvent {
  std::uint64_t tag = 0;
  std::uint32_t ready = 0;
};

// A descriptor this module opened, closed when it goes.
class OwnedDescriptor {
 public:
  OwnedDescriptor() = default;
  explicit OwnedDescriptor(int fd) : fd_(fd) {}
  OwnedDescriptor(OwnedDescriptor&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  OwnedDescriptor& operator=(OwnedDescriptor&& other) noexcept;
  OwnedDescriptor(const OwnedDescriptor&) = delete;
  OwnedDescriptor& operator=(const OwnedDescriptor&) = delete;
  ~OwnedDescriptor();
  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

class EventLoop {
 public:
  // Not valid() if the system refused one.
  static EventLoop Open();

  bool valid() const { return fd_.valid(); }
  // Watches `fd` for `interest`, reporting `tag`; Change replaces both.
  std::expected<void, std::error_code> Add(int fd, std::uint64_t tag, std::uint32_t interest);
  std::expected<void, std::error_code> Change(int fd, std::uint64_t tag, std::uint32_t interest);
  // Stops watching `fd` (before it is closed).
  void Remove(int fd);
  // Waits at most `timeout` (zero: not at all) for a registration to be
  // ready, and fills `out` from its start with at most 256 of them: how
  // many. A signal's interruption is std::errc::interrupted.
  std::expected<std::size_t, std::error_code> Wait(std::span<ReadyEvent> out,
                                                   std::chrono::milliseconds timeout);

 private:
  OwnedDescriptor fd_;
};

// A wake another thread can post: its descriptor is readable from Signal
// until Drain.
class Waker {
 public:
  // Not valid() if the system refused one.
  static Waker Open();

  bool valid() const { return fd_.valid(); }
  // What an EventLoop or poll watches.
  int descriptor() const { return fd_.get(); }
  void Signal() const;
  void Drain() const;

 private:
  OwnedDescriptor fd_;
};

// Signals delivered as readiness: the descriptor is readable while one of
// the watched signals is pending, and Take returns them one at a time.
// The signals must be blocked in every thread (the runtime blocks its stop
// signals first thing, runtime/main.cc), or they are delivered the usual
// way instead.
class SignalWatch {
 public:
  // Not valid() if the system refused one.
  static SignalWatch Open(std::span<const int> signals);

  bool valid() const { return fd_.valid(); }
  int descriptor() const { return fd_.get(); }
  // The next pending watched signal's number, or 0 when none is pending.
  // Never blocks.
  int Take() const;

 private:
  OwnedDescriptor fd_;
};

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_EVENT_LOOP_H_
