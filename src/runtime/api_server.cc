// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/api_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "base/report.h"
#include "base/surface_versions.h"
#include "platform/crash_policy.h"
#include "platform/event_loop.h"
#include "platform/memory_pressure.h"
#include "platform/sockets.h"
#include "runtime/api_server_internal.h"

namespace llmp::runtime::api {
namespace server_detail {
// Empties a buffer, giving back its allocation when that is large.
void Release(std::string& buffer) {
  if (buffer.capacity() > kKeptBufferBytes) {
    std::string().swap(buffer);
  } else {
    buffer.clear();
  }
}

Error Refusal(int status, std::string message, std::string code) {
  std::string type = "invalid_request_error";
  if (status == 429) {
    type = "rate_limit_error";
  } else if (status == 403) {
    type = "permission_error";
  } else if (status == 409) {
    type = "conflict_error";
  } else if (status >= 500) {
    type = "server_error";
  }
  return Error{.status = status,
               .type = std::move(type),
               .message = std::move(message),
               .param = {},
               .code = std::move(code)};
}

// While the backend is unhealthy (watchdog.h): what is queued, and what
// arrives, until its next progress.
Error Unresponsive() {
  return Refusal(503, "the model backend has stopped making progress; retry later",
                 "backend_unresponsive");
}

// The running request's end when its backend stalled, `idle` after its
// last progress.
Error Stalled(Clock::duration idle) {
  return Refusal(504,
                 std::format("the model backend made no progress for {} s",
                             std::chrono::duration_cast<std::chrono::seconds>(idle).count()),
                 "backend_stalled");
}

bool Readable(int fd) {
  pollfd p{.fd = fd, .events = POLLIN, .revents = 0};
  return ::poll(&p, 1, 0) > 0 && (p.revents & POLLIN) != 0;
}

std::string RequestId() {
  std::array<std::uint8_t, 12> bytes{};
  if (!platform::FillRandom(std::as_writable_bytes(std::span(bytes)))) {
    // Unique within the process is enough for an opaque ID.
    static std::atomic<std::uint64_t> counter{0};
    const std::uint64_t n = ++counter;
    for (std::size_t i = 0; i < 8; ++i) {
      bytes[i] = static_cast<std::uint8_t>(n >> (8 * i));
    }
  }
  std::string id = "chatcmpl-";
  for (const std::uint8_t b : bytes) {
    id += std::format("{:02x}", b);
  }
  return id;
}

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

bool PeerIsLoopback(const sockaddr_storage& peer) {
  if (peer.ss_family == AF_INET) {
    sockaddr_in four{};
    std::memcpy(&four, &peer, sizeof four);
    return (ntohl(four.sin_addr.s_addr) >> 24U) == 127U;
  }
  if (peer.ss_family == AF_INET6) {
    sockaddr_in6 six{};
    std::memcpy(&six, &peer, sizeof six);
    if (std::memcmp(&six.sin6_addr, &in6addr_loopback, sizeof six.sin6_addr) == 0) {
      return true;
    }
    return IN6_IS_ADDR_V4MAPPED(&six.sin6_addr) && six.sin6_addr.s6_addr[12] == 127;
  }
  return false;
}

}  // namespace server_detail
using server_detail::PeerIsLoopback;
using server_detail::Readable;
using server_detail::Refusal;
using server_detail::Release;
using server_detail::RequestId;
using server_detail::Seconds;
using server_detail::Stalled;
using server_detail::Unresponsive;
// ---------------------------------------------------------------- Server

Server::Server(Backend& backend, ServerOptions options)
    : backend_(backend),
      options_(std::move(options)),
      memory_(options_.memory != nullptr
                  ? options_.memory
                  : std::make_shared<RequestMemory>(options_.intake.request_capacity,
                                                    options_.intake.request_capacity)),
      models_(backend.Models()),
      created_(static_cast<std::int64_t>(std::time(nullptr))),
      loop_(platform::EventLoop::Open()),
      stop_(platform::Waker::Open()),
      io_wake_(platform::Waker::Open()),
      ready_(platform::Waker::Open()),
      watchdog_(options_.stall, Clock::now()) {
  if (options_.ladder != nullptr) {
    ladder_ = options_.ladder;
    return;
  }
  // A ladder of the server's own (the tests'): `hang`, or its default, and
  // `on_hang` its last resort.
  own_ladder_ = std::make_unique<HangLadder>(options_.hang.value_or(DefaultHang(options_.stall)),
                                             Clock::now());
  own_ladder_->set_log([this](std::string_view line) { Log(line); });
  if (options_.on_hang) {
    own_ladder_->set_last_resort(options_.on_hang);
  }
  ladder_ = own_ladder_.get();
}

std::chrono::milliseconds DefaultHang(std::chrono::milliseconds stall) {
  return std::max<std::chrono::milliseconds>(kHangFloor, kHangStalls * stall);
}

Server::~Server() {
  if (io_.joinable()) {
    stop_.Signal();
    io_.join();
  }
  if (parser_.joinable()) {
    parser_.request_stop();
    parser_.join();
  }
}

std::expected<std::vector<std::uint16_t>, std::string> Server::Listen() {
  if (!loop_.valid() || !stop_.valid() || !io_wake_.valid() || !ready_.valid()) {
    return std::unexpected("the event loop or a waker failed");
  }
  if (options_.bind.empty()) {
    return std::unexpected("there is nothing to listen on");
  }
  std::vector<std::uint16_t> ports;
  for (const config::ClientEndpoint& endpoint : options_.bind) {
    auto listener = http::Listen(endpoint);
    if (!listener) {
      listeners_.clear();
      return std::unexpected(listener.error());
    }
    listeners_.push_back(std::move(listener->fd));
    ports.push_back(listener->port);
    options_.hosts.AddPort(listener->port);
  }
  return ports;
}

void Server::Log(std::string_view line) const {
  if (options_.log == nullptr) {
    return;
  }
  const std::string text = std::format("llmp-runtime: {}\n", line);
  (void)std::fwrite(text.data(), 1, text.size(), options_.log);
  (void)std::fflush(options_.log);
}

void Server::WakeIo() const { io_wake_.Signal(); }

Health Server::health() const {
  const std::scoped_lock lock(mutex_);
  return watchdog_.health();
}

void Server::HealthChanged() const {
  if (!options_.on_health) {
    return;
  }
  // Serialized, each with the health as it is then: the last report is
  // the current state whichever thread noticed the change last.
  const std::scoped_lock lock(health_mutex_);
  options_.on_health(health());
}

std::optional<std::string> Server::BeatLocked(Phase next, double expected) {
  const auto now = Clock::now();
  const Health before = watchdog_.health();
  // The hang ladder (hang_ladder.h) sees the same beat: the unit that
  // follows and the allowance the watchdog gives it.
  ladder_->Unit(next != Phase::kIdle && next != Phase::kPaused,
                Allowance(watchdog_.stall(), expected), PhaseName(next), now);
  if (!watchdog_.Beat(next, expected, now)) {
    return std::nullopt;
  }
  return std::format(
      "the model backend made progress again ({}), {:.1f} s after its last; requests are "
      "accepted again",
      PhaseName(before.phase), Seconds(now - before.last_progress));
}

void Server::Recovered(const std::optional<std::string>& line) const {
  if (line) {
    Log(*line);
    HealthChanged();
  }
}

// ---------------------------------------------------------------- I/O thread

void Server::WatchBackend(Clock::time_point now, std::vector<std::uint64_t>& ended) {
  std::vector<std::string> lines;  // logged once mutex_ is released
  {
    const std::scoped_lock lock(mutex_);
    if (!watchdog_.Check(now)) {
      return;
    }
    const Health& health = watchdog_.health();
    const Clock::duration idle = now - health.last_progress;
    if (!options_.stall_fails) {
      // Detection is not a limit (D-102): reported, while the requests go
      // on (a hang that never returns is the node's to end).
      lines.push_back(std::format(
          "the model backend made no progress for {:.1f} s ({}; stall {}); it is reported "
          "unhealthy until it makes progress; its {} active and {} queued requests go on",
          Seconds(idle), PhaseName(health.phase), health.stalls, running_.size(), queue_.size()));
    } else {
      FailStalledLocked(idle, ended, lines);
    }
  }
  for (const std::string& line : lines) {
    Log(line);
  }
  HealthChanged();
}

void Server::WatchHang(Clock::time_point now) {
  // Progress of any kind: a unit's beat (the ladder sees each, BeatLocked)
  // or the backend's own activity moving (the node's progress count). The
  // ladder logs each rung it enters; entering rung 3 runs its last resort
  // here, on the I/O thread, which never waits on the model.
  if (options_.activity) {
    ladder_->Activity(options_.activity(), now);
  }
  (void)ladder_->Check(now);
}

void Server::FailStalledLocked(Clock::duration idle, std::vector<std::uint64_t>& ended,
                               std::vector<std::string>& lines) {
  const Health& health = watchdog_.health();
  {
    std::string which = "no request";
    for (const auto& running : running_) {
      if (running->ended || running->gone) {
        continue;
      }
      // Ended as a client that leaves ends it: the generation stops at the
      // backend's next step, and the lease is released as it returns.
      Channel& ch = *running;
      which = std::format("request {} ends with 504", ch.id);
      ch.PutError(Stalled(idle));
      ch.stalled = true;  // not `gone`: the connection stays to carry the error
      ended.push_back(ch.connection);
    }
    lines.push_back(
        std::format("the model backend made no progress for {:.1f} s ({}; stall {}): {}; it is "
                    "unhealthy, and requests get 503 until it makes progress",
                    Seconds(idle), PhaseName(health.phase), health.stalls, which));
    // Nothing queued (or parked) would run before the backend moves again.
    for (Pending& p : parked_) {
      p.channel->parked = false;
      queue_.push_back(std::move(p));
    }
    parked_.clear();
    for (Pending& p : queue_) {
      Channel& ch = *p.channel;
      ch.queued = false;
      lines.push_back(std::format("request {}: 503, the backend is not making progress", ch.id));
      ch.PutError(Unresponsive());
      ended.push_back(ch.connection);
    }
    queue_.clear();
    for (const PendingRelease& release : pending_releases_) {
      Channel& ch = *release.channel;
      lines.push_back(std::format("request {}: 503, the backend is not making progress", ch.id));
      ch.PutError(Unresponsive());
      ended.push_back(ch.connection);
    }
    pending_releases_.clear();
  }
  drained_.notify_all();  // a paused request ends too
}

}  // namespace llmp::runtime::api
