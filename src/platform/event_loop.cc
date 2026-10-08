// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/event_loop.h"

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/signalfd.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <system_error>
#include <utility>

namespace llmp::platform {
namespace {

constexpr std::size_t kMostEvents = 256;

std::unexpected<std::error_code> Errno() {
  return std::unexpected(std::error_code(errno, std::generic_category()));
}

std::uint32_t ToEpoll(std::uint32_t interest) {
  std::uint32_t events = 0;
  events |= (interest & kReadable) != 0 ? static_cast<std::uint32_t>(EPOLLIN) : 0U;
  events |= (interest & kWritable) != 0 ? static_cast<std::uint32_t>(EPOLLOUT) : 0U;
  events |= (interest & kPeerClosed) != 0 ? static_cast<std::uint32_t>(EPOLLRDHUP) : 0U;
  return events;
}

std::uint32_t FromEpoll(std::uint32_t events) {
  std::uint32_t ready = 0;
  ready |= (events & EPOLLIN) != 0 ? kReadable : 0U;
  ready |= (events & EPOLLOUT) != 0 ? kWritable : 0U;
  ready |= (events & EPOLLRDHUP) != 0 ? kPeerClosed : 0U;
  ready |= (events & EPOLLHUP) != 0 ? kHangUp : 0U;
  ready |= (events & EPOLLERR) != 0 ? kError : 0U;
  return ready;
}

}  // namespace

OwnedDescriptor& OwnedDescriptor::operator=(OwnedDescriptor&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

OwnedDescriptor::~OwnedDescriptor() {
  if (fd_ >= 0) {
    (void)::close(fd_);
  }
}

// ---------------------------------------------------------------- EventLoop

EventLoop EventLoop::Open() {
  EventLoop loop;
  loop.fd_ = OwnedDescriptor(::epoll_create1(EPOLL_CLOEXEC));
  return loop;
}

std::expected<void, std::error_code> EventLoop::Add(int fd, std::uint64_t tag,
                                                    std::uint32_t interest) {
  epoll_event event{.events = ToEpoll(interest), .data = {.u64 = tag}};
  if (::epoll_ctl(fd_.get(), EPOLL_CTL_ADD, fd, &event) != 0) {
    return Errno();
  }
  return {};
}

std::expected<void, std::error_code> EventLoop::Change(int fd, std::uint64_t tag,
                                                       std::uint32_t interest) {
  epoll_event event{.events = ToEpoll(interest), .data = {.u64 = tag}};
  if (::epoll_ctl(fd_.get(), EPOLL_CTL_MOD, fd, &event) != 0) {
    return Errno();
  }
  return {};
}

void EventLoop::Remove(int fd) { (void)::epoll_ctl(fd_.get(), EPOLL_CTL_DEL, fd, nullptr); }

std::expected<std::size_t, std::error_code> EventLoop::Wait(std::span<ReadyEvent> out,
                                                            std::chrono::milliseconds timeout) {
  std::array<epoll_event, kMostEvents> events{};
  const auto most = static_cast<int>(std::min(out.size(), events.size()));
  const auto wait = static_cast<int>(std::clamp<std::chrono::milliseconds::rep>(
      timeout.count(), 0, std::numeric_limits<int>::max()));
  const int n = ::epoll_wait(fd_.get(), events.data(), most, wait);
  if (n < 0) {
    return Errno();
  }
  const auto ready = static_cast<std::size_t>(n);
  for (std::size_t i = 0; i < ready; ++i) {
    out[i] = ReadyEvent{.tag = events[i].data.u64, .ready = FromEpoll(events[i].events)};
  }
  return ready;
}

// ---------------------------------------------------------------- Waker

Waker Waker::Open() {
  Waker waker;
  waker.fd_ = OwnedDescriptor(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK));
  return waker;
}

void Waker::Signal() const {
  const std::uint64_t one = 1;
  (void)!::write(fd_.get(), &one, sizeof one);
}

void Waker::Drain() const {
  std::uint64_t count = 0;
  (void)!::read(fd_.get(), &count, sizeof count);
}

// ---------------------------------------------------------------- SignalWatch

SignalWatch SignalWatch::Open(std::span<const int> signals) {
  sigset_t set{};
  (void)::sigemptyset(&set);
  for (const int s : signals) {
    (void)::sigaddset(&set, s);
  }
  SignalWatch watch;
  watch.fd_ = OwnedDescriptor(::signalfd(-1, &set, SFD_CLOEXEC | SFD_NONBLOCK));
  return watch;
}

int SignalWatch::Take() const {
  signalfd_siginfo info{};
  if (::read(fd_.get(), &info, sizeof info) != static_cast<ssize_t>(sizeof info)) {
    return 0;
  }
  return static_cast<int>(info.ssi_signo);
}

}  // namespace llmp::platform
