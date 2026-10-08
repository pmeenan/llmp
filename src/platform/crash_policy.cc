// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/crash_policy.h"

#include <fcntl.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstring>
#include <expected>
#include <format>
#include <string>
#include <string_view>

namespace llmp::platform {
namespace {

// Every signal whose default action dumps core (signal(7)).
constexpr std::array kCoreSignals = {SIGABRT, SIGBUS, SIGFPE,  SIGILL,  SIGQUIT,
                                     SIGSEGV, SIGSYS, SIGTRAP, SIGXCPU, SIGXFSZ};

// The handler's line, built before any signal can arrive: it may only use
// async-signal-safe calls.
std::array<char, 96> g_prefix{};
std::size_t g_prefix_size = 0;

// Room for the handler to run after a stack overflow.
constexpr std::size_t kAltStackSize = std::size_t{64} * 1024;

void Append(char* line, std::size_t& at, std::size_t cap, std::string_view text) {
  for (const char c : text) {
    if (at < cap) {
      line[at++] = c;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
  }
}

void OnFatalSignal(int signal) {
  std::array<char, 160> line{};
  std::size_t at = 0;
  Append(line.data(), at, line.size(), std::string_view(g_prefix.data(), g_prefix_size));
  // The signal number, without printf.
  std::array<char, 4> digits{};
  std::size_t count = 0;
  for (int n = signal; n > 0 && count < digits.size(); n /= 10) {
    digits[count++] = static_cast<char>('0' + (n % 10));
  }
  while (count > 0) {
    Append(line.data(), at, line.size(), std::string_view(&digits[--count], 1));
  }
  Append(line.data(), at, line.size(), "; exiting without a core dump\n");
  (void)::write(STDERR_FILENO, line.data(), at);
  ::_exit(kFatalSignalExitBase + signal);
}

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

}  // namespace

std::expected<void, std::string> MarkNonDumpable() {
  // Done already: it clears the filter before it marks the process, and a
  // non-dumpable process's /proc files belong to root.
  if (::prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) == 0) {
    return {};
  }
  // Exclude every mapping from any core image (core(5)). First: a
  // non-dumpable process's /proc files belong to root.
  const int filter = ::open("/proc/self/coredump_filter", O_WRONLY | O_CLOEXEC);
  if (filter < 0 || ::write(filter, "0\n", 2) != 2) {
    const int error = errno;
    if (filter >= 0) {
      (void)::close(filter);
    }
    return std::unexpected("cannot clear /proc/self/coredump_filter: " + Errno(error));
  }
  (void)::close(filter);

  if (::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) {
    return std::unexpected("cannot mark the process non-dumpable: " + Errno(errno));
  }
  return {};
}

std::expected<void, std::string> InstallThreadSignalStack() {
  // Freed never: a thread's handler may run until the thread's very end.
  alignas(16) thread_local std::array<std::byte, kAltStackSize> stack_memory{};
  stack_t stack{};
  stack.ss_sp = stack_memory.data();
  stack.ss_size = stack_memory.size();
  if (::sigaltstack(&stack, nullptr) != 0) {
    return std::unexpected("cannot set the signal stack: " + Errno(errno));
  }
  return {};
}

std::expected<void, std::string> InstallCrashPolicy(std::string_view name) {
  const std::string prefix = std::format("{}: fatal signal ", name.substr(0, 60));
  g_prefix_size = prefix.copy(g_prefix.data(), g_prefix.size());

  if (auto marked = MarkNonDumpable(); !marked) {
    return marked;
  }
  if (auto stack = InstallThreadSignalStack(); !stack) {
    return stack;
  }
  struct sigaction action{};
  action.sa_handler = OnFatalSignal;
  action.sa_flags = SA_ONSTACK;
  (void)::sigfillset(&action.sa_mask);  // nothing interrupts the handler
  for (const int signal : kCoreSignals) {
    if (::sigaction(signal, &action, nullptr) != 0) {
      return std::unexpected(std::format("cannot handle signal {}: {}", signal, Errno(errno)));
    }
  }
  return {};
}

}  // namespace llmp::platform
