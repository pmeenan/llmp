// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

#include "platform/crash_policy.h"
#include "runtime/commands.h"
#include "runtime/runtime.h"

namespace {

// Before this program's other initializers: non-dumpable early, in case
// something faults before main() installs the handlers. Shared libraries'
// initializers (the NVIDIA driver's libcuda.so.1) and the loader still run
// before it. Global constructors in llmpalooza do no work that can fault
// (architecture.md, layers: nothing at static initialization).
// A failure here is repeated, and reported, by InstallCrashPolicy.
[[gnu::constructor(101)]] void NonDumpableFromTheStart() {
  if (!llmp::platform::MarkNonDumpable()) {
    return;
  }
}

}  // namespace

int main(int argc, char** argv) {
  // First: a crash leaves no core image (D-014).
  if (auto policy = llmp::platform::InstallCrashPolicy("llmp-runtime"); !policy) {
    (void)std::fprintf(stderr, "llmp-runtime: %s\n", policy.error().c_str());
    return llmp::runtime::kExitFailure;
  }
  // A closed standard descriptor would be taken by the next file opened.
  for (int fd = 0; fd <= 2; ++fd) {
    if (::fcntl(fd, F_GETFD) == -1 && errno == EBADF &&
        ::open("/dev/null", O_RDWR | O_NOCTTY) != fd) {
      return llmp::runtime::kExitFailure;
    }
  }
  // The stop signals are waited for, not handled; blocked before any
  // thread starts, so every thread inherits the mask.
  sigset_t stop;
  (void)::sigemptyset(&stop);
  (void)::sigaddset(&stop, SIGTERM);
  (void)::sigaddset(&stop, SIGINT);
  (void)::sigaddset(&stop, SIGHUP);
  (void)::sigaddset(&stop, SIGCHLD);
  (void)::pthread_sigmask(SIG_BLOCK, &stop, nullptr);
  (void)::signal(SIGPIPE, SIG_IGN);
  // The runtime compiles nothing on the GPU (its code is SASS), so the
  // driver has no reason to keep a JIT cache in the service user's home,
  // the data directory. Nothing has started a thread yet.
  (void)::setenv("CUDA_CACHE_DISABLE", "1", 1);  // NOLINT(concurrency-mt-unsafe)
  const std::span<char*> all(argv, static_cast<std::size_t>(argc));
  const std::vector<std::string_view> args(all.begin() + (argc > 0 ? 1 : 0), all.end());
  return llmp::runtime::Run(args, stderr, &llmp::runtime::RunServing);
}
