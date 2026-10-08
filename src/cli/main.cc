// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <span>
#include <string_view>
#include <vector>

#include "cli/cli.h"

int main(int argc, char** argv) {
  // A closed standard descriptor would be taken by the next file opened
  // (such as by the CUDA driver) and written to as output. Hold it with
  // /dev/null opened read-only, so writing there still fails (exit 1) and
  // reading finds nothing. open() returns the lowest free descriptor.
  for (int fd = 0; fd <= 2; ++fd) {
    if (::fcntl(fd, F_GETFD) == -1 && errno == EBADF &&
        ::open("/dev/null", O_RDONLY | O_NOCTTY) != fd) {
      return llmp::cli::kExitFailure;
    }
  }
  // A closed pipe is a failed write (exit 1), not death by SIGPIPE.
  (void)std::signal(SIGPIPE, SIG_IGN);
  const std::span<char*> all(argv, static_cast<std::size_t>(argc));
  const std::vector<std::string_view> args(all.begin() + (argc > 0 ? 1 : 0), all.end());
  return llmp::cli::Run(args, stdout, stderr);
}
