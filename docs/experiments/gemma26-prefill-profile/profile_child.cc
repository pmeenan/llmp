// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Manual observer launcher: COMPLETION_FILE EXECUTABLE [ARG...].
// Wait for the actual model child, not merely the profiling CLI.
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const auto pid = fork();
  if (pid < 0) return 1;
  if (pid == 0) {
    execv(argv[2], argv + 2);
    _exit(127);
  }
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(pid, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return 1;
  // The frozen model helpers return zero only after proven native teardown
  // or original context/model/backend release. No new process group is made.
  const auto temporary = std::string(argv[1]) + ".partial." + std::to_string(getpid());
  const auto fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) return 1;
  char receipt[256];
  const auto length =
      std::snprintf(receipt, sizeof(receipt),
                    "{\"phase\":\"child_exit0_after_source_verified_release\",\"child_pid\":%ld,"
                    "\"child_status\":0}\n",
                    static_cast<long>(pid));
  bool good = length > 0 && static_cast<unsigned>(length) < sizeof(receipt);
  unsigned done = 0;
  while (good && done < static_cast<unsigned>(length)) {
    const auto count = write(fd, receipt + done, static_cast<unsigned>(length) - done);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      good = false;
      break;
    }
    done += static_cast<unsigned>(count);
  }
  if (fsync(fd) != 0) good = false;
  if (close(fd) != 0) good = false;
  // Publish only fully written, flushed and closed bytes. link is exclusive:
  // it cannot overwrite an earlier completion receipt, even under a race.
  if (good && link(temporary.c_str(), argv[1]) != 0) good = false;
  if (unlink(temporary.c_str()) != 0) good = false;
  return good ? 0 : 1;
}
