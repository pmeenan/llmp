// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/job.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "platform/files.h"

namespace llmp::platform {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kCgroupMount = "/sys/fs/cgroup";

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

std::expected<void, std::string> WriteFile(const fs::path& path, std::string_view text) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) {
    return std::unexpected(std::format("cannot open {}: {}", path.string(), Errno(errno)));
  }
  const ssize_t wrote = ::write(fd, text.data(), text.size());
  const int error = errno;
  if (::close(fd) != 0 || std::cmp_not_equal(wrote, text.size())) {
    return std::unexpected(std::format("cannot write {}: {}", path.string(), Errno(error)));
  }
  return {};
}

}  // namespace

std::expected<fs::path, std::string> OwnCgroup() {
  auto line = ReadFirstLine("/proc/self/cgroup");
  if (!line) {
    return std::unexpected("cannot read /proc/self/cgroup: " + line.error().message());
  }
  // cgroup v2 only: one line, "0::<path>".
  if (!line->starts_with("0::/")) {
    return std::unexpected(
        std::format("not in a cgroup v2 hierarchy (/proc/self/cgroup: {})", *line));
  }
  return fs::path(kCgroupMount) / fs::path(line->substr(4)).relative_path();
}

std::expected<fs::path, std::string> JobsCgroupRoot() {
  auto own = OwnCgroup();
  if (!own) {
    return own;
  }
  // Only where llmp.service put the runtime (DelegateSubgroup=runtime):
  // anywhere else, the parent is a cgroup systemd manages, not ours.
  if (own->filename() != "runtime") {
    return std::unexpected(
        std::format("{} is not a delegated service's `runtime` subgroup "
                    "(Delegate=, DelegateSubgroup=runtime)",
                    own->string()));
  }
  fs::path root = own->parent_path() / "jobs";
  if (::mkdir(root.c_str(), 0755) != 0 && errno != EEXIST) {
    return std::unexpected(std::format("cannot create {}: {}", root.string(), Errno(errno)));
  }
  return root;
}

std::expected<void, std::string> BecomeSubreaper() {
  if (::prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0) {
    return std::unexpected("cannot become a child subreaper: " + Errno(errno));
  }
  return {};
}

std::expected<StartedJob, std::string> StartJob(const fs::path& jobs_root, std::string_view id,
                                                const fs::path& lock_file,
                                                std::span<const std::string> argv) {
  if (argv.empty() || !fs::path(argv.front()).is_absolute()) {
    return std::unexpected("a job needs an absolute program path");
  }
  if (id.empty() || id.contains('/') || id.starts_with('.')) {
    return std::unexpected(std::format("'{}' is not a job id", id));
  }
  const fs::path cgroup = jobs_root / id;
  if (::mkdir(cgroup.c_str(), 0755) != 0) {
    return std::unexpected(
        std::format("cannot create the job's cgroup {}: {}", cgroup.string(), Errno(errno)));
  }
  // Close-on-exec until the job's own child clears it, so no other job
  // started meanwhile, by any thread, inherits this lock.
  const int lock = ::open(lock_file.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (lock < 0) {
    (void)::rmdir(cgroup.c_str());
    return std::unexpected(
        std::format("cannot open the job lock {}: {}", lock_file.string(), Errno(errno)));
  }
  if (::flock(lock, LOCK_EX | LOCK_NB) != 0) {
    const int error = errno;
    (void)::close(lock);
    (void)::rmdir(cgroup.c_str());
    return std::unexpected(
        error == EWOULDBLOCK
            ? std::format("the job lock {} is held: that job has not ended", lock_file.string())
            : std::format("cannot lock {}: {}", lock_file.string(), Errno(error)));
  }
  const int procs = ::open((cgroup / "cgroup.procs").c_str(), O_WRONLY | O_CLOEXEC);
  if (procs < 0) {
    const int error = errno;
    (void)::close(lock);
    (void)::rmdir(cgroup.c_str());
    return std::unexpected(
        std::format("cannot open {}/cgroup.procs: {}", cgroup.string(), Errno(error)));
  }
  // Everything the child needs, prepared before fork().
  std::vector<std::string> args(argv.begin(), argv.end());
  std::vector<char*> args_c;
  args_c.reserve(args.size() + 1);
  for (std::string& arg : args) {
    args_c.push_back(arg.data());
  }
  args_c.push_back(nullptr);
  std::vector<std::string> env = {std::format("{}={}", kJobLockFdVariable, lock)};
  for (char** entry = environ; *entry != nullptr;
       ++entry) {  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (!std::string_view(*entry).starts_with(std::format("{}=", kJobLockFdVariable))) {
      env.emplace_back(*entry);
    }
  }
  std::vector<char*> env_c;
  env_c.reserve(env.size() + 1);
  for (std::string& entry : env) {
    env_c.push_back(entry.data());
  }
  env_c.push_back(nullptr);

  // The child reports a failure on this pipe; its end closes on exec, so
  // end of file means the job runs its program inside its cgroup, and
  // StartJob returns only then: a kill right after cannot miss it.
  std::array<int, 2> report{-1, -1};
  if (::pipe2(report.data(), O_CLOEXEC) != 0) {
    const int error = errno;
    (void)::close(procs);
    (void)::close(lock);
    (void)::rmdir(cgroup.c_str());
    return std::unexpected("cannot start the job: " + Errno(error));
  }
  const pid_t pid = ::fork();
  if (pid == 0) {
    // In the child, async-signal-safe calls only: the default signal
    // mask and SIGPIPE (the runtime blocks its stop signals and ignores
    // SIGPIPE, which exec would pass on), into the job's cgroup, keep the
    // lock across exec, exec.
    sigset_t none;
    (void)::sigemptyset(&none);
    // The forked child has one thread.
    (void)::sigprocmask(SIG_SETMASK, &none, nullptr);  // NOLINT(concurrency-mt-unsafe)
    (void)::signal(SIGPIPE, SIG_DFL);
    char stage = 'e';
    if (::write(procs, "0\n", 2) != 2) {
      stage = 'c';
    } else if (::fcntl(lock, F_SETFD, 0) != 0) {
      stage = 'l';
    }
    if (stage == 'e') {
      ::execve(args_c.front(), args_c.data(), env_c.data());
    }
    const int child_error = errno;
    const std::array<char, 1 + sizeof child_error> message = [&] {
      std::array<char, 1 + sizeof child_error> m{stage};
      std::memcpy(m.data() + 1, &child_error, sizeof child_error);
      return m;
    }();
    (void)::write(report[1], message.data(), message.size());
    ::_exit(127);
  }
  const int fork_error = errno;
  (void)::close(report[1]);
  (void)::close(procs);
  (void)::close(lock);  // the job holds it now
  if (pid < 0) {
    (void)::close(report[0]);
    (void)::rmdir(cgroup.c_str());
    return std::unexpected("cannot start the job: " + Errno(fork_error));
  }
  std::array<char, 1 + sizeof(int)> message{};
  ssize_t got = 0;
  do {
    got = ::read(report[0], message.data(), message.size());
  } while (got < 0 && errno == EINTR);
  const int read_error = errno;
  (void)::close(report[0]);
  if (got < 0) {
    // Whether the job runs is unknown: end it rather than hand back a job
    // that may not be in its cgroup.
    (void)::kill(pid, SIGKILL);
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    (void)::rmdir(cgroup.c_str());
    return std::unexpected("cannot learn whether the job started: " + Errno(read_error));
  }
  if (got > 0) {
    int child_error = 0;
    std::memcpy(&child_error, message.data() + 1, sizeof child_error);
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    (void)::rmdir(cgroup.c_str());
    const char* what = "run";
    if (message[0] == 'c') {
      what = "join its cgroup";
    } else if (message[0] == 'l') {
      what = "keep its lock";
    }
    return std::unexpected(
        std::format("the job could not {} {}: {}", what, args.front(), Errno(child_error)));
  }
  return StartedJob{.pid = pid, .cgroup = cgroup};
}

std::expected<bool, std::string> CgroupPopulated(const fs::path& cgroup) {
  auto events = ReadSmallFile(cgroup / "cgroup.events");
  if (!events) {
    if (events.error() == std::errc::no_such_file_or_directory) {
      return false;
    }
    return std::unexpected(
        std::format("cannot read {}/cgroup.events: {}", cgroup.string(), events.error().message()));
  }
  if (events->contains("populated 1")) {
    return true;
  }
  if (events->contains("populated 0")) {
    return false;
  }
  return std::unexpected(std::format("{}/cgroup.events has no populated line", cgroup.string()));
}

std::expected<void, std::string> KillCgroup(const fs::path& cgroup) {
  return WriteFile(cgroup / "cgroup.kill", "1\n");
}

std::expected<bool, std::string> LockIsFree(const fs::path& lock_file) {
  const int fd = ::open(lock_file.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ENOENT) {
      return true;
    }
    return std::unexpected(std::format("cannot open {}: {}", lock_file.string(), Errno(errno)));
  }
  const bool taken = ::flock(fd, LOCK_EX | LOCK_NB) == 0;
  const int error = errno;
  (void)::close(fd);  // releases it if taken
  if (!taken && error != EWOULDBLOCK) {
    return std::unexpected(std::format("cannot test {}: {}", lock_file.string(), Errno(error)));
  }
  return taken;
}

std::expected<bool, std::string> JobEnded(const fs::path& cgroup, const fs::path& lock_file) {
  auto populated = CgroupPopulated(cgroup);
  if (!populated) {
    return std::unexpected(populated.error());
  }
  if (*populated) {
    return false;
  }
  return LockIsFree(lock_file);
}

int ReapExited() {
  int reaped = 0;
  int status = 0;
  while (::waitpid(-1, &status, WNOHANG) > 0) {
    ++reaped;
  }
  return reaped;
}

}  // namespace llmp::platform
