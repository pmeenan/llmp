// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The confined-job proof (plan.md M1, D-074): job processes in delegated
// cgroups, holding their record's lock, under what llmp.service gives
// the runtime. It needs a delegated cgroup, so tools/job-proof runs it in a
// transient systemd unit with the service's delegation (and, on a Spark,
// its user and sandbox); anywhere else it skips, unless LLMP_JOB_PROOF=1
// says it must not.

#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "platform/confine.h"
#include "platform/job.h"

namespace {

namespace fs = std::filesystem;
using ::testing::HasSubstr;
namespace job = ::llmp::platform;

const char* Env(const char* name) {
  return std::getenv(name);  // NOLINT(concurrency-mt-unsafe): read before any thread starts
}

class JobProof : public ::testing::Test {
 protected:
  void SetUp() override {
    auto root = job::JobsCgroupRoot();
    if (!root) {
      if (Env("LLMP_JOB_PROOF") != nullptr) {
        FAIL() << root.error();
      }
      GTEST_SKIP() << "not in a delegated cgroup (" << root.error() << "); run tools/job-proof";
    }
    root_ = *root;
    ASSERT_TRUE(job::BecomeSubreaper().has_value());
    std::string pattern =
        (fs::path(Env("LLMP_JOB_PROOF_STATE") != nullptr ? Env("LLMP_JOB_PROOF_STATE")
                                                         : ::testing::TempDir()) /
         "jobs-XXXXXX")
            .string();
    ASSERT_NE(::mkdtemp(pattern.data()), nullptr);
    state_ = pattern;
  }
  void TearDown() override {
    std::error_code error;
    if (!state_.empty()) {
      fs::remove_all(state_, error);
    }
  }

  // Starts a shell command as job `id`.
  job::StartedJob Start(std::string_view id, std::string_view command) {
    const std::vector<std::string> argv = {"/bin/sh", "-c", std::string(command)};
    auto started = job::StartJob(root_, id, Lock(id), argv);
    EXPECT_TRUE(started.has_value()) << (started ? "" : started.error());
    return started.value_or(job::StartedJob{});
  }
  fs::path Lock(std::string_view id) const { return state_ / std::format("{}.lock", id); }

  // Kills a job's cgroup and waits until it has ended, reaping what
  // re-parented here.
  void KillAndSettle(const job::StartedJob& started, std::string_view id) {
    ASSERT_TRUE(job::KillCgroup(started.cgroup).has_value());
    for (int i = 0; i < 1000; ++i) {
      (void)job::ReapExited();
      auto ended = job::JobEnded(started.cgroup, Lock(id));
      ASSERT_TRUE(ended.has_value()) << ended.error();
      if (*ended) {
        EXPECT_EQ(::rmdir(started.cgroup.c_str()), 0);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    FAIL() << "job " << id << " did not end within 10 s of being killed";
  }

  fs::path root_;
  fs::path state_;
};

// A child that outlives its job (it daemonizes: a new session, re-parented)
// stays in the job's cgroup and keeps the job's lock: the job has not ended
// until it too has exited and been reaped.
TEST_F(JobProof, AChildThatOutlivesItsJobIsContained) {
  const auto started =
      Start("outlives", "setsid /bin/sleep 600 </dev/null >/dev/null 2>&1 & exit 0");
  int status = 0;
  ASSERT_EQ(::waitpid(started.pid, &status, 0), started.pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  // The job's own process is gone; its daemon is not.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_EQ(job::CgroupPopulated(started.cgroup), true);
  EXPECT_EQ(job::LockIsFree(Lock("outlives")), false);
  EXPECT_EQ(job::JobEnded(started.cgroup, Lock("outlives")), false);
  KillAndSettle(started, "outlives");
  EXPECT_EQ(job::LockIsFree(Lock("outlives")), true);
}

// The lock goes with the job's processes across exec, on the descriptor
// the environment names, and the runtime keeps none.
TEST_F(JobProof, TheLockIsInheritedAcrossExec) {
  const auto started =
      Start("exec", R"(test -e "/proc/self/fd/$LLMP_JOB_LOCK_FD" && exec /bin/sleep 600)");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  std::error_code error;
  EXPECT_EQ(fs::read_symlink(std::format("/proc/{}/exe", started.pid), error).filename(), "sleep");
  EXPECT_EQ(job::LockIsFree(Lock("exec")), false);
  // A second job with the same record cannot start while it runs.
  const std::vector<std::string> argv = {"/bin/true"};
  auto second = job::StartJob(root_, "exec-again", Lock("exec"), argv);
  ASSERT_FALSE(second.has_value());
  EXPECT_THAT(second.error(), HasSubstr("that job has not ended"));
  KillAndSettle(started, "exec");
}

// A job starts with the default signal mask and dispositions, whatever the
// runtime blocks or ignores (it blocks its stop signals and ignores
// SIGPIPE), so a stop reaches it and a closed pipe ends it.
TEST_F(JobProof, AJobStartsWithDefaultSignals) {
  sigset_t stop;
  sigset_t before;
  (void)::sigemptyset(&stop);
  (void)::sigaddset(&stop, SIGTERM);
  ASSERT_EQ(::pthread_sigmask(SIG_BLOCK, &stop, &before), 0);
  auto* const previous = ::signal(SIGPIPE, SIG_IGN);
  const fs::path out = state_ / "signals";
  const auto started = Start(
      "signals",
      std::format(R"(exec /bin/grep -E "^Sig(Blk|Ign):" /proc/self/status > '{}')", out.string()));
  (void)::signal(SIGPIPE, previous);
  ASSERT_EQ(::pthread_sigmask(SIG_SETMASK, &before, nullptr), 0);
  int status = 0;
  ASSERT_EQ(::waitpid(started.pid, &status, 0), started.pid);
  std::stringstream text;
  text << std::ifstream(out).rdbuf();
  EXPECT_EQ(text.str(), "SigBlk:\t0000000000000000\nSigIgn:\t0000000000000000\n");
  KillAndSettle(started, "signals");
}

// No job inherits another's lock: killing one frees its lock while the
// other, started after it, still runs.
TEST_F(JobProof, NoJobInheritsAnothersLock) {
  const auto first = Start("first", "exec /bin/sleep 600");
  const auto second = Start("second", "exec /bin/sleep 600");
  KillAndSettle(first, "first");
  EXPECT_EQ(job::LockIsFree(Lock("first")), true);
  EXPECT_EQ(job::LockIsFree(Lock("second")), false);
  KillAndSettle(second, "second");
}

// A runtime that dies without stopping its job: a restarted one (here, the
// test process, the subreaper its orphans re-parent to) finds the job's
// lock held and its cgroup populated, so the job has not ended; it kills
// the cgroup, reaps, and then the record settles and can be reused.
TEST_F(JobProof, ARestartedRuntimeSettlesItsPredecessorsJob) {
  const fs::path cgroup = root_ / "orphaned";
  (void)std::fflush(nullptr);
  const pid_t first_runtime = ::fork();
  if (first_runtime == 0) {
    const std::vector<std::string> argv = {"/bin/sh", "-c", "exec /bin/sleep 600"};
    ::_exit(job::StartJob(root_, "orphaned", Lock("orphaned"), argv).has_value() ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(first_runtime, &status, 0), first_runtime);
  ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  const auto ended = job::JobEnded(cgroup, Lock("orphaned"));
  EXPECT_EQ(ended, false) << (ended ? "" : ended.error());
  const auto populated = job::CgroupPopulated(cgroup);
  EXPECT_EQ(populated, true) << (populated ? "" : populated.error());
  KillAndSettle({.pid = -1, .cgroup = cgroup}, "orphaned");
  const auto again = Start("orphaned", "exit 0");
  KillAndSettle(again, "orphaned");
}

// The same across a restart of the unit itself: the harness runs phase
// "crash", which starts a job and leaves it; the unit's main process exits
// and systemd stops the unit, killing its whole cgroup. Phase "recover", a
// new unit, finds the job ended: its cgroup gone and its lock free.
TEST(UnitRestart, Phase) {
  const char* phase = Env("LLMP_JOB_PROOF_PHASE");
  const char* state = Env("LLMP_JOB_PROOF_STATE");
  if (phase == nullptr || state == nullptr) {
    GTEST_SKIP() << "tools/job-proof runs this in two units";
  }
  const fs::path record = fs::path(state) / "unit-restart";
  const fs::path lock = fs::path(state) / "unit-restart.lock";
  if (std::string_view(phase) == "crash") {
    auto root = job::JobsCgroupRoot();
    ASSERT_TRUE(root.has_value()) << root.error();
    const std::vector<std::string> argv = {
        "/bin/sh", "-c", "setsid /bin/sleep 600 </dev/null >/dev/null 2>&1 & exec /bin/sleep 600"};
    auto started = job::StartJob(*root, "unit-restart", lock, argv);
    ASSERT_TRUE(started.has_value()) << started.error();
    std::ofstream(record) << started->cgroup.string() << "\n";
    EXPECT_EQ(job::LockIsFree(lock), false);
    return;  // the unit ends with the job still running
  }
  ASSERT_EQ(std::string_view(phase), "recover");
  std::string cgroup;
  std::ifstream(record) >> cgroup;
  ASSERT_FALSE(cgroup.empty());
  EXPECT_FALSE(fs::exists(cgroup)) << "systemd removes a stopped unit's cgroup";
  EXPECT_EQ(job::JobEnded(cgroup, lock), true);
}

// A stage that parses untrusted input reads only its inputs, writes only
// its staging directory, and has no network (Landlock and seccomp; RE-013
// rules out namespaces). Needs no delegation.
TEST(Confinement, AStageReachesOnlyItsInputsAndStaging) {
  std::string pattern = (fs::path(::testing::TempDir()) / "confine-XXXXXX").string();
  ASSERT_NE(::mkdtemp(pattern.data()), nullptr);
  const fs::path root = pattern;
  fs::create_directories(root / "input");
  fs::create_directories(root / "staging");
  fs::create_directories(root / "state");
  std::ofstream(root / "input/checkpoint") << "weights";
  std::ofstream(root / "state/record") << "secret";
  (void)std::fflush(nullptr);
  const pid_t child = ::fork();
  if (child == 0) {
    const std::array read = {root / "input"};
    const std::array write = {root / "staging"};
    if (!llmp::platform::ConfineSelf({.read = read, .write = write})) {
      ::_exit(100);
    }
    const auto opens = [](const fs::path& path, int flags) {
      const int fd = ::open(path.c_str(), flags | O_CLOEXEC, 0600);
      if (fd >= 0) {
        (void)::close(fd);
      }
      return fd >= 0;
    };
    // Each check that goes the wrong way is one letter of the exit report.
    std::string wrong;
    const auto expect = [&wrong](bool ok, char code) {
      if (!ok) {
        wrong += code;
      }
    };
    expect(opens(root / "input/checkpoint", O_RDONLY), 'a');         // may read its input
    expect(opens(root / "staging/shard", O_WRONLY | O_CREAT), 'b');  // may write staging
    expect(!opens(root / "state/record", O_RDONLY), 'c');            // may not read state
    expect(!opens(root / "input/checkpoint", O_WRONLY), 'd');        // may not change input
    expect(!opens(root / "planted", O_WRONLY | O_CREAT), 'e');       // may not create elsewhere
    expect(!opens("/etc/hostname", O_RDONLY), 'f');                  // may not read the system
    const int inet = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(inet < 0 && errno == EACCES, 'g');                           // no network
    expect(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0) < 0, 'h');  // no local sockets
    std::array<int, 2> pair{};
    expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair.data()) != 0, 'i');  // nor pairs
    std::array<char, 120> params{};  // struct io_uring_params, zeroed
    expect(::syscall(SYS_io_uring_setup, 1, params.data()) < 0, 'j');  // nor io_uring's sockets
    expect(::kill(::getppid(), 0) != 0, 'k');                          // nor signals outside
    (void)::write(STDERR_FILENO, wrong.data(), wrong.size());
    ::_exit(wrong.empty() ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "wait status " << status;
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "the letters on standard error name what went wrong; 100: could not confine";
  std::error_code error;
  fs::remove_all(root, error);
}

}  // namespace
