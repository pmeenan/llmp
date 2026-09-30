// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The runtime module's startup steps on scratch trees, and the platform
// module's process services it relies on: the crash policy, lock files and
// readiness notification; a prefill's chunks (prefill.h); the chat route's
// watchdog and scaled deadlines on a synthetic clock (watchdog.h); and the
// start's memory guard (memory_guard.h).

#include "runtime/runtime.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "platform/crash_policy.h"
#include "platform/files.h"
#include "platform/lock_file.h"
#include "platform/sd_notify.h"
#include "runtime/memory_guard.h"
#include "runtime/model_limits.h"
#include "runtime/prefill.h"
#include "runtime/watchdog.h"

namespace {

namespace fs = std::filesystem;
using ::testing::HasSubstr;

class Scratch {
 public:
  Scratch() {
    const char* base = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path parent = base != nullptr ? fs::path(base) : fs::path(::testing::TempDir());
    std::error_code error;
    fs::create_directories(parent, error);
    std::string pattern = (parent / "runtime-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
      (void)::chmod(path_.c_str(), 0755);
    } else {
      ADD_FAILURE() << "cannot create a directory from " << pattern;
    }
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  Scratch(Scratch&&) = delete;
  Scratch& operator=(Scratch&&) = delete;
  ~Scratch() {
    std::error_code error;
    fs::remove_all(path_, error);
  }
  const fs::path& path() const { return path_; }

  // A standalone configuration whose roles live here.
  jitllm::runtime::Options Options(std::string_view extra = "") const {
    const fs::path config = path_ / "jitllm.toml";
    std::ofstream(config) << std::format("schema_version = 2\n{}[storage]\ndata_dir = \"{}\"\n",
                                         extra, (path_ / "data").string());
    return {.config = config, .config_given = true, .anchor = path_ / "enrollment", .help = false};
  }

 private:
  fs::path path_;
};

// Start()'s log, and its exit status if it refused.
struct Outcome {
  std::string log;
  int status = -1;  // -1: started
};

Outcome StartWith(const jitllm::runtime::Options& options) {
  char* buffer = nullptr;
  std::size_t size = 0;
  std::FILE* log = ::open_memstream(&buffer, &size);
  Outcome outcome;
  {
    auto started = jitllm::runtime::Start(options, log);
    if (!started) {
      outcome.status = started.error();
    }
  }
  (void)std::fclose(log);
  outcome.log.assign(buffer, size);
  std::free(buffer);  // NOLINT(cppcoreguidelines-no-malloc)
  return outcome;
}

TEST(RuntimeArguments, Parse) {
  const std::vector<std::string_view> args = {"--config", "a.toml", "--anchor", "/x/anchor"};
  auto options = jitllm::runtime::ParseArguments(args);
  ASSERT_TRUE(options.has_value()) << options.error();
  EXPECT_EQ(options->config, "a.toml");
  EXPECT_TRUE(options->config_given);
  EXPECT_EQ(options->anchor, "/x/anchor");
  auto defaults = jitllm::runtime::ParseArguments({});
  ASSERT_TRUE(defaults.has_value());
  EXPECT_EQ(defaults->config, "/etc/jitllm/jitllm.toml");
  EXPECT_FALSE(defaults->config_given);
  EXPECT_EQ(defaults->anchor, "/var/lib/jitllm/enrollment");
  for (const std::vector<std::string_view>& bad :
       {std::vector<std::string_view>{"--config"}, {"--anchor", ""}, {"serve"}}) {
    EXPECT_FALSE(jitllm::runtime::ParseArguments(bad).has_value());
  }
  EXPECT_EQ(defaults->command.command, jitllm::runtime::Command::kService);
}

// The serving commands (D-096): everything after the command's name is its.
TEST(RuntimeArguments, Commands) {
  using jitllm::runtime::Command;
  const std::vector<std::string_view> chat = {
      "--config",      "a.toml", "chat",   "--max-tokens", "32",   "--ignore-stop",
      "--turn",        "ds",     "Hi",     "--turn",       "qwen", "--plain here stays text",
      "--image-noise", "n.bf16", "--plain"};
  auto options = jitllm::runtime::ParseArguments(chat);
  ASSERT_TRUE(options.has_value()) << options.error();
  EXPECT_EQ(options->config, "a.toml");
  EXPECT_EQ(options->command.command, Command::kChat);
  EXPECT_EQ(options->command.chat.max_tokens, 32U);
  EXPECT_TRUE(options->command.chat.ignore_stop);
  EXPECT_FALSE(options->command.chat.fresh);
  ASSERT_EQ(options->command.chat.turns.size(), 2U);
  EXPECT_EQ(options->command.chat.turns[0].model, "ds");
  EXPECT_EQ(options->command.chat.turns[1].text, "--plain here stays text");
  EXPECT_TRUE(options->command.serving.plain);
  EXPECT_EQ(options->command.serving.image_noise, "n.bf16");

  const std::vector<std::string_view> table = {
      "swap-table", "--pairs",        "a:b,b:a", "--context-text", "t.md",  "--cycles",
      "1",          "--zero-context", "off",     "--report",       "r.json"};
  options = jitllm::runtime::ParseArguments(table);
  ASSERT_TRUE(options.has_value()) << options.error();
  EXPECT_EQ(options->command.command, Command::kSwapTable);
  ASSERT_EQ(options->command.table.pairs.size(), 2U);
  EXPECT_EQ(options->command.table.pairs[1].first, "b");
  EXPECT_EQ(options->command.table.cycles, 1U);
  EXPECT_FALSE(options->command.table.zero_context);
  EXPECT_EQ(options->command.table.context_tokens, 8192U);
  EXPECT_EQ(options->command.serving.report, "r.json");

  options = jitllm::runtime::ParseArguments(
      std::array<std::string_view, 3>{"swap-table", "--context-tokens", "1048576"});
  ASSERT_TRUE(options.has_value()) << options.error();
  EXPECT_EQ(options->command.table.context_tokens, 1048576U);

  const std::string long_text(jitllm::runtime::kMaxTurnBytes + 1, 'x');
  const std::string upper_sha(64, 'A');  // hex, but not as the table prints it
  for (const std::vector<std::string_view>& bad : {
           std::vector<std::string_view>{"chat"},  // no turn
           {"chat", "--turn", "ds"},
           {"chat", "--turn", "ds", ""},
           {"chat", "--turn", "ds", long_text},
           {"chat", "--max-tokens", "0", "--turn", "ds", "Hi"},
           {"chat", "--max-tokens", "8193", "--turn", "ds", "Hi"},
           {"chat", "--cycles", "1", "--turn", "ds", "Hi"},  // swap-table's
           {"swap-table", "--turn", "ds", "Hi"},             // chat's
           {"swap-table", "--pairs", "a:a"},
           {"swap-table", "--pairs", "a"},
           {"swap-table", "--pairs", ""},
           {"swap-table", "--cycles", "9"},
           {"swap-table", "--cycles", "0"},
           {"swap-table", "--context-tokens", "1048577"},
           {"swap-table", "--image-expect", upper_sha},
           {"swap-table", "--zero-context", "yes"},
           {"swap-table", "--image-expect", "abc"},
           {"swap-table", "--report"},
       }) {
    EXPECT_FALSE(jitllm::runtime::ParseArguments(bad).has_value()) << bad.front();
  }
  std::vector<std::string_view> many = {"chat"};
  for (std::size_t i = 0; i <= jitllm::runtime::kMaxTurns; ++i) {
    many.insert(many.end(), {"--turn", "ds", "Hi"});
  }
  EXPECT_FALSE(jitllm::runtime::ParseArguments(many).has_value());
}

// A host without a GPU this build targets (or, as in the workstation
// presets' tests, without the driver: they load its stub) refuses at the
// platform step; the CPU-only build, a Spark and a targeted discrete GPU
// (D-082) get to readiness.
TEST(RuntimeStart, CreatesItsRolesAndTakesTheLock) {
  const Scratch scratch;
  const Outcome outcome = StartWith(scratch.Options());
  EXPECT_THAT(outcome.log,
              HasSubstr("storage: installed " + (scratch.path() / "data/models").string()));
  EXPECT_TRUE(fs::is_regular_file(scratch.path() / "data/spill/.jitllm-spill"));
  EXPECT_TRUE(fs::is_regular_file(scratch.path() / "enrollment.lock"));
  if (outcome.status != -1) {
    EXPECT_EQ(outcome.status, jitllm::runtime::kExitHostNotReady);
    EXPECT_THAT(outcome.log, HasSubstr("refusing to start: this host cannot run this build now"));
  }
}

TEST(RuntimeStart, RefusesWhileTheAnchorExists) {
  const Scratch scratch;
  const auto options = scratch.Options();
  std::ofstream(options.anchor) << "anchor";
  const Outcome outcome = StartWith(options);
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log,
              HasSubstr("the enrollment anchor " + options.anchor.string() + " exists"));
  EXPECT_FALSE(fs::exists(scratch.path() / "data"));
}

TEST(RuntimeStart, RefusesAnInvalidOrMemberConfiguration) {
  const Scratch scratch;
  Outcome outcome = StartWith(scratch.Options("bogus = 1\n"));
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log, HasSubstr(":2:9: unknown key bogus"));
  outcome = StartWith(scratch.Options(
      "cluster_file = \"/etc/jitllm/cluster.toml\"\nnode_id = "
      "\"af564a6b-8b4e-4528-8140-e50b92b40002\"\n"
      "credentials.ca_file = \"/c/ca.pem\"\ncredentials.certificate_file = \"/c/n.pem\"\n"
      "credentials.private_key_file = \"/c/k.pem\"\ncontrol.port = 7443\n"));
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log, HasSubstr("a cluster member's, and this build has no cluster support"));
  jitllm::runtime::Options missing = scratch.Options();
  missing.config = scratch.path() / "missing.toml";
  outcome = StartWith(missing);
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log, HasSubstr("missing.toml: does not exist"));
}

TEST(RuntimeStart, OneRuntimePerNode) {
  const Scratch scratch;
  const auto options = scratch.Options();
  fs::path lock_path = options.anchor;
  lock_path += ".lock";
  auto held = jitllm::platform::LockFile::Acquire(lock_path, ::geteuid());
  ASSERT_TRUE(held.has_value()) << held.error();
  const Outcome outcome = StartWith(options);
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log, HasSubstr("enrollment.lock is locked: another process holds it"));
  EXPECT_FALSE(fs::exists(scratch.path() / "data"));
}

TEST(LockFile, RefusesLinksAndOpenModes) {
  const Scratch scratch;
  std::ofstream(scratch.path() / "open.lock") << "";
  ASSERT_EQ(::chmod((scratch.path() / "open.lock").c_str(), 0644), 0);
  auto lock = jitllm::platform::LockFile::Acquire(scratch.path() / "open.lock", ::geteuid());
  ASSERT_FALSE(lock.has_value());
  EXPECT_THAT(lock.error(), HasSubstr("with mode 0600, not uid"));
  fs::create_symlink(scratch.path() / "elsewhere", scratch.path() / "link.lock");
  lock = jitllm::platform::LockFile::Acquire(scratch.path() / "link.lock", ::geteuid());
  ASSERT_FALSE(lock.has_value());
  EXPECT_THAT(lock.error(), HasSubstr("symbolic link"));
  EXPECT_FALSE(fs::exists(scratch.path() / "elsewhere"));
  // Released with its last descriptor.
  {
    auto first = jitllm::platform::LockFile::Acquire(scratch.path() / "a.lock", ::geteuid());
    ASSERT_TRUE(first.has_value()) << first.error();
  }
  EXPECT_TRUE(
      jitllm::platform::LockFile::Acquire(scratch.path() / "a.lock", ::geteuid()).has_value());
}

// Runs body in a child with the crash policy installed; returns its wait
// status.
template <typename Body>
int CrashChild(Body body) {
  (void)std::fflush(nullptr);
  const pid_t child = ::fork();
  if (child == 0) {
    if (!jitllm::platform::InstallCrashPolicy("crash-test")) {
      ::_exit(99);
    }
    body();
    ::_exit(98);
  }
  int status = 0;
  EXPECT_EQ(::waitpid(child, &status, 0), child);
  return status;
}

// D-014: a crash never becomes a core dump. The process ends with an exit
// status, so the kernel never starts one, whatever core_pattern says.
TEST(CrashPolicy, FatalSignalsExitWithoutACoreDump) {
  for (const int signal : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGQUIT, SIGSYS, SIGTRAP}) {
    const int status = CrashChild([signal] { (void)std::raise(signal); });
    EXPECT_TRUE(WIFEXITED(status)) << signal << ": wait status " << status;
    EXPECT_EQ(WEXITSTATUS(status), jitllm::platform::kFatalSignalExitBase + signal) << signal;
  }
  const int aborted = CrashChild([] { std::abort(); });
  EXPECT_TRUE(WIFEXITED(aborted));
  EXPECT_EQ(WEXITSTATUS(aborted), jitllm::platform::kFatalSignalExitBase + SIGABRT);
}

TEST(CrashPolicy, StackOverflowStillExits) {
  const int status = CrashChild([] {
    // Recurses until the stack runs out; the flag keeps the compiler from
    // proving it never ends.
    struct Recurse {
      static int Deeper(int depth, const volatile bool& go_on) {
        std::array<volatile char, 4096> frame{};
        frame[0] = static_cast<char>(depth);
        return go_on ? Deeper(depth + 1, go_on) + frame[0] : 0;
      }
    };
    static volatile bool go_on = true;
    (void)Recurse::Deeper(0, go_on);
  });
  EXPECT_TRUE(WIFEXITED(status)) << "wait status " << status;
  EXPECT_EQ(WEXITSTATUS(status), jitllm::platform::kFatalSignalExitBase + SIGSEGV);
}

// A thread that installs its own signal stack is covered too.
TEST(CrashPolicy, AThreadsStackOverflowStillExits) {
  const int status = CrashChild([] {
    std::thread worker([] {
      if (!jitllm::platform::InstallThreadSignalStack()) {
        ::_exit(97);
      }
      struct Recurse {
        static int Deeper(int depth, const volatile bool& go_on) {
          std::array<volatile char, 4096> frame{};
          frame[0] = static_cast<char>(depth);
          return go_on ? Deeper(depth + 1, go_on) + frame[0] : 0;
        }
      };
      static volatile bool go_on = true;
      (void)Recurse::Deeper(0, go_on);
    });
    worker.join();
  });
  EXPECT_TRUE(WIFEXITED(status)) << "wait status " << status;
  EXPECT_EQ(WEXITSTATUS(status), jitllm::platform::kFatalSignalExitBase + SIGSEGV);
}

TEST(CrashPolicy, MarkingTwiceIsFine) {
  const int status = CrashChild([] {
    ::_exit(jitllm::platform::MarkNonDumpable() && jitllm::platform::MarkNonDumpable() ? 0 : 1);
  });
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(CrashPolicy, MarksTheProcessNonDumpable) {
  const int status = CrashChild([] {
    auto filter = jitllm::platform::ReadFirstLine("/proc/self/coredump_filter");
    const bool filtered = filter && std::strtoul(filter->c_str(), nullptr, 16) == 0;
    ::_exit(::prctl(PR_GET_DUMPABLE) == 0 && filtered ? 0 : 1);
  });
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(NotifyServiceManager, SendsToTheSocket) {
  const Scratch scratch;
  const fs::path socket_path = scratch.path() / "notify";
  const int fd = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  ASSERT_GE(fd, 0);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.string().size() >= sizeof(address.sun_path)) {
    (void)::close(fd);
    GTEST_SKIP() << "the scratch path is too long for a socket";
  }
  (void)socket_path.string().copy(address.sun_path, sizeof(address.sun_path) - 1);
  ASSERT_EQ(::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);  // NOLINT
  ASSERT_EQ(::setenv("NOTIFY_SOCKET", socket_path.c_str(), 1), 0);  // NOLINT(concurrency-mt-unsafe)
  auto sent = jitllm::platform::NotifyServiceManager("READY=1");
  (void)::unsetenv("NOTIFY_SOCKET");  // NOLINT(concurrency-mt-unsafe)
  ASSERT_TRUE(sent.has_value()) << sent.error();
  EXPECT_TRUE(*sent);
  std::array<char, 64> received{};
  const ssize_t got = ::recv(fd, received.data(), received.size(), MSG_DONTWAIT);
  (void)::close(fd);
  EXPECT_EQ(std::string_view(received.data(), got > 0 ? static_cast<std::size_t>(got) : 0),
            "READY=1");
  auto unset = jitllm::platform::NotifyServiceManager("READY=1");
  ASSERT_TRUE(unset.has_value());
  EXPECT_FALSE(*unset);
}

TEST(ModelContext, ChecksTheCheckpointCeilingBeforeSetup) {
  using jitllm::runtime::CheckModelContext;
  for (const std::string_view architecture : {"deepseek4", "qwen4exp"}) {
    EXPECT_TRUE(CheckModelContext(architecture, 512).has_value());
    EXPECT_TRUE(CheckModelContext(architecture, jitllm::config::kDefaultContext).has_value());
    EXPECT_FALSE(CheckModelContext(architecture, 511).has_value());
  }
  EXPECT_TRUE(CheckModelContext("deepseek4", 1048576).has_value());
  auto refused = CheckModelContext("deepseek4", 1048577);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("deepseek4's supported range 512 to 1048576"));
  refused = CheckModelContext("qwen4exp", 262145);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("qwen4exp's supported range 512 to 262144"));
  EXPECT_FALSE(CheckModelContext("qwen4exp", 1048576).has_value());
  EXPECT_FALSE(CheckModelContext("unknown", 262144).has_value());
}

// A prefill's chunk (runtime/prefill.h): the configured or default rows,
// capped by the model and below the context.
TEST(PrefillChunk, TakesTheConfiguredOrDefaultRowsWithinTheModelAndContext) {
  using jitllm::runtime::PrefillChunkRows;
  EXPECT_EQ(PrefillChunkRows(8704, std::nullopt, 2048, 8192), 2048U);
  EXPECT_EQ(PrefillChunkRows(8704, 4096U, 2048, 8192), 4096U);
  EXPECT_EQ(PrefillChunkRows(8704, 65536U, 2048, 8192), 8192U);  // the model's most
  EXPECT_EQ(PrefillChunkRows(1048576, std::nullopt, 2048, 1048448), 2048U);
  EXPECT_EQ(PrefillChunkRows(262144, std::nullopt, 4096, 8192), 4096U);
  // The minimum context: a chunk below it, whatever the default.
  EXPECT_EQ(PrefillChunkRows(512, std::nullopt, 2048, 512), 504U);  // whole tiles
  EXPECT_EQ(PrefillChunkRows(512, std::nullopt, 2048, 384), 384U);
  EXPECT_EQ(PrefillChunkRows(512, 512U, 512, 512), 504U);
  EXPECT_EQ(PrefillChunkRows(8704, 1500U, 512, 8192), 1496U);
  EXPECT_EQ(PrefillChunkRows(8704, 5U, 512, 8192), 5U);
  EXPECT_EQ(PrefillChunkRows(2, std::nullopt, 512, 2), 1U);
  EXPECT_EQ(PrefillChunkRows(1, std::nullopt, 512, 1), 0U);
  EXPECT_EQ(PrefillChunkRows(0, std::nullopt, 512, 0), 0U);
  EXPECT_EQ(PrefillChunkRows(8704, std::nullopt, 512, 0), 0U);  // the model allows none
}

// The chunks run in order, in rows of at most the chunk, and `go_on` asked
// before each (with its rows) stops the loop between chunks, never inside
// one.
TEST(PrefillChunk, RunsChunksUntilToldToStop) {
  using jitllm::runtime::RunPrefillChunks;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ran;
  const auto chunk = [&](std::uint32_t at, std::uint32_t rows) -> std::expected<void, std::string> {
    ran.emplace_back(at, rows);
    return {};
  };
  auto whole = RunPrefillChunks(100, 1300, 512, chunk, {});
  ASSERT_TRUE(whole.has_value()) << whole.error();
  EXPECT_EQ(ran, (std::vector<std::pair<std::uint32_t, std::uint32_t>>{
                     {100, 512}, {612, 512}, {1124, 176}}));
  EXPECT_EQ(whole->end, 1300U);
  EXPECT_EQ(whole->chunks, 3U);
  EXPECT_FALSE(whole->stopped);
  EXPECT_GE(whole->longest, 0.0);

  // Told to stop before the third chunk: the first two ran, and the end is
  // where they reached (what the state then holds).
  ran.clear();
  std::vector<std::uint32_t> asked;
  auto stopped = RunPrefillChunks(0, 1300, 512, chunk, [&](std::uint32_t rows) {
    asked.push_back(rows);
    return asked.size() < 3;
  });
  ASSERT_TRUE(stopped.has_value());
  EXPECT_EQ(asked, (std::vector<std::uint32_t>{512, 512, 276}));
  EXPECT_EQ(ran.size(), 2U);
  EXPECT_EQ(stopped->end, 1024U);
  EXPECT_EQ(stopped->chunks, 2U);
  EXPECT_TRUE(stopped->stopped);

  // Told before the first: nothing runs.
  ran.clear();
  auto none = RunPrefillChunks(40, 1300, 512, chunk, [](std::uint32_t) { return false; });
  ASSERT_TRUE(none.has_value());
  EXPECT_TRUE(ran.empty());
  EXPECT_EQ(none->end, 40U);
  EXPECT_TRUE(none->stopped);

  // Resumed from where it stopped, the chunks fall where an unstopped
  // prefill's would.
  ran.clear();
  auto resumed =
      RunPrefillChunks(stopped->end, 1300, 512, chunk, [](std::uint32_t) { return true; });
  ASSERT_TRUE(resumed.has_value());
  EXPECT_EQ(ran, (std::vector<std::pair<std::uint32_t, std::uint32_t>>{{1024, 276}}));
  EXPECT_FALSE(resumed->stopped);

  // Nothing to run is not a chunk.
  EXPECT_EQ(RunPrefillChunks(7, 7, 512, chunk, {})->chunks, 0U);
}

// A chunk of 1,024 rows or more runs in whole 8-row tiles (the attention's
// mask pre-pass reads them; RE-036), its remainder a chunk of its own.
TEST(PrefillChunk, WideChunksRunInWholeTiles) {
  using jitllm::runtime::RunPrefillChunks;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ran;
  const auto chunk = [&](std::uint32_t at, std::uint32_t rows) -> std::expected<void, std::string> {
    ran.emplace_back(at, rows);
    return {};
  };
  std::vector<std::uint32_t> asked;  // go_on hears the rows each chunk runs
  ASSERT_TRUE(RunPrefillChunks(0, 1500, 2048, chunk, [&](std::uint32_t rows) {
                asked.push_back(rows);
                return true;
              }).has_value());
  EXPECT_EQ(ran, (std::vector<std::pair<std::uint32_t, std::uint32_t>>{{0, 1496}, {1496, 4}}));
  EXPECT_EQ(asked, (std::vector<std::uint32_t>{1496, 4}));
  ran.clear();
  ASSERT_TRUE(RunPrefillChunks(3, 4100, 2048, chunk, {}).has_value());
  EXPECT_EQ(ran, (std::vector<std::pair<std::uint32_t, std::uint32_t>>{
                     {3, 2048}, {2051, 2048}, {4099, 1}}));
  // Under 1,024 rows a chunk is whatever is left.
  ran.clear();
  ASSERT_TRUE(RunPrefillChunks(0, 1023, 2048, chunk, {}).has_value());
  EXPECT_EQ(ran, (std::vector<std::pair<std::uint32_t, std::uint32_t>>{{0, 1023}}));
}

TEST(PrefillChunk, AFailedChunkIsTheError) {
  using jitllm::runtime::RunPrefillChunks;
  int calls = 0;
  auto failed =
      RunPrefillChunks(0, 2048, 512,
                       [&](std::uint32_t at, std::uint32_t) -> std::expected<void, std::string> {
                         ++calls;
                         if (at == 1024) {
                           return std::unexpected("the device faulted");
                         }
                         return {};
                       },
                       {});
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(calls, 3);
  EXPECT_THAT(failed.error(), HasSubstr("the chunk at 1024: the device faulted"));
  EXPECT_FALSE(RunPrefillChunks(
                   0, 10, 0, [](auto, auto) -> std::expected<void, std::string> { return {}; }, {})
                   .has_value());
}

// ---------------------------------------------------------------- the watchdog

using jitllm::runtime::Allowance;
using jitllm::runtime::ExpectedSeconds;
using jitllm::runtime::Floors;
using jitllm::runtime::Phase;
using jitllm::runtime::ScaledDeadline;
using jitllm::runtime::WatchClock;
using jitllm::runtime::Watchdog;
using std::chrono::milliseconds;
using std::chrono::minutes;
using std::chrono::seconds;

// A unit's expected time at the floors, and its allowance: the stall time
// plus three times that.
TEST(Watchdog, AllowsEachUnitItsExpectedTime) {
  const Floors floors{.prefill = 100, .decode = 5};
  EXPECT_DOUBLE_EQ(ExpectedSeconds(Phase::kPrefill, 2048, floors), 20.48);
  EXPECT_DOUBLE_EQ(ExpectedSeconds(Phase::kDecode, 10, floors), 2.0);
  EXPECT_DOUBLE_EQ(ExpectedSeconds(Phase::kSwap, 100'000'000'000ULL, floors), 100.0);
  EXPECT_DOUBLE_EQ(ExpectedSeconds(Phase::kStarting, 5, floors), 0.0);
  EXPECT_EQ(Allowance(seconds(120), 0), seconds(120));
  EXPECT_EQ(Allowance(seconds(120), 20.48), milliseconds(120'000 + 61'440));
  EXPECT_EQ(Allowance(seconds(120), -1), seconds(120));
  EXPECT_LE(Allowance(seconds(120), 1e30), std::chrono::hours(24 * 30));  // saturates
  // A floor of 0 (never configured: the bounds start at 1) counts as 1.
  EXPECT_DOUBLE_EQ(ExpectedSeconds(Phase::kPrefill, 7, Floors{.prefill = 0, .decode = 0}), 7.0);
}

// A non-streaming deadline: min(cap, stall + 3 × (swap + prompt + completion
// at the floors)). DeepSeek's 128K prompt at the default floors fits well
// inside the default cap; a completion as long as the context is capped.
TEST(Watchdog, ScalesANonStreamingDeadlineToItsWork) {
  const Floors floors;  // 100 and 5 tokens a second
  EXPECT_EQ(ScaledDeadline(seconds(120), seconds(14400), floors, 0, 131072, 4096),
            milliseconds(120'000 + (3 * (1'310'720 + 819'200))));
  EXPECT_EQ(ScaledDeadline(seconds(120), seconds(14400), floors, 0, 1000, 262144), seconds(14400));
  EXPECT_EQ(ScaledDeadline(seconds(120), seconds(14400), floors, 50'000'000'000ULL, 100, 10),
            milliseconds(120'000 + (3 * (50'000 + 1'000 + 2'000))));
  EXPECT_EQ(ScaledDeadline(seconds(30), seconds(60), floors, 0, 100'000, 0), seconds(60));
}

// Beats keep a request alive however long it runs: two hours of synthetic
// time (twelve times the old fixed 600 s deadline) with a chunk every
// minute never trips a 120 s stall.
TEST(Watchdog, ProgressKeepsALongRequestAlive) {
  auto now = WatchClock::time_point{} + std::chrono::hours(1);
  Watchdog dog(seconds(120), now);
  EXPECT_FALSE(dog.Check(now + std::chrono::hours(5)));  // idle: nothing watched
  EXPECT_FALSE(dog.due().has_value());
  EXPECT_FALSE(dog.Beat(Phase::kStarting, 0, now));
  for (int i = 0; i < 120; ++i) {
    now += minutes(1);
    EXPECT_FALSE(dog.Check(now)) << i;
    EXPECT_FALSE(dog.Beat(Phase::kPrefill, 0.5, now));
  }
  EXPECT_TRUE(dog.health().healthy);
  EXPECT_EQ(dog.health().stalls, 0U);
  EXPECT_EQ(dog.health().phase, Phase::kPrefill);
  EXPECT_EQ(dog.health().last_progress, now);
  EXPECT_FALSE(dog.Idle(now));
  EXPECT_FALSE(dog.due().has_value());
}

// No beat within the allowance is a stall, noticed once: the backend is
// unhealthy until its next beat, which recovers it.
TEST(Watchdog, AStallMarksTheBackendUnhealthyUntilItsNextBeat) {
  const auto start = WatchClock::time_point{} + std::chrono::hours(1);
  Watchdog dog(seconds(120), start);
  (void)dog.Beat(Phase::kDecode, 0, start);
  EXPECT_EQ(dog.due(), std::optional(start + seconds(120)));
  EXPECT_FALSE(dog.Check(start + seconds(119)));
  EXPECT_TRUE(dog.Check(start + seconds(120)));
  EXPECT_FALSE(dog.health().healthy);
  EXPECT_EQ(dog.health().stalls, 1U);
  EXPECT_EQ(dog.health().stalled_at, start + seconds(120));
  EXPECT_EQ(dog.health().phase, Phase::kDecode);  // where it hung
  EXPECT_EQ(dog.health().last_progress, start);
  EXPECT_FALSE(dog.Check(start + seconds(500)));  // once a stall
  EXPECT_FALSE(dog.due().has_value());
  EXPECT_TRUE(dog.Beat(Phase::kDecode, 0, start + seconds(600)));  // recovered
  EXPECT_TRUE(dog.health().healthy);
  EXPECT_FALSE(dog.Beat(Phase::kDecode, 0, start + seconds(601)));
  // A unit declared long is allowed its expected time: a 2,048-row chunk
  // at 100 tokens a second, 120 s + 3 × 20.48 s.
  const auto at = start + seconds(700);
  (void)dog.Beat(Phase::kPrefill, ExpectedSeconds(Phase::kPrefill, 2048, Floors{}), at);
  EXPECT_FALSE(dog.Check(at + seconds(181)));
  EXPECT_TRUE(dog.Check(at + seconds(182)));
  EXPECT_EQ(dog.health().stalls, 2U);
  EXPECT_TRUE(dog.Idle(at + seconds(300)));  // the request ended: recovered, idle
  EXPECT_FALSE(dog.Check(at + seconds(9000)));
}

// The start's memory guard (memory_guard.h): the largest model's weights,
// the host-built chunk inputs and a 6 GiB margin against what is available.
TEST(MemoryGuard, CountsTheWeightsTheHostInputsAndTheMargin) {
  using jitllm::runtime::CheckMemoryGuard;
  using jitllm::runtime::kUncountedMargin;
  constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
  EXPECT_EQ(kUncountedMargin, 6 * kGiB);
  // DeepSeek's 90.3 GiB of weights, 1 GiB of host inputs: 97.3 GiB needed.
  const std::uint64_t weights = (903 * kGiB) / 10;
  EXPECT_TRUE(CheckMemoryGuard({.largest = weights, .host_inputs = kGiB, .available = 100 * kGiB})
                  .has_value());
  // Exactly enough fits; a byte less does not.
  const std::uint64_t exact = weights + kGiB + kUncountedMargin;
  EXPECT_TRUE(
      CheckMemoryGuard({.largest = weights, .host_inputs = kGiB, .available = exact}).has_value());
  const auto short_by_one = CheckMemoryGuard(
      {.largest = weights, .host_inputs = kGiB, .available = exact - 1, .fixed = 12 * kGiB});
  ASSERT_FALSE(short_by_one.has_value());
  EXPECT_THAT(short_by_one.error(), HasSubstr("host-built chunk inputs (1.0 GiB)"));
  EXPECT_THAT(short_by_one.error(), HasSubstr("6 GiB margin"));
  EXPECT_THAT(short_by_one.error(), HasSubstr("fixed memory (12.0 GiB)"));
  // The host inputs count: what fits without them does not with them.
  const std::uint64_t without = weights + kUncountedMargin;
  EXPECT_TRUE(
      CheckMemoryGuard({.largest = weights, .host_inputs = 0, .available = without}).has_value());
  EXPECT_FALSE(CheckMemoryGuard({.largest = weights, .host_inputs = 3 * kGiB, .available = without})
                   .has_value());
  // The old 4 GiB margin's fit is refused now.
  EXPECT_FALSE(
      CheckMemoryGuard({.largest = weights, .available = weights + (4 * kGiB)}).has_value());
  // Terms past what is available are refused, never wrapped; an unknown
  // availability (0) passes.
  EXPECT_FALSE(
      CheckMemoryGuard(
          {.largest = ~std::uint64_t{0}, .host_inputs = ~std::uint64_t{0}, .available = 100 * kGiB})
          .has_value());
  EXPECT_TRUE(
      CheckMemoryGuard({.largest = weights, .host_inputs = kGiB, .available = 0}).has_value());
}

}  // namespace
