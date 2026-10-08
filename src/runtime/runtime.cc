// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/runtime.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/build_info.h"
#include "base/report.h"
#include "config/node_config.h"
#include "config/storage_roles.h"
#include "platform/files.h"
#include "platform/host_probe.h"
#include "platform/job.h"
#include "platform/lock_file.h"
#include "platform/path_trust.h"
#include "platform/sd_notify.h"
#include "providers/device_probe.h"
#include "runtime/model_settings.h"

namespace llmp::runtime {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kUsage =
    "Usage: llmp-runtime [--config FILE] [--anchor PATH] [COMMAND [ARGS...]]\n"
    "\n"
    "The llmpalooza node runtime, which llmp.service starts without a command.\n"
    "  --config FILE  the node's configuration (default /etc/llmp/llmp.toml\n"
    "                 and /etc/llmp/llmp.d/)\n"
    "  --anchor PATH  the enrollment anchor (default /var/lib/llmp/enrollment);\n"
    "                 the process lock is PATH.lock\n"
    "\n";

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// One line to the journal; nothing in text can start another (D-014).
void Log(std::FILE* log, std::string_view text) {
  const std::string line = std::format("llmp-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log);
  (void)std::fflush(log);
}

// Opens each RDMA device node the runtime will use, as the unit's device
// policy decides; returns what could not be opened.
std::vector<std::string> OpenRdmaNodes(std::FILE* log) {
  std::vector<std::string> failed;
  auto names = platform::ListDirectory("/dev/infiniband");
  if (!names) {
    Log(log, "no RDMA device nodes (/dev/infiniband)");
    return failed;
  }
  std::vector<std::string> opened;
  for (const std::string& name : *names) {
    if (!name.starts_with("uverbs") && name != "rdma_cm") {
      continue;
    }
    const fs::path node = fs::path("/dev/infiniband") / name;
    const int fd = ::open(node.c_str(), O_RDWR | O_CLOEXEC | O_NOCTTY);
    if (fd < 0) {
      failed.push_back(std::format("{} ({})", node.string(), Errno(errno)));
      continue;
    }
    (void)::close(fd);  // opened only to prove access
    opened.push_back(name);
  }
  std::string list;
  for (const std::string& name : opened) {
    list += (list.empty() ? "" : ", ") + name;
  }
  Log(log, std::format("RDMA device nodes opened read-write: {}", list.empty() ? "none" : list));
  return failed;
}

}  // namespace

std::expected<Options, std::string> ParseArguments(std::span<const std::string_view> args) {
  Options options;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (!arg.starts_with("-")) {
      // A serving command, and every argument after it its own.
      auto command = ParseCommand(arg, args.subspan(i + 1));
      if (!command) {
        return std::unexpected(command.error());
      }
      options.command = std::move(*command);
      break;
    }
    if (arg == "--help" || arg == "-h") {
      options.help = true;
    } else if (arg == "--config" || arg == "--anchor") {
      if (i + 1 >= args.size() || args[i + 1].empty()) {
        return std::unexpected(std::format("{} needs a path", arg));
      }
      const fs::path value(args[++i]);
      if (arg == "--config") {
        options.config = value;
        options.config_given = true;
      } else {
        options.anchor = fs::absolute(value).lexically_normal();
      }
    } else {
      return std::unexpected(std::format("unexpected argument '{}'", arg));
    }
  }
  return options;
}

std::expected<Started, int> Start(const Options& options, std::FILE* log) {
  const base::BuildInfo& build = base::GetBuildInfo();
  Log(log, std::format("llmpalooza {} ({}, license profile {}), uid {}", build.version,
                       build.target, build.license_profile, ::geteuid()));
  const uid_t self = ::geteuid();

  // 1. The enrollment anchor, before anything else (D-063).
  struct stat anchor{};
  if (::lstat(options.anchor.c_str(), &anchor) == 0) {
    Log(log,
        std::format("refusing to start: the enrollment anchor {} exists, and this build has no "
                    "cluster support (D-063)",
                    options.anchor.string()));
    return std::unexpected(kExitRefused);
  }
  if (errno != ENOENT) {
    Log(log, std::format("refusing to start: cannot examine the enrollment anchor {}: {}",
                         options.anchor.string(), Errno(errno)));
    return std::unexpected(kExitRefused);
  }

  // 2. The configuration.
  config::LoadOptions load{.main_file = options.config,
                           .main_file_optional = !options.config_given,
                           .anchor = options.anchor,
                           .trusted_uid = self};
  auto loaded = config::LoadNodeConfig(load);
  if (!loaded) {
    for (const config::Diagnostic& diagnostic : loaded.error()) {
      Log(log, "configuration: " + config::FormatDiagnostic(diagnostic));
    }
    Log(log, "refusing to start: the configuration is not valid");
    return std::unexpected(kExitRefused);
  }
  if (loaded->files.empty()) {
    Log(log, "configuration: none found; the built-in defaults, a standalone node");
  }
  for (const fs::path& file : loaded->files) {
    Log(log, "configuration: read " + file.string());
  }
  if (loaded->membership) {
    Log(log,
        "refusing to start: the configuration is a cluster member's, and this build has no cluster "
        "support");
    return std::unexpected(kExitRefused);
  }

  // 3. The per-node process lock, on a path nobody else could re-point.
  fs::path lock_path = options.anchor;
  lock_path += ".lock";
  if (auto walked = platform::WalkTrusted(lock_path, self, false); !walked) {
    Log(log, std::format("refusing to start: the process lock {}: {}", lock_path.string(),
                         walked.error()));
    return std::unexpected(kExitRefused);
  }
  auto lock = platform::LockFile::Acquire(lock_path, self);
  if (!lock) {
    Log(log, "refusing to start: " + lock.error());
    return std::unexpected(kExitRefused);
  }

  // 4-5. The storage roles.
  auto roles = config::PrepareRuntimeRoles(loaded->storage, self, options.anchor);
  if (!roles) {
    for (const std::string& problem : roles.error()) {
      Log(log, problem);
    }
    Log(log, "refusing to start: the storage roles are not usable");
    return std::unexpected(kExitRefused);
  }
  Log(log, std::format("storage: installed {}, spill {}, state {}", roles->installed.string(),
                       roles->spill.string(), roles->state.string()));

  // Jobs (none run yet in M1): orphans of theirs re-parent here, and they
  // live in cgroups beside the runtime's own, which llmp.service
  // delegates (D-074).
  // (qemu-user, where the tests run, has no subreapers.)
  if (auto subreaper = platform::BecomeSubreaper(); !subreaper) {
    Log(log, "warning: jobs cannot run here: " + subreaper.error());
  } else if (auto jobs = platform::JobsCgroupRoot(); jobs) {
    Log(log, "jobs: cgroups under " + jobs->string());
  } else {
    Log(log, "warning: jobs cannot run here: " + jobs.error());
  }

  // 7. The platform.
  base::Report report;
  platform::DescribeHost("/", report);
  providers::DescribeDevices("/", report);
  for (const std::string& warning : report.warnings) {
    Log(log, "warning: " + warning);
  }
  for (const std::string& failed : OpenRdmaNodes(log)) {
    Log(log, "warning: cannot open " + failed);
  }
  if (!report.problems.empty()) {
    for (const std::string& problem : report.problems) {
      Log(log, "problem: " + problem);
    }
    // Restartable: at boot the driver or its devices may not be ready yet.
    Log(log, "refusing to start: this host cannot run this build now (see `llmp doctor`)");
    return std::unexpected(kExitHostNotReady);
  }
  for (const config::ModelEntry& model : loaded->models) {
    Log(log,
        std::format("model {}: {} {}{}", model.name, model.composition ? "composition" : "artifact",
                    model.composition ? *model.composition : model.artifact.value_or(""),
                    model.drafter ? " with drafter " + *model.drafter : std::string()));
  }
  return Started{
      .lock = std::move(*lock), .roles = std::move(*roles), .config = std::move(*loaded)};
}

int Run(std::span<const std::string_view> args, std::FILE* log, ServeFunction serve) {
  auto options = ParseArguments(args);
  if (!options) {
    Log(log, options.error());
    (void)std::fwrite(kUsage.data(), 1, kUsage.size(), log);
    return kExitUsage;
  }
  if (options->help) {
    (void)std::fwrite(kUsage.data(), 1, kUsage.size(), stdout);
    (void)std::fwrite(kCommandUsage.data(), 1, kCommandUsage.size(), stdout);
    return std::fflush(stdout) == 0 ? kExitOk : kExitFailure;
  }
  if (options->command.command == Command::kSettings) {
    // Reads only: the configuration as the service reads it, then each
    // model's artifacts (D-103). No lock, roles or device.
    config::LoadOptions load{.main_file = options->config,
                             .main_file_optional = !options->config_given,
                             .anchor = options->anchor,
                             .trusted_uid = ::geteuid()};
    auto loaded = config::LoadNodeConfig(load);
    if (!loaded) {
      for (const config::Diagnostic& diagnostic : loaded.error()) {
        Log(log, "configuration: " + config::FormatDiagnostic(diagnostic));
      }
      Log(log, "the configuration is not valid");
      return kExitRefused;
    }
    if (loaded->models.empty()) {
      Log(log, "the configuration names no models ([models.<name>], D-096)");
    }
    return PrintSettings(*loaded, options->command.json, stdout, log);
  }
  auto started = Start(*options, log);
  if (!started) {
    return started.error();
  }
  if (options->command.command != Command::kService) {
    // A serving command in this process: no readiness, no waiting. Run by
    // hand, it stops on SIGINT or SIGTERM at once (their default action,
    // unblocked before any of its threads start); the kernel then frees
    // its device memory and the unnamed spill files.
    sigset_t stop;
    (void)::sigemptyset(&stop);
    (void)::sigaddset(&stop, SIGTERM);
    (void)::sigaddset(&stop, SIGINT);
    (void)::sigaddset(&stop, SIGHUP);
    (void)::pthread_sigmask(SIG_UNBLOCK, &stop, nullptr);
    return serve(started->config, started->roles, options->command, stdout, log);
  }
  if (!started->config.models.empty()) {
    // The service with models to serve (D-097): they are registered, the
    // loopback chat route listens, and readiness follows; the stop signals
    // stay blocked, watched by the serving loop.
    return serve(started->config, started->roles, options->command, stdout, log);
  }
  // 10. Ready. Without models there is nothing to serve (the front door is
  // M5's).
  if (auto notified =
          platform::NotifyServiceManager("READY=1\nSTATUS=standalone node; nothing to serve yet");
      !notified) {
    Log(log, notified.error());
    return kExitFailure;
  }
  Log(log, "ready");

  sigset_t stop;
  (void)::sigemptyset(&stop);
  (void)::sigaddset(&stop, SIGTERM);
  (void)::sigaddset(&stop, SIGINT);
  (void)::sigaddset(&stop, SIGHUP);
  (void)::sigaddset(&stop, SIGCHLD);
  for (;;) {
    int signal = 0;
    if (::sigwait(&stop, &signal) != 0) {
      Log(log, "cannot wait for signals");
      return kExitFailure;
    }
    if (signal == SIGCHLD) {
      // Orphans re-parented to this subreaper (M1 starts no jobs of its own).
      (void)platform::ReapExited();
      continue;
    }
    if (signal == SIGHUP) {
      Log(log, "the configuration is read only at startup; restart the runtime to apply a change");
      continue;
    }
    break;
  }
  if (auto stopping = platform::NotifyServiceManager("STOPPING=1"); !stopping) {
    Log(log, stopping.error());
  }
  Log(log, "stopping: nothing to drain");
  return kExitOk;
}

}  // namespace llmp::runtime
