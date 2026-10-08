// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The node runtime process (D-005; /usr/libexec/llmp/llmp-runtime,
// started by llmp.service). In M1 it runs the startup order of
// docs/architecture.md#errors-faults-startup-and-shutdown as far as this
// build has steps for, reports readiness, and waits to be stopped:
//
//  1. the enrollment anchor: this build has no cluster support, so it
//     refuses to start while one exists (D-063);
//  2. the node's configuration (D-073), refusing a cluster member's;
//  3. the per-node process lock, `<anchor>.lock` beside the anchor, which
//     the unit's StateDirectory keeps and which only the runtime's user can
//     open;
//  4-5. its storage roles, created and checked, with D-034's direct-I/O
//     probe and the spill marker (config/storage_roles.h);
//  (6.) the job side of step 6: it becomes the subreaper for its jobs'
//     orphans and finds the delegated cgroup its jobs would run in
//     (platform/job.h), warning if there is none;
//  7. the platform: the host and device probes that `llmp doctor` runs,
//     refusing (for now) a host with problems, and an open of each RDMA device node,
//     which shows whether the unit's sandbox lets it through;
//  10. readiness (sd_notify).
//
// Steps 6, 8 and 9 (job records, the artifact index, authority records)
// have nothing to act on yet. Stopping (SIGTERM or SIGINT) has nothing to
// drain yet either.
//
// With a serving command (commands.h: chat, swap-table; D-096), the same
// startup steps run, then instead of readiness the process registers the
// configured models on its node, runs the command and exits; it holds the
// process lock throughout, so it never runs beside the service.
//
// The service with models configured (D-097) registers them the same way,
// serves the loopback chat route (api_server.h) and then reports
// readiness; without models it waits as above.

#ifndef LLMP_RUNTIME_RUNTIME_H_
#define LLMP_RUNTIME_RUNTIME_H_

#include <cstdio>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

#include "config/node_config.h"
#include "config/storage_roles.h"
#include "platform/lock_file.h"
#include "runtime/commands.h"

namespace llmp::runtime {

// Exit statuses. A refusal at startup that a restart would only repeat
// (configuration, anchor, lock, roles) is kExitRefused, EX_CONFIG from
// sysexits.h, which the unit does not restart after
// (RestartPreventExitStatus=). A host that cannot run this build now (the
// driver or its devices, perhaps not ready at boot) is kExitHostNotReady,
// EX_TEMPFAIL, which it does.
inline constexpr int kExitOk = 0;
inline constexpr int kExitFailure = 1;
inline constexpr int kExitUsage = 2;
inline constexpr int kExitHostNotReady = 75;
inline constexpr int kExitRefused = 78;

struct Options {
  // The configuration's main file (D-063): the packaged default, which may
  // be absent, unless --config names one, which must exist.
  std::filesystem::path config{config::kDefaultConfigFile};
  bool config_given = false;
  // The enrollment anchor; development runs name their own (D-063).
  std::filesystem::path anchor{config::kDefaultAnchor};
  bool help = false;
  // A serving command (after the options); kService without one.
  // Initialized so callers may designate only the fields before it.
  // NOLINTNEXTLINE(readability-redundant-member-init)
  CommandOptions command = {};
};

std::expected<Options, std::string> ParseArguments(std::span<const std::string_view> args);

// What a started runtime holds until it stops.
struct Started {
  platform::LockFile lock;
  config::RuntimeRoles roles;
  config::NodeConfig config;
};

// Runs the startup steps, writing what it does and every problem to log.
// Returns the exit status on refusal.
std::expected<Started, int> Start(const Options& options, std::FILE* log);

// The whole program: arguments, startup, readiness, then waiting for
// SIGTERM or SIGINT, which the caller has blocked in every thread; or with
// a serving command, `serve` (commands.h RunServing, which main links from
// the serving library) instead of readiness.
using ServeFunction = int (*)(const config::NodeConfig&, const config::RuntimeRoles&,
                              const CommandOptions&, std::FILE*, std::FILE*);
int Run(std::span<const std::string_view> args, std::FILE* log, ServeFunction serve);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_RUNTIME_H_
