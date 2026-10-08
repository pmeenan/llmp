// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Job processes contained in delegated cgroups (D-074;
// docs/architecture.md#import-install-and-archive-jobs). llmp.service
// delegates its cgroup to the runtime's user and runs the runtime one level
// down (Delegate=, DelegateSubgroup=runtime), so the runtime can make a
// cgroup per job beside its own: <unit>/jobs/<id>. Every process a job
// starts stays in that cgroup, however it forks, daemonizes or re-parents,
// until it exits, and the cgroup can be killed whole.
//
// Each job also holds its record's lock (flock) from before it starts: the
// descriptor is inherited by every process of the job, across exec, and by
// nothing else. A job has ended only when its cgroup has no process left
// and its lock is free (JobEnded). A process group alone contains nothing:
// setsid() leaves it. This contains ordinary process trees; a hostile job
// running unconfined as the runtime's user could move itself to another
// cgroup of the delegated tree, which is why stages that parse untrusted
// input also confine themselves (platform/confine.h), leaving them no
// access to /sys/fs/cgroup.

#ifndef LLMP_PLATFORM_JOB_H_
#define LLMP_PLATFORM_JOB_H_

#include <sys/types.h>

#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace llmp::platform {

// The name of the environment variable that tells a job which descriptor
// holds its lock.
inline constexpr std::string_view kJobLockFdVariable = "LLMP_JOB_LOCK_FD";

// The cgroup v2 directory the calling process is in, under /sys/fs/cgroup.
std::expected<std::filesystem::path, std::string> OwnCgroup();

// <parent of the calling process's cgroup>/jobs, created (0755) if
// missing: where a delegated runtime puts its jobs. Fails, creating
// nothing, unless the calling process runs in a cgroup named `runtime`, as
// DelegateSubgroup=runtime places it.
std::expected<std::filesystem::path, std::string> JobsCgroupRoot();

// Makes the calling process a child subreaper (prctl(2)): orphans of its
// jobs re-parent to it, so it reaps them.
std::expected<void, std::string> BecomeSubreaper();

struct StartedJob {
  pid_t pid = -1;
  std::filesystem::path cgroup;
};

// Starts argv[0] (an absolute path) with argv as a job: a new cgroup
// jobs_root/<id>, and the lock file (created 0600 if missing), locked
// before the job starts and inherited by it on the descriptor that
// LLMP_JOB_LOCK_FD names. The caller keeps no descriptor for the lock.
// Fails if the lock is held or the cgroup exists.
std::expected<StartedJob, std::string> StartJob(const std::filesystem::path& jobs_root,
                                                std::string_view id,
                                                const std::filesystem::path& lock_file,
                                                std::span<const std::string> argv);

// Whether any process remains in a job's cgroup (cgroup.events).
std::expected<bool, std::string> CgroupPopulated(const std::filesystem::path& cgroup);

// Kills every process in a job's cgroup (cgroup.kill), at once.
std::expected<void, std::string> KillCgroup(const std::filesystem::path& cgroup);

// Whether nobody holds a job's lock: taken and released at once through a
// descriptor of its own. A missing lock file is free.
std::expected<bool, std::string> LockIsFree(const std::filesystem::path& lock_file);

// Whether a job has ended: its cgroup is gone or empty, and its lock free.
std::expected<bool, std::string> JobEnded(const std::filesystem::path& cgroup,
                                          const std::filesystem::path& lock_file);

// Reaps every child that has exited (waitpid, WNOHANG); returns how many.
// One reaper only: before jobs start from a thread other than the one that
// reaps, StartJob's own waits must go through it (or through pidfds).
int ReapExited();

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_JOB_H_
