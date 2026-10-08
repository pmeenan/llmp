// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Confinement for the job stages that parse untrusted input (D-074;
// docs/architecture.md#import-install-and-archive-jobs): such a stage may
// read only its inputs and write only its own staging directory, with no
// network. Unprivileged namespaces are blocked on the hosts (RE-013), so
// this uses what an unprivileged process can apply to itself: a Landlock
// ruleset (ABI 6) for the filesystem, TCP, abstract unix sockets and
// signals to processes outside the stage, and a seccomp filter that
// refuses creating sockets, directly or through io_uring. Both are
// irreversible and inherited by every child and across exec. The stage is
// also made non-dumpable, since exec made it dumpable again. "No network"
// means no new sockets and no TCP through any: a job must not hand a stage
// an open socket, since UDP through one would still work. Landlock
// does not cover metadata (chmod, chown, times, extended attributes) of
// the files a stage can reach, nor descriptors it inherited: the job must
// start it with only the descriptors it needs.

#ifndef LLMP_PLATFORM_CONFINE_H_
#define LLMP_PLATFORM_CONFINE_H_

#include <expected>
#include <filesystem>
#include <span>
#include <string>

namespace llmp::platform {

struct Confinement {
  // Trees the stage may read (files and directory listings) and execute.
  std::span<const std::filesystem::path> read;
  // Trees the stage may also write, create in and remove from.
  std::span<const std::filesystem::path> write;
};

// Confines the calling thread and what it starts: filesystem access
// outside `read` and `write` fails with EACCES, as do socket(2),
// socketpair(2) and io_uring; TCP binds and connects and signals to other
// processes are refused. Fails if the kernel lacks Landlock ABI 6 (Linux
// 6.12) or seccomp; a failure part way can leave part of the
// confinement applied, so a job that cannot be confined must exit. Call it
// in a single-threaded job process before it touches untrusted input.
std::expected<void, std::string> ConfineSelf(const Confinement& confinement);

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_CONFINE_H_
