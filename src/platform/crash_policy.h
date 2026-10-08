// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a crash may leave behind (D-014; docs/architecture.md#errors-faults-startup-and-shutdown):
// no core image that could hold request bodies or state. The hosts pipe
// core dumps to apport, which ignores RLIMIT_CORE and, with
// fs.suid_dumpable=2, still receives a non-dumpable process's dump
// (docs/environment.md#crash-dump-handling-2026-09-23). So the runtime
// never lets a fatal signal kill it: a handler writes one bounded line and
// ends the process with _exit(128 + signal), and the kernel never starts a
// core dump at all. As defence in depth, the process is also marked
// non-dumpable and its core-dump filter excludes every mapping, before
// main() (MarkNonDumpable, from a high-priority constructor) and again
// by InstallCrashPolicy, so a dump the handler cannot prevent holds no
// memory contents, only registers and file names. The handler cannot run
// for a fault before it is installed, for a fault inside itself, for a
// hardware fault a thread has blocked, or for a stack overflow on a thread
// without an alternate signal stack: every thread llmpalooza starts calls
// InstallThreadSignalStack first.

#ifndef LLMP_PLATFORM_CRASH_POLICY_H_
#define LLMP_PLATFORM_CRASH_POLICY_H_

#include <expected>
#include <string>
#include <string_view>

namespace llmp::platform {

// Exits with this status plus the signal number after a fatal signal, as a
// shell reports a death by signal.
inline constexpr int kFatalSignalExitBase = 128;

// Marks the process non-dumpable and clears its core-dump filter; what
// InstallCrashPolicy does first, callable earlier (from a constructor).
std::expected<void, std::string> MarkNonDumpable();

// Gives the calling thread its own alternate signal stack, so the handler
// runs after a stack overflow there. Every thread llmpalooza starts calls it
// before anything else; the stack lives as long as the thread.
std::expected<void, std::string> InstallThreadSignalStack();

// Installs the policy for the calling process. `name` starts the line the
// handler writes to standard error. Call it first thing in main(), before
// any thread starts; it is not undone. The alternate signal stack, which
// lets the handler run after a stack overflow, is the calling thread's:
// threads started later need their own.
std::expected<void, std::string> InstallCrashPolicy(std::string_view name);

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_CRASH_POLICY_H_
