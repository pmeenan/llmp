# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Linux services (docs/architecture.md#layers-and-dependency-rules), the one
# place, with the Linux provider implementations, that Linux-specific
# headers and calls appear (docs/portability.md, tools/llmp_boundaries.py):
# small reads of /proc and /sys, the host half of the capability probe and
# the host's available memory, the trust walk that checks who could replace
# a path, the storage roles' filesystem and direct-I/O probe and the
# direct-I/O opens, and process services: the crash policy, lock files, the
# service manager's readiness protocol, and job processes in delegated
# cgroups with their confinement; a minimal io_uring ring; the event loop,
# wakers and signal watch the chat route runs on, its socket calls and
# random bytes; the node's interface addresses and names for its
# listeners; memory pressure from outside the runtime (MemAvailable,
# PSI) with the heap's free pages returned; and the owner-only files kept
# across a restart (D-105).
add_library(llmp_platform STATIC files.cc host_probe.cc path_trust.cc direct_io.cc crash_policy.cc
                                   lock_file.cc sd_notify.cc job.cc confine.cc io_uring.cc
                                   interfaces.cc event_loop.cc sockets.cc memory_pressure.cc
                                   kept_files.cc)
target_include_directories(llmp_platform PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_platform PUBLIC llmp_report PRIVATE llmp_warnings)
