// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The storage roles on the filesystem (D-054, D-055, D-063): what the
// runtime checks, and creates when missing, before it uses its own three
// roles (installed, spill and state), and what `llmp doctor` reports
// about them without changing anything.
//
// The runtime opens only its own roles. It compares the job-only paths
// (checkpoints, archive and long_term) with its resolved roles by text
// alone, so that a hung mount cannot stall startup.

#ifndef LLMP_CONFIG_STORAGE_ROLES_H_
#define LLMP_CONFIG_STORAGE_ROLES_H_

#include <sys/types.h>

#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "base/report.h"
#include "config/node_config.h"

namespace llmp::config {

// The spill directory's marker file (D-055): startup refuses a non-empty
// spill directory without it.
inline constexpr std::string_view kSpillMarker = ".llmp-spill";

struct RuntimeRoles {
  // Each role resolved, with links followed.
  std::filesystem::path installed;
  std::filesystem::path spill;
  std::filesystem::path state;
};

// Resolves the runtime's roles and, before creating anything, checks that
// nobody but root and the runtime's user could replace them or the way to
// them, that no two resolve to the same directory or nest, that none
// equals, contains or lies inside the enrollment anchor (resolved too),
// and that the job-only paths stay clear of them. Then creates any that
// are missing (with missing parents, 0755) owned by the calling user, who
// must be `runtime`: installed 0755, spill 0700 with its marker, state
// 0700. Then checks that the runtime's user owns them with those modes
// (spill and state exactly; installed writable by nobody else), that they
// are on a writable local filesystem, that installed and spill pass the
// direct-I/O probe, and that a non-empty spill directory has its marker.
// Every problem found is returned.
std::expected<RuntimeRoles, std::vector<std::string>> PrepareRuntimeRoles(
    const Storage& storage, uid_t runtime, const std::filesystem::path& anchor);

// Adds the `storage` section: each role's path, whether it exists, its
// owner, mode and filesystem, and the problems (a runtime role that
// exists on a filesystem the runtime refuses, or read-only) and warnings
// (one that does not exist yet and would be created on such a
// filesystem) found without touching anything. `trusted` is the runtime's
// user. writable_under, if not empty, is where the packaged unit's sandbox
// lets the runtime write (D-063): a runtime role outside it is reported.
void DescribeStorage(const Storage& storage, uid_t trusted,
                     const std::filesystem::path& writable_under, base::Report& report);

}  // namespace llmp::config

#endif  // LLMP_CONFIG_STORAGE_ROLES_H_
