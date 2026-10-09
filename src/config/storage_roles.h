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
  // What the runtime starts despite (D-073 as amended on 2026-10-09): a
  // role another user owns, a spill or state directory others may read, an
  // installed one anyone may write, a default ACL.
  std::vector<std::string> warnings;
};

// Resolves the runtime's roles (links followed) and, before creating
// anything, checks that no two resolve to the same directory or nest,
// that none equals, contains or lies inside the enrollment anchor
// (resolved too), and that the job-only paths stay clear of them. Then
// creates any that are missing (with missing parents, 0755) as the calling
// user, who must be `runtime`: installed 0755, spill 0700 with its marker,
// state 0700, exactly. Then checks that each is a directory on a local
// filesystem the runtime's user can write, that installed and spill pass
// the direct-I/O probe, and that a non-empty spill directory has its
// marker. Every problem found is returned. An existing role's owner, mode
// and default ACL are never refused, only reported as warnings, since
// filesystem access control is its owner's.
std::expected<RuntimeRoles, std::vector<std::string>> PrepareRuntimeRoles(
    const Storage& storage, uid_t runtime, const std::filesystem::path& anchor);

// Adds the `storage` section: each role's path, whether it exists, its
// owner, mode and filesystem, and the problems (a runtime role that
// exists on a filesystem the runtime refuses, or read-only, or is not a
// directory) and warnings (one that does not exist yet and would be
// created on such a filesystem, and what PrepareRuntimeRoles warns of)
// found without touching anything. `runtime` is the runtime's user.
// writable_under, if not empty, is where the packaged unit's sandbox lets
// the runtime write (D-063): a runtime role outside it is reported.
void DescribeStorage(const Storage& storage, uid_t runtime,
                     const std::filesystem::path& writable_under, base::Report& report);

}  // namespace llmp::config

#endif  // LLMP_CONFIG_STORAGE_ROLES_H_
