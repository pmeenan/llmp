// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "config/storage_roles.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/report.h"
#include "config/node_config.h"
#include "platform/direct_io.h"
#include "platform/path_trust.h"

namespace llmp::config {
namespace {

namespace fs = std::filesystem;

// What the marker holds: its format's version, for D-062.
constexpr std::string_view kSpillMarkerText = "llmp spill directory, version 1\n";

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

bool Within(const fs::path& a, const fs::path& b) {
  return std::ranges::mismatch(a, b).in1 == a.end();
}

struct RoleSpec {
  std::string_view name;
  const fs::path* path;
  mode_t mode;
  bool exact_mode;  // spill and state: exactly this mode
  bool direct_io;   // installed and spill: D-034's probe
};

// Creates the missing directories from `existing` down to `path`, parents
// 0755 and the last one `mode`, exactly (not through the umask).
std::expected<void, std::string> Create(const fs::path& existing, const fs::path& path,
                                        mode_t mode) {
  fs::path at = existing;
  const fs::path tail = path.lexically_relative(existing);
  for (auto part = tail.begin(); part != tail.end(); ++part) {
    at /= *part;
    const bool last = std::next(part) == tail.end();
    const mode_t want = last ? mode : 0755;
    if (::mkdir(at.c_str(), want) != 0) {
      // Another role's creation may have made a shared parent; the walk
      // after creation checks it like any other.
      if (!last && errno == EEXIST) {
        continue;
      }
      return std::unexpected(std::format("cannot create {}: {}", at.string(), Errno(errno)));
    }
    if (::chmod(at.c_str(), want) != 0) {
      return std::unexpected(
          std::format("cannot set the mode of {}: {}", at.string(), Errno(errno)));
    }
  }
  return {};
}

// A spill directory is empty or holds its marker (D-055); an empty one
// gets the marker.
std::expected<void, std::string> CheckSpillMarker(const fs::path& spill, uid_t runtime) {
  const int dir = ::open(spill.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (dir < 0) {
    return std::unexpected(std::format("cannot open {}: {}", spill.string(), Errno(errno)));
  }
  std::optional<std::string> problem;
  bool empty = true;
  bool marked = false;
  const int listing = ::dup(dir);
  DIR* stream = listing < 0 ? nullptr : ::fdopendir(listing);
  if (stream == nullptr) {
    if (listing >= 0) {
      (void)::close(listing);
    }
    problem = std::format("cannot list {}: {}", spill.string(), Errno(errno));
  } else {
    errno = 0;
    while (const dirent* entry = ::readdir(stream)) {  // NOLINT(concurrency-mt-unsafe): one reader
      const std::string_view name = entry->d_name;
      if (name == "." || name == "..") {
        continue;
      }
      empty = false;
      if (name == kSpillMarker) {
        marked = true;
      }
    }
    if (errno != 0) {
      // A directory not fully listed is not known to be empty.
      problem = std::format("cannot list {}: {}", spill.string(), Errno(errno));
    }
    (void)::closedir(stream);
  }
  if (!problem && marked) {
    // Ours: a regular file owned by the runtime's user, holding what this
    // version writes.
    const int fd = ::openat(dir, std::string(kSpillMarker).c_str(),
                            O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    struct stat status{};
    std::array<char, 128> text{};
    const ssize_t size = fd < 0 ? -1 : ::read(fd, text.data(), text.size());
    const bool regular =
        fd >= 0 && ::fstat(fd, &status) == 0 && S_ISREG(status.st_mode) && status.st_uid == runtime;
    if (fd >= 0) {
      (void)::close(fd);  // read-only
    }
    if (!regular || size < 0 ||
        std::string_view(text.data(), static_cast<std::size_t>(size)) != kSpillMarkerText) {
      problem = std::format(
          "{}/{} is not the marker this runtime writes (a regular file owned by uid {} holding "
          "\"{}\")",
          spill.string(), kSpillMarker, runtime,
          kSpillMarkerText.substr(0, kSpillMarkerText.size() - 1));
    }
  } else if (!problem && !empty) {
    problem = std::format(
        "{} is not empty and has no {} marker, so it may hold someone else's files; point "
        "storage.spill at a new or empty directory (D-055)",
        spill.string(), kSpillMarker);
  } else if (!problem) {
    const int fd = ::openat(dir, std::string(kSpillMarker).c_str(),
                            O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) {
      problem = std::format("cannot create {}/{}: {}", spill.string(), kSpillMarker, Errno(errno));
    } else {
      const auto wrote = ::write(fd, kSpillMarkerText.data(), kSpillMarkerText.size());
      if (std::cmp_not_equal(wrote, kSpillMarkerText.size()) || ::close(fd) != 0) {
        problem = std::format("cannot write {}/{}", spill.string(), kSpillMarker);
      }
    }
  }
  (void)::close(dir);
  if (problem) {
    return std::unexpected(*problem);
  }
  return {};
}

std::string Owner(uid_t uid) { return std::format("uid {}", uid); }

}  // namespace

std::expected<RuntimeRoles, std::vector<std::string>> PrepareRuntimeRoles(const Storage& storage,
                                                                          uid_t runtime,
                                                                          const fs::path& anchor) {
  const std::array<RoleSpec, 3> roles = {{
      {.name = "installed",
       .path = &storage.installed,
       .mode = 0755,
       .exact_mode = false,
       .direct_io = true},
      {.name = "spill",
       .path = &storage.spill,
       .mode = 0700,
       .exact_mode = true,
       .direct_io = true},
      {.name = "state",
       .path = &storage.state,
       .mode = 0700,
       .exact_mode = true,
       .direct_io = false},
  }};
  std::vector<std::string> problems;
  // Each role as walked, and whether it is still in the running.
  std::array<platform::TrustedPath, 3> at{};
  std::array<bool, 3> ok{};

  // First, where each role is or would be, with nothing created yet.
  for (std::size_t i = 0; i < roles.size(); ++i) {
    auto walked = platform::WalkTrusted(*roles[i].path, runtime, false);
    if (!walked) {
      problems.push_back(std::format("storage.{} ({}): {}", roles[i].name, roles[i].path->string(),
                                     walked.error()));
      continue;
    }
    at[i] = std::move(*walked);
    ok[i] = true;
  }

  // Links and bind mounts cannot make two roles one directory, and no role
  // may resolve onto the enrollment anchor (D-063).
  for (std::size_t i = 0; i < roles.size(); ++i) {
    for (std::size_t j = i + 1; j < roles.size(); ++j) {
      if (!ok[i] || !ok[j]) {
        continue;
      }
      const fs::path& a = at[i].resolved;
      const fs::path& b = at[j].resolved;
      const bool same_inode = at[i].exists && at[j].exists &&
                              at[i].status.st_dev == at[j].status.st_dev &&
                              at[i].status.st_ino == at[j].status.st_ino;
      if (same_inode || a == b) {
        problems.push_back(std::format("storage.{} and storage.{} are the same directory ({})",
                                       roles[i].name, roles[j].name, a.string()));
        ok[j] = false;
      } else if (Within(a, b) || Within(b, a)) {
        problems.push_back(std::format(
            "storage.{} resolves to {} and storage.{} to {}; no role may contain another",
            roles[i].name, a.string(), roles[j].name, b.string()));
        ok[j] = false;
      }
    }
  }
  const auto anchor_walk = platform::WalkTrusted(anchor, runtime, true);
  if (!anchor_walk) {
    problems.push_back(
        std::format("the enrollment anchor {}: {}", anchor.string(), anchor_walk.error()));
    return std::unexpected(problems);
  }
  const fs::path& anchor_at = anchor_walk->resolved;
  for (std::size_t i = 0; i < roles.size(); ++i) {
    if (ok[i] && (Within(anchor_at, at[i].resolved) || Within(at[i].resolved, anchor_at))) {
      problems.push_back(
          std::format("storage.{} resolves to {}, which equals, contains or lies inside the "
                      "enrollment anchor {}",
                      roles[i].name, at[i].resolved.string(), anchor_at.string()));
      ok[i] = false;
    }
  }
  // The job-only paths, by text only (D-054).
  struct Other {
    std::string_view name;
    const fs::path* path;
  };
  std::vector<Other> others = {{.name = "checkpoints", .path = &storage.checkpoints}};
  if (storage.archive) {
    others.push_back({.name = "archive", .path = &*storage.archive});
  }
  if (storage.long_term) {
    others.push_back({.name = "long_term", .path = &*storage.long_term});
  }
  for (std::size_t i = 0; i < roles.size(); ++i) {
    if (!ok[i]) {
      continue;
    }
    for (const Other& other : others) {
      if (Within(at[i].resolved, *other.path) || Within(*other.path, at[i].resolved)) {
        problems.push_back(std::format("storage.{} ({}) overlaps storage.{}, which resolves to {}",
                                       other.name, other.path->string(), roles[i].name,
                                       at[i].resolved.string()));
      }
    }
  }
  if (!problems.empty()) {
    return std::unexpected(problems);
  }

  // Then create what is missing, and walk again: each role must be where
  // the first walk said, and as its role requires.
  for (std::size_t i = 0; i < roles.size(); ++i) {
    const RoleSpec& role = roles[i];
    const std::string label = std::format("storage.{} ({})", role.name, role.path->string());
    if (!at[i].exists) {
      if (auto created = Create(at[i].existing, at[i].resolved, role.mode); !created) {
        problems.push_back(std::format("{}: {}", label, created.error()));
        ok[i] = false;
        continue;
      }
      auto walked = platform::WalkTrusted(*role.path, runtime, false);
      if (!walked || !walked->exists || walked->resolved != at[i].resolved) {
        problems.push_back(std::format("{} changed while it was being created", label));
        ok[i] = false;
        continue;
      }
      at[i] = std::move(*walked);
    }
    const struct stat& status = at[i].status;
    std::optional<std::string> problem;
    if (!S_ISDIR(status.st_mode)) {
      problem = std::format("{} is not a directory", label);
    } else if (status.st_uid != runtime) {
      problem = std::format("{} is owned by {}, not the runtime's user ({})", label,
                            Owner(status.st_uid), Owner(runtime));
    } else if (role.exact_mode && (status.st_mode & 07777U) != role.mode) {
      problem = std::format("{} has mode {}, not {}", label, platform::OctalMode(status.st_mode),
                            platform::OctalMode(role.mode));
    } else if (platform::OthersCanWrite(status, runtime, at[i].resolved)) {
      problem = std::format("{} can be written by users other than root and {} (mode {})", label,
                            Owner(runtime), platform::OctalMode(status.st_mode));
    } else if (platform::HasDefaultAcl(at[i].resolved)) {
      // What the runtime later creates inside would inherit it.
      problem = std::format("{} has a default ACL; remove it (setfacl -k)", label);
    }
    if (problem) {
      problems.push_back(*problem);
      ok[i] = false;
    }
  }

  for (std::size_t i = 0; i < roles.size(); ++i) {
    if (!ok[i]) {
      continue;
    }
    const RoleSpec& role = roles[i];
    const fs::path& path = at[i].resolved;
    const auto filesystem = platform::DescribeFilesystem(path);
    if (!filesystem) {
      problems.push_back(std::format("storage.{}: {}", role.name, filesystem.error()));
      continue;
    }
    if (!filesystem->accepted) {
      problems.push_back(
          std::format("storage.{} ({}) is on {}, not a local block-device filesystem (ext4, xfs or "
                      "btrfs)",
                      role.name, path.string(), filesystem->type));
      continue;
    }
    if (filesystem->read_only) {
      problems.push_back(
          std::format("storage.{} ({}) is on a read-only filesystem; a role outside /var/lib/llmp "
                      "needs ReadWritePaths= in a llmp.service drop-in",
                      role.name, path.string()));
      continue;
    }
    if (role.direct_io) {
      if (auto probe = platform::ProbeDirectIo(path); !probe) {
        problems.push_back(std::format("storage.{}: {}", role.name, probe.error()));
        continue;
      }
    }
    if (role.name == "spill") {
      if (auto marker = CheckSpillMarker(path, runtime); !marker) {
        problems.push_back(std::format("storage.spill: {}", marker.error()));
      }
    }
  }
  if (!problems.empty()) {
    return std::unexpected(problems);
  }
  return RuntimeRoles{
      .installed = at[0].resolved, .spill = at[1].resolved, .state = at[2].resolved};
}

void DescribeStorage(const Storage& storage, uid_t trusted, const fs::path& writable_under,
                     base::Report& report) {
  base::ReportSection& section = report.AddSection("storage");
  section.Add("data_dir", storage.data_dir.string());
  struct Role {
    std::string_view name;
    const fs::path* path;
    bool runtime;
  };
  std::vector<Role> roles = {
      {.name = "installed", .path = &storage.installed, .runtime = true},
      {.name = "spill", .path = &storage.spill, .runtime = true},
      {.name = "state", .path = &storage.state, .runtime = true},
      {.name = "checkpoints", .path = &storage.checkpoints, .runtime = false}};
  if (storage.long_term) {
    section.Add("long_term", storage.long_term->string() + " (job processes only; not examined)");
  }
  if (storage.archive) {
    roles.push_back({.name = "archive", .path = &*storage.archive, .runtime = false});
  }
  for (const Role& role : roles) {
    if (!role.runtime) {
      section.Add(std::string(role.name),
                  role.path->string() + " (job processes only; not examined)");
      continue;
    }
    auto walked = platform::WalkTrusted(*role.path, trusted, false);
    if (!walked) {
      section.Add(std::string(role.name), role.path->string());
      report.problems.push_back(
          std::format("storage.{} ({}): {}", role.name, role.path->string(), walked.error()));
      continue;
    }
    const fs::path& at = walked->exists ? walked->resolved : walked->existing;
    const auto filesystem = platform::DescribeFilesystem(at);
    std::string facts = role.path->string();
    if (walked->resolved != *role.path) {
      facts += std::format(" -> {}", walked->resolved.string());
    }
    if (walked->exists) {
      facts += std::format(", owner uid {}, mode {}", walked->status.st_uid,
                           platform::OctalMode(walked->status.st_mode));
    } else {
      facts += std::format(", not created yet (the runtime creates it under {})",
                           walked->existing.string());
    }
    if (filesystem) {
      facts += std::format(", {}{}", filesystem->type, filesystem->read_only ? " (read-only)" : "");
    }
    section.Add(std::string(role.name), facts);
    if (!filesystem) {
      report.warnings.push_back(std::format("storage.{}: {}", role.name, filesystem.error()));
      continue;
    }
    std::vector<std::string>& judged = walked->exists ? report.problems : report.warnings;
    const std::string where = walked->exists ? "is" : "would be created";
    if (!filesystem->accepted) {
      judged.push_back(
          std::format("storage.{} {} on {}, which the runtime refuses: it needs a local "
                      "block-device filesystem (ext4, xfs or btrfs)",
                      role.name, where, filesystem->type));
    } else if (filesystem->read_only) {
      judged.push_back(std::format("storage.{} {} on a read-only filesystem", role.name, where));
    }
    if (!writable_under.empty() && !Within(writable_under, walked->resolved)) {
      report.warnings.push_back(
          std::format("storage.{} ({}) is outside {}, which is all the packaged "
                      "llmp.service can write: add ReadWritePaths={} in a drop-in",
                      role.name, walked->resolved.string(), writable_under.string(),
                      walked->resolved.string()));
    }
  }
}

}  // namespace llmp::config
