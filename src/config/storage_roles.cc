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
#include "platform/kept_files.h"
#include "platform/paths.h"

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
  bool exact_mode;  // spill and state: exactly this mode when created, and private
  bool direct_io;   // installed and spill: D-034's probe
};

std::array<RoleSpec, 3> RuntimeRoleSpecs(const Storage& storage) {
  return {{
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
}

// What an existing role's owner, mode and default ACL leave open: never a
// reason to refuse (the filesystem's access control is its owner's, D-073
// as amended on 2026-10-09), but reported, by the runtime at start and by
// `llmp doctor`.
void NoteAccess(const RoleSpec& role, const std::string& label, const struct stat& status,
                uid_t runtime, const fs::path& path, std::vector<std::string>& warnings) {
  if (status.st_uid != runtime) {
    warnings.push_back(std::format("{} is owned by uid {}, not the runtime's user (uid {})", label,
                                   status.st_uid, runtime));
  }
  if (role.exact_mode && (status.st_mode & 077U) != 0) {
    warnings.push_back(
        std::format("{} has mode {}, not {}: other users may read the conversation state it holds",
                    label, platform::OctalMode(status.st_mode), platform::OctalMode(role.mode)));
  } else if (!role.exact_mode && (status.st_mode & S_IWOTH) != 0) {
    warnings.push_back(std::format("{} can be written by any user (mode {})", label,
                                   platform::OctalMode(status.st_mode)));
  }
  if (platform::HasDefaultAcl(path)) {
    warnings.push_back(std::format(
        "{} has a default ACL, which what the runtime creates inside inherits (setfacl -k removes "
        "it)",
        label));
  }
}

// An existing directory the runtime keeps state in beneath a role (the kept
// conversations and each model's directory in them, the calibration
// records: D-105, D-103) and how it departs from owner-only: a warning,
// never a refusal.
void NoteKept(const fs::path& dir, const std::string& label, bool descend,
              std::vector<std::string>& warnings) {
  const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) {
    return;
  }
  if (const std::string note = platform::PrivacyNote(fd); !note.empty()) {
    warnings.push_back(
        std::format("{} {}: {}; the runtime uses it as found", label, dir.string(), note));
  }
  if (descend) {
    if (const auto entries = platform::ListPrivateDirectory(fd)) {
      for (const platform::DirectoryEntry& entry : *entries) {
        if (!entry.regular) {  // a directory, or a link to one (followed)
          NoteKept(dir / entry.name, label, false, warnings);
        }
      }
    }
  }
  (void)::close(fd);
}

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
      // Another role's creation may have made a shared parent.
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
std::expected<void, std::string> CheckSpillMarker(const fs::path& spill) {
  const int dir = ::open(spill.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
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
    // Ours: a regular file holding what this version writes (through a
    // link or not). Never waiting on a FIFO, and nothing else read.
    const int fd = ::openat(dir, std::string(kSpillMarker).c_str(),
                            O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOCTTY);
    struct stat status{};
    std::array<char, 128> text{};
    const bool regular = fd >= 0 && ::fstat(fd, &status) == 0 && S_ISREG(status.st_mode);
    const ssize_t size = regular ? ::read(fd, text.data(), text.size()) : -1;
    if (fd >= 0) {
      (void)::close(fd);  // read-only
    }
    if (!regular || size < 0 ||
        std::string_view(text.data(), static_cast<std::size_t>(size)) != kSpillMarkerText) {
      problem = std::format(
          "{}/{} is not the marker this runtime writes (a regular file holding \"{}\")",
          spill.string(), kSpillMarker, kSpillMarkerText.substr(0, kSpillMarkerText.size() - 1));
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

}  // namespace

std::expected<RuntimeRoles, std::vector<std::string>> PrepareRuntimeRoles(const Storage& storage,
                                                                          uid_t runtime,
                                                                          const fs::path& anchor) {
  const std::array<RoleSpec, 3> roles = RuntimeRoleSpecs(storage);
  std::vector<std::string> problems;
  std::vector<std::string> warnings;
  // Each role as resolved, and whether it is still in the running.
  std::array<platform::ResolvedPath, 3> at{};
  std::array<bool, 3> ok{};

  // First, where each role is or would be, with nothing created yet.
  for (std::size_t i = 0; i < roles.size(); ++i) {
    auto resolved = platform::ResolvePath(*roles[i].path);
    if (!resolved) {
      problems.push_back(std::format("storage.{} ({}): {}", roles[i].name, roles[i].path->string(),
                                     resolved.error()));
      continue;
    }
    at[i] = std::move(*resolved);
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
  const auto anchor_at = platform::ResolvePath(anchor);
  if (!anchor_at) {
    problems.push_back(
        std::format("the enrollment anchor {}: {}", anchor.string(), anchor_at.error()));
    return std::unexpected(problems);
  }
  for (std::size_t i = 0; i < roles.size(); ++i) {
    if (ok[i] && (Within(anchor_at->resolved, at[i].resolved) ||
                  Within(at[i].resolved, anchor_at->resolved))) {
      problems.push_back(
          std::format("storage.{} resolves to {}, which equals, contains or lies inside the "
                      "enrollment anchor {}",
                      roles[i].name, at[i].resolved.string(), anchor_at->resolved.string()));
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

  // Then create what is missing, private where the role is (spill and
  // state 0700, exactly), and resolve it again.
  for (std::size_t i = 0; i < roles.size(); ++i) {
    const RoleSpec& role = roles[i];
    const std::string label = std::format("storage.{} ({})", role.name, role.path->string());
    if (!at[i].exists) {
      if (auto created = Create(at[i].existing, at[i].resolved, role.mode); !created) {
        problems.push_back(std::format("{}: {}", label, created.error()));
        ok[i] = false;
        continue;
      }
      auto resolved = platform::ResolvePath(*role.path);
      if (!resolved || !resolved->exists || resolved->resolved != at[i].resolved) {
        problems.push_back(std::format("{} changed while it was being created", label));
        ok[i] = false;
        continue;
      }
      at[i] = std::move(*resolved);
    }
    if (!S_ISDIR(at[i].status.st_mode)) {
      problems.push_back(std::format("{} is not a directory", label));
      ok[i] = false;
      continue;
    }
    NoteAccess(role, label, at[i].status, runtime, at[i].resolved, warnings);
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
    // The runtime writes every role: through its owner's, group's or
    // others' bits or an ACL, whichever grants it.
    if (::faccessat(AT_FDCWD, path.c_str(), W_OK | X_OK, AT_EACCESS) != 0) {
      problems.push_back(std::format("storage.{} ({}) cannot be written by the runtime's user: {}",
                                     role.name, path.string(), Errno(errno)));
      continue;
    }
    if (role.direct_io) {
      if (auto probe = platform::ProbeDirectIo(path); !probe) {
        problems.push_back(std::format("storage.{}: {}", role.name, probe.error()));
        continue;
      }
    }
    if (role.name == "spill") {
      if (auto marker = CheckSpillMarker(path); !marker) {
        problems.push_back(std::format("storage.spill: {}", marker.error()));
      }
    }
  }
  if (!problems.empty()) {
    return std::unexpected(problems);
  }
  return RuntimeRoles{.installed = at[0].resolved,
                      .spill = at[1].resolved,
                      .state = at[2].resolved,
                      .warnings = std::move(warnings)};
}

std::string DirectIoCaveat(std::string_view role, const fs::path& path,
                           std::string_view filesystem) {
  const Storage defaults;
  const auto roles = RuntimeRoleSpecs(defaults);
  const bool direct = std::ranges::any_of(
      roles, [&](const RoleSpec& spec) { return spec.name == role && spec.direct_io; });
  if (!direct || filesystem != "btrfs") {
    return "";
  }
  return std::format(
      "storage.{} ({}) is on btrfs, which serves misaligned or compressed direct reads through "
      "the page cache without saying so (RE-018), so page-ins from it may fill the cache that "
      "D-034 keeps weights out of: ext4 or xfs does not",
      role, path.string());
}

void DescribeStorage(const Storage& storage, uid_t runtime, const fs::path& writable_under,
                     base::Report& report) {
  base::ReportSection& section = report.AddSection("storage");
  section.Add("data_dir", storage.data_dir.string());
  if (storage.long_term) {
    section.Add("long_term", storage.long_term->string() + " (job processes only; not examined)");
  }
  for (const RoleSpec& role : RuntimeRoleSpecs(storage)) {
    auto resolved = platform::ResolvePath(*role.path);
    if (!resolved) {
      section.Add(std::string(role.name), role.path->string());
      report.problems.push_back(
          std::format("storage.{} ({}): {}", role.name, role.path->string(), resolved.error()));
      continue;
    }
    const fs::path& at = resolved->exists ? resolved->resolved : resolved->existing;
    const auto filesystem = platform::DescribeFilesystem(at);
    std::string facts = role.path->string();
    if (resolved->resolved != *role.path) {
      facts += std::format(" -> {}", resolved->resolved.string());
    }
    if (resolved->exists) {
      facts += std::format(", owner uid {}, mode {}", resolved->status.st_uid,
                           platform::OctalMode(resolved->status.st_mode));
    } else {
      facts += std::format(", not created yet (the runtime creates it under {})",
                           resolved->existing.string());
    }
    if (filesystem) {
      facts += std::format(", {}{}", filesystem->type, filesystem->read_only ? " (read-only)" : "");
    }
    section.Add(std::string(role.name), facts);
    if (resolved->exists) {
      const std::string label = std::format("storage.{} ({})", role.name, role.path->string());
      if (!S_ISDIR(resolved->status.st_mode)) {
        report.problems.push_back(std::format("{} is not a directory", label));
      } else {
        NoteAccess(role, label, resolved->status, runtime, resolved->resolved, report.warnings);
        if (role.name == "spill") {
          NoteKept(resolved->resolved / kKeptConversationsDirectory,
                   "storage.spill: kept conversations", true, report.warnings);
        } else if (role.name == "state") {
          NoteKept(resolved->resolved / kCalibrationDirectory, "storage.state: calibration", false,
                   report.warnings);
        }
      }
    }
    if (!filesystem) {
      report.warnings.push_back(std::format("storage.{}: {}", role.name, filesystem.error()));
      continue;
    }
    std::vector<std::string>& judged = resolved->exists ? report.problems : report.warnings;
    const std::string where = resolved->exists ? "is" : "would be created";
    if (!filesystem->accepted) {
      judged.push_back(
          std::format("storage.{} {} on {}, which the runtime refuses: it needs a local "
                      "block-device filesystem (ext4, xfs or btrfs)",
                      role.name, where, filesystem->type));
    } else if (filesystem->read_only) {
      judged.push_back(std::format("storage.{} {} on a read-only filesystem", role.name, where));
    } else if (std::string caveat = DirectIoCaveat(role.name, resolved->resolved, filesystem->type);
               !caveat.empty()) {
      report.warnings.push_back(std::move(caveat));
    }
    if (!writable_under.empty() && !Within(writable_under, resolved->resolved)) {
      report.warnings.push_back(
          std::format("storage.{} ({}) is outside {}, which is all the packaged "
                      "llmp.service can write: add ReadWritePaths={} in a drop-in",
                      role.name, resolved->resolved.string(), writable_under.string(),
                      resolved->resolved.string()));
    }
  }
  section.Add("checkpoints", storage.checkpoints.string() + " (job processes only; not examined)");
  if (storage.archive) {
    section.Add("archive", storage.archive->string() + " (job processes only; not examined)");
  }
}

}  // namespace llmp::config
