// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/path_trust.h"

#include <grp.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <deque>
#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

namespace llmp::platform {
namespace {

namespace fs = std::filesystem;

// Links a walk may follow, as the kernel's own limit (MAXSYMLINKS).
constexpr int kLinkLimit = 40;

bool TrustedOwner(uid_t owner, uid_t trusted) { return owner == 0 || owner == trusted; }

// Whether nobody but root and the trusted user belongs to a group: the
// trusted user's own private group (the Debian and Ubuntu default), with
// no other members listed and no other account using it as its primary
// group. Group write through such a group is the owner's own.
bool PrivateGroup(gid_t group, uid_t trusted) {
  std::array<char, 16384> buffer{};
  passwd user{};
  passwd* found_user = nullptr;
  if (::getpwuid_r(trusted, &user, buffer.data(), buffer.size(), &found_user) != 0 ||
      found_user == nullptr || user.pw_gid != group) {
    return false;
  }
  const std::string name = user.pw_name;
  std::array<char, 65536> group_buffer{};
  struct group entry{};
  struct group* found_group = nullptr;
  if (::getgrgid_r(group, &entry, group_buffer.data(), group_buffer.size(), &found_group) != 0 ||
      found_group == nullptr) {
    return false;
  }
  for (char** member = entry.gr_mem; member != nullptr && *member != nullptr; ++member) {  // NOLINT
    if (name != *member && std::string_view(*member) != "root") {
      return false;
    }
  }
  bool shared = false;
  ::setpwent();  // NOLINT(concurrency-mt-unsafe): startup checks, one thread
  while (const passwd* other = ::getpwent()) {  // NOLINT(concurrency-mt-unsafe)
    if (other->pw_gid == group && other->pw_uid != trusted && other->pw_uid != 0) {
      shared = true;
      break;
    }
  }
  ::endpwent();  // NOLINT(concurrency-mt-unsafe)
  return !shared;
}

// Whether path (not followed if a link) carries a POSIX access ACL. An
// error other than "none" counts as one, which fails closed.
bool AclAnswer(ssize_t size) { return size >= 0 || (errno != ENODATA && errno != ENOTSUP); }

bool HasAccessAcl(const fs::path& path) {
  return AclAnswer(::lgetxattr(path.c_str(), "system.posix_acl_access", nullptr, 0));
}

// Whether path is on FUSE, whose mounting user reports whatever ownership
// it likes.
bool OnFuse(const fs::path& path) {
  struct statfs fs{};
  constexpr auto kFuseMagic = 0x65735546;
  return ::statfs(path.c_str(), &fs) == 0 && fs.f_type == kFuseMagic;
}

std::string OwnerText(uid_t trusted) {
  return trusted == 0 ? std::string("root") : std::format("root and uid {}", trusted);
}

std::deque<std::string> Components(const fs::path& path) {
  std::deque<std::string> parts;
  for (const fs::path& part : path.relative_path()) {
    parts.push_back(part.string());
  }
  return parts;
}

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

}  // namespace

namespace {

// With an access ACL, the group bits are its mask: a named user or group
// may be what can write. Only group write through the private group, and
// no ACL, is the owner's own.
template <typename HasAcl>
bool OthersWrite(const struct stat& status, uid_t trusted, HasAcl has_acl) {
  if ((status.st_mode & S_IWOTH) != 0) {
    return true;
  }
  if ((status.st_mode & S_IWGRP) == 0) {
    return false;
  }
  return !PrivateGroup(status.st_gid, trusted) || has_acl();
}

}  // namespace

bool HasDefaultAcl(const fs::path& path) {
  return AclAnswer(::lgetxattr(path.c_str(), "system.posix_acl_default", nullptr, 0));
}

bool OthersCanWrite(const struct stat& status, uid_t trusted, const fs::path& path) {
  return OthersWrite(status, trusted, [&path] { return HasAccessAcl(path); });
}

bool OthersCanWrite(const struct stat& status, uid_t trusted, int fd) {
  return OthersWrite(status, trusted, [fd] {
    return AclAnswer(::fgetxattr(fd, "system.posix_acl_access", nullptr, 0));
  });
}

std::string OctalMode(mode_t mode) { return std::format("{:04o}", mode & 07777U); }

std::expected<void, std::string> CheckPrivateDirectory(const fs::path& path,
                                                       const struct stat& status, uid_t trusted) {
  if (!S_ISDIR(status.st_mode)) {
    return std::unexpected(std::format("{} is not a directory", path.string()));
  }
  if (!TrustedOwner(status.st_uid, trusted)) {
    return std::unexpected(std::format("{} is owned by uid {}, not {}", path.string(),
                                       status.st_uid, OwnerText(trusted)));
  }
  if (OthersCanWrite(status, trusted, path)) {
    return std::unexpected(std::format("users other than {} can add files to {} (mode {})",
                                       OwnerText(trusted), path.string(),
                                       OctalMode(status.st_mode)));
  }
  return {};
}

std::expected<TrustedPath, std::string> WalkTrusted(const fs::path& path, uid_t trusted,
                                                    bool final_link_ok) {
  if (!path.is_absolute()) {
    return std::unexpected(std::format("{} is not an absolute path", path.string()));
  }
  const auto untrusted = [&](const fs::path& at, const std::string& why) {
    return std::unexpected(at == path ? why : at.string() + " " + why);
  };
  TrustedPath result;
  fs::path current = "/";
  struct stat status{};
  if (::lstat("/", &status) != 0) {
    return untrusted("/", "cannot be examined: " + Errno(errno));
  }
  // "/" is judged like any directory. (Under qemu-user it is the target
  // sysroot, which the developer owns.)
  if (!TrustedOwner(status.st_uid, trusted) ||
      (OthersCanWrite(status, trusted, "/") && (status.st_mode & S_ISVTX) == 0)) {
    return untrusted(
        "/", std::format("is not owned by {} and private (uid {}, mode {})", OwnerText(trusted),
                         status.st_uid, OctalMode(status.st_mode)));
  }
  std::deque<std::string> pending = Components(path);
  int links = 0;
  while (!pending.empty()) {
    const std::string part = pending.front();
    pending.pop_front();
    if (part.empty() || part == ".") {
      continue;
    }
    if (part == "..") {
      // Every directory above `current` was walked on the way down.
      current = current.parent_path();
      continue;
    }
    const fs::path next = current / part;
    if (::lstat(next.c_str(), &status) != 0) {
      const int error = errno;
      if (error != ENOENT) {
        return untrusted(next, "cannot be examined: " + Errno(error));
      }
      fs::path rest = next;
      for (const std::string& more : pending) {
        if (more == "..") {
          // The kernel would stop here; a lexical ".." would land elsewhere.
          return untrusted(next, "does not exist, and the path goes on to '..' from inside it");
        }
        rest /= more;
      }
      struct stat parent{};
      if (::stat(current.c_str(), &parent) != 0) {
        return untrusted(current, "cannot be examined: " + Errno(errno));
      }
      result.resolved = rest.lexically_normal();
      result.exists = false;
      result.existing = current;
      result.status = parent;
      return result;
    }
    // The directory it came from may be sticky and writable by others, who
    // could then have created this entry.
    if (!TrustedOwner(status.st_uid, trusted)) {
      return untrusted(
          next, std::format("is owned by uid {}, not {}", status.st_uid, OwnerText(trusted)));
    }
    if (!S_ISLNK(status.st_mode) && OnFuse(next)) {
      return untrusted(next,
                       "is on a FUSE filesystem, which reports whatever owner its mounter likes");
    }
    const bool last = pending.empty();
    if (S_ISLNK(status.st_mode)) {
      if (last && !final_link_ok) {
        return untrusted(next, "is a symbolic link");
      }
      if (++links > kLinkLimit) {
        return untrusted(next, "is reached through too many symbolic links");
      }
      std::array<char, 4096> target{};
      const ssize_t size = ::readlink(next.c_str(), target.data(), target.size());
      if (size < 0 || static_cast<std::size_t>(size) >= target.size()) {
        return untrusted(next, "is a symbolic link that cannot be read");
      }
      const fs::path to(std::string(target.data(), static_cast<std::size_t>(size)));
      std::deque<std::string> parts = Components(to);
      pending.insert(pending.begin(), parts.begin(), parts.end());
      if (to.is_absolute()) {
        current = "/";
      }
      continue;
    }
    if (last) {
      result.resolved = next;
      result.exists = true;
      result.existing = next;
      result.status = status;
      return result;
    }
    if (!S_ISDIR(status.st_mode)) {
      return untrusted(next, "is not a directory");
    }
    if (OthersCanWrite(status, trusted, next) && (status.st_mode & S_ISVTX) == 0) {
      return untrusted(next, std::format("can be changed by users other than {} (mode {})",
                                         OwnerText(trusted), OctalMode(status.st_mode)));
    }
    current = next;
  }
  // The path was "/" (or resolved back to it).
  if (::lstat(current.c_str(), &status) != 0) {
    return untrusted(current, "cannot be examined: " + Errno(errno));
  }
  result.resolved = current;
  result.exists = true;
  result.existing = current;
  result.status = status;
  return result;
}

}  // namespace llmp::platform
