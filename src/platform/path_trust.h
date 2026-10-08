// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Whether a path is safe from other users (D-063, D-054): that nobody but
// root and one trusted user could change what it names, by writing to it,
// or by renaming, unlinking or re-pointing it or any directory or link on
// the way to it. The runtime asks this of its configuration files and its
// storage roles, trusting root and its own user.
//
// The walk resolves each component itself, following links the way the
// kernel would, and judges every directory it passes through and every
// entry it takes from one:
//
// - a directory is trusted if root or the trusted user owns it and nobody
//   else can write it; a directory others can write is still passable if
//   it is sticky (like /tmp), since they cannot rename or unlink what they
//   do not own there;
// - every entry taken from a directory, a link included, must be owned by
//   root or the trusted user, so that a sticky directory's other users
//   cannot have planted it.
//
// Group write counts as others' write unless the group is the trusted
// user's private group (Debian's default: its own primary group, with no
// other member listed or known) and there is no ACL, whose mask the group
// bits then show. A path on FUSE is untrusted, since its mounter reports
// the owners. The check is a snapshot: callers open what it approved
// without following links and compare device and inode.

#ifndef LLMP_PLATFORM_PATH_TRUST_H_
#define LLMP_PLATFORM_PATH_TRUST_H_

#include <sys/stat.h>
#include <sys/types.h>

#include <expected>
#include <filesystem>
#include <string>

namespace llmp::platform {

// Where a trusted walk ended.
struct TrustedPath {
  // The path with every existing link resolved; a missing tail is kept as
  // written.
  std::filesystem::path resolved;
  // Whether the whole path exists; if not, `existing` is its longest
  // existing prefix (resolved), and `status` describes that.
  bool exists = false;
  std::filesystem::path existing;
  struct stat status{};
};

// Walks an absolute path from "/". Fails, naming the component (unless it
// is the path itself) and why,
// if any directory or entry on the way is untrusted, if a component other
// than the last is not a directory, or if links loop. The last component
// is judged as an entry only: callers check its own type and mode. If
// `final_link_ok` is false, a link as the last component fails.
std::expected<TrustedPath, std::string> WalkTrusted(const std::filesystem::path& path,
                                                    uid_t trusted, bool final_link_ok);

// Whether a directory's own mode keeps others out: owned by root or the
// trusted user, and writable by nobody else (sticky or not). A directory
// others could add entries to fails.
std::expected<void, std::string> CheckPrivateDirectory(const std::filesystem::path& path,
                                                       const struct stat& status, uid_t trusted);

// Whether users other than root and the trusted user can write path,
// which status describes (see above); path is read for its ACL.
bool OthersCanWrite(const struct stat& status, uid_t trusted, const std::filesystem::path& path);
// The same for an open file.
bool OthersCanWrite(const struct stat& status, uid_t trusted, int fd);

// Whether a directory carries a default ACL, which what is later created
// in it inherits (not followed if a link; an error counts as one).
bool HasDefaultAcl(const std::filesystem::path& path);

// The mode bits as four octal digits, "0755".
std::string OctalMode(mode_t mode);

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_PATH_TRUST_H_
