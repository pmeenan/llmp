// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/confine.h"

#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <string>

#include "platform/crash_policy.h"

namespace llmp::platform {
namespace {

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// Newer than the sysroot's headers: LANDLOCK_ACCESS_FS_IOCTL_DEV (ABI 5)
// and the scopes (ABI 6), with the ruleset attribute that carries them.
constexpr std::uint64_t kAccessIoctlDev = std::uint64_t{1} << 15;
constexpr std::uint64_t kScopeAbstractUnixSocket = std::uint64_t{1} << 0;
constexpr std::uint64_t kScopeSignal = std::uint64_t{1} << 1;
struct RulesetAttr {
  std::uint64_t handled_access_fs;
  std::uint64_t handled_access_net;
  std::uint64_t scoped;
};
// The Landlock ABI a stage needs: filesystem truncation (3), TCP (4) and
// scopes (6), so it can neither connect out nor signal other processes.
constexpr long kLandlockAbi = 6;
constexpr std::uint64_t kReadAccess =
    LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR | LANDLOCK_ACCESS_FS_EXECUTE;

#ifdef __x86_64__
constexpr std::uint32_t kAuditArch = AUDIT_ARCH_X86_64;
#elifdef __aarch64__
constexpr std::uint32_t kAuditArch = AUDIT_ARCH_AARCH64;
#else
#error "no seccomp architecture for this target"
#endif

class Fd {
 public:
  explicit Fd(int fd) : fd_(fd) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&&) = delete;
  Fd& operator=(Fd&&) = delete;
  ~Fd() {
    if (fd_ >= 0) {
      (void)::close(fd_);  // a ruleset or a path: nothing to flush
    }
  }
  int get() const { return fd_; }

 private:
  int fd_;
};

std::expected<void, std::string> AddTree(int ruleset, const std::filesystem::path& tree,
                                         std::uint64_t access) {
  const Fd fd(::open(tree.c_str(), O_PATH | O_CLOEXEC));
  if (fd.get() < 0) {
    return std::unexpected(std::format("cannot open {}: {}", tree.string(), Errno(errno)));
  }
  landlock_path_beneath_attr rule{};
  rule.allowed_access = access;
  rule.parent_fd = fd.get();
  if (::syscall(SYS_landlock_add_rule, ruleset, LANDLOCK_RULE_PATH_BENEATH, &rule, 0) != 0) {
    return std::unexpected(std::format("cannot allow {}: {}", tree.string(), Errno(errno)));
  }
  return {};
}

// A seccomp filter: the calls that make or reach sockets fail with EACCES
// (socket and socketpair, and io_uring, whose operations create sockets
// without system calls); a system call of another architecture, or an
// x32 one, kills the process; everything else runs.
std::expected<void, std::string> DenySockets() {
  constexpr std::uint32_t kErrno = SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA);
  constexpr std::uint32_t kX32Bit = 0x40000000;
  std::array<sock_filter, 14> program = {{
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kAuditArch, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, kX32Bit, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socket, 5, 0),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socketpair, 4, 0),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_io_uring_setup, 3, 0),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_io_uring_enter, 2, 0),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_io_uring_register, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      BPF_STMT(BPF_RET | BPF_K, kErrno),
      BPF_STMT(BPF_RET | BPF_K, kErrno),
  }};
  const sock_fprog filter{.len = static_cast<unsigned short>(program.size()),
                          .filter = program.data()};
  if (::syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &filter) != 0) {
    return std::unexpected("cannot install the seccomp filter: " + Errno(errno));
  }
  return {};
}

}  // namespace

std::expected<void, std::string> ConfineSelf(const Confinement& confinement) {
  const long abi =
      ::syscall(SYS_landlock_create_ruleset, nullptr, 0, LANDLOCK_CREATE_RULESET_VERSION);
  if (abi < kLandlockAbi) {
    return std::unexpected(
        abi < 0 ? "this kernel has no Landlock: " + Errno(errno)
                : std::format("this kernel's Landlock ABI is {}; confinement needs {}", abi,
                              kLandlockAbi));
  }
  std::uint64_t handled =
      LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_READ_FILE |
      LANDLOCK_ACCESS_FS_READ_DIR | LANDLOCK_ACCESS_FS_REMOVE_DIR | LANDLOCK_ACCESS_FS_REMOVE_FILE |
      LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_DIR | LANDLOCK_ACCESS_FS_MAKE_REG |
      LANDLOCK_ACCESS_FS_MAKE_SOCK | LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_BLOCK |
      LANDLOCK_ACCESS_FS_MAKE_SYM | LANDLOCK_ACCESS_FS_REFER | LANDLOCK_ACCESS_FS_TRUNCATE;
  handled |= kAccessIoctlDev;
  // No rule grants TCP, so a stage can neither bind nor connect; and it
  // can neither signal processes outside its own Landlock domain (the
  // runtime, other jobs) nor reach abstract unix sockets.
  const RulesetAttr attr{
      .handled_access_fs = handled,
      .handled_access_net = LANDLOCK_ACCESS_NET_BIND_TCP | LANDLOCK_ACCESS_NET_CONNECT_TCP,
      .scoped = kScopeAbstractUnixSocket | kScopeSignal};
  const Fd ruleset(static_cast<int>(::syscall(SYS_landlock_create_ruleset, &attr, sizeof attr, 0)));
  if (ruleset.get() < 0) {
    return std::unexpected("cannot create a Landlock ruleset: " + Errno(errno));
  }
  for (const std::filesystem::path& tree : confinement.read) {
    if (auto added = AddTree(ruleset.get(), tree, kReadAccess); !added) {
      return added;
    }
  }
  const std::uint64_t write_access =
      handled & ~(LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_BLOCK | kAccessIoctlDev);
  for (const std::filesystem::path& tree : confinement.write) {
    if (auto added = AddTree(ruleset.get(), tree, write_access); !added) {
      return added;
    }
  }
  // A stage is its own program (exec makes a process dumpable again): its
  // crash leaves no memory image either (D-014).
  if (auto marked = MarkNonDumpable(); !marked) {
    return marked;
  }
  // Both need it, and it keeps a setuid program from undoing either.
  if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
    return std::unexpected("cannot set no_new_privs: " + Errno(errno));
  }
  if (auto sockets = DenySockets(); !sockets) {
    return sockets;
  }
  if (::syscall(SYS_landlock_restrict_self, ruleset.get(), 0) != 0) {
    return std::unexpected("cannot apply the Landlock ruleset: " + Errno(errno));
  }
  return {};
}

}  // namespace llmp::platform
