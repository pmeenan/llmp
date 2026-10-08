// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The platform module: small file reads and the host probe, on fake /proc,
// /sys and /dev trees; the direct-I/O opens, the event loop, signal watch,
// random bytes and available memory.

#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>

#include "base/report.h"
#include "platform/direct_io.h"
#include "platform/event_loop.h"
#include "platform/files.h"
#include "platform/host_probe.h"
#include "platform/interfaces.h"
#include "platform/memory_pressure.h"
#include "platform/path_trust.h"
#include "platform/sockets.h"

namespace {

namespace fs = std::filesystem;
using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

// A directory for one test's fake tree, removed afterwards.
class FakeRoot {
 public:
  FakeRoot() {
    std::string pattern = (fs::path(::testing::TempDir()) / "llmp-platform-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
    } else {
      ADD_FAILURE() << "cannot create a directory from " << pattern;
    }
  }
  FakeRoot(const FakeRoot&) = delete;
  FakeRoot& operator=(const FakeRoot&) = delete;
  FakeRoot(FakeRoot&&) = delete;
  FakeRoot& operator=(FakeRoot&&) = delete;
  ~FakeRoot() {
    if (!path_.empty()) {
      std::error_code error;
      fs::remove_all(path_, error);
    }
  }

  const fs::path& path() const { return path_; }

  // Writes text to a file below the root, creating its directories.
  // Nothing is created outside a root that exists.
  void Write(std::string_view name, std::string_view text) const {
    ASSERT_FALSE(path_.empty());
    const fs::path file = path_ / name;
    std::error_code error;
    fs::create_directories(file.parent_path(), error);
    ASSERT_FALSE(error) << error.message();
    std::ofstream out(file, std::ios::binary);
    out << text;
    ASSERT_TRUE(out.good()) << file;
  }

  void Directory(std::string_view name) const {
    ASSERT_FALSE(path_.empty());
    std::error_code error;
    fs::create_directories(path_ / name, error);
    ASSERT_FALSE(error) << error.message();
  }

 private:
  fs::path path_;
};

// The value of a line in a report section, or nullopt.
std::optional<std::string> Value(const llmp::base::Report& report, std::string_view title,
                                 std::string_view key) {
  for (const auto& section : report.sections) {
    if (section.title != title) {
      continue;
    }
    for (const auto& line : section.lines) {
      if (line.key == key) {
        return line.value;
      }
    }
  }
  return std::nullopt;
}

TEST(Files, ReadSmallFile) {
  const FakeRoot root;
  ASSERT_FALSE(root.path().empty());
  root.Write("a", "hello\nworld\n");
  EXPECT_EQ(llmp::platform::ReadSmallFile(root.path() / "a"), "hello\nworld\n");
  root.Write("empty", "");
  EXPECT_EQ(llmp::platform::ReadSmallFile(root.path() / "empty"), "");
}

TEST(Files, ReadSmallFileLimit) {
  const FakeRoot root;
  root.Write("four", "1234");
  EXPECT_EQ(llmp::platform::ReadSmallFile(root.path() / "four", 4), "1234");
  const auto longer = llmp::platform::ReadSmallFile(root.path() / "four", 3);
  ASSERT_FALSE(longer);
  EXPECT_EQ(longer.error(), std::errc::file_too_large);
}

TEST(Files, ReadSmallFileErrors) {
  const FakeRoot root;
  const auto missing = llmp::platform::ReadSmallFile(root.path() / "missing");
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error(), std::errc::no_such_file_or_directory);
  const auto directory = llmp::platform::ReadSmallFile(root.path());
  ASSERT_FALSE(directory);
  EXPECT_EQ(directory.error(), std::errc::is_a_directory);
}

TEST(Files, ReadFirstLine) {
  const FakeRoot root;
  root.Write("a", "580.178.04  \r\nnext\n");
  EXPECT_EQ(llmp::platform::ReadFirstLine(root.path() / "a"), "580.178.04");
  root.Write("b", "no newline");
  EXPECT_EQ(llmp::platform::ReadFirstLine(root.path() / "b"), "no newline");
  root.Write("c", "\n");
  EXPECT_EQ(llmp::platform::ReadFirstLine(root.path() / "c"), "");
  EXPECT_FALSE(llmp::platform::ReadFirstLine(root.path() / "missing"));
}

TEST(Files, ListDirectory) {
  const FakeRoot root;
  root.Write("d/b", "");
  root.Write("d/a", "");
  root.Directory("d/C");
  EXPECT_THAT(llmp::platform::ListDirectory(root.path() / "d").value(), ElementsAre("C", "a", "b"));
  root.Directory("empty");
  EXPECT_THAT(llmp::platform::ListDirectory(root.path() / "empty").value(), IsEmpty());
  const auto missing = llmp::platform::ListDirectory(root.path() / "missing");
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error(), std::errc::no_such_file_or_directory);
}

TEST(MemoryPressure, ParsesTheStallLines) {
  using llmp::platform::ParsePressure;
  const auto stall = ParsePressure(
      "some avg10=1.50 avg60=0.10 avg300=0.00 total=1234\n"
      "full avg10=0.25 avg60=0.00 avg300=0.00 total=56\n");
  ASSERT_TRUE(stall.has_value());
  const auto parsed = stall.value_or(llmp::platform::PressureStall{});
  EXPECT_DOUBLE_EQ(parsed.some_avg10, 1.5);
  EXPECT_DOUBLE_EQ(parsed.full_avg10, 0.25);
  EXPECT_FALSE(ParsePressure("some avg10=1.50 avg60=0.10 avg300=0.00 total=1\n").has_value());
  EXPECT_FALSE(ParsePressure("some avg10=x\nfull avg10=0.00\n").has_value());
  EXPECT_FALSE(ParsePressure("some avg10=101.0\nfull avg10=0.00\n").has_value());
  EXPECT_FALSE(ParsePressure("somewhere avg10=1.0\nfull avg10=0.00\n").has_value());
  EXPECT_FALSE(ParsePressure("").has_value());
  // This host's, when it has PSI: both lines, each a share in [0, 100].
  const auto now = llmp::platform::ReadMemoryPressure();
  if (now.stall) {
    EXPECT_LE(now.stall->full_avg10, 100.0);
    EXPECT_GE(now.stall->some_avg10, 0.0);
  }
}

TEST(Meminfo, Values) {
  constexpr std::string_view kMeminfo =
      "MemTotal:       127598832 kB\n"
      "MemFree:        1 kB\n"
      "MemAvailable:   123791920 kB\n"
      "HugePages_Total:       0\n";
  using llmp::platform::MeminfoBytes;
  EXPECT_EQ(MeminfoBytes(kMeminfo, "MemTotal"), std::uint64_t{127598832} * 1024);
  EXPECT_EQ(MeminfoBytes(kMeminfo, "MemAvailable"), std::uint64_t{123791920} * 1024);
  EXPECT_EQ(MeminfoBytes(kMeminfo, "HugePages_Total"), 0U);
  EXPECT_EQ(MeminfoBytes(kMeminfo, "Mem"), std::nullopt);
  EXPECT_EQ(MeminfoBytes(kMeminfo, "SwapTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("MemTotal: 12 MB\n", "MemTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("MemTotal: x kB\n", "MemTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("MemTotal: 18014398509481984 kB\n", "MemTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("MemTotal: 99999999999999999999 kB\n", "MemTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("", "MemTotal"), std::nullopt);
}

TEST(KernelModule, Found) {
  const FakeRoot root;
  root.Write("sys/module/nvidia/version", "580.178.04\n");
  root.Directory("sys/module/nvidia_uvm");
  const auto nvidia = llmp::platform::FindKernelModule(root.path(), "nvidia");
  EXPECT_TRUE(nvidia.loaded);
  EXPECT_EQ(nvidia.version, "580.178.04");
  const auto uvm = llmp::platform::FindKernelModule(root.path(), "nvidia_uvm");
  EXPECT_TRUE(uvm.loaded);
  EXPECT_EQ(uvm.version, "");
  const auto fs_module = llmp::platform::FindKernelModule(root.path(), "nvidia_fs");
  EXPECT_FALSE(fs_module.loaded);
}

// A Spark-like tree: two RDMA devices, one port each.
void WriteSparkLike(const FakeRoot& root) {
  root.Write("proc/meminfo", "MemTotal: 127598832 kB\nMemAvailable: 123791920 kB\n");
  root.Write("proc/sys/fs/protected_hardlinks", "1\n");
  for (const auto& [device, netdev, verbs, state] :
       {std::tuple{"rocep1s0f0", "enp1s0f0np0", "uverbs0", "1: DOWN"},
        std::tuple{"rocep1s0f1", "enp1s0f1np1", "uverbs1", "4: ACTIVE"}}) {
    const std::string dir = std::string("sys/class/infiniband/") + device;
    root.Write(dir + "/ports/1/state", std::string(state) + "\n");
    root.Write(dir + "/ports/1/link_layer", "Ethernet\n");
    root.Write(dir + "/ports/1/rate", "200 Gb/sec (2X NDR)\n");
    root.Directory(dir + "/device/net/" + netdev);
    root.Directory(dir + "/device/infiniband_verbs/" + verbs);
    root.Write(std::string("dev/infiniband/") + verbs, "");
  }
  root.Write("dev/infiniband/rdma_cm", "");
}

TEST(DescribeHost, SparkLike) {
  const FakeRoot root;
  WriteSparkLike(root);
  llmp::base::Report report;
  llmp::platform::DescribeHost(root.path(), report);
  EXPECT_EQ(Value(report, "host", "memory"), "121.7 GiB total, 118.1 GiB available");
  EXPECT_EQ(Value(report, "host", "fs.protected_hardlinks"), "1");
  EXPECT_TRUE(Value(report, "host", "glibc"));
  EXPECT_TRUE(Value(report, "host", "kernel"));
  EXPECT_EQ(Value(report, "host", "page size"),
            llmp::base::FormatBytes(static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE))));
  EXPECT_EQ(Value(report, "RDMA", "rocep1s0f0 port 1"),
            "DOWN, Ethernet, 200 Gb/sec (2X NDR), enp1s0f0np0, /dev/infiniband/uverbs0 read-write");
  EXPECT_EQ(
      Value(report, "RDMA", "rocep1s0f1 port 1"),
      "ACTIVE, Ethernet, 200 Gb/sec (2X NDR), enp1s0f1np1, /dev/infiniband/uverbs1 read-write");
  EXPECT_EQ(Value(report, "RDMA", "/dev/infiniband/rdma_cm"), "read-write");
  EXPECT_EQ(Value(report, "RDMA", "device access checked for"),
            "uid " + std::to_string(::geteuid()));
  EXPECT_THAT(report.problems, IsEmpty());
  EXPECT_THAT(report.warnings, IsEmpty());
}

TEST(DescribeHost, NothingThere) {
  const FakeRoot root;
  llmp::base::Report report;
  llmp::platform::DescribeHost(root.path(), report);
  EXPECT_EQ(Value(report, "host", "memory"), "unknown");
  EXPECT_EQ(Value(report, "host", "fs.protected_hardlinks"), "unknown");
  EXPECT_EQ(Value(report, "RDMA", "devices"), "none");
  EXPECT_THAT(report.problems, IsEmpty());
  ASSERT_EQ(report.warnings.size(), 2U);
  EXPECT_THAT(report.warnings[0], HasSubstr("cannot read /proc/meminfo"));
  EXPECT_THAT(report.warnings[1], HasSubstr("cannot read fs.protected_hardlinks"));
}

TEST(DescribeHost, Warnings) {
  const FakeRoot root;
  root.Write("proc/meminfo", "MemFree: 1 kB\n");
  root.Write("proc/sys/fs/protected_hardlinks", "0\n");
  root.Write("sys/class/infiniband/mlx5_0/ports/1/state", "4: ACTIVE\n");
  root.Directory("sys/class/infiniband/mlx5_0/device/infiniband_verbs/uverbs0");
  root.Directory("sys/class/infiniband/mlx5_1");  // no ports
  llmp::base::Report report;
  llmp::platform::DescribeHost(root.path(), report);
  EXPECT_EQ(Value(report, "RDMA", "mlx5_0 port 1"),
            "ACTIVE, unknown link layer, unknown rate, /dev/infiniband/uverbs0 No such file or "
            "directory");
  EXPECT_EQ(Value(report, "RDMA", "mlx5_1"), "no ports");
  EXPECT_EQ(Value(report, "RDMA", "/dev/infiniband/rdma_cm"), "No such file or directory");
  EXPECT_THAT(report.problems, IsEmpty());
  ASSERT_EQ(report.warnings.size(), 5U);
  EXPECT_EQ(report.warnings[0], "/proc/meminfo has no MemTotal or MemAvailable");
  EXPECT_THAT(report.warnings[1], HasSubstr("fs.protected_hardlinks is 0, not 1"));
  EXPECT_THAT(report.warnings[2], HasSubstr("RDMA device mlx5_0: this user cannot open "
                                            "/dev/infiniband/uverbs0"));
  EXPECT_THAT(report.warnings[3], HasSubstr("RDMA device mlx5_1 has no verbs device"));
  EXPECT_THAT(report.warnings[4], HasSubstr("/dev/infiniband/rdma_cm"));
}

// A device node this user cannot open (not testable as root, who can).
TEST(DescribeHost, InaccessibleVerbs) {
  if (::geteuid() == 0) {
    GTEST_SKIP() << "root can open any file";
  }
  const FakeRoot root;
  WriteSparkLike(root);
  ASSERT_EQ(::chmod((root.path() / "dev/infiniband/uverbs1").c_str(), 0), 0);
  llmp::base::Report report;
  llmp::platform::DescribeHost(root.path(), report);
  EXPECT_EQ(Value(report, "RDMA", "rocep1s0f1 port 1"),
            "ACTIVE, Ethernet, 200 Gb/sec (2X NDR), enp1s0f1np1, /dev/infiniband/uverbs1 "
            "Permission denied");
  ASSERT_EQ(report.warnings.size(), 1U);
  EXPECT_THAT(report.warnings[0], HasSubstr("RDMA device rocep1s0f1"));
}

// Scratch directories in the build tree (LLMP_TEST_SCRATCH), whose
// parents no other user shares, as they may /tmp's.
class Scratch {
 public:
  Scratch() {
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path parent = base != nullptr ? fs::path(base) : fs::path(::testing::TempDir());
    std::error_code error;
    fs::create_directories(parent, error);
    std::string pattern = (parent / "trust-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
    } else {
      ADD_FAILURE() << "cannot create a directory from " << pattern;
    }
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  Scratch(Scratch&&) = delete;
  Scratch& operator=(Scratch&&) = delete;
  ~Scratch() {
    std::error_code error;
    fs::permissions(path_, fs::perms::owner_all, error);
    fs::remove_all(path_, error);
  }
  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

TEST(WalkTrusted, ResolvesLinksAndMissingTails) {
  const Scratch scratch;
  const fs::path& root = scratch.path();
  fs::create_directories(root / "real/sub");
  fs::create_directory_symlink("real", root / "rel");
  fs::create_directory_symlink(root / "real", root / "abs");
  std::ofstream(root / "real/sub/file") << "x";
  const uid_t me = ::geteuid();

  auto walked = llmp::platform::WalkTrusted(root / "rel/sub/file", me, false);
  ASSERT_TRUE(walked.has_value()) << walked.error();
  EXPECT_TRUE(walked->exists);
  EXPECT_EQ(walked->resolved, fs::canonical(root / "real/sub/file"));
  EXPECT_TRUE(S_ISREG(walked->status.st_mode));

  walked = llmp::platform::WalkTrusted(root / "abs/sub/../missing/deeper", me, false);
  ASSERT_TRUE(walked.has_value()) << walked.error();
  EXPECT_FALSE(walked->exists);
  EXPECT_EQ(walked->resolved, fs::canonical(root / "real") / "missing/deeper");
  EXPECT_EQ(walked->existing, fs::canonical(root / "real"));

  // A link as the last component is refused unless allowed.
  fs::create_symlink("real/sub/file", root / "link");
  walked = llmp::platform::WalkTrusted(root / "link", me, false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("is a symbolic link"));
  walked = llmp::platform::WalkTrusted(root / "link", me, true);
  ASSERT_TRUE(walked.has_value()) << walked.error();
  EXPECT_EQ(walked->resolved, fs::canonical(root / "real/sub/file"));
}

TEST(WalkTrusted, RefusesLoopsAndNonDirectories) {
  const Scratch scratch;
  const fs::path& root = scratch.path();
  fs::create_symlink("b", root / "a");
  fs::create_symlink("a", root / "b");
  auto walked = llmp::platform::WalkTrusted(root / "a/x", ::geteuid(), false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("too many symbolic links"));
  std::ofstream(root / "file") << "x";
  walked = llmp::platform::WalkTrusted(root / "file/x", ::geteuid(), false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("file is not a directory"));
  // ".." after a missing component: the kernel stops at the missing one.
  fs::create_symlink("missing/../../x", root / "dotdot");
  walked = llmp::platform::WalkTrusted(root / "dotdot/y", ::geteuid(), false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("does not exist, and the path goes on to '..'"));
  walked = llmp::platform::WalkTrusted("relative/path", ::geteuid(), false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("not an absolute path"));
}

TEST(WalkTrusted, RefusesWhatOthersCanChange) {
  const Scratch scratch;
  const fs::path& root = scratch.path();
  fs::create_directories(root / "open/inner");
  const uid_t me = ::geteuid();
  ASSERT_EQ(::chmod((root / "open").c_str(), 0757), 0);
  auto walked = llmp::platform::WalkTrusted(root / "open/inner", me, false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("open can be changed by users other than root and uid"));
  // A sticky directory is passable: others cannot replace what it holds.
  ASSERT_EQ(::chmod((root / "open").c_str(), 01777), 0);
  walked = llmp::platform::WalkTrusted(root / "open/inner", me, false);
  EXPECT_TRUE(walked.has_value()) << walked.error();
  // But not as a directory to add files to.
  struct stat status{};
  ASSERT_EQ(::stat((root / "open").c_str(), &status), 0);
  const auto private_dir = llmp::platform::CheckPrivateDirectory(root / "open", status, me);
  ASSERT_FALSE(private_dir.has_value());
  EXPECT_THAT(private_dir.error(), HasSubstr("can add files to"));
  // Entries another user owns are untrusted: here, trusting root alone.
  if (me != 0) {
    walked = llmp::platform::WalkTrusted(root / "open/inner", 0, false);
    ASSERT_FALSE(walked.has_value());
    // Under qemu-user, "/" is the sysroot, which this user owns: the walk
    // stops there instead.
    EXPECT_THAT(walked.error(),
                AnyOf(HasSubstr(std::format("is owned by uid {}, not root", me)),
                      HasSubstr(std::format("/ is not owned by root and private (uid {}", me))));
  }
}

TEST(DirectIo, DescribesFilesystems) {
  auto proc = llmp::platform::DescribeFilesystem("/proc");
  ASSERT_TRUE(proc.has_value()) << proc.error();
  EXPECT_FALSE(proc->accepted);
  EXPECT_FALSE(llmp::platform::DescribeFilesystem("/nonexistent/llmp").has_value());
  auto refused = llmp::platform::ProbeDirectIo("/proc");
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("not a local block-device filesystem"));
}

TEST(DirectIo, ProbesTheBuildTreesFilesystem) {
  const Scratch scratch;
  auto filesystem = llmp::platform::DescribeFilesystem(scratch.path());
  ASSERT_TRUE(filesystem.has_value()) << filesystem.error();
  if (!filesystem->accepted) {
    GTEST_SKIP() << "the build tree is on " << filesystem->type
                 << ", which the storage roles refuse";
  }
  auto probe = llmp::platform::ProbeDirectIo(scratch.path());
  ASSERT_TRUE(probe.has_value()) << probe.error();
  EXPECT_LE(probe->offset_alignment, llmp::platform::kDirectIoAlignment);
  // The probe's file had no name: nothing is left behind.
  EXPECT_TRUE(fs::is_empty(scratch.path()));
}

// The chat route's view of the node's network (platform/interfaces.h):
// every host has its loopback addresses, and the open-file limit only
// rises.
TEST(InterfacesTest, ReadsTheLoopbackAddress) {
  auto addresses = llmp::platform::ReadInterfaceAddresses();
  ASSERT_TRUE(addresses.has_value()) << addresses.error();
  bool loopback = false;
  for (const llmp::platform::InterfaceAddress& a : *addresses) {
    if (!a.ipv6 && a.Text() == "127.0.0.1") {
      loopback = a.loopback;
    }
  }
  EXPECT_TRUE(loopback);
  const std::uint64_t now = llmp::platform::RaiseOpenFileLimit(0);
  EXPECT_GT(now, 0U);
  EXPECT_GE(llmp::platform::RaiseOpenFileLimit(now + 1), now);
}

// A reverse lookup never holds its caller past its limit, whatever the
// resolver does (here it may answer at once, from /etc/hosts, or not).
TEST(InterfacesTest, AReverseLookupKeepsToItsLimit) {
  llmp::platform::InterfaceAddress address;
  address.bytes = {192, 0, 2, 1};  // TEST-NET-1: no name anywhere, perhaps a slow no
  for (const auto limit : {std::chrono::milliseconds(0), std::chrono::milliseconds(50)}) {
    const auto start = std::chrono::steady_clock::now();
    (void)llmp::platform::ReverseName(address, limit);
    EXPECT_LT(std::chrono::steady_clock::now() - start, limit + std::chrono::seconds(1));
  }
}

// The event loop and its wakers (platform/event_loop.h): a waker is
// reported readable with its tag until drained; a connection's peer
// closing is reported; a removed descriptor is no longer.
TEST(EventLoopTest, ReportsWakersAndPeers) {
  using llmp::platform::ReadyEvent;
  auto loop = llmp::platform::EventLoop::Open();
  auto waker = llmp::platform::Waker::Open();
  ASSERT_TRUE(loop.valid());
  ASSERT_TRUE(waker.valid());
  std::array<ReadyEvent, 8> events{};
  const auto wait = [&] { return loop.Wait(events, std::chrono::milliseconds(0)); };
  ASSERT_TRUE(loop.Add(waker.descriptor(), 7, llmp::platform::kReadable));
  EXPECT_EQ(wait().value_or(9), 0U);
  waker.Signal();
  waker.Signal();
  ASSERT_EQ(wait().value_or(9), 1U);
  EXPECT_EQ(events[0].tag, 7U);
  EXPECT_EQ(events[0].ready, llmp::platform::kReadable);
  waker.Drain();
  EXPECT_EQ(wait().value_or(9), 0U);

  std::array<int, 2> pair{-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair.data()), 0);
  const llmp::platform::OwnedDescriptor ours(pair[0]);
  llmp::platform::OwnedDescriptor theirs(pair[1]);
  ASSERT_TRUE(loop.Add(
      ours.get(), 8,
      llmp::platform::kReadable | llmp::platform::kWritable | llmp::platform::kPeerClosed));
  ASSERT_EQ(wait().value_or(9), 1U);
  EXPECT_EQ(events[0].tag, 8U);
  EXPECT_EQ(events[0].ready, llmp::platform::kWritable);
  ASSERT_TRUE(loop.Change(ours.get(), 9, llmp::platform::kPeerClosed));
  EXPECT_EQ(wait().value_or(9), 0U);
  theirs = llmp::platform::OwnedDescriptor();  // the peer closes
  ASSERT_EQ(wait().value_or(9), 1U);
  EXPECT_EQ(events[0].tag, 9U);
  EXPECT_NE(events[0].ready & (llmp::platform::kPeerClosed | llmp::platform::kHangUp), 0U);
  loop.Remove(ours.get());
  EXPECT_EQ(wait().value_or(9), 0U);
  // Nothing to wait for but the limit, which is kept.
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(loop.Wait(events, std::chrono::milliseconds(20)).value_or(9), 0U);
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(19));
}

// A blocked signal is reported through the watch, once, and never
// delivered the usual way.
TEST(EventLoopTest, WatchesBlockedSignals) {
  sigset_t block{};
  (void)::sigemptyset(&block);
  (void)::sigaddset(&block, SIGUSR1);
  sigset_t previous{};
  ASSERT_EQ(::pthread_sigmask(SIG_BLOCK, &block, &previous), 0);
  {
    constexpr std::array kWatched = {SIGUSR1};
    const auto watch = llmp::platform::SignalWatch::Open(kWatched);
    ASSERT_TRUE(watch.valid());
    EXPECT_EQ(watch.Take(), 0);
    ASSERT_EQ(::raise(SIGUSR1), 0);
    pollfd p{.fd = watch.descriptor(), .events = POLLIN, .revents = 0};
    ASSERT_EQ(::poll(&p, 1, 1000), 1);
    EXPECT_EQ(watch.Take(), SIGUSR1);
    EXPECT_EQ(watch.Take(), 0);
  }
  ASSERT_EQ(::pthread_sigmask(SIG_SETMASK, &previous, nullptr), 0);
}

TEST(SocketsTest, RandomBytesDiffer) {
  std::array<std::byte, 32> a{};
  std::array<std::byte, 32> b{};
  ASSERT_TRUE(llmp::platform::FillRandom(a));
  ASSERT_TRUE(llmp::platform::FillRandom(b));
  EXPECT_NE(a, b);
}

TEST(Files, AvailableMemoryAndAnonymousFiles) {
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available.has_value());
  EXPECT_GT(available.value_or(0), 0U);
  const auto memory = llmp::platform::OpenAnonymousMemoryFile("llmp-test");
  ASSERT_TRUE(memory.has_value());
  const llmp::platform::OwnedDescriptor owned(*memory);
  EXPECT_EQ(::ftruncate(owned.get(), 4096), 0);
  // The test's own executable: a size and a time (the calibration key's
  // stamp of an untracked build).
  const auto stamp = llmp::platform::RunningExecutableStamp();
  ASSERT_TRUE(stamp.has_value());
  EXPECT_GT(stamp.value_or(llmp::platform::ExecutableStamp{}).size, 0U);
  EXPECT_GT(stamp.value_or(llmp::platform::ExecutableStamp{}).modified_ns, 0);
}

// The direct-I/O opens: a shard opened for direct reads (and, on a
// filesystem that refuses direct I/O, refused or, when asked, opened
// through the cache), and an unnamed spill file that leaves nothing behind.
TEST(DirectIo, OpensFilesForDirectIo) {
  const Scratch scratch;
  auto filesystem = llmp::platform::DescribeFilesystem(scratch.path());
  ASSERT_TRUE(filesystem.has_value()) << filesystem.error();
  if (!filesystem->accepted) {
    GTEST_SKIP() << "the build tree is on " << filesystem->type
                 << ", which the storage roles refuse";
  }
  {
    std::ofstream(scratch.path() / "shard") << std::string(8192, 'x');
  }
  const llmp::platform::OwnedDescriptor dir(
      ::open(scratch.path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_TRUE(dir.valid());
  auto shard = llmp::platform::OpenForDirectRead(dir.get(), "shard");
  ASSERT_TRUE(shard.has_value()) << shard.error();
  const llmp::platform::OwnedDescriptor shard_fd(shard->fd);
  EXPECT_TRUE(shard->direct);
  EXPECT_EQ(llmp::platform::FileGeneration(shard_fd.get()),
            llmp::platform::FileGeneration(shard_fd.get()));
  EXPECT_EQ(llmp::platform::OpenForDirectRead(dir.get(), "missing").error_or(0), ENOENT);

  auto spill = llmp::platform::OpenUnnamedDirectFile(scratch.path());
  ASSERT_TRUE(spill.has_value()) << spill.error();
  const llmp::platform::OwnedDescriptor spill_fd(*spill);
  EXPECT_EQ(::ftruncate(spill_fd.get(), 2 << 20), 0);
  std::error_code error;
  EXPECT_EQ(std::distance(fs::directory_iterator(scratch.path(), error), fs::directory_iterator()),
            1);  // the shard alone

  // procfs refuses direct I/O.
  const llmp::platform::OwnedDescriptor proc(::open("/proc", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_TRUE(proc.valid());
  EXPECT_EQ(llmp::platform::OpenForDirectRead(proc.get(), "meminfo").error_or(0), EINVAL);
  auto buffered = llmp::platform::OpenForDirectRead(proc.get(), "meminfo", true);
  ASSERT_TRUE(buffered.has_value()) << buffered.error();
  const llmp::platform::OwnedDescriptor buffered_fd(buffered->fd);
  EXPECT_FALSE(buffered->direct);
}

}  // namespace
