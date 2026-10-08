// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Experiment-only physical-memory pressure. Release on stdin EOF or termination.
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

int main(int argc, char** argv) {
  constexpr std::uint64_t gib = 1ULL << 30;
  std::uint64_t amount = 0;
  if (argc != 2) return 2;
  const auto end = argv[1] + std::strlen(argv[1]);
  auto parsed = std::from_chars(argv[1], end, amount);
  if (parsed.ec != std::errc{} || parsed.ptr != end || amount == 0 || amount > 80) return 2;
  std::ifstream memory("/proc/meminfo");
  std::string key, line;
  std::uint64_t available_kib = 0;
  while (memory >> key) {
    if (key == "MemAvailable:") { memory >> available_kib; break; }
    std::getline(memory, line);
  }
  const auto bytes = amount * gib;
  if (available_kib * 1024 < bytes + 24 * gib) {
    std::fputs("Insufficient headroom before pressure allocation\n", stderr);
    return 1;
  }
  const rlimit limit{bytes, bytes};
  if (setrlimit(RLIMIT_MEMLOCK, &limit) != 0) { std::perror("setrlimit"); return 1; }
  void* data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (data == MAP_FAILED) { std::perror("mmap"); return 1; }
  if (madvise(data, bytes, MADV_NOHUGEPAGE) != 0 || mlock(data, bytes) != 0) {
    std::perror("madvise/mlock"); munmap(data, bytes); return 1;
  }
  // Touch every page with nonzero data; no demand-zero or reclaimable placeholder.
  std::memset(data, 0x5a, bytes);
  std::printf("{\"pid\":%d,\"locked_bytes\":%llu}\n", getpid(),
              static_cast<unsigned long long>(bytes));
  std::fflush(stdout);
  char stop = 0;
  while (read(STDIN_FILENO, &stop, 1) < 0 && errno == EINTR) {}
  const auto unlocked = munlock(data, bytes);
  const auto unmapped = munmap(data, bytes);
  return unlocked == 0 && unmapped == 0 ? 0 : 1;
}
