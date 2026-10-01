// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_BENCHMARKS_QWEN38_REFERENCE_H_
#define JITLLM_BENCHMARKS_QWEN38_REFERENCE_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace jitllm::benchmarks::draft_vocab {

// Private diagnostic file, never an engine restore path. The caller owns
// two bounded, cataloged pinned views and supplies a completed D2H copy.
struct PageRange {
  std::size_t region = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  bool operator==(const PageRange&) const = default;
};

struct ReferenceFailure {
  std::string detail;
  // A failed copy does not prove retirement. Keep its live staging allocation.
  bool unknown_copy = false;
};

struct ReferenceStats {
  std::uint64_t captures = 0;
  std::uint64_t comparisons = 0;
  std::uint64_t capture_bytes = 0;
  std::uint64_t comparison_bytes = 0;
  std::uint64_t write_bytes = 0;
  std::uint64_t read_bytes = 0;
  double capture_seconds = 0;
  double comparison_seconds = 0;
  double capture_copy_seconds = 0;
  double comparison_copy_seconds = 0;
  double sha_seconds = 0;
  double write_seconds = 0;
  double read_seconds = 0;
  double byte_compare_seconds = 0;
};

class ReferencePages {
 public:
  using Copy = std::function<std::expected<void, std::string>(void*, const PageRange&)>;
  static constexpr std::uint64_t kPage = 2U << 20;
  static constexpr std::uint64_t kMaxBytes = 2ULL << 30;
  static constexpr std::size_t kMaxRanges = 1024;

  ReferencePages() = default;
  ReferencePages(const ReferencePages&) = delete;
  ReferencePages& operator=(const ReferencePages&) = delete;
  ReferencePages(ReferencePages&& other) noexcept;
  ReferencePages& operator=(ReferencePages&& other) noexcept;
  ~ReferencePages();

  // All ranges are nonempty physical pages in strict region/offset order.
  // Success authenticates geometry and every byte once with the canonical SHA.
  static std::expected<ReferencePages, ReferenceFailure> Capture(
      const std::filesystem::path& directory, std::span<const PageRange> ranges,
      std::uint32_t cursor, std::span<std::byte> live, const Copy& copy, ReferenceStats& stats);
  // Every current range and cursor must match before any D2H copy. Each saved
  // page is read completely before comparing the completed live copy. A byte
  // mismatch is a terminal diagnostic failure, with no borrowed work pending.
  std::expected<void, ReferenceFailure> Compare(std::span<const PageRange> ranges,
                                                std::uint32_t cursor, std::span<std::byte> expected,
                                                std::span<std::byte> live, const Copy& copy,
                                                ReferenceStats& stats) const;
  const std::string& sha256() const { return sha256_; }
  std::uint64_t bytes() const { return bytes_; }

 private:
  // Allows a CPU control to truncate the real unnamed file and prove EOF
  // refusal; no alternate production reader or file format is introduced.
  friend struct ReferencePagesTestAccess;
  int fd_ = -1;
  std::uint64_t bytes_ = 0;
  std::uint64_t file_bytes_ = 0;
  std::uint32_t cursor_ = 0;
  std::vector<PageRange> ranges_;
  std::string sha256_;
};

}  // namespace jitllm::benchmarks::draft_vocab

#endif  // JITLLM_BENCHMARKS_QWEN38_REFERENCE_H_
