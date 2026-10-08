// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's n-gram table paged by rows (engine/ple_rows.h): each
// lookup's slot holds its row's bytes once the plan's reads have landed and
// the gather has run; reads are 4 KiB-aligned direct reads within the
// table's stored range, merged where their blocks touch, up to the most a
// read may carry; the whole-chunk count is the chunks the rows touch; rows
// outside the table, a landing too small and too many distinct rows are
// refused; an unknown submission is waited for, and a short or failed read
// refuses the rows once every read has drained. On a GB10: the reads
// through io_uring from a real file and the gather kernel
// (kernels/paging/paging.h) put every row in its slot.

#include "engine/ple_rows.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <vector>

#include "kernels/paging/paging.h"
#include "providers/fake/fake_storage.h"
#include "providers/storage.h"
#include "providers/uring_storage.h"

namespace {

using llmp::engine::kPleBlock;
using llmp::engine::PlanPleRows;
using llmp::engine::PleLandingBound;
using llmp::engine::PleRowPlan;
using llmp::engine::PleTable;
using llmp::engine::ReadPleRows;
using llmp::kernels::paging::GatherPleRows;

constexpr std::uint64_t kChunk = std::uint64_t{2} << 20U;

// A file of `bytes` whose byte i is a function of i.
std::vector<std::byte> Pattern(std::uint64_t bytes) {
  std::vector<std::byte> file(bytes);
  for (std::uint64_t i = 0; i < bytes; ++i) {
    file[i] = static_cast<std::byte>((i * 131U) ^ (i >> 11U));
  }
  return file;
}

// A table of 90-byte rows starting 128 bytes into a group at 8 KiB, the
// group's stored range ending on a block: 70,000 rows cross three 2 MiB
// chunks and many blocks.
PleTable Table() {
  const std::uint64_t group = 8192;
  const std::uint64_t rows = 70000;
  const std::uint64_t end = group + 128 + (rows * 90);
  return {.fd = -1,
          .file_offset = group + 128,
          .rows = rows,
          .row_bytes = 90,
          .chunk_file_offset = group,
          .file_bytes = (end + kPleBlock - 1) / kPleBlock * kPleBlock};
}

// The plan's reads applied to `file` in host memory, as the direct reads
// would land them.
std::vector<std::byte> Land(const PleRowPlan& plan, std::span<const std::byte> file) {
  std::vector<std::byte> landing(plan.landing_bytes);
  for (const auto& read : plan.reads) {
    std::memcpy(landing.data() + read.landing, file.data() + read.file_offset, read.length);
  }
  return landing;
}

void CheckReads(const PleRowPlan& plan, const PleTable& table, std::uint64_t max_read) {
  std::uint64_t landing = 0;
  std::uint64_t last_end = 0;
  for (const auto& read : plan.reads) {
    EXPECT_EQ(read.file_offset % kPleBlock, 0U);
    EXPECT_EQ(read.length % kPleBlock, 0U);
    EXPECT_GT(read.length, 0U);
    EXPECT_LE(read.length, max_read);
    EXPECT_EQ(read.landing, landing);  // packed, each block-aligned
    EXPECT_GE(read.file_offset, table.chunk_file_offset);
    EXPECT_LE(read.file_offset + read.length, table.file_bytes);
    // Ascending; a read split at the limit may repeat the last block of the
    // one before (a row must lie within one read), never more.
    EXPECT_GE(read.file_offset + kPleBlock, last_end);
    EXPECT_GT(read.file_offset + read.length, last_end);
    last_end = read.file_offset + read.length;
    landing += read.length;
  }
  EXPECT_EQ(landing, plan.landing_bytes);
}

TEST(PleRowsTest, EachLookupsSlotHoldsItsRowOnceTheReadsLand) {
  const PleTable table = Table();
  const auto file = Pattern(table.file_bytes);
  std::mt19937 rng(38);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::uniform_int_distribution<std::int32_t> row(0, static_cast<std::int32_t>(table.rows) - 1);
  std::vector<std::int32_t> lookups;
  lookups.reserve(4004);
  for (int i = 0; i < 4000; ++i) {
    lookups.push_back(row(rng));
  }
  // Duplicates, both ends, rows crossing a block and a chunk boundary.
  lookups.push_back(lookups.front());
  lookups.push_back(0);
  lookups.push_back(static_cast<std::int32_t>(table.rows - 1));
  for (std::uint64_t r = 0; r < table.rows; ++r) {
    const std::uint64_t start = table.file_offset + (r * 90);
    if (start / kPleBlock != (start + 89) / kPleBlock) {
      lookups.push_back(static_cast<std::int32_t>(r));
      break;
    }
  }
  for (std::uint64_t r = 0; r < table.rows; ++r) {
    const std::uint64_t start = table.file_offset + (r * 90) - table.chunk_file_offset;
    if (start / kChunk != (start + 89) / kChunk) {
      lookups.push_back(static_cast<std::int32_t>(r));
      break;
    }
  }
  const auto plan = PlanPleRows(table, lookups, PleLandingBound(lookups.size()), lookups.size());
  ASSERT_TRUE(plan.has_value()) << plan.error();
  CheckReads(*plan, table, llmp::engine::kPleMaxRead);
  const std::set<std::int32_t> distinct(lookups.begin(), lookups.end());
  EXPECT_EQ(plan->sources.size(), distinct.size());
  EXPECT_EQ(plan->useful_bytes, distinct.size() * 90);
  ASSERT_EQ(plan->slots.size(), lookups.size());
  const auto landing = Land(*plan, file);
  for (std::size_t i = 0; i < lookups.size(); ++i) {
    const auto slot = static_cast<std::size_t>(plan->slots[i]);
    ASSERT_LT(slot, plan->sources.size());
    const std::uint64_t source = plan->sources[slot];
    ASSERT_LE(source + 90, landing.size());
    const std::uint64_t at = table.file_offset + (static_cast<std::uint64_t>(lookups[i]) * 90);
    EXPECT_EQ(std::memcmp(landing.data() + source, file.data() + at, 90), 0) << "lookup " << i;
  }
  // Whole chunks: those holding any row's first or last byte.
  std::set<std::uint64_t> chunks;
  for (const std::int32_t r : distinct) {
    const std::uint64_t start =
        table.file_offset + (static_cast<std::uint64_t>(r) * 90) - table.chunk_file_offset;
    chunks.insert(start / kChunk);
    chunks.insert((start + 89) / kChunk);
  }
  EXPECT_EQ(plan->extents, chunks.size());
}

TEST(PleRowsTest, RowsWhoseBlocksTouchShareARead) {
  const PleTable table = Table();
  // Rows 0 and 1 share the first block; a row two blocks on is read apart.
  const auto far = static_cast<std::int32_t>((3 * kPleBlock) / 90);
  const std::vector<std::int32_t> lookups = {1, 0, far, 1};
  const auto plan = PlanPleRows(table, lookups, PleLandingBound(4), 4);
  ASSERT_TRUE(plan.has_value()) << plan.error();
  ASSERT_EQ(plan->reads.size(), 2U);
  EXPECT_EQ(plan->reads[0].length, kPleBlock);
  EXPECT_EQ(plan->slots, (std::vector<std::int32_t>{1, 0, 2, 1}));
  // Every row of the first 64 KiB: one read of 64 KiB, then the rest.
  std::vector<std::int32_t> dense;
  dense.reserve(2000);
  for (std::int32_t r = 0; r < 2000; ++r) {
    dense.push_back(r);
  }
  const auto merged = PlanPleRows(table, dense, PleLandingBound(dense.size()), dense.size());
  ASSERT_TRUE(merged.has_value()) << merged.error();
  CheckReads(*merged, table, llmp::engine::kPleMaxRead);
  EXPECT_EQ(merged->reads.front().length, llmp::engine::kPleMaxRead);
  EXPECT_GT(merged->reads.size(), 1U);
  // A smaller limit makes more reads over the same rows.
  const auto small =
      PlanPleRows(table, dense, PleLandingBound(dense.size()), dense.size(), 2 * kPleBlock);
  ASSERT_TRUE(small.has_value()) << small.error();
  CheckReads(*small, table, 2 * kPleBlock);
  EXPECT_GT(small->reads.size(), merged->reads.size());
}

TEST(PleRowsTest, RowsOutsideTheTableATooSmallLandingAndTooManyRowsAreRefused) {
  const PleTable table = Table();
  const auto rows = static_cast<std::int32_t>(table.rows);
  EXPECT_FALSE(PlanPleRows(table, std::vector<std::int32_t>{rows}, 1 << 20, 8).has_value());
  EXPECT_FALSE(PlanPleRows(table, std::vector<std::int32_t>{-1}, 1 << 20, 8).has_value());
  // Two rows a chunk apart need two blocks; room for one is refused.
  const std::vector<std::int32_t> two = {0, 30000};
  EXPECT_FALSE(PlanPleRows(table, two, kPleBlock, 8).has_value());
  EXPECT_TRUE(PlanPleRows(table, two, 2 * kPleBlock, 8).has_value());
  EXPECT_FALSE(PlanPleRows(table, two, 2 * kPleBlock, 1).has_value());
  // A table the reads cannot address: its rows past its stored range.
  PleTable longer = table;
  longer.rows += 100;
  EXPECT_FALSE(PlanPleRows(longer, two, 2 * kPleBlock, 8).has_value());
}

// The reads' outcomes, on the storage fake, under the storage lane's rules:
// an unknown submission is in flight and waited for (its row counts once
// its completion proves it); a short read or an error refuses the chunk's
// rows, and only after every read in flight has completed.
TEST(PleRowsTest, UnknownSubmissionsAreWaitedForAndShortOrFailedReadsRefuseAfterDraining) {
  using llmp::providers::Submission;
  using llmp::providers::fake::FakeStorage;
  PleTable table = Table();
  const auto file = Pattern(table.file_bytes);
  // Rows far enough apart that each is a read of its own.
  std::vector<std::int32_t> lookups;
  lookups.reserve(6);
  for (std::int32_t r = 0; r < 6; ++r) {
    lookups.push_back(r * 1000);
  }
  const auto plan = PlanPleRows(table, lookups, PleLandingBound(lookups.size()), lookups.size());
  ASSERT_TRUE(plan.has_value()) << plan.error();
  ASSERT_EQ(plan->reads.size(), lookups.size());
  const auto run = [&](const std::vector<FakeStorage::Script>& scripts, bool& drained) {
    FakeStorage storage(4, static_cast<std::uint32_t>(kPleBlock));
    table.fd = storage.AddFile(file);
    for (const FakeStorage::Script& s : scripts) {
      storage.ScriptNext(s);
    }
    std::vector<std::byte> landing(plan->landing_bytes + kPleBlock);
    const auto at = reinterpret_cast<std::uintptr_t>(landing.data());
    std::byte* aligned = landing.data() + ((kPleBlock - (at % kPleBlock)) % kPleBlock);
    auto read = ReadPleRows(storage, table.fd, *plan, aligned);
    drained = storage.in_flight() == 0;
    if (read) {
      const std::vector<std::byte> want = Land(*plan, file);
      EXPECT_EQ(std::memcmp(aligned, want.data(), want.size()), 0);
    }
    return read.has_value();
  };
  const auto script = [](Submission submission, std::optional<std::int64_t> result) {
    return FakeStorage::Script{.submission = submission, .result = result, .hold = false};
  };
  const FakeStorage::Script plain = script(Submission::kAccepted, std::nullopt);
  const FakeStorage::Script unknown = script(Submission::kUnknown, std::nullopt);
  bool drained = false;
  EXPECT_TRUE(run({unknown, plain, unknown}, drained));
  EXPECT_TRUE(drained);
  EXPECT_FALSE(run({plain, script(Submission::kAccepted, 100)}, drained));  // short
  EXPECT_TRUE(drained);
  EXPECT_FALSE(run({plain, plain, script(Submission::kAccepted, -EIO)}, drained));
  EXPECT_TRUE(drained);
}

// On a GB10: a real file read through io_uring into pinned memory, the rows
// gathered by the kernel into device memory.
TEST(CudaPleRowsTest, ReadsLandAndTheGatherPutsEveryRowInItsSlot) {
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path directory = scratch != nullptr
                                              ? std::filesystem::path(scratch)
                                              : std::filesystem::path(::testing::TempDir());
  std::filesystem::create_directories(directory);
  PleTable table = Table();
  const auto file = Pattern(table.file_bytes);
  const std::filesystem::path path = directory / "ple_rows_test.bin";
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(file.data()),
              static_cast<std::streamsize>(file.size()));
  }
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  ASSERT_GE(fd, 0);
  table.fd = fd;
  std::mt19937 rng(8);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::uniform_int_distribution<std::int32_t> row(0, static_cast<std::int32_t>(table.rows) - 1);
  std::vector<std::int32_t> lookups(2048);
  for (std::int32_t& l : lookups) {
    l = row(rng);
  }
  const std::uint64_t capacity = PleLandingBound(lookups.size());
  const auto plan = PlanPleRows(table, lookups, capacity, lookups.size());
  ASSERT_TRUE(plan.has_value()) << plan.error();
  auto ring = llmp::providers::UringStorage::Create(8);
  ASSERT_TRUE(ring.has_value());
  void* landing = nullptr;
  void* sources = nullptr;
  void* slots = nullptr;
  // The gather's grid past the rows: the count it reads (pinned, after the
  // sources) bounds what it writes, as a replayed graph's does.
  const std::size_t count = plan->sources.size();
  const std::size_t grid = count + 16;
  ASSERT_EQ(cudaMallocHost(&landing, capacity), cudaSuccess);
  ASSERT_EQ(cudaMallocHost(&sources, (count + 1) * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&slots, grid * 90), cudaSuccess);
  ASSERT_EQ(cudaMemset(slots, 0xAB, grid * 90), cudaSuccess);
  const auto read = ReadPleRows(**ring, fd, *plan, static_cast<std::byte*>(landing));
  ASSERT_TRUE(read.has_value()) << read.error();
  std::memcpy(sources, plan->sources.data(), count * 4);
  auto* const rows_count = static_cast<std::uint32_t*>(sources) + count;
  *rows_count = static_cast<std::uint32_t>(count);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_TRUE(GatherPleRows(
      static_cast<const std::byte*>(landing), static_cast<const std::uint32_t*>(sources),
      rows_count, static_cast<std::uint32_t>(grid), 90, static_cast<std::byte*>(slots), stream));
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::vector<std::byte> gathered(grid * 90);
  ASSERT_EQ(cudaMemcpy(gathered.data(), slots, gathered.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (std::size_t i = 0; i < lookups.size(); ++i) {
    const auto slot = static_cast<std::size_t>(plan->slots[i]);
    const std::uint64_t at = table.file_offset + (static_cast<std::uint64_t>(lookups[i]) * 90);
    EXPECT_EQ(std::memcmp(gathered.data() + (slot * 90), file.data() + at, 90), 0)
        << "lookup " << i;
  }
  for (std::size_t i = count * 90; i < grid * 90; ++i) {
    ASSERT_EQ(gathered[i], std::byte{0xAB}) << "a slot past the count at byte " << i;
  }
  (void)cudaStreamDestroy(stream);
  (void)cudaFree(slots);
  (void)cudaFreeHost(sources);
  (void)cudaFreeHost(landing);
  (void)::close(fd);
  std::filesystem::remove(path);
}

}  // namespace
