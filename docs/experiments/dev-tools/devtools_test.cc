// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// M0 tool smoke: exercises the GoogleTest features M1 expects to use.
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <expected>
#include <limits>
#include <string>

namespace {

struct Bytes {
  std::uint64_t value{};
  auto operator<=>(const Bytes&) const = default;
};

constexpr Bytes align_up(Bytes size, Bytes alignment) {
  return Bytes{(size.value + alignment.value - 1) / alignment.value * alignment.value};
}

enum class ReadError : std::uint8_t { short_read, cancelled };

std::expected<Bytes, ReadError> finish_read(Bytes requested, Bytes transferred) {
  if (transferred < requested) {
    return std::unexpected(ReadError::short_read);
  }
  return transferred;
}

void require_released(bool released) {
  if (!released) {
    std::abort();
  }
}

class CompletionSink {
 public:
  CompletionSink() = default;
  CompletionSink(const CompletionSink&) = delete;
  CompletionSink& operator=(const CompletionSink&) = delete;
  CompletionSink(CompletionSink&&) = delete;
  CompletionSink& operator=(CompletionSink&&) = delete;
  virtual ~CompletionSink() = default;
  virtual void complete(std::uint64_t op, bool success) = 0;
  virtual void retire(std::uint64_t op) = 0;
};

class MockSink : public CompletionSink {
 public:
  MOCK_METHOD(void, complete, (std::uint64_t op, bool success), (override));
  MOCK_METHOD(void, retire, (std::uint64_t op), (override));
};

void finish(CompletionSink& sink, std::uint64_t op) {
  sink.complete(op, true);
  sink.retire(op);
}

TEST(Bytes, AlignsToFourKiB) {
  EXPECT_EQ(align_up(Bytes{1}, Bytes{4096}), Bytes{4096});
  EXPECT_EQ(align_up(Bytes{4096}, Bytes{4096}), Bytes{4096});
}

TEST(Expected, ReportsShortRead) {
  const auto result = finish_read(Bytes{8}, Bytes{4});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), ReadError::short_read);
  EXPECT_EQ(finish_read(Bytes{8}, Bytes{8}).value(), Bytes{8});
}

struct AlignCase {
  std::uint64_t size;
  std::uint64_t expected;
};

class ChunkAlign : public testing::TestWithParam<AlignCase> {};

TEST_P(ChunkAlign, RoundsToTwoMiB) {
  constexpr Bytes chunk{2ULL << 20U};
  EXPECT_EQ(align_up(Bytes{GetParam().size}, chunk), Bytes{GetParam().expected});
}

INSTANTIATE_TEST_SUITE_P(Sizes, ChunkAlign,
                         testing::Values(AlignCase{1, 2ULL << 20U},
                                         AlignCase{2ULL << 20U, 2ULL << 20U},
                                         AlignCase{(2ULL << 20U) + 1, 4ULL << 20U}),
                         [](const testing::TestParamInfo<AlignCase>& info) {
                           return "bytes_" + std::to_string(info.param.size);
                         });

template <typename T>
class Unsigned : public testing::Test {};
using UnsignedTypes = testing::Types<std::uint16_t, std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(Unsigned, UnsignedTypes);

TYPED_TEST(Unsigned, WrapsAtMaximum) {
  const TypeParam max = std::numeric_limits<TypeParam>::max();
  EXPECT_EQ(static_cast<TypeParam>(max + 1U), TypeParam{0});
}

TEST(Mock, CompletesBeforeRetiring) {
  MockSink sink;
  const testing::InSequence order;
  EXPECT_CALL(sink, complete(7, true));
  EXPECT_CALL(sink, retire(7));
  finish(sink, 7);
}

TEST(DeathTest, AbortsOnUnreleasedBacking) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(require_released(false), "");
  require_released(true);
}

}  // namespace
