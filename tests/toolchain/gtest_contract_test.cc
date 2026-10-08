// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The test-framework contract (D-059, D-066): GoogleTest 1.18.0 and gMock,
// built from the source lock (D-057) without exceptions, run the features
// llmpalooza's tests use, in every profile, death tests of libstdc++'s
// assertions (D-083) where the preset has them. NegativeControl.DISABLED_Fails runs
// only when asked for, to show that a failure fails the run
// (check_gtest_failure.cmake).

#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <expected>
#include <limits>
#include <optional>
#include <print>
#include <string>
#include <vector>

#include "expected_error.h"

#if GTEST_HAS_EXCEPTIONS
#error "GoogleTest must build without exceptions, like the tests that use it (D-066)"
#endif

namespace {

using llmp::test_support::Failed;

struct Bytes {
  std::uint64_t value{};
  auto operator<=>(const Bytes&) const = default;
};

constexpr Bytes AlignUp(Bytes size, Bytes alignment) {
  return Bytes{(size.value + alignment.value - 1) / alignment.value * alignment.value};
}

enum class ReadError : std::uint8_t { kShortRead, kCancelled };

std::expected<Bytes, ReadError> FinishRead(Bytes requested, Bytes transferred) {
  if (transferred < requested) {
    return std::unexpected(ReadError::kShortRead);
  }
  return transferred;
}

void RequireReleased(bool released) {
  if (!released) {
    std::abort();
  }
}

// Under -fno-exceptions a library path that would throw ends the process.
void ReadPastTheEnd() {
  const std::vector<int> empty;
  std::println("{}", empty.at(1));
}

class CompletionSink {
 public:
  CompletionSink() = default;
  CompletionSink(const CompletionSink&) = delete;
  CompletionSink& operator=(const CompletionSink&) = delete;
  CompletionSink(CompletionSink&&) = delete;
  CompletionSink& operator=(CompletionSink&&) = delete;
  virtual ~CompletionSink() = default;
  virtual void Complete(std::uint64_t op, bool success) = 0;
  virtual void Retire(std::uint64_t op) = 0;
};

class MockSink : public CompletionSink {
 public:
  MOCK_METHOD(void, Complete, (std::uint64_t op, bool success), (override));
  MOCK_METHOD(void, Retire, (std::uint64_t op), (override));
};

void Finish(CompletionSink& sink, std::uint64_t op) {
  sink.Complete(op, true);
  sink.Retire(op);
}

TEST(Bytes, AlignsToFourKiB) {
  EXPECT_EQ(AlignUp(Bytes{1}, Bytes{4096}), Bytes{4096});
  EXPECT_EQ(AlignUp(Bytes{4096}, Bytes{4096}), Bytes{4096});
}

TEST(Expected, ReportsShortRead) {
  const auto result = FinishRead(Bytes{8}, Bytes{4});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), ReadError::kShortRead);
  EXPECT_EQ(FinishRead(Bytes{8}, Bytes{8}).value_or(Bytes{}), Bytes{8});
}

// The safe reading of an error (D-083): a result that wrongly succeeded
// fails the expectation that names its error, whatever its value's bytes.
TEST(Expected, AWrongSuccessFailsTheExpectedError) {
  EXPECT_EQ(Failed(FinishRead(Bytes{8}, Bytes{4})), ReadError::kShortRead);
  EXPECT_EQ(Failed(FinishRead(Bytes{0}, Bytes{0})), std::nullopt);
  EXPECT_NONFATAL_FAILURE(EXPECT_EQ(Failed(FinishRead(Bytes{0}, Bytes{0})), ReadError::kShortRead),
                          "Failed");
}

struct AlignCase {
  std::uint64_t size;
  std::uint64_t expected;
};

class ChunkAlign : public testing::TestWithParam<AlignCase> {};

TEST_P(ChunkAlign, RoundsToTwoMiB) {
  constexpr Bytes kChunk{2ULL << 20U};
  EXPECT_EQ(AlignUp(Bytes{GetParam().size}, kChunk), Bytes{GetParam().expected});
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
  EXPECT_CALL(sink, Complete(7, true));
  EXPECT_CALL(sink, Retire(7));
  Finish(sink, 7);
}

TEST(Matchers, DescribeContainers) {
  const std::vector<int> values{1, 2, 3};
  EXPECT_THAT(values, testing::ElementsAre(1, testing::Gt(1), testing::Lt(4)));
  EXPECT_THAT(values, testing::Contains(2));
}

TEST(DeathTest, AbortsOnUnreleasedBacking) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(RequireReleased(false), "");
  RequireReleased(true);
}

TEST(DeathTest, ThrowingLibraryPathTerminates) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(ReadPastTheEnd(), "out_of_range");
}

#ifdef _GLIBCXX_ASSERTIONS
// With libstdc++'s assertions (D-083), a death test sees a precondition
// violation end the process, with the runtime's report of the condition.
TEST(DeathTest, ErrorOfAValueAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto success = FinishRead(Bytes{8}, Bytes{8});
  EXPECT_DEATH(std::println("{}", static_cast<int>(success.error())), "_M_has_value");
}
#endif

TEST(NegativeControl, DISABLED_Fails) {
  ADD_FAILURE() << "negative control: this test always fails";
}

}  // namespace
