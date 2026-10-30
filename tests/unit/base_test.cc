// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The base module: diagnostic reports and SHA-256.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/process_retention.h"
#include "base/report.h"
#include "base/sha256.h"
#include "base/sha256_internal.h"

namespace {

using llmp::base::FormatBytes;

TEST(FormatBytes, BelowOneKibibyteIsBytes) {
  EXPECT_EQ(FormatBytes(0), "0 B");
  EXPECT_EQ(FormatBytes(1023), "1023 B");
}

TEST(FormatBytes, ExactUnitsAreWhole) {
  EXPECT_EQ(FormatBytes(1024), "1 KiB");
  EXPECT_EQ(FormatBytes(4096), "4 KiB");
  EXPECT_EQ(FormatBytes(std::uint64_t{2} << 20), "2 MiB");
  EXPECT_EQ(FormatBytes(std::uint64_t{3} << 40), "3 TiB");
  EXPECT_EQ(FormatBytes(std::uint64_t{1} << 63), "8 EiB");
}

TEST(FormatBytes, OthersHaveOneDecimal) {
  EXPECT_EQ(FormatBytes(1536), "1.5 KiB");
  EXPECT_EQ(FormatBytes(std::uint64_t{127598832} * 1024), "121.7 GiB");
  EXPECT_EQ(FormatBytes(std::numeric_limits<std::uint64_t>::max()), "16.0 EiB");
}

TEST(FormatBytes, RoundsUpToTheNextUnit) {
  EXPECT_EQ(FormatBytes((std::uint64_t{1} << 30) - 1), "1.0 GiB");
  EXPECT_EQ(FormatBytes((std::uint64_t{1} << 20) - 1), "1.0 MiB");
  EXPECT_EQ(FormatBytes((std::uint64_t{1023} << 20) + (std::uint64_t{900} << 10)), "1023.9 MiB");
}

TEST(Report, SectionsKeepTheirOrder) {
  llmp::base::Report report;
  report.AddSection("first").Add("a", "1");
  llmp::base::ReportSection& second = report.AddSection("second");
  second.Add("b", "2");
  second.Add("c", "3");
  ASSERT_EQ(report.sections.size(), 2U);
  EXPECT_EQ(report.sections[0].title, "first");
  ASSERT_EQ(report.sections[1].lines.size(), 2U);
  EXPECT_EQ(report.sections[1].lines[1].key, "c");
  EXPECT_EQ(report.sections[1].lines[1].value, "3");
}

TEST(Printable, EscapesWhatCouldForgeOrHideText) {
  using llmp::base::Printable;
  EXPECT_EQ(Printable("plain caf\xc3\xa9"), "plain caf\xc3\xa9");
  EXPECT_EQ(Printable("a\nb\tc\x7f"), "a\\x0ab\\x09c\\x7f");
  EXPECT_EQ(Printable("\xc2\x9b"
                      "31m"),
            "\\u009b31m");  // C1 CSI
  // NOLINTNEXTLINE(misc-misleading-bidirectional): the override is the input
  EXPECT_EQ(Printable("x\xe2\x80\xae"
                      "y"),
            "x\\u202ey");                                   // right-to-left override
  EXPECT_EQ(Printable("\xff\xc3"), "\\xff\\xc3");           // not UTF-8
  EXPECT_EQ(Printable("\xe0\x80\x80"), "\\xe0\\x80\\x80");  // overlong
}

TEST(Printable, EscapesSeparatorsAndTags) {
  using llmp::base::Printable;
  EXPECT_EQ(Printable("a\xe2\x80\xa8"
                      "b"),
            "a\\u2028b");                                // line separator
  EXPECT_EQ(Printable("\xf3\xa0\x80\x81"), "\\ue0001");  // a tag character
}

std::string Sha256Hex(std::string_view text) {
  return llmp::base::ToHex(llmp::base::Sha256().Update(text).Finish());
}

// FIPS 180-4's examples (NIST's SHA-256 example values).
// An owner kept for the process stays reachable and is counted; a null one
// is not kept.
TEST(ProcessRetention, KeepsEachOwnerAndIgnoresNull) {
  const auto before = llmp::base::ProcessRetained();
  static const int owner = 0;
  llmp::base::RetainForProcess(&owner);
  llmp::base::RetainForProcess(nullptr);
  EXPECT_EQ(llmp::base::ProcessRetained(), before + 1);
}

TEST(Sha256, MatchesTheStandardsExamples) {
  EXPECT_EQ(Sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(Sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(Sha256Hex(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

// Lengths around the padding's boundaries, against Python's hashlib.
TEST(Sha256, PadsEveryLengthAroundABlock) {
  EXPECT_EQ(Sha256Hex(std::string(55, 'x')),
            "d5e285683cd4efc02d021a5c62014694958901005d6f71e89e0989fac77e4072");
  EXPECT_EQ(Sha256Hex(std::string(56, 'x')),
            "04c26261370ee7541549d16dee320c723e3fd14671e66a099afe0a377c16888e");
  EXPECT_EQ(Sha256Hex(std::string(63, 'x')),
            "75220b47218278e656f2013bb8f0c455a25eaf01e86c64924e9d48d89776d6f2");
  EXPECT_EQ(Sha256Hex(std::string(64, 'x')),
            "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c");
  EXPECT_EQ(Sha256Hex(std::string(65, 'x')),
            "9537c5fdf120482f7d58d25e9ed583f52c02b4e304ea814db1633ad565aed7e9");
  EXPECT_EQ(Sha256Hex(std::string(119, 'x')),
            "000b48d4edf0fa7bee3c6236ecd2785baa5db4eeb8bb54341b029e0d9fa5fb0c");
  EXPECT_EQ(Sha256Hex(std::string(120, 'x')),
            "13f05a0b594787f5ecd315edc96141bd3243203d1b7d4f0836f37308b276ba98");
}

// Any split of the input gives the digest of the whole.
TEST(Sha256, UpdatesInPiecesHashTheWhole) {
  std::vector<std::byte> bytes;
  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < 256; ++i) {
      bytes.push_back(static_cast<std::byte>(i));
    }
  }
  const std::string want = "f3a25aa93aa2fbba28d79260535bbd6a5eb0fc1c24a8b0f04e12b484c1dfe363";
  EXPECT_EQ(llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish()), want);
  for (const std::size_t piece : {std::size_t{1}, std::size_t{7}, std::size_t{63}, std::size_t{64},
                                  std::size_t{65}, std::size_t{200}}) {
    llmp::base::Sha256 hash;
    for (std::size_t at = 0; at < bytes.size(); at += piece) {
      hash.Update(std::span(bytes).subspan(at, std::min(piece, bytes.size() - at)));
      // An empty update, whose span has no data pointer, changes nothing.
      hash.Update(std::span<const std::byte>());
    }
    EXPECT_EQ(llmp::base::ToHex(hash.Finish()), want) << piece;
  }
}

namespace sha = llmp::base::sha256_internal;

llmp::base::Sha256Digest ScalarDigest(std::span<const std::byte> message) {
  sha::State state = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::vector<std::uint8_t> padded;
  for (auto byte : message) padded.push_back(std::to_integer<std::uint8_t>(byte));
  padded.push_back(0x80);
  while (padded.size() % 64 != 56) padded.push_back(0);
  const auto bits = static_cast<std::uint64_t>(message.size()) * 8;
  for (unsigned i = 0; i < 8; ++i)
    padded.push_back(static_cast<std::uint8_t>(bits >> (56 - 8 * i)));
  sha::Scalar(state, padded.data(), padded.size() / 64);
  llmp::base::Sha256Digest digest{};
  for (std::size_t i = 0; i < state.size(); ++i)
    for (unsigned byte = 0; byte < 4; ++byte)
      digest[i * 4 + byte] = static_cast<std::uint8_t>(state[i] >> (24 - 8 * byte));
  return digest;
}

TEST(Sha256, CompressionDispatchPreservesStatesAcrossBlocksAndAlignments) {
  EXPECT_EQ(sha::Select(false), sha::Scalar);
  const auto hardware = sha::ArmCompressor();
  EXPECT_EQ(sha::Select(true), hardware ? hardware : sha::Scalar);
  for (const auto compressor : {sha::Select(false), sha::Select(true)}) {
    sha::State empty{1, 2, 3, 4, 5, 6, 7, 8};
    const auto before = empty;
    compressor(empty, nullptr, 0);
    EXPECT_EQ(empty, before);
    for (std::size_t offset = 0; offset < 32; ++offset) {
      for (std::size_t count : {1U, 2U, 3U, 16U, 31U}) {
        // The last block ends exactly at the allocation boundary.
        std::vector<std::uint8_t> input(offset + count * 64);
        for (std::size_t i = 0; i < input.size(); ++i)
          input[i] = static_cast<std::uint8_t>((i * 29 + offset * 17) % 256);
        sha::State expected{0, 0xffffffff, 0x87654321, 17, 0x80000000, 0x12345678, 91, 0xfedcba98};
        auto actual = expected;
        for (std::size_t block = 0; block < count; ++block)
          sha::Scalar(expected, input.data() + offset + block * 64, 1);
        compressor(actual, input.data() + offset, count);
        EXPECT_EQ(actual, expected) << offset << " " << count;
      }
    }
  }
}

TEST(Sha256, IncrementalCopiesAndPaddingMatchScalarAcrossEverySplit) {
  for (std::size_t length :
       {0U, 1U, 55U, 56U, 63U, 64U, 65U, 119U, 120U, 127U, 128U, 129U, 511U, 1024U}) {
    for (std::size_t offset : {0U, 1U, 7U, 15U}) {
      std::vector<std::byte> storage(length + offset);
      for (std::size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<std::byte>((i * 131 + length) % 256);
      const auto message = std::span<const std::byte>(storage).subspan(offset);
      const auto expected = ScalarDigest(message);
      EXPECT_EQ(llmp::base::Sha256().Update(message).Finish(), expected);
      for (std::size_t split = 0; split <= length; ++split) {
        llmp::base::Sha256 prefix;
        prefix.Update(message.first(split));
        auto checkpoint = prefix;
        prefix.Update(message.subspan(split));
        for (auto byte : message.subspan(split)) {
          checkpoint.Update(std::span(&byte, 1));
          checkpoint.Update(std::span<const std::byte>());
        }
        EXPECT_EQ(prefix.Finish(), expected) << length << " " << offset << " " << split;
        EXPECT_EQ(checkpoint.Finish(), expected) << length << " " << offset << " " << split;
      }
    }
  }
}

}  // namespace
