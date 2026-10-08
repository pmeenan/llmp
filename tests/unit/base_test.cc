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

#include "base/report.h"
#include "base/sha256.h"

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

}  // namespace
