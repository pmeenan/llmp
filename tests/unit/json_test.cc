// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The general JSON parser (base/json.h), as untrusted input.

#include "base/json.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <string_view>

#include "expected_error.h"
#include "tokenizer_fixtures.h"

namespace {

namespace json = llmp::base::json;
using llmp::test_support::Failed;
using llmp::test_support::Get;

std::optional<std::uint64_t> FailOffset(std::string_view text, const json::Limits& limits = {}) {
  return Failed(json::Parse(text, limits), &json::Error::offset);
}

TEST(Json, ParsesEveryKind) {
  const auto doc = json::Parse(R"( {"a": [1, -2.5e3, true, false, null, "x"], "b": {}, "c": []} )");
  ASSERT_TRUE(doc.has_value()) << doc.error().ToString();
  const json::Value root = doc->root();
  ASSERT_TRUE(root.is_object());
  ASSERT_EQ(root.size(), 3U);
  EXPECT_EQ(root.key(0), "a");
  const json::Value a = Get(root, "a");
  ASSERT_EQ(a.size(), 6U);
  EXPECT_EQ(a.at(0).int64(), 1);
  EXPECT_TRUE(a.at(0).is_integer());
  EXPECT_FALSE(a.at(1).is_integer());
  EXPECT_EQ(a.at(1).number(), "-2.5e3");
  EXPECT_EQ(a.at(1).float64(), -2500.0);
  EXPECT_EQ(a.at(1).int64(), std::nullopt);
  EXPECT_TRUE(a.at(2).boolean());
  EXPECT_TRUE(a.at(3).is_bool());
  EXPECT_FALSE(a.at(3).boolean());
  EXPECT_TRUE(a.at(4).is_null());
  EXPECT_EQ(a.at(5).string(), "x");
  EXPECT_TRUE(Get(root, "b").is_object());
  EXPECT_EQ(Get(root, "b").size(), 0U);
  EXPECT_TRUE(Get(root, "c").is_array());
  EXPECT_EQ(Get(root, "c").size(), 0U);
  EXPECT_FALSE(root.find("d").has_value());
}

TEST(Json, UnescapesStrings) {
  // The \u escapes are spelled with doubled backslashes, so that the JSON
  // text holds them and not the characters.
  const auto doc = json::Parse(
      "[\"\\\"\\\\\\/\\b\\f\\n\\r\\t\", \"\\u00e9\\u4E2D\", \"\\ud83d\\ude00\", \"\xC3\xA9\"]");
  ASSERT_TRUE(doc.has_value()) << doc.error().ToString();
  const json::Value r = doc->root();
  EXPECT_EQ(r.at(0).string(), "\"\\/\b\f\n\r\t");
  EXPECT_EQ(r.at(1).string(), "é中");
  EXPECT_EQ(r.at(2).string(), "😀");
  EXPECT_EQ(r.at(3).string(), "é");
}

TEST(Json, KeepsNulInStrings) {
  const auto doc = json::Parse(R"(["a\u0000b"])");
  ASSERT_TRUE(doc.has_value());
  EXPECT_EQ(doc->root().at(0).string(), std::string_view("a\0b", 3));
}

TEST(Json, RefusesWhatRfc8259Refuses) {
  EXPECT_EQ(FailOffset(""), 0U);
  EXPECT_EQ(FailOffset("[1,]"), 3U);
  EXPECT_TRUE(FailOffset("{\"a\" 1}"));
  EXPECT_TRUE(FailOffset("01"));
  EXPECT_TRUE(FailOffset("1."));
  EXPECT_TRUE(FailOffset(".5"));
  EXPECT_TRUE(FailOffset("-"));
  EXPECT_TRUE(FailOffset("+1"));
  EXPECT_TRUE(FailOffset("NaN"));
  EXPECT_TRUE(FailOffset("tru"));
  EXPECT_TRUE(FailOffset("[1] x"));
  EXPECT_TRUE(FailOffset("\"a\tb\""));  // raw control character
  EXPECT_TRUE(FailOffset(R"("\x")"));
  EXPECT_TRUE(FailOffset(R"("\u12")"));
  EXPECT_TRUE(FailOffset("\"abc"));
  EXPECT_TRUE(FailOffset("{'a': 1}"));
}

TEST(Json, RefusesLoneSurrogates) {
  EXPECT_TRUE(FailOffset(R"("\ud800")"));
  EXPECT_TRUE(FailOffset(R"("\udc00")"));
  EXPECT_TRUE(FailOffset("\"\\ud800\\u0041\""));  // a high surrogate, then an escape but no low one
  EXPECT_TRUE(FailOffset(R"("\ud800x")"));
}

TEST(Json, RefusesIllFormedUtf8WithItsOffset) {
  EXPECT_EQ(FailOffset("[\"a\xC3\""), 3U);
  EXPECT_EQ(FailOffset("[\"\xED\xA0\x80\"]"), 2U);      // a surrogate
  EXPECT_EQ(FailOffset("[\"\xC0\xAF\"]"), 2U);          // overlong
  EXPECT_EQ(FailOffset("[\"\xF4\x90\x80\x80\"]"), 2U);  // above U+10FFFF
}

TEST(Json, RefusesDuplicateKeys) {
  EXPECT_TRUE(FailOffset(R"({"a": 1, "b": 2, "a": 3})"));
  EXPECT_TRUE(json::Parse(R"({"a": {"x": 1}, "b": {"x": 2}})").has_value());
}

TEST(Json, EnforcesItsLimits) {
  json::Limits limits;
  limits.max_depth = 3;
  EXPECT_TRUE(json::Parse("[[[1]]]", limits).has_value());
  EXPECT_TRUE(FailOffset("[[[[1]]]]", limits));
  limits = {};
  limits.max_values = 4;
  EXPECT_TRUE(json::Parse("[1, 2, 3]", limits).has_value());
  EXPECT_TRUE(FailOffset("[1, 2, 3, 4]", limits));
  limits = {};
  limits.max_string_bytes = 4;
  EXPECT_TRUE(json::Parse(R"(["ab", "cd"])", limits).has_value());
  EXPECT_TRUE(FailOffset(R"(["ab", "cde"])", limits));
  limits = {};
  limits.max_bytes = 3;
  EXPECT_TRUE(FailOffset("[11]", limits));
}

TEST(Json, DeepNestingIsBoundedNotRecursedWithoutLimit) {
  const std::string deep(100000, '[');
  EXPECT_TRUE(FailOffset(deep));
}

TEST(Json, ConvertsNumbers) {
  const auto doc =
      json::Parse("[9223372036854775807, 9223372036854775808, -0, 1e400, -1e400, 1e-400, 12e-2]");
  ASSERT_TRUE(doc.has_value());
  const json::Value r = doc->root();
  EXPECT_EQ(r.at(0).int64(), INT64_MAX);
  EXPECT_EQ(r.at(1).int64(), std::nullopt);
  EXPECT_EQ(r.at(1).float64(), 9223372036854775808.0);
  EXPECT_EQ(r.at(2).int64(), 0);
  EXPECT_EQ(r.at(3).float64(), INFINITY);
  EXPECT_EQ(r.at(4).float64(), -INFINITY);
  EXPECT_EQ(r.at(5).float64(), 0.0);
  EXPECT_EQ(r.at(6).float64(), 0.12);
}

// Corrupting a document anywhere gives an error or a document, never a
// crash; a document then reads fully.
TEST(Json, RandomCorruptionsNeverCrash) {
  const std::string base =
      "{\"a\": [1, -2.5e3, true, false, null, \"x\\u00e9\\ud83d\\ude00\"], \"b\": {\"c\": [[], "
      "{}]}, "
      "\"d\": \"\\n\\t\\\"\"}";
  std::mt19937 random(5);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::size_t parsed = 0;
  for (int round = 0; round < 20000; ++round) {
    std::string text = base;
    const int edits = 1 + static_cast<int>(random() % 3);
    for (int k = 0; k < edits; ++k) {
      text[random() % text.size()] = static_cast<char>(random() & 0xFFU);
    }
    const auto doc = json::Parse(text);
    if (!doc) {
      EXPECT_LE(doc.error().offset, text.size());
      continue;
    }
    ++parsed;
    std::size_t visited = 0;
    auto walk = [&](auto&& self, json::Value v) -> void {
      ++visited;
      if (v.is_number()) {
        (void)v.int64();
        (void)v.float64();
      }
      for (std::size_t i = 0; i < v.size(); ++i) {
        self(self, v.is_object() ? v.member(i) : v.at(i));
      }
    };
    walk(walk, doc->root());
    EXPECT_GT(visited, 0U);
  }
  EXPECT_GT(parsed, 0U);
}

TEST(Json, FirstInvalidUtf8FollowsTable3_7) {
  EXPECT_EQ(json::FirstInvalidUtf8("plain ascii"), std::nullopt);
  EXPECT_EQ(
      json::FirstInvalidUtf8(
          "\xC2\x80\xDF\xBF\xE0\xA0\x80\xED\x9F\xBF\xEE\x80\x80\xF0\x90\x80\x80\xF4\x8F\xBF\xBF"),
      std::nullopt);
  EXPECT_EQ(json::FirstInvalidUtf8("\x80"), 0U);
  EXPECT_EQ(json::FirstInvalidUtf8("a\xC1\xBF"), 1U);
  EXPECT_EQ(json::FirstInvalidUtf8("ab\xE0\x9F\x80"), 2U);
  EXPECT_EQ(json::FirstInvalidUtf8("\xED\xA0\x80"), 0U);
  EXPECT_EQ(json::FirstInvalidUtf8("\xF0\x8F\xBF\xBF"), 0U);
  EXPECT_EQ(json::FirstInvalidUtf8("\xF4\x90\x80\x80"), 0U);
  EXPECT_EQ(json::FirstInvalidUtf8("\xF5\x80\x80\x80"), 0U);
  EXPECT_EQ(json::FirstInvalidUtf8("xyz\xE2\x82"), 3U);
}

TEST(Json, QuotesMinimally) {
  std::string out;
  json::AppendQuoted(std::string_view("a\"b\\c\n\x01\x7F é", 11), out);
  EXPECT_EQ(out, "\"a\\\"b\\\\c\\n\\u0001\x7F é\"");
}

}  // namespace
