// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The general JSON parser (base/json.h), as untrusted input.

#include "base/json.h"

#include <gtest/gtest.h>
#include <pthread.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "expected_error.h"
#include "tokenizer_fixtures.h"

namespace {

namespace json = llmp::base::json;
using llmp::test_support::Failed;
using llmp::test_support::Get;

std::optional<std::uint64_t> FailOffset(std::string_view text, const json::Limits& limits = {}) {
  return Failed(json::Parse(text, limits), &json::Error::offset);
}

// The reason and offset a parse fails with, or nothing when it succeeds.
std::optional<std::pair<std::string_view, std::uint64_t>> Failure(std::string_view text,
                                                                  const json::Limits& limits = {}) {
  const auto result = json::Parse(text, limits);
  if (result) {
    return std::nullopt;
  }
  return std::pair{result.error().reason, result.error().offset};
}

using Parsed = std::expected<json::Document, json::Error>;

// Parses on a thread with a 256 KiB stack: far less than a parser that
// recursed per nesting level would need for the deep documents below.
Parsed ParseOnSmallStack(std::string_view text, const json::Limits& limits) {
  struct Call {
    std::string_view text;
    const json::Limits* limits;
    std::optional<Parsed> result;
  } call{.text = text, .limits = &limits, .result = std::nullopt};
  pthread_attr_t attr{};
  EXPECT_EQ(pthread_attr_init(&attr), 0);
  EXPECT_EQ(pthread_attr_setstacksize(&attr, std::size_t{256} << 10U), 0);
  pthread_t thread{};
  const int created = pthread_create(
      &thread, &attr,
      [](void* p) -> void* {
        auto* c = static_cast<Call*>(p);
        c->result = json::Parse(c->text, *c->limits);
        return nullptr;
      },
      &call);
  pthread_attr_destroy(&attr);
  EXPECT_EQ(created, 0);
  if (created != 0) {
    return std::unexpected(json::Error{"thread not created", 0});
  }
  EXPECT_EQ(pthread_join(thread, nullptr), 0);
  if (!call.result) {
    return std::unexpected(json::Error{"thread did not parse", 0});
  }
  return std::move(*call.result);
}

json::Limits Unbounded(std::size_t bytes) {
  json::Limits limits;
  limits.max_bytes = bytes;
  limits.max_depth = std::numeric_limits<std::size_t>::max();
  limits.max_values = bytes + 1;
  limits.max_string_bytes = bytes;
  return limits;
}

// Writes a document back as JSON with no space, without recursing.
std::string Serialize(json::Value root) {
  struct Frame {
    json::Value value;
    std::size_t next;
  };
  std::string out;
  std::vector<Frame> open;
  auto begin = [&](json::Value v) {
    switch (v.kind()) {
      case json::Kind::kNull:
        out += "null";
        break;
      case json::Kind::kFalse:
        out += "false";
        break;
      case json::Kind::kTrue:
        out += "true";
        break;
      case json::Kind::kNumber:
        out += v.number();
        break;
      case json::Kind::kString:
        json::AppendQuoted(v.string(), out);
        break;
      case json::Kind::kArray:
        out += '[';
        open.push_back({.value = v, .next = 0});
        break;
      case json::Kind::kObject:
        out += '{';
        open.push_back({.value = v, .next = 0});
        break;
    }
  };
  begin(root);
  while (!open.empty()) {
    const json::Value v = open.back().value;
    const std::size_t i = open.back().next;
    if (i == v.size()) {
      out += v.is_object() ? '}' : ']';
      open.pop_back();
      continue;
    }
    ++open.back().next;
    if (i > 0) {
      out += ',';
    }
    if (v.is_object()) {
      json::AppendQuoted(v.key(i), out);
      out += ':';
      begin(v.member(i));
    } else {
      begin(v.at(i));
    }
  }
  return out;
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

// Every refusal's reason and offset, which callers report.
TEST(Json, RefusalsGiveTheirReasonAndOffset) {
  using F = std::pair<std::string_view, std::uint64_t>;
  EXPECT_EQ(Failure(""), F("value expected", 0));
  EXPECT_EQ(Failure("["), F("value expected", 1));
  EXPECT_EQ(Failure("[1,]"), F("invalid number", 3));
  EXPECT_EQ(Failure("[1"), F("',' or ']' expected", 2));
  EXPECT_EQ(Failure("[1 2]"), F("',' or ']' expected", 3));
  EXPECT_EQ(Failure("[tru]"), F("invalid literal", 1));
  EXPECT_EQ(Failure("{"), F("member name expected", 1));
  EXPECT_EQ(Failure("{1}"), F("member name expected", 1));
  EXPECT_EQ(Failure(R"({"a" 1})"), F("':' expected", 5));
  EXPECT_EQ(Failure(R"({"a":)"), F("value expected", 5));
  EXPECT_EQ(Failure(R"({"a":1 "b":2})"), F("',' or '}' expected", 7));
  EXPECT_EQ(Failure("[[1]]x"), F("text after the value", 5));
  EXPECT_EQ(Failure("[1]]"), F("text after the value", 3));
  EXPECT_EQ(Failure(R"([{"a":1,"a":2}])"), F("duplicate member name", 14));
  EXPECT_EQ(Failure(R"({"a":[],"a":{}})"), F("duplicate member name", 15));
  json::Limits limits;
  limits.max_depth = 3;
  EXPECT_EQ(Failure("[[[[1]]]]", limits), F("nesting too deep", 3));
  limits.max_depth = 0;
  EXPECT_EQ(Failure("[]", limits), F("nesting too deep", 0));
  EXPECT_FALSE(Failure("1", limits));
  limits = {};
  limits.max_values = 4;
  EXPECT_EQ(Failure("[1, 2, 3, 4]", limits), F("too many values", 10));
  limits = {};
  limits.max_string_bytes = 4;
  EXPECT_EQ(Failure(R"(["ab", "cde"])", limits), F("too many string bytes", 12));
  limits.max_string_bytes = 1;
  EXPECT_EQ(Failure(R"({"ab": 1})", limits), F("too many string bytes", 5));
}

// A caller may allow any depth: the parser does not recurse, so a million
// open brackets fail cleanly on a small stack.
TEST(Json, UnterminatedDeepNestingFailsOnASmallStack) {
  const std::string deep(1'000'000, '[');
  const auto result = ParseOnSmallStack(deep, Unbounded(deep.size()));
  EXPECT_EQ(Failed(result, &json::Error::reason), "value expected");
  EXPECT_EQ(Failed(result, &json::Error::offset), deep.size());
}

TEST(Json, BalancedDeepNestingParsesOnASmallStack) {
  constexpr std::size_t kDepth = 1'000'000;
  const std::string deep = std::string(kDepth, '[') + std::string(kDepth, ']');
  const auto doc = ParseOnSmallStack(deep, Unbounded(deep.size()));
  ASSERT_TRUE(doc.has_value()) << doc.error().ToString();
  json::Value v = doc->root();
  for (std::size_t level = 1; level < kDepth; ++level) {
    ASSERT_TRUE(v.is_array()) << level;
    ASSERT_EQ(v.size(), 1U) << level;
    v = v.at(0);
  }
  EXPECT_TRUE(v.is_array());
  EXPECT_EQ(v.size(), 0U);

  // Exactly max_depth levels parse; one more is refused at its bracket.
  json::Limits limits = Unbounded(deep.size());
  limits.max_depth = kDepth;
  EXPECT_TRUE(ParseOnSmallStack(deep, limits).has_value());
  limits.max_depth = kDepth - 1;
  const auto refused = ParseOnSmallStack(deep, limits);
  EXPECT_EQ(Failed(refused, &json::Error::reason), "nesting too deep");
  EXPECT_EQ(Failed(refused, &json::Error::offset), kDepth - 1);

  // The default depth refuses the 65th bracket at its offset.
  EXPECT_EQ(Failure(deep), std::pair(std::string_view("nesting too deep"), std::uint64_t{64}));
}

// Objects and arrays nested deeply, each with members before and after the
// nested one, keep their layout: the document writes back as its text.
TEST(Json, DeepMixedNestingRoundTrips) {
  constexpr std::size_t kLevels = 200'000;
  std::string text;
  for (std::size_t i = 0; i < kLevels; ++i) {
    text += i % 2 == 0 ? std::format(R"({{"n":{},"d":)", i) : std::format(R"([true,"{}",)", i);
  }
  text += "-1.5e3";
  for (std::size_t i = kLevels; i-- > 0;) {
    text += i % 2 == 0 ? R"(,"e":[]})" : ",null]";
  }
  const auto doc = ParseOnSmallStack(text, Unbounded(text.size()));
  ASSERT_TRUE(doc.has_value()) << doc.error().ToString();
  EXPECT_EQ(Serialize(doc->root()), text);
  json::Value v = doc->root();
  for (std::size_t i = 0; i < kLevels; ++i) {
    if (i % 2 == 0) {
      ASSERT_TRUE(v.is_object()) << i;
      ASSERT_EQ(v.size(), 3U) << i;
      EXPECT_EQ(Get(v, "n").int64(), static_cast<std::int64_t>(i));
      v = Get(v, "d");
    } else {
      ASSERT_TRUE(v.is_array()) << i;
      ASSERT_EQ(v.size(), 4U) << i;
      EXPECT_EQ(v.at(1).string(), std::to_string(i));
      v = v.at(2);
    }
  }
  EXPECT_EQ(v.float64(), -1500.0);

  // A repeated member name deep inside is refused where its object closes.
  const std::string dup =
      std::string(kLevels, '[') + R"({"a":1,"a":2})" + std::string(kLevels, ']');
  const auto refused = ParseOnSmallStack(dup, Unbounded(dup.size()));
  EXPECT_EQ(Failed(refused, &json::Error::reason), "duplicate member name");
  EXPECT_EQ(Failed(refused, &json::Error::offset), kLevels + 13);
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
  // Exponents at the ends of 64 bits, which parse, beside a mantissa's own
  // magnitude (found by benchmarks/import_fuzz under UBSan: the sum
  // overflowed).
  const auto edges = json::Parse(
      "[5449e9223372036854775807, -5449e9223372036854775807, 0.0001e-9223372036854775808, "
      "5449e-9223372036854775808]");
  ASSERT_TRUE(edges.has_value());
  EXPECT_EQ(edges->root().at(0).float64(), INFINITY);
  EXPECT_EQ(edges->root().at(1).float64(), -INFINITY);
  EXPECT_EQ(edges->root().at(2).float64(), 0.0);
  EXPECT_EQ(edges->root().at(3).float64(), 0.0);
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

TEST(Json, AccessorsCheckKindAndBounds) {
  json::Document empty_doc;
  EXPECT_TRUE(empty_doc.root().is_null());
  EXPECT_EQ(empty_doc.root().size(), 0U);
  EXPECT_TRUE(empty_doc.root().at(0).is_null());
  EXPECT_EQ(empty_doc.root().key(0), "");
  EXPECT_TRUE(empty_doc.root().member(0).is_null());

  const auto doc = json::Parse(R"({"arr":[10,20],"obj":{"k":"v"},"num":42})");
  ASSERT_TRUE(doc.has_value());
  const auto root = doc->root();
  EXPECT_TRUE(root.at(0).is_null());
  EXPECT_EQ(root.key(3), "");
  EXPECT_TRUE(root.member(3).is_null());

  const auto arr = root.find("arr");
  ASSERT_TRUE(arr.has_value());
  EXPECT_EQ(arr->at(0).int64(), 10);
  EXPECT_TRUE(arr->at(2).is_null());
  EXPECT_EQ(arr->key(0), "");
  EXPECT_TRUE(arr->member(0).is_null());
}

}  // namespace
