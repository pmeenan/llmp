// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Python's str case operations (chat/pycase.h) against Python 3.12.3 itself
// (tests/unit/data/chat/pycase-reference.json, written by
// docs/experiments/chat-template-corpus/pycase_reference.py): upper, lower,
// title and capitalize of every code point, islower and isupper of every
// code point, and every operation, Jinja2's title filter included, on
// strings that put Final_Sigma, case-ignorable and title-case characters
// and SpecialCasing's expansions in every context.

#include "chat/pycase.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/json.h"
#include "tokenizer/unicode.h"
#include "tokenizer_fixtures.h"

namespace {

namespace chat = llmp::chat;
namespace json = llmp::base::json;
namespace unicode = llmp::tokenizer::unicode;
using llmp::test_support::Get;
using llmp::test_support::LoadJson;

std::string Utf8(char32_t cp) {
  std::string s;
  unicode::AppendUtf8(cp, s);
  return s;
}

std::string Utf8(json::Value code_points) {
  std::string s;
  for (std::size_t i = 0; i < code_points.size(); ++i) {
    unicode::AppendUtf8(static_cast<char32_t>(code_points.at(i).int64().value_or(0)), s);
  }
  return s;
}

std::unordered_map<char32_t, std::string> Singles(json::Value table) {
  std::unordered_map<char32_t, std::string> out;
  for (std::size_t i = 0; i < table.size(); ++i) {
    out.emplace(static_cast<char32_t>(std::stoul(std::string(table.key(i)), nullptr, 16)),
                Utf8(table.member(i)));
  }
  return out;
}

std::vector<bool> Members(json::Value ranges) {
  std::vector<bool> in(0x110000, false);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const auto first = ranges.at(i).at(0).int64().value_or(0);
    const auto last = ranges.at(i).at(1).int64().value_or(-1);
    for (auto cp = first; cp <= last; ++cp) {
      in[static_cast<std::size_t>(cp)] = true;
    }
  }
  return in;
}

TEST(PythonCase, EveryCodePointMapsAsPythonMapsIt) {
  const json::Document doc = LoadJson("chat/pycase-reference.json");
  const json::Value singles = Get(doc.root(), "singles");
  const auto upper = Singles(Get(singles, "upper"));
  const auto lower = Singles(Get(singles, "lower"));
  const auto title = Singles(Get(singles, "title"));
  const auto islower = Members(Get(doc.root(), "islower"));
  const auto isupper = Members(Get(doc.root(), "isupper"));
  EXPECT_GT(upper.size(), 1000U);
  std::size_t wrong = 0;
  const auto expect = [&](const std::unordered_map<char32_t, std::string>& table, char32_t cp,
                          const std::string& text, chat::CaseOp op, std::string_view name) {
    const auto it = table.find(cp);
    const std::string& want = it != table.end() ? it->second : text;
    const std::string got = chat::PythonCase(text, op);
    if (got != want && ++wrong <= 20) {
      ADD_FAILURE() << name << " of U+" << std::hex << static_cast<std::uint32_t>(cp);
    }
  };
  for (char32_t cp = 0; cp < 0x110000; ++cp) {
    if (cp >= 0xD800 && cp <= 0xDFFF) {
      continue;
    }
    const std::string text = Utf8(cp);
    expect(upper, cp, text, chat::CaseOp::kUpper, "upper");
    expect(lower, cp, text, chat::CaseOp::kLower, "lower");
    expect(title, cp, text, chat::CaseOp::kTitle, "title");
    expect(title, cp, text, chat::CaseOp::kCapitalize, "capitalize");
    if (chat::PythonIsLower(text) != islower[cp] && ++wrong <= 20) {
      ADD_FAILURE() << "islower of U+" << std::hex << static_cast<std::uint32_t>(cp);
    }
    if (chat::PythonIsUpper(text) != isupper[cp] && ++wrong <= 20) {
      ADD_FAILURE() << "isupper of U+" << std::hex << static_cast<std::uint32_t>(cp);
    }
  }
  EXPECT_EQ(wrong, 0U);
}

TEST(PythonCase, StringsMapAsPythonAndJinjaMapThem) {
  const json::Document doc = LoadJson("chat/pycase-reference.json");
  const json::Value strings = Get(doc.root(), "strings");
  ASSERT_GE(strings.size(), 1000U);
  const std::vector<std::pair<std::string_view, chat::CaseOp>> ops = {
      {"upper", chat::CaseOp::kUpper},
      {"lower", chat::CaseOp::kLower},
      {"capitalize", chat::CaseOp::kCapitalize},
      {"title", chat::CaseOp::kTitle},
      {"jinja_title", chat::CaseOp::kJinjaTitle}};
  for (std::size_t i = 0; i < strings.size(); ++i) {
    const json::Value s = strings.at(i);
    const std::string_view text = Get(s, "text").string();
    for (const auto& [name, op] : ops) {
      EXPECT_EQ(chat::PythonCase(text, op), Get(s, name).string()) << name << " of " << text;
    }
    EXPECT_EQ(chat::PythonIsLower(text), Get(s, "islower").boolean()) << text;
    EXPECT_EQ(chat::PythonIsUpper(text), Get(s, "isupper").boolean()) << text;
  }
}

// The cases that matter most, spelled out.
TEST(PythonCase, SpecialCasingAndFinalSigma) {
  EXPECT_EQ(chat::PythonCase("straße", chat::CaseOp::kUpper), "STRASSE");
  EXPECT_EQ(chat::PythonCase("İ", chat::CaseOp::kLower), "i̇");
  EXPECT_EQ(chat::PythonCase("ﬃ", chat::CaseOp::kUpper), "FFI");
  EXPECT_EQ(chat::PythonCase("ﬃ", chat::CaseOp::kTitle), "Ffi");
  EXPECT_EQ(chat::PythonCase("ΟΔΟΣ ΟΔΟΣ'", chat::CaseOp::kLower), "οδος οδος'");
  EXPECT_EQ(chat::PythonCase("ΣΑ", chat::CaseOp::kLower), "σα");
  // ASCII context, which the fast path maps: . ' : ^ ` are case-ignorable.
  EXPECT_EQ(chat::PythonCase("ΑΣ.", chat::CaseOp::kLower), "ας.");
  EXPECT_EQ(chat::PythonCase("ΑΣ'a", chat::CaseOp::kLower), "ασ'a");
  EXPECT_EQ(chat::PythonCase("a.Σ", chat::CaseOp::kLower), "a.ς");
  EXPECT_EQ(chat::PythonCase(".Σ", chat::CaseOp::kLower), ".σ");
  EXPECT_EQ(chat::PythonCase("A'Σ1", chat::CaseOp::kLower), "a'ς1");
  EXPECT_EQ(chat::PythonCase("строка", chat::CaseOp::kUpper), "СТРОКА");
  EXPECT_EQ(chat::PythonCase("ǆ", chat::CaseOp::kTitle), "ǅ");
  EXPECT_EQ(chat::PythonCase("ǆ", chat::CaseOp::kJinjaTitle), "Ǆ");
  EXPECT_EQ(chat::PythonCase("o'neil-o'NEIL", chat::CaseOp::kTitle), "O'Neil-O'Neil");
  EXPECT_EQ(chat::PythonCase("o'neil-o'NEIL", chat::CaseOp::kJinjaTitle), "O'neil-O'neil");
}

}  // namespace
