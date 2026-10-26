// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The native tokenizer on synthetic vocabularies: Unicode tables and UTF-8,
// NFC, the pre-tokenizers, BPE, special tokens, bounds, decoding, and the
// GGUF and tokenizer.json readers as untrusted input. The real vocabularies
// and the references' outputs are tokenizer_models_test's.

#include "tokenizer/tokenizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/json.h"
#include "base/work_pulse.h"
#include "expected_error.h"
#include "tokenizer/error.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/sentencepiece.h"
#include "tokenizer/unicode.h"

namespace {

namespace tok = llmp::tokenizer;
namespace uni = llmp::tokenizer::unicode;
using llmp::test_support::Failed;
using tok::PreTokenizer;
using tok::Rule;
using tok::TokenId;
using tok::TokenKind;

std::string Utf8(std::u32string_view cps) {
  std::string out;
  for (const char32_t c : cps) {
    uni::AppendUtf8(c, out);
  }
  return out;
}

// --- Unicode ------------------------------------------------------------------

TEST(Unicode, TablesAreUcd15_1) {
  EXPECT_EQ(uni::Version(), "15.1.0");
  EXPECT_EQ(uni::GetCategory(U'A'), uni::Category::kLu);
  EXPECT_EQ(uni::GetCategory(U'é'), uni::Category::kLl);
  EXPECT_EQ(uni::GetCategory(U'中'), uni::Category::kLo);
  EXPECT_EQ(uni::GetCategory(U'٣'), uni::Category::kNd);
  EXPECT_EQ(uni::GetCategory(0x0301), uni::Category::kMn);
  EXPECT_EQ(uni::GetCategory(U' '), uni::Category::kZs);
  EXPECT_EQ(uni::GetCategory(0x0378), uni::Category::kCn);
  EXPECT_EQ(uni::GetCategory(0xE000), uni::Category::kCo);
  EXPECT_EQ(uni::GetCategory(0x1F600), uni::Category::kSo);
  EXPECT_EQ(uni::GetCategory(0x10FFFF), uni::Category::kCn);
  EXPECT_EQ(uni::GetCategory(0x110000), uni::Category::kCn);
  // New in 15.1 (CJK extension I), and new only in 16.0 (unassigned here).
  EXPECT_EQ(uni::GetCategory(0x2EBF0), uni::Category::kLo);
  EXPECT_EQ(uni::GetCategory(0x1FAE9), uni::Category::kCn);
  EXPECT_EQ(uni::GetCategory(0xAC00), uni::Category::kLo);
  EXPECT_EQ(uni::GetCategory(0xD7A3), uni::Category::kLo);
  EXPECT_EQ(uni::GetCategory(0xD800), uni::Category::kCs);
}

TEST(Unicode, DefaultFullCaseFoldingUsesPinnedUcdData) {
  const auto fold = [](char32_t cp) {
    const auto mapped = uni::CaseFold(cp);
    return std::u32string(mapped.view().begin(), mapped.view().end());
  };
  EXPECT_EQ(fold(U'ſ'), U"s");
  EXPECT_EQ(fold(U'K'), U"k");
  EXPECT_EQ(fold(U'ß'), U"ss");
  EXPECT_EQ(fold(U'ẞ'), U"ss");
  EXPECT_EQ(fold(U'İ'), U"i\u0307");
  EXPECT_EQ(fold(U'ς'), U"σ");
  EXPECT_EQ(fold(U'ﬃ'), U"ffi");
  EXPECT_EQ(fold(U'中'), U"中");
}

TEST(Unicode, WhiteSpaceIsThePropListSet) {
  for (const char32_t c :
       {U'\x09', U'\x0A', U'\x0B', U'\x0C', U'\x0D', U'\x20', U'\x85', U'\xA0', U'\x1680',
        U'\x2000', U'\x200A', U'\x2028', U'\x2029', U'\x202F', U'\x205F', U'\x3000'}) {
    EXPECT_TRUE(uni::IsWhiteSpace(c)) << static_cast<std::uint32_t>(c);
  }
  for (const char32_t c : {U'\x1C', U'\x1F', U'\x200B', U'\x180E', U'\xFEFF', U'\x41'}) {
    EXPECT_FALSE(uni::IsWhiteSpace(c)) << static_cast<std::uint32_t>(c);
  }
}

TEST(Unicode, DecodesStrictUtf8) {
  std::vector<char32_t> out;
  ASSERT_TRUE(uni::DecodeUtf8("a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80", out).has_value());
  EXPECT_EQ(out, (std::vector<char32_t>{U'a', 0xE9, 0x4E2D, 0x1F600}));
  auto offset = [](std::string_view s) {
    std::vector<char32_t> o;
    return Failed(uni::DecodeUtf8(s, o), &tok::Error::item);
  };
  EXPECT_EQ(offset("ab\x80"), 2U);
  EXPECT_EQ(offset("\xC0\x80"), 0U);          // overlong NUL
  EXPECT_EQ(offset("x\xE0\x80\x80"), 1U);     // overlong
  EXPECT_EQ(offset("\xED\xA0\x80"), 0U);      // surrogate
  EXPECT_EQ(offset("\xF4\x90\x80\x80"), 0U);  // above U+10FFFF
  EXPECT_EQ(offset("\xF8\x88\x80\x80\x80"), 0U);
  EXPECT_EQ(offset("abc\xE2\x82"), 3U);  // truncated
  EXPECT_EQ(Failed(uni::ValidateUtf8("ok\xFF"), &tok::Error::item), 2U);
  EXPECT_TRUE(uni::ValidateUtf8("").has_value());
}

TEST(Unicode, ReplacesMaximalSubparts) {
  // The Unicode Standard's example (chapter 3, U+FFFD substitution).
  EXPECT_EQ(uni::ReplaceInvalidUtf8("\x61\xF1\x80\x80\xE1\x80\xC2\x62\x80\x63\x80\xBF\x64"),
            "a���b�c��d");
  EXPECT_EQ(uni::ReplaceInvalidUtf8("\xED\xA0\x80"), "���");
  EXPECT_EQ(uni::ReplaceInvalidUtf8("ok é"), "ok é");
  EXPECT_EQ(uni::ReplaceInvalidUtf8("\xF0\x9F\x98"), "�");
}

TEST(Unicode, CompletePrefixHoldsBackOnlyWhatCanComplete) {
  EXPECT_EQ(uni::CompleteUtf8Prefix("abc"), 3U);
  EXPECT_EQ(uni::CompleteUtf8Prefix("ab\xC3"), 2U);
  EXPECT_EQ(uni::CompleteUtf8Prefix("ab\xF0\x9F\x98"), 2U);
  EXPECT_EQ(uni::CompleteUtf8Prefix("ab\xF0\x9F\x98\x80"), 6U);
  EXPECT_EQ(uni::CompleteUtf8Prefix("ab\x80"), 3U);      // cannot complete: emit (as U+FFFD)
  EXPECT_EQ(uni::CompleteUtf8Prefix("ab\xE0\x80"), 4U);  // ill formed already
  EXPECT_EQ(uni::CompleteUtf8Prefix("ab\xFF"), 3U);
}

std::string Nfc(std::u32string_view in) {
  std::vector<char32_t> v(in.begin(), in.end());
  uni::ToNfc(v);
  return Utf8(std::u32string_view(v.data(), v.size()));
}

TEST(Unicode, Nfc) {
  EXPECT_EQ(Nfc(U"é"), "é");
  EXPECT_EQ(Nfc(U"abc"), "abc");
  EXPECT_EQ(Nfc(U"각"), "각");  // Hangul L V T
  EXPECT_EQ(Nfc(U"각"), "각");   // LV + T
  EXPECT_EQ(Nfc(U"ạ́"), "ạ́");      // reordered: dot below first
  EXPECT_EQ(Nfc(U"ạ́"), "ạ́");
  EXPECT_EQ(Nfc(U"Å"), "Å");  // singletons
  EXPECT_EQ(Nfc(U"Ω"), "Ω");
  EXPECT_EQ(Nfc(U"क़"), "क़");  // composition exclusion stays decomposed
  EXPECT_EQ(Nfc(U"ḍ̇"), "ḍ̇");
  EXPECT_EQ(Nfc(U"́e"), "́e");  // no starter to compose with
  EXPECT_EQ(Nfc(U"Tiếng"), "Tiếng");
  EXPECT_EQ(Nfc(U"̈́"), "̈́");  // NFC_QC=No, decomposes
}

TEST(Unicode, NfcOfALongCombiningRunIsNotQuadratic) {
  std::vector<char32_t> v(200000, 0x0301);
  v.insert(v.begin(), U'e');
  v.push_back(0x0323);
  uni::ToNfc(v);
  ASSERT_EQ(v.size(), 200001U);
  EXPECT_EQ(v[0], 0x1EB9U);  // e + dot below, then the acutes
}

// --- pre-tokenizers ------------------------------------------------------------

std::vector<std::string> Pieces(std::string_view p, std::string_view text) {
  std::vector<char32_t> cps;
  EXPECT_TRUE(uni::DecodeUtf8(text, cps).has_value());
  std::vector<std::uint32_t> lengths;
  const auto preset = tok::ReadPreTokenizerPreset(p);
  EXPECT_TRUE(preset.has_value());
  if (!preset) return {};
  const auto plan = tok::SplitPlan::Compile(preset->expressions, preset->dialect);
  EXPECT_TRUE(plan.has_value());
  if (!plan) return {};
  auto budget = plan->Budget(text.size() * 4 + 1);
  EXPECT_TRUE(plan->Split(cps, lengths, budget).has_value());
  std::vector<std::string> out;
  std::size_t at = 0;
  for (const std::uint32_t n : lengths) {
    EXPECT_GT(n, 0U);
    out.push_back(Utf8(std::u32string_view(cps.data() + at, n)));
    at += n;
  }
  EXPECT_EQ(at, cps.size());
  return out;
}

using Strings = std::vector<std::string>;

TEST(PreTokenize, Qwen2) {
  const std::string_view q = "qwen2";
  EXPECT_EQ(Pieces(q, "Hello world"), (Strings{"Hello", " world"}));
  EXPECT_EQ(Pieces(q, "I'm OK'LL"), (Strings{"I", "'m", " OK", "'LL"}));
  EXPECT_EQ(Pieces(q, "abc123"), (Strings{"abc", "1", "2", "3"}));
  EXPECT_EQ(Pieces(q, "a  b"), (Strings{"a", " ", " b"}));
  EXPECT_EQ(Pieces(q, "x   "), (Strings{"x", "   "}));
  EXPECT_EQ(Pieces(q, "a\n\nb"), (Strings{"a", "\n\n", "b"}));
  EXPECT_EQ(Pieces(q, "a \n \n b"), (Strings{"a", " \n \n", " b"}));
  EXPECT_EQ(Pieces(q, "!!!\n\nx"), (Strings{"!!!\n\n", "x"}));
  EXPECT_EQ(Pieces(q, " !!"), (Strings{" !!"}));
  EXPECT_EQ(Pieces(q, "éx"), (Strings{"e", "́x"}));
  EXPECT_EQ(Pieces(q, " "), (Strings{" "}));
  EXPECT_EQ(Pieces(q, "\r\nx"), (Strings{"\r\n", "x"}));
  EXPECT_EQ(Pieces(q, "\tx"), (Strings{"\tx"}));
}

TEST(PreTokenize, Qwen35TakesMarksIntoWords) {
  const std::string_view q = "qwen35";
  EXPECT_EQ(Pieces(q, "éx"), (Strings{"éx"}));
  // A mark is no longer a symbol: it ends the symbol run and starts a word.
  EXPECT_EQ(Pieces(q, "क्ष !́"), (Strings{"क्ष", " !", "́"}));
  EXPECT_EQ(Pieces("qwen2", "क्ष !́"), (Strings{"क", "्ष", " !́"}));
  EXPECT_EQ(Pieces(q, " ́́"), (Strings{" ́́"}));
}

TEST(PreTokenize, ContractionsMatchAsciiLettersOfEitherCase) {
  const std::string_view q = "qwen35";
  EXPECT_EQ(Pieces(q, "a'Sx"), (Strings{"a", "'S", "x"}));
  EXPECT_EQ(Pieces(q, "a'lLx"), (Strings{"a", "'lL", "x"}));
  EXPECT_EQ(Pieces(q, "a'ſx"), (Strings{"a", "'ſx"}));  // long s is no s
  EXPECT_EQ(Pieces(q, "a'Kx"), (Strings{"a", "'Kx"}));
  EXPECT_EQ(Pieces(q, "'"), (Strings{"'"}));
  EXPECT_EQ(Pieces(q, "'r"), (Strings{"'r"}));  // 're needs its e: the word takes ' as its prefix
}

TEST(PreTokenize, DeepSeekV3Stages) {
  const std::string_view d = "deepseek-v3";
  EXPECT_EQ(Pieces(d, "abc12345def"), (Strings{"abc", "123", "45", "def"}));
  EXPECT_EQ(Pieces(d, "日本語テキスト"), (Strings{"日本語テキスト"}));
  EXPECT_EQ(Pieces(d, "hello世界!"), (Strings{"hello", "世界", "!"}));
  EXPECT_EQ(Pieces(d, "#include x"), (Strings{"#include", " x"}));
  EXPECT_EQ(Pieces(d, " !!\n\nx"), (Strings{" !!\n\n", "x"}));
  EXPECT_EQ(Pieces(d, "a  b"), (Strings{"a", " ", " b"}));
  EXPECT_EQ(Pieces(d, "‍‍x"), (Strings{"‍", "‍x"}));  // one Cf prefixes the word
  EXPECT_EQ(Pieces(d, "‍‍"), (Strings{"‍‍"}));        // no match: one gap
  EXPECT_EQ(Pieces(d, "5‍"), (Strings{"5", "‍"}));
  EXPECT_EQ(Pieces(d, "Hí!"), (Strings{"Hí", "!"}));
  EXPECT_EQ(Pieces(d, "don't"), (Strings{"don", "'t"}));
  EXPECT_EQ(Pieces(d, "x   "), (Strings{"x", "   "}));
  EXPECT_EQ(Pieces(d, "x   y"), (Strings{"x", "  ", " y"}));
}

std::vector<std::string> RegexPieces(std::string_view expression, std::u32string_view text,
                                     tok::RegexDialect dialect = tok::RegexDialect::kHuggingFace) {
  auto regex = tok::Regex::Compile(expression, dialect);
  EXPECT_TRUE(regex) << Failed(regex, &tok::Error::ToString).value_or("");
  if (!regex) return {};
  tok::RegexBudget budget{tok::kRegexWorkFactor * (text.size() + 1) * (regex->instructions() + 1)};
  std::vector<std::uint32_t> lengths;
  const auto split = regex->Split(text, lengths, budget);
  EXPECT_TRUE(split) << Failed(split, &tok::Error::ToString).value_or("");
  std::vector<std::string> pieces;
  std::size_t at = 0;
  for (const auto n : lengths) {
    EXPECT_GT(n, 0U);
    pieces.push_back(Utf8(text.substr(at, n)));
    at += n;
  }
  EXPECT_EQ(at, text.size());
  return pieces;
}

// Independent HF expectations: tokenizers 0.22.2 isolated Split probe,
// r8-final-hf-probes.log. GGUF expectations follow the locked llama.cpp
// non-multiline ECMAScript fallback, not Oniguruma's default line anchors.
TEST(Regex, SourceDialectKeepsLineAndAbsoluteAnchorsDistinct) {
  EXPECT_EQ(RegexPieces("^a", U"x\na y"), (Strings{"x\n", "a", " y"}));
  EXPECT_EQ(RegexPieces("b$", U"xb\ny"), (Strings{"x", "b", "\ny"}));
  EXPECT_EQ(RegexPieces("$", U"ab\ncd\n"), (Strings{"ab", "\ncd", "\n"}));
  EXPECT_EQ(RegexPieces("^", U"ab\ncd\n"), (Strings{"ab\n", "cd\n"}));
  EXPECT_EQ(RegexPieces(R"(\A)", U"ab\ncd\n"), (Strings{"ab\ncd\n"}));
  EXPECT_EQ(RegexPieces(R"(\z)", U"ab\ncd\n"), (Strings{"ab\ncd\n"}));
  EXPECT_EQ(RegexPieces(R"(\Aa)", U"x\na y"), (Strings{"x\na y"}));
  EXPECT_EQ(RegexPieces(R"(b\z)", U"xb\ny"), (Strings{"xb\ny"}));
  EXPECT_EQ(RegexPieces(R"(b\z)", U"xb\n"), (Strings{"xb\n"}));
  EXPECT_EQ(RegexPieces(R"(b\z)", U"xb"), (Strings{"x", "b"}));
  EXPECT_EQ(RegexPieces("^a", U"x\na y", tok::RegexDialect::kGguf), (Strings{"x\na y"}));
  EXPECT_EQ(RegexPieces("b$", U"xb\ny", tok::RegexDialect::kGguf), (Strings{"xb\ny"}));
  EXPECT_EQ(RegexPieces("b$", U"xb\n", tok::RegexDialect::kGguf), (Strings{"xb\n"}));
  EXPECT_EQ(RegexPieces(R"(\s+$)", U"a \nb \n", tok::RegexDialect::kGguf),
            (Strings{"a \nb", " \n"}));
  EXPECT_EQ(RegexPieces(R"(\s+$)", U"a \nb \n"), (Strings{"a", " ", "\nb", " \n"}));
  EXPECT_EQ(RegexPieces(".+", U"a\rb\nc\u2028d"), (Strings{"a\rb", "\n", "c\u2028d"}));
  EXPECT_EQ(RegexPieces(".+", U"a\rb\nc\u2028d", tok::RegexDialect::kGguf),
            (Strings{"a", "\r", "b", "\n", "c", "\u2028", "d"}));
  // Oniguruma's m flag controls dotall, not its default line anchors.
  // It remains explicitly unsupported rather than being treated as an alias.
  EXPECT_FALSE(tok::Regex::Compile("(?m:^a)"));
  auto preset = tok::ReadPreTokenizerPreset("deepseek-llm");
  ASSERT_TRUE(preset);
  EXPECT_EQ(preset->dialect, tok::RegexDialect::kGguf);
  EXPECT_NE(std::ranges::find(preset->expressions, R"(\s+$)"), preset->expressions.end());
  EXPECT_EQ(Pieces("deepseek-llm", "a \nb \n"), (Strings{"a", " ", "\n", "b", " ", "\n"}));
}

TEST(Regex, CharacterClassAlgebraIsNotLiteralAmpersands) {
  EXPECT_FALSE(tok::Regex::Compile("[[:alpha:]]"));
  // Additional independent class-precedence probe a52665ed63d2874e... .
  const std::u32string_view sample = U"za&bcdeiou\n";
  EXPECT_EQ(RegexPieces(R"([a-z&&[^aeiou]])", sample),
            (Strings{"z", "a&", "b", "c", "d", "eiou\n"}));
  EXPECT_EQ(RegexPieces("[ab&&bc]", sample), (Strings{"za&", "b", "cdeiou\n"}));
  EXPECT_EQ(RegexPieces("[[ab]c&&[bc]]", sample), (Strings{"za&", "b", "c", "deiou\n"}));
  for (const auto expression : {"[a&&]", "[&&a]"}) {
    EXPECT_EQ(RegexPieces(expression, sample), (Strings{"za&bcdeiou\n"}));
    EXPECT_EQ(RegexPieces(expression, U""), Strings{});
  }
  EXPECT_EQ(RegexPieces("[a&b]", sample), (Strings{"z", "a", "&", "b", "cdeiou\n"}));
  // Removed-mode pinned probes establish membership; isolated partitions alone
  // cannot distinguish a one-character match from a retained unmatched gap.
  EXPECT_EQ(RegexPieces("x[^ab&&bc]x", U"!xbx~"), (Strings{"!xbx~"}));
  EXPECT_EQ(RegexPieces("x[^ab&&bc]x", U"!xax~"), (Strings{"!", "xax", "~"}));
  EXPECT_EQ(RegexPieces("x[^ab&&bc]x", U"!xcx~"), (Strings{"!", "xcx", "~"}));
  EXPECT_EQ(RegexPieces("[^a&&b]+", sample), (Strings{"za&bcdeiou\n"}));
  EXPECT_EQ(RegexPieces("[a&&b]", U"za&bz"), (Strings{"za&bz"}));
  EXPECT_EQ(RegexPieces("[a&&a]", U"za&bz"), (Strings{"z", "a", "&bz"}));
  EXPECT_EQ(RegexPieces(R"([a\&\&a])", U"za&bz"), (Strings{"z", "a", "&", "bz"}));
  EXPECT_EQ(RegexPieces("[a&&b]", U"za&bz", tok::RegexDialect::kGguf),
            (Strings{"z", "a", "&", "b", "z"}));
  EXPECT_EQ(RegexPieces(R"([\p{L}&&[^a-z]]+)", U"aÉ中b!"), (Strings{"a", "É中", "b!"}));
  // A deep expression has source-sized explicit compiler/matcher stacks.
  std::string nested;
  for (unsigned i = 0; i < 256; ++i) nested += "[a&&";
  nested += "a" + std::string(256, ']');
  EXPECT_EQ(RegexPieces(nested, U"zaaz"), (Strings{"z", "a", "a", "z"}));
  const auto program = tok::Regex::Compile(nested);
  ASSERT_TRUE(program);
  EXPECT_LT(program->WorkingBytes(4), tok::Regex::ConstructionBytes(nested.size()));
}

TEST(Regex, OrderedAlternationGreedyLazyAndPossessive) {
  EXPECT_EQ(RegexPieces("a|ab", U"ab"), (Strings{"a", "b"}));
  EXPECT_EQ(RegexPieces("ab|a", U"ab"), (Strings{"ab"}));
  EXPECT_EQ(RegexPieces("a+a|a", U"aaaa"), (Strings{"aaaa"}));
  EXPECT_EQ(RegexPieces("a+?a", U"aaaa"), (Strings{"aa", "aa"}));
  EXPECT_EQ(RegexPieces("a++a|a", U"aaaa"), (Strings{"a", "a", "a", "a"}));
  EXPECT_EQ(RegexPieces("(a|ab)++b|.", U"abb"), (Strings{"ab", "b"}));
  EXPECT_EQ(RegexPieces("(ab|a){1,2}+b|.", U"ababb"), (Strings{"ababb"}));
  EXPECT_EQ(RegexPieces("a{2,3}?", U"aaaaaa"), (Strings{"aa", "aa", "aa"}));
  EXPECT_EQ(RegexPieces("(a?)*", U"aab"), (Strings{"aa", "b"}));
}

TEST(Regex, GapsEmptyMatchesAndPositionPredicatesNeverDropInput) {
  EXPECT_EQ(RegexPieces("x+", U"abxxcdx"), (Strings{"ab", "xx", "cd", "x"}));
  EXPECT_EQ(RegexPieces("", U"abc"), (Strings{"a", "b", "c"}));
  EXPECT_EQ(RegexPieces("a*?", U"abc"), (Strings{"a", "b", "c"}));
  EXPECT_EQ(RegexPieces("^a|b$", U"ab\n"), (Strings{"a", "b", "\n"}));
  EXPECT_EQ(RegexPieces(R"(\s+(?!\S)|\s+|\p{L}+)", U"a   b\r\n"),
            (Strings{"a", "  ", " ", "b", "\r\n"}));
  EXPECT_EQ(RegexPieces(R"(\s+)", U"\x1c\u0085\u00a0x"),
            (Strings{std::string(1, '\x1c'), "\u0085\u00a0", "x"}));
  EXPECT_EQ(RegexPieces(R"((?i:'s|'ll))", U"'S'Ll'ſ"), (Strings{"'S", "'Ll", "'ſ"}));
}

// Independent tokenizers 0.22.2 / Oniguruma probes, rather than Python
// regex or the implementation under test: r8-regex-fold-probes.json SHA256
// 06c11ad9c6a028297e2657a241730c1efc50bd67c583d6040ab6325141f94d15.
TEST(Regex, ScopedCaseFoldingPreservesLiteralAndInstructionBoundaries) {
  EXPECT_EQ(RegexPieces("(?i:s)", U"!ſ~"), (Strings{"!", "ſ", "~"}));
  EXPECT_EQ(RegexPieces("(?i:k)", U"!K~"), (Strings{"!", "K", "~"}));
  EXPECT_EQ(RegexPieces("(?i:é)", U"!É~"), (Strings{"!", "É", "~"}));
  EXPECT_EQ(RegexPieces("(?i:é)", U"!é~"), (Strings{"!é~"}));
  EXPECT_EQ(RegexPieces("(?i:σ)", U"!Σσς~"), (Strings{"!", "Σ", "σ", "ς", "~"}));
  EXPECT_EQ(RegexPieces("(?i:ss)", U"!ßẞssSS~"), (Strings{"!", "ß", "ẞ", "ss", "SS", "~"}));
  EXPECT_EQ(RegexPieces("(?i:ß)", U"!ssSS~"), (Strings{"!", "ss", "SS", "~"}));
  for (const auto expression :
       {"(?i:s{2})", "(?i:[s][s])", "(?i:s)(?i:s)", "(?i:(s)(s))", "(?i:s+)", "(?i:[A-Z])"}) {
    EXPECT_EQ(RegexPieces(expression, U"!ßẞ~"), (Strings{"!ßẞ~"})) << expression;
  }
  EXPECT_EQ(RegexPieces("(?i:[A-Z])", U"!ſK~"), (Strings{"!", "ſ", "K", "~"}));
  EXPECT_EQ(RegexPieces("(?i:[A-Z])", U"!İıßσ~"), (Strings{"!İıßσ~"}));
  EXPECT_EQ(RegexPieces("(?i:i)", U"!İı~"), (Strings{"!İı~"}));
  EXPECT_EQ(RegexPieces("(?i:ss)x", U"!ßx~"), (Strings{"!", "ßx", "~"}));
  EXPECT_EQ(RegexPieces("(?i:(s|ss))x", U"!ßx~"), (Strings{"!", "ßx", "~"}));
  EXPECT_EQ(RegexPieces("(?i:s)s", U"!ß~"), (Strings{"!ß~"}));
}

TEST(Regex, SourceContractionsUseUnicodeFoldingWithoutChangingGgufClasses) {
  constexpr std::string_view hf =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  // Bare-source probes SHA 5f1c76c044512c286c7afcc943e9b94bfbdbad6192b955143b06c7b1e755a4b3.
  EXPECT_EQ(RegexPieces(hf, U"'ſ"), (Strings{"'ſ"}));
  EXPECT_EQ(RegexPieces(hf, U" 'ſ"), (Strings{" '", "ſ"}));
  EXPECT_EQ(RegexPieces(hf, U"a'ſ"), (Strings{"a", "'ſ"}));
  EXPECT_EQ(RegexPieces(hf, U"x'ſz"), (Strings{"x", "'ſ", "z"}));
  EXPECT_EQ(RegexPieces(hf, U"a'Kz"), (Strings{"a", "'Kz"}));
  EXPECT_EQ(RegexPieces(hf, U"!x'ſz~"), (Strings{"!x", "'ſ", "z", "~"}));
  EXPECT_EQ(RegexPieces(hf, U"!x'Sz~"), (Strings{"!x", "'S", "z", "~"}));
  // GGUF's actual pinned expression spells its contraction classes in ASCII.
  EXPECT_EQ(Pieces("qwen2", "!x'ſz~"), (Strings{"!x", "'ſz", "~"}));
}

TEST(Regex, PositiveClassLookaheadRetainsOriginalPositionAndCategory) {
  EXPECT_EQ(RegexPieces(R"((?=[\p{L}])[^a-z]+)", U"!A中a~"), (Strings{"!", "A中", "a~"}));
  EXPECT_EQ(RegexPieces(R"(((?=[\p{L}])([^A-Z]))+)", U"!é中A~"), (Strings{"!", "é中", "A~"}));
}

// Independent tokenizers 0.22.2 isolated Split oracle, log SHA256
// b5fe2135b53ca401a76c47b6d7309883283d3b925f0a1ff38a20f09e9d90420d.
TEST(Regex, SuccessfulZeroWidthMatchesAreBoundariesWithoutEmptyPieces) {
  EXPECT_EQ(RegexPieces(R"((?=[a]))", U"baac"), (Strings{"b", "a", "ac"}));
  EXPECT_EQ(RegexPieces(R"((?=[a]))", U"aaa"), (Strings{"a", "a", "a"}));
  EXPECT_EQ(RegexPieces("^", U"abc"), (Strings{"abc"}));
  EXPECT_EQ(RegexPieces("$", U"abc"), (Strings{"abc"}));
  EXPECT_EQ(RegexPieces("a*", U"baac"), (Strings{"b", "aa", "c"}));
  EXPECT_EQ(RegexPieces("(?:)", U"abc"), (Strings{"a", "b", "c"}));
  EXPECT_EQ(RegexPieces("a*?", U"baac"), (Strings{"b", "a", "a", "c"}));
  EXPECT_EQ(RegexPieces("(a?)*?", U"baac"), (Strings{"b", "a", "a", "c"}));
  EXPECT_EQ(RegexPieces("x", U"baac"), (Strings{"baac"}));
}

TEST(Regex, FundedSourceAndFrameStorageReplaceSmallPolicyLimits) {
  const std::string nested = std::string(4096, '(') + "a" + std::string(4096, ')');
  llmp::base::WorkPulse pulse;
  llmp::base::SetThreadPulse(&pulse);
  const auto regex = tok::Regex::Compile(nested);
  llmp::base::SetThreadPulse(nullptr);
  ASSERT_TRUE(regex) << Failed(regex, &tok::Error::ToString).value_or("");
  EXPECT_GT(pulse.beats(), 16U);
  EXPECT_EQ(RegexPieces(nested, U"ab"), (Strings{"a", "b"}));
  const std::string long_literal(16385, 'a');
  EXPECT_GT(tok::Regex::ConstructionBytes(long_literal.size()),
            tok::Regex::ConstructionBytes(long_literal.size() - 1));
  EXPECT_EQ(RegexPieces(long_literal, std::u32string(16385, U'a')), (Strings{long_literal}));
  EXPECT_EQ(tok::Regex::ConstructionBytes(SIZE_MAX), UINT64_MAX);
}

TEST(Regex, GeneralCategoryAggregatesInversesAndCaseCategories) {
  EXPECT_EQ(RegexPieces(R"(\p{Lu}+|\p{Ll}+|\p{Lt}+|\p{Lm}+|\p{Lo}+|\p{M}+|\p{N}+)",
                        U"ABabǅʰ中\u0301\u0903Ⅳ٢²"),
            (Strings{"AB", "ab", "ǅ", "ʰ", "中", "\u0301\u0903", "Ⅳ٢²"}));
  EXPECT_EQ(RegexPieces(R"([\p{L}\p{M}]+|\P{L}+)", U"a中́!! 42"), (Strings{"a中́", "!! 42"}));
  EXPECT_EQ(RegexPieces(R"(\p{P}+|\p{S}+|\p{Z}+|\p{C}+)", U"!?€🙂 \u200d"),
            (Strings{"!?", "€🙂", " ", "\u200d"}));
  EXPECT_EQ(RegexPieces(R"([^a-c]+|[a-c]+)", U"xyabcz"), (Strings{"xy", "abc", "z"}));
}

TEST(Regex, ParserLimitsUnsupportedSyntaxAndCompactCounts) {
  for (const auto expression :
       {R"(\1)", R"(\p{Han})", R"((?=a))", R"((?i:[é]))", R"((?i:\p{L}))", R"(\w)", "[a-", "[z-a]",
        "a{3,2}", "(", "a**", "\\", "a{4294967296}"}) {
    EXPECT_FALSE(tok::Regex::Compile(expression)) << expression;
  }
  EXPECT_TRUE(tok::Regex::Compile(std::string(65, '(') + "a" + std::string(65, ')')));
  EXPECT_TRUE(tok::Regex::Compile(std::string(16385, 'a')));
  EXPECT_EQ(tok::Regex::ConstructionBytes(SIZE_MAX), UINT64_MAX);
  EXPECT_FALSE(tok::Regex::Compile(std::string(1, static_cast<char>(0xff))));
  auto compact = tok::Regex::Compile("(a?){4294967294}");
  ASSERT_TRUE(compact);
  EXPECT_LT(compact->instructions(), 20U);
  tok::RegexBudget budget{1000};
  std::vector<std::uint32_t> lengths;
  EXPECT_EQ(Failed(compact->Split({}, lengths, budget), &tok::Error::rule), Rule::kBounds);
}

TEST(Regex, AdversarialWorkAndStorageHaveDeterministicBounds) {
  const auto regex = tok::Regex::Compile("(a+)+b");
  ASSERT_TRUE(regex);
  const std::u32string input(64, U'a');
  std::uint64_t spent = 0;
  for (unsigned run = 0; run < 2; ++run) {
    tok::RegexBudget budget{16 * (input.size() + 1) * (regex->instructions() + 1)};
    std::vector<std::uint32_t> lengths;
    const auto split = regex->Split(input, lengths, budget);
    EXPECT_EQ(Failed(split, &tok::Error::rule), Rule::kBounds);
    if (run) EXPECT_EQ(budget.spent, spent);
    spent = budget.spent;
  }
  const std::vector<std::string> expressions = {"x", "(a+)+b"};
  const auto plan = tok::SplitPlan::Compile(expressions);
  ASSERT_TRUE(plan);
  tok::RegexBudget budget{200};
  std::vector<std::uint32_t> lengths;
  const auto split = plan->Split(U"axaaaaaaaaaaaaaaaaaaaa", lengths, budget);
  EXPECT_EQ(Failed(split, &tok::Error::rule), Rule::kBounds);
  EXPECT_LE(budget.spent, 200U);
}

TEST(Regex, ReproducibleSyntaxFuzzAndCancellation) {
  std::mt19937 rng(808);
  constexpr std::string_view alphabet = "abc|()*+?{}[],^$\\123";
  for (unsigned round = 0; round < 500; ++round) {
    std::string expression;
    for (unsigned n = rng() % 40; n > 0; --n) expression += alphabet[rng() % alphabet.size()];
    const auto regex = tok::Regex::Compile(expression);
    if (!regex) continue;
    tok::RegexBudget budget{10000};
    std::vector<std::uint32_t> lengths;
    const auto split = regex->Split(std::u32string_view(U"aabcbaaa"), lengths, budget);
    if (split) {
      std::size_t total = 0;
      for (const auto length : lengths) {
        EXPECT_GT(length, 0U);
        total += length;
      }
      EXPECT_EQ(total, 8U);
    } else
      EXPECT_EQ(split.error().rule, Rule::kBounds);
  }
  llmp::base::WorkPulse pulse;
  pulse.Cancel();
  llmp::base::SetThreadPulse(&pulse);
  const auto cancelled = tok::Regex::Compile("a");
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(Failed(cancelled, &tok::Error::rule), Rule::kCancelled);
}

TEST(PreTokenize, RequiredPresetsCompileAndConserveLongUnicodeInput) {
  for (const auto name :
       {"llama-bpe", "gpt-2", "phi-2", "gpt-4o", "tekken", "starcoder", "falcon", "command-r",
        "qwen2", "qwen35", "deepseek-llm", "deepseek-coder", "deepseek-v3", "joyai-llm"}) {
    const auto preset = tok::ReadPreTokenizerPreset(name);
    ASSERT_TRUE(preset) << name;
    const auto plan = tok::SplitPlan::Compile(preset->expressions, preset->dialect);
    ASSERT_TRUE(plan) << name << " " << Failed(plan, &tok::Error::ToString).value_or("");
    std::string input;
    for (unsigned i = 0; i < 300; ++i) input += "Hello'LL 世界क्षé́12345\r\n  λΑǅʰǂ🙂! ";
    std::vector<char32_t> cps;
    ASSERT_TRUE(uni::DecodeUtf8(input, cps));
    auto budget = plan->Budget(input.size() * 4 + 1);
    std::vector<std::uint32_t> lengths;
    ASSERT_TRUE(plan->Split(cps, lengths, budget)) << name;
    std::size_t total = 0;
    for (const auto n : lengths) {
      EXPECT_GT(n, 0U);
      total += n;
    }
    EXPECT_EQ(total, cps.size()) << name;
  }
}

// --- BPE on a synthetic byte-level vocabulary ---------------------------------------

// GPT-2's byte alphabet, written independently of the tokenizer's.
std::string ByteText(unsigned b) {
  unsigned n = 0;
  for (unsigned x = 0; x < b; ++x) {
    const bool printable = (x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || x >= 0xAE;
    n += printable ? 0 : 1;
  }
  const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
  std::string out;
  uni::AppendUtf8(printable ? b : 256 + n, out);
  return out;
}

std::string Alphabet(std::string_view bytes) {
  std::string out;
  for (const char c : bytes) {
    out += ByteText(static_cast<unsigned char>(c));
  }
  return out;
}

struct Vocab {
  tok::TokenizerSpec spec;

  Vocab() {
    for (unsigned b = 0; b < 256; ++b) {
      spec.tokens.push_back(ByteText(b));
      spec.kinds.push_back(TokenKind::kNormal);
    }
  }
  TokenId Add(std::string text, TokenKind kind = TokenKind::kNormal) {
    spec.tokens.push_back(std::move(text));
    spec.kinds.push_back(kind);
    return static_cast<TokenId>(spec.tokens.size() - 1);
  }
  // Merges two byte strings; adds the result as a token. Returns its ID.
  TokenId Merge(std::string_view a, std::string_view b) {
    spec.merges.emplace_back(Alphabet(a), Alphabet(b));
    return Add(Alphabet(std::string(a) + std::string(b)));
  }
  tok::Tokenizer Build() const {
    auto t = tok::Tokenizer::Create(spec);
    EXPECT_TRUE(t.has_value()) << t.error().ToString();
    return std::move(*t);
  }
};

std::vector<TokenId> Encode(const tok::Tokenizer& t, std::string_view text,
                            tok::EncodeOptions o = {}) {
  std::vector<TokenId> ids;
  const auto r = t.Encode(text, o, ids);
  EXPECT_TRUE(r.has_value()) << r.error().ToString();
  return ids;
}

TokenId Byte(char c) { return static_cast<unsigned char>(c); }

void CheckMarkedNormalizedControl(const tok::Tokenizer& tokenizer, std::string_view control,
                                  TokenId id) {
  const auto plain = Encode(tokenizer, control);
  EXPECT_EQ(std::ranges::find(plain, id), plain.end());
  tok::EncodeOptions parse{.special = tok::SpecialTokens::kParse, .max_tokens = plain.size()};
  std::vector<TokenId> out;
  ASSERT_TRUE(tokenizer.EncodeMarked(control, {}, parse, out));
  EXPECT_EQ(out, plain);

  const auto text = std::string(control) + std::string(control);
  const std::array spans = {tok::SpecialSpan{0, control.size()}};
  std::vector<TokenId> expected{id};
  expected.insert(expected.end(), plain.begin(), plain.end());
  parse.max_tokens = expected.size();
  out = {-1};  // max_tokens bounds the append, not the caller's existing prefix.
  std::vector<std::size_t> positions;
  ASSERT_TRUE(tokenizer.EncodeMarked(text, spans, parse, out, &positions));
  EXPECT_EQ(std::vector<TokenId>(out.begin() + 1, out.end()), expected);
  EXPECT_EQ(out.front(), -1);
  EXPECT_EQ(positions, (std::vector<std::size_t>{1}));

  --parse.max_tokens;
  out = {-1};
  const auto bounded = tokenizer.EncodeMarked(text, spans, parse, out);
  EXPECT_EQ(Failed(bounded, &tok::Error::rule), Rule::kOutputTooLarge);
  EXPECT_LE(out.size(), 1 + parse.max_tokens);

  llmp::base::WorkPulse pulse;
  pulse.Cancel();
  out.clear();
  llmp::base::SetThreadPulse(&pulse);
  const auto cancelled = tokenizer.EncodeMarked(text, spans, parse, out);
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(Failed(cancelled, &tok::Error::rule), Rule::kCancelled);
  EXPECT_TRUE(out.empty());
}

TEST(Create, ConstructionPaymentCountsMovedCapacitiesAndSpecialTrie) {
  Vocab v;
  auto first = tok::Tokenizer::ConstructionBytes(v.spec);
  ASSERT_TRUE(first);
  v.spec.tokens.reserve(v.spec.tokens.capacity() + 1024);
  auto capacity = tok::Tokenizer::ConstructionBytes(v.spec);
  ASSERT_TRUE(capacity);
  EXPECT_GE(*capacity - *first, 1024 * sizeof(std::string));
  v.Add("<|startup-special|>", TokenKind::kControl);
  v.Merge("a", "b");
  auto full = tok::Tokenizer::ConstructionBytes(v.spec);
  ASSERT_TRUE(full);
  EXPECT_GT(*full, *capacity);
  auto tokenizer = tok::Tokenizer::Create(std::move(v.spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
  EXPECT_TRUE(tokenizer->Find("<|startup-special|>"));
  tok::TokenizerSpec malformed;
  EXPECT_FALSE(tok::Tokenizer::ConstructionBytes(malformed));
  malformed.tokens.push_back("a");
  EXPECT_FALSE(tok::Tokenizer::ConstructionBytes(malformed));
}

TEST(Bpe, MergesByRankThenLeftmost) {
  Vocab v;
  v.spec.pre_tokenizer = PreTokenizer::kByteBpe;
  const TokenId bc = v.Merge("b", "c");  // rank 0
  v.Merge("a", "b");                     // rank 1: never applies to "abc"
  const TokenId aa = v.Merge("a", "a");  // rank 2
  const TokenId he = v.Merge("h", "e");
  const TokenId ll = v.Merge("l", "l");
  const TokenId hell = v.Merge("he", "ll");
  const TokenId hello = v.Merge("hell", "o");
  const tok::Tokenizer t = v.Build();
  EXPECT_EQ(Encode(t, "abc"), (std::vector<TokenId>{Byte('a'), bc}));
  EXPECT_EQ(Encode(t, "aaa"), (std::vector<TokenId>{aa, Byte('a')}));
  EXPECT_EQ(Encode(t, "aaaa"), (std::vector<TokenId>{aa, aa}));
  EXPECT_EQ(Encode(t, "hello"), (std::vector<TokenId>{hello}));
  EXPECT_EQ(Encode(t, "hell"), (std::vector<TokenId>{hell}));
  EXPECT_EQ(Encode(t, "helo"), (std::vector<TokenId>{he, Byte('l'), Byte('o')}));
  EXPECT_EQ(Encode(t, "ll ll"), (std::vector<TokenId>{ll, Byte(' '), ll}));
  EXPECT_EQ(Encode(t, ""), (std::vector<TokenId>{}));
}

TEST(Bpe, EveryByteEncodesAndRoundTrips) {
  Vocab v;
  v.Merge("\xC3", "\xA9");  // é
  const tok::Tokenizer t = v.Build();
  const std::string text = "é\x7F\x01 \t\n中😀";
  const auto ids = Encode(t, text);
  std::string back;
  ASSERT_TRUE(t.Decode(ids, {}, back).has_value());
  EXPECT_EQ(back, text);
  EXPECT_EQ(Encode(t, "é"), (std::vector<TokenId>{256}));
}

TEST(Bpe, IgnoreMergesTakesWholePieces) {
  Vocab v;
  v.Merge("a", "b");
  const TokenId whole = v.Add(Alphabet("abc"));  // no merge makes it
  v.spec.ignore_merges = true;
  const tok::Tokenizer t = v.Build();
  EXPECT_EQ(Encode(t, "abc"), (std::vector<TokenId>{whole}));
  EXPECT_EQ(Encode(t, "abcd").size(), 3U);  // "abcd" is no token: BPE
}

TEST(Bpe, NfcNormalizationAppliesToText) {
  Vocab v;
  v.spec.normalization = tok::Normalization::kNfc;
  const tok::Tokenizer t = v.Build();
  EXPECT_EQ(Encode(t, "é"), Encode(t, "é"));
  Vocab raw;
  const tok::Tokenizer r = raw.Build();
  EXPECT_NE(Encode(r, "é"), Encode(r, "é"));
}

TEST(SpecialTokens, ControlOnlyWhenAskedUserDefinedAlways) {
  Vocab v;
  const TokenId end = v.Add("<|end|>", TokenKind::kControl);
  const TokenId think = v.Add("<think>", TokenKind::kUserDefined);
  const tok::Tokenizer t = v.Build();
  tok::EncodeOptions parse;
  parse.special = tok::SpecialTokens::kParse;
  const auto parsed = Encode(t, "x<|end|>y<think>", parse);
  EXPECT_EQ(parsed, (std::vector<TokenId>{Byte('x'), end, Byte('y'), think}));
  const auto plain = Encode(t, "x<|end|>y<think>");
  ASSERT_EQ(plain.size(), 10U);  // x < | end | > as bytes, y, <think>
  EXPECT_EQ(plain.back(), think);
  EXPECT_EQ(std::count(plain.begin(), plain.end(), end), 0);
}

TEST(SpecialTokens, LeftmostLongest) {
  Vocab v;
  const TokenId ab = v.Add("<ab>", TokenKind::kUserDefined);
  const TokenId abc = v.Add("<ab>c", TokenKind::kUserDefined);
  const tok::Tokenizer t = v.Build();
  EXPECT_EQ(Encode(t, "<ab>c<ab>"), (std::vector<TokenId>{abc, ab}));
}

TEST(SpecialTokens, MarkedSpansAreTheOnlyControlTokens) {
  Vocab v;
  const TokenId start = v.Add("<s>", TokenKind::kControl);
  const TokenId think = v.Add("<think>", TokenKind::kUserDefined);
  const tok::Tokenizer t = v.Build();
  const std::string text = "<s>a<s><think>";
  const std::vector<tok::SpecialSpan> spans = {{0, 3}};
  std::vector<TokenId> ids;
  std::vector<std::size_t> at;
  tok::EncodeOptions parse;
  parse.special = tok::SpecialTokens::kParse;  // ignored for the text between spans
  ASSERT_TRUE(t.EncodeMarked(text, spans, parse, ids, &at).has_value());
  ASSERT_EQ(ids.size(), 6U);
  EXPECT_EQ(ids[0], start);
  EXPECT_EQ(ids[1], Byte('a'));
  EXPECT_EQ(ids[2], Byte('<'));  // the unmarked <s> stays text
  EXPECT_EQ(ids[5], think);
  EXPECT_EQ(at, (std::vector<std::size_t>{0}));

  auto rule = [&](std::vector<tok::SpecialSpan> s) {
    std::vector<TokenId> o;
    return Failed(t.EncodeMarked(text, s, {}, o), &tok::Error::rule);
  };
  EXPECT_EQ(rule({{1, 3}}), Rule::kSpecialToken);          // not a token's text
  EXPECT_EQ(rule({{0, 3}, {0, 3}}), Rule::kSpecialToken);  // overlapping
  EXPECT_EQ(rule({{12, 5}}), Rule::kSpecialToken);         // past the end
  EXPECT_EQ(rule({{4, 3}}), std::nullopt);                 // the second <s>
  EXPECT_EQ(rule({{3, 1}}), Rule::kSpecialToken);          // "a" is a normal token
}

TEST(Encode, RefusesIllFormedInputAndBounds) {
  Vocab v;
  const tok::Tokenizer t = v.Build();
  std::vector<TokenId> ids;
  EXPECT_EQ(Failed(t.Encode("ok\xC3(", {}, ids), &tok::Error::rule), Rule::kInvalidUtf8);
  EXPECT_EQ(Failed(t.Encode("ok\xC3(", {}, ids), &tok::Error::item), 2U);
  tok::EncodeOptions small;
  small.max_bytes = 4;
  EXPECT_EQ(Failed(t.Encode("hello", small, ids), &tok::Error::rule), Rule::kInputTooLarge);
  small = {};
  small.max_tokens = 3;
  ids.assign(10, 0);  // bounds count only this call's tokens
  EXPECT_TRUE(t.Encode("abc", small, ids).has_value());
  EXPECT_EQ(Failed(t.Encode("abcd", small, ids), &tok::Error::rule), Rule::kOutputTooLarge);
}

TEST(Encode, LongRunsStayLinear) {
  Vocab v;
  v.Merge(" ", " ");
  const tok::Tokenizer t = v.Build();
  const std::string spaces(1U << 20U, ' ');
  const auto ids = Encode(t, spaces);
  EXPECT_EQ(ids.size(), 1U << 19U);
}

// Arbitrary source expressions can join across the old fixed-window cuts.
TEST(Encode, GenericRegexKeepsMergesAcrossLongFragments) {
  Vocab v;
  v.spec.split_expressions = {".+"};
  v.Merge("a", " ");
  const auto merged = v.Merge("a ", "b");
  const auto t = v.Build();
  std::string text((std::size_t{1} << 20U) - 1, '!');
  text += "a b";
  const auto ids = Encode(t, text);
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids.back(), merged);
  EXPECT_EQ(ids.size(), text.size() - 2);
  EXPECT_GE(t.WorkingBytes(text), (text.size() + 1) * tok::kEncodeBytesPerInputByte);
}

TEST(Encode, AddsBosAndEosOnlyWhenAsked) {
  Vocab v;
  const TokenId bos = v.Add("<bos>", TokenKind::kControl);
  const TokenId eos = v.Add("<eos>", TokenKind::kControl);
  v.spec.bos = bos;
  v.spec.eos = eos;
  v.spec.add_bos = true;
  v.spec.add_eos = true;
  const tok::Tokenizer t = v.Build();
  tok::EncodeOptions o;
  EXPECT_EQ(Encode(t, "x", o), (std::vector<TokenId>{Byte('x')}));
  o.add_bos_eos = true;
  EXPECT_EQ(Encode(t, "x", o), (std::vector<TokenId>{bos, Byte('x'), eos}));
}

TEST(Decode, KindsDecodeAsDocumented) {
  Vocab v;
  const TokenId ctl = v.Add("<c>", TokenKind::kControl);
  const TokenId user = v.Add("<u>", TokenKind::kUserDefined);
  const TokenId unused = v.Add("[PAD9]", TokenKind::kUnused);
  const tok::Tokenizer t = v.Build();
  const std::vector<TokenId> ids = {Byte('a'), ctl, user, unused, Byte(' ')};
  std::string out;
  ASSERT_TRUE(t.Decode(ids, {}, out).has_value());
  EXPECT_EQ(out, "a<u> ");
  out.clear();
  ASSERT_TRUE(t.Decode(ids, {.control_tokens = true}, out).has_value());
  EXPECT_EQ(out, "a<c><u> ");
  const std::vector<TokenId> bad = {Byte('a'), static_cast<TokenId>(t.size())};
  EXPECT_EQ(Failed(t.Decode(bad, {}, out), &tok::Error::item), 1U);
  const std::vector<TokenId> negative = {-1};
  EXPECT_EQ(Failed(t.Decode(negative, {}, out), &tok::Error::rule), Rule::kInvalidToken);
  // The unused token's text is not matchable either.
  EXPECT_EQ(t.Find("[PAD9]"), std::nullopt);
}

TEST(Decode, StreamHoldsPartialSequences) {
  Vocab v;
  const tok::Tokenizer t = v.Build();
  tok::StreamDecoder d(t, {});
  std::string out;
  ASSERT_TRUE(d.Push(Byte('\xC3'), out).has_value());
  EXPECT_EQ(out, "");
  ASSERT_TRUE(d.Push(Byte('\xA9'), out).has_value());
  EXPECT_EQ(out, "é");
  ASSERT_TRUE(d.Push(Byte('\xF0'), out).has_value());
  ASSERT_TRUE(d.Push(Byte('\x9F'), out).has_value());
  EXPECT_EQ(out, "é");
  d.Finish(out);
  EXPECT_EQ(out, "é�");
  out.clear();
  ASSERT_TRUE(d.Push(Byte('\x80'), out).has_value());  // cannot complete: replaced at once
  EXPECT_EQ(out, "�");
}

TEST(Create, RefusesInconsistentVocabularies) {
  auto rule = [](const Vocab& v) {
    return Failed(tok::Tokenizer::Create(v.spec), &tok::Error::rule);
  };
  {
    Vocab v;
    v.spec.tokens.erase(v.spec.tokens.begin() + 'z');
    v.spec.kinds.pop_back();
    EXPECT_EQ(rule(v), Rule::kVocabulary);  // byte 'z' has no token
  }
  {
    Vocab v;
    v.spec.merges.emplace_back(Alphabet("x"), Alphabet("y"));  // "xy" is not a token
    EXPECT_EQ(rule(v), Rule::kVocabulary);
  }
  {
    Vocab v;
    v.Merge("a", "b");
    v.spec.merges.emplace_back(Alphabet("a"), Alphabet("b"));
    EXPECT_EQ(rule(v), Rule::kVocabulary);  // a repeated merge
  }
  {
    Vocab v;
    v.Add(ByteText('a'));
    EXPECT_EQ(rule(v), Rule::kVocabulary);  // one text, two IDs
  }
  {
    Vocab v;
    v.Add("é");  // U+00E9 is in the alphabet, as byte 0xE9; but "中" is not
    v.Add("中");
    EXPECT_EQ(rule(v), Rule::kVocabulary);
  }
  {
    Vocab v;
    v.Add(std::string(300, '<'), TokenKind::kControl);
    EXPECT_EQ(rule(v), Rule::kBounds);
  }
  {
    Vocab v;
    v.Add("", TokenKind::kUserDefined);
    EXPECT_EQ(rule(v), Rule::kVocabulary);
  }
  {
    Vocab v;
    v.spec.bos = 1000;
    EXPECT_EQ(rule(v), Rule::kVocabulary);
  }
  {
    Vocab v;
    v.spec.add_eos = true;
    EXPECT_EQ(rule(v), Rule::kVocabulary);
  }
  {
    Vocab v;
    v.spec.kinds.pop_back();
    EXPECT_EQ(rule(v), Rule::kVocabulary);
  }
  {
    Vocab v;
    v.Add("\xC3", TokenKind::kUserDefined);  // not UTF-8
    EXPECT_EQ(rule(v), Rule::kVocabulary);
  }
}

// A hostile vocabulary whose merge keys, (left << 32 | right), all fall in
// one bucket of an identity-hashed table of that size: creation stays fast
// (0.24 s on spark-b; quadratic, 62 s, before the keys were mixed).
TEST(Create, CollidingMergeKeysStayFast) {
  constexpr std::size_t kMerges = 200000;
  std::unordered_map<std::uint64_t, int> sizing;
  sizing.reserve(kMerges);
  const std::uint64_t buckets = sizing.bucket_count();
  Vocab v;
  auto text = [](std::size_t i) {  // five letters, distinct per i
    std::string s;
    for (int k = 0; k < 5; ++k, i /= 26) {
      s.push_back(static_cast<char>('a' + (i % 26)));
    }
    return s;
  };
  const std::size_t words = buckets + 1000;
  for (std::size_t i = 256; i < words; ++i) {
    v.Add(text(i));
  }
  std::size_t merges = 0;
  for (std::uint64_t left = 256; left < words && merges < kMerges; ++left) {
    const std::uint64_t right = (buckets - ((left << 32U) % buckets)) % buckets;
    if (right < 256 || right >= words) {
      continue;
    }
    v.spec.merges.emplace_back(v.spec.tokens[left], v.spec.tokens[right]);
    v.Add(v.spec.tokens[left] + v.spec.tokens[right]);
    ++merges;
  }
  ASSERT_GT(merges, kMerges / 2);
  const auto start = std::chrono::steady_clock::now();
  const auto t = tok::Tokenizer::Create(v.spec);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(t.has_value()) << t.error().ToString();
  EXPECT_LT(elapsed, std::chrono::seconds(20));
}

// --- GGUF -------------------------------------------------------------------------

class GgufWriter {
 public:
  GgufWriter& Header(std::uint64_t keys, std::uint32_t version = 3) {
    bytes_ = "GGUF";
    U(version, 4);
    U(0, 8);
    U(keys, 8);
    return *this;
  }
  void U(std::uint64_t v, int n) {
    for (int i = 0; i < n; ++i) {
      bytes_.push_back(static_cast<char>((v >> (8 * i)) & 0xFFU));
    }
  }
  void Str(std::string_view s) {
    U(s.size(), 8);
    bytes_ += s;
  }
  GgufWriter& String(std::string_view key, std::string_view value) {
    Str(key);
    U(8, 4);
    Str(value);
    return *this;
  }
  GgufWriter& Strings(std::string_view key, const std::vector<std::string>& values) {
    Str(key);
    U(9, 4);
    U(8, 4);
    U(values.size(), 8);
    for (const auto& v : values) {
      Str(v);
    }
    return *this;
  }
  GgufWriter& Ints(std::string_view key, const std::vector<std::int32_t>& values) {
    Str(key);
    U(9, 4);
    U(5, 4);
    U(values.size(), 8);
    for (const auto v : values) {
      U(static_cast<std::uint32_t>(v), 4);
    }
    return *this;
  }
  GgufWriter& Floats(std::string_view key, const std::vector<float>& values) {
    Str(key);
    U(9, 4);
    U(6, 4);
    U(values.size(), 8);
    for (const auto value : values) {
      U(std::bit_cast<std::uint32_t>(value), 4);
    }
    return *this;
  }
  GgufWriter& U32(std::string_view key, std::uint32_t value) {
    Str(key);
    U(4, 4);
    U(value, 4);
    return *this;
  }
  GgufWriter& Bool(std::string_view key, bool value) {
    Str(key);
    U(7, 4);
    U(value ? 1 : 0, 1);
    return *this;
  }
  std::span<const std::byte> Bytes() const { return std::as_bytes(std::span(bytes_)); }
  std::string& raw() { return bytes_; }

 private:
  std::string bytes_;
};

GgufWriter SmallGguf(std::string_view pre = "qwen35") {
  Vocab v;
  v.Merge("a", "b");
  const TokenId end = v.Add("<|end|>", TokenKind::kControl);
  v.Add("[PAD1]", TokenKind::kUnused);
  auto type = [](TokenKind k) -> std::int32_t {
    switch (k) {
      case TokenKind::kNormal:
        return 1;
      case TokenKind::kByte:
        return 6;
      case TokenKind::kControl:
        return 3;
      case TokenKind::kUserDefined:
        return 4;
      case TokenKind::kUnused:
        return 5;
    }
    return 0;
  };
  std::vector<std::int32_t> types;
  types.reserve(v.spec.kinds.size());
  for (const TokenKind k : v.spec.kinds) {
    types.push_back(type(k));
  }
  std::vector<std::string> merges;
  merges.reserve(v.spec.merges.size());
  for (const auto& [a, b] : v.spec.merges) {
    std::string m = a;
    m += ' ';
    m += b;
    merges.push_back(std::move(m));
  }
  GgufWriter w;
  w.Header(11);
  w.String("general.architecture", "test");
  w.U32("some.other.key", 7);
  w.String("tokenizer.ggml.model", "gpt2");
  w.String("tokenizer.ggml.pre", pre);
  w.Strings("tokenizer.ggml.tokens", v.spec.tokens);
  w.Ints("tokenizer.ggml.token_type", types);
  w.Strings("tokenizer.ggml.merges", merges);
  w.U32("tokenizer.ggml.eos_token_id", static_cast<std::uint32_t>(end));
  w.Bool("tokenizer.ggml.add_bos_token", false);
  w.String("tokenizer.chat_template", "{{ messages }}");
  w.Ints("general.nested.ints", {1, 2, 3});
  return w;
}

// Raw UTF-8 BPE uses code points, whitespace markers and byte fallback.
// Its merge paths can cross spaces, unlike the M3 byte-BPE windows.
tok::TokenizerSpec RawSpec() {
  tok::TokenizerSpec spec;
  spec.pre_tokenizer = PreTokenizer::kRawBpe;
  constexpr std::string_view hex = "0123456789ABCDEF";
  for (unsigned b = 0; b < 256; ++b) {
    std::string text = "<0x00>";
    text[3] = hex[b >> 4U];
    text[4] = hex[b & 15U];
    spec.tokens.push_back(text);
    spec.kinds.push_back(TokenKind::kByte);
  }
  for (const std::string_view text :
       {"a", "▁", "▁a", "a▁a", "\n", "\n\n", "\n\n\n", "é", "<bos>"}) {
    spec.tokens.emplace_back(text);
    spec.kinds.push_back(text == "<bos>" ? TokenKind::kControl : TokenKind::kNormal);
  }
  spec.merges = {{"▁", "a"}, {"a", "▁a"}, {"\n", "\n"}};
  spec.bos = 264;
  spec.add_bos = true;
  return spec;
}

TEST(RawBpe, MergesAcrossSpacesPreservesWhitespaceAndFallsBackByByte) {
  auto t = tok::Tokenizer::Create(RawSpec());
  ASSERT_TRUE(t.has_value()) << t.error().ToString();
  EXPECT_EQ(Encode(*t, "a a"), (std::vector<TokenId>{259}));
  EXPECT_EQ(Encode(*t, "é🙂"), (std::vector<TokenId>{263, 0xF0, 0x9F, 0x99, 0x82}));
  EXPECT_EQ(Encode(*t, "\n\n\n"), (std::vector<TokenId>{262}));
  EXPECT_EQ(Encode(*t, "\n\n\n\n"), (std::vector<TokenId>{261, 261}));
  const std::string text = " a a\r\n\t  é🙂";
  std::string decoded;
  ASSERT_TRUE(t->Decode(Encode(*t, text), {}, decoded).has_value());
  EXPECT_EQ(decoded, text);
  EXPECT_EQ(Encode(*t, "a", {.add_bos_eos = true}), (std::vector<TokenId>{264, 256}));
  EXPECT_EQ(Encode(*t, "<bos>", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{264}));
  EXPECT_NE(Encode(*t, "<bos>"), (std::vector<TokenId>{264}));
  EXPECT_EQ(Encode(*t, "▁"), (std::vector<TokenId>{257}));
  EXPECT_EQ(t->longest_token_bytes(), 5U);
}

TEST(RawBpe, BoundsAndAccountingIncludeTheWholeInput) {
  auto t = tok::Tokenizer::Create(RawSpec());
  ASSERT_TRUE(t.has_value());
  std::string text;
  for (std::size_t i = 0; i < (std::size_t{1} << 20U) / 2; ++i) {
    text += "a a\n";
  }
  EXPECT_EQ(t->WorkingBytes(text), (text.size() + 1) * tok::kEncodeBytesPerInputByte);
  llmp::base::WorkPulse pulse;
  llmp::base::SetThreadPulse(&pulse);
  const auto ids = Encode(*t, text);
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_GT(pulse.beats(), 2U);
  ASSERT_EQ(ids.size(), (std::size_t{1} << 20U));
  for (std::size_t i = 0; i < ids.size(); i += 2) {
    ASSERT_EQ(ids[i], 259);
    ASSERT_EQ(ids[i + 1], 260);
  }
  std::vector<TokenId> out;
  pulse.Cancel();
  llmp::base::SetThreadPulse(&pulse);
  auto cancelled = t->Encode("a a", {}, out);
  llmp::base::SetThreadPulse(nullptr);
  ASSERT_FALSE(cancelled.has_value());
  EXPECT_EQ(cancelled.error().rule, Rule::kCancelled);
  auto r = t->Encode("a a", {.max_bytes = 2}, out);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().rule, Rule::kInputTooLarge);
  r = t->Encode("🙂", {.max_tokens = 3}, out);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().rule, Rule::kOutputTooLarge);
  out.clear();
  const std::array spans = {tok::SpecialSpan{0, 6}};
  r = t->EncodeMarked("<0x41>", spans, {}, out);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().rule, Rule::kSpecialToken);
}

TEST(RawBpe, RejectsMissingOrMalformedFallbackAndUnsafeSpecOptions) {
  auto spec = RawSpec();
  spec.kinds[0] = TokenKind::kUnused;
  EXPECT_FALSE(tok::Tokenizer::Create(spec).has_value());
  spec = RawSpec();
  spec.tokens[0] = "<0xgg>";
  EXPECT_FALSE(tok::Tokenizer::Create(spec).has_value());
  spec = RawSpec();
  spec.normalization = tok::Normalization::kNfc;
  EXPECT_FALSE(tok::Tokenizer::Create(spec).has_value());
  spec = RawSpec();
  spec.kinds[257] = TokenKind::kUnused;
  EXPECT_FALSE(tok::Tokenizer::Create(spec).has_value());
}

TEST(RawBpe, SpecialOnlyTextHonorsCancellationAndReportsProgress) {
  auto spec = RawSpec();
  spec.tokens.emplace_back("<user>");
  spec.kinds.push_back(TokenKind::kUserDefined);
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t.has_value());
  llmp::base::WorkPulse pulse;
  pulse.Cancel();
  std::vector<TokenId> out;
  llmp::base::SetThreadPulse(&pulse);
  const auto parsed = t->Encode("<bos><user>", {.special = tok::SpecialTokens::kParse}, out);
  const std::array spans = {tok::SpecialSpan{0, 5}};
  const auto marked = t->EncodeMarked("<bos>", spans, {}, out);
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(Failed(parsed, &tok::Error::rule), Rule::kCancelled);
  EXPECT_EQ(Failed(marked, &tok::Error::rule), Rule::kCancelled);
  EXPECT_TRUE(out.empty());

  pulse.Clear();
  std::string text;
  for (std::size_t i = 0; i < 65536; ++i) {
    text += "<bos><user>";
  }
  const auto before = pulse.beats();
  out.clear();
  llmp::base::SetThreadPulse(&pulse);
  const auto progress = t->Encode(text, {.special = tok::SpecialTokens::kParse}, out);
  llmp::base::SetThreadPulse(nullptr);
  ASSERT_TRUE(progress.has_value());
  EXPECT_EQ(out.size(), 131072U);
  EXPECT_GT(pulse.beats() - before, 2U);
}

TEST(RawBpe, HostileOverlapsRespectRanksAndPieceBoundaries) {
  auto spec = RawSpec();
  auto add = [&](std::string_view text) {
    spec.tokens.emplace_back(text);
    spec.kinds.push_back(TokenKind::kNormal);
    return static_cast<TokenId>(spec.tokens.size() - 1);
  };
  const auto aa = add("aa");
  const auto aaa = add("aaa");
  add("aaaa");  // A whole token without a merge path must not be greedy.
  add("a\n");   // A merge cannot escape the newline piece boundary.
  spec.merges = {{"aa", "a"}, {"a", "aa"}, {"a", "a"}, {"a", "\n"}, {"\n", "\n"}};
  spec.kinds[262] = TokenKind::kControl;  // The newline shortcut must check kind.
  auto t = tok::Tokenizer::Create(spec);
  ASSERT_TRUE(t.has_value());
  EXPECT_EQ(Encode(*t, "aaaa"), (std::vector<TokenId>{aaa, 256}));
  EXPECT_EQ(Encode(*t, "aaaaa"), (std::vector<TokenId>{aaa, aa}));
  EXPECT_EQ(Encode(*t, "a\n\n\n"), (std::vector<TokenId>{256, 261, 260}));
  EXPECT_EQ(Encode(*t, "\n\n\n", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{262}));
  spec.merges.emplace_back("a", "a");
  EXPECT_EQ(Failed(tok::Tokenizer::Create(spec), &tok::Error::rule), Rule::kVocabulary);
  spec = RawSpec();
  spec.tokens.emplace_back("<0x41>a");
  spec.kinds.push_back(TokenKind::kNormal);
  spec.merges.emplace_back("<0x41>", "a");
  EXPECT_EQ(Failed(tok::Tokenizer::Create(spec), &tok::Error::rule), Rule::kVocabulary);
}

TEST(RawBpe, AllFallbackBytesDecodeExactlyAndStreamAcrossTokenBoundaries) {
  auto t = tok::Tokenizer::Create(RawSpec());
  ASSERT_TRUE(t.has_value());
  std::vector<TokenId> ids;
  std::string expected;
  for (unsigned b = 0; b < 256; ++b) {
    ids.push_back(static_cast<TokenId>(b));
    expected.push_back(static_cast<char>(b));
  }
  std::string decoded;
  ASSERT_TRUE(t->Decode(ids, {}, decoded).has_value());
  EXPECT_EQ(decoded, expected);
  tok::StreamDecoder stream(*t, {});
  std::string text;
  for (const TokenId id : {0xE2, 0x82}) {
    ASSERT_TRUE(stream.Push(id, text).has_value());
    EXPECT_TRUE(text.empty());
  }
  ASSERT_TRUE(stream.Push(0xAC, text).has_value());
  EXPECT_EQ(text, "€");
  for (const TokenId id : {0xE2, 0x28, 0xA1, 0xF0, 0x9F}) {
    ASSERT_TRUE(stream.Push(id, text).has_value());
  }
  EXPECT_EQ(text, "€�(�");
  stream.Finish(text);
  EXPECT_EQ(text, "€�(��");
  EXPECT_EQ(Encode(*t, "<0x41>", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{'<', '0', 'x', '4', '1', '>'}));
}

TEST(RawBpe, RefusesIndexOverflowWithoutReadingTheInput) {
  auto t = tok::Tokenizer::Create(RawSpec());
  ASSERT_TRUE(t.has_value());
  // The view is intentionally not backed by that many bytes: refusal must
  // precede validation, scanning or allocation.
  const std::size_t too_large = std::size_t{1} << 32U;
  const std::string_view hostile("a", too_large);
  std::vector<TokenId> out = {123};
  const auto result = t->Encode(hostile, {.max_bytes = too_large, .max_tokens = too_large}, out);
  EXPECT_EQ(Failed(result, &tok::Error::rule), Rule::kInputTooLarge);
  EXPECT_EQ(out, (std::vector<TokenId>{123}));
  // A literal marker occupies three input bytes but still fits one token.
  EXPECT_EQ(Encode(*t, "▁", {.max_tokens = 1}), (std::vector<TokenId>{257}));
}

GgufWriter RawGguf(std::string_view model = "gemma4", std::string_view pre = "",
                   bool prefix = false) {
  const auto spec = RawSpec();
  GgufWriter w;
  w.Header(pre.empty() ? 7 : 8);
  w.String("tokenizer.ggml.model", model);
  if (!pre.empty()) {
    w.String("tokenizer.ggml.pre", pre);
  }
  w.Strings("tokenizer.ggml.tokens", spec.tokens);
  std::vector<std::int32_t> kinds(256, 6);
  kinds.insert(kinds.end(), 8, 1);
  kinds.push_back(3);
  w.Ints("tokenizer.ggml.token_type", kinds);
  w.Strings("tokenizer.ggml.merges", {"▁ a", "a ▁a", "\n \n"});
  w.U32("tokenizer.ggml.bos_token_id", 264);
  w.Bool("tokenizer.ggml.add_bos_token", true);
  w.Bool("tokenizer.ggml.add_space_prefix", prefix);
  return w;
}

TEST(Gguf, RawUtf8BpeDefaultsAndStrictRefusals) {
  for (const std::string_view pre : {"", "gemma4"}) {
    const auto g = tok::ReadGgufTokenizer(RawGguf("gemma4", pre).Bytes());
    ASSERT_TRUE(g.has_value()) << g.error().ToString();
    EXPECT_EQ(g->pre, "gemma4");
    EXPECT_EQ(g->spec.pre_tokenizer, PreTokenizer::kRawBpe);
    auto t = tok::Tokenizer::Create(g->spec);
    ASSERT_TRUE(t.has_value()) << t.error().ToString();
    EXPECT_EQ(Encode(*t, "a a"), (std::vector<TokenId>{259}));
  }
  EXPECT_FALSE(tok::ReadGgufTokenizer(RawGguf("gemma4", "qwen2").Bytes()).has_value());
  EXPECT_FALSE(tok::ReadGgufTokenizer(RawGguf("gemma4", "", true).Bytes()).has_value());
  EXPECT_FALSE(tok::ReadGgufTokenizer(RawGguf("gpt2", "qwen2").Bytes()).has_value());
  const auto good = RawGguf();
  for (std::size_t n = 0; n < good.Bytes().size(); ++n) {
    EXPECT_FALSE(tok::ReadGgufTokenizer(good.Bytes().first(n)).has_value()) << n;
  }
}

// --- classic score-ordered SentencePiece ------------------------------------------

tok::TokenizerSpec SpmSpec(bool prefix = false) {
  auto spec = RawSpec();
  spec.pre_tokenizer = PreTokenizer::kSentencePiece;
  spec.merges.clear();
  spec.scores.assign(spec.tokens.size(), -10.0F);
  spec.add_space_prefix = prefix;
  return spec;
}

TokenId SpmAdd(tok::TokenizerSpec& spec, std::string text, float score,
               TokenKind kind = TokenKind::kNormal) {
  const auto id = static_cast<TokenId>(spec.tokens.size());
  spec.tokens.push_back(std::move(text));
  spec.kinds.push_back(kind);
  spec.scores.push_back(score);
  return id;
}

std::string ProtoVarint(std::uint64_t value) {
  std::string bytes;
  do {
    const auto low = value & 127U;
    value >>= 7U;
    bytes.push_back(static_cast<char>(low | (value != 0 ? 128U : 0U)));
  } while (value != 0);
  return bytes;
}

std::string ProtoScalar(unsigned field, std::uint64_t value) {
  return ProtoVarint(std::uint64_t{field} << 3U) + ProtoVarint(value);
}

std::string ProtoBytes(unsigned field, std::string_view bytes) {
  return ProtoVarint((std::uint64_t{field} << 3U) | 2U) + ProtoVarint(bytes.size()) +
         std::string(bytes);
}

std::string ProtoPiece(std::string_view text, unsigned type, float score = 0) {
  std::string piece = ProtoBytes(1, text);
  piece += ProtoVarint((2U << 3U) | 5U);
  const auto bits = std::bit_cast<std::uint32_t>(score);
  for (unsigned b = 0; b < 4; ++b) piece.push_back(static_cast<char>(bits >> (8U * b)));
  return ProtoBytes(1, piece + ProtoScalar(3, type));
}

std::string SentencePieceModel(std::string_view normalizer = {}, std::string_view extra = {},
                               std::string_view trainer_extra = {}) {
  auto spec = SpmSpec();
  const auto unknown = SpmAdd(spec, "<unk>", 0, TokenKind::kControl);
  const auto eos = SpmAdd(spec, "<eos>", 0, TokenKind::kControl);
  std::string model;
  for (std::size_t i = 0; i < spec.tokens.size(); ++i) {
    unsigned type = spec.kinds[i] == TokenKind::kNormal ? 1
                    : spec.kinds[i] == TokenKind::kByte ? 6
                                                        : 3;
    if (i == static_cast<std::size_t>(unknown)) type = 2;
    model += ProtoPiece(spec.tokens[i], type, spec.scores[i]);
  }
  model += ProtoBytes(3, normalizer.empty() ? ProtoScalar(3, 0) + ProtoScalar(4, 0) : normalizer);
  model += ProtoBytes(2, ProtoScalar(3, 2) + ProtoScalar(35, 1) +
                             ProtoScalar(40, static_cast<std::uint64_t>(unknown)) +
                             ProtoScalar(41, static_cast<std::uint64_t>(*spec.bos)) +
                             ProtoScalar(42, static_cast<std::uint64_t>(eos)) +
                             std::string(trainer_extra));
  return model + std::string(extra);
}

TEST(SentencePieceModel, NativeReaderUsesScoresIdentityAndByteFallback) {
  auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel());
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  EXPECT_FALSE(spec->add_space_prefix);
  EXPECT_FALSE(spec->remove_extra_whitespaces);
  EXPECT_FALSE(spec->add_bos);
  EXPECT_FALSE(spec->add_eos);
  EXPECT_EQ(spec->bos, 264);
  EXPECT_EQ(spec->eos, 266);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "a a"), (std::vector<TokenId>{259}));
  EXPECT_EQ(Encode(*tokenizer, "猫"), (std::vector<TokenId>{0xE7, 0x8C, 0xAB}));
}

TEST(SentencePieceModel, IdentityWhitespaceOptionsTrimCollapseAndPrefix) {
  auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel(ProtoBytes(1, "identity")));
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  EXPECT_TRUE(spec->add_space_prefix);
  EXPECT_TRUE(spec->remove_extra_whitespaces);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "  a   a  "), Encode(*tokenizer, "a a"));
  EXPECT_TRUE(Encode(*tokenizer, "   ").empty());
  EXPECT_NE(Encode(*tokenizer, "\ta"), Encode(*tokenizer, "a"));
  tok::EncodeOptions options;
  options.max_tokens = 1;
  options.max_bytes = 1000;
  std::vector<TokenId> ids;
  EXPECT_TRUE(tokenizer->Encode(std::string(1000, ' '), options, ids));
  EXPECT_TRUE(ids.empty());
}

TEST(SentencePieceModel, NativeSymbolsDoNotRestartNormalizationOfTheSurroundingText) {
  const auto symbol = ProtoPiece("<symbol>", 4);
  for (const bool prefix : {false, true}) {
    for (const bool trim : {false, true}) {
      SCOPED_TRACE(std::to_string(prefix) + "/" + std::to_string(trim));
      auto spec = tok::ReadSentencePieceTokenizer(
          SentencePieceModel(ProtoScalar(3, prefix) + ProtoScalar(4, trim), symbol));
      ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
      auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
      ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
      const auto first = prefix ? 258 : 256;
      EXPECT_EQ(Encode(*tokenizer, "a<symbol>a"), (std::vector<TokenId>{first, 267, 256}));
      EXPECT_EQ(Encode(*tokenizer, "a <symbol> a"), (std::vector<TokenId>{first, 257, 267, 258}));
      EXPECT_EQ(Encode(*tokenizer, "<symbol>a"),
                prefix ? (std::vector<TokenId>{257, 267, 256}) : (std::vector<TokenId>{267, 256}));
      if (trim)
        EXPECT_EQ(Encode(*tokenizer, "  a  <symbol>  a  "),
                  (std::vector<TokenId>{first, 257, 267, 258}));
    }
  }
}

TEST(SentencePieceModel, NativeSymbolsProtectTheirOwnWhitespaceDuringNormalization) {
  auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel(
      ProtoScalar(3, 0) + ProtoScalar(4, 1), ProtoPiece("x  y", 4) + ProtoPiece("x▁▁y", 4)));
  ASSERT_TRUE(spec);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  // The raw native symbol protects its internal spaces, but the model then
  // matches its exact marker spelling. Never invent a normalized alias for 267.
  EXPECT_EQ(Encode(*tokenizer, "a x  y a"), (std::vector<TokenId>{256, 257, 268, 258}));
  // Native normalization removes a trailing marker, including a literal
  // marker in input, once at the end of the complete region.
  EXPECT_EQ(Encode(*tokenizer, "a▁"), (std::vector<TokenId>{256}));
}

TEST(SentencePieceModel, ConfigAddedTokensKeepTheirDeclaredRawExtractionPhase) {
  auto spec =
      tok::ReadSentencePieceTokenizer(SentencePieceModel(ProtoScalar(3, 1) + ProtoScalar(4, 1)));
  ASSERT_TRUE(spec);
  ASSERT_TRUE(tok::ApplyHfTokenizerConfig(
      R"({"added_tokens_decoder":{"267":{"content":"<added>","special":false,"normalized":false}}})",
      *spec));
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "a<added>a"), (std::vector<TokenId>{258, 267, 258}));
}

TEST(SentencePieceModel, NormalizedConfigControlsHonorTheParsingModeAfterNormalization) {
  auto spec =
      tok::ReadSentencePieceTokenizer(SentencePieceModel(ProtoScalar(3, 1) + ProtoScalar(4, 1)));
  ASSERT_TRUE(spec);
  ASSERT_TRUE(tok::ApplyHfTokenizerConfig(
      R"({"added_tokens_decoder":{"267":{"content":"<control>","special":true,"normalized":true}}})",
      *spec));
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "a<control>a", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{258, 267, 256}));
  const auto plain = Encode(*tokenizer, "a<control>a");
  EXPECT_EQ(std::ranges::find(plain, 267), plain.end());
}

TEST(SentencePieceModel, MarkedEncodingDoesNotParseUnmarkedNormalizedControls) {
  auto spec =
      tok::ReadSentencePieceTokenizer(SentencePieceModel(ProtoScalar(3, 1) + ProtoScalar(4, 1)));
  ASSERT_TRUE(spec);
  ASSERT_TRUE(tok::ApplyHfTokenizerConfig(
      R"({"added_tokens_decoder":{"267":{"content":"<control>","special":true,"normalized":true}}})",
      *spec));
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  CheckMarkedNormalizedControl(*tokenizer, "<control>", 267);
}

TEST(SentencePieceModel, UnusedPiecesCanMergeIntoLivePiecesAndResegmentAtEmission) {
  const auto pieces =
      ProtoPiece("b", 1) + ProtoPiece("c", 1) + ProtoPiece("ab", 5, 10) + ProtoPiece("abc", 1, 20);
  auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel({}, pieces));
  ASSERT_TRUE(spec);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "abc"), (std::vector<TokenId>{270}));
  EXPECT_EQ(Encode(*tokenizer, "ab"), (std::vector<TokenId>{256, 267}));
  EXPECT_EQ(tokenizer->WorkingBytes("abc"), 4U * 128U + tokenizer->size() * sizeof(std::uint32_t));
}

TEST(SentencePieceModel, AnUnmergedUnusedSingletonKeepsItsNativeIdAndDecodedText) {
  auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel({}, ProtoPiece("b", 5)));
  ASSERT_TRUE(spec);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  const auto ids = Encode(*tokenizer, "b");
  EXPECT_EQ(ids, (std::vector<TokenId>{267}));
  std::string decoded;
  ASSERT_TRUE(tokenizer->Decode(ids, {}, decoded));
  EXPECT_EQ(decoded, "b");
}

TEST(SentencePieceModel, InconsistentModelAndMatcherMetadataRefusesBeforeConstruction) {
  auto parsed = tok::ReadSentencePieceTokenizer(SentencePieceModel());
  ASSERT_TRUE(parsed);
  for (const unsigned field : {0U, 1U, 2U, 3U}) {
    auto malformed = *parsed;
    auto& flags = field == 0   ? malformed.model_tokens
                  : field == 1 ? malformed.model_symbols
                  : field == 2 ? malformed.added_tokens
                               : malformed.normalized_tokens;
    flags.pop_back();
    EXPECT_FALSE(tok::Tokenizer::ConstructionBytes(malformed));
    EXPECT_FALSE(tok::Tokenizer::Create(std::move(malformed)));
  }
}

TEST(SentencePieceModel, DeepUnusedChainsResegmentIterativelyWithinOutputAndWorkBounds) {
  std::string pieces;
  for (std::size_t size = 2; size <= tok::kMaxTokenBytes; ++size)
    pieces += ProtoPiece(std::string(size, 'a'), 5);
  auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel({}, pieces));
  ASSERT_TRUE(spec);
  const auto cost = tok::Tokenizer::ConstructionBytes(*spec);
  ASSERT_TRUE(cost);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  const std::string text(128 * tok::kMaxTokenBytes + 1, 'a');
  EXPECT_EQ(tokenizer->WorkingBytes(text),
            (text.size() + 1) * 128U + tokenizer->size() * sizeof(std::uint32_t));
  llmp::base::WorkPulse pulse;
  llmp::base::SetThreadPulse(&pulse);
  std::vector<TokenId> ids;
  const auto encoded = tokenizer->Encode(text, {.max_tokens = text.size()}, ids);
  llmp::base::SetThreadPulse(nullptr);
  ASSERT_TRUE(encoded) << Failed(encoded, &tok::Error::ToString).value_or("");
  EXPECT_GT(pulse.beats(), 10U);
  ASSERT_EQ(ids.size(), text.size());
  EXPECT_TRUE(std::ranges::all_of(ids, [](TokenId id) { return id == 256; }));
  ids.assign(1, 264);
  const auto limited = tokenizer->Encode(std::string_view(text).substr(0, tok::kMaxTokenBytes),
                                         {.max_tokens = 3}, ids);
  EXPECT_EQ(Failed(limited, &tok::Error::rule), Rule::kOutputTooLarge);
  EXPECT_EQ(ids, (std::vector<TokenId>{264, 256, 256, 256}));
  pulse.Cancel();
  llmp::base::SetThreadPulse(&pulse);
  const auto cancelled = tokenizer->Encode(text, {}, ids);
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(Failed(cancelled, &tok::Error::rule), Rule::kCancelled);
}

TEST(SentencePieceModel, WireErrorsAndUnsupportedNormalizationRefuseExplicitly) {
  EXPECT_FALSE(tok::ReadSentencePieceTokenizer(""));
  for (const auto& bad : {std::string("\x00", 1), std::string("\x0F", 1),
                          std::string("\x0A\xFF", 2), std::string(12, '\xFF')}) {
    EXPECT_FALSE(tok::ReadSentencePieceTokenizer(bad));
  }
  EXPECT_EQ(Failed(tok::ReadSentencePieceTokenizer(SentencePieceModel(ProtoBytes(2, "map"))),
                   &tok::Error::rule),
            Rule::kUnsupported);
  EXPECT_EQ(Failed(tok::ReadSentencePieceTokenizer(SentencePieceModel(ProtoScalar(5, 0))),
                   &tok::Error::rule),
            Rule::kUnsupported);
  EXPECT_EQ(Failed(tok::ReadSentencePieceTokenizer(SentencePieceModel(ProtoScalar(3, 2))),
                   &tok::Error::rule),
            Rule::kFormat);
  EXPECT_EQ(Failed(tok::ReadSentencePieceTokenizer(
                       SentencePieceModel(ProtoScalar(3, 0) + ProtoScalar(3, 0))),
                   &tok::Error::rule),
            Rule::kFormat);
  EXPECT_FALSE(tok::ReadSentencePieceTokenizer(SentencePieceModel({}, ProtoBytes(2, ""))));
  EXPECT_FALSE(tok::ReadSentencePieceTokenizer(SentencePieceModel({}, ProtoBytes(5, "map"))));
  EXPECT_FALSE(tok::ReadSentencePieceTokenizer(SentencePieceModel({}, ProtoBytes(6, ""))));
}

TEST(SentencePieceModel, HfConfigAddsSpecialMetadataAndControlsInsertion) {
  auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel());
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  const auto applied = tok::ApplyHfTokenizerConfig(
      R"({"bos_token":{"content":"<bos>","lstrip":false},"eos_token":"<eos>",)"
      R"("add_bos_token":true,"add_eos_token":true,"added_tokens_decoder":{)"
      R"("267":{"content":"<turn>","special":false,"normalized":false}}})",
      *spec);
  ASSERT_TRUE(applied) << Failed(applied, &tok::Error::ToString).value_or("");
  EXPECT_EQ(spec->kinds[267], TokenKind::kUserDefined);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "a<turn>"), (std::vector<TokenId>{256, 267}));
  EXPECT_EQ(Encode(*tokenizer, "a<turn>", {.add_bos_eos = true}),
            (std::vector<TokenId>{264, 256, 267, 266}));
}

TEST(SentencePieceModel, ExtraSpecialListsAndMapsPreserveRstripMatching) {
  for (const std::string_view extras : {R"([{"content":"<bos>","rstrip":true}])",
                                        R"({"named":{"content":"<bos>","rstrip":true}})"}) {
    auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel());
    ASSERT_TRUE(spec);
    ASSERT_TRUE(tok::ApplyHfTokenizerConfig(
        R"({"extra_special_tokens":)" + std::string(extras) + "}", *spec));
    ASSERT_TRUE(spec->rstrip[264]);
    auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
    ASSERT_TRUE(tokenizer);
    EXPECT_EQ(Encode(*tokenizer, "<bos>  a", {.special = tok::SpecialTokens::kParse}),
              (std::vector<TokenId>{264, 256}));
  }
}

TEST(SentencePieceModel, HfConfigRefusesUnrepresentableOrInconsistentSpecialMetadata) {
  for (const std::string_view json :
       {R"({"bos_token":"missing"})", R"({"add_bos_token":"true"})",
        R"({"sp_model_kwargs":{"enable_sampling":true}})",
        R"({"added_tokens_decoder":{"267":{"content":"<x>","special":true,"lstrip":true}}})",
        R"({"added_tokens_decoder":{"267":{"content":"<x>","special":true,"single_word":true}}})",
        R"({"added_tokens_decoder":{"268":{"content":"<x>","special":true}}})",
        R"({"added_tokens_decoder":{"264":{"content":"changed","special":true}}})"}) {
    auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel());
    ASSERT_TRUE(spec);
    EXPECT_FALSE(tok::ApplyHfTokenizerConfig(json, *spec)) << json;
  }
}

TEST(SentencePieceModel, RandomCorruptionsStayWithinTheWireInput) {
  const auto model = SentencePieceModel();
  std::mt19937 random(0x53A1);
  for (unsigned i = 0; i < 250; ++i) {
    std::string bad = model;
    bad[random() % bad.size()] ^= static_cast<char>(1U << (random() % 8));
    const auto spec = tok::ReadSentencePieceTokenizer(bad);
    if (spec) (void)tok::Tokenizer::Create(*spec);
  }
}

TEST(SplitPlan, EmptyPlanKeepsWholeNonemptyFragment) {
  const auto plan = tok::SplitPlan::Compile({});
  ASSERT_TRUE(plan);
  auto budget = plan->Budget(10);
  std::vector<std::uint32_t> lengths;
  ASSERT_TRUE(plan->Split({}, lengths, budget));
  EXPECT_TRUE(lengths.empty());
  const std::u32string_view fragment = U"a b";
  ASSERT_TRUE(plan->Split(fragment, lengths, budget));
  EXPECT_EQ(lengths, (std::vector<std::uint32_t>{3}));
}

TEST(SentencePiece, GreatestScoreWinsAndEqualScoresChooseLeftmost) {
  for (const float score : {1.0F, 2.0F}) {
    auto spec = SpmSpec();
    SpmAdd(spec, "b", 0);
    SpmAdd(spec, "c", 0);
    const auto ab = SpmAdd(spec, "ab", 1);
    const auto bc = SpmAdd(spec, "bc", score);
    auto t = tok::Tokenizer::Create(std::move(spec));
    ASSERT_TRUE(t);
    EXPECT_EQ(Encode(*t, "abc"), score == 1 ? (std::vector<TokenId>{ab, *t->Find("c")})
                                            : (std::vector<TokenId>{*t->Find("a"), bc}));
  }
}

TEST(SentencePiece, LinkedHeapMatchesIndependentGreedyReductionOnOverlappingPairs) {
  auto spec = SpmSpec();
  SpmAdd(spec, "b", 0);
  for (const auto text :
       {"aa", "ab", "ba", "bb", "aaa", "aab", "aba", "abb", "baa", "bab", "bba", "bbb"}) {
    SpmAdd(spec, text, static_cast<float>(std::string_view(text).back() == 'a' ? 1 : 2));
  }
  auto tokenizer = tok::Tokenizer::Create(spec);
  ASSERT_TRUE(tokenizer);
  std::mt19937 random(3551);
  for (unsigned trial = 0; trial < 300; ++trial) {
    std::string text;
    std::vector<std::string> pieces;
    for (unsigned i = 0; i < 40; ++i) {
      const char c = (random() % 3) == 0 ? ' ' : (random() % 2) == 0 ? 'a' : 'b';
      text += c;
      pieces.emplace_back(c == ' ' ? "▁" : std::string(1, c));
    }
    // Deliberately slow oracle: scan every adjacent concatenation afresh
    // and reduce one maximum-score pair; no heap, links or stale entries.
    for (;;) {
      std::optional<std::size_t> best;
      float best_score = 0;
      for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
        const auto joined = pieces[i] + pieces[i + 1];
        const auto found = std::ranges::find(spec.tokens, joined);
        if (found != spec.tokens.end()) {
          const auto id = static_cast<std::size_t>(found - spec.tokens.begin());
          if (!best || spec.scores[id] > best_score) {
            best = i;
            best_score = spec.scores[id];
          }
        }
      }
      if (!best) {
        break;
      }
      pieces[*best] += pieces[*best + 1];
      pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(*best + 1));
    }
    std::vector<TokenId> expected;
    for (const auto& piece : pieces) {
      const auto found = std::ranges::find(spec.tokens, piece);
      ASSERT_NE(found, spec.tokens.end());
      expected.push_back(static_cast<TokenId>(found - spec.tokens.begin()));
    }
    EXPECT_EQ(Encode(*tokenizer, text), expected) << text;
  }
}

TEST(SentencePiece, MissingCodePointsRemainMergeableUntilByteFallback) {
  auto spec = SpmSpec();
  const auto joined = SpmAdd(spec, "中😀", 5);
  const auto longer = SpmAdd(spec, "中😀a", 6);
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  EXPECT_EQ(Encode(*t, "中😀"), (std::vector<TokenId>{joined}));
  EXPECT_EQ(Encode(*t, "中😀a"), (std::vector<TokenId>{longer}));
  EXPECT_EQ(Encode(*t, "中"), (std::vector<TokenId>{0xE4, 0xB8, 0xAD}));
}

TEST(SentencePiece, PlainTextCannotMergeOrEmitControls) {
  auto spec = SpmSpec();
  const auto joined = SpmAdd(spec, "ab", 100, TokenKind::kControl);
  const auto single = SpmAdd(spec, "中", 100, TokenKind::kControl);
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  EXPECT_EQ(Encode(*t, "ab中"), (std::vector<TokenId>{256, 'b', 0xE4, 0xB8, 0xAD}));
  EXPECT_EQ(Encode(*t, "ab中", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{joined, single}));
  std::vector<TokenId> marked;
  ASSERT_TRUE(t->EncodeMarked("ab中", {}, {.special = tok::SpecialTokens::kParse}, marked));
  EXPECT_EQ(marked, (std::vector<TokenId>{256, 'b', 0xE4, 0xB8, 0xAD}));
  marked.clear();
  const std::array spans = {tok::SpecialSpan{0, 2}};
  ASSERT_TRUE(t->EncodeMarked("ab中", spans, {}, marked));
  EXPECT_EQ(marked, (std::vector<TokenId>{joined, 0xE4, 0xB8, 0xAD}));
}

TEST(SentencePiece, LiteralByteTokenMarkupStaysText) {
  auto spec = SpmSpec();
  for (const auto prefix : {"<0", "<0x", "<0x6", "<0x61"}) {
    SpmAdd(spec, prefix, 1);
  }
  spec.scores[0x61] = 100;
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  const auto ids = Encode(*t, "<0x61>");
  EXPECT_EQ(std::ranges::find(ids, 0x61), ids.end());
  std::string decoded;
  ASSERT_TRUE(t->Decode(ids, {}, decoded));
  EXPECT_EQ(decoded, "<0x61>");
}

TEST(SentencePiece, DummyPrefixAppliesAfterControlsAndUserDefinedTokens) {
  auto spec = SpmSpec(true);
  const auto added = SpmAdd(spec, "<added>", 0, TokenKind::kUserDefined);
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  EXPECT_EQ(Encode(*t, ""), (std::vector<TokenId>{}));
  EXPECT_EQ(Encode(*t, "a"), (std::vector<TokenId>{258}));
  EXPECT_EQ(Encode(*t, "a<added>a"), (std::vector<TokenId>{258, added, 258}));
  EXPECT_EQ(Encode(*t, "a<bos>a", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{258, 264, 258}));
  std::vector<TokenId> marked;
  const std::array spans = {tok::SpecialSpan{1, 5}};
  ASSERT_TRUE(t->EncodeMarked("a<bos>a", spans, {}, marked));
  EXPECT_EQ(marked, (std::vector<TokenId>{258, 264, 258}));
  EXPECT_EQ(Encode(*t, "", {.add_bos_eos = true}), (std::vector<TokenId>{264}));
}

TEST(SentencePiece, CrossSpaceMergesKeepWhitespaceWithoutNfc) {
  auto spec = SpmSpec();
  spec.scores[258] = 1;
  spec.scores[259] = 2;
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  EXPECT_EQ(Encode(*t, "a a"), (std::vector<TokenId>{259}));
  EXPECT_EQ(Encode(*t, "a  a"), (std::vector<TokenId>{256, 257, 258}));
  EXPECT_EQ(Encode(*t, "é"), (std::vector<TokenId>{'e', 0xCC, 0x81}));
  EXPECT_EQ(Encode(*t, "é"), (std::vector<TokenId>{263}));
}

TEST(SentencePiece, SpecialStrippingIsBoundedAndDoesNotTriggerFalseTokenRefusal) {
  auto spec = SpmSpec(true);
  spec.rstrip.assign(spec.tokens.size(), false);
  spec.rstrip[264] = true;
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  const std::string text = "<bos>" + std::string(70000, ' ') + "a";
  EXPECT_EQ(Encode(*t, text, {.special = tok::SpecialTokens::kParse, .max_tokens = 2}),
            (std::vector<TokenId>{264, 258}));
  EXPECT_EQ(Encode(*t, "<bos>\t\r\na", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{264, 258}));
  std::vector<TokenId> marked;
  const std::array spans = {tok::SpecialSpan{0, 5}};
  ASSERT_TRUE(t->EncodeMarked(text, spans, {.max_tokens = 2}, marked));
  EXPECT_EQ(marked, (std::vector<TokenId>{264, 258}));
}

TEST(SentencePiece, DecodingPiecesPreservesGeneratedLeadingSpaces) {
  auto t = tok::Tokenizer::Create(SpmSpec(true));
  ASSERT_TRUE(t);
  const auto ids = Encode(*t, "a a");
  std::string pieces;
  ASSERT_TRUE(t->Decode(ids, {}, pieces));
  EXPECT_EQ(pieces, " a a");
  std::string text;
  ASSERT_TRUE(t->Decode(ids, {.remove_space_prefix = true}, text));
  EXPECT_EQ(text, "a a");
  tok::StreamDecoder generated(*t, {});
  std::string output;
  ASSERT_TRUE(generated.Push(258, output));
  generated.Finish(output);
  EXPECT_EQ(output, " a");
  tok::StreamDecoder full(*t, {.remove_space_prefix = true});
  output.clear();
  for (const auto id : ids) {
    ASSERT_TRUE(full.Push(id, output));
  }
  full.Finish(output);
  EXPECT_EQ(output, "a a");
  const auto with_bos = Encode(*t, "a", {.add_bos_eos = true});
  text.clear();
  ASSERT_TRUE(t->Decode(with_bos, {.remove_space_prefix = true}, text));
  EXPECT_EQ(text, " a");
  tok::StreamDecoder bos_first(*t, {.remove_space_prefix = true});
  output.clear();
  for (const auto id : with_bos) {
    ASSERT_TRUE(bos_first.Push(id, output));
  }
  bos_first.Finish(output);
  EXPECT_EQ(output, " a");
}

TEST(SentencePiece, LongWholeFragmentUsesFundedBoundAndReportsProgress) {
  auto spec = SpmSpec();
  spec.kinds[258] = TokenKind::kUnused;
  spec.kinds[259] = TokenKind::kUnused;
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  std::string text;
  for (std::size_t i = 0; i < (std::size_t{1} << 20U); ++i) {
    text += "a ";
  }
  EXPECT_EQ(t->WorkingBytes(text), (text.size() + 1) * tok::kEncodeBytesPerInputByte);
  llmp::base::WorkPulse pulse;
  llmp::base::SetThreadPulse(&pulse);
  std::vector<TokenId> ids;
  const auto encoded = t->Encode(text, {}, ids);
  llmp::base::SetThreadPulse(nullptr);
  ASSERT_TRUE(encoded);
  EXPECT_GT(pulse.beats(), 40U);
  EXPECT_EQ(ids.size(), text.size());
  pulse.Cancel();
  llmp::base::SetThreadPulse(&pulse);
  const auto refused = t->Encode("a", {}, ids);
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(Failed(refused, &tok::Error::rule), Rule::kCancelled);
}

TEST(SentencePiece, DenseEqualScoreOverlapsRespectMaximumLengthAndAppendBounds) {
  auto spec = SpmSpec();
  TokenId full = -1;
  for (std::size_t length = 2; length <= tok::kMaxTokenBytes; ++length) {
    full = SpmAdd(spec, std::string(length, 'a'), 0);
  }
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  // Equal scores repeatedly extend the leftmost piece. Seeds belonging to
  // its absorbed symbols remain stale in the heap until they are popped.
  const std::string text(128 * tok::kMaxTokenBytes + 1, 'a');
  std::vector<TokenId> ids{264};
  ASSERT_TRUE(t->Encode(text, {.max_tokens = 129}, ids));
  ASSERT_EQ(ids.size(), 130U);
  EXPECT_EQ(ids.front(), 264);
  EXPECT_EQ(ids.back(), 256);
  EXPECT_TRUE(
      std::all_of(ids.begin() + 1, ids.end() - 1, [full](TokenId id) { return id == full; }));

  // Automatic BOS uses one of the appended-token slots. The final one-byte
  // piece then refuses, retaining both prior output and completed pieces.
  ids.assign(1, 264);
  const auto refused = t->Encode(text, {.add_bos_eos = true, .max_tokens = 129}, ids);
  EXPECT_EQ(Failed(refused, &tok::Error::rule), Rule::kOutputTooLarge);
  ASSERT_EQ(ids.size(), 130U);
  EXPECT_EQ(ids[0], 264);
  EXPECT_EQ(ids[1], 264);
  EXPECT_TRUE(std::all_of(ids.begin() + 2, ids.end(), [full](TokenId id) { return id == full; }));
}

TEST(SentencePiece, ExpandedSpacesAndDummySymbolRespectMaximumTokenBytes) {
  auto spec = SpmSpec(true);
  std::string marker = "▁";
  TokenId full = -1;
  const auto symbols_per_token = tok::kMaxTokenBytes / marker.size();
  for (std::size_t length = 2; length <= symbols_per_token; ++length) {
    marker += "▁";
    full = SpmAdd(spec, marker, 0);
  }
  auto t = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(t);
  const std::string text(256 * symbols_per_token - 1, ' ');
  EXPECT_EQ(t->WorkingBytes(text), 96 * (text.size() + 1));
  const auto ids = Encode(*t, text, {.max_tokens = 256});
  ASSERT_EQ(ids.size(), 256U);
  EXPECT_TRUE(std::all_of(ids.begin(), ids.end(), [full](TokenId id) { return id == full; }));
  std::string decoded;
  ASSERT_TRUE(t->Decode(ids, {.remove_space_prefix = true}, decoded));
  EXPECT_EQ(decoded, text);
}

TEST(SentencePiece, RejectsNonfiniteScoresAndIncompatibleOrMalformedSpecs) {
  for (const float value :
       {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    auto spec = SpmSpec();
    spec.scores[256] = value;
    EXPECT_FALSE(tok::Tokenizer::Create(std::move(spec)));
  }
  auto spec = SpmSpec();
  spec.scores.pop_back();
  EXPECT_FALSE(tok::Tokenizer::Create(spec));
  spec = SpmSpec();
  spec.merges = {{"a", "a"}};
  EXPECT_FALSE(tok::Tokenizer::Create(spec));
  spec = SpmSpec();
  spec.normalization = tok::Normalization::kNfc;
  EXPECT_FALSE(tok::Tokenizer::Create(spec));
  spec = SpmSpec();
  spec.tokens[10] = "<0xgg>";
  EXPECT_FALSE(tok::Tokenizer::Create(spec));
  spec = SpmSpec();
  spec.kinds[10] = TokenKind::kUnused;
  EXPECT_FALSE(tok::Tokenizer::Create(spec));
  spec = SpmSpec();
  spec.rstrip.resize(1);
  EXPECT_FALSE(tok::Tokenizer::Create(spec));
}

GgufWriter SpmGguf(std::string_view name = "Phi 3.5 Mini Instruct",
                   std::optional<bool> prefix = std::nullopt, std::string_view special = "<s>") {
  auto spec = SpmSpec();
  spec.tokens[264] = special;
  GgufWriter w;
  w.Header(prefix ? 8 : 7);
  w.String("general.name", name);
  w.String("tokenizer.ggml.model", "llama");
  w.Strings("tokenizer.ggml.tokens", spec.tokens);
  std::vector<std::int32_t> kinds(256, 6);
  kinds.insert(kinds.end(), 8, 1);
  kinds.push_back(special == "</s>" ? 4 : 3);
  w.Ints("tokenizer.ggml.token_type", kinds);
  w.Floats("tokenizer.ggml.scores", spec.scores);
  w.U32("tokenizer.ggml.bos_token_id", 264);
  w.Bool("tokenizer.ggml.add_bos_token", false);
  if (prefix) {
    w.Bool("tokenizer.ggml.add_space_prefix", *prefix);
  }
  return w;
}

TEST(Gguf, SentencePieceFlagsScoresAndPhiNameOverrideAreExact) {
  const auto g = tok::ReadGgufTokenizer(SpmGguf().Bytes());
  ASSERT_TRUE(g);
  EXPECT_TRUE(g->spec.add_space_prefix);
  EXPECT_FALSE(g->spec.add_bos);
  EXPECT_EQ(g->spec.scores.size(), g->spec.tokens.size());
  EXPECT_TRUE(g->spec.rstrip.empty());  // approved spaced name does not trigger pinned override
  const auto named = tok::ReadGgufTokenizer(SpmGguf("Phi-3.5").Bytes());
  ASSERT_TRUE(named);
  EXPECT_EQ(named->spec.rstrip.size(), named->spec.tokens.size());
  EXPECT_FALSE(named->spec.rstrip[264]);  // <s> is explicitly exempted
  const auto gemma = tok::ReadGgufTokenizer(SpmGguf("Gemma", false).Bytes());
  ASSERT_TRUE(gemma);
  EXPECT_FALSE(gemma->spec.add_space_prefix);
}

TEST(Gguf, SentencePieceSpecialOverridesAndStripExemptionsMatchThePin) {
  for (const auto name : {"Phi 3.5 Mini Instruct", "Phi-3.5"}) {
    auto g = tok::ReadGgufTokenizer(SpmGguf(name, std::nullopt, "</s>").Bytes());
    ASSERT_TRUE(g);
    EXPECT_EQ(g->spec.kinds[264], TokenKind::kControl);  // metadata originally user-defined
    auto t = tok::Tokenizer::Create(std::move(g->spec));
    ASSERT_TRUE(t);
    const auto plain = Encode(*t, "</s>");
    EXPECT_EQ(std::count(plain.begin(), plain.end(), 264), 0);
    EXPECT_EQ(Encode(*t, "</s>  a", {.special = tok::SpecialTokens::kParse}),
              name == std::string_view("Phi-3.5") ? (std::vector<TokenId>{264, 258})
                                                  : (std::vector<TokenId>{264, 257, 257, 258}));
  }
}

TEST(Gguf, SentencePieceRefusesNonfiniteWrongTypesAndScoreCounts) {
  for (const auto bits : {0x7F800000U, 0xFF800000U, 0x7FC00000U}) {
    auto w = SpmGguf();
    const auto at =
        w.raw().find("tokenizer.ggml.scores") + std::string_view("tokenizer.ggml.scores").size();
    for (unsigned byte = 0; byte < 4; ++byte) {
      w.raw()[at + 16 + byte] = static_cast<char>((bits >> (8U * byte)) & 255U);
    }
    EXPECT_EQ(Failed(tok::ReadGgufTokenizer(w.Bytes()), &tok::Error::rule), Rule::kVocabulary);
  }
  auto w = SpmGguf();
  const auto at =
      w.raw().find("tokenizer.ggml.scores") + std::string_view("tokenizer.ggml.scores").size();
  w.raw()[at + 4] = 5;  // INT32 arrays are unsupported for this F32 score contract
  EXPECT_EQ(Failed(tok::ReadGgufTokenizer(w.Bytes()), &tok::Error::rule), Rule::kFormat);
  w = SpmGguf();
  w.raw()[at + 8] = 8;
  w.raw()[at + 9] = 1;  // 264 scores for 265 tokens; preserve following keys
  w.raw().erase(at + 16 + 264 * 4, 4);
  EXPECT_EQ(Failed(tok::ReadGgufTokenizer(w.Bytes()), &tok::Error::rule), Rule::kVocabulary);
}

TEST(Gguf, SentencePieceEveryTruncationFailsBeforeAllocation) {
  const auto w = SpmGguf();
  for (std::size_t size = 0; size < w.Bytes().size(); ++size) {
    EXPECT_FALSE(tok::ReadGgufTokenizer(w.Bytes().first(size))) << size;
  }
}

TEST(Gguf, SentencePieceHostileScoreCountsRefuseWithoutPayload) {
  for (const auto count : {static_cast<std::uint64_t>(tok::kMaxGgufArray),
                           static_cast<std::uint64_t>(tok::kMaxGgufArray) + 1,
                           std::numeric_limits<std::uint64_t>::max()}) {
    GgufWriter w;
    w.Header(1);
    w.Str("tokenizer.ggml.scores");
    w.U(9, 4);  // array
    w.U(6, 4);  // F32
    w.U(count, 8);
    EXPECT_EQ(Failed(tok::ReadGgufTokenizer(w.Bytes()), &tok::Error::rule),
              count > tok::kMaxGgufArray ? Rule::kBounds : Rule::kFormat);
  }
}

TEST(Gguf, ReadsTheTokenizer) {
  const GgufWriter w = SmallGguf();
  const auto g = tok::ReadGgufTokenizer(w.Bytes());
  ASSERT_TRUE(g.has_value()) << g.error().ToString();
  EXPECT_EQ(g->architecture, "test");
  EXPECT_EQ(g->pre, "qwen35");
  EXPECT_EQ(g->spec.pre_tokenizer, PreTokenizer::kByteBpe);
  EXPECT_EQ(g->chat_template, "{{ messages }}");
  EXPECT_EQ(g->spec.tokens.size(), 259U);
  EXPECT_EQ(g->spec.kinds[258], TokenKind::kUnused);
  EXPECT_EQ(g->spec.eos, 257);
  auto t = tok::Tokenizer::Create(g->spec);
  ASSERT_TRUE(t.has_value()) << t.error().ToString();
  EXPECT_EQ(Encode(*t, "ab"), (std::vector<TokenId>{256}));
}

TEST(Gguf, LlamaPresetDefaultsAndExplicitFalseOverride) {
  auto original = SmallGguf("llama-bpe");
  auto explicit_false = tok::ReadGgufTokenizer(original.Bytes());
  ASSERT_TRUE(explicit_false);
  EXPECT_TRUE(explicit_false->spec.ignore_merges);
  EXPECT_FALSE(explicit_false->spec.add_bos);
  // Replace the explicit false insertion flag with a named BOS, keeping
  // the metadata count fixed. The absent flag must select the preset default.
  auto& bytes = original.raw();
  const auto offset = bytes.find("tokenizer.ggml.add_bos_token");
  ASSERT_NE(offset, std::string::npos);
  bytes.erase(offset - 8, 8 + std::string_view("tokenizer.ggml.add_bos_token").size() + 4 + 1);
  original.U32("tokenizer.ggml.bos_token_id", 257);
  auto defaults = tok::ReadGgufTokenizer(original.Bytes());
  ASSERT_TRUE(defaults) << Failed(defaults, &tok::Error::ToString).value_or("");
  EXPECT_TRUE(defaults->spec.add_bos);
  auto tokenizer = tok::Tokenizer::Create(std::move(defaults->spec));
  ASSERT_TRUE(tokenizer);
  EXPECT_EQ(Encode(*tokenizer, "", {.add_bos_eos = true}), (std::vector<TokenId>{257}));
  EXPECT_EQ(Encode(*tokenizer, "ab", {.add_bos_eos = true}), (std::vector<TokenId>{257, 256}));
}

TEST(Encode, DirectPieceVocabularyLookupFollowsSourceFlag) {
  Vocab v;
  const auto whole = v.Add(Alphabet("abc"));
  auto preset = tok::ReadPreTokenizerPreset("llama-bpe");
  ASSERT_TRUE(preset);
  v.spec.split_expressions = preset->expressions;
  v.spec.split_dialect = preset->dialect;
  v.spec.ignore_merges = preset->ignore_merges;
  EXPECT_EQ(Encode(v.Build(), "abc"), (std::vector<TokenId>{whole}));
  v.spec.ignore_merges = false;
  EXPECT_EQ(Encode(v.Build(), "abc"), (std::vector<TokenId>{97, 98, 99}));
}

TEST(Gguf, ExplicitEmptyAndAbsentBytePresetUseThePinnedDefault) {
  auto explicit_default = tok::ReadGgufTokenizer(SmallGguf("default").Bytes());
  auto empty = tok::ReadGgufTokenizer(SmallGguf("").Bytes());
  ASSERT_TRUE(explicit_default);
  ASSERT_TRUE(empty);
  auto absent = SmallGguf("default");
  const auto key = absent.raw().find("tokenizer.ggml.pre");
  ASSERT_NE(key, std::string::npos);
  const auto begin = key - 8;  // uint64 key length
  const auto bytes = 8 + std::string_view("tokenizer.ggml.pre").size() + 4 + 8 + 7;
  absent.raw().erase(begin, bytes);
  absent.raw()[16] = 10;  // metadata count; the source fixture has eleven keys
  auto read_absent = tok::ReadGgufTokenizer(absent.Bytes());
  ASSERT_TRUE(read_absent) << Failed(read_absent, &tok::Error::ToString).value_or("");
  for (const auto* read : {&*explicit_default, &*empty, &*read_absent}) {
    EXPECT_EQ(read->pre, "default");
    EXPECT_EQ(read->spec.pre_tokenizer, PreTokenizer::kByteBpe);
    EXPECT_EQ(read->spec.split_expressions, explicit_default->spec.split_expressions);
    EXPECT_EQ(read->spec.split_expressions.size(), 4U);
    EXPECT_FALSE(read->spec.ignore_merges);
  }
  EXPECT_EQ(Pieces("default", "1234567"), (Strings{"123", "456", "7"}));
  EXPECT_EQ(Pieces("gpt-2", "1234567"), (Strings{"1234567"}));
}

TEST(Gguf, EveryTruncationFailsCleanly) {
  const GgufWriter w = SmallGguf();
  const auto all = w.Bytes();
  for (std::size_t n = 0; n < all.size(); ++n) {
    const auto g = tok::ReadGgufTokenizer(all.first(n));
    ASSERT_FALSE(g.has_value()) << n;
  }
}

// Corrupting bytes anywhere gives an error or a tokenizer, never a crash
// (the test presets check bounds, D-083); whatever reads, Create validates.
TEST(Gguf, RandomCorruptionsNeverCrash) {
  const GgufWriter w = SmallGguf();
  std::mt19937 random(7);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::size_t read = 0;
  for (int round = 0; round < 3000; ++round) {
    std::string bytes(reinterpret_cast<const char*>(w.Bytes().data()), w.Bytes().size());
    const int edits = 1 + static_cast<int>(random() % 4);
    for (int k = 0; k < edits; ++k) {
      bytes[random() % bytes.size()] = static_cast<char>(random() & 0xFFU);
    }
    const auto g = tok::ReadGgufTokenizer(std::as_bytes(std::span(bytes)));
    if (g) {
      ++read;
      const auto t = tok::Tokenizer::Create(g->spec);
      if (t) {
        std::vector<TokenId> ids;
        tok::EncodeOptions parse;
        parse.special = tok::SpecialTokens::kParse;
        EXPECT_TRUE(t->Encode("ab<|end|>\xC3\xA9 x", parse, ids).has_value());
      }
    }
  }
  EXPECT_GT(read, 0U);  // some corruptions only touch token texts
}

TEST(Encode, RandomTextNeverCrashes) {
  Vocab v;
  v.Merge("a", "b");
  v.Merge(" ", "a");
  v.Add("<|end|>", TokenKind::kControl);
  v.Add("<think>", TokenKind::kUserDefined);
  v.spec.normalization = tok::Normalization::kNfc;
  for (const auto p : {"qwen2", "qwen35", "deepseek-v3"}) {
    v.spec.split_expressions = tok::ReadPreTokenizerPreset(p)->expressions;
    const tok::Tokenizer t = v.Build();
    std::mt19937 random(11);  // NOLINT(bugprone-random-generator-seed): reproducible
    const std::array<std::string_view, 14> atoms = {
        "a", "b",        " ",        "\n",           "\r",      "\t",      "1",
        "!", "\xC3\xA9", "\xCC\x81", "\xE4\xB8\xAD", "<|end|>", "<think>", "'s"};
    for (int round = 0; round < 2000; ++round) {
      std::string text;
      const int n = static_cast<int>(random() % 40);
      for (int k = 0; k < n; ++k) {
        text += atoms[random() % atoms.size()];
      }
      for (const auto special :
           {tok::SpecialTokens::kParse, tok::SpecialTokens::kUserDefinedOnly}) {
        std::vector<TokenId> ids;
        tok::EncodeOptions o;
        o.special = special;
        ASSERT_TRUE(t.Encode(text, o, ids).has_value());
        std::string back;
        ASSERT_TRUE(t.Decode(ids, {.control_tokens = true}, back).has_value());
        std::vector<char32_t> cps;
        ASSERT_TRUE(uni::DecodeUtf8(back, cps).has_value());
      }
    }
  }
}

TEST(Gguf, RefusesMalformedAndUnsupportedFiles) {
  auto rule = [](const GgufWriter& w) {
    return Failed(tok::ReadGgufTokenizer(w.Bytes()), &tok::Error::rule);
  };
  {
    GgufWriter w = SmallGguf();
    w.raw()[0] = 'X';
    EXPECT_EQ(rule(w), Rule::kFormat);
  }
  EXPECT_TRUE(tok::ReadGgufTokenizer(SmallGguf("llama-bpe").Bytes()));
  {
    GgufWriter w;
    w.Header(2).String("tokenizer.ggml.model", "gpt2").String("tokenizer.ggml.model", "gpt2");
    EXPECT_EQ(rule(w), Rule::kFormat);  // repeated key
  }
  {
    GgufWriter w;
    w.Header(1);
    w.Str("tokenizer.ggml.tokens");
    w.U(9, 4);
    w.U(8, 4);
    w.U(std::uint64_t{1} << 40U, 8);  // an absurd count, before allocating
    EXPECT_EQ(rule(w), Rule::kBounds);
  }
  {
    GgufWriter w;
    w.Header(1);
    w.Str("tokenizer.ggml.tokens");
    w.U(9, 4);
    w.U(8, 4);
    w.U(1000, 8);  // more strings than the bytes could hold
    EXPECT_EQ(rule(w), Rule::kFormat);
  }
  {
    GgufWriter w;
    w.Header(1);
    w.Str("x");
    w.U(99, 4);  // unknown type
    EXPECT_EQ(rule(w), Rule::kFormat);
  }
  {
    GgufWriter w;
    w.Header(std::uint64_t{1} << 20U);
    EXPECT_EQ(rule(w), Rule::kBounds);
  }
  {
    GgufWriter w;
    w.Header(1, 1);
    EXPECT_EQ(rule(w), Rule::kUnsupported);
  }
  {
    GgufWriter w;
    w.Header(3)
        .String("tokenizer.ggml.model", "llama")
        .String("tokenizer.ggml.pre", "qwen2")
        .U32("x", 1);
    EXPECT_EQ(rule(w), Rule::kUnsupported);
  }
  {
    GgufWriter w;
    w.Header(2).U32("tokenizer.ggml.model", 1).String("tokenizer.ggml.pre", "qwen2");
    EXPECT_EQ(rule(w), Rule::kFormat);  // wrong type
  }
}

TEST(Gguf, RefusesUnknownTokenTypesAndBadMerges) {
  Vocab v;
  std::vector<std::int32_t> types(v.spec.tokens.size(), 1);
  types[5] = 6;  // byte (SPM)
  GgufWriter w;
  w.Header(5).String("tokenizer.ggml.model", "gpt2").String("tokenizer.ggml.pre", "qwen2");
  w.Strings("tokenizer.ggml.tokens", v.spec.tokens).Ints("tokenizer.ggml.token_type", types);
  w.Strings("tokenizer.ggml.merges", {});
  EXPECT_EQ(Failed(tok::ReadGgufTokenizer(w.Bytes()), &tok::Error::rule), Rule::kUnsupported);

  types[5] = 1;
  GgufWriter m;
  m.Header(5).String("tokenizer.ggml.model", "gpt2").String("tokenizer.ggml.pre", "qwen2");
  m.Strings("tokenizer.ggml.tokens", v.spec.tokens).Ints("tokenizer.ggml.token_type", types);
  m.Strings("tokenizer.ggml.merges", {"a  b"});
  EXPECT_EQ(Failed(tok::ReadGgufTokenizer(m.Bytes()), &tok::Error::rule), Rule::kVocabulary);
}

// --- tokenizer.json ---------------------------------------------------------------

std::string Quote(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    out += c;
  }
  return out + "\"";
}

constexpr std::string_view kQwen35Regex =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)re";

std::string HfJson(
    std::string_view normalizer = R"({"type": "NFC"})", std::string_view regex = kQwen35Regex,
    std::string_view added =
        R"([{"id": 257, "content": "<|end|>", "single_word": false, "lstrip": false,
      "rstrip": false, "normalized": false, "special": true},
     {"id": 258, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false,
      "normalized": false, "special": false}])",
    std::string_view model_extra = R"("byte_fallback": false, "ignore_merges": false)") {
  std::string vocab;
  for (unsigned b = 0; b < 256; ++b) {
    vocab += Quote(ByteText(b)) + ": " + std::to_string(b) + ", ";
  }
  vocab += Quote(Alphabet("ab")) + ": 256";
  std::string json = R"({"version": "1.0", "truncation": null, "padding": null, "added_tokens": )";
  json += added;
  json += R"(, "normalizer": )";
  json += normalizer;
  json += R"(, "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": ")";
  json += regex;
  json += R"("}, "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
    "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
    "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
    "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
      "end_of_word_suffix": "", "fuse_unk": false, )";
  json += model_extra;
  json += R"(, "vocab": {)" + vocab + R"(}, "merges": [[)" + Quote(Alphabet("a")) + ", " +
          Quote(Alphabet("b")) + "]]}}";
  return json;
}

TEST(HfTokenizer, ExtraSpecialTokenListsAndNamedMapsShareControlSemantics) {
  for (
      const std::string_view extras :
      {R"(["<|end|>",{"content":"<think>","rstrip":false,"lstrip":false,"single_word":false}])",
       R"({"plain":"<|end|>","structured":{"content":"<think>","rstrip":false,"lstrip":false,"single_word":false}})"}) {
    SCOPED_TRACE(extras);
    auto spec = tok::ReadHfTokenizer(HfJson());
    ASSERT_TRUE(spec);
    const auto count = spec->tokens.size();
    ASSERT_TRUE(tok::ApplyHfTokenizerConfig(
        R"({"extra_special_tokens":)" + std::string(extras) + "}", *spec));
    EXPECT_EQ(spec->tokens.size(), count);  // Existing IDs are resolved, never allocated.
    EXPECT_EQ(spec->tokens[257], "<|end|>");
    EXPECT_EQ(spec->tokens[258], "<think>");
    for (const auto id : {257U, 258U}) {
      EXPECT_EQ(spec->kinds[id], TokenKind::kControl);
    }
    EXPECT_TRUE(spec->added_tokens.empty());
    EXPECT_TRUE(spec->model_tokens.empty());
    EXPECT_TRUE(spec->rstrip.empty());
    EXPECT_EQ(spec->kinds[97], TokenKind::kNormal);
    auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
    ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
    EXPECT_EQ(Encode(*tokenizer, "ab"), (std::vector<TokenId>{256}));
    EXPECT_EQ(Encode(*tokenizer, "<|end|>", {.special = tok::SpecialTokens::kParse}),
              (std::vector<TokenId>{257}));
    EXPECT_EQ(Encode(*tokenizer, "<think>a", {.special = tok::SpecialTokens::kParse}),
              (std::vector<TokenId>{258, 97}));
    EXPECT_NE(Encode(*tokenizer, "<think>"), (std::vector<TokenId>{258}));
  }
}

TEST(HfTokenizer, ExtraSpecialTokenEmptyFormsLeaveVocabularyUnchanged) {
  for (const std::string_view config :
       {R"({})", R"({"extra_special_tokens":null})", R"({"extra_special_tokens":[]})",
        R"({"extra_special_tokens":{}})"}) {
    auto spec = tok::ReadHfTokenizer(HfJson());
    ASSERT_TRUE(spec);
    const auto before = *spec;
    ASSERT_TRUE(tok::ApplyHfTokenizerConfig(config, *spec));
    EXPECT_EQ(spec->tokens, before.tokens);
    EXPECT_EQ(spec->kinds, before.kinds);
    EXPECT_EQ(spec->added_tokens, before.added_tokens);
    EXPECT_EQ(spec->model_tokens, before.model_tokens);
  }
}

TEST(HfTokenizer, ExtraSpecialTokenFormsPreserveValidationAndRefusals) {
  for (const auto& [value, rule] :
       {std::pair{R"(42)", Rule::kFormat}, std::pair{R"("ab")", Rule::kFormat},
        std::pair{R"(true)", Rule::kFormat}, std::pair{R"([null])", Rule::kFormat},
        std::pair{R"([{"content":42}])", Rule::kFormat},
        std::pair{R"({"named":{}})", Rule::kFormat}, std::pair{R"(["absent"])", Rule::kVocabulary},
        std::pair{R"({"named":"absent"})", Rule::kVocabulary},
        std::pair{R"([{"content":"ab","lstrip":true}])", Rule::kUnsupported},
        std::pair{R"({"named":{"content":"ab","single_word":true}})", Rule::kUnsupported},
        std::pair{R"([{"content":"ab","rstrip":"true"}])", Rule::kUnsupported},
        std::pair{R"([{"content":"ab","single_word":1}])", Rule::kUnsupported}}) {
    SCOPED_TRACE(value);
    auto spec = tok::ReadHfTokenizer(HfJson());
    ASSERT_TRUE(spec);
    EXPECT_EQ(Failed(tok::ApplyHfTokenizerConfig(
                         R"({"extra_special_tokens":)" + std::string(value) + "}", *spec),
                     &tok::Error::rule),
              rule);
  }
  // The list does not bypass ID/content checks in added_tokens_decoder.
  for (const std::string_view added : {R"({"260":{"content":"<new>","special":true}})",
                                       R"({"256":{"content":"<new>","special":true}})"}) {
    auto spec = tok::ReadHfTokenizer(HfJson());
    ASSERT_TRUE(spec);
    EXPECT_EQ(Failed(tok::ApplyHfTokenizerConfig(
                         R"({"extra_special_tokens":["<new>"],"added_tokens_decoder":)" +
                             std::string(added) + "}",
                         *spec),
                     &tok::Error::rule),
              Rule::kVocabulary);
  }
}

std::string HfWithPost(std::string_view post) {
  auto text = HfJson();
  const auto begin = text.find("\"post_processor\":");
  const auto end = text.find("\"decoder\":", begin);
  text.replace(begin, end - begin, "\"post_processor\":" + std::string(post) + ",\n");
  return text;
}

std::string HfWithPre(std::string_view pre) {
  auto text = HfJson();
  const auto begin = text.find("\"pre_tokenizer\":");
  const auto end = text.find("\"post_processor\":", begin);
  text.replace(begin, end - begin, "\"pre_tokenizer\":" + std::string(pre) + ",\n");
  return text;
}

TEST(HfTokenizer, SourceDialectTravelsFromReaderIntoTheEncodingPlan) {
  auto hf = tok::ReadHfTokenizer(HfJson("null", "b$"));
  ASSERT_TRUE(hf);
  EXPECT_EQ(hf->split_dialect, tok::RegexDialect::kHuggingFace);
  auto gguf = tok::ReadGgufTokenizer(SmallGguf("deepseek-llm").Bytes());
  ASSERT_TRUE(gguf);
  EXPECT_EQ(gguf->spec.split_dialect, tok::RegexDialect::kGguf);
  // The source vocab contains the ab merge. A line-end match isolates b
  // before BPE, whereas an absolute-end expression leaves ab together.
  auto absolute = *hf;
  absolute.split_dialect = tok::RegexDialect::kGguf;
  auto line_tokenizer = tok::Tokenizer::Create(std::move(*hf));
  auto absolute_tokenizer = tok::Tokenizer::Create(std::move(absolute));
  ASSERT_TRUE(line_tokenizer);
  ASSERT_TRUE(absolute_tokenizer);
  EXPECT_EQ(Encode(*line_tokenizer, "ab\ncd"), (std::vector<TokenId>{97, 98, 10, 99, 100}));
  EXPECT_EQ(Encode(*absolute_tokenizer, "ab\ncd"), (std::vector<TokenId>{256, 10, 99, 100}));
}

constexpr std::string_view kTemplate =
    R"({"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"start","type_id":0}},)"
    R"({"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"end","type_id":0}}],)"
    R"("pair":"start $A end $B:1","special_tokens":{)"
    R"("start":{"id":"start","ids":[257,258],"tokens":["<|end|>","<think>"]},)"
    R"("end":{"id":"end","ids":[257],"tokens":["<|end|>"]}}})";

std::vector<TokenId> WrapperIds(const tok::PostProcessorPlan& plan, bool prefix) {
  std::vector<TokenId> result;
  for (const auto piece : prefix ? plan.prefix : plan.suffix)
    for (const auto id : std::span(plan.ids).subspan(piece.offset, piece.length))
      result.push_back(id);
  return result;
}

TEST(HfTokenizer, SharedPostprocessorMultiIdsSequenceOrderAndRoleSeparation) {
  auto second = std::string(kTemplate);
  const auto begin = second.find("\"single\":");
  const auto end = second.find("\"pair\":", begin);
  second.replace(begin, end - begin, R"("single":"end:1 $A:0 start",)");
  const auto post =
      R"({"type":"Sequence","processors":[)" + std::string(kTemplate) +
      R"(,{"type":"Sequence","processors":[{"type":"ByteLevel","trim_offsets":true},)" + second +
      "]}]}";
  auto spec = tok::ReadHfTokenizer(HfWithPost(post));
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  EXPECT_FALSE(spec->bos);
  EXPECT_FALSE(spec->eos);
  EXPECT_FALSE(spec->add_bos);
  EXPECT_EQ(WrapperIds(spec->post_processor, true), (std::vector<TokenId>{257, 257, 258}));
  EXPECT_EQ(WrapperIds(spec->post_processor, false), (std::vector<TokenId>{257, 257, 258}));
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "ab"), (std::vector<TokenId>{256}));
  EXPECT_EQ(Encode(*tokenizer, "ab", {.add_bos_eos = true}),
            (std::vector<TokenId>{257, 257, 258, 256, 257, 257, 258}));
  EXPECT_EQ(Encode(*tokenizer, "", {.add_bos_eos = true}),
            (std::vector<TokenId>{257, 257, 258, 257, 257, 258}));
  EXPECT_GE(tokenizer->OutputTokensBound(0, true), 6U);
  std::vector<TokenId> out{9};
  EXPECT_EQ(Failed(tokenizer->Encode("ab", {.add_bos_eos = true, .max_tokens = 6}, out),
                   &tok::Error::rule),
            Rule::kOutputTooLarge);
  EXPECT_EQ(out.front(), 9);
  const std::string control = "<|end|>";
  const std::array<tok::SpecialSpan, 1> spans{{{0, control.size()}}};
  out = {9};
  std::vector<std::size_t> positions;
  ASSERT_TRUE(tokenizer->EncodeMarked(control, spans, {.add_bos_eos = true}, out, &positions));
  EXPECT_EQ(positions, (std::vector<std::size_t>{4}));
  EXPECT_EQ(out, (std::vector<TokenId>{9, 257, 257, 258, 257, 257, 257, 258}));
}

TEST(HfTokenizer, SourceSizedStageAndPostprocessorPlansAcceptAffordableLargeInputs) {
  std::string pre = R"({"type":"Sequence","pretokenizers":[)";
  for (unsigned i = 0; i < 33; ++i)
    pre += R"({"type":"Split","pattern":{"Regex":"a"},"behavior":"Isolated","invert":false},)";
  pre += R"({"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]})";
  auto split = tok::ReadHfTokenizer(HfWithPre(pre));
  ASSERT_TRUE(split) << Failed(split, &tok::Error::ToString).value_or("");
  EXPECT_EQ(split->split_expressions.size(), 33U);
  auto tokenizer = tok::Tokenizer::Create(std::move(*split));
  ASSERT_TRUE(tokenizer);
  EXPECT_EQ(Encode(*tokenizer, "aab"), (std::vector<TokenId>{97, 97, 98}));

  std::string post = R"({"type":"Sequence","processors":[)";
  for (unsigned i = 0; i < 1025; ++i) {
    if (i) post += ',';
    post += R"({"type":"ByteLevel"})";
  }
  post += "]}";
  auto offsets = tok::ReadHfTokenizer(HfWithPost(post));
  ASSERT_TRUE(offsets) << Failed(offsets, &tok::Error::ToString).value_or("");
  EXPECT_TRUE(offsets->post_processor.ids.empty());
  EXPECT_TRUE(offsets->post_processor.prefix.empty());

  post = R"({"type":"TemplateProcessing","single":")";
  for (unsigned i = 0; i < 65537; ++i) post += "x ";
  post += R"($A","special_tokens":{"x":{"id":"x","ids":[257],"tokens":["<|end|>"]}}})";
  auto wrappers = tok::ReadHfTokenizer(HfWithPost(post));
  ASSERT_TRUE(wrappers) << Failed(wrappers, &tok::Error::ToString).value_or("");
  EXPECT_EQ(wrappers->post_processor.ids.size(), 1U);
  EXPECT_EQ(wrappers->post_processor.prefix.size(), 65537U);
  const auto payment = tok::Tokenizer::ConstructionBytes(*wrappers);
  ASSERT_TRUE(payment);
  EXPECT_GE(*payment, wrappers->post_processor.prefix.capacity() * sizeof(tok::PostProcessorSlice));
  auto large = tok::Tokenizer::Create(std::move(*wrappers));
  ASSERT_TRUE(large);
  EXPECT_EQ(Encode(*large, "", {.add_bos_eos = true, .max_tokens = 65537}).size(), 65537U);
}

TEST(HfTokenizer, MultiIdReferencesStayCompactAndEmissionPaysItsOutputBound) {
  constexpr std::size_t count = 4096;
  std::string post = R"({"type":"TemplateProcessing","single":")";
  for (std::size_t i = 0; i < count; ++i) post += "x ";
  post += R"($A","special_tokens":{"x":{"id":"x","ids":[)";
  for (std::size_t i = 0; i < count; ++i) post += i ? ",257" : "257";
  post += R"(],"tokens":[)";
  for (std::size_t i = 0; i < count; ++i) post += i ? ",\"<|end|>\"" : "\"<|end|>\"";
  post += "]}}}";
  auto source = tok::ReadHfTokenizer(HfWithPost(post));
  ASSERT_TRUE(source) << Failed(source, &tok::Error::ToString).value_or("");
  EXPECT_EQ(source->post_processor.ids.size(), count);
  EXPECT_EQ(source->post_processor.prefix.size(), count);
  const auto bytes = tok::Tokenizer::ConstructionBytes(*source);
  ASSERT_TRUE(bytes);
  EXPECT_LT(*bytes, count * count * sizeof(TokenId));
  auto tokenizer = tok::Tokenizer::Create(*source);
  ASSERT_TRUE(tokenizer);
  EXPECT_GE(tokenizer->OutputTokensBound(0, true), count * count);
  std::vector<TokenId> out;
  const auto encoded = tokenizer->Encode("", {.add_bos_eos = true, .max_tokens = 8}, out);
  EXPECT_EQ(Failed(encoded, &tok::Error::rule), Rule::kOutputTooLarge);
  EXPECT_EQ(out.size(), 8U);
  source->post_processor.prefix[0] = {source->post_processor.ids.size(), 1};
  EXPECT_EQ(Failed(tok::Tokenizer::ConstructionBytes(*source), &tok::Error::rule),
            Rule::kVocabulary);
  EXPECT_EQ(Failed(tok::Tokenizer::Create(*source), &tok::Error::rule), Rule::kVocabulary);

  llmp::base::WorkPulse pulse;
  pulse.Cancel();
  llmp::base::SetThreadPulse(&pulse);
  const auto cancelled_read = tok::ReadHfTokenizer(HfWithPost(post));
  const auto cancelled_create = tok::Tokenizer::Create(*source);
  out.clear();
  const auto cancelled_emit = tokenizer->Encode("", {.add_bos_eos = true}, out);
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(Failed(cancelled_read, &tok::Error::rule), Rule::kCancelled);
  EXPECT_EQ(Failed(cancelled_create, &tok::Error::rule), Rule::kCancelled);
  EXPECT_EQ(Failed(cancelled_emit, &tok::Error::rule), Rule::kCancelled);
  EXPECT_TRUE(out.empty());
}

TEST(HfTokenizer, CompactInsertionReservesCapacityBeforeEmissionAndKeepsAppendLimits) {
  Vocab vocab;
  const auto control = vocab.Add("<control>", TokenKind::kControl);
  vocab.spec.has_post_processor = true;
  vocab.spec.post_processor.ids.assign(512, control);
  vocab.spec.post_processor.prefix.assign(256, {0, 512});
  vocab.spec.post_processor.prefix.push_back({0, 1});
  auto tokenizer = vocab.Build();
  const tok::EncodeOptions options{.add_bos_eos = true, .max_tokens = 200000};
  std::vector<TokenId> out{11, 12};
  ASSERT_TRUE(tokenizer.ReserveOutput("", options, out, 1));
  const auto capacity = out.capacity();
  const auto* address = out.data();
  EXPECT_EQ(capacity, 2 + tokenizer.OutputTokensBound(0, true) + 1);
  ASSERT_TRUE(tokenizer.Encode("", options, out));
  EXPECT_EQ(out.data(), address);
  EXPECT_EQ(out.capacity(), capacity);
  EXPECT_EQ(out.size(), 2 + 131073U);
  EXPECT_EQ(out[0], 11);
  EXPECT_EQ(out[1], 12);

  std::vector<TokenId> limited(100, 11);
  const auto result = tokenizer.Encode("", {.add_bos_eos = true, .max_tokens = 8}, limited);
  EXPECT_EQ(Failed(result, &tok::Error::rule), Rule::kOutputTooLarge);
  EXPECT_EQ(limited.size(), 108U);
  EXPECT_EQ(limited.capacity(), 108U);
  std::vector<TokenId> invalid{11};
  EXPECT_EQ(Failed(tokenizer.ReserveOutput("", {}, invalid, SIZE_MAX), &tok::Error::rule),
            Rule::kBounds);
  EXPECT_EQ(invalid, (std::vector<TokenId>{11}));
  const std::string bad_utf8(1, static_cast<char>(0xff));
  const auto original_capacity = invalid.capacity();
  EXPECT_FALSE(tokenizer.ReserveOutput(bad_utf8, options, invalid));
  EXPECT_EQ(invalid.capacity(), original_capacity);

  const std::string text = "<control>";
  const std::array<tok::SpecialSpan, 1> spans{{{0, text.size()}}};
  std::vector<TokenId> marked{11};
  std::vector<std::size_t> positions{7, 8};
  ASSERT_TRUE(tokenizer.EncodeMarked(text, spans, {.max_tokens = 1}, marked, &positions));
  EXPECT_EQ(marked, (std::vector<TokenId>{11, control}));
  EXPECT_EQ(positions, (std::vector<std::size_t>{7, 8, 1}));
  EXPECT_EQ(positions.capacity(), 3U);
}

TEST(HfTokenizer, ManyDefinitionsAndLateReferencesHaveBoundedLookupWork) {
  constexpr std::size_t definitions = 2048, references = 8192;
  const auto name = [](std::size_t i) { return "definition_" + std::to_string(i); };
  const auto last = name(definitions - 1);
  std::string post = R"({"type":"TemplateProcessing","single":")";
  for (std::size_t i = 0; i < references; ++i) post += last + ' ';
  post += name(0) + " $A " + name(2) + ' ' + last;
  post += R"(","special_tokens":{)";
  for (std::size_t i = 0; i < definitions; ++i) {
    if (i) post += ',';
    const auto id = name(i);
    post += Quote(id) + R"(:{"id":)" + Quote(id);
    post +=
        i % 2 ? R"(,"ids":[258],"tokens":["<think>"]})" : R"(,"ids":[257],"tokens":["<|end|>"]})";
  }
  post += "}}";
  const auto input = HfWithPost(post);
  llmp::base::WorkPulse pulse;
  llmp::base::SetThreadPulse(&pulse);
  auto source = tok::ReadHfTokenizer(input);
  llmp::base::SetThreadPulse(nullptr);
  ASSERT_TRUE(source) << Failed(source, &tok::Error::ToString).value_or("");
  // Parsing pulses also cover JSON, source scanning, IDs and index comparisons.
  // The old two-pass late-name scan alone beats >131,000 times here. Bound
  // metered work, not elapsed time, while allowing all source-linear work.
  EXPECT_LT(pulse.beats(), 4 * (definitions + references));
  EXPECT_EQ(source->post_processor.ids.size(), definitions);
  ASSERT_EQ(source->post_processor.prefix.size(), references + 1);
  ASSERT_EQ(source->post_processor.suffix.size(), 2U);
  for (std::size_t i = 0; i < references; ++i) {
    EXPECT_EQ(source->post_processor.prefix[i].offset, definitions - 1);
    EXPECT_EQ(source->post_processor.prefix[i].length, 1U);
  }
  EXPECT_EQ(source->post_processor.prefix.back().offset, 0U);
  EXPECT_EQ(source->post_processor.suffix[0].offset, 2U);
  EXPECT_EQ(source->post_processor.suffix[1].offset, definitions - 1);
  // Index nodes are transient parser storage, conservatively 128 bytes each
  // inside the existing source allowance. Retained construction funds only
  // the definition-ID pool and ordered slices, not a definitions*references
  // table or an expanded insertion list.
  EXPECT_LE(128 * definitions, 4 * post.size());
  const auto payment = tok::Tokenizer::ConstructionBytes(*source);
  ASSERT_TRUE(payment);
  const auto& plan = source->post_processor;
  EXPECT_GE(*payment, plan.ids.capacity() * sizeof(TokenId) +
                          (plan.prefix.capacity() + plan.suffix.capacity()) *
                              sizeof(tok::PostProcessorSlice));
  EXPECT_LT(*payment, definitions * references * sizeof(TokenId));
  auto tokenizer = tok::Tokenizer::Create(std::move(*source));
  ASSERT_TRUE(tokenizer);
  const auto output = Encode(*tokenizer, "ab", {.add_bos_eos = true, .max_tokens = references + 4});
  ASSERT_EQ(output.size(), references + 4);
  EXPECT_TRUE(std::ranges::all_of(std::span(output).first(references),
                                  [](TokenId id) { return id == 258; }));
  EXPECT_EQ(output[references], 257);
  EXPECT_EQ(output[references + 1], 256);
  EXPECT_EQ(output[references + 2], 257);
  EXPECT_EQ(output[references + 3], 258);

  const auto repeated_name = post.substr(0, post.size() - 2) + ',' + Quote(name(0)) + R"(:{"id":)" +
                             Quote(name(0)) + R"(,"ids":[257],"tokens":["<|end|>"]}}})";
  EXPECT_EQ(Failed(tok::ReadHfTokenizer(HfWithPost(repeated_name)), &tok::Error::rule),
            Rule::kFormat);
}

TEST(HfTokenizer, TemplateStringSequenceAliasesAndMalformedDefinitions) {
  auto post = std::string(kTemplate);
  auto begin = post.find("\"single\":");
  auto end = post.find("\"pair\":", begin);
  post.replace(begin, end - begin, R"("single":"start $0 end",)");
  auto spec = tok::ReadHfTokenizer(HfWithPost(post));
  ASSERT_TRUE(spec);
  EXPECT_EQ(WrapperIds(spec->post_processor, true), (std::vector<TokenId>{257, 258}));
  EXPECT_EQ(WrapperIds(spec->post_processor, false), (std::vector<TokenId>{257}));
  for (const auto& [from, to] : {
           std::pair{R"("ids":[257,258])", R"("ids":[257])"},
           std::pair{R"("ids":[257,258])", R"("ids":[257,-1])"},
           std::pair{R"("tokens":["<|end|>","<think>"])", R"("tokens":["<|end|>","wrong"] )"},
           std::pair{R"("type_id":0)", R"("type_id":4294967296)"},
           std::pair{R"("id":"A")", R"("id":"B")"},
           std::pair{R"("id":"start","type_id":0)", R"("id":"missing","type_id":0)"},
           std::pair{R"("pair":"start $A end $B:1")", R"("pair":"start $A end $A:1")"},
       }) {
    auto bad = std::string(kTemplate);
    const auto offset = bad.find(from);
    ASSERT_NE(offset, std::string::npos);
    bad.replace(offset, std::string_view(from).size(), to);
    EXPECT_FALSE(tok::ReadHfTokenizer(HfWithPost(bad))) << from;
  }
  for (const auto single : {"start end", "$A $A", "$B", "$A:no", "$4294967296"}) {
    auto bad = std::string(kTemplate);
    begin = bad.find("\"single\":");
    end = bad.find("\"pair\":", begin);
    bad.replace(begin, end - begin, "\"single\":" + Quote(single) + ",");
    EXPECT_FALSE(tok::ReadHfTokenizer(HfWithPost(bad))) << single;
  }
  EXPECT_FALSE(tok::ReadHfTokenizer(HfWithPost(R"({"type":"Sequence","processors":null})")));
  EXPECT_FALSE(tok::ReadHfTokenizer(HfWithPost(R"({"type":"ByteLevel","trim_offsets":"true"})")));
}

TEST(HfTokenizer, GenericNestedSplitsAndFaithfulByteLevelOptions) {
  auto text = HfJson();
  const auto begin = text.find("\"pre_tokenizer\":");
  const auto end = text.find("\"post_processor\":", begin);
  const auto with = [&](std::string_view pre) {
    auto result = text;
    result.replace(begin, end - begin, "\"pre_tokenizer\":" + std::string(pre) + ",");
    return result;
  };
  const auto generic = with(
      R"({"type":"Sequence","pretokenizers":[{"type":"Sequence","pretokenizers":[)"
      R"({"type":"Split","pattern":{"Regex":"b+"},"behavior":"Isolated","invert":false}]},)"
      R"({"type":"ByteLevel","add_prefix_space":false,"use_regex":false,"trim_offsets":true}]})");
  auto spec = tok::ReadHfTokenizer(generic);
  ASSERT_TRUE(spec);
  EXPECT_EQ(spec->split_expressions, (std::vector<std::string>{"b+"}));
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer);
  EXPECT_EQ(Encode(*tokenizer, "ab"), (std::vector<TokenId>{97, 98}));
  auto byte_only = tok::ReadHfTokenizer(
      with(R"({"type":"ByteLevel","add_prefix_space":false,"use_regex":false})"));
  ASSERT_TRUE(byte_only);
  EXPECT_TRUE(byte_only->split_expressions.empty());
  auto gpt2 = tok::ReadHfTokenizer(with(R"({"type":"ByteLevel","add_prefix_space":false})"));
  ASSERT_TRUE(gpt2);
  EXPECT_EQ(gpt2->split_expressions, tok::ReadPreTokenizerPreset("gpt-2")->expressions);
  auto prefix_space = tok::ReadHfTokenizer(with(R"({"type":"ByteLevel","add_prefix_space":true})"));
  ASSERT_TRUE(prefix_space);
  EXPECT_TRUE(prefix_space->add_space_prefix);
  for (
      const auto pre :
      {R"({"type":"ByteLevel","add_prefix_space":false,"use_regex":"false"})",
       R"({"type":"Sequence","pretokenizers":[{"type":"ByteLevel","add_prefix_space":false,"use_regex":false},)"
       R"({"type":"Split","pattern":{"Regex":"."},"behavior":"Isolated"}]})"}) {
    EXPECT_FALSE(tok::ReadHfTokenizer(with(pre))) << pre;
  }
}

std::string RawHfJson(bool literal_tab = false) {
  auto spec = RawSpec();
  spec.tokens.push_back("▁▁");  // a non-special added token that remains in the BPE model
  spec.tokens.push_back("a\n");
  spec.merges.emplace_back("▁", "▁");
  spec.merges.emplace_back("a", "\n");
  if (literal_tab) {
    // Official Gemma 2's JSON has a normal tab instead of <0x09>, and
    // declares merges between tab tokens.
    spec.tokens[9] = "\t";
    spec.tokens.push_back("\t\t");
    spec.merges.emplace_back("\t", "\t");
  }
  const auto quoted = [](std::string_view text) {
    std::string out;
    llmp::base::json::AppendQuoted(text, out);
    return out;
  };
  std::string json =
      R"({"truncation":null,"padding":null,"normalizer":{"type":"Replace",)"
      R"("pattern":{"String":" "},"content":"▁"},"pre_tokenizer":{"type":"Split",)"
      R"("pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},)"
      R"("decoder":{"type":"Sequence","decoders":[{"type":"Replace",)"
      R"("pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},)"
      R"("post_processor":{"type":"TemplateProcessing","single":[)"
      R"({"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],)"
      R"("special_tokens":{"<bos>":{"id":"<bos>","ids":[264],"tokens":["<bos>"]}}},)"
      R"("added_tokens":[{"id":264,"content":"<bos>","special":true,"normalized":false},)"
      R"({"id":265,"content":"▁▁","special":false,"normalized":false}],)"
      R"("model":{"type":"BPE","dropout":null,"unk_token":"<bos>","fuse_unk":true,)"
      R"("byte_fallback":true,"vocab":{)";
  for (std::size_t i = 0; i < spec.tokens.size(); ++i) {
    if (i != 0) json += ',';
    json += quoted(spec.tokens[i]) + ':' + std::to_string(i);
  }
  json += R"(},"merges":[)";
  for (std::size_t i = 0; i < spec.merges.size(); ++i) {
    if (i != 0) json += ',';
    json += '[' + quoted(spec.merges[i].first) + ',' + quoted(spec.merges[i].second) + ']';
  }
  return json + "]}}";
}

TEST(HfTokenizer, ExtraSpecialTokensUpdateMaterializedMembershipFlags) {
  for (const std::string_view extras : {R"(["▁▁"])", R"({"named":"▁▁"})"}) {
    auto spec = tok::ReadHfTokenizer(RawHfJson());
    ASSERT_TRUE(spec);
    ASSERT_TRUE(spec->model_tokens[265]);
    // The config reader also accepts specs whose container materializes added membership.
    spec->added_tokens.assign(spec->tokens.size(), false);
    ASSERT_TRUE(tok::ApplyHfTokenizerConfig(
        R"({"extra_special_tokens":)" + std::string(extras) + "}", *spec));
    EXPECT_EQ(spec->kinds[265], TokenKind::kControl);
    EXPECT_FALSE(spec->model_tokens[265]);
    EXPECT_TRUE(spec->added_tokens[265]);
    EXPECT_TRUE(spec->model_tokens[256]);
    EXPECT_FALSE(spec->added_tokens[256]);
  }
}

TEST(HfTokenizer, RawPipelinePreservesWhitespaceModelMembershipAndCrossNewlineMerges) {
  auto spec = tok::ReadHfTokenizer(RawHfJson());
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  EXPECT_EQ(spec->pre_tokenizer, PreTokenizer::kRawBpe);
  EXPECT_TRUE(spec->model_tokens[265]);
  EXPECT_FALSE(spec->model_tokens[264]);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "  "), (std::vector<TokenId>{265}));
  EXPECT_EQ(Encode(*tokenizer, "a\n"), (std::vector<TokenId>{266}));
  EXPECT_EQ(Encode(*tokenizer, "a\n", {.add_bos_eos = true}), (std::vector<TokenId>{264, 266}));
  std::string decoded;
  ASSERT_TRUE(tokenizer->Decode(std::vector<TokenId>{265, 266}, {}, decoded));
  EXPECT_EQ(decoded, "  a\n");
}

TEST(HfTokenizer, RawAsciiModelTokensCoverMissingByteFallbackWithoutLosingMerges) {
  auto spec = tok::ReadHfTokenizer(RawHfJson(true));
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  EXPECT_EQ(spec->kinds[9], TokenKind::kNormal);
  EXPECT_TRUE(spec->model_tokens[9]);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "\t"), (std::vector<TokenId>{9}));
  EXPECT_EQ(Encode(*tokenizer, "\t\t\t"), (std::vector<TokenId>{267, 9}));
  const std::string text = "a\t\t\t🙂\n";
  std::string decoded;
  ASSERT_TRUE(tokenizer->Decode(Encode(*tokenizer, text), {}, decoded));
  EXPECT_EQ(decoded, text);
}

TEST(HfTokenizer, RawByteCoverageRejectsMissingOrSpecialAsciiAndNonAsciiSubstitutes) {
  for (const auto& [from, to] : {
           std::pair{R"("<0x09>":9)", R"("x":9)"},
           std::pair{R"("<0x20>":32)", R"(" ":32)"},
           std::pair{R"("<0xA9>":169)", R"("©":169)"},
           std::pair{R"("<0x09>":9)", R"("<0xGG>":9)"},
           std::pair{R"("added_tokens":[)",
                     R"("added_tokens":[{"id":9,"content":"<0x09>","special":true},)"},
       }) {
    auto json = RawHfJson();
    const auto at = json.find(from);
    ASSERT_NE(at, std::string::npos);
    json.replace(at, std::string_view(from).size(), to);
    auto spec = tok::ReadHfTokenizer(json);
    ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
    const auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
    ASSERT_FALSE(tokenizer) << from;
    EXPECT_EQ(Failed(tokenizer, &tok::Error::rule), Rule::kVocabulary);
  }
  // A literal tab that only matches as an added control is not model
  // coverage when controls are parsed as ordinary text.
  auto json = RawHfJson(true);
  const auto at = json.find(R"("added_tokens":[)");
  ASSERT_NE(at, std::string::npos);
  json.insert(at + std::string_view(R"("added_tokens":[)").size(),
              R"({"id":9,"content":"\t","special":true},)");
  auto spec = tok::ReadHfTokenizer(json);
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  EXPECT_FALSE(tok::Tokenizer::Create(std::move(*spec)));
}

TEST(HfTokenizer, NormalizedAddedTokensFreezeBeforeRawBpeMerges) {
  auto json = RawHfJson();
  const std::string_view from = R"("content":"▁▁","special":false,"normalized":false)";
  const auto at = json.find(from);
  ASSERT_NE(at, std::string::npos);
  json.replace(at, from.size(), R"("content":"▁▁","special":false,"normalized":true)");
  auto spec = tok::ReadHfTokenizer(json);
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  ASSERT_TRUE(spec->normalized_tokens[265]);
  // Prefer the model's ▁+a merge, and then allow a larger token containing
  // both spaces. Normalized added-token extraction must precede either merge.
  spec->tokens.push_back("▁▁a");
  spec->kinds.push_back(TokenKind::kNormal);
  spec->model_tokens.push_back(true);
  spec->normalized_tokens.push_back(false);
  spec->merges.emplace_back("▁", "▁a");
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, "  a"), (std::vector<TokenId>{265, 256}));
  EXPECT_EQ(Encode(*tokenizer, "▁▁a"), (std::vector<TokenId>{265, 256}));
}

TEST(HfTokenizer, ConfigNormalizedTokensUseTransformedTextAndSeparateMatchingPhases) {
  auto spec = tok::ReadHfTokenizer(RawHfJson());
  ASSERT_TRUE(spec) << Failed(spec, &tok::Error::ToString).value_or("");
  const auto applied = tok::ApplyHfTokenizerConfig(
      R"({"added_tokens_decoder":{"265":{"content":"▁▁","special":false,"normalized":true},)"
      R"("267":{"content":" a▁","special":false,"normalized":true},)"
      R"("268":{"content":"<x>","special":true,"normalized":true}}})",
      *spec);
  ASSERT_TRUE(applied) << Failed(applied, &tok::Error::ToString).value_or("");
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  EXPECT_EQ(Encode(*tokenizer, " a▁"), (std::vector<TokenId>{267}));
  EXPECT_EQ(Encode(*tokenizer, "▁a▁", {.max_tokens = 1}), (std::vector<TokenId>{267}));
  // The raw matcher must not consume normalized ▁▁ before the longer
  // normalized token starting at the preceding space can match.
  EXPECT_EQ(Encode(*tokenizer, " a▁▁"), (std::vector<TokenId>{267, 257}));
  EXPECT_EQ(Encode(*tokenizer, "<x>", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{268}));
  EXPECT_NE(Encode(*tokenizer, "<x>"), (std::vector<TokenId>{268}));
}

TEST(HfTokenizer, MarkedEncodingDoesNotParseUnmarkedNormalizedControls) {
  auto spec = tok::ReadHfTokenizer(RawHfJson());
  ASSERT_TRUE(spec);
  ASSERT_TRUE(tok::ApplyHfTokenizerConfig(
      R"({"added_tokens_decoder":{"267":{"content":"<control>","special":true,"normalized":true}}})",
      *spec));
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  CheckMarkedNormalizedControl(*tokenizer, "<control>", 267);
}

TEST(HfTokenizer, RawPipelineRefusesUnsupportedOrMalformedDeclaredOptions) {
  for (const auto& [from, to] :
       {std::pair{R"("content":"▁")", R"("content":"x")"},
        std::pair{R"("invert":false)", R"("invert":true)"},
        std::pair{R"("behavior":"MergedWithPrevious")", R"("behavior":"Removed")"},
        std::pair{R"("ids":[264])", R"("ids":[263])"},
        std::pair{R"("Sequence":{"id":"A")", R"("Sequence":{"id":"B")"},
        std::pair{R"("byte_fallback":true)", R"("byte_fallback":false)"},
        std::pair{R"("normalized":false)", R"("normalized":"false")"}}) {
    auto json = RawHfJson();
    const auto at = json.find(from);
    ASSERT_NE(at, std::string::npos);
    json.replace(at, std::string_view(from).size(), to);
    EXPECT_FALSE(tok::ReadHfTokenizer(json)) << from;
  }
}

TEST(SentencePiece, NormalizedUserDefinedSymbolsAreFrozenBeforeScoreMerging) {
  auto spec = SpmSpec();
  spec.decode_added_whitespace = true;
  const auto whitespace = SpmAdd(spec, "▁▁", 0, TokenKind::kUserDefined);
  SpmAdd(spec, "▁▁a", 100);
  auto tokenizer = tok::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(tokenizer) << Failed(tokenizer, &tok::Error::ToString).value_or("");
  const auto ids = Encode(*tokenizer, "  a");
  EXPECT_EQ(ids, (std::vector<TokenId>{whitespace, 256}));
  std::string decoded;
  ASSERT_TRUE(tokenizer->Decode(ids, {}, decoded));
  EXPECT_EQ(decoded, "  a");
}

TEST(HfTokenizer, ReadsASupportedFile) {
  const auto spec = tok::ReadHfTokenizer(HfJson());
  ASSERT_TRUE(spec.has_value()) << spec.error().ToString();
  EXPECT_EQ(spec->pre_tokenizer, PreTokenizer::kByteBpe);
  EXPECT_EQ(spec->normalization, tok::Normalization::kNfc);
  ASSERT_EQ(spec->tokens.size(), 259U);
  EXPECT_EQ(spec->kinds[257], TokenKind::kControl);
  EXPECT_EQ(spec->kinds[258], TokenKind::kUserDefined);
  auto t = tok::Tokenizer::Create(*spec);
  ASSERT_TRUE(t.has_value()) << t.error().ToString();
  EXPECT_EQ(Encode(*t, "ab<think>"), (std::vector<TokenId>{256, 258}));
}

TEST(SentencePieceModel, SkipsUnknownTrainerFieldsAbove54Cleanly) {
  auto spec = tok::ReadSentencePieceTokenizer(SentencePieceModel({}, {}, ProtoScalar(60, 42)));
  EXPECT_TRUE(spec);
  auto spec_too_high =
      tok::ReadSentencePieceTokenizer(SentencePieceModel({}, {}, ProtoScalar(130, 42)));
  EXPECT_EQ(Failed(spec_too_high, &tok::Error::rule), Rule::kUnsupported);
}

TEST(HfTokenizer, AcceptsRstripAndDigits) {
  auto rstrip_spec = tok::ReadHfTokenizer(
      HfJson(R"({"type": "NFC"})", kQwen35Regex,
             R"([{"id": 257, "content": "<|end|>", "rstrip": true, "special": true}])"));
  ASSERT_TRUE(rstrip_spec);
  EXPECT_FALSE(rstrip_spec->rstrip.empty());
  EXPECT_TRUE(rstrip_spec->rstrip[257]);

  auto digits_json = HfWithPre(R"({"type": "Sequence", "pretokenizers": [
      {"type": "Digits", "individual_digits": true},
      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
  ]})");
  auto digits_spec = tok::ReadHfTokenizer(digits_json);
  ASSERT_TRUE(digits_spec);
  ASSERT_EQ(digits_spec->split_expressions.size(), 1U);
  EXPECT_EQ(digits_spec->split_expressions[0], R"(\p{N})");
  auto tok_digits = tok::Tokenizer::Create(*digits_spec);
  ASSERT_TRUE(tok_digits) << tok_digits.error().ToString();
  EXPECT_EQ(Encode(*tok_digits, "12"), (std::vector<TokenId>{'1', '2'}));
}

TEST(HfTokenizer, RefusesWhatItDoesNotImplement) {
  auto rule = [](const std::string& json) {
    return Failed(tok::ReadHfTokenizer(json), &tok::Error::rule);
  };
  EXPECT_EQ(rule(HfJson(R"({"type": "NFKC"})")), Rule::kUnsupported);
  EXPECT_EQ(rule(HfJson(R"({"type": "UnsupportedType"})")), Rule::kUnsupported);
  EXPECT_TRUE(tok::ReadHfTokenizer(HfJson(R"({"type": "NFC"})", R"(\\p{L}+)")));
  EXPECT_EQ(rule(HfJson(R"({"type": "NFC"})", kQwen35Regex, "[]", R"("byte_fallback": true)")),
            Rule::kUnsupported);
  EXPECT_EQ(rule(HfJson(R"({"type": "NFC"})", kQwen35Regex,
                        R"([{"id": 257, "content": "<x>", "lstrip": true, "special": true}])")),
            Rule::kUnsupported);
  EXPECT_EQ(rule(HfJson(R"({"type": "NFC"})", kQwen35Regex,
                        R"([{"id": 257, "content": "<x>", "normalized": true, "special": true}])")),
            Rule::kUnsupported);  // matched after NFC: not implemented
  EXPECT_EQ(rule(HfJson(R"({"type": "NFC"})", kQwen35Regex,
                        R"([{"id": 300, "content": "<x>", "special": true}])")),
            Rule::kFormat);  // an ID past the vocabulary
  EXPECT_EQ(rule(HfJson(R"({"type": "NFC"})", kQwen35Regex,
                        R"([{"id": 1, "content": "<x>", "special": true}])")),
            Rule::kVocabulary);  // changes an entry
  EXPECT_EQ(rule("{"), Rule::kFormat);
  EXPECT_EQ(rule("[]"), Rule::kFormat);
}

tok::ForcedSpanOptions ForcedOptions(std::size_t tokens = 64) {
  tok::ForcedSpanOptions options;
  options.encode.max_bytes = 4096;
  options.encode.max_tokens = tokens;
  options.remaining_tokens = tokens;
  options.remaining_context = tokens;
  return options;
}

std::string ForcedText(const tok::Tokenizer& tokenizer, std::span<const TokenId> ids,
                       tok::DecodeOptions options = {.control_tokens = true}) {
  std::string out;
  auto decoded = tokenizer.Decode(ids, options, out);
  EXPECT_TRUE(decoded) << Failed(decoded, &tok::Error::ToString).value_or("");
  return out;
}

TEST(ForcedSpan, HealsOnlyTheUnpublishedSelectionAndHoldsAnUnstableTail) {
  Vocab v;
  const auto ab = v.Merge("a", "b");
  auto tokenizer = v.Build();
  auto plan = tokenizer.PlanForcedSpan(Byte('a'), "b!", ForcedOptions());
  ASSERT_TRUE(plan) << Failed(plan, &tok::Error::ToString).value_or("");
  EXPECT_EQ(plan->tokens, (std::vector<TokenId>{ab}));
  EXPECT_TRUE(plan->replaces_selected);
  EXPECT_TRUE(plan->selected_covered);
  EXPECT_EQ(plan->committed_bytes, 2U);
  EXPECT_EQ(plan->consumed_forced_bytes, 1U);
  EXPECT_EQ(plan->held, "!");
  EXPECT_EQ(ForcedText(tokenizer, plan->tokens) + plan->held, "ab!");
}

TEST(ForcedSpan, StableEarlierBlocksBecomeOneCertainSpanIncludingTheAnchor) {
  Vocab v;
  const auto marker = v.Add("<call>", TokenKind::kUserDefined);
  auto tokenizer = v.Build();
  auto plan = tokenizer.PlanForcedSpan(Byte('a'), " b c<call>", ForcedOptions());
  ASSERT_TRUE(plan) << Failed(plan, &tok::Error::ToString).value_or("");
  EXPECT_GT(plan->tokens.size(), 3U);
  EXPECT_EQ(plan->tokens.front(), Byte('a'));
  EXPECT_EQ(plan->tokens.back(), marker);
  EXPECT_EQ(ForcedText(tokenizer, plan->tokens), "a b c<call>");
  EXPECT_EQ(plan->consumed_forced_bytes, 10U);
  EXPECT_TRUE(plan->held.empty());
  EXPECT_FALSE(plan->replaces_selected);
}

TEST(ForcedSpan, CompleteAllowedFollowingBytesProveTheLastBoundary) {
  Vocab v;
  v.spec.split_expressions = {"[a-z]+|[0-9]+|[^a-z0-9]+"};
  auto tokenizer = v.Build();
  auto options = ForcedOptions();
  auto open = tokenizer.PlanForcedSpan(std::nullopt, "abc>", options);
  ASSERT_TRUE(open);
  EXPECT_EQ(open->held, ">");
  const std::array<std::uint8_t, 2> next{'a', 'b'};
  options.following_known = true;
  options.following_bytes = next;
  auto closed = tokenizer.PlanForcedSpan(std::nullopt, "abc>", options);
  ASSERT_TRUE(closed);
  EXPECT_EQ(ForcedText(tokenizer, closed->tokens), "abc>");
  EXPECT_TRUE(closed->held.empty());
  const std::array<std::uint8_t, 2> mixed{'a', 0xc3};
  options.following_bytes = mixed;
  auto unicode = tokenizer.PlanForcedSpan(std::nullopt, "abc>", options);
  ASSERT_TRUE(unicode);
  EXPECT_EQ(unicode->held, ">");  // no sampled Unicode representatives
}

TEST(ForcedSpan, CoverRejectsShortSplitsAndAllowsLexicallyCheckedLongerCandidates) {
  Vocab v;
  v.spec.split_expressions = {"ab|.+"};
  const auto ab = v.Merge("a", "b");
  const auto abc = v.Merge("ab", "c");
  const auto unicode = v.Merge("\xc3", "\xa9");
  auto tokenizer = v.Build();
  auto cover = tokenizer.ForcedCoverBytes("ab", ForcedOptions());
  ASSERT_TRUE(cover);
  ASSERT_EQ(*cover, 2U);
  EXPECT_FALSE(tokenizer.CoversForced(Byte('a'), *cover));
  EXPECT_TRUE(tokenizer.CoversForced(ab, *cover));
  EXPECT_TRUE(tokenizer.CoversForced(abc, *cover));
  EXPECT_TRUE(tokenizer.CoversForced(unicode, *cover));  // caller checks actual lexical bytes
  EXPECT_FALSE(tokenizer.CoversForced(-1, *cover));
  EXPECT_FALSE(tokenizer.CoversForced(static_cast<TokenId>(tokenizer.size()), *cover));
}

TEST(ForcedSpan, EndAssertionsCannotJustifyACoverThatWouldRejectCanonicalContinuation) {
  Vocab v;
  v.spec.split_expressions = {"ab$|a"};
  v.Merge("a", "b");
  auto tokenizer = v.Build();
  auto cover = tokenizer.ForcedCoverBytes("ab", ForcedOptions());
  ASSERT_TRUE(cover);
  EXPECT_EQ(*cover, 0U);  // after x, the actual pretoken split is a / bx
  EXPECT_EQ(Encode(tokenizer, "abx").front(), Byte('a'));
}

TEST(ForcedSpan, FailedHigherPriorityFutureDependentBranchHoldsEarlierBlocks) {
  Vocab v;
  v.spec.split_expressions = {"a.*z|a|!"};
  auto tokenizer = v.Build();
  auto plan = tokenizer.PlanForcedSpan(Byte('a'), "!", ForcedOptions());
  ASSERT_TRUE(plan);
  EXPECT_TRUE(plan->tokens.empty());
  EXPECT_FALSE(plan->selected_covered);
  EXPECT_EQ(plan->held, "a!");
  auto options = ForcedOptions();
  options.terminal = true;
  auto terminal = tokenizer.PlanForcedSpan(Byte('a'), "!", options);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(ForcedText(tokenizer, terminal->tokens), "a!");
}

TEST(ForcedSpan, ActualAddedAndControlAtomsRespectLongerTriePrefixes) {
  Vocab v;
  const auto short_atom = v.Add("<t>", TokenKind::kControl);
  const auto long_atom = v.Add("<t>x", TokenKind::kControl);
  auto tokenizer = v.Build();
  auto options = ForcedOptions();
  auto held = tokenizer.PlanForcedSpan(std::nullopt, "<t>", options);
  ASSERT_TRUE(held);
  EXPECT_TRUE(held->tokens.empty());
  EXPECT_EQ(held->held, "<t>");
  const std::array<std::uint8_t, 1> next{'y'};
  options.following_known = true;
  options.following_bytes = next;
  auto proven = tokenizer.PlanForcedSpan(std::nullopt, "<t>", options);
  ASSERT_TRUE(proven);
  EXPECT_EQ(proven->tokens, (std::vector<TokenId>{short_atom}));
  auto longest = tokenizer.PlanForcedSpan(std::nullopt, "<t>x", options);
  ASSERT_TRUE(longest);
  EXPECT_EQ(longest->tokens, (std::vector<TokenId>{long_atom}));
  options.decode.control_tokens = false;
  EXPECT_FALSE(tokenizer.PlanForcedSpan(std::nullopt, "<t>x", options));
}

TEST(ForcedSpan, PartialUtf8IsHeldOrCompletedExactlyWithoutReplacementCharacters) {
  Vocab v;
  const auto e = v.Merge("\xc3", "\xa9");
  auto tokenizer = v.Build();
  auto incomplete = tokenizer.PlanForcedSpan(Byte('\xc3'), "", ForcedOptions());
  ASSERT_TRUE(incomplete);
  EXPECT_FALSE(incomplete->selected_covered);
  EXPECT_TRUE(incomplete->tokens.empty());
  EXPECT_EQ(incomplete->held, "\xc3");
  auto complete = tokenizer.PlanForcedSpan(Byte('\xc3'), "\xa9!", ForcedOptions());
  ASSERT_TRUE(complete);
  EXPECT_EQ(complete->tokens, (std::vector<TokenId>{e}));
  EXPECT_EQ(complete->consumed_forced_bytes, 1U);
  EXPECT_EQ(complete->held, "!");
  EXPECT_FALSE(tokenizer.PlanForcedSpan(Byte('\xc3'), "x", ForcedOptions()));
}

TEST(ForcedSpan, NfcAndWhitespaceLossAreExplicitCapabilityErrors) {
  Vocab v;
  v.spec.normalization = tok::Normalization::kNfc;
  auto tokenizer = v.Build();
  auto changed = tokenizer.PlanForcedSpan(Byte('e'), "\xcc\x81!", ForcedOptions());
  ASSERT_FALSE(changed);
  EXPECT_EQ(changed.error().rule, Rule::kUnsupported);
  auto stripping = SpmSpec(false);
  auto preserved = tok::Tokenizer::Create(stripping);
  ASSERT_TRUE(preserved);
  auto options = ForcedOptions();
  options.terminal = true;
  auto original = preserved->PlanForcedSpan(std::nullopt, "<bos>  a", options);
  ASSERT_TRUE(original);
  EXPECT_EQ(ForcedText(*preserved, original->tokens), "<bos>  a");
  stripping.rstrip.assign(stripping.tokens.size(), false);
  stripping.rstrip[264] = true;  // existing supported SentencePiece special rule
  auto strip = tok::Tokenizer::Create(std::move(stripping));
  ASSERT_TRUE(strip);
  auto lost = strip->PlanForcedSpan(std::nullopt, "<bos>  a", options);
  ASSERT_FALSE(lost);
  EXPECT_EQ(lost.error().rule, Rule::kUnsupported);
}

TEST(ForcedSpan, RawNewlineRunsAreTheRealBlocksAndLiteralMarkersMustReproduce) {
  auto tokenizer = tok::Tokenizer::Create(RawSpec());
  ASSERT_TRUE(tokenizer);
  auto plan = tokenizer->PlanForcedSpan(std::nullopt, "a\n\nb", ForcedOptions());
  ASSERT_TRUE(plan) << Failed(plan, &tok::Error::ToString).value_or("");
  EXPECT_EQ(ForcedText(*tokenizer, plan->tokens), "a\n\n");
  EXPECT_EQ(plan->held, "b");
  auto nonreproducing = tokenizer->PlanForcedSpan(std::nullopt, "▁", ForcedOptions());
  ASSERT_FALSE(nonreproducing);
  EXPECT_EQ(nonreproducing.error().rule, Rule::kUnsupported);
}

TEST(ForcedSpan, SentencePieceForcedBytesContinueTheStreamWithoutADummyPrefix) {
  auto tokenizer = tok::Tokenizer::Create(SpmSpec(true));
  ASSERT_TRUE(tokenizer);
  // A plain encoding starts a stream: its dummy ▁ would decode as a space.
  EXPECT_EQ(Encode(*tokenizer, "a"), (std::vector<TokenId>{258}));
  EXPECT_EQ(Encode(*tokenizer, "a", {.continuation = true}), (std::vector<TokenId>{256}));
  EXPECT_EQ(
      Encode(*tokenizer, "a<bos>a", {.special = tok::SpecialTokens::kParse, .continuation = true}),
      (std::vector<TokenId>{256, 264, 256}));
  auto options = ForcedOptions();
  options.terminal = true;
  auto middle = tokenizer->PlanForcedSpan(std::nullopt, "a", options);
  ASSERT_TRUE(middle) << Failed(middle, &tok::Error::ToString).value_or("");
  EXPECT_EQ(middle->tokens, (std::vector<TokenId>{256}));
  EXPECT_EQ(ForcedText(*tokenizer, middle->tokens), "a");
  // After a selected token, and after a control token inside the span.
  auto healed = tokenizer->PlanForcedSpan(256, " a", options);
  ASSERT_TRUE(healed) << Failed(healed, &tok::Error::ToString).value_or("");
  EXPECT_EQ(healed->tokens, (std::vector<TokenId>{259}));
  EXPECT_TRUE(healed->replaces_selected);
  auto control = tokenizer->PlanForcedSpan(std::nullopt, "<bos>a", options);
  ASSERT_TRUE(control) << Failed(control, &tok::Error::ToString).value_or("");
  EXPECT_EQ(control->tokens, (std::vector<TokenId>{264, 256}));
  auto cover = tokenizer->ForcedCoverBytes("a", options);
  ASSERT_TRUE(cover) << Failed(cover, &tok::Error::ToString).value_or("");
  EXPECT_EQ(*cover, 1U);
  // A span that starts the stream keeps the selected token's dummy-space policy.
  options.decode.remove_space_prefix = true;
  auto start = tokenizer->PlanForcedSpan(std::nullopt, "a", options);
  ASSERT_TRUE(start) << Failed(start, &tok::Error::ToString).value_or("");
  EXPECT_EQ(start->tokens, (std::vector<TokenId>{258}));
  EXPECT_EQ(ForcedText(*tokenizer, start->tokens, options.decode), "a");
}

TEST(ForcedSpan, ActualTokenAndContextLimitsCutOnlyCanonicalTokens) {
  Vocab v;
  const auto ab = v.Merge("a", "b");
  auto tokenizer = v.Build();
  auto options = ForcedOptions(2);
  options.terminal = true;
  auto cut = tokenizer.PlanForcedSpan(Byte('a'), "bcd", options);
  ASSERT_TRUE(cut);
  EXPECT_EQ(cut->tokens, (std::vector<TokenId>{ab, Byte('c')}));
  EXPECT_EQ(cut->held, "d");
  EXPECT_TRUE(cut->limited);
  EXPECT_EQ(cut->consumed_forced_bytes, 2U);
  options.remaining_context = 1;
  auto context = tokenizer.PlanForcedSpan(Byte('a'), "bcd", options);
  ASSERT_TRUE(context);
  EXPECT_EQ(context->tokens, (std::vector<TokenId>{ab}));
  EXPECT_EQ(context->held, "cd");
  EXPECT_TRUE(context->limited);
  options.remaining_context = 0;
  auto zero = tokenizer.PlanForcedSpan(Byte('a'), "bcd", options);
  ASSERT_TRUE(zero);
  EXPECT_TRUE(zero->tokens.empty());
  EXPECT_FALSE(zero->selected_covered);
  EXPECT_EQ(zero->held, "abcd");
  EXPECT_TRUE(zero->limited);
}

TEST(ForcedSpan, BoundsAreResourceDerivedAndPostprocessorInsertionIsRefused) {
  Vocab v;
  auto tokenizer = v.Build();
  auto options = ForcedOptions();
  options.encode.max_bytes = 2;
  EXPECT_FALSE(tokenizer.PlanForcedSpan(Byte('a'), "bc", options));
  options.encode.add_bos_eos = true;
  EXPECT_FALSE(tokenizer.PlanForcedSpan(std::nullopt, "a", options));
  EXPECT_EQ(tokenizer.ForcedSpanBytes(std::numeric_limits<std::size_t>::max()),
            std::numeric_limits<std::uint64_t>::max());
  EXPECT_GT(tokenizer.ForcedSpanBytes(1024), tokenizer.WorkingBytes(1024));
}

TEST(ForcedSpan, AnOpenSingleTokenIsNotACoverProofForLaterCrossingMerges) {
  Vocab v;
  const auto bc = v.Merge("b", "c");
  v.Merge("a", "b");
  auto tokenizer = v.Build();
  auto cover = tokenizer.ForcedCoverBytes("ab", ForcedOptions());
  ASSERT_TRUE(cover);
  EXPECT_EQ(*cover, 0U);
  EXPECT_EQ(Encode(tokenizer, "abc"), (std::vector<TokenId>{Byte('a'), bc}));
}

TEST(ForcedSpan, NormalizedAddedAtomsUseActualSourceOffsets) {
  auto json = RawHfJson();
  const std::string_view from = R"("content":"▁▁","special":false,"normalized":false)";
  const auto at = json.find(from);
  ASSERT_NE(at, std::string::npos);
  json.replace(at, from.size(), R"("content":"▁▁","special":false,"normalized":true)");
  auto spec = tok::ReadHfTokenizer(json);
  ASSERT_TRUE(spec);
  auto tokenizer = tok::Tokenizer::Create(std::move(*spec));
  ASSERT_TRUE(tokenizer);
  auto plan = tokenizer->PlanForcedSpan(std::nullopt, "  a", ForcedOptions());
  ASSERT_TRUE(plan) << Failed(plan, &tok::Error::ToString).value_or("");
  EXPECT_EQ(plan->tokens, (std::vector<TokenId>{265}));
  EXPECT_EQ(plan->committed_bytes, 2U);
  EXPECT_EQ(ForcedText(*tokenizer, plan->tokens), "  ");
  EXPECT_EQ(plan->held, "a");
}

TEST(ForcedSpan, PlanningAndCoverRespectCancellation) {
  Vocab v;
  auto tokenizer = v.Build();
  llmp::base::WorkPulse pulse;
  pulse.Cancel();
  llmp::base::SetThreadPulse(&pulse);
  const auto plan = tokenizer.PlanForcedSpan(std::nullopt, "abc", ForcedOptions());
  const auto cover = tokenizer.ForcedCoverBytes("abc", ForcedOptions());
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(Failed(plan, &tok::Error::rule), Rule::kCancelled);
  EXPECT_EQ(Failed(cover, &tok::Error::rule), Rule::kCancelled);
}

}  // namespace

TEST(ForcedSpan, FollowingByteCannotPromoteARepartitionedCanonicalPrefix) {
  Vocab v;
  v.spec.split_expressions = {"ab(?=[c])|a"};
  const auto ab = v.Merge("a", "b");
  auto tokenizer = v.Build();
  EXPECT_EQ(Encode(tokenizer, "ab"), (std::vector<TokenId>{Byte('a'), Byte('b')}));
  EXPECT_EQ(Encode(tokenizer, "abc").front(), ab);
  auto options = ForcedOptions();
  const std::array<std::uint8_t, 1> following{'c'};
  options.following_known = true;
  options.following_bytes = following;
  auto plan = tokenizer.PlanForcedSpan(std::nullopt, "ab", options);
  ASSERT_TRUE(plan);
  EXPECT_TRUE(plan->tokens.empty());
  EXPECT_EQ(plan->held, "ab");
}

TEST(ForcedSpan, MultiStageSplitPlanAndUnmatchedGapPropagateStablePrefixExactly) {
  // 1. An unmatched gap stays open until a match ends it: appending "c" to
  // "axb" extends the gap "xb", so only the match "a" is stable.
  {
    auto regex = tok::Regex::Compile("a", tok::RegexDialect::kHuggingFace);
    ASSERT_TRUE(regex);
    const std::u32string text = U"axb";
    std::vector<std::uint32_t> lengths;
    tok::RegexBudget budget{.remaining = 1024};
    std::size_t stable_prefix = 0;
    ASSERT_TRUE(regex->Split(text, lengths, budget, &stable_prefix));
    EXPECT_EQ(lengths, (std::vector<std::uint32_t>{1, 2}));
    EXPECT_EQ(stable_prefix, 1U);
    // A gap a match closes is stable with it.
    const std::u32string closed = U"axa!";
    lengths.clear();
    ASSERT_TRUE(regex->Split(closed, lengths, budget, &stable_prefix));
    EXPECT_EQ(lengths, (std::vector<std::uint32_t>{1, 1, 1, 1}));
    EXPECT_EQ(stable_prefix, 3U);
  }
  // 2. Multi-stage SplitPlan propagates stable_prefix across all stages instead of overwriting
  // stage 0's instability with stage 1's local result.
  {
    const std::array<std::string, 2> expressions{"[a-z]+", "a"};
    auto plan = tok::SplitPlan::Compile(expressions, tok::RegexDialect::kHuggingFace);
    ASSERT_TRUE(plan);
    const std::u32string text = U"ab1c";
    std::vector<std::uint32_t> lengths;
    auto budget = plan->Budget(text.size());
    std::size_t stable_prefix = 0;
    ASSERT_TRUE(plan->Split(text, lengths, budget, &stable_prefix));
    EXPECT_EQ(lengths, (std::vector<std::uint32_t>{1, 1, 1, 1}));
    // Stage 0 proves "ab" (its match stopped at "1"); the gap "1" is closed only
    // by the match "c", which consulted EOF. Stage 1's own split of "1" and "c"
    // cannot extend stage 0's proof.
    EXPECT_EQ(stable_prefix, 2U);
  }
}

TEST(ForcedSpan, MarkedSpansKeepControlTextInClientBytesAsText) {
  Vocab v;
  const auto call = v.Add("<call>", TokenKind::kControl);
  const auto eos = v.Add("<eos>", TokenKind::kControl);
  const auto added = v.Add("<u>", TokenKind::kUserDefined);
  auto tokenizer = v.Build();
  const std::string marker = "<call>";
  const std::string forced = marker + "x<eos><u>y";  // a client name holding control text
  auto spans = tokenizer.ControlSpans(marker);
  ASSERT_TRUE(spans);
  ASSERT_EQ(spans->size(), 1U);
  auto options = ForcedOptions();
  options.terminal = true;
  // Unmarked kParse turns the client's text into a control ID.
  auto parsed = tokenizer.PlanForcedSpan(std::nullopt, forced, options);
  ASSERT_TRUE(parsed);
  EXPECT_NE(std::ranges::find(parsed->tokens, eos), parsed->tokens.end());
  options.marked = true;
  options.encode.special = tok::SpecialTokens::kUserDefinedOnly;
  options.specials = *spans;
  auto marked = tokenizer.PlanForcedSpan(std::nullopt, forced, options);
  ASSERT_TRUE(marked) << Failed(marked, &tok::Error::ToString).value_or("");
  EXPECT_EQ(marked->tokens.front(), call);
  EXPECT_EQ(std::ranges::find(marked->tokens, eos), marked->tokens.end());
  EXPECT_NE(std::ranges::find(marked->tokens, added), marked->tokens.end());  // still matches
  EXPECT_EQ(ForcedText(tokenizer, marked->tokens), forced);
  std::vector<TokenId> expected;
  ASSERT_TRUE(tokenizer.EncodeMarked(forced, *spans, {}, expected));
  EXPECT_EQ(marked->tokens, expected);
  // encode.special is ignored for control tokens once marked.
  options.encode.special = tok::SpecialTokens::kParse;
  auto ignored = tokenizer.PlanForcedSpan(std::nullopt, forced, options);
  ASSERT_TRUE(ignored);
  EXPECT_EQ(ignored->tokens, expected);
  // The cover follows the same policy: only a marked span is one atom.
  auto marked_cover = tokenizer.ForcedCoverBytes(marker, options);
  ASSERT_TRUE(marked_cover);
  EXPECT_EQ(*marked_cover, marker.size());
  options.specials = {};
  auto text_cover = tokenizer.ForcedCoverBytes(marker, options);
  ASSERT_TRUE(text_cover);
  EXPECT_EQ(*text_cover, 0U);
}

TEST(ForcedSpan, MarkedSpansHealASelectionInsideThemAndShiftIntoTheHeldTail) {
  Vocab v;
  const auto call = v.Add("<call>", TokenKind::kControl);
  auto tokenizer = v.Build();
  auto options = ForcedOptions();
  options.terminal = true;
  options.marked = true;
  options.encode.special = tok::SpecialTokens::kUserDefinedOnly;
  // The selected "<" begins the marked span: the plan replaces it.
  const std::array inside = {tok::SpecialSpan{0, 6}};
  options.specials = inside;
  auto healed = tokenizer.PlanForcedSpan(Byte('<'), "call>x", options);
  ASSERT_TRUE(healed) << Failed(healed, &tok::Error::ToString).value_or("");
  EXPECT_EQ(healed->tokens, (std::vector<TokenId>{call, Byte('x')}));
  EXPECT_TRUE(healed->replaces_selected);
  EXPECT_TRUE(healed->selected_covered);
  EXPECT_EQ(healed->consumed_forced_bytes, 6U);
  // A selected control token is its own span; an identical span merges.
  options.specials = {};
  auto own = tokenizer.PlanForcedSpan(call, "x", options);
  ASSERT_TRUE(own) << Failed(own, &tok::Error::ToString).value_or("");
  EXPECT_EQ(own->tokens, (std::vector<TokenId>{call, Byte('x')}));
  EXPECT_FALSE(own->replaces_selected);
  options.specials = inside;
  auto merged = tokenizer.PlanForcedSpan(call, "x", options);
  ASSERT_TRUE(merged);
  EXPECT_EQ(merged->tokens, own->tokens);
  // The held tail keeps its spans, shifted: here it is exactly one atom.
  options.terminal = false;
  options.remaining_tokens = 3;
  const std::array later = {tok::SpecialSpan{3, 6}};
  options.specials = later;
  auto held = tokenizer.PlanForcedSpan(Byte('a'), " b<call>", options);
  ASSERT_TRUE(held) << Failed(held, &tok::Error::ToString).value_or("");
  EXPECT_TRUE(held->limited);
  EXPECT_EQ(held->tokens.size(), 3U);
  EXPECT_EQ(held->held, "<call>");
  EXPECT_EQ(held->cover_bytes, 6U);
}

TEST(ForcedSpan, MarkedSpansAreCheckedLikeEncodeMarked) {
  Vocab v;
  const auto call = v.Add("<call>", TokenKind::kControl);
  auto tokenizer = v.Build();
  auto options = ForcedOptions();
  options.marked = true;
  const auto refused = [&](std::optional<TokenId> selected, std::string_view forced,
                           std::span<const tok::SpecialSpan> spans) {
    options.specials = spans;
    return Failed(tokenizer.PlanForcedSpan(selected, forced, options), &tok::Error::rule);
  };
  const std::array partial = {tok::SpecialSpan{0, 3}};
  EXPECT_EQ(refused(std::nullopt, "<call>", partial), Rule::kSpecialToken);
  const std::array outside = {tok::SpecialSpan{2, 6}};
  EXPECT_EQ(refused(std::nullopt, "<call>", outside), Rule::kSpecialToken);
  const std::array disordered = {tok::SpecialSpan{6, 6}, tok::SpecialSpan{0, 6}};
  EXPECT_EQ(refused(std::nullopt, "<call><call>", disordered), Rule::kSpecialToken);
  const std::array overlap = {tok::SpecialSpan{2, 6}};
  EXPECT_EQ(refused(call, "xx<call>", overlap), Rule::kSpecialToken);  // inside the selected
  const std::array empty = {tok::SpecialSpan{0, 0}};
  EXPECT_EQ(refused(std::nullopt, "<call>", empty), Rule::kSpecialToken);
  options.specials = partial;
  EXPECT_EQ(Failed(tokenizer.ForcedCoverBytes("<call>", options), &tok::Error::rule),
            Rule::kSpecialToken);
  options.marked = false;
  const std::array whole = {tok::SpecialSpan{0, 6}};
  EXPECT_EQ(refused(std::nullopt, "<call>", whole), Rule::kUnsupported);
  options.specials = whole;
  EXPECT_EQ(Failed(tokenizer.ForcedCoverBytes("<call>", options), &tok::Error::rule),
            Rule::kUnsupported);
}

TEST(ForcedSpan, ControlSpansAndControlTextFollowTheParsingMatcher) {
  Vocab v;
  v.Add("<a>", TokenKind::kControl);
  v.Add("<a>b", TokenKind::kUserDefined);
  v.Add("<c>", TokenKind::kControl);
  auto tokenizer = v.Build();
  auto spans = tokenizer.ControlSpans("x<a>y<a>b<c>");
  ASSERT_TRUE(spans);
  ASSERT_EQ(spans->size(), 2U);  // the longer user-defined "<a>b" hides a control
  EXPECT_EQ((*spans)[0].offset, 1U);
  EXPECT_EQ((*spans)[0].length, 3U);
  EXPECT_EQ((*spans)[1].offset, 9U);
  EXPECT_EQ((*spans)[1].length, 3U);
  auto none = tokenizer.ControlSpans("<a");
  ASSERT_TRUE(none);
  EXPECT_TRUE(none->empty());
  const auto contains = [&](std::string_view text) {
    auto found = tokenizer.ContainsControlText(text);
    EXPECT_TRUE(found);
    return found.value_or(false);
  };
  EXPECT_FALSE(contains(""));
  EXPECT_FALSE(contains("x<a"));
  EXPECT_FALSE(contains("<c"));
  EXPECT_TRUE(contains("x<a>"));
  EXPECT_TRUE(contains("<a>b"));  // a substring, even inside a user-defined token
  EXPECT_TRUE(contains("zz<c>zz"));
  llmp::base::WorkPulse pulse;
  pulse.Cancel();
  llmp::base::SetThreadPulse(&pulse);
  const std::string long_text(std::size_t{1} << 17U, '<');
  const auto cancelled = tokenizer.ContainsControlText(long_text);
  const auto cancelled_spans = tokenizer.ControlSpans(long_text);
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(Failed(cancelled, &tok::Error::rule), Rule::kCancelled);
  EXPECT_EQ(Failed(cancelled_spans, &tok::Error::rule), Rule::kCancelled);
}

TEST(ForcedSpan, AnOpenAtomPrefixHoldsBlocksItsClosedFragmentWouldSplitDifferently) {
  Vocab v;
  const auto spaces = v.Merge(" ", " ");
  const auto call = v.Add("<call>", TokenKind::kControl);
  auto tokenizer = v.Build();
  // If "<call>" does follow, the fragment "a  " ends in one two-space block.
  EXPECT_EQ(Encode(tokenizer, "a  <call>", {.special = tok::SpecialTokens::kParse}),
            (std::vector<TokenId>{Byte('a'), spaces, call}));
  auto plan = tokenizer.PlanForcedSpan(std::nullopt, "a  <c", ForcedOptions());
  ASSERT_TRUE(plan) << Failed(plan, &tok::Error::ToString).value_or("");
  EXPECT_EQ(plan->tokens, (std::vector<TokenId>{Byte('a')}));
  EXPECT_EQ(plan->held, "  <c");
  // Blocks the closed fragment reproduces are still committed.
  auto kept = tokenizer.PlanForcedSpan(std::nullopt, "abc<c", ForcedOptions());
  ASSERT_TRUE(kept);
  EXPECT_EQ(ForcedText(tokenizer, kept->tokens), "abc");
  EXPECT_EQ(kept->held, "<c");
  // A prefix only of a control token holds nothing once control text is text.
  auto options = ForcedOptions();
  options.marked = true;
  auto marked = tokenizer.PlanForcedSpan(std::nullopt, "a  <c", options);
  ASSERT_TRUE(marked);
  EXPECT_EQ(ForcedText(tokenizer, marked->tokens), "a  <");
  EXPECT_EQ(marked->held, "c");
}

TEST(ForcedSpan, ForcedBytesThatFinishAPublishedCharacterAreHeldNotRefused) {
  Vocab v;
  const auto tail = v.Merge("\xbd", "\x9c");  // splits U+FF5C, as in DeepSeek's markers
  auto tokenizer = v.Build();
  const std::string forced = "\xef\xbd\x9c!";  // "｜!"
  EXPECT_EQ(Encode(tokenizer, forced), (std::vector<TokenId>{Byte('\xef'), tail, Byte('!')}));
  // Masking: every candidate whose bytes run along the forced text is planned,
  // from every offset an accepted candidate can publish.
  const auto options = ForcedOptions();
  std::vector<bool> reached(forced.size() + 1, false);
  std::vector<std::size_t> pending{0};
  reached[0] = true;
  while (!pending.empty()) {
    const auto at = pending.back();
    pending.pop_back();
    if (at == forced.size()) continue;
    const auto rest = std::string_view(forced).substr(at);
    auto cover = tokenizer.ForcedCoverBytes(rest, options);
    ASSERT_TRUE(cover) << at << ": " << Failed(cover, &tok::Error::ToString).value_or("");
    bool accepted = false;
    for (TokenId id = 0; std::cmp_less(id, tokenizer.size()); ++id) {
      if (!tokenizer.CoversForced(id, *cover)) continue;
      const auto piece = ForcedText(tokenizer, std::span(&id, 1));
      if (piece.empty() || !rest.starts_with(piece)) continue;
      std::size_t next = forced.size();
      if (piece.size() < rest.size()) {
        auto plan = tokenizer.PlanForcedSpan(id, rest.substr(piece.size()), options);
        ASSERT_TRUE(plan) << at << ": " << Failed(plan, &tok::Error::ToString).value_or("");
        if (plan->selected_covered && !plan->tokens.empty())
          next = at + plan->committed_bytes;
        else if (plan->selected_matches_canonical_prefix)
          next = at + piece.size();
        else
          continue;
      }
      accepted = true;
      if (!reached[next]) {
        reached[next] = true;
        pending.push_back(next);
      }
    }
    EXPECT_TRUE(accepted) << at;
  }
  EXPECT_TRUE(reached[1]);  // the lead byte alone was published
  EXPECT_TRUE(reached[forced.size()]);
  // A token allowance can end a committed span inside a character too.
  Vocab w;
  const auto lead = w.Merge("\xef", "\xbd");
  auto limited = w.Build();
  auto cut_options = ForcedOptions(1);
  cut_options.terminal = true;
  auto cut = limited.PlanForcedSpan(std::nullopt,
                                    "\xef\xbd\x9c"
                                    "a b",
                                    cut_options);
  ASSERT_TRUE(cut) << Failed(cut, &tok::Error::ToString).value_or("");
  EXPECT_EQ(cut->tokens, (std::vector<TokenId>{lead}));
  EXPECT_TRUE(cut->limited);
  EXPECT_EQ(cut->held,
            "\x9c"
            "a b");
  EXPECT_EQ(cut->cover_bytes, 0U);
  // More continuation bytes than one character can hold are refused.
  EXPECT_EQ(Failed(limited.ForcedCoverBytes("\x9c\x9c\x9c\x9c", cut_options), &tok::Error::rule),
            Rule::kInvalidUtf8);
}
