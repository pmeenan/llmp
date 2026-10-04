// SPDX-FileCopyrightText: 2026 jitLLM contributors
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

#include "base/work_pulse.h"
#include "expected_error.h"
#include "tokenizer/error.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

namespace {

namespace tok = jitllm::tokenizer;
namespace uni = jitllm::tokenizer::unicode;
using jitllm::test_support::Failed;
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

std::vector<std::string> Pieces(PreTokenizer p, std::string_view text) {
  std::vector<char32_t> cps;
  EXPECT_TRUE(uni::DecodeUtf8(text, cps).has_value());
  std::vector<std::uint32_t> lengths;
  tok::PreTokenize(p, cps, lengths);
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
  const PreTokenizer q = PreTokenizer::kQwen2;
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
  const PreTokenizer q = PreTokenizer::kQwen35;
  EXPECT_EQ(Pieces(q, "éx"), (Strings{"éx"}));
  // A mark is no longer a symbol: it ends the symbol run and starts a word.
  EXPECT_EQ(Pieces(q, "क्ष !́"), (Strings{"क्ष", " !", "́"}));
  EXPECT_EQ(Pieces(PreTokenizer::kQwen2, "क्ष !́"), (Strings{"क", "्ष", " !́"}));
  EXPECT_EQ(Pieces(q, " ́́"), (Strings{" ́́"}));
}

TEST(PreTokenize, ContractionsMatchAsciiLettersOfEitherCase) {
  const PreTokenizer q = PreTokenizer::kQwen35;
  EXPECT_EQ(Pieces(q, "a'Sx"), (Strings{"a", "'S", "x"}));
  EXPECT_EQ(Pieces(q, "a'lLx"), (Strings{"a", "'lL", "x"}));
  EXPECT_EQ(Pieces(q, "a'ſx"), (Strings{"a", "'ſx"}));  // long s is no s
  EXPECT_EQ(Pieces(q, "a'Kx"), (Strings{"a", "'Kx"}));
  EXPECT_EQ(Pieces(q, "'"), (Strings{"'"}));
  EXPECT_EQ(Pieces(q, "'r"), (Strings{"'r"}));  // 're needs its e: the word takes ' as its prefix
}

TEST(PreTokenize, DeepSeekV3Stages) {
  const PreTokenizer d = PreTokenizer::kDeepSeekV3;
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

TEST(Bpe, MergesByRankThenLeftmost) {
  Vocab v;
  v.spec.pre_tokenizer = PreTokenizer::kQwen2;
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

// The windows' cut rule as tokenizer.h documents it (kEncodeWindowBytes),
// written independently: before an ASCII letter or digit after a newline,
// or before a space between an ASCII letter or digit and an ASCII letter.
bool DocumentedCut(std::u32string_view t, std::size_t p) {
  const auto letter = [](char32_t c) {
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
  };
  const auto alnum = [&](char32_t c) { return letter(c) || (c >= U'0' && c <= U'9'); };
  if (t[p - 1] == U'\n') {
    return alnum(t[p]);
  }
  return t[p] == U' ' && p + 1 < t.size() && alnum(t[p - 1]) && letter(t[p + 1]);
}

std::vector<std::uint32_t> PieceLengths(PreTokenizer p, std::u32string_view cps) {
  std::vector<std::uint32_t> lengths;
  tok::PreTokenize(p, std::span(cps.data(), cps.size()), lengths);
  return lengths;
}

// Every cut point splits the pieces of every pre-tokenizer exactly where
// the halves' own pieces meet, and NFC of the halves is NFC of the whole:
// so encoding window by window gives the whole text's tokens.
TEST(Encode, WindowCutsSplitNoPieceAndNoComposition) {
  // Code points by value, so nothing invisible sits in the source: e and a
  // combining acute, the acute alone, CJK, katakana, a zero-width joiner,
  // Hangul jamo L and V (composing), an emoji, an Arabic-Indic digit, and
  // a precomposed e acute.
  const auto cps = [](std::initializer_list<char32_t> c) { return std::u32string(c); };
  const std::array<std::u32string, 28> atoms = {U"a",
                                                U"Zed",
                                                U"q",
                                                U"7",
                                                U"421",
                                                U" ",
                                                U"  ",
                                                U"\n",
                                                U"\r\n",
                                                U"\n\n",
                                                U"\t",
                                                U"'s",
                                                U"'LL",
                                                U"!",
                                                U"#inc",
                                                U"...",
                                                cps({0x65, 0x301}),
                                                cps({0x301}),
                                                cps({0x4E2D, 0x6587}),
                                                cps({0x30C6}),
                                                cps({0x200D}),
                                                cps({0x1100, 0x1161}),
                                                cps({0x1F600}),
                                                cps({0x663}),
                                                U" !!",
                                                U"x",
                                                cps({0xE9}),
                                                U"\n "};
  // NOLINTNEXTLINE(bugprone-random-generator-seed): reproducible
  std::mt19937 rng(20261003);
  std::uniform_int_distribution<std::size_t> pick(0, atoms.size() - 1);
  std::uniform_int_distribution<std::size_t> count(1, 120);
  std::size_t cuts = 0;
  for (int round = 0; round < 400; ++round) {
    std::u32string text;
    for (std::size_t n = count(rng); n > 0; --n) {
      text += atoms[pick(rng)];
    }
    for (std::size_t p = 1; p < text.size(); ++p) {
      if (!DocumentedCut(text, p)) {
        continue;
      }
      ++cuts;
      const std::u32string_view whole(text);
      for (const PreTokenizer pre :
           {PreTokenizer::kQwen2, PreTokenizer::kQwen35, PreTokenizer::kDeepSeekV3}) {
        std::vector<std::uint32_t> halves = PieceLengths(pre, whole.substr(0, p));
        const std::vector<std::uint32_t> right = PieceLengths(pre, whole.substr(p));
        halves.insert(halves.end(), right.begin(), right.end());
        ASSERT_EQ(halves, PieceLengths(pre, whole))
            << tok::PreTokenizerName(pre) << " at " << p << " of " << Utf8(text);
      }
      std::vector<char32_t> nfc(text.begin(), text.end());
      std::vector<char32_t> left(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(p));
      std::vector<char32_t> right(text.begin() + static_cast<std::ptrdiff_t>(p), text.end());
      uni::ToNfc(nfc);
      uni::ToNfc(left);
      uni::ToNfc(right);
      left.insert(left.end(), right.begin(), right.end());
      ASSERT_EQ(left, nfc) << "NFC at " << p << " of " << Utf8(text);
    }
  }
  EXPECT_GT(cuts, 300U);  // the texts had cut points to check
}

// A text past kEncodeWindowBytes encodes window by window, with the tokens
// of its lines encoded one at a time (each line under a window), including
// a stretch longer than a window with no cut point (scanned forward), under
// NFC and each pre-tokenizer; its working set follows the widest window.
TEST(Encode, WindowedEncodingMatchesLineByLine) {
  std::string text;
  while (text.size() < 2 * tok::kEncodeWindowBytes) {
    text += "hello world 123 h\xC3\xA9llo e\xCC\x81 don't\n";
    text += "  indented!!\n\nWell said\r\n9 lives\n";
  }
  const std::size_t stretch = tok::kEncodeWindowBytes + (tok::kEncodeWindowBytes / 2);
  text += std::string(stretch, '!');
  text += "\nlast line";
  // The widest window: from the cut in "9| lives\n" through the stretch to
  // the cut before "last" (7 bytes before it, 1 after).
  EXPECT_EQ(tok::Tokenizer::EncodeWorkingBytes(text),
            std::uint64_t{stretch + 8} * tok::kEncodeBytesPerWindowByte);
  EXPECT_EQ(tok::Tokenizer::EncodeWorkingBytes("short text"),
            std::uint64_t{10} * tok::kEncodeBytesPerWindowByte);
  for (const PreTokenizer pre :
       {PreTokenizer::kQwen2, PreTokenizer::kQwen35, PreTokenizer::kDeepSeekV3}) {
    Vocab v;
    v.spec.pre_tokenizer = pre;
    v.spec.normalization = tok::Normalization::kNfc;
    v.Merge("h", "e");
    v.Merge("l", "l");
    v.Merge("he", "ll");
    v.Merge(" ", "w");
    v.Merge("!", "!");
    v.Merge("\n", "\n");
    const tok::Tokenizer t = v.Build();
    tok::EncodeOptions o;
    o.max_bytes = text.size();
    o.max_tokens = text.size();
    const auto whole = Encode(t, text, o);
    std::vector<TokenId> lines;
    std::size_t start = 0;
    for (std::size_t p = 1; p <= text.size(); ++p) {
      const bool cut = p < text.size() && text[p - 1] == '\n' &&
                       std::isalnum(static_cast<unsigned char>(text[p])) != 0;
      if (cut || p == text.size()) {
        const auto line = Encode(t, std::string_view(text).substr(start, p - start), o);
        lines.insert(lines.end(), line.begin(), line.end());
        start = p;
      }
    }
    EXPECT_EQ(whole, lines) << tok::PreTokenizerName(pre);
  }
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
  spec.pre_tokenizer = PreTokenizer::kGemma4;
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
  for (std::size_t i = 0; i < tok::kEncodeWindowBytes / 2; ++i) {
    text += "a a\n";
  }
  EXPECT_EQ(t->WorkingBytes(text), text.size() * tok::kEncodeBytesPerWindowByte);
  EXPECT_GT(t->WorkingBytes(text), tok::Tokenizer::EncodeWorkingBytes(text));
  jitllm::base::WorkPulse pulse;
  jitllm::base::SetThreadPulse(&pulse);
  const auto ids = Encode(*t, text);
  jitllm::base::SetThreadPulse(nullptr);
  EXPECT_GT(pulse.beats(), 2U);
  ASSERT_EQ(ids.size(), tok::kEncodeWindowBytes);
  for (std::size_t i = 0; i < ids.size(); i += 2) {
    ASSERT_EQ(ids[i], 259);
    ASSERT_EQ(ids[i + 1], 260);
  }
  std::vector<TokenId> out;
  pulse.Cancel();
  jitllm::base::SetThreadPulse(&pulse);
  auto cancelled = t->Encode("a a", {}, out);
  jitllm::base::SetThreadPulse(nullptr);
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
  jitllm::base::WorkPulse pulse;
  pulse.Cancel();
  std::vector<TokenId> out;
  jitllm::base::SetThreadPulse(&pulse);
  const auto parsed = t->Encode("<bos><user>", {.special = tok::SpecialTokens::kParse}, out);
  const std::array spans = {tok::SpecialSpan{0, 5}};
  const auto marked = t->EncodeMarked("<bos>", spans, {}, out);
  jitllm::base::SetThreadPulse(nullptr);
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
  jitllm::base::SetThreadPulse(&pulse);
  const auto progress = t->Encode(text, {.special = tok::SpecialTokens::kParse}, out);
  jitllm::base::SetThreadPulse(nullptr);
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
    EXPECT_EQ(g->spec.pre_tokenizer, PreTokenizer::kGemma4);
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

TEST(SentencePiece, PretokenizationKeepsWholeNonemptyFragment) {
  std::vector<std::uint32_t> lengths;
  tok::PreTokenize(PreTokenizer::kSentencePiece, {}, lengths);
  EXPECT_TRUE(lengths.empty());
  const std::u32string_view fragment = U"a b";
  tok::PreTokenize(PreTokenizer::kSentencePiece, fragment, lengths);
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
  for (std::size_t i = 0; i < tok::kEncodeWindowBytes; ++i) {
    text += "a ";
  }
  EXPECT_EQ(t->WorkingBytes(text), (text.size() + 1) * tok::kEncodeBytesPerWindowByte);
  EXPECT_GT(t->WorkingBytes(text), tok::Tokenizer::EncodeWorkingBytes(text));
  jitllm::base::WorkPulse pulse;
  jitllm::base::SetThreadPulse(&pulse);
  std::vector<TokenId> ids;
  const auto encoded = t->Encode(text, {}, ids);
  jitllm::base::SetThreadPulse(nullptr);
  ASSERT_TRUE(encoded);
  EXPECT_GT(pulse.beats(), 40U);
  EXPECT_EQ(ids.size(), text.size());
  pulse.Cancel();
  jitllm::base::SetThreadPulse(&pulse);
  const auto refused = t->Encode("a", {}, ids);
  jitllm::base::SetThreadPulse(nullptr);
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
  EXPECT_EQ(g->spec.pre_tokenizer, PreTokenizer::kQwen35);
  EXPECT_EQ(g->chat_template, "{{ messages }}");
  EXPECT_EQ(g->spec.tokens.size(), 259U);
  EXPECT_EQ(g->spec.kinds[258], TokenKind::kUnused);
  EXPECT_EQ(g->spec.eos, 257);
  auto t = tok::Tokenizer::Create(g->spec);
  ASSERT_TRUE(t.has_value()) << t.error().ToString();
  EXPECT_EQ(Encode(*t, "ab"), (std::vector<TokenId>{256}));
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
  for (const auto p : {PreTokenizer::kQwen2, PreTokenizer::kQwen35, PreTokenizer::kDeepSeekV3}) {
    v.spec.pre_tokenizer = p;
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
  EXPECT_EQ(rule(SmallGguf("llama-bpe")), Rule::kUnsupported);
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

TEST(HfTokenizer, ReadsASupportedFile) {
  const auto spec = tok::ReadHfTokenizer(HfJson());
  ASSERT_TRUE(spec.has_value()) << spec.error().ToString();
  EXPECT_EQ(spec->pre_tokenizer, PreTokenizer::kQwen35);
  EXPECT_EQ(spec->normalization, tok::Normalization::kNfc);
  ASSERT_EQ(spec->tokens.size(), 259U);
  EXPECT_EQ(spec->kinds[257], TokenKind::kControl);
  EXPECT_EQ(spec->kinds[258], TokenKind::kUserDefined);
  auto t = tok::Tokenizer::Create(*spec);
  ASSERT_TRUE(t.has_value()) << t.error().ToString();
  EXPECT_EQ(Encode(*t, "ab<think>"), (std::vector<TokenId>{256, 258}));
}

TEST(HfTokenizer, RefusesWhatItDoesNotImplement) {
  auto rule = [](const std::string& json) {
    return Failed(tok::ReadHfTokenizer(json), &tok::Error::rule);
  };
  EXPECT_EQ(rule(HfJson(R"({"type": "NFKC"})")), Rule::kUnsupported);
  EXPECT_EQ(rule(HfJson(R"({"type": "NFC"})", R"(\\p{L}+)")), Rule::kUnsupported);
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

}  // namespace
