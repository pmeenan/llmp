// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/hf.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"
#include "tokenizer/error.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/tokenizer.h"

namespace llmp::tokenizer {
namespace {

namespace json = base::json;

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t item = kNoItem) {
  return std::unexpected(Error{rule, reason, item});
}

// The Split expressions each pre-tokenizer stands for, as tokenizer.json
// holds them (after JSON unescaping).
constexpr std::string_view kQwen2Split =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr std::string_view kQwen35Split =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr std::string_view kDeepSeekDigits = R"re(\p{N}{1,3})re";
// [一-龥぀-ゟ゠-ヿ]+
constexpr std::string_view kDeepSeekCjk =
    "[\xE4\xB8\x80-\xE9\xBE\xA5\xE3\x81\x80-\xE3\x82\x9F\xE3\x82\xA0-\xE3\x83\xBF]+";
// With literal CR and LF inside the classes, as DeepSeek's file has them.
constexpr std::string_view kDeepSeekWords =
    R"re([!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^)re"
    "\r\n"
    R"re(\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[)re"
    "\r\n"
    R"re(]*|\s*[)re"
    "\r\n"
    R"re(]+|\s+(?!\S)|\s+)re";

std::optional<std::string_view> StringMember(json::Value object, std::string_view name) {
  const auto v = object.find(name);
  if (!v || !v->is_string()) {
    return std::nullopt;
  }
  return v->string();
}

// A member that must be absent, null or false.
bool Off(json::Value object, std::string_view name) {
  const auto v = object.find(name);
  return !v || v->is_null() || (v->is_bool() && !v->boolean());
}

// A member that must be absent, null or "".
bool Empty(json::Value object, std::string_view name) {
  const auto v = object.find(name);
  return !v || v->is_null() || (v->is_string() && v->string().empty());
}

std::expected<Normalization, Error> ReadNormalizer(json::Value root) {
  const auto n = root.find("normalizer");
  if (!n || n->is_null()) {
    return Normalization::kNone;
  }
  const auto type = StringMember(*n, "type");
  if (type == "NFC") {
    return Normalization::kNfc;
  }
  if (type == "Sequence") {
    const auto list = n->find("normalizers");
    if (list && list->is_array()) {
      if (list->size() == 0) {
        return Normalization::kNone;
      }
      if (list->size() == 1 && StringMember(list->at(0), "type") == "NFC") {
        return Normalization::kNfc;
      }
    }
  }
  return Fail(Rule::kUnsupported, "normalizer");
}

std::expected<PreTokenizer, Error> ReadPreTokenizer(json::Value root) {
  const auto p = root.find("pre_tokenizer");
  if (!p || StringMember(*p, "type") != "Sequence") {
    return Fail(Rule::kUnsupported, "pre-tokenizer other than a Sequence");
  }
  const auto list = p->find("pretokenizers");
  if (!list || !list->is_array() || list->size() < 2) {
    return Fail(Rule::kUnsupported, "pre-tokenizer sequence");
  }
  const json::Value last = list->at(list->size() - 1);
  if (StringMember(last, "type") != "ByteLevel" || !Off(last, "add_prefix_space") ||
      !Off(last, "use_regex")) {
    return Fail(Rule::kUnsupported, "pre-tokenizer sequence not ending in a plain ByteLevel");
  }
  std::vector<std::string_view> splits;
  for (std::size_t i = 0; i + 1 < list->size(); ++i) {
    const json::Value s = list->at(i);
    const auto pattern = s.find("pattern");
    const auto regex = pattern ? StringMember(*pattern, "Regex") : std::nullopt;
    if (StringMember(s, "type") != "Split" || !regex || StringMember(s, "behavior") != "Isolated" ||
        !Off(s, "invert")) {
      return Fail(Rule::kUnsupported, "pre-tokenizer stage other than an isolated regex Split", i);
    }
    splits.push_back(*regex);
  }
  if (splits.size() == 1 && splits[0] == kQwen2Split) {
    return PreTokenizer::kQwen2;
  }
  if (splits.size() == 1 && splits[0] == kQwen35Split) {
    return PreTokenizer::kQwen35;
  }
  if (splits.size() == 3 && splits[0] == kDeepSeekDigits && splits[1] == kDeepSeekCjk &&
      splits[2] == kDeepSeekWords) {
    return PreTokenizer::kDeepSeekV3;
  }
  return Fail(Rule::kUnsupported, "pre-tokenizer expressions");
}

}  // namespace

std::expected<TokenizerSpec, Error> ReadHfTokenizer(std::string_view text) {
  auto parsed = json::Parse(text);
  if (!parsed) {
    return Fail(Rule::kFormat, "tokenizer.json is not JSON", parsed.error().offset);
  }
  const json::Value root = parsed->root();
  if (!root.is_object()) {
    return Fail(Rule::kFormat, "tokenizer.json is not an object");
  }
  if (!Off(root, "truncation") || !Off(root, "padding")) {
    return Fail(Rule::kUnsupported, "truncation or padding");
  }
  TokenizerSpec spec;
  auto normalization = ReadNormalizer(root);
  if (!normalization) {
    return std::unexpected(normalization.error());
  }
  spec.normalization = *normalization;
  auto pre = ReadPreTokenizer(root);
  if (!pre) {
    return std::unexpected(pre.error());
  }
  spec.pre_tokenizer = *pre;
  const auto decoder = root.find("decoder");
  if (!decoder || StringMember(*decoder, "type") != "ByteLevel") {
    return Fail(Rule::kUnsupported, "decoder other than ByteLevel");
  }
  if (const auto post = root.find("post_processor");
      post && !post->is_null() && StringMember(*post, "type") != "ByteLevel") {
    return Fail(Rule::kUnsupported, "post-processor other than ByteLevel");
  }

  const auto model = root.find("model");
  if (!model || StringMember(*model, "type") != "BPE") {
    return Fail(Rule::kUnsupported, "model other than BPE");
  }
  if (!Off(*model, "dropout") || !Off(*model, "unk_token") ||
      !Empty(*model, "continuing_subword_prefix") || !Empty(*model, "end_of_word_suffix") ||
      !Off(*model, "fuse_unk") || !Off(*model, "byte_fallback")) {
    return Fail(Rule::kUnsupported, "BPE option");
  }
  if (const auto ignore = model->find("ignore_merges"); ignore && !ignore->is_null()) {
    if (!ignore->is_bool()) {
      return Fail(Rule::kFormat, "ignore_merges is not a boolean");
    }
    spec.ignore_merges = ignore->boolean();
  }
  const auto vocab = model->find("vocab");
  const auto merges = model->find("merges");
  const auto added = root.find("added_tokens");
  if (!vocab || !vocab->is_object() || !merges || !merges->is_array() || !added ||
      !added->is_array()) {
    return Fail(Rule::kFormat, "vocab, merges or added_tokens missing");
  }
  const std::size_t total = vocab->size() + added->size();
  if (total > kMaxVocabulary) {
    return Fail(Rule::kBounds, "vocabulary size", total);
  }
  std::vector<std::string> texts(total);
  std::vector<bool> present(total, false);
  std::vector<TokenKind> kinds(total, TokenKind::kNormal);
  std::size_t count = 0;
  for (std::size_t i = 0; i < vocab->size(); ++i) {
    const auto id = vocab->member(i).int64();
    if (!id || *id < 0 || std::cmp_greater_equal(*id, total) ||
        present[static_cast<std::size_t>(*id)]) {
      return Fail(Rule::kVocabulary, "vocabulary ID repeated or out of range", i);
    }
    texts[static_cast<std::size_t>(*id)] = std::string(vocab->key(i));
    present[static_cast<std::size_t>(*id)] = true;
    count = std::max(count, static_cast<std::size_t>(*id) + 1);
  }
  for (std::size_t i = 0; i < added->size(); ++i) {
    const json::Value a = added->at(i);
    std::optional<std::int64_t> id;
    if (const auto v = a.find("id")) {
      id = v->int64();
    }
    const auto content = StringMember(a, "content");
    const auto special = a.find("special");
    if (!id || *id < 0 || std::cmp_greater_equal(*id, total) || !content || !special ||
        !special->is_bool()) {
      return Fail(Rule::kFormat, "added token without an ID, content or special flag", i);
    }
    if (!Off(a, "lstrip") || !Off(a, "rstrip") || !Off(a, "single_word")) {
      return Fail(Rule::kUnsupported, "added token stripping or single-word matching", i);
    }
    if (!Off(a, "normalized") && spec.normalization != Normalization::kNone) {
      return Fail(Rule::kUnsupported, "added token matched after normalization", i);
    }
    const auto slot = static_cast<std::size_t>(*id);
    if (present[slot] && texts[slot] != *content) {
      return Fail(Rule::kVocabulary, "added token changes a vocabulary entry", i);
    }
    texts[slot] = std::string(*content);
    present[slot] = true;
    kinds[slot] = special->boolean() ? TokenKind::kControl : TokenKind::kUserDefined;
    count = std::max(count, slot + 1);
  }
  for (std::size_t id = 0; id < count; ++id) {
    if (!present[id]) {
      return Fail(Rule::kVocabulary, "token IDs are not dense", id);
    }
  }
  texts.resize(count);
  spec.tokens = std::move(texts);
  kinds.resize(count);
  spec.kinds = std::move(kinds);

  spec.merges.reserve(merges->size());
  for (std::size_t i = 0; i < merges->size(); ++i) {
    const json::Value m = merges->at(i);
    if (m.is_string()) {
      const std::string_view s = m.string();
      const std::size_t space = s.find(' ');
      if (space == 0 || space == std::string_view::npos || space + 1 == s.size() ||
          s.find(' ', space + 1) != std::string_view::npos) {
        return Fail(Rule::kVocabulary, "a merge is not two tokens separated by one space", i);
      }
      spec.merges.emplace_back(std::string(s.substr(0, space)), std::string(s.substr(space + 1)));
    } else if (m.is_array() && m.size() == 2 && m.at(0).is_string() && m.at(1).is_string()) {
      spec.merges.emplace_back(std::string(m.at(0).string()), std::string(m.at(1).string()));
    } else {
      return Fail(Rule::kFormat, "a merge is neither a string nor a pair", i);
    }
  }
  return spec;
}

}  // namespace llmp::tokenizer
