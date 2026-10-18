// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/hf.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
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

bool Only(json::Value value, std::initializer_list<std::string_view> names) {
  if (!value.is_object()) return false;
  for (std::size_t i = 0; i < value.size(); ++i)
    if (std::ranges::find(names, value.key(i)) == names.end()) return false;
  return true;
}

bool Replace(json::Value value, std::string_view from, std::string_view to) {
  const auto pattern = value.find("pattern");
  return Only(value, {"type", "pattern", "content"}) && StringMember(value, "type") == "Replace" &&
         pattern && Only(*pattern, {"String"}) && StringMember(*pattern, "String") == from &&
         StringMember(value, "content") == to;
}

std::expected<void, Error> ReadRawPipeline(json::Value root, TokenizerSpec& spec) {
  const auto normalizer = root.find("normalizer");
  if (!normalizer || !Replace(*normalizer, " ", "▁"))
    return Fail(Rule::kUnsupported, "raw BPE normalizer other than space replacement");
  const auto pre = root.find("pre_tokenizer");
  if (pre && !pre->is_null()) {
    const auto pattern = pre->find("pattern");
    if (!Only(*pre, {"type", "pattern", "behavior", "invert"}) ||
        StringMember(*pre, "type") != "Split" || !pattern || !Only(*pattern, {"String"}) ||
        StringMember(*pattern, "String") != " " ||
        StringMember(*pre, "behavior") != "MergedWithPrevious" || !Off(*pre, "invert"))
      return Fail(Rule::kUnsupported, "raw BPE pre-tokenizer");
    // Space replacement runs first and leaves no literal spaces. Therefore
    // this declared Split has no matches; the model receives the whole fragment.
  }
  const auto decoder = root.find("decoder");
  const auto list = decoder ? decoder->find("decoders") : std::nullopt;
  if (!decoder || !Only(*decoder, {"type", "decoders"}) ||
      StringMember(*decoder, "type") != "Sequence" || !list || !list->is_array() ||
      list->size() != 3 || !Replace(list->at(0), "▁", " ") || !Only(list->at(1), {"type"}) ||
      StringMember(list->at(1), "type") != "ByteFallback" || !Only(list->at(2), {"type"}) ||
      StringMember(list->at(2), "type") != "Fuse")
    return Fail(Rule::kUnsupported, "raw BPE decoder sequence");
  spec.pre_tokenizer = PreTokenizer::kRawBpe;
  spec.raw_bpe_split_newlines = false;
  spec.decode_added_whitespace = true;
  return {};
}

std::expected<void, Error> ReadRawPostProcessor(json::Value root, TokenizerSpec& spec) {
  const auto post = root.find("post_processor");
  if (!post || post->is_null()) return {};
  if (!Only(*post, {"type", "single", "pair", "special_tokens"}) ||
      StringMember(*post, "type") != "TemplateProcessing")
    return Fail(Rule::kUnsupported, "raw BPE post-processor other than TemplateProcessing");
  const auto definitions = post->find("special_tokens");
  const auto single = post->find("single");
  if (!definitions || !definitions->is_object() || !single || !single->is_array())
    return Fail(Rule::kFormat, "TemplateProcessing definitions or single template missing");
  for (std::size_t i = 0; i < definitions->size(); ++i) {
    const auto definition = definitions->member(i);
    const auto ids = definition.find("ids");
    const auto tokens = definition.find("tokens");
    if (!Only(definition, {"id", "ids", "tokens"}) ||
        StringMember(definition, "id") != definitions->key(i) || !ids || !ids->is_array() ||
        !tokens || !tokens->is_array())
      return Fail(Rule::kFormat, "malformed TemplateProcessing special token", i);
    if (ids->size() != 1 || tokens->size() != 1)
      return Fail(Rule::kUnsupported, "multi-token TemplateProcessing special token", i);
    const auto id = ids->at(0).int64();
    if (!id || *id < 0 || std::cmp_greater_equal(*id, spec.tokens.size()) ||
        !tokens->at(0).is_string() ||
        spec.tokens[static_cast<std::size_t>(*id)] != tokens->at(0).string())
      return Fail(Rule::kVocabulary, "TemplateProcessing special token ID/text mismatch", i);
  }
  const auto read_template = [&](json::Value list, bool pair) -> std::expected<void, Error> {
    if (!list.is_array()) return Fail(Rule::kFormat, "TemplateProcessing template is not an array");
    if (list.size() > (pair ? 6U : 3U))
      return Fail(Rule::kUnsupported, "template exceeds single BOS/EOS wrappers");
    unsigned sequences = 0;
    std::optional<TokenId> prefix, suffix;
    for (std::size_t i = 0; i < list.size(); ++i) {
      const auto entry = list.at(i);
      if (!entry.is_object() || entry.size() != 1)
        return Fail(Rule::kFormat, "TemplateProcessing item is not a tagged object", i);
      const bool sequence = entry.key(0) == "Sequence";
      if (!sequence && entry.key(0) != "SpecialToken")
        return Fail(Rule::kUnsupported, "TemplateProcessing item kind", i);
      const auto item = entry.member(0);
      const auto name = StringMember(item, "id");
      const auto type = item.find("type_id");
      const auto id = type ? type->int64() : std::nullopt;
      if (!Only(item, {"id", "type_id"}) || !name || !id || *id < 0 || *id > 0xFFFFFFFFLL)
        return Fail(Rule::kFormat, "TemplateProcessing item has invalid fields", i);
      if (sequence) {
        const unsigned bit = *name == "A" ? 1U : pair && *name == "B" ? 2U : 0U;
        if (bit == 0 || (sequences & bit) != 0)
          return Fail(Rule::kUnsupported, "TemplateProcessing repeated or unknown sequence", i);
        sequences |= bit;
      } else {
        const auto definition = definitions->find(*name);
        if (!definition)
          return Fail(Rule::kVocabulary, "undefined TemplateProcessing special token", i);
        const auto token = static_cast<TokenId>(*definition->find("ids")->at(0).int64());
        if (!pair) {
          auto& insertion = sequences == 0 ? prefix : suffix;
          if (insertion) return Fail(Rule::kUnsupported, "multiple BOS or EOS template tokens", i);
          insertion = token;
        }
      }
    }
    if (sequences != (pair ? 3U : 1U))
      return Fail(Rule::kUnsupported, "TemplateProcessing sequence is missing");
    if (!pair) {
      spec.bos = prefix;
      spec.eos = suffix;
      spec.add_bos = prefix.has_value();
      spec.add_eos = suffix.has_value();
    }
    return {};
  };
  if (auto value = read_template(*single, false); !value) return value;
  if (const auto pair = post->find("pair"); pair) return read_template(*pair, true);
  return {};
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
  const auto decoder = root.find("decoder");
  const bool raw = decoder && StringMember(*decoder, "type") == "Sequence";
  if (raw) {
    if (auto pipeline = ReadRawPipeline(root, spec); !pipeline)
      return std::unexpected(pipeline.error());
  } else {
    auto normalization = ReadNormalizer(root);
    if (!normalization) return std::unexpected(normalization.error());
    spec.normalization = *normalization;
    auto pre = ReadPreTokenizer(root);
    if (!pre) return std::unexpected(pre.error());
    spec.pre_tokenizer = *pre;
    if (!decoder || StringMember(*decoder, "type") != "ByteLevel")
      return Fail(Rule::kUnsupported, "decoder other than ByteLevel");
    if (const auto post = root.find("post_processor");
        post && !post->is_null() && StringMember(*post, "type") != "ByteLevel")
      return Fail(Rule::kUnsupported, "post-processor other than ByteLevel");
  }

  const auto model = root.find("model");
  if (!model || StringMember(*model, "type") != "BPE") {
    return Fail(Rule::kUnsupported, "model other than BPE");
  }
  if (!Off(*model, "dropout") || (!raw && !Off(*model, "unk_token")) ||
      !Empty(*model, "continuing_subword_prefix") || !Empty(*model, "end_of_word_suffix") ||
      (!raw && (!Off(*model, "fuse_unk") || !Off(*model, "byte_fallback")))) {
    return Fail(Rule::kUnsupported, "BPE option");
  }
  if (raw) {
    const auto fallback = model->find("byte_fallback");
    const auto fuse = model->find("fuse_unk");
    if (!fallback || !fallback->is_bool() || !fallback->boolean() || (fuse && !fuse->is_bool()) ||
        !StringMember(*model, "unk_token"))
      return Fail(Rule::kUnsupported, "raw BPE requires byte fallback and a named unknown token");
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
  std::vector<bool> model_tokens(raw ? total : 0, false);
  std::vector<bool> normalized_tokens(raw ? total : 0, false);
  std::size_t count = 0;
  for (std::size_t i = 0; i < vocab->size(); ++i) {
    const auto id = vocab->member(i).int64();
    if (!id || *id < 0 || std::cmp_greater_equal(*id, total) ||
        present[static_cast<std::size_t>(*id)]) {
      return Fail(Rule::kVocabulary, "vocabulary ID repeated or out of range", i);
    }
    texts[static_cast<std::size_t>(*id)] = std::string(vocab->key(i));
    present[static_cast<std::size_t>(*id)] = true;
    if (raw) {
      const auto index = static_cast<std::size_t>(*id);
      model_tokens[index] = true;
      const auto token = vocab->key(i);
      if (token.size() == 6 && token.starts_with("<0x") && token.back() == '>')
        kinds[index] = TokenKind::kByte;
    }
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
    const auto normalized = a.find("normalized");
    if (normalized && !normalized->is_bool())
      return Fail(Rule::kFormat, "added token normalized flag is not a boolean", i);
    if (!raw && !Off(a, "normalized") && spec.normalization != Normalization::kNone)
      return Fail(Rule::kUnsupported, "added token matched after normalization", i);
    const auto slot = static_cast<std::size_t>(*id);
    if (present[slot] && texts[slot] != *content) {
      return Fail(Rule::kVocabulary, "added token changes a vocabulary entry", i);
    }
    texts[slot] = std::string(*content);
    present[slot] = true;
    kinds[slot] = special->boolean() ? TokenKind::kControl : TokenKind::kUserDefined;
    if (raw) {
      if (special->boolean()) model_tokens[slot] = false;
      normalized_tokens[slot] = normalized && normalized->boolean();
    }
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
  if (raw) {
    model_tokens.resize(count);
    spec.model_tokens = std::move(model_tokens);
    normalized_tokens.resize(count);
    spec.normalized_tokens = std::move(normalized_tokens);
    const auto unknown = StringMember(*model, "unk_token");
    if (std::ranges::find(spec.tokens, *unknown) == spec.tokens.end())
      return Fail(Rule::kVocabulary, "raw BPE unknown token is absent");
  }

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
  if (raw) {
    if (auto post = ReadRawPostProcessor(root, spec); !post) return std::unexpected(post.error());
  }
  return spec;
}

std::expected<void, Error> ApplyHfTokenizerConfig(std::string_view text, TokenizerSpec& spec) {
  auto parsed = json::Parse(text);
  if (!parsed || !parsed->root().is_object())
    return Fail(Rule::kFormat, "tokenizer_config is not a JSON object");
  const auto root = parsed->root();
  const auto token_text = [](json::Value value) -> std::optional<std::string_view> {
    if (value.is_string()) return value.string();
    return value.is_object() ? StringMember(value, "content") : std::nullopt;
  };
  if (const auto kwargs = root.find("sp_model_kwargs");
      kwargs && !kwargs->is_null() && (!kwargs->is_object() || kwargs->size() != 0))
    return Fail(Rule::kUnsupported, "SentencePiece sampling options");
  if (const auto added = root.find("added_tokens_decoder"); added && !added->is_null()) {
    if (!added->is_object()) return Fail(Rule::kFormat, "added_tokens_decoder is not an object");
    std::size_t count = spec.tokens.size();
    const auto id_of = [](std::string_view name) -> std::optional<std::size_t> {
      std::size_t id = 0;
      const auto read = std::from_chars(name.data(), name.data() + name.size(), id);
      if (read.ec != std::errc{} || read.ptr != name.data() + name.size() || id >= kMaxVocabulary)
        return std::nullopt;
      return id;
    };
    for (std::size_t i = 0; i < added->size(); ++i) {
      const auto id = id_of(added->key(i));
      if (!id) return Fail(Rule::kVocabulary, "added_tokens_decoder ID outside the vocabulary");
      count = std::max(count, *id + 1);
    }
    spec.tokens.resize(count);
    spec.kinds.resize(count, TokenKind::kUnused);
    if (!spec.model_tokens.empty()) spec.model_tokens.resize(count, false);
    if (!spec.model_symbols.empty()) spec.model_symbols.resize(count, false);
    if (!spec.added_tokens.empty()) spec.added_tokens.resize(count, false);
    if (!spec.normalized_tokens.empty() || spec.pre_tokenizer == PreTokenizer::kRawBpe)
      spec.normalized_tokens.resize(count, false);
    if (!spec.scores.empty()) spec.scores.resize(count, 0);
    for (std::size_t i = 0; i < added->size(); ++i) {
      const auto id = *id_of(added->key(i));
      const auto token = added->member(i);
      const auto content = token.is_object() ? StringMember(token, "content") : std::nullopt;
      const auto special = token.find("special");
      if (!content || !special || !special->is_bool())
        return Fail(Rule::kFormat, "added_tokens_decoder token without content or special flag",
                    id);
      if (!Off(token, "lstrip") || !Off(token, "rstrip") || !Off(token, "single_word"))
        return Fail(Rule::kUnsupported, "added token stripping or single-word matching", id);
      const auto normalized = token.find("normalized");
      if (normalized && !normalized->is_bool())
        return Fail(Rule::kFormat, "added token normalized flag is not a boolean", id);
      if (spec.pre_tokenizer != PreTokenizer::kRawBpe && !Off(token, "normalized") &&
          (spec.normalization != Normalization::kNone ||
           content->find(' ') != std::string_view::npos))
        return Fail(Rule::kUnsupported, "added token matched after normalization", id);
      if (!spec.normalized_tokens.empty())
        spec.normalized_tokens[id] = normalized && normalized->boolean();
      if (!spec.tokens[id].empty() && spec.tokens[id] != *content)
        return Fail(Rule::kVocabulary, "added_tokens_decoder changes a token's text", id);
      spec.tokens[id] = std::string(*content);
      if (!spec.added_tokens.empty()) spec.added_tokens[id] = true;
      spec.kinds[id] = special->boolean() ? TokenKind::kControl : TokenKind::kUserDefined;
      if (!spec.model_tokens.empty() && special->boolean()) spec.model_tokens[id] = false;
    }
    for (std::size_t i = 0; i < count; ++i)
      if (spec.tokens[i].empty())
        return Fail(Rule::kVocabulary, "added token IDs are not dense", i);
  }
  std::vector<std::pair<std::string_view, TokenId>> by_text;
  by_text.reserve(spec.tokens.size());
  for (std::size_t i = 0; i < spec.tokens.size(); ++i)
    by_text.emplace_back(spec.tokens[i], static_cast<TokenId>(i));
  std::ranges::sort(by_text, {}, &std::pair<std::string_view, TokenId>::first);
  const auto find_special = [&](json::Value value) -> std::expected<TokenId, Error> {
    const auto content = token_text(value);
    if (!content) return Fail(Rule::kFormat, "special token without text");
    if (value.is_object() &&
        (!Off(value, "lstrip") || !Off(value, "rstrip") || !Off(value, "single_word")))
      return Fail(Rule::kUnsupported, "special token stripping or single-word matching");
    const auto found = std::ranges::lower_bound(by_text, *content, {},
                                                &std::pair<std::string_view, TokenId>::first);
    if (found == by_text.end() || found->first != *content)
      return Fail(Rule::kVocabulary, "named special token is absent");
    const auto id = static_cast<std::size_t>(found->second);
    spec.kinds[id] = TokenKind::kControl;
    if (!spec.added_tokens.empty()) spec.added_tokens[id] = true;
    if (!spec.model_tokens.empty()) spec.model_tokens[id] = false;
    return static_cast<TokenId>(id);
  };
  for (const std::string_view name : {"bos_token", "eos_token", "unk_token", "pad_token"}) {
    const auto value = root.find(name);
    if (!value || value->is_null()) continue;
    auto id = find_special(*value);
    if (!id) return std::unexpected(id.error());
    if (name == "bos_token") spec.bos = *id;
    if (name == "eos_token") spec.eos = *id;
  }
  if (const auto extra = root.find("additional_special_tokens"); extra && !extra->is_null()) {
    if (!extra->is_array()) return Fail(Rule::kFormat, "additional_special_tokens is not an array");
    for (std::size_t i = 0; i < extra->size(); ++i) {
      auto id = find_special(extra->at(i));
      if (!id) return std::unexpected(id.error());
    }
  }
  if (const auto extra = root.find("extra_special_tokens"); extra && !extra->is_null()) {
    if (!extra->is_object()) return Fail(Rule::kFormat, "extra_special_tokens is not an object");
    for (std::size_t i = 0; i < extra->size(); ++i) {
      auto id = find_special(extra->member(i));
      if (!id) return std::unexpected(id.error());
    }
  }
  for (const std::string_view name : {"add_bos_token", "add_eos_token"}) {
    const auto value = root.find(name);
    if (!value) continue;
    if (!value->is_bool()) return Fail(Rule::kFormat, "special insertion flag is not a boolean");
    if (name == "add_bos_token")
      spec.add_bos = value->boolean();
    else
      spec.add_eos = value->boolean();
  }
  if ((spec.add_bos && !spec.bos) || (spec.add_eos && !spec.eos))
    return Fail(Rule::kVocabulary, "special insertion without a named token");
  return {};
}

}  // namespace llmp::tokenizer
