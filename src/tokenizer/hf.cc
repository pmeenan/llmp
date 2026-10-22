// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/hf.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"
#include "base/work_pulse.h"
#include "tokenizer/error.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/tokenizer.h"

namespace llmp::tokenizer {
namespace {

namespace json = base::json;

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t item = kNoItem) {
  return std::unexpected(Error{rule, reason, item});
}

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

// Serialized boolean stage flags never reinterpret null or another type.
bool FalseFlag(json::Value object, std::string_view name) {
  const auto v = object.find(name);
  return !v || (v->is_bool() && !v->boolean());
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
  std::vector<json::Value> pending{*n};
  bool has_nfc = false;
  while (!pending.empty()) {
    const auto value = pending.back();
    pending.pop_back();
    const auto type = StringMember(value, "type");
    if (!type) return Fail(Rule::kFormat, "normalizer missing type");
    if (*type == "Sequence") {
      const auto list = value.find("normalizers");
      if (list && list->is_array()) {
        for (std::size_t i = list->size(); i > 0; --i) {
          pending.push_back(list->at(i - 1));
        }
      }
    } else if (*type == "NFC") {
      has_nfc = true;
    } else if (*type == "Strip") {
      const auto left = value.find("strip_left");
      const auto right = value.find("strip_right");
      const bool strip_left = left && left->is_bool() && left->boolean();
      const bool strip_right = right && right->is_bool() && right->boolean();
      if (strip_left || strip_right) return Fail(Rule::kUnsupported, "normalizer");
    } else {
      return Fail(Rule::kUnsupported, "normalizer");
    }
  }
  return has_nfc ? Normalization::kNfc : Normalization::kNone;
}

bool Only(json::Value value, std::initializer_list<std::string_view> names) {
  if (!value.is_object()) return false;
  for (std::size_t i = 0; i < value.size(); ++i)
    if (std::ranges::find(names, value.key(i)) == names.end()) return false;
  return true;
}

std::expected<void, Error> ReadPreTokenizer(json::Value root, TokenizerSpec& spec) {
  const auto pre = root.find("pre_tokenizer");
  if (!pre || pre->is_null())
    return Fail(Rule::kUnsupported, "byte BPE requires ByteLevel pre-tokenizer");
  std::vector<json::Value> pending{*pre};
  spec.split_expressions.clear();
  spec.split_dialect = RegexDialect::kHuggingFace;
  bool byte_level = false;
  while (!pending.empty()) {
    if (!base::Pulse()) return Fail(Rule::kCancelled, "pre-tokenizer parsing was asked to stop");
    const auto value = pending.back();
    pending.pop_back();
    const auto type = StringMember(value, "type");
    if (type == "Sequence") {
      const auto list = value.find("pretokenizers");
      if (!Only(value, {"type", "pretokenizers"}) || !list || !list->is_array())
        return Fail(Rule::kFormat, "pre-tokenizer Sequence fields");
      if (list->size() > pending.max_size() - pending.size())
        return Fail(Rule::kBounds, "pre-tokenizer traversal storage overflows");
      for (std::size_t i = list->size(); i > 0; --i) {
        if ((i & 255U) == 0 && !base::Pulse())
          return Fail(Rule::kCancelled, "pre-tokenizer parsing was asked to stop");
        pending.push_back(list->at(i - 1));
      }
    } else if (type == "ByteLevel") {
      if (!Only(value, {"type", "add_prefix_space", "trim_offsets", "use_regex"}) || byte_level)
        return Fail(Rule::kUnsupported, "repeated or malformed ByteLevel stage");
      for (const auto name : {"add_prefix_space", "trim_offsets", "use_regex"}) {
        const auto flag = value.find(name);
        if (flag && !flag->is_bool()) return Fail(Rule::kFormat, "ByteLevel flag is not boolean");
      }
      const auto prefix = value.find("add_prefix_space");
      if (prefix && prefix->boolean()) {
        spec.add_space_prefix = true;
      }
      const auto regex = value.find("use_regex");
      if (!regex || regex->boolean()) {
        auto preset = ReadPreTokenizerPreset("gpt-2");
        spec.split_expressions.insert(spec.split_expressions.end(), preset->expressions.begin(),
                                      preset->expressions.end());
      }
      byte_level = true;
    } else if (type == "Split") {
      const auto pattern = value.find("pattern");
      const auto regex = pattern ? StringMember(*pattern, "Regex") : std::nullopt;
      const auto behavior = StringMember(value, "behavior");
      if (!Only(value, {"type", "pattern", "behavior", "invert"}) || !pattern ||
          !Only(*pattern, {"Regex"}) || !regex || behavior != "Isolated" ||
          !FalseFlag(value, "invert"))
        return Fail(Rule::kUnsupported, "pre-tokenizer stage other than isolated regex Split");
      if (byte_level) return Fail(Rule::kUnsupported, "Split after byte alphabet conversion");
      if (Regex::ConstructionBytes(regex->size()) == UINT64_MAX)
        return Fail(Rule::kBounds, "regex construction representation overflows");
      spec.split_expressions.emplace_back(*regex);
    } else if (type == "Digits") {
      if (!Only(value, {"type", "individual_digits"}))
        return Fail(Rule::kUnsupported, "Digits pre-tokenizer fields");
      const auto ind = value.find("individual_digits");
      if (ind && !ind->is_bool())
        return Fail(Rule::kFormat, "Digits individual_digits flag is not boolean");
      if (byte_level) return Fail(Rule::kUnsupported, "Digits after byte alphabet conversion");
      spec.split_expressions.emplace_back((ind && ind->boolean()) ? R"(\p{N})" : R"(\p{N}+)");
    } else
      return Fail(Rule::kUnsupported, "pre-tokenizer stage");
  }
  if (!byte_level) return Fail(Rule::kUnsupported, "byte BPE requires ByteLevel pre-tokenizer");
  spec.pre_tokenizer = PreTokenizer::kByteBpe;
  return {};
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
        StringMember(*pre, "behavior") != "MergedWithPrevious" || !FalseFlag(*pre, "invert"))
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

std::expected<void, Error> ReadPostProcessor(json::Value root, TokenizerSpec& spec) {
  const auto post = root.find("post_processor");
  if (!post || post->is_null()) return {};
  spec.has_post_processor = true;
  std::size_t prefix_count = 0, suffix_count = 0;
  const auto parse = [&](bool measuring) -> std::expected<void, Error> {
    std::vector<json::Value> pending{*post};
    std::uint64_t expanded_tokens = 0;
    std::size_t definition_ids = 0;
    while (!pending.empty()) {
      if (!base::Pulse()) return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
      const auto value = pending.back();
      pending.pop_back();
      const auto type = StringMember(value, "type");
      if (type == "Sequence") {
        const auto list = value.find("processors");
        if (!Only(value, {"type", "processors"}) || !list || !list->is_array())
          return Fail(Rule::kFormat, "postprocessor Sequence fields");
        if (list->size() > pending.max_size() - pending.size())
          return Fail(Rule::kBounds, "postprocessor traversal storage overflows");
        for (std::size_t i = list->size(); i > 0; --i) {
          if ((i & 255U) == 0 && !base::Pulse())
            return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
          pending.push_back(list->at(i - 1));
        }
        continue;
      }
      if (type == "ByteLevel") {
        if (!Only(value, {"type", "add_prefix_space", "trim_offsets", "use_regex"}))
          return Fail(Rule::kUnsupported, "ByteLevel postprocessor fields");
        for (const auto name : {"add_prefix_space", "trim_offsets", "use_regex"})
          if (const auto flag = value.find(name); flag && !flag->is_bool())
            return Fail(Rule::kFormat, "ByteLevel postprocessor flag is not boolean");
        continue;  // offsets-only; this tokenizer returns IDs, never offsets
      }
      if (type != "TemplateProcessing" ||
          !Only(value, {"type", "single", "pair", "special_tokens"}))
        return Fail(Rule::kUnsupported, "postprocessor type or fields");
      const auto definitions = value.find("special_tokens");
      const auto single = value.find("single");
      if (!definitions || !definitions->is_object() || !single)
        return Fail(Rule::kFormat, "TemplateProcessing definitions or single template missing");
      std::uint64_t comparisons = 0, pulsed_comparisons = 0;
      auto by_name = [&](std::string_view left, std::string_view right) {
        ++comparisons;
        return left < right;
      };
      // Borrow names from the JSON document. Ordered nodes bound adversarial
      // lookup to logarithmic comparisons without hash-collision assumptions.
      // The existing 32*source-byte parser allowance funds 128 bytes per node:
      // this build's value/tree links use 64, plus allocation/alignment overhead.
      // Each accepted definition has at least 32 source bytes, so the index
      // costs at most 4*source bytes and is gone before Create's retained charge.
      using Definitions = std::map<std::string_view, PostProcessorSlice, decltype(by_name)>;
      static_assert(sizeof(Definitions::value_type) <= 64);
      Definitions defined(by_name);
      if (definitions->size() > defined.max_size())
        return Fail(Rule::kBounds, "postprocessor definition storage overflows");
      const auto index_progress = [&] {
        const auto now = comparisons >> 8U;
        if (now == pulsed_comparisons) return true;
        pulsed_comparisons = now;
        return base::Pulse();
      };
      for (std::size_t i = 0; i < definitions->size(); ++i) {
        const auto definition = definitions->member(i);
        const auto ids = definition.find("ids"), tokens = definition.find("tokens");
        if (!Only(definition, {"id", "ids", "tokens"}) ||
            StringMember(definition, "id") != definitions->key(i) || !ids || !ids->is_array() ||
            !tokens || !tokens->is_array() || ids->size() == 0 || ids->size() != tokens->size())
          return Fail(Rule::kFormat, "malformed TemplateProcessing special token", i);
        if (ids->size() > spec.post_processor.ids.max_size() - definition_ids)
          return Fail(Rule::kBounds, "postprocessor definition storage overflows");
        const auto inserted =
            defined.emplace(definitions->key(i), PostProcessorSlice{definition_ids, ids->size()});
        // Observe cancellation after the balanced operation, never by changing
        // comparator ordering halfway through an insertion or lookup.
        if (!index_progress())
          return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
        if (!inserted.second)
          return Fail(Rule::kFormat, "duplicate TemplateProcessing special token", i);
        definition_ids += ids->size();
        for (std::size_t j = 0; j < ids->size(); ++j) {
          if ((j & 255U) == 0 && !base::Pulse())
            return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
          const auto id = ids->at(j).int64();
          if (!id || *id < 0 || std::cmp_greater_equal(*id, spec.tokens.size()) ||
              !tokens->at(j).is_string() ||
              spec.tokens[static_cast<std::size_t>(*id)] != tokens->at(j).string())
            return Fail(Rule::kVocabulary, "TemplateProcessing special token ID/text mismatch", i);
          if (!measuring) spec.post_processor.ids.push_back(static_cast<TokenId>(*id));
        }
      }
      const auto prefix_begin = prefix_count;
      const auto read_template = [&](json::Value list, bool pair) -> std::expected<void, Error> {
        if (!list.is_array() && !list.is_string())
          return Fail(Rule::kFormat, "TemplateProcessing template is neither string nor array");
        unsigned sequences = 0;
        std::size_t string_pos = 0, entries = 0, examined = 0;
        const auto text = list.is_string() ? list.string() : std::string_view{};
        while (list.is_array() ? entries < list.size() : string_pos < text.size()) {
          if ((entries & 255U) == 0 && !base::Pulse())
            return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
          ++entries;
          std::string_view name;
          bool sequence = false;
          if (list.is_array()) {
            const auto entry = list.at(entries - 1);
            if (!entry.is_object() || entry.size() != 1)
              return Fail(Rule::kFormat, "TemplateProcessing item is not tagged");
            sequence = entry.key(0) == "Sequence";
            if (!sequence && entry.key(0) != "SpecialToken")
              return Fail(Rule::kUnsupported, "TemplateProcessing item kind");
            const auto item = entry.member(0);
            const auto token_name = StringMember(item, "id");
            const auto type_value = item.find("type_id");
            const auto type_id = type_value ? type_value->int64() : std::nullopt;
            if (!Only(item, {"id", "type_id"}) || !token_name || !type_id || *type_id < 0 ||
                *type_id > 0xFFFFFFFFLL)
              return Fail(Rule::kFormat, "TemplateProcessing item fields");
            name = *token_name;
          } else {
            while (string_pos < text.size() && text[string_pos] == ' ') {
              if ((++examined & 255U) == 0 && !base::Pulse())
                return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
              ++string_pos;
            }
            if (string_pos == text.size()) break;
            const auto begin = string_pos;
            while (string_pos < text.size() && text[string_pos] != ' ') {
              if ((++examined & 255U) == 0 && !base::Pulse())
                return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
              ++string_pos;
            }
            name = text.substr(begin, string_pos - begin);
            if (const auto colon = name.rfind(':'); colon != std::string_view::npos) {
              std::uint32_t type_id = 0;
              const auto digits = name.substr(colon + 1);
              const auto result =
                  std::from_chars(digits.data(), digits.data() + digits.size(), type_id);
              if (digits.empty() || result.ec != std::errc{} ||
                  result.ptr != digits.data() + digits.size())
                return Fail(Rule::kFormat, "TemplateProcessing type ID");
              name = name.substr(0, colon);
            }
            sequence = name.starts_with('$');
            if (sequence) {
              name.remove_prefix(1);
              if (!name.empty() && name.front() >= '0' && name.front() <= '9') {
                std::uint32_t type_id = 0;
                const auto result =
                    std::from_chars(name.data(), name.data() + name.size(), type_id);
                if (result.ec != std::errc{} || result.ptr != name.data() + name.size())
                  return Fail(Rule::kFormat, "TemplateProcessing sequence ID");
                name = "A";
              }
            }
          }
          if (sequence) {
            const unsigned bit = name == "A" ? 1U : pair && name == "B" ? 2U : 0U;
            if (bit == 0 || (sequences & bit) != 0)
              return Fail(Rule::kUnsupported, "TemplateProcessing repeated or unknown sequence");
            sequences |= bit;
          } else {
            const auto definition = defined.find(name);
            if (!index_progress())
              return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
            if (definition == defined.end())
              return Fail(Rule::kVocabulary, "undefined TemplateProcessing special token");
            if (!pair) {
              const bool before = sequences == 0;
              auto& insertion = before ? spec.post_processor.prefix : spec.post_processor.suffix;
              auto& count = before ? prefix_count : suffix_count;
              if (count == insertion.max_size() ||
                  __builtin_add_overflow(expanded_tokens,
                                         static_cast<std::uint64_t>(definition->second.length),
                                         &expanded_tokens))
                return Fail(Rule::kBounds, "postprocessor insertion representation overflows");
              ++count;
              if (!measuring) insertion.push_back(definition->second);
            }
          }
        }
        if (sequences != (pair ? 3U : 1U))
          return Fail(Rule::kUnsupported, "TemplateProcessing sequence is missing");
        return {};
      };
      if (auto parsed = read_template(*single, false); !parsed) return parsed;
      if (const auto pair = value.find("pair"); pair) {
        if (auto parsed = read_template(*pair, true); !parsed) return parsed;
      }
      if (!measuring) {
        // Append each template's prefix reversed. One final reversal produces
        // later-prefix/earlier-prefix order without any insertion copies.
        auto& prefix = spec.post_processor.prefix;
        for (std::size_t left = prefix_begin, right = prefix.size(); left < right && left < --right;
             ++left) {
          if ((left & 255U) == 0 && !base::Pulse())
            return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
          std::swap(prefix[left], prefix[right]);
        }
      }
    }
    if (measuring) {
      // Count first: these allocations and all later retained storage are
      // proportional to JSON definitions/references, never expanded output.
      spec.post_processor.ids.reserve(definition_ids);
      spec.post_processor.prefix.reserve(prefix_count);
      spec.post_processor.suffix.reserve(suffix_count);
    }
    return {};
  };
  if (auto counted = parse(true); !counted) return counted;
  prefix_count = suffix_count = 0;
  if (auto filled = parse(false); !filled) return filled;
  auto& prefix = spec.post_processor.prefix;
  for (std::size_t left = 0, right = prefix.size(); left < right && left < --right; ++left) {
    if ((left & 255U) == 0 && !base::Pulse())
      return Fail(Rule::kCancelled, "postprocessor parsing was asked to stop");
    std::swap(prefix[left], prefix[right]);
  }
  return {};
}

}  // namespace

std::expected<TokenizerSpec, Error> ReadHfTokenizer(std::string_view text) {
  if (!base::Pulse()) return Fail(Rule::kCancelled, "tokenizer parsing was asked to stop");
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
    if (auto pre = ReadPreTokenizer(root, spec); !pre) return std::unexpected(pre.error());
    if (!decoder || StringMember(*decoder, "type") != "ByteLevel")
      return Fail(Rule::kUnsupported, "decoder other than ByteLevel");
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
  std::vector<bool> rstrips(total, false);
  bool has_rstrip = false;
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
    const auto rstrip_val = a.find("rstrip");
    const bool rstrip = rstrip_val && rstrip_val->is_bool() && rstrip_val->boolean();
    if (!Off(a, "lstrip") || (rstrip_val && !rstrip_val->is_bool()) || !Off(a, "single_word")) {
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
    if (rstrip) {
      rstrips[slot] = true;
      has_rstrip = true;
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
  if (has_rstrip) {
    rstrips.resize(count);
    spec.rstrip = std::move(rstrips);
  }
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
  if (auto post = ReadPostProcessor(root, spec); !post) return std::unexpected(post.error());
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
    if (!spec.rstrip.empty()) spec.rstrip.resize(count, false);
    for (std::size_t i = 0; i < added->size(); ++i) {
      const auto id = *id_of(added->key(i));
      const auto token = added->member(i);
      const auto content = token.is_object() ? StringMember(token, "content") : std::nullopt;
      const auto special = token.find("special");
      if (!content || !special || !special->is_bool())
        return Fail(Rule::kFormat, "added_tokens_decoder token without content or special flag",
                    id);
      const auto rstrip_val = token.find("rstrip");
      const bool rstrip = rstrip_val && rstrip_val->is_bool() && rstrip_val->boolean();
      if (!Off(token, "lstrip") || (rstrip_val && !rstrip_val->is_bool()) ||
          !Off(token, "single_word"))
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
      if (rstrip) {
        if (spec.rstrip.empty()) spec.rstrip.resize(count, false);
        spec.rstrip[id] = true;
      }
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
    const auto rstrip_val = value.is_object() ? value.find("rstrip") : std::nullopt;
    const bool rstrip = rstrip_val && rstrip_val->is_bool() && rstrip_val->boolean();
    if (value.is_object() && (!Off(value, "lstrip") || (rstrip_val && !rstrip_val->is_bool()) ||
                              !Off(value, "single_word")))
      return Fail(Rule::kUnsupported, "special token stripping or single-word matching");
    const auto found = std::ranges::lower_bound(by_text, *content, {},
                                                &std::pair<std::string_view, TokenId>::first);
    if (found == by_text.end() || found->first != *content)
      return Fail(Rule::kVocabulary, "named special token is absent");
    const auto id = static_cast<std::size_t>(found->second);
    spec.kinds[id] = TokenKind::kControl;
    if (rstrip) {
      if (spec.rstrip.empty()) spec.rstrip.resize(spec.tokens.size(), false);
      spec.rstrip[id] = true;
    }
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
    if (!extra->is_object() && !extra->is_array())
      return Fail(Rule::kFormat, "extra_special_tokens is not an object or array");
    for (std::size_t i = 0; i < extra->size(); ++i) {
      auto id = find_special(extra->is_array() ? extra->at(i) : extra->member(i));
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
