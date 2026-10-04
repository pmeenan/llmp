// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/gguf.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tokenizer/error.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::tokenizer {
namespace {

enum class Type : std::uint32_t {
  kU8 = 0,
  kI8 = 1,
  kU16 = 2,
  kI16 = 3,
  kU32 = 4,
  kI32 = 5,
  kF32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kU64 = 10,
  kI64 = 11,
  kF64 = 12,
};

std::size_t ScalarSize(std::uint32_t type) {
  switch (static_cast<Type>(type)) {
    case Type::kU8:
    case Type::kI8:
    case Type::kBool:
      return 1;
    case Type::kU16:
    case Type::kI16:
      return 2;
    case Type::kU32:
    case Type::kI32:
    case Type::kF32:
      return 4;
    case Type::kU64:
    case Type::kI64:
    case Type::kF64:
      return 8;
    case Type::kString:
    case Type::kArray:
      return 0;
  }
  return 0;
}

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t item = kNoItem) {
  return std::unexpected(Error{rule, reason, item});
}

class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::size_t offset() const { return pos_; }
  std::size_t remaining() const { return bytes_.size() - pos_; }

  std::expected<std::uint64_t, Error> Unsigned(std::size_t size) {
    if (remaining() < size) {
      return Fail(Rule::kFormat, "truncated", pos_);
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {  // little endian
      value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes_[pos_ + i]))
               << (8U * i);
    }
    pos_ += size;
    return value;
  }

  std::expected<std::string_view, Error> String() {
    auto length = Unsigned(8);
    if (!length) {
      return std::unexpected(length.error());
    }
    if (*length > kMaxGgufString) {
      return Fail(Rule::kBounds, "string longer than the cap", pos_);
    }
    if (*length > remaining()) {
      return Fail(Rule::kFormat, "truncated", pos_);
    }
    const std::string_view s(reinterpret_cast<const char*>(bytes_.data() + pos_), *length);
    pos_ += *length;
    return s;
  }

  std::expected<void, Error> Skip(std::size_t size) {
    if (remaining() < size) {
      return Fail(Rule::kFormat, "truncated", pos_);
    }
    pos_ += size;
    return {};
  }

 private:
  std::span<const std::byte> bytes_;
  std::size_t pos_ = 0;
};

// A value the tokenizer uses, or a skipped one.
struct Value {
  std::uint32_t type = 0;
  std::uint32_t element_type = 0;  // arrays
  std::uint64_t scalar = 0;        // integers and bools
  std::string_view string;
  std::vector<std::string_view> strings;  // arrays of strings
  std::vector<float> floats;              // arrays of f32 scores
  std::vector<std::int32_t> ints;         // arrays of i32
};

std::expected<void, Error> SkipValue(Reader& r, std::uint32_t type, int depth) {
  if (const std::size_t size = ScalarSize(type); size != 0) {
    return r.Skip(size);
  }
  if (type == static_cast<std::uint32_t>(Type::kString)) {
    // A string nothing reads needs no cap beyond the bytes it spans (some
    // GGUFs embed large documents under other keys).
    auto length = r.Unsigned(8);
    if (!length) {
      return std::unexpected(length.error());
    }
    if (*length > r.remaining()) {
      return Fail(Rule::kFormat, "truncated", r.offset());
    }
    return r.Skip(static_cast<std::size_t>(*length));
  }
  if (type != static_cast<std::uint32_t>(Type::kArray)) {
    return Fail(Rule::kFormat, "unknown value type", r.offset());
  }
  if (depth > 1) {
    return Fail(Rule::kBounds, "arrays nested too deeply", r.offset());
  }
  auto element = r.Unsigned(4);
  auto count = element ? r.Unsigned(8) : element;
  if (!count) {
    return std::unexpected(count.error());
  }
  // Every element takes at least a byte (a string its 8-byte length, an
  // array 12), so the bytes left bound the count before any loop.
  const auto e = static_cast<std::uint32_t>(*element);
  const std::size_t size = ScalarSize(e);
  std::size_t least = 12;
  if (size != 0) {
    least = size;
  } else if (e == static_cast<std::uint32_t>(Type::kString)) {
    least = 8;
  }
  if (*count > r.remaining() / least) {
    return Fail(Rule::kFormat, "truncated", r.offset());
  }
  if (size != 0) {
    return r.Skip(static_cast<std::size_t>(*count) * size);
  }
  for (std::uint64_t i = 0; i < *count; ++i) {
    if (auto s = SkipValue(r, e, depth + 1); !s) {
      return s;
    }
  }
  return {};
}

std::expected<Value, Error> ReadValue(Reader& r, std::uint32_t type) {
  Value v;
  v.type = type;
  if (const std::size_t size = ScalarSize(type); size != 0) {
    auto x = r.Unsigned(size);
    if (!x) {
      return std::unexpected(x.error());
    }
    v.scalar = *x;
    return v;
  }
  if (type == static_cast<std::uint32_t>(Type::kString)) {
    auto s = r.String();
    if (!s) {
      return std::unexpected(s.error());
    }
    v.string = *s;
    return v;
  }
  if (type != static_cast<std::uint32_t>(Type::kArray)) {
    return Fail(Rule::kFormat, "unknown value type", r.offset());
  }
  auto element = r.Unsigned(4);
  auto count = element ? r.Unsigned(8) : element;
  if (!count) {
    return std::unexpected(count.error());
  }
  if (*count > kMaxGgufArray) {
    return Fail(Rule::kBounds, "array longer than the cap", r.offset());
  }
  v.element_type = static_cast<std::uint32_t>(*element);
  if (v.element_type == static_cast<std::uint32_t>(Type::kString)) {
    // Each string takes at least its 8-byte length, so the count is bounded
    // by the bytes left before anything is reserved.
    if (*count > r.remaining() / 8) {
      return Fail(Rule::kFormat, "truncated", r.offset());
    }
    v.strings.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t i = 0; i < *count; ++i) {
      auto s = r.String();
      if (!s) {
        return std::unexpected(s.error());
      }
      v.strings.push_back(*s);
    }
    return v;
  }
  if (v.element_type == static_cast<std::uint32_t>(Type::kF32)) {
    if (*count > r.remaining() / 4) {
      return Fail(Rule::kFormat, "truncated", r.offset());
    }
    v.floats.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t i = 0; i < *count; ++i) {
      auto value = r.Unsigned(4);
      if (!value) {
        return std::unexpected(value.error());
      }
      v.floats.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(*value)));
    }
    return v;
  }
  if (v.element_type == static_cast<std::uint32_t>(Type::kI32)) {
    if (*count > r.remaining() / 4) {
      return Fail(Rule::kFormat, "truncated", r.offset());
    }
    v.ints.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t i = 0; i < *count; ++i) {
      auto x = r.Unsigned(4);
      if (!x) {
        return std::unexpected(x.error());
      }
      v.ints.push_back(static_cast<std::int32_t>(static_cast<std::uint32_t>(*x)));
    }
    return v;
  }
  // Another array: skip its elements.
  const std::uint32_t e = v.element_type;
  if (const std::size_t size = ScalarSize(e); size != 0) {
    if (auto s = r.Skip(static_cast<std::size_t>(*count) * size); !s) {
      return std::unexpected(s.error());
    }
    return v;
  }
  for (std::uint64_t i = 0; i < *count; ++i) {
    if (auto s = SkipValue(r, e, 1); !s) {
      return std::unexpected(s.error());
    }
  }
  return v;
}

// The keys the reader uses; every other key is skipped unread.
bool Wanted(std::string_view key) {
  static constexpr std::array<std::string_view, 15> kKeys = {
      "general.architecture",
      "general.name",
      "tokenizer.ggml.scores",
      "tokenizer.ggml.model",
      "tokenizer.ggml.pre",
      "tokenizer.ggml.tokens",
      "tokenizer.ggml.token_type",
      "tokenizer.ggml.merges",
      "tokenizer.ggml.bos_token_id",
      "tokenizer.ggml.eos_token_id",
      "tokenizer.ggml.add_bos_token",
      "tokenizer.ggml.add_eos_token",
      "tokenizer.ggml.add_space_prefix",
      "tokenizer.ggml.remove_extra_whitespaces",
      "tokenizer.chat_template",
  };
  return std::ranges::find(kKeys, key) != kKeys.end();
}

}  // namespace

std::expected<GgufTokenizer, Error> ReadGgufTokenizer(std::span<const std::byte> bytes) {
  Reader r(bytes);
  auto magic = r.Unsigned(4);
  if (!magic) {
    return std::unexpected(magic.error());
  }
  if (*magic != 0x46554747U) {  // "GGUF"
    return Fail(Rule::kFormat, "not a GGUF file", 0);
  }
  auto version = r.Unsigned(4);
  if (!version) {
    return std::unexpected(version.error());
  }
  if (*version != 2 && *version != 3) {
    return Fail(Rule::kUnsupported, "GGUF version", *version);
  }
  auto tensors = r.Unsigned(8);
  auto keys = tensors ? r.Unsigned(8) : tensors;
  if (!keys) {
    return std::unexpected(keys.error());
  }
  if (*keys > kMaxGgufKeys) {
    return Fail(Rule::kBounds, "more keys than the cap", *keys);
  }

  std::set<std::string_view> seen;
  std::vector<std::pair<std::string_view, Value>> values;
  for (std::uint64_t i = 0; i < *keys; ++i) {
    auto key = r.String();
    if (!key) {
      return std::unexpected(key.error());
    }
    if (!seen.insert(*key).second) {
      return Fail(Rule::kFormat, "a key repeats", i);
    }
    auto type = r.Unsigned(4);
    if (!type) {
      return std::unexpected(type.error());
    }
    const auto t = static_cast<std::uint32_t>(*type);
    if (Wanted(*key)) {
      auto v = ReadValue(r, t);
      if (!v) {
        return std::unexpected(v.error());
      }
      values.emplace_back(*key, std::move(*v));
    } else if (auto s = SkipValue(r, t, 0); !s) {
      return std::unexpected(s.error());
    }
  }

  auto find = [&](std::string_view key) -> const Value* {
    for (const auto& [k, v] : values) {
      if (k == key) {
        return &v;
      }
    }
    return nullptr;
  };
  // A string key into *value; whether it was present.
  auto string_key = [&](std::string_view key, std::string* value) -> std::expected<bool, Error> {
    const Value* v = find(key);
    if (v == nullptr) {
      return false;
    }
    if (v->type != static_cast<std::uint32_t>(Type::kString)) {
      return Fail(Rule::kFormat, "a tokenizer key has the wrong type");
    }
    *value = std::string(v->string);
    return true;
  };

  GgufTokenizer out;
  std::string model;
  auto has_architecture = string_key("general.architecture", &out.architecture);
  auto has_model = has_architecture ? string_key("tokenizer.ggml.model", &model) : has_architecture;
  auto has_pre = has_model ? string_key("tokenizer.ggml.pre", &out.pre) : has_model;
  auto has_template = has_pre ? string_key("tokenizer.chat_template", &out.chat_template) : has_pre;
  if (!has_template) {
    return std::unexpected(has_template.error());
  }
  if (!*has_model || (!*has_pre && model != "gemma4" && model != "llama")) {
    return Fail(Rule::kFormat, "tokenizer model or pre-tokenizer missing");
  }
  out.has_chat_template = *has_template;
  if (model != "gpt2" && model != "gemma4" && model != "llama") {
    return Fail(Rule::kUnsupported, "tokenizer model other than gpt2, gemma4 or llama");
  }
  const bool spm = model == "llama";
  if (spm) {
    if (*has_pre && out.pre != "default") {
      return Fail(Rule::kUnsupported, "SentencePiece pre-tokenizer");
    }
    out.pre = "default";
    out.spec.pre_tokenizer = PreTokenizer::kSentencePiece;
  } else if (model == "gemma4") {
    if (*has_pre && out.pre != "gemma4") {
      return Fail(Rule::kUnsupported, "Gemma4 pre-tokenizer");
    }
    out.pre = "gemma4";
    out.spec.pre_tokenizer = PreTokenizer::kGemma4;
  } else if (out.pre == "qwen2") {
    out.spec.pre_tokenizer = PreTokenizer::kQwen2;
  } else if (out.pre == "qwen35") {
    out.spec.pre_tokenizer = PreTokenizer::kQwen35;
  } else if (out.pre == "deepseek-v3" || out.pre == "joyai-llm") {
    out.spec.pre_tokenizer = PreTokenizer::kDeepSeekV3;
  } else {
    return Fail(Rule::kUnsupported, "pre-tokenizer");
  }
  out.spec.normalization = Normalization::kNone;

  const Value* tokens = find("tokenizer.ggml.tokens");
  const Value* types = find("tokenizer.ggml.token_type");
  const Value* merges = find("tokenizer.ggml.merges");
  const Value* scores = find("tokenizer.ggml.scores");
  if (tokens == nullptr || types == nullptr || (spm ? scores == nullptr : merges == nullptr)) {
    return Fail(Rule::kFormat, "tokens, token types or merges missing");
  }
  const auto kString = static_cast<std::uint32_t>(Type::kString);
  const auto kArray = static_cast<std::uint32_t>(Type::kArray);
  if (tokens->type != kArray || tokens->element_type != kString || types->type != kArray ||
      types->element_type != static_cast<std::uint32_t>(Type::kI32) ||
      (spm ? scores->type != kArray ||
                 scores->element_type != static_cast<std::uint32_t>(Type::kF32)
           : merges->type != kArray || merges->element_type != kString)) {
    return Fail(Rule::kFormat, "tokens, token types or merges have the wrong type");
  }
  if (types->ints.size() != tokens->strings.size()) {
    return Fail(Rule::kVocabulary, "token types and tokens differ in number", types->ints.size());
  }
  const std::size_t n = tokens->strings.size();
  if (spm) {
    if (scores->floats.size() != n) {
      return Fail(Rule::kVocabulary, "token scores and tokens differ in number");
    }
    if (merges != nullptr) {
      return Fail(Rule::kUnsupported, "SentencePiece uses scores rather than a merge list");
    }
    if (std::ranges::any_of(scores->floats, [](float score) { return !std::isfinite(score); })) {
      return Fail(Rule::kVocabulary, "nonfinite SentencePiece token score");
    }
    out.spec.scores = scores->floats;
  }
  out.spec.tokens.reserve(n);
  out.spec.kinds.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.spec.tokens.emplace_back(tokens->strings[i]);
    switch (types->ints[i]) {
      case 1:
        out.spec.kinds.push_back(TokenKind::kNormal);
        break;
      case 2:
        if (!spm) {
          return Fail(Rule::kUnsupported, "unknown token in a BPE vocabulary", i);
        }
        [[fallthrough]];
      case 3:
        out.spec.kinds.push_back(TokenKind::kControl);
        break;
      case 4:
        out.spec.kinds.push_back(TokenKind::kUserDefined);
        break;
      case 5:
        out.spec.kinds.push_back(TokenKind::kUnused);
        break;
      case 6:
        if (model == "gemma4" || spm) {
          out.spec.kinds.push_back(TokenKind::kByte);
          break;
        }
        [[fallthrough]];
      default:  // 2 unknown (SPM vocabularies), 0 undefined
        return Fail(Rule::kUnsupported, "token type", i);
    }
  }
  if (merges != nullptr) {
    out.spec.merges.reserve(merges->strings.size());
    for (std::size_t i = 0; i < merges->strings.size(); ++i) {
      const std::string_view m = merges->strings[i];
      const std::size_t space = m.find(' ');
      if (space == 0 || space == std::string_view::npos || space + 1 == m.size() ||
          m.find(' ', space + 1) != std::string_view::npos) {
        return Fail(Rule::kVocabulary, "a merge is not two tokens separated by one space", i);
      }
      out.spec.merges.emplace_back(std::string(m.substr(0, space)),
                                   std::string(m.substr(space + 1)));
    }
  }
  auto id_key = [&](std::string_view key) -> std::expected<std::optional<TokenId>, Error> {
    const Value* v = find(key);
    if (v == nullptr) {
      return std::optional<TokenId>{};
    }
    if (v->type != static_cast<std::uint32_t>(Type::kU32)) {
      return Fail(Rule::kFormat, "a token ID key has the wrong type");
    }
    if (v->scalar >= n) {
      return Fail(Rule::kVocabulary, "a special token ID outside the vocabulary", v->scalar);
    }
    return std::optional<TokenId>{static_cast<TokenId>(v->scalar)};
  };
  auto bool_key = [&](std::string_view key) -> std::expected<std::optional<bool>, Error> {
    const Value* v = find(key);
    if (v == nullptr) {
      return std::optional<bool>{};
    }
    if (v->type != static_cast<std::uint32_t>(Type::kBool) || v->scalar > 1) {
      return Fail(Rule::kFormat, "a flag key has the wrong type");
    }
    return std::optional<bool>{v->scalar == 1};
  };
  auto bos = id_key("tokenizer.ggml.bos_token_id");
  auto eos = bos ? id_key("tokenizer.ggml.eos_token_id") : bos;
  if (!eos) {
    return std::unexpected(eos.error());
  }
  out.spec.bos = *bos;
  out.spec.eos = *eos;
  auto add_bos = bool_key("tokenizer.ggml.add_bos_token");
  auto add_eos = add_bos ? bool_key("tokenizer.ggml.add_eos_token") : add_bos;
  auto prefix = add_eos ? bool_key("tokenizer.ggml.add_space_prefix") : add_eos;
  auto spaces = prefix ? bool_key("tokenizer.ggml.remove_extra_whitespaces") : prefix;
  if (!spaces) {
    return std::unexpected(spaces.error());
  }
  out.spec.add_bos = add_bos->value_or(spm);
  out.spec.add_eos = add_eos->value_or(false);
  out.spec.add_space_prefix = spm && prefix->value_or(true);
  if ((!spm && prefix->value_or(false)) || spaces->value_or(false)) {
    return Fail(Rule::kUnsupported, "space prefix or whitespace removal");
  }
  if (spm) {
    // Pinned llama.cpp overrides these EOG spellings to control before
    // partitioning, even when their original GGUF token type is user-defined.
    static constexpr std::string_view eog[] = {"<|eot_id|>",
                                               "<|im_end|>",
                                               "<|end|>",
                                               "<|return|>",
                                               "<|call|>",
                                               "<|flush|>",
                                               "<|calls|>",
                                               "<end_of_turn>",
                                               "<|endoftext|>",
                                               "</s>",
                                               "<|eom_id|>",
                                               "<EOT>",
                                               "_<EOT>",
                                               "[EOT]",
                                               "[EOS]",
                                               "<|end_of_text|>",
                                               "<end_of_utterance>",
                                               "<eos>",
                                               "<turn|>",
                                               "<|tool_response>",
                                               "<｜end▁of▁sentence｜>",
                                               "[e~["};
    for (std::size_t i = 0; i < n; ++i) {
      if (std::ranges::find(eog, out.spec.tokens[i]) != std::end(eog) &&
          out.spec.kinds[i] != TokenKind::kUnused) {
        out.spec.kinds[i] = TokenKind::kControl;
      }
    }
    std::string name;
    if (auto present = string_key("general.name", &name); !present) {
      return std::unexpected(present.error());
    }
    for (char& c : name) {
      if (c >= 'A' && c <= 'Z') {
        c = static_cast<char>(c + ('a' - 'A'));
      }
    }
    if (name.find("phi-3") != std::string::npos || name.find("phi3") != std::string::npos) {
      out.spec.rstrip.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        const auto& token = out.spec.tokens[i];
        out.spec.rstrip[i] = (out.spec.kinds[i] == TokenKind::kControl ||
                              out.spec.kinds[i] == TokenKind::kUserDefined) &&
                             token != "<unk>" && token != "<s>" && token != "<|endoftext|>";
      }
    }
  }
  return out;
}

}  // namespace jitllm::tokenizer
