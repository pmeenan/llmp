// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/sentencepiece.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "base/work_pulse.h"
#include "tokenizer/unicode.h"

namespace llmp::tokenizer {
namespace {

// Wire field numbers/types follow Google's published ModelProto schema:
// sentencepiece v0.2.1, src/sentencepiece_model.proto. This reader is
// independently implemented; training-only fields do not affect encoding.
constexpr std::size_t kMaxModelBytes = std::size_t{64} << 20U;

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::size_t offset = 0) {
  return std::unexpected(Error{rule, reason, offset});
}

struct Field {
  std::uint32_t number;
  unsigned wire;
  std::uint64_t scalar = 0;
  std::string_view bytes;
  std::size_t offset;
};

class Wire {
 public:
  explicit Wire(std::string_view bytes, std::size_t offset = 0) : bytes_(bytes), offset_(offset) {}

  std::expected<std::optional<Field>, Error> Next() {
    if ((++work_ & 0xFFFFU) == 0 && !base::Pulse())
      return Fail(Rule::kCancelled, "the tokenizer reader was asked to stop", offset_ + at_);
    if (at_ == bytes_.size()) return std::nullopt;
    const auto start = at_;
    auto key = Varint();
    if (!key) return std::unexpected(key.error());
    if ((*key >> 3U) == 0 || (*key >> 3U) > 0x1FFFFFFF)
      return Fail(Rule::kFormat, "invalid protobuf field number", offset_ + start);
    Field field{.number = static_cast<std::uint32_t>(*key >> 3U),
                .wire = static_cast<unsigned>(*key & 7U),
                .bytes = {},
                .offset = offset_ + start};
    if (field.wire == 0) {
      auto scalar = Varint();
      if (!scalar) return std::unexpected(scalar.error());
      field.scalar = *scalar;
    } else if (field.wire == 1 || field.wire == 5) {
      const std::size_t length = field.wire == 1 ? 8 : 4;
      if (bytes_.size() - at_ < length)
        return Fail(Rule::kFormat, "truncated protobuf scalar", offset_ + start);
      for (std::size_t i = 0; i < length; ++i)
        field.scalar |= std::uint64_t{static_cast<unsigned char>(bytes_[at_++])} << (8U * i);
    } else if (field.wire == 2) {
      auto length = Varint();
      if (!length) return std::unexpected(length.error());
      if (*length > bytes_.size() - at_)
        return Fail(Rule::kFormat, "truncated protobuf message", offset_ + start);
      field.bytes = bytes_.substr(at_, static_cast<std::size_t>(*length));
      field.offset = offset_ + at_;
      at_ += static_cast<std::size_t>(*length);
    } else {
      return Fail(Rule::kFormat, "unsupported protobuf wire type", offset_ + start);
    }
    return field;
  }

 private:
  std::expected<std::uint64_t, Error> Varint() {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 10; ++i) {
      if (at_ == bytes_.size())
        return Fail(Rule::kFormat, "truncated protobuf varint", offset_ + at_);
      const auto c = static_cast<unsigned char>(bytes_[at_++]);
      if (i == 9 && c > 1)
        return Fail(Rule::kFormat, "overflowing protobuf varint", offset_ + at_ - 1);
      value |= std::uint64_t{c & 127U} << (7U * i);
      if ((c & 128U) == 0) return value;
    }
    return Fail(Rule::kFormat, "overflowing protobuf varint", offset_ + at_);
  }

  std::string_view bytes_;
  std::size_t offset_;
  std::size_t at_ = 0;
  std::size_t work_ = 0;
};

std::expected<bool, Error> Boolean(const Field& field) {
  if (field.wire != 0 || field.scalar > 1)
    return Fail(Rule::kFormat, "protobuf boolean has the wrong type or value", field.offset);
  return field.scalar != 0;
}

std::expected<std::int32_t, Error> Integer(const Field& field) {
  if (field.wire != 0 ||
      (field.scalar > std::uint64_t{std::numeric_limits<std::int32_t>::max()} &&
       field.scalar < std::numeric_limits<std::uint64_t>::max() -
                          std::uint64_t{std::numeric_limits<std::int32_t>::max()}))
    return Fail(Rule::kFormat, "protobuf int32 has the wrong type or value", field.offset);
  return std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(field.scalar));
}

std::expected<void, Error> Piece(const Field& message, TokenizerSpec& spec,
                                 std::optional<TokenId>& unknown) {
  Wire wire(message.bytes, message.offset);
  std::string_view text;
  float score = 0;
  std::uint64_t type = 1;
  unsigned seen = 0;
  for (;;) {
    auto read = wire.Next();
    if (!read) return std::unexpected(read.error());
    if (!*read) break;
    const auto& f = **read;
    if (f.number < 1 || f.number > 3)
      return Fail(Rule::kUnsupported, "SentencePiece piece extension", f.offset);
    const auto bit = 1U << f.number;
    if ((seen & bit) != 0)
      return Fail(Rule::kFormat, "repeated SentencePiece piece field", f.offset);
    seen |= bit;
    if (f.number == 1 && f.wire == 2)
      text = f.bytes;
    else if (f.number == 2 && f.wire == 5)
      score = std::bit_cast<float>(static_cast<std::uint32_t>(f.scalar));
    else if (f.number == 3 && f.wire == 0)
      type = f.scalar;
    else
      return Fail(Rule::kFormat, "SentencePiece piece field has the wrong type", f.offset);
  }
  if (text.empty() || text.size() > kMaxTokenBytes || !std::isfinite(score) ||
      !unicode::ValidateUtf8(text))
    return Fail(Rule::kVocabulary, "invalid SentencePiece piece text or score", message.offset);
  TokenKind kind;
  switch (type) {
    case 1:
      kind = TokenKind::kNormal;
      break;
    case 2:
      if (unknown) return Fail(Rule::kVocabulary, "multiple SentencePiece unknown pieces");
      unknown = static_cast<TokenId>(spec.tokens.size());
      kind = TokenKind::kControl;
      break;
    case 3:
      kind = TokenKind::kControl;
      break;
    case 4:
      kind = TokenKind::kUserDefined;
      break;
    case 5:
      kind = TokenKind::kUnused;
      break;
    case 6:
      kind = TokenKind::kByte;
      break;
    default:
      return Fail(Rule::kUnsupported, "SentencePiece piece kind", message.offset);
  }
  if (spec.tokens.size() == kMaxVocabulary)
    return Fail(Rule::kBounds, "SentencePiece vocabulary size", spec.tokens.size());
  spec.tokens.emplace_back(text);
  spec.kinds.push_back(kind);
  spec.scores.push_back(score);
  spec.model_tokens.push_back(type == 1 || type == 4 || type == 5);
  spec.model_symbols.push_back(type == 4);
  spec.added_tokens.push_back(type == 2 || type == 3);
  spec.normalized_tokens.push_back(false);
  return {};
}

struct Trainer {
  std::int32_t model = 1;
  std::int32_t unknown = 0;
  std::int32_t bos = 1;
  std::int32_t eos = 2;
  bool byte_fallback = false;
};

std::expected<Trainer, Error> ReadTrainer(const Field& message) {
  Wire wire(message.bytes, message.offset);
  Trainer trainer;
  std::array<bool, 128> seen{};
  for (;;) {
    auto read = wire.Next();
    if (!read) return std::unexpected(read.error());
    if (!*read) break;
    const auto& f = **read;
    if (f.number >= seen.size())
      return Fail(Rule::kUnsupported, "SentencePiece trainer extension", f.offset);
    if (f.number >= 55) {
      seen[f.number] = true;
      continue;
    }
    const bool repeated = f.number == 1 || f.number == 5 || f.number == 30 || f.number == 31;
    if (!repeated && seen[f.number])
      return Fail(Rule::kFormat, "repeated SentencePiece trainer field", f.offset);
    seen[f.number] = true;
    if (f.number == 3 || f.number == 40 || f.number == 41 || f.number == 42) {
      auto value = Integer(f);
      if (!value) return std::unexpected(value.error());
      if (f.number == 3) trainer.model = *value;
      if (f.number == 40) trainer.unknown = *value;
      if (f.number == 41) trainer.bos = *value;
      if (f.number == 42) trainer.eos = *value;
    } else if (f.number == 35 || f.number == 24) {
      auto value = Boolean(f);
      if (!value) return std::unexpected(value.error());
      if (f.number == 35)
        trainer.byte_fallback = *value;
      else if (*value)
        return Fail(Rule::kUnsupported, "SentencePiece whitespace suffix", f.offset);
    }
    // Other known trainer fields describe training or special-piece names.
    // Piece IDs/text/kinds in the model are authoritative for execution.
  }
  return trainer;
}

std::expected<void, Error> ReadNormalizer(const Field& message, TokenizerSpec& spec) {
  Wire wire(message.bytes, message.offset);
  std::array<bool, 8> seen{};
  spec.add_space_prefix = true;
  spec.remove_extra_whitespaces = true;
  for (;;) {
    auto read = wire.Next();
    if (!read) return std::unexpected(read.error());
    if (!*read) break;
    const auto& f = **read;
    if (f.number >= seen.size())
      return Fail(Rule::kUnsupported, "SentencePiece normalizer extension", f.offset);
    if (seen[f.number])
      return Fail(Rule::kFormat, "repeated SentencePiece normalizer field", f.offset);
    seen[f.number] = true;
    if (f.number == 1 || f.number == 2 || f.number == 6) {
      if (f.wire != 2)
        return Fail(Rule::kFormat, "SentencePiece normalizer field has the wrong type", f.offset);
      if (f.number == 2 && !f.bytes.empty())
        return Fail(Rule::kUnsupported, "SentencePiece compiled normalization map", f.offset);
    } else {
      auto value = Boolean(f);
      if (!value) return std::unexpected(value.error());
      if (f.number == 3) spec.add_space_prefix = *value;
      if (f.number == 4) spec.remove_extra_whitespaces = *value;
      // Public Gemma3 models carry an unassigned normalizer field 7=0.
      // The published SentencePiece v0.2.1 schema ignores this extension;
      // admit its disabled form only, rather than guessing a true meaning.
      if (f.number == 7 && *value)
        return Fail(Rule::kUnsupported, "enabled SentencePiece normalizer extension", f.offset);
      if (f.number == 5 && !*value)
        return Fail(Rule::kUnsupported, "SentencePiece without whitespace escaping", f.offset);
    }
  }
  return {};
}

}  // namespace

std::expected<TokenizerSpec, Error> ReadSentencePieceTokenizer(std::string_view bytes) {
  if (bytes.size() > kMaxModelBytes)
    return Fail(Rule::kBounds, "SentencePiece model exceeds its byte bound", bytes.size());
  TokenizerSpec spec;
  spec.pre_tokenizer = PreTokenizer::kSentencePiece;
  spec.decode_added_whitespace = true;
  spec.add_space_prefix = true;
  spec.remove_extra_whitespaces = true;
  Wire wire(bytes);
  Trainer trainer;
  std::optional<TokenId> unknown;
  std::array<bool, 6> seen{};
  for (;;) {
    auto read = wire.Next();
    if (!read) return std::unexpected(read.error());
    if (!*read) break;
    const auto& f = **read;
    if (f.wire != 2)
      return Fail(Rule::kFormat, "SentencePiece model field has the wrong type", f.offset);
    if (f.number > 5) return Fail(Rule::kUnsupported, "SentencePiece model extension", f.offset);
    if (f.number != 1 && seen[f.number])
      return Fail(Rule::kFormat, "repeated SentencePiece model field", f.offset);
    seen[f.number] = true;
    if (f.number == 1) {
      if ((spec.tokens.size() & 0xFFFFU) == 0 && !base::Pulse())
        return Fail(Rule::kCancelled, "the tokenizer reader was asked to stop", f.offset);
      if (auto piece = Piece(f, spec, unknown); !piece) return std::unexpected(piece.error());
    } else if (f.number == 2) {
      auto value = ReadTrainer(f);
      if (!value) return std::unexpected(value.error());
      trainer = *value;
    } else if (f.number == 3) {
      if (auto normalizer = ReadNormalizer(f, spec); !normalizer)
        return std::unexpected(normalizer.error());
    } else if (f.number == 5 && !f.bytes.empty()) {
      return Fail(Rule::kUnsupported, "SentencePiece denormalizer", f.offset);
    }
  }
  if (trainer.model != 2) return Fail(Rule::kUnsupported, "SentencePiece model is not scored BPE");
  if (!trainer.byte_fallback)
    return Fail(Rule::kUnsupported, "SentencePiece without byte fallback");
  if (!unknown || *unknown != trainer.unknown)
    return Fail(Rule::kVocabulary, "SentencePiece unknown ID does not name its unknown piece");
  const auto special = [&](std::int32_t id) {
    return id == -1 || (id >= 0 && std::cmp_less(id, spec.tokens.size()) &&
                        spec.kinds[static_cast<std::size_t>(id)] == TokenKind::kControl);
  };
  if (!special(trainer.bos) || !special(trainer.eos))
    return Fail(Rule::kVocabulary, "SentencePiece BOS or EOS does not name a control piece");
  if (trainer.bos >= 0) spec.bos = trainer.bos;
  if (trainer.eos >= 0) spec.eos = trainer.eos;
  // ModelProto names BOS/EOS; the tokenizer_config, rather than ModelProto,
  // determines whether a Transformers caller inserts them automatically.
  return spec;
}

}  // namespace llmp::tokenizer
