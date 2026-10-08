// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Native byte-level BPE, Gemma4 raw UTF-8 BPE and classic SentencePiece
// tokenization (D-067, D-088). It encodes text
// to token IDs and decodes them back, from a vocabulary that a loader
// (gguf.h, hf.h) reads from untrusted model files, validated when the
// tokenizer is created.
//
// Byte-level BPE encoding, as llama.cpp and Hugging Face tokenizers do it for these
// vocabularies:
//   1. Special tokens are matched in the raw text first, leftmost-longest.
//      User-defined tokens (Hugging Face's non-special added tokens) always
//      match; control tokens only under SpecialTokens::kParse, or where a
//      renderer marks them (EncodeMarked). Each match becomes its token.
//   2. Each fragment between them is normalized (NFC, when the tokenizer
//      has that normalizer), split by the pre-tokenizer, and each piece's
//      UTF-8 bytes are mapped to the GPT-2 byte alphabet.
//   3. BPE merges adjacent symbols, lowest merge rank first, leftmost on a
//      tie, until no ranked pair remains; each final symbol is a token.
// Byte fallback: every byte has a token (checked at creation), and every
// merge's result is a token, so any well-formed text encodes, and nothing
// is ever dropped.
//
// Inputs are untrusted and bounded: text must be well-formed UTF-8 (refused
// with its offset otherwise; unicode::ReplaceInvalidUtf8 is the caller's
// explicit lossy choice), at most EncodeOptions::max_bytes long, and the
// output at most max_tokens long. A text with more bytes than max_tokens
// times the longest token's (four times that under NFC, which shortens text
// at most threefold) cannot fit and is refused before any work, unless
// special whitespace stripping can shorten the input. Gemma4
// uses raw code-point symbols with ▁ for spaces, newline-run splitting
// and <0xNN> fallback; it encodes whole fragments (WorkingBytes), since
// merges may cross spaces. Classic SentencePiece likewise uses raw code
// points, optional dummy ▁ prefixes and highest-score, leftmost merging
// across a whole fragment. Missing symbols emit byte fallbacks; normal
// raw tokens decode ▁ to a space. Work is
// O(n log n), with up to 256 special-token trie steps per input byte, and
// byte-BPE memory follows the longest window, not the text (D-102): a fragment is
// encoded in windows of about kEncodeWindowBytes, cut only where neither
// normalization nor its byte-BPE pre-tokenizer joins across (EncodeWorkingBytes).
// A Tokenizer is immutable once created; Encode and Decode are safe to call
// from several threads at once.

#ifndef LLMP_TOKENIZER_TOKENIZER_H_
#define LLMP_TOKENIZER_TOKENIZER_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tokenizer/error.h"
#include "tokenizer/pretokenize.h"

namespace llmp::tokenizer {

using TokenId = std::int32_t;

enum class TokenKind : std::uint8_t {
  kNormal,       // text; produced by BPE
  kByte,         // raw UTF-8 BPE fallback, <0xNN>; decodes to one byte
  kControl,      // special: matched only when asked, decoded only when asked
  kUserDefined,  // always matched in text, decoded as its text
  kUnused,       // padding past the real vocabulary: never produced, decodes to nothing
};

enum class Normalization : std::uint8_t { kNone, kNfc };

// A vocabulary as a loader reads it; Tokenizer::Create validates it.
struct TokenizerSpec {
  std::vector<std::string> tokens;  // by ID; byte alphabet, or raw UTF-8 for raw modes
  std::vector<TokenKind> kinds;     // by ID
  std::vector<std::pair<std::string, std::string>> merges;  // in rank order
  PreTokenizer pre_tokenizer = PreTokenizer::kQwen2;
  // Classic SentencePiece uses token scores, rather than a merge list.
  std::vector<float> scores;
  bool add_space_prefix = false;
  std::vector<bool> rstrip;  // optional per-special ASCII whitespace stripping
  Normalization normalization = Normalization::kNone;
  bool ignore_merges = false;  // a piece that is a whole token skips BPE
  std::optional<TokenId> bos;
  std::optional<TokenId> eos;
  bool add_bos = false;
  bool add_eos = false;
};

// Caps on what Create accepts.
inline constexpr std::size_t kMaxVocabulary = std::size_t{1} << 22U;
inline constexpr std::size_t kMaxTokenBytes = 1024;

// Byte-BPE encoding's windows: a fragment longer than this is cut, at the last cut
// point within it (or the first after it, if none), into windows encoded
// one at a time. A cut point is before an ASCII letter or digit that
// follows a newline, or before a space between an ASCII letter or digit and
// an ASCII letter: no pre-tokenizer's piece and no NFC composition spans
// one, so the tokens are those of the whole fragment.
inline constexpr std::size_t kEncodeWindowBytes = std::size_t{1} << 20U;
// What encoding allocates beside its output, at most, per byte of its
// longest window: the code points (4), NFC's decomposition (up to four code
// points a character, its vector grown by doubling: 32), the pieces'
// lengths and the pre-tokenizer's stages (12), and a piece's merging
// (symbols, links and candidates, 36; its bytes, up to 3 more), rounded up.
inline constexpr std::size_t kEncodeBytesPerWindowByte = 96;

enum class SpecialTokens : std::uint8_t {
  kUserDefinedOnly,  // llama.cpp's parse_special = false
  kParse,            // also control tokens: llama.cpp's parse_special = true, HF's default
};

struct EncodeOptions {
  SpecialTokens special = SpecialTokens::kUserDefinedOnly;
  bool add_bos_eos = false;  // add BOS and EOS where the vocabulary asks for them
  std::size_t max_bytes = std::size_t{4} << 20U;
  std::size_t max_tokens = std::size_t{1} << 22U;
};

// A control token a renderer placed: `length` bytes at `offset` of the
// rendered text, which must be exactly that token's text.
struct SpecialSpan {
  std::size_t offset = 0;
  std::size_t length = 0;
};

struct DecodeOptions {
  bool control_tokens = false;       // decode control tokens as their text
  bool remove_space_prefix = false;  // remove SentencePiece dummy space at stream start
};

class Tokenizer {
 public:
  static std::expected<Tokenizer, Error> Create(TokenizerSpec spec);

  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  ~Tokenizer();

  std::size_t size() const;
  PreTokenizer pre_tokenizer() const;
  Normalization normalization() const;
  std::optional<TokenId> bos() const;
  std::optional<TokenId> eos() const;
  bool adds_bos() const;
  // A token's text and kind (id < size()).
  std::string_view Text(TokenId id) const;
  TokenKind Kind(TokenId id) const;
  // The token whose text is exactly `text`, of any kind.
  std::optional<TokenId> Find(std::string_view text) const;
  // The longest token's bytes: a normal token's decoded bytes (raw UTF-8
  // bytes for raw modes, accounting for literal ▁ in the input), a special
  // token's text. Without stripping, n bytes need at least n / this tokens.
  std::size_t longest_token_bytes() const;

  // Byte-BPE estimate (use WorkingBytes when the tokenizer is known).
  // The bytes Encode or EncodeMarked allocates for `text` at most, beside
  // its output: kEncodeBytesPerWindowByte a byte of its longest window
  // (kEncodeWindowBytes, or longer where the text has no cut point). A
  // caller charges this before encoding untrusted text. A scan of the text.
  static std::uint64_t EncodeWorkingBytes(std::string_view text);
  // Mode-aware estimate: raw modes can merge across spaces, so they
  // charge the whole input rather than byte-BPE windows. SentencePiece
  // adds one input byte to cover its optional dummy prefix.
  std::uint64_t WorkingBytes(std::string_view text) const;

  // Appends text's tokens to out.
  std::expected<void, Error> Encode(std::string_view text, const EncodeOptions& options,
                                    std::vector<TokenId>& out) const;
  // Encodes a renderer's text: the spans are control tokens wherever they
  // are (in order, not overlapping), and control tokens elsewhere stay text
  // whatever options.special says; user-defined tokens match as always.
  // span_tokens, when given, receives each span's index in out.
  std::expected<void, Error> EncodeMarked(std::string_view text, std::span<const SpecialSpan> spans,
                                          const EncodeOptions& options, std::vector<TokenId>& out,
                                          std::vector<std::size_t>* span_tokens = nullptr) const;

  // Appends the tokens' bytes to out. Byte-level tokens can end inside a
  // UTF-8 sequence, so a partial sequence may end the output; StreamDecoder
  // holds those back.
  std::expected<void, Error> Decode(std::span<const TokenId> tokens, const DecodeOptions& options,
                                    std::string& out) const;

 private:
  struct Impl;
  explicit Tokenizer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Decodes a token stream into complete UTF-8: each Push appends what the
// token completes, holding back a trailing partial sequence; Finish flushes
// it. Ill-formed bytes become U+FFFD (unicode::ReplaceInvalidUtf8).
class StreamDecoder {
 public:
  StreamDecoder(const Tokenizer& tokenizer, DecodeOptions options)
      : tokenizer_(&tokenizer), options_(options) {}

  std::expected<void, Error> Push(TokenId token, std::string& out);
  void Finish(std::string& out);

 private:
  const Tokenizer* tokenizer_;
  DecodeOptions options_;
  std::string pending_;
  bool started_ = false;
};

}  // namespace llmp::tokenizer

#endif  // LLMP_TOKENIZER_TOKENIZER_H_
