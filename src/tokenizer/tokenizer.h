// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Native byte-level BPE, raw UTF-8 BPE and scored SentencePiece BPE
// tokenization (D-067, D-088). It encodes text
// to token IDs and decodes them back, from a vocabulary that a loader
// (gguf.h, hf.h, sentencepiece.h) reads from untrusted model files, validated when the
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
// Byte coverage: every byte has a fallback token (checked at creation), or
// raw BPE has a normal model token for that unchanged ASCII character. Every
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
// raw tokens decode ▁ to a space. Native ModelProto user symbols normalize
// with their entire region and freeze afterward with their exact model spelling;
// external added-token extraction remains independent of model membership.
// Native UNUSED pieces can merge, then resegment iteratively at emission. Work is
// O(n log n), with up to 256 special-token trie steps per input byte, and
// regex work is bounded by input length and compiled program size. Whole
// fragments are encoded and funded: arbitrary expressions may join across
// spaces or newlines.
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

// Compact source recipe: definitions are stored once, while the ordered
// wrappers refer to checked slices. Repeated multi-ID names never expand
// construction storage; encoding streams the actual IDs under its output bound.
struct PostProcessorSlice {
  std::size_t offset = 0, length = 0;
};
struct PostProcessorPlan {
  std::vector<TokenId> ids;
  std::vector<PostProcessorSlice> prefix, suffix;
};

// A vocabulary as a loader reads it; Tokenizer::Create validates it.
struct TokenizerSpec {
  std::vector<std::string> tokens;  // by ID; byte alphabet, or raw UTF-8 for raw modes
  std::vector<TokenKind> kinds;     // by ID
  std::vector<bool> model_tokens;   // optional BPE membership, independent of added-token matching
  std::vector<bool> model_symbols;  // native SentencePiece user symbols protected by normalization
  std::vector<bool> added_tokens;   // optional external token matcher membership
  std::vector<bool> normalized_tokens;  // optional added-token matching after normalization
  std::vector<std::pair<std::string, std::string>> merges;  // in rank order
  PreTokenizer pre_tokenizer = PreTokenizer::kByteBpe;
  std::vector<std::string> split_expressions = DefaultByteSplits();
  RegexDialect split_dialect = RegexDialect::kHuggingFace;
  bool decode_added_whitespace = false;  // declared decoder also replaces markers in added tokens
  bool raw_bpe_split_newlines = true;    // GGUF raw BPE; HF may merge across newlines
  // Classic SentencePiece uses token scores, rather than a merge list.
  std::vector<float> scores;
  bool add_space_prefix = false;
  bool remove_extra_whitespaces = false;  // SentencePiece identity-map trim/collapse
  std::vector<bool> rstrip;               // optional per-special ASCII whitespace stripping
  Normalization normalization = Normalization::kNone;
  bool ignore_merges = false;  // a piece that is a whole token skips BPE
  std::optional<TokenId> bos;
  std::optional<TokenId> eos;
  bool add_bos = false;
  bool add_eos = false;
  // Source insertion recipe, independent of the named BOS/EOS roles. An
  // explicitly empty processor overrides the GGUF/config insertion flags.
  bool has_post_processor = false;
  PostProcessorPlan post_processor;
};

// Caps on what Create accepts.
inline constexpr std::size_t kMaxVocabulary = std::size_t{1} << 22U;
inline constexpr std::size_t kMaxTokenBytes = 1024;

// Per normalized input byte, excluding regex VM scratch and output. Whole
// fragments are charged because user-supplied expressions can cross spaces.
inline constexpr std::size_t kEncodeBytesPerInputByte = 96;

enum class SpecialTokens : std::uint8_t {
  kUserDefinedOnly,  // llama.cpp's parse_special = false
  kParse,            // also control tokens: llama.cpp's parse_special = true, HF's default
};

struct EncodeOptions {
  SpecialTokens special = SpecialTokens::kUserDefinedOnly;
  bool add_bos_eos = false;  // apply the complete declared source postprocessor
  std::size_t max_bytes = std::size_t{4} << 20U;
  std::size_t max_tokens = std::size_t{1} << 22U;
  // The text continues a stream: SentencePiece adds no dummy ▁ prefix to any
  // of its fragments, so the bytes encode as written. Other modes ignore it.
  bool continuation = false;
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

// Planning never accepts published IDs: selected is still a local sampling
// candidate. All returned tokens replace that candidate and include its bytes.
// Forced text is a continuation: SentencePiece adds no dummy prefix unless
// decode.remove_space_prefix says the span starts the stream.
struct ForcedSpanOptions {
  EncodeOptions encode{.special = SpecialTokens::kParse};
  DecodeOptions decode{.control_tokens = true};
  // Marked: exactly `specials` are control tokens, as EncodeMarked places
  // them, and control-token text elsewhere is text (encode.special is then
  // ignored; user-defined tokens still match). Offsets are in the planned
  // text: for PlanForcedSpan the selected token's decoded bytes followed by
  // `forced`, so a span may begin inside the selected token's bytes; a
  // selected control token is itself a span (a supplied identical span is
  // merged, any other overlap with it is refused). For ForcedCoverBytes they
  // are in `forced`. Spans are in order, do not overlap and are each exactly
  // a special token's text.
  bool marked = false;
  std::span<const SpecialSpan> specials;
  // Complete allowed NEXT-BYTE set, not Unicode representatives. High-byte
  // possibilities conservatively retain an open tail; byte-cover accepts them.
  std::span<const std::uint8_t> following_bytes;
  bool following_known = false;
  bool terminal = false;  // actual hard grammar boundary, never max_tokens
  std::size_t remaining_tokens = 0;
  std::size_t remaining_context = 0;
};
struct ForcedSpanPlan {
  std::vector<TokenId> tokens;
  std::size_t committed_bytes = 0;        // decoded selected + forced prefix
  std::size_t consumed_forced_bytes = 0;  // excludes the selected token's bytes
  std::string held;                       // exact uncommitted suffix of both inputs
  std::size_t cover_bytes = 0;
  bool replaces_selected = false;
  // False: this plan supplies no stable replacement or certain suffix. An
  // unchanged ordinary sampler choice is a separate caller disposition.
  bool selected_covered = false;
  // Local consistency with the finite known suffix only, NOT a proof against
  // unknown future merges. Permits an unchanged genuine sampler choice while
  // an open block remains held; never authorizes inserting unstable IDs.
  bool selected_matches_canonical_prefix = false;
  bool limited = false;  // real token/context allowance cut the plan
};

class Tokenizer {
 public:
  // Allocation-free bound for Create's peak and retained storage, including
  // capacities moved from the reader. Pay before passing ownership to Create.
  // This does not replace Create's semantic validation.
  static std::expected<std::uint64_t, Error> ConstructionBytes(const TokenizerSpec& spec);
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

  // Saturating bounds, including whole-fragment normalization, merging,
  // split vectors and the bounded VM's registers/choices. Output is separate.
  std::uint64_t WorkingBytes(std::string_view text) const;
  std::uint64_t WorkingBytes(std::size_t input_bytes) const;
  std::uint64_t OutputTokensBound(std::size_t input_bytes, bool source_postprocessor = false) const;
  // Reserve the complete bounded append before emission. The caller funds
  // existing storage plus this replacement capacity (including their overlap).
  // Extra slots are for caller-owned suffix/prefix policy, not emitted tokens;
  // max_tokens still bounds only this encoding's append. Encode/EncodeMarked
  // do this automatically with no extra slots, after input preflight.
  std::expected<void, Error> ReserveOutput(std::string_view text, const EncodeOptions& options,
                                           std::vector<TokenId>& out,
                                           std::size_t extra_tokens = 0) const;

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
  // The control tokens SpecialTokens::kParse encoding finds in text, in order:
  // leftmost-longest matches, where a longer user-defined match hides a
  // control one, as EncodeMarked's spans. Fails only when cancelled.
  std::expected<std::vector<SpecialSpan>, Error> ControlSpans(std::string_view text) const;
  // True when some substring of text is the text of a control token the
  // matcher knows (in its normalized spelling for normalized added tokens),
  // wherever it starts. O(text bytes * longest special token); fails only
  // when cancelled.
  std::expected<bool, Error> ContainsControlText(std::string_view text) const;

  // Peak working + result ownership, before either planner/cover operation.
  // input_bytes includes the selected token's decoded bytes and forced bytes.
  std::uint64_t ForcedSpanBytes(std::size_t input_bytes) const;
  std::expected<ForcedSpanPlan, Error> PlanForcedSpan(std::optional<TokenId> selected,
                                                      std::string_view forced,
                                                      const ForcedSpanOptions& options) const;
  // Boundary cover applies only when the forced prefix is one proven closed
  // actual block represented by one exact canonical token. Open blocks return
  // zero: a later crossing merge can split a token encoded in isolation. Lexical permission for any
  // extra candidate bytes is checked separately by the caller's constraint cursor.
  std::expected<std::size_t, Error> ForcedCoverBytes(std::string_view forced,
                                                     const ForcedSpanOptions& options) const;
  bool CoversForced(TokenId candidate, std::size_t cover_bytes,
                    const DecodeOptions& options = {.control_tokens = true}) const;

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
