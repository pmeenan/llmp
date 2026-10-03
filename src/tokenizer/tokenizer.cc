// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizer/error.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

namespace jitllm::tokenizer {
namespace {

// Special tokens longer than this are refused, which bounds the matcher's
// work per input byte.
constexpr std::size_t kMaxSpecialBytes = 256;
constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

bool AsciiLetter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool AsciiAlnum(char c) { return AsciiLetter(c) || (c >= '0' && c <= '9'); }

// Whether encoding may cut `text` before byte p (0 < p < size): before an
// ASCII letter or digit that follows a newline, or before a space between
// an ASCII letter or digit and an ASCII letter. Every pre-tokenizer's piece
// ends there whatever follows (a newline ends its run, and a letter or
// digit run ends at a space; no alternative joins a newline to a following
// letter or digit, or a space to a preceding one), and matching from there
// looks only forward; NFC composes neither (each side is a starter that
// composes with nothing before it).
bool CutAt(std::string_view text, std::size_t p) {
  const char before = text[p - 1];
  const char at = text[p];
  if (before == '\n') {
    return AsciiAlnum(at);
  }
  return at == ' ' && p + 1 < text.size() && AsciiAlnum(before) && AsciiLetter(text[p + 1]);
}

// The end of the window that starts at `start`: the last cut point within
// kEncodeWindowBytes of it, else the first after, else the text's end.
// Linear over a text: a window's backward scan crosses only bytes past the
// last window's cut point, or ends in a forward scan the next window starts
// after.
std::size_t WindowEnd(std::string_view text, std::size_t start) {
  if (text.size() - start <= kEncodeWindowBytes) {
    return text.size();
  }
  const std::size_t limit = start + kEncodeWindowBytes;
  for (std::size_t p = limit; p > start; --p) {
    if (CutAt(text, p)) {
      return p;
    }
  }
  for (std::size_t p = limit + 1; p < text.size(); ++p) {
    if (CutAt(text, p)) {
      return p;
    }
  }
  return text.size();
}

// a * b, saturating.
std::size_t Times(std::size_t a, std::size_t b) {
  return a != 0 && b > std::numeric_limits<std::size_t>::max() / a
             ? std::numeric_limits<std::size_t>::max()
             : a * b;
}

// GPT-2's byte alphabet: each byte is one code point; printable Latin-1
// bytes are themselves, the rest are 256 and up in byte order.
std::array<char32_t, 256> ByteAlphabet() {
  std::array<char32_t, 256> map{};
  char32_t next = 256;
  for (unsigned b = 0; b < 256; ++b) {
    const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
    map[b] = printable ? static_cast<char32_t>(b) : next++;
  }
  return map;
}

struct MergeResult {
  std::uint32_t rank;
  TokenId token;
};

std::uint64_t PairKey(TokenId a, TokenId b) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32U) |
         static_cast<std::uint32_t>(b);
}

// std::hash of an integer is the identity in libstdc++, so a hostile
// vocabulary could choose merges or special tokens whose keys all fall in
// one bucket and make creation quadratic; mix the bits first (SplitMix64's
// finalizer).
struct MixHash {
  std::size_t operator()(std::uint64_t x) const {
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9U;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBU;
    return static_cast<std::size_t>(x ^ (x >> 31U));
  }
};

}  // namespace

std::string_view RuleName(Rule rule) {
  switch (rule) {
    case Rule::kInvalidUtf8:
      return "invalid-utf8";
    case Rule::kInputTooLarge:
      return "input-too-large";
    case Rule::kOutputTooLarge:
      return "output-too-large";
    case Rule::kInvalidToken:
      return "invalid-token";
    case Rule::kSpecialToken:
      return "special-token";
    case Rule::kFormat:
      return "format";
    case Rule::kUnsupported:
      return "unsupported";
    case Rule::kVocabulary:
      return "vocabulary";
    case Rule::kBounds:
      return "bounds";
  }
  return "unknown";
}

std::string Error::ToString() const {
  std::string s(RuleName(rule));
  s += ": ";
  s += reason;
  if (item != kNoItem) {
    s += " (item ";
    s += std::to_string(item);
    s += ")";
  }
  return s;
}

struct Tokenizer::Impl {
  std::vector<std::string> tokens;
  std::vector<TokenKind> kinds;
  std::vector<std::string> decoded;  // normal tokens' bytes
  std::unordered_map<std::string_view, TokenId> by_text;
  std::array<TokenId, 256> byte_token{};
  std::array<std::string, 256> byte_text;  // each byte's alphabet character, UTF-8
  std::unordered_map<std::uint64_t, MergeResult, MixHash> merges;
  // The special-token trie: node 0 is the root; edges keyed by
  // (node << 8 | byte); terminal[node] is the token ending there, or -1.
  std::unordered_map<std::uint64_t, std::uint32_t, MixHash> edges;
  std::vector<TokenId> terminal;
  std::array<bool, 256> special_first{};
  PreTokenizer pre_tokenizer = PreTokenizer::kQwen2;
  Normalization normalization = Normalization::kNone;
  bool ignore_merges = false;
  std::optional<TokenId> bos;
  std::optional<TokenId> eos;
  bool add_bos = false;
  bool add_eos = false;
  std::size_t longest = 1;  // the longest token's bytes (decoded, or a special's text)

  // The longest special token matching at text[p], allowed by `special`,
  // or -1.
  TokenId MatchSpecial(std::string_view text, std::size_t p, SpecialTokens special,
                       std::size_t* length) const {
    if (!special_first[static_cast<unsigned char>(text[p])]) {
      return -1;
    }
    std::uint32_t node = 0;
    TokenId best = -1;
    for (std::size_t q = p; q < text.size() && q - p < kMaxSpecialBytes; ++q) {
      const auto it = edges.find((static_cast<std::uint64_t>(node) << 8U) |
                                 static_cast<unsigned char>(text[q]));
      if (it == edges.end()) {
        break;
      }
      node = it->second;
      const TokenId t = terminal[node];
      if (t >= 0 && (special == SpecialTokens::kParse ||
                     kinds[static_cast<std::size_t>(t)] == TokenKind::kUserDefined)) {
        best = t;
        *length = q - p + 1;
      }
    }
    return best;
  }

  static std::expected<void, Error> Emit(TokenId id, const EncodeOptions& options,
                                         std::vector<TokenId>& out, std::size_t base) {
    if (out.size() - base >= options.max_tokens) {
      return std::unexpected(
          Error{Rule::kOutputTooLarge, "more tokens than max_tokens", options.max_tokens});
    }
    out.push_back(id);
    return {};
  }

  // BPE over one piece's symbols (token IDs of its bytes), appending the
  // result.
  std::expected<void, Error> Merge(std::vector<TokenId>& symbols, const EncodeOptions& options,
                                   std::vector<TokenId>& out, std::size_t base) const {
    const std::size_t n = symbols.size();
    if (n > 1) {
      std::vector<std::uint32_t> next(n);
      std::vector<std::uint32_t> prev(n);
      for (std::size_t i = 0; i < n; ++i) {
        next[i] = i + 1 < n ? static_cast<std::uint32_t>(i + 1) : kNone;
        prev[i] = i > 0 ? static_cast<std::uint32_t>(i - 1) : kNone;
      }
      // (rank, left position): the lowest rank first, then the leftmost.
      using Candidate = std::pair<std::uint32_t, std::uint32_t>;
      // At most n - 1 first candidates and two a merge, so its storage
      // never doubles past them (kEncodeBytesPerWindowByte).
      std::vector<Candidate> storage;
      storage.reserve((3 * n) - 3);
      std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue(
          std::greater<>(), std::move(storage));
      auto consider = [&](std::uint32_t left) {
        if (left == kNone || next[left] == kNone) {
          return;
        }
        const auto it = merges.find(PairKey(symbols[left], symbols[next[left]]));
        if (it != merges.end()) {
          queue.emplace(it->second.rank, left);
        }
      };
      for (std::size_t i = 0; i + 1 < n; ++i) {
        consider(static_cast<std::uint32_t>(i));
      }
      while (!queue.empty()) {
        const auto [rank, left] = queue.top();
        queue.pop();
        const std::uint32_t right = next[left];
        if (symbols[left] < 0 || right == kNone) {
          continue;  // merged away since
        }
        const auto it = merges.find(PairKey(symbols[left], symbols[right]));
        if (it == merges.end() || it->second.rank != rank) {
          continue;  // the pair here changed since
        }
        symbols[left] = it->second.token;
        symbols[right] = -1;
        next[left] = next[right];
        if (next[right] != kNone) {
          prev[next[right]] = left;
        }
        consider(prev[left]);
        consider(left);
      }
    }
    for (const TokenId s : symbols) {
      if (s >= 0) {
        if (auto e = Emit(s, options, out, base); !e) {
          return e;
        }
      }
    }
    return {};
  }

  // Encodes text containing no special token (a fragment), a window at a
  // time (kEncodeWindowBytes).
  std::expected<void, Error> EncodeFragment(std::string_view text, const EncodeOptions& options,
                                            std::vector<TokenId>& out, std::size_t base) const {
    for (std::size_t start = 0; start < text.size();) {
      const std::size_t end = WindowEnd(text, start);
      if (auto e = EncodeWindow(text.substr(start, end - start), options, out, base); !e) {
        return e;
      }
      start = end;
    }
    return {};
  }

  std::expected<void, Error> EncodeWindow(std::string_view text, const EncodeOptions& options,
                                          std::vector<TokenId>& out, std::size_t base) const {
    if (text.empty()) {
      return {};
    }
    std::vector<char32_t> cps;
    if (auto d = unicode::DecodeUtf8(text, cps); !d) {
      return d;
    }
    if (normalization == Normalization::kNfc) {
      unicode::ToNfc(cps);
    }
    std::vector<std::uint32_t> lengths;
    lengths.reserve(cps.size());  // a piece a code point at most
    PreTokenize(pre_tokenizer, cps, lengths);
    std::size_t start = 0;
    std::string bytes;
    std::string alphabet;
    std::vector<TokenId> symbols;
    for (const std::uint32_t length : lengths) {
      bytes.clear();
      for (std::size_t i = start; i < start + length; ++i) {
        unicode::AppendUtf8(cps[i], bytes);
      }
      start += length;
      if (ignore_merges) {
        alphabet.clear();
        for (const char b : bytes) {
          alphabet += byte_text[static_cast<unsigned char>(b)];
        }
        if (const auto it = by_text.find(alphabet);
            it != by_text.end() &&
            kinds[static_cast<std::size_t>(it->second)] == TokenKind::kNormal) {
          if (auto e = Emit(it->second, options, out, base); !e) {
            return e;
          }
          continue;
        }
      }
      symbols.clear();
      for (const char b : bytes) {
        symbols.push_back(byte_token[static_cast<unsigned char>(b)]);
      }
      if (auto m = Merge(symbols, options, out, base); !m) {
        return m;
      }
    }
    return {};
  }

  std::expected<void, Error> Begin(std::string_view text, const EncodeOptions& options,
                                   std::vector<TokenId>& out, std::size_t base) const {
    if (text.size() > options.max_bytes) {
      return std::unexpected(
          Error{Rule::kInputTooLarge, "text longer than max_bytes", options.max_bytes});
    }
    // Every token covers at most `longest` bytes of normalized text, and NFC
    // shortens text at most threefold: a longer text has more tokens than
    // max_tokens, which is known before any work.
    if (text.size() >
        Times(Times(options.max_tokens, longest), normalization == Normalization::kNfc ? 4 : 1)) {
      return std::unexpected(
          Error{Rule::kOutputTooLarge, "more tokens than max_tokens", options.max_tokens});
    }
    if (auto valid = unicode::ValidateUtf8(text); !valid) {
      return valid;
    }
    if (options.add_bos_eos && add_bos && bos) {  // Create checked add_bos names a BOS
      return Emit(*bos, options, out, base);
    }
    return {};
  }

  std::expected<void, Error> End(const EncodeOptions& options, std::vector<TokenId>& out,
                                 std::size_t base) const {
    if (options.add_bos_eos && add_eos && eos) {
      return Emit(*eos, options, out, base);
    }
    return {};
  }

  // Encodes text, matching special tokens as `special` allows.
  std::expected<void, Error> EncodeText(std::string_view text, SpecialTokens special,
                                        const EncodeOptions& options, std::vector<TokenId>& out,
                                        std::size_t base) const {
    std::size_t fragment = 0;
    for (std::size_t p = 0; p < text.size();) {
      std::size_t length = 0;
      const TokenId t = MatchSpecial(text, p, special, &length);
      if (t < 0) {
        ++p;
        continue;
      }
      if (auto e = EncodeFragment(text.substr(fragment, p - fragment), options, out, base); !e) {
        return e;
      }
      if (auto e = Emit(t, options, out, base); !e) {
        return e;
      }
      p += length;
      fragment = p;
    }
    return EncodeFragment(text.substr(fragment), options, out, base);
  }
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

std::expected<Tokenizer, Error> Tokenizer::Create(TokenizerSpec spec) {
  auto fail = [](Rule rule, std::string_view reason, std::uint64_t item = kNoItem) {
    return std::unexpected(Error{rule, reason, item});
  };
  const std::size_t n = spec.tokens.size();
  if (n == 0 || n > kMaxVocabulary) {
    return fail(Rule::kBounds, "vocabulary size", n);
  }
  if (spec.kinds.size() != n) {
    return fail(Rule::kVocabulary, "token kinds and tokens differ in number", spec.kinds.size());
  }
  if (spec.merges.size() > 2 * kMaxVocabulary) {
    return fail(Rule::kBounds, "merge count", spec.merges.size());
  }
  auto impl = std::make_unique<Impl>();
  impl->tokens = std::move(spec.tokens);
  impl->kinds = std::move(spec.kinds);
  impl->pre_tokenizer = spec.pre_tokenizer;
  impl->normalization = spec.normalization;
  impl->ignore_merges = spec.ignore_merges;
  impl->bos = spec.bos;
  impl->eos = spec.eos;
  impl->add_bos = spec.add_bos;
  impl->add_eos = spec.add_eos;

  const std::array<char32_t, 256> alphabet = ByteAlphabet();
  std::unordered_map<char32_t, unsigned char> byte_of;
  for (unsigned b = 0; b < 256; ++b) {
    byte_of.emplace(alphabet[b], static_cast<unsigned char>(b));
    unicode::AppendUtf8(alphabet[b], impl->byte_text[b]);
  }

  impl->decoded.resize(n);
  impl->by_text.reserve(n);
  impl->terminal.push_back(-1);
  std::vector<char32_t> cps;
  for (std::size_t id = 0; id < n; ++id) {
    const std::string& text = impl->tokens[id];
    const TokenKind kind = impl->kinds[id];
    if (kind == TokenKind::kUnused) {
      continue;
    }
    if (text.empty() || text.size() > kMaxTokenBytes) {
      return fail(Rule::kVocabulary, "token text empty or too long", id);
    }
    cps.clear();
    if (!unicode::DecodeUtf8(text, cps)) {
      return fail(Rule::kVocabulary, "token text is not UTF-8", id);
    }
    if (!impl->by_text.emplace(text, static_cast<TokenId>(id)).second) {
      return fail(Rule::kVocabulary, "two tokens with one text", id);
    }
    if (kind == TokenKind::kNormal) {
      std::string& bytes = impl->decoded[id];
      for (const char32_t cp : cps) {
        const auto it = byte_of.find(cp);
        if (it == byte_of.end()) {
          return fail(Rule::kVocabulary, "normal token outside the byte alphabet", id);
        }
        bytes.push_back(static_cast<char>(it->second));
      }
      impl->longest = std::max(impl->longest, bytes.size());
      continue;
    }
    impl->longest = std::max(impl->longest, text.size());
    // A special token: add it to the trie.
    if (text.size() > kMaxSpecialBytes) {
      return fail(Rule::kBounds, "special token longer than the matcher's bound", id);
    }
    std::uint32_t node = 0;
    for (const char c : text) {
      const std::uint64_t key =
          (static_cast<std::uint64_t>(node) << 8U) | static_cast<unsigned char>(c);
      const auto [it, added] =
          impl->edges.emplace(key, static_cast<std::uint32_t>(impl->terminal.size()));
      if (added) {
        impl->terminal.push_back(-1);
      }
      node = it->second;
    }
    impl->terminal[node] = static_cast<TokenId>(id);
    impl->special_first[static_cast<unsigned char>(text[0])] = true;
  }

  for (unsigned b = 0; b < 256; ++b) {
    const auto it = impl->by_text.find(impl->byte_text[b]);
    if (it == impl->by_text.end() ||
        impl->kinds[static_cast<std::size_t>(it->second)] != TokenKind::kNormal) {
      return fail(Rule::kVocabulary, "a byte has no normal token", b);
    }
    impl->byte_token[b] = it->second;
  }

  impl->merges.reserve(spec.merges.size());
  auto normal = [&impl](std::string_view text) -> std::optional<TokenId> {
    const auto it = impl->by_text.find(text);
    if (it == impl->by_text.end() ||
        impl->kinds[static_cast<std::size_t>(it->second)] != TokenKind::kNormal) {
      return std::nullopt;
    }
    return it->second;
  };
  std::string joined;
  for (std::size_t rank = 0; rank < spec.merges.size(); ++rank) {
    const auto& [left, right] = spec.merges[rank];
    const auto a = normal(left);
    const auto b = normal(right);
    joined = left;
    joined += right;
    const auto merged = normal(joined);
    if (!a || !b || !merged) {
      return fail(Rule::kVocabulary, "a merge names a token the vocabulary lacks", rank);
    }
    if (!impl->merges
             .emplace(PairKey(*a, *b), MergeResult{static_cast<std::uint32_t>(rank), *merged})
             .second) {
      return fail(Rule::kVocabulary, "a merge repeats an earlier one", rank);
    }
  }

  for (const auto& id : {impl->bos, impl->eos}) {
    if (id && (*id < 0 || std::cmp_greater_equal(*id, n) ||
               impl->kinds[static_cast<std::size_t>(*id)] == TokenKind::kUnused)) {
      return fail(Rule::kVocabulary, "BOS or EOS outside the vocabulary",
                  static_cast<std::uint64_t>(*id));
    }
  }
  if ((impl->add_bos && !impl->bos) || (impl->add_eos && !impl->eos)) {
    return fail(Rule::kVocabulary, "BOS or EOS added but not named");
  }
  return Tokenizer(std::move(impl));
}

std::size_t Tokenizer::size() const { return impl_->tokens.size(); }
PreTokenizer Tokenizer::pre_tokenizer() const { return impl_->pre_tokenizer; }
Normalization Tokenizer::normalization() const { return impl_->normalization; }
std::optional<TokenId> Tokenizer::bos() const { return impl_->bos; }
bool Tokenizer::adds_bos() const { return impl_->add_bos; }
std::optional<TokenId> Tokenizer::eos() const { return impl_->eos; }
std::string_view Tokenizer::Text(TokenId id) const {
  return impl_->tokens[static_cast<std::size_t>(id)];
}
TokenKind Tokenizer::Kind(TokenId id) const { return impl_->kinds[static_cast<std::size_t>(id)]; }

std::size_t Tokenizer::longest_token_bytes() const { return impl_->longest; }

std::uint64_t Tokenizer::EncodeWorkingBytes(std::string_view text) {
  std::size_t widest = 0;
  for (std::size_t start = 0; start < text.size();) {
    const std::size_t end = WindowEnd(text, start);
    widest = std::max(widest, end - start);
    start = end;
  }
  return std::uint64_t{widest} * kEncodeBytesPerWindowByte;
}

std::optional<TokenId> Tokenizer::Find(std::string_view text) const {
  const auto it = impl_->by_text.find(text);
  if (it == impl_->by_text.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::expected<void, Error> Tokenizer::Encode(std::string_view text, const EncodeOptions& options,
                                             std::vector<TokenId>& out) const {
  const std::size_t base = out.size();
  if (auto b = impl_->Begin(text, options, out, base); !b) {
    return b;
  }
  if (auto e = impl_->EncodeText(text, options.special, options, out, base); !e) {
    return e;
  }
  return impl_->End(options, out, base);
}

std::expected<void, Error> Tokenizer::EncodeMarked(std::string_view text,
                                                   std::span<const SpecialSpan> spans,
                                                   const EncodeOptions& options,
                                                   std::vector<TokenId>& out,
                                                   std::vector<std::size_t>* span_tokens) const {
  const std::size_t base = out.size();
  if (auto b = impl_->Begin(text, options, out, base); !b) {
    return b;
  }
  std::size_t at = 0;
  for (std::size_t i = 0; i < spans.size(); ++i) {
    const SpecialSpan& s = spans[i];
    if (s.offset < at || s.length == 0 || s.offset > text.size() ||
        text.size() - s.offset < s.length) {
      return std::unexpected(
          Error{Rule::kSpecialToken, "spans out of order or outside the text", i});
    }
    const auto token = Find(text.substr(s.offset, s.length));
    if (!token || Kind(*token) == TokenKind::kNormal || Kind(*token) == TokenKind::kUnused) {
      return std::unexpected(Error{Rule::kSpecialToken, "a span is not a special token", i});
    }
    if (auto e = impl_->EncodeText(text.substr(at, s.offset - at), SpecialTokens::kUserDefinedOnly,
                                   options, out, base);
        !e) {
      return e;
    }
    if (span_tokens != nullptr) {
      span_tokens->push_back(out.size());
    }
    if (auto e = Impl::Emit(*token, options, out, base); !e) {
      return e;
    }
    at = s.offset + s.length;
  }
  if (auto e =
          impl_->EncodeText(text.substr(at), SpecialTokens::kUserDefinedOnly, options, out, base);
      !e) {
    return e;
  }
  return impl_->End(options, out, base);
}

std::expected<void, Error> Tokenizer::Decode(std::span<const TokenId> tokens,
                                             const DecodeOptions& options, std::string& out) const {
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const TokenId id = tokens[i];
    if (id < 0 || static_cast<std::size_t>(id) >= impl_->tokens.size()) {
      return std::unexpected(Error{Rule::kInvalidToken, "token ID outside the vocabulary", i});
    }
    const auto u = static_cast<std::size_t>(id);
    switch (impl_->kinds[u]) {
      case TokenKind::kNormal:
        out += impl_->decoded[u];
        break;
      case TokenKind::kUserDefined:
        out += impl_->tokens[u];
        break;
      case TokenKind::kControl:
        if (options.control_tokens) {
          out += impl_->tokens[u];
        }
        break;
      case TokenKind::kUnused:
        break;
    }
  }
  return {};
}

std::expected<void, Error> StreamDecoder::Push(TokenId token, std::string& out) {
  if (auto d = tokenizer_->Decode(std::span(&token, 1), options_, pending_); !d) {
    return d;
  }
  const std::size_t complete = unicode::CompleteUtf8Prefix(pending_);
  out += unicode::ReplaceInvalidUtf8(std::string_view(pending_).substr(0, complete));
  pending_.erase(0, complete);
  return {};
}

void StreamDecoder::Finish(std::string& out) {
  out += unicode::ReplaceInvalidUtf8(pending_);
  pending_.clear();
}

}  // namespace jitllm::tokenizer
