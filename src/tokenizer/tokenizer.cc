// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <iterator>
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

#include "base/check.h"
#include "base/work_pulse.h"
#include "tokenizer/error.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

namespace llmp::tokenizer {
namespace {

// Special tokens longer than this are refused, which bounds the matcher's
// work per input byte.
constexpr std::size_t kMaxSpecialBytes = 256;
constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

// Leading UTF-8 continuation bytes: the rest of a character whose lead byte
// precedes the text. Four or more can never be well formed.
std::size_t ContinuationBytes(std::string_view text) {
  std::size_t n = 0;
  while (n < text.size() && n < 4 && (static_cast<unsigned char>(text[n]) & 0xC0U) == 0x80U) ++n;
  return n;
}

bool AsciiSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool Raw(PreTokenizer p) { return p == PreTokenizer::kRawBpe || p == PreTokenizer::kSentencePiece; }

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
    case Rule::kCancelled:
      return "cancelled";
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
  std::vector<bool> model_tokens;
  std::vector<bool> model_symbols;
  std::vector<bool> added_tokens;
  std::vector<bool> normalized_tokens;
  std::vector<float> scores;
  std::vector<bool> rstrip;
  bool add_space_prefix = false;
  bool remove_extra_whitespaces = false;
  bool strips_space = false;
  std::vector<std::string> decoded;  // normal tokens' bytes
  std::unordered_map<std::string_view, TokenId> by_text;
  std::array<TokenId, 256> byte_token{};
  std::array<std::string, 256> byte_text;  // each byte's alphabet character, UTF-8
  std::unordered_map<std::uint64_t, MergeResult, MixHash> merges;
  // Roots 0/1 extract added tokens before/after normalization; root 2
  // protects native SentencePiece model symbols during normalization.
  // Edges are keyed by (node << 8 | byte); terminal[node] is the token ID.
  // user_defined_below[node]: a user-defined token ends strictly below it.
  std::unordered_map<std::uint64_t, std::uint32_t, MixHash> edges;
  std::vector<TokenId> terminal;
  std::vector<bool> user_defined_below;
  std::array<std::array<bool, 256>, 3> special_first{};
  bool unused_model_tokens = false;
  PreTokenizer pre_tokenizer = PreTokenizer::kByteBpe;
  SplitPlan split_plan;
  bool has_post_processor = false;
  PostProcessorPlan post_processor;
  std::uint64_t post_processor_tokens = 0;
  bool decode_added_whitespace = false;
  bool raw_bpe_split_newlines = true;
  Normalization normalization = Normalization::kNone;
  bool ignore_merges = false;
  std::optional<TokenId> bos;
  std::optional<TokenId> eos;
  bool add_bos = false;
  bool add_eos = false;
  std::size_t longest = 1;  // the longest token's bytes (decoded, or a special's text)

  bool ModelToken(TokenId id) const {
    const auto i = static_cast<std::size_t>(id);
    return model_tokens.empty() ? kinds[i] == TokenKind::kNormal : model_tokens[i];
  }

  // The longest special token matching at text[p], allowed by `special`,
  // or -1.
  TokenId MatchSpecial(std::string_view text, std::size_t p, SpecialTokens special,
                       std::size_t* length, unsigned phase = 0) const {
    if (!special_first[phase][static_cast<unsigned char>(text[p])]) {
      return -1;
    }
    std::uint32_t node = phase;
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

  // True only for a trie prefix that reaches the available end and may
  // continue into a token `special` allows. Unlike a matched atom, such a
  // prefix is not a cut vertex yet.
  bool OpenSpecial(std::string_view text, std::size_t at, unsigned phase,
                   SpecialTokens special) const {
    std::uint32_t node = phase;
    for (std::size_t p = at; p < text.size(); ++p) {
      if (p - at >= kMaxSpecialBytes) return false;
      const auto found = edges.find((static_cast<std::uint64_t>(node) << 8U) |
                                    static_cast<unsigned char>(text[p]));
      if (found == edges.end()) return false;
      node = found->second;
    }
    if (text.size() - at >= kMaxSpecialBytes) return false;
    if (special == SpecialTokens::kUserDefinedOnly) return user_defined_below[node];
    // Every edge leads to some token, and kParse allows them all.
    for (unsigned byte = 0; byte < 256; ++byte)
      if (edges.contains((static_cast<std::uint64_t>(node) << 8U) | byte)) return true;
    return false;
  }

  // Marked spans in order, inside text, each exactly a special token's text
  // (EncodeMarked's rule).
  std::expected<void, Error> CheckSpans(std::string_view text,
                                        std::span<const SpecialSpan> spans) const {
    std::size_t at = 0;
    for (std::size_t i = 0; i < spans.size(); ++i) {
      if (i != 0 && (i & 0xFFFFU) == 0 && !base::Pulse())
        return std::unexpected(
            Error{Rule::kCancelled, "forced span planning was asked to stop", i});
      const SpecialSpan& s = spans[i];
      if (s.offset < at || s.length == 0 || s.offset > text.size() ||
          text.size() - s.offset < s.length)
        return std::unexpected(
            Error{Rule::kSpecialToken, "spans out of order or outside the text", i});
      const auto found = by_text.find(text.substr(s.offset, s.length));
      if (found == by_text.end() || !Special(found->second))
        return std::unexpected(Error{Rule::kSpecialToken, "a span is not a special token", i});
      at = s.offset + s.length;
    }
    return {};
  }

  bool Special(TokenId id) const {
    const auto kind = kinds[static_cast<std::size_t>(id)];
    return kind == TokenKind::kControl || kind == TokenKind::kUserDefined;
  }

  struct Blocks {
    std::vector<std::size_t> ends;
    std::size_t stable = 0;
  };

  // Uses the encoder's matchers, compiled SplitPlan, and raw newline rule.
  // Offsets are in the ORIGINAL bytes, only after exact reproduction has been
  // established by the caller. A normalized nonidentity fragment is refused.
  // Marked spans (checked by the caller) are atoms wherever they are, and
  // `special` then matches only user-defined tokens between them, as
  // EncodeMarked encodes: no match or strip crosses into a span.
  std::expected<Blocks, Error> TraceBlocks(std::string_view text,
                                           std::span<const SpecialSpan> spans,
                                           SpecialTokens special, bool final_boundary) const {
    Blocks out;
    out.ends.reserve(text.size());
    std::size_t uncertain = text.size();
    std::size_t work = 0;
    const auto pulse = [&] { return (++work & 0xffffU) != 0 || base::Pulse(); };
    auto budget = split_plan.Budget(Times(text.size() + 1, 4));
    const auto piece = [&](std::size_t end, bool stable) {
      if (out.ends.empty() || out.ends.back() != end) out.ends.push_back(end);
      if (stable) out.stable = end;
    };
    const auto fragment = [&](std::size_t begin, std::size_t end,
                              bool closed) -> std::expected<void, Error> {
      if (begin == end) return {};
      const auto input = text.substr(begin, end - begin);
      std::vector<char32_t> cps;
      if (auto decoded_text = unicode::DecodeUtf8(input, cps); !decoded_text) return decoded_text;
      if (normalization == Normalization::kNfc) {
        auto normalized = cps;
        unicode::ToNfc(normalized);
        if (normalized != cps)
          return std::unexpected(Error{Rule::kUnsupported, "forced bytes are changed by NFC"});
      }
      std::vector<std::size_t> offsets;
      offsets.reserve(cps.size() + 1);
      std::size_t at = begin;
      for (const auto cp : cps) {
        if (!pulse())
          return std::unexpected(Error{Rule::kCancelled, "forced span planning was asked to stop"});
        offsets.push_back(at);
        at += cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
      }
      offsets.push_back(end);
      std::vector<std::uint32_t> lengths;
      lengths.reserve(cps.size());
      std::size_t stable = 0;
      if (pre_tokenizer == PreTokenizer::kByteBpe) {
        if (auto split = split_plan.Split(cps, lengths, budget, &stable); !split) return split;
      } else if (pre_tokenizer == PreTokenizer::kRawBpe && raw_bpe_split_newlines) {
        for (std::size_t p = 0; p < cps.size();) {
          std::size_t next = p + 1;
          while (next < cps.size() && (cps[next] == U'\n') == (cps[p] == U'\n')) {
            if (!pulse())
              return std::unexpected(
                  Error{Rule::kCancelled, "forced span planning was asked to stop"});
            ++next;
          }
          lengths.push_back(static_cast<std::uint32_t>(next - p));
          if (next < cps.size()) stable = next;
          p = next;
        }
      } else {
        lengths.push_back(static_cast<std::uint32_t>(cps.size()));
      }
      if (closed) stable = cps.size();
      std::size_t normalization_cut = end;
      if (!closed && normalization == Normalization::kNfc) {
        // A future combining mark or Hangul jamo may change the final starter
        // and its marks. Earlier starters are independent. An ASCII lookahead
        // supplied by the caller is itself the held final starter.
        std::size_t starter = cps.size();
        while (starter > 0) {
          if (!pulse())
            return std::unexpected(
                Error{Rule::kCancelled, "forced span planning was asked to stop"});
          --starter;
          if (unicode::CombiningClass(cps[starter]) == 0) break;
        }
        stable = std::min(stable, starter);
        normalization_cut = offsets[starter];
      }
      std::size_t first = 0;
      for (const auto length : lengths) {
        const auto last = first + length;
        if (pre_tokenizer == PreTokenizer::kByteBpe) {
          piece(offsets[last], last <= stable);
          first = last;
          continue;
        }
        // Raw modes freeze normalized added/model symbols inside the merge
        // region. Map their actual normalized spellings back to source bytes.
        std::string mapped;
        std::vector<std::size_t> source;
        mapped.reserve(3 * (offsets[last] - offsets[first]));
        source.reserve(3 * (offsets[last] - offsets[first]) + 1);
        for (std::size_t c = first; c < last; ++c) {
          if (!pulse())
            return std::unexpected(
                Error{Rule::kCancelled, "forced span planning was asked to stop"});
          const auto start = mapped.size();
          unicode::AppendUtf8(cps[c] == U' ' ? U'▁' : cps[c], mapped);
          source.insert(source.end(), mapped.size() - start, offsets[c]);
        }
        source.push_back(offsets[last]);
        // Match EncodeWindow's newline whole-token shortcut before looking
        // for normalized atoms: that shortcut bypasses the symbol matcher.
        const auto whole = by_text.find(mapped);
        if (pre_tokenizer == PreTokenizer::kRawBpe && raw_bpe_split_newlines &&
            cps[first] == U'\n' && whole != by_text.end() &&
            kinds[static_cast<std::size_t>(whole->second)] == TokenKind::kNormal) {
          piece(offsets[last], last <= stable);
          first = last;
          continue;
        }

        const unsigned phase =
            pre_tokenizer == PreTokenizer::kSentencePiece && model_symbols.empty() ? 0U : 1U;
        const auto policy = pre_tokenizer == PreTokenizer::kSentencePiece && model_symbols.empty()
                                ? SpecialTokens::kUserDefinedOnly
                                : special;
        std::size_t normal = 0;
        for (std::size_t p = 0; p < mapped.size();) {
          if (!pulse())
            return std::unexpected(
                Error{Rule::kCancelled, "forced span planning was asked to stop"});
          if (!closed && last == cps.size() && OpenSpecial(mapped, p, phase, policy))
            uncertain = std::min(uncertain, source[p]);
          std::size_t size = 0;
          const auto id = MatchSpecial(mapped, p, policy, &size, phase);
          if (id >= 0) {
            if (p > normal) piece(source[p], true);
            p += size;
            piece(source[p], true);
            normal = p;
          } else {
            const auto byte = static_cast<unsigned char>(mapped[p]);
            p += byte < 0x80 ? 1 : byte < 0xe0 ? 2 : byte < 0xf0 ? 3 : 4;
          }
        }
        if (normal < mapped.size()) piece(offsets[last], last <= stable);
        first = last;
      }
      out.stable = std::min(out.stable, normalization_cut);
      return {};
    };
    std::size_t start = 0;
    std::size_t next = 0;  // the next marked span
    // Where a longer top-level atom may still begin. Only the text after the
    // last marked span can hold one.
    std::vector<std::size_t> open_atoms;
    for (std::size_t p = 0; p < text.size();) {
      if ((p & 0xffffU) == 0 && !base::Pulse())
        return std::unexpected(Error{Rule::kCancelled, "forced span planning was asked to stop"});
      const auto limit = next < spans.size() ? spans[next].offset : text.size();
      std::size_t length = 0;
      TokenId id = -1;
      if (p == limit) {
        length = spans[next].length;
        id = by_text.find(text.substr(p, length))->second;
        ++next;
      } else {
        const auto segment = text.substr(0, limit);
        if (!final_boundary && limit == text.size() && OpenSpecial(segment, p, 0, special))
          open_atoms.push_back(p);
        id = MatchSpecial(segment, p, special, &length);
        if (id < 0) {
          ++p;
          continue;
        }
      }
      if (auto traced = fragment(start, p, true); !traced) return std::unexpected(traced.error());
      p += length;
      piece(p, true);
      if (!rstrip.empty() && rstrip[static_cast<std::size_t>(id)]) {
        // Any stripped bytes would violate the exact reproduction check.
        const auto end = next < spans.size() ? spans[next].offset : text.size();
        while (p < end && AsciiSpace(text[p])) ++p;
      }
      start = p;
    }
    if (auto traced = fragment(start, text.size(), final_boundary); !traced)
      return std::unexpected(traced.error());
    out.stable = std::min(out.stable, uncertain);
    // If an atom does begin at q, the fragment before it closes at q. That is
    // not an append, so the fragment's own split may differ there (spaces
    // before "<" split differently at the end of text). Commit only blocks
    // whose partition the closed trace of text[0, q) reproduces exactly.
    for (const auto q : open_atoms) {
      std::size_t proven = 0;
      if (q != 0) {
        auto retraced = TraceBlocks(text.substr(0, q), spans, special, true);
        if (!retraced) return std::unexpected(retraced.error());
        for (std::size_t i = 0; i < out.ends.size() && i < retraced->ends.size() &&
                                out.ends[i] == retraced->ends[i] && out.ends[i] <= out.stable;
             ++i)
          proven = out.ends[i];
      }
      out.stable = std::min(out.stable, proven);
      if (out.stable == 0) break;
    }
    // A possible longer atom may begin within an earlier pretoken block.
    // Commit only whole blocks ending before that earliest uncertain byte.
    auto cut = std::upper_bound(out.ends.begin(), out.ends.end(), out.stable);
    out.stable = cut == out.ends.begin() ? 0 : *std::prev(cut);
    return out;
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
      // never doubles past them (kEncodeBytesPerInputByte).
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
      std::size_t work = 0;
      while (!queue.empty()) {
        if ((++work & 0xFFFFU) == 0 && !base::Pulse()) {
          return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", work});
        }
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
    std::size_t emitted = 0;
    for (const TokenId s : symbols) {
      if ((++emitted & 0xFFFFU) == 0 && !base::Pulse()) {
        return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", emitted});
      }
      if (s >= 0) {
        if (auto e = Emit(s, options, out, base); !e) {
          return e;
        }
      }
    }
    return {};
  }

  // Classic SentencePiece keeps missing code points as raw symbols until
  // after merging. Expanding them into fallback bytes first changes which
  // scored pairs exist (and can lose a known multi-code-point token).
  std::expected<void, Error> EncodeSentencePiece(std::string_view text,
                                                 const EncodeOptions& options,
                                                 std::vector<TokenId>& out,
                                                 std::size_t base) const {
    if (text.empty()) {
      return {};
    }
    std::string bytes;
    bytes.reserve(3 * (text.size() + 1));
    if (!model_symbols.empty() || remove_extra_whitespaces) {
      // ModelProto normalizes a complete region before freezing its model
      // symbols. External added/control tokens still delimit regions in
      // EncodeText; a native user symbol does not restart dummy prefixes or
      // turn adjacent internal whitespace into leading/trailing whitespace.
      bool previous_space = remove_extra_whitespaces;
      bool started = false;
      std::size_t work = 0;
      for (std::size_t at = 0; at < text.size();) {
        if ((++work & 0xFFFFU) == 0 && !base::Pulse())
          return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", at});
        std::size_t length = 0;
        if (MatchSpecial(text, at, SpecialTokens::kUserDefinedOnly, &length, 2) < 0) {
          const auto first = static_cast<unsigned char>(text[at]);
          length = first < 0x80 ? 1 : first < 0xE0 ? 2 : first < 0xF0 ? 3 : 4;
        }
        auto piece = text.substr(at, length);
        at += length;
        if (!started && remove_extra_whitespaces && piece == " ") continue;
        if (!started) {
          if (add_space_prefix && !options.continuation) bytes = "▁";
          started = true;
        }
        while (previous_space && piece.starts_with(' ')) piece.remove_prefix(1);
        for (const char c : piece) {
          if (c == ' ')
            bytes += "▁";
          else
            bytes.push_back(c);
        }
        if (!piece.empty()) previous_space = remove_extra_whitespaces && piece.back() == ' ';
      }
      if (remove_extra_whitespaces) {
        while (bytes.ends_with("▁")) {
          if ((++work & 0xFFFFU) == 0 && !base::Pulse())
            return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop"});
          bytes.resize(bytes.size() - std::string_view("▁").size());
        }
      }
    } else {
      if (add_space_prefix && !options.continuation) bytes = "▁";
      for (std::size_t i = 0; i < text.size(); ++i) {
        if ((i & 0xFFFFU) == 0 && !base::Pulse())
          return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", i});
        if (text[i] == ' ')
          bytes += "▁";
        else
          bytes.push_back(text[i]);
      }
    }
    if (bytes.empty()) return {};
    struct Symbol {
      std::size_t offset;
      std::uint32_t length;
      std::uint32_t prev;
      std::uint32_t next;
      TokenId frozen;
    };
    std::vector<Symbol> symbols;
    symbols.reserve(text.size() + 1);
    for (std::size_t at = 0; at < bytes.size();) {
      if ((symbols.size() & 0xFFFFU) == 0 && !base::Pulse()) {
        return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", at});
      }
      const auto b = static_cast<unsigned char>(bytes[at]);
      std::size_t special_length = 0;
      const auto frozen = MatchSpecial(
          bytes, at, model_symbols.empty() ? SpecialTokens::kUserDefinedOnly : options.special,
          &special_length, model_symbols.empty() ? 0U : 1U);
      const std::uint32_t length = frozen >= 0 ? static_cast<std::uint32_t>(special_length)
                                   : b < 0x80  ? 1U
                                   : b < 0xE0  ? 2U
                                   : b < 0xF0  ? 3U
                                               : 4U;
      const auto index = static_cast<std::uint32_t>(symbols.size());
      symbols.push_back({.offset = at,
                         .length = length,
                         .prev = index == 0 ? kNone : index - 1,
                         .next = at + length < bytes.size() ? index + 1 : kNone,
                         .frozen = frozen});
      at += length;
    }
    struct Candidate {
      float score;
      std::uint32_t left;
      std::uint32_t right;
      std::uint32_t length;
      TokenId token;
    };
    struct Lower {
      bool operator()(const Candidate& a, const Candidate& b) const {
        return a.score < b.score || (a.score == b.score && a.left > b.left);
      }
    };
    // At most n-1 seeds and two candidates per successful merge. Together
    // with symbols (24n), escaped bytes (3n), and this heap (60n), this fits
    // WorkingBytes's 96*(input bytes+1), including the optional dummy symbol.
    static_assert(sizeof(Symbol) <= 24 && sizeof(Candidate) <= 20);
    std::vector<Candidate> storage;
    storage.reserve(3 * symbols.size() - 3);
    std::priority_queue<Candidate, std::vector<Candidate>, Lower> queue(Lower{},
                                                                        std::move(storage));
    // Native UNUSED entries may be intermediate merges. Record the most
    // recently considered split per piece, then resegment only surviving
    // UNUSED pieces after all merges. A direct ID table bounds lookup and
    // avoids recursively walking a hostile merge chain on the C++ stack.
    std::vector<std::uint32_t> unused_splits(unused_model_tokens ? tokens.size() : 0, 0);
    const auto consider = [&](std::uint32_t left) {
      if (left == kNone || symbols[left].next == kNone || symbols[left].frozen >= 0 ||
          symbols[symbols[left].next].frozen >= 0) {
        return;
      }
      const auto right = symbols[left].next;
      const auto length = symbols[left].length + symbols[right].length;
      if (length > kMaxTokenBytes) {
        return;
      }
      const auto found = by_text.find(std::string_view(bytes).substr(symbols[left].offset, length));
      if (found != by_text.end() && ModelToken(found->second)) {
        const auto token = found->second;
        if (kinds[static_cast<std::size_t>(token)] == TokenKind::kUnused)
          unused_splits[static_cast<std::size_t>(token)] = symbols[left].length;
        queue.push({.score = scores[static_cast<std::size_t>(token)],
                    .left = left,
                    .right = right,
                    .length = length,
                    .token = token});
      }
    };
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      if ((i & 0xFFFFU) == 0 && !base::Pulse()) {
        return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", i});
      }
      consider(static_cast<std::uint32_t>(i));
    }
    std::size_t work = 0;
    while (!queue.empty()) {
      if ((++work & 0xFFFFU) == 0 && !base::Pulse()) {
        return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", work});
      }
      const auto candidate = queue.top();
      queue.pop();
      auto& left = symbols[candidate.left];
      auto& right = symbols[candidate.right];
      if (left.length == 0 || right.length == 0 || left.next != candidate.right ||
          left.length + right.length != candidate.length) {
        continue;
      }
      left.length += right.length;
      right.length = 0;
      left.next = right.next;
      if (right.next != kNone) {
        symbols[right.next].prev = candidate.left;
      }
      consider(left.prev);
      consider(candidate.left);
    }
    std::size_t emission_work = 0;
    std::vector<std::pair<std::size_t, std::uint32_t>> pending;
    static_assert(sizeof(decltype(pending)::value_type) <= 16);
    if (unused_model_tokens) pending.reserve(text.size() + 1);
    for (std::uint32_t i = 0; i != kNone; i = symbols[i].next) {
      if ((++emission_work & 0xFFFFU) == 0 && !base::Pulse()) {
        return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", i});
      }
      const auto& symbol = symbols[i];
      if (symbol.frozen >= 0) {
        if (auto emitted = Emit(symbol.frozen, options, out, base); !emitted) return emitted;
        continue;
      }
      const auto emit_piece = [&](std::size_t offset,
                                  std::uint32_t length) -> std::expected<void, Error> {
        const auto piece = std::string_view(bytes).substr(offset, length);
        const auto found = by_text.find(piece);
        if (found != by_text.end() && ModelToken(found->second)) {
          const auto id = static_cast<std::size_t>(found->second);
          if (kinds[id] == TokenKind::kUnused && unused_splits[id] != 0) {
            const auto split = unused_splits[id];
            pending.emplace_back(offset + split, length - split);
            pending.emplace_back(offset, split);
            return {};
          }
          return Emit(found->second, options, out, base);
        }
        for (const char b : piece)
          if (auto emitted = Emit(byte_token[static_cast<unsigned char>(b)], options, out, base);
              !emitted)
            return emitted;
        return {};
      };
      if (auto emitted = emit_piece(symbol.offset, symbol.length); !emitted) return emitted;
      while (!pending.empty()) {
        if ((++emission_work & 0xFFFFU) == 0 && !base::Pulse())
          return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop"});
        const auto [offset, length] = pending.back();
        pending.pop_back();
        if (auto emitted = emit_piece(offset, length); !emitted) return emitted;
      }
    }
    return {};
  }

  // Encodes a whole fragment; generic expressions may join across spaces.
  std::expected<void, Error> EncodeFragment(std::string_view text, const EncodeOptions& options,
                                            std::vector<TokenId>& out, std::size_t base,
                                            RegexBudget& budget) const {
    if (pre_tokenizer == PreTokenizer::kSentencePiece) {
      return EncodeSentencePiece(text, options, out, base);
    }
    if (!base::Pulse())
      return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop"});
    return EncodeWindow(text, options, out, base, budget);
  }

  std::expected<void, Error> EncodeWindow(std::string_view text, const EncodeOptions& options,
                                          std::vector<TokenId>& out, std::size_t base,
                                          RegexBudget& budget) const {
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
    if (pre_tokenizer == PreTokenizer::kRawBpe) {
      for (char32_t& cp : cps) {
        if (cp == U' ') {
          cp = U'▁';
        }
      }
    }
    std::vector<std::uint32_t> lengths;
    lengths.reserve(cps.size());  // a piece a code point at most
    if (pre_tokenizer == PreTokenizer::kRawBpe) {
      if (!raw_bpe_split_newlines) {
        lengths.push_back(static_cast<std::uint32_t>(cps.size()));
      } else
        for (std::size_t at = 0; at < cps.size();) {
          std::size_t end = at + 1;
          while (end < cps.size() && (cps[end] == U'\n') == (cps[at] == U'\n')) {
            if (auto paid = budget.Pay(); !paid) return paid;
            ++end;
          }
          lengths.push_back(static_cast<std::uint32_t>(end - at));
          at = end;
        }
    } else if (auto split = split_plan.Split(cps, lengths, budget); !split) {
      return split;
    }
    std::size_t start = 0;
    std::string bytes;
    std::string alphabet;
    std::vector<TokenId> symbols;
    std::size_t pieces = 0;
    for (const std::uint32_t length : lengths) {
      if ((++pieces & 0xFFFFU) == 0 && !base::Pulse()) {
        return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", pieces});
      }
      bytes.clear();
      for (std::size_t i = start; i < start + length; ++i) {
        unicode::AppendUtf8(cps[i], bytes);
      }
      start += length;
      if (pre_tokenizer == PreTokenizer::kRawBpe) {
        // The pinned reference treats a whole newline run that is a token
        // atomically, even when its merge path would choose other tokens.
        const auto whole = by_text.find(bytes);
        if (raw_bpe_split_newlines && bytes[0] == '\n' && whole != by_text.end() &&
            kinds[static_cast<std::size_t>(whole->second)] == TokenKind::kNormal) {
          if (auto e = Emit(whole->second, options, out, base); !e) {
            return e;
          }
          continue;
        }
        symbols.clear();
        for (std::size_t at = 0; at < bytes.size();) {
          if (at != 0 && (symbols.size() & 0xFFFFU) == 0 && !base::Pulse()) {
            return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", at});
          }
          std::size_t special_length = 0;
          const auto frozen = MatchSpecial(bytes, at, options.special, &special_length, true);
          if (frozen >= 0) {
            if (auto merged = Merge(symbols, options, out, base); !merged) return merged;
            symbols.clear();
            if (auto emitted = Emit(frozen, options, out, base); !emitted) return emitted;
            at += special_length;
            continue;
          }
          const unsigned char b = static_cast<unsigned char>(bytes[at]);
          const std::size_t width = b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
          const auto it = by_text.find(std::string_view(bytes).substr(at, width));
          if (it != by_text.end() && ModelToken(it->second)) {
            symbols.push_back(it->second);
          } else {
            for (std::size_t i = at; i < at + width; ++i) {
              symbols.push_back(byte_token[static_cast<unsigned char>(bytes[i])]);
            }
          }
          at += width;
        }
        if (auto m = Merge(symbols, options, out, base); !m) {
          return m;
        }
        continue;
      }
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

  std::expected<void, Error> Insert(std::span<const PostProcessorSlice> pieces,
                                    const EncodeOptions& options, std::vector<TokenId>& out,
                                    std::size_t base) const {
    for (const auto piece : pieces) {
      if (!base::Pulse())
        return std::unexpected(Error{Rule::kCancelled, "postprocessing was asked to stop"});
      for (const auto id : std::span(post_processor.ids).subspan(piece.offset, piece.length)) {
        if (!base::Pulse())
          return std::unexpected(Error{Rule::kCancelled, "postprocessing was asked to stop"});
        if (auto emitted = Emit(id, options, out, base); !emitted) return emitted;
      }
    }
    return {};
  }

  std::uint64_t OutputTokensBound(std::size_t n, bool source) const {
    const auto limit = std::numeric_limits<std::uint64_t>::max();
    const auto bytes = n == limit ? limit : Times(n + 1, 16);
    const auto wrappers = !source              ? 0
                          : has_post_processor ? post_processor_tokens
                                               : static_cast<std::uint64_t>(add_bos) +
                                                     static_cast<std::uint64_t>(add_eos);
    return bytes > limit - wrappers ? limit : bytes + wrappers;
  }

  std::expected<void, Error> ReserveOutput(std::size_t n, const EncodeOptions& options,
                                           std::vector<TokenId>& out, std::size_t extra) const {
    const auto append =
        std::min<std::uint64_t>(OutputTokensBound(n, options.add_bos_eos), options.max_tokens);
    std::uint64_t capacity = 0;
    if (__builtin_add_overflow(append, static_cast<std::uint64_t>(out.size()), &capacity) ||
        __builtin_add_overflow(capacity, static_cast<std::uint64_t>(extra), &capacity) ||
        capacity > out.max_size())
      return std::unexpected(Error{Rule::kBounds, "output reservation overflows"});
    if (!base::Pulse())
      return std::unexpected(Error{Rule::kCancelled, "output reservation was asked to stop"});
    if (capacity > out.capacity()) {
      out.reserve(static_cast<std::size_t>(capacity));
      // The pinned library's reserve allocates exactly n, never geometric growth.
      base::Check(out.capacity() == capacity, "token output exceeded its reserved capacity");
    }
    return {};
  }

  std::expected<void, Error> Preflight(std::string_view text, const EncodeOptions& options) const {
    if (text.size() > options.max_bytes ||
        (text.size() > kNone ||
         (pre_tokenizer == PreTokenizer::kSentencePiece && text.size() == kNone))) {
      return std::unexpected(
          Error{Rule::kInputTooLarge, "text longer than max_bytes", options.max_bytes});
    }
    // Every token covers at most `longest` bytes of normalized text, and NFC
    // shortens text at most threefold: a longer text has more tokens than
    // max_tokens, which is known before any work.
    if (!strips_space && !remove_extra_whitespaces &&
        text.size() > Times(Times(options.max_tokens, longest),
                            normalization == Normalization::kNfc ? 4 : 1)) {
      return std::unexpected(
          Error{Rule::kOutputTooLarge, "more tokens than max_tokens", options.max_tokens});
    }
    if (!base::Pulse()) {
      return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop"});
    }
    if (auto valid = unicode::ValidateUtf8(text); !valid) {
      return valid;
    }
    return {};
  }

  std::expected<void, Error> Begin(std::string_view text, const EncodeOptions& options,
                                   std::vector<TokenId>& out, std::size_t base) const {
    if (auto valid = Preflight(text, options); !valid) return valid;
    if (auto reserved = ReserveOutput(text.size(), options, out, 0); !reserved) return reserved;
    if (options.add_bos_eos) {
      if (has_post_processor) {
        if (auto inserted = Insert(post_processor.prefix, options, out, base); !inserted)
          return inserted;
      } else if (add_bos && bos)
        return Emit(*bos, options, out, base);
    }
    return {};
  }

  std::expected<void, Error> End(const EncodeOptions& options, std::vector<TokenId>& out,
                                 std::size_t base) const {
    if (options.add_bos_eos) {
      if (has_post_processor) {
        if (auto inserted = Insert(post_processor.suffix, options, out, base); !inserted)
          return inserted;
      } else if (add_eos && eos)
        return Emit(*eos, options, out, base);
    }
    return {};
  }

  // One matching policy applies before and after normalization.
  std::expected<void, Error> EncodeText(std::string_view text, const EncodeOptions& options,
                                        std::vector<TokenId>& out, std::size_t base,
                                        RegexBudget& budget) const {
    std::size_t fragment = 0;
    std::size_t checkpoint = 65536;
    for (std::size_t p = 0; p < text.size();) {
      if (p >= checkpoint) {
        if (!base::Pulse()) {
          return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", p});
        }
        checkpoint = p + 65536;
      }
      std::size_t length = 0;
      const TokenId t = MatchSpecial(text, p, options.special, &length);
      if (t < 0) {
        ++p;
        continue;
      }
      if (auto e = EncodeFragment(text.substr(fragment, p - fragment), options, out, base, budget);
          !e) {
        return e;
      }
      if (auto e = Emit(t, options, out, base); !e) {
        return e;
      }
      p += length;
      if (!rstrip.empty() && rstrip[static_cast<std::size_t>(t)]) {
        while (p < text.size() && AsciiSpace(text[p])) {
          if ((p & 0xFFFFU) == 0 && !base::Pulse()) {
            return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", p});
          }
          ++p;
        }
      }
      fragment = p;
    }
    return EncodeFragment(text.substr(fragment), options, out, base, budget);
  }
};

namespace {
std::expected<std::uint64_t, Error> PostProcessorTokens(const PostProcessorPlan& plan) {
  std::uint64_t count = 0;
  for (const auto& list : {std::span(plan.prefix), std::span(plan.suffix)}) {
    for (const auto piece : list) {
      if (!base::Pulse())
        return std::unexpected(
            Error{Rule::kCancelled, "postprocessor validation was asked to stop"});
      if (piece.offset > plan.ids.size() || piece.length > plan.ids.size() - piece.offset)
        return std::unexpected(Error{Rule::kVocabulary, "postprocessor slice outside definitions"});
      if (__builtin_add_overflow(count, static_cast<std::uint64_t>(piece.length), &count))
        return std::unexpected(
            Error{Rule::kBounds, "expanded postprocessor token count overflows"});
    }
  }
  return count;
}
}  // namespace

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

std::expected<std::uint64_t, Error> Tokenizer::ConstructionBytes(const TokenizerSpec& spec) {
  const auto fail = [] {
    return std::unexpected(Error{Rule::kBounds, "tokenizer construction bound overflows"});
  };
  if (spec.tokens.empty() || spec.tokens.size() > kMaxVocabulary ||
      spec.kinds.size() != spec.tokens.size() || spec.merges.size() > 2 * kMaxVocabulary ||
      (!spec.model_tokens.empty() && spec.model_tokens.size() != spec.tokens.size()) ||
      (!spec.model_symbols.empty() && spec.model_symbols.size() != spec.tokens.size()) ||
      (!spec.added_tokens.empty() && spec.added_tokens.size() != spec.tokens.size()) ||
      (!spec.normalized_tokens.empty() && spec.normalized_tokens.size() != spec.tokens.size()))
    return fail();
  // The byte alphabet's 256 strings/map nodes, fixed arrays and objects.
  // Hash entries below include node allocation, bucket storage and rehash
  // overlap under the pinned standard library; vector growth counts old+new.
  std::uint64_t bytes = sizeof(Impl) + sizeof(TokenizerSpec) + 256 * 192;
  const auto add = [&](std::uint64_t count, std::uint64_t unit) {
    std::uint64_t part = 0;
    return !__builtin_mul_overflow(count, unit, &part) &&
           !__builtin_add_overflow(bytes, part, &bytes);
  };
  if (!add(spec.tokens.capacity(), sizeof(std::string)) ||
      !add(spec.kinds.capacity(), sizeof(TokenKind)) || !add(spec.model_tokens.capacity(), 1) ||
      !add(spec.model_symbols.capacity(), 1) || !add(spec.added_tokens.capacity(), 1) ||
      !add(spec.normalized_tokens.capacity(), 1) || !add(spec.scores.capacity(), sizeof(float)) ||
      !add(spec.rstrip.capacity(), 1) ||
      !add(spec.merges.capacity(), sizeof(std::pair<std::string, std::string>)) ||
      !add(spec.tokens.size(), sizeof(std::string) + 192) || !add(spec.merges.size(), 192))
    return fail();
  if (auto count = PostProcessorTokens(spec.post_processor); !count)
    return std::unexpected(count.error());
  if (!add(spec.split_expressions.capacity(), sizeof(std::string)) ||
      !add(spec.post_processor.ids.capacity(), sizeof(TokenId)) ||
      !add(spec.post_processor.prefix.capacity(), sizeof(PostProcessorSlice)) ||
      !add(spec.post_processor.suffix.capacity(), sizeof(PostProcessorSlice)))
    return fail();
  for (const auto& expression : spec.split_expressions) {
    if (!add(expression.capacity(), 1) || !add(1, Regex::ConstructionBytes(expression.size())))
      return fail();
  }
  std::uint64_t longest = 0, longest_merge = 0;
  for (std::size_t i = 0; i < spec.tokens.size(); ++i) {
    const auto& text = spec.tokens[i];
    if (!add(text.capacity(), 1) || !add(1, 1) || !add(text.size(), 3)) return fail();
    longest = std::max(longest, static_cast<std::uint64_t>(text.size()));
    if (spec.kinds[i] != TokenKind::kNormal && spec.kinds[i] != TokenKind::kUnused &&
        spec.kinds[i] != TokenKind::kByte) {
      const bool normalized = !spec.normalized_tokens.empty() && spec.normalized_tokens[i];
      const bool added = spec.added_tokens.empty() || spec.added_tokens[i];
      std::uint64_t copies = added ? (normalized ? 3 : 1) : 0;
      if (!spec.model_symbols.empty() && spec.model_symbols[i] &&
          spec.kinds[i] == TokenKind::kUserDefined)
        copies += 2;  // protected raw spelling and exact model spelling after normalization
      if (!add(text.size(), copies * 192)) return fail();
    }
  }
  for (const auto& [left, right] : spec.merges) {
    std::uint64_t size = 0;
    if (!add(left.capacity(), 1) || !add(right.capacity(), 1) || !add(2, 1) ||
        __builtin_add_overflow(static_cast<std::uint64_t>(left.size()),
                               static_cast<std::uint64_t>(right.size()), &size))
      return fail();
    longest_merge = std::max(longest_merge, size);
  }
  // UTF-8 code points grow by doubling; the merge's joined string may grow
  // while retaining its previous buffer. Three special-trie roots are separate.
  if (!add(longest, 12) || !add(longest_merge, 3) || !add(3, 192)) return fail();
  return bytes;
}

std::expected<Tokenizer, Error> Tokenizer::Create(TokenizerSpec spec) {
  auto fail = [](Rule rule, std::string_view reason, std::uint64_t item = kNoItem) {
    return std::unexpected(Error{rule, reason, item});
  };
  auto wrapper_count = PostProcessorTokens(spec.post_processor);
  if (!wrapper_count) return std::unexpected(wrapper_count.error());
  const std::size_t n = spec.tokens.size();
  if (n == 0 || n > kMaxVocabulary) {
    return fail(Rule::kBounds, "vocabulary size", n);
  }
  if (spec.kinds.size() != n || (!spec.model_tokens.empty() && spec.model_tokens.size() != n) ||
      (!spec.model_symbols.empty() && spec.model_symbols.size() != n) ||
      (!spec.added_tokens.empty() && spec.added_tokens.size() != n) ||
      (!spec.normalized_tokens.empty() && spec.normalized_tokens.size() != n)) {
    return fail(Rule::kVocabulary, "token kinds and tokens differ in number", spec.kinds.size());
  }
  if (spec.merges.size() > 2 * kMaxVocabulary) {
    return fail(Rule::kBounds, "merge count", spec.merges.size());
  }
  if (spec.pre_tokenizer == PreTokenizer::kRawBpe &&
      (spec.normalization != Normalization::kNone || spec.ignore_merges)) {
    return fail(Rule::kUnsupported, "raw UTF-8 BPE normalization or ignore-merges");
  }
  if (spec.decode_added_whitespace && !Raw(spec.pre_tokenizer))
    return fail(Rule::kVocabulary, "added-token whitespace decoder requires a raw model");
  const bool spm = spec.pre_tokenizer == PreTokenizer::kSentencePiece;
  if ((spm && (spec.scores.size() != n || !spec.merges.empty() ||
               spec.normalization != Normalization::kNone || spec.ignore_merges)) ||
      (!spm && (!spec.scores.empty() || spec.add_space_prefix || spec.remove_extra_whitespaces ||
                !spec.rstrip.empty() || !spec.model_symbols.empty()))) {
    return fail(Rule::kVocabulary, "incompatible SentencePiece configuration");
  }
  if ((!spec.rstrip.empty() && spec.rstrip.size() != n) ||
      std::ranges::any_of(spec.scores, [](float score) { return !std::isfinite(score); })) {
    return fail(Rule::kVocabulary, "invalid SentencePiece scores or special attributes");
  }
  auto impl = std::make_unique<Impl>();
  impl->tokens = std::move(spec.tokens);
  impl->kinds = std::move(spec.kinds);
  impl->model_tokens = std::move(spec.model_tokens);
  impl->model_symbols = std::move(spec.model_symbols);
  impl->added_tokens = std::move(spec.added_tokens);
  impl->normalized_tokens = std::move(spec.normalized_tokens);
  impl->scores = std::move(spec.scores);
  impl->rstrip = std::move(spec.rstrip);
  impl->add_space_prefix = spec.add_space_prefix;
  impl->remove_extra_whitespaces = spec.remove_extra_whitespaces;
  impl->strips_space = std::ranges::any_of(impl->rstrip, [](bool value) { return value; });
  auto plan = SplitPlan::Compile(spec.pre_tokenizer == PreTokenizer::kByteBpe
                                     ? std::span<const std::string>(spec.split_expressions)
                                     : std::span<const std::string>{},
                                 spec.split_dialect);
  if (!plan) return std::unexpected(plan.error());
  impl->split_plan = std::move(*plan);
  impl->has_post_processor = spec.has_post_processor;
  impl->post_processor = std::move(spec.post_processor);
  impl->post_processor_tokens = *wrapper_count;
  for (const auto id : impl->post_processor.ids) {
    if (!base::Pulse()) return fail(Rule::kCancelled, "postprocessor validation was asked to stop");
    if (id < 0 || std::cmp_greater_equal(id, n) ||
        impl->kinds[static_cast<std::size_t>(id)] == TokenKind::kUnused)
      return fail(Rule::kVocabulary, "postprocessor ID outside vocabulary");
  }
  impl->pre_tokenizer = spec.pre_tokenizer;
  impl->raw_bpe_split_newlines = spec.raw_bpe_split_newlines;
  impl->decode_added_whitespace = spec.decode_added_whitespace;
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
  impl->terminal.resize(3, -1);
  impl->user_defined_below.resize(3, false);
  std::vector<char32_t> cps;
  for (std::size_t id = 0; id < n; ++id) {
    const std::string& text = impl->tokens[id];
    const TokenKind kind = impl->kinds[id];
    if (!impl->model_tokens.empty() && impl->model_tokens[id] &&
        (kind == TokenKind::kControl || (kind == TokenKind::kUnused && !spm)))
      return fail(Rule::kVocabulary, "control or unused token is in the BPE model", id);
    if (!impl->model_symbols.empty() && impl->model_symbols[id] &&
        (kind != TokenKind::kUserDefined && kind != TokenKind::kControl))
      return fail(Rule::kVocabulary, "a model symbol is not a user-defined or control token", id);
    if (!impl->rstrip.empty() && impl->rstrip[id] && kind != TokenKind::kControl &&
        kind != TokenKind::kUserDefined) {
      return fail(Rule::kVocabulary, "whitespace stripping on a nonspecial token", id);
    }
    if (kind == TokenKind::kUnused && !impl->ModelToken(static_cast<TokenId>(id))) {
      continue;
    }
    if (kind == TokenKind::kUnused) impl->unused_model_tokens = true;
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
    if (kind == TokenKind::kByte) {
      if (!Raw(spec.pre_tokenizer) || text.size() != 6 || !text.starts_with("<0x") ||
          text.back() != '>') {
        return fail(Rule::kVocabulary, "invalid byte fallback token", id);
      }
      auto hex = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
      };
      const int high = hex(text[3]);
      const int low = hex(text[4]);
      if (high < 0 || low < 0) {
        return fail(Rule::kVocabulary, "invalid byte fallback token", id);
      }
      impl->decoded[id].push_back(static_cast<char>((high << 4) | low));
      continue;
    }
    if (kind == TokenKind::kNormal || kind == TokenKind::kUnused) {
      std::string& bytes = impl->decoded[id];
      if (Raw(spec.pre_tokenizer)) {
        for (const char32_t cp : cps) {
          unicode::AppendUtf8(cp == U'▁' ? U' ' : cp, bytes);
        }
        impl->longest = std::max(impl->longest, text.size());
        continue;
      }
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
    if (Raw(spec.pre_tokenizer)) {
      for (const char32_t cp : cps) unicode::AppendUtf8(cp == U'▁' ? U' ' : cp, impl->decoded[id]);
    }
    // A special token: add it to the trie.
    if (text.size() > kMaxSpecialBytes) {
      return fail(Rule::kBounds, "special token longer than the matcher's bound", id);
    }
    const bool normalized = !impl->normalized_tokens.empty() && impl->normalized_tokens[id];
    if (normalized && !Raw(impl->pre_tokenizer))
      return fail(Rule::kUnsupported, "normalized added tokens require a raw pipeline", id);
    std::string normalized_text;
    std::string_view match = text;
    if (normalized) {
      for (const char c : text) {
        if (c == ' ')
          normalized_text += "▁";
        else
          normalized_text.push_back(c);
      }
      match = normalized_text;
    }
    if (match.size() > kMaxSpecialBytes)
      return fail(Rule::kBounds, "normalized special token exceeds the matcher's bound", id);
    impl->longest = std::max(impl->longest, match.size());
    const auto insert = [&](unsigned phase, std::string_view value) {
      std::uint32_t node = phase;
      for (const char c : value) {
        if (kind == TokenKind::kUserDefined) impl->user_defined_below[node] = true;
        const std::uint64_t key =
            (static_cast<std::uint64_t>(node) << 8U) | static_cast<unsigned char>(c);
        const auto [it, added] =
            impl->edges.emplace(key, static_cast<std::uint32_t>(impl->terminal.size()));
        if (added) {
          impl->terminal.push_back(-1);
          impl->user_defined_below.push_back(false);
        }
        node = it->second;
      }
      if (impl->terminal[node] >= 0 && impl->terminal[node] != static_cast<TokenId>(id))
        return false;
      impl->terminal[node] = static_cast<TokenId>(id);
      impl->special_first[phase][static_cast<unsigned char>(value[0])] = true;
      return true;
    };
    const bool added = impl->added_tokens.empty() || impl->added_tokens[id];
    if (added && !insert(normalized ? 1U : 0U, match))
      return fail(Rule::kVocabulary, "added tokens have the same normalized text", id);
    if (!impl->model_symbols.empty() && impl->model_symbols[id] &&
        kind == TokenKind::kUserDefined) {
      if (!insert(1, text) || !insert(2, text))
        return fail(Rule::kVocabulary, "model symbols conflict with added-token matching", id);
    }
  }

  for (unsigned b = 0; b < 256; ++b) {
    if (Raw(spec.pre_tokenizer)) {
      constexpr std::string_view hex = "0123456789ABCDEF";
      impl->byte_text[b] = "<0x00>";
      impl->byte_text[b][3] = hex[b >> 4U];
      impl->byte_text[b][4] = hex[b & 15U];
    }
    const auto it = impl->by_text.find(impl->byte_text[b]);
    // Raw BPE seeds complete code points before using byte fallback. An
    // unchanged ASCII character can therefore be covered by its normal
    // model token (Gemma 2 stores tab this way), without changing its kind
    // or merge eligibility. A non-ASCII character cannot stand in for an
    // individual UTF-8 byte, and spaces normalize to the marker instead.
    if (it == impl->by_text.end() && spec.pre_tokenizer == PreTokenizer::kRawBpe && b < 0x80 &&
        b != ' ') {
      const char literal = static_cast<char>(b);
      const auto direct = impl->by_text.find(std::string_view(&literal, 1));
      if (direct != impl->by_text.end() &&
          impl->kinds[static_cast<std::size_t>(direct->second)] == TokenKind::kNormal &&
          impl->ModelToken(direct->second)) {
        impl->byte_token[b] = direct->second;
        continue;
      }
    }
    if (it == impl->by_text.end() ||
        impl->kinds[static_cast<std::size_t>(it->second)] !=
            (Raw(spec.pre_tokenizer) ? TokenKind::kByte : TokenKind::kNormal)) {
      return fail(Rule::kVocabulary, "a byte has no fallback token", b);
    }
    impl->byte_token[b] = it->second;
  }

  if (Raw(spec.pre_tokenizer)) {
    const auto space = impl->by_text.find("▁");
    if (space == impl->by_text.end() ||
        impl->kinds[static_cast<std::size_t>(space->second)] != TokenKind::kNormal) {
      return fail(Rule::kVocabulary, "raw UTF-8 BPE requires a whitespace marker token");
    }
  }

  impl->merges.reserve(spec.merges.size());
  auto normal = [&impl](std::string_view text) -> std::optional<TokenId> {
    const auto it = impl->by_text.find(text);
    if (it == impl->by_text.end() || !impl->ModelToken(it->second)) {
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

std::uint64_t Tokenizer::WorkingBytes(std::string_view text) const {
  return WorkingBytes(text.size());
}
std::uint64_t Tokenizer::WorkingBytes(std::size_t n) const {
  const auto limit = std::numeric_limits<std::uint64_t>::max();
  if (n == limit) return limit;
  const auto bytes = Times(n + 1, impl_->normalization == Normalization::kNfc ? 4 : 1);
  auto bound = Times(bytes, impl_->unused_model_tokens ? 128 : kEncodeBytesPerInputByte);
  const auto scratch =
      impl_->pre_tokenizer == PreTokenizer::kByteBpe ? impl_->split_plan.WorkingBytes(bytes) : 0;
  const auto table =
      impl_->unused_model_tokens ? Times(impl_->tokens.size(), sizeof(std::uint32_t)) : 0;
  if (bound > limit - scratch || bound + scratch > limit - table) return limit;
  return bound + scratch + table;
}
std::uint64_t Tokenizer::OutputTokensBound(std::size_t n, bool source) const {
  return impl_->OutputTokensBound(n, source);
}
std::expected<void, Error> Tokenizer::ReserveOutput(std::string_view text,
                                                    const EncodeOptions& options,
                                                    std::vector<TokenId>& out,
                                                    std::size_t extra) const {
  if (auto valid = impl_->Preflight(text, options); !valid) return valid;
  return impl_->ReserveOutput(text.size(), options, out, extra);
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
  auto budget = impl_->split_plan.Budget(Times(text.size() + 1, 4));
  if (auto b = impl_->Begin(text, options, out, base); !b) {
    return b;
  }
  if (auto e = impl_->EncodeText(text, options, out, base, budget); !e) {
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
  auto budget = impl_->split_plan.Budget(Times(text.size() + 1, 4));
  if (auto b = impl_->Begin(text, options, out, base); !b) {
    return b;
  }
  if (span_tokens != nullptr) {
    if (spans.size() > span_tokens->max_size() - span_tokens->size())
      return std::unexpected(Error{Rule::kBounds, "span index reservation overflows"});
    span_tokens->reserve(span_tokens->size() + spans.size());
  }
  auto text_options = options;
  text_options.special = SpecialTokens::kUserDefinedOnly;
  std::size_t at = 0;
  for (std::size_t i = 0; i < spans.size(); ++i) {
    if (i != 0 && (i & 0xFFFFU) == 0 && !base::Pulse()) {
      return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", i});
    }
    const SpecialSpan& s = spans[i];
    if (s.offset < at || s.length == 0 || s.offset > text.size() ||
        text.size() - s.offset < s.length) {
      return std::unexpected(
          Error{Rule::kSpecialToken, "spans out of order or outside the text", i});
    }
    const auto token = Find(text.substr(s.offset, s.length));
    if (!token || Kind(*token) == TokenKind::kNormal || Kind(*token) == TokenKind::kByte ||
        Kind(*token) == TokenKind::kUnused) {
      return std::unexpected(Error{Rule::kSpecialToken, "a span is not a special token", i});
    }
    if (auto e = impl_->EncodeText(text.substr(at, s.offset - at), text_options, out, base, budget);
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
    if (!impl_->rstrip.empty() && impl_->rstrip[static_cast<std::size_t>(*token)]) {
      // Never consume a renderer-marked token while stripping its preceding whitespace.
      const auto end =
          i + 1 < spans.size() ? std::min(text.size(), spans[i + 1].offset) : text.size();
      while (at < end && AsciiSpace(text[at])) {
        if ((at & 0xFFFFU) == 0 && !base::Pulse()) {
          return std::unexpected(Error{Rule::kCancelled, "the encoding was asked to stop", at});
        }
        ++at;
      }
    }
  }
  if (auto e = impl_->EncodeText(text.substr(at), text_options, out, base, budget); !e) {
    return e;
  }
  return impl_->End(options, out, base);
}

std::expected<std::vector<SpecialSpan>, Error> Tokenizer::ControlSpans(
    std::string_view text) const {
  std::vector<SpecialSpan> out;
  std::size_t work = 0;
  for (std::size_t p = 0; p < text.size();) {
    if ((++work & 0xFFFFU) == 0 && !base::Pulse())
      return std::unexpected(Error{Rule::kCancelled, "the span search was asked to stop", p});
    std::size_t length = 0;
    const auto id = impl_->MatchSpecial(text, p, SpecialTokens::kParse, &length);
    if (id < 0) {
      ++p;
      continue;
    }
    if (Kind(id) == TokenKind::kControl) out.push_back({.offset = p, .length = length});
    p += length;
    if (!impl_->rstrip.empty() && impl_->rstrip[static_cast<std::size_t>(id)]) {
      while (p < text.size() && AsciiSpace(text[p])) {
        if ((++work & 0xFFFFU) == 0 && !base::Pulse())
          return std::unexpected(Error{Rule::kCancelled, "the span search was asked to stop", p});
        ++p;
      }
    }
  }
  return out;
}

std::expected<bool, Error> Tokenizer::ContainsControlText(std::string_view text) const {
  std::size_t work = 0;
  const auto search = [&](std::string_view bytes, unsigned phase) -> std::expected<bool, Error> {
    for (std::size_t p = 0; p < bytes.size(); ++p) {
      if (!impl_->special_first[phase][static_cast<unsigned char>(bytes[p])]) continue;
      std::uint32_t node = phase;
      for (std::size_t q = p; q < bytes.size() && q - p < kMaxSpecialBytes; ++q) {
        if ((++work & 0xFFFFU) == 0 && !base::Pulse())
          return std::unexpected(
              Error{Rule::kCancelled, "the control search was asked to stop", p});
        const auto edge = impl_->edges.find((static_cast<std::uint64_t>(node) << 8U) |
                                            static_cast<unsigned char>(bytes[q]));
        if (edge == impl_->edges.end()) break;
        node = edge->second;
        const auto id = impl_->terminal[node];
        if (id >= 0 && Kind(id) == TokenKind::kControl) return true;
      }
    }
    return false;
  };
  auto found = search(text, 0);
  if (!found || *found || impl_->normalized_tokens.empty()) return found;
  // Normalized added tokens match the text with its spaces as ▁.
  std::string normalized;
  normalized.reserve(text.size());
  for (const char c : text) {
    if ((++work & 0xFFFFU) == 0 && !base::Pulse())
      return std::unexpected(Error{Rule::kCancelled, "the control search was asked to stop"});
    if (c == ' ')
      normalized += "▁";
    else
      normalized.push_back(c);
  }
  return search(normalized, 1);
}

std::uint64_t Tokenizer::ForcedSpanBytes(std::size_t input_bytes) const {
  const auto most = std::numeric_limits<std::uint64_t>::max();
  if (input_bytes > most - 4) return most;
  const auto n = static_cast<std::uint64_t>(input_bytes) + 4;
  // Original/decoded/extended strings, canonical and committed IDs, three
  // trace vectors (one an open atom prefix's closed retrace), raw
  // source-offset maps and two copies of marked spans (at most one per byte)
  // coexist with one encoding/regex pass.
  const auto working = Times(WorkingBytes(static_cast<std::size_t>(n)), 4);
  const auto owned = Times(n, 160);
  return working > most - owned ? most : working + owned;
}

bool Tokenizer::CoversForced(TokenId candidate, std::size_t cover,
                             const DecodeOptions& options) const {
  if (candidate < 0 || static_cast<std::size_t>(candidate) >= size()) return false;
  const auto id = static_cast<std::size_t>(candidate);
  const auto kind = impl_->kinds[id];
  std::string_view bytes;
  if (kind == TokenKind::kControl && !options.control_tokens) return false;
  if (kind == TokenKind::kUnused && !impl_->ModelToken(candidate)) return false;
  if (kind == TokenKind::kControl || kind == TokenKind::kUserDefined)
    bytes = impl_->decode_added_whitespace ? impl_->decoded[id] : impl_->tokens[id];
  else {
    bytes = impl_->decoded[id];
    if (impl_->add_space_prefix && options.remove_space_prefix && bytes.starts_with(' '))
      bytes.remove_prefix(1);
  }
  return !bytes.empty() && bytes.size() >= cover;
}

namespace {
// The planner's encoding: forced bytes continue a stream, so SentencePiece
// adds no dummy prefix, unless the span is declared to start the stream.
// Forced text of n bytes encodes to at most n + 1 tokens: every token
// covers at least one byte, and the planner refuses text its normalization
// changes, so that is the encoding's whole reservation.
EncodeOptions ForcedEncoding(const ForcedSpanOptions& options, std::size_t bytes) {
  auto encode = options.encode;
  encode.continuation = encode.continuation || !options.decode.remove_space_prefix;
  encode.max_tokens = bytes + 1;
  return encode;
}
}  // namespace

std::expected<std::size_t, Error> Tokenizer::ForcedCoverBytes(
    std::string_view forced, const ForcedSpanOptions& options) const {
  if (options.encode.add_bos_eos)
    return std::unexpected(
        Error{Rule::kUnsupported, "forced spans cannot insert postprocessor tokens"});
  if (!options.marked && !options.specials.empty())
    return std::unexpected(Error{Rule::kUnsupported, "special spans require marked planning"});
  if (forced.size() > options.encode.max_bytes)
    return std::unexpected(Error{Rule::kInputTooLarge, "forced span exceeds max_bytes"});
  const auto orphan = ContinuationBytes(forced);
  if (orphan > 3) return std::unexpected(Error{Rule::kInvalidUtf8, "ill-formed UTF-8", 3});
  const auto complete = orphan + unicode::CompleteUtf8Prefix(forced.substr(orphan));
  if (auto valid = unicode::ValidateUtf8(forced.substr(orphan, complete - orphan)); !valid)
    return std::unexpected(
        Error{valid.error().rule, valid.error().reason, valid.error().item + orphan});
  if (auto spans = impl_->CheckSpans(forced.substr(0, complete), options.specials); !spans)
    return std::unexpected(spans.error());
  // Bytes that finish an already published character have no canonical
  // encoding of their own, so they prove no cover.
  if (orphan != 0 || forced.empty() || complete != forced.size()) return 0;
  const auto encode = ForcedEncoding(options, forced.size());
  if (auto valid = impl_->Preflight(forced, encode); !valid) return std::unexpected(valid.error());
  std::vector<char32_t> cps;
  if (auto valid = unicode::DecodeUtf8(forced, cps); !valid) return std::unexpected(valid.error());
  // Whitespace lookahead can split a formerly single block into a shorter
  // prefix plus a space attached to the next word. Do not impose that cover.
  if (unicode::IsWhiteSpace(cps.back())) return 0;
  const auto special = options.marked ? SpecialTokens::kUserDefinedOnly : options.encode.special;
  auto blocks = impl_->TraceBlocks(forced, options.specials, special, false);
  if (!blocks) return std::unexpected(blocks.error());
  // A token for an open block alone is not a cover proof: a higher-priority
  // crossing merge can turn ab + c into a / bc. Failed earlier regex branches
  // can likewise introduce a shorter boundary after more input arrives.
  if (blocks->ends.size() != 1 || blocks->stable != forced.size()) return 0;
  std::vector<TokenId> ids;
  if (auto encoded = options.marked ? EncodeMarked(forced, options.specials, encode, ids)
                                    : Encode(forced, encode, ids);
      !encoded)
    return std::unexpected(encoded.error());
  std::string reproduced;
  if (auto decoded = Decode(ids, options.decode, reproduced); !decoded)
    return std::unexpected(decoded.error());
  if (reproduced != forced)
    return std::unexpected(
        Error{Rule::kUnsupported, "canonical forced token does not reproduce its bytes"});
  return ids.size() == 1 ? forced.size() : 0;
}

// What PlanForcedSpan and PlanForcedPrefixes share: the planned text's
// canonical encoding, its stable blocks, and the canonical prefix that
// both bound and the actual allowance let a plan commit.
struct Tokenizer::ForcedTrace {
  bool continuation = false;  // the text finishes an already published character
  std::vector<TokenId> committed;
  std::size_t bytes = 0;         // the committed tokens' decoded bytes
  bool limited = false;          // the real token/context allowance cut them
  std::optional<TokenId> first;  // the canonical encoding's first token
};

std::expected<Tokenizer::ForcedTrace, Error> Tokenizer::TraceForced(
    std::string_view text, std::span<const SpecialSpan> spans,
    const ForcedSpanOptions& options) const {
  ForcedTrace out;
  const auto orphan = ContinuationBytes(text);
  if (orphan > 3) return std::unexpected(Error{Rule::kInvalidUtf8, "ill-formed UTF-8", 3});
  const auto complete = orphan + unicode::CompleteUtf8Prefix(text.substr(orphan));
  const auto available = text.substr(0, complete);
  if (auto valid = unicode::ValidateUtf8(available.substr(orphan)); !valid)
    return std::unexpected(
        Error{valid.error().rule, valid.error().reason, valid.error().item + orphan});
  if (auto checked = impl_->CheckSpans(available, spans); !checked)
    return std::unexpected(checked.error());
  if (orphan != 0) {
    // The text finishes a character whose lead byte is already published, so
    // it has no canonical encoding from here: an unchanged sampler choice on
    // the forced bytes continues the character.
    out.continuation = true;
    return out;
  }
  const auto special = options.marked ? SpecialTokens::kUserDefinedOnly : options.encode.special;
  const auto encode = ForcedEncoding(options, available.size());
  std::vector<TokenId> canonical;
  if (auto encoded = options.marked ? EncodeMarked(available, spans, encode, canonical)
                                    : Encode(available, encode, canonical);
      !encoded)
    return std::unexpected(encoded.error());
  std::string reproduced;
  if (auto decoded = Decode(canonical, options.decode, reproduced); !decoded)
    return std::unexpected(decoded.error());
  if (reproduced != available)
    return std::unexpected(
        Error{Rule::kUnsupported, "canonical forced span does not reproduce its bytes"});
  auto blocks = impl_->TraceBlocks(available, spans, special, false);
  if (!blocks) return std::unexpected(blocks.error());
  const std::size_t cut = blocks->stable;
  // Check canonical IDs for the complete known span, not an independently
  // encoded short prefix. This includes healing an unpublished sampled ID.
  // Canonicalization is funded by input bytes; publication is independently
  // clipped to the actual token/context allowance, never inside a token.
  const auto allowance =
      std::min({options.remaining_tokens, options.remaining_context, options.encode.max_tokens});
  out.committed.reserve(std::min(canonical.size(), allowance));
  std::string token_bytes;
  for (std::size_t i = 0; i < canonical.size(); ++i) {
    token_bytes.clear();
    auto decode = options.decode;
    decode.remove_space_prefix = decode.remove_space_prefix && i == 0;
    if (auto decoded = Decode(std::span(&canonical[i], 1), decode, token_bytes); !decoded)
      return std::unexpected(decoded.error());
    if (out.bytes + token_bytes.size() > cut) break;
    if (out.committed.size() == allowance) {
      out.limited = true;
      break;
    }
    out.committed.push_back(canonical[i]);
    out.bytes += token_bytes.size();
  }
  if (!canonical.empty()) out.first = canonical.front();
  return out;
}

std::expected<ForcedSpanPlan, Error> Tokenizer::PlanForcedSpan(
    std::optional<TokenId> selected, std::string_view forced,
    const ForcedSpanOptions& options) const {
  if (options.encode.add_bos_eos)
    return std::unexpected(
        Error{Rule::kUnsupported, "forced spans cannot insert postprocessor tokens"});
  if (!options.marked && !options.specials.empty())
    return std::unexpected(Error{Rule::kUnsupported, "special spans require marked planning"});
  ForcedSpanPlan out;
  std::string text;
  if (selected) {
    if (auto decoded = Decode(std::span(&*selected, 1), options.decode, text); !decoded)
      return std::unexpected(decoded.error());
    if (text.empty())
      return std::unexpected(Error{Rule::kUnsupported, "selected token has no decoded bytes"});
  }
  const auto selected_bytes = text.size();
  if (text.size() > options.encode.max_bytes ||
      forced.size() > options.encode.max_bytes - text.size())
    return std::unexpected(Error{Rule::kInputTooLarge, "forced span exceeds max_bytes"});
  text += forced;
  // Valid spans are disjoint and nonempty: at most one per planned byte.
  if (options.specials.size() > text.size())
    return std::unexpected(Error{Rule::kSpecialToken, "more spans than planned bytes"});
  // A selected control token is a marked span of its own bytes.
  std::vector<SpecialSpan> own;
  std::span<const SpecialSpan> spans = options.specials;
  if (options.marked && selected && Kind(*selected) == TokenKind::kControl) {
    auto rest = options.specials;
    if (!rest.empty() && rest.front().offset == 0 && rest.front().length == selected_bytes)
      rest = rest.subspan(1);
    else if (!rest.empty() && rest.front().offset < selected_bytes)
      return std::unexpected(
          Error{Rule::kSpecialToken, "a span overlaps the selected control token", 0});
    own.reserve(rest.size() + 1);
    own.push_back({.offset = 0, .length = selected_bytes});
    own.insert(own.end(), rest.begin(), rest.end());
    spans = own;
  }
  auto trace = TraceForced(text, spans, options);
  if (!trace) return std::unexpected(trace.error());
  if (trace->continuation) {
    // Hold all of it: an unchanged sampler choice continues the character.
    out.selected_covered = !selected;
    out.selected_matches_canonical_prefix = selected.has_value();
    return out;
  }
  out.limited = trace->limited;
  out.selected_matches_canonical_prefix = selected && trace->first == selected;
  if (selected && trace->bytes < selected_bytes) {
    // The selected token has not been published. Retain all of it until one
    // canonical candidate covers it; never publish a partial replacement.
    out.selected_covered = false;
    return out;
  }
  out.selected_covered = true;
  out.tokens = std::move(trace->committed);
  return out;
}

std::expected<ForcedPrefixPlan, Error> Tokenizer::PlanForcedPrefixes(
    std::string_view forced, const ForcedSpanOptions& options) const {
  if (options.encode.add_bos_eos)
    return std::unexpected(
        Error{Rule::kUnsupported, "forced spans cannot insert postprocessor tokens"});
  if (!options.marked && !options.specials.empty())
    return std::unexpected(Error{Rule::kUnsupported, "special spans require marked planning"});
  if (forced.size() > options.encode.max_bytes)
    return std::unexpected(Error{Rule::kInputTooLarge, "forced span exceeds max_bytes"});
  if (options.specials.size() > forced.size())
    return std::unexpected(Error{Rule::kSpecialToken, "more spans than planned bytes"});
  auto trace = TraceForced(forced, options.specials, options);
  if (!trace) return std::unexpected(trace.error());
  return ForcedPrefixPlan{.continuation = trace->continuation,
                          .committed_bytes = trace->bytes,
                          .limited = trace->limited,
                          .first = trace->first};
}

std::string_view Tokenizer::Piece(TokenId id, const DecodeOptions& options) const {
  if (id < 0 || static_cast<std::size_t>(id) >= impl_->tokens.size()) return {};
  const auto u = static_cast<std::size_t>(id);
  switch (impl_->kinds[u]) {
    case TokenKind::kUnused:
      if (!impl_->ModelToken(id)) return {};
      [[fallthrough]];
    case TokenKind::kByte:
    case TokenKind::kNormal: {
      std::string_view piece = impl_->decoded[u];
      if (impl_->add_space_prefix && options.remove_space_prefix && piece.starts_with(' '))
        piece.remove_prefix(1);
      return piece;
    }
    case TokenKind::kUserDefined:
      return impl_->decode_added_whitespace ? impl_->decoded[u] : impl_->tokens[u];
    case TokenKind::kControl:
      if (!options.control_tokens) return {};
      return impl_->decode_added_whitespace ? impl_->decoded[u] : impl_->tokens[u];
  }
  return {};
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
      case TokenKind::kUnused:
        if (!impl_->ModelToken(id)) break;
        [[fallthrough]];
      case TokenKind::kByte:
      case TokenKind::kNormal: {
        std::string_view piece = impl_->decoded[u];
        if (i == 0 && impl_->add_space_prefix && options.remove_space_prefix &&
            piece.starts_with(' ')) {
          piece.remove_prefix(1);
        }
        out += piece;
      } break;
      case TokenKind::kUserDefined:
        out += impl_->decode_added_whitespace ? impl_->decoded[u] : impl_->tokens[u];
        break;
      case TokenKind::kControl:
        if (options.control_tokens) {
          out += impl_->decode_added_whitespace ? impl_->decoded[u] : impl_->tokens[u];
        }
        break;
    }
  }
  return {};
}

std::expected<void, Error> StreamDecoder::Push(TokenId token, std::string& out) {
  auto options = options_;
  options.remove_space_prefix = options.remove_space_prefix && !started_;
  if (auto d = tokenizer_->Decode(std::span(&token, 1), options, pending_); !d) {
    return d;
  }
  started_ = true;
  const std::size_t complete = unicode::CompleteUtf8Prefix(pending_);
  out += unicode::ReplaceInvalidUtf8(std::string_view(pending_).substr(0, complete));
  pending_.erase(0, complete);
  return {};
}

void StreamDecoder::Finish(std::string& out) {
  out += unicode::ReplaceInvalidUtf8(pending_);
  pending_.clear();
}

}  // namespace llmp::tokenizer
