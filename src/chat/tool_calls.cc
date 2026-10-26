// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "chat/tool_calls.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <unordered_set>
#include <utility>

namespace llmp::chat {
namespace {
namespace j = base::json;
using detail::ToolCursor;
using detail::ToolEnvelopeCursor;
using detail::ToolObjectCursor;
using detail::ToolScan;
using EnvelopeInfo = detail::ToolEnvelopeInfo;

std::unexpected<std::string> Bad(std::string_view message) {
  return std::unexpected(std::string(message));
}
// Internal representations can grow beyond admitted raw bytes through JSON
// quoting and envelope keys. Share these checked bounds with prepayment; the
// caller's depth, value-count and decoded-string bounds still apply to JSON.
std::optional<std::size_t> OutputJsonBytes(std::size_t raw) {
  std::size_t bytes = 0;
  if (__builtin_mul_overflow(raw, std::size_t{6}, &bytes) ||
      __builtin_add_overflow(bytes, std::size_t{32}, &bytes))
    return {};
  return bytes;
}
std::optional<std::size_t> DefinitionJsonBytes(const ToolDefinition& definition) {
  std::size_t bytes = 0;
  if (__builtin_add_overflow(definition.name.size(), definition.description.size(), &bytes) ||
      __builtin_mul_overflow(bytes, std::size_t{6}, &bytes) ||
      __builtin_add_overflow(bytes, definition.parameters_json.size(), &bytes) ||
      __builtin_add_overflow(bytes, definition.function_json.size(), &bytes) ||
      __builtin_add_overflow(bytes, std::size_t{128}, &bytes))
    return {};
  return bytes;
}
bool Space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
std::string_view Strip(std::string_view s) {
  while (!s.empty() && Space(s.front())) s.remove_prefix(1);
  while (!s.empty() && Space(s.back())) s.remove_suffix(1);
  return s;
}
std::string_view Unwrap(std::string_view s, bool trim) {
  if (!trim) return s;
  if (s.starts_with('\n')) s.remove_prefix(1);
  if (s.ends_with('\n')) s.remove_suffix(1);
  return s;
}
// The decoded strings that generated documents may carry; nesting follows
// the caller's JSON depth (api::kMaxJsonDepth for served requests).
j::Limits SmallLimits(std::size_t bytes) {
  return {.max_bytes = bytes, .max_depth = 1, .max_values = 1, .max_string_bytes = bytes};
}

std::string Json(j::Value value) {
  std::string out;
  j::AppendValue(value, out);
  return out;
}

// Schema conversion reads `type` (a name or a list) and, minimally, the
// alternatives of anyOf/oneOf (schemas with a `type`, or bare type names).
enum TypeBit : unsigned {
  kString = 1,
  kInteger = 2,
  kNumber = 4,
  kBoolean = 8,
  kObject = 16,
  kArray = 32,
  kNull = 64,
};
unsigned TypeBits(std::string_view type) {
  return type == "string"    ? kString
         : type == "integer" ? kInteger
         : type == "number"  ? kNumber
         : type == "boolean" ? kBoolean
         : type == "object"  ? kObject
         : type == "array"   ? kArray
         : type == "null"    ? kNull
                             : 0U;
}
unsigned TypeField(std::optional<j::Value> type) {
  if (!type) return 0;
  if (type->is_string()) return TypeBits(type->string());
  unsigned mask = 0;
  if (type->is_array())
    for (std::size_t i = 0; i < type->size(); ++i)
      if (type->at(i).is_string()) mask |= TypeBits(type->at(i).string());
  return mask;
}
unsigned SchemaTypes(std::optional<j::Value> schema) {
  if (!schema || !schema->is_object()) return 0;
  unsigned mask = TypeField(schema->find("type"));
  for (const std::string_view key : {std::string_view("anyOf"), std::string_view("oneOf")}) {
    const auto alternatives = schema->find(key);
    if (!alternatives || !alternatives->is_array()) continue;
    for (std::size_t i = 0; i < alternatives->size(); ++i) {
      const auto alternative = alternatives->at(i);
      if (alternative.is_string())
        mask |= TypeBits(alternative.string());
      else if (alternative.is_object())
        mask |= TypeField(alternative.find("type"));
    }
  }
  return mask;
}
bool SchemaAccepts(unsigned mask, j::Value value) {
  switch (value.kind()) {
    case j::Kind::kNull:
      return (mask & kNull) != 0;
    case j::Kind::kFalse:
    case j::Kind::kTrue:
      return (mask & kBoolean) != 0;
    case j::Kind::kNumber:
      return (mask & kNumber) != 0 || ((mask & kInteger) != 0 && value.is_integer());
    case j::Kind::kString:
      return (mask & kString) != 0;
    case j::Kind::kArray:
      return (mask & kArray) != 0;
    case j::Kind::kObject:
      return (mask & kObject) != 0;
  }
  return false;
}
// A raw tagged value: a JSON string unless an explicit encoding or a schema
// admitting a non-string type says JSON; only those values are parsed.
ToolStatus AppendScalar(std::string_view value, unsigned schema, std::optional<bool> string,
                        const j::Limits& limits, std::string& out) {
  const bool json = string ? !*string : (schema & ~static_cast<unsigned>(kString)) != 0;
  if (!json) {
    j::AppendQuoted(value, out);
    return {};
  }
  auto parsed = j::Parse(value, limits);
  if (string) {
    if (!parsed) return Bad("tool parameter contains invalid JSON");
    j::AppendValue(parsed->root(), out);
    return {};
  }
  if (parsed && !parsed->root().is_string() && SchemaAccepts(schema, parsed->root()))
    j::AppendValue(parsed->root(), out);
  else
    j::AppendQuoted(value, out);
  return {};
}

// A candidate token extends the committed pending bytes without copying them.
// Subviews and searches can span the join; only short name/header lookup keys
// are materialized. Argument payload is never copied for vocabulary probes.
class JoinedText {
 public:
  JoinedText(std::string_view first, std::string_view second = {})
      : first_(first), second_(second) {}
  std::size_t size() const { return first_.size() + second_.size(); }
  bool empty() const { return size() == 0; }
  char operator[](std::size_t i) const {
    return i < first_.size() ? first_[i] : second_[i - first_.size()];
  }
  char front() const { return (*this)[0]; }
  char back() const { return (*this)[size() - 1]; }
  JoinedText substr(std::size_t at, std::size_t count = std::string_view::npos) const {
    at = std::min(at, size());
    count = std::min(count, size() - at);
    if (at >= first_.size()) return {second_.substr(at - first_.size(), count)};
    const auto a = first_.substr(at, count);
    return {a, second_.substr(0, count - a.size())};
  }
  void remove_prefix(std::size_t n) { *this = substr(n); }
  void remove_suffix(std::size_t n) { *this = substr(0, size() - n); }
  bool starts_with(std::string_view prefix) const {
    if (prefix.size() > size()) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i)
      if ((*this)[i] != prefix[i]) return false;
    return true;
  }
  std::size_t find(std::string_view needle, std::size_t at = 0) const {
    for (; at <= size() && needle.size() <= size() - at; ++at)
      if (substr(at).starts_with(needle)) return at;
    return std::string_view::npos;
  }
  // A view when the bytes are in one piece (the parser's own buffer).
  std::optional<std::string_view> contiguous() const {
    if (second_.empty()) return first_;
    if (first_.empty()) return second_;
    return std::nullopt;
  }
  explicit operator std::string() const { return std::string(first_) + std::string(second_); }

 private:
  std::string_view first_, second_;
};
JoinedText Strip(JoinedText s) {
  while (!s.empty() && Space(s.front())) s.remove_prefix(1);
  while (!s.empty() && Space(s.back())) s.remove_suffix(1);
  return s;
}
bool PrefixOf(JoinedText prefix, std::string_view text) {
  if (prefix.size() > text.size()) return false;
  for (std::size_t i = 0; i < prefix.size(); ++i)
    if (prefix[i] != text[i]) return false;
  return true;
}
bool PrefixOf(std::string_view prefix, std::string_view text) { return text.starts_with(prefix); }
bool HasText(JoinedText text) { return !Strip(text).empty(); }

// The first needle outside a quoted string. A partial needle (or quote) at
// the end holds the scan there. Nesting is a count: only whether a closing
// bracket is inside a value matters. `watch` tokens are recorded (the first
// one) without ending the scan.
template <class Text>
std::optional<std::size_t> Outside(Text text, std::string_view needle, std::string_view quote,
                                   ToolScan& scan, std::string_view second = {},
                                   std::string_view third = {}, bool structured = true,
                                   std::span<const std::string> watch = {}) {
  while (scan.at < text.size()) {
    const auto rest = text.substr(scan.at);
    const auto boundary = [&](std::string_view token) {
      if (token.empty() || !rest.starts_with(token)) return false;
      // An XML-like boundary outside a quoted string closes a malformed JSON
      // value too: unclosed braces must not turn a fully closed bad call into
      // a length-truncated call that would silently disappear.
      return scan.depth == 0 ||
             (token.front() != '}' && token.front() != ']' && token.front() != ',');
    };
    if (!scan.quoted && (boundary(needle) || boundary(second) || boundary(third))) return scan.at;
    if (scan.quoted) {
      if (quote == "\"" && !scan.escaped && rest.front() == '\\') {
        scan.escaped = true;
        ++scan.at;
        continue;
      }
      if (!scan.escaped && rest.starts_with(quote)) {
        scan.quoted = false;
        scan.at += quote.size();
        continue;
      }
      if (!scan.escaped && rest.size() < quote.size() && PrefixOf(rest, quote)) return {};
      scan.escaped = false;
    } else {
      if (structured && rest.starts_with(quote)) {
        scan.quoted = true;
        scan.at += quote.size();
        continue;
      }
      if ((structured && rest.size() < quote.size() && PrefixOf(rest, quote)) ||
          (rest.size() < needle.size() && PrefixOf(rest, needle)) ||
          (rest.size() < second.size() && PrefixOf(rest, second)) ||
          (rest.size() < third.size() && PrefixOf(rest, third)))
        return {};
      for (const auto& token : watch) {
        if (token.empty()) continue;
        if (rest.size() < token.size() && PrefixOf(rest, token)) return {};
        if (scan.alias_at == std::string::npos && rest.starts_with(token)) {
          scan.alias_at = scan.at;
          scan.alias_bytes = token.size();
        }
      }
      const char c = rest.front();
      if (structured && (c == '{' || c == '[')) ++scan.depth;
      if ((c == '}' || c == ']') && scan.depth != 0) --scan.depth;
    }
    ++scan.at;
  }
  return {};
}

// Advance only through newly appended bytes. Projections of the current field
// remain provisional until a top-level comma; this preserves the lexical
// treatment of malformed values and duplicate names. Only bounded key/name
// slices are materialized, never argument payloads.
template <class Text>
EnvelopeInfo Envelope(Text text, const ToolCallFormat& format, const ToolRequest& request,
                      ToolEnvelopeCursor& scan) {
  using Phase = ToolEnvelopeCursor::Phase;
  const auto project = [&](bool ended) {
    auto info = scan.committed;
    if (scan.arguments_key) info.arguments = true;
    if (scan.name_key) {
      info.name = true;
      const auto encoded = text.substr(scan.value_start, scan.trim_end - scan.value_start);
      // A name string still open (its last quote escaped) is incomplete.
      if (!scan.quoted && encoded.size() >= 2 && encoded.front() == '"' && encoded.back() == '"' &&
          encoded.size() <= 6 * request.NameBytes() + 2) {
        const auto bytes = static_cast<std::string>(encoded);
        auto parsed = j::Parse(bytes, SmallLimits(bytes.size()));
        if (parsed && parsed->root().is_string()) {
          const auto name = parsed->root().string();
          info.allowed = request.choice().kind == ToolChoiceKind::kNamed
                             ? name == request.choice().name
                             : request.Declared(name);
        }
      } else if (!ended) {
        info.name = false;
      }
    }
    return info;
  };
  const auto previous_trim = scan.trim_end;
  const auto previous_phase = scan.phase;
  while (scan.at < text.size() && scan.phase != Phase::kStopped) {
    const char ch = text[scan.at];
    switch (scan.phase) {
      case Phase::kBetween:
        if (Space(ch) || ch == ',') {
          ++scan.at;
        } else if (ch == '"') {
          scan.key_start = scan.at++;
          scan.escaped = false;
          scan.phase = Phase::kKey;
        } else {
          scan.phase = Phase::kStopped;
        }
        break;
      case Phase::kKey:
        ++scan.at;
        if (!scan.escaped && ch == '"') {
          scan.name_key = scan.arguments_key = false;
          if (scan.at - scan.key_start <=
              6 * std::max(format.name_key.size(), format.arguments_key.size()) + 2) {
            const auto bytes =
                static_cast<std::string>(text.substr(scan.key_start, scan.at - scan.key_start));
            auto parsed = j::Parse(bytes, SmallLimits(bytes.size()));
            if (parsed && parsed->root().is_string()) {
              scan.name_key = parsed->root().string() == format.name_key;
              scan.arguments_key = parsed->root().string() == format.arguments_key;
            }
          }
          scan.phase = Phase::kColon;
        } else {
          scan.escaped = !scan.escaped && ch == '\\';
        }
        break;
      case Phase::kColon:
        if (Space(ch)) {
          ++scan.at;
        } else if (ch == ':') {
          ++scan.at;
          scan.phase = Phase::kValueStart;
        } else {
          scan.phase = Phase::kStopped;
        }
        break;
      case Phase::kValueStart:
        if (Space(ch)) {
          ++scan.at;
        } else {
          scan.value_start = scan.trim_end = scan.at;
          scan.depth = 0;
          scan.quoted = scan.escaped = false;
          scan.phase = Phase::kValue;
        }
        break;
      case Phase::kValue:
        if (!scan.quoted && scan.depth == 0 && ch == ',') {
          scan.committed = scan.projected = project(true);
          ++scan.at;
          scan.phase = Phase::kBetween;
          break;
        }
        if (scan.quoted) {
          if (!scan.escaped && ch == '\\') {
            scan.escaped = true;
          } else {
            if (!scan.escaped && ch == '"') scan.quoted = false;
            scan.escaped = false;
          }
        } else if (ch == '"') {
          scan.quoted = true;
        } else if (ch == '{' || ch == '[') {
          ++scan.depth;
        } else if ((ch == '}' || ch == ']') && scan.depth != 0) {
          --scan.depth;
        }
        ++scan.at;
        if (!Space(ch)) scan.trim_end = scan.at;
        break;
      case Phase::kStopped:
        break;
    }
  }
  if (scan.phase == Phase::kValue &&
      (scan.trim_end != previous_trim || previous_phase != Phase::kValue))
    scan.projected = project(false);
  else if (scan.phase != Phase::kValue)
    scan.projected = scan.committed;
  return scan.projected;
}
// The decoded prefix of an open JSON string (raw bytes after its opening
// quote), stopping before an incomplete escape. Empty optional: not a valid
// JSON string prefix.
std::optional<std::string> JsonStringPrefix(std::string_view raw) {
  std::string out;
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const char ch = raw[i];
    if (static_cast<unsigned char>(ch) < 0x20 || ch == '"') return std::nullopt;
    if (ch != '\\') {
      out += ch;
      continue;
    }
    if (i + 1 == raw.size()) return out;
    const char e = raw[++i];
    if (e == 'u') {
      if (raw.size() - i - 1 < 4) return out;
      // A complete \u escape: decode through the JSON parser (surrogates).
      std::string piece = "\"\\u" + std::string(raw.substr(i + 1, 4));
      if (raw.size() - i - 5 >= 6 && raw.substr(i + 5, 2) == "\\u") {
        piece += raw.substr(i + 5, 6);
        piece += '"';
        if (auto pair = j::Parse(piece, SmallLimits(piece.size())); pair) {
          out += pair->root().string();
          i += 10;
          continue;
        }
        piece.resize(6);
      }
      piece += '"';
      auto single = j::Parse(piece, SmallLimits(piece.size()));
      if (!single) {
        // A high surrogate whose low half has not arrived yet.
        const auto hex = raw.substr(i + 1, 4);
        if ((hex[0] == 'd' || hex[0] == 'D') &&
            std::string_view("89abAB").find(hex[1]) != std::string_view::npos)
          return out;
        return std::nullopt;
      }
      out += single->root().string();
      i += 4;
      continue;
    }
    const std::string_view simple = "\"\\/bfnrt";
    const auto at = simple.find(e);
    if (at == std::string_view::npos) return std::nullopt;
    out += std::string_view("\"\\/\b\f\n\r\t")[at];
  }
  return out;
}

// Implicit conversion only recognizes schema-admitted non-string values.
// Keep the lexical decision identical in streaming and final parsing: unknown
// schemas and non-container types cannot make raw braces/quotes hide tags.
int ParameterSyntax(unsigned schema, std::optional<bool> string) {
  if (string) return *string ? 0 : 1;  // raw text, or explicitly encoded JSON
  return ((schema & kObject) != 0 ? 2 : 0) | ((schema & kArray) != 0 ? 4 : 0);
}
template <class Text>
bool StructuredValue(Text body, int syntax) {
  if (syntax == 1) return true;
  body = Strip(body);
  return !body.empty() &&
         (((syntax & 2) != 0 && body.front() == '{') || ((syntax & 4) != 0 && body.front() == '['));
}

struct TaggedBoundary {
  std::size_t at, bytes;
};
struct ParameterHeader {
  std::string_view begin, name_end;
};
template <class Text>
std::string_view CallEnd(Text rest, const ToolCallFormat& format) {
  if (rest.starts_with(format.call_end)) return format.call_end;
  for (const auto& alias : format.call_end_aliases)
    if (rest.starts_with(alias)) return alias;
  return {};
}
template <class Text>
ParameterHeader Header(Text rest, const ToolCallFormat& format) {
  if (rest.starts_with(format.parameter_begin))
    return {format.parameter_begin, format.parameter_name_end};
  for (const auto& alias : format.parameter_aliases)
    if (rest.starts_with(alias.begin)) return {alias.begin, alias.name_end};
  return {};
}
template <class Text>
bool PartialTaggedBoundary(Text rest, const ToolCallFormat& format) {
  if (PrefixOf(rest, format.call_end) || PrefixOf(rest, format.parameter_begin)) return true;
  for (const auto& alias : format.call_end_aliases)
    if (PrefixOf(rest, alias)) return true;
  for (const auto& alias : format.parameter_aliases)
    if (PrefixOf(rest, alias.begin)) return true;
  return false;
}
// Phases: 0 the call name, 1 between parameters, 3 a value's start, 2 inside
// a value. Text between parameters that cannot begin a boundary is outside
// the grammar at once, so a probe never scans it again.
using TaggedFound = std::expected<std::optional<TaggedBoundary>, std::string>;
template <class Text>
TaggedFound TaggedEnd(Text body, const ToolCallFormat& format, const ToolRequest& request,
                      ToolScan& scan, unsigned& phase, int& json, std::size_t& call_name_end,
                      std::size_t& header_search) {
  if (phase == 0) {
    call_name_end = body.find(format.name_end, scan.at);
    if (call_name_end == std::string_view::npos) {
      scan.at =
          body.size() >= format.name_end.size() ? body.size() - format.name_end.size() + 1 : 0;
      return std::nullopt;
    }
    scan.at = call_name_end + format.name_end.size();
    phase = 1;
  }
  const auto call_name = Strip(body.substr(0, call_name_end));
  for (;;) {
    if (phase == 1) {
      while (scan.at < body.size() && Space(body[scan.at])) ++scan.at;
      const auto rest = body.substr(scan.at);
      if (const auto end = CallEnd(rest, format); !end.empty())
        return TaggedBoundary{scan.at, end.size()};
      if (PartialTaggedBoundary(rest, format)) return std::nullopt;
      const auto header = Header(rest, format);
      if (header.begin.empty()) return Bad("invalid tool parameter boundary");
      const auto name_at = scan.at + header.begin.size();
      const auto end = body.find(header.name_end, std::max(name_at, header_search));
      if (end == std::string_view::npos) {
        header_search = body.size() >= header.name_end.size()
                            ? std::max(name_at, body.size() - header.name_end.size() + 1)
                            : name_at;
        return std::nullopt;
      }
      header_search = end;
      const auto name = body.substr(name_at, end - name_at);
      auto value_at = end + header.name_end.size();
      std::optional<bool> string;
      if (!format.string_suffix.empty()) {
        const auto suffix = body.substr(value_at);
        if (suffix.starts_with(format.string_suffix)) {
          string = true;
          value_at += format.string_suffix.size();
        } else if (suffix.starts_with(format.json_suffix)) {
          string = false;
          value_at += format.json_suffix.size();
        } else if (PrefixOf(suffix, format.string_suffix) || PrefixOf(suffix, format.json_suffix)) {
          return std::nullopt;
        } else {
          return Bad("tool parameter needs an encoding suffix");
        }
      }
      json = ParameterSyntax(SchemaTypes(request.Parameter(static_cast<std::string>(call_name),
                                                           static_cast<std::string>(name))),
                             string);
      scan.at = value_at;
      header_search = 0;
      phase = 3;
    }
    if (phase == 3) {
      while (scan.at < body.size() && Space(body[scan.at])) ++scan.at;
      if (scan.at == body.size()) return std::nullopt;
      json = StructuredValue(body.substr(scan.at), json) ? 1 : 0;
      scan.alias_at = std::string::npos;
      phase = 2;
    }
    const auto end = Outside(body, format.call_end, "\"", scan, format.parameter_begin,
                             format.parameter_end, json == 1, format.call_end_aliases);
    if (!end) return std::nullopt;
    if (body.substr(*end).starts_with(format.call_end))
      return TaggedBoundary{*end, format.call_end.size()};
    if (body.substr(*end).starts_with(format.parameter_end)) scan.at += format.parameter_end.size();
    scan.alias_at = std::string::npos;
    phase = 1;
  }
}

// ---- Object notation: one byte-stepped grammar --------------------------
//
// object := '{' (member (',' member)*)? '}'; member := key ':' value;
// key := quoted | bare (bytes up to ':', trimmed, nonempty, no ",{}[]");
// value := object | '[' (value (',' value)*)? ']' | quoted | scalar;
// quoted := quote raw-bytes quote (a JSON string when the quote is '"');
// scalar := true | false | null | null_word | a JSON number.
// Whitespace separates tokens. The caller treats call_end outside a string
// as the call's boundary, before stepping.
constexpr std::uint8_t kNumberDead = 9;
std::uint8_t NumberStep(std::uint8_t state, char ch) {
  const bool digit = ch >= '0' && ch <= '9';
  switch (state) {
    case 0:  // start
      return ch == '-' ? 1 : ch == '0' ? 2 : digit ? 3 : kNumberDead;
    case 1:  // after '-'
      return ch == '0' ? 2 : digit ? 3 : kNumberDead;
    case 2:  // a leading zero
      return ch == '.' ? 4 : (ch == 'e' || ch == 'E') ? 6 : kNumberDead;
    case 3:  // integer digits
      return digit ? 3 : ch == '.' ? 4 : (ch == 'e' || ch == 'E') ? 6 : kNumberDead;
    case 4:  // after '.'
      return digit ? 5 : kNumberDead;
    case 5:  // fraction digits
      return digit ? 5 : (ch == 'e' || ch == 'E') ? 6 : kNumberDead;
    case 6:  // after 'e'
      return (ch == '+' || ch == '-') ? 7 : digit ? 8 : kNumberDead;
    case 7:  // after the exponent sign
      return digit ? 8 : kNumberDead;
    case 8:  // exponent digits
      return digit ? 8 : kNumberDead;
    default:
      return kNumberDead;
  }
}
bool NumberComplete(std::uint8_t state) {
  return state == 2 || state == 3 || state == 5 || state == 8;
}
std::array<std::string_view, 4> Words(const ToolCallFormat& format) {
  return {"true", "false", "null", format.null_word};
}
int HexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// One grammar for sampling probes (out == nullptr) and conversion to JSON.
class ObjectGrammar {
 public:
  // Conversion also refuses a key repeated within one object, as JSON
  // documents in this runtime do (keys: one set per open container).
  using Keys = std::vector<std::unordered_set<std::string>>;
  ObjectGrammar(const ToolCallFormat& format, std::size_t max_depth, Keys* keys = nullptr)
      : format_(format), max_depth_(std::min<std::size_t>(max_depth, 64)), keys_(keys) {}
  // False: the byte is outside the grammar (the cursor is then rejected).
  bool Step(ToolObjectCursor& c, char ch, std::string* out, std::string& scratch) const {
    using S = ToolObjectCursor::State;
    const std::string_view q = format_.quote;
    const auto reject = [&c] {
      c.state = S::kRejected;
      return false;
    };
    for (;;) {
      switch (c.state) {
        case S::kValue:
        case S::kValueOrClose:
          if (Space(ch)) return true;
          if (c.state == S::kValueOrClose && ch == ']') return Close(c, ch, out);
          if (ch == '{' || ch == '[') {
            if ((c.depth == 0 && ch != '{') || c.depth + 1 > max_depth_) return reject();
            if (ch == '{')
              c.objects |= std::uint64_t{1} << c.depth;
            else
              c.objects &= ~(std::uint64_t{1} << c.depth);
            ++c.depth;
            if (keys_) keys_->emplace_back();
            if (out) *out += ch;
            c.state = ch == '{' ? S::kKeyOrClose : S::kValueOrClose;
            return true;
          }
          if (c.depth == 0) return reject();
          if (!q.empty() && ch == q.front()) {
            c.key = false;
            return OpenQuote(c, out, scratch);
          }
          return StartScalar(c, ch, scratch, out) || reject();
        case S::kKeyOrClose:
        case S::kKey:
          if (Space(ch)) return true;
          if (c.state == S::kKeyOrClose && ch == '}') return Close(c, ch, out);
          if (!q.empty() && ch == q.front()) {
            c.key = true;
            return OpenQuote(c, out, scratch);
          }
          if (ch == ':' || ch == ',' || ch == '{' || ch == '}' || ch == '[' || ch == ']')
            return reject();
          c.key_text = true;
          if (out) scratch.assign(1, ch);
          c.state = S::kBareKey;
          return true;
        case S::kQuoteOpen:
          if (ch == q[c.matched]) {
            if (++c.matched == q.size()) EnterString(c, out, scratch);
            return true;
          }
          // Not a quote: only a bare key may begin with these bytes.
          if (!c.key) return reject();
          c.key_text = HasText(JoinedText(q.substr(0, c.matched)));
          if (out) scratch.assign(q.substr(0, c.matched));
          c.matched = 0;
          c.state = S::kBareKey;
          continue;
        case S::kBareKey:
          if (ch == ':') {
            if (!c.key_text) return reject();
            if (out) {
              const auto key = Strip(std::string_view(scratch));
              if (!NewKey(key)) return reject();
              j::AppendQuoted(key, *out);
              *out += ':';
            }
            c.state = S::kValue;
            return true;
          }
          if (ch == ',' || ch == '{' || ch == '}' || ch == '[' || ch == ']') return reject();
          c.key_text = c.key_text || !Space(ch);
          if (out) scratch += ch;
          return true;
        case S::kColon:
          if (Space(ch)) return true;
          if (ch != ':') return reject();
          if (out) *out += ':';
          c.state = S::kValue;
          return true;
        case S::kQuoted:
          return QuotedByte(c, ch, out, scratch) || reject();
        case S::kJsonString:
          return JsonStringByte(c, ch, out, scratch) || reject();
        case S::kScalar:
          if (Space(ch) || ch == ',' || ch == '}' || ch == ']') {
            if (!FinishScalar(c, scratch, out)) return reject();
            c.state = S::kAfterValue;
            continue;
          }
          return ScalarByte(c, ch, scratch, out) || reject();
        case S::kAfterValue: {
          if (Space(ch)) return true;
          const bool object = ((c.objects >> (c.depth - 1)) & 1U) != 0;
          if (ch == ',') {
            if (out) *out += ',';
            c.state = object ? S::kKey : S::kValue;
            return true;
          }
          if (ch == (object ? '}' : ']')) return Close(c, ch, out);
          return reject();
        }
        case S::kDone:
          return Space(ch) || reject();
        case S::kRejected:
          return false;
      }
      return reject();
    }
  }
  // Inside a string, call_end text is literal; elsewhere it bounds the call.
  static bool InString(const ToolObjectCursor& c) {
    return c.state == ToolObjectCursor::State::kQuoted ||
           c.state == ToolObjectCursor::State::kJsonString;
  }

 private:
  bool Close(ToolObjectCursor& c, char ch, std::string* out) const {
    if (out) *out += ch;
    if (keys_ && !keys_->empty()) keys_->pop_back();
    --c.depth;
    c.state = c.depth == 0 ? ToolObjectCursor::State::kDone : ToolObjectCursor::State::kAfterValue;
    return true;
  }
  bool OpenQuote(ToolObjectCursor& c, std::string* out, std::string& scratch) const {
    c.matched = 1;
    if (format_.quote.size() == 1)
      EnterString(c, out, scratch);
    else
      c.state = ToolObjectCursor::State::kQuoteOpen;
    return true;
  }
  void EnterString(ToolObjectCursor& c, std::string* out, std::string& scratch) const {
    c.matched = 0;
    c.escape = 0;
    c.high_surrogate = false;
    if (out) scratch.clear();
    if (format_.quote == "\"") {
      if (out) scratch = "\"";
      c.state = ToolObjectCursor::State::kJsonString;
    } else {
      c.state = ToolObjectCursor::State::kQuoted;
    }
  }
  bool NewKey(std::string_view key) const {
    return !keys_ || keys_->empty() || keys_->back().emplace(key).second;
  }
  bool CloseString(ToolObjectCursor& c, std::string* out, std::string& scratch) const {
    c.matched = 0;
    c.state = c.key ? ToolObjectCursor::State::kColon : ToolObjectCursor::State::kAfterValue;
    if (!out) return true;
    if (format_.quote == "\"") {
      scratch += '"';
      if (c.key) {
        auto decoded = j::Parse(scratch, SmallLimits(scratch.size()));
        if (!decoded || !NewKey(decoded->root().string())) return false;
      }
      *out += scratch;  // already validated JSON string text
      return true;
    }
    if (c.key && !NewKey(scratch)) return false;
    j::AppendQuoted(scratch, *out);
    return true;
  }
  bool QuotedByte(ToolObjectCursor& c, char ch, std::string* out, std::string& scratch) const {
    const std::string_view q = format_.quote;
    if (ch == q[c.matched]) {
      if (++c.matched == q.size()) return CloseString(c, out, scratch);
      return true;
    }
    // The held delimiter bytes were content; keep the longest suffix of them
    // plus this byte that can still begin the delimiter.
    std::array<char, 256> held{};
    const std::size_t size = std::min<std::size_t>(c.matched, held.size() - 1);
    std::copy_n(q.data(), size, held.data());
    held[size] = ch;
    const std::string_view bytes(held.data(), size + 1);
    std::size_t keep = 0;
    for (std::size_t k = std::min(size, q.size() - 1); k > 0; --k)
      if (bytes.ends_with(q.substr(0, k))) {
        keep = k;
        break;
      }
    if (out) scratch += bytes.substr(0, bytes.size() - keep);
    c.matched = static_cast<std::uint8_t>(keep);
    return true;
  }
  bool JsonStringByte(ToolObjectCursor& c, char ch, std::string* out, std::string& scratch) const {
    if (out) scratch += ch;
    if (c.escape == 1) {
      if (ch == 'u') {
        c.escape = 2;
        c.hex = 0;
        return true;
      }
      if (std::string_view("\"\\/bfnrt").find(ch) == std::string_view::npos || c.high_surrogate)
        return false;
      c.escape = 0;
      return true;
    }
    if (c.escape >= 2) {
      const int digit = HexDigit(ch);
      if (digit < 0) return false;
      c.hex = static_cast<std::uint16_t>((static_cast<unsigned>(c.hex) << 4U) |
                                         static_cast<unsigned>(digit));
      if (++c.escape < 6) return true;
      c.escape = 0;
      const bool high = c.hex >= 0xD800 && c.hex <= 0xDBFF;
      const bool low = c.hex >= 0xDC00 && c.hex <= 0xDFFF;
      if (c.high_surrogate) {
        if (!low) return false;
        c.high_surrogate = false;
        return true;
      }
      if (low) return false;
      c.high_surrogate = high;
      return true;
    }
    if (ch == '\\') {
      c.escape = 1;
      return true;
    }
    if (c.high_surrogate || static_cast<unsigned char>(ch) < 0x20) return false;
    if (ch == '"') {
      if (out) scratch.pop_back();  // CloseString appends the closing quote
      return CloseString(c, out, scratch);
    }
    return true;
  }
  bool StartScalar(ToolObjectCursor& c, char ch, std::string& scratch, std::string* out) const {
    c.number = 0;
    c.word_at = 0;
    c.words = 0;
    const auto words = Words(format_);
    for (std::size_t i = 0; i < words.size(); ++i)
      if (!words[i].empty()) c.words = static_cast<std::uint8_t>(c.words | (1U << i));
    if (out) scratch.clear();
    c.state = ToolObjectCursor::State::kScalar;
    return ScalarByte(c, ch, scratch, out);
  }
  bool ScalarByte(ToolObjectCursor& c, char ch, std::string& scratch, std::string* out) const {
    c.number = NumberStep(c.number, ch);
    const auto words = Words(format_);
    for (std::size_t i = 0; i < words.size(); ++i)
      if ((c.words & (1U << i)) != 0 && (c.word_at >= words[i].size() || words[i][c.word_at] != ch))
        c.words = static_cast<std::uint8_t>(c.words & ~(1U << i));
    if (c.word_at < std::numeric_limits<std::uint16_t>::max()) ++c.word_at;
    if (out) scratch += ch;
    return c.number != kNumberDead || c.words != 0;
  }
  bool FinishScalar(ToolObjectCursor& c, const std::string& scratch, std::string* out) const {
    const auto words = Words(format_);
    std::optional<std::size_t> word;
    for (std::size_t i = 0; i < words.size(); ++i)
      if ((c.words & (1U << i)) != 0 && words[i].size() == c.word_at) word = i;
    if (!word && !NumberComplete(c.number)) return false;
    if (out) *out += word == 3 ? std::string_view("null") : std::string_view(scratch);
    return true;
  }

  const ToolCallFormat& format_;
  std::size_t max_depth_;
  Keys* keys_;
};

// Converts a complete call's object (starting at its '{') to JSON.
std::expected<std::string, std::string> ObjectArguments(std::string_view text,
                                                        const ToolCallFormat& format,
                                                        const j::Limits& limits) {
  ObjectGrammar::Keys keys;
  const ObjectGrammar grammar(format, limits.max_depth, &keys);
  ToolObjectCursor cursor;
  std::string out, scratch;
  for (const char ch : text)
    if (!grammar.Step(cursor, ch, &out, scratch)) return Bad("invalid tool object notation");
  if (cursor.state != ToolObjectCursor::State::kDone) return Bad("invalid tool object notation");
  return out;
}

struct ParsedCall {
  std::string name, arguments;
};
std::expected<ParsedCall, std::string> ParseCall(std::string_view body,
                                                 const ToolCallFormat& format,
                                                 const ToolRequest& request,
                                                 const j::Limits& limits) {
  ParsedCall call;
  // The arguments object wraps each value: values nest one level less.
  auto value_limits = limits;
  value_limits.max_depth = limits.max_depth != 0 ? limits.max_depth - 1 : 0;
  if (format.arguments == ToolArguments::kJsonEnvelope) {
    const auto envelope = format.bare_envelope ? "{" + std::string(body) + "}" : std::string(body);
    auto doc = j::Parse(envelope, limits);
    if (!doc || !doc->root().is_object()) return Bad("tool call must contain a JSON object");
    const auto name = doc->root().find(format.name_key),
               args = doc->root().find(format.arguments_key);
    if (!name || !name->is_string() || !args)
      return Bad("tool call needs a name and argument object");
    call.name = name->string();
    if (args->is_object()) {
      call.arguments = Json(*args);
    } else if (args->is_string()) {
      // Arguments encoded as a JSON string, as some templates and vLLM accept.
      auto inner = j::Parse(args->string(), value_limits);
      if (!inner || !inner->root().is_object())
        return Bad("tool call needs a name and argument object");
      call.arguments = Json(inner->root());
    } else {
      return Bad("tool call needs a name and argument object");
    }
  } else {
    const auto end = body.find(format.name_end);
    if (end == std::string_view::npos) return Bad("tool call has no name boundary");
    call.name = Strip(body.substr(0, end));
    if (format.arguments == ToolArguments::kObjectNotation) {
      // name_end is the opening brace, consumed with the name.
      auto args = ObjectArguments(body.substr(end), format, limits);
      if (!args) return std::unexpected(args.error());
      call.arguments = std::move(*args);
    } else {
      body.remove_prefix(end + format.name_end.size());
      call.arguments = "{";
      bool first = true;
      std::unordered_set<std::string_view> keys;
      while (!(body = Strip(body)).empty()) {
        const auto header = Header(body, format);
        if (header.begin.empty()) return Bad("invalid tool parameter boundary");
        body.remove_prefix(header.begin.size());
        const auto name_end = body.find(header.name_end);
        if (name_end == std::string_view::npos) return Bad("tool parameter has no name boundary");
        const auto name = body.substr(0, name_end);
        if (name.empty()) return Bad("tool parameter name is empty");
        if (!keys.insert(name).second) return Bad("duplicate tool parameter");
        body.remove_prefix(name_end + header.name_end.size());
        std::optional<bool> string;
        if (!format.string_suffix.empty()) {
          if (body.starts_with(format.string_suffix)) {
            string = true;
            body.remove_prefix(format.string_suffix.size());
          } else if (body.starts_with(format.json_suffix)) {
            string = false;
            body.remove_prefix(format.json_suffix.size());
          } else {
            return Bad("tool parameter needs an encoding suffix");
          }
        }
        const auto schema = SchemaTypes(request.Parameter(call.name, name));
        ToolScan boundary;
        const auto found =
            Outside(body, format.parameter_end, "\"", boundary, format.parameter_begin, {},
                    StructuredValue(body, ParameterSyntax(schema, string)));
        const auto cut = found.value_or(body.size());
        const auto value = Unwrap(body.substr(0, cut), format.trim_wrapping_newline);
        if (!first) call.arguments += ',';
        first = false;
        j::AppendQuoted(name, call.arguments);
        call.arguments += ':';
        if (auto appended = AppendScalar(value, schema, string, value_limits, call.arguments);
            !appended)
          return std::unexpected(appended.error());
        body.remove_prefix(cut);
        if (body.starts_with(format.parameter_end)) body.remove_prefix(format.parameter_end.size());
      }
      call.arguments += '}';
    }
  }
  if (call.name.empty()) return Bad("tool name is empty");
  if (j::FirstInvalidUtf8(call.name)) return Bad("tool name is not UTF-8");
  if (j::FirstInvalidUtf8(call.arguments)) return Bad("tool arguments are not UTF-8");
  // Each piece was parsed or validated under the caller's limits; the
  // converted object as a whole keeps the caller's value bound too.
  if (j::MaxValues(call.arguments) > limits.max_values)
    return Bad("tool arguments exceed their value bound");
  return call;
}

// ---- The interpreter's one grammar -----------------------------------

//
// Advance runs the output grammar over buffered bytes from a cursor, for the
// output parser (events emitted, the buffer consumed after) and for probes
// (a copied cursor over the buffer plus candidate bytes, nothing emitted).
// Both run this code, so the sampler never refuses what the parser accepts,
// nor allows what it refuses while a call is open.
//
// One exception: a call that closes but cannot be converted (invalid JSON, a
// repeated parameter or key, a second call when parallel calls are disabled)
// is the parser's to refuse, not the probe's. Its body is fixed by then, so
// no continuation could repair it: masking the closing token would only let a
// constrained reply run on to its token allowance and end by length. The
// probe allows the closing bytes and stops there; the parser (and the
// sampler's own Accept) then refuses the request at once.
struct ProbeOut {
  using Result = std::expected<bool, std::string>;  // true: paused
  Result Text(JoinedText) { return false; }
  Result Call(ParsedCall&&) { return false; }
  Result ReasoningEnd() { return false; }
  Result ReasoningBegin() { return false; }
  Result Closed(std::string) { return true; }  // allowed: the parser's verdict
};
struct EmitOut {
  using Result = std::expected<bool, std::string>;
  const ToolInterpreter::Sink& sink;
  bool& paused;
  Result Emit(ToolEvent event) {
    if (const auto* text = std::get_if<TextPart>(&event); text && j::FirstInvalidUtf8(text->text))
      return Bad("generated text is not UTF-8");
    const auto flow = sink(std::move(event));
    if (flow.action == ToolInterpreter::Flow::kCancel) return Bad("tool output was cancelled");
    paused = flow.action == ToolInterpreter::Flow::kPause;
    return paused;
  }
  Result Text(JoinedText text) { return Emit(TextPart{static_cast<std::string>(text)}); }
  Result Call(ParsedCall&& call) {
    return Emit(ToolUsePart{
        .id = {}, .name = std::move(call.name), .input_json = std::move(call.arguments)});
  }
  Result ReasoningEnd() { return Emit(ToolReasoningEnd{}); }
  Result ReasoningBegin() { return Emit(ToolReasoningBegin{}); }
  Result Closed(std::string error) {
    return Bad(std::string(ToolInterpreter::kClosedToolCall) + error);
  }
};

std::expected<std::size_t, std::string> TextBytes(JoinedText text, bool final) {
  // Generated text is sanitized (StreamDecoder) before both the parser and the
  // sampler see it; a direct caller can still hold back a partial sequence.
  const auto bytes = text.contiguous() ? std::string() : static_cast<std::string>(text);
  const std::string_view view = text.contiguous() ? *text.contiguous() : std::string_view(bytes);
  const auto invalid = j::FirstInvalidUtf8(view);
  if (!invalid) return view.size();
  const auto tail = view.substr(*invalid);
  const auto lead = static_cast<unsigned char>(tail.front());
  const std::size_t need = lead >= 0xC2 && lead <= 0xDF   ? 2U
                           : lead >= 0xE0 && lead <= 0xEF ? 3U
                           : lead >= 0xF0 && lead <= 0xF4 ? 4U
                                                          : 0U;
  if (!final && need != 0 && tail.size() < need) return *invalid;
  return Bad("generated text is not UTF-8");
}

bool Constrained(const ToolRequest& request) {
  return request.choice().kind == ToolChoiceKind::kRequired ||
         request.choice().kind == ToolChoiceKind::kNamed;
}

class Grammar {
 public:
  Grammar(const ToolCallFormat& format, const ToolRequest& request, const j::Limits& limits)
      : format_(format), request_(request), limits_(limits) {}

  // true: an emitted event paused (stop now); false: everything available is
  // consumed or held for more input. `final` resolves holds and open calls
  // the way Finish does.
  template <class Out>
  std::expected<bool, std::string> Advance(ToolCursor& s, JoinedText all, bool final,
                                           Out& out) const {
    if (request_.choice().kind == ToolChoiceKind::kNone) {
      const auto rest = all.substr(s.start);
      if (rest.empty()) return false;
      auto bytes = TextBytes(rest, final);
      if (!bytes) return std::unexpected(bytes.error());
      if (*bytes == 0) return false;
      s.start += *bytes;
      return out.Text(rest.substr(0, *bytes));
    }
    for (;;) {
      if (s.in_call) {
        auto step = CallStep(s, all, final, out);
        if (!step) return std::unexpected(step.error());
        if (*step == Next::kPaused) return true;
        if (*step == Next::kMore) return false;
        continue;
      }
      auto step = TextStep(s, all, final, out);
      if (!step) return std::unexpected(step.error());
      if (*step == Next::kPaused) return true;
      if (*step == Next::kMore) return false;
    }
  }

 private:
  enum class Next { kAgain, kMore, kPaused };
  using Stepped = std::expected<Next, std::string>;
  static Stepped Emitted(std::expected<bool, std::string> result) {
    if (!result) return std::unexpected(result.error());
    return *result ? Next::kPaused : Next::kAgain;
  }

  static void ResetCall(ToolCursor& s) {
    s.scan = {};
    s.tag_phase = 0;
    s.tag_json = -1;
    s.tag_name_end = std::string::npos;
    s.tag_header_search = 0;
    s.envelope = {};
    s.object = {};
  }

  // A bare candidate that is not a call: its opener is answer text, and its
  // bytes are scanned again as text (no later opener is a candidate).
  template <class Out>
  Stepped Demote(ToolCursor& s, Out& out) const {
    s.in_call = false;
    s.content = true;
    s.started = true;
    ResetCall(s);
    return Emitted(out.Text(JoinedText(format_.call_begin)));
  }

  template <class Out>
  Stepped CallStep(ToolCursor& s, JoinedText all, bool final, Out& out) const {
    const auto body = all.substr(s.start);
    const bool bare_auto = format_.bare_envelope && request_.choice().kind == ToolChoiceKind::kAuto;
    if (Constrained(request_) && !format_.bare_envelope &&
        format_.arguments != ToolArguments::kJsonEnvelope) {
      const auto header = static_cast<std::string>(
          body.substr(0, std::min(body.size(), request_.NameBytes() + format_.name_end.size())));
      if (!request_.NamePrefix(header, format_.name_end))
        return Bad("generated tool name violates tool choice");
    }
    std::optional<TaggedBoundary> end;
    switch (format_.arguments) {
      case ToolArguments::kTaggedParameters: {
        auto found = TaggedEnd(body, format_, request_, s.scan, s.tag_phase, s.tag_json,
                               s.tag_name_end, s.tag_header_search);
        if (!found) return std::unexpected(found.error());
        end = *found;
        // A value left open at the end is closed by its first call-end alias.
        if (!end && final && s.tag_phase == 2 && s.scan.alias_at != std::string::npos)
          end = TaggedBoundary{s.scan.alias_at, s.scan.alias_bytes};
        break;
      }
      case ToolArguments::kObjectNotation: {
        auto found = ObjectEnd(body, s);
        if (!found) return std::unexpected(found.error());
        end = *found;
        break;
      }
      case ToolArguments::kJsonEnvelope: {
        if (const auto at = Outside(body, format_.call_end, format_.quote, s.scan))
          end = TaggedBoundary{*at, format_.call_end.size()};
        Envelope(body.substr(0, end ? end->at : s.scan.at), format_, request_, s.envelope);
        if (bare_auto && s.envelope.phase == ToolEnvelopeCursor::Phase::kStopped)
          return Demote(s, out);  // not a JSON object with string keys
        if (format_.bare_envelope && (!end || final)) {
          auto whole = s.envelope;
          const auto info = Envelope(body, format_, request_, whole);
          if (Constrained(request_)) {
            if (info.name && !info.allowed) return Bad("generated tool name violates tool choice");
            if (auto partial = OpenName(body, whole, s.envelope); !partial)
              return Bad("generated tool name violates tool choice");
          }
          if (!end && final && bare_auto && !(info.name && info.arguments) && !info.allowed)
            return Demote(s, out);
        }
        break;
      }
    }
    if (!end) return Next::kMore;
    const auto content = body.substr(0, end->at);
    if (bare_auto) {
      const auto info = s.envelope.projected;
      if (!info.name || !info.arguments) return Demote(s, out);
    }
    std::string copied;
    std::string_view view;
    if (const auto piece = content.contiguous()) {
      view = *piece;
    } else {
      copied = static_cast<std::string>(content);
      view = copied;
    }
    auto call = ParseCall(view, format_, request_, limits_);
    if (!call) {
      if (bare_auto) return Demote(s, out);
      return Emitted(out.Closed(std::move(call.error())));
    }
    const auto kind = request_.choice().kind;
    // Names are checked as they are written (a probe refuses a wrong prefix);
    // this is the closed call's own check.
    if ((kind == ToolChoiceKind::kRequired && !request_.Declared(call->name)) ||
        (kind == ToolChoiceKind::kNamed && call->name != request_.choice().name))
      return Bad("generated tool name violates tool choice");
    if (!request_.parallel() && s.calls != 0)
      return Emitted(out.Closed("parallel tool calls are disabled"));
    ++s.calls;
    s.start += end->at + end->bytes;
    s.in_call = false;
    s.started = true;
    ResetCall(s);
    return Emitted(out.Call(std::move(*call)));
  }

  // The name field of a constrained bare envelope, while it is open: its
  // string must stay a prefix of an allowed name, and once that string closes
  // it must be an allowed name with nothing but space after it. False: no
  // continuation can make it one. (Checked on every probe, whatever bytes
  // arrive together: a verdict never depends on how the output is chunked.)
  // `scanned` stops before a held prefix of the call's end, which is not junk.
  bool OpenName(JoinedText body, const ToolEnvelopeCursor& whole,
                const ToolEnvelopeCursor& scanned) const {
    if (whole.phase != ToolEnvelopeCursor::Phase::kValue || !whole.name_key ||
        whole.value_start >= body.size() || body[whole.value_start] != '"')
      return true;
    const std::size_t most = 6 * request_.NameBytes() + 6;
    std::size_t close = std::string::npos;
    bool escaped = false;
    for (std::size_t at = whole.value_start + 1; at < whole.at; ++at) {
      if (at - whole.value_start - 1 > most) return false;
      const char ch = body[at];
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        close = at;
        break;
      }
    }
    const auto end = close == std::string::npos ? whole.at : close;
    const auto raw = body.substr(whole.value_start + 1, end - whole.value_start - 1);
    if (raw.size() > most) return false;
    const auto decoded = JsonStringPrefix(static_cast<std::string>(raw));
    if (!decoded) return false;
    if (close == std::string::npos) return request_.NamePrefix(*decoded, {});
    // The last byte that is not space, in the field's scanned bytes.
    const bool same = scanned.phase == ToolEnvelopeCursor::Phase::kValue && scanned.name_key &&
                      scanned.value_start == whole.value_start;
    if (same && scanned.trim_end > close + 1) return false;  // bytes after the name string
    return request_.choice().kind == ToolChoiceKind::kNamed ? *decoded == request_.choice().name
                                                            : request_.Declared(*decoded);
  }

  // Object notation: the name up to name_end ('{'), then one grammar cursor.
  // call_end outside a string ends the call (a malformed body then fails to
  // parse rather than waiting for a length finish).
  std::expected<std::optional<TaggedBoundary>, std::string> ObjectEnd(JoinedText body,
                                                                      ToolCursor& s) const {
    if (s.tag_name_end == std::string::npos) {
      const auto name_at = body.find(format_.name_end, s.scan.at);
      const auto end_at = body.find(format_.call_end, s.scan.at);
      if (end_at != std::string_view::npos && end_at < name_at)
        return TaggedBoundary{end_at, format_.call_end.size()};
      if (name_at == std::string_view::npos) {
        const auto longest = std::max(format_.name_end.size(), format_.call_end.size());
        if (body.size() >= longest) s.scan.at = std::max(s.scan.at, body.size() - longest + 1);
        return std::nullopt;
      }
      s.tag_name_end = name_at;
      s.object = {};
    }
    const ObjectGrammar grammar(format_, limits_.max_depth);
    std::string unused;
    const auto object = body.substr(s.tag_name_end);
    while (s.object.at < object.size()) {
      const auto rest = object.substr(s.object.at);
      if (!ObjectGrammar::InString(s.object)) {
        if (rest.starts_with(format_.call_end))
          return TaggedBoundary{s.tag_name_end + s.object.at, format_.call_end.size()};
        if (rest.size() < format_.call_end.size() && PrefixOf(rest, format_.call_end))
          return std::nullopt;
      }
      if (!grammar.Step(s.object, rest.front(), nullptr, unused))
        return Bad("invalid tool object notation");
      ++s.object.at;
    }
    return std::nullopt;
  }

  template <class Out>
  Stepped TextStep(ToolCursor& s, JoinedText all, bool final, Out& out) const {
    const auto rest = all.substr(s.start);
    if (rest.empty()) return Next::kMore;
    const bool allow_openers = !s.reasoning_active || format_.opener_ends_reasoning;
    const bool opens_reasoning =
        !s.started && !s.reasoning_active && !s.reasoning_ended && !format_.reasoning_begin.empty();
    // Markers: 0 call, 1 block, 2 block end, 3 reasoning end, 4 reasoning begin.
    const std::array<std::string_view, 5> markers{
        allow_openers && !(format_.bare_envelope && s.content)
            ? std::string_view(format_.call_begin)
            : std::string_view{},
        allow_openers ? std::string_view(format_.block_begin) : std::string_view{},
        allow_openers && s.in_block ? std::string_view(format_.block_end) : std::string_view{},
        s.reasoning_active ? std::string_view(format_.reasoning_end) : std::string_view{},
        opens_reasoning ? std::string_view(format_.reasoning_begin) : std::string_view{}};
    std::size_t found = std::string::npos, marker = 0;
    for (std::size_t i = 0; i < markers.size(); ++i) {
      if (markers[i].empty()) continue;
      std::size_t at = std::string::npos;
      if (i == 4) {
        // Only the output's very start opens reasoning; wait while it may.
        if (rest.starts_with(markers[i]))
          at = 0;
        else if (!final && PrefixOf(rest, markers[i]))
          return Next::kMore;
      } else {
        at = rest.find(markers[i]);
        // A bare candidate opens only after whitespace since the last call
        // (or the answer's start).
        if (i == 0 && format_.bare_envelope && at != std::string::npos &&
            HasText(rest.substr(0, at)))
          at = std::string::npos;
      }
      if (at < found) {
        found = at;
        marker = i;
      }
    }
    if (found == std::string::npos) {
      std::size_t held = 0;
      // The reasoning opener counts only at the output's start, which waited
      // above: holding its prefix later would make text depend on chunking.
      if (!final)
        for (std::size_t i = 0; i < 4; ++i) {
          const auto token = markers[i];
          for (std::size_t n = 1; n < token.size() && n <= rest.size(); ++n)
            if (rest.substr(rest.size() - n).starts_with(token.substr(0, n)))
              held = std::max(held, n);
        }
      auto size = rest.size() - held;
      if (!s.in_block) {
        const auto bytes = TextBytes(rest.substr(0, size), final);
        if (!bytes) return std::unexpected(bytes.error());
        size = *bytes;
      } else if (HasText(rest.substr(0, size))) {
        return Bad("unrecognized text inside a tool calls block");
      }
      const auto text = rest.substr(0, size);
      s.start += size;
      if (s.in_block || size == 0) return Next::kMore;
      Answered(s, text);
      auto emitted = Emitted(out.Text(text));
      if (!emitted || *emitted == Next::kPaused) return emitted;
      return Next::kMore;
    }
    const auto prefix = rest.substr(0, found);
    if (s.in_block && HasText(prefix)) return Bad("unrecognized text inside a tool calls block");
    if (!s.in_block && !prefix.empty()) {
      if (auto bytes = TextBytes(prefix, true); !bytes) return std::unexpected(bytes.error());
      s.start += found;
      Answered(s, prefix);
      return Emitted(out.Text(prefix));
    }
    s.start += found + markers[marker].size();
    switch (marker) {
      case 3:
        s.reasoning_active = false;
        s.reasoning_ended = true;
        return Emitted(out.ReasoningEnd());
      case 4:
        s.reasoning_active = true;
        return Emitted(out.ReasoningBegin());
      case 2:
        if (s.calls == s.block_calls) return Bad("empty tool call block");
        s.in_block = false;
        return Next::kAgain;
      case 1:
        if (s.in_block) return Bad("nested tool call block");
        if (!request_.parallel() && s.calls != 0) return Bad("parallel tool calls are disabled");
        s.in_block = true;
        s.block_calls = s.calls;
        break;
      default:
        if (!format_.bare_envelope && !request_.parallel() && s.calls != 0)
          return Bad("parallel tool calls are disabled");
        s.in_call = true;
        ResetCall(s);
        break;
    }
    s.started = true;
    if (format_.opener_ends_reasoning && !s.reasoning_ended) {
      s.reasoning_ended = true;
      s.reasoning_active = false;
      return Emitted(out.ReasoningEnd());
    }
    return Next::kAgain;
  }

  static void Answered(ToolCursor& s, JoinedText text) {
    s.started = true;
    if (!s.reasoning_active && HasText(text)) s.content = true;
  }

  const ToolCallFormat& format_;
  const ToolRequest& request_;
  const j::Limits& limits_;
};
}  // namespace

std::optional<std::size_t> ToolOutputBytes(const ToolCallFormat& format, std::size_t open_bytes) {
  // The open bytes are buffered by the output parser and the sampler's
  // cursor, and by up to two speculative cursor copies (each string at twice
  // its size). When the call closes, one conversion holds its quoted
  // arguments and serialization temporaries (JSON quoting expands a byte to
  // at most six) and one JSON parse of at most the call's bytes.
  const auto escaped = OutputJsonBytes(open_bytes);
  std::size_t buffers = 0, strings = 0, total = 0, input = 0;
  if (!escaped || __builtin_mul_overflow(open_bytes, std::size_t{8}, &buffers) ||
      __builtin_mul_overflow(*escaped, std::size_t{4}, &strings) ||
      __builtin_add_overflow(open_bytes, std::size_t{2}, &input))
    return {};
  const auto parsed = j::ParseWorkingBytesBound(input);
  if (!parsed || __builtin_add_overflow(buffers, strings, &total) ||
      __builtin_add_overflow(total, *parsed, &total) ||
      __builtin_add_overflow(
          total, 4 * sizeof(ToolInterpreter) + sizeof(ParsedCall) + 2 * sizeof(j::Document),
          &total))
    return {};
  for (const auto* field :
       {&format.block_begin, &format.block_end, &format.call_begin, &format.call_end,
        &format.name_end, &format.name_key, &format.arguments_key, &format.parameter_begin,
        &format.parameter_name_end, &format.parameter_end, &format.string_suffix,
        &format.json_suffix, &format.quote, &format.null_word, &format.reasoning_end,
        &format.reasoning_begin}) {
    std::size_t owned = 0;
    if (__builtin_mul_overflow(field->size(), std::size_t{2}, &owned) ||
        __builtin_add_overflow(total, owned, &total))
      return {};
  }
  const auto add_string = [&](const std::string& field) {
    std::size_t owned = 0;
    return !__builtin_mul_overflow(field.size(), std::size_t{2}, &owned) &&
           !__builtin_add_overflow(total, owned, &total);
  };
  std::size_t entries = 0;
  if (__builtin_add_overflow(format.call_end_aliases.size(), format.continued_route_stops.size(),
                             &entries) ||
      __builtin_mul_overflow(entries, 2 * sizeof(std::string), &entries) ||
      __builtin_add_overflow(total, entries, &total) ||
      __builtin_mul_overflow(format.parameter_aliases.size(), 2 * sizeof(ToolParameterHeader),
                             &entries) ||
      __builtin_add_overflow(total, entries, &total))
    return {};
  for (const auto& alias : format.call_end_aliases)
    if (!add_string(alias)) return {};
  for (const auto& alias : format.parameter_aliases)
    if (!add_string(alias.begin) || !add_string(alias.name_end)) return {};
  for (const auto& marker : format.continued_route_stops)
    if (!add_string(marker)) return {};
  return total;
}
std::optional<std::size_t> ToolRequestBytes(std::span<const ToolDefinition> definitions) {
  std::size_t total = sizeof(ToolRequest) + sizeof(ToolRequest::Data) + 2 * sizeof(void*), peak = 0;
  for (const auto& d : definitions) {
    const auto escaped = DefinitionJsonBytes(d);
    if (!escaped) return {};
    std::size_t retained = 0;
    const auto parsed = j::ParseWorkingBytesBound(*escaped);
    const auto transient = ToolOutputBytes(ToolCallFormat{}, *escaped);
    // The name index: a node with its view and index per definition.
    constexpr std::size_t kIndexEntry = 4 * sizeof(void*) + sizeof(std::string_view);
    if (!parsed || !transient ||
        __builtin_add_overflow(*parsed, 2 * sizeof(ToolDefinition) + sizeof(j::Document),
                               &retained) ||
        __builtin_add_overflow(retained, 2 * d.name.size() + kIndexEntry, &retained) ||
        __builtin_add_overflow(total, retained, &total))
      return {};
    peak = std::max(peak, *transient);
  }
  if (__builtin_add_overflow(total, peak, &total)) return {};
  return total;
}

std::expected<ToolRequest, std::string> ToolRequest::Create(
    std::span<const ToolDefinition> definitions, ToolChoice choice, bool parallel,
    const j::Limits& limits) {
  ToolRequest out;
  auto data = std::make_shared<Data>();
  data->choice = std::move(choice);
  data->parallel = parallel;
  data->definitions.reserve(definitions.size());
  std::unordered_set<std::string_view> names;
  names.reserve(definitions.size());
  std::size_t bytes = 0;
  for (const auto& definition : definitions) {
    if (definition.name.empty()) return Bad("tool name is empty");
    if (!names.insert(definition.name).second) return Bad("duplicate tool name");
    const auto add = [&](std::size_t size) {
      if (size > limits.max_bytes - bytes) return false;
      bytes += size;
      return true;
    };
    if (!add(definition.name.size()) || !add(definition.description.size()) ||
        !add(definition.parameters_json.size()) || !add(definition.function_json.size()))
      return Bad("tool definitions exceed their byte budget");
    auto parameters = j::Parse(definition.parameters_json, limits);
    if (!parameters || !parameters->root().is_object())
      return Bad("tool parameters must be a schema object");
    std::string text = "{\"type\":\"function\",\"function\":";
    if (!definition.function_json.empty()) {
      // The client's own function object: exactly the fields it sent.
      auto function = j::Parse(definition.function_json, limits);
      if (!function || !function->root().is_object()) return Bad("tool function must be an object");
      const auto name = function->root().find("name");
      if (!name || !name->is_string() || name->string() != definition.name)
        return Bad("tool function name differs from its definition");
      j::AppendValue(function->root(), text);
    } else {
      text += "{\"name\":";
      j::AppendQuoted(definition.name, text);
      text += ",\"description\":";
      j::AppendQuoted(definition.description, text);
      text += ",\"parameters\":";
      j::AppendValue(parameters->root(), text);
      text += '}';
    }
    text += '}';
    const auto expanded = DefinitionJsonBytes(definition);
    if (!expanded) return Bad("normalized tool definition exceeds addressable bytes");
    auto normalized_limits = limits;
    normalized_limits.max_bytes = *expanded;
    auto document = j::Parse(text, normalized_limits);
    if (!document) return Bad("normalized tool definition exceeds its JSON budget");
    data->definitions.push_back(
        {definition.name, std::make_shared<const j::Document>(std::move(*document)), text.size()});
  }
  data->index.reserve(data->definitions.size());
  for (std::size_t i = 0; i < data->definitions.size(); ++i) {
    data->index.emplace(data->definitions[i].name, i);
    data->name_bytes = std::max(data->name_bytes, data->definitions[i].name.size());
  }
  out.data_ = data;
  if ((data->choice.kind == ToolChoiceKind::kRequired ||
       data->choice.kind == ToolChoiceKind::kNamed) &&
      data->definitions.empty())
    return Bad("required tool choice needs declared tools");
  if (data->choice.kind == ToolChoiceKind::kNamed && !out.Declared(data->choice.name))
    return Bad("named tool choice must name a declared tool");
  if (data->choice.kind != ToolChoiceKind::kNamed && !data->choice.name.empty())
    return Bad("only named tool choice carries a name");
  return out;
}
const ToolRequest::Definition* ToolRequest::Find(std::string_view name) const {
  const auto found = data_->index.find(name);
  return found == data_->index.end() ? nullptr : &data_->definitions[found->second];
}
bool ToolRequest::Declared(std::string_view name) const { return Find(name) != nullptr; }
std::size_t ToolRequest::TemplateBytes() const {
  if (choice().kind == ToolChoiceKind::kNone) return 0;
  std::size_t bytes = 0;
  for (const auto& d : data_->definitions)
    if (choice().kind != ToolChoiceKind::kNamed || d.name == choice().name) bytes += d.bytes;
  return bytes;
}
std::size_t ToolRequest::NameBytes() const {
  return std::max(choice().name.size(), data_->name_bytes);
}
std::vector<std::string_view> ToolRequest::ExposedNames() const {
  std::vector<std::string_view> names;
  if (choice().kind == ToolChoiceKind::kNone) return names;
  names.reserve(data_->definitions.size());
  for (const auto& d : data_->definitions)
    if (choice().kind != ToolChoiceKind::kNamed || d.name == choice().name) names.push_back(d.name);
  return names;
}
ToolStatus ToolRequest::ValidateNames(const ToolCallFormat& format) const {
  if (choice().kind == ToolChoiceKind::kNone || format.arguments == ToolArguments::kJsonEnvelope)
    return {};
  for (const auto name : ExposedNames()) {
    const JoinedText header(name, format.name_end);
    if (Strip(name) != name || format.name_end.empty() ||
        header.find(format.name_end) != name.size())
      return Bad("tool name cannot round-trip through this template's name boundary");
    if (format.arguments == ToolArguments::kObjectNotation) {
      // Object notation scans its raw name as part of the call: a call end
      // inside the name would close the call early.
      ToolScan scan;
      const auto end = Outside(header, format.name_end, format.quote, scan, format.call_end);
      if (!end || *end != name.size())
        return Bad("tool name cannot round-trip through this template's call boundary");
    }
  }
  return {};
}
bool ToolRequest::NamePrefix(std::string_view prefix, std::string_view terminator) const {
  const auto allowed = [&](std::string_view name) {
    const auto first = prefix.substr(0, std::min(prefix.size(), name.size()));
    if (!name.starts_with(first)) return false;
    if (prefix.size() <= name.size()) return true;
    const auto rest = prefix.substr(name.size());
    return !terminator.empty() && (rest.starts_with(terminator) || terminator.starts_with(rest));
  };
  if (choice().kind == ToolChoiceKind::kNamed) return allowed(choice().name);
  if (choice().kind == ToolChoiceKind::kRequired)
    return std::ranges::any_of(data_->definitions, [&](const auto& d) { return allowed(d.name); });
  return true;
}
std::optional<j::Value> ToolRequest::Parameter(std::string_view name, std::string_view key) const {
  const auto* d = Find(name);
  if (d == nullptr) return {};
  const auto function = d->owner->root().find("function");
  const auto parameters = function ? function->find("parameters") : std::nullopt;
  const auto properties = parameters ? parameters->find("properties") : std::nullopt;
  return properties ? properties->find(key) : std::nullopt;
}
std::vector<j::Value> ToolRequest::TemplateTools() const {
  std::vector<j::Value> out;
  if (data_->choice.kind == ToolChoiceKind::kNone) return out;
  for (const auto& d : data_->definitions)
    if (data_->choice.kind != ToolChoiceKind::kNamed || data_->choice.name == d.name)
      out.push_back(d.owner->root());
  return out;
}

const ToolCallFormat& QwenToolCalls() {
  static const ToolCallFormat format{.block_begin = "<tool_call>",
                                     .block_end = "</tool_call>",
                                     .call_begin = "<function=",
                                     .call_end = "</function>",
                                     .name_end = ">",
                                     .arguments = ToolArguments::kTaggedParameters,
                                     .parameter_begin = "<parameter=",
                                     .parameter_name_end = ">",
                                     .parameter_end = "</parameter>",
                                     .trim_wrapping_newline = true,
                                     .opener_ends_reasoning = true,
                                     .reasoning_end = "</think>",
                                     .reasoning_begin = "<think>",
                                     .call_end_aliases = {"</invoke>"},
                                     .parameter_aliases = {{"<parameter name=\"", "\">"}}};
  return format;
}
const ToolCallFormat& DsmlToolCalls() {
  static const ToolCallFormat format{.block_begin = "<｜DSML｜tool_calls>",
                                     .block_end = "</｜DSML｜tool_calls>",
                                     .call_begin = "<｜DSML｜invoke name=\"",
                                     .call_end = "</｜DSML｜invoke>",
                                     .name_end = "\">",
                                     .arguments = ToolArguments::kTaggedParameters,
                                     .parameter_begin = "<｜DSML｜parameter name=\"",
                                     .parameter_name_end = "\"",
                                     .parameter_end = "</｜DSML｜parameter>",
                                     .string_suffix = " string=\"true\">",
                                     .json_suffix = " string=\"false\">",
                                     .reasoning_end = "</think>",
                                     .reasoning_begin = "<think>"};
  return format;
}
const ToolCallFormat& GemmaToolCalls() {
  static const ToolCallFormat format{.call_begin = "<|tool_call>call:",
                                     .call_end = "<tool_call|>",
                                     .name_end = "{",
                                     .arguments = ToolArguments::kObjectNotation,
                                     .reasoning_end = "<channel|>",
                                     .reasoning_begin = "<|channel>thought\n",
                                     .quote = "<|\"|>",
                                     .null_word = "None",
                                     .continued_route_stops = {"<|tool_call>"}};
  return format;
}
const ToolCallFormat& JsonToolCalls() {
  static const ToolCallFormat format{.call_begin = "{",
                                     .call_end = "}",
                                     .arguments_key = "parameters",
                                     .arguments = ToolArguments::kJsonEnvelope,
                                     .bare_envelope = true};
  return format;
}
const ToolCallFormat* ToolCallsForTemplate(std::string_view sha256) {
  // nvidia/Llama-3.3-70B-Instruct-NVFP4@f792b707002f2c613e0729c6b37bbcedc4efa94e,
  // tokenizer_config's actual template. Grammar binding is independent of the
  // renderer implementation; modified/unverified templates do not inherit it.
  if (sha256 == "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65")
    return &JsonToolCalls();
  return nullptr;
}
ToolTrigger ToolTriggerFor(const ToolCallFormat& format, const ToolRequest& request) {
  return {
      .begin = format.block_begin.empty() ? format.call_begin : format.block_begin,
      .bound = format.reasoning_end.empty() ? std::nullopt : std::optional(format.reasoning_end),
      .required = Constrained(request),
      .name = request.choice().kind == ToolChoiceKind::kNamed ? std::optional(request.choice().name)
                                                              : std::nullopt,
      .ends_reasoning = format.opener_ends_reasoning};
}

ToolInterpreter::ToolInterpreter(ToolCallFormat format, ToolRequest request, j::Limits limits)
    : format_(std::move(format)), request_(std::move(request)), limits_(limits) {
  if (!ToolOutputBytes(format_, limits_.max_bytes))
    error_ = "tool output budget exceeds addressable bytes";
  if (format_.call_begin.empty() || format_.call_end.empty() || format_.quote.empty() ||
      format_.quote.size() > 64 || format_.null_word.size() > 64 ||
      format_.string_suffix.empty() != format_.json_suffix.empty() ||
      (format_.arguments == ToolArguments::kObjectNotation && format_.name_end != "{") ||
      (format_.arguments != ToolArguments::kJsonEnvelope && format_.name_end.empty()) ||
      (format_.arguments == ToolArguments::kTaggedParameters &&
       (format_.parameter_begin.empty() || format_.parameter_name_end.empty() ||
        format_.parameter_end.empty())))
    error_ = "incomplete tool format declaration";
  const auto overlaps = [](std::string_view a, std::string_view b) {
    return a.starts_with(b) || b.starts_with(a);
  };
  if ((!format_.call_end_aliases.empty() || !format_.parameter_aliases.empty()) &&
      format_.arguments != ToolArguments::kTaggedParameters)
    error_ = "tagged aliases require tagged parameters";
  for (std::size_t i = 0; i < format_.call_end_aliases.size(); ++i) {
    const auto& alias = format_.call_end_aliases[i];
    if (alias.empty() || overlaps(alias, format_.call_end) ||
        overlaps(alias, format_.parameter_begin) || overlaps(alias, format_.parameter_end))
      error_ = "ambiguous tagged boundary alias";
    for (std::size_t previous = 0; previous < i; ++previous)
      if (overlaps(alias, format_.call_end_aliases[previous]))
        error_ = "ambiguous tagged boundary alias";
    for (const auto& header : format_.parameter_aliases)
      if (overlaps(alias, header.begin)) error_ = "ambiguous tagged boundary alias";
  }
  for (std::size_t i = 0; i < format_.parameter_aliases.size(); ++i) {
    const auto& alias = format_.parameter_aliases[i];
    if (alias.begin.empty() || alias.name_end.empty() ||
        overlaps(alias.begin, format_.parameter_begin) || overlaps(alias.begin, format_.call_end))
      error_ = "ambiguous tagged boundary alias";
    for (std::size_t previous = 0; previous < i; ++previous)
      if (overlaps(alias.begin, format_.parameter_aliases[previous].begin))
        error_ = "ambiguous tagged boundary alias";
  }
  for (const auto& marker : format_.continued_route_stops)
    if (marker.empty()) error_ = "empty continued route stop";
}

bool ToolInterpreter::CanStop() const {
  if (error_ || ended_) return false;
  // Exactly Finish(kStop)'s decision, on a copied cursor.
  auto cursor = cursor_;
  ProbeOut out;
  if (!Grammar(format_, request_, limits_).Advance(cursor, JoinedText(pending_), true, out))
    return false;
  if (cursor.in_call || (cursor.in_block && cursor.block_calls == cursor.calls)) return false;
  return cursor.calls != 0 || !Constrained(request_);
}
bool ToolInterpreter::Allows(std::string_view candidate) const {
  if (error_ || ended_ || candidate.size() > limits_.max_bytes - received_) return false;
  auto cursor = cursor_;
  ProbeOut out;
  return Grammar(format_, request_, limits_)
      .Advance(cursor, JoinedText(pending_, candidate), false, out)
      .has_value();
}

ToolStatus ToolInterpreter::Fail(std::string error) {
  error_ = std::move(error);
  return std::unexpected(*error_);
}
ToolStatus ToolInterpreter::Push(std::string_view generated, const Sink& sink) {
  if (busy_) return Fail("tool output callbacks cannot reenter the interpreter");
  struct Active {
    bool& busy;
    explicit Active(bool& b) : busy(b) { busy = true; }
    ~Active() { busy = false; }
  } active(busy_);
  if (error_) return std::unexpected(*error_);
  if (ended_) return Fail("tool output already ended");
  if (paused_) return Fail("resume paused tool output before appending bytes");
  if (generated.size() > limits_.max_bytes - received_)
    return Fail("generated output exceeds its byte budget");
  received_ += generated.size();
  pending_ += generated;
  return Drain(false, sink);
}

ToolStatus ToolInterpreter::Drain(bool final, const Sink& sink) {
  EmitOut out{.sink = sink, .paused = paused_};
  const auto advanced =
      Grammar(format_, request_, limits_).Advance(cursor_, JoinedText(pending_), final, out);
  pending_.erase(0, cursor_.start);
  cursor_.start = 0;
  if (error_) return std::unexpected(*error_);  // a callback reentered
  if (!advanced) return Fail(advanced.error());
  return {};
}

ToolStatus ToolInterpreter::Resume(const Sink& sink) {
  if (busy_) return Fail("tool output callbacks cannot reenter the interpreter");
  if (error_) return std::unexpected(*error_);
  paused_ = false;
  if (finishing_) return Finish(*finishing_, sink);
  busy_ = true;
  const auto result = Drain(false, sink);
  busy_ = false;
  return result;
}

ToolStatus ToolInterpreter::Finish(ToolFinish reason, const Sink& sink) {
  if (busy_) return Fail("tool output callbacks cannot reenter the interpreter");
  struct Active {
    bool& busy;
    explicit Active(bool& b) : busy(b) { busy = true; }
    ~Active() { busy = false; }
  } active(busy_);
  if (error_) return std::unexpected(*error_);
  if (ended_) return Fail("tool output already ended");
  finishing_ = reason;
  if (paused_) return {};
  if (auto drained = Drain(true, sink); !drained) return drained;
  if (paused_) return {};
  ended_ = true;
  if (cursor_.in_call || (cursor_.in_block && cursor_.block_calls == cursor_.calls)) {
    if (reason != ToolFinish::kLength) return Fail("incomplete tool call at natural completion");
    pending_.clear();
  }
  // An output-limit end is the request's own token allowance: a required call
  // it cut short ends the reply by length (as a forced span the allowance
  // cuts does), not as an invalid output. A natural end needs the call.
  if (Constrained(request_) && cursor_.calls == 0 && reason == ToolFinish::kStop)
    return Fail("generation did not complete a required tool call");
  return {};
}

std::expected<std::vector<ToolResultPart>, std::string> OrderToolResults(
    std::span<const ToolUsePart> calls, std::span<const ToolResultPart> results) {
  std::vector<ToolResultPart> out(results.begin(), results.end());
  std::unordered_map<std::string_view, std::size_t> call_index;
  call_index.reserve(calls.size());
  bool missing = false;
  for (std::size_t i = 0; i < calls.size(); ++i) {
    if (calls[i].id.empty()) {
      missing = true;
      continue;
    }
    if (!call_index.emplace(calls[i].id, i).second) return Bad("duplicate tool call ID");
  }
  std::unordered_set<std::string_view> seen;
  seen.reserve(results.size());
  std::vector<std::size_t> order;
  order.reserve(results.size());
  for (const auto& result : results) {
    if (result.tool_use_id.empty()) {
      missing = true;
      continue;
    }
    if (!seen.insert(result.tool_use_id).second) return Bad("duplicate tool result ID");
    const auto found = call_index.find(result.tool_use_id);
    if (found == call_index.end()) return Bad("unknown tool result ID");
    order.push_back(found->second);
  }
  if (missing) return out;
  std::vector<std::size_t> indices(results.size());
  for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
  std::ranges::sort(indices, {}, [&](std::size_t i) { return order[i]; });
  for (std::size_t i = 0; i < indices.size(); ++i) out[i] = results[indices[i]];
  return out;
}
}  // namespace llmp::chat
