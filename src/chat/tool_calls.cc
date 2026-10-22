// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "chat/tool_calls.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <utility>

namespace llmp::chat {
namespace {
namespace j = base::json;

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

std::expected<std::size_t, std::string> TextBytes(std::string_view text, bool final) {
  const auto invalid = j::FirstInvalidUtf8(text);
  if (!invalid) return text.size();
  const auto tail = text.substr(*invalid);
  const auto lead = static_cast<unsigned char>(tail.front());
  const std::size_t need = lead >= 0xC2 && lead <= 0xDF   ? 2U
                           : lead >= 0xE0 && lead <= 0xEF ? 3U
                           : lead >= 0xF0 && lead <= 0xF4 ? 4U
                                                          : 0U;
  if (!final && need != 0 && tail.size() < need) return *invalid;
  return Bad("generated text is not UTF-8");
}

// RFC8259 serialization preserves exact number text; Python's presentation
// serializer would turn overflowing finite JSON literals into Infinity.
void Json(j::Value value, std::string& out) {
  switch (value.kind()) {
    case j::Kind::kNull:
      out += "null";
      break;
    case j::Kind::kFalse:
      out += "false";
      break;
    case j::Kind::kTrue:
      out += "true";
      break;
    case j::Kind::kNumber:
      out += value.number();
      break;
    case j::Kind::kString:
      j::AppendQuoted(value.string(), out);
      break;
    case j::Kind::kArray:
      out += '[';
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i) out += ',';
        Json(value.at(i), out);
      }
      out += ']';
      break;
    case j::Kind::kObject:
      out += '{';
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i) out += ',';
        j::AppendQuoted(value.key(i), out);
        out += ':';
        Json(value.member(i), out);
      }
      out += '}';
      break;
  }
}
std::string Json(j::Value value) {
  std::string out;
  Json(value, out);
  return out;
}

bool Type(j::Value value, std::string_view type) {
  return type == "string"    ? value.is_string()
         : type == "integer" ? value.is_integer()
         : type == "number"  ? value.is_number()
         : type == "boolean" ? value.is_bool()
         : type == "object"  ? value.is_object()
         : type == "array"   ? value.is_array()
                             : type == "null" && value.is_null();
}
bool SchemaAccepts(j::Value schema, j::Value value) {
  const auto type = schema.find("type");
  if (!type) return false;
  if (type->is_string()) return Type(value, type->string());
  if (type->is_array())
    for (std::size_t i = 0; i < type->size(); ++i)
      if (type->at(i).is_string() && Type(value, type->at(i).string())) return true;
  return false;
}
std::string Scalar(std::string_view value, std::optional<j::Value> schema,
                   std::optional<bool> string, const j::Limits& limits) {
  auto parsed = j::Parse(value, limits);
  if (parsed && ((!string && schema && !parsed->root().is_string() &&
                  SchemaAccepts(*schema, parsed->root())) ||
                 (string && !*string)))
    return Json(parsed->root());
  std::string out;
  j::AppendQuoted(value, out);
  return out;
}

// JSON/object syntax keeps delimiter-looking text inside strings and nested
// containers opaque. Tagged raw string values instead obey their explicit
// parameter boundaries; a schema or string=false can select JSON syntax.
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

struct DelimiterScan {
  std::size_t at = 0;
  std::vector<char> nesting;
  bool quoted = false, escaped = false;
};
template <class Text>
std::optional<std::size_t> Outside(Text text, std::string_view needle, std::string_view quote,
                                   DelimiterScan& scan, std::string_view second = {},
                                   std::string_view third = {}, bool structured = true) {
  while (scan.at < text.size()) {
    const auto rest = text.substr(scan.at);
    const auto boundary = [&](std::string_view token) {
      if (token.empty() || !rest.starts_with(token)) return false;
      // An XML-like boundary outside a quoted string closes a malformed JSON
      // value too: unclosed braces must not turn a fully closed bad call into
      // a length-truncated call that would silently disappear.
      return scan.nesting.empty() ||
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
      const char c = rest.front();
      if (structured && (c == '{' || c == '[')) scan.nesting.push_back(c);
      if ((c == '}' || c == ']') && !scan.nesting.empty()) scan.nesting.pop_back();
    }
    ++scan.at;
  }
  return {};
}

using EnvelopeInfo = detail::ToolEnvelopeInfo;
using EnvelopeCursor = detail::ToolEnvelopeCursor;

// Advance only through newly appended bytes. Projections of the current field
// remain provisional until a top-level comma; this preserves the old lexical
// treatment of malformed values and duplicate names. Only bounded key/name
// slices are materialized, never argument payloads.
template <class Text>
EnvelopeInfo Envelope(Text text, const ToolCallFormat& format, const ToolRequest& request,
                      EnvelopeCursor& scan) {
  using Phase = EnvelopeCursor::Phase;
  const auto project = [&](bool ended) {
    auto info = scan.committed;
    if (scan.arguments_key) info.arguments = true;
    if (scan.name_key) {
      info.name = true;
      const auto encoded = text.substr(scan.value_start, scan.trim_end - scan.value_start);
      if (!encoded.empty() && encoded.front() == '"' && encoded.back() == '"' &&
          encoded.size() <= 6 * request.NameBytes() + 2) {
        const auto bytes = static_cast<std::string>(encoded);
        auto parsed = j::Parse(bytes, {.max_bytes = bytes.size(),
                                       .max_depth = bytes.size(),
                                       .max_values = bytes.size(),
                                       .max_string_bytes = bytes.size()});
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
            auto parsed = j::Parse(bytes, {.max_bytes = bytes.size(),
                                           .max_depth = bytes.size(),
                                           .max_values = bytes.size(),
                                           .max_string_bytes = bytes.size()});
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

// Implicit conversion only recognizes schema-admitted non-string values.
// Keep the lexical decision identical in streaming and final parsing: unknown
// schemas and non-container types cannot make raw braces/quotes hide tags.
int ParameterSyntax(std::optional<j::Value> schema, std::optional<bool> string) {
  if (string) return *string ? 0 : 1;  // raw text, or explicitly encoded JSON
  const auto type = schema ? schema->find("type") : std::nullopt;
  if (!type) return 0;
  const auto flag = [](std::string_view name) {
    return name == "object" ? 2 : name == "array" ? 4 : 0;
  };
  if (type->is_string()) return flag(type->string());
  int syntax = 0;
  if (type->is_array())
    for (std::size_t i = 0; i < type->size(); ++i)
      if (type->at(i).is_string()) syntax |= flag(type->at(i).string());
  return syntax;
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
std::optional<TaggedBoundary> FindCallEnd(Text body, std::size_t from,
                                          const ToolCallFormat& format) {
  const auto canonical = body.find(format.call_end, from);
  std::optional<TaggedBoundary> found;
  if (canonical != std::string_view::npos)
    found = TaggedBoundary{canonical, format.call_end.size()};
  for (const auto& alias : format.call_end_aliases) {
    const auto at = body.find(alias, from);
    if (at != std::string_view::npos && (!found || at < found->at))
      found = TaggedBoundary{at, alias.size()};
  }
  return found;
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
template <class Text>
std::optional<TaggedBoundary> TaggedEnd(Text body, const ToolCallFormat& format,
                                        const ToolRequest& request, DelimiterScan& scan,
                                        unsigned& phase, int& json, std::size_t& call_name_end,
                                        std::size_t& header_search) {
  if (phase == 0) {
    call_name_end = body.find(format.name_end, scan.at);
    if (call_name_end == std::string_view::npos) {
      scan.at =
          body.size() >= format.name_end.size() ? body.size() - format.name_end.size() + 1 : 0;
      return {};
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
      if (PartialTaggedBoundary(rest, format)) return {};
      const auto header = Header(rest, format);
      if (header.begin.empty()) return FindCallEnd(body, scan.at, format);
      const auto name_at = scan.at + header.begin.size();
      const auto end = body.find(header.name_end, std::max(name_at, header_search));
      if (end == std::string_view::npos) {
        header_search = body.size() >= header.name_end.size()
                            ? body.size() - header.name_end.size() + 1
                            : name_at;
        return {};
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
        } else if (PrefixOf(suffix, format.string_suffix) || PrefixOf(suffix, format.json_suffix))
          return {};
        else
          return FindCallEnd(body, value_at, format);
      }
      json = ParameterSyntax(
          request.Parameter(static_cast<std::string>(call_name), static_cast<std::string>(name)),
          string);
      scan.at = value_at;
      header_search = 0;
      phase = 3;
    }
    if (phase == 3) {
      while (scan.at < body.size() && Space(body[scan.at])) ++scan.at;
      if (scan.at == body.size()) return {};
      json = StructuredValue(body.substr(scan.at), json) ? 1 : 0;
      phase = 2;
    }
    const auto end = Outside(body, format.call_end, "\"", scan, format.parameter_begin,
                             format.parameter_end, json == 1);
    if (!end) return {};
    if (body.substr(*end).starts_with(format.call_end))
      return TaggedBoundary{*end, format.call_end.size()};
    if (body.substr(*end).starts_with(format.parameter_end)) scan.at += format.parameter_end.size();
    phase = 1;
  }
}

// Converts the declarative object notation to JSON. Its quotes and null spelling
// are format data; recursion is bounded by the same JSON stack limit.
class ObjectReader {
 public:
  ObjectReader(std::string_view text, const ToolCallFormat& format, const j::Limits& limits)
      : text_(text), format_(format), limits_(limits) {}
  std::expected<std::string, std::string> Read() {
    if (!Value(0)) return Bad("invalid tool object notation");
    White();
    if (at_ != text_.size()) return Bad("trailing tool argument text");
    return std::move(out_);
  }

 private:
  void White() {
    while (at_ < text_.size() && Space(text_[at_])) ++at_;
  }
  bool String() {
    const auto first = at_;
    at_ += format_.quote.size();
    if (format_.quote == "\"") {
      bool escaped = false;
      for (; at_ < text_.size(); ++at_) {
        const char ch = text_[at_];
        if (!escaped && ch == '"') {
          ++at_;
          const auto json = j::Parse(text_.substr(first, at_ - first), limits_);
          if (!json || !json->root().is_string()) return false;
          j::AppendQuoted(json->root().string(), out_);
          return true;
        }
        escaped = !escaped && ch == '\\';
      }
      return false;
    }
    const auto end = text_.find(format_.quote, at_);
    if (end == std::string_view::npos) return false;
    j::AppendQuoted(text_.substr(at_, end - at_), out_);
    at_ = end + format_.quote.size();
    return true;
  }
  bool Value(std::size_t depth) {
    if (depth > limits_.max_depth) return false;
    White();
    if (at_ == text_.size()) return false;
    if (text_.substr(at_).starts_with(format_.quote)) return String();
    const char begin = text_[at_];
    if (begin == '{' || begin == '[') {
      const char end = begin == '{' ? '}' : ']';
      out_ += begin;
      ++at_;
      White();
      if (at_ < text_.size() && text_[at_] == end) {
        ++at_;
        out_ += end;
        return true;
      }
      for (;;) {
        White();
        if (begin == '{') {
          if (text_.substr(at_).starts_with(format_.quote)) {
            if (!String()) return false;
          } else {
            const auto first = at_;
            while (at_ < text_.size() && text_[at_] != ':' && text_[at_] != ',' &&
                   text_[at_] != '}' && text_[at_] != '{')
              ++at_;
            const auto key = Strip(text_.substr(first, at_ - first));
            if (key.empty()) return false;
            j::AppendQuoted(key, out_);
          }
          White();
          if (at_ == text_.size() || text_[at_++] != ':') return false;
          out_ += ':';
        }
        if (!Value(depth + 1)) return false;
        White();
        if (at_ == text_.size()) return false;
        const char separator = text_[at_++];
        out_ += separator;
        if (separator == end) return true;
        if (separator != ',') return false;
      }
    }
    const auto first = at_;
    while (at_ < text_.size() && text_[at_] != ',' && text_[at_] != '}' && text_[at_] != ']') ++at_;
    const auto scalar = Strip(text_.substr(first, at_ - first));
    if (scalar == format_.null_word)
      out_ += "null";
    else
      out_ += scalar;
    return !scalar.empty();
  }
  std::string_view text_;
  const ToolCallFormat& format_;
  const j::Limits& limits_;
  std::size_t at_ = 0;
  std::string out_;
};

struct ParsedCall {
  std::string name, arguments;
};
std::expected<ParsedCall, std::string> ParseCall(std::string_view body,
                                                 const ToolCallFormat& format,
                                                 const ToolRequest& request,
                                                 const j::Limits& limits) {
  ParsedCall call;
  if (format.arguments == ToolArguments::kJsonEnvelope) {
    const auto envelope = format.bare_envelope ? "{" + std::string(body) + "}" : std::string(body);
    auto doc = j::Parse(envelope, limits);
    if (!doc || !doc->root().is_object()) return Bad("tool call must contain a JSON object");
    const auto name = doc->root().find(format.name_key),
               args = doc->root().find(format.arguments_key);
    if (!name || !name->is_string() || !args || !args->is_object())
      return Bad("tool call needs a name and argument object");
    call.name = name->string();
    call.arguments = Json(*args);
  } else {
    const auto end = body.find(format.name_end);
    if (end == std::string_view::npos) return Bad("tool call has no name boundary");
    call.name = Strip(body.substr(0, end));
    body.remove_prefix(end + format.name_end.size());
    if (format.arguments == ToolArguments::kObjectNotation) {
      // name_end is the opening brace, consumed with the name.
      auto args = ObjectReader(std::string("{") + std::string(body), format, limits).Read();
      if (!args) return std::unexpected(args.error());
      call.arguments = std::move(*args);
    } else {
      call.arguments = "{";
      bool first = true;
      while (!(body = Strip(body)).empty()) {
        const auto header = Header(body, format);
        if (header.begin.empty()) return Bad("invalid tool parameter boundary");
        body.remove_prefix(header.begin.size());
        const auto name_end = body.find(header.name_end);
        if (name_end == std::string_view::npos) return Bad("tool parameter has no name boundary");
        const auto name = body.substr(0, name_end);
        if (name.empty()) return Bad("tool parameter name is empty");
        body.remove_prefix(name_end + header.name_end.size());
        std::optional<bool> string;
        if (!format.string_suffix.empty()) {
          if (body.starts_with(format.string_suffix)) {
            string = true;
            body.remove_prefix(format.string_suffix.size());
          } else if (body.starts_with(format.json_suffix)) {
            string = false;
            body.remove_prefix(format.json_suffix.size());
          } else
            return Bad("tool parameter needs an encoding suffix");
        }
        const auto schema = request.Parameter(call.name, name);
        DelimiterScan boundary;
        const auto found =
            Outside(body, format.parameter_end, "\"", boundary, format.parameter_begin, {},
                    StructuredValue(body, ParameterSyntax(schema, string)));
        const auto cut = found.value_or(body.size());
        const auto value = Unwrap(body.substr(0, cut), format.trim_wrapping_newline);
        if (!first) call.arguments += ',';
        first = false;
        j::AppendQuoted(name, call.arguments);
        call.arguments += ':';
        if (string && !*string) {
          auto json = j::Parse(value, limits);
          if (!json) return Bad("tool parameter contains invalid JSON");
        }
        call.arguments += Scalar(value, schema, string, limits);
        body.remove_prefix(cut);
        if (body.starts_with(format.parameter_end)) body.remove_prefix(format.parameter_end.size());
      }
      call.arguments += '}';
    }
  }
  if (call.name.empty()) return Bad("tool name is empty");
  if (j::FirstInvalidUtf8(call.name)) return Bad("tool name is not UTF-8");
  const auto expanded = OutputJsonBytes(limits.max_bytes);
  if (!expanded) return Bad("converted tool arguments exceed addressable bytes");
  auto converted_limits = limits;
  converted_limits.max_bytes = *expanded;
  auto arguments = j::Parse(call.arguments, converted_limits);
  if (!arguments || !arguments->root().is_object())
    return Bad("tool arguments must be a JSON object");
  return call;
}
}  // namespace

std::optional<std::size_t> ToolOutputBytes(const ToolCallFormat& format,
                                           std::size_t generated_bytes) {
  // JSON quoting expands one input byte to at most six. At peak, retain the
  // input/scanner, converted arguments, scalar conversion and object-notation
  // temporary strings. Each growing string/vector is counted at twice its
  // logical bound; JSON reserves exact table sizes (base/json.cc).
  const auto escaped = OutputJsonBytes(generated_bytes);
  std::size_t strings = 0, total = 0;
  if (!escaped || __builtin_mul_overflow(*escaped, std::size_t{8}, &strings)) return {};
  const auto parsed = j::ParseWorkingBytesBound(*escaped);
  if (!parsed || __builtin_mul_overflow(*parsed, std::size_t{2}, &total) ||
      __builtin_add_overflow(total, strings, &total) ||
      __builtin_add_overflow(
          total, sizeof(ToolInterpreter) + sizeof(ParsedCall) + 2 * sizeof(j::Document), &total))
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
    if (!parsed || !transient ||
        __builtin_add_overflow(*parsed, 2 * sizeof(ToolDefinition) + sizeof(j::Document),
                               &retained) ||
        __builtin_add_overflow(retained, 2 * d.name.size(), &retained) ||
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
  out.data_ = data;
  std::size_t bytes = 0;
  for (const auto& definition : definitions) {
    if (definition.name.empty()) return Bad("tool name is empty");
    if (out.Declared(definition.name)) return Bad("duplicate tool name");
    if (definition.strict) return Bad("strict tool arguments require constrained output");
    const auto add = [&](std::size_t size) {
      if (size > limits.max_bytes - bytes) return false;
      bytes += size;
      return true;
    };
    if (!add(definition.name.size()) || !add(definition.description.size()) ||
        !add(definition.parameters_json.size()))
      return Bad("tool definitions exceed their byte budget");
    auto parameters = j::Parse(definition.parameters_json, limits);
    if (!parameters || !parameters->root().is_object())
      return Bad("tool parameters must be a schema object");
    std::string text = "{\"type\":\"function\",\"function\":{\"name\":";
    j::AppendQuoted(definition.name, text);
    text += ",\"description\":";
    j::AppendQuoted(definition.description, text);
    text += ",\"parameters\":";
    Json(parameters->root(), text);
    text += "}}";
    const auto expanded = DefinitionJsonBytes(definition);
    if (!expanded) return Bad("normalized tool definition exceeds addressable bytes");
    auto normalized_limits = limits;
    normalized_limits.max_bytes = *expanded;
    auto document = j::Parse(text, normalized_limits);
    if (!document) return Bad("normalized tool definition exceeds its JSON budget");
    data->definitions.push_back(
        {definition.name, std::make_shared<const j::Document>(std::move(*document)), text.size()});
  }
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
bool ToolRequest::Declared(std::string_view name) const {
  return std::ranges::any_of(data_->definitions, [&](const auto& d) { return d.name == name; });
}
std::size_t ToolRequest::TemplateBytes() const {
  if (choice().kind == ToolChoiceKind::kNone) return 0;
  std::size_t bytes = 0;
  for (const auto& d : data_->definitions)
    if (choice().kind != ToolChoiceKind::kNamed || d.name == choice().name) bytes += d.bytes;
  return bytes;
}
std::size_t ToolRequest::NameBytes() const {
  std::size_t size = choice().name.size();
  for (const auto& d : data_->definitions) size = std::max(size, d.name.size());
  return size;
}
ToolStatus ToolRequest::ValidateNames(const ToolCallFormat& format) const {
  if (choice().kind == ToolChoiceKind::kNone || format.arguments == ToolArguments::kJsonEnvelope)
    return {};
  DelimiterScan scan;
  if (format.arguments == ToolArguments::kObjectNotation) scan.nesting.reserve(NameBytes());
  for (const auto& definition : data_->definitions) {
    const auto& name = definition.name;
    if (choice().kind == ToolChoiceKind::kNamed && name != choice().name) continue;
    const JoinedText header(name, format.name_end);
    if (Strip(std::string_view(name)) != name || format.name_end.empty() ||
        header.find(format.name_end) != name.size())
      return Bad("tool name cannot round-trip through this template's name boundary");
    if (format.arguments == ToolArguments::kObjectNotation) {
      // Object notation scans its raw name as part of the call. Reuse that
      // scanner so quoted or premature call boundaries cannot hide the end.
      scan.at = 0;
      scan.nesting.clear();
      scan.quoted = scan.escaped = false;
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
    return rest.starts_with(terminator) || terminator.starts_with(rest);
  };
  if (choice().kind == ToolChoiceKind::kNamed) return allowed(choice().name);
  if (choice().kind == ToolChoiceKind::kRequired)
    return std::ranges::any_of(data_->definitions, [&](const auto& d) { return allowed(d.name); });
  return true;
}
std::optional<j::Value> ToolRequest::Parameter(std::string_view name, std::string_view key) const {
  for (const auto& d : data_->definitions) {
    if (d.name != name) continue;
    const auto function = d.owner->root().find("function");
    const auto parameters = function->find("parameters");
    const auto properties = parameters->find("properties");
    return properties ? properties->find(key) : std::nullopt;
  }
  return {};
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
      .required = request.choice().kind == ToolChoiceKind::kRequired ||
                  request.choice().kind == ToolChoiceKind::kNamed,
      .name = request.choice().kind == ToolChoiceKind::kNamed ? std::optional(request.choice().name)
                                                              : std::nullopt,
      .ends_reasoning = format.opener_ends_reasoning};
}

ToolInterpreter::ToolInterpreter(ToolCallFormat format, ToolRequest request, j::Limits limits)
    : format_(std::move(format)), request_(std::move(request)), limits_(limits) {
  if (!ToolOutputBytes(format_, limits_.max_bytes))
    error_ = "tool output budget exceeds addressable bytes";
  if (format_.call_begin.empty() || format_.call_end.empty() || format_.quote.empty() ||
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
        overlaps(alias, format_.parameter_begin))
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
void ToolInterpreter::ResetEnvelope() {
  envelope_ = {};
  envelope_whole_ = {};
  envelope_size_ = std::string::npos;
}
void ToolInterpreter::RefreshEnvelope() const {
  if (!format_.bare_envelope || !in_call_ || envelope_size_ == pending_.size()) return;
  const JoinedText pending(pending_);
  auto stable = scan_at_;
  if (stable < pending_.size()) {
    DelimiterScan scan{scan_at_, scan_nesting_, scan_quoted_, scan_escaped_};
    Outside(pending, format_.call_end, format_.quote, scan);
    stable = scan.at;
  }
  // Outside holds a partial closing delimiter. Keep the resumable body cursor
  // before those bytes: a later append can make its body end precede the old
  // whole-pending projection. Probes revisit only the held delimiter suffix.
  Envelope(pending.substr(0, stable), format_, request_, envelope_);
  envelope_whole_ = envelope_;
  // A paused callback can retain a closed call and later text. CanStop and the
  // provisional required-name check historically examine that complete buffer.
  Envelope(pending, format_, request_, envelope_whole_);
  envelope_size_ = pending_.size();
}
bool ToolInterpreter::CanStop() const {
  if (error_ || ended_ || (in_block_ && block_calls_ == calls_)) return false;
  if (in_call_) {
    if (!format_.bare_envelope || request_.choice().kind != ToolChoiceKind::kAuto) return false;
    // The same ordinary-answer fallback as Finish: an unmatched JSON opener
    // without an authenticated call name/argument pair is still answer text.
    RefreshEnvelope();
    const auto info = envelope_whole_.projected;
    if ((info.name && info.arguments) || info.allowed) return false;
  }
  return calls_ != 0 || (request_.choice().kind != ToolChoiceKind::kRequired &&
                         request_.choice().kind != ToolChoiceKind::kNamed);
}
bool ToolInterpreter::Allows(std::string_view candidate) const {
  if (error_ || ended_ || candidate.size() > limits_.max_bytes - received_) return false;
  if (request_.choice().kind == ToolChoiceKind::kNone) return true;
  RefreshEnvelope();
  JoinedText pending(pending_, candidate);
  auto envelope = envelope_;
  auto whole_envelope = envelope_whole_;
  DelimiterScan scan{scan_at_, scan_nesting_, scan_quoted_, scan_escaped_};
  auto phase = tag_phase_;
  auto json = tag_json_;
  auto name_end = tag_name_end_, search = tag_header_search_;
  auto calls = calls_;
  bool in_call = in_call_;
  for (;;) {
    if (in_call) {
      if (request_.choice().kind == ToolChoiceKind::kNamed ||
          request_.choice().kind == ToolChoiceKind::kRequired) {
        if (format_.bare_envelope) {
          auto probe = whole_envelope;
          const auto info = Envelope(pending, format_, request_, probe);
          if (info.name && !info.allowed) return false;
        } else {
          if (format_.arguments == ToolArguments::kJsonEnvelope) return false;
          const auto header = static_cast<std::string>(pending.substr(
              0, std::min(pending.size(), request_.NameBytes() + format_.name_end.size())));
          if (!request_.NamePrefix(header, format_.name_end)) return false;
        }
      }
      std::optional<TaggedBoundary> end;
      if (format_.arguments == ToolArguments::kTaggedParameters)
        end = TaggedEnd(pending, format_, request_, scan, phase, json, name_end, search);
      else if (const auto at = Outside(pending, format_.call_end, format_.quote, scan))
        end = TaggedBoundary{*at, format_.call_end.size()};
      if (!end) return true;
      bool is_call = true;
      if (format_.bare_envelope) {
        auto probe = envelope;
        const auto info = Envelope(pending.substr(0, end->at), format_, request_, probe);
        is_call = info.name && info.arguments;
        const auto kind = request_.choice().kind;
        if ((kind == ToolChoiceKind::kRequired || kind == ToolChoiceKind::kNamed) &&
            (!info.name || !info.arguments || !info.allowed))
          return false;
      }
      pending.remove_prefix(end->at + end->bytes);
      in_call = false;
      if (is_call) {
        if (!request_.parallel() && calls != 0) return false;
        ++calls;
      }
      scan = {};
      envelope = {};
      whole_envelope = {};
      phase = 0;
      name_end = std::string::npos;
      search = 0;
      continue;
    }
    const std::array<std::string_view, 3> markers{format_.call_begin, format_.block_begin,
                                                  format_.block_end};
    std::size_t found = std::string::npos, marker = 0;
    for (std::size_t i = 0; i < markers.size(); ++i) {
      if (markers[i].empty()) continue;
      const auto at = pending.find(markers[i]);
      if (at < found) {
        found = at;
        marker = i;
      }
    }
    if (found == std::string::npos) return true;
    pending.remove_prefix(found + markers[marker].size());
    if (marker == 0) {
      if (!format_.bare_envelope && !request_.parallel() && calls != 0) return false;
      in_call = true;
    }
  }
}

ToolStatus ToolInterpreter::Fail(std::string error) {
  error_ = std::move(error);
  return std::unexpected(*error_);
}
ToolStatus ToolInterpreter::Emit(ToolEvent event, const Sink& sink) {
  if (const auto* text = std::get_if<TextPart>(&event); text && j::FirstInvalidUtf8(text->text))
    return Fail("generated text is not UTF-8");
  const auto flow = sink(std::move(event));
  if (flow.action == Flow::kCancel) return Fail("tool output was cancelled");
  paused_ = flow.action == Flow::kPause;
  if (error_) return std::unexpected(*error_);
  return {};
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
  if (request_.choice().kind == ToolChoiceKind::kNone) {
    if (pending_.empty()) return {};
    auto bytes = TextBytes(pending_, final);
    if (!bytes) return Fail(bytes.error());
    if (*bytes == 0) return {};
    auto text = pending_.substr(0, *bytes);
    pending_.erase(0, *bytes);
    return Emit(TextPart{std::move(text)}, sink);
  }
  for (;;) {
    if (in_call_) {
      // State retained across Push calls; only the final candidate is parsed.
      DelimiterScan scan{scan_at_, std::move(scan_nesting_), scan_quoted_, scan_escaped_};
      std::optional<TaggedBoundary> end;
      if (format_.arguments == ToolArguments::kTaggedParameters)
        end = TaggedEnd(std::string_view(pending_), format_, request_, scan, tag_phase_, tag_json_,
                        tag_name_end_, tag_header_search_);
      else if (const auto at =
                   Outside(std::string_view(pending_), format_.call_end, format_.quote, scan))
        end = TaggedBoundary{*at, format_.call_end.size()};
      scan_at_ = scan.at;
      scan_nesting_ = std::move(scan.nesting);
      scan_quoted_ = scan.quoted;
      scan_escaped_ = scan.escaped;
      RefreshEnvelope();
      if (!end) return {};
      if (format_.bare_envelope && request_.choice().kind == ToolChoiceKind::kAuto) {
        const auto body = std::string_view(pending_).substr(0, end->at);
        const auto info = envelope_.projected;
        if (!info.name || !info.arguments) {
          auto text = format_.call_begin + std::string(body) + format_.call_end;
          pending_.erase(0, end->at + end->bytes);
          in_call_ = false;
          ResetEnvelope();
          scan_at_ = 0;
          scan_nesting_.clear();
          scan_quoted_ = scan_escaped_ = false;
          if (auto emitted = Emit(TextPart{std::move(text)}, sink); !emitted) return emitted;
          if (paused_) return {};
          continue;
        }
      }
      auto call = syntax_only_ ? std::expected<ParsedCall, std::string>(ParsedCall{})
                               : ParseCall(std::string_view(pending_).substr(0, end->at), format_,
                                           request_, limits_);
      if (!call) return Fail(call.error());
      const auto kind = request_.choice().kind;
      if (!syntax_only_ &&
          ((kind == ToolChoiceKind::kRequired && !request_.Declared(call->name)) ||
           (kind == ToolChoiceKind::kNamed && call->name != request_.choice().name)))
        return Fail("generated tool name violates tool choice");
      if (!request_.parallel() && calls_ != 0) return Fail("parallel tool calls are disabled");
      ++calls_;
      pending_.erase(0, end->at + end->bytes);
      in_call_ = false;
      ResetEnvelope();
      scan_at_ = 0;
      scan_nesting_.clear();
      scan_quoted_ = scan_escaped_ = false;
      tag_phase_ = 0;
      tag_name_end_ = std::string::npos;
      tag_header_search_ = 0;
      if (auto emitted = Emit(ToolUsePart{.id = {},
                                          .name = std::move(call->name),
                                          .input_json = std::move(call->arguments)},
                              sink);
          !emitted)
        return emitted;
      if (paused_) return {};
      continue;
    }
    if (pending_.empty()) return {};
    const std::array<std::string_view, 4> markers{
        format_.call_begin, format_.block_begin, format_.block_end,
        reasoning_active_ ? format_.reasoning_end : std::string_view{}};
    std::size_t found = std::string::npos, marker = 0;
    for (std::size_t i = 0; i < markers.size(); ++i) {
      if (markers[i].empty()) continue;
      const auto at = pending_.find(markers[i]);
      if (at < found) {
        found = at;
        marker = i;
      }
    }
    if (found == std::string::npos) {
      std::size_t held = 0;
      if (!final)
        for (const auto token : markers)
          for (std::size_t n = 1; n < token.size() && n <= pending_.size(); ++n)
            if (std::string_view(pending_).ends_with(token.substr(0, n))) held = std::max(held, n);
      auto size = pending_.size() - held;
      if (!in_block_) {
        const auto bytes = TextBytes(std::string_view(pending_).substr(0, size), final);
        if (!bytes) return Fail(bytes.error());
        size = *bytes;
      }
      if (in_block_) {
        if (!Strip(std::string_view(pending_).substr(0, size)).empty())
          return Fail("unrecognized text inside a tool calls block");
      }
      auto text = pending_.substr(0, size);
      pending_.erase(0, size);
      if (!in_block_ && size != 0) return Emit(TextPart{std::move(text)}, sink);
      return {};
    }
    const auto prefix = std::string_view(pending_).substr(0, found);
    if (in_block_ && !Strip(prefix).empty())
      return Fail("unrecognized text inside a tool calls block");
    if (!in_block_ && !prefix.empty()) {
      auto text = std::string(prefix);
      pending_.erase(0, found);
      if (auto emitted = Emit(TextPart{std::move(text)}, sink); !emitted) return emitted;
      if (paused_) return {};
      continue;
    }
    pending_.erase(0, found + markers[marker].size());
    if (marker == 3) {
      reasoning_active_ = false;
      reasoning_ended_ = true;
      reasoning_active_ = false;
      if (auto emitted = Emit(ToolReasoningEnd{}, sink); !emitted) return emitted;
      if (paused_) return {};
      continue;
    }
    if (marker == 2) {
      in_block_ = false;
      continue;
    }
    if (marker == 1) {
      in_block_ = true;
      block_calls_ = calls_;
    }
    if (marker == 0) {
      if (!format_.bare_envelope && !request_.parallel() && calls_ != 0)
        return Fail("parallel tool calls are disabled");
      in_call_ = true;
      ResetEnvelope();
    }
    if (format_.opener_ends_reasoning && !reasoning_ended_) {
      reasoning_ended_ = true;
      reasoning_active_ = false;
      if (auto emitted = Emit(ToolReasoningEnd{}, sink); !emitted) return emitted;
      if (paused_) return {};
    }
  }
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
  if (in_call_ && format_.bare_envelope && request_.choice().kind == ToolChoiceKind::kAuto) {
    RefreshEnvelope();
    const auto info = envelope_whole_.projected;
    if ((!info.name || !info.arguments) && !info.allowed) {
      auto text = format_.call_begin + std::exchange(pending_, {});
      in_call_ = false;
      if (auto emitted = Emit(TextPart{std::move(text)}, sink); !emitted) return emitted;
    }
  }
  if (paused_) return {};
  ended_ = true;
  if (in_call_ || (in_block_ && block_calls_ == calls_)) {
    if (reason != ToolFinish::kLength) return Fail("incomplete tool call at natural completion");
    pending_.clear();
  }
  if ((request_.choice().kind == ToolChoiceKind::kRequired ||
       request_.choice().kind == ToolChoiceKind::kNamed) &&
      calls_ == 0)
    return Fail("generation did not complete a required tool call");
  return {};
}

std::expected<std::vector<ToolResultPart>, std::string> OrderToolResults(
    std::span<const ToolUsePart> calls, std::span<const ToolResultPart> results) {
  std::vector<ToolResultPart> out(results.begin(), results.end());
  std::vector<std::size_t> order;
  bool missing = false;
  for (std::size_t i = 0; i < calls.size(); ++i) {
    if (calls[i].id.empty()) {
      missing = true;
      continue;
    }
    for (std::size_t p = 0; p < i; ++p)
      if (calls[p].id == calls[i].id) return Bad("duplicate tool call ID");
  }
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results[i].tool_use_id.empty()) {
      missing = true;
      continue;
    }
    for (std::size_t p = 0; p < i; ++p)
      if (results[p].tool_use_id == results[i].tool_use_id) return Bad("duplicate tool result ID");
    const auto found = std::ranges::find(calls, results[i].tool_use_id, &ToolUsePart::id);
    if (found == calls.end()) return Bad("unknown tool result ID");
    order.push_back(static_cast<std::size_t>(found - calls.begin()));
  }
  if (missing) return out;
  std::vector<std::size_t> indices(results.size());
  for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
  std::ranges::sort(indices, {}, [&](std::size_t i) { return order[i]; });
  for (std::size_t i = 0; i < indices.size(); ++i) out[i] = results[indices[i]];
  return out;
}
}  // namespace llmp::chat
