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
struct DelimiterScan {
  std::size_t at = 0;
  std::vector<char> nesting;
  bool quoted = false, escaped = false;
};
std::optional<std::size_t> Outside(std::string_view text, std::string_view needle,
                                   std::string_view quote, DelimiterScan& scan,
                                   std::string_view second = {}, std::string_view third = {},
                                   bool structured = true) {
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
      if (!scan.escaped && rest.size() < quote.size() && quote.starts_with(rest)) return {};
      scan.escaped = false;
    } else {
      if (structured && rest.starts_with(quote)) {
        scan.quoted = true;
        scan.at += quote.size();
        continue;
      }
      if ((structured && rest.size() < quote.size() && quote.starts_with(rest)) ||
          (rest.size() < needle.size() && needle.starts_with(rest)) ||
          (rest.size() < second.size() && second.starts_with(rest)) ||
          (rest.size() < third.size() && third.starts_with(rest)))
        return {};
      const char c = rest.front();
      if (structured && (c == '{' || c == '[')) scan.nesting.push_back(c);
      if ((c == '}' || c == ']') && !scan.nesting.empty()) scan.nesting.pop_back();
    }
    ++scan.at;
  }
  return {};
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
bool StructuredValue(std::string_view body, int syntax) {
  if (syntax == 1) return true;
  body = Strip(body);
  return !body.empty() &&
         (((syntax & 2) != 0 && body.front() == '{') || ((syntax & 4) != 0 && body.front() == '['));
}

std::optional<std::size_t> TaggedEnd(std::string_view body, const ToolCallFormat& format,
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
      if (rest.starts_with(format.call_end)) return scan.at;
      if (format.call_end.starts_with(rest) || format.parameter_begin.starts_with(rest)) return {};
      if (!rest.starts_with(format.parameter_begin)) {
        const auto end = body.find(format.call_end, scan.at);
        return end == std::string_view::npos ? std::nullopt : std::optional(end);
      }
      const auto name_at = scan.at + format.parameter_begin.size();
      const auto end = body.find(format.parameter_name_end, std::max(name_at, header_search));
      if (end == std::string_view::npos) {
        header_search = body.size() >= format.parameter_name_end.size()
                            ? body.size() - format.parameter_name_end.size() + 1
                            : name_at;
        return {};
      }
      header_search = end;
      const auto name = body.substr(name_at, end - name_at);
      auto value_at = end + format.parameter_name_end.size();
      std::optional<bool> string;
      if (!format.string_suffix.empty()) {
        const auto suffix = body.substr(value_at);
        if (suffix.starts_with(format.string_suffix)) {
          string = true;
          value_at += format.string_suffix.size();
        } else if (suffix.starts_with(format.json_suffix)) {
          string = false;
          value_at += format.json_suffix.size();
        } else if (format.string_suffix.starts_with(suffix) ||
                   format.json_suffix.starts_with(suffix))
          return {};
        else {
          const auto closing = body.find(format.call_end, value_at);
          return closing == std::string_view::npos ? std::nullopt : std::optional(closing);
        }
      }
      json = ParameterSyntax(request.Parameter(call_name, name), string);
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
    if (body.substr(*end).starts_with(format.call_end)) return end;
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
    auto doc = j::Parse(body, limits);
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
        if (!body.starts_with(format.parameter_begin))
          return Bad("invalid tool parameter boundary");
        body.remove_prefix(format.parameter_begin.size());
        const auto name_end = body.find(format.parameter_name_end);
        if (name_end == std::string_view::npos) return Bad("tool parameter has no name boundary");
        const auto name = body.substr(0, name_end);
        if (name.empty()) return Bad("tool parameter name is empty");
        body.remove_prefix(name_end + format.parameter_name_end.size());
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
        &format.json_suffix, &format.quote, &format.null_word, &format.reasoning_end}) {
    std::size_t owned = 0;
    if (__builtin_mul_overflow(field->size(), std::size_t{2}, &owned) ||
        __builtin_add_overflow(total, owned, &total))
      return {};
  }
  return total;
}
std::optional<std::size_t> ToolRequestBytes(std::span<const ToolDefinition> definitions) {
  std::size_t total = sizeof(ToolRequest), peak = 0;
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
  out.choice_ = std::move(choice);
  out.parallel_ = parallel;
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
    out.definitions_.push_back(
        {definition.name, std::make_shared<const j::Document>(std::move(*document))});
  }
  if ((out.choice_.kind == ToolChoiceKind::kRequired ||
       out.choice_.kind == ToolChoiceKind::kNamed) &&
      out.definitions_.empty())
    return Bad("required tool choice needs declared tools");
  if (out.choice_.kind == ToolChoiceKind::kNamed && !out.Declared(out.choice_.name))
    return Bad("named tool choice must name a declared tool");
  if (out.choice_.kind != ToolChoiceKind::kNamed && !out.choice_.name.empty())
    return Bad("only named tool choice carries a name");
  return out;
}
bool ToolRequest::Declared(std::string_view name) const {
  return std::ranges::any_of(definitions_, [&](const auto& d) { return d.name == name; });
}
std::optional<j::Value> ToolRequest::Parameter(std::string_view name, std::string_view key) const {
  for (const auto& d : definitions_) {
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
  if (choice_.kind == ToolChoiceKind::kNone) return out;
  for (const auto& d : definitions_)
    if (choice_.kind != ToolChoiceKind::kNamed || choice_.name == d.name)
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
                                     .reasoning_end = "</think>"};
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
                                     .reasoning_end = "</think>"};
  return format;
}
const ToolCallFormat& GemmaToolCalls() {
  static const ToolCallFormat format{.call_begin = "<|tool_call>call:",
                                     .call_end = "<tool_call|>",
                                     .name_end = "{",
                                     .arguments = ToolArguments::kObjectNotation,
                                     .reasoning_end = "<channel|>",
                                     .quote = "<|\"|>",
                                     .null_word = "None"};
  return format;
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
}
ToolStatus ToolInterpreter::Fail(std::string error) {
  error_ = std::move(error);
  return std::unexpected(*error_);
}
ToolStatus ToolInterpreter::Emit(ToolEvent event, const Sink& sink) {
  if (const auto* text = std::get_if<TextPart>(&event); text && j::FirstInvalidUtf8(text->text))
    return Fail("generated text is not UTF-8");
  if (!sink(std::move(event))) return Fail("tool output was cancelled");
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
      auto end = format_.arguments == ToolArguments::kTaggedParameters
                     ? TaggedEnd(pending_, format_, request_, scan, tag_phase_, tag_json_,
                                 tag_name_end_, tag_header_search_)
                     : Outside(pending_, format_.call_end, format_.quote, scan);
      scan_at_ = scan.at;
      scan_nesting_ = std::move(scan.nesting);
      scan_quoted_ = scan.quoted;
      scan_escaped_ = scan.escaped;
      if (!end) return {};
      auto call = ParseCall(std::string_view(pending_).substr(0, *end), format_, request_, limits_);
      if (!call) return Fail(call.error());
      const auto kind = request_.choice().kind;
      if ((kind == ToolChoiceKind::kRequired && !request_.Declared(call->name)) ||
          (kind == ToolChoiceKind::kNamed && call->name != request_.choice().name))
        return Fail("generated tool name violates tool choice");
      if (auto emitted = Emit(ToolUsePart{.id = {},
                                          .name = std::move(call->name),
                                          .input_json = std::move(call->arguments)},
                              sink);
          !emitted)
        return emitted;
      ++calls_;
      pending_.erase(0, *end + format_.call_end.size());
      in_call_ = false;
      scan_at_ = 0;
      scan_nesting_.clear();
      scan_quoted_ = scan_escaped_ = false;
      tag_phase_ = 0;
      tag_name_end_ = std::string::npos;
      tag_header_search_ = 0;
      continue;
    }
    if (pending_.empty()) return {};
    const std::array<std::string_view, 3> markers{format_.call_begin, format_.block_begin,
                                                  format_.block_end};
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
      } else if (size != 0) {
        if (auto emitted = Emit(TextPart{pending_.substr(0, size)}, sink); !emitted) return emitted;
      }
      pending_.erase(0, size);
      return {};
    }
    const auto prefix = std::string_view(pending_).substr(0, found);
    if (in_block_ && !Strip(prefix).empty())
      return Fail("unrecognized text inside a tool calls block");
    if (!in_block_ && !prefix.empty())
      if (auto emitted = Emit(TextPart{std::string(prefix)}, sink); !emitted) return emitted;
    pending_.erase(0, found + markers[marker].size());
    if (marker == 2) {
      in_block_ = false;
      continue;
    }
    if (marker == 1) {
      in_block_ = true;
      block_calls_ = calls_;
    }
    if (format_.opener_ends_reasoning && !reasoning_ended_) {
      if (auto emitted = Emit(ToolReasoningEnd{}, sink); !emitted) return emitted;
      reasoning_ended_ = true;
    }
    if (marker == 0) {
      if (!request_.parallel() && calls_ != 0) return Fail("parallel tool calls are disabled");
      in_call_ = true;
    }
  }
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
  if (auto drained = Drain(true, sink); !drained) return drained;
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
