// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/check.h"
#include "base/json.h"

namespace llmp::runtime::api {
namespace {

namespace json = base::json;

// One entry of an unordered index of names (a node with its cached hash and a
// bucket, with allocator rounding).
constexpr std::uint64_t kNameIndexEntryBytes = 64;

std::unexpected<Error> Bad(std::string message, std::string param = {}, std::string code = {}) {
  return std::unexpected(Error{.status = 400,
                               .type = "invalid_request_error",
                               .message = std::move(message),
                               .param = std::move(param),
                               .code = std::move(code)});
}

std::string Quoted(std::string_view text) {
  std::string out;
  json::AppendQuoted(text, out);
  return out;
}

// A number that must equal `value` (a parameter accepted only at the
// value that means "off").
bool NumberIs(const json::Value& v, double value) {
  if (!v.is_number()) {
    return false;
  }
  const std::optional<double> x = v.float64();
  return x && *x == value;
}

// The unknown fields' names a request carries (ChatRequest::ignored).
class Ignored {
 public:
  explicit Ignored(std::vector<std::string>& names) : names_(names) {}

  void Add(std::string_view prefix, std::string_view key) {
    if (names_.size() >= kMaxIgnoredPerRequest) {
      return;
    }
    std::string name(prefix);
    name += key;
    if (name.size() > kMaxIgnoredNameBytes) {
      // Cut at a character: never inside a UTF-8 sequence (the parser
      // checked the key's encoding).
      std::size_t cut = kMaxIgnoredNameBytes;
      while (cut > 0 && (static_cast<unsigned char>(name[cut]) & 0xC0U) == 0x80U) {
        --cut;
      }
      name.resize(cut);
    }
    if (std::ranges::find(names_, name) == names_.end()) {
      names_.push_back(std::move(name));
    }
  }

 private:
  std::vector<std::string>& names_;
};

// A message's content: a string, or owned ordered text parts.
std::expected<chat::Content, Error> Content(const json::Value& content, const std::string& at,
                                            Ignored& ignored, RequestMemory* memory,
                                            MemoryCharge& charge) {
  if (content.is_string()) {
    return chat::Content(std::string(content.string()));
  }
  if (!content.is_array()) {
    return Bad(std::format("{}.content must be a string or an array of text parts", at),
               at + ".content");
  }
  // Any number of parts: their bytes are the body's (D-102).
  const std::uint64_t parts_bytes = content.size() * sizeof(chat::MessagePart);
  if (memory != nullptr && !charge.Add(*memory, parts_bytes)) {
    return std::unexpected(MemoryRefusal(*memory, parts_bytes, "message parts"));
  }
  std::vector<chat::MessagePart> text;
  text.reserve(content.size());
  for (std::size_t j = 0; j < content.size(); ++j) {
    const json::Value part = content.at(j);
    const std::string where = std::format("{}.content[{}]", at, j);
    if (!part.is_object()) {
      return Bad(where + " must be an object", where);
    }
    auto type = part.find("type");
    if (!type || !type->is_string()) {
      return Bad(where + ".type must be a string", where + ".type");
    }
    if (type->string() != "text") {
      return Bad(
          std::format("{}.type is {}: this route takes text only", where, Quoted(type->string())),
          where + ".type", "unsupported_content_type");
    }
    for (std::size_t k = 0; k < part.size(); ++k) {
      const std::string_view key = part.key(k);
      if (key == "type" || key == "text" || key == "cache_control") {
        continue;  // cache_control: an advisory retention hint (D-045), ignored
      }
      ignored.Add("messages[].content[].", key);
    }
    auto piece = part.find("text");
    if (!piece || !piece->is_string()) {
      return Bad(where + ".text must be a string", where + ".text");
    }
    text.emplace_back(chat::TextPart{std::string(piece->string())});
  }
  return chat::Content(std::move(text));
}

// Minimal JSON text that preserves number spellings, including values outside
// machine floating point; iterative (any nesting is safe on any stack).
void AppendJson(json::Value v, std::string& out) { json::AppendValue(v, out); }

bool SameJson(json::Value a, json::Value b) {
  if (a.kind() != b.kind()) return false;
  if (a.is_string()) return a.string() == b.string();
  if (a.is_number()) return a.number() == b.number();
  if (!a.is_array() && !a.is_object()) return true;
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a.is_array()) {
      if (!SameJson(a.at(i), b.at(i))) return false;
    } else {
      const auto member = b.find(a.key(i));
      if (!member || !SameJson(a.member(i), *member)) return false;
    }
  }
  return true;
}

std::expected<chat::ToolUsePart, Error> ParseCall(json::Value value, bool legacy,
                                                  const std::string& where, Ignored& ignored,
                                                  RequestMemory* memory) {
  if (!value.is_object()) return Bad(where + " must be an object", where);
  chat::ToolUsePart out;
  if (!legacy) {
    auto type = value.find("type");
    if (!type || !type->is_string() || type->string() != "function")
      return Bad(where + ".type must be function", where);
    if (auto id = value.find("id")) {
      if (!id->is_string()) return Bad(where + ".id must be a string", where + ".id");
      out.id = std::string(id->string());
    }
    for (std::size_t i = 0; i < value.size(); ++i)
      if (value.key(i) != "id" && value.key(i) != "type" && value.key(i) != "function")
        ignored.Add("messages[].tool_calls[].", value.key(i));
    const auto function = value.find("function");
    if (!function || !function->is_object())
      return Bad(where + ".function must be an object", where);
    value = *function;
  }
  auto name = value.find("name");
  auto args = value.find("arguments");
  if (!name || !name->is_string() || name->string().empty() || !args || !args->is_string())
    return Bad(where + " requires a nonempty name and JSON string arguments", where);
  MemoryCharge working;
  const auto bytes = json::ParseWorkingBytes(args->string());
  if (memory != nullptr && !working.Add(*memory, bytes))
    return std::unexpected(MemoryRefusal(*memory, bytes, "tool call arguments"));
  const auto size = args->string().size();
  auto document = json::Parse(args->string(), {.max_bytes = size,
                                               .max_depth = kMaxJsonDepth,
                                               .max_values = size + 1,
                                               .max_string_bytes = size + 1});
  if (!document || !document->root().is_object())
    return Bad(where + ".arguments must encode a JSON object", where);
  out.name = std::string(name->string());
  out.input_json = std::string(args->string());
  for (std::size_t i = 0; i < value.size(); ++i)
    if (value.key(i) != "name" && value.key(i) != "arguments")
      ignored.Add(legacy ? "messages[].function_call." : "messages[].tool_calls[].function.",
                  value.key(i));
  return out;
}

std::expected<Message, Error> ParseMessage(const json::Value& m, std::size_t index,
                                           Ignored& ignored, RequestMemory* memory,
                                           MemoryCharge& charge) {
  const std::string at = std::format("messages[{}]", index);
  if (!m.is_object()) return Bad(at + " must be an object", at);
  Message out;
  auto role = m.find("role");
  if (!role || !role->is_string()) return Bad(at + ".role must be a string", at + ".role");
  const std::string_view r = role->string();
  if (r == "system" || r == "developer")
    out.role = Role::kSystem;
  else if (r == "user")
    out.role = Role::kUser;
  else if (r == "assistant")
    out.role = Role::kAssistant;
  else if (r == "tool" || r == "function")
    out.role = Role::kTool;
  else
    return Bad(at + ".role must be system, developer, user, assistant, tool or function",
               at + ".role");
  const bool assistant = out.role == Role::kAssistant;
  bool has_content = false;
  bool null_content = false;
  chat::ToolResultPart result;
  for (std::size_t k = 0; k < m.size(); ++k) {
    const std::string_view key = m.key(k);
    const json::Value v = m.member(k);
    const std::string where = std::format("{}.{}", at, key);
    if (key == "role") continue;
    if (key == "content") {
      if (v.is_null() && assistant) {
        null_content = true;
        has_content = true;
        continue;
      }
      auto text = Content(v, at, ignored, memory, charge);
      if (!text) return std::unexpected(text.error());
      out.content = std::move(*text);
      has_content = true;
    } else if (key == "name") {
      if (!v.is_string()) return Bad(where + " must be a string", where);
      if (out.role == Role::kTool) result.name = std::string(v.string());
    } else if (out.role == Role::kTool && key == "tool_call_id") {
      if (!v.is_string()) return Bad(where + " must be a string", where);
      result.tool_use_id = std::string(v.string());
    } else if (out.role == Role::kTool && key == "is_error") {
      if (!v.is_bool()) return Bad(where + " must be a boolean", where);
      result.is_error = v.boolean();
    } else if (assistant && (key == "reasoning" || key == "reasoning_content")) {
      if (v.is_null()) continue;
      if (!v.is_string()) return Bad(where + " must be a string", where);
      if (out.reasoning) return Bad(at + " has both reasoning and reasoning_content", where);
      out.reasoning = std::string(v.string());
    } else if (assistant && (key == "tool_calls" || key == "function_call")) {
      // Decode together below, independent of object-member order.
      continue;
    } else if (assistant && ((key == "annotations" && v.is_array()) ||
                             ((key == "refusal" || key == "audio") && v.is_null())))
      continue;
    else if (assistant && key == "refusal")
      return Bad(where + ": send a refusal as content", where);
    else if (assistant && key == "audio")
      return Bad(where + ": audio is not supported by this route", where);
    else
      ignored.Add("messages[].", key);
  }
  if (!assistant && !has_content) return Bad(at + ".content is required", at + ".content");
  if (out.role == Role::kTool) {
    if (const auto* parts = out.content.parts()) {
      for (const auto& part : *parts) result.text += std::get<chat::TextPart>(part).text;
    } else if (out.content.text())
      result.text = *out.content.text();
    if (memory != nullptr && !charge.Add(*memory, sizeof(chat::MessagePart)))
      return std::unexpected(MemoryRefusal(*memory, sizeof(chat::MessagePart), "tool result"));
    std::vector<chat::MessagePart> parts;
    parts.emplace_back(std::move(result));
    out.content = chat::Content(std::move(parts));
  }
  if (assistant) {
    auto modern = m.find("tool_calls");
    auto legacy = m.find("function_call");
    const bool has_modern = modern && !modern->is_null();
    const bool has_legacy = legacy && !legacy->is_null();
    if (has_modern && !modern->is_array())
      return Bad(at + ".tool_calls must be an array", at + ".tool_calls");
    const std::size_t count = has_modern ? modern->size() : (has_legacy ? 1 : 0);
    if (has_legacy && has_modern && count != 1)
      return Bad(at + " has conflicting tool_calls and function_call", at + ".function_call");
    if (!has_content && count == 0) return Bad(at + ".content is required", at + ".content");
    // Call-only assistant history may omit content. Preserve its absence,
    // exactly as explicit null, while validating every call below.
    null_content = null_content || !has_content;
    const auto* old = out.content.parts();
    const std::size_t text_count =
        old ? old->size() : (!null_content && out.content.text() ? 1 : 0);
    const std::uint64_t bytes = (count + text_count) * sizeof(chat::MessagePart);
    if (count && memory != nullptr && !charge.Add(*memory, bytes))
      return std::unexpected(MemoryRefusal(*memory, bytes, "assistant tool calls"));
    std::vector<chat::MessagePart> parts;
    if (count) {
      parts.reserve(count + text_count);
      if (old)
        parts = *old;
      else if (!null_content && out.content.text())
        parts.emplace_back(chat::TextPart{*out.content.text()});
    }
    for (std::size_t i = 0; i < count; ++i) {
      const auto where =
          has_modern ? std::format("{}.tool_calls[{}]", at, i) : at + ".function_call";
      auto call =
          ParseCall(has_modern ? modern->at(i) : *legacy, !has_modern, where, ignored, memory);
      if (!call) return std::unexpected(call.error());
      if (has_legacy && has_modern) {
        auto old_call = ParseCall(*legacy, true, at + ".function_call", ignored, memory);
        if (!old_call) return std::unexpected(old_call.error());
        // Parsed JSON equality below admits irrelevant whitespace/key ordering.
        MemoryCharge comparison;
        const auto compare_bytes = json::ParseWorkingBytes(call->input_json) +
                                   json::ParseWorkingBytes(old_call->input_json);
        if (memory != nullptr && !comparison.Add(*memory, compare_bytes))
          return std::unexpected(MemoryRefusal(*memory, compare_bytes, "tool call comparison"));
        auto a = json::Parse(call->input_json), b = json::Parse(old_call->input_json);
        if (call->name != old_call->name || !a || !b || !SameJson(a->root(), b->root()))
          return Bad(at + " has conflicting tool_calls and function_call", at + ".function_call");
      }
      parts.emplace_back(std::move(*call));
    }
    if (count) out.content = chat::Content(std::move(parts));
  }
  return out;
}

std::expected<chat::ToolChoice, Error> ParseToolChoice(json::Value v, bool legacy,
                                                       Ignored& ignored) {
  const std::string field = legacy ? "function_call" : "tool_choice";
  if (v.is_string()) {
    if (v.string() == "auto") return chat::ToolChoice{chat::ToolChoiceKind::kAuto, {}};
    if (v.string() == "none") return chat::ToolChoice{chat::ToolChoiceKind::kNone, {}};
    if (v.string() == "required") return chat::ToolChoice{chat::ToolChoiceKind::kRequired, {}};
  } else if (v.is_object()) {
    if (!legacy) {
      auto type = v.find("type"), function = v.find("function");
      if (!type || !type->is_string() || type->string() != "function" || !function ||
          !function->is_object())
        return Bad("tool_choice must select a function", field);
      for (std::size_t i = 0; i < v.size(); ++i)
        if (v.key(i) != "type" && v.key(i) != "function") ignored.Add("tool_choice.", v.key(i));
      v = *function;
    }
    auto name = v.find("name");
    if (name && name->is_string() && !name->string().empty()) {
      for (std::size_t i = 0; i < v.size(); ++i)
        if (v.key(i) != "name")
          ignored.Add(legacy ? "function_call." : "tool_choice.function.", v.key(i));
      return chat::ToolChoice{chat::ToolChoiceKind::kNamed, std::string(name->string())};
    }
  }
  return Bad(field + " must be auto, none, required or a named function", field);
}

std::expected<std::vector<chat::ToolDefinition>, Error> ParseDefinitions(json::Value v, bool legacy,
                                                                         Ignored& ignored,
                                                                         RequestMemory* memory,
                                                                         MemoryCharge& charge) {
  const std::string field = legacy ? "functions" : "tools";
  if (!v.is_array()) return Bad(field + " must be an array", field);
  const std::uint64_t bytes = v.size() * sizeof(chat::ToolDefinition);
  if (memory != nullptr && !charge.Add(*memory, bytes))
    return std::unexpected(MemoryRefusal(*memory, bytes, "tool definitions"));
  std::vector<chat::ToolDefinition> out;
  out.reserve(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    auto definition = v.at(i);
    const auto where = std::format("{}[{}]", field, i);
    if (!definition.is_object()) return Bad(where + " must be an object", field);
    if (!legacy) {
      auto type = definition.find("type"), function = definition.find("function");
      if (!type || !type->is_string() || type->string() != "function" || !function ||
          !function->is_object())
        return Bad(where + " must contain a function definition", field);
      for (std::size_t k = 0; k < definition.size(); ++k)
        if (definition.key(k) != "type" && definition.key(k) != "function")
          ignored.Add("tools[].", definition.key(k));
      definition = *function;
    }
    auto name = definition.find("name");
    if (!name || !name->is_string() || name->string().empty())
      return Bad(where + ".name must be nonempty", field);
    chat::ToolDefinition parsed;
    parsed.name = std::string(name->string());
    // The template receives the function object as the client sent it.
    AppendJson(definition, parsed.function_json);
    if (memory != nullptr && !charge.Add(*memory, parsed.function_json.capacity()))
      return std::unexpected(
          MemoryRefusal(*memory, parsed.function_json.capacity(), "tool definitions"));
    for (std::size_t k = 0; k < definition.size(); ++k) {
      auto key = definition.key(k);
      auto value = definition.member(k);
      if (key == "name") continue;
      if (key == "description") {
        if (!value.is_string()) return Bad(where + ".description must be a string", field);
        parsed.description = std::string(value.string());
      } else if (key == "parameters") {
        if (!value.is_object()) return Bad(where + ".parameters must be an object", field);
        parsed.parameters_json.clear();
        AppendJson(value, parsed.parameters_json);
      } else if (key == "strict") {
        if (!value.is_bool()) return Bad(where + ".strict must be a boolean", field);
        parsed.strict = value.boolean();
      } else
        ignored.Add(legacy ? "functions[]." : "tools[].function.", key);
    }
    out.push_back(std::move(parsed));
  }
  return out;
}

std::expected<std::shared_ptr<const ToolOptions>, Error> ParseTools(json::Value root,
                                                                    Ignored& ignored,
                                                                    RequestMemory* memory,
                                                                    MemoryCharge& charge) {
  auto tools = root.find("tools"), functions = root.find("functions");
  auto modern_choice = root.find("tool_choice"), legacy_choice = root.find("function_call");
  auto parallel = root.find("parallel_tool_calls");
  const auto present = [](const std::optional<json::Value>& v) { return v && !v->is_null(); };
  if (!present(tools) && !present(functions) && !present(modern_choice) &&
      !present(legacy_choice) && !present(parallel))
    return std::shared_ptr<const ToolOptions>();
  std::vector<chat::ToolDefinition> definitions;
  if (present(tools) || present(functions)) {
    auto parsed = ParseDefinitions(present(tools) ? *tools : *functions, !present(tools), ignored,
                                   memory, charge);
    if (!parsed) return std::unexpected(parsed.error());
    definitions = std::move(*parsed);
  }
  if (present(tools) && present(functions)) {
    auto legacy = ParseDefinitions(*functions, true, ignored, memory, charge);
    if (!legacy) return std::unexpected(legacy.error());
    if (definitions.size() != legacy->size())
      return Bad("tools and functions conflict", "functions");
    // Each definition's legacy twin, by name (the first of a repeated name).
    MemoryCharge index_charge;
    const std::uint64_t index_bytes = legacy->size() * kNameIndexEntryBytes;
    if (memory != nullptr && !index_charge.Add(*memory, index_bytes))
      return std::unexpected(MemoryRefusal(*memory, index_bytes, "tool definition comparison"));
    std::unordered_map<std::string_view, std::size_t> by_name;
    by_name.reserve(legacy->size());
    for (std::size_t i = 0; i < legacy->size(); ++i) by_name.try_emplace((*legacy)[i].name, i);
    for (const auto& definition : definitions) {
      const auto found = by_name.find(definition.name);
      if (found == by_name.end()) return Bad("tools and functions conflict", "functions");
      const auto other = legacy->begin() + static_cast<std::ptrdiff_t>(found->second);
      if (definition.description != other->description || definition.strict != other->strict)
        return Bad("tools and functions conflict", "functions");
      MemoryCharge comparison;
      const auto bytes = json::ParseWorkingBytes(definition.parameters_json) +
                         json::ParseWorkingBytes(other->parameters_json);
      if (memory != nullptr && !comparison.Add(*memory, bytes))
        return std::unexpected(MemoryRefusal(*memory, bytes, "tool definition comparison"));
      auto a = json::Parse(definition.parameters_json), b = json::Parse(other->parameters_json);
      if (!a || !b || !SameJson(a->root(), b->root()))
        return Bad("tools and functions conflict", "functions");
    }
  }
  chat::ToolChoice choice{
      definitions.empty() ? chat::ToolChoiceKind::kNone : chat::ToolChoiceKind::kAuto, {}};
  if (present(modern_choice) || present(legacy_choice)) {
    auto parsed = ParseToolChoice(present(modern_choice) ? *modern_choice : *legacy_choice,
                                  !present(modern_choice), ignored);
    if (!parsed) return std::unexpected(parsed.error());
    choice = std::move(*parsed);
  }
  if (present(modern_choice) && present(legacy_choice)) {
    auto parsed = ParseToolChoice(*legacy_choice, true, ignored);
    if (!parsed) return std::unexpected(parsed.error());
    if (choice.kind != parsed->kind || choice.name != parsed->name)
      return Bad("tool_choice and function_call conflict", "function_call");
  }
  // `auto` with nothing declared is no tools, as when nothing is sent (a
  // required or named choice still needs its definitions).
  if (definitions.empty() && choice.kind == chat::ToolChoiceKind::kAuto)
    choice.kind = chat::ToolChoiceKind::kNone;
  if (present(parallel) && !parallel->is_bool())
    return Bad("parallel_tool_calls must be a boolean", "parallel_tool_calls");
  const auto bound = chat::ToolRequestBytes(definitions);
  if (!bound || *bound > std::numeric_limits<std::uint64_t>::max() - sizeof(ToolOptions) -
                             choice.name.capacity())
    return Bad("tool definitions exceed addressable memory", "tools");
  const std::uint64_t bytes = *bound + sizeof(ToolOptions) + choice.name.capacity();
  MemoryCharge owned;
  if (memory != nullptr && !owned.Add(*memory, bytes))
    return std::unexpected(MemoryRefusal(*memory, bytes, "normalized tool definitions"));
  // The paid bound covers canonical envelope expansion as well as the source fields.
  const json::Limits limits{.max_bytes = *bound,
                            .max_depth = kMaxJsonDepth,
                            .max_values = *bound,
                            .max_string_bytes = *bound};
  auto request = chat::ToolRequest::Create(definitions, std::move(choice),
                                           !present(parallel) || parallel->boolean(), limits);
  if (!request)
    return Bad(request.error(),
               present(tools) ? "tools" : (present(functions) ? "functions" : "tool_choice"));
  return std::make_shared<const ToolOptions>(ToolOptions{std::move(*request), std::move(owned)});
}

// The stop strings, as one shared matcher. Any number and length: their
// bytes are the body's, and matching costs a few steps an output byte
// whatever they are (StopMatcher, D-102). The list and the matcher's tables
// are charged to `charge` (the parse's) as they are made.
std::expected<std::shared_ptr<const StopMatcher>, Error> ParseStop(const json::Value& v,
                                                                   RequestMemory* memory,
                                                                   MemoryCharge& charge) {
  std::vector<std::string_view> stops;
  const auto one = [&](const json::Value& s, const std::string& where) -> std::optional<Error> {
    if (!s.is_string()) {
      return Bad(where + " must be a string", where).error();
    }
    if (s.string().empty()) {
      return Bad(where + " must not be empty", where).error();
    }
    stops.push_back(s.string());  // the document's, which outlives the build
    return std::nullopt;
  };
  if (v.is_string()) {
    if (auto e = one(v, "stop")) {
      return std::unexpected(*e);
    }
  } else if (!v.is_array()) {
    return Bad("stop must be a string or an array of strings", "stop");
  } else {
    if (memory != nullptr && !charge.Add(*memory, v.size() * sizeof(std::string_view))) {
      return std::unexpected(MemoryRefusal(*memory, v.size() * sizeof(std::string_view), "stop"));
    }
    stops.reserve(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (auto e = one(v.at(i), std::format("stop[{}]", i))) {
        return std::unexpected(*e);
      }
    }
  }
  if (stops.empty()) {
    return std::shared_ptr<const StopMatcher>();
  }
  // Its tables charged as they grow: a node a distinct prefix, so a
  // million copies of one stop string cost what one does.
  auto built = StopMatcher::Build(stops, memory, &charge);
  if (!built) {
    // What the request needed by then: more than the pool can never fit.
    return std::unexpected(MemoryRefusal(*memory, charge.bytes() + built.error(), "stop"));
  }
  return std::make_shared<const StopMatcher>(std::move(*built));
}

std::expected<std::uint32_t, Error> ParseMaxTokens(const json::Value& v, std::string_view name,
                                                   std::int64_t least = 1) {
  const std::optional<std::int64_t> n = v.int64();
  if (!v.is_integer() || !n || *n < least || std::cmp_greater(*n, kMaxTokensCeiling)) {
    return Bad(std::format("{} must be an integer from {} to {}", name, least, kMaxTokensCeiling),
               std::string(name));
  }
  return static_cast<std::uint32_t>(*n);
}

// The fields this route knows, whose null means their default.
constexpr std::array<std::string_view, 31> kKnown = {"max_tokens",
                                                     "max_completion_tokens",
                                                     "temperature",
                                                     "top_p",
                                                     "top_k",
                                                     "min_p",
                                                     "seed",
                                                     "stop",
                                                     "stream",
                                                     "stream_options",
                                                     "n",
                                                     "presence_penalty",
                                                     "frequency_penalty",
                                                     "repetition_penalty",
                                                     "logprobs",
                                                     "top_logprobs",
                                                     "tools",
                                                     "tool_choice",
                                                     "functions",
                                                     "function_call",
                                                     "response_format",
                                                     "logit_bias",
                                                     "modalities",
                                                     "audio",
                                                     "user",
                                                     "safety_identifier",
                                                     "prompt_cache_key",
                                                     "metadata",
                                                     "service_tier",
                                                     "parallel_tool_calls",
                                                     "store"};

std::expected<ChatRequest, Error> ParseRequest(std::string_view body, CompletionRequest* literal,
                                               RequestMemory* memory) {
  // The server bounds the body's bytes (its max_body, from memory). Every
  // value takes at least a byte, so a body's own size bounds its values and
  // strings: a token-ID prompt is one value a token. The parser's 32-bit
  // offsets bound a body (config::kMaxBodyCeiling); depth is its stack's.
  // The working set is charged before parsing: the document's tables and
  // strings, and the request built from them (its strings are the body's
  // at most; a prompt ID, four bytes, takes at least two of the body's).
  MemoryCharge charge;
  if (memory != nullptr) {
    const std::uint64_t working =
        std::uint64_t{json::ParseWorkingBytes(body)} + (std::uint64_t{2} * body.size());
    if (!charge.Add(*memory, working)) {
      return std::unexpected(MemoryRefusal(*memory, working, "the request body"));
    }
  }
  const std::size_t bytes = std::min<std::size_t>(body.size(), config::kMaxBodyCeiling);
  const json::Limits limits{.max_bytes = static_cast<std::size_t>(config::kMaxBodyCeiling),
                            .max_depth = kMaxJsonDepth,
                            .max_values = bytes + 1,
                            .max_string_bytes = bytes + 1};
  auto document = json::Parse(body, limits);
  if (!document) {
    return Bad(std::format("the body is not valid JSON: {}", document.error().ToString()));
  }
  const json::Value root = document->root();
  if (!root.is_object()) {
    return Bad("the body must be a JSON object");
  }
  ChatRequest request;
  Ignored ignored(request.ignored);
  bool has_model = false;
  bool has_messages = false;
  bool has_prompt = false;
  std::optional<std::uint32_t> max_tokens;
  std::optional<std::uint32_t> max_completion_tokens;
  for (std::size_t k = 0; k < root.size(); ++k) {
    const std::string_view key = root.key(k);
    const json::Value v = root.member(k);
    const std::string name(key);
    if (key == "transforms" || key == "plugins") {
      // OpenRouter's, refused at the wire whatever their value (D-046).
      return Bad(std::format("{} is not supported: this is not a routing service", key), name);
    }
    if (key == "model") {
      if (!v.is_string() || v.string().empty() || v.string().size() > kMaxModelBytes) {
        return Bad(std::format("model must be a string of 1 to {} bytes", kMaxModelBytes), "model");
      }
      request.model = std::string(v.string());
      has_model = true;
    } else if (literal != nullptr && key == "prompt") {
      has_prompt = true;
      if (v.is_string()) {
        literal->prompt = std::string(v.string());  // its bytes are the body's
      } else if (v.is_array() && v.size() > 0) {
        // One value a token, bounded by the body; the model's context decides.
        literal->token_ids.reserve(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
          const auto token = v.at(i).int64();
          if (!v.at(i).is_integer() || !token || *token < 0 ||
              *token > std::numeric_limits<std::int32_t>::max()) {
            return Bad(
                "prompt must be one string or one nonempty array of nonnegative int32 token IDs",
                name);
          }
          literal->token_ids.push_back(static_cast<std::int32_t>(*token));
        }
      } else {
        return Bad(
            "prompt must be one string or one nonempty token-ID array; batches are not supported",
            name);
      }
    } else if (literal != nullptr && (key == "echo" || key == "add_special_tokens" ||
                                      key == "return_tokens_as_token_ids")) {
      if (!v.is_null()) {
        if (!v.is_bool()) {
          return Bad(name + " must be a boolean", name);
        }
        if (key == "echo") {
          literal->echo = v.boolean();
        } else if (key == "add_special_tokens") {
          literal->add_special_tokens = v.boolean();
        } else {
          literal->return_tokens_as_token_ids = v.boolean();
        }
      }
    } else if (literal != nullptr && (key == "logprobs" || key == "prompt_logprobs")) {
      if (!v.is_null()) {
        // Up to the vocabulary: the largest a tokenizer accepts here, the
        // model's own and the response's bytes at admission (D-102).
        const auto count = v.int64();
        if (!v.is_integer() || !count || *count < 0 ||
            std::cmp_greater(*count, kMaxTopLogprobsAtParse)) {
          return Bad(
              std::format("{} must be an integer from 0 to the model's vocabulary size", name),
              name);
        }
        (key == "logprobs" ? literal->logprobs : literal->prompt_logprobs) =
            static_cast<std::uint32_t>(*count);
      }
    } else if (literal != nullptr && key == "best_of") {
      if (!v.is_null() && !NumberIs(v, 1)) {
        return Bad("best_of must be 1", name);
      }
    } else if (literal != nullptr && key == "suffix") {
      if (!v.is_null()) {
        return Bad("suffix insertion is not supported", name);
      }
    } else if (literal != nullptr && (key == "truncate_prompt_tokens" || key == "prompt_embeds" ||
                                      key == "allowed_token_ids" || key == "logprob_token_ids")) {
      if (!v.is_null()) {
        return Bad(
            name + " is not supported; no prompt truncation or distribution restriction is applied",
            name);
      }
    } else if (literal != nullptr && (key == "ignore_eos" || key == "use_beam_search")) {
      if (!v.is_null() && (!v.is_bool() || v.boolean())) {
        return Bad(name + " must be false", name);
      }
    } else if (literal != nullptr && key == "min_tokens") {
      if (!v.is_null() && !NumberIs(v, 0)) {
        return Bad("min_tokens must be 0", name);
      }
    } else if (literal != nullptr && key == "skip_special_tokens") {
      if (!v.is_null() && (!v.is_bool() || !v.boolean())) {
        return Bad("skip_special_tokens must be true", name);
      }
    } else if (key == "messages") {
      if (literal != nullptr) {
        return Bad("messages is not accepted by the literal completions route", name);
      }
      if (!v.is_array() || v.size() == 0) {
        return Bad("messages must be a non-empty array", "messages");
      }
      // Any number: their bytes are the body's, their tokens the context's.
      const std::uint64_t message_bytes = v.size() * sizeof(Message);
      if (memory != nullptr && !charge.Add(*memory, message_bytes)) {
        return std::unexpected(MemoryRefusal(*memory, message_bytes, "messages"));
      }
      request.messages.reserve(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) {
        auto m = ParseMessage(v.at(i), i, ignored, memory, charge);
        if (!m) {
          return std::unexpected(m.error());
        }
        request.messages.push_back(std::move(*m));
      }
      has_messages = true;
    } else if (std::ranges::find(kKnown, key) == kKnown.end()) {
      ignored.Add({}, key);  // unknown: ignored, its name counted
    } else if (v.is_null()) {
      continue;  // a known field's default
    } else if (key == "max_tokens") {
      auto n = ParseMaxTokens(v, key, literal != nullptr ? 0 : 1);
      if (!n) {
        return std::unexpected(n.error());
      }
      max_tokens = *n;
    } else if (key == "max_completion_tokens") {
      auto n = ParseMaxTokens(v, key, literal != nullptr ? 0 : 1);
      if (!n) {
        return std::unexpected(n.error());
      }
      max_completion_tokens = *n;
    } else if (key == "temperature") {
      const std::optional<double> t = v.is_number() ? v.float64() : std::nullopt;
      if (!t || !std::isfinite(*t) || *t < 0 || *t > kMaxTemperature) {
        return Bad(std::format("temperature must be a number from 0 to {}", kMaxTemperature), name);
      }
      request.temperature = *t;
      request.sent |= ChatRequest::kSentTemperature;
    } else if (key == "top_p") {
      const std::optional<double> p = v.is_number() ? v.float64() : std::nullopt;
      // The sampler takes a float: a value that rounds to 0 there would be
      // refused mid-generation, a node failure, so it is refused here.
      if (!p || !std::isfinite(*p) || *p <= 0 || *p > 1 || static_cast<float>(*p) <= 0.0F) {
        return Bad("top_p must be a number greater than 0 and at most 1", name);
      }
      request.top_p = *p;
      request.sent |= ChatRequest::kSentTopP;
    } else if (key == "top_k") {
      const std::optional<std::int64_t> n = v.int64();
      if (!v.is_integer() || !n || *n < -1 || *n >= kMaxTopK) {
        return Bad(std::format("top_k must be an integer from -1 (or 0: off) to {}", kMaxTopK - 1),
                   name);
      }
      request.top_k = *n < 0 ? 0U : static_cast<std::uint32_t>(*n);
      request.sent |= ChatRequest::kSentTopK;
    } else if (key == "min_p") {
      const std::optional<double> p = v.is_number() ? v.float64() : std::nullopt;
      if (!p || !std::isfinite(*p) || *p < 0 || *p > 1) {
        return Bad("min_p must be a number from 0 to 1", name);
      }
      request.min_p = *p;
      request.sent |= ChatRequest::kSentMinP;
    } else if (key == "seed") {
      const std::optional<std::int64_t> s = v.int64();
      if (!v.is_integer() || !s) {
        return Bad("seed must be an integer that fits in 64 signed bits", name);
      }
      request.seed = static_cast<std::uint64_t>(*s);
    } else if (key == "stop") {
      auto stops = ParseStop(v, memory, charge);
      if (!stops) {
        return std::unexpected(stops.error());
      }
      request.stop = std::move(*stops);
    } else if (key == "stream") {
      if (!v.is_bool()) {
        return Bad("stream must be a boolean", name);
      }
      request.stream = v.boolean();
    } else if (key == "stream_options") {
      if (!v.is_object()) {
        return Bad("stream_options must be an object", name);
      }
      for (std::size_t j = 0; j < v.size(); ++j) {
        const std::string where = std::format("stream_options.{}", v.key(j));
        const bool usage = v.key(j) == "include_usage";
        if (!usage && v.key(j) != "include_obfuscation") {
          ignored.Add("stream_options.", v.key(j));
          continue;
        }
        if (!v.member(j).is_bool()) {
          return Bad(where + " must be a boolean", where);
        }
        if (usage) {
          request.include_usage = v.member(j).boolean();
        }  // include_obfuscation: none is sent on this route
      }
    } else if (key == "n") {
      if (!NumberIs(v, 1)) {
        return Bad("n must be 1: this route returns one choice", name);
      }
    } else if (key == "presence_penalty" || key == "frequency_penalty") {
      if (!NumberIs(v, 0)) {
        return Bad(std::format("{} is not supported by this route yet; only 0 is accepted", key),
                   name);
      }
    } else if (key == "repetition_penalty") {
      if (!NumberIs(v, 1)) {
        return Bad("repetition_penalty is not supported by this route yet; only 1 is accepted",
                   name);
      }
    } else if (key == "logprobs") {
      if (!v.is_bool() || v.boolean()) {
        return Bad("logprobs are not supported by this route yet", name);
      }
    } else if (key == "top_logprobs") {
      if (!NumberIs(v, 0)) {
        return Bad("logprobs are not supported by this route yet", name);
      }
    } else if (key == "tools" || key == "functions" || key == "tool_choice" ||
               key == "function_call") {
      // Normalize the related modern and legacy fields together after parsing.
    } else if (key == "response_format") {
      auto type = v.is_object() && v.size() == 1 ? v.find("type") : std::nullopt;
      if (!type || !type->is_string() || type->string() != "text") {
        return Bad(
            "response_format must be {\"type\": \"text\"}: structured output is not "
            "supported by this route yet",
            name);
      }
    } else if (key == "logit_bias") {
      if (!v.is_object() || v.size() != 0) {
        return Bad("logit_bias is not supported by this route yet", name);
      }
    } else if (key == "modalities") {
      if (!v.is_array() || v.size() != 1 || !v.at(0).is_string() || v.at(0).string() != "text") {
        return Bad("modalities must be [\"text\"]", name);
      }
    } else if (key == "audio") {
      return Bad("audio output is not supported by this route", name);
    } else if (key == "user" || key == "safety_identifier" || key == "prompt_cache_key" ||
               key == "service_tier") {
      if (!v.is_string()) {
        return Bad(std::format("{} must be a string", key), name);
      }
    } else if (key == "metadata") {
      if (!v.is_object()) {
        return Bad("metadata must be an object", name);
      }
    } else if (key == "parallel_tool_calls") {
      if (!v.is_bool()) {
        return Bad("parallel_tool_calls must be a boolean", name);
      }
    } else if (key == "store") {
      if (!v.is_bool() || v.boolean()) {
        return Bad("store must be false: nothing is stored", name);
      }
    }
  }
  if (!has_model) {
    return Bad("model is required", "model");
  }
  if (literal != nullptr && !has_prompt) {
    return Bad("prompt is required", "prompt");
  }
  if (literal == nullptr && !has_messages) {
    return Bad("messages is required", "messages");
  }
  if (max_tokens && max_completion_tokens && *max_tokens != *max_completion_tokens) {
    return Bad("max_tokens and max_completion_tokens differ; give one", "max_completion_tokens");
  }
  request.max_tokens = max_completion_tokens ? max_completion_tokens : max_tokens;
  if (literal == nullptr && request.messages.back().role != Role::kUser &&
      request.messages.back().role != Role::kTool) {
    return Bad("the last message must be the user's or a tool result", "messages");
  }
  auto tools = ParseTools(root, ignored, memory, charge);
  if (!tools) return std::unexpected(tools.error());
  request.tools = std::move(*tools);
  if (literal != nullptr && request.tools &&
      request.tools->request.choice().kind != chat::ToolChoiceKind::kNone)
    return Bad("tools are not supported by the literal completions route", "tools");
  return request;
}

}  // namespace

Error MemoryRefusal(const RequestMemory& memory, std::uint64_t bytes, std::string_view what) {
  if (bytes > memory.capacity()) {
    return Error{
        .status = 413,
        .type = "invalid_request_error",
        .message = std::format("{} needs {} bytes of request memory, more than the node's {} "
                               "([client] request_memory_bytes)",
                               what, bytes, memory.capacity()),
        .param = {},
        .code = "request_too_large"};
  }
  return Error{
      .status = 503,
      .type = "server_error",
      .message = std::format("{} needs {} bytes of request memory, which other requests hold now "
                             "({} of {}; [client] request_memory_bytes); retry",
                             what, bytes, memory.used(), memory.capacity()),
      .param = {},
      .code = "request_memory_busy"};
}

std::uint64_t RequestBytes(const ChatRequest& request) {
  // ToolOptions owns its separate prepaid charge across all request copies.
  std::uint64_t bytes = sizeof(ChatRequest) + request.model.capacity();
  bytes += request.messages.capacity() * sizeof(Message);
  for (const Message& m : request.messages) {
    bytes += m.content.StorageBytes() + (m.reasoning ? m.reasoning->capacity() : 0);
  }
  bytes += request.ignored.capacity() * sizeof(std::string);
  for (const std::string& name : request.ignored) {
    bytes += name.capacity();
  }
  if (request.stop) {
    bytes += request.stop->bytes();
  }
  return bytes;
}

std::expected<std::shared_ptr<const PromptTokens>, Error> PromptTokens::Hold(
    std::vector<std::int32_t> tokens, RequestMemory& memory) {
  auto held = std::make_shared<PromptTokens>();
  const std::uint64_t bytes = sizeof(PromptTokens) + tokens.capacity() * sizeof(std::int32_t);
  if (!held->charge_.Add(memory, bytes)) {
    return std::unexpected(MemoryRefusal(memory, bytes, "the prepared prompt tokens"));
  }
  held->tokens_ = std::move(tokens);
  return held;
}

std::uint64_t RequestBytes(const CompletionRequest& request) {
  return RequestBytes(request.options) + (sizeof(CompletionRequest) - sizeof(ChatRequest)) +
         (request.prompt ? request.prompt->capacity() : 0) +
         (request.token_ids.capacity() * sizeof(std::int32_t));
}

std::expected<ChatRequest, Error> ParseChatRequest(std::string_view body, RequestMemory* memory) {
  return ParseRequest(body, nullptr, memory);
}

std::expected<CompletionRequest, Error> ParseCompletionRequest(std::string_view body,
                                                               RequestMemory* memory) {
  CompletionRequest literal;
  auto options = ParseRequest(body, &literal, memory);
  if (!options) {
    return std::unexpected(options.error());
  }
  if (options->stream) {
    return Bad("literal completions currently support stream=false only", "stream");
  }
  if (!options->max_tokens) {
    options->max_tokens = 16;  // legacy OpenAI completion default
  }
  literal.options = std::move(*options);
  return literal;
}

// ---------------------------------------------------------------- IgnoredFields

std::vector<std::string> IgnoredFields::Record(const std::vector<std::string>& names,
                                               std::int64_t now) {
  std::vector<std::string> fresh;
  const std::scoped_lock lock(mutex_);
  for (const std::string& name : names) {
    auto it = names_.find(name);
    if (it == names_.end()) {
      if (names_.size() >= kMaxNames) {
        ++unrecorded_;
        continue;
      }
      it = names_.emplace(name, Entry{.count = 0, .first = now, .last = now}).first;
      fresh.push_back(name);
    }
    ++it->second.count;
    it->second.last = now;
  }
  return fresh;
}

std::string IgnoredFields::Json() const {
  const std::scoped_lock lock(mutex_);
  std::string data;
  for (const auto& [name, entry] : names_) {
    data +=
        std::format(R"({}{{"name":{},"count":{},"first_seen":{},"last_seen":{}}})",
                    data.empty() ? "" : ",", Quoted(name), entry.count, entry.first, entry.last);
  }
  return std::format(R"({{"object":"list","data":[{}],"unrecorded":{}}})", data, unrecorded_);
}

// ---------------------------------------------------------------- output

std::string ErrorJson(const Error& error) {
  return std::format(R"({{"error":{{"message":{},"type":{},"param":{},"code":{}}}}})",
                     Quoted(error.message), Quoted(error.type),
                     error.param.empty() ? "null" : Quoted(error.param),
                     error.code.empty() ? "null" : Quoted(error.code));
}

namespace {

std::string_view FinishName(Finish finish) {
  return finish == Finish::kStop ? "stop" : finish == Finish::kToolCalls ? "tool_calls" : "length";
}

std::string UsageJson(const Usage& u) {
  return std::format(R"({{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{},)"
                     R"("prompt_tokens_details":{{"cached_tokens":{}}}}})",
                     u.prompt_tokens, u.completion_tokens,
                     std::uint64_t{u.prompt_tokens} + std::uint64_t{u.completion_tokens},
                     u.cached_tokens);
}

}  // namespace

std::optional<std::size_t> CompletionJsonBytes(std::string_view id, std::string_view model,
                                               std::string_view content,
                                               std::optional<std::string_view> reasoning,
                                               std::span<const chat::ToolUsePart> calls) {
  // Fixed keys, punctuation, bounded-width integer formatting and its temporary
  // usage string. The variable payload is counted with its actual escaping.
  std::size_t bytes = 1024;
  const auto quoted = [&](std::string_view value) {
    if (__builtin_add_overflow(bytes, std::size_t{2}, &bytes)) return false;
    for (const char ch : value) {
      const auto c = static_cast<unsigned char>(ch);
      const std::size_t width =
          c == '"' || c == '\\' || c == '\n' || c == '\r' || c == '\t' || c == '\b' || c == '\f' ? 2
          : c < 0x20                                                                             ? 6
                     : 1;
      if (__builtin_add_overflow(bytes, width, &bytes)) return false;
    }
    return true;
  };
  if (!quoted(id) || !quoted(model) || !quoted(content) || (reasoning && !quoted(*reasoning)))
    return std::nullopt;
  for (const auto& call : calls) {
    if (__builtin_add_overflow(bytes, std::size_t{128}, &bytes) || !quoted(call.id) ||
        !quoted(call.name) || !quoted(call.input_json))
      return std::nullopt;
  }
  return bytes;
}

std::string CompletionJson(std::string_view id, std::int64_t created, std::string_view model,
                           std::string_view content, std::optional<std::string_view> reasoning,
                           Finish finish, const Usage& usage,
                           std::span<const chat::ToolUsePart> calls) {
  const auto bound = CompletionJsonBytes(id, model, content, reasoning, calls);
  base::Check(bound.has_value(), "completion serialization exceeds addressable storage");
  std::string out;
  out.reserve(*bound);
  out = "{\"id\":";
  json::AppendQuoted(id, out);
  out += std::format(",\"object\":\"chat.completion\",\"created\":{},\"model\":", created);
  json::AppendQuoted(model, out);
  out += ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":";
  // A reply of tool calls alone has no content (null, as OpenAI sends it);
  // the fixed allowance in CompletionJsonBytes covers the two extra bytes.
  if (content.empty() && !calls.empty())
    out += "null";
  else
    json::AppendQuoted(content, out);
  if (reasoning) {
    out += ",\"reasoning\":";
    json::AppendQuoted(*reasoning, out);
  }
  if (!calls.empty()) {
    out += ",\"tool_calls\":[";
    for (std::size_t i = 0; i < calls.size(); ++i) {
      if (i != 0) out += ',';
      out += "{\"id\":";
      json::AppendQuoted(calls[i].id, out);
      out += ",\"type\":\"function\",\"function\":{\"name\":";
      json::AppendQuoted(calls[i].name, out);
      out += ",\"arguments\":";
      json::AppendQuoted(calls[i].input_json, out);
      out += "}}";
    }
    out += ']';
  }
  out += "},\"logprobs\":null,\"finish_reason\":\"";
  out += FinishName(finish);
  out += "\"}],\"usage\":";
  out += UsageJson(usage);
  out += '}';
  base::Check(out.size() <= *bound, "completion serialization exceeded its paid bound");
  return out;
}

std::expected<std::string, Error> LiteralCompletionJson(std::string_view id, std::int64_t created,
                                                        const CompletionRequest& request,
                                                        std::string_view content,
                                                        const LiteralResult& result, Finish finish,
                                                        const Usage& usage, std::size_t max_bytes) {
  std::size_t allowance = 1024;
  const auto charge = [&](std::size_t bytes) {
    if (allowance > max_bytes || bytes > max_bytes - allowance) {
      return false;
    }
    allowance += bytes;
    return true;
  };
  const auto charge_text = [&](std::string_view text) {
    return text.size() <= max_bytes / 6 && charge(6 * text.size());
  };
  // A row's top scores: those asked for, and the supplied token beside them.
  const std::size_t most_top =
      std::size_t{std::max(
          {request.logprobs.value_or(0), request.prompt_logprobs.value_or(0), std::uint32_t{1}})} +
      1;
  const auto charge_rows = [&](std::span<const TokenLogprob> rows) {
    for (const auto& row : rows) {
      if (!charge(kScoreRowBytes) || !charge_text(row.token) || row.top.size() > most_top ||
          (row.logprob && (!std::isfinite(*row.logprob) || *row.logprob > 0))) {
        return false;
      }
      for (const auto& score : row.top) {
        if (!charge(kTopScoreBytes) || !charge_text(score.token) || !std::isfinite(score.logprob) ||
            score.logprob > 0 || score.rank == 0) {
          return false;
        }
      }
    }
    return true;
  };
  if (!charge_text(content) || (request.echo && !charge_text(result.prompt_text)) ||
      (request.logprobs && !charge_rows(result.logprobs)) ||
      (request.prompt_logprobs && !charge_rows(result.prompt_logprobs))) {
    return std::unexpected(Error{.status = 413,
                                 .type = "invalid_request_error",
                                 .message = "completion scores exceed the finite bounded response",
                                 .param = "logprobs",
                                 .code = "response_too_large"});
  }
  const auto arrays = [&](std::span<const TokenLogprob> rows) {
    std::string tokens = "[";
    std::string scores = "[";
    std::string top = "[";
    std::string offsets = "[";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto& row = rows[i];
      if (i != 0) {
        tokens += ',';
        scores += ',';
        top += ',';
        offsets += ',';
      }
      tokens += Quoted(row.token);
      scores += row.logprob ? std::format("{:.17g}", *row.logprob) : "null";
      offsets += std::to_string(row.text_offset);
      if (!row.logprob) {
        top += "null";
      } else {
        top += '{';
        // Different byte tokens may decode to the same replacement text.
        // Keep the highest score for a key so top-1/greedy remains correct;
        // return_tokens_as_token_ids avoids that representational collision.
        std::map<std::string, double> unique;
        for (std::size_t j = 0; j < row.top.size(); ++j) {
          const auto& score = row.top[j];
          if (j >= *request.logprobs && score.id != row.id) {
            continue;
          }
          const auto [at, inserted] = unique.try_emplace(score.token, score.logprob);
          if (!inserted) {
            at->second = std::max(at->second, score.logprob);
          }
        }
        bool first = true;
        for (const auto& [token, score] : unique) {
          if (!first) {
            top += ',';
          }
          first = false;
          top += Quoted(token) + ':' + std::format("{:.17g}", score);
        }
        top += '}';
      }
    }
    return "{\"tokens\":" + tokens + "],\"token_logprobs\":" + scores +
           "],\"top_logprobs\":" + top + "],\"text_offset\":" + offsets + "]}";
  };
  std::string text = request.echo ? result.prompt_text : std::string();
  text += content;
  std::string prompt_scores;
  if (request.prompt_logprobs) {
    prompt_scores = ",\"prompt_logprobs\":[";
    for (std::size_t i = 0; i < result.prompt_logprobs.size(); ++i) {
      if (i != 0) {
        prompt_scores += ',';
      }
      const auto& row = result.prompt_logprobs[i];
      if (!row.logprob) {
        prompt_scores += "null";
      } else {
        prompt_scores += '{';
        // vLLM's prompt scores are keyed by token ID, independently of the
        // legacy token text arrays. The supplied ID is always present.
        bool first = true;
        for (std::size_t j = 0; j < row.top.size(); ++j) {
          const auto& score = row.top[j];
          if (j >= *request.prompt_logprobs && score.id != row.id) {
            continue;
          }
          if (!first) {
            prompt_scores += ',';
          }
          first = false;
          prompt_scores += std::format(R"({}:{{"logprob":{:.17g},"rank":{},"decoded_token":{}}})",
                                       Quoted(std::to_string(score.id)), score.logprob, score.rank,
                                       Quoted(score.token));
        }
        prompt_scores += '}';
      }
    }
    prompt_scores += ']';
  }
  std::string body = std::format(
      R"({{"id":{},"object":"text_completion","created":{},"model":{},"choices":[{{"index":0,"text":{},"logprobs":{},"finish_reason":"{}"{}}}],"usage":{}}})",
      Quoted(id), created, Quoted(request.options.model), Quoted(text),
      request.logprobs ? arrays(result.logprobs) : "null", FinishName(finish), prompt_scores,
      UsageJson(usage));
  if (body.size() > max_bytes) {
    return std::unexpected(Error{.status = 413,
                                 .type = "invalid_request_error",
                                 .message = "completion exceeds the bounded response size",
                                 .param = "logprobs",
                                 .code = "response_too_large"});
  }
  return body;
}

std::string ChunkJson(std::string_view id, std::int64_t created, std::string_view model,
                      Delta delta, std::string_view text, Finish finish) {
  std::string body;
  switch (delta) {
    case Delta::kRole:
      body = R"({"role":"assistant","content":""})";
      break;
    case Delta::kReasoning:
      body = "{\"reasoning\":" + Quoted(text) + "}";
      break;
    case Delta::kContent:
      body = "{\"content\":" + Quoted(text) + "}";
      break;
    case Delta::kFinish:
      body = "{}";
      break;
  }
  return std::format(
      R"({{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{)"
      R"("index":0,"delta":{},"logprobs":null,"finish_reason":{}}}]}})",
      Quoted(id), created, Quoted(model), body,
      delta == Delta::kFinish ? Quoted(FinishName(finish)) : std::string("null"));
}

std::string UsageChunkJson(std::string_view id, std::int64_t created, std::string_view model,
                           const Usage& usage) {
  return std::format(
      R"({{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[],)"
      R"("usage":{}}})",
      Quoted(id), created, Quoted(model), UsageJson(usage));
}

std::string ModelJson(const ModelInfo& model, std::int64_t created) {
  return std::format(R"({{"id":{},"object":"model","created":{},"owned_by":"llmp"}})",
                     Quoted(model.name), created);
}

std::string ModelsJson(const std::vector<ModelInfo>& models, std::int64_t created) {
  std::string data;
  for (const ModelInfo& m : models) {
    data += (data.empty() ? "" : ",") + ModelJson(m, created);
  }
  return std::format(R"({{"object":"list","data":[{}]}})", data);
}

// ---------------------------------------------------------------- OutputText

OutputText::Out OutputText::Reasoning(std::string_view text) {
  if (stopped_) {
    return {};
  }
  had_reasoning_ = true;
  return {.reasoning = std::string(text), .content = {}};
}

// ---------------------------------------------------------------- StopMatcher

std::uint64_t StopMatcher::BuildBytes(std::size_t count, std::size_t bytes) {
  // A node a byte at most, and the root, its table grown by doubling (every
  // allocation charged as it is made); the breadth-first queue beside.
  static_cast<void>(count);
  return sizeof(StopMatcher) +
         ((std::uint64_t{bytes} + 1) * ((2 * sizeof(Node)) + sizeof(std::uint32_t)));
}

std::size_t StopMatcher::bytes() const {
  return sizeof(StopMatcher) + (nodes_.capacity() * sizeof(Node));
}

std::uint32_t StopMatcher::Child(std::uint32_t node, unsigned char byte) const {
  if (node == 0) {
    return root_[byte];
  }
  for (std::uint32_t c = nodes_[node].child; c != 0; c = nodes_[c].sibling) {
    if (nodes_[c].byte == byte) {
      return c;
    }
  }
  return 0;
}

std::uint32_t StopMatcher::Next(std::uint32_t state, unsigned char byte) const {
  for (;;) {
    const std::uint32_t c = Child(state, byte);
    if (c != 0 || state == 0) {
      return c;
    }
    state = nodes_[state].fail;
  }
}

StopMatcher StopMatcher::Build(std::span<const std::string_view> stops) {
  auto built = Build(stops, nullptr, nullptr);
  return built ? std::move(*built) : StopMatcher();  // uncharged: always built
}

std::expected<StopMatcher, std::uint64_t> StopMatcher::Build(
    std::span<const std::string_view> stops, RequestMemory* memory, MemoryCharge* charge) {
  // Each allocation charged before it is made: the table grows by doubling.
  const auto grow = [&](std::size_t bytes) {
    return memory == nullptr || charge == nullptr || charge->Add(*memory, bytes);
  };
  StopMatcher m;
  if (!grow(64 * sizeof(Node))) {
    return std::unexpected(std::uint64_t{64 * sizeof(Node)});
  }
  m.nodes_.reserve(64);
  m.nodes_.emplace_back();  // the root
  m.count_ = stops.size();
  for (std::string_view s : stops) {
    m.longest_ = std::max(m.longest_, s.size());
    std::uint32_t node = 0;
    for (const char ch : s) {
      const auto byte = static_cast<unsigned char>(ch);
      std::uint32_t next = m.Child(node, byte);
      if (next == 0) {
        if (m.nodes_.size() == m.nodes_.capacity()) {
          const std::size_t more = 2 * m.nodes_.capacity();
          if (!grow(more * sizeof(Node))) {
            return std::unexpected(std::uint64_t{more * sizeof(Node)});
          }
          m.nodes_.reserve(more);
        }
        next = static_cast<std::uint32_t>(m.nodes_.size());
        Node n;
        n.depth = m.nodes_[node].depth + 1;
        n.byte = byte;
        if (node == 0) {
          m.root_[byte] = next;
        } else {
          n.sibling = m.nodes_[node].child;
          m.nodes_[node].child = next;
        }
        m.nodes_.push_back(n);
      }
      node = next;
    }
    if (node != 0) {
      m.nodes_[node].out = m.nodes_[node].depth;  // a stop string ends here
    }
  }
  // Failure links breadth first, so a node's suffix (shallower) is linked
  // before it; `out` is a node's own stop string, else its suffix's.
  std::vector<std::uint32_t> queue;
  if (!grow(m.nodes_.size() * sizeof(std::uint32_t))) {
    return std::unexpected(std::uint64_t{m.nodes_.size() * sizeof(std::uint32_t)});
  }
  queue.reserve(m.nodes_.size());
  for (const std::uint32_t c : m.root_) {
    if (c != 0) {
      queue.push_back(c);  // fail = 0
    }
  }
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const std::uint32_t u = queue[head];
    for (std::uint32_t v = m.nodes_[u].child; v != 0; v = m.nodes_[v].sibling) {
      const unsigned char byte = m.nodes_[v].byte;
      std::uint32_t f = m.nodes_[u].fail;
      std::uint32_t link = m.Child(f, byte);
      while (link == 0 && f != 0) {
        f = m.nodes_[f].fail;
        link = m.Child(f, byte);
      }
      m.nodes_[v].fail = link;
      if (m.nodes_[v].out == 0) {
        m.nodes_[v].out = m.nodes_[link].out;
      }
      queue.push_back(v);
    }
  }
  // Grown by doubling: the request holds the matcher as long as it lives,
  // so it is copied to an exact allocation (libstdc++'s shrink_to_fit does
  // nothing without exceptions, RE-044).
  if (m.nodes_.capacity() > m.nodes_.size()) {
    if (!grow(m.nodes_.size() * sizeof(Node))) {
      return std::unexpected(std::uint64_t{m.nodes_.size() * sizeof(Node)});
    }
    std::vector<Node>(m.nodes_.begin(), m.nodes_.end()).swap(m.nodes_);
  }
  return m;
}

// ---------------------------------------------------------------- OutputText

OutputText::OutputText(const std::vector<std::string>& stops) {
  std::vector<std::string_view> views;
  views.reserve(stops.size());
  for (const std::string& s : stops) {
    if (!s.empty()) {
      views.emplace_back(s);  // an empty one matches nowhere (the route refuses one)
    }
  }
  if (!views.empty()) {
    stops_ = std::make_shared<const StopMatcher>(StopMatcher::Build(views));
  }
}

OutputText::Out OutputText::Content(std::string_view text) {
  if (stopped_) {
    return {};
  }
  if (!content_started_ && had_reasoning_) {
    while (!text.empty() && (text.front() == '\n' || text.front() == ' ')) {
      text.remove_prefix(1);
    }
  }
  if (text.empty()) {
    return {};
  }
  content_started_ = true;
  if (!stops_) {
    return {.reasoning = {}, .content = std::string(text)};
  }
  const std::size_t fresh = held_.size();
  held_ += text;
  // The matcher reads the new bytes; the stop string that starts first,
  // among those the text so far holds, ends the answer before it. At each
  // byte the longest stop string ending there starts first, and the held
  // text always covers the matcher's depth, so a start is never before it.
  std::size_t first = std::string::npos;
  std::uint32_t state = state_;
  for (std::size_t i = fresh; i < held_.size(); ++i) {
    state = stops_->Next(state, static_cast<unsigned char>(held_[i]));
    if (const std::uint32_t matched = stops_->matched(state); matched != 0) {
      first = std::min<std::size_t>(first, i + 1 - matched);
    }
  }
  state_ = state;
  if (first != std::string::npos) {
    Out out{.reasoning = {}, .content = held_.substr(held_start_, first - held_start_)};
    held_.clear();
    held_start_ = 0;
    stopped_ = true;
    return out;
  }
  // Hold back the longest tail that begins some stop string.
  const std::size_t hold = stops_->depth(state);
  const std::size_t end = held_.size() - hold;
  Out out{.reasoning = {}, .content = held_.substr(held_start_, end - held_start_)};
  held_start_ = end;
  // Moving a long retained prefix after every small piece is quadratic.
  // Compact only once the consumed prefix is at least the retained tail;
  // each moved byte is then paid for by a byte already emitted.
  if (held_start_ >= hold) {
    held_.erase(0, held_start_);
    held_start_ = 0;
  }
  return out;
}

OutputText::Out OutputText::Boundary() {
  state_ = 0;
  return Finish();
}

OutputText::Out OutputText::Finish() {
  held_.erase(0, held_start_);
  held_start_ = 0;
  Out out{.reasoning = {}, .content = std::move(held_)};
  held_.clear();
  return out;
}

}  // namespace llmp::runtime::api
