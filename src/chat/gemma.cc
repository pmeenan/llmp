// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Gemma family's chat formats, as their templates render them through
// Hugging Face transformers (docs/tokenizer.md). Written from the formats,
// not translated from the templates:
//
// - Gemma 3 (SHA-256 7de1c58e..., the template of Unsloth's Gemma 3 copies
//   and of the GGUFs converted from Google's checkpoints): turns
//   <start_of_turn>ROLE ... <end_of_turn>, the assistant as `model`, a
//   leading system message prefixed to the first user turn, roles
//   alternating user/assistant (the template's own refusal). Gemma 3n's
//   and Unsloth's 270m templates render text alike.
// - Gemma 4 (SHA-256 ae53464b..., google/gemma-4-31B-it and its siblings
//   since 2026-07-15): turns <|turn>ROLE ... <turn|>, a system turn holding
//   the thinking switch, the system message and tool declarations in
//   Gemma's own notation, a thought channel, tool calls and the tool
//   messages after them as responses inside the assistant's turn. Variants:
//   the E2B/E4B template (no empty thought channel in the generation
//   prompt), Unsloth's (tool-call arguments given as a string render
//   instead of raising) and Google's 2026-04-28 template (NVIDIA's NVFP4
//   checkpoints: no preserve_thinking, earlier reasoning only with tool
//   calls, `None` for nulls, no turn continuation; its 2026-05-18 revision
//   differs only in tool results given as content parts, which llmpalooza's
//   messages cannot express). Google's two earlier templates (2026-04-02
//   and 2026-04-10) have no renderer: the variant probes tell them apart.
//
// The templates rescan the conversation for every message (quadratic); the
// renderers are linear in its length, and their output is bounded
// (writer.h). Case mapping is Python's, in full (chat/pycase.h).

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/json.h"
#include "chat/chat.h"
#include "chat/pycase.h"
#include "chat/pyjson.h"
#include "chat/writer.h"

namespace llmp::chat {
namespace {

using base::json::Value;

constexpr std::string_view kBos = "<bos>";

// Gemma 3.
constexpr std::string_view kStartOfTurn = "<start_of_turn>";
constexpr std::string_view kEndOfTurn = "<end_of_turn>";

// Gemma 4.
constexpr std::string_view kTurn = "<|turn>";
constexpr std::string_view kTurnEnd = "<turn|>";
constexpr std::string_view kThink = "<|think|>";
constexpr std::string_view kChannel = "<|channel>";
constexpr std::string_view kChannelEnd = "<channel|>";
constexpr std::string_view kTool = "<|tool>";
constexpr std::string_view kToolEnd = "<tool|>";
constexpr std::string_view kToolCall = "<|tool_call>";
constexpr std::string_view kToolCallEnd = "<tool_call|>";
constexpr std::string_view kToolResponse = "<|tool_response>";
constexpr std::string_view kToolResponseEnd = "<tool_response|>";
constexpr std::string_view kQuote = R"(<|"|>)";

// Deeper JSON than any client sends (base::json's default bound is 64):
// refused rather than recursed into.
constexpr std::size_t kMaxDepth = 256;

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::size_t item = kNoItem) {
  return std::unexpected(Error{rule, reason, item});
}

std::string_view RoleName(Role role) {
  switch (role) {
    case Role::kSystem:
      return "system";
    case Role::kUser:
      return "user";
    case Role::kAssistant:
      return "assistant";
    case Role::kTool:
      return "tool";
  }
  return "user";
}

// ------------------------------------------------- Python's view of JSON

// A value as the template sees it: the client's JSON, Jinja's undefined (a
// key a mapping lacks), or a bound method (Jinja's attribute fallback finds
// one for a key a mapping lacks that names a dict method, such as `items`).
// No key the templates look up names an attribute of a string, list or
// number, so on those every lookup is undefined.
// A JSON null that stands in for the value of an undefined or a method.
Value Placeholder() {
  static const auto kNull = base::json::Parse("null");
  return kNull->root();
}

struct Py {
  enum class Kind : std::uint8_t { kUndefined, kJson, kMethod };
  Kind kind = Kind::kUndefined;
  Value json = Placeholder();  // kJson's value

  static Py Of(Value v) { return {Kind::kJson, v}; }
  static Py Method() { return {Kind::kMethod, Placeholder()}; }
  bool undefined() const { return kind == Kind::kUndefined; }
  bool Is(base::json::Kind k) const { return kind == Kind::kJson && json.kind() == k; }
  bool mapping() const { return Is(base::json::Kind::kObject); }
  bool string() const { return Is(base::json::Kind::kString); }
  bool none() const { return Is(base::json::Kind::kNull); }
};

bool DictMethod(std::string_view key) {
  constexpr std::array<std::string_view, 11> kMethods = {
      "clear", "copy",    "fromkeys",   "get",    "items", "keys",
      "pop",   "popitem", "setdefault", "update", "values"};
  return std::ranges::any_of(kMethods, [&](std::string_view m) { return m == key; });
}

// x[key]. Subscripting an undefined value raises (Jinja's UndefinedError).
std::expected<Py, Error> Item(const Py& x, std::string_view key) {
  if (x.undefined()) {
    return Fail(Rule::kInvalid, "a member of an undefined value");
  }
  if (x.mapping()) {
    if (const auto m = x.json.find(key)) {
      return Py::Of(*m);
    }
    if (DictMethod(key)) {
      return Py::Method();
    }
  }
  return Py{};
}

bool NumberIsZero(Value v) {
  if (v.is_integer()) {
    const auto i = v.int64();
    return i.has_value() && *i == 0;  // beyond 64 bits it is not zero
  }
  const auto x = v.float64();
  return x.has_value() && *x == 0.0;
}

// Python's truth of a value.
bool Truthy(const Py& x) {
  if (x.kind != Py::Kind::kJson) {
    return x.kind == Py::Kind::kMethod;
  }
  const Value v = x.json;
  switch (v.kind()) {
    case base::json::Kind::kNull:
    case base::json::Kind::kFalse:
      return false;
    case base::json::Kind::kTrue:
      return true;
    case base::json::Kind::kNumber:
      return !NumberIsZero(v);
    case base::json::Kind::kString:
      return !v.string().empty();
    case base::json::Kind::kArray:
    case base::json::Kind::kObject:
      return v.size() != 0;
  }
  return false;
}

// What {{ x }} prints: str(x), empty for undefined.
std::expected<void, Error> AppendStr(const Py& x, std::string& out) {
  if (x.kind == Py::Kind::kMethod) {
    return Fail(Rule::kUnsupported, "a method printed as text");
  }
  if (x.kind == Py::Kind::kJson) {
    AppendPythonStr(x.json, out);
  }
  return {};
}

std::expected<std::string, Error> Str(const Py& x) {
  std::string s;
  if (auto r = AppendStr(x, s); !r) {
    return std::unexpected(r.error());
  }
  return s;
}

// Visits UTF-8 code points, assuming well-formed text (as JSON's is).
template <typename F>
void ForEachCodePoint(std::string_view s, F&& f) {
  for (std::size_t at = 0; at < s.size();) {
    std::size_t n = 0;
    const char32_t cp = DecodeUtf8At(s, at, n);
    f(cp, s.substr(at, n));
    at += n;
  }
}

// x | upper: str(x).upper(), Python's full mapping (chat/pycase.h).
std::expected<std::string, Error> Upper(const Py& x) {
  auto s = Str(x);
  if (!s) {
    return s;
  }
  return PythonCase(*s, CaseOp::kUpper);
}

// `for item in x`, each item's str(): a list's elements, a string's
// characters, a mapping's keys; undefined iterates as empty. Anything else
// is not iterable, which the template raises.
template <typename F>
std::expected<void, Error> ForEachStr(const Py& x, F&& f) {
  if (x.undefined()) {
    return {};
  }
  if (x.kind == Py::Kind::kMethod) {
    return Fail(Rule::kInvalid, "iterating a value that is not iterable");
  }
  const Value v = x.json;
  switch (v.kind()) {
    case base::json::Kind::kArray:
      for (std::size_t i = 0; i < v.size(); ++i) {
        std::string s;
        AppendPythonStr(v.at(i), s);
        if (auto r = f(std::string_view(s)); !r) {
          return r;
        }
      }
      return {};
    case base::json::Kind::kString: {
      std::expected<void, Error> result;
      ForEachCodePoint(v.string(), [&](char32_t, std::string_view ch) {
        if (result) {
          result = f(ch);
        }
      });
      return result;
    }
    case base::json::Kind::kObject:
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (auto r = f(v.key(i)); !r) {
          return r;
        }
      }
      return {};
    default:
      return Fail(Rule::kInvalid, "iterating a value that is not iterable");
  }
}

// `needle in x`.
std::expected<bool, Error> Contains(const Py& x, std::string_view needle) {
  if (x.undefined()) {
    return false;
  }
  if (x.kind == Py::Kind::kJson) {
    const Value v = x.json;
    if (v.is_object()) {
      return v.find(needle).has_value();
    }
    if (v.is_string()) {
      return v.string().contains(needle);
    }
    if (v.is_array()) {
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (v.at(i).is_string() && v.at(i).string() == needle) {
          return true;
        }
      }
      return false;
    }
  }
  return Fail(Rule::kInvalid, "`in` a value that is not a container");
}

// x | dictsort: the members' indexes by key, case-insensitively, stably.
std::expected<std::vector<std::size_t>, Error> DictSorted(const Py& x) {
  if (!x.mapping()) {
    return Fail(Rule::kInvalid, "dictsort of a value that is not a mapping");
  }
  const Value v = x.json;
  std::vector<std::size_t> order(v.size());
  std::vector<std::string> folded(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    order[i] = i;
    folded[i] = PythonCase(v.key(i), CaseOp::kLower);  // Jinja2's ignore_case: str.lower()
  }
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) {
    return folded[a] < folded[b];  // UTF-8 bytes order as code points do
  });
  return order;
}

// ------------------------------------------------ Gemma 4's notation

class Gemma4Writer {
 public:
  Gemma4Writer(bool null_word, std::size_t max_bytes) : w_(max_bytes), null_word_(null_word) {}

  Writer& w() { return w_; }

  std::expected<void, Error> Print(const Py& x) {
    auto s = Str(x);
    if (!s) {
      return std::unexpected(s.error());
    }
    w_.Text(*s);
    return {};
  }

  void Quoted(std::string_view text) {
    w_.Special(kQuote);
    w_.Text(text);
    w_.Special(kQuote);
  }

  // A value in Gemma's notation: strings between <|"|>, mappings by sorted
  // key (keys quoted too when escape_keys), lists, null (`None` in the
  // 2026-04 template, which prints it as Python does), numbers as Python
  // prints them.
  std::expected<void, Error> Argument(const Py& x, bool escape_keys, std::size_t depth = 0) {
    if (depth > kMaxDepth) {
      return Fail(Rule::kUnsupported, "JSON nested deeper than the renderer's bound");
    }
    if (x.none() && null_word_) {
      w_.Text("null");
    } else if (x.string()) {
      Quoted(x.json.string());
    } else if (x.Is(base::json::Kind::kTrue) || x.Is(base::json::Kind::kFalse)) {
      w_.Text(x.json.boolean() ? "true" : "false");
    } else if (x.mapping()) {
      auto order = DictSorted(x);
      if (!order) {
        return std::unexpected(order.error());
      }
      w_.Text("{");
      for (std::size_t k = 0; k < order->size(); ++k) {
        const std::size_t i = (*order)[k];
        if (k != 0) {
          w_.Text(",");
        }
        if (escape_keys) {
          Quoted(x.json.key(i));
        } else {
          w_.Text(x.json.key(i));
        }
        w_.Text(":");
        if (auto r = Argument(Py::Of(x.json.member(i)), escape_keys, depth + 1); !r) {
          return r;
        }
      }
      w_.Text("}");
    } else if (x.Is(base::json::Kind::kArray)) {
      w_.Text("[");
      for (std::size_t i = 0; i < x.json.size(); ++i) {
        if (i != 0) {
          w_.Text(",");
        }
        if (auto r = Argument(Py::Of(x.json.at(i)), escape_keys, depth + 1); !r) {
          return r;
        }
      }
      w_.Text("]");
    } else if (x.undefined()) {
      w_.Text("[]");  // Jinja's `sequence` test accepts an undefined, which iterates as empty
    } else {
      return Print(x);  // a number or a 2026-04 null
    }
    return {};
  }

  // `required:[<|"|>a<|"|>,...]`'s items: str() of each item iterated.
  std::expected<void, Error> QuotedItems(const Py& x) {
    bool first = true;
    return ForEachStr(x, [&](std::string_view s) -> std::expected<void, Error> {
      if (!first) {
        w_.Text(",");
      }
      first = false;
      Quoted(s);
      return {};
    });
  }

  // A schema's properties: each property's description, enum, items,
  // nullable, nested properties, required and upper-cased type, in
  // Gemma's notation. With filter_keys (an object property without
  // `properties`), the property's own keys other than the standard ones.
  std::expected<void, Error> Parameters(const Py& properties, bool filter_keys, std::size_t depth) {
    if (depth > kMaxDepth) {
      return Fail(Rule::kUnsupported, "JSON nested deeper than the renderer's bound");
    }
    auto order = DictSorted(properties);
    if (!order) {
      return std::unexpected(order.error());
    }
    bool found_first = false;
    for (const std::size_t i : *order) {
      const std::string_view key = properties.json.key(i);
      if (filter_keys && (key == "description" || key == "type" || key == "properties" ||
                          key == "required" || key == "nullable")) {
        continue;
      }
      const Py value = Py::Of(properties.json.member(i));
      if (found_first) {
        w_.Text(",");
      }
      found_first = true;
      w_.Text(key);
      w_.Text(":{");
      bool add_comma = false;
      const auto comma = [&] {
        if (add_comma) {
          w_.Text(",");
        } else {
          add_comma = true;
        }
      };
      const Py description = *Item(value, "description");
      if (Truthy(description)) {
        w_.Text("description:");
        w_.Special(kQuote);
        if (auto r = Print(description); !r) {
          return r;
        }
        w_.Special(kQuote);
        add_comma = true;
      }
      auto type = Upper(*Item(value, "type"));
      if (!type) {
        return std::unexpected(type.error());
      }
      if (*type == "STRING") {
        const Py e = *Item(value, "enum");
        if (Truthy(e)) {
          comma();
          w_.Text("enum:");
          if (auto r = Argument(e, true, depth + 1); !r) {
            return r;
          }
        }
      } else if (*type == "ARRAY") {
        const Py items = *Item(value, "items");
        if (items.mapping() && Truthy(items)) {
          comma();
          w_.Text("items:{");
          if (auto r = Items(items, depth + 1); !r) {
            return r;
          }
          w_.Text("}");
        }
      }
      if (Truthy(*Item(value, "nullable"))) {
        comma();
        w_.Text("nullable:true");
      }
      if (*type == "OBJECT") {
        const Py props = *Item(value, "properties");
        if (props.mapping()) {
          comma();
          w_.Text("properties:{");
          if (auto r = Parameters(props, false, depth + 1); !r) {
            return r;
          }
          w_.Text("}");
        } else if (value.mapping()) {
          comma();
          w_.Text("properties:{");
          if (auto r = Parameters(value, true, depth + 1); !r) {
            return r;
          }
          w_.Text("}");
        }
        const Py required = *Item(value, "required");
        if (Truthy(required)) {
          comma();
          w_.Text("required:[");
          if (auto r = QuotedItems(required); !r) {
            return r;
          }
          w_.Text("]");
        }
      }
      comma();
      w_.Text("type:");
      w_.Special(kQuote);
      w_.Text(*type);
      w_.Special(kQuote);
      w_.Text("}");
    }
    return {};
  }

  // An array property's `items` schema: its members by sorted key, each
  // not null.
  std::expected<void, Error> Items(const Py& items, std::size_t depth) {
    auto order = DictSorted(items);
    if (!order) {
      return std::unexpected(order.error());
    }
    bool found_first = false;
    for (const std::size_t i : *order) {
      const std::string_view key = items.json.key(i);
      const Py v = Py::Of(items.json.member(i));
      if (v.none()) {
        continue;
      }
      if (found_first) {
        w_.Text(",");
      }
      found_first = true;
      if (key == "properties") {
        w_.Text("properties:{");
        if (v.mapping()) {
          if (auto r = Parameters(v, false, depth + 1); !r) {
            return r;
          }
        }
        w_.Text("}");
      } else if (key == "required") {
        w_.Text("required:[");
        if (auto r = QuotedItems(v); !r) {
          return r;
        }
        w_.Text("]");
      } else if (key == "type") {
        w_.Text("type:");
        if (v.string()) {
          auto t = Upper(v);
          if (!t) {
            return std::unexpected(t.error());
          }
          Quoted(*t);
        } else {
          // item_value | map('upper') | list, printed as a list of strings.
          std::vector<std::string> upper;
          auto r = ForEachStr(v, [&](std::string_view s) -> std::expected<void, Error> {
            upper.push_back(PythonCase(s, CaseOp::kUpper));
            return {};
          });
          if (!r) {
            return r;
          }
          w_.Text("[");
          for (std::size_t k = 0; k < upper.size(); ++k) {
            if (k != 0) {
              w_.Text(",");
            }
            Quoted(upper[k]);
          }
          w_.Text("]");
        }
      } else {
        w_.Text(key);
        w_.Text(":");
        if (auto r = Argument(v, true, depth + 1); !r) {
          return r;
        }
      }
    }
    return {};
  }

  // One tool's declaration: name, description, parameters and response.
  std::expected<void, Error> Declaration(Value tool) {
    const Py function = *Item(Py::Of(tool), "function");
    auto name = Item(function, "name");
    if (!name) {
      return std::unexpected(name.error());
    }
    w_.Text("declaration:");
    if (auto r = Print(*name); !r) {
      return r;
    }
    w_.Text("{description:");
    w_.Special(kQuote);
    if (auto r = Print(*Item(function, "description")); !r) {
      return r;
    }
    w_.Special(kQuote);
    const Py params = *Item(function, "parameters");
    if (Truthy(params)) {
      w_.Text(",parameters:{");
      const Py properties = *Item(params, "properties");
      if (Truthy(properties)) {
        w_.Text("properties:{");
        if (auto r = Parameters(properties, false, 1); !r) {
          return r;
        }
        w_.Text("},");
      }
      const Py required = *Item(params, "required");
      if (Truthy(required)) {
        w_.Text("required:[");
        if (auto r = QuotedItems(required); !r) {
          return r;
        }
        w_.Text("],");
      }
      const Py type = *Item(params, "type");
      if (Truthy(type)) {
        auto t = Upper(type);
        if (!t) {
          return std::unexpected(t.error());
        }
        w_.Text("type:");
        Quoted(*t);
        w_.Text("}");
      }
    }
    auto has_response = Contains(function, "response");
    if (!has_response) {
      return std::unexpected(has_response.error());
    }
    if (*has_response) {
      const Py response = *Item(function, "response");
      auto description = Item(response, "description");
      if (!description) {
        return std::unexpected(description.error());
      }
      w_.Text(",response:{");
      if (Truthy(*description)) {
        w_.Text("description:");
        w_.Special(kQuote);
        if (auto r = Print(*description); !r) {
          return r;
        }
        w_.Special(kQuote);
        w_.Text(",");
      }
      auto type = Upper(*Item(response, "type"));
      if (!type) {
        return std::unexpected(type.error());
      }
      if (*type == "OBJECT") {
        w_.Text("type:");
        Quoted(*type);
        w_.Text("}");
      }
    }
    w_.Text("}");
    return {};
  }

 private:
  Writer w_;
  bool null_word_;
};

// The assistant's content without thought channels: each part between
// <channel|> markers up to its first <|channel>, joined, then stripped.
std::string StripThinking(std::string_view text) {
  std::string out;
  for (std::size_t start = 0;;) {
    const std::size_t end = text.find(kChannelEnd, start);
    const std::string_view part =
        text.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);
    out += part.substr(0, part.find(kChannel));
    if (end == std::string_view::npos) {
      break;
    }
    start = end + kChannelEnd.size();
  }
  return std::string(PythonStrip(out));
}

// Where Gemma 4's templates part (see the top of this file).
struct Gemma4Variant {
  bool july = true;                // Google's template since 2026-07-15
  bool empty_thought = true;       // the prompt closes an empty thought channel unless thinking
  bool unsloth_arguments = false;  // string arguments render (outer braces stripped)
};

std::expected<Rendered, Error> RenderGemma4Variant(const Conversation& c,
                                                   const Gemma4Variant& variant) {
  const std::vector<Message>& messages = c.messages;
  if (!variant.july && messages.empty()) {
    return Fail(Rule::kInvalid, "no messages");  // the 2026-04 template reads messages[0]
  }
  const bool thinking = c.enable_thinking.value_or(false);
  const bool preserve = variant.july && c.preserve_thinking.value_or(false);
  enum class Prev : std::uint8_t { kNone, kThink, kTool, kToolCall, kToolResponse };
  Prev prev = Prev::kNone;
  Gemma4Writer g(variant.july, c.max_render_bytes);
  Writer& w = g.w();

  w.Special(kBos);
  const bool system_first = !messages.empty() && messages[0].role == Role::kSystem;
  std::size_t start = 0;
  if (thinking || !c.tools.empty() || system_first) {
    w.Special(kTurn);
    w.Text("system\n");
    if (thinking) {
      w.Special(kThink);
      w.Text("\n");
      prev = Prev::kThink;
    }
    if (system_first) {
      if (messages[0].content) {
        w.Text(PythonStrip(*messages[0].content));
      }
      start = 1;
    }
    if (!c.tools.empty()) {
      for (const Value tool : c.tools) {
        w.Special(kTool);
        if (auto r = g.Declaration(tool); !r) {
          return std::unexpected(r.error());
        }
        w.Special(kToolEnd);
      }
      prev = Prev::kTool;
    }
    w.Special(kTurnEnd);
    w.Text("\n");
  }
  w.Mark(BoundaryKind::kPrefixEnd);

  // The last user message, and for each message the next one that is not
  // a tool result (the templates scan for both, per message).
  const std::size_t n = messages.size();
  std::optional<std::size_t> last_user;
  for (std::size_t i = start; i < n; ++i) {
    if (messages[i].role == Role::kUser) {
      last_user = i;
    }
  }
  std::vector<std::optional<Role>> next_non_tool(n);
  for (std::size_t i = n; i-- > start + 1;) {
    next_non_tool[i - 1] =
        messages[i].role != Role::kTool ? std::optional<Role>(messages[i].role) : next_non_tool[i];
  }

  std::optional<Role> prev_non_tool;
  for (std::size_t i = start; i < n; ++i) {
    const Message& m = messages[i];
    if (m.role == Role::kTool) {
      continue;  // rendered with the assistant message it answers, or not at all
    }
    prev = Prev::kNone;
    const bool model = m.role == Role::kAssistant;
    if (!(model && prev_non_tool == Role::kAssistant)) {
      w.Special(kTurn);
      w.Text(model ? std::string_view("model") : RoleName(m.role));
      w.Text("\n");
    }
    const bool calls = !m.tool_calls.empty();
    const bool after_last_user = !last_user || i > *last_user;
    const bool gate =
        variant.july ? (after_last_user || (preserve && calls)) : (after_last_user && calls);
    if (m.reasoning_content && !m.reasoning_content->empty() && gate) {
      w.Special(kChannel);
      w.Text("thought\n");
      w.Text(*m.reasoning_content);
      w.Text("\n");
      w.Special(kChannelEnd);
    }
    for (const ToolCall& call : m.tool_calls) {
      w.Special(kToolCall);
      w.Text("call:");
      w.Text(call.name);
      w.Text("{");
      const Value args = call.arguments;
      if (args.is_object()) {
        auto order = DictSorted(Py::Of(args));
        if (!order) {
          return std::unexpected(order.error());
        }
        for (std::size_t k = 0; k < order->size(); ++k) {
          if (k != 0) {
            w.Text(",");
          }
          w.Text(args.key((*order)[k]));
          w.Text(":");
          if (auto r = g.Argument(Py::Of(args.member((*order)[k])), false); !r) {
            return std::unexpected(r.error());
          }
        }
      } else if (args.is_null() && variant.july) {
        // nothing
      } else if (args.is_string() && variant.unsloth_arguments) {
        const std::string_view s = PythonStrip(args.string());
        if (s.size() >= 2 && s.front() == '{' && s.back() == '}') {
          w.Text(s.substr(1, s.size() - 2));
        } else {
          w.Text(args.string());
        }
      } else if (args.is_string() && !variant.july) {
        w.Text(args.string());
      } else if (variant.july && !variant.unsloth_arguments) {
        return Fail(Rule::kInvalid, "tool call arguments are not an object", i);
      }
      w.Text("}");
      w.Special(kToolCallEnd);
      prev = Prev::kToolCall;
    }
    // The tool results after a message with tool calls, each answering
    // (the templates match call IDs, which llmpalooza's messages lack, so every
    // call matches) the last call.
    bool responded = false;
    if (calls) {
      const std::string_view name = m.tool_calls.back().name;
      for (std::size_t k = i + 1; k < n && messages[k].role == Role::kTool; ++k) {
        w.Special(kToolResponse);
        w.Text("response:");
        w.Text(name);
        w.Text("{value:");
        if (messages[k].content) {
          g.Quoted(*messages[k].content);
        } else {
          w.Text(variant.july ? "null" : "None");
        }
        w.Text("}");
        w.Special(kToolResponseEnd);
        responded = true;
        prev = Prev::kToolResponse;
      }
    }
    std::string content;
    if (m.content) {
      content = model ? StripThinking(*m.content) : std::string(PythonStrip(*m.content));
    }
    w.Text(content);
    const bool has_content = !content.empty();
    if (prev == Prev::kToolCall && !responded) {
      w.Special(kToolResponse);
    } else if (variant.july) {
      const bool continues = model && next_non_tool[i] == Role::kAssistant && (!calls || responded);
      // A turn of tool results without content closes only before more messages.
      const bool open = responded && !has_content && !next_non_tool[i];
      if (!continues && !open) {
        w.Special(kTurnEnd);
        w.Text("\n");
      }
    } else if (!responded || has_content) {
      w.Special(kTurnEnd);
      w.Text("\n");
    }
    w.Mark(BoundaryKind::kMessageEnd);
    prev_non_tool = m.role;
  }

  if (c.add_generation_prompt) {
    if (prev != Prev::kToolResponse && prev != Prev::kToolCall) {
      w.Mark(BoundaryKind::kGenerationPrompt);
      w.Special(kTurn);
      w.Text("model\n");
      if (!thinking && variant.empty_thought) {
        w.Special(kChannel);
        w.Text("thought\n");
        w.Special(kChannelEnd);
      }
    } else if (variant.july && prev == Prev::kToolResponse && thinking) {
      w.Mark(BoundaryKind::kGenerationPrompt);
      w.Special(kChannel);
      w.Text("thought\n");
    }
  }
  return w.Finish();
}

}  // namespace

std::expected<Rendered, Error> RenderGemma3(const Conversation& c) {
  const std::vector<Message>& messages = c.messages;
  if (messages.empty()) {
    return Fail(Rule::kInvalid, "no messages");  // the template reads messages[0]
  }
  Writer w(c.max_render_bytes);
  w.Special(kBos);
  std::string prefix;
  std::size_t start = 0;
  if (messages[0].role == Role::kSystem) {
    const std::optional<std::string>& system = messages[0].content;
    if (!system) {
      return Fail(Rule::kInvalid, "a system message without content", 0);
    }
    prefix = *system + "\n\n";
    start = 1;
  } else {
    w.Mark(BoundaryKind::kPrefixEnd);
  }
  for (std::size_t i = start; i < messages.size(); ++i) {
    const Message& m = messages[i];
    const std::size_t index = i - start;
    if ((m.role == Role::kUser) != (index % 2 == 0)) {
      return Fail(Rule::kInvalid, "roles do not alternate user/assistant", i);
    }
    w.Special(kStartOfTurn);
    w.Text(m.role == Role::kAssistant ? std::string_view("model") : RoleName(m.role));
    w.Text("\n");
    if (index == 0 && start == 1) {
      w.Text(prefix);
      w.Mark(BoundaryKind::kPrefixEnd);
    }
    if (!m.content) {
      return Fail(Rule::kInvalid, "a message without content", i);
    }
    w.Text(PythonStrip(*m.content));
    w.Special(kEndOfTurn);
    w.Text("\n");
    w.Mark(BoundaryKind::kMessageEnd);
  }
  if (c.add_generation_prompt) {
    w.Mark(BoundaryKind::kGenerationPrompt);
    w.Special(kStartOfTurn);
    w.Text("model\n");
  }
  return w.Finish();
}

std::expected<Rendered, Error> RenderGemma4(const Conversation& c) {
  return RenderGemma4Variant(c, {});
}

std::expected<Rendered, Error> RenderGemma4E(const Conversation& c) {
  return RenderGemma4Variant(c, {.empty_thought = false});
}

std::expected<Rendered, Error> RenderGemma4Unsloth(const Conversation& c) {
  return RenderGemma4Variant(c, {.unsloth_arguments = true});
}

std::expected<Rendered, Error> RenderGemma4EUnsloth(const Conversation& c) {
  return RenderGemma4Variant(c, {.empty_thought = false, .unsloth_arguments = true});
}

std::expected<Rendered, Error> RenderGemma4April(const Conversation& c) {
  return RenderGemma4Variant(c, {.july = false});
}

}  // namespace llmp::chat
