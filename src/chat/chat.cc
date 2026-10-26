// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "chat/chat.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "base/sha256.h"
#include "chat/tool_calls.h"
#include "tokenizer/tokenizer.h"

namespace llmp::chat {
namespace {
std::uint64_t AddBytes(std::uint64_t a, std::uint64_t b) {
  constexpr auto most = std::numeric_limits<std::uint64_t>::max();
  return b > most - a ? most : a + b;
}

std::uint64_t MultiplyBytes(std::uint64_t count, std::uint64_t bytes) {
  constexpr auto most = std::numeric_limits<std::uint64_t>::max();
  return bytes != 0 && count > most / bytes ? most : count * bytes;
}

// One entry of an unordered index of views (a node with its cached hash and
// a bucket, with allocator rounding): ProjectContent's association tables.
constexpr std::uint64_t kIndexEntryBytes = 64;
constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();  // no call or offset

// The chat route's limits for a tool call's arguments (runtime/api.cc
// ParseCall): bounded by the text itself, nested at most 64 deep
// (runtime::api::kMaxJsonDepth). Projection reparses what intake accepted.
base::json::Limits ToolArgumentLimits(std::string_view text) {
  constexpr std::size_t kMaxDepth = 64;  // runtime::api::kMaxJsonDepth
  return {.max_bytes = text.size(),
          .max_depth = kMaxDepth,
          .max_values = text.size() + 1,
          .max_string_bytes = text.size() + 1};
}

std::uint64_t PartBytes(const MessagePart& part, bool storage) {
  const auto bytes = [storage](const std::string& s) -> std::uint64_t {
    return storage ? s.capacity() : s.size();
  };
  return std::visit(
      [&](const auto& p) -> std::uint64_t {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, TextPart>) {
          return bytes(p.text);
        } else if constexpr (std::is_same_v<T, ToolUsePart>) {
          return AddBytes(AddBytes(bytes(p.id), bytes(p.name)), bytes(p.input_json));
        } else if constexpr (std::is_same_v<T, ToolResultPart>) {
          return AddBytes(AddBytes(bytes(p.tool_use_id), bytes(p.text)), bytes(p.name));
        } else {
          return AddBytes(bytes(p.media_type), bytes(p.source));
        }
      },
      part);
}

}  // namespace

std::uint64_t ProjectionStorageBytes(const Conversation& conversation) {
  auto bytes = AddBytes(MultiplyBytes(conversation.messages.capacity(), sizeof(Message)),
                        MultiplyBytes(conversation.tools.capacity(), sizeof(base::json::Value)));
  if (conversation.reasoning_effort) {
    bytes = AddBytes(bytes, conversation.reasoning_effort->capacity());
  }
  for (const Message& message : conversation.messages) {
    bytes = AddBytes(bytes, message.content.ProjectionBytes());
    if (message.reasoning_content) {
      bytes = AddBytes(bytes, message.reasoning_content->capacity());
    }
    bytes = AddBytes(
        bytes, AddBytes(message.tool_call_id.capacity(), message.tool_result_name.capacity()));
    // Group normalization's index tables and result copies; a resolved name is
    // borrowed by index, not copied once per result.
    bytes = AddBytes(bytes, sizeof(ToolResultPart) * 3 + sizeof(std::size_t) * 4);
    if (message.role == Role::kTool)
      bytes = AddBytes(bytes, MultiplyBytes(3, message.content.PayloadBytes()));
    // Each call's copy, its identity for ordering, and its entries in the
    // group's ID and name indexes.
    bytes = AddBytes(bytes, MultiplyBytes(message.tool_calls.capacity(),
                                          sizeof(ToolCall) + sizeof(ToolUsePart) +
                                              sizeof(std::size_t) * 2 + kIndexEntryBytes * 2));
    for (const ToolCall& call : message.tool_calls) {
      bytes = AddBytes(bytes, AddBytes(call.name.capacity(), MultiplyBytes(3, call.id.capacity())));
      // Legacy JSON values remain views and their owners remain shared. Only
      // structured tool parts parse new documents, in ProjectionBytes above.
    }
  }
  return bytes;
}

std::uint64_t Content::PayloadBytes() const {
  if (const auto* p = parts()) {
    std::uint64_t bytes = 0;
    for (const auto& part : *p) {
      bytes = AddBytes(bytes, PartBytes(part, false));
    }
    return bytes;
  }
  return text() ? text()->size() : 0;
}

std::uint64_t Content::StorageBytes() const {
  if (const auto* p = parts()) {
    std::uint64_t bytes = MultiplyBytes(p->capacity(), sizeof(MessagePart));
    for (const auto& part : *p) {
      bytes = AddBytes(bytes, PartBytes(part, true));
    }
    return bytes;
  }
  return text() ? text()->capacity() : 0;
}

std::uint64_t Content::ProjectionBytes() const {
  const auto* p = parts();
  if (p == nullptr) {
    return StorageBytes();
  }
  auto bytes =
      AddBytes(AddBytes(MultiplyBytes(3, StorageBytes()), MultiplyBytes(4, PayloadBytes())),
               MultiplyBytes(p->size(), sizeof(ToolCall) + sizeof(ToolResultPart) * 3 +
                                            sizeof(std::size_t) * 4 + 32));
  for (const auto& part : *p) {
    if (const auto* use = std::get_if<ToolUsePart>(&part)) {
      // Its parsed arguments, and its entries in the message's distinct-ID
      // set and the group's ID and name indexes.
      bytes = AddBytes(bytes, AddBytes(base::json::ParseWorkingBytes(use->input_json),
                                       sizeof(base::json::Document) + kIndexEntryBytes * 3));
    }
  }
  return bytes;
}

bool HasStructuredContent(const Conversation& conversation) {
  return std::ranges::any_of(conversation.messages, [](const Message& m) {
    return m.content.parts() != nullptr || (m.role == Role::kTool && !m.tool_result_normalized);
  });
}

std::expected<Conversation, Error> ProjectContent(const Conversation& conversation,
                                                  bool supports_tools) {
  // Bound the projection before copying or parsing. Serving funds its storage
  // from request memory (max_projection_bytes); direct callers retain the
  // render bound.
  std::uint64_t payload = 0;
  for (const Message& m : conversation.messages) {
    payload = AddBytes(payload, m.content.PayloadBytes());
    if (payload > conversation.max_render_bytes) {
      return std::unexpected(Error{.rule = Rule::kUnsupported,
                                   .reason = "message content exceeds the render bound",
                                   .bound = true});
    }
  }
  const std::uint64_t storage = ProjectionStorageBytes(conversation);
  if (conversation.max_projection_bytes != 0 && storage > conversation.max_projection_bytes) {
    return std::unexpected(Error{.rule = Rule::kUnsupported,
                                 .reason = "message projection exceeds its funded storage",
                                 .bound = true});
  }
  Conversation out = conversation;
  std::unordered_set<std::string_view> ids;  // one message's tool-use IDs, viewing its parts
  for (std::size_t i = 0; i < out.messages.size(); ++i) {
    Message& m = out.messages[i];
    const auto* parts = conversation.messages[i].content.parts();
    if (parts == nullptr) {
      continue;
    }
    const auto fail = [i](std::string_view reason) {
      return std::unexpected(Error{.rule = Rule::kUnsupported, .reason = reason, .item = i});
    };
    std::string text;
    bool calls = false;
    bool has_text = false;
    if (!m.tool_calls.empty()) {
      return fail("structured content and legacy tool calls cannot be mixed");
    }
    const auto uses = static_cast<std::size_t>(std::ranges::count_if(
        *parts, [](const MessagePart& part) { return std::holds_alternative<ToolUsePart>(part); }));
    m.tool_calls.reserve(uses);
    ids.clear();
    if (uses > 1) ids.reserve(uses);
    for (const MessagePart& part : *parts) {
      if (const auto* t = std::get_if<TextPart>(&part)) {
        if (calls) {
          return fail("text after tool use cannot be represented by this template boundary");
        }
        has_text = true;
        text += t->text;
      } else if (const auto* use = std::get_if<ToolUsePart>(&part)) {
        if (!supports_tools) {
          return fail("this template does not render tool use");
        }
        if (m.role != Role::kAssistant || use->name.empty()) {
          return fail("tool use requires an assistant and a name");
        }
        if (!use->id.empty() && !ids.insert(use->id).second) {
          return fail("tool use IDs must be distinct within a message");
        }
        auto document = base::json::Parse(use->input_json, ToolArgumentLimits(use->input_json));
        if (!document || !document->root().is_object()) {
          return fail("tool input must be a JSON object");
        }
        auto owner = std::make_shared<const base::json::Document>(std::move(*document));
        m.tool_calls.push_back({.name = use->name,
                                .arguments = owner->root(),
                                .id = use->id,
                                .owner = std::move(owner)});
        calls = true;
      } else if (const auto* result = std::get_if<ToolResultPart>(&part)) {
        if (!supports_tools) {
          return fail("this template does not render tool results");
        }
        if (parts->size() != 1 || (m.role != Role::kUser && m.role != Role::kTool)) {
          return fail("tool results require one result in a user or tool message");
        }
        m.role = Role::kTool;
        m.tool_call_id = result->tool_use_id;
        m.tool_result_name = result->name;
        m.tool_result_is_error = result->is_error;
        m.tool_result_normalized = false;
        text = result->text;
      } else {
        return fail("image content requires a multimodal renderer");
      }
    }
    m.content = calls && !has_text ? Content(std::nullopt) : Content(std::move(text));
  }
  // Resolve one contiguous group at a time, in time linear in its calls and
  // results. Known IDs constrain association first; remaining missing-ID
  // results retain their supplied relative order.
  for (std::size_t first = 0; first < out.messages.size();) {
    if (out.messages[first].role != Role::kTool) {
      ++first;
      continue;
    }
    std::size_t end = first + 1;
    while (end < out.messages.size() && out.messages[end].role == Role::kTool) ++end;
    const auto fail = [first](std::string_view reason) {
      return std::unexpected(Error{.rule = Rule::kUnsupported, .reason = reason, .item = first});
    };
    if (!supports_tools) return fail("this template does not render tool results");
    if (first == 0 || out.messages[first - 1].role != Role::kAssistant ||
        out.messages[first - 1].tool_calls.empty())
      return fail("a tool result requires preceding assistant tool calls");
    // The calls stay in place while the group's results move.
    const auto& calls = out.messages[first - 1].tool_calls;
    const std::size_t count = end - first;
    if (count > calls.size()) return fail("more tool results than preceding calls");
    std::vector<ToolUsePart> identities;
    identities.reserve(calls.size());
    for (const auto& call : calls) identities.push_back({call.id, {}, {}});
    std::vector<ToolResultPart> results;
    results.reserve(count);
    for (std::size_t i = first; i < end; ++i) {
      const auto& m = out.messages[i];
      results.push_back({m.tool_call_id, {}, m.tool_result_is_error, {}});
    }
    // The shared order helper owns duplicate/unknown-ID validation. Empty text
    // avoids copying payload merely to discover the order of existing messages.
    auto ordered = OrderToolResults(identities, results);
    if (!ordered || ordered->size() != count)
      return fail("tool results contain duplicate or unknown IDs");
    std::unordered_map<std::string_view, std::size_t> by_id;  // calls by their (distinct) IDs
    by_id.reserve(calls.size());
    for (std::size_t c = 0; c < calls.size(); ++c)
      if (!calls[c].id.empty()) by_id.try_emplace(calls[c].id, c);
    const auto call_of = [&by_id](std::string_view id) {
      const auto found = by_id.find(id);
      return found == by_id.end() ? kNoIndex : found->second;
    };
    const bool all_ids =
        std::ranges::all_of(identities, [](const auto& c) { return !c.id.empty(); }) &&
        std::ranges::all_of(results, [](const auto& r) { return !r.tool_use_id.empty(); });
    if (all_ids) {
      // Every result moves to its offset in the helper's order: one
      // permutation of the group's whole messages, applied in place.
      std::vector<std::size_t> supplied(calls.size(), kNoIndex);  // a call's result's offset
      for (std::size_t r = 0; r < count; ++r) {
        const std::size_t c = call_of(results[r].tool_use_id);
        if (c == kNoIndex || supplied[c] != kNoIndex)
          return fail("tool results contain duplicate or unknown IDs");
        supplied[c] = r;
      }
      std::vector<std::size_t> target(count, kNoIndex);
      for (std::size_t offset = 0; offset < count; ++offset) {
        const std::size_t c = call_of((*ordered)[offset].tool_use_id);
        if (c == kNoIndex || supplied[c] == kNoIndex || target[supplied[c]] != kNoIndex)
          return fail("tool results contain duplicate or unknown IDs");
        target[supplied[c]] = offset;
      }
      for (std::size_t r = 0; r < count; ++r) {
        while (target[r] != r) {
          const std::size_t to = target[r];
          std::swap(out.messages[first + r], out.messages[first + to]);
          std::swap(target[r], target[to]);
        }
      }
    }
    std::vector<bool> used(calls.size(), false);
    for (std::size_t i = first; i < end; ++i) {
      auto& m = out.messages[i];
      m.resolved_tool_call.reset();
      m.implied_name_call.reset();
      if (!m.tool_call_id.empty()) {
        const std::size_t index = call_of(m.tool_call_id);
        if (index == kNoIndex) return fail("tool results contain duplicate or unknown IDs");
        if (!m.tool_result_name.empty() && m.tool_result_name != calls[index].name)
          return fail("tool result name contradicts its call ID");
        m.resolved_tool_call = index;
        used[index] = true;
      }
    }
    // Calls by name: each name's calls in call order (a list through
    // next_named, from `next`), how many no result has claimed, and how many
    // unresolved results give the name.
    struct Named {
      std::size_t next = kNoIndex;
      std::size_t unused = 0;
      std::size_t supplied = 0;
    };
    std::unordered_map<std::string_view, Named> by_name;
    std::vector<std::size_t> next_named(calls.size(), kNoIndex);
    by_name.reserve(calls.size());
    for (std::size_t c = calls.size(); c-- > 0;) {
      Named& named = by_name[calls[c].name];
      next_named[c] = named.next;
      named.next = c;
      named.unused += used[c] ? 0 : 1;
    }
    const auto named_by = [&by_name](const Message& m) -> Named* {
      const auto found = by_name.find(m.tool_result_name);
      return found == by_name.end() ? nullptr : &found->second;
    };
    // Names are evidence even when no IDs were supplied (legacy function
    // messages). A name with one unmatched call resolves to it, including one
    // made unique by an earlier known-ID match; a name with no unmatched call,
    // or one call that several results give, never silently resolves.
    for (std::size_t i = first; i < end; ++i) {
      const auto& m = out.messages[i];
      if (m.resolved_tool_call || m.tool_result_name.empty()) continue;
      Named* named = named_by(m);
      if (named == nullptr) return fail("tool result name has no unmatched preceding call");
      ++named->supplied;
    }
    for (std::size_t i = first; i < end; ++i) {
      auto& m = out.messages[i];
      if (m.resolved_tool_call || m.tool_result_name.empty()) continue;
      Named& named = *named_by(m);
      if (named.unused == 0 || (named.unused == 1 && named.supplied > 1))
        return fail("tool result name has no unmatched preceding call");
      if (named.unused == 1) {
        std::size_t c = named.next;
        while (used[c]) c = next_named[c];
        m.resolved_tool_call = c;
        used[c] = true;
        named.unused = 0;
        named.supplied = 0;
      }
    }
    // Repeated function names may identify the required rendering name without
    // identifying a call occurrence, but cannot claim more calls than exist.
    for (std::size_t i = first; i < end; ++i) {
      const auto& m = out.messages[i];
      if (m.resolved_tool_call || m.tool_result_name.empty()) continue;
      const Named& named = *named_by(m);
      if (named.supplied > named.unused) return fail("too many results for the named tool calls");
    }
    const auto remaining = static_cast<std::size_t>(std::ranges::count(used, false));
    const auto unresolved = static_cast<std::size_t>(
        std::ranges::count_if(out.messages.begin() + static_cast<std::ptrdiff_t>(first),
                              out.messages.begin() + static_cast<std::ptrdiff_t>(end),
                              [](const Message& m) { return !m.resolved_tool_call; }));
    // In a complete group, reserve every name-constrained result first, then
    // associate unnamed results with the remaining calls in call order. This
    // fallback never overrides known IDs or name evidence, and moves no result.
    if (remaining == unresolved) {
      for (std::size_t i = first; i < end; ++i) {
        auto& m = out.messages[i];
        if (m.resolved_tool_call || m.tool_result_name.empty()) continue;
        Named& named = *named_by(m);
        while (named.next != kNoIndex && used[named.next]) named.next = next_named[named.next];
        if (named.next == kNoIndex) continue;
        m.resolved_tool_call = named.next;
        used[named.next] = true;
      }
      std::size_t next = 0;
      for (std::size_t i = first; i < end; ++i) {
        auto& m = out.messages[i];
        if (m.resolved_tool_call) continue;
        while (next < calls.size() && used[next]) ++next;
        if (next == calls.size()) break;
        m.resolved_tool_call = next;
        used[next++] = true;
      }
    }
    // The one name the calls no result claims share, if they share one: what
    // a renderer that needs only a result's function name may use
    // (Message::implied_name_call). An empty name is no evidence.
    std::optional<std::size_t> implied;
    std::string_view implied_name;
    for (std::size_t c = 0; c < calls.size(); ++c) {
      if (used[c]) continue;
      if (!implied_name.empty() && implied_name != calls[c].name) {
        implied_name = {};
        break;
      }
      implied_name = calls[c].name;
      implied = c;
    }
    if (implied_name.empty()) implied.reset();
    for (std::size_t i = first; i < end; ++i) {
      auto& m = out.messages[i];
      m.implied_name_call = implied;
      if (m.tool_result_is_error && !m.tool_result_normalized)
        m.content = Content("[tool error] " + m.content.text().value_or(""));
      m.tool_result_normalized = true;
    }
    first = end;
  }
  return out;
}

namespace {

constexpr auto kDeepSeekEfforts = std::to_array<std::string_view>({"low", "high", "max"});
constexpr auto kChatV2Efforts = std::to_array<std::string_view>({"high", "unknown"});
constexpr auto kQwenEfforts = std::to_array<std::string_view>({"xhigh", "medium", "low"});
constexpr auto kQwenUnslothEfforts =
    std::to_array<std::string_view>({"xhigh", "high", "medium", "low"});
// Gemma's templates read no reasoning effort: probes check it is ignored.
constexpr auto kGemmaEfforts = std::to_array<std::string_view>({"high"});

// The native renderers. Each hash is of the template's exact UTF-8 bytes
// as the checkpoint ships it, pinned by that template's fixtures
// (docs/tokenizer.md records where from); a template with another hash may
// still be recognized by probe equivalence. Unsloth's Qwen3.8 GGUF variant
// and the Gemma 4 variants have no pinned hash: probe equivalence alone
// selects them.
const std::array<Template, 10> kTemplates = {{
    {"e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645", "deepseek-v4-flash-0731",
     &RenderDeepSeekV4, StopRules{{"<｜end▁of▁sentence｜>"}}, kDeepSeekEfforts, false, true,
     &DsmlToolCalls()},
    {"872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27",
     "deepseek-v4-flash-chat-v2", &RenderDeepSeekV4ChatV2, StopRules{{"<｜end▁of▁sentence｜>"}},
     kChatV2Efforts, false, true, &DsmlToolCalls()},
    {"c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", "qwen3.8-flash-next",
     &RenderQwen38, StopRules{{"<|im_end|>", "<|endoftext|>"}}, kQwenEfforts, false, true,
     &QwenToolCalls()},
    {"", "qwen3.8-flash-next-unsloth", &RenderQwen38Unsloth,
     StopRules{{"<|im_end|>", "<|endoftext|>"}}, kQwenUnslothEfforts, false, true,
     &QwenToolCalls()},
    {"7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4", "gemma-3", &RenderGemma3,
     StopRules{{"<end_of_turn>"}}, kGemmaEfforts, false, false},
    {"ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4", "gemma-4", &RenderGemma4,
     StopRules{{"<turn|>", "<|tool_response>"}}, kGemmaEfforts, true, true, &GemmaToolCalls(),
     true},
    {"0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5", "gemma-4-e",
     &RenderGemma4E, StopRules{{"<turn|>", "<|tool_response>"}}, kGemmaEfforts, true, true,
     &GemmaToolCalls(), true},
    {"", "gemma-4-unsloth", &RenderGemma4Unsloth, StopRules{{"<turn|>", "<|tool_response>"}},
     kGemmaEfforts, true, true, &GemmaToolCalls(), true},
    {"", "gemma-4-e-unsloth", &RenderGemma4EUnsloth, StopRules{{"<turn|>", "<|tool_response>"}},
     kGemmaEfforts, true, true, &GemmaToolCalls(), true},
    {"", "gemma-4-2604", &RenderGemma4April, StopRules{{"<turn|>", "<|tool_response>"}},
     kGemmaEfforts, true, true, &GemmaToolCalls(), true},
}};

}  // namespace

std::string Error::ToString() const {
  std::string s;
  switch (rule) {
    case Rule::kUnknownTemplate:
      s = "unknown-template";
      break;
    case Rule::kUnsupported:
      s = "unsupported";
      break;
    case Rule::kInvalid:
      s = "invalid";
      break;
  }
  s += ": ";
  s += reason;
  if (item != kNoItem) {
    s += " (item ";
    s += std::to_string(item);
    s += ")";
  }
  if (template_line != 0) {
    s += " (template line ";
    s += std::to_string(template_line);
    s += ")";
  }
  return s;
}

const Template* FindTemplate(std::string_view sha256) {
  for (const Template& t : kTemplates) {
    if (!t.sha256.empty() && t.sha256 == sha256) {
      return &t;
    }
  }
  return nullptr;
}

std::span<const Template> NativeTemplates() { return kTemplates; }

std::expected<const Template*, std::string> FindTemplateForText(std::string_view template_text) {
  const std::string sha256 = base::ToHex(base::Sha256().Update(template_text).Finish());
  if (const Template* t = FindTemplate(sha256)) {
    return t;
  }
  return std::unexpected(
      std::format("no native renderer is pinned to its chat template (SHA-256 {})", sha256));
}

std::string_view PythonStrip(std::string_view text) {
  // str.isspace(): bidirectional class WS, B or S, or category Zs.
  auto space_at_start = [](std::string_view s) -> std::size_t {
    if (s.empty()) {
      return 0;
    }
    const auto b0 = static_cast<unsigned char>(s[0]);
    if ((b0 >= 0x09 && b0 <= 0x0D) || (b0 >= 0x1C && b0 <= 0x20)) {
      return 1;
    }
    for (const std::string_view w :
         {"\xC2\x85", "\xC2\xA0", "\xE1\x9A\x80", "\xE2\x80\xA8", "\xE2\x80\xA9", "\xE2\x80\xAF",
          "\xE2\x81\x9F", "\xE3\x80\x80"}) {
      if (s.starts_with(w)) {
        return w.size();
      }
    }
    if (s.size() >= 3 && s[0] == '\xE2' && s[1] == '\x80' &&
        static_cast<unsigned char>(s[2]) >= 0x80 && static_cast<unsigned char>(s[2]) <= 0x8A) {
      return 3;  // U+2000..U+200A
    }
    return 0;
  };
  auto space_at_end = [&](std::string_view s) -> std::size_t {
    for (std::size_t n = 1; n <= 3 && n <= s.size(); ++n) {
      const std::string_view tail = s.substr(s.size() - n);
      if (space_at_start(tail) == n) {
        return n;
      }
    }
    return 0;
  };
  while (const std::size_t n = space_at_start(text)) {
    text.remove_prefix(n);
  }
  while (const std::size_t n = space_at_end(text)) {
    text.remove_suffix(n);
  }
  return text;
}

std::expected<std::vector<tokenizer::TokenId>, Error> StopTokens(
    const StopRules& rules, const tokenizer::Tokenizer& tokenizer) {
  std::vector<tokenizer::TokenId> ids;
  for (std::size_t i = 0; i < rules.tokens.size(); ++i) {
    const auto id = tokenizer.Find(rules.tokens[i]);
    if (!id) {
      return std::unexpected(Error{Rule::kInvalid, "a stop token is not in the vocabulary", i});
    }
    ids.push_back(*id);
  }
  return ids;
}

std::expected<std::vector<tokenizer::TokenId>, Error> StopTokens(
    const std::vector<std::string>& texts, const tokenizer::Tokenizer& tokenizer) {
  StopRules rules;
  for (const std::string& t : texts) {
    rules.tokens.emplace_back(t);
  }
  return StopTokens(rules, tokenizer);
}

}  // namespace llmp::chat
