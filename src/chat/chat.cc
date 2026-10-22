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
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "base/sha256.h"
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
          return AddBytes(bytes(p.tool_use_id), bytes(p.text));
        } else {
          return AddBytes(bytes(p.media_type), bytes(p.source));
        }
      },
      part);
}

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
    bytes = AddBytes(bytes, message.tool_call_id.capacity());
    bytes = AddBytes(bytes, MultiplyBytes(message.tool_calls.capacity(), sizeof(ToolCall)));
    for (const ToolCall& call : message.tool_calls) {
      bytes = AddBytes(bytes, AddBytes(call.name.capacity(), call.id.capacity()));
      // Legacy JSON values remain views and their owners remain shared. Only
      // structured tool parts parse new documents, in ProjectionBytes above.
    }
  }
  return bytes;
}
}  // namespace

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
  auto bytes = AddBytes(AddBytes(StorageBytes(), MultiplyBytes(2, PayloadBytes())),
                        MultiplyBytes(p->size(), sizeof(ToolCall)));
  for (const auto& part : *p) {
    if (const auto* use = std::get_if<ToolUsePart>(&part)) {
      bytes = AddBytes(bytes, AddBytes(base::json::ParseWorkingBytes(use->input_json),
                                       sizeof(base::json::Document)));
    }
  }
  return bytes;
}

bool HasStructuredContent(const Conversation& conversation) {
  return std::ranges::any_of(conversation.messages,
                             [](const Message& m) { return m.content.parts() != nullptr; });
}

std::expected<Conversation, Error> ProjectContent(const Conversation& conversation,
                                                  bool supports_tools) {
  // Bound the projection before copying or parsing. Serving separately funds
  // its allocation from request memory; direct callers retain this render bound.
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
  if (conversation.max_live_bytes != 0 && storage > conversation.max_live_bytes) {
    return std::unexpected(Error{.rule = Rule::kUnsupported,
                                 .reason = "message projection exceeds the live memory bound",
                                 .bound = true});
  }
  Conversation out = conversation;
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
    m.tool_calls.reserve(
        static_cast<std::size_t>(std::ranges::count_if(*parts, [](const MessagePart& part) {
          return std::holds_alternative<ToolUsePart>(part);
        })));
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
        if (m.role != Role::kAssistant || use->id.empty() || use->name.empty()) {
          return fail("tool use requires an assistant, an ID and a name");
        }
        if (std::ranges::any_of(m.tool_calls,
                                [&](const ToolCall& call) { return call.id == use->id; })) {
          return fail("tool use IDs must be distinct within a message");
        }
        auto document = base::json::Parse(use->input_json);
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
        if (parts->size() != 1 || (m.role != Role::kUser && m.role != Role::kTool) ||
            result->tool_use_id.empty() || result->is_error) {
          return fail("tool results require a single successful result in a user or tool message");
        }
        std::size_t previous = i;
        while (previous > 0 && out.messages[previous - 1].role == Role::kTool) {
          --previous;
          if (out.messages[previous].tool_call_id == result->tool_use_id) {
            return fail("a tool use cannot receive duplicate results");
          }
        }
        if (previous == 0 ||
            !std::ranges::any_of(out.messages[previous - 1].tool_calls, [&](const ToolCall& call) {
              return call.id == result->tool_use_id;
            })) {
          return fail("a tool result must reference the preceding assistant's tool use");
        }
        const auto& preceding_calls = out.messages[previous - 1].tool_calls;
        const std::size_t result_index = i - previous;
        if (result_index >= preceding_calls.size() ||
            preceding_calls[result_index].id != result->tool_use_id) {
          return fail("tool results must follow their call order for text templates");
        }
        m.role = Role::kTool;
        m.tool_call_id = result->tool_use_id;
        text = result->text;
      } else {
        return fail("image content requires a multimodal renderer");
      }
    }
    m.content = calls && !has_text ? Content(std::nullopt) : Content(std::move(text));
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
     &RenderDeepSeekV4, StopRules{{"<｜end▁of▁sentence｜>"}}, kDeepSeekEfforts},
    {"872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27",
     "deepseek-v4-flash-chat-v2", &RenderDeepSeekV4ChatV2, StopRules{{"<｜end▁of▁sentence｜>"}},
     kChatV2Efforts},
    {"c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", "qwen3.8-flash-next",
     &RenderQwen38, StopRules{{"<|im_end|>", "<|endoftext|>"}}, kQwenEfforts},
    {"", "qwen3.8-flash-next-unsloth", &RenderQwen38Unsloth,
     StopRules{{"<|im_end|>", "<|endoftext|>"}}, kQwenUnslothEfforts},
    {"7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4", "gemma-3", &RenderGemma3,
     StopRules{{"<end_of_turn>"}}, kGemmaEfforts, false, false},
    {"ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4", "gemma-4", &RenderGemma4,
     StopRules{{"<turn|>", "<|tool_response>"}}, kGemmaEfforts, true},
    {"0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5", "gemma-4-e",
     &RenderGemma4E, StopRules{{"<turn|>", "<|tool_response>"}}, kGemmaEfforts, true},
    {"", "gemma-4-unsloth", &RenderGemma4Unsloth, StopRules{{"<turn|>", "<|tool_response>"}},
     kGemmaEfforts, true},
    {"", "gemma-4-e-unsloth", &RenderGemma4EUnsloth, StopRules{{"<turn|>", "<|tool_response>"}},
     kGemmaEfforts, true},
    {"", "gemma-4-2604", &RenderGemma4April, StopRules{{"<turn|>", "<|tool_response>"}},
     kGemmaEfforts, true},
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
