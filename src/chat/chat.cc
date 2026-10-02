// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "chat/chat.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::chat {
namespace {

constexpr auto kDeepSeekEfforts = std::to_array<std::string_view>({"low", "high", "max"});
constexpr auto kChatV2Efforts = std::to_array<std::string_view>({"high", "unknown"});
constexpr auto kQwenEfforts = std::to_array<std::string_view>({"xhigh", "medium", "low"});
constexpr auto kQwenUnslothEfforts =
    std::to_array<std::string_view>({"xhigh", "high", "medium", "low"});

// The native renderers. Each hash is of the template's exact UTF-8 bytes
// as the checkpoint ships it, pinned by that template's fixtures
// (docs/tokenizer.md records where from); a template with another hash may
// still be recognized by probe equivalence. Unsloth's Qwen3.8 GGUF variant
// has no pinned hash: probe equivalence alone selects it.
const std::array<Template, 4> kTemplates = {{
    {"e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645", "deepseek-v4-flash-0731",
     &RenderDeepSeekV4, StopRules{{"<｜end▁of▁sentence｜>"}}, kDeepSeekEfforts},
    {"872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27",
     "deepseek-v4-flash-chat-v2", &RenderDeepSeekV4ChatV2, StopRules{{"<｜end▁of▁sentence｜>"}},
     kChatV2Efforts},
    {"c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", "qwen3.8-flash-next",
     &RenderQwen38, StopRules{{"<|im_end|>", "<|endoftext|>"}}, kQwenEfforts},
    {"", "qwen3.8-flash-next-unsloth", &RenderQwen38Unsloth,
     StopRules{{"<|im_end|>", "<|endoftext|>"}}, kQwenUnslothEfforts},
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

}  // namespace jitllm::chat
