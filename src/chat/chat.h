// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Chat templates as native renderers (D-067): a protocol-neutral
// conversation in, the exact text the model's pinned template renders out,
// with the control tokens the template placed marked so that the tokenizer
// never takes a control token from message content
// (tokenizer::Tokenizer::EncodeMarked). A renderer is chosen by the SHA-256
// of the template's bytes; no template text is ever evaluated.
//
// Supported templates, each byte-equal to its reference renderer on the
// fixtures in tests/unit/data/chat (docs/tokenizer.md):
//
// - DeepSeek V4 Flash 0731, the GGUF's embedded template (Unsloth's port of
//   DeepSeek's encoding_dsv4.py): roles system, user, assistant and tool;
//   request-level tools, DSML tool calls, reasoning, `thinking` and
//   `reasoning_effort`.
// - DeepSeek V4 Flash "chat-v2", the community GGUF's embedded template
//   (antirez/deepseek-v4-gguf@f71f23d5): the same roles, tools, DSML tool
//   calls, reasoning and `thinking`; it has no reasoning effort, so any
//   `reasoning_effort` renders as none.
// - Qwen3.8 Flash Next, the NVFP4 checkpoint's chat_template.jinja: roles
//   system (first only), user, assistant and tool; tools with XML tool
//   calls, <think> blocks, `enable_thinking`, `reasoning_effort` and
//   `preserve_thinking`.
// - Qwen-Image 2.1's text-to-image prompt (the pinned diffusers pipeline's
//   fixed template, not a chat template): RenderQwenImagePrompt.
//
// Anything a template does not support (another role, an image, a
// DeepSeek task or latest_reminder, a message-level tool list) is refused,
// not approximated. Where the reference template raises an exception, the
// renderer returns kInvalid.

#ifndef JITLLM_CHAT_CHAT_H_
#define JITLLM_CHAT_CHAT_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/json.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::chat {

enum class Rule : std::uint8_t {
  kUnknownTemplate,  // no renderer for this template hash
  kUnsupported,      // a role, content type or option the renderer does not implement
  kInvalid,          // what the template itself rejects
};

inline constexpr std::uint64_t kNoItem = std::numeric_limits<std::uint64_t>::max();

struct Error {
  Rule rule = Rule::kInvalid;
  std::string_view reason;       // static text
  std::uint64_t item = kNoItem;  // usually a message index

  std::string ToString() const;
};

enum class Role : std::uint8_t { kSystem, kUser, kAssistant, kTool };

struct ToolCall {
  std::string name;
  // The arguments as a JSON object, parsed from the client's string: what
  // vLLM and llama.cpp hand the template.
  base::json::Value arguments;
};

// JSON values point into documents the caller keeps alive while rendering.
struct Message {
  Role role = Role::kUser;
  std::optional<std::string> content;  // text; text blocks already joined
  std::optional<std::string> reasoning_content;
  std::vector<ToolCall> tool_calls;
};

struct Conversation {
  std::vector<Message> messages;
  std::vector<base::json::Value> tools;  // each as the client sent it
  bool add_generation_prompt = true;
  // Template options; absent means the template's own default.
  std::optional<bool> enable_thinking;
  std::optional<std::string> reasoning_effort;
  std::optional<bool> preserve_thinking;  // Qwen3.8 only
};

enum class BoundaryKind : std::uint8_t {
  kPrefixEnd,         // end of the leading system and tool-definition segment
  kMessageEnd,        // end of a message's rendering
  kGenerationPrompt,  // where the generation prompt starts
};

struct Boundary {
  BoundaryKind kind = BoundaryKind::kMessageEnd;
  std::size_t offset = 0;  // in Rendered::text
};

struct Rendered {
  std::string text;
  std::vector<tokenizer::SpecialSpan> specials;  // control tokens the template placed
  std::vector<Boundary> boundaries;
};

// How generation stops for a template's model: its end-of-turn tokens (by
// text; the tokenizer gives their IDs).
struct StopRules {
  std::vector<std::string_view> tokens;
};

struct Template {
  std::string_view sha256;  // of the template's UTF-8 bytes, lower-case hex
  std::string_view name;
  std::expected<Rendered, Error> (*render)(const Conversation& conversation);
  StopRules stop;
};

// The renderer for a template hash, or nullptr: such a model has no chat
// routes (D-067).
const Template* FindTemplate(std::string_view sha256);

// The renderer for a template's text, found by its SHA-256; otherwise an
// error naming the hash. A model whose template has none is refused where
// it registers, not at its first turn.
std::expected<const Template*, std::string> FindTemplateForText(std::string_view template_text);

std::expected<Rendered, Error> RenderDeepSeekV4(const Conversation& conversation);
std::expected<Rendered, Error> RenderDeepSeekV4ChatV2(const Conversation& conversation);
std::expected<Rendered, Error> RenderQwen38(const Conversation& conversation);

// Qwen-Image 2.1's text-to-image prompt around `prompt` (an empty prompt
// becomes " ", as the pipeline does), and how many leading tokens of the
// encoder's output the pipeline drops: the system segment's tokens.
struct ImagePrompt {
  Rendered rendered;
  std::size_t drop_tokens = 0;
};
std::expected<ImagePrompt, Error> RenderQwenImagePrompt(std::string_view prompt,
                                                        const tokenizer::Tokenizer& tokenizer);

// Python's str.strip(): the characters str.isspace() accepts, which
// Jinja's `trim` removes.
std::string_view PythonStrip(std::string_view text);

// Token IDs for a stop rule's texts; an error names the first missing one.
std::expected<std::vector<tokenizer::TokenId>, Error> StopTokens(
    const StopRules& rules, const tokenizer::Tokenizer& tokenizer);

}  // namespace jitllm::chat

#endif  // JITLLM_CHAT_CHAT_H_
