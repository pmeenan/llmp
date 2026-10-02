// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Chat templates (D-067 as amended 2026-10-02): a protocol-neutral
// conversation in, the exact text the model's own template renders out,
// with the control tokens the template placed marked so that the tokenizer
// never takes a control token from message content
// (tokenizer::Tokenizer::EncodeMarked).
//
// A model's template (ChatTemplate::ForText) is rendered by:
//
// 1. A native family renderer, when the template's SHA-256 is one whose
//    fixtures it passes, or when it reproduces the template's own output,
//    rendered by the interpreter, byte for byte on every conversation of
//    the family's probe corpus (probe equivalence: a whitespace-tweaked or
//    repackaged copy of a supported template is still recognized).
// 2. Otherwise the template itself, through the bounded, sandboxed
//    Jinja-subset interpreter (chat/jinja.h). Its control tokens are those
//    the template's own text placed; message content stays text.
//
// Native renderers, each byte-equal to its reference on the fixtures in
// tests/unit/data/chat (docs/tokenizer.md):
//
// - DeepSeek V4 Flash 0731, the GGUF's embedded template (Unsloth's port of
//   DeepSeek's encoding_dsv4.py): roles system, user, assistant and tool;
//   request-level tools, DSML tool calls, reasoning, `thinking` and
//   `reasoning_effort`.
// - DeepSeek V4 Flash "chat-v2", the community GGUF's embedded template
//   (antirez/deepseek-v4-gguf@f71f23d5): the same roles, tools, DSML tool
//   calls, reasoning and `thinking`; it has no reasoning effort, so any
//   `reasoning_effort` renders as none.
// - Qwen3.8, the NVFP4 checkpoint's chat_template.jinja: roles system
//   (first only), user, assistant and tool; tools with XML tool calls,
//   <think> blocks, `enable_thinking`, `reasoning_effort` and
//   `preserve_thinking`; and Unsloth's GGUF variant of it (merged leading
//   system messages, `high` effort as `xhigh`, no user query required).
// - Qwen-Image 2.1's text-to-image prompt (the pinned diffusers pipeline's
//   fixed template, not a chat template): RenderQwenImagePrompt.
//
// A native renderer refuses what its template does not support (another
// role, an image, a DeepSeek task or latest_reminder, a message-level tool
// list) rather than approximating it. Where the template raises an
// exception, rendering returns kInvalid.

#ifndef JITLLM_CHAT_CHAT_H_
#define JITLLM_CHAT_CHAT_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/json.h"
#include "chat/jinja.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::chat {

enum class Rule : std::uint8_t {
  kUnknownTemplate,  // no renderer for this template
  kUnsupported,      // a role, content type, option or template construct not implemented
  kInvalid,          // what the template itself rejects
};

inline constexpr std::uint64_t kNoItem = std::numeric_limits<std::uint64_t>::max();

struct Error {
  Rule rule = Rule::kInvalid;
  std::string_view reason;          // static text
  std::uint64_t item = kNoItem;     // usually a message index
  std::uint32_t template_line = 0;  // where an interpreted template failed, from 1

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

// A native renderer and the template hash its fixtures pin.
struct Template {
  std::string_view sha256;  // of the template's UTF-8 bytes, lower-case hex
  std::string_view name;
  std::expected<Rendered, Error> (*render)(const Conversation& conversation);
  StopRules stop;
  // The reasoning_effort values the renderer accepts, which probes try.
  std::span<const std::string_view> probe_efforts;
};

// The native renderer for a template hash, or nullptr.
const Template* FindTemplate(std::string_view sha256);

// The native renderer pinned to a template's exact text by its SHA-256;
// otherwise an error naming the hash. For harnesses that require a pinned
// renderer; serving chooses with ChatTemplate::ForText.
std::expected<const Template*, std::string> FindTemplateForText(std::string_view template_text);

// The native renderers, for probe recognition.
std::span<const Template> NativeTemplates();

// What a template needs from the model's tokenizer: the BOS and EOS texts
// (bos_token and eos_token, trusted) and the control tokens' texts, which
// an interpreted template's own text may place.
struct TokenFacts {
  std::optional<std::string> bos;
  std::optional<std::string> eos;
  std::vector<std::string> control;

  static TokenFacts From(const tokenizer::Tokenizer& tokenizer);
};

// How a model's template is rendered (see the top of this file).
class ChatTemplate {
 public:
  enum class How : std::uint8_t {
    kNativeByHash,   // a native renderer pinned to this exact template
    kNativeByProbe,  // a native renderer equal to this template on its probe corpus
    kInterpreted,    // the template itself, through the interpreter
  };

  // The renderer for a template's text; an error names the template's hash
  // and what the interpreter could not accept.
  static std::expected<ChatTemplate, std::string> ForText(std::string_view template_text,
                                                          TokenFacts tokens);

  ChatTemplate(ChatTemplate&&) noexcept;
  ChatTemplate& operator=(ChatTemplate&&) noexcept;
  ChatTemplate(const ChatTemplate&) = delete;
  ChatTemplate& operator=(const ChatTemplate&) = delete;
  ~ChatTemplate();

  // `now` is strftime_now's clock for an interpreted template.
  std::expected<Rendered, Error> Render(const Conversation& conversation,
                                        const std::optional<jinja::CivilTime>& now = {}) const;

  How how() const { return how_; }
  std::string_view sha256() const { return sha256_; }
  // The native renderer's name, or "interpreted".
  std::string_view name() const;
  // End-of-turn token texts: the native renderer's; for an interpreted
  // template, the control token it places after an assistant message (the
  // tokenizer's EOS is added where the model registers).
  const std::vector<std::string>& stop() const { return stop_; }

 private:
  ChatTemplate() = default;
  How how_ = How::kInterpreted;
  std::string sha256_;
  const Template* native_ = nullptr;
  std::unique_ptr<jinja::Template> program_;
  TokenFacts tokens_;
  std::vector<std::string> stop_;
};

// The interpreter's rendering of a conversation, with the specials its
// template text placed and the generation prompt's boundary where the
// rendering without it is a prefix of the rendering with it.
// `budget` lowers the interpreter's bounds (probes use it).
std::expected<Rendered, Error> RenderInterpreted(const jinja::Template& program,
                                                 const Conversation& conversation,
                                                 const TokenFacts& tokens,
                                                 const std::optional<jinja::CivilTime>& now,
                                                 jinja::Budget budget = {});

// A native renderer's output equals the template's own on the family's
// probe corpus (docs/tokenizer.md): the evidence for How::kNativeByProbe.
// Probes draw on one pool, a single rendering's bounds: `spent` (its own
// when null) carries what earlier probes used, across families; once it is
// spent the template is no family's.
bool ProbeEquivalent(const Template& native, const jinja::Template& program,
                     const TokenFacts& tokens, jinja::Usage* spent = nullptr);

std::expected<Rendered, Error> RenderDeepSeekV4(const Conversation& conversation);
std::expected<Rendered, Error> RenderDeepSeekV4ChatV2(const Conversation& conversation);
std::expected<Rendered, Error> RenderQwen38(const Conversation& conversation);
std::expected<Rendered, Error> RenderQwen38Unsloth(const Conversation& conversation);

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

// Token IDs for stop texts; an error names the first missing one.
std::expected<std::vector<tokenizer::TokenId>, Error> StopTokens(
    const StopRules& rules, const tokenizer::Tokenizer& tokenizer);
std::expected<std::vector<tokenizer::TokenId>, Error> StopTokens(
    const std::vector<std::string>& texts, const tokenizer::Tokenizer& tokenizer);

}  // namespace jitllm::chat

#endif  // JITLLM_CHAT_CHAT_H_
