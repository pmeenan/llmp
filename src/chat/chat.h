// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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
// - Gemma 3, the template Unsloth's and the converted Gemma 3 GGUFs carry:
//   roles system (first, prefixed to the first user turn), user and
//   assistant (`model`) alternating, as the template requires.
// - Gemma 4, google/gemma-4-31B-it's template (2026-07-15) and the E2B/E4B
//   checkpoints': roles system, user, assistant and tool; tool declarations
//   in Gemma's notation, tool calls and the tool results after them, the
//   thought channel, `enable_thinking` and `preserve_thinking`; and
//   variants chosen by probe: Unsloth's two (string tool-call arguments
//   render) and Google's 2026-04-28 template (NVIDIA's NVFP4 checkpoints).
// - Qwen-Image 2.1's text-to-image prompt (the pinned diffusers pipeline's
//   fixed template, not a chat template): RenderQwenImagePrompt.
//
// A native renderer refuses what its template does not support (another
// role, an image, a DeepSeek task or latest_reminder, a message-level tool
// list) rather than approximating it (kUnsupported), and ChatTemplate then
// renders that case through the interpreter. Where the template raises an
// exception, rendering returns kInvalid. A native rendering is bounded as
// the interpreter's output is (Conversation::max_render_bytes, which
// serving sets for each request within the model's bound).

#ifndef LLMP_CHAT_CHAT_H_
#define LLMP_CHAT_CHAT_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "base/json.h"
#include "chat/jinja.h"
#include "tokenizer/tokenizer.h"

namespace llmp::chat {

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
  // A native rendering grew past the interpreter's output bound: the
  // interpreter would refuse it too, so ChatTemplate does not retry it there.
  bool bound = false;
  // The caller's cancellation stopped an interpreted rendering (its request
  // ended; jinja::Budget::cancelled).
  bool cancelled = false;

  std::string ToString() const;
};

enum class Role : std::uint8_t { kSystem, kUser, kAssistant, kTool };

// Ordered content is owned by the conversation, including tool JSON and image
// sources. Image payloads are descriptions only: text rendering never fetches them.
struct TextPart {
  std::string text;
};
struct ToolUsePart {
  std::string id;
  std::string name;
  std::string input_json;
};
struct ToolResultPart {
  std::string tool_use_id;
  std::string text;
  bool is_error = false;
};
struct ImagePart {
  std::string media_type;
  std::string source;
};
using MessagePart = std::variant<TextPart, ToolUsePart, ToolResultPart, ImagePart>;

class Content {
 public:
  Content() = default;
  Content(std::nullopt_t) {}
  Content(std::string text) : value_(std::optional<std::string>(std::move(text))) {}
  Content(const char* text) : Content(std::string(text)) {}
  Content(std::optional<std::string> text) : value_(std::move(text)) {}
  Content(std::vector<MessagePart> parts) : value_(std::move(parts)) {}

  // Scalar text is a template projection, not a flattening accessor. Callers
  // must project structured content before using it.
  const std::optional<std::string>& text() const { return std::get<0>(value_); }
  const std::vector<MessagePart>* parts() const { return std::get_if<1>(&value_); }
  std::uint64_t PayloadBytes() const;
  std::uint64_t StorageBytes() const;
  // Extra owned storage and JSON parser working memory during projection.
  std::uint64_t ProjectionBytes() const;

 private:
  std::variant<std::optional<std::string>, std::vector<MessagePart>> value_;
};

struct ToolCall {
  std::string name;
  // The arguments as a JSON object, parsed from the client's string: what
  // vLLM and llama.cpp hand the template.
  base::json::Value arguments;
  std::string id = {};
  std::shared_ptr<const base::json::Document> owner = {};
};

// JSON values point into documents the caller keeps alive while rendering.
struct Message {
  Role role = Role::kUser;
  Content content;
  std::optional<std::string> reasoning_content = {};
  std::vector<ToolCall> tool_calls = {};
  std::string tool_call_id = {};
};

struct Conversation {
  std::vector<Message> messages;
  std::vector<base::json::Value> tools;  // each as the client sent it
  bool add_generation_prompt = true;
  // Template options; absent means the template's own default.
  std::optional<bool> enable_thinking;
  std::optional<std::string> reasoning_effort;
  std::optional<bool> preserve_thinking;  // Qwen3.8 and Gemma 4 read it
  // A rendering's most bytes (an interpreted one's at most its template's
  // jinja::Limits::max_output_bytes too), and an interpreted rendering's
  // values' (0: its template's): serving sets them from what the request is
  // charged, within the model's bound (Llm::RenderCharge, D-102).
  std::size_t max_render_bytes = jinja::Limits{}.max_output_bytes;
  std::size_t max_live_bytes = 0;
};

// The shared template boundary. Text parts concatenate exactly; tools project
// to template fields with IDs preserved. Unrepresentable ordering, error results
// and images are rejected before any native or interpreted template runs.
bool HasStructuredContent(const Conversation& conversation);
std::expected<Conversation, Error> ProjectContent(const Conversation& conversation,
                                                  bool supports_tools = true);

using jinja::Boundary;
using jinja::BoundaryKind;

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
  // Probes first try the conversations where the family's template variants
  // part (Gemma 4's): unparsed tool-call arguments, a system message without
  // content, properties named like schema keys, blank content beside tool
  // results (template.cc's VariantConversations).
  bool probe_variants = false;
  // Whether structured tool parts can be projected without losing their work.
  bool supports_tools = true;
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
  // and what the interpreter could not accept. `limits` bound its
  // interpreted renderings (serving derives their memory bounds,
  // runtime/intake_limits.h RenderBytes); max_bytes() is their output's.
  static std::expected<ChatTemplate, std::string> ForText(std::string_view template_text,
                                                          TokenFacts tokens,
                                                          const jinja::Limits& limits = {});

  ChatTemplate(ChatTemplate&&) noexcept;
  ChatTemplate& operator=(ChatTemplate&&) noexcept;
  ChatTemplate(const ChatTemplate&) = delete;
  ChatTemplate& operator=(const ChatTemplate&) = delete;
  ~ChatTemplate();

  // `now` is strftime_now's clock for an interpreted template. A case a
  // native renderer does not implement (kUnsupported, short of the output
  // bound) renders through the interpreter, from the template's own text;
  // the stop tokens stay the native renderer's. An interpreted rendering
  // has no time or work bound (D-102): `cancelled`, when given, is asked
  // as it goes (jinja::Budget::cancelled), and true ends it (an Error
  // with `cancelled`). A native rendering is linear in its conversation.
  std::expected<Rendered, Error> Render(const Conversation& conversation,
                                        const std::optional<jinja::CivilTime>& now = {},
                                        const std::function<bool()>* cancelled = nullptr) const;

  How how() const { return how_; }
  // Whether a rendering may run through the interpreter (an interpreted
  // template, or a native one's fallback).
  bool interprets() const { return program_ != nullptr; }
  // The interpreter's live-value bound (its Limits').
  std::size_t max_live_bytes() const { return max_live_bytes_; }
  // An interpreted rendering's output bound (its Limits'); a conversation's
  // max_render_bytes may lower it.
  std::size_t max_bytes() const { return max_bytes_; }
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
  std::size_t max_bytes_ = jinja::Limits{}.max_output_bytes;
  std::size_t max_live_bytes_ = jinja::Limits{}.max_live_bytes;
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

// All of one registration's probes: a heuristic with its own fixed basis
// (what one rendering's former bounds were, about 6 s of a GB10 core),
// since renderings themselves are no longer capped (D-102).
inline constexpr jinja::Usage kProbePool{.steps = 50'000'000,
                                         .work_bytes = std::uint64_t{2} << 30U};

// A native renderer's output equals the template's own on the family's
// probe corpus (docs/tokenizer.md): the evidence for How::kNativeByProbe.
// Probes draw on one pool, kProbePool: `spent` (its own when null) carries
// what earlier probes used, across families; once it is spent the template
// is no family's.
bool ProbeEquivalent(const Template& native, const jinja::Template& program,
                     const TokenFacts& tokens, jinja::Usage* spent = nullptr);

std::expected<Rendered, Error> RenderDeepSeekV4(const Conversation& conversation);
std::expected<Rendered, Error> RenderDeepSeekV4ChatV2(const Conversation& conversation);
std::expected<Rendered, Error> RenderQwen38(const Conversation& conversation);
std::expected<Rendered, Error> RenderQwen38Unsloth(const Conversation& conversation);
std::expected<Rendered, Error> RenderGemma3(const Conversation& conversation);
std::expected<Rendered, Error> RenderGemma4(const Conversation& conversation);
std::expected<Rendered, Error> RenderGemma4E(const Conversation& conversation);
std::expected<Rendered, Error> RenderGemma4Unsloth(const Conversation& conversation);
std::expected<Rendered, Error> RenderGemma4EUnsloth(const Conversation& conversation);
std::expected<Rendered, Error> RenderGemma4April(const Conversation& conversation);

// Image prompts return a renderer-authored special-span index at the end
// of the dropped system prefix. Encode the complete rendering once with
// EncodeMarked(span_tokens): span_tokens[drop_span] is the exact token count
// to drop, under that encoding's normalization/splitting policy.
struct ImagePrompt {
  Rendered rendered;
  std::size_t drop_span = 0;
};
// The native image renderer's exact output bytes (empty becomes one space).
// No escaping is performed. Saturates on overflow; fund output, tokenization
// and span-token indexes before rendering. Every registered renderer honors it.
std::uint64_t ImagePromptBytesBound(std::size_t input_bytes);
std::expected<ImagePrompt, Error> RenderQwenImagePrompt(std::string_view prompt);

using ImagePromptRenderer = std::expected<ImagePrompt, Error> (*)(std::string_view prompt);
ImagePromptRenderer FindImagePromptRenderer(std::string_view name);

// Python's str.strip(): the characters str.isspace() accepts, which
// Jinja's `trim` removes.
std::string_view PythonStrip(std::string_view text);

// Token IDs for stop texts; an error names the first missing one.
std::expected<std::vector<tokenizer::TokenId>, Error> StopTokens(
    const StopRules& rules, const tokenizer::Tokenizer& tokenizer);
std::expected<std::vector<tokenizer::TokenId>, Error> StopTokens(
    const std::vector<std::string>& texts, const tokenizer::Tokenizer& tokenizer);

}  // namespace llmp::chat

#endif  // LLMP_CHAT_CHAT_H_
