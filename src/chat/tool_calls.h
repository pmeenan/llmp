// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Generated output only: never feed rendered prompts or conversation content
// through this interpreter. Format syntax is data, independent of model identity.
#ifndef LLMP_CHAT_TOOL_CALLS_H_
#define LLMP_CHAT_TOOL_CALLS_H_

#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "base/json.h"
#include "chat/chat.h"

namespace llmp::chat {

using ToolStatus = std::expected<void, std::string>;

enum class ToolChoiceKind { kAuto, kNone, kRequired, kNamed };
struct ToolChoice {
  ToolChoiceKind kind = ToolChoiceKind::kAuto;
  std::string name;
};

// Protocol decoders supply this neutral contract; neither route translates
// through another route's JSON. Values are copied into owned stable documents.
struct ToolDefinition {
  std::string name;
  std::string description;
  std::string parameters_json = "{}";
  bool strict = false;
};
// Conservative peak bytes to prepay before constructing the owned request or
// output interpreter. The caller derives limits.max_bytes from real request
// memory/context. Includes temporary parsing/escaping; excludes the caller's
// input and events after the sink takes ownership. Empty means size overflow.
std::optional<std::size_t> ToolRequestBytes(std::span<const ToolDefinition> definitions);
struct ToolCallFormat;
std::optional<std::size_t> ToolOutputBytes(const ToolCallFormat& format,
                                           std::size_t generated_bytes);

class ToolRequest {
 public:
  // max_bytes bounds aggregate raw definition fields. Normalized envelopes
  // use the checked expansion included in ToolRequestBytes; other JSON limits
  // also apply to those envelopes.
  static std::expected<ToolRequest, std::string> Create(std::span<const ToolDefinition> definitions,
                                                        ToolChoice choice, bool parallel,
                                                        const base::json::Limits& limits);
  const ToolChoice& choice() const { return data_->choice; }
  bool parallel() const { return data_->parallel; }
  bool Declared(std::string_view name) const;
  // Checks a partial name plus its declarative terminator without allocating.
  std::size_t NameBytes() const;
  // Exact name identity in the selected grammar. Named choice checks only
  // its exposed definition; none is harmless. Caller prepays NameBytes()
  // scratch (the object scanner); JSON envelopes escape arbitrary names.
  ToolStatus ValidateNames(const ToolCallFormat& format) const;
  std::size_t TemplateBytes() const;
  bool NamePrefix(std::string_view prefix, std::string_view terminator) const;
  std::optional<base::json::Value> Parameter(std::string_view name, std::string_view key) const;
  // OpenAI-shaped definitions are the template contract, owned by this request.
  // Named choice presents only its selected definition. None presents no tools.
  std::vector<base::json::Value> TemplateTools() const;

 private:
  friend std::optional<std::size_t> ToolRequestBytes(std::span<const ToolDefinition> definitions);
  struct Definition {
    std::string name;
    std::shared_ptr<const base::json::Document> owner;
    std::size_t bytes = 0;
  };
  struct Data {
    ToolChoice choice;
    bool parallel = true;
    std::vector<Definition> definitions;
  };
  std::shared_ptr<const Data> data_;
};

enum class ToolArguments { kJsonEnvelope, kTaggedParameters, kObjectNotation };
struct ToolCallFormat {
  std::string block_begin = {}, block_end = {};  // optional wrappers; inner calls stand alone
  std::string call_begin = {}, call_end = {};
  // Name immediately after call_begin, terminated by name_end. For an object
  // envelope, name_key/arguments_key select fields instead.
  std::string name_end = {};
  std::string name_key = "name", arguments_key = "arguments";
  ToolArguments arguments = ToolArguments::kJsonEnvelope;
  // Bare JSON object: call_begin/end delimit the envelope itself. Auto mode
  // preserves objects without both call fields as ordinary answer text.
  bool bare_envelope = false;
  std::string parameter_begin = {}, parameter_name_end = {}, parameter_end = {};
  // An optional parameter-header suffix says string or JSON explicitly.
  std::string string_suffix = {}, json_suffix = {};
  bool trim_wrapping_newline = false;
  bool opener_ends_reasoning = false;
  std::string reasoning_end = {};
  std::string reasoning_begin = {};  // actual rendered generation-prefix bytes
  // Object notation: strings use this delimiter; keys may be bare. null_word
  // accepts a template's spelling in addition to JSON null.
  std::string quote = "\"", null_word = "null";
};

// Declarations from the checked-in renderers and pinned templates, shared by
// their hash/probe registrations. No family selection in the interpreter.
const ToolCallFormat& QwenToolCalls();
const ToolCallFormat& DsmlToolCalls();
const ToolCallFormat& GemmaToolCalls();
const ToolCallFormat& JsonToolCalls();
const ToolCallFormat* ToolCallsForTemplate(std::string_view sha256);

// The later sampler/verify integration binds these texts to tokenizer IDs.
// This is a constraint intent, not evidence that required/named decoding is
// already enforced. End-of-reasoning is the optional forcing boundary.
struct ToolTrigger {
  std::string begin;
  std::optional<std::string> bound;
  bool required = false;
  std::optional<std::string> name;
  bool ends_reasoning = false;
};
ToolTrigger ToolTriggerFor(const ToolCallFormat& format, const ToolRequest& request);

struct ToolReasoningEnd {};
using ToolEvent = std::variant<TextPart, ToolUsePart, ToolReasoningEnd>;
enum class ToolFinish { kStop, kLength };

class ToolInterpreter {
 public:
  // Caller prepays ToolOutputBytes(format, limits.max_bytes) before construction.
  // max_bytes bounds raw generated bytes. Converted JSON uses the checked
  // expansion already included in ToolOutputBytes; the other JSON limits
  // still bound depth, values and decoded strings. No call-count or time limit.
  ToolInterpreter(ToolCallFormat format, ToolRequest request, base::json::Limits limits);
  struct Flow {
    enum Action { kContinue, kPause, kCancel } action;
    Flow(bool go) : action(go ? kContinue : kCancel) {}
    Flow(Action value) : action(value) {}
  };
  using Sink = std::function<Flow(ToolEvent)>;  // every event is accepted exactly once
  // A pause retains the unconsumed input suffix; Resume drains it before new input.
  ToolStatus Resume(const Sink& sink);
  bool paused() const { return paused_; }
  bool ended() const { return ended_; }
  // Sampling probes borrow pending argument bytes; only lexical cursor state is
  // cloned per candidate. Syntax-only cursors do not build argument documents.
  bool Allows(std::string_view candidate) const;
  void SyntaxOnly() { syntax_only_ = true; }
  void Reasoning(bool active) { reasoning_active_ = active; }
  bool reasoning() const { return reasoning_active_; }
  std::optional<std::string_view> error() const {
    return error_ ? std::optional<std::string_view>(*error_) : std::nullopt;
  }
  ToolStatus Push(std::string_view generated, const Sink& sink);
  ToolStatus Finish(ToolFinish reason, const Sink& sink);
  std::size_t calls() const { return calls_; }

 private:
  ToolStatus Drain(bool final, const Sink& sink);
  ToolStatus Emit(ToolEvent event, const Sink& sink);
  ToolStatus Fail(std::string error);
  ToolCallFormat format_;
  ToolRequest request_;
  base::json::Limits limits_;
  std::string pending_;
  std::size_t received_ = 0, calls_ = 0, block_calls_ = 0, scan_at_ = 0;
  std::vector<char> scan_nesting_;
  bool scan_quoted_ = false, scan_escaped_ = false;
  std::size_t tag_name_end_ = std::string::npos, tag_header_search_ = 0;
  unsigned tag_phase_ = 0;
  int tag_json_ = -1;
  bool in_call_ = false, in_block_ = false, ended_ = false, reasoning_ended_ = false, busy_ = false;
  bool paused_ = false, syntax_only_ = false, reasoning_active_ = false;
  std::optional<ToolFinish> finishing_;
  std::optional<std::string> error_;
};

// Reorder a parallel result group only when IDs prove its association. Missing
// IDs are tolerated in supplied order; they never invent a stable association.
// A duplicate/unknown supplied ID is an error. Returned parts own all payloads.
std::expected<std::vector<ToolResultPart>, std::string> OrderToolResults(
    std::span<const ToolUsePart> calls, std::span<const ToolResultPart> results);

}  // namespace llmp::chat
#endif  // LLMP_CHAT_TOOL_CALLS_H_
