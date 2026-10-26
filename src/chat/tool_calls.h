// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Generated output only: never feed rendered prompts or conversation content
// through this interpreter. Format syntax is data, independent of model identity.
#ifndef LLMP_CHAT_TOOL_CALLS_H_
#define LLMP_CHAT_TOOL_CALLS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
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
  // Accepted and ignored, as vLLM does: arguments are not schema-validated.
  bool strict = false;
  // The client's function object as JSON text, when the route has one. The
  // template then receives exactly the fields the client sent; otherwise the
  // fields above are normalized into one.
  std::string function_json = {};
};
// Conservative peak bytes to prepay before constructing the owned request.
// The caller derives limits.max_bytes from real request memory/context.
// Includes temporary parsing/escaping; excludes the caller's input. Empty
// means size overflow.
std::optional<std::size_t> ToolRequestBytes(std::span<const ToolDefinition> definitions);
struct ToolCallFormat;
// Peak bytes one request's tool-output state holds while its open call (or
// held text) is `open_bytes` raw bytes: the output parser's and the
// sampler's buffered bytes with their speculative copies, and one call's
// conversion and JSON working set when it closes. Serving funds this as the
// output grows (never from max_tokens). Empty means size overflow.
std::optional<std::size_t> ToolOutputBytes(const ToolCallFormat& format, std::size_t open_bytes);

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
  std::size_t NameBytes() const;
  // Exact name identity in the selected grammar. Named choice checks only
  // its exposed definition; none is harmless. Caller prepays NameBytes()
  // scratch (the object scanner); JSON envelopes escape arbitrary names.
  ToolStatus ValidateNames(const ToolCallFormat& format) const;
  // Each exposed definition's name, in declaration order (for admission
  // checks that need the tokenizer).
  std::vector<std::string_view> ExposedNames() const;
  std::size_t TemplateBytes() const;
  // Checks a partial name plus its declarative terminator without allocating.
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
    // Name -> index into definitions; keys borrow definitions' names.
    std::unordered_map<std::string_view, std::size_t> index;
    std::size_t name_bytes = 0;  // the longest name (probes ask on every byte)
  };
  const Definition* Find(std::string_view name) const;
  std::shared_ptr<const Data> data_;
};

enum class ToolArguments { kJsonEnvelope, kTaggedParameters, kObjectNotation };
struct ToolParameterHeader {
  std::string begin, name_end;
};
struct ToolCallFormat {
  std::string block_begin = {}, block_end = {};  // optional wrappers; inner calls stand alone
  std::string call_begin = {}, call_end = {};
  // Name immediately after call_begin, terminated by name_end. For an object
  // envelope, name_key/arguments_key select fields instead.
  std::string name_end = {};
  std::string name_key = "name", arguments_key = "arguments";
  ToolArguments arguments = ToolArguments::kJsonEnvelope;
  // Bare JSON object: call_begin/end delimit the envelope itself. Only an
  // opener at the start of the answer (after whitespace) or right after a
  // call is a candidate (as vLLM and llama.cpp); a candidate that is not a
  // call is ordinary answer text in auto mode.
  bool bare_envelope = false;
  std::string parameter_begin = {}, parameter_name_end = {}, parameter_end = {};
  // An optional parameter-header suffix says string or JSON explicitly.
  std::string string_suffix = {}, json_suffix = {};
  bool trim_wrapping_newline = false;
  bool opener_ends_reasoning = false;
  std::string reasoning_end = {};
  // The rendered reasoning opener. Output that starts with it opens
  // reasoning the template did not prefill (the model's own choice).
  std::string reasoning_begin = {};
  // Object notation: strings use this delimiter; keys may be bare. null_word
  // accepts a template's spelling in addition to JSON null.
  std::string quote = "\"", null_word = "null";
  // Alternate tagged spellings are recognized only between parameters, with
  // one exception: a call that ends with an unclosed parameter is closed by
  // the first call-end alias in that value. They never change the rendered form.
  std::vector<std::string> call_end_aliases = {};
  std::vector<ToolParameterHeader> parameter_aliases = {};
  // Exact route-only stop spellings consumed by this tool protocol. Actual
  // model/EOS stops still end generation; arbitrary substrings do not qualify.
  std::vector<std::string> continued_route_stops = {};
};

// Declarations from the checked-in renderers and pinned templates, shared by
// their hash/probe registrations. No family selection in the interpreter.
const ToolCallFormat& QwenToolCalls();
const ToolCallFormat& DsmlToolCalls();
const ToolCallFormat& GemmaToolCalls();
const ToolCallFormat& JsonToolCalls();
const ToolCallFormat* ToolCallsForTemplate(std::string_view sha256);

// The sampler binds these texts to tokenizer IDs. End-of-reasoning is the
// optional forcing boundary.
struct ToolTrigger {
  std::string begin;
  std::optional<std::string> bound;
  bool required = false;
  std::optional<std::string> name;
  bool ends_reasoning = false;
};
ToolTrigger ToolTriggerFor(const ToolCallFormat& format, const ToolRequest& request);

struct ToolReasoningBegin {};
struct ToolReasoningEnd {};
using ToolEvent = std::variant<TextPart, ToolUsePart, ToolReasoningEnd, ToolReasoningBegin>;
enum class ToolFinish { kStop, kLength };

namespace detail {
// Copyable lexical cursors only; argument bytes remain in the interpreter's
// buffer. No member allocates: a probe copies the whole cursor.
struct ToolScan {
  std::size_t at = 0, depth = 0;
  bool quoted = false, escaped = false;
  // A tagged value's first call-end alias (raw position, its bytes): it closes
  // the call only if the value never closes before generation ends.
  std::size_t alias_at = std::string::npos, alias_bytes = 0;
};
struct ToolEnvelopeInfo {
  bool name = false, arguments = false, allowed = false;
};
struct ToolEnvelopeCursor {
  enum class Phase { kBetween, kKey, kColon, kValueStart, kValue, kStopped };
  Phase phase = Phase::kBetween;
  std::size_t at = 0, key_start = 0, value_start = 0, trim_end = 0, depth = 0;
  bool escaped = false, quoted = false, name_key = false, arguments_key = false;
  ToolEnvelopeInfo committed, projected;
};
// Object notation (GemmaToolCalls): one byte-stepped grammar validates
// sampled prefixes and converts complete calls, so they cannot disagree.
// Nesting is bounded by limits.max_depth and at most 64 levels.
struct ToolObjectCursor {
  enum class State : std::uint8_t {
    kValue,         // a value is next
    kValueOrClose,  // after '[': a value or ']'
    kKeyOrClose,    // after '{': a key or '}'
    kKey,           // after ',' in an object: a key
    kQuoteOpen,     // matching the quote delimiter that opens a string
    kBareKey,       // inside an unquoted key, up to ':'
    kColon,         // after a quoted key: ':'
    kScalar,        // inside true/false/null/null_word/number
    kQuoted,        // inside a delimiter-quoted string (raw bytes)
    kJsonString,    // inside a JSON string (quote is '"')
    kAfterValue,    // ',' or the container's close
    kDone,          // the top-level object closed
    kRejected,
  };
  State state = State::kValue;
  bool key = false;             // the open string or quote match is a key
  std::uint8_t matched = 0;     // quote delimiter bytes matched
  std::uint8_t number = 0;      // number recognizer state
  std::uint8_t words = 0;       // literal words still possible (bit per word)
  std::uint8_t escape = 0;      // JSON string escape state
  std::uint16_t word_at = 0;    // bytes into the literal words
  std::uint16_t hex = 0;        // \u escape value so far
  bool high_surrogate = false;  // a \u high surrogate needs its low half
  bool key_text = false;        // an unquoted key has non-space bytes
  std::uint32_t depth = 0;
  std::uint64_t objects = 0;  // bit d: level d is an object
  std::size_t at = 0;         // bytes stepped
};
struct ToolCursor {
  // Offsets are relative to `start`: the first unconsumed buffered byte in
  // text mode, the open call's body in call mode.
  std::size_t start = 0;
  std::size_t calls = 0, block_calls = 0;
  bool in_call = false, in_block = false;
  bool reasoning_active = false, reasoning_ended = false;
  bool started = false;  // any output beyond a reasoning opener was consumed
  bool content = false;  // non-space answer text was emitted
  ToolScan scan;
  unsigned tag_phase = 0;
  int tag_json = -1;
  std::size_t tag_name_end = std::string::npos, tag_header_search = 0;
  ToolEnvelopeCursor envelope;
  ToolObjectCursor object;
};
}  // namespace detail

class ToolInterpreter {
 public:
  // max_bytes bounds raw generated bytes (a consistency bound: the caller
  // funds actual growth with ToolOutputBytes). The other JSON limits bound
  // depth, values and decoded strings. No call-count or time limit.
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
  // Whether appending candidate bytes keeps the output in the grammar this
  // interpreter's Push enforces: the same code runs on a copied cursor over
  // the buffered bytes plus the candidate, without emitting or copying them.
  // A false answer means Push refuses them. The one refusal it leaves to
  // Push is a call that closes but cannot be converted (its body is fixed,
  // so masking the close would only run a constrained reply to its length):
  // Push's error then starts with kClosedToolCall.
  bool Allows(std::string_view candidate) const;
  static constexpr std::string_view kClosedToolCall = "the closed tool call is invalid: ";
  void Reasoning(bool active) { cursor_.reasoning_active = active; }
  bool reasoning() const { return cursor_.reasoning_active; }
  // Whether Finish(kStop) now would succeed.
  bool CanStop() const;
  std::optional<std::string_view> error() const {
    return error_ ? std::optional<std::string_view>(*error_) : std::nullopt;
  }
  ToolStatus Push(std::string_view generated, const Sink& sink);
  ToolStatus Finish(ToolFinish reason, const Sink& sink);
  std::size_t calls() const { return cursor_.calls; }
  // Buffered raw bytes: the open call's body or held text (ToolOutputBytes).
  std::size_t buffered() const { return pending_.size(); }

 private:
  ToolStatus Drain(bool final, const Sink& sink);
  ToolStatus Fail(std::string error);
  ToolCallFormat format_;
  ToolRequest request_;
  base::json::Limits limits_;
  std::string pending_;
  detail::ToolCursor cursor_;
  std::size_t received_ = 0;
  bool ended_ = false, busy_ = false, paused_ = false;
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
