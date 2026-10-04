// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Choosing how a model's chat template renders (chat.h): a native renderer
// by hash or by probe equivalence, else the template through the
// interpreter, with the control tokens its own text placed marked and the
// generation prompt's boundary found by rendering without it.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "chat/jinja.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::chat {
namespace {

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

// The variables transformers' apply_chat_template passes: OpenAI-shaped
// messages (tool call arguments as objects, as vLLM passes them), `tools`
// and `documents` (none when absent), add_generation_prompt, the special
// tokens' texts, and the template options the conversation sets
// (enable_thinking also as `thinking`, which DeepSeek's templates read).
std::vector<std::pair<std::string, jinja::Input>> Variables(const Conversation& c,
                                                            const TokenFacts& tokens,
                                                            bool add_generation_prompt) {
  using jinja::Input;
  std::vector<Input> messages;
  messages.reserve(c.messages.size());
  for (const Message& m : c.messages) {
    std::vector<std::pair<std::string, Input>> fields;
    fields.emplace_back("role", Input::String(RoleName(m.role)));
    fields.emplace_back("content", m.content ? Input::String(*m.content) : Input::None());
    if (m.reasoning_content) {
      fields.emplace_back("reasoning_content", Input::String(*m.reasoning_content));
    }
    if (!m.tool_calls.empty()) {
      std::vector<Input> calls;
      calls.reserve(m.tool_calls.size());
      for (const ToolCall& call : m.tool_calls) {
        calls.push_back(
            Input::Map({{"type", Input::String("function")},
                        {"function", Input::Map({{"name", Input::String(call.name)},
                                                 {"arguments", Input::Json(call.arguments)}})}}));
      }
      fields.emplace_back("tool_calls", Input::List(std::move(calls)));
    }
    messages.push_back(Input::Map(std::move(fields)));
  }
  std::vector<std::pair<std::string, Input>> v;
  v.emplace_back("messages", Input::List(std::move(messages)));
  if (c.tools.empty()) {
    v.emplace_back("tools", Input::None());
  } else {
    std::vector<Input> tools;
    tools.reserve(c.tools.size());
    for (const base::json::Value& t : c.tools) {
      tools.push_back(Input::Json(t));
    }
    v.emplace_back("tools", Input::List(std::move(tools)));
  }
  v.emplace_back("documents", Input::None());
  v.emplace_back("add_generation_prompt", Input::Bool(add_generation_prompt));
  if (tokens.bos) {
    v.emplace_back("bos_token", Input::String(*tokens.bos, true));
  }
  if (tokens.eos) {
    v.emplace_back("eos_token", Input::String(*tokens.eos, true));
  }
  if (c.enable_thinking) {
    v.emplace_back("enable_thinking", Input::Bool(*c.enable_thinking));
    v.emplace_back("thinking", Input::Bool(*c.enable_thinking));
  }
  if (c.reasoning_effort) {
    v.emplace_back("reasoning_effort", Input::String(*c.reasoning_effort));
  }
  if (c.preserve_thinking) {
    v.emplace_back("preserve_thinking", Input::Bool(*c.preserve_thinking));
  }
  return v;
}

Error FromJinja(const jinja::Error& e) {
  Rule rule = Rule::kUnsupported;
  if (e.code == jinja::Code::kRaised || e.code == jinja::Code::kRuntime) {
    rule = Rule::kInvalid;  // what Jinja2 itself raises
  }
  Error out{rule, e.reason, kNoItem, e.line};
  out.cancelled = e.code == jinja::Code::kCancelled;
  return out;
}

// The control tokens in the template's own text: leftmost, longest.
class SpecialScanner {
 public:
  explicit SpecialScanner(const std::vector<std::string>& control) {
    for (const std::string& t : control) {
      if (t.empty()) {
        continue;
      }
      texts_.insert(t);
      first_[static_cast<unsigned char>(t[0])] = true;
      if (std::ranges::find(lengths_, t.size()) == lengths_.end()) {
        lengths_.push_back(t.size());
      }
    }
    std::ranges::sort(lengths_, std::greater<>());
    for (const std::size_t n : lengths_) {
      per_start_ += 16 + n;
    }
  }

  // What Scan costs at most, as bytes: each position a control token may
  // start at hashes a candidate of every length (16 a lookup besides).
  std::uint64_t Cost(std::string_view text, std::size_t offset, std::size_t length) const {
    std::uint64_t starts = 0;
    for (std::size_t at = offset; at < offset + length; ++at) {
      starts += first_[static_cast<unsigned char>(text[at])] ? 1 : 0;
    }
    return length + (starts * per_start_);
  }

  enum class End : std::uint8_t { kDone, kCancelled, kBound };
  // kCancelled when `cancelled` (asked every jinja::kCancelWorkBytes of
  // Cost's measure, as the interpreter asks) said to stop; kBound before
  // `out` would hold more than `max_specials`.
  End Scan(std::string_view text, std::size_t offset, std::size_t length,
           std::vector<tokenizer::SpecialSpan>& out, std::size_t max_specials,
           const std::function<bool()>* cancelled, std::uint64_t& work) const {
    const std::size_t end = offset + length;
    std::uint64_t next_check = work + jinja::kCancelWorkBytes;
    for (std::size_t at = offset; at < end;) {
      if (cancelled != nullptr && work >= next_check) {
        next_check = work + jinja::kCancelWorkBytes;
        if (*cancelled && (*cancelled)()) {
          return End::kCancelled;
        }
      }
      ++work;
      std::size_t found = 0;
      if (first_[static_cast<unsigned char>(text[at])]) {
        work += per_start_;
        for (const std::size_t n : lengths_) {
          if (n <= end - at && texts_.contains(text.substr(at, n))) {
            found = n;
            break;
          }
        }
      }
      if (found != 0) {
        if (out.size() >= max_specials) {
          return End::kBound;
        }
        out.push_back({at, found});
        at += found;
      } else {
        ++at;
      }
    }
    return End::kDone;
  }

 private:
  std::unordered_set<std::string_view> texts_;
  std::array<bool, 256> first_{};
  std::vector<std::size_t> lengths_;
  std::uint64_t per_start_ = 0;
};

// A probe's bounds: a family's template renders a probe in thousands of
// steps and well under a megabyte of work; one that needs more is no
// family's. A template reaching either differs from the family at its
// first probe, so registration renders a hostile template a handful of
// times at these bounds, not hundreds of times at the full ones.
constexpr jinja::Budget kProbeBudget{.max_steps = 500'000, .max_work_bytes = 16U << 20U};

// The next probe's budget: kProbeBudget, within what is left of kProbePool
// after `spent`. Every probe of a registration draws on that one pool, so a
// template that passes probe after probe, each just under kProbeBudget,
// costs no more than the pool; none once it is spent.
std::optional<jinja::Budget> ProbeBudget(jinja::Usage& spent) {
  if (spent.steps >= kProbePool.steps || spent.work_bytes >= kProbePool.work_bytes) {
    return std::nullopt;
  }
  return jinja::Budget{
      .max_steps = std::min(kProbeBudget.max_steps, kProbePool.steps - spent.steps),
      .max_work_bytes =
          std::min(kProbeBudget.max_work_bytes, kProbePool.work_bytes - spent.work_bytes),
      .usage = &spent,
      .cancelled = nullptr,
      .max_live_bytes = 0,
      .max_output_bytes = 0};
}

// A fixed clock for probes: no family renderer reads one.
constexpr jinja::CivilTime kProbeTime{.year = 2026,
                                      .month = 1,
                                      .day = 1,
                                      .hour = 0,
                                      .minute = 0,
                                      .second = 0,
                                      .weekday = 4,
                                      .yearday = 0};

// ----------------------------------------------------------- probe corpus

constexpr std::string_view kWeatherTool =
    R"({"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "days": {"type": "integer"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}})";
constexpr std::string_view kSearchTool =
    R"({"type": "function", "function": {"name": "search", "description": "Search the web <fast> & \"exact\".", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}})";
constexpr std::string_view kArgsParis = R"({"city": "Paris", "days": 3, "unit": "celsius"})";
constexpr std::string_view kArgsTokyo =
    R"({"city": "東京", "days": 3, "extra": {"nested": [1, 2.5, null, true], "s": "a\nb"}})";
constexpr std::string_view kArgsOslo = R"({"city": "Oslo"})";
// Arguments a client sent as JSON that is not an object: a string holding
// an object, a plain string, a list.
constexpr std::string_view kArgsString = R"(" {\"city\": \"Oslo\"} ")";
constexpr std::string_view kArgsPlain = R"("Oslo, 3 days")";
constexpr std::string_view kArgsList = R"(["Oslo", 3])";
// A tool whose properties are named like schema keys.
constexpr std::string_view kSchemaKeysTool =
    R"({"type": "function", "function": {"name": "lookup", "description": "Look up.", "parameters": {"type": "object", "properties": {"type": {"type": "string"}, "description": {"type": "string", "description": "what"}, "city": {"type": "string"}}, "required": ["type"]}}})";

// The documents the probe conversations' JSON points into.
struct ProbeDocs {
  base::json::Document weather;
  base::json::Document search;
  base::json::Document paris;
  base::json::Document tokyo;
  base::json::Document oslo;
  base::json::Document string;
  base::json::Document plain;
  base::json::Document list;
  base::json::Document schema_keys;
};

std::optional<ProbeDocs> ParseProbeDocs() {
  auto weather = base::json::Parse(kWeatherTool);
  auto search = base::json::Parse(kSearchTool);
  auto paris = base::json::Parse(kArgsParis);
  auto tokyo = base::json::Parse(kArgsTokyo);
  auto oslo = base::json::Parse(kArgsOslo);
  auto string = base::json::Parse(kArgsString);
  auto plain = base::json::Parse(kArgsPlain);
  auto list = base::json::Parse(kArgsList);
  auto schema_keys = base::json::Parse(kSchemaKeysTool);
  if (!weather || !search || !paris || !tokyo || !oslo || !string || !plain || !list ||
      !schema_keys) {
    return std::nullopt;
  }
  return ProbeDocs{std::move(*weather), std::move(*search), std::move(*paris),
                   std::move(*tokyo),   std::move(*oslo),   std::move(*string),
                   std::move(*plain),   std::move(*list),   std::move(*schema_keys)};
}

Message M(Role role, std::optional<std::string> content,
          std::optional<std::string> reasoning = std::nullopt, std::vector<ToolCall> calls = {}) {
  return Message{role, std::move(content), std::move(reasoning), std::move(calls)};
}

// Conversations that exercise every part of the supported families'
// formats: roles and their orders, tools, tool calls and results,
// reasoning, empty and missing content, Unicode and whitespace, and tags in
// content. Each is probed with thinking unset, on and off.
std::vector<Conversation> ProbeConversations(const ProbeDocs& d) {
  using R = Role;
  std::vector<Conversation> out;
  const auto add = [&](std::vector<Message> messages, bool tools = false, bool prompt = true) {
    Conversation c;
    c.messages = std::move(messages);
    if (tools) {
      c.tools = {d.weather.root(), d.search.root()};
    }
    c.add_generation_prompt = prompt;
    out.push_back(std::move(c));
  };
  add({M(R::kUser, "What is 2+2?")});
  add({M(R::kSystem, "You are a helpful assistant."), M(R::kUser, "Hello!")});
  add({M(R::kSystem, "Be brief."), M(R::kUser, "Hi"),
       M(R::kAssistant, "Hello! How can I help?", "The user greets me."),
       M(R::kUser, "Capital of France?")});
  add({M(R::kUser, "Hi"), M(R::kAssistant, "Hello!", "Greeting.")}, false, false);
  add({M(R::kSystem, "You can call tools."), M(R::kUser, "Weather in Paris and Tokyo for 3 days?"),
       M(R::kAssistant, "", "I need two weather calls.",
         {{"get_weather", d.paris.root()}, {"get_weather", d.tokyo.root()}}),
       M(R::kTool, R"({"temp": 21, "sky": "sunny"})"), M(R::kTool, R"({"temp": 18.5})"),
       M(R::kAssistant, "Paris: 21°C. Tokyo: 18.5°C.", "Summarize both."),
       M(R::kUser, "Thanks! Any news on umbrellas?")},
      true);
  add({M(R::kUser, "Find news about tokenizers.")}, true);
  add({M(R::kSystem, ""), M(R::kUser, "Find news.")}, true);
  add({M(R::kSystem, "Tools allowed."), M(R::kUser, "Weather?"),
       M(R::kAssistant, "Let me check.", std::nullopt, {{"get_weather", d.oslo.root()}}),
       M(R::kTool, "cold")},
      true);
  add({M(R::kSystem, "Réponds en français. 用中文也可以。"),
       M(R::kUser, "Tiếng Việt 😀 👨‍👩‍👧 \t tabs and  spaces  ")});
  add({M(R::kSystem, "  \n system with spaces 　"), M(R::kUser, "\n\n  question?  \n\x1c")});
  add({M(R::kUser, "Explain <think> and </think> and <tool_call> tags.")});
  add({M(R::kUser, std::nullopt)});
  add({M(R::kSystem, "First."), M(R::kSystem, "Second."), M(R::kUser, "Go.")});
  add({M(R::kUser, "one"), M(R::kUser, "two")});
  add({M(R::kTool, "orphan"), M(R::kUser, "What was that?")});
  add({M(R::kUser, "Count."), M(R::kAssistant, "One.", "First."),
       M(R::kAssistant, "Two.", "Second."), M(R::kUser, "Go on.")});
  add({M(R::kUser, "Hi"), M(R::kSystem, "Late rule."), M(R::kAssistant, "Hello.", "Greet."),
       M(R::kUser, "Bye")});
  add({M(R::kUser, "Hi")}, false, false);
  add({M(R::kSystem, "Just a system.")});
  add({M(R::kUser, "Real question"), M(R::kUser, "<tool_response>\nresult\n</tool_response>")});
  add({M(R::kUser, "<tool_response>x</tool_response>")});
  add({M(R::kUser, "Hi"), M(R::kAssistant, "Hello!", " "), M(R::kUser, "Bye"),
       M(R::kAssistant, "Bye!", "farewell")},
      false, false);
  add({M(R::kAssistant, "Hello, I am ready.", "Opening."), M(R::kUser, "Hi")});
  return out;
}

// Conversations where the variants of a family's template part (Template's
// probe_variants; Gemma 4's), probed first, as they tell those variants
// apart: tool calls whose arguments are not an object, which a variant may
// render, ignore or refuse; a system message without content, which one
// prints as `None`; a property named like a schema key (`type`,
// `description`), which one always filters out; and whitespace content
// beside tool results, after which one closes the turn.
std::vector<Conversation> VariantConversations(const ProbeDocs& d) {
  using R = Role;
  std::vector<Conversation> out;
  for (const base::json::Document* args : {&d.string, &d.plain, &d.list}) {
    Conversation c;
    c.messages = {M(R::kUser, "Weather in Oslo?"),
                  M(R::kAssistant, "", std::nullopt, {{"get_weather", args->root()}}),
                  M(R::kTool, "cold"), M(R::kUser, "Thanks.")};
    c.tools = {d.weather.root()};
    out.push_back(std::move(c));
  }
  Conversation none;
  none.messages = {M(R::kSystem, std::nullopt), M(R::kUser, "Hi")};
  out.push_back(std::move(none));
  Conversation keys;
  keys.messages = {M(R::kUser, "Look it up.")};
  keys.tools = {d.schema_keys.root()};
  out.push_back(std::move(keys));
  Conversation blank;
  blank.messages = {M(R::kUser, "Weather in Oslo?"),
                    M(R::kAssistant, "  ", std::nullopt, {{"get_weather", d.oslo.root()}}),
                    M(R::kTool, "cold"), M(R::kUser, "Thanks.")};
  blank.tools = {d.weather.root()};
  out.push_back(std::move(blank));
  return out;
}

// One probe: the native renderer and the template agree, or both refuse,
// or the native renderer does not implement the case (no evidence).
enum class Verdict : std::uint8_t { kSame, kBothRefuse, kSkip, kDiffer };

Verdict Probe(const Template& native, const jinja::Template& program, const TokenFacts& tokens,
              const Conversation& c, jinja::Usage& spent) {
  const auto mine = native.render(c);
  if (!mine && mine.error().rule == Rule::kUnsupported) {
    return Verdict::kSkip;
  }
  const auto budget = ProbeBudget(spent);
  if (!budget) {
    return Verdict::kDiffer;  // the pool is spent: no family's template costs that much
  }
  auto theirs = program.Render(Variables(c, tokens, c.add_generation_prompt), kProbeTime, *budget);
  if (mine && theirs) {
    return mine->text == theirs->text ? Verdict::kSame : Verdict::kDiffer;
  }
  if (!mine && !theirs &&
      (theirs.error().code == jinja::Code::kRaised ||
       theirs.error().code == jinja::Code::kRuntime)) {
    return Verdict::kBothRefuse;
  }
  return Verdict::kDiffer;
}

// The end-of-turn token an interpreted template places after an assistant
// message's content, if any.
std::optional<std::string> EndOfTurn(const jinja::Template& program, const TokenFacts& tokens,
                                     jinja::Usage& spent) {
  const auto budget = ProbeBudget(spent);
  if (!budget) {
    return std::nullopt;
  }
  constexpr std::string_view kProbe = "PROBE-ASSISTANT-CONTENT";
  Conversation c;
  c.messages = {M(Role::kUser, "PROBE-USER"), M(Role::kAssistant, std::string(kProbe))};
  c.add_generation_prompt = false;
  auto r = RenderInterpreted(program, c, tokens, kProbeTime, *budget);
  if (!r) {
    return std::nullopt;
  }
  const std::size_t at = r->text.rfind(kProbe);
  if (at == std::string::npos) {
    return std::nullopt;
  }
  for (const tokenizer::SpecialSpan& s : r->specials) {
    if (s.offset >= at + kProbe.size()) {
      return r->text.substr(s.offset, s.length);
    }
  }
  return std::nullopt;
}

}  // namespace

TokenFacts TokenFacts::From(const tokenizer::Tokenizer& tokenizer) {
  TokenFacts facts;
  if (const auto bos = tokenizer.bos()) {
    facts.bos = std::string(tokenizer.Text(*bos));
  }
  if (const auto eos = tokenizer.eos()) {
    facts.eos = std::string(tokenizer.Text(*eos));
  }
  for (std::size_t id = 0; id < tokenizer.size(); ++id) {
    const auto t = static_cast<tokenizer::TokenId>(id);
    if (tokenizer.Kind(t) == tokenizer::TokenKind::kControl && !tokenizer.Text(t).empty()) {
      facts.control.emplace_back(tokenizer.Text(t));
    }
  }
  return facts;
}

std::expected<Rendered, Error> RenderInterpreted(const jinja::Template& program,
                                                 const Conversation& conversation,
                                                 const TokenFacts& tokens,
                                                 const std::optional<jinja::CivilTime>& now,
                                                 jinja::Budget budget) {
  // The scan for control tokens is charged to the rendering's work: to a
  // probe's budget, when it has one (an ordinary rendering's work is not
  // capped, D-102; the scan is linear in its output).
  const std::uint64_t max_work = budget.max_work_bytes != 0
                                     ? budget.max_work_bytes
                                     : std::numeric_limits<std::uint64_t>::max();
  jinja::Usage own;
  jinja::Usage& used = budget.usage != nullptr ? *budget.usage : own;
  const jinja::Usage before = used;
  budget.usage = &used;
  auto rendered = program.Render(
      Variables(conversation, tokens, conversation.add_generation_prompt), now, budget);
  if (!rendered) {
    return std::unexpected(FromJinja(rendered.error()));
  }
  const SpecialScanner scanner(tokens.control);
  for (const auto& [offset, length] : rendered->trusted) {
    used.work_bytes += scanner.Cost(rendered->text, offset, length);
  }
  if (used.work_bytes - before.work_bytes > max_work) {
    return std::unexpected(
        Error{.rule = Rule::kUnsupported,
              .reason = "the template built or scanned more bytes than its bound"});
  }
  // The specials are held with the text and its trusted ranges until the
  // ranges are released: all three within the output bound, as the
  // interpreter keeps the text and ranges (a template repeating a short
  // control token would otherwise hold several times its output in spans).
  std::size_t bound = program.limits().max_output_bytes;
  if (budget.max_output_bytes != 0) {
    bound = std::min(bound, budget.max_output_bytes);
  }
  const std::size_t held =
      rendered->text.size() + (rendered->trusted.size() * sizeof(jinja::TextRange));
  const std::size_t max_specials = (bound - std::min(bound, held)) / sizeof(tokenizer::SpecialSpan);
  Rendered out;
  std::uint64_t scanned = 0;
  for (const auto& [offset, length] : rendered->trusted) {
    const auto end = scanner.Scan(rendered->text, offset, length, out.specials, max_specials,
                                  budget.cancelled, scanned);
    if (end == SpecialScanner::End::kCancelled) {
      Error cancelled{.rule = Rule::kUnsupported,
                      .reason = "the rendering was cancelled: its request ended"};
      cancelled.cancelled = true;
      return std::unexpected(cancelled);
    }
    if (end == SpecialScanner::End::kBound) {
      return std::unexpected(FromJinja(jinja::Error{
          .code = jinja::Code::kLimit,
          .reason = "the rendering and its control tokens are longer than its bound"}));
    }
  }
  std::vector<jinja::TextRange>().swap(rendered->trusted);  // released before the next rendering
  out.text = std::move(rendered->text);
  if (conversation.add_generation_prompt) {
    // Where the generation prompt starts: the end of the rendering without
    // it, when that is a prefix of this one (a rendering of its own bounds).
    auto without = program.Render(Variables(conversation, tokens, false), now, budget);
    if (without && without->text.size() < out.text.size() &&
        std::string_view(out.text).starts_with(without->text)) {
      out.boundaries.push_back({BoundaryKind::kGenerationPrompt, without->text.size()});
    }
  }
  return out;
}

bool ProbeEquivalent(const Template& native, const jinja::Template& program,
                     const TokenFacts& tokens, jinja::Usage* spent) {
  jinja::Usage own;
  jinja::Usage& pool = spent != nullptr ? *spent : own;
  const auto docs = ParseProbeDocs();
  if (!docs) {
    return false;
  }
  // Each conversation with thinking unset, on and off; each accepted
  // reasoning effort with thinking unset and on; preserve_thinking off.
  std::size_t same = 0;
  const auto probe = [&](const Conversation& c) {
    const Verdict v = Probe(native, program, tokens, c, pool);
    same += v == Verdict::kSame ? 1 : 0;
    return v != Verdict::kDiffer;
  };
  constexpr std::array<std::optional<bool>, 3> kThinking = {std::nullopt, true, false};
  std::vector<Conversation> conversations;
  if (native.probe_variants) {
    conversations = VariantConversations(*docs);
  }
  for (Conversation& c : ProbeConversations(*docs)) {
    conversations.push_back(std::move(c));
  }
  for (Conversation& c : conversations) {
    for (const std::optional<bool> thinking : kThinking) {
      c.enable_thinking = thinking;
      if (!probe(c)) {
        return false;
      }
      if (thinking == false) {
        continue;
      }
      for (const std::string_view effort : native.probe_efforts) {
        c.reasoning_effort = std::string(effort);
        if (!probe(c)) {
          return false;
        }
      }
      c.reasoning_effort.reset();
      c.preserve_thinking = false;
      if (!probe(c)) {
        return false;
      }
      c.preserve_thinking.reset();
    }
  }
  return same >= 30;
}

std::expected<ChatTemplate, std::string> ChatTemplate::ForText(std::string_view template_text,
                                                               TokenFacts tokens,
                                                               const jinja::Limits& limits) {
  ChatTemplate t;
  t.sha256_ = base::ToHex(base::Sha256().Update(template_text).Finish());
  t.tokens_ = std::move(tokens);
  t.max_bytes_ = limits.max_output_bytes;
  t.max_live_bytes_ = limits.max_live_bytes;
  jinja::Usage probes;  // what every probe of this registration used (ProbeBudget)
  auto program = jinja::Template::Parse(template_text, limits);
  if (program) {
    // Kept beside a native renderer too, for the cases it does not implement.
    t.program_ = std::make_unique<jinja::Template>(std::move(*program));
  }
  if (const Template* native = FindTemplate(t.sha256_)) {
    t.how_ = How::kNativeByHash;
    t.native_ = native;
  } else {
    if (!t.program_) {
      return std::unexpected(std::format(
          "its chat template (SHA-256 {}) has no native renderer and the interpreter does not "
          "accept it: {}, so it has no chat turns (D-067)",
          t.sha256_, program.error().ToString()));
    }
    for (const Template& family : NativeTemplates()) {
      if (ProbeEquivalent(family, *t.program_, t.tokens_, &probes)) {
        t.how_ = How::kNativeByProbe;
        t.native_ = &family;
        break;
      }
    }
  }
  if (t.native_ != nullptr) {
    for (const std::string_view s : t.native_->stop.tokens) {
      t.stop_.emplace_back(s);
    }
    return t;
  }
  t.how_ = How::kInterpreted;
  if (auto eot = EndOfTurn(*t.program_, t.tokens_, probes)) {
    t.stop_.push_back(std::move(*eot));
  }
  return t;
}

ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;
ChatTemplate::~ChatTemplate() = default;

std::string_view ChatTemplate::name() const {
  return native_ != nullptr ? native_->name : std::string_view("interpreted");
}

std::expected<Rendered, Error> ChatTemplate::Render(const Conversation& conversation,
                                                    const std::optional<jinja::CivilTime>& now,
                                                    const std::function<bool()>* cancelled) const {
  if (native_ != nullptr) {
    auto rendered = native_->render(conversation);
    // A case the native renderer does not implement is the template's to
    // render (or refuse), unless it reached the output bound, which the
    // interpreter shares.
    if (rendered || rendered.error().rule != Rule::kUnsupported || rendered.error().bound ||
        !program_) {
      return rendered;
    }
  }
  return RenderInterpreted(*program_, conversation, tokens_, now,
                           jinja::Budget{.max_steps = 0,
                                         .max_work_bytes = 0,
                                         .usage = nullptr,
                                         .cancelled = cancelled,
                                         .max_live_bytes = conversation.max_live_bytes,
                                         .max_output_bytes = conversation.max_render_bytes});
}

}  // namespace jitllm::chat
