// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The tool-call grammar shared by the output parser and the sampler (B1, B5-B8,
// B11, B12 of the M3.5 foundation review), and an adversarial fuzz of it.

#include <gtest/gtest.h>
#include <pthread.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "chat/tool_calls.h"
#include "chat/tool_sampling.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace c = llmp::chat;
namespace j = llmp::base::json;
namespace tok = llmp::tokenizer;

j::Limits Limits(std::size_t bytes = 65536) {
  return {.max_bytes = bytes, .max_depth = 64, .max_values = bytes, .max_string_bytes = bytes};
}
std::vector<c::ToolDefinition> Definitions() {
  return {
      {.name = "lookup",
       .description = "find a thing",
       .parameters_json =
           R"({"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"},"obj":{"type":"object"},"xs":{"type":"array"},"maybe":{"anyOf":[{"type":"integer"},{"type":"null"}]}}})"},
      {.name = "other", .description = {}}};
}
c::ToolRequest Request(c::ToolChoice choice = {}, bool parallel = true) {
  return c::ToolRequest::Create(Definitions(), std::move(choice), parallel, Limits()).value();
}
struct Output {
  std::string text, reasoning;
  std::vector<c::ToolUsePart> calls;
  std::size_t begins = 0, ends = 0;
  bool thinking = false;
  c::ToolInterpreter::Sink Sink() {
    return [this](c::ToolEvent event) {
      if (auto* part = std::get_if<c::TextPart>(&event))
        (thinking ? reasoning : text) += part->text;
      else if (auto* call = std::get_if<c::ToolUsePart>(&event))
        calls.push_back(std::move(*call));
      else if (std::holds_alternative<c::ToolReasoningBegin>(event)) {
        ++begins;
        thinking = true;
      } else {
        ++ends;
        thinking = false;
      }
      return true;
    };
  }
};

// The probe's contract with the parser: Allows refuses only what Push
// refuses, and admits a refused chunk only when a closed call's conversion
// refused it (ToolInterpreter::kClosedToolCall): masking that close would
// only run a constrained reply on to its length.
testing::AssertionResult Agree(bool allows, const c::ToolInterpreter& pushed, bool accepted) {
  if (allows == accepted) return testing::AssertionSuccess();
  if (!allows) return testing::AssertionFailure() << "the probe refused what the parser accepts";
  if (pushed.error() && pushed.error()->starts_with(c::ToolInterpreter::kClosedToolCall))
    return testing::AssertionSuccess();
  return testing::AssertionFailure()
         << "the probe allowed what the parser refuses: " << pushed.error().value_or("(no error)");
}

// Runs `work` on a thread with a 256 KiB stack: recursion per nesting level
// would overflow it long before 10^6 levels.
void OnSmallStack(const std::function<void()>& work) {
  pthread_attr_t attr{};
  ASSERT_EQ(pthread_attr_init(&attr), 0);
  ASSERT_EQ(pthread_attr_setstacksize(&attr, std::size_t{256} << 10U), 0);
  pthread_t thread{};
  auto copy = work;
  const int created = pthread_create(
      &thread, &attr,
      [](void* p) -> void* {
        (*static_cast<std::function<void()>*>(p))();
        return nullptr;
      },
      &copy);
  pthread_attr_destroy(&attr);
  ASSERT_EQ(created, 0);
  ASSERT_EQ(pthread_join(thread, nullptr), 0);
}

TEST(ToolGrammar, DeepNestingIsBoundedAndNeverRecursesOnTheStack) {
  const std::string deep(1'000'000, '[');
  const auto limits = Limits(8 * deep.size());
  struct Case {
    c::ToolCallFormat format;
    std::string text;
    bool parses;  // a call, else a request-local refusal or answer text
  };
  const std::vector<Case> cases{
      // A schema-typed array too deep to parse stays the raw string it is.
      {c::QwenToolCalls(), "<function=lookup><parameter=xs>" + deep + "</parameter></function>",
       true},
      // Explicit JSON that is too deep is an invalid call (found when it closes).
      {c::DsmlToolCalls(),
       "<｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"xs\" string=\"false\">" + deep +
           "</｜DSML｜parameter></｜DSML｜invoke>",
       false},
      {c::GemmaToolCalls(), "<|tool_call>call:lookup{xs:" + deep + "<tool_call|>", false},
      // Auto bare JSON that cannot parse is answer text.
      {c::JsonToolCalls(),
       R"({"name":"lookup","parameters":{"xs":)" + deep + std::string(deep.size(), ']') + "}}",
       true},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.format.call_begin);
    OnSmallStack([&] {
      c::ToolInterpreter parser(test.format, Request(), limits);
      Output out;
      const bool pushed = parser.Push(test.text, out.Sink()).has_value();
      const bool finished = pushed && parser.Finish(c::ToolFinish::kStop, out.Sink()).has_value();
      EXPECT_EQ(finished, test.parses);
      EXPECT_LE(parser.buffered(), test.text.size());
      if (test.format.arguments == c::ToolArguments::kTaggedParameters && test.parses) {
        ASSERT_EQ(out.calls.size(), 1U);
        auto arguments = j::Parse(out.calls[0].input_json, limits);
        ASSERT_TRUE(arguments);
        EXPECT_EQ(arguments->root().find("xs")->string(), deep);
      }
      // The sampler's probe runs the same grammar on the same stack.
      c::ToolInterpreter probe(test.format, Request(), limits);
      EXPECT_TRUE(Agree(probe.Allows(test.text), parser, pushed));
    });
  }
}

TEST(ToolGrammar, ObjectNotationHasOneGrammarForSamplingAndParsing) {
  struct Case {
    std::string body;
    std::optional<std::string> json;  // nullopt: outside the grammar
    bool closed = false;              // refused only when the call closes
  };
  const std::vector<Case> cases{
      {R"(n:1,s:<|"|>a<|"|>})", R"({"n":1,"s":"a"})"},
      {R"(<|"|>quoted key<|"|>:None})", R"({"quoted key":null})"},
      {R"(  spaced key :true})", R"({"spaced key":true})"},
      {R"(s:<|"|><tool_call|>,}]<|"|>})", R"({"s":"<tool_call|>,}]"})"},
      {R"(s:<|"|>a<|<|"|>})", R"({"s":"a<|"})"},
      {R"(obj:{a:[1,{b:[]},-0.5e+3]}})", R"({"obj":{"a":[1,{"b":[]},-0.5e+3]}})"},
      // A JSON-quoted string is not this format's string (one grammar: refused
      // by the parser as by the sampler).
      {R"(s:"abc"})", std::nullopt},
      {R"("k":1})", R"({"\"k\"":1})"},      // an unquoted key keeps its bytes
      {R"(n:1,n:2})", std::nullopt, true},  // a repeated key
      {R"(n:01})", std::nullopt},
      {R"(n:tru})", std::nullopt},
      {R"(n:1,})", std::nullopt},
      {R"(:1})", std::nullopt},
      {R"(n:1 2})", std::nullopt},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.body);
    const std::string text = "<|tool_call>call:lookup{" + test.body + "<tool_call|>";
    for (std::size_t split = 0; split <= text.size(); ++split) {
      c::ToolInterpreter parser(c::GemmaToolCalls(), Request(), Limits());
      Output out;
      const auto first = std::string_view(text).substr(0, split);
      const auto rest = std::string_view(text).substr(split);
      EXPECT_EQ(parser.Allows(text), test.json.has_value() || test.closed);
      auto probe = parser;
      Output probed;
      const bool first_pushed = probe.Push(first, probed.Sink()).has_value();
      ASSERT_TRUE(Agree(parser.Allows(first), probe, first_pushed));
      if (!parser.Push(first, out.Sink())) {
        EXPECT_FALSE(test.json);
        continue;
      }
      EXPECT_EQ(parser.Allows(rest), test.json.has_value() || test.closed);
      const bool pushed = parser.Push(rest, out.Sink()).has_value();
      EXPECT_EQ(pushed, test.json.has_value());
      if (test.closed) {
        ASSERT_TRUE(parser.error());
        EXPECT_TRUE(parser.error()->starts_with(c::ToolInterpreter::kClosedToolCall));
      }
      if (!pushed) continue;
      ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
      ASSERT_EQ(out.calls.size(), 1U);
      EXPECT_EQ(out.calls[0].input_json, *test.json);
    }
  }
}

TEST(ToolGrammar, BareJsonRequiredNameIsCheckedAsAPrefix) {
  const auto required = Request({c::ToolChoiceKind::kRequired, {}});
  c::ToolInterpreter parser(c::JsonToolCalls(), required, Limits());
  EXPECT_TRUE(parser.Allows(R"({"name":"lo)"));
  EXPECT_TRUE(parser.Allows(R"({"name":"ot)"));
  EXPECT_TRUE(parser.Allows(R"({"name":"\u006co)"));  // escapes decode before the check
  EXPECT_TRUE(parser.Allows(R"({"name":"\u00)"));     // an incomplete escape waits
  EXPECT_FALSE(parser.Allows(R"({"name":"x)"));
  EXPECT_FALSE(parser.Allows(R"({"name":"lookupx)"));
  // A closed name is the whole name, whatever arrives with it (found by the
  // fuzz below: bytes after an undeclared name hid it from a whole chunk).
  EXPECT_TRUE(parser.Allows(R"({"name":"lookup" )"));
  EXPECT_FALSE(parser.Allows(R"({"name":"lo")"));
  EXPECT_FALSE(parser.Allows(R"({"name":"arguments"other)"));
  EXPECT_FALSE(parser.Allows(R"({"name":"lookup"x)"));
  EXPECT_FALSE(parser.Allows(R"({"name":"lo\"x)"));  // an escaped quote stays inside
  EXPECT_FALSE(parser.Allows(R"({"name":"lookup","parameters":{}} {"name":"zz)"));
  EXPECT_TRUE(parser.Allows(R"({"name":"lookup","parameters":{}})"));
  Output out;
  EXPECT_FALSE(parser.Push(R"({"name":"x)", out.Sink()));
}

TEST(ToolGrammar, ArgumentsMayBeAJsonStringAndSchemasMayBeAlternatives) {
  c::ToolInterpreter llama(c::JsonToolCalls(), Request(), Limits());
  Output out;
  ASSERT_TRUE(llama.Push(R"({"name":"lookup","parameters":"{\"n\":1}"})", out.Sink()));
  ASSERT_TRUE(llama.Finish(c::ToolFinish::kStop, out.Sink()));
  ASSERT_EQ(out.calls.size(), 1U);
  EXPECT_EQ(out.calls[0].input_json, R"({"n":1})");
  c::ToolInterpreter qwen(c::QwenToolCalls(), Request(), Limits());
  Output typed;
  ASSERT_TRUE(
      qwen.Push("<function=lookup><parameter=maybe>5</parameter></function>"
                "<function=lookup><parameter=maybe>null</parameter></function>"
                "<function=lookup><parameter=maybe>five</parameter></function>",
                typed.Sink()));
  ASSERT_TRUE(qwen.Finish(c::ToolFinish::kStop, typed.Sink()));
  ASSERT_EQ(typed.calls.size(), 3U);
  EXPECT_EQ(typed.calls[0].input_json, R"({"maybe":5})");
  EXPECT_EQ(typed.calls[1].input_json, R"({"maybe":null})");
  EXPECT_EQ(typed.calls[2].input_json, R"({"maybe":"five"})");
}

TEST(ToolGrammar, UnclosedParameterIsClosedByTheInvokeAliasAtTheEnd) {
  const std::string text =
      "<tool_call>\n<function=lookup>\n<parameter=s>\nmarker.txt\n</invoke>\n</tool_call>";
  for (std::size_t split = 0; split <= text.size(); ++split) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    Output out;
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(0, split), out.Sink()));
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(split), out.Sink()));
    EXPECT_TRUE(parser.CanStop());  // a natural end is allowed: the call completes
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_EQ(out.calls[0].input_json, R"({"s":"marker.txt"})");
  }
  // A value that closes later keeps the alias as its own text.
  c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
  Output out;
  ASSERT_TRUE(
      parser.Push("<function=lookup><parameter=s>a</invoke>b</parameter></function>", out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  ASSERT_EQ(out.calls.size(), 1U);
  EXPECT_EQ(out.calls[0].input_json, R"({"s":"a</invoke>b"})");
}

TEST(ToolGrammar, ModelOpenedReasoningIsReasoningOnTheToolsPath) {
  {
    c::ToolInterpreter gemma(c::GemmaToolCalls(), Request(), Limits());
    Output out;
    const std::string text =
        "<|channel>thought\nweigh it<channel|>answer<|tool_call>call:lookup{n:1}<tool_call|>";
    for (const char ch : text) ASSERT_TRUE(gemma.Push(std::string_view(&ch, 1), out.Sink()));
    ASSERT_TRUE(gemma.Finish(c::ToolFinish::kStop, out.Sink()));
    EXPECT_EQ(out.begins, 1U);
    EXPECT_EQ(out.ends, 1U);
    EXPECT_EQ(out.reasoning, "weigh it");
    EXPECT_EQ(out.text, "answer");
    ASSERT_EQ(out.calls.size(), 1U);
  }
  {
    c::ToolInterpreter qwen(c::QwenToolCalls(), Request(), Limits());
    Output out;
    ASSERT_TRUE(qwen.Push("<think>hmm</think>done", out.Sink()));
    ASSERT_TRUE(qwen.Finish(c::ToolFinish::kStop, out.Sink()));
    EXPECT_EQ(out.reasoning, "hmm");
    EXPECT_EQ(out.text, "done");
  }
  {
    // Only the output's start opens reasoning.
    c::ToolInterpreter qwen(c::QwenToolCalls(), Request(), Limits());
    Output out;
    ASSERT_TRUE(qwen.Push("x<think>y", out.Sink()));
    ASSERT_TRUE(qwen.Finish(c::ToolFinish::kStop, out.Sink()));
    EXPECT_EQ(out.begins, 0U);
    EXPECT_EQ(out.text, "x<think>y");
  }
}

TEST(ToolGrammar, OrderingAndDeclarationsAreLinearInTheirSizes) {
  constexpr std::size_t kCount = 50'000;
  std::vector<c::ToolUsePart> calls(kCount);
  std::vector<c::ToolResultPart> results(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    calls[i].id = "id" + std::to_string(i);
    results[kCount - 1 - i].tool_use_id = calls[i].id;
    results[kCount - 1 - i].text = std::to_string(i);
  }
  auto ordered = c::OrderToolResults(calls, results);
  ASSERT_TRUE(ordered);
  EXPECT_EQ(ordered->front().text, "0");
  EXPECT_EQ(ordered->back().text, std::to_string(kCount - 1));
  results.back().tool_use_id = results.front().tool_use_id;
  EXPECT_FALSE(c::OrderToolResults(calls, results));
  std::vector<c::ToolDefinition> definitions(kCount);
  for (std::size_t i = 0; i < kCount; ++i) definitions[i].name = "tool" + std::to_string(i);
  auto request = c::ToolRequest::Create(definitions, {}, true, Limits(std::size_t{64} << 20U));
  ASSERT_TRUE(request);
  EXPECT_TRUE(request->Declared("tool49999"));
  EXPECT_FALSE(request->Declared("tool50000"));
  definitions.push_back(definitions[kCount / 2]);
  EXPECT_FALSE(c::ToolRequest::Create(definitions, {}, true, Limits(std::size_t{64} << 20U)));
}

TEST(ToolGrammar, ClientFunctionObjectsReachTheTemplateAsSent) {
  std::vector<c::ToolDefinition> definitions{
      {.name = "f",
       .description = {},
       .parameters_json = "{}",
       .strict = true,
       .function_json = R"({"name":"f","strict":true,"x-extra":[1]})"}};
  auto request = c::ToolRequest::Create(definitions, {}, true, Limits());
  ASSERT_TRUE(request);
  const auto tools = request->TemplateTools();
  ASSERT_EQ(tools.size(), 1U);
  const auto function = tools[0].find("function");
  ASSERT_TRUE(function);
  EXPECT_FALSE(function->find("description"));  // never sent, never printed
  EXPECT_FALSE(function->find("parameters"));
  EXPECT_TRUE(function->find("x-extra"));
  EXPECT_FALSE(request->Parameter("f", "s"));
  definitions[0].function_json = R"({"name":"g"})";
  EXPECT_FALSE(c::ToolRequest::Create(definitions, {}, true, Limits()));
}

// ---- Sampler cursor -------------------------------------------------------
struct Vocabulary {
  tok::TokenizerSpec spec;
  Vocabulary() {
    spec.pre_tokenizer = tok::PreTokenizer::kRawBpe;
    constexpr std::string_view hex = "0123456789ABCDEF";
    for (unsigned b = 0; b < 256; ++b) {
      std::string value = "<0x00>";
      value[3] = hex[b >> 4U];
      value[4] = hex[b & 15U];
      Add(value, tok::TokenKind::kByte);
    }
    for (char ch = 'a'; ch <= 'z'; ++ch) Add(std::string(1, ch));
    Add("▁");
    for (const char* text : {":", "{", "}", "\n", "<", "|", ">"}) Add(text);
  }
  std::int32_t Add(std::string value, tok::TokenKind kind = tok::TokenKind::kNormal) {
    spec.tokens.push_back(std::move(value));
    spec.kinds.push_back(kind);
    return static_cast<std::int32_t>(spec.tokens.size() - 1);
  }
  std::int32_t Merge(std::string a, std::string b) {
    spec.merges.emplace_back(a, b);
    return Add(a + b);
  }
};

TEST(ToolGrammar, SamplerSeesTheParsersSanitizedBytesAndNeverRefusesInAuto) {
  Vocabulary vocabulary;
  auto tokenizer = tok::Tokenizer::Create(std::move(vocabulary.spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
  c::ToolSampling sampling(*tokenizer, c::QwenToolCalls(), Request(), Limits(), false, {});
  sampling.Budget(64, 64);
  // An ill-formed byte sequence (a lead byte, then ASCII): the parser sees
  // U+FFFD; so does the sampler's cursor.
  for (const std::int32_t token : std::array<std::int32_t, 4>{0xE4, 'a', 0xE4, 0xB8}) {
    EXPECT_TRUE(sampling.Plan(token));
    ASSERT_TRUE(sampling.Accept(token));
  }
  EXPECT_FALSE(sampling.refused());
  std::vector<float> row(tokenizer->size(), 0), scratch;
  EXPECT_EQ(sampling.Mask(row, scratch).data(), row.data());  // nothing masked
}

TEST(ToolGrammar, RequiredOpenerWaitsForReasoningTheModelOpens) {
  Vocabulary vocabulary;
  const auto channel = vocabulary.Add("<|channel>", tok::TokenKind::kControl);
  const auto channel_end = vocabulary.Add("<channel|>", tok::TokenKind::kControl);
  const auto opener = vocabulary.Add("<|tool_call>", tok::TokenKind::kControl);
  const auto eos = vocabulary.Add("<eos>", tok::TokenKind::kControl);
  auto tokenizer = tok::Tokenizer::Create(std::move(vocabulary.spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
  const std::array terminals{eos};
  const auto at = [](std::span<const float> masked, std::int32_t token) {
    return std::isfinite(masked[static_cast<std::size_t>(token)]);
  };
  std::vector<float> row(tokenizer->size(), 0), scratch;
  for (const bool may_reason : {false, true}) {
    c::ToolSampling sampling(*tokenizer, c::GemmaToolCalls(),
                             Request({c::ToolChoiceKind::kRequired, {}}), Limits(), false,
                             terminals, {}, may_reason);
    sampling.Budget(256, 256);
    EXPECT_TRUE(sampling.Preserves(channel));
    const auto first = sampling.Mask(row, scratch);
    EXPECT_TRUE(at(first, opener));
    EXPECT_EQ(at(first, channel), may_reason);
    EXPECT_FALSE(at(first, 'x'));
    EXPECT_FALSE(at(first, eos));
    if (!may_reason) continue;
    ASSERT_TRUE(sampling.Accept(channel));
    for (const char ch : std::string("thought\nhmm")) ASSERT_TRUE(sampling.Accept(ch));
    const auto thinking = sampling.Mask(row, scratch);
    EXPECT_TRUE(at(thinking, 'x'));
    EXPECT_TRUE(at(thinking, channel_end));
    ASSERT_TRUE(sampling.Accept(channel_end));
    const auto after = sampling.Mask(row, scratch);
    EXPECT_TRUE(at(after, opener));
    EXPECT_FALSE(at(after, 'x'));
  }
}

TEST(ToolGrammar, ForcedNamesStayTextEvenWithControlTokenSpellings) {
  Vocabulary vocabulary;
  const auto opener = vocabulary.Add("<|tool_call>", tok::TokenKind::kControl);
  const auto eos = vocabulary.Add("<eos>", tok::TokenKind::kControl);
  auto tokenizer = tok::Tokenizer::Create(std::move(vocabulary.spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
  const std::string name = "a<eos>b";
  const std::array<c::ToolDefinition, 1> definitions{{{.name = name, .description = {}}}};
  auto request =
      c::ToolRequest::Create(definitions, {c::ToolChoiceKind::kNamed, name}, true, Limits());
  ASSERT_TRUE(request);
  ASSERT_TRUE(request->ValidateNames(c::GemmaToolCalls()));
  EXPECT_TRUE(*tokenizer->ContainsControlText(name));  // admission refuses it (B11)
  c::ToolSampling sampling(*tokenizer, c::GemmaToolCalls(), *request, Limits(), false,
                           std::array{eos});
  sampling.Budget(256, 256);
  // Follow the forced span greedily: the first allowed candidate each step.
  std::vector<std::int32_t> published;
  std::vector<float> row(tokenizer->size(), 0), scratch;
  std::string decoded;
  const std::string forced = "<|tool_call>call:" + name + "{";
  for (std::size_t step = 0; step < 64 && decoded.size() < forced.size(); ++step) {
    std::vector<std::int32_t> next;
    if (!sampling.certain().empty()) {
      next.assign(sampling.certain().begin(), sampling.certain().end());
      sampling.Certain({});
    } else {
      const auto masked = sampling.Mask(row, scratch);
      const auto found = std::ranges::find_if(masked, [](float v) { return std::isfinite(v); });
      ASSERT_NE(found, masked.end());
      auto plan = sampling.Plan(static_cast<std::int32_t>(found - masked.begin()));
      ASSERT_TRUE(plan) << plan.error();
      next.push_back(plan->tokens.front());
      sampling.Certain({plan->tokens.begin() + 1, plan->tokens.end()});
    }
    for (const auto token : next) {
      ASSERT_TRUE(sampling.Accept(token));
      published.push_back(token);
      ASSERT_TRUE(tokenizer->Decode(std::span(&token, 1), {.control_tokens = true}, decoded));
    }
  }
  EXPECT_EQ(decoded.substr(0, forced.size()), forced);
  EXPECT_EQ(published.front(), opener);
  EXPECT_EQ(std::ranges::count(published, eos), 0);  // the name's <eos> is text
}

TEST(ToolGrammar, ForcedPlanCutByTheAllowanceEndsByLength) {
  Vocabulary vocabulary;
  vocabulary.Merge("a", "b");
  vocabulary.Merge("c", "d");
  const auto abc = vocabulary.Merge("ab", "c");
  auto tokenizer = tok::Tokenizer::Create(std::move(vocabulary.spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
  auto format = c::GemmaToolCalls();
  format.call_begin = "abcd\n";
  for (const std::size_t left : {std::size_t{1}, std::size_t{8}}) {
    c::ToolSampling sampling(*tokenizer, format, Request({c::ToolChoiceKind::kRequired, {}}),
                             Limits(), false, {});
    sampling.Budget(left, left);
    // "abcd" canonically encodes as ab + cd: abc needs two tokens to cover.
    auto plan = sampling.Plan(abc);
    ASSERT_TRUE(plan) << plan.error();
    if (left == 1) {
      EXPECT_EQ(plan->kind, c::ToolSampling::Selection::Kind::kOrdinarySelected);
    } else {
      EXPECT_EQ(plan->kind, c::ToolSampling::Selection::Kind::kStableSpan);
      EXPECT_EQ(plan->tokens.size(), 2U);
    }
    EXPECT_FALSE(sampling.refused());
  }
}

TEST(ToolGrammar, ConstrainedRefusalsAreRecordedOnTheRequestCursor) {
  Vocabulary vocabulary;
  auto tokenizer = tok::Tokenizer::Create(std::move(vocabulary.spec));
  ASSERT_TRUE(tokenizer);
  c::ToolSampling sampling(*tokenizer, c::QwenToolCalls(),
                           Request({c::ToolChoiceKind::kNamed, "lookup"}), Limits(), false, {});
  sampling.Budget(64, 64);
  auto copy = sampling;
  EXPECT_FALSE(copy.Plan('z'));
  EXPECT_TRUE(copy.refused());
  EXPECT_FALSE(sampling.refused());  // copies are independent until reported
  copy.Refuse();
  EXPECT_FALSE(sampling.Accept('z'));
  EXPECT_TRUE(sampling.refused());
}

// ---- Adversarial fuzz --------------------------------------------------
//
// Random and truncated model output built from each format's own fragments.
// At every step the probe (Allows) must agree with the parser (Push) on a
// copy, CanStop with Finish, and nothing may crash or grow beyond the input.
// The accepted output then streams again, whole and byte by byte: how the
// bytes arrive must never change what the parser emits or decides.

// Bytes as printable ASCII (failure messages must stay valid UTF-8).
std::string Escaped(std::string_view bytes) {
  std::string out;
  for (const char ch : bytes) {
    const auto byte = static_cast<unsigned char>(ch);
    if (byte >= 0x20 && byte < 0x7F && ch != '\\') {
      out += ch;
    } else {
      constexpr std::string_view kHex = "0123456789abcdef";
      out += "\\x";
      out += kHex[byte >> 4U];
      out += kHex[byte & 15U];
    }
  }
  return out;
}

// Events as a log: adjacent text merges, so arrival chunking does not show.
struct EventLog {
  std::vector<std::string> events;
  c::ToolInterpreter::Sink Sink() {
    return [this](c::ToolEvent event) {
      if (const auto* part = std::get_if<c::TextPart>(&event)) {
        if (!events.empty() && events.back().starts_with("T:"))
          events.back() += part->text;
        else
          events.push_back("T:" + part->text);
      } else if (const auto* call = std::get_if<c::ToolUsePart>(&event)) {
        events.push_back("C:" + call->name + ":" + call->input_json);
      } else {
        events.push_back(std::holds_alternative<c::ToolReasoningBegin>(event) ? "B" : "E");
      }
      return true;
    };
  }
  void Append(const EventLog& more) {
    for (const auto& event : more.events) {
      if (event.starts_with("T:") && !events.empty() && events.back().starts_with("T:"))
        events.back() += event.substr(2);
      else
        events.push_back(event);
    }
  }
};

TEST(ToolGrammar, FuzzedOutputKeepsParserAndProbeInAgreement) {
  struct Grammar {
    const c::ToolCallFormat* format;
    std::vector<std::string> pieces;
  };
  const std::vector<std::string> common{"lookup",   "other",
                                        "s",        "n",
                                        "xs",       "obj",
                                        " ",        "\n",
                                        "\"",       "\\",
                                        "{",        "}",
                                        "[",        "]",
                                        ",",        ":",
                                        "1",        "-",
                                        ".5",       "e9",
                                        "true",     "None",
                                        "null",     "é",
                                        "世",       "\xEF\xBF\xBD",
                                        "<",        ">",
                                        "</",       "x",
                                        "</think>", "<think>",
                                        "\\u00e9",  "\\ud83d",
                                        "\\ude00",  "\\\"",
                                        "\\n",      "lo"};
  const std::vector<Grammar> grammars{
      {&c::QwenToolCalls(),
       {"<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>",
        "<parameter name=\"", "\">", "</invoke>", "<tool_", "</func"}},
      {&c::DsmlToolCalls(),
       {"<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>", "<｜DSML｜invoke name=\"",
        "</｜DSML｜invoke>", "<｜DSML｜parameter name=\"", "</｜DSML｜parameter>",
        " string=\"true\">", " string=\"false\">", "\">", "<｜DSML｜"}},
      {&c::GemmaToolCalls(),
       {"<|tool_call>call:", "<tool_call|>", "<|\"|>", "<|channel>thought\n", "<channel|>",
        "<|tool_", "<|\""}},
      {&c::JsonToolCalls(), {"{\"name\":", "\"parameters\":", "\"lookup\"", "{}", "\"arguments\""}},
  };
  const std::vector<c::ToolChoice> choices{
      {}, {c::ToolChoiceKind::kRequired, {}}, {c::ToolChoiceKind::kNamed, "lookup"}};
  std::size_t checked = 0;
  for (const std::uint32_t seed : {20261026U, 1U, 2U, 3U}) {
    std::mt19937 random(seed);
    for (const auto& grammar : grammars) {
      for (const auto& choice : choices) {
        for (const bool parallel : {true, false}) {
          const auto request = Request(choice, parallel);
          for (int sample = 0; sample < 400; ++sample) {
            c::ToolInterpreter parser(*grammar.format, request, Limits());
            parser.Reasoning(sample % 5 == 0);
            std::size_t pushed = 0;
            std::string all;  // every accepted chunk
            EventLog log;
            const auto pieces = std::uniform_int_distribution<int>(1, 40)(random);
            for (int i = 0; i < pieces; ++i) {
              std::string chunk;
              const auto parts = std::uniform_int_distribution<int>(1, 3)(random);
              for (int p = 0; p < parts; ++p) {
                const bool own = std::uniform_int_distribution<int>(0, 1)(random) == 0;
                const auto& source = own ? grammar.pieces : common;
                auto piece = source[std::uniform_int_distribution<std::size_t>(
                    0, source.size() - 1)(random)];
                // Truncate a piece sometimes: partial markers mid-stream.
                if (piece.size() > 1 && std::uniform_int_distribution<int>(0, 3)(random) == 0)
                  piece.resize(
                      std::uniform_int_distribution<std::size_t>(1, piece.size() - 1)(random));
                chunk += piece;
              }
              const bool allows = parser.Allows(chunk);
              auto attempt = parser;
              EventLog emitted;
              const bool accepted = attempt.Push(chunk, emitted.Sink()).has_value();
              ASSERT_TRUE(Agree(allows, attempt, accepted))
                  << "seed " << seed << " format " << grammar.format->call_begin << " chunk "
                  << Escaped(chunk);
              ++checked;
              if (!accepted) break;
              parser = std::move(attempt);
              log.Append(emitted);
              all += chunk;
              pushed += chunk.size();
              ASSERT_LE(parser.buffered(), pushed);
              auto finish = parser;
              Output finished;
              ASSERT_EQ(parser.CanStop(),
                        finish.Finish(c::ToolFinish::kStop, finished.Sink()).has_value())
                  << "format " << grammar.format->call_begin;
            }
            EventLog tail;
            const bool ended = parser.Finish(c::ToolFinish::kLength, tail.Sink()).has_value();
            log.Append(tail);
            // The same accepted bytes, whole and then one byte at a time.
            for (const bool bytewise : {false, true}) {
              c::ToolInterpreter again(*grammar.format, request, Limits());
              again.Reasoning(sample % 5 == 0);
              EventLog replay;
              bool ok = true;
              if (bytewise) {
                for (std::size_t b = 0; ok && b < all.size(); ++b)
                  ok = again.Push(std::string_view(all).substr(b, 1), replay.Sink()).has_value();
              } else {
                ok = again.Push(all, replay.Sink()).has_value();
              }
              ASSERT_TRUE(ok) << "format " << grammar.format->call_begin << " bytewise " << bytewise
                              << " output " << Escaped(all) << ": "
                              << Escaped(again.error().value_or(""));
              EXPECT_EQ(again.Finish(c::ToolFinish::kLength, replay.Sink()).has_value(), ended)
                  << "format " << grammar.format->call_begin << " output " << Escaped(all);
              EXPECT_EQ(replay.events, log.events)
                  << "format " << grammar.format->call_begin << " bytewise " << bytewise
                  << " output " << Escaped(all);
            }
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 120000U);
}
}  // namespace
