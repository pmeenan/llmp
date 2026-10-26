// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "chat/tool_calls.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "chat/tool_sampling.h"

namespace {
namespace c = llmp::chat;
namespace j = llmp::base::json;

j::Limits Limits() {
  return {.max_bytes = 65536, .max_depth = 32, .max_values = 65536, .max_string_bytes = 65536};
}
std::vector<c::ToolDefinition> Definitions() {
  return {
      {.name = "lookup",
       .description = "find a thing",
       .parameters_json =
           R"({"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"},"b":{"type":"boolean"},"obj":{"type":"object"},"xs":{"type":"array"}}})"},
      {.name = "other", .description = {}}};
}
c::ToolRequest Request(c::ToolChoice choice = {}, bool parallel = true) {
  return c::ToolRequest::Create(Definitions(), std::move(choice), parallel, Limits()).value();
}
struct Output {
  std::string text;
  std::vector<c::ToolUsePart> calls;
  std::size_t reasoning_ends = 0;
  c::ToolInterpreter::Sink Sink() {
    return [this](c::ToolEvent event) {
      if (auto* text_part = std::get_if<c::TextPart>(&event))
        text += text_part->text;
      else if (auto* call = std::get_if<c::ToolUsePart>(&event))
        calls.push_back(std::move(*call));
      else
        ++reasoning_ends;
      return true;
    };
  }
};
std::string Qwen(std::string_view args, std::string_view name = "lookup") {
  return "<tool_call>\n<function=" + std::string(name) + ">\n" + std::string(args) +
         "</function>\n</tool_call>";
}

TEST(ToolCalls, NormalizedDefinitionsOwnTheirJsonAndNamedChoiceFiltersOnlyDefinitions) {
  auto definitions = Definitions();
  auto request =
      c::ToolRequest::Create(definitions, {c::ToolChoiceKind::kNamed, "lookup"}, true, Limits());
  ASSERT_TRUE(request);
  definitions.clear();
  auto moved = std::move(*request);
  const auto tools = moved.TemplateTools();
  ASSERT_EQ(tools.size(), 1U);
  EXPECT_EQ(tools[0].find("function")->find("name")->string(), "lookup");
  EXPECT_TRUE(moved.Declared("other"));
  EXPECT_EQ(moved.Parameter("lookup", "n")->find("type")->string(), "integer");
  const auto trigger = c::ToolTriggerFor(c::QwenToolCalls(), moved);
  EXPECT_TRUE(trigger.required);
  EXPECT_EQ(trigger.name, "lookup");
  EXPECT_EQ(trigger.begin, "<tool_call>");
  EXPECT_EQ(trigger.bound, "</think>");
  EXPECT_TRUE(trigger.ends_reasoning);
}
TEST(ToolCalls, ContractsRefuseStrictDuplicatesAndInvalidRequiredChoices) {
  auto definitions = Definitions();
  definitions[0].strict = true;
  EXPECT_FALSE(c::ToolRequest::Create(definitions, {}, true, Limits()));
  definitions[0].strict = false;
  definitions.push_back(definitions[0]);
  EXPECT_FALSE(c::ToolRequest::Create(definitions, {}, true, Limits()));
  EXPECT_FALSE(c::ToolRequest::Create({}, {c::ToolChoiceKind::kRequired, {}}, true, Limits()));
  EXPECT_FALSE(c::ToolRequest::Create(Definitions(), {c::ToolChoiceKind::kNamed, "missing"}, true,
                                      Limits()));
  EXPECT_TRUE(Request({c::ToolChoiceKind::kNone, {}}).TemplateTools().empty());
}
TEST(ToolCalls, ActiveNamesMustRoundTripThroughTheSelectedFormat) {
  const std::array formats{c::QwenToolCalls(), c::DsmlToolCalls(), c::GemmaToolCalls()};
  for (const auto& format : formats) {
    for (const auto& name : {std::string("a") + format.name_end + "b", std::string(" padded"),
                             std::string("padded\t")}) {
      const std::array<c::ToolDefinition, 2> definitions{
          {{.name = name, .description = {}}, {.name = "valid", .description = {}}}};
      for (const auto kind :
           {c::ToolChoiceKind::kAuto, c::ToolChoiceKind::kRequired, c::ToolChoiceKind::kNamed}) {
        auto request = c::ToolRequest::Create(
            definitions, {kind, kind == c::ToolChoiceKind::kNamed ? name : ""}, true, Limits());
        ASSERT_TRUE(request);
        EXPECT_FALSE(request->ValidateNames(format)) << name;
        EXPECT_TRUE(request->ValidateNames(c::JsonToolCalls())) << name;
      }
      auto none =
          c::ToolRequest::Create(definitions, {c::ToolChoiceKind::kNone, {}}, true, Limits());
      ASSERT_TRUE(none);
      EXPECT_TRUE(none->ValidateNames(format));
      auto named =
          c::ToolRequest::Create(definitions, {c::ToolChoiceKind::kNamed, "valid"}, true, Limits());
      ASSERT_TRUE(named);
      EXPECT_TRUE(named->ValidateNames(format));  // only the exposed definition is active
    }
  }
  // A multibyte terminator can start inside the name and finish at its boundary.
  auto overlap = c::DsmlToolCalls();
  overlap.name_end = "aba";
  auto request = c::ToolRequest::Create(
      std::array<c::ToolDefinition, 1>{{{.name = "ab", .description = {}}}}, {}, true, Limits());
  ASSERT_TRUE(request);
  EXPECT_FALSE(request->ValidateNames(overlap));
}

TEST(ToolCalls, ObjectNamesObeyTheSameLexicalBoundaryAndJsonNamesRemainEscaped) {
  for (const auto& name : {std::string("a<|\"|>b"), std::string("a<tool_call|>b")}) {
    auto request = c::ToolRequest::Create(
        std::array<c::ToolDefinition, 1>{{{.name = name, .description = {}}}}, {}, true, Limits());
    ASSERT_TRUE(request);
    EXPECT_FALSE(request->ValidateNames(c::GemmaToolCalls()));
  }
  for (const auto& name : {std::string("quoted<|\"|>text<|\"|>"), std::string("工具")}) {
    auto request = c::ToolRequest::Create(
        std::array<c::ToolDefinition, 1>{{{.name = name, .description = {}}}}, {}, true, Limits());
    ASSERT_TRUE(request);
    ASSERT_TRUE(request->ValidateNames(c::GemmaToolCalls()));
    c::ToolInterpreter parser(c::GemmaToolCalls(), *request, Limits());
    Output out;
    ASSERT_TRUE(parser.Push("<|tool_call>call:" + name + "{}<tool_call|>", out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_EQ(out.calls[0].name, name);
  }
  const std::string name = " a>b\"\\\t ";
  auto request =
      c::ToolRequest::Create(std::array<c::ToolDefinition, 1>{{{.name = name, .description = {}}}},
                             {c::ToolChoiceKind::kNamed, name}, true, Limits());
  ASSERT_TRUE(request);
  ASSERT_TRUE(request->ValidateNames(c::JsonToolCalls()));
  std::string text = "{\"name\":";
  j::AppendQuoted(name, text);
  text += ",\"parameters\":{}}";
  for (std::size_t split = 0; split <= text.size(); ++split) {
    c::ToolInterpreter parser(c::JsonToolCalls(), *request, Limits());
    Output out;
    ASSERT_TRUE(parser.Push(text.substr(0, split), out.Sink()));
    ASSERT_TRUE(parser.Push(text.substr(split), out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_EQ(out.calls[0].name, name);
  }
}

TEST(ToolCalls, QwenAcceptsEveryFragmentBoundaryAndEmitsOnlyCompleteCalls) {
  const auto text = std::string("answer ") +
                    Qwen("<parameter=s>\nhi\n</parameter>\n<parameter=n>\n12\n</parameter>\n") +
                    " after";
  for (std::size_t split = 0; split <= text.size(); ++split) {
    SCOPED_TRACE(split);
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    Output out;
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(0, split), out.Sink()));
    if (split < text.find("</function>") + std::string_view("</function>").size())
      EXPECT_TRUE(out.calls.empty());
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(split), out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_EQ(out.calls[0].name, "lookup");
    EXPECT_EQ(out.calls[0].input_json, R"({"s":"hi","n":12})");
    EXPECT_EQ(out.text, "answer  after");
    EXPECT_EQ(out.reasoning_ends, 1U);
  }
}
TEST(ToolCalls, OmittedOuterTagsAndUnclosedParametersUseNextBoundary) {
  const std::string text =
      "<function=lookup><parameter=s>\n\n  value  \n\n<parameter=n>\n42\n</function>";
  c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
  Output out;
  for (const char ch : text) ASSERT_TRUE(parser.Push(std::string_view(&ch, 1), out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  ASSERT_EQ(out.calls.size(), 1U);
  EXPECT_EQ(out.calls[0].input_json, R"({"s":"\n  value  \n","n":42})");
}
TEST(ToolCalls, FunctionClosureCompletesCallEvenWithoutOptionalOuterClosure) {
  c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
  Output out;
  ASSERT_TRUE(parser.Push("<tool_call><function=lookup></function>", out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  ASSERT_EQ(out.calls.size(), 1U);
}
TEST(ToolCalls, QwenObservedAttributeHeaderAndInvokeCloseAreDeclaredAlternatives) {
  // Text recovered from the failed native 7175 tool qualification's persisted
  // token history. Token IDs and the complete request remain external evidence.
  const std::string text =
      "The user wants me to read the marker.txt file using the read_marker tool. I should call "
      "the tool with the path \"marker.txt\".\n</think><tool_call><function=read_marker>\n"
      "<parameter name=\"path\">marker.txt</parameter>\n</invoke>";
  const std::array<c::ToolDefinition, 1> definitions{
      {{.name = "read_marker",
        .description = {},
        .parameters_json = R"({"type":"object","properties":{"path":{"type":"string"}}})"}}};
  auto request = c::ToolRequest::Create(definitions, {c::ToolChoiceKind::kNamed, "read_marker"},
                                        true, Limits());
  ASSERT_TRUE(request);
  for (std::size_t split = 0; split <= text.size(); ++split) {
    SCOPED_TRACE(split);
    c::ToolInterpreter parser(c::QwenToolCalls(), *request, Limits());
    parser.Reasoning(true);
    Output out;
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(0, split), out.Sink()));
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(split), out.Sink()));
    EXPECT_TRUE(parser.CanStop());
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_EQ(out.calls[0].name, "read_marker");
    EXPECT_EQ(out.calls[0].input_json, R"({"path":"marker.txt"})");
    EXPECT_EQ(out.reasoning_ends, 1U);
    EXPECT_FALSE(out.text.ends_with('>'));  // the shorter alias is erased exactly
  }
}
TEST(ToolCalls, TaggedAlternativesMixOnlyAtStructuralBoundariesAndKeepValuesOpaque) {
  const std::string text =
      R"(<tool_call><function=lookup><parameter=s>literal </invoke> and <parameter name="n"> text</parameter><parameter name="n">12</parameter><parameter=obj>{"x":"</invoke><parameter name=\"s\">"}</parameter></invoke></tool_call>after)";
  for (std::size_t split = 0; split <= text.size(); ++split) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    Output out;
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(0, split), out.Sink()));
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(split), out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    auto parsed = j::Parse(out.calls[0].input_json);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed->root().find("s")->string(),
              R"(literal </invoke> and <parameter name="n"> text)");
    EXPECT_EQ(parsed->root().find("n")->number(), "12");
    EXPECT_EQ(parsed->root().find("obj")->find("x")->string(), R"(</invoke><parameter name="s">)");
    EXPECT_EQ(out.text, "after");
  }
  c::ToolInterpreter bytes(c::QwenToolCalls(), Request(), Limits());
  Output out;
  for (const char ch : text) ASSERT_TRUE(bytes.Push(std::string_view(&ch, 1), out.Sink()));
  ASSERT_TRUE(bytes.Finish(c::ToolFinish::kStop, out.Sink()));
  ASSERT_EQ(out.calls.size(), 1U);
  for (const std::string malformed : {"<function=lookup><parameter name=\"s\">unclosed</invoke>",
                                      "<function=lookup><unsupported=s>x</unsupported></function>",
                                      "<function=lookup><parameter name=\"s\">x</parameter>"
                                      "<parameter=s>y</parameter></invoke>"}) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    Output bad;
    const auto pushed = parser.Push(malformed, bad.Sink());
    EXPECT_TRUE(!pushed || !parser.Finish(c::ToolFinish::kStop, bad.Sink()));
    EXPECT_TRUE(bad.calls.empty());
  }
}
TEST(ToolCalls, TaggedAlternativesRejectAmbiguousDeclarationsAndArePaidFor) {
  const auto base = *c::ToolOutputBytes(c::QwenToolCalls(), Limits().max_bytes);
  auto format = c::QwenToolCalls();
  format.call_end_aliases.push_back("</much_longer_call_end>");
  format.parameter_aliases.push_back({"<arg name=\"", "\">"});
  format.continued_route_stops.push_back("<route_marker>");
  EXPECT_GT(*c::ToolOutputBytes(format, Limits().max_bytes), base);
  for (const auto alias : {"", "</function>", "</function>suffix", "<parameter="}) {
    auto invalid = format;
    invalid.call_end_aliases.emplace_back(alias);
    c::ToolInterpreter parser(invalid, Request(), Limits());
    EXPECT_TRUE(parser.error());
    EXPECT_FALSE(parser.CanStop());
  }
}
TEST(ToolCalls, MalformedBodiesClosedByAliasesRemainErrorsAtEverySplit) {
  for (const std::string text : {"<function=lookup>junk</invoke>",
                                 "<function=lookup><unsupported=s>x</unsupported></invoke>"}) {
    for (std::size_t split = 0; split <= text.size(); ++split) {
      c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
      Output out;
      const auto first = parser.Push(std::string_view(text).substr(0, split), out.Sink());
      const auto last = parser.Push(std::string_view(text).substr(split), out.Sink());
      EXPECT_FALSE(first && last);
      EXPECT_FALSE(parser.Finish(c::ToolFinish::kLength, out.Sink()));
      EXPECT_TRUE(out.calls.empty());
    }
  }
  auto suffix = c::DsmlToolCalls();
  suffix.call_end_aliases = {"</invoke>"};
  const std::string invalid =
      "<｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"s\" "
      "invalid=\"true\">x</invoke>";
  for (std::size_t split = 0; split <= invalid.size(); ++split) {
    c::ToolInterpreter parser(suffix, Request(), Limits());
    Output out;
    const auto first = parser.Push(std::string_view(invalid).substr(0, split), out.Sink());
    const auto last = parser.Push(std::string_view(invalid).substr(split), out.Sink());
    EXPECT_FALSE(first && last);
    EXPECT_FALSE(parser.Finish(c::ToolFinish::kLength, out.Sink()));
    EXPECT_TRUE(out.calls.empty());
  }
}
TEST(ToolCalls, NonstrictSchemaConversionDoesNotRejectUnconvertibleScalars) {
  c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
  Output out;
  ASSERT_TRUE(parser.Push(
      Qwen("<parameter=n>not a "
           "number</parameter><parameter=b>true</parameter><parameter=s>12</parameter>"),
      out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  ASSERT_EQ(out.calls.size(), 1U);
  EXPECT_EQ(out.calls[0].input_json, R"({"n":"not a number","b":true,"s":"12"})");
}
TEST(ToolCalls, TaggedJsonValuesKeepQuotedNestedDelimitersOpaque) {
  const auto text = Qwen(
      R"(<parameter=obj>{"text":"</function><parameter=fake>","xs":[{"v":"</parameter>"}]}</parameter><parameter=xs>["</function>",{"x":"</parameter>"}]</parameter>)");
  for (std::size_t split = 0; split <= text.size(); ++split) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    Output out;
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(0, split), out.Sink()));
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(split), out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    auto json = j::Parse(out.calls[0].input_json);
    ASSERT_TRUE(json);
    EXPECT_EQ(json->root().find("obj")->find("text")->string(), "</function><parameter=fake>");
    EXPECT_EQ(json->root().find("xs")->at(0).string(), "</function>");
    EXPECT_EQ(json->root().find("xs")->at(1).find("x")->string(), "</parameter>");
  }
}
TEST(ToolCalls, PlainStringQuotesDoNotHideParameterBoundaries) {
  c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
  Output out;
  ASSERT_TRUE(parser.Push(Qwen("<parameter=s>an unmatched \" quote</parameter>"), out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  ASSERT_EQ(out.calls.size(), 1U);
  EXPECT_EQ(out.calls[0].input_json, R"({"s":"an unmatched \" quote"})");
}
TEST(ToolCalls, DsmlDeclaresExplicitStringAndJsonEncodings) {
  const std::string text =
      "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n"
      "<｜DSML｜parameter name=\"n\" string=\"true\">12</｜DSML｜parameter>\n"
      "<｜DSML｜parameter name=\"obj\" "
      "string=\"false\">{\"x\":\"</｜DSML｜invoke>\"}</｜DSML｜parameter>\n"
      "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
  c::ToolInterpreter parser(c::DsmlToolCalls(), Request(), Limits());
  Output out;
  for (const char ch : text) ASSERT_TRUE(parser.Push(std::string_view(&ch, 1), out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  ASSERT_EQ(out.calls.size(), 1U);
  EXPECT_EQ(out.calls[0].input_json, R"({"n":"12","obj":{"x":"</｜DSML｜invoke>"}})");
}
TEST(ToolCalls, GenericKeyValueGrammarPreservesUnwrappedWhitespace) {
  c::ToolCallFormat format{.call_begin = "<call>",
                           .call_end = "</call>",
                           .name_end = "\n",
                           .arguments = c::ToolArguments::kTaggedParameters,
                           .parameter_begin = "<key>",
                           .parameter_name_end = "</key><value>",
                           .parameter_end = "</value>"};
  c::ToolInterpreter parser(format, Request(), Limits());
  Output out;
  ASSERT_TRUE(parser.Push("<call>lookup\n<key>s</key><value>\n  x\n</value></call>", out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  EXPECT_EQ(out.calls[0].input_json, R"({"s":"\n  x\n"})");
}
TEST(ToolCalls, GemmaObjectNotationSupportsNestedValuesAndDelimiterText) {
  const std::string text =
      R"(<|tool_call>call:lookup{obj:{x:[1,true,None,<|"|><tool_call|> text<|"|>]},s:<|"|>héllo<|"|>}<tool_call|>)";
  for (std::size_t split = 0; split <= text.size(); ++split) {
    c::ToolInterpreter parser(c::GemmaToolCalls(), Request(), Limits());
    Output out;
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(0, split), out.Sink()));
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(split), out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_EQ(out.calls[0].input_json,
              R"({"obj":{"x":[1,true,null,"<tool_call|> text"]},"s":"héllo"})");
  }
}
TEST(ToolCalls, JsonEnvelopeKeepsEscapesAndQuotedCallEndOpaque) {
  c::ToolCallFormat format{.call_begin = "<call>", .call_end = "</call>"};
  const std::string text =
      R"(<call>{"name":"lookup","arguments":{"s":"\"</call>\"","n":1e999}}</call>)";
  c::ToolInterpreter parser(format, Request(), Limits());
  Output out;
  for (const char ch : text) ASSERT_TRUE(parser.Push(std::string_view(&ch, 1), out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  EXPECT_EQ(out.calls[0].input_json, R"({"s":"\"</call>\"","n":1e999})");
}
TEST(ToolCalls, AutoReturnsUndeclaredNamesButRequiredAndNamedRejectThem) {
  for (const auto& choice : {c::ToolChoice{}, c::ToolChoice{c::ToolChoiceKind::kRequired, {}},
                             c::ToolChoice{c::ToolChoiceKind::kNamed, "lookup"}}) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(choice), Limits());
    Output out;
    const auto parsed = parser.Push(Qwen("", "unknown"), out.Sink());
    EXPECT_EQ(parsed.has_value(), choice.kind == c::ToolChoiceKind::kAuto);
    EXPECT_EQ(out.calls.size(), choice.kind == c::ToolChoiceKind::kAuto ? 1U : 0U);
  }
}
TEST(ToolCalls, NonePreservesMarkupAsTextAndDoesNotExtractCalls) {
  c::ToolInterpreter parser(c::QwenToolCalls(), Request({c::ToolChoiceKind::kNone, {}}), Limits());
  Output out;
  const auto text = Qwen("");
  ASSERT_TRUE(parser.Push(text, out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  EXPECT_EQ(out.text, text);
  EXPECT_TRUE(out.calls.empty());
}
TEST(ToolCalls, ParallelFalseFailsOnSecondOpenerBeforeItCanPublish) {
  c::ToolInterpreter parser(c::QwenToolCalls(), Request({}, false), Limits());
  Output out;
  ASSERT_TRUE(parser.Push(Qwen(""), out.Sink()));
  EXPECT_FALSE(parser.Push("<function=", out.Sink()));
  EXPECT_EQ(out.calls.size(), 1U);
  EXPECT_FALSE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
}
TEST(ToolCalls, LengthDropsOnlyIncompleteCallNaturalCompletionFails) {
  for (const auto finish : {c::ToolFinish::kStop, c::ToolFinish::kLength}) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    Output out;
    ASSERT_TRUE(parser.Push("kept " + Qwen("") + "<function=lookup><parameter=s>cut", out.Sink()));
    EXPECT_EQ(parser.Finish(finish, out.Sink()).has_value(), finish == c::ToolFinish::kLength);
    EXPECT_EQ(out.text, "kept ");
    ASSERT_EQ(out.calls.size(), 1U);
  }
  c::ToolInterpreter required(c::QwenToolCalls(), Request({c::ToolChoiceKind::kRequired, {}}),
                              Limits());
  Output out;
  ASSERT_TRUE(required.Push("<function=lookup>", out.Sink()));
  EXPECT_FALSE(required.Finish(c::ToolFinish::kLength, out.Sink()));
}
TEST(ToolCalls, FullyClosedMalformedCallsAreErrorsNotSilentlyDropped) {
  for (const auto text :
       {"<function=></function>", "<function=lookup>bad</function>",
        "<function=lookup><parameter=s>x</parameter><parameter=s>y</parameter></function>"}) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    Output out;
    EXPECT_FALSE(parser.Push(text, out.Sink()));
    EXPECT_TRUE(out.calls.empty());
    EXPECT_FALSE(parser.Finish(c::ToolFinish::kLength, out.Sink()));
  }
}
TEST(ToolCalls, InputBudgetAndCancellationAreTerminal) {
  auto limits = Limits();
  limits.max_bytes = 10;
  c::ToolInterpreter parser(c::QwenToolCalls(), Request(), limits);
  Output out;
  ASSERT_TRUE(parser.Push("12345678", out.Sink()));
  EXPECT_FALSE(parser.Push("901", out.Sink()));
  EXPECT_EQ(out.text, "12345678");
  EXPECT_FALSE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  c::ToolInterpreter cancelled(c::QwenToolCalls(), Request(), Limits());
  EXPECT_FALSE(cancelled.Push("text", [](c::ToolEvent) { return false; }));
  EXPECT_FALSE(cancelled.Push(Qwen(""), out.Sink()));
}
TEST(ToolCalls, ResultOrderingMovesOwnedPayloadsTogetherAndToleratesMissingIds) {
  const std::array<c::ToolUsePart, 2> calls{{{.id = "a", .name = "lookup", .input_json = "{}"},
                                             {.id = "b", .name = "other", .input_json = "{}"}}};
  const std::array<c::ToolResultPart, 2> results{
      {{.tool_use_id = "b", .text = "second", .is_error = true},
       {.tool_use_id = "a", .text = "first"}}};
  auto ordered = c::OrderToolResults(calls, results);
  ASSERT_TRUE(ordered);
  EXPECT_EQ((*ordered)[0].text, "first");
  EXPECT_EQ((*ordered)[1].text, "second");
  EXPECT_TRUE((*ordered)[1].is_error);
  auto missing = results;
  missing[1].tool_use_id.clear();
  ordered = c::OrderToolResults(calls, missing);
  ASSERT_TRUE(ordered);
  EXPECT_EQ((*ordered)[0].text, "second");
  auto invalid = results;
  invalid[1].tool_use_id = "b";
  EXPECT_FALSE(c::OrderToolResults(calls, invalid));
  invalid[1].tool_use_id = "foreign";
  EXPECT_FALSE(c::OrderToolResults(calls, invalid));
}
TEST(ToolCalls, NativeTemplateRegistrationsExposeOnlyVerifiedGrammars) {
  for (const auto& format : c::NativeTemplates()) {
    if (format.name == "gemma-3")
      EXPECT_EQ(format.tool_calls, nullptr);
    else
      EXPECT_NE(format.tool_calls, nullptr) << format.name;
  }
}
TEST(ToolCalls, TextEventsNeverSplitUtf8EvenWhenInputDoes) {
  for (const auto& choice : {c::ToolChoice{}, c::ToolChoice{c::ToolChoiceKind::kNone, {}}}) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(choice), Limits());
    std::string collected;
    auto sink = [&](c::ToolEvent event) {
      const auto* text = std::get_if<c::TextPart>(&event);
      EXPECT_NE(text, nullptr);
      if (text) {
        EXPECT_FALSE(j::FirstInvalidUtf8(text->text));
        collected += text->text;
      }
      return true;
    };
    const std::string text = "héllø 世界";
    for (const char ch : text) ASSERT_TRUE(parser.Push(std::string_view(&ch, 1), sink));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, sink));
    EXPECT_EQ(collected, text);
  }
}
TEST(ToolCalls, GrammarRoundTripsActualNativeRendererCallFragments) {
  auto arguments = j::Parse(R"({"s":"\"quoted\"","n":4,"obj":{"x":[false,null]}})");
  ASSERT_TRUE(arguments);
  c::Conversation conversation;
  conversation.enable_thinking = false;
  conversation.messages = {{.role = c::Role::kUser, .content = "question"},
                           {.role = c::Role::kAssistant,
                            .content = "",
                            .tool_calls = {{.name = "lookup", .arguments = arguments->root()}}}};
  for (const auto& native : c::NativeTemplates()) {
    if (!native.tool_calls) continue;
    SCOPED_TRACE(native.name);
    const auto rendered = native.render(conversation);
    ASSERT_TRUE(rendered);
    const auto& format = *native.tool_calls;
    const auto begin = rendered->text.find(format.call_begin);
    ASSERT_NE(begin, std::string::npos);
    const auto end = rendered->text.find(format.call_end, begin);
    ASSERT_NE(end, std::string::npos);
    // Test only the assistant call fragment emitted by the renderer. The
    // interpreter's runtime contract excludes the rendered prompt itself.
    const auto call =
        std::string_view(rendered->text).substr(begin, end + format.call_end.size() - begin);
    c::ToolInterpreter parser(format, Request(), Limits());
    Output out;
    ASSERT_TRUE(parser.Push(call, out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_EQ(out.calls[0].name, "lookup");
    auto parsed = j::Parse(out.calls[0].input_json);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed->root().find("s")->string(), "\"quoted\"");
    EXPECT_EQ(parsed->root().find("n")->int64(), 4);
    EXPECT_EQ(parsed->root().find("obj")->find("x")->size(), 2U);
  }
}
TEST(ToolCalls, QwenRendererRawSyntaxSurvivesUnknownAndNonstrictSchemas) {
  for (const std::string value :
       {"\"unfinished quote", "{\"unfinished quote", "[\"unfinished quote"}) {
    std::string input = "{\"s\":";
    j::AppendQuoted(value, input);
    input += '}';
    auto arguments = j::Parse(input);
    ASSERT_TRUE(arguments);
    std::vector<std::string_view> schemas{
        R"({})", R"({"properties":{"s":{"type":["string","number"]}}})",
        R"({"properties":{"s":{"type":"integer"}}})", R"({"properties":{"s":{"type":"number"}}})"};
    if (value.front() == '{') schemas.push_back(R"({"properties":{"s":{"type":"array"}}})");
    if (value.front() == '[') schemas.push_back(R"({"properties":{"s":{"type":"object"}}})");
    for (const auto schema : schemas) {
      auto definitions = Definitions();
      definitions[0].parameters_json = schema;
      for (const std::string name : {"lookup", "unknown"}) {
        auto request = c::ToolRequest::Create(definitions, {}, true, Limits());
        ASSERT_TRUE(request);
        c::Conversation conversation;
        conversation.enable_thinking = false;
        conversation.messages = {{.role = c::Role::kUser, .content = "question"},
                                 {.role = c::Role::kAssistant,
                                  .content = "",
                                  .tool_calls = {{.name = name, .arguments = arguments->root()}}}};
        for (const auto& native : c::NativeTemplates()) {
          if (native.tool_calls != &c::QwenToolCalls()) continue;
          SCOPED_TRACE(native.name);
          SCOPED_TRACE(schema);
          SCOPED_TRACE(name);
          const auto rendered = native.render(conversation);
          ASSERT_TRUE(rendered);
          const auto& format = *native.tool_calls;
          const auto begin = rendered->text.find(format.call_begin);
          ASSERT_NE(begin, std::string::npos);
          const auto end = rendered->text.find(format.call_end, begin);
          ASSERT_NE(end, std::string::npos);
          const auto call =
              std::string_view(rendered->text).substr(begin, end + format.call_end.size() - begin);
          for (std::size_t split = 0; split <= call.size(); ++split) {
            SCOPED_TRACE(split);
            c::ToolInterpreter parser(format, *request, Limits());
            Output out;
            ASSERT_TRUE(parser.Push(call.substr(0, split), out.Sink()));
            ASSERT_TRUE(parser.Push(call.substr(split), out.Sink()));
            ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
            ASSERT_EQ(out.calls.size(), 1U);
            EXPECT_EQ(out.calls[0].name, name);
            auto parsed = j::Parse(out.calls[0].input_json);
            ASSERT_TRUE(parsed);
            EXPECT_EQ(parsed->root().find("s")->string(), value);
          }
        }
      }
    }
  }
}
TEST(ToolCalls, EscapedArgumentsUseFundedExpansionWithoutIncreasingRawBudget) {
  // Both ordinary escapes and six-byte control escapes grow beyond raw input.
  for (const char ch : {'\\', '\x01'}) {
    const std::string value(200, ch);
    const auto text = "<function=lookup><parameter=s>" + value + "</parameter></function>";
    auto limits = Limits();
    limits.max_bytes = text.size();
    ASSERT_TRUE(c::ToolOutputBytes(c::QwenToolCalls(), limits.max_bytes));
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), limits);
    Output out;
    ASSERT_TRUE(parser.Push(text, out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_GT(out.calls[0].input_json.size(), limits.max_bytes);
    auto parsed = j::Parse(out.calls[0].input_json);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed->root().find("s")->string(), value);
    c::ToolInterpreter exceeded(c::QwenToolCalls(), Request(), limits);
    Output rejected;
    EXPECT_FALSE(exceeded.Push(text + " ", rejected.Sink()));
    EXPECT_TRUE(rejected.calls.empty());
    EXPECT_FALSE(exceeded.Finish(c::ToolFinish::kLength, rejected.Sink()));
    limits.max_values = 1;
    c::ToolInterpreter values(c::QwenToolCalls(), Request(), limits);
    EXPECT_FALSE(values.Push(text, rejected.Sink()));
  }
  auto impossible = Limits();
  impossible.max_bytes = std::numeric_limits<std::size_t>::max();
  c::ToolInterpreter overflow(c::QwenToolCalls(), Request(), impossible);
  Output out;
  EXPECT_FALSE(overflow.Push("", out.Sink()));
}
TEST(ToolCalls, NormalizedDefinitionEnvelopeUsesItsFundedExpansion) {
  const std::vector<c::ToolDefinition> definitions{
      {.name = "lookup", .description = std::string(200, '\\'), .parameters_json = "{}"}};
  auto limits = Limits();
  limits.max_bytes = definitions[0].name.size() + definitions[0].description.size() +
                     definitions[0].parameters_json.size();
  ASSERT_TRUE(c::ToolRequestBytes(definitions));
  auto request = c::ToolRequest::Create(definitions, {}, true, limits);
  ASSERT_TRUE(request);
  const auto tools = request->TemplateTools();
  ASSERT_EQ(tools.size(), 1U);
  EXPECT_EQ(tools[0].find("function")->find("description")->string(), definitions[0].description);
  --limits.max_bytes;
  EXPECT_FALSE(c::ToolRequest::Create(definitions, {}, true, limits));
  ++limits.max_bytes;
  limits.max_values = 1;  // Expansion does not erase the caller's JSON structure bound.
  EXPECT_FALSE(c::ToolRequest::Create(definitions, {}, true, limits));
}
TEST(ToolCalls, PrepaymentBoundsRejectOverflowAndCoverDenseJson) {
  EXPECT_FALSE(c::ToolOutputBytes(c::QwenToolCalls(), std::numeric_limits<std::size_t>::max()));
  EXPECT_FALSE(j::ParseWorkingBytesBound(std::numeric_limits<std::size_t>::max()));
  EXPECT_TRUE(c::ToolRequestBytes(Definitions()));
  for (const std::string_view text : {"{}", "[0,0,0,0]", R"({"a":[{"b":1}],"s":"escaped\n"})"}) {
    const auto bound = j::ParseWorkingBytesBound(text.size());
    ASSERT_TRUE(bound);
    EXPECT_GE(*bound, j::ParseWorkingBytes(text));
    const auto output = c::ToolOutputBytes(c::QwenToolCalls(), text.size());
    ASSERT_TRUE(output);
    EXPECT_GE(*output, *bound);
  }
}
TEST(ToolCalls, ExplicitJsonSyntaxCannotMasqueradeAsTruncationOrAString) {
  const std::array<std::pair<c::ToolCallFormat, std::string>, 2> cases{
      {{c::ToolCallFormat{.call_begin = "<call>", .call_end = "</call>"},
        R"(<call>{"name":"lookup","arguments":{bad</call>)"},
       {c::DsmlToolCalls(),
        "<｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"obj\" "
        "string=\"false\">{bad</｜DSML｜parameter></｜DSML｜invoke>"}}};
  for (const auto& [format, text] : cases) {
    c::ToolInterpreter parser(format, Request(), Limits());
    Output out;
    EXPECT_FALSE(parser.Push(text, out.Sink()));
    EXPECT_FALSE(parser.Finish(c::ToolFinish::kLength, out.Sink()));
    EXPECT_TRUE(out.calls.empty());
  }
}
}  // namespace

TEST(ToolCalls, AcceptedPauseRetainsCurrentPushSuffixAndFinalDrainExactlyOnce) {
  c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
  Output out;
  std::size_t events = 0;
  const auto sink = [&](c::ToolEvent event) -> c::ToolInterpreter::Flow {
    out.Sink()(std::move(event));
    ++events;
    return c::ToolInterpreter::Flow::kPause;
  };
  ASSERT_TRUE(parser.Push("before" + Qwen("<parameter=n>7</parameter>") + "after", sink));
  EXPECT_TRUE(parser.paused());
  EXPECT_EQ(out.text, "before");
  while (parser.paused()) ASSERT_TRUE(parser.Resume(sink));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, sink));
  while (parser.paused()) ASSERT_TRUE(parser.Resume(sink));
  EXPECT_EQ(out.text, "beforeafter");
  ASSERT_EQ(out.calls.size(), 1U);
  EXPECT_EQ(out.calls[0].input_json, R"({"n":7})");
  EXPECT_EQ(out.reasoning_ends, 1U);
  EXPECT_EQ(events, 4U);
  EXPECT_TRUE(parser.ended());
}

TEST(ToolCalls, CompleteCandidateCannotHideWrongSecondNameOrQuotedDelimiter) {
  const auto request = Request({c::ToolChoiceKind::kNamed, "lookup"});
  for (std::size_t split = 0; split <= std::string("<function=lookup>").size(); ++split) {
    c::ToolInterpreter parser(c::QwenToolCalls(), request, Limits());
    parser.SyntaxOnly();
    const std::string header = "<function=lookup>";
    ASSERT_TRUE(parser.Push(header.substr(0, split), [](c::ToolEvent) { return true; }));
    EXPECT_TRUE(
        parser.Allows(header.substr(split) + "<parameter=s></function></parameter></function>"));
    EXPECT_FALSE(parser.Allows(header.substr(split) + "</function><function=other></function>"));
  }
  c::ToolInterpreter parser(c::QwenToolCalls(), request, Limits());
  parser.SyntaxOnly();
  EXPECT_FALSE(parser.Allows("<tool_call><function=other>"));
  EXPECT_TRUE(
      parser.Allows("<tool_call><function=lookup><parameter=obj>{\"x\":\"</"
                    "function><function=other>\"}</parameter></function></tool_call>"));
}

TEST(ToolCalls, DeclaredReasoningEndIsRecognizedAcrossEverySplitOnlyOutsideArguments) {
  const std::string bytes = "thought</think>answer" + Qwen("<parameter=s></think></parameter>");
  for (std::size_t split = 0; split <= bytes.size(); ++split) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    parser.Reasoning(true);
    Output out;
    ASSERT_TRUE(parser.Push(bytes.substr(0, split), out.Sink()));
    ASSERT_TRUE(parser.Push(bytes.substr(split), out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    EXPECT_EQ(out.reasoning_ends, 1U);
    EXPECT_EQ(out.text, "thoughtanswer");
    ASSERT_EQ(out.calls.size(), 1U);
    EXPECT_EQ(out.calls[0].input_json, R"({"s":"</think>"})");
  }
}

namespace {
namespace tok = llmp::tokenizer;
struct RawTools {
  tok::TokenizerSpec spec;
  RawTools() {
    spec.pre_tokenizer = tok::PreTokenizer::kRawBpe;
    constexpr std::string_view hex = "0123456789ABCDEF";
    for (unsigned b = 0; b < 256; ++b) {
      std::string value = "<0x00>";
      value[3] = hex[b >> 4U];
      value[4] = hex[b & 15U];
      Add(value, tok::TokenKind::kByte);
    }
    for (char ch = 'a'; ch <= 'z'; ++ch) Add(std::string(1, ch));
    Add("▁");  // actual raw-BPE whitespace alphabet required by the constructor
    Add(":");
    Add("{");
    Add("}");
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

TEST(ToolSampling, GemmaRawCallTailAdvancesOnlyByUnchangedOrdinarySelections) {
  RawTools raw;
  const auto marker = raw.Add("<|tool_call>", tok::TokenKind::kControl);
  raw.Merge("c", "a");
  raw.Merge("ca", "l");
  const auto call = raw.Merge("cal", "l");
  const auto colon = std::ranges::find(raw.spec.tokens, ":") - raw.spec.tokens.begin();
  auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
  c::ToolSampling sampling(*tokenizer, c::GemmaToolCalls(),
                           Request({c::ToolChoiceKind::kRequired, {}}), Limits(), false, {});
  sampling.Budget(64, 64);
  auto first = sampling.Plan(marker);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->kind, c::ToolSampling::Selection::Kind::kStableSpan);
  EXPECT_EQ(first->tokens, (std::vector<std::int32_t>{marker}));
  ASSERT_TRUE(sampling.Accept(marker));
  std::vector<float> row(tokenizer->size(), 0), scratch;
  const auto masked = sampling.Mask(row, scratch);
  EXPECT_TRUE(std::isfinite(masked[static_cast<std::size_t>(call)]));
  EXPECT_FALSE(std::isfinite(masked[static_cast<std::size_t>(colon)]));
  auto ordinary = sampling.Plan(call);
  ASSERT_TRUE(ordinary);
  EXPECT_EQ(ordinary->kind, c::ToolSampling::Selection::Kind::kOrdinarySelected);
  EXPECT_EQ(ordinary->tokens, (std::vector<std::int32_t>{call}));
  EXPECT_TRUE(sampling.certain().empty());
  ASSERT_TRUE(sampling.Accept(call));
  auto punctuation = sampling.Plan(static_cast<std::int32_t>(colon));
  ASSERT_TRUE(punctuation);
  EXPECT_EQ(punctuation->kind, c::ToolSampling::Selection::Kind::kOrdinarySelected);
  ASSERT_TRUE(sampling.Accept(static_cast<std::int32_t>(colon)));
  EXPECT_TRUE(sampling.certain().empty());
}

TEST(ToolSampling, FutureCrossingMergeDoesNotTurnOrdinaryRawTailIntoCertainIds) {
  RawTools raw;
  const auto bc = raw.Merge("b", "c");  // higher priority than ab
  const auto ab = raw.Merge("a", "b");
  auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
  std::vector<std::int32_t> future;
  ASSERT_TRUE(tokenizer->Encode("abc", {}, future));
  ASSERT_EQ(future.size(), 2U);
  EXPECT_EQ(future.back(), bc);
  EXPECT_NE(future.front(), ab);
  auto format = c::GemmaToolCalls();
  format.call_begin = "ab";
  c::ToolSampling sampling(*tokenizer, format, Request({c::ToolChoiceKind::kRequired, {}}),
                           Limits(), false, {});
  sampling.Budget(64, 64);
  auto selected = sampling.Plan(ab);
  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->kind, c::ToolSampling::Selection::Kind::kOrdinarySelected);
  EXPECT_EQ(selected->tokens, (std::vector<std::int32_t>{ab}));
  EXPECT_TRUE(sampling.certain().empty());
}
TEST(ToolSampling, OnlyActualTerminalsAreBlockedUntilAnOpenCallCompletes) {
  for (const auto& choice : {c::ToolChoice{c::ToolChoiceKind::kAuto, {}},
                             c::ToolChoice{c::ToolChoiceKind::kRequired, {}},
                             c::ToolChoice{c::ToolChoiceKind::kNamed, "lookup"}}) {
    RawTools raw;
    const auto eos = raw.Add("<eos>", tok::TokenKind::kControl);
    const auto other = raw.Add("<hidden>", tok::TokenKind::kControl);
    auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
    ASSERT_TRUE(tokenizer);
    const std::array terminals{eos};
    c::ToolSampling sampling(*tokenizer, c::QwenToolCalls(), Request(choice), Limits(), false,
                             terminals);
    sampling.Budget(256, 256);
    std::vector<float> row(tokenizer->size(), 0), scratch;
    EXPECT_EQ(std::isfinite(sampling.Mask(row, scratch)[static_cast<std::size_t>(eos)]),
              choice.kind == c::ToolChoiceKind::kAuto);
    for (const unsigned char ch : std::string("<tool_call><function=lookup><parameter=s>cut"))
      ASSERT_TRUE(sampling.Accept(ch));
    const auto masked = sampling.Mask(row, scratch);
    EXPECT_FALSE(std::isfinite(masked[static_cast<std::size_t>(eos)]));
    EXPECT_TRUE(std::isfinite(masked[static_cast<std::size_t>(other)]));
    EXPECT_FALSE(sampling.Plan(eos));
    auto rejected = sampling;
    EXPECT_FALSE(rejected.Accept(eos));
    for (const unsigned char ch : std::string("</parameter></invoke>"))
      ASSERT_TRUE(sampling.Accept(ch));
    EXPECT_TRUE(std::isfinite(sampling.Mask(row, scratch)[static_cast<std::size_t>(eos)]));
    EXPECT_TRUE(sampling.Plan(eos));
    EXPECT_TRUE(sampling.Accept(eos));
  }
}
TEST(ToolSampling, ReasoningBoundaryAndExactRouteMarkerContinueButRealEosWins) {
  RawTools raw;
  const auto eos = raw.Add("<eos>", tok::TokenKind::kControl);
  const auto reasoning = raw.Add("</think>", tok::TokenKind::kControl);
  const auto opener = raw.Add("<|tool_call>", tok::TokenKind::kControl);
  auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
  ASSERT_TRUE(tokenizer);
  const std::array terminals{eos, reasoning};
  c::ToolSampling thinking(*tokenizer, c::QwenToolCalls(),
                           Request({c::ToolChoiceKind::kRequired, {}}), Limits(), true, terminals);
  thinking.Budget(256, 256);
  std::vector<float> row(tokenizer->size(), 0), scratch;
  const auto masked = thinking.Mask(row, scratch);
  EXPECT_FALSE(std::isfinite(masked[static_cast<std::size_t>(eos)]));
  EXPECT_TRUE(std::isfinite(masked[static_cast<std::size_t>(reasoning)]));
  EXPECT_TRUE(thinking.Plan(reasoning));
  EXPECT_TRUE(thinking.Accept(reasoning));
  const std::array route{opener};
  c::ToolSampling gemma(*tokenizer, c::GemmaToolCalls(),
                        Request({c::ToolChoiceKind::kRequired, {}}), Limits(), false,
                        std::span<const std::int32_t>{}, route);
  gemma.Budget(256, 256);
  EXPECT_TRUE(gemma.ContinuesRouteStop(opener));
  EXPECT_FALSE(gemma.ContinuesRouteStop(eos));
  EXPECT_TRUE(std::isfinite(gemma.Mask(row, scratch)[static_cast<std::size_t>(opener)]));
  c::ToolSampling real_eos(*tokenizer, c::GemmaToolCalls(),
                           Request({c::ToolChoiceKind::kRequired, {}}), Limits(), false, route,
                           route);
  real_eos.Budget(256, 256);
  EXPECT_FALSE(std::isfinite(real_eos.Mask(row, scratch)[static_cast<std::size_t>(opener)]));
  EXPECT_FALSE(real_eos.Plan(opener));
  EXPECT_FALSE(real_eos.Accept(opener));
}
TEST(ToolSampling, DeclaredAliasControlTokensReachBothSamplingAndOutputParsers) {
  RawTools raw;
  const auto header = raw.Add("<parameter name=\"", tok::TokenKind::kControl);
  const auto name_end = raw.Add("\">", tok::TokenKind::kControl);
  const auto parameter_end = raw.Add("</parameter>", tok::TokenKind::kControl);
  const auto call_end = raw.Add("</invoke>", tok::TokenKind::kControl);
  const auto eos = raw.Add("<eos>", tok::TokenKind::kControl);
  auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
  ASSERT_TRUE(tokenizer);
  const auto request = Request({c::ToolChoiceKind::kNamed, "lookup"});
  const std::array terminals{eos};
  c::ToolSampling sampling(*tokenizer, c::QwenToolCalls(), request, Limits(), false, terminals);
  c::ToolInterpreter parser(c::QwenToolCalls(), request, Limits());
  sampling.Budget(256, 256);
  Output output;
  for (const unsigned char ch : std::string("<tool_call><function=lookup>")) {
    ASSERT_TRUE(sampling.Accept(ch));
    const char byte = static_cast<char>(ch);
    ASSERT_TRUE(parser.Push(std::string_view(&byte, 1), output.Sink()));
  }
  const std::array<std::int32_t, 6> suffix{header, 's', name_end, 'x', parameter_end, call_end};
  std::vector<float> row(tokenizer->size(), 0), scratch;
  for (const auto token : suffix) {
    if (token >= 256) EXPECT_TRUE(sampling.Preserves(token));
    EXPECT_TRUE(std::isfinite(sampling.Mask(row, scratch)[static_cast<std::size_t>(token)]));
    ASSERT_TRUE(sampling.Plan(token));
    ASSERT_TRUE(sampling.Accept(token));
    std::string piece;
    ASSERT_TRUE(tokenizer->Decode(std::span(&token, 1), {.control_tokens = true}, piece));
    ASSERT_TRUE(parser.Push(piece, output.Sink()));
  }
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, output.Sink()));
  ASSERT_EQ(output.calls.size(), 1U);
  EXPECT_EQ(output.calls[0].input_json, R"({"s":"x"})");
  EXPECT_TRUE(sampling.Plan(eos));
  EXPECT_TRUE(sampling.Accept(eos));
}
TEST(ToolSampling, BareJsonAutoOrdinaryOpenBraceCanEndButRecognizedPartialCallCannot) {
  RawTools raw;
  const auto eos = raw.Add("<eos>", tok::TokenKind::kControl);
  auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
  ASSERT_TRUE(tokenizer);
  for (const auto text : {"Use { to open a block", "{\"name\":\"ordinary\",\"x\":",
                          "{\"name\":\"lookup\",\"parameters\":{"}) {
    const bool ordinary = std::string_view(text).find("lookup") == std::string_view::npos;
    const std::array terminals{eos};
    c::ToolSampling sampling(*tokenizer, c::JsonToolCalls(), Request(), Limits(), false, terminals);
    sampling.Budget(256, 256);
    c::ToolInterpreter parser(c::JsonToolCalls(), Request(), Limits());
    Output output;
    for (const unsigned char ch : std::string(text)) ASSERT_TRUE(sampling.Accept(ch));
    ASSERT_TRUE(parser.Push(text, output.Sink()));
    EXPECT_EQ(parser.CanStop(), ordinary);
    std::vector<float> row(tokenizer->size(), 0), scratch;
    EXPECT_EQ(std::isfinite(sampling.Mask(row, scratch)[static_cast<std::size_t>(eos)]), ordinary);
    EXPECT_EQ(sampling.Plan(eos).has_value(), ordinary);
    EXPECT_EQ(sampling.Accept(eos).has_value(), ordinary);
    EXPECT_EQ(parser.Finish(c::ToolFinish::kStop, output.Sink()).has_value(), ordinary);
    if (ordinary) EXPECT_EQ(output.text, text);
  }
}
}  // namespace

namespace {
TEST(ToolCalls, BareJsonEnvelopeKeepsKeyOrderEscapesAndOrdinaryJsonAcrossEverySplit) {
  const std::string text =
      R"(before {"answer":42} {"parameters":{"s":"} { \"name\": \"other\""}, "na\u006de":"unknown"} after)";
  for (std::size_t split = 0; split <= text.size(); ++split) {
    c::ToolInterpreter parser(c::JsonToolCalls(), Request(), Limits());
    Output output;
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(0, split), output.Sink()));
    ASSERT_TRUE(parser.Push(std::string_view(text).substr(split), output.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, output.Sink()));
    ASSERT_EQ(output.calls.size(), 1U);
    EXPECT_EQ(output.calls[0].name, "unknown");
    EXPECT_EQ(output.text, "before {\"answer\":42}  after");
  }
}
TEST(ToolCalls, BareJsonNameConstraintChecksEveryCompleteTokenIncludingSecondCall) {
  const auto request = Request({c::ToolChoiceKind::kNamed, "lookup"});
  const std::string good = R"({"parameters":{"s":"other"},"name":"lookup"})";
  const std::string bad = R"({"parameters":{},"name":"other"})";
  for (std::size_t split = 0; split <= bad.size(); ++split) {
    c::ToolInterpreter parser(c::JsonToolCalls(), request, Limits());
    parser.SyntaxOnly();
    if (!parser.Allows(std::string_view(bad).substr(0, split))) continue;
    ASSERT_TRUE(
        parser.Push(std::string_view(bad).substr(0, split), [](c::ToolEvent) { return true; }));
    EXPECT_FALSE(parser.Allows(std::string_view(bad).substr(split)));
  }
  c::ToolInterpreter parser(c::JsonToolCalls(), request, Limits());
  EXPECT_TRUE(parser.Allows(good));
  EXPECT_FALSE(parser.Allows(good + bad));
}
TEST(ToolCalls, BareJsonIncompleteAnswerIsRetainedButTruncatedCallIsDropped) {
  for (const auto finish : {c::ToolFinish::kStop, c::ToolFinish::kLength}) {
    c::ToolInterpreter answer(c::JsonToolCalls(), Request(), Limits());
    Output plain;
    ASSERT_TRUE(answer.Push(R"({"answer":"unfinished)", plain.Sink()));
    ASSERT_TRUE(answer.Finish(finish, plain.Sink()));
    EXPECT_EQ(plain.text, R"({"answer":"unfinished)");
    c::ToolInterpreter call(c::JsonToolCalls(), Request(), Limits());
    Output tool;
    ASSERT_TRUE(call.Push(R"({"name":"lookup","parameters":{"s":"unfinished)", tool.Sink()));
    EXPECT_EQ(call.Finish(finish, tool.Sink()).has_value(), finish == c::ToolFinish::kLength);
    EXPECT_TRUE(tool.text.empty());
    EXPECT_TRUE(tool.calls.empty());
  }
  c::ToolInterpreter malformed(c::JsonToolCalls(), Request(), Limits());
  Output output;
  EXPECT_FALSE(malformed.Push(R"({"name":"lookup","parameters":nope})", output.Sink()));
}
}  // namespace

namespace {
// Frozen e0338017 envelope projection, independent of the incremental cursor.
bool Space(char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; }
bool PrefixOf(std::string_view prefix, std::string_view text) { return text.starts_with(prefix); }
std::string_view Strip(std::string_view text) {
  while (!text.empty() && Space(text.front())) text.remove_prefix(1);
  while (!text.empty() && Space(text.back())) text.remove_suffix(1);
  return text;
}
struct DelimiterScan {
  std::size_t at = 0;
  std::vector<char> nesting;
  bool quoted = false, escaped = false;
};
std::optional<std::size_t> Outside(std::string_view text, std::string_view needle,
                                   std::string_view quote, DelimiterScan& scan,
                                   std::string_view second = {}, std::string_view third = {},
                                   bool structured = true) {
  while (scan.at < text.size()) {
    const auto rest = text.substr(scan.at);
    const auto boundary = [&](std::string_view token) {
      if (token.empty() || !rest.starts_with(token)) return false;
      // An XML-like boundary outside a quoted string closes a malformed JSON
      // value too: unclosed braces must not turn a fully closed bad call into
      // a length-truncated call that would silently disappear.
      return scan.nesting.empty() ||
             (token.front() != '}' && token.front() != ']' && token.front() != ',');
    };
    if (!scan.quoted && (boundary(needle) || boundary(second) || boundary(third))) return scan.at;
    if (scan.quoted) {
      if (quote == "\"" && !scan.escaped && rest.front() == '\\') {
        scan.escaped = true;
        ++scan.at;
        continue;
      }
      if (!scan.escaped && rest.starts_with(quote)) {
        scan.quoted = false;
        scan.at += quote.size();
        continue;
      }
      if (!scan.escaped && rest.size() < quote.size() && PrefixOf(rest, quote)) return {};
      scan.escaped = false;
    } else {
      if (structured && rest.starts_with(quote)) {
        scan.quoted = true;
        scan.at += quote.size();
        continue;
      }
      if ((structured && rest.size() < quote.size() && PrefixOf(rest, quote)) ||
          (rest.size() < needle.size() && PrefixOf(rest, needle)) ||
          (rest.size() < second.size() && PrefixOf(rest, second)) ||
          (rest.size() < third.size() && PrefixOf(rest, third)))
        return {};
      const char c = rest.front();
      if (structured && (c == '{' || c == '[')) scan.nesting.push_back(c);
      if ((c == '}' || c == ']') && !scan.nesting.empty()) scan.nesting.pop_back();
    }
    ++scan.at;
  }
  return {};
}

struct EnvelopeInfo {
  bool name = false, arguments = false, allowed = false;
};
// Scan top-level envelope keys without copying argument payloads. Both the
// output interpreter and vocabulary probes use this same lexical projection;
// full JSON validation occurs once when an accepted call closes.
EnvelopeInfo OriginalEnvelope(std::string_view text, const c::ToolCallFormat& format,
                              const c::ToolRequest& request) {
  EnvelopeInfo info;
  std::size_t at = 0;
  while (at < text.size()) {
    while (at < text.size() && (Space(text[at]) || text[at] == ',')) ++at;
    if (at == text.size() || text[at] != '"') break;
    const auto key_start = at++;
    bool escaped = false;
    while (at < text.size()) {
      const char ch = text[at++];
      if (!escaped && ch == '"') break;
      if (!escaped && ch == '\\')
        escaped = true;
      else
        escaped = false;
    }
    if (at > text.size() || text[at - 1] != '"') break;
    std::string key;
    if (at - key_start <= 6 * std::max(format.name_key.size(), format.arguments_key.size()) + 2) {
      const auto bytes = static_cast<std::string>(text.substr(key_start, at - key_start));
      auto parsed = j::Parse(bytes, {.max_bytes = bytes.size(),
                                     .max_depth = bytes.size(),
                                     .max_values = bytes.size(),
                                     .max_string_bytes = bytes.size()});
      if (parsed && parsed->root().is_string()) key = parsed->root().string();
    }
    while (at < text.size() && Space(text[at])) ++at;
    if (at == text.size() || text[at++] != ':') break;
    while (at < text.size() && Space(text[at])) ++at;
    if (at == text.size()) break;
    const auto value = at;
    DelimiterScan scan;
    const auto end = Outside(text.substr(value), ",", "\"", scan);
    at = end ? value + *end : text.size();
    if (key == format.arguments_key) info.arguments = true;
    if (key == format.name_key) {
      info.name = true;
      // The entire name, including JSON escapes, is checked before any token
      // containing its closing quote can be chosen/kept. Other fields may come
      // first, and whitespace/key order are ordinary JSON semantics.
      auto encoded = Strip(text.substr(value, at - value));
      if (!encoded.empty() && encoded.front() == '"' && encoded.back() == '"' &&
          encoded.size() <= 6 * request.NameBytes() + 2) {
        const auto bytes = static_cast<std::string>(encoded);
        auto parsed = j::Parse(bytes, {.max_bytes = bytes.size(),
                                       .max_depth = bytes.size(),
                                       .max_values = bytes.size(),
                                       .max_string_bytes = bytes.size()});
        if (parsed && parsed->root().is_string()) {
          const auto name = parsed->root().string();
          info.allowed = request.choice().kind == c::ToolChoiceKind::kNamed
                             ? name == request.choice().name
                             : request.Declared(name);
        }
      } else if (!end) {
        // An unfinished name is still a viable prefix; final closure must
        // authenticate a complete name. No minimum-completion grammar here.
        info.name = false;
      }
    }
    if (!end) break;
    ++at;
  }
  return info;
}

TEST(ToolCalls, IncrementalEnvelopeMatchesOriginalAtEveryHostilePrefix) {
  const std::vector<std::string> bodies{
      R"("parameters":{"s":"a\\\"b","xs":[{},[],{"n":1}]},"na\u006de":"lookup"   )",
      R"("name":"lookup","name":"unknown","parameters":{)",
      R"("name":"lookup","name":nope,"parameters":{)",
      R"("name":"unknown","name":"lookup","name":"unfinished)",
      R"("name":"lookup","name":"bad\q","parameters":{)",
      R"("parameters":,"name":null, "name":"lookup"x)",
      R"("parameters":{},"name":"lookup",not_a_key,"name":"unknown")",
      R"("na\u006de":"look\u0075p","parameters":{"s":"},\"name\":\"unknown\""})",
      R"("name\"":"ignored","parameters":{},"name":"unknown")",
      "\"" + std::string(1024, 'x') + "\":\"ignored\",\"name\":\"lookup\"",
      "\"parameters\":{\"s\":\"" + std::string(4096, 'x')};
  const auto auto_request = Request();
  const auto named_request = Request({c::ToolChoiceKind::kNamed, "lookup"});
  for (const auto& body : bodies) {
    c::ToolInterpreter incremental(c::JsonToolCalls(), auto_request, Limits());
    c::ToolInterpreter named(c::JsonToolCalls(), named_request, Limits());
    incremental.SyntaxOnly();
    named.SyntaxOnly();
    const auto sink = [](c::ToolEvent) { return true; };
    ASSERT_TRUE(incremental.Push("{", sink));
    ASSERT_TRUE(named.Push("{", sink));
    for (std::size_t size = 0; size <= body.size(); ++size) {
      const auto prefix = std::string_view(body).substr(0, size);
      const auto ordinary = OriginalEnvelope(prefix, c::JsonToolCalls(), auto_request);
      const bool can_stop = !(ordinary.name && ordinary.arguments) && !ordinary.allowed;
      EXPECT_EQ(incremental.CanStop(), can_stop) << prefix;
      EXPECT_EQ(incremental.CanStop(), can_stop) << prefix;
      const auto restricted = OriginalEnvelope(prefix, c::JsonToolCalls(), named_request);
      EXPECT_EQ(named.Allows({}), !restricted.name || restricted.allowed) << prefix;
      if (size == body.size()) break;
      const auto next = std::string_view(body).substr(size, 1);
      const auto future = OriginalEnvelope(std::string_view(body).substr(0, size + 1),
                                           c::JsonToolCalls(), named_request);
      EXPECT_EQ(named.Allows(next), !future.name || future.allowed) << prefix << next;
      ASSERT_TRUE(incremental.Push(next, sink));
      ASSERT_TRUE(named.Push(next, sink));
    }
  }
}

TEST(ToolCalls, IncrementalEnvelopeCopiesKeepIndependentProvisionalNamesAndRollback) {
  const std::string prefix = R"({"parameters":{"s":"value"},"name":"look)";
  c::ToolInterpreter original(c::JsonToolCalls(), Request(), Limits());
  Output output;
  ASSERT_TRUE(original.Push(prefix, output.Sink()));
  EXPECT_TRUE(original.CanStop());
  auto accepted = original;
  auto rejected = original;
  Output good, bad;
  ASSERT_TRUE(accepted.Push("up\"", good.Sink()));
  ASSERT_TRUE(rejected.Push("else\"", bad.Sink()));
  EXPECT_FALSE(accepted.CanStop());
  EXPECT_FALSE(rejected.CanStop());  // both fields authenticate a call, even an undeclared name
  EXPECT_TRUE(original.CanStop());
  EXPECT_TRUE(original.Allows("up\"}"));
  ASSERT_TRUE(accepted.Push("}", good.Sink()));
  ASSERT_TRUE(accepted.Finish(c::ToolFinish::kStop, good.Sink()));
  ASSERT_EQ(good.calls.size(), 1U);
  EXPECT_EQ(good.calls[0].name, "lookup");
  ASSERT_TRUE(original.Finish(c::ToolFinish::kStop, output.Sink()));
  EXPECT_EQ(output.text, prefix);
}

TEST(ToolCalls, PausedBareEnvelopeRetainsWholePendingAndFirstBodyProjections) {
  for (const std::string close : {"}", "}\n", "</call>"}) {
    for (const bool second : {false, true}) {
      const std::string suffix =
          second ? R"( {"name":"other","parameters":{})" + close : " trailing answer";
      auto format = c::JsonToolCalls();
      format.opener_ends_reasoning = true;
      format.call_end = close;
      const std::string input = R"({"name":"lookup","parameters":{})" + close + suffix;
      c::ToolInterpreter parser(format, Request(), Limits());
      Output output;
      ASSERT_TRUE(parser.Push(input, [&](c::ToolEvent event) {
        EXPECT_TRUE(std::holds_alternative<c::ToolReasoningEnd>(event));
        return c::ToolInterpreter::Flow::kPause;
      }));
      ASSERT_TRUE(parser.paused());
      const auto info = OriginalEnvelope(std::string_view(input).substr(1), format, Request());
      EXPECT_EQ(parser.CanStop(), !(info.name && info.arguments) && !info.allowed);
      EXPECT_TRUE(parser.Allows({}));
      auto speculative = parser;
      EXPECT_TRUE(speculative.Allows(" more"));
      ASSERT_TRUE(speculative.Finish(c::ToolFinish::kStop, output.Sink()));
      EXPECT_TRUE(speculative.paused());
      ASSERT_TRUE(speculative.Resume(output.Sink()));
      EXPECT_TRUE(speculative.ended());
      ASSERT_EQ(output.calls.size(), second ? 2U : 1U);
      EXPECT_EQ(output.calls.front().name, "lookup");
      EXPECT_TRUE(parser.paused());
      Output resumed;
      ASSERT_TRUE(parser.Resume(resumed.Sink()));
      EXPECT_TRUE(parser.CanStop());
      EXPECT_TRUE(parser.Allows(" more"));
      ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, resumed.Sink()));
      EXPECT_EQ(resumed.calls.size(), output.calls.size());
      EXPECT_EQ(resumed.text, output.text);
    }
  }
}

}  // namespace

namespace {
TEST(ToolCalls, IncrementalEnvelopeHoldsEverySplitOfDeclaredClosingDelimiter) {
  for (const std::string close : {"}\n", "END", "</call>"}) {
    auto format = c::JsonToolCalls();
    format.call_end = close;
    const std::string body = R"({"parameters":{"s":"value"},"name":"lookup")";
    for (std::size_t split = 0; split < close.size(); ++split) {
      c::ToolInterpreter parser(format, Request(), Limits());
      c::ToolInterpreter named(format, Request({c::ToolChoiceKind::kNamed, "lookup"}), Limits());
      Output output, named_output;
      ASSERT_TRUE(parser.Push(body + close.substr(0, split), output.Sink()));
      ASSERT_TRUE(named.Push(body + close.substr(0, split), named_output.Sink()));
      const auto suffix = std::string_view(close).substr(split);
      EXPECT_TRUE(parser.Allows(suffix)) << close << split;
      EXPECT_TRUE(named.Allows(suffix)) << close << split;
      ASSERT_TRUE(parser.Push(suffix, output.Sink()));
      ASSERT_TRUE(named.Push(suffix, named_output.Sink()));
      ASSERT_EQ(output.calls.size(), 1U) << close << split;
      ASSERT_EQ(named_output.calls.size(), 1U) << close << split;
      EXPECT_EQ(output.calls.front().name, "lookup");
      ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, output.Sink()));
    }
  }
}

TEST(ToolCalls, ReasoningBlocksOpenersUnlessOpenerEndsReasoning) {
  auto format = c::GemmaToolCalls();
  format.reasoning_end = "<channel|>";
  ASSERT_FALSE(format.opener_ends_reasoning);
  const auto named = Request({c::ToolChoiceKind::kNamed, "lookup"});
  c::ToolInterpreter parser(format, named, Limits());
  parser.Reasoning(true);
  // While reasoning is active and opener_ends_reasoning is false, tool call markers
  // in reasoning text are ordinary text rather than tool calls.
  EXPECT_TRUE(parser.Allows("<|tool_call>call:other{}<tool_call|>"));
  Output out;
  ASSERT_TRUE(
      parser.Push("thinking <|tool_call>call:other{}<tool_call|> done<channel|>"
                  "<|tool_call>call:lookup{n:1}<tool_call|>",
                  out.Sink()));
  ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
  EXPECT_EQ(out.reasoning_ends, 1U);
  EXPECT_EQ(out.text, "thinking <|tool_call>call:other{}<tool_call|> done");
  ASSERT_EQ(out.calls.size(), 1U);
  EXPECT_EQ(out.calls[0].name, "lookup");
  EXPECT_EQ(out.calls[0].input_json, R"({"n":1})");

  // Once reasoning has ended, undeclared/wrong-name calls are rejected by Allows.
  c::ToolInterpreter after_reasoning(format, named, Limits());
  after_reasoning.Reasoning(true);
  Output after_out;
  ASSERT_TRUE(after_reasoning.Push("thought<channel|>", after_out.Sink()));
  EXPECT_FALSE(after_reasoning.Allows("<|tool_call>call:other{}"));
  EXPECT_TRUE(after_reasoning.Allows("<|tool_call>call:lookup{}"));
}

TEST(ToolSampling, SpontaneousOpenerDuringWaitingRespectsOpenerEndsReasoningAndForcesSuffix) {
  {
    RawTools raw;
    const auto open_tok = raw.Add("<tool_call>", tok::TokenKind::kControl);
    raw.Add("</think>", tok::TokenKind::kControl);
    auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
    ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
    c::ToolSampling sampling(*tokenizer, c::QwenToolCalls(),
                             Request({c::ToolChoiceKind::kNamed, "lookup"}), Limits(), true, {});
    sampling.Budget(128, 128);
    // Spontaneous <tool_call> during reasoning ends waiting and leaves "<function=lookup>"
    // still forced rather than marking the entire prefix as already forced.
    ASSERT_TRUE(sampling.Accept(open_tok));
    std::vector<float> row(tokenizer->size(), 0), scratch;
    const auto masked = sampling.Mask(row, scratch);
    EXPECT_TRUE(std::isfinite(masked[static_cast<std::size_t>('<')]));
    EXPECT_FALSE(std::isfinite(masked[static_cast<std::size_t>('x')]));
  }
  {
    RawTools raw;
    const auto open_tok = raw.Add("<|tool_call>", tok::TokenKind::kControl);
    const auto end_tok = raw.Add("<channel|>", tok::TokenKind::kControl);
    auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
    ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
    auto format = c::GemmaToolCalls();
    format.reasoning_end = "<channel|>";
    c::ToolSampling sampling(*tokenizer, format, Request({c::ToolChoiceKind::kNamed, "lookup"}),
                             Limits(), true, {});
    sampling.Budget(128, 128);
    // Because opener_ends_reasoning is false for Gemma, <|tool_call> during reasoning
    // is ordinary reasoning text and does not satisfy or trigger the post-reasoning prefix.
    ASSERT_TRUE(sampling.Accept(open_tok));
    std::vector<float> row(tokenizer->size(), 0), scratch;
    const auto during_reasoning = sampling.Mask(row, scratch);
    EXPECT_TRUE(std::isfinite(during_reasoning[static_cast<std::size_t>('x')]));
    ASSERT_TRUE(sampling.Accept(end_tok));
    const auto after_reasoning = sampling.Mask(row, scratch);
    EXPECT_TRUE(std::isfinite(after_reasoning[static_cast<std::size_t>(open_tok)]));
    EXPECT_FALSE(std::isfinite(after_reasoning[static_cast<std::size_t>('x')]));
  }
}

TEST(ToolCalls, BlockBoundariesRequireOpenBlockAndAtLeastOneCall) {
  // Stray block_end outside a block is ordinary text.
  {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    Output out;
    EXPECT_TRUE(parser.Allows("hello </tool_call> world"));
    ASSERT_TRUE(parser.Push("hello </tool_call> world", out.Sink()));
    ASSERT_TRUE(parser.Finish(c::ToolFinish::kStop, out.Sink()));
    EXPECT_EQ(out.text, "hello </tool_call> world");
    EXPECT_TRUE(out.calls.empty());
  }
  // Empty block, nested block opener, and non-whitespace inside a block are rejected.
  for (const std::string_view bad : {"<tool_call></tool_call>", "<tool_call>\n</tool_call>",
                                     "<tool_call>\n<tool_call>", "<tool_call>hello"}) {
    c::ToolInterpreter parser(c::QwenToolCalls(), Request(), Limits());
    EXPECT_FALSE(parser.Allows(bad)) << bad;
    Output out;
    EXPECT_FALSE(parser.Push(bad, out.Sink())) << bad;
  }
}

TEST(ToolCalls, ObjectNotationAllowsValidPrefixesAndRejectsMalformedArgumentsDuringSampling) {
  c::ToolInterpreter parser(c::GemmaToolCalls(), Request(), Limits());
  Output out;
  ASSERT_TRUE(parser.Push("<|tool_call>call:lookup{", out.Sink()));
  for (const std::string_view valid :
       {"}", "s:<|\"|>hi", "s:<|\"|>hi<|\"|>,n:-12", "obj:{a:[true,null,1.5e-2]}",
        "s:<|\"|>hi<|\"|>}<tool_call|>"}) {
    EXPECT_TRUE(parser.Allows(valid)) << valid;
  }
  for (const std::string_view invalid :
       {":1", "s:\"hi\"", "s:truX", "s:01", "s:--1", "s:1,}", "s:1,]", "s:1,,"}) {
    EXPECT_FALSE(parser.Allows(invalid)) << invalid;
  }

  RawTools raw;
  auto tokenizer = tok::Tokenizer::Create(std::move(raw.spec));
  ASSERT_TRUE(tokenizer) << tokenizer.error().ToString();
  c::ToolSampling sampling(*tokenizer, c::GemmaToolCalls(), Request({}, false), Limits(), false,
                           {});
  sampling.Budget(128, 128);
  for (const unsigned char ch : std::string("<|tool_call>call:lookup{"))
    ASSERT_TRUE(sampling.Accept(ch));
  std::vector<float> row(tokenizer->size(), 0), scratch;
  const auto at_key = sampling.Mask(row, scratch);
  const auto colon = static_cast<std::size_t>(*tokenizer->Find(":"));
  const auto rbrace = static_cast<std::size_t>(*tokenizer->Find("}"));
  const auto n_tok = static_cast<std::size_t>(*tokenizer->Find("n"));
  EXPECT_FALSE(std::isfinite(at_key[colon]));
  EXPECT_TRUE(std::isfinite(at_key[rbrace]));
  EXPECT_TRUE(std::isfinite(at_key[n_tok]));
  ASSERT_TRUE(sampling.Accept(static_cast<std::int32_t>(n_tok)));
  ASSERT_TRUE(sampling.Accept(static_cast<std::int32_t>(colon)));
  const auto at_val = sampling.Mask(row, scratch);
  EXPECT_TRUE(std::isfinite(at_val[static_cast<std::size_t>('1')]));
  EXPECT_TRUE(std::isfinite(at_val[static_cast<std::size_t>('-')]));
  EXPECT_FALSE(std::isfinite(at_val[rbrace]));
  EXPECT_FALSE(std::isfinite(at_val[colon]));
}
}  // namespace
