// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "base/json.h"
#include "chat/chat.h"
#include "chat/tool_calls.h"
#include "expected_error.h"
#include "runtime/api.h"
#include "runtime/intake_limits.h"

namespace {
namespace api = llmp::runtime::api;
namespace chat = llmp::chat;
namespace json = llmp::base::json;
using llmp::test_support::Failed;

std::string ToolBody(std::string_view fields) {
  return "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]," +
         std::string(fields) + "}";
}
constexpr std::string_view kDefinitions =
    R"("tools":[{"type":"function","function":{"name":"weather","description":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},{"type":"function","function":{"name":"clock"}}])";

TEST(ApiTools, DefaultsAndNamedDefinitionFilteringAreOwned) {
  api::ChatRequest owned;
  {
    std::string body = ToolBody(kDefinitions);
    auto parsed = api::ParseChatRequest(body);
    ASSERT_TRUE(parsed) << Failed(parsed, &api::Error::message).value_or("");
    owned = std::move(*parsed);
  }
  ASSERT_NE(owned.tools, nullptr);
  EXPECT_EQ(owned.tools->request.choice().kind, chat::ToolChoiceKind::kAuto);
  EXPECT_TRUE(owned.tools->request.parallel());
  EXPECT_TRUE(owned.tools->request.Declared("weather"));
  EXPECT_EQ(owned.tools->request.TemplateTools().size(), 2U);
  EXPECT_EQ(owned.tools->request.Parameter("weather", "city")->find("type")->string(), "string");
  auto named = api::ParseChatRequest(ToolBody(
      std::string(kDefinitions) +
      R"(,"tool_choice":{"type":"function","function":{"name":"clock"}},"parallel_tool_calls":false)"));
  ASSERT_TRUE(named) << Failed(named, &api::Error::message).value_or("");
  EXPECT_EQ(named->tools->request.choice().kind, chat::ToolChoiceKind::kNamed);
  EXPECT_FALSE(named->tools->request.parallel());
  const auto exposed = named->tools->request.TemplateTools();
  ASSERT_EQ(exposed.size(), 1U);
  EXPECT_EQ(exposed[0].find("function")->find("name")->string(), "clock");
}

TEST(ApiTools, EmptyAndNoneRequestsNeedNoToolTemplate) {
  // `auto` with nothing declared is no tools too.
  for (const auto fields :
       {R"("tools":[])", R"("tool_choice":"none")", R"("parallel_tool_calls":false)",
        R"("functions":[])", R"("tools":[],"tool_choice":"auto")", R"("tool_choice":"auto")",
        R"("functions":[],"function_call":"auto")"}) {
    auto parsed = api::ParseChatRequest(ToolBody(fields));
    ASSERT_TRUE(parsed) << fields << Failed(parsed, &api::Error::message).value_or("");
    ASSERT_NE(parsed->tools, nullptr);
    EXPECT_EQ(parsed->tools->request.choice().kind, chat::ToolChoiceKind::kNone);
    EXPECT_TRUE(parsed->tools->request.TemplateTools().empty());
  }
  auto parsed =
      api::ParseChatRequest(ToolBody(std::string(kDefinitions) + R"(,"tool_choice":"none")"));
  ASSERT_TRUE(parsed);
  EXPECT_TRUE(parsed->tools->request.TemplateTools().empty());
}

TEST(ApiTools, LegacyInputsNormalizeAndConflictingRepresentationsFail) {
  auto legacy = api::ParseChatRequest(ToolBody(
      R"("functions":[{"name":"f","parameters":{"type":"object"}}],"function_call":{"name":"f"})"));
  ASSERT_TRUE(legacy);
  EXPECT_EQ(legacy->tools->request.choice().kind, chat::ToolChoiceKind::kNamed);
  EXPECT_EQ(legacy->tools->request.choice().name, "f");
  auto same = api::ParseChatRequest(ToolBody(
      R"("tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}],"functions":[{"name":"f","parameters":{"properties":{},"type":"object"}}],"tool_choice":"required","function_call":"required")"));
  ASSERT_TRUE(same) << Failed(same, &api::Error::message).value_or("");
  for (const auto fields :
       {R"("tools":[],"functions":[{"name":"f"}])",
        R"("tools":[{"type":"function","function":{"name":"f"}}],"functions":[{"name":"g"}])",
        R"("tool_choice":"none","function_call":"auto")"}) {
    auto bad = api::ParseChatRequest(ToolBody(fields));
    ASSERT_FALSE(bad) << fields;
    EXPECT_NE(bad.error().message.find("conflict"), std::string::npos);
  }
}

TEST(ApiTools, InvalidDefinitionsFailExplicitly) {
  for (const auto fields :
       {R"("tools":[{"type":"function","function":{"name":"f","strict":"yes"}}])",
        R"("functions":[{"name":"f","parameters":[]}])",
        R"("functions":[{"name":"f"},{"name":"f"}])",
        R"("functions":[{"name":"f"}],"function_call":{"name":"absent"})",
        R"("tool_choice":"required")", R"("tools":[],"tool_choice":"required")",
        R"("tools":[],"tool_choice":{"type":"function","function":{"name":"f"}})",
        R"("parallel_tool_calls":3)", R"("tools":[{"type":"custom","function":{"name":"f"}}])"}) {
    auto parsed = api::ParseChatRequest(ToolBody(fields));
    ASSERT_FALSE(parsed) << fields;
    EXPECT_EQ(parsed.error().status, 400);
    EXPECT_FALSE(parsed.error().param.empty());
  }
}

// `strict: true` is accepted, as `strict: false` is: a strict definition is
// not refused.
TEST(ApiTools, StrictDefinitionsAreAccepted) {
  for (
      const auto fields :
      {R"("tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object","properties":{"x":{"type":"string"}}}}}])",
       R"("functions":[{"name":"f","strict":true}])"}) {
    auto parsed = api::ParseChatRequest(ToolBody(fields));
    ASSERT_TRUE(parsed) << fields << ": " << Failed(parsed, &api::Error::message).value_or("");
    ASSERT_NE(parsed->tools, nullptr);
    EXPECT_EQ(parsed->tools->request.choice().kind, chat::ToolChoiceKind::kAuto);
    EXPECT_TRUE(parsed->tools->request.Declared("f"));
  }
}

// Both representations of many definitions compare by name in linear time,
// in any order; one missing or different definition is still a conflict.
TEST(ApiTools, ManyEquivalentToolsAndFunctionsCompareByName) {
  constexpr std::size_t kDefinitionsCount = 10'000;
  // The functions in reverse order; the one at `differ` (if any) described
  // differently.
  const auto lists = [](std::size_t differ) {
    std::string tools = R"("tools":[)";
    std::string functions = R"("functions":[)";
    for (std::size_t i = 0; i < kDefinitionsCount; ++i) {
      const std::size_t j = kDefinitionsCount - 1 - i;
      if (i != 0) {
        tools += ',';
        functions += ',';
      }
      tools += R"({"type":"function","function":{"name":"f)" + std::to_string(i) + R"("}})";
      functions += R"({"name":"f)" + std::to_string(j) + '"' +
                   (j == differ ? R"(,"description":"other")" : "") + '}';
    }
    return tools + "]," + functions + "]";
  };
  auto same = api::ParseChatRequest(ToolBody(lists(kDefinitionsCount)));
  ASSERT_TRUE(same) << Failed(same, &api::Error::message).value_or("");
  EXPECT_EQ(same->tools->request.TemplateTools().size(), kDefinitionsCount);
  auto conflict = api::ParseChatRequest(ToolBody(lists(kDefinitionsCount / 2)));
  ASSERT_FALSE(conflict);
  EXPECT_NE(conflict.error().message.find("conflict"), std::string::npos);
}

TEST(ApiTools, AssistantCallsAndLegacyResultEvidenceSurviveInputDestruction) {
  api::ChatRequest request;
  {
    auto parsed = api::ParseChatRequest(R"({"model":"m","messages":[
      {"role":"user","content":"go"},
      {"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{\"x\":1e999}"}}]},
      {"role":"tool","tool_call_id":"a","content":[{"type":"text","text":"bad"}],"is_error":true},
      {"role":"assistant","content":null,"function_call":{"name":"g","arguments":"{}"}},
      {"role":"function","name":"g","content":"ok"}]})");
    ASSERT_TRUE(parsed) << Failed(parsed, &api::Error::message).value_or("");
    request = std::move(*parsed);
  }
  const auto& use = std::get<chat::ToolUsePart>(request.messages[1].content.parts()->at(0));
  EXPECT_EQ(use.input_json, R"({"x":1e999})");
  EXPECT_EQ(use.id, "a");
  const auto& result = std::get<chat::ToolResultPart>(request.messages[2].content.parts()->at(0));
  EXPECT_TRUE(result.is_error);
  EXPECT_EQ(result.text, "bad");
  EXPECT_EQ(request.messages.back().role, api::Role::kTool);
  const auto& old = std::get<chat::ToolResultPart>(request.messages.back().content.parts()->at(0));
  EXPECT_EQ(old.name, "g");
  EXPECT_TRUE(old.tool_use_id.empty());
  EXPECT_TRUE(std::get<chat::ToolUsePart>(request.messages[3].content.parts()->at(0)).id.empty());
}

TEST(ApiTools, OmittedAssistantCallContentMatchesNullAfterInputDestruction) {
  for (
      const auto fields :
      {R"("tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}])",
       R"("function_call":{"name":"f","arguments":"{\"x\":1}"})"}) {
    std::vector<chat::Conversation> projections;
    for (const auto content : {"", "\"content\":null,"}) {
      api::ChatRequest request;
      {
        const auto body = std::string(R"({"model":"m","messages":[{"role":"assistant",)") +
                          content + fields + R"(},{"role":"user","content":"next"}]})";
        auto parsed = api::ParseChatRequest(body);
        ASSERT_TRUE(parsed) << Failed(parsed, &api::Error::message).value_or("");
        request = std::move(*parsed);
      }
      const auto* parts = request.messages[0].content.parts();
      ASSERT_NE(parts, nullptr);
      ASSERT_EQ(parts->size(), 1U);  // no synthetic empty TextPart
      const auto& call = std::get<chat::ToolUsePart>(parts->front());
      EXPECT_EQ(call.name, "f");
      EXPECT_EQ(call.input_json, R"({"x":1})");
      chat::Conversation conversation;
      conversation.messages.push_back(
          {.role = chat::Role::kAssistant, .content = request.messages[0].content});
      auto projected = chat::ProjectContent(conversation);
      ASSERT_TRUE(projected);
      projections.push_back(std::move(*projected));
    }
    const auto& omitted = projections[0].messages[0];
    const auto& explicit_null = projections[1].messages[0];
    ASSERT_EQ(omitted.tool_calls.size(), 1U);
    ASSERT_EQ(explicit_null.tool_calls.size(), 1U);
    EXPECT_EQ(omitted.content.text(), explicit_null.content.text());
    EXPECT_EQ(omitted.tool_calls[0].name, explicit_null.tool_calls[0].name);
    EXPECT_EQ(omitted.tool_calls[0].id, explicit_null.tool_calls[0].id);
    EXPECT_EQ(omitted.tool_calls[0].arguments.find("x")->number(), "1");
    EXPECT_EQ(explicit_null.tool_calls[0].arguments.find("x")->number(), "1");
  }
}

TEST(ApiTools, MissingContentRequiresAValidNonemptyAssistantCall) {
  for (const auto message :
       {R"({"role":"user"})", R"({"role":"tool","tool_call_id":"c"})", R"({"role":"assistant"})",
        R"({"role":"assistant","tool_calls":[]})", R"({"role":"assistant","function_call":null})",
        R"({"role":"assistant","tool_calls":[{}]})",
        R"({"role":"assistant","function_call":{"name":"f","arguments":"bad"}})"}) {
    const auto body = std::string(R"({"model":"m","messages":[)") + message +
                      R"(,{"role":"user","content":"next"}]})";
    auto parsed = api::ParseChatRequest(body);
    ASSERT_FALSE(parsed) << message;
    EXPECT_EQ(parsed.error().status, 400);
    EXPECT_TRUE(parsed.error().param.starts_with("messages[0]."));
  }
}

TEST(ApiTools, EchoedModernAndLegacyCallsMustAgree) {
  auto body = [](std::string_view arguments) {
    return std::string(
               R"({"model":"m","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{\"a\":1,\"b\":2}"}}],"function_call":{"name":"f","arguments":)") +
           std::string(arguments) + R"(}},{"role":"user","content":"next"}]})";
  };
  EXPECT_TRUE(api::ParseChatRequest(body(R"("{\"b\":2,\"a\":1}")")));
  auto conflict = api::ParseChatRequest(body(R"("{\"a\":2}")"));
  ASSERT_FALSE(conflict);
  EXPECT_EQ(conflict.error().param, "messages[0].function_call");
}

TEST(ApiTools, MalformedHistoryArgumentsNeverReachRendering) {
  for (const auto value : {R"("[]")", R"("{broken}")", R"({})", R"(null)"}) {
    const auto body =
        std::string(
            R"({"model":"m","messages":[{"role":"assistant","content":null,"function_call":{"name":"f","arguments":)") +
        value + R"(}},{"role":"user","content":"next"}]})";
    auto parsed = api::ParseChatRequest(body);
    ASSERT_FALSE(parsed) << body;
    EXPECT_EQ(parsed.error().param, "messages[0].function_call");
  }
}

TEST(ApiTools, UnknownFieldAccountingRetainsNestedNames) {
  auto parsed = api::ParseChatRequest(ToolBody(
      R"("tools":[{"type":"function","extension":7,"function":{"name":"f","note":8}}],"tool_choice":{"type":"function","function":{"name":"f","tag":9}})"));
  ASSERT_TRUE(parsed);
  for (const auto name : {"tools[].extension", "tools[].function.note", "tool_choice.function.tag"})
    EXPECT_NE(std::ranges::find(parsed->ignored, name), parsed->ignored.end()) << name;
}

TEST(ApiTools, DefinitionPaymentFollowsSharedOwnerAndIsExcludedFromRequestBytes) {
  llmp::runtime::RequestMemory memory(1U << 24U);
  auto parsed = api::ParseChatRequest(ToolBody(kDefinitions), &memory);
  ASSERT_TRUE(parsed) << Failed(parsed, &api::Error::message).value_or("");
  const auto held = memory.used();
  ASSERT_GT(held, 0U);
  EXPECT_EQ(held, parsed->tools->charge.bytes());
  auto shared = parsed->tools;
  const auto bytes = api::RequestBytes(*parsed);
  parsed->tools.reset();
  EXPECT_EQ(api::RequestBytes(*parsed), bytes);
  EXPECT_EQ(memory.used(), held);
  shared.reset();
  EXPECT_EQ(memory.used(), 0U);
}

TEST(ApiTools, NormalizationRefusesMemoryBeforeOwnedAllocationAndReleasesOnFailure) {
  const auto body = ToolBody(kDefinitions);
  // Enough for the request parse but insufficient for normalized tool schemas.
  const auto parse_bytes = json::ParseWorkingBytes(body) + 2 * body.size() + sizeof(api::Message) +
                           2 * sizeof(chat::ToolDefinition);
  llmp::runtime::RequestMemory memory(parse_bytes + 256);
  auto parsed = api::ParseChatRequest(body, &memory);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().code, "request_too_large");
  EXPECT_NE(parsed.error().message.find("normalized tool definitions"), std::string::npos);
  EXPECT_EQ(memory.used(), 0U);
}
}  // namespace

TEST(ApiTools, CallsSeparateStopMatchingAndNonStreamingUsesNormalizedArguments) {
  api::OutputText text(std::vector<std::string>{"abc"});
  EXPECT_EQ(text.Content("ab").content, "");
  EXPECT_EQ(text.Boundary().content, "ab");
  EXPECT_EQ(text.Content("c").content, "c");
  EXPECT_FALSE(text.stopped());
  const std::vector<chat::ToolUsePart> calls{{"call_1", "weather", R"({"city":"a\\b"})"}};
  const auto body =
      api::CompletionJson("chat_1", 0, "m", "abc", {}, api::Finish::kToolCalls, {}, calls);
  auto parsed = json::Parse(body);
  ASSERT_TRUE(parsed);
  const auto choice = parsed->root().find("choices")->at(0);
  EXPECT_EQ(choice.find("finish_reason")->string(), "tool_calls");
  const auto call = choice.find("message")->find("tool_calls")->at(0);
  EXPECT_EQ(call.find("id")->string(), "call_1");
  EXPECT_EQ(call.find("function")->find("arguments")->string(), calls[0].input_json);
  // A reply of tool calls alone has null content, as OpenAI sends it; text
  // replies keep their (possibly empty) string.
  for (const bool with_calls : {true, false}) {
    const auto reply = api::CompletionJson(
        "chat_2", 0, "m", "", {}, with_calls ? api::Finish::kToolCalls : api::Finish::kStop, {},
        with_calls ? std::span<const chat::ToolUsePart>(calls)
                   : std::span<const chat::ToolUsePart>());
    EXPECT_LE(reply.size(),
              *api::CompletionJsonBytes("chat_2", "m", "", {},
                                        with_calls ? std::span<const chat::ToolUsePart>(calls)
                                                   : std::span<const chat::ToolUsePart>()));
    auto doc = json::Parse(reply);
    ASSERT_TRUE(doc) << reply;
    const auto content = doc->root().find("choices")->at(0).find("message")->find("content");
    ASSERT_TRUE(content);
    if (with_calls) {
      EXPECT_TRUE(content->is_null()) << reply;
    } else {
      ASSERT_TRUE(content->is_string()) << reply;
      EXPECT_EQ(content->string(), "");
    }
  }
  api::OutputText stopped(std::vector<std::string>{"abc"});
  EXPECT_EQ(stopped.Content("xabc").content, "x");
  EXPECT_TRUE(stopped.stopped());
  EXPECT_TRUE(stopped.Boundary().content.empty());
  EXPECT_TRUE(stopped.Content("later").content.empty());
}

// The finish reason a reply reports: a call it emitted is tool_calls even
// when a stop string matched after it; only an allowance's end is length.
TEST(ApiTools, ToolCallsOutrankAStopStringButNotTheTokenAllowance) {
  struct Case {
    bool text_stopped, generation_stopped;
    std::uint64_t calls;
    api::Finish expected;
  };
  for (const auto& c : std::vector<Case>{
           {false, true, 0, api::Finish::kStop},
           {false, true, 2, api::Finish::kToolCalls},
           {true, false, 0, api::Finish::kStop},
           {true, false, 1, api::Finish::kToolCalls},  // a stop string after a call
           {true, true, 1, api::Finish::kToolCalls},
           {false, false, 0, api::Finish::kLength},
           {false, false, 1, api::Finish::kLength},
       }) {
    EXPECT_EQ(api::FinishReason(c.text_stopped, c.generation_stopped, c.calls), c.expected);
  }
}

// An assistant's text, scalar or in parts, comes before its calls (moved
// into the message's parts, not copied).
TEST(ApiTools, AssistantTextPrecedesItsCalls) {
  for (const std::string_view content :
       {std::string_view(R"("before")"),
        std::string_view(R"([{"type":"text","text":"before"},{"type":"text","text":"again"}])")}) {
    SCOPED_TRACE(content);
    auto parsed = api::ParseChatRequest(
        std::string(R"({"model":"m","messages":[
      {"role":"user","content":"go"},
      {"role":"assistant","content":)") +
        std::string(content) +
        R"(,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},
      {"role":"tool","tool_call_id":"a","content":"ok"}]})");
    ASSERT_TRUE(parsed) << Failed(parsed, &api::Error::message).value_or("");
    const auto* parts = parsed->messages[1].content.parts();
    ASSERT_NE(parts, nullptr);
    const bool listed = content.front() == '[';
    ASSERT_EQ(parts->size(), listed ? 3U : 2U);
    EXPECT_EQ(std::get<chat::TextPart>(parts->at(0)).text, "before");
    if (listed) EXPECT_EQ(std::get<chat::TextPart>(parts->at(1)).text, "again");
    EXPECT_EQ(std::get<chat::ToolUsePart>(parts->back()).name, "f");
  }
}
