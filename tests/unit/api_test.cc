// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The chat route (D-097 as amended 2026-09-28; runtime/api.h, http.h,
// binding.h, api_server.h): request parsing and every intake bound, the
// unknown fields' names, the output's stop strings and reasoning split,
// the JSON shapes, the bind resolution over fake interface lists and the
// Host and Origin guard, and the server end to end over real loopback
// sockets with a fake backend: routes, browser guards, HTTP bounds,
// timeouts, keep-alive, pipelining, idle connections, the queue,
// keepalive comments, slow clients, streaming and errors after the
// headers, and the progress watchdog and scaled deadlines (watchdog.h).

#include "runtime/api.h"

#include <arpa/inet.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/json.h"
#include "config/node_config.h"
#include "platform/interfaces.h"
#include "runtime/api_server.h"
#include "runtime/binding.h"
#include "runtime/completion_tokens.h"
#include "runtime/http.h"
#include "runtime/watchdog.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

namespace {

namespace api = jitllm::runtime::api;
namespace json = jitllm::base::json;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::StartsWith;

// A body's parse result: the request, or the error's status and message.
std::expected<api::ChatRequest, api::Error> Parse(std::string_view body) {
  return api::ParseChatRequest(body);
}

std::string ErrorOf(std::string_view body) {
  auto r = Parse(body);
  if (r.has_value()) {
    return "(accepted)";
  }
  return std::format("{} {} [{}]", r.error().status, r.error().message, r.error().param);
}

// The member at `path` below `v`, which the test expects; a missing one
// fails the test and yields the last value found.
json::Value In(json::Value v, std::initializer_list<std::string_view> path) {
  for (const std::string_view name : path) {
    const std::optional<json::Value> found = v.find(name);
    if (!found.has_value()) {
      ADD_FAILURE() << "no member " << name;
      return v;
    }
    v = *found;
  }
  return v;
}

constexpr std::string_view kMinimal =
    R"({"model":"m","messages":[{"role":"user","content":"hi"}]})";

std::string WithField(std::string_view field) {
  return std::format(R"({{"model":"m","messages":[{{"role":"user","content":"hi"}}],{}}})", field);
}

TEST(ChatRequest, ReadsTheHonoredFields) {
  auto minimal = Parse(kMinimal);
  ASSERT_TRUE(minimal.has_value());
  EXPECT_EQ(minimal->model, "m");
  ASSERT_EQ(minimal->messages.size(), 1U);
  EXPECT_EQ(minimal->messages[0].content, "hi");
  EXPECT_FALSE(minimal->max_tokens.has_value());
  EXPECT_EQ(minimal->temperature, 1.0);
  EXPECT_EQ(minimal->top_p, 1.0);
  EXPECT_FALSE(minimal->stream);

  auto full = Parse(R"({"model":"deepseek","messages":[
      {"role":"developer","content":"be brief"},
      {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b","cache_control":{}}]},
      {"role":"assistant","content":"c","reasoning_content":"r","refusal":null,"annotations":[]},
      {"role":"user","content":"d","name":"pat"}],
    "max_completion_tokens":32,"temperature":0,"top_p":0.5,"seed":-1,"stop":["x","yz"],
    "stream":true,"stream_options":{"include_usage":true},"n":1,"presence_penalty":0,
    "logprobs":false,"tools":[],"tool_choice":"none","response_format":{"type":"text"},
    "user":"u","metadata":{"k":"v"},"store":false,"parallel_tool_calls":true,
    "prompt_cache_key":"p","frequency_penalty":null})");
  ASSERT_TRUE(full.has_value()) << full.error().message;
  EXPECT_EQ(full->messages[0].role, api::Role::kSystem);
  EXPECT_EQ(full->messages[1].content, "ab");
  EXPECT_EQ(full->messages[2].role, api::Role::kAssistant);
  EXPECT_EQ(full->messages[2].reasoning, std::optional<std::string>("r"));
  EXPECT_EQ(full->max_tokens, std::optional<std::uint32_t>(32));
  EXPECT_EQ(full->temperature, 0.0);
  EXPECT_EQ(full->top_p, 0.5);
  EXPECT_EQ(full->seed, std::optional<std::uint64_t>(~std::uint64_t{0}));
  EXPECT_THAT(full->stop, ::testing::ElementsAre("x", "yz"));
  EXPECT_TRUE(full->stream);
  EXPECT_TRUE(full->include_usage);
  auto one_stop = Parse(WithField(R"("stop":"end","max_tokens":5,"max_completion_tokens":5)"));
  ASSERT_TRUE(one_stop.has_value());
  EXPECT_THAT(one_stop->stop, ::testing::ElementsAre("end"));
  EXPECT_EQ(one_stop->max_tokens, std::optional<std::uint32_t>(5));
}

TEST(ChatRequest, RefusesWhatItDoesNotHonor) {
  EXPECT_THAT(ErrorOf("{"), StartsWith("400 the body is not valid JSON"));
  EXPECT_THAT(ErrorOf("[]"), StartsWith("400 the body must be a JSON object"));
  EXPECT_THAT(ErrorOf(R"({"messages":[{"role":"user","content":"x"}]})"), HasSubstr("[model]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m"})"), HasSubstr("[messages]"));
  // Known fields that ask for what the route does not do are refused, and
  // OpenRouter's transforms and plugins whatever their value (D-046).
  EXPECT_THAT(ErrorOf(WithField(R"("transforms":["middle-out"])")), HasSubstr("[transforms]"));
  EXPECT_THAT(ErrorOf(WithField(R"("transforms":null)")), HasSubstr("[transforms]"));
  EXPECT_THAT(ErrorOf(WithField(R"("plugins":[])")), HasSubstr("[plugins]"));
  EXPECT_THAT(ErrorOf(WithField(R"("n":2)")), HasSubstr("[n]"));
  EXPECT_THAT(ErrorOf(WithField(R"("logprobs":true)")), HasSubstr("[logprobs]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_logprobs":2)")), HasSubstr("[top_logprobs]"));
  EXPECT_THAT(ErrorOf(WithField(R"("presence_penalty":0.5)")), HasSubstr("[presence_penalty]"));
  EXPECT_THAT(ErrorOf(WithField(R"("repetition_penalty":1.1)")), HasSubstr("[repetition_penalty]"));
  EXPECT_THAT(ErrorOf(WithField(R"("tools":[{"type":"function"}])")), HasSubstr("[tools]"));
  EXPECT_THAT(ErrorOf(WithField(R"("tool_choice":"required")")), HasSubstr("[tool_choice]"));
  EXPECT_THAT(ErrorOf(WithField(R"("response_format":{"type":"json_object"})")),
              HasSubstr("[response_format]"));
  EXPECT_THAT(ErrorOf(WithField(R"("logit_bias":{"1":2})")), HasSubstr("[logit_bias]"));
  EXPECT_THAT(ErrorOf(WithField(R"("modalities":["text","audio"])")), HasSubstr("[modalities]"));
  EXPECT_THAT(ErrorOf(WithField(R"("audio":{"voice":"x"})")), HasSubstr("[audio]"));
  EXPECT_THAT(ErrorOf(WithField(R"("store":true)")), HasSubstr("[store]"));
  EXPECT_THAT(ErrorOf(WithField(R"("stream":"yes")")), HasSubstr("[stream]"));
  EXPECT_THAT(ErrorOf(WithField(R"("stream_options":{"include_usage":1})")),
              HasSubstr("[stream_options.include_usage]"));
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":4,"max_completion_tokens":5)")),
              HasSubstr("[max_completion_tokens]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_k":-2)")), HasSubstr("[top_k]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_k":2147483648)")), HasSubstr("[top_k]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_k":1.5)")), HasSubstr("[top_k]"));
  EXPECT_THAT(ErrorOf(WithField(R"("min_p":1.01)")), HasSubstr("[min_p]"));
  EXPECT_THAT(ErrorOf(WithField(R"("min_p":-0.1)")), HasSubstr("[min_p]"));
  // Messages.
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[]})"), HasSubstr("[messages]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"tool","content":"x"}]})"),
              HasSubstr("tools are not supported"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"robot","content":"x"}]})"),
              HasSubstr("[messages[0].role]"));
  EXPECT_THAT(
      ErrorOf(R"({"model":"m","messages":[{"role":"user","content":[{"type":"image_url"}]}]})"),
      HasSubstr("[messages[0].content[0].type]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"user"}]})"),
              HasSubstr("[messages[0].content]"));
  EXPECT_THAT(
      ErrorOf(R"({"model":"m","messages":[{"role":"assistant","content":"x","tool_calls":[{}]}]})"),
      HasSubstr("tool calls are not supported"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[
                 {"role":"assistant","content":"x","function_call":{"name":"f"}},
                 {"role":"user","content":"y"}]})"),
              HasSubstr("[messages[0].function_call]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[
                 {"role":"assistant","content":"x","audio":{"id":"a"}},
                 {"role":"user","content":"y"}]})"),
              HasSubstr("[messages[0].audio]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[
                 {"role":"assistant","content":null,"refusal":"no"},
                 {"role":"user","content":"y"}]})"),
              HasSubstr("[messages[0].refusal]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"user","content":"a"},
                                                  {"role":"assistant","content":"b"}]})"),
              HasSubstr("the last message must be the user's"));
}

// Unknown fields are ignored and their names (never their values) kept,
// each once, with their place; known ones are not.
TEST(ChatRequest, IgnoresUnknownFieldsByName) {
  auto r = Parse(R"({"model":"m","frobnicate":{"deep":[1,2]},"zeta":null,"reasoning_effort":"low",
      "messages":[
        {"role":"system","content":"s","x_note":1},
        {"role":"assistant","content":"a","function_call":null,"audio":null,"refusal":null,
         "tool_calls":[],"annotations":[],"x_note":2},
        {"role":"user","content":[{"type":"text","text":"u","x_part":true}],"x_note":3}],
      "stream_options":{"include_usage":true,"x_opt":"v"},
      "top_k":40,"min_p":0.05,"repetition_penalty":1,"user":"u","frobnicate2":1})");
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_THAT(r->ignored,
              ::testing::UnorderedElementsAre("frobnicate", "zeta", "reasoning_effort",
                                              "messages[].x_note", "messages[].content[].x_part",
                                              "stream_options.x_opt", "frobnicate2"));
  EXPECT_EQ(r->top_k, 40U);
  EXPECT_FLOAT_EQ(static_cast<float>(r->min_p), 0.05F);
  EXPECT_TRUE(r->include_usage);
  auto off = Parse(WithField(R"("top_k":-1,"min_p":0)"));
  ASSERT_TRUE(off.has_value());
  EXPECT_EQ(off->top_k, 0U);
  EXPECT_TRUE(Parse(kMinimal)->ignored.empty());
  // A long name is cut at a character; a request's names are bounded.
  const std::string long_name = std::string(62, 'a') + "\xC3\xA9\xC3\xA9";
  auto cut = Parse(WithField(std::format(R"("{}":1)", long_name)));
  ASSERT_TRUE(cut.has_value());
  EXPECT_THAT(cut->ignored, ElementsAre(std::string(62, 'a') + "\xC3\xA9"));
  std::string many;
  for (std::size_t i = 0; i < api::kMaxIgnoredPerRequest + 10; ++i) {
    many += std::format(R"({}"u{}":0)", i == 0 ? "" : ",", i);
  }
  auto bounded = Parse(WithField(many));
  ASSERT_TRUE(bounded.has_value());
  EXPECT_EQ(bounded->ignored.size(), api::kMaxIgnoredPerRequest);
}

TEST(IgnoredFields, CountsNamesUpToItsSize) {
  api::IgnoredFields table;
  EXPECT_THAT(table.Record({"a", "b"}, 100), ElementsAre("a", "b"));
  EXPECT_THAT(table.Record({"a", "c"}, 200), ElementsAre("c"));
  auto doc = json::Parse(table.Json());
  ASSERT_TRUE(doc.has_value()) << table.Json();
  const json::Value a = In(doc->root(), {"data"}).at(0);
  EXPECT_EQ(In(a, {"name"}).string(), "a");
  EXPECT_EQ(In(a, {"count"}).int64(), 2);
  EXPECT_EQ(In(a, {"first_seen"}).int64(), 100);
  EXPECT_EQ(In(a, {"last_seen"}).int64(), 200);
  for (std::size_t i = 0; i < api::IgnoredFields::kMaxNames + 5; ++i) {
    (void)table.Record({std::format("n{}", i)}, 300);
  }
  auto full = json::Parse(table.Json());
  ASSERT_TRUE(full.has_value());
  EXPECT_EQ(In(full->root(), {"data"}).size(), api::IgnoredFields::kMaxNames);
  EXPECT_EQ(In(full->root(), {"unrecorded"}).int64(), 8);  // 3 names were in before
  EXPECT_THAT(table.Record({"a"}, 400), IsEmpty());        // a known name is still counted
}

// Every numeric bound, at its edge and one past it.
TEST(ChatRequest, EnforcesItsBounds) {
  // Messages: count and bytes.
  const auto messages = [](std::size_t n, std::size_t bytes) {
    std::string list;
    for (std::size_t i = 0; i < n; ++i) {
      list += std::format(R"({}{{"role":"user","content":"{}"}})", i == 0 ? "" : ",",
                          std::string(bytes, 'a'));
    }
    return std::format(R"({{"model":"m","messages":[{}]}})", list);
  };
  EXPECT_TRUE(Parse(messages(api::kMaxMessages, 1)).has_value());
  EXPECT_THAT(ErrorOf(messages(api::kMaxMessages + 1, 1)), HasSubstr("more than 1024"));
  EXPECT_TRUE(Parse(messages(1, api::kMaxMessageBytes)).has_value());
  EXPECT_THAT(ErrorOf(messages(1, api::kMaxMessageBytes + 1)), HasSubstr("longer than"));
  // Parts: count and joined bytes.
  const auto parts = [](std::size_t n, std::size_t bytes) {
    std::string list;
    for (std::size_t i = 0; i < n; ++i) {
      list += std::format(R"({}{{"type":"text","text":"{}"}})", i == 0 ? "" : ",",
                          std::string(bytes, 'a'));
    }
    return std::format(R"({{"model":"m","messages":[{{"role":"user","content":[{}]}}]}})", list);
  };
  EXPECT_TRUE(Parse(parts(api::kMaxContentParts, 1)).has_value());
  EXPECT_THAT(ErrorOf(parts(api::kMaxContentParts + 1, 1)), HasSubstr("more than 64"));
  EXPECT_THAT(ErrorOf(parts(2, (api::kMaxMessageBytes / 2) + 1)), HasSubstr("longer than"));
  // The model's name.
  EXPECT_TRUE(Parse(std::format(R"({{"model":"{}","messages":[{{"role":"user","content":"x"}}]}})",
                                std::string(api::kMaxModelBytes, 'm')))
                  .has_value());
  EXPECT_THAT(
      ErrorOf(std::format(R"({{"model":"{}","messages":[{{"role":"user","content":"x"}}]}})",
                          std::string(api::kMaxModelBytes + 1, 'm'))),
      HasSubstr("[model]"));
  EXPECT_THAT(ErrorOf(R"({"model":"","messages":[{"role":"user","content":"x"}]})"),
              HasSubstr("[model]"));
  // Tokens, temperature, top_p, seed.
  EXPECT_TRUE(Parse(WithField(R"("max_tokens":262144)")).has_value());
  EXPECT_TRUE(Parse(WithField(R"("max_tokens":1048576)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":1048577)")), HasSubstr("[max_tokens]"));
  EXPECT_TRUE(Parse(WithField(R"("max_completion_tokens":1048576)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("max_completion_tokens":1048577)")),
              HasSubstr("[max_completion_tokens]"));
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":0)")), HasSubstr("[max_tokens]"));
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":1.5)")), HasSubstr("[max_tokens]"));
  EXPECT_TRUE(Parse(WithField(R"("temperature":2)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("temperature":2.01)")), HasSubstr("[temperature]"));
  EXPECT_THAT(ErrorOf(WithField(R"("temperature":-0.1)")), HasSubstr("[temperature]"));
  EXPECT_THAT(ErrorOf(WithField(R"("temperature":1e999)")), HasSubstr("[temperature]"));
  EXPECT_TRUE(Parse(WithField(R"("top_p":1)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("top_p":0)")), HasSubstr("[top_p]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_p":1.0001)")), HasSubstr("[top_p]"));
  // A top_p the sampler's float rounds to 0 would fail mid-generation.
  EXPECT_THAT(ErrorOf(WithField(R"("top_p":1e-50)")), HasSubstr("[top_p]"));
  EXPECT_TRUE(Parse(WithField(R"("top_p":1e-30)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("seed":9223372036854775808)")), HasSubstr("[seed]"));
  EXPECT_THAT(ErrorOf(WithField(R"("seed":1.0)")), HasSubstr("[seed]"));
  // Stop strings: count and bytes.
  EXPECT_TRUE(Parse(WithField(R"("stop":["a","b","c","d"])")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("stop":["a","b","c","d","e"])")), HasSubstr("[stop]"));
  EXPECT_TRUE(Parse(WithField(std::format(R"("stop":"{}")", std::string(api::kMaxStopBytes, 's'))))
                  .has_value());
  EXPECT_THAT(
      ErrorOf(WithField(std::format(R"("stop":"{}")", std::string(api::kMaxStopBytes + 1, 's')))),
      HasSubstr("[stop]"));
  EXPECT_THAT(ErrorOf(WithField(R"("stop":[""])")), HasSubstr("[stop[0]]"));
  // JSON depth and size.
  std::string deep(api::kMaxJsonDepth + 1, '[');
  deep += std::string(api::kMaxJsonDepth + 1, ']');
  EXPECT_THAT(ErrorOf(WithField(std::format(R"("metadata":{{"a":{}}})", deep))),
              HasSubstr("not valid JSON"));
  EXPECT_THAT(ErrorOf(std::string(api::kMaxBodyBytes + 1, ' ')), HasSubstr("not valid JSON"));
}

TEST(HttpRequest, ChecksTheCodingBodyCeilingFromTheHead) {
  const auto head = [](std::size_t bytes) {
    return std::format(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        bytes);
  };
  auto request = jitllm::runtime::http::ParseHead(head(api::kMaxBodyBytes), {});
  ASSERT_TRUE(request.has_value()) << request.error().message;
  EXPECT_EQ(request->body_bytes, std::size_t{16} << 20U);
  EXPECT_TRUE(request->body.empty());
  request = jitllm::runtime::http::ParseHead(head(api::kMaxBodyBytes + 1), {});
  ASSERT_FALSE(request.has_value());
  EXPECT_EQ(request.error().status, 413);
}

TEST(ChatRequest, AcceptsLargeCodingTextWithinBoundedIntake) {
  // A roughly 4 MiB coding prompt fits with doubled JSON escaping. This
  // checks the byte envelope; the selected model checks token counts later.
  const std::string content(std::size_t{4} << 20U, '\\');
  auto request =
      Parse(std::format(R"({{"model":"m","messages":[{{"role":"user","content":"{}"}}]}})",
                        std::string(std::size_t{8} << 20U, '\\')));
  ASSERT_TRUE(request.has_value()) << request.error().message;
  ASSERT_EQ(request->messages.size(), 1U);
  EXPECT_EQ(request->messages.front().content, content);
  // The complete body still has a hard limit even when each field fits.
  std::string body = WithField(R"("max_tokens":1)");
  body.resize(api::kMaxBodyBytes, ' ');
  EXPECT_TRUE(Parse(body).has_value());
  body.push_back(' ');
  EXPECT_THAT(ErrorOf(body), HasSubstr("not valid JSON"));
}

TEST(OutputText, EndsAtAStopStringAcrossPieces) {
  api::OutputText text({"STOP", "\n\n"});
  EXPECT_EQ(text.Content("Hello ST").content, "Hello ");  // "ST" might begin STOP
  EXPECT_EQ(text.Content("ORM").content, "STORM");
  EXPECT_EQ(text.Content(" and\n").content, " and");
  EXPECT_EQ(text.Content("\nmore").content, "");
  EXPECT_TRUE(text.stopped());
  EXPECT_EQ(text.Content("ignored").content, "");
  EXPECT_EQ(text.Finish().content, "");
}

TEST(OutputText, FlushesWhatItHeldAtTheEnd) {
  api::OutputText text({"STOP"});
  EXPECT_EQ(text.Content("abcSTO").content, "abc");
  EXPECT_EQ(text.Finish().content, "STO");
  EXPECT_FALSE(text.stopped());
}

TEST(OutputText, SplitsReasoningAndTrimsTheAnswersStart) {
  api::OutputText text({"x"});
  EXPECT_EQ(text.Reasoning("I think x").reasoning, "I think x");  // no stop in reasoning
  EXPECT_EQ(text.Content("\n\n").content, "");
  EXPECT_EQ(text.Content(" Paris").content, "Paris");
  EXPECT_EQ(text.Content("\n ok").content, "\n ok");
  api::OutputText plain({});
  EXPECT_EQ(plain.Content("\nkept").content, "\nkept");
  // A reasoning block that said nothing still trims the answer's start.
  api::OutputText empty({});
  EXPECT_EQ(empty.Reasoning({}).reasoning, "");
  EXPECT_EQ(empty.Content("\n\nParis").content, "Paris");
}

// Untrusted text: NUL and escapes pass through as text; ill-formed UTF-8
// is refused.
TEST(ChatRequest, TakesTextAsText) {
  auto nul = Parse(R"({"model":"m","messages":[{"role":"user","content":"a\u0000b"}]})");
  ASSERT_TRUE(nul.has_value());
  using namespace std::string_view_literals;
  EXPECT_EQ(nul->messages[0].content, "a\0b"sv);
  EXPECT_THAT(ErrorOf("{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"\xC3\"}]}"),
              HasSubstr("not valid JSON"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","model":"n","messages":[{"role":"user","content":"x"}]})"),
              HasSubstr("not valid JSON"));  // a duplicate key
}

TEST(LiteralRequest, ReadsTextAndExactIdsWithBothScoringConventions) {
  auto text = api::ParseCompletionRequest(
      R"({"model":"m","prompt":"a\u0000é","echo":true,"logprobs":1,"max_tokens":1,"temperature":0})");
  ASSERT_TRUE(text.has_value());
  EXPECT_EQ(text->prompt, std::optional<std::string>(std::string("a\0é", 4)));
  EXPECT_TRUE(text->echo);
  EXPECT_EQ(text->logprobs, 1U);
  EXPECT_EQ(text->options.max_tokens, 1U);
  EXPECT_TRUE(text->options.messages.empty());
  auto ids = api::ParseCompletionRequest(
      R"({"model":"m","prompt":[3,0,2147483647],"max_tokens":0,"prompt_logprobs":0,"echo":false,"add_special_tokens":true,"return_tokens_as_token_ids":true})");
  ASSERT_TRUE(ids.has_value());
  EXPECT_FALSE(ids->prompt.has_value());
  EXPECT_THAT(ids->token_ids, ElementsAre(3, 0, 2147483647));
  EXPECT_EQ(ids->options.max_tokens, 0U);
  EXPECT_EQ(ids->prompt_logprobs, 0U);
  EXPECT_FALSE(ids->echo);
  EXPECT_TRUE(ids->return_tokens_as_token_ids);
  auto empty = api::ParseCompletionRequest(R"({"model":"m","prompt":""})");
  ASSERT_TRUE(empty.has_value());  // model BOS availability checked at admission
  EXPECT_EQ(empty->options.max_tokens, 16U);
}

TEST(LiteralRequest, RefusesMalformedIdsAndUnimplementedSemanticFields) {
  for (const std::string_view prompt : {"null", "[]", "[[]]", "[\"x\"]", "[true]", "[-1]", "[1.0]",
                                        "[2147483648]", "[[1],[2]]", R"(["a","b"])", "{}"}) {
    auto r = api::ParseCompletionRequest(std::format(R"({{"model":"m","prompt":{}}})", prompt));
    ASSERT_FALSE(r.has_value()) << prompt;
    EXPECT_EQ(r.error().param, "prompt");
  }
  for (const std::string_view field :
       {R"("logprobs":true)", R"("logprobs":-1)", R"("prompt_logprobs":6)", R"("echo":1)",
        R"("stream":true)", R"("best_of":2)", R"("suffix":"x")", R"("truncate_prompt_tokens":1)",
        R"("allowed_token_ids":[1])", R"("ignore_eos":true)", R"("use_beam_search":true)",
        R"("min_tokens":1)", R"("skip_special_tokens":false)", R"("messages":[])",
        R"("max_tokens":-1)", R"("max_tokens":0,"max_completion_tokens":1)"}) {
    EXPECT_FALSE(
        api::ParseCompletionRequest(std::format(R"({{"model":"m","prompt":"x",{}}})", field))
            .has_value())
        << field;
  }
  EXPECT_FALSE(api::ParseCompletionRequest(R"({"model":"m"})").has_value());
  EXPECT_FALSE(
      api::ParseCompletionRequest(R"({"model":"m","prompt":"a","prompt":[1]})").has_value());
  EXPECT_FALSE(api::ParseCompletionRequest("{\"model\":\"m\",\"prompt\":\"\xC3\"}").has_value());
  EXPECT_FALSE(Parse(WithField(R"("max_tokens":0)")).has_value());
}

TEST(LiteralRequest, SharesTheHardIntakeAndUnknownNameBounds) {
  std::string body = R"({"model":"m","prompt":"x","unrecognized":{"secret":"not logged"}})";
  auto r = api::ParseCompletionRequest(body);
  ASSERT_TRUE(r.has_value());
  EXPECT_THAT(r->options.ignored, ElementsAre("unrecognized"));
  body.resize(api::kMaxBodyBytes, ' ');
  EXPECT_TRUE(api::ParseCompletionRequest(body).has_value());
  body.push_back(' ');
  EXPECT_FALSE(api::ParseCompletionRequest(body).has_value());
  const std::string too_long(api::kMaxMessageBytes + 1, 'x');
  EXPECT_FALSE(
      api::ParseCompletionRequest(std::format(R"({{"model":"m","prompt":"{}"}})", too_long))
          .has_value());
}

TEST(LiteralJson, EchoArraysAlignWithHarnessAndVllmSuppliedTokenScores) {
  auto request = api::ParseCompletionRequest(
      R"({"model":"m","prompt":[1,2],"echo":true,"max_tokens":1,"logprobs":1,"prompt_logprobs":1})");
  ASSERT_TRUE(request.has_value());
  const api::TokenLogprob first{
      .id = 1, .token = "é", .logprob = std::nullopt, .top = {}, .text_offset = 0};
  const api::TokenLogprob supplied{.id = 2,
                                   .token = "x",
                                   .logprob = -7,
                                   .top = {{.id = 3, .token = "y", .logprob = -0.1, .rank = 1},
                                           {.id = 2, .token = "x", .logprob = -7, .rank = 8}},
                                   .text_offset = 1};
  auto generated = supplied;
  generated.text_offset = 2;
  const api::LiteralResult rows{.prompt_text = "éx",
                                .logprobs = {first, supplied, generated},
                                .prompt_logprobs = {first, supplied}};
  auto body = api::LiteralCompletionJson("cmpl-1", 5, *request, "x", rows, api::Finish::kLength,
                                         {.prompt_tokens = 2, .completion_tokens = 1});
  ASSERT_TRUE(body.has_value());
  auto doc = json::Parse(*body);
  ASSERT_TRUE(doc.has_value()) << *body;
  EXPECT_EQ(In(doc->root(), {"object"}).string(), "text_completion");
  const auto choice = In(doc->root(), {"choices"}).at(0);
  EXPECT_EQ(In(choice, {"text"}).string(), "éxx");
  const auto scores = In(choice, {"logprobs"});
  for (const std::string_view key : {"tokens", "token_logprobs", "top_logprobs", "text_offset"}) {
    EXPECT_EQ(In(scores, {key}).size(), 3U);
  }
  EXPECT_TRUE(In(scores, {"token_logprobs"}).at(0).is_null());
  // lm_eval reads [ctxlen:-1]: only supplied continuation token x, even
  // when x is outside top-1; the final generated row is separate.
  EXPECT_EQ(In(scores, {"token_logprobs"}).at(1).float64(), -7);
  EXPECT_EQ(In(In(scores, {"top_logprobs"}).at(1), {"y"}).float64(), -0.1);
  EXPECT_EQ(In(scores, {"text_offset"}).at(1).int64(), 1);
  const auto prompt = In(choice, {"prompt_logprobs"});
  EXPECT_TRUE(prompt.at(0).is_null());
  EXPECT_EQ(In(prompt.at(1), {"2", "logprob"}).float64(), -7);
  EXPECT_EQ(In(prompt.at(1), {"2", "rank"}).int64(), 8);
}

TEST(LiteralJson, PureScoringDoesNotRequireEchoOrLegacyLogprobs) {
  auto request = api::ParseCompletionRequest(
      R"({"model":"m","prompt":[1,2],"max_tokens":0,"prompt_logprobs":0})");
  ASSERT_TRUE(request.has_value());
  const api::LiteralResult rows{
      .prompt_text = "AB",
      .logprobs = {},
      .prompt_logprobs = {
          {.id = 1, .token = "A", .logprob = std::nullopt, .top = {}, .text_offset = 0},
          {.id = 2,
           .token = "B",
           .logprob = -2,
           .top = {{.id = 2, .token = "B", .logprob = -2, .rank = 2}},
           .text_offset = 1}}};
  auto body = api::LiteralCompletionJson("cmpl-0", 1, *request, "", rows, api::Finish::kStop,
                                         {.prompt_tokens = 2});
  ASSERT_TRUE(body.has_value());
  auto doc = json::Parse(*body);
  ASSERT_TRUE(doc.has_value());
  const auto choice = In(doc->root(), {"choices"}).at(0);
  EXPECT_EQ(In(choice, {"text"}).string(), "");
  EXPECT_TRUE(In(choice, {"logprobs"}).is_null());
  EXPECT_EQ(In(choice, {"prompt_logprobs"}).size(), 2U);
  EXPECT_EQ(In(doc->root(), {"usage", "completion_tokens"}).int64(), 0);
}

TEST(LiteralJson, RefusesOversizedOrNonfiniteScoresBeforeSerialization) {
  api::CompletionRequest request;
  request.options.model = "m";
  request.prompt = "x";
  request.logprobs = 1;
  api::LiteralResult result;
  result.logprobs.push_back({.id = 1,
                             .token = "x",
                             .logprob = std::numeric_limits<double>::infinity(),
                             .top = {},
                             .text_offset = 0});
  EXPECT_FALSE(
      api::LiteralCompletionJson("c", 0, request, "", result, api::Finish::kStop, {}).has_value());
  result.logprobs.front().logprob = -1;
  result.logprobs.front().token.resize((api::kMaxCompletionResponseBytes / 6) + 1, 'x');
  EXPECT_FALSE(
      api::LiteralCompletionJson("c", 0, request, "", result, api::Finish::kStop, {}).has_value());
}

jitllm::tokenizer::TokenizerSpec LiteralVocabulary(bool add_bos) {
  jitllm::tokenizer::TokenizerSpec spec;
  char32_t next = 256;
  for (unsigned byte = 0; byte < 256; ++byte) {
    const bool printable =
        (byte >= 0x21 && byte <= 0x7E) || (byte >= 0xA1 && byte <= 0xAC) || byte >= 0xAE;
    std::string text;
    jitllm::tokenizer::unicode::AppendUtf8(printable ? static_cast<char32_t>(byte) : next++, text);
    spec.tokens.push_back(std::move(text));
    spec.kinds.push_back(jitllm::tokenizer::TokenKind::kNormal);
  }
  spec.tokens.emplace_back("<bos>");
  spec.tokens.emplace_back("<eos>");
  spec.tokens.emplace_back("unused");
  spec.kinds.push_back(jitllm::tokenizer::TokenKind::kControl);
  spec.kinds.push_back(jitllm::tokenizer::TokenKind::kControl);
  spec.kinds.push_back(jitllm::tokenizer::TokenKind::kUnused);
  spec.bos = 256;
  spec.eos = 257;
  spec.add_bos = add_bos;
  return spec;
}

TEST(LiteralTokens, TextHonorsAddBosPolicyButIdsRemainExact) {
  auto with = jitllm::tokenizer::Tokenizer::Create(LiteralVocabulary(true));
  auto without = jitllm::tokenizer::Tokenizer::Create(LiteralVocabulary(false));
  ASSERT_TRUE(with.has_value());
  ASSERT_TRUE(without.has_value());
  auto request = api::ParseCompletionRequest(
      R"({"model":"m","prompt":"A","max_tokens":0,"echo":true,"logprobs":1})");
  ASSERT_TRUE(request.has_value());
  auto prepared = api::PrepareLiteralPrompt(*request, *with, 2);
  ASSERT_TRUE(prepared.has_value());
  EXPECT_THAT(prepared->tokens, ElementsAre(256, 65));
  EXPECT_TRUE(prepared->added_bos);
  EXPECT_EQ(prepared->text, "A");
  prepared = api::PrepareLiteralPrompt(*request, *without, 2);
  ASSERT_TRUE(prepared.has_value());
  EXPECT_THAT(prepared->tokens, ElementsAre(65));
  EXPECT_FALSE(prepared->added_bos);
  request->prompt = "<bos>A";
  prepared = api::PrepareLiteralPrompt(*request, *with, 4);
  ASSERT_TRUE(prepared.has_value());
  EXPECT_THAT(prepared->tokens, ElementsAre(256, 65));
  EXPECT_FALSE(prepared->added_bos);
  request->prompt.reset();
  request->token_ids = {257, 65};
  prepared = api::PrepareLiteralPrompt(*request, *with, 2);
  ASSERT_TRUE(prepared.has_value());
  EXPECT_THAT(prepared->tokens, ElementsAre(257, 65));  // supplied stop is not dropped
  EXPECT_FALSE(prepared->added_bos);
  request->options.max_tokens = 1;
  EXPECT_FALSE(api::PrepareLiteralPrompt(*request, *with, 2).has_value());
  request->options.max_tokens = 0;
  request->token_ids = {258};
  EXPECT_FALSE(api::PrepareLiteralPrompt(*request, *with, 2).has_value());
  request->token_ids = {259};
  EXPECT_FALSE(api::PrepareLiteralPrompt(*request, *with, 2).has_value());
}

TEST(LiteralTokens, EmptyTextRequiresEnabledBosAndNormalizationIsExplicit) {
  auto with = jitllm::tokenizer::Tokenizer::Create(LiteralVocabulary(true));
  auto spec = LiteralVocabulary(false);
  spec.normalization = jitllm::tokenizer::Normalization::kNfc;
  auto normalized = jitllm::tokenizer::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(with.has_value());
  ASSERT_TRUE(normalized.has_value());
  auto request = api::ParseCompletionRequest(
      R"({"model":"m","prompt":"","max_tokens":0,"prompt_logprobs":0})");
  ASSERT_TRUE(request.has_value());
  auto prepared = api::PrepareLiteralPrompt(*request, *with, 1);
  ASSERT_TRUE(prepared.has_value());
  EXPECT_THAT(prepared->tokens, ElementsAre(256));
  EXPECT_FALSE(api::PrepareLiteralPrompt(*request, *normalized, 1).has_value());
  request->add_special_tokens = false;
  EXPECT_FALSE(api::PrepareLiteralPrompt(*request, *with, 1).has_value());
  request->prompt = "e\xCC\x81";
  auto refused = api::PrepareLiteralPrompt(*request, *normalized, 8);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().code, "tokenization_changes_text");
}

TEST(LiteralTokens, ByteTokensShareOffsetsAndFlushAtThePromptBoundary) {
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(LiteralVocabulary(false));
  ASSERT_TRUE(tokenizer.has_value());
  std::size_t budget = 1024;
  api::LiteralRows rows(*tokenizer, false, true, 0, budget);
  std::vector<float> logits(tokenizer->size(), 0);
  auto lead = rows.Add(0xC3, {}, 1, true);
  ASSERT_TRUE(lead.has_value());
  EXPECT_EQ(lead->text_offset, 0U);
  auto tail = rows.Add(0xA9, logits, 1);
  ASSERT_TRUE(tail.has_value());
  EXPECT_EQ(tail->text_offset, 0U);
  EXPECT_EQ(rows.offset(), 1U);  // é is one decoded character, not two byte tokens
  ASSERT_TRUE(rows.Add(0xE2, logits, 1).has_value());
  EXPECT_EQ(rows.offset(), 1U);
  rows.Finish();
  EXPECT_EQ(rows.offset(), 2U);  // trailing partial byte becomes U+FFFD
  api::LiteralRows generated(*tokenizer, true, false, rows.offset(), budget);
  auto next = generated.Add(65, logits, 0);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next->text_offset, 2U);
  EXPECT_EQ(next->token, "token_id:65");
  EXPECT_EQ(generated.offset(), 3U);
  auto eos = generated.Add(257, logits, 0);
  ASSERT_TRUE(eos.has_value());
  EXPECT_EQ(eos->text_offset, 3U);
  EXPECT_EQ(generated.offset(), 3U);  // EOS has a score but no generated text
  budget = api::kMaxCompletionResponseBytes;
  EXPECT_FALSE(generated.Add(65, logits, 0).has_value());
  EXPECT_EQ(budget, api::kMaxCompletionResponseBytes);
}

TEST(LiteralTokens, RefusesScoreCountBeforeDecodingAndPreservesBpeIds) {
  auto spec = LiteralVocabulary(false);
  spec.tokens.emplace_back("ab");
  spec.kinds.push_back(jitllm::tokenizer::TokenKind::kNormal);
  spec.merges.emplace_back("a", "b");
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(std::move(spec));
  ASSERT_TRUE(tokenizer.has_value());
  auto request = api::ParseCompletionRequest(
      R"({"model":"m","prompt":"ab","max_tokens":0,"prompt_logprobs":1})");
  ASSERT_TRUE(request.has_value());
  auto merged = api::PrepareLiteralPrompt(*request, *tokenizer, 8);
  ASSERT_TRUE(merged.has_value());
  EXPECT_THAT(merged->tokens, ElementsAre(259));
  request->prompt.reset();
  request->token_ids = {97, 98};  // client-specified continuation boundary stays separate
  auto exact = api::PrepareLiteralPrompt(*request, *tokenizer, 8);
  ASSERT_TRUE(exact.has_value());
  EXPECT_THAT(exact->tokens, ElementsAre(97, 98));
  request->token_ids.assign(api::kMaxCompletionScoreRows + 1, 65);
  auto refused = api::PrepareLiteralPrompt(*request, *tokenizer, 262144);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().code, "score_limit_exceeded");
}

TEST(LiteralJson, ZeroTopCountKeepsActualTokenOnlyWhenOtherFormAskedForTopOne) {
  auto request = api::ParseCompletionRequest(
      R"({"model":"m","prompt":[1,2],"max_tokens":0,"echo":true,"logprobs":1,"prompt_logprobs":0})");
  ASSERT_TRUE(request.has_value());
  api::TokenLogprob row{.id = 2,
                        .token = "B",
                        .logprob = -2,
                        .top = {{.id = 3, .token = "C", .logprob = -0.1, .rank = 1},
                                {.id = 2, .token = "B", .logprob = -2, .rank = 2}},
                        .text_offset = 1};
  api::LiteralResult result{.prompt_text = "AB", .logprobs = {row}, .prompt_logprobs = {row}};
  auto body = api::LiteralCompletionJson("c", 0, *request, "", result, api::Finish::kStop, {});
  ASSERT_TRUE(body.has_value());
  auto doc = json::Parse(*body);
  ASSERT_TRUE(doc.has_value());
  const auto choice = In(doc->root(), {"choices"}).at(0);
  EXPECT_EQ(In(choice, {"logprobs", "top_logprobs"}).at(0).size(), 2U);
  EXPECT_EQ(In(choice, {"prompt_logprobs"}).at(0).size(), 1U);
  EXPECT_TRUE(In(choice, {"prompt_logprobs"}).at(0).find("2").has_value());
}

TEST(ApiJson, ShapesParseBack) {
  const api::Usage usage{.prompt_tokens = 7, .completion_tokens = 3, .cached_tokens = 2};
  const std::string completion = api::CompletionJson("chatcmpl-1", 5, "m", "a\"b", std::string("r"),
                                                     api::Finish::kLength, usage);
  auto doc = json::Parse(completion);
  ASSERT_TRUE(doc.has_value()) << completion;
  const json::Value choice = In(doc->root(), {"choices"}).at(0);
  EXPECT_EQ(In(choice, {"message", "content"}).string(), "a\"b");
  EXPECT_EQ(In(choice, {"message", "reasoning"}).string(), "r");
  EXPECT_EQ(In(choice, {"finish_reason"}).string(), "length");
  EXPECT_EQ(In(doc->root(), {"usage", "total_tokens"}).int64(), 10);
  EXPECT_EQ(In(doc->root(), {"usage", "prompt_tokens_details", "cached_tokens"}).int64(), 2);
  for (const std::string& chunk :
       {api::ChunkJson("i", 1, "m", api::Delta::kRole, {}),
        api::ChunkJson("i", 1, "m", api::Delta::kContent, "x\n"),
        api::ChunkJson("i", 1, "m", api::Delta::kFinish, {}, api::Finish::kStop),
        api::UsageChunkJson("i", 1, "m", usage), api::ModelsJson({{"a", true, 1}}, 1),
        api::ErrorJson({.status = 400, .type = "t", .message = "m", .param = {}, .code = "c"})}) {
    EXPECT_TRUE(json::Parse(chunk).has_value()) << chunk;
  }
  EXPECT_THAT(api::ChunkJson("i", 1, "m", api::Delta::kFinish, {}, api::Finish::kStop),
              HasSubstr(R"("delta":{},"logprobs":null,"finish_reason":"stop")"));
}

TEST(ApiHost, OnlyLoopbackNames) {
  for (const std::string_view ok : {"localhost", "localhost:8114", "127.0.0.1", "127.0.0.1:8114",
                                    "127.9.9.9:1", "[::1]", "[::1]:8114", "LocalHost:8114"}) {
    EXPECT_TRUE(api::IsLoopbackHost(ok)) << ok;
  }
  for (const std::string_view bad :
       {"", "example.com", "evil.localhost", "10.0.0.1:8114", "[::2]:8114", "[::1", "[::1]x",
        "localhost.:80", "localhost:", "localhost:80:80"}) {
    EXPECT_FALSE(api::IsLoopbackHost(bad)) << bad;
  }
}

// ---------------------------------------------------------------- binding

jitllm::platform::InterfaceAddress Address(std::string_view interface, std::string_view text,
                                           bool loopback = false) {
  jitllm::platform::InterfaceAddress a;
  a.interface = std::string(interface);
  a.ipv6 = text.contains(':');
  const std::string owned(text);
  EXPECT_EQ(::inet_pton(a.ipv6 ? AF_INET6 : AF_INET, owned.c_str(), a.bytes.data()), 1) << text;
  a.up = true;
  a.loopback = loopback;
  return a;
}

// spark-b's interfaces as `ip addr` showed them on 2026-09-28, shortened.
std::vector<jitllm::platform::InterfaceAddress> SparkAddresses(bool tailscale = true) {
  std::vector<jitllm::platform::InterfaceAddress> list = {
      Address("lo", "127.0.0.1", true),
      Address("lo", "::1", true),
      Address("enP7s7", "192.168.0.101"),
      Address("enP7s7", "fdd0:5b0c:6852:482c::1"),
      Address("enP7s7", "fe80::1864:c1f4:4501:7925"),
      Address("docker0", "172.17.0.1")};
  if (tailscale) {
    list.push_back(Address("tailscale0", "100.114.118.63"));
    list.push_back(Address("tailscale0", "fd7a:115c:a1e0::2e31:7640"));
    list.push_back(Address("tailscale0", "fe80::a007:23ed:e98f:b08e"));
  }
  return list;
}

// A resolver that knows the tailnet's names, as MagicDNS answers them.
std::optional<std::string> SparkReverse(const jitllm::platform::InterfaceAddress& a) {
  const std::string text = a.Text();
  if (text == "100.114.118.63" || text == "fd7a:115c:a1e0::2e31:7640") {
    return "spark-b.coati-puffin.ts.net";
  }
  if (text == "192.168.0.101") {
    return "spark-56f5.lan";
  }
  return std::nullopt;
}

std::vector<std::string> Endpoints(const api::Listening& l) {
  std::vector<std::string> out;
  out.reserve(l.endpoints.size());
  for (const auto& e : l.endpoints) {
    out.push_back(e.ipv6 ? std::format("[{}]:{}", e.address, e.port)
                         : std::format("{}:{}", e.address, e.port));
  }
  return out;
}

jitllm::config::ClientConfig Bind(const std::vector<std::string_view>& entries) {
  jitllm::config::ClientConfig client;
  client.bind.clear();
  for (const std::string_view e : entries) {
    auto entry = jitllm::config::ParseBindEntry(e);
    EXPECT_TRUE(entry.has_value()) << e;
    client.bind.push_back(entry.value_or(jitllm::config::BindEntry{}));
  }
  return client;
}

TEST(Binding, TheDefaultServesLoopbackAndTheTailnet) {
  const api::Listening l = api::ResolveListening(jitllm::config::ClientConfig{}, SparkAddresses(),
                                                 "spark-56f5", SparkReverse);
  EXPECT_THAT(Endpoints(l), ElementsAre("127.0.0.1:8114", "[::1]:8114", "100.114.118.63:8114",
                                        "[fd7a:115c:a1e0::2e31:7640]:8114"));
  EXPECT_THAT(l.unauthenticated, IsEmpty());
  EXPECT_THAT(l.notes, ElementsAre(HasSubstr("spark-b.coati-puffin.ts.net")));
  api::HostGuard hosts = l.hosts;
  for (const std::string_view ok :
       {"spark-b.coati-puffin.ts.net", "spark-b.coati-puffin.ts.net:8114", "Spark-B:8114",
        "spark-56f5", "100.114.118.63:8114", "[fd7a:115c:a1e0::2e31:7640]:8114", "localhost"}) {
    EXPECT_TRUE(hosts.AllowsHost(ok)) << ok;
  }
  for (const std::string_view bad :
       {"evil.coati-puffin.ts.net", "spark-b.coati-puffin.ts.net.", "coati-puffin.ts.net",
        "192.168.0.101:8114", "spark-56f5.lan", "100.114.118.64", "evil.example", "[fe80::1]"}) {
    EXPECT_FALSE(hosts.AllowsHost(bad)) << bad;
  }
}

TEST(Binding, WithoutTailscaleItServesLoopbackOnly) {
  auto addresses = SparkAddresses(false);
  // No ::1 either: an IPv6-less host.
  std::erase_if(addresses, [](const auto& a) { return a.ipv6; });
  const api::Listening l =
      api::ResolveListening(jitllm::config::ClientConfig{}, addresses, "spark-56f5", SparkReverse);
  EXPECT_THAT(Endpoints(l), ElementsAre("127.0.0.1:8114"));
  EXPECT_THAT(l.notes, ElementsAre(HasSubstr("no tailnet interface found")));
  EXPECT_THAT(l.unauthenticated, IsEmpty());
}

// 100.64.0.0/10 is also carriers' shared address space: only Tailscale's
// interface makes it the tailnet.
TEST(Binding, ACarriersSharedAddressIsNotTheTailnet) {
  std::vector<jitllm::platform::InterfaceAddress> addresses = {Address("lo", "127.0.0.1", true),
                                                               Address("wwan0", "100.72.1.2")};
  api::Listening l = api::ResolveListening(Bind({"tailscale"}), addresses, "h", nullptr);
  EXPECT_THAT(Endpoints(l), IsEmpty());
  // Tailscale's IPv6 range marks a renamed tunnel as the tailnet.
  addresses.push_back(Address("ts9", "100.100.1.2"));
  addresses.push_back(Address("ts9", "fd7a:115c:a1e0::9"));
  l = api::ResolveListening(Bind({"tailscale"}), addresses, "h", nullptr);
  EXPECT_THAT(Endpoints(l), ElementsAre("100.100.1.2:8114", "[fd7a:115c:a1e0::9]:8114"));
  EXPECT_THAT(l.notes, ElementsAre(HasSubstr("MagicDNS name is unknown")));
}

TEST(Binding, ExplicitAddressesAreUnauthenticatedUnlessLoopbackOrTailnet) {
  // A wildcard: every address of its family passes the Host check, it
  // covers loopback's on the same port, and the start log says it is
  // served without authentication.
  api::Listening l = api::ResolveListening(Bind({"loopback", "0.0.0.0", "tailscale"}),
                                           SparkAddresses(), "spark-56f5", SparkReverse);
  EXPECT_THAT(Endpoints(l),
              ElementsAre("[::1]:8114", "0.0.0.0:8114", "[fd7a:115c:a1e0::2e31:7640]:8114"));
  ASSERT_THAT(l.unauthenticated, ElementsAre(HasSubstr("serving without authentication")));
  EXPECT_THAT(l.unauthenticated[0], HasSubstr("0.0.0.0:8114"));
  EXPECT_TRUE(l.hosts.AllowsHost("192.168.0.101:8114"));
  EXPECT_TRUE(l.hosts.AllowsHost("172.17.0.1"));
  EXPECT_TRUE(l.hosts.AllowsHost("spark-56f5.lan"));
  EXPECT_TRUE(l.hosts.AllowsHost("spark-b"));
  EXPECT_FALSE(l.hosts.AllowsHost("[fdd0:5b0c:6852:482c::1]"));  // IPv6 is not wildcarded
  // A LAN address is said to be unauthenticated; a tailnet or loopback
  // address is not; a port of its own is kept, and [client] port fills in
  // the rest.
  jitllm::config::ClientConfig client =
      Bind({"192.168.0.101:9000", "100.114.118.63", "127.0.0.2:9001", "[::]:9002"});
  client.port = 9100;
  l = api::ResolveListening(client, SparkAddresses(), "spark-56f5", SparkReverse);
  EXPECT_THAT(Endpoints(l), ElementsAre("192.168.0.101:9000", "100.114.118.63:9100",
                                        "127.0.0.2:9001", "[::]:9002"));
  EXPECT_THAT(
      l.unauthenticated,
      ElementsAre(HasSubstr("on 192.168.0.101:9000, which is neither loopback nor the tailnet"),
                  HasSubstr("on [::]:9002 (every IPv6 interface)")));
  EXPECT_TRUE(l.hosts.AllowsHost("spark-56f5.lan:9000"));
  EXPECT_TRUE(l.hosts.AllowsHost("[fdd0:5b0c:6852:482c::1]:9002"));
  EXPECT_FALSE(l.hosts.AllowsHost("[fe80::1864:c1f4:4501:7925]"));  // link-local
  EXPECT_TRUE(l.hosts.AllowsHost("spark-b.coati-puffin.ts.net"));   // the tailnet address's
  // A tailnet address is named only under ts.net, bound by "tailscale" or
  // by its address: a resolver's other name for it is not trusted.
  const auto spoofed = [](const jitllm::platform::InterfaceAddress& a) {
    return api::InTailnetRange(a) ? std::optional<std::string>("evil.example") : SparkReverse(a);
  };
  for (const std::string_view entry : {"100.114.118.63", "tailscale"}) {
    l = api::ResolveListening(Bind({entry}), SparkAddresses(), "spark-56f5", spoofed);
    EXPECT_FALSE(l.hosts.AllowsHost("evil.example")) << entry;
    EXPECT_FALSE(l.hosts.AllowsHost("evil")) << entry;
    EXPECT_TRUE(l.hosts.AllowsHost("100.114.118.63:8114")) << entry;
  }
  // Reverse lookups are bounded.
  std::size_t lookups = 0;
  std::vector<jitllm::platform::InterfaceAddress> many = {Address("lo", "127.0.0.1", true)};
  for (int i = 1; i <= 40; ++i) {
    many.push_back(Address("eth0", std::format("10.0.0.{}", i)));
  }
  (void)api::ResolveListening(Bind({"0.0.0.0"}), many, "h",
                              [&](const jitllm::platform::InterfaceAddress&) {
                                ++lookups;
                                return std::optional<std::string>();
                              });
  EXPECT_EQ(lookups, api::kMaxReverseLookups);
}

TEST(Binding, OriginsMustBeTheNodeOnItsPort) {
  api::HostGuard hosts;
  hosts.AddName("spark-b.coati-puffin.ts.net");
  hosts.AddPort(8114);
  EXPECT_TRUE(hosts.AllowsOrigin("http://spark-b.coati-puffin.ts.net:8114"));
  EXPECT_TRUE(hosts.AllowsOrigin("HTTP://localhost:8114"));
  EXPECT_TRUE(hosts.AllowsOrigin("http://[::1]:8114"));
  // A scheme alone ("http", "https") once read past its end: the process
  // aborted.
  for (const std::string_view bad :
       {"http://spark-b.coati-puffin.ts.net", "https://spark-b.coati-puffin.ts.net:3000",
        "http://evil.example:8114", "null", "file://", "http://localhost:8114/",
        "ftp://localhost:8114", "http://127.0.0.1:3000", "http", "https", "HTTP", "http:", "http:/",
        "http://", "", "://localhost:8114"}) {
    EXPECT_FALSE(hosts.AllowsOrigin(bad)) << bad;
  }
  hosts.AddPort(443);
  EXPECT_TRUE(hosts.AllowsOrigin("https://spark-b.coati-puffin.ts.net"));
}

// ---------------------------------------------------------------- the server

// Runs requests as the test directs: "block" waits for release, making
// progress; "hang" waits for release making none (a hung unit), then asks
// to go on; "slow" makes progress every 50 ms for 1.5 s; "wide" runs one
// prefill chunk of 100 rows (1 s at the default floor) in 1 s without a
// beat; "fail" fails before admission, "late" after it; "flood" streams
// until told to stop; "big" answers 16 MiB at once; otherwise, and after
// "slow" and "wide", reasoning, then the last message's content echoed in
// two pieces. Each is admitted with `admission`.
class FakeBackend final : public api::Backend {
 public:
  std::vector<api::ModelInfo> Models() const override {
    return {{.name = "alpha", .chat = true, .context = 100},
            {.name = "image", .chat = false, .context = 0}};
  }
  std::expected<api::Completion, api::Error> Complete(const api::ChatRequest& request,
                                                      api::Exchange& exchange) override {
    ++chat_calls;
    const std::string& text = request.messages.back().content;
    if (text == "fail") {
      return std::unexpected(api::Error{.status = 400,
                                        .type = "invalid_request_error",
                                        .message = "too long",
                                        .param = "messages",
                                        .code = "context_length_exceeded"});
    }
    if (!exchange.Admit(admission)) {
      return api::Completion{};
    }
    if (text == "late") {
      return std::unexpected(api::Error{
          .status = 500, .type = "server_error", .message = "failed", .param = {}, .code = {}});
    }
    if (text == "block") {
      started.store(true);
      while (!release.load()) {
        if (!exchange.Continue()) {
          cancelled.store(true);
          return api::Completion{.completion_tokens = 1, .cached_tokens = 0, .stopped = false};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
    if (text == "hang") {
      started.store(true);
      while (!release.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      if (!exchange.Continue()) {
        cancelled.store(true);
        return api::Completion{.completion_tokens = 1, .cached_tokens = 0, .stopped = false};
      }
    }
    if (text == "slow") {
      for (int i = 0; i < 30; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!exchange.Next(jitllm::runtime::Phase::kPrefill, 1)) {
          cancelled.store(true);
          return api::Completion{};
        }
      }
    }
    if (text == "wide") {
      if (!exchange.Next(jitllm::runtime::Phase::kPrefill, 100)) {
        cancelled.store(true);
        return api::Completion{};
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (text == "flood") {
      started.store(true);
      const std::string piece(1024, 'x');
      for (int i = 0; i < 200000; ++i) {
        if (!exchange.Content(piece)) {
          flooded.store(true);
          break;
        }
      }
      return api::Completion{.completion_tokens = 1, .cached_tokens = 0, .stopped = false};
    }
    if (text == "big") {
      (void)exchange.Content(std::string(std::size_t{16} << 20U, 'y'));
      return api::Completion{.completion_tokens = 1, .cached_tokens = 0, .stopped = true};
    }
    (void)exchange.Reasoning("hmm");
    const std::size_t half = text.size() / 2;
    const bool go = exchange.Content(text.substr(0, half)) && exchange.Content(text.substr(half));
    return api::Completion{.completion_tokens = go ? 4U : 3U, .cached_tokens = 2, .stopped = go};
  }

  std::expected<api::Completion, api::Error> Complete(const api::CompletionRequest& request,
                                                      api::Exchange& exchange) override {
    ++literal_calls;
    if (std::ranges::any_of(request.token_ids, [](std::int32_t id) { return id >= 10; })) {
      return std::unexpected(api::Error{.status = 400,
                                        .type = "invalid_request_error",
                                        .message = "outside vocabulary",
                                        .param = "prompt",
                                        .code = "invalid_token_id"});
    }
    const auto prompt = static_cast<std::uint32_t>(request.prompt ? request.prompt->size()
                                                                  : request.token_ids.size());
    const auto max_tokens = request.options.max_tokens.value_or(16);
    if (!exchange.Admit(
            {.prompt_tokens = prompt, .max_tokens = max_tokens, .swap_bytes = 0, .floors = {}})) {
      return api::Completion{};
    }
    if (request.prompt == "block") {
      started.store(true);
      while (!release.load()) {
        if (!exchange.Next(jitllm::runtime::Phase::kPrefill, 1)) {
          cancelled.store(true);
          return api::Completion{};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
    api::Completion result;
    result.literal.prompt_text = request.prompt.value_or(std::string(prompt, 'A'));
    if (request.prompt == "huge") {
      result.literal.prompt_text.assign(std::size_t{4} << 20U, 'x');
    }
    for (std::uint32_t i = 0; i < prompt; ++i) {
      api::TokenLogprob row{
          .id = 1, .token = "A", .logprob = std::nullopt, .top = {}, .text_offset = i};
      if (i != 0) {
        row.logprob = -2;
        row.top = {{.id = 2, .token = "B", .logprob = -0.1, .rank = 1},
                   {.id = 1, .token = "A", .logprob = -2, .rank = 2}};
      }
      if (request.echo && request.logprobs) {
        result.literal.logprobs.push_back(row);
      }
      if (request.prompt_logprobs) {
        result.literal.prompt_logprobs.push_back(std::move(row));
      }
    }
    if (max_tokens != 0) {
      (void)exchange.Content("B");
      result.completion_tokens = 1;
      if (request.logprobs) {
        result.literal.logprobs.push_back(
            {.id = 2,
             .token = "B",
             .logprob = -0.1,
             .top = {{.id = 2, .token = "B", .logprob = -0.1, .rank = 1}},
             .text_offset = request.echo ? prompt : 0});
      }
    }
    result.stopped = max_tokens == 0;
    return result;
  }

  // Set before the server starts.
  api::Exchange::Admission admission{
      .prompt_tokens = 10, .max_tokens = 1000, .swap_bytes = 0, .floors = {}};
  std::atomic<bool> started{false};
  std::atomic<bool> release{false};
  std::atomic<bool> cancelled{false};
  std::atomic<bool> flooded{false};
  std::atomic<unsigned> literal_calls{0};
  std::atomic<unsigned> chat_calls{0};
  api::CooperativeBackend* cooperative() override { return cooperative_backend; }
  api::CooperativeBackend* cooperative_backend = nullptr;
};

// Native-unit boundaries are explicit gates. Tests can block a wave or its
// retirement while the transport continues to observe channels and health.
class FakeCooperative final : public api::CooperativeBackend {
 public:
  class Job final : public Work {
   public:
    Job(FakeCooperative& backend, const Request& request, api::Exchange& into)
        : owner(backend),
          request(request),
          exchange(into),
          limit(request.options.max_tokens.value_or(4)) {
      ++owner.live;
    }
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;
    Job(Job&&) = delete;
    Job& operator=(Job&&) = delete;
    ~Job() override {
      if (!retired) {
        owner.early_destruction.store(true);
      }
      ++owner.destroyed;
      --owner.live;
    }
    bool terminal() const override { return cancelled || ticks >= limit; }
    void Cancel() override {
      if (!cancelled) {
        cancelled = true;
        ++owner.cancelled;
      }
    }
    FakeCooperative& owner;
    const Request& request;  // borrows the descriptor itself through retirement
    api::Exchange& exchange;
    std::uint32_t limit = 0;
    std::uint32_t ticks = 0;
    bool cancelled = false;
    bool retired = false;
  };

  bool Supports(const Request& request) const override {
    return request.options.model == "alpha" &&
           (request.literal != nullptr || request.options.messages.back().content != "serial");
  }
  std::expected<std::unique_ptr<Work>, api::Error> Start(const Request& request,
                                                         api::Exchange& exchange) override {
    if (live.load() >= capacity.load()) {
      ++capacity_deferrals;
      return std::unique_ptr<Work>{};  // No owner, Admit, output or borrowed descriptor.
    }
    if (request.literal == nullptr && request.options.messages.back().content == "bad") {
      ++rejections;
      return std::unexpected(api::Error{.status = 400,
                                        .type = "invalid_request_error",
                                        .message = "injected request refusal",
                                        .param = "messages",
                                        .code = {}});
    }
    if (request.literal == nullptr && request.options.messages.back().content == "defer") {
      ++deferrals;
      return std::unique_ptr<Work>{};
    }
    if (request.literal == nullptr && request.options.messages.back().content == "later") {
      later_after_serial.store(serial_calls != nullptr && serial_calls->load() != 0);
    }
    auto work = std::make_unique<Job>(*this, request, exchange);
    if (!exchange.Admit({.prompt_tokens = request.literal != nullptr ? 4U : 10U,
                         .max_tokens = work->limit,
                         .swap_bytes = 0,
                         .floors = {.prefill = 1000, .decode = 1000}})) {
      work->Cancel();
    }
    ++started;
    return std::unique_ptr<Work>(std::move(work));
  }
  std::expected<Unit, std::string> NextUnit(std::span<Work* const> work) override {
    if (work.size() > peak.load()) {
      peak.store(static_cast<unsigned>(work.size()));
    }
    return Unit{.phase = jitllm::runtime::Phase::kDecode, .expected_seconds = 0};
  }
  std::expected<void, std::string> Advance(std::span<Work* const> work) override {
    const unsigned now = rejections.load();
    const unsigned since_last = now - previous_rejections_;
    previous_rejections_ = now;
    if (since_last > rejections_between_units.load()) {
      rejections_between_units.store(since_last);
    }
    ++advances;
    while (pause_units.load() && !release.load()) {
      for (Work* item : work) {
        auto& job = static_cast<Job&>(*item);
        (void)job.exchange.Continue();  // channel polling must not defeat the watchdog
        ++polls;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (fail_unit.load()) {
      return std::unexpected("injected native completion failure");
    }
    for (Work* item : work) {
      auto& job = static_cast<Job&>(*item);
      if (job.request.options.model != "alpha" ||
          job.request.options.max_tokens.value_or(4) != job.limit) {
        return std::unexpected("the borrowed request changed during native work");
      }
      if (job.terminal()) {
        continue;
      }
      if (!job.exchange.Content("x")) {
        job.Cancel();
      }
      ++job.ticks;
    }
    return {};
  }
  Retirement Retire(Work& work) override {
    auto& job = static_cast<Job&>(work);
    retirement_entered.store(true);
    while (pause_retirement.load() && !release_retirement.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    api::Completion result{
        .completion_tokens = job.ticks, .cached_tokens = 0, .stopped = job.ticks >= job.limit};
    if (job.request.literal != nullptr) {
      result.literal.prompt_text = job.request.literal->prompt.value_or("");
      if (job.request.literal->prompt == "huge") {
        result.literal.prompt_text.assign(std::size_t{4} << 20U, 'x');
      }
    }
    job.retired = true;
    ++retired;
    return {.result = std::move(result), .references_retired = true};
  }

  std::atomic<unsigned> started{0}, advances{0}, peak{0}, polls{0}, live{0};
  std::atomic<unsigned> capacity{api::kMaxActiveRequests}, capacity_deferrals{0};
  std::atomic<unsigned> cancelled{0}, retired{0}, destroyed{0}, deferrals{0};
  std::atomic<unsigned> rejections{0}, rejections_between_units{0};
  std::atomic<bool> pause_units{false}, release{false}, fail_unit{false};
  std::atomic<bool> pause_retirement{false}, release_retirement{false};
  std::atomic<bool> retirement_entered{false}, early_destruction{false};
  std::atomic<bool> later_after_serial{false};
  const std::atomic<unsigned>* serial_calls = nullptr;

 private:
  unsigned previous_rejections_ = 0;  // driver-owned
};

// A client socket's reads, with a timeout: false at the end or on none.
bool Recv(int fd, std::string& into) {
  std::array<char, 16384> buf{};
  const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
  if (n <= 0) {
    return false;
  }
  into.append(buf.data(), static_cast<std::size_t>(n));
  return true;
}

std::size_t Number(std::string_view text, int base) {
  std::size_t n = 0;
  (void)std::from_chars(text.data(), text.data() + text.size(), n, base);
  return n;
}

// One response from a connection that may persist: its head and its body,
// decoded (by Content-Length, chunks, or to the close); what follows stays
// in `pending`. What a connection sent before closing, if no head.
std::string ReadResponse(int fd, std::string& pending) {
  std::size_t head_end = 0;
  while ((head_end = pending.find("\r\n\r\n")) == std::string::npos) {
    if (!Recv(fd, pending)) {
      return std::exchange(pending, {});
    }
  }
  head_end += 4;
  const std::string head = pending.substr(0, head_end);
  std::string body;
  std::size_t used = 0;
  if (const std::size_t field = head.find("Content-Length: "); field != std::string::npos) {
    const std::size_t length = Number(std::string_view(head).substr(field + 16), 10);
    while (pending.size() < head_end + length && Recv(fd, pending)) {
    }
    body = pending.substr(head_end, length);
    used = std::min(pending.size(), head_end + length);
  } else if (head.contains("Transfer-Encoding: chunked\r\n")) {
    std::size_t at = head_end;
    for (;;) {
      std::size_t eol = 0;
      while ((eol = pending.find("\r\n", at)) == std::string::npos) {
        if (!Recv(fd, pending)) {
          pending.clear();
          return head + body;
        }
      }
      const std::size_t size = Number(std::string_view(pending).substr(at, eol - at), 16);
      if (size == 0) {
        while (pending.size() < eol + 4 && Recv(fd, pending)) {
        }
        used = std::min(pending.size(), eol + 4);
        break;
      }
      while (pending.size() < eol + 2 + size + 2 && Recv(fd, pending)) {
      }
      body += pending.substr(eol + 2, size);
      at = eol + 2 + size + 2;
      if (pending.size() < at) {
        pending.clear();
        return head + body;
      }
    }
  } else {
    while (Recv(fd, pending)) {
    }
    body = pending.substr(head_end);
    used = pending.size();
  }
  pending.erase(0, used);
  return head + body;
}

// Reads until `needle` has arrived (or the connection ends); all of it.
std::string ReadUntil(int fd, std::string& pending, std::string_view needle) {
  while (!pending.contains(needle) && Recv(fd, pending)) {
  }
  return pending;
}

// Whether the server has closed the connection (after what it sent).
bool Closed(int fd) {
  std::string ignored;
  while (Recv(fd, ignored)) {
  }
  const ssize_t n = ::recv(fd, ignored.data(), 0, MSG_DONTWAIT);
  return n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK);
}

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override { Start({}); }

  void Start(const api::ServerOptions& overrides) {
    api::ServerOptions options = overrides;
    options.bind = {{.address = "127.0.0.1", .ipv6 = false, .port = 0}};
    server_.emplace(backend_, options);
    auto ports = server_->Listen();
    ASSERT_TRUE(ports.has_value()) << ports.error();
    port_ = ports->front();
    wake_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    thread_ = std::jthread([this] {
      result_ = server_->Run(wake_, [this] {
        std::uint64_t n = 0;
        (void)!::read(wake_, &n, sizeof n);
        return true;
      });
    });
  }

  void TearDown() override { Stop(); }

  void Stop() {
    if (thread_.joinable()) {
      backend_.release.store(true);
      if (cooperative_) {
        cooperative_->release.store(true);
        cooperative_->release_retirement.store(true);
      }
      const std::uint64_t one = 1;
      (void)!::write(wake_, &one, sizeof one);
      thread_.join();
      (void)::close(wake_);
    }
  }

  void StartCooperative(const api::ServerOptions& options = {}) {
    Stop();
    cooperative_ = std::make_unique<FakeCooperative>();
    cooperative_->serial_calls = &backend_.chat_calls;
    backend_.cooperative_backend = cooperative_.get();
    Start(options);
  }

  static bool WaitFor(const std::function<bool()>& predicate) {
    for (int i = 0; i < 2000 && !predicate(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return predicate();
  }

  // A connection, with a receive timeout.
  int Open(int receive_buffer = 0) const {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
      ADD_FAILURE() << "socket: " << errno;
      return fd;
    }
    if (receive_buffer > 0) {
      (void)::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer, sizeof receive_buffer);
    }
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(port_);
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the sockets API
    EXPECT_EQ(::connect(fd, reinterpret_cast<const sockaddr*>(&to), sizeof to), 0);
    timeval tv{.tv_sec = 20, .tv_usec = 0};
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    return fd;
  }

  // A connection that has sent `bytes`.
  int Connect(std::string_view bytes) const {
    const int fd = Open();
    EXPECT_TRUE(jitllm::runtime::http::WriteAll(fd, bytes));
    return fd;
  }

  // One request's response on a connection of its own.
  std::string Exchange(std::string_view bytes) const {
    const int fd = Connect(bytes);
    std::string pending;
    std::string response = ReadResponse(fd, pending);
    (void)::close(fd);
    return response;
  }

  static std::string Post(std::string_view body, std::string_view extra = "") {
    return std::format(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: "
        "application/json\r\n{}Content-Length: {}\r\n\r\n{}",
        extra, body.size(), body);
  }

  static std::string Literal(std::string_view body) {
    return std::format(
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.size(), body);
  }

  static std::string Chat(std::string_view text, std::string_view fields = "",
                          std::string_view model = "alpha") {
    return std::format(R"({{"model":"{}","messages":[{{"role":"user","content":"{}"}}]{}}})", model,
                       text, fields);
  }

  static std::string BodyOf(const std::string& response) {
    const std::size_t at = response.find("\r\n\r\n");
    return at == std::string::npos ? std::string() : response.substr(at + 4);
  }

  // The backend's health as the server sees it (a failure without a server).
  jitllm::runtime::Health BackendHealth() const {
    if (!server_.has_value()) {
      ADD_FAILURE() << "no server";
      return {};
    }
    return server_->health();
  }

  void WaitStarted() const {
    for (int i = 0; i < 2000 && !backend_.started.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(backend_.started.load());
  }

  FakeBackend backend_;
  std::unique_ptr<FakeCooperative> cooperative_;
  std::optional<api::Server> server_;
  std::uint16_t port_ = 0;
  int wake_ = -1;
  std::expected<void, std::string> result_;
  std::jthread thread_;
};

TEST_F(ServerTest, AnswersAChatCompletion) {
  const std::string response = Exchange(Post(Chat("Hello world")));
  EXPECT_THAT(response, StartsWith("HTTP/1.1 200 OK\r\n"));
  EXPECT_THAT(response, HasSubstr("Content-Type: application/json\r\n"));
  EXPECT_THAT(response, HasSubstr("Connection: keep-alive\r\nKeep-Alive: timeout=60\r\n"));
  auto doc = json::Parse(BodyOf(response));
  ASSERT_TRUE(doc.has_value()) << response;
  const json::Value root = doc->root();
  EXPECT_THAT(std::string(In(root, {"id"}).string()), StartsWith("chatcmpl-"));
  EXPECT_EQ(In(root, {"model"}).string(), "alpha");
  const json::Value message = In(In(root, {"choices"}).at(0), {"message"});
  EXPECT_EQ(In(message, {"content"}).string(), "Hello world");
  EXPECT_EQ(In(message, {"reasoning"}).string(), "hmm");
  EXPECT_EQ(In(In(root, {"choices"}).at(0), {"finish_reason"}).string(), "stop");
  EXPECT_EQ(In(root, {"usage", "prompt_tokens"}).int64(), 10);
  EXPECT_EQ(In(root, {"usage", "completion_tokens"}).int64(), 4);
}

TEST_F(ServerTest, ServesLegacyEchoAndVllmPureScoringOnTheLiteralRoute) {
  auto response = Exchange(
      Literal(R"({"model":"alpha","prompt":[1,2],"max_tokens":1,"echo":true,"logprobs":1})"));
  EXPECT_THAT(response, StartsWith("HTTP/1.1 200 OK\r\n"));
  EXPECT_THAT(response, HasSubstr("jitllm-inference-version: 1\r\n"));
  auto doc = json::Parse(BodyOf(response));
  ASSERT_TRUE(doc.has_value());
  EXPECT_THAT(std::string(In(doc->root(), {"id"}).string()), StartsWith("cmpl-"));
  EXPECT_EQ(In(doc->root(), {"object"}).string(), "text_completion");
  const auto choice = In(doc->root(), {"choices"}).at(0);
  EXPECT_EQ(In(choice, {"text"}).string(), "AAB");
  EXPECT_EQ(In(choice, {"logprobs", "token_logprobs"}).size(), 3U);
  response = Exchange(Literal(
      R"({"model":"alpha","prompt":[1,2],"max_tokens":0,"echo":false,"prompt_logprobs":0})"));
  doc = json::Parse(BodyOf(response));
  ASSERT_TRUE(doc.has_value());
  const auto scored = In(doc->root(), {"choices"}).at(0);
  EXPECT_EQ(In(scored, {"text"}).string(), "");
  EXPECT_TRUE(In(scored, {"logprobs"}).is_null());
  EXPECT_EQ(In(scored, {"prompt_logprobs"}).size(), 2U);
  EXPECT_EQ(In(doc->root(), {"usage", "completion_tokens"}).int64(), 0);
  EXPECT_EQ(In(scored, {"finish_reason"}).string(), "stop");
  EXPECT_EQ(backend_.literal_calls, 2U);
}

TEST_F(ServerTest, LiteralParserRefusesBeforeBackendAndAdmissionChecksVocabulary) {
  for (const std::string_view body :
       {R"({"model":"alpha","prompt":[-1]})", R"({"model":"alpha","prompt":[true]})",
        R"({"model":"alpha","prompt":[[1]]})", R"({"model":"alpha","prompt":"x","stream":true})",
        R"({"model":"image","prompt":"x"})"}) {
    EXPECT_THAT(Exchange(Literal(body)), StartsWith("HTTP/1.1 400 "));
  }
  EXPECT_EQ(backend_.literal_calls, 0U);
  EXPECT_THAT(Exchange(Literal(R"({"model":"alpha","prompt":[10]})")),
              AllOf(StartsWith("HTTP/1.1 400 "), HasSubstr("invalid_token_id")));
  EXPECT_EQ(backend_.literal_calls, 1U);
}

TEST_F(ServerTest, LiteralScoringStopsWhenTheClientDisconnects) {
  const int fd =
      Connect(Literal(R"({"model":"alpha","prompt":"block","max_tokens":0,"prompt_logprobs":1})"));
  WaitStarted();
  linger reset{.l_onoff = 1, .l_linger = 0};
  (void)::setsockopt(fd, SOL_SOCKET, SO_LINGER, &reset, sizeof reset);
  (void)::close(fd);
  for (int i = 0; i < 2000 && !backend_.cancelled.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(backend_.cancelled.load());
}

TEST_F(ServerTest, LiteralResponseBudgetFollowsSlowBuffersThroughSendAndDrop) {
  constexpr std::string_view body =
      R"({"model":"alpha","prompt":"huge","max_tokens":0,"echo":true})";
  auto request = api::ParseCompletionRequest(body);
  ASSERT_TRUE(request.has_value());
  // Measure the allocator's charge for this exact response shape rather
  // than assuming a particular string growth factor in the test.
  api::LiteralResult large;
  large.prompt_text.assign(std::size_t{4} << 20U, 'x');
  auto serialized =
      api::LiteralCompletionJson("cmpl-000000000000000000000000", 1000000000, *request, "", large,
                                 api::Finish::kStop, {.prompt_tokens = 4});
  ASSERT_TRUE(serialized.has_value());
  api::ServerOptions options;
  options.response_budget = serialized->capacity() + (serialized->capacity() / 2);
  options.write_timeout = std::chrono::seconds(5);
  Stop();
  Start(options);
  if (!server_.has_value()) {
    ADD_FAILURE() << "the response-budget server did not start";
    return;
  }
  auto& server = *server_;
  jitllm::runtime::http::Fd slow(Open(4096));
  ASSERT_TRUE(jitllm::runtime::http::WriteAll(slow.get(), Literal(body)));
  const auto wait_for_charge = [&](bool charged) {
    for (int i = 0; i < 1000 && (server.response_bytes() != 0) != charged; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return (server.response_bytes() != 0) == charged;
  };
  ASSERT_TRUE(wait_for_charge(true));
  EXPECT_LE(server.response_bytes(), options.response_budget);
  jitllm::runtime::http::Fd second(Open(4096));
  ASSERT_TRUE(jitllm::runtime::http::WriteAll(second.get(), Literal(body)));
  std::string pending;
  EXPECT_THAT(ReadResponse(second.get(), pending),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("response_budget_exceeded")));
  EXPECT_LE(server.response_bytes(), options.response_budget);
  // A reset releases a partially sent allocation, allowing the next large
  // response; draining that response releases its charge as well.
  linger reset{.l_onoff = 1, .l_linger = 0};
  (void)::setsockopt(slow.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof reset);
  slow = jitllm::runtime::http::Fd();
  ASSERT_TRUE(wait_for_charge(false));
  EXPECT_THAT(Exchange(Literal(body)), StartsWith("HTTP/1.1 200 "));
  ASSERT_TRUE(wait_for_charge(false));
}

TEST_F(ServerTest, StreamsChunksThenDone) {
  const std::string response = Exchange(
      Post(Chat("Hello world", R"(,"stream":true,"stream_options":{"include_usage":true})")));
  EXPECT_THAT(response, StartsWith("HTTP/1.1 200 OK\r\n"));
  EXPECT_THAT(response, HasSubstr("Content-Type: text/event-stream\r\n"));
  EXPECT_THAT(response, HasSubstr("Transfer-Encoding: chunked\r\n"));
  const std::string body = BodyOf(response);
  EXPECT_THAT(body, StartsWith("data: {"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"role":"assistant","content":""})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"reasoning":"hmm"})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"content":"Hello"})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"content":" world"})"));
  EXPECT_THAT(body, HasSubstr(R"("finish_reason":"stop")"));
  EXPECT_THAT(body, HasSubstr(R"("choices":[],"usage":{"prompt_tokens":10)"));
  EXPECT_TRUE(body.ends_with("data: [DONE]\n\n")) << body;
  // HTTP/1.0 has no chunks: the stream ends with the connection.
  const std::string old = Exchange(
      std::format("POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\nContent-Type: "
                  "application/json\r\nContent-Length: {}\r\n\r\n{}",
                  Chat("Hi", R"(,"stream":true)").size(), Chat("Hi", R"(,"stream":true)")));
  EXPECT_THAT(old, AllOf(HasSubstr("Connection: close\r\n"), Not(HasSubstr("chunked"))));
  EXPECT_TRUE(BodyOf(old).ends_with("data: [DONE]\n\n")) << old;
}

TEST_F(ServerTest, StopStringsEndTheAnswer) {
  const std::string response = Exchange(Post(Chat("Hello world", R"(,"stop":"lo w")")));
  auto doc = json::Parse(BodyOf(response));
  ASSERT_TRUE(doc.has_value()) << response;
  const json::Value choice = In(doc->root(), {"choices"}).at(0);
  EXPECT_EQ(In(choice, {"message", "content"}).string(), "Hel");
  EXPECT_EQ(In(choice, {"finish_reason"}).string(), "stop");
}

TEST_F(ServerTest, ListsModels) {
  const std::string list = Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost:8114\r\n\r\n");
  EXPECT_THAT(list, StartsWith("HTTP/1.1 200 OK\r\n"));
  EXPECT_THAT(BodyOf(list), HasSubstr(R"({"object":"list","data":[{"id":"alpha")"));
  EXPECT_THAT(Exchange("GET /v1/models/image HTTP/1.1\r\nHost: [::1]\r\n\r\n"),
              HasSubstr(R"({"id":"image","object":"model")"));
  EXPECT_THAT(Exchange("GET /v1/models/beta HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"),
              AllOf(StartsWith("HTTP/1.1 404 "), HasSubstr("model_not_found")));
}

TEST_F(ServerTest, RefusesBeforeAnyWork) {
  // Model problems: unknown, or not a chat model; and the backend's own
  // refusal before admission.
  EXPECT_THAT(Exchange(Post(Chat("x", "", "beta"))),
              AllOf(StartsWith("HTTP/1.1 404 "), HasSubstr(R"("code":"model_not_found")")));
  EXPECT_THAT(Exchange(Post(Chat("x", "", "image"))),
              AllOf(StartsWith("HTTP/1.1 400 "), HasSubstr(R"("code":"model_not_supported")")));
  EXPECT_THAT(Exchange(Post(Chat("fail"))),
              AllOf(StartsWith("HTTP/1.1 400 "), HasSubstr("context_length_exceeded")));
  EXPECT_THAT(Exchange(Post(Chat("x", R"(,"temperature":3)"))),
              AllOf(StartsWith("HTTP/1.1 400 "), HasSubstr(R"("param":"temperature")")));
  // Routes and methods.
  EXPECT_THAT(Exchange("GET /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              AllOf(StartsWith("HTTP/1.1 405 "), HasSubstr("Allow: POST\r\n")));
  EXPECT_THAT(Exchange("GET /v1/other HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 404 "));
  // Browser guards.
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: evil.example\r\n\r\n"),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\n\r\n"), StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(Post(Chat("x"), "Origin: http://127.0.0.1:3000\r\n")),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(Post(Chat("x"), "Origin: null\r\n")), StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(Post(Chat("x"), "Sec-Fetch-Site: cross-site\r\n")),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(Post(Chat("x"), "Sec-Fetch-Site: same-site\r\n")),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                   "Content-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                                   Chat("x").size(), Chat("x"))),
              StartsWith("HTTP/1.1 415 "));
  // The same origin (a page this listener served) passes.
  EXPECT_THAT(Exchange(Post(Chat("x"), std::format("Origin: http://127.0.0.1:{}\r\n", port_))),
              StartsWith("HTTP/1.1 200 "));
}

TEST_F(ServerTest, BoundsTheHttpRequest) {
  // The body's size, before it is read.
  EXPECT_THAT(Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                   "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                                   api::kMaxBodyBytes + 1)),
              StartsWith("HTTP/1.1 413 "));
  // The head's size and header count.
  EXPECT_THAT(Exchange(std::format("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX: {}\r\n\r\n",
                                   std::string(api::kMaxHeaderBytes, 'x'))),
              StartsWith("HTTP/1.1 413 "));
  std::string many = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n";
  for (std::size_t i = 0; i < api::kMaxHeaders; ++i) {
    many += std::format("X-{}: y\r\n", i);
  }
  EXPECT_THAT(Exchange(many + "\r\n"), StartsWith("HTTP/1.1 413 "));
  // The target's length.
  EXPECT_THAT(Exchange(std::format("GET /{} HTTP/1.1\r\nHost: localhost\r\n\r\n",
                                   std::string(api::kMaxTargetBytes, 'a'))),
              StartsWith("HTTP/1.1 414 "));
  // Framing: chunked bodies, a missing length, bare LF, folding, versions.
  EXPECT_THAT(Exchange("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                       "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"),
              StartsWith("HTTP/1.1 501 "));
  EXPECT_THAT(Exchange("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 411 "));
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\nHost: localhost\n\n\r\n\r\n"),
              StartsWith("HTTP/1.1 400 "));
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n folded\r\n\r\n"),
              StartsWith("HTTP/1.1 400 "));
  EXPECT_THAT(Exchange("GET /v1/models HTTP/2.0\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 505 "));
  EXPECT_THAT(Exchange("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                       "Content-Length: 1\r\nContent-Length: 2\r\n\r\nx"),
              StartsWith("HTTP/1.1 400 "));
  // Content-Length's forms: only decimal digits, at most 18 of them.
  for (const std::string_view bad :
       {"+5", "-1", "5, 5", "0x5", "5 5", "", "99999999999999999999"}) {
    EXPECT_THAT(Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                     "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                                     bad)),
                StartsWith("HTTP/1.1 400 "))
        << bad;
  }
  // A length and a chunked encoding together (smuggling's shape).
  EXPECT_THAT(Exchange("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                       "Content-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"),
              StartsWith("HTTP/1.1 501 "));
  // Two Hosts, and a control byte in a value.
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nHost: evil.example\r\n\r\n"),
              StartsWith("HTTP/1.1 400 "));
  using namespace std::string_view_literals;
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX: a\0b\r\n\r\n"sv),
              StartsWith("HTTP/1.1 400 "));
  // A refusal of the head closes the connection: where the next request
  // would begin is unknown.
  EXPECT_THAT(Exchange("GET /v1/models HTTP/2.0\r\nHost: localhost\r\n\r\n"),
              HasSubstr("Connection: close\r\n"));
  // Expect: 100-continue is answered before the body.
  const std::string body = Chat("Hi there");
  const int fd = Connect(std::format(
      "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
      "Expect: 100-continue\r\nContent-Length: {}\r\n\r\n",
      body.size()));
  std::array<char, 64> interim{};
  const ssize_t got = ::recv(fd, interim.data(), interim.size(), 0);
  ASSERT_GT(got, 0);
  EXPECT_EQ(std::string_view(interim.data(), static_cast<std::size_t>(got)),
            "HTTP/1.1 100 Continue\r\n\r\n");
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(fd, body));
  std::string pending;
  EXPECT_THAT(ReadResponse(fd, pending), StartsWith("HTTP/1.1 200 OK"));
  (void)::close(fd);
}

// Several requests on one connection, refusals included; Connection:
// close and HTTP/1.0 end it.
TEST_F(ServerTest, KeepsAConnectionAlive) {
  const int fd = Open();
  std::string pending;
  const auto send = [&](const std::string& bytes) {
    EXPECT_TRUE(jitllm::runtime::http::WriteAll(fd, bytes));
    return ReadResponse(fd, pending);
  };
  EXPECT_THAT(send(Post(Chat("One"))), AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr("One")));
  const std::string stream = send(Post(Chat("Two", R"(,"stream":true)")));
  EXPECT_THAT(stream, AllOf(HasSubstr(R"("content":"wo")"), HasSubstr("Connection: keep-alive")));
  EXPECT_TRUE(stream.ends_with("data: [DONE]\n\n")) << stream;
  EXPECT_THAT(send("GET /v1/nothing HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              AllOf(StartsWith("HTTP/1.1 404 "), HasSubstr("Connection: keep-alive")));
  EXPECT_THAT(send(Post(Chat("x", R"(,"n":3)"))), StartsWith("HTTP/1.1 400 "));
  EXPECT_THAT(send("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 200 "));
  EXPECT_THAT(send(Post(Chat("Three"))), HasSubstr("Three"));
  EXPECT_THAT(send("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr("Connection: close\r\n")));
  EXPECT_TRUE(Closed(fd));
  (void)::close(fd);
  const int old = Open();
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(
      old, "GET /v1/models HTTP/1.0\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n"));
  std::string old_pending;
  EXPECT_THAT(ReadResponse(old, old_pending), HasSubstr("Connection: keep-alive\r\n"));
  EXPECT_TRUE(
      jitllm::runtime::http::WriteAll(old, "GET /v1/models HTTP/1.0\r\nHost: localhost\r\n\r\n"));
  EXPECT_THAT(ReadResponse(old, old_pending), HasSubstr("Connection: close\r\n"));
  EXPECT_TRUE(Closed(old));
  (void)::close(old);
}

// A request sent before the previous response ended is not served: that
// response says Connection: close, and the connection closes after it.
TEST_F(ServerTest, RefusesPipelinedRequests) {
  const int both = Connect(Post(Chat("One")) + Post(Chat("Two")));
  std::string pending;
  EXPECT_THAT(ReadResponse(both, pending), AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("One"),
                                                 HasSubstr("Connection: close\r\n")));
  EXPECT_TRUE(Closed(both));
  EXPECT_THAT(pending, Not(HasSubstr("Two")));
  (void)::close(both);
  // The second arrives while the first runs.
  Stop();
  backend_.release.store(false);
  Start({});
  const int later = Connect(Post(Chat("block")));
  WaitStarted();
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(later, Post(Chat("Two"))));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  backend_.release.store(true);
  std::string rest;
  EXPECT_THAT(ReadResponse(later, rest),
              AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("Connection: close\r\n")));
  EXPECT_TRUE(Closed(later));
  EXPECT_THAT(rest, Not(HasSubstr("Two")));
  (void)::close(later);
}

TEST_F(ServerTest, TimesOutASlowHead) {
  Stop();
  api::ServerOptions options;
  options.head_timeout = std::chrono::milliseconds(200);
  options.body_timeout = std::chrono::milliseconds(400);
  options.idle_timeout = std::chrono::milliseconds(300);
  Start(options);
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n"),
              AllOf(StartsWith("HTTP/1.1 408 "), HasSubstr("Connection: close")));
  EXPECT_THAT(
      Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                           "Content-Type: application/json\r\nContent-Length: 50\r\n\r\n{{")),
      StartsWith("HTTP/1.1 408 "));
  // A connection that sends nothing is closed, once idle, without a
  // response; so is one kept alive after a response.
  EXPECT_EQ(Exchange(""), "");
  const int kept = Connect("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n");
  std::string pending;
  EXPECT_THAT(ReadResponse(kept, pending), StartsWith("HTTP/1.1 200 "));
  EXPECT_TRUE(Closed(kept));
  (void)::close(kept);
}

// Many idle connections are held; at the limit the oldest idle one makes
// room for a new one.
TEST_F(ServerTest, HoldsManyIdleConnections) {
  std::vector<int> idle;
  idle.reserve(200);
  for (int i = 0; i < 200; ++i) {
    idle.push_back(Open());
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_THAT(Exchange(Post(Chat("still here"))), HasSubstr("still here"));
  // Each idle one still works.
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(
      idle[150], "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"));
  std::string pending;
  EXPECT_THAT(ReadResponse(idle[150], pending), StartsWith("HTTP/1.1 200 "));
  for (const int fd : idle) {
    (void)::close(fd);
  }
  Stop();
  api::ServerOptions options;
  options.max_connections = 8;
  Start(options);
  idle.clear();
  for (int i = 0; i < 8; ++i) {
    idle.push_back(Open());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 200 "));
  EXPECT_TRUE(Closed(idle[0]));
  for (const int fd : idle) {
    (void)::close(fd);
  }
}

TEST_F(ServerTest, QueuesThenRefusesConcurrentRequests) {
  Stop();
  backend_.release.store(false);
  api::ServerOptions options;
  options.max_queued = 1;
  Start(options);
  const int running = Connect(Post(Chat("block")));
  WaitStarted();
  const int queued = Connect(Post(Chat("second")));
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // the I/O thread queues it
  const std::string refused = Exchange(Post(Chat("third")));
  EXPECT_THAT(refused, StartsWith("HTTP/1.1 429 "));
  EXPECT_THAT(refused, HasSubstr("Retry-After: 10\r\n"));
  EXPECT_THAT(refused, HasSubstr("x-should-retry: true\r\n"));
  EXPECT_THAT(refused, HasSubstr("rate_limit_error"));
  backend_.release.store(true);
  std::string a;
  std::string b;
  EXPECT_THAT(ReadResponse(running, a), StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(ReadResponse(queued, b), AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("second")));
  (void)::close(running);
  (void)::close(queued);
}

// A stream that waits its turn starts once it has waited a keepalive
// interval, and then hears `: keepalive` until its tokens come; so does
// one whose model is busy before its first token. A non-streaming request
// whose queue wait runs out gets a 429; a stream waits past it.
TEST_F(ServerTest, KeepsStreamsAliveWhileTheyWait) {
  Stop();
  backend_.release.store(false);
  api::ServerOptions options;
  options.keepalive = std::chrono::milliseconds(100);
  Start(options);
  const int running = Connect(Post(Chat("block")));
  WaitStarted();
  const int waiting = Connect(Post(Chat("second", R"(,"stream":true)")));
  std::string early;
  const std::string heard = ReadUntil(waiting, early, ": keepalive\n\n");
  EXPECT_THAT(heard, AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("text/event-stream"),
                           HasSubstr(R"("delta":{"role":"assistant","content":""})"),
                           HasSubstr(": keepalive\n\n"), Not(HasSubstr("second"))));
  backend_.release.store(true);
  std::string pending;
  EXPECT_THAT(ReadResponse(running, pending), StartsWith("HTTP/1.1 200 OK"));
  const std::string rest = ReadUntil(waiting, early, "data: [DONE]\n\n");
  EXPECT_THAT(rest, AllOf(HasSubstr(R"("delta":{"content":"sec"})"),
                          HasSubstr(R"("delta":{"content":"ond"})")));
  EXPECT_EQ(rest.find("\"role\":\"assistant\""), rest.rfind("\"role\":\"assistant\""));  // once
  // A running stream hears keepalives before its first token.
  backend_.started.store(false);
  backend_.release.store(false);
  const int busy = Connect(Post(Chat("block", R"(,"stream":true)")));
  std::string busy_pending;
  EXPECT_THAT(ReadUntil(busy, busy_pending, ": keepalive\n\n"), HasSubstr(": keepalive\n\n"));
  backend_.release.store(true);
  EXPECT_THAT(ReadUntil(busy, busy_pending, "data: [DONE]\n\n"), HasSubstr("data: [DONE]"));
  for (const int fd : {running, waiting, busy}) {
    (void)::close(fd);
  }
  // The queue wait runs out for a non-streaming request (a 429); a stream,
  // held by keepalives, waits past it for its turn.
  Stop();
  backend_.started.store(false);
  backend_.release.store(false);
  options.queue_wait = std::chrono::milliseconds(400);
  Start(options);
  const int blocker = Connect(Post(Chat("block")));
  WaitStarted();
  const int streamed = Connect(Post(Chat("s", R"(,"stream":true)")));
  const int plain = Connect(Post(Chat("p")));
  std::string s;
  std::string p;
  EXPECT_THAT(ReadResponse(plain, p),
              AllOf(StartsWith("HTTP/1.1 429 "), HasSubstr("Retry-After: 10\r\n")));
  EXPECT_THAT(ReadUntil(streamed, s, ": keepalive\n\n"), StartsWith("HTTP/1.1 200 OK"));
  std::this_thread::sleep_for(std::chrono::milliseconds(400));  // twice the queue's wait
  backend_.release.store(true);
  std::string b;
  EXPECT_THAT(ReadResponse(blocker, b), StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(ReadUntil(streamed, s, "data: [DONE]\n\n"),
              AllOf(HasSubstr(R"("delta":{"content":"s"})"), Not(HasSubstr("rate_limit_error")),
                    HasSubstr("data: [DONE]")));
  for (const int fd : {blocker, streamed, plain}) {
    (void)::close(fd);
  }
}

// A client that stalls mid-request, or stops reading its response, holds
// up nobody else; one that stops reading a stream ends its generation.
TEST_F(ServerTest, SlowClientsDoNotHoldUpOthers) {
  Stop();
  api::ServerOptions options;
  options.max_unsent = std::size_t{64} << 10U;
  options.write_timeout = std::chrono::milliseconds(300);
  Start(options);
  const int stalled = Connect("POST /v1/chat/completions HTTP/1.1\r\nHost: loc");
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_THAT(Exchange(Post(Chat("quick"))), HasSubstr("quick"));
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(2));
  // A stream nobody reads.
  const int flood = Open(4096);
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(flood, Post(Chat("flood", R"(,"stream":true)"))));
  WaitStarted();
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 200 "));
  for (int i = 0; i < 2000 && !backend_.flooded.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(backend_.flooded.load());
  EXPECT_THAT(Exchange(Post(Chat("after"))), HasSubstr("after"));
  // A whole response nobody reads: the connection is dropped once its
  // writes stall, and the rest is never buffered further.
  const int big = Open(4096);
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(big, Post(Chat("big"))));
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  EXPECT_THAT(Exchange(Post(Chat("still"))), HasSubstr("still"));
  std::string got;
  while (Recv(big, got)) {
  }
  EXPECT_LT(got.size(), std::size_t{16} << 20U);
  for (const int fd : {stalled, flood, big}) {
    (void)::close(fd);
  }
}

// An unknown field is ignored and counted by name on a loopback-only
// route; its name is logged once, its value never.
TEST_F(ServerTest, IgnoresUnknownFieldsAndCountsThem) {
  Stop();
  std::FILE* log = std::tmpfile();
  if (log == nullptr) {
    ADD_FAILURE() << "no temporary file";
    return;
  }
  api::ServerOptions options;
  options.log = log;
  Start(options);
  EXPECT_THAT(Exchange(Post(Chat("one", R"(,"frobnicate":"secret-value-1")"))),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr("one")));
  EXPECT_THAT(Exchange(Post(Chat("two", R"(,"frobnicate":{"a":"secret-value-2"})"))),
              StartsWith("HTTP/1.1 200 "));
  const std::string table =
      BodyOf(Exchange("GET /jitllm/v1/ignored-fields HTTP/1.1\r\nHost: localhost\r\n\r\n"));
  // No early return from here until the log is closed.
  auto doc = json::Parse(table);
  EXPECT_TRUE(doc.has_value()) << table;
  if (doc.has_value()) {
    const json::Value data = In(doc->root(), {"data"});
    EXPECT_EQ(data.size(), 1U);
    if (data.size() == 1) {
      EXPECT_EQ(In(data.at(0), {"name"}).string(), "frobnicate");
      EXPECT_EQ(In(data.at(0), {"count"}).int64(), 2);
    }
  }
  EXPECT_THAT(Exchange("POST /jitllm/v1/ignored-fields HTTP/1.1\r\nHost: localhost\r\n"
                       "Content-Length: 0\r\n\r\n"),
              StartsWith("HTTP/1.1 405 "));
  Stop();
  (void)std::fflush(log);
  EXPECT_EQ(std::fseek(log, 0, SEEK_SET), 0);
  std::string text;
  std::array<char, 4096> buf{};
  for (std::size_t n = 0; (n = std::fread(buf.data(), 1, buf.size(), log)) > 0;) {
    text.append(buf.data(), n);
  }
  (void)std::fclose(log);
  const std::string line = "an unknown request field is ignored: frobnicate";
  EXPECT_NE(text.find(line), std::string::npos) << text;
  EXPECT_EQ(text.find(line), text.rfind(line));
  EXPECT_THAT(text, Not(HasSubstr("secret-value")));
}

// The node's other names, as the resolution gives them (the tailnet's
// here), pass the Host and Origin guards; others do not.
TEST_F(ServerTest, AcceptsTheNodesNames) {
  Stop();
  api::ServerOptions options;
  options.hosts.AddName("spark-b.coati-puffin.ts.net");
  options.hosts.AddName("spark-b");
  std::array<std::uint8_t, 16> tailnet{100, 114, 118, 63};
  options.hosts.AddAddress(false, tailnet);
  Start(options);
  const auto get = [&](std::string_view host, std::string_view extra = "") {
    return Exchange(std::format("GET /v1/models HTTP/1.1\r\nHost: {}\r\n{}\r\n", host, extra));
  };
  for (const std::string_view ok :
       {"spark-b.coati-puffin.ts.net:8114", "SPARK-B", "100.114.118.63:8114", "localhost"}) {
    EXPECT_THAT(get(ok), StartsWith("HTTP/1.1 200 ")) << ok;
  }
  for (const std::string_view bad : {"evil.coati-puffin.ts.net", "spark-b.coati-puffin.ts.net.",
                                     "100.114.118.64", "[fd7a:115c:a1e0::1]"}) {
    EXPECT_THAT(get(bad), StartsWith("HTTP/1.1 403 ")) << bad;
  }
  EXPECT_THAT(
      get("spark-b", std::format("Origin: http://spark-b.coati-puffin.ts.net:{}\r\n", port_)),
      StartsWith("HTTP/1.1 200 "));
  EXPECT_THAT(get("spark-b", "Origin: http://spark-b.coati-puffin.ts.net:3000\r\n"),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(get("spark-b", std::format("Origin: http://evil.example:{}\r\n", port_)),
              StartsWith("HTTP/1.1 403 "));
  // A bare scheme once aborted the process; it is a 403, and the route
  // goes on serving.
  for (const std::string_view origin : {"http", "https", "null"}) {
    EXPECT_THAT(get("localhost", std::format("Origin: {}\r\n", origin)),
                StartsWith("HTTP/1.1 403 "))
        << origin;
  }
  EXPECT_THAT(get("localhost"), StartsWith("HTTP/1.1 200 "));
}

TEST_F(ServerTest, AFailureAfterTheHeadersEndsTheStreamWithoutDone) {
  const std::string response = Exchange(Post(Chat("late", R"(,"stream":true)")));
  EXPECT_THAT(response, StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(response, HasSubstr(R"(data: {"error":{"message":"failed")"));
  EXPECT_THAT(response, Not(HasSubstr("[DONE]")));
  EXPECT_THAT(Exchange(Post(Chat("late"))), StartsWith("HTTP/1.1 500 "));
}

TEST_F(ServerTest, AClientThatLeavesCancelsItsGeneration) {
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block")));
  WaitStarted();
  (void)::close(fd);
  for (int i = 0; i < 400 && !backend_.cancelled.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(backend_.cancelled.load());
  // The server goes on serving.
  EXPECT_THAT(Exchange(Post(Chat("after"))), StartsWith("HTTP/1.1 200 OK"));
  // A client leaving is not the backend stalling (the watchdog, watchdog.h).
  EXPECT_TRUE(BackendHealth().healthy);
  EXPECT_EQ(BackendHealth().stalls, 0U);
}

// A connection kept alive after its request keeps none of the request's
// or the response's bytes allocated: the accounting sees a body arrive,
// and then nothing large on connections that sit idle.
TEST_F(ServerTest, IdleConnectionsKeepNoLargeBuffers) {
  const std::string text(std::size_t{900} << 10U, 'a');  // a bounded message
  const std::string whole = Post(Chat(text));
  if (!server_.has_value()) {
    ADD_FAILURE() << "no server";
    return;
  }
  const api::Server& server = *server_;
  const auto wait_for = [&](const auto& done) {
    for (int i = 0; i < 400 && !done(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return done();
  };
  // Half a body: the buffer the rest arrives in is held and counted.
  const int first = Connect(std::string_view(whole).substr(0, whole.size() / 2));
  EXPECT_TRUE(wait_for([&] { return server.held_bytes() >= whole.size(); })) << server.held_bytes();
  EXPECT_TRUE(
      jitllm::runtime::http::WriteAll(first, std::string_view(whole).substr(whole.size() / 2)));
  std::vector<int> kept{first};
  for (int i = 0; i < 8; ++i) {
    if (i > 0) {
      kept.push_back(Connect(whole));
    }
    std::string pending;
    const std::string response = ReadResponse(kept.back(), pending);
    EXPECT_THAT(response,
                AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("Connection: keep-alive")));
    EXPECT_GT(response.size(), text.size());  // the answer echoes it
  }
  // Nine requests of about 1 MiB each, answered at about 1 MiB each: the
  // connections, all still open, hold at most a small buffer each way.
  const std::size_t bound = kept.size() * 2 * api::kKeptBufferBytes;
  EXPECT_TRUE(wait_for([&] { return server.held_bytes() <= bound; })) << server.held_bytes();
  // Each still serves.
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(
      kept[3], "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"));
  std::string pending;
  EXPECT_THAT(ReadResponse(kept[3], pending), StartsWith("HTTP/1.1 200 "));
  for (const int fd : kept) {
    (void)::close(fd);
  }
}

// A client that shuts its sending side after a whole request still gets
// its response (after an interim 102 when it half-closed first, and only
// when not streaming), then the connection closes; the generation is not
// cancelled. A shutdown before the request is whole is a disconnect: no
// response.
TEST_F(ServerTest, AHalfClosedClientGetsItsResponse) {
  constexpr std::string_view kInterim = "HTTP/1.1 102 Processing\r\n\r\n";
  const auto half = [&](const std::string& bytes, bool strip = true) {
    const int fd = Connect(bytes);
    EXPECT_EQ(::shutdown(fd, SHUT_WR), 0);
    std::string all;
    while (Recv(fd, all)) {
    }
    // The server closed the connection (nothing more can arrive), rather
    // than the receive timing out on a connection it kept open.
    char byte = 0;
    const ssize_t last = ::recv(fd, &byte, 1, MSG_DONTWAIT);
    EXPECT_FALSE(last < 0 && errno == EAGAIN) << "the connection was kept open after a half-close";
    (void)::close(fd);
    if (strip && all.starts_with(kInterim)) {
      all.erase(0, kInterim.size());  // sent when the shutdown was seen first
    }
    return all;
  };
  // The reviewer's case: a whole POST, then SHUT_WR, then read. The server
  // learns of the shutdown only when its I/O thread sees it, which may come
  // after the backend made the response (a race the client starts): then
  // no interim response came and the head says keep-alive, true when it
  // was made. Seen first, the interim response comes and the head says
  // close. Either way the response is whole and the connection then closes
  // (`half` reads to the end), since nothing more can arrive.
  const std::string raw = half(Post(Chat("Hello world")), false);
  const bool seen_first = raw.starts_with(kInterim);
  const std::string plain = seen_first ? raw.substr(kInterim.size()) : raw;
  EXPECT_THAT(plain,
              AllOf(StartsWith("HTTP/1.1 200 OK\r\n"),
                    HasSubstr(seen_first ? "Connection: close\r\n" : "Connection: keep-alive\r\n"),
                    HasSubstr(R"("content":"Hello world")")));
  // A stream is never sent a 1xx: its probe is its own start.
  const std::string streamed = half(Post(Chat("Hello world", R"(,"stream":true)")), false);
  EXPECT_THAT(streamed, AllOf(StartsWith("HTTP/1.1 200 OK\r\n"), HasSubstr(R"("content":"Hello")"),
                              HasSubstr("data: [DONE]\n\n")));
  EXPECT_TRUE(streamed.ends_with("0\r\n\r\n")) << streamed;
  EXPECT_THAT(half("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 200 OK\r\n"));
  // Before the request is whole: nobody to answer.
  EXPECT_EQ(half("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: "
                 "application/json\r\nContent-Length: 50\r\n\r\n{"),
            "");
  EXPECT_EQ(half("GET /v1/models HTTP/1.1\r\nHost: loc"), "");
  // A generation under way when the client half-closes: the interim
  // response comes at once, the generation goes on, the answer follows.
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block")));
  WaitStarted();
  ASSERT_EQ(::shutdown(fd, SHUT_WR), 0);
  std::string got;
  EXPECT_THAT(ReadUntil(fd, got, "\r\n\r\n"), StartsWith(kInterim));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(backend_.cancelled.load());
  backend_.release.store(true);
  while (Recv(fd, got)) {
  }
  (void)::close(fd);
  EXPECT_FALSE(backend_.cancelled.load());
  EXPECT_THAT(got.substr(std::min(got.size(), kInterim.size())),
              AllOf(StartsWith("HTTP/1.1 200 OK\r\n"), HasSubstr("Connection: close\r\n"),
                    HasSubstr(R"("content":"block")")));
}

// A client that closes entirely while a stream waits for its first token
// is told apart from a half-close by the probe, and its generation ends.
TEST_F(ServerTest, AStreamsClientThatLeavesCancelsItsGeneration) {
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block", R"(,"stream":true)")));
  WaitStarted();
  std::string head;
  EXPECT_THAT(ReadUntil(fd, head, "\n\n"), StartsWith("HTTP/1.1 200 OK"));  // admitted
  (void)::close(fd);
  for (int i = 0; i < 400 && !backend_.cancelled.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(backend_.cancelled.load());
  EXPECT_THAT(Exchange(Post(Chat("after"))), StartsWith("HTTP/1.1 200 OK"));
  EXPECT_TRUE(BackendHealth().healthy);
  EXPECT_EQ(BackendHealth().stalls, 0U);
}

// A queued stream whose client half-closes starts at once (headers and
// role chunk, no 1xx, well before `keepalive`) and is answered after its
// turn. A queued request whose client closes entirely leaves the queue
// (the probe draws a reset), streaming or not: its place is free.
TEST_F(ServerTest, QueuedRequestsTellAHalfCloseFromALeave) {
  Stop();
  backend_.release.store(false);
  api::ServerOptions options;
  options.max_queued = 1;
  options.keepalive = std::chrono::seconds(30);
  Start(options);
  const int running = Connect(Post(Chat("block")));
  WaitStarted();
  const int waiting = Connect(Post(Chat("second", R"(,"stream":true)")));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // the I/O thread queues it
  ASSERT_EQ(::shutdown(waiting, SHUT_WR), 0);
  std::string early;
  EXPECT_THAT(ReadUntil(waiting, early, "\n\n"),
              AllOf(StartsWith("HTTP/1.1 200 OK\r\n"), HasSubstr("Connection: close\r\n"),
                    HasSubstr(R"("delta":{"role":"assistant","content":""})")));
  backend_.release.store(true);
  std::string pending;
  EXPECT_THAT(ReadResponse(running, pending), StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(ReadUntil(waiting, early, "data: [DONE]\n\n"), HasSubstr(R"("content":"ond")"));
  (void)::close(running);
  (void)::close(waiting);

  for (const bool stream : {true, false}) {
    backend_.release.store(false);
    backend_.started.store(false);
    const int busy = Connect(Post(Chat("block")));
    WaitStarted();
    const int leaving =
        Connect(Post(Chat("second", stream ? R"(,"stream":true)" : std::string_view{})));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    (void)::close(leaving);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // The queue's one place is free again.
    const int next = Connect(Post(Chat("third")));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    backend_.release.store(true);
    std::string a;
    std::string b;
    EXPECT_THAT(ReadResponse(busy, a), StartsWith("HTTP/1.1 200 OK")) << stream;
    EXPECT_THAT(ReadResponse(next, b), AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("third")))
        << stream;
    (void)::close(busy);
    (void)::close(next);
  }
}

// A request that makes progress runs as long as it needs: here 1.5 s,
// 7.5 times the stall time (as 15 minutes would be to the default's 120 s,
// past the old fixed 600 s deadline), streaming or not; and one unit that
// declares its size (a 100-row chunk: 1 s at the floor, allowed 3.2 s) is
// not cut short by the stall time alone.
TEST_F(ServerTest, AProgressingRequestOutlivesTheStallTime) {
  Stop();
  api::ServerOptions options;
  options.stall = std::chrono::milliseconds(200);
  Start(options);
  const auto started = std::chrono::steady_clock::now();
  const std::string plain = Exchange(Post(Chat("slow")));
  EXPECT_GE(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(1500));
  EXPECT_THAT(plain, AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr(R"("content":"slow")")));
  const std::string streamed = Exchange(Post(Chat("slow", R"(,"stream":true)")));
  EXPECT_THAT(streamed, AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("data: [DONE]")));
  EXPECT_THAT(Exchange(Post(Chat("wide"))),
              AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr(R"("content":"wide")")));
  const jitllm::runtime::Health health = BackendHealth();
  EXPECT_TRUE(health.healthy);
  EXPECT_EQ(health.stalls, 0U);
  EXPECT_FALSE(backend_.cancelled.load());
}

// A backend that makes no progress for the stall time trips the watchdog
// while it is still hung: the request ends (a 504 before the headers, an
// in-stream error after, without [DONE]), what is queued and what arrives
// gets a 503, and the backend is unhealthy until the unit it hung in
// returns; its generation then ends at that step, as a client's leaving
// ends it, and the service serves on.
TEST_F(ServerTest, AStalledBackendTripsTheWatchdog) {
  Stop();
  backend_.release.store(false);
  api::ServerOptions options;
  options.stall = std::chrono::milliseconds(500);
  struct Reports {
    std::atomic<int> unhealthy{0};
    std::atomic<int> healthy{0};
  };
  const auto reports = std::make_shared<Reports>();
  options.on_health = [reports](const jitllm::runtime::Health& health) {
    (health.healthy ? reports->healthy : reports->unhealthy).fetch_add(1);
  };
  Start(options);
  const auto started = std::chrono::steady_clock::now();
  const int hung = Connect(Post(Chat("hang")));
  WaitStarted();
  const int queued = Connect(Post(Chat("queued")));
  std::string a;
  std::string b;
  EXPECT_THAT(ReadResponse(hung, a),
              AllOf(StartsWith("HTTP/1.1 504 "), HasSubstr("backend_stalled")));
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(10));
  EXPECT_THAT(ReadResponse(queued, b),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("Retry-After: 10\r\n"),
                    HasSubstr("backend_unresponsive")));
  const jitllm::runtime::Health health = BackendHealth();
  EXPECT_FALSE(health.healthy);
  EXPECT_EQ(health.stalls, 1U);
  EXPECT_EQ(health.phase, jitllm::runtime::Phase::kStarting);  // where it hung
  EXPECT_EQ(reports->unhealthy.load(), 1);
  EXPECT_THAT(Exchange(Post(Chat("refused"))),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("backend_unresponsive")));
  EXPECT_FALSE(backend_.cancelled.load());  // still hung

  const auto recover = [&] {
    backend_.release.store(true);
    for (int i = 0; i < 2000 && !(backend_.cancelled.load() && BackendHealth().healthy); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(backend_.cancelled.load());  // ended at its next step
    EXPECT_TRUE(BackendHealth().healthy);
  };
  recover();
  EXPECT_GE(reports->healthy.load(), 1);
  EXPECT_THAT(Exchange(Post(Chat("after"))), StartsWith("HTTP/1.1 200 OK"));

  // A stream that stalls after its headers ends with an in-stream error.
  backend_.started.store(false);
  backend_.release.store(false);
  backend_.cancelled.store(false);
  const int streamed = Connect(Post(Chat("hang", R"(,"stream":true)")));
  WaitStarted();
  std::string s;
  const std::string heard = ReadResponse(streamed, s);
  EXPECT_THAT(heard, AllOf(StartsWith("HTTP/1.1 200 OK"),
                           HasSubstr(R"("delta":{"role":"assistant","content":"")"),
                           HasSubstr(R"(data: {"error":)"), HasSubstr("backend_stalled"),
                           Not(HasSubstr("[DONE]"))));
  EXPECT_EQ(BackendHealth().stalls, 2U);
  recover();
  EXPECT_THAT(Exchange(Post(Chat("again"))), StartsWith("HTTP/1.1 200 OK"));
  for (const int fd : {hung, queued, streamed}) {
    (void)::close(fd);
  }
  Stop();  // before `reports`' last user goes
}

// A non-streaming request's deadline is its work at the model's floors
// (watchdog.h ScaledDeadline), capped; a stream has none and runs past it
// while it makes progress.
TEST_F(ServerTest, OnlyANonStreamingRequestHasADeadline) {
  Stop();
  backend_.release.store(false);
  // 10 prompt tokens and 100 to generate at 1,000 tokens a second: 0.11 s,
  // three times over, plus the stall time: 0.53 s.
  backend_.admission.max_tokens = 100;
  backend_.admission.floors = {.prefill = 1000, .decode = 1000};
  api::ServerOptions options;
  options.stall = std::chrono::milliseconds(200);
  Start(options);
  auto started = std::chrono::steady_clock::now();
  const std::string scaled = Exchange(Post(Chat("block")));
  auto took = std::chrono::steady_clock::now() - started;
  EXPECT_THAT(scaled, AllOf(StartsWith("HTTP/1.1 504 "), HasSubstr("deadline")));
  EXPECT_GE(took, std::chrono::milliseconds(530));
  EXPECT_LT(took, std::chrono::seconds(5));
  EXPECT_TRUE(backend_.cancelled.load());
  EXPECT_TRUE(BackendHealth().healthy);  // a deadline is not a stall

  // The same work as a stream runs until it is done, well past that.
  backend_.started.store(false);
  backend_.cancelled.store(false);
  const int fd = Connect(Post(Chat("block", R"(,"stream":true)")));
  WaitStarted();
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  backend_.release.store(true);
  std::string pending;
  EXPECT_THAT(ReadUntil(fd, pending, "data: [DONE]\n\n"),
              AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("data: [DONE]")));
  EXPECT_FALSE(backend_.cancelled.load());
  (void)::close(fd);

  // The cap bounds the scaled deadline.
  Stop();
  backend_.started.store(false);
  backend_.release.store(false);
  backend_.cancelled.store(false);
  backend_.admission.max_tokens = 1000;
  backend_.admission.floors = {};
  options.stall = std::chrono::seconds(10);
  options.deadline_cap = std::chrono::milliseconds(300);
  Start(options);
  started = std::chrono::steady_clock::now();
  EXPECT_THAT(Exchange(Post(Chat("block"))), StartsWith("HTTP/1.1 504 "));
  took = std::chrono::steady_clock::now() - started;
  EXPECT_GE(took, std::chrono::milliseconds(300));
  EXPECT_LT(took, std::chrono::seconds(5));
  EXPECT_TRUE(backend_.cancelled.load());
  EXPECT_THAT(Exchange(Post(Chat("after"))), StartsWith("HTTP/1.1 200 OK"));
}

TEST_F(ServerTest, StoppingEndsARunningRequest) {
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block")));
  WaitStarted();
  const int queued = Connect(Post(Chat("queued")));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const std::uint64_t one = 1;
  (void)!::write(wake_, &one, sizeof one);
  std::string a;
  std::string b;
  EXPECT_THAT(ReadResponse(fd, a),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("Connection: close\r\n")));
  EXPECT_THAT(ReadResponse(queued, b),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("Retry-After: 10\r\n")));
  thread_.join();
  (void)::close(wake_);
  (void)::close(fd);
  (void)::close(queued);
  EXPECT_TRUE(result_.has_value());
}

TEST_F(ServerTest, CooperativeRequestsJoinAnExistingDecodeAndStayBounded) {
  StartCooperative();
  cooperative_->pause_units.store(true);
  const int first = Connect(Post(Chat("long", R"(,"max_tokens":12)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->advances.load() != 0; }));
  std::array<int, 4> later{};
  for (int& fd : later) {
    fd = Connect(Post(Chat("short", R"(,"max_tokens":4)")));
  }
  // All connections are handled while one backend unit remains in flight.
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"),
              StartsWith("HTTP/1.1 200 "));
  cooperative_->release.store(true);
  ASSERT_TRUE(WaitFor([&] { return cooperative_->peak.load() == api::kMaxActiveRequests; }));
  EXPECT_EQ(cooperative_->peak.load(), api::kMaxActiveRequests);
  std::string pending;
  for (const int fd : later) {
    EXPECT_THAT(ReadResponse(fd, pending),
                AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr(R"("content":"xxxx")")));
    (void)::close(fd);
  }
  EXPECT_THAT(ReadResponse(first, pending),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr(R"("completion_tokens":12)")));
  (void)::close(first);
  ASSERT_TRUE(WaitFor([&] { return cooperative_->destroyed.load() == 5; }));
  EXPECT_EQ(cooperative_->retired.load(), 5U);
  EXPECT_EQ(backend_.chat_calls.load(), 0U);
  EXPECT_FALSE(cooperative_->early_destruction.load());
}

TEST_F(ServerTest, CooperativeCapacityDefersThenRefillsWithoutSerialFallback) {
  StartCooperative();
  cooperative_->capacity.store(2);
  cooperative_->pause_units.store(true);
  const int first = Connect(Post(Chat("long", R"(,"max_tokens":12)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->advances.load() != 0; }));
  const int second = Connect(Post(Chat("short", R"(,"max_tokens":4,"capacity_second":true)")));
  const int third = Connect(Post(Chat("short", R"(,"max_tokens":4,"capacity_third":true)")));
  // Authenticate both pending requests before completing the blocked unit.
  ASSERT_TRUE(WaitFor([&] {
    const auto ignored =
        Exchange("GET /jitllm/v1/ignored-fields HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
    return ignored.contains("capacity_second") && ignored.contains("capacity_third");
  }));
  cooperative_->release.store(true);
  std::string pending;
  for (const int fd : {second, third}) {
    EXPECT_THAT(ReadResponse(fd, pending),
                AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr(R"("content":"xxxx")")));
    (void)::close(fd);
  }
  EXPECT_THAT(ReadResponse(first, pending),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr(R"("completion_tokens":12)")));
  (void)::close(first);
  ASSERT_TRUE(WaitFor([&] { return cooperative_->destroyed.load() == 3; }));
  EXPECT_EQ(cooperative_->peak.load(), 2U);
  EXPECT_GT(cooperative_->capacity_deferrals.load(), 0U);
  EXPECT_EQ(cooperative_->started.load(), 3U);
  EXPECT_EQ(cooperative_->retired.load(), 3U);
  EXPECT_EQ(backend_.chat_calls.load(), 0U);
  EXPECT_FALSE(cooperative_->early_destruction.load());
}

TEST_F(ServerTest, CooperativeCancellationAndHalfCloseBelongToOneRequest) {
  StartCooperative();
  const int leaving = Connect(Post(Chat("long", R"(,"stream":true,"max_tokens":40)")));
  std::string pending;
  ASSERT_THAT(ReadUntil(leaving, pending, "data: "), StartsWith("HTTP/1.1 200 "));
  const int half = Connect(Post(Chat("half", R"(,"max_tokens":8)")));
  EXPECT_EQ(::shutdown(half, SHUT_WR), 0);
  ASSERT_TRUE(WaitFor([&] { return cooperative_->peak.load() == 2; }));
  linger reset{.l_onoff = 1, .l_linger = 0};
  (void)::setsockopt(leaving, SOL_SOCKET, SO_LINGER, &reset, sizeof reset);
  (void)::close(leaving);
  pending.clear();
  const std::string response = ReadUntil(half, pending, R"("finish_reason")");
  EXPECT_THAT(response, AllOf(HasSubstr("HTTP/1.1 200 OK"), HasSubstr(R"("content":"xxxxxxxx")")));
  (void)::close(half);
  ASSERT_TRUE(WaitFor([&] { return cooperative_->destroyed.load() == 2; }));
  EXPECT_EQ(cooperative_->cancelled.load(), 1U);
  EXPECT_TRUE(BackendHealth().healthy);
  EXPECT_FALSE(cooperative_->early_destruction.load());
}

TEST_F(ServerTest, CooperativeDeadlinesDoNotEndAnotherRequestsStream) {
  api::ServerOptions options;
  options.deadline_cap = std::chrono::milliseconds(100);
  options.stall = std::chrono::seconds(2);
  StartCooperative(options);
  const int timed = Connect(Post(Chat("timed", R"(,"max_tokens":100)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->advances.load() != 0; }));
  const int streamed = Connect(Post(Chat("stream", R"(,"stream":true,"max_tokens":30)")));
  std::string pending;
  EXPECT_THAT(ReadResponse(timed, pending),
              AllOf(StartsWith("HTTP/1.1 504 "), HasSubstr("deadline")));
  pending.clear();
  EXPECT_THAT(ReadResponse(streamed, pending),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr("data: [DONE]")));
  for (const int fd : {timed, streamed}) {
    (void)::close(fd);
  }
  ASSERT_TRUE(WaitFor([&] { return cooperative_->destroyed.load() == 2; }));
  EXPECT_EQ(cooperative_->cancelled.load(), 1U);
  EXPECT_EQ(BackendHealth().stalls, 0U);
}

TEST_F(ServerTest, CooperativeCohortDrainsBeforeSerialFallbackAndLaterRefill) {
  StartCooperative();
  cooperative_->pause_units.store(true);
  const int first = Connect(Post(Chat("long", R"(,"max_tokens":10)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->advances.load() != 0; }));
  const int serial = Connect(Post(Chat("serial", R"(,"cooperative_fifo_marker":true)")));
  // The diagnostic route authenticates that this whole request was parsed
  // before a later eligible one is sent; the driver is still blocked.
  ASSERT_TRUE(WaitFor([&] {
    return Exchange("GET /jitllm/v1/ignored-fields HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .contains("cooperative_fifo_marker");
  }));
  const int later = Connect(Post(Chat("later", R"(,"max_tokens":4)")));
  cooperative_->release.store(true);
  std::string pending;
  for (const int fd : {first, serial, later}) {
    EXPECT_THAT(ReadResponse(fd, pending), StartsWith("HTTP/1.1 200 "));
    (void)::close(fd);
  }
  EXPECT_TRUE(cooperative_->later_after_serial.load());
  EXPECT_EQ(cooperative_->peak.load(), 1U);
  EXPECT_EQ(backend_.chat_calls.load(), 1U);
  EXPECT_FALSE(cooperative_->early_destruction.load());
  EXPECT_THAT(Exchange(Post(Chat("defer"))),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr(R"("content":"defer")")));
  EXPECT_EQ(cooperative_->deferrals.load(), 1U);
  EXPECT_EQ(backend_.chat_calls.load(), 2U);
}

TEST_F(ServerTest, CooperativeChannelPollingCannotHideAGlobalStall) {
  api::ServerOptions options;
  options.stall = std::chrono::milliseconds(200);
  StartCooperative(options);
  const int first = Connect(Post(Chat("long", R"(,"max_tokens":100)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->advances.load() != 0; }));
  const int second = Connect(Post(Chat("long", R"(,"max_tokens":100)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->peak.load() == 2; }));
  cooperative_->pause_units.store(true);
  ASSERT_TRUE(WaitFor([&] { return cooperative_->polls.load() != 0; }));
  const int queued = Connect(Post(Chat("serial")));
  std::string pending;
  for (const int fd : {first, second}) {
    EXPECT_THAT(ReadResponse(fd, pending),
                AllOf(StartsWith("HTTP/1.1 504 "), HasSubstr("backend_stalled")));
    (void)::close(fd);
  }
  EXPECT_THAT(ReadResponse(queued, pending),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("backend_unresponsive")));
  (void)::close(queued);
  EXPECT_FALSE(BackendHealth().healthy);
  EXPECT_EQ(cooperative_->destroyed.load(), 0U);
  cooperative_->release.store(true);
  ASSERT_TRUE(
      WaitFor([&] { return cooperative_->destroyed.load() == 2 && BackendHealth().healthy; }));
  EXPECT_EQ(cooperative_->cancelled.load(), 2U);
  EXPECT_THAT(Exchange(Post(Chat("serial"))), StartsWith("HTTP/1.1 200 "));
}

TEST_F(ServerTest, CooperativeFaultKeepsBorrowedFramesUntilRetirementCompletes) {
  StartCooperative();
  cooperative_->fail_unit.store(true);
  cooperative_->pause_retirement.store(true);
  const int fd = Connect(Post(Chat("fault", R"(,"max_tokens":4)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->retirement_entered.load(); }));
  EXPECT_EQ(cooperative_->live.load(), 1U);
  EXPECT_EQ(cooperative_->retired.load(), 0U);
  EXPECT_EQ(cooperative_->destroyed.load(), 0U);
  cooperative_->release_retirement.store(true);
  std::string pending;
  EXPECT_THAT(ReadResponse(fd, pending),
              AllOf(StartsWith("HTTP/1.1 500 "), HasSubstr("backend_failed")));
  (void)::close(fd);
  Stop();
  EXPECT_FALSE(result_.has_value());
  EXPECT_EQ(cooperative_->retired.load(), 1U);
  EXPECT_EQ(cooperative_->destroyed.load(), 1U);
  EXPECT_FALSE(cooperative_->early_destruction.load());
}

TEST_F(ServerTest, CooperativeLiteralResponsesShareTheSocketBufferBudget) {
  constexpr std::string_view body =
      R"({"model":"alpha","prompt":"huge","max_tokens":0,"echo":true})";
  api::ServerOptions options;
  auto request = api::ParseCompletionRequest(body);
  ASSERT_TRUE(request.has_value());
  api::LiteralResult large;
  large.prompt_text.assign(std::size_t{4} << 20U, 'x');
  auto serialized =
      api::LiteralCompletionJson("cmpl-000000000000000000000000", 1000000000, *request, "", large,
                                 api::Finish::kStop, {.prompt_tokens = 4});
  ASSERT_TRUE(serialized.has_value());
  options.response_budget = serialized->capacity() + (serialized->capacity() / 2);
  options.write_timeout = std::chrono::seconds(5);
  StartCooperative(options);
  if (!server_.has_value()) {
    ADD_FAILURE() << "the cooperative response-budget server did not start";
    return;
  }
  auto& server = *server_;
  jitllm::runtime::http::Fd slow(Open(4096));
  ASSERT_TRUE(jitllm::runtime::http::WriteAll(slow.get(), Literal(body)));
  ASSERT_TRUE(WaitFor([&] { return server.response_bytes() != 0; }));
  EXPECT_LE(server.response_bytes(), options.response_budget);
  EXPECT_THAT(Exchange(Literal(body)),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("response_budget_exceeded")));
  EXPECT_LE(server.response_bytes(), options.response_budget);
  linger reset{.l_onoff = 1, .l_linger = 0};
  (void)::setsockopt(slow.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof reset);
  slow = jitllm::runtime::http::Fd();
  ASSERT_TRUE(WaitFor([&] { return server.response_bytes() == 0; }));
  EXPECT_THAT(Exchange(Literal(body)), StartsWith("HTTP/1.1 200 "));
  ASSERT_TRUE(WaitFor([&] { return server.response_bytes() == 0; }));
  EXPECT_FALSE(cooperative_->early_destruction.load());
}

TEST_F(ServerTest, CooperativeShutdownRetiresEveryActiveRequest) {
  StartCooperative();
  const int first = Connect(Post(Chat("long", R"(,"max_tokens":100)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->advances.load() != 0; }));
  const int second = Connect(Post(Chat("long", R"(,"max_tokens":100)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->peak.load() == 2; }));
  const std::uint64_t one = 1;
  (void)!::write(wake_, &one, sizeof one);
  std::string pending;
  for (const int fd : {first, second}) {
    EXPECT_THAT(ReadResponse(fd, pending),
                AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("Connection: close\r\n")));
    (void)::close(fd);
  }
  Stop();
  EXPECT_TRUE(result_.has_value());
  EXPECT_EQ(cooperative_->retired.load(), 2U);
  EXPECT_EQ(cooperative_->destroyed.load(), 2U);
  EXPECT_FALSE(cooperative_->early_destruction.load());
}

TEST_F(ServerTest, CooperativeRefusedIntakeCannotStarveAnExistingDecode) {
  StartCooperative();
  cooperative_->pause_units.store(true);
  const int first = Connect(Post(Chat("long", R"(,"max_tokens":20)")));
  ASSERT_TRUE(WaitFor([&] { return cooperative_->advances.load() != 0; }));
  std::array<int, 16> rejected{};
  for (int& fd : rejected) {
    fd = Connect(Post(Chat("bad")));
  }
  cooperative_->release.store(true);
  std::string pending;
  for (const int fd : rejected) {
    EXPECT_THAT(ReadResponse(fd, pending), StartsWith("HTTP/1.1 400 "));
    (void)::close(fd);
  }
  EXPECT_THAT(ReadResponse(first, pending),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr(R"("completion_tokens":20)")));
  (void)::close(first);
  EXPECT_EQ(cooperative_->rejections.load(), rejected.size());
  EXPECT_LE(cooperative_->rejections_between_units.load(), api::kMaxActiveRequests);
  EXPECT_EQ(backend_.chat_calls.load(), 0U);
  EXPECT_FALSE(cooperative_->early_destruction.load());
}

}  // namespace
