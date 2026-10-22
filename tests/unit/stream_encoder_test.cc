// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/stream_encoder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "base/json.h"

namespace {
namespace api = llmp::runtime::api;
namespace json = llmp::base::json;
using ::testing::HasSubstr;
using ::testing::Not;

const api::StreamIdentity kIdentity{.id = "request\"1", .model = "model\n1", .created = 123};
const api::Usage kUsage{.prompt_tokens = 13, .completion_tokens = 7, .cached_tokens = 4};

struct Capture {
  std::vector<std::string> events;
  api::StreamEncoder::Sink sink = [this](std::string_view event) { events.emplace_back(event); };

  std::string_view Data(std::size_t index) const {
    const auto& event = events.at(index);
    const auto pos = event.find("data: ");
    if (pos == std::string::npos || !event.ends_with("\n\n")) {
      ADD_FAILURE() << event;
      return {};
    }
    return std::string_view(event).substr(pos + 6, event.size() - pos - 8);
  }
  void ValidateJson() const {
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (!events[i].starts_with(":") && Data(i) != "[DONE]") {
        const auto parsed = json::Parse(Data(i));
        EXPECT_TRUE(parsed) << Data(i);
      }
    }
  }
};

TEST(StreamEncoder, OpenAiPreservesEveryExistingEventAndBoundary) {
  api::StreamEncoder encoder;
  Capture output;
  ASSERT_TRUE(encoder.Start(kIdentity, {}, output.sink));
  ASSERT_TRUE(encoder.Start(kIdentity, kUsage, output.sink));
  ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kReasoning, "why\n\"", output.sink));
  ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kContent, "answer\t", output.sink));
  encoder.Keepalive(output.sink);
  ASSERT_TRUE(encoder.End(kIdentity, api::StreamStop::kLength, kUsage, true, output.sink));
  EXPECT_EQ(
      output.events,
      (std::vector<std::string>{
          "data: " + api::ChunkJson(kIdentity.id, 123, kIdentity.model, api::Delta::kRole, {}) +
              "\n\n",
          "data: " +
              api::ChunkJson(kIdentity.id, 123, kIdentity.model, api::Delta::kReasoning,
                             "why\n\"") +
              "\n\n",
          "data: " +
              api::ChunkJson(kIdentity.id, 123, kIdentity.model, api::Delta::kContent, "answer\t") +
              "\n\n",
          ": keepalive\n\n",
          "data: " +
              api::ChunkJson(kIdentity.id, 123, kIdentity.model, api::Delta::kFinish, {},
                             api::Finish::kLength) +
              "\n\n",
          "data: " + api::UsageChunkJson(kIdentity.id, 123, kIdentity.model, kUsage) + "\n\n",
          "data: [DONE]\n\n"}));
  output.ValidateJson();
  encoder.Keepalive(output.sink);
  EXPECT_EQ(output.events.size(), 7U);
  EXPECT_FALSE(encoder.End(kIdentity, api::StreamStop::kStop, kUsage, true, output.sink));
}

TEST(StreamEncoder, OpenAiUsageIsOptionalAndErrorsHaveNoDone) {
  for (const bool fail : {false, true}) {
    api::StreamEncoder encoder;
    Capture output;
    ASSERT_TRUE(encoder.Start(kIdentity, {}, output.sink));
    const api::Error error{.status = 503, .message = "try\nagain", .param = {}, .code = {}};
    if (fail) {
      EXPECT_EQ(encoder.ErrorBody(error), api::ErrorJson(error));
      encoder.Fail(error, output.sink);
      EXPECT_EQ(output.events.back(), "data: " + api::ErrorJson(error) + "\n\n");
      EXPECT_FALSE(encoder.Text(kIdentity, api::StreamText::kContent, "late", output.sink));
    } else {
      ASSERT_TRUE(encoder.End(kIdentity, api::StreamStop::kStop, kUsage, false, output.sink));
      EXPECT_EQ(output.events.back(), "data: [DONE]\n\n");
    }
    EXPECT_EQ(output.events.size(), fail ? 2U : 3U);
    output.ValidateJson();
  }
}

TEST(StreamEncoder, MessagesWaitsForAdmissionAndClosesIndexedText) {
  api::StreamEncoder encoder(api::StreamRoute::kMessages);
  Capture output;
  ASSERT_TRUE(encoder.Start(kIdentity, {}, output.sink));
  EXPECT_EQ(output.events,
            (std::vector<std::string>{"event: ping\ndata: {\"type\":\"ping\"}\n\n"}));
  EXPECT_FALSE(encoder.Text(kIdentity, api::StreamText::kContent, "early", output.sink));
  ASSERT_TRUE(encoder.Start(kIdentity, kUsage, output.sink));
  ASSERT_TRUE(encoder.Start(kIdentity, kUsage, output.sink));
  const auto start = json::Parse(output.Data(1));
  ASSERT_TRUE(start);
  const auto message = start->root().find("message");
  ASSERT_TRUE(message);
  EXPECT_EQ(message->find("id")->string(), kIdentity.id);
  EXPECT_EQ(message->find("model")->string(), kIdentity.model);
  EXPECT_EQ(message->find("usage")->find("input_tokens")->int64(), 9);
  EXPECT_EQ(message->find("usage")->find("cache_read_input_tokens")->int64(), 4);
  EXPECT_EQ(message->find("usage")->find("output_tokens")->int64(), 0);
  EXPECT_FALSE(encoder.Text(kIdentity, api::StreamText::kReasoning, "unsigned", output.sink));
  ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kContent, "", output.sink));
  EXPECT_EQ(output.events.size(), 2U);
  ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kContent, "one\n", output.sink));
  ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kContent, "two", output.sink));
  ASSERT_TRUE(encoder.End(kIdentity, api::StreamStop::kStop, kUsage, false, output.sink));
  ASSERT_EQ(output.events.size(), 8U);
  EXPECT_THAT(output.events[2], HasSubstr("\"index\":0"));
  EXPECT_THAT(output.events[3], HasSubstr("text_delta"));
  EXPECT_THAT(output.events[4], HasSubstr("text_delta"));
  EXPECT_EQ(output.events[5],
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n");
  EXPECT_THAT(output.events[6], HasSubstr("\"stop_reason\":\"end_turn\""));
  EXPECT_THAT(output.events[6], HasSubstr("\"output_tokens\":7"));
  EXPECT_EQ(output.events[7], "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
  output.ValidateJson();
}

TEST(StreamEncoder, MessagesFinalUsageReportsCacheReuseDiscoveredAfterAdmission) {
  api::StreamEncoder encoder(api::StreamRoute::kMessages);
  Capture output;
  const api::Usage admitted{.prompt_tokens = kUsage.prompt_tokens};
  ASSERT_TRUE(encoder.Start(kIdentity, admitted, output.sink));
  const auto start = json::Parse(output.Data(0));
  ASSERT_TRUE(start);
  EXPECT_EQ(start->root().find("message")->find("usage")->find("input_tokens")->int64(), 13);
  const api::Usage invalid{.prompt_tokens = 1, .cached_tokens = 2};
  EXPECT_FALSE(encoder.End(kIdentity, api::StreamStop::kStop, invalid, false, output.sink));
  EXPECT_EQ(output.events.size(), 1U);
  ASSERT_TRUE(encoder.End(kIdentity, api::StreamStop::kLength, kUsage, false, output.sink));
  ASSERT_EQ(output.events.size(), 3U);  // no invented empty content block
  const auto end = json::Parse(output.Data(1));
  ASSERT_TRUE(end);
  const auto usage = end->root().find("usage");
  ASSERT_TRUE(usage);
  EXPECT_EQ(usage->find("input_tokens")->int64(), 9);
  EXPECT_EQ(usage->find("cache_read_input_tokens")->int64(), 4);
  EXPECT_EQ(usage->find("output_tokens")->int64(), 7);
}

TEST(StreamEncoder, MessagesToolJsonSurvivesEveryByteBoundaryAndEscaping) {
  const std::string arguments = "{\"text\":\"quote\\\" slash\\\\ line\\n euro € emoji 😀\"}";
  for (std::size_t split = 0; split <= arguments.size(); ++split) {
    SCOPED_TRACE(split);
    api::StreamEncoder encoder(api::StreamRoute::kMessages);
    Capture output;
    ASSERT_TRUE(encoder.Start(kIdentity, kUsage, output.sink));
    ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kContent, "before", output.sink));
    ASSERT_TRUE(encoder.ToolStart(kIdentity, "call\"1", "tool\n1", output.sink));
    ASSERT_TRUE(
        encoder.ToolInput(kIdentity, std::string_view(arguments).substr(0, split), output.sink));
    ASSERT_TRUE(
        encoder.ToolInput(kIdentity, std::string_view(arguments).substr(split), output.sink));
    ASSERT_TRUE(encoder.ToolEnd(output.sink));
    ASSERT_TRUE(encoder.ToolStart(kIdentity, "call2", "tool2", output.sink));
    ASSERT_TRUE(encoder.ToolInput(kIdentity, "{}", output.sink));
    ASSERT_TRUE(encoder.ToolEnd(output.sink));
    ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kContent, "after", output.sink));
    ASSERT_TRUE(encoder.End(kIdentity, api::StreamStop::kToolUse, kUsage, false, output.sink));
    std::string reconstructed;
    std::vector<std::int64_t> starts;
    std::vector<std::int64_t> stops;
    for (std::size_t i = 0; i < output.events.size(); ++i) {
      const auto parsed = json::Parse(output.Data(i));
      ASSERT_TRUE(parsed) << output.events[i];
      const auto root = parsed->root();
      const auto type = root.find("type")->string();
      if (type == "content_block_start") {
        starts.push_back(*root.find("index")->int64());
        if (starts.back() == 1) {
          EXPECT_EQ(root.find("content_block")->find("id")->string(), "call\"1");
          EXPECT_EQ(root.find("content_block")->find("name")->string(), "tool\n1");
        }
      } else if (type == "content_block_stop") {
        stops.push_back(*root.find("index")->int64());
      } else if (type == "content_block_delta" && root.find("index")->int64() == 1) {
        EXPECT_EQ(root.find("delta")->find("type")->string(), "input_json_delta");
        reconstructed += root.find("delta")->find("partial_json")->string();
      }
    }
    EXPECT_EQ(starts, (std::vector<std::int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(stops, starts);
    EXPECT_EQ(reconstructed, arguments);
    EXPECT_THAT(output.events[output.events.size() - 2], HasSubstr("\"stop_reason\":\"tool_use\""));
  }
}

TEST(StreamEncoder, OpenAiToolIndexesExcludeTextAndArgumentsStreamAsStrings) {
  api::StreamEncoder encoder;
  Capture output;
  ASSERT_TRUE(encoder.Start(kIdentity, {}, output.sink));
  ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kContent, "before", output.sink));
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(encoder.ToolStart(kIdentity, "call", "tool", output.sink));
    const auto start = json::Parse(output.Data(output.events.size() - 1));
    ASSERT_TRUE(start);
    const auto call = start->root().find("choices")->at(0).find("delta")->find("tool_calls")->at(0);
    EXPECT_EQ(call.find("index")->int64(), i);
    EXPECT_EQ(call.find("id")->string(), "call");
    EXPECT_EQ(call.find("function")->find("name")->string(), "tool");
    ASSERT_TRUE(encoder.ToolInput(kIdentity, "{\"x\":", output.sink));
    ASSERT_TRUE(encoder.ToolInput(kIdentity, "1}", output.sink));
    ASSERT_TRUE(encoder.ToolEnd(output.sink));
    ASSERT_TRUE(encoder.Text(kIdentity, api::StreamText::kContent, "between", output.sink));
  }
  ASSERT_TRUE(encoder.End(kIdentity, api::StreamStop::kToolUse, kUsage, true, output.sink));
  EXPECT_THAT(output.events[output.events.size() - 3],
              HasSubstr("\"finish_reason\":\"tool_calls\""));
  output.ValidateJson();
}

TEST(StreamEncoder, RejectsBrokenToolTransitionsAndInvalidUtf8WithoutEmitting) {
  api::StreamEncoder encoder(api::StreamRoute::kMessages);
  Capture output;
  ASSERT_TRUE(encoder.Start(kIdentity, kUsage, output.sink));
  EXPECT_FALSE(encoder.ToolInput(kIdentity, "{}", output.sink));
  EXPECT_FALSE(encoder.ToolEnd(output.sink));
  ASSERT_TRUE(encoder.ToolStart(kIdentity, "id", "name", output.sink));
  const auto count = output.events.size();
  EXPECT_FALSE(encoder.ToolStart(kIdentity, "other", "name", output.sink));
  EXPECT_FALSE(encoder.Text(kIdentity, api::StreamText::kContent, "wrong", output.sink));
  EXPECT_FALSE(encoder.End(kIdentity, api::StreamStop::kStop, kUsage, false, output.sink));
  for (const std::string_view invalid : {"\xff", "\xc0\xaf", "\xed\xa0", "\xf4\x90", "\xe2x"}) {
    EXPECT_FALSE(encoder.ToolInput(kIdentity, invalid, output.sink));
  }
  EXPECT_EQ(output.events.size(), count);
  ASSERT_TRUE(encoder.ToolInput(kIdentity, "\xf0\x9f", output.sink));
  EXPECT_FALSE(encoder.ToolEnd(output.sink));
  EXPECT_EQ(output.events.size(), count);
  ASSERT_TRUE(encoder.ToolInput(kIdentity, "\x98\x80", output.sink));
  ASSERT_TRUE(encoder.ToolEnd(output.sink));
  ASSERT_TRUE(encoder.End(kIdentity, api::StreamStop::kLength, kUsage, false, output.sink));
  EXPECT_THAT(output.events[output.events.size() - 2], HasSubstr("\"stop_reason\":\"max_tokens\""));
  output.ValidateJson();
}

TEST(StreamEncoder, ErrorIsTerminalBeforeAndDuringMessagesBlocks) {
  for (const bool admitted : {false, true}) {
    api::StreamEncoder encoder(api::StreamRoute::kMessages);
    Capture output;
    ASSERT_TRUE(
        encoder.Start(kIdentity, admitted ? std::optional(kUsage) : std::nullopt, output.sink));
    if (admitted) {
      ASSERT_TRUE(encoder.ToolStart(kIdentity, "id", "tool", output.sink));
      ASSERT_TRUE(encoder.ToolInput(kIdentity, "\xf0", output.sink));
    }
    const api::Error error{.status = 503, .message = "gone\n\"", .param = {}, .code = {}};
    encoder.Fail(error, output.sink);
    const auto count = output.events.size();
    const auto parsed = json::Parse(output.Data(count - 1));
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed->root().find("error")->find("type")->string(), "overloaded_error");
    EXPECT_EQ(parsed->root().find("error")->find("message")->string(), error.message);
    EXPECT_THAT(output.events.back(), Not(HasSubstr("message_stop")));
    encoder.Keepalive(output.sink);
    encoder.Fail(error, output.sink);
    EXPECT_FALSE(encoder.Start(kIdentity, kUsage, output.sink));
    EXPECT_FALSE(encoder.ToolEnd(output.sink));
    EXPECT_FALSE(encoder.End(kIdentity, api::StreamStop::kStop, kUsage, false, output.sink));
    EXPECT_EQ(output.events.size(), count);
    output.ValidateJson();
  }
}

TEST(StreamEncoder, ToolFragmentsDoNotRetainTheArgumentDocument) {
  api::StreamEncoder encoder(api::StreamRoute::kMessages);
  std::size_t events = 0;
  std::size_t bytes = 0;
  const api::StreamEncoder::Sink sink = [&](std::string_view event) {
    ++events;
    bytes += event.size();
  };
  ASSERT_TRUE(encoder.Start(kIdentity, kUsage, sink));
  ASSERT_TRUE(encoder.ToolStart(kIdentity, "id", "tool", sink));
  ASSERT_TRUE(encoder.ToolInput(kIdentity, "{\"value\":\"", sink));
  const std::string fragment(8192, 'x');
  for (int i = 0; i < 256; ++i) {
    ASSERT_TRUE(encoder.ToolInput(kIdentity, fragment, sink));
  }
  ASSERT_TRUE(encoder.ToolInput(kIdentity, "\"}", sink));
  ASSERT_TRUE(encoder.ToolEnd(sink));
  EXPECT_EQ(events, 261U);
  EXPECT_GT(bytes, fragment.size() * 256);
  // The entire encoder is fixed state, including its UTF-8 carry; an input
  // document's size never becomes a retained allocation between calls.
  static_assert(sizeof(api::StreamEncoder) <= 32);
}
}  // namespace
