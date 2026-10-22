// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/stream_encoder.h"

#include <algorithm>
#include <format>
#include <limits>

#include "base/json.h"

namespace llmp::runtime::api {
namespace {
std::string Quote(std::string_view text) {
  std::string result;
  base::json::AppendQuoted(text, result);
  return result;
}
void Event(std::string_view name, std::string_view json, const StreamEncoder::Sink& sink) {
  std::string event;
  if (!name.empty()) {
    event = std::format("event: {}\n", name);
  }
  event += "data: ";
  event += json;
  event += "\n\n";
  sink(event);
}
}  // namespace

StreamEncoder::Status StreamEncoder::Start(const StreamIdentity& identity,
                                           std::optional<Usage> usage, const Sink& sink) {
  if (ended_) {
    return std::unexpected("stream already ended");
  }
  if (started_) {
    return {};
  }
  if (route_ == StreamRoute::kChatCompletions) {
    Event({}, ChunkJson(identity.id, identity.created, identity.model, Delta::kRole, {}), sink);
  } else {
    if (!usage) {
      Keepalive(sink);
      return {};
    }
    if (usage->cached_tokens > usage->prompt_tokens) {
      return std::unexpected("cached prompt tokens exceed input tokens");
    }
    Event("message_start",
          std::format(
              R"({{"type":"message_start","message":{{"id":{},"type":"message",)"
              R"("role":"assistant","model":{},"content":[],"stop_reason":null,)"
              R"("stop_sequence":null,"usage":{{"input_tokens":{},"cache_read_input_tokens":{},)"
              R"("output_tokens":0}}}}}})",
              Quote(identity.id), Quote(identity.model),
              usage->prompt_tokens - usage->cached_tokens, usage->cached_tokens),
          sink);
  }
  started_ = true;
  return {};
}

void StreamEncoder::CloseBlock(const Sink& sink) {
  if (block_ != Block::kNone) {
    if (route_ == StreamRoute::kMessages) {
      Event("content_block_stop",
            std::format(R"({{"type":"content_block_stop","index":{}}})", index_), sink);
    }
    ++index_;  // guarded when a block starts
    block_ = Block::kNone;
  }
}

StreamEncoder::Status StreamEncoder::Text(const StreamIdentity& identity, StreamText kind,
                                          std::string_view text, const Sink& sink) {
  if (!started_ || ended_) {
    return std::unexpected("text requires an active stream");
  }
  if (block_ == Block::kTool) {
    return std::unexpected("text cannot interrupt a tool input");
  }
  if (route_ == StreamRoute::kChatCompletions) {
    if (!text.empty()) {
      Event({},
            ChunkJson(identity.id, identity.created, identity.model,
                      kind == StreamText::kContent ? Delta::kContent : Delta::kReasoning, text),
            sink);
    }
    return {};
  }
  if (kind == StreamText::kReasoning) {
    return std::unexpected("Messages reasoning requires signed thinking blocks");
  }
  if (text.empty()) {
    return {};
  }
  if (block_ == Block::kNone) {
    if (index_ == std::numeric_limits<std::uint64_t>::max()) {
      return std::unexpected("content block index overflows");
    }
    Event("content_block_start",
          std::format(R"({{"type":"content_block_start","index":{},)"
                      R"("content_block":{{"type":"text","text":""}}}})",
                      index_),
          sink);
    block_ = Block::kText;
  }
  Event("content_block_delta",
        std::format(R"({{"type":"content_block_delta","index":{},)"
                    R"("delta":{{"type":"text_delta","text":{}}}}})",
                    index_, Quote(text)),
        sink);
  return {};
}

std::string StreamEncoder::ToolDelta(const StreamIdentity& identity, std::string_view delta) const {
  return std::format(R"({{"id":{},"object":"chat.completion.chunk","created":{},"model":{},)"
                     R"("choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},{}}}]}},)"
                     R"("logprobs":null,"finish_reason":null}}]}})",
                     Quote(identity.id), identity.created, Quote(identity.model), index_, delta);
}

StreamEncoder::Status StreamEncoder::ToolStart(const StreamIdentity& identity, std::string_view id,
                                               std::string_view name, const Sink& sink) {
  if (!started_ || ended_ || block_ == Block::kTool) {
    return std::unexpected("tool start requires an active stream without a tool input");
  }
  // Closing a text block consumes an index before the tool's own index.
  if (index_ >= std::numeric_limits<std::uint64_t>::max() - (block_ == Block::kText ? 1U : 0U)) {
    return std::unexpected("content block index overflows");
  }
  CloseBlock(sink);
  if (route_ == StreamRoute::kMessages) {
    Event("content_block_start",
          std::format(R"({{"type":"content_block_start","index":{},)"
                      R"("content_block":{{"type":"tool_use","id":{},"name":{},"input":{{}}}}}})",
                      index_, Quote(id), Quote(name)),
          sink);
  } else {
    Event({},
          ToolDelta(identity, std::format(R"("id":{},"type":"function",)"
                                          R"("function":{{"name":{},"arguments":""}})",
                                          Quote(id), Quote(name))),
          sink);
  }
  block_ = Block::kTool;
  return {};
}

StreamEncoder::Status StreamEncoder::ToolInput(const StreamIdentity& identity,
                                               std::string_view fragment, const Sink& sink) {
  if (!started_ || ended_ || block_ != Block::kTool) {
    return std::unexpected("tool input requires an open tool block");
  }
  std::string joined(tail_.data(), tail_size_);
  joined += fragment;
  std::size_t complete = joined.size();
  if (const auto invalid = base::json::FirstInvalidUtf8(joined)) {
    // The only accepted invalid suffix is a valid but incomplete character.
    const std::string_view suffix = std::string_view(joined).substr(*invalid);
    const auto lead = static_cast<unsigned char>(suffix[0]);
    const std::size_t width = lead >= 0xc2 && lead <= 0xdf   ? 2U
                              : lead >= 0xe0 && lead <= 0xef ? 3U
                              : lead >= 0xf0 && lead <= 0xf4 ? 4U
                                                             : 0U;
    if (width == 0 || suffix.size() >= width ||
        !std::ranges::all_of(suffix.substr(1),
                             [](unsigned char c) { return c >= 0x80 && c <= 0xbf; }) ||
        (suffix.size() >= 2 && ((lead == 0xe0 && static_cast<unsigned char>(suffix[1]) < 0xa0) ||
                                (lead == 0xed && static_cast<unsigned char>(suffix[1]) >= 0xa0) ||
                                (lead == 0xf0 && static_cast<unsigned char>(suffix[1]) < 0x90) ||
                                (lead == 0xf4 && static_cast<unsigned char>(suffix[1]) >= 0x90)))) {
      return std::unexpected("tool input contains invalid UTF-8");
    }
    complete = *invalid;
  }
  tail_size_ = static_cast<std::uint8_t>(joined.size() - complete);
  std::ranges::copy(std::string_view(joined).substr(complete), tail_.begin());
  const std::string_view text(joined.data(), complete);
  if (text.empty()) {
    return {};
  }
  if (route_ == StreamRoute::kMessages) {
    Event("content_block_delta",
          std::format(R"({{"type":"content_block_delta","index":{},)"
                      R"("delta":{{"type":"input_json_delta","partial_json":{}}}}})",
                      index_, Quote(text)),
          sink);
  } else {
    Event({}, ToolDelta(identity, std::format(R"("function":{{"arguments":{}}})", Quote(text))),
          sink);
  }
  return {};
}

StreamEncoder::Status StreamEncoder::ToolEnd(const Sink& sink) {
  if (!started_ || ended_ || block_ != Block::kTool) {
    return std::unexpected("tool end requires an open tool block");
  }
  if (tail_size_ != 0) {
    return std::unexpected("tool input ends inside a UTF-8 character");
  }
  CloseBlock(sink);
  return {};
}

StreamEncoder::Status StreamEncoder::End(const StreamIdentity& identity, StreamStop stop,
                                         const Usage& usage, bool include_usage, const Sink& sink) {
  if (!started_ || ended_ || block_ == Block::kTool) {
    return std::unexpected("finish requires an active stream without an open tool input");
  }
  if (route_ == StreamRoute::kMessages && usage.cached_tokens > usage.prompt_tokens) {
    return std::unexpected("cached prompt tokens exceed input tokens");
  }
  CloseBlock(sink);
  if (route_ == StreamRoute::kChatCompletions) {
    if (stop == StreamStop::kToolUse) {
      // The tool finish shares the chunk envelope, but no tool delta.
      Event(
          {},
          std::format(
              R"({{"id":{},"object":"chat.completion.chunk","created":{},"model":{},)"
              R"("choices":[{{"index":0,"delta":{{}},"logprobs":null,"finish_reason":"tool_calls"}}]}})",
              Quote(identity.id), identity.created, Quote(identity.model)),
          sink);
    } else {
      Event({},
            ChunkJson(identity.id, identity.created, identity.model, Delta::kFinish, {},
                      stop == StreamStop::kStop ? Finish::kStop : Finish::kLength),
            sink);
    }
    if (include_usage) {
      Event({}, UsageChunkJson(identity.id, identity.created, identity.model, usage), sink);
    }
    sink("data: [DONE]\n\n");
  } else {
    const std::string_view reason = stop == StreamStop::kStop     ? "end_turn"
                                    : stop == StreamStop::kLength ? "max_tokens"
                                                                  : "tool_use";
    Event("message_delta",
          std::format(
              R"({{"type":"message_delta","delta":{{"stop_reason":{},)"
              R"("stop_sequence":null}},"usage":{{"input_tokens":{},"cache_read_input_tokens":{},)"
              R"("output_tokens":{}}}}})",
              Quote(reason), usage.prompt_tokens - usage.cached_tokens, usage.cached_tokens,
              usage.completion_tokens),
          sink);
    Event("message_stop", R"({"type":"message_stop"})", sink);
  }
  ended_ = true;
  return {};
}

void StreamEncoder::Keepalive(const Sink& sink) const {
  if (ended_) {
    return;
  }
  if (route_ == StreamRoute::kMessages) {
    Event("ping", R"({"type":"ping"})", sink);
  } else {
    sink(": keepalive\n\n");
  }
}

std::string StreamEncoder::ErrorBody(const Error& error) const {
  if (route_ == StreamRoute::kChatCompletions) {
    return ErrorJson(error);
  }
  const std::string_view type = error.status == 401   ? "authentication_error"
                                : error.status == 403 ? "permission_error"
                                : error.status == 404 ? "not_found_error"
                                : error.status == 413 ? "request_too_large"
                                : error.status == 429 ? "rate_limit_error"
                                : error.status == 503 ? "overloaded_error"
                                : error.status >= 500 ? "api_error"
                                                      : "invalid_request_error";
  return std::format(R"({{"type":"error","error":{{"type":{},"message":{}}}}})", Quote(type),
                     Quote(error.message));
}

void StreamEncoder::Fail(const Error& error, const Sink& sink) {
  if (!ended_) {
    Event(route_ == StreamRoute::kMessages ? "error" : "", ErrorBody(error), sink);
    ended_ = true;
    tail_size_ = 0;
  }
}
}  // namespace llmp::runtime::api
