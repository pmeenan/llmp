// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Route-owned SSE serialization. The sink consumes one complete event during
// the call; neither it nor input views are retained. HTTP chunking, request
// memory, backpressure and cancellation remain the channel's responsibility.
// Messages is an encoder seam, not an enabled HTTP route. Its request parser,
// tool extraction, signed reasoning and non-streaming response are still owed.
#ifndef LLMP_RUNTIME_STREAM_ENCODER_H_
#define LLMP_RUNTIME_STREAM_ENCODER_H_

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "runtime/api.h"

namespace llmp::runtime::api {

enum class StreamRoute : std::uint8_t { kChatCompletions, kMessages };
enum class StreamText : std::uint8_t { kContent, kReasoning };
enum class StreamStop : std::uint8_t { kStop, kLength, kToolUse };

struct StreamIdentity {
  std::string_view id;
  std::string_view model;
  std::int64_t created = 0;
};

class StreamEncoder {
 public:
  using Sink = std::function<void(std::string_view)>;
  using Status = std::expected<void, std::string_view>;

  explicit StreamEncoder(StreamRoute route = StreamRoute::kChatCompletions) : route_(route) {}

  // Idempotent: the queue may open the stream before the driver admits it.
  // Messages sends pings until prompt usage is known, then message_start;
  // OpenAI sends its role immediately. Usage here is the admitted prompt's.
  Status Start(const StreamIdentity& identity, std::optional<Usage> usage, const Sink& sink);
  // Text comes from the tokenizer's complete UTF-8 output; tool inputs may
  // instead split a UTF-8 character between arbitrary JSON fragments.
  Status Text(const StreamIdentity& identity, StreamText kind, std::string_view text,
              const Sink& sink);
  // Tools are serialized one block at a time. The extractor owns JSON syntax
  // and tool identity. Input forwards fragments, without parsing or retaining
  // the whole JSON document; at most three trailing UTF-8 bytes are retained.
  Status ToolStart(const StreamIdentity& identity, std::string_view id, std::string_view name,
                   const Sink& sink);
  Status ToolInput(const StreamIdentity& identity, std::string_view fragment, const Sink& sink);
  Status ToolEnd(const Sink& sink);
  Status End(const StreamIdentity& identity, StreamStop stop, const Usage& usage,
             bool include_usage, const Sink& sink);
  void Keepalive(const Sink& sink) const;
  // A terminal error emits no successful finish marker. Also works before
  // Start (e.g. Messages sent queue pings but admission subsequently failed).
  void Fail(const Error& error, const Sink& sink);
  std::string ErrorBody(const Error& error) const;

 private:
  enum class Block : std::uint8_t { kNone, kText, kTool };
  void CloseBlock(const Sink& sink);
  std::string ToolDelta(const StreamIdentity& identity, std::string_view delta) const;
  StreamRoute route_;
  bool started_ = false;
  bool ended_ = false;
  Block block_ = Block::kNone;
  std::uint64_t index_ = 0;  // Messages content block, OpenAI tool call
  std::array<char, 3> tail_{};
  std::uint8_t tail_size_ = 0;
};

}  // namespace llmp::runtime::api
#endif  // LLMP_RUNTIME_STREAM_ENCODER_H_
