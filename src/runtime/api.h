// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's bounded inference routes (D-097, D-100; docs/runtime-serving.md):
// an OpenAI-shaped POST /v1/chat/completions request read from untrusted
// bytes under fixed numeric bounds, and the JSON the route answers with.
// The full front door (client-api-baseline.md) is M5's; this is a subset of
// its Chat Completions profile, vendor-free and CPU-tested.
// Literal /v1/completions additionally accepts raw text/exact IDs and
// unfiltered target likelihoods through legacy echo and vLLM prompt scores.
//
// What a request may hold. `model`, `messages` (system, developer, user and
// assistant text; a string or an array of text parts), `max_tokens` or
// `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
// `seed`, `stop`, `stream` and `stream_options.include_usage` are honored.
// Fields this route knows but does not implement are accepted only at the
// value that means "off" (n = 1, zero penalties, no logprobs, no tools,
// text responses, no audio) and refused otherwise, since ignoring them
// would silently answer a different question; OpenRouter's `transforms` and
// `plugins` are always refused (D-046). A documented set of metadata is
// ignored (user, metadata, prompt_cache_key, safety_identifier,
// service_tier, parallel_tool_calls, store = false). Any other field, at
// the top, in a message, a text part or stream_options, is unknown: it is
// ignored (the owner's 2026-09-28 amendment of D-097), and its name, never
// its value, is counted in IgnoredFields.

#ifndef JITLLM_RUNTIME_API_H_
#define JITLLM_RUNTIME_API_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config/node_config.h"

namespace jitllm::runtime::api {

// The intake bounds, fixed before the route accepts input
// (client-api-baseline.md#shared-correctness-and-limits); the reasons are
// in runtime-serving.md.
inline constexpr std::size_t kMaxHeaderBytes = std::size_t{16} << 10U;  // request line + headers
inline constexpr std::size_t kMaxHeaders = 64;
inline constexpr std::size_t kMaxTargetBytes = 2048;
inline constexpr std::size_t kMaxBodyBytes = std::size_t{16} << 20U;
inline constexpr std::size_t kMaxJsonDepth = 16;
inline constexpr std::size_t kMaxJsonValues = std::size_t{1} << 18U;
inline constexpr std::size_t kMaxMessages = 1024;
inline constexpr std::size_t kMaxMessageBytes = std::size_t{8} << 20U;  // one message's text
inline constexpr std::size_t kMaxContentParts = 64;
inline constexpr std::size_t kMaxModelBytes = 64;
inline constexpr std::size_t kMaxStops = 4;
inline constexpr std::size_t kMaxStopBytes = 128;
inline constexpr std::uint32_t kMaxTokensCeiling = config::kMaxContext;
inline constexpr double kMaxTemperature = 2.0;
inline constexpr std::int64_t kMaxTopK = std::int64_t{1} << 31U;
// Unknown fields: names recorded per request, and a name's recorded bytes.
inline constexpr std::size_t kMaxIgnoredPerRequest = 64;
inline constexpr std::size_t kMaxIgnoredNameBytes = 64;
// Timeouts, connections and the queue, in the server (api_server.h).
inline constexpr std::uint32_t kHeaderTimeoutMs = 10'000;  // from a request's first byte
inline constexpr std::uint32_t kBodyTimeoutMs = 30'000;    // likewise
inline constexpr std::uint32_t kWriteTimeoutMs = 30'000;   // output pending without progress
inline constexpr std::uint32_t kIdleTimeoutMs = 60'000;  // a kept-alive connection between requests
inline constexpr std::uint32_t kKeepaliveMs = 15'000;    // SSE comment interval
inline constexpr std::uint32_t kQueueWaitMs = 120'000;   // a non-streaming request's
inline constexpr std::uint32_t kMaintenanceMs = 250;     // Backend::Maintain while idle
// A running request has no fixed deadline: the watchdog and a non-streaming
// request's scaled deadline (watchdog.h) take its place.
inline constexpr std::size_t kMaxQueued = 64;  // config::kDefaultMaxQueued
inline constexpr std::size_t kMaxConnections = 1024;
inline constexpr std::size_t kMaxUnsentBytes = std::size_t{1} << 20U;    // a stream's, per client
inline constexpr std::size_t kBodyBudgetBytes = std::size_t{64} << 20U;  // bodies arriving at once
inline constexpr std::uint32_t kRetryAfterSeconds = 10;

// An OpenAI-shaped error: the HTTP status and the body's error object.
struct Error {
  int status = 400;
  std::string type = "invalid_request_error";
  std::string message;
  std::string param;  // empty: null
  std::string code;   // empty: null
};

enum class Role : std::uint8_t { kSystem, kUser, kAssistant };

struct Message {
  Role role = Role::kUser;               // "developer" is read as system
  std::string content;                   // text parts joined
  std::optional<std::string> reasoning;  // an assistant's reasoning, sent back
};

struct ChatRequest {
  std::string model;
  std::vector<Message> messages;  // 1 to kMaxMessages, the last the user's
  std::optional<std::uint32_t> max_tokens;
  double temperature = 1.0;  // [0, 2]; 0 is greedy (OpenAI's default is 1)
  double top_p = 1.0;        // (0, 1]
  std::uint32_t top_k = 0;   // 0 (or -1 sent): off
  double min_p = 0.0;        // [0, 1]; 0: off
  std::optional<std::uint64_t> seed;
  std::vector<std::string> stop;  // matched against the answer's text
  bool stream = false;
  bool include_usage = false;
  // The unknown fields' names, each once, at most kMaxIgnoredPerRequest:
  // "x" at the top, "messages[].x", "messages[].content[].x",
  // "stream_options.x"; a name cut to kMaxIgnoredNameBytes at a character.
  std::vector<std::string> ignored;
};

// Literal /v1/completions, with one text prompt or one exact token-ID
// sequence. No chat template is applied. Token IDs never acquire a BOS.
// Scoring and its response are bounded independently of the model context.
inline constexpr std::size_t kMaxCompletionScoreRows = 131072;
inline constexpr std::size_t kMaxCompletionResponseBytes = std::size_t{64} << 20U;
inline constexpr std::uint32_t kMaxCompletionTopLogprobs = 5;
struct CompletionRequest {
  ChatRequest options;  // the shared model/sampling/stop controls; no messages
  std::optional<std::string> prompt;
  std::vector<std::int32_t> token_ids;
  bool echo = false;
  bool add_special_tokens = true;  // text only: tokenizer's add-BOS policy, never EOS
  bool return_tokens_as_token_ids = false;
  std::optional<std::uint32_t> logprobs;
  std::optional<std::uint32_t> prompt_logprobs;
};

// Parses and checks a request body (JSON), before any model work.
std::expected<ChatRequest, Error> ParseChatRequest(std::string_view body);
std::expected<CompletionRequest, Error> ParseCompletionRequest(std::string_view body);

// The unknown fields requests have carried, by name (never a value;
// D-014): how often, and when first and last (Unix seconds). At most
// kMaxNames names; later new names are only counted. Thread-safe.
class IgnoredFields {
 public:
  static constexpr std::size_t kMaxNames = 256;

  // Counts a request's names; returns those never seen before (to log
  // once each).
  std::vector<std::string> Record(const std::vector<std::string>& names, std::int64_t now);
  // {"object":"list","data":[{"name","count","first_seen","last_seen"}],
  //  "unrecorded":N}, names in order.
  std::string Json() const;

 private:
  struct Entry {
    std::uint64_t count = 0;
    std::int64_t first = 0;
    std::int64_t last = 0;
  };
  mutable std::mutex mutex_;
  std::map<std::string, Entry> names_;
  std::uint64_t unrecorded_ = 0;  // occurrences of names past kMaxNames
};

// ---------------------------------------------------------------- output

enum class Finish : std::uint8_t { kStop, kLength };

struct Usage {
  std::uint32_t prompt_tokens = 0;
  std::uint32_t completion_tokens = 0;
  std::uint32_t cached_tokens = 0;  // the prompt's tokens the conversation state already held
};

struct TokenLogprob {
  struct Ranked {
    std::int32_t id = 0;
    std::string token;
    double logprob = 0;
    std::uint32_t rank = 0;
  };
  std::int32_t id = 0;
  std::string token;
  std::optional<double> logprob;  // null only for the first supplied token
  std::vector<Ranked> top;
  std::size_t text_offset = 0;  // Unicode characters in the decoded text
};
struct LiteralResult {
  std::string prompt_text;
  std::vector<TokenLogprob> logprobs;
  std::vector<TokenLogprob> prompt_logprobs;
};

// {"error":{...}}
std::string ErrorJson(const Error& error);
// A non-streaming response (object "chat.completion").
std::string CompletionJson(std::string_view id, std::int64_t created, std::string_view model,
                           std::string_view content, const std::optional<std::string>& reasoning,
                           Finish finish, const Usage& usage);
// Legacy OpenAI arrays and the vLLM prompt_logprobs extension. The result's
// scores come from the natural target distribution, before sampling filters.
std::expected<std::string, Error> LiteralCompletionJson(std::string_view id, std::int64_t created,
                                                        const CompletionRequest& request,
                                                        std::string_view content,
                                                        const LiteralResult& result, Finish finish,
                                                        const Usage& usage);
// One streamed chunk (object "chat.completion.chunk"): a delta of the role,
// the reasoning or the content, or with `finish` the last one.
enum class Delta : std::uint8_t { kRole, kReasoning, kContent, kFinish };
std::string ChunkJson(std::string_view id, std::int64_t created, std::string_view model,
                      Delta delta, std::string_view text, Finish finish = Finish::kStop);
// The usage chunk (stream_options.include_usage): empty choices.
std::string UsageChunkJson(std::string_view id, std::int64_t created, std::string_view model,
                           const Usage& usage);

// A model on the route's list (GET /v1/models).
struct ModelInfo {
  std::string name;
  bool chat = true;  // false: an image pipeline, which this route does not serve
  std::uint32_t context = 0;
};
std::string ModelJson(const ModelInfo& model, std::int64_t created);
std::string ModelsJson(const std::vector<ModelInfo>& models, std::int64_t created);

// The generated text on its way out: the reasoning (before the template's
// end-of-reasoning marker) and the answer, with the request's stop strings
// matched in the answer. What might still become a stop string is held
// back until it cannot; the answer's leading whitespace after reasoning is
// dropped, as the chat command does.
class OutputText {
 public:
  explicit OutputText(std::vector<std::string> stops) : stops_(std::move(stops)) {}

  struct Out {
    std::string reasoning;
    std::string content;
  };
  // Adds generated text; returns what may be sent now. After a stop
  // string, stopped() is true and nothing more is taken. Reasoning with
  // empty text marks a reasoning block that said nothing, so the answer's
  // leading whitespace is still dropped.
  Out Reasoning(std::string_view text);
  Out Content(std::string_view text);
  // At the end: whatever was held back.
  Out Finish();
  bool stopped() const { return stopped_; }

 private:
  std::vector<std::string> stops_;
  std::string held_;  // answer text that may begin a stop string
  bool had_reasoning_ = false;
  bool content_started_ = false;
  bool stopped_ = false;
};

}  // namespace jitllm::runtime::api

#endif  // JITLLM_RUNTIME_API_H_
