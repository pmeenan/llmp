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

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config/node_config.h"
#include "runtime/intake_limits.h"

namespace jitllm::runtime::api {

// The intake bounds (client-api-baseline.md#shared-correctness-and-limits);
// the resource each protects is in runtime-serving.md. Only real resources
// bound a request (D-102): a body's bytes (the server's, from memory or
// [client] max_body_bytes, intake_limits.h), the parser's stack, a
// header's buffer, the model's context and vocabulary, and the types'
// ranges. Messages, content parts and stop strings have no count; their
// bytes are the body's.
inline constexpr std::size_t kMaxHeaderBytes = std::size_t{16} << 10U;  // request line + headers
inline constexpr std::size_t kMaxHeaders = 64;
inline constexpr std::size_t kMaxTargetBytes = 2048;
// The JSON parser's recursion: a stack bound, not a size bound.
inline constexpr std::size_t kMaxJsonDepth = 64;
inline constexpr std::size_t kMaxModelBytes = 64;
// max_tokens at parse: the type's; the model's usable context decides.
inline constexpr std::uint32_t kMaxTokensCeiling = 0xFFFF'FFFFU;
inline constexpr double kMaxTemperature = 2.0;
inline constexpr std::int64_t kMaxTopK = std::int64_t{1} << 31U;
// Unknown fields: names recorded per request, and a name's recorded bytes.
inline constexpr std::size_t kMaxIgnoredPerRequest = 64;
inline constexpr std::size_t kMaxIgnoredNameBytes = 64;
// The SSE comment interval, Backend::Maintain's interval while idle, and
// the Retry-After of a 429 or 503. The server's timeouts, queue and
// connection limits are its options (api_server.h), each from [client].
inline constexpr std::uint32_t kKeepaliveMs = 15'000;
inline constexpr std::uint32_t kMaintenanceMs = 250;
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

// A request's stop strings, matched together (Aho-Corasick): a trie of
// their bytes with failure links, so the output's matching costs a few
// steps a byte however many stop strings there are, and duplicates and
// shared prefixes take no more room. Built once when the request is
// parsed and shared by its output; immutable.
class StopMatcher {
 public:
  // From the stop strings (each non-empty), in any order. With `charge`,
  // the tables are charged to `memory` as they grow (a node a distinct
  // prefix, so shared prefixes and duplicates cost nothing more); when a
  // growth does not fit, the bytes it needed.
  static std::expected<StopMatcher, std::uint64_t> Build(std::span<const std::string_view> stops,
                                                         RequestMemory* memory,
                                                         MemoryCharge* charge);
  static StopMatcher Build(std::span<const std::string_view> stops);
  // The bytes Build allocates for `count` stop strings of `bytes` bytes in
  // all, at most: a node a byte, grown by doubling, and its queue.
  static std::uint64_t BuildBytes(std::size_t count, std::size_t bytes);

  bool empty() const { return nodes_.size() <= 1; }
  std::size_t count() const { return count_; }      // stop strings, duplicates included
  std::size_t longest() const { return longest_; }  // the longest's bytes
  std::size_t bytes() const;                        // the tables' allocation

  // From `state` (0: nothing matched) past `byte`.
  std::uint32_t Next(std::uint32_t state, unsigned char byte) const;
  // The longest prefix of a stop string the text so far ends with.
  std::uint32_t depth(std::uint32_t state) const { return nodes_[state].depth; }
  // The longest stop string the text so far ends with (0: none).
  std::uint32_t matched(std::uint32_t state) const { return nodes_[state].out; }

 private:
  struct Node {
    std::uint32_t child = 0;    // the first child (0: none; the root is never one)
    std::uint32_t sibling = 0;  // the next child of the same parent, by byte
    std::uint32_t fail = 0;     // the longest proper suffix that is a node
    std::uint32_t depth = 0;
    std::uint32_t out = 0;  // the longest stop string ending here or along `fail`
    unsigned char byte = 0;
  };
  std::uint32_t Child(std::uint32_t node, unsigned char byte) const;

  std::vector<Node> nodes_;
  std::array<std::uint32_t, 256> root_{};  // the root's children by byte (0: none)
  std::size_t count_ = 0;
  std::size_t longest_ = 0;
};

// A model's sampling defaults (its settings, D-103): what a request that
// sends none of a field samples with.
struct SamplingDefaults {
  double temperature = 1.0;
  double top_p = 1.0;
  std::uint32_t top_k = 0;
  double min_p = 0.0;
};

struct ChatRequest {
  std::string model;
  std::vector<Message> messages;  // at least one, the last the user's
  std::optional<std::uint32_t> max_tokens;
  // Each as sent; one not sent takes the model's default (Sampling).
  double temperature = 1.0;  // [0, 2]; 0 is greedy (OpenAI's default is 1)
  double top_p = 1.0;        // (0, 1]
  std::uint32_t top_k = 0;   // 0 (or -1 sent): off
  double min_p = 0.0;        // [0, 1]; 0: off
  // Which of the four the request sent (kSent* bits).
  std::uint8_t sent = 0;
  static constexpr std::uint8_t kSentTemperature = 1;
  static constexpr std::uint8_t kSentTopP = 2;
  static constexpr std::uint8_t kSentTopK = 4;
  static constexpr std::uint8_t kSentMinP = 8;
  // The request's sampling: each field it sent, else `defaults`'.
  SamplingDefaults Sampling(const SamplingDefaults& defaults) const {
    return {.temperature = (sent & kSentTemperature) != 0 ? temperature : defaults.temperature,
            .top_p = (sent & kSentTopP) != 0 ? top_p : defaults.top_p,
            .top_k = (sent & kSentTopK) != 0 ? top_k : defaults.top_k,
            .min_p = (sent & kSentMinP) != 0 ? min_p : defaults.min_p};
  }
  std::optional<std::uint64_t> seed;
  // Matched against the answer's text; shared, never copied (null: none).
  std::shared_ptr<const StopMatcher> stop;
  bool stream = false;
  bool include_usage = false;
  // The unknown fields' names, each once, at most kMaxIgnoredPerRequest:
  // "x" at the top, "messages[].x", "messages[].content[].x",
  // "stream_options.x"; a name cut to kMaxIgnoredNameBytes at a character.
  std::vector<std::string> ignored;
};

// Literal /v1/completions, with one text prompt or one exact token-ID
// sequence. No chat template is applied. Token IDs never acquire a BOS.
// Score rows are bounded by the model's context (prompt plus output) and
// with their top scores by the response's bytes (the request memory's,
// intake_limits.h); top scores at parse by the
// largest vocabulary a tokenizer accepts, then by the model's own.
inline constexpr std::uint32_t kMaxTopLogprobsAtParse = std::uint32_t{1} << 22U;
// The response bytes a score row and each of its top scores are charged
// (twice: prompt rows can appear in both response forms), which admission
// figures before any work.
inline constexpr std::size_t kScoreRowBytes = 256;
inline constexpr std::size_t kTopScoreBytes = 128;
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

// Parses and checks a request body (JSON), before any model work. With
// `memory`, the parse's working set (json::ParseWorkingBytes and the
// request it builds) and the stop matcher's building are charged to it
// while they last, and a request that cannot fit is refused (413 past the
// whole pool, else 503) before the work.
std::expected<ChatRequest, Error> ParseChatRequest(std::string_view body,
                                                   RequestMemory* memory = nullptr);
std::expected<CompletionRequest, Error> ParseCompletionRequest(std::string_view body,
                                                               RequestMemory* memory = nullptr);

// The bytes a parsed request holds (its strings, prompt IDs and stop
// matcher, by allocation), which the server charges while it lives.
std::uint64_t RequestBytes(const ChatRequest& request);
std::uint64_t RequestBytes(const CompletionRequest& request);

// The refusal for a charge of `bytes` that `memory` could not take: a 413
// when it is more than the whole pool, a 503 (retry) otherwise; `what`
// says what needed it.
Error MemoryRefusal(const RequestMemory& memory, std::uint64_t bytes, std::string_view what);

// Immutable prepared prompt IDs shared by active work and continuations.
// Their charge follows the allocation until its last owner releases it.
class PromptTokens {
 public:
  static std::expected<std::shared_ptr<const PromptTokens>, Error> Hold(
      std::vector<std::int32_t> tokens, RequestMemory& memory);
  const std::vector<std::int32_t>& values() const { return tokens_; }

 private:
  std::vector<std::int32_t> tokens_;
  MemoryCharge charge_;
};

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
// The body is at most `max_bytes` (the request memory's capacity).
std::expected<std::string, Error> LiteralCompletionJson(std::string_view id, std::int64_t created,
                                                        const CompletionRequest& request,
                                                        std::string_view content,
                                                        const LiteralResult& result, Finish finish,
                                                        const Usage& usage, std::size_t max_bytes);
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
  // The most text a prompt that fits the context could render to
  // (runtime/intake_limits.h RenderBytes; 0: unknown): a request whose text
  // is longer is refused before it is queued (D-102).
  std::uint64_t render_bytes = 0;
};
std::string ModelJson(const ModelInfo& model, std::int64_t created);
std::string ModelsJson(const std::vector<ModelInfo>& models, std::int64_t created);

// The generated text on its way out: the reasoning (before the template's
// end-of-reasoning marker) and the answer, with the request's stop strings
// matched in the answer. What might still become a stop string is held
// back until it cannot (at most the longest stop string, less a byte); the
// answer's leading whitespace after reasoning is dropped, as the chat
// command does. The stop strings are matched together, incrementally
// (StopMatcher), so the work is a few steps a byte whatever their number
// and lengths (D-102 removed their count and length caps; the body and the
// request memory bound their bytes).
class OutputText {
 public:
  // The request's shared matcher (null: no stop strings).
  explicit OutputText(std::shared_ptr<const StopMatcher> stops) : stops_(std::move(stops)) {}
  // Builds a matcher from the strings (tests, commands).
  explicit OutputText(const std::vector<std::string>& stops);

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
  std::shared_ptr<const StopMatcher> stops_;
  std::uint32_t state_ = 0;  // the matcher's, over the answer so far
  // The unconsumed tail may begin a stop string. An emitted prefix stays
  // allocated until compacting it moves at most as many bytes as emitted.
  std::string held_;
  std::size_t held_start_ = 0;
  bool had_reasoning_ = false;
  bool content_started_ = false;
  bool stopped_ = false;
};

}  // namespace jitllm::runtime::api

#endif  // JITLLM_RUNTIME_API_H_
