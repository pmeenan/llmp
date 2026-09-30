// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"

namespace jitllm::runtime::api {
namespace {

namespace json = base::json;

std::unexpected<Error> Bad(std::string message, std::string param = {}, std::string code = {}) {
  return std::unexpected(Error{.status = 400,
                               .type = "invalid_request_error",
                               .message = std::move(message),
                               .param = std::move(param),
                               .code = std::move(code)});
}

std::string Quoted(std::string_view text) {
  std::string out;
  json::AppendQuoted(text, out);
  return out;
}

// A number that must equal `value` (a parameter accepted only at the
// value that means "off").
bool NumberIs(const json::Value& v, double value) {
  if (!v.is_number()) {
    return false;
  }
  const std::optional<double> x = v.float64();
  return x && *x == value;
}

// The unknown fields' names a request carries (ChatRequest::ignored).
class Ignored {
 public:
  explicit Ignored(std::vector<std::string>& names) : names_(names) {}

  void Add(std::string_view prefix, std::string_view key) {
    if (names_.size() >= kMaxIgnoredPerRequest) {
      return;
    }
    std::string name(prefix);
    name += key;
    if (name.size() > kMaxIgnoredNameBytes) {
      // Cut at a character: never inside a UTF-8 sequence (the parser
      // checked the key's encoding).
      std::size_t cut = kMaxIgnoredNameBytes;
      while (cut > 0 && (static_cast<unsigned char>(name[cut]) & 0xC0U) == 0x80U) {
        --cut;
      }
      name.resize(cut);
    }
    if (std::ranges::find(names_, name) == names_.end()) {
      names_.push_back(std::move(name));
    }
  }

 private:
  std::vector<std::string>& names_;
};

// A message's content: a string, or text parts joined.
std::expected<std::string, Error> Content(const json::Value& content, const std::string& at,
                                          Ignored& ignored) {
  if (content.is_string()) {
    return std::string(content.string());
  }
  if (!content.is_array()) {
    return Bad(std::format("{}.content must be a string or an array of text parts", at),
               at + ".content");
  }
  if (content.size() > kMaxContentParts) {
    return Bad(
        std::format("{}.content has {} parts, more than {}", at, content.size(), kMaxContentParts),
        at + ".content");
  }
  std::string text;
  for (std::size_t j = 0; j < content.size(); ++j) {
    const json::Value part = content.at(j);
    const std::string where = std::format("{}.content[{}]", at, j);
    if (!part.is_object()) {
      return Bad(where + " must be an object", where);
    }
    auto type = part.find("type");
    if (!type || !type->is_string()) {
      return Bad(where + ".type must be a string", where + ".type");
    }
    if (type->string() != "text") {
      return Bad(
          std::format("{}.type is {}: this route takes text only", where, Quoted(type->string())),
          where + ".type", "unsupported_content_type");
    }
    for (std::size_t k = 0; k < part.size(); ++k) {
      const std::string_view key = part.key(k);
      if (key == "type" || key == "text" || key == "cache_control") {
        continue;  // cache_control: an advisory retention hint (D-045), ignored
      }
      ignored.Add("messages[].content[].", key);
    }
    auto piece = part.find("text");
    if (!piece || !piece->is_string()) {
      return Bad(where + ".text must be a string", where + ".text");
    }
    if (text.size() + piece->string().size() > kMaxMessageBytes) {
      return Bad(std::format("{} is longer than {} bytes", at, kMaxMessageBytes), at);
    }
    text += piece->string();
  }
  return text;
}

std::expected<Message, Error> ParseMessage(const json::Value& m, std::size_t index,
                                           Ignored& ignored) {
  const std::string at = std::format("messages[{}]", index);
  if (!m.is_object()) {
    return Bad(at + " must be an object", at);
  }
  Message out;
  auto role = m.find("role");
  if (!role || !role->is_string()) {
    return Bad(at + ".role must be a string", at + ".role");
  }
  const std::string_view r = role->string();
  if (r == "system" || r == "developer") {
    out.role = Role::kSystem;
  } else if (r == "user") {
    out.role = Role::kUser;
  } else if (r == "assistant") {
    out.role = Role::kAssistant;
  } else if (r == "tool" || r == "function") {
    return Bad(
        std::format("{}.role is {}: tools are not supported by this route yet", at, Quoted(r)),
        at + ".role");
  } else {
    return Bad(std::format("{}.role is {}; it must be system, developer, user or assistant", at,
                           Quoted(r)),
               at + ".role");
  }
  const bool assistant = out.role == Role::kAssistant;
  bool has_content = false;
  for (std::size_t k = 0; k < m.size(); ++k) {
    const std::string_view key = m.key(k);
    const json::Value v = m.member(k);
    const std::string where = std::format("{}.{}", at, key);
    if (key == "role") {
      continue;
    }
    if (key == "content") {
      if (v.is_null() && assistant) {
        has_content = true;
        continue;
      }
      auto text = Content(v, at, ignored);
      if (!text) {
        return std::unexpected(text.error());
      }
      out.content = std::move(*text);
      has_content = true;
    } else if (key == "name") {
      if (!v.is_string()) {
        return Bad(where + " must be a string", where);
      }
      // Ignored: neither served template renders a speaker's name.
    } else if (assistant && (key == "reasoning" || key == "reasoning_content")) {
      if (v.is_null()) {
        continue;
      }
      if (!v.is_string()) {
        return Bad(where + " must be a string", where);
      }
      if (out.reasoning) {
        return Bad(at + " has both reasoning and reasoning_content", where);
      }
      out.reasoning = std::string(v.string());
    } else if (assistant && ((key == "annotations" && v.is_array()) ||
                             ((key == "refusal" || key == "audio" || key == "tool_calls" ||
                               key == "function_call") &&
                              v.is_null()) ||
                             (key == "tool_calls" && v.is_array() && v.size() == 0))) {
      // Response metadata, a response's null fields or an empty tool_calls,
      // which a client echoes back.
      continue;
    } else if (assistant && key == "refusal") {
      return Bad(where + ": a refusal cannot be sent back to this route; send it as content",
                 where);
    } else if (assistant && key == "audio") {
      return Bad(where + ": audio is not supported by this route", where);
    } else if (assistant && (key == "tool_calls" || key == "function_call")) {
      return Bad(where + ": tool calls are not supported by this route yet", where);
    } else {
      ignored.Add("messages[].", key);
    }
  }
  if (!has_content) {
    return Bad(at + ".content is required", at + ".content");
  }
  if (out.content.size() > kMaxMessageBytes ||
      (out.reasoning && out.reasoning->size() > kMaxMessageBytes)) {
    return Bad(std::format("{} is longer than {} bytes", at, kMaxMessageBytes), at);
  }
  return out;
}

std::expected<std::vector<std::string>, Error> ParseStop(const json::Value& v) {
  std::vector<std::string> stops;
  const auto one = [&](const json::Value& s, const std::string& where) -> std::optional<Error> {
    if (!s.is_string()) {
      return Bad(where + " must be a string", where).error();
    }
    if (s.string().empty() || s.string().size() > kMaxStopBytes) {
      return Bad(std::format("{} must be 1 to {} bytes", where, kMaxStopBytes), where).error();
    }
    stops.emplace_back(s.string());
    return std::nullopt;
  };
  if (v.is_string()) {
    if (auto e = one(v, "stop")) {
      return std::unexpected(*e);
    }
    return stops;
  }
  if (!v.is_array()) {
    return Bad("stop must be a string or an array of strings", "stop");
  }
  if (v.size() > kMaxStops) {
    return Bad(std::format("stop has {} strings, more than {}", v.size(), kMaxStops), "stop");
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (auto e = one(v.at(i), std::format("stop[{}]", i))) {
      return std::unexpected(*e);
    }
  }
  return stops;
}

std::expected<std::uint32_t, Error> ParseMaxTokens(const json::Value& v, std::string_view name,
                                                   std::int64_t least = 1) {
  const std::optional<std::int64_t> n = v.int64();
  if (!v.is_integer() || !n || *n < least || std::cmp_greater(*n, kMaxTokensCeiling)) {
    return Bad(std::format("{} must be an integer from {} to {}", name, least, kMaxTokensCeiling),
               std::string(name));
  }
  return static_cast<std::uint32_t>(*n);
}

// The fields this route knows, whose null means their default.
constexpr std::array<std::string_view, 31> kKnown = {"max_tokens",
                                                     "max_completion_tokens",
                                                     "temperature",
                                                     "top_p",
                                                     "top_k",
                                                     "min_p",
                                                     "seed",
                                                     "stop",
                                                     "stream",
                                                     "stream_options",
                                                     "n",
                                                     "presence_penalty",
                                                     "frequency_penalty",
                                                     "repetition_penalty",
                                                     "logprobs",
                                                     "top_logprobs",
                                                     "tools",
                                                     "tool_choice",
                                                     "functions",
                                                     "function_call",
                                                     "response_format",
                                                     "logit_bias",
                                                     "modalities",
                                                     "audio",
                                                     "user",
                                                     "safety_identifier",
                                                     "prompt_cache_key",
                                                     "metadata",
                                                     "service_tier",
                                                     "parallel_tool_calls",
                                                     "store"};

std::expected<ChatRequest, Error> ParseRequest(std::string_view body, CompletionRequest* literal) {
  const json::Limits limits{.max_bytes = kMaxBodyBytes,
                            .max_depth = kMaxJsonDepth,
                            .max_values = kMaxJsonValues,
                            .max_string_bytes = kMaxBodyBytes};
  auto document = json::Parse(body, limits);
  if (!document) {
    return Bad(std::format("the body is not valid JSON: {}", document.error().ToString()));
  }
  const json::Value root = document->root();
  if (!root.is_object()) {
    return Bad("the body must be a JSON object");
  }
  ChatRequest request;
  Ignored ignored(request.ignored);
  bool has_model = false;
  bool has_messages = false;
  bool has_prompt = false;
  std::optional<std::uint32_t> max_tokens;
  std::optional<std::uint32_t> max_completion_tokens;
  for (std::size_t k = 0; k < root.size(); ++k) {
    const std::string_view key = root.key(k);
    const json::Value v = root.member(k);
    const std::string name(key);
    if (key == "transforms" || key == "plugins") {
      // OpenRouter's, refused at the wire whatever their value (D-046).
      return Bad(std::format("{} is not supported: this is not a routing service", key), name);
    }
    if (key == "model") {
      if (!v.is_string() || v.string().empty() || v.string().size() > kMaxModelBytes) {
        return Bad(std::format("model must be a string of 1 to {} bytes", kMaxModelBytes), "model");
      }
      request.model = std::string(v.string());
      has_model = true;
    } else if (literal != nullptr && key == "prompt") {
      has_prompt = true;
      if (v.is_string()) {
        if (v.string().size() > kMaxMessageBytes) {
          return Bad("prompt exceeds the text byte limit", name);
        }
        literal->prompt = std::string(v.string());
      } else if (v.is_array() && v.size() > 0 && v.size() <= kMaxTokensCeiling) {
        for (std::size_t i = 0; i < v.size(); ++i) {
          const auto token = v.at(i).int64();
          if (!v.at(i).is_integer() || !token || *token < 0 ||
              *token > std::numeric_limits<std::int32_t>::max()) {
            return Bad(
                "prompt must be one string or one nonempty array of nonnegative int32 token IDs",
                name);
          }
          literal->token_ids.push_back(static_cast<std::int32_t>(*token));
        }
      } else {
        return Bad(
            "prompt must be one string or one nonempty token-ID array; batches are not supported",
            name);
      }
    } else if (literal != nullptr && (key == "echo" || key == "add_special_tokens" ||
                                      key == "return_tokens_as_token_ids")) {
      if (!v.is_null()) {
        if (!v.is_bool()) {
          return Bad(name + " must be a boolean", name);
        }
        if (key == "echo") {
          literal->echo = v.boolean();
        } else if (key == "add_special_tokens") {
          literal->add_special_tokens = v.boolean();
        } else {
          literal->return_tokens_as_token_ids = v.boolean();
        }
      }
    } else if (literal != nullptr && (key == "logprobs" || key == "prompt_logprobs")) {
      if (!v.is_null()) {
        const auto count = v.int64();
        if (!v.is_integer() || !count || *count < 0 ||
            std::cmp_greater(*count, kMaxCompletionTopLogprobs)) {
          return Bad(
              std::format("{} must be an integer from 0 to {}", name, kMaxCompletionTopLogprobs),
              name);
        }
        (key == "logprobs" ? literal->logprobs : literal->prompt_logprobs) =
            static_cast<std::uint32_t>(*count);
      }
    } else if (literal != nullptr && key == "best_of") {
      if (!v.is_null() && !NumberIs(v, 1)) {
        return Bad("best_of must be 1", name);
      }
    } else if (literal != nullptr && key == "suffix") {
      if (!v.is_null()) {
        return Bad("suffix insertion is not supported", name);
      }
    } else if (literal != nullptr && (key == "truncate_prompt_tokens" || key == "prompt_embeds" ||
                                      key == "allowed_token_ids" || key == "logprob_token_ids")) {
      if (!v.is_null()) {
        return Bad(
            name + " is not supported; no prompt truncation or distribution restriction is applied",
            name);
      }
    } else if (literal != nullptr && (key == "ignore_eos" || key == "use_beam_search")) {
      if (!v.is_null() && (!v.is_bool() || v.boolean())) {
        return Bad(name + " must be false", name);
      }
    } else if (literal != nullptr && key == "min_tokens") {
      if (!v.is_null() && !NumberIs(v, 0)) {
        return Bad("min_tokens must be 0", name);
      }
    } else if (literal != nullptr && key == "skip_special_tokens") {
      if (!v.is_null() && (!v.is_bool() || !v.boolean())) {
        return Bad("skip_special_tokens must be true", name);
      }
    } else if (key == "messages") {
      if (literal != nullptr) {
        return Bad("messages is not accepted by the literal completions route", name);
      }
      if (!v.is_array() || v.size() == 0) {
        return Bad("messages must be a non-empty array", "messages");
      }
      if (v.size() > kMaxMessages) {
        return Bad(std::format("messages has {} entries, more than {}", v.size(), kMaxMessages),
                   "messages");
      }
      for (std::size_t i = 0; i < v.size(); ++i) {
        auto m = ParseMessage(v.at(i), i, ignored);
        if (!m) {
          return std::unexpected(m.error());
        }
        request.messages.push_back(std::move(*m));
      }
      has_messages = true;
    } else if (std::ranges::find(kKnown, key) == kKnown.end()) {
      ignored.Add({}, key);  // unknown: ignored, its name counted
    } else if (v.is_null()) {
      continue;  // a known field's default
    } else if (key == "max_tokens") {
      auto n = ParseMaxTokens(v, key, literal != nullptr ? 0 : 1);
      if (!n) {
        return std::unexpected(n.error());
      }
      max_tokens = *n;
    } else if (key == "max_completion_tokens") {
      auto n = ParseMaxTokens(v, key, literal != nullptr ? 0 : 1);
      if (!n) {
        return std::unexpected(n.error());
      }
      max_completion_tokens = *n;
    } else if (key == "temperature") {
      const std::optional<double> t = v.is_number() ? v.float64() : std::nullopt;
      if (!t || !std::isfinite(*t) || *t < 0 || *t > kMaxTemperature) {
        return Bad(std::format("temperature must be a number from 0 to {}", kMaxTemperature), name);
      }
      request.temperature = *t;
    } else if (key == "top_p") {
      const std::optional<double> p = v.is_number() ? v.float64() : std::nullopt;
      // The sampler takes a float: a value that rounds to 0 there would be
      // refused mid-generation, a node failure, so it is refused here.
      if (!p || !std::isfinite(*p) || *p <= 0 || *p > 1 || static_cast<float>(*p) <= 0.0F) {
        return Bad("top_p must be a number greater than 0 and at most 1", name);
      }
      request.top_p = *p;
    } else if (key == "top_k") {
      const std::optional<std::int64_t> n = v.int64();
      if (!v.is_integer() || !n || *n < -1 || *n >= kMaxTopK) {
        return Bad(std::format("top_k must be an integer from -1 (or 0: off) to {}", kMaxTopK - 1),
                   name);
      }
      request.top_k = *n < 0 ? 0U : static_cast<std::uint32_t>(*n);
    } else if (key == "min_p") {
      const std::optional<double> p = v.is_number() ? v.float64() : std::nullopt;
      if (!p || !std::isfinite(*p) || *p < 0 || *p > 1) {
        return Bad("min_p must be a number from 0 to 1", name);
      }
      request.min_p = *p;
    } else if (key == "seed") {
      const std::optional<std::int64_t> s = v.int64();
      if (!v.is_integer() || !s) {
        return Bad("seed must be an integer that fits in 64 signed bits", name);
      }
      request.seed = static_cast<std::uint64_t>(*s);
    } else if (key == "stop") {
      auto stops = ParseStop(v);
      if (!stops) {
        return std::unexpected(stops.error());
      }
      request.stop = std::move(*stops);
    } else if (key == "stream") {
      if (!v.is_bool()) {
        return Bad("stream must be a boolean", name);
      }
      request.stream = v.boolean();
    } else if (key == "stream_options") {
      if (!v.is_object()) {
        return Bad("stream_options must be an object", name);
      }
      for (std::size_t j = 0; j < v.size(); ++j) {
        const std::string where = std::format("stream_options.{}", v.key(j));
        const bool usage = v.key(j) == "include_usage";
        if (!usage && v.key(j) != "include_obfuscation") {
          ignored.Add("stream_options.", v.key(j));
          continue;
        }
        if (!v.member(j).is_bool()) {
          return Bad(where + " must be a boolean", where);
        }
        if (usage) {
          request.include_usage = v.member(j).boolean();
        }  // include_obfuscation: none is sent on this route
      }
    } else if (key == "n") {
      if (!NumberIs(v, 1)) {
        return Bad("n must be 1: this route returns one choice", name);
      }
    } else if (key == "presence_penalty" || key == "frequency_penalty") {
      if (!NumberIs(v, 0)) {
        return Bad(std::format("{} is not supported by this route yet; only 0 is accepted", key),
                   name);
      }
    } else if (key == "repetition_penalty") {
      if (!NumberIs(v, 1)) {
        return Bad("repetition_penalty is not supported by this route yet; only 1 is accepted",
                   name);
      }
    } else if (key == "logprobs") {
      if (!v.is_bool() || v.boolean()) {
        return Bad("logprobs are not supported by this route yet", name);
      }
    } else if (key == "top_logprobs") {
      if (!NumberIs(v, 0)) {
        return Bad("logprobs are not supported by this route yet", name);
      }
    } else if (key == "tools" || key == "functions") {
      if (!v.is_array() || v.size() != 0) {
        return Bad("tools are not supported by this route yet", name);
      }
    } else if (key == "tool_choice" || key == "function_call") {
      if (!v.is_string() || (v.string() != "none" && v.string() != "auto")) {
        return Bad("tools are not supported by this route yet", name);
      }
    } else if (key == "response_format") {
      auto type = v.is_object() && v.size() == 1 ? v.find("type") : std::nullopt;
      if (!type || !type->is_string() || type->string() != "text") {
        return Bad(
            "response_format must be {\"type\": \"text\"}: structured output is not "
            "supported by this route yet",
            name);
      }
    } else if (key == "logit_bias") {
      if (!v.is_object() || v.size() != 0) {
        return Bad("logit_bias is not supported by this route yet", name);
      }
    } else if (key == "modalities") {
      if (!v.is_array() || v.size() != 1 || !v.at(0).is_string() || v.at(0).string() != "text") {
        return Bad("modalities must be [\"text\"]", name);
      }
    } else if (key == "audio") {
      return Bad("audio output is not supported by this route", name);
    } else if (key == "user" || key == "safety_identifier" || key == "prompt_cache_key" ||
               key == "service_tier") {
      if (!v.is_string()) {
        return Bad(std::format("{} must be a string", key), name);
      }
    } else if (key == "metadata") {
      if (!v.is_object()) {
        return Bad("metadata must be an object", name);
      }
    } else if (key == "parallel_tool_calls") {
      if (!v.is_bool()) {
        return Bad("parallel_tool_calls must be a boolean", name);
      }
    } else if (key == "store") {
      if (!v.is_bool() || v.boolean()) {
        return Bad("store must be false: nothing is stored", name);
      }
    }
  }
  if (!has_model) {
    return Bad("model is required", "model");
  }
  if (literal != nullptr && !has_prompt) {
    return Bad("prompt is required", "prompt");
  }
  if (literal == nullptr && !has_messages) {
    return Bad("messages is required", "messages");
  }
  if (max_tokens && max_completion_tokens && *max_tokens != *max_completion_tokens) {
    return Bad("max_tokens and max_completion_tokens differ; give one", "max_completion_tokens");
  }
  request.max_tokens = max_completion_tokens ? max_completion_tokens : max_tokens;
  if (literal == nullptr && request.messages.back().role != Role::kUser) {
    return Bad("the last message must be the user's", "messages");
  }
  return request;
}

}  // namespace

std::expected<ChatRequest, Error> ParseChatRequest(std::string_view body) {
  return ParseRequest(body, nullptr);
}

std::expected<CompletionRequest, Error> ParseCompletionRequest(std::string_view body) {
  CompletionRequest literal;
  auto options = ParseRequest(body, &literal);
  if (!options) {
    return std::unexpected(options.error());
  }
  if (options->stream) {
    return Bad("literal completions currently support stream=false only", "stream");
  }
  if (!options->max_tokens) {
    options->max_tokens = 16;  // legacy OpenAI completion default
  }
  literal.options = std::move(*options);
  return literal;
}

// ---------------------------------------------------------------- IgnoredFields

std::vector<std::string> IgnoredFields::Record(const std::vector<std::string>& names,
                                               std::int64_t now) {
  std::vector<std::string> fresh;
  const std::scoped_lock lock(mutex_);
  for (const std::string& name : names) {
    auto it = names_.find(name);
    if (it == names_.end()) {
      if (names_.size() >= kMaxNames) {
        ++unrecorded_;
        continue;
      }
      it = names_.emplace(name, Entry{.count = 0, .first = now, .last = now}).first;
      fresh.push_back(name);
    }
    ++it->second.count;
    it->second.last = now;
  }
  return fresh;
}

std::string IgnoredFields::Json() const {
  const std::scoped_lock lock(mutex_);
  std::string data;
  for (const auto& [name, entry] : names_) {
    data +=
        std::format(R"({}{{"name":{},"count":{},"first_seen":{},"last_seen":{}}})",
                    data.empty() ? "" : ",", Quoted(name), entry.count, entry.first, entry.last);
  }
  return std::format(R"({{"object":"list","data":[{}],"unrecorded":{}}})", data, unrecorded_);
}

// ---------------------------------------------------------------- output

std::string ErrorJson(const Error& error) {
  return std::format(R"({{"error":{{"message":{},"type":{},"param":{},"code":{}}}}})",
                     Quoted(error.message), Quoted(error.type),
                     error.param.empty() ? "null" : Quoted(error.param),
                     error.code.empty() ? "null" : Quoted(error.code));
}

namespace {

std::string_view FinishName(Finish finish) { return finish == Finish::kStop ? "stop" : "length"; }

std::string UsageJson(const Usage& u) {
  return std::format(R"({{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{},)"
                     R"("prompt_tokens_details":{{"cached_tokens":{}}}}})",
                     u.prompt_tokens, u.completion_tokens,
                     std::uint64_t{u.prompt_tokens} + std::uint64_t{u.completion_tokens},
                     u.cached_tokens);
}

}  // namespace

std::string CompletionJson(std::string_view id, std::int64_t created, std::string_view model,
                           std::string_view content, const std::optional<std::string>& reasoning,
                           Finish finish, const Usage& usage) {
  return std::format(
      R"({{"id":{},"object":"chat.completion","created":{},"model":{},"choices":[{{"index":0,)"
      R"("message":{{"role":"assistant","content":{}{}}},"logprobs":null,"finish_reason":"{}"}}],)"
      R"("usage":{}}})",
      Quoted(id), created, Quoted(model), Quoted(content),
      reasoning ? ",\"reasoning\":" + Quoted(*reasoning) : std::string(), FinishName(finish),
      UsageJson(usage));
}

std::expected<std::string, Error> LiteralCompletionJson(std::string_view id, std::int64_t created,
                                                        const CompletionRequest& request,
                                                        std::string_view content,
                                                        const LiteralResult& result, Finish finish,
                                                        const Usage& usage) {
  std::size_t allowance = 1024;
  const auto charge = [&](std::size_t bytes) {
    if (bytes > kMaxCompletionResponseBytes - allowance) {
      return false;
    }
    allowance += bytes;
    return true;
  };
  const auto charge_text = [&](std::string_view text) {
    return text.size() <= kMaxCompletionResponseBytes / 6 && charge(6 * text.size());
  };
  const auto charge_rows = [&](std::span<const TokenLogprob> rows) {
    if (rows.size() > kMaxCompletionScoreRows) {
      return false;
    }
    for (const auto& row : rows) {
      if (!charge(256) || !charge_text(row.token) || row.top.size() > 6 ||
          (row.logprob && (!std::isfinite(*row.logprob) || *row.logprob > 0))) {
        return false;
      }
      for (const auto& score : row.top) {
        if (!charge(128) || !charge_text(score.token) || !std::isfinite(score.logprob) ||
            score.logprob > 0 || score.rank == 0) {
          return false;
        }
      }
    }
    return true;
  };
  if (!charge_text(content) || (request.echo && !charge_text(result.prompt_text)) ||
      (request.logprobs && !charge_rows(result.logprobs)) ||
      (request.prompt_logprobs && !charge_rows(result.prompt_logprobs))) {
    return std::unexpected(Error{.status = 413,
                                 .type = "invalid_request_error",
                                 .message = "completion scores exceed the finite bounded response",
                                 .param = "logprobs",
                                 .code = "response_too_large"});
  }
  const auto arrays = [&](std::span<const TokenLogprob> rows) {
    std::string tokens = "[";
    std::string scores = "[";
    std::string top = "[";
    std::string offsets = "[";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto& row = rows[i];
      if (i != 0) {
        tokens += ',';
        scores += ',';
        top += ',';
        offsets += ',';
      }
      tokens += Quoted(row.token);
      scores += row.logprob ? std::format("{:.17g}", *row.logprob) : "null";
      offsets += std::to_string(row.text_offset);
      if (!row.logprob) {
        top += "null";
      } else {
        top += '{';
        // Different byte tokens may decode to the same replacement text.
        // Keep the highest score for a key so top-1/greedy remains correct;
        // return_tokens_as_token_ids avoids that representational collision.
        std::map<std::string, double> unique;
        for (std::size_t j = 0; j < row.top.size(); ++j) {
          const auto& score = row.top[j];
          if (j >= *request.logprobs && score.id != row.id) {
            continue;
          }
          const auto [at, inserted] = unique.try_emplace(score.token, score.logprob);
          if (!inserted) {
            at->second = std::max(at->second, score.logprob);
          }
        }
        bool first = true;
        for (const auto& [token, score] : unique) {
          if (!first) {
            top += ',';
          }
          first = false;
          top += Quoted(token) + ':' + std::format("{:.17g}", score);
        }
        top += '}';
      }
    }
    return "{\"tokens\":" + tokens + "],\"token_logprobs\":" + scores +
           "],\"top_logprobs\":" + top + "],\"text_offset\":" + offsets + "]}";
  };
  std::string text = request.echo ? result.prompt_text : std::string();
  text += content;
  std::string prompt_scores;
  if (request.prompt_logprobs) {
    prompt_scores = ",\"prompt_logprobs\":[";
    for (std::size_t i = 0; i < result.prompt_logprobs.size(); ++i) {
      if (i != 0) {
        prompt_scores += ',';
      }
      const auto& row = result.prompt_logprobs[i];
      if (!row.logprob) {
        prompt_scores += "null";
      } else {
        prompt_scores += '{';
        // vLLM's prompt scores are keyed by token ID, independently of the
        // legacy token text arrays. The supplied ID is always present.
        bool first = true;
        for (std::size_t j = 0; j < row.top.size(); ++j) {
          const auto& score = row.top[j];
          if (j >= *request.prompt_logprobs && score.id != row.id) {
            continue;
          }
          if (!first) {
            prompt_scores += ',';
          }
          first = false;
          prompt_scores += std::format(R"({}:{{"logprob":{:.17g},"rank":{},"decoded_token":{}}})",
                                       Quoted(std::to_string(score.id)), score.logprob, score.rank,
                                       Quoted(score.token));
        }
        prompt_scores += '}';
      }
    }
    prompt_scores += ']';
  }
  std::string body = std::format(
      R"({{"id":{},"object":"text_completion","created":{},"model":{},"choices":[{{"index":0,"text":{},"logprobs":{},"finish_reason":"{}"{}}}],"usage":{}}})",
      Quoted(id), created, Quoted(request.options.model), Quoted(text),
      request.logprobs ? arrays(result.logprobs) : "null", FinishName(finish), prompt_scores,
      UsageJson(usage));
  if (body.size() > kMaxCompletionResponseBytes) {
    return std::unexpected(Error{.status = 413,
                                 .type = "invalid_request_error",
                                 .message = "completion exceeds the bounded response size",
                                 .param = "logprobs",
                                 .code = "response_too_large"});
  }
  return body;
}

std::string ChunkJson(std::string_view id, std::int64_t created, std::string_view model,
                      Delta delta, std::string_view text, Finish finish) {
  std::string body;
  switch (delta) {
    case Delta::kRole:
      body = R"({"role":"assistant","content":""})";
      break;
    case Delta::kReasoning:
      body = "{\"reasoning\":" + Quoted(text) + "}";
      break;
    case Delta::kContent:
      body = "{\"content\":" + Quoted(text) + "}";
      break;
    case Delta::kFinish:
      body = "{}";
      break;
  }
  return std::format(
      R"({{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{)"
      R"("index":0,"delta":{},"logprobs":null,"finish_reason":{}}}]}})",
      Quoted(id), created, Quoted(model), body,
      delta == Delta::kFinish ? Quoted(FinishName(finish)) : std::string("null"));
}

std::string UsageChunkJson(std::string_view id, std::int64_t created, std::string_view model,
                           const Usage& usage) {
  return std::format(
      R"({{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[],)"
      R"("usage":{}}})",
      Quoted(id), created, Quoted(model), UsageJson(usage));
}

std::string ModelJson(const ModelInfo& model, std::int64_t created) {
  return std::format(R"({{"id":{},"object":"model","created":{},"owned_by":"jitllm"}})",
                     Quoted(model.name), created);
}

std::string ModelsJson(const std::vector<ModelInfo>& models, std::int64_t created) {
  std::string data;
  for (const ModelInfo& m : models) {
    data += (data.empty() ? "" : ",") + ModelJson(m, created);
  }
  return std::format(R"({{"object":"list","data":[{}]}})", data);
}

// ---------------------------------------------------------------- OutputText

OutputText::Out OutputText::Reasoning(std::string_view text) {
  if (stopped_) {
    return {};
  }
  had_reasoning_ = true;
  return {.reasoning = std::string(text), .content = {}};
}

OutputText::Out OutputText::Content(std::string_view text) {
  if (stopped_) {
    return {};
  }
  if (!content_started_ && had_reasoning_) {
    while (!text.empty() && (text.front() == '\n' || text.front() == ' ')) {
      text.remove_prefix(1);
    }
  }
  if (text.empty()) {
    return {};
  }
  content_started_ = true;
  held_ += text;
  // The earliest stop string ends the answer before it.
  std::size_t first = std::string::npos;
  for (const std::string& stop : stops_) {
    first = std::min(first, held_.find(stop));
  }
  if (first != std::string::npos) {
    Out out{.reasoning = {}, .content = held_.substr(0, first)};
    held_.clear();
    stopped_ = true;
    return out;
  }
  // Hold back the longest tail that begins some stop string.
  std::size_t hold = 0;
  for (const std::string& stop : stops_) {
    for (std::size_t n = std::min(stop.size() - 1, held_.size()); n > hold; --n) {
      if (std::string_view(held_).substr(held_.size() - n) == std::string_view(stop).substr(0, n)) {
        hold = n;
        break;
      }
    }
  }
  Out out{.reasoning = {}, .content = held_.substr(0, held_.size() - hold)};
  held_.erase(0, held_.size() - hold);
  return out;
}

OutputText::Out OutputText::Finish() {
  Out out{.reasoning = {}, .content = std::move(held_)};
  held_.clear();
  return out;
}

}  // namespace jitllm::runtime::api
