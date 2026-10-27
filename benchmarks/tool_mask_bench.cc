// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The tool-call sampler's host cost per generated token, on real
// vocabularies (docs/experiments/tool-sampling-speed): the full constrained
// mask with the greedy choice through it, and the lazy choice that checks
// the argmax first, in each tool choice and lexical state.
//
//     llmp_tool_mask_bench QWEN_TOKENIZER_JSON GEMMA4_GGUF [REPS]
//
// QWEN_TOKENIZER_JSON is Qwen3.8's tokenizer.json (its tokenizer_config.json
// beside it is applied when present); GEMMA4_GGUF a GGUF whose metadata
// holds Gemma 4's tokenizer (a header prefix suffices). Each state is
// reached by accepting a fixed text's tokens (a forced opener by the
// sampler's own stable spans), then timed REPS times (default 30). One line
// per state: the mean milliseconds of the full masked greedy choice, of the
// lazy choice when the argmax is allowed and when it is not, and of the
// speculative judge's cursor copy.

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"
#include "chat/tool_calls.h"
#include "chat/tool_sampling.h"
#include "execution/sampling.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace chat = llmp::chat;
namespace tok = llmp::tokenizer;
namespace ex = llmp::execution;
using Clock = std::chrono::steady_clock;
constexpr float kNone = -std::numeric_limits<float>::infinity();

std::optional<std::string> ReadFile(const std::string& path, std::size_t most) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  std::string bytes(most, '\0');
  in.read(bytes.data(), static_cast<std::streamsize>(most));
  bytes.resize(static_cast<std::size_t>(in.gcount()));
  return bytes;
}

std::unique_ptr<tok::Tokenizer> LoadHf(const std::string& path) {
  const auto bytes = ReadFile(path, std::size_t{512} << 20U);
  if (!bytes) return nullptr;
  auto spec = tok::ReadHfTokenizer(*bytes);
  if (!spec) {
    std::println(stderr, "{}: {}", path, spec.error().ToString());
    return nullptr;
  }
  const auto dir = path.substr(0, path.rfind('/') + 1);
  if (const auto config = ReadFile(dir + "tokenizer_config.json", std::size_t{64} << 20U))
    if (auto applied = tok::ApplyHfTokenizerConfig(*config, *spec); !applied) {
      std::println(stderr, "tokenizer_config.json: {}", applied.error().ToString());
      return nullptr;
    }
  auto created = tok::Tokenizer::Create(std::move(*spec));
  if (!created) return nullptr;
  return std::make_unique<tok::Tokenizer>(std::move(*created));
}

std::unique_ptr<tok::Tokenizer> LoadGguf(const std::string& path) {
  const auto bytes = ReadFile(path, std::size_t{256} << 20U);
  if (!bytes) return nullptr;
  auto read = tok::ReadGgufTokenizer(std::as_bytes(std::span(*bytes)));
  if (!read) {
    std::println(stderr, "{}: {}", path, read.error().ToString());
    return nullptr;
  }
  auto created = tok::Tokenizer::Create(std::move(read->spec));
  if (!created) return nullptr;
  return std::make_unique<tok::Tokenizer>(std::move(*created));
}

std::vector<std::int32_t> Encode(const tok::Tokenizer& t, std::string_view text) {
  std::vector<std::int32_t> ids;
  if (!t.Encode(text, {.special = tok::SpecialTokens::kParse}, ids)) ids.clear();
  return ids;
}

struct Family {
  std::string name;
  const tok::Tokenizer* tokenizer;
  const chat::ToolCallFormat* format;
  bool reasoning;
  std::vector<std::int32_t> terminals;
};

struct Mode {
  std::string name;
  chat::ToolChoice choice;
  bool parallel = true;
};

struct State {
  std::string name;
  std::string before = {};  // accepted first
  std::string until = {};   // then the forced opener, by its stable spans, to this
  std::string after = {};   // then this
  std::string next = {};    // an allowed continuation (its first token is the argmax)
};

// The lazy choice exists from this change on; the first measurement's
// binary has only the full mask.
template <class T>
concept Lazy = requires(const T& t) {
  { t.Allowed(std::int32_t{}) } -> std::same_as<bool>;
};

// A refused argmax's masked argmax from the row's leading tokens (F13,
// runtime/llm.cc LazyChoice): the first allowed in rank order.
template <class T>
concept LeadingLazy = Lazy<T> && requires(const T& t) {
  {
    t.AllowedOf(std::span<const std::int32_t>{}, std::span<std::int32_t>{})
  } -> std::same_as<std::size_t>;
};

template <class Sampling>
std::int32_t LazyGreedy(const Sampling& sampling, std::span<const float> row,
                        std::vector<float>& scratch) {
  if constexpr (Lazy<Sampling>) {
    if (const auto first = ex::Greedy(row); first && sampling.Allowed(*first)) return *first;
  }
  if constexpr (LeadingLazy<Sampling>) {
    std::vector<ex::SamplingCandidate> ranked;
    std::vector<std::int32_t> ids;
    std::array<std::int32_t, 1> found{};
    for (std::size_t want = 64;; want = std::min<std::size_t>(1024, 4 * want)) {
      ex::Ranked(row, want, ranked);
      ids.clear();
      for (const auto& candidate : ranked) ids.push_back(candidate.id);
      if (sampling.AllowedOf(ids, found) == 1) return found[0];
      if (ranked.size() < want || want == 1024) break;
    }
  }
  return ex::Greedy(sampling.Mask(row, scratch)).value_or(-1);
}

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

std::size_t RawBytes(const tok::Tokenizer& t, std::int32_t id) {
  std::string bytes;
  if (!t.Decode(std::span(&id, 1), {.control_tokens = true}, bytes)) return 0;
  return bytes.size();
}

bool AcceptAll(chat::ToolSampling& s, std::span<const std::int32_t> ids) {
  for (const auto id : ids)
    if (!s.Accept(id)) return false;
  return true;
}

// The forced opener: the longest allowed token, replaced by the sampler's
// stable span, until the accepted bytes hold `until`.
bool Force(chat::ToolSampling& s, const tok::Tokenizer& t, std::string_view until) {
  std::vector<float> zeros(t.size(), 0.0F), scratch;
  std::string accepted;
  for (int step = 0; step < 64; ++step) {
    const auto masked = s.Mask(zeros, scratch);
    std::size_t best_bytes = 0;
    std::int32_t best = -1;
    for (std::size_t i = 0; i < masked.size(); ++i) {
      if (masked[i] == kNone) continue;
      const auto bytes = RawBytes(t, static_cast<std::int32_t>(i));
      if (bytes > best_bytes) {
        best_bytes = bytes;
        best = static_cast<std::int32_t>(i);
      }
    }
    if (best < 0) return false;
    auto planned = s.Plan(best);
    if (!planned) return false;
    std::vector<std::int32_t> span = planned->tokens;
    span.insert(span.end(), s.certain().begin(), s.certain().end());
    s.Certain({});
    if (!AcceptAll(s, span)) return false;
    for (const auto id : span)
      (void)t.Decode(std::span(&id, 1), {.control_tokens = true}, accepted);
    if (accepted.find(until) != std::string::npos) return true;
  }
  return false;
}

void Run(const Family& family, const Mode& mode, const State& state, int reps) {
  const auto& t = *family.tokenizer;
  std::vector<chat::ToolDefinition> definitions{
      {.name = "get_weather",
       .description = "The current weather in a city.",
       .parameters_json = R"({"type":"object","properties":{"city":{"type":"string"},)"
                          R"("unit":{"type":"string","enum":["c","f"]}},"required":["city"]})"},
      {.name = "search",
       .description = "Search the web.",
       .parameters_json =
           R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})"}};
  const llmp::base::json::Limits limits{.max_bytes = std::size_t{1} << 20U,
                                        .max_depth = 64,
                                        .max_values = std::size_t{1} << 20U,
                                        .max_string_bytes = std::size_t{1} << 20U};
  auto request = chat::ToolRequest::Create(definitions, mode.choice, mode.parallel, limits);
  if (!request) {
    std::println("TOOL_MASK family={} mode={} state={} error=\"{}\"", family.name, mode.name,
                 state.name, request.error());
    return;
  }
  chat::ToolSampling s(t, *family.format, *request, limits, family.reasoning, family.terminals);
  s.Budget(4096, 32768);
  const auto fail = [&](std::string_view what) {
    std::println("TOOL_MASK family={} mode={} state={} error=\"{}\"", family.name, mode.name,
                 state.name, what);
  };
  if (!AcceptAll(s, Encode(t, state.before))) return fail("before text refused");
  if (!state.until.empty() && !Force(s, t, state.until)) return fail("forced opener not reached");
  if (!AcceptAll(s, Encode(t, state.after))) return fail("after text refused");
  const auto next = Encode(t, state.next);
  if (next.empty()) return fail("no continuation token");

  // A deterministic row of ordinary logits with one clear maximum.
  std::vector<float> row(t.size());
  std::uint32_t x = 2463534242U;
  for (auto& v : row) {
    x ^= x << 13U;
    x ^= x >> 17U;
    x ^= x << 5U;
    v = static_cast<float>(x % 10000U) / 1000.0F - 5.0F;
  }
  auto accept_row = row;
  accept_row[static_cast<std::size_t>(next.front())] = 20.0F;
  auto reject_row = row;
  const auto terminal = family.terminals.front();
  reject_row[static_cast<std::size_t>(terminal)] = 20.0F;
  // The allowed continuation next: the masked argmax a refused terminal
  // leaves (without it, the first allowed token may rank past 1,024 of a
  // random row and the leading search falls back to the full mask).
  reject_row[static_cast<std::size_t>(next.front())] = 19.0F;

  std::vector<float> scratch;
  std::int32_t full_choice = -1, lazy_choice = -1, reject_choice = -1, full_reject = -1;
  const auto time = [&](auto&& body) {
    body();  // warm
    const auto start = Clock::now();
    for (int i = 0; i < reps; ++i) body();
    return Ms(Clock::now() - start) / reps;
  };
  const double full = time([&] {
    (void)ex::ValidateLogits(accept_row);
    full_choice = ex::Greedy(s.Mask(accept_row, scratch)).value_or(-1);
  });
  const double lazy = time([&] {
    (void)ex::ValidateLogits(accept_row);
    lazy_choice = LazyGreedy(s, accept_row, scratch);
  });
  const double full_r = time([&] {
    (void)ex::ValidateLogits(reject_row);
    full_reject = ex::Greedy(s.Mask(reject_row, scratch)).value_or(-1);
  });
  const double lazy_r = time([&] {
    (void)ex::ValidateLogits(reject_row);
    reject_choice = LazyGreedy(s, reject_row, scratch);
  });
  const double copy = time([&] {
    auto copied = s;
    (void)copied.refused();
  });
  // How many tokens the mask allows (narrow states are forced or names).
  std::size_t allowed = 0;
  for (const auto v : s.Mask(row, scratch)) allowed += v == kNone ? 0 : 1;
  std::println(
      "TOOL_MASK family={} mode={} state={} vocab={} allowed={} full_ms={:.3f} lazy_ms={:.3f} "
      "full_reject_ms={:.3f} lazy_reject_ms={:.3f} copy_us={:.2f} lazy={} same={} "
      "terminal_allowed={}",
      family.name, mode.name, state.name, t.size(), allowed, full, lazy, full_r, lazy_r,
      copy * 1000.0, Lazy<chat::ToolSampling> ? "yes" : "no",
      full_choice == lazy_choice && full_reject == reject_choice ? "yes" : "NO",
      full_reject == terminal ? "yes" : "no");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::println(stderr, "usage: {} QWEN_TOKENIZER_JSON GEMMA4_GGUF [REPS]", argv[0]);
    return 2;
  }
  const int reps = argc > 3 ? std::max(1, std::atoi(argv[3])) : 30;
  const auto qwen = LoadHf(argv[1]);
  const auto gemma = LoadGguf(argv[2]);
  if (!qwen || !gemma) {
    std::println(stderr, "a tokenizer did not load");
    return 1;
  }
  const auto ids = [](const tok::Tokenizer& t, std::initializer_list<std::string_view> texts) {
    std::vector<std::int32_t> out;
    for (const auto text : texts)
      if (const auto id = t.Find(text)) out.push_back(*id);
    return out;
  };
  const std::vector<Family> families{{.name = "qwen3.8",
                                      .tokenizer = qwen.get(),
                                      .format = &chat::QwenToolCalls(),
                                      .reasoning = true,
                                      .terminals = ids(*qwen, {"<|im_end|>", "<|endoftext|>"})},
                                     {.name = "gemma4",
                                      .tokenizer = gemma.get(),
                                      .format = &chat::GemmaToolCalls(),
                                      .reasoning = false,
                                      .terminals = ids(*gemma, {"<turn|>", "<eos>"})}};
  const std::vector<Mode> modes{
      {.name = "auto", .choice = {}, .parallel = true},
      {.name = "auto-serial", .choice = {}, .parallel = false},
      {.name = "required", .choice = {.kind = chat::ToolChoiceKind::kRequired, .name = {}}},
      {.name = "named", .choice = {.kind = chat::ToolChoiceKind::kNamed, .name = "get_weather"}}};
  for (const auto& family : families) {
    if (family.terminals.empty()) {
      std::println(stderr, "{}: no terminal tokens", family.name);
      return 1;
    }
    const bool qwen_family = family.name == "qwen3.8";
    for (const auto& mode : modes) {
      const bool forced = mode.choice.kind == chat::ToolChoiceKind::kRequired ||
                          mode.choice.kind == chat::ToolChoiceKind::kNamed;
      const bool named = mode.choice.kind == chat::ToolChoiceKind::kNamed;
      std::vector<State> states;
      if (qwen_family) {
        const std::string thought = "The user wants the weather in Paris, so I call the tool.";
        states.push_back({.name = "reasoning", .before = thought, .next = " Then"});
        if (!forced) {
          states.push_back(
              {.name = "text", .before = thought + "</think>\n\nThe", .next = " weather"});
          if (!named)
            states.push_back({.name = "name",
                              .before = thought + "</think>\n\n<tool_call>\n<function=get_",
                              .next = "weather"});
          states.push_back(
              {.name = "value",
               .before = thought +
                         "</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
               .next = "is"});
          // A long argument: what the speculative judge's cursor copy holds.
          std::string body;
          while (body.size() < 4096) body += "the city of light, ";
          states.push_back({.name = "value-4k",
                            .before = thought +
                                      "</think>\n\n<tool_call>\n<function=get_weather>\n"
                                      "<parameter=city>\n" +
                                      body,
                            .next = "is"});
        } else {
          if (!named)
            states.push_back({.name = "name",
                              .before = thought + "</think>",
                              .until = "<tool_call>",
                              .after = "\n<function=get_",
                              .next = "weather"});
          states.push_back({.name = "value",
                            .before = thought + "</think>",
                            .until = named ? "get_weather>" : "<tool_call>",
                            .after = named ? "\n<parameter=city>\nPar"
                                           : "\n<function=get_weather>\n<parameter=city>\nPar",
                            .next = "is"});
        }
      } else {
        if (!forced) {
          states.push_back({.name = "text", .before = "The", .next = " weather"});
          states.push_back({.name = "name", .before = "<|tool_call>call:get_", .next = "weather"});
          states.push_back({.name = "value",
                            .before = "<|tool_call>call:get_weather{city:<|\"|>Par",
                            .next = "is"});
        } else {
          if (!named)
            states.push_back({.name = "name",
                              .before = "",
                              .until = "call:",
                              .after = "get_",
                              .next = "weather"});
          states.push_back({.name = "value",
                            .before = "",
                            .until = named ? "get_weather{" : "call:",
                            .after = named ? "city:<|\"|>Par" : "get_weather{city:<|\"|>Par",
                            .next = "is"});
        }
      }
      for (const auto& state : states) Run(family, mode, state, reps);
    }
  }
  return 0;
}
