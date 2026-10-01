// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "long_context_tasks.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>

#include "base/json.h"

namespace jitllm::benchmarks::long_context {
namespace {
template <class T>
std::unexpected<std::string> Fail(T&& message) {
  return std::unexpected<std::string>(std::forward<T>(message));
}

std::expected<std::int64_t, std::string> Integer(std::string_view text) {
  std::int64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size())
    return Fail("expected a complete bounded integer");
  return value;
}
std::expected<void, std::string> CheckFacts(const Blueprint& b) {
  std::set<std::string> lefts;
  std::set<std::string> values;
  std::map<std::string, std::int64_t> bindings;
  std::optional<std::int64_t> initial;
  for (const auto& fact : b.facts) {
    if (fact.text.empty()) return Fail("empty fact");
    if (b.kind == AnswerKind::kQualitative) continue;
    const std::string_view prefix = b.kind == AnswerKind::kVariables ? "Binding " : "Record ";
    const auto equal = fact.text.find(" = ");
    if (!fact.text.starts_with(prefix) || equal == std::string::npos || !fact.text.ends_with(".\n"))
      return Fail("fact grammar differs from the fixed generator");
    const auto start = std::string_view(prefix).size();
    const std::string left = fact.text.substr(start, equal - start);
    const std::string right = fact.text.substr(equal + 3, fact.text.size() - equal - 5);
    if (left.empty() || left.starts_with(b.kind == AnswerKind::kVariables ? "n" : "d") ||
        !lefts.insert(left).second)
      return Fail("duplicate or distractor-reserved fact key or variable");
    if (b.kind == AnswerKind::kNumbers) {
      auto value = Integer(right);
      if (!value || *value < 100000 || *value >= 800000000 || std::to_string(*value) != right ||
          !values.insert(right).second)
        return Fail("target values must be unique and outside the distractor range");
    } else {
      auto value = Integer(right);
      if (!value) {
        const auto found = bindings.find(right);
        if (found == bindings.end()) return Fail("chain references an unbound variable");
        value = found->second;
      }
      if (!initial) {
        if (*value < 0 || *value >= 800000000)
          return Fail("initial chain value collides with the distractor range");
        initial = *value;
      }
      bindings.emplace(left, *value);
    }
  }
  if (b.kind == AnswerKind::kVariables) {
    for (const auto& [name, value] : bindings)
      if (initial && value == *initial) values.insert(name);
  }
  const std::set<std::string> expected(b.answers.begin(), b.answers.end());
  if (b.kind != AnswerKind::kQualitative &&
      (expected.size() != b.answers.size() || expected != values))
    return Fail("known answer does not match the complete chronological facts");
  return {};
}

std::string Intro(AnswerKind kind) {
  if (kind == AnswerKind::kQualitative)
    return "Review the fictional miniature repository below. The repeated registry records are "
           "unrelated background. Use only the supplied files and requirements.\n";
  return "The following registry contains unrelated distractors and authoritative records. "
         "Read the complete registry, then answer the final question. Return only one JSON "
         "array, without Markdown or explanation.\n";
}
std::string Distractors(std::uint32_t count, std::uint32_t slot, AnswerKind kind) {
  std::string result;
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto id = (std::uint64_t{slot} * 16384) + i;
    result +=
        std::format("{} {}{:06} = {}.\n", kind == AnswerKind::kVariables ? "Binding" : "Record",
                    kind == AnswerKind::kVariables ? "n" : "d", id,
                    800000000 + (std::uint64_t{kSeed} * 200000) + id);
  }
  return result;
}
std::string Compose(const Blueprint& b, std::span<const std::string> filler) {
  std::string user = Intro(b.kind);
  for (std::size_t i = 0; i < b.facts.size(); ++i) {
    user += filler[i];
    user += b.facts[i].text;
  }
  user += filler.back();
  // Facts and neutral filler already end in a newline. A second newline
  // here can merge with the final fact's terminator in the native tokenizer.
  user += "Question: " + b.question + "\n";
  return user;
}
std::expected<FactSpan, std::string> Span(const Encoding& e, std::string_view fact) {
  const auto at = e.rendered.find(fact);
  if (at == std::string::npos || e.rendered.find(fact, at + 1) != std::string::npos)
    return Fail("fact must occur exactly once in the final rendering");
  const auto end_byte = at + fact.size();
  const auto first = std::ranges::upper_bound(e.byte_ends, at);
  const auto end = std::ranges::lower_bound(e.byte_ends, end_byte);
  if (first == e.byte_ends.end() || end == e.byte_ends.end() || *end != end_byte ||
      (first != e.byte_ends.begin() && *(first - 1) != at))
    return Fail("a fact boundary falls inside a token");
  return FactSpan{at, end_byte, static_cast<std::uint32_t>(first - e.byte_ends.begin()),
                  static_cast<std::uint32_t>((end - e.byte_ends.begin()) + 1)};
}
std::expected<void, std::string> CheckEncoding(const Encoding& e, const Limits& limits) {
  if (e.rendered.empty() || e.rendered.size() > limits.max_bytes || e.ids.empty() ||
      e.ids.size() > 131072 || e.ids.size() != e.byte_ends.size() ||
      e.byte_ends.back() != e.rendered.size() || e.byte_ends.front() == 0 ||
      std::ranges::any_of(e.ids, [](auto id) { return id < 0 || id >= 129280; }))
    return Fail("codec returned an out-of-bound or incomplete rendering");
  for (std::size_t i = 1; i < e.byte_ends.size(); ++i)
    if (e.byte_ends[i] <= e.byte_ends[i - 1]) return Fail("non-increasing decoded token endpoints");
  return {};
}
}  // namespace

std::vector<Blueprint> FixedCases() {
  Blueprint r1{"r1-single",
               30720,
               32768,
               128,
               AnswerKind::kNumbers,
               {{"Record k004201 = 731942.\n", 16384}},
               "What value is assigned to k004201? Return a JSON array containing exactly "
               "that integer.",
               {"731942"},
               {}};
  Blueprint r2{"r2-four",
               126976,
               131072,
               128,
               AnswerKind::kNumbers,
               {{"Record k004211 = 620143.\n", 8192},
                {"Record k004212 = 815207.\n", 24576},
                {"Record k004213 = 390461.\n", 106496},
                {"Record k004214 = 574829.\n", 122880}},
               "What are the values assigned to k004211, k004212, k004213 and k004214? "
               "Return a JSON array of exactly those four integers, in any order.",
               {"620143", "815207", "390461", "574829"},
               {}};
  Blueprint r3{"r3-chain",
               126976,
               131072,
               128,
               AnswerKind::kVariables,
               {{"Binding v_seed42 = 314159.\n", 8192},
                {"Binding v_hop42a = v_seed42.\n", 24576},
                {"Binding v_hop42b = v_hop42a.\n", 57344},
                {"Binding v_hop42c = v_hop42b.\n", 102400},
                {"Binding v_hop42d = v_hop42c.\n", 120832}},
               "After all chronological bindings, which five variables hold the initial "
               "value of v_seed42? Return a JSON array of exactly their names, in any order.",
               {"v_seed42", "v_hop42a", "v_hop42b", "v_hop42c", "v_hop42d"},
               {}};
  Blueprint c1{"c1-code-review",
               30720,
               32768,
               768,
               AnswerKind::kQualitative,
               {{"FILE frame.h:\n"
                 "struct View { const std::byte* data; std::size_t size; };\n"
                 "bool Parse(std::span<const std::byte> input, std::size_t offset, View& out) {\n"
                 "  auto n = ReadU32LE(input.data() + offset);\n"
                 "  auto end = offset + 4 + n;\n"
                 "  if (end > input.size()) return false;\n"
                 "  out = {input.data() + offset + 4, n};\n"
                 "  return true;\n"
                 "}\nEND FILE frame.h\n",
                 8192},
                {"FILE dispatch.cc:\n"
                 "void Handle(std::span<const std::byte> input) {\n"
                 "  View frame{};\n"
                 "  if (Parse(input, 0, frame)) queue.Push([frame] { Consume(frame); });\n"
                 "}\nEND FILE dispatch.cc\n"
                 "CONTRACT: input belongs to a reusable network receive buffer. The queue "
                 "runs later. ReadU32LE requires four readable bytes. Payloads may be empty "
                 "but are at most 65536 bytes. Reject truncated/oversized frames; preserve "
                 "the valid-frame API behavior.\n",
                 24576}},
               "Provide a concise concrete C++ patch for the supplied parser and asynchronous "
               "dispatch, then explain the essential tests. Identify both bounds and lifetime "
               "bugs. Do not invent unrelated repository files.",
               {},
               "Qualitative only: validate offset/header before pointer arithmetic; use "
               "subtraction or checked addition for length and cap65536; retain owned or "
               "completion-fenced payload until queue consumption; test truncated header, "
               "truncated body, zero/max/oversized payload, overflow offset and receive-buffer "
               "reuse; preserve valid semantics with bounded buffering."};
  return {std::move(r1), std::move(r2), std::move(r3), std::move(c1)};
}

std::expected<std::uint32_t, std::string> HcaCalls(std::uint32_t prompt_tokens) {
  if (prompt_tokens == 0 || prompt_tokens > 131072 || prompt_tokens % kChunkRows != 0)
    return Fail("HCA count requires complete2048-row chunks within131072");
  std::uint32_t calls = 0;
  for (std::uint32_t chunk = 1; chunk <= prompt_tokens / kChunkRows; ++chunk) {
    const auto used = (chunk * kChunkRows) / 128;
    const auto padded = ((used + 255) / 256) * 256;
    if (padded == 256 || padded == 1024) calls += 20;
  }
  return calls;
}

std::expected<Prepared, std::string> Prepare(const Blueprint& b, const Codec& codec,
                                             const Limits& limits) {
  if ((b.kind != AnswerKind::kNumbers && b.kind != AnswerKind::kVariables &&
       b.kind != AnswerKind::kQualitative) ||
      b.name.empty() || b.question.empty() || b.facts.empty() || b.facts.size() > 8 ||
      b.prompt_tokens == 0 || b.capacity > 131072 || b.prompt_tokens >= b.capacity ||
      b.output_tokens == 0 || b.output_tokens > b.capacity - b.prompt_tokens ||
      limits.max_bytes == 0 || limits.max_bytes > (std::size_t{4} << 20U) ||
      limits.max_encoding_calls == 0 || limits.max_encoding_calls > 512 ||
      limits.max_distractors == 0 || limits.max_distractors > 16384)
    return Fail("invalid bounded task geometry");
  if (auto checked = CheckFacts(b); !checked) return Fail(checked.error());
  auto hca = HcaCalls(b.prompt_tokens);
  if (!hca) return Fail(hca.error());
  std::uint32_t previous = 0;
  for (const auto& fact : b.facts) {
    if (fact.target_token <= previous || fact.target_token >= b.prompt_tokens)
      return Fail("fact positions must be strictly increasing within the prompt");
    previous = fact.target_token;
  }
  Prepared result;
  result.blueprint = b;
  result.hca_calls = *hca;
  std::vector<std::string> filler(b.facts.size() + 1);
  const auto encode = [&]() -> std::expected<Encoding, std::string> {
    if (result.encoding_calls >= limits.max_encoding_calls)
      return Fail("bounded full-render encoding budget exhausted");
    ++result.encoding_calls;
    auto user = Compose(b, filler);
    if (user.size() > limits.max_bytes) return Fail("synthetic text exceeds the byte bound");
    auto e = codec.Encode(user);
    if (!e) return Fail(e.error());
    if (auto checked = CheckEncoding(*e, limits); !checked) return Fail(checked.error());
    return e;
  };
  const auto metric = [&](const Encoding& e,
                          std::size_t slot) -> std::expected<std::uint32_t, std::string> {
    if (slot == b.facts.size()) return static_cast<std::uint32_t>(e.ids.size());
    auto span = Span(e, b.facts[slot].text);
    if (!span) return Fail(span.error());
    return span->first_token;
  };
  // Every trial measures the complete native rendering. Estimates only
  // select the next neutral-filler trial; they never establish placement.
  for (std::size_t slot = 0; slot < filler.size(); ++slot) {
    const auto target = slot == b.facts.size() ? b.prompt_tokens : b.facts[slot].target_token;
    auto e = encode();
    if (!e) return Fail(e.error());
    auto start = metric(*e, slot);
    if (!start) return Fail(start.error());
    if (*start > target) return Fail("immutable facts/question already exceed a target");
    if (*start == target) continue;
    const auto baseline = *start;
    filler[slot] = Distractors(1, static_cast<std::uint32_t>(slot), b.kind);
    e = encode();
    if (!e) return Fail(e.error());
    auto one = metric(*e, slot);
    if (!one || *one <= baseline)
      return Fail("distractor does not advance the final token position");
    const auto step = *one - baseline;
    std::uint32_t count = std::min((target - baseline) / step, limits.max_distractors);
    std::uint32_t best = 0;
    std::uint32_t best_metric = baseline;
    for (unsigned trial = 0; trial < 12; ++trial) {
      filler[slot] = Distractors(count, static_cast<std::uint32_t>(slot), b.kind);
      e = encode();
      if (!e) return Fail(e.error());
      auto m = metric(*e, slot);
      if (!m) return Fail(m.error());
      if (*m <= target && *m >= best_metric) {
        best = count;
        best_metric = *m;
      }
      if (*m == target || (best_metric <= target && target - best_metric < step * 2)) break;
      const auto delta = std::int64_t{target} - *m;
      const auto adjustment = delta / step;
      const auto direction = delta > 0 ? 1 : -1;
      const auto change = adjustment == 0 ? direction : adjustment;
      const auto next = std::clamp(std::int64_t{count} + change, std::int64_t{0},
                                   std::int64_t{limits.max_distractors});
      if (std::cmp_equal(next, count)) break;
      count = static_cast<std::uint32_t>(next);
    }
    if (target - best_metric == 1 && best != 0) {
      --best;
      filler[slot] = Distractors(best, static_cast<std::uint32_t>(slot), b.kind);
      e = encode();
      if (!e) return Fail(e.error());
      auto m = metric(*e, slot);
      if (!m || *m > target) return Fail("coarse retreat changed the target ordering");
      best_metric = *m;
    }
    const auto coarse = Distractors(best, static_cast<std::uint32_t>(slot), b.kind);
    bool placed = best_metric == target;
    filler[slot] = coarse;
    // Find an actual one-token neutral word at this boundary. The final
    // full re-encoding, not a per-piece token count, proves the result.
    for (std::string_view word : {" note", " a", " x", " 7", " .", " ", "\n"}) {
      if (placed) break;
      filler[slot] = coarse + std::string(word) + "\n";
      e = encode();
      if (!e) return Fail(e.error());
      auto one_word = metric(*e, slot);
      if (!one_word || *one_word <= best_metric || *one_word > target) continue;
      const auto word_step = *one_word - best_metric;
      std::uint32_t words = std::max(std::uint32_t{1}, (target - best_metric) / word_step);
      for (unsigned trial = 0; trial < 8; ++trial) {
        if (words > 4096) break;
        filler[slot] = coarse;
        for (std::uint32_t i = 0; i < words; ++i) filler[slot] += word;
        filler[slot] += '\n';
        e = encode();
        if (!e) return Fail(e.error());
        auto m = metric(*e, slot);
        if (!m) return Fail(m.error());
        if (*m == target) {
          placed = true;
          break;
        }
        const auto delta = std::int64_t{target} - *m;
        const auto direction = delta > 0 ? 1 : -1;
        const auto adjustment = delta / word_step;
        const auto next = std::int64_t{words} + (adjustment == 0 ? direction : adjustment);
        if (next <= 0 || std::cmp_equal(next, words) || next > 4096) break;
        words = static_cast<std::uint32_t>(next);
      }
    }
    if (!placed) return Fail("bounded neutral filler cannot reach the exact token target");
  }
  auto final = encode();
  if (!final) return Fail(final.error());
  if (final->ids.size() != b.prompt_tokens) return Fail("final rendered prompt length changed");
  for (const auto& fact : b.facts) {
    auto span = Span(*final, fact.text);
    if (!span || span->first_token != fact.target_token)
      return Fail("final rendering changed an exact fact token span");
    result.spans.push_back(*span);
  }
  result.user = Compose(b, filler);
  result.encoded = std::move(*final);
  return result;
}

std::expected<bool, std::string> ExactAnswer(const Blueprint& b, std::string_view response,
                                             std::string_view stop_reason) {
  if (b.kind == AnswerKind::kQualitative) return Fail("qualitative sample requires human review");
  if (stop_reason != "eos") return false;
  auto document = base::json::Parse(
      response, {.max_bytes = 4096, .max_depth = 2, .max_values = 16, .max_string_bytes = 1024});
  if (!document) return false;
  const auto root = document->root();
  if (!root.is_array() || root.size() != b.answers.size()) return false;
  std::set<std::string> actual;
  for (std::size_t i = 0; i < root.size(); ++i) {
    const auto value = root.at(i);
    std::string item;
    if (b.kind == AnswerKind::kNumbers) {
      const auto number = value.int64();
      if (!number) return false;
      item = std::to_string(*number);
    } else {
      if (!value.is_string()) return false;
      item = value.string();
    }
    if (!actual.insert(std::move(item)).second) return false;
  }
  return actual == std::set<std::string>(b.answers.begin(), b.answers.end());
}

std::expected<std::vector<std::int32_t>, std::string> ParseStopIds(std::string_view text,
                                                                   std::uint32_t vocab) {
  if (text.empty() || text.size() > 256 || vocab == 0) return Fail("empty or oversized stop list");
  std::vector<std::int32_t> ids;
  std::size_t first = 0;
  for (;;) {
    const auto comma = text.find(',', first);
    const auto word = text.substr(first, comma == std::string_view::npos ? comma : comma - first);
    auto id = Integer(word);
    if (!id || *id < 0 || *id > std::numeric_limits<std::int32_t>::max() ||
        std::cmp_greater_equal(*id, vocab) || ids.size() >= 16 ||
        std::ranges::find(ids, *id) != ids.end())
      return Fail("invalid, duplicate or excessive stop IDs");
    ids.push_back(static_cast<std::int32_t>(*id));
    if (comma == std::string_view::npos) break;
    first = comma + 1;
  }
  return ids;
}
bool IsStop(std::span<const std::int32_t> stops, std::int32_t token) {
  return std::ranges::find(stops, token) != stops.end();
}
}  // namespace jitllm::benchmarks::long_context
