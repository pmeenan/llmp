// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Private, CPU-only preparation and scoring for three fixed synthetic
// long-context tasks and one separately judged code-review prompt.
#ifndef JITLLM_BENCHMARKS_LONG_CONTEXT_TASKS_H_
#define JITLLM_BENCHMARKS_LONG_CONTEXT_TASKS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jitllm::benchmarks::long_context {

inline constexpr std::uint32_t kSeed = 42;
inline constexpr std::uint32_t kChunkRows = 2048;
inline constexpr std::string_view kArtifact =
    "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";
inline constexpr std::string_view kProtocol = "ds4-long-context-tasks-v1";

enum class AnswerKind : std::uint8_t { kNumbers, kVariables, kQualitative };
struct Fact {
  std::string text;
  std::uint32_t target_token = 0;
};
struct Blueprint {
  std::string name;
  std::uint32_t prompt_tokens = 0;
  std::uint32_t capacity = 0;
  std::uint32_t output_tokens = 128;
  AnswerKind kind = AnswerKind::kNumbers;
  std::vector<Fact> facts;
  std::string question;
  std::vector<std::string> answers;
  std::string rubric;
};

// Full rendered bytes and IDs, plus each token's cumulative decoded byte
// endpoint (controls included). The codec must prove exact reconstruction.
struct Encoding {
  std::string rendered;
  std::vector<std::int32_t> ids;
  std::vector<std::size_t> byte_ends;
};
class Codec {
 public:
  Codec() = default;
  Codec(const Codec&) = default;
  Codec& operator=(const Codec&) = default;
  Codec(Codec&&) = default;
  Codec& operator=(Codec&&) = default;
  virtual ~Codec() = default;
  virtual std::expected<Encoding, std::string> Encode(std::string_view user) const = 0;
};
struct FactSpan {
  std::size_t first_byte = 0;
  std::size_t end_byte = 0;
  std::uint32_t first_token = 0;
  std::uint32_t end_token = 0;
};
struct Prepared {
  Blueprint blueprint;
  std::string user;
  Encoding encoded;
  std::vector<FactSpan> spans;
  std::uint32_t encoding_calls = 0;
  std::uint32_t hca_calls = 0;
};
struct Limits {
  std::size_t max_bytes = std::size_t{4} << 20U;
  std::uint32_t max_encoding_calls = 256;
  std::uint32_t max_distractors = 16384;
};

std::vector<Blueprint> FixedCases();
std::expected<Prepared, std::string> Prepare(const Blueprint& blueprint, const Codec& codec,
                                             const Limits& limits = {});
// Derive eligibility from the USED padded compressor extent per full
// chunk; configured context capacity alone does not determine selection.
std::expected<std::uint32_t, std::string> HcaCalls(std::uint32_t prompt_tokens);

// A strict JSON array: exact cardinality, correct types, no duplicates or
// extra text. Qualitative cases are deliberately not machine-accepted.
std::expected<bool, std::string> ExactAnswer(const Blueprint& blueprint, std::string_view response,
                                             std::string_view stop_reason);

// The private native benchmark consumes an explicit list, never a guessed
// EOS. Stops remain in preserved raw IDs but are not fed for another step.
std::expected<std::vector<std::int32_t>, std::string> ParseStopIds(std::string_view text,
                                                                   std::uint32_t vocab);
bool IsStop(std::span<const std::int32_t> stops, std::int32_t token);

}  // namespace jitllm::benchmarks::long_context
#endif  // JITLLM_BENCHMARKS_LONG_CONTEXT_TASKS_H_
