// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Choosing the next token from a step's logits, on the host: greedy, and
// seeded sampling that is reproducible from its key alone.
//
// Greedy takes the highest logit, the lowest ID among equals (as
// torch.argmax and llama.cpp's greedy sampler do). Sampling applies, in
// order: temperature (0 means greedy), top-k, softmax, min-p, top-p.
// Top-k 1 also takes the greedy path after validating the parameters,
// without candidates or random draws. Otherwise sampling draws from what
// remains with one uniform number, by inverse CDF over the kept tokens in
// a fixed order. Nothing sorts the vocabulary (a draw is a
// few linear passes; docs/tokenizer.md#sampling has its cost): top-k keeps
// the k highest logits in one pass, ranked by logit then by ID, so ties
// never depend on sort stability; min-p is a threshold; top-p buckets the
// weights by their bits and sorts only the bucket its boundary falls in,
// ranked by weight then by ID. The weights are float exponentials relative
// to the most likely token's. The distribution is the exact one (float
// rounding aside: no candidate pool, no approximate cutoff); without
// truncation the draw walks the tokens in ID order. Which token a key
// draws is a property of this implementation, not a contract across
// builds (it changed in M3, when the full sort went).
//
// The uniform number is Philox4x32-10 (Salmon et al., SC'11) keyed by the
// seed, over a counter made of the stream and the position: the same (seed,
// stream, position) always draws the same token, whatever came before, so a
// speculative verifier can redraw any position and a restored request
// continues exactly (D-068).

#ifndef JITLLM_EXECUTION_SAMPLING_H_
#define JITLLM_EXECUTION_SAMPLING_H_

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace jitllm::execution {

struct SamplingParams {
  float temperature = 1.0F;  // 0: greedy
  std::uint32_t top_k = 0;   // 0: off
  float top_p = 1.0F;        // (0, 1]; 1: off
  float min_p = 0.0F;        // [0, 1]; 0: off
};

struct SamplingKey {
  std::uint64_t seed = 0;
  std::uint64_t stream = 0;    // e.g. the request's sequence
  std::uint64_t position = 0;  // the generated token's position
};

enum class SamplingError : std::uint8_t {
  kNoLogits,       // empty, or no finite logit
  kInvalidLogits,  // a NaN or +infinity logit
  kInvalidParams,  // a parameter outside its range
};

std::string_view SamplingErrorName(SamplingError e);

// The greedy choice.
std::expected<std::int32_t, SamplingError> Greedy(std::span<const float> logits);

// Natural target log probabilities, before temperature or truncation. At
// least the top-1 and supplied token are returned, even when top_count is
// zero. Negative-infinity padding is excluded; a supplied token with zero
// probability cannot be represented as a finite JSON score and is refused.
struct TokenScores {
  struct Ranked {
    std::int32_t id = 0;
    double logprob = 0;
    std::uint32_t rank = 0;
  };
  double logprob = 0;
  std::vector<Ranked> top;
};
std::expected<TokenScores, SamplingError> ScoreToken(std::span<const float> logits,
                                                     std::int32_t token, std::uint32_t top_count);

// A seeded draw. `scratch` is reused between calls to avoid allocating (it
// grows to one candidate a logit).
struct SamplingCandidate {
  double value;
  std::int32_t id;
};
std::expected<std::int32_t, SamplingError> Sample(std::span<const float> logits,
                                                  const SamplingParams& params,
                                                  const SamplingKey& key,
                                                  std::vector<SamplingCandidate>& scratch);

// Speculative sampling's verdict on a token a greedy drafter proposed for
// the position `key` names, whose draft distribution is therefore a point
// mass (Leviathan et al., ICML 2023, and Chen et al. 2023, with q =
// δ(draft)): accepted with the probability `params` give it; otherwise a
// token drawn from that distribution without it, renormalized. Either way
// the token is distributed as Sample's. The acceptance draws UniformAt with
// the key's stream xor kAcceptStream, the replacement the key itself.
// Temperature 0 or top-k 1 accepts exactly the greedy token, and otherwise
// gives it.
struct DraftVerdict {
  bool accepted = false;
  std::int32_t token = 0;  // the draft if accepted, else its replacement
};
inline constexpr std::uint64_t kAcceptStream = std::uint64_t{1} << 63U;
std::expected<DraftVerdict, SamplingError> VerifyDraft(std::span<const float> logits,
                                                       std::int32_t draft,
                                                       const SamplingParams& params,
                                                       const SamplingKey& key,
                                                       std::vector<SamplingCandidate>& scratch);

// Philox4x32-10's block for a 128-bit counter and 64-bit key.
std::array<std::uint32_t, 4> Philox4x32(std::array<std::uint32_t, 4> counter,
                                        std::array<std::uint32_t, 2> key);

// The uniform number in [0, 1) a key draws, with 53 random bits.
double UniformAt(const SamplingKey& key);

}  // namespace jitllm::execution

#endif  // JITLLM_EXECUTION_SAMPLING_H_
