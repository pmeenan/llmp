// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Host sampling's cost a draw (execution/sampling.h) at the served models'
// vocabularies: DeepSeek V4 Flash's 129,280 and Qwen3.8's 248,320 logits.
// The logits are synthetic and fixed by a seed: a peaked row (a handful of
// likely tokens over a normal spread, as a decode step's usually is) and a
// flat one (high entropy, the nucleus's worst case). Each case draws at
// `--draws` positions and prints the median and the 90th percentile
// (docs/tokenizer.md#sampling records them).
//
//   jitllm_sampling_bench [--draws N]

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <print>
#include <random>
#include <string_view>
#include <system_error>
#include <vector>

#include "execution/sampling.h"

namespace {

namespace ex = jitllm::execution;

std::vector<float> Row(std::size_t vocab, bool peaked, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> spread(0.0F, peaked ? 2.5F : 1.0F);
  std::vector<float> logits(vocab);
  for (float& x : logits) {
    x = spread(rng);
  }
  if (peaked) {
    const std::array<float, 5> tops = {14.0F, 13.0F, 12.5F, 11.0F, 10.0F};
    for (std::size_t i = 0; i < tops.size(); ++i) {
      logits[((i * 7919U) + 17U) % vocab] = tops[i];
    }
  }
  return logits;
}

struct Case {
  std::string_view name;
  ex::SamplingParams params;
  bool verify = false;
};

}  // namespace

int main(int argc, char** argv) {
  int draws = 400;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--draws" && i + 1 < argc) {
      const std::string_view value = argv[++i];
      int parsed = 0;
      if (std::from_chars(value.data(), value.data() + value.size(), parsed).ec == std::errc()) {
        draws = std::max(1, parsed);
      }
    }
  }
  const std::array<Case, 10> cases = {{
      {.name = "T=0 greedy", .params = {.temperature = 0}, .verify = false},
      {.name = "T=1 top_k=1", .params = {.top_k = 1}, .verify = false},
      {.name = "T=1 top_k=1 verify", .params = {.top_k = 1}, .verify = true},
      {.name = "T=1", .params = {}, .verify = false},
      {.name = "T=1 verify", .params = {}, .verify = true},
      {.name = "T=1 top_p=0.95", .params = {.top_p = 0.95F}, .verify = false},
      {.name = "T=0.6 top_k=20 top_p=0.95",
       .params = {.temperature = 0.6F, .top_k = 20, .top_p = 0.95F},
       .verify = false},
      {.name = "T=1 min_p=0.05", .params = {.min_p = 0.05F}, .verify = false},
      {.name = "T=1 top_k=50", .params = {.top_k = 50}, .verify = false},
      {.name = "T=1 top_p=0.95 verify", .params = {.top_p = 0.95F}, .verify = true},
  }};
  std::println("{:>8} {:>6} {:<28} {:>10} {:>10}", "vocab", "row", "params", "median ms", "p90 ms");
  std::vector<ex::SamplingCandidate> scratch;
  for (const std::size_t vocab : {std::size_t{129280}, std::size_t{248320}}) {
    for (const bool peaked : {true, false}) {
      const std::vector<float> logits = Row(vocab, peaked, 42);
      std::int32_t draft = 0;
      if (auto g = ex::Greedy(logits); g) {
        draft = *g;
      }
      for (const Case& c : cases) {
        std::vector<double> ms;
        ms.reserve(static_cast<std::size_t>(draws));
        std::int64_t sink = 0;
        for (int d = 0; d < draws; ++d) {
          const ex::SamplingKey key{
              .seed = 7, .stream = 0, .position = static_cast<std::uint64_t>(d)};
          const auto start = std::chrono::steady_clock::now();
          if (c.verify) {
            auto v = ex::VerifyDraft(logits, draft, c.params, key, scratch);
            sink += v ? v->token : -1;
          } else {
            auto t = ex::Sample(logits, c.params, key, scratch);
            sink += t ? *t : -1;
          }
          ms.push_back(
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                  .count());
        }
        std::ranges::sort(ms);
        std::println("{:>8} {:>6} {:<28} {:>10.3f} {:>10.3f}   (sum {})", vocab,
                     peaked ? "peaked" : "flat", c.name, ms[ms.size() / 2],
                     ms[(ms.size() * 9) / 10], sink);
      }
    }
  }
  return 0;
}
