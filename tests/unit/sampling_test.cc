// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Greedy and seeded sampling (execution/sampling.h).

#include "execution/sampling.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <utility>
#include <vector>

#include "expected_error.h"

namespace {

namespace ex = llmp::execution;
using llmp::test_support::Failed;

// Random123's published known-answer vectors for Philox4x32-10 (its
// tests/kat_vectors).
TEST(Philox, KnownAnswers) {
  EXPECT_EQ(ex::Philox4x32({0, 0, 0, 0}, {0, 0}),
            (std::array<std::uint32_t, 4>{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}));
  EXPECT_EQ(
      ex::Philox4x32({0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, {0xffffffff, 0xffffffff}),
      (std::array<std::uint32_t, 4>{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}));
  EXPECT_EQ(
      ex::Philox4x32({0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, {0xa4093822, 0x299f31d0}),
      (std::array<std::uint32_t, 4>{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}));
}

TEST(Uniform, InRangeAndKeyedByEveryField) {
  const double u = ex::UniformAt({1, 2, 3});
  EXPECT_GE(u, 0.0);
  EXPECT_LT(u, 1.0);
  EXPECT_EQ(u, ex::UniformAt({1, 2, 3}));
  EXPECT_NE(u, ex::UniformAt({2, 2, 3}));
  EXPECT_NE(u, ex::UniformAt({1, 3, 3}));
  EXPECT_NE(u, ex::UniformAt({1, 2, 4}));
  double sum = 0;
  for (std::uint64_t p = 0; p < 100000; ++p) {
    sum += ex::UniformAt({7, 0, p});
  }
  EXPECT_NEAR(sum / 100000, 0.5, 0.01);
}

TEST(Greedy, TakesTheLowestIdAmongEqualMaxima) {
  const std::vector<float> logits = {1.0F, 3.0F, -2.0F, 3.0F};
  EXPECT_EQ(ex::Greedy(logits), 1);
  const std::vector<float> masked = {-INFINITY, -INFINITY, 0.5F};
  EXPECT_EQ(ex::Greedy(masked), 2);
}

TEST(Greedy, RefusesBadLogits) {
  EXPECT_EQ(Failed(ex::Greedy(std::vector<float>{})), ex::SamplingError::kNoLogits);
  EXPECT_EQ(Failed(ex::Greedy(std::vector<float>{-INFINITY, -INFINITY})),
            ex::SamplingError::kNoLogits);
  EXPECT_EQ(Failed(ex::Greedy(std::vector<float>{1.0F, NAN})), ex::SamplingError::kInvalidLogits);
  EXPECT_EQ(Failed(ex::Greedy(std::vector<float>{1.0F, INFINITY})),
            ex::SamplingError::kInvalidLogits);
}

TEST(TokenScores, ScoresTheSuppliedTokenOutsideTopOneWithoutSamplingBias) {
  const std::vector<float> row = {0, 2, -3, -INFINITY};
  const auto score = ex::ScoreToken(row, 2, 0);
  ASSERT_TRUE(score.has_value());
  const double norm = std::log(std::exp(-2.0) + 1 + std::exp(-5.0));
  EXPECT_DOUBLE_EQ(score->logprob, -5 - norm);
  ASSERT_EQ(score->top.size(), 2U);
  EXPECT_EQ(score->top[0].id, 1);
  EXPECT_DOUBLE_EQ(score->top[0].logprob, -norm);
  EXPECT_EQ(score->top[0].rank, 1U);
  EXPECT_EQ(score->top[1].id, 2);
  EXPECT_EQ(score->top[1].rank, 3U);
  EXPECT_DOUBLE_EQ(score->top[1].logprob, score->logprob);
  const auto shifted = ex::ScoreToken(std::vector<float>{100, 102, 97, -INFINITY}, 2, 5);
  ASSERT_TRUE(shifted.has_value());
  EXPECT_DOUBLE_EQ(shifted->logprob, score->logprob);
  EXPECT_EQ(shifted->top.size(), 3U);  // padding never becomes a candidate
}

TEST(TokenScores, TiesHaveEqualProbabilitiesAndStableLowestIdOrdering) {
  const auto score = ex::ScoreToken(std::vector<float>{2, -2, 2, 2}, 3, 2);
  ASSERT_TRUE(score.has_value());
  ASSERT_EQ(score->top.size(), 3U);
  EXPECT_EQ(score->top[0].id, 0);
  EXPECT_EQ(score->top[1].id, 2);
  EXPECT_EQ(score->top[2].id, 3);
  EXPECT_DOUBLE_EQ(score->logprob, score->top[0].logprob);
  EXPECT_EQ(score->top[2].rank, 1U);
}

TEST(TokenScores, RefusesInvalidTargetsAndNonJsonDistributions) {
  for (const auto& row : {std::vector<float>{NAN, 0}, std::vector<float>{INFINITY, 0},
                          std::vector<float>{-INFINITY, -INFINITY}}) {
    EXPECT_FALSE(ex::ScoreToken(row, 1, 1));
  }
  EXPECT_FALSE(ex::ScoreToken(std::vector<float>{0, -INFINITY}, 1, 1));
  EXPECT_FALSE(ex::ScoreToken(std::vector<float>{0}, -1, 1));
  EXPECT_FALSE(ex::ScoreToken(std::vector<float>{0}, 1, 1));
  // More top scores than the row asks for all of it (D-102: no cap of 5).
  const auto all = ex::ScoreToken(std::vector<float>{0}, 0, 6);
  ASSERT_TRUE(all.has_value());
  EXPECT_EQ(all->top.size(), 1U);
}

// Many top scores (D-102 raised the API's 5 to the vocabulary) keep the
// few-score ordering and ranks: more likely first, ties by the lower ID,
// equal logits sharing the first one's rank; the whole row in order.
TEST(TokenScores, ManyTopScoresKeepTheOrderAndRanks) {
  std::vector<float> row(5000);
  for (std::size_t i = 0; i < row.size(); ++i) {
    row[i] = static_cast<float>((i * 7919) % 97) - 40.0F;  // many ties
  }
  row[17] = -INFINITY;  // padding: never a candidate
  const auto few = ex::ScoreToken(row, 3, 16);
  const auto many = ex::ScoreToken(row, 3, 4000);
  const auto whole = ex::ScoreToken(row, 3, 5000);
  ASSERT_TRUE(few.has_value() && many.has_value() && whole.has_value());
  ASSERT_GE(many->top.size(), 4000U);
  EXPECT_EQ(whole->top.size(), row.size() - 1);  // every finite logit, the actual among them
  for (std::size_t j = 0; j < 16; ++j) {
    EXPECT_EQ(few->top[j].id, many->top[j].id) << j;
    EXPECT_EQ(few->top[j].rank, many->top[j].rank) << j;
    EXPECT_DOUBLE_EQ(few->top[j].logprob, many->top[j].logprob) << j;
  }
  for (std::size_t j = 1; j < whole->top.size(); ++j) {
    const auto& a = whole->top[j - 1];
    const auto& b = whole->top[j];
    EXPECT_TRUE(a.logprob > b.logprob || (a.logprob == b.logprob && a.id < b.id)) << j;
    EXPECT_EQ(b.rank, a.logprob == b.logprob ? a.rank : static_cast<std::uint32_t>(j + 1)) << j;
  }
  EXPECT_DOUBLE_EQ(whole->logprob, many->logprob);
}

TEST(Sample, TemperatureZeroIsGreedy) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {0.1F, 0.9F, 0.3F};
  for (std::uint64_t p = 0; p < 20; ++p) {
    EXPECT_EQ(ex::Sample(logits, {.temperature = 0}, {5, 0, p}, scratch), 1);
  }
}

TEST(Sampling, TopKOneIsGreedyForSamplesAndDraftVerdicts) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::array<std::vector<float>, 4> rows = {{
      {0.1F, 0.9F, 0.3F},
      {-INFINITY, 3.0F, 3.0F, -1.0F},  // ties choose the lower ID
      {-INFINITY, 3.0F, -INFINITY},    // just one finite logit
      {0.0F},
  }};
  for (const auto& logits : rows) {
    const auto greedy = ex::Greedy(logits);
    ASSERT_TRUE(greedy.has_value());
    for (const float temperature : {0.0F, std::numeric_limits<float>::denorm_min(), 0.6F,
                                    std::numeric_limits<float>::max()}) {
      const ex::SamplingParams params{.temperature = temperature,
                                      .top_k = 1,
                                      .top_p = std::numeric_limits<float>::denorm_min(),
                                      .min_p = 1.0F};
      for (std::uint64_t s = 0; s < 20; ++s) {
        const ex::SamplingKey key{.seed = s, .stream = s + 1, .position = s + 2};
        EXPECT_EQ(ex::Sample(logits, params, key, scratch), greedy);
        for (std::int32_t draft = -1; std::cmp_less_equal(draft, logits.size()); ++draft) {
          const auto verdict = ex::VerifyDraft(logits, draft, params, key, scratch);
          ASSERT_TRUE(verdict.has_value());
          EXPECT_EQ(verdict->accepted, draft == *greedy);
          EXPECT_EQ(verdict->token, *greedy);
        }
      }
    }
  }
  EXPECT_EQ(scratch.capacity(), 0U);  // no candidate storage allocated
}

TEST(Sampling, TopKOneStillValidatesParametersAndLogits) {
  std::vector<ex::SamplingCandidate> scratch;
  for (ex::SamplingParams params :
       {ex::SamplingParams{.temperature = -1.0F}, ex::SamplingParams{.temperature = NAN},
        ex::SamplingParams{.temperature = INFINITY}, ex::SamplingParams{.top_p = 0.0F},
        ex::SamplingParams{.top_p = 1.5F}, ex::SamplingParams{.top_p = NAN},
        ex::SamplingParams{.min_p = -0.1F}, ex::SamplingParams{.min_p = 1.5F},
        ex::SamplingParams{.min_p = NAN}}) {
    params.top_k = 1;
    // Parameters must be checked even before invalid logits.
    for (const auto& logits : {std::vector<float>{0.0F, 1.0F}, std::vector<float>{NAN}}) {
      EXPECT_EQ(Failed(ex::Sample(logits, params, {}, scratch)), ex::SamplingError::kInvalidParams);
      EXPECT_EQ(Failed(ex::VerifyDraft(logits, 1, params, {}, scratch)),
                ex::SamplingError::kInvalidParams);
    }
  }
  for (const auto& logits : {std::vector<float>{}, std::vector<float>{-INFINITY, -INFINITY}}) {
    EXPECT_EQ(Failed(ex::Sample(logits, {.top_k = 1}, {}, scratch)), ex::SamplingError::kNoLogits);
    EXPECT_EQ(Failed(ex::VerifyDraft(logits, 0, {.top_k = 1}, {}, scratch)),
              ex::SamplingError::kNoLogits);
  }
  for (const auto& logits : {std::vector<float>{1.0F, NAN}, std::vector<float>{1.0F, INFINITY}}) {
    EXPECT_EQ(Failed(ex::Sample(logits, {.top_k = 1}, {}, scratch)),
              ex::SamplingError::kInvalidLogits);
    EXPECT_EQ(Failed(ex::VerifyDraft(logits, 0, {.top_k = 1}, {}, scratch)),
              ex::SamplingError::kInvalidLogits);
  }
}

TEST(Sample, IsReproducibleFromItsKey) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {1.0F, 1.2F, 0.8F, 1.1F, 0.0F};
  for (std::uint64_t p = 0; p < 50; ++p) {
    const auto a = ex::Sample(logits, {}, {9, 1, p}, scratch);
    const auto b = ex::Sample(logits, {}, {9, 1, p}, scratch);
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(a, b);
  }
}

TEST(Sample, FollowsTheSoftmax) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {std::log(0.5F), std::log(0.3F), std::log(0.2F)};
  std::map<int, int> counts;
  constexpr int kDraws = 60000;
  for (int p = 0; p < kDraws; ++p) {
    ++counts[*ex::Sample(logits, {}, {3, 0, static_cast<std::uint64_t>(p)}, scratch)];
  }
  EXPECT_NEAR(counts[0] / double{kDraws}, 0.5, 0.01);
  EXPECT_NEAR(counts[1] / double{kDraws}, 0.3, 0.01);
  EXPECT_NEAR(counts[2] / double{kDraws}, 0.2, 0.01);
}

TEST(Sample, FiltersAsDocumented) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {std::log(0.5F), std::log(0.3F), std::log(0.15F),
                                     std::log(0.05F)};
  auto seen = [&](ex::SamplingParams params) {
    std::map<int, int> counts;
    for (std::uint64_t p = 0; p < 4000; ++p) {
      ++counts[*ex::Sample(logits, params, {11, 0, p}, scratch)];
    }
    return counts;
  };
  EXPECT_EQ(seen({.top_k = 1}).size(), 1U);
  EXPECT_EQ(seen({.top_k = 2}).size(), 2U);
  EXPECT_EQ(seen({.top_p = 0.45F}).size(), 1U);  // 0.5 reaches 0.45
  EXPECT_EQ(seen({.top_p = 0.79F}).size(), 2U);  // 0.5 + 0.3
  EXPECT_EQ(seen({.min_p = 0.5F}).size(), 2U);   // 0.3 >= 0.25, 0.15 is not
  EXPECT_EQ(seen({.min_p = 0.05F}).size(), 4U);
}

TEST(Sample, TiesOrderByIdNotBySortStability) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits(1000, 0.0F);
  const auto a = ex::Sample(logits, {.top_k = 10}, {1, 0, 0}, scratch);
  ASSERT_TRUE(a.has_value());
  EXPECT_LT(*a, 10);  // top_k keeps the ten lowest IDs among equals
}

TEST(Sample, RefusesBadParameters) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {0.0F, 1.0F};
  EXPECT_EQ(Failed(ex::Sample(logits, {.temperature = -1}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(logits, {.temperature = NAN}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(logits, {.top_p = 0}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(logits, {.top_p = 1.5F}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(logits, {.min_p = -0.1F}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(std::vector<float>{NAN}, {}, {}, scratch)),
            ex::SamplingError::kInvalidLogits);
}

// Speculative sampling with a greedy drafter: the verdict's token is
// distributed as Sample's, the draft accepted with its probability.
TEST(VerifyDraft, PreservesTheDistribution) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {2.0F, 1.0F, 0.0F, -1.0F, 0.5F};
  std::vector<double> p(logits.size());
  double total = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    p[i] = std::exp(static_cast<double>(logits[i]));
    total += p[i];
  }
  for (double& x : p) {
    x /= total;
  }
  constexpr std::uint64_t kDraws = 200000;
  for (const std::int32_t draft : {0, 1, 3}) {
    std::vector<double> seen(logits.size(), 0.0);
    std::uint64_t accepted = 0;
    for (std::uint64_t s = 0; s < kDraws; ++s) {
      auto v = ex::VerifyDraft(logits, draft, {}, {.seed = s, .stream = 0, .position = 9}, scratch);
      ASSERT_TRUE(v.has_value());
      seen[static_cast<std::size_t>(v->token)] += 1.0;
      accepted += v->accepted ? 1 : 0;
      // A rejected draft is never the token.
      EXPECT_TRUE(v->accepted || v->token != draft);
    }
    EXPECT_NEAR(static_cast<double>(accepted) / kDraws, p[static_cast<std::size_t>(draft)], 0.01)
        << draft;
    for (std::size_t i = 0; i < p.size(); ++i) {
      EXPECT_NEAR(seen[i] / kDraws, p[i], 0.01) << draft << " " << i;
    }
  }
  // Reproducible from the key.
  const auto a = ex::VerifyDraft(logits, 1, {}, {.seed = 3, .stream = 1, .position = 2}, scratch);
  const auto b = ex::VerifyDraft(logits, 1, {}, {.seed = 3, .stream = 1, .position = 2}, scratch);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->accepted, b->accepted);
  EXPECT_EQ(a->token, b->token);
}

// The distribution the documented order gives, computed the plain way (a
// full sort, double softmax): each token's probability.
std::vector<double> Reference(const std::vector<float>& logits, const ex::SamplingParams& p) {
  struct C {
    double value;
    std::int32_t id;
  };
  std::vector<C> c;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (std::isfinite(logits[i])) {
      c.push_back({static_cast<double>(logits[i]) / p.temperature, static_cast<std::int32_t>(i)});
    }
  }
  std::ranges::sort(c, [](const C& a, const C& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
  });
  if (p.top_k != 0 && p.top_k < c.size()) {
    c.resize(p.top_k);
  }
  const double top = c[0].value;
  double total = 0;
  for (C& x : c) {
    x.value = std::exp(x.value - top);
    total += x.value;
  }
  std::size_t keep = c.size();
  while (p.min_p > 0 && keep > 1 && c[keep - 1].value < p.min_p * c[0].value) {
    total -= c[--keep].value;
  }
  if (p.top_p < 1) {
    double cumulative = 0;
    std::size_t n = 0;
    while (n < keep && cumulative < p.top_p * total) {
      cumulative += c[n++].value;
    }
    keep = n;
    total = cumulative;
  }
  std::vector<double> probability(logits.size(), 0.0);
  for (std::size_t i = 0; i < keep; ++i) {
    probability[static_cast<std::size_t>(c[i].id)] = c[i].value / total;
  }
  return probability;
}

// Total variation between draws' counts and a distribution; `outside`:
// draws of tokens it gives no probability.
double TotalVariation(const std::vector<std::uint64_t>& counts, const std::vector<double>& p,
                      std::uint64_t draws, std::uint64_t& outside) {
  double tv = 0;
  outside = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    tv += std::abs((static_cast<double>(counts[i]) / static_cast<double>(draws)) - p[i]);
    outside += p[i] == 0 ? counts[i] : 0;
  }
  return tv / 2;
}

// A bound on the total variation of `draws` exact draws from `p`: three
// times its expectation (each count's mean absolute deviation, about
// sqrt(2 p (1 - p) / (pi draws))), plus a little.
double TvBound(const std::vector<double>& p, std::uint64_t draws) {
  double expected = 0;
  for (const double x : p) {
    expected += std::sqrt(2 * x * (1 - x) / (std::numbers::pi * static_cast<double>(draws)));
  }
  return (3 * expected / 2) + 0.002;
}

// Logits spread over [-4, 4] by a fixed formula, with ties and a masked
// token, so each filter's boundary falls between tokens.
std::vector<float> Spread(std::size_t n) {
  std::vector<float> logits(n);
  for (std::size_t i = 0; i < n; ++i) {
    logits[i] = 4.0F * std::sin((static_cast<float>(i) * 1.7F) + 0.3F);
  }
  logits[n / 2] = -INFINITY;
  logits[3] = logits[5];  // a tie
  return logits;
}

// The draws follow the exact softmax with each filter and their mix
// (total variation over many draws; never a token the filters drop), for
// plain draws and for speculative verdicts on a likely and an unlikely
// draft, on a small vocabulary.
TEST(Sample, DrawsFollowTheExactDistribution) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = Spread(48);
  constexpr std::uint64_t kDraws = 200000;
  const std::array<ex::SamplingParams, 7> cases = {{
      {},
      {.temperature = 0.6F},
      {.temperature = 1.0F, .top_k = 5},
      {.temperature = 1.0F, .top_k = 0, .top_p = 0.8F},
      {.temperature = 1.0F, .top_k = 0, .top_p = 1.0F, .min_p = 0.1F},
      {.temperature = 1.3F, .top_k = 12, .top_p = 0.9F, .min_p = 0.02F},
      {.temperature = 0.9F, .top_k = 0, .top_p = 0.7F, .min_p = 0.05F},
  }};
  const auto greedy = ex::Greedy(logits);
  ASSERT_TRUE(greedy.has_value());
  for (std::size_t k = 0; k < cases.size(); ++k) {
    const ex::SamplingParams& p = cases[k];
    const std::vector<double> want = Reference(logits, p);
    // An unlikely draft that the filters may or may not keep.
    const std::int32_t unlikely = 7;
    for (int mode = 0; mode < 3; ++mode) {
      std::vector<std::uint64_t> counts(logits.size(), 0);
      for (std::uint64_t s = 0; s < kDraws; ++s) {
        const ex::SamplingKey key{.seed = 17, .stream = k, .position = s};
        std::int32_t token = -1;
        if (mode == 0) {
          token = ex::Sample(logits, p, key, scratch).value_or(-1);
        } else {
          const auto v = ex::VerifyDraft(logits, mode == 1 ? *greedy : unlikely, p, key, scratch);
          token = v ? v->token : -1;
        }
        ASSERT_GE(token, 0);
        ++counts[static_cast<std::size_t>(token)];
      }
      std::uint64_t outside = 0;
      EXPECT_LT(TotalVariation(counts, want, kDraws, outside), TvBound(want, kDraws))
          << "case " << k << " mode " << mode;
      EXPECT_EQ(outside, 0U) << "case " << k << " mode " << mode;
      // Each token's count within five standard deviations of its
      // expectation: the total variation alone would miss a 1% mass error
      // (its bound is about 0.02 here), which this catches on any token
      // (at 200,000 draws, 1% is 2,000 draws, five deviations at most
      // about 1,120).
      for (std::size_t i = 0; i < want.size(); ++i) {
        const double mean = want[i] * static_cast<double>(kDraws);
        EXPECT_LE(std::abs(static_cast<double>(counts[i]) - mean),
                  (5 * std::sqrt(mean * (1 - want[i]))) + 5)
            << "case " << k << " mode " << mode << " token " << i;
      }
    }
  }
}

// Filter edges: every logit equal (top-p's boundary bucket holds them
// all, ties by ID), a temperature near 0, top-k beyond the vocabulary,
// one finite logit among -infinity, and min-p with top-p (min-p first).
TEST(Sample, FilterEdges) {
  std::vector<ex::SamplingCandidate> scratch;
  const auto drawn = [&](const std::vector<float>& logits, const ex::SamplingParams& p) {
    std::vector<std::uint64_t> counts(logits.size(), 0);
    for (std::uint64_t s = 0; s < 20000; ++s) {
      const auto t = ex::Sample(logits, p, {.seed = 5, .stream = 0, .position = s}, scratch);
      EXPECT_TRUE(t.has_value());
      if (t) {
        ++counts[static_cast<std::size_t>(*t)];
      }
    }
    return counts;
  };
  const auto kept = [](const std::vector<std::uint64_t>& counts) {
    std::vector<std::size_t> ids;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      if (counts[i] != 0) {
        ids.push_back(i);
      }
    }
    return ids;
  };
  // 1000 equal logits, top-p 0.25: the 250 lowest IDs.
  const std::vector<float> flat(1000, 0.5F);
  const auto nucleus = drawn(flat, {.temperature = 1.0F, .top_k = 0, .top_p = 0.25F});
  const auto ids = kept(nucleus);
  ASSERT_EQ(ids.size(), 250U);
  EXPECT_EQ(ids.front(), 0U);
  EXPECT_EQ(ids.back(), 249U);
  // Top-k past the vocabulary is off; top-p 1 is off.
  EXPECT_EQ(kept(drawn(flat, {.temperature = 1.0F, .top_k = 5000, .top_p = 1.0F})).size(), 1000U);
  // A temperature near 0 draws the most likely token only.
  std::vector<float> spread(64);
  for (std::size_t i = 0; i < spread.size(); ++i) {
    spread[i] = std::sin(static_cast<float>(i));
  }
  const auto cold = kept(drawn(spread, {.temperature = 1e-30F}));
  ASSERT_EQ(cold.size(), 1U);
  EXPECT_EQ(static_cast<std::int32_t>(cold[0]), *ex::Greedy(spread));
  // One finite logit: it, whatever the filters.
  std::vector<float> one(50, -INFINITY);
  one[17] = -3.0F;
  for (const ex::SamplingParams& p :
       {ex::SamplingParams{}, ex::SamplingParams{.temperature = 2.0F, .top_k = 3},
        ex::SamplingParams{.temperature = 0.5F, .top_k = 0, .top_p = 0.2F, .min_p = 0.9F}}) {
    EXPECT_EQ(kept(drawn(one, p)), std::vector<std::size_t>{17});
  }
  // min-p before top-p. Weights 1, 0.8, 0.1, 0.1: min-p 0.5 leaves 1 and
  // 0.8 (total 1.8), and top-p 0.55 of that (0.99) is reached by the first
  // alone; top-p first would need 0.55 of 2.0 (1.1), keeping two.
  const std::vector<float> four = {0.0F, std::log(0.8F), std::log(0.1F), std::log(0.1F)};
  EXPECT_EQ(kept(drawn(four, {.temperature = 1.0F, .top_k = 0, .top_p = 0.55F, .min_p = 0.5F})),
            std::vector<std::size_t>{0});
  EXPECT_EQ(kept(drawn(four, {.temperature = 1.0F, .top_k = 0, .top_p = 0.55F, .min_p = 0.0F})),
            (std::vector<std::size_t>{0, 1}));
  // The same through top-k's path (top-k 3 keeps 1, 0.8 and the lower ID's
  // 0.1).
  EXPECT_EQ(kept(drawn(four, {.temperature = 1.0F, .top_k = 3, .top_p = 0.55F, .min_p = 0.5F})),
            std::vector<std::size_t>{0});
  EXPECT_EQ(kept(drawn(four, {.temperature = 1.0F, .top_k = 3, .top_p = 1.0F, .min_p = 0.0F})),
            (std::vector<std::size_t>{0, 1, 2}));
}

// At a real vocabulary's size: top-k's one pass cuts back many times, and
// top-p's boundary falls in a crowded bucket. The kept set is exactly the
// reference's (every kept token drawn, none other) and the draws follow it.
TEST(Sample, LargeVocabulariesKeepExactlyTheReferenceSet) {
  std::vector<ex::SamplingCandidate> scratch;
  std::vector<float> logits(60000);
  for (std::size_t i = 0; i < logits.size(); ++i) {
    // Ascending in parts, so the top-k pass meets ever better candidates.
    logits[i] = (static_cast<float>(i % 5000) / 1000.0F) + (0.5F * std::sin(static_cast<float>(i)));
  }
  constexpr std::uint64_t kDraws = 10000;
  for (const ex::SamplingParams& p :
       {ex::SamplingParams{.temperature = 0.2F, .top_k = 40},
        ex::SamplingParams{.temperature = 0.05F, .top_k = 0, .top_p = 0.9F},
        ex::SamplingParams{.temperature = 0.05F, .top_k = 0, .top_p = 0.9F, .min_p = 0.2F}}) {
    const std::vector<double> want = Reference(logits, p);
    std::vector<std::uint64_t> counts(logits.size(), 0);
    for (std::uint64_t s = 0; s < kDraws; ++s) {
      const auto t = ex::Sample(logits, p, {.seed = 3, .stream = p.top_k, .position = s}, scratch);
      ASSERT_TRUE(t.has_value());
      ++counts[static_cast<std::size_t>(*t)];
    }
    std::uint64_t outside = 0;
    std::size_t kept = 0;
    std::size_t unseen = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      kept += want[i] > 0 ? 1 : 0;
      unseen += want[i] > 20.0 / kDraws && counts[i] == 0 ? 1 : 0;
    }
    EXPECT_GT(kept, 20U);
    EXPECT_EQ(unseen, 0U) << "tokens the reference keeps, never drawn";
    EXPECT_LT(TotalVariation(counts, want, kDraws, outside), TvBound(want, kDraws))
        << kept << " kept";
    EXPECT_EQ(outside, 0U);
  }
}

TEST(VerifyDraft, GreedyAndFilteredDrafts) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {0.0F, 3.0F, 1.0F};
  // Temperature 0: accepted exactly when the draft is the greedy token.
  auto yes = ex::VerifyDraft(logits, 1, {.temperature = 0}, {}, scratch);
  auto no = ex::VerifyDraft(logits, 2, {.temperature = 0}, {}, scratch);
  ASSERT_TRUE(yes && no);
  EXPECT_TRUE(yes->accepted);
  EXPECT_EQ(yes->token, 1);
  EXPECT_FALSE(no->accepted);
  EXPECT_EQ(no->token, 1);
  // A draft top-k filters out has probability 0: always rejected, and the
  // replacement is from what top-k keeps.
  for (std::uint64_t s = 0; s < 1000; ++s) {
    auto v =
        ex::VerifyDraft(logits, 0, {.top_k = 2}, {.seed = s, .stream = 0, .position = 0}, scratch);
    ASSERT_TRUE(v.has_value());
    EXPECT_FALSE(v->accepted);
    EXPECT_NE(v->token, 0);
  }
  // A draft holding all the weight is always accepted.
  auto only =
      ex::VerifyDraft(logits, 1, {.top_k = 1}, {.seed = 5, .stream = 0, .position = 0}, scratch);
  ASSERT_TRUE(only.has_value());
  EXPECT_TRUE(only->accepted);
  EXPECT_EQ(Failed(ex::VerifyDraft(logits, 1, {.top_p = 0}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
}

}  // namespace
