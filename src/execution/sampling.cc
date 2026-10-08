// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/sampling.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace llmp::execution {

std::string_view SamplingErrorName(SamplingError e) {
  switch (e) {
    case SamplingError::kNoLogits:
      return "no-logits";
    case SamplingError::kInvalidLogits:
      return "invalid-logits";
    case SamplingError::kInvalidParams:
      return "invalid-params";
  }
  return "unknown";
}

std::array<std::uint32_t, 4> Philox4x32(std::array<std::uint32_t, 4> c,
                                        std::array<std::uint32_t, 2> k) {
  constexpr std::uint64_t kM0 = 0xD2511F53U;
  constexpr std::uint64_t kM1 = 0xCD9E8D57U;
  constexpr std::uint32_t kW0 = 0x9E3779B9U;
  constexpr std::uint32_t kW1 = 0xBB67AE85U;
  for (int round = 0; round < 10; ++round) {
    const std::uint64_t p0 = kM0 * c[0];
    const std::uint64_t p1 = kM1 * c[2];
    const auto hi0 = static_cast<std::uint32_t>(p0 >> 32U);
    const auto lo0 = static_cast<std::uint32_t>(p0);
    const auto hi1 = static_cast<std::uint32_t>(p1 >> 32U);
    const auto lo1 = static_cast<std::uint32_t>(p1);
    c = {hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0};
    k[0] += kW0;
    k[1] += kW1;
  }
  return c;
}

double UniformAt(const SamplingKey& key) {
  const std::array<std::uint32_t, 4> counter = {
      static_cast<std::uint32_t>(key.position), static_cast<std::uint32_t>(key.position >> 32U),
      static_cast<std::uint32_t>(key.stream), static_cast<std::uint32_t>(key.stream >> 32U)};
  const std::array<std::uint32_t, 2> k = {static_cast<std::uint32_t>(key.seed),
                                          static_cast<std::uint32_t>(key.seed >> 32U)};
  const auto r = Philox4x32(counter, k);
  const std::uint64_t bits = (static_cast<std::uint64_t>(r[0] >> 5U) << 26U) | (r[1] >> 6U);
  return static_cast<double>(bits) * 0x1.0p-53;
}

namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

// One pass over the logits: whether they may be sampled, their largest
// and how many are finite.
struct Scan {
  float max = -kInfinity;
  std::size_t finite = 0;
};

std::expected<Scan, SamplingError> ScanLogits(std::span<const float> logits) {
  if (logits.empty()) {
    return std::unexpected(SamplingError::kNoLogits);
  }
  // Branch-free, so it vectorizes: a NaN or +infinity anywhere refuses them.
  unsigned bad = 0;
  Scan scan;
  for (const float x : logits) {
    bad |= static_cast<unsigned>(std::isnan(x)) | static_cast<unsigned>(x == kInfinity);
    scan.finite += static_cast<std::size_t>(x > -kInfinity);
    scan.max = x > scan.max ? x : scan.max;
  }
  if (bad != 0) {
    return std::unexpected(SamplingError::kInvalidLogits);
  }
  if (scan.finite == 0) {
    return std::unexpected(SamplingError::kNoLogits);
  }
  return scan;
}

}  // namespace

std::expected<std::int32_t, SamplingError> Greedy(std::span<const float> logits) {
  const auto scan = ScanLogits(logits);
  if (!scan) {
    return std::unexpected(scan.error());
  }
  // The validating scan already found the maximum; its first occurrence
  // is the lowest ID among ties.
  return static_cast<std::int32_t>(std::ranges::find(logits, scan->max) - logits.begin());
}

std::expected<TokenScores, SamplingError> ScoreToken(std::span<const float> logits,
                                                     std::int32_t token, std::uint32_t top_count) {
  // Up to the whole row (D-102 raised the API's 5 to the vocabulary); more
  // than the row asks for all of it.
  if (token < 0 || std::cmp_greater_equal(token, logits.size())) {
    return std::unexpected(SamplingError::kInvalidParams);
  }
  const auto scan = ScanLogits(logits);
  if (!scan) {
    return std::unexpected(scan.error());
  }
  if (!std::isfinite(logits[static_cast<std::size_t>(token)])) {
    return std::unexpected(SamplingError::kInvalidLogits);
  }
  const auto at = [&](std::int32_t id) { return logits[static_cast<std::size_t>(id)]; };
  // More likely first, then the lower ID.
  const auto before = [&](std::int32_t a, std::int32_t b) {
    return at(a) > at(b) || (at(a) == at(b) && a < b);
  };
  // Few top scores (the common case) are kept by insertion as the scan
  // goes; many by a partial sort of every finite logit's ID: O(V log k)
  // either way, never O(V k).
  constexpr std::size_t kInsertionMost = 16;
  double sum = 0;
  std::uint32_t token_rank = 1;
  std::vector<std::int32_t> best;
  const std::size_t keep = std::min<std::size_t>(std::max(top_count, 1U), logits.size());
  if (keep > kInsertionMost) {
    best.reserve(logits.size());
  }
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double value = logits[i];
    if (!std::isfinite(value)) {
      continue;  // the scan already refused NaN and +infinity
    }
    sum += std::exp(value - scan->max);
    token_rank += static_cast<std::uint32_t>(value > logits[static_cast<std::size_t>(token)]);
    if (keep > kInsertionMost) {
      best.push_back(static_cast<std::int32_t>(i));
      continue;
    }
    const auto place = std::ranges::find_if(
        best, [&](std::int32_t id) { return value > logits[static_cast<std::size_t>(id)]; });
    if (std::cmp_less(best.size(), keep) || place != best.end()) {
      best.insert(place, static_cast<std::int32_t>(i));
      if (best.size() > keep) {
        best.pop_back();
      }
    }
  }
  if (keep > kInsertionMost) {
    const auto end = best.begin() + static_cast<std::ptrdiff_t>(std::min(keep, best.size()));
    std::partial_sort(best.begin(), end, best.end(), before);
    best.erase(end, best.end());
  }
  const double norm = std::log(sum);
  const auto score = [&](std::int32_t id) {
    return (static_cast<double>(at(id)) - scan->max) - norm;
  };
  TokenScores out{.logprob = score(token), .top = {}};
  out.top.reserve(best.size() + 1);
  // `best` is in order: a rank is one more than the entries strictly above,
  // so equal logits share the first one's.
  std::uint32_t rank = 1;
  for (std::size_t j = 0; j < best.size(); ++j) {
    if (j > 0 && at(best[j]) != at(best[j - 1])) {
      rank = static_cast<std::uint32_t>(j + 1);
    }
    out.top.push_back({.id = best[j], .logprob = score(best[j]), .rank = rank});
  }
  if (std::ranges::find(best, token) == best.end()) {
    out.top.push_back({.id = token, .logprob = out.logprob, .rank = token_rank});
  }
  return out;
}

namespace {

bool ParamsValid(const SamplingParams& p) {
  // NaN fails every comparison, so each range is checked for it too.
  return std::isfinite(p.temperature) && p.temperature >= 0 && !std::isnan(p.top_p) &&
         p.top_p > 0 && p.top_p <= 1 && !std::isnan(p.min_p) && p.min_p >= 0 && p.min_p <= 1;
}

// Candidates ranked for top-k and top-p: more likely first, then the lower
// ID, so ties never depend on sort stability.
bool Before(const SamplingCandidate& a, const SamplingCandidate& b) {
  return a.value > b.value || (a.value == b.value && a.id < b.id);
}

// What a draw takes from: `scratch`'s first `keep` candidates, in draw
// order, each with its unnormalized weight (0: never drawn), `total` their
// sum. Dense: every token, candidate i being token i.
struct Kept {
  std::size_t keep = 0;
  double total = 0;
  bool dense = false;
};

// The bucket of a weight in (0, 1] for top-p's first cut, from its bits
// (monotonic in a positive float's value): eight a binary octave below
// 1.0, which is bucket 0, down to 2^-64; everything smaller in the last.
constexpr std::uint32_t kBuckets = 512;
std::uint32_t Bucket(float w) {
  constexpr std::uint32_t kOne = 0x3f800000U;  // 1.0F
  return std::min((kOne - std::bit_cast<std::uint32_t>(w)) >> 20U, kBuckets - 1);
}

// top-p over candidates ranked by Before: the shortest prefix whose weight
// reaches `target`.
Kept Nucleus(std::vector<SamplingCandidate>& scratch, std::size_t count, double target) {
  std::sort(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(count), Before);
  double cumulative = 0;
  std::size_t n = 0;
  while (n < count) {
    cumulative += scratch[n].value;
    ++n;
    if (cumulative >= target) {
      break;
    }
  }
  return {.keep = n, .total = cumulative, .dense = false};
}

// The top_k highest logits (ties to the lower ID) into `scratch`, as
// candidates valued by logit: one pass that keeps what beats the k-th best
// seen so far, cutting back to k whenever the buffer fills.
void TopK(std::span<const float> logits, std::size_t k, std::vector<SamplingCandidate>& scratch) {
  const std::size_t cap = std::max<std::size_t>(2 * k, 1024);
  scratch.clear();
  scratch.reserve(cap);
  const auto cut = [&] {
    std::nth_element(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(k - 1),
                     scratch.end(), Before);
    scratch.resize(k);
  };
  // A later logit equal to the k-th best ranks after it (a higher ID), so
  // only a greater one can enter.
  float floor = -kInfinity;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (logits[i] > floor) {
      scratch.push_back({static_cast<double>(logits[i]), static_cast<std::int32_t>(i)});
      if (scratch.size() == cap) {
        cut();
        floor =
            static_cast<float>(std::ranges::min_element(scratch, [](const auto& a, const auto& b) {
                                 return a.value < b.value;
                               })->value);
      }
    }
  }
  if (scratch.size() > k) {
    cut();
  }
}

// Temperature, top-k, softmax, min-p and top-p, in that order (p.temperature
// above 0, the logits scanned). Weights are relative to the most likely
// token's, which is exactly 1.
Kept Distribution(std::span<const float> logits, const SamplingParams& p, const Scan& scan,
                  std::vector<SamplingCandidate>& scratch) {
  const double inverse = 1.0 / static_cast<double>(p.temperature);
  const auto weight = [&](float x) {
    // A float exponential: the weights need no more, and it is the pass's cost.
    return std::exp(static_cast<float>((static_cast<double>(x) - scan.max) * inverse));
  };
  const float floor = p.min_p;  // min-p: less likely than min_p times the most likely
  if (p.top_k != 0 && p.top_k < scan.finite) {
    TopK(logits, p.top_k, scratch);
    std::size_t kept = 0;
    double total = 0;
    for (const SamplingCandidate& c : scratch) {
      const float w = weight(static_cast<float>(c.value));
      if (w >= floor) {
        scratch[kept++] = {static_cast<double>(w), c.id};
        total += w;
      }
    }
    if (p.top_p < 1) {
      return Nucleus(scratch, kept, static_cast<double>(p.top_p) * total);
    }
    return {.keep = kept, .total = total, .dense = false};
  }
  // Every token, in ID order.
  scratch.resize(logits.size());
  std::array<double, kBuckets> buckets{};
  const bool nucleus = p.top_p < 1;
  double total = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    float w = weight(logits[i]);
    w = w >= floor ? w : 0.0F;
    scratch[i] = {static_cast<double>(w), static_cast<std::int32_t>(i)};
    total += w;
    if (nucleus && w > 0) {
      buckets[Bucket(w)] += w;
    }
  }
  if (!nucleus) {
    return {.keep = logits.size(), .total = total, .dense = true};
  }
  // top-p without a full sort. The buckets, most likely first, until their
  // weight reaches the target: those before the last are wholly in the
  // nucleus, in any order; only the last one's candidates are sorted, and
  // taken until the target is reached.
  const double target = static_cast<double>(p.top_p) * total;
  double cumulative = 0;
  std::uint32_t last = kBuckets - 1;
  for (std::uint32_t b = 0; b < kBuckets; ++b) {
    cumulative += buckets[b];
    if (cumulative >= target) {
      last = b;
      break;
    }
  }
  std::size_t count = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const auto w = static_cast<float>(scratch[i].value);
    if (w > 0 && Bucket(w) <= last) {
      scratch[count++] = scratch[i];
    }
  }
  const auto end = scratch.begin() + static_cast<std::ptrdiff_t>(count);
  const auto boundary = std::partition(scratch.begin(), end, [last](const SamplingCandidate& c) {
    return Bucket(static_cast<float>(c.value)) < last;
  });
  double kept = 0;
  for (auto it = scratch.begin(); it != boundary; ++it) {
    kept += it->value;
  }
  std::sort(boundary, end, Before);
  auto it = boundary;
  while (it != end && kept < target) {
    kept += it->value;
    ++it;
  }
  // (Only the summation's rounding can leave the target unreached with
  // the last bucket taken whole: then that is the nucleus.)
  return {.keep = static_cast<std::size_t>(it - scratch.begin()), .total = kept, .dense = false};
}

// The candidate `u` (in [0, 1)) falls on, weights summing to `total`,
// skipping `skip` (-1: none).
std::int32_t Draw(const std::vector<SamplingCandidate>& scratch, Kept kept, double u,
                  std::int32_t skip) {
  const double target = u * kept.total;
  double cumulative = 0;
  std::int32_t last = -1;
  for (std::size_t i = 0; i < kept.keep; ++i) {
    const SamplingCandidate& c = scratch[i];
    if (c.value == 0 || c.id == skip) {
      continue;
    }
    last = c.id;
    cumulative += c.value;
    if (target < cumulative) {
      return c.id;
    }
  }
  return last;
}

}  // namespace

std::expected<std::int32_t, SamplingError> Sample(std::span<const float> logits,
                                                  const SamplingParams& p, const SamplingKey& key,
                                                  std::vector<SamplingCandidate>& scratch) {
  if (!ParamsValid(p)) {
    return std::unexpected(SamplingError::kInvalidParams);
  }
  if (p.temperature == 0 || p.top_k == 1) {
    return Greedy(logits);
  }
  auto scan = ScanLogits(logits);
  if (!scan) {
    return std::unexpected(scan.error());
  }
  const Kept kept = Distribution(logits, p, *scan, scratch);
  return Draw(scratch, kept, UniformAt(key), -1);
}

std::expected<DraftVerdict, SamplingError> VerifyDraft(std::span<const float> logits,
                                                       std::int32_t draft, const SamplingParams& p,
                                                       const SamplingKey& key,
                                                       std::vector<SamplingCandidate>& scratch) {
  if (!ParamsValid(p)) {
    return std::unexpected(SamplingError::kInvalidParams);
  }
  if (p.temperature == 0 || p.top_k == 1) {
    auto greedy = Greedy(logits);
    if (!greedy) {
      return std::unexpected(greedy.error());
    }
    return DraftVerdict{.accepted = *greedy == draft, .token = *greedy};
  }
  auto scan = ScanLogits(logits);
  if (!scan) {
    return std::unexpected(scan.error());
  }
  const Kept kept = Distribution(logits, p, *scan, scratch);
  double weight = 0;
  if (kept.dense) {
    if (draft >= 0 && std::cmp_less(draft, kept.keep)) {
      weight = scratch[static_cast<std::size_t>(draft)].value;
    }
  } else {
    for (std::size_t i = 0; i < kept.keep; ++i) {
      if (scratch[i].id == draft) {
        weight = scratch[i].value;
      }
    }
  }
  // Accepted with the draft's probability.
  const SamplingKey accept{
      .seed = key.seed, .stream = key.stream ^ kAcceptStream, .position = key.position};
  if (weight > 0 && UniformAt(accept) * kept.total < weight) {
    return DraftVerdict{.accepted = true, .token = draft};
  }
  // Rejected: the distribution without the draft, renormalized. (A draft
  // holding all the weight is never rejected: its probability is 1.)
  const Kept rest{.keep = kept.keep, .total = kept.total - weight, .dense = kept.dense};
  return DraftVerdict{.accepted = false, .token = Draw(scratch, rest, UniformAt(key), draft)};
}

}  // namespace llmp::execution
