// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_EXECUTION_ADAPTIVE_DEPTH_H_
#define JITLLM_EXECUTION_ADAPTIVE_DEPTH_H_

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace jitllm::execution {

// Compare adjacent draft depths by accepted tokens per calibrated step
// cost. The default relative cost (depth3/depth2) is 1.16, measured on
// Qwen3.8 at 128K; callers may supply another model's measurement. Wall
// time never changes a schedule: copying this value with conversation
// state preserves uninterrupted-versus-restored results. Explore each
// depth for four complete steps, longer first, then use a moving acceptance average;
// probe the other depth for four steps after 32 observations. Change the
// preference only for a 3% gain. Tail steps teach nothing: the output or
// context limit truncated their verify, not the drafter's acceptance.
class AdaptiveDepth {
 public:
  explicit AdaptiveDepth(std::uint32_t maximum = 3, double relative_cost = 1.16)
      : maximum_(std::clamp(maximum, 1U, 7U)),
        minimum_(std::max(1U, maximum_ - 1)),
        preferred_(maximum_),
        relative_cost_(std::isfinite(relative_cost) && relative_cost > 0 ? relative_cost : 1.16) {}

  std::uint32_t Choose() const {
    if (minimum_ == maximum_) {
      return maximum_;
    }
    if (stats_[1].seen < 4) {
      return maximum_;
    }
    if (stats_[0].seen < 4) {
      return minimum_;
    }
    return probe_left_ > 0 ? probe_depth_ : preferred_;
  }

  void Observe(std::uint32_t depth, std::uint32_t kept, bool complete = true) {
    if (!complete || (depth != minimum_ && depth != maximum_) || kept == 0 || kept > depth + 1) {
      return;
    }
    Stats& s = stats_[depth == minimum_ ? 0 : 1];
    ++s.seen;
    if (s.seen == 1) {
      s.kept = kept;
    } else {
      s.kept += (kept - s.kept) / 8;
    }
    if (probe_left_ > 0) {
      --probe_left_;
    }
    if (stats_[0].seen >= 4 && stats_[1].seen >= 4 && probe_left_ == 0) {
      const auto current = preferred_ == minimum_ ? 0U : 1U;
      const auto other = 1U - current;
      const auto cost = [&](std::uint32_t index) { return index == 0 ? 1.0 : relative_cost_; };
      if (stats_[other].kept / cost(other) > 1.03 * stats_[current].kept / cost(current)) {
        preferred_ = Other(preferred_);
      }
      if (++since_probe_ >= 32) {
        since_probe_ = 0;
        probe_depth_ = Other(preferred_);
        probe_left_ = 4;
      }
    }
  }

  bool operator==(const AdaptiveDepth&) const = default;

  std::uint32_t maximum() const { return maximum_; }
  double relative_cost() const { return relative_cost_; }
  // The same state at another relative cost (a calibration measured since
  // it was saved, D-103); an invalid cost keeps its own.
  AdaptiveDepth WithCost(double relative_cost) const {
    AdaptiveDepth out = *this;
    if (std::isfinite(relative_cost) && relative_cost > 0) {
      out.relative_cost_ = relative_cost;
    }
    return out;
  }

  // Its whole state as words (a kept conversation's record, D-105), and
  // back: Load refuses words that no state Save could give (a depth out of
  // its range, a probe of another depth, a count or average not a number).
  static constexpr std::size_t kWords = 11;
  std::array<std::uint64_t, kWords> Save() const {
    return {maximum_,
            minimum_,
            preferred_,
            std::bit_cast<std::uint64_t>(relative_cost_),
            since_probe_,
            probe_depth_,
            probe_left_,
            stats_[0].seen,
            std::bit_cast<std::uint64_t>(stats_[0].kept),
            stats_[1].seen,
            std::bit_cast<std::uint64_t>(stats_[1].kept)};
  }
  static std::optional<AdaptiveDepth> Load(std::span<const std::uint64_t> words) {
    if (words.size() != kWords || words[0] < 1 || words[0] > 7) {
      return std::nullopt;
    }
    const auto cost = std::bit_cast<double>(words[3]);
    AdaptiveDepth out(static_cast<std::uint32_t>(words[0]), cost);
    const auto kept0 = std::bit_cast<double>(words[8]);
    const auto kept1 = std::bit_cast<double>(words[10]);
    const auto valid_kept = [](double kept) { return std::isfinite(kept) && kept >= 0; };
    if (out.relative_cost_ != cost || words[1] != out.minimum_ ||
        (words[2] != out.minimum_ && words[2] != out.maximum_) || words[4] >= 32 ||
        (words[5] != 0 && words[5] != out.minimum_ && words[5] != out.maximum_) || words[6] > 4 ||
        (words[6] != 0 && words[5] == 0) || !valid_kept(kept0) || !valid_kept(kept1)) {
      return std::nullopt;
    }
    out.preferred_ = static_cast<std::uint32_t>(words[2]);
    out.since_probe_ = static_cast<std::uint32_t>(words[4]);
    out.probe_depth_ = static_cast<std::uint32_t>(words[5]);
    out.probe_left_ = static_cast<std::uint32_t>(words[6]);
    out.stats_[0] = {.seen = words[7], .kept = kept0};
    out.stats_[1] = {.seen = words[9], .kept = kept1};
    return out;
  }

 private:
  struct Stats {
    std::uint64_t seen = 0;
    double kept = 0;
    bool operator==(const Stats&) const = default;
  };
  std::uint32_t Other(std::uint32_t depth) const { return depth == minimum_ ? maximum_ : minimum_; }
  std::uint32_t maximum_;
  std::uint32_t minimum_;
  std::uint32_t preferred_;
  double relative_cost_;
  std::uint32_t since_probe_ = 0;
  std::uint32_t probe_depth_ = 0;
  std::uint32_t probe_left_ = 0;
  std::array<Stats, 2> stats_{};
};

}  // namespace jitllm::execution

#endif  // JITLLM_EXECUTION_ADAPTIVE_DEPTH_H_
