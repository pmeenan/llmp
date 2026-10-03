// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_EXECUTION_ADAPTIVE_WAVE_MODE_H_
#define JITLLM_EXECUTION_ADAPTIVE_WAVE_MODE_H_

#include <algorithm>
#include <array>
#include <cstdint>

namespace jitllm::execution {

// How a speculative model steps a wave of concurrent requests: each one's
// draft and one joined verify, or one plain decode row each (the drafter
// still fed). A draft-verify wave of `width` requests commits `kept` tokens
// a request (the anchor and its accepted drafts) and costs cost[width]
// plain waves; it is chosen while kept exceeds that cost. The cost is a
// calibration recorded per model (each width's draft-verify wave time over
// its plain wave time, measured); kept is a moving average of the tokens
// draft-verify waves commit a request, counted from the tokens alone. So,
// as AdaptiveDepth says, wall time never changes a schedule: the same
// requests in the same waves choose the same forms, and runs with the same
// waves repeat. Only full verifies in joined waves without a sampling
// member feed the average.
//
// - A lone request, a width without a cost, or a wave with a sampling
//   member speculates (plain and speculative sampling turn a seed into
//   different tokens, so seeded requests take one form throughout).
// - Speculation first; after three draft-verify waves the average decides,
//   switching only past a 3% margin either way. While plain waves run, two
//   draft-verify waves follow every 64 of them, so the average follows
//   acceptance.
// - A forced form (an exactness control) takes every wave of two or more.
// - A width whose verify is cut to fewer rows a request than a narrow
//   wave's (`rows`: DeepSeek's past four requests, each its share of a
//   wave's sixteen rows) commits at most those rows a request: the
//   average counts against that bound (min(kept, rows)), its waves do not
//   feed the average (they would pull it below the full verifies') and it
//   never probes: it follows the average narrower waves' full verifies
//   keep, plain until they have kept enough, and always where the bound
//   cannot pay the width's cost.
class AdaptiveWaveMode {
 public:
  enum class Mode : std::uint8_t { kSpeculative, kPlain };
  enum class Force : std::uint8_t { kNone, kSpeculative, kPlain };
  // The widest wave costed: a speculative model's most request slots (a
  // DSpark verify of at least two of a wave's sixteen rows,
  // docs/runtime-serving.md#request-slots).
  static constexpr std::uint32_t kMaxWidth = 8;
  using Costs = std::array<double, kMaxWidth + 1>;  // by width; 0: none
  // By width, the most tokens a draft-verify wave commits a request when
  // its verify is cut below a full one's; 0: a full verify.
  using Rows = std::array<std::uint32_t, kMaxWidth + 1>;

  explicit AdaptiveWaveMode(Costs cost = {}, Force force = Force::kNone, Rows rows = {})
      : cost_(cost), force_(force), rows_(rows) {}

  Mode Choose(std::uint32_t width, bool sampling = false) const {
    if (width < 2 || width > kMaxWidth) {
      return Mode::kSpeculative;
    }
    if (force_ != Force::kNone) {
      return force_ == Force::kPlain ? Mode::kPlain : Mode::kSpeculative;
    }
    if (sampling || cost_[width] <= 0) {
      return Mode::kSpeculative;
    }
    if (rows_[width] != 0) {
      // Its own waves cannot teach the average: it follows what the full
      // verifies of narrower waves have shown, plain until they have.
      const bool pays = rows_[width] > cost_[width] && seen_ >= kExplore;
      return pays ? preferred_[width] : Mode::kPlain;
    }
    if (probe_left_ > 0 || seen_ < kExplore) {
      return Mode::kSpeculative;
    }
    return preferred_[width];
  }

  // A completed wave of `width` requests in `mode`. For a draft-verify
  // wave, the tokens committed by its `complete` requests: those whose
  // verify took its full rows in the joined wave (as AdaptiveDepth counts
  // only complete steps: a verify cut short by a mask width or the reply's
  // end, or run alone, says little about acceptance). A draft-verify wave
  // with none is not observed. Waves chosen for a sampling member should
  // not be observed either (speculative sampling accepts differently).
  void Observe(std::uint32_t width, Mode mode, std::uint32_t tokens = 0,
               std::uint32_t complete = 0) {
    if (width < 2 || width > kMaxWidth) {
      return;
    }
    if (mode == Mode::kPlain) {
      if (++since_probe_ >= kProbeEvery) {
        since_probe_ = 0;
        probe_left_ = kProbe;
      }
      return;
    }
    if (complete == 0 || complete > width || tokens == 0) {
      return;
    }
    if (probe_left_ > 0) {
      --probe_left_;
    }
    if (rows_[width] != 0) {
      return;  // a cut verify: below the full verifies' acceptance
    }
    const double kept = static_cast<double>(tokens) / complete;
    ++seen_;
    kept_ = seen_ == 1 ? kept : kept_ + ((kept - kept_) / 8);
    if (seen_ < kExplore) {
      return;
    }
    for (std::uint32_t w = 2; w <= kMaxWidth; ++w) {
      if (cost_[w] <= 0) {
        continue;
      }
      const double can = rows_[w] != 0 ? std::min<double>(kept_, rows_[w]) : kept_;
      if (preferred_[w] == Mode::kSpeculative && can * 1.03 < cost_[w]) {
        preferred_[w] = Mode::kPlain;
      } else if (preferred_[w] == Mode::kPlain && can > cost_[w] * 1.03) {
        preferred_[w] = Mode::kSpeculative;
      }
    }
  }

  // The moving average of tokens a draft-verify wave commits a request.
  double kept() const { return kept_; }

  bool operator==(const AdaptiveWaveMode&) const = default;

 private:
  static constexpr std::uint64_t kExplore = 3;
  static constexpr std::uint32_t kProbe = 2;
  static constexpr std::uint32_t kProbeEvery = 64;
  Costs cost_;
  Force force_;
  Rows rows_;
  std::uint64_t seen_ = 0;
  double kept_ = 0;
  std::array<Mode, kMaxWidth + 1> preferred_ = [] {
    std::array<Mode, kMaxWidth + 1> all{};
    all.fill(Mode::kSpeculative);
    return all;
  }();
  std::uint32_t since_probe_ = 0;
  std::uint32_t probe_left_ = 0;
};

}  // namespace jitllm::execution

#endif  // JITLLM_EXECUTION_ADAPTIVE_WAVE_MODE_H_
