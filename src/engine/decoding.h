// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Decoding modes (D-068; docs/engine.md, "Decoding modes";
// docs/engine-components.md, "Execution"): how a unit of generation runs on
// the one model runner, written once for every family.
//
// - Plain, sampled and scored modes are single-pass: one decode row a slot
//   (its token chosen on the device when greedy, else its logits returned
//   for the caller's sampler), or a scored run of given tokens (every row's
//   outputs).
// - Speculative decoding with any number of drafters, each a pipeline stage
//   with its own state (SlotLifecycle): an autoregressive drafter proposes
//   a token a pass (Qwen3.8's MTP, Gemma4's assistant), a block drafter a
//   block of them in one pass from its anchor and a fill token (DSpark).
//   The drafters combine as a cascade (drafter i extends drafter i - 1's
//   proposal; one verify checks them all) or as alternatives (one a unit,
//   chosen by the alias policy, else by the acceptance each has shown).
//   Then one verify on the target (1 + every proposal's rows, its plan
//   row-invariant in the exact mode, D-092) saves what it writes through
//   the stages' state contracts (model::WritesFor), the caller's judge
//   (greedy agreement, speculative sampling) says how many rows stay, and
//   the slot accepts them (truncation, or a snapshot's restore and the
//   model's commit) or, when the judge refuses, discards the verify whole.
// - An autoregressive drafter whose state for committed positions is
//   rewritten from the target's features (`catch_up`: MTP) starts each
//   unit's first pass where its state stops being the committed tokens'
//   (the verify's first row after a verify), so its first pass catches up
//   the rows the last verify kept. One further behind than its chunk bound
//   (an alternative not chosen for a while) catches up its last chunk's
//   rows only.
// - A drafter that reads the target's state (`borrows_target`: Gemma4's
//   assistant reading the target's KV) borrows the target's stage frozen
//   for its passes (SlotLifecycle::Borrow), after what the last verify
//   owes has run. An anchored one (`anchored`: the assistant) runs every
//   pass at the committed positions, its token the latest proposal; one
//   with `features` reads, beside its tokens, the target's tapped features
//   on a unit's first pass and its own last next features after (the
//   runner copies both on the device).
// - Depth adapts between units (execution::AdaptiveDepth, per slot and
//   drafter), and only complete steps teach it (a verify cut by the
//   reply's end or the context, or a cascade's earlier drafter rejected,
//   says nothing of this drafter's acceptance).
// - Block diffusion (DiffusionGemma's canvas) is a declared kind, not built:
//   a mode of it is refused.
//
// Isolation: every step runs through RunWave (wave_composer.h), so one
// slot's refusal never fails its peers; a drafter refused for a slot
// leaves that slot's unit with the proposals it has (none: a verify of its
// anchor alone, a plain decode in verify form).
//
// What bytes mean (a drafter's proposals, a verify's verdict) is the
// caller's (DecodingPolicy): the serving adapter's samplers, the tests'
// fakes.
//
// Part of the engine core.

#ifndef LLMP_ENGINE_DECODING_H_
#define LLMP_ENGINE_DECODING_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "engine/model_runner.h"
#include "engine/run_stats.h"
#include "engine/wave_composer.h"
#include "execution/adaptive_depth.h"

namespace llmp::engine {

struct DrafterSpec {
  enum class Shape : std::uint8_t {
    kAutoregressive,  // a proposal a pass, each pass one row (its first, a catch-up)
    kBlock,           // max_depth proposals in one pass of that many rows
  };
  std::uint16_t stage = 1;  // its SlotLifecycle stage (a drafter: not 0)
  Shape shape = Shape::kAutoregressive;
  std::uint32_t max_depth = 1;  // proposals a unit, at most
  // Autoregressive: its first pass of a unit starts where its state stops
  // being the committed tokens' (its catch-up: MTP).
  bool catch_up = false;
  // Block: the token of every row after its anchor (DSpark's mask token).
  std::int32_t fill = 0;
  // Reads the target's state, borrowed frozen while it drafts.
  bool borrows_target = false;
  // Autoregressive: every pass at the slot's committed positions (Gemma 4's
  // assistant, whose stage reads the target's frozen state through them
  // and stores none of its rows): a pass's one row is its latest
  // proposal's token (the anchor's first), at the anchor's position.
  bool anchored = false;
  // Its rows read a feature beside their tokens (ModelRunner FeatureFrom):
  // a unit's first pass its stage's tap at the positions before its rows
  // (the target's features there), a later pass its own last pass's next
  // feature (the runner's carry).
  bool features = false;
  // Its depth adapts between units (execution::AdaptiveDepth over
  // max_depth and max_depth - 1, at that cost ratio); else max_depth.
  bool adaptive = false;
  double depth_cost = 1.16;
  // Alternatives: its unit's cost relative to the others' (the choice
  // weighs the tokens it commits by it).
  double unit_cost = 1.0;
  // Return native I32 vocabulary tokens (including a selected-head ID map),
  // instead of F32 head rows. Any next-feature output follows those tokens.
  bool device_tokens = false;
  // Run the complete requested autoregressive pass count before truncating
  // the verify at the reply boundary. Refused unless this is the sole drafter.
  bool draft_before_limit = false;
};

struct DecodingMode {
  enum class Kind : std::uint8_t { kPlain, kSampled, kScored, kSpeculative, kBlockDiffusion };
  enum class Drafters : std::uint8_t { kCascade, kAlternatives };
  Kind kind = Kind::kPlain;
  std::vector<DrafterSpec> drafters{};
  Drafters combine = Drafters::kCascade;
  // The verify's plan is row-invariant (D-092's exact mode): each owner
  // reads what it reads alone.
  bool row_invariant = false;
  // A verify's most rows: the anchor and every drafter's depth (a
  // cascade's), or the deepest's (alternatives').
  std::uint32_t verify_rows() const;
};

// One slot's work for a unit.
struct DecodeWork {
  std::uint32_t slot = 0;
  // Every token through the anchor (the last sampled token, at the slot's
  // positions, not yet in its state): history.size() == positions + 1.
  std::span<const std::int32_t> history{};
  // Tokens it may still emit (at least 1): a verify takes at most so many
  // rows.
  std::uint32_t left = std::numeric_limits<std::uint32_t>::max();
  bool greedy = false;  // plain or verify: tokens chosen on the device
  // Scored: the tokens whose every row's outputs return, from the slot's
  // positions.
  std::span<const std::int32_t> score{};
};

// A verify's verdict: its first `keep` rows stay (the anchor and the
// accepted drafts, so 1 to its rows), and the tokens they commit.
struct Verdict {
  std::uint32_t keep = 1;
  std::vector<std::int32_t> tokens{};
};

struct DecodingPolicy {
  // A drafter step's proposals from its segment's own outputs: at most
  // `want` (fewer stop the drafter for this unit).
  std::function<std::expected<std::vector<std::int32_t>, std::string>(
      std::uint32_t slot, std::size_t drafter, std::span<const std::byte> outputs,
      std::uint32_t want)>
      propose;
  // The verdict on a verify of `verified` (the anchor, then the proposals)
  // from its segment's own outputs (every row's); refused: the verify is
  // discarded whole.
  std::function<std::expected<Verdict, std::string>(std::uint32_t slot,
                                                    std::span<const std::int32_t> verified,
                                                    std::span<const std::byte> outputs)>
      judge;
  // Alternatives: the drafter for this slot's unit (the alias policy);
  // none: the adaptive choice.
  std::function<std::optional<std::size_t>(std::uint32_t slot)> choose;
  // Whether a unit of `width` slots speculates (execution::
  // AdaptiveWaveMode's choice of a wave's form); false: one plain decode
  // row each, which still feeds the drafters' taps. Unset: always.
  std::function<bool(std::uint32_t width)> speculate;
  // Optional caller-owned schedule (for conversation-persisted adaptation).
  // A depth is validated against the declared maximum before drafting.
  std::function<std::uint32_t(std::uint32_t slot, std::size_t drafter)> depth;
  std::function<void(std::uint32_t slot, std::size_t drafter, std::uint32_t depth,
                     std::uint32_t kept, bool complete)>
      observe;
};

// One drafter's part of a slot's unit.
struct DraftRun {
  std::size_t drafter = 0;
  std::uint32_t depth = 0;     // asked
  std::uint32_t proposed = 0;  // given
  std::uint32_t steps = 0;
  std::optional<StepRefusal> refusal{};  // it stopped early: why
};

struct DecodeOutcome {
  // The unit's step (the plain step, or the verify) did not run, or the
  // verdict was not taken: nothing committed.
  std::optional<StepRefusal> refusal{};
  std::vector<DraftRun> drafts{};
  std::vector<std::int32_t> proposed{};  // what the verify checked after its anchor
  std::uint32_t kept = 0;                // positions committed
  std::vector<std::int32_t> tokens{};    // the verdict's
  bool discarded = false;                // the judge refused: the verify undone whole
  // The step's (plain, scored or the verify's) own outputs, its joined
  // outputs and where it sat in its step (WaveOutcome's).
  std::vector<std::byte> outputs{};
  std::vector<std::byte> joined{};
  std::uint32_t segment = 0, first_row = 0, first_output = 0;
  RunPath path = RunPath::kEager;
};

struct DecodeReport {
  std::vector<DecodeOutcome> outcomes;  // by work
  std::uint32_t steps = 0;              // steps run, every drafter's and the verify's
  bool speculated = false;              // drafted and verified, not a plain decode
};

// The choice among alternative drafters: each explored for four units in
// order, then the one whose units commit the most tokens a unit (a moving
// average) for its cost, switching only past a 3% margin.
class DrafterChoice {
 public:
  // Each alternative's unit cost (DrafterSpec::unit_cost).
  explicit DrafterChoice(std::vector<double> costs = {});
  std::size_t Choose() const;
  // A unit of `drafter` committed `committed` tokens.
  void Observe(std::size_t drafter, std::uint32_t committed);
  double kept(std::size_t drafter) const { return stats_.at(drafter).kept; }
  std::uint64_t seen(std::size_t drafter) const { return stats_.at(drafter).seen; }

 private:
  static constexpr std::uint64_t kExplore = 4;
  struct Stats {
    std::uint64_t seen = 0;
    double kept = 0;
    double cost = 1;
  };
  std::vector<Stats> stats_;
  std::size_t preferred_ = 0;
};

class DecodingOrchestrator {
 public:
  // Over `runner` and `composer`, which outlive it (it keeps both by
  // reference). Refused for a block-diffusion mode (declared, not built),
  // drafters in a mode without speculation, a speculative mode without
  // drafters or a policy, a drafter on the target's stage, a stage twice or
  // none of the runner's, a verify wider than the runner's snapshots or a
  // drafter's chunk bound, a drafter reading the target's state without
  // borrowing it, or a composer whose stages are not the runner's.
  static std::expected<DecodingOrchestrator, std::string> Make(ModelRunner& runner,
                                                               const WaveComposer& composer,
                                                               DecodingMode mode,
                                                               DecodingPolicy policy);

  const DecodingMode& mode() const { return mode_; }
  // One unit for each work's slot.
  DecodeReport Step(std::span<const DecodeWork> work);

  // A slot's drafters' state now matches its committed positions (after a
  // prefill, a clear, a restore): each catch-up starts there.
  void Reset(std::uint32_t slot);
  std::uint32_t cursor(std::uint32_t slot, std::size_t drafter) const;
  // A slot's drafter's adaptive depth (a kept conversation's, D-105).
  const execution::AdaptiveDepth& depth(std::uint32_t slot, std::size_t drafter) const;
  void SetDepth(std::uint32_t slot, std::size_t drafter, const execution::AdaptiveDepth& depth);
  const DrafterChoice& choice(std::uint32_t slot) const { return choice_.at(slot); }

 private:
  DecodingOrchestrator(ModelRunner& runner, const WaveComposer& composer, DecodingMode mode,
                       DecodingPolicy policy);
  // One plain (or sampled, or scored: `scored`) step a slot.
  DecodeReport Single(std::span<const DecodeWork> work, bool scored, bool greedy);
  DecodeReport Speculate(std::span<const DecodeWork> work);
  std::size_t At(std::uint32_t slot, std::size_t drafter) const {
    return (std::size_t{slot} * mode_.drafters.size()) + drafter;
  }

  ModelRunner* runner_;
  const WaveComposer* composer_;
  DecodingMode mode_;
  DecodingPolicy policy_;
  std::vector<std::uint32_t> cursor_;               // by slot, then drafter
  std::vector<execution::AdaptiveDepth> adaptive_;  // by slot, then drafter
  std::vector<DrafterChoice> choice_;               // by slot
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_DECODING_H_
