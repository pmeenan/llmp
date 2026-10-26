// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/prefill.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "engine/prompt_wave.h"

namespace llmp::engine {

namespace pl = llmp::planner;

std::uint32_t PrefillOrchestrator::ChunkRows(const PromptWork& prompt, std::uint32_t at) const {
  const auto size = static_cast<std::uint32_t>(
      std::min<std::size_t>(prompt.tokens.size(), std::numeric_limits<std::uint32_t>::max()));
  if (at >= size || options_.chunk_rows == 0 || prompt.stage >= composer_.stages()) return 0;
  std::uint32_t rows = std::min(options_.chunk_rows, size - at);
  if (composer_.caps().max_segment_rows != 0) {
    rows = std::min(rows, composer_.caps().max_segment_rows);
  }
  const std::uint32_t context = composer_.context(prompt.stage);
  const std::uint64_t start = std::uint64_t{prompt.past} + at;
  if (start >= context) return 0;
  rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, context - start));
  if (prompt.boundary > at) rows = std::min(rows, prompt.boundary - at);
  return rows;
}

PendingWork PrefillOrchestrator::ChunkAt(const PromptWork& prompt, std::uint32_t at) const {
  const std::uint32_t rows = ChunkRows(prompt, at);
  const bool last = at + rows == prompt.tokens.size();
  const bool head = last ? prompt.want_head : !options_.state_only;
  return {.slot = prompt.slot,
          .stage = prompt.stage,
          .phase = pl::Phase::kPrefill,
          .past = static_cast<std::uint32_t>(std::min<std::uint64_t>(
              std::uint64_t{prompt.past} + at, std::numeric_limits<std::uint32_t>::max())),
          .rows = rows,
          .outputs = !head                ? 0U
                     : prompt.all_outputs ? rows
                                          : 1U,
          .want_head = head,
          .greedy = false,
          .row_invariant = false,
          .alone = false};
}

std::expected<StepAhead, std::string> PrefillOrchestrator::Predict(
    std::span<const PrefillUnit> units, std::span<const PrefillPeer> peers,
    const PrefillContinuation& continuation, std::span<const std::size_t> owners) const {
  if (options_.lookahead == 0 && !options_.capture_ahead && !options_.prepare_ahead)
    return StepAhead{};
  if (owners.empty()) return StepAhead{};
  const std::size_t n = units.size() + peers.size();
  if (n < units.size()) return std::unexpected("owner extent");
  const std::uint32_t max_verify = continuation.speculate ? continuation.verify_rows : 1;
  const std::uint32_t min_verify = continuation.speculate ? continuation.min_verify_rows : 1;
  if (min_verify == 0 || max_verify < min_verify || max_verify > composer_.context())
    return std::unexpected("verify extent");
  // The capacity follows actual owner count, configured depth alternatives,
  // peer exits and the explicit semantic-search budget. Admission happens
  // before any candidate loops or allocation. No fixed candidate/build ceiling.
  std::uint64_t events = 0, capacity = 0, bytes = 0;
  const auto mul = [](std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    return !__builtin_mul_overflow(a, b, &out);
  };
  const auto add = [](std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    return !__builtin_add_overflow(a, b, &out);
  };
  if (!mul(std::uint64_t{n} + 1, std::uint64_t{max_verify} - min_verify + 2, events) ||
      !mul(events, std::uint64_t{options_.specialization_lookahead} + 1, events) ||
      !add(events, 2, events) || !mul(events, n, capacity) ||
      capacity > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    return std::unexpected("candidate extent");
  constexpr std::uint64_t kAllocatorMetadata = 64;
  std::uint64_t one = 0, scratch = 0;
  if (!mul(n, sizeof(pl::Segment), one) || !add(one, kAllocatorMetadata, one) ||
      !add(one, sizeof(pl::StepShape) + sizeof(std::optional<pl::StepShape>), one) ||
      !mul(capacity, one, bytes) ||
      !mul(n, sizeof(PendingWork) + sizeof(PrefillPeer) + sizeof(std::size_t), scratch) ||
      !add(bytes, scratch, bytes) || !add(bytes, 5 * kAllocatorMetadata, bytes))
    return std::unexpected("workspace extent");
  auto payment = runner_.PayHostMetadata(bytes);
  if (!payment) return std::unexpected(payment.error());
  // Payment declared first: every capacity is destroyed before its refund.
  std::vector<pl::StepShape> candidates(static_cast<std::size_t>(capacity));
  for (auto& candidate : candidates) candidate.segments.reserve(n);
  std::vector<std::optional<pl::StepShape>> inventory;
  inventory.reserve(static_cast<std::size_t>(capacity));
  std::vector<PendingWork> pending;
  pending.reserve(n);
  std::vector<PrefillPeer> cohort;
  cohort.reserve(n);
  std::vector<std::size_t> order(n);
  std::size_t filled = 0;
  const auto compose = [&]() -> std::optional<std::pair<std::size_t, std::size_t>> {
    if (pending.empty()) return std::pair(filled, std::size_t{0});
    auto count = composer_.ComposePrediction(pending, std::span(candidates).subspan(filled), order);
    if (!count) return std::nullopt;
    const auto result = std::pair(filled, *count);
    filled += *count;
    return result;
  };
  bool has_prompt_future = false;
  for (std::size_t depth = 0; depth < 2; ++depth) {
    pending.clear();
    for (const auto owner : owners) {
      if (owner >= units.size()) return std::unexpected("owner index");
      const auto& unit = units[owner];
      if (unit.current.stage >= composer_.stages()) return std::unexpected("stage index");
      if (unit.current.phase != pl::Phase::kPrefill) continue;
      const auto stages = PredictPrefill(
          unit.current.past, unit.current.rows, composer_.context(unit.current.stage),
          composer_.caps().max_segment_rows == 0 ? composer_.context(unit.current.stage)
                                                 : composer_.caps().max_segment_rows,
          unit.next);
      if (!stages[depth].rows) continue;
      auto next = unit.current;
      next.past = stages[depth].first;
      next.rows = stages[depth].rows;
      next.want_head = stages[depth].want_head;
      next.outputs = next.want_head ? (unit.all_outputs ? next.rows : 1U) : 0U;
      pending.push_back(next);
    }
    if (!pending.empty()) {
      has_prompt_future = true;
    }
    if (!compose()) return std::unexpected("prompt geometry");
  }
  // Add exact ready peers, then owners whose session hints prove a prompt
  // frontier. Missing next geometry can name a checkpoint instead of the end;
  // only the explicit full prompt extent proves a generation frontier.
  for (const auto& peer : peers) {
    if (!peer.ready || (peer.remaining && *peer.remaining == 0)) continue;
    if (peer.next.stage != 0 || peer.next.phase != pl::Phase::kDecode || peer.next.rows != 1 ||
        peer.next.past >= composer_.context())
      return std::unexpected("peer geometry");
    if (std::ranges::any_of(cohort, [&](const auto& p) { return p.next.slot == peer.next.slot; }))
      return std::unexpected("duplicate peer");
    cohort.push_back(peer);
  }
  bool any_owner_in_cohort = false;
  for (const auto owner : owners) {
    const auto& unit = units[owner];
    if (unit.current.stage != 0) continue;
    if (unit.current.phase == pl::Phase::kDecode) {
      if (unit.remaining && *unit.remaining <= 1) continue;
      const auto end = std::uint64_t{unit.current.past} + unit.current.rows;
      if (end >= composer_.context()) continue;
      if (std::ranges::any_of(cohort,
                              [&](const auto& p) { return p.next.slot == unit.current.slot; }))
        return std::unexpected("duplicate frontier");
      auto next = unit.current;
      next.phase = pl::Phase::kDecode;
      next.past = static_cast<std::uint32_t>(end);
      next.rows = next.outputs = 1;
      next.want_head = true;
      next.greedy = unit.decode_greedy;
      next.row_invariant = continuation.row_invariant;
      cohort.push_back(
          {.next = next,
           .ready = true,
           .verify_greedy = unit.verify_greedy,
           .sampling = unit.sampling,
           .remaining = unit.remaining ? std::optional(*unit.remaining - 1) : std::nullopt});
      any_owner_in_cohort = true;
      continue;
    }
    if (!unit.prompt_end || !unit.generation_expected || (unit.remaining && *unit.remaining == 0))
      continue;
    const auto stages = PredictPrefill(unit.current.past, unit.current.rows, composer_.context(),
                                       composer_.context(), unit.next);
    const auto current_end = std::uint64_t{unit.current.past} + unit.current.rows;
    const auto end = *unit.prompt_end;
    if (end != current_end && (!stages[0].rows || stages[0].end() != end) &&
        (!stages[1].rows || stages[1].end() != end))
      continue;
    if (end >= composer_.context()) continue;
    if (std::ranges::any_of(cohort,
                            [&](const auto& p) { return p.next.slot == unit.current.slot; }))
      return std::unexpected("duplicate frontier");
    auto next = unit.current;
    next.phase = pl::Phase::kDecode;
    next.past = end;
    next.rows = next.outputs = 1;
    next.want_head = true;
    next.greedy = unit.decode_greedy;
    next.row_invariant = continuation.row_invariant;
    cohort.push_back({.next = next,
                      .ready = true,
                      .verify_greedy = unit.verify_greedy,
                      .sampling = unit.sampling,
                      .remaining = unit.remaining});
    any_owner_in_cohort = true;
  }
  // One scenario for the ready cohort, then each actual known peer exit.
  // Unknown generation budgets never become a guessed death at 128 tokens.
  const bool predict_continuation = any_owner_in_cohort;
  if (predict_continuation && cohort.size() > 1 && composer_.caps().max_segments <= 1) {
    std::erase_if(cohort, [&](const auto& peer) {
      return !std::ranges::any_of(
          owners, [&](auto owner) { return units[owner].current.slot == peer.next.slot; });
    });
  }
  std::uint32_t elapsed = 0;
  for (std::size_t scenario = 0; predict_continuation && scenario <= cohort.size(); ++scenario) {
    std::size_t survivors = 0;
    bool sampled = false, one_remaining = false;
    for (const auto& peer : cohort) {
      if (peer.remaining && *peer.remaining <= elapsed) continue;
      if (elapsed >= composer_.context() - peer.next.past) continue;
      ++survivors;
      sampled |= peer.sampling;
      one_remaining = peer.remaining && *peer.remaining - elapsed == 1;
    }
    if (!survivors) break;
    const bool plain =
        !continuation.speculate || (survivors == 1 && one_remaining) ||
        (continuation.plain && continuation.plain(static_cast<std::uint32_t>(survivors), sampled));
    const std::uint32_t first_rows = plain ? 1 : max_verify;
    const std::uint32_t last_rows = plain ? 1 : min_verify;
    for (std::uint32_t rows = first_rows;; --rows) {
      pending.clear();
      for (const auto& peer : cohort) {
        if ((peer.remaining && *peer.remaining <= elapsed) ||
            elapsed >= composer_.context() - peer.next.past)
          continue;
        auto next = peer.next;
        next.past += elapsed;
        next.rows = std::min(rows, composer_.context() - next.past);
        if (peer.remaining && (plain || min_verify == 1))
          next.rows = std::min(next.rows, *peer.remaining - elapsed);
        next.outputs = next.rows;
        next.row_invariant = continuation.row_invariant;
        if (!plain) {
          next.phase = pl::Phase::kVerify;
          next.greedy = peer.verify_greedy;
        }
        pending.push_back(next);
      }
      auto composed = compose();
      if (!composed) return std::unexpected("continuation geometry");
      if (composed->second == 1) {
        std::size_t at = composed->first;
        for (std::uint32_t event = 0; event < options_.specialization_lookahead; ++event) {
          const auto& shape = candidates[at];
          std::optional<std::uint32_t> advance;
          bool complete = true;
          std::uint32_t horizon = std::numeric_limits<std::uint32_t>::max();
          for (const auto& segment : shape.segments) {
            const auto peer = std::ranges::find_if(
                cohort, [&](const auto& p) { return p.next.slot == segment.slot; });
            auto available = composer_.context() - segment.rows - segment.past;
            if (peer->remaining) {
              const auto consumed = segment.past - peer->next.past;
              if (*peer->remaining <= consumed || *peer->remaining - consumed <= segment.rows) {
                available = 0;
              } else {
                available = std::min(available, *peer->remaining - consumed - segment.rows);
              }
            }
            horizon = std::min(horizon, available);
          }
          if (!horizon) break;
          for (std::size_t owner = 0; owner < shape.segments.size(); ++owner) {
            const auto& segment = shape.segments[owner];
            const auto through = segment.past + horizon;
            const SpecializationRequest query{.shape = shape,
                                              .owner = owner,
                                              .after = segment.past,
                                              .through = through,
                                              .context = composer_.context()};
            auto next = runner_.NextSpecializationChange(query);
            if (!next || next->coverage != SpecializationChange::Coverage::kComplete) {
              complete = false;
              break;
            }
            if (next->next) {
              const auto delta = next->next->position - segment.past;
              advance = advance ? std::min(*advance, delta) : delta;
            }
          }
          if (!complete || !advance) break;
          for (auto& next : pending) {
            if (*advance > composer_.context() - next.rows - next.past)
              return std::unexpected("boundary extent");
            next.past += *advance;
          }
          composed = compose();
          if (!composed || composed->second != 1) return std::unexpected("boundary geometry");
          at = composed->first;
        }
      }
      if (rows == last_rows) break;
    }
    if (!plain && survivors == 1 && (scenario > 0 || has_prompt_future)) {
      pending.clear();
      for (const auto& peer : cohort) {
        if ((peer.remaining && *peer.remaining <= elapsed) ||
            elapsed >= composer_.context() - peer.next.past)
          continue;
        auto next = peer.next;
        next.past += elapsed;
        next.rows = next.outputs = 1;
        next.row_invariant = continuation.row_invariant;
        pending.push_back(next);
      }
      if (!compose()) return std::unexpected("continuation geometry");
    }
    std::optional<std::uint32_t> exit;
    for (const auto& peer : cohort) {
      auto end = composer_.context() - peer.next.past;
      if (peer.remaining) end = std::min(end, *peer.remaining);
      if (end > elapsed) exit = exit ? std::min(*exit, end) : end;
    }
    if (!exit) break;
    elapsed = *exit;
  }
  for (std::size_t i = 0; i < filled; ++i) inventory.emplace_back(std::move(candidates[i]));
  const std::uint32_t builds =
      !has_prompt_future && predict_continuation && continuation.speculate &&
              continuation.verify_rows > 1 && options_.lookahead > 0
          ? std::max<std::uint32_t>(options_.lookahead,
                                    static_cast<std::uint32_t>(inventory.size()))
          : options_.lookahead;
  return runner_.Predict(inventory, builds, has_prompt_future && options_.capture_ahead,
                         has_prompt_future && options_.prepare_ahead);
}

PrefillReport PrefillOrchestrator::Step(std::span<const PromptWork> prompts) {
  if (prompts.empty()) return PrefillReport{};
  constexpr std::uint64_t kAllocatorMetadata = 64;
  std::uint64_t bytes = 0;
  if (__builtin_mul_overflow(std::uint64_t{prompts.size()}, std::uint64_t{sizeof(PrefillUnit)},
                             &bytes) ||
      __builtin_add_overflow(bytes, kAllocatorMetadata, &bytes)) {
    PrefillReport refused;
    refused.refusal = StepRefusal{.what = "prompt metadata extent overflows"};
    return refused;
  }
  auto payment = runner_.PayHostMetadata(bytes);
  if (!payment) {
    PrefillReport refused;
    refused.refusal = StepRefusal{.what = payment.error(), .capacity_refused = true};
    return refused;
  }
  std::vector<PrefillUnit> units;
  units.reserve(prompts.size());
  for (const auto& prompt : prompts) {
    const auto chunk = ChunkAt(prompt, 0);
    const auto next = ChunkAt(prompt, chunk.rows);
    const auto after = ChunkAt(prompt, chunk.rows + next.rows);
    units.push_back(
        {.current = chunk,
         .work = {.tokens = prompt.tokens.first(chunk.rows),
                  .bytes = {},
                  .feature = {},
                  .history =
                      prompt.history.empty() && prompt.past == 0 ? prompt.tokens : prompt.history},
         .next = {.rows = next.rows,
                  .want_head = next.want_head,
                  .after_rows = after.rows,
                  .after_want_head = after.want_head},
         .prompt_end =
             prompt.tokens.size() <= UINT32_MAX - prompt.past
                 ? std::optional(prompt.past + static_cast<std::uint32_t>(prompt.tokens.size()))
                 : std::nullopt,
         .remaining = {},
         .decode_greedy = false,
         .verify_greedy = false,
         .all_outputs = prompt.all_outputs});
  }
  return StepPrepared(units);
}

PrefillReport PrefillOrchestrator::StepPrepared(std::span<const PrefillUnit> units,
                                                std::span<const PrefillPeer> peers,
                                                const PrefillContinuation& continuation) {
  PrefillReport report;
  if (units.empty()) return report;
  std::uint64_t outcome_bytes = 0, temporary_bytes = 0;
  constexpr std::uint64_t kAllocatorMetadata = 64;
  if (__builtin_mul_overflow(std::uint64_t{units.size()}, std::uint64_t{sizeof(PrefillOutcome)},
                             &outcome_bytes) ||
      __builtin_add_overflow(outcome_bytes, kAllocatorMetadata, &outcome_bytes) ||
      __builtin_mul_overflow(
          std::uint64_t{units.size()},
          std::uint64_t{sizeof(PendingWork) + sizeof(SegmentWork) +
                        sizeof(std::span<const std::int32_t>) + sizeof(std::size_t)},
          &temporary_bytes) ||
      __builtin_add_overflow(temporary_bytes, 4 * kAllocatorMetadata, &temporary_bytes)) {
    report.refusal = StepRefusal{.what = "prefill dispatch metadata extent overflows"};
    return report;
  }
  auto outcomes_payment = runner_.PayHostMetadata(outcome_bytes);
  if (!outcomes_payment) {
    report.refusal = StepRefusal{.what = outcomes_payment.error(), .capacity_refused = true};
    return report;
  }
  report.payment_ = std::move(*outcomes_payment);
  auto temporary_payment = runner_.PayHostMetadata(temporary_bytes);
  if (!temporary_payment) {
    report.payment_.Reset();
    report.refusal = StepRefusal{.what = temporary_payment.error(), .capacity_refused = true};
    return report;
  }
  report.outcomes_.resize(units.size());
  std::vector<PendingWork> pending;
  std::vector<SegmentWork> work;
  std::vector<std::span<const std::int32_t>> histories;
  std::vector<std::size_t> of;
  pending.reserve(units.size());
  work.reserve(units.size());
  histories.reserve(units.size());
  of.reserve(units.size());
  for (std::size_t i = 0; i < units.size(); ++i) {
    const auto& unit = units[i];
    if (unit.current.rows == 0 ||
        (unit.current.phase != pl::Phase::kPrefill &&
         (unit.current.phase != pl::Phase::kScore || !unit.all_outputs) &&
         !(composer_.caps().mixed_causal && unit.current.phase == pl::Phase::kDecode &&
           unit.current.stage == 0 && unit.current.rows == 1)) ||
        unit.work.tokens.size() != unit.current.rows) {
      report.outcomes_[i].refusal = StepRefusal{.what = "the prompt chunk has no exact token rows"};
      continue;
    }
    pending.push_back(unit.current);
    work.push_back(unit.work);
    histories.push_back(unit.work.history);
    of.push_back(i);
  }
  struct PredictionContext {
    const PrefillOrchestrator& owner;
    std::span<const PrefillUnit> units;
    std::span<const PrefillPeer> peers;
    const PrefillContinuation& continuation;
    const std::vector<std::size_t>& of;
    PrefillReport& report;
    StepAhead Predict(std::span<const std::size_t> owners) const {
      // A single captured pointer keeps AheadFor in the pinned library's
      // inline callable storage; no uncounted closure allocation is needed.
      std::array<std::size_t, std::numeric_limits<SlotMask>::digits> input_owners{};
      if (owners.size() > input_owners.size()) {
        ++report.prediction_refused;
        return StepAhead{};
      }
      for (std::size_t i = 0; i < owners.size(); ++i) input_owners[i] = of[owners[i]];
      auto predicted =
          owner.Predict(units, peers, continuation, std::span(input_owners).first(owners.size()));
      if (!predicted) {
        ++report.prediction_refused;
        return StepAhead{};
      }
      return std::move(*predicted);
    }
  } prediction{*this, units, peers, continuation, of, report};
  const AheadFor ahead = [&prediction](const ComposedStep&, std::span<const std::size_t> owners) {
    return prediction.Predict(owners);
  };
  const auto wave = RunPromptWave(runner_, composer_, pending, work, histories, ahead);
  report.steps = wave.steps;
  for (std::size_t j = 0; j < pending.size(); ++j) {
    auto& outcome = report.outcomes_[of[j]];
    const auto& ran = wave.outcomes[j];
    outcome.phase = pending[j].phase;
    outcome.refusal = ran.refusal;
    if (ran.refusal) continue;
    outcome.rows = pending[j].rows;
    outcome.last = units[of[j]].prompt_end &&
                   std::uint64_t{pending[j].past} + pending[j].rows == *units[of[j]].prompt_end;
    outcome.outputs = ran.outputs;
    outcome.joined = ran.joined;
    outcome.segment = ran.segment;
    outcome.first_row = ran.first_row;
    outcome.first_output = ran.first_output;
    outcome.path = ran.path;
  }
  return report;
}

}  // namespace llmp::engine
