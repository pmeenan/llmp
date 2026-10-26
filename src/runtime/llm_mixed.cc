// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <utility>

#include "base/check.h"
#include "runtime/serving.h"
#include "runtime/serving_support.h"

namespace llmp::runtime {
using serving_support::Clock;
using serving_support::Error;
using serving_support::Seconds;

Status Llm::RunCausalWave(std::span<PromptSession* const> prompts,
                          std::span<const PrefillGoOn> go_on,
                          std::span<GenerationSession* const> generations, bool defer_capacity) {
  const auto count = prompts.size() + generations.size();
  if (!supports_causal_waves() || speculate_ || prompts.empty() || generations.empty() ||
      count > kMaxBranches || count > generation_wave_capacity() ||
      count > prefill_wave_capacity() || go_on.size() != prompts.size())
    return Error("mixed causal dispatch needs compatible plain prompt and decode sessions");
  std::array<bool, kMaxBranches> seen{};
  std::array<PromptSession::Unit, kMaxBranches> next{};
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    auto* session = prompts[i];
    if (!session || &session->model_ != this || session->done() || session->advancing_ ||
        session->scoring_)
      return Error("mixed causal dispatch has an inactive or foreign prompt");
    const auto unit = session->NextUnit();
    if (!unit || unit->phase != PromptSession::Phase::kChunk ||
        (i != 0 && !prompts.front()->CanJoin(*session)))
      return Error("mixed causal dispatch needs a ready prompt chunk");
    const auto slot = BranchIndex(session->branch_);
    if (slot >= seen.size() || std::exchange(seen[slot], true))
      return Error("mixed causal dispatch repeats a branch");
    next[i] = *unit;
  }
  for (auto* session : generations) {
    if (!session || &session->model_ != this || session->done() || session->prepared_ ||
        session->finished_ || session->publishing_)
      return Error("mixed causal dispatch has an inactive or foreign generation");
    const auto slot = BranchIndex(session->branch_);
    if (slot >= seen.size() || std::exchange(seen[slot], true))
      return Error("mixed causal dispatch repeats a branch");
  }
  struct PromptBoundary {
    std::span<PromptSession* const> sessions;
    ~PromptBoundary() {
      for (auto* session : sessions) session->advancing_ = false;
    }
  } boundary{prompts};
  for (auto* session : prompts) {
    session->advancing_ = true;
    session->refused_ = false;
    session->unit_result_ = {};
  }
  const auto fail = [&](const std::string& error) {
    for (auto* session : prompts) {
      if (session->done()) continue;
      session->unit_result_ = Error(error);
      session->branch_.needs_clear_ = true;
      (void)session->Fail(error);
    }
    for (auto* session : generations) {
      if (session->done() || session->finished_) continue;
      if (session->prepared_)
        (void)session->FailStep(error);
      else {
        session->ran_ = Error(error);
        session->refusal_.clear();
        session->failed_prefix_valid_ = false;
      }
    }
  };
  // Stack inventories follow the actual fixed slot representation. Native
  // control, programs, resources and results are independently prepaid together
  // by the canonical complete-batch execution path.
  std::array<PreparedPrefill, kMaxBranches> prompt_storage{};
  std::array<PreparedGeneration, kMaxBranches> generation_storage{};
  std::size_t pcount = 0, gcount = 0;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    auto& session = *prompts[i];
    CheckIdleGeneration(session.branch_, &session);
    if (go_on[i] && !go_on[i](next[i].rows)) {
      session.Stop();
      continue;
    }
    if (auto prepared = session.PrepareChunk(next[i].rows, defer_capacity); !prepared) {
      session.unit_result_ = std::move(prepared);
      continue;
    }
    const auto at = static_cast<std::uint32_t>(session.branch_.history_.size());
    if (auto prepared = PreparePrefillStateFor(session.branch_, at, next[i].rows); !prepared) {
      session.unit_result_ =
          session.CompleteChunk(next[i].rows, 0, 0, std::move(prepared), defer_capacity);
      if (!GenerationCohortUsable()) {
        fail("mixed causal preparation lost its cohort");
        return Error("mixed causal preparation lost its cohort");
      }
      continue;
    }
    prompt_storage[pcount++] = {.session = &session,
                                .branch = &session.branch_,
                                .all = std::span(session.tokens_).first(at + next[i].rows),
                                .past = at,
                                .rows = next[i].rows,
                                .want_head = next[i].want_head,
                                .next = session.NextPrefillHint(at + next[i].rows),
                                .logits = &session.last_,
                                .result = {}};
  }
  for (auto* session : generations) {
    auto step = session->PrepareStep(defer_capacity);
    if (!step) {
      if (!GenerationCohortUsable()) {
        fail(step.error());
        return Error(step.error());
      }
      continue;
    }
    base::Check(!step->speculative, "plain mixed session prepared speculation");
    generation_storage[gcount++] = {.session = session,
                                    .branch = &session->branch_,
                                    .step = *step,
                                    .row = {},
                                    .kept = {},
                                    .logits = {},
                                    .drafted = 0,
                                    .result = {},
                                    .failed_prefix_valid = false,
                                    .anchor_processed = false,
                                    .chosen = std::nullopt};
  }
  auto prepared_prompts = std::span(prompt_storage).first(pcount);
  auto prepared_generations = std::span(generation_storage).first(gcount);
  if (pcount + gcount == 0) return {};
  std::ranges::sort(prepared_prompts, {},
                    [this](const auto& unit) { return BranchIndex(*unit.branch); });
  std::ranges::sort(prepared_generations, {},
                    [this](const auto& unit) { return BranchIndex(*unit.branch); });
  const auto started = Clock::now();
  const StateKeeper::Quiet quiet(kept_.keeper);
  auto result = RunPreparedCausalWave(prepared_prompts, prepared_generations);
  if (!result || !GenerationCohortUsable()) {
    const auto error =
        result ? std::string("mixed causal dispatch lost its cohort") : result.error();
    fail(error);
    return Error(error);
  }
  const double seconds = Seconds(Clock::now() - started);
  std::uint64_t rows = gcount;
  for (const auto& unit : prepared_prompts) rows += unit.rows;
  if (!prepared_generations.empty() && rows != 0) {
    std::size_t tokens = 0;
    std::uint32_t at = std::numeric_limits<std::uint32_t>::max();
    for (const PreparedGeneration& unit : prepared_generations) {
      if (unit.result) ++tokens;
      at = std::min(at, unit.step.position);
    }
    calibration_samples_.DecodeStep(
        at, static_cast<double>(tokens) / static_cast<double>(prepared_generations.size()),
        seconds * static_cast<double>(gcount) / static_cast<double>(rows));
  }
  for (auto& unit : prepared_prompts) {
    unit.session->unit_result_ = unit.session->CompleteChunk(
        unit.rows, seconds, seconds * unit.rows / static_cast<double>(rows), std::move(unit.result),
        defer_capacity);
    if (unit.session->unit_result_) unit.session->NextPhase();
    if (!GenerationCohortUsable()) {
      fail("mixed causal publication lost its cohort");
      return Error("mixed causal publication lost its cohort");
    }
  }
  for (auto& unit : prepared_generations) {
    if (!unit.result)
      (void)(unit.anchor_processed
                 ? unit.session->FailAfterAnchor(unit.result.error())
                 : unit.session->FailStep(unit.result.error(), unit.failed_prefix_valid));
    else if (unit.step.forced)
      (void)unit.session->ApplyCertain(std::move(unit.row), std::move(unit.logits));
    else if (unit.chosen)
      (void)unit.session->ApplyChosen(*unit.chosen);
    else
      (void)unit.session->ApplyPlain(std::move(unit.row));
    if (!GenerationCohortUsable()) {
      fail("mixed causal publication lost its cohort");
      return Error("mixed causal publication lost its cohort");
    }
  }
  return {};
}
}  // namespace llmp::runtime
