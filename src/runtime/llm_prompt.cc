// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/check.h"
#include "chat/chat.h"
#include "engine/checkpoint_file.h"
#include "engine/live_state.h"
#include "execution/adaptive_depth.h"
#include "execution/sampling.h"
#include "memory/reclaim.h"
#include "platform/files.h"
#include "platform/kept_files.h"
#include "runtime/intake_limits.h"
#include "runtime/kept_record.h"
#include "runtime/llm_internal.h"
#include "runtime/model_settings.h"
#include "runtime/prefill.h"
#include "runtime/serving.h"
#include "runtime/serving_support.h"
#include "runtime/state_keeper.h"
#include "runtime/turn_reuse.h"
#include "tokenizer/tokenizer.h"

namespace llmp::runtime {
namespace {
using llm_detail::HistoryFundingGuard;
using serving_support::Error;
using serving_support::Seconds;
}  // namespace
Status Llm::PreparePrompt(Branch& branch, std::span<const std::int32_t> tokens,
                          std::uint32_t stable_boundary, std::vector<float>& last,
                          std::uint32_t& reused, const PrefillGoOn& go_on, PrefillRun* run,
                          bool fresh) {
  CheckIdleGeneration(branch);
  branch.capacity_refused_ = false;
  auto opened = BeginPrompt(branch, tokens, stable_boundary, fresh);
  if (!opened) {
    return std::unexpected(opened.error());
  }
  auto& session = **opened;
  std::size_t reclaimed = 0;
  while (!session.done()) {
    auto advanced = session.Advance(go_on, true);
    if (advanced) {
      reclaimed = 0;
      continue;
    }
    if (session.refused()) {
      // The same unit again once capacity was freed; otherwise it ends at
      // its completed prefix, as an undeferred refusal does.
      if (reclaimed < kMaxBranches && ReclaimFor(branch)) {
        ++reclaimed;
        continue;
      }
      branch.capacity_refused_ = true;
      [[maybe_unused]] const auto failed = session.Fail(std::move(advanced.error()));
    }
    break;
  }
  auto result = session.Finish();
  last = std::move(session.last_);
  reused = session.reused();
  if (run != nullptr) {
    *run = session.run();
  }
  return result;
}

Llm::PromptSession::PromptSession(Llm& model, Branch& branch, std::uint32_t stable_boundary,
                                  bool fresh, bool resume, bool scoring,
                                  std::function<bool(std::int32_t, std::span<const float>)> on_row)
    : model_(model),
      branch_(branch),
      stable_boundary_(stable_boundary),
      fresh_(fresh),
      resume_(resume),
      scoring_(scoring),
      on_row_(std::move(on_row)) {}

Llm::PromptSession::~PromptSession() {
  base::Check(finished_, "an unfinished prompt session was destroyed");
}

std::expected<std::unique_ptr<Llm::PromptSession>, std::string> Llm::BeginPrompt(
    Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t stable_boundary, bool fresh,
    bool resume, bool scoring, std::function<bool(std::int32_t, std::span<const float>)> on_row) {
  CheckBranch(branch);
  if (tokens.empty() || tokens.size() > context_ || stable_boundary >= tokens.size()) {
    return Error("the prompt or its turn boundary is outside the context");
  }
  if (max_rows_ == 0 || (scoring && ScoreChunkRowsFor(branch) == 0)) {
    return Error("a prompt needs an admitted positive row capacity");
  }
  if (branch.generation_active_ || branch.prompt_session_ != nullptr) {
    return Error("the conversation already has an active session");
  }
  auto session = std::unique_ptr<PromptSession>(
      new PromptSession(*this, branch, stable_boundary, fresh, resume, scoring, std::move(on_row)));
  branch.prompt_session_ = session.get();
  if (!ReserveTokens(branch, session->tokens_, session->tokens_charge_, tokens.size())) {
    session->Cancel();
    (void)session->Finish();
    return Error("native prompt history exceeds the execution budget");
  }
  session->tokens_.assign(tokens.begin(), tokens.end());
  return session;
}

bool Llm::PromptSession::done() const { return complete_ || finished_; }

bool Llm::PromptSession::started() const { return phase_ != Phase::kReuse; }

std::expected<Llm::PromptSession::Unit, std::string> Llm::PromptSession::NextUnit() const {
  if (done() || advancing_) {
    return Error("the prompt has no next unit");
  }
  Unit unit{.phase = phase_};
  if (phase_ == Phase::kChunk) {
    const auto at = static_cast<std::uint32_t>(branch_.history_.size());
    const auto end =
        checkpoint_pending_ ? stable_boundary_ : static_cast<std::uint32_t>(tokens_.size());
    base::Check(at < end, "a prompt chunk has no rows");
    unit.rows =
        PrefillRows(end - at, scoring_ ? model_.ScoreChunkRowsFor(branch_) : model_.max_rows_);
    if (unit.rows == 0) return Error("the model has no admitted scoring row capacity");
    unit.want_head = scoring_ || at + unit.rows == tokens_.size();
  }
  return unit;
}

std::uint32_t Llm::PromptSession::remaining_rows() const {
  if (done()) {
    return 0;
  }
  std::size_t at = branch_.history_.size();
  if (phase_ == Phase::kReuse) {
    // Not reused yet: what the reuse would keep (none of a stale history
    // that merely shares a few tokens with the prompt).
    at = model_.ReusableFor(branch_, tokens_, fresh_, resume_);
  }
  return at >= tokens_.size() ? 0 : static_cast<std::uint32_t>(tokens_.size() - at);
}

void Llm::PromptSession::NextPhase() {
  const auto at = branch_.history_.size();
  if (scoring_ && !branch_.score_rows_.bytes.empty()) {
    phase_ = Phase::kScores;
  } else if (checkpoint_pending_ && at == stable_boundary_) {
    phase_ = Phase::kCheckpoint;
  } else if (at < tokens_.size()) {
    phase_ = Phase::kChunk;
  } else {
    complete_ = true;
  }
}

Status Llm::PromptSession::Fail(std::string error) {
  branch_.needs_clear_ = branch_.needs_clear_ || !model_.StateUsableFor(branch_);
  if (branch_.needs_clear_) {
    model_.ReleaseTokens(branch_.history_, branch_.history_charge_);
  }
  last_.clear();
  complete_ = true;
  ran_ = std::unexpected(std::move(error));
  return ran_;
}

bool Llm::PromptSession::CanJoin(const PromptSession& peer) const {
  if (&model_ != &peer.model_ || this == &peer || scoring_ || peer.scoring_ || model_.speculate_)
    return false;
  const auto a = NextUnit(), b = peer.NextUnit();
  return a && b && a->phase == Phase::kChunk && b->phase == Phase::kChunk &&
         model_.CompatiblePrefill(static_cast<std::uint32_t>(branch_.history_.size()), a->rows,
                                  static_cast<std::uint32_t>(peer.branch_.history_.size()), b->rows,
                                  a->want_head, b->want_head);
}

std::optional<std::size_t> Llm::PromptSession::SelectForWave(
    bool decode, std::optional<std::size_t> chosen, std::span<PromptSession* const> sessions) {
  if (decode || !chosen || *chosen >= sessions.size() || sessions[*chosen] == nullptr)
    return std::nullopt;
  const auto& anchor = *sessions[*chosen];
  const auto unit = anchor.NextUnit();
  if (!unit || unit->phase != Phase::kChunk || anchor.scoring_ || anchor.model_.speculate_ ||
      anchor.model_.prefill_wave_capacity() <= 1)
    return chosen;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    const auto* peer = sessions[i];
    if (peer == nullptr || peer == &anchor || &peer->model_ != &anchor.model_ || peer->scoring_)
      continue;
    const auto next = peer->NextUnit();
    if (!next || (next->phase != Phase::kReuse && next->phase != Phase::kCheckpoint)) continue;
    const auto remaining = peer->remaining_rows();
    if (remaining == 0) continue;
    const auto at = static_cast<std::uint32_t>(peer->tokens_.size()) - remaining;
    // Reuse is a hint until its ordinary unit actually settles. A cache
    // miss may change this geometry; CanJoin checks the resulting chunks.
    const auto end = next->phase == Phase::kReuse && peer->stable_boundary_ > at
                         ? peer->stable_boundary_
                         : static_cast<std::uint32_t>(peer->tokens_.size());
    const auto rows = PrefillRows(end - at, peer->model_.max_rows_);
    const bool want_head = at + rows == peer->tokens_.size();
    if (anchor.model_.CompatiblePrefill(static_cast<std::uint32_t>(anchor.branch_.history_.size()),
                                        unit->rows, at, rows, unit->want_head, want_head))
      return i;
  }
  return chosen;
}

PrefillHint Llm::PromptSession::NextPrefillHint(std::uint32_t current_end) const {
  PrefillHint hint;
  const auto boundary =
      checkpoint_pending_ ? stable_boundary_ : static_cast<std::uint32_t>(tokens_.size());
  if (scoring_) return hint;  // Paid scoring batches admit their next capacity on demand.
  if (current_end < boundary) {
    hint.rows = scoring_ ? 1U : PrefillRows(boundary - current_end, model_.max_rows_);
    hint.want_head = scoring_ || current_end + hint.rows == tokens_.size();
    if (const auto after = current_end + hint.rows; after < boundary) {
      hint.after_rows = scoring_ ? 1U : PrefillRows(boundary - after, model_.max_rows_);
      hint.after_want_head = scoring_ || after + hint.after_rows == tokens_.size();
    }
  }
  return hint;
}

Status Llm::PromptSession::PrepareChunk(std::uint32_t rows, bool defer_capacity) {
  const auto end = branch_.history_.size() + rows;
  if (!model_.ReserveTokens(branch_, branch_.history_, branch_.history_charge_, end)) {
    refused_ = defer_capacity;
    if (defer_capacity) return Error("native retained history exceeds the execution budget");
    return Fail("native retained history exceeds the execution budget");
  }
  return {};
}

Status Llm::PromptSession::CompleteChunk(std::uint32_t rows, double wall_seconds,
                                         double accounted_seconds, Status result,
                                         bool defer_capacity, ScoreRows scored) {
  const auto at = static_cast<std::uint32_t>(branch_.history_.size());
  const auto end = at + rows;
  if (!result) {
    std::string error =
        scoring_
            ? std::format("{}'s scoring: the row at {}: {}", model_.name_, at, result.error())
            : std::format("{}'s prefill: the chunk at {}: {}", model_.name_, at, result.error());
    if (defer_capacity && model_.StateRefusedFor(branch_) && model_.StateUsableFor(branch_)) {
      refused_ = true;
      return std::unexpected(std::move(error));
    }
    return Fail(std::move(error));
  }
  branch_.history_.insert(branch_.history_.end(), tokens_.begin() + at, tokens_.begin() + end);
  branch_.history_used_ = Clock::now();
  run_.end = end;
  ++run_.chunks;
  run_.longest = std::max(run_.longest, wall_seconds);
  if (!scoring_) model_.calibration_samples_.PrefillChunk(at, rows, accounted_seconds);
  from_zero_ = from_zero_ || at == 0;
  chunk_seconds_ += accounted_seconds;
  if (!scoring_ && from_zero_ && end == tokens_.size())
    model_.calibration_samples_.Prefill(end, chunk_seconds_);
  if (scoring_) {
    if (!scored.bytes.empty()) {
      if (!scored.valid() || scored.first != at || scored.rows != rows) {
        branch_.needs_clear_ = true;
        return Fail("the retired scoring batch has inconsistent owned rows");
      }
      scored.published = branch_.score_rows_.published;
      branch_.score_rows_ = std::move(scored);
      return PublishScores();
    }
    if (rows != 1) {
      branch_.needs_clear_ = true;
      return Fail("a multirow scoring step returned no owned distributions");
    }
    if (end < tokens_.size()) {
      run_.published = branch_.score_rows_.published = end;
      if (on_row_ && !on_row_(tokens_[end], last_)) Stop();
    }
  }
  return {};
}

Status Llm::PromptSession::PublishScores() {
  auto& scored = branch_.score_rows_;
  if (!scored.valid() || std::uint64_t{scored.first} + scored.rows > branch_.history_.size())
    return Fail("pending score publication has no retired history");
  while (scored.next < scored.rows) {
    const auto row_index = scored.next++;
    const auto position = scored.first + row_index + 1;
    if (position >= tokens_.size()) continue;  // final head, no following prompt token
    run_.published = scored.published = position;
    if (on_row_ && !on_row_(tokens_[position], scored.row(row_index))) {
      Stop();
      return {};
    }
  }
  if (std::uint64_t{scored.first} + scored.rows == tokens_.size()) {
    const auto final = scored.row(scored.rows - 1);
    last_.assign(final.begin(), final.end());
  }
  scored.ClearRows();
  return {};
}

Llm::PrefillHeadModes Llm::FuturePrefillHeads(std::span<const PreparedPrefill> prepared) {
  PrefillHeadModes heads;
  bool mixed_next = false, mixed_after = false;
  for (const auto& unit : prepared) {
    if (unit.next.rows != 0) {
      if (heads.next.has_value() && *heads.next != unit.next.want_head)
        mixed_next = true;
      else if (!heads.next.has_value())
        heads.next = unit.next.want_head;
    }
    if (unit.next.after_rows != 0) {
      if (heads.after.has_value() && *heads.after != unit.next.after_want_head)
        mixed_after = true;
      else if (!heads.after.has_value())
        heads.after = unit.next.after_want_head;
    }
  }
  if (mixed_next) heads.next.reset();
  if (mixed_after) heads.after.reset();
  return heads;
}

Status Llm::RunPreparedPrefillWave(std::span<PreparedPrefill> prepared) {
  for (auto& unit : prepared) {
    unit.result = RunPrefillChunkFor(*unit.branch, unit.all, unit.past, speculate_, unit.want_head,
                                     *unit.logits, unit.next);
    if (!unit.result && !GenerationCohortUsable()) return unit.result;
  }
  return {};
}

Status Llm::RunPromptWave(std::span<PromptSession* const> sessions,
                          std::span<const PrefillGoOn> go_on, bool defer_capacity) {
  if (sessions.empty() || sessions.size() > kMaxBranches ||
      sessions.size() > prefill_wave_capacity() || go_on.size() != sessions.size())
    return Error("the prompt cohort exceeds this model's capability");
  std::array<bool, kMaxBranches> seen{};
  std::array<PromptSession::Unit, kMaxBranches> units{};
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    const auto* session = sessions[i];
    if (!session || &session->model_ != this || session->done() || session->advancing_ ||
        session->scoring_ || speculate_)
      return Error("a prompt cohort contains an inactive, scoring or foreign session");
    const auto next = session->NextUnit();
    if (!next || next->phase != PromptSession::Phase::kChunk ||
        (i != 0 && !sessions.front()->CanJoin(*session)))
      return Error("a prompt cohort needs compatible plain chunks");
    const auto id = BranchIndex(session->branch_);
    if (id >= seen.size() || seen[id]) return Error("a prompt cohort repeats a native branch");
    seen[id] = true;
    units[i] = *next;
  }
  // Keep every session borrowed through dispatch and all peer publication.
  // The HTTP driver may Finish/Cancel/BeginGeneration only after this returns.
  struct CompletedWave {
    std::span<PromptSession* const> sessions;
    ~CompletedWave() {
      for (auto* session : sessions) session->advancing_ = false;
    }
  } completed{sessions};
  for (auto* session : sessions) {
    session->refused_ = false;
    session->unit_result_ = {};
    session->advancing_ = true;
  }
  std::array<PreparedPrefill, kMaxBranches> storage{};
  std::size_t count = 0;
  const auto fail_cohort = [&](const std::string& error) {
    for (auto* session : sessions) {
      session->unit_result_ = Error(error);
      session->branch_.needs_clear_ = true;
      (void)session->Fail(error);
    }
  };
  const auto started = Clock::now();
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    auto& session = *sessions[i];
    const auto& next = units[i];
    CheckIdleGeneration(session.branch_, &session);
    if (go_on[i] && !go_on[i](next.rows)) {
      session.Stop();
      continue;
    }
    if (auto funded = session.PrepareChunk(next.rows, defer_capacity); !funded) {
      session.unit_result_ = std::move(funded);
      continue;
    }
    const auto at = static_cast<std::uint32_t>(session.branch_.history_.size());
    if (auto state = PreparePrefillStateFor(session.branch_, at, next.rows); !state) {
      session.unit_result_ =
          session.CompleteChunk(next.rows, 0, 0, std::move(state), defer_capacity);
      if (!GenerationCohortUsable()) {
        const std::string error = "the native prompt cohort became unusable during preparation";
        fail_cohort(error);
        return Error(error);
      }
      continue;
    }
    storage[count++] = {.session = &session,
                        .branch = &session.branch_,
                        .all = std::span(session.tokens_).first(at + next.rows),
                        .past = at,
                        .rows = next.rows,
                        .want_head = next.want_head,
                        .next = session.NextPrefillHint(at + next.rows),
                        .logits = &session.last_,
                        .result = {}};
  }
  auto prepared = std::span(storage).first(count);
  if (prepared.empty()) return {};
  std::ranges::sort(prepared, {}, [this](const auto& unit) { return BranchIndex(*unit.branch); });
  const StateKeeper::Quiet quiet(kept_.keeper);
  if (auto ran = RunPreparedPrefillWave(prepared); !ran) {
    fail_cohort(ran.error());
    return ran;
  }
  if (!GenerationCohortUsable()) {
    const std::string error = "the native prompt cohort became unusable after dispatch";
    fail_cohort(error);
    return Error(error);
  }
  const double seconds = Seconds(Clock::now() - started);
  std::uint32_t total_rows = 0;
  for (const auto& unit : prepared) total_rows += unit.rows;
  for (auto& unit : prepared) {
    // Wall latency belongs to every participant; calibration apportions the
    // shared cost by actual rows rather than charging the same work twice.
    const double attributed = seconds * unit.rows / total_rows;
    unit.session->unit_result_ = unit.session->CompleteChunk(
        unit.rows, seconds, attributed, std::move(unit.result), defer_capacity);
    if (unit.session->unit_result_) unit.session->NextPhase();
  }
  return {};
}

Status Llm::PromptSession::Advance(const PrefillGoOn& go_on, bool defer_capacity) {
  refused_ = false;
  auto next = NextUnit();
  if (!next) {
    return Fail(next.error());
  }
  model_.CheckIdleGeneration(branch_, this);
  advancing_ = true;
  struct CompletedUnit {
    explicit CompletedUnit(bool& value) noexcept : advancing(value) {}
    CompletedUnit(const CompletedUnit&) = delete;
    CompletedUnit& operator=(const CompletedUnit&) = delete;
    CompletedUnit(CompletedUnit&&) = delete;
    CompletedUnit& operator=(CompletedUnit&&) = delete;
    bool& advancing;
    ~CompletedUnit() { advancing = false; }
  } completed{advancing_};
  if (go_on && !go_on(next->rows)) {
    Stop();
    return {};
  }
  if (next->phase == Phase::kReuse) {
    Status reused;
    // Serial literal scoring historically starts from an already-empty healthy
    // state. Preserve that completed boundary instead of clearing it twice.
    const bool empty_scorer = scoring_ && fresh_ && branch_.history_.empty() &&
                              !branch_.needs_clear_ && !model_.SpilledFor(branch_) &&
                              model_.StateUsableFor(branch_) && model_.CursorFor(branch_) == 0;
    if (!empty_scorer)
      reused =
          model_.ReusePrompt(branch_, tokens_, reused_, fresh_, resume_, go_on, run_.stopped, this);
    if (!reused) {
      if (defer_capacity && model_.SpilledFor(branch_) && model_.StateRefusedFor(branch_) &&
          model_.StateUsableFor(branch_)) {
        // Its spilled state's restore refused for capacity, before any
        // change: the same unit again once there is room.
        refused_ = true;
        return std::unexpected(reused.error());
      }
      return Fail(reused.error());
    }
    run_.end = reused_;
    if (scoring_) run_.published = branch_.score_rows_.published;
    if (run_.stopped) {
      Stop();
      return {};
    }
    checkpoint_pending_ = stable_boundary_ != 0 && stable_boundary_ >= branch_.history_.size();
  } else if (next->phase == Phase::kCheckpoint) {
    auto captured = model_.CaptureTurnCheckpoint(branch_, go_on, run_.stopped, this);
    if (!captured) {
      return Fail(captured.error());
    }
    if (run_.stopped) {
      Stop();
      return {};
    }
    checkpoint_pending_ = false;
  } else if (next->phase == Phase::kScores) {
    if (auto published = PublishScores(); !published) return published;
  } else {
    const auto at = static_cast<std::uint32_t>(branch_.history_.size());
    auto rows = next->rows;
    for (;;) {
      auto prepared = PrepareChunk(rows, scoring_ || defer_capacity);
      if (prepared) break;
      if (scoring_ && rows > 1) {
        rows = std::max(1U, rows / 2);
        continue;
      }
      if (scoring_ && !defer_capacity) return Fail(prepared.error());
      return prepared;
    }
    refused_ = false;
    const auto end = at + rows;
    const auto hint = NextPrefillHint(end);
    const auto started = Clock::now();
    const StateKeeper::Quiet quiet(model_.kept_.keeper);  // records' hashing waits
    // Scalar sessions carry the same authoritative continuation facts as a
    // shared prompt wave, including scoring/speculative sessions that cannot
    // join the public RunPromptWave entry point.
    PreparedPrefill unit{.session = this,
                         .branch = &branch_,
                         .all = std::span(tokens_).first(end),
                         .past = at,
                         .rows = rows,
                         .want_head = scoring_ || end == tokens_.size(),
                         .next = hint,
                         .logits = &last_,
                         .result = {}};
    auto dispatched = model_.RunPreparedPrefillWave(std::span(&unit, 1));
    auto chunk = !dispatched && unit.result ? std::move(dispatched) : std::move(unit.result);
    const double seconds = Seconds(Clock::now() - started);
    if (auto completed_chunk = CompleteChunk(unit.rows, seconds, seconds, std::move(chunk),
                                             defer_capacity, std::move(unit.scored));
        !completed_chunk)
      return completed_chunk;
  }
  NextPhase();
  return {};
}

void Llm::PromptSession::Cancel() {
  base::Check(!advancing_, "cancelling a prompt before its unit completed");
  Stop();
}

void Llm::PromptSession::Stop() {
  if (finished_) {
    return;
  }
  run_.stopped = true;
  last_.clear();
  complete_ = true;
}

Status Llm::PromptSession::Finish() {
  base::Check(!advancing_, "finishing a prompt before its unit completed");
  if (!done()) {
    return Error("a prompt still has unprocessed units");
  }
  if (!finished_) {
    base::Check(branch_.prompt_session_ == this, "a prompt lost its branch ownership");
    if (scoring_ && ran_ && run_.chunks != 0) {
      if (auto settled = model_.SettleFor(branch_); !settled) {
        branch_.needs_clear_ = true;
        model_.ReleaseTokens(branch_.history_, branch_.history_charge_);
        last_.clear();
        ran_ = std::move(settled);
      }
    }
    model_.ReleaseTokens(tokens_, tokens_charge_);
    branch_.prompt_session_ = nullptr;
    finished_ = true;
  }
  return ran_;
}

Status Llm::Prefill(Branch& branch, std::span<const std::int32_t> tokens, std::vector<float>& last,
                    const PrefillGoOn& go_on, PrefillRun* run) {
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  // Records' hashing waits while the prompt prefills (StateKeeper::Quiet).
  const StateKeeper::Quiet quiet(kept_.keeper);
  branch.capacity_refused_ = false;
  if (branch.needs_clear_) {
    if (auto r = Clear(branch); !r) {
      return r;
    }
  }
  if (tokens.empty()) {
    return Error("nothing to prefill");
  }
  const std::size_t total = branch.history_.size() + tokens.size();
  if (total > context_) {
    return Error("the prompt exceeds the context");
  }
  MemoryCharge all_charge;
  std::vector<std::int32_t> all;
  if (!ReserveTokens(branch, all, all_charge, total)) {
    branch.capacity_refused_ = true;
    return Error("native prefill history exceeds the execution budget");
  }
  all.insert(all.end(), branch.history_.begin(), branch.history_.end());
  all.insert(all.end(), tokens.begin(), tokens.end());
  if (all.size() > context_) {
    return Error(
        std::format("{} tokens do not fit {}'s context of {}", all.size(), name_, context_));
  }
  last.clear();
  auto completed = static_cast<std::uint32_t>(branch.history_.size());
  const bool from_zero = branch.history_.empty();
  double chunk_seconds = 0;
  auto ran = RunPrefillChunks(
      static_cast<std::uint32_t>(branch.history_.size()), static_cast<std::uint32_t>(all.size()),
      max_rows_,
      [&](std::uint32_t at, std::uint32_t n) {
        const auto remaining = static_cast<std::uint32_t>(all.size()) - at - n;
        const auto next_rows = PrefillRows(remaining, max_rows_);
        const auto after_rows = PrefillRows(remaining - next_rows, max_rows_);
        const PrefillHint hint{next_rows, next_rows == remaining, after_rows,
                               after_rows == remaining - next_rows};
        auto started = Clock::now();
        auto chunk = RunPrefillChunkFor(branch, std::span(all).first(at + n), at, speculate_,
                                        at + n == all.size(), last, hint);
        // A capacity refusal ran nothing: the same chunk again once freed.
        for (std::size_t reclaimed = 0;
             !chunk && CapacityRefused(branch) && reclaimed < kMaxBranches && ReclaimFor(branch);
             ++reclaimed) {
          started = Clock::now();
          chunk = RunPrefillChunkFor(branch, std::span(all).first(at + n), at, speculate_,
                                     at + n == all.size(), last, hint);
        }
        if (chunk) {
          completed = at + n;
          // Its speed, and a whole prefill's cost a token (calibration.h).
          const double seconds = Seconds(Clock::now() - started);
          calibration_samples_.PrefillChunk(at, n, seconds);
          chunk_seconds += seconds;
          if (from_zero && completed == all.size()) {
            calibration_samples_.Prefill(completed, chunk_seconds);
          }
        } else {
          branch.capacity_refused_ = CapacityRefused(branch);
        }
        return chunk;
      },
      go_on);
  if (!ran) {
    branch.needs_clear_ = !StateUsableFor(branch);
    if (branch.needs_clear_) {
      ReleaseTokens(branch.history_, branch.history_charge_);
      ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
    } else {
      PublishTokens(branch, all, all_charge, completed);
    }
    ReleaseTokens(all, all_charge);
    last.clear();
    return Error(std::format("{}'s prefill: {}", name_, ran.error()));
  }
  if (run != nullptr) {
    *run = *ran;
  }
  // The state holds exactly the chunks that ran: stopped between chunks,
  // the history is their prefix (which the next prefill continues from,
  // through the same chunk boundaries), and there are no logits to
  // generate from.
  PublishTokens(branch, all, all_charge, ran->end);
  branch.history_used_ = Clock::now();
  if (ran->stopped) {
    last.clear();
  }
  return {};
}

Status Llm::ScorePrompt(Branch& branch, std::span<const std::int32_t> tokens,
                        std::vector<float>& last,
                        const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                        const PrefillGoOn& go_on, PrefillRun* run) {
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  branch.capacity_refused_ = false;
  if (!branch.history_.empty())
    return Error("literal scoring needs an empty history and a nonempty prompt within context");
  auto opened = branch.BeginScoringPrompt(tokens, on_row);
  if (!opened) return Error(opened.error());
  auto& session = **opened;
  std::size_t reclaimed = 0;
  while (!session.done()) {
    const PrefillGoOn score_go_on =
        go_on ? PrefillGoOn([&](std::uint32_t rows) { return rows == 0 || go_on(rows); })
              : PrefillGoOn{};
    auto advanced = session.Advance(score_go_on, true);
    if (advanced) {
      reclaimed = 0;
      continue;
    }
    if (session.refused()) {
      if (reclaimed < kMaxBranches && ReclaimFor(branch)) {
        ++reclaimed;
        continue;
      }
      branch.capacity_refused_ = true;
      (void)session.Fail(std::move(advanced.error()));
    }
    break;
  }
  auto finished = session.Finish();
  last = std::move(session.last_);
  if (run) *run = session.run();
  return finished;
}

}  // namespace llmp::runtime
